//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    /// Original packed 16-bit format (86 or 115) when the host texture was
    /// widened to R8G8B8A8_UNORM, so detiled 2-byte texels expand into the
    /// 4-byte staging buffer.
    widened_from: Option<u32>,
    last: Mutex<Last>,
    #[cfg(test)]
    gpu_uploads: std::sync::atomic::AtomicU64,
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    asked: u64,
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

thread_local! {
    /// Texture memory read for hashing and detiling (both watched and unwatched
    /// paths): one per submitting thread, as large as the largest texture it read.
    static SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// State recorded when a host copy or resolve writes a texture.
///
/// While the texture's console memory stays untouched after this point,
/// [`upload`] keeps the host-written contents instead of overwriting them with
/// stale title memory. Once the title writes the memory again, the entry is
/// cleared and detiling resumes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum HostWritten {
    /// Write-watched memory (`MEM_WRITE_WATCH`): journal asking number at the
    /// host copy.
    Watched { start: usize, end: usize, at: u64 },
    /// Unwatched memory (e.g. physical-page views): content hash at the host
    /// copy, or `None` if the memory was not committed at copy time.
    Hashed {
        memory: usize,
        size: usize,
        hash: Option<blake3::Hash>,
    },
    /// Non-title-written texture or no detileable console layout: stays
    /// host-written until forgotten.
    Sticky,
}

/// Textures the host GPU wrote by a copy or resolve: their contents live on
/// the host until the title writes their console memory again.
static HOST_WRITTEN: Mutex<Option<HashMap<usize, HostWritten>>> = Mutex::new(None);

/// Console memory pointer and layout byte size for a title-written, detileable
/// texture. Returns `None` for render targets, depth/UAV textures, buffers, or
/// unsupported layouts.
fn texture_span(resource: This) -> Option<(*const u8, usize)> {
    {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = map.as_ref().and_then(|m| m.get(&(resource as usize))) {
            return Some((s.memory, s.layout.size as usize));
        }
    }
    if !crate::resource::host_resource(resource).is_some_and(title_written) {
        return None;
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let tex = crate::resource::texture_layout(&desc)?;
    let layout = &tex.planes[0].layout;
    if layout.desc.swizzle.xor() || layout.size == 0 {
        return None;
    }
    usize::try_from(layout.size).ok().map(|size| (memory, size))
}

/// Hashes `[memory, memory + size)` when committed, reading through the kernel
/// so concurrently unmapped pages do not fault.
fn hash_unwatched(memory: *const u8, size: usize) -> Option<blake3::Hash> {
    if !committed(memory, size) {
        return None;
    }
    let mut hasher = blake3::Hasher::new();
    let mut chunk = vec![0u8; size.min(64 << 10)];
    let mut offset = 0;
    while offset < size {
        let n = (size - offset).min(chunk.len());
        // SAFETY: `offset < size`; `copy_mapped` reads via the kernel, zeroing
        // any page unmapped concurrently.
        let src = unsafe { memory.add(offset) };
        crate::host::copy_mapped(chunk.as_mut_ptr(), src, n, false);
        hasher.update(&chunk[..n]);
        offset += n;
    }
    Some(hasher.finalize())
}

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let state = match texture_span(resource) {
        Some((memory, size)) => {
            let start = memory as usize;
            if let Some(end) = start.checked_add(size) {
                if let Some((at, _)) = crate::watch::writes(start, end, false) {
                    HostWritten::Watched { start, end, at }
                } else {
                    HostWritten::Hashed {
                        memory: start,
                        size,
                        hash: hash_unwatched(memory, size),
                    }
                }
            } else {
                HostWritten::Sticky
            }
        }
        None => HostWritten::Sticky,
    };
    let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    map.get_or_insert_with(HashMap::new)
        .insert(resource as usize, state);
}

/// Whether `resource` still holds host-written contents (no title write to its
/// console memory since the copy). Removes the entry once title memory changes.
fn still_host_written(resource: This) -> bool {
    let key = resource as usize;
    let state = {
        let map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        let Some(&state) = map.as_ref().and_then(|m| m.get(&key)) else {
            return false;
        };
        state
    };
    let unchanged = match state {
        HostWritten::Sticky => true,
        HostWritten::Watched { start, end, at } => match crate::watch::writes(start, end, false) {
            Some((_, written)) => written.iter().all(|&(_, w)| w <= at),
            None => true,
        },
        HostWritten::Hashed { memory, size, hash } => {
            let Some(current) = hash_unwatched(memory as *const u8, size) else {
                return true;
            };
            hash == Some(current)
        }
    };
    if !unchanged {
        let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(map) = map.as_mut()
            && map.get(&key) == Some(&state)
        {
            map.remove(&key);
        }
    }
    unchanged
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Registers a title-written texture with `list` (once per list) so that at
/// submission time its staging buffer is checked for CPU modifications and,
/// only when its contents changed (including first upload), copied into its
/// host texture on the list's prelude command list.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    if !list.first_use(resource as usize) || still_host_written(resource) {
        return;
    }
    let Some(copy) = crate::resource::host_copy(resource) else {
        return;
    };
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    // Once per submission however many of its lists sample the texture;
    // `copy` keeps the host texture alive even if the title releases it
    // before submission, and `CopyTextureRegion` is only recorded onto the
    // prelude when `staging.refresh()` reports dirty contents.
    list.defer_texture(staging, copy);
}

impl Staging {
    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all). Returns `true` when the
    /// staging buffer received new contents that must be copied to the host
    /// texture (including initial upload).
    pub(crate) fn refresh(&self) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let (read, dirty) = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
        dirty
    }

    /// Returns `(bytes_read, dirty)`.
    fn refresh_now(&self) -> (usize, bool) {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some((now, written)) = crate::watch::writes(start, start + size, false) {
            if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                return (0, false);
            }
            last.asked = now;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            SCRATCH.with_borrow_mut(|read| {
                read.resize(size, 0);
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                self.detile(read);
            });
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return (size, true);
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return (0, false);
        }
        // Read through the kernel, as above: the title may unmap the pages
        // (evicting a texture a big table lists) while we read them, and a
        // direct read would fault. The copy is reused, not kept per texture.
        SCRATCH.with_borrow_mut(|tiled| {
            tiled.resize(size, 0);
            crate::host::copy_mapped(tiled.as_mut_ptr(), self.memory, size, false);
            self.changed(last, tiled)
        })
    }

    /// After reading the texture's memory into `tiled`: detiles it unless
    /// its hash is the last one's. Returns `(bytes_read, dirty)`.
    fn changed(&self, last: &mut Last, tiled: &[u8]) -> (usize, bool) {
        let size = tiled.len();
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return (size, false);
        }
        last.hash = Some(hash);
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        (size, true)
    }

    /// Records `CopyTextureRegion` for every subresource from this staging
    /// buffer into `texture` (which must already be in `COPY_DEST`).
    pub(crate) fn record_copies(&self, list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource) {
        for (i, footprint) in self.footprints.iter().enumerate() {
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: i as u32,
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(&self.buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: *footprint,
                },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
            #[cfg(test)]
            self.gpu_uploads
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    fn gpu_uploads(&self) -> u64 {
        self.gpu_uploads.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        let mut packed = Vec::new();
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let lines = rows as usize * f.Depth as usize;
            let len = if self.widened_from.is_some() && lines > 0 {
                // The last subresource allocation stops at the last row's used bytes.
                pitch * (lines - 1) + f.Width as usize * 4
            } else {
                pitch * lines
            };
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if let Err(e) = detile_subresource(
                &self.layout,
                tiled,
                mip,
                slice,
                linear,
                pitch,
                rows as usize,
                f.Depth as usize,
                self.widened_from,
                &mut packed,
            ) {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn detile_subresource(
    layout: &victory_formats::texture::Layout,
    tiled: &[u8],
    mip: u32,
    slice: u32,
    linear: &mut [u8],
    row_pitch: usize,
    rows: usize,
    depth: usize,
    widened_from: Option<u32>,
    packed: &mut Vec<u8>,
) -> Result<(), victory_formats::texture::Error> {
    let Some(packed_format) = widened_from else {
        return layout.detile(tiled, mip, slice, linear, row_pitch, row_pitch * rows);
    };
    let width = layout
        .mips
        .get(mip as usize)
        .map_or(0, |m| m.width as usize);
    let packed_pitch = width * 2;
    let packed_plane = packed_pitch * rows;
    packed.resize(packed_plane * depth, 0);
    layout.detile(tiled, mip, slice, packed, packed_pitch, packed_plane)?;
    crate::resource::expand_packed16_rows(
        packed_format,
        width,
        rows * depth,
        packed,
        packed_pitch,
        linear,
        row_pitch,
    );
    Ok(())
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    let title_format = crate::resource::pc_format(desc.format);
    let s = create_staging(host, resource, texture, title_format, layout, memory)?;
    tracing::debug!(?desc, "texture contents queued for staging upload");
    map.insert(resource as usize, s.clone());
    Some(s)
}

fn create_staging(
    host: &crate::host::Host,
    resource: This,
    texture: &ID3D12Resource,
    title_format: u32,
    layout: victory_formats::texture::Layout,
    memory: *const u8,
) -> Option<Arc<Staging>> {
    // SAFETY: a live resource.
    let mut d = unsafe { texture.GetDesc() };
    let widened_from = (d.Format.0 as u32 == crate::resource::RGBA8_UNORM
        && crate::resource::PACKED_16.contains(&title_format))
    .then_some(title_format);
    // Use the title's concrete format when the host texture is typeless so
    // `CopyTextureRegion` footprints never carry a `TYPELESS` format.
    if widened_from.is_none()
        && let Some(concrete) = crate::resource::view_format(resource)
    {
        d.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(concrete as i32);
    }
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    Some(Arc::new(Staging {
        buffer,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        widened_from,
        last: Mutex::default(),
        #[cfg(test)]
        gpu_uploads: std::sync::atomic::AtomicU64::new(0),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::Host;
    use crate::resource::HostTexture;
    use std::sync::atomic::{AtomicU32, Ordering};
    use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

    const SHADER_RESOURCE: u32 = (D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0
        | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0) as u32;

    fn make_rgba_texture(host: &Host, width: u32, height: u32) -> (Arc<HostTexture>, Texture) {
        let layout_tex = Texture::new(&TextureDesc {
            dimension: Dimension::Tex2d,
            format: DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32,
            width,
            height,
            depth_or_array: 1,
            mips: 1,
            samples: 1,
            swizzle: Swizzle::LINEAR,
        })
        .expect("linear RGBA layout");
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: plain default-heap 2D texture creation on our test device.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut resource,
                )
                .expect("create host texture");
        }
        let copy = Arc::new(HostTexture {
            texture: resource.unwrap(),
            state: AtomicU32::new(D3D12_RESOURCE_STATE_COMMON.0 as u32),
            twin: None,
        });
        (copy, layout_tex)
    }

    /// Reads a 2D RGBA8 texture back from `host` into a tightly packed row buffer.
    fn read_back_rgba(host: &Host, copy: &Arc<HostTexture>, width: u32, height: u32) -> Vec<u8> {
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut rows = 0u32;
        let mut total = 0u64;
        // SAFETY: querying footprint of a live texture.
        let desc = unsafe { copy.texture.GetDesc() };
        // SAFETY: single-subresource footprint query.
        unsafe {
            host.device.GetCopyableFootprints(
                &desc,
                0,
                1,
                0,
                Some(&mut footprint),
                Some(&mut rows),
                None,
                Some(&mut total),
            );
        }
        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain readback buffer creation.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .expect("create readback buffer");
        }
        let readback = readback.unwrap();
        let list = host.new_list().expect("readback list");
        let before = copy.state.load(Ordering::Relaxed);
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed COM pointer for the copy call.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed COM pointer for the copy call.
            pResource: unsafe { std::mem::transmute_copy(&copy.texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        // SAFETY: recording a valid readback copy on an open list.
        unsafe {
            if before != D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32 {
                list.list.ResourceBarrier(&[crate::host::transition(
                    &copy.texture,
                    D3D12_RESOURCE_STATES(before as i32),
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
            }
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            if before != D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32 {
                list.list.ResourceBarrier(&[crate::host::transition(
                    &copy.texture,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATES(before as i32),
                )]);
            }
        }
        list.close();
        host.execute(&[&list]);

        let mut ptr = std::ptr::null_mut();
        // SAFETY: GPU finished writing `readback`; map for CPU read.
        unsafe {
            readback.Map(0, None, Some(&mut ptr)).expect("map readback");
            let pitch = footprint.Footprint.RowPitch as usize;
            let row_bytes = width as usize * 4;
            let src = std::slice::from_raw_parts(ptr.cast::<u8>(), total as usize);
            let mut out = Vec::with_capacity(row_bytes * height as usize);
            for y in 0..height as usize {
                let start = footprint.Offset as usize + y * pitch;
                out.extend_from_slice(&src[start..start + row_bytes]);
            }
            readback.Unmap(0, None);
            out
        }
    }

    fn fill_linear_rgba(layout: &victory_formats::texture::Layout, mem: &mut [u8], pixel: [u8; 4]) {
        let mut linear = vec![0u8; layout.desc.width as usize * layout.desc.height as usize * 4];
        for chunk in linear.as_chunks_mut::<4>().0 {
            *chunk = pixel;
        }
        let pitch = layout.desc.width as usize * 4;
        layout
            .tile(mem, 0, 0, &linear, pitch, linear.len())
            .expect("tile pixels");
    }

    fn all_pixels_eq(bytes: &[u8], expected: [u8; 4]) -> bool {
        let (chunks, rem) = bytes.as_chunks::<4>();
        rem.is_empty() && !chunks.is_empty() && chunks.iter().all(|&px| px == expected)
    }

    /// Exercises the real D3D12 WARP prelude upload path end to end:
    /// (a) initial use uploads title memory to GPU,
    /// (b) subsequent frames with unchanged title memory record zero GPU copies,
    /// (c) multi-list submissions after a modification upload only once,
    /// (d) modifying title memory triggers a fresh GPU upload with new pixels.
    #[test]
    fn warp_skips_clean_texture_uploads_and_reuploads_on_change() {
        let Some(host) = Host::new_warp() else {
            return; // WARP software adapter unavailable here
        };
        let (width, height) = (4, 4);
        let (copy, tex) = make_rgba_texture(&host, width, height);
        let layout = tex.planes[0].layout.clone();
        let mut title_mem = vec![0u8; layout.size as usize];
        let v1 = [0x11, 0x22, 0x33, 0xff];
        fill_linear_rgba(&layout, &mut title_mem, v1);

        let staging = create_staging(
            &host,
            std::ptr::null(),
            &copy.texture,
            28,
            layout.clone(),
            title_mem.as_ptr(),
        )
        .expect("create staging");
        assert_eq!(staging.gpu_uploads(), 0);

        let make_sampling_list = || {
            let list = host.new_list().expect("new list");
            list.defer_texture(staging.clone(), copy.clone());
            list.expect_texture_state(&copy, SHADER_RESOURCE);
            list.close();
            list
        };

        // (a) Submission 1: first use must upload onto the prelude and land on GPU.
        let list1 = make_sampling_list();
        host.execute(&[&list1]);
        assert_eq!(staging.gpu_uploads(), 1, "initial use must record 1 copy");
        assert_eq!(copy.state.load(Ordering::Relaxed), SHADER_RESOURCE);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v1
        ));

        // (b) Submission 2: unchanged title memory across two lists records ZERO copies.
        let list2a = make_sampling_list();
        let list2b = make_sampling_list();
        host.execute(&[&list2a, &list2b]);
        assert_eq!(
            staging.gpu_uploads(),
            1,
            "unchanged texture must record zero CopyTextureRegion uploads"
        );
        // Re-submitting the same closed list with unchanged memory also records zero copies.
        host.execute(&[&list2a]);
        assert_eq!(staging.gpu_uploads(), 1);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v1
        ));

        // (c) Submission 3: CPU modifies title memory; two lists sample it -> exactly 1 upload.
        let v2 = [0xaa, 0xbb, 0xcc, 0xff];
        fill_linear_rgba(&layout, &mut title_mem, v2);
        let list3a = make_sampling_list();
        let list3b = make_sampling_list();
        host.execute(&[&list3a, &list3b]);
        assert_eq!(
            staging.gpu_uploads(),
            2,
            "modified texture across two lists must upload exactly once"
        );
        assert_eq!(copy.state.load(Ordering::Relaxed), SHADER_RESOURCE);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v2
        ));

        // (d) Submission 4: unchanged again -> zero additional uploads.
        let list4 = make_sampling_list();
        host.execute(&[&list4]);
        assert_eq!(staging.gpu_uploads(), 2);
    }

    /// Verifies the write-watched (`MEM_WRITE_WATCH`) dirty-tracking path on
    /// real D3D12 WARP: first use uploads, clean frames skip, and a CPU page
    /// write triggers a single fresh upload.
    #[test]
    fn warp_write_watched_texture_uploads_only_when_written() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;

        let Some(host) = Host::new_warp() else {
            return;
        };
        let (width, height) = (4, 4);
        let (copy, tex) = make_rgba_texture(&host, width, height);
        let layout = tex.planes[0].layout.clone();
        let alloc_size = (layout.size as usize).max(crate::watch::PAGE);
        // SAFETY: allocating a private write-watched page buffer freed before returning.
        let raw = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                alloc_size,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as *mut u8;
        assert!(!raw.is_null());

        let v1 = [0x33, 0x44, 0x55, 0xff];
        // SAFETY: `raw` is committed read-write memory of `alloc_size` bytes.
        unsafe {
            let mem = std::slice::from_raw_parts_mut(raw, layout.size as usize);
            fill_linear_rgba(&layout, mem, v1);
        }

        let staging = create_staging(&host, std::ptr::null(), &copy.texture, 28, layout.clone(), raw)
            .expect("create staging");

        let run_once = || {
            let list = host.new_list().expect("new list");
            list.defer_texture(staging.clone(), copy.clone());
            list.expect_texture_state(&copy, SHADER_RESOURCE);
            list.close();
            host.execute(&[&list]);
        };

        // Initial submission uploads.
        run_once();
        assert_eq!(staging.gpu_uploads(), 1);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v1
        ));

        // Unchanged write-watched memory skips GPU upload.
        run_once();
        assert_eq!(staging.gpu_uploads(), 1);

        // CPU write dirties the watched page -> next submission uploads.
        let v2 = [0xde, 0xad, 0xbe, 0xff];
        // SAFETY: `raw` is still committed read-write memory.
        unsafe {
            let mem = std::slice::from_raw_parts_mut(raw, layout.size as usize);
            fill_linear_rgba(&layout, mem, v2);
        }
        run_once();
        assert_eq!(staging.gpu_uploads(), 2);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v2
        ));

        // Clean again -> zero upload.
        run_once();
        assert_eq!(staging.gpu_uploads(), 2);

        // SAFETY: freeing our own allocation.
        unsafe { VirtualFree(raw.cast(), 0, MEM_RELEASE) };
    }

    #[test]
    fn host_written_texture_never_queues_cpu_upload() {
        let Some(host) = Host::new_warp() else {
            return;
        };
        let (copy, _) = make_rgba_texture(&host, 4, 4);
        let fake_resource = 0x1234_5678usize as This;
        host_written(fake_resource);
        let list = host.new_list().expect("new list");
        upload(&list, fake_resource, &copy.texture);
        assert!(
            !list.has_deferred_work(),
            "host-written textures must not queue CPU-to-GPU uploads"
        );
        forget(fake_resource as usize);
    }
}

#[cfg(test)]
mod widened_tests {
    use super::*;
    use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};

    #[test]
    fn detile_widened_packed16_expands_into_rgba8_staging() {
        for format in crate::resource::PACKED_16 {
            let desc = TextureDesc {
                dimension: Dimension::Tex2d,
                width: 17,
                height: 9,
                depth_or_array: 1,
                mips: 2,
                format,
                samples: 1,
                swizzle: Swizzle::SW_4KB_S,
            };
            let tex = Texture::new(&desc).unwrap();
            let layout = &tex.planes[0].layout;
            let mut tiled = vec![0u8; layout.size as usize];
            let mut expected_mips = Vec::new();

            for mip in 0..2u32 {
                let m = layout.mips[mip as usize];
                let row_bytes = m.width as usize * 2;
                let mut src_mip = vec![0u8; row_bytes * m.height as usize];
                for y in 0..m.height as usize {
                    for x in 0..m.width as usize {
                        let px = ((mip as u16 + 1) * 0x1111)
                            .wrapping_add((y as u16) * 0x0103 + x as u16);
                        let at = y * row_bytes + x * 2;
                        src_mip[at..at + 2].copy_from_slice(&px.to_le_bytes());
                    }
                }
                layout
                    .tile(&mut tiled, mip, 0, &src_mip, row_bytes, src_mip.len())
                    .unwrap();
                expected_mips.push(src_mip);
            }

            let mut packed = Vec::new();
            for (mip, src_mip) in expected_mips.iter().enumerate() {
                let m = layout.mips[mip];
                let width = m.width as usize;
                let rows = m.height as usize;
                let row_pitch = (width * 4).next_multiple_of(256);
                let exact_len = row_pitch * (rows - 1) + width * 4;
                let mut staging = vec![0x5au8; exact_len];
                detile_subresource(
                    layout,
                    &tiled,
                    mip as u32,
                    0,
                    &mut staging,
                    row_pitch,
                    rows,
                    1,
                    Some(format),
                    &mut packed,
                )
                .unwrap();
                for y in 0..rows {
                    let mut expected = vec![0u8; width * 4];
                    crate::resource::expand_packed16(
                        format,
                        &src_mip[y * width * 2..][..width * 2],
                        &mut expected,
                    );
                    assert_eq!(
                        &staging[y * row_pitch..][..width * 4],
                        &expected[..],
                        "format {format} mip {mip} row {y}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{HRESULT, S_OK, guid, slot};
    use crate::resource::{
        buffer_bytes, host_resource, host_transition, test_buffer, test_texture_at, texture_layout,
        texture_memory,
    };

    type Release = extern "system" fn(This) -> u32;

    const RGBA8: u32 = 28;
    const W: u32 = 4;
    const H: u32 = 4;
    const PITCH: u32 = 256;

    #[repr(C)]
    struct Location {
        resource: This,
        kind: u32,
        footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    }

    fn rows_footprint() -> D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(RGBA8 as i32),
                Width: W,
                Height: H,
                Depth: 1,
                RowPitch: PITCH,
            },
        }
    }

    fn pattern(seed: u8) -> Vec<u8> {
        let mut linear = vec![0u8; (H * PITCH) as usize];
        for row in 0..H as usize {
            for col in 0..(W as usize * 4) {
                linear[row * PITCH as usize + col] = seed.wrapping_add((row * 17 + col * 3) as u8);
            }
        }
        linear
    }

    fn write_tiled(resource: This, mem: *mut u8, size: usize, linear: &[u8]) {
        let (_, desc) = texture_memory(resource).expect("texture memory");
        let layout = texture_layout(&desc).expect("texture layout").planes[0]
            .layout
            .clone();
        assert_eq!(layout.size as usize, size);
        let mut tiled = vec![0u8; size];
        layout
            .tile(
                &mut tiled,
                0,
                0,
                linear,
                PITCH as usize,
                (H * PITCH) as usize,
            )
            .expect("tile");
        // SAFETY: `mem` points to at least `size` writable bytes owned by the test.
        unsafe { std::ptr::copy_nonoverlapping(tiled.as_ptr(), mem, size) };
    }

    fn copy_buffer_to_texture(queue: This, allocator: This, list: This, dst: This, linear: &[u8]) {
        let src = test_buffer(linear.len() as u64);
        let (data, _) = buffer_bytes(src).expect("buffer bytes");
        // SAFETY: `src` is a committed buffer of `linear.len()` bytes.
        unsafe { std::ptr::copy_nonoverlapping(linear.as_ptr(), data, linear.len()) };

        let rows = rows_footprint();
        let to = Location {
            resource: dst,
            kind: 0,
            footprint: rows,
        };
        let from = Location {
            resource: src,
            kind: 1,
            footprint: rows,
        };
        type Copy =
            extern "system" fn(This, *const Location, u32, u32, u32, *const Location, *const u8);
        slot::<Copy>(list, 0x80)(list, &to, 0, 0, 0, &from, std::ptr::null());
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        assert_eq!(
            slot::<extern "system" fn(This, This, This) -> HRESULT>(list, 0x50)(
                list,
                allocator,
                std::ptr::null(),
            ),
            S_OK
        );
        assert_eq!(slot::<Release>(src, 0x10)(src), 0);
    }

    fn run_upload(host: &crate::host::Host, resource: This) {
        let list = host.new_list().expect("new host list");
        let tex = host_resource(resource).expect("host texture");
        upload(&list, resource, tex);
        list.close();
        host.execute(&[&list]);
    }

    fn read_back(host: &crate::host::Host, resource: This) -> Vec<u8> {
        use windows::core::Interface;
        let bytes = (H * PITCH) as u64;
        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .expect("readback buffer");
        }
        let readback = readback.expect("readback");
        let host_image = host_resource(resource).expect("host texture");
        let back = host.new_list().expect("readback list");
        const COPY_SOURCE: u32 = 0x800;
        let before = host_transition(resource, COPY_SOURCE).map(|(_, b)| b);
        let rows = rows_footprint();
        let out_to = Location {
            resource: readback.as_raw().cast_const().cast(),
            kind: 1,
            footprint: rows,
        };
        let out_from = Location {
            resource: host_image.as_raw().cast_const().cast(),
            kind: 0,
            footprint: rows,
        };
        // SAFETY: an open list, live resources, locations in the host layout.
        unsafe {
            if let Some(before) = before {
                back.list.ResourceBarrier(&[crate::host::transition(
                    host_image,
                    D3D12_RESOURCE_STATES(before as i32),
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
            }
            back.list.CopyTextureRegion(
                (&raw const out_to).cast(),
                0,
                0,
                0,
                (&raw const out_from).cast(),
                None,
            );
        }
        back.close();
        host.execute(&[&back]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy finished (`execute` waits); the buffer has `bytes` bytes.
        unsafe {
            readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: bytes as usize,
                    }),
                    Some(&mut mapped),
                )
                .expect("map readback");
            let slice = std::slice::from_raw_parts(mapped.cast::<u8>(), bytes as usize).to_vec();
            readback.Unmap(0, None);
            slice
        }
    }

    fn assert_rows_eq(actual: &[u8], expected: &[u8], label: &str) {
        for row in 0..H as usize {
            let at = row * PITCH as usize;
            let row_bytes = W as usize * 4;
            assert_eq!(
                &actual[at..at + row_bytes],
                &expected[at..at + row_bytes],
                "{label}: row {row}"
            );
        }
    }

    fn make_queue_and_list() -> (This, This, This) {
        let queue_iid = guid(
            0x0ec870a6,
            0x5d7e,
            0x4c22,
            [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
        );
        let allocator_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let (mut queue, mut allocator, mut list) =
            (std::ptr::null(), std::ptr::null(), std::ptr::null());
        let queue_desc = [0u32; 4];
        assert_eq!(
            crate::queue::create(
                std::ptr::null(),
                queue_desc.as_ptr().cast(),
                &queue_iid,
                &mut queue,
            ),
            S_OK
        );
        assert_eq!(
            crate::list::create_allocator(std::ptr::null(), 0, &allocator_iid, &mut allocator),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(
                std::ptr::null(),
                0,
                0,
                allocator,
                std::ptr::null(),
                &list_iid,
                &mut list,
            ),
            S_OK
        );
        (queue, allocator, list)
    }

    /// A write-watched texture that receives a GPU copy keeps the copied
    /// pixels while untouched, then resumes detiling once the title writes
    /// its memory again (issue #758).
    #[test]
    fn host_written_watched_texture_uploads_after_title_write() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;

        let Some(host) = host() else {
            return;
        };
        let alloc_size = 64 << 10;
        // SAFETY: fresh write-watched allocation, freed after the resource is released.
        let mem = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                alloc_size,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        assert!(!mem.is_null());

        let image = test_texture_at(mem as u64, u64::from(W), H, RGBA8, 0);
        let (_, desc) = texture_memory(image).expect("texture memory");
        let layout_size = texture_layout(&desc).expect("layout").planes[0].layout.size as usize;
        assert!(layout_size <= alloc_size);

        let (queue, allocator, list) = make_queue_and_list();

        // Title writes pattern A *before* the GPU copy: must not falsely
        // invalidate the GPU copy that comes next.
        let pat_a = pattern(0x10);
        let pat_b = pattern(0x50);
        let pat_c = pattern(0x90);
        let pat_d = pattern(0xc0);
        let pat_e = pattern(0xe0);

        write_tiled(image, mem, layout_size, &pat_a);
        copy_buffer_to_texture(queue, allocator, list, image, &pat_b);

        // No title write since the GPU copy: upload must keep pattern B.
        run_upload(host, image);
        assert_rows_eq(
            &read_back(host, image),
            &pat_b,
            "keeps GPU copy before write",
        );

        // Title writes pattern C into the texture's memory: upload must detile C.
        write_tiled(image, mem, layout_size, &pat_c);
        run_upload(host, image);
        assert_rows_eq(&read_back(host, image), &pat_c, "uploads after title write");

        // Second GPU copy -> keep D -> later title write -> upload E (with
        // Staging already allocated).
        copy_buffer_to_texture(queue, allocator, list, image, &pat_d);
        run_upload(host, image);
        assert_rows_eq(&read_back(host, image), &pat_d, "keeps second GPU copy");

        write_tiled(image, mem, layout_size, &pat_e);
        run_upload(host, image);
        assert_rows_eq(
            &read_back(host, image),
            &pat_e,
            "uploads after second title write",
        );

        assert_eq!(slot::<Release>(image, 0x10)(image), 0);
        assert_eq!(slot::<Release>(list, 0x10)(list), 0);
        assert_eq!(slot::<Release>(allocator, 0x10)(allocator), 0);
        assert_eq!(slot::<Release>(queue, 0x10)(queue), 0);
        // SAFETY: `image` is released; freeing our allocation.
        unsafe { VirtualFree(mem.cast(), 0, MEM_RELEASE) };
    }

    /// An unwatched texture (physical-page view / content-hashed path) also
    /// keeps GPU-copied pixels until its console memory changes, then uploads
    /// the new contents (issue #758).
    #[test]
    fn host_written_unwatched_texture_uploads_after_title_write() {
        let Some(host) = host() else {
            return;
        };
        let mut backing = vec![0u8; 64 << 10];
        let mem = backing.as_mut_ptr();
        let image = test_texture_at(mem as u64, u64::from(W), H, RGBA8, 0);
        let (_, desc) = texture_memory(image).expect("texture memory");
        let layout_size = texture_layout(&desc).expect("layout").planes[0].layout.size as usize;
        assert!(layout_size <= backing.len());

        let (queue, allocator, list) = make_queue_and_list();
        let pat_a = pattern(0x21);
        let pat_b = pattern(0x61);
        let pat_c = pattern(0xa1);

        // Initial CPU upload creates Staging, then GPU copy overwrites host texture.
        write_tiled(image, mem, layout_size, &pat_a);
        run_upload(host, image);
        assert_rows_eq(&read_back(host, image), &pat_a, "initial CPU upload");

        copy_buffer_to_texture(queue, allocator, list, image, &pat_b);
        run_upload(host, image);
        assert_rows_eq(
            &read_back(host, image),
            &pat_b,
            "keeps GPU copy while unchanged",
        );

        // Later title write changes content hash: upload must detile pattern C.
        write_tiled(image, mem, layout_size, &pat_c);
        run_upload(host, image);
        assert_rows_eq(
            &read_back(host, image),
            &pat_c,
            "uploads after unwatched write",
        );

        assert_eq!(slot::<Release>(image, 0x10)(image), 0);
        assert_eq!(slot::<Release>(list, 0x10)(list), 0);
        assert_eq!(slot::<Release>(allocator, 0x10)(allocator), 0);
        assert_eq!(slot::<Release>(queue, 0x10)(queue), 0);
    }

    /// Non-title-written textures (such as render targets) stay sticky in
    /// `HOST_WRITTEN` without hashing or write-watch queries, and clear on
    /// release.
    #[test]
    fn host_written_render_target_stays_sticky_until_forgotten() {
        let Some(_) = host() else {
            return;
        };
        let mut backing = vec![0u8; 64 << 10];
        const ALLOW_RENDER_TARGET: u32 = 1;
        let rt = test_texture_at(
            backing.as_mut_ptr() as u64,
            u64::from(W),
            H,
            RGBA8,
            ALLOW_RENDER_TARGET,
        );
        host_written(rt);
        assert!(still_host_written(rt));
        backing[0] ^= 0xff;
        assert!(still_host_written(rt));
        assert_eq!(slot::<Release>(rt, 0x10)(rt), 0);
        assert!(!still_host_written(rt));
    }
}
