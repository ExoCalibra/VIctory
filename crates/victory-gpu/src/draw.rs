//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, depth bounds, topology, targets) is set on the host
//! list as the title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::emit::{AddressView, UserData};
use victory_shader::resource::RootIndexedView;
use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_UNKNOWN,
};
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    pub(crate) address: u64,
    pub(crate) size: u32,
    pub(crate) stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

const SHADER_RESOURCE: u32 = 0x40 | 0x80;
const DEPTH_WRITE: u32 = 0x10;
const DEPTH_READ: u32 = 0x20;

#[derive(Clone, Copy)]
struct DepthTarget {
    resource: usize,
    state: u32,
}

pub(crate) enum UavClear {
    Uint([u32; 4]),
    Float([f32; 4]),
}
/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<(usize, Option<std::ops::Range<u32>>)>,
    depth: Option<DepthTarget>,
    /// The current draw also samples its read-only depth target.
    depth_sampled: bool,
    /// None leaves the host list's default bounds (0..1) in place.
    depth_bounds: Option<(f32, f32)>,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, slot count, contents) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u32, u64), u32>,
    /// The dynamic slots the list holds (title heap, first, count), given
    /// back when the list is reset or released, or earlier when a heap runs
    /// short after the list ran (see [`crate::descriptor::lease`]).
    lease: std::sync::Arc<crate::descriptor::lease::Lease>,
    /// A fresh table of the stage being bound could not be made.
    table_failed: bool,
}

impl Drop for State {
    fn drop(&mut self) {
        for (heap, first, n) in self.lease.take() {
            crate::descriptor::release_dynamic(heap, first, n);
        }
    }
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootView {
    Cbv,
    Srv,
    Uav,
}

/// Permanently mapped zero-filled read-only fallback buffer (for unset or
/// unmapped root CBVs/SRVs) and writable sink buffer (for unset or unmapped
/// root UAVs, kept separate so shader writes never dirty the zero buffer).
fn fallback_root_va(uav: bool) -> Option<u64> {
    const SIZE: u64 = 4 << 20;
    static ZERO: std::sync::OnceLock<Option<(ID3D12Resource, u64)>> = std::sync::OnceLock::new();
    static SINK: std::sync::OnceLock<Option<(ID3D12Resource, u64)>> = std::sync::OnceLock::new();
    let slot = if uav { &SINK } else { &ZERO };
    slot.get_or_init(|| {
        let host = crate::host::host()?;
        let (buffer, data) = if uav {
            host.shared_buffer(SIZE)?
        } else {
            host.upload_buffer(SIZE)?
        };
        let label = if uav {
            "fallback sink root uav"
        } else {
            "fallback zero root view"
        };
        crate::host::name(&buffer, || label.into());
        // SAFETY: `data` points to `SIZE` mapped writable bytes, and `buffer` is live.
        let gpu = unsafe {
            data.write_bytes(0, SIZE as usize);
            buffer.GetGPUVirtualAddress()
        };
        Some((buffer, gpu))
    })
    .as_ref()
    .map(|&(_, gpu)| gpu)
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory. Constant buffers refresh up to the D3D12 64 KiB
/// maximum; SRVs and UAVs refresh up to [`ROOT_SRV_MAX`] / [`ROOT_UAV_MAX`]
/// of the backing buffer (dirty-page watched) and UAVs also schedule
/// write-back of that range (see [`root_len`]).
fn root_view(list: &HostList, address: u64, kind: RootView) -> Option<u64> {
    if address == 0 {
        return None;
    }
    let shadow = crate::shadow::find(address, 1)?;
    let gpu = shadow.gpu(address);
    match kind {
        RootView::Cbv => {
            let len = root_len(shadow.left(address), RootView::Cbv);
            if gpu.is_multiple_of(256) {
                shadow.refresh(list, address, len);
                list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
                return Some(gpu);
            }
            let (gpu, to) = list.scratch(len)?;
            list.defer_fill(to, address as *const u8, len as usize);
            Some(gpu)
        }
        RootView::Srv => {
            if !gpu.is_multiple_of(4) {
                return None;
            }
            let len = root_len(shadow.left(address), RootView::Srv);
            shadow.refresh(list, address, len);
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
            Some(gpu)
        }
        RootView::Uav => {
            if !gpu.is_multiple_of(4) {
                return None;
            }
            let len = root_len(shadow.left(address), RootView::Uav);
            shadow.refresh(list, address, len);
            shadow.gpu_writes(list, address, len);
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_WRITE);
            Some(gpu)
        }
    }
}

fn emit_uav_barrier(list: &HostList) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    };
    // SAFETY: an open command list and a global UAV barrier.
    unsafe { list.list.ResourceBarrier(&[barrier]) };
}

/// Addresses below this are taken as unset (null, or no buffer): a root SRV
/// at an address in root constants is not bound there and its draw is
/// dropped.
const MIN_ADDRESS: u64 = 0x1_0000;

/// The address a root SRV at an address in root constants starts at: the
/// two constants' current values, low half first. `None` when the title has
/// not set either or the address is below [`MIN_ADDRESS`].
fn pair_address(roots: &[Root], view: &AddressView) -> Option<u64> {
    let half = |u: &UserData| match roots.get(usize::from(u.slot))? {
        Root::Constants(c) => c.get(u.dword as usize).copied(),
        _ => None,
    };
    let address = u64::from(half(&view.lo)?) | u64::from(half(&view.hi)?) << 32;
    (address >= MIN_ADDRESS).then_some(address)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// Largest root SRV range refreshed per bind.
const ROOT_SRV_MAX: u64 = 4 << 20;
/// Largest root UAV range refreshed, kept and copied back per submission.
const ROOT_UAV_MAX: u64 = 2 << 20;

/// The bytes of a root view's buffer copy (`left` remaining after its
/// address) that a bind refreshes: constant buffers up to the D3D12 64 KiB
/// maximum, SRVs and UAVs up to their caps, warning once when a cap cuts.
///
/// shortcut: a root view has no size, and the translated shader does not
/// yet report the range it reads or writes, so SRVs and UAVs are capped
/// rather than taking the rest of the copy: behind most title buffers is
/// one host copy of well over 100 MiB, and a UAV's range is kept and
/// compared on every submission. The UAV cap is the smaller for that.
fn root_len(left: u64, kind: RootView) -> u64 {
    let cap = match kind {
        RootView::Cbv => return left.min(1 << 16),
        RootView::Srv => ROOT_SRV_MAX,
        RootView::Uav => ROOT_UAV_MAX,
    };
    if left > cap {
        match kind {
            RootView::Srv => once!(left, cap, "a root SRV cut to its refresh cap"),
            _ => once!(left, cap, "a root UAV cut to its refresh and copy-back cap"),
        }
    }
    left.min(cap)
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    #[cfg(test)]
    pub(crate) fn current_pipeline(&self) -> This {
        self.pipeline as This
    }

    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The list's slots, to start and finish its submissions.
    pub(crate) fn lease(&self) -> &std::sync::Arc<crate::descriptor::lease::Lease> {
        &self.lease
    }

    /// `n` consecutive dynamic slots of `host` (the host heap of the title
    /// heap at address `key`), held by this list. When the heap has no room,
    /// the slots of the oldest list that already ran are taken back first,
    /// one list at a time, instead of dropping the draw.
    fn take_dynamic(
        &mut self,
        host: &crate::descriptor::host::HostHeap,
        key: usize,
        n: u32,
    ) -> Option<u32> {
        let mut reclaimed = false;
        loop {
            if let Some(first) = host.dynamic(n) {
                self.lease.add(key, first, n);
                return Some(first);
            }
            let more = crate::descriptor::lease::reclaim(key, |(heap, first, n)| {
                if heap == key {
                    host.release(first, n);
                } else {
                    crate::descriptor::release_dynamic(heap, first, n);
                }
            });
            if !more {
                return None;
            }
            if !reclaimed {
                reclaimed = true;
                once!("dynamic descriptor slots of lists that already ran are taken back");
            }
        }
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        let host = match topology {
            RECTANGLES => 4,
            t => crate::primitive::ExpandedTopology::from_console(t)
                .map_or(t, crate::primitive::ExpandedTopology::host_topology),
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.targets.clear();
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            if record.resource.is_null() {
                // A typed null RTV discards writes without changing the output slot.
                if let Some(desc) = record.desc() {
                    rtvs.push(host.rtv(None, Some(&desc)));
                }
                continue;
            }
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                continue;
            };
            let desc = record.desc();
            self.targets.push((
                record.resource as usize,
                crate::descriptor::rtv_mips(texture, desc.as_ref()),
            ));
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            rtvs.push(host.rtv(Some(texture), desc.as_ref()));
        }
        self.depth = None;
        self.depth_sampled = false;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            let desc = record.depth_desc();
            self.depth = Some(DepthTarget {
                resource: record.resource as usize,
                state: depth_state(texture, desc.as_ref()),
            });
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            Some(host.dsv(texture, desc.as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// The value of user-data dword `u` of a translated shader: a root
    /// argument, or a bound heap's guard slot.
    fn user_dword(&self, compute: bool, u: &victory_shader::emit::UserData) -> u32 {
        let guard = |samplers: bool| {
            let h = self.heaps[usize::from(samplers)];
            (h != 0)
                .then(|| crate::descriptor::heap(h as This).host())
                .flatten()
                .map_or(0, |h| h.guard_slot())
        };
        match u.root {
            RootType::ResourceHeap => guard(false),
            RootType::SamplerHeap => guard(true),
            _ => self.user_value(compute, u.slot, u.root, u.dword),
        }
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// Table `slot` starts at a title slot the host heap does not have.
    fn past_host_heap(&self, compute: bool, slot: u8) -> bool {
        let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
            return false;
        };
        self.heap_of(table)
            .and_then(|heap| Some((heap.host()?, heap.index(table)?)))
            .is_some_and(|(host, index)| index >= host.count())
    }

    /// The bytes root view `slot` holds from its start, as [`root_view`]
    /// binds it. Shaders check their accesses against it: a root view has
    /// no bounds on the host, and an access past the memory it maps is a
    /// GPU page fault, which some hosts stall on until the driver resets
    /// the device. 0 for a root view the title did not set or outside mapped
    /// title memory.
    fn root_view_bytes(&self, compute: bool, slot: u8, kind: RootView) -> u32 {
        let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) else {
            return 0;
        };
        if address == 0 {
            return 0;
        }
        let Some(shadow) = crate::shadow::find(address, 1) else {
            return 0;
        };
        u32::try_from(root_len(shadow.left(address), kind)).unwrap_or(u32::MAX)
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    pub(crate) fn set_depth_bounds(&mut self, list: &HostList, min: f32, max: f32) {
        self.depth_bounds = Some((min, max));
        if list.depth_bounds.is_none() {
            crate::host::warn_depth_bounds();
        }
        self.apply_depth_bounds(list);
    }

    fn apply_depth_bounds(&self, list: &HostList) {
        if let (Some(list), Some((min, max))) = (&list.depth_bounds, self.depth_bounds) {
            // SAFETY: the host supports depth bounds and the list is open.
            unsafe { list.OMSetDepthBounds(min, max) };
        }
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(
                mesh,
                "a draw or mesh dispatch with another kind of pipeline"
            );
            return false;
        }
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            // After `bind`: a shader may have written the buffer before.
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        for (resource, mips) in &self.targets {
            crate::list::ensure_mips(list, *resource as This, RENDER_TARGET, mips.clone());
            #[cfg(feature = "diagnostics")]
            crate::producers::live_target(*resource);
        }
        self.ensure_depth_state(list);
        true
    }

    fn ensure_depth_state(&self, list: &HostList) {
        if let Some(depth) = self.depth {
            let state = depth.state | if self.depth_sampled { SHADER_RESOURCE } else { 0 };
            crate::list::ensure_state(list, depth.resource as This, state);
        }
    }

    fn read_only_depth(&self, compute: bool) -> Option<usize> {
        self.depth
            .filter(|depth| !compute && depth.state == DEPTH_READ)
            .map(|depth| depth.resource)
    }

    /// Binds the title's shader-visible heaps when they change.
    fn bind_heaps(&mut self, list: &HostList) {
        let l = &list.list;
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
    }

    /// Records a clear of a resource-backed texture UAV in the bound heap.
    pub(crate) fn clear_texture_uav(
        &mut self,
        list: &HostList,
        gpu: u64,
        cpu: u64,
        resource: This,
        value: UavClear,
        rects: &[RECT],
    ) {
        let heap_id = self.heaps[0];
        if heap_id == 0 {
            return;
        }
        let heap = crate::descriptor::heap(heap_id as This);
        if heap.index(gpu).is_none() {
            return;
        }
        let (Some(View::Unordered(cpu_resource, _, words)), Some(View::Unordered(gpu_resource, _, _))) =
            (crate::descriptor::view(cpu), crate::descriptor::view(gpu))
        else {
            return;
        };
        if cpu_resource != resource as usize || gpu_resource != cpu_resource {
            return;
        }
        let (Some(texture), Some(native_heap)) =
            (crate::resource::host_resource(resource), heap.host())
        else {
            return;
        };
        let Some((slot, gpu, cpu)) = native_heap.clear_texture_uav(list, texture, words.as_ref())
        else {
            return;
        };
        self.lease.add(heap_id, slot, 1);
        self.bind_heaps(list);
        crate::list::ensure_state(list, resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32);
        // SAFETY: matched views, a live texture in UAV state, and the
        // shader-visible descriptor's heap is bound on this list.
        unsafe {
            match value {
                UavClear::Uint(values) => {
                    list.list.ClearUnorderedAccessViewUint(gpu, cpu, texture, &values, rects);
                }
                UavClear::Float(values) => {
                    list.list.ClearUnorderedAccessViewFloat(gpu, cpu, texture, &values, rects);
                }
            }
        }
    }

    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a table it reads could
    /// not be made, or a root SRV at an address in root constants has no
    /// address to bind.
    pub(crate) fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        self.depth_sampled = false;
        let mut bound = true;
        let l = &list.list;
        self.bind_heaps(list);
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            // Also reached after the indirect clamp invalidates the pipeline.
            if !compute {
                self.apply_depth_bounds(list);
            }
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        let mut bound_params = 0u64;
        if p.stages.iter().any(|s| s.translated().resource_heap)
            && self.heaps[0] != 0
            && let Some(host_heap) = crate::descriptor::heap(self.heaps[0] as This).host_arc()
        {
            host_heap.prepare_bindless_record(list);
            let writes = p.stages.iter().any(|s| s.translated().writes_memory);
            list.defer_bindless(&host_heap, writes);
        }
        #[cfg(feature = "diagnostics")]
        crate::producers::set_reader(&p.name);
        for stage in &p.stages {
            let t = stage.translated();
            self.table_failed = false;
            let made = self.tables(list, &t.used, &t.host.root_indexed_views, compute);
            if self.table_failed {
                static DROPPED_TABLES: AtomicU64 = AtomicU64::new(0);
                let n = DROPPED_TABLES.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::warn!(
                        dropped = n,
                        "a draw or dispatch dropped: no dynamic slots for its tables"
                    );
                }
                return false;
            }
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        // The title's slots past a capped host heap have no
                        // host slot: a shader given that offset reads past
                        // the heap's end (garbage descriptors on NVIDIA; the
                        // device hangs, #71).
                        None if u.root == RootType::Table
                            && self.past_host_heap(compute, u.slot) =>
                        {
                            once!(
                                slot = u.slot,
                                "a draw or dispatch dropped: its table lies past the host heap"
                            );
                            return false;
                        }
                        None if u.root == RootType::Descriptor => {
                            let kind = if stage.root_uavs.iter().any(|r| r.0 == u.slot) {
                                RootView::Uav
                            } else if stage.root_srvs.iter().any(|r| r.0 == u.slot) {
                                RootView::Srv
                            } else {
                                RootView::Cbv
                            };
                            self.root_view_bytes(compute, u.slot, kind)
                        }
                        None => self.user_dword(compute, u),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, kind) in [
                (&stage.root_cbvs, RootView::Cbv),
                (&stage.root_srvs, RootView::Srv),
                (&stage.root_uavs, RootView::Uav),
            ] {
                for &(slot, param) in views {
                    if param < 64 {
                        if bound_params & (1u64 << param) != 0 {
                            continue;
                        }
                        bound_params |= 1u64 << param;
                    }
                    let uav = kind == RootView::Uav;
                    let gpu = match self.roots_ref(compute).get(usize::from(slot)) {
                        Some(&Root::View(address)) if address != 0 => {
                            match root_view(list, address, kind) {
                                Some(gpu) => gpu,
                                None => {
                                    once!(
                                        slot,
                                        address = format_args!("{address:#x}"),
                                        ?kind,
                                        "root view outside mapped title memory; binding zeroed fallback"
                                    );
                                    let Some(gpu) = fallback_root_va(uav) else {
                                        return false;
                                    };
                                    gpu
                                }
                            }
                        }
                        Some(Root::View(_)) => {
                            once!(
                                slot,
                                ?kind,
                                "null root view address; binding zeroed fallback"
                            );
                            let Some(gpu) = fallback_root_va(uav) else {
                                return false;
                            };
                            gpu
                        }
                        _ => {
                            once!(
                                slot,
                                ?kind,
                                "a shader reads a root view the title did not set; binding zeroed fallback"
                            );
                            let Some(gpu) = fallback_root_va(uav) else {
                                return false;
                            };
                            gpu
                        }
                    };
                    // SAFETY: an open list; `gpu` lies in a live host or fallback buffer.
                    unsafe {
                        match (compute, kind) {
                            (false, RootView::Cbv) => {
                                l.SetGraphicsRootConstantBufferView(param, gpu)
                            }
                            (false, RootView::Srv) => {
                                l.SetGraphicsRootShaderResourceView(param, gpu)
                            }
                            (false, RootView::Uav) => {
                                l.SetGraphicsRootUnorderedAccessView(param, gpu)
                            }
                            (true, RootView::Cbv) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, RootView::Srv) => l.SetComputeRootShaderResourceView(param, gpu),
                            (true, RootView::Uav) => {
                                l.SetComputeRootUnorderedAccessView(param, gpu)
                            }
                        }
                    }
                }
            }
            for &(view, param) in &stage.address_srvs {
                let gpu = pair_address(self.roots_ref(compute), &view)
                    .and_then(|address| root_view(list, address, RootView::Srv));
                let Some(gpu) = gpu else {
                    once!(
                        slot = view.lo.slot,
                        dword = view.lo.dword,
                        "a shader reads through an address in root constants that is unset; draw dropped"
                    );
                    bound = false;
                    continue;
                };
                // SAFETY: an open list; the address is in live memory.
                unsafe {
                    if compute {
                        l.SetComputeRootShaderResourceView(param, gpu);
                    } else {
                        l.SetGraphicsRootShaderResourceView(param, gpu);
                    }
                }
            }
        }
        bound
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            self.note_dropped_writes();
            self.frame_op(&format!("dispatch {x}x{y}x{z}"), true, false);
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_drop(self.pipeline as This);
                self.note_dropped_writes();
                self.frame_op(&format!("dispatch {x}x{y}x{z}"), true, false);
            }
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            #[cfg(feature = "diagnostics")]
            self.frame_op_why(
                &format!("dispatch {x}x{y}x{z}"),
                true,
                false,
                Some("a graphics pipeline"),
            );
            return;
        }
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_drop(self.pipeline as This);
                self.frame_op_why(
                    &format!("dispatch {x}x{y}x{z}"),
                    true,
                    false,
                    Some("a table or address it reads could not be bound"),
                );
            }
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        #[cfg(feature = "diagnostics")]
        self.frame_op(&format!("dispatch {x}x{y}x{z}"), true, true);
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        if p.has_root_uavs() {
            emit_uav_barrier(list);
        }
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// In frames listed for capture, at debug level: one line per draw or
    /// dispatch with its render targets and depth (named as in
    /// `VICTORY_DUMP_TARGETS` files) and whether it ran, else why it was
    /// dropped (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn frame_op(&self, kind: &str, compute: bool, ran: bool) {
        self.frame_op_why(kind, compute, ran, None);
    }

    /// [`State::frame_op`] with the reason a drop happened before a pipeline
    /// was looked at (else the pipeline's).
    #[cfg(feature = "diagnostics")]
    fn frame_op_why(&self, kind: &str, compute: bool, ran: bool, why: Option<&str>) {
        if !tracing::enabled!(tracing::Level::DEBUG)
            || !crate::capture::listed(crate::queue::presented() + 1)
        {
            return;
        }
        let targets: Vec<String> = if compute {
            Vec::new()
        } else {
            self.targets.iter().map(|(t, _)| format!("{t:#x}")).collect()
        };
        let outcome = if ran {
            "ran".to_owned()
        } else {
            match why {
                Some(why) => format!("DROPPED: {why}"),
                None => format!(
                    "DROPPED: {}",
                    crate::pipeline::drop_reason(self.pipeline as This)
                ),
            }
        };
        let text = self.op_text(kind.to_owned(), compute);
        tracing::debug!(
            "frame {} op: pipeline {:#x} topology {} targets {targets:?} depth {:#x} -> {outcome} || {text}",
            crate::queue::presented() + 1,
            self.pipeline,
            self.topology,
            self.depth.map_or(0, |d| d.resource)
        );
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = self.user_dword(compute, u);
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, _) in stage
                .root_cbvs
                .iter()
                .chain(&stage.root_srvs)
                .chain(&stage.root_uavs)
            {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
            for (view, _) in &stage.address_srvs {
                let address = pair_address(self.roots_ref(compute), view);
                let host =
                    address.and_then(|a| crate::shadow::find(a, 1).map(|s| (s.gpu(a), s.left(a))));
                let _ = write!(
                    text,
                    " u{}:u{}={address:#x?} host {host:#x?}",
                    view.lo.index, view.hi.index
                );
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        text
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    if crate::pipeline::host_pipeline(self.pipeline as This)
                        .is_some_and(|p| p.has_root_uavs())
                    {
                        emit_uav_barrier(list);
                    }
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        #[cfg(feature = "diagnostics")]
        self.frame_op("mesh", false, ran);
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
            #[cfg(feature = "diagnostics")]
            self.note_dropped_writes();
        }
    }

    /// Diagnostics (`VICTORY_PRODUCERS=1`): what the dropped draw or
    /// dispatch bound now would have written, for [`crate::producers`].
    #[cfg(feature = "diagnostics")]
    fn note_dropped_writes(&self) {
        if !crate::producers::on() {
            return;
        }
        let mut writer = String::new();
        for (stage, hash, used) in crate::pipeline::native_used(self.pipeline as This) {
            let compute = stage == "compute";
            if writer.is_empty() || stage == "pixel" || compute {
                writer = format!("{stage} {hash}");
            }
            for u in used
                .iter()
                .filter(|u| u.kind == ResourceType::Uav && u.root == RootType::Table)
            {
                let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                    continue;
                };
                let handle = table + u64::from(u.table_offset_dwords) * 4;
                if let Some(view) = crate::descriptor::view(handle) {
                    crate::producers::dropped_write(&format!("{stage} {hash}"), &view);
                }
            }
        }
        if !writer.starts_with("compute") {
            for &(t, _) in &self.targets {
                crate::producers::dropped_target(&writer, t);
            }
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&mut self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true) {
            self.depth_sampled |= readable(list, u.kind, &view, self.read_only_depth(compute));
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer that is not 256-byte aligned on the host (or whose
    /// 256-byte-padded range leaves its shadow buffer) are made afresh for
    /// the draw in dynamic slots, with the constant data copied to upload
    /// memory when the list is submitted: the host wants constant buffers
    /// 256-byte aligned, the console only 16. Tables whose constant buffers
    /// already meet the host's alignment mirror straight into the title's
    /// own heap slots, avoiding dynamic slot churn on capped view heaps.
    /// Returns the fresh tables' new byte offsets by root parameter; the
    /// other tables' slots are made from the title's.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once: its constant data is copied at submission either way, and the
    /// big tables the world binds would otherwise take hundreds of slots per
    /// draw, wrapping the dynamic slots onto lists not yet submitted.
    pub(crate) fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        indexed: &[RootIndexedView],
        compute: bool,
    ) -> Vec<(u8, u32)> {
        let read_only_depth = self.read_only_depth(compute);
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = Vec::new();
        let mut cbv_slots: Vec<u8> = used
            .iter()
            .filter(tables)
            .filter(|u| u.kind == ResourceType::Cbv)
            .map(|u| u.slot)
            .collect();
        cbv_slots.sort_unstable();
        cbv_slots.dedup();
        for slot in cbv_slots {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let n = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .map(|u| u.table_offset_dwords / 8 + 1)
                .max()
                .unwrap_or(1);
            let needs_fresh = (0..n).any(|k| {
                matches!(
                    crate::descriptor::view(table + u64::from(k) * 32),
                    Some(View::Constant(address, size))
                        if size == 0
                            || !crate::shadow::find(address, u64::from(size.next_multiple_of(256)))
                                .is_some_and(|s| s.gpu(address).is_multiple_of(256))
                )
            });
            if needs_fresh {
                fresh.push(slot);
            }
        }
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA, 2,048 samplers anywhere)
        // are made afresh too: the title's slots there have no host slot of
        // their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        // Root-selected entries beyond a capped host heap also require a fresh table.
        for idx in indexed {
            if fresh.contains(&idx.table_slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(idx.table_slot))
            else {
                continue;
            };
            let constant = self.user_value(
                compute,
                idx.constant_slot,
                RootType::Constants,
                idx.constant_dword,
            );
            let slot_off = idx.byte_offset.wrapping_add(constant << idx.shift) / 32;
            let handle = table + u64::from(slot_off) * 32;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(idx.table_slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        // Mirror root-selected entries for tables that stayed in place.
        for idx in indexed {
            if fresh.contains(&idx.table_slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(idx.table_slot))
            else {
                continue;
            };
            let constant = self.user_value(
                compute,
                idx.constant_slot,
                RootType::Constants,
                idx.constant_dword,
            );
            let slot_off = idx.byte_offset.wrapping_add(constant << idx.shift) / 32;
            let handle = table + u64::from(slot_off) * 32;
            let Some(heap) = self.heap_of(handle) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
                continue;
            };
            if let Some(view) = host.mirror(index, handle, true) {
                let kind = match view {
                    View::Constant(..) => ResourceType::Cbv,
                    View::Unordered(..) | View::Placed(true, ..) => ResourceType::Uav,
                    _ => ResourceType::Srv,
                };
                self.depth_sampled |= readable(list, kind, &view, read_only_depth);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            if heap.samplers() || (table - heap.start) % 32 != 0 {
                continue;
            }
            let Some(host) = heap.host() else {
                continue;
            };
            let mut selected: Vec<u32> = indexed
                .iter()
                .filter(|index| index.table_slot == slot)
                .map(|index| {
                    let constant = self.user_value(
                        compute,
                        index.constant_slot,
                        RootType::Constants,
                        index.constant_dword,
                    );
                    // The shader adds i32 byte offsets and shifts by 5.
                    index.byte_offset.wrapping_add(constant << index.shift) / 32
                })
                .collect();
            selected.sort_unstable();
            selected.dedup();
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8)
                .chain(selected.last().copied())
                .max()
                .unwrap_or(0)
                + 1;
            let last = table.checked_add(u64::from(n - 1) * 32);
            if n > crate::descriptor::host::DYNAMIC
                || last.and_then(|handle| heap.index(handle)).is_none()
            {
                once!(slot, n, "fresh descriptor table exceeds the available span");
                continue;
            }
            // The table and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                }
                (table, n, h.finish())
            };
            if let Some(&first) = self.fresh_tables.get(&key) {
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        self.depth_sampled |= readable(list, u.kind, &view, read_only_depth);
                    }
                }
                for k in selected {
                    if entries.iter().any(|u| u.table_offset_dwords / 8 == k) {
                        continue;
                    }
                    let handle = table + u64::from(k) * 32;
                    if let Some(view) = crate::descriptor::view(handle)
                        && let Some(kind) = view_kind(&view)
                    {
                        self.depth_sampled |= readable(list, kind, &view, read_only_depth);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = self.take_dynamic(host, std::ptr::from_ref(heap) as usize, n) else {
                // Without its fresh table the shader would read the title's
                // slots, which for constant buffer tables hold no constant
                // view: a texture or null descriptor read as a buffer gives
                // a wild address on AMD and the device is removed (#59).
                // The draw or dispatch is dropped instead.
                self.table_failed = true;
                continue;
            };
            self.fresh_tables.insert(key, first);
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    Some(View::Constant(address, size)) => {
                        let Some((gpu, to)) = list.scratch(u64::from(size)) else {
                            host.make_null_kind(index, ResourceType::Cbv, 4);
                            continue;
                        };
                        list.defer_fill(to, address as *const u8, size as usize);
                        host.constant_at(index, gpu, size);
                    }
                    Some(view) => {
                        host.make_at(index, &view);
                        let kind = used.map(|u| u.kind).or_else(|| {
                            selected.binary_search(&k).ok().and_then(|_| view_kind(&view))
                        });
                        if let Some(kind) = kind {
                            self.depth_sampled |= readable(list, kind, &view, read_only_depth);
                        }
                    }
                    None => {
                        if let Some(u) = used {
                            host.make_null_kind(index, u.kind, u.dwords);
                            once!(slot, "a table entry without a view");
                        } else {
                            host.make_null(index);
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        // Listed frames: `frame_op` logs every operation instead.
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = !listed && TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&(resource, _)| {
                let resource = resource as This;
                let texture = crate::resource::host_resource(resource)?;
                // SAFETY: a live resource.
                let d = unsafe { texture.GetDesc() };
                Some(format!("{}x{} f{} {resource:?}", d.Width, d.Height, d.Format.0))
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let rects = self.topology == RECTANGLES;
        let expanded = crate::primitive::ExpandedTopology::from_console(self.topology);
        let ran = self.prepare(list, false) && (!rects || self.rectangles(list, vertices, first));
        if ran {
            if let Some(kind) = expanded {
                if let Some((count, base)) = self.expand_non_indexed(list, kind, vertices, first) {
                    if count > 0 {
                        #[cfg(feature = "diagnostics")]
                        list.note_op(|| {
                            self.op_text(format!("draw {kind:?} {count}x{instances}"), false)
                        });
                        // SAFETY: an open list with expanded scratch index buffer bound.
                        unsafe {
                            list.list
                                .DrawIndexedInstanced(count, instances, 0, base, first_instance)
                        };
                        if crate::pipeline::host_pipeline(self.pipeline as This)
                            .is_some_and(|p| p.has_root_uavs())
                        {
                            emit_uav_barrier(list);
                        }
                        DRAWN.fetch_add(1, Ordering::Relaxed);
                    }
                    return;
                }
            } else {
                // Rectangles were expanded into their own vertex buffers (their
                // count checked to fit by `rectangles`).
                let (count, from) = if rects {
                    (expanded_vertices(vertices).unwrap_or(0), 0)
                } else {
                    (vertices, first)
                };
                #[cfg(feature = "diagnostics")]
                list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
                // SAFETY: an open list with everything set.
                unsafe {
                    list.list
                        .DrawInstanced(count, instances, from, first_instance)
                };
                if crate::pipeline::host_pipeline(self.pipeline as This)
                    .is_some_and(|p| p.has_root_uavs())
                {
                    emit_uav_barrier(list);
                }
                DRAWN.fetch_add(1, Ordering::Relaxed);
                #[cfg(feature = "diagnostics")]
                self.frame_op("draw", false, true);
                return;
            }
        }
        #[cfg(feature = "diagnostics")]
        self.frame_op("draw", false, false);
        DROPPED.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            dropped(self.pipeline);
            self.note_dropped_writes();
        }
    }

    /// Binds a synthesized scratch index buffer expanding a non-indexed
    /// console-only topology draw into a host triangle or line list.
    fn expand_non_indexed(
        &self,
        list: &HostList,
        kind: crate::primitive::ExpandedTopology,
        vertices: u32,
        first: u32,
    ) -> Option<(u32, i32)> {
        let out_count = kind.max_output_indices(vertices);
        if out_count == 0 {
            return Some((0, 0));
        }
        let strip_cut = crate::pipeline::host_pipeline(self.pipeline as This)
            .map_or(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED, |p| p.strip_cut);
        let use_u16 = strip_cut == D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF;
        if use_u16 {
            if vertices > u32::from(u16::MAX) {
                once!(
                    vertices,
                    "non-indexed console topology exceeds 16-bit strip-cut index range"
                );
                return None;
            }
            let base = i32::try_from(first).ok()?;
            let bytes = u64::from(out_count).checked_mul(2)?;
            let size_bytes = u32::try_from(bytes).ok()?;
            let (gpu, to) = list.scratch(bytes)?;
            // SAFETY: `to` points to `out_count` writable `u16` elements in scratch upload memory.
            let dst = unsafe { std::slice::from_raw_parts_mut(to.cast::<u16>(), out_count as usize) };
            kind.expand_non_indexed_u16(vertices, 0, dst);
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: size_bytes,
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16_UINT,
            };
            // SAFETY: an open list; the view names live scratch upload memory.
            unsafe { list.list.IASetIndexBuffer(Some(&view)) };
            return Some((out_count, base));
        }
        let (start, base) = match i32::try_from(first) {
            Ok(base) => (0u32, base),
            Err(_) => {
                first.checked_add(vertices)?;
                (first, 0)
            }
        };
        let bytes = u64::from(out_count).checked_mul(4)?;
        let size_bytes = u32::try_from(bytes).ok()?;
        let (gpu, to) = list.scratch(bytes)?;
        // SAFETY: `to` points to `out_count` writable `u32` elements in scratch upload memory.
        let dst = unsafe { std::slice::from_raw_parts_mut(to.cast::<u32>(), out_count as usize) };
        kind.expand_non_indexed_u32(vertices, start, dst);
        let view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: gpu,
            SizeInBytes: size_bytes,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
        };
        // SAFETY: an open list; the view names live scratch upload memory.
        unsafe { list.list.IASetIndexBuffer(Some(&view)) };
        Some((out_count, base))
    }

    /// Binds a submission-filled scratch index buffer expanding an indexed
    /// console-only topology draw into a host triangle or line list, preserving
    /// index format (`R16_UINT`/`R32_UINT`) and strip-cut semantics.
    fn expand_indexed(
        &self,
        list: &HostList,
        kind: crate::primitive::ExpandedTopology,
        indices: u32,
        first: u32,
        base: i32,
    ) -> Option<u32> {
        const R32_UINT: u32 = 42;
        const R16_UINT: u32 = 57;
        let v = self.index;
        let is_u16 = match v.format {
            R16_UINT => true,
            R32_UINT => false,
            _ => {
                once!(format = v.format, "unsupported index format for console topology");
                return None;
            }
        };
        if v.address == 0 || v.size == 0 {
            return None;
        }
        let elem_bytes: u64 = if is_u16 { 2 } else { 4 };
        let avail = (u64::from(v.size) / elem_bytes).saturating_sub(u64::from(first));
        let count = u64::from(indices).min(avail) as u32;
        let max_out = kind.max_output_indices(count);
        if max_out == 0 {
            return Some(0);
        }
        let src_addr = v.address.checked_add(u64::from(first).checked_mul(elem_bytes)?)?;
        let src_bytes = usize::try_from(u64::from(count).checked_mul(elem_bytes)?).ok()?;
        let out_bytes = u64::from(max_out).checked_mul(elem_bytes)?;
        let out_bytes_u32 = u32::try_from(out_bytes).ok()?;
        let (gpu, to) = list.scratch(out_bytes)?;
        let strip_cut = crate::pipeline::host_pipeline(self.pipeline as This)
            .map_or(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED, |p| p.strip_cut);
        let to_addr = to as usize;
        let fallback_pad = (-i64::from(base)).max(0) as u32;
        if is_u16 {
            let cut = (strip_cut == D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF).then_some(u16::MAX);
            let pad = fallback_pad.min(u32::from(u16::MAX)) as u16;
            list.defer_host(move || {
                let mut src = vec![0u16; count as usize];
                crate::host::copy_mapped(
                    src.as_mut_ptr().cast::<u8>(),
                    src_addr as *const u8,
                    src_bytes,
                    false,
                );
                // SAFETY: `to_addr` points to `max_out` writable `u16` slots owned by this list.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(to_addr as *mut u16, max_out as usize)
                };
                kind.expand_indexed_u16(&src, cut, pad, dst);
            });
        } else {
            let cut =
                (strip_cut == D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF).then_some(u32::MAX);
            list.defer_host(move || {
                let mut src = vec![0u32; count as usize];
                crate::host::copy_mapped(
                    src.as_mut_ptr().cast::<u8>(),
                    src_addr as *const u8,
                    src_bytes,
                    false,
                );
                // SAFETY: `to_addr` points to `max_out` writable `u32` slots owned by this list.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(to_addr as *mut u32, max_out as usize)
                };
                kind.expand_indexed_u32(&src, cut, fallback_pad, dst);
            });
        }
        let view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: gpu,
            SizeInBytes: out_bytes_u32,
            Format: DXGI_FORMAT(v.format as i32),
        };
        // SAFETY: an open list; the view names live scratch upload memory.
        unsafe { list.list.IASetIndexBuffer(Some(&view)) };
        Some(max_out)
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        // The draw runs `count / 3 * 6` vertices, a count the host takes in
        // 32 bits.
        if expanded_vertices(count).is_none() {
            once!(count, "too many rectangle vertices");
            return false;
        }
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let source = rect_source(first, count, v.stride, v.size)
                .filter(|_| v.address != 0)
                .and_then(|offset| v.address.checked_add(offset))
                .filter(|&from| crate::shadow::find(from, rects * 3 * stride).is_some());
            let Some(from) = source else {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            };
            // The expanded vertices' view takes its size in 32 bits.
            let Ok(span) = u32::try_from(rects * 6 * stride) else {
                once!(slot, "rectangle vertices too many to expand");
                return false;
            };
            let Some((gpu, to)) = list.scratch(u64::from(span)) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: span,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        if self.topology == RECTANGLES {
            once!("indexed rectangles not drawn");
        }
        let expanded = crate::primitive::ExpandedTopology::from_console(self.topology);
        let ran = self.topology != RECTANGLES && self.prepare(list, expanded.is_none());
        if ran {
            if let Some(kind) = expanded {
                if let Some(count) = self.expand_indexed(list, kind, indices, first, base) {
                    if count > 0 {
                        #[cfg(feature = "diagnostics")]
                        list.note_op(|| {
                            self.op_text(format!("indexed {kind:?} {count}x{instances}"), false)
                        });
                        // SAFETY: an open list with expanded scratch index buffer bound.
                        unsafe {
                            list.list
                                .DrawIndexedInstanced(count, instances, 0, base, first_instance)
                        };
                        if crate::pipeline::host_pipeline(self.pipeline as This)
                            .is_some_and(|p| p.has_root_uavs())
                        {
                            emit_uav_barrier(list);
                        }
                        DRAWN.fetch_add(1, Ordering::Relaxed);
                    }
                    return;
                }
            } else {
                #[cfg(feature = "diagnostics")]
                list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
                // SAFETY: an open list with everything set.
                unsafe {
                    list.list
                        .DrawIndexedInstanced(indices, instances, first, base, first_instance)
                };
                if crate::pipeline::host_pipeline(self.pipeline as This)
                    .is_some_and(|p| p.has_root_uavs())
                {
                    emit_uav_barrier(list);
                }
                DRAWN.fetch_add(1, Ordering::Relaxed);
                #[cfg(feature = "diagnostics")]
                self.frame_op("draw_indexed", false, true);
                return;
            }
        }
        #[cfg(feature = "diagnostics")]
        self.frame_op("draw_indexed", false, false);
        DROPPED.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            dropped(self.pipeline);
            self.note_dropped_writes();
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            self.note_dropped_writes();
            self.frame_op(&format!("indirect kind {kind}"), kind == 2, false);
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        };
        let matches_kind = match kind {
            DRAW | INDEXED => self.topology != RECTANGLES && !p.compute && !p.mesh,
            MESH => !p.compute && p.mesh,
            DISPATCH => p.compute,
            _ => false,
        };
        if !matches_kind {
            why(&format!(
                "pipeline mismatch (compute {}, mesh {}, topology {})",
                p.compute, p.mesh, self.topology
            ));
            return false;
        }
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find(arguments, len) else {
            why("arguments outside title buffers");
            #[cfg(feature = "diagnostics")]
            self.frame_op_why(
                &format!("indirect kind {kind}"),
                kind == DISPATCH,
                false,
                Some("arguments outside title buffers"),
            );
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    #[cfg(feature = "diagnostics")]
                    self.frame_op_why(
                        &format!("indirect kind {kind}"),
                        kind == DISPATCH,
                        false,
                        Some("count outside title buffers"),
                    );
                    return false;
                }
            },
            None => None,
        };
        if kind == INDEXED && self.topology == RECTANGLES {
            once!("indirect indexed rectangles not drawn");
            why("indexed rectangles not supported");
            return false;
        }
        if kind == DRAW && self.topology == RECTANGLES {
            let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
                #[cfg(feature = "diagnostics")]
                crate::pipeline::note_drop(self.pipeline as This);
                return false;
            };
            if p.compute || p.mesh {
                why(&format!(
                    "pipeline mismatch (compute {}, mesh {}, topology {})",
                    p.compute, p.mesh, self.topology
                ));
                return false;
            }
            let Some(signature) = host_signature(DRAW, 16) else {
                why("no host signature");
                return false;
            };
            if max == 0 {
                DRAWN.fetch_add(1, Ordering::Relaxed);
                return true;
            }
            let prepared = match crate::rectangles::expand(
                list,
                p,
                &self.vertex,
                (&args, arguments, stride, max),
                counter.as_ref().map(|(s, at)| (s, *at)),
            ) {
                Ok(p) => p,
                Err(reason) => {
                    why(reason);
                    return false;
                }
            };
            self.host_pipeline = 0;
            self.host_roots[1] = 0;
            if !self.prepare(list, false) {
                why("prepare failed for indirect rectangles");
                return false;
            }
            prepared.before_draw(list);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| {
                self.op_text(
                    format!("indirect rect draw max {max} at {arguments:#x}"),
                    false,
                )
            });
            // SAFETY: an open list with everything set; `prepared.args_buf` is
            // list-owned, `counter` is kept alive by refresh, signature cached.
            unsafe {
                list.list.ExecuteIndirect(
                    &signature,
                    max,
                    &prepared.args_buf,
                    0,
                    counter.as_ref().map(|(s, _)| &s.buffer),
                    counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
                )
            };
            prepared.after_draw(list);
            if p.has_root_uavs() {
                emit_uav_barrier(list);
            }
            DRAWN.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            why("no host signature");
            #[cfg(feature = "diagnostics")]
            self.frame_op_why(
                &format!("indirect kind {kind}"),
                kind == DISPATCH,
                false,
                Some("no host signature"),
            );
            return false;
        };
        args.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        // Bounded first: the clamp binds a compute pipeline of its own and
        // copies clamped active records into a scratch buffer at offset 0.
        let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
        let Some(bounded) = crate::indirect::bound(
            list,
            (&args.buffer, args.offset(arguments)),
            count_arg,
            (kind, stride, max),
        ) else {
            why("not bounded (no clamp, or arguments not 4-byte aligned)");
            #[cfg(feature = "diagnostics")]
            self.frame_op_why(
                &format!("indirect kind {kind}"),
                kind == DISPATCH,
                false,
                Some("not bounded (no clamp, or arguments not 4-byte aligned)"),
            );
            return false;
        };
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
        let ran = match kind {
            DRAW | INDEXED => self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => self.bind(list, p, true),
            _ => false,
        };
        #[cfg(feature = "diagnostics")]
        self.frame_op("indirect", kind == DISPATCH, ran);
        if !ran {
            crate::indirect::done(list, &bounded);
            #[cfg(feature = "diagnostics")]
            match crate::pipeline::host_pipeline(self.pipeline as This) {
                None => {
                    crate::pipeline::note_drop(self.pipeline as This);
                    self.note_dropped_writes();
                }
                Some(p) => why(&format!(
                    "prepare failed (compute {}, mesh {})",
                    p.compute, p.mesh
                )),
            }
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(
                format!("indirect kind {kind} max {max} at {arguments:#x}"),
                kind == DISPATCH,
            )
        });
        if let Some((s, _)) = &counter {
            list.buffer_state(&s.buffer, crate::host::BUFFER_READ);
        }
        // SAFETY: an open list with everything set; the scratch buffer and
        // host copies are kept alive by the list, and the signature is cached.
        unsafe {
            list.list.ExecuteIndirect(
                &signature,
                max,
                &bounded.scratch,
                0,
                counter.as_ref().map(|(s, _)| &s.buffer),
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        crate::indirect::done(list, &bounded);
        if crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.has_root_uavs())
        {
            emit_uav_barrier(list);
        }
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// The tracker uses whole resources, so both planes must be read-only
/// when the depth format carries stencil.
fn depth_state(texture: &ID3D12Resource, desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>) -> u32 {
    let Some(desc) = desc else {
        return DEPTH_WRITE;
    };
    let format = if desc.Format == DXGI_FORMAT_UNKNOWN {
        // SAFETY: a live texture; a default format inherits its format.
        unsafe { texture.GetDesc() }.Format
    } else {
        desc.Format
    };
    let mut required = D3D12_DSV_FLAG_READ_ONLY_DEPTH.0;
    if format != DXGI_FORMAT_D16_UNORM && format != DXGI_FORMAT_D32_FLOAT {
        required |= D3D12_DSV_FLAG_READ_ONLY_STENCIL.0;
    }
    if desc.Flags.0 & required == required {
        DEPTH_READ
    } else {
        DEPTH_WRITE
    }
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need. Returns whether the view
/// also reads the draw's bound depth target.
fn view_kind(view: &View) -> Option<ResourceType> {
    match view {
        View::Shader(..) | View::Placed(false, ..) => Some(ResourceType::Srv),
        View::Unordered(..) | View::Placed(true, ..) => Some(ResourceType::Uav),
        _ => None,
    }
}

fn readable(
    list: &HostList,
    kind: ResourceType,
    view: &View,
    read_only_depth: Option<usize>,
) -> bool {
    #[cfg(feature = "diagnostics")]
    if kind == ResourceType::Uav {
        crate::producers::live_write(view);
    } else {
        crate::producers::read(view);
    }
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((address, len)) = crate::descriptor::host::buffer_range(view)
        && len > 0
        && let Some(shadow) = crate::shadow::find(address, len)
    {
        shadow.refresh(list, address, len);
        if kind == ResourceType::Uav {
            shadow.gpu_writes(list, address, len);
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_WRITE);
        } else {
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
        }
    }
    match (view, kind) {
        (View::Shader(resource, desc), ResourceType::Srv) => {
            let Some(t) = crate::resource::host_resource(*resource as This) else {
                return false;
            };
            if crate::texture::title_written(t) {
                crate::texture::upload(list, *resource as This, t);
            }
            // Keep the same read bits the depth target will request at draw time.
            let samples_depth = read_only_depth == Some(*resource);
            let state = if samples_depth {
                DEPTH_READ | SHADER_RESOURCE
            } else {
                SHADER_RESOURCE
            };
            let mips = (!samples_depth)
                .then(|| desc.as_ref().and_then(|d| crate::descriptor::host::srv_mips(t, d)))
                .flatten();
            crate::list::ensure_mips(list, *resource as This, state, mips);
            #[cfg(feature = "diagnostics")]
            {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
            samples_depth
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            false
        }
        (&View::Constant(address, size), ResourceType::Cbv) if size > 0 => {
            if let Some(shadow) = crate::shadow::find(address, u64::from(size)) {
                shadow.refresh(list, address, u64::from(size));
                list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
            }
            false
        }
        _ => false,
    }
}

#[cfg(test)]
mod depth_tests {
    use super::*;

    #[test]
    fn shader_table_keeps_read_only_depth_state() {
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_SAMPLE_DESC,
        };

        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            ..Default::default()
        };
        let resource_iid = ID3D12Resource::IID;
        let mut depth = null;
        assert_eq!(
            crate::resource::create_committed(
                null,
                (&raw const heap).cast(),
                0,
                (&raw const desc).cast(),
                D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32,
                null.cast(),
                (&raw const resource_iid).cast(),
                &mut depth,
            ),
            crate::object::S_OK
        );
        #[repr(C)]
        struct ViewHeap {
            desc: D3D12_DESCRIPTOR_HEAP_DESC,
            placement: [u64; 2],
        }
        let desc = ViewHeap {
            desc: D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            },
            placement: [0; 2],
        };
        let heap_iid = ID3D12DescriptorHeap::IID;
        let mut views = null;
        assert_eq!(
            crate::descriptor::create_heap(
                null,
                (&raw const desc).cast(),
                (&raw const heap_iid).cast(),
                &mut views,
            ),
            crate::object::S_OK
        );
        let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        let srv = crate::descriptor::heap(views).start;
        crate::descriptor::create_srv(null, depth, (&raw const srv_desc).cast(), srv);
        let used = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };

        let shader_resource = (D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
            | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
            .0 as u32;
        let combined = D3D12_RESOURCE_STATE_DEPTH_READ.0 as u32 | shader_resource;
        for (flags, unsampled_state, shader_state, draw_state) in [
            (
                D3D12_DSV_FLAG_READ_ONLY_DEPTH,
                DEPTH_READ,
                combined,
                combined,
            ),
            (
                D3D12_DSV_FLAG_NONE,
                DEPTH_WRITE,
                shader_resource,
                DEPTH_WRITE,
            ),
        ] {
            let list = host.new_list().expect("host list");
            let desc = [
                DXGI_FORMAT_D32_FLOAT.0 as u32,
                D3D12_DSV_DIMENSION_TEXTURE2D.0 as u32,
                flags.0 as u32,
                0,
                0,
                0,
            ];
            let mut slot = [0u64; 16];
            let dsv = slot.as_mut_ptr() as u64;
            crate::descriptor::create_dsv(null, depth, &desc, dsv);
            let mut state = State::default();
            state.set_targets(&list, &[], dsv);
            state.set_heaps(&[views]);
            state.set_table(false, 0, srv);
            let no_transition = |wanted| {
                let mut changed = false;
                list.transition(depth, wanted, None, |_, _, _, _| changed = true);
                !changed
            };
            crate::list::ensure_state(&list, depth, D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32);
            state.ensure_depth_state(&list);
            assert!(
                no_transition(unsampled_state),
                "unsampled depth state for {flags:?}"
            );
            state.tables(&list, &[used], &[], false);
            assert!(
                no_transition(shader_state),
                "shader-table state for {flags:?}"
            );
            state.ensure_depth_state(&list);
            assert!(
                no_transition(draw_state),
                "depth state for {flags:?}"
            );
            list.close();
            host.execute(&[&list]);
        }
        let release = |object: This| {
            crate::object::slot::<extern "system" fn(This) -> u32>(object, 0x10)(object)
        };
        assert_eq!(release(views), 0);
        assert_eq!(release(depth), 0);
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_resource(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!(
                "{}x{} f{} flags {:#x}",
                d.Width, d.Height, d.Format.0, d.Flags.0
            )
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        // A buffer has no host texture: its title range, as for a shader view.
        Some(v @ View::Unordered(r, ..)) => texture(*r as This).map_or_else(
            || format!("uav buffer {:?}", crate::descriptor::host::buffer_range(v)),
            |t| format!("uav {t}"),
        ),
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// Console rectangle list topology (expanded on vertex inputs in [`crate::primitive`]).
const RECTANGLES: u32 = 17;

/// The host vertices `vertices` rectangle-list vertices become (six per
/// rectangle), if the count fits in 32 bits.
fn expanded_vertices(vertices: u32) -> Option<u32> {
    (vertices / 3).checked_mul(6)
}

/// The byte offset of vertex `first` in a buffer of `size` bytes holding
/// `stride`-byte vertices, if vertices `first..first + count` all lie in
/// it. Computed wide: `first + count` and the byte offsets overflow 32
/// bits with the values a title may pass.
fn rect_source(first: u32, count: u32, stride: u32, size: u32) -> Option<u64> {
    let stride = u64::from(stride);
    let end = (u64::from(first) + u64::from(count)).checked_mul(stride)?;
    (stride != 0 && end <= u64::from(size)).then(|| u64::from(first) * stride)
}

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

#[cfg(all(test, feature = "diagnostics"))]
mod rtv_slot_tests {
    use super::*;

    #[test]
    fn typed_null_rtv_keeps_the_next_color_slot() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
        };
        use windows::core::s;

        struct Texture(This);

        impl Drop for Texture {
            fn drop(&mut self) {
                let release = crate::object::slot::<extern "system" fn(This) -> u32>(self.0, 0x10);
                release(self.0);
            }
        }

        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let device = &host.device;
        let list = host.new_list().expect("host command list");
        let l = &list.list;
        let format = DXGI_FORMAT_R8G8B8A8_UNORM;
        let source = r#"
            [RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)")]
            float4 vs(uint id : SV_VertexID) : SV_Position {
                return float4(id == 2 ? 3 : -1, id == 1 ? -3 : 1, 0, 1);
            }
            struct Colors { float4 red : SV_Target0; float4 green : SV_Target1; };
            Colors ps() {
                Colors c;
                c.red = float4(1, 0, 0, 1);
                c.green = float4(0, 1, 0, 1);
                return c;
            }
        "#;
        let compile = |entry, target| {
            let mut code = None;
            // SAFETY: source and shader names are live, and code is our output pointer.
            unsafe {
                D3DCompile(
                    source.as_ptr().cast(),
                    source.len(),
                    s!("rtv-slot-test"),
                    None,
                    None,
                    entry,
                    target,
                    0,
                    0,
                    &mut code,
                    None,
                )
                .expect("compile fixture shader");
            }
            code.expect("fixture bytecode")
        };
        let vs = compile(s!("vs"), s!("vs_5_1"));
        let ps = compile(s!("ps"), s!("ps_5_1"));
        // SAFETY: compiled fixture shaders and their embedded root signature are live.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let bytecode = |blob: &ID3DBlob| D3D12_SHADER_BYTECODE {
                pShaderBytecode: blob.GetBufferPointer(),
                BytecodeLength: blob.GetBufferSize(),
            };
            let code = bytecode(&vs);
            let root = device
                .CreateRootSignature(
                    0,
                    std::slice::from_raw_parts(code.pShaderBytecode.cast(), code.BytecodeLength),
                )
                .expect("graphics root signature");
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code,
                PS: bytecode(&ps),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 2,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            // D3D12 requires UNKNOWN in the PSO slot that binds a null RTV.
            desc.RTVFormats[1] = format;
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask =
                D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            let pso = device
                .CreateGraphicsPipelineState(&desc)
                .expect("graphics pipeline");
            (root, pso)
        };

        const SIDE: u32 = 4;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(SIDE),
            Height: SIDE,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let mut target = std::ptr::null();
        let null = std::ptr::null();
        let iid = ID3D12Resource::IID;
        assert_eq!(
            crate::resource::create_committed(
                null,
                (&raw const heap).cast(),
                0,
                (&raw const desc).cast(),
                D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32,
                null.cast(),
                (&raw const iid).cast(),
                &mut target,
            ),
            crate::object::S_OK
        );
        let _target = Texture(target);
        let color = crate::resource::host_resource(target).expect("color target");
        let mut slots = [[0u64; 16]; 2];
        let handles = slots.each_mut().map(|slot| slot.as_mut_ptr() as u64);
        let null_desc = [
            format.0 as u32,
            D3D12_RTV_DIMENSION_TEXTURE2D.0 as u32,
            0,
            0,
            0,
            0,
        ];
        crate::descriptor::create_rtv(null, null, &null_desc, handles[0]);
        crate::descriptor::create_rtv(null, target, null.cast(), handles[1]);
        crate::list::ensure_state(&list, target, D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32);
        let clear_rtv = crate::descriptor::rtv_record(handles[1])
            .expect("rtv record")
            .desc();
        // SAFETY: open list, live target and graphics pipeline.
        unsafe {
            l.ClearRenderTargetView(
                host.rtv(Some(color), clear_rtv.as_ref()),
                &[0.0, 0.0, 1.0, 1.0],
                None,
            );
            l.SetGraphicsRootSignature(&root);
            l.SetPipelineState(&pso);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        }
        viewports(
            &list,
            &[D3D12_VIEWPORT {
                Width: SIDE as f32,
                Height: SIDE as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }],
        );
        scissors(
            &list,
            &[RECT {
                right: SIDE as i32,
                bottom: SIDE as i32,
                ..Default::default()
            }],
        );
        State::default().set_targets(&list, &handles, 0);
        // SAFETY: complete graphics state on the open list.
        unsafe { l.DrawInstanced(3, 1, 0, 0) };

        crate::list::ensure_state(&list, target, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
        let (readback, footprint) =
            crate::capture::read_back(device, l, color).expect("color readback");
        list.close();
        host.execute(&[&list]);
        let mut mapped = std::ptr::null_mut();
        let mut pixels = Vec::new();
        // SAFETY: execution has completed and all pixels lie in the mapped footprint.
        unsafe {
            readback
                .Map(0, None, Some(&mut mapped))
                .expect("map readback");
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let at = footprint.Offset as usize
                        + (y * footprint.Footprint.RowPitch + x * 4) as usize;
                    pixels.push(mapped.cast::<u8>().add(at).cast::<[u8; 4]>().read());
                }
            }
            readback.Unmap(0, None);
        }
        assert_eq!(pixels, [[0, 255, 0, 255]; (SIDE * SIDE) as usize]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_shader::native::{ComputeIo, StageIo};
    use victory_shader::{
        Container, HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource,
    };
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
        VirtualFree,
    };

    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    struct Allocation {
        ptr: *mut u8,
    }

    impl Allocation {
        fn watched(len: usize) -> Self {
            // SAFETY: allocates a fresh write-watched virtual range freed in Drop.
            let ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    len,
                    MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            }
            .cast::<u8>();
            assert!(!ptr.is_null());
            Self { ptr }
        }

        fn reserved_unmapped(len: usize) -> Self {
            // SAFETY: reserves an uncommitted VA range so accesses are unmapped.
            let ptr = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE, PAGE_NOACCESS) }
                .cast::<u8>();
            assert!(!ptr.is_null());
            Self { ptr }
        }

        fn addr(&self) -> u64 {
            self.ptr as u64
        }
    }

    impl Drop for Allocation {
        fn drop(&mut self) {
            // SAFETY: `self.ptr` came from `VirtualAlloc`.
            unsafe {
                VirtualFree(self.ptr.cast(), 0, MEM_RELEASE);
            }
        }
    }

    /// Synthetic GFX10 compute shader reading Root CBV (slot 0, dword 0) and
    /// Root SRV (slot 1, byte offset `+1.5 MiB`), and storing `cbv ^ srv ^
    /// 0xA5A5_5A5A` to Root UAV (slot 2, byte offset `+1.5 MiB`).
    fn build_root_view_cs() -> (victory_shader::emit::HostShader, Vec<UsedResource>) {
        let s_mov_b32 =
            |sdst: u8, src0: u8| 0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32;
        let s_load_dword = |dst: u8, base: u8, byte_off: u32| -> [u32; 2] {
            [
                0x3D << 26 | (dst as u32) << 6 | (base as u32 >> 1),
                125 << 25 | (byte_off & 0x1F_FFFF),
            ]
        };
        let vop1 = |op: u16, vdst: u8, src0: u16| -> u32 {
            0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
        };
        let buffer_load_dword_offen = |vdst: u8, vaddr: u8, srsrc: u8| -> [u32; 2] {
            [
                0x38 << 26 | 0x0C << 18 | (1 << 12),
                0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdst as u32) << 8 | vaddr as u32,
            ]
        };
        let buffer_store_dword_offen = |vdata: u8, vaddr: u8, srsrc: u8| -> [u32; 2] {
            [
                0x38 << 26 | 0x1C << 18 | (1 << 12),
                0x80 << 24
                    | ((srsrc as u32 >> 2) & 0x1F) << 16
                    | (vdata as u32) << 8
                    | vaddr as u32,
            ]
        };

        let mut words = Vec::new();
        // s[0:1] = CBV 0, s[2:3] = SRV 1, s[4:5] = UAV 2.
        words.extend_from_slice(&s_load_dword(6, 0, 0));
        // SRV descriptor in s[8:11], UAV descriptor in s[12:15].
        words.push(s_mov_b32(8, 2));
        words.push(s_mov_b32(9, 3));
        words.push(s_mov_b32(10, 0xC1));
        words.extend_from_slice(&[s_mov_b32(11, 0xFF), 0x3101_4FAC]);
        words.push(s_mov_b32(12, 4));
        words.push(s_mov_b32(13, 5));
        words.push(s_mov_b32(14, 0xC1));
        words.extend_from_slice(&[s_mov_b32(15, 0xFF), 0x3101_4FAC]);
        let vop2 = |op: u8, vdst: u8, src0: u16, vsrc1: u8| -> u32 {
            (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
        };
        // v0 = 0x18_0000 (+1.5 MiB); v1 = srv[0x18_0000]; v2 = s6 ^ v1 ^ 0xA5A5_5A5A.
        words.extend_from_slice(&[vop1(0x001, 0, 0xFF), 0x0018_0000]);
        words.extend_from_slice(&buffer_load_dword_offen(1, 0, 8));
        words.push(vop2(0x1D, 2, 6, 1)); // v_xor_b32 v2, s6, v1
        words.extend_from_slice(&[vop2(0x1D, 2, 0xFF, 2), 0xA5A5_5A5A]); // v_xor_b32 v2, literal, v2
        words.extend_from_slice(&buffer_store_dword_offen(2, 0, 12));
        words.push(0xBF81_0000); // s_endpgm

        let used = vec![
            UsedResource {
                kind: ResourceType::Cbv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 2,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 4,
                slot: 2,
            },
        ];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![
                RootParam {
                    root: RootType::Descriptor,
                    table_offset_dwords: 0,
                    sgprs: 2,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                RootParam {
                    root: RootType::Descriptor,
                    table_offset_dwords: 0,
                    sgprs: 2,
                    base_register: 1,
                    user_data: 2,
                    slot: 1,
                },
                RootParam {
                    root: RootType::Descriptor,
                    table_offset_dwords: 0,
                    sgprs: 2,
                    base_register: 2,
                    user_data: 4,
                    slot: 2,
                },
            ],
            used: used.clone(),
            io: StageIo::Cs(ComputeIo {
                rsrc2: 6 << 1,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = victory_shader::ir::build(&native.code, false, 8).unwrap();
        let res = victory_shader::resource::track(&p, &native.layout, native.stage, None).unwrap();
        let io = victory_shader::io::raise(&p, &native).unwrap();
        let st = victory_shader::structure::structure(&p).unwrap();
        let c = Container {
            digest: [0; 16],
            version: (1, 0),
            parts: Vec::new(),
        };
        let shader = victory_shader::emit::emit(
            &c,
            &native,
            &p,
            &st,
            &io,
            &res,
            victory_shader::emit::Target::ANY,
        )
        .unwrap();
        (shader, used)
    }

    #[test]
    fn null_and_unmapped_root_views_use_safe_distinct_fallbacks() {
        assert!(crate::shadow::find(0, 1).is_none());
        assert!(crate::shadow::find(0x1000, 64).is_none());
        assert!(crate::shadow::find(u64::MAX - 16, 1).is_none());
        assert!(crate::shadow::find(u64::MAX - (1 << 20), 1).is_none());

        let unmapped = Allocation::reserved_unmapped(8 << 20);
        let mid = unmapped.addr() + (4 << 20);
        assert!(crate::shadow::find(mid, 64).is_none());

        let Some(host) = crate::host::host() else {
            return;
        };
        let list = host.new_list().expect("new_list");
        assert!(root_view(&list, 0, RootView::Cbv).is_none());
        assert!(root_view(&list, 0, RootView::Srv).is_none());
        assert!(root_view(&list, 0, RootView::Uav).is_none());
        assert!(root_view(&list, mid, RootView::Srv).is_none());
        assert!(root_view(&list, mid, RootView::Uav).is_none());
        assert!(root_view(&list, u64::MAX - 16, RootView::Srv).is_none());
        assert!(root_view(&list, u64::MAX - 16, RootView::Uav).is_none());

        let zero_va = fallback_root_va(false).expect("zero fallback VA");
        let sink_va = fallback_root_va(true).expect("sink fallback VA");
        assert_ne!(zero_va, 0);
        assert_ne!(sink_va, 0);
        assert_ne!(zero_va, sink_va);
        assert!(zero_va.is_multiple_of(256));
        assert!(sink_va.is_multiple_of(256));
        list.close();
    }

    #[test]
    fn root_view_ranges_are_capped() {
        let big = 1 << 30;
        assert_eq!(root_len(big, RootView::Cbv), 64 << 10);
        assert_eq!(root_len(big, RootView::Srv), ROOT_SRV_MAX);
        assert_eq!(root_len(big, RootView::Uav), ROOT_UAV_MAX);
        // A shorter remaining extent is taken whole.
        for kind in [RootView::Cbv, RootView::Srv, RootView::Uav] {
            assert_eq!(root_len(4096, kind), 4096);
        }
    }

    #[test]
    fn root_srv_refreshes_dirty_data_past_one_mib() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let len = 2 << 20;
        let alloc = Allocation::watched(len);
        let base = alloc.addr();
        let resource_id = alloc.ptr as usize;
        crate::shadow::register(resource_id, base, len as u64);

        let off_a = 0x18_0000usize; // +1.5 MiB
        let off_b = 0x1C_0000usize; // +1.75 MiB
        // SAFETY: `off_a` and `off_b` lie within the 2 MiB committed allocation.
        unsafe {
            alloc.ptr.add(off_a).cast::<u32>().write(0xCAFE_1234);
        }

        let list = host.new_list().expect("new_list");
        let shadow = crate::shadow::find(base, 4).expect("shadow");
        assert_eq!(
            root_view(&list, base, RootView::Srv),
            Some(shadow.gpu(base))
        );
        list.close();
        host.execute(&[&list]);
        assert_eq!(shadow.read_u32(off_a), 0xCAFE_1234);

        // Second submission mutates another page past +1 MiB and verifies
        // incremental dirty-page refresh updates it too.
        list.reset();
        // SAFETY: within the 2 MiB allocation.
        unsafe {
            alloc.ptr.add(off_b).cast::<u32>().write(0xBEEF_5678);
        }
        assert_eq!(
            root_view(&list, base, RootView::Srv),
            Some(shadow.gpu(base))
        );
        list.close();
        host.execute(&[&list]);
        assert_eq!(shadow.read_u32(off_a), 0xCAFE_1234);
        assert_eq!(shadow.read_u32(off_b), 0xBEEF_5678);

        crate::shadow::unregister(resource_id, base);
    }

    #[test]
    fn dispatch_binds_fallbacks_and_copies_root_uav_writes_back() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let (shader, used) = build_root_view_cs();
        let pso = crate::pipeline::test_compute_pipeline(shader, used);

        let unmapped = Allocation::reserved_unmapped(8 << 20);
        let mid_unmapped = unmapped.addr() + (4 << 20);

        let len = 2 << 20;
        let srv_buf = Allocation::watched(len);
        let uav_buf = Allocation::watched(len);
        let cbv_buf = Allocation::watched(64 << 10);
        crate::shadow::register(srv_buf.ptr as usize, srv_buf.addr(), len as u64);
        crate::shadow::register(uav_buf.ptr as usize, uav_buf.addr(), len as u64);
        crate::shadow::register(cbv_buf.ptr as usize, cbv_buf.addr(), 64 << 10);

        let off = 0x18_0000usize; // +1.5 MiB
        // SAFETY: offsets are inside the committed allocations.
        unsafe {
            cbv_buf.ptr.cast::<u32>().write(0x1357_2468);
            srv_buf.ptr.add(off).cast::<u32>().write(0xDEAD_BEEF);
        }

        // 1) Dispatch with slot 0 unset, slot 1 = null (0x0), slot 2 = unmapped VA:
        //    ZERO fallback serves CBV/SRV as 0, SINK absorbs the UAV store at +1.5 MiB
        //    without faulting or dirtying ZERO.
        let list = host.new_list().expect("new_list");
        let mut state = State::default();
        state.set_pipeline(pso);
        state.set_view(true, 1, 0);
        state.set_view(true, 2, mid_unmapped);
        state.dispatch(&list, 1, 1, 1);
        list.close();
        host.execute(&[&list]);

        // 2) Dispatch with CBV & SRV still unbound (verifying ZERO remained 0!)
        //    and UAV bound to `uav_buf`: uav_buf[+1.5 MiB] must receive
        //    0 ^ 0 ^ 0xA5A5_5A5A = 0xA5A5_5A5A in CPU title memory.
        list.reset();
        state.reset(pso);
        state.set_view(true, 2, uav_buf.addr());
        state.dispatch(&list, 1, 1, 1);
        list.close();
        host.execute(&[&list]);
        // SAFETY: reading back the u32 at `+1.5 MiB` in title memory.
        let from_zero = unsafe { uav_buf.ptr.add(off).cast::<u32>().read() };
        assert_eq!(
            from_zero, 0xA5A5_5A5A,
            "sink UAV write must not dirty the zero fallback buffer"
        );

        // 3) Dispatch with all three root views bound (SRV reading +1.5 MiB and
        //    UAV writing +1.5 MiB) and verify full extent refresh + write-back.
        list.reset();
        state.reset(pso);
        state.set_view(true, 0, cbv_buf.addr());
        state.set_view(true, 1, srv_buf.addr());
        state.set_view(true, 2, uav_buf.addr());
        state.dispatch(&list, 1, 1, 1);
        list.close();
        host.execute(&[&list]);
        // SAFETY: reading back the u32 at `+1.5 MiB` in title memory.
        let got = unsafe { uav_buf.ptr.add(off).cast::<u32>().read() };
        assert_eq!(got, 0x1357_2468 ^ 0xDEAD_BEEF ^ 0xA5A5_5A5A);

        // 4) Indirect dispatch (`execute_indirect`) with Root UAVs bound also
        //    emits its UAV barrier cleanly and copies updated writes back.
        let args_buf = Allocation::watched(64 << 10);
        crate::shadow::register(args_buf.ptr as usize, args_buf.addr(), 64 << 10);
        // SAFETY: updating CBV[0] and writing [1, 1, 1] dispatch args inside committed allocations.
        unsafe {
            cbv_buf.ptr.cast::<u32>().write(0x2468_ACE0);
            args_buf.ptr.cast::<[u32; 3]>().write([1, 1, 1]);
        }
        list.reset();
        state.reset(pso);
        state.set_view(true, 0, cbv_buf.addr());
        state.set_view(true, 1, srv_buf.addr());
        state.set_view(true, 2, uav_buf.addr());
        assert!(state.execute_indirect(&list, (2, 12, 1), args_buf.addr(), None));
        list.close();
        host.execute(&[&list]);
        // SAFETY: reading back the u32 at `+1.5 MiB` in title memory.
        let indirect_got = unsafe { uav_buf.ptr.add(off).cast::<u32>().read() };
        assert_eq!(indirect_got, 0x2468_ACE0 ^ 0xDEAD_BEEF ^ 0xA5A5_5A5A);
        crate::shadow::unregister(args_buf.ptr as usize, args_buf.addr());

        crate::shadow::unregister(srv_buf.ptr as usize, srv_buf.addr());
        crate::shadow::unregister(uav_buf.ptr as usize, uav_buf.addr());
        crate::shadow::unregister(cbv_buf.ptr as usize, cbv_buf.addr());
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(pso, 0x10);
        assert_eq!(release(pso), 0);
    }
}

#[cfg(test)]
mod lease_tests {
    use super::*;
    use crate::descriptor::host::HostHeap;

    /// The dynamic slots of an empty heap (and the heap left empty).
    fn capacity(heap: &HostHeap) -> u32 {
        let mut n = 0;
        while heap.dynamic(1).is_some() {
            n += 1;
        }
        heap.release(heap.count(), n);
        n
    }

    /// A heap with no room takes back the slots of a list that already ran,
    /// not of one that has not, and the list that lost them is refused.
    #[test]
    fn a_full_heap_takes_back_the_slots_of_a_list_that_ran() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let key = 0x2000_0001;
        let total = capacity(&heap);
        let chunk = 1000;
        let mut ran = State::default();
        let mut taken = 0;
        while ran.take_dynamic(&heap, key, chunk).is_some() {
            taken += chunk;
        }
        assert!(taken >= total - chunk, "the list filled the heap");
        // Nothing has run yet: nothing is given back and the draw is dropped.
        let mut other = State::default();
        assert_eq!(other.take_dynamic(&heap, key, chunk), None);
        assert!(!ran.lease().is_lost());
        // The list ran: its slots serve the next list, once.
        assert!(ran.lease().start());
        ran.lease().finish();
        assert!(other.take_dynamic(&heap, key, chunk).is_some());
        assert!(ran.lease().is_lost());
        assert!(
            !ran.lease().start(),
            "a list that lost its slots must not run"
        );
        // Everything of the first list is free again except what the other
        // list now holds, and nothing was given back twice.
        assert!(ran.lease().take().is_empty());
        assert_eq!(capacity_left(&heap), total - chunk);
        // Another request finds no finished list left and is dropped.
        let mut third = State::default();
        assert_eq!(third.take_dynamic(&heap, key, total), None);
    }

    /// Free slots of the heap now (taken and released again).
    fn capacity_left(heap: &HostHeap) -> u32 {
        capacity(heap)
    }

    /// A list that is running when another needs room keeps its slots; a
    /// list reset by the title gives back its own and is not taken twice.
    #[test]
    fn a_running_or_reset_list_is_not_taken_back() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let key = 0x2000_0002;
        let total = capacity(&heap);
        let mut running = State::default();
        let first = running.take_dynamic(&heap, key, total).unwrap();
        assert_eq!(first, heap.count());
        running.lease().finish();
        // Submitted again: it is running and cannot be taken back.
        assert!(running.lease().start());
        let mut other = State::default();
        assert_eq!(other.take_dynamic(&heap, key, 1), None);
        assert!(!running.lease().is_lost());
        // It ran: now it can, whole.
        running.lease().finish();
        assert_eq!(other.take_dynamic(&heap, key, 1), Some(heap.count()));
        assert!(running.lease().is_lost());
        // A lease reset by the title (its slots taken, as `Drop` does) is
        // skipped: nothing is held, so nothing is taken back.
        let mut reset = State::default();
        assert!(reset.take_dynamic(&heap, key, 1).is_some());
        reset.lease().finish();
        assert_eq!(reset.lease().take().len(), 1);
        let mut last = State::default();
        assert_eq!(last.take_dynamic(&heap, key, total), None);
        assert!(!reset.lease().is_lost());
    }
}

#[cfg(test)]
mod address_srv_tests {
    use super::*;

    fn half(slot: u8, dword: u32) -> UserData {
        UserData {
            index: 0,
            slot,
            root: RootType::Constants,
            dword,
        }
    }

    /// The low half at dword 1 of root parameter 0, the high half at dword 0
    /// of parameter 2.
    fn view() -> AddressView {
        AddressView {
            lo: half(0, 1),
            hi: half(2, 0),
        }
    }

    fn roots(lo: u32, hi: u32) -> Vec<Root> {
        vec![
            Root::Constants(vec![7, lo]),
            Root::None,
            Root::Constants(vec![hi]),
        ]
    }

    #[test]
    fn addresses_in_root_constants_are_composed_from_both_halves() {
        assert_eq!(
            pair_address(&roots(0x2345_0000, 1), &view()),
            Some(0x1_2345_0000)
        );
        assert_eq!(pair_address(&roots(0x1_0000, 0), &view()), Some(0x1_0000));
    }

    #[test]
    fn an_address_with_a_missing_half_is_unset() {
        let set = roots(0x2345_0000, 1);
        assert_eq!(pair_address(&set[..2], &view()), None);
        let mut no_high = set.clone();
        no_high[2] = Root::Constants(Vec::new());
        assert_eq!(pair_address(&no_high, &view()), None);
        let mut not_constants = set.clone();
        not_constants[0] = Root::View(0x1_0000_0000);
        assert_eq!(pair_address(&not_constants, &view()), None);
        let mut short = set;
        short[0] = Root::Constants(vec![7]);
        assert_eq!(pair_address(&short, &view()), None);
    }

    #[test]
    fn null_and_low_addresses_are_unset() {
        assert_eq!(pair_address(&roots(0, 0), &view()), None);
        assert_eq!(pair_address(&roots(0xFFFF, 0), &view()), None);
    }

    #[test]
    fn rectangle_vertices_inside_their_buffer() {
        assert_eq!(rect_source(0, 3, 16, 48), Some(0));
        assert_eq!(rect_source(3, 3, 16, 96), Some(48));
        assert_eq!(rect_source(3, 3, 16, 95), None);
        assert_eq!(rect_source(0, 3, 0, 48), None);
    }

    /// `first + count` past 32 bits wrapped to a small count before, which
    /// passed the bounds check and read far outside the buffer.
    #[test]
    fn rectangle_bounds_do_not_wrap() {
        assert_eq!(rect_source(u32::MAX, 3, 16, u32::MAX), None);
        assert_eq!(rect_source(u32::MAX - 1, 3, 1, u32::MAX), None);
        assert_eq!(rect_source(u32::MAX, u32::MAX, u32::MAX, u32::MAX), None);
    }

    #[test]
    fn expanded_counts_fit_32_bits() {
        assert_eq!(expanded_vertices(3), Some(6));
        assert_eq!(expanded_vertices(7), Some(12));
        assert_eq!(expanded_vertices(u32::MAX / 6 * 3), Some(u32::MAX / 6 * 6));
        assert_eq!(expanded_vertices(u32::MAX), None);
    }

    #[test]
    fn aligned_cbv_tables_mirror_in_place_without_dynamic_slots() {
        use crate::object::{S_OK, slot};

        let Some(host) = crate::host::host() else {
            return;
        };
        let list = host.new_list().expect("host list");
        let null: This = std::ptr::null();

        // Create a shader-visible view heap with 8 title slots (plain desc + zero placement tail).
        let mut heap_obj: This = null;
        let desc = [0u32, 8, 1, 0, 0, 0, 0, 0]; // CBV_SRV_UAV, count=8, SHADER_VISIBLE, node=0 + zero tail
        let heap_iid = ID3D12DescriptorHeap::IID;
        assert_eq!(
            crate::descriptor::create_heap(
                null,
                desc.as_ptr().cast(),
                std::ptr::from_ref(&heap_iid).cast(),
                &mut heap_obj,
            ),
            S_OK
        );
        let heap_start = slot::<extern "system" fn(This) -> u64>(heap_obj, 0x48)(heap_obj);

        // Create a 4096-byte committed buffer so its shadow buffer is 256-byte aligned.
        let props = [2u32, 0, 0, 0, 0]; // UPLOAD heap
        let buf_desc = [
            1u32, // BUFFER
            0, 0, 0, 4096, 0, // alignment=0, width=4096
            1, 0x0001_0001, 0, 1, 0, 1, 0, 0,
        ];
        let mut buf: This = null;
        let res_iid = ID3D12Resource::IID;
        assert_eq!(
            crate::resource::create_committed(
                null,
                props.as_ptr().cast(),
                0,
                buf_desc.as_ptr().cast(),
                0,
                null.cast(),
                std::ptr::from_ref(&res_iid).cast(),
                &mut buf,
            ),
            S_OK
        );
        let (data, len) = crate::resource::buffer_bytes(buf).expect("buffer bytes");
        assert!(len >= 4096);
        let base = data as u64;

        let write_cbv = |slot_idx: u64, addr: u64, size: u32| {
            let cbv = [addr as u32, (addr >> 32) as u32, size, 0];
            crate::descriptor::create_cbv(null, &cbv, heap_start + slot_idx * 32);
        };
        // Slot 0: 256-byte aligned CBV within shadow bounds -> mirrors in place.
        write_cbv(0, base, 256);
        // Slot 1: 16-byte aligned (misaligned for PC) CBV -> needs dynamic slots.
        write_cbv(1, base + 16, 256);
        // Slot 2: 256-byte aligned start, but padded 256-byte size runs past shadow end -> needs dynamic slots.
        write_cbv(2, base + 4096 - 128, 128);

        let used = [UsedResource {
            kind: ResourceType::Cbv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }];

        let mut state = State::default();
        state.set_heaps(&[heap_obj]);

        // 1) Aligned CBV table: zero dynamic slots allocated, not remapped.
        state.set_table(false, 0, heap_start);
        let made = state.tables(&list, &used, &[], false);
        assert!(!state.table_failed);
        assert!(made.is_empty(), "aligned CBV table should not allocate dynamic slots");

        // 2) Misaligned (+16) CBV table: allocates dynamic slots.
        state.set_table(false, 0, heap_start + 32);
        let made = state.tables(&list, &used, &[], false);
        assert!(!state.table_failed);
        assert_eq!(made.len(), 1, "misaligned CBV table must use dynamic slots");

        // 3) CBV whose 256-byte-padded size exceeds shadow end: allocates dynamic slots.
        state.set_table(false, 0, heap_start + 64);
        let made = state.tables(&list, &used, &[], false);
        assert!(!state.table_failed);
        assert_eq!(made.len(), 1, "CBV padded past shadow end must use dynamic slots");

        list.close();
    }
}
