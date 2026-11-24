//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed, or when a bindless
//! `ResourceDescriptorHeap[]` draw or dispatch materializes and refreshes
//! deferred buffer, constant-buffer, and placed-texture slots.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use victory_shader::emit::{RawWindow, StrideEntry};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, stale_views, view_at};
use crate::host::{Host, host};

/// What a mirrored host descriptor slot (`0..count`) currently holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    /// Default null descriptor (`null_view` or `null_sampler`).
    Null,
    /// Typed null descriptor made for DXIL `ResourceProperties`.
    TypedNull(u32, u32),
    /// Made from a view record `(index, generation)` (`generation >= 1`).
    View(u64),
}

/// A null descriptor description derived from a shader's DXIL
/// `ResourceProperties` annotation.
#[derive(Clone, Copy)]
pub(crate) enum NullDesc {
    Srv(D3D12_SHADER_RESOURCE_VIEW_DESC),
    Uav(D3D12_UNORDERED_ACCESS_VIEW_DESC),
    Cbv,
}

impl std::fmt::Debug for NullDesc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NullDesc::Srv(d) => write!(f, "Srv(dim={}, fmt={})", d.ViewDimension.0, d.Format.0),
            NullDesc::Uav(d) => write!(f, "Uav(dim={}, fmt={})", d.ViewDimension.0, d.Format.0),
            NullDesc::Cbv => write!(f, "Cbv"),
        }
    }
}

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; `spare` dynamic slots follow them, and view
    /// heaps end with one null guard slot ([`HostHeap::guard_slot`]).
    count: u32,
    /// Dynamic slots past the title's: [`DYNAMIC`] (or [`DYNAMIC_CAPPED`] when
    /// capped) in a view heap, [`SAMPLER_DYNAMIC`] in a capped sampler heap,
    /// none in another.
    spare: u32,
    sampler: bool,
    /// What each mirrored slot was last made from.
    made: Mutex<Vec<SlotState>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    kept: Mutex<Vec<Option<ID3D12Resource>>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
    /// Populated buffer, constant-buffer, and placed-texture slots (`< count`)
    /// tracked for bindless `ResourceDescriptorHeap[]` materialization.
    bindless: Mutex<Bindless>,
    /// Per slot, what shaders need to read a buffer whose stride shader
    /// analysis could not recover statically ([`StrideEntry`]).
    strides: Option<StrideTable>,
    /// The first of the [`ALIASES`] slots after the dynamic ones (view
    /// heaps), and which slots have a raw view there.
    alias_base: u32,
    aliases: Mutex<Aliases>,
}

// SAFETY: D3D12 descriptor heaps and resources are free-threaded; mutable
// slot tracking is synchronized behind mutexes and atomics.
unsafe impl Send for HostHeap {}
// SAFETY: as above.
unsafe impl Sync for HostHeap {}

const BOUNCE_CHUNK: u64 = 256 << 10;

/// Persistent 256-byte-aligned bounce slice for a misaligned bindless CBV.
struct BounceSlot {
    gpu: u64,
    cpu: usize,
    cap: u32,
    resource: ID3D12Resource,
    active: bool,
}

#[derive(Default)]
struct Bindless {
    /// Active `View::is_buffer()` slots: `slot_index -> (view_index, generation)`.
    slots: BTreeMap<u32, (u32, u32)>,
    /// Of them, those viewing placed textures.
    placed_set: BTreeMap<u32, (u32, u32)>,
    /// Placed-texture slots changed since draws last prepared them.
    record_pending: BTreeSet<u32>,
    /// Persistent 256-byte-aligned bounce slices by `slot_index`.
    cbv_bounce: HashMap<u32, BounceSlot>,
    /// Backing 256 KiB shared buffers `(resource, cpu_ptr, gpu_va, used_bytes)`.
    bounce_chunks: Vec<(ID3D12Resource, usize, u64, u64)>,
    /// Slots written since the plan was made: only they are made again.
    dirty: BTreeSet<u32>,
    /// What each slot made for bindless access needs at submission.
    uses: BTreeMap<u32, SlotUse>,
    /// The released-view and placed-texture epochs `uses` was checked at.
    checked: (u64, u64),
    /// What submissions do for the slots, until they or what they view change.
    plan: Option<Arc<BindlessPlan>>,
    /// Placed-texture slots as last prepared at recording: view key,
    /// texture.
    recorded: HashMap<u32, (u64, usize)>,
    /// Placed textures reached through slots, by resource object: the host
    /// texture (held, so its address names it alone) and the frame last
    /// uploaded.
    uploaded: HashMap<usize, (ID3D12Resource, u64)>,
    /// Uploaded textures by frame, oldest first, to look at again.
    recheck: VecDeque<(u64, usize)>,
    /// The placed-texture epoch `recorded` and `uploaded` were checked at.
    recorded_epoch: u64,
    /// A frame and the uploads of known textures done in it.
    rechecks: (u64, u32),
    /// When the slot-making budget was last refilled, and the microseconds
    /// left: plans at submission, placed textures at recording.
    budget: Option<(std::time::Instant, [u64; 2])>,
    /// The frame (plus one) released views and evicted textures were last
    /// looked for in: at submission, at recording. Once per frame: the title
    /// releases and evicts all the time, and each look is over all slots.
    looked: [u64; 2],
}

/// Time for making changed slots, at submission and at recording: an eighth
/// of the time passing (4 ms of a 30 fps frame), banked up to this many
/// microseconds; what is left over waits.
///
/// shortcut: while the title streams thousands of slots in at once, the
/// slots not made yet read nothing (null views) for a few frames, as
/// streaming pops in; making all of them at once stalled frames for
/// seconds.
const SLOT_BUDGET_US: u64 = 8_000;
/// Slots taken to make at once (the rest wait without being looked at).
const SLOT_CHUNK: usize = 256;

impl Bindless {
    /// The slot-making budget, refilled for the time passed since.
    fn budget(&mut self) -> &mut [u64; 2] {
        let now = std::time::Instant::now();
        let (at, left) = self
            .budget
            .get_or_insert((now, [SLOT_BUDGET_US; 2]));
        let add = now.duration_since(*at).as_micros() as u64 / 8;
        if add > 0 {
            *at = now;
            for l in left.iter_mut() {
                *l = (*l + add).min(SLOT_BUDGET_US);
            }
        }
        left
    }
}

/// What one bindless slot needs at submission.
#[derive(Default)]
struct SlotUse {
    /// Title memory it reads (host copy, address, bytes).
    read: Option<(Arc<crate::shadow::Shadow>, usize, usize)>,
    /// What shaders may write through it.
    write: Option<(Arc<crate::shadow::Shadow>, usize, usize)>,
    /// A misaligned constant buffer: bounce slice, title address, bytes.
    bounce: Option<(usize, u64, u32)>,
    /// The host resource it views.
    retained: Option<ID3D12Resource>,
    /// It views a placed texture: the texture's object (0 for none), made
    /// again when the texture is evicted.
    placed: Option<usize>,
    /// It views unmapped memory (looked at again when memory is remapped).
    unmapped: bool,
}

/// Placed textures reached only through bindless slots are uploaded when a
/// slot first reaches them, and then looked at again (their memory hashed)
/// at most this often, in frames, ...
///
/// shortcut: a texture the title rewrites in place without writing a slot
/// shows its new contents up to this many frames late through bindless
/// access (tables still upload what their draws read). Hashing all of them
/// at each submission cost seconds per frame while the world streams in.
const RECHECK_FRAMES: u64 = 60;
/// ... and at most this many of them per frame.
const RECHECKS: u32 = 8;

/// Bumped when views go stale or title memory is mapped or unmapped: what
/// bindless slots view may have changed though no slot was written (their
/// views' generations are checked, all at once, at the next submission).
static VIEWED: AtomicU64 = AtomicU64::new(0);

/// Bindless slots must be checked again: a resource their views name was
/// released, or title memory was mapped or unmapped.
pub(crate) fn bindless_viewed_changed() {
    VIEWED.fetch_add(1, Ordering::Release);
}

/// Bindless unordered access ranges larger than this are not copied back.
///
/// shortcut: copy-back finds what shaders wrote by keeping a range's bytes
/// before each submission; a bindless slot viewing a title's big shared heap
/// would cost a copy and a compare of all of it per submission whether or
/// not a shader writes there. Such ranges stay on the GPU's side.
const WRITE_CAP: usize = 8 << 20;

/// Gap below which two ranges of one host copy are refreshed as one: the
/// write-watch answers by 64 KiB page, and fewer, larger askings cost less.
const READ_GAP: usize = 1 << 20;

/// What a submission does for a heap's bindless slots, made again only when
/// the slots or what they view change ([`HostHeap::bindless_plan`]).
pub(crate) struct BindlessPlan {
    /// Title memory the slots read, merged per host copy (address, bytes):
    /// refreshed at each submission using the heap.
    pub(crate) reads: Vec<(Arc<crate::shadow::Shadow>, usize, usize)>,
    /// The unordered access slots' ranges (up to [`WRITE_CAP`] each), merged:
    /// copied back after a submission whose shaders may write memory.
    pub(crate) writes: Vec<(Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Misaligned constant buffers: bounce slice, title address, bytes.
    bounces: Vec<(usize, u64, u32)>,
    /// The host resources the slots view, held while a submission runs.
    _retained: Vec<ID3D12Resource>,
}

// SAFETY: D3D12 resources are free-threaded; the rest is plain data.
unsafe impl Send for BindlessPlan {}
// SAFETY: as above.
unsafe impl Sync for BindlessPlan {}

impl BindlessPlan {
    /// Copies the misaligned constant buffers into their bounce slices.
    pub(crate) fn copy_constants(&self) {
        for &(cpu, address, len) in &self.bounces {
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::Copies, 1);
                add(C::CopyBytes, u64::from(len));
            }
            crate::host::copy_mapped(cpu as *mut u8, address as *const u8, len as usize, false);
            let us = started.elapsed().as_micros() as u64;
            crate::host::PREPARING[0].fetch_add(us, Ordering::Relaxed);
            crate::host::PREPARING[1].fetch_add(u64::from(len), Ordering::Relaxed);
        }
    }
}

/// `ranges` (host copy, address, bytes) sorted and merged per host copy
/// where they overlap or lie within `gap` of each other.
fn merged(
    mut ranges: Vec<(Arc<crate::shadow::Shadow>, usize, usize)>,
    gap: usize,
) -> Vec<(Arc<crate::shadow::Shadow>, usize, usize)> {
    ranges.sort_unstable_by_key(|(s, a, _)| (Arc::as_ptr(s) as usize, *a));
    let mut out: Vec<(Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
    for (shadow, address, len) in ranges {
        if let Some((s, a, l)) = out.last_mut()
            && Arc::ptr_eq(s, &shadow)
            && address <= *a + *l + gap
        {
            *l = (*l).max(address + len - *a);
            continue;
        }
        out.push((shadow, address, len));
    }
    out
}

impl Drop for HostHeap {
    fn drop(&mut self) {
        if let Ok(mut kept) = self.kept.lock() {
            for res in kept.drain(..).flatten() {
                crate::host::retire(res);
            }
        }
    }
}

/// The raw views of the bytes structured and typed buffer views read, for
/// shaders that read such a buffer without knowing its stride: they take a
/// raw handle, which reads a raw view only (over the title's own view the
/// host GPUs disagree: an RTX 5070 reads other dwords than a Radeon). The
/// runtime stride table names the slot.
struct Aliases {
    /// The pool slot of each heap slot that has a raw view.
    of: HashMap<u32, u32>,
    /// Pool slots given back.
    free: Vec<u32>,
    /// Pool slots from here on were never taken.
    next: u32,
}

/// Slots past the dynamic ones in a view heap for [`Aliases`]: one per
/// structured or typed buffer view in the heap at a time.
/// shortcut: a heap holding more such views at once leaves the rest without
/// a raw view (they read zeros through the unknown-stride path, with a
/// warning); the count is a guess, not a measured need.
const ALIASES: u32 = 1 << 16;

/// The pool's first slots: null raw views (a shader view, an unordered
/// access one) for views that cannot have a raw view of their own.
const NULL_ALIASES: u32 = 2;

struct StrideTable {
    _resource: ID3D12Resource,
    ptr: usize,
    gpu: u64,
    len: u32,
}

impl StrideTable {
    fn new(device: &ID3D12Device, len: u32) -> Option<Self> {
        let bytes = (u64::from(len) * u64::from(StrideEntry::BYTES)).max(8);
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
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
        let mut resource: Option<ID3D12Resource> = None;
        let mut p = std::ptr::null_mut();
        // SAFETY: plain committed upload buffer creation and persistent CPU map.
        let gpu = unsafe {
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_UPLOAD,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut resource,
                )
                .ok()?;
            let res = resource.as_ref()?;
            res.Map(0, None, Some(&mut p)).ok()?;
            std::ptr::write_bytes(p.cast::<u8>(), 0, bytes as usize);
            res.GetGPUVirtualAddress()
        };
        Some(Self {
            _resource: resource?,
            ptr: p as usize,
            gpu,
            len,
        })
    }
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!(
                "{:#x} past its end (slot {})",
                va - start - bytes,
                (va - start) / step
            )
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (#59, #71).
pub(crate) const DYNAMIC: u32 = 1 << 19;

/// Dynamic slots when the host caps the whole view heap at [`PORTABLE_VIEWS`]
/// (NVIDIA Tier 2/3 1M limit): 2^18 (262,144) leaves 737,856 mirrored title
/// slots while preventing world-load dynamic slot exhaustion.
const DYNAMIC_CAPPED: u32 = 1 << 18;

/// The host's limit for shader-visible sampler heaps, on every binding
/// tier.
const SAMPLERS: u32 = D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE;

/// Dynamic slots of a sampler heap capped at [`SAMPLERS`], for tables
/// reaching past the title slots it keeps.
const SAMPLER_DYNAMIC: u32 = 256;

impl HostHeap {
    #[cfg(all(test, feature = "diagnostics"))]
    pub(super) fn retains_resource(&self, index: u32) -> bool {
        self.kept.lock().unwrap_or_else(|e| e.into_inner())[index as usize].is_some()
    }

    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        if sampler && count > SAMPLERS {
            // shortcut: samplers past the host's limit are left out; tables
            // reaching them are made afresh in the dynamic slots.
            tracing::warn!(count, "host sampler heap capped at {SAMPLERS}");
            let heap = match make(SAMPLERS) {
                Ok(heap) => heap,
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            };
            return Self::with(
                host,
                heap,
                kind,
                SAMPLERS - SAMPLER_DYNAMIC,
                SAMPLER_DYNAMIC,
            );
        }
        let aliases = if sampler { 0 } else { ALIASES };
        // The larger dynamic range only where the host takes it on top of
        // every title slot; else the capped dynamic range, and the portable cap.
        // A view heap ends with its guard slot (see [`HostHeap::guard_slot`]).
        let (heap, count, extra) = if sampler {
            match make(count) {
                Ok(heap) => (heap, count, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if let Ok(heap) = make(count + DYNAMIC + aliases + 1) {
            (heap, count, DYNAMIC)
        } else {
            match make(count + DYNAMIC_CAPPED + aliases + 1) {
                Ok(heap) => (heap, count, DYNAMIC_CAPPED),
                Err(e) if count + DYNAMIC_CAPPED + aliases + 1 > PORTABLE_VIEWS => {
                    // shortcut: descriptors past the host's limit are left
                    // out; tables past `count` go through dynamic slots.
                    tracing::warn!(error = %e, count, dynamic = DYNAMIC_CAPPED, "host view heap capped at {PORTABLE_VIEWS}");
                    (
                        make(PORTABLE_VIEWS).ok()?,
                        PORTABLE_VIEWS - DYNAMIC_CAPPED - aliases - 1,
                        DYNAMIC_CAPPED,
                    )
                }
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        };
        Self::with(host, heap, kind, count, extra)
    }

    /// The host heap over `heap`: `count` title slots, then `extra` dynamic
    /// ones, plus one trailing guard slot on view heaps, all made null.
    fn with(
        host: &Host,
        heap: ID3D12DescriptorHeap,
        kind: D3D12_DESCRIPTOR_HEAP_TYPE,
        count: u32,
        extra: u32,
    ) -> Option<Self> {
        let sampler = kind == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        let aliases = if sampler { 0 } else { ALIASES };
        let total = count + extra + aliases + u32::from(!sampler);
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, extra, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(total) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let strides = if sampler {
            None
        } else {
            StrideTable::new(&host.device, count + extra)
        };
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            spare: extra,
            sampler,
            made: Mutex::new(vec![SlotState::Null; count as usize]),
            kept: Mutex::new(vec![None; total as usize]),
            next: Mutex::new((0, vec![false; extra as usize])),
            bindless: Mutex::default(),
            strides,
            alias_base: count + extra,
            aliases: Mutex::new(Aliases {
                of: HashMap::new(),
                free: Vec::new(),
                next: NULL_ALIASES,
            }),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, total) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
            return None;
        }
        if !sampler {
            null_buffer_view(h.cpu(h.alias_base));
            null_raw_uav(h.cpu(h.alias_base + 1));
        }
        tracing::debug!(
            ms = started.elapsed().as_millis(),
            "host descriptor heap made null"
        );
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        let Some(host) = host() else { return Ok(()) };
        let block_count = total.min(if self.sampler {
            D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE
        } else {
            1 << 16
        });
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: block_count,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < block_count {
                let n = filled.min(block_count - filled);
                host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = block_count.min(total - done);
                host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// The microseconds of slot making (`which`: 0 plans, 1 recording) left.
    fn budget_left(&self, which: usize) -> u64 {
        self.bindless
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .budget()[which]
    }

    /// Notes `us` microseconds of slot making spent.
    fn budget_spend(&self, which: usize, us: u64) {
        let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
        let left = &mut b.budget()[which];
        *left = left.saturating_sub(us);
    }

    /// `n` consecutive free dynamic slots (view heaps and capped sampler
    /// heaps), taken until [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if n == 0 || self.spare == 0 {
            return None;
        }
        if n > self.spare {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    n,
                    spare = self.spare,
                    sampler = self.sampler,
                    "a table larger than the dynamic slots is not made"
                );
            });
            return None;
        }
        let spare = self.spare as usize;
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let n = n as usize;
        let Some(first) = free_table(taken, *next as usize, n) else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                let (free, longest) = free_space(taken);
                tracing::warn!(
                    n,
                    cap = taken.len(),
                    free,
                    longest,
                    "no free dynamic descriptor slots: a table is not made"
                );
            });
            return None;
        };
        taken[first..first + n].fill(true);
        *next = ((first + n) % spare) as u32;
        Some(self.count + first as u32)
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// The slot translated shaders read in place of a computed heap index
    /// at or past it (a stale buffer can give any index, and one past the
    /// heap hangs some GPUs): in a view heap its last slot, after the
    /// dynamic ones, null from the start and never made; in a sampler heap
    /// its last slot (any sampler reads safely).
    pub(crate) fn guard_slot(&self) -> u32 {
        if self.sampler {
            (self.count + self.spare).saturating_sub(1)
        } else {
            self.count + self.spare + ALIASES
        }
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        self.release_with(first, n, Self::forget_view);
    }

    pub(super) fn release_with(&self, first: u32, n: u32, mut clear: impl FnMut(&Self, u32)) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        // Clear the retiring list's aliases before another list can lease
        // these slots and publish its own stride entries.
        for index in first..first.saturating_add(n).min(self.alias_base) {
            clear(self, index);
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = at.saturating_add(n).min(self.spare) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            slots.fill(false);
        }
    }

    /// Makes a matched native UAV pair from the CPU view. The GPU slot is
    /// held until list reset like the descriptors in a fresh table.
    pub(crate) fn clear_texture_uav(
        &self,
        list: &crate::host::HostList,
        texture: &ID3D12Resource,
        words: Option<&[u32; 10]>,
    ) -> Option<(u32, D3D12_GPU_DESCRIPTOR_HANDLE, D3D12_CPU_DESCRIPTOR_HANDLE)> {
        let mut desc = words.map(uav_desc);
        // SAFETY: a live host texture.
        let info = unsafe { texture.GetDesc() };
        if !fits(&info, desc.as_mut().map(|d| &mut d.Format), true) {
            return None;
        }
        let device = &host()?.device;
        let cpu = list.clear_uav_cpu()?;
        let slot = self.dynamic(1)?;
        // SAFETY: both handles name live heap slots. Copy from the
        // CPU-only heap to the shader-visible heap.
        let gpu = unsafe {
            device.CreateUnorderedAccessView(
                texture,
                None::<&ID3D12Resource>,
                desc.as_ref().map(std::ptr::from_ref),
                cpu,
            );
            device.CopyDescriptorsSimple(
                1, self.cpu(slot), cpu, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            );
            D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: self.heap.GetGPUDescriptorHandleForHeapStart().ptr
                    + u64::from(slot) * u64::from(self.step),
            }
        };
        // The list's texture state retains the resource through execution.
        self.keep(slot, None);
        Some((slot, gpu, cpu))
    }


    /// Makes slot `index` (a dynamic one) from `view`; a typed null (or a null
    /// matching `view`) if the host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View, null_props: Option<(u32, u32)>) {
        let kept = make(view, self.cpu(index));
        if kept.is_none() {
            if self.sampler {
                self.null_sampler(index);
            } else {
                match null_props.and_then(null_desc_for_props) {
                    Some(desc) => write_null(&desc, self.cpu(index)),
                    None => null_for(view, self.cpu(index)),
                }
            }
        }
        self.note_view(index, view, kept.as_ref());
        self.keep(index, kept.flatten());
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
        self.forget_view(index);
    }

    /// Writes a typed null descriptor into slot `index` matching the resource
    /// kind the shader expects (CBV, buffer/texture SRV, or buffer/texture UAV),
    /// rather than a generic 2D SRV (#135).
    pub(crate) fn make_null_kind(
        &self,
        index: u32,
        kind: victory_shader::ResourceType,
        dwords: u8,
    ) {
        if self.sampler {
            self.null_sampler(index);
            self.keep(index, None);
            return;
        }
        let at = self.cpu(index);
        match kind {
            victory_shader::ResourceType::Cbv => {
                if let Some(host) = host() {
                    let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
                    // SAFETY: a slot of our heap; zero location makes a null CBV.
                    unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
                }
            }
            victory_shader::ResourceType::Srv if dwords == 4 => null_buffer_view(at),
            victory_shader::ResourceType::Uav => {
                if let Some(host) = host() {
                    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                        Format: DXGI_FORMAT_R32_UINT,
                        ViewDimension: if dwords == 4 {
                            D3D12_UAV_DIMENSION_BUFFER
                        } else {
                            D3D12_UAV_DIMENSION_TEXTURE2D
                        },
                        ..Default::default()
                    };
                    // SAFETY: a slot of our heap; null resource makes a null UAV.
                    unsafe {
                        host.device.CreateUnorderedAccessView(
                            None::<&ID3D12Resource>,
                            None::<&ID3D12Resource>,
                            Some(&desc),
                            at,
                        )
                    };
                }
            }
            _ => null_view(at),
        }
        self.keep(index, None);
    }

    /// Makes slot `index` a typed null descriptor matching a shader's
    /// annotation (view heaps only; sampler heaps get a plain sampler).
    pub(crate) fn make_typed_null(&self, index: u32, desc: &NullDesc) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            write_null(desc, self.cpu(index));
        }
        self.keep(index, None);
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = kept.get_mut(index as usize) {
            // A running submission may still read the old resource through
            // this slot.
            if let Some(old) = std::mem::replace(slot, resource) {
                crate::host::retire(old);
            }
        }
    }

    fn set_entry(&self, index: u32, entry: StrideEntry) {
        if let Some(t) = &self.strides
            && index < t.len
        {
            // SAFETY: `t.ptr` points to a live mapped UPLOAD buffer of
            // `t.len` entries of two dwords.
            unsafe {
                (t.ptr as *mut [u32; 2])
                    .add(index as usize)
                    .write(entry.words());
            }
        }
    }

    /// GPU virtual address of the runtime stride table: a [`StrideEntry`]
    /// per slot.
    pub(crate) fn stride_table_gpu(&self) -> Option<u64> {
        self.strides.as_ref().map(|t| t.gpu)
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds).
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        self.keep(index, None);
        self.forget_view(index);
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: size.min(65_536).next_multiple_of(256),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        self.mirror_slot(index, handle, for_draw, None)
    }

    /// As [`HostHeap::mirror`] for a draw or dispatch, placing a typed null
    /// descriptor matching `null_props` when the title slot has no view (or its
    /// view cannot be made on the host).
    pub(crate) fn mirror_for_draw(
        &self,
        index: u32,
        handle: u64,
        null_props: Option<(u32, u32)>,
    ) -> Option<View> {
        self.mirror_slot(index, handle, true, null_props)
    }

    fn mirror_slot(
        &self,
        index: u32,
        handle: u64,
        for_draw: bool,
        null_props: Option<(u32, u32)>,
    ) -> Option<View> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): a typed null when the shader annotates the slot,
            // otherwise the default null descriptor.
            {
                let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
                if !self.sampler
                    && let Some(props) = null_props
                    && let Some(desc) = null_desc_for_props(props)
                {
                    let want = SlotState::TypedNull(props.0, props.1);
                    if made[index as usize] != want {
                        made[index as usize] = want;
                        self.make_typed_null(index, &desc);
                    }
                } else if made[index as usize] != SlotState::Null {
                    made[index as usize] = SlotState::Null;
                    self.make_null(index);
                }
            }
            if !self.sampler {
                self.untrack_bindless(index);
            }
            return None;
        };
        let is_buf = !self.sampler && view.is_buffer();
        let want = SlotState::View(key);
        {
            let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            if made[index as usize] != want {
                let later = !for_draw && is_buf;
                made[index as usize] = if later { SlotState::Null } else { want };
                let kept = if later {
                    None
                } else {
                    make(&view, self.cpu(index))
                };
                if kept.is_none() && !self.sampler {
                    match null_props.and_then(null_desc_for_props) {
                        Some(desc) => write_null(&desc, self.cpu(index)),
                        None => null_for(&view, self.cpu(index)),
                    }
                }
                self.keep(index, kept.flatten());
            }
        }
        if !self.sampler {
            if is_buf {
                let placed = matches!(&view, View::Placed(false, r, _) if r[0] != 1);
                self.track_bindless(index, (i, generation), placed);
            } else {
                self.untrack_bindless(index);
            }
        }
        Some(view)
    }

    fn track_bindless(&self, index: u32, entry: (u32, u32), placed: bool) {
        let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
        if b.slots.insert(index, entry) != Some(entry) {
            b.dirty.insert(index);
            if let Some(bounce) = b.cbv_bounce.get_mut(&index) {
                bounce.active = false;
            }
        }
        let changed = if placed {
            b.placed_set.insert(index, entry) != Some(entry)
        } else {
            b.placed_set.remove(&index).is_some()
        };
        if changed {
            b.record_pending.insert(index);
        }
    }

    fn untrack_bindless(&self, index: u32) {
        let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
        if b.slots.remove(&index).is_some() {
            b.dirty.insert(index);
            if let Some(bounce) = b.cbv_bounce.get_mut(&index) {
                bounce.active = false;
            }
        }
        if b.placed_set.remove(&index).is_some() {
            b.recorded.remove(&index);
        }
    }

    /// Slot `index` (viewing `entry`) is no longer one for bindless access.
    fn drop_bindless(&self, b: &mut Bindless, index: u32, entry: Option<(u32, u32)>) {
        b.uses.remove(&index);
        b.recorded.remove(&index);
        if entry.is_none_or(|e| b.slots.get(&index) == Some(&e)) {
            b.slots.remove(&index);
            if let Some(bounce) = b.cbv_bounce.get_mut(&index) {
                bounce.active = false;
            }
            b.placed_set.remove(&index);
        }
    }

    fn clear_stale_slot(&self, index: u32) {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if made[index as usize] != SlotState::Null {
            made[index as usize] = SlotState::Null;
            self.make_null(index);
        }
    }

    fn set_unmapped_null(&self, index: u32, view: &View) {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        made[index as usize] = SlotState::Null;
        if !self.sampler {
            null_for(view, self.cpu(index));
        }
        self.keep(index, None);
    }

    fn ensure_placed_slot(
        &self,
        index: u32,
        key: u64,
        shader_view: &View,
        raw_tex: *mut std::ffi::c_void,
    ) {
        use windows::core::Interface;
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let same_tex = self
            .kept
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)
            .and_then(Option::as_ref)
            .is_some_and(|r| r.as_raw() == raw_tex);
        let want = SlotState::View(key);
        if made[index as usize] != want || !same_tex {
            let kept = make(shader_view, self.cpu(index));
            if kept.is_none() {
                null_for(shader_view, self.cpu(index));
                made[index as usize] = SlotState::Null;
            } else {
                made[index as usize] = want;
            }
            self.note_view(index, &view, kept.as_ref());
            self.keep(index, kept.flatten());
        }
    }

    fn ensure_buffer_slot(&self, index: u32, key: u64, view: &View) -> Option<ID3D12Resource> {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let want = SlotState::View(key);
        if made[index as usize] != want {
            let kept = make(view, self.cpu(index));
            let res = kept.flatten();
            if res.is_none() {
                null_for(view, self.cpu(index));
                made[index as usize] = SlotState::Null;
            } else {
                made[index as usize] = want;
            }
            self.keep(index, res.clone());
            return res;
        }
        self.kept
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)
            .and_then(Option::clone)
    }

    fn ensure_cbv_bounce(
        &self,
        index: u32,
        aligned: u32,
    ) -> Option<(u64, usize, ID3D12Resource, bool)> {
        let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = b.cbv_bounce.get_mut(&index)
            && slot.cap >= aligned
        {
            let was_active = std::mem::replace(&mut slot.active, true);
            return Some((slot.gpu, slot.cpu, slot.resource.clone(), was_active));
        }
        let needed = u64::from(aligned);
        let need_chunk = b
            .bounce_chunks
            .last()
            .is_none_or(|&(_, _, _, used)| used + needed > BOUNCE_CHUNK);
        if need_chunk {
            let chunk_size = BOUNCE_CHUNK.max(needed);
            let (res, data) = host()?.shared_buffer(chunk_size)?;
            crate::host::name(&res, || "bindless cbv bounce".into());
            // SAFETY: a live buffer.
            let gpu = unsafe { res.GetGPUVirtualAddress() };
            b.bounce_chunks.push((res, data as usize, gpu, 0));
        }
        let last = b.bounce_chunks.last_mut()?;
        let at = last.3;
        last.3 += needed;
        let (gpu, cpu, res) = (last.2 + at, last.1 + at as usize, last.0.clone());
        b.cbv_bounce.insert(
            index,
            BounceSlot {
                gpu,
                cpu,
                cap: aligned,
                resource: res.clone(),
                active: true,
            },
        );
        Some((gpu, cpu, res, false))
    }

    fn materialize_bindless_cbv(
        &self,
        index: u32,
        key: u64,
        address: u64,
        capped: u32,
        reads: &mut Vec<(Arc<crate::shadow::Shadow>, usize, usize)>,
        bounces: &mut Vec<(usize, u64, u32)>,
    ) -> Option<ID3D12Resource> {
        let host = host()?;
        let aligned = capped.next_multiple_of(256);
        if address.is_multiple_of(256)
            && let Some(shadow) = crate::shadow::find(address, u64::from(capped))
            && shadow.gpu(address).is_multiple_of(256)
        {
            let bounce_active = self
                .bindless
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .cbv_bounce
                .get(&index)
                .is_some_and(|s| s.active);
            let want = SlotState::View(key);
            {
                let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
                if made[index as usize] != want || bounce_active {
                    let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                        BufferLocation: shadow.gpu(address),
                        SizeInBytes: aligned,
                    };
                    // SAFETY: a slot of our heap; `shadow` covers the range.
                    unsafe {
                        host.device
                            .CreateConstantBufferView(Some(&desc), self.cpu(index))
                    };
                    made[index as usize] = want;
                    self.keep(index, Some(shadow.buffer.clone()));
                }
            }
            if bounce_active
                && let Some(slot) = self
                    .bindless
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .cbv_bounce
                    .get_mut(&index)
            {
                slot.active = false;
            }
            let len = u64::from(capped).min(shadow.left(address)) as usize;
            if len > 0 {
                reads.push((shadow.clone(), address as usize, len));
            }
            return Some(shadow.buffer.clone());
        }

        let (gpu, cpu, res, was_active) = self.ensure_cbv_bounce(index, aligned)?;
        let want = SlotState::View(key);
        {
            let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            if made[index as usize] != want || !was_active {
                let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                    BufferLocation: gpu,
                    SizeInBytes: aligned,
                };
                // SAFETY: a slot of our heap; `gpu` is 256-byte aligned in `res`.
                unsafe {
                    host.device
                        .CreateConstantBufferView(Some(&desc), self.cpu(index))
                };
                made[index as usize] = want;
                self.keep(index, Some(res.clone()));
            }
        }
        bounces.push((cpu, address, capped));
        Some(res)
    }

    /// At draw/dispatch recording time: makes the placed-texture slots
    /// changed since (or whose textures were evicted), and records on `list`
    /// the upload of the textures they newly reach and of a few known ones
    /// due a look ([`RECHECK_FRAMES`]), each moved to the shader resource
    /// state. Buffer and constant-buffer views are materialized and refreshed
    /// at submission from [`HostHeap::bindless_plan`].
    ///
    /// shortcut: a texture is moved to the shader resource state in the list
    /// that uploads it and keeps that state after; one another list moves
    /// (a copy into it, say) is read in that state through bindless slots.
    pub(crate) fn prepare_bindless_record(&self, list: &crate::host::HostList) {
        if self.sampler {
            return;
        }
        let epoch = crate::resource::placed_epoch();
        let frame = crate::queue::presented();
        // Most draws find nothing to do: one look under the lock.
        let evictions = {
            let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
            let evictions = b.recorded_epoch != epoch && b.looked[1] != frame + 1;
            let have = b.budget()[1] > 0;
            let recheck_due = (b.rechecks.0 != frame || b.rechecks.1 < RECHECKS)
                && b
                    .recheck
                    .front()
                    .is_some_and(|&(f, _)| f + RECHECK_FRAMES <= frame);
            if !evictions && (b.record_pending.is_empty() || !have) && !recheck_due {
                return;
            }
            evictions
        };
        let started = std::time::Instant::now();
        // Asked before taking the heap's lock: the placed textures' lock is
        // taken first where both are held.
        let live = evictions.then(crate::resource::placed_live);
        let (todo, due) = {
            let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
            let b = &mut *b;
            if let Some(live) = &live {
                b.uploaded.retain(|t, _| live.contains(t));
                for (&slot, &(_, t)) in &b.recorded {
                    if !live.contains(&t) {
                        b.record_pending.insert(slot);
                    }
                }
                b.recorded.retain(|_, (_, t)| live.contains(t));
                b.recorded_epoch = epoch;
                b.looked[1] = frame + 1;
            }
            let recheck_due = b
                .recheck
                .front()
                .is_some_and(|&(f, _)| f + RECHECK_FRAMES <= frame);
            if b.record_pending.is_empty() && !recheck_due {
                return;
            }
            let mut todo: Vec<(u32, Option<(u32, u32)>)> = Vec::new();
            while todo.len() < SLOT_CHUNK
                && let Some(i) = b.record_pending.pop_first()
            {
                todo.push((i, b.placed_set.get(&i).copied()));
            }
            let mut due = Vec::new();
            while due.len() < RECHECKS as usize
                && (b.rechecks.0 != frame || b.rechecks.1 < RECHECKS)
                && let Some(&(f, t)) = b.recheck.front()
                && f + RECHECK_FRAMES <= frame
            {
                b.recheck.pop_front();
                due.push(t as crate::object::This);
            }
            (todo, due)
        };
        let mut stale = Vec::new();
        let mut learnt = Vec::new();
        let mut reached: Vec<(Option<(u32, u64)>, crate::object::This)> = Vec::new();
        let left = self.budget_left(1);
        let mut todo = todo.into_iter();
        for (index, entry) in todo.by_ref() {
            let Some((vi, generation)) = entry else {
                continue;
            };
            if started.elapsed().as_micros() as u64 > left {
                // Over budget: the rest waits (this one too).
                self.bindless
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .record_pending
                    .insert(index);
                break;
            }
            match self.resolve_placed(index, vi, generation) {
                Ok(Some(t)) => {
                    let key = u64::from(generation) << 32 | u64::from(vi);
                    reached.push((Some((index, key)), t));
                }
                Ok(None) => {}
                Err(()) => stale.push(index),
            }
        }
        {
            let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
            b.record_pending.extend(todo.map(|(i, _)| i));
        }
        reached.extend(due.into_iter().map(|t| (None, t)));
        let mut deferred_slots = Vec::new();
        let mut deferred_recheck = Vec::new();
        let mut reached_iter = reached.into_iter();
        for (k, (slot, texture)) in reached_iter.by_ref().enumerate() {
            // Check budget between texture uploads too: detiling and uploading
            // many large placed textures can otherwise stall one draw for seconds.
            if k > 0 && started.elapsed().as_micros() as u64 > left {
                match slot {
                    Some((index, _)) => deferred_slots.push(index),
                    None => deferred_recheck.push((frame.saturating_sub(RECHECK_FRAMES), texture as usize)),
                }
                break;
            }
            // The texture may be evicted meanwhile: it is used only while
            // it is known alive.
            crate::resource::with_placed(texture, || {
                let Some(host_tex) = crate::resource::host_resource(texture) else {
                    return;
                };
                if crate::texture::title_written(host_tex)
                    && self.upload_due(texture, host_tex, frame)
                {
                    crate::texture::upload(list, texture, host_tex);
                }
                const SHADER_RESOURCE: u32 = 0x40 | 0x80;
                crate::list::ensure_state(list, texture, SHADER_RESOURCE);
            });
            if let Some((index, key)) = slot {
                learnt.push((index, (key, texture as usize)));
            }
        }
        for (slot, texture) in reached_iter {
            match slot {
                Some((index, _)) => deferred_slots.push(index),
                None => deferred_recheck.push((frame.saturating_sub(RECHECK_FRAMES), texture as usize)),
            }
        }
        if !stale.is_empty()
            || !learnt.is_empty()
            || !deferred_slots.is_empty()
            || !deferred_recheck.is_empty()
        {
            let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
            b.recorded.extend(learnt);
            b.record_pending.extend(deferred_slots);
            for item in deferred_recheck {
                b.recheck.push_front(item);
            }
            for idx in stale {
                self.drop_bindless(&mut b, idx, None);
            }
        }
        let us = started.elapsed().as_micros() as u64;
        self.budget_spend(1, us);
        crate::host::BINDLESS_US[1].fetch_add(us, Ordering::Relaxed);
    }

    /// The placed texture slot `index` views, made in the slot: `Err` if the
    /// view went stale, `None` if it is not a placed texture or cannot be made
    /// (the slot is then null).
    fn resolve_placed(
        &self,
        index: u32,
        vi: u32,
        generation: u32,
    ) -> Result<Option<crate::object::This>, ()> {
        use windows::core::Interface;
        let Some(view) = view_at(vi, generation) else {
            self.clear_stale_slot(index);
            return Err(());
        };
        let View::Placed(false, resource, desc) = &view else {
            return Ok(None);
        };
        const BUFFER: u32 = 1;
        if resource[0] == BUFFER {
            return Ok(None);
        }
        let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
        if !plausible(address, 1) {
            self.set_unmapped_null(index, &view);
            return Ok(None);
        }
        let Some(shader_view @ View::Shader(tex_res, _)) = placed_texture(&view) else {
            self.set_unmapped_null(index, &view);
            return Ok(None);
        };
        let tex_this = tex_res as crate::object::This;
        let Some(host_tex) = crate::resource::host_resource(tex_this) else {
            self.set_unmapped_null(index, &view);
            return Ok(None);
        };
        let key = u64::from(generation) << 32 | u64::from(vi);
        self.ensure_placed_slot(index, key, &shader_view, host_tex.as_raw());
        Ok(Some(tex_this))
    }

    /// Whether to upload placed texture `texture` (reached through a slot)
    /// in a list recorded in `frame`: the first time, then as
    /// [`RECHECK_FRAMES`] allows.
    fn upload_due(&self, texture: crate::object::This, host: &ID3D12Resource, frame: u64) -> bool {
        use windows::core::Interface;
        let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
        let b = &mut *b;
        let due = match b.uploaded.get(&(texture as usize)) {
            None => true,
            // Another texture at the object's address.
            Some((h, _)) if h.as_raw() != host.as_raw() => true,
            Some((_, last)) if frame >= last + RECHECK_FRAMES => {
                if b.rechecks.0 != frame {
                    b.rechecks = (frame, 0);
                }
                b.rechecks.1 += 1;
                b.rechecks.1 <= RECHECKS
            }
            Some(_) => false,
        };
        if due {
            b.uploaded
                .insert(texture as usize, (host.clone(), frame));
            b.recheck.push_back((frame, texture as usize));
        }
        due
    }

    /// At submission: what the heap's bindless slots need. Only slots
    /// written since the last plan are made again; released views are found
    /// by one look at all slots' generations, and slots of evicted placed
    /// textures (or unmapped memory, once remapped) are made again. With
    /// nothing changed it is the last plan, whose reads the caller
    /// refreshes, which costs little when the title changed nothing.
    pub(crate) fn bindless_plan(&self) -> Option<Arc<BindlessPlan>> {
        if self.sampler {
            return None;
        }
        let epochs = (
            VIEWED.load(Ordering::Acquire),
            crate::resource::placed_epoch(),
        );
        // Asked before taking the heap's lock: the placed textures' lock is
        // taken first where both are held.
        let frame = crate::queue::presented() + 1;
        let evictions = {
            let b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
            b.checked.1 != epochs.1 && b.looked[0] != frame
        };
        let live = evictions.then(crate::resource::placed_live);
        let (mut todo, check) = {
            let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
            let b = &mut *b;
            // Released views and evicted textures are looked for once a frame.
            let look = b.looked[0] != frame && b.checked != epochs;
            if b.dirty.is_empty()
                && !look
                && let Some(plan) = &b.plan
            {
                return Some(plan.clone());
            }
            if !look
                && b.budget()[0] == 0
                && let Some(plan) = &b.plan
            {
                return Some(plan.clone());
            }
            let viewed = look && b.checked.0 != epochs.0;
            let live = if look { live.as_ref() } else { None };
            if look {
                b.checked = epochs;
                b.looked[0] = frame;
            }
            let mut dirty = BTreeSet::new();
            while dirty.len() < 4 * SLOT_CHUNK
                && let Some(i) = b.dirty.pop_first()
            {
                dirty.insert(i);
            }
            for (&i, u) in &b.uses {
                let evicted = live
                    .zip(u.placed)
                    .is_some_and(|(l, t)| !l.contains(&t));
                if evicted || viewed && u.unmapped {
                    dirty.insert(i);
                }
            }
            let check: Vec<(u32, (u32, u32))> = if viewed {
                b.slots
                    .iter()
                    .filter(|(i, _)| !dirty.contains(i))
                    .map(|(&i, &v)| (i, v))
                    .collect()
            } else {
                Vec::new()
            };
            let todo: Vec<(u32, Option<(u32, u32)>)> = dirty
                .into_iter()
                .map(|i| (i, b.slots.get(&i).copied()))
                .collect();
            (todo, check)
        };
        todo.extend(stale_views(&check).into_iter().map(|(i, v)| (i, Some(v))));
        let (started, left) = (std::time::Instant::now(), self.budget_left(0));
        let mut made_any = false;
        let mut todo = todo.into_iter();
        for (index, entry) in todo.by_ref() {
            if started.elapsed().as_micros() as u64 > left {
                // Over budget: the rest waits for the next submission.
                self.bindless
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .dirty
                    .insert(index);
                break;
            }
            made_any = true;
            let made = entry.map(|e| (e, self.use_slot(index, e)));
            let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
            match made {
                None => {
                    b.uses.remove(&index);
                }
                Some((_, Some(u))) => {
                    b.uses.insert(index, u);
                }
                Some((e, None)) => self.drop_bindless(&mut b, index, Some(e)),
            }
        }
        self.budget_spend(0, started.elapsed().as_micros() as u64);
        let mut b = self.bindless.lock().unwrap_or_else(|e| e.into_inner());
        b.dirty.extend(todo.map(|(i, _)| i));
        if !made_any && let Some(plan) = &b.plan {
            return Some(plan.clone());
        }
        let (mut reads, mut writes, mut bounces, mut retained) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for u in b.uses.values() {
            reads.extend(u.read.clone());
            writes.extend(u.write.clone());
            bounces.extend(u.bounce);
            retained.extend(u.retained.clone());
        }
        let plan = Arc::new(BindlessPlan {
            reads: merged(reads, READ_GAP),
            writes: merged(writes, 0),
            bounces,
            _retained: retained,
        });
        b.plan = Some(plan.clone());
        Some(plan)
    }

    /// Materializes bindless slot `index` from its view `(vi, generation)`
    /// and finds what it reads and writes; `None` if the slot is no longer
    /// one for bindless access (its view went stale, or is not of a buffer).
    fn use_slot(&self, index: u32, (vi, generation): (u32, u32)) -> Option<SlotUse> {
        use windows::core::Interface;
        let Some(view) = view_at(vi, generation) else {
            self.clear_stale_slot(index);
            return None;
        };
        if !view.is_buffer() {
            return None;
        }
        let unmapped = |view: &View| {
            self.set_unmapped_null(index, view);
            Some(SlotUse {
                unmapped: true,
                ..SlotUse::default()
            })
        };
        let key = u64::from(generation) << 32 | u64::from(vi);
        let mut slot = SlotUse::default();
        match &view {
            &View::Constant(address, size) => {
                let capped = size.min(65_536);
                if !valid_title_address(address, u64::from(capped)) {
                    return unmapped(&view);
                }
                let (mut reads, mut bounces) = (Vec::new(), Vec::new());
                match self.materialize_bindless_cbv(
                    index,
                    key,
                    address,
                    capped,
                    &mut reads,
                    &mut bounces,
                ) {
                    Some(res) => slot.retained = Some(res),
                    None => return unmapped(&view),
                }
                slot.read = reads.pop();
                slot.bounce = bounces.pop();
            }
            View::Placed(false, resource, desc) if resource[0] != 1 => {
                slot.placed = Some(0);
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                if !plausible(address, 1) {
                    self.set_unmapped_null(index, &view);
                    return Some(slot);
                }
                if let Some(shader_view @ View::Shader(tex_res, _)) = placed_texture(&view)
                    && let Some(host_tex) =
                        crate::resource::host_resource(tex_res as crate::object::This)
                {
                    self.ensure_placed_slot(index, key, &shader_view, host_tex.as_raw());
                    slot.placed = Some(tex_res);
                    slot.retained = Some(host_tex.clone());
                } else {
                    self.set_unmapped_null(index, &view);
                }
            }
            _ => {
                let Some((address, len)) = buffer_range(&view) else {
                    self.set_unmapped_null(index, &view);
                    return Some(slot);
                };
                if !valid_title_address(address, len) {
                    return unmapped(&view);
                }
                slot.retained = self.ensure_buffer_slot(index, key, &view);
                if let Some(shadow) = crate::shadow::find(address, len) {
                    let clamped = len.min(shadow.left(address)) as usize;
                    if clamped > 0 {
                        slot.read = Some((shadow.clone(), address as usize, clamped));
                        if matches!(view, View::Unordered(..) | View::Placed(true, ..)) {
                            if clamped <= WRITE_CAP {
                                slot.write = Some((shadow, address as usize, clamped));
                            } else {
                                static ONCE: std::sync::Once = std::sync::Once::new();
                                ONCE.call_once(|| {
                                    tracing::warn!(
                                        bytes = clamped,
                                        "a bindless unordered access view too big to copy back"
                                    );
                                });
                            }
                        }
                    }
                }
            }
        }
        Some(slot)
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let state = *self
            .made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)?;
        let key = match state {
            SlotState::Null => return Some("null".into()),
            SlotState::TypedNull(p0, p1) => return Some(format!("null ({p0:#x}, {p1:#x})")),
            SlotState::View(key) => key,
        };
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Whether `[address, address + len)` may be title memory: inside the
/// title's address range, not null.
fn plausible(address: u64, len: u64) -> bool {
    address >= 0x10000
        && len > 0
        && address
            .checked_add(len)
            .is_some_and(|end| end <= (1u64 << 48))
}

/// Whether `[address, address + len)` is title memory: plausible, and in a
/// title buffer or committed. The kernel is asked only outside buffers:
/// in the title's big pools each asking costs milliseconds.
fn valid_title_address(address: u64, len: u64) -> bool {
    plausible(address, len)
        && (crate::shadow::in_buffer(address, len)
            || crate::texture::committed(address as *const u8, 1))
}

/// Total free slots and the longest contiguous run in the physical heap.
fn free_space(taken: &[bool]) -> (usize, usize) {
    let mut free = 0;
    let mut longest = 0;
    for gap in taken.split(|&held| held) {
        free += gap.len();
        longest = longest.max(gap.len());
    }
    (free, longest)
}

/// Try each possible table start from the cursor, then from slot zero.
/// A table may cross the cursor on the second pass, but must fit in the heap.
fn free_table(taken: &[bool], cursor: usize, n: usize) -> Option<usize> {
    let end = taken.len().checked_sub(n)? + 1;
    let split = cursor.min(end);
    for starts in [split..end, 0..split] {
        let mut first = starts.start;
        while first < starts.end {
            match taken[first..first + n].iter().position(|&held| held) {
                Some(offset) => first += offset + 1,
                None => return Some(first),
            }
        }
    }
    None
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

pub(crate) fn srv_mips(
    texture: &ID3D12Resource,
    words: &[u32; 10],
) -> Option<std::ops::Range<u32>> {
    let count = crate::host::color_mip_count(texture)?;
    let desc = srv_desc(words);
    if desc.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D {
        return None;
    }
    // SAFETY: the view dimension selects this union member.
    let view = unsafe { desc.Anonymous.Texture2D };
    let start = view.MostDetailedMip;
    let end = start.saturating_add(view.MipLevels).min(count);
    (view.PlaneSlice == 0 && start < end && (start != 0 || end != count)).then_some(start..end)
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let first_off = buffer.FirstElement.checked_mul(element)?;
    let byte_len = u64::from(buffer.NumElements).checked_mul(element)?;
    Some((base.checked_add(first_off)?, byte_len))
}

fn uav_element_size(desc: &D3D12_UNORDERED_ACCESS_VIEW_DESC) -> Option<u64> {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        Some(u64::from(buffer.StructureByteStride))
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        Some(4)
    } else {
        crate::resource::element(desc.Format.0 as u32).map(|(bytes, _)| bytes)
    }
}

impl HostHeap {
    /// Slot `index` was just made from `view` (`made`: what [`make`] gave):
    /// writes its runtime stride table entry, and for a structured or typed
    /// buffer view makes the raw view of the same bytes the entry names.
    fn note_view(&self, index: u32, view: &View, made: Option<&Kept>) {
        if self.strides.is_none() {
            return;
        }
        let Some(kept) = made else {
            // A null view: nothing to read through any handle.
            return self.forget_view(index);
        };
        let stride = view_stride(view);
        let entry = match viewed_bytes(view, kept.as_ref()) {
            // Not a buffer view, or a raw one: the slot's own view.
            Viewed::Own => {
                self.unalias(index);
                StrideEntry {
                    stride,
                    ..StrideEntry::default()
                }
            }
            Viewed::Bytes {
                buffer,
                start,
                len,
                unordered,
            } => {
                let (raw_slot, bias) = self.alias(index, &buffer, start, len, unordered);
                StrideEntry {
                    stride,
                    raw_slot: Some(raw_slot),
                    bias,
                }
            }
            Viewed::Unknown => {
                once!("a buffer view whose bytes in its host copy are not found has no raw view");
                self.unalias(index);
                StrideEntry {
                    stride,
                    raw_slot: Some(self.null_alias(view_is_unordered(view))),
                    bias: 0,
                }
            }
        };
        self.set_entry(index, entry);
    }

    /// Slot `index`'s runtime stride table entry, as shaders read it.
    #[cfg(test)]
    pub(crate) fn entry_words(&self, index: u32) -> [u32; 2] {
        let t = self.strides.as_ref().expect("a view heap's table");
        assert!(index < t.len);
        // SAFETY: as in `set_entry`.
        unsafe { (t.ptr as *const [u32; 2]).add(index as usize).read() }
    }

    /// The first slot of the raw view pool.
    #[cfg(test)]
    pub(crate) fn alias_base(&self) -> u32 {
        self.alias_base
    }

    /// Slot `index` holds no view shaders read through the runtime stride
    /// table: its entry is cleared and its raw view given back.
    pub(super) fn forget_view(&self, index: u32) {
        if self.strides.is_none() {
            return;
        }
        self.unalias(index);
        self.set_entry(index, StrideEntry::default());
    }

    /// The slot of the null raw view standing in for a raw view the host
    /// cannot make: reads give zeros, writes are dropped.
    fn null_alias(&self, unordered: bool) -> u32 {
        self.alias_base + u32::from(unordered)
    }

    /// Gives back the raw view slot of `index`, if it has one.
    fn unalias(&self, index: u32) {
        let mut a = self.aliases.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = a.of.remove(&index) {
            a.free.push(slot);
        }
    }

    /// Makes the raw view of bytes `[start, start + len)` of `buffer` for
    /// slot `index` (in the slot it already has for one, or a free one): the
    /// heap slot it is in, and the bytes from its start to `start`.
    ///
    /// The raw view ends with the view's last element, so reads past it
    /// give zeros and writes are dropped, as through the view itself.
    /// shortcut: it ends on the dword holding the last byte and starts up
    /// to 12 bytes before the first (raw views start on 16 bytes); only an
    /// offset wrapping below the view's start reaches those bytes.
    fn alias(
        &self,
        index: u32,
        buffer: &ID3D12Resource,
        start: u64,
        len: u64,
        unordered: bool,
    ) -> (u32, u32) {
        let Some(window) = RawWindow::covering(start, len) else {
            // shortcut: a view starting off a dword of its host copy (an
            // element size that is not a multiple of 4) has no raw view; a
            // raw load could not address its bytes either.
            once!(start, len, "a buffer view off the dwords of its host copy has no raw view");
            self.unalias(index);
            return (self.null_alias(unordered), 0);
        };
        let slot = {
            let mut a = self.aliases.lock().unwrap_or_else(|e| e.into_inner());
            let slot = match a.of.get(&index) {
                Some(&slot) => Some(slot),
                None => a.free.pop().or_else(|| {
                    (a.next < ALIASES).then(|| {
                        a.next += 1;
                        a.next - 1
                    })
                }),
            };
            let Some(slot) = slot else {
                once!(
                    "no free raw view slots: a buffer read without a known stride reads zeros"
                );
                return (self.null_alias(unordered), 0);
            };
            a.of.insert(index, slot);
            self.alias_base + slot
        };
        let Some(host) = host() else {
            return (self.null_alias(unordered), 0);
        };
        // SAFETY: a live buffer holding the range (the slot's own view was
        // just made on it) and a slot of our heap.
        unsafe {
            if unordered {
                let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            FirstElement: window.first,
                            NumElements: window.dwords,
                            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                            ..Default::default()
                        },
                    },
                };
                host.device.CreateUnorderedAccessView(
                    buffer,
                    None::<&ID3D12Resource>,
                    Some(&desc),
                    self.cpu(slot),
                );
            } else {
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: window.first,
                            NumElements: window.dwords,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                };
                host.device
                    .CreateShaderResourceView(buffer, Some(&desc), self.cpu(slot));
            }
        }
        (slot, window.bias)
    }
}

/// What a view the host made reads, for its raw view.
enum Viewed {
    /// Not a buffer, or through a raw view already.
    Own,
    /// These bytes of this host buffer (the view's own), read or also
    /// written.
    Bytes {
        buffer: ID3D12Resource,
        start: u64,
        len: u64,
        unordered: bool,
    },
    /// A structured or typed buffer view whose host buffer is not the one
    /// the bytes are found in now.
    Unknown,
}

/// Whether `view` is for unordered access.
fn view_is_unordered(view: &View) -> bool {
    matches!(view, View::Unordered(..) | View::Placed(true, ..))
}

/// The bytes of its host copy the structured or typed buffer view `view`
/// reads: as [`make`] places it (`kept`: the resource it made the view on),
/// from the first element's byte in the copy for the view's length.
fn viewed_bytes(view: &View, kept: Option<&ID3D12Resource>) -> Viewed {
    use windows::core::Interface;
    const BUFFER: u32 = 1;
    // The title buffer's address and length, and the view's description.
    let (base, len, words, unordered): (u64, u64, &[u32; 10], bool) = match view {
        View::Shader(r, Some(desc)) | View::Unordered(r, _, Some(desc)) => {
            let r = *r as crate::object::This;
            if crate::resource::host_resource(r).is_some() {
                return Viewed::Own;
            }
            let Some((data, len)) = crate::resource::buffer_bytes(r) else {
                return Viewed::Own;
            };
            (data as u64, len, desc, view_is_unordered(view))
        }
        View::Placed(unordered, resource, desc) if resource[0] == BUFFER => {
            let Ok(words) = desc[..10].try_into() else {
                return Viewed::Own;
            };
            (
                u64::from(desc[10]) | u64::from(desc[11]) << 32,
                u64::from(resource[4]) | u64::from(resource[5]) << 32,
                words,
                *unordered,
            )
        }
        _ => return Viewed::Own,
    };
    // (first element, elements, bytes per element), none for a raw view.
    let elements = if unordered {
        let desc = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        (b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 == 0)
            .then(|| (b.FirstElement, b.NumElements, uav_element_size(&desc).unwrap_or(0)))
    } else {
        let desc = srv_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        (b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 == 0)
            .then(|| (b.FirstElement, b.NumElements, element_size(&desc)))
    };
    let Some((first, count, element)) = elements else {
        return Viewed::Own;
    };
    let (Some(kept), Some(shadow)) = (kept, crate::shadow::find(base, len)) else {
        return Viewed::Unknown;
    };
    if element == 0 || shadow.buffer.as_raw() != kept.as_raw() {
        return Viewed::Unknown;
    }
    Viewed::Bytes {
        buffer: shadow.buffer.clone(),
        start: shadow.offset(base) + first * element,
        len: u64::from(count) * element,
        unordered,
    }
}

/// Element byte stride of `view` for the runtime stride table (`0` for
/// non-buffer views or descriptors without a known element size).
fn view_stride(view: &View) -> u32 {
    let bytes = match view {
        View::Shader(_, Some(desc)) => element_size(&srv_desc(desc)),
        View::Placed(false, resource, desc) if resource[0] == 1 => desc[..10]
            .try_into()
            .ok()
            .map_or(0, |w: &[u32; 10]| element_size(&srv_desc(w))),
        View::Unordered(_, _, Some(desc)) => uav_element_size(&uav_desc(desc)).unwrap_or(0),
        View::Placed(true, resource, desc) if resource[0] == 1 => desc[..10]
            .try_into()
            .ok()
            .and_then(|w: &[u32; 10]| uav_element_size(&uav_desc(w)))
            .unwrap_or(0),
        _ => 0,
    };
    u32::try_from(bytes).unwrap_or(0)
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = uav_element_size(&desc)?;
    let first_off = buffer.FirstElement.checked_mul(element)?;
    let byte_len = u64::from(buffer.NumElements).checked_mul(element)?;
    Some((base.checked_add(first_off)?, byte_len))
}

/// Whether a raw buffer view may start at element `first` (4 bytes each):
/// the host wants 16-byte aligned starts, the console only 4, and refuses
/// the call (removing the device) for the rest.
/// shortcut: a view the host refuses reads nothing (a null view); starting
/// it 16 bytes lower would need the shader to add the difference.
fn raw_start_fits(first: u64) -> bool {
    first.is_multiple_of(4)
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

/// A structured view's bytes as a raw view of its host copy: the first
/// dword and the dword count, for a view `offset` bytes into the copy whose
/// description starts at element `first` and has `count` elements of
/// `stride` bytes.
///
/// The host places structured element `i` at byte `i * stride` of the
/// resource, so a structured view whose start in the copy is not a multiple
/// of its stride (a title buffer's view 16 bytes past a 32-byte element)
/// cannot be made there. A raw view of the same bytes can if its start is
/// 16-byte aligned (`D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT`; the host removes the
/// device for one that is not), and it reads and writes the copy every other
/// view of that memory uses. Translated shaders read buffer views raw unless
/// they index them (`victory-shader`, "Buffer views in the heap").
fn raw_window(offset: u64, first: u64, count: u32, stride: u32) -> Option<(u64, u32)> {
    if stride == 0 || !stride.is_multiple_of(4) {
        return None;
    }
    let start = offset.checked_add(first.checked_mul(u64::from(stride))?)?;
    if !start.is_multiple_of(u64::from(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT)) {
        return None;
    }
    let dwords = u64::from(count) * u64::from(stride / 4);
    Some((start / 4, u32::try_from(dwords).ok()?))
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    let mut element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let offset = shadow.offset(address);
    if element != 0 && offset % element == 0 {
        buffer.FirstElement += offset / element;
    } else if let Some((first, count)) = raw_window(
        offset,
        buffer.FirstElement,
        buffer.NumElements,
        buffer.StructureByteStride,
    ) {
        // shortcut: a shader that indexes the view declares it structured,
        // and D3D12 leaves structured reads of a raw view undefined (NVIDIA
        // and WARP read the right bytes).
        once!(
            offset,
            element,
            "structured buffer view off its host copy's elements made raw"
        );
        *buffer = D3D12_BUFFER_SRV {
            FirstElement: first,
            NumElements: count,
            StructureByteStride: 0,
            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
        };
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        element = 4;
    } else {
        once!(
            offset,
            element,
            "buffer view not on an element of its host copy"
        );
        return None;
    }
    if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 && !raw_start_fits(buffer.FirstElement) {
        let first = buffer.FirstElement;
        once!(first, "raw buffer view not on 16 bytes of its host copy");
        return None;
    }
    // A view past its host copy's end is an invalid call (the debug layer
    // removes the device; drivers may fault): none past the end, the rest
    // cut to fit.
    // SAFETY: a live buffer.
    let capacity = unsafe { shadow.buffer.GetDesc() }.Width / element;
    if buffer.FirstElement >= capacity {
        let first = buffer.FirstElement;
        once!(first, capacity, "buffer view past the end of its host copy");
        return None;
    }
    if buffer.FirstElement.saturating_add(u64::from(buffer.NumElements)) > capacity {
        buffer.NumElements = (capacity - buffer.FirstElement) as u32;
    }
    // A view of no elements is an invalid call too (the title's own record
    // count can be 0): null, which reads 0 and drops writes.
    if buffer.NumElements == 0 {
        let first = buffer.FirstElement;
        once!(first, "buffer view of no elements");
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let mut element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = shadow.offset(address);
    buffer.CounterOffsetInBytes = 0;
    if element != 0 && offset % element == 0 {
        buffer.FirstElement += offset / element;
    } else if let Some((first, count)) = raw_window(
        offset,
        buffer.FirstElement,
        buffer.NumElements,
        buffer.StructureByteStride,
    ) {
        // As in `buffer_srv`; what shaders write lands in the shared copy,
        // in the range `buffer_range` gives (copied back to title memory).
        once!(
            offset,
            element,
            "structured unordered buffer view off its host copy's elements made raw"
        );
        *buffer = D3D12_BUFFER_UAV {
            FirstElement: first,
            NumElements: count,
            StructureByteStride: 0,
            CounterOffsetInBytes: 0,
            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
        };
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        element = 4;
    } else {
        once!(offset, element, "unordered buffer view not on an element of its host copy");
        return None;
    }
    if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 && !raw_start_fits(buffer.FirstElement) {
        let first = buffer.FirstElement;
        once!(
            first,
            "raw unordered buffer view not on 16 bytes of its host copy"
        );
        return None;
    }
    // A view past its host copy's end is an invalid call (the debug layer
    // removes the device; drivers may fault): none past the end, the rest
    // cut to fit.
    // SAFETY: a live buffer.
    let capacity = unsafe { shadow.buffer.GetDesc() }.Width / element;
    if buffer.FirstElement >= capacity {
        let first = buffer.FirstElement;
        once!(first, capacity, "buffer view past the end of its host copy");
        return None;
    }
    if buffer.FirstElement.saturating_add(u64::from(buffer.NumElements)) > capacity {
        buffer.NumElements = (capacity - buffer.FirstElement) as u32;
    }
    // A view of no elements is an invalid call too (the title's own record
    // count can be 0): null, which reads 0 and drops writes.
    if buffer.NumElements == 0 {
        let first = buffer.FirstElement;
        once!(first, "buffer view of no elements");
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device.CreateUnorderedAccessView(
            &shadow.buffer,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
    Some(Some(shadow.buffer.clone()))
}

/// Adapts the requested view format to a format compatible with host D3D12.
///
/// Returns `true` if the view is permitted, mutating `format` to a valid typed format
/// when necessary (e.g. mapping a typeless view on a typeless texture to its primary
/// concrete format). Returns `false` if the view format belongs to an incompatible
/// family or cannot be created.
pub(crate) fn adapt_view_format(
    own: u32,
    mut format: Option<&mut DXGI_FORMAT>,
    unordered: bool,
) -> bool {
    use crate::resource::{family, first_concrete, is_typeless};
    let typeless = is_typeless(own);
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 && format.is_none() {
        return !typeless;
    }
    // A texture whose console format the host substitutes for render targets
    // or UAVs (packed 16-bit -> R8G8B8A8_UNORM, R9G9B9E5_SHAREDEXP ->
    // R16G16B16A16_FLOAT via `resource::target_format`): views name its host
    // format.
    if view != 0 && view != own && crate::resource::target_format(view) == own {
        if let Some(format) = format {
            *format = DXGI_FORMAT(own as i32);
        }
        return true;
    }
    if view != 0 && family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    if let Some(format) = format {
        if !typeless {
            *format = DXGI_FORMAT(own as i32);
        } else if view == 0 || is_typeless(view) {
            let concrete = first_concrete(own);
            tracing::debug!(
                own,
                view,
                concrete,
                "typeless view on typeless resource remapped to concrete format"
            );
            *format = DXGI_FORMAT(concrete as i32);
        }
    }
    true
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's. A view of a typed texture in another format of its
/// family takes the texture's format.
/// shortcut: such a view (sRGB of UNORM, say) reads without its
/// conversion; cast with castable formats if colors come out wrong.
fn fits(d: &D3D12_RESOURCE_DESC, format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::DEPTH_FORMATS;
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    adapt_view_format(own, format, unordered)
}

/// The unordered access view a null description makes of a 2D texture
/// (twins are): mip 0, every array slice; the format is the caller's.
fn whole_uav(texture: &ID3D12Resource) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    // SAFETY: a live resource.
    let slices = u32::from(unsafe { texture.GetDesc() }.DepthOrArraySize);
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    if slices > 1 {
        desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
            ArraySize: slices,
            ..Default::default()
        };
    }
    desc
}

/// Builds a default SRV description for a typeless host texture using the
/// title's concrete `format`.
fn default_srv_desc(
    texture: &ID3D12Resource,
    format: DXGI_FORMAT,
) -> Option<D3D12_SHADER_RESOURCE_VIEW_DESC> {
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let mut desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        ..Default::default()
    };
    match d.Dimension {
        D3D12_RESOURCE_DIMENSION_TEXTURE1D if d.DepthOrArraySize > 1 => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_SRV {
                MostDetailedMip: 0,
                MipLevels: u32::MAX,
                FirstArraySlice: 0,
                ArraySize: u32::from(d.DepthOrArraySize),
                ResourceMinLODClamp: 0.0,
            };
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE1D => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
            desc.Anonymous.Texture1D = D3D12_TEX1D_SRV {
                MostDetailedMip: 0,
                MipLevels: u32::MAX,
                ResourceMinLODClamp: 0.0,
            };
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE2D if d.SampleDesc.Count > 1 && d.DepthOrArraySize > 1 => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
            desc.Anonymous.Texture2DMSArray = D3D12_TEX2DMS_ARRAY_SRV {
                FirstArraySlice: 0,
                ArraySize: u32::from(d.DepthOrArraySize),
            };
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE2D if d.SampleDesc.Count > 1 => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE2D if d.DepthOrArraySize > 1 => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_SRV {
                MostDetailedMip: 0,
                MipLevels: u32::MAX,
                FirstArraySlice: 0,
                ArraySize: u32::from(d.DepthOrArraySize),
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            };
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE2D => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            desc.Anonymous.Texture2D = D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: u32::MAX,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            };
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE3D => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_SRV {
                MostDetailedMip: 0,
                MipLevels: u32::MAX,
                ResourceMinLODClamp: 0.0,
            };
        }
        _ => return None,
    }
    Some(desc)
}

/// Builds a default UAV description for a typeless host texture using the
/// title's concrete `format`.
fn default_uav_desc(
    texture: &ID3D12Resource,
    format: DXGI_FORMAT,
) -> Option<D3D12_UNORDERED_ACCESS_VIEW_DESC> {
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    if d.SampleDesc.Count > 1 {
        return None;
    }
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: format,
        ..Default::default()
    };
    match d.Dimension {
        D3D12_RESOURCE_DIMENSION_TEXTURE1D if d.DepthOrArraySize > 1 => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                MipSlice: 0,
                FirstArraySlice: 0,
                ArraySize: u32::from(d.DepthOrArraySize),
            };
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE1D => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE2D if d.DepthOrArraySize > 1 => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                MipSlice: 0,
                FirstArraySlice: 0,
                ArraySize: u32::from(d.DepthOrArraySize),
                PlaneSlice: 0,
            };
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE2D => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE3D => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                MipSlice: 0,
                FirstWSlice: 0,
                WSize: u32::MAX,
            };
        }
        _ => return None,
    }
    Some(desc)
}

/// A shader view of a typeless depth texture names the readable aspect.
/// For a 2D view D3D12 also requires the plane index to match that format.
fn texture_srv_desc(
    resource: &D3D12_RESOURCE_DESC,
    words: &[u32; 10],
) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut view = srv_desc(words);
    fixup_depth_srv(resource, &mut view);
    view
}

/// Normalizes a shader view of a typeless depth texture to its readable aspect
/// and matching plane slice.
fn fixup_depth_srv(
    resource: &D3D12_RESOURCE_DESC,
    view: &mut D3D12_SHADER_RESOURCE_VIEW_DESC,
) {
    if !resource
        .Flags
        .contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
    {
        return;
    }
    let (dsv, depth, stencil) = match resource.Format {
        DXGI_FORMAT_R32G8X24_TYPELESS => (
            DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
            DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
            Some(DXGI_FORMAT_X32_TYPELESS_G8X24_UINT),
        ),
        DXGI_FORMAT_R24G8_TYPELESS => (
            DXGI_FORMAT_D24_UNORM_S8_UINT,
            DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
            Some(DXGI_FORMAT_X24_TYPELESS_G8_UINT),
        ),
        DXGI_FORMAT_R32_TYPELESS => (DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32_FLOAT, None),
        DXGI_FORMAT_R16_TYPELESS => (DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_R16_UNORM, None),
        _ => return,
    };
    let plane = if Some(view.Format) == stencil {
        1
    } else if [DXGI_FORMAT_UNKNOWN, resource.Format, dsv, depth].contains(&view.Format) {
        view.Format = depth;
        0
    } else {
        return;
    };
    match view.ViewDimension {
        D3D12_SRV_DIMENSION_TEXTURE2D => view.Anonymous.Texture2D.PlaneSlice = plane,
        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => view.Anonymous.Texture2DArray.PlaneSlice = plane,
        _ => {}
    }
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            if address == 0 || size == 0 {
                return None;
            }
            let capped = size.min(65_536);
            let shadow = crate::shadow::find(address, u64::from(capped))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: capped.next_multiple_of(256),
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            Some(Some(shadow.buffer.clone()))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                // SAFETY: a live host texture.
                let info = unsafe { texture.GetDesc() };
                let mut desc = desc.as_ref().map(|words| texture_srv_desc(&info, words));
                if let Some(view_fmt) = crate::resource::view_format(resource) {
                    let concrete = DXGI_FORMAT(view_fmt as i32);
                    match desc.as_mut() {
                        Some(d) if d.Format == DXGI_FORMAT_UNKNOWN => d.Format = concrete,
                        Some(_) => {}
                        None => {
                            desc = default_srv_desc(texture, concrete);
                            if let Some(d) = desc.as_mut() {
                                fixup_depth_srv(&info, d);
                            }
                        }
                    }
                }
                if !fits(&info, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some((texture, twin_format)) = crate::resource::host_unordered(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            let mut desc = desc.as_ref().map(uav_desc);
            // A depth target's twin is typeless: views name a format of its
            // family; a depth one, the typeless one or none becomes the one
            // with the depth's bits.
            if let Some(format) = twin_format {
                use crate::resource::{DEPTH_FORMATS, family};
                let d = desc.get_or_insert_with(|| whole_uav(texture));
                let own = d.Format.0 as u32;
                if own == 0 || own == family(format) || DEPTH_FORMATS.iter().any(|&(f, _)| f == own)
                {
                    d.Format.0 = format as i32;
                }
            } else if let Some(view_fmt) = crate::resource::view_format(resource) {
                let concrete = DXGI_FORMAT(view_fmt as i32);
                match desc.as_mut() {
                    Some(d) if d.Format == DXGI_FORMAT_UNKNOWN => d.Format = concrete,
                    Some(_) => {}
                    None => desc = default_uav_desc(texture, concrete),
                }
            }
            // SAFETY: a live host texture.
            let info = unsafe { texture.GetDesc() };
            if !fits(&info, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(
                    dimension = resource[0],
                    "placed unordered texture view not mirrored"
                );
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(false, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    Some(View::Shader(texture as usize, desc[..10].try_into().ok()))
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            return unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
        }
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A raw unordered access view of a buffer that drops writes and reads
/// zeros.
fn null_raw_uav(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                NumElements: 1,
                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// An R32 DXGI format compatible with the DXIL component type in `props1`'s
/// low byte (signed integer, unsigned integer, or float).
fn comp_format(props1: u32) -> DXGI_FORMAT {
    match props1 & 0xFF {
        2 | 4 | 6 => DXGI_FORMAT_R32_SINT,
        1 | 3 | 5 | 7 => DXGI_FORMAT_R32_UINT,
        _ => DXGI_FORMAT_R32_FLOAT,
    }
}

/// Maps a DXIL `ResourceProperties` pair (`props`: resource kind in `props.0`
/// low byte, UAV bit `0x1000`, component type or structure stride in
/// `props.1`) to a matching null descriptor description. Returns `None` for
/// kinds with no view descriptor (samplers, unknown kinds, or MSAA UAVs).
pub(crate) fn null_desc_for_props(props: (u32, u32)) -> Option<NullDesc> {
    let kind = props.0 & 0xFF;
    let uav = props.0 & 0x1000 != 0;
    if uav {
        let desc = match kind {
            1 => D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: comp_format(props.1),
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE1D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture1D: D3D12_TEX1D_UAV { MipSlice: 0 },
                },
            },
            2 => D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: comp_format(props.1),
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_UAV {
                        MipSlice: 0,
                        PlaneSlice: 0,
                    },
                },
            },
            4 => D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: comp_format(props.1),
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture3D: D3D12_TEX3D_UAV {
                        MipSlice: 0,
                        FirstWSlice: 0,
                        WSize: 1,
                    },
                },
            },
            6 => D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: comp_format(props.1),
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE1DARRAY,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture1DArray: D3D12_TEX1D_ARRAY_UAV {
                        MipSlice: 0,
                        FirstArraySlice: 0,
                        ArraySize: 1,
                    },
                },
            },
            5 | 7 | 9 => D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: comp_format(props.1),
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                        MipSlice: 0,
                        FirstArraySlice: 0,
                        ArraySize: 1,
                        PlaneSlice: 0,
                    },
                },
            },
            10 => D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: comp_format(props.1),
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: 0,
                        NumElements: 1,
                        StructureByteStride: 0,
                        CounterOffsetInBytes: 0,
                        Flags: D3D12_BUFFER_UAV_FLAG_NONE,
                    },
                },
            },
            11 => D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: 0,
                        NumElements: 1,
                        StructureByteStride: 0,
                        CounterOffsetInBytes: 0,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    },
                },
            },
            12 => D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: 0,
                        NumElements: 1,
                        StructureByteStride: if props.1 > 0 { props.1 } else { 4 },
                        CounterOffsetInBytes: 0,
                        Flags: D3D12_BUFFER_UAV_FLAG_NONE,
                    },
                },
            },
            _ => return None,
        };
        return Some(NullDesc::Uav(desc));
    }
    if kind == 13 {
        return Some(NullDesc::Cbv);
    }
    let srv = |format, dim, anon| D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        ViewDimension: dim,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: anon,
    };
    let desc = match kind {
        1 => srv(
            comp_format(props.1),
            D3D12_SRV_DIMENSION_TEXTURE1D,
            D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture1D: D3D12_TEX1D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        ),
        2 => srv(
            comp_format(props.1),
            D3D12_SRV_DIMENSION_TEXTURE2D,
            D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        ),
        3 => srv(
            comp_format(props.1),
            D3D12_SRV_DIMENSION_TEXTURE2DMS,
            D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DMS: D3D12_TEX2DMS_SRV::default(),
            },
        ),
        4 => srv(
            comp_format(props.1),
            D3D12_SRV_DIMENSION_TEXTURE3D,
            D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture3D: D3D12_TEX3D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        ),
        5 => srv(
            comp_format(props.1),
            D3D12_SRV_DIMENSION_TEXTURECUBE,
            D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                TextureCube: D3D12_TEXCUBE_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        ),
        6 => srv(
            comp_format(props.1),
            D3D12_SRV_DIMENSION_TEXTURE1DARRAY,
            D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture1DArray: D3D12_TEX1D_ARRAY_SRV {
                    MipLevels: 1,
                    ArraySize: 1,
                    ..Default::default()
                },
            },
        ),
        7 => srv(
            comp_format(props.1),
            D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                    MipLevels: 1,
                    ArraySize: 1,
                    ..Default::default()
                },
            },
        ),
        8 => srv(
            comp_format(props.1),
            D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY,
            D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DMSArray: D3D12_TEX2DMS_ARRAY_SRV {
                    FirstArraySlice: 0,
                    ArraySize: 1,
                },
            },
        ),
        9 => srv(
            comp_format(props.1),
            D3D12_SRV_DIMENSION_TEXTURECUBEARRAY,
            D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                TextureCubeArray: D3D12_TEXCUBE_ARRAY_SRV {
                    MipLevels: 1,
                    NumCubes: 1,
                    ..Default::default()
                },
            },
        ),
        10 => srv(
            comp_format(props.1),
            D3D12_SRV_DIMENSION_BUFFER,
            D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    NumElements: 1,
                    Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                    ..Default::default()
                },
            },
        ),
        11 => srv(
            DXGI_FORMAT_R32_TYPELESS,
            D3D12_SRV_DIMENSION_BUFFER,
            D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    NumElements: 1,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                    ..Default::default()
                },
            },
        ),
        12 => srv(
            DXGI_FORMAT_UNKNOWN,
            D3D12_SRV_DIMENSION_BUFFER,
            D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    NumElements: 1,
                    StructureByteStride: if props.1 > 0 { props.1 } else { 4 },
                    Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                    ..Default::default()
                },
            },
        ),
        _ => return None,
    };
    Some(NullDesc::Srv(desc))
}

/// Writes `desc` as a null descriptor at host heap slot `at`.
fn write_null(desc: &NullDesc, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    match desc {
        NullDesc::Srv(d) => {
            // SAFETY: a slot of our heap; a null resource is allowed with a desc.
            unsafe {
                host.device
                    .CreateShaderResourceView(None::<&ID3D12Resource>, Some(d), at)
            };
        }
        NullDesc::Uav(d) => {
            // SAFETY: a slot of our heap; a null resource is allowed with a desc.
            unsafe {
                host.device.CreateUnorderedAccessView(
                    None::<&ID3D12Resource>,
                    None::<&ID3D12Resource>,
                    Some(d),
                    at,
                )
            };
        }
        NullDesc::Cbv => {
            let d = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null CBV.
            unsafe { host.device.CreateConstantBufferView(Some(&d), at) };
        }
    }
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all get this null 2D view unless a
/// shader annotates the slot (`table_views`, then a null of that kind);
/// untyped null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they alone are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{
        HeapDesc, copy_simple, create_cbv, create_heap, create_placed_srv, create_srv, create_uav,
        heap,
    };
    use crate::object::{Iid, S_OK, This, guid, slot};
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::core::s;

    const HEAP_IID: Iid = guid(
        0x8efb471d,
        0x616c,
        0x4f49,
        [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
    );
    const RESOURCE_IID: Iid = guid(
        0x696442be,
        0xa72e,
        0x4059,
        [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
    );

    #[repr(C)]
    struct ExtendedHeapDesc {
        desc: HeapDesc,
        addr: u64,
        size: u64,
    }

    fn make_shader_heap(count: u32) -> (This, u64) {
        let ext = ExtendedHeapDesc {
            desc: HeapDesc {
                kind: 0,
                count,
                flags: 1,
                nodes: 0,
            },
            addr: 0,
            size: 0,
        };
        let mut out = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &ext.desc, &HEAP_IID, &mut out),
            S_OK
        );
        let start = slot::<extern "system" fn(This) -> u64>(out, 0x48)(out);
        (out, start)
    }

    fn make_title_buffer(address: u64, size: u64, uav: bool) -> This {
        let words: [u32; 14] = [
            1, // BUFFER
            0,
            0,
            0,
            size as u32,
            (size >> 32) as u32,
            1,
            0x0001_0001,
            0,
            1,
            0,
            1, // ROW_MAJOR
            if uav { 4 } else { 0 },
            0,
        ];
        let mut out = std::ptr::null();
        assert_eq!(
            crate::resource::create_at_address(
                std::ptr::null(),
                address,
                words.as_ptr().cast(),
                0,
                std::ptr::null(),
                &RESOURCE_IID,
                &mut out,
            ),
            S_OK
        );
        out
    }

    fn compile_probe_cs(device: &ID3D12Device) -> (ID3D12RootSignature, ID3D12PipelineState) {
        const PROBE_HLSL: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=1)), \
            DescriptorTable(CBV(b0, numDescriptors=1)), \
            DescriptorTable(SRV(t1, numDescriptors=1)), \
            DescriptorTable(UAV(u0, numDescriptors=1)), \
            DescriptorTable(SRV(t2, numDescriptors=1)), \
            DescriptorTable(CBV(b1, numDescriptors=1))"
ByteAddressBuffer srv_buf : register(t0);
cbuffer Cbv16 : register(b0) { uint4 cb_vals; };
Texture2D<float4> placed_tex : register(t1);
RWByteAddressBuffer uav_buf : register(u0);
ByteAddressBuffer copied_srv : register(t2);
cbuffer NullCbv : register(b1) { uint4 null_vals; };

[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint v_srv = srv_buf.Load(0);
    uint v_cbv = cb_vals.x + cb_vals.y + cb_vals.z + cb_vals.w;
    float4 px = placed_tex.Load(int3(0, 0, 0));
    uint v_tex = (uint(round(px.x * 255.0)) << 0)
               | (uint(round(px.y * 255.0)) << 8)
               | (uint(round(px.z * 255.0)) << 16)
               | (uint(round(px.w * 255.0)) << 24);
    uint v_copied = copied_srv.Load(0);
    uint v_null = null_vals.x;
    uav_buf.Store(0, v_srv);
    uav_buf.Store(4, v_cbv);
    uav_buf.Store(8, v_tex);
    uav_buf.Store(12, v_copied);
    uav_buf.Store(16, v_null);
}
"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: static source and entry point strings; out pointer ours.
        unsafe {
            D3DCompile(
                PROBE_HLSL.as_ptr().cast(),
                PROBE_HLSL.len(),
                s!("bindless_probe"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
            .unwrap();
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).unwrap();
            let pso = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        }
    }

    /// The CPU-side block used to initialize null samplers must fit the
    /// same portable sampler limit as the shader-visible heap.
    #[test]
    fn portable_sampler_null_fill_fits_the_host_limit() {
        if host().is_none() {
            return; // no host GPU
        }
        let heap = HostHeap::new(true, D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE)
            .expect("portable sampler heap must initialize");
        assert!(
            heap.fill_null(
                D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE,
            )
            .is_ok()
        );
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A buffer view of no elements, which the host refuses (removing the
    /// device), makes a null slot; one of an element is made as usual.
    #[test]
    fn buffer_views_of_no_elements_are_null() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        // A title buffer the view's memory lies in (its copy is made on the
        // host; title memory is not read).
        let (address, len) = (0x7e00_0000_0000_u64, 0x1000_u64);
        crate::shadow::register(usize::MAX - 1, address, len);
        let mut resource = [0u32; 14];
        resource[0] = 1; // a buffer
        resource[4] = len as u32;
        let placed = |unordered: bool, elements: u32| {
            let mut desc = [0u32; 16];
            desc[1] = 1; // buffer dimension
            if unordered {
                // NumElements, StructureByteStride
                (desc[4], desc[5]) = (elements, 16);
            } else {
                desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                (desc[6], desc[7]) = (elements, 16);
            }
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(unordered, resource, desc)
        };
        let kept = |index: u32| heap.kept.lock().unwrap()[index as usize].is_some();
        for unordered in [false, true] {
            heap.make_at(0, &placed(unordered, 1), None);
            assert!(kept(0), "a view of an element (unordered: {unordered})");
            heap.make_at(1, &placed(unordered, 0), None);
            assert!(!kept(1), "a view of no elements (unordered: {unordered})");
            // SAFETY: a plain device query.
            let removed = unsafe { host.device.GetDeviceRemovedReason() };
            assert!(removed.is_ok(), "device removed: {removed:?} (unordered: {unordered})");
        }
        crate::shadow::unregister(usize::MAX - 1, address);
    }

    #[test]
    fn depth_srv_uses_the_requested_aspect() {
        for (own, request, expected, plane) in [
            (19, 20, 21, 0),
            (19, 22, 22, 1),
            (44, 45, 46, 0),
            (44, 47, 47, 1),
            (44, 120, 47, 1),
            (44, 119, 46, 0),
            (39, 40, 41, 0),
            (53, 55, 56, 0),
            (19, 0, 21, 0),
            (44, 44, 46, 0),
        ] {
            let resource = D3D12_RESOURCE_DESC {
                Format: DXGI_FORMAT(own),
                Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                ..Default::default()
            };
            for dim in [
                D3D12_SRV_DIMENSION_TEXTURE2D,
                D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            ] {
                let mut words = [0; 10];
                words[0] = request;
                words[1] = dim.0 as u32;
                words[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                words[4] = 2; // MostDetailedMip
                words[5] = 3; // MipLevels
                if dim == D3D12_SRV_DIMENSION_TEXTURE2DARRAY {
                    words[6] = 1; // FirstArraySlice
                    words[7] = 2; // ArraySize
                }
                let plane_word = if dim == D3D12_SRV_DIMENSION_TEXTURE2DARRAY {
                    8
                } else {
                    6
                };
                words[plane_word] = 1 - plane;
                let view = texture_srv_desc(&resource, &words);
                assert_eq!(view.Format.0, expected);
                // SAFETY: read the member selected by the input view dimension.
                unsafe {
                    if dim == D3D12_SRV_DIMENSION_TEXTURE2DARRAY {
                        let a = view.Anonymous.Texture2DArray;
                        assert_eq!(
                            (
                                a.MostDetailedMip,
                                a.MipLevels,
                                a.FirstArraySlice,
                                a.ArraySize,
                                a.PlaneSlice
                            ),
                            (2, 3, 1, 2, plane)
                        );
                    } else {
                        let a = view.Anonymous.Texture2D;
                        assert_eq!(
                            (a.MostDetailedMip, a.MipLevels, a.PlaneSlice),
                            (2, 3, plane)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn free_space_counts_separate_runs_at_the_heap_ends() {
        assert_eq!(free_space(&[true; 8]), (0, 0));
        assert_eq!(free_space(&[false; 8]), (8, 8));
        let fragmented = [false, false, true, false, true, false, false, false];
        assert_eq!(free_space(&fragmented), (6, 3));
    }

    #[test]
    fn free_table_chooses_the_first_contiguous_fit_in_cursor_order() {
        for mask in 0u16..256 {
            let taken: [bool; 8] = std::array::from_fn(|slot| mask & (1 << slot) != 0);
            for cursor in 0..taken.len() {
                for n in 1..=taken.len() {
                    let expected = (cursor..taken.len()).chain(0..cursor).find(|&start| {
                        taken
                            .get(start..start + n)
                            .is_some_and(|table| table.iter().all(|&held| !held))
                    });
                    assert_eq!(
                        free_table(&taken, cursor, n),
                        expected,
                        "{mask:#x} {cursor} {n}"
                    );
                }
            }
        }
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.spare - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.spare - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.spare - 40));
    }

    /// A structured view whose start in its host copy is not on an element
    /// becomes the raw view of the same bytes, where those start 16-byte
    /// aligned.
    #[test]
    fn structured_views_off_an_element_are_raw_views_of_the_same_bytes() {
        // (offset in the copy, first element, elements, stride)
        for (offset, first, count, stride) in [
            (16, 0, 4, 32),
            (48, 3, 7, 32),
            (4096 + 16, 1, 1, 48),
            (32, 0, 9, 12),
        ] {
            let (dword, dwords) = raw_window(offset, first, count, stride).unwrap();
            assert_eq!(dword * 4, offset + first * u64::from(stride));
            assert_eq!(u64::from(dwords) * 4, u64::from(count) * u64::from(stride));
        }
        // Not 16-byte aligned, strides raw views cannot cover, too long.
        assert_eq!(raw_window(8, 0, 4, 32), None);
        assert_eq!(raw_window(16, 1, 4, 12), None);
        assert_eq!(raw_window(16, 0, 4, 6), None);
        assert_eq!(raw_window(16, 0, 4, 0), None);
        assert_eq!(raw_window(16, 0, u32::MAX, 32), None);
    }

    /// A structured view (32-byte elements) 16 bytes into its host copy,
    /// read and another written by a shader: it reads the title's bytes from
    /// there, and what it writes lands at its own bytes of title memory.
    #[test]
    fn structured_views_off_an_element_read_and_write_their_bytes() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::s;
        let Some(host) = host() else {
            return; // no host GPU
        };
        const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0), UAV(u0, numDescriptors = 2))"
ByteAddressBuffer view : register(t0);
RWByteAddressBuffer window : register(u0);
RWByteAddressBuffer seen : register(u1);
[RootSignature(RS)]
[numthreads(32, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    seen.Store(id.x * 4, view.Load(id.x * 4));
    window.Store(id.x * 4, 0xc0de0000u | id.x);
}
"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; out pointer ours.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("views"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
        }
        .unwrap();
        let code = code.unwrap();
        let device = &host.device;
        // SAFETY: the blob's own bytes, with the root signature inside;
        // plain object creation.
        let (root, pso, views, step) = unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            let views: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 3,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .unwrap();
            let step =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            (root, pso, views, step as usize)
        };
        // SAFETY: a live heap.
        let first = unsafe { views.GetCPUDescriptorHandleForHeapStart() }.ptr;
        let at = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: first + i * step,
        };

        // A title buffer of 1 KiB, numbered dwords; four 32-byte elements
        // from byte 16 are read, four from byte 272 written.
        let before: Vec<u32> = (0..256).map(|i| 0x7171_0000 | i).collect();
        let mut title = before.clone();
        let start = title.as_mut_ptr() as u64;
        crate::shadow::register(start as usize, start, 1024);
        let shadow = crate::shadow::find(start, 1024).unwrap();
        let structured = D3D12_BUFFER_SRV {
            NumElements: 4,
            StructureByteStride: 32,
            ..Default::default()
        };
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: structured },
        };
        let uav = |buffer| D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: buffer },
        };
        let written = D3D12_BUFFER_UAV {
            NumElements: 4,
            StructureByteStride: 32,
            ..Default::default()
        };
        // Views the host refuses are left null, as `HostHeap::mirror` leaves them.
        let mut placed = [0; 14];
        placed[0] = 1; // a buffer
        if buffer_srv(start + 16, 128, srv, at(0)).is_none() {
            null_for(&View::Placed(false, placed, [0; 16]), at(0));
        }
        if buffer_uav(start + 272, 128, uav(written), at(1)).is_none() {
            null_for(&View::Placed(true, placed, [0; 16]), at(1));
        }
        // What the shader read, as a raw view of 32 dwords of our own.
        let (seen, out) = host.shared_buffer(128).unwrap();
        let mut raw = uav(D3D12_BUFFER_UAV {
            NumElements: 32,
            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
            ..Default::default()
        });
        raw.Format = DXGI_FORMAT_R32_TYPELESS;
        // SAFETY: a live buffer and a slot of our heap.
        unsafe {
            device.CreateUnorderedAccessView(&seen, None::<&ID3D12Resource>, Some(&raw), at(2))
        };

        let list = host.new_list().unwrap();
        shadow.refresh(&list, start, 1024);
        shadow.gpu_writes(&list, start + 272, 128);
        // SAFETY: an open list; our heap, root signature and pipeline.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(views.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list
                .SetComputeRootDescriptorTable(0, views.GetGPUDescriptorHandleForHeapStart());
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        crate::shadow::unregister(start as usize, start);

        // SAFETY: 32 dwords the GPU wrote, read after it finished.
        let seen = unsafe { std::slice::from_raw_parts(out.cast::<u32>(), 32) };
        assert_eq!(seen, &before[4..36], "what the read view gave");
        let marks: Vec<u32> = (0..32).map(|i| 0xc0de_0000 | i).collect();
        assert_eq!(&title[68..100], &marks[..], "what the written view left");
        assert_eq!(
            (&title[..68], &title[100..]),
            (&before[..68], &before[100..])
        );
    }

    /// A sampler heap past the host's limit is capped, not lost, and keeps
    /// dynamic slots for the tables reaching past what it holds.
    #[test]
    fn large_sampler_heap_is_capped() {
        let Some(heap) = HostHeap::new(true, 4 * SAMPLERS) else {
            return; // no host GPU
        };
        let count = SAMPLERS - SAMPLER_DYNAMIC;
        assert_eq!(heap.count(), count);
        assert_eq!(heap.dynamic(SAMPLER_DYNAMIC + 1), None);
        let a = heap.dynamic(SAMPLER_DYNAMIC).unwrap();
        assert_eq!(a, count);
        assert_eq!(heap.dynamic(1), None);
        heap.release(a, 8);
        assert_eq!(heap.dynamic(8), Some(count));
    }

    /// A sampler heap within the limit has a slot per title slot and no
    /// dynamic ones.
    #[test]
    fn small_sampler_heap_has_no_dynamic_slots() {
        let Some(heap) = HostHeap::new(true, 16) else {
            return; // no host GPU
        };
        assert_eq!(heap.count(), 16);
        assert_eq!(heap.dynamic(1), None);
    }

    /// sRGB-castable textures are backed by typeless host resources so explicit
    /// `_SRGB` SRVs and RTVs keep their sRGB format, UAVs map cleanly to linear
    /// UNORM, and null-description views still succeed.
    #[test]
    fn srgb_views_preserved_on_srgb_castable_textures() {
        use crate::object::{This, slot};

        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let resource = crate::resource::test_committed_texture(
            4,
            4,
            1,
            1,
            DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32,
            0x1 | 0x4, // ALLOW_RENDER_TARGET | ALLOW_UNORDERED_ACCESS
        );
        let tex = crate::resource::host_resource(resource).expect("host texture");
        // SAFETY: live host texture.
        let tex_desc = unsafe { tex.GetDesc() };
        assert_eq!(tex_desc.Format, DXGI_FORMAT_R8G8B8A8_TYPELESS);

        // Explicit `_SRGB` SRV format is preserved rather than forced linear.
        let mut srgb = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        assert!(fits(&tex_desc, Some(&mut srgb), false));
        assert_eq!(srgb, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);

        // UAV maps `_SRGB` to its linear UNORM twin as D3D12 requires.
        let mut uav_fmt = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        assert!(fits(&tex_desc, Some(&mut uav_fmt), true));
        assert_eq!(uav_fmt, DXGI_FORMAT_R8G8B8A8_UNORM);

        // Build real host descriptors for explicit `_SRGB` SRV, null-desc SRV,
        // UNKNOWN-format SRV, and null-desc UAV.
        let mut srgb_srv_words = [0u32; 10];
        srgb_srv_words[0] = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB.0 as u32;
        srgb_srv_words[1] = D3D12_SRV_DIMENSION_TEXTURE2D.0 as u32;
        srgb_srv_words[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        // MipLevels sits at word 5 (after 4-byte union alignment pad at word 3
        // and MostDetailedMip at word 4).
        srgb_srv_words[5] = 1;
        assert!(
            make(
                &View::Shader(resource as usize, Some(srgb_srv_words)),
                heap.cpu(0)
            )
            .is_some()
        );

        let mut unknown_srv_words = srgb_srv_words;
        unknown_srv_words[0] = 0;
        assert!(
            make(
                &View::Shader(resource as usize, Some(unknown_srv_words)),
                heap.cpu(1)
            )
            .is_some()
        );
        assert!(make(&View::Shader(resource as usize, None), heap.cpu(2)).is_some());
        assert!(make(&View::Unordered(resource as usize, 0, None), heap.cpu(3)).is_some());

        // RTV records preserve explicit `_SRGB` and fill `None`/UNKNOWN with concrete format.
        let mut rtv_slot = [0u8; 112];
        let srgb_rtv = [
            DXGI_FORMAT_R8G8B8A8_UNORM_SRGB.0 as u32,
            D3D12_RTV_DIMENSION_TEXTURE2D.0 as u32,
            0,
            0,
            0,
            0,
        ];
        crate::descriptor::create_rtv(
            std::ptr::null(),
            resource,
            &srgb_rtv,
            rtv_slot.as_mut_ptr() as u64,
        );
        let rec = crate::descriptor::rtv_record(rtv_slot.as_ptr() as u64).unwrap();
        assert_eq!(rec.desc().unwrap().Format, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);

        crate::descriptor::create_rtv(
            std::ptr::null(),
            resource,
            std::ptr::null(),
            rtv_slot.as_mut_ptr() as u64,
        );
        let rec = crate::descriptor::rtv_record(rtv_slot.as_ptr() as u64).unwrap();
        let rtv_desc = rec.desc().unwrap();
        assert_eq!(rtv_desc.Format, DXGI_FORMAT_R8G8B8A8_UNORM);
        assert_eq!(rtv_desc.ViewDimension, D3D12_RTV_DIMENSION_TEXTURE2D);
        let host = host().unwrap();
        let _ = host.rtv(Some(tex), Some(&rtv_desc));

        // Null-desc RTV on 2D-array and MSAA sRGB-castable textures must
        // synthesize the matching RTV dimension (not hardcoded TEXTURE2D).
        let array_res = crate::resource::test_committed_texture(
            4,
            4,
            2,
            1,
            DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32,
            0x1, // ALLOW_RENDER_TARGET
        );
        let array_tex = crate::resource::host_resource(array_res).expect("array texture");
        crate::descriptor::create_rtv(
            std::ptr::null(),
            array_res,
            std::ptr::null(),
            rtv_slot.as_mut_ptr() as u64,
        );
        let array_rec = crate::descriptor::rtv_record(rtv_slot.as_ptr() as u64).unwrap();
        let array_rtv = array_rec.desc().unwrap();
        assert_eq!(array_rtv.Format, DXGI_FORMAT_R8G8B8A8_UNORM);
        assert_eq!(array_rtv.ViewDimension, D3D12_RTV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: reading `Texture2DArray` union arm just asserted above.
        assert_eq!(unsafe { array_rtv.Anonymous.Texture2DArray.ArraySize }, 2);
        let _ = host.rtv(Some(array_tex), Some(&array_rtv));

        let msaa_res = crate::resource::test_committed_texture(
            4,
            4,
            1,
            4,
            DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32,
            0x1, // ALLOW_RENDER_TARGET
        );
        let msaa_tex = crate::resource::host_resource(msaa_res).expect("msaa texture");
        crate::descriptor::create_rtv(
            std::ptr::null(),
            msaa_res,
            std::ptr::null(),
            rtv_slot.as_mut_ptr() as u64,
        );
        let msaa_rec = crate::descriptor::rtv_record(rtv_slot.as_ptr() as u64).unwrap();
        let msaa_rtv = msaa_rec.desc().unwrap();
        assert_eq!(msaa_rtv.Format, DXGI_FORMAT_R8G8B8A8_UNORM);
        assert_eq!(msaa_rtv.ViewDimension, D3D12_RTV_DIMENSION_TEXTURE2DMS);
        let _ = host.rtv(Some(msaa_tex), Some(&msaa_rtv));

        // SAFETY: plain device health check after creating all views.
        let removed = unsafe { host.device.GetDeviceRemovedReason() };
        assert!(removed.is_ok());
        assert_eq!(
            slot::<extern "system" fn(This) -> u32>(msaa_res, 0x10)(msaa_res),
            0
        );
        assert_eq!(
            slot::<extern "system" fn(This) -> u32>(array_res, 0x10)(array_res),
            0
        );
        assert_eq!(
            slot::<extern "system" fn(This) -> u32>(resource, 0x10)(resource),
            0
        );
    }

    /// End-to-end GPU execution test proving that buffer SRVs, 16-byte-aligned
    /// CBVs, copied buffer SRVs, `View::Placed` streamed textures, and buffer
    /// UAVs in a shader-visible heap materialize non-null host descriptors,
    /// refresh modified title memory across submissions, copy bindless UAV
    /// writes back to title memory, and map null/invalid addresses safely to 0.
    #[test]
    fn bindless_slots_materialize_refresh_and_copy_back_on_gpu() {
        let Some(host) = host() else {
            return;
        };
        let (title_heap, base) = make_shader_heap(16);
        let slot_handle = |i: u64| base + i * 32;

        // Backing title memory (64 KiB aligned).
        let mem = crate::resource::Memory::zeroed(256 << 10, 64 << 10).unwrap();
        let mem_base = mem.address();
        let srv_addr = mem_base;
        let cbv_16b_addr = mem_base + 0x30; // 16B-aligned, NOT 256B-aligned!
        let copy_srv_addr = mem_base + 0x1000;
        let uav_addr = mem_base + 0x2000;
        let placed_tex_addr = mem_base + (64 << 10);

        let srv_res = make_title_buffer(srv_addr, 256, false);
        let copy_srv_res = make_title_buffer(copy_srv_addr, 256, false);
        let uav_res = make_title_buffer(uav_addr, 256, true);

        // Initial CPU data in title memory.
        // SAFETY: offsets lie within the 256 KiB allocation `mem`.
        unsafe {
            (srv_addr as *mut u32).write(13);
            (cbv_16b_addr as *mut [u32; 4]).write([10, 20, 30, 40]);
            (copy_srv_addr as *mut u32).write(0xCAFE_1234);
            (uav_addr as *mut [u32; 8]).write([0; 8]);
            // First pixel of linear 4x4 RGBA8 placed texture at `placed_tex_addr`.
            (placed_tex_addr as *mut [u8; 4]).write([0x12, 0x34, 0x56, 0x78]);
        }

        // Raw ByteAddressBuffer SRV/UAV descriptions in host D3D12 layout.
        let srv_desc_host = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: 16,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        let uav_desc_host = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: 0,
                    NumElements: 16,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        };
        // SAFETY: both D3D12 view descriptions are 40-byte POD values (`[u32; 10]`).
        let (raw_srv_words, raw_uav_words): ([u32; 10], [u32; 10]) = unsafe {
            (
                std::ptr::from_ref(&srv_desc_host)
                    .cast::<[u32; 10]>()
                    .read(),
                std::ptr::from_ref(&uav_desc_host)
                    .cast::<[u32; 10]>()
                    .read(),
            )
        };

        // Slot 0: Buffer SRV.
        create_srv(std::ptr::null(), srv_res, &raw_srv_words, slot_handle(0));
        // Slot 1: 16B-aligned CBV (size 64 bytes, address % 256 == 0x30).
        create_cbv(
            std::ptr::null(),
            &[cbv_16b_addr as u32, (cbv_16b_addr >> 32) as u32, 64, 0],
            slot_handle(1),
        );
        // Slot 2: Placed 4x4 R8G8B8A8_UNORM linear texture (`layout = 0x100`).
        let placed_res_words: [u32; 14] = [
            3, // TEXTURE2D
            0,
            0,
            0,
            4,
            0,
            4,
            0x0001_0001,
            DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32,
            1,
            0,
            0x100, // console LINEAR layout
            0,
            0,
        ];
        let tex_srv_host = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        let mut placed_view_words = [0u32; 16];
        // SAFETY: `D3D12_SHADER_RESOURCE_VIEW_DESC` is 40 bytes (`[u32; 10]`).
        placed_view_words[..10].copy_from_slice(&unsafe {
            std::ptr::from_ref(&tex_srv_host).cast::<[u32; 10]>().read()
        });
        placed_view_words[10] = placed_tex_addr as u32;
        placed_view_words[11] = (placed_tex_addr >> 32) as u32;
        create_placed_srv(
            std::ptr::null(),
            &placed_res_words,
            &placed_view_words,
            slot_handle(2),
        );
        // Slot 3: Buffer UAV.
        create_uav(
            std::ptr::null(),
            uav_res,
            std::ptr::null(),
            &raw_uav_words,
            slot_handle(3),
        );
        // Slot 6 -> copied to Slot 4 via `copy_simple`.
        create_srv(
            std::ptr::null(),
            copy_srv_res,
            &raw_srv_words,
            slot_handle(6),
        );
        copy_simple(std::ptr::null(), 1, slot_handle(4), slot_handle(6), 0);
        // Slot 5: Unmapped/null CBV (address = 0) -> must read 0 safely.
        create_cbv(std::ptr::null(), &[0, 0, 256, 0], slot_handle(5));

        let host_heap = heap(title_heap).host_arc().unwrap();
        // Before bindless preparation, `mirror(..., false)` deferred all
        // `View::is_buffer()` slots (`made[idx] == 0`).
        for idx in 0..=5 {
            assert_eq!(
                host_heap.made.lock().unwrap()[idx],
                SlotState::Null,
                "slot {idx} starts deferred"
            );
        }

        let (root, pso) = compile_probe_cs(&host.device);
        let list = host.new_list().unwrap();

        let record_probe = |list: &crate::host::HostList| {
            host_heap.prepare_bindless_record(list);
            list.defer_bindless(&host_heap, true);
            // SAFETY: open command list, shader-visible heap, and matching root/PSO.
            unsafe {
                let gpu_start = host_heap.heap.GetGPUDescriptorHandleForHeapStart().ptr;
                let gpu_at = |slot_idx: u32| D3D12_GPU_DESCRIPTOR_HANDLE {
                    ptr: gpu_start + u64::from(slot_idx) * u64::from(host_heap.step),
                };
                list.list
                    .SetDescriptorHeaps(&[Some(host_heap.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                list.list.SetComputeRootDescriptorTable(0, gpu_at(0));
                list.list.SetComputeRootDescriptorTable(1, gpu_at(1));
                list.list.SetComputeRootDescriptorTable(2, gpu_at(2));
                list.list.SetComputeRootDescriptorTable(3, gpu_at(3));
                list.list.SetComputeRootDescriptorTable(4, gpu_at(4));
                list.list.SetComputeRootDescriptorTable(5, gpu_at(5));
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
        };

        record_probe(&list);
        host.execute(&[&list]);
        list.reset();

        // SAFETY: GPU finished and `Host::execute` copied UAV writes back to `uav_addr`.
        let out1 = unsafe { (uav_addr as *const [u32; 5]).read_volatile() };
        assert_eq!(
            out1[0], 13,
            "bindless buffer SRV materialized and refreshed"
        );
        assert_eq!(
            out1[1], 100,
            "16B-aligned bindless CBV materialized via bounce"
        );
        assert_eq!(
            out1[2], 0x7856_3412,
            "placed streamed texture uploaded and sampled"
        );
        assert_eq!(out1[3], 0xCAFE_1234, "copied bindless SRV materialized");
        assert_eq!(out1[4], 0, "null-address CBV safely reads zero");

        // Modify CPU memory for both the SRV buffer and the 16B-aligned CBV,
        // overwrite slot 5 with a `u64::MAX` garbage address CBV, and release
        // `copy_srv_res` so slot 4 goes stale -> both slots 4 and 5 read 0!
        create_cbv(
            std::ptr::null(),
            &[u32::MAX, u32::MAX, 256, 0],
            slot_handle(5),
        );
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(copy_srv_res), 0);
        // SAFETY: offsets lie within `mem`.
        unsafe {
            (srv_addr as *mut u32).write_volatile(500);
            (cbv_16b_addr as *mut [u32; 4]).write_volatile([100, 200, 300, 400]);
        }

        // Released views are looked for once a frame.
        crate::queue::pass_frame();
        record_probe(&list);
        host.execute(&[&list]);
        list.reset();

        // SAFETY: GPU finished and copied back.
        let out2 = unsafe { (uav_addr as *const [u32; 5]).read_volatile() };
        assert_eq!(
            out2[0], 500,
            "dirty SRV title memory refreshed on second submit"
        );
        assert_eq!(
            out2[1], 1000,
            "dirty 16B-aligned CBV refreshed on second submit"
        );
        assert_eq!(out2[3], 0, "released resource slot pruned and reads zero");
        assert_eq!(
            out2[4], 0,
            "overflowing u64::MAX garbage address safely reads zero"
        );

        assert_eq!(release(srv_res), 0);
        assert_eq!(release(uav_res), 0);
        assert_eq!(release(title_heap), 0);
    }

    /// Submissions with nothing changed reuse a heap's bindless plan (no
    /// slot is made or range found again); writing a slot makes a new one,
    /// whose reads of one host copy are merged into one range.
    #[test]
    fn bindless_plan_is_remade_only_when_slots_change() {
        if host().is_none() {
            return;
        }
        let (title_heap, base) = make_shader_heap(8);
        let mem = crate::resource::Memory::zeroed(256 << 10, 64 << 10).unwrap();
        let cbv = |at: u64, slot: u64| {
            create_cbv(
                std::ptr::null(),
                &[at as u32, (at >> 32) as u32, 256, 0],
                base + slot * 32,
            );
        };
        cbv(mem.address(), 0);
        let host_heap = heap(title_heap).host_arc().unwrap();
        let first = host_heap.bindless_plan().unwrap();
        assert_eq!(first.reads.len(), 1);
        assert_eq!(first.reads[0].2, 256);
        assert!(first.writes.is_empty(), "constant buffers are not written");
        // Other tests may release resources meanwhile, which remakes plans.
        let reused = (0..100).any(|_| {
            let a = host_heap.bindless_plan().unwrap();
            Arc::ptr_eq(&a, &host_heap.bindless_plan().unwrap())
        });
        assert!(reused, "unchanged slots reuse the plan");

        // Slot 0 marked unmade: were it made again, it would be made anew.
        host_heap.made.lock().unwrap()[0] = SlotState::Null;
        cbv(mem.address() + 0x1000, 1);
        let second = host_heap.bindless_plan().unwrap();
        assert!(
            !Arc::ptr_eq(&first, &second),
            "a written slot remakes the plan"
        );
        assert_eq!(second.reads.len(), 1, "nearby reads merge");
        assert_eq!(second.reads[0].1, mem.address() as usize);
        assert_eq!(second.reads[0].2, 0x1100);
        assert_eq!(
            host_heap.made.lock().unwrap()[0],
            SlotState::Null,
            "only the written slot is made again"
        );
        assert_ne!(host_heap.made.lock().unwrap()[1], SlotState::Null);

        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(title_heap), 0);
    }

    /// Placed textures reached through slots upload the first time, then
    /// only after [`RECHECK_FRAMES`] (a few per frame), or when another
    /// texture takes the object's address.
    #[test]
    fn bindless_texture_uploads_are_throttled() {
        let Some(host) = host() else {
            return;
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let (a, _) = host.shared_buffer(1 << 16).unwrap();
        let (b, _) = host.shared_buffer(1 << 16).unwrap();
        let tex = 0x1000 as crate::object::This;
        assert!(heap.upload_due(tex, &a, 10), "first reach uploads");
        assert!(!heap.upload_due(tex, &a, 10));
        assert!(!heap.upload_due(tex, &a, 10 + RECHECK_FRAMES - 1));
        assert!(heap.upload_due(tex, &b, 11), "a new texture at the address");
        assert!(heap.upload_due(tex, &b, 11 + RECHECK_FRAMES), "looked at again");
        // At most `RECHECKS` known textures per frame.
        let frame = 1000;
        let many: Vec<_> = (0..2 * RECHECKS as usize)
            .map(|i| {
                let t = (0x2000 + i) as crate::object::This;
                assert!(heap.upload_due(t, &a, 0));
                t
            })
            .collect();
        let again = many.iter().filter(|&&t| heap.upload_due(t, &a, frame)).count();
        assert_eq!(again, RECHECKS as usize);
    }

    /// A list queues each heap once, keeping whether any draw may write.
    #[test]
    fn bindless_heaps_queue_once_per_list() {
        let Some(host) = host() else {
            return;
        };
        let (title_heap, _) = make_shader_heap(8);
        let host_heap = heap(title_heap).host_arc().unwrap();
        let list = host.new_list().unwrap();
        list.defer_bindless(&host_heap, false);
        list.defer_bindless(&host_heap, true);
        list.defer_bindless(&host_heap, false);
        assert_eq!(list.bindless_queued(), vec![true]);
        list.close();
        host.execute(&[&list]);
        list.reset();
        assert!(list.bindless_queued().is_empty());

        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(title_heap), 0);
    }

    /// Verifies `PLACED_TEXTURES` entries overlapping a remapped virtual range
    /// are evicted on `title_memory_remapped` while non-overlapping entries stay.
    #[test]
    fn placed_textures_evict_on_title_memory_remapped() {
        if host().is_none() {
            return;
        }
        let mem = crate::resource::Memory::zeroed(256 << 10, 64 << 10).unwrap();
        let addr_a = mem.address();
        let addr_b = mem.address() + (128 << 10);
        let desc: [u32; 14] = [
            3, // TEXTURE2D
            0,
            0,
            0,
            16,
            0,
            16,
            0x0001_0001,
            DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32,
            1,
            0,
            0x100, // LINEAR
            0,
            0,
        ];

        assert!(crate::resource::texture_at(addr_a, &desc).is_some());
        assert!(crate::resource::texture_at(addr_b, &desc).is_some());
        assert!(crate::resource::has_placed_texture(addr_a, &desc));
        assert!(crate::resource::has_placed_texture(addr_b, &desc));

        let epoch_before = crate::resource::placed_epoch();
        // Remap 256 bytes inside `addr_a`'s footprint (not even starting at `addr_a`!).
        crate::watch::title_memory_remapped((addr_a + 128) as usize, 256);

        assert!(
            !crate::resource::has_placed_texture(addr_a, &desc),
            "overlapping placed texture must be evicted"
        );
        assert!(
            crate::resource::has_placed_texture(addr_b, &desc),
            "non-overlapping placed texture must remain cached"
        );
        assert!(crate::resource::placed_epoch() > epoch_before);

        // Clean up `addr_b` too.
        crate::watch::title_memory_remapped(addr_b as usize, 64 << 10);
        assert!(!crate::resource::has_placed_texture(addr_b, &desc));
    }

    /// Typeless SRV and UAV views on typeless resources are remapped to the first
    /// concrete format of their family so D3D12 can create the view without E_INVALIDARG.
    #[test]
    fn typeless_view_on_typeless_resource_remaps_to_concrete_format() {
        // R8G8B8A8_TYPELESS (27) with R8G8B8A8_TYPELESS view (27) -> R8G8B8A8_UNORM (28)
        let mut fmt = DXGI_FORMAT(27);
        assert!(adapt_view_format(27, Some(&mut fmt), false));
        assert_eq!(fmt.0, 28);

        // B8G8R8A8_TYPELESS (90) with B8G8R8A8_TYPELESS view (90) -> B8G8R8A8_UNORM (87)
        let mut fmt = DXGI_FORMAT(90);
        assert!(adapt_view_format(90, Some(&mut fmt), false));
        assert_eq!(fmt.0, 87);

        // R32_TYPELESS (39) with R32_TYPELESS view (39) -> R32_FLOAT (41)
        let mut fmt = DXGI_FORMAT(39);
        assert!(adapt_view_format(39, Some(&mut fmt), false));
        assert_eq!(fmt.0, 41);

        // UNKNOWN view (0) on typeless texture -> first concrete format
        let mut fmt = DXGI_FORMAT(0);
        assert!(adapt_view_format(27, Some(&mut fmt), false));
        assert_eq!(fmt.0, 28);

        // Typed view (29 = R8G8B8A8_UNORM_SRGB) on typeless texture (27) stays 29
        let mut fmt = DXGI_FORMAT(29);
        assert!(adapt_view_format(27, Some(&mut fmt), false));
        assert_eq!(fmt.0, 29);

        // Typeless view (27) on typed texture (28) adopts texture's format (28)
        let mut fmt = DXGI_FORMAT(27);
        assert!(adapt_view_format(28, Some(&mut fmt), false));
        assert_eq!(fmt.0, 28);

        // Typeless texture with None format is refused
        assert!(!adapt_view_format(27, None, false));

        // Incompatible family is refused
        let mut fmt = DXGI_FORMAT(10);
        assert!(!adapt_view_format(27, Some(&mut fmt), false));

        // UAV with sRGB (29) converts to linear (28)
        let mut fmt = DXGI_FORMAT(29);
        assert!(adapt_view_format(27, Some(&mut fmt), true));
        assert_eq!(fmt.0, 28);
    }

    #[test]
    fn a_free_table_can_straddle_the_search_cursor() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let left = heap.dynamic(4).unwrap();
        let right = heap.dynamic(5).unwrap();
        let tail = heap.dynamic(DYNAMIC - 9).unwrap();

        heap.release(left, 4);
        assert_eq!(heap.dynamic(4), Some(left)); // cursor now separates the two tables
        heap.release(right, 5);
        heap.release(left, 4);
        assert_eq!(heap.dynamic(9), Some(left));
        assert_eq!(heap.dynamic(1), None, "the tail is still held");

        heap.release(tail, DYNAMIC - 9);
        assert_eq!(heap.dynamic(DYNAMIC - 9), Some(tail));
    }

    #[test]
    fn null_desc_maps_buffer_and_texture_annotations() {
        // Raw buffer SRV and UAV.
        match null_desc_for_props((11, 0)) {
            Some(NullDesc::Srv(d)) => {
                assert_eq!(d.Format, DXGI_FORMAT_R32_TYPELESS);
                assert_eq!(d.ViewDimension, D3D12_SRV_DIMENSION_BUFFER);
                assert_eq!(
                    d.Shader4ComponentMapping,
                    D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING
                );
                // SAFETY: ViewDimension is BUFFER.
                let b = unsafe { d.Anonymous.Buffer };
                assert_eq!(b.Flags, D3D12_BUFFER_SRV_FLAG_RAW);
                assert_eq!(b.StructureByteStride, 0);
            }
            other => panic!("expected raw buffer SRV, got {other:?}"),
        }
        match null_desc_for_props((11 | 0x1000, 0)) {
            Some(NullDesc::Uav(d)) => {
                assert_eq!(d.Format, DXGI_FORMAT_R32_TYPELESS);
                assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_BUFFER);
                // SAFETY: ViewDimension is BUFFER.
                let b = unsafe { d.Anonymous.Buffer };
                assert_eq!(b.Flags, D3D12_BUFFER_UAV_FLAG_RAW);
                assert_eq!(b.StructureByteStride, 0);
            }
            other => panic!("expected raw buffer UAV, got {other:?}"),
        }

        // Structured buffer SRV and UAV (with stride and 0-stride fallback).
        match null_desc_for_props((12, 64)) {
            Some(NullDesc::Srv(d)) => {
                assert_eq!(d.Format, DXGI_FORMAT_UNKNOWN);
                assert_eq!(d.ViewDimension, D3D12_SRV_DIMENSION_BUFFER);
                // SAFETY: ViewDimension is BUFFER.
                let b = unsafe { d.Anonymous.Buffer };
                assert_eq!(b.StructureByteStride, 64);
                assert_eq!(b.Flags, D3D12_BUFFER_SRV_FLAG_NONE);
            }
            other => panic!("expected structured buffer SRV, got {other:?}"),
        }
        match null_desc_for_props((12 | 0x1000, 0)) {
            Some(NullDesc::Uav(d)) => {
                assert_eq!(d.Format, DXGI_FORMAT_UNKNOWN);
                // SAFETY: ViewDimension is BUFFER.
                let b = unsafe { d.Anonymous.Buffer };
                assert_eq!(b.StructureByteStride, 4);
            }
            other => panic!("expected structured buffer UAV, got {other:?}"),
        }

        // Typed buffers respect component type (F32, U32, I32).
        for (p1, fmt) in [
            (0x409, DXGI_FORMAT_R32_FLOAT),
            (0x105, DXGI_FORMAT_R32_UINT),
            (0x104, DXGI_FORMAT_R32_SINT),
        ] {
            match null_desc_for_props((10, p1)) {
                Some(NullDesc::Srv(d)) => {
                    assert_eq!(d.Format, fmt);
                    assert_eq!(d.ViewDimension, D3D12_SRV_DIMENSION_BUFFER);
                }
                other => panic!("expected typed buffer SRV for {p1:#x}, got {other:?}"),
            }
            match null_desc_for_props((10 | 0x1000, p1)) {
                Some(NullDesc::Uav(d)) => {
                    assert_eq!(d.Format, fmt);
                    assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_BUFFER);
                }
                other => panic!("expected typed buffer UAV for {p1:#x}, got {other:?}"),
            }
        }

        // Texture SRVs cover every dimension.
        let srv_dims = [
            (1, D3D12_SRV_DIMENSION_TEXTURE1D),
            (2, D3D12_SRV_DIMENSION_TEXTURE2D),
            (3, D3D12_SRV_DIMENSION_TEXTURE2DMS),
            (4, D3D12_SRV_DIMENSION_TEXTURE3D),
            (5, D3D12_SRV_DIMENSION_TEXTURECUBE),
            (6, D3D12_SRV_DIMENSION_TEXTURE1DARRAY),
            (7, D3D12_SRV_DIMENSION_TEXTURE2DARRAY),
            (8, D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY),
            (9, D3D12_SRV_DIMENSION_TEXTURECUBEARRAY),
        ];
        for (kind, dim) in srv_dims {
            match null_desc_for_props((kind, 0x409)) {
                Some(NullDesc::Srv(d)) => {
                    assert_eq!(d.ViewDimension, dim);
                    assert_eq!(d.Format, DXGI_FORMAT_R32_FLOAT);
                    assert_eq!(
                        d.Shader4ComponentMapping,
                        D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING
                    );
                }
                other => panic!("expected texture SRV for kind {kind}, got {other:?}"),
            }
        }

        // Texture UAVs (including 3D WSize, 2D atomic uint, cube -> 2D array).
        let uav_dims = [
            (
                1,
                0x409,
                D3D12_UAV_DIMENSION_TEXTURE1D,
                DXGI_FORMAT_R32_FLOAT,
            ),
            (
                2,
                0x105,
                D3D12_UAV_DIMENSION_TEXTURE2D,
                DXGI_FORMAT_R32_UINT,
            ),
            (
                4,
                0x409,
                D3D12_UAV_DIMENSION_TEXTURE3D,
                DXGI_FORMAT_R32_FLOAT,
            ),
            (
                6,
                0x409,
                D3D12_UAV_DIMENSION_TEXTURE1DARRAY,
                DXGI_FORMAT_R32_FLOAT,
            ),
            (
                7,
                0x409,
                D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
                DXGI_FORMAT_R32_FLOAT,
            ),
            (
                5,
                0x409,
                D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
                DXGI_FORMAT_R32_FLOAT,
            ),
            (
                9,
                0x409,
                D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
                DXGI_FORMAT_R32_FLOAT,
            ),
        ];
        for (kind, p1, dim, fmt) in uav_dims {
            match null_desc_for_props((kind | 0x1000, p1)) {
                Some(NullDesc::Uav(d)) => {
                    assert_eq!(d.ViewDimension, dim);
                    assert_eq!(d.Format, fmt);
                    if kind == 4 {
                        // SAFETY: ViewDimension is TEXTURE3D.
                        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, 1);
                    }
                }
                other => panic!("expected texture UAV for kind {kind}, got {other:?}"),
            }
        }

        // CBV and unsupported kinds.
        assert!(matches!(
            null_desc_for_props((13, 0x1_0000)),
            Some(NullDesc::Cbv)
        ));
        for bad in [
            (0, 0),
            (14, 0),
            (3 | 0x1000, 0x409),
            (8 | 0x1000, 0x409),
            (13 | 0x1000, 0),
        ] {
            assert!(null_desc_for_props(bad).is_none(), "{bad:?}");
        }
    }

    #[test]
    fn host_accepts_typed_null_descriptors_and_mirror_transitions() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let Some(host) = host() else {
            return;
        };
        let slot = heap.dynamic(1).unwrap();
        let all = [
            (1, 0x409),
            (2, 0x409),
            (3, 0x409),
            (4, 0x409),
            (5, 0x409),
            (6, 0x409),
            (7, 0x409),
            (8, 0x409),
            (9, 0x409),
            (10, 0x409),
            (11, 0),
            (12, 16),
            (13, 0x1_0000),
            (1 | 0x1000, 0x409),
            (2 | 0x1000, 0x105),
            (4 | 0x1000, 0x409),
            (6 | 0x1000, 0x409),
            (7 | 0x1000, 0x409),
            (10 | 0x1000, 0x104),
            (11 | 0x1000, 0),
            (12 | 0x1000, 32),
        ];
        for props in all {
            let desc = null_desc_for_props(props).expect("valid props");
            heap.make_typed_null(slot, &desc);
        }
        heap.release(slot, 1);

        // Mirrored slot: empty -> typed null -> updated typed null -> real view.
        let mut title = [0u32; 8];
        let handle = title.as_mut_ptr() as u64;
        assert!(
            heap.mirror_for_draw(0, handle, Some((11 | 0x1000, 0)))
                .is_none()
        );
        assert_eq!(
            heap.made.lock().unwrap()[0],
            SlotState::TypedNull(11 | 0x1000, 0)
        );
        assert!(
            heap.mirror_for_draw(0, handle, Some((12 | 0x1000, 16)))
                .is_none()
        );
        assert_eq!(
            heap.made.lock().unwrap()[0],
            SlotState::TypedNull(12 | 0x1000, 16)
        );
        // Out-of-bounds mirrored slot is ignored cleanly.
        assert!(
            heap.mirror_for_draw(heap.count(), handle, Some((11 | 0x1000, 0)))
                .is_none()
        );

        // Writing a real view later still triggers `mirror` to make it.
        super::super::create_cbv(std::ptr::null(), &[0x1000, 0, 256, 0], handle);
        let view = heap.mirror_for_draw(0, handle, Some((13, 0x1_0000)));
        assert_eq!(view, Some(View::Constant(0x1000, 256)));
        assert!(matches!(heap.made.lock().unwrap()[0], SlotState::View(_)));

        // SAFETY: live host device.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }
}
