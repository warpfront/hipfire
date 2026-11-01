// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Weight slabs: small device-resident weight tensors packed into shared,
//! 2 MiB-multiple `hipMalloc` slabs.
//!
//! ROCr rounds every `hipMalloc` above 1 MiB up to a multiple of 2 MiB (and
//! anything at or below it to 4 KiB). A model whose weights are thousands of
//! separate 1-8 MiB tensors loses gigabytes to that rounding (Raon on the
//! 7900 XTX: 4,758 MiB of 21,624 MiB in fdinfo). A slab carries many tensors
//! back to back, each sub-allocated at its exact size on a 4 KiB boundary (the
//! alignment a plain `hipMalloc` of that tensor already gets), so the rounding
//! is paid once per slab instead of once per tensor.
//!
//! Lifecycle. A load *scope* ([`HipRuntime::slab_scope_begin`] /
//! [`HipRuntime::slab_scope_end`]) makes eligible uploads sub-allocate from the
//! open slabs; closing the outermost scope *seals* every open slab, so a slab
//! never mixes tensors of two model loads and its unused tail is not handed to
//! later, unrelated allocations. Each view is a [`DeviceBuffer`] tagged
//! `SlabView`: it looks like an owning `hipMalloc` buffer to ownership
//! predicates (`is_hip_allocation`), and `HipRuntime::free` of a view only
//! releases it. A sealed slab with no live view is retired; the real `hipFree`
//! happens in [`HipRuntime::slab_flush`] (the pool drain at unload) or at once
//! for a direct `free`.
//!
//! Zero fill. A slab is zeroed when it is created, so the bytes between two
//! tensors read as zeros exactly like the unwritten tail of a fresh
//! allocation, and a kernel's masked over-read of a tensor's tail is
//! deterministic.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use crate::{DeviceBuffer, DeviceBufferOwnership, HipError, HipResult, HipRuntime};

/// Alignment of every sub-allocation: the page granule a plain `hipMalloc` of
/// any size already returns.
pub const SLAB_ALIGN: usize = 4096;
/// ROCr's allocation granule for sizes above [`SMALL_ALLOC_LIMIT`].
pub const ALLOC_GRANULE: usize = 2 << 20;
/// Largest size ROCr does not round to [`ALLOC_GRANULE`].
pub const SMALL_ALLOC_LIMIT: usize = 1 << 20;
/// Granule of an allocation at or below [`SMALL_ALLOC_LIMIT`].
pub const SMALL_ALLOC_GRANULE: usize = 4096;
/// Open slabs a placement may first-fit into; the oldest is sealed beyond this.
const MAX_OPEN_SLABS: usize = 4;

/// Bytes a `hipMalloc(size)` occupies in device memory (the `drm-memory-vram`
/// fdinfo charge): 4 KiB-rounded up to 1 MiB, 2 MiB-rounded above it.
pub fn device_alloc_charge(size: usize) -> usize {
    let granule = if size > SMALL_ALLOC_LIMIT {
        ALLOC_GRANULE
    } else {
        SMALL_ALLOC_GRANULE
    };
    size.div_ceil(granule).saturating_mul(granule)
}

/// `HIPFIRE_WEIGHT_SLAB`: which weight tensors are packed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlabMode {
    /// Every weight is its own allocation.
    Off,
    /// Tensors of at most 2 MiB are packed.
    Up2M,
    /// Tensors of at most 16 MiB are packed.
    Up16M,
}

/// Mode when `HIPFIRE_WEIGHT_SLAB` is unset.
pub const DEFAULT_SLAB_MODE: SlabMode = SlabMode::Off;

impl SlabMode {
    /// Parse `off|2m|16m` (case-insensitive; `0`/`none` mean off).
    pub fn parse(text: &str) -> Result<SlabMode, String> {
        match text.trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "none" | "false" => Ok(SlabMode::Off),
            "2m" | "2mib" => Ok(SlabMode::Up2M),
            "16m" | "16mib" => Ok(SlabMode::Up16M),
            other => Err(format!(
                "HIPFIRE_WEIGHT_SLAB={other:?} is not one of off|2m|16m"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            SlabMode::Off => "off",
            SlabMode::Up2M => "2m",
            SlabMode::Up16M => "16m",
        }
    }

    /// Largest tensor the mode packs (0 = none).
    pub fn max_tensor_bytes(self) -> usize {
        match self {
            SlabMode::Off => 0,
            SlabMode::Up2M => 2 << 20,
            SlabMode::Up16M => 16 << 20,
        }
    }

    /// Capacity of a fresh slab (a multiple of [`ALLOC_GRANULE`], at least the
    /// largest tensor the mode packs).
    pub fn slab_bytes(self) -> usize {
        match self {
            SlabMode::Off => 0,
            SlabMode::Up2M => 32 << 20,
            SlabMode::Up16M => 64 << 20,
        }
    }

    /// Mode from `HIPFIRE_WEIGHT_SLAB` (environment, or `[developer]
    /// weight_slab` in the config snapshot). An unparsable value is reported
    /// on stderr and treated as the default.
    pub fn from_env() -> SlabMode {
        match hipfire_config::developer_var("HIPFIRE_WEIGHT_SLAB") {
            Ok(value) if !value.trim().is_empty() => SlabMode::parse(&value).unwrap_or_else(|e| {
                eprintln!("[weight-slab] {e}; using {}", DEFAULT_SLAB_MODE.name());
                DEFAULT_SLAB_MODE
            }),
            _ => DEFAULT_SLAB_MODE,
        }
    }
}

/// Counters of the slabs a runtime holds (and has ever created).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SlabStats {
    /// Slabs currently allocated (live or retired, not yet flushed).
    pub slabs: usize,
    /// Capacity bytes of those slabs.
    pub slab_bytes: usize,
    /// Views currently live.
    pub live_views: usize,
    /// Slabs with no live view still held for [`HipRuntime::slab_flush`].
    pub retired_slabs: usize,
    /// Views ever created.
    pub views_created: usize,
    /// Payload bytes of every view ever created.
    pub payload_bytes: usize,
    /// Slabs ever created.
    pub slabs_created: usize,
    /// Capacity bytes of every slab ever created.
    pub slab_bytes_created: usize,
}

/// What one outermost scope packed (`SlabStats` deltas).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SlabScopeSummary {
    pub mode: Option<SlabMode>,
    pub tensors: usize,
    pub payload_bytes: usize,
    pub slabs: usize,
    pub slab_bytes: usize,
}

#[derive(Debug)]
struct SlabRec {
    size: usize,
    cursor: usize,
    live: usize,
    sealed: bool,
    retired: bool,
}

/// Placement and lifetime bookkeeping, independent of the device.
#[derive(Debug, Default)]
pub(crate) struct SlabBook {
    slabs: BTreeMap<usize, SlabRec>,
    open: Vec<usize>,
    stats: SlabStats,
}

/// What releasing a view did to its slab.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Released {
    /// The slab still has live views (or is still open).
    Held,
    /// The slab is sealed with no live view; the caller owns its `hipFree`.
    Dead { base: usize },
}

fn align_up(value: usize) -> usize {
    (value + SLAB_ALIGN - 1) & !(SLAB_ALIGN - 1)
}

impl SlabBook {
    /// Best fit of `size` into the open slabs (the one left with the least
    /// free space, the oldest on a tie): the view address.
    pub(crate) fn place(&mut self, size: usize) -> Option<usize> {
        let mut best: Option<(usize, usize, usize)> = None; // (leftover, base, start)
        for &base in &self.open {
            let slab = self.slabs.get(&base)?;
            let start = align_up(slab.cursor);
            let Some(end) = start.checked_add(size).filter(|&end| end <= slab.size) else {
                continue;
            };
            let leftover = slab.size - end;
            if best.is_none_or(|(l, _, _)| leftover < l) {
                best = Some((leftover, base, start));
            }
        }
        let (_, base, start) = best?;
        let slab = self.slabs.get_mut(&base)?;
        slab.cursor = start + size;
        slab.live += 1;
        self.stats.views_created += 1;
        self.stats.payload_bytes += size;
        Some(base + start)
    }

    /// Register a freshly allocated, zeroed slab and open it. Returns the
    /// bases of older open slabs the bounded ring sealed that hold no view
    /// (to be freed).
    pub(crate) fn open_slab(&mut self, base: usize, size: usize) -> Vec<usize> {
        self.slabs.insert(
            base,
            SlabRec {
                size,
                cursor: 0,
                live: 0,
                sealed: false,
                retired: false,
            },
        );
        self.open.push(base);
        self.stats.slabs_created += 1;
        self.stats.slab_bytes_created += size;
        let mut dead = Vec::new();
        while self.open.len() > MAX_OPEN_SLABS {
            let oldest = self.open.remove(0);
            dead.extend(self.seal_one(oldest));
        }
        dead
    }

    fn seal_one(&mut self, base: usize) -> Option<usize> {
        let slab = self.slabs.get_mut(&base)?;
        slab.sealed = true;
        if slab.live == 0 && !slab.retired {
            slab.retired = true;
            return Some(base);
        }
        None
    }

    /// Seal every open slab; the bases that are already dead (to be freed).
    pub(crate) fn seal_all(&mut self) -> Vec<usize> {
        let open = std::mem::take(&mut self.open);
        open.into_iter()
            .filter_map(|base| self.seal_one(base))
            .collect()
    }

    /// Base of the slab containing `ptr`.
    fn locate(&self, ptr: usize) -> Option<usize> {
        let (&base, slab) = self.slabs.range(..=ptr).next_back()?;
        (ptr < base + slab.size).then_some(base)
    }

    /// Drop one live view at `ptr`.
    pub(crate) fn release(&mut self, ptr: usize) -> Result<Released, String> {
        let base = self
            .locate(ptr)
            .ok_or_else(|| format!("slab view 0x{ptr:x} belongs to no live slab"))?;
        let slab = self.slabs.get_mut(&base).expect("located slab");
        if slab.live == 0 {
            return Err(format!(
                "slab view 0x{ptr:x} released more often than created (slab 0x{base:x})"
            ));
        }
        slab.live -= 1;
        if slab.live == 0 && slab.sealed && !slab.retired {
            slab.retired = true;
            return Ok(Released::Dead { base });
        }
        Ok(Released::Held)
    }

    /// Forget a retired slab (its `hipFree` was issued).
    pub(crate) fn remove(&mut self, base: usize) {
        self.slabs.remove(&base);
    }

    /// Bases of retired slabs not yet removed.
    pub(crate) fn retired_bases(&self) -> Vec<usize> {
        self.slabs
            .iter()
            .filter(|(_, s)| s.retired)
            .map(|(&b, _)| b)
            .collect()
    }

    pub(crate) fn snapshot(&self) -> SlabStats {
        let mut stats = self.stats;
        stats.slabs = self.slabs.len();
        stats.slab_bytes = self.slabs.values().map(|s| s.size).sum();
        stats.live_views = self.slabs.values().map(|s| s.live).sum();
        stats.retired_slabs = self.slabs.values().filter(|s| s.retired).count();
        stats
    }
}

/// Per-runtime slab state.
pub(crate) struct WeightSlabs {
    inner: Mutex<SlabInner>,
}

struct SlabInner {
    /// Resolved at the first scope (the daemon installs its config snapshot
    /// after the runtime is created), or set explicitly.
    mode: Option<SlabMode>,
    depth: u32,
    /// Counters when the outermost scope opened.
    scope_start: SlabStats,
    book: SlabBook,
    owners: HashMap<usize, DeviceBuffer>,
}

impl Default for WeightSlabs {
    fn default() -> Self {
        WeightSlabs {
            inner: Mutex::new(SlabInner {
                mode: None,
                depth: 0,
                scope_start: SlabStats::default(),
                book: SlabBook::default(),
                owners: HashMap::new(),
            }),
        }
    }
}

impl WeightSlabs {
    fn lock(&self) -> std::sync::MutexGuard<'_, SlabInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl SlabInner {
    fn mode(&mut self) -> SlabMode {
        *self.mode.get_or_insert_with(SlabMode::from_env)
    }
}

fn view(ptr: usize, size: usize) -> DeviceBuffer {
    DeviceBuffer {
        ptr: ptr as *mut std::ffi::c_void,
        size,
        ownership: DeviceBufferOwnership::SlabView,
        token: None,
    }
}

impl HipRuntime {
    /// Packing mode of this runtime.
    pub fn slab_mode(&self) -> SlabMode {
        self.slabs.lock().mode()
    }

    /// Override the packing mode (resolved from `HIPFIRE_WEIGHT_SLAB` at the
    /// first scope). Only affects scopes opened afterwards.
    pub fn set_slab_mode(&self, mode: SlabMode) {
        self.slabs.lock().mode = Some(mode);
    }

    /// Whether uploads inside this scope are currently packed.
    pub fn slab_scope_active(&self) -> bool {
        let mut inner = self.slabs.lock();
        inner.depth > 0 && inner.mode() != SlabMode::Off
    }

    /// Open (or nest) a weight-load scope. Packing needs the mode on, no
    /// railgun registry recording and no plan tape, otherwise the scope is
    /// inert.
    pub fn slab_scope_begin(&self) {
        let mut inner = self.slabs.lock();
        if inner.depth == 0 && (crate::registry::enabled() || self.plan.active()) {
            // Recording modes pair allocations one to one; stay out of them.
            inner.mode = Some(SlabMode::Off);
        }
        if inner.depth == 0 {
            // Resolve now: the config snapshot is installed by the first load.
            let _ = inner.mode();
        }
        if inner.depth == 0 {
            inner.scope_start = inner.book.snapshot();
        }
        inner.depth += 1;
    }

    /// Close one scope level; the outermost close seals every open slab and
    /// returns what the scope packed (`None` when it packed nothing or the
    /// scope is still nested).
    pub fn slab_scope_end(&self) -> Option<SlabScopeSummary> {
        let (dead, summary) = {
            let mut inner = self.slabs.lock();
            if inner.depth == 0 {
                return None;
            }
            inner.depth -= 1;
            if inner.depth > 0 {
                return None;
            }
            let now = inner.book.snapshot();
            let start = inner.scope_start;
            let summary = SlabScopeSummary {
                mode: inner.mode,
                tensors: now.views_created - start.views_created,
                payload_bytes: now.payload_bytes - start.payload_bytes,
                slabs: now.slabs_created - start.slabs_created,
                slab_bytes: now.slab_bytes_created - start.slab_bytes_created,
            };
            (inner.book.seal_all(), summary)
        };
        // A slab sealed with no view never held a live tensor: free it now.
        for base in dead {
            let _ = self.slab_free_base(base);
        }
        (summary.tensors > 0).then_some(summary)
    }

    /// Sub-allocate `size` bytes from the open slabs when the scope is active
    /// and `size` is eligible. `Ok(None)`: not packed, the caller allocates
    /// directly. The memory is zero-filled.
    pub fn slab_alloc(&self, size: usize) -> HipResult<Option<DeviceBuffer>> {
        let mut inner = self.slabs.lock();
        let mode = inner.mode.unwrap_or(SlabMode::Off);
        if inner.depth == 0 || size == 0 || size > mode.max_tensor_bytes() {
            return Ok(None);
        }
        if let Some(ptr) = inner.book.place(size) {
            return Ok(Some(view(ptr, size)));
        }
        // New slab: the mode's capacity, falling back to the smallest slab
        // that holds this tensor when the device cannot spare the full size.
        let want = mode.slab_bytes();
        let minimal = size.div_ceil(ALLOC_GRANULE) * ALLOC_GRANULE;
        // A failure of both is left to the direct path, which reports the
        // same error a plain allocation would.
        let Some((buf, cap)) = [want, minimal]
            .into_iter()
            .find_map(|cap| self.malloc(cap).ok().map(|buf| (buf, cap)))
        else {
            return Ok(None);
        };
        if let Err(error) = self.memset(&buf, 0, cap) {
            let _ = self.free(buf);
            return Err(error);
        }
        let base = buf.as_ptr() as usize;
        inner.owners.insert(base, buf);
        let dead = inner.book.open_slab(base, cap);
        let ptr = inner
            .book
            .place(size)
            .ok_or_else(|| HipError::new(0, "weight slab: a fresh slab refused its first view"))?;
        drop(inner);
        for base in dead {
            let _ = self.slab_free_base(base);
        }
        Ok(Some(view(ptr, size)))
    }

    /// Release one view. With `defer`, a slab that becomes dead stays
    /// allocated until [`Self::slab_flush`]; otherwise it is freed now.
    pub fn slab_release(&self, buf: DeviceBuffer, defer: bool) -> HipResult<()> {
        if buf.ownership != DeviceBufferOwnership::SlabView {
            return Err(HipError::new(0, "slab_release: not a slab view"));
        }
        let outcome = {
            let mut inner = self.slabs.lock();
            inner
                .book
                .release(buf.ptr as usize)
                .map_err(|e| HipError::new(0, &e))?
        };
        match outcome {
            Released::Held => Ok(()),
            Released::Dead { .. } if defer => Ok(()),
            Released::Dead { base } => self.slab_free_base(base),
        }
    }

    fn slab_free_base(&self, base: usize) -> HipResult<()> {
        let owner = {
            let mut inner = self.slabs.lock();
            inner.book.remove(base);
            inner.owners.remove(&base)
        };
        match owner {
            Some(buf) => self.free(buf),
            None => Ok(()),
        }
    }

    /// Free every retired (sealed, no live view) slab; the bytes freed.
    pub fn slab_flush(&self) -> usize {
        let bases = self.slabs.lock().book.retired_bases();
        let mut freed = 0usize;
        for base in bases {
            let size = self
                .slabs
                .lock()
                .owners
                .get(&base)
                .map_or(0, DeviceBuffer::size);
            if self.slab_free_base(base).is_ok() {
                freed += size;
            }
        }
        freed
    }

    /// Counters of this runtime's slabs.
    pub fn slab_stats(&self) -> SlabStats {
        self.slabs.lock().book.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1 << 20;

    #[test]
    fn charge_rounds_like_rocr() {
        assert_eq!(device_alloc_charge(0), 0);
        assert_eq!(device_alloc_charge(1), 4096);
        assert_eq!(device_alloc_charge(MIB), MIB);
        assert_eq!(device_alloc_charge(MIB + 4096), 2 * MIB);
        assert_eq!(device_alloc_charge(3 * MIB), 4 * MIB);
        assert_eq!(device_alloc_charge(4 * MIB + 4096), 6 * MIB);
        assert_eq!(device_alloc_charge(6 * MIB + MIB / 4), 8 * MIB);
    }

    #[test]
    fn modes_parse() {
        assert_eq!(SlabMode::parse("off"), Ok(SlabMode::Off));
        assert_eq!(SlabMode::parse(" 2M "), Ok(SlabMode::Up2M));
        assert_eq!(SlabMode::parse("16m"), Ok(SlabMode::Up16M));
        assert!(SlabMode::parse("4m").is_err());
        for mode in [SlabMode::Up2M, SlabMode::Up16M] {
            assert_eq!(mode.slab_bytes() % ALLOC_GRANULE, 0);
            assert!(mode.slab_bytes() >= mode.max_tensor_bytes());
        }
    }

    #[test]
    fn placement_is_page_aligned_first_fit() {
        let mut book = SlabBook::default();
        assert_eq!(book.place(100), None);
        let _ = book.open_slab(0x1000_0000, 4 * MIB);
        let a = book.place(100).unwrap();
        let b = book.place(5000).unwrap();
        let c = book.place(MIB).unwrap();
        assert_eq!(a, 0x1000_0000);
        assert_eq!(b, 0x1000_0000 + 4096);
        assert_eq!(c, 0x1000_0000 + 4096 + 8192);
        assert_eq!(a % SLAB_ALIGN, 0);
        // 3 MiB does not fit the remainder: no placement.
        assert_eq!(book.place(3 * MIB), None);
        // A smaller one still does (first fit).
        assert!(book.place(MIB).is_some());
        assert_eq!(book.snapshot().live_views, 4);
    }

    #[test]
    fn sealed_slab_dies_with_its_last_view() {
        let mut book = SlabBook::default();
        let _ = book.open_slab(0x2000_0000, 2 * MIB);
        let a = book.place(4096).unwrap();
        let b = book.place(4096).unwrap();
        assert!(book.seal_all().is_empty());
        assert_eq!(book.release(a), Ok(Released::Held));
        assert_eq!(
            book.release(b),
            Ok(Released::Dead { base: 0x2000_0000 })
        );
        assert_eq!(book.retired_bases(), vec![0x2000_0000]);
        book.remove(0x2000_0000);
        assert_eq!(book.snapshot().slabs, 0);
        assert!(book.release(b).is_err());
    }

    #[test]
    fn open_slab_survives_a_transient_free() {
        let mut book = SlabBook::default();
        let _ = book.open_slab(0x3000_0000, 2 * MIB);
        let a = book.place(4096).unwrap();
        // Still open: freeing the only view must not retire the slab.
        assert_eq!(book.release(a), Ok(Released::Held));
        assert!(book.place(4096).is_some());
    }

    #[test]
    fn empty_sealed_slab_is_reported_dead() {
        let mut book = SlabBook::default();
        let _ = book.open_slab(0x4000_0000, 2 * MIB);
        assert_eq!(book.seal_all(), vec![0x4000_0000]);
    }

    #[test]
    fn open_ring_is_bounded() {
        let mut book = SlabBook::default();
        for i in 0..(MAX_OPEN_SLABS + 2) {
            let _ = book.open_slab(0x5000_0000 + i * 4 * MIB, 2 * MIB);
        }
        assert_eq!(book.open.len(), MAX_OPEN_SLABS);
        // The two evicted slabs had no views and are retired.
        assert_eq!(book.retired_bases().len(), 2);
    }

    #[test]
    fn views_of_one_slab_never_overlap() {
        let mut book = SlabBook::default();
        let _ = book.open_slab(0x6000_0000, 32 * MIB);
        let mut spans = Vec::new();
        let mut size = 1usize;
        while let Some(ptr) = book.place(size) {
            spans.push((ptr, size));
            size = size * 3 % (MIB + 17) + 1;
            if spans.len() > 10_000 {
                break;
            }
        }
        spans.sort();
        for pair in spans.windows(2) {
            assert!(pair[0].0 + pair[0].1 <= pair[1].0);
        }
    }
}
