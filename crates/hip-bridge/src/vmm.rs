//! Explicit ownership for HIP virtual-memory mappings.
//!
//! `DeviceBuffer` and the existing GPU pool are released with `hipFree`, which
//! is invalid for addresses reserved through the VMM API. `VmmArena` therefore
//! keeps physical handles and mapping ranges separate and requires an explicit
//! [`VmmArena::release`] call.
//!
//! A released arena's virtual address range is never returned to the driver.
//! On ROCm 7.14+/gfx1201 a VA that was mapped, unmapped, and mapped again keeps
//! translating to its first backing: kernels read the previous owner's pages.
//! `hipMemAddressFree` followed by a hint-less `hipMemAddressReserve` hands the
//! same range back, so every model unload -> load in one process hit that.
//! Keeping the range reserved (see [`retired_va_bytes`]) means no later
//! reservation or `hipMalloc` can land on a VA the GPU has translated before.

use crate::ffi::{HipMemLocation, HIP_MEM_ALLOCATION_GRANULARITY_MINIMUM};
use crate::{
    DeviceBuffer, HipError, HipMemAccessDesc, HipMemAllocationProp, HipMemGenericAllocationHandle,
    HipResult, HipRuntime, HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED,
};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::ffi::c_void;
#[cfg(unix)]
use std::os::fd::{BorrowedFd, OwnedFd};
use std::sync::Mutex;

/// The HIP calls [`VmmArena::reserve`], [`VmmArena::map_next`] and
/// [`VmmArena::release`] make. [`HipRuntime`] is the production driver; unit
/// tests substitute a recording driver so the map/unmap sequence of the arena
/// lifecycle is checkable without a device.
pub trait VmmDriver {
    fn device_count(&self) -> HipResult<i32>;
    fn set_device(&self, id: i32) -> HipResult<()>;
    fn can_access_peer(&self, device: i32, peer_device: i32) -> HipResult<bool>;
    fn device_synchronize(&self) -> HipResult<()>;
    fn mem_get_allocation_granularity(
        &self,
        prop: &HipMemAllocationProp,
        option: u32,
    ) -> HipResult<usize>;
    fn mem_address_reserve(&self, size: usize, alignment: usize) -> HipResult<*mut c_void>;
    fn mem_create(
        &self,
        size: usize,
        prop: &HipMemAllocationProp,
    ) -> HipResult<HipMemGenericAllocationHandle>;
    /// # Safety
    /// As [`HipRuntime::mem_map`].
    unsafe fn mem_map(
        &self,
        ptr: *mut c_void,
        size: usize,
        handle: HipMemGenericAllocationHandle,
    ) -> HipResult<()>;
    /// # Safety
    /// As [`HipRuntime::mem_set_access`].
    unsafe fn mem_set_access(
        &self,
        ptr: *mut c_void,
        size: usize,
        descs: &[HipMemAccessDesc],
    ) -> HipResult<()>;
    /// # Safety
    /// As [`HipRuntime::mem_unmap`].
    unsafe fn mem_unmap(&self, ptr: *mut c_void, size: usize) -> HipResult<()>;
    /// # Safety
    /// As [`HipRuntime::mem_release`].
    unsafe fn mem_release(&self, handle: HipMemGenericAllocationHandle) -> HipResult<()>;
}

impl VmmDriver for HipRuntime {
    fn device_count(&self) -> HipResult<i32> {
        HipRuntime::device_count(self)
    }
    fn set_device(&self, id: i32) -> HipResult<()> {
        HipRuntime::set_device(self, id)
    }
    fn can_access_peer(&self, device: i32, peer_device: i32) -> HipResult<bool> {
        HipRuntime::can_access_peer(self, device, peer_device)
    }
    fn device_synchronize(&self) -> HipResult<()> {
        HipRuntime::device_synchronize(self)
    }
    fn mem_get_allocation_granularity(
        &self,
        prop: &HipMemAllocationProp,
        option: u32,
    ) -> HipResult<usize> {
        HipRuntime::mem_get_allocation_granularity(self, prop, option)
    }
    fn mem_address_reserve(&self, size: usize, alignment: usize) -> HipResult<*mut c_void> {
        HipRuntime::mem_address_reserve(self, size, alignment)
    }
    fn mem_create(
        &self,
        size: usize,
        prop: &HipMemAllocationProp,
    ) -> HipResult<HipMemGenericAllocationHandle> {
        HipRuntime::mem_create(self, size, prop)
    }
    unsafe fn mem_map(
        &self,
        ptr: *mut c_void,
        size: usize,
        handle: HipMemGenericAllocationHandle,
    ) -> HipResult<()> {
        // SAFETY: forwarded under this method's contract.
        unsafe { HipRuntime::mem_map(self, ptr, size, handle) }
    }
    unsafe fn mem_set_access(
        &self,
        ptr: *mut c_void,
        size: usize,
        descs: &[HipMemAccessDesc],
    ) -> HipResult<()> {
        // SAFETY: forwarded under this method's contract.
        unsafe { HipRuntime::mem_set_access(self, ptr, size, descs) }
    }
    unsafe fn mem_unmap(&self, ptr: *mut c_void, size: usize) -> HipResult<()> {
        // SAFETY: forwarded under this method's contract.
        unsafe { HipRuntime::mem_unmap(self, ptr, size) }
    }
    unsafe fn mem_release(&self, handle: HipMemGenericAllocationHandle) -> HipResult<()> {
        // SAFETY: forwarded under this method's contract.
        unsafe { HipRuntime::mem_release(self, handle) }
    }
}

/// Retired VA a process may accumulate before new reservations are refused.
/// User VA is 47 bits (128 TiB); one model load reserves tens of GiB of KV.
const RETIRED_VA_CAP_BYTES: usize = 64 << 40;

/// Virtual ranges released by [`VmmArena::release`] but kept reserved.
struct RetiredVa {
    ranges: usize,
    bytes: usize,
}

impl RetiredVa {
    const fn new() -> Self {
        Self {
            ranges: 0,
            bytes: 0,
        }
    }

    fn retire(&mut self, bytes: usize) {
        self.ranges += 1;
        self.bytes += bytes;
    }

    fn admit(&self, requested_bytes: usize, cap_bytes: usize) -> HipResult<()> {
        if self.bytes.saturating_add(requested_bytes) > cap_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "VMM reserve of {requested_bytes} bytes refused: {} bytes of virtual address \
                     space in {} ranges are retired by earlier unloads (cap {cap_bytes}); restart \
                     the process to reclaim it",
                    self.bytes, self.ranges
                ),
            ));
        }
        Ok(())
    }
}

static RETIRED_VA: Mutex<RetiredVa> = Mutex::new(RetiredVa::new());

fn retired_va() -> std::sync::MutexGuard<'static, RetiredVa> {
    RETIRED_VA
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

/// Bytes of virtual address space retired (kept reserved) by released arenas
/// in this process.
pub fn retired_va_bytes() -> usize {
    retired_va().bytes
}

/// Physical granule granularity floor: handles are at least this large so a
/// 2 MiB-granularity device and a 4 KiB-granularity device share one policy.
const GRANULE_MIN_BYTES: usize = 2 << 20;

/// Generation stamped on every granule table entry. A granule's HIP handle is
/// never replaced, so the stamp is constant; lookups still compare it so an id
/// from a differently stamped entry can never alias a live one.
const GRANULE_GENERATION: u64 = 1;

/// Process-wide identity of one physical VMM granule (one HIP allocation
/// handle). `id` is monotonic and never reused within a process.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VmmPhysicalId {
    pub id: u64,
    pub generation: u64,
}

struct GranuleEntry {
    /// `hipMemGenericAllocationHandle_t` stored as an address so the table is `Send`.
    handle: usize,
    size: usize,
    /// Mapping leases (one per mapped granule segment) plus cache leases
    /// (one per [`VmmSharedPrefix`] granule). Zero means the handle release
    /// failed and is pending a retry.
    leases: usize,
    generation: u64,
}

/// Process-global physical granule table: id -> handle, size and lease count.
struct GranuleTable {
    next_id: u64,
    entries: BTreeMap<u64, GranuleEntry>,
}

impl GranuleTable {
    const fn new() -> Self {
        Self {
            next_id: 1,
            entries: BTreeMap::new(),
        }
    }

    /// Record a freshly created handle holding its first lease.
    fn register(&mut self, handle: usize, size: usize) -> VmmPhysicalId {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(
            id,
            GranuleEntry {
                handle,
                size,
                leases: 1,
                generation: GRANULE_GENERATION,
            },
        );
        VmmPhysicalId {
            id,
            generation: GRANULE_GENERATION,
        }
    }

    fn entry(&self, id: VmmPhysicalId) -> Option<&GranuleEntry> {
        self.entries
            .get(&id.id)
            .filter(|entry| entry.generation == id.generation)
    }

    fn leases(&self, id: VmmPhysicalId) -> Option<usize> {
        self.entry(id).map(|entry| entry.leases)
    }

    fn handle(&self, id: VmmPhysicalId) -> Option<usize> {
        self.entry(id).map(|entry| entry.handle)
    }

    fn live_bytes(&self) -> usize {
        self.entries.values().map(|entry| entry.size).sum()
    }

    fn check_leasable(&self, id: VmmPhysicalId) -> HipResult<()> {
        match self.entry(id) {
            Some(entry) if entry.leases > 0 => Ok(()),
            Some(_) => Err(HipError::new(
                0,
                &format!("VMM granule {} has no live lease (release pending)", id.id),
            )),
            None => Err(HipError::new(
                0,
                &format!("VMM granule {} is not in the granule table", id.id),
            )),
        }
    }

    /// Take one more lease on `id` and return its handle address.
    fn add_lease(&mut self, id: VmmPhysicalId) -> HipResult<usize> {
        self.check_leasable(id)?;
        let entry = self
            .entries
            .get_mut(&id.id)
            .ok_or_else(|| HipError::new(0, "VMM granule table entry vanished"))?;
        entry.leases += 1;
        Ok(entry.handle)
    }

    /// Take one more lease on every id, or none if any id is not leasable.
    fn add_leases(&mut self, ids: &[VmmPhysicalId]) -> HipResult<()> {
        for &id in ids {
            self.check_leasable(id)?;
        }
        for &id in ids {
            self.add_lease(id)?;
        }
        Ok(())
    }

    /// Drop one lease. At zero leases `release` runs on the handle; a failed
    /// release leaves the entry at zero leases (pending, still counted live).
    fn drop_lease(
        &mut self,
        id: VmmPhysicalId,
        release: &mut impl FnMut(usize) -> HipResult<()>,
    ) -> HipResult<()> {
        self.check_leasable(id)?;
        let entry = self
            .entries
            .get_mut(&id.id)
            .ok_or_else(|| HipError::new(0, "VMM granule table entry vanished"))?;
        entry.leases -= 1;
        if entry.leases > 0 {
            return Ok(());
        }
        release(entry.handle)?;
        self.entries.remove(&id.id);
        Ok(())
    }

    /// Retry every pending (zero-lease) release. Returns how many were
    /// released and the first failure, if any.
    fn retry_pending(
        &mut self,
        release: &mut impl FnMut(usize) -> HipResult<()>,
    ) -> (usize, Option<HipError>) {
        let pending: Vec<u64> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.leases == 0)
            .map(|(&id, _)| id)
            .collect();
        let mut released = 0;
        let mut first_error = None;
        for id in pending {
            let Some(entry) = self.entries.get(&id) else {
                continue;
            };
            match release(entry.handle) {
                Ok(()) => {
                    self.entries.remove(&id);
                    released += 1;
                }
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }
        (released, first_error)
    }
}

static GRANULES: Mutex<GranuleTable> = Mutex::new(GranuleTable::new());

fn granule_table() -> std::sync::MutexGuard<'static, GranuleTable> {
    GRANULES.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// Process-wide bytes of physical VMM granules that still hold at least one
/// lease or a pending (failed) `hipMemRelease`.
pub fn vmm_live_granule_bytes() -> usize {
    granule_table().live_bytes()
}

/// Retry every `hipMemRelease` that failed when a granule's last lease was
/// dropped. Returns the number of handles released; if any retry fails the
/// first error is returned (successful retries stay released).
pub fn retry_pending_granule_releases(hip: &HipRuntime) -> HipResult<usize> {
    let (released, first_error) =
        granule_table().retry_pending(&mut |handle| release_granule_handle(hip, handle));
    match first_error {
        Some(err) => Err(err),
        None => Ok(released),
    }
}

fn release_granule_handle(hip: &impl VmmDriver, handle: usize) -> HipResult<()> {
    if let Some(err) = take_fault(VmmFaultKind::Release) {
        return Err(err);
    }
    // SAFETY: `handle` was created by `mem_create` for a granule table entry
    // whose leases reached zero; every mapping lease is dropped only after its
    // range was unmapped, and shared-prefix leases never map, so no mapped
    // range still references it.
    unsafe { hip.mem_release(handle as HipMemGenericAllocationHandle) }
}

/// Drop one lease per id, continuing past failures; returns the first error.
fn drop_leases(
    table: &mut GranuleTable,
    ids: impl IntoIterator<Item = VmmPhysicalId>,
    release: &mut impl FnMut(usize) -> HipResult<()>,
) -> HipResult<()> {
    let mut first_error = None;
    for id in ids {
        if let Err(err) = table.drop_lease(id, release) {
            if first_error.is_none() {
                first_error = Some(err);
            }
        }
    }
    match first_error {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

/// Cache leases on the leading granules of a [`VmmArena`], taken by
/// [`VmmArena::share_prefix`]. The physical memory stays alive until every
/// lease (this one and each arena mapping) is dropped; the prefix itself maps
/// nothing and exposes no raw handles.
#[must_use = "shared VMM prefixes must be released explicitly"]
#[derive(Debug)]
pub struct VmmSharedPrefix {
    granules: Vec<(VmmPhysicalId, usize)>,
    owner_device: i32,
    granularity: usize,
}

impl VmmSharedPrefix {
    /// A prefix with zero granules; releasing it is a no-op.
    pub fn empty() -> Self {
        Self {
            granules: Vec::new(),
            owner_device: -1,
            granularity: 0,
        }
    }

    /// Sum of the granule sizes.
    pub fn bytes(&self) -> usize {
        self.granules.iter().map(|&(_, size)| size).sum()
    }

    /// `(id, physical bytes)` of every granule in offset order.
    pub fn granules(&self) -> &[(VmmPhysicalId, usize)] {
        &self.granules
    }

    /// Drop one cache lease per granule. A granule whose last lease this was
    /// is released unless a mapping still holds a lease; a failed release
    /// stays pending for [`retry_pending_granule_releases`]. Continues past
    /// failures and returns the first.
    pub fn release(mut self, hip: &HipRuntime) -> HipResult<()> {
        let granules = std::mem::take(&mut self.granules);
        drop_leases(
            &mut granule_table(),
            granules.into_iter().map(|(id, _)| id),
            &mut |handle| release_granule_handle(hip, handle),
        )
    }
}

impl Drop for VmmSharedPrefix {
    fn drop(&mut self) {
        // Dropping cannot call HIP (no runtime, possibly a foreign thread), so
        // the leases are leaked and stay counted by `vmm_live_granule_bytes`.
        if cfg!(debug_assertions) && !self.granules.is_empty() {
            eprintln!(
                "hip-bridge: VmmSharedPrefix with {} granules ({} bytes) dropped without release(); \
                 its leases are leaked",
                self.granules.len(),
                self.bytes()
            );
        }
    }
}

/// Deterministic, test-only fault injection for VMM teardown/access stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmmFaultKind {
    /// Fail the next `hipMemSetAccess` call(s).
    AccessReset,
    /// Fail the next `hipMemUnmap` call(s).
    Unmap,
    /// Fail the next physical-handle `hipMemRelease` call(s).
    Release,
    /// Fail the next granule `hipMemMap` call(s) in `map_next_granules` /
    /// `map_shared_prefix`.
    Map,
    /// Fail the next granule `hipMemCreate` call(s) in `map_next_granules`.
    Create,
}

thread_local! {
    static FAULT_ACCESS: Cell<u32> = const { Cell::new(0) };
    static FAULT_UNMAP: Cell<u32> = const { Cell::new(0) };
    static FAULT_RELEASE: Cell<u32> = const { Cell::new(0) };
    static FAULT_MAP: Cell<u32> = const { Cell::new(0) };
    static FAULT_CREATE: Cell<u32> = const { Cell::new(0) };
}

/// Queue `count` deterministic failures for `kind`. Test-only; real HIP is
/// unchanged when the counter is zero.
pub fn inject_vmm_fault(kind: VmmFaultKind, count: u32) {
    match kind {
        VmmFaultKind::AccessReset => FAULT_ACCESS.with(|c| c.set(count)),
        VmmFaultKind::Unmap => FAULT_UNMAP.with(|c| c.set(count)),
        VmmFaultKind::Release => FAULT_RELEASE.with(|c| c.set(count)),
        VmmFaultKind::Map => FAULT_MAP.with(|c| c.set(count)),
        VmmFaultKind::Create => FAULT_CREATE.with(|c| c.set(count)),
    }
}

/// Clear every pending injected VMM fault.
pub fn clear_vmm_faults() {
    FAULT_ACCESS.with(|c| c.set(0));
    FAULT_UNMAP.with(|c| c.set(0));
    FAULT_RELEASE.with(|c| c.set(0));
    FAULT_MAP.with(|c| c.set(0));
    FAULT_CREATE.with(|c| c.set(0));
}

fn take_fault(kind: VmmFaultKind) -> Option<HipError> {
    let cell = match kind {
        VmmFaultKind::AccessReset => &FAULT_ACCESS,
        VmmFaultKind::Unmap => &FAULT_UNMAP,
        VmmFaultKind::Release => &FAULT_RELEASE,
        VmmFaultKind::Map => &FAULT_MAP,
        VmmFaultKind::Create => &FAULT_CREATE,
    };
    cell.with(|c| {
        let left = c.get();
        if left == 0 {
            return None;
        }
        c.set(left - 1);
        let label = match kind {
            VmmFaultKind::AccessReset => "access-reset",
            VmmFaultKind::Unmap => "unmap",
            VmmFaultKind::Release => "release",
            VmmFaultKind::Map => "map",
            VmmFaultKind::Create => "create",
        };
        Some(HipError::new(
            0x564D_4D46, // 'VMMF'
            &format!("injected VMM {label} failure"),
        ))
    })
}

/// Lifecycle/protection class of a [`VmmArena`]. `Default` is the unchanged
/// private (KV/weights) arena; the others only come from the shared-weight
/// constructors and reject the multi-mapping/granule-sharing APIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VmmArenaKind {
    /// Ordinary private read-write arena.
    Default,
    /// Exportable (POSIX FD) arena still under construction: RW, single physical allocation.
    ExportBuilding,
    /// Exportable arena after [`VmmArena::publish_read_only`]: owner access is RO.
    ExportPublished,
    /// Arena mapping an imported POSIX-FD allocation: always RO.
    #[cfg_attr(not(unix), allow(dead_code))]
    ImportedReadOnly,
}

fn allocation_prop(kind: VmmArenaKind, device: i32) -> HipMemAllocationProp {
    match kind {
        VmmArenaKind::Default => HipMemAllocationProp::device_pinned(device),
        _ => HipMemAllocationProp::device_pinned_posix_fd(device),
    }
}

#[derive(Debug)]
struct VmmSegment {
    offset: usize,
    size: usize,
    handle: Option<HipMemGenericAllocationHandle>,
    /// Physical granule backing this segment; its mapping lease is held in the
    /// granule table. Exclusive segments (`map_next`) use `handle` instead.
    granule: Option<VmmPhysicalId>,
    mapped: bool,
}

#[must_use = "VMM arenas must be explicitly released with VmmArena::release"]
pub struct VmmArena {
    base: *mut c_void,
    owner_device: i32,
    granularity: usize,
    reserved_bytes: usize,
    mapped_bytes: usize,
    segments: Vec<VmmSegment>,
    access_devices: Vec<i32>,
    releasing: bool,
    /// True once any mapping succeeded; a never-mapped VA may adopt a shared prefix.
    ever_mapped: bool,
    kind: VmmArenaKind,
}

// SAFETY: VmmArena holds a device VA base and opaque HIP allocation handles that
// may move with model state across threads. Concurrent mutation is excluded
// because VmmArena is not Sync; callers must not free/unmap while another thread
// still has in-flight work on the mapped prefix.
unsafe impl Send for VmmArena {}

impl VmmArena {
    pub fn reserve(
        hip: &impl VmmDriver,
        owner_device: i32,
        requested_bytes: usize,
    ) -> HipResult<Self> {
        Self::reserve_kind(
            hip,
            owner_device,
            requested_bytes,
            VmmArenaKind::Default,
            false,
        )
    }

    /// Reserve an arena whose single physical allocation is created exportable
    /// as a POSIX FD (`requestedHandleTypes = POSIX_FD`). The caller maps it
    /// with exactly one [`Self::map_next`] of [`Self::reserved_bytes`] (owner
    /// device access only; a second mapping is refused), fills it while the
    /// arena is read-write, synchronizes, calls [`Self::publish_read_only`],
    /// then [`Self::export_posix`]. Retirement/release behave like [`Self::reserve`].
    pub fn reserve_exportable(hip: &HipRuntime, device: i32, bytes: usize) -> HipResult<Self> {
        Self::reserve_kind(hip, device, bytes, VmmArenaKind::ExportBuilding, false)
    }

    /// Import a POSIX-FD physical allocation exported by [`Self::export_posix`]
    /// and map it read-only at offset zero of a fresh reservation on `device`
    /// (the owner device; no peer access). `physical_bytes` must be the
    /// exporter's physical size, a non-zero multiple of this device's
    /// allocation granularity. `fd` is borrowed: it is not consumed or closed.
    /// The arena is read-only for its whole life and must be released with
    /// [`Self::release`]. On failure nothing is returned to retry: the VA is
    /// retired, and a handle whose release failed stays in the granule table
    /// pending [`retry_pending_granule_releases`].
    #[cfg(unix)]
    pub fn import_posix(
        hip: &HipRuntime,
        device: i32,
        fd: BorrowedFd<'_>,
        physical_bytes: usize,
    ) -> HipResult<Self> {
        // Emu: refuse an over-cap import before the (never-freed) VA reservation
        // is consumed. The authoritative reserve/commit happens in
        // `mem_import_posix_charged`, which also keeps the token with the handle
        // until `mem_release` succeeds.
        if let Some(budget) = crate::emu_budget::active() {
            crate::emu_budget::precheck(&budget, crate::emu_budget::Class::Device, physical_bytes)?;
        }
        let mut arena = Self::reserve_kind(
            hip,
            device,
            physical_bytes,
            VmmArenaKind::ImportedReadOnly,
            true,
        )?;
        let handle = match hip.mem_import_posix_charged(fd, physical_bytes) {
            Ok(handle) => handle,
            Err(err) => return Err(arena.abort_import(hip, None, false, err)),
        };
        let base = arena.base;
        let size = arena.reserved_bytes;
        let mapped = take_fault(VmmFaultKind::Map).map_or_else(
            // SAFETY: `base..+size` is the fresh, unmapped reservation of this
            // arena; `size` equals the imported allocation's physical size
            // (caller contract) and `handle` is a live imported handle owned here.
            || unsafe { hip.mem_map(base, size, handle) },
            Err,
        );
        if let Err(err) = mapped {
            return Err(arena.abort_import(hip, Some(handle), false, err));
        }
        let access = [HipMemAccessDesc::read_only_device(device)];
        let access_set = take_fault(VmmFaultKind::AccessReset).map_or_else(
            // SAFETY: `base..+size` was just mapped above; the only descriptor
            // is the owner device, read-only.
            || unsafe { hip.mem_set_access(base, size, &access) },
            Err,
        );
        if let Err(err) = access_set {
            return Err(arena.abort_import(hip, Some(handle), true, err));
        }
        arena.segments.push(VmmSegment {
            offset: 0,
            size,
            handle: Some(handle),
            granule: None,
            mapped: true,
        });
        arena.mapped_bytes = size;
        arena.ever_mapped = true;
        Ok(arena)
    }

    /// Failure cleanup for [`Self::import_posix`]: undo what was done, retire
    /// the reservation (VA is never freed), and keep any handle that could not
    /// be released owned by the granule table. Returns the error to report.
    #[cfg(unix)]
    fn abort_import(
        mut self,
        hip: &HipRuntime,
        handle: Option<HipMemGenericAllocationHandle>,
        mapped: bool,
        err: HipError,
    ) -> HipError {
        let size = self.reserved_bytes;
        let mut cleanup: Option<HipError> = None;
        if let Some(handle) = handle {
            let mut unmapped = !mapped;
            if mapped {
                let result = match take_fault(VmmFaultKind::Unmap) {
                    Some(fault) => Err(fault),
                    // SAFETY: `base..+size` was mapped by `import_posix` and
                    // never used by a kernel (arena not yet returned).
                    None => unsafe { hip.mem_unmap(self.base, size) },
                };
                match result {
                    Ok(()) => unmapped = true,
                    Err(unmap_err) => cleanup = Some(unmap_err),
                }
            }
            if unmapped {
                let result = match take_fault(VmmFaultKind::Release) {
                    Some(fault) => Err(fault),
                    // SAFETY: no mapped range references `handle`; it is owned here.
                    None => unsafe { hip.mem_release(handle) },
                };
                if let Err(release_err) = result {
                    // Pending release: counted live, retried by
                    // `retry_pending_granule_releases`.
                    let mut table = granule_table();
                    let id = table.register(handle as usize, size);
                    let _ = table.drop_lease(id, &mut |_| Err(release_err.clone()));
                    cleanup = Some(release_err);
                }
            } else {
                // Still mapped: cannot release; leaked but accounted as live.
                granule_table().register(handle as usize, size);
            }
        }
        retired_va().retire(self.reserved_bytes);
        self.base = std::ptr::null_mut();
        self.reserved_bytes = 0;
        match cleanup {
            None => err,
            Some(cleanup) => HipError::new(
                0,
                &format!("{err}; imported VMM cleanup also failed: {cleanup}; reservation retired"),
            ),
        }
    }

    /// Shared reserve path. `exact_granules` additionally requires
    /// `requested_bytes` to be a non-zero multiple of the allocation
    /// granularity (imports map a whole physical allocation).
    fn reserve_kind(
        hip: &impl VmmDriver,
        owner_device: i32,
        requested_bytes: usize,
        kind: VmmArenaKind,
        exact_granules: bool,
    ) -> HipResult<Self> {
        if requested_bytes == 0 {
            return Err(HipError::new(
                0,
                "VMM reserve size must be greater than zero",
            ));
        }
        let count = hip.device_count()?;
        if owner_device < 0 || owner_device >= count {
            return Err(HipError::new(
                0,
                &format!("VMM owner device {owner_device} is outside available range 0..{count}"),
            ));
        }

        hip.set_device(owner_device)?;
        let prop = allocation_prop(kind, owner_device);
        let granularity_option = match kind {
            VmmArenaKind::Default => HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED,
            // Exportable/imported owners use minimum rounding so small owners
            // do not pay recommended-granularity padding.
            _ => HIP_MEM_ALLOCATION_GRANULARITY_MINIMUM,
        };
        let granularity = hip.mem_get_allocation_granularity(&prop, granularity_option)?;
        if granularity == 0 {
            return Err(HipError::new(
                0,
                "HIP returned zero VMM allocation granularity",
            ));
        }
        if exact_granules && !requested_bytes.is_multiple_of(granularity) {
            return Err(HipError::new(
                0,
                &format!(
                    "VMM physical size {requested_bytes} must be a multiple of granularity {granularity}"
                ),
            ));
        }
        let reserved_bytes = round_up(requested_bytes, granularity)?;
        retired_va().admit(reserved_bytes, RETIRED_VA_CAP_BYTES)?;
        let base = hip.mem_address_reserve(reserved_bytes, granularity)?;

        Ok(Self {
            base,
            owner_device,
            granularity,
            reserved_bytes,
            mapped_bytes: 0,
            segments: Vec::new(),
            access_devices: vec![owner_device],
            releasing: false,
            ever_mapped: false,
            kind,
        })
    }

    pub const fn owner_device(&self) -> i32 {
        self.owner_device
    }

    pub const fn granularity(&self) -> usize {
        self.granularity
    }

    pub const fn reserved_bytes(&self) -> usize {
        self.reserved_bytes
    }

    /// The primary physical allocation handle backing this arena, if any mapped
    /// segment exists. Exposed so callers can query the handle's placement with
    /// `HipRuntime::mem_get_handle_properties` (fail-closed host-located check).
    pub fn primary_handle(&self) -> Option<HipMemGenericAllocationHandle> {
        self.segments.first().and_then(|seg| {
            seg.handle.or_else(|| {
                seg.granule.and_then(|id| {
                    granule_table()
                        .handle(id)
                        .map(|handle| handle as HipMemGenericAllocationHandle)
                })
            })
        })
    }
    pub const fn mapped_bytes(&self) -> usize {
        self.mapped_bytes
    }

    pub fn base_address(&self) -> usize {
        self.base as usize
    }

    pub fn is_released(&self) -> bool {
        self.base.is_null()
    }

    pub fn map_next(
        &mut self,
        hip: &impl VmmDriver,
        size: usize,
        access_devices: &[i32],
    ) -> HipResult<()> {
        if self.releasing || self.is_released() {
            return Err(HipError::new(
                0,
                "VMM arena is releasing or already released",
            ));
        }
        match self.kind {
            VmmArenaKind::Default => {}
            VmmArenaKind::ExportBuilding => {
                if !self.segments.is_empty() {
                    return Err(HipError::new(
                        0,
                        "exportable VMM arena holds exactly one physical allocation",
                    ));
                }
                if access_devices
                    .iter()
                    .any(|&device| device != self.owner_device)
                {
                    return Err(HipError::new(
                        0,
                        "exportable VMM arena grants access to its owner device only",
                    ));
                }
            }
            VmmArenaKind::ExportPublished | VmmArenaKind::ImportedReadOnly => {
                return Err(HipError::new(
                    0,
                    "read-only (published or imported) VMM arena cannot be remapped",
                ));
            }
        }
        if size == 0 || !size.is_multiple_of(self.granularity) {
            return Err(HipError::new(
                0,
                &format!(
                    "VMM map size {size} must be a non-zero multiple of granularity {}",
                    self.granularity
                ),
            ));
        }
        let next_mapped = self
            .mapped_bytes
            .checked_add(size)
            .ok_or_else(|| HipError::new(0, "VMM mapped byte count overflowed"))?;
        if next_mapped > self.reserved_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "VMM map would exceed reserve: {} + {size} > {}",
                    self.mapped_bytes, self.reserved_bytes
                ),
            ));
        }

        let count = hip.device_count()?;
        let mut devices = Vec::with_capacity(access_devices.len() + 1);
        devices.push(self.owner_device);
        for &device in access_devices {
            if device < 0 || device >= count {
                return Err(HipError::new(
                    0,
                    &format!("VMM access device {device} is outside available range 0..{count}"),
                ));
            }
            if device != self.owner_device && !hip.can_access_peer(device, self.owner_device)? {
                return Err(HipError::new(
                    0,
                    &format!(
                        "VMM access device {device} cannot access owner device {}",
                        self.owner_device
                    ),
                ));
            }
            if !devices.contains(&device) {
                devices.push(device);
            }
        }
        let mut next_access_devices = self.access_devices.clone();
        for device in devices {
            if !next_access_devices.contains(&device) {
                next_access_devices.push(device);
            }
        }
        let access: Vec<_> = next_access_devices
            .iter()
            .copied()
            .map(HipMemAccessDesc::read_write_device)
            .collect();

        hip.set_device(self.owner_device)?;
        let prop = allocation_prop(self.kind, self.owner_device);
        let handle = hip.mem_create(size, &prop)?;
        let address = offset_ptr(self.base, self.mapped_bytes);
        // SAFETY: `address` is base+mapped_bytes within the live reserved VA;
        // `size` is a granularity multiple and fits the remaining reserve;
        // `handle` is a fresh owned allocation covering `size`, not yet mapped.
        if let Err(err) = unsafe { hip.mem_map(address, size, handle) } {
            // SAFETY: map failed so no range references `handle`; release is exclusive.
            return match unsafe { hip.mem_release(handle) } {
                Ok(()) => Err(err),
                Err(cleanup) => {
                    self.segments.push(VmmSegment {
                        offset: self.mapped_bytes,
                        size,
                        handle: Some(handle),
                        granule: None,
                        mapped: false,
                    });
                    self.releasing = true;
                    Err(combined_cleanup_error(err, cleanup))
                }
            };
        }
        // ROCm 7.2 on gfx1100 accepts 4 KiB allocation granularity but rejects
        // hipMemSetAccess when a later subrange begins at some otherwise-valid
        // 4 KiB offsets (for example base+16 KiB). Reapplying access from the
        // reservation base over the contiguous mapped prefix is accepted and
        // also ensures newly-added peer devices gain access to older segments.
        // With >= 2 MiB granularity and no new peer device only the new range
        // is set (`access_start`).
        let access_start = self.access_start(&next_access_devices, self.mapped_bytes);
        if let Err(err) = take_fault(VmmFaultKind::AccessReset).map_or_else(
            // SAFETY: `access_start..next_mapped` is mapped (just extended by
            // mem_map; `access_start <= mapped_bytes`); `access` lists only
            // owner/peer devices already validated for peer access.
            || unsafe {
                hip.mem_set_access(
                    offset_ptr(self.base, access_start),
                    next_mapped - access_start,
                    &access,
                )
            },
            Err,
        ) {
            let err = HipError {
                code: err.code,
                message: format!(
                    "{}; VMM access range base=0x{:x} offset={} size={} (new segment address=0x{:x} offset={} size={}) granularity={}",
                    err.message,
                    self.base as usize,
                    access_start,
                    next_mapped - access_start,
                    address as usize,
                    self.mapped_bytes,
                    size,
                    self.granularity,
                ),
                context: err.context,
            };
            let mut segment = VmmSegment {
                offset: self.mapped_bytes,
                size,
                handle: Some(handle),
                granule: None,
                mapped: true,
            };
            // SAFETY: `address,size` is the segment just mapped; no kernels are
            // scheduled on it yet (map_next is pre-use). Unmap before release.
            let cleanup_error = match unsafe { hip.mem_unmap(address, size) } {
                Ok(()) => {
                    segment.mapped = false;
                    // SAFETY: unmapped so no mapped range still references handle.
                    match unsafe { hip.mem_release(handle) } {
                        Ok(()) => {
                            segment.handle = None;
                            None
                        }
                        Err(cleanup) => Some(cleanup),
                    }
                }
                Err(cleanup) => Some(cleanup),
            };
            // Poison the arena even when cleanup succeeded: a retried map_next
            // would map a new handle at `address`, which the GPU may still
            // translate to the handle released above.
            self.releasing = true;
            return match cleanup_error {
                None => Err(err),
                Some(cleanup) => {
                    self.segments.push(segment);
                    Err(combined_cleanup_error(err, cleanup))
                }
            };
        }

        // Commit newly requested peer permissions only after the driver has
        // accepted them.
        self.access_devices = next_access_devices;
        self.segments.push(VmmSegment {
            offset: self.mapped_bytes,
            size,
            handle: Some(handle),
            granule: None,
            mapped: true,
        });
        self.mapped_bytes = next_mapped;
        self.ever_mapped = true;
        Ok(())
    }

    /// Bytes of one physical granule: `max(granularity, 2 MiB)` rounded up to
    /// the granularity. Only the last granule at the reserve end may be smaller.
    pub fn granule_bytes(&self) -> usize {
        GRANULE_MIN_BYTES
            .max(self.granularity)
            .div_ceil(self.granularity)
            .saturating_mul(self.granularity)
    }

    /// End offset of the leading run of mapped granule segments: the bytes
    /// that can be aliased into another arena by [`Self::share_prefix`].
    pub fn granule_prefix_bytes(&self) -> usize {
        let mut end = 0;
        for segment in &self.segments {
            if segment.granule.is_none() || !segment.mapped || segment.offset != end {
                break;
            }
            end = segment.offset + segment.size;
        }
        end
    }

    /// End offset of the last mapped granule segment whose physical granule
    /// has more than one lease (shared with a checkpoint or another arena);
    /// zero when nothing is shared. Bytes below it must never be written.
    pub fn sealed_bytes(&self) -> usize {
        let table = granule_table();
        self.segments
            .iter()
            .rev()
            .find(|segment| {
                segment.mapped
                    && segment
                        .granule
                        .is_some_and(|id| table.leases(id).is_some_and(|leases| leases > 1))
            })
            .map_or(0, |segment| segment.offset + segment.size)
    }

    /// `(id, bytes)` of every mapped granule segment in offset order.
    pub fn physical_granules(&self) -> Vec<(VmmPhysicalId, usize)> {
        self.segments
            .iter()
            .filter(|segment| segment.mapped)
            .filter_map(|segment| segment.granule.map(|id| (id, segment.size)))
            .collect()
    }

    /// Take one cache lease on every granule of the leading `bytes`, which
    /// must be a granule boundary no greater than [`Self::granule_prefix_bytes`].
    /// Zero returns an empty prefix.
    pub fn share_prefix(&self, bytes: usize) -> HipResult<VmmSharedPrefix> {
        if self.releasing || self.is_released() {
            return Err(HipError::new(
                0,
                "VMM arena is releasing or already released",
            ));
        }
        if self.kind != VmmArenaKind::Default {
            return Err(HipError::new(
                0,
                "exportable/imported VMM arenas do not support granule sharing",
            ));
        }
        if bytes == 0 {
            return Ok(VmmSharedPrefix::empty());
        }
        let limit = self.granule_prefix_bytes();
        if bytes > limit {
            return Err(HipError::new(
                0,
                &format!("VMM share of {bytes} bytes exceeds the granule prefix of {limit} bytes"),
            ));
        }
        let mut granules = Vec::new();
        let mut end = 0;
        for segment in &self.segments {
            let Some(id) = segment.granule else { break };
            if end >= bytes {
                break;
            }
            end = segment.offset + segment.size;
            granules.push((id, segment.size));
        }
        if end != bytes {
            return Err(HipError::new(
                0,
                &format!("VMM share of {bytes} bytes does not end on a granule boundary"),
            ));
        }
        let ids: Vec<VmmPhysicalId> = granules.iter().map(|&(id, _)| id).collect();
        granule_table().add_leases(&ids)?;
        Ok(VmmSharedPrefix {
            granules,
            owner_device: self.owner_device,
            granularity: self.granularity,
        })
    }

    /// Map `bytes` (a non-zero multiple of the granularity that fits the
    /// reserve) as one physical granule per [`Self::granule_bytes`] (the last
    /// may be shorter), contiguously after the mapped prefix, with one
    /// `hipMemSetAccess` over the whole mapped prefix. Any failure unmaps and
    /// drops the leases created by this call and poisons the arena.
    pub fn map_next_granules(
        &mut self,
        hip: &HipRuntime,
        bytes: usize,
        access_devices: &[i32],
    ) -> HipResult<()> {
        self.ensure_mappable()?;
        if bytes == 0 || !bytes.is_multiple_of(self.granularity) {
            return Err(HipError::new(
                0,
                &format!(
                    "VMM granule map size {bytes} must be a non-zero multiple of granularity {}",
                    self.granularity
                ),
            ));
        }
        let next_mapped = self
            .mapped_bytes
            .checked_add(bytes)
            .ok_or_else(|| HipError::new(0, "VMM mapped byte count overflowed"))?;
        if next_mapped > self.reserved_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "VMM map would exceed reserve: {} + {bytes} > {}",
                    self.mapped_bytes, self.reserved_bytes
                ),
            ));
        }
        let (next_access_devices, access) = self.resolve_access(hip, access_devices)?;

        hip.set_device(self.owner_device)?;
        let prop = HipMemAllocationProp::device_pinned(self.owner_device);
        let granule = self.granule_bytes();
        let mut staged: Vec<VmmSegment> = Vec::with_capacity(bytes.div_ceil(granule));
        let mut offset = self.mapped_bytes;
        while offset < next_mapped {
            let size = (next_mapped - offset).min(granule);
            let handle = match take_fault(VmmFaultKind::Create)
                .map_or_else(|| hip.mem_create(size, &prop), Err)
            {
                Ok(handle) => handle,
                Err(err) => return Err(self.abort_granule_map(hip, staged, err)),
            };
            let id = granule_table().register(handle as usize, size);
            staged.push(VmmSegment {
                offset,
                size,
                handle: None,
                granule: Some(id),
                mapped: false,
            });
            let address = offset_ptr(self.base, offset);
            let mapped = take_fault(VmmFaultKind::Map).map_or_else(
                // SAFETY: `address` is base+offset inside the live reserve and
                // `offset + size <= next_mapped <= reserved_bytes`; `size` is a
                // granularity multiple; `handle` is a fresh granule covering
                // `size`, not mapped anywhere.
                || unsafe { hip.mem_map(address, size, handle) },
                Err,
            );
            if let Err(err) = mapped {
                return Err(self.abort_granule_map(hip, staged, err));
            }
            if let Some(last) = staged.last_mut() {
                last.mapped = true;
            }
            offset += size;
        }
        if let Err(err) = self.apply_access(
            hip,
            next_mapped,
            &access,
            &next_access_devices,
            self.mapped_bytes,
            bytes,
        ) {
            return Err(self.abort_granule_map(hip, staged, err));
        }

        self.access_devices = next_access_devices;
        self.segments.append(&mut staged);
        self.mapped_bytes = next_mapped;
        self.ever_mapped = true;
        Ok(())
    }

    /// Map every granule of `prefix` at `base + offset` of this never-mapped
    /// arena (one mapping lease per granule, one `hipMemSetAccess`). The
    /// destination must share the prefix's owner device and granularity. Any
    /// failure unmaps what it mapped, drops those leases and poisons the arena.
    pub fn map_shared_prefix(
        &mut self,
        hip: &HipRuntime,
        prefix: &VmmSharedPrefix,
        access_devices: &[i32],
    ) -> HipResult<()> {
        self.ensure_mappable()?;
        if self.ever_mapped || self.mapped_bytes != 0 || !self.segments.is_empty() {
            return Err(HipError::new(
                0,
                "VMM shared prefix needs a never-mapped arena (its VA must not have been translated)",
            ));
        }
        if prefix.granules.is_empty() {
            return Ok(());
        }
        if prefix.owner_device != self.owner_device || prefix.granularity != self.granularity {
            return Err(HipError::new(
                0,
                &format!(
                    "VMM shared prefix (device {}, granularity {}) does not match arena (device {}, granularity {})",
                    prefix.owner_device, prefix.granularity, self.owner_device, self.granularity
                ),
            ));
        }
        let prefix_bytes = prefix.bytes();
        if prefix_bytes > self.reserved_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "VMM shared prefix of {prefix_bytes} bytes exceeds reserve {}",
                    self.reserved_bytes
                ),
            ));
        }
        let (next_access_devices, access) = self.resolve_access(hip, access_devices)?;

        hip.set_device(self.owner_device)?;
        let mut staged: Vec<VmmSegment> = Vec::with_capacity(prefix.granules.len());
        let mut offset = 0;
        for &(id, size) in &prefix.granules {
            // Take the lease before matching: the table guard must not be held
            // while `abort_granule_map` re-locks the table.
            let leased = granule_table().add_lease(id);
            let handle = match leased {
                Ok(handle) => handle as HipMemGenericAllocationHandle,
                Err(err) => return Err(self.abort_granule_map(hip, staged, err)),
            };
            staged.push(VmmSegment {
                offset,
                size,
                handle: None,
                granule: Some(id),
                mapped: false,
            });
            let address = offset_ptr(self.base, offset);
            let mapped = take_fault(VmmFaultKind::Map).map_or_else(
                // SAFETY: `address` is base+offset inside the live reserve and
                // `offset + size <= prefix_bytes <= reserved_bytes`; the arena
                // was never mapped so this VA range is unmapped; `handle` is a
                // live granule (a lease was just taken) of exactly `size` bytes.
                || unsafe { hip.mem_map(address, size, handle) },
                Err,
            );
            if let Err(err) = mapped {
                return Err(self.abort_granule_map(hip, staged, err));
            }
            if let Some(last) = staged.last_mut() {
                last.mapped = true;
            }
            offset += size;
        }
        if let Err(err) = self.apply_access(
            hip,
            prefix_bytes,
            &access,
            &next_access_devices,
            0,
            prefix_bytes,
        ) {
            return Err(self.abort_granule_map(hip, staged, err));
        }

        self.access_devices = next_access_devices;
        self.segments.append(&mut staged);
        self.mapped_bytes = prefix_bytes;
        self.ever_mapped = true;
        Ok(())
    }

    fn ensure_mappable(&self) -> HipResult<()> {
        if self.releasing || self.is_released() {
            return Err(HipError::new(
                0,
                "VMM arena is releasing or already released",
            ));
        }
        if self.kind != VmmArenaKind::Default {
            return Err(HipError::new(
                0,
                "exportable/imported VMM arenas do not support granule mapping",
            ));
        }
        Ok(())
    }

    /// Validate `access_devices` like `map_next` and return the accumulated
    /// device list plus its access descriptors.
    fn resolve_access(
        &self,
        hip: &HipRuntime,
        access_devices: &[i32],
    ) -> HipResult<(Vec<i32>, Vec<HipMemAccessDesc>)> {
        let count = hip.device_count()?;
        let mut devices = Vec::with_capacity(access_devices.len() + 1);
        devices.push(self.owner_device);
        for &device in access_devices {
            if device < 0 || device >= count {
                return Err(HipError::new(
                    0,
                    &format!("VMM access device {device} is outside available range 0..{count}"),
                ));
            }
            if device != self.owner_device && !hip.can_access_peer(device, self.owner_device)? {
                return Err(HipError::new(
                    0,
                    &format!(
                        "VMM access device {device} cannot access owner device {}",
                        self.owner_device
                    ),
                ));
            }
            if !devices.contains(&device) {
                devices.push(device);
            }
        }
        let mut next_access_devices = self.access_devices.clone();
        for device in devices {
            if !next_access_devices.contains(&device) {
                next_access_devices.push(device);
            }
        }
        let access = next_access_devices
            .iter()
            .copied()
            .map(HipMemAccessDesc::read_write_device)
            .collect();
        Ok((next_access_devices, access))
    }

    /// Offset the post-map `hipMemSetAccess` starts at. `new_offset` (just the
    /// new range) only when the earlier mapped prefix already carries exactly
    /// the requested access: no device was added, so its permissions are
    /// unchanged, and the device allocation granularity is at least 2 MiB.
    /// The gfx1100 ROCm 7.2 failure at 4 KiB subranges (see `map_next`) is
    /// documented only for smaller granularities, so those, and any growth
    /// that adds a peer device, keep the whole-prefix call from the base.
    /// Whole-prefix `hipMemSetAccess` cost grows with the number of mapped
    /// segments (measured on gfx1201: 0.03 ms with 1 segment mapped, 0.9 ms
    /// with 512 one-granule segments), so small repeated growth would
    /// otherwise be quadratic. HIP only requires the range to start at a
    /// mapped sub-buffer and end on a sub-buffer boundary
    /// (`ValidateSubBufferCoverage` in `hip_vm.cpp`); a new range starting at
    /// the previous segment's end satisfies that.
    fn access_start(&self, next_access_devices: &[i32], new_offset: usize) -> usize {
        if self.granularity >= GRANULE_MIN_BYTES
            && next_access_devices.len() == self.access_devices.len()
        {
            new_offset
        } else {
            0
        }
    }

    /// One `hipMemSetAccess` over `[access_start, next_mapped)`: the whole
    /// mapped prefix from the base by default (see `map_next` for why), only
    /// the new range when [`Self::access_start`] allows it.
    fn apply_access(
        &self,
        hip: &HipRuntime,
        next_mapped: usize,
        access: &[HipMemAccessDesc],
        next_access_devices: &[i32],
        new_offset: usize,
        new_size: usize,
    ) -> HipResult<()> {
        let start = self.access_start(next_access_devices, new_offset);
        take_fault(VmmFaultKind::AccessReset)
            .map_or_else(
                // SAFETY: `start..next_mapped` is mapped (the caller just mapped
                // `[new_offset, +new_size)` at the prefix end and `start <=
                // new_offset`); `access` lists only owner/peer devices already
                // validated for peer access.
                || unsafe {
                    hip.mem_set_access(offset_ptr(self.base, start), next_mapped - start, access)
                },
                Err,
            )
            .map_err(|err| HipError {
                code: err.code,
                message: format!(
                    "{}; VMM access range base=0x{:x} offset={} size={} (new granules offset={} size={}) granularity={}",
                    err.message,
                    self.base as usize,
                    start,
                    next_mapped - start,
                    new_offset,
                    new_size,
                    self.granularity,
                ),
                context: err.context,
            })
    }

    /// Roll back the staged segments of a failed granule map: unmap each,
    /// drop its lease, and poison the arena (a retried map could land a new
    /// handle on a VA the GPU may still translate to the released one).
    /// Segments whose unmap failed keep their lease and are retried by `release`.
    fn abort_granule_map(
        &mut self,
        hip: &HipRuntime,
        mut staged: Vec<VmmSegment>,
        err: HipError,
    ) -> HipError {
        self.releasing = true;
        match teardown_segments(hip, self.base, &mut staged) {
            Ok(()) => err,
            Err(cleanup) => {
                self.segments.append(&mut staged);
                combined_cleanup_error(err, cleanup)
            }
        }
    }

    /// Return a non-owning buffer view over the reserved virtual address.
    ///
    /// Only the mapped prefix may be accessed. The returned buffer must never
    /// be passed to `HipRuntime::free` or a pool that eventually calls it.
    pub fn buffer(&self, logical_bytes: usize) -> HipResult<DeviceBuffer> {
        if self.releasing || self.is_released() {
            return Err(HipError::new(
                0,
                "cannot create a buffer view from a releasing or released VMM arena",
            ));
        }
        if logical_bytes > self.mapped_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "VMM buffer view {logical_bytes} exceeds mapped prefix {}",
                    self.mapped_bytes
                ),
            ));
        }
        // SAFETY: base is a live reserved VA; logical_bytes <= mapped_bytes
        // (checked above). Borrowed wrapper must not outlive the arena or be freed.
        Ok(unsafe { DeviceBuffer::from_raw(self.base, logical_bytes) })
    }

    /// Return the unique owning descriptor for a reserved dense tensor.
    ///
    /// # Safety
    ///
    /// The returned buffer's safe byte length is capped at `mapped_bytes()`;
    /// `logical_bytes` only validates the tensor's intended reserved extent.
    /// The caller must register exactly one returned owner for arena teardown.
    pub unsafe fn owner_buffer(&self, logical_bytes: usize) -> HipResult<DeviceBuffer> {
        if self.releasing || self.is_released() {
            return Err(HipError::new(
                0,
                "cannot create an owner from a releasing or released VMM arena",
            ));
        }
        if logical_bytes > self.reserved_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "VMM owner view {logical_bytes} exceeds reserve {}",
                    self.reserved_bytes
                ),
            ));
        }
        Ok(DeviceBuffer::from_vmm_owner(
            self.base,
            logical_bytes.min(self.mapped_bytes),
        ))
    }

    /// True once the arena is read-only for its owner device (published
    /// exportable arena, or an imported arena).
    pub fn is_read_only(&self) -> bool {
        matches!(
            self.kind,
            VmmArenaKind::ExportPublished | VmmArenaKind::ImportedReadOnly
        )
    }

    /// Make the exportable arena immutable: synchronize the owner device, then
    /// switch the owner mapping from read-write to read-only
    /// (`hipMemSetAccess` over the whole mapping). Construction (`map_next`)
    /// is RW only; after this call `map_next` is refused and mapping access
    /// can no longer be written. Idempotent once published; on failure the
    /// arena stays in its construction state (still releasable).
    pub fn publish_read_only(&mut self, hip: &HipRuntime) -> HipResult<()> {
        match self.kind {
            VmmArenaKind::ExportPublished => return Ok(()),
            VmmArenaKind::ExportBuilding => {}
            _ => {
                return Err(HipError::new(
                    0,
                    "publish_read_only requires an arena from reserve_exportable",
                ))
            }
        }
        self.ensure_published_state()?;
        hip.set_device(self.owner_device)?;
        hip.device_synchronize()?;
        let access = [HipMemAccessDesc::read_only_device(self.owner_device)];
        take_fault(VmmFaultKind::AccessReset).map_or_else(
            // SAFETY: `base..+mapped_bytes` is the arena's single mapped
            // allocation (checked by `ensure_published_state`); only the owner
            // device is listed, read-only; the device was synchronized above.
            || unsafe { hip.mem_set_access(self.base, self.mapped_bytes, &access) },
            Err,
        )?;
        self.kind = VmmArenaKind::ExportPublished;
        Ok(())
    }

    /// Export the published arena's physical allocation as a new POSIX FD
    /// (an independent reference; the caller owns it and may call this more
    /// than once). Requires [`Self::publish_read_only`] first.
    #[cfg(unix)]
    pub fn export_posix(&self, hip: &HipRuntime) -> HipResult<OwnedFd> {
        if self.kind != VmmArenaKind::ExportPublished {
            return Err(HipError::new(
                0,
                "export_posix requires a published exportable VMM arena",
            ));
        }
        self.ensure_published_state()?;
        let handle = self.segments[0].handle.ok_or_else(|| {
            HipError::new(0, "exportable VMM arena has no physical allocation handle")
        })?;
        hip.mem_export_posix(handle)
    }

    /// Query the owner device's access flags on this arena's mapping
    /// (`hipMemGetAccess`; `HIP_MEM_ACCESS_FLAGS_PROT_*`). Optional symbol:
    /// errors if the runtime lacks it. Sharing never depends on this call.
    pub fn owner_access_flags(&self, hip: &HipRuntime) -> HipResult<u32> {
        if self.releasing || self.is_released() {
            return Err(HipError::new(
                0,
                "cannot query access of a releasing or released VMM arena",
            ));
        }
        hip.mem_get_access(&HipMemLocation::device(self.owner_device), self.base)
    }

    /// Exportable arena must be live with exactly one fully mapped allocation.
    fn ensure_published_state(&self) -> HipResult<()> {
        if self.releasing || self.is_released() {
            return Err(HipError::new(
                0,
                "VMM arena is releasing or already released",
            ));
        }
        if self.segments.len() != 1 || !self.segments[0].mapped || self.mapped_bytes == 0 {
            return Err(HipError::new(
                0,
                "exportable VMM arena must hold exactly one mapped physical allocation",
            ));
        }
        Ok(())
    }

    /// Unmap every segment and release every physical handle. The VA range is
    /// kept reserved and retired, never freed (see the module docs).
    /// Cleanup continues after an individual failure and returns the first one.
    pub fn release(&mut self, hip: &impl VmmDriver) -> HipResult<()> {
        if self.is_released() {
            return Ok(());
        }
        self.releasing = true;
        for &device in &self.access_devices {
            hip.set_device(device)?;
            hip.device_synchronize()?;
        }
        hip.set_device(self.owner_device)?;
        let base = self.base;
        let result = teardown_segments(hip, base, &mut self.segments);

        if self.segments.is_empty() {
            retired_va().retire(self.reserved_bytes);
            self.base = std::ptr::null_mut();
            self.reserved_bytes = 0;
            self.mapped_bytes = 0;
            self.access_devices.clear();
        }

        result
    }
}

/// Unmap and release `segments` in reverse order (see [`cleanup_segments`]).
/// Granule segments drop their mapping lease; their handle is released only
/// at the last lease.
fn teardown_segments(
    hip: &impl VmmDriver,
    base: *mut c_void,
    segments: &mut Vec<VmmSegment>,
) -> HipResult<()> {
    cleanup_segments(
        segments,
        |offset, size| {
            if let Some(err) = take_fault(VmmFaultKind::Unmap) {
                return Err(err);
            }
            // SAFETY: offset/size come from tracked segments of the arena
            // reserved at `base`; callers either device_synchronize'd every
            // access device (release) or unmap granules staged by a failed map
            // call that never scheduled work on them.
            unsafe { hip.mem_unmap(offset_ptr(base, offset), size) }
        },
        |handle| {
            if let Some(err) = take_fault(VmmFaultKind::Release) {
                return Err(err);
            }
            // SAFETY: called after the segment's map is unmapped (cleanup_segments
            // order); handle is owned and no mapped range should still reference it.
            unsafe { hip.mem_release(handle) }
        },
        |id| granule_table().drop_lease(id, &mut |handle| release_granule_handle(hip, handle)),
    )
}

fn offset_ptr(base: *mut c_void, offset: usize) -> *mut c_void {
    // SAFETY: callers pass base from a live VmmArena reserve and offset within
    // reserved_bytes (map_next / release segment bookkeeping). u8 add; no deref.
    unsafe { (base as *mut u8).add(offset) as *mut c_void }
}

fn round_up(value: usize, alignment: usize) -> HipResult<usize> {
    if alignment == 0 {
        return Err(HipError::new(0, "VMM alignment must be greater than zero"));
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| HipError::new(0, "VMM reserve size overflowed during alignment"))
    }
}

fn combined_cleanup_error(operation: HipError, cleanup: HipError) -> HipError {
    HipError::new(
        0,
        &format!("{operation}; cleanup also failed: {cleanup}; VMM arena retained for retry"),
    )
}

fn cleanup_segments(
    segments: &mut Vec<VmmSegment>,
    mut unmap: impl FnMut(usize, usize) -> HipResult<()>,
    mut release: impl FnMut(HipMemGenericAllocationHandle) -> HipResult<()>,
    mut drop_lease: impl FnMut(VmmPhysicalId) -> HipResult<()>,
) -> HipResult<()> {
    let mut first_error = None;
    for segment in segments.iter_mut().rev() {
        if segment.mapped {
            match unmap(segment.offset, segment.size) {
                Ok(()) => segment.mapped = false,
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                    continue;
                }
            }
        }
        if let Some(handle) = segment.handle {
            match release(handle) {
                Ok(()) => segment.handle = None,
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }
        if let Some(id) = segment.granule.take() {
            // The mapping lease is consumed even when the handle release
            // fails: the granule table keeps that handle pending for
            // `retry_pending_granule_releases`.
            if let Err(err) = drop_lease(id) {
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }
    }
    segments
        .retain(|segment| segment.mapped || segment.handle.is_some() || segment.granule.is_some());
    match first_error {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(mapped: bool) -> VmmSegment {
        VmmSegment {
            offset: 4096,
            size: 4096,
            handle: Some(1usize as HipMemGenericAllocationHandle),
            granule: None,
            mapped,
        }
    }

    fn granule_segment(offset: usize, size: usize, id: VmmPhysicalId) -> VmmSegment {
        VmmSegment {
            offset,
            size,
            handle: None,
            granule: Some(id),
            mapped: true,
        }
    }

    fn host_arena(segments: Vec<VmmSegment>) -> VmmArena {
        let mapped_bytes = segments
            .iter()
            .map(|s| s.offset + s.size)
            .max()
            .unwrap_or(0);
        VmmArena {
            base: 0x1000_0000usize as *mut c_void,
            owner_device: 0,
            granularity: 4096,
            reserved_bytes: 64 << 20,
            mapped_bytes,
            segments,
            access_devices: vec![0],
            releasing: false,
            ever_mapped: true,
            kind: VmmArenaKind::Default,
        }
    }

    fn forget_prefix(mut prefix: VmmSharedPrefix) {
        let ids: Vec<VmmPhysicalId> = std::mem::take(&mut prefix.granules)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        drop_leases(&mut granule_table(), ids, &mut |_| Ok(())).unwrap();
    }

    fn forget_global(id: VmmPhysicalId) {
        granule_table().drop_lease(id, &mut |_| Ok(())).unwrap();
    }

    #[test]
    fn access_start_is_the_new_range_only_for_unchanged_devices_at_2mib() {
        const MIB: usize = 1 << 20;
        let mut arena = host_arena(Vec::new());
        arena.access_devices = vec![0];
        arena.granularity = 2 * MIB;
        // Default (>= 2 MiB, no added peer): only the new range.
        assert_eq!(arena.access_start(&[0], 6 * MIB), 6 * MIB);
        arena.granularity = 4 * MIB;
        assert_eq!(arena.access_start(&[0], 8 * MIB), 8 * MIB);
        // Small-page granularity keeps the whole-prefix call from the base.
        arena.granularity = 4096;
        assert_eq!(arena.access_start(&[0], 6 * MIB), 0);
        arena.granularity = 2 * MIB - 4096;
        assert_eq!(arena.access_start(&[0], 6 * MIB), 0);
        // An added peer device must reach the older segments: whole prefix.
        arena.granularity = 2 * MIB;
        assert_eq!(arena.access_start(&[0, 1], 6 * MIB), 0);
        // A never-mapped arena (first map, or a shared-prefix map at offset 0).
        arena.access_devices = Vec::new();
        assert_eq!(arena.access_start(&[0], 0), 0);
        arena.access_devices = vec![0];
        assert_eq!(arena.access_start(&[0], 0), 0);
    }

    #[test]
    fn failed_unmap_keeps_mapping_and_handle_for_retry() {
        let mut segments = vec![segment(true)];
        let mut releases = 0;
        let err = cleanup_segments(
            &mut segments,
            |_, _| Err(HipError::new(1, "injected unmap failure")),
            |_| {
                releases += 1;
                Ok(())
            },
            |_| panic!("exclusive segment holds no granule lease"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("injected unmap failure"));
        assert_eq!(releases, 0, "a still-mapped handle must not be released");
        assert!(segments[0].mapped);
        assert!(segments[0].handle.is_some());

        cleanup_segments(
            &mut segments,
            |_, _| Ok(()),
            |_| Ok(()),
            |_| panic!("exclusive segment holds no granule lease"),
        )
        .unwrap();
        assert!(segments.is_empty());
    }

    #[test]
    fn failed_handle_release_keeps_handle_for_retry() {
        let mut segments = vec![segment(false)];
        cleanup_segments(
            &mut segments,
            |_, _| panic!("unmap must not run for an unmapped segment"),
            |_| Err(HipError::new(2, "injected handle failure")),
            |_| panic!("exclusive segment holds no granule lease"),
        )
        .unwrap_err();
        assert!(!segments[0].mapped);
        assert!(segments[0].handle.is_some());

        cleanup_segments(
            &mut segments,
            |_, _| Ok(()),
            |_| Ok(()),
            |_| panic!("exclusive segment holds no granule lease"),
        )
        .unwrap();
        assert!(segments.is_empty());
    }

    #[test]
    fn inject_vmm_fault_counters_are_consumed_once_each() {
        clear_vmm_faults();
        inject_vmm_fault(VmmFaultKind::Unmap, 2);
        inject_vmm_fault(VmmFaultKind::Release, 1);
        inject_vmm_fault(VmmFaultKind::AccessReset, 1);

        let u1 = take_fault(VmmFaultKind::Unmap).unwrap();
        let u2 = take_fault(VmmFaultKind::Unmap).unwrap();
        assert!(take_fault(VmmFaultKind::Unmap).is_none());
        assert!(u1.to_string().contains("unmap"));
        assert!(u2.to_string().contains("unmap"));

        let r = take_fault(VmmFaultKind::Release).unwrap();
        assert!(r.to_string().contains("release"));
        assert!(take_fault(VmmFaultKind::Release).is_none());

        let a = take_fault(VmmFaultKind::AccessReset).unwrap();
        assert!(a.to_string().contains("access-reset"));
        assert!(take_fault(VmmFaultKind::AccessReset).is_none());
        clear_vmm_faults();
    }

    #[test]
    fn clear_vmm_faults_drops_pending_injections() {
        inject_vmm_fault(VmmFaultKind::Unmap, 5);
        inject_vmm_fault(VmmFaultKind::Release, 5);
        inject_vmm_fault(VmmFaultKind::AccessReset, 5);
        clear_vmm_faults();
        assert!(take_fault(VmmFaultKind::Unmap).is_none());
        assert!(take_fault(VmmFaultKind::Release).is_none());
        assert!(take_fault(VmmFaultKind::AccessReset).is_none());
    }

    #[test]
    fn retired_va_budget_admits_up_to_the_cap_then_refuses() {
        let cap = 64usize << 20;
        let mut retired = RetiredVa::new();
        retired.admit(cap, cap).unwrap();
        retired.retire(24 << 20);
        retired.retire(8 << 20);
        retired.admit(cap - (32 << 20), cap).unwrap();
        let err = retired.admit(cap - (32 << 20) + 1, cap).unwrap_err();
        let message = err.to_string();
        assert!(message.contains(&(32usize << 20).to_string()), "{message}");
        assert!(message.contains("2 ranges"), "{message}");
        assert!(message.contains("restart"), "{message}");
        assert!(retired.admit(usize::MAX, cap).is_err());
    }

    const G: usize = 2 << 20;

    /// Three granule segments registered in the global table with one mapping
    /// lease each (fake handles; no HIP call is ever made on them).
    fn three_granule_arena() -> (VmmArena, [VmmPhysicalId; 3]) {
        let mut table = granule_table();
        let ids = [
            table.register(0x10, G),
            table.register(0x20, G),
            table.register(0x30, G),
        ];
        drop(table);
        let segments = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| granule_segment(i * G, G, id))
            .collect();
        (host_arena(segments), ids)
    }

    fn leases(id: VmmPhysicalId) -> Option<usize> {
        granule_table().leases(id)
    }

    #[test]
    fn granule_released_only_at_last_lease() {
        let mut table = GranuleTable::new();
        let id = table.register(0x10, 4096);
        table.add_lease(id).unwrap();
        assert_eq!(table.leases(id), Some(2));
        let mut released = Vec::new();
        table
            .drop_lease(id, &mut |handle| {
                released.push(handle);
                Ok(())
            })
            .unwrap();
        assert!(released.is_empty(), "a leased granule must not be released");
        assert_eq!(table.live_bytes(), 4096);
        table
            .drop_lease(id, &mut |handle| {
                released.push(handle);
                Ok(())
            })
            .unwrap();
        assert_eq!(released, vec![0x10]);
        assert_eq!(table.live_bytes(), 0);
        assert!(table.leases(id).is_none());
        assert!(table.drop_lease(id, &mut |_| Ok(())).is_err());
    }

    #[test]
    fn granule_ids_are_monotonic_and_never_reused() {
        let mut table = GranuleTable::new();
        let a = table.register(1, 4096);
        table.drop_lease(a, &mut |_| Ok(())).unwrap();
        let b = table.register(1, 4096);
        assert!(b.id > a.id);
        assert!(table.leases(a).is_none());
    }

    #[test]
    fn shared_prefix_release_before_arena_release_keeps_granule_until_unmap() {
        let mut table = GranuleTable::new();
        let id = table.register(0x10, 4096);
        table.add_lease(id).unwrap(); // the shared prefix's cache lease
        let mut released = Vec::new();

        drop_leases(&mut table, [id], &mut |handle| {
            released.push(handle);
            Ok(())
        })
        .unwrap();
        assert!(released.is_empty(), "the arena mapping still holds a lease");

        let mut segments = vec![granule_segment(0, 4096, id)];
        cleanup_segments(
            &mut segments,
            |_, _| Ok(()),
            |_| panic!("granule segments never use the exclusive handle"),
            |id| {
                table.drop_lease(id, &mut |handle| {
                    released.push(handle);
                    Ok(())
                })
            },
        )
        .unwrap();
        assert!(segments.is_empty());
        assert_eq!(released, vec![0x10]);
        assert_eq!(table.live_bytes(), 0);
    }

    #[test]
    fn arena_release_before_shared_prefix_release_keeps_granule_until_prefix_drops() {
        let mut table = GranuleTable::new();
        let id = table.register(0x10, 4096);
        table.add_lease(id).unwrap(); // the shared prefix's cache lease
        let mut released = Vec::new();

        let mut segments = vec![granule_segment(0, 4096, id)];
        cleanup_segments(
            &mut segments,
            |_, _| Ok(()),
            |_| panic!("granule segments never use the exclusive handle"),
            |id| {
                table.drop_lease(id, &mut |handle| {
                    released.push(handle);
                    Ok(())
                })
            },
        )
        .unwrap();
        assert!(segments.is_empty());
        assert!(released.is_empty(), "the shared prefix still holds a lease");
        assert_eq!(table.live_bytes(), 4096);

        drop_leases(&mut table, [id], &mut |handle| {
            released.push(handle);
            Ok(())
        })
        .unwrap();
        assert_eq!(released, vec![0x10]);
        assert_eq!(table.live_bytes(), 0);
    }

    #[test]
    fn failed_granule_release_stays_pending_and_is_retried() {
        let mut table = GranuleTable::new();
        let id = table.register(0x10, 4096);
        let mut segments = vec![granule_segment(0, 4096, id)];
        let err = cleanup_segments(
            &mut segments,
            |_, _| Ok(()),
            |_| panic!("granule segments never use the exclusive handle"),
            |id| {
                table.drop_lease(id, &mut |_| {
                    Err(HipError::new(3, "injected release failure"))
                })
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("injected release failure"));
        assert!(segments.is_empty(), "the mapping lease is consumed");
        assert_eq!(table.live_bytes(), 4096, "a pending release is still live");
        assert_eq!(table.leases(id), Some(0));
        assert!(
            table.add_lease(id).is_err(),
            "a pending granule is not leasable"
        );

        let (released, error) =
            table.retry_pending(&mut |_| Err(HipError::new(4, "still failing")));
        assert_eq!(released, 0);
        assert!(error.is_some());
        assert_eq!(table.live_bytes(), 4096);

        let mut handles = Vec::new();
        let (released, error) = table.retry_pending(&mut |handle| {
            handles.push(handle);
            Ok(())
        });
        assert_eq!((released, handles), (1, vec![0x10]));
        assert!(error.is_none());
        assert_eq!(table.live_bytes(), 0);
    }

    #[test]
    fn failed_unmap_keeps_granule_lease_for_retry() {
        let mut table = GranuleTable::new();
        let id = table.register(0x10, 4096);
        let mut segments = vec![granule_segment(0, 4096, id)];
        cleanup_segments(
            &mut segments,
            |_, _| Err(HipError::new(1, "injected unmap failure")),
            |_| panic!("granule segments never use the exclusive handle"),
            |_| panic!("a still-mapped granule must keep its lease"),
        )
        .unwrap_err();
        assert_eq!(segments.len(), 1);
        assert!(segments[0].mapped);
        assert_eq!(table.leases(id), Some(1));
    }

    #[test]
    fn share_prefix_bounds_and_lease_accounting() {
        let (arena, ids) = three_granule_arena();
        assert_eq!(arena.granule_bytes(), G);
        assert_eq!(arena.granule_prefix_bytes(), 3 * G);

        let empty = arena.share_prefix(0).unwrap();
        assert_eq!(empty.bytes(), 0);
        assert!(empty.granules().is_empty());
        assert_eq!(leases(ids[0]), Some(1));

        let err = arena.share_prefix(G / 2).unwrap_err();
        assert!(err.to_string().contains("granule boundary"), "{err}");
        let err = arena.share_prefix(G + 4096).unwrap_err();
        assert!(err.to_string().contains("granule boundary"), "{err}");
        let err = arena.share_prefix(4 * G).unwrap_err();
        assert!(err.to_string().contains("granule prefix"), "{err}");
        assert!(
            ids.iter().all(|&id| leases(id) == Some(1)),
            "failed shares take no lease"
        );

        let prefix = arena.share_prefix(2 * G).unwrap();
        assert_eq!(prefix.bytes(), 2 * G);
        assert_eq!(prefix.granules(), &[(ids[0], G), (ids[1], G)]);
        assert_eq!(leases(ids[0]), Some(2));
        assert_eq!(leases(ids[1]), Some(2));
        assert_eq!(leases(ids[2]), Some(1));
        assert_eq!(arena.sealed_bytes(), 2 * G);

        forget_prefix(prefix);
        assert_eq!(arena.sealed_bytes(), 0);
        for id in ids {
            forget_global(id);
        }
    }

    #[test]
    fn granule_prefix_stops_at_exclusive_or_unmapped_segments() {
        let (mut arena, ids) = three_granule_arena();
        arena.segments[1] = VmmSegment {
            offset: G,
            size: G,
            handle: Some(1usize as HipMemGenericAllocationHandle),
            granule: None,
            mapped: true,
        };
        assert_eq!(arena.granule_prefix_bytes(), G);
        assert_eq!(arena.physical_granules(), vec![(ids[0], G), (ids[2], G)]);
        let err = arena.share_prefix(2 * G).unwrap_err();
        assert!(err.to_string().contains("granule prefix"), "{err}");
        for id in ids {
            forget_global(id);
        }
    }

    #[test]
    fn physical_granules_and_sealed_bytes_follow_leases() {
        let (arena, ids) = three_granule_arena();
        assert_eq!(
            arena.physical_granules(),
            vec![(ids[0], G), (ids[1], G), (ids[2], G)]
        );
        assert_eq!(arena.sealed_bytes(), 0);
        granule_table().add_lease(ids[1]).unwrap();
        assert_eq!(arena.sealed_bytes(), 2 * G);
        granule_table().add_lease(ids[2]).unwrap();
        assert_eq!(arena.sealed_bytes(), 3 * G);
        for id in ids {
            while leases(id).is_some() {
                forget_global(id);
            }
        }
    }

    #[test]
    fn share_prefix_refuses_a_poisoned_arena() {
        let (mut arena, ids) = three_granule_arena();
        arena.releasing = true;
        assert!(arena.share_prefix(G).is_err());
        assert_eq!(leases(ids[0]), Some(1));
        for id in ids {
            forget_global(id);
        }
    }

    #[test]
    fn dropping_an_unreleased_shared_prefix_touches_no_lease() {
        let id = granule_table().register(0x10, 4096);
        granule_table().add_lease(id).unwrap();
        let prefix = VmmSharedPrefix {
            granules: vec![(id, 4096)],
            owner_device: 0,
            granularity: 4096,
        };
        drop(prefix);
        assert_eq!(leases(id), Some(2), "Drop must not release or decrement");
        forget_global(id);
        forget_global(id);
    }

    #[test]
    fn granule_fault_kinds_are_consumed_once_each() {
        clear_vmm_faults();
        inject_vmm_fault(VmmFaultKind::Map, 1);
        inject_vmm_fault(VmmFaultKind::Create, 2);
        assert!(take_fault(VmmFaultKind::Map)
            .unwrap()
            .to_string()
            .contains("map"));
        assert!(take_fault(VmmFaultKind::Map).is_none());
        assert!(take_fault(VmmFaultKind::Create)
            .unwrap()
            .to_string()
            .contains("create"));
        assert!(take_fault(VmmFaultKind::Create).is_some());
        assert!(take_fault(VmmFaultKind::Create).is_none());
        inject_vmm_fault(VmmFaultKind::Map, 3);
        inject_vmm_fault(VmmFaultKind::Create, 3);
        clear_vmm_faults();
        assert!(take_fault(VmmFaultKind::Map).is_none());
        assert!(take_fault(VmmFaultKind::Create).is_none());
    }

    fn gpu_pattern(len: usize, mul: usize, add: usize) -> Vec<u8> {
        (0..len).map(|i| ((i * mul + add) % 251) as u8).collect()
    }

    #[test]
    #[ignore = "requires a HIP device with VMM support (select the card with ROCR_VISIBLE_DEVICES)"]
    fn vmm_granule_alias_two_fresh_vas_hw() {
        let device = 0;
        let hip = HipRuntime::load().expect("HIP runtime unavailable: cannot run the VMM test");
        let count = hip.device_count().expect("hipGetDeviceCount failed");
        assert!(device < count, "no HIP device visible ({count} devices)");
        let access = [device];
        let live_before = vmm_live_granule_bytes();

        let mut a = VmmArena::reserve(&hip, device, 64 << 20).expect("reserve A");
        let g = a.granule_bytes();
        assert!(
            4 * g <= a.reserved_bytes(),
            "reserve too small for 4 granules of {g} bytes"
        );
        a.map_next_granules(&hip, 3 * g, &access)
            .expect("map 3 granules in A");
        assert_eq!(a.mapped_bytes(), 3 * g);
        assert_eq!(a.granule_prefix_bytes(), 3 * g);
        assert_eq!(a.sealed_bytes(), 0);
        assert!(a.primary_handle().is_some());
        let pattern = gpu_pattern(3 * g, 31, 7);
        hip.memcpy_htod(&a.buffer(3 * g).unwrap(), &pattern)
            .expect("write A");

        let shared = a.share_prefix(2 * g).expect("share 2 granules");
        assert_eq!(shared.bytes(), 2 * g);
        assert_eq!(shared.granules().len(), 2);
        assert_eq!(a.sealed_bytes(), 2 * g);

        let mut b = VmmArena::reserve(&hip, device, 64 << 20).expect("reserve B");
        assert_ne!(a.base_address(), b.base_address());
        b.map_shared_prefix(&hip, &shared, &access)
            .expect("map shared prefix in B");
        assert_eq!(b.mapped_bytes(), 2 * g);
        b.map_next_granules(&hip, g, &access)
            .expect("map private granule in B");
        assert_eq!(b.mapped_bytes(), 3 * g);
        assert_eq!(b.sealed_bytes(), 2 * g);
        let physical_b = b.physical_granules();
        assert_eq!(&physical_b[..2], shared.granules());
        assert_eq!(&physical_b[..2], &a.physical_granules()[..2]);
        assert_ne!(physical_b[2].0, a.physical_granules()[2].0);
        assert!(
            b.map_shared_prefix(&hip, &shared, &access).is_err(),
            "B was already mapped"
        );

        let buf_a = a.buffer(3 * g).unwrap();
        let buf_b = b.buffer(3 * g).unwrap();
        let mut read = vec![0u8; 2 * g];
        hip.memcpy_dtoh_at(&mut read, &buf_b, 0)
            .expect("read B alias");
        assert_eq!(read, pattern[..2 * g]);

        let private = gpu_pattern(g, 13, 99);
        hip.memcpy_htod_offset(&buf_b, 2 * g, &private)
            .expect("write B private granule");
        let mut a_third = vec![0u8; g];
        hip.memcpy_dtoh_at(&mut a_third, &buf_a, 2 * g)
            .expect("read A third granule");
        assert_eq!(
            a_third,
            pattern[2 * g..3 * g],
            "A's private granule must be unchanged"
        );
        let mut b_third = vec![0u8; g];
        hip.memcpy_dtoh_at(&mut b_third, &buf_b, 2 * g)
            .expect("read B third granule");
        assert_eq!(b_third, private);

        a.release(&hip).expect("release A");
        assert!(a.is_released());
        hip.memcpy_dtoh_at(&mut read, &buf_b, 0)
            .expect("read B after A release");
        assert_eq!(
            read,
            pattern[..2 * g],
            "B must keep the aliased bytes after A is released"
        );

        b.release(&hip).expect("release B");
        shared.release(&hip).expect("release shared prefix");
        assert_eq!(vmm_live_granule_bytes(), live_before);
        assert_eq!(
            retry_pending_granule_releases(&hip).expect("retry pending"),
            0
        );
    }

    // ---- Map-only discipline (gfx1100 hardware defects), CPU only ----------
    //
    // gfx1100 defect B: after `hipMemUnmap` + `hipMemMap` of other handles a VA
    // keeps translating to the old backing, so a live VA must never be
    // unmapped and remapped mid-life (module docs). Defect A: `hipMemSetAccess`
    // rejects some 4 KiB-offset sub-ranges. The recording driver below stands
    // in for the HIP runtime so the real `reserve` / `map_next` / `release`
    // bodies run on a host with no device.

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum DriverOp {
        Reserve { size: usize },
        Create { size: usize },
        Map { offset: usize, size: usize },
        SetAccess { offset: usize, size: usize },
        Unmap { offset: usize, size: usize },
        Release,
        Sync,
    }

    struct RecordingDriver {
        granularity: usize,
        ops: std::cell::RefCell<Vec<DriverOp>>,
        /// Base of the most recent reservation; the next one is placed above
        /// it, never at a returned base (retired ranges stay reserved).
        next_base: Cell<usize>,
        bases: std::cell::RefCell<Vec<usize>>,
        fail_map: Cell<bool>,
        next_handle: Cell<usize>,
    }

    impl RecordingDriver {
        fn new(granularity: usize) -> Self {
            Self {
                granularity,
                ops: Default::default(),
                next_base: Cell::new(0x7f00_0000_0000),
                bases: Default::default(),
                fail_map: Cell::new(false),
                next_handle: Cell::new(0x100),
            }
        }

        fn ops(&self) -> Vec<DriverOp> {
            self.ops.borrow().clone()
        }

        fn take_ops(&self) -> Vec<DriverOp> {
            std::mem::take(&mut *self.ops.borrow_mut())
        }

        fn log(&self, op: DriverOp) {
            self.ops.borrow_mut().push(op);
        }

        fn offset_of(&self, ptr: *mut c_void) -> usize {
            let base = *self.bases.borrow().last().expect("a reservation");
            ptr as usize - base
        }
    }

    impl VmmDriver for RecordingDriver {
        fn device_count(&self) -> HipResult<i32> {
            Ok(1)
        }
        fn set_device(&self, _: i32) -> HipResult<()> {
            Ok(())
        }
        fn can_access_peer(&self, _: i32, _: i32) -> HipResult<bool> {
            Ok(true)
        }
        fn device_synchronize(&self) -> HipResult<()> {
            self.log(DriverOp::Sync);
            Ok(())
        }
        fn mem_get_allocation_granularity(
            &self,
            _: &HipMemAllocationProp,
            _: u32,
        ) -> HipResult<usize> {
            Ok(self.granularity)
        }
        fn mem_address_reserve(&self, size: usize, _: usize) -> HipResult<*mut c_void> {
            self.log(DriverOp::Reserve { size });
            let base = self.next_base.get();
            self.next_base.set(base + size.next_multiple_of(1 << 30));
            self.bases.borrow_mut().push(base);
            Ok(base as *mut c_void)
        }
        fn mem_create(
            &self,
            size: usize,
            _: &HipMemAllocationProp,
        ) -> HipResult<HipMemGenericAllocationHandle> {
            self.log(DriverOp::Create { size });
            let handle = self.next_handle.get();
            self.next_handle.set(handle + 1);
            Ok(handle as HipMemGenericAllocationHandle)
        }
        unsafe fn mem_map(
            &self,
            ptr: *mut c_void,
            size: usize,
            _: HipMemGenericAllocationHandle,
        ) -> HipResult<()> {
            self.log(DriverOp::Map {
                offset: self.offset_of(ptr),
                size,
            });
            if self.fail_map.replace(false) {
                return Err(HipError::new(1, "injected map failure"));
            }
            Ok(())
        }
        unsafe fn mem_set_access(
            &self,
            ptr: *mut c_void,
            size: usize,
            _: &[HipMemAccessDesc],
        ) -> HipResult<()> {
            self.log(DriverOp::SetAccess {
                offset: self.offset_of(ptr),
                size,
            });
            Ok(())
        }
        unsafe fn mem_unmap(&self, ptr: *mut c_void, size: usize) -> HipResult<()> {
            self.log(DriverOp::Unmap {
                offset: self.offset_of(ptr),
                size,
            });
            Ok(())
        }
        unsafe fn mem_release(&self, _: HipMemGenericAllocationHandle) -> HipResult<()> {
            self.log(DriverOp::Release);
            Ok(())
        }
    }

    /// Every VA range a driver ever mapped, with the rules the gfx1100 defects
    /// impose on a live arena: maps land exactly at the mapped frontier, never
    /// overlap a range that was mapped before (live or not), and access calls
    /// never begin at an offset only 4 KiB-aligned.
    #[derive(Default)]
    struct VaLedger {
        mapped_end: usize,
        ever_mapped: Vec<(usize, usize)>,
        unmaps: usize,
    }

    impl VaLedger {
        fn apply(&mut self, op: &DriverOp, small_granularity: bool) {
            match *op {
                DriverOp::Map { offset, size } => {
                    assert_eq!(
                        offset, self.mapped_end,
                        "a map must start at the mapped frontier, not inside or past it"
                    );
                    assert!(
                        self.ever_mapped
                            .iter()
                            .all(|&(start, end)| offset >= end || offset + size <= start),
                        "remapped a VA range that was mapped before: {offset}+{size}"
                    );
                    self.ever_mapped.push((offset, offset + size));
                    self.mapped_end = offset + size;
                }
                DriverOp::SetAccess { offset, size } => {
                    assert!(
                        offset + size <= self.mapped_end,
                        "access past the mapped prefix"
                    );
                    // Defect A: a sub-range starting at a 4 KiB-only offset is
                    // rejected by ROCm 7.2; allowed starts are the base or a
                    // 2 MiB boundary.
                    assert!(
                        offset == 0 || offset % (2 << 20) == 0,
                        "hipMemSetAccess at 4 KiB-only offset {offset}"
                    );
                    if small_granularity {
                        assert_eq!(offset, 0, "small pages re-apply access from the base");
                    }
                }
                DriverOp::Unmap { .. } => self.unmaps += 1,
                _ => {}
            }
        }
    }

    fn reserve_recorded(driver: &RecordingDriver, bytes: usize) -> VmmArena {
        VmmArena::reserve(driver, 0, bytes).expect("reserve on the recording driver")
    }

    /// QSA arena growth (`grow_context_arena` -> `grow_vmm_tensor` ->
    /// `map_next`, the gather workspace's `map_qsa_gather_prefix` likewise):
    /// every growth step maps one new segment at the frontier and sets access
    /// on the new range (>= 2 MiB pages) or from the base (4 KiB pages); no
    /// step unmaps, releases or remaps anything, at either page size the
    /// gfx1100 reports (4 KiB minimum, 2 MiB recommended).
    #[test]
    fn gfx1100_vmm_growth_is_map_only_at_the_frontier() {
        const MIB: usize = 1 << 20;
        for granularity in [2 * MIB, 4096] {
            let driver = RecordingDriver::new(granularity);
            let mut arena = reserve_recorded(&driver, 512 * MIB);
            let mut ledger = VaLedger::default();
            for op in driver.take_ops() {
                assert!(matches!(op, DriverOp::Reserve { .. }));
            }
            // Request-shaped growth: first tile, an 8K prefill, a decode
            // map-ahead, then single-page steps up to the reserve.
            let steps = [
                granularity,
                8 * MIB,
                2 * MIB,
                granularity,
                32 * MIB,
                MIB.max(granularity),
            ];
            for size in steps {
                let before = arena.mapped_bytes();
                arena.map_next(&driver, size, &[0]).expect("map_next");
                assert_eq!(arena.mapped_bytes(), before + size);
                let ops = driver.take_ops();
                assert!(
                    !ops.iter()
                        .any(|op| matches!(op, DriverOp::Unmap { .. } | DriverOp::Release)),
                    "growth unmapped or released: {ops:?}"
                );
                assert!(matches!(ops.first(), Some(DriverOp::Create { .. })));
                for op in &ops {
                    ledger.apply(op, granularity < 2 * MIB);
                }
                assert_eq!(ledger.mapped_end, arena.mapped_bytes());
            }
            assert_eq!(ledger.unmaps, 0);
            // A map past the reserve is refused before any driver call.
            let overshoot = 512 * MIB;
            assert!(arena.map_next(&driver, overshoot, &[0]).is_err());
            assert!(driver.take_ops().is_empty());
        }
    }

    /// A map the driver refuses leaves nothing mapped and the arena intact; the
    /// retry lands on the same never-mapped frontier (the only legal re-use of
    /// a VA offset), and the live prefix is never touched.
    #[test]
    fn gfx1100_vmm_failed_map_leaves_the_live_prefix_untouched() {
        const MIB: usize = 1 << 20;
        let driver = RecordingDriver::new(2 * MIB);
        let mut arena = reserve_recorded(&driver, 64 * MIB);
        arena.map_next(&driver, 2 * MIB, &[0]).unwrap();
        driver.take_ops();
        driver.fail_map.set(true);
        assert!(arena.map_next(&driver, 2 * MIB, &[0]).is_err());
        let ops = driver.take_ops();
        assert_eq!(
            ops,
            vec![
                DriverOp::Create { size: 2 * MIB },
                DriverOp::Map {
                    offset: 2 * MIB,
                    size: 2 * MIB
                },
                DriverOp::Release,
            ],
            "a failed map releases only its fresh handle"
        );
        assert_eq!(arena.mapped_bytes(), 2 * MIB);
        arena.map_next(&driver, 2 * MIB, &[0]).unwrap();
        assert!(!driver
            .ops()
            .iter()
            .any(|op| matches!(op, DriverOp::Unmap { .. })));
    }

    /// A failed `hipMemSetAccess` unmaps only the segment it just mapped (no
    /// kernel has touched it) and poisons the arena: a retry would remap that
    /// VA to a new handle the GPU may still translate to the released one.
    #[test]
    fn gfx1100_vmm_failed_access_poisons_instead_of_remapping() {
        const MIB: usize = 1 << 20;
        let driver = RecordingDriver::new(2 * MIB);
        let mut arena = reserve_recorded(&driver, 64 * MIB);
        arena.map_next(&driver, 2 * MIB, &[0]).unwrap();
        driver.take_ops();
        clear_vmm_faults();
        inject_vmm_fault(VmmFaultKind::AccessReset, 1);
        assert!(arena.map_next(&driver, 2 * MIB, &[0]).is_err());
        clear_vmm_faults();
        let ops = driver.take_ops();
        let unmaps: Vec<_> = ops
            .iter()
            .filter_map(|op| match op {
                DriverOp::Unmap { offset, size } => Some((*offset, *size)),
                _ => None,
            })
            .collect();
        assert_eq!(
            unmaps,
            vec![(2 * MIB, 2 * MIB)],
            "only the fresh, never-used segment may be unmapped"
        );
        // Poisoned: no later growth reaches the driver at all.
        assert!(arena.map_next(&driver, 2 * MIB, &[0]).is_err());
        assert!(driver.take_ops().is_empty(), "a poisoned arena remapped");
    }

    /// The only mid-life unmap is `release`: it synchronizes, unmaps every
    /// segment once (newest first) and releases its handles, retires the VA
    /// (never freed, so no later reserve can land on it) and refuses any
    /// further map on the released arena (the old range is never reused).
    #[test]
    fn gfx1100_vmm_release_retires_the_va_and_never_reuses_it() {
        const MIB: usize = 1 << 20;
        let driver = RecordingDriver::new(2 * MIB);
        let mut arena = reserve_recorded(&driver, 64 * MIB);
        let first_base = arena.base_address();
        for size in [2 * MIB, 4 * MIB, 2 * MIB] {
            arena.map_next(&driver, size, &[0]).unwrap();
        }
        driver.take_ops();
        let retired_before = retired_va_bytes();
        arena.release(&driver).expect("release");
        assert!(arena.is_released());
        assert!(retired_va_bytes() >= retired_before + 64 * MIB);
        let ops = driver.take_ops();
        let unmaps: Vec<_> = ops
            .iter()
            .filter_map(|op| match op {
                DriverOp::Unmap { offset, size } => Some((*offset, *size)),
                _ => None,
            })
            .collect();
        assert_eq!(
            unmaps,
            vec![(6 * MIB, 2 * MIB), (2 * MIB, 4 * MIB), (0, 2 * MIB)]
        );
        assert!(
            ops.iter().position(|op| *op == DriverOp::Sync)
                < ops
                    .iter()
                    .position(|op| matches!(op, DriverOp::Unmap { .. })),
            "release must synchronize before unmapping"
        );
        assert_eq!(ops.iter().filter(|op| **op == DriverOp::Release).count(), 3);
        // The released VA is gone: no remap, and a new reservation (the gather
        // workspace's replacement) is a different range.
        assert!(arena.map_next(&driver, 2 * MIB, &[0]).is_err());
        assert!(driver.take_ops().is_empty());
        let replacement = reserve_recorded(&driver, 64 * MIB);
        assert_ne!(replacement.base_address(), first_base);
        assert_eq!(arena.base_address(), 0);
    }

    /// `hipMemUnmap` has exactly three call sites in the VMM module, each a
    /// teardown of a segment that is not live: import abort, the fresh segment
    /// of a failed access call, and `release`'s teardown. A new call site must
    /// extend this list on purpose.
    #[test]
    fn mem_unmap_call_sites_in_the_vmm_module_are_the_audited_teardowns() {
        let source = include_str!("vmm.rs");
        let production = &source[..source.find("#[cfg(test)]").expect("test module")];
        let mut current = String::new();
        let mut sites = Vec::new();
        for line in production.lines() {
            let trimmed = line.trim_start();
            if let Some(rest) = trimmed
                .strip_prefix("pub fn ")
                .or_else(|| trimmed.strip_prefix("fn "))
                .or_else(|| trimmed.strip_prefix("pub unsafe fn "))
            {
                current = rest
                    .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .next()
                    .unwrap_or_default()
                    .to_string();
            }
            if !trimmed.starts_with("//") && line.contains(".mem_unmap(") {
                sites.push(current.clone());
            }
        }
        sites.sort();
        assert_eq!(
            sites,
            ["abort_import", "map_next", "teardown_segments"],
            "unreviewed hipMemUnmap call site in the VMM module"
        );
    }
}
