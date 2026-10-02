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

use crate::{
    DeviceBuffer, HipError, HipMemAccessDesc, HipMemAllocationProp, HipMemGenericAllocationHandle,
    HipResult, HipRuntime, HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED,
};
use std::cell::Cell;
use std::ffi::c_void;
use std::sync::Mutex;

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

/// Deterministic, test-only fault injection for VMM teardown/access stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmmFaultKind {
    /// Fail the next `hipMemSetAccess` call(s).
    AccessReset,
    /// Fail the next `hipMemUnmap` call(s).
    Unmap,
    /// Fail the next physical-handle `hipMemRelease` call(s).
    Release,
}

thread_local! {
    static FAULT_ACCESS: Cell<u32> = const { Cell::new(0) };
    static FAULT_UNMAP: Cell<u32> = const { Cell::new(0) };
    static FAULT_RELEASE: Cell<u32> = const { Cell::new(0) };
}

/// Queue `count` deterministic failures for `kind`. Test-only; real HIP is
/// unchanged when the counter is zero.
pub fn inject_vmm_fault(kind: VmmFaultKind, count: u32) {
    match kind {
        VmmFaultKind::AccessReset => FAULT_ACCESS.with(|c| c.set(count)),
        VmmFaultKind::Unmap => FAULT_UNMAP.with(|c| c.set(count)),
        VmmFaultKind::Release => FAULT_RELEASE.with(|c| c.set(count)),
    }
}

/// Clear every pending injected VMM fault.
pub fn clear_vmm_faults() {
    FAULT_ACCESS.with(|c| c.set(0));
    FAULT_UNMAP.with(|c| c.set(0));
    FAULT_RELEASE.with(|c| c.set(0));
}

fn take_fault(kind: VmmFaultKind) -> Option<HipError> {
    let cell = match kind {
        VmmFaultKind::AccessReset => &FAULT_ACCESS,
        VmmFaultKind::Unmap => &FAULT_UNMAP,
        VmmFaultKind::Release => &FAULT_RELEASE,
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
        };
        Some(HipError::new(
            0x564D_4D46, // 'VMMF'
            &format!("injected VMM {label} failure"),
        ))
    })
}

#[derive(Debug)]
struct VmmSegment {
    offset: usize,
    size: usize,
    handle: Option<HipMemGenericAllocationHandle>,
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
}

// SAFETY: VmmArena holds a device VA base and opaque HIP allocation handles that
// may move with model state across threads. Concurrent mutation is excluded
// because VmmArena is not Sync; callers must not free/unmap while another thread
// still has in-flight work on the mapped prefix.
unsafe impl Send for VmmArena {}

impl VmmArena {
    pub fn reserve(hip: &HipRuntime, owner_device: i32, requested_bytes: usize) -> HipResult<Self> {
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
        let prop = HipMemAllocationProp::device_pinned(owner_device);
        let granularity =
            hip.mem_get_allocation_granularity(&prop, HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED)?;
        if granularity == 0 {
            return Err(HipError::new(
                0,
                "HIP returned zero VMM allocation granularity",
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
        self.segments.first().and_then(|seg| seg.handle)
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
        hip: &HipRuntime,
        size: usize,
        access_devices: &[i32],
    ) -> HipResult<()> {
        if self.releasing || self.is_released() {
            return Err(HipError::new(
                0,
                "VMM arena is releasing or already released",
            ));
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
        let prop = HipMemAllocationProp::device_pinned(self.owner_device);
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
        if let Err(err) = take_fault(VmmFaultKind::AccessReset).map_or_else(
            // SAFETY: `self.base..+next_mapped` is the contiguous mapped prefix
            // (just extended by mem_map); `access` lists only owner/peer devices
            // already validated for peer access.
            || unsafe { hip.mem_set_access(self.base, next_mapped, &access) },
            Err,
        ) {
            let err = HipError {
                code: err.code,
                message: format!(
                    "{}; VMM access prefix base=0x{:x} size={} (new segment address=0x{:x} offset={} size={}) granularity={}",
                    err.message,
                    self.base as usize,
                    next_mapped,
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
            mapped: true,
        });
        self.mapped_bytes = next_mapped;
        Ok(())
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

    /// Unmap every segment and release every physical handle. The VA range is
    /// kept reserved and retired, never freed (see the module docs).
    /// Cleanup continues after an individual failure and returns the first one.
    pub fn release(&mut self, hip: &HipRuntime) -> HipResult<()> {
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
        let result = cleanup_segments(
            &mut self.segments,
            |offset, size| {
                if let Some(err) = take_fault(VmmFaultKind::Unmap) {
                    return Err(err);
                }
                // SAFETY: offset/size come from tracked segments; release() already
                // device_synchronize'd every access device so in-flight work is
                // quiesced before unmap. base is the arena's reserved VA.
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
        );

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
    }
    segments.retain(|segment| segment.mapped || segment.handle.is_some());
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
            mapped,
        }
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
        )
        .unwrap_err();
        assert!(err.to_string().contains("injected unmap failure"));
        assert_eq!(releases, 0, "a still-mapped handle must not be released");
        assert!(segments[0].mapped);
        assert!(segments[0].handle.is_some());

        cleanup_segments(&mut segments, |_, _| Ok(()), |_| Ok(())).unwrap();
        assert!(segments.is_empty());
    }

    #[test]
    fn failed_handle_release_keeps_handle_for_retry() {
        let mut segments = vec![segment(false)];
        cleanup_segments(
            &mut segments,
            |_, _| panic!("unmap must not run for an unmapped segment"),
            |_| Err(HipError::new(2, "injected handle failure")),
        )
        .unwrap_err();
        assert!(!segments[0].mapped);
        assert!(segments[0].handle.is_some());

        cleanup_segments(&mut segments, |_, _| Ok(()), |_| Ok(())).unwrap();
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
}
