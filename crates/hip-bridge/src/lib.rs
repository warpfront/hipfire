// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 alpineq
// hipfire — see LICENSE and NOTICE in the project root.

//! hip-bridge: Safe Rust FFI to AMD HIP runtime via dlopen.
//! Modeled after rustane's ane-bridge — no link-time dependency on libamdhip64.

#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(clippy::multiple_unsafe_ops_per_block)]

mod emu_budget;
mod error;
mod ffi;
mod kernarg;
mod launch_grid;
pub mod null_stream;
mod plan_tape;
mod rccl;
pub mod registry;
mod rocblas;
mod rocsolver;
pub mod site_ledger;
mod vmm;

pub use error::{
    register_first_illegal_address_hook, HipError, HipErrorCode, HipResult, LaunchContext,
    HIP_ERROR_ILLEGAL_ADDRESS, HIP_ERROR_INVALID_IMAGE, HIP_ERROR_OUT_OF_MEMORY,
    HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED, HIP_ERROR_PEER_ACCESS_NOT_ENABLED,
    HIP_ERROR_PEER_ACCESS_UNSUPPORTED,
};
#[cfg(feature = "kernel-timing")]
pub use ffi::kernel_timing::{
    reject_kernel_timing_backend, KernelTimingCoverage, KernelTimingPool, KernelTimingReport,
};
pub use ffi::weight_slab::{
    device_alloc_charge, SlabMode, SlabScopeSummary, SlabStats, ALLOC_GRANULE, DEFAULT_SLAB_MODE,
    SLAB_ALIGN,
};
pub use ffi::{
    arm_hip_fault, declare_qwen4_host_mapping, dispatch_on_worker_threads,
    keep_host_memory_out_of_reclaim,
};
pub use ffi::{launch_counters, memory_effects};
pub use ffi::{
    Event, Function, Graph, GraphExec, HipMemAccessDesc, HipMemAllocationProp,
    HipMemGenericAllocationHandle, HipMemLocation, HipPointerAttribute, HipRuntime, HostAllocOwner,
    InputReady, MappedHostBuffer, MappedPhase, MappedPublished, MappedWrite, Module,
    PinnedHostBuffer, PinnedTransferBuffer, SignalGen, SignalHostPtr, SignalWord, Stream,
    HIP_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE, HIP_ERROR_NOT_READY, HIP_EVENT_DISABLE_TIMING,
    HIP_EVENT_RELEASE_TO_SYSTEM, HIP_HOST_MALLOC_COHERENT, HIP_HOST_MALLOC_MAPPED,
    HIP_MEM_ACCESS_FLAGS_PROT_READ, HIP_MEM_ALLOCATION_GRANULARITY_MINIMUM,
    HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED, HIP_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
    HIP_MEM_LOCATION_TYPE_DEVICE, HIP_MEM_LOCATION_TYPE_HOST, HIP_STREAM_WAIT_VALUE_GTE,
    HIP_STREAM_WRITE_VALUE_DEFAULT, QWEN4_EXPERT_VRAM_LAYERS_ENV, SIGNAL_ABORT, SIGNAL_GEN_MAX,
};
pub use kernarg::KernargBlob;
pub use launch_grid::{check_launch_grid, grid_yz_limit_for_arch, GFX1201_MAX_GRID_YZ};
pub use plan_tape::{PlanEvent, PlanGate, PlanTape};
pub use rccl::{RcclComms, RcclDataType, RcclError, RcclRedOp, RcclResult, NCCL_SUCCESS};
pub use rocblas::{Rocblas, RocblasDatatype, RocblasError, RocblasOperation, RocblasResult};
pub use rocsolver::{
    RocblasDiagonal, RocblasFill, Rocsolver, RocsolverError, RocsolverResult,
    ROCSOLVER_STATUS_SUCCESS,
};
pub use vmm::{
    clear_vmm_faults, inject_vmm_fault, retired_va_bytes, retry_pending_granule_releases,
    vmm_live_granule_bytes, VmmArena, VmmDriver, VmmFaultKind, VmmPhysicalId, VmmSharedPrefix,
};

/// Re-export memory copy direction for callers.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

/// Mirrors `hipMemoryType`. FFI stores raw `u32`; use `from_raw` to convert.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    Unregistered = 0,
    Host = 1,
    Device = 2,
    Managed = 3,
    Array = 10,
    Unified = 11,
}

impl MemoryType {
    pub fn from_raw(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Unregistered),
            1 => Some(Self::Host),
            2 => Some(Self::Device),
            3 => Some(Self::Managed),
            10 => Some(Self::Array),
            11 => Some(Self::Unified),
            _ => None,
        }
    }
}

/// Opaque GPU buffer handle. Tracks pointer + size for safety.
pub struct DeviceBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
    ownership: DeviceBufferOwnership,
    /// Emu capacity charge for an owning `hipMalloc` allocation (`None` when
    /// emu is off, and always `None` for borrowed/VMM/host-mapped views). Not
    /// auto-releasing: dropping the buffer without `HipRuntime::free` keeps
    /// the charge, like the leaked allocation itself.
    token: Option<emu_budget::Token>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeviceBufferOwnership {
    HipMalloc,
    Vmm,
    /// Owner of a `hipHostMalloc(hipHostMallocMapped)` allocation: system RAM the
    /// GPU dereferences over PCIe. Released with `hipHostFree`, and — unlike a
    /// host-located VMM arena — it is NOT charged against the device heap, which
    /// is the whole point of the offload path using it.
    HostMapped,
    Borrowed,
    /// Sub-allocation of a weight slab (see `ffi::weight_slab`). It owns its
    /// range like a `hipMalloc` buffer — `is_hip_allocation` is true, so
    /// ownership checks accept it — but `HipRuntime::free` only releases the
    /// range; the slab's `hipFree` runs when its last view is gone.
    SlabView,
}

impl DeviceBuffer {
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Owning device allocation: a `hipMalloc` buffer or a weight-slab view.
    pub fn is_hip_allocation(&self) -> bool {
        matches!(
            self.ownership,
            DeviceBufferOwnership::HipMalloc | DeviceBufferOwnership::SlabView
        )
    }

    /// Sub-allocation of a weight slab: free it through `HipRuntime::free`
    /// (never pool it past the slab's lifetime).
    pub fn is_slab_view(&self) -> bool {
        self.ownership == DeviceBufferOwnership::SlabView
    }

    pub fn is_vmm_owner(&self) -> bool {
        self.ownership == DeviceBufferOwnership::Vmm
    }

    pub fn is_borrowed(&self) -> bool {
        self.ownership == DeviceBufferOwnership::Borrowed
    }

    pub fn is_host_mapped(&self) -> bool {
        self.ownership == DeviceBufferOwnership::HostMapped
    }

    /// Attach the emu capacity charge of this owning allocation.
    pub(crate) fn with_charge(mut self, token: Option<emu_budget::Token>) -> DeviceBuffer {
        self.token = token;
        self
    }

    /// Detach the emu capacity charge so `free` can release it only after the
    /// raw free succeeds.
    pub(crate) fn take_charge(&mut self) -> Option<emu_budget::Token> {
        self.token.take()
    }

    #[cfg(test)]
    fn is_charged(&self) -> bool {
        self.token.is_some()
    }

    /// Create a non-owning DeviceBuffer from a raw pointer and size.
    /// The caller must ensure the pointer is valid GPU memory.
    /// The resulting buffer must NOT be freed (it doesn't own the memory).
    ///
    /// # Safety
    ///
    /// `ptr` must point to at least `size` bytes of valid GPU-accessible
    /// memory for the lifetime of the returned non-owning wrapper.
    pub unsafe fn from_raw(ptr: *mut std::ffi::c_void, size: usize) -> DeviceBuffer {
        DeviceBuffer {
            ptr,
            size,
            ownership: DeviceBufferOwnership::Borrowed,
            token: None,
        }
    }

    /// Create the unique owner descriptor for a VMM arena base address.
    ///
    /// # Safety
    ///
    /// The caller must register exactly one such descriptor with the VMM owner
    /// that will unmap and release it. Aliases must use `from_raw` or `alias`.
    pub unsafe fn from_vmm_owner(ptr: *mut std::ffi::c_void, size: usize) -> DeviceBuffer {
        DeviceBuffer {
            ptr,
            size,
            ownership: DeviceBufferOwnership::Vmm,
            token: None,
        }
    }

    /// Create the unique owner descriptor for a mapped host allocation.
    ///
    /// `ptr` is the **device-visible** address (what kernels are handed), which
    /// `hipHostGetDevicePointer` may alias differently from the host pointer that
    /// `hipHostMalloc` returned. The host pointer is what `hipHostFree` needs, so
    /// the owner of a host-mapped allocation must register `(ptr -> host_ptr)`
    /// with the `Gpu` that will free it.
    ///
    /// # Safety
    ///
    /// `ptr` must be the device-visible alias of a live `hipHostMalloc` buffer of
    /// at least `size` bytes, freed exactly once through its registered host
    /// pointer.
    pub unsafe fn from_host_mapped(ptr: *mut std::ffi::c_void, size: usize) -> DeviceBuffer {
        DeviceBuffer {
            ptr,
            size,
            ownership: DeviceBufferOwnership::HostMapped,
            token: None,
        }
    }

    /// Create a non-owning alias to the same GPU memory.
    /// The alias must not outlive the original buffer.
    /// Used for reshaping tensors without reallocating.
    /// # Safety
    /// Caller must ensure the alias doesn't outlive the original.
    pub unsafe fn alias(&self) -> DeviceBuffer {
        DeviceBuffer {
            ptr: self.ptr,
            size: self.size,
            ownership: DeviceBufferOwnership::Borrowed,
            token: None,
        }
    }

    /// Non-owning byte-range view into this allocation: bytes
    /// `[byte_offset, byte_offset + byte_len)`.
    ///
    /// This is the single home of the device-pointer reinterpretation that
    /// used to be hand-rolled at every callsite (`as_ptr() as *mut u8`
    /// + `add` + `DeviceBuffer::from_raw`). The range is validated against
    /// the allocation size with checked arithmetic before any pointer
    /// arithmetic runs, so the `add` below is always in-bounds.
    ///
    /// Like [`DeviceBuffer::alias`], the view is `Borrowed`: it must NOT be
    /// freed, and it must not outlive the buffer it was cut from.
    ///
    /// # Safety
    ///
    /// The invariant that holds: `byte_offset + byte_len <= self.size()`,
    /// enforced by the checked range test below, so the derived pointer
    /// addresses only bytes of this live device allocation (`u8` has
    /// alignment 1, so there is no alignment precondition beyond the
    /// allocation itself). The caller must still ensure the view does not
    /// outlive the owning buffer and is never passed to a free path.
    pub fn byte_view(&self, byte_offset: usize, byte_len: usize) -> DeviceBuffer {
        let end = byte_offset
            .checked_add(byte_len)
            .expect("DeviceBuffer::byte_view range overflow");
        assert!(
            end <= self.size,
            "DeviceBuffer::byte_view [{byte_offset}, {end}) exceeds allocation of {} bytes",
            self.size,
        );
        // SAFETY: `end <= self.size` (checked above), so `add(byte_offset)`
        // stays in-bounds of the same allocation and names only live device
        // bytes. The wrapper is Borrowed, so drop never frees it.
        let ptr = unsafe { (self.ptr as *mut u8).add(byte_offset) as *mut std::ffi::c_void };
        DeviceBuffer {
            ptr,
            size: byte_len,
            ownership: DeviceBufferOwnership::Borrowed,
            token: None,
        }
    }
}

// SAFETY: DeviceBuffer is a raw GPU address plus size/ownership tag. Sending
// the handle between threads is fine; the pointees stay on the device. It is
// not Sync: concurrent use of the same allocation requires external stream
// synchronization (aliasing + in-flight lifetime are the caller's contract).
unsafe impl Send for DeviceBuffer {}

#[cfg(test)]
mod device_buffer_tests {
    use super::*;

    #[test]
    fn raw_and_alias_buffers_are_borrowed() {
        // SAFETY: dangling non-null ptr is never dereferenced; tests only check ownership tags.
        let raw = unsafe { DeviceBuffer::from_raw(std::ptr::dangling_mut(), 4096) };
        assert!(raw.is_borrowed());
        assert!(!raw.is_hip_allocation());
        assert!(!raw.is_vmm_owner());

        // SAFETY: alias must not outlive `raw`; both stay on the stack for this test.
        let alias = unsafe { raw.alias() };
        assert!(alias.is_borrowed());
    }

    #[test]
    fn vmm_owner_marker_is_distinct_from_views() {
        // SAFETY: dangling non-null ptr is never dereferenced; tests only check ownership tags.
        let owner = unsafe { DeviceBuffer::from_vmm_owner(std::ptr::dangling_mut(), 4096) };
        assert!(owner.is_vmm_owner());
        assert!(!owner.is_borrowed());
        // SAFETY: alias must not outlive `owner`; both stay on the stack for this test.
        let view = unsafe { owner.alias() };
        assert!(view.is_borrowed());
        assert!(!view.is_vmm_owner());
    }

    #[test]
    fn aliases_views_and_host_mapped_owners_carry_no_charge() {
        // SAFETY: dangling non-null ptr is never dereferenced; tests only check charge tags.
        let mapped = unsafe { DeviceBuffer::from_host_mapped(std::ptr::dangling_mut(), 4096) };
        // SAFETY: alias must not outlive `mapped`; both stay on the stack for this test.
        let alias = unsafe { mapped.alias() };
        let view = mapped.byte_view(0, 1024);
        assert!(!mapped.is_charged() && !alias.is_charged() && !view.is_charged());
        // SAFETY: dangling non-null ptr is never dereferenced.
        let vmm = unsafe { DeviceBuffer::from_vmm_owner(std::ptr::dangling_mut(), 4096) };
        assert!(!vmm.is_charged());
        // SAFETY: dangling non-null ptr is never dereferenced.
        let mut raw = unsafe { DeviceBuffer::from_raw(std::ptr::dangling_mut(), 4096) };
        assert!(raw.take_charge().is_none());
    }
}
