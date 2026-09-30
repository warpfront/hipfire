// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 alpineq
// hipfire — see LICENSE and NOTICE in the project root.

//! hip-bridge: Safe Rust FFI to AMD HIP runtime via dlopen.
//! Modeled after rustane's ane-bridge — no link-time dependency on libamdhip64.

mod error;
mod ffi;
mod kernarg;
mod rccl;
mod rocblas;
mod rocsolver;
mod vmm;

pub use error::{
    HipError, HipErrorCode, HipResult, LaunchContext, HIP_ERROR_INVALID_IMAGE,
    HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED, HIP_ERROR_PEER_ACCESS_NOT_ENABLED,
    HIP_ERROR_PEER_ACCESS_UNSUPPORTED,
};
pub use ffi::launch_counters;
pub use ffi::{
    Event, Function, Graph, GraphExec, HipMemAccessDesc, HipMemAllocationProp,
    HipMemGenericAllocationHandle, HipMemLocation, HipPointerAttribute, HipRuntime, Module, Stream,
    HIP_ERROR_NOT_READY, HIP_EVENT_DISABLE_TIMING, HIP_EVENT_RELEASE_TO_SYSTEM,
    HIP_HOST_MALLOC_MAPPED, HIP_MEM_ALLOCATION_GRANULARITY_MINIMUM,
    HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED, HIP_MEM_LOCATION_TYPE_DEVICE,
    HIP_MEM_LOCATION_TYPE_HOST,
};
pub use kernarg::KernargBlob;
pub use rccl::{RcclComms, RcclDataType, RcclError, RcclRedOp, RcclResult, NCCL_SUCCESS};
pub use rocblas::{Rocblas, RocblasDatatype, RocblasError, RocblasOperation, RocblasResult};
pub use rocsolver::{
    RocblasDiagonal, RocblasFill, Rocsolver, RocsolverError, RocsolverResult,
    ROCSOLVER_STATUS_SUCCESS,
};
pub use vmm::{clear_vmm_faults, inject_vmm_fault, VmmArena, VmmFaultKind};

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
}

impl DeviceBuffer {
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn is_hip_allocation(&self) -> bool {
        self.ownership == DeviceBufferOwnership::HipMalloc
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
        }
    }
}

// DeviceBuffer is Send — GPU pointers can be sent between threads.
// They are NOT Sync — concurrent access requires stream synchronization.
unsafe impl Send for DeviceBuffer {}

#[cfg(test)]
mod device_buffer_tests {
    use super::*;

    #[test]
    fn raw_and_alias_buffers_are_borrowed() {
        let raw = unsafe { DeviceBuffer::from_raw(std::ptr::dangling_mut(), 4096) };
        assert!(raw.is_borrowed());
        assert!(!raw.is_hip_allocation());
        assert!(!raw.is_vmm_owner());

        let alias = unsafe { raw.alias() };
        assert!(alias.is_borrowed());
    }

    #[test]
    fn vmm_owner_marker_is_distinct_from_views() {
        let owner = unsafe { DeviceBuffer::from_vmm_owner(std::ptr::dangling_mut(), 4096) };
        assert!(owner.is_vmm_owner());
        assert!(!owner.is_borrowed());
        let view = unsafe { owner.alias() };
        assert!(view.is_borrowed());
        assert!(!view.is_vmm_owner());
    }
}
