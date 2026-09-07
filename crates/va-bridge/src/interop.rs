// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipImportExternalMemory` over a VA-exported dma-buf.
//!
//! This HIP version exposes no `DmaBuf` handle type, so the import uses
//! `hipExternalMemoryHandleTypeOpaqueFd` — the same mechanism as the
//! Vulkan-dmabuf → HIP interop path. Proven by `experiments/vcn-jpeg/t0_probe`
//! (import + mapped pointer valid on gfx1201, 2026-09-07).
//!
//! The caller must have selected the HIP device already (e.g. via
//! `hip-bridge`'s `HipRuntime::set_device`); the import binds to the calling
//! thread's current device, which must be the GPU that owns the dma-buf.

use crate::ffi::VaError;
use libloading::{Library, Symbol};
use std::ffi::c_void;

// hip_runtime_api.h (ROCm 7.x, /opt/rocm/include/hip/hip_runtime_api.h)
const HIP_EXT_MEM_HANDLE_OPAQUE_FD: i32 = 1;

#[repr(C)]
struct HipExtMemHandleDesc {
    ty: i32,
    _pad0: u32, // enum tail padding: the union starts at offset 8
    handle_fd: i32,
    // union tail: the C union is 16 bytes (the win32 member is two
    // pointers), so `size` sits at offset 24, not 16.
    _pad1: [u8; 12],
    size: u64,
    flags: u32,
    reserved: [u32; 16],
}
// 4 + 4 + 16 + 8 + 4 + 64 = 100, +4 tail pad to align 8 = 104
const _: () = assert!(std::mem::size_of::<HipExtMemHandleDesc>() == 104);
const _: () = assert!(std::mem::size_of::<HipExtMemBufferDesc>() == 88);

#[repr(C)]
struct HipExtMemBufferDesc {
    offset: u64,
    size: u64,
    flags: u32,
    reserved: [u32; 16],
}
const _: () = assert!(std::mem::size_of::<HipExtMemBufferDesc>() == 88);

pub type HipExternalMemory = *mut c_void;

/// Device mapping of an imported dma-buf. Owns the `HipExternalMemory`
/// handle; `Drop` destroys it. The mapping stays valid after the VA surface
/// is destroyed (HIP holds its own dma-buf reference).
pub struct HipMapping {
    _lib: Library,
    handle: HipExternalMemory,
    ptr: *mut c_void,
    size: usize,
    fn_destroy: unsafe extern "C" fn(HipExternalMemory) -> u32,
}

impl HipMapping {
    /// Import `fd` (a VA-exported dma-buf of `size` bytes) and map it whole.
    /// Consumes nothing: the caller still owns `fd`.
    pub fn import_dma_buf(fd: i32, size: usize) -> Result<Self, VaError> {
        if fd < 0 || size == 0 {
            return Err(VaError::Corrupt("bad dma-buf for HIP import"));
        }
        // Resolve the SAME libamdhip64 instance `hip-bridge`'s HipRuntime
        // uses (hipfire_config::rocm::library_candidates). This host carries
        // two (ROCm 7.15 under /opt/rocm + 7.1 under /usr/lib); importing
        // through a different instance than the one holding the device
        // context fails with hipErrorInvalidValue.
        let candidates =
            hipfire_config::rocm::library_candidates(hipfire_config::rocm::HIP_RUNTIME_LIBRARIES);
        // SAFETY: system HIP runtime; no Rust invariants involved.
        let mut lib = None;
        let mut last_err = String::new();
        for c in &candidates {
            // SAFETY: see above.
            match unsafe { Library::new(c) } {
                Ok(l) => {
                    lib = Some(l);
                    break;
                }
                Err(e) => last_err = e.to_string(),
            }
        }
        let lib = lib.ok_or_else(|| VaError::Dlopen {
            lib: candidates.join(","),
            msg: last_err,
        })?;
        // SAFETY: signatures match hip_runtime_api.h; the library is the same
        // instance HipRuntime uses (candidate policy above).
        unsafe {
            let fn_import: Symbol<
                unsafe extern "C" fn(*mut HipExternalMemory, *const HipExtMemHandleDesc) -> u32,
            > = lib
                .get(b"hipImportExternalMemory")
                .map_err(|_| VaError::MissingSymbol {
                    symbol: "hipImportExternalMemory".to_string(),
                })?;
            let fn_map: Symbol<
                unsafe extern "C" fn(
                    *mut *mut c_void,
                    HipExternalMemory,
                    *const HipExtMemBufferDesc,
                ) -> u32,
            > = lib.get(b"hipExternalMemoryGetMappedBuffer").map_err(|_| {
                VaError::MissingSymbol {
                    symbol: "hipExternalMemoryGetMappedBuffer".to_string(),
                }
            })?;
            let fn_destroy: Symbol<unsafe extern "C" fn(HipExternalMemory) -> u32> = lib
                .get(b"hipDestroyExternalMemory")
                .map_err(|_| VaError::MissingSymbol {
                    symbol: "hipDestroyExternalMemory".to_string(),
                })?;
            // Copy the fn pointer out so no borrow of `lib` survives the move below.
            let fn_destroy_ptr: unsafe extern "C" fn(HipExternalMemory) -> u32 = *fn_destroy;
            let desc = HipExtMemHandleDesc {
                ty: HIP_EXT_MEM_HANDLE_OPAQUE_FD,
                _pad0: 0,
                handle_fd: fd,
                _pad1: [0; 12],
                size: size as u64,
                flags: 0,
                reserved: [0; 16],
            };
            if std::env::var_os("HIPFIRE_VCN_DEBUG").is_some() {
                let raw: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        (&desc as *const HipExtMemHandleDesc) as *const u8,
                        std::mem::size_of::<HipExtMemHandleDesc>(),
                    )
                };
                eprintln!(
                    "[va-bridge] import desc ({}B): {}",
                    raw.len(),
                    hex_bytes(raw)
                );
            }
            let mut handle: HipExternalMemory = std::ptr::null_mut();
            let code = fn_import(&mut handle, &desc);
            if code != 0 || handle.is_null() {
                return Err(VaError::Status {
                    op: "hipImportExternalMemory",
                    code: code as i32,
                    msg: format!("code {code} fd={fd} size={size}"),
                });
            }
            let buf = HipExtMemBufferDesc {
                offset: 0,
                size: size as u64,
                flags: 0,
                reserved: [0; 16],
            };
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let code = fn_map(&mut ptr, handle, &buf);
            if code != 0 || ptr.is_null() {
                fn_destroy_ptr(handle);
                return Err(VaError::Status {
                    op: "hipExternalMemoryGetMappedBuffer",
                    code: code as i32,
                    msg: format!("code {code}"),
                });
            }
            Ok(Self {
                _lib: lib,
                handle,
                ptr,
                size,
                fn_destroy: fn_destroy_ptr,
            })
        }
    }

    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }
    pub fn size(&self) -> usize {
        self.size
    }
    /// Copy `len` bytes at `offset` from the device mapping to host.
    /// Experiment `experiment/vcn-jpeg` staging helper (parity bisection).
    pub fn copy_to_host(&self, offset: usize, len: usize) -> Result<Vec<u8>, VaError> {
        if offset.saturating_add(len) > self.size {
            return Err(VaError::Corrupt("copy_to_host out of range"));
        }
        // SAFETY: signature matches hip_runtime_api.h; the library is the
        // same instance the mapping was created from.
        unsafe {
            let fn_memcpy: Symbol<
                unsafe extern "C" fn(*mut c_void, *const c_void, usize, u32) -> u32,
            > = self
                ._lib
                .get(b"hipMemcpy")
                .map_err(|_| VaError::MissingSymbol {
                    symbol: "hipMemcpy".to_string(),
                })?;
            let mut host = vec![0u8; len];
            let src = (self.ptr as *const u8).add(offset) as *const c_void;
            // hipMemcpyDeviceToHost = 2.
            let code = fn_memcpy(host.as_mut_ptr() as *mut c_void, src, len, 2);
            if code != 0 {
                return Err(VaError::Status {
                    op: "hipMemcpy(D2H)",
                    code: code as i32,
                    msg: format!("code {code}"),
                });
            }
            Ok(host)
        }
    }
}

/// Hex dump for `HIPFIRE_VCN_DEBUG` diagnostics.
fn hex_bytes(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

impl Drop for HipMapping {
    fn drop(&mut self) {
        // SAFETY: handle came from hipImportExternalMemory; exactly-once destroy.
        unsafe {
            (self.fn_destroy)(self.handle);
        }
    }
}

// Mapping is a device pointer wrapper; cross-thread transfer is the caller's responsibility.
unsafe impl Send for HipMapping {}
