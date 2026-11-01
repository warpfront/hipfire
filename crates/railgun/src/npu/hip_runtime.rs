// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! HIP VMM host-backed (system/GTT) memory, GPU initialisation and dma-buf export for NPU argument BOs, plus a
//! small run-time-loaded HIP compute runtime (`HipRuntime`) built on the same dynamic loader and VMM code.
//! `HipDmabufA::create_kind` exports pinned host memory or the distinct uncached-host pool selected with
//! `hipMemAllocationTypeUncached`.
//!
//! `hipHostMalloc` memory is not dma-buf exportable (ROCR rejects CPU-agent allocations) and device mallocs are
//! VRAM, so this uses the HIP virtual-memory-management host path instead. Sequence (`HipDmabufA::create_kind`):
//! `hipInit(0)`, `hipGetDeviceCount`/`hipSetDevice(0)`, `hipMemAllocationProp{type = hipMemAllocationTypePinned,
//! requestedHandleTypes = PosixFileDescriptor (1), location = hipMemLocationTypeHost}`,
//! `hipMemGetAllocationGranularity` (minimum) to round the size, `hipMemCreate`, `hipMemAddressReserve`,
//! `hipMemMap`, `hipMemSetAccess` (GPU 0, read/write), `hipMemcpyHtoD` of the packed A, `hipDeviceSynchronize`,
//! and `hipMemGetHandleForAddressRange(.., hipMemRangeHandleTypeDmaBufFd = 1, flags 0)` for the dma-buf fd.
//!
//! The HIP runtime is loaded at run time (`dlopen` + `dlsym`, libc only), through the same candidate list as the
//! GPU runtime (`hipfire_config::rocm::library_candidates`, never a bare `libamdhip64.so` search), so
//! consumers build and run without ROCm unless a HIP feature is used. Struct layouts and constants
//! follow the installed `hip_runtime_api.h` / `driver_types.h`. The symbols used only by `HipRuntime`
//! (device query, `hipMalloc`, modules, events, ...) are resolved lazily by `HipRuntime::load_for_pci`, so
//! `HipDmabufA` requires exactly the symbols it always did.
//!
//! Lifetime: the `HipDmabufA` value owns the physical allocation, the VA reservation and the exported fd. Drop
//! closes the fd, then `hipMemUnmap`, `hipMemAddressFree`, `hipMemRelease`. The NPU `Bo` imported from the fd (and
//! any hardware context that references it) MUST be dropped first: declare the `HipDmabufA` before the `Device`
//! so reverse declaration order drops it last. The HIP library handle is deliberately never `dlclose`d once HIP
//! has been initialised (the runtime owns threads and its teardown runs at process exit; unloading it earlier is
//! unsupported); it is only closed when symbol lookup fails before any HIP call. Every error string names the
//! stage; HIP failures carry the numeric `hipError_t` and `hipGetErrorString`, dynamic-loader failures the
//! `dlerror` text. No errno is invented for HIP statuses.
//!
//! `HipRuntime` objects (`HipBuffer`, `HipModule`, `HipFunction`, `HipEvent`) each hold a reference-counted
//! handle to the loaded API, so none can outlive the function pointers they release through. They are `!Send`:
//! `hipSetDevice` is per-thread state, so all use stays on the thread that called `load_for_pci`. Release order
//! between independent resources is the caller's: `synchronize()` before dropping buffers/modules used by
//! in-flight launches.
#![allow(dead_code)]
use core::ffi::{c_char, c_int, c_uint, c_ulonglong, c_void, CStr};
use std::ffi::CString;
#[cfg(unix)]
use std::os::fd::RawFd;
use std::rc::Rc;

#[cfg(unix)]
mod sys {
    use core::ffi::{c_char, c_int, c_void};
    extern "C" {
        pub fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        pub fn dlclose(handle: *mut c_void) -> c_int;
        pub fn dlerror() -> *mut c_char;
        pub fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
        pub fn close(fd: c_int) -> c_int;
    }
    /// `struct dl_addr` of `dladdr(3)`; only `dli_fname` is read.
    #[repr(C)]
    pub struct DlInfo {
        pub dli_fname: *const c_char,
        pub dli_fbase: *mut c_void,
        pub dli_sname: *const c_char,
        pub dli_saddr: *mut c_void,
    }
    pub const RTLD_NOW: c_int = 2;
    pub const RTLD_NOLOAD: c_int = 4;
}
/// Windows has no dlopen/dlsym/dlclose/dlerror/dladdr libc surface: every entry
/// fails closed (null handle/symbol, error return, message string), which the
/// existing null/error checks turn into `Err`. No HIP runtime can load through
/// this backend on Windows; Linux behavior is unchanged.
#[cfg(windows)]
mod sys {
    use core::ffi::{c_char, c_int, c_void};
    pub unsafe fn dlopen(_filename: *const c_char, _flags: c_int) -> *mut c_void {
        core::ptr::null_mut()
    }
    pub unsafe fn dlsym(_handle: *mut c_void, _symbol: *const c_char) -> *mut c_void {
        core::ptr::null_mut()
    }
    pub unsafe fn dlclose(_handle: *mut c_void) -> c_int {
        -1
    }
    pub unsafe fn dlerror() -> *mut c_char {
        b"NPU HIP loader unavailable on Windows\0".as_ptr() as *mut c_char
    }
    pub unsafe fn dladdr(_addr: *const c_void, _info: *mut DlInfo) -> c_int {
        0
    }
    /// `struct dl_addr` of `dladdr(3)`; only `dli_fname` is read.
    #[repr(C)]
    pub struct DlInfo {
        pub dli_fname: *const c_char,
        pub dli_fbase: *mut c_void,
        pub dli_sname: *const c_char,
        pub dli_saddr: *mut c_void,
    }
    pub const RTLD_NOW: c_int = 2;
    pub const RTLD_NOLOAD: c_int = 4;
}

/// HIP runtime label for error text.
const HIP_LIB_LABEL: &str = "libamdhip64.so";
/// Error-prefix of `HipDmabufA` (unchanged since it was the only user).
const DMABUF_PREFIX: &str = "a-from-hip";
/// Error-prefix of `HipRuntime`.
const RUNTIME_PREFIX: &str = "hip";
/// `hipMemRangeHandleTypeDmaBufFd`.
const HIP_MEM_RANGE_HANDLE_DMABUF_FD: c_int = 1;
/// `hipMemHandleTypePosixFileDescriptor`.
const HIP_MEM_HANDLE_TYPE_POSIX_FD: c_int = 1;
/// `hipMemAllocationTypePinned`.
const HIP_MEM_ALLOCATION_TYPE_PINNED: c_int = 0x1;
/// `hipMemAllocationTypeUncached` (HIP extension, hip_runtime_api.h).
const HIP_MEM_ALLOCATION_TYPE_UNCACHED: c_int = 0x40000000;
/// `hipMemLocationTypeDevice` / `hipMemLocationTypeHost`.
const HIP_MEM_LOCATION_DEVICE: c_int = 1;
const HIP_MEM_LOCATION_HOST: c_int = 2;
/// `hipMemAccessFlagsProtReadWrite`.
const HIP_MEM_ACCESS_READ_WRITE: c_int = 3;
/// `hipMemAllocationGranularityMinimum`.
const HIP_MEM_GRANULARITY_MINIMUM: c_int = 0;
/// `hipDeviceAttributeMultiprocessorCount` (installed hip_runtime_api.h enum value; counts WGPs, not CUs, when the
/// device runs in WGP mode).
const HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 63;
/// `hipDeviceAttributeWallClockRate` (installed hip_runtime_api.h: `hipDeviceAttributeAmdSpecificBegin` 10000 + 17), kHz of
/// the constant-rate counter behind `wall_clock64` / `s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME`.
const HIP_DEVICE_ATTRIBUTE_WALL_CLOCK_RATE: c_int = 10017;
/// `offsetof(hipDeviceProp_t, gcnArchName)` (`char[256]`) in `hipDeviceProp_tR0600`, whose `sizeof` is 1472
/// (hip_runtime_api.h; `hipGetDeviceProperties` is `#define`d to `hipGetDevicePropertiesR0600`).
const HIP_DEVICE_PROP_GCN_ARCH_NAME_OFFSET: usize = 1160;
const HIP_DEVICE_PROP_GCN_ARCH_NAME_LEN: usize = 256;
/// Scratch for `hipDeviceProp_tR0600` (1472 bytes today); oversized so a newer runtime cannot overrun it.
const HIP_DEVICE_PROP_SCRATCH: usize = 4096;
/// `HIP_LAUNCH_PARAM_BUFFER_POINTER` / `_BUFFER_SIZE` / `_END`.
const HIP_LAUNCH_PARAM_BUFFER_POINTER: usize = 0x01;
const HIP_LAUNCH_PARAM_BUFFER_SIZE: usize = 0x02;
const HIP_LAUNCH_PARAM_END: usize = 0x03;
/// Capacity for `hipDeviceGetPCIBusId` ("dddd:bb:dd.f" plus NUL needs 13).
const PCI_BUS_ID_CAP: usize = 64;

type HipStatus = c_int;
type Handle = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct MemLocation {
    ty: c_int,
    id: c_int,
}

/// `hipMemAllocationProp` (driver_types / hip_runtime_api.h): 32 bytes.
#[repr(C)]
struct AllocProp {
    ty: c_int,
    requested_handle_types: c_int,
    location: MemLocation,
    win32_handle_meta_data: *mut c_void,
    compression_type: u8,
    gpu_direct_rdma_capable: u8,
    usage: u16,
}

/// `hipMemAccessDesc`: 12 bytes.
#[repr(C)]
struct AccessDesc {
    location: MemLocation,
    flags: c_int,
}

const _: () =
    assert!(core::mem::size_of::<AllocProp>() == 32 && core::mem::size_of::<AccessDesc>() == 12);
const _: () = assert!(
    HIP_DEVICE_PROP_GCN_ARCH_NAME_OFFSET + HIP_DEVICE_PROP_GCN_ARCH_NAME_LEN
        <= HIP_DEVICE_PROP_SCRATCH
);

/// Every HIP symbol required by `Api`, in `Api` field order.
const SYMBOLS: [&str; 15] = [
    "hipInit",
    "hipGetDeviceCount",
    "hipSetDevice",
    "hipGetErrorString",
    "hipMemGetAllocationGranularity",
    "hipMemCreate",
    "hipMemAddressReserve",
    "hipMemMap",
    "hipMemSetAccess",
    "hipMemcpyHtoD",
    "hipDeviceSynchronize",
    "hipMemGetHandleForAddressRange",
    "hipMemUnmap",
    "hipMemAddressFree",
    "hipMemRelease",
];

/// Symbols required only by `HipRuntime`, in `GpuApi` field order.
const GPU_SYMBOLS: [&str; 22] = [
    "hipDeviceGetPCIBusId",
    "hipGetDevicePropertiesR0600",
    "hipDeviceGetAttribute",
    "hipMalloc",
    "hipFree",
    "hipMemcpyDtoH",
    "hipMemset",
    "hipModuleLoadData",
    "hipModuleGetFunction",
    "hipModuleUnload",
    "hipModuleLaunchKernel",
    "hipEventCreate",
    "hipEventRecord",
    "hipEventSynchronize",
    "hipEventElapsedTime",
    "hipEventDestroy",
    "hipHostRegister",
    "hipHostGetDevicePointer",
    "hipHostUnregister",
    "hipMemcpyDtoD",
    "hipStreamCreateWithFlags",
    "hipStreamDestroy",
];

/// Resolved HIP entry points shared by `HipDmabufA` and `HipRuntime` (all required).
struct Api {
    /// Library handle from `dlopen`; kept open for the life of the process once HIP is initialised.
    lib: *mut c_void,
    /// What `lib` was opened as: an already-loaded soname or the candidate path (error text).
    source: String,
    /// Error-string prefix of the owning consumer.
    prefix: &'static str,
    init: unsafe extern "C" fn(c_uint) -> HipStatus,
    get_device_count: unsafe extern "C" fn(*mut c_int) -> HipStatus,
    set_device: unsafe extern "C" fn(c_int) -> HipStatus,
    error_string: unsafe extern "C" fn(HipStatus) -> *const c_char,
    mem_granularity: unsafe extern "C" fn(*mut usize, *const AllocProp, c_int) -> HipStatus,
    mem_create:
        unsafe extern "C" fn(*mut Handle, usize, *const AllocProp, c_ulonglong) -> HipStatus,
    addr_reserve:
        unsafe extern "C" fn(*mut *mut c_void, usize, usize, *mut c_void, c_ulonglong) -> HipStatus,
    mem_map: unsafe extern "C" fn(*mut c_void, usize, usize, Handle, c_ulonglong) -> HipStatus,
    set_access: unsafe extern "C" fn(*mut c_void, usize, *const AccessDesc, usize) -> HipStatus,
    memcpy_htod: unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> HipStatus,
    device_synchronize: unsafe extern "C" fn() -> HipStatus,
    get_handle_for_address_range:
        unsafe extern "C" fn(*mut c_void, *mut c_void, usize, c_int, c_ulonglong) -> HipStatus,
    mem_unmap: unsafe extern "C" fn(*mut c_void, usize) -> HipStatus,
    addr_free: unsafe extern "C" fn(*mut c_void, usize) -> HipStatus,
    mem_release: unsafe extern "C" fn(Handle) -> HipStatus,
}

/// Entry points used only by `HipRuntime`, resolved lazily from the already-loaded library.
struct GpuApi {
    get_pci_bus_id: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> HipStatus,
    get_properties: unsafe extern "C" fn(*mut c_void, c_int) -> HipStatus,
    get_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> HipStatus,
    malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> HipStatus,
    free: unsafe extern "C" fn(*mut c_void) -> HipStatus,
    memcpy_dtoh: unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> HipStatus,
    memset: unsafe extern "C" fn(*mut c_void, c_int, usize) -> HipStatus,
    module_load_data: unsafe extern "C" fn(*mut Handle, *const c_void) -> HipStatus,
    module_get_function: unsafe extern "C" fn(*mut Handle, Handle, *const c_char) -> HipStatus,
    module_unload: unsafe extern "C" fn(Handle) -> HipStatus,
    module_launch_kernel: unsafe extern "C" fn(
        Handle,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        Handle,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> HipStatus,
    event_create: unsafe extern "C" fn(*mut Handle) -> HipStatus,
    event_record: unsafe extern "C" fn(Handle, Handle) -> HipStatus,
    event_synchronize: unsafe extern "C" fn(Handle) -> HipStatus,
    event_elapsed_time: unsafe extern "C" fn(*mut f32, Handle, Handle) -> HipStatus,
    event_destroy: unsafe extern "C" fn(Handle) -> HipStatus,
    host_register: unsafe extern "C" fn(*mut c_void, usize, c_uint) -> HipStatus,
    host_get_device_pointer:
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, c_uint) -> HipStatus,
    host_unregister: unsafe extern "C" fn(*mut c_void) -> HipStatus,
    memcpy_dtod: unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> HipStatus,
    stream_create: unsafe extern "C" fn(*mut Handle, c_uint) -> HipStatus,
    stream_destroy: unsafe extern "C" fn(Handle) -> HipStatus,
}

fn dl_error() -> String {
    let p = unsafe { sys::dlerror() };
    if p.is_null() {
        "no dlerror text".to_string()
    } else {
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

/// Loader-reported path backing `ptr`, for diagnostics only. Filesystem paths
/// can change after a library is loaded; they do not identify a live mapping.
pub(crate) fn dladdr_fname(ptr: *const c_void) -> Option<String> {
    let mut info = sys::DlInfo {
        dli_fname: core::ptr::null(),
        dli_fbase: core::ptr::null_mut(),
        dli_sname: core::ptr::null(),
        dli_saddr: core::ptr::null_mut(),
    };
    // SAFETY: `info` is a valid out-pointer for the lifetime of the call; dladdr
    // only writes into it.
    if unsafe { sys::dladdr(ptr, &mut info) } == 0 || info.dli_fname.is_null() {
        return None;
    }
    // SAFETY: dladdr leaves a NUL-terminated path there on success.
    Some(
        unsafe { CStr::from_ptr(info.dli_fname) }
            .to_string_lossy()
            .into_owned(),
    )
}

/// dlsym every name in `names` from `lib` (opened as `source`); `Err` lists the missing ones (the library is NOT
/// closed here).
fn resolve<const N: usize>(
    lib: *mut c_void,
    source: &str,
    names: [&str; N],
    prefix: &str,
) -> Result<[*mut c_void; N], String> {
    let mut ptrs = [core::ptr::null_mut::<c_void>(); N];
    let mut missing: Vec<&str> = Vec::new();
    for (slot, name) in ptrs.iter_mut().zip(names) {
        let cname = format!("{name}\0");
        *slot = unsafe { sys::dlsym(lib, cname.as_ptr() as *const c_char) };
        if slot.is_null() {
            missing.push(name);
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "{prefix} stage dlsym: missing in {source}: {} ({})",
            missing.join(", "),
            dl_error()
        ));
    }
    Ok(ptrs)
}

/// One `dlopen` attempt of the HIP runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Attempt<'a> {
    /// Soname (probe) or candidate path.
    name: &'a str,
    /// `RTLD_NOLOAD`: only reuse a copy this process already mapped, never load a new one.
    already_loaded: bool,
}

/// The ordered `dlopen` attempts: first every soname as an `RTLD_NOLOAD` probe, so a HIP runtime the process
/// already mapped (the GPU runtime's) is reused instead of mapping a second install; then each of
/// `candidates` (the output of `hipfire_config::rocm::library_candidates`, in its order) as a real load. No bare
/// soname is ever loaded, so ldconfig cannot mix in another ROCm install behind the selected root.
fn load_plan<'a>(
    sonames: &'a [&'a str],
    candidates: &'a [String],
) -> impl Iterator<Item = Attempt<'a>> + 'a {
    let probes = sonames.iter().map(|s| Attempt {
        name: s,
        already_loaded: true,
    });
    let loads = candidates.iter().map(|c| Attempt {
        name: c.as_str(),
        already_loaded: false,
    });
    probes.chain(loads)
}

/// Open the HIP runtime following `load_plan`; returns the handle and the name it opened as.
fn open_hip(prefix: &str) -> Result<(*mut c_void, String), String> {
    let sonames = hipfire_config::rocm::HIP_RUNTIME_LIBRARIES;
    let candidates = hipfire_config::rocm::library_candidates(sonames);
    let mut last_err = None;
    for attempt in load_plan(sonames, &candidates) {
        let Ok(cname) = CString::new(attempt.name) else {
            continue;
        };
        let flags = if attempt.already_loaded {
            sys::RTLD_NOW | sys::RTLD_NOLOAD
        } else {
            sys::RTLD_NOW
        };
        let lib = unsafe { sys::dlopen(cname.as_ptr(), flags) };
        if !lib.is_null() {
            return Ok((lib, attempt.name.to_owned()));
        }
        if !attempt.already_loaded {
            last_err = Some(format!("{}: {}", attempt.name, dl_error()));
        }
    }
    Err(format!(
        "{prefix} stage dlopen: {HIP_LIB_LABEL} failed: {}.\n{}",
        last_err.unwrap_or_else(|| "no candidates".into()),
        hipfire_config::rocm::resolution_failure("the HIP runtime (libamdhip64.so)", &candidates)
    ))
}

/// dlopen + dlsym every required symbol; on a missing symbol the not-yet-initialised library is closed again.
fn load(prefix: &'static str) -> Result<Api, String> {
    let (lib, source) = open_hip(prefix)?;
    let ptrs = match resolve(lib, &source, SYMBOLS, prefix) {
        Ok(p) => p,
        Err(e) => {
            unsafe { sys::dlclose(lib) };
            return Err(e);
        }
    };
    // SAFETY: each pointer is the HIP entry point of that name (SYMBOLS order); the declared signatures match
    // hip_runtime_api.h (hipDeviceptr_t = void*, hipMemGenericAllocationHandle_t = opaque pointer).
    unsafe {
        use core::mem::transmute as t;
        Ok(Api {
            lib,
            source,
            prefix,
            init: t(ptrs[0]),
            get_device_count: t(ptrs[1]),
            set_device: t(ptrs[2]),
            error_string: t(ptrs[3]),
            mem_granularity: t(ptrs[4]),
            mem_create: t(ptrs[5]),
            addr_reserve: t(ptrs[6]),
            mem_map: t(ptrs[7]),
            set_access: t(ptrs[8]),
            memcpy_htod: t(ptrs[9]),
            device_synchronize: t(ptrs[10]),
            get_handle_for_address_range: t(ptrs[11]),
            mem_unmap: t(ptrs[12]),
            addr_free: t(ptrs[13]),
            mem_release: t(ptrs[14]),
        })
    }
}

/// `load` plus the `HipRuntime`-only symbols; either lookup failing closes the not-yet-initialised library.
fn load_with_gpu(prefix: &'static str) -> Result<(Api, GpuApi), String> {
    let api = load(prefix)?;
    let ptrs = match resolve(api.lib, &api.source, GPU_SYMBOLS, prefix) {
        Ok(p) => p,
        Err(e) => {
            unsafe { sys::dlclose(api.lib) };
            return Err(e);
        }
    };
    // SAFETY: each pointer is the HIP entry point of that name (GPU_SYMBOLS order); signatures follow
    // hip_runtime_api.h (hipDeviceptr_t = void*, hipModule_t/hipFunction_t/hipEvent_t/hipStream_t = opaque pointers).
    let gpu = unsafe {
        use core::mem::transmute as t;
        GpuApi {
            get_pci_bus_id: t(ptrs[0]),
            get_properties: t(ptrs[1]),
            get_attribute: t(ptrs[2]),
            malloc: t(ptrs[3]),
            free: t(ptrs[4]),
            memcpy_dtoh: t(ptrs[5]),
            memset: t(ptrs[6]),
            module_load_data: t(ptrs[7]),
            module_get_function: t(ptrs[8]),
            module_unload: t(ptrs[9]),
            module_launch_kernel: t(ptrs[10]),
            event_create: t(ptrs[11]),
            event_record: t(ptrs[12]),
            event_synchronize: t(ptrs[13]),
            event_elapsed_time: t(ptrs[14]),
            event_destroy: t(ptrs[15]),
            host_register: t(ptrs[16]),
            host_get_device_pointer: t(ptrs[17]),
            host_unregister: t(ptrs[18]),
            memcpy_dtod: t(ptrs[19]),
            stream_create: t(ptrs[20]),
            stream_destroy: t(ptrs[21]),
        }
    };
    Ok((api, gpu))
}

impl Api {
    /// `Ok` for hipSuccess (0); else the stage, call, numeric status and `hipGetErrorString`.
    fn check(&self, stage: &str, call: &str, status: HipStatus) -> Result<(), String> {
        if status == 0 {
            return Ok(());
        }
        let p = unsafe { (self.error_string)(status) };
        let text = if p.is_null() {
            "<null hipGetErrorString>".to_string()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        };
        Err(format!(
            "{} stage {stage}: {call} failed: HIP status {status} ({text})",
            self.prefix
        ))
    }
}

/// Exportable host VMM physical-memory modes; neither requests a VRAM allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HipMemoryKind {
    VmmHost,
    VmmHostUncached,
}

impl HipMemoryKind {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "vmm-host" => Ok(Self::VmmHost),
            "vmm-host-uncached" => Ok(Self::VmmHostUncached),
            _ => Err(format!(
                "unknown HIP memory mode {value:?}; expected vmm-host or vmm-host-uncached"
            )),
        }
    }

    pub fn describe(self) -> &'static str {
        match self {
            Self::VmmHost => "vmm-host",
            Self::VmmHostUncached => "vmm-host-uncached",
        }
    }

    fn allocation_type(self) -> c_int {
        match self {
            Self::VmmHost => HIP_MEM_ALLOCATION_TYPE_PINNED,
            Self::VmmHostUncached => HIP_MEM_ALLOCATION_TYPE_UNCACHED,
        }
    }
}

/// One host VMM physical allocation mapped at a reserved VA and made read/write for one GPU. Shared by the
/// dma-buf exporter and `HipRuntime` buffers. Drop: `hipMemUnmap`, `hipMemAddressFree`, `hipMemRelease`.
struct Vmm {
    api: Rc<Api>,
    /// Physical allocation handle from `hipMemCreate` (null until created).
    handle: Handle,
    /// Reserved VA (null until reserved) of `alloc` bytes; `mapped` once `hipMemMap` succeeded.
    va: *mut c_void,
    mapped: bool,
    /// Granularity-rounded allocation size.
    alloc: usize,
}

impl Vmm {
    /// Granularity-round `bytes`, create, reserve, map and grant `device` read/write. Any failure releases what
    /// was acquired so far (via `Drop`).
    fn create(
        api: &Rc<Api>,
        bytes: usize,
        kind: HipMemoryKind,
        device: c_int,
    ) -> Result<Self, String> {
        let mut s = Vmm {
            api: Rc::clone(api),
            handle: core::ptr::null_mut(),
            va: core::ptr::null_mut(),
            mapped: false,
            alloc: 0,
        };
        let prop = AllocProp {
            ty: kind.allocation_type(),
            requested_handle_types: HIP_MEM_HANDLE_TYPE_POSIX_FD,
            location: MemLocation {
                ty: HIP_MEM_LOCATION_HOST,
                id: 0,
            },
            win32_handle_meta_data: core::ptr::null_mut(),
            compression_type: 0,
            gpu_direct_rdma_capable: 0,
            usage: 0,
        };
        let mut gran = 0usize;
        api.check(
            "hipMemGetAllocationGranularity",
            "hipMemGetAllocationGranularity(host, minimum)",
            unsafe { (api.mem_granularity)(&mut gran, &prop, HIP_MEM_GRANULARITY_MINIMUM) },
        )?;
        if gran == 0 {
            return Err(format!(
                "{} stage hipMemGetAllocationGranularity: granularity 0",
                api.prefix
            ));
        }
        let alloc = bytes
            .checked_add(gran - 1)
            .map(|n| n / gran * gran)
            .ok_or_else(|| "HIP VMM allocation size overflow".to_string())?;
        s.alloc = alloc;
        let mut handle: Handle = core::ptr::null_mut();
        api.check(
            "hipMemCreate",
            &format!("hipMemCreate({alloc} B, {})", kind.describe()),
            unsafe { (api.mem_create)(&mut handle, alloc, &prop, 0) },
        )?;
        s.handle = handle;
        let mut va: *mut c_void = core::ptr::null_mut();
        api.check(
            "hipMemAddressReserve",
            &format!("hipMemAddressReserve({alloc} B)"),
            unsafe { (api.addr_reserve)(&mut va, alloc, 0, core::ptr::null_mut(), 0) },
        )?;
        s.va = va;
        api.check("hipMemMap", &format!("hipMemMap({alloc} B)"), unsafe {
            (api.mem_map)(va, alloc, 0, handle, 0)
        })?;
        s.mapped = true;
        let access = AccessDesc {
            location: MemLocation {
                ty: HIP_MEM_LOCATION_DEVICE,
                id: device,
            },
            flags: HIP_MEM_ACCESS_READ_WRITE,
        };
        api.check(
            "hipMemSetAccess",
            &format!("hipMemSetAccess(device {device}, read/write)"),
            unsafe { (api.set_access)(va, alloc, &access, 1) },
        )?;
        Ok(s)
    }
}

impl Drop for Vmm {
    fn drop(&mut self) {
        unsafe {
            if self.mapped {
                (self.api.mem_unmap)(self.va, self.alloc);
            }
            if !self.va.is_null() {
                (self.api.addr_free)(self.va, self.alloc);
            }
            if !self.handle.is_null() {
                (self.api.mem_release)(self.handle);
            }
        }
    }
}

/// Packed A in HIP VMM host-backed memory plus its exported dma-buf fd. See the module docs for drop order.
#[cfg(unix)]
pub struct HipDmabufA {
    api: Rc<Api>,
    kind: HipMemoryKind,
    /// Host VMM allocation (None until created). Dropped after the fd is closed.
    vmm: Option<Vmm>,
    /// Bytes of packed A.
    bytes: usize,
    /// Exported dma-buf fd (-1 until exported).
    fd: RawFd,
}

#[cfg(unix)]
impl HipDmabufA {
    /// Initialise and export either supported host physical-memory pool. Does not fall back to another mode.
    pub fn create_kind(packed_a: &[u8], kind: HipMemoryKind) -> Result<Self, String> {
        if packed_a.is_empty() {
            return Err("a-from-hip stage args: packed A is empty".to_string());
        }
        let api = Rc::new(load(DMABUF_PREFIX)?);
        // From here Drop releases whatever was acquired.
        let mut s = HipDmabufA {
            api,
            kind,
            vmm: None,
            bytes: packed_a.len(),
            fd: -1,
        };
        s.build(packed_a)?;
        Ok(s)
    }

    fn build(&mut self, packed_a: &[u8]) -> Result<(), String> {
        let api = Rc::clone(&self.api);
        api.check("hipInit", "hipInit(0)", unsafe { (api.init)(0) })?;
        let mut count: c_int = 0;
        api.check("hipGetDeviceCount", "hipGetDeviceCount", unsafe {
            (api.get_device_count)(&mut count)
        })?;
        if count < 1 {
            return Err(format!(
                "a-from-hip stage hipGetDeviceCount: HIP reports {count} devices"
            ));
        }
        api.check("hipSetDevice", "hipSetDevice(0)", unsafe {
            (api.set_device)(0)
        })?;
        let vmm = Vmm::create(&api, packed_a.len(), self.kind, 0)?;
        let (va, alloc) = (vmm.va, vmm.alloc);
        self.vmm = Some(vmm);
        api.check(
            "hipMemcpyHtoD",
            &format!("hipMemcpyHtoD(packed A {} B)", packed_a.len()),
            unsafe { (api.memcpy_htod)(va, packed_a.as_ptr() as *const c_void, packed_a.len()) },
        )?;
        api.check("hipDeviceSynchronize", "hipDeviceSynchronize", unsafe {
            (api.device_synchronize)()
        })?;
        let mut fd: c_int = -1;
        api.check(
            "hipMemGetHandleForAddressRange",
            &format!("hipMemGetHandleForAddressRange(dmabuf fd, {alloc} B)"),
            unsafe {
                (api.get_handle_for_address_range)(
                    &mut fd as *mut c_int as *mut c_void,
                    va,
                    alloc,
                    HIP_MEM_RANGE_HANDLE_DMABUF_FD,
                    0,
                )
            },
        )?;
        if fd < 0 {
            return Err(format!("a-from-hip stage hipMemGetHandleForAddressRange: returned success with invalid fd {fd}"));
        }
        self.fd = fd;
        Ok(())
    }

    /// The exported dma-buf fd (owned by `self`; `import_dmabuf` takes its own reference).
    pub fn fd(&self) -> RawFd {
        self.fd
    }

    /// Bytes of packed A (the import size).
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// One-line description of the allocation for logs.
    pub fn describe(&self) -> String {
        let (handle, va, alloc) = match &self.vmm {
            Some(v) => (v.handle, v.va, v.alloc),
            None => (core::ptr::null_mut(), core::ptr::null_mut(), 0),
        };
        format!(
            "memory={} location=host handle={:p} gpu_va={:p} bytes={} alloc={} dmabuf_fd={}",
            self.kind.describe(),
            handle,
            va,
            self.bytes,
            alloc,
            self.fd
        )
    }
}

#[cfg(unix)]
impl Drop for HipDmabufA {
    fn drop(&mut self) {
        if self.fd >= 0 {
            unsafe { sys::close(self.fd) };
        }
        // `vmm` drops after this body: unmap, address free, release.
    }
}

/// Loaded API plus the selected device; shared by every `HipRuntime` resource.
struct Shared {
    api: Rc<Api>,
    gpu: GpuApi,
    /// HIP device ordinal bound to the PCI address given to `load_for_pci`.
    device: c_int,
}

impl Shared {
    fn check(&self, stage: &str, call: &str, status: HipStatus) -> Result<(), String> {
        self.api.check(stage, call, status)
    }
}

/// HIP compute runtime bound to one GPU selected by PCI address. See the module docs for threading/lifetime.
pub struct HipRuntime {
    shared: Rc<Shared>,
}

impl HipRuntime {
    /// Address of this runtime's resolved `hipEventRecord` implementation.
    /// Compare only while both runtime owners are alive, in the same process.
    /// This makes no HIP call; the optional pathname is diagnostic only.
    pub fn event_record_address(&self) -> usize {
        self.shared.gpu.event_record as usize
    }

    /// Loader-reported path of this runtime's `hipEventRecord` implementation,
    /// for diagnostics. A missing path never proves equality with another runtime.
    pub fn mapped_hip_library(&self) -> Option<String> {
        dladdr_fname(self.shared.gpu.event_record as *const c_void)
    }

    /// Load HIP, `hipInit`, find the device whose `hipDeviceGetPCIBusId` equals `pci` (e.g. `"0000:bf:00.0"`, ASCII
    /// case-insensitive) and `hipSetDevice` it. No other device is ever selected.
    pub fn load_for_pci(pci: &str) -> Result<Self, String> {
        let (api, gpu) = load_with_gpu(RUNTIME_PREFIX)?;
        api.check("hipInit", "hipInit(0)", unsafe { (api.init)(0) })?;
        let mut count: c_int = 0;
        api.check("hipGetDeviceCount", "hipGetDeviceCount", unsafe {
            (api.get_device_count)(&mut count)
        })?;
        if count < 1 {
            return Err(format!(
                "{RUNTIME_PREFIX} stage hipGetDeviceCount: HIP reports {count} devices"
            ));
        }
        let mut seen: Vec<String> = Vec::new();
        let mut found: Option<c_int> = None;
        for index in 0..count {
            let mut buf = [0 as c_char; PCI_BUS_ID_CAP];
            api.check(
                "hipDeviceGetPCIBusId",
                &format!("hipDeviceGetPCIBusId(device {index})"),
                unsafe { (gpu.get_pci_bus_id)(buf.as_mut_ptr(), PCI_BUS_ID_CAP as c_int, index) },
            )?;
            // SAFETY: buf is zero-initialised with a spare byte beyond what HIP wrote, so it is NUL-terminated.
            let id = unsafe { CStr::from_ptr(buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            if id.eq_ignore_ascii_case(pci) && found.is_none() {
                found = Some(index);
            }
            seen.push(format!("{index}={id}"));
        }
        let device = found.ok_or_else(|| {
            format!("{RUNTIME_PREFIX} stage hipDeviceGetPCIBusId: no HIP device at PCI {pci}; HIP devices: {}", seen.join(", "))
        })?;
        api.check(
            "hipSetDevice",
            &format!("hipSetDevice({device}) for PCI {pci}"),
            unsafe { (api.set_device)(device) },
        )?;
        Ok(HipRuntime {
            shared: Rc::new(Shared {
                api: Rc::new(api),
                gpu,
                device,
            }),
        })
    }

    /// GCN architecture name of the device (`gcnArchName` from `hipGetDevicePropertiesR0600`, e.g. `"gfx1151"`),
    /// with any `:feature+/-` suffix removed.
    pub fn device_arch(&self) -> Result<String, String> {
        let s = &*self.shared;
        #[repr(C, align(16))]
        struct PropScratch([u8; HIP_DEVICE_PROP_SCRATCH]);
        let mut prop = PropScratch([0; HIP_DEVICE_PROP_SCRATCH]);
        s.check(
            "hipGetDeviceProperties",
            &format!("hipGetDevicePropertiesR0600(device {})", s.device),
            unsafe { (s.gpu.get_properties)(prop.0.as_mut_ptr() as *mut c_void, s.device) },
        )?;
        let field = &prop.0[HIP_DEVICE_PROP_GCN_ARCH_NAME_OFFSET
            ..HIP_DEVICE_PROP_GCN_ARCH_NAME_OFFSET + HIP_DEVICE_PROP_GCN_ARCH_NAME_LEN];
        let name = CStr::from_bytes_until_nul(field)
            .map_err(|_| format!("{RUNTIME_PREFIX} stage hipGetDeviceProperties: gcnArchName is not NUL-terminated"))?
            .to_string_lossy();
        let arch = name.split(':').next().unwrap_or("");
        if arch.is_empty() {
            return Err(format!(
                "{RUNTIME_PREFIX} stage hipGetDeviceProperties: empty gcnArchName"
            ));
        }
        Ok(arch.to_string())
    }

    /// `hipDeviceAttributeMultiprocessorCount` of the device (WGPs rather than CUs when the device runs in WGP mode).
    pub fn compute_units(&self) -> Result<u32, String> {
        let s = &*self.shared;
        let mut value: c_int = 0;
        s.check(
            "hipDeviceGetAttribute",
            &format!(
                "hipDeviceGetAttribute(MultiprocessorCount, device {})",
                s.device
            ),
            unsafe {
                (s.gpu.get_attribute)(
                    &mut value,
                    HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                    s.device,
                )
            },
        )?;
        u32::try_from(value).map_err(|_| format!("{RUNTIME_PREFIX} stage hipDeviceGetAttribute: negative multiprocessor count {value}"))
    }

    /// `hipDeviceAttributeWallClockRate` in kHz (the GPU realtime counter's constant rate).
    pub fn wall_clock_khz(&self) -> Result<u32, String> {
        let s = &*self.shared;
        let mut value: c_int = 0;
        s.check(
            "hipDeviceGetAttribute",
            &format!("hipDeviceGetAttribute(WallClockRate, device {})", s.device),
            unsafe {
                (s.gpu.get_attribute)(&mut value, HIP_DEVICE_ATTRIBUTE_WALL_CLOCK_RATE, s.device)
            },
        )?;
        u32::try_from(value).ok().filter(|&v| v > 0).ok_or_else(|| {
            format!("{RUNTIME_PREFIX} stage hipDeviceGetAttribute: bad wall clock rate {value}")
        })
    }

    /// Allocate `bytes` (> 0): `None` is ordinary `hipMalloc` device memory, `Some(kind)` is host VMM memory mapped
    /// for this GPU (same allocation path as `HipDmabufA`, without dma-buf export).
    pub fn allocate(&self, bytes: usize, kind: Option<HipMemoryKind>) -> Result<HipBuffer, String> {
        if bytes == 0 {
            return Err(format!("{RUNTIME_PREFIX} stage args: allocation size is 0"));
        }
        let s = &*self.shared;
        let backing = match kind {
            None => {
                let mut p: *mut c_void = core::ptr::null_mut();
                s.check("hipMalloc", &format!("hipMalloc({bytes} B)"), unsafe {
                    (s.gpu.malloc)(&mut p, bytes)
                })?;
                if p.is_null() {
                    return Err(format!(
                        "{RUNTIME_PREFIX} stage hipMalloc: returned success with null pointer"
                    ));
                }
                Backing::Device(p)
            }
            Some(kind) => Backing::Vmm(Vmm::create(&s.api, bytes, kind, s.device)?),
        };
        Ok(HipBuffer {
            shared: Rc::clone(&self.shared),
            backing,
            len: bytes,
        })
    }

    /// Copy `bytes` (<= the buffer length) to the start of `buffer` (`hipMemcpyHtoD`, synchronous).
    pub fn upload(&self, buffer: &HipBuffer, bytes: &[u8]) -> Result<(), String> {
        self.upload_at(buffer, 0, bytes)
    }

    /// Copy `bytes` to `buffer` at byte `offset` (`offset + len` <= the buffer length; `hipMemcpyHtoD`, synchronous).
    pub fn upload_at(&self, buffer: &HipBuffer, offset: usize, bytes: &[u8]) -> Result<(), String> {
        self.owns(&buffer.shared, "upload buffer")?;
        if offset
            .checked_add(bytes.len())
            .is_none_or(|end| end > buffer.len)
        {
            return Err(format!(
                "{RUNTIME_PREFIX} stage args: upload of {} B at {offset} exceeds buffer of {} B",
                bytes.len(),
                buffer.len
            ));
        }
        let s = &*self.shared;
        s.check(
            "hipMemcpyHtoD",
            &format!("hipMemcpyHtoD({} B)", bytes.len()),
            unsafe {
                (s.api.memcpy_htod)(
                    (buffer.raw() as *mut u8).add(offset) as *mut c_void,
                    bytes.as_ptr() as *const c_void,
                    bytes.len(),
                )
            },
        )
    }

    /// Fill `bytes` (<= the buffer length) from the start of `buffer` (`hipMemcpyDtoH`, synchronous).
    pub fn download(&self, buffer: &HipBuffer, bytes: &mut [u8]) -> Result<(), String> {
        self.download_at(buffer, 0, bytes)
    }

    /// Fill `bytes` from `buffer` at byte `offset` (`offset + len` <= the buffer length; `hipMemcpyDtoH`, synchronous).
    pub fn download_at(
        &self,
        buffer: &HipBuffer,
        offset: usize,
        bytes: &mut [u8],
    ) -> Result<(), String> {
        self.owns(&buffer.shared, "download buffer")?;
        if offset
            .checked_add(bytes.len())
            .is_none_or(|end| end > buffer.len)
        {
            return Err(format!(
                "{RUNTIME_PREFIX} stage args: download of {} B at {offset} exceeds buffer of {} B",
                bytes.len(),
                buffer.len
            ));
        }
        let s = &*self.shared;
        s.check(
            "hipMemcpyDtoH",
            &format!("hipMemcpyDtoH({} B)", bytes.len()),
            unsafe {
                (s.gpu.memcpy_dtoh)(
                    bytes.as_mut_ptr() as *mut c_void,
                    (buffer.raw() as *mut u8).add(offset) as *mut c_void,
                    bytes.len(),
                )
            },
        )
    }

    /// Set every byte of `buffer` (its requested length) to `value` (`hipMemset`).
    pub fn memset(&self, buffer: &HipBuffer, value: u8) -> Result<(), String> {
        self.owns(&buffer.shared, "memset buffer")?;
        let s = &*self.shared;
        s.check(
            "hipMemset",
            &format!("hipMemset({} B, {value:#04x})", buffer.len),
            unsafe { (s.gpu.memset)(buffer.raw(), c_int::from(value), buffer.len) },
        )
    }

    /// `hipDeviceSynchronize`.
    pub fn synchronize(&self) -> Result<(), String> {
        let s = &*self.shared;
        s.check("hipDeviceSynchronize", "hipDeviceSynchronize", unsafe {
            (s.api.device_synchronize)()
        })
    }

    /// `hipModuleLoadData` of an in-memory code object (e.g. an AMDGPU ELF).
    pub fn load_module(&self, bytes: &[u8]) -> Result<HipModule, String> {
        if bytes.is_empty() {
            return Err(format!("{RUNTIME_PREFIX} stage args: code object is empty"));
        }
        let s = &*self.shared;
        let mut module: Handle = core::ptr::null_mut();
        s.check(
            "hipModuleLoadData",
            &format!("hipModuleLoadData({} B)", bytes.len()),
            unsafe { (s.gpu.module_load_data)(&mut module, bytes.as_ptr() as *const c_void) },
        )?;
        if module.is_null() {
            return Err(format!(
                "{RUNTIME_PREFIX} stage hipModuleLoadData: returned success with null module"
            ));
        }
        Ok(HipModule {
            inner: Rc::new(ModuleInner {
                shared: Rc::clone(&self.shared),
                handle: module,
            }),
        })
    }

    /// Launch `function` as a 1-D `grid` x `block` dispatch on the null stream (asynchronous; use `synchronize` or
    /// events). `kernarg` is passed as the kernel-argument blob through `HIP_LAUNCH_PARAM_BUFFER_POINTER` /
    /// `HIP_LAUNCH_PARAM_BUFFER_SIZE`.
    pub fn launch(
        &self,
        function: &HipFunction,
        grid: u32,
        block: u32,
        kernarg: &mut [u8],
    ) -> Result<(), String> {
        self.launch_on(None, function, grid, block, kernarg)
    }

    /// [`HipRuntime::launch`] on `stream` (`None`: the null stream).
    pub fn launch_on(
        &self,
        stream: Option<&HipStream>,
        function: &HipFunction,
        grid: u32,
        block: u32,
        kernarg: &mut [u8],
    ) -> Result<(), String> {
        if let Some(st) = stream {
            self.owns(&st.shared, "launch stream")?;
        }
        self.owns(&function.module.shared, "launch function")?;
        if grid == 0 || block == 0 {
            return Err(format!(
                "{RUNTIME_PREFIX} stage args: launch grid {grid} / block {block} must be non-zero"
            ));
        }
        let s = &*self.shared;
        let mut size: usize = kernarg.len();
        let mut extra: [*mut c_void; 5] = [
            HIP_LAUNCH_PARAM_BUFFER_POINTER as *mut c_void,
            kernarg.as_mut_ptr() as *mut c_void,
            HIP_LAUNCH_PARAM_BUFFER_SIZE as *mut c_void,
            &mut size as *mut usize as *mut c_void,
            HIP_LAUNCH_PARAM_END as *mut c_void,
        ];
        s.check(
            "hipModuleLaunchKernel",
            &format!(
                "hipModuleLaunchKernel(grid {grid}, block {block}, kernarg {} B)",
                kernarg.len()
            ),
            unsafe {
                (s.gpu.module_launch_kernel)(
                    function.handle,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    stream.map_or(core::ptr::null_mut(), |st| st.handle),
                    core::ptr::null_mut(),
                    extra.as_mut_ptr(),
                )
            },
        )
    }

    /// Launch `function` as a 3-D `grid` x `block` dispatch with `shared` bytes of dynamic LDS on the null stream
    /// (asynchronous). `kernarg` as in [`HipRuntime::launch`].
    pub fn launch_grid(
        &self,
        function: &HipFunction,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        kernarg: &mut [u8],
    ) -> Result<(), String> {
        self.owns(&function.module.shared, "launch function")?;
        if grid.contains(&0) || block.contains(&0) {
            return Err(format!("{RUNTIME_PREFIX} stage args: launch grid {grid:?} / block {block:?} must be non-zero"));
        }
        let s = &*self.shared;
        let mut size: usize = kernarg.len();
        let mut extra: [*mut c_void; 5] = [
            HIP_LAUNCH_PARAM_BUFFER_POINTER as *mut c_void,
            kernarg.as_mut_ptr() as *mut c_void,
            HIP_LAUNCH_PARAM_BUFFER_SIZE as *mut c_void,
            &mut size as *mut usize as *mut c_void,
            HIP_LAUNCH_PARAM_END as *mut c_void,
        ];
        s.check(
            "hipModuleLaunchKernel",
            &format!(
                "hipModuleLaunchKernel(grid {grid:?}, block {block:?}, lds {shared}, kernarg {} B)",
                kernarg.len()
            ),
            unsafe {
                (s.gpu.module_launch_kernel)(
                    function.handle,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    shared,
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    extra.as_mut_ptr(),
                )
            },
        )
    }

    /// `hipMemcpyDtoD` of `bytes` from GPU address `src` to GPU address `dst` (synchronous with respect to the host).
    pub fn copy_dtod(&self, dst: u64, src: u64, bytes: usize) -> Result<(), String> {
        let s = &*self.shared;
        s.check(
            "hipMemcpyDtoD",
            &format!("hipMemcpyDtoD({bytes} B)"),
            unsafe {
                (s.gpu.memcpy_dtod)(
                    dst as usize as *mut c_void,
                    src as usize as *mut c_void,
                    bytes,
                )
            },
        )
    }

    /// `hipEventCreate`.
    pub fn event(&self) -> Result<HipEvent, String> {
        let s = &*self.shared;
        let mut event: Handle = core::ptr::null_mut();
        s.check("hipEventCreate", "hipEventCreate", unsafe {
            (s.gpu.event_create)(&mut event)
        })?;
        if event.is_null() {
            return Err(format!(
                "{RUNTIME_PREFIX} stage hipEventCreate: returned success with null event"
            ));
        }
        Ok(HipEvent {
            shared: Rc::clone(&self.shared),
            handle: event,
        })
    }

    /// `hipStreamCreateWithFlags(hipStreamNonBlocking)`: an in-order queue that does not synchronise with the null
    /// stream (`synchronize` still waits for it).
    pub fn stream(&self) -> Result<HipStream, String> {
        let s = &*self.shared;
        let mut stream: Handle = core::ptr::null_mut();
        s.check(
            "hipStreamCreateWithFlags",
            "hipStreamCreateWithFlags(NonBlocking)",
            unsafe { (s.gpu.stream_create)(&mut stream, 1) },
        )?;
        Ok(HipStream {
            shared: Rc::clone(&self.shared),
            handle: stream,
        })
    }

    /// `hipEventRecord` on the null stream.
    pub fn record(&self, event: &HipEvent) -> Result<(), String> {
        self.record_on(event, None)
    }

    /// `hipEventRecord` on `stream` (`None`: the null stream).
    pub fn record_on(&self, event: &HipEvent, stream: Option<&HipStream>) -> Result<(), String> {
        self.owns(&event.shared, "record event")?;
        if let Some(st) = stream {
            self.owns(&st.shared, "record stream")?;
        }
        let s = &*self.shared;
        s.check("hipEventRecord", "hipEventRecord", unsafe {
            (s.gpu.event_record)(
                event.handle,
                stream.map_or(core::ptr::null_mut(), |st| st.handle),
            )
        })
    }

    /// `hipEventSynchronize(end)` then `hipEventElapsedTime(start, end)` in milliseconds.
    pub fn elapsed_ms(&self, start: &HipEvent, end: &HipEvent) -> Result<f32, String> {
        self.owns(&start.shared, "elapsed start event")?;
        self.owns(&end.shared, "elapsed end event")?;
        let s = &*self.shared;
        s.check("hipEventSynchronize", "hipEventSynchronize(end)", unsafe {
            (s.gpu.event_synchronize)(end.handle)
        })?;
        let mut ms: f32 = 0.0;
        s.check("hipEventElapsedTime", "hipEventElapsedTime", unsafe {
            (s.gpu.event_elapsed_time)(&mut ms, start.handle, end.handle)
        })?;
        Ok(ms)
    }

    /// `hipEventSynchronize(event)`: block the host until the event's stream position is complete.
    pub fn sync_event(&self, event: &HipEvent) -> Result<(), String> {
        self.owns(&event.shared, "sync event")?;
        let s = &*self.shared;
        s.check("hipEventSynchronize", "hipEventSynchronize", unsafe {
            (s.gpu.event_synchronize)(event.handle)
        })
    }

    /// dma-buf fd of a host-VMM `buffer` (`hipMemGetHandleForAddressRange(.., DmaBufFd)` over its whole
    /// granularity-rounded allocation), for `railgun::npu::Device::import_dmabuf`. `hipMalloc` buffers are refused.
    /// The buffer must outlive every importer of the fd (see the module docs for drop order).
    #[cfg(unix)]
    pub fn export_dmabuf(&self, buffer: &HipBuffer) -> Result<DmabufFd, String> {
        self.owns(&buffer.shared, "export buffer")?;
        let Backing::Vmm(vmm) = &buffer.backing else {
            return Err(format!(
                "{RUNTIME_PREFIX} stage args: only host-VMM buffers are dma-buf exportable"
            ));
        };
        let s = &*self.shared;
        let mut fd: c_int = -1;
        s.check(
            "hipMemGetHandleForAddressRange",
            &format!("hipMemGetHandleForAddressRange(dmabuf fd, {} B)", vmm.alloc),
            unsafe {
                (s.api.get_handle_for_address_range)(
                    &mut fd as *mut c_int as *mut c_void,
                    vmm.va,
                    vmm.alloc,
                    HIP_MEM_RANGE_HANDLE_DMABUF_FD,
                    0,
                )
            },
        )?;
        if fd < 0 {
            return Err(format!("{RUNTIME_PREFIX} stage hipMemGetHandleForAddressRange: returned success with invalid fd {fd}"));
        }
        Ok(DmabufFd {
            fd,
            alloc: vmm.alloc,
        })
    }

    /// `hipHostRegister(ptr, len, 0)` + `hipHostGetDevicePointer`: make caller-owned host pages (e.g. anonymous memory
    /// that another device has pinned) GPU-accessible. The pages must stay mapped until the returned registration is
    /// dropped (`hipHostUnregister`).
    pub fn register_host(&self, ptr: *mut u8, len: usize) -> Result<HostRegistration, String> {
        self.register_host_flags(ptr, len, 0)
    }

    /// [`HipRuntime::register_host`] with explicit `hipHostRegister` flags (e.g. `hipExtHostRegisterCoarseGrained` 0x8).
    pub fn register_host_flags(
        &self,
        ptr: *mut u8,
        len: usize,
        flags: u32,
    ) -> Result<HostRegistration, String> {
        if ptr.is_null() || len == 0 {
            return Err(format!(
                "{RUNTIME_PREFIX} stage args: register_host needs a non-null, non-empty range"
            ));
        }
        let s = &*self.shared;
        s.check(
            "hipHostRegister",
            &format!("hipHostRegister({len} B, flags {flags:#x})"),
            unsafe { (s.gpu.host_register)(ptr as *mut c_void, len, flags) },
        )?;
        // From here `reg` unregisters on any early return.
        let mut reg = HostRegistration {
            shared: Rc::clone(&self.shared),
            host: ptr,
            device: 0,
            len,
        };
        let mut dptr: *mut c_void = core::ptr::null_mut();
        s.check(
            "hipHostGetDevicePointer",
            "hipHostGetDevicePointer",
            unsafe { (s.gpu.host_get_device_pointer)(&mut dptr, ptr as *mut c_void, 0) },
        )?;
        if dptr.is_null() {
            return Err(format!("{RUNTIME_PREFIX} stage hipHostGetDevicePointer: returned success with null pointer"));
        }
        reg.device = dptr as usize as u64;
        Ok(reg)
    }

    /// System-release events for foreign-device handoff (never timing-only events).
    pub fn system_event(&self) -> Result<HipEvent, String> {
        let p = resolve(
            self.shared.api.lib,
            &self.shared.api.source,
            ["hipEventCreateWithFlags"],
            RUNTIME_PREFIX,
        )?;
        let create: unsafe extern "C" fn(*mut Handle, u32) -> HipStatus =
            unsafe { core::mem::transmute(p[0]) };
        let mut handle = core::ptr::null_mut();
        self.shared
            .check("hipEventCreateWithFlags", "system release event", unsafe {
                create(&mut handle, 0x8000_0000 | 2)
            })?;
        Ok(HipEvent {
            shared: Rc::clone(&self.shared),
            handle,
        })
    }

    /// Raw-stream counterpart of `record_on`, for streams owned by the hipfire GPU runtime.
    /// The stream must be live and belong to this GPU.
    pub unsafe fn record_raw(&self, event: &HipEvent, stream: *mut c_void) -> Result<(), String> {
        self.owns(&event.shared, "record event")?;
        self.shared
            .check("hipEventRecord", "foreign stream event", unsafe {
                (self.shared.gpu.event_record)(event.handle, stream)
            })
    }

    /// Order a foreign producer stream after the last consumer of its arena.
    pub unsafe fn wait_raw(&self, event: &HipEvent, stream: *mut c_void) -> Result<(), String> {
        self.owns(&event.shared, "wait event")?;
        let p = resolve(
            self.shared.api.lib,
            &self.shared.api.source,
            ["hipStreamWaitEvent"],
            RUNTIME_PREFIX,
        )?;
        let wait: unsafe extern "C" fn(Handle, Handle, u32) -> HipStatus =
            unsafe { core::mem::transmute(p[0]) };
        self.shared
            .check("hipStreamWaitEvent", "foreign stream retire wait", unsafe {
                wait(stream, event.handle, 0)
            })
    }

    /// Resolve only the relay's thread-safe entry points. No Rc-owned HIP object crosses threads.
    pub(crate) fn relay_api(&self) -> Result<RelayApi, String> {
        let p = resolve(
            self.shared.api.lib,
            &self.shared.api.source,
            ["hipEventQuery", "hipStreamWaitValue32"],
            RUNTIME_PREFIX,
        )?;
        Ok(RelayApi {
            device: self.shared.device,
            set_device: self.shared.api.set_device,
            query: unsafe { core::mem::transmute(p[0]) },
            wait: unsafe { core::mem::transmute(p[1]) },
        })
    }

    /// Bounded event completion; unlike hipEventSynchronize this cannot wait forever.
    pub fn wait_event_timeout(
        &self,
        event: &HipEvent,
        timeout: std::time::Duration,
    ) -> Result<(), String> {
        self.owns(&event.shared, "wait event")?;
        let api = self.relay_api()?;
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if api.query(event.raw())? {
                return Ok(());
            }
            if std::time::Instant::now() >= deadline {
                return Err("HIP event timed out".into());
            }
            std::thread::sleep(std::time::Duration::from_micros(20));
        }
    }

    /// A resource must come from this runtime, not another `load_for_pci` call.
    fn owns(&self, other: &Rc<Shared>, what: &str) -> Result<(), String> {
        if Rc::ptr_eq(&self.shared, other) {
            Ok(())
        } else {
            Err(format!(
                "{RUNTIME_PREFIX} stage args: {what} belongs to a different HipRuntime"
            ))
        }
    }
}

enum Backing {
    /// `hipMalloc` device pointer.
    Device(*mut c_void),
    /// Host VMM allocation.
    Vmm(Vmm),
}

/// Device or host-VMM memory; freed on drop (`hipFree` / VMM unmap+free+release).
pub struct HipBuffer {
    shared: Rc<Shared>,
    backing: Backing,
    len: usize,
}

impl HipBuffer {
    fn raw(&self) -> *mut c_void {
        match &self.backing {
            Backing::Device(p) => *p,
            Backing::Vmm(v) => v.va,
        }
    }

    /// GPU virtual address (the value a kernel receives as a pointer argument).
    pub fn ptr(&self) -> u64 {
        self.raw() as usize as u64
    }

    /// Requested length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A host range registered with [`HipRuntime::register_host`]; `hipHostUnregister` on drop.
pub struct HostRegistration {
    shared: Rc<Shared>,
    host: *mut u8,
    device: u64,
    len: usize,
}

impl HostRegistration {
    /// GPU virtual address of the registered range (what a kernel receives as a pointer).
    pub fn ptr(&self) -> u64 {
        self.device
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for HostRegistration {
    fn drop(&mut self) {
        unsafe { (self.shared.gpu.host_unregister)(self.host as *mut c_void) };
    }
}

/// Owned dma-buf fd exported by [`HipRuntime::export_dmabuf`]; closed on drop.
#[cfg(unix)]
pub struct DmabufFd {
    fd: RawFd,
    /// Granularity-rounded size of the exported allocation (the dma-buf size).
    pub alloc: usize,
}

#[cfg(unix)]
impl DmabufFd {
    pub fn fd(&self) -> RawFd {
        self.fd
    }
}

#[cfg(unix)]
impl Drop for DmabufFd {
    fn drop(&mut self) {
        unsafe { sys::close(self.fd) };
    }
}

impl Drop for HipBuffer {
    fn drop(&mut self) {
        if let Backing::Device(p) = self.backing {
            unsafe { (self.shared.gpu.free)(p) };
        }
        // Backing::Vmm releases itself.
    }
}

struct ModuleInner {
    shared: Rc<Shared>,
    handle: Handle,
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        unsafe { (self.shared.gpu.module_unload)(self.handle) };
    }
}

/// Loaded code object; unloaded when the module and every `HipFunction` taken from it are dropped.
pub struct HipModule {
    inner: Rc<ModuleInner>,
}

impl HipModule {
    /// `hipModuleGetFunction` by kernel symbol name.
    pub fn function(&self, name: &str) -> Result<HipFunction, String> {
        let s = &*self.inner.shared;
        let cname = CString::new(name).map_err(|_| {
            format!("{RUNTIME_PREFIX} stage args: kernel name {name:?} contains NUL")
        })?;
        let mut function: Handle = core::ptr::null_mut();
        s.check(
            "hipModuleGetFunction",
            &format!("hipModuleGetFunction({name})"),
            unsafe {
                (s.gpu.module_get_function)(&mut function, self.inner.handle, cname.as_ptr())
            },
        )?;
        if function.is_null() {
            return Err(format!("{RUNTIME_PREFIX} stage hipModuleGetFunction: returned success with null function for {name}"));
        }
        Ok(HipFunction {
            module: Rc::clone(&self.inner),
            handle: function,
        })
    }
}

/// Kernel entry point; keeps its module loaded.
#[derive(Clone)]
pub struct HipFunction {
    module: Rc<ModuleInner>,
    handle: Handle,
}

/// `hipEvent_t`; destroyed on drop.
pub struct HipEvent {
    shared: Rc<Shared>,
    handle: Handle,
}

impl HipEvent {
    pub(crate) fn raw(&self) -> usize {
        self.handle as usize
    }
}

impl Drop for HipEvent {
    fn drop(&mut self) {
        unsafe { (self.shared.gpu.event_destroy)(self.handle) };
    }
}

/// A HIP stream (`HipRuntime::stream`); destroyed on drop.
pub struct HipStream {
    shared: Rc<Shared>,
    handle: Handle,
}

impl HipStream {
    pub fn raw(&self) -> *mut c_void {
        self.handle
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        unsafe { (self.shared.gpu.stream_destroy)(self.handle) };
    }
}

/// Function pointers are immutable and the library stays loaded for the process.
/// The relay selects the original GPU on its own thread before querying events.
#[derive(Clone, Copy)]
pub(crate) struct RelayApi {
    device: i32,
    set_device: unsafe extern "C" fn(i32) -> HipStatus,
    query: unsafe extern "C" fn(Handle) -> HipStatus,
    wait: unsafe extern "C" fn(Handle, Handle, u32, u32, u32) -> HipStatus,
}

impl RelayApi {
    pub fn select(&self) -> Result<(), String> {
        let code = unsafe { (self.set_device)(self.device) };
        if code == 0 {
            Ok(())
        } else {
            Err(format!("relay hipSetDevice: status {code}"))
        }
    }
    pub fn query(&self, event: usize) -> Result<bool, String> {
        match unsafe { (self.query)(event as Handle) } {
            0 => Ok(true),
            600 => Ok(false),
            code => Err(format!("relay hipEventQuery: status {code}")),
        }
    }
    pub unsafe fn wait_value(&self, stream: Handle, ptr: u64, value: u32) -> Result<(), String> {
        let code = unsafe { (self.wait)(stream, ptr as Handle, value, 0, u32::MAX) };
        if code == 0 {
            Ok(())
        } else {
            Err(format!("hipStreamWaitValue32: status {code}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `dladdr_fname` must resolve a symbol that is definitely mapped: `dlopen`
    /// itself lives in the process's dynamic linker/libc on every Linux host.
    /// Test the guard's diagnostic on the real dynamic loader. The admission
    /// decision itself compares live symbol addresses, not this path.
    #[test]
    fn dladdr_fname_resolves_the_object_of_a_live_symbol() {
        let dlopen_ptr = sys::dlopen as unsafe extern "C" fn(*const c_char, c_int) -> *mut c_void;
        let path =
            dladdr_fname(dlopen_ptr as *const c_void).expect("dladdr must resolve dlopen itself");
        assert!(!path.is_empty());
        assert!(
            std::path::Path::new(&path).exists(),
            "mapped object should exist on disk: {path}"
        );
    }

    #[test]
    fn dladdr_fname_handles_an_unknown_address() {
        assert_eq!(dladdr_fname(core::ptr::null()), None);
    }

    fn name(a: Attempt<'_>) -> (&str, bool) {
        (a.name, a.already_loaded)
    }

    /// No GPU, no ROCm: the resolution order is the already-loaded probes (every soname, `RTLD_NOLOAD`), then the
    /// shared candidate list exactly as `hipfire_config::rocm::library_candidates` returned it.
    #[test]
    fn load_plan_probes_loaded_sonames_then_candidates_in_order() {
        let sonames = ["libamdhip64.so", "libamdhip64.so.7"];
        let candidates = vec![
            "/opt/rocm/lib/libamdhip64.so".to_owned(),
            "/opt/rocm/lib/libamdhip64.so.7".to_owned(),
        ];
        let got: Vec<_> = load_plan(&sonames, &candidates).map(name).collect();
        assert_eq!(
            got,
            [
                ("libamdhip64.so", true),
                ("libamdhip64.so.7", true),
                ("/opt/rocm/lib/libamdhip64.so", false),
                ("/opt/rocm/lib/libamdhip64.so.7", false),
            ]
        );
    }

    /// A host with no resolvable ROCm root yields no loading attempt: the bare `libamdhip64.so` search is gone, so
    /// ldconfig cannot supply another install.
    #[test]
    fn load_plan_never_loads_a_bare_soname() {
        let sonames = hipfire_config::rocm::HIP_RUNTIME_LIBRARIES;
        let mut plan = load_plan(sonames, &[]);
        assert!(plan.all(|a| a.already_loaded));
        assert_eq!(load_plan(sonames, &[]).count(), sonames.len());
    }
}
