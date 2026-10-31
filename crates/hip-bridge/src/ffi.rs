// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! FFI bindings to libamdhip64.so via dlopen.
//! No link-time dependency — runtime loads the shared library.

use crate::error::{HipError, HipResult};
use crate::plan_tape::{PlanEvent, PlanGate, PlanTape};
use crate::{DeviceBuffer, MemcpyKind};
#[cfg(unix)]
use core::ffi::CStr;
use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_uint, c_void, CString, OsStr, OsString};
use std::ptr;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

#[cfg(unix)]
extern "C" {
    fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
}

/// `struct dl_addr` of `dladdr(3)`; only `dli_fname` is read.
#[repr(C)]
#[cfg(unix)]
struct DlInfo {
    dli_fname: *const c_char,
    dli_fbase: *mut c_void,
    dli_sname: *const c_char,
    dli_saddr: *mut c_void,
}

/// Loader-reported path backing `ptr`, for diagnostics only, not object identity.
#[cfg(unix)]
fn dladdr_fname(ptr: *const c_void) -> Option<String> {
    let mut info = DlInfo {
        dli_fname: ptr::null(),
        dli_fbase: ptr::null_mut(),
        dli_sname: ptr::null(),
        dli_saddr: ptr::null_mut(),
    };
    // SAFETY: `info` is a valid out-pointer for the lifetime of the call.
    if unsafe { dladdr(ptr, &mut info) } == 0 || info.dli_fname.is_null() {
        return None;
    }
    // SAFETY: dladdr leaves a NUL-terminated path there on success.
    Some(
        unsafe { CStr::from_ptr(info.dli_fname) }
            .to_string_lossy()
            .into_owned(),
    )
}

/// Kernel32 entry points for [`dladdr_fname`]. Declared directly with an
/// explicit link so hip-bridge gains no dependency edge (and no Cargo.lock
/// change).
#[cfg(windows)]
mod win32 {
    use std::ffi::c_void;

    pub(super) const GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT: u32 = 0x0000_0002;
    pub(super) const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x0000_0004;

    #[link(name = "kernel32")]
    extern "system" {
        pub(super) fn GetModuleHandleExW(
            flags: u32,
            module_name_or_address: *const u16,
            module: *mut *mut c_void,
        ) -> i32;
        pub(super) fn GetModuleFileNameW(module: *mut c_void, filename: *mut u16, size: u32)
            -> u32;
    }
}

/// Loader-reported path backing `ptr` (the DLL that contains that address), for
/// diagnostics only, not object identity.
#[cfg(windows)]
fn dladdr_fname(ptr: *const c_void) -> Option<String> {
    use win32::*;
    let mut module: *mut c_void = ptr::null_mut();
    // SAFETY: FROM_ADDRESS makes the second argument an address, not a string;
    // UNCHANGED_REFCOUNT means the returned handle needs no FreeLibrary and
    // cannot outlive the caller's own hold on the mapping. `module` is a valid
    // out-pointer.
    let ok = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            ptr as *const u16,
            &mut module,
        )
    };
    if ok == 0 || module.is_null() {
        return None;
    }
    // Grow until the path fits: GetModuleFileNameW truncates silently and
    // returns the buffer size when it does. 32768 UTF-16 units is the NT limit.
    let mut buf = vec![0u16; 512];
    loop {
        // SAFETY: `buf` is valid for `buf.len()` UTF-16 units and the length is
        // passed as the size.
        let len =
            unsafe { GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32) } as usize;
        if len == 0 {
            return None;
        }
        if len < buf.len() {
            return Some(String::from_utf16_lossy(&buf[..len]));
        }
        if buf.len() >= 32768 {
            return None;
        }
        let grown = buf.len() * 2;
        buf.resize(grown, 0);
    }
}

#[cfg(not(any(unix, windows)))]
fn dladdr_fname(_ptr: *const c_void) -> Option<String> {
    None
}

/// Map one HIP runtime candidate. Non-Windows hosts keep plain `Library::new`.
///
/// # Safety
/// Same contract as `libloading::Library::new`: loading runs the library's
/// initialisation routines.
#[cfg(not(windows))]
unsafe fn open_library(candidate: &str) -> Result<Library, libloading::Error> {
    Library::new(candidate)
}

/// Map one HIP runtime candidate on Windows.
///
/// Plain `Library::new` is `LoadLibraryExW(path, NULL, 0)`: the named DLL loads
/// from its absolute path, but its own dependencies (amd_comgr*.dll, ...) go
/// through the standard search order, where System32 precedes PATH. The AMD
/// display driver installs an older amdhip64/comgr into System32, so a HIP SDK
/// (or shipped runtime) DLL would silently pair with the driver's comgr.
/// `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` makes the DLL's own directory the first
/// place its dependencies resolve; `LOAD_LIBRARY_SEARCH_DEFAULT_DIRS` keeps the
/// application directory, System32 and AddDllDirectory entries as fallbacks.
///
/// Both flags reject relative names, and DEFAULT_DIRS excludes PATH, so a bare
/// name (the legacy PATH fallback) keeps `Library::new`. If the flagged load of
/// an existing absolute path fails (a lone runtime DLL, e.g. the installer's
/// `~\.hipfire\runtime` copy, whose dependencies are reachable only through
/// PATH), the pre-flag `Library::new` load is retried so a setup that loaded
/// before still loads.
///
/// # Safety
/// Same contract as `libloading::Library::new`.
#[cfg(windows)]
unsafe fn open_library(candidate: &str) -> Result<Library, libloading::Error> {
    use libloading::os::windows::{
        Library as WinLibrary, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
    };
    if !std::path::Path::new(candidate).is_absolute() {
        return Library::new(candidate);
    }
    // The SEARCH_* flags want a normalised backslash path; a root taken from an
    // env var may carry forward slashes. Verbatim (`\\?\`) paths are literal.
    let normalized;
    let path = if candidate.starts_with(r"\\?\") || !candidate.contains('/') {
        candidate
    } else {
        normalized = candidate.replace('/', "\\");
        normalized.as_str()
    };
    match WinLibrary::load_with_flags(
        path,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
    ) {
        Ok(lib) => Ok(Library::from(lib)),
        Err(_) if std::path::Path::new(path).is_file() => Library::new(path),
        Err(error) => Err(error),
    }
}

#[cfg(feature = "kernel-timing")]
pub mod kernel_timing;

/// Per-thread accumulators for time spent inside HIP FFI calls. Used by
/// Phase 3a host-vs-GPU diagnostics to attribute the forward pass wall
/// clock to specific HIP runtime calls.
pub mod launch_counters {
    use std::cell::Cell;

    macro_rules! counter {
        ($mod_name:ident) => {
            pub mod $mod_name {
                use std::cell::Cell;
                thread_local! {
                    pub(super) static TIME_NS: Cell<u64> = const { Cell::new(0) };
                    pub(super) static COUNT: Cell<u64> = const { Cell::new(0) };
                    pub(super) static BYTES: Cell<u64> = const { Cell::new(0) };
                }
                #[inline]
                pub fn record(ns: u64) {
                    TIME_NS.with(|c| c.set(c.get() + ns));
                    COUNT.with(|c| c.set(c.get() + 1));
                }
                #[inline]
                pub fn record_bytes(ns: u64, bytes: u64) {
                    TIME_NS.with(|c| c.set(c.get() + ns));
                    COUNT.with(|c| c.set(c.get() + 1));
                    BYTES.with(|c| c.set(c.get() + bytes));
                }
                pub fn time_ns() -> u64 {
                    TIME_NS.with(|c| c.get())
                }
                pub fn count() -> u64 {
                    COUNT.with(|c| c.get())
                }
                pub fn bytes() -> u64 {
                    BYTES.with(|c| c.get())
                }
                pub fn reset() {
                    TIME_NS.with(|c| c.set(0));
                    COUNT.with(|c| c.set(0));
                    BYTES.with(|c| c.set(0));
                }
            }
        };
    }

    // Existing counter — kept for back-compat with profile_host_vs_gpu.
    thread_local! {
        static TIME_NS: Cell<u64> = const { Cell::new(0) };
        static COUNT: Cell<u64> = const { Cell::new(0) };
    }

    #[inline]
    pub(super) fn record(ns: u64) {
        TIME_NS.with(|c| c.set(c.get() + ns));
        COUNT.with(|c| c.set(c.get() + 1));
        launch_kernel::record(ns);
    }

    pub fn reset() {
        TIME_NS.with(|c| c.set(0));
        COUNT.with(|c| c.set(0));
        launch_kernel::reset();
        memcpy_dtod::reset();
        memcpy_htod::reset();
        memcpy_dtoh::reset();
        memset::reset();
        ensure_kernel_lookup::reset();
        stream_sync::reset();
        event_sync::reset();
        device_sync::reset();
        graph_launch::reset();
    }

    pub fn time_ns() -> u64 {
        TIME_NS.with(|c| c.get())
    }
    pub fn count() -> u64 {
        COUNT.with(|c| c.get())
    }

    // Per-API counters
    counter!(launch_kernel);
    counter!(memcpy_dtod);
    counter!(memcpy_htod);
    counter!(memcpy_dtoh);
    counter!(memset);
    counter!(ensure_kernel_lookup);
    counter!(stream_sync);
    counter!(event_sync);
    counter!(device_sync);
    counter!(graph_launch);
}

/// Per-thread tally of the HIP memory operations and kernel launches issued
/// on this thread.
///
/// A retained Redline tape replays recorded kernel dispatches only, so a
/// device copy, readback or memset issued inside its capture window is state
/// the tape cannot reproduce, and so is a kernel launched past the recorder.
/// The replay controller differences two snapshots
/// around the window. Unlike [`launch_counters`] this tally is never reset, so
/// a profiler resetting its counters mid-window cannot hide an effect, and
/// every synchronous and asynchronous variant counts.
pub mod memory_effects {
    use std::cell::Cell;

    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct MemoryEffects {
        /// Host→device uploads.
        pub htod: u64,
        /// Device→device copies, including peer copies.
        pub dtod: u64,
        /// Device→host readbacks.
        pub dtoh: u64,
        /// Memsets.
        pub memset: u64,
        /// Kernel launches (`hipModuleLaunchKernel`, both argument paths). A
        /// launch the replay recorder did not see is missing from the tape.
        pub launches: u64,
    }

    impl MemoryEffects {
        /// The operations issued since `earlier`, a snapshot taken on this thread.
        pub fn since(self, earlier: Self) -> Self {
            Self {
                htod: self.htod - earlier.htod,
                dtod: self.dtod - earlier.dtod,
                dtoh: self.dtoh - earlier.dtoh,
                memset: self.memset - earlier.memset,
                launches: self.launches - earlier.launches,
            }
        }

        /// Whether a capture window with these effects is fully described by
        /// its recorded dispatches. Host→device uploads are excluded: a
        /// retained-body boundary re-stages its inputs on every replay.
        pub fn replayable(self) -> bool {
            self.dtod == 0 && self.dtoh == 0 && self.memset == 0
        }
    }

    thread_local! {
        static TALLY: Cell<MemoryEffects> = const {
            Cell::new(MemoryEffects { htod: 0, dtod: 0, dtoh: 0, memset: 0, launches: 0 })
        };
    }

    /// This thread's running tally.
    pub fn snapshot() -> MemoryEffects {
        TALLY.with(Cell::get)
    }

    fn bump(update: impl FnOnce(&mut MemoryEffects)) {
        TALLY.with(|tally| {
            let mut value = tally.get();
            update(&mut value);
            tally.set(value);
        });
    }

    pub(crate) fn htod() {
        bump(|value| value.htod += 1);
    }
    pub(crate) fn dtod() {
        bump(|value| value.dtod += 1);
    }
    pub(crate) fn dtoh() {
        bump(|value| value.dtoh += 1);
    }
    pub(crate) fn memset() {
        bump(|value| value.memset += 1);
    }
    pub(crate) fn launch() {
        bump(|value| value.launches += 1);
    }
}

// Opaque HIP handles (pointers to internal structs)
type HipStream = *mut c_void;
type HipModule = *mut c_void;
type HipFunction = *mut c_void;
type HipEvent = *mut c_void;
type HipGraph = *mut c_void;
type HipGraphExec = *mut c_void;
pub type HipMemGenericAllocationHandle = *mut c_void;

// ── A19 HIP fault injection (oracle only) ──────────────────────────────
// HIPFIRE_FAULT_HIP=<class>[:<count>][,...] makes the FIRST `count` calls
// of the chosen class (`upload` = H2D memcpy, `launch` = kernel launch,
// `sync` = stream synchronize) fail with an injected error. This exercises
// the engine's fail-closed path for device faults — typed rejection,
// poisoned resources not reused, no same-forward fallback execution
// (spec §5.4 S4, oracle cell A19). Inert unless the variable is set by the
// time the first fault-class call occurs; the steady-state cost is one
// atomic load per API call.
const HIP_FAULT_UNSET: usize = usize::MAX;
static HIP_FAULT_UPLOAD: AtomicUsize = AtomicUsize::new(HIP_FAULT_UNSET);
static HIP_FAULT_LAUNCH: AtomicUsize = AtomicUsize::new(HIP_FAULT_UNSET);
static HIP_FAULT_SYNC: AtomicUsize = AtomicUsize::new(HIP_FAULT_UNSET);

/// The process environment is read EXACTLY ONCE (POSIX forbids mutating it
/// under concurrent readers, and the decode-hot path must not pay a getenv
/// per memcpy/launch/sync). Tests arm the seam through
/// [`arm_hip_fault`] instead of `set_var`.
///
/// The spec is read only from an already-installed process config: a HIP call
/// that runs before `install_process_config` must not install the local
/// fallback as a side effect (the daemon's own install would then fail and
/// its resolved config be dropped). Such a call injects nothing and the next
/// call re-checks.
fn fault_spec() -> Option<Option<String>> {
    hipfire_config::active_process_config().map(|config| config.legacy_value("HIPFIRE_FAULT_HIP"))
}

/// Developer-only module-load audit (`HIPFIRE_MODULE_LOAD_AUDIT=1`, default
/// off): one stderr line per `hipModuleLoad`/`hipModuleLoadData`/
/// `hipModuleGetFunction`, keyed by the opaque module handle, so an offline
/// route-closure audit can attribute every resolved symbol to its loaded
/// image. Read per call from the installed process config (module loads are
/// cold-path); nothing is logged before the config is installed.
fn module_load_audit_enabled() -> bool {
    hipfire_config::active_process_config()
        .and_then(|config| config.legacy_value("HIPFIRE_MODULE_LOAD_AUDIT"))
        .as_deref()
        == Some("1")
}

/// FNV-1a 64 over an embedded image; identifies the image in the audit line.
fn module_image_fnv64(image: &[u8]) -> u64 {
    image.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Arm `count` injected failures for one fault class (`upload`, `launch`,
/// `sync`), or disarm with `count == 0`. This is the test-facing arming API:
/// it writes the atomics directly, so no process-global environment state is
/// mutated from a running engine.
pub fn arm_hip_fault(class: &str, count: usize) {
    let cell = match class {
        "upload" => &HIP_FAULT_UPLOAD,
        "launch" => &HIP_FAULT_LAUNCH,
        "sync" => &HIP_FAULT_SYNC,
        other => {
            eprintln!("[hip-bridge] arm_hip_fault: unknown fault class '{other}'");
            return;
        }
    };
    cell.store(count, Ordering::Release);
}

fn hip_fault_consume(class_idx: usize, class: &'static str) -> bool {
    let cells = [&HIP_FAULT_UPLOAD, &HIP_FAULT_LAUNCH, &HIP_FAULT_SYNC];
    let cell = cells[class_idx];
    let mut cur = cell.load(Ordering::Acquire);
    if cur == HIP_FAULT_UNSET {
        // First use: latch the class from the (single) environment read.
        // Absent or unmatched means inert forever — the hot path never
        // consults the environment again.
        let Some(spec) = fault_spec() else {
            return false;
        };
        let n = spec
            .as_deref()
            .map(|spec| {
                spec.split(',')
                    .find_map(|part| {
                        let mut it = part.splitn(2, ':');
                        let name = it.next()?.trim();
                        if name != class {
                            return None;
                        }
                        Some(it.next().and_then(|v| v.parse().ok()).unwrap_or(1))
                    })
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        cell.store(n, Ordering::Release);
        cur = n;
    }
    while cur != 0 {
        match cell.compare_exchange(cur, cur - 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                eprintln!("[hip-bridge] A19: injected {class} fault (HIPFIRE_FAULT_HIP)");
                return true;
            }
            Err(actual) => cur = actual,
        }
    }
    false
}

const HIP_FAULT_UNKNOWN: u32 = 999; // hipErrorUnknown

fn hip_fault_err(class: &'static str) -> HipError {
    HipError::new(
        HIP_FAULT_UNKNOWN,
        &format!("HIPFIRE_FAULT_HIP: injected {class} failure (A19)"),
    )
}

const HIP_SUCCESS: u32 = 0;

/// `hipEventQuery` return when previously-recorded work is still outstanding.
/// Per `hip_runtime_api.h` (`hipErrorNotReady = 600`); a non-blocking poll
/// treats exactly this code as "not yet", every other nonzero code as failure.
pub const HIP_ERROR_NOT_READY: u32 = 600;
pub const HIP_MEM_LOCATION_TYPE_DEVICE: u32 = 1;
pub const HIP_MEM_LOCATION_TYPE_HOST: u32 = 2;
pub const HIP_MEM_ALLOCATION_TYPE_PINNED: u32 = 1;
pub const HIP_MEM_ACCESS_FLAGS_PROT_READ_WRITE: u32 = 3;
/// `hipMemAccessFlagsProtRead`: read-only mapping access.
pub const HIP_MEM_ACCESS_FLAGS_PROT_READ: u32 = 1;
/// `hipMemHandleTypePosixFileDescriptor`: shareable-handle type of a POSIX FD.
pub const HIP_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR: u32 = 1;
pub const HIP_MEM_ALLOCATION_GRANULARITY_MINIMUM: u32 = 0;
pub const HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED: u32 = 1;
/// Dependency event: omit profiling state and retain the default system fence.
pub const HIP_EVENT_DISABLE_TIMING: u32 = 0x2;

/// `hipHostMalloc` flag: map the allocation into device address space so device
/// code can dereference it (the zero-copy / mapped-pinned path). Without it the
/// pages are pinned but only reachable by the copy engines, which is *not*
/// enough for offloaded weights — the GEMV/GEMM kernels read them directly.
pub const HIP_HOST_MALLOC_MAPPED: u32 = 0x2;

/// `hipHostMalloc` flag: fine-grained, coherent host memory. Overrides
/// `HIP_HOST_COHERENT` for this allocation, so device system-scope stores are
/// visible to the host without waiting for a kernel-boundary release.
pub const HIP_HOST_MALLOC_COHERENT: u32 = 0x4000_0000;

/// Request an explicit system-scope release when recording an event.
pub const HIP_EVENT_RELEASE_TO_SYSTEM: u32 = 0x8000_0000;

/// `hipDeviceAttributeCanUseStreamWaitValue` in the `hipDeviceAttribute_t` enum
/// of the ROCm header this bridge targets (`hip_runtime_api.h`:
/// `hipDeviceAttributeAmdSpecificBegin = 10000`, thirteen enumerators later).
pub const HIP_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE: i32 = 10013;

/// `hipStreamWaitValueGte`: wait until `(*ptr & mask) >= (value & mask)`.
pub const HIP_STREAM_WAIT_VALUE_GTE: u32 = 0x0;

/// `hipStreamWriteValueDefault`.
pub const HIP_STREAM_WRITE_VALUE_DEFAULT: u32 = 0x0;

/// `hipMallocSignalMemory` payload of a [`SignalWord`] allocation.
const SIGNAL_WORD_BYTES: usize = 8;

const STREAM_WAIT_CAP_SLOTS: usize = 16;
const STREAM_WAIT_CAP_UNKNOWN: u8 = 0;
const STREAM_WAIT_CAP_NO: u8 = 1;
const STREAM_WAIT_CAP_YES: u8 = 2;

/// `hipPointerAttribute_t` per ROCm 6.4.3 layout. ROCm 5.x not supported.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HipPointerAttribute {
    pub mem_type: u32,
    pub device: c_int,
    pub device_pointer: *mut c_void,
    pub host_pointer: *mut c_void,
    pub is_managed: c_int,
    pub allocation_flags: c_uint,
}

impl Default for HipPointerAttribute {
    fn default() -> Self {
        Self {
            mem_type: 0,
            device: -1,
            device_pointer: ptr::null_mut(),
            host_pointer: ptr::null_mut(),
            is_managed: 0,
            allocation_flags: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HipMemLocation {
    pub type_: u32,
    pub id: c_int,
}

impl HipMemLocation {
    pub fn device(id: i32) -> Self {
        Self {
            type_: HIP_MEM_LOCATION_TYPE_DEVICE,
            id,
        }
    }
    pub fn host() -> Self {
        Self {
            type_: HIP_MEM_LOCATION_TYPE_HOST,
            id: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HipMemAllocationFlags {
    pub compression_type: u8,
    pub gpu_direct_rdma_capable: u8,
    pub usage: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HipMemAllocationProp {
    pub type_: u32,
    pub requested_handle_types: u32,
    pub location: HipMemLocation,
    pub win32_handle_meta_data: *mut c_void,
    pub alloc_flags: HipMemAllocationFlags,
}

impl HipMemAllocationProp {
    pub fn device_pinned(device: i32) -> Self {
        Self {
            type_: HIP_MEM_ALLOCATION_TYPE_PINNED,
            requested_handle_types: 0,
            location: HipMemLocation::device(device),
            win32_handle_meta_data: ptr::null_mut(),
            alloc_flags: HipMemAllocationFlags {
                compression_type: 0,
                gpu_direct_rdma_capable: 0,
                usage: 0,
            },
        }
    }

    /// Device pinned allocation exportable as a POSIX file descriptor
    /// (`requestedHandleTypes = hipMemHandleTypePosixFileDescriptor`).
    pub fn device_pinned_posix_fd(device: i32) -> Self {
        let mut prop = Self::device_pinned(device);
        prop.requested_handle_types = HIP_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        prop
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HipMemAccessDesc {
    pub location: HipMemLocation,
    pub flags: u32,
}

impl HipMemAccessDesc {
    pub fn read_write_device(device: i32) -> Self {
        Self {
            location: HipMemLocation::device(device),
            flags: HIP_MEM_ACCESS_FLAGS_PROT_READ_WRITE,
        }
    }

    /// Read-only access for one device.
    pub fn read_only_device(device: i32) -> Self {
        Self {
            location: HipMemLocation::device(device),
            flags: HIP_MEM_ACCESS_FLAGS_PROT_READ,
        }
    }
}

/// Loaded HIP runtime — holds the dlopen'd library and resolved function pointers.
pub struct HipRuntime {
    _lib: Arc<Library>,

    // Init
    fn_init: unsafe extern "C" fn(c_uint) -> u32,

    // Version
    fn_runtime_get_version: unsafe extern "C" fn(*mut c_int) -> u32,

    // Device management
    fn_get_device_count: unsafe extern "C" fn(*mut c_int) -> u32,
    fn_set_device: unsafe extern "C" fn(c_int) -> u32,
    fn_set_device_flags: unsafe extern "C" fn(c_uint) -> u32,
    fn_get_device: unsafe extern "C" fn(*mut c_int) -> u32,
    fn_device_get_uuid: unsafe extern "C" fn(*mut u8, c_int) -> u32,
    fn_device_get_pci_bus_id: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> u32,

    // Multi-device / peer access
    fn_device_can_access_peer: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> u32,
    fn_device_enable_peer_access: unsafe extern "C" fn(c_int, c_uint) -> u32,
    fn_memcpy_peer: unsafe extern "C" fn(*mut c_void, c_int, *const c_void, c_int, usize) -> u32,
    fn_memcpy_peer_async:
        unsafe extern "C" fn(*mut c_void, c_int, *const c_void, c_int, usize, HipStream) -> u32,
    fn_pointer_get_attributes: unsafe extern "C" fn(*mut HipPointerAttribute, *const c_void) -> u32,

    // Memory
    fn_malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> u32,
    fn_ext_malloc_with_flags: Option<unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> u32>,
    fn_free: unsafe extern "C" fn(*mut c_void) -> u32,
    fn_host_malloc: Option<unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> u32>,
    fn_host_get_device_pointer:
        Option<unsafe extern "C" fn(*mut *mut c_void, *mut c_void, c_uint) -> u32>,
    fn_host_free: Option<unsafe extern "C" fn(*mut c_void) -> u32>,
    fn_mem_get_address_range:
        unsafe extern "C" fn(*mut *mut c_void, *mut usize, *mut c_void) -> u32,
    fn_memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_uint) -> u32,
    fn_memcpy_async:
        unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_uint, HipStream) -> u32,
    /// `hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)`.
    /// Pitches and width are bytes; height is rows.
    fn_memcpy_2d_async: unsafe extern "C" fn(
        *mut c_void,
        usize,
        *const c_void,
        usize,
        usize,
        usize,
        c_uint,
        HipStream,
    ) -> u32,
    /// `hipMemcpyDtoD(dst, src, bytes)` (driver-style device pointers).
    fn_memcpy_dtod: Option<unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> u32>,
    fn_memset: unsafe extern "C" fn(*mut c_void, c_int, usize) -> u32,
    fn_memset_async: unsafe extern "C" fn(*mut c_void, c_int, usize, HipStream) -> u32,
    fn_memset_d32_async: unsafe extern "C" fn(*mut c_void, c_int, usize, HipStream) -> u32,
    fn_mem_address_reserve:
        Option<unsafe extern "C" fn(*mut *mut c_void, usize, usize, *mut c_void, u64) -> u32>,
    fn_mem_create: Option<
        unsafe extern "C" fn(
            *mut HipMemGenericAllocationHandle,
            usize,
            *const HipMemAllocationProp,
            u64,
        ) -> u32,
    >,
    fn_mem_get_allocation_granularity:
        Option<unsafe extern "C" fn(*mut usize, *const HipMemAllocationProp, u32) -> u32>,
    fn_mem_get_handle_properties: Option<
        unsafe extern "C" fn(*mut HipMemAllocationProp, HipMemGenericAllocationHandle) -> u32,
    >,
    fn_mem_map: Option<
        unsafe extern "C" fn(*mut c_void, usize, usize, HipMemGenericAllocationHandle, u64) -> u32,
    >,
    fn_mem_unmap: Option<unsafe extern "C" fn(*mut c_void, usize) -> u32>,
    fn_mem_set_access:
        Option<unsafe extern "C" fn(*mut c_void, usize, *const HipMemAccessDesc, usize) -> u32>,
    fn_mem_release: Option<unsafe extern "C" fn(HipMemGenericAllocationHandle) -> u32>,
    /// `hipMemExportToShareableHandle(void *out, handle, type, flags)`.
    #[cfg_attr(not(unix), allow(dead_code))]
    fn_mem_export_to_shareable_handle:
        Option<unsafe extern "C" fn(*mut c_void, HipMemGenericAllocationHandle, u32, u64) -> u32>,
    /// `hipMemImportFromShareableHandle(handle *out, void *osHandle, type)`.
    #[cfg_attr(not(unix), allow(dead_code))]
    fn_mem_import_from_shareable_handle:
        Option<unsafe extern "C" fn(*mut HipMemGenericAllocationHandle, *mut c_void, u32) -> u32>,
    /// `hipMemGetAccess(unsigned long long *flags, const hipMemLocation *, void *ptr)`.
    fn_mem_get_access:
        Option<unsafe extern "C" fn(*mut u64, *const HipMemLocation, *mut c_void) -> u32>,

    // Streams
    fn_stream_create: unsafe extern "C" fn(*mut HipStream) -> u32,
    fn_stream_create_with_flags: unsafe extern "C" fn(*mut HipStream, c_uint) -> u32,
    fn_stream_synchronize: unsafe extern "C" fn(HipStream) -> u32,
    fn_stream_destroy: unsafe extern "C" fn(HipStream) -> u32,

    // Modules & kernels
    fn_module_load: unsafe extern "C" fn(*mut HipModule, *const c_char) -> u32,
    fn_module_load_data: unsafe extern "C" fn(*mut HipModule, *const c_void) -> u32,
    fn_module_get_function: unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> u32,
    fn_module_launch_kernel: unsafe extern "C" fn(
        HipFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        HipStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> u32,
    fn_module_occupancy_max_active_blocks:
        Option<unsafe extern "C" fn(*mut c_int, HipFunction, c_int, usize) -> u32>,
    // Events
    fn_event_create: unsafe extern "C" fn(*mut HipEvent) -> u32,
    fn_event_create_with_flags: unsafe extern "C" fn(*mut HipEvent, c_uint) -> u32,
    fn_event_record: unsafe extern "C" fn(HipEvent, HipStream) -> u32,
    fn_event_query: unsafe extern "C" fn(HipEvent) -> u32,
    fn_event_synchronize: unsafe extern "C" fn(HipEvent) -> u32,
    fn_event_elapsed_time: unsafe extern "C" fn(*mut f32, HipEvent, HipEvent) -> u32,
    fn_event_destroy: unsafe extern "C" fn(HipEvent) -> u32,
    fn_stream_wait_event: unsafe extern "C" fn(HipStream, HipEvent, c_uint) -> u32,

    // Error
    fn_get_error_string: unsafe extern "C" fn(u32) -> *const i8,
    fn_get_last_error: unsafe extern "C" fn() -> u32,

    // Graph capture & replay
    fn_stream_begin_capture: unsafe extern "C" fn(HipStream, c_uint) -> u32,
    fn_stream_end_capture: unsafe extern "C" fn(HipStream, *mut HipGraph) -> u32,
    fn_stream_is_capturing: unsafe extern "C" fn(HipStream, *mut c_uint) -> u32,
    fn_graph_instantiate:
        unsafe extern "C" fn(*mut HipGraphExec, HipGraph, *mut HipGraph, *mut c_void, usize) -> u32,
    fn_graph_launch: unsafe extern "C" fn(HipGraphExec, HipStream) -> u32,
    fn_graph_exec_destroy: unsafe extern "C" fn(HipGraphExec) -> u32,
    fn_graph_destroy: unsafe extern "C" fn(HipGraph) -> u32,
    // Stream memory ops (HIP 7.2+)
    fn_stream_write_value32: unsafe extern "C" fn(HipStream, *mut c_void, u32, c_uint) -> u32,
    fn_stream_wait_value32:
        Option<unsafe extern "C" fn(HipStream, *mut c_void, u32, c_uint, u32) -> u32>,
    fn_device_synchronize: unsafe extern "C" fn() -> u32,
    fn_get_device_properties: unsafe extern "C" fn(*mut u8, c_int) -> u32,
    fn_get_device_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> u32,
    fn_mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> u32,
    /// `gridDim.y`/`gridDim.z` ceiling (`u32::MAX` = unguarded, the default).
    /// Per-runtime, not per-device or per-thread: it is set once by the owner
    /// of a runtime bound to a single device (`Gpu::init_with_device`) via
    /// `set_launch_grid_limit_for_arch`. A standalone `HipRuntime` that nobody
    /// configures is unguarded.
    launch_grid_yz_limit: AtomicU32,
    /// Device bytes this runtime has allocated and not yet freed (hipMalloc +
    /// fine-grained + VMM physical handles). The process's own footprint, so
    /// admission can plan from `total - own - reserve` instead of the
    /// instantaneous (foreign-allocation-sensitive) free figure.
    own_device_bytes: AtomicUsize,
    /// VMM physical handle -> size, so `mem_release` can debit the ledger.
    own_vmm_handles: std::sync::Mutex<std::collections::HashMap<usize, usize>>,
    /// Per-device cache of `hipDeviceAttributeCanUseStreamWaitValue`.
    stream_wait_caps: [AtomicU8; STREAM_WAIT_CAP_SLOTS],
    /// Plan-mode tape (railgun design A authoring); idle unless a plan is active.
    plan: PlanGate,
    #[cfg(feature = "kernel-timing")]
    kernel_timing_control: std::sync::OnceLock<Arc<kernel_timing::TimingControl>>,
}

// HipRuntime is Send+Sync — the underlying HIP runtime is thread-safe for API calls.
// SAFETY: HipRuntime holds only function pointers and a Library; HIP's C API
// is documented as safe to call from multiple threads. Sync allows
// shared refs; individual buffer/stream races remain the caller's.
unsafe impl Send for HipRuntime {}
// SAFETY: HipRuntime holds only function pointers and a Library; HIP's C API
// is documented as safe to call from multiple threads. Sync allows
// shared refs; individual buffer/stream races remain the caller's.
unsafe impl Sync for HipRuntime {}

macro_rules! load_fn {
    ($lib:expr, $name:expr, $ty:ty) => {{
        let sym: Symbol<'_, $ty> = $lib
            .get($name.as_bytes())
            .map_err(|e| HipError::new(0, &format!("failed to load symbol {}: {e}", $name)))?;
        *sym.into_raw()
    }};
}

macro_rules! load_optional_fn {
    ($lib:expr, $name:expr, $ty:ty) => {{
        $lib.get::<$ty>($name.as_bytes())
            .ok()
            .map(|sym| *sym.into_raw())
    }};
}

/// libhsakmt's choice for `hipHostMalloc` memory on a discrete GPU: non-zero
/// (ROCm's default) backs it with pageable userptr BOs, `0` with GTT BOs.
const HSA_USERPTR_FOR_PAGED_MEM: &str = "HSA_USERPTR_FOR_PAGED_MEM";

/// clr's size, in MB, from which a copy to or from pageable host memory pins
/// the pageable pages in place; smaller copies go through clr's own pinned
/// staging buffer.
const GPU_PINNED_MIN_XFER_SIZE: &str = "GPU_PINNED_MIN_XFER_SIZE";

/// Stage every pageable copy: larger than any host allocation.
const STAGE_ALL_PAGEABLE_COPIES_MB: &str = "100000";

/// Keep host memory the GPU reads out of the kernel's reclaim path, unless
/// the operator set the switches. `reason` names why, in the log line that
/// reports the switches set.
///
/// Under ROCm's defaults, host pages that the GPU reads are registered as
/// KFD userptr BOs. That covers every `hipHostMalloc` block, which is shared
/// anonymous memory, and the source of any large `hipMemcpy` from pageable
/// memory, such as the mapped model file, which clr pins in place. Reclaim or
/// migration may take those pages, and every invalidation evicts all of the
/// process's queues until KFD's restore worker has faulted them back. Under
/// sustained host-memory pressure the queues stay evicted. The GPU idles and
/// HIP's signal wait spins a core, with no error (ROCm/rocm-systems#12528).
/// This was seen with Qwen4's tens of GB of host-mapped routed experts, and
/// with weight uploads out of the mapped file.
///
/// `HSA_USERPTR_FOR_PAGED_MEM=0` backs `hipHostMalloc` memory with GTT BOs,
/// which are outside reclaim. TTM's `pages_limit` (half of RAM by default)
/// caps them instead. `GPU_PINNED_MIN_XFER_SIZE` set past any copy size
/// routes every pageable copy through clr's staging buffer, which is itself
/// `hipHostMalloc` memory. libhsakmt and clr read both switches once, when
/// the runtime initializes, so this must run before the first HIP runtime
/// call, while the process is still single-threaded. APUs ignore the first
/// switch.
///
/// The two switches cover the two classes of userptr page, and they are priced
/// very differently, so they are armed differently:
///
/// * `GPU_PINNED_MIN_XFER_SIZE` covers the **pageable copy source** — the
///   mapped model file's page-cache pages, which clr otherwise pins in place
///   for every large weight upload. That source exists on *every* weight load,
///   on every architecture, so it is armed in every Linux process, by
///   [`HipRuntime::load`] before the runtime is dlopen'd. Measured free on
///   gfx1201 / Qwen3.8-27B MQ3-Pro (warm cache, three reps each): weight sweep
///   1190-1198 ms armed vs 1204-1210 ms unarmed.
/// * `HSA_USERPTR_FOR_PAGED_MEM=0` covers `hipHostMalloc` blocks, and it is
///   the half that costs (same model and box: +8-11 % on the sweep; the commit
///   that introduced the gate measured ~+20 % on H2), so it stays scoped to a
///   process that loads a Qwen4 model, the one known to hold tens of GB of host
///   memory the GPU reads: [`HipRuntime::load`] sets it in a process configured
///   to host-map Qwen4 experts ([`QWEN4_EXPERT_VRAM_LAYERS_ENV`]), and a
///   process about to load a Qwen4 model on a discrete GPU calls this before
///   the runtime loads (`hipfire_loader::prepare_host_memory_for`).
///
/// Neither name is a published interface. `GPU_PINNED_MIN_XFER_SIZE` is absent
/// from HIP's own environment-variable reference and from AMD's env-var page
/// (checked against ROCm 7.2), and `HSA_USERPTR_FOR_PAGED_MEM` is a libhsakmt
/// knob; clr/libhsakmt read each once, at runtime initialization, so both are
/// set before the runtime is dlopen'd. They are used here because the
/// alternative is the stall described above (ROCm/rocm-systems#12528), and
/// `f5731a506` relied on them first, for host-mapped Qwen4 experts.
/// `docs/env-vars.md` records that provenance so neither reads as supported
/// API.
///
/// Evidence: under an 8 GiB host-memory ballast, an unarmed 27B load stalls at
/// a varying `loading layer N/64` on both a partial-GPU-offload and a
/// fully-resident configuration; `GPU_PINNED_MIN_XFER_SIZE` alone loads 7/7.
/// See the `fix/host-memory-reclaim-stall` branch commit for the full table.
pub fn keep_host_memory_out_of_reclaim(reason: &str) {
    if !cfg!(target_os = "linux") {
        return;
    }
    let mut set = Vec::new();
    for (name, value) in [
        (HSA_USERPTR_FOR_PAGED_MEM, "0"),
        (GPU_PINNED_MIN_XFER_SIZE, STAGE_ALL_PAGEABLE_COPIES_MB),
    ] {
        if std::env::var_os(name).is_none() {
            std::env::set_var(name, value);
            set.push(format!("{name}={value}"));
        }
    }
    if !set.is_empty() {
        eprintln!(
            "[hip-bridge] {} ({reason}): host memory out of reclaim",
            set.join(" ")
        );
    }
}

/// Exact opt-out from the process-wide [`GPU_PINNED_MIN_XFER_SIZE`] default:
/// when it is `0`, [`keep_pageable_copies_out_of_reclaim`] leaves the switch
/// unset, so clr keeps its own built-in threshold. That is an exact
/// restoration of the pre-default behavior, which setting
/// `GPU_PINNED_MIN_XFER_SIZE=0` is not (it pins every pageable copy source in
/// place, including copies below clr's built-in threshold).
const PINNED_XFER_DEFAULT_ENV: &str = "HIPFIRE_PINNED_XFER_DEFAULT";

/// What [`keep_pageable_copies_out_of_reclaim`] does with
/// `GPU_PINNED_MIN_XFER_SIZE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PageableCopyStaging {
    /// The operator already set the switch: respected untouched, whatever the
    /// opt-out says.
    Preset,
    /// [`PINNED_XFER_DEFAULT_ENV`] is `0`: leave the switch unset.
    OptedOut,
    /// Set the switch to [`STAGE_ALL_PAGEABLE_COPIES_MB`].
    Arm,
}

/// The decision behind [`keep_pageable_copies_out_of_reclaim`], pure so the
/// tests never touch the process environment. `opt_out` is the value of
/// [`PINNED_XFER_DEFAULT_ENV`] and `preset` that of `GPU_PINNED_MIN_XFER_SIZE`;
/// only an `opt_out` of exactly `0` opts out.
fn pageable_copy_staging(opt_out: Option<&OsStr>, preset: Option<&OsStr>) -> PageableCopyStaging {
    if preset.is_some() {
        PageableCopyStaging::Preset
    } else if opt_out == Some(OsStr::new("0")) {
        PageableCopyStaging::OptedOut
    } else {
        PageableCopyStaging::Arm
    }
}

/// [`PINNED_XFER_DEFAULT_ENV`] as this process sees it: the installed process
/// config when there is one, otherwise the ambient environment. Like
/// [`host_maps_qwen4_experts`], it never installs the local fallback config, so
/// it is safe before the HIP runtime loads.
fn pinned_xfer_default_opt_out() -> Option<OsString> {
    match hipfire_config::active_process_config() {
        Some(config) => config
            .legacy_value(PINNED_XFER_DEFAULT_ENV)
            .map(OsString::from),
        None => std::env::var_os(PINNED_XFER_DEFAULT_ENV),
    }
}

/// Arm clr's pageable-copy staging threshold (`GPU_PINNED_MIN_XFER_SIZE`) in
/// every Linux process, before the HIP runtime is dlopen'd, unless the operator
/// already set it or opted out with `HIPFIRE_PINNED_XFER_DEFAULT=0` (see
/// [`pageable_copy_staging`]). Separate from [`keep_host_memory_out_of_reclaim`]
/// because this switch is priced free while the `HSA_USERPTR_FOR_PAGED_MEM`
/// half is not, so the two are scoped differently. This is the tier that
/// covers the mapped-file copy source on *every* HFQ load:
/// [`HipRuntime::load`] is the one site every load path in every process
/// reaches before its first weight copy, where
/// `hipfire_loader::prepare_host_memory_for` runs earlier but only in a daemon
/// started for a model.
fn keep_pageable_copies_out_of_reclaim() {
    if !cfg!(target_os = "linux") {
        return;
    }
    let opt_out = pinned_xfer_default_opt_out();
    let preset = std::env::var_os(GPU_PINNED_MIN_XFER_SIZE);
    match pageable_copy_staging(opt_out.as_deref(), preset.as_deref()) {
        PageableCopyStaging::Preset => {}
        PageableCopyStaging::OptedOut => eprintln!(
            "[hip-bridge] {PINNED_XFER_DEFAULT_ENV}=0: {GPU_PINNED_MIN_XFER_SIZE} left unset (clr default)"
        ),
        PageableCopyStaging::Arm => {
            std::env::set_var(GPU_PINNED_MIN_XFER_SIZE, STAGE_ALL_PAGEABLE_COPIES_MB);
            eprintln!(
                "[hip-bridge] {GPU_PINNED_MIN_XFER_SIZE}={STAGE_ALL_PAGEABLE_COPIES_MB}: pageable weight copies staged out of reclaim"
            );
        }
    }
}

/// Places the routed experts of Qwen4 trunk layers at or past `N` in pinned,
/// device-mapped host RAM (`hipfire_arch_qwen4::expert_residency`). It is
/// named here because it decides [`keep_host_memory_out_of_reclaim`] before
/// the HIP runtime loads.
pub const QWEN4_EXPERT_VRAM_LAYERS_ENV: &str = "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS";

/// Whether this process is configured to host-map Qwen4 experts: the
/// installed process config when there is one (the daemon installs it before
/// any GPU runtime initializes), otherwise the ambient environment. Like
/// [`fault_spec`], it never installs the local fallback config.
fn host_maps_qwen4_experts() -> bool {
    match hipfire_config::active_process_config() {
        Some(config) => config.legacy_value(QWEN4_EXPERT_VRAM_LAYERS_ENV).is_some(),
        None => std::env::var_os(QWEN4_EXPERT_VRAM_LAYERS_ENV).is_some(),
    }
}

impl HipRuntime {
    /// Address of this runtime's resolved `hipEventRecord` implementation.
    ///
    /// Compare only while both runtime owners are alive, in the same process.
    /// Unlike a pathname, this identifies the live implementation even after a
    /// library is renamed or replaced. It makes no HIP call and does not use a
    /// process-global first-loader cache: each GPU's actual runtime is checked.
    pub fn event_record_address(&self) -> usize {
        self.fn_event_record as usize
    }

    /// Loader-reported library path for diagnostics. Identity comparisons must
    /// use [`Self::event_record_address`]; a missing path never proves equality.
    pub fn mapped_hip_library(&self) -> Option<String> {
        dladdr_fname(self.fn_event_record as *const c_void)
    }

    /// Load the HIP runtime via dlopen.
    /// Uses the shared ROCm resolver so runtime, headers, and hipcc stay within
    /// one selected installation.
    pub fn load() -> HipResult<Self> {
        // CU masks on RDNA hung the R9700 MES (2026-10-14): refuse before any
        // ROCm library is mapped.
        hipfire_config::devices::cu_mask_env_guard()
            .map_err(|message| HipError::new(0, &message))?;
        keep_pageable_copies_out_of_reclaim();
        if host_maps_qwen4_experts() {
            keep_host_memory_out_of_reclaim(&format!("{QWEN4_EXPERT_VRAM_LAYERS_ENV} set"));
        }
        // Windows and Unix share one candidate policy in hipfire_config::rocm so a
        // selected/configured root never falls through to another install's DLL
        // (user cache or bare PATH names). HIP_RUNTIME_LIBRARIES already encodes
        // the host's preferred filenames (amdhip64*.dll vs libamdhip64.so*).
        let candidates =
            hipfire_config::rocm::library_candidates(hipfire_config::rocm::HIP_RUNTIME_LIBRARIES);
        // SAFETY: open_library maps a shared object by path (plain
        // libloading::Library::new off Windows); on success the Library owns the
        // mapping for the process. Failure paths only read the error value.
        let lib = unsafe {
            let mut loaded = None;
            let mut last_err = None;
            for candidate in &candidates {
                match open_library(candidate) {
                    Ok(l) => {
                        loaded = Some(l);
                        break;
                    }
                    Err(e) => last_err = Some(e),
                }
            }
            match loaded {
                Some(l) => l,
                None => {
                    let runtime_label = if cfg!(target_os = "windows") {
                        "the HIP runtime (amdhip64.dll)"
                    } else {
                        "the HIP runtime (libamdhip64.so)"
                    };
                    let short_name = if cfg!(target_os = "windows") {
                        "amdhip64.dll"
                    } else {
                        "libamdhip64.so"
                    };
                    return Err(HipError::new(
                        0,
                        &format!(
                            "failed to load {short_name}: {}.\n{}",
                            last_err
                                .map(|e| e.to_string())
                                .unwrap_or_else(|| "no candidates".into()),
                            hipfire_config::rocm::resolution_failure(runtime_label, &candidates)
                        ),
                    ));
                }
            }
        };

        let runtime = Self {
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_init: unsafe { load_fn!(lib, "hipInit", unsafe extern "C" fn(c_uint) -> u32) },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_runtime_get_version: unsafe {
                load_fn!(
                    lib,
                    "hipRuntimeGetVersion",
                    unsafe extern "C" fn(*mut c_int) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_get_device_count: unsafe {
                load_fn!(
                    lib,
                    "hipGetDeviceCount",
                    unsafe extern "C" fn(*mut c_int) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_set_device: unsafe {
                load_fn!(lib, "hipSetDevice", unsafe extern "C" fn(c_int) -> u32)
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_set_device_flags: unsafe {
                load_fn!(
                    lib,
                    "hipSetDeviceFlags",
                    unsafe extern "C" fn(c_uint) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_get_device: unsafe {
                load_fn!(lib, "hipGetDevice", unsafe extern "C" fn(*mut c_int) -> u32)
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_device_get_uuid: unsafe {
                load_fn!(
                    lib,
                    "hipDeviceGetUuid",
                    unsafe extern "C" fn(*mut u8, c_int) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_device_get_pci_bus_id: unsafe {
                load_fn!(
                    lib,
                    "hipDeviceGetPCIBusId",
                    unsafe extern "C" fn(*mut c_char, c_int, c_int) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_device_can_access_peer: unsafe {
                load_fn!(
                    lib,
                    "hipDeviceCanAccessPeer",
                    unsafe extern "C" fn(*mut c_int, c_int, c_int) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_device_enable_peer_access: unsafe {
                load_fn!(
                    lib,
                    "hipDeviceEnablePeerAccess",
                    unsafe extern "C" fn(c_int, c_uint) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_memcpy_peer: unsafe {
                load_fn!(
                    lib,
                    "hipMemcpyPeer",
                    unsafe extern "C" fn(*mut c_void, c_int, *const c_void, c_int, usize) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_memcpy_peer_async: unsafe {
                load_fn!(
                    lib,
                    "hipMemcpyPeerAsync",
                    unsafe extern "C" fn(
                        *mut c_void,
                        c_int,
                        *const c_void,
                        c_int,
                        usize,
                        HipStream,
                    ) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_pointer_get_attributes: unsafe {
                load_fn!(
                    lib,
                    "hipPointerGetAttributes",
                    unsafe extern "C" fn(*mut HipPointerAttribute, *const c_void) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_malloc: unsafe {
                load_fn!(
                    lib,
                    "hipMalloc",
                    unsafe extern "C" fn(*mut *mut c_void, usize) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_ext_malloc_with_flags: unsafe {
                load_optional_fn!(
                    lib,
                    "hipExtMallocWithFlags",
                    unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_free: unsafe { load_fn!(lib, "hipFree", unsafe extern "C" fn(*mut c_void) -> u32) },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_host_malloc: unsafe {
                load_optional_fn!(
                    lib,
                    "hipHostMalloc",
                    unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_host_get_device_pointer: unsafe {
                load_optional_fn!(
                    lib,
                    "hipHostGetDevicePointer",
                    unsafe extern "C" fn(*mut *mut c_void, *mut c_void, c_uint) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_host_free: unsafe {
                load_optional_fn!(lib, "hipHostFree", unsafe extern "C" fn(*mut c_void) -> u32)
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_mem_get_address_range: unsafe {
                load_fn!(
                    lib,
                    "hipMemGetAddressRange",
                    unsafe extern "C" fn(*mut *mut c_void, *mut usize, *mut c_void) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_memcpy: unsafe {
                load_fn!(
                    lib,
                    "hipMemcpy",
                    unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_uint) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_memcpy_async: unsafe {
                load_fn!(
                    lib,
                    "hipMemcpyAsync",
                    unsafe extern "C" fn(
                        *mut c_void,
                        *const c_void,
                        usize,
                        c_uint,
                        HipStream,
                    ) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_memcpy_2d_async: unsafe {
                load_fn!(
                    lib,
                    "hipMemcpy2DAsync",
                    unsafe extern "C" fn(
                        *mut c_void,
                        usize,
                        *const c_void,
                        usize,
                        usize,
                        usize,
                        c_uint,
                        HipStream,
                    ) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_memcpy_dtod: unsafe {
                load_optional_fn!(
                    lib,
                    "hipMemcpyDtoD",
                    unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_memset: unsafe {
                load_fn!(
                    lib,
                    "hipMemset",
                    unsafe extern "C" fn(*mut c_void, c_int, usize) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_memset_async: unsafe {
                load_fn!(
                    lib,
                    "hipMemsetAsync",
                    unsafe extern "C" fn(*mut c_void, c_int, usize, HipStream) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_memset_d32_async: unsafe {
                load_fn!(
                    lib,
                    "hipMemsetD32Async",
                    unsafe extern "C" fn(*mut c_void, c_int, usize, HipStream) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_mem_address_reserve: unsafe {
                load_optional_fn!(
                    lib,
                    "hipMemAddressReserve",
                    unsafe extern "C" fn(*mut *mut c_void, usize, usize, *mut c_void, u64) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_mem_create: unsafe {
                load_optional_fn!(
                    lib,
                    "hipMemCreate",
                    unsafe extern "C" fn(
                        *mut HipMemGenericAllocationHandle,
                        usize,
                        *const HipMemAllocationProp,
                        u64,
                    ) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_mem_get_allocation_granularity: unsafe {
                load_optional_fn!(
                    lib,
                    "hipMemGetAllocationGranularity",
                    unsafe extern "C" fn(*mut usize, *const HipMemAllocationProp, u32) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_mem_get_handle_properties: unsafe {
                load_optional_fn!(
                    lib,
                    "hipMemGetAllocationPropertiesFromHandle",
                    unsafe extern "C" fn(
                        *mut HipMemAllocationProp,
                        HipMemGenericAllocationHandle,
                    ) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_mem_map: unsafe {
                load_optional_fn!(
                    lib,
                    "hipMemMap",
                    unsafe extern "C" fn(
                        *mut c_void,
                        usize,
                        usize,
                        HipMemGenericAllocationHandle,
                        u64,
                    ) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_mem_unmap: unsafe {
                load_optional_fn!(
                    lib,
                    "hipMemUnmap",
                    unsafe extern "C" fn(*mut c_void, usize) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_mem_set_access: unsafe {
                load_optional_fn!(
                    lib,
                    "hipMemSetAccess",
                    unsafe extern "C" fn(*mut c_void, usize, *const HipMemAccessDesc, usize) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_mem_release: unsafe {
                load_optional_fn!(
                    lib,
                    "hipMemRelease",
                    unsafe extern "C" fn(HipMemGenericAllocationHandle) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_mem_export_to_shareable_handle: unsafe {
                load_optional_fn!(
                    lib,
                    "hipMemExportToShareableHandle",
                    unsafe extern "C" fn(
                        *mut c_void,
                        HipMemGenericAllocationHandle,
                        u32,
                        u64,
                    ) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_mem_import_from_shareable_handle: unsafe {
                load_optional_fn!(
                    lib,
                    "hipMemImportFromShareableHandle",
                    unsafe extern "C" fn(
                        *mut HipMemGenericAllocationHandle,
                        *mut c_void,
                        u32,
                    ) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_mem_get_access: unsafe {
                load_optional_fn!(
                    lib,
                    "hipMemGetAccess",
                    unsafe extern "C" fn(*mut u64, *const HipMemLocation, *mut c_void) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_stream_create: unsafe {
                load_fn!(
                    lib,
                    "hipStreamCreate",
                    unsafe extern "C" fn(*mut HipStream) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_stream_create_with_flags: unsafe {
                load_fn!(
                    lib,
                    "hipStreamCreateWithFlags",
                    unsafe extern "C" fn(*mut HipStream, c_uint) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_stream_synchronize: unsafe {
                load_fn!(
                    lib,
                    "hipStreamSynchronize",
                    unsafe extern "C" fn(HipStream) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_stream_destroy: unsafe {
                load_fn!(
                    lib,
                    "hipStreamDestroy",
                    unsafe extern "C" fn(HipStream) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_module_load: unsafe {
                load_fn!(
                    lib,
                    "hipModuleLoad",
                    unsafe extern "C" fn(*mut HipModule, *const c_char) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_module_load_data: unsafe {
                load_fn!(
                    lib,
                    "hipModuleLoadData",
                    unsafe extern "C" fn(*mut HipModule, *const c_void) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_module_get_function: unsafe {
                load_fn!(
                    lib,
                    "hipModuleGetFunction",
                    unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_module_launch_kernel: unsafe {
                load_fn!(
                    lib,
                    "hipModuleLaunchKernel",
                    unsafe extern "C" fn(
                        HipFunction,
                        c_uint,
                        c_uint,
                        c_uint,
                        c_uint,
                        c_uint,
                        c_uint,
                        c_uint,
                        HipStream,
                        *mut *mut c_void,
                        *mut *mut c_void,
                    ) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_module_occupancy_max_active_blocks: unsafe {
                load_optional_fn!(
                    lib,
                    "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
                    unsafe extern "C" fn(*mut c_int, HipFunction, c_int, usize) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_event_create: unsafe {
                load_fn!(
                    lib,
                    "hipEventCreate",
                    unsafe extern "C" fn(*mut HipEvent) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_event_create_with_flags: unsafe {
                load_fn!(
                    lib,
                    "hipEventCreateWithFlags",
                    unsafe extern "C" fn(*mut HipEvent, c_uint) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_event_record: unsafe {
                load_fn!(
                    lib,
                    "hipEventRecord",
                    unsafe extern "C" fn(HipEvent, HipStream) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_event_query: unsafe {
                load_fn!(lib, "hipEventQuery", unsafe extern "C" fn(HipEvent) -> u32)
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_event_synchronize: unsafe {
                load_fn!(
                    lib,
                    "hipEventSynchronize",
                    unsafe extern "C" fn(HipEvent) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_event_elapsed_time: unsafe {
                load_fn!(
                    lib,
                    "hipEventElapsedTime",
                    unsafe extern "C" fn(*mut f32, HipEvent, HipEvent) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_event_destroy: unsafe {
                load_fn!(
                    lib,
                    "hipEventDestroy",
                    unsafe extern "C" fn(HipEvent) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_stream_wait_event: unsafe {
                load_fn!(
                    lib,
                    "hipStreamWaitEvent",
                    unsafe extern "C" fn(HipStream, HipEvent, c_uint) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_get_error_string: unsafe {
                load_fn!(
                    lib,
                    "hipGetErrorString",
                    unsafe extern "C" fn(u32) -> *const i8
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_get_last_error: unsafe {
                load_fn!(lib, "hipGetLastError", unsafe extern "C" fn() -> u32)
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_stream_begin_capture: unsafe {
                load_fn!(
                    lib,
                    "hipStreamBeginCapture",
                    unsafe extern "C" fn(HipStream, c_uint) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_stream_end_capture: unsafe {
                load_fn!(
                    lib,
                    "hipStreamEndCapture",
                    unsafe extern "C" fn(HipStream, *mut HipGraph) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_stream_is_capturing: unsafe {
                load_fn!(
                    lib,
                    "hipStreamIsCapturing",
                    unsafe extern "C" fn(HipStream, *mut c_uint) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_graph_instantiate: unsafe {
                load_fn!(
                    lib,
                    "hipGraphInstantiate",
                    unsafe extern "C" fn(
                        *mut HipGraphExec,
                        HipGraph,
                        *mut HipGraph,
                        *mut c_void,
                        usize,
                    ) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_graph_launch: unsafe {
                load_fn!(
                    lib,
                    "hipGraphLaunch",
                    unsafe extern "C" fn(HipGraphExec, HipStream) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_graph_exec_destroy: unsafe {
                load_fn!(
                    lib,
                    "hipGraphExecDestroy",
                    unsafe extern "C" fn(HipGraphExec) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_graph_destroy: unsafe {
                load_fn!(
                    lib,
                    "hipGraphDestroy",
                    unsafe extern "C" fn(HipGraph) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_stream_write_value32: unsafe {
                load_fn!(
                    lib,
                    "hipStreamWriteValue32",
                    unsafe extern "C" fn(HipStream, *mut c_void, u32, c_uint) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_stream_wait_value32: unsafe {
                load_optional_fn!(
                    lib,
                    "hipStreamWaitValue32",
                    unsafe extern "C" fn(HipStream, *mut c_void, u32, c_uint, u32) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_device_synchronize: unsafe {
                load_fn!(lib, "hipDeviceSynchronize", unsafe extern "C" fn() -> u32)
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_get_device_properties: unsafe {
                load_fn!(
                    lib,
                    "hipGetDeviceProperties",
                    unsafe extern "C" fn(*mut u8, c_int) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_get_device_attribute: unsafe {
                load_fn!(
                    lib,
                    "hipDeviceGetAttribute",
                    unsafe extern "C" fn(*mut c_int, c_int, c_int) -> u32
                )
            },
            // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
            fn_mem_get_info: unsafe {
                load_fn!(
                    lib,
                    "hipMemGetInfo",
                    unsafe extern "C" fn(*mut usize, *mut usize) -> u32
                )
            },
            _lib: Arc::new(lib),
            launch_grid_yz_limit: AtomicU32::new(u32::MAX),
            own_device_bytes: AtomicUsize::new(0),
            own_vmm_handles: Default::default(),
            stream_wait_caps: std::array::from_fn(|_| AtomicU8::new(STREAM_WAIT_CAP_UNKNOWN)),
            plan: PlanGate::default(),
            #[cfg(feature = "kernel-timing")]
            kernel_timing_control: std::sync::OnceLock::new(),
        };

        // Empirically required on ROCm 7.2 — hipcc-linked binaries get an
        // implicit init via shared-library constructors at process start;
        // a dlopen'd runtime doesn't, and ROCm 7.2's hipModuleLoad returns
        // 303 (hipErrorSharedObjectInitFailed) on otherwise-valid .hsaco
        // blobs without it. Older HIP libs implicitly init on the first
        // hipMalloc/hipGetDevice call, so the absence used to be tolerated.
        // hipInit(0) is documented as safe to call repeatedly.
        // SAFETY: HIP runtime symbol is live from HipRuntime::load; out-params are
        // stack locals (or caller-provided device ids). No host pointer aliasing.
        let code = unsafe { (runtime.fn_init)(0) };
        runtime.check(code, "hipInit")?;
        Ok(runtime)
    }

    // ── Plan mode (railgun design A authoring) ──────────────────

    /// Begin a plan tape on this thread. Blob launches, D2D copies and memsets
    /// are recorded instead of issued; every other HIP API refuses.
    pub fn plan_begin(&self) -> HipResult<()> {
        self.plan.begin()
    }

    /// Finish the plan tape and take its events / first refusal.
    pub fn plan_end(&self) -> PlanTape {
        self.plan.end()
    }

    /// True on the thread that began an active plan tape.
    pub fn plan_active(&self) -> bool {
        self.plan.active()
    }

    /// Mark the active plan refused (a no-op outside plan mode).
    pub fn plan_refuse(&self, why: &str) {
        if self.plan.active() {
            let _ = self.plan.refuse(why);
        }
    }

    /// Label stamped on every launch recorded from now on.
    pub fn plan_set_label(&self, label: &'static str) {
        if self.plan.active() {
            self.plan.set_label(label);
        }
    }

    /// Events recorded so far (0 outside plan mode).
    pub fn plan_event_count(&self) -> usize {
        self.plan.event_count()
    }

    /// Index of the most recent recorded launch.
    pub fn plan_last_launch_index(&self) -> Option<usize> {
        self.plan.last_launch_index()
    }

    /// Start recording every real blob launch of this thread (they still
    /// execute / get graph-captured): the oracle's ground truth for what an
    /// ordinary window launches, compared against a plan's launch list.
    pub fn launch_witness_begin(&self) -> HipResult<()> {
        self.plan.witness_begin()
    }

    /// Stop the launch witness and take its launches (in order).
    pub fn launch_witness_end(&self) -> Vec<PlanEvent> {
        self.plan.witness_end()
    }

    #[inline]
    fn plan_deny(&self, api: &str) -> HipResult<()> {
        if self.plan.active() {
            Err(self
                .plan
                .refuse(&format!("HIP API {api} reached in plan mode")))
        } else {
            Ok(())
        }
    }

    #[inline]
    fn plan_copy_at(
        &self,
        api: &str,
        dst: &DeviceBuffer,
        dst_off: usize,
        src: &DeviceBuffer,
        src_off: usize,
        size: usize,
    ) -> Option<HipResult<()>> {
        if !self.plan.active() {
            return None;
        }
        Some(self.plan.copy_range(
            api,
            dst.as_ptr() as u64,
            dst.size(),
            dst_off,
            src.as_ptr() as u64,
            src.size(),
            src_off,
            size,
        ))
    }

    /// Stream-ordered D2D copy of raw device addresses (the emitted-twin
    /// runner of a plan's `Copy` op). Not plan-recorded.
    ///
    /// # Safety
    /// `[src, src+size)` and `[dst, dst+size)` must lie in live, disjoint
    /// device allocations that outlive the copy.
    pub unsafe fn memcpy_dtod_async_raw(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        stream: Option<&Stream>,
    ) -> HipResult<()> {
        let stream_raw = stream.map_or(ptr::null_mut(), |s| s.0);
        // SAFETY: the caller guarantees both ranges are live disjoint device
        // allocations; Rust does not deref device pointers.
        let code = unsafe {
            (self.fn_memcpy_async)(
                dst,
                src,
                size,
                MemcpyKind::DeviceToDevice as c_uint,
                stream_raw,
            )
        };
        memory_effects::dtod();
        self.check(code, "hipMemcpyAsync D2D raw")
    }

    /// Stream-ordered 32-bit fill of raw device words (the emitted-twin
    /// runner of a plan's `Fill` op). Not plan-recorded.
    ///
    /// # Safety
    /// `[dst, dst + 4*words)` must lie in a live device allocation that
    /// outlives the fill.
    pub unsafe fn memset_d32_async_raw(
        &self,
        dst: *mut c_void,
        value: i32,
        words: usize,
        stream: Option<&Stream>,
    ) -> HipResult<()> {
        let stream_raw = stream.map_or(ptr::null_mut(), |s| s.0);
        // SAFETY: the caller guarantees the word range is a live device
        // allocation; Rust does not deref device pointers.
        let code = unsafe { (self.fn_memset_d32_async)(dst, value, words, stream_raw) };
        memory_effects::memset();
        self.check(code, "hipMemsetD32Async raw")
    }

    fn check(&self, code: u32, context: &str) -> HipResult<()> {
        if code == HIP_SUCCESS {
            Ok(())
        } else {
            Err(HipError::from_code(
                code,
                context,
                Some(&self.fn_get_error_string),
            ))
        }
    }

    fn missing_vmm_symbol<T>(&self, symbol: &str, value: Option<T>) -> HipResult<T> {
        value.ok_or_else(|| {
            HipError::new(0, &format!("{symbol} is not available in this HIP runtime"))
        })
    }

    // ── Version ────────────────────────────────────────────────

    /// Raw `hipRuntimeGetVersion` value (HIP encoding:
    /// `major * 10_000_000 + minor * 100_000 + patch`).
    pub fn runtime_version_raw(&self) -> HipResult<i32> {
        let mut version: c_int = 0;
        // SAFETY: HIP runtime symbol is live from HipRuntime::load; out-params are
        // stack locals (or caller-provided device ids). No host pointer aliasing.
        let code = unsafe { (self.fn_runtime_get_version)(&mut version) };
        self.check(code, "hipRuntimeGetVersion")?;
        Ok(version)
    }

    /// Get HIP runtime version as (major, minor). E.g. ROCm 6.3 → (6, 3).
    pub fn runtime_version(&self) -> HipResult<(i32, i32)> {
        let version = self.runtime_version_raw()?;
        // HIP version encoding: major * 10000000 + minor * 100000 + patch
        let major = version / 10_000_000;
        let minor = (version % 10_000_000) / 100_000;
        Ok((major, minor))
    }

    /// Raw `hipDriverGetVersion` value. The symbol is resolved from the held
    /// library on each call rather than in the eager table in [`Self::load`], so
    /// a runtime that lacks it still loads; the error names the missing symbol.
    pub fn driver_version(&self) -> HipResult<i32> {
        let mut version: c_int = 0;
        // SAFETY: the symbol is resolved from the live `Library` owned by
        // `_lib` and called while that borrow is held; the out-param is a stack
        // local.
        let code = unsafe {
            let f: Symbol<'_, unsafe extern "C" fn(*mut c_int) -> u32> =
                self._lib.get(b"hipDriverGetVersion").map_err(|e| {
                    HipError::new(
                        0,
                        &format!("failed to load symbol hipDriverGetVersion: {e}"),
                    )
                })?;
            f(&mut version)
        };
        self.check(code, "hipDriverGetVersion")?;
        Ok(version)
    }

    // ── Device management ───────────────────────────────────────

    pub fn device_count(&self) -> HipResult<i32> {
        let mut count: c_int = 0;
        // SAFETY: HIP runtime symbol is live from HipRuntime::load; out-params are
        // stack locals (or caller-provided device ids). No host pointer aliasing.
        let code = unsafe { (self.fn_get_device_count)(&mut count) };
        self.check(code, "hipGetDeviceCount")?;
        Ok(count)
    }
    /// ROCm encodes the 16 hex digits of the physical GPU UUID as ASCII in
    /// hipUUID.bytes. Retain that spelling so it matches GPU-... selectors.
    pub fn device_uuid(&self, id: i32) -> HipResult<String> {
        let mut bytes = [0u8; 16];
        // SAFETY: HIP runtime symbol is live from HipRuntime::load; out-params are
        // stack locals (or caller-provided device ids). No host pointer aliasing.
        let code = unsafe { (self.fn_device_get_uuid)(bytes.as_mut_ptr(), id) };
        self.check(code, "hipDeviceGetUuid")?;
        let mut uuid = String::with_capacity(36);
        uuid.push_str("GPU-");
        if bytes.iter().all(u8::is_ascii_hexdigit) {
            for byte in bytes {
                uuid.push((byte as char).to_ascii_lowercase());
            }
        } else {
            use std::fmt::Write;
            for byte in bytes {
                write!(&mut uuid, "{byte:02x}").expect("write to String");
            }
        }
        Ok(uuid)
    }

    pub fn device_pci_bus_id(&self, id: i32) -> HipResult<String> {
        let mut bytes = [0 as c_char; 32];
        // SAFETY: HIP runtime symbol is live from HipRuntime::load; out-params are
        // stack locals (or caller-provided device ids). No host pointer aliasing.
        let code = unsafe { (self.fn_device_get_pci_bus_id)(bytes.as_mut_ptr(), 32, id) };
        self.check(code, "hipDeviceGetPCIBusId")?;
        // SAFETY: on success HIP wrote a NUL-terminated C string into the local
        // buffer; the pointer addresses that buffer only for this read.
        Ok(unsafe { std::ffi::CStr::from_ptr(bytes.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    pub fn set_device(&self, id: i32) -> HipResult<()> {
        // SAFETY: HIP runtime symbol is live from HipRuntime::load; out-params are
        // stack locals (or caller-provided device ids). No host pointer aliasing.
        let code = unsafe { (self.fn_set_device)(id) };
        self.check(code, "hipSetDevice")
    }

    /// Set HIP device scheduling flags before the device context is created.
    ///
    /// HIP uses these flags to choose how host sync calls wait for GPU work:
    /// spin is lowest-latency but consumes a CPU core; yield/blocking are more
    /// cooperative with the OS and reduce CPU pressure during long prefill.
    pub fn set_device_flags(&self, flags: u32) -> HipResult<()> {
        // SAFETY: HIP runtime symbol is live from HipRuntime::load; out-params are
        // stack locals (or caller-provided device ids). No host pointer aliasing.
        let code = unsafe { (self.fn_set_device_flags)(flags as c_uint) };
        self.check(code, "hipSetDeviceFlags")
    }

    pub fn current_device(&self) -> HipResult<i32> {
        let mut id: c_int = 0;
        // SAFETY: HIP runtime symbol is live from HipRuntime::load; out-params are
        // stack locals (or caller-provided device ids). No host pointer aliasing.
        let code = unsafe { (self.fn_get_device)(&mut id) };
        self.check(code, "hipGetDevice")?;
        Ok(id)
    }

    // ── Multi-device / peer access ──────────────────────────────

    pub fn can_access_peer(&self, device: i32, peer_device: i32) -> HipResult<bool> {
        let mut can: c_int = 0;
        // SAFETY: HIP runtime symbol is live from HipRuntime::load; out-params are
        // stack locals (or caller-provided device ids). No host pointer aliasing.
        let code = unsafe { (self.fn_device_can_access_peer)(&mut can, device, peer_device) };
        self.check(code, "hipDeviceCanAccessPeer")?;
        Ok(can != 0)
    }

    /// Idempotent: hipErrorPeerAccessAlreadyEnabled (704) → Ok. Caller must
    /// have bound the source device first (`set_device`).
    pub fn enable_peer_access(&self, peer_device: i32) -> HipResult<()> {
        // SAFETY: HIP runtime symbol is live from HipRuntime::load; out-params are
        // stack locals (or caller-provided device ids). No host pointer aliasing.
        let code = unsafe { (self.fn_device_enable_peer_access)(peer_device, 0) };
        if code == HIP_SUCCESS || code == crate::HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED {
            Ok(())
        } else {
            Err(HipError::from_code(
                code,
                "hipDeviceEnablePeerAccess",
                Some(&self.fn_get_error_string),
            ))
        }
    }

    pub fn memcpy_peer(
        &self,
        dst: &DeviceBuffer,
        dst_device: i32,
        src: &DeviceBuffer,
        src_device: i32,
        size: usize,
    ) -> HipResult<()> {
        self.plan_deny("hipMemcpyPeer")?;
        assert!(
            size <= dst.size(),
            "size ({size}) exceeds dst ({})",
            dst.size()
        );
        assert!(
            size <= src.size(),
            "size ({size}) exceeds src ({})",
            src.size()
        );
        // SAFETY: pointers are live HIP allocations or host slices; sizes were
        // asserted to fit. Same-device/stream and host-buffer lifetime across
        // async copies are the caller's contract; Rust does not deref device ptrs.
        let code = unsafe {
            (self.fn_memcpy_peer)(
                dst.as_ptr(),
                dst_device,
                src.as_ptr() as *const c_void,
                src_device,
                size,
            )
        };
        memory_effects::dtod();
        self.check(code, "hipMemcpyPeer")
    }

    /// Stream must belong to one of the two devices involved.
    pub fn memcpy_peer_async(
        &self,
        dst: &DeviceBuffer,
        dst_device: i32,
        src: &DeviceBuffer,
        src_device: i32,
        size: usize,
        stream: &Stream,
    ) -> HipResult<()> {
        self.plan_deny("hipMemcpyPeerAsync")?;
        assert!(
            size <= dst.size(),
            "size ({size}) exceeds dst ({})",
            dst.size()
        );
        assert!(
            size <= src.size(),
            "size ({size}) exceeds src ({})",
            src.size()
        );
        // SAFETY: pointers are live HIP allocations or host slices; sizes were
        // asserted to fit. Same-device/stream and host-buffer lifetime across
        // async copies are the caller's contract; Rust does not deref device ptrs.
        let code = unsafe {
            (self.fn_memcpy_peer_async)(
                dst.as_ptr(),
                dst_device,
                src.as_ptr() as *const c_void,
                src_device,
                size,
                stream.0,
            )
        };
        memory_effects::dtod();
        self.check(code, "hipMemcpyPeerAsync")
    }

    pub fn pointer_get_attributes(&self, buf: &DeviceBuffer) -> HipResult<HipPointerAttribute> {
        let mut attr = HipPointerAttribute::default();
        // SAFETY: HIP runtime symbol is live from HipRuntime::load; out-params are
        // stack locals (or caller-provided device ids). No host pointer aliasing.
        let code = unsafe { (self.fn_pointer_get_attributes)(&mut attr, buf.as_ptr()) };
        self.check(code, "hipPointerGetAttributes")?;
        Ok(attr)
    }

    // ── Memory management ───────────────────────────────────────

    pub fn malloc(&self, size: usize) -> HipResult<DeviceBuffer> {
        self.plan_deny("hipMalloc")?;
        let Some(budget) = crate::emu_budget::active() else {
            return self.malloc_raw(size);
        };
        let (buf, token) = crate::emu_budget::charged_alloc(
            &budget,
            crate::emu_budget::Class::Device,
            size,
            || self.malloc_raw(size),
        )?;
        Ok(buf.with_charge(token))
    }

    fn malloc_raw(&self, size: usize) -> HipResult<DeviceBuffer> {
        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: FFI out-params are stack locals with valid writable provenance;
        // HIP symbols were resolved at load. Returned pointers are opaque to Rust.
        let code = unsafe { (self.fn_malloc)(&mut ptr, size) };
        self.check(code, "hipMalloc")?;
        crate::registry::record(ptr as usize, size, crate::registry::AllocationKind::Malloc);
        self.own_device_bytes.fetch_add(size, Ordering::Relaxed);
        Ok(DeviceBuffer {
            ptr,
            size,
            ownership: crate::DeviceBufferOwnership::HipMalloc,
            token: None,
        })
    }

    /// Allocate system-visible signal memory for stream wait/write operations.
    ///
    /// This is an optional ROCm capability. Loading `HipRuntime` remains
    /// compatible with runtimes that do not export `hipExtMallocWithFlags`;
    /// callers receive a scoped error only when they request signal memory.
    pub fn malloc_signal(&self, size: usize) -> HipResult<DeviceBuffer> {
        self.plan_deny("hipMalloc(signal)")?;
        let Some(budget) = crate::emu_budget::active() else {
            return self.malloc_signal_raw(size);
        };
        let (buf, token) = crate::emu_budget::charged_alloc(
            &budget,
            crate::emu_budget::Class::Device,
            size,
            || self.malloc_signal_raw(size),
        )?;
        Ok(buf.with_charge(token))
    }

    /// Allocate fine-grained device VRAM (`hipDeviceMallocFinegrained`, 0x1
    /// in the installed `hip_runtime_api.h`) through the optional
    /// `hipExtMallocWithFlags` entry. Kernels on the owning device and on
    /// peers with access enabled see system-scope stores and loads of this
    /// memory coherently without a kernel boundary. Allocate before enabling
    /// peer access so the peer mapping covers it. Success alone does not
    /// qualify the memory; callers still run a publication probe.
    pub fn malloc_peer_finegrained(&self, size: usize) -> HipResult<DeviceBuffer> {
        self.plan_deny("hipMalloc(finegrained)")?;
        let Some(budget) = crate::emu_budget::active() else {
            return self.malloc_finegrained_raw(size);
        };
        let (buf, token) = crate::emu_budget::charged_alloc(
            &budget,
            crate::emu_budget::Class::Device,
            size,
            || self.malloc_finegrained_raw(size),
        )?;
        Ok(buf.with_charge(token))
    }

    fn malloc_finegrained_raw(&self, size: usize) -> HipResult<DeviceBuffer> {
        const HIP_DEVICE_MALLOC_FINEGRAINED: c_uint = 0x1;
        let Some(ext_malloc) = self.fn_ext_malloc_with_flags else {
            return Err(HipError::new(
                0,
                "hipExtMallocWithFlags unavailable; fine-grained VRAM is unsupported",
            ));
        };
        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: FFI out-params are stack locals with valid writable provenance;
        // HIP symbols were resolved at load. Returned pointers are opaque to Rust.
        let code = unsafe { ext_malloc(&mut ptr, size, HIP_DEVICE_MALLOC_FINEGRAINED) };
        self.check(code, "hipExtMallocWithFlags(hipDeviceMallocFinegrained)")?;
        crate::registry::record(
            ptr as usize,
            size,
            crate::registry::AllocationKind::FineGrained,
        );
        self.own_device_bytes.fetch_add(size, Ordering::Relaxed);
        Ok(DeviceBuffer {
            ptr,
            size,
            ownership: crate::DeviceBufferOwnership::HipMalloc,
            token: None,
        })
    }

    fn malloc_signal_raw(&self, size: usize) -> HipResult<DeviceBuffer> {
        const HIP_MALLOC_SIGNAL_MEMORY: c_uint = 0x2;
        let Some(ext_malloc) = self.fn_ext_malloc_with_flags else {
            return Err(HipError::new(
                0,
                "hipExtMallocWithFlags unavailable; signal memory is unsupported",
            ));
        };
        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: FFI out-params are stack locals with valid writable provenance;
        // HIP symbols were resolved at load. Returned pointers are opaque to Rust.
        let code = unsafe { ext_malloc(&mut ptr, size, HIP_MALLOC_SIGNAL_MEMORY) };
        self.check(code, "hipExtMallocWithFlags(hipMallocSignalMemory)")?;
        crate::registry::record(ptr as usize, size, crate::registry::AllocationKind::Signal);
        Ok(DeviceBuffer {
            ptr,
            size,
            ownership: crate::DeviceBufferOwnership::HipMalloc,
            token: None,
        })
    }

    /// # Safety
    /// Caller must ensure the buffer is not in use by any pending GPU operations.
    pub fn free(&self, mut buf: DeviceBuffer) -> HipResult<()> {
        self.plan_deny("hipFree")?;
        if !buf.is_hip_allocation() {
            return Err(HipError::new(
                0,
                "hipFree rejected a borrowed or VMM DeviceBuffer",
            ));
        }
        // Emu charge (None when off): released only if hipFree succeeds; a
        // failed free drops the token without releasing, keeping the charge.
        let token = buf.take_charge();
        let size = buf.size;
        crate::emu_budget::charged_free::<crate::emu_budget::Process>(token, || {
            // SAFETY: buf.ptr is a live hipMalloc allocation (ownership checked above);
            // caller must ensure GPU work on it is quiesced (documented on free).
            let code = unsafe { (self.fn_free)(buf.ptr) };
            self.check(code, "hipFree")?;
            crate::registry::forget(buf.ptr as usize);
            self.own_device_bytes
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                    Some(v.saturating_sub(size))
                })
                .ok();
            Ok(())
        })
    }

    /// Device bytes this runtime currently owns (see `own_device_bytes`).
    pub fn own_device_bytes(&self) -> usize {
        self.own_device_bytes.load(Ordering::Relaxed)
    }

    /// Allocate host-pinned memory the GPU can read directly over PCIe.
    ///
    /// `flags` are the HIP host-malloc flags; bit 1 (`hipHostMallocMapped`) is
    /// what makes the pages reachable from device code, and the returned address
    /// is then the *host* pointer — use [`Self::host_get_device_pointer`] for the
    /// address to hand a kernel.
    pub fn host_malloc(&self, size: usize, flags: u32) -> HipResult<*mut c_void> {
        self.plan_deny("hipHostMalloc")?;
        let Some(budget) = crate::emu_budget::active() else {
            return self.host_malloc_raw(size, flags);
        };
        crate::emu_budget::charged_alloc_keyed(
            &budget,
            &crate::emu_budget::HOST_TOKENS,
            crate::emu_budget::Class::HostPinned,
            size,
            |ptr: &*mut c_void| *ptr as usize,
            || self.host_malloc_raw(size, flags),
        )
    }

    fn host_malloc_raw(&self, size: usize, flags: u32) -> HipResult<*mut c_void> {
        let func = self.missing_vmm_symbol("hipHostMalloc", self.fn_host_malloc)?;
        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: resolved VMM/host HIP symbol; pointer/handle args meet the method's
        // # Safety or documented preconditions; out-params are stack locals.
        let code = unsafe { func(&mut ptr, size, flags) };
        self.check(code, "hipHostMalloc")?;
        Ok(ptr)
    }

    /// Device-visible address of a `hipHostMalloc`'d buffer.
    ///
    /// HIP treats `host` as an opaque address; Rust never dereferences it.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn host_get_device_pointer(&self, host: *mut c_void, flags: u32) -> HipResult<*mut c_void> {
        self.plan_deny("hipHostGetDevicePointer")?;
        let func =
            self.missing_vmm_symbol("hipHostGetDevicePointer", self.fn_host_get_device_pointer)?;
        let mut dev: *mut c_void = ptr::null_mut();
        // SAFETY: resolved VMM/host HIP symbol; pointer/handle args meet the method's
        // # Safety or documented preconditions; out-params are stack locals.
        let code = unsafe { func(&mut dev, host, flags) };
        self.check(code, "hipHostGetDevicePointer")?;
        Ok(dev)
    }

    /// Release a `hipHostMalloc`'d buffer. # Safety: must not be in use on GPU.
    ///
    /// HIP treats `host` as an opaque address; Rust never dereferences it.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn host_free(&self, host: *mut c_void) -> HipResult<()> {
        self.plan_deny("hipHostFree")?;
        let func = self.missing_vmm_symbol("hipHostFree", self.fn_host_free)?;
        let raw = || {
            // SAFETY: resolved VMM/host HIP symbol; pointer/handle args meet the method's
            // # Safety or documented preconditions; out-params are stack locals.
            let code = unsafe { func(host) };
            self.check(code, "hipHostFree")
        };
        if crate::emu_budget::active().is_none() {
            return raw();
        }
        // Release the pinned-host charge only if hipHostFree succeeds.
        crate::emu_budget::release_keyed::<crate::emu_budget::Process>(
            &crate::emu_budget::HOST_TOKENS,
            host as usize,
            raw,
        )
    }

    /// Take shared ownership of a `hipHostMalloc`'d allocation so a stream-held lease can
    /// outlive both the registry that allocated it and this runtime.  The returned
    /// [`HostAllocOwner`] frees it (once, on last drop); nothing else may.
    ///
    /// # Safety
    /// `host` must be non-null, must have been returned by this runtime's `host_malloc`, must
    /// not have been freed, must be adopted AT MOST ONCE, and after adoption nothing but the
    /// returned owner (and the references cloned from it) may call `host_free` on it — the
    /// caller must stop freeing it directly (the registry that held it drops its record
    /// without freeing).
    pub unsafe fn adopt_host_allocation(&self, host: *mut c_void) -> HipResult<HostAllocOwner> {
        if host.is_null() {
            return Err(HipError::new(0, "adopt_host_allocation: null host pointer"));
        }
        let free = self.missing_vmm_symbol("hipHostFree", self.fn_host_free)?;
        Ok(HostAllocOwner {
            ptr: host,
            free,
            _lib: Some(Arc::clone(&self._lib)),
        })
    }

    /// Own reusable pinned host storage, portable across devices. The library
    /// stays loaded until the allocation is freed, even if the runtime drops.
    pub fn host_buffer(&self, size: usize) -> HipResult<PinnedHostBuffer> {
        self.plan_deny("pinned host_buffer alloc")?;
        if size == 0 {
            return Err(HipError::new(0, "host_buffer requires non-zero size"));
        }
        let free = self.missing_vmm_symbol("hipHostFree", self.fn_host_free)?;
        // hipHostMallocPortable: this thread-local stage may outlive a Gpu or
        // be reused with another device after a model swap.
        let (ptr, token) = match crate::emu_budget::active() {
            None => (self.host_malloc_raw(size, 1)?, None),
            // The buffer owns its token (not the pointer-keyed table): its
            // Drop calls the raw free function directly.
            Some(budget) => crate::emu_budget::charged_alloc(
                &budget,
                crate::emu_budget::Class::HostPinned,
                size,
                || self.host_malloc_raw(size, 1),
            )?,
        };
        // SAFETY: hipHostMalloc succeeded with a non-zero extent. Initialize
        // once so as_bytes_mut can expose ordinary initialized Rust bytes.
        unsafe { std::ptr::write_bytes(ptr.cast::<u8>(), 0, size) };
        Ok(PinnedHostBuffer {
            ptr,
            size,
            free,
            token,
            _lib: Arc::clone(&self._lib),
        })
    }

    /// Reusable pinned CPU/GPU transfer storage. Same-stream downloads may be
    /// queued after an outstanding upload without exposing host references.
    pub fn transfer_buffer(&self, size: usize) -> HipResult<PinnedTransferBuffer> {
        self.plan_deny("pinned transfer_buffer alloc")?;
        Ok(PinnedTransferBuffer {
            storage: Some(self.host_buffer(size)?),
            device: self.current_device()?,
            stream: ptr::null_mut(),
            ledger: TransferLedger::new(),
            ready_event: None,
            get_device: self.fn_get_device,
            set_device: self.fn_set_device,
            synchronize: self.fn_device_synchronize,
            event_destroy: self.fn_event_destroy,
        })
    }

    /// Mapped + coherent host storage with a non-owning device alias, for CPU
    /// results the GPU reads in place. See [`MappedHostBuffer`].
    ///
    /// Refuses a zero/oversized extent and any alias failure (including a null
    /// alias); there is no host-pointer fallback. Starts zeroed, `Retired`,
    /// generation 0. The library stays loaded until the allocation is freed.
    pub fn mapped_host_buffer(&self, size: usize) -> HipResult<MappedHostBuffer> {
        self.plan_deny("mapped_host_buffer alloc")?;
        check_mapped_extent(size)?;
        let free = self.missing_vmm_symbol("hipHostFree", self.fn_host_free)?;
        let device = self.current_device()?;
        let retire_event = self.event_create_with_flags(HIP_EVENT_DISABLE_TIMING)?;
        let flags = HIP_HOST_MALLOC_MAPPED | HIP_HOST_MALLOC_COHERENT;
        let allocated = match crate::emu_budget::active() {
            None => self.host_malloc_raw(size, flags).map(|p| (p, None)),
            Some(budget) => crate::emu_budget::charged_alloc(
                &budget,
                crate::emu_budget::Class::HostPinned,
                size,
                || self.host_malloc_raw(size, flags),
            ),
        };
        let (host, token) = match allocated {
            Ok(v) => v,
            Err(e) => {
                let _ = self.event_destroy(retire_event);
                return Err(e);
            }
        };
        // SAFETY: hipHostMalloc succeeded with a non-zero extent; initialize
        // once so the host slices handed out are ordinary initialized bytes.
        unsafe { std::ptr::write_bytes(host.cast::<u8>(), 0, size) };
        // From here the buffer owns the allocation: any refusal below drops
        // it in the Retired phase, which frees the host memory and the charge.
        let mut buffer = MappedHostBuffer {
            host,
            // SAFETY: placeholder null view of extent 0, replaced below
            // before the buffer escapes; never dereferenced.
            device_view: unsafe { DeviceBuffer::from_raw(ptr::null_mut(), 0) },
            size,
            ledger: MappedLedger::new(),
            retire_event,
            device,
            free,
            token,
            get_device: self.fn_get_device,
            set_device: self.fn_set_device,
            synchronize: self.fn_device_synchronize,
            event_synchronize: self.fn_event_synchronize,
            event_destroy: self.fn_event_destroy,
            _lib: Arc::clone(&self._lib),
        };
        let alias = self
            .host_get_device_pointer(host, 0)
            .and_then(|alias| checked_alias(alias, size))?;
        // SAFETY: `alias` is the non-null device address of the live `size`-byte
        // mapped allocation owned by `buffer`; the view is Borrowed (never
        // freed) and `buffer` outlives every use of it.
        buffer.device_view = unsafe { DeviceBuffer::from_raw(alias, size) };
        Ok(buffer)
    }

    pub fn mem_get_allocation_granularity(
        &self,
        prop: &HipMemAllocationProp,
        option: u32,
    ) -> HipResult<usize> {
        let func = self.missing_vmm_symbol(
            "hipMemGetAllocationGranularity",
            self.fn_mem_get_allocation_granularity,
        )?;
        let mut granularity: usize = 0;
        // SAFETY: resolved VMM/host HIP symbol; pointer/handle args meet the method's
        // # Safety or documented preconditions; out-params are stack locals.
        let code = unsafe {
            func(
                &mut granularity,
                prop as *const HipMemAllocationProp,
                option,
            )
        };
        self.check(code, "hipMemGetAllocationGranularity")?;
        Ok(granularity)
    }

    /// Query the physical placement of a VMM allocation handle. Callers read
    /// `prop.location.type_`: a host-located page reports
    /// `HIP_MEM_LOCATION_TYPE_HOST` (system RAM accessed over PCIe) — exactly what
    /// partial offload requires. Fail-closed: if this ROCm build lacks the symbol,
    /// returns an error instead of pretending the pages left VRAM.
    ///
    /// `handle` is an opaque HIP allocation handle; Rust never dereferences it.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn mem_get_handle_properties(
        &self,
        handle: HipMemGenericAllocationHandle,
    ) -> HipResult<HipMemAllocationProp> {
        let func = self.missing_vmm_symbol(
            "hipMemGetAllocationPropertiesFromHandle",
            self.fn_mem_get_handle_properties,
        )?;
        // SAFETY: HipMemAllocationProp is a POD FFI struct; zeroed bytes are a
        // valid representation overwritten by the subsequent HIP query.
        let mut prop: HipMemAllocationProp = unsafe { std::mem::zeroed() };
        // SAFETY: resolved VMM/host HIP symbol; pointer/handle args meet the method's
        // # Safety or documented preconditions; out-params are stack locals.
        let code = unsafe { func(&mut prop as *mut HipMemAllocationProp, handle) };
        self.check(code, "hipMemGetAllocationPropertiesFromHandle")?;
        Ok(prop)
    }
    pub fn mem_address_reserve(&self, size: usize, alignment: usize) -> HipResult<*mut c_void> {
        self.plan_deny("hipMemAddressReserve")?;
        let func = self.missing_vmm_symbol("hipMemAddressReserve", self.fn_mem_address_reserve)?;
        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: resolved VMM/host HIP symbol; pointer/handle args meet the method's
        // # Safety or documented preconditions; out-params are stack locals.
        let code = unsafe { func(&mut ptr, size, alignment, ptr::null_mut(), 0) };
        self.check(code, "hipMemAddressReserve")?;
        Ok(ptr)
    }

    pub fn mem_create(
        &self,
        size: usize,
        prop: &HipMemAllocationProp,
    ) -> HipResult<HipMemGenericAllocationHandle> {
        let handle = self.mem_create_inner(size, prop)?;
        // Only device-placed backing counts against device capacity.
        if prop.location.type_ == HIP_MEM_LOCATION_TYPE_DEVICE {
            self.own_vmm_handles
                .lock()
                .unwrap()
                .insert(handle as usize, size);
            self.own_device_bytes.fetch_add(size, Ordering::Relaxed);
        }
        Ok(handle)
    }

    fn mem_create_inner(
        &self,
        size: usize,
        prop: &HipMemAllocationProp,
    ) -> HipResult<HipMemGenericAllocationHandle> {
        self.plan_deny("hipMemCreate")?;
        let func = self.missing_vmm_symbol("hipMemCreate", self.fn_mem_create)?;
        let raw = || {
            let mut handle: HipMemGenericAllocationHandle = ptr::null_mut();
            // SAFETY: resolved VMM/host HIP symbol; pointer/handle args meet the method's
            // # Safety or documented preconditions; out-params are stack locals.
            let code = unsafe { func(&mut handle, size, prop as *const HipMemAllocationProp, 0) };
            self.check(code, "hipMemCreate")?;
            Ok(handle)
        };
        let Some(budget) = crate::emu_budget::active() else {
            return raw();
        };
        // Charged at physical backing creation, by placement.
        let class = crate::emu_budget::vmm_class(prop.location.type_)?;
        crate::emu_budget::charged_alloc_keyed(
            &budget,
            &crate::emu_budget::VMM_TOKENS,
            class,
            size,
            |handle: &HipMemGenericAllocationHandle| *handle as usize,
            raw,
        )
    }

    /// # Safety
    /// `ptr..ptr+size` must be a valid, currently unmapped VMM reservation and
    /// `handle` must be a live allocation handle covering at least `size`.
    pub unsafe fn mem_map(
        &self,
        ptr: *mut c_void,
        size: usize,
        handle: HipMemGenericAllocationHandle,
    ) -> HipResult<()> {
        self.plan_deny("hipMemMap")?;
        let func = self.missing_vmm_symbol("hipMemMap", self.fn_mem_map)?;
        // SAFETY: resolved VMM/host HIP symbol; pointer/handle args meet the method's
        // # Safety or documented preconditions; out-params are stack locals.
        let code = unsafe { func(ptr, size, 0, handle, 0) };
        self.check(code, "hipMemMap")?;
        crate::registry::record(
            ptr as usize,
            size,
            crate::registry::AllocationKind::VmmChunk,
        );
        Ok(())
    }

    /// # Safety
    /// `ptr..ptr+size` must describe a currently mapped VMM range.
    pub unsafe fn mem_unmap(&self, ptr: *mut c_void, size: usize) -> HipResult<()> {
        self.plan_deny("hipMemUnmap")?;
        let func = self.missing_vmm_symbol("hipMemUnmap", self.fn_mem_unmap)?;
        // SAFETY: resolved VMM/host HIP symbol; pointer/handle args meet the method's
        // # Safety or documented preconditions; out-params are stack locals.
        let code = unsafe { func(ptr, size) };
        self.check(code, "hipMemUnmap")?;
        crate::registry::forget(ptr as usize);
        Ok(())
    }

    /// # Safety
    /// `ptr..ptr+size` must describe a mapped VMM range and every descriptor
    /// must identify a device allowed by the backing allocation.
    pub unsafe fn mem_set_access(
        &self,
        ptr: *mut c_void,
        size: usize,
        descs: &[HipMemAccessDesc],
    ) -> HipResult<()> {
        self.plan_deny("hipMemSetAccess")?;
        let func = self.missing_vmm_symbol("hipMemSetAccess", self.fn_mem_set_access)?;
        // SAFETY: resolved VMM/host HIP symbol; pointer/handle args meet the method's
        // # Safety or documented preconditions; out-params are stack locals.
        let code = unsafe { func(ptr, size, descs.as_ptr(), descs.len()) };
        self.check(code, "hipMemSetAccess")
    }

    /// # Safety
    /// `handle` must be live, owned by the caller, and no mapped range may
    /// still reference it.
    pub unsafe fn mem_release(&self, handle: HipMemGenericAllocationHandle) -> HipResult<()> {
        // SAFETY: forwarded caller contract.
        unsafe { self.mem_release_inner(handle) }?;
        if let Some(size) = self
            .own_vmm_handles
            .lock()
            .unwrap()
            .remove(&(handle as usize))
        {
            self.own_device_bytes
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                    Some(v.saturating_sub(size))
                })
                .ok();
        }
        Ok(())
    }

    unsafe fn mem_release_inner(&self, handle: HipMemGenericAllocationHandle) -> HipResult<()> {
        self.plan_deny("hipMemRelease")?;
        let func = self.missing_vmm_symbol("hipMemRelease", self.fn_mem_release)?;
        let raw = || {
            // SAFETY: resolved VMM/host HIP symbol; pointer/handle args meet the method's
            // # Safety or documented preconditions; out-params are stack locals.
            let code = unsafe { func(handle) };
            self.check(code, "hipMemRelease")
        };
        if crate::emu_budget::active().is_none() {
            return raw();
        }
        // One token per physical handle; released only on a successful release.
        crate::emu_budget::release_keyed::<crate::emu_budget::Process>(
            &crate::emu_budget::VMM_TOKENS,
            handle as usize,
            raw,
        )
    }

    /// Export a live exportable (POSIX-FD) physical allocation handle as a new
    /// file descriptor. The FD is an independent reference to the allocation:
    /// the physical memory outlives `mem_release(handle)` until every FD and
    /// importer reference is gone. The exporting runtime must have created
    /// `handle` with `HipMemAllocationProp::device_pinned_posix_fd`.
    ///
    /// `handle` is an opaque HIP allocation handle; Rust never dereferences it.
    #[cfg(unix)]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn mem_export_posix(
        &self,
        handle: HipMemGenericAllocationHandle,
    ) -> HipResult<std::os::fd::OwnedFd> {
        use std::os::fd::FromRawFd;
        let func = self.missing_vmm_symbol(
            "hipMemExportToShareableHandle",
            self.fn_mem_export_to_shareable_handle,
        )?;
        // The runtime writes an integer FD through the output address.
        let mut fd: c_int = -1;
        // SAFETY: `fd` is a live stack c_int receiving the descriptor; the
        // handle is an opaque runtime handle; flags are zero per the HIP API.
        let code = unsafe {
            func(
                &mut fd as *mut c_int as *mut c_void,
                handle,
                HIP_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0,
            )
        };
        self.check(code, "hipMemExportToShareableHandle")?;
        if fd < 0 {
            return Err(HipError::new(
                0,
                "hipMemExportToShareableHandle returned success with an invalid file descriptor",
            ));
        }
        // SAFETY: on success the runtime transferred a fresh FD to this caller;
        // nothing else owns it.
        Ok(unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) })
    }

    /// Import a POSIX-FD shareable handle exported by another process (or
    /// this one). The FD is passed by value (`(void *)(intptr_t)fd`, the
    /// convention G0 verified on the installed runtime) and is NOT consumed:
    /// the caller keeps ownership and may close it after import. The returned
    /// handle is owned by the caller and must be released with `mem_release`
    /// after every mapping of it is unmapped.
    #[cfg(unix)]
    pub fn mem_import_posix(
        &self,
        fd: std::os::fd::BorrowedFd<'_>,
    ) -> HipResult<HipMemGenericAllocationHandle> {
        // Untyped: no byte/placement metadata to charge, so refuse under emu.
        crate::emu_budget::untyped_import_gate(crate::emu_budget::active().is_some())?;
        self.mem_import_posix_raw(fd)
    }

    /// Import with caller-supplied physical size: under emu the `Device`
    /// charge is reserved before the raw import, committed on success and kept
    /// with the handle until a successful `mem_release`. Off state is the
    /// plain raw import.
    #[cfg(unix)]
    pub(crate) fn mem_import_posix_charged(
        &self,
        fd: std::os::fd::BorrowedFd<'_>,
        physical_bytes: usize,
    ) -> HipResult<HipMemGenericAllocationHandle> {
        let Some(budget) = crate::emu_budget::active() else {
            return self.mem_import_posix_raw(fd);
        };
        crate::emu_budget::charged_alloc_keyed(
            &budget,
            &crate::emu_budget::VMM_TOKENS,
            crate::emu_budget::Class::Device,
            physical_bytes,
            |handle: &HipMemGenericAllocationHandle| *handle as usize,
            || self.mem_import_posix_raw(fd),
        )
    }

    #[cfg(unix)]
    fn mem_import_posix_raw(
        &self,
        fd: std::os::fd::BorrowedFd<'_>,
    ) -> HipResult<HipMemGenericAllocationHandle> {
        use std::os::fd::AsRawFd;
        let func = self.missing_vmm_symbol(
            "hipMemImportFromShareableHandle",
            self.fn_mem_import_from_shareable_handle,
        )?;
        let mut handle: HipMemGenericAllocationHandle = ptr::null_mut();
        // SAFETY: `handle` is a stack out-param; the FD is encoded by value as
        // the runtime expects and stays valid (borrowed) for the call.
        let code = unsafe {
            func(
                &mut handle,
                fd.as_raw_fd() as isize as *mut c_void,
                HIP_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            )
        };
        self.check(code, "hipMemImportFromShareableHandle")?;
        Ok(handle)
    }

    /// Query the access flags (`HIP_MEM_ACCESS_FLAGS_PROT_*`) `location` has
    /// on the mapped VMM address `ptr` (`hipMemGetAccess`). Optional symbol:
    /// errors when this runtime lacks it. `ptr` is an opaque GPU address.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn mem_get_access(&self, location: &HipMemLocation, ptr: *mut c_void) -> HipResult<u32> {
        let func = self.missing_vmm_symbol("hipMemGetAccess", self.fn_mem_get_access)?;
        let mut flags: u64 = 0;
        // SAFETY: `flags` and `location` are valid for the call; `ptr` is an
        // opaque GPU address the runtime validates.
        let code = unsafe { func(&mut flags, location as *const HipMemLocation, ptr) };
        self.check(code, "hipMemGetAccess")?;
        Ok(flags as u32)
    }

    /// Return the base and byte extent of the HIP allocation containing
    /// `device_ptr`. This is used by retained replay to conservatively treat
    /// distinct subviews of one allocation as aliasing resources.
    ///
    /// HIP treats `device_ptr` as an opaque GPU address; Rust never dereferences it.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn mem_get_address_range(
        &self,
        device_ptr: *mut c_void,
    ) -> HipResult<(*mut c_void, usize)> {
        let mut base = ptr::null_mut();
        let mut size = 0usize;
        // SAFETY: FFI out-params are stack locals with valid writable provenance;
        // HIP symbols were resolved at load. Returned pointers are opaque to Rust.
        let code = unsafe { (self.fn_mem_get_address_range)(&mut base, &mut size, device_ptr) };
        self.check(code, "hipMemGetAddressRange")?;
        if base.is_null() || size == 0 {
            return Err(HipError::new(
                0,
                "hipMemGetAddressRange returned an empty allocation",
            ));
        }
        Ok((base, size))
    }

    /// Copy host data into GPU buffer at a byte offset.
    pub fn memcpy_htod_offset(
        &self,
        dst: &DeviceBuffer,
        offset: usize,
        src: &[u8],
    ) -> HipResult<()> {
        self.plan_deny("hipMemcpy H2D offset")?;
        assert!(
            offset + src.len() <= dst.size,
            "offset ({}) + source ({}) exceeds device buffer ({})",
            offset,
            src.len(),
            dst.size
        );
        // SAFETY: byte offset was bounds-checked against the DeviceBuffer size;
        // `u8::add` stays in the same live allocation; no host deref of device memory.
        let dst_ptr = unsafe { (dst.ptr as *mut u8).add(offset) as *mut c_void };
        // SAFETY: pointers are live HIP allocations or host slices; sizes were
        // asserted to fit. Same-device/stream and host-buffer lifetime across
        // async copies are the caller's contract; Rust does not deref device ptrs.
        let code = unsafe {
            (self.fn_memcpy)(
                dst_ptr,
                src.as_ptr() as *const c_void,
                src.len(),
                MemcpyKind::HostToDevice as c_uint,
            )
        };
        memory_effects::htod();
        self.check(code, "hipMemcpy H2D offset")
    }

    /// Copy bytes between GPU buffers with offsets on both sides.
    pub fn memcpy_dtod_at(
        &self,
        dst: &DeviceBuffer,
        dst_offset: usize,
        src: &DeviceBuffer,
        src_offset: usize,
        size: usize,
    ) -> HipResult<()> {
        if let Some(r) = self.plan_copy_at(
            "hipMemcpy D2D at offset",
            dst,
            dst_offset,
            src,
            src_offset,
            size,
        ) {
            return r;
        }
        assert!(dst_offset + size <= dst.size);
        assert!(src_offset + size <= src.size);
        // SAFETY: byte offset was bounds-checked against the DeviceBuffer size;
        // `u8::add` stays in the same live allocation; no host deref of device memory.
        let dst_ptr = unsafe { (dst.ptr as *mut u8).add(dst_offset) as *mut c_void };
        // SAFETY: byte offset was bounds-checked against the DeviceBuffer size;
        // `u8::add` stays in the same live allocation; no host deref of device memory.
        let src_ptr = unsafe { (src.ptr as *const u8).add(src_offset) as *const c_void };
        let t = std::time::Instant::now();
        // SAFETY: pointers are live HIP allocations or host slices; sizes were
        // asserted to fit. Same-device/stream and host-buffer lifetime across
        // async copies are the caller's contract; Rust does not deref device ptrs.
        let code = unsafe {
            (self.fn_memcpy)(dst_ptr, src_ptr, size, MemcpyKind::DeviceToDevice as c_uint)
        };
        crate::ffi::launch_counters::memcpy_dtod::record(t.elapsed().as_nanos() as u64);
        memory_effects::dtod();
        self.check(code, "hipMemcpy D2D at offset")
    }

    /// Copy bytes from one GPU buffer at an offset to another GPU buffer.
    pub fn memcpy_dtod_offset(
        &self,
        dst: &DeviceBuffer,
        src: &DeviceBuffer,
        src_offset: usize,
        size: usize,
    ) -> HipResult<()> {
        if let Some(r) = self.plan_copy_at("hipMemcpy D2D offset", dst, 0, src, src_offset, size) {
            return r;
        }
        assert!(size <= dst.size, "size ({size}) exceeds dst ({})", dst.size);
        assert!(src_offset + size <= src.size, "src_offset+size exceeds src");
        // SAFETY: byte offset was bounds-checked against the DeviceBuffer size;
        // `u8::add` stays in the same live allocation; no host deref of device memory.
        let src_ptr = unsafe { (src.ptr as *const u8).add(src_offset) as *const c_void };
        let t = std::time::Instant::now();
        // SAFETY: pointers are live HIP allocations or host slices; sizes were
        // asserted to fit. Same-device/stream and host-buffer lifetime across
        // async copies are the caller's contract; Rust does not deref device ptrs.
        let code = unsafe {
            (self.fn_memcpy)(dst.ptr, src_ptr, size, MemcpyKind::DeviceToDevice as c_uint)
        };
        crate::ffi::launch_counters::memcpy_dtod::record(t.elapsed().as_nanos() as u64);
        memory_effects::dtod();
        self.check(code, "hipMemcpy D2D offset")
    }

    pub fn memcpy_htod(&self, dst: &DeviceBuffer, src: &[u8]) -> HipResult<()> {
        self.plan_deny("hipMemcpy H2D")?;
        if hip_fault_consume(0, "upload") {
            return Err(hip_fault_err("upload"));
        }
        assert!(
            src.len() <= dst.size,
            "source ({}) exceeds device buffer ({})",
            src.len(),
            dst.size
        );
        let t = std::time::Instant::now();
        // SAFETY: pointers are live HIP allocations or host slices; sizes were
        // asserted to fit. Same-device/stream and host-buffer lifetime across
        // async copies are the caller's contract; Rust does not deref device ptrs.
        let code = unsafe {
            (self.fn_memcpy)(
                dst.ptr,
                src.as_ptr() as *const c_void,
                src.len(),
                MemcpyKind::HostToDevice as c_uint,
            )
        };
        crate::ffi::launch_counters::memcpy_htod::record(t.elapsed().as_nanos() as u64);
        memory_effects::htod();
        self.check(code, "hipMemcpy H2D")
    }

    #[track_caller]
    pub fn memcpy_dtoh(&self, dst: &mut [u8], src: &DeviceBuffer) -> HipResult<()> {
        self.plan_deny("hipMemcpy D2H")?;
        assert!(
            dst.len() <= src.size,
            "destination ({}) exceeds device buffer ({})",
            dst.len(),
            src.size
        );
        let loc = std::panic::Location::caller();
        let t = std::time::Instant::now();
        // SAFETY: pointers are live HIP allocations or host slices; sizes were
        // asserted to fit. Same-device/stream and host-buffer lifetime across
        // async copies are the caller's contract; Rust does not deref device ptrs.
        let code = unsafe {
            (self.fn_memcpy)(
                dst.as_mut_ptr() as *mut c_void,
                src.ptr as *const c_void,
                dst.len(),
                MemcpyKind::DeviceToHost as c_uint,
            )
        };
        let elapsed = t.elapsed().as_nanos() as u64;
        crate::ffi::launch_counters::memcpy_dtoh::record_bytes(elapsed, dst.len() as u64);
        static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let dump = *DUMP.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_DTOH_DUMP")
                .ok()
                .as_deref()
                == Some("1")
        });
        if dump {
            eprintln!(
                "dtoh bytes={} us={} at {}:{}",
                dst.len(),
                elapsed / 1000,
                loc.file(),
                loc.line()
            );
        }
        memory_effects::dtoh();
        self.check(code, "hipMemcpy D2H")
    }

    /// Copy bytes from a GPU buffer at a given source offset to host.
    /// `dst.len()` bytes are copied starting from `src.ptr + src_offset`.
    #[track_caller]
    pub fn memcpy_dtoh_at(
        &self,
        dst: &mut [u8],
        src: &DeviceBuffer,
        src_offset: usize,
    ) -> HipResult<()> {
        self.plan_deny("hipMemcpy D2H at offset")?;
        assert!(
            src_offset + dst.len() <= src.size,
            "src_offset ({}) + dst len ({}) exceeds device buffer ({})",
            src_offset,
            dst.len(),
            src.size
        );
        // SAFETY: byte offset was bounds-checked against the DeviceBuffer size;
        // `u8::add` stays in the same live allocation; no host deref of device memory.
        let src_ptr = unsafe { (src.ptr as *const u8).add(src_offset) as *const c_void };
        let loc = std::panic::Location::caller();
        let t = std::time::Instant::now();
        // SAFETY: pointers are live HIP allocations or host slices; sizes were
        // asserted to fit. Same-device/stream and host-buffer lifetime across
        // async copies are the caller's contract; Rust does not deref device ptrs.
        let code = unsafe {
            (self.fn_memcpy)(
                dst.as_mut_ptr() as *mut c_void,
                src_ptr,
                dst.len(),
                MemcpyKind::DeviceToHost as c_uint,
            )
        };
        let elapsed = t.elapsed().as_nanos() as u64;
        crate::ffi::launch_counters::memcpy_dtoh::record_bytes(elapsed, dst.len() as u64);
        static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let dump = *DUMP.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_DTOH_DUMP")
                .ok()
                .as_deref()
                == Some("1")
        });
        if dump {
            eprintln!(
                "dtoh_at bytes={} us={} at {}:{}",
                dst.len(),
                elapsed / 1000,
                loc.file(),
                loc.line()
            );
        }
        memory_effects::dtoh();
        self.check(code, "hipMemcpy D2H at offset")
    }

    pub fn memcpy_dtod(
        &self,
        dst: &DeviceBuffer,
        src: &DeviceBuffer,
        size: usize,
    ) -> HipResult<()> {
        if let Some(r) = self.plan_copy_at("hipMemcpy D2D", dst, 0, src, 0, size) {
            return r;
        }
        assert!(size <= dst.size && size <= src.size);
        let t = std::time::Instant::now();
        // SAFETY: pointers are live HIP allocations or host slices; sizes were
        // asserted to fit. Same-device/stream and host-buffer lifetime across
        // async copies are the caller's contract; Rust does not deref device ptrs.
        let code = unsafe {
            (self.fn_memcpy)(
                dst.ptr,
                src.ptr as *const c_void,
                size,
                MemcpyKind::DeviceToDevice as c_uint,
            )
        };
        crate::ffi::launch_counters::memcpy_dtod::record(t.elapsed().as_nanos() as u64);
        memory_effects::dtod();
        self.check(code, "hipMemcpy D2D")
    }

    /// `hipMemcpyDtoD(dst, src, size)` on raw device addresses.
    ///
    /// # Safety
    /// `[src, src+size)` and `[dst, dst+size)` must lie in live device
    /// allocations and must not overlap.
    pub unsafe fn memcpy_dtod_raw(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
    ) -> HipResult<()> {
        if self.plan.active() {
            self.plan.copy_raw(dst as u64, src as u64, size as u64);
            return Ok(());
        }
        let f = self.fn_memcpy_dtod.ok_or_else(|| {
            HipError::new(0, "hipMemcpyDtoD is not exported by the loaded HIP runtime")
        })?;
        let t = std::time::Instant::now();
        // SAFETY: the caller guarantees both ranges lie in live, disjoint
        // device allocations; Rust does not deref device pointers.
        let code = unsafe { f(dst, src as *mut c_void, size) };
        crate::ffi::launch_counters::memcpy_dtod::record(t.elapsed().as_nanos() as u64);
        memory_effects::dtod();
        self.check(code, "hipMemcpyDtoD")
    }

    #[track_caller]
    pub fn memset(&self, buf: &DeviceBuffer, value: i32, size: usize) -> HipResult<()> {
        if self.plan.active() {
            return self.plan.memset_bytes(
                "hipMemset",
                buf.as_ptr() as u64,
                buf.size(),
                value,
                size,
            );
        }
        assert!(size <= buf.size);
        let loc = std::panic::Location::caller();
        let t = std::time::Instant::now();
        // SAFETY: pointers are live HIP allocations or host slices; sizes were
        // asserted to fit. Same-device/stream and host-buffer lifetime across
        // async copies are the caller's contract; Rust does not deref device ptrs.
        let code = unsafe { (self.fn_memset)(buf.ptr, value, size) };
        let elapsed = t.elapsed().as_nanos() as u64;
        crate::ffi::launch_counters::memset::record_bytes(elapsed, size as u64);
        static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let dump = *DUMP.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_MEMSET_DUMP")
                .ok()
                .as_deref()
                == Some("1")
        });
        if dump {
            eprintln!(
                "memset bytes={} us={} at {}:{}",
                size,
                elapsed / 1000,
                loc.file(),
                loc.line()
            );
        }
        memory_effects::memset();
        self.check(code, "hipMemset")
    }

    /// Fill `count` 32-bit words of `buf` with `value`, ordered on `stream`
    /// (the null stream when `None`) without blocking the host.
    pub fn memset_d32_async(
        &self,
        buf: &DeviceBuffer,
        value: i32,
        count: usize,
        stream: Option<&Stream>,
    ) -> HipResult<()> {
        if self.plan.active() {
            return self.plan.memset_words(
                "hipMemsetD32Async",
                buf.as_ptr() as u64,
                buf.size(),
                value,
                count,
            );
        }
        assert!(count * 4 <= buf.size);
        let stream_raw = stream.map_or(ptr::null_mut(), |s| s.0);
        // SAFETY: pointers are live HIP allocations or host slices; sizes were
        // asserted to fit. Same-device/stream and host-buffer lifetime across
        // async copies are the caller's contract; Rust does not deref device ptrs.
        let code = unsafe { (self.fn_memset_d32_async)(buf.ptr, value, count, stream_raw) };
        memory_effects::memset();
        self.check(code, "hipMemsetD32Async")
    }

    /// Async memset on a specific stream — does NOT block the host.
    /// Caller must ensure stream-ordering downstream work syncs correctly.
    #[track_caller]
    pub fn memset_async(
        &self,
        buf: &DeviceBuffer,
        value: i32,
        size: usize,
        stream: &Stream,
    ) -> HipResult<()> {
        if self.plan.active() {
            return self.plan.memset_bytes(
                "hipMemsetAsync",
                buf.as_ptr() as u64,
                buf.size(),
                value,
                size,
            );
        }
        assert!(size <= buf.size);
        let loc = std::panic::Location::caller();
        let t = std::time::Instant::now();
        // SAFETY: pointers are live HIP allocations or host slices; sizes were
        // asserted to fit. Same-device/stream and host-buffer lifetime across
        // async copies are the caller's contract; Rust does not deref device ptrs.
        let code = unsafe { (self.fn_memset_async)(buf.ptr, value, size, stream.0) };
        let elapsed = t.elapsed().as_nanos() as u64;
        crate::ffi::launch_counters::memset::record_bytes(elapsed, size as u64);
        static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let dump = *DUMP.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_MEMSET_DUMP")
                .ok()
                .as_deref()
                == Some("1")
        });
        if dump {
            eprintln!(
                "memset_async bytes={} us={} at {}:{}",
                size,
                elapsed / 1000,
                loc.file(),
                loc.line()
            );
        }
        memory_effects::memset();
        self.check(code, "hipMemsetAsync")
    }

    // ── Streams ─────────────────────────────────────────────────

    pub fn stream_create(&self) -> HipResult<Stream> {
        self.plan_deny("hipStreamCreate")?;
        let mut stream: HipStream = ptr::null_mut();
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_stream_create)(&mut stream) };
        self.check(code, "hipStreamCreate")?;
        Ok(Stream(stream))
    }

    /// A stream that does not synchronize with the legacy null stream
    /// (`hipStreamNonBlocking`).
    pub fn stream_create_non_blocking(&self) -> HipResult<Stream> {
        self.plan_deny("hipStreamCreateWithFlags")?;
        let mut stream: HipStream = ptr::null_mut();
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_stream_create_with_flags)(&mut stream, 0x1) };
        self.check(code, "hipStreamCreateWithFlags")?;
        Ok(Stream(stream))
    }

    pub fn stream_synchronize(&self, stream: &Stream) -> HipResult<()> {
        self.plan_deny("hipStreamSynchronize")?;
        if hip_fault_consume(2, "sync") {
            return Err(hip_fault_err("sync"));
        }
        let t = std::time::Instant::now();
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_stream_synchronize)(stream.0) };
        crate::ffi::launch_counters::stream_sync::record(t.elapsed().as_nanos() as u64);
        self.check(code, "hipStreamSynchronize")
    }

    /// Synchronize the legacy default (NULL) stream: the stream HIP uses when a
    /// caller passes no stream. Same fault injection and `stream_sync` counter as
    /// [`Self::stream_synchronize`]. It waits on that one stream only, never the
    /// whole device. `Stream` has no `Drop`, so the temporary NULL handle is never
    /// destroyed.
    pub fn stream_synchronize_default(&self) -> HipResult<()> {
        self.plan_deny("hipStreamSynchronize(default)")?;
        self.stream_synchronize(&Stream(ptr::null_mut()))
    }

    pub fn stream_destroy(&self, stream: Stream) -> HipResult<()> {
        self.plan_deny("hipStreamDestroy")?;
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_stream_destroy)(stream.0) };
        self.check(code, "hipStreamDestroy")
    }

    // ── Modules & Kernels ───────────────────────────────────────

    pub fn module_load(&self, path: &str) -> HipResult<Module> {
        self.plan_deny("hipModuleLoad")?;
        let c_path =
            CString::new(path).map_err(|_| HipError::new(1, "invalid path for module_load"))?;
        let mut module: HipModule = ptr::null_mut();
        // SAFETY: module/function handles are live HIP objects from earlier loads;
        // CString/image buffers outlive the call; launch arg validity is the
        // unsafe fn caller's contract when applicable.
        let code = unsafe { (self.fn_module_load)(&mut module, c_path.as_ptr()) };
        self.check(code, "hipModuleLoad")?;
        if module_load_audit_enabled() {
            eprintln!("HIPFIRE_MODULE_LOAD_AUDIT\tload\t{:p}\t{path}", module);
        }
        Ok(Module(module))
    }

    pub fn module_load_data(&self, image: &[u8]) -> HipResult<Module> {
        self.plan_deny("hipModuleLoadData")?;
        let mut module: HipModule = ptr::null_mut();
        let code =
            // SAFETY: module/function handles are live HIP objects from earlier loads;
            // CString/image buffers outlive the call; launch arg validity is the
            // unsafe fn caller's contract when applicable.
            unsafe { (self.fn_module_load_data)(&mut module, image.as_ptr() as *const c_void) };
        self.check(code, "hipModuleLoadData")?;
        if module_load_audit_enabled() {
            eprintln!(
                "HIPFIRE_MODULE_LOAD_AUDIT\tload_data\t{:p}\t{}\t{:016x}",
                module,
                image.len(),
                module_image_fnv64(image)
            );
        }
        Ok(Module(module))
    }

    pub fn module_get_function(&self, module: &Module, name: &str) -> HipResult<Function> {
        self.plan_deny("hipModuleGetFunction")?;
        let c_name = CString::new(name)
            .map_err(|_| HipError::new(1, "invalid kernel name for module_get_function"))?;
        let mut func: HipFunction = ptr::null_mut();
        // SAFETY: module/function handles are live HIP objects from earlier loads;
        // CString/image buffers outlive the call; launch arg validity is the
        // unsafe fn caller's contract when applicable.
        let code = unsafe { (self.fn_module_get_function)(&mut func, module.0, c_name.as_ptr()) };
        self.check(code, "hipModuleGetFunction")?;
        if module_load_audit_enabled() {
            eprintln!(
                "HIPFIRE_MODULE_LOAD_AUDIT\tget_function\t{:p}\t{name}",
                module.0
            );
        }
        Ok(Function(func))
    }

    /// Install the `gridDim.y`/`gridDim.z` ceiling for the architecture this
    /// runtime launches on ([`crate::launch_grid::grid_yz_limit_for_arch`]).
    /// Architectures without a recorded ceiling clear the guard.
    pub fn set_launch_grid_limit_for_arch(&self, arch: &str) {
        let limit = crate::launch_grid::grid_yz_limit_for_arch(arch).unwrap_or(u32::MAX);
        self.launch_grid_yz_limit.store(limit, Ordering::Relaxed);
    }

    /// The installed `gridDim.y`/`gridDim.z` ceiling, if any.
    pub fn launch_grid_yz_limit(&self) -> Option<u32> {
        match self.launch_grid_yz_limit.load(Ordering::Relaxed) {
            u32::MAX => None,
            limit => Some(limit),
        }
    }

    /// Fail-closed geometry check shared by the raw launch entry points and
    /// the dispatch record/capture funnels. Never clamps; see
    /// [`crate::launch_grid::check_launch_grid`].
    #[inline]
    pub fn validate_launch_grid(&self, grid: [u32; 3]) -> HipResult<()> {
        crate::launch_grid::check_launch_grid(grid, self.launch_grid_yz_limit())
    }

    /// Launch a kernel on the GPU.
    ///
    /// # Safety
    /// `params` must contain valid pointers to kernel arguments matching the kernel signature.
    pub unsafe fn launch_kernel(
        &self,
        func: &Function,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: Option<&Stream>,
        params: &mut [*mut c_void],
    ) -> HipResult<()> {
        if self.plan.active() {
            return Err(self.plan.refuse(&format!(
                "HIP API hipModuleLaunchKernel(kernelParams) function handle {:#x} reached in plan mode",
                func.0 as usize
            )));
        }
        if self.plan.witnessing() {
            // kernelParams form: the kernarg bytes are not recoverable here.
            self.plan
                .witness_launch(func.0 as usize, grid, block, shared_mem, &[]);
        }
        #[cfg(feature = "kernel-timing")]
        let mut timing = kernel_timing::LaunchTiming::new(self)?;
        self.validate_launch_grid(grid)?;
        if hip_fault_consume(1, "launch") {
            return Err(hip_fault_err("launch"));
        }
        let stream_raw = stream.map_or(ptr::null_mut(), |s| s.0);
        let t = std::time::Instant::now();
        #[cfg(feature = "kernel-timing")]
        timing.record_start(stream_raw)?;
        let code = (self.fn_module_launch_kernel)(
            func.0,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            shared_mem,
            stream_raw,
            params.as_mut_ptr(),
            ptr::null_mut(),
        );
        #[cfg(feature = "kernel-timing")]
        let timing_result = timing.record_stop(code);
        crate::ffi::launch_counters::record(t.elapsed().as_nanos() as u64);
        memory_effects::launch();
        #[cfg(feature = "kernel-timing")]
        timing_result?;
        self.check(code, "hipModuleLaunchKernel")
    }

    /// Launch a kernel using the `extra` path, passing a contiguous kernarg
    /// byte buffer instead of the traditional `void**` pointer-per-arg array.
    ///
    /// This path is REQUIRED for graph capture on gfx1100 / ROCm 6.3: when a
    /// launch is captured into a stream graph via `hipStreamBeginCapture`, the
    /// kernelParams path (`*mut *mut c_void`) only captures pointers, not the
    /// pointed-to values. By the time the graph is replayed, the stack frame
    /// that held those values is gone and the kernel reads garbage. The
    /// `extra` path, on the other hand, hands HIP a single blob pointer +
    /// size, and HIP copies the blob contents into the kernel node at capture
    /// time.
    ///
    /// The caller owns the `kernarg_blob` slice and is responsible for keeping
    /// it alive for the lifetime of any graph that captured this launch. For
    /// one-shot launches (no capture) the blob may be stack-local.
    ///
    /// Layout contract: `kernarg_blob` must be the kernel's full kernarg
    /// struct, laid out with natural alignment per field (matching the way
    /// hipcc emits the kernel's argument ABI). Total blob size is passed
    /// alongside the pointer via HIP_LAUNCH_PARAM_BUFFER_SIZE.
    ///
    /// # Safety
    /// `kernarg_blob` must have layout + size matching the kernel signature,
    /// and all contained pointers must be valid GPU addresses.
    pub unsafe fn launch_kernel_blob(
        &self,
        func: &Function,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: Option<&Stream>,
        kernarg_blob: &mut [u8],
    ) -> HipResult<()> {
        if self.plan.active() {
            self.validate_launch_grid(grid)?;
            self.plan
                .launch(func.0 as usize, grid, block, shared_mem, kernarg_blob);
            return Ok(());
        }
        if self.plan.witnessing() {
            self.plan
                .witness_launch(func.0 as usize, grid, block, shared_mem, kernarg_blob);
        }
        #[cfg(feature = "kernel-timing")]
        let mut timing = kernel_timing::LaunchTiming::new(self)?;
        self.validate_launch_grid(grid)?;
        if hip_fault_consume(1, "launch") {
            return Err(hip_fault_err("launch"));
        }
        // HIP `extra` mode sentinel constants (from hip_runtime.h):
        //   HIP_LAUNCH_PARAM_BUFFER_POINTER = 0x01
        //   HIP_LAUNCH_PARAM_BUFFER_SIZE    = 0x02
        //   HIP_LAUNCH_PARAM_END            = 0x03
        // The `extra` array alternates sentinel, value pointer, ..., END.
        let mut blob_size: usize = kernarg_blob.len();
        let blob_ptr: *mut c_void = kernarg_blob.as_mut_ptr() as *mut c_void;
        let size_ptr: *mut c_void = (&mut blob_size as *mut usize) as *mut c_void;
        let mut extra: [*mut c_void; 5] = [
            0x01 as *mut c_void, // HIP_LAUNCH_PARAM_BUFFER_POINTER
            blob_ptr,            // → persistent kernarg blob
            0x02 as *mut c_void, // HIP_LAUNCH_PARAM_BUFFER_SIZE
            size_ptr,            // → &blob_size (must live across the call)
            0x03 as *mut c_void, // HIP_LAUNCH_PARAM_END
        ];

        let stream_raw = stream.map_or(ptr::null_mut(), |s| s.0);
        let t = std::time::Instant::now();
        #[cfg(feature = "kernel-timing")]
        timing.record_start(stream_raw)?;
        let code = (self.fn_module_launch_kernel)(
            func.0,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            shared_mem,
            stream_raw,
            ptr::null_mut(), // kernelParams = null (we use extra)
            extra.as_mut_ptr(),
        );
        #[cfg(feature = "kernel-timing")]
        let timing_result = timing.record_stop(code);
        crate::ffi::launch_counters::record(t.elapsed().as_nanos() as u64);
        memory_effects::launch();
        #[cfg(feature = "kernel-timing")]
        timing_result?;
        self.check(code, "hipModuleLaunchKernel(extra blob)")
    }
    /// Max active blocks per multiprocessor for a loaded module function
    /// (oracle/occupancy probe; `dynamic_smem` = launch-time LDS bytes).
    pub fn occupancy_max_active_blocks(
        &self,
        func: &Function,
        block_size: u32,
        dynamic_smem: usize,
    ) -> HipResult<i32> {
        let Some(occ) = self.fn_module_occupancy_max_active_blocks else {
            return Err(HipError::new(
                0,
                "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor unavailable",
            ));
        };
        let mut n: c_int = 0;
        // SAFETY: occupancy symbol is present; func.0 is a live module function;
        // out-param `n` is a stack local.
        let code = unsafe {
            occ(
                &mut n as *mut c_int,
                func.0,
                block_size as c_int,
                dynamic_smem,
            )
        };
        self.check(code, "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor")?;
        Ok(n)
    }

    // ── Events ──────────────────────────────────────────────────

    pub fn event_create(&self) -> HipResult<Event> {
        self.plan_deny("hipEventCreate")?;
        let mut event: HipEvent = ptr::null_mut();
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_event_create)(&mut event) };
        self.check(code, "hipEventCreate")?;
        Ok(Event(event))
    }

    /// Create an event with HIP runtime flags.
    ///
    /// Dependency events should normally use [`HIP_EVENT_DISABLE_TIMING`]
    /// without `hipEventDisableSystemFence`. Cross-device producers may add
    /// [`HIP_EVENT_RELEASE_TO_SYSTEM`] to make the release scope explicit.
    pub fn event_create_with_flags(&self, flags: u32) -> HipResult<Event> {
        self.plan_deny("hipEventCreateWithFlags")?;
        let mut event: HipEvent = ptr::null_mut();
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_event_create_with_flags)(&mut event, flags) };
        self.check(code, "hipEventCreateWithFlags")?;
        Ok(Event(event))
    }

    pub fn event_record(&self, event: &Event, stream: Option<&Stream>) -> HipResult<()> {
        self.plan_deny("hipEventRecord")?;
        let stream_raw = stream.map_or(ptr::null_mut(), |s| s.0);
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_event_record)(event.0, stream_raw) };
        self.check(code, "hipEventRecord")
    }

    pub fn event_synchronize(&self, event: &Event) -> HipResult<()> {
        self.plan_deny("hipEventSynchronize")?;
        let t = std::time::Instant::now();
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_event_synchronize)(event.0) };
        crate::ffi::launch_counters::event_sync::record(t.elapsed().as_nanos() as u64);
        self.check(code, "hipEventSynchronize")
    }

    pub fn event_query(&self, event: &Event) -> HipResult<bool> {
        self.plan_deny("hipEventQuery")?;
        // Non-blocking by construction: hipSuccess = recorded and complete,
        // hipErrorNotReady (600) = work still outstanding, anything else is
        // a real failure (invalid handle, device lost) and propagates.
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_event_query)(event.0) };
        if code == HIP_SUCCESS {
            Ok(true)
        } else if code == HIP_ERROR_NOT_READY {
            Ok(false)
        } else {
            self.check(code, "hipEventQuery")?;
            // Unreachable: check errs on every nonzero code.
            Ok(false)
        }
    }

    pub fn event_elapsed_ms(&self, start: &Event, stop: &Event) -> HipResult<f32> {
        self.plan_deny("hipEventElapsedTime")?;
        let mut ms: f32 = 0.0;
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_event_elapsed_time)(&mut ms, start.0, stop.0) };
        self.check(code, "hipEventElapsedTime")?;
        Ok(ms)
    }

    pub fn event_destroy(&self, event: Event) -> HipResult<()> {
        self.plan_deny("hipEventDestroy")?;
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_event_destroy)(event.0) };
        self.check(code, "hipEventDestroy")
    }

    pub fn stream_wait_event(&self, stream: &Stream, event: &Event) -> HipResult<()> {
        self.plan_deny("hipStreamWaitEvent")?;
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_stream_wait_event)(stream.0, event.0, 0) };
        self.check(code, "hipStreamWaitEvent")
    }

    /// [`Self::stream_wait_event`] on an optional stream: `None` is the
    /// legacy/default (null) stream, the one `rdna_compute::Gpu` launches on
    /// while it has no active stream.
    pub fn stream_wait_event_on(&self, stream: Option<&Stream>, event: &Event) -> HipResult<()> {
        self.plan_deny("hipStreamWaitEvent")?;
        let stream_raw = stream.map_or(ptr::null_mut(), |s| s.0);
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_stream_wait_event)(stream_raw, event.0, 0) };
        self.check(code, "hipStreamWaitEvent")
    }

    // ── Error query ─────────────────────────────────────────────

    pub fn last_error(&self) -> u32 {
        // SAFETY: HIP runtime symbol is live from HipRuntime::load; out-params are
        // stack locals (or caller-provided device ids). No host pointer aliasing.
        let code = unsafe { (self.fn_get_last_error)() };
        crate::error::observe_raw_code(code);
        code
    }

    // ── Async memory ops ────────────────────────────────────────

    pub fn memcpy_htod_async(
        &self,
        dst: &DeviceBuffer,
        src: &[u8],
        stream: &Stream,
    ) -> HipResult<()> {
        self.plan_deny("hipMemcpyAsync H2D")?;
        if hip_fault_consume(0, "upload") {
            return Err(hip_fault_err("upload"));
        }
        assert!(src.len() <= dst.size);
        // SAFETY: pointers are live HIP allocations or host slices; sizes were
        // asserted to fit. Same-device/stream and host-buffer lifetime across
        // async copies are the caller's contract; Rust does not deref device ptrs.
        let code = unsafe {
            (self.fn_memcpy_async)(
                dst.ptr,
                src.as_ptr() as *const c_void,
                src.len(),
                MemcpyKind::HostToDevice as c_uint,
                stream.0,
            )
        };
        memory_effects::htod();
        self.check(code, "hipMemcpyAsync H2D")
    }

    /// Host-asynchronous H→D copy ordered on `stream` (`None` = the legacy
    /// stream, only when the caller has no stream). `src` must stay unchanged
    /// until the stream reaches the copy; a pinned `src` makes this a plain
    /// stream-ordered DMA.
    pub fn memcpy_htod_async_on(
        &self,
        dst: &DeviceBuffer,
        src: &[u8],
        stream: Option<&Stream>,
    ) -> HipResult<()> {
        match stream {
            Some(stream) => self.memcpy_htod_async(dst, src, stream),
            None => self.memcpy_htod_async_default(dst, src),
        }
    }

    /// Host-asynchronous H→D copy on the legacy/default stream: `src` must
    /// stay unchanged until the stream reaches the copy.
    pub fn memcpy_htod_async_default(&self, dst: &DeviceBuffer, src: &[u8]) -> HipResult<()> {
        self.plan_deny("hipMemcpyAsync H2D(default stream)")?;
        assert!(src.len() <= dst.size);
        // SAFETY: pointers are live HIP allocations or host slices; sizes were
        // asserted to fit. Same-device/stream and host-buffer lifetime across
        // async copies are the caller's contract; Rust does not deref device ptrs.
        let code = unsafe {
            (self.fn_memcpy_async)(
                dst.ptr,
                src.as_ptr() as *const c_void,
                src.len(),
                MemcpyKind::HostToDevice as c_uint,
                ptr::null_mut(),
            )
        };
        memory_effects::htod();
        self.check(code, "hipMemcpyAsync H2D default stream")
    }

    pub fn memcpy_dtoh_async(
        &self,
        dst: &mut [u8],
        src: &DeviceBuffer,
        stream: Option<&Stream>,
    ) -> HipResult<()> {
        self.plan_deny("hipMemcpyAsync D2H")?;
        assert!(dst.len() <= src.size);
        // SAFETY: pointers are live HIP allocations or host slices; sizes were
        // asserted to fit. Same-device/stream and host-buffer lifetime across
        // async copies are the caller's contract; Rust does not deref device ptrs.
        let code = unsafe {
            (self.fn_memcpy_async)(
                dst.as_mut_ptr() as *mut c_void,
                src.ptr as *const c_void,
                dst.len(),
                MemcpyKind::DeviceToHost as c_uint,
                stream.map_or(ptr::null_mut(), |s| s.0),
            )
        };
        memory_effects::dtoh();
        self.check(code, "hipMemcpyAsync D2H")
    }

    /// Async D→D copy with optional offsets on both sides. Ordered on
    /// `stream` and capturable by hipStreamBeginCapture — use this in
    /// place of sync `memcpy_dtod_at` wherever the copy needs to live
    /// inside a hipGraph.
    pub fn memcpy_dtod_async_at(
        &self,
        dst: &DeviceBuffer,
        dst_offset: usize,
        src: &DeviceBuffer,
        src_offset: usize,
        size: usize,
        stream: &Stream,
    ) -> HipResult<()> {
        if let Some(r) = self.plan_copy_at(
            "hipMemcpyAsync D2D offset",
            dst,
            dst_offset,
            src,
            src_offset,
            size,
        ) {
            return r;
        }
        assert!(dst_offset + size <= dst.size);
        assert!(src_offset + size <= src.size);
        // SAFETY: byte offset was bounds-checked against the DeviceBuffer size;
        // `u8::add` stays in the same live allocation; no host deref of device memory.
        let dst_ptr = unsafe { (dst.ptr as *mut u8).add(dst_offset) as *mut c_void };
        // SAFETY: byte offset was bounds-checked against the DeviceBuffer size;
        // `u8::add` stays in the same live allocation; no host deref of device memory.
        let src_ptr = unsafe { (src.ptr as *const u8).add(src_offset) as *const c_void };
        // SAFETY: pointers are live HIP allocations or host slices; sizes were
        // asserted to fit. Same-device/stream and host-buffer lifetime across
        // async copies are the caller's contract; Rust does not deref device ptrs.
        let code = unsafe {
            (self.fn_memcpy_async)(
                dst_ptr,
                src_ptr,
                size,
                MemcpyKind::DeviceToDevice as c_uint,
                stream.0,
            )
        };
        memory_effects::dtod();
        self.check(code, "hipMemcpyAsync D2D offset")
    }

    /// Async D→D copy with offsets on the legacy/default stream
    /// (`hipStream_t` null). Host-asynchronous and ordered on that stream;
    /// intended for non-capture hot paths where a sync `memcpy_dtod_at`
    /// would stall the host. Not suitable for hipGraph capture (use
    /// `memcpy_dtod_async_at` with an explicit stream instead).
    pub fn memcpy_dtod_async_default_at(
        &self,
        dst: &DeviceBuffer,
        dst_offset: usize,
        src: &DeviceBuffer,
        src_offset: usize,
        size: usize,
    ) -> HipResult<()> {
        if let Some(r) = self.plan_copy_at(
            "hipMemcpyAsync D2D offset default stream",
            dst,
            dst_offset,
            src,
            src_offset,
            size,
        ) {
            return r;
        }
        assert!(dst_offset + size <= dst.size);
        assert!(src_offset + size <= src.size);
        // SAFETY: byte offset was bounds-checked against the DeviceBuffer size;
        // `u8::add` stays in the same live allocation; no host deref of device memory.
        let dst_ptr = unsafe { (dst.ptr as *mut u8).add(dst_offset) as *mut c_void };
        // SAFETY: byte offset was bounds-checked against the DeviceBuffer size;
        // `u8::add` stays in the same live allocation; no host deref of device memory.
        let src_ptr = unsafe { (src.ptr as *const u8).add(src_offset) as *const c_void };
        // SAFETY: pointers are live HIP allocations or host slices; sizes were
        // asserted to fit. Same-device/stream and host-buffer lifetime across
        // async copies are the caller's contract; Rust does not deref device ptrs.
        let code = unsafe {
            (self.fn_memcpy_async)(
                dst_ptr,
                src_ptr,
                size,
                MemcpyKind::DeviceToDevice as c_uint,
                ptr::null_mut(),
            )
        };
        memory_effects::dtod();
        self.check(code, "hipMemcpyAsync D2D offset default stream")
    }

    /// Host-asynchronous pitched D→D copy (`hipMemcpy2DAsync`) of `height`
    /// rows of `width` bytes each, with a byte offset and a row pitch on
    /// each side.
    ///
    /// Units: `*_offset`, `*_pitch` and `width` are bytes; `height` is a row
    /// count. Row `r` copies `width` bytes from `src + src_offset +
    /// r * src_pitch` to `dst + dst_offset + r * dst_pitch`.
    ///
    /// Empty copies (`width == 0` or `height == 0`) return `Ok(())` before
    /// any pointer arithmetic, HIP call, counter or memory-effect update.
    ///
    /// Bounds: requires `width <= dst_pitch` and `width <= src_pitch`, and
    /// each side's last-row span `offset + (height - 1) * pitch + width`
    /// (not `height * pitch`) must fit in its buffer; overflow or violation
    /// panics before HIP is invoked.
    ///
    /// Async/lifetime: the copy is enqueued on `stream` (`None` = legacy/null
    /// stream) and may complete after this returns; both allocations must stay
    /// live until it completes, and the caller owns same-device, non-overlap
    /// and stream-ordering. This helper never synchronizes or allocates.
    /// hipGraph capture requires an explicit `Some(stream)`.
    #[allow(clippy::too_many_arguments)]
    pub fn memcpy_dtod_2d_async_at(
        &self,
        dst: &DeviceBuffer,
        dst_offset: usize,
        dst_pitch: usize,
        src: &DeviceBuffer,
        src_offset: usize,
        src_pitch: usize,
        width: usize,
        height: usize,
        stream: Option<&Stream>,
    ) -> HipResult<()> {
        self.plan_deny("hipMemcpy2DAsync D2D")?;
        if width == 0 || height == 0 {
            return Ok(());
        }
        assert!(
            width <= dst_pitch,
            "memcpy_dtod_2d: width exceeds dst pitch"
        );
        assert!(
            width <= src_pitch,
            "memcpy_dtod_2d: width exceeds src pitch"
        );
        let dst_end = (height - 1)
            .checked_mul(dst_pitch)
            .and_then(|v| v.checked_add(width))
            .and_then(|v| v.checked_add(dst_offset))
            .expect("memcpy_dtod_2d: dst span overflows usize");
        let src_end = (height - 1)
            .checked_mul(src_pitch)
            .and_then(|v| v.checked_add(width))
            .and_then(|v| v.checked_add(src_offset))
            .expect("memcpy_dtod_2d: src span overflows usize");
        assert!(
            dst_end <= dst.size,
            "memcpy_dtod_2d: dst span exceeds buffer"
        );
        assert!(
            src_end <= src.size,
            "memcpy_dtod_2d: src span exceeds buffer"
        );
        // SAFETY: byte offset was bounds-checked against the DeviceBuffer size;
        // `u8::add` stays in the same live allocation; no host deref of device memory.
        let dst_ptr = unsafe { (dst.ptr as *mut u8).add(dst_offset) as *mut c_void };
        // SAFETY: byte offset was bounds-checked against the DeviceBuffer size;
        // `u8::add` stays in the same live allocation; no host deref of device memory.
        let src_ptr = unsafe { (src.ptr as *const u8).add(src_offset) as *const c_void };
        let stream_raw = stream.map_or(ptr::null_mut(), |s| s.0);
        let t = std::time::Instant::now();
        // SAFETY: pointers are live HIP allocations; the full strided span of
        // both sides (last row ends at offset + (height-1)*pitch + width) was
        // asserted to fit, and width <= both pitches. Same-device/stream and
        // allocation lifetime until completion are the caller's contract; Rust
        // does not deref device ptrs.
        let code = unsafe {
            (self.fn_memcpy_2d_async)(
                dst_ptr,
                dst_pitch,
                src_ptr,
                src_pitch,
                width,
                height,
                MemcpyKind::DeviceToDevice as c_uint,
                stream_raw,
            )
        };
        crate::ffi::launch_counters::memcpy_dtod::record(t.elapsed().as_nanos() as u64);
        memory_effects::dtod();
        self.check(code, "hipMemcpy2DAsync D2D at offset")
    }

    // ── Graph capture & replay ──────────────────────────────────

    /// Begin capturing all operations on `stream` into a graph.
    /// mode=0 is hipStreamCaptureModeGlobal.
    pub fn stream_begin_capture(&self, stream: &Stream, mode: u32) -> HipResult<()> {
        self.plan_deny("hipStreamBeginCapture")?;
        #[cfg(feature = "kernel-timing")]
        kernel_timing::reject_active(self, "graph capture unsupported by kernel timing")?;
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_stream_begin_capture)(stream.0, mode as c_uint) };
        self.check(code, "hipStreamBeginCapture")
    }

    /// End capture on `stream`, returning the captured graph.
    pub fn stream_end_capture(&self, stream: &Stream) -> HipResult<Graph> {
        self.plan_deny("hipStreamEndCapture")?;
        let mut graph: HipGraph = ptr::null_mut();
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_stream_end_capture)(stream.0, &mut graph) };
        self.check(code, "hipStreamEndCapture")?;
        Ok(Graph(graph))
    }

    /// Whether `stream` is inside a `hipStreamBeginCapture` window. An
    /// invalidated capture still counts: work issued to it is not executed.
    pub fn stream_is_capturing(&self, stream: &Stream) -> HipResult<bool> {
        self.plan_deny("hipStreamIsCapturing")?;
        let mut status: c_uint = 0;
        // SAFETY: `stream` is a live opaque HIP handle owned by the wrapper;
        // `status` is a stack out-param.
        let code = unsafe { (self.fn_stream_is_capturing)(stream.0, &mut status) };
        self.check(code, "hipStreamIsCapturing")?;
        // hipStreamCaptureStatusNone = 0; Active = 1, Invalidated = 2.
        Ok(status != 0)
    }

    /// Instantiate an executable graph from a captured graph.
    pub fn graph_instantiate(&self, graph: &Graph) -> HipResult<GraphExec> {
        self.plan_deny("hipGraphInstantiate")?;
        let mut exec: HipGraphExec = ptr::null_mut();
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe {
            (self.fn_graph_instantiate)(&mut exec, graph.0, ptr::null_mut(), ptr::null_mut(), 0)
        };
        self.check(code, "hipGraphInstantiate")?;
        Ok(GraphExec(exec))
    }

    /// Launch an executable graph on `stream`.
    pub fn graph_launch(&self, exec: &GraphExec, stream: &Stream) -> HipResult<()> {
        self.plan_deny("hipGraphLaunch")?;
        #[cfg(feature = "kernel-timing")]
        kernel_timing::reject_active(self, "graph replay unsupported by kernel timing")?;
        let t = std::time::Instant::now();
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_graph_launch)(exec.0, stream.0) };
        crate::ffi::launch_counters::graph_launch::record(t.elapsed().as_nanos() as u64);
        self.check(code, "hipGraphLaunch")
    }

    pub fn graph_exec_destroy(&self, exec: GraphExec) -> HipResult<()> {
        self.plan_deny("hipGraphExecDestroy")?;
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_graph_exec_destroy)(exec.0) };
        self.check(code, "hipGraphExecDestroy")
    }

    pub fn graph_destroy(&self, graph: Graph) -> HipResult<()> {
        self.plan_deny("hipGraphDestroy")?;
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_graph_destroy)(graph.0) };
        self.check(code, "hipGraphDestroy")
    }

    /// Write a 32-bit value to a device address on the stream.
    /// The write is ordered with respect to other operations on the stream.
    /// Graph-safe: can be used before hipGraphLaunch to update device state
    /// that captured kernels will read (e.g., position buffers).
    pub fn stream_write_value32(
        &self,
        stream: &Stream,
        ptr: &DeviceBuffer,
        value: u32,
        flags: u32,
    ) -> HipResult<()> {
        self.plan_deny("hipStreamWriteValue32")?;
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_stream_write_value32)(stream.0, ptr.as_ptr(), value, flags) };
        self.check(code, "hipStreamWriteValue32")
    }

    /// Whether `device` reports `hipDeviceAttributeCanUseStreamWaitValue`.
    ///
    /// Symbol presence alone is not capability: a runtime may export
    /// `hipStreamWaitValue32` on a device whose queues cannot execute it. The
    /// answer is cached per device (a query error is not cached).
    pub fn can_use_stream_wait_value(&self, device: i32) -> HipResult<bool> {
        let slot = usize::try_from(device)
            .ok()
            .and_then(|d| self.stream_wait_caps.get(d));
        if let Some(slot) = slot {
            match slot.load(Ordering::Relaxed) {
                STREAM_WAIT_CAP_NO => return Ok(false),
                STREAM_WAIT_CAP_YES => return Ok(true),
                _ => {}
            }
        }
        let capable = if self.fn_stream_wait_value32.is_none() {
            false
        } else {
            self.get_device_attribute(HIP_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE, device)? != 0
        };
        if let Some(slot) = slot {
            let cached = if capable {
                STREAM_WAIT_CAP_YES
            } else {
                STREAM_WAIT_CAP_NO
            };
            slot.store(cached, Ordering::Relaxed);
        }
        Ok(capable)
    }

    /// Wait until a 32-bit signal-memory value satisfies the requested condition.
    /// Operations submitted later to `stream` remain blocked until it does.
    ///
    /// Refused unless the current device reports
    /// `hipDeviceAttributeCanUseStreamWaitValue` ([`Self::can_use_stream_wait_value`]).
    pub fn stream_wait_value32(
        &self,
        stream: &Stream,
        ptr: &DeviceBuffer,
        value: u32,
        flags: u32,
        mask: u32,
    ) -> HipResult<()> {
        self.plan_deny("hipStreamWaitValue32")?;
        let Some(wait_value32) = self.fn_stream_wait_value32 else {
            return Err(HipError::new(
                0,
                "hipStreamWaitValue32 unavailable; stream signal waits are unsupported",
            ));
        };
        let device = self.current_device()?;
        if !self.can_use_stream_wait_value(device)? {
            return Err(HipError::new(
                0,
                "hipDeviceAttributeCanUseStreamWaitValue is 0; stream signal waits are unsupported",
            ));
        }
        // SAFETY: ptr is a live device signal/buffer address from DeviceBuffer; stream
        // is a live handle. Value wait/write races are ordered by the stream.
        let code = unsafe { wait_value32(stream.0, ptr.as_ptr(), value, flags, mask) };
        self.check(code, "hipStreamWaitValue32")
    }

    /// Allocate one [`SignalWord`] (8 bytes of `hipMallocSignalMemory`),
    /// initialized to 0. Refused unless the current device can use stream
    /// value operations.
    pub fn signal_word(&self) -> HipResult<SignalWord> {
        let device = self.current_device()?;
        if !self.can_use_stream_wait_value(device)? {
            return Err(HipError::new(
                0,
                "hipDeviceAttributeCanUseStreamWaitValue is 0; signal words are unsupported",
            ));
        }
        let owner = self.malloc_signal(SIGNAL_WORD_BYTES)?;
        let host = owner.as_ptr().cast::<u32>();
        if host.is_null() || !(host as usize).is_multiple_of(std::mem::align_of::<u32>()) {
            let _ = self.free(owner);
            return Err(mapped_err(
                "signal memory returned a null or misaligned host pointer",
            ));
        }
        // SAFETY: the Borrowed view never outlives `owner`, which the word owns.
        let view = unsafe { owner.alias() };
        let word = SignalWord {
            owner: Some(owner),
            view,
            host,
            device,
            free: self.fn_free,
            get_device: self.fn_get_device,
            set_device: self.fn_set_device,
            synchronize: self.fn_device_synchronize,
            _lib: Arc::clone(&self._lib),
        };
        word.store(0);
        Ok(word)
    }

    /// Enqueue `hipStreamWriteValue32(word, generation)` on `stream`: a
    /// GPU-ordered release of everything queued before it (`None` = the default
    /// stream). `value` must be a generation, never [`SIGNAL_ABORT`].
    pub fn stream_write_signal(
        &self,
        stream: Option<&Stream>,
        word: &SignalWord,
        value: u32,
    ) -> HipResult<()> {
        self.plan_deny("hipStreamWriteSignal")?;
        if !SignalGen::is_generation(value) {
            return Err(mapped_err("signal write value is not a generation"));
        }
        let raw = Stream(stream.map_or(ptr::null_mut(), |s| s.0));
        self.stream_write_value32(
            &raw,
            word.device_buffer(),
            value,
            HIP_STREAM_WRITE_VALUE_DEFAULT,
        )
    }

    /// Enqueue `hipStreamWaitValue32(word >= generation)` on `stream`. Later
    /// work on the stream stays blocked until the word reaches `generation` or
    /// holds [`SIGNAL_ABORT`]. Every error path of a caller that queued this
    /// wait must store the abort value before any synchronize.
    pub fn stream_wait_signal_gte(
        &self,
        stream: Option<&Stream>,
        word: &SignalWord,
        generation: u32,
    ) -> HipResult<()> {
        self.plan_deny("hipStreamWaitSignal")?;
        if !SignalGen::is_generation(generation) {
            return Err(mapped_err("signal wait value is not a generation"));
        }
        let raw = Stream(stream.map_or(ptr::null_mut(), |s| s.0));
        self.stream_wait_value32(
            &raw,
            word.device_buffer(),
            generation,
            HIP_STREAM_WAIT_VALUE_GTE,
            u32::MAX,
        )
    }

    pub fn device_synchronize(&self) -> HipResult<()> {
        self.plan_deny("hipDeviceSynchronize")?;
        if hip_fault_consume(2, "sync") {
            return Err(hip_fault_err("sync"));
        }
        let t = std::time::Instant::now();
        // SAFETY: HIP runtime symbol is live from HipRuntime::load; out-params are
        // stack locals (or caller-provided device ids). No host pointer aliasing.
        let code = unsafe { (self.fn_device_synchronize)() };
        crate::ffi::launch_counters::device_sync::record(t.elapsed().as_nanos() as u64);
        self.check(code, "hipDeviceSynchronize")
    }

    /// Get GPU architecture string (e.g., "gfx1010", "gfx1030", "gfx1100").
    /// Allocates a large buffer for hipDeviceProp_t, reads gcnArchName from offset 0.
    pub fn get_arch(&self, device_id: i32) -> HipResult<String> {
        let mut buf = vec![0u8; 1024]; // hipDeviceProp_t varies by ROCm version, 1024 is safe
                                       // SAFETY: HIP runtime symbol is live from HipRuntime::load; out-params are
                                       // stack locals (or caller-provided device ids). No host pointer aliasing.
        let code = unsafe { (self.fn_get_device_properties)(buf.as_mut_ptr(), device_id as c_int) };
        self.check(code, "hipGetDeviceProperties")?;
        // gcnArchName is a null-terminated C string at the start of the struct
        // Actually it's at a fixed offset. On ROCm 5/6, gcnArchName is at offset 500+.
        // Safer: search for "gfx" in the buffer.
        let s = String::from_utf8_lossy(&buf);
        if let Some(pos) = s.find("gfx") {
            let arch_str = &s[pos..];
            let end = arch_str
                .find(|c: char| c == '\0' || c == ':' || c == ' ')
                .unwrap_or(arch_str.len());
            Ok(arch_str[..end].to_string())
        } else {
            // Fallback: read as null-terminated string from known offsets
            // gcnArchName is typically at offset 0 in older ROCm or at a named field
            // SAFETY: on success HIP wrote a NUL-terminated C string into the local
            // buffer; the pointer addresses that buffer only for this read.
            let cstr = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr() as *const c_char) };
            let name = cstr.to_string_lossy().to_string();
            if name.starts_with("gfx") {
                let end = name.find(':').unwrap_or(name.len());
                Ok(name[..end].to_string())
            } else {
                Ok("unknown".to_string())
            }
        }
    }

    /// Query a HIP device attribute by enum ID. See `hipDeviceAttribute_t` in
    /// `hip_runtime_api.h` for valid IDs. Used by the profiler to read CU count
    /// when sysfs/KFD is unavailable (Windows, restricted containers).
    pub fn get_device_attribute(&self, attr_id: i32, device_id: i32) -> HipResult<i32> {
        let mut value: c_int = 0;
        // SAFETY: HIP runtime symbol is live from HipRuntime::load; out-params are
        // stack locals (or caller-provided device ids). No host pointer aliasing.
        let code = unsafe {
            (self.fn_get_device_attribute)(&mut value, attr_id as c_int, device_id as c_int)
        };
        self.check(code, "hipDeviceGetAttribute")?;
        Ok(value as i32)
    }

    /// Get VRAM info: (free_bytes, total_bytes).
    ///
    /// RAW hardware query (`hipMemGetInfo`): never capped by an emu profile.
    /// The application-visible, capped query is `Gpu::device_mem_info` in
    /// rdna-compute.
    pub fn get_vram_info(&self) -> HipResult<(usize, usize)> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        // SAFETY: FFI out-params are stack locals with valid writable provenance;
        // HIP symbols were resolved at load. Returned pointers are opaque to Rust.
        let code = unsafe { (self.fn_mem_get_info)(&mut free, &mut total) };
        self.check(code, "hipMemGetInfo")?;
        Ok((free, total))
    }
}

// ── Handle wrappers ─────────────────────────────────────────────

/// Owned, initialized `hipHostMallocPortable` storage. Async users must finish
/// all copies before accessing or dropping it. On an unrecoverable drain error,
/// retain/leak the owner rather than freeing memory still used by the device.
pub struct PinnedHostBuffer {
    ptr: *mut c_void,
    size: usize,
    free: unsafe extern "C" fn(*mut c_void) -> u32,
    /// Emu `HostPinnedGtt` charge; `None` when emu is off. Released only after
    /// `free` succeeds in `Drop`; a failed free keeps the charge.
    token: Option<crate::emu_budget::Token>,
    _lib: Arc<Library>,
}

impl PinnedHostBuffer {
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: this owner holds size initialized bytes until Drop. The
        // mutable borrow excludes other host access; GPU quiescence is the
        // caller's async-copy contract.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.cast(), self.size) }
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        let Some(token) = self.token.take() else {
            // SAFETY: unique hipHostMalloc owner; its library is still live. The
            // caller must have drained async copies before dropping the owner.
            let code = unsafe { (self.free)(self.ptr) };
            crate::error::observe_raw_code(code);
            return;
        };
        // SAFETY: as above.
        let code = unsafe { (self.free)(self.ptr) };
        crate::error::observe_raw_code(code);
        let _ = crate::emu_budget::charged_free::<crate::emu_budget::Process>(Some(token), || {
            if code == HIP_SUCCESS {
                Ok(())
            } else {
                Err(HipError::new(code, "hipHostFree"))
            }
        });
    }
}

// ── Pinned transfer storage and input readiness ─────────────────

static NEXT_TRANSFER_ID: AtomicU64 = AtomicU64::new(1);

/// Process-wide enqueue order across every transfer buffer. Each enqueue (and
/// each readiness arm) draws the next value; the cross-buffer retirement proof
/// compares them.
static NEXT_TRANSFER_SEQ: AtomicU64 = AtomicU64::new(1);

/// What proves an [`InputReady`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadyKind {
    /// An event recorded on the buffer's stream after its queued downloads.
    Event,
    /// A `hipStreamWriteValue32` of this generation to a [`SignalWord`].
    Word(u32),
}

/// Proof obligation of one armed readiness point of a [`PinnedTransferBuffer`].
///
/// Minted by [`PinnedTransferBuffer::arm_ready`] /
/// [`PinnedTransferBuffer::arm_ready_word`] after the downloads are queued;
/// only the buffer that minted it, while the arm is still current, can complete
/// it. A completed token proves host readiness of that buffer and, through
/// [`PinnedTransferBuffer::retire_behind`], of earlier same-stream buffers.
/// Not `Clone`/`Copy`. It is a different type from [`MappedPublished`]: the
/// two cannot be exchanged.
///
/// ```compile_fail
/// fn takes_published(_: hip_bridge::MappedPublished) {}
/// fn forge(ready: hip_bridge::InputReady) {
///     takes_published(ready)
/// }
/// ```
///
/// ```compile_fail
/// fn takes_ready(_: hip_bridge::InputReady) {}
/// fn forge(published: hip_bridge::MappedPublished) {
///     takes_ready(published)
/// }
/// ```
#[must_use = "an armed readiness point must be completed before host access"]
#[derive(Debug, PartialEq, Eq)]
pub struct InputReady {
    buffer: u64,
    arm: u64,
    kind: ReadyKind,
    stream: usize,
    seq: u64,
    complete: bool,
}

impl InputReady {
    /// Whether the proof has been completed (by a query, a word or a blocking
    /// fallback) on the buffer that minted it.
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// The process-wide sequence number drawn when this readiness was armed.
    pub fn sequence(&self) -> u64 {
        self.seq
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Retire {
    Retired,
    /// The proof does not cover this buffer: complete it separately.
    NotCovered,
}

/// Pure, HIP-free ledger of one [`PinnedTransferBuffer`]: pending accesses,
/// the stream and sequence of the last enqueue, and the armed readiness.
#[derive(Debug)]
struct TransferLedger {
    id: u64,
    pending: bool,
    stream: usize,
    last_seq: u64,
    armed: Option<(u64, ReadyKind)>,
    next_arm: u64,
}

impl TransferLedger {
    fn new() -> Self {
        Self {
            id: NEXT_TRANSFER_ID.fetch_add(1, Ordering::Relaxed),
            pending: false,
            stream: 0,
            last_seq: 0,
            armed: None,
            next_arm: 1,
        }
    }

    /// A stream change while accesses are pending needs a device drain first.
    fn needs_drain(&self, raw: usize) -> bool {
        self.pending && self.stream != raw
    }

    /// Every pending access is retired.
    fn drained(&mut self) {
        self.pending = false;
        self.armed = None;
    }

    /// Mark before submission: even a failed API call must not expose storage
    /// while an earlier copy might still be using it. A new enqueue
    /// invalidates the armed readiness, which no longer covers it.
    fn enqueued(&mut self, raw: usize) {
        self.enqueued_at(raw, NEXT_TRANSFER_SEQ.fetch_add(1, Ordering::Relaxed));
    }

    fn enqueued_at(&mut self, raw: usize, seq: u64) {
        self.stream = raw;
        self.pending = true;
        self.last_seq = seq;
        self.armed = None;
    }

    fn arm(&mut self, kind: ReadyKind) -> Result<InputReady, &'static str> {
        self.arm_at(kind, NEXT_TRANSFER_SEQ.fetch_add(1, Ordering::Relaxed))
    }

    fn arm_at(&mut self, kind: ReadyKind, seq: u64) -> Result<InputReady, &'static str> {
        if !self.pending {
            return Err("no pending transfer to arm readiness for");
        }
        let arm = self.next_arm;
        self.next_arm += 1;
        self.armed = Some((arm, kind));
        Ok(InputReady {
            buffer: self.id,
            arm,
            kind,
            stream: self.stream,
            seq,
            complete: false,
        })
    }

    fn check(&self, token: &InputReady) -> Result<(), &'static str> {
        if token.buffer != self.id {
            return Err("readiness token belongs to a different transfer buffer");
        }
        if token.complete {
            return Err("readiness token was already completed");
        }
        match self.armed {
            Some((arm, kind)) if arm == token.arm && kind == token.kind => Ok(()),
            Some(_) => Err("readiness token is stale: the buffer was re-armed"),
            None => {
                Err("readiness token no longer covers the buffer (later enqueue or completion)")
            }
        }
    }

    /// Whether a word observation proves a `Word` token.
    fn observe_word(&self, token: &InputReady, observed: u32) -> Result<bool, &'static str> {
        self.check(token)?;
        match token.kind {
            ReadyKind::Word(want) => Ok(observed >= want),
            ReadyKind::Event => Err("an event readiness token is not proved by a signal word"),
        }
    }

    fn complete(&mut self, token: &mut InputReady) -> Result<(), &'static str> {
        self.check(token)?;
        self.drained();
        token.complete = true;
        Ok(())
    }

    /// A completed proof retires this buffer only when its last enqueue was on
    /// the proof's stream with a smaller sequence: same-stream FIFO then puts
    /// that access before the proof's marker.
    fn retire_behind(&mut self, proof: &InputReady) -> Result<Retire, &'static str> {
        if !proof.complete {
            return Err("readiness proof is not complete");
        }
        if !self.pending {
            return Ok(Retire::Retired);
        }
        if self.stream == proof.stream && self.last_seq < proof.seq {
            self.drained();
            Ok(Retire::Retired)
        } else {
            Ok(Retire::NotCovered)
        }
    }
}

/// Sole owner of one `hipHostMalloc`'d allocation, shared by `Arc` between the registry that
/// allocated it and any stream-held lease.  The LAST holder's `Drop` runs the original
/// `hipHostFree` exactly once (releasing the pointer-keyed emu pinned-host charge only if
/// the free succeeds; a failed free keeps the charge and leaks, never frees twice).  It holds
/// the free function and an `Arc<Library>` — never the [`HipRuntime`] — so it stays freeable
/// after the runtime that adopted it is gone.  Created only by
/// [`HipRuntime::adopt_host_allocation`].  Async users must have drained every GPU access
/// before the last reference drops.
pub struct HostAllocOwner {
    ptr: *mut c_void,
    free: unsafe extern "C" fn(*mut c_void) -> u32,
    /// `None` only for the CPU tests' fake free function.
    _lib: Option<Arc<Library>>,
}

// SAFETY: the owner is the unique owner of an opaque host allocation that Rust never
// dereferences; it holds a plain function pointer and an `Arc<Library>` (Send + Sync).
unsafe impl Send for HostAllocOwner {}
unsafe impl Sync for HostAllocOwner {}

impl HostAllocOwner {
    /// The host pointer `hipHostFree` will receive.
    pub fn host_ptr(&self) -> *mut c_void {
        self.ptr
    }

    #[cfg(test)]
    fn for_test(ptr: *mut c_void, free: unsafe extern "C" fn(*mut c_void) -> u32) -> Self {
        Self {
            ptr,
            free,
            _lib: None,
        }
    }
}

impl Drop for HostAllocOwner {
    fn drop(&mut self) {
        let raw = || {
            // SAFETY: the unique owner of the allocation (see `adopt_host_allocation`); its
            // library is still loaded (`_lib`). The caller drained async GPU access.
            let code = unsafe { (self.free)(self.ptr) };
            crate::error::observe_raw_code(code);
            if code == HIP_SUCCESS {
                Ok(())
            } else {
                Err(HipError::new(code, "hipHostFree"))
            }
        };
        // Same release rule as `HipRuntime::host_free`: the keyed charge is released only
        // if the free succeeds.
        let _ = if crate::emu_budget::active().is_none() {
            raw()
        } else {
            crate::emu_budget::release_keyed::<crate::emu_budget::Process>(
                &crate::emu_budget::HOST_TOKENS,
                self.ptr as usize,
                raw,
            )
        };
    }
}

/// Pinned transfer storage whose outstanding GPU accesses survive a CPU call.
///
/// Host access is available only after `complete` (or a completed readiness
/// token) drains the queued downloads. Uploads and following downloads on the
/// same stream reuse the allocation without an intervening host wait. Stream
/// changes and teardown must retire the old accesses first; a failed teardown
/// retains the allocation.
///
/// Event readiness: after the queued downloads, [`Self::arm_ready`] records an
/// event on the buffer's stream and returns an [`InputReady`];
/// [`Self::try_complete_ready`] polls it without a stream synchronization. A
/// completed token also retires other buffers whose last enqueue was on the
/// same stream earlier ([`Self::retire_behind`]).
pub struct PinnedTransferBuffer {
    storage: Option<PinnedHostBuffer>,
    device: i32,
    stream: HipStream,
    ledger: TransferLedger,
    ready_event: Option<Event>,
    get_device: unsafe extern "C" fn(*mut c_int) -> u32,
    set_device: unsafe extern "C" fn(c_int) -> u32,
    synchronize: unsafe extern "C" fn() -> u32,
    event_destroy: unsafe extern "C" fn(HipEvent) -> u32,
}

// SAFETY: this is the unique owner of its pinned allocation. Moving it does
// not move the allocation, expose host aliases, or change stream ordering.
// Submission, completion and host access all require exclusive `&mut self`;
// the caller must bind the recorded device before HIP operations. Drop binds
// that device itself, drains pending accesses and restores the prior binding,
// retaining storage if retirement fails. HIP stream and event handles are not
// thread-affine. Deliberately NOT Sync: concurrent access is not supported.
unsafe impl Send for PinnedTransferBuffer {}

const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<PinnedTransferBuffer>();
};

impl PinnedTransferBuffer {
    pub fn size(&self) -> usize {
        self.storage.as_ref().expect("live transfer storage").size()
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    /// Whether queued accesses are not yet proved complete.
    pub fn is_pending(&self) -> bool {
        self.ledger.pending
    }

    /// The runtime must be bound to this buffer's device, and the stream and
    /// device allocation must remain live until their queued work completes.
    pub fn download_async(
        &mut self,
        hip: &HipRuntime,
        src: &DeviceBuffer,
        offset: usize,
        size: usize,
        stream: Option<&Stream>,
    ) -> HipResult<()> {
        self.check_span(offset, size)?;
        if size > src.size() {
            return Err(HipError::new(0, "pinned download exceeds device buffer"));
        }
        self.begin(hip, stream)?;
        let storage = self.storage.as_ref().expect("live transfer storage");
        // SAFETY: checked allocation extent; no host reference exists while
        // pending. Same-stream copies order prior upload reads before these
        // writes, and the owner drains before host reuse/free.
        let code = unsafe {
            (hip.fn_memcpy_async)(
                storage.ptr.cast::<u8>().add(offset).cast(),
                src.ptr,
                size,
                MemcpyKind::DeviceToHost as c_uint,
                self.stream,
            )
        };
        memory_effects::dtoh();
        hip.check(code, "hipMemcpyAsync pinned D2H")
    }

    /// Enqueue a return from owned storage, not a caller's temporary slice.
    /// The runtime, stream and destination obey `download_async`'s contract.
    pub fn upload_async(
        &mut self,
        hip: &HipRuntime,
        dst: &DeviceBuffer,
        offset: usize,
        size: usize,
        stream: Option<&Stream>,
    ) -> HipResult<()> {
        self.check_span(offset, size)?;
        if size > dst.size() {
            return Err(HipError::new(0, "pinned upload exceeds device buffer"));
        }
        if hip_fault_consume(0, "upload") {
            return Err(hip_fault_err("upload"));
        }
        self.begin(hip, stream)?;
        let storage = self.storage.as_ref().expect("live transfer storage");
        // SAFETY: checked source extent, and no host access is allowed until
        // complete drains the stream. The pinned owner survives this enqueue.
        let code = unsafe {
            (hip.fn_memcpy_async)(
                dst.ptr,
                storage.ptr.cast::<u8>().add(offset).cast(),
                size,
                MemcpyKind::HostToDevice as c_uint,
                self.stream,
            )
        };
        memory_effects::htod();
        hip.check(code, "hipMemcpyAsync pinned H2D")
    }

    /// Complete the caller's existing D2H readiness boundary.
    pub fn complete(&mut self, hip: &HipRuntime) -> HipResult<()> {
        if self.ledger.pending {
            self.sync_stream(hip)?;
            self.ledger.drained();
        }
        Ok(())
    }

    fn sync_stream(&self, hip: &HipRuntime) -> HipResult<()> {
        if self.stream.is_null() {
            hip.stream_synchronize_default()
        } else {
            hip.stream_synchronize(&Stream(self.stream))
        }
    }

    /// Record an event on this buffer's stream after its queued transfers and
    /// return the token that [`Self::try_complete_ready`] polls. The event is
    /// created on first use and reused. A failed record returns no token and
    /// leaves the accesses pending.
    pub fn arm_ready(&mut self, hip: &HipRuntime) -> HipResult<InputReady> {
        if !self.ledger.pending {
            return Err(mapped_err("no pending transfer to arm readiness for"));
        }
        if self.ready_event.is_none() {
            self.ready_event = Some(hip.event_create_with_flags(HIP_EVENT_DISABLE_TIMING)?);
        }
        let event = self
            .ready_event
            .as_ref()
            .expect("ready event was just created");
        hip.event_record(event, Some(&Stream(self.stream)))?;
        self.ledger.arm(ReadyKind::Event).map_err(mapped_err)
    }

    /// Enqueue `hipStreamWriteValue32(word, value)` on this buffer's stream
    /// after its queued transfers and return the token that
    /// [`Self::try_complete_ready_word`] checks. `value` must be a generation.
    pub fn arm_ready_word(
        &mut self,
        hip: &HipRuntime,
        word: &SignalWord,
        value: u32,
    ) -> HipResult<InputReady> {
        if !self.ledger.pending {
            return Err(mapped_err("no pending transfer to arm readiness for"));
        }
        hip.stream_write_signal(Some(&Stream(self.stream)), word, value)?;
        self.ledger.arm(ReadyKind::Word(value)).map_err(mapped_err)
    }

    /// Non-blocking completion of an event token: `Ok(true)` once the event
    /// completed (the accesses are then retired and `ready` is a completed
    /// proof), `Ok(false)` while outstanding. A foreign, stale or wrong-kind
    /// token and any query error leave the accesses pending.
    pub fn try_complete_ready(
        &mut self,
        hip: &HipRuntime,
        ready: &mut InputReady,
    ) -> HipResult<bool> {
        self.ledger.check(ready).map_err(mapped_err)?;
        if ready.kind != ReadyKind::Event {
            return Err(mapped_err(
                "a signal-word readiness token is not an event token",
            ));
        }
        let event = self
            .ready_event
            .as_ref()
            .ok_or_else(|| mapped_err("event readiness without a recorded event"))?;
        if hip.event_query(event)? {
            self.ledger.complete(ready).map_err(mapped_err)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Non-blocking completion of a signal-word token: an acquire load of
    /// `word` at or above the armed generation completes it. An observed
    /// [`SIGNAL_ABORT`] also compares `>=` but is refused: the ready word is
    /// never aborted, so seeing the reserved value means the word is dead.
    pub fn try_complete_ready_word(
        &mut self,
        ready: &mut InputReady,
        word: &SignalWord,
    ) -> HipResult<bool> {
        let observed = word.load_acquire();
        if observed == SIGNAL_ABORT {
            return Err(mapped_err(
                "input ready word holds the reserved abort value",
            ));
        }
        if self
            .ledger
            .observe_word(ready, observed)
            .map_err(mapped_err)?
        {
            self.ledger.complete(ready).map_err(mapped_err)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Blocking fallback for an armed token: synchronize the stream, then mark
    /// the token complete. A foreign or stale token is refused before any wait.
    pub fn complete_ready(&mut self, hip: &HipRuntime, ready: &mut InputReady) -> HipResult<()> {
        self.ledger.check(ready).map_err(mapped_err)?;
        self.sync_stream(hip)?;
        self.ledger.complete(ready).map_err(mapped_err)
    }

    /// Retire this buffer's pending accesses using another buffer's completed
    /// proof: `Ok(true)` when they are retired (including when none were
    /// pending), `Ok(false)` when the proof does not cover this buffer (a
    /// different stream, or an enqueue not earlier than the proof) and the
    /// caller must [`Self::complete`] it. An incomplete proof is an error.
    pub fn retire_behind(&mut self, proof: &InputReady) -> HipResult<bool> {
        Ok(self.ledger.retire_behind(proof).map_err(mapped_err)? == Retire::Retired)
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        assert!(
            !self.ledger.pending,
            "host access before pinned transfer completion"
        );
        self.storage
            .as_mut()
            .expect("live transfer storage")
            .as_bytes_mut()
    }

    fn check_span(&self, offset: usize, size: usize) -> HipResult<()> {
        if offset.checked_add(size).is_none_or(|end| end > self.size()) {
            return Err(HipError::new(0, "pinned transfer extent exceeds storage"));
        }
        Ok(())
    }

    fn begin(&mut self, hip: &HipRuntime, stream: Option<&Stream>) -> HipResult<()> {
        let raw = stream.map_or(ptr::null_mut(), |s| s.0);
        if self.ledger.needs_drain(raw as usize) {
            // The old stream may have been destroyed already. Drain the owning
            // device, not a potentially stale handle, on this rare transition.
            hip.device_synchronize()?;
            self.ledger.drained();
        }
        self.stream = raw;
        self.ledger.enqueued(raw as usize);
        Ok(())
    }
}

impl Drop for PinnedTransferBuffer {
    fn drop(&mut self) {
        let event = self.ready_event.take();
        let pending = self.ledger.pending;
        if !pending && event.is_none() {
            return;
        }
        // Bind the owning device before draining; restore the caller's device
        // so cached thread binding remains valid outside this destructor.
        let drained = call_on_device(self.get_device, self.set_device, self.device, || {
            let mut code = HIP_SUCCESS;
            if pending {
                // SAFETY: the pinned owner keeps the function-pointer library live.
                code = unsafe { (self.synchronize)() };
            }
            if code == HIP_SUCCESS {
                if let Some(event) = event.as_ref() {
                    // SAFETY: the live event handle is owned by this buffer; the
                    // stream was drained above, so the record is complete.
                    let destroyed = unsafe { (self.event_destroy)(event.0) };
                    crate::error::observe_raw_code(destroyed);
                }
            }
            code
        });
        if !drained && pending {
            // Retain both allocation and library rather than freeing memory
            // that the device may still access after a failed drain.
            std::mem::forget(self.storage.take());
        }
    }
}

// ── Mapped host result buffer ───────────────────────────────────

/// Lifecycle phase of a [`MappedHostBuffer`].
///
/// `Retired → CpuWriting → Published → GpuReading → Retired`. Host bytes are
/// writable only in `CpuWriting`, which is reachable only from `Retired`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MappedPhase {
    /// No GPU consumer can be reading; the host may begin a write.
    Retired,
    /// The host holds the unique [`MappedWrite`].
    CpuWriting,
    /// The host finished writing; the consumer has not been recorded yet.
    Published,
    /// A consumer was enqueued; the host must not write until retirement.
    GpuReading,
}

static NEXT_MAPPED_ID: AtomicU64 = AtomicU64::new(1);

/// Proof that one generation of a [`MappedHostBuffer`] was published.
///
/// Carries the buffer identity and generation; consumed by
/// [`MappedHostBuffer::mark_gpu_reading`]. Not `Clone`/`Copy`, so one
/// publication can be marked in flight at most once.
#[must_use = "a published generation must be passed to mark_gpu_reading"]
#[derive(Debug, PartialEq, Eq)]
pub struct MappedPublished {
    buffer: u64,
    generation: u64,
}

impl MappedPublished {
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// How a quiescence request must be satisfied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MappedQuiesce {
    /// Nothing can be in flight.
    Done,
    /// Wait on (or query) the recorded retirement event.
    Event,
    /// No event covers the consumer; only a device synchronize proves it.
    Device,
}

/// Pure, HIP-free phase ledger embedded in [`MappedHostBuffer`].
#[derive(Debug)]
struct MappedLedger {
    id: u64,
    phase: MappedPhase,
    generation: u64,
    /// `GpuReading` entered without a recorded retirement event (the record
    /// call failed). Retirable only by device synchronization.
    unfenced: bool,
}

impl MappedLedger {
    fn new() -> Self {
        Self {
            id: NEXT_MAPPED_ID.fetch_add(1, Ordering::Relaxed),
            phase: MappedPhase::Retired,
            generation: 0,
            unfenced: false,
        }
    }

    fn begin_write(&mut self) -> Result<(), &'static str> {
        if self.phase != MappedPhase::Retired {
            return Err("mapped buffer is not retired; cannot begin a host write");
        }
        if self.generation == u64::MAX {
            return Err("mapped buffer generation counter exhausted");
        }
        self.phase = MappedPhase::CpuWriting;
        Ok(())
    }

    /// An unpublished write returns to `Retired`; the generation is unchanged.
    fn abandon_write(&mut self) {
        if self.phase == MappedPhase::CpuWriting {
            self.phase = MappedPhase::Retired;
        }
    }

    /// `begin_write` guarantees `generation < u64::MAX` while `CpuWriting`.
    fn publish(&mut self) -> Result<MappedPublished, &'static str> {
        if self.phase != MappedPhase::CpuWriting {
            return Err("mapped buffer is not being written; cannot publish");
        }
        self.generation += 1;
        self.phase = MappedPhase::Published;
        Ok(MappedPublished {
            buffer: self.id,
            generation: self.generation,
        })
    }

    fn check_token(&self, token: &MappedPublished) -> Result<(), &'static str> {
        if token.buffer != self.id {
            return Err("published token belongs to a different mapped buffer");
        }
        if self.phase != MappedPhase::Published {
            return Err("mapped buffer is not in the Published phase");
        }
        if token.generation != self.generation {
            return Err("published token is stale (generation mismatch)");
        }
        Ok(())
    }

    fn enter_gpu_reading(
        &mut self,
        token: &MappedPublished,
        fenced: bool,
    ) -> Result<(), &'static str> {
        self.check_token(token)?;
        self.phase = MappedPhase::GpuReading;
        self.unfenced = !fenced;
        Ok(())
    }

    fn retire_plan(&self) -> Result<MappedQuiesce, &'static str> {
        match self.phase {
            MappedPhase::Retired => Ok(MappedQuiesce::Done),
            MappedPhase::GpuReading if self.unfenced => Ok(MappedQuiesce::Device),
            MappedPhase::GpuReading => Ok(MappedQuiesce::Event),
            MappedPhase::CpuWriting => Err("mapped buffer is being written; nothing to retire"),
            MappedPhase::Published => {
                Err("mapped buffer was published but never marked GpuReading; refusing to retire")
            }
        }
    }

    /// Drop must assume any non-`Retired` phase may have a consumer in flight.
    fn drop_plan(&self) -> MappedQuiesce {
        match self.phase {
            MappedPhase::Retired => MappedQuiesce::Done,
            MappedPhase::GpuReading if !self.unfenced => MappedQuiesce::Event,
            _ => MappedQuiesce::Device,
        }
    }

    fn complete_retirement(&mut self) {
        debug_assert_eq!(self.phase, MappedPhase::GpuReading);
        self.phase = MappedPhase::Retired;
        self.unfenced = false;
    }
}

/// Reject extents that cannot back a Rust byte slice or a HIP allocation.
fn check_mapped_extent(size: usize) -> HipResult<()> {
    if size == 0 {
        return Err(HipError::new(
            0,
            "mapped_host_buffer requires non-zero size",
        ));
    }
    if size > isize::MAX as usize {
        return Err(HipError::new(
            0,
            "mapped_host_buffer size exceeds isize::MAX",
        ));
    }
    Ok(())
}

/// A mapped allocation is usable only with a non-null device alias. Never fall
/// back to the host pointer: the GPU is not guaranteed to dereference it.
fn checked_alias(alias: *mut c_void, size: usize) -> HipResult<*mut c_void> {
    if alias.is_null() {
        return Err(HipError::new(
            0,
            &format!("hipHostGetDevicePointer returned a null alias for {size} mapped bytes"),
        ));
    }
    Ok(alias)
}

/// `f32` view legality: a whole number of elements at an aligned address.
fn check_f32_extent(host_addr: usize, size: usize) -> HipResult<()> {
    if !size.is_multiple_of(4) {
        return Err(HipError::new(
            0,
            &format!("mapped buffer of {size} bytes is not a whole number of f32"),
        ));
    }
    if !host_addr.is_multiple_of(std::mem::align_of::<f32>()) {
        return Err(mapped_err("mapped host pointer is not f32-aligned"));
    }
    Ok(())
}

fn mapped_err(msg: &'static str) -> HipError {
    HipError::new(0, msg)
}

/// Host-written, GPU-read result storage: one
/// `hipHostMalloc(Mapped | Coherent)` allocation plus its device alias.
///
/// The allocation is owned here exactly once; [`MappedHostBuffer::device_buffer`]
/// is a non-owning (`Borrowed`) view that must never be freed or registered
/// as a second owner. Host bytes are reachable only through
/// [`MappedHostBuffer::begin_write`] from the `Retired` phase. After the
/// consumer is enqueued, [`MappedHostBuffer::mark_gpu_reading`] records a
/// retirement event; the host may write again only after
/// [`MappedHostBuffer::retire`] / [`MappedHostBuffer::try_retire`] prove that
/// event complete. Drop retires before freeing and retains the allocation if
/// retirement fails.
///
/// The runtime used by `mark_gpu_reading`/`retire`/`try_retire` must be bound
/// to [`MappedHostBuffer::device`], and the stream must outlive its work.
pub struct MappedHostBuffer {
    host: *mut c_void,
    device_view: DeviceBuffer,
    size: usize,
    ledger: MappedLedger,
    retire_event: Event,
    device: i32,
    free: unsafe extern "C" fn(*mut c_void) -> u32,
    /// Emu `HostPinned` charge; `None` when emu is off. Released only after
    /// `free` succeeds in `Drop`; a failed free keeps the charge.
    token: Option<crate::emu_budget::Token>,
    get_device: unsafe extern "C" fn(*mut c_int) -> u32,
    set_device: unsafe extern "C" fn(c_int) -> u32,
    synchronize: unsafe extern "C" fn() -> u32,
    event_synchronize: unsafe extern "C" fn(HipEvent) -> u32,
    event_destroy: unsafe extern "C" fn(HipEvent) -> u32,
    _lib: Arc<Library>,
}

// SAFETY: this is the unique owner of its mapped host allocation; moving it
// does not move the allocation. Host access requires `&mut self` in the
// Retired phase, and every GPU-facing operation requires exclusive
// `&mut self`; the caller must bind the recorded device before HIP calls.
// Drop binds that device itself, quiesces the consumer and restores the prior
// binding, retaining the allocation and library if quiescence fails. The
// non-owning device view is only a pointer value. Deliberately NOT Sync.
unsafe impl Send for MappedHostBuffer {}

const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<MappedHostBuffer>();
};

/// Unique host writer for one generation. Dropping it unpublished returns the
/// buffer to `Retired` without advancing the generation.
pub struct MappedWrite<'a> {
    buf: &'a mut MappedHostBuffer,
    published: bool,
}

impl MappedWrite<'_> {
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the buffer holds `size` initialized bytes until Drop, the
        // ledger is CpuWriting (no consumer in flight), and `&mut self` plus
        // the buffer's `&mut` borrow exclude other host access.
        unsafe { std::slice::from_raw_parts_mut(self.buf.host.cast::<u8>(), self.buf.size) }
    }

    /// The whole buffer as `f32`; refuses when the size is not a multiple of 4.
    pub fn f32_mut(&mut self) -> HipResult<&mut [f32]> {
        let size = self.buf.size;
        check_f32_extent(self.buf.host as usize, size)?;
        // SAFETY: extent is a checked multiple of 4, the pointer is checked
        // aligned, bytes are initialized, and exclusivity is as `bytes_mut`.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.buf.host.cast::<f32>(), size / 4) })
    }

    /// Finish the write and advance the generation.
    ///
    /// Issues `fence(SeqCst)`: this orders the host stores for other *host*
    /// threads only. It is NOT a proof that the GPU observes the bytes; GPU
    /// visibility is established only by an oracle on real hardware.
    pub fn publish(mut self) -> MappedPublished {
        std::sync::atomic::fence(Ordering::SeqCst);
        let token = self
            .buf
            .ledger
            .publish()
            .expect("MappedWrite exists only in the CpuWriting phase");
        self.published = true;
        token
    }
}

impl Drop for MappedWrite<'_> {
    fn drop(&mut self) {
        if !self.published {
            self.buf.ledger.abandon_write();
        }
    }
}

impl MappedHostBuffer {
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    pub fn phase(&self) -> MappedPhase {
        self.ledger.phase
    }

    /// Number of completed publications (0 until the first `publish`).
    pub fn generation(&self) -> u64 {
        self.ledger.generation
    }

    /// Non-owning device alias. It is `Borrowed`: never free it, never
    /// register it as an owner, and do not let it outlive this buffer.
    pub fn device_buffer(&self) -> &DeviceBuffer {
        &self.device_view
    }

    /// The whole host mapping as bytes.
    ///
    /// # Safety
    /// The caller guarantees every GPU write to this buffer completed (the
    /// writing stream was synchronized) and no GPU write is in flight. Reading
    /// does not change the ledger phase.
    pub unsafe fn host_bytes(&self) -> &[u8] {
        // SAFETY: `host` maps `size` bytes until Drop; the caller upholds the
        // no-concurrent-GPU-write contract above.
        unsafe { std::slice::from_raw_parts(self.host.cast::<u8>(), self.size) }
    }

    /// Start a host write. Only legal from `Retired`.
    pub fn begin_write(&mut self) -> HipResult<MappedWrite<'_>> {
        self.ledger.begin_write().map_err(mapped_err)?;
        Ok(MappedWrite {
            buf: self,
            published: false,
        })
    }

    /// Record that the consumer of `token`'s generation was just enqueued on
    /// `stream` (`None` = the default stream). Call it right after the launch.
    ///
    /// Refuses a foreign or stale token without changing the phase. If the
    /// event record fails the consumer may still be in flight, so the phase
    /// becomes `GpuReading` *without* an event: [`Self::try_retire`] then
    /// never succeeds and [`Self::retire`] uses a device synchronize.
    pub fn mark_gpu_reading(
        &mut self,
        hip: &HipRuntime,
        token: MappedPublished,
        stream: Option<&Stream>,
    ) -> HipResult<()> {
        self.ledger.check_token(&token).map_err(mapped_err)?;
        let recorded = hip.event_record(&self.retire_event, stream);
        self.ledger
            .enter_gpu_reading(&token, recorded.is_ok())
            .map_err(mapped_err)?;
        recorded
    }

    /// Block until the recorded consumer finished, then return to `Retired`.
    /// No-op when already `Retired`; refuses `Published`/`CpuWriting`. On error
    /// the phase is unchanged.
    pub fn retire(&mut self, hip: &HipRuntime) -> HipResult<()> {
        match self.ledger.retire_plan().map_err(mapped_err)? {
            MappedQuiesce::Done => return Ok(()),
            MappedQuiesce::Event => hip.event_synchronize(&self.retire_event)?,
            MappedQuiesce::Device => hip.device_synchronize()?,
        }
        self.ledger.complete_retirement();
        Ok(())
    }

    /// Non-blocking [`Self::retire`]: `Ok(true)` once retired, `Ok(false)`
    /// while the consumer is outstanding (always `false` after a failed event
    /// record: use `retire`). On error the phase is unchanged.
    pub fn try_retire(&mut self, hip: &HipRuntime) -> HipResult<bool> {
        match self.ledger.retire_plan().map_err(mapped_err)? {
            MappedQuiesce::Done => Ok(true),
            MappedQuiesce::Device => Ok(false),
            MappedQuiesce::Event => {
                if hip.event_query(&self.retire_event)? {
                    self.ledger.complete_retirement();
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        }
    }

    /// Run `call` with this buffer's device bound, restoring the caller's
    /// binding afterwards. `true` only if every HIP step succeeded.
    fn quiesce_on_device(&self, call: impl FnOnce() -> u32) -> bool {
        let mut previous = 0;
        // SAFETY: the owner keeps the function-pointer library live; out-param
        // is a stack local.
        let got = unsafe { (self.get_device)(&mut previous) };
        crate::error::observe_raw_code(got);
        if got != HIP_SUCCESS {
            return false;
        }
        // SAFETY: as above; binds the device recorded at construction.
        let bound = unsafe { (self.set_device)(self.device) };
        crate::error::observe_raw_code(bound);
        if bound != HIP_SUCCESS {
            return false;
        }
        let code = call();
        crate::error::observe_raw_code(code);
        // SAFETY: as above; restores the caller's device binding.
        let restored = unsafe { (self.set_device)(previous) };
        crate::error::observe_raw_code(restored);
        code == HIP_SUCCESS && restored == HIP_SUCCESS
    }
}

impl Drop for MappedHostBuffer {
    fn drop(&mut self) {
        let event_raw = self.retire_event.0;
        let quiesced = match self.ledger.drop_plan() {
            MappedQuiesce::Done => true,
            MappedQuiesce::Event => self.quiesce_on_device(|| {
                // SAFETY: live event owned by this buffer; device is bound.
                unsafe { (self.event_synchronize)(event_raw) }
            }),
            MappedQuiesce::Device => self.quiesce_on_device(|| {
                // SAFETY: device synchronize takes no arguments; device is bound.
                unsafe { (self.synchronize)() }
            }),
        };
        if !quiesced {
            // Retain allocation, event and library rather than freeing memory
            // the device may still read after a failed drain. The unreleased
            // emu token stays charged.
            std::mem::forget(Arc::clone(&self._lib));
            return;
        }
        // SAFETY: the consumer is quiesced, so the event is idle and unshared.
        let destroyed = unsafe { (self.event_destroy)(event_raw) };
        crate::error::observe_raw_code(destroyed);
        // SAFETY: unique hipHostMalloc owner of `host`; library still live.
        let code = unsafe { (self.free)(self.host) };
        crate::error::observe_raw_code(code);
        if let Some(token) = self.token.take() {
            let _ =
                crate::emu_budget::charged_free::<crate::emu_budget::Process>(Some(token), || {
                    if code == HIP_SUCCESS {
                        Ok(())
                    } else {
                        Err(HipError::new(code, "hipHostFree"))
                    }
                });
        }
    }
}

// ── Signal words (CP-side waits) ────────────────────────────────

/// Largest generation a [`SignalWord`] may carry: 31 bits. Generations count
/// up from 1; the word starts at 0.
pub const SIGNAL_GEN_MAX: u32 = 0x7FFF_FFFF;

/// Reserved release value. It compares `>=` every generation, so a
/// `Gte` wait on any generation is released by a store of `SIGNAL_ABORT`. It
/// is never a generation and is never reused: a word that holds it is dead.
pub const SIGNAL_ABORT: u32 = u32::MAX;

/// Pure monotonic generation source of one signal word pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignalGen {
    last: u32,
}

impl SignalGen {
    pub const fn new() -> Self {
        Self { last: 0 }
    }

    #[cfg(test)]
    fn at(last: u32) -> Self {
        Self { last }
    }

    /// The most recently issued generation (0 before the first).
    pub const fn last(&self) -> u32 {
        self.last
    }

    /// Whether `value` may be written to / waited on as a generation.
    pub const fn is_generation(value: u32) -> bool {
        value >= 1 && value <= SIGNAL_GEN_MAX
    }

    /// The next generation. A counter that would pass [`SIGNAL_GEN_MAX`] is
    /// refused instead of wrapping: a wrapped (smaller) generation would be
    /// already satisfied by the stale word value.
    pub fn next(&mut self) -> HipResult<u32> {
        if self.last >= SIGNAL_GEN_MAX {
            return Err(mapped_err(
                "signal generation counter exhausted; wrap refused",
            ));
        }
        self.last += 1;
        Ok(self.last)
    }
}

impl Default for SignalGen {
    fn default() -> Self {
        Self::new()
    }
}

/// Steps of destroying a signal word, in the only safe order.
trait SignalTeardown {
    /// Release any wait that may still be pending on the word.
    fn store_abort(&mut self);
    /// Drain the device; `false` when quiescence is not proved.
    fn quiesce(&mut self) -> bool;
    /// Release the allocation; `false` when it failed.
    fn free(&mut self) -> bool;
}

/// Abort FIRST (a drain with a pending wait never finishes), drain second,
/// free only after a proved drain. `false` means the allocation must be
/// retained.
fn teardown_signal(t: &mut impl SignalTeardown) -> bool {
    t.store_abort();
    if !t.quiesce() {
        return false;
    }
    t.free()
}

/// Run `call` with `device` bound, restoring the caller's binding. `true` only
/// if every HIP step succeeded.
fn call_on_device(
    get_device: unsafe extern "C" fn(*mut c_int) -> u32,
    set_device: unsafe extern "C" fn(c_int) -> u32,
    device: i32,
    call: impl FnOnce() -> u32,
) -> bool {
    let mut previous = 0;
    // SAFETY: the caller's owner keeps the function-pointer library live; the
    // out-param is a stack local.
    let got = unsafe { get_device(&mut previous) };
    crate::error::observe_raw_code(got);
    if got != HIP_SUCCESS {
        return false;
    }
    // SAFETY: as above; binds the device recorded at construction.
    let bound = unsafe { set_device(device) };
    crate::error::observe_raw_code(bound);
    if bound != HIP_SUCCESS {
        return false;
    }
    let code = call();
    crate::error::observe_raw_code(code);
    // SAFETY: as above; restores the caller's device binding.
    let restored = unsafe { set_device(previous) };
    crate::error::observe_raw_code(restored);
    code == HIP_SUCCESS && restored == HIP_SUCCESS
}

/// Host pointer of a [`SignalWord`] for a thread that must store
/// [`SIGNAL_ABORT`] (the join watchdog).
#[derive(Clone, Copy, Debug)]
pub struct SignalHostPtr(*mut u32);

// SAFETY: the pointee is signal memory accessed only through volatile
// loads/stores of one aligned 32-bit word; the holder's contract (see
// `SignalHostPtr::store`) keeps the owning word alive.
unsafe impl Send for SignalHostPtr {}
// SAFETY: as above; concurrent volatile stores of a u32 are the protocol.
unsafe impl Sync for SignalHostPtr {}

impl SignalHostPtr {
    /// Volatile store of `value` after a release fence.
    ///
    /// # Safety
    /// The [`SignalWord`] this pointer came from must still be alive. A holder
    /// must stop using the pointer (disarm) before the word is dropped.
    pub unsafe fn store(self, value: u32) {
        std::sync::atomic::fence(Ordering::Release);
        // SAFETY: the caller guarantees the word is live; it is an aligned u32.
        unsafe { ptr::write_volatile(self.0, value) }
    }
}

/// One 32-bit word of `hipMallocSignalMemory` storage: GPU-writable by
/// `hipStreamWriteValue32`, GPU-waitable by `hipStreamWaitValue32`, and
/// host-accessible only through volatile loads/stores plus explicit fences.
///
/// The allocation is owned here exactly once (`hipFree` after quiescence); the
/// device view is a non-owning `Borrowed` alias. Drop stores
/// [`SIGNAL_ABORT`] first, drains the device, and retains the allocation if
/// the drain is not proved: freeing signal memory that a queued wait may still
/// poll is never allowed.
pub struct SignalWord {
    owner: Option<DeviceBuffer>,
    view: DeviceBuffer,
    host: *mut u32,
    device: i32,
    free: unsafe extern "C" fn(*mut c_void) -> u32,
    get_device: unsafe extern "C" fn(*mut c_int) -> u32,
    set_device: unsafe extern "C" fn(c_int) -> u32,
    synchronize: unsafe extern "C" fn() -> u32,
    _lib: Arc<Library>,
}

// SAFETY: unique owner of its signal allocation; the host pointer is used only
// through volatile accesses of one aligned word. Moving it does not move the
// allocation. Deliberately NOT Sync: other threads get `SignalHostPtr`.
unsafe impl Send for SignalWord {}

const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<SignalWord>();
};

impl SignalWord {
    pub fn device(&self) -> i32 {
        self.device
    }

    /// Non-owning device view: pass it to the stream value operations. Never
    /// free it and do not let it outlive the word.
    pub fn device_buffer(&self) -> &DeviceBuffer {
        &self.view
    }

    /// Volatile load of the word.
    pub fn load(&self) -> u32 {
        // SAFETY: `host` is the live, aligned 4-byte signal allocation.
        unsafe { ptr::read_volatile(self.host) }
    }

    /// Volatile load followed by an acquire fence: bytes the GPU wrote before
    /// the value it just read are visible after this returns.
    pub fn load_acquire(&self) -> u32 {
        let value = self.load();
        std::sync::atomic::fence(Ordering::Acquire);
        value
    }

    /// Volatile store with no ordering of earlier stores.
    pub fn store(&self, value: u32) {
        // SAFETY: as `load`.
        unsafe { ptr::write_volatile(self.host, value) }
    }

    /// Release fence, then a volatile store: every earlier host store is
    /// ordered before `value` for any observer that acquires it. This is a
    /// host-side order only; GPU visibility is established by a hardware oracle.
    pub fn store_release(&self, value: u32) {
        std::sync::atomic::fence(Ordering::Release);
        self.store(value);
    }

    /// Pointer for the watchdog thread; see [`SignalHostPtr::store`].
    pub fn host_ptr(&self) -> SignalHostPtr {
        SignalHostPtr(self.host)
    }
}

struct SignalWordTeardown<'a>(&'a mut SignalWord);

impl SignalTeardown for SignalWordTeardown<'_> {
    fn store_abort(&mut self) {
        self.0.store_release(SIGNAL_ABORT);
    }

    fn quiesce(&mut self) -> bool {
        let word = &*self.0;
        call_on_device(word.get_device, word.set_device, word.device, || {
            // SAFETY: device synchronize takes no arguments; the device is bound.
            unsafe { (word.synchronize)() }
        })
    }

    fn free(&mut self) -> bool {
        let word = &mut *self.0;
        let Some(mut owner) = word.owner.take() else {
            return true;
        };
        let raw = owner.as_ptr();
        let token = owner.take_charge();
        let free = word.free;
        let freed = crate::emu_budget::charged_free::<crate::emu_budget::Process>(token, || {
            // SAFETY: unique hipExtMallocWithFlags owner; the drain was proved
            // and the abort value releases any wait, so nothing polls it.
            let code = unsafe { free(raw) };
            crate::error::observe_raw_code(code);
            if code == HIP_SUCCESS {
                crate::registry::forget(raw as usize);
                Ok(())
            } else {
                Err(HipError::new(code, "hipFree"))
            }
        });
        freed.is_ok()
    }
}

impl Drop for SignalWord {
    fn drop(&mut self) {
        if self.owner.is_none() {
            return;
        }
        if !teardown_signal(&mut SignalWordTeardown(self)) {
            // Retain the allocation and the library: a queued wait may still
            // poll the word. The unreleased emu charge stays.
            std::mem::forget(Arc::clone(&self._lib));
        }
    }
}

/// GPU stream handle.
pub struct Stream(HipStream);
impl Stream {
    pub fn as_raw(&self) -> *mut c_void {
        self.0
    }
    /// Alias for `as_raw` used by the TP/RCCL collective callers (which take
    /// a raw stream pointer to hand to `ncclAllReduce`).
    pub fn raw_ptr(&self) -> *mut c_void {
        self.0
    }
    /// Non-owning copy of this handle, for borrowing a stream where an owned
    /// `Stream` is expected (e.g. `Gpu::active_stream`).
    ///
    /// # Safety
    /// The alias must not outlive the stream and must never reach
    /// `stream_destroy`; the owner destroys the stream exactly once.
    pub unsafe fn alias(&self) -> Stream {
        Stream(self.0)
    }
}
// SAFETY: Stream is an opaque HIP handle (*mut c_void). Sending the handle
// between threads is fine; it is not Sync. Concurrent use needs
// external stream synchronization; destroy only when idle.
unsafe impl Send for Stream {}

/// Loaded GPU module (compiled kernels).
pub struct Module(HipModule);
// SAFETY: Module is an opaque HIP handle (*mut c_void). Sending the handle
// between threads is fine; it is not Sync. Concurrent use needs
// external stream synchronization; destroy only when idle.
unsafe impl Send for Module {}

impl Module {
    /// Raw `hipModule_t` handle, for identity keys only (never dereferenced).
    pub fn raw_ptr(&self) -> *mut c_void {
        self.0
    }
}

/// Handle to a specific kernel function within a module.
pub struct Function(HipFunction);
// SAFETY: Function is an opaque HIP handle (*mut c_void). Sending the handle
// between threads is fine; it is not Sync. Concurrent use needs
// external stream synchronization; destroy only when idle.
unsafe impl Send for Function {}

impl Function {
    /// Raw `hipFunction_t` handle, for identity keys only (never dereferenced).
    pub fn raw_ptr(&self) -> *mut c_void {
        self.0
    }
}

/// GPU event for timing.
pub struct Event(HipEvent);
// SAFETY: Event is an opaque HIP handle (*mut c_void). Sending the handle
// between threads is fine; it is not Sync. Concurrent use needs
// external stream synchronization; destroy only when idle.
unsafe impl Send for Event {}

/// Captured GPU operation graph.
pub struct Graph(HipGraph);
// SAFETY: Graph is an opaque HIP handle (*mut c_void). Sending the handle
// between threads is fine; it is not Sync. Concurrent use needs
// external stream synchronization; destroy only when idle.
unsafe impl Send for Graph {}

/// Executable (instantiated) graph ready for replay.
pub struct GraphExec(HipGraphExec);
// SAFETY: GraphExec is an opaque HIP handle (*mut c_void). Sending the handle
// between threads is fine; it is not Sync. Concurrent use needs
// external stream synchronization; destroy only when idle.
unsafe impl Send for GraphExec {}

#[cfg(test)]
mod mapped_library_tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn diagnostic_lookup_resolves_a_live_symbol_without_hip() {
        let path = dladdr_fname(dladdr as *const c_void)
            .expect("the dynamic loader must resolve its own dladdr symbol");
        assert!(!path.is_empty());
    }

    #[test]
    fn diagnostic_lookup_handles_an_unknown_address() {
        assert_eq!(dladdr_fname(ptr::null()), None);
    }
}

#[cfg(test)]
mod pageable_copy_staging_tests {
    use super::{pageable_copy_staging, PageableCopyStaging};
    use std::ffi::OsStr;

    fn decide(opt_out: Option<&str>, preset: Option<&str>) -> PageableCopyStaging {
        pageable_copy_staging(opt_out.map(OsStr::new), preset.map(OsStr::new))
    }

    #[test]
    fn both_unset_arms_the_default() {
        assert_eq!(decide(None, None), PageableCopyStaging::Arm);
    }

    #[test]
    fn opt_out_zero_leaves_the_switch_unset() {
        assert_eq!(decide(Some("0"), None), PageableCopyStaging::OptedOut);
    }

    #[test]
    fn any_other_opt_out_value_still_arms() {
        for value in ["1", "", "00", "false", " 0", "0 "] {
            assert_eq!(
                decide(Some(value), None),
                PageableCopyStaging::Arm,
                "opt-out {value:?}"
            );
        }
    }

    #[test]
    fn a_preset_switch_is_always_respected() {
        for preset in ["0", "100000", "64", ""] {
            assert_eq!(
                decide(None, Some(preset)),
                PageableCopyStaging::Preset,
                "preset {preset:?}"
            );
            assert_eq!(
                decide(Some("0"), Some(preset)),
                PageableCopyStaging::Preset,
                "preset {preset:?} with opt-out"
            );
            assert_eq!(
                decide(Some("1"), Some(preset)),
                PageableCopyStaging::Preset,
                "preset {preset:?} with =1"
            );
        }
    }
}

#[cfg(test)]
mod mapped_ledger_tests {
    use super::{
        check_f32_extent, check_mapped_extent, checked_alias, MappedLedger, MappedPhase,
        MappedPublished, MappedQuiesce,
    };
    use std::ffi::c_void;

    /// Drive a ledger through one write → publish.
    fn publish_one(l: &mut MappedLedger) -> MappedPublished {
        l.begin_write().unwrap();
        l.publish().unwrap()
    }

    fn full_cycle(l: &mut MappedLedger) {
        let t = publish_one(l);
        l.enter_gpu_reading(&t, true).unwrap();
        l.complete_retirement();
    }

    #[test]
    fn starts_retired_generation_zero() {
        let l = MappedLedger::new();
        assert_eq!(l.phase, MappedPhase::Retired);
        assert_eq!(l.generation, 0);
        assert_eq!(l.retire_plan(), Ok(MappedQuiesce::Done));
        assert_eq!(l.drop_plan(), MappedQuiesce::Done);
    }

    #[test]
    fn legal_cycle_walks_every_phase() {
        let mut l = MappedLedger::new();
        l.begin_write().unwrap();
        assert_eq!(l.phase, MappedPhase::CpuWriting);
        let t = l.publish().unwrap();
        assert_eq!(l.phase, MappedPhase::Published);
        assert_eq!(t.generation(), 1);
        l.enter_gpu_reading(&t, true).unwrap();
        assert_eq!(l.phase, MappedPhase::GpuReading);
        assert_eq!(l.retire_plan(), Ok(MappedQuiesce::Event));
        assert_eq!(l.drop_plan(), MappedQuiesce::Event);
        l.complete_retirement();
        assert_eq!(l.phase, MappedPhase::Retired);
    }

    #[test]
    fn begin_write_only_from_retired() {
        let mut l = MappedLedger::new();
        l.begin_write().unwrap();
        assert!(l.begin_write().is_err(), "CpuWriting");
        let t = l.publish().unwrap();
        assert!(l.begin_write().is_err(), "Published");
        l.enter_gpu_reading(&t, true).unwrap();
        assert!(l.begin_write().is_err(), "GpuReading");
        assert_eq!(l.phase, MappedPhase::GpuReading);
        l.complete_retirement();
        l.begin_write().unwrap();
    }

    #[test]
    fn publish_only_from_cpu_writing() {
        let mut l = MappedLedger::new();
        assert!(l.publish().is_err(), "Retired");
        let t = publish_one(&mut l);
        assert!(l.publish().is_err(), "Published");
        l.enter_gpu_reading(&t, true).unwrap();
        assert!(l.publish().is_err(), "GpuReading");
        assert_eq!(l.generation, 1);
    }

    #[test]
    fn mark_requires_published_phase() {
        let mut l = MappedLedger::new();
        let t = publish_one(&mut l);
        l.enter_gpu_reading(&t, true).unwrap();
        // Already GpuReading: a same-generation token (forged for the test) is refused.
        let again = MappedPublished {
            buffer: l.id,
            generation: l.generation,
        };
        assert!(l.enter_gpu_reading(&again, true).is_err());
        l.complete_retirement();
        assert!(l.enter_gpu_reading(&again, true).is_err(), "Retired");
        l.begin_write().unwrap();
        assert!(l.enter_gpu_reading(&again, true).is_err(), "CpuWriting");
        assert_eq!(l.phase, MappedPhase::CpuWriting);
    }

    #[test]
    fn generation_is_monotonic_one_per_publication() {
        let mut l = MappedLedger::new();
        for want in 1..=64u64 {
            let t = publish_one(&mut l);
            assert_eq!(t.generation(), want);
            assert_eq!(l.generation, want);
            l.enter_gpu_reading(&t, true).unwrap();
            l.complete_retirement();
        }
    }

    #[test]
    fn foreign_token_refused_without_state_change() {
        let mut a = MappedLedger::new();
        let mut b = MappedLedger::new();
        assert_ne!(a.id, b.id);
        let ta = publish_one(&mut a);
        let tb = publish_one(&mut b);
        assert_eq!(ta.generation(), tb.generation());
        assert!(a.enter_gpu_reading(&tb, true).is_err());
        assert_eq!(a.phase, MappedPhase::Published);
        assert!(b.enter_gpu_reading(&ta, true).is_err());
        assert_eq!(b.phase, MappedPhase::Published);
        a.enter_gpu_reading(&ta, true).unwrap();
        b.enter_gpu_reading(&tb, true).unwrap();
    }

    #[test]
    fn stale_token_refused() {
        let mut l = MappedLedger::new();
        let first = publish_one(&mut l);
        l.enter_gpu_reading(&first, true).unwrap();
        l.complete_retirement();
        let _second = publish_one(&mut l);
        assert!(l.check_token(&first).is_err());
        assert!(l.enter_gpu_reading(&first, true).is_err());
        assert_eq!(l.phase, MappedPhase::Published);
        assert_eq!(l.generation, 2);
    }

    #[test]
    fn abandoned_write_returns_retired_without_generation_bump() {
        let mut l = MappedLedger::new();
        full_cycle(&mut l);
        assert_eq!(l.generation, 1);
        l.begin_write().unwrap();
        l.abandon_write();
        assert_eq!(l.phase, MappedPhase::Retired);
        assert_eq!(l.generation, 1);
        // abandon is inert outside CpuWriting.
        let t = publish_one(&mut l);
        l.abandon_write();
        assert_eq!(l.phase, MappedPhase::Published);
        l.enter_gpu_reading(&t, true).unwrap();
        l.abandon_write();
        assert_eq!(l.phase, MappedPhase::GpuReading);
    }

    #[test]
    fn retire_plan_refuses_published_and_writing() {
        let mut l = MappedLedger::new();
        l.begin_write().unwrap();
        assert!(l.retire_plan().is_err(), "CpuWriting");
        let _t = l.publish().unwrap();
        assert!(l.retire_plan().is_err(), "Published without mark");
        assert_eq!(l.phase, MappedPhase::Published);
        // Drop must still quiesce an unmarked Published buffer by device sync.
        assert_eq!(l.drop_plan(), MappedQuiesce::Device);
    }

    #[test]
    fn failed_event_record_fails_closed_to_device_sync() {
        let mut l = MappedLedger::new();
        let t = publish_one(&mut l);
        l.enter_gpu_reading(&t, false).unwrap();
        assert_eq!(l.phase, MappedPhase::GpuReading);
        assert_eq!(l.retire_plan(), Ok(MappedQuiesce::Device));
        assert_eq!(l.drop_plan(), MappedQuiesce::Device);
        assert!(l.begin_write().is_err());
        l.complete_retirement();
        assert_eq!(l.phase, MappedPhase::Retired);
        assert!(!l.unfenced);
        // The next fenced cycle is event-based again.
        let t = publish_one(&mut l);
        l.enter_gpu_reading(&t, true).unwrap();
        assert_eq!(l.retire_plan(), Ok(MappedQuiesce::Event));
    }

    #[test]
    fn retire_error_keeps_phase_because_completion_is_explicit() {
        // retire/try_retire call complete_retirement only after the HIP wait
        // succeeds; planning alone never changes the phase.
        let mut l = MappedLedger::new();
        let t = publish_one(&mut l);
        l.enter_gpu_reading(&t, true).unwrap();
        for _ in 0..3 {
            assert_eq!(l.retire_plan(), Ok(MappedQuiesce::Event));
            assert_eq!(l.phase, MappedPhase::GpuReading);
        }
        assert_eq!(l.generation, 1);
    }

    #[test]
    fn generation_exhaustion_refuses_new_write() {
        let mut l = MappedLedger::new();
        l.generation = u64::MAX - 1;
        full_cycle(&mut l);
        assert_eq!(l.generation, u64::MAX);
        assert!(l.begin_write().is_err());
        assert_eq!(l.phase, MappedPhase::Retired);
    }

    #[test]
    fn extent_guards() {
        assert!(check_mapped_extent(0).is_err());
        assert!(check_mapped_extent(usize::MAX).is_err());
        assert!(check_mapped_extent(isize::MAX as usize + 1).is_err());
        assert!(check_mapped_extent(1).is_ok());
        assert!(check_mapped_extent(isize::MAX as usize).is_ok());
    }

    #[test]
    fn f32_extent_refusal() {
        assert!(check_f32_extent(4096, 0).is_ok());
        assert!(check_f32_extent(4096, 4).is_ok());
        assert!(check_f32_extent(4096, 102_400).is_ok());
        for bad in [1usize, 2, 3, 5, 102_401, 102_403] {
            assert!(check_f32_extent(4096, bad).is_err(), "size {bad}");
        }
        assert!(check_f32_extent(4097, 8).is_err(), "misaligned");
    }

    #[test]
    fn null_alias_refused() {
        assert!(checked_alias(std::ptr::null_mut(), 64).is_err());
        let mut byte = 0u8;
        let p = (&mut byte as *mut u8).cast::<c_void>();
        assert_eq!(checked_alias(p, 64).unwrap(), p);
    }

    #[test]
    fn token_is_not_clone_or_copy() {
        // Inherent-const-priority probe: resolves to `true` only if Clone.
        trait NotClone {
            const IS_CLONE: bool = false;
        }
        impl<T> NotClone for T {}
        struct Probe<T>(std::marker::PhantomData<T>);
        #[allow(dead_code)]
        impl<T: Clone> Probe<T> {
            const IS_CLONE: bool = true;
        }
        assert!(!<Probe<MappedPublished>>::IS_CLONE);
    }
}

#[cfg(test)]
mod transfer_ready_tests {
    use super::{InputReady, ReadyKind, Retire, TransferLedger};

    const S1: usize = 0x10;
    const S2: usize = 0x20;

    /// A ledger with one enqueue on `stream` at `seq`.
    fn enqueued(stream: usize, seq: u64) -> TransferLedger {
        let mut l = TransferLedger::new();
        l.enqueued_at(stream, seq);
        l
    }

    #[test]
    fn arm_requires_pending_work() {
        let mut l = TransferLedger::new();
        assert!(l.arm_at(ReadyKind::Event, 5).is_err(), "nothing queued");
        l.enqueued_at(S1, 1);
        assert!(l.arm_at(ReadyKind::Event, 2).is_ok());
    }

    #[test]
    fn only_the_matching_token_completes_the_buffer() {
        let mut a = enqueued(S1, 1);
        let mut b = enqueued(S1, 2);
        let mut ta = a.arm_at(ReadyKind::Event, 3).unwrap();
        let mut tb = b.arm_at(ReadyKind::Event, 4).unwrap();
        // A token of another buffer is refused and changes nothing.
        assert!(a.complete(&mut tb).is_err());
        assert!(b.pending && a.pending);
        a.complete(&mut ta).unwrap();
        assert!(!a.pending && ta.is_complete());
        assert!(b.pending, "the other buffer is untouched");
        b.complete(&mut tb).unwrap();
        assert!(!b.pending);
    }

    #[test]
    fn a_stale_or_replaced_token_cannot_complete() {
        let mut l = enqueued(S1, 1);
        let mut old = l.arm_at(ReadyKind::Event, 2).unwrap();
        let mut new = l.arm_at(ReadyKind::Event, 3).unwrap();
        assert!(
            l.complete(&mut old).is_err(),
            "re-armed: the old token is stale"
        );
        assert!(l.pending);
        // A later enqueue also invalidates the armed token: the event no
        // longer covers the buffer.
        l.enqueued_at(S1, 4);
        assert!(l.complete(&mut new).is_err(), "later enqueue");
        assert!(l.pending);
        let mut fresh = l.arm_at(ReadyKind::Event, 5).unwrap();
        l.complete(&mut fresh).unwrap();
        assert!(
            l.complete(&mut fresh).is_err(),
            "a completed token completes once"
        );
    }

    #[test]
    fn a_token_kind_must_match_the_armed_kind() {
        let mut l = enqueued(S1, 1);
        let mut ev = l.arm_at(ReadyKind::Event, 2).unwrap();
        let mut forged = InputReady {
            kind: ReadyKind::Word(7),
            ..l.arm_at(ReadyKind::Event, 3).unwrap()
        };
        assert!(l.complete(&mut forged).is_err(), "armed Event, token Word");
        assert!(
            l.complete(&mut ev).is_err(),
            "the first token was replaced by the second arm"
        );
    }

    #[test]
    fn a_word_token_completes_only_after_its_value_is_observed() {
        let mut l = enqueued(S1, 1);
        let mut t = l.arm_at(ReadyKind::Word(9), 2).unwrap();
        assert_eq!(l.observe_word(&t, 8), Ok(false));
        assert!(l.pending);
        assert_eq!(l.observe_word(&t, 9), Ok(true));
        l.complete(&mut t).unwrap();
        assert!(!l.pending);
        let mut e = enqueued(S1, 3);
        let ev = e.arm_at(ReadyKind::Event, 4).unwrap();
        assert!(
            e.observe_word(&ev, u32::MAX).is_err(),
            "an event token never reads a word"
        );
    }

    #[test]
    fn a_completed_proof_retires_an_earlier_buffer_on_the_same_stream() {
        let mut input = enqueued(S1, 10);
        let mut mask = enqueued(S1, 4);
        let mut proof = input.arm_at(ReadyKind::Event, 11).unwrap();
        assert!(
            mask.retire_behind(&proof).is_err(),
            "an uncompleted proof retires nothing"
        );
        input.complete(&mut proof).unwrap();
        assert_eq!(mask.retire_behind(&proof), Ok(Retire::Retired));
        assert!(!mask.pending);
        assert_eq!(
            mask.retire_behind(&proof),
            Ok(Retire::Retired),
            "idempotent"
        );
    }

    #[test]
    fn a_stream_mismatch_or_a_later_sequence_is_not_covered() {
        let mut input = enqueued(S1, 10);
        let mut proof = input.arm_at(ReadyKind::Event, 11).unwrap();
        input.complete(&mut proof).unwrap();
        let mut other_stream = enqueued(S2, 4);
        assert_eq!(other_stream.retire_behind(&proof), Ok(Retire::NotCovered));
        assert!(other_stream.pending);
        let mut later = enqueued(S1, 12);
        assert_eq!(
            later.retire_behind(&proof),
            Ok(Retire::NotCovered),
            "enqueued after the arm"
        );
        assert!(later.pending);
        let mut equal = enqueued(S1, 11);
        assert_eq!(
            equal.retire_behind(&proof),
            Ok(Retire::NotCovered),
            "same sequence is not smaller"
        );
    }

    #[test]
    fn real_sequences_are_process_wide_and_monotonic() {
        let mut a = TransferLedger::new();
        let mut b = TransferLedger::new();
        a.enqueued(S1);
        b.enqueued(S1);
        a.enqueued(S1);
        assert!(a.last_seq > b.last_seq && b.last_seq > 0);
        let mut proof = a.arm(ReadyKind::Event).unwrap();
        a.complete(&mut proof).unwrap();
        assert!(proof.sequence() > a.last_seq);
        assert_eq!(b.retire_behind(&proof), Ok(Retire::Retired));
    }

    #[test]
    fn a_stream_change_requires_a_drain_only_while_pending() {
        let mut l = TransferLedger::new();
        assert!(!l.needs_drain(S2));
        l.enqueued_at(S1, 1);
        assert!(!l.needs_drain(S1));
        assert!(l.needs_drain(S2));
        l.drained();
        assert!(!l.needs_drain(S2));
    }

    #[test]
    fn readiness_tokens_are_not_clone_or_copy() {
        trait NotClone {
            const IS_CLONE: bool = false;
        }
        impl<T> NotClone for T {}
        struct Probe<T>(std::marker::PhantomData<T>);
        #[allow(dead_code)]
        impl<T: Clone> Probe<T> {
            const IS_CLONE: bool = true;
        }
        // Compile-time: the trait default is chosen only when `InputReady` is
        // not `Clone` (the inherent constant needs `T: Clone`).
        const _: () = assert!(!<Probe<InputReady>>::IS_CLONE);
    }
}

#[cfg(test)]
mod signal_word_tests {
    use super::{teardown_signal, SignalGen, SignalTeardown, SIGNAL_ABORT, SIGNAL_GEN_MAX};

    #[test]
    fn generations_are_31_bit_monotonic_and_start_at_one() {
        let mut g = SignalGen::new();
        assert_eq!(g.last(), 0);
        for want in 1..=1000u32 {
            assert_eq!(g.next().unwrap(), want);
        }
        assert_eq!(SIGNAL_GEN_MAX, (1u32 << 31) - 1);
    }

    /// The abort sentinel sits strictly above every generation: a store of it
    /// releases a `Gte` wait on any generation and can never be mistaken for
    /// one. Checked at compile time.
    const _: () = assert!(SIGNAL_GEN_MAX < SIGNAL_ABORT);

    #[test]
    fn the_counter_never_issues_the_abort_value_up_to_exhaustion() {
        // Drive the real counter across its last generations to the refusal
        // boundary: every issued value is a valid generation strictly below
        // the abort sentinel (so an abort store releases a wait on it), and
        // the counter refuses instead of ever reaching or wrapping past it.
        let mut g = SignalGen::at(SIGNAL_GEN_MAX - 3);
        let mut issued = Vec::new();
        while let Ok(gen) = g.next() {
            issued.push(gen);
        }
        assert_eq!(
            issued,
            [SIGNAL_GEN_MAX - 2, SIGNAL_GEN_MAX - 1, SIGNAL_GEN_MAX]
        );
        for &gen in &issued {
            assert!(SignalGen::is_generation(gen));
            assert!(
                gen < SIGNAL_ABORT,
                "an abort store releases a Gte wait on {gen}"
            );
        }
        assert_eq!(g.last(), SIGNAL_GEN_MAX);
        assert!(!SignalGen::is_generation(SIGNAL_ABORT));
        assert!(
            !SignalGen::is_generation(0),
            "the initial word value is never a generation"
        );
        assert!(!SignalGen::is_generation(SIGNAL_GEN_MAX + 1));
    }

    #[test]
    fn wrap_is_refused_and_leaves_the_counter_unchanged() {
        let mut g = SignalGen::at(SIGNAL_GEN_MAX - 1);
        assert_eq!(g.next().unwrap(), SIGNAL_GEN_MAX);
        assert!(g.next().is_err());
        assert_eq!(g.last(), SIGNAL_GEN_MAX);
        assert!(g.next().is_err(), "stays refused");
    }

    #[derive(Default)]
    struct Rec {
        log: Vec<&'static str>,
        sync_ok: bool,
        free_ok: bool,
    }

    impl SignalTeardown for Rec {
        fn store_abort(&mut self) {
            self.log.push("abort");
        }
        fn quiesce(&mut self) -> bool {
            self.log.push("sync");
            self.sync_ok
        }
        fn free(&mut self) -> bool {
            self.log.push("free");
            self.free_ok
        }
    }

    #[test]
    fn teardown_stores_abort_before_the_drain_and_frees_last() {
        let mut r = Rec {
            sync_ok: true,
            free_ok: true,
            ..Rec::default()
        };
        assert!(teardown_signal(&mut r));
        assert_eq!(r.log, ["abort", "sync", "free"]);
    }

    #[test]
    fn teardown_never_frees_after_a_failed_drain() {
        let mut r = Rec {
            sync_ok: false,
            free_ok: true,
            ..Rec::default()
        };
        assert!(!teardown_signal(&mut r));
        assert_eq!(
            r.log,
            ["abort", "sync"],
            "a pending wait may still poll the word"
        );
    }
}

#[cfg(test)]
mod host_alloc_owner_tests {
    use super::{HostAllocOwner, HIP_SUCCESS};
    use std::ffi::c_void;
    use std::sync::{Arc, Mutex};

    /// Pointers the fake `hipHostFree` was called with, in order.
    static FREED: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    /// The pointer whose fake free reports a HIP error.
    const FAILING: usize = 0x7000_0000;
    /// The same, for the test that holds a real emu pinned-host charge on it.
    const FAILING_CHARGED: usize = 0x7000_1000;

    unsafe extern "C" fn fake_free(ptr: *mut c_void) -> u32 {
        FREED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(ptr as usize);
        if ptr as usize == FAILING || ptr as usize == FAILING_CHARGED {
            1
        } else {
            HIP_SUCCESS
        }
    }

    fn freed(ptr: usize) -> usize {
        FREED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&p| p == ptr)
            .count()
    }

    fn owner(ptr: usize) -> HostAllocOwner {
        HostAllocOwner::for_test(ptr as *mut c_void, fake_free)
    }

    #[test]
    fn host_alloc_owner_drop_frees_exactly_once() {
        let ptr = 0x1000_0000;
        let o = owner(ptr);
        assert_eq!(o.host_ptr() as usize, ptr);
        assert_eq!(freed(ptr), 0, "adopting frees nothing");
        drop(o);
        assert_eq!(freed(ptr), 1);
    }

    #[test]
    fn host_alloc_owner_last_arc_holder_frees_after_every_other_owner_is_gone() {
        let ptr = 0x1000_1000;
        // The registry's reference, a stream-held lease and a quarantined ring page.
        let registry = Arc::new(owner(ptr));
        let lease = Arc::clone(&registry);
        let quarantined = Arc::clone(&registry);
        // `free_tensor` / Gpu drop releases the registry's reference: nothing is freed yet.
        drop(registry);
        assert_eq!(freed(ptr), 0);
        drop(lease);
        assert_eq!(
            freed(ptr),
            0,
            "a quarantined page still holds the allocation"
        );
        // Quarantine-then-drain: the drain drops the last holder, which frees once.
        drop(quarantined);
        assert_eq!(freed(ptr), 1);
    }

    #[test]
    fn host_alloc_owner_outlives_the_structure_that_adopted_it() {
        // The owner holds its own free function and library reference, never the runtime:
        // dropping every other structure first leaves it freeable.
        let ptr = 0x1000_2000;
        let holders: Vec<Arc<HostAllocOwner>> = vec![Arc::new(owner(ptr))];
        let moved = holders.into_iter().next().unwrap();
        assert_eq!(freed(ptr), 0);
        drop(moved);
        assert_eq!(freed(ptr), 1);
    }

    #[test]
    fn host_alloc_owner_failed_free_is_not_retried_or_repeated() {
        // A failing hipHostFree leaks (the keyed emu charge is kept by `release_keyed`); the
        // owner never frees a second time.
        drop(owner(FAILING));
        assert_eq!(freed(FAILING), 1);
    }

    #[test]
    fn host_alloc_owner_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HostAllocOwner>();
        assert_send_sync::<Arc<HostAllocOwner>>();
    }

    /// Committed pinned-host bytes of the process emu ledger.
    fn committed_host_bytes(id: hipfire_config::emu_budget::BudgetId) -> u64 {
        hipfire_config::emu_budget::snapshot(id)
            .committed
            .host_bytes
    }

    #[test]
    fn host_alloc_owner_token_release_happens_once_and_only_after_a_successful_free() {
        use crate::emu_budget::{charged_alloc_keyed, release_keyed, Class, HOST_TOKENS};
        use hipfire_config::emu_budget::{install, BudgetLimits};
        let limits = BudgetLimits {
            device_bytes: 1 << 40,
            gtt_bytes: Some(1 << 40),
            shared_host_bytes: None,
            uma: false,
        };
        let id = install("0000:0f:00.0", limits).expect("the test ledger installs");
        let process = crate::emu_budget::active().expect("a budget is installed");
        let base = committed_host_bytes(id);

        // Success: the pointer-keyed charge of the adopted allocation is released exactly
        // once, by the last holder's free, and only after that free returned success.
        let ok_ptr = 0x2000_0000usize;
        charged_alloc_keyed(
            &process,
            &HOST_TOKENS,
            Class::HostPinned,
            4096,
            |p: &usize| *p,
            || Ok(ok_ptr),
        )
        .expect("the charge commits");
        assert_eq!(
            committed_host_bytes(id),
            base + 4096,
            "the allocation is charged"
        );
        let registry = Arc::new(owner(ok_ptr));
        let lease = Arc::clone(&registry);
        drop(registry);
        assert_eq!(
            committed_host_bytes(id),
            base + 4096,
            "a held lease keeps the charge"
        );
        assert_eq!(freed(ok_ptr), 0);
        drop(lease);
        assert_eq!(freed(ok_ptr), 1);
        assert_eq!(
            committed_host_bytes(id),
            base,
            "released once by the successful free"
        );

        // Failure: the free reports an error, so the charge is retained (restored under its
        // key), and a second drop path cannot release it.
        charged_alloc_keyed(
            &process,
            &HOST_TOKENS,
            Class::HostPinned,
            8192,
            |p: &usize| *p,
            || Ok(FAILING_CHARGED),
        )
        .expect("the charge commits");
        assert_eq!(committed_host_bytes(id), base + 8192);
        drop(owner(FAILING_CHARGED));
        assert_eq!(
            committed_host_bytes(id),
            base + 8192,
            "a failed free keeps the charge"
        );
        // The retained token is still keyed: a later successful release (a retry by
        // whoever can free it) releases it exactly once.
        release_keyed::<crate::emu_budget::Process>(&HOST_TOKENS, FAILING_CHARGED, || Ok(()))
            .expect("the retry succeeds");
        assert_eq!(committed_host_bytes(id), base);
        release_keyed::<crate::emu_budget::Process>(&HOST_TOKENS, FAILING_CHARGED, || Ok(()))
            .expect("a second release is a no-op");
        assert_eq!(committed_host_bytes(id), base, "no double release");
    }
}
