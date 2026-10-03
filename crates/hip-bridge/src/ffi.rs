// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! FFI bindings to libamdhip64.so via dlopen.
//! No link-time dependency — runtime loads the shared library.

use crate::error::{HipError, HipResult};
use crate::{DeviceBuffer, MemcpyKind};
use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_uint, c_void, CString};
use std::ptr;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

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

/// Per-thread tally of the HIP memory operations issued on this thread.
///
/// A retained Redline tape replays recorded kernel dispatches only, so a
/// device copy, readback or memset issued inside its capture window is state
/// the tape cannot reproduce. The replay controller differences two snapshots
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
    }

    impl MemoryEffects {
        /// The operations issued since `earlier`, a snapshot taken on this thread.
        pub fn since(self, earlier: Self) -> Self {
            Self {
                htod: self.htod - earlier.htod,
                dtod: self.dtod - earlier.dtod,
                dtoh: self.dtoh - earlier.dtoh,
                memset: self.memset - earlier.memset,
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
            Cell::new(MemoryEffects { htod: 0, dtod: 0, dtoh: 0, memset: 0 })
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
pub const HIP_MEM_ALLOCATION_GRANULARITY_MINIMUM: u32 = 0;
pub const HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED: u32 = 1;
/// Dependency event: omit profiling state and retain the default system fence.
pub const HIP_EVENT_DISABLE_TIMING: u32 = 0x2;

/// `hipHostMalloc` flag: map the allocation into device address space so device
/// code can dereference it (the zero-copy / mapped-pinned path). Without it the
/// pages are pinned but only reachable by the copy engines, which is *not*
/// enough for offloaded weights — the GEMV/GEMM kernels read them directly.
pub const HIP_HOST_MALLOC_MAPPED: u32 = 0x2;

/// Request an explicit system-scope release when recording an event.
pub const HIP_EVENT_RELEASE_TO_SYSTEM: u32 = 0x8000_0000;

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
}

/// Loaded HIP runtime — holds the dlopen'd library and resolved function pointers.
pub struct HipRuntime {
    _lib: Library,

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
/// Both switches are process-global, and together they slow other loads: on
/// gfx1201, H2's weight sweep took 1.20-1.22 s with them instead of
/// 1.00-1.01 s. So they are set only in a process that loads a Qwen4 model,
/// the one known to hold tens of GB of host memory the GPU reads:
/// [`HipRuntime::load`] sets them in a process configured to host-map Qwen4
/// experts ([`QWEN4_EXPERT_VRAM_LAYERS_ENV`]), and a process about to load a
/// Qwen4 model on a discrete GPU calls this before the runtime loads
/// (`hipfire_loader::prepare_host_memory_for`).
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
        eprintln!("[hip-bridge] {} ({reason}): host memory out of reclaim", set.join(" "));
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
    /// Load the HIP runtime via dlopen.
    /// Uses the shared ROCm resolver so runtime, headers, and hipcc stay within
    /// one selected installation.
    pub fn load() -> HipResult<Self> {
        if host_maps_qwen4_experts() {
            keep_host_memory_out_of_reclaim(&format!("{QWEN4_EXPERT_VRAM_LAYERS_ENV} set"));
        }
        // Windows and Unix share one candidate policy in hipfire_config::rocm so a
        // selected/configured root never falls through to another install's DLL
        // (user cache or bare PATH names). HIP_RUNTIME_LIBRARIES already encodes
        // the host's preferred filenames (amdhip64*.dll vs libamdhip64.so*).
        let candidates =
            hipfire_config::rocm::library_candidates(hipfire_config::rocm::HIP_RUNTIME_LIBRARIES);
        // SAFETY: libloading::Library::new maps a shared object by path; on success
        // the Library owns the mapping for the process. Failure paths only
        // read the error value.
        let lib = unsafe {
            let mut loaded = None;
            let mut last_err = None;
            for candidate in &candidates {
                match Library::new(candidate) {
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
                fn_runtime_get_version: unsafe { load_fn!(
                    lib,
                    "hipRuntimeGetVersion",
                    unsafe extern "C" fn(*mut c_int) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_get_device_count: unsafe { load_fn!(
                    lib,
                    "hipGetDeviceCount",
                    unsafe extern "C" fn(*mut c_int) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_set_device: unsafe { load_fn!(lib, "hipSetDevice", unsafe extern "C" fn(c_int) -> u32) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_set_device_flags: unsafe { load_fn!(
                    lib,
                    "hipSetDeviceFlags",
                    unsafe extern "C" fn(c_uint) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_get_device: unsafe { load_fn!(
                    lib,
                    "hipGetDevice",
                    unsafe extern "C" fn(*mut c_int) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_device_get_uuid: unsafe { load_fn!(
                    lib,
                    "hipDeviceGetUuid",
                    unsafe extern "C" fn(*mut u8, c_int) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_device_get_pci_bus_id: unsafe { load_fn!(
                    lib,
                    "hipDeviceGetPCIBusId",
                    unsafe extern "C" fn(*mut c_char, c_int, c_int) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_device_can_access_peer: unsafe { load_fn!(
                    lib,
                    "hipDeviceCanAccessPeer",
                    unsafe extern "C" fn(*mut c_int, c_int, c_int) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_device_enable_peer_access: unsafe { load_fn!(
                    lib,
                    "hipDeviceEnablePeerAccess",
                    unsafe extern "C" fn(c_int, c_uint) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_memcpy_peer: unsafe { load_fn!(
                    lib,
                    "hipMemcpyPeer",
                    unsafe extern "C" fn(*mut c_void, c_int, *const c_void, c_int, usize) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_memcpy_peer_async: unsafe { load_fn!(
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
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_pointer_get_attributes: unsafe { load_fn!(
                    lib,
                    "hipPointerGetAttributes",
                    unsafe extern "C" fn(*mut HipPointerAttribute, *const c_void) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_malloc: unsafe { load_fn!(
                    lib,
                    "hipMalloc",
                    unsafe extern "C" fn(*mut *mut c_void, usize) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_ext_malloc_with_flags: unsafe { load_optional_fn!(
                    lib,
                    "hipExtMallocWithFlags",
                    unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_free: unsafe { load_fn!(lib, "hipFree", unsafe extern "C" fn(*mut c_void) -> u32) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_host_malloc: unsafe { load_optional_fn!(
                    lib,
                    "hipHostMalloc",
                    unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_host_get_device_pointer: unsafe { load_optional_fn!(
                    lib,
                    "hipHostGetDevicePointer",
                    unsafe extern "C" fn(*mut *mut c_void, *mut c_void, c_uint) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_host_free: unsafe { load_optional_fn!(
                    lib,
                    "hipHostFree",
                    unsafe extern "C" fn(*mut c_void) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_mem_get_address_range: unsafe { load_fn!(
                    lib,
                    "hipMemGetAddressRange",
                    unsafe extern "C" fn(*mut *mut c_void, *mut usize, *mut c_void) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_memcpy: unsafe { load_fn!(
                    lib,
                    "hipMemcpy",
                    unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_uint) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_memcpy_async: unsafe { load_fn!(
                    lib,
                    "hipMemcpyAsync",
                    unsafe extern "C" fn(
                        *mut c_void,
                        *const c_void,
                        usize,
                        c_uint,
                        HipStream,
                    ) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_memcpy_dtod: unsafe { load_optional_fn!(
                    lib,
                    "hipMemcpyDtoD",
                    unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_memset: unsafe { load_fn!(
                    lib,
                    "hipMemset",
                    unsafe extern "C" fn(*mut c_void, c_int, usize) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_memset_async: unsafe { load_fn!(
                    lib,
                    "hipMemsetAsync",
                    unsafe extern "C" fn(*mut c_void, c_int, usize, HipStream) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_memset_d32_async: unsafe { load_fn!(
                    lib,
                    "hipMemsetD32Async",
                    unsafe extern "C" fn(*mut c_void, c_int, usize, HipStream) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_mem_address_reserve: unsafe { load_optional_fn!(
                    lib,
                    "hipMemAddressReserve",
                    unsafe extern "C" fn(*mut *mut c_void, usize, usize, *mut c_void, u64) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_mem_create: unsafe { load_optional_fn!(
                    lib,
                    "hipMemCreate",
                    unsafe extern "C" fn(
                        *mut HipMemGenericAllocationHandle,
                        usize,
                        *const HipMemAllocationProp,
                        u64,
                    ) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_mem_get_allocation_granularity: unsafe { load_optional_fn!(
                    lib,
                    "hipMemGetAllocationGranularity",
                    unsafe extern "C" fn(*mut usize, *const HipMemAllocationProp, u32) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_mem_get_handle_properties: unsafe { load_optional_fn!(
                    lib,
                    "hipMemGetAllocationPropertiesFromHandle",
                    unsafe extern "C" fn(
                        *mut HipMemAllocationProp,
                        HipMemGenericAllocationHandle,
                    ) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_mem_map: unsafe { load_optional_fn!(
                    lib,
                    "hipMemMap",
                    unsafe extern "C" fn(
                        *mut c_void,
                        usize,
                        usize,
                        HipMemGenericAllocationHandle,
                        u64,
                    ) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_mem_unmap: unsafe { load_optional_fn!(
                    lib,
                    "hipMemUnmap",
                    unsafe extern "C" fn(*mut c_void, usize) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_mem_set_access: unsafe { load_optional_fn!(
                    lib,
                    "hipMemSetAccess",
                    unsafe extern "C" fn(*mut c_void, usize, *const HipMemAccessDesc, usize) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_mem_release: unsafe { load_optional_fn!(
                    lib,
                    "hipMemRelease",
                    unsafe extern "C" fn(HipMemGenericAllocationHandle) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_stream_create: unsafe { load_fn!(
                    lib,
                    "hipStreamCreate",
                    unsafe extern "C" fn(*mut HipStream) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_stream_create_with_flags: unsafe { load_fn!(
                    lib,
                    "hipStreamCreateWithFlags",
                    unsafe extern "C" fn(*mut HipStream, c_uint) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_stream_synchronize: unsafe { load_fn!(
                    lib,
                    "hipStreamSynchronize",
                    unsafe extern "C" fn(HipStream) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_stream_destroy: unsafe { load_fn!(
                    lib,
                    "hipStreamDestroy",
                    unsafe extern "C" fn(HipStream) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_module_load: unsafe { load_fn!(
                    lib,
                    "hipModuleLoad",
                    unsafe extern "C" fn(*mut HipModule, *const c_char) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_module_load_data: unsafe { load_fn!(
                    lib,
                    "hipModuleLoadData",
                    unsafe extern "C" fn(*mut HipModule, *const c_void) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_module_get_function: unsafe { load_fn!(
                    lib,
                    "hipModuleGetFunction",
                    unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_module_launch_kernel: unsafe { load_fn!(
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
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_module_occupancy_max_active_blocks: unsafe { load_optional_fn!(
                    lib,
                    "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
                    unsafe extern "C" fn(*mut c_int, HipFunction, c_int, usize) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_event_create: unsafe { load_fn!(
                    lib,
                    "hipEventCreate",
                    unsafe extern "C" fn(*mut HipEvent) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_event_create_with_flags: unsafe { load_fn!(
                    lib,
                    "hipEventCreateWithFlags",
                    unsafe extern "C" fn(*mut HipEvent, c_uint) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_event_record: unsafe { load_fn!(
                    lib,
                    "hipEventRecord",
                    unsafe extern "C" fn(HipEvent, HipStream) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_event_query: unsafe { load_fn!(
                    lib,
                    "hipEventQuery",
                    unsafe extern "C" fn(HipEvent) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_event_synchronize: unsafe { load_fn!(
                    lib,
                    "hipEventSynchronize",
                    unsafe extern "C" fn(HipEvent) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_event_elapsed_time: unsafe { load_fn!(
                    lib,
                    "hipEventElapsedTime",
                    unsafe extern "C" fn(*mut f32, HipEvent, HipEvent) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_event_destroy: unsafe { load_fn!(
                    lib,
                    "hipEventDestroy",
                    unsafe extern "C" fn(HipEvent) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_stream_wait_event: unsafe { load_fn!(
                    lib,
                    "hipStreamWaitEvent",
                    unsafe extern "C" fn(HipStream, HipEvent, c_uint) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_get_error_string: unsafe { load_fn!(
                    lib,
                    "hipGetErrorString",
                    unsafe extern "C" fn(u32) -> *const i8
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_get_last_error: unsafe { load_fn!(lib, "hipGetLastError", unsafe extern "C" fn() -> u32) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_stream_begin_capture: unsafe { load_fn!(
                    lib,
                    "hipStreamBeginCapture",
                    unsafe extern "C" fn(HipStream, c_uint) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_stream_end_capture: unsafe { load_fn!(
                    lib,
                    "hipStreamEndCapture",
                    unsafe extern "C" fn(HipStream, *mut HipGraph) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_stream_is_capturing: unsafe { load_fn!(
                    lib,
                    "hipStreamIsCapturing",
                    unsafe extern "C" fn(HipStream, *mut c_uint) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_graph_instantiate: unsafe { load_fn!(
                    lib,
                    "hipGraphInstantiate",
                    unsafe extern "C" fn(
                        *mut HipGraphExec,
                        HipGraph,
                        *mut HipGraph,
                        *mut c_void,
                        usize,
                    ) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_graph_launch: unsafe { load_fn!(
                    lib,
                    "hipGraphLaunch",
                    unsafe extern "C" fn(HipGraphExec, HipStream) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_graph_exec_destroy: unsafe { load_fn!(
                    lib,
                    "hipGraphExecDestroy",
                    unsafe extern "C" fn(HipGraphExec) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_graph_destroy: unsafe { load_fn!(
                    lib,
                    "hipGraphDestroy",
                    unsafe extern "C" fn(HipGraph) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_stream_write_value32: unsafe { load_fn!(
                    lib,
                    "hipStreamWriteValue32",
                    unsafe extern "C" fn(HipStream, *mut c_void, u32, c_uint) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_stream_wait_value32: unsafe { load_optional_fn!(
                    lib,
                    "hipStreamWaitValue32",
                    unsafe extern "C" fn(HipStream, *mut c_void, u32, c_uint, u32) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_device_synchronize: unsafe { load_fn!(
                    lib,
                    "hipDeviceSynchronize",
                    unsafe extern "C" fn() -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_get_device_properties: unsafe { load_fn!(
                    lib,
                    "hipGetDeviceProperties",
                    unsafe extern "C" fn(*mut u8, c_int) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_get_device_attribute: unsafe { load_fn!(
                    lib,
                    "hipDeviceGetAttribute",
                    unsafe extern "C" fn(*mut c_int, c_int, c_int) -> u32
                ) },
                // SAFETY: resolve one HIP symbol from the live `Library`; pointer stays valid while `_lib` owns the mapping.
                fn_mem_get_info: unsafe { load_fn!(
                    lib,
                    "hipMemGetInfo",
                    unsafe extern "C" fn(*mut usize, *mut usize) -> u32
                ) },
                _lib: lib,
                launch_grid_yz_limit: AtomicU32::new(u32::MAX),
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

    /// Get HIP runtime version as (major, minor). E.g. ROCm 6.3 → (6, 3).
    pub fn runtime_version(&self) -> HipResult<(i32, i32)> {
        let mut version: c_int = 0;
        // SAFETY: HIP runtime symbol is live from HipRuntime::load; out-params are
        // stack locals (or caller-provided device ids). No host pointer aliasing.
        let code = unsafe { (self.fn_runtime_get_version)(&mut version) };
        self.check(code, "hipRuntimeGetVersion")?;
        // HIP version encoding: major * 10000000 + minor * 100000 + patch
        let major = version / 10_000_000;
        let minor = (version % 10_000_000) / 100_000;
        Ok((major, minor))
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
        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: FFI out-params are stack locals with valid writable provenance;
        // HIP symbols were resolved at load. Returned pointers are opaque to Rust.
        let code = unsafe { (self.fn_malloc)(&mut ptr, size) };
        self.check(code, "hipMalloc")?;
        crate::registry::record(ptr as usize, size, crate::registry::AllocationKind::Malloc);
        Ok(DeviceBuffer {
            ptr,
            size,
            ownership: crate::DeviceBufferOwnership::HipMalloc,
        })
    }

    /// Allocate system-visible signal memory for stream wait/write operations.
    ///
    /// This is an optional ROCm capability. Loading `HipRuntime` remains
    /// compatible with runtimes that do not export `hipExtMallocWithFlags`;
    /// callers receive a scoped error only when they request signal memory.
    pub fn malloc_signal(&self, size: usize) -> HipResult<DeviceBuffer> {
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
        })
    }

    /// # Safety
    /// Caller must ensure the buffer is not in use by any pending GPU operations.
    pub fn free(&self, buf: DeviceBuffer) -> HipResult<()> {
        if !buf.is_hip_allocation() {
            return Err(HipError::new(
                0,
                "hipFree rejected a borrowed or VMM DeviceBuffer",
            ));
        }
        // SAFETY: buf.ptr is a live hipMalloc allocation (ownership checked above);
        // caller must ensure GPU work on it is quiesced (documented on free).
        let code = unsafe { (self.fn_free)(buf.ptr) };
        self.check(code, "hipFree")?;
        crate::registry::forget(buf.ptr as usize);
        Ok(())
    }

    /// Allocate host-pinned memory the GPU can read directly over PCIe.
    ///
    /// `flags` are the HIP host-malloc flags; bit 1 (`hipHostMallocMapped`) is
    /// what makes the pages reachable from device code, and the returned address
    /// is then the *host* pointer — use [`Self::host_get_device_pointer`] for the
    /// address to hand a kernel.
    pub fn host_malloc(&self, size: usize, flags: u32) -> HipResult<*mut c_void> {
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
        let func = self.missing_vmm_symbol("hipHostFree", self.fn_host_free)?;
        // SAFETY: resolved VMM/host HIP symbol; pointer/handle args meet the method's
        // # Safety or documented preconditions; out-params are stack locals.
        let code = unsafe { func(host) };
        self.check(code, "hipHostFree")
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
        let func = self.missing_vmm_symbol("hipMemCreate", self.fn_mem_create)?;
        let mut handle: HipMemGenericAllocationHandle = ptr::null_mut();
        // SAFETY: resolved VMM/host HIP symbol; pointer/handle args meet the method's
        // # Safety or documented preconditions; out-params are stack locals.
        let code = unsafe { func(&mut handle, size, prop as *const HipMemAllocationProp, 0) };
        self.check(code, "hipMemCreate")?;
        Ok(handle)
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
        let func = self.missing_vmm_symbol("hipMemMap", self.fn_mem_map)?;
        // SAFETY: resolved VMM/host HIP symbol; pointer/handle args meet the method's
        // # Safety or documented preconditions; out-params are stack locals.
        let code = unsafe { func(ptr, size, 0, handle, 0) };
        self.check(code, "hipMemMap")?;
        crate::registry::record(ptr as usize, size, crate::registry::AllocationKind::VmmChunk);
        Ok(())
    }

    /// # Safety
    /// `ptr..ptr+size` must describe a currently mapped VMM range.
    pub unsafe fn mem_unmap(&self, ptr: *mut c_void, size: usize) -> HipResult<()> {
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
        let func = self.missing_vmm_symbol("hipMemRelease", self.fn_mem_release)?;
        // SAFETY: resolved VMM/host HIP symbol; pointer/handle args meet the method's
        // # Safety or documented preconditions; out-params are stack locals.
        let code = unsafe { func(handle) };
        self.check(code, "hipMemRelease")
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
        let mut stream: HipStream = ptr::null_mut();
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_stream_create_with_flags)(&mut stream, 0x1) };
        self.check(code, "hipStreamCreateWithFlags")?;
        Ok(Stream(stream))
    }

    pub fn stream_synchronize(&self, stream: &Stream) -> HipResult<()> {
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

    pub fn stream_destroy(&self, stream: Stream) -> HipResult<()> {
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_stream_destroy)(stream.0) };
        self.check(code, "hipStreamDestroy")
    }

    // ── Modules & Kernels ───────────────────────────────────────

    pub fn module_load(&self, path: &str) -> HipResult<Module> {
        let c_path =
            CString::new(path).map_err(|_| HipError::new(1, "invalid path for module_load"))?;
        let mut module: HipModule = ptr::null_mut();
        // SAFETY: module/function handles are live HIP objects from earlier loads;
        // CString/image buffers outlive the call; launch arg validity is the
        // unsafe fn caller's contract when applicable.
        let code = unsafe { (self.fn_module_load)(&mut module, c_path.as_ptr()) };
        self.check(code, "hipModuleLoad")?;
        Ok(Module(module))
    }

    pub fn module_load_data(&self, image: &[u8]) -> HipResult<Module> {
        let mut module: HipModule = ptr::null_mut();
        let code =
            // SAFETY: module/function handles are live HIP objects from earlier loads;
            // CString/image buffers outlive the call; launch arg validity is the
            // unsafe fn caller's contract when applicable.
            unsafe { (self.fn_module_load_data)(&mut module, image.as_ptr() as *const c_void) };
        self.check(code, "hipModuleLoadData")?;
        Ok(Module(module))
    }

    pub fn module_get_function(&self, module: &Module, name: &str) -> HipResult<Function> {
        let c_name = CString::new(name)
            .map_err(|_| HipError::new(1, "invalid kernel name for module_get_function"))?;
        let mut func: HipFunction = ptr::null_mut();
        // SAFETY: module/function handles are live HIP objects from earlier loads;
        // CString/image buffers outlive the call; launch arg validity is the
        // unsafe fn caller's contract when applicable.
        let code = unsafe { (self.fn_module_get_function)(&mut func, module.0, c_name.as_ptr()) };
        self.check(code, "hipModuleGetFunction")?;
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
        self.validate_launch_grid(grid)?;
        if hip_fault_consume(1, "launch") {
            return Err(hip_fault_err("launch"));
        }
        let stream_raw = stream.map_or(ptr::null_mut(), |s| s.0);
        let t = std::time::Instant::now();
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
        crate::ffi::launch_counters::record(t.elapsed().as_nanos() as u64);
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
        crate::ffi::launch_counters::record(t.elapsed().as_nanos() as u64);
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
        let mut event: HipEvent = ptr::null_mut();
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_event_create_with_flags)(&mut event, flags) };
        self.check(code, "hipEventCreateWithFlags")?;
        Ok(Event(event))
    }

    pub fn event_record(&self, event: &Event, stream: Option<&Stream>) -> HipResult<()> {
        let stream_raw = stream.map_or(ptr::null_mut(), |s| s.0);
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_event_record)(event.0, stream_raw) };
        self.check(code, "hipEventRecord")
    }

    pub fn event_synchronize(&self, event: &Event) -> HipResult<()> {
        let t = std::time::Instant::now();
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_event_synchronize)(event.0) };
        crate::ffi::launch_counters::event_sync::record(t.elapsed().as_nanos() as u64);
        self.check(code, "hipEventSynchronize")
    }

    pub fn event_query(&self, event: &Event) -> HipResult<bool> {
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
        let mut ms: f32 = 0.0;
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_event_elapsed_time)(&mut ms, start.0, stop.0) };
        self.check(code, "hipEventElapsedTime")?;
        Ok(ms)
    }

    pub fn event_destroy(&self, event: Event) -> HipResult<()> {
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_event_destroy)(event.0) };
        self.check(code, "hipEventDestroy")
    }

    pub fn stream_wait_event(&self, stream: &Stream, event: &Event) -> HipResult<()> {
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_stream_wait_event)(stream.0, event.0, 0) };
        self.check(code, "hipStreamWaitEvent")
    }

    // ── Error query ─────────────────────────────────────────────

    pub fn last_error(&self) -> u32 {
        // SAFETY: HIP runtime symbol is live from HipRuntime::load; out-params are
        // stack locals (or caller-provided device ids). No host pointer aliasing.
        unsafe { (self.fn_get_last_error)() }
    }

    // ── Async memory ops ────────────────────────────────────────

    pub fn memcpy_htod_async(
        &self,
        dst: &DeviceBuffer,
        src: &[u8],
        stream: &Stream,
    ) -> HipResult<()> {
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

    /// Host-asynchronous H→D copy on the legacy/default stream: `src` must
    /// stay unchanged until the stream reaches the copy.
    pub fn memcpy_htod_async_default(&self, dst: &DeviceBuffer, src: &[u8]) -> HipResult<()> {
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
        stream: &Stream,
    ) -> HipResult<()> {
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
                stream.0,
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

    // ── Graph capture & replay ──────────────────────────────────

    /// Begin capturing all operations on `stream` into a graph.
    /// mode=0 is hipStreamCaptureModeGlobal.
    pub fn stream_begin_capture(&self, stream: &Stream, mode: u32) -> HipResult<()> {
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_stream_begin_capture)(stream.0, mode as c_uint) };
        self.check(code, "hipStreamBeginCapture")
    }

    /// End capture on `stream`, returning the captured graph.
    pub fn stream_end_capture(&self, stream: &Stream) -> HipResult<Graph> {
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
        let t = std::time::Instant::now();
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_graph_launch)(exec.0, stream.0) };
        crate::ffi::launch_counters::graph_launch::record(t.elapsed().as_nanos() as u64);
        self.check(code, "hipGraphLaunch")
    }

    pub fn graph_exec_destroy(&self, exec: GraphExec) -> HipResult<()> {
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_graph_exec_destroy)(exec.0) };
        self.check(code, "hipGraphExecDestroy")
    }

    pub fn graph_destroy(&self, graph: Graph) -> HipResult<()> {
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
        // SAFETY: stream/event/graph handles are live opaque HIP objects owned by the
        // wrappers; out-params are stack locals. Destroy/sync require the caller
        // to have drained dependent work first where HIP demands it.
        let code = unsafe { (self.fn_stream_write_value32)(stream.0, ptr.as_ptr(), value, flags) };
        self.check(code, "hipStreamWriteValue32")
    }

    /// Wait until a 32-bit signal-memory value satisfies the requested condition.
    /// Operations submitted later to `stream` remain blocked until it does.
    pub fn stream_wait_value32(
        &self,
        stream: &Stream,
        ptr: &DeviceBuffer,
        value: u32,
        flags: u32,
        mask: u32,
    ) -> HipResult<()> {
        let Some(wait_value32) = self.fn_stream_wait_value32 else {
            return Err(HipError::new(
                0,
                "hipStreamWaitValue32 unavailable; stream signal waits are unsupported",
            ));
        };
        // SAFETY: ptr is a live device signal/buffer address from DeviceBuffer; stream
        // is a live handle. Value wait/write races are ordered by the stream.
        let code = unsafe { wait_value32(stream.0, ptr.as_ptr(), value, flags, mask) };
        self.check(code, "hipStreamWaitValue32")
    }

    pub fn device_synchronize(&self) -> HipResult<()> {
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

/// Handle to a specific kernel function within a module.
pub struct Function(HipFunction);
// SAFETY: Function is an opaque HIP handle (*mut c_void). Sending the handle
// between threads is fine; it is not Sync. Concurrent use needs
// external stream synchronization; destroy only when idle.
unsafe impl Send for Function {}

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
