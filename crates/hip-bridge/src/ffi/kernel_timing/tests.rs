// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Native-call mock contract tests for the strict kernel timing collector.
//!
//! The pool wraps REAL resolved HIP entry points; these tests substitute the
//! function table with scriptable `extern "C"` shims so every closure path is
//! exercised without a GPU. No test uses a fake passing collector: every case
//! drives one genuine [`KernelTimingPool`] through its actual
//! reserve/drain/begin/finish/cancel/close state machine plus the real
//! [`LaunchTiming`](super::LaunchTiming) wrapper hooks or the real free guard.

use super::{KernelTimingPool, Pair, State, TimingControl};
use crate::error::HipResult;
use crate::ffi::{Event, Function, HipEvent, HipRuntime, HipStream, Stream, HIP_ERROR_NOT_READY};
use libloading::Library;
use std::ffi::{c_char, c_int, c_uint, c_void};
use std::ptr;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

// ── Mock native-call plan ─────────────────────────────────────────────────

/// Scriptable outcomes for one mock HIP table. Atomics keep the `extern "C"`
/// shims callable from the collector without aliasing mutable statics.
struct MockPlan {
    /// Device ordinal reported by `hipGetDevice`.
    device: AtomicU32,
    /// Next synthetic HIP event handle.
    next_event: AtomicU64,
    /// `hipEventRecord` failures before success.
    record_failures: AtomicU32,
    /// Fail exactly the Nth `hipEventRecord` call (1-based, `u32::MAX` = never).
    fail_record_at: AtomicU32,
    /// `hipEventSynchronize` failures before success.
    sync_failures: AtomicU32,
    /// `hipEventElapsedTime` failures before success.
    elapsed_failures: AtomicU32,
    /// Duration reported by `hipEventElapsedTime`.
    elapsed_ms: AtomicU32,
    /// `hipEventDestroy` failures before success.
    destroy_failures: AtomicU32,
    /// `hipStreamSynchronize` failures before success.
    stream_sync_failures: AtomicU32,
    /// `hipModuleLaunchKernel` status code.
    launch_code: AtomicU32,
    /// `hipStreamIsCapturing` capture status (0 = none, 1 = active).
    capturing: AtomicU32,
    /// `hipGetDevice` failures before success.
    device_failures: AtomicU32,
    /// `hipSetDevice` failures before success.
    select_failures: AtomicU32,
    /// Observed counters.
    records: AtomicU32,
    syncs: AtomicU32,
    elapseds: AtomicU32,
    destroys: AtomicU32,
    stream_syncs: AtomicU32,
    launches: AtomicU32,
    next_stream: AtomicU32,
}

impl MockPlan {
    const fn new() -> Self {
        Self {
            device: AtomicU32::new(0),
            next_event: AtomicU64::new(0x1000),
            record_failures: AtomicU32::new(0),
            fail_record_at: AtomicU32::new(u32::MAX),
            sync_failures: AtomicU32::new(0),
            elapsed_failures: AtomicU32::new(0),
            elapsed_ms: AtomicU32::new(0x3f800000),
            destroy_failures: AtomicU32::new(0),
            stream_sync_failures: AtomicU32::new(0),
            launch_code: AtomicU32::new(0),
            capturing: AtomicU32::new(0),
            device_failures: AtomicU32::new(0),
            select_failures: AtomicU32::new(0),
            records: AtomicU32::new(0),
            syncs: AtomicU32::new(0),
            elapseds: AtomicU32::new(0),
            destroys: AtomicU32::new(0),
            stream_syncs: AtomicU32::new(0),
            launches: AtomicU32::new(0),
            next_stream: AtomicU32::new(0),
        }
    }

    fn reset(&self) {
        self.device.store(0, Ordering::SeqCst);
        self.next_event.store(0x1000, Ordering::SeqCst);
        self.record_failures.store(0, Ordering::SeqCst);
        self.fail_record_at.store(u32::MAX, Ordering::SeqCst);
        self.sync_failures.store(0, Ordering::SeqCst);
        self.elapsed_failures.store(0, Ordering::SeqCst);
        self.elapsed_ms.store(0x3f800000, Ordering::SeqCst);
        self.destroy_failures.store(0, Ordering::SeqCst);
        self.stream_sync_failures.store(0, Ordering::SeqCst);
        self.launch_code.store(0, Ordering::SeqCst);
        self.capturing.store(0, Ordering::SeqCst);
        self.device_failures.store(0, Ordering::SeqCst);
        self.select_failures.store(0, Ordering::SeqCst);
        self.records.store(0, Ordering::SeqCst);
        self.syncs.store(0, Ordering::SeqCst);
        self.elapseds.store(0, Ordering::SeqCst);
        self.destroys.store(0, Ordering::SeqCst);
        self.stream_syncs.store(0, Ordering::SeqCst);
        self.launches.store(0, Ordering::SeqCst);
        self.next_stream.store(0, Ordering::SeqCst);
    }

    /// Consume one scripted failure; returns true when the call must fail.
    fn take_failure(&self, slot: &AtomicU32) -> bool {
        let mut current = slot.load(Ordering::SeqCst);
        loop {
            if current == 0 {
                return false;
            }
            match slot.compare_exchange_weak(
                current,
                current - 1,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return true,
                Err(next) => current = next,
            }
        }
    }
}

// Shared mock table; tests hold `SERIAL` so per-case scripts never race.
static PLAN: LazyLock<MockPlan> = LazyLock::new(MockPlan::new);
static SERIAL: Mutex<()> = Mutex::new(());

fn lock_plan() -> (std::sync::MutexGuard<'static, ()>, &'static MockPlan) {
    let guard = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    PLAN.reset();
    super::ACTIVE.with(|active| active.set(ptr::null_mut()));
    (guard, &PLAN)
}

// ── Mock shims ────────────────────────────────────────────────────────────

unsafe extern "C" fn mock_init(_flags: c_uint) -> u32 {
    0
}

unsafe extern "C" fn mock_get_version(version: *mut c_int) -> u32 {
    // SAFETY: caller passes a writable stack out-parameter.
    unsafe { *version = 602_00000 };
    0
}

unsafe extern "C" fn mock_get_count(count: *mut c_int) -> u32 {
    // SAFETY: caller passes a writable stack out-parameter.
    unsafe { *count = 1 };
    0
}

unsafe extern "C" fn mock_set_device(_device: c_int) -> u32 {
    if PLAN.take_failure(&PLAN.select_failures) {
        return 1;
    }
    0
}

unsafe extern "C" fn mock_set_device_flags(_flags: c_uint) -> u32 {
    0
}

unsafe extern "C" fn mock_get_device(device: *mut c_int) -> u32 {
    if PLAN.take_failure(&PLAN.device_failures) {
        return 1;
    }
    // SAFETY: caller passes a writable stack out-parameter.
    unsafe { *device = PLAN.device.load(Ordering::SeqCst) as c_int };
    0
}

unsafe extern "C" fn mock_uuid(_uuid: *mut u8, _device: c_int) -> u32 {
    0
}

unsafe extern "C" fn mock_pci(_buf: *mut c_char, _len: c_int, _device: c_int) -> u32 {
    0
}

unsafe extern "C" fn mock_can_access_peer(can: *mut c_int, _device: c_int, _peer: c_int) -> u32 {
    // SAFETY: caller passes a writable stack out-parameter.
    unsafe { *can = 0 };
    0
}

unsafe extern "C" fn mock_enable_peer(_peer: c_int, _flags: c_uint) -> u32 {
    0
}

unsafe extern "C" fn mock_memcpy_peer(
    _dst: *mut c_void,
    _dst_device: c_int,
    _src: *const c_void,
    _src_device: c_int,
    _size: usize,
) -> u32 {
    0
}

unsafe extern "C" fn mock_memcpy_peer_async(
    _dst: *mut c_void,
    _dst_device: c_int,
    _src: *const c_void,
    _src_device: c_int,
    _size: usize,
    _stream: HipStream,
) -> u32 {
    0
}

unsafe extern "C" fn mock_pointer_attrs(
    _attrs: *mut crate::ffi::HipPointerAttribute,
    _ptr: *const c_void,
) -> u32 {
    0
}

unsafe extern "C" fn mock_malloc(_ptr: *mut *mut c_void, _size: usize) -> u32 {
    1
}

unsafe extern "C" fn mock_free(_ptr: *mut c_void) -> u32 {
    0
}

unsafe extern "C" fn mock_mem_range(
    _base: *mut *mut c_void,
    _size: *mut usize,
    _ptr: *mut c_void,
) -> u32 {
    1
}

unsafe extern "C" fn mock_memcpy(
    _dst: *mut c_void,
    _src: *const c_void,
    _size: usize,
    _kind: c_uint,
) -> u32 {
    0
}

unsafe extern "C" fn mock_memcpy_async(
    _dst: *mut c_void,
    _src: *const c_void,
    _size: usize,
    _kind: c_uint,
    _stream: HipStream,
) -> u32 {
    0
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn mock_memcpy_2d(
    _dst: *mut c_void,
    _dpitch: usize,
    _src: *const c_void,
    _spitch: usize,
    _width: usize,
    _height: usize,
    _kind: c_uint,
    _stream: HipStream,
) -> u32 {
    0
}

unsafe extern "C" fn mock_memset(_ptr: *mut c_void, _value: c_int, _size: usize) -> u32 {
    0
}

unsafe extern "C" fn mock_memset_async(
    _ptr: *mut c_void,
    _value: c_int,
    _size: usize,
    _stream: HipStream,
) -> u32 {
    0
}

unsafe extern "C" fn mock_memset_d32(
    _ptr: *mut c_void,
    _value: c_int,
    _size: usize,
    _stream: HipStream,
) -> u32 {
    0
}

unsafe extern "C" fn mock_stream_create(stream: *mut HipStream) -> u32 {
    let id = PLAN.next_stream.fetch_add(1, Ordering::SeqCst) as u64 + 0x2000;
    // SAFETY: caller passes a writable stack out-parameter.
    unsafe { *stream = id as HipStream };
    0
}

unsafe extern "C" fn mock_stream_create_flags(stream: *mut HipStream, _flags: c_uint) -> u32 {
    mock_stream_create(stream)
}

unsafe extern "C" fn mock_stream_sync(_stream: HipStream) -> u32 {
    PLAN.stream_syncs.fetch_add(1, Ordering::SeqCst);
    if PLAN.take_failure(&PLAN.stream_sync_failures) {
        return 1;
    }
    0
}

unsafe extern "C" fn mock_stream_destroy(_stream: HipStream) -> u32 {
    0
}

unsafe extern "C" fn mock_module_load(_module: *mut HipStream, _path: *const c_char) -> u32 {
    1
}

unsafe extern "C" fn mock_module_load_data(_module: *mut HipStream, _image: *const c_void) -> u32 {
    1
}

unsafe extern "C" fn mock_module_get_function(
    _func: *mut HipStream,
    _module: HipStream,
    _name: *const c_char,
) -> u32 {
    1
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn mock_launch(
    _func: *mut c_void,
    _gx: c_uint,
    _gy: c_uint,
    _gz: c_uint,
    _bx: c_uint,
    _by: c_uint,
    _bz: c_uint,
    _smem: c_uint,
    _stream: HipStream,
    _params: *mut *mut c_void,
    _extra: *mut *mut c_void,
) -> u32 {
    PLAN.launches.fetch_add(1, Ordering::SeqCst);
    PLAN.launch_code.load(Ordering::SeqCst)
}

unsafe extern "C" fn mock_event_create(event: *mut HipEvent) -> u32 {
    let id = PLAN.next_event.fetch_add(1, Ordering::SeqCst);
    // SAFETY: caller passes a writable stack out-parameter.
    unsafe { *event = id as HipEvent };
    0
}

unsafe extern "C" fn mock_event_create_flags(event: *mut HipEvent, _flags: c_uint) -> u32 {
    mock_event_create(event)
}

unsafe extern "C" fn mock_event_record(_event: HipEvent, _stream: HipStream) -> u32 {
    let observed = PLAN.records.fetch_add(1, Ordering::SeqCst) + 1;
    if observed == PLAN.fail_record_at.load(Ordering::SeqCst) {
        return 1;
    }
    if PLAN.take_failure(&PLAN.record_failures) {
        return 1;
    }
    0
}

unsafe extern "C" fn mock_event_query(_event: HipEvent) -> u32 {
    0
}

unsafe extern "C" fn mock_event_sync(_event: HipEvent) -> u32 {
    PLAN.syncs.fetch_add(1, Ordering::SeqCst);
    if PLAN.take_failure(&PLAN.sync_failures) {
        return 1;
    }
    0
}

unsafe extern "C" fn mock_event_elapsed(ms: *mut f32, _start: HipEvent, _stop: HipEvent) -> u32 {
    PLAN.elapseds.fetch_add(1, Ordering::SeqCst);
    if PLAN.take_failure(&PLAN.elapsed_failures) {
        return 1;
    }
    // SAFETY: caller passes a writable stack out-parameter.
    unsafe {
        *ms = f32::from_bits(PLAN.elapsed_ms.load(Ordering::SeqCst));
    }
    0
}

unsafe extern "C" fn mock_event_destroy(_event: HipEvent) -> u32 {
    PLAN.destroys.fetch_add(1, Ordering::SeqCst);
    if PLAN.take_failure(&PLAN.destroy_failures) {
        return 1;
    }
    0
}

/// Drive one genuine launch through the real wrapper funnel.
fn drive_launch(runtime: &HipRuntime, stream_raw: HipStream) -> HipResult<()> {
    // `Function`/`Stream` are tuple structs whose fields are visible to this
    // descendant of `ffi`; the mock launch shim ignores every argument.
    let func = Function(stream_raw);
    let stream = Stream(stream_raw);
    let mut params: [*mut c_void; 1] = [ptr::null_mut()];
    // SAFETY: mock launch shim ignores kernel, grid, and params; stream and
    // params are live mock handles owned by the caller.
    unsafe { runtime.launch_kernel(&func, [1, 1, 1], [32, 1, 1], 0, Some(&stream), &mut params) }
}

unsafe extern "C" fn mock_stream_wait(_stream: HipStream, _event: HipEvent, _flags: c_uint) -> u32 {
    0
}

unsafe extern "C" fn mock_error_string(_code: u32) -> *const i8 {
    c"mock".as_ptr()
}

unsafe extern "C" fn mock_last_error() -> u32 {
    0
}

unsafe extern "C" fn mock_begin_capture(_stream: HipStream, _mode: c_uint) -> u32 {
    0
}

unsafe extern "C" fn mock_end_capture(_stream: HipStream, _graph: *mut HipStream) -> u32 {
    0
}

unsafe extern "C" fn mock_is_capturing(_stream: HipStream, status: *mut c_uint) -> u32 {
    // SAFETY: caller passes a writable stack out-parameter.
    unsafe { *status = PLAN.capturing.load(Ordering::SeqCst) };
    0
}

unsafe extern "C" fn mock_graph_instantiate(
    _exec: *mut HipStream,
    _graph: HipStream,
    _log: *mut HipStream,
    _params: *mut c_void,
    _size: usize,
) -> u32 {
    1
}

unsafe extern "C" fn mock_graph_launch(_exec: HipStream, _stream: HipStream) -> u32 {
    1
}

unsafe extern "C" fn mock_graph_exec_destroy(_exec: HipStream) -> u32 {
    0
}

unsafe extern "C" fn mock_graph_destroy(_graph: HipStream) -> u32 {
    0
}

unsafe extern "C" fn mock_write_value(
    _stream: HipStream,
    _ptr: *mut c_void,
    _value: u32,
    _flags: c_uint,
) -> u32 {
    0
}

unsafe extern "C" fn mock_device_sync() -> u32 {
    0
}

unsafe extern "C" fn mock_device_props(_props: *mut u8, _device: c_int) -> u32 {
    0
}

unsafe extern "C" fn mock_device_attr(_value: *mut c_int, _attr: c_int, _device: c_int) -> u32 {
    0
}

unsafe extern "C" fn mock_mem_info(_free: *mut usize, _total: *mut usize) -> u32 {
    0
}

// ── Mock runtime construction ─────────────────────────────────────────────

/// Build a [`HipRuntime`] whose function table is the scriptable mock set
/// above. Only the collector-relevant entries are scripted; every other
/// entry returns a neutral success/failure that never touches hardware.
fn mock_runtime() -> HipRuntime {
    // `Library::this` borrows the test binary's own handle; the
    // function-pointer fields below are mock shims, not dlsym lookups, so no
    // symbol resolution ever observes the borrowed handle.
    let lib: Library = libloading::os::unix::Library::this().into();
    HipRuntime {
        _lib: Arc::new(lib),
        fn_init: mock_init,
        fn_runtime_get_version: mock_get_version,
        fn_get_device_count: mock_get_count,
        fn_set_device: mock_set_device,
        fn_set_device_flags: mock_set_device_flags,
        fn_get_device: mock_get_device,
        fn_device_get_uuid: mock_uuid,
        fn_device_get_pci_bus_id: mock_pci,
        fn_device_can_access_peer: mock_can_access_peer,
        fn_device_enable_peer_access: mock_enable_peer,
        fn_memcpy_peer: mock_memcpy_peer,
        fn_memcpy_peer_async: mock_memcpy_peer_async,
        fn_pointer_get_attributes: mock_pointer_attrs,
        fn_malloc: mock_malloc,
        fn_ext_malloc_with_flags: None,
        fn_free: mock_free,
        fn_host_malloc: None,
        fn_host_get_device_pointer: None,
        fn_host_free: None,
        fn_mem_get_address_range: mock_mem_range,
        fn_memcpy: mock_memcpy,
        fn_memcpy_async: mock_memcpy_async,
        fn_memcpy_2d_async: mock_memcpy_2d,
        fn_memcpy_dtod: None,
        fn_memset: mock_memset,
        fn_memset_async: mock_memset_async,
        fn_memset_d32_async: mock_memset_d32,
        fn_mem_address_reserve: None,
        fn_mem_create: None,
        fn_mem_get_allocation_granularity: None,
        fn_mem_get_handle_properties: None,
        fn_mem_map: None,
        fn_mem_unmap: None,
        fn_mem_set_access: None,
        fn_mem_release: None,
        fn_mem_export_to_shareable_handle: None,
        fn_mem_import_from_shareable_handle: None,
        fn_mem_get_access: None,
        fn_stream_create: mock_stream_create,
        fn_stream_create_with_flags: mock_stream_create_flags,
        fn_stream_synchronize: mock_stream_sync,
        fn_stream_destroy: mock_stream_destroy,
        fn_module_load: mock_module_load,
        fn_module_load_data: mock_module_load_data,
        fn_module_get_function: mock_module_get_function,
        fn_module_launch_kernel: mock_launch,
        fn_module_occupancy_max_active_blocks: None,
        fn_event_create: mock_event_create,
        fn_event_create_with_flags: mock_event_create_flags,
        fn_event_record: mock_event_record,
        fn_event_query: mock_event_query,
        fn_event_synchronize: mock_event_sync,
        fn_event_elapsed_time: mock_event_elapsed,
        fn_event_destroy: mock_event_destroy,
        fn_stream_wait_event: mock_stream_wait,
        fn_get_error_string: mock_error_string,
        fn_get_last_error: mock_last_error,
        fn_stream_begin_capture: mock_begin_capture,
        fn_stream_end_capture: mock_end_capture,
        fn_stream_is_capturing: mock_is_capturing,
        fn_graph_instantiate: mock_graph_instantiate,
        fn_graph_launch: mock_graph_launch,
        fn_graph_exec_destroy: mock_graph_exec_destroy,
        fn_graph_destroy: mock_graph_destroy,
        fn_stream_write_value32: mock_write_value,
        fn_stream_wait_value32: None,
        fn_device_synchronize: mock_device_sync,
        fn_get_device_properties: mock_device_props,
        fn_get_device_attribute: mock_device_attr,
        fn_mem_get_info: mock_mem_info,
        launch_grid_yz_limit: AtomicU32::new(u32::MAX),
        own_device_bytes: AtomicUsize::new(0),
        own_vmm_handles: Default::default(),
        stream_wait_caps: std::array::from_fn(|_| AtomicU8::new(0)),
        kernel_timing_control: std::sync::OnceLock::new(),
    }
}

fn mock_stream(raw: u64) -> HipStream {
    raw as HipStream
}

/// Allocate a pool against the shared mock plan. The returned runtime owns a
/// fresh [`TimingControl`], so each test gets an isolated control identity.
fn pool_with_plan(capacity: usize) -> (HipRuntime, KernelTimingPool) {
    let runtime = mock_runtime();
    let destroys_before = PLAN.destroys.load(Ordering::SeqCst);
    let pool = runtime
        .kernel_timing_pool(capacity)
        .expect("mock pool allocates");
    assert_eq!(
        PLAN.destroys.load(Ordering::SeqCst),
        destroys_before,
        "preallocation must not destroy events"
    );
    (runtime, pool)
}

// ── Pure state semantics ──────────────────────────────────────────────────

#[test]
fn reserve_assigns_slots_without_allocation() {
    let (_serial, _plan) = lock_plan();
    let control = Arc::new(TimingControl::default());
    let mut state = State {
        control,
        device: 0,
        pairs: vec![
            Pair {
                start: Event(ptr::null_mut()),
                stop: Event(ptr::null_mut()),
                stream: ptr::null_mut(),
                started: false,
                completed: false,
            },
            Pair {
                start: Event(ptr::null_mut()),
                stop: Event(ptr::null_mut()),
                stream: ptr::null_mut(),
                started: false,
                completed: false,
            },
        ],
        used: 0,
        in_flight: false,
        failure: None,
    };
    let first = state.reserve(mock_stream(1)).expect("first slot");
    let second = state.reserve(mock_stream(2)).expect("second slot");
    assert_eq!((first, second), (0, 1));
    assert_eq!(state.used, 2);
    assert!(state
        .pairs
        .iter()
        .all(|pair| !pair.started && !pair.completed));
    assert!(
        state.reserve(mock_stream(3)).is_err(),
        "exhaustion must fail"
    );
    assert!(state.failure.is_some(), "exhaustion poisons the scope");
    assert!(
        state.reserve(mock_stream(4)).is_err(),
        "poisoned scope stays failed"
    );
}

#[test]
fn state_fail_keeps_first_error() {
    let (_serial, _plan) = lock_plan();
    let control = Arc::new(TimingControl::default());
    let mut state = State {
        control,
        device: 0,
        pairs: Vec::new(),
        used: 0,
        in_flight: false,
        failure: None,
    };
    state.fail(crate::error::HipError::new(7, "first"));
    state.fail(crate::error::HipError::new(9, "second"));
    assert_eq!(state.failure.as_ref().expect("failure").code, 7);
}

// ── Fixture flag semantics ────────────────────────────────────────────────

#[test]
fn in_flight_blocks_finish() {
    let (_serial, _plan) = lock_plan();
    let (runtime, mut pool) = pool_with_plan(2);
    pool.begin().expect("begin");
    // Simulate one wrapper that recorded start but never paired stop.
    super::ACTIVE.with(|active| {
        let state = active.get();
        assert!(!state.is_null());
        // SAFETY: TLS cell points at this pool's live stable boxed state.
        unsafe { (*state).in_flight = true };
    });
    assert!(pool.finish().is_err(), "unpaired launch must not report");
    assert!(!pool.is_active());
    drop(runtime);
}

#[test]
fn started_completed_flags_gate_report() {
    let (_serial, _plan) = lock_plan();
    let (runtime, mut pool) = pool_with_plan(2);
    pool.begin().expect("begin");
    super::ACTIVE.with(|active| {
        let state = active.get();
        // SAFETY: TLS cell points at this pool's live stable boxed state.
        unsafe { (*state).used = 1 };
        // SAFETY: same live state; the pairs vector is stable and exclusive.
        unsafe { (&mut (*state).pairs)[0].started = true };
        // SAFETY: same live state; no native call while borrowed.
        unsafe { (&mut (*state).pairs)[0].completed = false };
    });
    assert!(
        pool.finish().is_err(),
        "started-but-uncompleted pair must not report"
    );
    drop(runtime);
}

// ── Exact launch/event behavior ───────────────────────────────────────────

#[test]
fn successful_launch_pair_reports_mock_duration() {
    let (_serial, plan) = lock_plan();
    plan.elapsed_ms.store(0x40400000, Ordering::SeqCst); // 3.0 ms
    let (runtime, mut pool) = pool_with_plan(1);
    assert!(!runtime.kernel_timing_is_active());
    pool.begin().expect("begin");
    assert!(pool.is_active());
    assert!(runtime.kernel_timing_is_active());
    drive_launch(&runtime, mock_stream(0x3000)).expect("launch");
    assert_eq!(plan.records.load(Ordering::SeqCst), 2, "start + stop");
    let report = pool.finish().expect("finish");
    assert_eq!(report.kernel_count, 1);
    assert_eq!(report.per_kernel_ms.len(), 1);
    assert!((report.per_kernel_ms[0] - 3.0).abs() < f32::EPSILON);
    assert_eq!(
        report.coverage,
        super::KernelTimingCoverage::EagerHipLaunchWrappers
    );
    assert_eq!(plan.elapseds.load(Ordering::SeqCst), 1);
    assert_eq!(
        plan.syncs.load(Ordering::SeqCst),
        1,
        "paired stop is synced"
    );
    // Explicit close observes destruction of exactly the preallocated pair.
    let destroys_before = plan.destroys.load(Ordering::SeqCst);
    pool.close().expect("close");
    assert_eq!(plan.destroys.load(Ordering::SeqCst) - destroys_before, 2);
    drop(runtime);
}

#[test]
fn record_start_failure_rejects_launch() {
    let (_serial, plan) = lock_plan();
    plan.record_failures.store(1, Ordering::SeqCst);
    let (runtime, mut pool) = pool_with_plan(1);
    pool.begin().expect("begin");
    assert!(drive_launch(&runtime, mock_stream(0x3000)).is_err());
    assert!(pool.finish().is_err(), "failed start poisons finish");
    drop(runtime);
}

#[test]
fn record_stop_failure_rejects_and_drains() {
    let (_serial, plan) = lock_plan();
    // Fail exactly the stop record of the first launch: start OK, stop fails.
    plan.fail_record_at.store(2, Ordering::SeqCst);
    let (runtime, mut pool) = pool_with_plan(1);
    pool.begin().expect("begin");
    assert!(drive_launch(&runtime, mock_stream(0x3000)).is_err());
    assert!(pool.finish().is_err());
    drop(runtime);
}

#[test]
fn kernel_failure_wins_over_timing_error() {
    let (_serial, plan) = lock_plan();
    plan.launch_code.store(1, Ordering::SeqCst);
    let (runtime, mut pool) = pool_with_plan(1);
    pool.begin().expect("begin");
    let err = drive_launch(&runtime, mock_stream(0x3000)).expect_err("native fails");
    assert!(
        err.message.contains("hipModuleLaunchKernel"),
        "native launch error takes precedence, got: {err}"
    );
    // The pool still refuses a report: the failed launch poisoned the scope.
    assert!(pool.finish().is_err());
    drop(runtime);
}

#[test]
fn runtime_thread_mismatch_rejects() {
    let (_serial, _plan) = lock_plan();
    let (runtime, mut pool) = pool_with_plan(1);
    pool.begin().expect("begin");
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                assert!(drive_launch(&runtime, mock_stream(0x3000)).is_err());
            })
            .join()
            .expect("thread");
    });
    // Cross-thread probe invalidated the claim.
    assert!(pool.finish().is_err());
    drop(pool);
    drop(runtime);
}

#[test]
fn nesting_rejected_and_poisoned() {
    let (_serial, _plan) = lock_plan();
    let (runtime, mut pool) = pool_with_plan(1);
    pool.begin().expect("begin");
    assert!(pool.begin().is_err(), "nested begin rejected");
    assert!(pool.finish().is_err(), "nesting poisons the scope");
    drop(runtime);
}

// ── Free TLS guard ────────────────────────────────────────────────────────

#[test]
fn free_guard_inactive_returns_ok() {
    let (_serial, _plan) = lock_plan();
    assert!(super::reject_kernel_timing_backend("actual backend").is_ok());
}

#[test]
fn free_guard_active_poisons_and_rejects() {
    let (_serial, _plan) = lock_plan();
    let (runtime, mut pool) = pool_with_plan(1);
    pool.begin().expect("begin");
    let err = super::reject_kernel_timing_backend("actual backend").expect_err("reject");
    assert!(err.message.contains("actual backend"));
    assert!(pool.finish().is_err(), "backend work invalidates the claim");
    drop(runtime);
}

// ── Allocation lifecycle: no per-kernel allocation ────────────────────────

#[test]
fn zero_capacity_rejected() {
    let (_serial, _plan) = lock_plan();
    let runtime = mock_runtime();
    assert!(runtime.kernel_timing_pool(0).is_err());
    drop(runtime);
}

#[test]
fn pool_exhaustion_fails_closed() {
    let (_serial, _plan) = lock_plan();
    let (runtime, mut pool) = pool_with_plan(1);
    pool.begin().expect("begin");
    drive_launch(&runtime, mock_stream(0x3000)).expect("first");
    assert!(
        drive_launch(&runtime, mock_stream(0x3001)).is_err(),
        "second launch exceeds capacity"
    );
    assert!(pool.finish().is_err(), "exhaustion poisons the scope");
    // A poisoned pool cannot silently retry: begin refuses.
    assert!(pool.begin().is_err());
    drop(runtime);
}

#[test]
fn cancel_then_close_without_report() {
    let (_serial, _plan) = lock_plan();
    let (runtime, mut pool) = pool_with_plan(1);
    pool.begin().expect("begin");
    drive_launch(&runtime, mock_stream(0x3000)).expect("launch");
    pool.cancel().expect("cancel drains");
    assert!(!pool.is_active());
    assert!(pool.finish().is_err(), "cancelled scope never reports");
    pool.close().expect("close after cancel");
    drop(runtime);
}

#[test]
fn active_pool_rejects_second_pool_and_backend() {
    let (_serial, _plan) = lock_plan();
    let (runtime, mut pool) = pool_with_plan(1);
    pool.begin().expect("begin");
    assert!(runtime.kernel_timing_pool(1).is_err());
    assert!(runtime.kernel_timing_reject_backend("actual PM4").is_err());
    assert!(
        pool.finish().is_err(),
        "backend rejection invalidates claim"
    );
    drop(runtime);
}

#[test]
fn close_destroy_failure_reaches_caller() {
    let (_serial, plan) = lock_plan();
    let (runtime, pool) = pool_with_plan(1);
    plan.destroy_failures.store(1, Ordering::SeqCst);
    assert!(pool.close().is_err(), "destroy error must reach close");
    drop(runtime);
}

#[test]
fn drain_sync_failure_fails_finish() {
    let (_serial, plan) = lock_plan();
    let (runtime, mut pool) = pool_with_plan(1);
    pool.begin().expect("begin");
    drive_launch(&runtime, mock_stream(0x3000)).expect("launch");
    plan.sync_failures.store(1, Ordering::SeqCst);
    assert!(pool.finish().is_err(), "drain sync failure fails finish");
    drop(runtime);
}

#[test]
fn capture_mode_rejects_launch() {
    let (_serial, plan) = lock_plan();
    plan.capturing.store(1, Ordering::SeqCst);
    let (runtime, mut pool) = pool_with_plan(1);
    pool.begin().expect("begin");
    assert!(drive_launch(&runtime, mock_stream(0x3000)).is_err());
    plan.capturing.store(0, Ordering::SeqCst);
    assert!(pool.finish().is_err());
    drop(runtime);
}

#[test]
fn device_change_rejects_launch() {
    let (_serial, plan) = lock_plan();
    let (runtime, mut pool) = pool_with_plan(1);
    pool.begin().expect("begin");
    plan.device.store(1, Ordering::SeqCst);
    assert!(drive_launch(&runtime, mock_stream(0x3000)).is_err());
    plan.device.store(0, Ordering::SeqCst);
    assert!(pool.finish().is_err());
    drop(runtime);
}

#[test]
fn not_ready_code_is_not_success() {
    let (_serial, _plan) = lock_plan();
    assert_ne!(HIP_ERROR_NOT_READY, 0);
}
