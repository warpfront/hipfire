// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Strict preallocated event spans for the two eager HIP launch wrappers.
//!
//! Allocate the pool before the measured host window; begin, issue eager work,
//! then finish on the same thread. Each event pair brackets exactly one successful
//! HIP kernel launch on its actual stream. Uploads, JIT and snapshots are outside
//! the pairs. Raw GPU event milliseconds are not host wall time; summing streams
//! measures work, not critical-path latency. Event resolution/overhead still apply.
//!
//! Coverage is NOT a whole-backend claim: PM4, AQL, rocBLAS and calls bypassing the
//! wrappers are unsupported. The owner MUST admit an eager HIP-only route first.
//! Graph capture/replay through this runtime is rejected while active. Concurrent
//! same-runtime launches from other threads invalidate rather than vanish.
//!
//! Pool ownership is independent of `Gpu`: it owns the library mapping, event API
//! pointers and stable boxed state. No borrow of `Gpu` survives begin, and there is
//! no self-reference. The pool is thread-bound. Keep streams alive until finish.
//! `cancel`/Drop detach and drain an abandoned scope; only finish returns a report.
//! Call `close` outside the host window to observe event destruction errors.

use super::{Event, HipError, HipEvent, HipResult, HipRuntime, HipStream, Stream};
use libloading::Library;
use std::cell::Cell;
use std::ffi::c_int;
use std::marker::PhantomData;
use std::ptr;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

thread_local! {
    // Stable pointee belongs to the thread-bound pool until detach or Drop.
    static ACTIVE: Cell<*mut State> = const { Cell::new(ptr::null_mut()) };
}

const VIOLATION: usize = 1;

fn thread_token() -> usize {
    ACTIVE.with(|slot| slot as *const Cell<*mut State> as usize)
}

fn error(message: &str) -> HipError {
    HipError::new(0, message)
}

/// Runtime identity survives moves of HipRuntime and retains cross-thread errors.
#[derive(Default)]
pub(super) struct TimingControl {
    // TLS Cell alignment leaves bit zero available for sticky failure. A single
    // atomic makes detach vs concurrent violation linearizable (no reset race).
    owner: AtomicUsize,
}

impl TimingControl {
    fn invalidate(&self) {
        let _ = self
            .owner
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |owner| {
                if owner == 0 {
                    None
                } else {
                    Some(owner | VIOLATION)
                }
            });
    }
}

/// Exact source coverage, never a claim about PM4/AQL or graph execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelTimingCoverage {
    EagerHipLaunchWrappers,
}

/// A complete successful measurement: no missing or unpaired wrapper launches.
#[derive(Debug)]
pub struct KernelTimingReport {
    pub per_kernel_ms: Vec<f32>,
    pub kernel_count: usize,
    pub coverage: KernelTimingCoverage,
}

// Copy only event/context operations; retain their dlopen mapping independently
// of HipRuntime. All calls are real resolved HIP APIs, not a host timer fallback.
struct EventApi {
    _lib: Arc<Library>,
    get_device: unsafe extern "C" fn(*mut c_int) -> u32,
    set_device: unsafe extern "C" fn(c_int) -> u32,
    event_create: unsafe extern "C" fn(*mut HipEvent) -> u32,
    event_sync: unsafe extern "C" fn(HipEvent) -> u32,
    event_elapsed: unsafe extern "C" fn(*mut f32, HipEvent, HipEvent) -> u32,
    event_destroy: unsafe extern "C" fn(HipEvent) -> u32,
    stream_sync: unsafe extern "C" fn(HipStream) -> u32,
}

impl EventApi {
    fn check(&self, code: u32, operation: &str) -> HipResult<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(HipError::new(code, operation))
        }
    }

    fn device(&self) -> HipResult<i32> {
        let mut device = 0;
        // SAFETY: mapping retained by self; stack out-parameter is writable.
        self.check(
            unsafe { (self.get_device)(&mut device) },
            "hipGetDevice (kernel timing)",
        )?;
        Ok(device)
    }

    fn select(&self, device: i32) -> HipResult<()> {
        // SAFETY: device is a previously queried HIP ordinal; mapping is live.
        self.check(
            unsafe { (self.set_device)(device) },
            "hipSetDevice (kernel timing)",
        )
    }
}

struct Pair {
    start: Event,
    stop: Event,
    stream: HipStream,
    started: bool,
    completed: bool,
}

struct State {
    control: Arc<TimingControl>,
    device: i32,
    pairs: Vec<Pair>,
    used: usize,
    in_flight: bool,
    failure: Option<HipError>,
}

impl State {
    fn fail(&mut self, failure: HipError) {
        if self.failure.is_none() {
            self.failure = Some(failure);
        }
    }

    fn reserve(&mut self, stream: HipStream) -> HipResult<usize> {
        if self.used == self.pairs.len() {
            let failure = error("kernel timing event pool exhausted");
            self.fail(failure.clone());
            return Err(failure);
        }
        let index = self.used;
        self.used += 1;
        let pair = &mut self.pairs[index];
        pair.stream = stream;
        pair.started = false;
        pair.completed = false;
        Ok(index)
    }
}

/// Owned thread-bound event pool. Allocate before entering the host verify
/// window. Failed or cancelled scopes poison their pool; no silent retry/reset.
#[must_use]
pub struct KernelTimingPool {
    api: EventApi,
    state: Box<State>,
    active: bool,
    closed: bool,
    _thread_bound: PhantomData<Rc<()>>,
}

impl HipRuntime {
    /// Allocate two timing-enabled HIP events per possible kernel before begin.
    /// The pool owns the mapping and remains safe if this runtime is moved/dropped.
    pub fn kernel_timing_pool(&self, capacity: usize) -> HipResult<KernelTimingPool> {
        reject_active(self, "cannot allocate a timing pool inside an active scope")?;
        KernelTimingPool::allocate(
            EventApi {
                _lib: Arc::clone(&self._lib),
                get_device: self.fn_get_device,
                set_device: self.fn_set_device,
                event_create: self.fn_event_create,
                event_sync: self.fn_event_synchronize,
                event_elapsed: self.fn_event_elapsed_time,
                event_destroy: self.fn_event_destroy,
                stream_sync: self.fn_stream_synchronize,
            },
            Arc::clone(
                self.kernel_timing_control
                    .get_or_init(|| Arc::new(TimingControl::default())),
            ),
            capacity,
        )
    }

    /// Whether this runtime (or this calling thread) has a strict timing scope.
    pub fn kernel_timing_is_active(&self) -> bool {
        self.kernel_timing_control
            .get()
            .is_some_and(|control| control.owner.load(Ordering::Acquire) != 0)
            || ACTIVE.with(|active| !active.get().is_null())
    }

    /// Native backend owners must call this BEFORE enqueueing PM4/AQL or other
    /// work outside the eager HIP wrappers. Inactive production behavior is
    /// unchanged; an active strict timing claim is invalidated and rejected.
    pub fn kernel_timing_reject_backend(&self, backend: &str) -> HipResult<()> {
        if self.kernel_timing_is_active() {
            reject_active(
                self,
                &format!("kernel timing unsupported backend: {backend}"),
            )
        } else {
            Ok(())
        }
    }
}

impl KernelTimingPool {
    fn allocate(api: EventApi, control: Arc<TimingControl>, capacity: usize) -> HipResult<Self> {
        if capacity == 0 {
            return Err(error("kernel timing capacity must be nonzero"));
        }
        let device = api.device()?;
        let mut pool = Self {
            api,
            state: Box::new(State {
                control,
                device,
                pairs: Vec::with_capacity(capacity),
                used: 0,
                in_flight: false,
                failure: None,
            }),
            active: false,
            closed: false,
            _thread_bound: PhantomData,
        };
        for _ in 0..capacity {
            let mut start = ptr::null_mut();
            // SAFETY: live mapping and valid stack out-parameter.
            pool.api.check(
                unsafe { (pool.api.event_create)(&mut start) },
                "hipEventCreate (start)",
            )?;
            let mut stop = ptr::null_mut();
            // SAFETY: live mapping and valid stack out-parameter.
            let code = unsafe { (pool.api.event_create)(&mut stop) };
            if let Err(e) = pool.api.check(code, "hipEventCreate (stop)") {
                // Preserve original creation error; earlier pairs are cleaned by
                // Drop. Destroy this unpaired allocation too.
                // SAFETY: start was successfully created on the current device.
                let _ = unsafe { (pool.api.event_destroy)(start) };
                return Err(e);
            }
            pool.state.pairs.push(Pair {
                start: Event(start),
                stop: Event(stop),
                stream: ptr::null_mut(),
                started: false,
                completed: false,
            });
        }
        Ok(pool)
    }

    /// True from successful begin until finish/cancel/Drop detaches collection.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Activate collection without borrowing Gpu/runtime. A scope cannot nest;
    /// an attempted nested begin also invalidates the existing scope.
    pub fn begin(&mut self) -> HipResult<()> {
        ACTIVE.with(|active| {
            let state = active.get();
            if !state.is_null() {
                // SAFETY: ACTIVE always points to this thread's stable live Box.
                unsafe {
                    (*state).fail(error("nested kernel timing scope"));
                }
                return Err(error("nested kernel timing scope"));
            }
            Ok(())
        })?;
        if self.closed || self.state.failure.is_some() {
            return Err(error("kernel timing pool is closed or failed"));
        }
        if self.api.device()? != self.state.device {
            return Err(error("kernel timing begin device differs from pool device"));
        }
        if self
            .state
            .control
            .owner
            .compare_exchange(0, thread_token(), Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            self.state.control.invalidate();
            return Err(error("runtime already has an active timing scope"));
        }
        self.state.used = 0;
        self.state.in_flight = false;
        self.active = true;
        ACTIVE.with(|active| active.set(&mut *self.state));
        Ok(())
    }

    fn detach(&mut self) {
        if self.active {
            ACTIVE.with(|active| active.set(ptr::null_mut()));
            let owner = self.state.control.owner.swap(0, Ordering::AcqRel);
            if owner & VIOLATION != 0 {
                self.state.fail(error(
                    "kernel timing runtime/thread/backend coverage violated",
                ));
            }
            self.active = false;
        }
    }

    // Drain every used pair even if another pair fails. For an unpaired stop,
    // synchronize its actual stream so cancellation cannot retire live events.
    fn drain(&mut self) -> HipResult<()> {
        let original = self.api.device()?;
        if original != self.state.device {
            self.state
                .fail(error("kernel timing finish device changed"));
            self.api.select(self.state.device)?;
        }
        let mut failure = None;
        for pair in &mut self.state.pairs[..self.state.used] {
            if !pair.started {
                continue;
            }
            // SAFETY: pool owns the live events; caller retains stream lifetime.
            let code = unsafe {
                if pair.completed {
                    (self.api.event_sync)(pair.stop.0)
                } else {
                    (self.api.stream_sync)(pair.stream)
                }
            };
            match self.api.check(code, "HIP kernel timing drain") {
                Ok(()) => pair.started = false,
                Err(e) => {
                    if failure.is_none() {
                        failure = Some(e);
                    }
                }
            }
        }
        if original != self.state.device {
            if let Err(e) = self.api.select(original) {
                if failure.is_none() {
                    failure = Some(e);
                }
            }
        }
        match failure {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Detach, wait on every paired stop event, then read raw GPU milliseconds.
    /// Empty, unpaired, failed or unsupported work can never produce a report.
    /// Result allocation happens after collection is detached, not per launch.
    pub fn finish(&mut self) -> HipResult<KernelTimingReport> {
        if !self.active {
            return Err(error("kernel timing scope is not active"));
        }
        self.detach();
        if self.state.in_flight {
            self.state.fail(error("unpaired kernel timing launch"));
        }
        if let Err(e) = self.drain() {
            self.state.fail(e);
        }
        if self.state.used == 0 {
            self.state
                .fail(error("kernel timing scope observed no HIP launches"));
        }
        if let Some(e) = &self.state.failure {
            return Err(e.clone());
        }
        let mut per_kernel_ms = Vec::with_capacity(self.state.used);
        for index in 0..self.state.used {
            let pair = &self.state.pairs[index];
            if !pair.completed {
                let e = error("unpaired kernel timing events");
                self.state.fail(e.clone());
                return Err(e);
            }
            let mut ms = 0.0;
            // SAFETY: both events recorded successfully and stop is synchronized.
            let code = unsafe { (self.api.event_elapsed)(&mut ms, pair.start.0, pair.stop.0) };
            if let Err(e) = self.api.check(code, "hipEventElapsedTime (kernel timing)") {
                self.state.fail(e.clone());
                return Err(e);
            }
            if !ms.is_finite() || ms < 0.0 {
                let e = error("invalid HIP event duration");
                self.state.fail(e.clone());
                return Err(e);
            }
            per_kernel_ms.push(ms);
        }
        Ok(KernelTimingReport {
            kernel_count: per_kernel_ms.len(),
            per_kernel_ms,
            coverage: KernelTimingCoverage::EagerHipLaunchWrappers,
        })
    }

    /// Cancel and drain on a host forward error. Cancellation permanently poisons
    /// this pool, even when GPU cleanup succeeds. Safe to call when inactive.
    pub fn cancel(&mut self) -> HipResult<()> {
        if self.active {
            self.detach();
            self.state
                .fail(error("kernel timing scope cancelled without finish"));
        }
        self.drain()
    }

    fn destroy(&mut self) -> HipResult<()> {
        if self.closed {
            return Ok(());
        }
        let mut failure = self.cancel().err();
        let original = self.api.device()?;
        if original != self.state.device {
            self.api.select(self.state.device)?;
        }
        for pair in self.state.pairs.drain(..) {
            for event in [pair.start, pair.stop] {
                // SAFETY: unique event ownership, allocation device selected.
                let code = unsafe { (self.api.event_destroy)(event.0) };
                if let Err(e) = self.api.check(code, "hipEventDestroy (kernel timing)") {
                    if failure.is_none() {
                        failure = Some(e);
                    }
                }
            }
        }
        self.state.used = 0;
        self.closed = true;
        if original != self.state.device {
            if let Err(e) = self.api.select(original) {
                if failure.is_none() {
                    failure = Some(e);
                }
            }
        }
        match failure {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Destroy all events; callers must propagate this error before publishing
    /// their strict timing claim. Drop alone cannot report destruction errors.
    pub fn close(mut self) -> HipResult<()> {
        self.destroy()
    }
}

impl Drop for KernelTimingPool {
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}

/// Free thread-local backend guard for actual native submission entrypoints.
/// Slice C calls this without a runtime handle BEFORE enqueueing PM4/AQL or
/// other work outside the eager HIP wrappers. Inactive collection (no active
/// scope on the calling thread) leaves behavior unchanged and returns `Ok`.
/// An active strict timing claim is poisoned first and the backend rejected.
/// Native launch wrappers cannot use this: they are issued on the pool's bound
/// device/stream with full control identity, not a bare TLS probe.
pub fn reject_kernel_timing_backend(backend: &str) -> HipResult<()> {
    ACTIVE.with(|active| {
        let state = active.get();
        if state.is_null() {
            return Ok(());
        }
        // SAFETY: thread-local pointer addresses a live stable boxed state.
        unsafe {
            (*state).fail(error(backend));
        }
        Err(error(backend))
    })
}

/// Graph/native-unsupported operations must reject before enqueue and invalidate
/// the existing claim, even if their caller catches the returned error.
pub(super) fn reject_active(runtime: &HipRuntime, message: &str) -> HipResult<()> {
    if let Some(control) = runtime.kernel_timing_control.get() {
        if control.owner.load(Ordering::Acquire) != 0 {
            control.invalidate();
            return Err(error(message));
        }
    }
    reject_kernel_timing_backend(message)
}

// Stack-only hook, compiled entirely out without the kernel-timing feature.
pub(super) struct LaunchTiming<'runtime> {
    runtime: &'runtime HipRuntime,
    state: *mut State,
    index: Option<usize>,
    finished: bool,
}

impl<'runtime> LaunchTiming<'runtime> {
    pub(super) fn new(runtime: &'runtime HipRuntime) -> HipResult<Self> {
        let state = ACTIVE.with(Cell::get);
        let control = runtime.kernel_timing_control.get();
        let owner = control.map_or(0, |control| control.owner.load(Ordering::Acquire));
        if owner != 0 && owner & !VIOLATION != thread_token() {
            if let Some(control) = control {
                control.invalidate();
            }
            return Err(error("HIP launch from wrong timing thread"));
        }
        if !state.is_null() {
            // SAFETY: same-thread short-lived borrow; no reference crosses HIP.
            let state_ref = unsafe { &mut *state };
            if !control.is_some_and(|control| Arc::ptr_eq(&state_ref.control, control)) {
                state_ref.fail(error("HIP launch from wrong timing runtime"));
                return Err(error("HIP launch from wrong timing runtime"));
            }
            if state_ref.in_flight {
                state_ref.fail(error("nested HIP timing launch"));
                return Err(error("nested HIP timing launch"));
            }
            if let Some(e) = &state_ref.failure {
                return Err(e.clone());
            }
            state_ref.in_flight = true;
        }
        Ok(Self {
            runtime,
            state,
            index: None,
            finished: false,
        })
    }

    pub(super) fn record_start(&mut self, stream: HipStream) -> HipResult<()> {
        if self.state.is_null() {
            return Ok(());
        }
        // SAFETY: same-thread ticket owns its live state until synchronous return.
        let device = unsafe { (*self.state).device };
        if self.runtime.current_device()? != device {
            return Err(error("HIP launch changed timing device"));
        }
        if self.runtime.stream_is_capturing(&Stream(stream))? {
            return Err(error("HIP graph capture unsupported by kernel timing"));
        }
        // SAFETY: exclusive short-lived access; fixed vector never grows.
        let index = unsafe { (*self.state).reserve(stream)? };
        self.index = Some(index);
        // SAFETY: same ticket owns the index; copy handle before native call.
        let start = unsafe { (&(*self.state).pairs)[index].start.0 };
        self.runtime
            .event_record(&Event(start), Some(&Stream(stream)))?;
        // SAFETY: stable state/index; no native call while this borrow is live.
        unsafe {
            (&mut (*self.state).pairs)[index].started = true;
        }
        Ok(())
    }

    pub(super) fn record_stop(&mut self, launch_code: u32) -> HipResult<()> {
        self.finished = true;
        if self.state.is_null() {
            return Ok(());
        }
        // SAFETY: same-thread short-lived state access, copy handles before HIP.
        let (stop, stream) = unsafe {
            let state = &mut *self.state;
            state.in_flight = false;
            if launch_code != 0 {
                state.fail(error("HIP kernel launch failed in timing scope"));
                // Native launch error wins; unfinished start drains actual stream.
                return Ok(());
            }
            let Some(index) = self.index else {
                state.fail(error("HIP timing launch has no start event"));
                return Err(error("HIP timing launch has no start event"));
            };
            let pair = &state.pairs[index];
            (pair.stop.0, pair.stream)
        };
        let result = self
            .runtime
            .event_record(&Event(stop), Some(&Stream(stream)));
        // SAFETY: native call ended, state still exclusively owned on this thread.
        unsafe {
            let state = &mut *self.state;
            match &result {
                Ok(()) => state.pairs[self.index.expect("paired start")].completed = true,
                Err(e) => state.fail(e.clone()),
            }
        }
        result
    }
}

impl Drop for LaunchTiming<'_> {
    fn drop(&mut self) {
        if !self.state.is_null() && !self.finished {
            // SAFETY: ticket is synchronous and cannot outlive the owning pool.
            unsafe {
                let state = &mut *self.state;
                state.in_flight = false;
                state.fail(error("HIP launch returned before timing events paired"));
            }
        }
    }
}

#[cfg(test)]
mod tests;
