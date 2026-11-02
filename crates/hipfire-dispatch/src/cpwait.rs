// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! HIP-free state machines of the D2 readiness and CP-wait join transports.
//!
//! * [`spin_until_ready`]: the bounded input-readiness spin of
//!   `HIPFIRE_QWEN4_D2_READY=event|cpwait` with its poll accounting.
//! * [`JoinGate`]: the one funnel through which a queued `hipStreamWaitValue32`
//!   is either released with its generation or aborted. Every error path that
//!   can follow the wait enqueue goes through [`JoinGate::fail`], which stores
//!   [`SIGNAL_ABORT`] BEFORE it may synchronize the device: a device
//!   synchronize with a pending wait never returns.
//! * [`Watchdog`]: per-arm deadlines. On expiry it stores the abort value and
//!   marks the arm's transaction poisoned; one lazily started thread serves the
//!   process. The core is HIP-free and runs against an injectable [`Clock`].

use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use hip_bridge::SIGNAL_ABORT;

/// How long `prepare` spins on the event readiness proof
/// before it falls back to the blocking stream synchronization.
pub const READY_SPIN_BUDGET: Duration = Duration::from_millis(2);

/// Deadline of one armed CP wait. The CPU arithmetic of a layer takes
/// milliseconds; ten seconds only ever expires when the join thread is stuck or
/// gone, and then the watchdog releases the queue with the abort value.
pub const CPWAIT_WATCHDOG_TIMEOUT: Duration = Duration::from_secs(10);

// ── Readiness spin ──────────────────────────────────────────────

/// How a bounded readiness spin ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpinOutcome {
    /// The proof completed after `polls` polls.
    Ready { polls: u64 },
    /// The budget ran out after `polls` polls; the caller blocks.
    TimedOut { polls: u64 },
}

impl SpinOutcome {
    pub fn polls(self) -> u64 {
        match self {
            Self::Ready { polls } | Self::TimedOut { polls } => polls,
        }
    }
}

/// Poll until `poll` reports readiness or `elapsed` reaches `budget`. Always
/// polls at least once, so an already-complete proof never counts a timeout; a
/// poll error ends the spin and propagates (the proof stays unmanufactured).
pub fn spin_until_ready<E>(
    mut poll: impl FnMut() -> Result<bool, E>,
    mut elapsed: impl FnMut() -> Duration,
    budget: Duration,
) -> Result<SpinOutcome, E> {
    let mut polls = 0u64;
    loop {
        polls += 1;
        if poll()? {
            return Ok(SpinOutcome::Ready { polls });
        }
        if elapsed() >= budget {
            return Ok(SpinOutcome::TimedOut { polls });
        }
        std::hint::spin_loop();
    }
}

// ── Join gate ───────────────────────────────────────────────────

/// What a [`JoinGate`] needs from the world. Implemented over the signal word,
/// the watchdog and the device by the executor, and by a recorder in tests.
pub trait GatePort {
    /// Volatile (release) store into the join word.
    fn store_join(&self, value: u32);
    /// Stop the watchdog for this arm.
    fn disarm_watchdog(&self) -> Disarm;
    /// Drain the device. Legal only once no wait can be pending.
    fn device_sync(&self) -> Result<(), String>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateState {
    /// No wait is queued.
    Idle,
    /// The wait for `gen` is (or may be) queued and the flag is not stored.
    Armed,
    /// `gen` was stored: the wait is released with valid data.
    Released,
    /// The abort value was stored: the wait is released with garbage.
    Aborted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateError {
    /// `arm`/`release` out of order.
    State(&'static str),
    /// The watchdog fired first: the abort value is stored, the consumer ran on
    /// garbage and the transaction is poisoned.
    WatchdogFired,
    /// The watchdog no longer knew the arm; the caller must `fail`.
    ArmLost,
}

impl std::fmt::Display for GateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::State(what) => f.write_str(what),
            Self::WatchdogFired => {
                f.write_str("the join watchdog fired before the CPU released the wait")
            }
            Self::ArmLost => f.write_str("the join watchdog lost its arm"),
        }
    }
}

/// Pure state of one layer's CP wait.
#[derive(Debug)]
pub struct JoinGate {
    state: GateState,
    gen: u32,
    aborts: u64,
}

impl JoinGate {
    pub const fn new() -> Self {
        Self {
            state: GateState::Idle,
            gen: 0,
            aborts: 0,
        }
    }

    pub fn state(&self) -> GateState {
        self.state
    }

    pub fn is_armed(&self) -> bool {
        self.state == GateState::Armed
    }

    /// Aborts this gate stored itself (a watchdog abort is counted by the
    /// watchdog target).
    pub fn aborts(&self) -> u64 {
        self.aborts
    }

    /// A new call begins: a released gate is reusable; an armed one means the
    /// previous call never joined, and an aborted one is dead.
    pub fn begin_call(&mut self) -> Result<(), GateError> {
        match self.state {
            GateState::Idle => Ok(()),
            GateState::Released => {
                self.state = GateState::Idle;
                Ok(())
            }
            GateState::Armed => Err(GateError::State("the previous CP wait was never joined")),
            GateState::Aborted => Err(GateError::State("the join gate was aborted")),
        }
    }

    /// Record the wait for `gen` BEFORE its enqueue call: a failed enqueue may
    /// still have queued the packet, so from here an error must abort.
    pub fn arm(&mut self, gen: u32) -> Result<(), GateError> {
        if self.state != GateState::Idle {
            return Err(GateError::State("arm over a non-idle join gate"));
        }
        self.state = GateState::Armed;
        self.gen = gen;
        Ok(())
    }

    /// The CPU is done and the slots are published: stop the watchdog, then
    /// store `gen`. When the watchdog already fired the abort value stays and
    /// the call fails (the combine consumed garbage).
    pub fn release(&mut self, port: &impl GatePort) -> Result<(), GateError> {
        if self.state != GateState::Armed {
            return Err(GateError::State("release of a join gate that is not armed"));
        }
        match port.disarm_watchdog() {
            Disarm::Disarmed => {
                port.store_join(self.gen);
                self.state = GateState::Released;
                Ok(())
            }
            Disarm::AlreadyFired => {
                self.state = GateState::Aborted;
                Err(GateError::WatchdogFired)
            }
            // Stay armed: the caller's `fail` stores the abort value.
            Disarm::Unknown => Err(GateError::ArmLost),
        }
    }

    /// Release a pending wait with the abort value, without any device call.
    /// Returns whether this call stored it.
    pub fn abort(&mut self, port: &impl GatePort) -> bool {
        if self.state != GateState::Armed {
            return false;
        }
        port.disarm_watchdog();
        port.store_join(SIGNAL_ABORT);
        self.state = GateState::Aborted;
        self.aborts += 1;
        true
    }

    /// The single error funnel: abort FIRST, then drain the device. Idle,
    /// released and already-aborted gates just drain.
    pub fn fail(&mut self, port: &impl GatePort) -> Result<(), String> {
        self.abort(port);
        port.device_sync()
    }
}

impl Default for JoinGate {
    fn default() -> Self {
        Self::new()
    }
}

// ── Watchdog ────────────────────────────────────────────────────

/// Monotonic nanoseconds. Injectable so the watchdog logic runs on a fake clock.
pub trait Clock: Send + Sync + 'static {
    fn now_ns(&self) -> u64;
}

pub struct SystemClock {
    origin: Instant,
}

impl SystemClock {
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn now_ns(&self) -> u64 {
        self.origin.elapsed().as_nanos().min(u64::MAX as u128) as u64
    }
}

/// What the watchdog does on expiry. `fire` runs under the watchdog's lock: it
/// must not block and must not call back into the watchdog.
pub trait WatchdogTarget: Send + Sync {
    fn fire(&self);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArmId(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Disarm {
    /// The arm was removed before its deadline: the target is never touched.
    Disarmed,
    /// The deadline passed first and the target was fired.
    AlreadyFired,
    /// No such arm.
    Unknown,
}

struct Entry {
    id: u64,
    deadline_ns: u64,
    fired: bool,
    target: Arc<dyn WatchdogTarget>,
}

/// The deadline table. `wake_at` is the instant the serving thread wakes next
/// (`None` while it sleeps without a timeout), so an arm only needs to wake it
/// when its deadline is earlier than that.
struct WatchdogCore {
    entries: Vec<Entry>,
    next_id: u64,
    wake_at: Option<u64>,
}

impl WatchdogCore {
    const fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_id: 1,
            wake_at: None,
        }
    }

    /// Returns the arm and whether the serving thread must be woken.
    fn arm(&mut self, deadline_ns: u64, target: Arc<dyn WatchdogTarget>) -> (ArmId, bool) {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(Entry {
            id,
            deadline_ns,
            fired: false,
            target,
        });
        let wake = match self.wake_at {
            None => true,
            Some(at) => deadline_ns < at,
        };
        (ArmId(id), wake)
    }

    fn disarm(&mut self, id: ArmId) -> Disarm {
        match self.entries.iter().position(|e| e.id == id.0) {
            None => Disarm::Unknown,
            Some(at) => {
                let entry = self.entries.swap_remove(at);
                if entry.fired {
                    Disarm::AlreadyFired
                } else {
                    Disarm::Disarmed
                }
            }
        }
    }

    /// Fire every unfired entry whose deadline is at or before `now_ns`.
    fn expire(&mut self, now_ns: u64) -> u64 {
        let mut fired = 0;
        for entry in &mut self.entries {
            if !entry.fired && entry.deadline_ns <= now_ns {
                entry.fired = true;
                entry.target.fire();
                fired += 1;
            }
        }
        fired
    }

    fn next_deadline(&self) -> Option<u64> {
        self.entries
            .iter()
            .filter(|e| !e.fired)
            .map(|e| e.deadline_ns)
            .min()
    }
}

struct Shared<C> {
    core: Mutex<WatchdogCore>,
    wake: Condvar,
    clock: C,
    timeout_ns: u64,
    fires: AtomicU64,
}

impl<C> Shared<C> {
    fn lock(&self) -> MutexGuard<'_, WatchdogCore> {
        // A panicking target must not wedge every later arm.
        self.core
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Per-arm deadlines served by one thread (a `Condvar` sleep, never a busy
/// loop). The thread only exists after [`Watchdog::spawn`]; a watchdog built by
/// [`Watchdog::manual`] is driven by [`Watchdog::expire`] (tests).
pub struct Watchdog<C: Clock> {
    shared: Arc<Shared<C>>,
}

impl<C: Clock> Watchdog<C> {
    /// A watchdog with no serving thread: call [`Self::expire`] to advance it.
    pub fn manual(clock: C, timeout: Duration) -> Self {
        Self {
            shared: Arc::new(Shared {
                core: Mutex::new(WatchdogCore::new()),
                wake: Condvar::new(),
                clock,
                timeout_ns: timeout.as_nanos().min(u64::MAX as u128) as u64,
                fires: AtomicU64::new(0),
            }),
        }
    }

    /// A watchdog with its serving thread started.
    pub fn spawn(clock: C, timeout: Duration) -> io::Result<Self> {
        let dog = Self::manual(clock, timeout);
        let shared = Arc::clone(&dog.shared);
        std::thread::Builder::new()
            .name("hipfire-cpwait-watchdog".into())
            .spawn(move || serve(&shared))?;
        Ok(dog)
    }

    /// Arm a deadline of `now + timeout` for `target`.
    pub fn arm(&self, target: Arc<dyn WatchdogTarget>) -> ArmId {
        let deadline = self
            .shared
            .clock
            .now_ns()
            .saturating_add(self.shared.timeout_ns);
        let (id, wake) = self.shared.lock().arm(deadline, target);
        if wake {
            self.shared.wake.notify_one();
        }
        id
    }

    /// Remove an arm. Serialized with [`Self::expire`]: after this returns the
    /// target is never touched again, so the word it points at may be freed.
    pub fn disarm(&self, id: ArmId) -> Disarm {
        self.shared.lock().disarm(id)
    }

    /// Fire every arm whose deadline is due on the clock; returns how many.
    pub fn expire(&self) -> u64 {
        let fired = self.shared.lock().expire(self.shared.clock.now_ns());
        self.shared.fires.fetch_add(fired, Ordering::Relaxed);
        fired
    }

    /// Fires since creation.
    pub fn fires(&self) -> u64 {
        self.shared.fires.load(Ordering::Relaxed)
    }
}

/// The serving loop: expire, then sleep until the next deadline (or until an
/// arm needs an earlier wake).
fn serve<C: Clock>(shared: &Shared<C>) {
    let mut core = shared.lock();
    loop {
        let now = shared.clock.now_ns();
        let fired = core.expire(now);
        shared.fires.fetch_add(fired, Ordering::Relaxed);
        core = match core.next_deadline() {
            None => {
                core.wake_at = None;
                shared.wake.wait(core).unwrap_or_else(|p| p.into_inner())
            }
            Some(deadline) => {
                core.wake_at = Some(deadline);
                let wait = Duration::from_nanos(deadline.saturating_sub(now));
                shared
                    .wake
                    .wait_timeout(core, wait)
                    .unwrap_or_else(|p| p.into_inner())
                    .0
            }
        };
    }
}

/// The process-wide watchdog, started on first use. `Err` when the thread
/// could not be created (the caller then keeps the host join).
pub fn global_watchdog() -> Result<&'static Watchdog<SystemClock>, String> {
    static GLOBAL: LazyLock<Result<Watchdog<SystemClock>, String>> = LazyLock::new(|| {
        Watchdog::spawn(SystemClock::new(), CPWAIT_WATCHDOG_TIMEOUT)
            .map_err(|e| format!("cannot start the CP-wait watchdog thread: {e}"))
    });
    GLOBAL.as_ref().map_err(Clone::clone)
}

/// The watchdog target of one executor: on expiry run `store_abort` (the abort
/// store into its join word) and poison the transaction through an atomic the
/// executor checks.
pub struct JoinTarget {
    store_abort: Box<dyn Fn() + Send + Sync>,
    poisoned: AtomicBool,
    fires: AtomicU64,
}

impl JoinTarget {
    /// `store_abort` runs on the watchdog thread under its lock: it must only
    /// store [`SIGNAL_ABORT`] into the join word, and the word must stay alive
    /// until the arm is disarmed (the executor disarms before it drops the word).
    pub fn new(store_abort: Box<dyn Fn() + Send + Sync>) -> Self {
        Self {
            store_abort,
            poisoned: AtomicBool::new(false),
            fires: AtomicU64::new(0),
        }
    }

    /// The watchdog (or an abort) poisoned this executor's transactions.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::SeqCst)
    }

    pub fn poison(&self) {
        self.poisoned.store(true, Ordering::SeqCst);
    }

    pub fn fires(&self) -> u64 {
        self.fires.load(Ordering::Relaxed)
    }
}

impl WatchdogTarget for JoinTarget {
    fn fire(&self) {
        (self.store_abort)();
        self.poisoned.store(true, Ordering::SeqCst);
        self.fires.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::sync::atomic::AtomicUsize;

    // ── spin accounting ──

    #[test]
    fn a_ready_first_poll_is_one_poll_and_no_timeout() {
        let out: Result<_, ()> = spin_until_ready(
            || Ok(true),
            || Duration::from_secs(99),
            Duration::from_millis(2),
        );
        assert_eq!(out, Ok(SpinOutcome::Ready { polls: 1 }));
    }

    #[test]
    fn the_spin_counts_polls_until_ready() {
        let mut n = 0;
        let out: Result<_, ()> = spin_until_ready(
            || {
                n += 1;
                Ok(n == 5)
            },
            || Duration::ZERO,
            READY_SPIN_BUDGET,
        );
        assert_eq!(out, Ok(SpinOutcome::Ready { polls: 5 }));
    }

    #[test]
    fn an_expired_budget_falls_back_and_counts_every_poll() {
        let mut clock = Duration::ZERO;
        let mut polls = 0;
        let out: Result<_, ()> = spin_until_ready(
            || {
                polls += 1;
                Ok(false)
            },
            || {
                clock += Duration::from_micros(500);
                clock
            },
            READY_SPIN_BUDGET,
        );
        // elapsed reads 0.5, 1.0, 1.5, 2.0 ms: the fourth poll ends the spin.
        assert_eq!(out, Ok(SpinOutcome::TimedOut { polls: 4 }));
        assert_eq!(polls, 4);
        assert_eq!(out.unwrap().polls(), 4);
    }

    #[test]
    fn a_poll_error_ends_the_spin_without_a_proof() {
        let out = spin_until_ready(|| Err("device lost"), || Duration::ZERO, READY_SPIN_BUDGET);
        assert_eq!(out, Err("device lost"));
    }

    #[test]
    fn the_documented_constants() {
        assert_eq!(READY_SPIN_BUDGET, Duration::from_millis(2));
        assert_eq!(CPWAIT_WATCHDOG_TIMEOUT, Duration::from_secs(10));
    }

    // ── join gate: abort ordering ──

    struct Rec {
        log: RefCell<Vec<String>>,
        watchdog: Disarm,
        sync: Result<(), String>,
    }

    impl Rec {
        fn new() -> Self {
            Self {
                log: RefCell::new(Vec::new()),
                watchdog: Disarm::Disarmed,
                sync: Ok(()),
            }
        }
        fn log(&self) -> Vec<String> {
            self.log.borrow().clone()
        }
        fn position(&self, what: &str) -> Option<usize> {
            self.log.borrow().iter().position(|e| e == what)
        }
    }

    impl GatePort for Rec {
        fn store_join(&self, value: u32) {
            let what = if value == SIGNAL_ABORT {
                "store ABORT".into()
            } else {
                format!("store {value}")
            };
            self.log.borrow_mut().push(what);
        }
        fn disarm_watchdog(&self) -> Disarm {
            self.log.borrow_mut().push("disarm".into());
            self.watchdog
        }
        fn device_sync(&self) -> Result<(), String> {
            self.log.borrow_mut().push("sync".into());
            self.sync.clone()
        }
    }

    /// Every site between the wait enqueue and the flag store funnels into
    /// `fail`: the combine enqueue, the mapped-buffer marking, the CPU
    /// arithmetic, the publication, the lost watchdog arm, the abandoned
    /// pipeline step, `free_gpu`. ABORT must precede the sync in all of them.
    #[test]
    fn the_abort_is_stored_before_the_sync_at_every_error_site() {
        for site in [
            "wait enqueue failed (the packet may be queued)",
            "combine launch failed",
            "mapped marking failed",
            "cpu arithmetic failed",
            "publish failed",
            "free_gpu while armed",
        ] {
            let port = Rec::new();
            let mut gate = JoinGate::new();
            gate.arm(7).unwrap();
            gate.fail(&port).unwrap();
            let abort = port
                .position("store ABORT")
                .unwrap_or_else(|| panic!("{site}: no abort"));
            let sync = port
                .position("sync")
                .unwrap_or_else(|| panic!("{site}: no sync"));
            assert!(abort < sync, "{site}: {:?}", port.log());
            assert_eq!(gate.state(), GateState::Aborted, "{site}");
            assert_eq!(gate.aborts(), 1, "{site}");
        }
    }

    #[test]
    fn the_watchdog_is_disarmed_before_the_abort_store() {
        let port = Rec::new();
        let mut gate = JoinGate::new();
        gate.arm(3).unwrap();
        gate.fail(&port).unwrap();
        assert_eq!(port.log(), ["disarm", "store ABORT", "sync"]);
    }

    #[test]
    fn an_unarmed_gate_only_drains() {
        for prepare in [
            |_: &mut JoinGate, _: &Rec| {},
            |g: &mut JoinGate, p: &Rec| {
                g.arm(1).unwrap();
                g.release(p).unwrap();
                p.log.borrow_mut().clear();
            },
        ] {
            let port = Rec::new();
            let mut gate = JoinGate::new();
            prepare(&mut gate, &port);
            gate.fail(&port).unwrap();
            assert_eq!(port.log(), ["sync"], "no abort without a pending wait");
            assert_eq!(gate.aborts(), 0);
        }
    }

    #[test]
    fn a_failed_gate_never_stores_twice() {
        let port = Rec::new();
        let mut gate = JoinGate::new();
        gate.arm(2).unwrap();
        gate.fail(&port).unwrap();
        gate.fail(&port).unwrap();
        let stores = port.log().iter().filter(|e| e.starts_with("store")).count();
        assert_eq!(stores, 1);
        assert_eq!(gate.aborts(), 1);
    }

    #[test]
    fn a_sync_error_still_reports_after_the_abort() {
        let mut port = Rec::new();
        port.sync = Err("device lost".into());
        let mut gate = JoinGate::new();
        gate.arm(2).unwrap();
        assert_eq!(gate.fail(&port), Err("device lost".to_string()));
        assert_eq!(port.log(), ["disarm", "store ABORT", "sync"]);
    }

    #[test]
    fn abort_alone_never_syncs() {
        let port = Rec::new();
        let mut gate = JoinGate::new();
        gate.arm(2).unwrap();
        assert!(gate.abort(&port));
        assert!(!gate.abort(&port));
        assert_eq!(port.log(), ["disarm", "store ABORT"]);
    }

    #[test]
    fn release_stops_the_watchdog_then_stores_the_generation() {
        let port = Rec::new();
        let mut gate = JoinGate::new();
        gate.arm(41).unwrap();
        gate.release(&port).unwrap();
        assert_eq!(port.log(), ["disarm", "store 41"]);
        assert_eq!(gate.state(), GateState::Released);
        assert!(gate.release(&port).is_err(), "released once");
    }

    #[test]
    fn a_fired_watchdog_keeps_the_abort_and_fails_the_release() {
        let mut port = Rec::new();
        port.watchdog = Disarm::AlreadyFired;
        let mut gate = JoinGate::new();
        gate.arm(5).unwrap();
        assert_eq!(gate.release(&port), Err(GateError::WatchdogFired));
        assert_eq!(
            port.log(),
            ["disarm"],
            "the generation is never stored over the abort"
        );
        assert_eq!(gate.state(), GateState::Aborted);
        gate.fail(&port).unwrap();
        assert_eq!(
            port.log(),
            ["disarm", "sync"],
            "the watchdog already stored it"
        );
    }

    #[test]
    fn a_lost_arm_stays_armed_so_fail_stores_the_abort() {
        let mut port = Rec::new();
        port.watchdog = Disarm::Unknown;
        let mut gate = JoinGate::new();
        gate.arm(5).unwrap();
        assert_eq!(gate.release(&port), Err(GateError::ArmLost));
        assert!(gate.is_armed());
        gate.fail(&port).unwrap();
        assert!(port.position("store ABORT").unwrap() < port.position("sync").unwrap());
    }

    #[test]
    fn gate_calls_must_follow_the_protocol() {
        let port = Rec::new();
        let mut gate = JoinGate::new();
        assert!(gate.release(&port).is_err(), "release before arm");
        gate.arm(1).unwrap();
        assert!(gate.arm(2).is_err(), "double arm");
        assert!(
            gate.begin_call().is_err(),
            "a new call over an unjoined wait"
        );
        gate.release(&port).unwrap();
        gate.begin_call().unwrap();
        assert_eq!(gate.state(), GateState::Idle);
        gate.arm(2).unwrap();
        gate.fail(&port).unwrap();
        assert!(gate.begin_call().is_err(), "an aborted gate is dead");
    }

    // ── watchdog on a fake clock ──

    struct FakeClock(Arc<AtomicU64>);

    impl Clock for FakeClock {
        fn now_ns(&self) -> u64 {
            self.0.load(Ordering::SeqCst)
        }
    }

    struct CountTarget(AtomicUsize);

    impl WatchdogTarget for CountTarget {
        fn fire(&self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn fixture() -> (Watchdog<FakeClock>, Arc<AtomicU64>) {
        let now = Arc::new(AtomicU64::new(1_000));
        (
            Watchdog::manual(FakeClock(Arc::clone(&now)), Duration::from_secs(10)),
            now,
        )
    }

    fn at(now: &AtomicU64, d: Duration) {
        now.store(d.as_nanos() as u64, Ordering::SeqCst);
    }

    #[test]
    fn nothing_fires_before_the_deadline() {
        let (dog, now) = fixture();
        let target = Arc::new(CountTarget(AtomicUsize::new(0)));
        dog.arm(target.clone());
        at(&now, Duration::from_millis(9_999));
        assert_eq!(dog.expire(), 0);
        assert_eq!(target.0.load(Ordering::SeqCst), 0);
        assert_eq!(dog.fires(), 0);
    }

    #[test]
    fn expiry_fires_once_and_disarm_reports_it() {
        let (dog, now) = fixture();
        let target = Arc::new(CountTarget(AtomicUsize::new(0)));
        let id = dog.arm(target.clone());
        // Armed at 1 us with a 10 s timeout: due at 10.000001 s.
        at(&now, Duration::from_secs(10));
        assert_eq!(dog.expire(), 0, "one microsecond early");
        at(&now, Duration::from_secs(10) + Duration::from_micros(1));
        assert_eq!(dog.expire(), 1);
        assert_eq!(dog.expire(), 0, "an arm fires once");
        assert_eq!(target.0.load(Ordering::SeqCst), 1);
        assert_eq!(dog.fires(), 1);
        assert_eq!(dog.disarm(id), Disarm::AlreadyFired);
        assert_eq!(dog.disarm(id), Disarm::Unknown, "disarm consumes the arm");
    }

    #[test]
    fn a_disarmed_arm_never_fires() {
        let (dog, now) = fixture();
        let target = Arc::new(CountTarget(AtomicUsize::new(0)));
        let id = dog.arm(target.clone());
        assert_eq!(dog.disarm(id), Disarm::Disarmed);
        at(&now, Duration::from_secs(3600));
        assert_eq!(dog.expire(), 0);
        assert_eq!(target.0.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn arms_have_independent_deadlines() {
        let (dog, now) = fixture();
        let a = Arc::new(CountTarget(AtomicUsize::new(0)));
        let b = Arc::new(CountTarget(AtomicUsize::new(0)));
        let ia = dog.arm(a.clone());
        at(&now, Duration::from_secs(8));
        let ib = dog.arm(b.clone());
        at(&now, Duration::from_secs(11));
        assert_eq!(dog.expire(), 1);
        assert_eq!(
            (a.0.load(Ordering::SeqCst), b.0.load(Ordering::SeqCst)),
            (1, 0)
        );
        assert_eq!(dog.disarm(ib), Disarm::Disarmed);
        assert_eq!(dog.disarm(ia), Disarm::AlreadyFired);
    }

    #[test]
    fn the_serving_thread_is_woken_only_for_an_earlier_deadline() {
        let target: Arc<dyn WatchdogTarget> = Arc::new(CountTarget(AtomicUsize::new(0)));
        let mut core = WatchdogCore::new();
        assert!(
            core.arm(100, target.clone()).1,
            "an idle thread sleeps without a timeout"
        );
        core.wake_at = Some(100);
        assert!(
            !core.arm(200, target.clone()).1,
            "the thread wakes at 100 anyway"
        );
        assert!(!core.arm(100, target.clone()).1);
        assert!(
            core.arm(99, target.clone()).1,
            "an earlier deadline needs a wake"
        );
        assert_eq!(core.next_deadline(), Some(99));
    }

    #[test]
    fn a_fired_entry_leaves_the_next_deadline() {
        let target: Arc<dyn WatchdogTarget> = Arc::new(CountTarget(AtomicUsize::new(0)));
        let mut core = WatchdogCore::new();
        core.arm(10, target.clone());
        core.arm(20, target.clone());
        assert_eq!(core.expire(15), 1);
        assert_eq!(core.next_deadline(), Some(20));
        assert_eq!(core.expire(25), 1);
        assert_eq!(core.next_deadline(), None);
    }

    /// The real serving thread, with a 30 ms timeout: a fired arm stores the
    /// abort value and poisons; a disarmed arm leaves both untouched.
    #[test]
    fn the_serving_thread_fires_expired_arms_and_spares_disarmed_ones() {
        struct Flag(AtomicBool);
        impl WatchdogTarget for Flag {
            fn fire(&self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let dog = Watchdog::spawn(SystemClock::new(), Duration::from_millis(30)).unwrap();
        let spared = Arc::new(Flag(AtomicBool::new(false)));
        let fired = Arc::new(Flag(AtomicBool::new(false)));
        let spared_id = dog.arm(spared.clone());
        assert_eq!(dog.disarm(spared_id), Disarm::Disarmed);
        let fired_id = dog.arm(fired.clone());
        let start = Instant::now();
        while !fired.0.load(Ordering::SeqCst) {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "the watchdog never fired"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(!spared.0.load(Ordering::SeqCst));
        assert_eq!(dog.fires(), 1);
        assert_eq!(dog.disarm(fired_id), Disarm::AlreadyFired);
    }

    #[test]
    fn a_fired_join_target_stores_the_abort_and_poisons() {
        let word = Arc::new(AtomicU64::new(0));
        let sink = Arc::clone(&word);
        let target = JoinTarget::new(Box::new(move || {
            sink.store(u64::from(SIGNAL_ABORT), Ordering::SeqCst);
        }));
        assert!(!target.is_poisoned());
        target.fire();
        assert!(target.is_poisoned());
        assert_eq!(target.fires(), 1);
        assert_eq!(word.load(Ordering::SeqCst), u64::from(SIGNAL_ABORT));
    }
}
