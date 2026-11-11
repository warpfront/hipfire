//! The process-wide CPU routed-expert splice pool.
//!
//! The splice's GEMV regions are memory-bandwidth bound row dots; running them
//! on every logical CPU (SMT siblings share a core's load ports and L1/L2)
//! measured slower than one worker per physical core (Strix Halo: 10 experts ×
//! 1 row, 365 µs at 16 threads vs 438 µs at 32). This module builds one
//! lazily-initialised rayon pool sized to the physical cores of the process's
//! allowed CPU set, with each worker pinned to one logical CPU of a distinct
//! core. The global rayon pool and every other caller are untouched.
//!
//! Arithmetic is independent of the pool: the shared GEMV computes each output
//! element as one row dot, so the same bytes come out for any worker count
//! (see the `gemv` determinism test).

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

/// Splice pool configuration, resolved by the caller from its knobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplicePoolConfig {
    /// Worker count; `0` = one per physical core in the allowed CPU set.
    pub threads: usize,
    /// Pin each worker to a logical CPU of a distinct physical core.
    pub pin: bool,
}

static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// The splice pool. Built on first use from `cfg`; later calls return the same
/// pool and ignore their `cfg` (the knobs are process-lifetime).
pub fn splice_pool(cfg: SplicePoolConfig) -> &'static rayon::ThreadPool {
    POOL.get_or_init(|| build_pool(cfg))
}

/// Run `f` on the splice pool (built from `cfg` on first use). Every rayon
/// region `f` starts uses the pool's workers, not the global pool's.
pub fn install<R: Send>(cfg: SplicePoolConfig, f: impl FnOnce() -> R + Send) -> R {
    splice_pool(cfg).install(f)
}

/// Run `f(i, chunk_i)` for every `chunk`-long piece of `data` in the current
/// rayon pool (the splice pool when called inside [`install`]). The pieces are
/// disjoint, so `f` may write its piece freely; this is how the arm's per-slot
/// middle (SwiGLU + hidden rotation) joins the GEMV regions on the same
/// workers instead of running serially on the calling worker.
pub fn par_for_each_chunk_mut<T: Send>(
    data: &mut [T],
    chunk: usize,
    f: impl Fn(usize, &mut [T]) + Sync + Send,
) {
    use rayon::prelude::*;
    data.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(i, c)| f(i, c));
}

// ---- spin wake / spin completion -----------------------------------------------------------
//
// An idle rayon worker parks on a futex after a few dozen yield rounds, so a D2 layer that runs
// the CPU arm once per ~0.7 ms pays a futex wake of every worker when `install` injects the arm,
// and a Mutex/Condvar wake of the calling thread when it completes. [`PoolWarm`] moves the first
// off the critical path (the workers are woken, and held hot, from the layer's input-readiness
// wait until the arm has run) and [`install_spin`] removes the second (the caller polls an atomic
// instead of blocking). Neither changes what runs: the arm is the same closure on the same pool,
// so its outputs are byte-identical; only when the workers are awake differs.
//
// Both spins are bounded and yield: a spinner calls `sched_yield` every [`SPIN_BATCH`] pauses, so
// a thread that shares its CPU (the calling thread can land on an SMT sibling of a pinned worker)
// is never starved for a scheduler quantum, and each spin gives up after its budget and falls back
// to the ordinary park/sleep.

/// `spin_loop` hints between two `sched_yield` calls of a spinner.
const SPIN_BATCH: u32 = 32;

/// Longest the caller waits for the workers' spinners to report in before the arm is published.
const WARM_START_WAIT: Duration = Duration::from_micros(150);

#[inline]
fn spin_step(n: &mut u32) {
    *n += 1;
    if *n % SPIN_BATCH == 0 {
        std::thread::yield_now();
    } else {
        std::hint::spin_loop();
    }
}

/// State shared by one [`PoolWarm`] and the spinner every worker runs for it.
struct WarmCtl {
    /// Set when the arm has finished (or the warm is dropped): spinners return.
    stop: AtomicBool,
    /// Set before the arm is published: a spinner that starts later does not spin. A worker
    /// that polled the spinner job from inside the arm's own latch wait would otherwise spin
    /// there and stall the arm it is part of until the budget ran out.
    late: AtomicBool,
    /// Spinners that have entered their loop.
    started: AtomicUsize,
    workers: usize,
    budget: Duration,
}

/// Workers held awake for an upcoming CPU arm; dropping it (or the arm completing through
/// [`install_spin`] after [`PoolWarm::arm_next_install`]) lets them go back to sleep.
pub struct PoolWarm {
    ctl: Arc<WarmCtl>,
}

impl PoolWarm {
    /// Wake every worker of the splice pool (built from `cfg` on first use) and keep it spinning
    /// for at most `budget`, or until this value is dropped.
    pub fn start(cfg: SplicePoolConfig, budget: Duration) -> PoolWarm {
        let pool = splice_pool(cfg);
        let ctl = Arc::new(WarmCtl {
            stop: AtomicBool::new(false),
            late: AtomicBool::new(false),
            started: AtomicUsize::new(0),
            workers: pool.current_num_threads(),
            budget,
        });
        let spinner = ctl.clone();
        pool.spawn_broadcast(move |_| spin_until_stopped(&spinner));
        PoolWarm { ctl }
    }

    /// The next [`install_spin`] on THIS thread publishes its arm only once the spinners are in,
    /// and then lets none start. Call it immediately before the arm; [`clear_next_install`]
    /// undoes it when the arm turns out not to install (nothing to compute).
    pub fn arm_next_install(&self) {
        NEXT_INSTALL.with(|c| *c.borrow_mut() = Some(self.ctl.clone()));
    }
}

impl Drop for PoolWarm {
    fn drop(&mut self) {
        self.ctl.stop.store(true, Ordering::Release);
    }
}

thread_local! {
    static NEXT_INSTALL: RefCell<Option<Arc<WarmCtl>>> = const { RefCell::new(None) };
}

/// Forget a pending [`PoolWarm::arm_next_install`] of this thread.
pub fn clear_next_install() {
    NEXT_INSTALL.with(|c| *c.borrow_mut() = None);
}

fn spin_until_stopped(ctl: &WarmCtl) {
    let skip = ctl.late.load(Ordering::Acquire) || ctl.stop.load(Ordering::Acquire);
    ctl.started.fetch_add(1, Ordering::AcqRel);
    if skip {
        return;
    }
    let t0 = Instant::now();
    let mut n = 0u32;
    // `yield_now` runs pending pool jobs (the arm's root job and its stolen halves) in this
    // loop, so a spinning worker is also an eager one.
    while !ctl.stop.load(Ordering::Acquire) && t0.elapsed() < ctl.budget {
        match rayon::yield_now() {
            Some(rayon::Yield::Executed) => n = 0,
            _ => spin_step(&mut n),
        }
    }
}

/// [`ThreadPool::install`] for a caller that is not a worker of `pool`, minus the Condvar wake of
/// the caller: the closure is spawned on the pool and the caller polls for its completion for up
/// to `spin` (with a `sched_yield` every few pauses), then parks. From inside a worker of `pool`
/// it is plain `install`. A panic of `f` is caught on the worker and resumed on the caller, as
/// `install` does.
pub fn install_spin<R: Send>(
    pool: &rayon::ThreadPool,
    spin: Duration,
    f: impl FnOnce() -> R + Send,
) -> R {
    if pool.current_thread_index().is_some() {
        return pool.install(f);
    }
    if let Some(ctl) = NEXT_INSTALL.with(|c| c.borrow_mut().take()) {
        let t0 = Instant::now();
        let mut n = 0u32;
        while ctl.started.load(Ordering::Acquire) < ctl.workers && t0.elapsed() < WARM_START_WAIT {
            spin_step(&mut n);
        }
        ctl.late.store(true, Ordering::Release);
    }
    struct Slot<R> {
        done: AtomicBool,
        result: std::cell::UnsafeCell<Option<std::thread::Result<R>>>,
        waiter: std::thread::Thread,
    }
    // SAFETY: `result` is written once by the worker before the Release store of `done` and read
    // once by the caller after its Acquire load of `done`.
    unsafe impl<R: Send> Sync for Slot<R> {}
    let slot = Arc::new(Slot::<R> {
        done: AtomicBool::new(false),
        result: std::cell::UnsafeCell::new(None),
        waiter: std::thread::current(),
    });
    let job_slot = slot.clone();
    let job: Box<dyn FnOnce() + Send + '_> = Box::new(move || {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        // SAFETY: sole writer, before `done` is published.
        unsafe { *job_slot.result.get() = Some(r) };
        job_slot.done.store(true, Ordering::Release);
        job_slot.waiter.unpark();
    });
    // SAFETY: `f` (and every borrow it holds) is consumed by the job body before `done` is set,
    // and this function does not return before it observes `done`; the job's remaining captures
    // (the `Arc`) are `'static`. The wait loop below cannot unwind.
    let job: Box<dyn FnOnce() + Send + 'static> = unsafe { std::mem::transmute(job) };
    pool.spawn(job);
    let t0 = Instant::now();
    let mut n = 0u32;
    while !slot.done.load(Ordering::Acquire) {
        if t0.elapsed() < spin {
            spin_step(&mut n);
        } else {
            std::thread::park_timeout(Duration::from_millis(1));
        }
    }
    // SAFETY: `done` was observed with Acquire; the worker wrote `result` before it.
    let r =
        unsafe { (*slot.result.get()).take() }.expect("splice pool job finished without a result");
    match r {
        Ok(v) => v,
        Err(p) => std::panic::resume_unwind(p),
    }
}

fn build_pool(cfg: SplicePoolConfig) -> rayon::ThreadPool {
    let fallback_threads = || {
        let n = if cfg.threads > 0 {
            cfg.threads
        } else {
            (std::thread::available_parallelism().map_or(2, |n| n.get()) / 2).max(1)
        };
        n.max(1)
    };
    let unpinned = |n: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("hipfire-splice-{i}"))
            .build()
            .expect("splice pool: thread pool build failed")
    };
    if !cfg.pin {
        let n = if cfg.threads > 0 {
            cfg.threads
        } else {
            physical_core_cpus().map_or_else(fallback_threads, |c| c.primaries.len().max(1))
        };
        eprintln!("cpu exec: splice pool {n} threads, unpinned (HIPFIRE_CPU_EXPERT_PIN=0)");
        return unpinned(n);
    }
    let Some(cpus) = physical_core_cpus() else {
        let n = fallback_threads();
        eprintln!(
            "cpu exec: splice pool falling back to {n} unpinned threads (CPU topology unreadable)"
        );
        return unpinned(n);
    };
    let cores = cpus.primaries.len();
    let n = if cfg.threads > 0 { cfg.threads } else { cores };
    // Workers 0..cores take one CPU per physical core; beyond that (explicit
    // override) they take the remaining SMT siblings, and past the allowed set
    // they run unpinned.
    let order: Vec<usize> = cpus
        .primaries
        .iter()
        .chain(cpus.secondaries.iter())
        .copied()
        .collect();
    let failed = std::sync::Arc::new(AtomicBool::new(false));
    let handler_failed = failed.clone();
    let handler_order = order.clone();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("hipfire-splice-{i}"))
        .start_handler(move |i| {
            if let Some(&cpu) = handler_order.get(i) {
                if !pin_current_thread(cpu) {
                    handler_failed.store(true, Ordering::Release);
                }
            }
        })
        .build()
        .expect("splice pool: thread pool build failed");
    // `start_handler` runs on each worker before it takes work; a broadcast
    // completes only after every worker has started, so `failed` is final here.
    pool.broadcast(|_| {});
    if failed.load(Ordering::Acquire) {
        let n = fallback_threads();
        eprintln!("cpu exec: splice pool pinning failed; using {n} unpinned threads");
        drop(pool);
        return unpinned(n);
    }
    eprintln!(
        "cpu exec: splice pool {n} threads pinned ({cores} physical cores in the allowed CPU set)"
    );
    pool
}

/// Allowed logical CPUs grouped by physical core.
struct PhysicalCpus {
    /// One allowed CPU (the lowest id) per distinct physical core, by core.
    primaries: Vec<usize>,
    /// The remaining allowed SMT siblings, in ascending id order.
    secondaries: Vec<usize>,
}

fn physical_core_cpus() -> Option<PhysicalCpus> {
    let allowed = allowed_cpus()?;
    let mut cores: Vec<(usize, usize)> = Vec::new(); // (core key, chosen cpu)
    let mut secondaries = Vec::new();
    for &cpu in &allowed {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        let text = std::fs::read_to_string(path).ok()?;
        let key = parse_cpu_list(text.trim())?.into_iter().min()?;
        if cores.iter().any(|&(k, _)| k == key) {
            secondaries.push(cpu);
        } else {
            cores.push((key, cpu));
        }
    }
    if cores.is_empty() {
        return None;
    }
    Some(PhysicalCpus {
        primaries: cores.into_iter().map(|(_, c)| c).collect(),
        secondaries,
    })
}

/// Parse a kernel CPU list such as `0,16` or `0-3,8`.
fn parse_cpu_list(s: &str) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return None;
        }
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b): (usize, usize) = (a.parse().ok()?, b.parse().ok()?);
                if a > b {
                    return None;
                }
                out.extend(a..=b);
            }
            None => out.push(part.parse().ok()?),
        }
    }
    Some(out)
}

#[cfg(target_os = "linux")]
fn allowed_cpus() -> Option<Vec<usize>> {
    // SAFETY: `set` is a plain bitmask the kernel fills; size matches the call.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return None;
        }
        let v: Vec<usize> = (0..libc::CPU_SETSIZE as usize)
            .filter(|&c| libc::CPU_ISSET(c, &set))
            .collect();
        (!v.is_empty()).then_some(v)
    }
}

#[cfg(not(target_os = "linux"))]
fn allowed_cpus() -> Option<Vec<usize>> {
    None
}

#[cfg(target_os = "linux")]
fn pin_current_thread(cpu: usize) -> bool {
    // SAFETY: `set` is initialised zeroed and only the target bit is set.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_current_thread(_cpu: usize) -> bool {
    false
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn cpu_list_parses_ranges_and_commas() {
        assert_eq!(parse_cpu_list("0,16"), Some(vec![0, 16]));
        assert_eq!(parse_cpu_list("0-3,8"), Some(vec![0, 1, 2, 3, 8]));
        assert_eq!(parse_cpu_list("5"), Some(vec![5]));
        assert_eq!(parse_cpu_list(""), None);
        assert_eq!(parse_cpu_list("3-1"), None);
        assert_eq!(parse_cpu_list("a"), None);
    }

    #[test]
    fn splice_pool_runs_work() {
        let pool = splice_pool(SplicePoolConfig {
            threads: 0,
            pin: true,
        });
        let sum: usize = pool.install(|| {
            use rayon::prelude::*;
            (0..1000usize).into_par_iter().sum()
        });
        assert_eq!(sum, 499_500);
        assert!(pool.current_num_threads() >= 1);
    }

    fn pool() -> &'static rayon::ThreadPool {
        splice_pool(SplicePoolConfig {
            threads: 0,
            pin: true,
        })
    }

    /// The spin install returns exactly what the plain install computes, on the pool's workers.
    #[test]
    fn install_spin_matches_install() {
        use rayon::prelude::*;
        let work = || {
            (0..100_000u64)
                .into_par_iter()
                .map(|x| x.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 7)
                .reduce(|| 0, u64::wrapping_add)
        };
        let want = pool().install(work);
        for _ in 0..20 {
            assert_eq!(install_spin(pool(), Duration::from_micros(500), work), want);
        }
        // Past its spin budget the caller parks and still sees the result.
        let slow = install_spin(pool(), Duration::ZERO, || {
            std::thread::sleep(Duration::from_millis(5));
            7u32
        });
        assert_eq!(slow, 7);
        let on_worker = install_spin(pool(), Duration::ZERO, || rayon::current_thread_index());
        assert!(on_worker.is_some());
    }

    /// Borrowed data and a panic both cross the spin install like `install`.
    #[test]
    fn install_spin_borrows_and_propagates_panics() {
        let mut v = vec![1u32, 2, 3];
        let total = install_spin(pool(), Duration::from_micros(100), || {
            v.push(4);
            v.iter().sum::<u32>()
        });
        assert_eq!((total, v.len()), (10, 4));
        let caught = std::panic::catch_unwind(|| {
            install_spin(pool(), Duration::from_micros(100), || -> u32 {
                panic!("splice arm failed")
            })
        });
        assert!(caught.is_err());
        // The pool is intact afterwards.
        assert_eq!(install_spin(pool(), Duration::from_micros(100), || 5), 5);
    }

    /// A warm pool with a pending-arm handshake: the spinners are all in before the arm is
    /// published, the arm runs and returns the same value, and dropping the warm releases them.
    #[test]
    fn warm_pool_handshake_and_release() {
        use rayon::prelude::*;
        let cfg = SplicePoolConfig {
            threads: 0,
            pin: true,
        };
        let work = || (0..50_000u64).into_par_iter().sum::<u64>();
        let want = pool().install(work);
        for _ in 0..20 {
            let warm = PoolWarm::start(cfg, Duration::from_millis(5));
            warm.arm_next_install();
            assert_eq!(install_spin(pool(), Duration::from_micros(500), work), want);
            assert_eq!(
                warm.ctl.started.load(Ordering::Acquire) <= warm.ctl.workers,
                true
            );
            drop(warm);
        }
        // A warm that expires unused, and a handshake that never installs, leave nothing behind.
        let warm = PoolWarm::start(cfg, Duration::from_micros(200));
        std::thread::sleep(Duration::from_millis(2));
        warm.arm_next_install();
        clear_next_install();
        drop(warm);
        assert_eq!(install_spin(pool(), Duration::from_micros(500), work), want);
    }
}
