// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generic per-rank replicated execution: one persistent worker thread per
//! peer rank, rank 0 on the caller's thread.
//!
//! A [`RankPool`] fans one job out to every peer (and, when supplied, runs a
//! local closure for rank 0 on the caller's thread), then compares every
//! host-visible result with [`agree`]. The first failure or panic on any rank
//! records the pool as poisoned and releases every rank blocked on a barrier
//! through the caller-installed abort callback, so no rank is left waiting on a
//! dead peer.
//!
//! The machinery is generic over the peer state `P` so it is testable without
//! a device; a family instantiates it with its own per-rank handle (for
//! example, `Qwen4EpPeer`).

use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

pub fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// A job every targeted worker runs on its peer. Implementations keep their
/// own per-rank result slots; the `Err` is the rank's failure text.
trait PoolJob<P>: Sync {
    fn run(&self, rank: usize, peer: &mut P) -> Result<(), String>;
}

/// Lifetime-erased pointer to a job on the caller's stack. The caller waits on
/// the job's [`Latch`] before the job (or anything it borrows) is dropped, and
/// workers touch the job only before counting the latch down.
struct JobPtr<P: 'static>(*const (dyn PoolJob<P> + 'static));

impl<P: 'static> Clone for JobPtr<P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: 'static> Copy for JobPtr<P> {}

// SAFETY: the pointee is `Sync` (`PoolJob: Sync`) and outlives every use (see
// `JobPtr`); the pointer is only dereferenced as `&dyn PoolJob`.
unsafe impl<P: 'static> Send for JobPtr<P> {}

enum Msg<P: 'static> {
    Job { job: JobPtr<P>, latch: Arc<Latch> },
    Stop,
}

/// Count-down completion latch.
struct Latch {
    left: Mutex<usize>,
    done: Condvar,
}

impl Latch {
    fn new(count: usize) -> Self {
        Self {
            left: Mutex::new(count),
            done: Condvar::new(),
        }
    }

    fn count_down(&self) {
        let mut left = lock(&self.left);
        *left = left.saturating_sub(1);
        if *left == 0 {
            self.done.notify_all();
        }
    }

    fn wait(&self) {
        let mut left = lock(&self.left);
        while *left > 0 {
            left = self
                .done
                .wait(left)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }
}

/// First-failure record shared by the caller and every worker.
struct Shared {
    poison: Mutex<Option<String>>,
    /// Releases every rank blocked on a barrier: one callback per registered
    /// barrier family (row gather, tile exchange, trunk split, ...), all run on
    /// every abort. Callbacks must be idempotent.
    aborts: Mutex<Vec<Arc<dyn Fn() + Send + Sync>>>,
}

impl Shared {
    /// Record the first failure (chronologically) and abort the barrier.
    fn fail(&self, rank: usize, message: &str) {
        {
            let mut poison = lock(&self.poison);
            if poison.is_none() {
                *poison = Some(format!("rank {rank}: {message}"));
            }
        }
        self.run_aborts();
    }

    /// Run every registered abort callback, outside the registry lock so a
    /// callback may register or abort again without deadlocking.
    fn run_aborts(&self) {
        let aborts = lock(&self.aborts).clone();
        for abort in aborts {
            abort();
        }
    }

    fn poison(&self) -> Option<String> {
        lock(&self.poison).clone()
    }

    /// Record a failure that belongs to no single rank (the ranks diverged)
    /// and abort the barrier. The first failure wins.
    fn fail_mesh(&self, message: &str) {
        {
            let mut poison = lock(&self.poison);
            if poison.is_none() {
                *poison = Some(message.to_string());
            }
        }
        self.run_aborts();
    }
}

/// Peers load one after another in ascending rank order: the loader's host-RAM
/// check for pinned experts assumes the lower ranks have already pinned theirs.
/// A failed load releases every later rank without loading it.
struct Baton {
    /// Index of the worker allowed to construct next; `None` once one failed.
    turn: Mutex<Option<usize>>,
    changed: Condvar,
}

impl Baton {
    fn wait(&self, index: usize) -> bool {
        let mut turn = lock(&self.turn);
        while matches!(*turn, Some(next) if next < index) {
            turn = self
                .changed
                .wait(turn)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        *turn == Some(index)
    }

    fn pass(&self, index: usize, built: bool) {
        *lock(&self.turn) = built.then_some(index + 1);
        self.changed.notify_all();
    }
}

struct Worker<P: 'static> {
    tx: Sender<Msg<P>>,
    handle: Option<JoinHandle<Result<(), String>>>,
}

struct RunJob<'a, P, R> {
    f: &'a (dyn Fn(usize, &mut P) -> Result<R, String> + Sync),
    /// Indexed by worker (`rank - 1`); `None` = never ran or panicked.
    results: Mutex<Vec<Option<Result<R, String>>>>,
}

impl<P, R: Send> PoolJob<P> for RunJob<'_, P, R> {
    fn run(&self, rank: usize, peer: &mut P) -> Result<(), String> {
        let result = (self.f)(rank, peer);
        let status = result.as_ref().map(|_| ()).map_err(String::clone);
        lock(&self.results)[rank - 1] = Some(result);
        status
    }
}

fn worker_main<P: 'static>(
    rank: usize,
    init: Box<dyn FnOnce(usize) -> Result<P, String> + Send>,
    rx: Receiver<Msg<P>>,
    ready: Sender<(usize, Result<(), String>)>,
    shared: Arc<Shared>,
    baton: Arc<Baton>,
    teardown: fn(P) -> Result<(), String>,
) -> Result<(), String> {
    let index = rank - 1;
    if !baton.wait(index) {
        let _ = ready.send((
            rank,
            Err("not loaded: an earlier rank failed to load".to_string()),
        ));
        return Ok(());
    }
    let mut peer = match catch_unwind(AssertUnwindSafe(|| init(rank))) {
        Ok(Ok(peer)) => {
            let _ = ready.send((rank, Ok(())));
            baton.pass(index, true);
            peer
        }
        Ok(Err(error)) => {
            let _ = ready.send((rank, Err(error)));
            baton.pass(index, false);
            return Ok(());
        }
        Err(payload) => {
            let _ = ready.send((rank, Err(format!("panicked: {}", panic_message(&*payload)))));
            baton.pass(index, false);
            return Ok(());
        }
    };
    drop(ready);
    while let Ok(Msg::Job { job, latch }) = rx.recv() {
        // SAFETY: see `JobPtr`; the caller blocks on `latch` until we count
        // down, and we do not touch the job afterwards.
        let job: &(dyn PoolJob<P> + 'static) = unsafe { &*job.0 };
        match catch_unwind(AssertUnwindSafe(|| job.run(rank, &mut peer))) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => shared.fail(rank, &error),
            Err(payload) => shared.fail(rank, &format!("panicked: {}", panic_message(&*payload))),
        }
        latch.count_down();
    }
    match catch_unwind(AssertUnwindSafe(|| teardown(peer))) {
        Ok(result) => result,
        Err(payload) => Err(format!(
            "rank {rank}: teardown panicked: {}",
            panic_message(&*payload)
        )),
    }
}

/// Persistent worker threads, one per peer rank (`rank = worker index + 1`),
/// each owning a peer value `P` constructed on its own thread.
pub struct RankPool<P: 'static> {
    workers: Vec<Worker<P>>,
    shared: Arc<Shared>,
    /// One call in flight at a time (calls are `&self`).
    call: Mutex<()>,
}

impl<P: 'static> RankPool<P> {
    /// Spawn one worker per `init`; worker `i` is rank `i + 1` and builds its
    /// peer by calling `init(rank)` on its own thread, one rank at a time in
    /// ascending order. Returns once every peer exists, or the first
    /// construction error (after stopping the rest). Worker threads are named
    /// `{thread_name}-rank{rank}`.
    /// `teardown` consumes a peer on its thread at close/drop.
    pub fn spawn(
        thread_name: &str,
        inits: Vec<Box<dyn FnOnce(usize) -> Result<P, String> + Send>>,
        teardown: fn(P) -> Result<(), String>,
    ) -> Result<Self, String> {
        let shared = Arc::new(Shared {
            poison: Mutex::new(None),
            aborts: Mutex::new(Vec::new()),
        });
        let (ready_tx, ready_rx) = channel();
        let baton = Arc::new(Baton {
            turn: Mutex::new(Some(0)),
            changed: Condvar::new(),
        });
        let count = inits.len();
        let mut pool = Self {
            workers: Vec::with_capacity(count),
            shared: shared.clone(),
            call: Mutex::new(()),
        };
        for (index, init) in inits.into_iter().enumerate() {
            let rank = index + 1;
            let (tx, rx) = channel();
            let ready = ready_tx.clone();
            let shared = shared.clone();
            let baton = baton.clone();
            let handle = std::thread::Builder::new()
                .name(format!("{thread_name}-rank{rank}"))
                .spawn(move || worker_main(rank, init, rx, ready, shared, baton, teardown))
                .map_err(|error| format!("spawn rank {rank} worker: {error}"))?;
            pool.workers.push(Worker {
                tx,
                handle: Some(handle),
            });
        }
        drop(ready_tx);
        let mut first = None;
        for _ in 0..count {
            match ready_rx.recv() {
                Ok((_, Ok(()))) => {}
                Ok((rank, Err(error))) => {
                    first.get_or_insert(format!("rank {rank}: {error}"));
                }
                Err(_) => {
                    first.get_or_insert("a worker exited during construction".to_string());
                    break;
                }
            }
        }
        match first {
            // `pool` drops here: constructed peers are torn down, workers joined.
            Some(error) => Err(error),
            None => Ok(pool),
        }
    }

    /// Register a barrier-abort callback. Every registered callback runs on every
    /// failure or mesh poison, so each barrier family (row gather, tile exchange,
    /// trunk split) registers its own and none is left waiting out its timeout.
    /// Callbacks must be idempotent: a later registration does not replace
    /// earlier ones, and one failure may run them repeatedly.
    pub fn set_abort(&self, abort: impl Fn() + Send + Sync + 'static) {
        lock(&self.shared.aborts).push(Arc::new(abort));
    }

    /// The first recorded failure, if the pool is poisoned.
    pub fn poisoned(&self) -> Option<String> {
        self.shared.poison()
    }

    /// Poison the pool with a failure no single rank owns.
    pub fn poison_mesh(&self, message: &str) {
        self.shared.fail_mesh(message);
    }

    /// Run `f` on the workers `targets` (indices) while `local` runs on the
    /// calling thread; returns once every targeted worker finished. A failure
    /// or panic anywhere poisons the pool and aborts the barrier at once.
    #[allow(clippy::type_complexity)]
    fn exec<R: Send, L: FnOnce() -> Result<R, String>>(
        &self,
        targets: &[usize],
        f: &(dyn Fn(usize, &mut P) -> Result<R, String> + Sync),
        local: Option<L>,
        check_poison: bool,
    ) -> Result<(Option<R>, Vec<Option<R>>), String> {
        let _call = lock(&self.call);
        if check_poison {
            if let Some(poison) = self.shared.poison() {
                return Err(format!(
                    "rank pool is poisoned by an earlier failure (reload the model): {poison}"
                ));
            }
        }
        let job = RunJob {
            f,
            results: Mutex::new((0..self.workers.len()).map(|_| None).collect()),
        };
        let latch = Arc::new(Latch::new(targets.len()));
        let erased: &(dyn PoolJob<P> + '_) = &job;
        // SAFETY: `latch.wait()` below runs on every path (the local call is
        // unwind-contained) before `job` and its borrows are dropped.
        let ptr = JobPtr(unsafe {
            std::mem::transmute::<*const (dyn PoolJob<P> + '_), *const (dyn PoolJob<P> + 'static)>(
                erased as *const _,
            )
        });
        for &index in targets {
            let sent = self.workers[index].tx.send(Msg::Job {
                job: ptr,
                latch: latch.clone(),
            });
            if sent.is_err() {
                latch.count_down();
                self.shared.fail(
                    index + 1,
                    "worker thread is gone (it exited or panicked earlier)",
                );
            }
        }
        let local_result = local.map(|local| match catch_unwind(AssertUnwindSafe(local)) {
            Ok(result) => result,
            Err(payload) => Err(format!("panicked: {}", panic_message(&*payload))),
        });
        if let Some(Err(error)) = &local_result {
            self.shared.fail(0, error);
        }
        latch.wait();
        let slots = std::mem::take(&mut *lock(&job.results));
        let mut failure = None;
        let mut peer_values = Vec::with_capacity(slots.len());
        let local_value = match local_result {
            Some(Ok(value)) => Some(value),
            Some(Err(error)) => {
                failure = Some(format!("rank 0: {error}"));
                None
            }
            None => None,
        };
        for (index, slot) in slots.into_iter().enumerate() {
            match slot {
                Some(Ok(value)) => peer_values.push(Some(value)),
                Some(Err(error)) => {
                    failure.get_or_insert(format!("rank {}: {error}", index + 1));
                    peer_values.push(None);
                }
                None => {
                    if targets.contains(&index) {
                        failure.get_or_insert(format!("rank {}: job did not complete", index + 1));
                    }
                    peer_values.push(None);
                }
            }
        }
        match failure {
            None => Ok((local_value, peer_values)),
            Some(own) => Err(self.shared.poison().unwrap_or(own)),
        }
    }

    /// Run `f` on every peer while `local` runs on the caller. Results in rank
    /// order (`[rank 0, rank 1, ..]`). Fails fast if the pool is poisoned.
    pub fn run_all<R: Send>(
        &self,
        local: impl FnOnce() -> Result<R, String>,
        f: &(dyn Fn(usize, &mut P) -> Result<R, String> + Sync),
    ) -> Result<Vec<R>, String> {
        self.run_all_inner(local, f, true)
    }

    /// [`Self::run_all`] that ignores an earlier poisoning (teardown drains).
    pub fn run_all_forced<R: Send>(
        &self,
        local: impl FnOnce() -> Result<R, String>,
        f: &(dyn Fn(usize, &mut P) -> Result<R, String> + Sync),
    ) -> Result<Vec<R>, String> {
        self.run_all_inner(local, f, false)
    }

    fn run_all_inner<R: Send>(
        &self,
        local: impl FnOnce() -> Result<R, String>,
        f: &(dyn Fn(usize, &mut P) -> Result<R, String> + Sync),
        check_poison: bool,
    ) -> Result<Vec<R>, String> {
        let targets: Vec<usize> = (0..self.workers.len()).collect();
        let (local_value, peers) = self.exec(&targets, f, Some(local), check_poison)?;
        let mut values = Vec::with_capacity(1 + peers.len());
        values.extend(local_value);
        values.extend(peers.into_iter().flatten());
        Ok(values)
    }

    /// Run `f` on the single peer `rank` (1-based) while the caller waits.
    pub fn run_rank<R: Send>(
        &self,
        rank: usize,
        f: &(dyn Fn(usize, &mut P) -> Result<R, String> + Sync),
    ) -> Result<R, String> {
        if rank == 0 || rank > self.workers.len() {
            return Err(format!(
                "rank {rank} is not a peer (peers are 1..={})",
                self.workers.len()
            ));
        }
        let (_, mut peers) = self.exec(&[rank - 1], f, None::<fn() -> Result<R, String>>, true)?;
        peers
            .swap_remove(rank - 1)
            .ok_or_else(|| format!("rank {rank}: job produced no result"))
    }

    fn stop_and_join(&mut self) -> Result<(), String> {
        for worker in &self.workers {
            let _ = worker.tx.send(Msg::Stop);
        }
        let mut first = None;
        for (index, worker) in self.workers.iter_mut().enumerate() {
            let Some(handle) = worker.handle.take() else {
                continue;
            };
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    first.get_or_insert(format!("rank {}: {error}", index + 1));
                }
                Err(payload) => {
                    first.get_or_insert(format!(
                        "rank {}: worker panicked: {}",
                        index + 1,
                        panic_message(&*payload)
                    ));
                }
            }
        }
        self.workers.clear();
        first.map_or(Ok(()), Err)
    }

    /// Stop the workers: each tears its peer down on its own thread; returns
    /// the first teardown error.
    pub fn close(mut self) -> Result<(), String> {
        self.stop_and_join()
    }
}

impl<P: 'static> Drop for RankPool<P> {
    fn drop(&mut self) {
        if let Err(error) = self.stop_and_join() {
            eprintln!("  rank pool: peer teardown failed: {error}");
        }
    }
}

/// Require every rank's host-visible result to be identical; returns rank 0's.
/// `results` is in rank order.
pub fn agree<T: PartialEq + std::fmt::Debug>(what: &str, results: Vec<T>) -> Result<T, String> {
    if let Some(first) = results.first() {
        for (rank, other) in results.iter().enumerate().skip(1) {
            if other != first {
                return Err(format!(
                    "rank pool: {what} disagrees across ranks: rank 0 = {first:?}, rank {rank} = {other:?}"
                ));
            }
        }
    }
    results
        .into_iter()
        .next()
        .ok_or_else(|| format!("rank pool: {what} produced no result"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Fake {
        rank: usize,
        value: i64,
    }

    fn teardown_ok(_: Fake) -> Result<(), String> {
        Ok(())
    }

    fn pool(n: usize) -> RankPool<Fake> {
        let inits = (0..n)
            .map(|_| {
                Box::new(|rank: usize| {
                    Ok(Fake {
                        rank,
                        value: rank as i64 * 10,
                    })
                }) as Box<dyn FnOnce(usize) -> Result<Fake, String> + Send>
            })
            .collect();
        RankPool::spawn("test", inits, teardown_ok).expect("spawn")
    }

    #[test]
    fn runs_every_rank_concurrently_and_returns_in_rank_order() {
        let pool = pool(3);
        let barrier = std::sync::Barrier::new(4);
        let out = pool
            .run_all(
                || {
                    barrier.wait();
                    Ok(0i64)
                },
                &|_, peer: &mut Fake| {
                    // All four ranks must be inside the job at once.
                    barrier.wait();
                    peer.value += 1;
                    Ok(peer.value)
                },
            )
            .expect("run");
        assert_eq!(out, vec![0, 11, 21, 31]);
        // Peer state persists across calls.
        let out = pool
            .run_all(|| Ok(0), &|_, peer: &mut Fake| Ok(peer.value))
            .expect("run");
        assert_eq!(out, vec![0, 11, 21, 31]);
        pool.close().expect("close");
    }

    #[test]
    fn job_may_borrow_caller_stack_data() {
        let pool = pool(2);
        let tokens = vec![3u32, 5, 7];
        let sum = |t: &[u32]| t.iter().sum::<u32>() as i64;
        let out = pool
            .run_all(|| Ok(sum(&tokens)), &|_, peer: &mut Fake| {
                Ok(sum(&tokens) + peer.rank as i64)
            })
            .expect("run");
        assert_eq!(out, vec![15, 16, 17]);
    }

    #[test]
    fn peer_error_poisons_aborts_and_fails_later_calls_fast() {
        let pool = pool(2);
        let aborts = Arc::new(AtomicUsize::new(0));
        let counter = aborts.clone();
        pool.set_abort(move || {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        let error = pool
            .run_all(|| Ok(()), &|rank, _: &mut Fake| {
                if rank == 2 {
                    Err("boom".to_string())
                } else {
                    Ok(())
                }
            })
            .expect_err("rank 2 fails");
        assert!(
            error.contains("rank 2") && error.contains("boom"),
            "{error}"
        );
        assert!(aborts.load(Ordering::SeqCst) >= 1);
        let again = pool
            .run_all(|| Ok(()), &|_, _: &mut Fake| Ok(()))
            .expect_err("poisoned");
        assert!(
            again.contains("poisoned") && again.contains("boom"),
            "{again}"
        );
        assert!(pool.poisoned().is_some());
        // A forced run (teardown drain) still reaches the workers.
        pool.run_all_forced(|| Ok(()), &|_, _: &mut Fake| Ok(()))
            .expect("forced run ignores poison");
    }

    #[test]
    fn local_error_aborts_before_peers_finish_and_reports_the_first_failure() {
        let pool = pool(1);
        let aborted = Arc::new(AtomicUsize::new(0));
        let flag = aborted.clone();
        pool.set_abort(move || {
            flag.fetch_add(1, Ordering::SeqCst);
        });
        let seen_abort = aborted.clone();
        let error = pool
            .run_all(
                || Err::<(), _>("rank0 failed".to_string()),
                &move |_, _: &mut Fake| {
                    // The peer waits for the abort that rank 0's failure raises
                    // (what a rank blocked on the barrier relies on).
                    let start = std::time::Instant::now();
                    while seen_abort.load(Ordering::SeqCst) == 0 {
                        if start.elapsed() > std::time::Duration::from_secs(5) {
                            return Err("never aborted".to_string());
                        }
                        std::thread::yield_now();
                    }
                    Err("barrier aborted".to_string())
                },
            )
            .expect_err("fails");
        assert!(
            error.contains("rank 0") && error.contains("rank0 failed"),
            "{error}"
        );
    }

    #[test]
    fn peer_panic_becomes_an_error_and_the_worker_survives_for_teardown() {
        let pool = pool(2);
        let error = pool
            .run_all(|| Ok(()), &|rank, _: &mut Fake| {
                if rank == 1 {
                    panic!("kernel exploded");
                }
                Ok(())
            })
            .expect_err("panic is an error");
        assert!(
            error.contains("rank 1") && error.contains("kernel exploded"),
            "{error}"
        );
        pool.run_all_forced(|| Ok(()), &|_, _: &mut Fake| Ok(()))
            .expect("workers still answer");
        pool.close().expect("close");
    }

    #[test]
    fn local_panic_is_contained_and_peers_are_still_joined() {
        let pool = pool(2);
        let finished = Arc::new(AtomicUsize::new(0));
        let counter = finished.clone();
        let error = pool
            .run_all(
                || -> Result<(), String> { panic!("rank0 panic") },
                &move |_, _: &mut Fake| {
                    std::thread::sleep(std::time::Duration::from_millis(30));
                    counter.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                },
            )
            .expect_err("local panic");
        assert!(error.contains("rank0 panic"), "{error}");
        // The call returned only after both peers finished the borrowed job.
        assert_eq!(finished.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn run_rank_targets_one_peer() {
        let pool = pool(3);
        let value = pool
            .run_rank(2, &|rank, peer: &mut Fake| Ok((rank, peer.value)))
            .expect("run");
        assert_eq!(value, (2, 20));
        assert!(pool.run_rank(0, &|_, _: &mut Fake| Ok(())).is_err());
        assert!(pool.run_rank(4, &|_, _: &mut Fake| Ok(())).is_err());
    }

    #[test]
    fn construction_failure_tears_down_the_peers_that_did_build() {
        static TORN_DOWN: AtomicUsize = AtomicUsize::new(0);
        fn count(_: Fake) -> Result<(), String> {
            TORN_DOWN.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        let inits: Vec<Box<dyn FnOnce(usize) -> Result<Fake, String> + Send>> = vec![
            Box::new(|rank| Ok(Fake { rank, value: 0 })),
            Box::new(|_| Err("load failed".to_string())),
            Box::new(|rank| Ok(Fake { rank, value: 0 })),
        ];
        let error = RankPool::spawn("test", inits, count)
            .err()
            .expect("spawn fails");
        assert!(
            error.contains("rank 2") && error.contains("load failed"),
            "{error}"
        );
        // Rank 1 built and is torn down; rank 3 never loads after rank 2 failed.
        assert_eq!(TORN_DOWN.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn peers_construct_one_at_a_time_in_ascending_rank_order() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let active = Arc::new(AtomicUsize::new(0));
        let inits: Vec<Box<dyn FnOnce(usize) -> Result<Fake, String> + Send>> = (0..4)
            .map(|_| {
                let order = order.clone();
                let active = active.clone();
                Box::new(move |rank: usize| {
                    assert_eq!(
                        active.fetch_add(1, Ordering::SeqCst),
                        0,
                        "two loads overlapped"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(10 * (5 - rank as u64)));
                    lock(&order).push(rank);
                    active.fetch_sub(1, Ordering::SeqCst);
                    Ok(Fake { rank, value: 0 })
                }) as Box<dyn FnOnce(usize) -> Result<Fake, String> + Send>
            })
            .collect();
        let pool = RankPool::spawn("test", inits, teardown_ok).expect("spawn");
        assert_eq!(*lock(&order), vec![1, 2, 3, 4]);
        pool.close().expect("close");
    }

    #[test]
    fn a_failed_load_skips_the_later_ranks() {
        let loaded = Arc::new(AtomicUsize::new(0));
        let inits: Vec<Box<dyn FnOnce(usize) -> Result<Fake, String> + Send>> = (0..3)
            .map(|_| {
                let loaded = loaded.clone();
                Box::new(move |rank: usize| {
                    loaded.fetch_add(1, Ordering::SeqCst);
                    if rank == 2 {
                        Err("out of host memory".to_string())
                    } else {
                        Ok(Fake { rank, value: 0 })
                    }
                }) as Box<dyn FnOnce(usize) -> Result<Fake, String> + Send>
            })
            .collect();
        let error = RankPool::spawn("test", inits, teardown_ok)
            .err()
            .expect("fails");
        assert!(
            error.contains("rank 2") && error.contains("out of host memory"),
            "{error}"
        );
        assert_eq!(loaded.load(Ordering::SeqCst), 2, "rank 3 must not load");
    }

    #[test]
    fn close_reports_a_teardown_error() {
        fn bad(peer: Fake) -> Result<(), String> {
            if peer.rank == 2 {
                Err("free failed".to_string())
            } else {
                Ok(())
            }
        }
        let inits = (0..2)
            .map(|_| {
                Box::new(|rank| Ok(Fake { rank, value: 0 }))
                    as Box<dyn FnOnce(usize) -> Result<Fake, String> + Send>
            })
            .collect();
        let pool = RankPool::spawn("test", inits, bad).expect("spawn");
        let error = pool.close().expect_err("teardown error surfaces");
        assert!(
            error.contains("rank 2") && error.contains("free failed"),
            "{error}"
        );
    }

    #[test]
    fn agreement_names_both_values_on_disagreement() {
        assert_eq!(agree("token", vec![7u32, 7, 7]).unwrap(), 7);
        assert_eq!(agree("unit", vec![(), ()]).unwrap(), ());
        let error = agree("token", vec![7u32, 7, 9, 7]).unwrap_err();
        assert!(
            error.contains("token") && error.contains("rank 0 = 7") && error.contains("rank 2 = 9"),
            "{error}"
        );
        assert!(agree::<u32>("none", Vec::new()).is_err());
    }

    #[test]
    fn every_registered_abort_callback_runs_on_each_failure() {
        let pool = pool(1);
        let counts: Vec<Arc<AtomicUsize>> = (0..3).map(|_| Arc::new(AtomicUsize::new(0))).collect();
        for count in &counts {
            let count = count.clone();
            pool.set_abort(move || {
                count.fetch_add(1, Ordering::SeqCst);
            });
        }
        pool.run_all(|| Ok(()), &|_, _: &mut Fake| {
            Err::<(), _>("boom".to_string())
        })
        .expect_err("the peer fails");
        for count in &counts {
            assert!(count.load(Ordering::SeqCst) >= 1, "every family is aborted");
        }
        let before: Vec<usize> = counts.iter().map(|c| c.load(Ordering::SeqCst)).collect();
        pool.poison_mesh("diverged");
        for (count, before) in counts.iter().zip(before) {
            assert!(
                count.load(Ordering::SeqCst) > before,
                "a mesh poison aborts too"
            );
        }
    }
}
