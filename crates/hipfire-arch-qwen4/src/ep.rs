// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next expert-parallel (EP) mesh: one persistent worker thread per peer
//! rank.
//!
//! Rank 0 is the [`Qwen4Bundle`] and [`Gpu`] the daemon already drives, on the
//! caller's thread. Ranks `1..N` are peer bundles owned by worker threads that
//! construct them (a `Gpu` never crosses threads). Every daemon-facing bundle
//! call is replicated: the caller runs rank 0 itself while the workers run the
//! same call on their own bundle, then every host result is compared.
//!
//! The thread/latch/poison machinery ([`PeerPool`]) is generic over the peer
//! state so it is testable without a device. [`Qwen4EpMesh`] is the Qwen4
//! instantiation: peers are [`Qwen4EpPeer`] and a failure on any rank aborts
//! the row-gather barrier so no rank is left waiting on a dead peer. The mesh is
//! rank-generic over EP2..=EP8 (`attach_ep_mesh` admits `1 + peers` of 2..=8).

use crate::bundle::{BundleError, Qwen4Bundle};
use crate::gpu_forward::{
    qwen4_ep_buffer_index, Qwen4BalancedPlan, Qwen4PeerExpertSource, QWEN4_EP_GATHER_BUFFERS,
    QWEN4_EP_GATHER_MAX_ROWS,
};
use crate::weights::Qwen4EpPlacement;
use hipfire_dispatch::families::moe::{MoeRowGather, MoeTileExchange};
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::ep_row_gather::{EpRowGather, EpRowGatherSpec, EpSlotOwners};
use hipfire_runtime::ep_tile_exchange::{EpTileExchange, EpTileExchangeSpec};
use hipfire_runtime::sampler::PenaltyTable;
use hipfire_runtime::serve_contract::{
    sha256_len_prefixed, ArchPolicy, CacheDomain, DeviceTopology, KvLayout, SharingNamespace,
    TemplateIdentity, TokenizerIdentity,
};
use hipfire_runtime::session_cache::{SessionCache, SessionRoute};
use rdna_compute::moe::{Qwen4EpOwnerTables, QWEN4_EP_OWNER_EXPERTS};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
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

// ───────────────────────────── generic peer pool ─────────────────────────────

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
    /// Releases every rank blocked on the row-gather barrier. Installed once
    /// the transport exists.
    abort: OnceLock<Box<dyn Fn() + Send + Sync>>,
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
        if let Some(abort) = self.abort.get() {
            abort();
        }
    }

    fn poison(&self) -> Option<String> {
        lock(&self.poison).clone()
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

/// Persistent worker threads, one per peer rank (`rank = worker index + 1`),
/// each owning a peer value `P` constructed on its own thread.
pub(crate) struct PeerPool<P: 'static> {
    workers: Vec<Worker<P>>,
    shared: Arc<Shared>,
    /// One call in flight at a time (calls are `&self`).
    call: Mutex<()>,
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

impl<P: 'static> PeerPool<P> {
    /// Spawn one worker per `init`; worker `i` is rank `i + 1` and builds its
    /// peer by calling `init(rank)` on its own thread, one rank at a time in
    /// ascending order. Returns once every peer exists, or the first
    /// construction error (after stopping the rest).
    /// `teardown` consumes a peer on its thread at close/drop.
    pub(crate) fn spawn(
        inits: Vec<Box<dyn FnOnce(usize) -> Result<P, String> + Send>>,
        teardown: fn(P) -> Result<(), String>,
    ) -> Result<Self, String> {
        let shared = Arc::new(Shared {
            poison: Mutex::new(None),
            abort: OnceLock::new(),
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
                .name(format!("hipfire-qwen4-ep-rank{rank}"))
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

    /// Install the barrier-abort callback (once).
    pub(crate) fn set_abort(&self, abort: impl Fn() + Send + Sync + 'static) {
        let _ = self.shared.abort.set(Box::new(abort));
    }

    /// The first recorded failure, if the pool is poisoned.
    pub(crate) fn poisoned(&self) -> Option<String> {
        self.shared.poison()
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
                    "qwen4 EP mesh is poisoned by an earlier failure (reload the model): {poison}"
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
    pub(crate) fn run_all<R: Send>(
        &self,
        local: impl FnOnce() -> Result<R, String>,
        f: &(dyn Fn(usize, &mut P) -> Result<R, String> + Sync),
    ) -> Result<Vec<R>, String> {
        self.run_all_inner(local, f, true)
    }

    /// [`Self::run_all`] that ignores an earlier poisoning (teardown drains).
    pub(crate) fn run_all_forced<R: Send>(
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
    pub(crate) fn run_rank<R: Send>(
        &self,
        rank: usize,
        f: &(dyn Fn(usize, &mut P) -> Result<R, String> + Sync),
    ) -> Result<R, String> {
        if rank == 0 || rank > self.workers.len() {
            return Err(format!(
                "qwen4 EP rank {rank} is not a peer (peers are 1..={})",
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
    pub(crate) fn close(mut self) -> Result<(), String> {
        self.stop_and_join()
    }
}

impl<P: 'static> Drop for PeerPool<P> {
    fn drop(&mut self) {
        if let Err(error) = self.stop_and_join() {
            eprintln!("  qwen4 EP: peer teardown failed: {error}");
        }
    }
}

/// Require every rank's host-visible result to be identical; returns rank 0's.
/// `results` is in rank order.
pub(crate) fn agree<T: PartialEq + std::fmt::Debug>(
    what: &str,
    results: Vec<T>,
) -> Result<T, String> {
    if let Some(first) = results.first() {
        for (rank, other) in results.iter().enumerate().skip(1) {
            if other != first {
                return Err(format!(
                    "qwen4 EP: {what} disagrees across ranks: rank 0 = {first:?}, rank {rank} = {other:?}"
                ));
            }
        }
    }
    results
        .into_iter()
        .next()
        .ok_or_else(|| format!("qwen4 EP: {what} produced no result"))
}

// ─────────────────────────────── tensor mirrors ───────────────────────────────

/// Shape and dtype of a rank-0 caller-owned device tensor; peers allocate a
/// same-shaped stand-in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl TensorSpec {
    pub fn of(tensor: &GpuTensor) -> Self {
        Self {
            shape: tensor.shape.clone(),
            dtype: tensor.dtype,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MirrorRole {
    Logits,
    Top1,
}

/// A peer's stand-ins for the rank-0 caller-owned device tensors of a call
/// (logits, top1): allocated lazily, cached by role + shape + dtype, freed at
/// teardown. Nobody reads a peer's mirror in serving; its logits equal rank
/// 0's bitwise.
#[derive(Default)]
pub struct PeerMirrors {
    entries: Vec<(MirrorRole, TensorSpec, GpuTensor)>,
    last_logits: Option<usize>,
    last_top1: Option<usize>,
    /// This peer's replica of rank 0's native MTP drafter, built on the first
    /// drafter fan-out ([`Qwen4Bundle::ep_drafter_fanout`]) and freed with the
    /// mirrors.
    drafter: Option<crate::mtp_spec::Qwen4MtpDrafter>,
}

impl PeerMirrors {
    /// This peer's drafter replica, built from `params` (rank 0's construction
    /// parameters) on first use.
    pub(crate) fn drafter(
        &mut self,
        params: crate::mtp_spec::ReplicaParams,
    ) -> &mut crate::mtp_spec::Qwen4MtpDrafter {
        self.drafter
            .get_or_insert_with(|| crate::mtp_spec::Qwen4MtpDrafter::new_replica(params))
    }

    fn ensure(
        &mut self,
        gpu: &mut Gpu,
        role: MirrorRole,
        spec: &TensorSpec,
    ) -> Result<usize, String> {
        if let Some(index) = self
            .entries
            .iter()
            .position(|(r, s, _)| *r == role && s == spec)
        {
            return Ok(index);
        }
        let tensor = gpu
            .zeros(&spec.shape, spec.dtype)
            .map_err(|error| format!("qwen4 EP mirror {role:?} {spec:?}: {error:?}"))?;
        self.entries.push((role, spec.clone(), tensor));
        Ok(self.entries.len() - 1)
    }

    /// The logits mirror for `spec` (allocated on first use) and, when the
    /// caller passed one, the top1 mirror.
    pub(crate) fn pair(
        &mut self,
        gpu: &mut Gpu,
        logits: &TensorSpec,
        top1: Option<&TensorSpec>,
    ) -> Result<(&GpuTensor, Option<&GpuTensor>), String> {
        let logits_index = self.ensure(gpu, MirrorRole::Logits, logits)?;
        self.last_logits = Some(logits_index);
        let top1_index = match top1 {
            Some(spec) => Some(self.ensure(gpu, MirrorRole::Top1, spec)?),
            None => None,
        };
        self.last_top1 = top1_index;
        Ok((
            &self.entries[logits_index].2,
            top1_index.map(|index| &self.entries[index].2),
        ))
    }

    /// The logits mirror of the most recent logits-taking call.
    pub fn logits(&self) -> Option<&GpuTensor> {
        self.last_logits.map(|index| &self.entries[index].2)
    }

    /// The top1 mirror of the most recent call that passed a top1 tensor.
    pub fn top1(&self) -> Option<&GpuTensor> {
        self.last_top1.map(|index| &self.entries[index].2)
    }

    fn free(self, gpu: &mut Gpu) -> Result<(), String> {
        if let Some(drafter) = self.drafter {
            <crate::mtp_spec::Qwen4MtpDrafter as hipfire_runtime::spec::MtpDrafter>::mtp_free(
                Box::new(drafter),
                gpu,
            );
        }
        let mut first = None;
        for (_, _, tensor) in self.entries {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(format!("free mirror: {error:?}"));
            }
        }
        first.map_or(Ok(()), Err)
    }
}

// ─────────────────────────────── Qwen4 mesh ───────────────────────────────

/// Loads one peer rank on the worker's own thread: rank → the peer's `Gpu`
/// (on the seal's physical device for the rank) and a fully attached bundle
/// bound to rank 0's seal at that rank (`Qwen4Bundle::bind_ep_seal`).
pub type PeerLoadFn = dyn Fn(usize) -> Result<(Gpu, Qwen4Bundle), String> + Send + Sync;

/// What a peer worker owns.
pub struct Qwen4EpPeer {
    rank: usize,
    gpu: Gpu,
    bundle: Qwen4Bundle,
    mirrors: PeerMirrors,
    gather: Option<Arc<EpRowGather>>,
    tile: Option<Arc<EpTileExchange>>,
}

impl Qwen4EpPeer {
    /// Drain the device, then free everything this rank owns.
    fn teardown(mut self) -> Result<(), String> {
        let mut first: Option<String> = None;
        let mut note = |result: Result<(), String>| {
            if let Err(error) = result {
                first.get_or_insert(format!("rank {}: {error}", self.rank));
            }
        };
        note(sync_device(&mut self.gpu));
        note(self.mirrors.free(&mut self.gpu));
        if let Some(gather) = self.gather.take() {
            note(
                gather
                    .release(self.rank, &mut self.gpu)
                    .map_err(|error| format!("release gather events: {error}")),
            );
        }
        if let Some(tile) = self.tile.take() {
            note(
                tile.release(self.rank, &mut self.gpu)
                    .map_err(|error| format!("release tile exchange events: {error}")),
            );
        }
        note(
            self.bundle
                .free_gpu(&mut self.gpu)
                .map_err(|error| format!("free bundle: {error}")),
        );
        first.map_or(Ok(()), Err)
    }
}

fn sync_device(gpu: &mut Gpu) -> Result<(), String> {
    gpu.bind_thread().map_err(|error| error.to_string())?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| format!("device synchronize: {error}"))
}

/// `last_buffer` before any invocation.
const NO_BUFFER: usize = usize::MAX;

/// The row-gather hook one rank installs into its forward
/// (`Qwen4GpuForward::attach_ep`).
pub(crate) struct Qwen4EpRowGatherHook {
    rank: usize,
    trunk_layers: usize,
    gather: Arc<EpRowGather>,
    /// Device addresses of this rank's gather buffers, to check the expanded
    /// rows the stage hands over are the buffer it exchanges.
    buffer_ptrs: [usize; QWEN4_EP_GATHER_BUFFERS],
    /// Buffer index of this rank's previous invocation. The transport's hazard
    /// rule forbids the same buffer on two consecutive invocations.
    last_buffer: AtomicUsize,
    /// Balanced-ownership layers (default on under EP; `HIPFIRE_EP_BALANCED_OWNERS=0`
    /// disables), installed once after the forward attached its live tables.
    /// Never set when disabled.
    balanced: OnceLock<BalancedOwners>,
}

/// What the hook knows of the rank's balanced trunk layers.
struct BalancedOwners {
    /// Owner-assign table addresses indexed by trunk layer label.
    tables: Vec<Option<Qwen4EpOwnerTables>>,
    /// Non-owning view of the rank's slot-owner buffer (the forward owns it).
    slot_owner: Mutex<GpuTensor>,
}

/// The owner-assign tables of a balanced trunk layer; `None` for every other
/// layer, including the MTP head range (`>= trunk_layers`), whatever
/// `tables` holds.
fn balanced_layer_tables(
    tables: &[Option<Qwen4EpOwnerTables>],
    layer_idx: u16,
    trunk_layers: usize,
) -> Option<&Qwen4EpOwnerTables> {
    let layer = usize::from(layer_idx);
    if layer >= trunk_layers {
        return None;
    }
    tables.get(layer).and_then(Option::as_ref)
}

impl BalancedOwners {
    fn layer(&self, layer_idx: u16, trunk_layers: usize) -> Option<&Qwen4EpOwnerTables> {
        balanced_layer_tables(&self.tables, layer_idx, trunk_layers)
    }
}

impl Qwen4EpRowGatherHook {
    fn new(
        rank: usize,
        trunk_layers: usize,
        gather: Arc<EpRowGather>,
        buffer_ptrs: [usize; QWEN4_EP_GATHER_BUFFERS],
    ) -> Self {
        Self {
            rank,
            trunk_layers,
            gather,
            buffer_ptrs,
            last_buffer: AtomicUsize::new(NO_BUFFER),
            balanced: OnceLock::new(),
        }
    }

    /// Install the rank's balanced layers (once, after `attach_ep`).
    fn install_balanced(&self, plan: Qwen4BalancedPlan) -> Result<(), String> {
        let mut tables: Vec<Option<Qwen4EpOwnerTables>> = vec![None; self.trunk_layers];
        for (label, layer_tables) in plan.layers {
            let slot = tables.get_mut(label).ok_or_else(|| {
                format!(
                    "qwen4 EP rank {}: balanced layer {label} is outside the {} trunk layers",
                    self.rank, self.trunk_layers
                )
            })?;
            *slot = Some(layer_tables);
        }
        self.balanced
            .set(BalancedOwners {
                tables,
                slot_owner: Mutex::new(plan.slot_owner),
            })
            .map_err(|_| {
                format!(
                    "qwen4 EP rank {}: balanced owners are already installed",
                    self.rank
                )
            })
    }
}

/// Record `buffer` as the rank's current invocation in `last`; `Err` when the
/// previous invocation used the same one.
fn note_gather_buffer(last: &AtomicUsize, rank: usize, buffer: usize) -> Result<(), String> {
    if last.swap(buffer, Ordering::Relaxed) == buffer {
        return Err(format!(
            "qwen4 EP row gather: gather buffer {buffer} of rank {rank} used by two consecutive invocations"
        ));
    }
    Ok(())
}

impl MoeRowGather for Qwen4EpRowGatherHook {
    fn gather_rows(
        &self,
        gpu: &mut Gpu,
        layer_idx: u16,
        slots: usize,
        down_expanded: &GpuTensor,
        topk_indices: &GpuTensor,
    ) -> Result<(), String> {
        let buffer = qwen4_ep_buffer_index(usize::from(layer_idx), self.trunk_layers);
        let ptr = down_expanded.buf.as_ptr() as usize;
        if self.buffer_ptrs.get(buffer) != Some(&ptr) {
            return Err(format!(
                "qwen4 EP row gather: layer {layer_idx} expanded rows at {ptr:#x} are not gather buffer {buffer} of rank {}",
                self.rank
            ));
        }
        note_gather_buffer(&self.last_buffer, self.rank, buffer)?;
        // A balanced layer gathers from the owners its own assignment wrote
        // this call; every other layer keeps the static `e % ranks` placement.
        let balanced = self
            .balanced
            .get()
            .filter(|balanced| balanced.layer(layer_idx, self.trunk_layers).is_some());
        let result = match balanced {
            Some(balanced) => {
                let owners = lock(&balanced.slot_owner);
                self.gather.exchange(
                    gpu,
                    self.rank,
                    buffer,
                    EpSlotOwners::Explicit(&owners),
                    slots,
                )
            }
            None => self.gather.exchange(
                gpu,
                self.rank,
                buffer,
                EpSlotOwners::ExpertModulo(topk_indices, QWEN4_EP_OWNER_EXPERTS),
                slots,
            ),
        };
        result.map_err(|error| {
            format!(
                "qwen4 EP row gather layer {layer_idx} rank {}: {error}",
                self.rank
            )
        })
    }

    fn assigns_owners(&self, layer_idx: u16) -> bool {
        self.balanced
            .get()
            .is_some_and(|balanced| balanced.layer(layer_idx, self.trunk_layers).is_some())
    }

    fn assign_owners(
        &self,
        gpu: &mut Gpu,
        layer_idx: u16,
        slots: usize,
        topk_indices: &GpuTensor,
    ) -> Result<(), String> {
        let tables = self
            .balanced
            .get()
            .and_then(|balanced| balanced.layer(layer_idx, self.trunk_layers))
            .ok_or_else(|| {
                format!(
                    "qwen4 EP rank {}: layer {layer_idx} is not a balanced trunk layer",
                    self.rank
                )
            })?;
        gpu.qwen4_ep_owner_assign(
            topk_indices,
            slots,
            self.gather.spec().ranks,
            self.rank,
            tables,
        )
        .map_err(|error| {
            format!(
                "qwen4 EP owner assignment layer {layer_idx} rank {}: {error}",
                self.rank
            )
        })
    }
}

#[cfg(test)]
mod gather_hook_tests {
    use super::*;

    #[test]
    fn consecutive_gathers_never_reuse_a_buffer() {
        let last = AtomicUsize::new(NO_BUFFER);
        // Trunk parity, then the head's two buffers interleaved with the trunk.
        for buffer in [0, 1, 0, 1, 2, 0, 3, 1, 2, 3, 0] {
            note_gather_buffer(&last, 1, buffer).unwrap();
        }
        let error = note_gather_buffer(&last, 1, 0).unwrap_err();
        assert!(
            error.contains("buffer 0") && error.contains("rank 1"),
            "{error}"
        );
        // The same buffer twice in a row is refused wherever it occurs.
        assert!(note_gather_buffer(&last, 1, 2).is_ok());
        assert!(note_gather_buffer(&last, 1, 2).is_err());
    }

    #[test]
    fn only_balanced_trunk_layers_assign_owners_never_the_mtp_head() {
        let tables = Qwen4EpOwnerTables {
            full_gate_up: 0x1000,
            full_down: 0x2000,
            live_gate_up: 0x3000,
            live_down: 0x4000,
            slot_owner: 0x5000,
            dummy_gate_up: 0x6000,
            dummy_down: 0x7000,
        };
        // Four trunk layers, two of them balanced; the table vector is longer
        // than the trunk to prove the head range is excluded by the guard.
        let by_layer = [None, Some(tables), None, Some(tables), Some(tables)];
        let trunk_layers = 4;
        assert_eq!(
            balanced_layer_tables(&by_layer, 1, trunk_layers),
            Some(&tables)
        );
        assert_eq!(
            balanced_layer_tables(&by_layer, 3, trunk_layers),
            Some(&tables)
        );
        assert_eq!(balanced_layer_tables(&by_layer, 0, trunk_layers), None);
        assert_eq!(balanced_layer_tables(&by_layer, 2, trunk_layers), None);
        // The MTP head's layer indices start at the trunk layer count.
        assert_eq!(balanced_layer_tables(&by_layer, 4, trunk_layers), None);
        assert_eq!(balanced_layer_tables(&by_layer, 5, trunk_layers), None);
        assert_eq!(balanced_layer_tables(&[], 0, trunk_layers), None);
    }
}

/// The grouped-row capacity (padded tiles) a rank registered for the owner-routed chunk
/// MoE, `None` when the rank did not admit it.
fn tile_capacity(bundle: &Qwen4Bundle) -> Option<usize> {
    bundle
        .execution
        .as_ref()
        .and_then(|forward| forward.ep_tile_registration())
        .map(|(_, tiles)| tiles)
}

/// A rank's per-layer symmetric verdicts (`attach_ep`), `None` when the lever was not
/// admitted on this rank.
fn tile_verdicts(bundle: &Qwen4Bundle) -> Option<Vec<bool>> {
    bundle
        .execution
        .as_ref()
        .and_then(|forward| forward.ep_tile_verdicts())
}

/// The per-layer verdict vector every rank agreed on: `Some` only when every rank admitted
/// the lever, the vectors are identical, non-empty and every layer is symmetric. Any `None`,
/// mismatch or false layer refuses the lever for the whole mesh.
fn agreed_tile_verdicts(verdicts: &[Option<Vec<bool>>]) -> Option<&Vec<bool>> {
    let first = verdicts.first()?.as_ref()?;
    (!first.is_empty()
        && first.iter().all(|&symmetric| symmetric)
        && verdicts.iter().all(|layers| layers.as_ref() == Some(first)))
    .then_some(first)
}

/// The owner-routed chunk exchange one rank installs into its forward
/// (`Qwen4GpuForward::install_tile_exchange`): the shared transport, this rank and a
/// non-owning view of the rank's owner table (`e % ranks`).
pub(crate) struct Qwen4EpTileExchangeHook {
    rank: usize,
    tile: Arc<EpTileExchange>,
    owner_table: GpuTensor,
    experts: usize,
}

impl Qwen4EpTileExchangeHook {
    pub(crate) fn new(
        rank: usize,
        tile: Arc<EpTileExchange>,
        owner_table: GpuTensor,
        experts: usize,
    ) -> Self {
        Self {
            rank,
            tile,
            owner_table,
            experts,
        }
    }
}

// SAFETY: `owner_table` is a `Borrowed` alias of a device buffer the forward owns until
// `free_gpu` (read-only after upload: the hook only hands its device address to launches on
// the calling rank's own stream) and the hook never outlives that forward; HIP device
// pointers carry no thread affinity.
unsafe impl Send for Qwen4EpTileExchangeHook {}
unsafe impl Sync for Qwen4EpTileExchangeHook {}

impl MoeTileExchange for Qwen4EpTileExchangeHook {
    fn owners(&self) -> (usize, &GpuTensor, usize) {
        (self.rank, &self.owner_table, self.experts)
    }

    fn before_down(&self, gpu: &mut Gpu) -> Result<(), String> {
        self.tile
            .before_down(gpu, self.rank)
            .map_err(|error| error.to_string())
    }

    fn host_barrier(&self) -> Result<(), String> {
        self.tile.host_barrier().map_err(|error| error.to_string())
    }

    fn exchange(
        &self,
        gpu: &mut Gpu,
        _layer_idx: u16,
        tile_ids: &GpuTensor,
        y_down_grouped: &GpuTensor,
        tiles: usize,
    ) -> Result<(), String> {
        self.tile
            .exchange(
                gpu,
                self.rank,
                &self.owner_table,
                tile_ids,
                y_down_grouped,
                tiles,
            )
            .map_err(|error| error.to_string())
    }
}

/// The attached mesh, owned by rank 0's bundle (`Qwen4Bundle::ep`).
pub struct Qwen4EpMesh {
    pool: PeerPool<Qwen4EpPeer>,
    gather: Arc<EpRowGather>,
    /// Owner-routed chunk MoE transport (`HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`); `None`
    /// when the lever is off or was not admitted on every rank.
    tile: Option<Arc<EpTileExchange>>,
    ranks: usize,
    device_ids: Vec<i32>,
    used_bytes: Vec<u64>,
}

type PeerJob<'a, R> =
    &'a (dyn Fn(&mut Qwen4Bundle, &mut Gpu, &mut PeerMirrors) -> Result<R, String> + Sync);

impl Qwen4EpMesh {
    /// Ranks including rank 0.
    pub fn ranks(&self) -> usize {
        self.ranks
    }

    /// HIP device id of every rank (rank 0 first).
    pub fn device_ids(&self) -> &[i32] {
        &self.device_ids
    }

    /// Device memory in use on every rank's GPU when the mesh was attached
    /// (rank 0 first): resident weights, state, scratch, expert tables.
    pub fn used_bytes(&self) -> &[u64] {
        &self.used_bytes
    }

    /// Whether an earlier failure poisoned the mesh.
    pub fn poisoned(&self) -> Option<String> {
        self.pool.poisoned()
    }

    /// Run `job` on every peer while `local` runs rank 0 on the caller's
    /// thread. Results in rank order. Any rank's error or panic aborts the
    /// row-gather barrier, poisons the mesh, and is returned (the first one).
    pub(crate) fn run<R: Send>(
        &self,
        local: impl FnOnce() -> Result<R, String>,
        job: PeerJob<'_, R>,
    ) -> Result<Vec<R>, String> {
        self.pool.run_all(local, &|_, peer: &mut Qwen4EpPeer| {
            job(&mut peer.bundle, &mut peer.gpu, &mut peer.mirrors)
        })
    }

    /// Run `job` on peer `rank` alone (rank 0 is not a peer).
    pub(crate) fn run_peer<R: Send>(&self, rank: usize, job: PeerJob<'_, R>) -> Result<R, String> {
        self.pool.run_rank(rank, &|_, peer: &mut Qwen4EpPeer| {
            job(&mut peer.bundle, &mut peer.gpu, &mut peer.mirrors)
        })
    }

    /// Drain every rank, stop the workers (each frees its own bundle on its
    /// thread) and release rank 0's gather events. Always attempts every step.
    pub(crate) fn close(self, gpu: &mut Gpu) -> Result<(), String> {
        let Self {
            pool, gather, tile, ..
        } = self;
        let mut first: Option<String> = None;
        let drained = pool.run_all_forced(|| sync_device(gpu), &|_, peer: &mut Qwen4EpPeer| {
            sync_device(&mut peer.gpu)
        });
        if let Err(error) = drained {
            first.get_or_insert(error);
        }
        if let Err(error) = pool.close() {
            first.get_or_insert(error);
        }
        if let Err(error) = gather.release(0, gpu) {
            first.get_or_insert(format!("rank 0: release gather events: {error}"));
        }
        if let Some(tile) = tile {
            if let Err(error) = tile.release(0, gpu) {
                first.get_or_insert(format!("rank 0: release tile exchange events: {error}"));
            }
        }
        first.map_or(Ok(()), Err)
    }
}

fn ep_err(message: impl Into<String>) -> BundleError {
    BundleError::Forward(format!("qwen4 EP: {}", message.into()))
}

impl Qwen4Bundle {
    /// Number of ranks serving this bundle: 1 without EP.
    pub fn ranks(&self) -> usize {
        self.ep.as_ref().map_or(1, |mesh| mesh.ranks())
    }

    /// The attached EP mesh, if any.
    pub fn ep_mesh(&self) -> Option<&Qwen4EpMesh> {
        self.ep.as_deref()
    }

    /// Run `f` on rank `rank`'s bundle and `Gpu`: rank 0 is `self` with the
    /// caller's `gpu` (the mesh is detached for the call, so calls on the
    /// bundle inside `f` do not fan out); a peer runs on its own worker thread
    /// and the call blocks until `f` returns. `f` also gets that rank's
    /// [`PeerMirrors`] (always empty on rank 0).
    pub fn with_rank_mut<R: Send>(
        &mut self,
        gpu: &mut Gpu,
        rank: usize,
        f: impl FnOnce(&mut Qwen4Bundle, &mut Gpu, &PeerMirrors) -> Result<R, String> + Send,
    ) -> Result<R, String> {
        let ranks = self.ranks();
        if rank >= ranks {
            return Err(format!(
                "qwen4 EP rank {rank} out of range (ranks = {ranks})"
            ));
        }
        if rank == 0 {
            let mesh = self.ep.take();
            let result = f(self, gpu, &PeerMirrors::default());
            self.ep = mesh;
            return result;
        }
        let mesh = self
            .ep
            .as_ref()
            .ok_or_else(|| "qwen4 EP mesh is not attached".to_string())?;
        let cell = Mutex::new(Some(f));
        mesh.run_peer(rank, &|bundle, peer_gpu, mirrors| {
            let f = lock(&cell)
                .take()
                .ok_or_else(|| "qwen4 EP: rank job ran twice".to_string())?;
            f(bundle, peer_gpu, mirrors)
        })
    }

    /// Replicate the loaded single-rank bundle into an EP mesh.
    ///
    /// `self` is rank 0 on `gpu`: loaded with its forward attached and bound
    /// to its load's seal ([`Qwen4Bundle::bind_ep_seal`]), from which the
    /// rank count and every rank's physical device come. `loader` builds one
    /// peer on its worker thread (hipfire-loader supplies it). Sequence:
    ///
    /// 1. every peer loads on its own thread; it must be bound to rank 0's
    ///    seal at its own rank on the seal's device, and its config and EP
    ///    placement are checked against rank 0;
    /// 2. each rank registers its gather buffers with one
    ///    [`EpRowGatherBuilder`](hipfire_runtime::ep_row_gather::EpRowGatherBuilder)
    ///    (on its own thread), then the transport is built;
    /// 3. every rank enables P2P (after every load/allocation exists);
    /// 4. every rank reports its compact expert sources;
    /// 5. every rank builds its full (P2P) tables from all sources and installs
    ///    the row-gather hook (`Qwen4GpuForward::attach_ep`);
    /// 5b. with `HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`, every rank's tile-exchange
    ///    registration and symmetric verdicts are collected (`run_all`, every rank, any
    ///    admitted rank count); the exchange is installed only if all ranks admitted and agree;
    /// 6. per-rank memory is recorded and the mesh is stored in `self`.
    ///
    /// On `Err` the bundle must be freed, not served (rank 0's forward may
    /// already hold the hook); peers are already torn down.
    pub fn attach_ep_mesh(
        &mut self,
        gpu: &mut Gpu,
        loader: Arc<PeerLoadFn>,
    ) -> Result<(), BundleError> {
        if self.ep.is_some() {
            return Err(ep_err("the mesh is already attached"));
        }
        let seal = match &self.ep_seal {
            Some(bound) if bound.rank == 0 => bound.seal.clone(),
            Some(bound) => {
                return Err(ep_err(format!(
                    "this bundle is rank {}, not rank 0",
                    bound.rank
                )))
            }
            None => return Err(ep_err("rank 0 is not bound to an EP seal")),
        };
        let ranks = seal.ranks();
        if !(2..=8).contains(&ranks) {
            return Err(ep_err(format!("{ranks} ranks requested; supported: 2..=8")));
        }
        // The sealed planner admitted the physical devices (distinct unless
        // `HIPFIRE_EMULATE_GPUS` aliases logical ranks onto one device).
        let device_ids = seal.physical_devices().to_vec();
        if device_ids[0] != gpu.device_id {
            return Err(ep_err(format!(
                "rank 0 runs on device {}, the seal places it on {}",
                gpu.device_id, device_ids[0]
            )));
        }
        let own = seal
            .placement(0)
            .map_err(|error| ep_err(error.to_string()))?;
        let forward = self
            .execution
            .as_ref()
            .ok_or_else(|| ep_err("rank 0 has no attached forward"))?;
        if forward.ep_placement() != Some(own) {
            return Err(ep_err(format!(
                "rank 0 was loaded with placement {:?}, expected {own:?}",
                forward.ep_placement()
            )));
        }
        let trunk_layers = self.config.num_hidden_layers;
        let spec = EpRowGatherSpec {
            ranks,
            slots: QWEN4_EP_GATHER_MAX_ROWS * self.config.num_experts_per_tok,
            row_bytes: self.config.hidden_size * std::mem::size_of::<f32>(),
        };
        let fingerprint = self.ep_replication_fingerprint();

        let inits: Vec<Box<dyn FnOnce(usize) -> Result<Qwen4EpPeer, String> + Send>> = (1..ranks)
            .map(|_| {
                let loader = loader.clone();
                let config = self.config.clone();
                let fingerprint = fingerprint.clone();
                let seal = seal.clone();
                Box::new(move |rank: usize| {
                    let (gpu, mut bundle) = loader(rank)?;
                    let placement = seal.placement(rank).map_err(|error| error.to_string())?;
                    align_peer_session(&mut bundle, placement, &fingerprint);
                    let peer = Qwen4EpPeer {
                        rank,
                        gpu,
                        bundle,
                        mirrors: PeerMirrors::default(),
                        gather: None,
                        tile: None,
                    };
                    let checked = peer
                        .check_seal(&seal)
                        .and_then(|()| peer.check_replica(&config, placement, &fingerprint));
                    match checked {
                        Ok(()) => Ok(peer),
                        Err(error) => {
                            let _ = peer.teardown();
                            Err(error)
                        }
                    }
                }) as Box<dyn FnOnce(usize) -> Result<Qwen4EpPeer, String> + Send>
            })
            .collect();
        let pool = PeerPool::spawn(inits, Qwen4EpPeer::teardown).map_err(ep_err)?;

        match self.ep_attach_steps(gpu, pool, spec, trunk_layers, device_ids) {
            Ok(mesh) => {
                self.ep = Some(Box::new(mesh));
                Ok(())
            }
            Err(error) => Err(ep_err(error)),
        }
    }

    /// Steps 2-6 of [`Self::attach_ep_mesh`]. Consumes the pool: on error the
    /// peers are torn down by its drop and rank 0's gather events released.
    fn ep_attach_steps(
        &mut self,
        gpu: &mut Gpu,
        pool: PeerPool<Qwen4EpPeer>,
        spec: EpRowGatherSpec,
        trunk_layers: usize,
        device_ids: Vec<i32>,
    ) -> Result<Qwen4EpMesh, String> {
        let ranks = spec.ranks;
        // 2. Register every rank's gather buffers, build the transport.
        let builder = Mutex::new(Some(EpRowGather::builder(spec)));
        let register = |rank: usize, bundle: &Qwen4Bundle, gpu: &mut Gpu| -> Result<(), String> {
            let buffers = bundle
                .execution
                .as_ref()
                .and_then(|forward| forward.ep_gather_buffers())
                .ok_or("no EP gather buffers (the forward has no EP placement)")?;
            let refs: [&GpuTensor; QWEN4_EP_GATHER_BUFFERS] = buffers;
            lock(&builder)
                .as_mut()
                .ok_or("gather builder already consumed")?
                .register(rank, gpu, &refs)
                .map_err(|error| format!("register gather buffers: {error}"))
        };
        {
            let me = &*self;
            let gpu0 = &mut *gpu;
            pool.run_all(|| register(0, me, gpu0), &|rank, peer: &mut Qwen4EpPeer| {
                register(rank, &peer.bundle, &mut peer.gpu)
            })?;
        }
        let gather = lock(&builder)
            .take()
            .ok_or("gather builder already consumed")?
            .build()
            .map_err(|error| format!("build gather transport: {error}"))?;
        let abort = gather.clone();
        pool.set_abort(move || abort.abort());
        let tile_slot: Mutex<Option<Arc<EpTileExchange>>> = Mutex::new(None);
        let release_rank0 = |gpu: &mut Gpu, error: String| -> String {
            gather.abort();
            let _ = gather.release(0, gpu);
            if let Some(tile) = lock(&tile_slot).as_ref() {
                tile.abort();
                let _ = tile.release(0, gpu);
            }
            error
        };

        // 2b. Owner-routed chunk MoE (`HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`): every rank that
        // admitted it at construction registers its grouped-row buffer (allocated before
        // P2P is enabled below) with one transport.
        let capacities = {
            let me = &*self;
            pool.run_all(|| Ok(tile_capacity(me)), &|_, peer: &mut Qwen4EpPeer| {
                Ok(tile_capacity(&peer.bundle))
            })
        };
        let capacities = match capacities {
            Ok(capacities) => capacities,
            Err(error) => return Err(release_rank0(gpu, error)),
        };
        let admitted = capacities.iter().filter(|c| c.is_some()).count();
        if admitted != 0 && admitted != ranks {
            eprintln!(
                "qwen4 EP: owner-routed prefill admitted on {admitted}/{ranks} ranks only; chunks keep the full-table route"
            );
        }
        if admitted == ranks {
            let tile_spec = EpTileExchangeSpec {
                ranks,
                experts: self.config.num_experts,
                row_u4: self.config.hidden_size * std::mem::size_of::<u16>() / 16,
                max_tiles: capacities.iter().flatten().copied().min().unwrap_or(0),
            };
            let tile_builder = Mutex::new(Some(EpTileExchange::builder(tile_spec)));
            let register_tile =
                |rank: usize, bundle: &Qwen4Bundle, gpu: &mut Gpu| -> Result<(), String> {
                    let (buffer, _) = bundle
                        .execution
                        .as_ref()
                        .and_then(|forward| forward.ep_tile_registration())
                        .ok_or("no EP tile registration")?;
                    lock(&tile_builder)
                        .as_mut()
                        .ok_or("tile exchange builder already consumed")?
                        .register(rank, gpu, buffer)
                        .map_err(|error| format!("register tile exchange buffers: {error}"))
                };
            let registered = {
                let me = &*self;
                let gpu0 = &mut *gpu;
                pool.run_all(
                    || register_tile(0, me, gpu0),
                    &|rank, peer: &mut Qwen4EpPeer| {
                        register_tile(rank, &peer.bundle, &mut peer.gpu)
                    },
                )
            };
            if let Err(error) = registered {
                return Err(release_rank0(gpu, error));
            }
            let built = lock(&tile_builder)
                .take()
                .ok_or_else(|| "tile exchange builder already consumed".to_string())
                .and_then(|builder| {
                    builder
                        .build()
                        .map_err(|error| format!("build tile exchange transport: {error}"))
                });
            match built {
                Ok(tile) => {
                    let abort = tile.clone();
                    let gather_abort = gather.clone();
                    pool.set_abort(move || {
                        gather_abort.abort();
                        abort.abort();
                    });
                    *lock(&tile_slot) = Some(tile);
                }
                Err(error) => return Err(release_rank0(gpu, error.to_string())),
            }
        }

        // 3. P2P after every load/allocation exists.
        let tile = lock(&tile_slot).clone();
        let enable = |rank: usize, gpu: &mut Gpu| -> Result<(), String> {
            gather
                .enable_peers(rank, gpu)
                .map_err(|error| format!("enable peer access: {error}"))?;
            if let Some(tile) = tile.as_ref() {
                tile.enable_peers(rank, gpu)
                    .map_err(|error| format!("enable tile exchange peer access: {error}"))?;
            }
            Ok(())
        };
        let enabled = {
            let gpu0 = &mut *gpu;
            pool.run_all(|| enable(0, gpu0), &|rank, peer: &mut Qwen4EpPeer| {
                peer.gather = Some(gather.clone());
                peer.tile = tile.clone();
                enable(rank, &mut peer.gpu)
            })
        };
        if let Err(error) = enabled {
            return Err(release_rank0(gpu, error));
        }

        // 4. Compact expert sources of every rank.
        let sources = {
            let me = &*self;
            pool.run_all(|| rank_sources(me), &|_, peer: &mut Qwen4EpPeer| {
                rank_sources(&peer.bundle)
            })
        };
        let sources: Vec<Vec<Qwen4PeerExpertSource>> = match sources {
            Ok(sources) => sources,
            Err(error) => return Err(release_rank0(gpu, error)),
        };

        // 5. Full tables + hooks.
        let attach = |rank: usize, bundle: &mut Qwen4Bundle, gpu: &mut Gpu| -> Result<(), String> {
            let buffers = bundle
                .execution
                .as_ref()
                .and_then(|forward| forward.ep_gather_buffers())
                .ok_or("no EP gather buffers (the forward has no EP placement)")?;
            let hook = Arc::new(Qwen4EpRowGatherHook::new(
                rank,
                trunk_layers,
                gather.clone(),
                buffers.map(|buffer| buffer.buf.as_ptr() as usize),
            ));
            let mut forward = bundle
                .execution
                .take()
                .ok_or("the forward is not attached")?;
            let installed: Arc<dyn MoeRowGather + Send + Sync> = hook.clone();
            let result = forward
                .attach_ep(gpu, bundle, &sources, installed)
                .map_err(|error| format!("attach EP tables: {error}"))
                .and_then(|()| match forward.balanced_plan() {
                    // Default on (`HIPFIRE_EP_BALANCED_OWNERS=0` disables) with a
                    // host-mapped layer on gfx1201 at 2..=4 ranks; other arches and
                    // more ranks run the exact static split (`attach_ep` builds no
                    // balanced layers).
                    Some(plan) => hook.install_balanced(plan),
                    None => Ok(()),
                });
            bundle.execution = Some(forward);
            result
        };
        let attached = {
            let me = &mut *self;
            let gpu0 = &mut *gpu;
            pool.run_all(|| attach(0, me, gpu0), &|rank, peer: &mut Qwen4EpPeer| {
                attach(rank, &mut peer.bundle, &mut peer.gpu)
            })
        };
        if let Err(error) = attached {
            return Err(release_rank0(gpu, error));
        }

        // 5b. Owner-routed chunk MoE (EP2..EP8): every rank's per-layer verdicts (symmetric IU4
        // headers on gfx1201, QT44/QT53 formats on gfx1100's F16 WMMA route; taken on the full
        // tables in `attach_ep`) must be identical and all-true on ALL ranks; only
        // then does every rank propagate them to its compact layers and install the exchange.
        // Any mismatch refuses the lever for this load: never a per-rank route divergence.
        if let Some(tile) = tile.as_ref() {
            let verdicts = {
                let me = &*self;
                pool.run_all(|| Ok(tile_verdicts(me)), &|_, peer: &mut Qwen4EpPeer| {
                    Ok(tile_verdicts(&peer.bundle))
                })
            };
            let verdicts = match verdicts {
                Ok(verdicts) => verdicts,
                Err(error) => return Err(release_rank0(gpu, error)),
            };
            let agreed = agreed_tile_verdicts(&verdicts);
            let layer_count = agreed.map_or(0, Vec::len);
            if agreed.is_some() {
                let experts = self.config.num_experts;
                let install = |rank: usize, bundle: &mut Qwen4Bundle| -> Result<(), String> {
                    let mut forward = bundle
                        .execution
                        .take()
                        .ok_or("the forward is not attached")?;
                    let result = (|| {
                        let table = forward
                            .ep_tile_owner_table_view()
                            .ok_or("the forward has no owner table")?;
                        let hook: Arc<dyn MoeTileExchange + Send + Sync> = Arc::new(
                            Qwen4EpTileExchangeHook::new(rank, tile.clone(), table, experts),
                        );
                        forward
                            .install_tile_exchange(hook)
                            .map_err(|error| format!("install tile exchange: {error}"))
                    })();
                    bundle.execution = Some(forward);
                    result
                };
                let installed = {
                    let me = &mut *self;
                    pool.run_all(|| install(0, me), &|rank, peer: &mut Qwen4EpPeer| {
                        install(rank, &mut peer.bundle)
                    })
                };
                if let Err(error) = installed {
                    return Err(release_rank0(gpu, error));
                }
                eprintln!(
                    "qwen4 EP: owner-routed prefill on ({} MoE layers admitted on every rank, \
                     owners e % {ranks})",
                    layer_count
                );
            } else {
                let counts: Vec<String> = verdicts
                    .iter()
                    .map(|layers| {
                        layers.as_ref().map_or("n/a".to_string(), |layers| {
                            format!("{}/{}", layers.iter().filter(|&&s| s).count(), layers.len())
                        })
                    })
                    .collect();
                eprintln!(
                    "qwen4 EP: owner-routed prefill refused (per-rank admitted layers {counts:?} \
                     are not identical and complete); chunks keep the full-table route"
                );
            }
        }

        // 6. Per-rank device memory in use.
        let used = |gpu: &Gpu| -> Result<u64, String> {
            let (free, total) = gpu
                .device_mem_info()
                .map_err(|error| format!("device memory query: {error}"))?;
            Ok(total.saturating_sub(free) as u64)
        };
        let used_bytes = {
            let gpu0 = &*gpu;
            pool.run_all(|| used(gpu0), &|_, peer: &mut Qwen4EpPeer| used(&peer.gpu))
        };
        let used_bytes = match used_bytes {
            Ok(used_bytes) => used_bytes,
            Err(error) => return Err(release_rank0(gpu, error)),
        };
        debug_assert_eq!(used_bytes.len(), ranks);
        Ok(Qwen4EpMesh {
            pool,
            gather,
            tile,
            ranks,
            device_ids,
            used_bytes,
        })
    }

    /// Fan a call out to every rank (see the module docs). Without EP this is
    /// exactly `local`. The mesh is detached from `self` during rank 0's own
    /// call so nested bundle calls run locally, and restored afterwards.
    /// Host results must agree on every rank; rank 0's is returned.
    pub(crate) fn ep_fanout<R: Send + PartialEq + std::fmt::Debug>(
        &mut self,
        what: &str,
        local: impl FnOnce(&mut Self) -> Result<R, BundleError>,
        job: PeerJob<'_, R>,
    ) -> Result<R, BundleError> {
        let Some(mesh) = self.ep.take() else {
            return local(self);
        };
        let result = mesh.run(|| local(&mut *self).map_err(|error| error.to_string()), job);
        self.ep = Some(mesh);
        result
            .and_then(|values| agree(what, values))
            .map_err(BundleError::Forward)
    }

    /// Fan a native-MTP drafter call out to every rank. `lead` runs rank 0's
    /// `leader` drafter with this bundle (mesh detached, so nested calls run
    /// locally) on the caller's thread and `gpu`; `follow` runs on every peer
    /// with that peer's drafter replica (created from rank 0's construction
    /// parameters on the first call) and its own bundle. Before `follow`, the
    /// replica replays the target-less calls rank 0 ran since the last fan-out
    /// (`configure_request`, `mtp_reset`, in order) and mirrors rank 0's n-gram
    /// arming. The host results must agree on every rank; rank 0's is returned.
    /// Without a mesh this is exactly `lead`. Neither closure may poll an
    /// abort: no rank may stop alone.
    pub(crate) fn ep_drafter_fanout<R: Send + PartialEq + std::fmt::Debug>(
        &mut self,
        what: &str,
        gpu: &mut Gpu,
        leader: &mut crate::mtp_spec::Qwen4MtpDrafter,
        lead: impl FnOnce(
            &mut crate::mtp_spec::Qwen4MtpDrafter,
            &mut Qwen4Bundle,
            &mut Gpu,
        ) -> Result<R, String>,
        follow: &(dyn Fn(
            &mut crate::mtp_spec::Qwen4MtpDrafter,
            &mut Qwen4Bundle,
            &mut Gpu,
        ) -> Result<R, String>
              + Sync),
    ) -> Result<R, String> {
        let Some(mesh) = self.ep.take() else {
            return lead(leader, self, gpu);
        };
        let params = leader.replica_params();
        let ngram_active = leader.ngram_active();
        let owed = leader.pending_replay().to_vec();
        let result = mesh.run(
            || lead(&mut *leader, &mut *self, &mut *gpu),
            &|bundle, peer_gpu, mirrors| {
                let replica = mirrors.drafter(params);
                replica.replay(peer_gpu, &owed)?;
                replica.mirror_ngram_active(ngram_active);
                follow(replica, bundle, peer_gpu)
            },
        );
        self.ep = Some(mesh);
        let agreed = result.and_then(|values| agree(what, values));
        if agreed.is_ok() {
            leader.clear_replay();
        }
        agreed
    }

    /// Refuse an entry point that has no EP replication (lanes, MTP/spec, state
    /// snapshots) instead of running it on rank 0 alone and stalling the gather
    /// barrier.
    pub(crate) fn ep_refuse(&self, what: &str) -> Result<(), BundleError> {
        if self.pp.is_some() {
            return Err(BundleError::Forward(format!(
                "{what} is not supported under qwen4 topology.pp yet"
            )));
        }
        if self.ep.is_some() {
            return Err(BundleError::Forward(format!(
                "{what} is unsupported under qwen4 EP"
            )));
        }
        Ok(())
    }

    /// Stop and free the mesh (before the bundle itself is freed).
    pub(crate) fn ep_close(&mut self, gpu: &mut Gpu) -> Result<(), BundleError> {
        match self.ep.take() {
            Some(mesh) => mesh.close(gpu).map_err(ep_err),
            None => Ok(()),
        }
    }
}

/// The replicated forms of the daemon-facing bundle calls. Each runs the
/// identical call on every rank (rank 0 on the caller's thread with the caller's
/// tensors, peers on their workers with their [`PeerMirrors`]) and requires the
/// host-visible results to agree. They are reached only when the mesh is
/// attached; inside the fan-out every rank's mesh is detached, so the nested
/// public calls run locally.
impl Qwen4Bundle {
    pub(crate) fn ep_forward_token(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
    ) -> Result<(), BundleError> {
        let logits_spec = TensorSpec::of(logits);
        let top1_spec = top1.map(TensorSpec::of);
        self.ep_fanout(
            "forward_token",
            |bundle| bundle.forward_token(gpu, token, logits, top1),
            &|bundle, peer_gpu, mirrors| {
                let (logits, top1) = mirrors.pair(peer_gpu, &logits_spec, top1_spec.as_ref())?;
                bundle
                    .forward_token(peer_gpu, token, logits, top1)
                    .map_err(|error| error.to_string())
            },
        )
    }

    pub(crate) fn ep_forward_token_or_argmax(
        &mut self,
        gpu: &mut Gpu,
        token: Option<u32>,
        logits: &GpuTensor,
    ) -> Result<u32, BundleError> {
        let logits_spec = TensorSpec::of(logits);
        self.ep_fanout(
            "forward_token_or_argmax token",
            |bundle| bundle.forward_token_or_argmax(gpu, token, logits),
            &|bundle, peer_gpu, mirrors| {
                let (logits, _) = mirrors.pair(peer_gpu, &logits_spec, None)?;
                bundle
                    .forward_token_or_argmax(peer_gpu, token, logits)
                    .map_err(|error| error.to_string())
            },
        )
    }

    /// `forward_chunk` (`final_row_only == false`) / `forward_chunk_final`.
    pub(crate) fn ep_forward_chunk(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
        final_row_only: bool,
    ) -> Result<(), BundleError> {
        let logits_spec = TensorSpec::of(logits);
        let top1_spec = top1.map(TensorSpec::of);
        let run = |bundle: &mut Qwen4Bundle,
                   gpu: &mut Gpu,
                   logits: &GpuTensor,
                   top1: Option<&GpuTensor>| {
            if final_row_only {
                bundle.forward_chunk_final(gpu, tokens, logits, top1)
            } else {
                bundle.forward_chunk(gpu, tokens, logits, top1)
            }
        };
        self.ep_fanout(
            "forward_chunk",
            |bundle| run(bundle, gpu, logits, top1),
            &|bundle, peer_gpu, mirrors| {
                let (logits, top1) = mirrors.pair(peer_gpu, &logits_spec, top1_spec.as_ref())?;
                run(bundle, peer_gpu, logits, top1).map_err(|error| error.to_string())
            },
        )
    }

    pub(crate) fn ep_prefill_final(
        &mut self,
        gpu: &mut Gpu,
        prompt: &[u32],
        reused: usize,
        logits: &GpuTensor,
    ) -> Result<(), BundleError> {
        let logits_spec = TensorSpec::of(logits);
        self.ep_fanout(
            "prefill_final",
            |bundle| bundle.prefill_final(gpu, prompt, reused, logits),
            &|bundle, peer_gpu, mirrors| {
                let (logits, _) = mirrors.pair(peer_gpu, &logits_spec, None)?;
                bundle
                    .prefill_final(peer_gpu, prompt, reused, logits)
                    .map_err(|error| error.to_string())
            },
        )
    }

    pub(crate) fn ep_apply_penalty_table(
        &mut self,
        gpu: &mut Gpu,
        logits: &GpuTensor,
        table: &PenaltyTable,
    ) -> Result<(), BundleError> {
        let logits_spec = TensorSpec::of(logits);
        self.ep_fanout(
            "apply_penalty_table",
            |bundle| bundle.apply_penalty_table(gpu, logits, table),
            &|bundle, peer_gpu, mirrors| {
                let (logits, _) = mirrors.pair(peer_gpu, &logits_spec, None)?;
                bundle
                    .apply_penalty_table(peer_gpu, logits, table)
                    .map_err(|error| error.to_string())
            },
        )
    }

    pub(crate) fn ep_reset(&mut self, gpu: &mut Gpu) -> Result<(), BundleError> {
        self.ep_fanout(
            "reset",
            |bundle| bundle.reset(gpu),
            &|bundle, peer_gpu, _| bundle.reset(peer_gpu).map_err(|error| error.to_string()),
        )
    }

    pub(crate) fn ep_session_clear(&mut self, gpu: &mut Gpu) {
        let cleared = self.ep_fanout(
            "session_clear",
            |bundle| {
                bundle.session_clear(gpu);
                Ok(())
            },
            &|bundle, peer_gpu, _| {
                bundle.session_clear(peer_gpu);
                Ok(())
            },
        );
        if let Err(error) = cleared {
            eprintln!("  qwen4 EP: session_clear: {error}");
        }
    }

    /// `ArchModel::session_commit` (`consumed == None`) /
    /// `session_commit_live(consumed)` on every rank (host state only).
    pub(crate) fn ep_session_commit(&mut self, consumed: Option<&[u32]>) {
        let commit = |bundle: &mut Qwen4Bundle| {
            match consumed {
                Some(consumed) => ArchModel::session_commit_live(bundle, consumed),
                None => ArchModel::session_commit(bundle),
            }
            Ok(())
        };
        if let Err(error) = self.ep_fanout("session_commit", commit, &|bundle, _, _| {
            match consumed {
                Some(consumed) => ArchModel::session_commit_live(bundle, consumed),
                None => ArchModel::session_commit(bundle),
            }
            Ok(())
        }) {
            eprintln!("  qwen4 EP: session_commit: {error}");
        }
    }

    /// The reuse point every rank can serve: each rank plans over the prompt
    /// prefix `cap` allows, and `cap` drops to the lowest answer until all
    /// ranks agree (each rank's cache evicts independently under memory
    /// pressure, so they may hold different snapshots). A mesh failure plans 0;
    /// the following replicated call reports the poisoning.
    pub(crate) fn ep_session_plan(&self, prompt: &[u32], route: SessionRoute) -> usize {
        let Some(mesh) = self.ep.as_ref() else {
            return self.session_plan_capped(prompt, route, prompt.len());
        };
        let mut cap = prompt.len();
        loop {
            let plans = mesh.run(
                || Ok(self.session_plan_capped(prompt, route, cap)),
                &|bundle, _, _| Ok(bundle.session_plan_capped(prompt, route, cap)),
            );
            let Ok(plans) = plans else {
                return 0;
            };
            let lowest = plans.iter().copied().min().unwrap_or(0);
            if plans.iter().all(|&plan| plan == lowest) {
                return lowest;
            }
            cap = lowest;
        }
    }
}

/// A peer loads without rank 0's tokenizer, so it has no session cache and no
/// `<|im_end|>` id. Every rank plans prefill boundaries and snapshots from the
/// same host configuration, so give the peer a cache of rank 0's budget and
/// rank 0's turn settings. Its cache domain only keys that peer's own cache.
fn align_peer_session(
    bundle: &mut Qwen4Bundle,
    placement: Qwen4EpPlacement,
    rank0: &ReplicaFingerprint,
) {
    if rank0.session_cache && bundle.session_cache().is_none() {
        let tag = format!("rank{}/{}", placement.rank, placement.ranks);
        let digest = sha256_len_prefixed(&[b"qwen4-ep-peer-session-cache", tag.as_bytes()]);
        let domain = CacheDomain {
            model_content_digest: digest.clone(),
            model_load_epoch: 1,
            sidecar_digests: Vec::new(),
            tokenizer: TokenizerIdentity {
                vocab_digest: digest.clone(),
                config_digest: digest.clone(),
            },
            template: TemplateIdentity {
                template_digest: digest,
                normalization_tag: "ep-peer".to_string(),
            },
            arch_policy: ArchPolicy {
                arch_tag: "qwen4".to_string(),
                state_abi_tag: String::new(),
                position_attention_tag: String::new(),
            },
            kv_layout: KvLayout {
                k_stride_bytes: Vec::new(),
                v_stride_bytes: Vec::new(),
                layout_tag: String::new(),
            },
            device: DeviceTopology {
                device_id: format!("ep-{tag}"),
                topology_id: "ep".to_string(),
                allocation_epoch: 1,
            },
            namespace: SharingNamespace {
                domain_id: "default".to_string(),
            },
        };
        bundle.attach_session_cache(SessionCache::new(
            domain,
            hipfire_config::memory::session_cache_bytes(),
        ));
    }
    bundle.set_turn_end_token(rank0.turn_end_token);
    bundle.set_turn_snapshots(rank0.turn_snapshots);
}

fn rank_sources(bundle: &Qwen4Bundle) -> Result<Vec<Qwen4PeerExpertSource>, String> {
    bundle
        .execution
        .as_ref()
        .ok_or("the forward is not attached")?
        .ep_expert_sources(bundle)
        .map_err(|error| format!("expert sources: {error}"))
}

/// Host configuration that must match on every rank for the replicated
/// session/prefill planning to stay in lockstep.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ReplicaFingerprint {
    pub(crate) session_cache: bool,
    pub(crate) chunk_rows: Option<usize>,
    pub(crate) state_format: crate::state::Qwen4StateFormat,
    pub(crate) turn_end_token: Option<u32>,
    pub(crate) turn_snapshots: bool,
    pub(crate) tail_snapshot: bool,
    pub(crate) tail_snapshot_min_gap: usize,
    pub(crate) tail_snapshot_min_prompt: usize,
    pub(crate) has_mtp: bool,
    pub(crate) has_lanes: bool,
}

impl Qwen4EpPeer {
    /// A loaded peer must be bound to rank 0's seal itself (not an equal
    /// one planned from another mesh epoch) at its own rank, on the seal's
    /// physical device for that rank.
    fn check_seal(&self, seal: &Arc<crate::ep_seal::Qwen4EpSeal>) -> Result<(), String> {
        let bound = self
            .bundle
            .ep_seal()
            .ok_or("the loaded peer is not bound to an EP seal")?;
        if !Arc::ptr_eq(&bound.seal, seal) || bound.rank != self.rank {
            return Err(format!(
                "the loaded peer is bound to rank {} of a seal of epoch {}, expected rank {} of epoch {}",
                bound.rank,
                bound.seal.mesh().epoch().as_u64(),
                self.rank,
                seal.mesh().epoch().as_u64()
            ));
        }
        let device = seal.physical_devices()[self.rank];
        if self.gpu.device_id != device {
            return Err(format!(
                "the loaded peer runs on device {}, the seal places rank {} on {device}",
                self.gpu.device_id, self.rank
            ));
        }
        Ok(())
    }

    /// A loaded peer must be the same model as rank 0, at its own placement,
    /// with the same host-side request configuration.
    fn check_replica(
        &self,
        config: &crate::config::Qwen4Config,
        placement: Qwen4EpPlacement,
        fingerprint: &ReplicaFingerprint,
    ) -> Result<(), String> {
        let forward = self
            .bundle
            .execution
            .as_ref()
            .ok_or("the loaded peer has no attached forward")?;
        if forward.ep_placement() != Some(placement) {
            return Err(format!(
                "the loaded peer has placement {:?}, expected {placement:?}",
                forward.ep_placement()
            ));
        }
        if self.bundle.config != *config {
            return Err("the loaded peer's model configuration differs from rank 0".to_string());
        }
        let theirs = self.bundle.ep_replication_fingerprint();
        if theirs != *fingerprint {
            return Err(format!(
                "the loaded peer's request configuration differs from rank 0: {theirs:?} vs {fingerprint:?}"
            ));
        }
        Ok(())
    }
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

    fn pool(n: usize) -> PeerPool<Fake> {
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
        PeerPool::spawn(inits, teardown_ok).expect("spawn")
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
        let error = PeerPool::spawn(inits, count).err().expect("spawn fails");
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
        let pool = PeerPool::spawn(inits, teardown_ok).expect("spawn");
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
        let error = PeerPool::spawn(inits, teardown_ok).err().expect("fails");
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
        let pool = PeerPool::spawn(inits, bad).expect("spawn");
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
    fn mirror_specs_compare_by_shape_and_dtype() {
        let a = TensorSpec {
            shape: vec![248_320],
            dtype: DType::F32,
        };
        assert_eq!(a, a.clone());
        assert_ne!(
            a,
            TensorSpec {
                shape: vec![248_320],
                dtype: DType::F16,
            }
        );
        assert_ne!(
            a,
            TensorSpec {
                shape: vec![2, 124_160],
                dtype: DType::F32,
            }
        );
    }
}

#[cfg(test)]
mod tile_verdict_tests {
    use super::agreed_tile_verdicts;

    #[test]
    fn agreement_needs_every_rank_identical_and_complete() {
        for ranks in [2usize, 4] {
            let all = vec![Some(vec![true; 40]); ranks];
            assert_eq!(
                agreed_tile_verdicts(&all).map(Vec::len),
                Some(40),
                "{ranks}"
            );
            // One rank that did not admit the lever refuses it for all.
            for absent in 0..ranks {
                let mut mixed = all.clone();
                mixed[absent] = None;
                assert!(
                    agreed_tile_verdicts(&mixed).is_none(),
                    "{ranks} absent {absent}"
                );
            }
            // One rank with one asymmetric layer, or a different layer count, refuses.
            for odd in 0..ranks {
                let mut asymmetric = all.clone();
                asymmetric[odd].as_mut().unwrap()[7] = false;
                assert!(
                    agreed_tile_verdicts(&asymmetric).is_none(),
                    "{ranks} asym {odd}"
                );
                let mut shorter = all.clone();
                shorter[odd].as_mut().unwrap().pop();
                assert!(
                    agreed_tile_verdicts(&shorter).is_none(),
                    "{ranks} short {odd}"
                );
            }
        }
        assert!(agreed_tile_verdicts(&[]).is_none());
        assert!(agreed_tile_verdicts(&[Some(Vec::new()), Some(Vec::new())]).is_none());
    }
}
