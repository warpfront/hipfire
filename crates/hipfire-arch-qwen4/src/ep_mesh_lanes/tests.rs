// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host-only tests of the EP step driver: two scripted ranks on two real
//! threads, each crossing a simulated row-gather barrier exactly as many times
//! as its messages say. A rank that executes a different op list than its peer
//! therefore stalls (and the test fails by timeout) instead of passing; a
//! failing rank aborts the barrier so the peer wakes. No GPU is involved.

use super::*;
use crate::lane::LabRoutes;
use crate::mesh_lane_protocol::scripted::{Call, FailAt, Script, ScriptedExecutor, WindowScript};
use crate::mesh_lane_protocol::{run_mesh_step, GreedyRule, LaneAdmission, MeshKind, SamplingMode};
use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

fn epoch(tag: u64) -> RequestEpoch {
    RequestEpoch {
        request_tag: tag,
        owner_generation: 1,
    }
}

fn budgets(lab: LabRoutes, k: usize) -> WindowBudgets {
    WindowBudgets {
        max_lanes: 8,
        verify_rows_per_lane: k + 1,
        max_rows: 8 * (k + 1),
        copy_take_cap: 7,
        head_lag_rows: 0,
        lab,
        pp_microbatch_lanes: None,
    }
}

fn req(tag: u64, remaining: usize) -> LaneRequest {
    LaneRequest {
        wide_verify: false,
        epoch: epoch(tag),
        remaining,
        context_left: 10_000,
        forced: None,
        sampling: SamplingMode::Greedy,
    }
}

fn root(floor: bool) -> MeshRoot {
    root_with_moe_layers(floor, 48)
}

fn root_with_moe_layers(floor: bool, moe_layers: usize) -> MeshRoot {
    let world = vec![RankOwnership::ep(0), RankOwnership::ep(1)];
    MeshRoot::new(
        RankOwnership::ep(0),
        &world,
        RootConfig {
            topology_generation: 3,
            stage: StagePolicy {
                moe_layers,
                moe_once: true,
                stream_bf16: false,
            },
            floor_enabled: floor,
        },
    )
    .unwrap()
}

// ---- simulated row-gather barrier ----------------------------------------

struct RvState {
    arrived: usize,
    generation: u64,
    aborted: bool,
    crossings: u64,
}

/// A host barrier with a timeout and an abort, standing in for the EP row
/// gather: every crossing needs both ranks.
struct Rendezvous {
    state: Mutex<RvState>,
    cv: Condvar,
    ranks: usize,
    timeout: Duration,
}

impl Rendezvous {
    fn new(ranks: usize, timeout: Duration) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(RvState {
                arrived: 0,
                generation: 0,
                aborted: false,
                crossings: 0,
            }),
            cv: Condvar::new(),
            ranks,
            timeout,
        })
    }

    fn cross(&self, rank: usize) -> Result<(), RankError> {
        let mut s = self.state.lock().unwrap();
        if s.aborted {
            return Err(rank_err(rank, "row gather aborted"));
        }
        s.arrived += 1;
        if s.arrived == self.ranks {
            s.arrived = 0;
            s.generation += 1;
            s.crossings += 1;
            self.cv.notify_all();
            return Ok(());
        }
        let generation = s.generation;
        let (mut s, timed) = self
            .cv
            .wait_timeout_while(s, self.timeout, |s| {
                s.generation == generation && !s.aborted
            })
            .unwrap();
        if s.aborted {
            return Err(rank_err(rank, "row gather aborted"));
        }
        if timed.timed_out() {
            s.aborted = true;
            self.cv.notify_all();
            return Err(rank_err(rank, "row gather barrier timed out"));
        }
        Ok(())
    }

    fn abort(&self) {
        self.state.lock().unwrap().aborted = true;
        self.cv.notify_all();
    }

    fn crossings(&self) -> u64 {
        self.state.lock().unwrap().crossings
    }
}

/// A scripted rank whose collective phases cross the simulated gather: a
/// draft round once per op, a verify once per target exchange the sealed
/// ledger names (a rank replaying a different op list stalls its peer).
struct GatherSim {
    inner: ScriptedExecutor,
    rv: Arc<Rendezvous>,
}

impl LaneMeshExecutor for GatherSim {
    fn ownership(&self) -> RankOwnership {
        self.inner.ownership()
    }
    fn admit_lane(&mut self, a: &LaneAdmit) -> Result<(), RankError> {
        self.inner.admit_lane(a)
    }
    fn release_lane(&mut self, e: &RequestEpoch) -> Result<(), RankError> {
        self.inner.release_lane(e)
    }
    fn begin_window(&mut self, s: &WindowStart) -> RankReady {
        self.inner.begin_window(s)
    }
    fn reject_window(&mut self, k: &MeshStepKey) {
        self.inner.reject_window(k)
    }
    fn run_draft_round(&mut self, round: &DraftRound) -> Result<RankDraftResult, RankError> {
        let rank = self.ownership().rank;
        let result = self.inner.run_draft_round(round)?;
        for _ in 0..round.ops.len() {
            self.rv.cross(rank)?;
        }
        Ok(result)
    }
    fn run_verify(&mut self, plan: &VerifyPlan) -> Result<RankVerifyResult, RankError> {
        let rank = self.ownership().rank;
        let result = self.inner.run_verify(plan)?;
        for _ in 0..plan.ledger.target_exchanges {
            self.rv.cross(rank)?;
        }
        Ok(result)
    }
    fn prepare_commit(&mut self, d: &CommitDecision) -> Result<PreparedReceipt, RankError> {
        self.inner.prepare_commit(d)
    }
    fn publish_commit(&mut self, d: &CommitDecision) {
        self.inner.publish_commit(d)
    }
    fn abort_step(&mut self, k: &MeshStepKey, p: StepPhase, r: &str) {
        self.inner.abort_step(k, p, r)
    }
    fn evidence(&mut self, k: &MeshStepKey) -> Option<RankCostEvidence> {
        self.inner.evidence(k)
    }
}

// ---- threaded fan-out --------------------------------------------------------

type Job = Box<dyn FnOnce(&mut GatherSim) -> (Box<dyn Any + Send>, bool) + Send>;

/// What a rank thread hands back at the end (the scripted executor itself
/// holds `Rc`s and never leaves its thread).
struct SimView {
    calls: Vec<Call>,
    aborts: Vec<String>,
    published: Vec<u64>,
}

struct RankThread {
    tx: Sender<Option<Job>>,
    rx: Receiver<Box<dyn Any + Send>>,
    handle: Option<JoinHandle<SimView>>,
}

/// Two ranks on two threads: every phase is sent to all ranks before any reply
/// is awaited (the concurrency a real pool has), and a failing rank aborts the
/// gather before replying so its peer wakes.
struct ThreadedFanout {
    ranks: Vec<RankThread>,
    rv: Arc<Rendezvous>,
    publish_fatal: bool,
}

struct RankSetup {
    script: HashMap<(u64, u64), WindowScript>,
    fail_at: Option<FailAt>,
    skew_us: f32,
    rogue: bool,
    capacity_error: Option<String>,
}

impl Default for RankSetup {
    fn default() -> Self {
        Self {
            script: HashMap::new(),
            fail_at: None,
            skew_us: 0.0,
            rogue: false,
            capacity_error: None,
        }
    }
}

impl ThreadedFanout {
    fn new(setups: Vec<RankSetup>, timeout: Duration) -> Self {
        let rv = Rendezvous::new(setups.len(), timeout);
        let ranks = setups
            .into_iter()
            .enumerate()
            .map(|(rank, setup)| {
                let (tx, job_rx) = channel::<Option<Job>>();
                let (reply_tx, rx) = channel::<Box<dyn Any + Send>>();
                let rv_thread = rv.clone();
                let handle = thread::spawn(move || {
                    // The scripted executor holds `Rc`s: built on its own thread.
                    let script = Rc::new(RefCell::new(Script {
                        windows: setup.script,
                    }));
                    let mut inner = ScriptedExecutor::new(RankOwnership::ep(rank), script);
                    inner.fail_at = setup.fail_at;
                    inner.skew_us = setup.skew_us;
                    inner.rogue_sampler_output = setup.rogue;
                    inner.capacity_error = setup.capacity_error;
                    let mut sim = GatherSim {
                        inner,
                        rv: rv_thread.clone(),
                    };
                    while let Ok(Some(job)) = job_rx.recv() {
                        let (reply, failed) = job(&mut sim);
                        if failed {
                            rv_thread.abort();
                        }
                        if reply_tx.send(reply).is_err() {
                            break;
                        }
                    }
                    SimView {
                        calls: sim.inner.calls.clone(),
                        aborts: sim.inner.aborts.clone(),
                        published: sim.inner.published.clone(),
                    }
                });
                RankThread {
                    tx,
                    rx,
                    handle: Some(handle),
                }
            })
            .collect();
        Self {
            ranks,
            rv,
            publish_fatal: false,
        }
    }

    /// Send `f` to every rank, then collect every reply (rank order).
    fn all<T: Send + 'static>(
        &mut self,
        f: impl Fn(&mut GatherSim) -> (T, bool) + Send + Sync + Clone + 'static,
    ) -> Vec<T> {
        for r in &self.ranks {
            let f = f.clone();
            r.tx.send(Some(Box::new(move |sim: &mut GatherSim| {
                let (value, failed) = f(sim);
                (Box::new(value) as Box<dyn Any + Send>, failed)
            })))
            .unwrap();
        }
        self.ranks
            .iter()
            .map(|r| *r.rx.recv().unwrap().downcast::<T>().unwrap())
            .collect()
    }

    fn strict<T: Send + 'static>(
        &mut self,
        f: impl Fn(&mut GatherSim) -> Result<T, RankError> + Send + Sync + Clone + 'static,
    ) -> Result<Vec<T>, RankError> {
        let all = self.all(move |sim| {
            let r = f(sim);
            let failed = r.is_err();
            (r, failed)
        });
        // The first failing rank in rank order is the record's rank.
        all.into_iter().collect()
    }

    fn finish(mut self) -> Vec<SimView> {
        for r in &self.ranks {
            let _ = r.tx.send(None);
        }
        self.ranks
            .iter_mut()
            .map(|r| r.handle.take().unwrap().join().unwrap())
            .collect()
    }
}

impl MeshFanout for ThreadedFanout {
    fn admit(&mut self, admit: &LaneAdmit) -> Result<(), RankError> {
        let admit = admit.clone();
        self.all(move |s| (s.admit_lane(&admit), false))
            .into_iter()
            .collect()
    }
    fn release(&mut self, epoch: &RequestEpoch) -> Result<(), RankError> {
        let epoch = *epoch;
        self.all(move |s| (s.release_lane(&epoch), false))
            .into_iter()
            .collect()
    }
    fn begin(&mut self, start: &WindowStart) -> Result<Vec<RankReady>, RankError> {
        let start = start.clone();
        Ok(self.all(move |s| (s.begin_window(&start), false)))
    }
    fn reject(&mut self, key: &MeshStepKey) {
        let key = key.clone();
        self.all(move |s| (s.reject_window(&key), false));
    }
    fn draft(&mut self, round: &DraftRound) -> Result<Vec<RankDraftResult>, RankError> {
        let round = round.clone();
        self.strict(move |s| s.run_draft_round(&round))
    }
    fn verify(&mut self, plan: &VerifyPlan) -> Result<Vec<RankVerifyResult>, RankError> {
        let plan = plan.clone();
        self.strict(move |s| s.run_verify(&plan))
    }
    fn prepare(&mut self, d: &CommitDecision) -> Result<Vec<PreparedReceipt>, RankError> {
        let d = d.clone();
        self.strict(move |s| s.prepare_commit(&d))
    }
    fn publish(&mut self, d: &CommitDecision) -> Vec<Option<String>> {
        let d = d.clone();
        let fatal = self.publish_fatal;
        self.all(move |s| {
            s.publish_commit(&d);
            (fatal.then(|| "scripted publish failure".to_string()), false)
        })
    }
    fn abort(&mut self, key: &MeshStepKey, phase: StepPhase, reason: &str) {
        self.rv.abort();
        let (key, reason) = (key.clone(), reason.to_string());
        self.all(move |s| (s.abort_step(&key, phase, &reason), false));
    }
    fn evidence(&mut self, key: &MeshStepKey) -> Vec<Option<RankCostEvidence>> {
        let key = key.clone();
        self.all(move |s| (s.evidence(&key), false))
    }
}

fn script(tag: u64, seq: u64, drafts: &[u32], picks: &[u32]) -> ((u64, u64), WindowScript) {
    (
        (tag, seq),
        WindowScript {
            drafts: drafts.to_vec(),
            picks: picks.to_vec(),
            keep_going: Vec::new(),
            drop_last: Vec::new(),
        },
    )
}

fn fanout_with(
    windows: Vec<((u64, u64), WindowScript)>,
    mutate: impl Fn(usize, &mut RankSetup),
) -> ThreadedFanout {
    let mut setups: Vec<RankSetup> = (0..2)
        .map(|_| RankSetup {
            script: windows.iter().cloned().collect(),
            ..RankSetup::default()
        })
        .collect();
    for (rank, setup) in setups.iter_mut().enumerate() {
        mutate(rank, setup);
    }
    ThreadedFanout::new(setups, Duration::from_secs(5))
}

fn admit(
    root: &mut MeshRoot,
    fan: &mut ThreadedFanout,
    tag: u64,
    position: usize,
    seed: u32,
    eos: Option<u32>,
    spec_k: Option<usize>,
) {
    let a = root
        .admit_lane(LaneAdmission {
            epoch: epoch(tag),
            position,
            seed,
            eos,
            rng_seed: 99 + tag,
            prompt: vec![1, 2, 3],
            spec_k,
            head: HeadState::Aligned,
            copy: false,
            head_lag: Vec::new(),
        })
        .unwrap();
    fan.admit(&a).unwrap();
}

fn calls_of(sims: &[SimView]) -> Vec<Vec<Call>> {
    sims.iter().map(|s| s.calls.clone()).collect()
}

// ---- tests ---------------------------------------------------------------------

#[test]
fn mesh_lane_ep_ranks_replay_identical_ordered_calls_and_the_ledger_merges() {
    let mut root = root(false);
    let mut fan = fanout_with(
        vec![
            // Lane 1: three drafts, the target diverges at the second.
            script(1, 0, &[10, 11, 12], &[10, 77, 5, 6]),
            // Lane 2: one draft, fully accepted; the head hints it is done.
            {
                let (key, mut w) = script(2, 0, &[20], &[20, 8]);
                w.keep_going = vec![false];
                (key, w)
            },
        ],
        |_, _| {},
    );
    admit(&mut root, &mut fan, 1, 10, 9, None, Some(3));
    admit(&mut root, &mut fan, 2, 10, 9, None, Some(3));
    let driven = drive_step(
        &mut root,
        &mut fan,
        None,
        &[req(1, 64), req(2, 64)],
        &budgets(LabRoutes::Batched, 3),
        &mut GreedyRule,
    )
    .expect("a clean two-rank step");
    // Zero acceptance after one match: accepted 1, consumed 2; the bonus (the
    // target's own pick) is emitted but not consumed.
    let lane1 = &driven.output.lanes[0];
    assert_eq!(lane1.emitted, vec![10, 77]);
    assert_eq!(lane1.consumed_rows, 2);
    let lane2 = &driven.output.lanes[1];
    assert_eq!(lane2.emitted, vec![20, 8]);
    assert_eq!(lane2.consumed_rows, 2);
    // Both ranks answered the evidence and the merged counts are coherent.
    assert_eq!(driven.receipts.len(), 2);
    let merged = root
        .merge_evidence(
            &driven.receipts,
            ControllerInterval {
                wall_us: 40_000.0,
                prefill_ran: false,
            },
        )
        .expect("coherent rank receipts");
    assert_eq!(merged.ranks, 2);
    // Lane 1: rows 1+3=4, lane 2: rows 2; one row group of six rows: 48
    // target exchanges; head Draft calls: 3 + 1.
    assert_eq!(merged.counts.target_exchanges, 48);
    assert_eq!(merged.counts.head_draft_calls, 4);
    // Head rounds cross once per op: 4; verify crosses 48.
    assert_eq!(fan.rv.crossings(), 4 + 48);
    let sims = fan.finish();
    let calls = calls_of(&sims);
    assert_eq!(calls[0], calls[1], "ranks must see the same ordered calls");
    assert!(sims.iter().all(|s| s.aborts.is_empty()));
    assert_eq!(sims[0].published, vec![driven.key.step_seq]);
    assert_eq!(sims[1].published, vec![driven.key.step_seq]);
}

/// The trunk shard's gatherless trunk: the sealed plan holds zero target exchanges, the verify
/// crosses the row-gather barrier zero times, and the merged ledger agrees (production sealed 48
/// against a measured 0 and aborted every drafted step).
#[test]
fn mesh_lane_trunk_shard_plans_zero_target_exchanges() {
    assert_eq!(planned_gather_layers(48, false), 48);
    assert_eq!(planned_gather_layers(48, true), 0);
    let mut root = root_with_moe_layers(false, planned_gather_layers(48, true));
    let mut fan = fanout_with(
        vec![script(1, 0, &[10, 11, 12], &[10, 77, 5, 6])],
        |_, _| {},
    );
    admit(&mut root, &mut fan, 1, 10, 9, None, Some(3));
    let driven = drive_step(
        &mut root,
        &mut fan,
        None,
        &[req(1, 64)],
        &budgets(LabRoutes::Batched, 3),
        &mut GreedyRule,
    )
    .expect("a clean two-rank step");
    let merged = root
        .merge_evidence(
            &driven.receipts,
            ControllerInterval {
                wall_us: 40_000.0,
                prefill_ran: false,
            },
        )
        .expect("coherent rank receipts");
    assert_eq!(merged.counts.target_exchanges, 0);
    assert_eq!(merged.counts.head_draft_calls, 3);
    // Head rounds cross once per op (3); the gatherless verify crosses none.
    assert_eq!(fan.rv.crossings(), 3);
    let sims = fan.finish();
    assert!(sims.iter().all(|s| s.aborts.is_empty()));
}

#[test]
fn mesh_lane_ep_serial_dispatch_of_a_collective_phase_cannot_rendezvous() {
    // `run_mesh_step` dispatches serially: rank 0 enters the row gather and
    // waits for a peer that has not been called yet. The EP driver exists
    // because of exactly this.
    let rv = Rendezvous::new(2, Duration::from_millis(200));
    let script_rc = Rc::new(RefCell::new(Script::default()));
    script_rc
        .borrow_mut()
        .windows
        .extend([script(1, 0, &[10], &[10, 4])]);
    let mut ranks: Vec<GatherSim> = (0..2)
        .map(|r| GatherSim {
            inner: ScriptedExecutor::new(RankOwnership::ep(r), script_rc.clone()),
            rv: rv.clone(),
        })
        .collect();
    let mut root = root(false);
    let a = root
        .admit_lane(LaneAdmission {
            epoch: epoch(1),
            position: 10,
            seed: 9,
            eos: None,
            rng_seed: 5,
            prompt: vec![1],
            spec_k: Some(1),
            head: HeadState::Aligned,
            copy: false,
            head_lag: Vec::new(),
        })
        .unwrap();
    for r in &mut ranks {
        r.admit_lane(&a).unwrap();
    }
    let mut refs: Vec<&mut dyn LaneMeshExecutor> = ranks
        .iter_mut()
        .map(|r| r as &mut dyn LaneMeshExecutor)
        .collect();
    let error = run_mesh_step(
        &mut root,
        &mut refs,
        None,
        &[req(1, 8)],
        &budgets(LabRoutes::Batched, 1),
        &mut GreedyRule,
    )
    .expect_err("the first head gather cannot be crossed by one rank");
    assert!(matches!(error, ProtocolError::Aborted(_)), "{error}");
}

#[test]
fn mesh_lane_ep_a_rank_rejection_rejects_the_whole_step_before_any_collective() {
    let mut root = root(false);
    let mut fan = fanout_with(vec![script(1, 0, &[10], &[10, 4])], |rank, setup| {
        if rank == 1 {
            setup.capacity_error = Some("rank 1: lane store needs 9 bytes, 3 free".to_string());
        }
    });
    admit(&mut root, &mut fan, 1, 10, 9, None, Some(1));
    let error = drive_step(
        &mut root,
        &mut fan,
        None,
        &[req(1, 8)],
        &budgets(LabRoutes::Batched, 1),
        &mut GreedyRule,
    )
    .expect_err("rank 1 refuses the plan");
    let ProtocolError::Preflight(rejections) = error else {
        panic!("a preflight rejection is not an abort: {error}");
    };
    assert_eq!(rejections.len(), 1);
    assert_eq!(rejections[0].0, 1);
    assert_eq!(root.phase(), StepPhase::Idle);
    assert!(root.quarantined().is_none());
    // No rank crossed a gather or ran a head/verify call.
    assert_eq!(fan.rv.crossings(), 0);
    let sims = fan.finish();
    for sim in &sims {
        assert!(
            sim.calls.iter().all(|c| matches!(c, Call::Begin(_))),
            "{:?}",
            sim.calls
        );
        assert!(sim.published.is_empty());
        assert!(sim.aborts.is_empty(), "a rejection is not an abort");
    }
}

#[test]
fn mesh_lane_ep_a_verify_failure_on_one_rank_aborts_every_rank_and_wakes_the_gather() {
    let mut root = root(false);
    let mut fan = fanout_with(
        vec![script(1, 0, &[10, 11], &[10, 11, 4])],
        |rank, setup| {
            if rank == 1 {
                setup.fail_at = Some(FailAt::Verify);
            }
        },
    );
    admit(&mut root, &mut fan, 1, 10, 9, None, Some(2));
    let started = std::time::Instant::now();
    let error = drive_step(
        &mut root,
        &mut fan,
        None,
        &[req(1, 16)],
        &budgets(LabRoutes::Batched, 2),
        &mut GreedyRule,
    )
    .expect_err("one rank's forward fails");
    assert!(matches!(error, ProtocolError::Aborted(_)), "{error}");
    // Rank 0 was inside the verify's gather: the failure woke it at once (the
    // barrier timeout is 5 s).
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "the peer waited for the barrier timeout"
    );
    assert!(root.quarantined().is_some());
    let sims = fan.finish();
    for sim in &sims {
        assert!(!sim.aborts.is_empty(), "every rank aborts");
        assert!(sim.published.is_empty(), "nothing is published");
    }
    let record = root.quarantined().unwrap();
    assert_eq!(record.phase, StepPhase::Armed.max(record.phase));
}

#[test]
fn mesh_lane_ep_a_head_failure_between_rounds_aborts_every_rank() {
    let mut root = root(false);
    let mut fan = fanout_with(
        vec![script(1, 0, &[10, 11, 12], &[10, 11, 12, 4])],
        |rank, setup| {
            if rank == 1 {
                setup.fail_at = Some(FailAt::Draft(1));
            }
        },
    );
    admit(&mut root, &mut fan, 1, 10, 9, None, Some(3));
    let error = drive_step(
        &mut root,
        &mut fan,
        None,
        &[req(1, 16)],
        &budgets(LabRoutes::Batched, 3),
        &mut GreedyRule,
    )
    .expect_err("round 1 fails on rank 1");
    assert!(matches!(error, ProtocolError::Aborted(_)), "{error}");
    let sims = fan.finish();
    assert!(sims.iter().all(|s| !s.aborts.is_empty()));
    assert!(sims.iter().all(|s| s.published.is_empty()));
}

#[test]
fn mesh_lane_ep_a_repair_failure_after_verify_aborts_before_any_publication() {
    let mut root = root(false);
    let mut fan = fanout_with(vec![script(1, 0, &[10], &[10, 4])], |rank, setup| {
        if rank == 0 {
            setup.fail_at = Some(FailAt::Prepare);
        }
    });
    admit(&mut root, &mut fan, 1, 10, 9, None, Some(1));
    let error = drive_step(
        &mut root,
        &mut fan,
        None,
        &[req(1, 16)],
        &budgets(LabRoutes::Batched, 1),
        &mut GreedyRule,
    )
    .expect_err("a prepare failure poisons the step");
    assert!(matches!(error, ProtocolError::Aborted(_)), "{error}");
    assert_eq!(root.counters().steps_published, 0);
    assert!(
        root.generated(epoch(1)).unwrap().is_empty(),
        "no IDs leaked"
    );
    let sims = fan.finish();
    assert!(sims.iter().all(|s| s.published.is_empty()));
}

#[test]
fn mesh_lane_ep_a_follower_that_returns_proposals_is_an_abort() {
    let mut root = root(false);
    let mut fan = fanout_with(vec![script(1, 0, &[10], &[10, 4])], |rank, setup| {
        setup.rogue = rank == 1;
    });
    admit(&mut root, &mut fan, 1, 10, 9, None, Some(1));
    let error = drive_step(
        &mut root,
        &mut fan,
        None,
        &[req(1, 16)],
        &budgets(LabRoutes::Batched, 1),
        &mut GreedyRule,
    )
    .expect_err("the root alone reads proposals");
    assert!(matches!(error, ProtocolError::Aborted(_)), "{error}");
    let sims = fan.finish();
    assert!(sims.iter().all(|s| s.published.is_empty()));
}

#[test]
fn mesh_lane_ep_local_timing_skew_cannot_change_the_plan_or_the_calls() {
    let windows = vec![
        script(1, 0, &[10, 11], &[10, 11, 4]),
        script(1, 1, &[5, 6], &[5, 99, 1]),
    ];
    let run = |skew: f32| {
        let mut root = root(false);
        let mut fan = fanout_with(windows.clone(), |rank, setup| {
            if rank == 1 {
                setup.skew_us = skew;
            }
        });
        admit(&mut root, &mut fan, 1, 10, 9, None, Some(2));
        let mut prior = None;
        let mut keys = Vec::new();
        for _ in 0..2 {
            let driven = drive_step(
                &mut root,
                &mut fan,
                prior.as_ref(),
                &[req(1, 64)],
                &budgets(LabRoutes::Batched, 2),
                &mut GreedyRule,
            )
            .unwrap();
            keys.push(driven.key.clone());
            prior = Some(
                root.merge_evidence(
                    &driven.receipts,
                    ControllerInterval {
                        wall_us: 30_000.0,
                        prefill_ran: false,
                    },
                )
                .unwrap(),
            );
        }
        let sims = fan.finish();
        (keys, calls_of(&sims), prior.unwrap().critical_path_us)
    };
    let (keys_a, calls_a, path_a) = run(0.0);
    let (keys_b, calls_b, path_b) = run(5_000.0);
    assert_eq!(keys_a, keys_b, "the root's plan ignores rank-local timing");
    assert_eq!(calls_a, calls_b);
    assert_eq!(calls_a[0], calls_a[1]);
    // The skew is evidence (the critical path is the slower rank), not policy.
    assert!(path_b > path_a);
}

#[test]
fn mesh_lane_ep_cycle_routes_mix_ar_and_native_windows_on_matched_ranks() {
    // Cycle: Batched{K}, Batched{1}, ArSynced, Batched{2}.
    let windows = vec![
        script(1, 0, &[10, 11, 12], &[10, 11, 12, 4]),
        script(1, 1, &[20], &[20, 5]),
        script(1, 2, &[], &[8]),
        script(1, 3, &[30, 31], &[30, 31, 9]),
    ];
    let mut root = root(false);
    let mut fan = fanout_with(windows, |_, _| {});
    admit(&mut root, &mut fan, 1, 10, 9, None, Some(3));
    let mut emitted = Vec::new();
    for _ in 0..4 {
        let driven = drive_step(
            &mut root,
            &mut fan,
            None,
            &[req(1, 64)],
            &budgets(LabRoutes::Cycle, 3),
            &mut GreedyRule,
        )
        .expect("every cycle window");
        emitted.push(driven.output.lanes[0].emitted.clone());
    }
    assert_eq!(
        emitted,
        vec![vec![10, 11, 12, 4], vec![20, 5], vec![8], vec![30, 31, 9]]
    );
    let sims = fan.finish();
    let calls = calls_of(&sims);
    assert_eq!(calls[0], calls[1]);
}

#[test]
fn mesh_lane_ep_a_failure_after_the_all_prepared_boundary_quarantines_the_mesh() {
    let mut root = root(false);
    let mut fan = fanout_with(vec![script(1, 0, &[10], &[10, 4])], |_, _| {});
    fan.publish_fatal = true;
    admit(&mut root, &mut fan, 1, 10, 9, None, Some(1));
    let error = drive_step(
        &mut root,
        &mut fan,
        None,
        &[req(1, 16)],
        &budgets(LabRoutes::Batched, 1),
        &mut GreedyRule,
    )
    .expect_err("a publish failure is fatal");
    assert!(matches!(error, ProtocolError::Aborted(_)), "{error}");
    assert!(root.quarantined().is_some());
    let sims = fan.finish();
    assert!(sims.iter().all(|s| !s.aborts.is_empty()));
}

#[test]
fn mesh_lane_ep_step_errors_map_preflight_abort_and_protocol() {
    let preflight = MeshStepError::from(ProtocolError::Preflight(vec![(1, "full".into())]));
    assert!(matches!(&preflight, MeshStepError::Preflight(v) if v[0].0 == 1));
    let record = AbortRecord {
        key: None,
        phase: StepPhase::Armed,
        rank: Some(1),
        reason: "x".into(),
    };
    assert!(matches!(
        MeshStepError::from(ProtocolError::Aborted(record)),
        MeshStepError::Aborted(_)
    ));
    assert!(matches!(
        MeshStepError::from(ProtocolError::Budget("b".into())),
        MeshStepError::Protocol(_)
    ));
    assert_eq!(MeshKind::Ep, MeshKind::Ep);
}

// ---- sampled lanes through the EP driver -----------------------------------

const SAMPLED_VOCAB: usize = 10;

struct ConstRows(Vec<f32>);

impl TargetRowSource for ConstRows {
    fn row_logits(
        &mut self,
        _key: &LaneWindowKey,
        _row: usize,
        host: &mut RowBuf,
    ) -> Result<(), String> {
        host.set(&self.0);
        Ok(())
    }
}

/// [`ThreadedFanout`] with what a real sampler owner adds: the head's
/// candidate read on every native proposal (rank 0) and the row handle in the
/// verify answer. `reads: false` is a sampler owner that forgot the read.
struct SampledFan {
    inner: ThreadedFanout,
    reads: bool,
}

impl MeshFanout for SampledFan {
    fn admit(&mut self, admit: &LaneAdmit) -> Result<(), RankError> {
        self.inner.admit(admit)
    }
    fn release(&mut self, epoch: &RequestEpoch) -> Result<(), RankError> {
        self.inner.release(epoch)
    }
    fn begin(&mut self, start: &WindowStart) -> Result<Vec<RankReady>, RankError> {
        self.inner.begin(start)
    }
    fn reject(&mut self, key: &MeshStepKey) {
        self.inner.reject(key)
    }
    fn draft(&mut self, round: &DraftRound) -> Result<Vec<RankDraftResult>, RankError> {
        let mut results = self.inner.draft(round)?;
        if self.reads {
            for proposal in results
                .iter_mut()
                .filter_map(|r| r.proposals.as_mut())
                .flatten()
            {
                proposal.read = Some(DraftRead::Candidates(vec![
                    (0, 1.8),
                    (2, 1.7),
                    (1, 0.2),
                    (5, 1.0),
                    (3, 0.1),
                    (7, 0.9),
                    (4, -0.5),
                    (6, -1.0),
                ]));
            }
        }
        Ok(results)
    }
    fn verify(&mut self, plan: &VerifyPlan) -> Result<Vec<RankVerifyResult>, RankError> {
        let mut results = self.inner.verify(plan)?;
        let rows = RowHandle::new(ConstRows(
            (0..SAMPLED_VOCAB).map(|i| 2.0 - 0.4 * i as f32).collect(),
        ));
        for sampler in results.iter_mut().filter_map(|r| r.sampler.as_mut()) {
            sampler.rows = Some(rows.clone());
        }
        Ok(results)
    }
    fn prepare(&mut self, d: &CommitDecision) -> Result<Vec<PreparedReceipt>, RankError> {
        self.inner.prepare(d)
    }
    fn publish(&mut self, d: &CommitDecision) -> Vec<Option<String>> {
        self.inner.publish(d)
    }
    fn abort(&mut self, key: &MeshStepKey, phase: StepPhase, reason: &str) {
        self.inner.abort(key, phase, reason)
    }
    fn evidence(&mut self, key: &MeshStepKey) -> Vec<Option<RankCostEvidence>> {
        self.inner.evidence(key)
    }
}

fn sampled_req(tag: u64) -> LaneRequest {
    LaneRequest {
        wide_verify: false,
        sampling: SamplingMode::Leviathan,
        ..req(tag, 64)
    }
}

fn sampled_run(seed: u64, reads: bool) -> Result<Vec<u32>, ProtocolError> {
    let mut root = root(false);
    let mut fan = SampledFan {
        inner: fanout_with(vec![script(1, 0, &[3, 3, 3], &[0, 0, 0, 0])], |_, _| {}),
        reads,
    };
    let a = root
        .admit_lane(LaneAdmission {
            epoch: epoch(1),
            position: 10,
            seed: 9,
            eos: None,
            rng_seed: seed,
            prompt: vec![1, 2, 3],
            spec_k: Some(3),
            head: HeadState::Aligned,
            copy: false,
            head_lag: Vec::new(),
        })
        .unwrap();
    fan.admit(&a).unwrap();
    let mut sampler = LaneSampler::new(SAMPLED_VOCAB);
    sampler
        .admit(
            epoch(1),
            &SpecRequestConfig {
                temp: 1.0,
                rng_seed: seed,
                ..SpecRequestConfig::default()
            },
            &[1, 2, 3],
            9,
        )
        .unwrap();
    let driven = drive_step(
        &mut root,
        &mut fan,
        None,
        &[sampled_req(1)],
        &budgets(LabRoutes::Batched, 3),
        &mut sampler,
    );
    let emitted = driven.map(|d| d.output.lanes[0].emitted.clone());
    if emitted.is_ok() {
        // `on_publish` ran: the lane stays registered (not terminal) and the
        // sampler dropped its row handle.
        assert!(sampler.is_registered(&epoch(1)));
    }
    fan.inner.finish();
    emitted
}

#[test]
fn mesh_lane_ep_sampled_lane_draws_on_the_root_from_the_ranks_reads() {
    let first = sampled_run(7, true).expect("a sampled step on two scripted ranks");
    assert!(!first.is_empty() && first.iter().all(|&t| (t as usize) < SAMPLED_VOCAB));
    // Root-only RNG: the same seed replays the same tokens regardless of the
    // ranks (the scripted picks are all 0, the draws are not).
    assert_eq!(sampled_run(7, true).unwrap(), first);
    // Over a few seeds the draws are not the scripted argmax stream.
    let any_drawn = (0..16u64).any(|seed| sampled_run(seed, true).unwrap() != vec![0, 0, 0, 0]);
    assert!(
        any_drawn,
        "sampled acceptance must draw, not follow the picks"
    );
}

#[test]
fn mesh_lane_ep_sampled_lane_without_the_head_read_aborts_the_step() {
    // The driver hands the rule every proposal (`collect_draft_with`): a
    // Leviathan lane with no candidate read is an abort, never a silent
    // greedy proposal.
    let error = sampled_run(7, false).expect_err("a Leviathan proposal needs its read");
    assert!(matches!(error, ProtocolError::Aborted(_)), "{error}");
}

#[test]
fn a_full_row_read_round_with_several_ops_is_sequential() {
    // Only a read of a head without re-scoring (the whole shared row) over more
    // than one op needs the one-op-at-a-time schedule.
    assert_eq!(draft_read_plan(true, false, 3), DraftReadPlan::Sequential);
    assert_eq!(draft_read_plan(true, false, 2), DraftReadPlan::Sequential);
    // One op, a re-scoring head (host-resident candidates) and a round with
    // no distribution read keep the one batched download.
    assert_eq!(draft_read_plan(true, false, 1), DraftReadPlan::Batched);
    assert_eq!(draft_read_plan(true, true, 4), DraftReadPlan::Batched);
    assert_eq!(draft_read_plan(false, false, 4), DraftReadPlan::Batched);
    assert_eq!(draft_read_plan(false, true, 4), DraftReadPlan::Batched);
    assert_eq!(draft_read_plan(true, false, 0), DraftReadPlan::Batched);
}

#[test]
fn the_read_schedule_covers_every_op_once_in_order() {
    assert_eq!(DraftReadPlan::Batched.chunks(3), vec![0..3]);
    assert_eq!(DraftReadPlan::Sequential.chunks(3), vec![0..1, 1..2, 2..3]);
    assert!(DraftReadPlan::Sequential.chunks(0).is_empty());
}

/// A fake head: every enqueued op overwrites one shared "logit row" with its
/// own marker (the real draft head's single `logits` tensor); a FullRow read
/// copies that row at read time.
fn run_round_on_shared_row(plan: DraftReadPlan, ops: usize) -> Vec<u32> {
    let mut shared_row = u32::MAX;
    let mut reads = Vec::new();
    for chunk in plan.chunks(ops) {
        for op in chunk.clone() {
            shared_row = 100 + op as u32; // enqueue: the op's step writes the row
        }
        for _ in chunk {
            reads.push(shared_row); // finish + read right after the chunk's enqueues
        }
    }
    reads
}

#[test]
fn full_row_reads_of_two_ops_are_distinct_only_when_sequential() {
    // The hazard the plan removes: batched, every op reads the last op's row.
    assert_eq!(
        run_round_on_shared_row(DraftReadPlan::Batched, 3),
        vec![102, 102, 102]
    );
    // Sequential: each op reads the row its own step wrote.
    let plan = draft_read_plan(true, false, 3);
    assert_eq!(plan, DraftReadPlan::Sequential);
    assert_eq!(run_round_on_shared_row(plan, 3), vec![100, 101, 102]);
}

// ---- eight sampled lanes: every lane draws from its OWN reads and rows ------

/// Lane `tag`'s target rows and head candidates are a spike at token `tag`
/// (the lane's marker): whatever a lane emits must be its own marker, so a
/// read or a row taken from another lane's slot, offset or key shows as a
/// foreign token.
struct KeyedRows;

impl TargetRowSource for KeyedRows {
    fn row_logits(
        &mut self,
        key: &LaneWindowKey,
        row: usize,
        host: &mut RowBuf,
    ) -> Result<(), String> {
        assert!(
            row < 4,
            "a K<=3 window has at most 4 verify rows, asked row {row}"
        );
        let row: Vec<f32> = (0..SAMPLED_VOCAB)
            .map(|t| {
                if t as u64 == key.request_epoch.request_tag {
                    60.0
                } else {
                    0.0
                }
            })
            .collect();
        host.set(&row);
        Ok(())
    }
}

struct KeyedFan {
    inner: ThreadedFanout,
}

impl MeshFanout for KeyedFan {
    fn admit(&mut self, admit: &LaneAdmit) -> Result<(), RankError> {
        self.inner.admit(admit)
    }
    fn release(&mut self, epoch: &RequestEpoch) -> Result<(), RankError> {
        self.inner.release(epoch)
    }
    fn begin(&mut self, start: &WindowStart) -> Result<Vec<RankReady>, RankError> {
        self.inner.begin(start)
    }
    fn reject(&mut self, key: &MeshStepKey) {
        self.inner.reject(key)
    }
    fn draft(&mut self, round: &DraftRound) -> Result<Vec<RankDraftResult>, RankError> {
        let mut results = self.inner.draft(round)?;
        for proposal in results
            .iter_mut()
            .filter_map(|r| r.proposals.as_mut())
            .flatten()
        {
            let marker = proposal.key.request_epoch.request_tag as u32;
            let mut candidates = vec![(marker, 60.0f32)];
            candidates.extend(
                (0..SAMPLED_VOCAB as u32)
                    .filter(|&t| t != marker)
                    .take(7)
                    .map(|t| (t, -60.0f32)),
            );
            proposal.read = Some(DraftRead::Candidates(candidates));
        }
        Ok(results)
    }
    fn verify(&mut self, plan: &VerifyPlan) -> Result<Vec<RankVerifyResult>, RankError> {
        let mut results = self.inner.verify(plan)?;
        let rows = RowHandle::new(KeyedRows);
        for sampler in results.iter_mut().filter_map(|r| r.sampler.as_mut()) {
            sampler.rows = Some(rows.clone());
        }
        Ok(results)
    }
    fn prepare(&mut self, d: &CommitDecision) -> Result<Vec<PreparedReceipt>, RankError> {
        self.inner.prepare(d)
    }
    fn publish(&mut self, d: &CommitDecision) -> Vec<Option<String>> {
        self.inner.publish(d)
    }
    fn abort(&mut self, key: &MeshStepKey, phase: StepPhase, reason: &str) {
        self.inner.abort(key, phase, reason)
    }
    fn evidence(&mut self, key: &MeshStepKey) -> Vec<Option<RankCostEvidence>> {
        self.inner.evidence(key)
    }
}

#[test]
fn mesh_lane_ep_eight_sampled_lanes_each_emit_their_own_marker() {
    for k in [2usize, 3] {
        let tags: Vec<u64> = (1..=8).collect();
        let mut root = root(false);
        let mut fan = KeyedFan {
            inner: fanout_with(
                tags.iter()
                    .map(|&tag| script(tag, 0, &vec![0; k], &vec![0; k + 1]))
                    .collect(),
                |_, _| {},
            ),
        };
        let mut sampler = LaneSampler::new(SAMPLED_VOCAB);
        for &tag in &tags {
            let a = root
                .admit_lane(LaneAdmission {
                    epoch: epoch(tag),
                    position: 10,
                    seed: 9,
                    eos: None,
                    rng_seed: 1000 + tag,
                    prompt: vec![1, 2, 3],
                    spec_k: Some(k),
                    head: HeadState::Aligned,
                    copy: false,
                    head_lag: Vec::new(),
                })
                .unwrap();
            fan.admit(&a).unwrap();
            sampler
                .admit(
                    epoch(tag),
                    &SpecRequestConfig {
                        temp: 1.0,
                        rng_seed: 1000 + tag,
                        ..SpecRequestConfig::default()
                    },
                    &[1, 2, 3],
                    9,
                )
                .unwrap();
        }
        let requests: Vec<LaneRequest> = tags.iter().map(|&tag| sampled_req(tag)).collect();
        let driven = drive_step(
            &mut root,
            &mut fan,
            None,
            &requests,
            &budgets(LabRoutes::Batched, k),
            &mut sampler,
        )
        .expect("an eight-lane sampled step on two scripted ranks");
        assert_eq!(driven.output.lanes.len(), 8);
        for (lane, &tag) in driven.output.lanes.iter().zip(&tags) {
            assert!(
                !lane.emitted.is_empty() && lane.emitted.iter().all(|&t| u64::from(t) == tag),
                "K={k}: lane {tag} (slot {}) emitted {:?}, expected only its marker {tag}",
                tag - 1,
                lane.emitted
            );
            assert_eq!(
                lane.emitted.len(),
                k + 1,
                "K={k}: lane {tag} emitted {:?}",
                lane.emitted
            );
        }
        fan.inner.finish();
    }
}

// ---- device-drawn sampled lanes (HIPFIRE_QWEN4_LANE_DEVICE_SAMPLE) ------------

/// A rank fan whose sampler owner answers a verify like the device sampler does:
/// every lane the root lent a spec is drawn by the shared device sampler's CPU
/// model (`hipfire_runtime::lane_draw::reference_draw`) over that lane's keyed
/// rows, drafts, q, history and RNG, and the rows themselves are withheld
/// (`rows: None`) unless `host_rows`. The head's candidate read is the keyed
/// spike of [`KeyedFan`] (a Leviathan lane draws its drafts from it on the root).
struct DeviceFan {
    inner: ThreadedFanout,
    /// Also hand the root a row handle (what the sampler owner does for a lane
    /// it was not asked to draw on device).
    host_rows: bool,
    /// Answer without the device picks the plan asked for.
    withhold: bool,
    /// Specs the plans lent, per verify.
    lent: Vec<usize>,
    /// The draw made for each lent lane (by request tag), per verify.
    draws: Vec<(u64, hipfire_runtime::lane_draw::LaneDraw)>,
}

impl MeshFanout for DeviceFan {
    fn admit(&mut self, admit: &LaneAdmit) -> Result<(), RankError> {
        self.inner.admit(admit)
    }
    fn release(&mut self, epoch: &RequestEpoch) -> Result<(), RankError> {
        self.inner.release(epoch)
    }
    fn begin(&mut self, start: &WindowStart) -> Result<Vec<RankReady>, RankError> {
        self.inner.begin(start)
    }
    fn reject(&mut self, key: &MeshStepKey) {
        self.inner.reject(key)
    }
    fn draft(&mut self, round: &DraftRound) -> Result<Vec<RankDraftResult>, RankError> {
        let mut results = self.inner.draft(round)?;
        for proposal in results
            .iter_mut()
            .filter_map(|r| r.proposals.as_mut())
            .flatten()
        {
            let marker = proposal.key.request_epoch.request_tag as u32;
            let mut candidates = vec![(marker, 60.0f32)];
            candidates.extend(
                (0..SAMPLED_VOCAB as u32)
                    .filter(|&t| t != marker)
                    .take(7)
                    .map(|t| (t, -60.0f32)),
            );
            proposal.read = Some(DraftRead::Candidates(candidates));
        }
        Ok(results)
    }
    fn verify(&mut self, plan: &VerifyPlan) -> Result<Vec<RankVerifyResult>, RankError> {
        let mut results = self.inner.verify(plan)?;
        self.lent.push(plan.samples.iter().flatten().count());
        for sampler in results.iter_mut().filter_map(|r| r.sampler.as_mut()) {
            if self.host_rows {
                sampler.rows = Some(RowHandle::new(KeyedRows));
            }
            if self.withhold {
                continue;
            }
            sampler.device = plan
                .lanes
                .iter()
                .zip(&plan.samples)
                .map(|(vl, spec)| {
                    let spec = spec.as_ref()?;
                    let tag = vl.key.request_epoch.request_tag;
                    let marker = tag as usize;
                    let row: Vec<f32> = (0..SAMPLED_VOCAB)
                        .map(|t| if t == marker { 60.0 } else { 0.0 })
                        .collect();
                    let rows = row.repeat(vl.rows.len());
                    let draw = hipfire_runtime::lane_draw::reference_draw(
                        &rows,
                        SAMPLED_VOCAB,
                        &vl.rows[1..],
                        &spec.q,
                        &spec.history,
                        &spec.cfg,
                        spec.rng_state,
                        spec.eos,
                    );
                    self.draws.push((tag, draw.clone()));
                    Some(DevicePick {
                        tokens: draw.tokens,
                        accepted: draw.accepted,
                        advanced_rng: draw.advanced_rng,
                    })
                })
                .collect();
        }
        Ok(results)
    }
    fn prepare(&mut self, d: &CommitDecision) -> Result<Vec<PreparedReceipt>, RankError> {
        self.inner.prepare(d)
    }
    fn publish(&mut self, d: &CommitDecision) -> Vec<Option<String>> {
        self.inner.publish(d)
    }
    fn abort(&mut self, key: &MeshStepKey, phase: StepPhase, reason: &str) {
        self.inner.abort(key, phase, reason)
    }
    fn evidence(&mut self, key: &MeshStepKey) -> Vec<Option<RankCostEvidence>> {
        self.inner.evidence(key)
    }
}

/// Admit eight Leviathan sampled lanes (each lane's head read and target rows
/// are its own marker spike) and run one step; returns each lane's emitted
/// tokens, its naive RNG state after the step, the specs lent per verify, the
/// naive states before the step and the device draws the sampler owner made.
#[allow(clippy::type_complexity)]
fn device_step(
    device: bool,
    host_rows: bool,
    withhold: bool,
    k: usize,
) -> (
    Result<Vec<(Vec<u32>, u32)>, ProtocolError>,
    Vec<usize>,
    Vec<u32>,
    Vec<(u64, hipfire_runtime::lane_draw::LaneDraw)>,
) {
    let tags: Vec<u64> = (1..=8).collect();
    let mut root = root(false);
    let mut fan = DeviceFan {
        inner: fanout_with(
            tags.iter()
                .map(|&tag| script(tag, 0, &vec![tag as u32; k], &vec![tag as u32; k + 1]))
                .collect(),
            |_, _| {},
        ),
        host_rows,
        withhold,
        lent: Vec::new(),
        draws: Vec::new(),
    };
    let mut sampler = LaneSampler::new(SAMPLED_VOCAB);
    sampler.set_device_sample(device);
    let mut initial = Vec::new();
    for &tag in &tags {
        let a = root
            .admit_lane(LaneAdmission {
                epoch: epoch(tag),
                position: 10,
                seed: 9,
                eos: None,
                rng_seed: 1000 + tag,
                prompt: vec![1, 2, 3],
                spec_k: Some(k),
                head: HeadState::Aligned,
                copy: false,
                head_lag: Vec::new(),
            })
            .unwrap();
        fan.admit(&a).unwrap();
        sampler
            .admit(
                epoch(tag),
                &SpecRequestConfig {
                    temp: 1.0,
                    rng_seed: 1000 + tag,
                    ..SpecRequestConfig::default()
                },
                &[1, 2, 3],
                9,
            )
            .unwrap();
        initial.push(root.rng_fingerprint(epoch(tag)).unwrap().1);
    }
    let requests: Vec<LaneRequest> = tags.iter().map(|&tag| sampled_req(tag)).collect();
    let driven = drive_step(
        &mut root,
        &mut fan,
        None,
        &requests,
        &budgets(LabRoutes::Batched, k),
        &mut sampler,
    );
    let lanes = driven.map(|d| {
        d.output
            .lanes
            .iter()
            .zip(&tags)
            .map(|(lane, &tag)| {
                (
                    lane.emitted.clone(),
                    root.rng_fingerprint(epoch(tag)).unwrap().1,
                )
            })
            .collect()
    });
    let lent = fan.lent.clone();
    let draws = fan.draws.clone();
    fan.inner.finish();
    (lanes, lent, initial, draws)
}

#[test]
fn mesh_lane_ep_device_picks_decide_eight_sampled_lanes_without_row_reads() {
    for k in [2usize, 3] {
        // Device: the sampler owner returns no rows at all (a row read by the
        // root would abort the step), only the lent lanes' device draws.
        let (device, lent, initial, draws) = device_step(true, false, false, k);
        let device = device.expect("a device-drawn eight-lane step on two scripted ranks");
        assert_eq!(lent, vec![8], "K={k}: every sampled lane was lent a spec");
        assert_eq!(draws.len(), 8, "K={k}: one reference draw per lane");
        // The host Leviathan path over the same keyed rows and head reads: the
        // flag off lends nothing, the rows are read and the naive RNG stays put.
        let (host, host_lent, _, host_draws) = device_step(false, true, false, k);
        let host = host.expect("the host Leviathan step");
        assert_eq!(host_lent, vec![0], "K={k}: the flag off lends nothing");
        assert!(host_draws.is_empty(), "K={k}: the host path draws nothing");
        for (i, (emitted, rng)) in device.iter().enumerate() {
            let tag = i as u32 + 1;
            // Keyed spikes make the law a point mass on the lane's marker, so
            // the device and host paths emit the same window.
            assert_eq!(
                emitted, &host[i].0,
                "K={k}: lane {tag} device vs host window"
            );
            assert!(
                emitted.len() == k + 1 && emitted.iter().all(|&t| t == tag),
                "K={k}: lane {tag} emitted {emitted:?}"
            );
            // The root decided from the stand-in's own reference draw and took
            // its advanced state back.
            let (_, draw) = draws
                .iter()
                .find(|(t, _)| *t == u64::from(tag))
                .expect("a draw for every lane");
            assert_eq!(emitted, &draw.tokens, "K={k}: lane {tag} tokens");
            assert_eq!(*rng, draw.advanced_rng, "K={k}: lane {tag} advanced RNG");
            assert_ne!(*rng, initial[i], "K={k}: lane {tag} RNG never advanced");
            assert_eq!(
                host[i].1, initial[i],
                "K={k}: the host path left the naive RNG"
            );
        }
    }
}

#[test]
fn mesh_lane_ep_device_step_aborts_when_the_owner_withholds_the_lent_draws() {
    // The root lent specs; an owner that answers with greedy picks and no device
    // draws (and no rows) is a protocol violation: abort, never a silent
    // greedy decision.
    let (result, lent, _, draws) = device_step(true, false, true, 3);
    assert!(draws.is_empty(), "the owner made no draw");
    assert_eq!(lent, vec![8]);
    let error = result.expect_err("missing device draws must abort");
    assert!(matches!(error, ProtocolError::Aborted(_)), "{error}");
}
