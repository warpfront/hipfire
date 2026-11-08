// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native-MTP lanes on a Flash-Next EP mesh: the EP adapter of the mesh lane
//! protocol ([`crate::mesh_lane_protocol`], DESIGN §3–§5, §7).
//!
//! # Who does what
//!
//! * **Rank 0 is the root** ([`EpMeshLanes`] owns the [`MeshRoot`]): it alone
//!   chooses route, depth and proposal source, reads the head's draft tokens,
//!   reads the target picks and decides the accepted prefix. Nothing else
//!   chooses: a follower owns no RNG, no chooser and no cost model.
//! * **Every rank** is a [`LaneMeshExecutor`] ([`RankLane`] over a
//!   [`RankExec`]): it validates the root's messages on its own lane store,
//!   in the same order, and runs the head work of **the lanes it heads**
//!   ([`heads_lane`]). A replicated lane (`LanePlan::head_rank == None`: every
//!   lane of a window with fewer than two head lanes) is replayed by every
//!   rank with the same op lists, so every row gather (target and head) has a
//!   matched crossing on both ranks. An owned lane (`Some(rank)`: the X2 draft
//!   split, promoted by the root once a window carries two or more head lanes,
//!   split evenly) is drafted, caught up, probed and head-repaired by its owner
//!   alone: the owner runs the head MoE lane-local (full expert table, no
//!   crossing), the peer keeps only the target side of the lane (a head-free
//!   [`rank_view`]). A rank answers the ops of its owned lanes (and, if it is
//!   the sampler owner, the replicated ones): `proposals` carry exactly the
//!   answered ops, the root merges them in round order. Receipts (state
//!   positions, the operation ledger the rank actually executed, evidence)
//!   are per rank; `sampler` output stays the sampler owner's.
//! * Each phase is dispatched to every rank **concurrently**
//!   ([`Qwen4EpMesh::run`] over the existing [`RankPool`]): rank 0 runs on the
//!   caller's thread, peers on their workers. A phase that contains a
//!   collective (a head draft round, the verify forward) returns `Err` on any
//!   failure, which poisons the pool and aborts the row-gather barrier, so no
//!   rank is left waiting on a dead peer; the driver then aborts the root and
//!   every rank (one record, quarantine). A preflight rejection is *not* a
//!   failure: [`drive_step`] rejects the whole step before any collective.
//!
//! # The window on a rank
//!
//! `begin_window` validates the lane tables against the plan, maps capacity
//! (trunk and head arenas) and, for every native lane, takes the head ticket
//! and arms the GDN ring and takes the target ticket (a lane a peer owns takes
//! no head ticket here). `run_draft_round` runs
//! the root's ordered head `Draft` ops of the lanes the rank heads (one
//! deferred head step per op, or one batched call over them, then one
//! download; only the answering rank reads a proposal). `run_verify` runs the one
//! trunk forward over the sealed rows of every lane (`forward_lanes`, row
//! groups of whole lanes) and returns the executed ledger. `prepare_commit`
//! repairs the rank's own target and head to the root's decision with
//! [`prepare_native_window`] (tickets stay live); `publish_commit` consumes the
//! validated tickets ([`publish_native_window`]) and mirrors host metadata.
//! No fallible GPU operation remains after every rank prepared.
//!
//! Lanes prefill through the ordinary lane executor (a Prefill-only plan runs
//! identically on every rank: no chooser is involved); a lane enters the mesh
//! protocol when its prompt is done ([`EpMeshLanes::admit_lane`]).
//!
//! # Not covered here
//!
//! Copy proposals, forced advances and head-lag lanes are refused at
//! preflight (the root keeps them off a window by not offering them).
//!
//! # Sampled lanes
//!
//! A sampled lane (`SamplingMode::Leviathan`) is served by the root's
//! [`LaneSampler`] ([`EpMeshLanes::step_sampled`]). Rank 0 (the sampler owner)
//! copies the head's exact rescored candidates into every native
//! [`DraftProposal::read`] right after the draft finish of each op, before
//! another lane's op can overwrite them; the root draws every proposal from its
//! `q` on the lane's staged RNG.
//!
//! With `HIPFIRE_QWEN4_LANE_DEVICE_SAMPLE` on (the default on exact gfx1201) the
//! root lends the sampler owner each sampled lane's staged verify RNG, sampler
//! policy, penalty-window source and draft distributions in
//! [`VerifyPlan::samples`] ([`crate::mesh_lane_protocol::DeviceSampleSpec`]).
//! Right after the verify forward on the same stream, rank 0 draws every
//! sampled lane's window with ONE `lane_draws` launch ([`device_draws`]:
//! Leviathan acceptance under the shared device sampler's law, one D2H of ids,
//! accept counts and RNG for all lanes) and returns them in
//! [`SamplerOutput::device`]; the root decides from them
//! (`mesh_lane_protocol::device_accept`). No logits row crosses to the host
//! and no host scan runs.
//!
//! With the flag off (`=0`) rank 0 returns a [`RowHandle`] over its resident
//! verify logits in [`SamplerOutput::rows`] and the root draws every
//! accept/residual/bonus on the host. The handle is valid from the verify
//! until the step's prepare (or abort). The ranks only replay the committed
//! tokens, exactly as for greedy lanes.

use crate::bundle::{BundleError, Qwen4Bundle};
use crate::ep::{gather_tally, GatherTally, PeerMirrors};
use crate::ep_vocab_shard::{VocabExchange, SLOT_CAND, SLOT_LOGITS, SLOT_PARTIAL, SLOT_TOP1};
use crate::gpu_forward_lanes::{
    forward_lanes, row_group_count, LaneAllRowsRule, LaneOutputs, LaneRowKind, LaneRows, StageName,
    StageShare, VocabShardView,
};
use crate::lane::download_lane_rows;
use crate::lane::{LabRoutes, LaneSpec, LaneStatus, Qwen4LaneState, Qwen4LaneStore, StepParts};
use crate::lane_exec::pick_at;
use crate::mesh_lane_protocol::SamplingMode;
use crate::mesh_lane_protocol::{ep_draft_split_enabled, heads_lane};
use crate::mesh_lane_protocol::{
    AbortRecord, AcceptRule, CommitDecision, ControllerInterval, DevicePick, DraftOp,
    DraftProposal, DraftRound, EvidenceError, HeadAppend, HeadEnd, HeadState, LaneAdmission,
    LaneAdmit, LaneDecision, LaneKind, LaneMeshExecutor, LanePlan, LaneRequest, LaneWindowKey,
    MergedCostEvidence, MeshRoot, MeshStepKey, OpCounts, OpLedger, PhaseTiming, PreparedReceipt,
    ProposalSource, ProtocolError, RankCostEvidence, RankDraftResult, RankError, RankOwnership,
    RankReady, RankSession, RankVerifyResult, ReadyStatus, RootConfig, SamplerOutput, StagePolicy,
    StepOutput, StepPhase, TimedPhase, VerifyPlan, WindowBudgets, WindowStart,
};
use crate::mesh_rank_copy::{
    arm_lane_verify, begin_lane_window, catch_up_lane, head_ticket_point, is_probe_op,
    native_keep_cap, prepare_rank_lane, probe_outcome, publish_rank_lane, save_probe_state,
    settle_probe, take_head_ticket, HeadTicketPoint, RankPrepare,
};
use crate::mtp_gpu::S4Round;
use crate::mtp_lanes::lane_call;
use crate::mtp_spec::{
    draft_accept_estimate, HeadTicket, TargetTicket, DRAFT_ACCEPT_MAX, MTP_ROW_WORTH,
};
use crate::sampled_lane::{DraftRead, LaneSampler, RowBuf, RowHandle, TargetRowSource};
use hipfire_dispatch::pipeline::DraftShard;
use hipfire_dispatch::types::DispatchError;
use hipfire_runtime::lane_draw::{
    lane_draws, LaneDraw, LaneDrawArena, LaneDrawJob, LaneDrawStaged,
};
use hipfire_runtime::slot_batch::RequestEpoch;
use hipfire_runtime::spec::SpecRequestConfig;
use rdna_compute::tensor_ops::ArgmaxPairMerge;
use rdna_compute::{Gpu, GpuTensor};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

type R<T> = Result<T, String>;

fn rank_err(rank: usize, reason: impl Into<String>) -> RankError {
    RankError {
        rank,
        reason: reason.into(),
    }
}

// ---------------------------------------------------------------------------
// Rank-side window state
// ---------------------------------------------------------------------------

/// Diagnostic counters of one rank (not protocol evidence).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RankLaneStats {
    pub rank: usize,
    pub steps: u64,
    /// Windows whose target/head repair this rank prepared.
    pub native_windows: u64,
    /// Head tokens this rank's own head read differently from the root's
    /// kept token (numerical divergence between ranks; evidence only, the
    /// root's token is authoritative).
    pub draft_mismatches: u64,
    /// Emitted tokens that differ from this rank's own target pick at the
    /// same row (the same kind of evidence for the verify forward).
    pub pick_mismatches: u64,
}

/// One lane's window on this rank.
/// One lane's window on this rank. The device window itself (tickets, head
/// steps, the prepared repair) lives in the lane's `spec.window`
/// ([`crate::mesh_rank_copy`]); this is the rank-protocol side.
struct LaneWin {
    plan: LanePlan,
    /// Probability the draft prefix is accepted (the singleton's early stop;
    /// read on the root only).
    prefix: f32,
    /// Tokens this rank's own head read for kept draft steps.
    local_drafts: Vec<u32>,
    /// The sealed verify rows.
    rows: Vec<u32>,
    /// First decode row of the lane among the step's decode rows.
    row_begin: usize,
    /// This rank's own target picks (the root's are the decision's input).
    picks: Vec<u32>,
}

impl LaneWin {
    /// `own`'s window of `plan` ([`rank_view`]).
    fn new(plan: &LanePlan, own: &RankOwnership) -> Self {
        Self {
            plan: rank_view(own, plan).into_owned(),
            prefix: 1.0,
            local_drafts: Vec::new(),
            rows: Vec::new(),
            row_begin: 0,
            picks: Vec::new(),
        }
    }
}

/// This rank's view of a lane's plan. A lane whose head a peer owns
/// ([`heads_lane`] is false though the window touches the head) is head-free
/// here: no head arena mapping, ticket, catch-up, append or owed row (the same
/// treatment as a head-free lane, which already runs target-only). Every other
/// lane's plan is unchanged.
fn rank_view<'a>(own: &RankOwnership, plan: &'a LanePlan) -> Cow<'a, LanePlan> {
    if plan.head == HeadState::Free || heads_lane(own, plan) {
        Cow::Borrowed(plan)
    } else {
        Cow::Owned(LanePlan {
            head: HeadState::Free,
            catch_up: None,
            ..plan.clone()
        })
    }
}

/// The decision as this rank repairs it: a head-free view of a lane
/// ([`rank_view`]) carries no head schedule.
fn rank_decision<'a>(d: &'a LaneDecision, view: &LanePlan) -> Cow<'a, LaneDecision> {
    if view.head == HeadState::Free
        && (d.head_append != HeadAppend::default() || d.head_end != HeadEnd::Free)
    {
        Cow::Owned(LaneDecision {
            head_append: HeadAppend::default(),
            head_end: HeadEnd::Free,
            ..d.clone()
        })
    } else {
        Cow::Borrowed(d)
    }
}

/// Whether the rank that executes an op of `plan` answers it (reads its
/// proposal): the owner of an owned lane, the sampler owner for a replicated
/// one.
fn answers_op(plan: &LanePlan, sampler: bool) -> bool {
    plan.head_rank.is_some() || sampler
}

/// The sharded-MoE pass (S4, `HIPFIRE_QWEN4_DRAFT_SHARDED_MOE`) of a draft
/// round, decided from the root's whole round and the plan alone so every rank
/// decides alike (`rank` only names this rank's share): the switch and the S4
/// head tables are on, the round holds two to eight ops, every op is a lane one
/// of two ranks owns (X2), and the round is one batched head call
/// ([`crate::mtp_lanes::draft_round_batchable`] over the whole round). Any other
/// round (a lone op, replicated lanes) keeps its X2 / replicated path unchanged.
fn s4_round_for<'a>(
    bundle: &Qwen4Bundle,
    ops: &[DraftOp],
    plan_of: impl Fn(&LaneWindowKey) -> Option<&'a LanePlan>,
    rank: usize,
) -> R<Option<S4Round>> {
    if !crate::mtp_gpu::draft_sharded_moe_enabled()
        || !bundle.mtp_s4_ready()
        || !(2..=crate::mtp_gpu::MTP_DRAFT_LANES_MAX).contains(&ops.len())
    {
        return Ok(None);
    }
    let mut lanes: SmallVec<[(Option<usize>, bool); crate::mtp_gpu::MTP_DRAFT_LANES_MAX]> =
        SmallVec::new();
    for op in ops {
        let plan =
            plan_of(&op.key).ok_or_else(|| "draft op for a lane outside the plan".to_string())?;
        lanes.push((plan.head_rank, plan.wants_candidate_read()));
    }
    Ok(s4_round_shape(&lanes, rank, |any_read| {
        crate::mtp_lanes::draft_round_batchable(bundle, lanes.len(), any_read)
    }))
}

/// The host-only part of [`s4_round_for`]: `lanes` holds each op's
/// `(head_rank, wants_candidate_read)` in the root's op order. `None` unless the
/// round holds two to eight ops, every op has an X2 owner (`head_rank` 0 or 1;
/// a replicated lane, `None`, keeps the round on its unchanged path) and
/// `batchable(any_candidate_read)` holds. Otherwise `owners` carries op `s`'s
/// owner in bit `s`.
fn s4_round_shape(
    lanes: &[(Option<usize>, bool)],
    rank: usize,
    batchable: impl FnOnce(bool) -> bool,
) -> Option<S4Round> {
    if !(2..=crate::mtp_gpu::MTP_DRAFT_LANES_MAX).contains(&lanes.len()) {
        return None;
    }
    let (mut owners, mut any_read) = (0u8, false);
    for (slot, &(head_rank, wants_read)) in lanes.iter().enumerate() {
        match head_rank {
            Some(owner) if owner < 2 => owners |= (owner as u8) << slot,
            _ => return None,
        }
        any_read |= wants_read;
    }
    batchable(any_read).then_some(S4Round {
        g: lanes.len(),
        owners,
        rank,
    })
}

struct ExecStep {
    key: MeshStepKey,
    lanes: Vec<LaneWin>,
    draft_us: f64,
    verify_us: f64,
    prepare_us: f64,
    draft_tally: GatherTally,
    verify_tally: GatherTally,
    head_calls: usize,
    catch_up_rows: usize,
    verify: Option<VerifyPlan>,
    row_groups: usize,
    wide_groups: usize,
    totals: Option<(usize, usize, usize, usize)>,
    completed: StepPhase,
    unpriced: bool,
}

/// One EP rank's lane-protocol state: the embedded [`RankSession`] reducer,
/// the open window and the evidence of the last finished step. Lives next to
/// the rank's bundle (rank 0 in [`EpMeshLanes`], a peer in its
/// `PeerMirrors`); the bundle and `Gpu` arrive with each call
/// ([`RankLane`]).
pub struct RankExec {
    session: RankSession,
    rank: usize,
    step: Option<ExecStep>,
    last: Option<RankCostEvidence>,
    /// Native depths this rank has run: a first-seen depth (and the very
    /// first step: warm kernel binds) is unpriced.
    seen_depths: HashSet<usize>,
    stepped: bool,
    stats: RankLaneStats,
    /// A failure after the all-prepared boundary (publish is infallible by
    /// construction; this records the impossible case).
    fatal: Option<String>,
    abort_reason: Option<String>,
    /// Validity flag of the row source the last verify handed out.
    rows_valid: Option<Arc<AtomicBool>>,
    /// The vocabulary-shard exchange of this rank's mesh, when attached.
    vocab: Option<Arc<VocabExchange>>,
}

impl RankExec {
    pub fn new(own: RankOwnership) -> Self {
        Self {
            session: RankSession::new(own),
            rank: own.rank,
            step: None,
            last: None,
            seen_depths: HashSet::new(),
            stepped: false,
            stats: RankLaneStats {
                rank: own.rank,
                ..RankLaneStats::default()
            },
            fatal: None,
            abort_reason: None,
            rows_valid: None,
            vocab: None,
        }
    }

    /// This rank's vocabulary-shard exchange (`HIPFIRE_QWEN4_EP_VOCAB_SHARD`; `None`: the switch
    /// is off or the load declined it, so a sharded plan is refused).
    pub(crate) fn with_vocab(mut self, vocab: Option<Arc<VocabExchange>>) -> Self {
        self.vocab = vocab;
        self
    }

    pub fn stats(&self) -> RankLaneStats {
        self.stats
    }

    /// Why this rank was aborted (idempotent), if it was.
    pub fn abort_reason(&self) -> Option<&str> {
        self.abort_reason.as_deref()
    }

    /// Invalidate the row source of the last verify: its rows are only
    /// readable from the verify until the step's prepare (or abort).
    fn drop_rows(&mut self) {
        if let Some(valid) = self.rows_valid.take() {
            valid.store(false, Ordering::Release);
        }
    }

    fn take_fatal(&mut self) -> Option<String> {
        self.fatal.take()
    }
}

/// A rank's executor for one call: its [`RankExec`] with the rank's bundle
/// and device.
pub(crate) struct RankLane<'a> {
    pub exec: &'a mut RankExec,
    pub bundle: &'a mut Qwen4Bundle,
    pub gpu: &'a mut Gpu,
}

/// Run `f` with the lane store taken out of the bundle so it may borrow the
/// bundle (weights, formats, forward) next to the store's lanes. The store is
/// back in place on every return path.
fn with_store<T>(
    bundle: &mut Qwen4Bundle,
    f: impl FnOnce(&mut Qwen4LaneStore, &mut Qwen4Bundle) -> R<T>,
) -> R<T> {
    let mut store = bundle
        .lanes
        .take()
        .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
    let result = f(&mut store, bundle);
    bundle.lanes = Some(store);
    result
}

fn lane_mut<'a>(store: &'a mut Qwen4LaneStore, epoch: &RequestEpoch) -> R<&'a mut Qwen4LaneState> {
    store
        .lane_mut(epoch)
        .ok_or_else(|| format!("unknown or stale epoch {epoch:?}"))
}

/// Release a lane's window tickets unconsumed-by-effect: the state is
/// unchanged since the snapshot, so committing a valid ticket only retires it.
/// Drop a lane's pending window: release its tickets (the state is unchanged
/// since the snapshots, so committing a valid ticket only retires it) and
/// disarm the ring. Used for a rejected or aborted window.
fn clear_window(bundle: &mut Qwen4Bundle, lane: &mut Qwen4LaneState) {
    let Qwen4LaneState { state, spec, .. } = lane;
    state.row_capture_armed = false;
    let Some(spec) = spec.as_mut() else {
        return;
    };
    let LaneSpec {
        head,
        draft,
        window,
        ..
    } = spec;
    let Some(w) = window.take() else {
        return;
    };
    if let Some(TargetTicket::Local(ticket)) = w.target_ticket {
        let _ = bundle.with_lane(state, None, |b| {
            if b.validate_commit(ticket).is_ok() {
                b.commit_validated(ticket);
            }
        });
    }
    if let Some(HeadTicket::Local(ticket)) = w.head_ticket {
        let _ = bundle.with_lane(state, Some((head, draft)), |b| {
            if b.mtp_validate_commit(ticket).is_ok() {
                b.mtp_commit_validated(ticket);
            }
        });
    }
}

/// What one draft op read on a rank.
struct OpRead {
    token: u32,
    margin: f32,
    /// Draft steps in effect before this op (`op.position - window.position`).
    j: usize,
    /// The confirmation probe abandoned its head step (a hit, or a miss with
    /// no native route).
    abandoned: bool,
    /// The head's rescored candidates (rank 0, native Leviathan lane only).
    read: Option<DraftRead>,
    /// FNV-1a of the op's `draft_top1` slot (lab draft trace).
    #[cfg(feature = "lab")]
    top1_fnv: u64,
}

/// The sampler owner's resident verify logits, read one row at a time by the
/// root's sampled rule ([`TargetRowSource`]). It reads rank 0's device from
/// the root's thread (rank 0 runs on the caller's thread) between the verify
/// and the prepare phase of one step.
struct EpRowSource {
    /// SAFETY: the rank's `Gpu` outlives every use of the handle: the handle
    /// is only readable while `valid` is set, and `valid` is cleared at the
    /// start of this rank's next verify, prepare, publish and abort, all of
    /// which run inside the `EpMeshLanes::step` call that borrows the `Gpu`.
    gpu: *const Gpu,
    /// A view of the step's `n_dec * vocab` logits rows.
    logits: GpuTensor,
    vocab: usize,
    /// `(window key, first row, rows)` per lane in plan order.
    lanes: Vec<(LaneWindowKey, usize, usize)>,
    valid: Arc<AtomicBool>,
}

// SAFETY: the raw `Gpu` pointer and the device view are only dereferenced on
// the root's thread (the thread that runs rank 0), while the step is in
// flight (see `gpu`).
unsafe impl Send for EpRowSource {}

impl TargetRowSource for EpRowSource {
    fn row_logits(
        &mut self,
        key: &LaneWindowKey,
        row: usize,
        host: &mut RowBuf,
    ) -> Result<(), String> {
        if !self.valid.load(Ordering::Acquire) {
            return Err("the step's target rows are no longer readable".to_string());
        }
        let &(_, first, rows) = self
            .lanes
            .iter()
            .find(|(k, _, _)| k == key)
            .ok_or_else(|| format!("no verify rows for lane window {key:?}"))?;
        if row >= rows {
            return Err(format!("verify row {row} of a {rows}-row window"));
        }
        // SAFETY: see `gpu`.
        let gpu = unsafe { &*self.gpu };
        download_lane_rows(
            gpu,
            &self.logits,
            first + row,
            1,
            self.vocab,
            host.scratch_mut(),
        )?;
        host.set_len(self.vocab);
        Ok(())
    }
}

/// What one rank's verify forward returned.
struct VerifyOut {
    /// The argmax pick at every row, per lane.
    picks: Vec<Vec<u32>>,
    /// Each lane's resulting target position.
    ends: Vec<usize>,
    /// A view of the step's `n_dec * vocab` decode logits rows.
    logits: GpuTensor,
    /// The device draws of the lanes the root lent a spec (sampler owner only).
    device: Vec<Option<DevicePick>>,
}

impl RankLane<'_> {
    /// The root's lazy access to this rank's resident verify logits.
    fn rows_handle(&mut self, plan: &VerifyPlan, logits: GpuTensor) -> RowHandle {
        let valid = Arc::new(AtomicBool::new(true));
        self.exec.rows_valid = Some(valid.clone());
        let mut first = 0usize;
        let lanes = plan
            .lanes
            .iter()
            .map(|l| {
                let entry = (l.key, first, l.rows.len());
                first += l.rows.len();
                entry
            })
            .collect();
        RowHandle::new(EpRowSource {
            gpu: &*self.gpu as *const Gpu,
            logits,
            vocab: self.bundle.lane_vocab(),
            lanes,
            valid,
        })
    }

    fn rank(&self) -> usize {
        self.exec.rank
    }

    fn own(&self) -> RankOwnership {
        self.exec.session.ownership()
    }

    /// The host-and-mapping checks of a plan and the device half of every
    /// lane's `begin_window` ([`begin_lane_window`]: the pending window, the
    /// head's position, arenas and the head ticket where the plan wants it).
    /// Nothing here moves lane state beyond mapping and tickets.
    fn preflight(&mut self, start: &WindowStart) -> R<()> {
        // A mesh window runs draft rounds, the verify and the repair on the
        // resident bank over whole donor layers: an open borrowed-bank epoch
        // (EP reserve-sharing load) refuses the window.
        crate::ep_reserve::mesh_window_epoch_check(self.bundle.ep_reserve_epoch_open())?;
        // The all-rows MoE rule the root sealed the plan's exchanges with is this rank's own
        // (every gather count of the step follows from it).
        let rule = all_rows_rule_of(&*self.gpu, &*self.bundle)?;
        if rule != start.stage.all_rows {
            return Err(format!(
                "rank {} all-rows MoE rule {rule:?} differs from the root's {:?}",
                self.exec.rank, start.stage.all_rows
            ));
        }
        let own = self.own();
        let Self { bundle, gpu, .. } = self;
        let (spec_k, rows_cap, max_lanes) = {
            let store = bundle
                .lanes
                .as_ref()
                .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
            (
                store.spec().map(|spec| spec.k),
                store.prefill_logits_row(),
                store.max_lanes(),
            )
        };
        if start.lanes.len() > max_lanes {
            return Err(format!(
                "{} lanes exceed the staged {max_lanes}",
                start.lanes.len()
            ));
        }
        let total: usize = start.lanes.iter().map(LanePlan::max_rows).sum();
        if total > rows_cap {
            return Err(format!(
                "{total} verify rows exceed the store's {rows_cap} step rows"
            ));
        }
        with_store(bundle, |store, bundle| {
            for plan in &start.lanes {
                let epoch = plan.key.request_epoch;
                let lane = lane_mut(store, &epoch)?;
                if lane.host.status != LaneStatus::Decoding {
                    return Err(format!("{epoch:?} is {:?}, not Decoding", lane.host.status));
                }
                if lane.host.pending_seed != Some(plan.seed) {
                    return Err(format!(
                        "{epoch:?}: pending seed {:?} != planned {}",
                        lane.host.pending_seed, plan.seed
                    ));
                }
                if lane.state.position != plan.position {
                    return Err(format!(
                        "{epoch:?}: lane position {} != planned {}",
                        lane.state.position, plan.position
                    ));
                }
                let end = plan
                    .position
                    .checked_add(plan.max_rows())
                    .ok_or_else(|| "position overflow".to_string())?;
                if end > lane.state.max_seq_len {
                    return Err(format!(
                        "{epoch:?}: end position {end} exceeds the context {}",
                        lane.state.max_seq_len
                    ));
                }
                if lane.spec.is_none() {
                    if plan.source != ProposalSource::None
                        || plan.head != HeadState::Free
                        || matches!(plan.kind, LaneKind::Forced(_))
                    {
                        return Err(format!("{epoch:?}: an AR lane cannot run a head window"));
                    }
                } else {
                    if spec_k.is_none() {
                        return Err(format!("{epoch:?}: lane has MTP state in an AR store"));
                    }
                    if plan.max_rows() > 1 && lane.state.row_capture_rows() < plan.max_rows() {
                        return Err(format!(
                            "{epoch:?}: ring holds {} rows, the window needs {} (stage the store with the copy budget for copy windows)",
                            lane.state.row_capture_rows(),
                            plan.max_rows()
                        ));
                    }
                }
                lane.state.ensure_mapped_capacity(gpu, end).map_err(|e| {
                    // A mapping refusal is a capacity condition (the page budget
                    // normally keeps it from happening), not diverged state:
                    // `lane_error_wire` answers it with a retryable overload.
                    format!(
                        "{}{end} positions for {epoch:?}: {e}",
                        crate::lane_exec::PROVISION_MAP_REFUSED
                    )
                })?;
                let view = rank_view(&own, plan);
                begin_lane_window(bundle, gpu, lane, &view)?;
            }
            Ok(())
        })
    }

    /// Unwind every lane window of `plan_lanes` (a rejected or aborted step).
    fn clear_windows(&mut self, lanes: &[LanePlan]) {
        let Self { bundle, .. } = self;
        let _ = with_store(bundle, |store, bundle| {
            for plan in lanes {
                if let Some(lane) = store.lane_mut(&plan.key.request_epoch) {
                    clear_window(bundle, lane);
                }
            }
            Ok(())
        });
    }

    /// One head round on this rank: the plan's catch-ups (round 0, before any
    /// Draft op), then every op's deferred head step in the root's order, one
    /// download for all of them and the head's own parse of each. Returns the
    /// catch-up rows appended and the reads per op.
    fn draft_round_device(
        &mut self,
        round: &DraftRound,
        sampler: bool,
        s4: Option<S4Round>,
    ) -> R<(usize, Vec<OpRead>)> {
        let Self {
            exec, bundle, gpu, ..
        } = self;
        let step = exec
            .step
            .as_ref()
            .ok_or_else(|| "draft round without a begun window".to_string())?;
        let plan_of = |key: &LaneWindowKey| -> R<&LanePlan> {
            step.lanes
                .iter()
                .find(|w| w.plan.key == *key)
                .map(|w| &w.plan)
                .ok_or_else(|| "draft op for a lane outside the plan".to_string())
        };
        // A vocabulary-sharded head round (`DraftRound::vocab_shard`): every draft of the round
        // splits its ranked rows with the peer rank ([`draft_shard`]). The root decided it from
        // the plan alone, so both ranks make the same number of exchange calls in the same order
        // (the same replicated ops, the same batched / serial branch).
        let shard_round = round.vocab_shard && !round.ops.is_empty();
        if shard_round {
            let shard = draft_shard(exec)?;
            bundle
                .mtp_set_draft_shard(Some(shard))
                .map_err(|e| e.to_string())?;
        }
        let result = with_store(bundle, |store, bundle| {
            // Round 0: owed-row catch-ups, then the head tickets that follow one.
            let mut catch_rows = 0usize;
            if !round.catch_up.is_empty() {
                let epochs: Vec<RequestEpoch> = round
                    .catch_up
                    .iter()
                    .map(|(k, _)| k.request_epoch)
                    .collect();
                let (mut lanes, _) = store.draft_parts(&epochs)?;
                for ((key, catch_up), lane) in round.catch_up.iter().zip(lanes.iter_mut()) {
                    catch_rows += catch_up_lane(bundle, gpu, &mut **lane, catch_up)?;
                    if head_ticket_point(plan_of(key)?) == HeadTicketPoint::AfterCatchUp {
                        take_head_ticket(bundle, gpu, &mut **lane)?;
                    }
                }
            }
            if round.ops.is_empty() {
                // A sharded-MoE pass (S4) is a collective: a rank that owns none of
                // its rows still runs the activation all-gather and the global MoE
                // call, in lockstep with the peer.
                if let Some(s4) = s4 {
                    bundle
                        .mtp_set_s4_round(Some(s4))
                        .map_err(|e| format!("sharded MoE mode: {e}"))?;
                    let ran = bundle
                        .mtp_s4_participate(gpu)
                        .map_err(|e| format!("sharded MoE participant: {e}"));
                    let off = bundle
                        .mtp_set_s4_round(None)
                        .map_err(|e| format!("sharded MoE mode off: {e}"));
                    ran?;
                    off?;
                }
                return Ok((catch_rows, Vec::new()));
            }
            // The round's ops are every op this rank executes: all owned by this
            // rank's lanes, or all replicated (the protocol refuses a mix). An
            // owned round runs the head lane-local (the full-table MoE runtime:
            // no crossing, no `moe_steps` advance); the mode is restored on every
            // path out.
            let lane_local = s4.is_none()
                && round.ops.iter().try_fold(false, |any, op| {
                    Ok::<_, String>(any || plan_of(&op.key)?.head_rank.is_some())
                })?;
            if lane_local {
                bundle
                    .mtp_set_lane_local(true)
                    .map_err(|e| format!("head lane-local mode: {e}"))?;
            }
            if let Some(s4) = s4 {
                bundle
                    .mtp_set_s4_round(Some(s4))
                    .map_err(|e| format!("sharded MoE mode: {e}"))?;
            }
            let ran = (|| -> R<Vec<OpRead>> {
                let epochs: Vec<RequestEpoch> =
                    round.ops.iter().map(|o| o.key.request_epoch).collect();
                let (mut lanes, spec_store) = store.draft_parts(&epochs)?;
                let spec_store = spec_store
                    .ok_or_else(|| "head draft in a store without MTP lanes".to_string())?;
                let slot_bytes = Gpu::TOPK8_OUT_BYTES;
                let bytes = round.ops.len() * slot_bytes;
                if bytes > spec_store.draft_top1_host.len() {
                    return Err(format!(
                        "{} draft ops exceed the {} draft slots",
                        round.ops.len(),
                        spec_store.draft_top1_host.len() / slot_bytes
                    ));
                }
                // Lab settling test of the draft-history mask (PARITY.md 4.3): head
                // arena rows beyond each lane's valid lengths are overwritten
                // (`HIPFIRE_LAB_MTP_STALE_FILL`); every output must stay identical.
                #[cfg(feature = "lab")]
                crate::lab_draft_rounds::fill_stale(gpu, &lanes)?;
                // Lab oracle of the batched head round: compare this round's lanes'
                // serial and batched head steps on scratch twins of their states
                // (no effect on the lanes; armed by env only, `lab` builds only).
                #[cfg(feature = "lab")]
                if crate::lab_draft_rounds::probe_enabled()
                    && s4.is_none()
                    && round.ops.len() >= 2
                    && !round.ops.iter().try_fold(false, |any, op| {
                        Ok::<_, String>(any || is_probe_op(plan_of(&op.key)?, op.position))
                    })?
                {
                    let ops: Vec<(u32, usize)> =
                        round.ops.iter().map(|op| (op.input, op.position)).collect();
                    crate::lab_draft_rounds::probe_round(
                        bundle,
                        gpu,
                        &mut lanes,
                        &ops,
                        exec.rank,
                        round.round,
                    )?;
                }
                // The draft head's logit row is one shared buffer: a sampled lane's
                // read of a head without re-scoring is that whole row, so with
                // more than one op in the round each op must be enqueued, finished
                // and read before the next op's enqueue overwrites it. Candidate
                // reads are host-resident after each finish and keep the one
                // batched download. Ops of a round are distinct lanes, so the
                // order of finishing does not move any lane's state.
                let want_read_any = round.ops.iter().try_fold(false, |any, op| {
                    let plan = plan_of(&op.key)?;
                    Ok::<_, String>(
                        any || (answers_op(plan, sampler) && plan.wants_candidate_read()),
                    )
                })?;
                // One batched head call per round (`HIPFIRE_QWEN4_MTP_DRAFT_ROUNDS=1`):
                // decided from the root's plan alone over the ops this rank executes
                // (a replicated round: every rank takes the same branch, the head's
                // gather parity advances once per batched call).
                let any_distribution_read = round.ops.iter().try_fold(false, |any, op| {
                    let plan = plan_of(&op.key)?;
                    Ok::<_, String>(any || plan.wants_candidate_read())
                })?;
                let batched = s4.is_some()
                    || crate::mtp_lanes::draft_round_batchable(
                        bundle,
                        round.ops.len(),
                        any_distribution_read,
                    );
                let chunks =
                    draft_read_plan(want_read_any, bundle.mtp_draft_rescored(), round.ops.len())
                        .chunks(round.ops.len());
                let mut reads = Vec::with_capacity(round.ops.len());
                for chunk in chunks {
                    if batched {
                        let mut inputs = Vec::with_capacity(round.ops.len());
                        for (slot, op) in round.ops.iter().enumerate() {
                            if is_probe_op(plan_of(&op.key)?, op.position) {
                                save_probe_state(&mut *lanes[slot]);
                            }
                            inputs.push((op.input, op.position));
                        }
                        crate::mtp_lanes::enqueue_batched_draft(
                            bundle,
                            gpu,
                            &mut lanes,
                            &inputs,
                            &spec_store.draft_top1,
                        )?;
                    } else {
                        for (slot, op) in round
                            .ops
                            .iter()
                            .enumerate()
                            .skip(chunk.start)
                            .take(chunk.len())
                        {
                            // The confirmation probe's Draft is abandoned on a hit: save
                            // the head's request state it is about to move.
                            if is_probe_op(plan_of(&op.key)?, op.position) {
                                save_probe_state(&mut *lanes[slot]);
                            }
                            let Qwen4LaneState { state, spec, .. } = &mut *lanes[slot];
                            let spec = spec
                                .as_mut()
                                .ok_or_else(|| "drafting lane lost its MTP state".to_string())?;
                            let LaneSpec {
                                head,
                                draft,
                                pending_hidden,
                                window,
                                ..
                            } = spec;
                            let window_position = window
                                .as_ref()
                                .ok_or_else(|| "draft op without a pending window".to_string())?
                                .position;
                            let top1 = spec_store
                                .draft_top1
                                .sub_offset(slot * slot_bytes, slot_bytes);
                            let j = op.position - window_position;
                            let hidden = (j == 0).then_some(&*pending_hidden);
                            lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
                                b.mtp_forward_token_deferred(
                                    gpu,
                                    op.input,
                                    hidden,
                                    op.position,
                                    j == 0,
                                    &top1,
                                )
                            }))
                            .map_err(|e| format!("draft step {j}: {e}"))?;
                        }
                    }
                    let (from, to) = (chunk.start * slot_bytes, chunk.end * slot_bytes);
                    let chunk_top1 = spec_store.draft_top1.sub_offset(from, to - from);
                    gpu.dtoh_stream_ordered(
                        &mut spec_store.draft_top1_host[from..to],
                        &chunk_top1.buf,
                    )
                    .map_err(|e| format!("draft download: {e}"))?;
                    gpu.take_device_wait_error()
                        .map_err(|e| format!("draft download: {e}"))?;
                    for (slot, op) in round
                        .ops
                        .iter()
                        .enumerate()
                        .skip(chunk.start)
                        .take(chunk.len())
                    {
                        let top =
                            &spec_store.draft_top1_host[slot * slot_bytes..(slot + 1) * slot_bytes];
                        let plan = plan_of(&op.key)?;
                        let want_read = answers_op(plan, sampler) && plan.wants_candidate_read();
                        let (token, margin, read) = if batched {
                            let (token, margin) = crate::mtp_lanes::finish_batched_draft(
                                bundle,
                                &mut *lanes[slot],
                                top,
                            )
                            .map_err(|e| format!("draft finish: {e}"))?;
                            let read = if want_read {
                                Some(
                                    bundle
                                        .mtp_draft_read(gpu)
                                        .map_err(|e| format!("draft finish: {e}"))?,
                                )
                            } else {
                                None
                            };
                            (token, margin, read)
                        } else {
                            let Qwen4LaneState { state, spec, .. } = &mut *lanes[slot];
                            let spec = spec
                                .as_mut()
                                .ok_or_else(|| "drafting lane lost its MTP state".to_string())?;
                            let LaneSpec { head, draft, .. } = spec;
                            lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
                                let token = b.mtp_draft_finish(top)?;
                                let read = if want_read {
                                    Some(b.mtp_draft_read(gpu)?)
                                } else {
                                    None
                                };
                                Ok::<_, BundleError>((token, b.mtp_draft_margin(), read))
                            }))
                            .map_err(|e| format!("draft finish: {e}"))?
                        };
                        let j = {
                            let w = lanes[slot]
                                .spec
                                .as_ref()
                                .and_then(|s| s.window.as_ref())
                                .ok_or_else(|| "draft op without a pending window".to_string())?;
                            op.position - w.position
                        };
                        // The probe: the head's argmax at the seed against the first
                        // candidate (the root's own rule, evaluated on this rank's
                        // read; the verify rows the root seals validate it). A hit,
                        // or a miss with no native route, abandons the probe's step.
                        let mut abandoned = false;
                        if is_probe_op(plan, op.position) {
                            let outcome =
                                probe_outcome(&plan.copy_drafts, token, plan.native_depth);
                            settle_probe(bundle, gpu, &mut *lanes[slot], outcome)?;
                            abandoned = outcome.restores_head();
                        }
                        if !abandoned {
                            if let Some(w) =
                                lanes[slot].spec.as_mut().and_then(|s| s.window.as_mut())
                            {
                                w.steps += 1;
                            }
                        }
                        reads.push(OpRead {
                            token,
                            margin,
                            j,
                            abandoned,
                            read: if abandoned { None } else { read },
                            #[cfg(feature = "lab")]
                            top1_fnv: lab_trace::fnv1a(top),
                        });
                    }
                }
                Ok(reads)
            })();
            if s4.is_some() {
                let off = bundle
                    .mtp_set_s4_round(None)
                    .map_err(|e| format!("sharded MoE mode off: {e}"));
                let reads = ran?;
                off?;
                return Ok((catch_rows, reads));
            }
            if lane_local {
                let off = bundle
                    .mtp_set_lane_local(false)
                    .map_err(|e| format!("head lane-local mode off: {e}"));
                let reads = ran?;
                off?;
                return Ok((catch_rows, reads));
            }
            ran.map(|reads| (catch_rows, reads))
        });
        if shard_round {
            // Removed on every path out of the round (an error aborts the mesh anyway).
            let removed = bundle.mtp_set_draft_shard(None).map_err(|e| e.to_string());
            return result.and_then(|ok| removed.map(|()| ok));
        }
        result
    }

    /// The one trunk forward over every lane's sealed rows (arming the ring and
    /// taking the target ticket of every multi-row lane first); the picks per
    /// lane and each lane's resulting target position. On the sampler owner,
    /// right after the forward on the same stream, every lane the root lent a
    /// [`crate::mesh_lane_protocol::DeviceSampleSpec`] is drawn on device
    /// ([`device_draws`]): only ids, accept counts and RNG states come back.
    ///
    /// With [`VerifyPlan::vocab_shard`] every rank runs only its half of the LM head
    /// (`LaneOutputs::vocab_shard`) and the sampler owner merges the two halves into the whole
    /// head's picks and draws ([`sharded_tail`]; the ordering and its reuse-safety argument are
    /// in [`crate::ep_vocab_shard`]). The non-sampler rank's picks are then shard-local ids.
    fn verify_device(&mut self, plan: &VerifyPlan) -> R<VerifyOut> {
        let sampler = self.own().sampler;
        let rank = self.rank();
        // A sharded plan runs only where the mesh attached its exchange (the rank refuses a plan
        // it cannot run: never a fallback to the whole head, the ranks' halves would not meet).
        let shard_exchange: Option<Arc<VocabExchange>> = if plan.vocab_shard {
            Some(self.exec.vocab.clone().ok_or_else(|| {
                "a vocabulary-sharded verify on a rank without the shard exchange".to_string()
            })?)
        } else {
            None
        };
        let Self { bundle, gpu, .. } = self;
        let vocab = bundle.lane_vocab();
        let epochs: Vec<RequestEpoch> = plan.lanes.iter().map(|l| l.key.request_epoch).collect();
        with_store(bundle, |store, bundle| {
            let StepParts {
                mut lanes,
                logits,
                top1,
                policy,
                top1_host,
                rows_cap,
                sample_arena,
                spec,
                ..
            } = store.step_parts(&epochs)?;
            let hidden: Option<&GpuTensor> = spec.as_deref().map(|spec| &spec.hidden);
            let n_dec = plan.total_rows();
            if n_dec > rows_cap {
                return Err(format!(
                    "{n_dec} decode rows exceed the step buffers ({rows_cap})"
                ));
            }
            // A vocabulary-sharded verify (`VerifyPlan::vocab_shard`): this rank's head buffers.
            let shard = match &shard_exchange {
                Some(exchange) => Some(shard_head(exchange, rank, sampler, n_dec, logits, top1)?),
                None => None,
            };
            for (vl, lane) in plan.lanes.iter().zip(lanes.iter_mut()) {
                if vl.rows.len() > 1 && lane.spec.is_some() {
                    // The sealed rows are the window's proposals (the root's
                    // seal, after any draft truncation / probe outcome).
                    if let Some(w) = lane.spec.as_mut().and_then(|s| s.window.as_mut()) {
                        w.drafts = vl.rows[1..].to_vec();
                    }
                    arm_lane_verify(bundle, gpu, &mut **lane, vl)?;
                }
            }
            let forwarded = {
                let mut rows: Vec<LaneRows<'_>> = Vec::with_capacity(lanes.len());
                for (lane, v) in lanes.iter_mut().zip(&plan.lanes) {
                    rows.push(LaneRows {
                        state: &mut lane.state,
                        tokens: &v.rows,
                        wide: v.wide,
                        exact: v.exact,
                        kind: if v.rows.len() == 1 {
                            LaneRowKind::Ar
                        } else {
                            LaneRowKind::Verify {
                                draft_len: v.rows.len() - 1,
                            }
                        },
                    });
                }
                let full_logits = logits.sub_offset(0, n_dec * vocab);
                let full_top1 = top1.sub_offset(0, n_dec * std::mem::size_of::<i32>());
                let out = LaneOutputs {
                    logits: shard.as_ref().map_or(&full_logits, |s| &s.logits),
                    top1: shard.as_ref().map_or(&full_top1, |s| &s.top1),
                    wide_hidden: hidden,
                    vocab_shard: shard.as_ref().map(|s| s.view),
                };
                forward_lanes(gpu, bundle, &mut rows, &out, &policy).map_err(|e| e.to_string())
            };
            // The ring is disarmed on every path out of the forward.
            for lane in lanes.iter_mut() {
                lane.state.row_capture_armed = false;
            }
            forwarded?;
            // The sampled lanes' device draws, enqueued right behind the
            // forward on its stream (the sampler owner only); a vocabulary-
            // sharded verify instead runs its select / exchange / merge
            // ([`sharded_tail`]) on every rank.
            let device = match &shard {
                Some(s) => sharded_tail(gpu, sample_arena, plan, s, sampler)?,
                None if sampler => device_draws(gpu, sample_arena, plan, logits, vocab)?,
                None => vec![None; plan.lanes.len()],
            };
            // The top-1 ids the host reads: the lane store's (the whole head's, or the sampler
            // owner's merged ids), or a shard-local rank's own (evidence only).
            let top1_src: &GpuTensor = match &shard {
                Some(s) => &s.top1,
                None => top1,
            };
            gpu.dtoh_stream_ordered(
                &mut top1_host[..n_dec * std::mem::size_of::<i32>()],
                &top1_src.buf,
            )
            .map_err(|e| e.to_string())?;
            gpu.take_device_wait_error().map_err(|e| e.to_string())?;
            // The window's forward is retired on this rank: the D2 expert tail (when installed)
            // judges it and rewrites its decode tables before the next window launches.
            if let Some(gather) = bundle
                .execution
                .as_ref()
                .and_then(crate::gpu_forward::Qwen4GpuForward::mtp_ep_gather)
            {
                hipfire_dispatch::families::moe::MoeRowGather::decode_step_end(gather.hook, gpu);
            }
            let mut picks = Vec::with_capacity(plan.lanes.len());
            let mut first = 0usize;
            for v in &plan.lanes {
                picks.push(
                    (0..v.rows.len())
                        .map(|j| pick_at(top1_host, first + j))
                        .collect(),
                );
                first += v.rows.len();
            }
            let ends = lanes.iter().map(|lane| lane.state.position).collect();
            // A view of the step's decode logits rows (the root's host sampled
            // rule reads single rows from it before the prepare phase).
            Ok(VerifyOut {
                picks,
                ends,
                logits: logits.sub_offset(0, n_dec * vocab),
                device,
            })
        })
    }

    /// Repair every lane to the decision ([`prepare_rank_lane`]: a window's
    /// target rollback and head rebuild with its tickets kept live, a row's
    /// head append or owed row, a target-only window's rollback) and store the
    /// prepared window in the lane. Returns the head rows appended.
    fn prepare_device(&mut self, decision: &CommitDecision) -> R<usize> {
        let Self {
            exec, bundle, gpu, ..
        } = self;
        let step = exec
            .step
            .as_ref()
            .ok_or_else(|| "prepare without a begun window".to_string())?;
        let verify = step
            .verify
            .as_ref()
            .ok_or_else(|| "prepare: the verify was never sealed".to_string())?;
        // A non-sampler rank of a vocabulary-sharded verify holds shard-local picks: its greedy
        // windows replay the root's decision as a sampled window does.
        let shard_local_picks = verify.vocab_shard && !exec.session.ownership().sampler;
        let epochs: Vec<RequestEpoch> = verify.lanes.iter().map(|l| l.key.request_epoch).collect();
        let mut head_rows = step.catch_up_rows;
        with_store(bundle, |store, bundle| {
            let spec_config = store.spec();
            let skip_rollback = store.skip_rollback();
            let StepParts {
                mut lanes, spec, ..
            } = store.step_parts(&epochs)?;
            let commit_fill = crate::mtp_lanes::lane_commit_fill_enabled(gpu);
            let (hidden, mut fill) = match spec {
                Some(s) => (
                    Some(&s.hidden),
                    s.append_scratch.as_mut().filter(|_| commit_fill),
                ),
                None => (None, None),
            };
            for (i, (d, lane)) in decision.lanes.iter().zip(lanes.iter_mut()).enumerate() {
                if lane.spec.is_none() {
                    continue;
                }
                let win = &step.lanes[i];
                // A lane a peer owns repairs target-only here (no head schedule).
                let local_decision = rank_decision(d, &win.plan);
                let config = spec_config
                    .ok_or_else(|| "MTP lane in a store without MTP lanes".to_string())?;
                let hidden =
                    hidden.ok_or_else(|| "MTP lane without a hidden capture".to_string())?;
                let prepared = prepare_rank_lane(
                    bundle,
                    gpu,
                    &mut **lane,
                    RankPrepare {
                        plan: &win.plan,
                        decision: &local_decision,
                        verify: &verify.lanes[i],
                        config,
                        hidden,
                        row_begin: win.row_begin,
                        picks: &win.picks,
                        shard_local_picks,
                        skip_rollback,
                        fill: fill.as_deref_mut(),
                    },
                )?;
                head_rows += prepared.head_rows;
                if let Some(w) = lane.spec.as_mut().and_then(|s| s.window.as_mut()) {
                    w.prepared = Some(prepared.lane);
                }
            }
            Ok(head_rows)
        })
    }
}

/// How one draft round's ops are enqueued and read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DraftReadPlan {
    /// Every op is enqueued, then one download and every op finished and read
    /// in order (nothing read depends on a buffer a later op overwrites).
    Batched,
    /// Each op is enqueued, downloaded, finished and read before the next one
    /// is enqueued: a sampled lane's read of a head without re-scoring is the
    /// whole shared draft logit row ([`crate::sampled_lane::DraftRead::FullRow`]),
    /// which the next op's step overwrites.
    Sequential,
}

/// The read plan of a round of `ops` ops: sequential only when some op reads
/// a distribution (`want_read_any`), the head does not re-score (its read is
/// the shared full row) and the round has more than one op.
fn draft_read_plan(want_read_any: bool, rescored: bool, ops: usize) -> DraftReadPlan {
    if want_read_any && !rescored && ops > 1 {
        DraftReadPlan::Sequential
    } else {
        DraftReadPlan::Batched
    }
}

impl DraftReadPlan {
    /// The op ranges of a round of `ops` ops, in execution order.
    fn chunks(self, ops: usize) -> Vec<std::ops::Range<usize>> {
        match self {
            Self::Batched => vec![0..ops],
            Self::Sequential => (0..ops).map(|i| i..i + 1).collect(),
        }
    }
}

/// The all-rows MoE rule of `bundle`'s forward on `gpu` under its lane store's stage policy:
/// what the root seals its ledger with (rank 0's) and every rank checks itself against.
fn all_rows_rule_of(gpu: &Gpu, bundle: &Qwen4Bundle) -> R<LaneAllRowsRule> {
    let store = bundle
        .lanes()
        .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
    let forward = bundle
        .execution
        .as_ref()
        .ok_or_else(|| "Qwen4 forward resources are not attached".to_string())?;
    LaneAllRowsRule::for_forward(gpu, forward, &store.stage_policy()).map_err(|e| e.to_string())
}

/// MoE layers whose expert-row gather the sealed ledger plans per row group. The trunk shard's FFN
/// is a partial route with the all-reduce fused into the HC write: its static trunk layers run NO
/// gather, so only its balanced trunk layers (`balanced_trunk`, `HIPFIRE_EP_TRUNK_BALANCED_OWNERS=1`)
/// gather (each slot's row from its producer, for the fixed `e % ranks` reduction partition), exactly
/// what each rank's gather tally measures (the head's gathers are tallied as head ones). Sealing
/// `moe_count` aborted every drafted step on an `OpLedger` mismatch (planned 48, measured 0).
fn planned_gather_layers(moe_count: usize, trunk_shard: bool, balanced_trunk: usize) -> usize {
    if trunk_shard {
        balanced_trunk
    } else {
        moe_count
    }
}

fn rank_counts(step: &ExecStep, rank_head: bool) -> OpCounts {
    let totals = step.totals.unwrap_or((0, 0, 0, 0));
    OpCounts {
        target_exchanges: step.verify_tally.trunk as usize,
        gather_bytes: step.draft_tally.bytes + step.verify_tally.bytes,
        head_draft_calls: if rank_head { step.head_calls } else { 0 },
        head_append_rows: if rank_head { totals.3 } else { 0 },
        row_groups: step.row_groups,
        wide_groups: step.wide_groups,
        edge_count: 0,
        edge_bytes: 0,
        drafts_generated: totals.0,
        accepted: totals.1,
        consumed_rows: totals.2,
    }
}

impl LaneMeshExecutor for RankLane<'_> {
    fn ownership(&self) -> RankOwnership {
        self.exec.session.ownership()
    }

    fn admit_lane(&mut self, admit: &LaneAdmit) -> Result<(), RankError> {
        let rank = self.rank();
        let store = self
            .bundle
            .lanes
            .as_ref()
            .ok_or_else(|| rank_err(rank, "Qwen4 lanes are not staged"))?;
        let lane = store
            .lane(&admit.epoch)
            .ok_or_else(|| rank_err(rank, format!("admit_lane: no lane for {:?}", admit.epoch)))?;
        if lane.host.status != LaneStatus::Decoding
            || lane.state.position != admit.position
            || lane.spec.is_some() != admit.spec
        {
            return Err(rank_err(
                rank,
                format!(
                    "admit_lane: lane {:?} is {:?} at {} (spec {}), the root admits it at {} (spec {})",
                    admit.epoch,
                    lane.host.status,
                    lane.state.position,
                    lane.spec.is_some(),
                    admit.position,
                    admit.spec
                ),
            ));
        }
        self.exec.session.admit(admit)
    }

    fn release_lane(&mut self, epoch: &RequestEpoch) -> Result<(), RankError> {
        self.exec.session.release(epoch)
    }

    fn begin_window(&mut self, start: &WindowStart) -> RankReady {
        let capacity = if self.exec.session.phase() == StepPhase::Idle {
            self.preflight(start)
        } else {
            Ok(())
        };
        let failed_preflight = capacity.is_err();
        let ready = self.exec.session.begin(start, capacity);
        if !matches!(ready.status, ReadyStatus::Ready { .. }) {
            // Windows opened for lanes before the failing one (or before the
            // reducer's refusal) are unwound: a rejected step leaves no ticket.
            if self.exec.session.phase() == StepPhase::Idle || failed_preflight {
                self.clear_windows(&start.lanes);
            }
            return ready;
        }
        let own = self.own();
        let lanes: Vec<LaneWin> = start
            .lanes
            .iter()
            .map(|plan| LaneWin::new(plan, &own))
            .collect();
        let unpriced = !self.exec.stepped
            || lanes
                .iter()
                .filter(|w| w.plan.source == ProposalSource::Native)
                .any(|w| !self.exec.seen_depths.contains(&w.plan.max_depth));
        self.exec.step = Some(ExecStep {
            key: start.key.clone(),
            lanes,
            draft_us: 0.0,
            verify_us: 0.0,
            prepare_us: 0.0,
            draft_tally: GatherTally::default(),
            verify_tally: GatherTally::default(),
            head_calls: 0,
            catch_up_rows: 0,
            verify: None,
            row_groups: 0,
            wide_groups: 0,
            totals: None,
            completed: StepPhase::Armed,
            unpriced,
        });
        ready
    }

    fn reject_window(&mut self, key: &MeshStepKey) {
        self.exec.session.reject(key);
        if let Some(step) = self.exec.step.take() {
            if step.key == *key {
                let plans: Vec<LanePlan> = step.lanes.iter().map(|w| w.plan.clone()).collect();
                self.clear_windows(&plans);
            } else {
                self.exec.step = Some(step);
            }
        }
    }

    fn run_draft_round(&mut self, round: &DraftRound) -> Result<RankDraftResult, RankError> {
        let rank = self.rank();
        let own = self.own();
        let sampler = own.sampler;
        let mut result = self.exec.session.on_draft_round(round)?;
        // This rank's slice of the round: the catch-ups and ops of the lanes it
        // heads (every replicated lane's, and its own lanes'). The reducer
        // validated the whole round and counted the same slice.
        let exec_round = {
            let step = self
                .exec
                .step
                .as_ref()
                .ok_or_else(|| rank_err(rank, "draft round without a begun window"))?;
            let heads = |key: &LaneWindowKey| {
                step.lanes
                    .iter()
                    .find(|w| w.plan.key == *key)
                    .is_some_and(|w| heads_lane(&own, &w.plan))
            };
            DraftRound {
                key: round.key.clone(),
                round: round.round,
                ops: round
                    .ops
                    .iter()
                    .filter(|op| heads(&op.key))
                    .copied()
                    .collect(),
                catch_up: round
                    .catch_up
                    .iter()
                    .filter(|(key, _)| heads(key))
                    .cloned()
                    .collect(),
                // A sharded round holds replicated ops only (`draft_vocab_shard_decision`), so
                // the filter keeps every op on both ranks and the halves line up.
                vocab_shard: round.vocab_shard,
            }
        };
        if exec_round.ops.len() != result.ops_executed {
            return Err(rank_err(
                rank,
                format!(
                    "the rank's window heads {} ops of the round, the reducer counted {}",
                    exec_round.ops.len(),
                    result.ops_executed
                ),
            ));
        }
        let s4 = {
            let step = self
                .exec
                .step
                .as_ref()
                .ok_or_else(|| rank_err(rank, "draft round without a begun window"))?;
            s4_round_for(
                &*self.bundle,
                &round.ops,
                |key| {
                    step.lanes
                        .iter()
                        .find(|w| w.plan.key == *key)
                        .map(|w| &w.plan)
                },
                rank,
            )
            .map_err(|e| rank_err(rank, e))?
        };
        let before = gather_tally();
        let t0 = Instant::now();
        let (catch_rows, reads) = self
            .draft_round_device(&exec_round, sampler, s4)
            .map_err(|e| rank_err(rank, e))?;
        let tally = gather_tally().since(before);
        if catch_rows != result.catch_up_rows {
            return Err(rank_err(
                rank,
                format!(
                    "the rank appended {catch_rows} owed head rows, the round scheduled {}",
                    result.catch_up_rows
                ),
            ));
        }
        let Some(step) = self.exec.step.as_mut() else {
            return Err(rank_err(rank, "draft round without a begun window"));
        };
        step.draft_us += t0.elapsed().as_secs_f64() * 1e6;
        step.draft_tally = GatherTally {
            trunk: step.draft_tally.trunk + tally.trunk,
            head: step.draft_tally.head + tally.head,
            bytes: step.draft_tally.bytes + tally.bytes,
        };
        step.head_calls += exec_round.ops.len();
        step.catch_up_rows += catch_rows;
        // A sharded-MoE pass crosses twice per pass (activation rows, then the
        // MoE's own slot rows) for an owned op; the reducer's head-gather count
        // is of replicated ops, of which an owned round has none.
        result.head_gathers = if s4.is_some() { 0 } else { tally.head as usize };
        let mut proposals = Vec::with_capacity(exec_round.ops.len());
        #[cfg(feature = "lab")]
        let mut traced: Vec<serde_json::Value> = Vec::new();
        for (op, read) in exec_round.ops.iter().zip(reads) {
            let win = step
                .lanes
                .iter_mut()
                .find(|w| w.plan.key == op.key)
                .ok_or_else(|| rank_err(rank, "draft op for a lane outside the plan"))?;
            let (keep_going, drop_last) = if read.abandoned {
                // The probe's head step was abandoned: the root confirms (or
                // falls back) from this token; no draft step stays.
                (false, false)
            } else {
                win.local_drafts.push(read.token);
                // The singleton's early stop, statement for statement: a token
                // after the first whose prefix acceptance fell below the row worth
                // is drafted but not kept; deeper drafting stops once even the
                // best per-draft acceptance cannot make the next row worth it.
                win.prefix *= draft_accept_estimate(read.margin);
                let drop_last = read.j > 0 && win.prefix < MTP_ROW_WORTH;
                let done = win.prefix * DRAFT_ACCEPT_MAX < MTP_ROW_WORTH
                    || read.j + 1 >= native_keep_cap(&win.plan);
                (!drop_last && !done, drop_last)
            };
            // The ops this rank answers: its own lanes', and the replicated
            // ones on the sampler owner (every other rank only keeps its own
            // head state in step).
            if !answers_op(&win.plan, sampler) {
                continue;
            }
            #[cfg(feature = "lab")]
            traced.push(serde_json::json!({
                "step_seq": round.key.step_seq,
                "round": round.round,
                "request_tag": op.key.request_epoch.request_tag,
                "position": op.position,
                "input": op.input,
                "token": read.token,
                "margin_bits": read.margin.to_bits(),
                "top1_fnv": format!("{:016x}", read.top1_fnv),
                "keep_going": keep_going,
                "drop_last": drop_last,
                "owned": win.plan.head_rank.is_some(),
                "s4": s4.is_some(),
            }));
            proposals.push(DraftProposal {
                key: op.key,
                token: read.token,
                keep_going,
                drop_last,
                // Greedy lanes propose the head's argmax; a sampled lane's
                // distribution read (S4) is the answering rank's alone.
                read: read.read,
            });
        }
        #[cfg(feature = "lab")]
        if let Some(dir) = lab_trace::dir() {
            lab_trace::append(&dir, rank, &traced).map_err(|e| rank_err(rank, e))?;
        }
        if !proposals.is_empty() {
            result.proposals = Some(proposals);
        }
        Ok(result)
    }

    fn run_verify(&mut self, plan: &VerifyPlan) -> Result<RankVerifyResult, RankError> {
        let rank = self.rank();
        let sampler = self.own().sampler;
        let mut result = self.exec.session.on_verify(plan)?;
        let before = gather_tally();
        let t0 = Instant::now();
        self.exec.drop_rows();
        let VerifyOut {
            picks,
            ends,
            logits,
            device,
        } = self.verify_device(plan).map_err(|e| rank_err(rank, e))?;
        let tally = gather_tally().since(before);
        let row_counts: Vec<usize> = plan.lanes.iter().map(|l| l.rows.len()).collect();
        let Some(step) = self.exec.step.as_mut() else {
            return Err(rank_err(rank, "verify without a begun window"));
        };
        step.verify_us += t0.elapsed().as_secs_f64() * 1e6;
        step.verify_tally = tally;
        let wide: Vec<bool> = plan.lanes.iter().map(|l| l.wide).collect();
        step.row_groups = row_group_count(&row_counts, &wide);
        step.wide_groups = plan.groups.iter().filter(|g| g.wide).count();
        step.completed = StepPhase::Verified;
        step.verify = Some(plan.clone());
        let mut first = 0usize;
        let mut mismatches = 0u64;
        for ((win, v), lane_picks) in step.lanes.iter_mut().zip(&plan.lanes).zip(&picks) {
            win.rows = v.rows.clone();
            win.row_begin = first;
            win.picks = lane_picks.clone();
            first += v.rows.len();
            // Evidence only: the rank's own head read vs the root's kept token.
            if win.plan.source == ProposalSource::Native {
                for (kept, local) in v.rows[1..].iter().zip(&win.local_drafts) {
                    if kept != local {
                        mismatches += 1;
                    }
                }
            }
        }
        self.exec.stats.draft_mismatches += mismatches;
        result.lane_ends = plan
            .lanes
            .iter()
            .zip(&ends)
            .map(|(l, &end)| (l.key, end))
            .collect();
        result.ledger = OpLedger {
            row_groups: self.exec.step.as_ref().map_or(0, |s| s.row_groups),
            wide_groups: self.exec.step.as_ref().map_or(0, |s| s.wide_groups),
            target_exchanges: tally.trunk as usize,
            microbatches: 0,
            head_draft_calls: plan.ledger.head_draft_calls,
            head_draft_calls_by_rank: plan.ledger.head_draft_calls_by_rank.clone(),
        };
        if sampler {
            let mut out = SamplerOutput::picks_only(
                plan.lanes
                    .iter()
                    .zip(picks)
                    .map(|(l, p)| (l.key, p))
                    .collect(),
            );
            // The host rows are needed only by a sampled lane the root did not
            // lend a device draw (`=0`, or a Leviathan lane).
            let host_rows = self.exec.step.as_ref().is_some_and(|s| {
                s.lanes
                    .iter()
                    .zip(&device)
                    .any(|(w, d)| w.plan.sampling != SamplingMode::Greedy && d.is_none())
            });
            if host_rows && plan.vocab_shard {
                // Unreachable by the root's decision; refused rather than run on half a head.
                return Err(rank_err(
                    rank,
                    "a vocabulary-sharded verify left a sampled lane without a device draw",
                ));
            }
            if host_rows {
                out.rows = Some(self.rows_handle(plan, logits));
            }
            out.device = device;
            result.sampler = Some(out);
        }
        Ok(result)
    }

    fn prepare_commit(&mut self, decision: &CommitDecision) -> Result<PreparedReceipt, RankError> {
        let rank = self.rank();
        let receipt = self.exec.session.on_prepare(decision)?;
        self.exec.drop_rows();
        let t0 = Instant::now();
        let head_rows = self
            .prepare_device(decision)
            .map_err(|e| rank_err(rank, e))?;
        if head_rows != receipt.head_append_rows {
            return Err(rank_err(
                rank,
                format!(
                    "the rank appended {head_rows} head rows, the decision's schedule is {}",
                    receipt.head_append_rows
                ),
            ));
        }
        // Evidence only: the emitted tokens against this rank's own picks.
        let mut mismatches = 0u64;
        // The emitted-vs-own-picks evidence needs this rank's picks to be target picks: a
        // non-sampler rank of a vocabulary-sharded verify holds shard-local ids, so it skips it.
        let sampler = self.own().sampler;
        if let Some(step) = self
            .exec
            .step
            .as_ref()
            .filter(|s| sampler || !s.verify.as_ref().is_some_and(|v| v.vocab_shard))
        {
            for (win, d) in step.lanes.iter().zip(&decision.lanes) {
                if d.forced {
                    continue;
                }
                for (i, token) in d.emitted.iter().enumerate() {
                    if win.picks.get(i).is_some_and(|p| p != token) {
                        mismatches += 1;
                    }
                }
            }
        }
        // Every fallible device operation completes (and is attributed) here.
        self.gpu
            .hip
            .device_synchronize()
            .map_err(|e| rank_err(rank, format!("prepare synchronize: {e}")))?;
        self.gpu
            .take_device_wait_error()
            .map_err(|e| rank_err(rank, format!("prepare: {e}")))?;
        self.exec.stats.pick_mismatches += mismatches;
        let Some(step) = self.exec.step.as_mut() else {
            return Err(rank_err(rank, "prepare without a begun window"));
        };
        step.prepare_us = t0.elapsed().as_secs_f64() * 1e6;
        step.totals = Some((
            decision.lanes.iter().map(|l| l.source.offered).sum(),
            decision.lanes.iter().map(|l| l.source.accepted).sum(),
            decision.lanes.iter().map(|l| l.consumed_rows).sum(),
            receipt.head_append_rows,
        ));
        step.completed = StepPhase::Prepared;
        Ok(receipt)
    }

    fn publish_commit(&mut self, decision: &CommitDecision) {
        self.exec.drop_rows();
        let rank = self.rank();
        let Some(step) = self.exec.step.take() else {
            return;
        };
        let Self {
            exec, bundle, gpu, ..
        } = self;
        let published = with_store(bundle, |store, bundle| {
            for (win, d) in step.lanes.iter().zip(&decision.lanes) {
                let epoch = win.plan.key.request_epoch;
                let lane = lane_mut(store, &epoch)?;
                let prepared = lane
                    .spec
                    .as_mut()
                    .and_then(|s| s.window.as_mut())
                    .and_then(|w| w.prepared.take());
                if let Some(prepared) = prepared {
                    publish_rank_lane(bundle, gpu, lane, prepared)
                        .map_err(|e| format!("{epoch:?}: publish: {e}"))?;
                }
                if let Some(spec) = lane.spec.as_mut() {
                    spec.window = None;
                }
                // The lane's host metadata, as `commit_verify`/`commit_host`.
                lane.host.generated.extend_from_slice(&d.emitted);
                lane.host.pending_seed = Some(d.next_seed);
            }
            Ok(())
        });
        if let Err(error) = published {
            exec.fatal = Some(format!("rank {rank}: {error}"));
            exec.session.on_abort();
            return;
        }
        exec.session.on_publish(decision);
        exec.stepped = true;
        exec.stats.steps += 1;
        let native = step
            .lanes
            .iter()
            .filter(|w| w.plan.source == ProposalSource::Native)
            .count() as u64;
        exec.stats.native_windows += native;
        for win in step
            .lanes
            .iter()
            .filter(|w| w.plan.source == ProposalSource::Native)
        {
            exec.seen_depths.insert(win.plan.max_depth);
        }
        let own = exec.session.ownership();
        let mut phases = vec![
            PhaseTiming {
                phase: TimedPhase::Verify,
                us: step.verify_us as f32,
                synchronized: true,
            },
            PhaseTiming {
                phase: TimedPhase::Prepare,
                us: step.prepare_us as f32,
                synchronized: true,
            },
        ];
        if step.head_calls > 0 {
            phases.insert(
                0,
                PhaseTiming {
                    phase: TimedPhase::Draft,
                    us: step.draft_us as f32,
                    synchronized: true,
                },
            );
        }
        exec.last = Some(RankCostEvidence {
            key: step.key.clone(),
            rank,
            completed: StepPhase::Published,
            phases,
            nodes: Vec::new(),
            counts: rank_counts(&step, own.head),
            unpriced: step.unpriced,
        });
    }

    fn abort_step(&mut self, _key: &MeshStepKey, _phase: StepPhase, reason: &str) {
        self.exec.drop_rows();
        if self.exec.abort_reason.is_none() {
            self.exec.abort_reason = Some(reason.to_string());
        }
        self.exec.session.on_abort();
        // A half-run window's device state is untrusted: poison every lane of
        // the step (store.poison drops the window and disarms the ring); the
        // mesh is quarantined, nothing is retried in place.
        if let Some(step) = self.exec.step.take() {
            if let Some(store) = self.bundle.lanes.as_mut() {
                for win in &step.lanes {
                    store.poison(&win.plan.key.request_epoch);
                }
            }
        }
    }

    fn evidence(&mut self, key: &MeshStepKey) -> Option<RankCostEvidence> {
        self.exec.last.clone().filter(|e| e.key == *key)
    }
}

// ---------------------------------------------------------------------------
// Concurrent step driver
// ---------------------------------------------------------------------------

/// Dispatch of each protocol phase to **every** rank at once. Results are in
/// rank order (rank 0 first). Implemented over the EP [`RankPool`]
/// ([`EpFanout`]) and, in tests, over scripted ranks on real threads.
pub(crate) trait MeshFanout {
    fn admit(&mut self, admit: &LaneAdmit) -> Result<(), RankError>;
    fn release(&mut self, epoch: &RequestEpoch) -> Result<(), RankError>;
    /// `Err` only for a mesh failure; a rank's rejection is a [`RankReady`].
    fn begin(&mut self, start: &WindowStart) -> Result<Vec<RankReady>, RankError>;
    fn reject(&mut self, key: &MeshStepKey);
    /// A phase with a collective: any failure must wake every rank.
    fn draft(&mut self, round: &DraftRound) -> Result<Vec<RankDraftResult>, RankError>;
    fn verify(&mut self, plan: &VerifyPlan) -> Result<Vec<RankVerifyResult>, RankError>;
    fn prepare(&mut self, decision: &CommitDecision) -> Result<Vec<PreparedReceipt>, RankError>;
    /// A failure after the all-prepared boundary, per rank (`None`: fine).
    fn publish(&mut self, decision: &CommitDecision) -> Vec<Option<String>>;
    fn abort(&mut self, key: &MeshStepKey, phase: StepPhase, reason: &str);
    fn evidence(&mut self, key: &MeshStepKey) -> Vec<Option<RankCostEvidence>>;
}

/// What one finished step hands back: the output to stream and the receipts
/// to merge into the next plan's prior.
#[derive(Debug)]
pub struct DrivenStep {
    pub output: StepOutput,
    pub key: MeshStepKey,
    pub receipts: Vec<RankCostEvidence>,
}

fn abort_all<F: MeshFanout>(
    root: &mut MeshRoot,
    fan: &mut F,
    key: Option<&MeshStepKey>,
    reason: &str,
) -> ProtocolError {
    let phase = root.phase();
    let record = match root.abort_step(key, phase, reason) {
        Ok(record) => record,
        Err(error) => return error,
    };
    if let Some(k) = record.key.clone().or_else(|| key.cloned()) {
        fan.abort(&k, record.phase, &record.reason);
    }
    ProtocolError::Aborted(record)
}

/// The reducer already recorded a rank-originated violation; make the ranks
/// follow. A pure caller-sequencing error is returned as is.
fn abort_after<F: MeshFanout>(
    root: &mut MeshRoot,
    fan: &mut F,
    key: &MeshStepKey,
    error: ProtocolError,
) -> ProtocolError {
    if let ProtocolError::Aborted(record) = &error {
        let at = record.key.clone().unwrap_or_else(|| key.clone());
        fan.abort(&at, record.phase, &record.reason);
        return error;
    }
    let reason = error.to_string();
    abort_all(root, fan, Some(key), &reason)
}

/// One mesh step in the protocol's order, every phase concurrent on all
/// ranks. Strict from the first draft on: any rank error or reducer-detected
/// disagreement aborts the root and every rank (one record, quarantine, no
/// fallback on uncertain device state); a preflight rejection rejects the
/// whole step before any collective and leaves the mesh usable.
pub(crate) fn drive_step<F: MeshFanout>(
    root: &mut MeshRoot,
    fan: &mut F,
    prior: Option<&MergedCostEvidence>,
    requests: &[LaneRequest],
    budgets: &WindowBudgets,
    rule: &mut dyn AcceptRule,
) -> Result<DrivenStep, ProtocolError> {
    let start = root.plan_window(prior, requests, budgets)?;
    let key = start.key.clone();
    let ready = match fan.begin(&start) {
        Ok(ready) => ready,
        Err(error) => {
            return Err(abort_all(
                root,
                fan,
                Some(&key),
                &format!("rank {}: {}", error.rank, error.reason),
            ))
        }
    };
    match root.collect_ready(&ready) {
        Ok(()) => {}
        Err(ProtocolError::Preflight(rejections)) => {
            fan.reject(&key);
            return Err(ProtocolError::Preflight(rejections));
        }
        Err(error) => return Err(abort_after(root, fan, &key, error)),
    }
    loop {
        let round = match root.next_draft_round() {
            Ok(Some(round)) => round,
            Ok(None) => break,
            Err(error) => return Err(abort_after(root, fan, &key, error)),
        };
        let results = match fan.draft(&round) {
            Ok(results) => results,
            Err(error) => {
                return Err(abort_all(
                    root,
                    fan,
                    Some(&key),
                    &format!("rank {}: {}", error.rank, error.reason),
                ))
            }
        };
        if let Err(error) = root.collect_draft_with(&round, &results, rule) {
            return Err(abort_after(root, fan, &key, error));
        }
    }
    let plan = match root.seal_verify_with(rule) {
        Ok(plan) => plan,
        Err(error) => return Err(abort_after(root, fan, &key, error)),
    };
    let results = match fan.verify(&plan) {
        Ok(results) => results,
        Err(error) => {
            return Err(abort_all(
                root,
                fan,
                Some(&key),
                &format!("rank {}: {}", error.rank, error.reason),
            ))
        }
    };
    if let Err(error) = root.collect_verify(&results) {
        return Err(abort_after(root, fan, &key, error));
    }
    let decision = match root.decide_commit(rule) {
        Ok(decision) => decision,
        Err(error) => return Err(abort_after(root, fan, &key, error)),
    };
    let receipts = match fan.prepare(&decision) {
        Ok(receipts) => receipts,
        Err(error) => {
            return Err(abort_all(
                root,
                fan,
                Some(&key),
                &format!("rank {}: {}", error.rank, error.reason),
            ))
        }
    };
    if let Err(error) = root.collect_prepared(&receipts) {
        return Err(abort_after(root, fan, &key, error));
    }
    let output = root.publish_commit()?;
    rule.on_publish(&output);
    let fatal: Vec<String> = fan.publish(&decision).into_iter().flatten().collect();
    if !fatal.is_empty() {
        // The step is already published at the root (no step is in flight):
        // quarantine the mesh with no step key and make every rank follow.
        let reason = format!(
            "publish failed after all ranks prepared: {}",
            fatal.join("; ")
        );
        let phase = root.phase();
        let record = match root.abort_step(None, phase, &reason) {
            Ok(record) => record,
            Err(error) => return Err(error),
        };
        fan.abort(&key, record.phase, &record.reason);
        return Err(ProtocolError::Aborted(record));
    }
    let receipts = fan.evidence(&key).into_iter().flatten().collect();
    Ok(DrivenStep {
        output,
        key,
        receipts,
    })
}

// ---------------------------------------------------------------------------
// EP dispatch (RankPool)
// ---------------------------------------------------------------------------

/// The EP mesh's phase dispatch: rank 0 on the caller's thread, the peers on
/// their workers, concurrently ([`Qwen4EpMesh::run`]).
pub(crate) struct EpFanout<'a> {
    bundle: &'a mut Qwen4Bundle,
    gpu: &'a mut Gpu,
    rank0: &'a mut RankExec,
}

fn peer_rank(bundle: &Qwen4Bundle) -> usize {
    bundle
        .weights
        .ep_placement()
        .map_or(0, |placement| placement.rank)
}

/// A peer's lane executor state, built on its first call (holding the mesh's vocabulary-shard
/// exchange).
fn peer_exec(mirrors: &mut PeerMirrors, rank: usize) -> &mut RankExec {
    let vocab = mirrors.vocab.clone();
    mirrors
        .mesh_lanes
        .get_or_insert_with(|| RankExec::new(RankOwnership::ep(rank)).with_vocab(vocab))
}

impl EpFanout<'_> {
    /// Run `local` (rank 0) and `job` (every peer) concurrently, results in
    /// rank order. The mesh is detached from the bundle for the call so the
    /// nested bundle calls run locally; a failure of any rank poisons the pool
    /// and aborts the gather barrier.
    fn fan<T: Send>(
        &mut self,
        local: impl FnOnce(&mut RankLane<'_>) -> Result<T, String>,
        job: &(dyn Fn(&mut RankLane<'_>) -> Result<T, String> + Sync),
    ) -> Result<Vec<T>, String> {
        let Some(mesh) = self.bundle.ep.take() else {
            return Err("qwen4 EP mesh is not attached".to_string());
        };
        let result = {
            let rank0 = &mut *self.rank0;
            let bundle = &mut *self.bundle;
            let gpu = &mut *self.gpu;
            mesh.run(
                || {
                    local(&mut RankLane {
                        exec: rank0,
                        bundle,
                        gpu,
                    })
                },
                &|peer_bundle, peer_gpu, mirrors| {
                    let rank = peer_rank(peer_bundle);
                    let exec = peer_exec(mirrors, rank);
                    job(&mut RankLane {
                        exec,
                        bundle: peer_bundle,
                        gpu: peer_gpu,
                    })
                },
            )
        };
        self.bundle.ep = Some(mesh);
        result
    }

    /// [`Self::fan`] with a rank-level `Result` value that leaves the mesh
    /// usable (no collective inside: a rank's refusal is data, not a failure).
    fn soft<T: Send>(
        &mut self,
        f: &(dyn Fn(&mut RankLane<'_>) -> T + Sync),
    ) -> Result<Vec<T>, RankError> {
        self.fan(|lane| Ok(f(lane)), &|lane| Ok(f(lane)))
            .map_err(|reason| rank_err(0, reason))
    }

    /// A phase with collectives: a rank error becomes `Err(String)` so the
    /// pool poisons and the barrier aborts.
    fn strict<T: Send>(
        &mut self,
        f: &(dyn Fn(&mut RankLane<'_>) -> Result<T, RankError> + Sync),
    ) -> Result<Vec<T>, RankError> {
        let named = |e: RankError| format!("rank {}: {}", e.rank, e.reason);
        self.fan(|lane| f(lane).map_err(named), &|lane| {
            f(lane).map_err(named)
        })
        .map_err(|reason| {
            let rank = reason
                .strip_prefix("rank ")
                .and_then(|r| r.split(':').next())
                .and_then(|r| r.parse().ok())
                .unwrap_or(0);
            let text = reason
                .split_once(": ")
                .filter(|_| reason.starts_with("rank "))
                .map_or(reason.clone(), |(_, rest)| rest.to_string());
            rank_err(rank, text)
        })
    }
}

impl MeshFanout for EpFanout<'_> {
    fn admit(&mut self, admit: &LaneAdmit) -> Result<(), RankError> {
        let all = self.soft(&|lane| lane.admit_lane(admit))?;
        all.into_iter().collect()
    }

    fn release(&mut self, epoch: &RequestEpoch) -> Result<(), RankError> {
        let all = self.soft(&|lane| lane.release_lane(epoch))?;
        all.into_iter().collect()
    }

    fn begin(&mut self, start: &WindowStart) -> Result<Vec<RankReady>, RankError> {
        self.soft(&|lane| lane.begin_window(start))
    }

    fn reject(&mut self, key: &MeshStepKey) {
        let _ = self.soft(&|lane| lane.reject_window(key));
    }

    fn draft(&mut self, round: &DraftRound) -> Result<Vec<RankDraftResult>, RankError> {
        self.strict(&|lane| {
            if lane.own().head {
                lane.run_draft_round(round).map(Some)
            } else {
                Ok(None)
            }
        })
        .map(|all| all.into_iter().flatten().collect())
    }

    fn verify(&mut self, plan: &VerifyPlan) -> Result<Vec<RankVerifyResult>, RankError> {
        self.strict(&|lane| lane.run_verify(plan))
    }

    fn prepare(&mut self, decision: &CommitDecision) -> Result<Vec<PreparedReceipt>, RankError> {
        self.strict(&|lane| lane.prepare_commit(decision))
    }

    fn publish(&mut self, decision: &CommitDecision) -> Vec<Option<String>> {
        match self.soft(&|lane| {
            lane.publish_commit(decision);
            lane.exec.take_fatal()
        }) {
            Ok(all) => all,
            Err(error) => vec![Some(format!("rank {}: {}", error.rank, error.reason))],
        }
    }

    fn abort(&mut self, key: &MeshStepKey, phase: StepPhase, reason: &str) {
        // Wake any rank still inside a collective and quarantine the pool; the
        // per-rank abort then runs on every rank (forced: it must run on a
        // poisoned pool).
        if let Some(mesh) = self.bundle.ep.as_ref() {
            mesh.poison(reason);
        }
        let Some(mesh) = self.bundle.ep.take() else {
            return;
        };
        let rank0 = &mut *self.rank0;
        let bundle = &mut *self.bundle;
        let gpu = &mut *self.gpu;
        let _ = mesh.run_forced(
            || {
                RankLane {
                    exec: rank0,
                    bundle,
                    gpu,
                }
                .abort_step(key, phase, reason);
                Ok(())
            },
            &|peer_bundle, peer_gpu, mirrors| {
                let rank = peer_rank(peer_bundle);
                let exec = peer_exec(mirrors, rank);
                RankLane {
                    exec,
                    bundle: peer_bundle,
                    gpu: peer_gpu,
                }
                .abort_step(key, phase, reason);
                Ok(())
            },
        );
        self.bundle.ep = Some(mesh);
    }

    fn evidence(&mut self, key: &MeshStepKey) -> Vec<Option<RankCostEvidence>> {
        self.soft(&|lane| lane.evidence(key)).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Public driver
// ---------------------------------------------------------------------------

/// A lane entering the mesh protocol: its prompt is done and the lane's
/// pending seed is the window's first row.
#[derive(Clone, Debug)]
pub struct MeshLaneAdmission {
    pub epoch: RequestEpoch,
    /// The lane's target position (prompt length after prefill).
    pub position: usize,
    pub seed: u32,
    pub eos: Option<u32>,
    /// The request's RNG seed (root-only; greedy acceptance never draws).
    pub rng_seed: u64,
    /// Full prompt (seeds the root's copy history).
    pub prompt: Vec<u32>,
    /// `Some(k)`: a native-MTP lane; `None`: a plain AR lane.
    pub spec_k: Option<usize>,
    /// The lane's slot holds the widened capture ring a copy window needs
    /// (`Qwen4LaneStore::lane_copy_capable`); only such a lane arms copy.
    pub copy_capable: bool,
}

/// The EP mesh's lane transaction owner: the root plus rank 0's executor
/// state. One per staged EP lane store; the lanes' prefill runs through the
/// ordinary lane executor, decode through [`Self::step`].
pub struct EpMeshLanes {
    root: MeshRoot,
    rank0: RankExec,
    ranks: usize,
    copy_armed: bool,
    pending: Option<(MeshStepKey, Vec<RankCostEvidence>)>,
    interval_start: Option<Instant>,
    prefill_ran: bool,
    last_merged: Option<MergedCostEvidence>,
    steps: u64,
    sampler: Option<LaneSampler>,
    /// Whether the root's draft-split setting has been decided from every
    /// rank's head ([`Self::decide_draft_split`]); done once, at the first step.
    draft_split_decided: bool,
}

/// What [`EpMeshLanes::step`] reports besides the output.
#[derive(Clone, Debug)]
pub struct MeshStepReport {
    pub output: StepOutput,
    pub key: MeshStepKey,
    /// The previous step's merged evidence the plan consumed (`None` on the
    /// first step or when its merge was refused).
    pub prior_priced: Option<bool>,
    pub wall_us: f64,
}

/// Why a step did not run, as the driver reports it.
#[derive(Debug)]
pub enum MeshStepError {
    /// A rank refused before any collective; the mesh is usable and the root
    /// planned nothing.
    Preflight(Vec<(usize, String)>),
    /// The step aborted; the mesh is quarantined (reload to recover).
    Aborted(AbortRecord),
    /// Any other root-side error (a caller-sequencing bug or a bad request).
    Protocol(ProtocolError),
}

impl std::fmt::Display for MeshStepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preflight(v) => write!(f, "mesh lane preflight rejected by {v:?}"),
            Self::Aborted(r) => write!(f, "{r}"),
            Self::Protocol(p) => write!(f, "{p}"),
        }
    }
}

impl std::error::Error for MeshStepError {}

impl From<ProtocolError> for MeshStepError {
    fn from(error: ProtocolError) -> Self {
        match error {
            ProtocolError::Preflight(v) => Self::Preflight(v),
            ProtocolError::Aborted(r) => Self::Aborted(r),
            other => Self::Protocol(other),
        }
    }
}

impl EpMeshLanes {
    /// The root and rank 0's state for `bundle`'s staged EP lane store, run on rank 0's `gpu`
    /// (whose flags and forward fix the all-rows MoE rule the plan is sealed with).
    /// `topology_generation` names this mesh load; `floor_enabled` is the
    /// production floor (lab route modes ignore it).
    pub fn new(
        gpu: &Gpu,
        bundle: &Qwen4Bundle,
        topology_generation: u64,
        floor_enabled: bool,
    ) -> Result<Self, String> {
        let ranks = bundle.ranks();
        if bundle.ep_mesh().is_none() || ranks < 2 {
            return Err("EP mesh lanes need an attached EP mesh".to_string());
        }
        let store = bundle
            .lanes()
            .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
        let (moe_layers, balanced_trunk) = bundle
            .execution
            .as_ref()
            .map(|forward| (forward.moe_count(), forward.ep_balanced_layer_count()))
            .ok_or_else(|| "Qwen4 forward resources are not attached".to_string())?;
        let moe_layers = planned_gather_layers(
            moe_layers,
            bundle.config.trunk_shard.is_some(),
            balanced_trunk,
        );
        let moe_once = store.stage_policy().share(StageName::Moe) == StageShare::OnceAllRows;
        // The one predicate the executor's `moe_all` is decided by, captured once from rank 0's
        // own `Gpu` / forward: the sealed ledger plans its exchanges from it, and every rank
        // checks its own rule against it at `begin_window`.
        let all_rows = all_rows_rule_of(gpu, bundle)?;
        let world: Vec<RankOwnership> = (0..ranks).map(RankOwnership::ep).collect();
        let mut root = MeshRoot::new(
            RankOwnership::ep(0),
            &world,
            RootConfig {
                topology_generation,
                stage: StagePolicy {
                    moe_layers,
                    moe_once,
                    stream_bf16: false,
                    all_rows,
                },
                floor_enabled,
            },
        )
        .map_err(|e| e.to_string())?;
        // A store staged with the copy budget arms the root's copy lane (the
        // request-local n-gram context, the take-length chooser and the head
        // lag bookkeeping): the ranks run its windows through the shared rank
        // helper ([`crate::mesh_rank_copy`]).
        let copy_armed = match store.copy() {
            Some(copy) => {
                root = root
                    .with_copy_lane(copy.config.ngram)
                    .map_err(|e| e.to_string())?
                    .with_copy_max_lanes(copy.config.wide_lanes);
                true
            }
            None => false,
        };
        // The vocabulary-sharded verify (`HIPFIRE_QWEN4_EP_VOCAB_SHARD`): armed when the mesh
        // attached its exchange and the forward is eager (no replay recording or graph capture:
        // the exchange is host-ordered and never replayed); the root then seals
        // `VerifyPlan::vocab_shard` for every step the exchange's buffers hold whose sampled
        // lanes are all device-drawn.
        let vocab = bundle
            .ep_mesh()
            .and_then(|mesh| mesh.vocab_exchange().cloned());
        if let Some(exchange) = vocab.as_ref().filter(|_| all_rows.eager) {
            root = root
                .with_vocab_shard_rows(exchange.rows())
                // The MQ2 draft head's shard (`HIPFIRE_QWEN4_EP_DRAFT_VOCAB_SHARD`): the exchange
                // carries its partial scratch only when that switch is on (0 rows: never sharded).
                .with_draft_vocab_shard_rows(exchange.draft_rows());
        }
        Ok(Self {
            copy_armed,
            root,
            rank0: RankExec::new(RankOwnership::ep(0)).with_vocab(vocab),
            ranks,
            pending: None,
            interval_start: None,
            prefill_ran: false,
            last_merged: None,
            steps: 0,
            sampler: Some(LaneSampler::new(bundle.lane_vocab())),
            draft_split_decided: false,
        })
    }

    /// The window budgets of the staged store (verify rows per lane, copy take
    /// cap and head-lag rows from the store's own copy budget) with route
    /// policy `lab`.
    pub fn window_budgets(
        &self,
        bundle: &Qwen4Bundle,
        lab: LabRoutes,
    ) -> Result<WindowBudgets, String> {
        let store = bundle
            .lanes()
            .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
        let copy = store.copy();
        let verify_rows = store.verify_rows_per_lane();
        Ok(WindowBudgets {
            max_lanes: store.max_lanes(),
            verify_rows_per_lane: verify_rows,
            max_rows: store.max_lanes() * verify_rows,
            copy_take_cap: copy.map_or(0, |c| c.take_cap),
            head_lag_rows: copy.map_or(0, |c| c.config.head_lag_rows),
            lab,
            pp_microbatch_lanes: None,
        })
    }

    pub fn ranks(&self) -> usize {
        self.ranks
    }

    pub fn root(&self) -> &MeshRoot {
        &self.root
    }

    pub fn last_evidence(&self) -> Option<&MergedCostEvidence> {
        self.last_merged.as_ref()
    }

    /// A prefill tile ran since the previous step: its interval is not a
    /// decode price.
    pub fn note_prefill(&mut self) {
        self.prefill_ran = true;
    }

    /// Decide the X2 draft split once, at the first step: the root may give a
    /// window's head lanes owners only while `HIPFIRE_QWEN4_EP_DRAFT_SPLIT` is not
    /// `0` and every rank's MTP head can run lane-local
    /// ([`Qwen4Bundle::mtp_lane_local_ready`]: rank 0 and every peer, asked
    /// through the pool). Logs the decision once.
    fn decide_draft_split(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
    ) -> Result<(), MeshStepError> {
        let ready = {
            let (mut fan, _) = self.fan(bundle, gpu);
            fan.soft(&|lane| lane.bundle.mtp_lane_local_ready())
        }
        .map_err(|e| {
            MeshStepError::Protocol(ProtocolError::Config(format!(
                "draft split readiness, rank {}: {}",
                e.rank, e.reason
            )))
        })?;
        let (want, reason) = if !ep_draft_split_enabled() {
            (false, "HIPFIRE_QWEN4_EP_DRAFT_SPLIT=0".to_string())
        } else if let Some(rank) = ready.iter().position(|ready| !ready) {
            (
                false,
                format!("rank {rank}'s MTP head has no lane-local MoE"),
            )
        } else {
            (
                true,
                format!("every one of {} ranks is lane-local ready", ready.len()),
            )
        };
        let on = self.root.set_draft_split(want);
        eprintln!(
            "qwen4 EP draft split: {} ({reason})",
            if on { "on" } else { "off" }
        );
        self.draft_split_decided = true;
        Ok(())
    }

    fn fan<'a>(
        &'a mut self,
        bundle: &'a mut Qwen4Bundle,
        gpu: &'a mut Gpu,
    ) -> (EpFanout<'a>, &'a mut MeshRoot) {
        (
            EpFanout {
                bundle,
                gpu,
                rank0: &mut self.rank0,
            },
            &mut self.root,
        )
    }

    /// Admit a lane whose prompt is done to the root and mirror it on every
    /// rank (identity only: no RNG, no policy).
    pub fn admit_lane(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        admission: MeshLaneAdmission,
    ) -> Result<(), String> {
        let head = if admission.spec_k.is_some() {
            HeadState::Aligned
        } else {
            HeadState::Free
        };
        // The lane cost profile of the loaded arch, bound before the first
        // lane's policy is born (later calls change nothing).
        self.root.bind_cost_arch(&gpu.arch);
        let admit = self
            .root
            .admit_lane(LaneAdmission {
                epoch: admission.epoch,
                position: admission.position,
                seed: admission.seed,
                eos: admission.eos,
                rng_seed: admission.rng_seed,
                prompt: admission.prompt,
                spec_k: admission.spec_k,
                head,
                copy: self.copy_armed && admission.copy_capable && admission.spec_k.is_some(),
                head_lag: Vec::new(),
            })
            .map_err(|e| e.to_string())?;
        let (mut fan, _) = self.fan(bundle, gpu);
        fan.admit(&admit)
            .map_err(|e| format!("rank {}: {}", e.rank, e.reason))
    }

    /// Release a finished/cancelled lane on the root and every rank.
    pub fn release_lane(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        epoch: RequestEpoch,
    ) -> Result<(), String> {
        if let Some(sampler) = self.sampler.as_mut() {
            sampler.release(&epoch);
        }
        self.root.release_lane(epoch).map_err(|e| e.to_string())?;
        let (mut fan, _) = self.fan(bundle, gpu);
        fan.release(&epoch)
            .map_err(|e| format!("rank {}: {}", e.rank, e.reason))
    }

    /// One decode step over `requests`: the root plans on the previous step's
    /// merged rank evidence, every rank replays the root's messages
    /// concurrently, the root decides. See the module docs.
    pub fn step(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        requests: &[LaneRequest],
        budgets: &WindowBudgets,
        rule: &mut dyn AcceptRule,
    ) -> Result<MeshStepReport, MeshStepError> {
        let now = Instant::now();
        // The previous step's price: merged rank receipts and the controller's
        // previous-window-start -> now interval.
        let mut prior: Option<MergedCostEvidence> = None;
        if let Some((_, receipts)) = self.pending.take() {
            let interval = ControllerInterval {
                wall_us: self
                    .interval_start
                    .map_or(f64::NAN, |t| now.duration_since(t).as_secs_f64() * 1e6),
                prefill_ran: self.prefill_ran,
            };
            match self.root.merge_evidence(&receipts, interval) {
                Ok(merged) => prior = Some(merged),
                Err(EvidenceError::Stale(_)) => {}
                Err(error) => {
                    return Err(MeshStepError::Protocol(ProtocolError::Evidence(error)));
                }
            }
        }
        self.prefill_ran = false;
        self.interval_start = Some(now);
        let prior_priced = prior.as_ref().map(|m| m.priced);
        self.last_merged = prior.clone().or(self.last_merged.take());
        if !self.draft_split_decided {
            self.decide_draft_split(bundle, gpu)?;
        }
        let (mut fan, root) = self.fan(bundle, gpu);
        let driven = drive_step(root, &mut fan, prior.as_ref(), requests, budgets, rule)?;
        self.steps += 1;
        self.pending = Some((driven.key.clone(), driven.receipts));
        Ok(MeshStepReport {
            output: driven.output,
            key: driven.key,
            prior_priced,
            wall_us: now.elapsed().as_secs_f64() * 1e6,
        })
    }

    /// Every rank's diagnostic counters (rank order).
    pub fn rank_stats(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
    ) -> Result<Vec<RankLaneStats>, String> {
        let (mut fan, _) = self.fan(bundle, gpu);
        fan.soft(&|lane| lane.exec.stats()).map_err(|e| e.reason)
    }

    /// Serve-driver addition (EpServe): the root's request-termination rule
    /// ([`MeshRoot::set_driver_terminates`]): the driver closes lanes on
    /// EOS/stop/length, the root never does and drafts through an EOS draft.
    pub fn set_driver_terminates(&mut self, on: bool) {
        self.root.set_driver_terminates(on);
        if let Some(sampler) = self.sampler.as_mut() {
            sampler.set_driver_terminates(on);
        }
    }

    /// Whether the root holds `epoch`'s lane.
    pub fn has_lane(&self, epoch: RequestEpoch) -> bool {
        self.root.has_lane(epoch)
    }

    /// Replace a live lane's pending seed at the root (the emitter's forced
    /// close tokens). The ranks' lane tables take the same token through
    /// `Qwen4Bundle::force_lane_seed`; the rank preflight checks the two agree.
    pub fn force_seed(&mut self, epoch: RequestEpoch, token: u32) -> Result<(), String> {
        self.root
            .force_seed(epoch, token)
            .map_err(|e| e.to_string())
    }

    /// [`Self::release_lane`] without a device handle: the release is host
    /// bookkeeping on the root and on every rank's [`RankSession`]. Runs on a
    /// quarantined mesh too (a failed step's lanes must still leave): once the
    /// root is quarantined a refusal is not an error of the release itself.
    pub fn release_lane_host(
        &mut self,
        bundle: &mut Qwen4Bundle,
        epoch: RequestEpoch,
    ) -> Result<(), String> {
        if let Some(sampler) = self.sampler.as_mut() {
            sampler.release(&epoch);
        }
        let root = self.root.release_lane(epoch).map_err(|e| e.to_string());
        let Some(mesh) = bundle.ep.take() else {
            return Err("qwen4 EP mesh is not attached".to_string());
        };
        let rank0 = &mut self.rank0;
        let ranks = mesh.run_forced(
            || {
                rank0
                    .session
                    .release(&epoch)
                    .map_err(|e| format!("rank {}: {}", e.rank, e.reason))
            },
            &|peer_bundle, _, mirrors| {
                let rank = peer_rank(peer_bundle);
                let exec = peer_exec(mirrors, rank);
                exec.session
                    .release(&epoch)
                    .map_err(|e| format!("rank {}: {}", e.rank, e.reason))
            },
        );
        bundle.ep = Some(mesh);
        let quarantined = self.root.quarantined().is_some();
        if quarantined {
            return Ok(());
        }
        root?;
        ranks.map(|_| ())
    }
}

impl EpMeshLanes {
    /// [`Self::admit_lane`] for a sampled request: the root's [`LaneSampler`]
    /// registers the request's sampler (temperature/top-p/top-k/min-p and
    /// penalty history over the prompt) before the lane enters the root; the
    /// request's RNG stream is `admission.rng_seed`. Registration is undone
    /// when the lane is not admitted.
    pub fn admit_lane_sampled(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        admission: MeshLaneAdmission,
        cfg: &SpecRequestConfig,
    ) -> Result<(), String> {
        let epoch = admission.epoch;
        self.sampler
            .as_mut()
            .ok_or_else(|| "the mesh lane sampler is in use".to_string())?
            .admit(epoch, cfg, &admission.prompt, admission.seed)?;
        let admitted = self.admit_lane(bundle, gpu, admission);
        if admitted.is_err() {
            if let Some(sampler) = self.sampler.as_mut() {
                sampler.release(&epoch);
            }
        }
        admitted
    }

    /// [`Self::step`] with the root's [`LaneSampler`] as the acceptance rule:
    /// the lanes' [`LaneRequest::sampling`] selects greedy, naive or
    /// Leviathan acceptance per lane.
    pub fn step_sampled(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        requests: &[LaneRequest],
        budgets: &WindowBudgets,
    ) -> Result<MeshStepReport, MeshStepError> {
        let mut sampler = self.sampler.take().ok_or_else(|| {
            MeshStepError::Protocol(ProtocolError::Config(
                "the mesh lane sampler is in use".to_string(),
            ))
        })?;
        sampler.set_device_sample(
            gpu.flags.qwen4_lane_device_sample_enabled()
                && hipfire_runtime::lane_draw::lane_draws_supported(gpu),
        );
        let report = self.step(bundle, gpu, requests, budgets, &mut sampler);
        self.sampler = Some(sampler);
        report
    }
}

/// The sampler owner's device draws of one verify: every lane whose
/// [`VerifyPlan::samples`] entry is set is drawn by ONE
/// [`lane_draws`] call over its rows of the step's `logits` (Leviathan
/// acceptance against the lane's draft distributions under the shared device
/// sampler's law, enqueued right behind the forward on its stream), with one
/// D2H of ids, accept counts and RNG states for all lanes.
fn device_draws(
    gpu: &mut Gpu,
    arena: &mut Option<LaneDrawArena>,
    plan: &VerifyPlan,
    logits: &GpuTensor,
    vocab: usize,
) -> R<Vec<Option<DevicePick>>> {
    let mut device: Vec<Option<DevicePick>> = vec![None; plan.lanes.len()];
    if plan.samples.iter().all(Option::is_none) {
        return Ok(device);
    }
    if plan.samples.len() != plan.lanes.len() {
        return Err(format!(
            "{} device-sample entries for {} lanes",
            plan.samples.len(),
            plan.lanes.len()
        ));
    }
    let arena = arena.get_or_insert_with(LaneDrawArena::new);
    // Each sampled lane's rows are a view of the step logits.
    let mut views: Vec<(usize, GpuTensor)> = Vec::new();
    let mut first = 0usize;
    for (i, vl) in plan.lanes.iter().enumerate() {
        if plan.samples[i].is_some() {
            views.push((i, logits.sub_offset(first * vocab, vl.rows.len() * vocab)));
        }
        first += vl.rows.len();
    }
    let jobs: Vec<LaneDrawJob<'_>> = views
        .iter()
        .map(|(i, view)| {
            let vl = &plan.lanes[*i];
            let spec = plan.samples[*i].as_ref().expect("view of a sampled lane");
            LaneDrawJob {
                logits: view,
                rows: vl.rows.len(),
                vocab,
                base: 0,
                drafts: &vl.rows[1..],
                q: &spec.q,
                history: &spec.history,
                cfg: &spec.cfg,
                rng_state: spec.rng_state,
                eos: spec.eos,
            }
        })
        .collect();
    let draws: Vec<LaneDraw> =
        lane_draws(gpu, arena, &jobs).map_err(|e| format!("device lane sample: {e}"))?;
    if draws.len() != jobs.len() {
        return Err(format!(
            "device lane sample returned {} draws for {} lanes",
            draws.len(),
            jobs.len()
        ));
    }
    for ((i, _), draw) in views.iter().zip(draws) {
        device[*i] = Some(DevicePick {
            tokens: draw.tokens,
            accepted: draw.accepted,
            advanced_rng: draw.advanced_rng,
        });
    }
    Ok(device)
}

/// Lab draft trace (`HIPFIRE_LAB_MTP_DRAFT_TRACE=<dir>`): one JSON line per
/// answered draft op per rank, appended to `<dir>/rank<r>.jsonl`, so the ops of
/// a split run merge into the incumbent's (`HIPFIRE_QWEN4_EP_DRAFT_SPLIT=0`) rank 0
/// trace op for op.
#[cfg(feature = "lab")]
mod lab_trace {
    use std::io::Write;

    /// Directory the per-rank trace files are written to.
    const TRACE_DIR_ENV: &str = "HIPFIRE_LAB_MTP_DRAFT_TRACE";

    pub(super) fn dir() -> Option<String> {
        hipfire_config::developer_var(TRACE_DIR_ENV)
            .ok()
            .filter(|dir| !dir.is_empty())
    }

    pub(super) fn fnv1a(bytes: &[u8]) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    pub(super) fn append(
        dir: &str,
        rank: usize,
        lines: &[serde_json::Value],
    ) -> Result<(), String> {
        if lines.is_empty() {
            return Ok(());
        }
        std::fs::create_dir_all(dir).map_err(|e| format!("draft trace dir {dir}: {e}"))?;
        let path = std::path::Path::new(dir).join(format!("rank{rank}.jsonl"));
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("draft trace {}: {e}", path.display()))?;
        for line in lines {
            writeln!(file, "{line}").map_err(|e| format!("draft trace: {e}"))?;
        }
        Ok(())
    }
}

/// This rank's side of a vocabulary-sharded head round: its partial scratch (a view of the
/// exchange's), the peer's scratch address and device, and the exchange of one draft call —
/// publish this rank's event on its stream, meet the peer's host at the barrier, make the stream
/// wait for the peer's event on the device. `Err` when the rank has no exchange (the mesh did
/// not attach one, or its draft scratch is off): a sharded round is refused, never run unsharded.
fn draft_shard(exec: &RankExec) -> R<DraftShard> {
    let exchange = exec.vocab.clone().ok_or_else(|| {
        "a vocabulary-sharded head round on a rank without the shard exchange".to_string()
    })?;
    let rank = exec.rank;
    if exchange.draft_rows() == 0 {
        return Err("a vocabulary-sharded head round without the exchange's draft scratch".into());
    }
    let partial = exchange.own_view(rank, SLOT_PARTIAL)?;
    let peer_partial = exchange.peer_slot(rank, SLOT_PARTIAL)?;
    let peer_device = exchange
        .device_of(1 - rank)
        .ok_or_else(|| "the peer rank registered no device".to_string())?;
    let gate = exchange.clone();
    let hip = |e: hip_bridge::HipError| DispatchError::Hip(e.to_string());
    DraftShard::new(
        rank,
        partial,
        peer_partial,
        peer_device,
        Box::new(move |gpu: &mut Gpu| {
            gate.publish(rank, gpu).map_err(hip)?;
            gate.barrier().map_err(DispatchError::Hip)?;
            gate.wait_peer(rank, gpu).map_err(hip)
        }),
    )
    .map_err(|e| format!("{e:?}"))
}

/// One rank's head buffers of a vocabulary-sharded verify ([`VerifyPlan::vocab_shard`]).
struct ShardHead<'a> {
    exchange: &'a VocabExchange,
    rank: usize,
    /// The head rows this rank projects and scans.
    view: VocabShardView,
    /// Packed `[n_dec][width]` shard logits and `n_dec` shard-local top-1 ids, as the head and
    /// the argmax write them: the lane store's own buffers on the sampler owner (nobody reads
    /// its half), the exchange's peer-readable ones on the other rank.
    logits: GpuTensor,
    top1: GpuTensor,
    /// Decode rows of the step.
    rows: usize,
}

/// [`ShardHead`] of `rank` for a step of `n_dec` rows; `store_*` are the lane store's step
/// buffers (the sampler owner's head destination). `Err` when the rank is not one of the two
/// shard ranks, the sampler owner is not rank 0, or the exchange's buffers cannot hold the rows.
fn shard_head<'a>(
    exchange: &'a VocabExchange,
    rank: usize,
    sampler: bool,
    n_dec: usize,
    store_logits: &GpuTensor,
    store_top1: &GpuTensor,
) -> R<ShardHead<'a>> {
    if rank >= crate::ep_vocab_shard::VOCAB_SHARD_RANKS || sampler != (rank == 0) {
        return Err(format!(
            "a vocabulary-sharded verify needs the sampler owner to be rank 0 of two ranks \
             (rank {rank}, sampler owner: {sampler})"
        ));
    }
    if n_dec == 0 || n_dec > exchange.rows() {
        return Err(format!(
            "{n_dec} decode rows do not fit the vocabulary-shard exchange ({} rows)",
            exchange.rows()
        ));
    }
    let width = exchange.width();
    let int = std::mem::size_of::<i32>();
    let (logits, top1) = if sampler {
        (
            store_logits.sub_offset(0, n_dec * width),
            store_top1.sub_offset(0, n_dec * int),
        )
    } else {
        (
            exchange
                .own_view(rank, SLOT_LOGITS)?
                .sub_offset(0, n_dec * width),
            exchange
                .own_view(rank, SLOT_TOP1)?
                .sub_offset(0, n_dec * int),
        )
    };
    Ok(ShardHead {
        exchange,
        rank,
        view: exchange.view(rank),
        logits,
        top1,
        rows: n_dec,
    })
}

/// Stage every sampled lane's draw on THIS rank's shard and run the select mode over it: the
/// jobs' row pointers are views of the rank's packed shard logits, `base` is the shard's first
/// global id and `vocab` its width, and each row's sorted top-g lands in row `r` of the rank's
/// candidate table (`r` counts sampled rows in plan order, the same on both ranks). `None` when
/// no lane is sampled. Returns the staged jobs (for the merged launch and the readback) and the
/// sampled lanes' plan indices.
fn shard_select(
    gpu: &mut Gpu,
    arena: &mut Option<LaneDrawArena>,
    plan: &VerifyPlan,
    shard: &ShardHead<'_>,
) -> R<Option<(LaneDrawStaged, Vec<usize>)>> {
    if plan.samples.iter().all(Option::is_none) {
        return Ok(None);
    }
    if plan.samples.len() != plan.lanes.len() {
        return Err(format!(
            "{} device-sample entries for {} lanes",
            plan.samples.len(),
            plan.lanes.len()
        ));
    }
    let arena = arena.get_or_insert_with(LaneDrawArena::new);
    let width = shard.view.width;
    let mut views: Vec<(usize, GpuTensor)> = Vec::new();
    let mut first = 0usize;
    for (i, vl) in plan.lanes.iter().enumerate() {
        if plan.samples[i].is_some() {
            views.push((
                i,
                shard
                    .logits
                    .sub_offset(first * width, vl.rows.len() * width),
            ));
        }
        first += vl.rows.len();
    }
    let jobs: Vec<LaneDrawJob<'_>> = views
        .iter()
        .map(|(i, view)| {
            let vl = &plan.lanes[*i];
            let spec = plan.samples[*i].as_ref().expect("view of a sampled lane");
            LaneDrawJob {
                logits: view,
                rows: vl.rows.len(),
                vocab: width,
                base: shard.view.base,
                drafts: &vl.rows[1..],
                q: &spec.q,
                history: &spec.history,
                cfg: &spec.cfg,
                rng_state: spec.rng_state,
                eos: spec.eos,
            }
        })
        .collect();
    let staged = arena
        .stage(gpu, &jobs)
        .map_err(|e| format!("vocab-shard lane select: {e}"))?;
    let cand = shard.exchange.own_view(shard.rank, SLOT_CAND)?;
    arena
        .launch_select(gpu, &staged, &cand)
        .map_err(|e| format!("vocab-shard lane select: {e}"))?;
    Ok(Some((staged, views.into_iter().map(|(i, _)| i).collect())))
}

/// What a vocabulary-sharded verify does after the shard head on one rank, in the order the
/// exchange's module docs prove ([`crate::ep_vocab_shard`]):
///
/// 1. Both ranks stage and launch the select mode over their shard for every sampled lane.
/// 2. A rank other than the sampler owner records its event on its stream, meets the barrier
///    and is done: its top-1 ids stay shard-local (evidence only) and it returns no draws.
/// 3. The sampler owner meets the barrier, makes its stream wait on the peer's event on the
///    device, merges the two shards' per-row argmax into the whole head's picks
///    (`argmax_f32_pair_merge`, written over its top-1 ids) and, for the sampled lanes, merges
///    the two candidate tables, runs the verdicts and reads the draws back.
///
/// Every rank reaches the barrier exactly once whatever the lanes (an error before it aborts
/// the mesh, which releases the other rank).
fn sharded_tail(
    gpu: &mut Gpu,
    arena: &mut Option<LaneDrawArena>,
    plan: &VerifyPlan,
    shard: &ShardHead<'_>,
    sampler: bool,
) -> R<Vec<Option<DevicePick>>> {
    let exchange = shard.exchange;
    let rank = shard.rank;
    let mut device: Vec<Option<DevicePick>> = vec![None; plan.lanes.len()];
    let selected = shard_select(gpu, arena, plan, shard)?;
    if !sampler {
        exchange
            .publish(rank, gpu)
            .map_err(|e| format!("vocab-shard publish: {e}"))?;
        exchange.barrier()?;
        return Ok(device);
    }
    exchange.barrier()?;
    exchange
        .wait_peer(rank, gpu)
        .map_err(|e| format!("vocab-shard peer wait: {e}"))?;
    let peer = exchange.peer(rank)?;
    let width = shard.view.width;
    gpu.argmax_f32_pair_merge(&ArgmaxPairMerge {
        local_logits: &shard.logits,
        local_top1: &shard.top1,
        peer_logits: peer.logits,
        peer_top1: peer.top1,
        out_top1: &shard.top1,
        rows: shard.rows,
        w0: width,
        w1: width,
    })
    .map_err(|e| format!("vocab-shard argmax merge: {e}"))?;
    if let Some((staged, lanes)) = selected {
        let arena = arena
            .as_mut()
            .ok_or_else(|| "vocab-shard select left no draw arena".to_string())?;
        let cand = exchange.own_view(rank, SLOT_CAND)?;
        arena
            .launch_merged(gpu, &staged, &cand, peer.cand)
            .map_err(|e| format!("vocab-shard lane verdict: {e}"))?;
        let draws = arena
            .readback(gpu, &staged)
            .map_err(|e| format!("vocab-shard lane verdict: {e}"))?;
        if draws.len() != lanes.len() {
            return Err(format!(
                "vocab-shard lane verdict returned {} draws for {} lanes",
                draws.len(),
                lanes.len()
            ));
        }
        for (i, draw) in lanes.into_iter().zip(draws) {
            device[i] = Some(DevicePick {
                tokens: draw.tokens,
                accepted: draw.accepted,
                advanced_rng: draw.advanced_rng,
            });
        }
    }
    Ok(device)
}
#[cfg(test)]
mod tests;
