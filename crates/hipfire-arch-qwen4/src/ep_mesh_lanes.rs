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
//!   [`RankExec`]): it replays the root's messages on its own replicated lane
//!   store, in the same order, with the same op lists, so every row gather
//!   (target and head) has a matched crossing on both ranks. A follower's
//!   answers are receipts only (state positions, the operation ledger it
//!   actually executed, evidence): `proposals`/`sampler` stay `None`.
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
//! and arms the GDN ring and takes the target ticket. `run_draft_round` runs
//! the root's ordered head `Draft` ops (one deferred head step per op, then
//! one download; only the root reads proposals). `run_verify` runs the one
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
//! preflight (the root keeps them off a window by not offering them); EP
//! sampled acceptance arrives from the root's `AcceptRule`.

use crate::bundle::{BundleError, Qwen4Bundle};
use crate::ep::{gather_tally, GatherTally};
use crate::gpu_forward_lanes::{
    forward_lanes, row_group_count, LaneOutputs, LaneRowKind, LaneRows, StageName, StageShare,
};
use crate::lane::{LaneSpec, LaneStatus, Qwen4LaneState, Qwen4LaneStore, StepParts};
use crate::lane_exec::{pick_at, LaneRoute};
use crate::mesh_lane_protocol::{
    AbortRecord, AcceptRule, CommitDecision, ControllerInterval, DraftProposal, DraftRound,
    EvidenceError, HeadState, LaneAdmission, LaneAdmit, LaneKind, LaneMeshExecutor,
    LanePlan, LaneRequest, MergedCostEvidence, MeshRoot, MeshStepKey, OpCounts,
    OpLedger, PhaseTiming, PreparedReceipt, ProposalSource, ProtocolError, RankCostEvidence,
    RankDraftResult, RankError, RankOwnership, RankReady, RankSession, RankVerifyResult,
    ReadyStatus, RootConfig, SamplerOutput, StagePolicy, StepOutput, StepPhase, TimedPhase,
    VerifyPlan, WindowBudgets, WindowStart,
};
use crate::mtp_lanes::lane_call;
use crate::mtp_spec::{
    draft_accept_estimate, prepare_native_window, publish_native_window, target_commit_accept_len,
    HeadTicket, NativeFinish, PreparedNativeWindow, Qwen4Ops, TargetTicket, DRAFT_ACCEPT_MAX,
    MTP_ROW_WORTH,
};
use hipfire_runtime::slot_batch::RequestEpoch;
use hipfire_runtime::spec::GreedyAccept;
use rdna_compute::{Gpu, GpuTensor};
use std::collections::HashSet;
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
struct LaneWin {
    plan: LanePlan,
    head_ticket: Option<HeadTicket>,
    target_ticket: Option<TargetTicket>,
    /// Head draft steps run so far this window.
    steps: usize,
    /// Probability the draft prefix is accepted (the singleton's early stop;
    /// read on the root only).
    prefix: f32,
    /// Tokens this rank's own head read, per draft step.
    local_drafts: Vec<u32>,
    /// The sealed verify rows.
    rows: Vec<u32>,
    /// First decode row of the lane among the step's decode rows.
    row_begin: usize,
    /// This rank's own target picks (the root's are the decision's input).
    picks: Vec<u32>,
    prepared: Option<PreparedNativeWindow>,
}

impl LaneWin {
    fn new(plan: &LanePlan) -> Self {
        Self {
            plan: plan.clone(),
            head_ticket: None,
            target_ticket: None,
            steps: 0,
            prefix: 1.0,
            local_drafts: Vec::new(),
            rows: Vec::new(),
            row_begin: 0,
            picks: Vec::new(),
            prepared: None,
        }
    }

    fn native(&self) -> bool {
        self.plan.source == ProposalSource::Native
    }

    /// The lane's head takes an AR-row append after its window (a spec lane's
    /// `ArSynced` row; a retired lane whose head the root keeps aligned).
    fn ar_head_append(&self) -> bool {
        !self.native()
            && self.plan.head == HeadState::Aligned
            && matches!(
                self.plan.route,
                LaneRoute::ArSynced | LaneRoute::ArRetired
            )
    }
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
    row_groups: usize,
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
        }
    }

    pub fn stats(&self) -> RankLaneStats {
        self.stats
    }

    /// Why this rank was aborted (idempotent), if it was.
    pub fn abort_reason(&self) -> Option<&str> {
        self.abort_reason.as_deref()
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
fn release_lane_tickets(bundle: &mut Qwen4Bundle, lane: &mut Qwen4LaneState, win: &mut LaneWin) {
    let Qwen4LaneState { state, spec, .. } = lane;
    state.row_capture_armed = false;
    if let Some(TargetTicket::Local(ticket)) = win.target_ticket.take() {
        let _ = bundle.with_lane(state, None, |b| {
            if b.validate_commit(ticket).is_ok() {
                b.commit_validated(ticket);
            }
        });
    }
    if let (Some(HeadTicket::Local(ticket)), Some(spec)) = (win.head_ticket.take(), spec.as_mut())
    {
        let LaneSpec { head, draft, .. } = spec;
        let _ = bundle.with_lane(state, Some((head, draft)), |b| {
            if b.mtp_validate_commit(ticket).is_ok() {
                b.mtp_commit_validated(ticket);
            }
        });
    }
}

impl RankLane<'_> {
    fn rank(&self) -> usize {
        self.exec.rank
    }

    fn own(&self) -> RankOwnership {
        self.exec.session.ownership()
    }

    // ---- begin_window ---------------------------------------------------

    /// The host-and-mapping checks of a plan: nothing here changes lane state
    /// (mapping only grows capacity).
    fn preflight(&mut self, start: &WindowStart) -> R<()> {
        let Self { bundle, gpu, .. } = self;
        let store = bundle
            .lanes
            .as_mut()
            .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
        let spec_k = store.spec().map(|spec| spec.k);
        let rows_cap = store.prefill_logits_row();
        if start.lanes.len() > store.max_lanes() {
            return Err(format!(
                "{} lanes exceed the staged {}",
                start.lanes.len(),
                store.max_lanes()
            ));
        }
        let total: usize = start.lanes.iter().map(LanePlan::max_rows).sum();
        if total > rows_cap {
            return Err(format!(
                "{total} verify rows exceed the store's {rows_cap} step rows"
            ));
        }
        for plan in &start.lanes {
            let epoch = plan.key.request_epoch;
            if matches!(plan.kind, LaneKind::Forced(_)) {
                return Err(format!(
                    "{epoch:?}: forced advances are not supported by the EP lane executor"
                ));
            }
            if plan.source == ProposalSource::Copy {
                return Err(format!(
                    "{epoch:?}: copy proposals are not supported by the EP lane executor"
                ));
            }
            if matches!(plan.head, HeadState::Lag { .. }) {
                return Err(format!(
                    "{epoch:?}: head-lag lanes are not supported by the EP lane executor"
                ));
            }
            let lane = store
                .lane_mut(&epoch)
                .ok_or_else(|| format!("unknown or stale epoch {epoch:?}"))?;
            if lane.host.status != LaneStatus::Decoding {
                return Err(format!(
                    "{epoch:?} is {:?}, not Decoding",
                    lane.host.status
                ));
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
            let needs_head = plan.head == HeadState::Aligned
                && (plan.source == ProposalSource::Native
                    || matches!(plan.route, LaneRoute::ArSynced | LaneRoute::ArRetired));
            match lane.spec.as_mut() {
                None => {
                    if plan.source != ProposalSource::None || needs_head {
                        return Err(format!("{epoch:?}: an AR lane cannot run a head window"));
                    }
                }
                Some(spec) => {
                    if spec_k.is_none() {
                        return Err(format!("{epoch:?}: lane has MTP state in an AR store"));
                    }
                    if spec.window.is_some() {
                        return Err(format!(
                            "{epoch:?}: an unconsumed legacy window is pending"
                        ));
                    }
                    if plan.source == ProposalSource::Native {
                        if plan.max_depth > spec_k.unwrap_or(0) {
                            return Err(format!(
                                "{epoch:?}: depth {} exceeds the staged k {:?}",
                                plan.max_depth, spec_k
                            ));
                        }
                        if lane.state.row_capture_rows() < plan.max_rows() {
                            return Err(format!(
                                "{epoch:?}: ring holds {} rows, the window needs {}",
                                lane.state.row_capture_rows(),
                                plan.max_rows()
                            ));
                        }
                    }
                    if needs_head {
                        if spec.head.position() != plan.position {
                            return Err(format!(
                                "{epoch:?}: head position {} != lane position {}",
                                spec.head.position(),
                                plan.position
                            ));
                        }
                        spec.head.ensure_mapped_capacity(gpu, end).map_err(|e| {
                            format!("{epoch:?}: map {end} head positions: {e}")
                        })?;
                    }
                }
            }
            lane.state
                .ensure_mapped_capacity(gpu, end)
                .map_err(|e| format!("{epoch:?}: map {end} positions: {e}"))?;
        }
        Ok(())
    }

    /// Take the head ticket and arm the ring, then the target ticket, of every
    /// native lane.
    fn arm(&mut self, start: &WindowStart) -> R<Vec<LaneWin>> {
        let Self { bundle, gpu, .. } = self;
        let mut wins: Vec<LaneWin> = start.lanes.iter().map(LaneWin::new).collect();
        let result = with_store(bundle, |store, bundle| {
            for win in wins.iter_mut().filter(|w| w.native()) {
                let epoch = win.plan.key.request_epoch;
                let lane = lane_mut(store, &epoch)?;
                let Qwen4LaneState { state, spec, .. } = lane;
                let spec = spec
                    .as_mut()
                    .ok_or_else(|| format!("{epoch:?}: native window on an AR lane"))?;
                let LaneSpec { head, draft, .. } = spec;
                let head_ticket = lane_call(bundle.with_lane(
                    state,
                    Some((head, draft)),
                    |b| b.mtp_snapshot(gpu),
                ))
                .map_err(|e| format!("{epoch:?}: head snapshot: {e}"))?;
                win.head_ticket = Some(HeadTicket::Local(head_ticket));
                // Armed before the snapshot: the GDN states then need no copy.
                state.row_capture_armed = true;
                match lane_call(bundle.with_lane(state, None, |b| b.snapshot(gpu))) {
                    Ok(ticket) => win.target_ticket = Some(TargetTicket::Local(ticket)),
                    Err(error) => {
                        state.row_capture_armed = false;
                        return Err(format!("{epoch:?}: target snapshot: {error}"));
                    }
                }
            }
            Ok(())
        });
        match result {
            Ok(()) => Ok(wins),
            Err(error) => {
                self.release_wins(&mut wins);
                Err(error)
            }
        }
    }

    /// Unwind every ticket of `wins` (an unexecuted or aborted window).
    fn release_wins(&mut self, wins: &mut [LaneWin]) {
        let Self { bundle, .. } = self;
        let _ = with_store(bundle, |store, bundle| {
            for win in wins.iter_mut() {
                if let Some(lane) = store.lane_mut(&win.plan.key.request_epoch) {
                    release_lane_tickets(bundle, lane, win);
                }
            }
            Ok(())
        });
    }

    // ---- draft ----------------------------------------------------------

    /// One head round on this rank: enqueue every op's deferred head step in
    /// the root's order, then one download for all of them and the head's own
    /// parse of each. Returns `(token, margin)` per op.
    fn draft_round_device(&mut self, round: &DraftRound) -> R<Vec<(u32, f32)>> {
        let Self {
            exec, bundle, gpu, ..
        } = self;
        let step = exec
            .step
            .as_ref()
            .ok_or_else(|| "draft round without a begun window".to_string())?;
        let mut epochs = Vec::with_capacity(round.ops.len());
        let mut windows = Vec::with_capacity(round.ops.len());
        for op in &round.ops {
            let win = step
                .lanes
                .iter()
                .find(|w| w.plan.key == op.key)
                .ok_or_else(|| "draft op for a lane outside the plan".to_string())?;
            epochs.push(op.key.request_epoch);
            windows.push(win.plan.position);
        }
        with_store(bundle, |store, bundle| {
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
            for (slot, op) in round.ops.iter().enumerate() {
                let Qwen4LaneState { state, spec, .. } = &mut *lanes[slot];
                let spec = spec
                    .as_mut()
                    .ok_or_else(|| "drafting lane lost its MTP state".to_string())?;
                let LaneSpec {
                    head,
                    draft,
                    pending_hidden,
                    ..
                } = spec;
                let top1 = spec_store.draft_top1.sub_offset(slot * slot_bytes, slot_bytes);
                let j = op.position - windows[slot];
                let hidden = (j == 0).then_some(&*pending_hidden);
                lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
                    b.mtp_forward_token_deferred(gpu, op.input, hidden, op.position, j == 0, &top1)
                }))
                .map_err(|e| format!("draft step {j}: {e}"))?;
            }
            gpu.hip
                .memcpy_dtoh(
                    &mut spec_store.draft_top1_host[..bytes],
                    &spec_store.draft_top1.buf,
                )
                .map_err(|e| format!("draft download: {e}"))?;
            gpu.take_device_wait_error()
                .map_err(|e| format!("draft download: {e}"))?;
            let mut reads = Vec::with_capacity(round.ops.len());
            for slot in 0..round.ops.len() {
                let top = &spec_store.draft_top1_host[slot * slot_bytes..(slot + 1) * slot_bytes];
                let Qwen4LaneState { state, spec, .. } = &mut *lanes[slot];
                let spec = spec
                    .as_mut()
                    .ok_or_else(|| "drafting lane lost its MTP state".to_string())?;
                let LaneSpec { head, draft, .. } = spec;
                let read = lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
                    let token = b.mtp_draft_finish(top)?;
                    Ok::<_, BundleError>((token, b.mtp_draft_margin()))
                }))
                .map_err(|e| format!("draft finish: {e}"))?;
                reads.push(read);
            }
            Ok(reads)
        })
    }

    // ---- verify ---------------------------------------------------------

    /// The one trunk forward over every lane's sealed rows; the picks per lane
    /// and each lane's resulting target position.
    fn verify_device(&mut self, plan: &VerifyPlan) -> R<(Vec<Vec<u32>>, Vec<usize>)> {
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
            let forwarded = {
                let mut rows: Vec<LaneRows<'_>> = Vec::with_capacity(lanes.len());
                for (lane, v) in lanes.iter_mut().zip(&plan.lanes) {
                    rows.push(LaneRows {
                        state: &mut lane.state,
                        tokens: &v.rows,
                        kind: if v.rows.len() == 1 {
                            LaneRowKind::Ar
                        } else {
                            LaneRowKind::Verify {
                                draft_len: v.rows.len() - 1,
                            }
                        },
                    });
                }
                let out = LaneOutputs {
                    logits: &logits.sub_offset(0, n_dec * vocab),
                    top1: &top1.sub_offset(0, n_dec * std::mem::size_of::<i32>()),
                    wide_hidden: hidden,
                };
                forward_lanes(gpu, bundle, &mut rows, &out, &policy).map_err(|e| e.to_string())
            };
            // The ring is disarmed on every path out of the forward.
            for lane in lanes.iter_mut() {
                lane.state.row_capture_armed = false;
            }
            forwarded?;
            gpu.hip
                .memcpy_dtoh(&mut top1_host[..n_dec * std::mem::size_of::<i32>()], &top1.buf)
                .map_err(|e| e.to_string())?;
            gpu.take_device_wait_error().map_err(|e| e.to_string())?;
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
            Ok((picks, ends))
        })
    }

    // ---- prepare / publish ------------------------------------------------

    /// Repair one lane to its decision: a native window's target rollback and
    /// head rebuild ([`prepare_native_window`], tickets kept live), or an AR
    /// lane's head append.
    fn prepare_lane(&mut self, index: usize, d: &crate::mesh_lane_protocol::LaneDecision) -> R<()> {
        let Self {
            exec, bundle, gpu, ..
        } = self;
        let step = exec
            .step
            .as_mut()
            .ok_or_else(|| "prepare without a begun window".to_string())?;
        let win = &mut step.lanes[index];
        let epoch = win.plan.key.request_epoch;
        let (row_begin, rows) = (win.row_begin, win.rows.len());
        let wide = bundle
            .config
            .hc_count
            .checked_mul(bundle.config.hidden_size)
            .ok_or_else(|| "hidden width overflow".to_string())?;
        let want_native = win.native();
        let want_ar_head = win.ar_head_append();
        let (position, seed) = (win.plan.position, win.plan.seed);
        let steps = win.steps;
        let block = win.rows.clone();
        let tickets = (win.target_ticket, win.head_ticket);
        let prepared = with_store(bundle, |store, bundle| {
            let hidden_tensor: Option<GpuTensor> = store
                .step_parts(&[epoch])
                .ok()
                .and_then(|parts| parts.spec.map(|spec| spec.hidden.sub_offset(0, spec.hidden.numel())));
            let lane = lane_mut(store, &epoch)?;
            let Qwen4LaneState { state, spec, .. } = lane;
            if want_native {
                let spec = spec
                    .as_mut()
                    .ok_or_else(|| format!("{epoch:?}: native window on an AR lane"))?;
                let LaneSpec {
                    head,
                    draft,
                    pending_hidden,
                    ..
                } = spec;
                let hidden = hidden_tensor
                    .as_ref()
                    .ok_or_else(|| "verify without a hidden capture".to_string())?;
                let (Some(target_snapshot), Some(mtp_ticket)) = tickets else {
                    return Err(format!("{epoch:?}: window holds no tickets"));
                };
                // I1 at prepare: the verify published every sealed row, the
                // head ran `steps` draft steps from the window's frontier.
                if state.position != position + rows {
                    return Err(format!(
                        "{epoch:?}: I1: trunk position {} != {} after the verify",
                        state.position,
                        position + rows
                    ));
                }
                if head.position() != position + steps {
                    return Err(format!(
                        "{epoch:?}: I1: head position {} != {} after {steps} draft steps",
                        head.position(),
                        position + steps
                    ));
                }
                let verify_hidden = hidden.sub_offset(row_begin * wide, rows * wide);
                // Taken before `with_lane`: it names the forward scratch.
                let ple_normed = bundle
                    .lane_ple_normed_rows(row_begin, rows)
                    .map_err(|e| format!("{epoch:?}: PLE rows: {e}"))?;
                let acceptance = GreedyAccept {
                    committed: d.emitted.clone(),
                    accepted: d.accepted,
                    hit_eos: d.hit_eos,
                };
                if 1 + target_commit_accept_len(&acceptance) != d.consumed_rows {
                    return Err(format!(
                        "{epoch:?}: decision consumes {} rows, its acceptance implies {}",
                        d.consumed_rows,
                        1 + target_commit_accept_len(&acceptance)
                    ));
                }
                let native = lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
                    prepare_native_window(
                        b,
                        &mut Qwen4Ops::Single(&mut *gpu),
                        NativeFinish {
                            position,
                            block: &block,
                            acceptance,
                            target_snapshot,
                            mtp_ticket,
                            pending_hidden,
                            verify_hidden: &verify_hidden,
                            ple_normed: Some(&ple_normed),
                            retain: false,
                            head_at_start: false,
                        },
                        None,
                    )
                    .map_err(BundleError::Forward)
                }))
                .map_err(|e| format!("{epoch:?}: prepare: {e}"))?;
                Ok(Some(native))
            } else if want_ar_head {
                let spec = spec
                    .as_mut()
                    .ok_or_else(|| format!("{epoch:?}: head append on an AR lane"))?;
                let LaneSpec {
                    head,
                    draft,
                    pending_hidden,
                    ..
                } = spec;
                let hidden = hidden_tensor
                    .as_ref()
                    .ok_or_else(|| "verify without a hidden capture".to_string())?;
                if state.position != position + 1 {
                    return Err(format!(
                        "{epoch:?}: lane position {} != {} after the AR row",
                        state.position,
                        position + 1
                    ));
                }
                let row = hidden.sub_offset(row_begin * wide, wide);
                // Fills use the same (token p, target hidden p - 1) pairing as
                // drafting; `pending_hidden` stays intact until the append
                // consumed it.
                lane_call(bundle.with_lane(state, Some((&mut *head, draft)), |b| {
                    b.mtp_append_token(gpu, seed, Some(pending_hidden), position)
                }))
                .map_err(|e| format!("{epoch:?}: head append: {e}"))?;
                if head.position() != position + 1 {
                    return Err(format!(
                        "{epoch:?}: head ended at {}, expected {}",
                        head.position(),
                        position + 1
                    ));
                }
                gpu.copy_d2d(&row, pending_hidden, pending_hidden.byte_size())
                    .map_err(|e| format!("{epoch:?}: pending hidden copy: {e}"))?;
                Ok(None)
            } else {
                Ok(None)
            }
        })?;
        win.prepared = prepared;
        Ok(())
    }
}

/// Window state to evidence: the counts this rank actually executed.
fn rank_counts(step: &ExecStep, rank_head: bool) -> OpCounts {
    let totals = step.totals.unwrap_or((0, 0, 0, 0));
    OpCounts {
        target_exchanges: step.verify_tally.trunk as usize,
        gather_bytes: step.draft_tally.bytes + step.verify_tally.bytes,
        head_draft_calls: if rank_head { step.head_calls } else { 0 },
        head_append_rows: if rank_head { totals.3 } else { 0 },
        row_groups: step.row_groups,
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
        let mut ready = self.exec.session.begin(start, capacity);
        if !matches!(ready.status, ReadyStatus::Ready { .. }) {
            return ready;
        }
        match self.arm(start) {
            Ok(lanes) => {
                let unpriced = !self.exec.stepped
                    || lanes
                        .iter()
                        .filter(|w| w.native())
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
                    row_groups: 0,
                    totals: None,
                    completed: StepPhase::Armed,
                    unpriced,
                });
            }
            Err(reason) => {
                self.exec.session.reject(&start.key);
                ready.status = ReadyStatus::Rejected { reason };
            }
        }
        ready
    }

    fn reject_window(&mut self, key: &MeshStepKey) {
        self.exec.session.reject(key);
        if let Some(mut step) = self.exec.step.take() {
            if step.key == *key {
                self.release_wins(&mut step.lanes);
            } else {
                self.exec.step = Some(step);
            }
        }
    }

    fn run_draft_round(&mut self, round: &DraftRound) -> Result<RankDraftResult, RankError> {
        let rank = self.rank();
        let sampler = self.own().sampler;
        let mut result = self.exec.session.on_draft_round(round)?;
        let before = gather_tally();
        let t0 = Instant::now();
        let reads = self
            .draft_round_device(round)
            .map_err(|e| rank_err(rank, e))?;
        let tally = gather_tally().since(before);
        let Some(step) = self.exec.step.as_mut() else {
            return Err(rank_err(rank, "draft round without a begun window"));
        };
        step.draft_us += t0.elapsed().as_secs_f64() * 1e6;
        step.draft_tally = GatherTally {
            trunk: step.draft_tally.trunk + tally.trunk,
            head: step.draft_tally.head + tally.head,
            bytes: step.draft_tally.bytes + tally.bytes,
        };
        step.head_calls += round.ops.len();
        result.head_gathers = tally.head as usize;
        let mut proposals = Vec::with_capacity(round.ops.len());
        for (op, (token, margin)) in round.ops.iter().zip(reads) {
            let win = step
                .lanes
                .iter_mut()
                .find(|w| w.plan.key == op.key)
                .ok_or_else(|| rank_err(rank, "draft op for a lane outside the plan"))?;
            let j = win.steps;
            win.steps += 1;
            win.local_drafts.push(token);
            // The singleton's early stop, statement for statement: a token
            // after the first whose prefix acceptance fell below the row worth
            // is drafted but not kept; deeper drafting stops once even the
            // best per-draft acceptance cannot make the next row worth it.
            win.prefix *= draft_accept_estimate(margin);
            let drop_last = j > 0 && win.prefix < MTP_ROW_WORTH;
            let keep_going = !drop_last && win.prefix * DRAFT_ACCEPT_MAX >= MTP_ROW_WORTH;
            proposals.push(DraftProposal {
                key: op.key,
                token,
                keep_going,
                drop_last,
                // Greedy lanes propose the head's argmax; a sampled lane's
                // distribution read (S4) is the sampler owner's alone.
                read: None,
            });
        }
        if sampler {
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
        let (picks, ends) = self.verify_device(plan).map_err(|e| rank_err(rank, e))?;
        let tally = gather_tally().since(before);
        let row_counts: Vec<usize> = plan.lanes.iter().map(|l| l.rows.len()).collect();
        let Some(step) = self.exec.step.as_mut() else {
            return Err(rank_err(rank, "verify without a begun window"));
        };
        step.verify_us += t0.elapsed().as_secs_f64() * 1e6;
        step.verify_tally = tally;
        step.row_groups = row_group_count(&row_counts);
        step.completed = StepPhase::Verified;
        let mut first = 0usize;
        let mut mismatches = 0u64;
        for ((win, v), lane_picks) in step.lanes.iter_mut().zip(&plan.lanes).zip(&picks) {
            win.rows = v.rows.clone();
            win.row_begin = first;
            win.picks = lane_picks.clone();
            first += v.rows.len();
            // Evidence only: the rank's own head read vs the root's kept token.
            for (kept, local) in v.rows[1..].iter().zip(&win.local_drafts) {
                if kept != local {
                    mismatches += 1;
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
            row_groups: self
                .exec
                .step
                .as_ref()
                .map_or(0, |s| s.row_groups),
            target_exchanges: tally.trunk as usize,
            microbatches: 0,
            head_draft_calls: plan.ledger.head_draft_calls,
        };
        if sampler {
            result.sampler = Some(SamplerOutput::picks_only(
                plan.lanes
                    .iter()
                    .zip(picks)
                    .map(|(l, p)| (l.key, p))
                    .collect(),
            ));
        }
        Ok(result)
    }

    fn prepare_commit(&mut self, decision: &CommitDecision) -> Result<PreparedReceipt, RankError> {
        let rank = self.rank();
        let receipt = self.exec.session.on_prepare(decision)?;
        let t0 = Instant::now();
        let mut mismatches = 0u64;
        for (index, d) in decision.lanes.iter().enumerate() {
            self.prepare_lane(index, d).map_err(|e| rank_err(rank, e))?;
            // Evidence only: the emitted tokens against this rank's own picks.
            if let Some(win) = self.exec.step.as_ref().map(|s| &s.lanes[index]) {
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
        let rank = self.rank();
        let Some(mut step) = self.exec.step.take() else {
            return;
        };
        let Self {
            exec, bundle, gpu, ..
        } = self;
        let published = with_store(bundle, |store, bundle| {
            for (win, d) in step.lanes.iter_mut().zip(&decision.lanes) {
                let epoch = win.plan.key.request_epoch;
                let lane = lane_mut(store, &epoch)?;
                if let Some(prepared) = win.prepared.take() {
                    let Qwen4LaneState { state, spec, .. } = &mut *lane;
                    let spec = spec
                        .as_mut()
                        .ok_or_else(|| format!("{epoch:?}: native window on an AR lane"))?;
                    let LaneSpec { head, draft, .. } = spec;
                    lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
                        publish_native_window(b, &mut Qwen4Ops::Single(&mut *gpu), prepared)
                            .map_err(BundleError::Forward)
                    }))
                    .map_err(|e| format!("{epoch:?}: publish: {e}"))?;
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
        exec.stats.native_windows += step.lanes.iter().filter(|w| w.native()).count() as u64;
        for win in step.lanes.iter().filter(|w| w.native()) {
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
        if self.exec.abort_reason.is_none() {
            self.exec.abort_reason = Some(reason.to_string());
        }
        self.exec.session.on_abort();
        // A half-run window's device state is untrusted: poison every lane of
        // the step (store.poison drops the legacy window, disarms the ring);
        // the mesh is quarantined, nothing is retried in place.
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
        if let Err(error) = root.collect_draft(&round, &results) {
            return Err(abort_after(root, fan, &key, error));
        }
    }
    let plan = match root.seal_verify() {
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
    let fatal: Vec<String> = fan.publish(&decision).into_iter().flatten().collect();
    if !fatal.is_empty() {
        // The step is already published at the root (no step is in flight):
        // quarantine the mesh with no step key and make every rank follow.
        let reason = format!("publish failed after all ranks prepared: {}", fatal.join("; "));
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
                    let exec = mirrors
                        .mesh_lanes
                        .get_or_insert_with(|| RankExec::new(RankOwnership::ep(rank)));
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
        self.fan(|lane| f(lane).map_err(named), &|lane| f(lane).map_err(named))
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
                let exec = mirrors
                    .mesh_lanes
                    .get_or_insert_with(|| RankExec::new(RankOwnership::ep(rank)));
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
}

/// The EP mesh's lane transaction owner: the root plus rank 0's executor
/// state. One per staged EP lane store; the lanes' prefill runs through the
/// ordinary lane executor, decode through [`Self::step`].
pub struct EpMeshLanes {
    root: MeshRoot,
    rank0: RankExec,
    ranks: usize,
    pending: Option<(MeshStepKey, Vec<RankCostEvidence>)>,
    interval_start: Option<Instant>,
    prefill_ran: bool,
    last_merged: Option<MergedCostEvidence>,
    steps: u64,
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
    /// The root and rank 0's state for `bundle`'s staged EP lane store.
    /// `topology_generation` names this mesh load; `floor_enabled` is the
    /// production floor (lab route modes ignore it).
    pub fn new(
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
        let moe_layers = bundle
            .execution
            .as_ref()
            .map(|forward| forward.moe_count())
            .ok_or_else(|| "Qwen4 forward resources are not attached".to_string())?;
        let moe_once = store.stage_policy().share(StageName::Moe) == StageShare::OnceAllRows;
        let world: Vec<RankOwnership> = (0..ranks).map(RankOwnership::ep).collect();
        let root = MeshRoot::new(
            RankOwnership::ep(0),
            &world,
            RootConfig {
                topology_generation,
                stage: StagePolicy {
                    moe_layers,
                    moe_once,
                    stream_bf16: false,
                },
                floor_enabled,
            },
        )
        .map_err(|e| e.to_string())?;
        Ok(Self {
            root,
            rank0: RankExec::new(RankOwnership::ep(0)),
            ranks,
            pending: None,
            interval_start: None,
            prefill_ran: false,
            last_merged: None,
            steps: 0,
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

    fn fan<'a>(&'a mut self, bundle: &'a mut Qwen4Bundle, gpu: &'a mut Gpu) -> (EpFanout<'a>, &'a mut MeshRoot) {
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
        fan.soft(&|lane| lane.exec.stats())
            .map_err(|e| e.reason)
    }
}

#[cfg(test)]
mod tests;
