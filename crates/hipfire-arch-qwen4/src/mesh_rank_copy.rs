// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The device side of the root's copy windows, forced advances, head lag,
//! round-0 catch-up and first-candidate confirmation
//! (`release-0.4.3/mgmtp/DESIGN.md` §4 "Native repair details", §5.2;
//! `INTERFACES.md` §9.3), written once over one lane's own states so the
//! single-card rank ([`crate::lane_rank::LocalLaneRank`]), an EP rank
//! (`ep_mesh_lanes`) and a PP output band can all call it.
//!
//! Every function takes `(bundle, gpu, lane, ..)`: the bundle's weights and
//! forward program, the device, and ONE [`Qwen4LaneState`] (its trunk state,
//! head state, pending hidden, owed-row storage and pending window). Nothing
//! here reaches for the single-card store, its step buffers or its
//! `LaneProto`: the caller owns the step's wide-hidden capture
//! (`hidden`, the lane's rows starting at `row_begin`) and hands the root's
//! plan/decision lane in.
//!
//! # What the rank executes
//!
//! | root plan | rank work |
//! |---|---|
//! | `catch_up` (round 0) | [`catch_up_lane`] before any Draft op, then the head ticket ([`take_head_ticket`]) |
//! | `copy_confirm` | the round-0 Draft at the seed is the *probe*; [`settle_probe`] restores the head on a hit or on a miss with no native route |
//! | copy window (no head Draft) | `[seed, copy_drafts..]` verify rows, ring armed, target + head ticket at begin; prepare with `head_at_start = (steps == 0)` |
//! | forced advance | `[seed, f1..f_{n-1}]`, all consumed, the acceptance is synthesized (no picks, no RNG), the head is appended as the singleton's forced advance does |
//! | retired AR token with a copy future | [`owe_lane_head_row`]: the target hidden goes to the lane's owed-row storage, the head is untouched |
//! | owed rows at prepare | [`flush_owed_head_rows`] before the window's own head work |
//!
//! # `head_at_start` (the rule, derived from the singleton)
//!
//! `NativeFinish.head_at_start` says the head sits at the window's *start*
//! (nothing was drafted, or the draft was abandoned), so the finish appends
//! the seed's row instead of truncating to the draft chain's row 0.
//! Singleton `run_takeover` never drafts inside the takeover: its
//! confirmation probe (`takeover_probe`) is snapshot / Predict / **restore**,
//! so the head is back at the window start when the takeover begins, and
//! `takeover_batched` teacher-forces every consumed row with
//! `fill_head_rows(.., &block[..consumed], position)`. The lane rule is the
//! same fact stated on the head's state, not on the window's kind:
//!
//! > `head_at_start = (steps == 0)`, where `steps` is the number of head
//! > Draft steps still *in effect* in the window: [`settle_probe`] restores
//! > the head and zeroes `steps` for an abandoned probe (a confirmed copy
//! > window, or a miss with no native route), so the confirmed-copy,
//! > established-copy and forced windows all finish from the start; a native
//! > window (and a missed probe that continues natively) has `steps >= 1`
//! > and keeps its draft chain's row 0.
//!
//! A window with `k == 0` verified drafts and `steps == 0` is the AR row; it
//! never reaches [`prepare_native_window`](crate::mtp_spec::prepare_native_window).

use crate::bundle::{BundleError, Qwen4Bundle};
use crate::config::Qwen4Config;
use crate::kv_backend::Qwen4ContextCommit;
use crate::lane::census::LaneSpecBudget;
use crate::lane::{LaneCopyConfig, LaneSpec, LaneSpecConfig, Qwen4LaneState};
use crate::mesh_lane_protocol::{
    CatchUp, HeadAppend, HeadState, LaneDecision, LaneKind, LanePlan, ProposalSource, SamplingMode,
    VerifyLane,
};
use crate::mtp_lanes::{
    ar_synced_head, lane_call, prepare_lane_window_with, publish_lane_window, LaneEos, LaneFinish,
    PendingWindow, PreparedLaneWindow,
};
use crate::mtp_spec::{qwen4_ngram_take_cap, HeadLag, HeadTicket, TargetTicket};
use hipfire_dispatch::pipeline::DraftHeadRequestState;
use hipfire_runtime::spec::GreedyAccept;
use rdna_compute::{DType, Gpu, GpuTensor};

// ---------------------------------------------------------------------------
// Host rules (pure: unit-tested on the CPU)
// ---------------------------------------------------------------------------

/// What the round-0 head Draft at the seed (the confirmation *probe* of an
/// unestablished copy source) decided. The root decides the same fact from
/// the same inputs (`collect_draft`: `first_candidate == Some(p.token)`, and
/// `hit || native_depth == 0` abandons the head's draft); the rank applies it
/// at once, the way the singleton restores right after its probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProbeOutcome {
    /// The head's argmax equalled the first candidate: a copy window, the
    /// probe's head state is abandoned.
    Confirmed,
    /// A miss with a native route: the probe is the first native draft and
    /// drafting continues.
    MissedNative,
    /// A miss with no native route (`native_depth == 0`): the window is an AR
    /// row and the probe is abandoned.
    MissedAr,
}

impl ProbeOutcome {
    /// The probe's head step is abandoned: the head returns to its ticket
    /// (and its request state to the pre-probe one) and `steps` is back to 0.
    pub(crate) fn restores_head(self) -> bool {
        !matches!(self, Self::MissedNative)
    }
}

/// [`ProbeOutcome`] of a probe token against the plan's candidates.
pub(crate) fn probe_outcome(
    copy_drafts: &[u32],
    probe_token: u32,
    native_depth: usize,
) -> ProbeOutcome {
    if copy_drafts.first() == Some(&probe_token) {
        ProbeOutcome::Confirmed
    } else if native_depth > 0 {
        ProbeOutcome::MissedNative
    } else {
        ProbeOutcome::MissedAr
    }
}

/// `NativeFinish.head_at_start` of a window whose head ran `steps` Draft
/// steps that are still in effect (see the module docs).
pub(crate) fn head_at_start(steps: usize) -> bool {
    steps == 0
}

/// Whether `op` (a draft op of `plan` at `position`) is the confirmation
/// probe: the first head op of a copy plan that asks for one.
pub(crate) fn is_probe_op(plan: &LanePlan, position: usize) -> bool {
    plan.source == ProposalSource::Copy && plan.copy_confirm && position == plan.position
}

/// Most proposals the head keeps in a window of `plan` (the native route's
/// depth for a copy plan whose fallback it is, else the plan's own).
pub(crate) fn native_keep_cap(plan: &LanePlan) -> usize {
    if plan.source == ProposalSource::Copy {
        plan.native_depth
    } else {
        plan.max_depth
    }
}

/// When a lane's head ticket (the head repair point of `prepare_commit`) is
/// taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeadTicketPoint {
    /// The window cannot verify more than one row, or the head is free: no
    /// ticket (an AR row's head work is append-only).
    Never,
    /// At `begin_window`, the head as it stands (no catch-up precedes it).
    AtBegin,
    /// In round 0, after the plan's catch-up and before any Draft op
    /// (`CatchUp` docs: the ticket is the head right after the catch-up).
    AfterCatchUp,
}

/// The ticket point of `plan`'s head.
pub(crate) fn head_ticket_point(plan: &LanePlan) -> HeadTicketPoint {
    if plan.head == HeadState::Free {
        return HeadTicketPoint::Never;
    }
    let multi_row = match &plan.kind {
        LaneKind::Forced(t) => t.len() > 1,
        LaneKind::Decode => matches!(plan.source, ProposalSource::Native | ProposalSource::Copy),
    };
    match (multi_row, plan.catch_up.is_some()) {
        (false, _) => HeadTicketPoint::Never,
        (true, true) => HeadTicketPoint::AfterCatchUp,
        (true, false) => HeadTicketPoint::AtBegin,
    }
}

/// Where the lane's head must stand when the window begins: the window's
/// position, or the lag's start (before a catch-up or a prepare-time flush).
pub(crate) fn expected_head_position(plan: &LanePlan) -> Option<usize> {
    match plan.head {
        HeadState::Aligned => Some(plan.position),
        HeadState::Lag { from } => Some(from),
        HeadState::Free => None,
    }
}

/// What `prepare_commit` does to one lane's head and target, from the plan,
/// the sealed rows and the decision's head schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeadPrepare {
    /// No head work (a head-free lane): the target alone is repaired.
    TargetOnly,
    /// A one-row window whose token owes its head row: the target hidden goes
    /// to the lane's owed-row storage; the head is not touched.
    OweRow,
    /// A one-row window the head keeps up with: flush `owed` rows, then
    /// append the seed and take the AR row's hidden as pending
    /// ([`ar_synced_head`]).
    SyncRow { owed: usize },
    /// A verify window: flush `owed` rows, then the finish with
    /// `head_at_start` and `consumed` retained rows.
    Window { owed: usize, head_at_start: bool },
}

/// The head half of `prepare_commit`'s schedule: the one place the rank
/// decides what it does with `append`. Errors name a schedule the plan does
/// not allow (the session already refuses the shapes it can check; this is
/// the rank's own re-check against what it holds).
pub(crate) fn head_prepare(
    head: HeadState,
    rows: usize,
    consumed: usize,
    append: HeadAppend,
    steps: usize,
    owed_held: usize,
) -> Result<HeadPrepare, String> {
    if head == HeadState::Free {
        return if append == HeadAppend::default() {
            Ok(HeadPrepare::TargetOnly)
        } else {
            Err(format!(
                "head schedule {append:?} on a head-free lane (the lane has no head rows to append)"
            ))
        };
    }
    // A schedule that appends nothing leaves every held row owed (the AR row
    // owes its own); one that appends flushes exactly the rows the rank holds
    // (the record is the root's, row for row).
    if append.retained == 0 {
        if append.owed != 0 || rows != 1 {
            return Err(format!(
                "a {rows}-row window cannot schedule {append:?} head rows"
            ));
        }
        return Ok(HeadPrepare::OweRow);
    }
    if append.owed != owed_held {
        return Err(format!(
            "head schedule flushes {} owed rows, the rank holds {owed_held}",
            append.owed
        ));
    }
    if rows == 1 {
        return if append.retained == 1 {
            Ok(HeadPrepare::SyncRow { owed: append.owed })
        } else {
            Err(format!(
                "a one-row window cannot append {} head rows",
                append.retained
            ))
        };
    }
    if append.retained != consumed {
        return Err(format!(
            "a {rows}-row window retains {} head rows, it consumed {consumed}",
            append.retained
        ));
    }
    Ok(HeadPrepare::Window {
        owed: append.owed,
        head_at_start: head_at_start(steps),
    })
}

/// Head rows a prepared lane appended, for the receipt's `head_append_rows`
/// (the catch-up rows of round 0 are counted where they run).
pub(crate) fn prepared_head_rows(prep: HeadPrepare, consumed: usize) -> usize {
    match prep {
        HeadPrepare::TargetOnly | HeadPrepare::OweRow => 0,
        HeadPrepare::SyncRow { owed } => owed + 1,
        HeadPrepare::Window { owed, .. } => owed + consumed,
    }
}

/// The acceptance a forced window finishes with: every forced row is kept,
/// nothing is sampled and no RNG is drawn (`LaneKind::Forced(t)`: rows
/// `[seed, t[..n-1]]` are consumed, `t[n-1]` becomes the pending seed).
pub(crate) fn forced_acceptance(forced: &[u32]) -> GreedyAccept {
    GreedyAccept {
        committed: forced.to_vec(),
        accepted: forced.len().saturating_sub(1),
        hit_eos: false,
    }
}

/// Whether a window routed `route` with planned source `source` may verify
/// more than one row: native windows, copy windows and forced windows.
pub(crate) fn verify_capable(
    route: crate::lane_exec::LaneRoute,
    source: ProposalSource,
    forced: bool,
) -> bool {
    matches!(route, crate::lane_exec::LaneRoute::Batched { .. })
        || source == ProposalSource::Copy
        || forced
}

// ---------------------------------------------------------------------------
// Owed-row storage and the copy budget
// ---------------------------------------------------------------------------

/// Elements of a lane's owed-row hidden storage: `rows` wide hidden rows.
pub(crate) fn lag_hidden_elements(config: &Qwen4Config, rows: usize) -> Option<usize> {
    config
        .hc_count
        .checked_mul(config.hidden_size)?
        .checked_mul(rows)
}

/// The device allocations of one lane's owed-row storage, in allocation
/// order: ONE F32 hidden tensor of `rows` wide rows (the owed *tokens* are
/// host values the appends embed, never device memory). The lane census
/// ([`crate::lane::census`]) charges exactly this list.
pub(crate) fn lag_allocation_sizes(config: &Qwen4Config, rows: usize) -> Option<Vec<u64>> {
    if rows == 0 {
        return Some(Vec::new());
    }
    let bytes = lag_hidden_elements(config, rows)?.checked_mul(std::mem::size_of::<f32>())?;
    Some(vec![u64::try_from(bytes).ok()?])
}

/// One lane's owed-head-row storage: the target hidden of every owed token
/// (token `i`'s own hidden is row `i`, the pair of token `i + 1`) and the
/// rank's host record of the owed tokens (the same record the root keeps; the
/// plan's catch-up and the decision's `owed` count are checked against it).
/// Fixed capacity (`WindowBudgets.head_lag_rows`): the root flushes before it
/// overflows.
pub(crate) struct HeadLagStore {
    hidden: GpuTensor,
    book: HeadLag,
    capacity: usize,
}

impl HeadLagStore {
    /// `capacity` (at least 1) wide rows on the device.
    pub(crate) fn new(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        capacity: usize,
    ) -> Result<Self, String> {
        let elements = lag_hidden_elements(config, capacity)
            .ok_or_else(|| "lane head lag: hidden extent overflow".to_string())?;
        let hidden = gpu
            .zeros(&[elements], DType::F32)
            .map_err(|e| format!("lane head lag: hidden rows: {e}"))?;
        Ok(Self {
            hidden,
            book: HeadLag::default(),
            capacity,
        })
    }

    /// Back to an empty record (a request's start).
    pub(crate) fn clear(&mut self) {
        self.book.clear();
    }

    /// Owed rows held.
    pub(crate) fn owed(&self) -> usize {
        self.book.tokens.len()
    }

    /// The record: the head position and the owed tokens.
    pub(crate) fn book(&self) -> &HeadLag {
        &self.book
    }

    /// Fixed capacity in rows.
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> Result<(), String> {
        gpu.free_tensor(self.hidden)
            .map_err(|e| format!("lane head lag: {e}"))
    }
}

/// The budget of a copy-armed store: the legacy budget widened to the copy
/// take cap (`verify_rows_per_lane = 1 + max(k, take_cap)`, one capture ring
/// of that width per lane) with `copy.head_lag_rows` owed rows per lane.
/// `row_exact_rows` is the arch's row-exact verify width
/// ([`crate::bundle::Qwen4Bundle`]`::row_exact_verify_rows`); the take cap is
/// `qwen4_ngram_take_cap`. The store allocates exactly this budget and the
/// census charges exactly this budget, so census == allocation by
/// construction ([`copy_budget`] is the only place either derives it).
pub fn copy_budget(
    lanes: usize,
    native_k: usize,
    copy: &LaneCopyConfig,
    row_exact_rows: usize,
    context: Qwen4ContextCommit,
    owner_mask: u64,
) -> Result<LaneSpecBudget, String> {
    let take_cap = qwen4_ngram_take_cap(&copy.ngram, row_exact_rows);
    if take_cap == 0 {
        return Err(format!(
            "lane copy: the n-gram config (n_max {}) takes no candidates within the {row_exact_rows}-row verify",
            copy.ngram.n_max
        ));
    }
    if copy.head_lag_rows == 0 {
        return Err("lane copy: head_lag_rows must be at least 1".to_string());
    }
    Ok(
        LaneSpecBudget::new(lanes, native_k, take_cap, 0, context, owner_mask)?
            // Copy windows (up to `1 + take cap` rows) run in steps of at most
            // `wide_lanes` lanes and every other window is native, so the step
            // buffers keep the native `k + 1` rows per lane.
            .with_step_rows_per_lane(native_k + 1)
            .with_head_lag_rows(copy.head_lag_rows),
    )
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

fn spec_of(lane: &mut Qwen4LaneState, what: &str) -> Result<(), String> {
    if lane.spec.is_none() {
        return Err(format!("{what}: lane has no MTP state"));
    }
    Ok(())
}

fn wide_width(bundle: &Qwen4Bundle) -> Result<usize, String> {
    bundle
        .config
        .hc_count
        .checked_mul(bundle.config.hidden_size)
        .ok_or_else(|| "hidden width overflow".to_string())
}

/// The retired AR token `seed` at `position`, just forwarded by the target
/// alone: copy its wide hidden (row `dec_row` of `hidden`, valid until the
/// next forward) to the lane's next owed-row slot and record the token. The
/// head is not touched (singleton `owe_head_row`, minus its in-line flush:
/// the root schedules the flush at capacity as `owed + retained`).
pub(crate) fn owe_lane_head_row(
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    hidden: &GpuTensor,
    dec_row: usize,
    position: usize,
    seed: u32,
) -> Result<(), String> {
    spec_of(lane, "owe_lane_head_row")?;
    let wide = wide_width(bundle)?;
    let Some(spec) = lane.spec.as_mut() else {
        unreachable!("checked above");
    };
    let LaneSpec { head, lag, .. } = spec;
    let lag = lag
        .as_mut()
        .ok_or_else(|| "owe_lane_head_row: the lane has no owed-row storage".to_string())?;
    let row = lag.book.next_row(position, lag.capacity)?;
    let expect = if lag.book.tokens.is_empty() {
        position
    } else {
        lag.book.start
    };
    if head.position() != expect {
        return Err(format!(
            "owe_lane_head_row: I1: head position {} != {expect}",
            head.position()
        ));
    }
    let src = hidden.sub_offset(dec_row * wide, wide);
    let dst = lag.hidden.sub_offset(row * wide, wide);
    gpu.copy_d2d(&src, &dst, wide * std::mem::size_of::<f32>())
        .map_err(|e| format!("owe_lane_head_row: hidden copy: {e}"))?;
    lag.book.push(position, seed);
    Ok(())
}

/// Append every owed row to the head, teacher-forced (singleton
/// `catch_up_head`): token `p` with the target hidden of `p - 1` (the pending
/// hidden for the first owed token, owed row `i - 1` after it), one
/// [`mtp_append_token`](Qwen4Bundle::mtp_append_token) per row. The lane's
/// head ends at the target position and its pending hidden is the last owed
/// row. Returns the rows appended (0: nothing owed, nothing touched).
///
/// The lane's rows are appended one by one: the batched fill
/// (`fill_head_rows`' `mtp_append_rows` branch) reads the bundle's resident
/// spec-hidden capture, which a lane does not own; the per-row branch is the
/// same `mtp_append_token` pairing it falls back to.
pub(crate) fn flush_owed_head_rows(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
) -> Result<usize, String> {
    spec_of(lane, "flush_owed_head_rows")?;
    let Qwen4LaneState { state, spec, .. } = lane;
    let Some(spec) = spec.as_mut() else {
        unreachable!("checked above");
    };
    let LaneSpec {
        head,
        draft,
        pending_hidden,
        lag,
        ..
    } = spec;
    let Some(lag) = lag.as_mut() else {
        return Ok(0);
    };
    let owed = lag.book.tokens.len();
    if owed == 0 {
        return Ok(0);
    }
    let start = lag.book.start;
    if head.position() != start {
        return Err(format!(
            "flush_owed_head_rows: I1: head position {} != owed start {start}",
            head.position()
        ));
    }
    let HeadLagStore { hidden, book, .. } = lag;
    let tokens = std::mem::take(&mut book.tokens);
    lane_call(bundle.with_lane(state, Some((&mut *head, draft)), |b| {
        for (i, &token) in tokens.iter().enumerate() {
            b.mtp_append_token(gpu, token, Some(&*pending_hidden), start + i)?;
            // Token `i`'s own hidden is the pair of token `i + 1` and, after
            // the last row, the head's pending hidden.
            b.copy_hidden_row(gpu, hidden, i, &*pending_hidden)?;
        }
        Ok::<(), BundleError>(())
    }))
    .map_err(|e| format!("flush_owed_head_rows: {e}"))?;
    if head.position() != start + owed {
        return Err(format!(
            "flush_owed_head_rows: head ended at {}, expected {}",
            head.position(),
            start + owed
        ));
    }
    Ok(owed)
}

/// The plan's round-0 catch-up: the rank's own record must be the plan's
/// (`from`, tokens), the head must stand at `from`, and after the flush the
/// head is at the lane's window position. Returns the rows appended (the
/// round's `catch_up_rows`).
pub(crate) fn catch_up_lane(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    catch_up: &CatchUp,
) -> Result<usize, String> {
    {
        let spec = lane
            .spec
            .as_ref()
            .ok_or_else(|| "catch_up_lane: lane has no MTP state".to_string())?;
        let lag = spec
            .lag
            .as_ref()
            .ok_or_else(|| "catch_up_lane: the lane has no owed-row storage".to_string())?;
        if lag.book.start != catch_up.from || lag.book.tokens != catch_up.tokens {
            return Err(format!(
                "catch_up_lane: the plan owes tokens {:?} from {}, the rank holds {:?} from {}",
                catch_up.tokens, catch_up.from, lag.book.tokens, lag.book.start
            ));
        }
    }
    let rows = flush_owed_head_rows(bundle, gpu, lane)?;
    let end = catch_up.from + catch_up.tokens.len();
    if rows != catch_up.tokens.len() || lane.state.position != end {
        return Err(format!(
            "catch_up_lane: appended {rows} rows, the lane sits at {} (expected {end})",
            lane.state.position
        ));
    }
    Ok(rows)
}

/// Take the lane's head ticket into its pending window (the head repair
/// point). The head must be live.
pub(crate) fn take_head_ticket(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
) -> Result<(), String> {
    spec_of(lane, "take_head_ticket")?;
    let Qwen4LaneState { state, spec, .. } = lane;
    let Some(spec) = spec.as_mut() else {
        unreachable!("checked above");
    };
    let LaneSpec {
        head,
        draft,
        window,
        ..
    } = spec;
    let window = window
        .as_mut()
        .ok_or_else(|| "take_head_ticket: lane has no pending window".to_string())?;
    let ticket = lane_call(bundle.with_lane(state, Some((head, draft)), |b| b.mtp_snapshot(gpu)))
        .map_err(|e| format!("take_head_ticket: {e}"))?;
    window.head_ticket = Some(HeadTicket::Local(ticket));
    Ok(())
}

/// Rows the lane's arenas must map for `plan`'s window: the verify block at
/// most, and a native route's full depth plus the seed (the head's last draft
/// step and its full-accept append reach `position + k_max`).
pub(crate) fn mapped_rows(plan: &LanePlan) -> usize {
    match plan.route {
        crate::lane_exec::LaneRoute::Batched { k_max } => (k_max + 1).max(plan.max_rows()),
        _ => plan.max_rows(),
    }
}

/// The capacity half of `begin_window` for one spec lane: open the lane's
/// pending window from the plan, check the head stands where the plan says
/// (I1: aligned at the window's position, or at the lag's start), map the
/// lane's target and head arenas through the window's end and take the head
/// ticket when the plan wants it at begin ([`head_ticket_point`]). A lane
/// whose head ticket follows a catch-up gets it in round 0
/// ([`take_head_ticket`]).
pub(crate) fn begin_lane_window(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    plan: &LanePlan,
) -> Result<(), String> {
    let epoch = plan.key.request_epoch;
    let Qwen4LaneState { state, spec, .. } = &mut *lane;
    let Some(spec) = spec.as_mut() else {
        return Ok(());
    };
    if spec.window.is_some() {
        return Err(format!(
            "draft_step: lane {epoch:?} has an unconsumed window (commit or abort the step first)"
        ));
    }
    let position = state.position;
    spec.window = Some(PendingWindow {
        route: plan.route,
        position,
        seed: plan.seed,
        drafts: Vec::new(),
        steps: 0,
        head_ticket: None,
        target_ticket: None,
        prepared: None,
        source: plan.source,
        forced: matches!(plan.kind, LaneKind::Forced(_)),
        head_live: plan.head != HeadState::Free,
        probe_saved: None,
        probe: None,
    });
    if let Some(expect) = expected_head_position(plan) {
        if spec.head.position() != expect {
            return Err(format!(
                "draft_step: lane {epoch:?} I1: head position {} != {expect} (trunk position {position})",
                spec.head.position()
            ));
        }
    }
    let point = head_ticket_point(plan);
    if point == HeadTicketPoint::Never {
        return Ok(());
    }
    let end = position
        .checked_add(mapped_rows(plan))
        .ok_or_else(|| "draft_step: position overflow".to_string())?;
    // A mapping refusal is a capacity condition, classified like `provision_step`'s
    // ([`crate::lane_exec::PROVISION_MAP_REFUSED`]): a retryable overload, no poison.
    let refused = crate::lane_exec::PROVISION_MAP_REFUSED;
    spec.head
        .ensure_mapped_capacity(gpu, end)
        .map_err(|e| format!("{refused}{end} head positions for {epoch:?}: {e}"))?;
    state
        .ensure_mapped_capacity(gpu, end)
        .map_err(|e| format!("{refused}{end} positions for {epoch:?}: {e}"))?;
    if point == HeadTicketPoint::AtBegin {
        take_head_ticket(bundle, gpu, lane)
            .map_err(|e| format!("draft_step: head snapshot of {epoch:?}: {e}"))?;
    }
    Ok(())
}

/// Save the head's request state before the probe's Draft op (the singleton
/// saves and restores it around its probe: it is not part of the head ticket).
pub(crate) fn save_probe_state(lane: &mut Qwen4LaneState) {
    if let Some(spec) = lane.spec.as_mut() {
        if let Some(window) = spec.window.as_mut() {
            window.probe_saved = Some(spec.draft);
        }
    }
}

/// Apply the probe's outcome after its token was read: record it, and for an
/// abandoned probe ([`ProbeOutcome::restores_head`]) put the head back to its
/// ticket (the ticket stays live for the finish's validation), restore the
/// request state and zero the window's `steps`.
pub(crate) fn settle_probe(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    outcome: ProbeOutcome,
) -> Result<(), String> {
    spec_of(lane, "settle_probe")?;
    let Qwen4LaneState { state, spec, .. } = lane;
    let Some(spec) = spec.as_mut() else {
        unreachable!("checked above");
    };
    let LaneSpec {
        head,
        draft,
        window,
        ..
    } = spec;
    let window = window
        .as_mut()
        .ok_or_else(|| "settle_probe: lane has no pending window".to_string())?;
    window.probe = Some(outcome);
    if !outcome.restores_head() {
        return Ok(());
    }
    let Some(HeadTicket::Local(ticket)) = window.head_ticket else {
        return Err("settle_probe: the probe lane holds no head ticket".to_string());
    };
    lane_call(bundle.with_lane(state, Some((head, &mut *draft)), |b| {
        b.mtp_restore_retain(gpu, ticket)
    }))
    .map_err(|e| format!("settle_probe: head restore: {e}"))?;
    *draft = window
        .probe_saved
        .take()
        .ok_or_else(|| "settle_probe: the probe saved no request state".to_string())?;
    window.steps = 0;
    Ok(())
}

/// Arm the ring and take the target ticket of a verify lane (`rows > 1`),
/// the same discipline for native, copy and forced windows: ring armed
/// before the snapshot (the GDN states then need no copy), the window's
/// seed/drafts/position checked against the sealed rows, I3 restored by the
/// caller after the forward.
pub(crate) fn arm_lane_verify(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    vl: &VerifyLane,
) -> Result<(), String> {
    let epoch = vl.key.request_epoch;
    let rows = vl.rows.len();
    let Qwen4LaneState { state, spec, .. } = lane;
    let window = spec
        .as_mut()
        .and_then(|spec| spec.window.as_mut())
        .ok_or_else(|| format!("Verify lane {epoch:?} has no window"))?;
    if !verify_capable(window.route, window.source, window.forced)
        || window.seed != vl.rows[0]
        || window.drafts != vl.rows[1..]
        || window.position != state.position
    {
        return Err(format!("Verify lane {epoch:?} disagrees with its window"));
    }
    if window.target_ticket.is_some() {
        return Err(format!(
            "Verify lane {epoch:?} already holds a target ticket"
        ));
    }
    if state.row_capture_rows() < rows {
        return Err(format!(
            "Verify lane {epoch:?}: the ring holds {} rows, the verify needs {rows}",
            state.row_capture_rows()
        ));
    }
    // Armed before the snapshot: the GDN states then need no copy.
    state.row_capture_armed = true;
    match lane_call(bundle.with_lane(state, None, |b| b.snapshot(gpu))) {
        Ok(ticket) => {
            window.target_ticket = Some(TargetTicket::Local(ticket));
            Ok(())
        }
        Err(error) => {
            state.row_capture_armed = false;
            Err(format!("target snapshot of {epoch:?}: {error}"))
        }
    }
}

/// A lane's prepared window: every fallible operation done, tickets still
/// live. [`publish_rank_lane`] consumes them.
pub(crate) enum PreparedRankLane {
    /// A verify window finished through `prepare_native_window` (head live).
    Window(PreparedLaneWindow),
    /// A verify window of a head-free lane: the target ticket, validated.
    TargetOnly { target: TargetTicket },
    /// A one-row window: only an unused head ticket (an abandoned probe) is
    /// left to consume.
    Row { head_ticket: Option<HeadTicket> },
}

/// What [`prepare_rank_lane`] needs of the step.
pub(crate) struct RankPrepare<'a> {
    pub plan: &'a LanePlan,
    pub decision: &'a LaneDecision,
    pub verify: &'a VerifyLane,
    pub config: LaneSpecConfig,
    /// The step's wide-hidden capture; the lane's rows start at `row_begin`.
    pub hidden: &'a GpuTensor,
    pub row_begin: usize,
    /// The lane's target pick at every verify row.
    pub picks: &'a [u32],
    /// Lab negative control (rejected verify keeps its rows).
    pub skip_rollback: bool,
    /// The store's Append scratch when the lane commit fill is on
    /// ([`crate::mtp_lanes::lane_commit_fill_enabled`]), else `None`.
    pub fill: Option<&'a mut crate::mtp_gpu::MtpAppendScratch>,
}

/// A prepared lane and the head rows it appended (owed flush + window rows).
pub(crate) struct RankPrepared {
    pub lane: PreparedRankLane,
    pub head_rows: usize,
}

/// `prepare_commit` of one lane: the target repair and the head schedule of
/// the decision, under the lane's tickets. Every fallible operation is here
/// and ends in a non-consuming ticket validation; [`publish_rank_lane`]
/// consumes. The performed head rows are checked against the decision's
/// schedule (`PreparedLane.head_append` must equal it).
pub(crate) fn prepare_rank_lane(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    p: RankPrepare<'_>,
) -> Result<RankPrepared, String> {
    let rows = p.verify.rows.len();
    let d = p.decision;
    let epoch = p.verify.key.request_epoch;
    let (steps, owed_held) = {
        let spec = lane
            .spec
            .as_ref()
            .ok_or_else(|| "prepare_rank_lane: lane has no MTP state".to_string())?;
        let window = spec
            .window
            .as_ref()
            .ok_or_else(|| "prepare_rank_lane: lane has no pending window".to_string())?;
        (
            window.steps,
            spec.lag.as_ref().map_or(0, HeadLagStore::owed),
        )
    };
    let prep = head_prepare(
        p.plan.head,
        rows,
        d.consumed_rows,
        d.head_append,
        steps,
        owed_held,
    )
    .map_err(|e| format!("lane {epoch:?}: {e}"))?;
    let forced = matches!(p.plan.kind, LaneKind::Forced(_));
    match prep {
        HeadPrepare::TargetOnly if rows > 1 => {
            let target = prepare_target_only(bundle, gpu, lane, &p)?;
            Ok(RankPrepared {
                lane: PreparedRankLane::TargetOnly { target },
                head_rows: 0,
            })
        }
        HeadPrepare::TargetOnly => Ok(RankPrepared {
            lane: PreparedRankLane::Row {
                head_ticket: take_unused_head_ticket(bundle, lane)?,
            },
            head_rows: 0,
        }),
        HeadPrepare::OweRow => {
            let (position, seed) = window_row(lane)?;
            owe_lane_head_row(bundle, gpu, lane, p.hidden, p.row_begin, position, seed)?;
            Ok(RankPrepared {
                lane: PreparedRankLane::Row {
                    head_ticket: take_unused_head_ticket(bundle, lane)?,
                },
                head_rows: 0,
            })
        }
        HeadPrepare::SyncRow { owed } => {
            let flushed = flush_owed_head_rows(bundle, gpu, lane)?;
            if flushed != owed {
                return Err(format!(
                    "lane {epoch:?}: flushed {flushed} owed rows, the decision schedules {owed}"
                ));
            }
            ar_synced_head(bundle, gpu, lane, p.hidden, p.row_begin)?;
            Ok(RankPrepared {
                lane: PreparedRankLane::Row {
                    head_ticket: take_unused_head_ticket(bundle, lane)?,
                },
                head_rows: owed + 1,
            })
        }
        HeadPrepare::Window {
            owed,
            head_at_start,
        } => {
            let flushed = flush_owed_head_rows(bundle, gpu, lane)?;
            if flushed != owed {
                return Err(format!(
                    "lane {epoch:?}: flushed {flushed} owed rows, the decision schedules {owed}"
                ));
            }
            let acceptance = match &p.plan.kind {
                LaneKind::Forced(t) => Some(forced_acceptance(t)),
                // A sampled window's tokens are the root's draws: the rank
                // replays the decision (its own argmax picks are evidence only).
                LaneKind::Decode if p.plan.sampling != SamplingMode::Greedy => Some(GreedyAccept {
                    committed: d.emitted.clone(),
                    accepted: d.accepted,
                    hit_eos: d.hit_eos,
                }),
                LaneKind::Decode => None,
            };
            let prepared = prepare_lane_window_with(
                bundle,
                gpu,
                lane,
                p.config,
                p.hidden,
                p.row_begin,
                p.picks,
                p.skip_rollback,
                LaneFinish {
                    acceptance,
                    head_at_start,
                    eos: LaneEos::Admitted(p.plan.eos),
                },
                p.fill,
            )?;
            let window = &prepared.result;
            if window.committed != d.emitted || window.accepted != d.accepted {
                return Err(format!(
                    "{} lane {epoch:?}: the window finished {:?} (accepted {}) but the root decided {:?} (accepted {})",
                    if forced { "Forced" } else { "Verify" },
                    window.committed,
                    window.accepted,
                    d.emitted,
                    d.accepted
                ));
            }
            let retained = prepared.native.committed_end - p.verify.position;
            let performed = HeadAppend { owed, retained };
            if performed != d.head_append {
                return Err(format!(
                    "lane {epoch:?}: the rank appended {performed:?} head rows, the decision schedules {:?}",
                    d.head_append
                ));
            }
            Ok(RankPrepared {
                lane: PreparedRankLane::Window(prepared),
                head_rows: owed + retained,
            })
        }
    }
}

fn window_row(lane: &Qwen4LaneState) -> Result<(usize, u32), String> {
    lane.spec
        .as_ref()
        .and_then(|s| s.window.as_ref())
        .map(|w| (w.position, w.seed))
        .ok_or_else(|| "lane has no pending window".to_string())
}

/// An abandoned probe's head ticket (still live), validated and handed to the
/// prepared row; `None` when the window never took one.
fn take_unused_head_ticket(
    bundle: &mut Qwen4Bundle,
    lane: &mut Qwen4LaneState,
) -> Result<Option<HeadTicket>, String> {
    let Qwen4LaneState { state, spec, .. } = lane;
    let Some(spec) = spec.as_mut() else {
        return Ok(None);
    };
    let LaneSpec {
        head,
        draft,
        window,
        ..
    } = spec;
    let Some(ticket) = window.as_mut().and_then(|w| w.head_ticket.take()) else {
        return Ok(None);
    };
    let HeadTicket::Local(snapshot) = ticket else {
        return Err("a mesh ticket on a local lane".to_string());
    };
    lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
        b.mtp_validate_commit(snapshot)
    }))
    .map_err(|e| format!("head ticket of an AR row is stale: {e}"))?;
    Ok(Some(ticket))
}

/// The target half of a verify window on a head-free lane: the ring rollback
/// to the consumed rows and the target ticket's validation (no head, no
/// pending hidden: a head-free lane never drafts again).
fn prepare_target_only(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    p: &RankPrepare<'_>,
) -> Result<TargetTicket, String> {
    let rows = p.verify.rows.len();
    let consumed = p.decision.consumed_rows;
    if consumed == 0 || consumed > rows {
        return Err(format!(
            "target-only window consumed {consumed} of {rows} rows"
        ));
    }
    let position = p.verify.position;
    let Qwen4LaneState { state, spec, .. } = lane;
    let window = spec
        .as_mut()
        .and_then(|spec| spec.window.as_mut())
        .ok_or_else(|| "target-only window: lane has no pending window".to_string())?;
    let target = window
        .target_ticket
        .ok_or_else(|| "target-only window: no target ticket".to_string())?;
    let TargetTicket::Local(snapshot) = target else {
        return Err("a mesh ticket on a local lane".to_string());
    };
    if state.position != position + rows {
        return Err(format!(
            "target-only window: I1: trunk position {} != {} after the verify",
            state.position,
            position + rows
        ));
    }
    if consumed < rows {
        if state.row_capture_rows() < rows {
            return Err(format!(
                "target-only window: the lane ring holds {} rows, the verify has {rows}",
                state.row_capture_rows()
            ));
        }
        let ple_normed = bundle
            .lane_ple_normed_rows(p.row_begin, rows)
            .map_err(|e| format!("target-only window: PLE rows: {e}"))?;
        lane_call(bundle.with_lane(state, None, |b| {
            b.rollback_verify_rows_retain_with(gpu, snapshot, consumed, &p.verify.rows, &ple_normed)
        }))
        .map_err(|e| format!("target-only window: ring rollback: {e}"))?;
    }
    if state.position != position + consumed {
        return Err(format!(
            "target-only window ended at {}, expected {}",
            state.position,
            position + consumed
        ));
    }
    lane_call(bundle.with_lane(state, None, |b| b.validate_commit(snapshot)))
        .map_err(|e| format!("target-only window: stale target ticket: {e}"))?;
    Ok(target)
}

/// Consume the tickets of a prepared lane (no GPU operation).
pub(crate) fn publish_rank_lane(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    prepared: PreparedRankLane,
) -> Result<(), String> {
    match prepared {
        PreparedRankLane::Window(window) => {
            publish_lane_window(bundle, gpu, lane, window).map(|_| ())
        }
        PreparedRankLane::TargetOnly { target } => {
            let TargetTicket::Local(snapshot) = target else {
                return Err("a mesh ticket on a local lane".to_string());
            };
            let Qwen4LaneState { state, .. } = lane;
            lane_call(bundle.with_lane(state, None, |b| {
                b.commit_validated(snapshot);
                Ok::<(), BundleError>(())
            }))
            .map_err(|e| format!("publish target-only window: {e}"))
        }
        PreparedRankLane::Row { head_ticket: None } => Ok(()),
        PreparedRankLane::Row {
            head_ticket: Some(HeadTicket::Local(snapshot)),
        } => {
            let Qwen4LaneState { state, spec, .. } = lane;
            let Some(LaneSpec { head, draft, .. }) = spec.as_mut() else {
                return Ok(());
            };
            lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
                b.mtp_commit_validated(snapshot);
                Ok::<(), BundleError>(())
            }))
            .map_err(|e| format!("publish row head ticket: {e}"))
        }
        PreparedRankLane::Row {
            head_ticket: Some(HeadTicket::Mesh(_)),
        } => Err("a mesh ticket on a local lane".to_string()),
    }
}

/// The head's lab-readable consistency facts of one lane, for the oracle:
/// (head position, target position, owed rows held, owed-record start).
#[cfg(feature = "lab")]
pub(crate) fn head_facts(lane: &Qwen4LaneState) -> Option<(usize, usize, usize, usize)> {
    let spec = lane.spec.as_ref()?;
    let (owed, start) = spec
        .lag
        .as_ref()
        .map_or((0, 0), |l| (l.owed(), l.book().start));
    Some((spec.head.position(), lane.state.position, owed, start))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lane_exec::LaneRoute;
    use crate::mesh_lane_protocol::{LaneWindowKey, SamplingMode};
    use hipfire_runtime::slot_batch::RequestEpoch;

    fn key() -> LaneWindowKey {
        LaneWindowKey {
            request_epoch: RequestEpoch {
                request_tag: 1,
                owner_generation: 1,
            },
            lane_generation: 1,
            window_seq: 0,
        }
    }

    fn plan() -> LanePlan {
        LanePlan {
            wide_verify: false,
            key: key(),
            kind: LaneKind::Decode,
            position: 40,
            seed: 9,
            route: LaneRoute::ArSynced,
            source: ProposalSource::None,
            max_depth: 0,
            copy_drafts: Vec::new(),
            copy_confirm: false,
            native_depth: 0,
            catch_up: None,
            sampling: SamplingMode::Greedy,
            eos: Some(2),
            remaining: 100,
            budget: 0,
            head: HeadState::Aligned,
        }
    }

    #[test]
    fn mesh_rank_copy_probe_outcome_mirrors_the_roots_confirmation_rule() {
        // Hit: the head's own argmax is the first candidate (any native depth).
        assert_eq!(probe_outcome(&[5, 6, 7], 5, 3), ProbeOutcome::Confirmed);
        assert_eq!(probe_outcome(&[5, 6, 7], 5, 0), ProbeOutcome::Confirmed);
        // Miss with a native fallback: the probe is the first native draft.
        assert_eq!(probe_outcome(&[5, 6, 7], 8, 3), ProbeOutcome::MissedNative);
        // Miss without one: an AR row, the probe abandoned.
        assert_eq!(probe_outcome(&[5, 6, 7], 8, 0), ProbeOutcome::MissedAr);
        // No candidate can never be confirmed.
        assert_eq!(probe_outcome(&[], 5, 2), ProbeOutcome::MissedNative);
        // Only a continuing native draft keeps the head advanced.
        assert!(ProbeOutcome::Confirmed.restores_head());
        assert!(ProbeOutcome::MissedAr.restores_head());
        assert!(!ProbeOutcome::MissedNative.restores_head());
    }

    #[test]
    fn mesh_rank_copy_head_at_start_rule() {
        // A window whose head ran no step that is still in effect finishes
        // from the start: established copy, forced, a confirmed or abandoned
        // probe (settle_probe zeroes `steps`).
        assert!(head_at_start(0));
        // A native window and a missed probe that continues natively keep
        // the draft chain's row 0.
        assert!(!head_at_start(1));
        assert!(!head_at_start(5));
        // The composition the rank applies to a probe lane.
        let steps_after = |outcome: ProbeOutcome, probe_steps: usize| {
            if outcome.restores_head() {
                0
            } else {
                probe_steps
            }
        };
        assert!(head_at_start(steps_after(ProbeOutcome::Confirmed, 1)));
        assert!(head_at_start(steps_after(ProbeOutcome::MissedAr, 1)));
        assert!(!head_at_start(steps_after(ProbeOutcome::MissedNative, 1)));
    }

    #[test]
    fn mesh_rank_copy_head_ticket_point_follows_the_catch_up() {
        let mut p = plan();
        // AR rows never take a head ticket.
        assert_eq!(head_ticket_point(&p), HeadTicketPoint::Never);
        p.route = LaneRoute::ArRetired;
        p.head = HeadState::Lag { from: 38 };
        assert_eq!(head_ticket_point(&p), HeadTicketPoint::Never);
        // A native window: at begin, or after its catch-up.
        p.route = LaneRoute::Batched { k_max: 3 };
        p.source = ProposalSource::Native;
        p.head = HeadState::Aligned;
        assert_eq!(head_ticket_point(&p), HeadTicketPoint::AtBegin);
        p.head = HeadState::Lag { from: 38 };
        p.catch_up = Some(CatchUp {
            from: 38,
            tokens: vec![1, 2],
        });
        assert_eq!(head_ticket_point(&p), HeadTicketPoint::AfterCatchUp);
        // An established copy window on a lagging head (no catch-up): at
        // begin; the owed rows flush at prepare.
        p.source = ProposalSource::Copy;
        p.catch_up = None;
        assert_eq!(head_ticket_point(&p), HeadTicketPoint::AtBegin);
        // A head-free lane takes none, even for a copy or forced window.
        p.head = HeadState::Free;
        assert_eq!(head_ticket_point(&p), HeadTicketPoint::Never);
        p.kind = LaneKind::Forced(vec![1, 2, 3]);
        assert_eq!(head_ticket_point(&p), HeadTicketPoint::Never);
        // A forced window with more than one token takes one; one token is a
        // single AR-style row.
        p.head = HeadState::Aligned;
        p.source = ProposalSource::None;
        assert_eq!(head_ticket_point(&p), HeadTicketPoint::AtBegin);
        p.kind = LaneKind::Forced(vec![4]);
        assert_eq!(head_ticket_point(&p), HeadTicketPoint::Never);
    }

    #[test]
    fn mesh_rank_copy_expected_head_position_and_keep_cap() {
        let mut p = plan();
        assert_eq!(expected_head_position(&p), Some(40));
        p.head = HeadState::Lag { from: 36 };
        assert_eq!(expected_head_position(&p), Some(36));
        p.head = HeadState::Free;
        assert_eq!(expected_head_position(&p), None);
        // A copy plan's head keeps at most its native fallback depth.
        p.source = ProposalSource::Copy;
        p.max_depth = 7;
        p.native_depth = 3;
        assert_eq!(native_keep_cap(&p), 3);
        p.source = ProposalSource::Native;
        p.max_depth = 4;
        assert_eq!(native_keep_cap(&p), 4);
        // Only the first head op of a confirm plan is the probe.
        p.source = ProposalSource::Copy;
        p.copy_confirm = true;
        assert!(is_probe_op(&p, 40));
        assert!(!is_probe_op(&p, 41));
        p.copy_confirm = false;
        assert!(!is_probe_op(&p, 40));
    }

    #[test]
    fn mesh_rank_copy_head_prepare_schedule() {
        let none = HeadAppend::default();
        let h = HeadState::Aligned;
        // Head-free: nothing may be scheduled.
        assert_eq!(
            head_prepare(HeadState::Free, 1, 1, none, 0, 0),
            Ok(HeadPrepare::TargetOnly)
        );
        assert!(head_prepare(
            HeadState::Free,
            4,
            4,
            HeadAppend {
                owed: 0,
                retained: 4
            },
            0,
            0
        )
        .is_err());
        // A retired AR token owes its row; rows already owed stay owed.
        assert_eq!(
            head_prepare(HeadState::Lag { from: 38 }, 1, 1, none, 0, 2),
            Ok(HeadPrepare::OweRow)
        );
        // An appending schedule must flush every row the rank holds.
        assert_eq!(
            head_prepare(
                HeadState::Lag { from: 38 },
                1,
                1,
                HeadAppend {
                    owed: 0,
                    retained: 1
                },
                0,
                2
            ),
            Err("head schedule flushes 0 owed rows, the rank holds 2".to_string())
        );
        // Owed rows alone are never scheduled, and no verify row owes.
        assert!(head_prepare(
            HeadState::Lag { from: 38 },
            1,
            1,
            HeadAppend {
                owed: 2,
                retained: 0
            },
            0,
            2
        )
        .is_err());
        assert!(head_prepare(HeadState::Aligned, 4, 3, none, 0, 0).is_err());
        assert_eq!(
            head_prepare(HeadState::Lag { from: 38 }, 1, 1, none, 0, 0),
            Ok(HeadPrepare::OweRow)
        );
        // An AR row the head keeps up with (ArSynced), with and without a flush.
        let sync = |owed| HeadAppend { owed, retained: 1 };
        assert_eq!(
            head_prepare(h, 1, 1, sync(0), 0, 0),
            Ok(HeadPrepare::SyncRow { owed: 0 })
        );
        assert_eq!(
            head_prepare(HeadState::Lag { from: 37 }, 1, 1, sync(3), 0, 3),
            Ok(HeadPrepare::SyncRow { owed: 3 })
        );
        // A verify window: head_at_start follows the steps in effect.
        let win = |owed, retained| HeadAppend { owed, retained };
        assert_eq!(
            head_prepare(h, 4, 3, win(0, 3), 0, 0),
            Ok(HeadPrepare::Window {
                owed: 0,
                head_at_start: true
            })
        );
        assert_eq!(
            head_prepare(h, 4, 3, win(0, 3), 2, 0),
            Ok(HeadPrepare::Window {
                owed: 0,
                head_at_start: false
            })
        );
        assert_eq!(
            head_prepare(HeadState::Lag { from: 38 }, 8, 8, win(2, 8), 0, 2),
            Ok(HeadPrepare::Window {
                owed: 2,
                head_at_start: true
            })
        );
        // Retained rows are all of the consumed rows or the schedule is wrong.
        assert!(head_prepare(h, 4, 3, win(0, 2), 1, 0).is_err());
        assert!(head_prepare(h, 1, 1, win(0, 2), 0, 0).is_err());
    }

    #[test]
    fn mesh_rank_copy_prepared_head_rows_match_the_schedule_total() {
        // The receipt's per-lane head rows are the decision's `owed +
        // retained` for every shape the rank executes.
        for (rows, consumed, append, steps, owed_held) in [
            (
                1,
                1,
                HeadAppend {
                    owed: 0,
                    retained: 1,
                },
                0,
                0,
            ),
            (
                1,
                1,
                HeadAppend {
                    owed: 3,
                    retained: 1,
                },
                0,
                3,
            ),
            (1, 1, HeadAppend::default(), 0, 0),
            (
                5,
                3,
                HeadAppend {
                    owed: 0,
                    retained: 3,
                },
                1,
                0,
            ),
            (
                8,
                8,
                HeadAppend {
                    owed: 4,
                    retained: 8,
                },
                0,
                4,
            ),
            (
                3,
                3,
                HeadAppend {
                    owed: 0,
                    retained: 3,
                },
                0,
                0,
            ),
        ] {
            let head = if owed_held > 0 {
                HeadState::Lag { from: 10 }
            } else {
                HeadState::Aligned
            };
            let prep = head_prepare(head, rows, consumed, append, steps, owed_held).unwrap();
            assert_eq!(
                prepared_head_rows(prep, consumed),
                append.total(),
                "{prep:?}"
            );
        }
        assert_eq!(prepared_head_rows(HeadPrepare::TargetOnly, 4), 0);
    }

    #[test]
    fn mesh_rank_copy_forced_acceptance_keeps_every_row_and_draws_nothing() {
        let a = forced_acceptance(&[11, 12, 13]);
        assert_eq!(a.committed, vec![11, 12, 13]);
        assert_eq!(a.accepted, 2);
        assert!(!a.hit_eos);
        // The target consumes `1 + accepted` = the forced rows: the seed and
        // f1..f_{n-1}; f_n stays pending.
        assert_eq!(1 + crate::mtp_spec::target_commit_accept_len(&a), 3);
        let one = forced_acceptance(&[7]);
        assert_eq!((one.committed.len(), one.accepted), (1, 0));
        assert_eq!(1 + crate::mtp_spec::target_commit_accept_len(&one), 1);
    }

    #[test]
    fn mesh_rank_copy_verify_capable_windows() {
        let batched = LaneRoute::Batched { k_max: 3 };
        assert!(verify_capable(batched, ProposalSource::Native, false));
        assert!(verify_capable(
            LaneRoute::ArRetired,
            ProposalSource::Copy,
            false
        ));
        assert!(verify_capable(LaneRoute::Ar, ProposalSource::None, true));
        for route in [LaneRoute::Ar, LaneRoute::ArSynced, LaneRoute::ArRetired] {
            assert!(!verify_capable(route, ProposalSource::None, false));
        }
    }

    fn compact() -> Qwen4Config {
        crate::config::compact_test_config()
    }

    #[test]
    fn mesh_rank_copy_lag_allocation_is_one_hidden_tensor_per_lane() {
        let config = compact();
        let wide = (config.hc_count * config.hidden_size) as u64;
        assert_eq!(lag_allocation_sizes(&config, 0), Some(Vec::new()));
        assert_eq!(lag_allocation_sizes(&config, 1), Some(vec![wide * 4]));
        assert_eq!(lag_allocation_sizes(&config, 8), Some(vec![wide * 4 * 8]));
        assert_eq!(lag_hidden_elements(&config, 8), Some(wide as usize * 8));
        assert_eq!(lag_allocation_sizes(&config, usize::MAX), None);
    }

    #[test]
    fn mesh_rank_copy_census_charges_exactly_the_allocated_lag_storage() {
        use crate::lane::census::{
            lane_spec_census, CensusComponent, CensusInputs, LaneSpecBudget,
        };
        use crate::state::Qwen4StateFormat;
        use crate::weights::Qwen4BandScope;
        use crate::{GdnStateFormat, QsaKvFormat};
        let config = compact();
        let context = Qwen4ContextCommit::legacy(64);
        let scope = Qwen4BandScope::single(&config);
        let format = Qwen4StateFormat {
            gdn: GdnStateFormat::F32,
            qsa: QsaKvFormat::Fp8,
            qsa_f16_arena: false,
            qsa_pooled_bf16: false,
        };
        let inputs = CensusInputs {
            config: &config,
            format,
            scope: &scope,
            rank: 0,
            append_scratch: None,
        };
        let copy = LaneCopyConfig {
            ngram: hipfire_runtime::ngram_mod::NgramModConfig {
                n_max: 7,
                n_min: 1,
                n_match: 4,
                ..Default::default()
            },
            head_lag_rows: 5,
            wide_lanes: usize::MAX,
        };
        let budget = copy_budget(3, 2, &copy, 8, context, 1).unwrap();
        assert_eq!(budget.copy_take_cap, 7);
        assert_eq!(budget.verify_rows_per_lane, 8);
        assert_eq!(budget.capture_rows_per_lane, 8);
        assert_eq!(budget.head_lag_rows, 5);
        let census = lane_spec_census(&budget, &inputs).unwrap();
        let lines: Vec<_> = census
            .lines
            .iter()
            .filter(|l| l.component == CensusComponent::HeadLag)
            .collect();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].copies, 3, "one owed-row store per lane");
        assert_eq!(
            lines[0].allocations,
            lag_allocation_sizes(&config, 5).unwrap(),
            "the census list is the allocator's list"
        );
        // Without copy there is no lag line at all.
        let legacy = LaneSpecBudget::legacy(3, 2, context, 1).unwrap();
        let census = lane_spec_census(&legacy, &inputs).unwrap();
        assert!(census
            .lines
            .iter()
            .all(|l| l.component != CensusComponent::HeadLag));
    }

    #[test]
    fn mesh_rank_copy_budget_refuses_unusable_configs() {
        let context = Qwen4ContextCommit::legacy(64);
        let ngram = hipfire_runtime::ngram_mod::NgramModConfig {
            n_max: 3,
            n_min: 1,
            n_match: 4,
            ..Default::default()
        };
        let ok = LaneCopyConfig {
            ngram,
            head_lag_rows: 2,
            wide_lanes: usize::MAX,
        };
        // The take cap is min(n_max, row_exact - 1).
        let b = copy_budget(2, 1, &ok, 8, context, 1).unwrap();
        assert_eq!((b.copy_take_cap, b.verify_rows_per_lane), (3, 4));
        let b = copy_budget(2, 5, &ok, 8, context, 1).unwrap();
        assert_eq!((b.copy_take_cap, b.verify_rows_per_lane), (3, 6));
        // A row-exact width of one leaves no candidate row.
        assert!(copy_budget(2, 1, &ok, 1, context, 1).is_err());
        // A lag-free budget could never hold an owed row.
        let none = LaneCopyConfig {
            head_lag_rows: 0,
            wide_lanes: usize::MAX,
            ..ok
        };
        assert!(copy_budget(2, 1, &none, 8, context, 1).is_err());
    }

    // The device glue cannot be built host-only (`Qwen4Bundle`/`Gpu`); these
    // pin the ordering the contract needs on the source itself.
    const SRC: &str = include_str!("mesh_rank_copy.rs");

    fn fn_body<'a>(name: &str) -> &'a str {
        let head = format!("pub(crate) fn {name}(");
        let start = SRC.find(&head).unwrap_or_else(|| panic!("{name} missing"));
        let rest = &SRC[start..];
        let end = rest.find("\n}\n").expect("fn end");
        &rest[..end]
    }

    #[test]
    fn mesh_rank_copy_flush_appends_before_it_moves_pending_and_uses_the_shared_append() {
        let body = fn_body("flush_owed_head_rows");
        let append = body.find("mtp_append_token(").expect("append");
        let copy = body.find("copy_hidden_row(").expect("pending move");
        assert!(
            append < copy,
            "token i pairs with the pending hidden of i-1"
        );
        assert!(body.contains("Some(&*pending_hidden)"));
        assert!(
            !body.contains("mtp_append_rows"),
            "no second append arithmetic"
        );
    }

    #[test]
    fn mesh_rank_copy_probe_restore_precedes_the_zeroed_steps_and_keeps_the_ticket_live() {
        let body = fn_body("settle_probe");
        let restore = body.find("mtp_restore_retain(").expect("retain restore");
        let steps = body.find("window.steps = 0").expect("steps zeroed");
        assert!(restore < steps);
        assert!(
            !body.contains("mtp_restore(gpu"),
            "the ticket must stay live"
        );
    }

    #[test]
    fn mesh_rank_copy_prepare_flushes_before_the_finish_and_checks_the_schedule() {
        let body = fn_body("prepare_rank_lane");
        let flush = body.find("flush_owed_head_rows(").expect("flush");
        let finish = body.find("prepare_lane_window_with(").expect("finish");
        assert!(flush < finish, "owed rows go first");
        assert!(body.contains("performed != d.head_append"));
        for consume in [
            "publish_lane_window(",
            "commit_validated(",
            "retain_window(",
        ] {
            assert!(
                !body.contains(consume),
                "prepare must not consume: {consume}"
            );
        }
        let publish = fn_body("publish_rank_lane");
        assert!(publish.contains("publish_lane_window("));
        assert!(!publish.contains("flush_owed_head_rows("));
    }

    // ---- the rank's host model against the real root ---------------------------------

    use crate::lane::LabRoutes;
    use crate::mesh_lane_protocol::scripted::{Script, ScriptedExecutor, WindowScript};
    use crate::mesh_lane_protocol::{
        CommitDecision, GreedyRule, HeadEnd, LaneAdmission, LaneMeshExecutor, LaneRequest,
        MeshRoot, RankOwnership, RootConfig, StagePolicy, WindowBudgets, WindowStart,
    };
    use crate::mtp_lanes::LaneSpecPolicy;
    use hipfire_runtime::ngram_mod::NgramModConfig;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn epoch(tag: u64) -> RequestEpoch {
        RequestEpoch {
            request_tag: tag,
            owner_generation: 1,
        }
    }

    /// The rank's host-visible state of one lane: where its head stands and
    /// which rows it owes (what the device does is pinned by source text and
    /// the GPU gate; this is the arithmetic that drives it).
    #[derive(Default)]
    struct Sim {
        head: usize,
        lag_start: usize,
        lag: Vec<u32>,
    }

    impl Sim {
        fn flush(&mut self) -> usize {
            let n = self.lag.len();
            if n > 0 {
                assert_eq!(
                    self.head, self.lag_start,
                    "the head stands at the lag start"
                );
                self.head += n;
                self.lag.clear();
            }
            n
        }
    }

    /// The facts of one driven window the tests assert on.
    struct Driven {
        start: WindowStart,
        decision: CommitDecision,
        rows: Vec<u32>,
        prep: HeadPrepare,
        steps: usize,
        probe: Option<ProbeOutcome>,
        catch_rows: usize,
    }

    struct Harness {
        root: MeshRoot,
        exec: ScriptedExecutor,
        script: Rc<RefCell<Script>>,
        sim: Sim,
    }

    const LAG_ROWS: usize = 4;

    fn copy_cfg() -> NgramModConfig {
        NgramModConfig {
            capacity: 1 << 10,
            n_match: 3,
            n_min: 2,
            n_max: 7,
        }
    }

    fn cycle_prompt() -> Vec<u32> {
        (20..80).chain(20..80).collect()
    }

    fn budgets() -> WindowBudgets {
        WindowBudgets {
            max_lanes: 8,
            verify_rows_per_lane: 8,
            max_rows: 64,
            copy_take_cap: 7,
            lab: LabRoutes::Batched,
            pp_microbatch_lanes: None,
            head_lag_rows: LAG_ROWS,
        }
    }

    impl Harness {
        fn new() -> Self {
            let own = RankOwnership::single();
            let root = MeshRoot::new(
                own,
                &[own],
                RootConfig {
                    topology_generation: 1,
                    stage: StagePolicy {
                        moe_layers: 0,
                        moe_once: false,
                        stream_bf16: false,
                        all_rows: crate::gpu_forward_lanes::LaneAllRowsRule::never(),
                    },
                    floor_enabled: false,
                },
            )
            .unwrap()
            .with_copy_lane(copy_cfg())
            .unwrap();
            let script = Rc::new(RefCell::new(Script::default()));
            Self {
                root,
                exec: ScriptedExecutor::new(own, script.clone()),
                script,
                sim: Sim::default(),
            }
        }

        fn admit(&mut self, position: usize, seed: u32, prompt: Vec<u32>, retired: bool) {
            let e = epoch(1);
            let a = self
                .root
                .admit_lane(LaneAdmission {
                    epoch: e,
                    position,
                    seed,
                    eos: None,
                    rng_seed: 7,
                    prompt,
                    spec_k: Some(3),
                    head: HeadState::Aligned,
                    copy: true,
                    head_lag: Vec::new(),
                })
                .unwrap();
            self.exec.admit_lane(&a).unwrap();
            if retired {
                self.root.test_set_policy(
                    e,
                    LaneSpecPolicy::promoted_with_floor(
                        3,
                        &[hipfire_runtime::spec_economics::AGREEMENT_PRIOR; 3],
                        true,
                        false,
                    ),
                );
            }
            self.sim = Sim {
                head: position,
                ..Sim::default()
            };
        }

        fn script(&self, seq: u64, drafts: &[u32], picks: &[u32]) {
            self.script.borrow_mut().windows.insert(
                (1, seq),
                WindowScript {
                    drafts: drafts.to_vec(),
                    picks: picks.to_vec(),
                    keep_going: Vec::new(),
                    drop_last: Vec::new(),
                },
            );
        }

        /// One whole step of the lane, with the rank's host model applying
        /// exactly what `prepare_rank_lane` / `run_draft_round` apply.
        fn drive(&mut self, forced: Option<Vec<u32>>) -> Driven {
            let e = epoch(1);
            let req = LaneRequest {
                wide_verify: false,
                epoch: e,
                remaining: 99,
                context_left: 99,
                forced,
                sampling: SamplingMode::Greedy,
            };
            let start = self.root.plan_window(None, &[req], &budgets()).unwrap();
            let plan = start.lanes[0].clone();
            let ready = self.exec.begin_window(&start);
            self.root.collect_ready(&[ready]).unwrap();
            // begin_lane_window: I1 against the plan.
            if let Some(expect) = expected_head_position(&plan) {
                assert_eq!(self.sim.head, expect, "the head stands where the plan says");
            }
            let mut steps = 0usize;
            let mut probe = None;
            let mut catch_rows = 0usize;
            while let Some(round) = self.root.next_draft_round().unwrap() {
                for (_, c) in &round.catch_up {
                    assert_eq!(c.from, self.sim.lag_start);
                    assert_eq!(
                        c.tokens, self.sim.lag,
                        "the rank holds the plan's owed rows"
                    );
                    catch_rows += self.sim.flush();
                    assert_eq!(self.sim.head, plan.position);
                    assert_eq!(head_ticket_point(&plan), HeadTicketPoint::AfterCatchUp);
                }
                let res = self.exec.run_draft_round(&round).unwrap();
                assert_eq!(res.catch_up_rows, catch_rows);
                let proposals = res.proposals.clone().unwrap();
                for (op, prop) in round.ops.iter().zip(&proposals) {
                    if is_probe_op(&plan, op.position) {
                        let outcome =
                            probe_outcome(&plan.copy_drafts, prop.token, plan.native_depth);
                        probe = Some(outcome);
                        steps = usize::from(!outcome.restores_head());
                    } else {
                        steps += 1;
                    }
                }
                self.root.collect_draft(&round, &[res]).unwrap();
            }
            if steps > 0 {
                // The draft steps advanced a head that stood at the window.
                assert_eq!(self.sim.head, plan.position);
                self.sim.head = plan.position + steps;
            }
            let sealed = self.root.seal_verify().unwrap();
            let v = self.exec.run_verify(&sealed).unwrap();
            self.root.collect_verify(&[v]).unwrap();
            let decision = self.root.decide_commit(&mut GreedyRule).unwrap();
            let d = decision.lanes[0].clone();
            let rows = sealed.lanes[0].rows.clone();
            // prepare_rank_lane's schedule over the rank's own record.
            let prep = head_prepare(
                plan.head,
                rows.len(),
                d.consumed_rows,
                d.head_append,
                steps,
                self.sim.lag.len(),
            )
            .unwrap_or_else(|e| panic!("{e}: plan {plan:?} decision {d:?}"));
            match prep {
                HeadPrepare::TargetOnly => {}
                HeadPrepare::OweRow => {
                    if self.sim.lag.is_empty() {
                        self.sim.lag_start = plan.position;
                        assert_eq!(self.sim.head, plan.position, "the head is aligned");
                    }
                    assert_eq!(
                        self.sim.lag_start + self.sim.lag.len(),
                        plan.position,
                        "the owed rows continue"
                    );
                    self.sim.lag.push(plan.seed);
                }
                HeadPrepare::SyncRow { owed } => {
                    assert_eq!(self.sim.flush(), owed);
                    assert_eq!(self.sim.head, plan.position);
                    self.sim.head = plan.position + 1;
                }
                HeadPrepare::Window {
                    owed,
                    head_at_start: at_start,
                } => {
                    assert_eq!(self.sim.flush(), owed);
                    // prepare_lane_window_with's own I1 and head_at_start checks.
                    assert_eq!(self.sim.head, plan.position + steps);
                    assert_eq!(at_start, steps == 0);
                    self.sim.head = plan.position + d.consumed_rows;
                }
            }
            let receipt = self.exec.prepare_commit(&decision).unwrap();
            assert_eq!(
                catch_rows + prepared_head_rows(prep, d.consumed_rows),
                receipt.head_append_rows,
                "the rank's rows are the decision's schedule"
            );
            self.root.collect_prepared(&[receipt]).unwrap();
            self.root.publish_commit().unwrap();
            self.exec.publish_commit(&decision);
            // The root and the rank agree on the head afterwards.
            let (state, owed) = self.root.head_state(e).unwrap();
            match state {
                HeadState::Aligned => {
                    let (position, _) = self.root.lane_position(e).unwrap();
                    assert!(self.sim.lag.is_empty() && owed.is_empty());
                    assert_eq!(self.sim.head, position);
                    assert_eq!(
                        d.head_end,
                        HeadEnd::Aligned { end: position },
                        "the decision's head end"
                    );
                }
                HeadState::Lag { from } => {
                    assert_eq!(self.sim.lag_start, from);
                    assert_eq!(self.sim.lag, owed);
                    assert_eq!(self.sim.head, from, "an owing head is not touched");
                }
                HeadState::Free => assert_eq!(d.head_end, HeadEnd::Free),
            }
            Driven {
                start,
                decision,
                rows,
                prep,
                steps,
                probe,
                catch_rows,
            }
        }
    }

    #[test]
    fn mesh_rank_copy_rank_model_follows_a_retired_lane_through_lag_catch_up_and_a_confirmed_takeover(
    ) {
        let mut h = Harness::new();
        h.admit(30, 5, (1..=30).collect(), true);
        // Three retired AR tokens owe their rows; the head is not touched.
        for (seq, pick) in [6u32, 7, 8].into_iter().enumerate() {
            h.script(seq as u64, &[], &[pick]);
            let w = h.drive(None);
            assert_eq!(w.prep, HeadPrepare::OweRow, "window {seq}");
            assert_eq!(h.sim.lag.len(), seq + 1);
        }
        assert_eq!(h.sim.lag, vec![5, 6, 7]);
        // The copy source takes the next window: catch-up first, then the
        // probe confirms and the window finishes from the window's start.
        h.script(3, &[9], &[9, 10, 11, 12, 13, 14, 15, 16]);
        let w = h.drive(None);
        assert_eq!(w.catch_rows, 3);
        assert_eq!(w.probe, Some(ProbeOutcome::Confirmed));
        assert_eq!(w.steps, 0, "the abandoned probe is not a step in effect");
        let consumed = w.decision.lanes[0].consumed_rows;
        assert_eq!(
            w.prep,
            HeadPrepare::Window {
                owed: 0,
                head_at_start: true
            }
        );
        assert_eq!(w.rows.len(), 1 + w.decision.lanes[0].source.offered);
        assert_eq!(h.sim.head, 33 + consumed);
        assert!(h.sim.lag.is_empty());
    }

    #[test]
    fn mesh_rank_copy_rank_model_a_missed_probe_owes_a_fresh_row_after_its_catch_up() {
        let mut h = Harness::new();
        h.admit(30, 5, (1..=30).collect(), true);
        for (seq, pick) in [6u32, 7, 8].into_iter().enumerate() {
            h.script(seq as u64, &[], &[pick]);
            h.drive(None);
        }
        // The head's row-0 draft (777) is not the first candidate: an AR
        // window that already caught up, so the head owes only this row.
        h.script(3, &[777], &[501, 502]);
        let w = h.drive(None);
        assert_eq!(w.probe, Some(ProbeOutcome::MissedAr));
        assert_eq!(w.steps, 0);
        assert_eq!(w.catch_rows, 3);
        assert_eq!(w.prep, HeadPrepare::OweRow);
        assert_eq!((h.sim.lag_start, h.sim.lag.clone()), (33, vec![8]));
    }

    #[test]
    fn mesh_rank_copy_rank_model_lag_flushes_at_capacity_with_the_row_that_fills_it() {
        let mut h = Harness::new();
        h.admit(30, 5, (1..=30).collect(), true);
        let mut flushes = 0;
        for seq in 0..9u64 {
            // Never a repeated 3-gram: no copy proposal, pure lag growth.
            h.script(seq, &[], &[1000 + seq as u32]);
            let w = h.drive(None);
            match w.prep {
                HeadPrepare::OweRow => assert!(h.sim.lag.len() < LAG_ROWS),
                HeadPrepare::SyncRow { owed } => {
                    flushes += 1;
                    assert_eq!(owed, LAG_ROWS - 1, "window {seq}");
                    assert!(h.sim.lag.is_empty());
                    assert_eq!(h.sim.head, w.start.lanes[0].position + 1);
                }
                other => panic!("window {seq}: {other:?}"),
            }
        }
        assert_eq!(flushes, 2, "windows 3 and 7 fill the 4-row record");
    }

    #[test]
    fn mesh_rank_copy_rank_model_live_lane_runs_native_confirmed_and_established_copy_windows() {
        let mut h = Harness::new();
        h.admit(120, 20, cycle_prompt(), false);
        let (mut at_start, mut kept_row0, mut established) = (0, 0, 0);
        let mut seen_probe = false;
        for seq in 0..12u64 {
            let (_, v) = h.root.lane_position(epoch(1)).unwrap();
            let drafts: Vec<u32> = (1..=7).map(|i| v + i).collect();
            let picks: Vec<u32> = (1..=8).map(|i| v + i).collect();
            h.script(seq, &drafts, &picks);
            let w = h.drive(None);
            seen_probe |= w.probe.is_some();
            let plan = &w.start.lanes[0];
            if plan.source == ProposalSource::Copy && !plan.copy_confirm {
                established += 1;
                assert_eq!(w.steps, 0, "an established copy window runs no head Draft");
            }
            match w.prep {
                HeadPrepare::Window { head_at_start, .. } => {
                    if head_at_start {
                        at_start += 1;
                    } else {
                        kept_row0 += 1;
                    }
                }
                other => panic!("window {seq}: {other:?}"),
            }
        }
        assert!(
            seen_probe && established >= 1 && at_start >= 1,
            "{at_start} {established}"
        );
        let _ = kept_row0;
    }

    #[test]
    fn mesh_rank_copy_rank_model_a_missed_probe_continues_as_native_drafts() {
        let mut h = Harness::new();
        h.admit(120, 20, cycle_prompt(), false);
        h.script(0, &[999, 998, 997], &[500, 501, 502, 503]);
        let w = h.drive(None);
        assert_eq!(w.probe, Some(ProbeOutcome::MissedNative));
        assert_eq!(w.steps, 3, "the probe is the first native draft");
        assert_eq!(
            w.prep,
            HeadPrepare::Window {
                owed: 0,
                head_at_start: false
            }
        );
        assert_eq!(w.rows, vec![20, 999, 998, 997]);
    }

    #[test]
    fn mesh_rank_copy_rank_model_forced_advances_realign_a_lagging_head_and_never_draw() {
        // An aligned head: rows [seed, f1, f2], all consumed, appended from
        // the start.
        let mut h = Harness::new();
        h.admit(120, 20, cycle_prompt(), false);
        h.script(0, &[], &[1, 2, 3]);
        let w = h.drive(Some(vec![41, 42, 43]));
        assert_eq!(w.rows, vec![20, 41, 42]);
        let d = &w.decision.lanes[0];
        assert!(d.forced && d.consumed_rows == 3 && d.emitted == vec![41, 42, 43]);
        assert_eq!(d.next_seed, 43);
        assert_eq!(
            w.prep,
            HeadPrepare::Window {
                owed: 0,
                head_at_start: true
            }
        );
        assert_eq!(h.sim.head, 123);
        // One forced token is a single AR-style row the head keeps up with.
        h.script(1, &[], &[9]);
        let w = h.drive(Some(vec![44]));
        assert_eq!(w.prep, HeadPrepare::SyncRow { owed: 0 });
        // A lagging head: the owed rows flush first, then the forced rows.
        let mut h = Harness::new();
        h.admit(30, 5, (1..=30).collect(), true);
        for (seq, pick) in [6u32, 7].into_iter().enumerate() {
            h.script(seq as u64, &[], &[pick]);
            assert_eq!(h.drive(None).prep, HeadPrepare::OweRow);
        }
        h.script(2, &[], &[1, 2, 3]);
        let w = h.drive(Some(vec![71, 72, 73]));
        assert_eq!(
            w.prep,
            HeadPrepare::Window {
                owed: 2,
                head_at_start: true
            },
            "owed rows ahead of the forced rows"
        );
        assert_eq!(h.sim.head, 32 + 3);
        assert!(h.sim.lag.is_empty());
    }
}
