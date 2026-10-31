// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The §4.2 step executor over the exact lanes
//! ([`crate::lane::Qwen4LaneStore`]): AR lanes, and MTP lanes when the store
//! is staged with a [`crate::lane::LaneSpecConfig`].
//!
//! A plan carries `Ar` requests (one row: the lane's pending seed at the
//! lane's position), `Verify { draft_len }` requests (`1 + draft_len` rows:
//! `[seed, drafts..]` of an MTP lane's drafted window) and `Prefill`
//! requests (one singleton prefill tile of the lane's prompt,
//! [`Qwen4Bundle::lane_prefill_tile_len`] rows). `Forced` rows are refused.
//! Per step:
//!
//! 0. [`Qwen4LaneExecutor::draft_step`] (MTP stores; AR lanes just answer
//!    `Ar`) routes the next window of every decoding lane and runs the
//!    step-synchronous batched draft of the `Batched` ones. A spec lane's
//!    decode row must be planned from its [`LaneDraft`]: `Verify` for
//!    `Batched`, `Ar` for the other routes.
//! 1. [`Qwen4LaneExecutor::provision_step`] validates the plan against the
//!    lane table (and the lanes' pending windows) and maps every lane's QSA
//!    arenas (an MTP lane's head arenas too) up to the step's end position.
//! 2. [`Qwen4LaneExecutor::forward_step`] runs the decode rows of all lanes
//!    as one exact trunk step (`forward_lanes`; each Verify lane under its
//!    armed ring and a target snapshot, the wide hidden captured), downloads
//!    the argmax picks of every decode row once, then, for each lane that
//!    carries a [`crate::lane::LaneSampling`], replaces them with the lane's
//!    sampler draws (below), finishes every Verify window (greedy accept
//!    over the picks, ring rollback, head commit), runs the head append of
//!    every `ArSynced` lane, then each prefill tile through the singleton
//!    chunk program on the lane's own state (an MTP lane's tile also fills
//!    its head). A pure-decode step's wall clock (draft start to the final
//!    sync) feeds the store's cost model, then every MTP lane's window
//!    policy observes its window.
//! 3. [`Qwen4LaneExecutor::commit_step`] applies the lane bookkeeping. The
//!    forwards publish the device and host trunk state (QSA lengths, PLE
//!    history, `state.position`) themselves, exactly as the singleton route
//!    does; commit never touches device state.
//!
//! Sampled lanes (`serve.batch_sampled`; [`crate::lane::LaneSampling`] set
//! after admission) draw every token they commit on the host with the AR
//! singleton's sampler (`llama::sample_top_k_p`, neutral penalties) from the
//! lane's own F32 logits row against the lane's own xorshift32 state, in
//! emission order: an AR row (`Ar`, `ArSynced`, `ArRetired`) is one draw, a
//! prompt-completing tile is one draw from its final row, and a `Verify`
//! window `[seed, d1..dk]` is the SpecInfer naive rule
//! (`spec_sampling::accept_naive_prefix`: drafts stay the head's argmax, each
//! verify row is one AR-sampler draw, a draft is accepted iff it equals its
//! row's draw, EOS stops as in the greedy rule), so every emitted token is
//! exactly one AR draw in AR order. The draws replace the lane's argmax
//! picks before the window is finished (the greedy accept over them commits
//! exactly the naive window; a disagreement is an error) and are the step
//! output's picks. A sampled lane costs one extra synchronous D2H per step
//! (its request's contiguous logits rows; the prompt-completing tile's row
//! likewise); a step with no sampled lane issues exactly the greedy device
//! work and downloads.
//!
//! Under an attached expert-parallel mesh every mutating entry below runs the
//! identical call on every rank (rank 0 on the caller's thread, peers on
//! their workers; `Qwen4Bundle::ep_lane_*`), and the host results (drafts,
//! step output, advances, errors) must agree across ranks. `forward_step`
//! shares the MoE row-gather barrier, so any failure there poisons the mesh;
//! the host-only entries report an error every rank agreed on without
//! poisoning it.
//!
//! Invariants of the lane plan (PLAN §3.2), and how each is enforced:
//!
//! - I1 (`state.position == mtp.position`): AR lanes carry no MTP state; the
//!   position every row is checked against is the lane's `state.position`.
//!   MTP lanes check it when a batched window is drafted, before an
//!   `ArSynced` head append, before each prefill tile, and at the end of
//!   every verify (`mtp_lanes`).
//! - I2 (one snapshot ticket, only in a verify): AR lanes snapshot nothing.
//!   An MTP lane holds one head ticket (taken at `draft_step`) and one target
//!   ticket (taken at `forward_step`) only inside a `Batched` window; both
//!   are committed by the window's finish, or dropped with the lane by
//!   poison/abort (the next admission's reset invalidates them).
//! - I3 (`row_capture_armed` only during a verify forward): armed per Verify
//!   lane right before the target snapshot and disarmed right after
//!   `forward_lanes` returns, success or not; `poison` and `abort_step`
//!   disarm too.
//! - I4 (mapping only at provision): `ensure_mapped_capacity` is called in
//!   `provision_step` and, for the head and trunk arenas a batched window's
//!   draft steps and append reach, in `draft_step`, which precedes the
//!   forward; the forwards run on mapped arenas.
//! - I5 (one plan identity across provision, forward, commit): the
//!   [`StepKey`] recorded at provision must equal the plan at forward and at
//!   commit, and commit also needs the step id forward issued. A failed
//!   forward poisons its participants and returns to idle.
//! - I6 (a lane step never touches `bundle.state`/`bundle.mtp`): lane work
//!   runs on the lanes' own states; a prefill tile or head step swaps the
//!   lane's states in for the call only (`Qwen4Bundle::with_lane`).
//! - I7 (rings are never shared): each MTP lane owns its `k + 1`-row ring.
//! - I8 (session bookkeeping keyed by epoch): lanes keep no session cache in
//!   this route; every lookup here is by the full epoch (tag and generation).
//! - I9 (arenas cover the step's end position before the first launch):
//!   `provision_step` maps `state.position + rows` for every request and
//!   refuses ends past `state.max_seq_len`.

use crate::bundle::Qwen4Bundle;
use crate::gpu_forward_lanes::{
    forward_lanes, row_group_count, LaneOutputs, LaneRowKind, LaneRows, LANE_MAX_ROWS,
};
use crate::lane::{
    download_lane_rows, lane_sample_step, LaneHost, LaneSampleOutcome, LaneSpecConfig, LaneStatus,
    Qwen4LaneState, Qwen4LaneStore, SpecStore, StepParts,
};
use crate::mtp_gpu::MtpAppendScratch;
use crate::mtp_lanes::{
    ar_synced_head, draft_lanes, finish_lane_window, prefill_tile, LaneCostModel,
};
use hipfire_runtime::slot_batch::{
    BatchStepPlan, RequestAdvance, RequestEpoch, RequestRows, RequestStepKind, StepOutput,
};
use hipfire_runtime::spec::MtpWindow;
use rdna_compute::tensor_ops::{argmax_f32, ArgmaxF32};
use rdna_compute::{Gpu, GpuTensor};
use std::time::Instant;

/// How a lane's next window runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneRoute {
    /// A non-spec lane: one AR row.
    Ar,
    /// An MTP lane's AR row followed by the head append of its seed (the
    /// singleton's calibration window): the head stays in step.
    ArSynced,
    /// An MTP lane retired to head-free AR rows.
    ArRetired,
    /// Up to `k_max` drafts verified in one `1 + drafts` row block.
    Batched { k_max: usize },
}

/// One lane's routed window, as `draft_step` returns it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneDraft {
    pub epoch: RequestEpoch,
    pub route: LaneRoute,
    /// The kept drafts (`Batched`: at least one; empty for every other route).
    pub drafts: Vec<u32>,
}

/// Plan identity recorded by provision/forward and checked by commit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StepKey {
    rows: Vec<(RequestEpoch, usize, usize, RequestStepKind)>,
    tokens: Vec<u32>,
    positions: Vec<i32>,
}

impl StepKey {
    pub(crate) fn of(plan: &BatchStepPlan) -> Self {
        Self {
            rows: plan
                .requests
                .iter()
                .map(|r| (r.epoch, r.rows.begin, r.rows.len, r.kind))
                .collect(),
            tokens: plan.batch.tokens.clone(),
            positions: plan.batch.positions.clone(),
        }
    }
}

/// Where the store's step is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum LanePhase {
    Idle,
    Provisioned(StepKey),
    Forwarded(StepKey, u64),
}

impl LanePhase {
    pub(crate) fn check_idle(&self) -> Result<(), String> {
        match self {
            Self::Idle => Ok(()),
            _ => Err("provision_step: previous step not committed or aborted".to_string()),
        }
    }

    pub(crate) fn check_draft(&self) -> Result<(), String> {
        match self {
            Self::Idle => Ok(()),
            _ => Err("draft_step: previous step not committed or aborted".to_string()),
        }
    }

    pub(crate) fn check_forward(&self, key: &StepKey) -> Result<(), String> {
        match self {
            Self::Provisioned(provisioned) if provisioned == key => Ok(()),
            _ => Err("forward_step: plan was not provisioned".to_string()),
        }
    }

    pub(crate) fn check_commit(&self, key: &StepKey, step_id: u64) -> Result<(), String> {
        match self {
            Self::Forwarded(forwarded, id) if forwarded == key && *id == step_id => Ok(()),
            _ => Err("commit_step: output does not belong to the forwarded plan".to_string()),
        }
    }
}

/// What plan validation reads of a lane's pending window.
struct WindowView<'a> {
    route: LaneRoute,
    /// The lane's position when the window was routed.
    position: usize,
    seed: u32,
    drafts: &'a [u32],
}

/// What plan validation reads of an MTP lane.
struct SpecView<'a> {
    window: Option<WindowView<'a>>,
}

/// What plan validation reads of one lane.
pub(crate) struct LaneView<'a> {
    slot: usize,
    host: &'a LaneHost,
    /// `state.position`: the next row's absolute position.
    position: usize,
    /// `state.max_seq_len`.
    max_seq_len: usize,
    /// `Some` on an MTP lane.
    spec: Option<SpecView<'a>>,
}

pub(crate) fn view_of(lane: &Qwen4LaneState) -> LaneView<'_> {
    LaneView {
        slot: lane.host.slot,
        host: &lane.host,
        position: lane.state.position,
        max_seq_len: lane.state.max_seq_len,
        spec: lane.spec.as_ref().map(|spec| SpecView {
            window: spec.window.as_ref().map(|w| WindowView {
                route: w.route,
                position: w.position,
                seed: w.seed,
                drafts: &w.drafts,
            }),
        }),
    }
}

#[derive(Clone, Copy)]
pub(crate) struct PlanLimits {
    pub(crate) max_lanes: usize,
    pub(crate) row_budget: usize,
}

/// Every host-only check of `provision_step`. `view` finds a live lane by
/// epoch; `prefill_len` is the singleton tile length at the lane's
/// `prefilled`. Returns each request's end position (`state.position +
/// rows`), in plan order.
pub(crate) fn validate_plan<'a>(
    plan: &BatchStepPlan,
    limits: PlanLimits,
    view: impl Fn(&RequestEpoch) -> Option<LaneView<'a>>,
    prefill_len: impl Fn(&LaneHost) -> usize,
) -> Result<Vec<usize>, String> {
    let b = &plan.batch;
    let n = b.total_rows();
    if n == 0 || plan.requests.is_empty() {
        return Err("provision_step: empty plan".to_string());
    }
    if b.m_per_slot.len() > limits.max_lanes
        || b.tokens.len() != n
        || b.positions.len() != n
        || b.row_slot.len() != n
    {
        return Err("provision_step: malformed SlotBatch".to_string());
    }
    if !b.pos3.is_empty() || b.ext_emb.iter().any(|&e| e >= 0) {
        return Err("provision_step: VL rows are not admitted on AR lanes".to_string());
    }
    let mut covered = vec![false; n];
    let mut seen_slots = vec![false; limits.max_lanes];
    let mut decode_lanes = 0usize;
    let mut completing = 0usize;
    let mut ends = Vec::with_capacity(plan.requests.len());
    for r in &plan.requests {
        if matches!(r.kind, RequestStepKind::Forced) {
            return Err(format!(
                "provision_step: {:?} rows are not admitted on AR lanes",
                r.kind
            ));
        }
        let lane = view(&r.epoch)
            .ok_or_else(|| format!("provision_step: stale or unknown epoch {:?}", r.epoch))?;
        let host = lane.host;
        if host.status == LaneStatus::Poisoned {
            return Err(format!("provision_step: epoch {:?} is poisoned", r.epoch));
        }
        if lane.slot >= limits.max_lanes {
            return Err(format!("provision_step: slot {} out of range", lane.slot));
        }
        if std::mem::replace(&mut seen_slots[lane.slot], true) {
            return Err(format!("provision_step: slot {} planned twice", lane.slot));
        }
        if r.rows.len == 0 || r.rows.end() > n {
            return Err(format!("provision_step: bad row range {:?}", r.rows));
        }
        if b.m_per_slot.get(lane.slot).copied() != Some(r.rows.len) {
            return Err(format!(
                "provision_step: m_per_slot[{}] disagrees with row range len {}",
                lane.slot, r.rows.len
            ));
        }
        let tokens = &b.tokens[r.rows.begin..r.rows.end()];
        match r.kind {
            RequestStepKind::Ar => {
                if r.rows.len != 1 {
                    return Err("provision_step: AR request must contribute one row".to_string());
                }
                if host.status != LaneStatus::Decoding {
                    return Err(format!(
                        "provision_step: AR row for {:?} lane {:?}",
                        host.status, r.epoch
                    ));
                }
                let seed = host.pending_seed.ok_or_else(|| {
                    format!("provision_step: epoch {:?} has no pending seed", r.epoch)
                })?;
                if tokens[0] != seed {
                    return Err(format!(
                        "provision_step: AR token {} != pending seed {seed} for {:?}",
                        tokens[0], r.epoch
                    ));
                }
                // An MTP lane's AR row is an `ArSynced`/`ArRetired` window.
                if let Some(spec) = &lane.spec {
                    let window = spec.window.as_ref().ok_or_else(|| {
                        format!(
                            "provision_step: MTP lane {:?} has no drafted window (draft_step first)",
                            r.epoch
                        )
                    })?;
                    if !matches!(window.route, LaneRoute::ArSynced | LaneRoute::ArRetired) {
                        return Err(format!(
                            "provision_step: AR row for MTP lane {:?} whose window is routed {:?}",
                            r.epoch, window.route
                        ));
                    }
                    if window.position != lane.position || window.seed != tokens[0] {
                        return Err(format!(
                            "provision_step: lane {:?} window (position {}, seed {}) is stale",
                            r.epoch, window.position, window.seed
                        ));
                    }
                }
                decode_lanes += 1;
            }
            RequestStepKind::Verify { draft_len } => {
                let Some(spec) = &lane.spec else {
                    return Err(format!(
                        "provision_step: Verify rows are not admitted on AR lanes (epoch {:?} has no MTP state)",
                        r.epoch
                    ));
                };
                if host.status != LaneStatus::Decoding {
                    return Err(format!(
                        "provision_step: Verify rows for {:?} lane {:?}",
                        host.status, r.epoch
                    ));
                }
                if draft_len == 0 || r.rows.len != draft_len + 1 || r.rows.len > LANE_MAX_ROWS {
                    return Err(format!(
                        "provision_step: Verify of {draft_len} drafts must contribute {} rows within {LANE_MAX_ROWS}, got {}",
                        draft_len + 1,
                        r.rows.len
                    ));
                }
                let seed = host.pending_seed.ok_or_else(|| {
                    format!("provision_step: epoch {:?} has no pending seed", r.epoch)
                })?;
                let window = spec.window.as_ref().ok_or_else(|| {
                    format!(
                        "provision_step: MTP lane {:?} has no drafted window (draft_step first)",
                        r.epoch
                    )
                })?;
                if !matches!(window.route, LaneRoute::Batched { .. }) {
                    return Err(format!(
                        "provision_step: Verify rows for MTP lane {:?} whose window is routed {:?}, not Batched",
                        r.epoch, window.route
                    ));
                }
                if tokens[0] != seed || window.seed != seed {
                    return Err(format!(
                        "provision_step: Verify seed {} != pending seed {seed} for {:?}",
                        tokens[0], r.epoch
                    ));
                }
                if window.drafts.len() != draft_len || tokens[1..] != *window.drafts {
                    return Err(format!(
                        "provision_step: Verify drafts of {:?} differ from its drafted window",
                        r.epoch
                    ));
                }
                if window.position != lane.position {
                    return Err(format!(
                        "provision_step: lane {:?} window position {} != state position {}",
                        r.epoch, window.position, lane.position
                    ));
                }
                decode_lanes += 1;
            }
            RequestStepKind::Prefill => {
                if host.status != LaneStatus::Prefilling {
                    return Err(format!(
                        "provision_step: prefill rows for {:?} lane {:?}",
                        host.status, r.epoch
                    ));
                }
                if lane.position != host.prefilled {
                    return Err(format!(
                        "provision_step: lane {:?} state position {} != prefilled {}",
                        r.epoch, lane.position, host.prefilled
                    ));
                }
                let want = prefill_len(host);
                if want == 0 || host.prefilled + want > host.prompt.len() {
                    return Err(format!(
                        "provision_step: no prefill tile left for {:?} ({} of {})",
                        r.epoch,
                        host.prefilled,
                        host.prompt.len()
                    ));
                }
                if r.rows.len != want {
                    return Err(format!(
                        "provision_step: prefill tile for {:?} is {} rows, the singleton route uses {want}",
                        r.epoch, r.rows.len
                    ));
                }
                if tokens != &host.prompt[host.prefilled..host.prefilled + want] {
                    return Err(format!(
                        "provision_step: prefill tokens for {:?} differ from its prompt",
                        r.epoch
                    ));
                }
                if host.prefilled + want == host.prompt.len() {
                    completing += 1;
                }
            }
            RequestStepKind::Forced => unreachable!(),
        }
        for (j, row) in (r.rows.begin..r.rows.end()).enumerate() {
            if std::mem::replace(&mut covered[row], true) {
                return Err(format!("provision_step: row {row} in two requests"));
            }
            if b.row_slot[row] != lane.slot as i32 {
                return Err(format!(
                    "provision_step: row {row} row_slot {} != owner slot {}",
                    b.row_slot[row], lane.slot
                ));
            }
            if i64::from(b.positions[row]) != (lane.position + j) as i64 {
                return Err(format!(
                    "provision_step: row {row} position {} != committed frontier {}",
                    b.positions[row],
                    lane.position + j
                ));
            }
        }
        let end = lane.position + r.rows.len;
        if end > lane.max_seq_len {
            return Err(format!(
                "provision_step: epoch {:?} needs {end} positions > context {}",
                r.epoch, lane.max_seq_len
            ));
        }
        ends.push(end);
    }
    if covered.iter().any(|c| !c) {
        return Err("provision_step: rows outside every request range".to_string());
    }
    if decode_lanes > limits.row_budget {
        return Err(format!(
            "provision_step: {decode_lanes} decode rows exceed row budget {}",
            limits.row_budget
        ));
    }
    if completing > limits.max_lanes {
        return Err(format!(
            "provision_step: {completing} prompt-completing tiles exceed {} lanes",
            limits.max_lanes
        ));
    }
    Ok(ends)
}

/// Whether `r` ends in a greedy pick: a decode row (AR or Verify), or the
/// tile that completes the prompt.
pub(crate) fn wants_pick(host: &LaneHost, r: &RequestRows) -> bool {
    match r.kind {
        RequestStepKind::Ar | RequestStepKind::Verify { .. } => true,
        RequestStepKind::Prefill => host.prefilled + r.rows.len == host.prompt.len(),
        RequestStepKind::Forced => false,
    }
}

/// Lane bookkeeping of one committed AR or Prefill request; returns the
/// committed id.
pub(crate) fn commit_host(host: &mut LaneHost, r: &RequestRows, pick: u32) -> Option<u32> {
    let picks = wants_pick(host, r);
    if r.kind == RequestStepKind::Prefill {
        host.prefilled += r.rows.len;
    }
    if !picks {
        return None;
    }
    if r.kind == RequestStepKind::Prefill {
        host.status = LaneStatus::Decoding;
    }
    host.generated.push(pick);
    host.pending_seed = Some(pick);
    Some(pick)
}

/// Lane bookkeeping of one committed Verify window: its committed tokens
/// (accepted drafts and the bonus, the seed excluded) extend the lane and
/// the last one is the next seed.
fn commit_verify(host: &mut LaneHost, committed: &[u32]) -> Option<u32> {
    let last = *committed.last()?;
    host.generated.extend_from_slice(committed);
    host.pending_seed = Some(last);
    Some(last)
}

/// The lane executor: borrows the bundle (weights, forward program) and its
/// staged [`Qwen4LaneStore`].
pub struct Qwen4LaneExecutor<'a> {
    pub(crate) bundle: &'a mut Qwen4Bundle,
}

impl Qwen4LaneExecutor<'_> {
    fn store(&self) -> Result<&Qwen4LaneStore, String> {
        self.bundle
            .lanes
            .as_ref()
            .ok_or_else(|| "Qwen4 lanes are not staged".to_string())
    }

    /// The singleton tile length the lane's next prefill step must use.
    pub fn exact_prefill_chunk_len(&self, epoch: &RequestEpoch) -> Result<usize, String> {
        let lane = self
            .store()?
            .lane(epoch)
            .ok_or_else(|| format!("exact_prefill_chunk_len: unknown or stale epoch {epoch:?}"))?;
        if lane.host.status != LaneStatus::Prefilling {
            return Err(format!(
                "exact_prefill_chunk_len: lane {epoch:?} is {:?}, not Prefilling",
                lane.host.status
            ));
        }
        Ok(self
            .bundle
            .lane_prefill_tile_len(&lane.host.prompt, lane.host.prefilled))
    }

    /// Route the next window of every listed Decoding lane (`budget`: most
    /// drafts allowed this window; 0 = AR) and run the step-synchronous
    /// batched draft for the `Batched` ones (one D2H per draft step for all
    /// lanes). Must precede `provision_step` whenever the plan holds an MTP
    /// lane's decode row, and every listed lane must then be planned:
    /// `Verify` for a `Batched` window (its kept drafts), `Ar` for the
    /// others. Non-spec lanes answer `Ar`. A failure poisons every listed
    /// lane.
    pub fn draft_step(
        &mut self,
        gpu: &mut Gpu,
        requests: &[(RequestEpoch, usize)],
    ) -> Result<Vec<LaneDraft>, String> {
        if self.bundle.ep.is_some() {
            return self.bundle.ep_lane_draft_step(gpu, requests);
        }
        self.draft_step_local(gpu, requests)
    }

    fn draft_step_local(
        &mut self,
        gpu: &mut Gpu,
        requests: &[(RequestEpoch, usize)],
    ) -> Result<Vec<LaneDraft>, String> {
        let mut store = self
            .bundle
            .lanes
            .take()
            .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
        let result = match store.phase.check_draft() {
            Ok(()) => {
                store.step_started = Some(Instant::now());
                let drafted = draft_lanes(self.bundle, gpu, &mut store, requests);
                if drafted.is_err() {
                    for (epoch, _) in requests {
                        store.poison(epoch);
                    }
                    store.step_started = None;
                }
                drafted
            }
            Err(error) => Err(error),
        };
        self.bundle.lanes = Some(store);
        result
    }

    /// Validate the plan against the lane table and map every lane's QSA
    /// arenas (an MTP lane's head arenas too) through the step's end
    /// position. Mapping happens here, never inside a forward (I4).
    pub fn provision_step(&mut self, gpu: &mut Gpu, plan: &BatchStepPlan) -> Result<(), String> {
        if self.bundle.ep.is_some() {
            return self.bundle.ep_lane_provision_step(gpu, plan);
        }
        self.provision_step_local(gpu, plan)
    }

    fn provision_step_local(&mut self, gpu: &mut Gpu, plan: &BatchStepPlan) -> Result<(), String> {
        let ends = {
            let bundle: &Qwen4Bundle = &*self.bundle;
            let store = bundle
                .lanes
                .as_ref()
                .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
            store.phase.check_idle()?;
            validate_plan(
                plan,
                PlanLimits {
                    max_lanes: store.max_lanes(),
                    row_budget: store.row_budget(),
                },
                |epoch| store.lane(epoch).map(view_of),
                |host| bundle.lane_prefill_tile_len(&host.prompt, host.prefilled),
            )?
        };
        let store = self
            .bundle
            .lanes
            .as_mut()
            .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
        // A step with no draft phase starts its wall clock here.
        store.step_started.get_or_insert_with(Instant::now);
        for (r, &end) in plan.requests.iter().zip(&ends) {
            let lane = store
                .lane_mut(&r.epoch)
                .ok_or_else(|| format!("provision_step: epoch {:?} vanished", r.epoch))?;
            lane.state.ensure_mapped_capacity(gpu, end).map_err(|e| {
                format!("provision_step: map {end} positions for {:?}: {e}", r.epoch)
            })?;
            if let Some(spec) = lane.spec.as_mut() {
                // A retired lane's head lags the trunk and is never touched.
                let retired = spec
                    .window
                    .as_ref()
                    .is_some_and(|w| w.route == LaneRoute::ArRetired);
                if !retired {
                    spec.head.ensure_mapped_capacity(gpu, end).map_err(|e| {
                        format!(
                            "provision_step: map {end} head positions for {:?}: {e}",
                            r.epoch
                        )
                    })?;
                }
            }
        }
        store.phase = LanePhase::Provisioned(StepKey::of(plan));
        Ok(())
    }

    /// Per-request decode map-ahead for a Decoding lane, once, before its
    /// first draft/decode step: map the lane's QSA arenas (an MTP lane's head
    /// arenas too, unless retired) through `min(position +
    /// remaining_new_tokens + draft rows, admitted context)`, so
    /// `provision_step` never remaps on the steady path. The draft rows are
    /// those of the widest Verify window the plan validator admits
    /// ([`LANE_MAX_ROWS`] rows), or the configured `k` if larger, not `k`
    /// alone. Mapping only grows capacity (no position or length commits) and
    /// `provision_step` still maps the actual step end on demand. Best effort:
    /// a failure (or an unknown / non-Decoding lane, a zero budget) is logged /
    /// skipped and `provision_step` grows on demand as before. Under EP every
    /// rank maps its own lane copy.
    pub fn map_ahead_decode(
        &mut self,
        gpu: &mut Gpu,
        epoch: &RequestEpoch,
        remaining_new_tokens: usize,
    ) {
        if self.bundle.ep.is_some() {
            self.bundle
                .ep_lane_map_ahead_decode(gpu, epoch, remaining_new_tokens);
            return;
        }
        self.map_ahead_decode_local(gpu, epoch, remaining_new_tokens);
    }

    fn map_ahead_decode_local(
        &mut self,
        gpu: &mut Gpu,
        epoch: &RequestEpoch,
        remaining_new_tokens: usize,
    ) {
        let Some(store) = self.bundle.lanes.as_mut() else {
            return;
        };
        let draft_rows = store.spec().map_or(0, |spec| {
            crate::state::map_ahead_draft_rows(spec.k, LANE_MAX_ROWS)
        });
        let Some(lane) = store
            .lane_mut(epoch)
            .filter(|lane| lane.host.status == LaneStatus::Decoding)
        else {
            return;
        };
        let Some(target) = crate::state::decode_map_ahead_tokens(
            lane.state.position,
            Some(remaining_new_tokens),
            draft_rows,
            lane.state.max_seq_len,
        ) else {
            return;
        };
        let mapped = lane
            .state
            .ensure_mapped_capacity(gpu, target)
            .map_err(|error| error.to_string())
            .and_then(
                |()| match lane.spec.as_mut().filter(|spec| !spec.stats.retired) {
                    Some(spec) => spec
                        .head
                        .ensure_mapped_capacity(gpu, target.min(spec.head.context_capacity()))
                        .map_err(|error| error.to_string()),
                    None => Ok(()),
                },
            );
        match mapped {
            Ok(()) => eprintln!(
                "  qwen4 QSA context mapped ahead for {target} tokens (decode budget {remaining_new_tokens})"
            ),
            Err(error) => eprintln!(
                "  qwen4 QSA map-ahead through {target} tokens failed ({error}); growing on demand"
            ),
        }
    }

    /// The step's trunk forward(s) and one pick per decode / prompt-completing
    /// row. Decode rows (AR and Verify) of all lanes run as one exact trunk
    /// step, every Verify window and `ArSynced` head append finishes, then
    /// each prefill tile runs the singleton chunk program on its own lane
    /// state: a decode row is never co-batched with a prefill tile. Waits
    /// for the picks (the download synchronizes). On error every participant
    /// is poisoned and the executor returns to idle.
    pub fn forward_step(
        &mut self,
        gpu: &mut Gpu,
        plan: &BatchStepPlan,
    ) -> Result<StepOutput, String> {
        if self.bundle.ep.is_some() {
            return self.bundle.ep_lane_forward_step(gpu, plan);
        }
        self.forward_step_local(gpu, plan)
    }

    fn forward_step_local(
        &mut self,
        gpu: &mut Gpu,
        plan: &BatchStepPlan,
    ) -> Result<StepOutput, String> {
        let key = StepKey::of(plan);
        let store = self
            .bundle
            .lanes
            .as_mut()
            .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
        store.phase.check_forward(&key)?;
        let step_id = store.next_step_id;
        store.phase = LanePhase::Forwarded(key, step_id);
        let mut store = self.bundle.lanes.take().expect("store checked above");
        let result = forward_inner(self.bundle, gpu, &mut store, plan);
        let output = match result {
            Ok(target_picks) => {
                store.next_step_id += 1;
                Ok(StepOutput {
                    step_id,
                    target_picks,
                })
            }
            Err(error) => {
                for r in &plan.requests {
                    store.poison(&r.epoch);
                }
                store.step_started = None;
                store.phase = LanePhase::Idle;
                Err(format!("forward_step: {error}"))
            }
        };
        self.bundle.lanes = Some(store);
        output
    }

    /// Apply the forwarded step's lane bookkeeping. Validates the plan
    /// identity, step id, pick count and every participant before mutating
    /// anything. A Verify request advances by its window's committed tokens
    /// (`committed_ids`, `accepted_drafts`, `verified_rows = drafts + 1`).
    pub fn commit_step(
        &mut self,
        gpu: &mut Gpu,
        plan: &BatchStepPlan,
        output: StepOutput,
    ) -> Result<Vec<RequestAdvance>, String> {
        if self.bundle.ep.is_some() {
            return self.bundle.ep_lane_commit_step(gpu, plan, output);
        }
        self.commit_step_local(gpu, plan, output)
    }

    fn commit_step_local(
        &mut self,
        _gpu: &mut Gpu,
        plan: &BatchStepPlan,
        output: StepOutput,
    ) -> Result<Vec<RequestAdvance>, String> {
        let store = self
            .bundle
            .lanes
            .as_mut()
            .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
        store
            .phase
            .check_commit(&StepKey::of(plan), output.step_id)?;
        if output.target_picks.len() != plan.total_rows() {
            return Err("commit_step: pick count != planned rows".to_string());
        }
        for r in &plan.requests {
            let lane = match store.lane(&r.epoch) {
                Some(lane) if lane.host.status != LaneStatus::Poisoned => lane,
                _ => {
                    return Err(format!(
                        "commit_step: epoch {:?} no longer owns its lane",
                        r.epoch
                    ));
                }
            };
            if r.rows.len == 0 || r.rows.end() > output.target_picks.len() {
                return Err(format!("commit_step: bad row range {:?}", r.rows));
            }
            if wants_pick(&lane.host, r) && output.target_picks[r.rows.end() - 1] == u32::MAX {
                return Err("commit_step: head row without a target pick".to_string());
            }
            let window = lane.spec.as_ref().and_then(|spec| spec.window.as_ref());
            match (r.kind, window) {
                (RequestStepKind::Verify { .. }, Some(w)) => {
                    if w.outcome.as_ref().map_or(true, |o| o.committed.is_empty()) {
                        return Err(format!(
                            "commit_step: Verify window of {:?} never finished",
                            r.epoch
                        ));
                    }
                }
                (RequestStepKind::Verify { .. }, None) => {
                    return Err(format!(
                        "commit_step: Verify of {:?} has no window",
                        r.epoch
                    ));
                }
                _ => {}
            }
        }
        let mut advances = Vec::with_capacity(plan.requests.len());
        for r in &plan.requests {
            let pick = output.target_picks[r.rows.end() - 1];
            let lane = store.lane_mut(&r.epoch).expect("validated above");
            let window = lane.spec.as_mut().and_then(|spec| spec.window.take());
            let (committed_ids, accepted_drafts, verified_rows) = match r.kind {
                RequestStepKind::Verify { draft_len } => {
                    let outcome: MtpWindow =
                        window.and_then(|w| w.outcome).expect("validated above");
                    commit_verify(&mut lane.host, &outcome.committed);
                    (outcome.committed, outcome.accepted, draft_len + 1)
                }
                _ => (
                    commit_host(&mut lane.host, r, pick).into_iter().collect(),
                    0,
                    0,
                ),
            };
            let position = lane.state.position;
            advances.push(RequestAdvance {
                epoch: r.epoch,
                committed_ids,
                committed_position: position,
                accepted_drafts,
                verified_rows,
                finish: (position >= lane.state.max_seq_len).then(|| "length".to_string()),
            });
        }
        store.step_started = None;
        store.phase = LanePhase::Idle;
        Ok(advances)
    }

    /// Drop the step: every participant is poisoned (its device state may be
    /// half-advanced), its pending window dropped, and the executor returns
    /// to idle.
    pub fn abort_step(&mut self, plan: &BatchStepPlan) {
        if self.bundle.ep.is_some() {
            self.bundle.ep_lane_abort_step(plan);
            return;
        }
        self.abort_step_local(plan);
    }

    fn abort_step_local(&mut self, plan: &BatchStepPlan) {
        if let Some(store) = self.bundle.lanes.as_mut() {
            for r in &plan.requests {
                store.poison(&r.epoch);
            }
            store.step_started = None;
            store.phase = LanePhase::Idle;
        }
    }
}

/// One planned request inside a forward.
struct Entry<'a> {
    r: &'a RequestRows,
    lane: &'a mut Qwen4LaneState,
    /// First decode row of a decode request (`None` for a prefill tile).
    dec: Option<usize>,
}

pub(crate) fn pick_at(host: &[u8], row: usize) -> u32 {
    let b = &host[row * 4..row * 4 + 4];
    u32::from_ne_bytes([b[0], b[1], b[2], b[3]])
}

/// The forward of a provisioned plan; the store is out of the bundle so the
/// lanes' states can be borrowed next to it.
fn forward_inner(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    store: &mut Qwen4LaneStore,
    plan: &BatchStepPlan,
) -> Result<Vec<u32>, String> {
    let vocab = bundle.lane_vocab();
    let epochs: Vec<RequestEpoch> = plan.requests.iter().map(|r| r.epoch).collect();
    let started = store.step_started.take().unwrap_or_else(Instant::now);
    let spec_config: Option<LaneSpecConfig> = store.spec();
    let StepParts {
        lanes,
        logits,
        top1,
        policy,
        top1_host,
        rows_cap,
        sample_host,
        spec,
    } = store.step_parts(&epochs)?;
    let mut hidden: Option<&GpuTensor> = None;
    let mut append: Option<&mut Option<MtpAppendScratch>> = None;
    let mut cost: Option<&mut LaneCostModel> = None;
    let mut skip_rollback = false;
    if let Some(spec) = spec {
        let SpecStore {
            hidden: h,
            append_scratch,
            cost: c,
            skip_rollback: skip,
            ..
        } = spec;
        hidden = Some(&*h);
        append = Some(append_scratch);
        cost = Some(c);
        skip_rollback = *skip;
    }

    // Plan order, with each decode request's first row among the step's
    // decode rows.
    let mut entries: Vec<Entry<'_>> = Vec::with_capacity(plan.requests.len());
    let (mut n_dec, mut n_prefill, mut decode_lanes, mut verify_windows, mut verified) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    // Row counts of the decode requests in entry order: what `forward_lanes`
    // groups into once-over-all-rows launches.
    let mut dec_rows: Vec<usize> = Vec::with_capacity(plan.requests.len());
    for (r, lane) in plan.requests.iter().zip(lanes) {
        let dec = match r.kind {
            RequestStepKind::Ar | RequestStepKind::Verify { .. } => {
                if let RequestStepKind::Verify { draft_len } = r.kind {
                    verify_windows += 1;
                    verified += draft_len;
                }
                let first = n_dec;
                n_dec += r.rows.len;
                dec_rows.push(r.rows.len);
                decode_lanes += 1;
                Some(first)
            }
            RequestStepKind::Prefill => {
                n_prefill += 1;
                None
            }
            RequestStepKind::Forced => {
                return Err(format!("{:?} rows are not admitted on AR lanes", r.kind));
            }
        };
        entries.push(Entry { r, lane, dec });
    }
    if n_dec > rows_cap {
        return Err(format!(
            "{n_dec} decode rows exceed the step buffers ({rows_cap})"
        ));
    }

    // Arm the ring, then take the target ticket, of every Verify lane.
    for e in entries.iter_mut() {
        let RequestStepKind::Verify { draft_len } = e.r.kind else {
            continue;
        };
        let rows = draft_len + 1;
        let tokens = &plan.batch.tokens[e.r.rows.begin..e.r.rows.end()];
        let Qwen4LaneState { state, spec, .. } = &mut *e.lane;
        let window = spec
            .as_mut()
            .and_then(|spec| spec.window.as_mut())
            .ok_or_else(|| format!("Verify lane {:?} has no window", e.r.epoch))?;
        if !matches!(window.route, LaneRoute::Batched { .. })
            || window.seed != tokens[0]
            || window.drafts != tokens[1..]
            || window.position != state.position
        {
            return Err(format!(
                "Verify lane {:?} disagrees with its window",
                e.r.epoch
            ));
        }
        if window.target_ticket.is_some() {
            return Err(format!(
                "Verify lane {:?} already holds a target ticket",
                e.r.epoch
            ));
        }
        if state.row_capture_rows() < rows {
            return Err(format!(
                "Verify lane {:?}: the ring holds {} rows, the verify needs {rows}",
                e.r.epoch,
                state.row_capture_rows()
            ));
        }
        // Armed before the snapshot: the GDN states then need no copy.
        state.row_capture_armed = true;
        let ticket =
            crate::mtp_lanes::lane_call(bundle.with_lane(state, None, |b| b.snapshot(gpu)));
        match ticket {
            Ok(ticket) => window.target_ticket = Some(crate::mtp_spec::TargetTicket::Local(ticket)),
            Err(error) => {
                state.row_capture_armed = false;
                return Err(format!("target snapshot of {:?}: {error}", e.r.epoch));
            }
        }
    }

    // One exact trunk step over every decode row, in plan order.
    if n_dec > 0 {
        let mut rows: Vec<LaneRows<'_>> = Vec::new();
        for e in entries.iter_mut() {
            let kind = match e.r.kind {
                RequestStepKind::Ar => LaneRowKind::Ar,
                RequestStepKind::Verify { draft_len } => LaneRowKind::Verify { draft_len },
                _ => continue,
            };
            rows.push(LaneRows {
                state: &mut e.lane.state,
                tokens: &plan.batch.tokens[e.r.rows.begin..e.r.rows.end()],
                kind,
            });
        }
        let out = LaneOutputs {
            logits: &logits.sub_offset(0, n_dec * vocab),
            top1: &top1.sub_offset(0, n_dec * std::mem::size_of::<i32>()),
            wide_hidden: hidden,
        };
        let forwarded =
            forward_lanes(gpu, bundle, &mut rows, &out, &policy).map_err(|e| e.to_string());
        drop(rows);
        // I3: disarmed on every path out of the forward.
        for e in entries.iter_mut() {
            if matches!(e.r.kind, RequestStepKind::Verify { .. }) {
                e.lane.state.row_capture_armed = false;
            }
        }
        forwarded?;
        gpu.hip
            .memcpy_dtoh(
                &mut top1_host[..n_dec * std::mem::size_of::<i32>()],
                &top1.buf,
            )
            .map_err(|e| e.to_string())?;
        gpu.take_device_wait_error().map_err(|e| e.to_string())?;
    }
    let mut picks = vec![u32::MAX; plan.total_rows()];
    for e in &entries {
        if let Some(first) = e.dec {
            for j in 0..e.r.rows.len {
                picks[e.r.rows.begin + j] = pick_at(top1_host, first + j);
            }
        }
    }

    // Verify windows, then the head appends of the ArSynced lanes. A sampled
    // lane's argmax picks are replaced by its sampler's draws first
    // (`lane_sample_step`: one D2H of the request's logits rows); a greedy lane
    // takes the unchanged argmax path.
    let lane_eos = spec_config
        .and_then(|config| config.end_of_turn)
        .unwrap_or(bundle.config.eos_token_id);
    for e in entries.iter_mut() {
        let Some(first) = e.dec else {
            continue;
        };
        let mut lane_picks: Vec<u32> = (0..e.r.rows.len)
            .map(|j| pick_at(top1_host, first + j))
            .collect();
        let mut naive = None;
        if e.lane.host.sampling.is_some() {
            let rows = e.r.rows.len;
            let drafts = matches!(e.r.kind, RequestStepKind::Verify { .. })
                .then(|| &plan.batch.tokens[e.r.rows.begin + 1..e.r.rows.end()]);
            let gpu_ref: &Gpu = gpu;
            let logits_ref: &GpuTensor = logits;
            let sampled = lane_sample_step(
                &mut e.lane.host,
                drafts,
                &mut lane_picks,
                lane_eos,
                vocab,
                sample_host,
                |scratch| download_lane_rows(gpu_ref, logits_ref, first, rows, vocab, scratch),
            )
            .map_err(|error| format!("lane {:?}: {error}", e.r.epoch))?;
            if let LaneSampleOutcome::Naive(window) = sampled {
                naive = Some(window);
            }
            // The step output's picks are the draws too.
            picks[e.r.rows.begin..e.r.rows.end()].copy_from_slice(&lane_picks);
        }
        let route = e
            .lane
            .spec
            .as_ref()
            .and_then(|spec| spec.window.as_ref())
            .map(|w| w.route);
        let outcome = match (e.r.kind, route) {
            (RequestStepKind::Verify { .. }, _) => {
                let config = spec_config
                    .ok_or_else(|| "Verify row in a store without MTP lanes".to_string())?;
                let hidden =
                    hidden.ok_or_else(|| "Verify row without a hidden capture".to_string())?;
                let window = finish_lane_window(
                    bundle,
                    gpu,
                    &mut *e.lane,
                    config,
                    hidden,
                    first,
                    &lane_picks,
                    skip_rollback,
                )?;
                if let Some(naive) = &naive {
                    if window.committed != naive.committed || window.accepted != naive.accepted {
                        return Err(format!(
                            "sampled Verify lane {:?}: the finish committed {:?} (accepted {}) \
                             but the naive window is {:?} (accepted {})",
                            e.r.epoch,
                            window.committed,
                            window.accepted,
                            naive.committed,
                            naive.accepted
                        ));
                    }
                }
                window
            }
            (RequestStepKind::Ar, Some(route)) => {
                if route == LaneRoute::ArSynced {
                    let hidden = hidden
                        .ok_or_else(|| "ArSynced row without a hidden capture".to_string())?;
                    ar_synced_head(bundle, gpu, &mut *e.lane, hidden, first)?;
                }
                MtpWindow {
                    committed: lane_picks,
                    accepted: 0,
                    drafts_generated: 0,
                }
            }
            _ => continue,
        };
        if let Some(window) = e.lane.spec.as_mut().and_then(|spec| spec.window.as_mut()) {
            window.outcome = Some(outcome);
        }
    }

    // Prefill tiles.
    // (prefill ordinal, plan row of its pick, entry index) of AR-lane
    // completing tiles.
    let mut final_tiles: Vec<(usize, usize, usize)> = Vec::new();
    let mut ordinal = 0usize;
    // A step's prefill tiles are one whole-prompt entry: on an EP
    // reserve-sharing load the donors are borrowed by every tile of more than
    // the resident rows and restored ONCE when the entry ends (the decode rows
    // of this step already ran, and the next step's decode rows need them
    // restored), not once per tile. A no-op on every other load.
    if n_prefill > 0 {
        bundle.share_prompt_begin();
    }
    let tiles: Result<(), String> = (|| {
        for (entry, e) in entries.iter_mut().enumerate() {
            if e.r.kind != RequestStepKind::Prefill {
                continue;
            }
            let i = ordinal;
            ordinal += 1;
            let tokens = &plan.batch.tokens[e.r.rows.begin..e.r.rows.end()];
            let completes = wants_pick(&e.lane.host, e.r);
            if e.lane.spec.is_some() {
                let scratch = append.as_deref_mut().and_then(Option::as_mut);
                let pick = prefill_tile(bundle, gpu, &mut *e.lane, tokens, scratch, sample_host)?;
                if completes {
                    picks[e.r.rows.end() - 1] = pick
                        .ok_or_else(|| "the prompt-completing tile produced no pick".to_string())?;
                }
            } else if completes {
                let row = logits.sub_offset((rows_cap + i) * vocab, vocab);
                bundle
                    .lane_prefill_tile(gpu, &mut e.lane.state, tokens, Some(&row))
                    .map_err(|e| e.to_string())?;
                let slot = top1.sub_offset(
                    (rows_cap + i) * std::mem::size_of::<i32>(),
                    std::mem::size_of::<i32>(),
                );
                argmax_f32(
                    gpu,
                    &ArgmaxF32 {
                        logits: &row,
                        indices: &slot,
                        rows: 1,
                        vocab,
                    },
                )
                .map_err(|e| e.to_string())?;
                final_tiles.push((i, e.r.rows.end() - 1, entry));
            } else {
                bundle
                    .lane_prefill_tile(gpu, &mut e.lane.state, tokens, None)
                    .map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    })();
    if n_prefill > 0 {
        let ended = bundle
            .share_prompt_end(gpu, tiles.is_ok())
            .map_err(|error| error.to_string());
        tiles?;
        ended?;
    } else {
        tiles?;
    }
    let prefill_ran = ordinal > 0;
    if !final_tiles.is_empty() {
        let used = (rows_cap + n_prefill) * std::mem::size_of::<i32>();
        gpu.hip
            .memcpy_dtoh(&mut top1_host[..used], &top1.buf)
            .map_err(|e| e.to_string())?;
        gpu.take_device_wait_error().map_err(|e| e.to_string())?;
        for &(i, row, entry) in &final_tiles {
            picks[row] = pick_at(top1_host, rows_cap + i);
            // A sampled lane's prompt-completing pick is its first draw, from
            // the tile's final logits row (one D2H); a greedy lane keeps the
            // argmax.
            let lane = &mut *entries[entry].lane;
            if lane.host.sampling.is_some() {
                let gpu_ref: &Gpu = gpu;
                let logits_ref: &GpuTensor = logits;
                let mut one = [picks[row]];
                lane_sample_step(
                    &mut lane.host,
                    None,
                    &mut one,
                    lane_eos,
                    vocab,
                    sample_host,
                    |scratch| {
                        download_lane_rows(gpu_ref, logits_ref, rows_cap + i, 1, vocab, scratch)
                    },
                )
                .map_err(|error| format!("lane {:?}: {error}", lane.host.epoch))?;
                picks[row] = one[0];
            }
        }
    }

    // The step's wall clock feeds the cost model only on a pure-decode
    // step; every MTP lane's policy observes its window either way.
    if let Some(cost) = cost {
        if decode_lanes > 0 {
            // Row groups beyond the first are the step's spill; each Batched
            // window is priced at its share of it.
            let spill = row_group_count(&dec_rows).saturating_sub(1);
            let spill_per_window = spill as f64 / verify_windows.max(1) as f64;
            if !prefill_ran {
                gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
                let wall_us = started.elapsed().as_secs_f64() * 1e6;
                cost.observe(decode_lanes, spill, verified, wall_us as f32);
            }
            for e in entries.iter_mut() {
                if e.dec.is_none() {
                    continue;
                }
                let Some(spec) = e.lane.spec.as_mut() else {
                    continue;
                };
                let (Some(window), policy, stats) =
                    (spec.window.as_ref(), &mut spec.policy, &mut spec.stats)
                else {
                    continue;
                };
                let Some(outcome) = window.outcome.as_ref() else {
                    continue;
                };
                match window.route {
                    LaneRoute::Batched { k_max } => {
                        let units = cost.window_units(
                            decode_lanes,
                            outcome.drafts_generated,
                            spill_per_window,
                        );
                        policy.observe_batched(outcome, units, k_max, window.budget);
                        stats.windows += 1;
                        stats.drafts += outcome.drafts_generated as u64;
                        stats.accepted += outcome.accepted as u64;
                    }
                    LaneRoute::ArSynced => {
                        policy.observe_ar_synced(outcome, window.budget);
                        stats.windows += 1;
                    }
                    LaneRoute::ArRetired => stats.ar_windows += 1,
                    LaneRoute::Ar => {}
                }
                stats.retired = policy.retired();
            }
        }
    }
    Ok(picks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::slot_batch::RowRange;
    use std::collections::HashMap;

    const MAX_LANES: usize = 4;
    const ROW_BUDGET: usize = 2;
    const CONTEXT: usize = 64;
    const TILE: usize = 4;

    fn epoch(tag: u64) -> RequestEpoch {
        RequestEpoch {
            request_tag: tag,
            owner_generation: 1,
        }
    }

    /// A pending window of an MTP lane.
    #[derive(Clone)]
    struct TestWindow {
        route: LaneRoute,
        position: usize,
        seed: u32,
        drafts: Vec<u32>,
    }

    struct Lane {
        host: LaneHost,
        position: usize,
        /// `Some`: an MTP lane (with its pending window, if any).
        spec: Option<Option<TestWindow>>,
    }

    fn host(
        tag: u64,
        slot: usize,
        status: LaneStatus,
        prompt_len: usize,
        prefilled: usize,
    ) -> LaneHost {
        LaneHost {
            epoch: epoch(tag),
            slot,
            prompt: (0..prompt_len as u32).map(|t| 100 + t).collect(),
            prefilled,
            generated: Vec::new(),
            pending_seed: (status == LaneStatus::Decoding).then_some(77),
            status,
            stamp: tag,
            sampling: None,
            rng: 0,
        }
    }

    /// Lane 1 decodes (seed 77 at position 10), lane 2 decodes (seed 77 at
    /// position 20), lane 3 prefills tile 2 of a 10-token prompt.
    fn lanes() -> HashMap<RequestEpoch, Lane> {
        let mut lanes = HashMap::new();
        for (tag, slot, status, prompt, prefilled, position) in [
            (1, 0, LaneStatus::Decoding, 10, 10, 10),
            (2, 1, LaneStatus::Decoding, 20, 20, 20),
            (3, 2, LaneStatus::Prefilling, 10, 4, 4),
        ] {
            lanes.insert(
                epoch(tag),
                Lane {
                    host: host(tag, slot, status, prompt, prefilled),
                    position,
                    spec: None,
                },
            );
        }
        lanes
    }

    /// `lanes()` with lane 1 an MTP lane holding `window`.
    fn spec_lanes(window: Option<TestWindow>) -> HashMap<RequestEpoch, Lane> {
        let mut table = lanes();
        table.get_mut(&epoch(1)).unwrap().spec = Some(window);
        table
    }

    fn batched_window() -> TestWindow {
        TestWindow {
            route: LaneRoute::Batched { k_max: 3 },
            position: 10,
            seed: 77,
            drafts: vec![5, 6],
        }
    }

    struct Entry {
        tag: u64,
        slot: usize,
        kind: RequestStepKind,
        tokens: Vec<u32>,
        start: usize,
    }

    fn ar(tag: u64, slot: usize, token: u32, position: usize) -> Entry {
        Entry {
            tag,
            slot,
            kind: RequestStepKind::Ar,
            tokens: vec![token],
            start: position,
        }
    }

    fn verify(tag: u64, slot: usize, tokens: &[u32], position: usize) -> Entry {
        Entry {
            tag,
            slot,
            kind: RequestStepKind::Verify {
                draft_len: tokens.len() - 1,
            },
            tokens: tokens.to_vec(),
            start: position,
        }
    }

    fn tile(tag: u64, slot: usize, tokens: &[u32], start: usize) -> Entry {
        Entry {
            tag,
            slot,
            kind: RequestStepKind::Prefill,
            tokens: tokens.to_vec(),
            start,
        }
    }

    fn plan_of(entries: &[Entry]) -> BatchStepPlan {
        let mut plan = BatchStepPlan::default();
        plan.batch.m_per_slot = vec![0; MAX_LANES];
        for e in entries {
            let begin = plan.batch.tokens.len();
            plan.batch.m_per_slot[e.slot] = e.tokens.len();
            for (j, &t) in e.tokens.iter().enumerate() {
                plan.batch.tokens.push(t);
                plan.batch.positions.push((e.start + j) as i32);
                plan.batch.row_slot.push(e.slot as i32);
            }
            plan.requests.push(RequestRows {
                epoch: epoch(e.tag),
                rows: RowRange {
                    begin,
                    len: e.tokens.len(),
                },
                kind: e.kind,
            });
        }
        plan
    }

    fn validate(
        plan: &BatchStepPlan,
        lanes: &HashMap<RequestEpoch, Lane>,
    ) -> Result<Vec<usize>, String> {
        validate_plan(
            plan,
            PlanLimits {
                max_lanes: MAX_LANES,
                row_budget: ROW_BUDGET,
            },
            |e| {
                lanes.get(e).map(|l| LaneView {
                    slot: l.host.slot,
                    host: &l.host,
                    position: l.position,
                    max_seq_len: CONTEXT,
                    spec: l.spec.as_ref().map(|window| SpecView {
                        window: window.as_ref().map(|w| WindowView {
                            route: w.route,
                            position: w.position,
                            seed: w.seed,
                            drafts: &w.drafts,
                        }),
                    }),
                })
            },
            |h| TILE.min(h.prompt.len() - h.prefilled),
        )
    }

    /// Prompt tokens of lane 3's second tile (prompt 100.., prefilled 4).
    fn tile3() -> Vec<u32> {
        vec![104, 105, 106, 107]
    }

    fn valid_entries() -> Vec<Entry> {
        vec![ar(1, 0, 77, 10), ar(2, 1, 77, 20), tile(3, 2, &tile3(), 4)]
    }

    fn refused(entries: &[Entry], needle: &str) {
        refused_on(entries, &lanes(), needle);
    }

    fn refused_on(entries: &[Entry], table: &HashMap<RequestEpoch, Lane>, needle: &str) {
        let error = validate(&plan_of(entries), table).unwrap_err();
        assert!(error.contains(needle), "{error:?} lacks {needle:?}");
    }

    #[test]
    fn a_mixed_ar_and_prefill_plan_is_accepted() {
        let plan = plan_of(&valid_entries());
        assert_eq!(validate(&plan, &lanes()), Ok(vec![11, 21, 8]));
    }

    #[test]
    fn ar_rows_are_exactly_one_token_at_the_lane_frontier() {
        let wide = Entry {
            tag: 1,
            slot: 0,
            kind: RequestStepKind::Ar,
            tokens: vec![77, 78],
            start: 10,
        };
        refused(&[wide], "one row");
        refused(&[ar(1, 0, 78, 10)], "pending seed");
        refused(&[ar(1, 0, 77, 11)], "position");
        refused(&[ar(1, 1, 77, 10)], "m_per_slot");
        // Wrong row_slot with a consistent m_per_slot.
        let mut plan = plan_of(&[ar(1, 0, 77, 10)]);
        plan.batch.row_slot[0] = 1;
        let error = validate(&plan, &lanes()).unwrap_err();
        assert!(error.contains("row_slot"), "{error}");
        // An AR row for a lane still prefilling.
        refused(&[ar(3, 2, 77, 4)], "Prefilling");
    }

    #[test]
    fn prefill_tiles_follow_the_singleton_schedule_and_prompt() {
        refused(&[tile(3, 2, &tile3()[..3], 4)], "singleton route uses 4");
        refused(
            &[tile(3, 2, &[104, 105, 106, 999], 4)],
            "differ from its prompt",
        );
        refused(&[tile(3, 2, &tile3(), 5)], "position");
        // A prefill row for a decoding lane.
        refused(&[tile(1, 0, &[1, 2, 3, 4], 10)], "Decoding");
    }

    #[test]
    fn forced_rows_are_not_admitted() {
        let mut entry = ar(1, 0, 77, 10);
        entry.kind = RequestStepKind::Forced;
        refused(&[entry], "not admitted on AR lanes");
    }

    #[test]
    fn verify_rows_need_an_mtp_lane() {
        // Lane 1 is an AR lane here.
        refused(&[verify(1, 0, &[77, 5, 6], 10)], "not admitted on AR lanes");
    }

    #[test]
    fn a_verify_row_matching_the_pending_window_is_accepted() {
        let table = spec_lanes(Some(batched_window()));
        let plan = plan_of(&[
            verify(1, 0, &[77, 5, 6], 10),
            ar(2, 1, 77, 20),
            tile(3, 2, &tile3(), 4),
        ]);
        assert_eq!(validate(&plan, &table), Ok(vec![13, 21, 8]));
        // One draft is fine too, when that is what the window kept.
        let mut one = batched_window();
        one.drafts = vec![5];
        let table = spec_lanes(Some(one));
        assert_eq!(
            validate(&plan_of(&[verify(1, 0, &[77, 5], 10)]), &table),
            Ok(vec![12])
        );
    }

    #[test]
    fn a_verify_row_must_equal_its_window() {
        let table = spec_lanes(Some(batched_window()));
        // Drafts differ.
        refused_on(
            &[verify(1, 0, &[77, 5, 9], 10)],
            &table,
            "differ from its drafted window",
        );
        // Fewer drafts than the window kept.
        refused_on(
            &[verify(1, 0, &[77, 5], 10)],
            &table,
            "differ from its drafted window",
        );
        // Seed differs.
        refused_on(&[verify(1, 0, &[78, 5, 6], 10)], &table, "pending seed");
        // Rows at another position.
        refused_on(&[verify(1, 0, &[77, 5, 6], 11)], &table, "position");
        // draft_len disagrees with the row count.
        let mut lying = verify(1, 0, &[77, 5, 6], 10);
        lying.kind = RequestStepKind::Verify { draft_len: 1 };
        refused_on(&[lying], &table, "must contribute");
        let mut empty = verify(1, 0, &[77, 5, 6], 10);
        empty.kind = RequestStepKind::Verify { draft_len: 0 };
        refused_on(&[empty], &table, "must contribute");
        // A window drafted at another position is stale.
        let mut stale = batched_window();
        stale.position = 9;
        refused_on(
            &[verify(1, 0, &[77, 5, 6], 10)],
            &spec_lanes(Some(stale)),
            "window position",
        );
        // No window, or a window of another route.
        refused_on(
            &[verify(1, 0, &[77, 5, 6], 10)],
            &spec_lanes(None),
            "no drafted window",
        );
        let mut synced = batched_window();
        synced.route = LaneRoute::ArSynced;
        synced.drafts = Vec::new();
        refused_on(
            &[verify(1, 0, &[77, 5, 6], 10)],
            &spec_lanes(Some(synced)),
            "not Batched",
        );
        // A verify for a lane still prefilling.
        let mut table = spec_lanes(Some(batched_window()));
        table.get_mut(&epoch(3)).unwrap().spec = Some(None);
        refused_on(&[verify(3, 2, &[77, 5], 4)], &table, "Prefilling");
    }

    #[test]
    fn an_ar_row_of_an_mtp_lane_needs_a_pending_ar_route() {
        refused_on(&[ar(1, 0, 77, 10)], &spec_lanes(None), "no drafted window");
        refused_on(
            &[ar(1, 0, 77, 10)],
            &spec_lanes(Some(batched_window())),
            "routed",
        );
        for route in [LaneRoute::ArSynced, LaneRoute::ArRetired] {
            let table = spec_lanes(Some(TestWindow {
                route,
                position: 10,
                seed: 77,
                drafts: Vec::new(),
            }));
            assert_eq!(
                validate(&plan_of(&[ar(1, 0, 77, 10)]), &table),
                Ok(vec![11]),
                "{route:?}"
            );
        }
        // A window routed at another position is stale.
        let stale = spec_lanes(Some(TestWindow {
            route: LaneRoute::ArSynced,
            position: 9,
            seed: 77,
            drafts: Vec::new(),
        }));
        refused_on(&[ar(1, 0, 77, 10)], &stale, "stale");
        // A non-spec lane needs no window.
        assert!(validate(&plan_of(&[ar(2, 1, 77, 20)]), &spec_lanes(None)).is_ok());
    }

    #[test]
    fn overlapping_and_uncovered_rows_are_refused() {
        let mut plan = plan_of(&[ar(1, 0, 77, 10), ar(2, 1, 77, 20)]);
        plan.requests[1].rows = RowRange { begin: 0, len: 1 };
        let error = validate(&plan, &lanes()).unwrap_err();
        assert!(error.contains("two requests"), "{error}");

        let mut plan = plan_of(&[ar(1, 0, 77, 10), ar(2, 1, 77, 20)]);
        plan.requests.pop();
        plan.batch.m_per_slot[1] = 0;
        let error = validate(&plan, &lanes()).unwrap_err();
        assert!(error.contains("outside every request"), "{error}");

        let mut plan = plan_of(&[ar(1, 0, 77, 10)]);
        plan.requests[0].rows = RowRange { begin: 1, len: 1 };
        assert!(validate(&plan, &lanes()).is_err());

        let mut plan = plan_of(&[ar(1, 0, 77, 10)]);
        plan.requests.push(plan.requests[0]);
        assert!(validate(&plan, &lanes())
            .unwrap_err()
            .contains("planned twice"));
    }

    #[test]
    fn unknown_stale_and_poisoned_epochs_are_refused() {
        refused(&[ar(9, 0, 77, 10)], "stale or unknown");
        let mut plan = plan_of(&[ar(1, 0, 77, 10)]);
        plan.requests[0].epoch.owner_generation = 2;
        assert!(validate(&plan, &lanes())
            .unwrap_err()
            .contains("stale or unknown"));
        let mut table = lanes();
        table.get_mut(&epoch(1)).unwrap().host.status = LaneStatus::Poisoned;
        let error = validate(&plan_of(&[ar(1, 0, 77, 10)]), &table).unwrap_err();
        assert!(error.contains("poisoned"), "{error}");
    }

    #[test]
    fn the_row_budget_limits_decode_lanes_only() {
        let mut table = lanes();
        for (tag, slot) in [(4u64, 3usize)] {
            table.insert(
                epoch(tag),
                Lane {
                    host: host(tag, slot, LaneStatus::Decoding, 5, 5),
                    position: 5,
                    spec: None,
                },
            );
        }
        let three = [ar(1, 0, 77, 10), ar(2, 1, 77, 20), ar(4, 3, 77, 5)];
        let error = validate(&plan_of(&three), &table).unwrap_err();
        assert!(error.contains("row budget"), "{error}");
        // The prefill tile does not count against it.
        assert!(validate(&plan_of(&valid_entries()), &table).is_ok());
        // A verify lane counts as one decode lane, however many rows it has.
        let mut spec = spec_lanes(Some(batched_window()));
        spec.insert(
            epoch(4),
            Lane {
                host: host(4, 3, LaneStatus::Decoding, 5, 5),
                position: 5,
                spec: None,
            },
        );
        let plan = plan_of(&[verify(1, 0, &[77, 5, 6], 10), ar(2, 1, 77, 20)]);
        assert!(validate(&plan, &spec).is_ok());
        let plan = plan_of(&[
            verify(1, 0, &[77, 5, 6], 10),
            ar(2, 1, 77, 20),
            ar(4, 3, 77, 5),
        ]);
        assert!(validate(&plan, &spec).unwrap_err().contains("row budget"));
    }

    #[test]
    fn rows_past_the_context_are_refused() {
        let mut table = lanes();
        table.get_mut(&epoch(1)).unwrap().position = CONTEXT;
        let error = validate(&plan_of(&[ar(1, 0, 77, CONTEXT)]), &table).unwrap_err();
        assert!(error.contains("context"), "{error}");
        // A verify block reaching past it is refused too.
        let mut spec = spec_lanes(Some(TestWindow {
            position: CONTEXT - 2,
            ..batched_window()
        }));
        spec.get_mut(&epoch(1)).unwrap().position = CONTEXT - 2;
        let error =
            validate(&plan_of(&[verify(1, 0, &[77, 5, 6], CONTEXT - 2)]), &spec).unwrap_err();
        assert!(error.contains("context"), "{error}");
    }

    #[test]
    fn malformed_batches_are_refused() {
        assert!(validate(&BatchStepPlan::default(), &lanes())
            .unwrap_err()
            .contains("empty plan"));
        let mut plan = plan_of(&[ar(1, 0, 77, 10)]);
        plan.batch.positions.clear();
        assert!(validate(&plan, &lanes()).unwrap_err().contains("malformed"));
        let mut plan = plan_of(&[ar(1, 0, 77, 10)]);
        plan.batch.ext_emb = vec![0];
        assert!(validate(&plan, &lanes()).is_err());
        let mut plan = plan_of(&[ar(1, 0, 77, 10)]);
        plan.batch.pos3 = vec![[10, 10, 10]];
        assert!(validate(&plan, &lanes()).is_err());
    }

    #[test]
    fn the_phase_machine_pins_forward_and_commit_to_the_provisioned_plan() {
        let plan = plan_of(&valid_entries());
        let other = plan_of(&[ar(1, 0, 77, 10)]);
        let (key, other_key) = (StepKey::of(&plan), StepKey::of(&other));
        assert_ne!(key, other_key);

        // Forward before provision is refused.
        assert!(LanePhase::Idle.check_forward(&key).is_err());
        // A provisioned phase forwards only its own plan.
        let provisioned = LanePhase::Provisioned(key.clone());
        assert!(provisioned.check_forward(&key).is_ok());
        assert!(provisioned.check_forward(&other_key).is_err());
        assert!(provisioned.check_idle().is_err());
        assert!(LanePhase::Idle.check_idle().is_ok());
        // Drafting needs an idle store.
        assert!(LanePhase::Idle.check_draft().is_ok());
        assert!(provisioned.check_draft().is_err());
        // Commit needs the forwarded phase, the same plan and the step id.
        assert!(provisioned.check_commit(&key, 0).is_err());
        let forwarded = LanePhase::Forwarded(key.clone(), 7);
        assert!(forwarded.check_commit(&key, 7).is_ok());
        assert!(forwarded.check_commit(&key, 8).is_err());
        assert!(forwarded.check_commit(&other_key, 7).is_err());
        assert!(LanePhase::Idle.check_commit(&key, 7).is_err());
        // A changed token or position changes the key.
        let mut moved = plan.clone();
        moved.batch.positions[0] += 1;
        assert_ne!(StepKey::of(&moved), key);
        let mut retokened = plan.clone();
        retokened.batch.tokens[0] += 1;
        assert_ne!(StepKey::of(&retokened), key);
    }

    #[test]
    fn commit_bookkeeping_advances_decode_and_prefill_lanes() {
        let plan = plan_of(&valid_entries());
        let mut table = lanes();
        // AR: the pick becomes the next seed.
        let h = &mut table.get_mut(&epoch(1)).unwrap().host;
        assert_eq!(commit_host(h, &plan.requests[0], 5), Some(5));
        assert_eq!((h.pending_seed, h.generated.clone()), (Some(5), vec![5]));
        // A non-final tile only advances `prefilled`.
        let h = &mut table.get_mut(&epoch(3)).unwrap().host;
        assert!(!wants_pick(h, &plan.requests[2]));
        assert_eq!(commit_host(h, &plan.requests[2], u32::MAX), None);
        assert_eq!(
            (h.prefilled, h.status, h.pending_seed),
            (8, LaneStatus::Prefilling, None)
        );
        // The prompt-completing tile flips the lane to Decoding with a seed.
        let last = plan_of(&[tile(3, 2, &[108, 109], 8)]);
        assert!(wants_pick(h, &last.requests[0]));
        assert_eq!(commit_host(h, &last.requests[0], 42), Some(42));
        assert_eq!(
            (h.prefilled, h.status, h.pending_seed, h.generated.clone()),
            (10, LaneStatus::Decoding, Some(42), vec![42])
        );
    }

    #[test]
    fn a_committed_verify_window_extends_the_lane_and_seeds_the_last_token() {
        let plan = plan_of(&[verify(1, 0, &[77, 5, 6], 10)]);
        assert!(wants_pick(
            &host(1, 0, LaneStatus::Decoding, 10, 10),
            &plan.requests[0]
        ));
        let mut h = host(1, 0, LaneStatus::Decoding, 10, 10);
        h.generated = vec![77];
        assert_eq!(commit_verify(&mut h, &[5, 6, 8]), Some(8));
        assert_eq!(
            (h.pending_seed, h.generated.clone()),
            (Some(8), vec![77, 5, 6, 8])
        );
        // A window that committed nothing changes nothing.
        assert_eq!(commit_verify(&mut h, &[]), None);
        assert_eq!(h.generated, vec![77, 5, 6, 8]);
    }
}
