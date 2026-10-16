// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native MTP windows on the exact lanes of the Flash-Next batch route.
//!
//! Every window of a lane is byte-identical to one singleton window on that
//! lane's own states (`fn-batch/S5-CONTRACT.md`, exactness model):
//!
//! | [`LaneRoute`] | singleton twin |
//! |---|---|
//! | `Batched { k_max }` | `mtp_step`'s batched route at depth `k_max` |
//! | `ArSynced` | `mtp_ar_step(calibrate = true)` without the timing |
//! | `ArRetired` | `mtp_ar_step(calibrate = false)` |
//! | `Ar` | non-spec lane: the S3 AR row |
//!
//! The phases the executor ([`crate::lane_exec`]) runs around its one trunk
//! forward:
//!
//! - [`draft_lanes`]: route every lane and run the step-synchronous batched
//!   draft (every still-drafting lane enqueues its draft step, then one D2H
//!   reads all of them).
//! - [`finish_lane_window`]: a Verify lane's greedy accept and the singleton's
//!   window tail ([`crate::mtp_spec::finish_native_window`]).
//! - [`ar_synced_head`]: the head append of an `ArSynced` lane.
//! - [`prefill_tile`]: a spec lane's prompt-fill chunk (trunk tile with wide
//!   hidden capture plus the head append of every row).
//!
//! The per-lane floor ([`LaneSpecPolicy`]) keeps the singleton's
//! [`MtpFloor`] and agreement, fed in AR-row units from the store-level
//! [`LaneCostModel`] instead of wall clocks (see its docs).

use crate::bundle::{BundleError, Qwen4Bundle};
use crate::lane::Qwen4LaneStore;
use crate::lane::{LabRoutes, LaneSpec, LaneSpecConfig, Qwen4LaneState};
use crate::lane_exec::{LaneDraft, LaneRoute};
use crate::mtp_gpu::MtpAppendScratch;
use crate::mtp_spec::{
    accept_native_greedy, batched_depth, draft_accept_estimate, draft_pairing,
    finish_native_window, mtp_prefill_chunk, mtp_prefill_chunk_needs_pick,
    mtp_reuse_prefill_ids_enabled, mtp_skip_intermediate_pick_enabled, observe_agreement,
    Agreement, DraftPairing, FloorRoute, HeadTicket, MtpFloor, NativeFinish, NativeWindow,
    Qwen4Ops, TargetTicket, WindowStats, DRAFT_ACCEPT_MAX, MTP_AGREEMENT_DECAY,
    MTP_AGREEMENT_PRIOR, MTP_MAX_DEPTH, MTP_ROW_WORTH,
};

use hipfire_runtime::slot_batch::RequestEpoch;
use hipfire_runtime::spec::{GreedyAccept, MtpWindow};
use rdna_compute::{Gpu, GpuTensor};

// ---------------------------------------------------------------------------
// Lane cost model
// ---------------------------------------------------------------------------

/// Prior `[F, a, r, G]` (µs): the fixed cost of a step, the cost per
/// decoding lane, per verified draft row, and per row-group spill. From the
/// C2 and C8 serve step traces on gfx1151 (PM rows on): 8 AR lanes 91.7 ms
/// median, 8 lanes with one 2.7-draft window 137.4 ms, 2 lanes going from
/// one to two windows +17.2 ms; it reproduces 8 lanes all drafting (295.5 ms)
/// and 7 (269.8 ms).
const COST_PRIOR_US: [f64; 4] = [16_900.0, 9_700.0, 8_200.0, 18_500.0];
/// Weight of the ridge prior (never decays), in typical observations.
const COST_PRIOR_WEIGHT: f64 = 8.0;
/// Per-parameter scale of the prior's weight: the squared typical regressor
/// (`C` about 8, `e` about 3 per window, `s` about 1), so each parameter's
/// prior counts about [`COST_PRIOR_WEIGHT`] observations.
const COST_PRIOR_SCALE: [f64; 4] = [1.0, 64.0, 9.0, 1.0];
/// Weight kept per later observation.
const COST_DECAY: f64 = 0.99;
/// `α(C)` never fits below this.
const COST_ALPHA_FLOOR_US: f64 = 1_000.0;

/// Store-level model of a pure-decode step's wall time `W` (µs, draft start
/// to accept end, synced):
///
/// `W = F + a·C + r·e + G·s`
///
/// with `C` the decoding lanes of the step (AR and Verify), `e` its verified
/// drafts (the sum of the `draft_len`s) and `s` its spill: the row groups of
/// its decode lanes (`gpu_forward_lanes::row_group_count`) beyond the first.
/// One fit is pooled over every lane count: decayed sufficient statistics
/// `Σ λⁿ x xᵀ` and `Σ λⁿ x W` of `x = [1, C, e, s]`, plus a constant ridge
/// prior [`COST_PRIOR_US`] (weight `P = 8 · diag(1, 64, 9, 1)`, never
/// decaying), solved as `(XᵀX + P) θ = XᵀW + P θ₀`; then `a, r, G ≥ 0`.
///
/// A lane's AR row costs its proportional share `α(C) / C` of the AR step
/// `α(C) = F + a·C` (floored): the unit of every lane floor. A Batched
/// window of `d` verified drafts that carries `share` of the step's spill
/// costs `1 + (r·d + G·share) / (α(C) / C)` units.
///
/// The cost a Verify window adds depends on `C` through the row groups: the
/// once-over-all-rows stages run in groups of at most `LANE_MAX_ROWS` rows,
/// and eight lanes plus one window overflow the first group. The C2 trace
/// (no spill) fits 0.76 ms per window, the C8 trace 14.6 ms. The
/// per-lane-count fit `W = α_C + m_C·e` this replaced could not see that: at
/// C8 every lane drafts, so `e` barely varies and `α_8` and `m_8` traded off
/// (`window_units(8, 3)` swung 1.16–5.42 on the C8 trace, and lanes kept or
/// retired MTP by chance). With the priors a 3-draft window prices at about
/// `1 + (24.6 + 18.5·share) / 11.8` units at C8, so AR wins, while at C2 (no
/// spill) it prices at about 2.4 units for about 3.1 tokens.
#[derive(Clone, Debug)]
pub(crate) struct LaneCostModel {
    xtx: [[f64; 4]; 4],
    xty: [f64; 4],
}

impl LaneCostModel {
    pub(crate) fn new() -> Self {
        Self {
            xtx: [[0.0; 4]; 4],
            xty: [0.0; 4],
        }
    }

    /// One finished pure-decode step of `lanes` decoding lanes, `spill` row
    /// groups beyond the first and `verified` verified drafts in all,
    /// `wall_us` long. An unusable wall is dropped.
    pub(crate) fn observe(&mut self, lanes: usize, spill: usize, verified: usize, wall_us: f32) {
        if !(wall_us.is_finite() && wall_us > 0.0) {
            return;
        }
        let x = [1.0, lanes as f64, verified as f64, spill as f64];
        let y = f64::from(wall_us);
        for i in 0..4 {
            self.xty[i] = self.xty[i] * COST_DECAY + x[i] * y;
            for j in 0..4 {
                self.xtx[i][j] = self.xtx[i][j] * COST_DECAY + x[i] * x[j];
            }
        }
    }

    /// The ridge fit `[F, a, r, G]` (µs), `a, r, G` clamped at zero.
    fn solve(&self) -> [f64; 4] {
        // Augmented `[XᵀX + P | XᵀW + P θ₀]`, eliminated with partial
        // pivoting; the matrix is symmetric positive definite.
        let mut m = [[0.0f64; 5]; 4];
        for i in 0..4 {
            let prior = COST_PRIOR_WEIGHT * COST_PRIOR_SCALE[i];
            m[i][..4].copy_from_slice(&self.xtx[i]);
            m[i][i] += prior;
            m[i][4] = self.xty[i] + prior * COST_PRIOR_US[i];
        }
        for col in 0..4 {
            let pivot = (col..4)
                .max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))
                .unwrap_or(col);
            m.swap(col, pivot);
            for row in col + 1..4 {
                let factor = m[row][col] / m[col][col];
                for k in col..5 {
                    m[row][k] -= factor * m[col][k];
                }
            }
        }
        let mut theta = [0.0f64; 4];
        for i in (0..4).rev() {
            let tail: f64 = (i + 1..4).map(|j| m[i][j] * theta[j]).sum();
            theta[i] = (m[i][4] - tail) / m[i][i];
        }
        if theta.iter().any(|t| !t.is_finite()) {
            theta = COST_PRIOR_US;
        }
        for t in &mut theta[1..] {
            *t = t.max(0.0);
        }
        theta
    }

    /// The clamped `(F, a, r, G)` in µs.
    #[cfg(test)]
    pub(crate) fn fit(&self) -> (f64, f64, f64, f64) {
        let [f, a, r, g] = self.solve();
        (f, a, r, g)
    }

    /// Cost of a window with `drafts` verified drafts carrying `spill_share`
    /// of its step's spill, in a step of `lanes` decoding lanes, in AR-row
    /// units: `1 + (r·d + G·share) / (α(C) / C)`, and exactly `1` without
    /// drafts.
    pub(crate) fn window_units(&self, lanes: usize, drafts: usize, spill_share: f64) -> f32 {
        if drafts == 0 {
            return 1.0;
        }
        let [f, a, r, g] = self.solve();
        let alpha = (f + a * lanes as f64).max(COST_ALPHA_FLOOR_US);
        let share = alpha / lanes.max(1) as f64;
        (1.0 + (r * drafts as f64 + g * spill_share) / share) as f32
    }
}

// ---------------------------------------------------------------------------
// Lane floor policy
// ---------------------------------------------------------------------------

/// `HIPFIRE_MTP_AR_FLOOR` (`0` opts out), read the way the drafter's
/// `configure_request` reads it.
fn floor_env_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_MTP_AR_FLOOR").as_deref() != Ok("0")
}

/// One lane's window router: the singleton's agreement and AR floor, priced
/// in AR-row units (calibrated once at `ar_us = 1.0`; batched windows
/// observed at [`LaneCostModel::window_units`], `ArSynced` windows as
/// interleaved windows of one unit).
#[derive(Clone, Debug)]
pub(crate) struct LaneSpecPolicy {
    agreement: Agreement,
    floor: MtpFloor,
    floor_enabled: bool,
    max_k: usize,
    calibrated: bool,
    /// Windows routed under a lab route mode (the cycle's position).
    lab_windows: usize,
    /// A promoted singleton whose floor already retired it: the head lags
    /// the trunk, so the lane is head-free AR for good (every route mode).
    promoted_retired: bool,
}

impl LaneSpecPolicy {
    /// A request's fresh policy for drafts of at most `k`.
    pub(crate) fn new(k: usize) -> Self {
        Self::with_floor(k, floor_env_enabled())
    }

    pub(crate) fn with_floor(k: usize, floor_enabled: bool) -> Self {
        Self {
            agreement: [MTP_AGREEMENT_PRIOR; MTP_MAX_DEPTH],
            floor: MtpFloor::new(),
            floor_enabled,
            max_k: k.min(MTP_MAX_DEPTH),
            calibrated: false,
            lab_windows: 0,
            promoted_retired: false,
        }
    }

    /// The policy of a promoted singleton: the drafter's per-depth
    /// `agreement` seeds the lane (performance only); `retired` (the
    /// singleton's floor retired it) keeps the lane `ArRetired`, otherwise
    /// the lane floor starts fresh like any admitted lane.
    pub(crate) fn promoted(k: usize, agreement: Agreement, retired: bool) -> Self {
        Self::promoted_with_floor(k, agreement, retired, floor_env_enabled())
    }

    fn promoted_with_floor(
        k: usize,
        agreement: Agreement,
        retired: bool,
        floor_enabled: bool,
    ) -> Self {
        Self {
            agreement,
            promoted_retired: retired,
            ..Self::with_floor(k, floor_enabled)
        }
    }

    /// Most drafts a window of this lane may carry.
    pub(crate) fn max_k(&self) -> usize {
        self.max_k
    }

    /// The floor retired this lane to head-free AR (sticky).
    pub(crate) fn retired(&self) -> bool {
        self.promoted_retired || (self.floor_enabled && self.floor.retired())
    }

    /// Route the lane's next window; `budget` is the most drafts it may
    /// carry (0 = AR). Retired lanes stay `ArRetired`.
    pub(crate) fn route(&mut self, budget: usize, lab: LabRoutes) -> LaneRoute {
        if self.promoted_retired {
            return LaneRoute::ArRetired;
        }
        if lab != LabRoutes::Floor {
            return self.lab_route(budget, lab);
        }
        if !self.calibrated {
            self.floor.observe_calibration(1.0, &self.agreement);
            self.calibrated = true;
        }
        if self.retired() {
            return LaneRoute::ArRetired;
        }
        if budget == 0 {
            return LaneRoute::ArSynced;
        }
        let k = budget.min(self.max_k);
        let provisional = batched_depth(&self.agreement, k);
        let batched = |depth: usize| match depth.min(k) {
            0 => LaneRoute::ArSynced,
            k_max => LaneRoute::Batched { k_max },
        };
        if !self.floor_enabled {
            return batched(provisional);
        }
        match self.floor.route(&self.agreement, k, provisional) {
            FloorRoute::Ar => LaneRoute::ArRetired,
            FloorRoute::ArSample | FloorRoute::Interleaved => LaneRoute::ArSynced,
            FloorRoute::Batched(depth) => batched(depth),
        }
    }

    /// Lab route modes bypass the floor: `Batched` drafts the whole budget;
    /// `Cycle` walks `Batched{K}`, `Batched{1}`, `ArSynced`, `Batched{2}`
    /// three times, then `ArRetired` for good (a zero budget is `ArSynced`).
    fn lab_route(&mut self, budget: usize, lab: LabRoutes) -> LaneRoute {
        let k = budget.min(self.max_k);
        let batched = |depth: usize| match depth.min(k) {
            0 => LaneRoute::ArSynced,
            k_max => LaneRoute::Batched { k_max },
        };
        let route = match lab {
            LabRoutes::Floor => unreachable!("lab_route is not the floor"),
            LabRoutes::Batched => batched(k),
            LabRoutes::Cycle => match self.lab_windows {
                n if n >= 12 => LaneRoute::ArRetired,
                n => match n % 4 {
                    0 => batched(self.max_k),
                    1 => batched(1),
                    2 => LaneRoute::ArSynced,
                    _ => batched(2),
                },
            },
        };
        self.lab_windows += 1;
        route
    }

    /// A finished batched window routed at `cap` drafts, of budget `k`,
    /// priced at `units` AR rows.
    pub(crate) fn observe_batched(&mut self, window: &MtpWindow, units: f32, cap: usize, k: usize) {
        observe_agreement(&mut self.agreement, window, MTP_AGREEMENT_DECAY);
        if self.floor_enabled {
            let stats = WindowStats::of(window, cap);
            self.floor
                .observe_window(NativeWindow::Batched, stats, units, k, &self.agreement);
        }
    }

    /// A finished `ArSynced` window of budget `k`: one AR row, observed as an
    /// interleaved window of one unit. A zero budget observes nothing.
    pub(crate) fn observe_ar_synced(&mut self, window: &MtpWindow, k: usize) {
        if k == 0 {
            return;
        }
        observe_agreement(&mut self.agreement, window, MTP_AGREEMENT_DECAY);
        if self.floor_enabled {
            self.floor
                .observe_native(NativeWindow::Interleaved, window, 1.0, k, &self.agreement);
        }
    }
}

// ---------------------------------------------------------------------------
// Pending window
// ---------------------------------------------------------------------------

/// One routed window between `draft_step` and `commit_step`.
pub(crate) struct PendingWindow {
    pub route: LaneRoute,
    /// The lane's `state.position` when the window was routed (the seed's).
    pub position: usize,
    pub seed: u32,
    /// The kept drafts (`Batched` only).
    pub drafts: Vec<u32>,
    /// Draft steps the head ran (one more than `drafts` when the last step
    /// drafted a dropped token).
    pub steps: usize,
    /// The drafts budget (clamped to the lane's `K`) the floor observes.
    pub budget: usize,
    /// The head ticket taken before the draft steps (`Batched`).
    pub head_ticket: Option<HeadTicket>,
    /// The target ticket taken before the verify forward (`Batched`). Lanes
    /// run on one card: always [`TargetTicket::Local`].
    pub target_ticket: Option<TargetTicket>,
    /// Set by the forward once the window is finished; consumed by commit.
    pub outcome: Option<MtpWindow>,
}

// ---------------------------------------------------------------------------
// Phases
// ---------------------------------------------------------------------------

/// Flatten `with_lane`'s `Result<Result<R, _>, _>`.
pub(crate) fn lane_call<R>(
    result: Result<Result<R, BundleError>, BundleError>,
) -> Result<R, String> {
    result
        .and_then(|inner| inner)
        .map_err(|error| error.to_string())
}

/// One lane's drafting state inside the step-synchronous loop.
struct Drafting {
    lane: usize,
    position: usize,
    k_max: usize,
    input: u32,
    /// Probability the whole draft prefix is accepted.
    prefix: f32,
    drafts: Vec<u32>,
    steps: usize,
    done: bool,
}

/// Route the next window of every listed Decoding lane and run the batched
/// draft of the `Batched` ones. See `Qwen4LaneExecutor::draft_step`.
pub(crate) fn draft_lanes(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    store: &mut Qwen4LaneStore,
    requests: &[(RequestEpoch, usize)],
) -> Result<Vec<LaneDraft>, String> {
    let epochs: Vec<RequestEpoch> = requests.iter().map(|(epoch, _)| *epoch).collect();
    let (mut lanes, mut spec_store) = store.draft_parts(&epochs)?;
    let routes = spec_store.as_ref().map(|s| s.routes);
    let mut out: Vec<LaneDraft> = Vec::with_capacity(requests.len());
    let mut drafting: Vec<Drafting> = Vec::new();

    for (index, ((epoch, budget), lane)) in requests.iter().zip(lanes.iter_mut()).enumerate() {
        let Qwen4LaneState { host, state, spec } = &mut **lane;
        if host.status != crate::lane::LaneStatus::Decoding {
            return Err(format!(
                "draft_step: lane {epoch:?} is {:?}, not Decoding",
                host.status
            ));
        }
        let seed = host
            .pending_seed
            .ok_or_else(|| format!("draft_step: lane {epoch:?} has no pending seed"))?;
        let Some(spec) = spec.as_mut() else {
            out.push(LaneDraft {
                epoch: *epoch,
                route: LaneRoute::Ar,
                drafts: Vec::new(),
            });
            continue;
        };
        if spec.window.is_some() {
            return Err(format!(
                "draft_step: lane {epoch:?} has an unconsumed window (commit or abort the step first)"
            ));
        }
        let route = spec.policy.route(*budget, routes.unwrap_or_default());
        let position = state.position;
        spec.window = Some(PendingWindow {
            route,
            position,
            seed,
            drafts: Vec::new(),
            steps: 0,
            budget: (*budget).min(spec.policy.max_k()),
            head_ticket: None,
            target_ticket: None,
            outcome: None,
        });
        out.push(LaneDraft {
            epoch: *epoch,
            route,
            drafts: Vec::new(),
        });
        let LaneRoute::Batched { k_max } = route else {
            continue;
        };
        // I1: the head sits at the lane's frontier.
        if state.position != spec.head.position() {
            return Err(format!(
                "draft_step: lane {epoch:?} I1: trunk position {} != head position {}",
                state.position,
                spec.head.position()
            ));
        }
        // The head's last draft step and its full-accept append reach
        // `position + k_max`; the verify's rows end at the same bound.
        let end = position
            .checked_add(k_max + 1)
            .ok_or_else(|| "draft_step: position overflow".to_string())?;
        spec.head
            .ensure_mapped_capacity(gpu, end)
            .map_err(|e| format!("draft_step: map head arenas of {epoch:?}: {e}"))?;
        state
            .ensure_mapped_capacity(gpu, end)
            .map_err(|e| format!("draft_step: map trunk arenas of {epoch:?}: {e}"))?;
        let ticket = lane_call(bundle.with_lane(
            state,
            Some((&mut spec.head, &mut spec.draft)),
            |b| b.mtp_snapshot(gpu),
        ))
        .map_err(|e| format!("draft_step: head snapshot of {epoch:?}: {e}"))?;
        if let Some(window) = spec.window.as_mut() {
            window.head_ticket = Some(HeadTicket::Local(ticket));
        }
        drafting.push(Drafting {
            lane: index,
            position,
            k_max,
            input: seed,
            prefix: 1.0,
            drafts: Vec::with_capacity(k_max),
            steps: 0,
            done: false,
        });
    }

    let slot_bytes = Gpu::TOPK8_OUT_BYTES;
    let max_steps = drafting.iter().map(|d| d.k_max).max().unwrap_or(0);
    for j in 0..max_steps {
        let active: Vec<usize> = (0..drafting.len()).filter(|&d| !drafting[d].done).collect();
        if active.is_empty() {
            break;
        }
        let store = spec_store
            .as_mut()
            .ok_or_else(|| "draft_step: batched window in a store without MTP lanes".to_string())?;
        // Every still-drafting lane enqueues its draft step, in slot order.
        for (slot, &d) in active.iter().enumerate() {
            let st = &drafting[d];
            let Qwen4LaneState { state, spec, .. } = &mut *lanes[st.lane];
            let spec = spec
                .as_mut()
                .ok_or_else(|| "draft_step: drafting lane lost its MTP state".to_string())?;
            let LaneSpec {
                head,
                draft,
                pending_hidden,
                ..
            } = spec;
            let top1 = store.draft_top1.sub_offset(slot * slot_bytes, slot_bytes);
            let hidden = (j == 0).then_some(&*pending_hidden);
            let (input, position) = (st.input, st.position + j);
            lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
                b.mtp_forward_token_deferred(gpu, input, hidden, position, j == 0, &top1)
            }))
            .map_err(|e| format!("draft_step: draft step {j}: {e}"))?;
        }
        // One D2H for every active lane.
        let bytes = active.len() * slot_bytes;
        gpu.hip
            .memcpy_dtoh(&mut store.draft_top1_host[..bytes], &store.draft_top1.buf)
            .map_err(|e| format!("draft_step: draft download: {e}"))?;
        gpu.take_device_wait_error()
            .map_err(|e| format!("draft_step: draft download: {e}"))?;
        for (slot, &d) in active.iter().enumerate() {
            let top = &store.draft_top1_host[slot * slot_bytes..(slot + 1) * slot_bytes];
            let st = &mut drafting[d];
            let Qwen4LaneState { state, spec, .. } = &mut *lanes[st.lane];
            let spec = spec
                .as_mut()
                .ok_or_else(|| "draft_step: drafting lane lost its MTP state".to_string())?;
            let LaneSpec { head, draft, .. } = spec;
            let (token, margin) = lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
                let token = b.mtp_draft_finish(top)?;
                Ok::<_, BundleError>((token, b.mtp_draft_margin()))
            }))
            .map_err(|e| format!("draft_step: draft finish {j}: {e}"))?;
            // The singleton's early stop, statement for statement.
            st.input = token;
            st.steps += 1;
            st.prefix *= draft_accept_estimate(margin);
            if j > 0 && st.prefix < MTP_ROW_WORTH {
                st.done = true;
                continue;
            }
            st.drafts.push(token);
            if st.prefix * DRAFT_ACCEPT_MAX < MTP_ROW_WORTH || j + 1 >= st.k_max {
                st.done = true;
            }
        }
    }

    for st in drafting {
        let lane = &mut *lanes[st.lane];
        let window = lane
            .spec
            .as_mut()
            .and_then(|spec| spec.window.as_mut())
            .ok_or_else(|| "draft_step: window vanished".to_string())?;
        window.steps = st.steps;
        window.drafts = st.drafts.clone();
        out[st.lane].drafts = st.drafts;
    }
    Ok(out)
}

/// Finish a Verify lane after the trunk forward: greedy accept over the
/// lane's picks, then the singleton's window tail on the lane's own states.
/// `hidden` is the store's wide-hidden capture, the lane's rows starting at
/// `row_begin`; `picks` are the lane's `1 + d` greedy picks. Returns the
/// window (the lane's `window.outcome` is the caller's to set).
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_lane_window(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    config: LaneSpecConfig,
    hidden: &GpuTensor,
    row_begin: usize,
    picks: &[u32],
    skip_rollback: bool,
) -> Result<MtpWindow, String> {
    let wide = bundle
        .config
        .hc_count
        .checked_mul(bundle.config.hidden_size)
        .ok_or_else(|| "finish_lane_window: hidden width overflow".to_string())?;
    let Qwen4LaneState { state, spec, .. } = lane;
    let spec = spec
        .as_mut()
        .ok_or_else(|| "finish_lane_window: lane has no MTP state".to_string())?;
    let LaneSpec {
        head,
        draft,
        pending_hidden,
        window,
        ..
    } = spec;
    let window = window
        .as_ref()
        .ok_or_else(|| "finish_lane_window: lane has no pending window".to_string())?;
    if !matches!(window.route, LaneRoute::Batched { .. }) {
        return Err(format!(
            "finish_lane_window: window routed {:?} has no verify",
            window.route
        ));
    }
    let rows = window.drafts.len() + 1;
    if picks.len() != rows {
        return Err(format!(
            "finish_lane_window: {} picks for a {rows}-row verify",
            picks.len()
        ));
    }
    // The ring must cover the verify: the replay branch of the singleton is
    // unreachable on lanes.
    if state.row_capture_rows() < rows {
        return Err(format!(
            "finish_lane_window: the lane ring holds {} rows, the verify has {rows}",
            state.row_capture_rows()
        ));
    }
    // I1 at finish: the verify published `rows` rows, the head ran `steps`
    // draft steps from the window's frontier.
    if state.position != window.position + rows {
        return Err(format!(
            "finish_lane_window: I1: trunk position {} != {} after the verify",
            state.position,
            window.position + rows
        ));
    }
    if head.position() != window.position + window.steps {
        return Err(format!(
            "finish_lane_window: I1: head position {} != {} after {} draft steps",
            head.position(),
            window.position + window.steps,
            window.steps
        ));
    }
    let target_snapshot = window
        .target_ticket
        .ok_or_else(|| "finish_lane_window: no target ticket".to_string())?;
    let mtp_ticket = window
        .head_ticket
        .ok_or_else(|| "finish_lane_window: no head ticket".to_string())?;
    let eos = config.end_of_turn.unwrap_or(bundle.config.eos_token_id);
    let drafts = window.drafts.clone();
    let acceptance = accept_native_greedy(&drafts, picks, Some(eos))?;
    let mut block = Vec::with_capacity(rows);
    block.push(window.seed);
    block.extend_from_slice(&drafts);
    let (position, steps) = (window.position, window.steps);
    let verify_hidden = hidden.sub_offset(row_begin * wide, rows * wide);
    // Taken before `with_lane`: it names the forward scratch.
    let ple_normed = bundle
        .lane_ple_normed_rows(row_begin, rows)
        .map_err(|e| format!("finish_lane_window: PLE rows: {e}"))?;
    let result = MtpWindow {
        committed: acceptance.committed.clone(),
        accepted: acceptance.accepted,
        drafts_generated: drafts.len(),
    };
    // Lab negative control: a rejected verify finishes as a full accept, so
    // the lane keeps every verified row (the rejected tail included) while
    // the stream still follows the true acceptance.
    let finish_acceptance = if skip_rollback && acceptance.accepted < drafts.len() {
        GreedyAccept {
            committed: drafts
                .iter()
                .copied()
                .chain(std::iter::once(picks[drafts.len()]))
                .collect(),
            accepted: drafts.len(),
            hit_eos: false,
        }
    } else {
        acceptance
    };
    lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
        finish_native_window(
            b,
            &mut Qwen4Ops::Single(&mut *gpu),
            NativeFinish {
                position,
                block: &block,
                steps,
                acceptance: finish_acceptance,
                target_snapshot,
                mtp_ticket,
                pending_hidden,
                verify_hidden: &verify_hidden,
                ple_normed: Some(&ple_normed),
                retain: false,
            },
            None,
        )
        .map_err(BundleError::Forward)
    }))
    .map_err(|e| format!("finish_lane_window: {e}"))?;
    Ok(result)
}

/// The head work of an `ArSynced` lane after its AR row: append the seed
/// with the `DraftPairing` conditioning, then keep the AR row's wide hidden
/// as the pending hidden. `dec_row` is the lane's row in `hidden`.
pub(crate) fn ar_synced_head(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    hidden: &GpuTensor,
    dec_row: usize,
) -> Result<(), String> {
    let wide = bundle
        .config
        .hc_count
        .checked_mul(bundle.config.hidden_size)
        .ok_or_else(|| "ar_synced_head: hidden width overflow".to_string())?;
    let Qwen4LaneState { state, spec, .. } = lane;
    let spec = spec
        .as_mut()
        .ok_or_else(|| "ar_synced_head: lane has no MTP state".to_string())?;
    let LaneSpec {
        head,
        draft,
        pending_hidden,
        window,
        ..
    } = spec;
    let window = window
        .as_ref()
        .ok_or_else(|| "ar_synced_head: lane has no pending window".to_string())?;
    let (position, seed) = (window.position, window.seed);
    if head.position() != position {
        return Err(format!(
            "ar_synced_head: I1: head position {} != lane position {position}",
            head.position()
        ));
    }
    if state.position != position + 1 {
        return Err(format!(
            "ar_synced_head: lane position {} != {} after the AR row",
            state.position,
            position + 1
        ));
    }
    let row = hidden.sub_offset(dec_row * wide, wide);
    let conditioning = match draft_pairing() {
        DraftPairing::HeadState => None,
        DraftPairing::AlignedHead | DraftPairing::AlignedTarget => Some(&row),
    };
    lane_call(bundle.with_lane(state, Some((&mut *head, draft)), |b| {
        b.mtp_append_token(gpu, seed, conditioning, position)
    }))
    .map_err(|e| format!("ar_synced_head: head append: {e}"))?;
    if head.position() != position + 1 {
        return Err(format!(
            "ar_synced_head: head ended at {}, expected {}",
            head.position(),
            position + 1
        ));
    }
    gpu.copy_d2d(&row, pending_hidden, pending_hidden.byte_size())
        .map_err(|e| format!("ar_synced_head: pending hidden copy: {e}"))
}

/// One prompt tile of a spec lane: the singleton `mtp_prefill` chunk body on
/// the lane's own states. Returns the trunk pick when the tile asks for one
/// (always on the prompt-completing tile).
pub(crate) fn prefill_tile(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    tokens: &[u32],
    append_scratch: Option<&mut MtpAppendScratch>,
) -> Result<Option<u32>, String> {
    let Qwen4LaneState { host, state, spec } = lane;
    let spec = spec
        .as_mut()
        .ok_or_else(|| "prefill_tile: lane has no MTP state".to_string())?;
    let LaneSpec {
        head,
        draft,
        pending_hidden,
        ..
    } = spec;
    let start = host.prefilled;
    if state.position != start || head.position() != start {
        return Err(format!(
            "prefill_tile: I1: trunk position {}, head position {}, prefilled {start}",
            state.position,
            head.position()
        ));
    }
    let end = start + tokens.len();
    let needs_pick =
        mtp_prefill_chunk_needs_pick(end, host.prompt.len(), mtp_skip_intermediate_pick_enabled());
    // `mtp_prefill_chunk` applies batched fill / the DET prompt fill to the scratch.
    let reuse = mtp_reuse_prefill_ids_enabled();
    lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
        b.poll_npu_trunk(gpu)?;
        let filled = mtp_prefill_chunk(
            b,
            &mut Qwen4Ops::Single(&mut *gpu),
            tokens,
            start,
            needs_pick,
            pending_hidden,
            append_scratch,
            reuse,
            &|| false,
        )
        .map_err(BundleError::Forward)?;
        match filled {
            std::ops::ControlFlow::Continue(pick) => Ok(pick),
            std::ops::ControlFlow::Break(()) => unreachable!("a never-abort fill cannot break"),
        }
    }))
    .map_err(|e| format!("prefill_tile: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(committed: usize, accepted: usize, drafts: usize) -> MtpWindow {
        MtpWindow {
            committed: vec![7; committed],
            accepted,
            drafts_generated: drafts,
        }
    }

    /// A deterministic spread of pure-decode step shapes `(C, e, s)`:
    /// `C` in `1..=8`, `e` in `0..=3C`, `s` in `0..=3`.
    fn step_shape(k: usize) -> (usize, usize, usize) {
        let lanes = 1 + (k * 5) % 8;
        let verified = (k * 7 + k / 3) % (3 * lanes + 1);
        let spill = (k * 3 + k / 5) % 4;
        (lanes, verified, spill)
    }

    /// The noise-free wall `F + a·C + r·e + G·s` of a `[F, a, r, G]` truth.
    fn truth_wall(truth: [f64; 4], (lanes, verified, spill): (usize, usize, usize)) -> f64 {
        truth[0] + truth[1] * lanes as f64 + truth[2] * verified as f64 + truth[3] * spill as f64
    }

    /// Feed `steps` steps of `truth`, starting at shape index `from`.
    fn feed(model: &mut LaneCostModel, truth: [f64; 4], from: usize, steps: usize) {
        for k in from..from + steps {
            let shape = step_shape(k);
            model.observe(shape.0, shape.2, shape.1, truth_wall(truth, shape) as f32);
        }
    }

    fn fit_array(model: &LaneCostModel) -> [f64; 4] {
        let (f, a, r, g) = model.fit();
        [f, a, r, g]
    }

    /// Worst relative error of the fit's predicted wall against `truth`
    /// over the held-out step shapes `from..from + steps`.
    fn worst_wall_error(model: &LaneCostModel, truth: [f64; 4], from: usize, steps: usize) -> f64 {
        let fit = fit_array(model);
        (from..from + steps)
            .map(|k| {
                let shape = step_shape(k);
                (truth_wall(fit, shape) / truth_wall(truth, shape) - 1.0).abs()
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn a_fresh_model_reproduces_its_priors() {
        let model = LaneCostModel::new();
        let fit = fit_array(&model);
        for (got, want) in fit.iter().zip(COST_PRIOR_US) {
            assert!(
                (got - want).abs() < want * 1e-9,
                "{fit:?} vs {COST_PRIOR_US:?}"
            );
        }
        // A window of d drafts costs 1 + (r d + G share) / (alpha(C) / C).
        let units = model.window_units(8, 3, 0.375);
        let want = 1.0 + (3.0 * 8_200.0 + 18_500.0 * 0.375) / ((16_900.0 + 8.0 * 9_700.0) / 8.0);
        assert!((f64::from(units) - want).abs() < 1e-4, "{units} vs {want}");
    }

    #[test]
    fn the_priors_keep_mtp_at_c2_and_retire_it_at_c8() {
        // A 3-draft window earns about 3.1 tokens. At C2 (no spill) it
        // costs less than that in AR rows; at C8 (one window spills a row
        // group over eight lanes) it costs more.
        let model = LaneCostModel::new();
        let c2 = model.window_units(2, 3, 0.0);
        let c8 = model.window_units(8, 3, 0.375);
        assert!(c2 < 3.1, "C2 window {c2}");
        assert!(c8 > 3.1, "C8 window {c8}");
    }

    #[test]
    fn regression_recovers_the_pooled_model_from_synthetic_steps() {
        // The ridge prior (a constant eight observations' weight) keeps the
        // fit from reaching a truth far from it exactly: after 400 steps
        // (about 100 effective under the decay) it lands within 15% of each
        // parameter, well inside the prior's own distance, and predicts
        // held-out walls within 10%.
        let truth = [24_000.0, 12_000.0, 6_000.0, 25_000.0];
        let mut model = LaneCostModel::new();
        feed(&mut model, truth, 0, 400);
        let fit = fit_array(&model);
        for i in 0..4 {
            let err = (fit[i] / truth[i] - 1.0).abs();
            let prior_err = (COST_PRIOR_US[i] / truth[i] - 1.0).abs();
            assert!(err < 0.15, "param {i}: {fit:?} vs {truth:?}");
            assert!(
                err < 0.6 * prior_err,
                "param {i} barely left the prior: {fit:?}"
            );
        }
        let wall_err = worst_wall_error(&model, truth, 400, 200);
        assert!(wall_err < 0.10, "held-out wall error {wall_err}");
        // Units follow the fit: share a = alpha(C) / C.
        let units = f64::from(model.window_units(8, 3, 0.5));
        let want = 1.0 + (3.0 * truth[2] + 0.5 * truth[3]) / ((truth[0] + 8.0 * truth[1]) / 8.0);
        assert!((units / want - 1.0).abs() < 0.15, "{units} vs {want}");
    }

    #[test]
    fn the_fit_tracks_a_drifting_cost_through_the_decay() {
        let before = [24_000.0, 12_000.0, 6_000.0, 25_000.0];
        let after = [30_000.0, 7_000.0, 4_000.0, 12_000.0];
        let mut model = LaneCostModel::new();
        feed(&mut model, before, 0, 400);
        feed(&mut model, after, 400, 400);
        let to_after = worst_wall_error(&model, after, 800, 200);
        let to_before = worst_wall_error(&model, before, 800, 200);
        assert!(
            to_after < 0.20,
            "wall error against the new truth {to_after}"
        );
        assert!(
            to_after < 0.5 * to_before,
            "fit still follows the old truth: {to_after} vs {to_before}"
        );
    }

    #[test]
    fn the_fit_respects_the_clamps() {
        // A truth with a negative per-draft cost fits r = 0 (the unclamped
        // solution is negative), and the other parameters stay usable.
        let mut model = LaneCostModel::new();
        feed(&mut model, [24_000.0, 12_000.0, -2_500.0, 25_000.0], 0, 400);
        let (f, a, r, g) = model.fit();
        assert_eq!(r, 0.0);
        assert!(f.is_finite() && a >= 0.0 && g >= 0.0, "{:?}", (f, a, r, g));
        let units = model.window_units(8, 3, 0.5);
        assert!(units.is_finite() && units > 1.0, "{units}");
    }

    #[test]
    fn unusable_walls_are_ignored() {
        let mut model = LaneCostModel::new();
        feed(&mut model, [24_000.0, 12_000.0, 6_000.0, 25_000.0], 0, 50);
        let before = model.fit();
        let units = model.window_units(8, 3, 0.5);
        for wall in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -4.0, 0.0] {
            model.observe(8, 1, 24, wall);
        }
        assert_eq!(model.fit(), before);
        assert_eq!(model.window_units(8, 3, 0.5), units);
    }

    #[test]
    fn collinear_steps_leave_the_draft_cost_near_its_prior() {
        // Every lane drafts every step (C = 8, one spill group) and e barely
        // varies: the per-draft cost cannot be told apart from the
        // intercept. The prior holds the fit; `window_units(8, 3, ..)`
        // stays a sane price.
        let mut model = LaneCostModel::new();
        for k in 0..400 {
            let verified = 22 + k % 3;
            let wall = 1.1 * truth_wall(COST_PRIOR_US, (8, verified, 1));
            model.observe(8, 1, verified, wall as f32);
        }
        let (_, _, r, _) = model.fit();
        assert!((r / COST_PRIOR_US[2] - 1.0).abs() < 0.25, "r {r}");
        let units = model.window_units(8, 3, 0.375);
        assert!(units.is_finite() && units > 1.0, "{units}");
    }

    #[test]
    fn a_window_without_drafts_costs_one_unit() {
        let mut model = LaneCostModel::new();
        for lanes in 0..=16 {
            assert_eq!(model.window_units(lanes, 0, 0.5), 1.0, "fresh, C {lanes}");
        }
        feed(&mut model, [24_000.0, 12_000.0, 6_000.0, 25_000.0], 0, 200);
        for lanes in 0..=16 {
            assert_eq!(model.window_units(lanes, 0, 0.5), 1.0, "fed, C {lanes}");
        }
    }

    #[test]
    fn fresh_and_promoted_lanes_keep_the_short_agreement_decay() {
        assert_eq!(MTP_AGREEMENT_DECAY, 0.875);
        for floor_enabled in [false, true] {
            let inherited = [(4.0, 5.0); MTP_MAX_DEPTH];
            for mut policy in [
                LaneSpecPolicy::with_floor(3, floor_enabled),
                LaneSpecPolicy::promoted_with_floor(3, inherited, false, floor_enabled),
            ] {
                let mut expected = policy.agreement[0];
                policy.observe_batched(&window(1, 0, 1), 1.0, 1, 3);
                expected = (expected.0 * 0.875, expected.1 * 0.875 + 1.0);
                assert_eq!(policy.agreement[0], expected);
                policy.observe_ar_synced(&window(2, 1, 1), 3);
                expected = (expected.0 * 0.875 + 1.0, expected.1 * 0.875 + 1.0);
                assert_eq!(policy.agreement[0], expected);
                let unchanged = policy.agreement;
                policy.observe_ar_synced(&window(2, 1, 1), 0);
                assert_eq!(policy.agreement, unchanged);
            }
        }
    }

    #[test]
    fn lane_floor_retains_the_short_cost_decay() {
        // These three windows pay at 0.875, but not at 0.97, after the
        // lane floor's three-window probe budget.
        let mut policy = LaneSpecPolicy::with_floor(1, true);
        policy.route(1, LabRoutes::Floor);
        for units in [3.0, 0.8, 2.05] {
            policy.observe_batched(&window(2, 1, 1), units, 1, 1);
        }
        assert!(!policy.retired());
        assert_eq!(
            policy.route(1, LabRoutes::Floor),
            LaneRoute::Batched { k_max: 1 }
        );
    }

    #[test]
    fn a_zero_budget_routes_ar_synced() {
        let mut policy = LaneSpecPolicy::with_floor(3, true);
        assert_eq!(policy.route(0, LabRoutes::Floor), LaneRoute::ArSynced);
        let mut off = LaneSpecPolicy::with_floor(3, false);
        assert_eq!(off.route(0, LabRoutes::Floor), LaneRoute::ArSynced);
    }

    #[test]
    fn floor_off_routes_the_agreement_depth() {
        let mut policy = LaneSpecPolicy::with_floor(3, false);
        let depth = batched_depth(&policy.agreement, 3);
        let want = match depth {
            0 => LaneRoute::ArSynced,
            d => LaneRoute::Batched { k_max: d },
        };
        assert_eq!(policy.route(3, LabRoutes::Floor), want);
        // A smaller budget caps the depth.
        let capped = batched_depth(&policy.agreement, 1);
        let want = match capped {
            0 => LaneRoute::ArSynced,
            d => LaneRoute::Batched { k_max: d },
        };
        assert_eq!(policy.route(1, LabRoutes::Floor), want);
        // Floor off never retires, whatever it observes.
        for _ in 0..8 {
            policy.observe_batched(&window(1, 0, 1), 50.0, 1, 3);
        }
        assert!(!policy.retired());
        assert_ne!(policy.route(3, LabRoutes::Floor), LaneRoute::ArRetired);
    }

    #[test]
    fn a_fresh_floor_probes_batched_then_retires_stickily() {
        let mut policy = LaneSpecPolicy::with_floor(3, true);
        let first = policy.route(3, LabRoutes::Floor);
        let depth = batched_depth(&policy.agreement, 3);
        assert_eq!(
            first,
            match depth {
                0 => LaneRoute::ArSynced,
                d => LaneRoute::Batched { k_max: d },
            }
        );
        // Three probe windows far costlier than their emitted tokens.
        for _ in 0..3 {
            policy.observe_batched(&window(1, 0, 1), 6.0, 1, 3);
        }
        assert!(policy.retired());
        assert_eq!(policy.route(3, LabRoutes::Floor), LaneRoute::ArRetired);
        // Sticky: a later cheap window changes nothing, nor does budget 0.
        policy.observe_batched(&window(4, 3, 3), 0.5, 3, 3);
        assert_eq!(policy.route(3, LabRoutes::Floor), LaneRoute::ArRetired);
        assert_eq!(policy.route(0, LabRoutes::Floor), LaneRoute::ArRetired);
    }

    #[test]
    fn a_low_agreement_probe_routes_ar_synced() {
        let mut policy = LaneSpecPolicy::with_floor(3, true);
        policy.agreement = [(0.0, 2.0); MTP_MAX_DEPTH];
        assert_eq!(batched_depth(&policy.agreement, 3), 0);
        // The floor's probe budget routes the provisional depth: none is
        // interleaved, which a lane runs as one synced AR row.
        assert_eq!(policy.route(3, LabRoutes::Floor), LaneRoute::ArSynced);
        policy.observe_ar_synced(&window(1, 0, 0), 3);
        assert!(!policy.retired());
        // A zero budget observes nothing.
        policy.observe_ar_synced(&window(1, 0, 0), 0);
    }

    #[test]
    fn a_promoted_retired_policy_stays_ar_retired_in_every_mode() {
        for floor_enabled in [true, false] {
            let mut policy = LaneSpecPolicy::promoted_with_floor(
                3,
                [(0.9, 1.0); MTP_MAX_DEPTH],
                true,
                floor_enabled,
            );
            assert!(policy.retired());
            for lab in [LabRoutes::Floor, LabRoutes::Batched, LabRoutes::Cycle] {
                for budget in [0, 1, 3] {
                    assert_eq!(policy.route(budget, lab), LaneRoute::ArRetired);
                }
            }
            policy.observe_batched(&window(4, 3, 3), 0.5, 3, 3);
            policy.observe_ar_synced(&window(1, 0, 0), 3);
            assert!(policy.retired());
            assert_eq!(policy.route(3, LabRoutes::Floor), LaneRoute::ArRetired);
        }
    }

    #[test]
    fn a_promoted_live_policy_routes_like_a_fresh_one_with_the_given_agreement() {
        let low = [(0.0, 2.0); MTP_MAX_DEPTH];
        let mut promoted = LaneSpecPolicy::promoted_with_floor(3, low, false, true);
        assert!(!promoted.retired());
        assert_eq!(promoted.agreement, low);
        assert_eq!(batched_depth(&promoted.agreement, 3), 0);
        let mut fresh = LaneSpecPolicy::with_floor(3, true);
        fresh.agreement = low;
        assert_eq!(
            promoted.route(3, LabRoutes::Floor),
            fresh.route(3, LabRoutes::Floor)
        );
        assert_eq!(promoted.route(0, LabRoutes::Floor), LaneRoute::ArSynced);
        // The floor starts fresh: it still retires on costly probe windows.
        let mut probing = LaneSpecPolicy::promoted_with_floor(
            3,
            [MTP_AGREEMENT_PRIOR; MTP_MAX_DEPTH],
            false,
            true,
        );
        assert_ne!(probing.route(3, LabRoutes::Floor), LaneRoute::ArRetired);
        for _ in 0..3 {
            probing.observe_batched(&window(1, 0, 1), 6.0, 1, 3);
        }
        assert!(probing.retired());
        assert_eq!(probing.route(3, LabRoutes::Floor), LaneRoute::ArRetired);
        // Lab modes still bypass the floor on a live promoted lane.
        let mut lab = LaneSpecPolicy::promoted_with_floor(3, low, false, true);
        assert_eq!(
            lab.route(2, LabRoutes::Batched),
            LaneRoute::Batched { k_max: 2 }
        );
    }

    #[test]
    fn lab_route_modes_bypass_the_floor() {
        let mut policy = LaneSpecPolicy::with_floor(3, true);
        assert_eq!(
            policy.route(2, LabRoutes::Batched),
            LaneRoute::Batched { k_max: 2 }
        );
        assert_eq!(
            policy.route(9, LabRoutes::Batched),
            LaneRoute::Batched { k_max: 3 }
        );
        assert_eq!(policy.route(0, LabRoutes::Batched), LaneRoute::ArSynced);

        let mut cycle = LaneSpecPolicy::with_floor(3, true);
        let mut routes = Vec::new();
        for _ in 0..14 {
            routes.push(cycle.route(3, LabRoutes::Cycle));
        }
        let pattern = [
            LaneRoute::Batched { k_max: 3 },
            LaneRoute::Batched { k_max: 1 },
            LaneRoute::ArSynced,
            LaneRoute::Batched { k_max: 2 },
        ];
        for (n, route) in routes.iter().take(12).enumerate() {
            assert_eq!(*route, pattern[n % 4], "window {n}");
        }
        assert_eq!(routes[12], LaneRoute::ArRetired);
        assert_eq!(routes[13], LaneRoute::ArRetired);
        // The budget caps the cycle's depths.
        let mut capped = LaneSpecPolicy::with_floor(3, true);
        assert_eq!(
            capped.route(1, LabRoutes::Cycle),
            LaneRoute::Batched { k_max: 1 }
        );
    }
}
