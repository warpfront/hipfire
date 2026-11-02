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
//! [`Qwen4Floor`] and agreement, fed in AR-row units from the store-level
//! [`LaneCostModel`] instead of wall clocks (see its docs).

use crate::bundle::{BundleError, Qwen4Bundle};
use crate::lane::{
    download_lane_rows, lane_sample_step, LabRoutes, LaneSpec, LaneSpecConfig, Qwen4LaneState,
};
use crate::lane_exec::LaneRoute;
use crate::mesh_lane_protocol::ProposalSource;
use crate::mesh_rank_copy::{verify_capable, PreparedRankLane, ProbeOutcome};
use crate::mtp_gpu::MtpAppendScratch;
use crate::mtp_spec::{
    accept_native_greedy, floor_with_call_log, mtp_prefill_chunk, mtp_prefill_chunk_needs_pick,
    mtp_reuse_prefill_ids_enabled, mtp_skip_intermediate_pick_enabled, prepare_native_window,
    publish_native_window, HeadTicket, NativeFinish, PreparedNativeWindow, Qwen4Floor, Qwen4Ops,
    TargetTicket, MTP_WINDOW_COST,
};

use hipfire_dispatch::pipeline::DraftHeadRequestState;
use hipfire_runtime::spec::{GreedyAccept, MtpWindow};
use hipfire_runtime::spec_economics::{
    observe_agreement, Action, NativeWindow, WindowStats, AGREEMENT_PRIOR, DEFAULT_DECAY,
};
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
pub(crate) fn floor_env_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_MTP_AR_FLOOR").as_deref() != Ok("0")
}

const LANE_MAX_DRAFTS: usize = crate::gpu_forward_lanes::LANE_MAX_ROWS - 1;

/// One lane's window router: the singleton's agreement and AR floor, priced
/// in AR-row units (calibrated once at `ar_us = 1.0`; batched windows
/// observed at [`LaneCostModel::window_units`], `ArSynced` windows as
/// interleaved windows of one unit).
/// Promotion retains the singleton's full agreement history without widening
/// the lane's physical draft capacity.
#[derive(Clone, Debug)]
pub(crate) struct LaneSpecPolicy {
    agreement: Vec<(f32, f32)>,
    floor: Qwen4Floor,
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
    pub(crate) fn with_floor(k: usize, floor_enabled: bool) -> Self {
        let max_k = k.min(LANE_MAX_DRAFTS);
        Self::with_agreement(max_k, vec![AGREEMENT_PRIOR; max_k], false, floor_enabled)
    }

    fn with_agreement(
        max_k: usize,
        agreement: Vec<(f32, f32)>,
        promoted_retired: bool,
        floor_enabled: bool,
    ) -> Self {
        let mut floor = Qwen4Floor::new(MTP_WINDOW_COST);
        floor.set_capacity(max_k);
        let floor = floor_with_call_log(floor, "lane", floor_enabled);
        Self {
            agreement,
            floor,
            floor_enabled,
            max_k,
            calibrated: false,
            lab_windows: 0,
            promoted_retired,
        }
    }

    /// The policy of a promoted singleton: the drafter's per-depth
    /// `agreement` seeds the lane (performance only); `retired` (the
    /// singleton's floor retired it) keeps the lane `ArRetired`, otherwise
    /// the lane floor starts fresh like any admitted lane.
    pub(crate) fn promoted_with_floor(
        k: usize,
        agreement: &[(f32, f32)],
        retired: bool,
        floor_enabled: bool,
    ) -> Self {
        Self::with_agreement(
            k.min(LANE_MAX_DRAFTS),
            agreement.to_vec(),
            retired,
            floor_enabled,
        )
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
        let provisional = self.floor.provisional_depth(&self.agreement, k);
        let batched = |depth: usize| match depth.min(k) {
            0 => LaneRoute::ArSynced,
            k_max => LaneRoute::Batched { k_max },
        };
        if !self.floor_enabled {
            return batched(provisional);
        }
        match self.floor.route(&self.agreement, k, provisional) {
            Action::ArHeadFree => LaneRoute::ArRetired,
            Action::ArSynced | Action::Interleaved => LaneRoute::ArSynced,
            Action::Native(depth) => batched(depth),
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
        observe_agreement(&mut self.agreement, window, DEFAULT_DECAY);
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
        observe_agreement(&mut self.agreement, window, DEFAULT_DECAY);
        if self.floor_enabled {
            self.floor
                .observe_native(NativeWindow::Interleaved, window, 1.0, k, &self.agreement);
        }
    }
}

/// The copy lane's view of the lane's own floor (`crate::mesh_lane_copy`):
/// the same measured prices and AR comparator the singleton's copy chooser
/// reads, in AR-row units. The native chooser and copy chooser stay
/// distinct; copy windows are priced as external windows
/// ([`Qwen4Floor::observe_external_copy`]) and never un-retire the lane.
impl LaneSpecPolicy {
    /// The floor's AR calibration (or a retirement probe) still has to run,
    /// so the copy source must not take the window. A retired lane is past
    /// calibration by definition (a promoted retired lane's floor is fresh
    /// but its verdict is not). Lab route modes bypass the floor, as the
    /// singleton's developer-forced route does.
    pub(crate) fn copy_blocked(&self, lab: LabRoutes) -> bool {
        self.floor_enabled
            && lab == LabRoutes::Floor
            && !self.retired()
            && self.floor.needs_ar_sample()
    }

    /// Native window cost at `depth` drafts in AR-row units: the floor's
    /// measurement/fit once probing finished, else the static table.
    fn window_cost(&self, depth: usize) -> f32 {
        if self.floor_enabled {
            if let Some(cost) = self.floor.batched_cost(depth) {
                return cost;
            }
        }
        MTP_WINDOW_COST.get(depth).copied().unwrap_or(f32::INFINITY)
    }

    /// Best expected emitted tokens per unit of window cost the native
    /// route offers within `k` drafts at the current agreement (the
    /// interleaved route is 1.0).
    pub(crate) fn native_rate(&self, k: usize) -> f32 {
        let depths = k.min(self.max_k).min(self.agreement.len());
        if self.floor_enabled {
            if let Some(rate) = self.floor.candidate_rate(&self.agreement, depths) {
                return rate;
            }
        }
        let mut best = 1.0f32;
        let mut prefix = 1.0f32;
        let mut expected = 1.0f32;
        for (i, &(accepted, total)) in self.agreement.iter().take(depths).enumerate() {
            prefix *= accepted / total.max(f32::MIN_POSITIVE);
            expected += prefix;
            best = best.max(expected / self.window_cost(i + 1));
        }
        best
    }

    /// Window cost of a takeover of `d` candidates in AR-row units: this
    /// request's measured takeover costs ([`Qwen4Floor::takeover_cost`]),
    /// else the native window cost.
    pub(crate) fn takeover_cost(&self, d: usize) -> f32 {
        if self.floor_enabled {
            if let Some(cost) = self.floor.takeover_cost(d) {
                return cost;
            }
        }
        self.window_cost(d)
    }

    /// One finished copy window (`drafts` offered, `emitted` committed,
    /// `units` AR rows, NaN for an unpriced bind window). A retired floor
    /// keeps pricing it and stays retired. Returns `true` when the copy
    /// source must stop for the rest of the request (takeovers cost no less
    /// per token than AR); never with the floor opted out.
    pub(crate) fn observe_external_copy(
        &mut self,
        emitted: usize,
        units: f32,
        drafts: usize,
    ) -> bool {
        self.floor_enabled
            && self
                .floor
                .observe_external_copy(emitted, units, drafts, &self.agreement)
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
    /// The kept proposals: the native drafts, the copy candidates or the
    /// forced tokens `f1..f_{n-1}` (the sealed rows after the seed); empty
    /// for an AR row.
    pub drafts: Vec<u32>,
    /// Draft steps the head ran and that are still in effect (one more than
    /// `drafts` when the last step drafted a dropped token; 0 again after an
    /// abandoned confirmation probe). `head_at_start = (steps == 0)`.
    pub steps: usize,
    /// The head ticket: before the draft steps of a native window, and
    /// (after the plan's catch-up) before a copy confirmation probe or a copy
    /// or forced window's finish.
    pub head_ticket: Option<HeadTicket>,
    /// The target ticket taken before the verify forward (`rows > 1`). Lanes
    /// run on one card: always [`TargetTicket::Local`].
    pub target_ticket: Option<TargetTicket>,
    /// The prepared window of the step (live tickets), between the rank's
    /// `prepare_commit` and `publish_commit`.
    pub prepared: Option<PreparedRankLane>,
    /// The root's planned source (a copy plan may still fall back after its
    /// confirmation).
    pub source: ProposalSource,
    /// An emitter-forced window (`LaneKind::Forced`).
    pub forced: bool,
    /// The plan's head is live (aligned or lagging): its arenas map with the
    /// window's, its rows are appended at prepare.
    pub head_live: bool,
    /// The head's request state saved before a confirmation probe's Draft.
    pub probe_saved: Option<DraftHeadRequestState>,
    /// What the confirmation probe decided (`None`: no probe ran).
    pub probe: Option<ProbeOutcome>,
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

/// Whether one draft round of `ops` ops runs as ONE batched head call
/// (`HIPFIRE_QWEN4_MTP_DRAFT_ROUNDS=1`, rows scratch allocated): 2..=8 ops, and
/// no op reads a distribution the batched ranking does not leave in the head's
/// shared logit row (`any_distribution_read` with a head that does not
/// re-score; a re-scoring head's read is its host-resident candidates).
///
/// The decision must be the same on every EP rank (the head's gather parity
/// advances once per batched call and once per op serially): it depends only
/// on the root's plan, the process flag and the head's build, never on which
/// rank owns the sampler.
pub(crate) fn draft_round_batchable(
    bundle: &Qwen4Bundle,
    ops: usize,
    any_distribution_read: bool,
) -> bool {
    batchable_round(
        ops,
        bundle.mtp_rows_ready(),
        any_distribution_read,
        bundle.mtp_draft_rescored(),
    )
}

/// [`draft_round_batchable`]'s pure rule.
fn batchable_round(
    ops: usize,
    rows_ready: bool,
    any_distribution_read: bool,
    rescored: bool,
) -> bool {
    rows_ready
        && (2..=crate::mtp_gpu::MTP_DRAFT_LANES_MAX).contains(&ops)
        && (rescored || !any_distribution_read)
}

/// The device work of one draft round as one batched head call: `ops[i]` is
/// `(input token, position)` of `lanes[i]`; lane `i`'s result lands in slot
/// `i` of `top1_out`. Each lane's own head state and draft request state are
/// passed in place (nothing is swapped); the step is the lane's `j`-th draft
/// (`j = position - window.position`), fresh selection exactly at `j == 0`.
pub(crate) fn enqueue_batched_draft(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lanes: &mut [&mut Qwen4LaneState],
    ops: &[(u32, usize)],
    top1_out: &GpuTensor,
) -> Result<(), String> {
    let mut rows: Vec<crate::mtp_gpu::DraftRow<'_>> = Vec::with_capacity(ops.len());
    for (lane, &(token, position)) in lanes.iter_mut().zip(ops) {
        let LaneSpec {
            head,
            draft,
            pending_hidden,
            window,
            ..
        } = lane
            .spec
            .as_mut()
            .ok_or_else(|| "drafting lane lost its MTP state".to_string())?;
        let window_position = window
            .as_ref()
            .ok_or_else(|| "draft op without a pending window".to_string())?
            .position;
        let j = position - window_position;
        rows.push(crate::mtp_gpu::DraftRow {
            state: head,
            request: draft,
            token,
            backbone_hidden: (j == 0).then_some(&*pending_hidden),
            position,
            fresh_qsa_selection: j == 0,
        });
    }
    bundle
        .mtp_forward_rows_deferred(gpu, &mut rows, top1_out)
        .map_err(|e| format!("batched draft step: {e}"))
}

/// Parse lane's slot of a batched draft round: the draft token and the lane's
/// own margin (written to the lane's request state, the head's resident one
/// untouched). A re-scoring head's candidates are the head's, valid until the
/// next lane's finish: read them (`mtp_draft_read`) before finishing another.
pub(crate) fn finish_batched_draft(
    bundle: &mut Qwen4Bundle,
    lane: &mut Qwen4LaneState,
    top: &[u8],
) -> Result<(u32, f32), String> {
    let spec = lane
        .spec
        .as_mut()
        .ok_or_else(|| "drafting lane lost its MTP state".to_string())?;
    let token = bundle
        .mtp_draft_finish_request(&mut spec.draft, top)
        .map_err(|e| e.to_string())?;
    Ok((token, spec.draft.margin))
}

#[cfg(test)]
mod batched_draft_tests {
    use super::batchable_round;

    #[test]
    fn batched_round_needs_two_to_eight_ops_and_the_rows_scratch() {
        for ops in [0, 1, 9, 16] {
            assert!(!batchable_round(ops, true, false, true), "{ops} ops");
        }
        for ops in 2..=8 {
            assert!(batchable_round(ops, true, false, true), "{ops} ops");
            assert!(
                !batchable_round(ops, false, false, true),
                "{ops} ops, no scratch"
            );
        }
    }

    #[test]
    fn a_full_row_read_keeps_the_round_serial_unless_the_head_rescores() {
        assert!(!batchable_round(4, true, true, false));
        assert!(batchable_round(4, true, true, true));
        assert!(batchable_round(4, true, false, false));
    }
}

/// A lane window whose every fallible operation succeeded and whose tickets
/// validated but are still live; [`publish_lane_window`] consumes them.
pub(crate) struct PreparedLaneWindow {
    /// The window outcome (the root decided it; the lane's pending window
    /// holds it until the publish).
    pub result: MtpWindow,
    /// The native prepared handle (live tickets, committed position).
    pub native: PreparedNativeWindow,
}

/// How a lane window's tail differs from the greedy native window.
pub(crate) struct LaneFinish {
    /// The acceptance, when the root fixes it without the target's picks (a
    /// forced advance keeps every row: [`crate::mesh_rank_copy::forced_acceptance`]);
    /// `None`: greedy accept over the lane's picks.
    pub acceptance: Option<GreedyAccept>,
    /// `NativeFinish.head_at_start` (`mesh_rank_copy::head_at_start(steps)`):
    /// no head Draft step is in effect, so the seed's row is appended.
    pub head_at_start: bool,
    /// The end-of-stream rule of the greedy accept when `acceptance` is
    /// `None`.
    pub eos: LaneEos,
}

/// Which end-of-stream token a lane window's greedy accept stops at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LaneEos {
    /// The store's configured end of turn, else the model's EOS (the
    /// singleton's rule).
    Config,
    /// The rule the root admitted the request with (`LanePlan::eos`; `None`
    /// ignores EOS, as the lab oracle's fixed-count lanes do). A rank must
    /// replay the root's decision, so it never re-derives the rule.
    Admitted(Option<u32>),
}

impl LaneEos {
    /// The EOS token the greedy accept stops at (`None`: never).
    pub(crate) fn resolve(self, end_of_turn: Option<u32>, model_eos: u32) -> Option<u32> {
        match self {
            Self::Config => Some(end_of_turn.unwrap_or(model_eos)),
            Self::Admitted(eos) => eos,
        }
    }
}

impl LaneFinish {
    /// A native window: greedy accept, the draft chain's row 0 kept.
    pub(crate) const NATIVE: Self = Self {
        acceptance: None,
        head_at_start: false,
        eos: LaneEos::Config,
    };
}

/// Prepare a Verify lane after the trunk forward: greedy accept over the
/// lane's picks, then every fallible operation of the singleton's window tail
/// on the lane's own states, ending in a non-consuming ticket validation.
/// On `Err` no ticket was consumed. `hidden` is the store's wide-hidden
/// capture, the lane's rows starting at `row_begin`; `picks` are the lane's
/// `1 + d` greedy picks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_lane_window(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    config: LaneSpecConfig,
    hidden: &GpuTensor,
    row_begin: usize,
    picks: &[u32],
    skip_rollback: bool,
) -> Result<PreparedLaneWindow, String> {
    prepare_lane_window_with(
        bundle,
        gpu,
        lane,
        config,
        hidden,
        row_begin,
        picks,
        skip_rollback,
        LaneFinish::NATIVE,
    )
}

/// [`prepare_lane_window`] for any verify window of a lane (native, copy,
/// forced): `finish` carries the acceptance a forced window fixes and the
/// `head_at_start` rule. The head must already stand where the finish
/// expects it (`window.position + window.steps`): a catch-up or an owed-row
/// flush has run.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_lane_window_with(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    config: LaneSpecConfig,
    hidden: &GpuTensor,
    row_begin: usize,
    picks: &[u32],
    skip_rollback: bool,
    finish: LaneFinish,
) -> Result<PreparedLaneWindow, String> {
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
    if !verify_capable(window.route, window.source, window.forced) {
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
    if finish.head_at_start != (window.steps == 0) {
        return Err(format!(
            "finish_lane_window: head_at_start {} disagrees with {} draft steps in effect",
            finish.head_at_start, window.steps
        ));
    }
    let target_snapshot = window
        .target_ticket
        .ok_or_else(|| "finish_lane_window: no target ticket".to_string())?;
    let mtp_ticket = window
        .head_ticket
        .ok_or_else(|| "finish_lane_window: no head ticket".to_string())?;
    let eos = finish
        .eos
        .resolve(config.end_of_turn, bundle.config.eos_token_id);
    let drafts = window.drafts.clone();
    let forced = finish.acceptance.is_some();
    let acceptance = match finish.acceptance {
        Some(acceptance) => acceptance,
        None => accept_native_greedy(&drafts, picks, eos)?,
    };
    let mut block = Vec::with_capacity(rows);
    block.push(window.seed);
    block.extend_from_slice(&drafts);
    let position = window.position;
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
    let finish_acceptance = if skip_rollback && !forced && acceptance.accepted < drafts.len() {
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
    let native = lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
        prepare_native_window(
            b,
            &mut Qwen4Ops::Single(&mut *gpu),
            NativeFinish {
                position,
                block: &block,
                acceptance: finish_acceptance,
                target_snapshot,
                mtp_ticket,
                pending_hidden,
                verify_hidden: &verify_hidden,
                ple_normed: Some(&ple_normed),
                retain: false,
                head_at_start: finish.head_at_start,
            },
            None,
        )
        .map_err(BundleError::Forward)
    }))
    .map_err(|e| format!("finish_lane_window: {e}"))?;
    Ok(PreparedLaneWindow { result, native })
}

/// Consume the tickets of a prepared lane window ([`publish_native_window`]
/// on the lane's own states). No GPU operation runs; the lane's tickets were
/// validated by [`prepare_lane_window`]. Returns the window.
pub(crate) fn publish_lane_window(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    prepared: PreparedLaneWindow,
) -> Result<MtpWindow, String> {
    let Qwen4LaneState { state, spec, .. } = lane;
    let spec = spec
        .as_mut()
        .ok_or_else(|| "publish_lane_window: lane has no MTP state".to_string())?;
    let LaneSpec { head, draft, .. } = spec;
    let PreparedLaneWindow { result, native } = prepared;
    lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
        publish_native_window(b, &mut Qwen4Ops::Single(&mut *gpu), native)
            .map_err(BundleError::Forward)
    }))
    .map_err(|e| format!("finish_lane_window: {e}"))?;
    Ok(result)
}

/// The head work of an `ArSynced` lane after its AR row: append the seed
/// with the previous position's target hidden, then keep the AR row's wide
/// hidden as pending for the next head row. `dec_row` is the lane's row in `hidden`.
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
    // Fills use the same (token p, target hidden p - 1) pairing as drafting.
    // Keep `pending_hidden` intact until the append has consumed it.
    lane_call(bundle.with_lane(state, Some((&mut *head, draft)), |b| {
        b.mtp_append_token(gpu, seed, Some(pending_hidden), position)
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
/// (always on the prompt-completing tile). On a sampled lane's
/// prompt-completing tile the pick is the lane's first draw: one draw of its
/// sampler (`LaneSampling`, the lane's own RNG) from the final-row logits the
/// tile's forward left in row 0 of the bundle's spec logits, downloaded with
/// one D2H into `sample_host`, exactly as the singleton draws its first
/// token; the device argmax is returned for a greedy lane and for the tiles
/// that do not complete the prompt.
pub(crate) fn prefill_tile(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    tokens: &[u32],
    append_scratch: Option<&mut MtpAppendScratch>,
    sample_host: &mut Vec<f32>,
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
    let pick = lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
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
    .map_err(|e| format!("prefill_tile: {e}"))?;
    // A sampled lane's prompt-completing pick is a draw from the tile's final
    // row (row 0 of the spec logits). Intermediate tiles' picks are never
    // committed and must not advance the lane's RNG.
    if let (Some(pick), true) = (pick, end == host.prompt.len()) {
        let vocab = bundle.config.vocab_size;
        let mut one = [pick];
        let gpu_ref: &Gpu = gpu;
        let bundle_ref: &Qwen4Bundle = bundle;
        lane_sample_step(host, None, &mut one, 0, vocab, sample_host, |scratch| {
            let row = bundle_ref
                .spec_logits_rows(0, 1)
                .map_err(|e| e.to_string())?;
            download_lane_rows(gpu_ref, &row, 0, 1, vocab, scratch)
        })
        .map_err(|e| format!("prefill_tile: {e}"))?;
        return Ok(Some(one[0]));
    }
    Ok(pick)
}

#[cfg(test)]
mod tests {
    // The finish split's ordering guarantee. `Qwen4Bundle`/`Gpu` cannot be built
    // host-only, so the real functions cannot run on a fake; these pin the
    // contract structurally on the source and the signatures.
    const SPEC_SRC: &str = include_str!("mtp_spec.rs");
    const LANES_SRC: &str = include_str!("mtp_lanes.rs");

    /// A rank replays the root's greedy decision, so its accept must stop at
    /// the EOS the root admitted the request with. The lab oracle admits with
    /// `eos: None` (fixed-count lanes ignore EOS) while the model's EOS is
    /// 248044: a rank that re-derived the store's EOS finished a window as
    /// [EOS] (accepted 1) where the root decided [EOS, x, y] (accepted 2).
    #[test]
    fn rank_accept_follows_the_admitted_eos_rule() {
        use super::LaneEos;
        use crate::mtp_spec::accept_native_greedy;
        use hipfire_runtime::spec::accept_greedy_prefix;
        const MODEL_EOS: u32 = 248044;
        let drafts = [248044, 248045];
        let picks = [248044, 248045, 198];
        // The root (None: EOS ignored) and the rank agree.
        let root = accept_greedy_prefix(&drafts, &picks, None);
        let rank_eos = LaneEos::Admitted(None).resolve(None, MODEL_EOS);
        let rank = accept_native_greedy(&drafts, &picks, rank_eos).unwrap();
        assert_eq!(
            (rank.committed.clone(), rank.accepted),
            (root.committed, root.accepted)
        );
        assert_eq!(
            (rank.committed, rank.accepted, rank.hit_eos),
            (vec![248044, 248045, 198], 2, false)
        );
        // The store-derived rule is what used to run on the rank.
        let stale = accept_native_greedy(&drafts, &picks, LaneEos::Config.resolve(None, MODEL_EOS))
            .unwrap();
        assert_eq!(
            (stale.committed, stale.accepted, stale.hit_eos),
            (vec![248044], 1, true)
        );
        // An admitted EOS stops the rank where it stops the root.
        for eos in [Some(MODEL_EOS), Some(248045), Some(7)] {
            let root = accept_greedy_prefix(&drafts, &picks, eos);
            let rank = accept_native_greedy(
                &drafts,
                &picks,
                LaneEos::Admitted(eos).resolve(Some(1), MODEL_EOS),
            )
            .unwrap();
            assert_eq!(
                (rank.committed, rank.accepted, rank.hit_eos),
                (root.committed, root.accepted, root.hit_eos)
            );
        }
        // The singleton rule: the configured end of turn, else the model's.
        assert_eq!(LaneEos::Config.resolve(Some(5), MODEL_EOS), Some(5));
        assert_eq!(LaneEos::Config.resolve(None, MODEL_EOS), Some(MODEL_EOS));
    }

    fn fn_body<'a>(src: &'a str, name: &str) -> &'a str {
        let head = format!("pub(crate) fn {name}(");
        let start = src.find(&head).unwrap_or_else(|| panic!("{name} missing"));
        let rest = &src[start..];
        let end = rest.find("\n}\n").expect("fn end");
        &rest[..end]
    }

    #[test]
    fn finish_split_prepare_validates_and_never_consumes() {
        let body = fn_body(SPEC_SRC, "prepare_native_window");
        assert!(body.contains("ops_window_valid("), "prepare validates");
        for consume in [
            "ops_finish_window(",
            "commit_validated(",
            "retain_window(",
            "pp_window_finalize(",
            "ops_release_window(",
        ] {
            assert!(!body.contains(consume), "prepare must not call {consume}");
        }
        // Validation is the last fallible op: nothing fallible after it but
        // the Ok(..) construction.
        let valid = body.find("ops_window_valid(").unwrap();
        let tail = &body[valid..];
        assert!(!tail.contains("ops.gpu()") && !tail.contains("copy_hidden_row"));
        // Every fallible GPU op precedes the validation.
        for gpu_op in [
            "mtp_append_token(",
            "copy_hidden_row(",
            "ops_rollback_rows(",
        ] {
            assert!(
                body.rfind(gpu_op).unwrap() < valid,
                "{gpu_op} after validation"
            );
        }
    }

    #[test]
    fn finish_split_publish_is_consume_only() {
        let body = fn_body(SPEC_SRC, "publish_native_window");
        assert!(body.contains("ops_finish_window("));
        for gpu_op in [
            "ops.gpu()",
            "copy_hidden_row(",
            "mtp_append_token(",
            "restore",
        ] {
            assert!(!body.contains(gpu_op), "publish must not run {gpu_op}");
        }
        let body = fn_body(LANES_SRC, "publish_lane_window");
        assert!(body.contains("publish_native_window("));
        assert!(!body.contains("prepare_native_window("));
    }

    #[test]
    fn finish_split_wrappers_are_prepare_then_publish() {
        let native = fn_body(SPEC_SRC, "finish_native_window");
        let (p, q) = (
            native.find("prepare_native_window(").expect("prepare"),
            native.find("publish_native_window(").expect("publish"),
        );
        assert!(p < q, "prepare before publish");
        assert!(native.contains("prepare_native_window(picks, ops, f, timers)?"));
        // The lane route calls prepare in the rank's `prepare_commit` and
        // consumes the tickets in its `publish_commit` (the protocol phases).
        let rank_src = include_str!("lane_rank.rs");
        let method = |name: &str| {
            let head = format!("    fn {name}(");
            let start = rank_src
                .find(&head)
                .unwrap_or_else(|| panic!("{name} missing"));
            let rest = &rank_src[start..];
            &rest[..rest.find("\n    }\n").expect("method end")]
        };
        // The rank's phases call the shared rank helper
        // (`mesh_rank_copy::{prepare,publish}_rank_lane`), which holds the
        // prepare/publish split of the lane window.
        let prepare = method("prepare_commit");
        assert!(prepare.contains("prepare_rank_lane(") && !prepare.contains("publish_rank_lane("));
        let publish = method("publish_commit");
        assert!(publish.contains("publish_rank_lane(") && !publish.contains("prepare_rank_lane("));
        let copy_src = include_str!("mesh_rank_copy.rs");
        let helper = |name: &str| {
            let head = format!("pub(crate) fn {name}(");
            let start = copy_src
                .find(&head)
                .unwrap_or_else(|| panic!("{name} missing"));
            let rest = &copy_src[start..];
            &rest[..rest.find("\n}\n").expect("fn end")]
        };
        let prepare = helper("prepare_rank_lane");
        assert!(
            prepare.contains("prepare_lane_window_with(")
                && !prepare.contains("publish_lane_window(")
        );
        let publish = helper("publish_rank_lane");
        assert!(
            publish.contains("publish_lane_window(") && !publish.contains("prepare_lane_window")
        );
        let prepare = fn_body(LANES_SRC, "prepare_lane_window_with");
        assert!(prepare.contains("prepare_native_window("));
        assert!(!prepare.contains("publish_native_window("));
        let wrapper = fn_body(LANES_SRC, "prepare_lane_window");
        assert!(
            wrapper.contains("prepare_lane_window_with(") && wrapper.contains("LaneFinish::NATIVE")
        );
    }

    #[test]
    fn finish_split_signatures() {
        let _: fn(
            &mut Qwen4Bundle,
            &mut Qwen4Ops<'_>,
            NativeFinish<'_>,
            Option<&mut crate::mtp_spec::MtpPhaseTimers>,
        ) -> Result<PreparedNativeWindow, String> = prepare_native_window;
        let _: fn(
            &mut Qwen4Bundle,
            &mut Qwen4Ops<'_>,
            PreparedNativeWindow,
        ) -> Result<MtpWindow, String> = publish_native_window;
        let _: fn(
            &mut Qwen4Bundle,
            &mut Gpu,
            &mut Qwen4LaneState,
            PreparedLaneWindow,
        ) -> Result<MtpWindow, String> = publish_lane_window;
    }

    use super::*;
    use hipfire_runtime::spec_economics::batched_depth;

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

    /// Per-position acceptance on the parity fixture (`REDLINE-TP2` 11, hipfire
    /// singleton baseline): depths 4 and 5 are not measured and take the depth-3
    /// rate. Story at T=1 and code at T=1.
    const STORY_AGREEMENT: [f32; 5] = [0.666, 0.568, 0.617, 0.617, 0.617];
    const CODE_AGREEMENT: [f32; 5] = [1.0, 0.955, 0.914, 0.914, 0.914];

    fn agreement_of(rates: &[f32]) -> Vec<(f32, f32)> {
        rates.iter().map(|&r| (r * 1000.0, 1000.0)).collect()
    }

    /// The lane's own depth choice (`batched_depth` over the per-depth window
    /// price) at lane count `lanes`, the model's per-window price in AR rows
    /// (`window_units`, spill share `(groups - 1) / lanes`), for a cap `k`.
    /// Returns `(depth, tokens per AR-row-unit at that depth)`: above 1.0 the
    /// native window beats an AR row.
    fn chosen_depth(
        model: &LaneCostModel,
        lanes: usize,
        agreement: &[(f32, f32)],
        k: usize,
    ) -> (usize, f32) {
        // A window of depth `d` makes every lane carry `1 + d` rows, so the
        // spill (row groups beyond the first) is the depth's own.
        let mut cost = vec![1.0f32];
        cost.extend((1..=k).map(|d| {
            let rows = vec![1 + d; lanes];
            let spill = crate::gpu_forward_lanes::row_group_count(&rows).saturating_sub(1);
            model.window_units(lanes, d, spill as f64 / lanes as f64)
        }));
        let depth = batched_depth(agreement, k, &cost);
        if depth == 0 {
            return (0, 1.0);
        }
        let mut prefix = 1.0f32;
        let mut expected = 1.0f32;
        for &(a, t) in &agreement[..depth] {
            prefix *= a / t;
            expected += prefix;
        }
        (depth, expected / cost[depth])
    }

    /// The R9700 EP2 truth `[F, a, r, G]` (us) that FLOOR.md's receipts imply:
    /// AR step 19.4 ms at C1 and 64 ms at C8 (F + a = 19.4, F + 8a = 64: a 6.4,
    /// F 13.0 ms), 5.5 ms per extra draft at C1 (`FLOOR` 13), and the 8-lane
    /// 4-row window 259 ms (3 spill groups: 24 verified drafts x 5.5 + 3 G
    /// = 259 - 64, so G 21 ms). ARITH on receipts, not a fit to a trace: the
    /// lane cost model fits these online from the serve's own step walls.
    const R9700_FLOOR_TRUTH_US: [f64; 4] = [13_000.0, 6_400.0, 5_500.0, 21_000.0];

    #[test]
    fn adaptive_depth_follows_acceptance_and_the_lane_count_on_the_r9700_floor_model() {
        let mut model = LaneCostModel::new();
        feed(&mut model, R9700_FLOOR_TRUTH_US, 0, 800);
        let story = agreement_of(&STORY_AGREEMENT);
        let code = agreement_of(&CODE_AGREEMENT);
        let mut table = String::from("lanes | story depth/gain | code depth/gain\n");
        for lanes in 1..=8 {
            let (sd, sg) = chosen_depth(&model, lanes, &story, 5);
            let (cd, cg) = chosen_depth(&model, lanes, &code, 5);
            table.push_str(&format!("{lanes} | {sd} / {sg:.2} | {cd} / {cg:.2}\n"));
            // A confident class is never routed shallower than an unsure one.
            assert!(cd >= sd, "C{lanes}: code depth {cd} < story depth {sd}");
            // At C1 both classes pay natively.
            if lanes == 1 {
                assert!(sg > 1.0 && cg > 1.0, "{table}");
            }
        }
        // Code at its measured acceptance keeps a deep window at low
        // concurrency; story stays shallow (<= 3) everywhere.
        let (cd1, _) = chosen_depth(&model, 1, &code, 5);
        assert!(cd1 >= 4, "{table}");
        for lanes in 1..=8 {
            assert!(chosen_depth(&model, lanes, &story, 5).0 <= 3, "{table}");
        }
        eprintln!("{table}");
    }

    #[test]
    fn deeper_lane_caps_never_lower_the_gain_the_lane_can_pick() {
        // A larger store cap only adds depths to choose from.
        let mut model = LaneCostModel::new();
        feed(&mut model, R9700_FLOOR_TRUTH_US, 0, 800);
        let code = agreement_of(&CODE_AGREEMENT);
        for lanes in 1..=8 {
            let mut last = 0.0f32;
            for k in 1..=5 {
                let (_, gain) = chosen_depth(&model, lanes, &code, k);
                assert!(gain >= last - 1e-3, "C{lanes} K{k}: {gain} < {last}");
                last = gain;
            }
        }
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
        assert_eq!(DEFAULT_DECAY, 0.875);
        for floor_enabled in [false, true] {
            let inherited = [(4.0, 5.0); 63];
            for mut policy in [
                LaneSpecPolicy::with_floor(3, floor_enabled),
                LaneSpecPolicy::promoted_with_floor(3, &inherited, false, floor_enabled),
            ] {
                let mut expected = policy.agreement[0];
                policy.observe_batched(&window(1, 0, 1), 1.0, 1, 3);
                expected = (expected.0 * 0.875, expected.1 * 0.875 + 1.0);
                assert_eq!(policy.agreement[0], expected);
                policy.observe_ar_synced(&window(2, 1, 1), 3);
                expected = (expected.0 * 0.875 + 1.0, expected.1 * 0.875 + 1.0);
                assert_eq!(policy.agreement[0], expected);
                let unchanged = policy.agreement.clone();
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
        let depth = batched_depth(&policy.agreement, 3, &MTP_WINDOW_COST);
        let want = match depth {
            0 => LaneRoute::ArSynced,
            d => LaneRoute::Batched { k_max: d },
        };
        assert_eq!(policy.route(3, LabRoutes::Floor), want);
        // A smaller budget caps the depth.
        let capped = batched_depth(&policy.agreement, 1, &MTP_WINDOW_COST);
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
        let depth = batched_depth(&policy.agreement, 3, &MTP_WINDOW_COST);
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
        policy.agreement.fill((0.0, 2.0));
        assert_eq!(batched_depth(&policy.agreement, 3, &MTP_WINDOW_COST), 0);
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
            let mut policy =
                LaneSpecPolicy::promoted_with_floor(3, &[(0.9, 1.0); 63], true, floor_enabled);
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
        let low = [(0.0, 2.0); 63];
        let mut promoted = LaneSpecPolicy::promoted_with_floor(3, &low, false, true);
        assert!(!promoted.retired());
        assert_eq!(promoted.agreement, low);
        assert_eq!(batched_depth(&promoted.agreement, 3, &MTP_WINDOW_COST), 0);
        let mut fresh = LaneSpecPolicy::with_floor(3, true);
        fresh.agreement = low.to_vec();
        assert_eq!(
            promoted.route(3, LabRoutes::Floor),
            fresh.route(3, LabRoutes::Floor)
        );
        assert_eq!(promoted.route(0, LabRoutes::Floor), LaneRoute::ArSynced);
        // The floor starts fresh: it still retires on costly probe windows.
        let mut probing =
            LaneSpecPolicy::promoted_with_floor(3, &[AGREEMENT_PRIOR; 63], false, true);
        assert_ne!(probing.route(3, LabRoutes::Floor), LaneRoute::ArRetired);
        for _ in 0..3 {
            probing.observe_batched(&window(1, 0, 1), 6.0, 1, 3);
        }
        assert!(probing.retired());
        assert_eq!(probing.route(3, LabRoutes::Floor), LaneRoute::ArRetired);
        // Lab modes still bypass the floor on a live promoted lane.
        let mut lab = LaneSpecPolicy::promoted_with_floor(3, &low, false, true);
        assert_eq!(
            lab.route(2, LabRoutes::Batched),
            LaneRoute::Batched { k_max: 2 }
        );
    }

    #[test]
    fn agreement_history_does_not_widen_the_physical_lane_capacity() {
        for k in [0, 1, 4, LANE_MAX_DRAFTS, usize::MAX] {
            let mut fresh = LaneSpecPolicy::with_floor(k, false);
            let max_k = k.min(LANE_MAX_DRAFTS);
            assert_eq!(fresh.max_k(), max_k);
            assert_eq!(fresh.agreement, vec![AGREEMENT_PRIOR; max_k]);
            let want = if max_k == 0 {
                LaneRoute::ArSynced
            } else {
                LaneRoute::Batched { k_max: max_k }
            };
            assert_eq!(fresh.route(usize::MAX, LabRoutes::Batched), want);
        }

        let mut inherited: Vec<_> = (0..63).map(|i| (i as f32, i as f32 + 1.0)).collect();
        let mut promoted = LaneSpecPolicy::promoted_with_floor(usize::MAX, &inherited, false, true);
        assert_eq!(promoted.agreement, inherited);
        assert_eq!(promoted.max_k(), LANE_MAX_DRAFTS);
        inherited[62] = (0.0, 0.0);
        assert_eq!(promoted.agreement[62], (62.0, 63.0));
        let storage = promoted.agreement.as_ptr();
        let capacity = promoted.agreement.capacity();
        assert_eq!(
            promoted.route(usize::MAX, LabRoutes::Batched),
            LaneRoute::Batched {
                k_max: LANE_MAX_DRAFTS
            }
        );
        promoted.observe_batched(&window(2, 1, 1), 1.0, 1, LANE_MAX_DRAFTS);
        promoted.observe_ar_synced(&window(2, 1, 1), LANE_MAX_DRAFTS);
        assert_eq!(promoted.agreement.len(), 63);
        assert_ne!(promoted.agreement[62], (62.0, 63.0));
        assert_eq!(promoted.agreement.as_ptr(), storage);
        assert_eq!(promoted.agreement.capacity(), capacity);
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
