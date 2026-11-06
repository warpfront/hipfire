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
    accept_native_greedy, floor_with_call_log, mtp_append_prefill_chunk, mtp_prefill_chunk,
    mtp_prefill_chunk_needs_pick, mtp_reuse_prefill_ids_enabled,
    mtp_skip_intermediate_pick_enabled, prepare_native_window, publish_native_window,
    singleton_floor_ar_probe_tokens, singleton_floor_min_windows, CommitFill, HeadTicket,
    NativeFinish, PreparedNativeWindow, Qwen4Floor, Qwen4Ops, TargetTicket, MTP_WINDOW_COST,
};

use hipfire_dispatch::pipeline::DraftHeadRequestState;
use hipfire_runtime::spec::{GreedyAccept, MtpWindow};
use hipfire_runtime::spec_economics::{
    observe_agreement, Action, NativeWindow, WindowStats, AGREEMENT_PRIOR, DEFAULT_DECAY,
    FLOOR_PROBE_WINDOWS,
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
/// Prior `[F, a, r, G]` (µs) of the EP2 R9700 mesh (gfx1201). Fitted to the
/// served lane steps of the cp5 receipts (`mgmtp/beatstack/runs/cp5`, beta
/// 7eb5fe3674, `HIPFIRE_CB_PHASES` `[fn-lane-step]` lines, 45 061 pure-decode
/// steps; script `stepshep/lcm-runs/fit_cp5.py`, non-negative least squares
/// over the per-(lanes, native windows) medians): AR steps 18.9 ms at one lane
/// to 37.5 ms at eight (`F = 15.7 ms`, `a = 2.7 ms`), 3.2 ms per verified
/// draft, and a step whose rows overflow the first row group (4 lanes, 3
/// native windows: 77.2 ms against the 45.9 ms the no-spill terms predict)
/// pays `G = 31 ms`. The gfx1151 prior above prices a one-lane two-draft
/// window at `1 + 16.4 / 26.6 = 1.62` AR rows, the measured 24.0 / 17.4 ms is
/// 1.38, and its `F + 8a = 94.5 ms` against 37.5 ms measured is what drove
/// the pooled fit's `F` negative after a C8 AR history.
const COST_PRIOR_GFX1201_US: [f64; 4] = [15_700.0, 2_700.0, 3_250.0, 31_000.0];
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
    /// The ridge prior `θ₀` (µs) of this model's arch.
    prior: [f64; 4],
    /// Steps observed (usable walls only).
    observed: u64,
}

impl LaneCostModel {
    /// The gfx1151 prior (every arch without a measured profile).
    pub(crate) fn new() -> Self {
        Self::with_prior(COST_PRIOR_US)
    }

    /// A fresh model whose ridge prior is `prior` = `[F, a, r, G]` (µs).
    pub(crate) fn with_prior(prior: [f64; 4]) -> Self {
        Self {
            xtx: [[0.0; 4]; 4],
            xty: [0.0; 4],
            prior,
            observed: 0,
        }
    }

    /// One finished pure-decode step of `lanes` decoding lanes, `spill` row
    /// groups beyond the first and `verified` verified drafts in all,
    /// `wall_us` long. An unusable wall is dropped.
    pub(crate) fn observe(&mut self, lanes: usize, spill: usize, verified: usize, wall_us: f32) {
        if !(wall_us.is_finite() && wall_us > 0.0) {
            return;
        }
        self.observed += 1;
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
            m[i][4] = self.xty[i] + prior * self.prior[i];
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
            theta = self.prior;
        }
        for t in &mut theta[1..] {
            *t = t.max(0.0);
        }
        theta
    }

    /// The clamped `(F, a, r, G)` in µs.
    pub(crate) fn fit(&self) -> (f64, f64, f64, f64) {
        let [f, a, r, g] = self.solve();
        (f, a, r, g)
    }

    /// Steps this model has observed.
    pub(crate) fn observed(&self) -> u64 {
        self.observed
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

/// `HIPFIRE_QWEN4_EP_LANE_AR_FLOOR` (developer; default off, `1` on): whether EP mesh
/// native-MTP lanes run the AR floor (measured window pricing with sticky retirement to
/// head-free AR). Off (the default), a mesh lane routes its draft depth from its own agreement
/// ([`Qwen4Floor::provisional_depth`]; depth 0 is a one-token `ArSynced` window, never sticky)
/// and the n-gram copy lane is never stopped by the floor. On, the floor runs unless
/// `HIPFIRE_MTP_AR_FLOOR=0`.
///
/// Why off: a lane's floor prices each drafting window an even share of its step's row-group
/// spill and AR none, so retiring one lane looks like it saves a share of the step, while the
/// other lanes keep drafting and the step does not shrink; and a lane whose first window
/// rejected early holds the shallow cap it was moved to (the deeper cap is never re-measured)
/// until it retires. On EP2 R9700 code C8 (cp8/cp9, CpFix j8164) one or two of eight lanes
/// retired after 16 windows in every load and decoded 1 token per 72 ms step while the others
/// emitted about 6: the last stream ran alone for about 1.8 s and the cell failed the
/// concurrency check (all-8-decoding 0.39-0.48 of its span) at 269-292 tok/s, against 434-443
/// tok/s and a pass with the floor off. Story C4 1.22x, story C8 1.05x, code C8 1.34x with
/// the floor off (CpFix j8168, 2 loads per arm).
pub(crate) fn ep_mesh_lane_floor_enabled() -> bool {
    floor_env_enabled() && hipfire_config::developer_bool("HIPFIRE_QWEN4_EP_LANE_AR_FLOOR", false)
}

const LANE_MAX_DRAFTS: usize = crate::gpu_forward_lanes::LANE_MAX_ROWS - 1;

/// `HIPFIRE_QWEN4_LANE_COST` (developer; `0` opts out): the measured lane
/// cost profile of an arch ([`LaneCostProfile::for_arch`]). Off, every arch
/// runs the gfx1151-prior, store-pooled, probe-budget-retire lane floor
/// this module had before profiles.
pub(crate) fn lane_cost_profile_enabled() -> bool {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_LANE_COST", true)
}

/// `HIPFIRE_QWEN4_LANE_ROW_CAP` (developer; default off, `1` on): the gfx1201
/// profile's even split of one 8-row group over the step's lanes
/// ([`LaneCostProfile::step_row_cap`]). Off (the default since the all-rows
/// lane step, `HIPFIRE_QWEN4_LANE_ALLROWS`, runs a >8-row step as one pass), the
/// lanes of a step draft up to their own budgets and the floors learn the
/// spill (`G`) from the priced steps; `1` restores the one-group cap.
pub(crate) fn lane_row_cap_enabled() -> bool {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_LANE_ROW_CAP", false)
}

/// What a lane's floor prices windows with and how it retires, per arch.
///
/// The lane floor is the singleton's [`Qwen4Floor`]; the profile is what the
/// lane feeds it and with which rules:
///
/// - `legacy` (every arch without a measured profile, gfx1151 included, and
///   every arch under `HIPFIRE_QWEN4_LANE_COST=0`): the gfx1151 prior
///   [`COST_PRIOR_US`], one fit pooled over the store's lanes and requests
///   (the root's [`LaneCostModel`]), retirement at the probe budget with no
///   AR probes: the floor this module shipped with, bit for bit.
/// - `gfx1201`: the measured prior [`COST_PRIOR_GFX1201_US`]; **one fit per
///   request** (the lane's own [`LaneCostModel`], born at admission from the
///   prior and fed only the steps that lane takes part in), so the C8 AR
///   history of an earlier request or of other lanes cannot steer a new
///   request's price; and the singleton's retire rules
///   ([`singleton_floor_min_windows`] observed windows before a verdict may
///   retire it, [`singleton_floor_ar_probe_tokens`] AR probe windows at a
///   would-be retirement, both read from the singleton's own
///   `HIPFIRE_MTP_FLOOR_MIN_WINDOWS` / `HIPFIRE_MTP_FLOOR_AR_PROBE`);
///   and a **row cap** on the step ([`LaneCostProfile::step_row_cap`]): the
///   root splits one first row group evenly over the step's lanes (in request
///   order, the odd rows to the first lanes) and a lane drafts at most its
///   share minus its seed row, because a spilled group costs `G = 31 ms` on
///   this arch (cp5: 4 lanes, 3 native windows 77 ms against 40.6 ms for 2
///   native and 24.5 ms AR) and nothing a window emits pays for it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LaneCostProfile {
    name: &'static str,
    prior_us: [f64; 4],
    per_request: bool,
    retire_after: usize,
    probe_tokens: usize,
    /// Most verify rows one step may carry before its rows spill into a second
    /// once-over-all-rows group (0: the step's rows are not capped).
    step_rows: usize,
}

impl LaneCostProfile {
    /// The pre-profile lane floor.
    pub(crate) const LEGACY: Self = Self {
        name: "legacy",
        prior_us: COST_PRIOR_US,
        per_request: false,
        retire_after: FLOOR_PROBE_WINDOWS,
        probe_tokens: 0,
        step_rows: 0,
    };

    /// The gfx1201 (EP2 R9700) profile under the singleton's retire rules.
    pub(crate) fn gfx1201() -> Self {
        Self {
            name: "gfx1201",
            prior_us: COST_PRIOR_GFX1201_US,
            per_request: true,
            retire_after: singleton_floor_min_windows(),
            probe_tokens: singleton_floor_ar_probe_tokens(),
            step_rows: if lane_row_cap_enabled() {
                crate::gpu_forward_lanes::LANE_MAX_ROWS
            } else {
                0
            },
        }
    }

    /// The profile of `arch` (a `Gpu::arch` string).
    pub(crate) fn for_arch(arch: &str) -> Self {
        Self::for_arch_with(arch, lane_cost_profile_enabled())
    }

    fn for_arch_with(arch: &str, enabled: bool) -> Self {
        match arch {
            "gfx1201" if enabled => Self::gfx1201(),
            _ => Self::LEGACY,
        }
    }

    pub(crate) fn name(&self) -> &'static str {
        self.name
    }

    /// The rows a step may carry without spilling into a second row group
    /// (`G`), `None` when the profile does not cap them.
    pub(crate) fn step_row_cap(&self) -> Option<usize> {
        (self.step_rows > 0).then_some(self.step_rows)
    }

    /// The prior `[F, a, r, G]` (µs).
    pub(crate) fn prior_us(&self) -> [f64; 4] {
        self.prior_us
    }

    /// A fresh floor under this profile's retire rules. The legacy profile is
    /// exactly [`Qwen4Floor::new`].
    fn floor(&self) -> Qwen4Floor {
        Qwen4Floor::with_retire_after(MTP_WINDOW_COST, self.retire_after)
            .with_retire_evidence(self.probe_tokens)
    }
}

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
    /// The profile this lane was admitted under.
    profile: LaneCostProfile,
    /// This request's own cost fit (`profile.per_request()`); `None` prices
    /// from the store-pooled model.
    cost: Option<LaneCostModel>,
}

impl LaneSpecPolicy {
    /// The legacy-profile policy (tests).
    #[cfg(test)]
    pub(crate) fn with_floor(k: usize, floor_enabled: bool) -> Self {
        Self::with_profile(k, floor_enabled, &LaneCostProfile::LEGACY)
    }

    /// An admitted lane's policy under `profile`: a fresh floor, fresh
    /// agreement and, for a per-request profile, a fresh cost fit.
    pub(crate) fn with_profile(k: usize, floor_enabled: bool, profile: &LaneCostProfile) -> Self {
        let max_k = k.min(LANE_MAX_DRAFTS);
        Self::with_agreement(
            max_k,
            vec![AGREEMENT_PRIOR; max_k],
            false,
            floor_enabled,
            profile,
        )
    }

    fn with_agreement(
        max_k: usize,
        agreement: Vec<(f32, f32)>,
        promoted_retired: bool,
        floor_enabled: bool,
        profile: &LaneCostProfile,
    ) -> Self {
        let mut floor = profile.floor();
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
            profile: *profile,
            cost: profile
                .per_request
                .then(|| LaneCostModel::with_prior(profile.prior_us)),
        }
    }

    /// The policy of a promoted singleton: the drafter's per-depth
    /// `agreement` seeds the lane (performance only); `retired` (the
    /// singleton's floor retired it) keeps the lane `ArRetired`, otherwise
    /// the lane floor starts fresh like any admitted lane.
    #[cfg(test)]
    pub(crate) fn promoted_with_floor(
        k: usize,
        agreement: &[(f32, f32)],
        retired: bool,
        floor_enabled: bool,
    ) -> Self {
        Self::promoted_with_profile(
            k,
            agreement,
            retired,
            floor_enabled,
            &LaneCostProfile::LEGACY,
        )
    }

    /// [`Self::promoted_with_floor`] under `profile`.
    pub(crate) fn promoted_with_profile(
        k: usize,
        agreement: &[(f32, f32)],
        retired: bool,
        floor_enabled: bool,
        profile: &LaneCostProfile,
    ) -> Self {
        Self::with_agreement(
            k.min(LANE_MAX_DRAFTS),
            agreement.to_vec(),
            retired,
            floor_enabled,
            profile,
        )
    }

    /// One finished pure-decode step this lane took part in: trains the
    /// lane's own cost fit (a per-request profile; the pooled model of the
    /// store trains elsewhere).
    pub(crate) fn observe_step(
        &mut self,
        lanes: usize,
        spill: usize,
        verified: usize,
        wall_us: f32,
    ) {
        if let Some(cost) = self.cost.as_mut() {
            cost.observe(lanes, spill, verified, wall_us);
        }
    }

    /// This request's own cost fit (tests).
    #[cfg(test)]
    pub(crate) fn own_cost(&self) -> Option<&LaneCostModel> {
        self.cost.as_ref()
    }

    /// AR-row price of a window of `drafts` verified drafts: this request's
    /// own fit under a per-request profile, else the store's `pooled` model.
    pub(crate) fn window_units(
        &self,
        pooled: &LaneCostModel,
        lanes: usize,
        drafts: usize,
        spill_share: f64,
    ) -> f32 {
        self.cost
            .as_ref()
            .unwrap_or(pooled)
            .window_units(lanes, drafts, spill_share)
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

    /// A finished `ArSynced` window of budget `k`: one AR row. While the
    /// floor is taking the AR probe tokens of a would-be retirement (only a
    /// profile with retire evidence) the row is one of them: the singleton's
    /// `floor_observe_ar` re-prices AR with it, here one unit like the
    /// calibration. Otherwise it is observed as an interleaved window of one
    /// unit. A zero budget observes nothing.
    pub(crate) fn observe_ar_synced(&mut self, window: &MtpWindow, k: usize) {
        if k == 0 {
            return;
        }
        observe_agreement(&mut self.agreement, window, DEFAULT_DECAY);
        if self.floor_enabled {
            if self.floor.needs_ar_sample() {
                self.floor.observe_calibration(1.0, &self.agreement);
            } else {
                self.floor.observe_native(
                    NativeWindow::Interleaved,
                    window,
                    1.0,
                    k,
                    &self.agreement,
                );
            }
        }
    }
}

/// `HIPFIRE_QWEN4_LANE_COST_TRACE=1` (developer; default off): one
/// [`LANE_COST_TRACE_PREFIX`] JSON line on stderr per lane route decision
/// (`"event":"route"`: the fitted `F, a, r, G`, the predicted native price per
/// emitted token at each depth against the AR unit, the floor's own state,
/// the chosen route, lane request tag / generation / window and step) and one
/// per priced step the cost fits learn from (`"event":"step"`).
pub(crate) fn lane_cost_trace_enabled() -> bool {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_LANE_COST_TRACE", false)
}

/// The prefix of the lane cost trace lines.
pub(crate) const LANE_COST_TRACE_PREFIX: &str = "QWEN4_LANE_COST_TRACE";

/// Where a traced route decision was taken.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RouteTraceCtx {
    pub request_tag: u64,
    pub owner_generation: u64,
    pub window_seq: u64,
    pub step_seq: u64,
    /// Lanes the step carries.
    pub lanes: usize,
}

fn route_name(route: LaneRoute) -> String {
    match route {
        LaneRoute::Ar => "ar".to_string(),
        LaneRoute::ArSynced => "ar_synced".to_string(),
        LaneRoute::ArRetired => "ar_retired".to_string(),
        LaneRoute::Batched { k_max } => format!("batched:{k_max}"),
    }
}

impl LaneSpecPolicy {
    /// The `"event":"route"` trace line of a decision: `pooled` is the
    /// store's model (the price of a lane without its own fit), `budget` the
    /// window's draft budget and `lab` the route mode. `native` lists, per
    /// depth within the budget, the window's price in AR rows at this step's
    /// lane count with no spill, the tokens it emits at the lane's measured
    /// agreement and their ratio, the number the floor weighs against the AR
    /// unit (1 per token).
    pub(crate) fn route_trace_line(
        &self,
        ctx: &RouteTraceCtx,
        pooled: &LaneCostModel,
        budget: usize,
        lab: LabRoutes,
        route: LaneRoute,
    ) -> String {
        let model = self.cost.as_ref().unwrap_or(pooled);
        let (f, a, r, g) = model.fit();
        let mut native = String::new();
        let depths = budget.min(self.max_k).min(self.agreement.len());
        let mut prefix = 1.0f32;
        let mut tokens = 1.0f32;
        for (i, &(accepted, total)) in self.agreement.iter().take(depths).enumerate() {
            prefix *= accepted / total.max(f32::MIN_POSITIVE);
            tokens += prefix;
            let units = model.window_units(ctx.lanes, i + 1, 0.0);
            if !native.is_empty() {
                native.push(',');
            }
            native.push_str(&format!(
                "{{\"depth\":{},\"units\":{units:.4},\"tokens\":{tokens:.4},\"units_per_token\":{:.4}}}",
                i + 1,
                units / tokens
            ));
        }
        let floor = self
            .floor
            .trace_line("", &self.agreement, "route", 0.0, 0, 0);
        format!(
            "{LANE_COST_TRACE_PREFIX} {{\"event\":\"route\",\"tag\":{},\"gen\":{},\"window_seq\":{},\"step_seq\":{},\"lanes\":{},\"profile\":\"{}\",\"fit\":\"{}\",\"obs\":{},\"fit_us\":{{\"F\":{f:.1},\"a\":{a:.1},\"r\":{r:.1},\"G\":{g:.1}}},\"budget\":{budget},\"lab\":{},\"floor_enabled\":{},\"retired\":{},\"route\":\"{}\",\"ar_units_per_token\":1.0,\"native\":[{native}],\"floor\":{}}}",
            ctx.request_tag,
            ctx.owner_generation,
            ctx.window_seq,
            ctx.step_seq,
            ctx.lanes,
            self.profile.name(),
            if self.cost.is_some() { "request" } else { "pooled" },
            model.observed(),
            lab != LabRoutes::Floor,
            self.floor_enabled,
            self.retired(),
            route_name(route),
            floor.trim_start(),
        )
    }

    /// Print the decision's trace line to stderr.
    pub(crate) fn trace_route(
        &self,
        ctx: &RouteTraceCtx,
        pooled: &LaneCostModel,
        budget: usize,
        lab: LabRoutes,
        route: LaneRoute,
    ) {
        eprintln!("{}", self.route_trace_line(ctx, pooled, budget, lab, route));
    }
}

/// The `"event":"step"` trace line of a priced step: what every lane's fit
/// learned from it.
pub(crate) fn step_trace_line(
    step_seq: u64,
    lanes: usize,
    spill: usize,
    verified: usize,
    wall_us: f64,
    priced: bool,
    phase_us: [f64; 3],
) -> String {
    format!(
        "{LANE_COST_TRACE_PREFIX} {{\"event\":\"step\",\"step_seq\":{step_seq},\"lanes\":{lanes},\"spill\":{spill},\"verified\":{verified},\"rows\":{},\"row_groups\":{},\"wall_us\":{wall_us:.1},\"priced\":{priced},\"draft_us\":{:.1},\"verify_us\":{:.1},\"prepare_us\":{:.1}}}",
        lanes + verified,
        spill + 1,
        phase_us[0],
        phase_us[1],
        phase_us[2]
    )
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
        None,
    )
}

/// Whether a lane step's prepare rebuilds a window's kept drafts with one
/// batched decode-fill Append per lane ([`CommitFill`]) instead of one
/// `mtp_append_token` per row (`HIPFIRE_QWEN4_LANE_COMMIT_FILL`): on unless
/// `0`, exact gfx1201 only. The batched Append is the per-row kernels' bytes
/// (`append_rows`' `DecodeFill` route, as the singleton takeover fill).
pub(crate) fn lane_commit_fill_enabled(gpu: &Gpu) -> bool {
    gpu.arch_caps.is_gfx1201()
        && hipfire_config::developer_bool("HIPFIRE_QWEN4_LANE_COMMIT_FILL", true)
}

/// [`prepare_lane_window`] for any verify window of a lane (native, copy,
/// forced): `finish` carries the acceptance a forced window fixes and the
/// `head_at_start` rule. The head must already stand where the finish
/// expects it (`window.position + window.steps`): a catch-up or an owed-row
/// flush has run. `fill` is the store's Append scratch when the commit fill
/// is on ([`lane_commit_fill_enabled`]); the kept drafts then embed from the
/// lane forward's device ids at `row_begin + 1..` when they are still there.
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
    fill: Option<&mut MtpAppendScratch>,
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
                fill: fill.map(|scratch| CommitFill {
                    scratch,
                    ids_row: row_begin,
                }),
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
    sample_host: &mut crate::lane::SampleScratch,
    sample_arena: &mut Option<hipfire_runtime::lane_draw::LaneDrawArena>,
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
        if crate::lane::lane_device_sample(gpu) {
            let logits = bundle
                .spec_logits_rows(0, 1)
                .map_err(|e| format!("prefill_tile: {e}"))?;
            let mut request = [crate::lane::LaneDeviceRequest {
                host: &mut *host,
                drafts: None,
                picks: &mut one,
                eos: 0,
                logits,
            }];
            crate::lane::lane_sample_device(gpu, sample_arena, vocab, &mut request)
                .map_err(|e| format!("prefill_tile: {e}"))?;
            return Ok(Some(one[0]));
        }
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

/// The head half of [`prefill_tile`] for an MTP lane whose trunk rows a prefill wave already
/// forwarded (`Qwen4Bundle::prefill_wave_wide`): the lane's `tokens` are rows
/// `hidden_row0..hidden_row0 + tokens.len()` of the wave's wide-hidden capture, which the caller
/// took into the bundle's resident spec hidden buffer, the buffer
/// [`mtp_prefill_chunk`]'s head append reads. The head append is `mtp_prefill_chunk`'s own
/// (`mtp_append_prefill_chunk`, with the chunk's first row taking `pending` and the capture row
/// offset `hidden_row0`), on the lane's own head state, in the same `with_lane` scope, so the head
/// bytes are the serial tile's: token `p` pairs with the target hidden of `p - 1`, `pending`
/// ends on the chunk's last row. The trunk already advanced `state.position` to the tile's end
/// (as the serial tile's forward has by the time its head fill runs); the head must still be at
/// the tile's start. The trunk ids are not reused (`reuse_trunk_ids = false`: the wave's token
/// ids are the whole burst's, the head uploads the chunk's own, the same bytes).
pub(crate) fn fill_head_after_wave(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    tokens: &[u32],
    hidden_row0: usize,
    append_scratch: Option<&mut MtpAppendScratch>,
) -> Result<(), String> {
    let Qwen4LaneState { host, state, spec } = lane;
    let spec = spec
        .as_mut()
        .ok_or_else(|| "fill_head_after_wave: lane has no MTP state".to_string())?;
    let LaneSpec {
        head,
        draft,
        pending_hidden,
        ..
    } = spec;
    let start = host.prefilled;
    let end = start + tokens.len();
    if state.position != end || head.position() != start {
        return Err(format!(
            "fill_head_after_wave: I1: trunk position {} (want {end}), head position {} (want {start})",
            state.position,
            head.position()
        ));
    }
    lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
        b.poll_npu_trunk(gpu)?;
        match mtp_append_prefill_chunk(
            b,
            &mut Qwen4Ops::Single(&mut *gpu),
            tokens,
            start,
            hidden_row0,
            pending_hidden,
            append_scratch,
            false,
            &|| false,
        )
        .map_err(BundleError::Forward)?
        {
            std::ops::ControlFlow::Continue(()) => Ok(()),
            std::ops::ControlFlow::Break(()) => unreachable!("a never-abort fill cannot break"),
        }
    }))
    .map_err(|e| format!("fill_head_after_wave: {e}"))
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

    /// The wave's head fill is the serial tile's own head append (same function, same scope,
    /// same pairing) over the captured rows: it never runs a trunk forward, checks I1 with the
    /// trunk already at the tile's end, takes the lane's own row offset in the capture, and
    /// embeds from its own ids (the trunk forward's ids are the whole burst's).
    #[test]
    fn wave_head_fill_is_the_serial_tiles_head_append() {
        let body = fn_body(LANES_SRC, "fill_head_after_wave");
        assert!(body.contains("mtp_append_prefill_chunk("));
        assert!(body.contains("bundle.with_lane(state, Some((head, draft))"));
        assert!(body.contains("state.position != end || head.position() != start"));
        assert!(body.contains("start,\n            hidden_row0,"));
        assert!(
            body.contains("false,\n            &|| false,"),
            "no trunk-id reuse, never aborts"
        );
        for trunk in [
            "mtp_prefill_chunk(",
            "ops_prefill_rows",
            "spec_prefill_rows",
            "forward_chunk",
        ] {
            assert!(
                !body.contains(trunk),
                "the head fill must not run the trunk: {trunk}"
            );
        }
        // The serial tile pairs with the same append (through `mtp_prefill_chunk`).
        assert!(fn_body(LANES_SRC, "prefill_tile").contains("mtp_prefill_chunk("));
        assert!(SPEC_SRC.contains("pub(crate) fn mtp_append_prefill_chunk("));
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
            let spill = crate::gpu_forward_lanes::row_group_count(&rows, &vec![false; rows.len()])
                .saturating_sub(1);
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

    // ---- lane cost profiles (gfx1201 measured prior, per-request fit) ----

    /// Eight AR lanes, no drafts, no spill, `wall_us` long: the C8 history of
    /// an earlier request (or of the other lanes).
    fn c8_ar_history(model: &mut LaneCostModel, steps: usize, wall_us: f32) {
        for _ in 0..steps {
            model.observe(8, 0, 0, wall_us);
        }
    }

    #[test]
    fn the_gfx1151_prior_is_pinned_and_the_legacy_profile_is_the_old_floor() {
        assert_eq!(COST_PRIOR_US, [16_900.0, 9_700.0, 8_200.0, 18_500.0]);
        assert_eq!(LaneCostProfile::LEGACY.prior_us(), COST_PRIOR_US);
        assert_eq!(
            LaneCostModel::new().fit(),
            LaneCostModel::with_prior(COST_PRIOR_US).fit()
        );
        // The legacy floor is `Qwen4Floor::new` exactly: the probe-budget
        // retire, no AR probes.
        let policy = LaneSpecPolicy::with_profile(3, true, &LaneCostProfile::LEGACY);
        let diag = policy.floor.diagnostics();
        let old = Qwen4Floor::new(MTP_WINDOW_COST).diagnostics();
        assert_eq!(diag.retire_after, FLOOR_PROBE_WINDOWS);
        assert_eq!(diag.probe_tokens, 0);
        assert_eq!(
            (diag.retire_after, diag.probe_tokens, diag.decay),
            (old.retire_after, old.probe_tokens, old.decay)
        );
        assert!(
            policy.cost.is_none(),
            "legacy lanes price from the store's pooled model"
        );
    }

    #[test]
    fn profiles_follow_the_arch_and_gfx1151_stays_legacy() {
        for arch in ["gfx1151", "gfx1100", "gfx1200", "gfx942", "", "gfx1201x"] {
            assert_eq!(
                LaneCostProfile::for_arch_with(arch, true),
                LaneCostProfile::LEGACY,
                "{arch}"
            );
        }
        // The kill switch: gfx1201 with HIPFIRE_QWEN4_LANE_COST=0.
        assert_eq!(
            LaneCostProfile::for_arch_with("gfx1201", false),
            LaneCostProfile::LEGACY
        );
        let p = LaneCostProfile::for_arch_with("gfx1201", true);
        assert_eq!(p.name(), "gfx1201");
        assert_eq!(p.prior_us(), COST_PRIOR_GFX1201_US);
        assert_ne!(p.prior_us(), COST_PRIOR_US);
        assert!(p.per_request);
        // A lane admitted under it carries a fit of its own, born at the prior.
        let lane = LaneSpecPolicy::with_profile(2, true, &p);
        let own = lane.cost.as_ref().expect("per-request fit");
        assert_eq!(fit_array(own), COST_PRIOR_GFX1201_US);
        assert_eq!(own.observed(), 0);
    }

    #[test]
    fn gfx1201_prior_prices_a_one_lane_native_window_below_its_tokens() {
        // The measured C1 step: 24.0 ms native (2 drafts) against 17.4-18.9 ms
        // AR, 1.3-1.4 AR rows for about 2.1 tokens.
        let model = LaneCostModel::with_prior(COST_PRIOR_GFX1201_US);
        let units = f64::from(model.window_units(1, 2, 0.0));
        let want = 1.0 + 2.0 * 3_250.0 / (15_700.0 + 2_700.0);
        assert!((units - want).abs() < 1e-4, "{units} vs {want}");
        assert!((1.25..1.5).contains(&units), "{units}");
        // A C4 window that spills a row group costs more than the 2-3 tokens
        // it emits (cp5: 3 spilled windows 77 ms against 24.5 ms AR).
        let spilled = model.window_units(4, 2, 1.0 / 3.0);
        assert!(spilled > 2.5, "spilled C4 window {spilled}");
    }

    #[test]
    fn a_fresh_request_does_not_inherit_a_c8_ar_history() {
        // The failure this replaces: a pooled model trained on C8 AR steps on
        // a card whose C8 step (37.5 ms) is far below the gfx1151 prior's
        // F + 8a (94.5 ms) drives F negative, alpha(1) to its 1 ms floor and
        // the price of a 2-draft window at C1 to 17 AR rows: AR for good.
        let mut pooled = LaneCostModel::new();
        c8_ar_history(&mut pooled, 3_000, 37_500.0);
        assert!(pooled.fit().0 < 0.0, "{:?}", pooled.fit());
        assert!(pooled.window_units(1, 2, 0.0) > 10.0);

        // A per-request fit sees only its own lane's steps; a new request's
        // lane is born at the prior whatever happened before it.
        let profile = LaneCostProfile::gfx1201();
        let mut earlier = LaneSpecPolicy::with_profile(2, true, &profile);
        for _ in 0..3_000 {
            earlier.observe_step(8, 0, 0, 37_500.0);
        }
        let earlier_fit = fit_array(earlier.cost.as_ref().unwrap());
        assert_ne!(earlier_fit, COST_PRIOR_GFX1201_US);
        let fresh = LaneSpecPolicy::with_profile(2, true, &profile);
        assert_eq!(
            fit_array(fresh.cost.as_ref().unwrap()),
            COST_PRIOR_GFX1201_US
        );
        let units = f64::from(fresh.window_units(&pooled, 1, 2, 0.0));
        assert!(units < 1.5, "fresh request C1 window {units}");
        // Even the earlier request's own fit stays sane at C1 (the gfx1201
        // prior is within the ridge's reach of the data).
        assert!(f64::from(earlier.window_units(&pooled, 1, 2, 0.0)) < 1.6);
    }

    #[test]
    fn lanes_and_requests_do_not_leak_into_each_other() {
        let profile = LaneCostProfile::gfx1201();
        let mut a = LaneSpecPolicy::with_profile(2, true, &profile);
        let b = LaneSpecPolicy::with_profile(2, true, &profile);
        for k in 0..200 {
            let (lanes, verified, spill) = step_shape(k);
            a.observe_step(lanes, spill, verified, 60_000.0);
        }
        assert_eq!(a.cost.as_ref().unwrap().observed(), 200);
        assert_eq!(b.cost.as_ref().unwrap().observed(), 0);
        assert_eq!(fit_array(b.cost.as_ref().unwrap()), COST_PRIOR_GFX1201_US);
        // A per-request lane ignores the pooled model entirely.
        let mut poisoned = LaneCostModel::new();
        c8_ar_history(&mut poisoned, 3_000, 37_500.0);
        let clean = LaneCostModel::with_prior(COST_PRIOR_GFX1201_US);
        for (lanes, drafts, share) in [(1, 2, 0.0), (3, 2, 0.0), (4, 3, 0.5)] {
            assert_eq!(
                b.window_units(&poisoned, lanes, drafts, share),
                clean.window_units(lanes, drafts, share)
            );
        }
        // A legacy lane does the opposite: it prices from the pooled model it
        // is handed and its own observe_step is a no-op.
        let mut legacy = LaneSpecPolicy::with_profile(2, true, &LaneCostProfile::LEGACY);
        legacy.observe_step(8, 0, 0, 37_500.0);
        assert_eq!(
            legacy.window_units(&poisoned, 1, 2, 0.0),
            poisoned.window_units(1, 2, 0.0)
        );
    }

    /// Route sequence of a lane that keeps losing: every native window
    /// emits one token (nothing accepted) and prices at `native_units` AR
    /// rows; AR windows price one unit and feed the probes. Stops at the
    /// first retired route. The second vector is the floor's `probe_left`
    /// after each window, the third its `observed` windows.
    fn losing_run(
        policy: &mut LaneSpecPolicy,
        native_units: f32,
        windows: usize,
    ) -> (Vec<LaneRoute>, Vec<usize>, Vec<usize>) {
        let (mut routes, mut probe_left, mut observed) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..windows {
            let route = policy.route(2, LabRoutes::Floor);
            routes.push(route);
            match route {
                LaneRoute::Batched { k_max } => {
                    policy.observe_batched(&window(1, 0, k_max), native_units, k_max, 2)
                }
                LaneRoute::ArSynced => policy.observe_ar_synced(&window(1, 0, 0), 2),
                LaneRoute::ArRetired | LaneRoute::Ar => break,
            }
            let diag = policy.floor.diagnostics();
            probe_left.push(diag.probe_left);
            observed.push(diag.observed);
        }
        (routes, probe_left, observed)
    }

    /// The singleton's floor, driven with the same calls the lane policy
    /// makes (the singleton's own configuration): AR calibration of one
    /// unit, then route / observe with the agreement the windows teach.
    fn singleton_floor_run(native_units: f32, windows: usize) -> Vec<LaneRoute> {
        let mut floor =
            Qwen4Floor::with_retire_after(MTP_WINDOW_COST, singleton_floor_min_windows())
                .with_retire_evidence(singleton_floor_ar_probe_tokens());
        floor.set_capacity(2);
        let mut agreement = vec![AGREEMENT_PRIOR; 2];
        floor.observe_calibration(1.0, &agreement);
        let mut routes = Vec::new();
        for _ in 0..windows {
            let provisional = floor.provisional_depth(&agreement, 2);
            let batched = |depth: usize| match depth.min(2) {
                0 => LaneRoute::ArSynced,
                k_max => LaneRoute::Batched { k_max },
            };
            let route = match floor.route(&agreement, 2, provisional) {
                Action::ArHeadFree => LaneRoute::ArRetired,
                Action::ArSynced | Action::Interleaved => LaneRoute::ArSynced,
                Action::Native(depth) => batched(depth),
            };
            routes.push(route);
            match route {
                LaneRoute::Batched { k_max } => {
                    let w = window(1, 0, k_max);
                    observe_agreement(&mut agreement, &w, DEFAULT_DECAY);
                    floor.observe_window(
                        NativeWindow::Batched,
                        WindowStats::of(&w, k_max),
                        native_units,
                        2,
                        &agreement,
                    );
                }
                LaneRoute::ArSynced => {
                    let w = window(1, 0, 0);
                    observe_agreement(&mut agreement, &w, DEFAULT_DECAY);
                    if floor.needs_ar_sample() {
                        floor.observe_calibration(1.0, &agreement);
                    } else {
                        floor.observe_native(NativeWindow::Interleaved, &w, 1.0, 2, &agreement);
                    }
                }
                LaneRoute::ArRetired | LaneRoute::Ar => break,
            }
        }
        routes
    }

    #[test]
    fn the_gfx1201_lane_floor_retires_and_reprobes_like_the_singleton() {
        let profile = LaneCostProfile::gfx1201();
        let mut policy = LaneSpecPolicy::with_profile(2, true, &profile);
        // The floor is built the singleton's way.
        let singleton =
            Qwen4Floor::with_retire_after(MTP_WINDOW_COST, singleton_floor_min_windows())
                .with_retire_evidence(singleton_floor_ar_probe_tokens())
                .diagnostics();
        let lane = policy.floor.diagnostics();
        assert_eq!(
            (lane.retire_after, lane.probe_tokens, lane.decay),
            (
                singleton.retire_after,
                singleton.probe_tokens,
                singleton.decay
            )
        );
        assert!(
            lane.retire_after > FLOOR_PROBE_WINDOWS,
            "{}",
            lane.retire_after
        );
        assert!(lane.probe_tokens > 0);

        // A losing lane (windows price 3 AR rows for 1 token): the lane's
        // route sequence is the singleton floor's, call for call.
        let (lane_routes, probe_left, observed) = losing_run(&mut policy, 3.0, 200);
        assert_eq!(lane_routes, singleton_floor_run(3.0, 200));
        assert_eq!(lane_routes.last(), Some(&LaneRoute::ArRetired));
        assert!(policy.retired());
        // No verdict before `retire_after` observed windows...
        let at_retire = *observed.last().unwrap();
        assert!(
            at_retire >= singleton_floor_min_windows(),
            "retired after {at_retire} observed windows: {lane_routes:?}"
        );
        // ...then exactly the singleton's AR probe tokens, each one an AR
        // row that the floor counts down...
        let n = singleton_floor_ar_probe_tokens();
        let countdown: Vec<usize> = (0..=n).rev().collect();
        assert!(
            probe_left.windows(n + 1).any(|w| w == &countdown[..]),
            "probe_left {probe_left:?}"
        );
        // ...and the last one is followed by the sticky retirement.
        assert_eq!(policy.route(2, LabRoutes::Floor), LaneRoute::ArRetired);
    }

    #[test]
    fn the_legacy_lane_floor_retires_at_the_probe_budget_with_no_probes() {
        let mut policy = LaneSpecPolicy::with_profile(2, true, &LaneCostProfile::LEGACY);
        let mut old = LaneSpecPolicy::with_floor(2, true);
        let (routes, probe_left, observed) = losing_run(&mut policy, 3.0, 200);
        assert_eq!(routes, losing_run(&mut old, 3.0, 200).0);
        assert_eq!(routes.last(), Some(&LaneRoute::ArRetired));
        assert!(routes.len() <= FLOOR_PROBE_WINDOWS + 2, "{routes:?}");
        assert!(probe_left.iter().all(|&p| p == 0), "{probe_left:?}");
        assert!(
            *observed.last().unwrap() <= FLOOR_PROBE_WINDOWS,
            "{observed:?}"
        );
    }

    #[test]
    fn a_winning_gfx1201_lane_keeps_drafting_and_never_probes() {
        let mut policy = LaneSpecPolicy::with_profile(2, true, &LaneCostProfile::gfx1201());
        for _ in 0..64 {
            let route = policy.route(2, LabRoutes::Floor);
            match route {
                LaneRoute::Batched { k_max } => {
                    // 3 tokens for 1.4 AR rows.
                    policy.observe_batched(&window(3, k_max, k_max), 1.4, k_max, 2)
                }
                other => panic!("a winning lane routed {other:?}"),
            }
        }
        assert!(!policy.retired());
    }

    #[test]
    fn the_route_trace_line_carries_fit_prediction_and_choice() {
        let profile = LaneCostProfile::gfx1201();
        let mut policy = LaneSpecPolicy::with_profile(2, true, &profile);
        let pooled = LaneCostModel::new();
        let route = policy.route(2, LabRoutes::Floor);
        let ctx = RouteTraceCtx {
            request_tag: 9,
            owner_generation: 3,
            window_seq: 5,
            step_seq: 17,
            lanes: 1,
        };
        let line = policy.route_trace_line(&ctx, &pooled, 2, LabRoutes::Floor, route);
        let json = line
            .strip_prefix(LANE_COST_TRACE_PREFIX)
            .expect("prefix")
            .trim();
        let v: serde_json::Value = serde_json::from_str(json).expect("one JSON object");
        assert_eq!(v["event"], "route");
        assert_eq!(v["tag"], 9);
        assert_eq!(v["gen"], 3);
        assert_eq!(v["window_seq"], 5);
        assert_eq!(v["step_seq"], 17);
        assert_eq!(v["profile"], "gfx1201");
        assert_eq!(v["fit"], "request");
        assert_eq!(v["obs"], 0);
        assert_eq!(v["fit_us"]["F"], 15_700.0);
        assert_eq!(v["fit_us"]["G"], 31_000.0);
        assert_eq!(v["route"], "batched:2");
        assert_eq!(v["ar_units_per_token"], 1.0);
        let native = v["native"].as_array().expect("native predictions");
        assert_eq!(native.len(), 2);
        let d1 = native[0]["units"].as_f64().unwrap();
        assert!((d1 - (1.0 + 3_250.0 / 18_400.0)).abs() < 1e-3, "{d1}");
        assert!(v["floor"]["ar_us"].as_f64().unwrap() > 0.0);
        // The pooled lane says so.
        let legacy = LaneSpecPolicy::with_profile(2, true, &LaneCostProfile::LEGACY);
        let line = legacy.route_trace_line(&ctx, &pooled, 2, LabRoutes::Floor, LaneRoute::ArSynced);
        let v: serde_json::Value =
            serde_json::from_str(line.strip_prefix(LANE_COST_TRACE_PREFIX).unwrap().trim())
                .unwrap();
        assert_eq!(
            (v["profile"].as_str(), v["fit"].as_str()),
            (Some("legacy"), Some("pooled"))
        );
        assert_eq!(v["fit_us"]["F"], 16_900.0);
        let step: serde_json::Value = serde_json::from_str(
            step_trace_line(17, 4, 1, 6, 77_200.0, true, [1.0, 2.0, 3.0])
                .strip_prefix(LANE_COST_TRACE_PREFIX)
                .unwrap()
                .trim(),
        )
        .unwrap();
        assert_eq!(
            (step["event"].as_str(), step["spill"].as_u64()),
            (Some("step"), Some(1))
        );
    }
}
