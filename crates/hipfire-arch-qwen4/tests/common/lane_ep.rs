// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The EP2 lane-arm executor of the lane suite
//! (`HIPFIRE_SAMPLED_MTP_EXECUTOR=lanes`, `HIPFIRE_SAMPLED_MTP_TOPOLOGY=ep2`).
//!
//! A child of `sampled_mtp_distribution_hw` (shares its config and identity
//! helpers). One process runs ONE arm over ONE `START..END` trial shard: it
//! loads the artifact once through the product loader on the EP mesh (with the
//! serve's `serve.vmm_batch` lane reservation), stages ONE lane store, and runs
//! every trial of the shard through it, recycling lanes (admit -> prefill ->
//! decode -> release) so load and staging are amortised.
//!
//! | arm | store | decode |
//! |---|---|---|
//! | `lane_ar` | AR lanes (no spec), sampled | lane executor AR rows; each row's token drawn from the lane's own RNG (device launch or host sampler, see below) |
//! | `lane_batched` / `lane_adaptive` | native-MTP lanes, every lane MTP (`spec_lanes: None`), no copy lane | `Qwen4Bundle::ep_mesh_decode_step` (Leviathan acceptance against each draft's q) |
//! | `lane_copy` | the same lanes with the copy lane the serve stages by default (`ep::ep_default_copy_config`, `stage_lanes_with_copy`); at most `lane_suite::EP2_COPY_LANES` lanes | the same mesh step: pool-hit windows are copy takeovers verified against the point-mass draft distribution; run on the primed copied-span fixture (`HIPFIRE_SAMPLED_MTP_PROMPT=copy`), its coverage needs copy windows from the counters |
//! | `lane_cross` | native-MTP lanes as above, `HIPFIRE_EP_MTP_LANE_MAX_ACTIVE=3` (the shipped `lane_switch::DEFAULT_MAX_ACTIVE`, pinned) | the same mesh step under the native<->AR switch at 3; 8 lanes only, staggered admission |
//!
//! # Draw route
//!
//! With `HIPFIRE_QWEN4_LANE_DEVICE_SAMPLE` on (the default on gfx1201), every
//! sampled draw (the AR rows, the prompt-completing first token, the native and
//! the copy verify windows) is ONE lane-wide device launch
//! (`hipfire_runtime::lane_draw::lane_draws`): Leviathan against each draft's q
//! (a copy proposal is a point mass) under the shared device sampler's law, with
//! the lane's xorshift32 stream as the RNG. With `=0` (or on a device the launch
//! does not support) the host sampler draws, as before. The two routes draw the
//! same distribution from different streams, so seeded ids differ between
//! them. The arm resolves the route from the loaded `Gpu` (`lane_draw_route`)
//! and records it as `lane_suite.lane_draw_route` (`device` or `host`) in its
//! config; ids produced under one route are a hard config error under the
//! other, a merge refuses mixed routes, and the verdict prints each arm's route
//! and refuses a cohort that ran under different ones.
//!
//! The native arms (`lane_batched`, `lane_adaptive`, `lane_cross`) draft at the
//! lane depth the cohort chose with `HIPFIRE_SAMPLED_MTP_LANE_K` (2, the
//! shipped `ep_lane_spec_k` default, or 3), pinned through
//! `HIPFIRE_EP_MTP_LANE_K` and checked against the depth the store resolves
//! ([`check_lane_k`]). The evidence records `lane_k`, the most drafts any window
//! offered (`max_offered`) and the drafts offered by each request's first
//! window (`first_window_offered`); the coverage refuses a window above K and a
//! run whose first windows never offered K.
//!
//! A trial is `lanes` concurrent requests of the same fixture prompt, admitted
//! together and kept in lockstep (all prefill tiles of a tick, then one mesh
//! step carrying every decoding lane), `TOKENS` ids per request including the
//! prefill-completing first token, EOS padded. Request `i` of global trial `t`
//! is seeded `lane_suite::lane_request_seed(trial_seed, lane_salt(arm), lanes,
//! t, i)` truncated to the lane RNG's 32 bits (`LaneSampling::seed`; the
//! singleton AR arm truncates its seeds the same way). Nothing depends on the
//! slot a request ran in.
//!
//! `lane_cross` is the one arm whose trial is not admitted together
//! (`lane_suite::admit_schedule`: `[3, 5]` at 8 lanes): the first three requests
//! are admitted at tick 0 and the other five on the tick after the first group
//! has forwarded its whole prompt, so the first group's first decode window
//! runs native with three active lanes and its later windows run with eight
//! (AR windows of `lane_switch::native_mode`: AR from `max_active + 2` = 5
//! lanes, a step at 4 lanes holds the previous mode). Request `i` is
//! lane `i` either way. The evidence counts the requests that ran both a
//! native and an AR window (`crossing_requests`) and keeps the occupancy
//! histogram of the mesh steps (occupancy 4, the hold case, is recorded; the
//! coverage requires steps at <= 3 and >= 5); the lane coverage refuses a run without both.
//! A prompt that needs several prefill tiles lets the first group run
//! more native windows before the second joins; the counters record what ran.
//!
//! The evidence the verdict reads is derived from real counters only: each
//! mesh window is classified from the lane's `LaneSpecStats` delta
//! (`Qwen4Bundle::lane_spec_stats_for`) and the step's `lane_switch` decision
//! (`MeshStepFacts::native_mode`: a switched step's draft-less windows are AR
//! windows), and the sums are cross-checked against the step's
//! `MeshStepFacts`; a disagreement fails the shard.

use super::*;
use hipfire_arch_qwen4::ep_mesh_serve::{MeshDecodeRequest, MeshStepFacts};
use hipfire_arch_qwen4::lane::{LaneSampling, LaneSpecConfig, LaneSpecStats};
use hipfire_runtime::slot_batch::{
    BatchStepPlan, RequestAdvance, RequestEpoch, RequestRows, RequestStepKind, RowRange, SlotBatch,
};
use rdna_compute::slot_pool::SlotId;

/// The EP degree of the `ep2` topology.
const EP_DEGREE: usize = 2;

// ------------------------------------------------------------------- arms

/// What a lane arm stages and drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ArmKind {
    /// Target-only AR lanes (the reference).
    Ar,
    /// Native-MTP mesh lanes (no copy lane staged: the store the serve builds
    /// with `HIPFIRE_EP_MTP_LANE_COPY=0`) with the fixed-count pin: native
    /// windows at every lane count.
    Mtp,
    /// Native-MTP mesh lanes with the copy lane the serve stages by default
    /// (`ep::ep_default_copy_config`, steps of at most `COPY_WIDE_LANES`
    /// lanes), driven by the same mesh step: pool-hit windows are copy
    /// takeovers verified against the point-mass draft distribution. At most
    /// `lane_suite::EP2_COPY_LANES` lanes.
    Copy,
    /// Native-MTP mesh lanes under the product step switch
    /// (`lane_suite::CROSS_MAX_ACTIVE`) with staggered admission.
    Cross,
}

impl ArmKind {
    pub(super) fn of(arm: &str) -> Result<Self, String> {
        match arm {
            "lane_ar" => Ok(Self::Ar),
            "lane_batched" | "lane_adaptive" => Ok(Self::Mtp),
            "lane_cross" => Ok(Self::Cross),
            "lane_copy" => Ok(Self::Copy),
            other => Err(format!("{other} is not a lane arm")),
        }
    }

    fn mesh(self) -> bool {
        matches!(self, Self::Mtp | Self::Copy | Self::Cross)
    }
}

/// The lane store's slot count: the serve never stages fewer than two lanes
/// (`BatchLoadRequest::lanes`), so a one-request arm still runs on a two-slot
/// store. The copy arm runs on the serve's eight-lane store
/// ([`COPY_STORE_LANES`]): the copy lane keeps the native `k + 1` step rows per
/// lane (`mesh_rank_copy::copy_budget`) and a step of `COPY_WIDE_LANES` copy
/// windows verifies up to `1 + take cap` rows each, so a store of fewer than
/// six lanes (`k` = 2) rejects such a step on the ranks ("verify rows exceed
/// the store's step rows"); the serve's default concurrency (8) never does.
pub(super) fn store_lanes(kind: ArmKind, lanes: usize) -> usize {
    match kind {
        ArmKind::Copy => COPY_STORE_LANES.max(lanes),
        _ => lanes.max(2),
    }
}

/// Slots of the copy arm's lane store: the serve's default `serve.continuous_batch_size`
/// (the shipped C8 lane count, `lane_suite::CROSS_LANES`).
pub(super) const COPY_STORE_LANES: usize = lane_suite::CROSS_LANES;

/// Host-only environment check of an EP2 lane arm. `max_active` is
/// `HIPFIRE_EP_MTP_LANE_MAX_ACTIVE` as the runtime resolves it.
///
/// * Fixed-count native arms (`Mtp`): at least the lane count; below it the
///   mesh step would switch to AR windows and the arm would not run the native
///   windows its name claims.
/// * The copy arm (`Copy`): the fixed-count rule, and at most
///   `lane_suite::EP2_COPY_LANES` lanes (the copy lane takes windows only in
///   steps of at most that many lanes).
/// * The crossing arm: exactly 8 lanes and exactly `lane_suite::CROSS_MAX_ACTIVE`
///   (pinned explicitly): any other value changes which steps run native.
pub(super) fn check_ep_env(
    kind: ArmKind,
    lanes: usize,
    max_active: usize,
    ep_degree: usize,
) -> Result<(), String> {
    if ep_degree != EP_DEGREE {
        return Err(format!(
            "{}=ep2 needs topology.ep = {EP_DEGREE} (HIPFIRE_TOPOLOGY_EP={EP_DEGREE}); the process resolves {ep_degree}",
            lane_suite::TOPOLOGY_ENV
        ));
    }
    match kind {
        ArmKind::Ar => {}
        ArmKind::Mtp | ArmKind::Copy => {
            if kind == ArmKind::Copy && lanes > lane_suite::EP2_COPY_LANES {
                return Err(format!(
                    "lane_copy runs at most {} lanes (the serve's copy lane takes windows only in steps of at most that many); the process has {lanes}",
                    lane_suite::EP2_COPY_LANES
                ));
            }
            if max_active < lanes {
                return Err(format!(
                    "HIPFIRE_EP_MTP_LANE_MAX_ACTIVE={max_active} is below the {lanes} lanes: the mesh step would run AR windows instead of native ones; pin it to at least {lanes}"
                ));
            }
        }
        ArmKind::Cross => {
            if lanes != lane_suite::CROSS_LANES {
                return Err(format!(
                    "lane_cross runs {} lanes (it admits {} requests, then the rest); the process has {lanes}",
                    lane_suite::CROSS_LANES,
                    lane_suite::CROSS_FIRST_GROUP
                ));
            }
            if max_active != lane_suite::CROSS_MAX_ACTIVE_N {
                return Err(format!(
                    "HIPFIRE_EP_MTP_LANE_MAX_ACTIVE={max_active}: lane_cross pins {} so eight lanes cross the native<->AR switch; any other value changes which steps run native windows",
                    lane_suite::CROSS_MAX_ACTIVE
                ));
            }
        }
    }
    Ok(())
}

/// The lane draft depth check: the EP lane store drafts at `spec_k`
/// (`ep_lane_spec_k`: the load's K capped at `HIPFIRE_EP_MTP_LANE_K`), which
/// must be the cohort's depth `lane_k` (`HIPFIRE_SAMPLED_MTP_LANE_K`) for the
/// native arms (the AR reference drafts nothing).
pub(super) fn check_lane_k(kind: ArmKind, spec_k: usize, lane_k: usize) -> Result<(), String> {
    if kind.mesh() && spec_k != lane_k {
        return Err(format!(
            "the EP lane draft depth resolves to {spec_k} ({}={:?} caps it; the oracle loads K={MTP_K}) but the cohort runs lane K {lane_k} ({}): pin {}={lane_k}",
            lane_suite::EP_LANE_K_ENV,
            std::env::var(lane_suite::EP_LANE_K_ENV).ok(),
            lane_suite::LANE_K_ENV,
            lane_suite::EP_LANE_K_ENV
        ));
    }
    Ok(())
}

// ------------------------------------------------------- seeds and rows

/// The lane RNG seed of request `lane` of global `trial` (32 bits: the lane
/// sampler's state width).
pub(super) fn lane_seed(arm: &str, lanes: usize, trial: usize, lane: usize) -> Result<u32, String> {
    Ok(
        lane_suite::lane_request_seed(trial_seed, lane_suite::lane_salt(arm)?, lanes, trial, lane)
            as u32,
    )
}

/// A request's row: cut after the first EOS, then padded with EOS to `TOKENS`
/// ("positions after an EOS read as EOS").
pub(super) fn finish_row(ids: &[u32], eos: u32) -> Vec<u32> {
    let end = ids
        .iter()
        .position(|&id| id == eos)
        .map_or(ids.len(), |at| at + 1)
        .min(TOKENS);
    let mut row = ids[..end].to_vec();
    row.resize(TOKENS, eos);
    row
}

/// The tokens a request may still emit, the pending seed included, as the
/// serve driver hands them to the mesh (`max_tokens - streamed` with the
/// pending seed unstreamed): `TOKENS + 1 - ids.len()`, at least 1.
pub(super) fn tokens_left(ids_len: usize) -> usize {
    (TOKENS + 1).saturating_sub(ids_len).max(1)
}

/// Whether a request holding `ids` is finished: `TOKENS` ids, or an EOS
/// anywhere (the mesh root never closes a lane on EOS, so a window may carry
/// ids past it; the row is cut after the first EOS).
pub(super) fn request_done(ids: &[u32], eos: u32) -> bool {
    ids.len() >= TOKENS || ids.contains(&eos)
}

// --------------------------------------------------------------- windows

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WindowKind {
    /// A native window (drafts from the head; zero drafts = the calibration
    /// or a draft-less window).
    Native,
    /// A copy window (one-hot proposals).
    Copy,
    /// A head-free target-only window (a retired lane, or an AR row).
    Ar,
}

impl WindowKind {
    fn name(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Copy => "copy",
            Self::Ar => "ar",
        }
    }
}

/// One decode window of one request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct WindowFact {
    pub kind: WindowKind,
    pub offered: usize,
    pub accepted: usize,
    pub emitted: usize,
}

impl WindowFact {
    fn json(&self) -> Value {
        json!({
            "kind": self.kind.name(),
            "offered": self.offered,
            "accepted": self.accepted,
            "emitted": self.emitted,
        })
    }
}

/// The one window a lane ran between two `LaneSpecStats` reads. Exactly one of
/// the native / copy / AR window counters moves by one, no forced window ran,
/// and the draft counters move only with their window kind. `native_mode` is
/// the step's `lane_switch` decision (`MeshStepFacts::native_mode`): in a
/// switched step (`false`) every lane runs the one-token `ArSynced` window,
/// which the lane counters book as a draft-less native window, so it is an
/// AR window here (a drafted window there is an error).
pub(super) fn classify_window(
    before: &LaneSpecStats,
    after: &LaneSpecStats,
    emitted: usize,
    native_mode: bool,
) -> Result<WindowFact, String> {
    let delta = |name: &str, before: u64, after: u64| {
        after
            .checked_sub(before)
            .ok_or_else(|| format!("lane counter {name} went backwards ({before} -> {after})"))
    };
    let windows = delta("windows", before.windows, after.windows)?;
    let ar = delta("ar_windows", before.ar_windows, after.ar_windows)?;
    let copy = delta("copy_windows", before.copy_windows, after.copy_windows)?;
    let forced = delta(
        "forced_windows",
        before.forced_windows,
        after.forced_windows,
    )?;
    let drafts = delta("drafts", before.drafts, after.drafts)? as usize;
    let accepted = delta("accepted", before.accepted, after.accepted)? as usize;
    let copy_drafts = delta("copy_drafts", before.copy_drafts, after.copy_drafts)? as usize;
    let copy_accepted = delta("copy_accepted", before.copy_accepted, after.copy_accepted)? as usize;
    if forced != 0 {
        return Err(format!(
            "a forced window ran in a lane suite request ({forced})"
        ));
    }
    match (windows, ar, copy) {
        (1, 0, 0) if !native_mode => {
            if drafts != 0 || accepted != 0 || copy_drafts != 0 || copy_accepted != 0 {
                return Err(format!(
                    "a window of a switched (AR) mesh step moved a draft counter ({drafts} drafts, {accepted} accepted)"
                ));
            }
            Ok(WindowFact {
                kind: WindowKind::Ar,
                offered: 0,
                accepted: 0,
                emitted,
            })
        }
        (1, 0, 0) => {
            if copy_drafts != 0 || copy_accepted != 0 {
                return Err("a native window moved the copy counters".into());
            }
            Ok(WindowFact {
                kind: WindowKind::Native,
                offered: drafts,
                accepted,
                emitted,
            })
        }
        (0, 0, 1) if !native_mode => Err("a switched (AR) mesh step ran a copy window".into()),
        (0, 0, 1) => {
            if drafts != 0 || accepted != 0 {
                return Err("a copy window moved the native draft counters".into());
            }
            Ok(WindowFact {
                kind: WindowKind::Copy,
                offered: copy_drafts,
                accepted: copy_accepted,
                emitted,
            })
        }
        (0, 1, 0) => {
            if drafts != 0 || accepted != 0 || copy_drafts != 0 || copy_accepted != 0 {
                return Err("a target-only window moved a draft counter".into());
            }
            Ok(WindowFact {
                kind: WindowKind::Ar,
                offered: 0,
                accepted: 0,
                emitted,
            })
        }
        other => Err(format!(
            "a window must move exactly one of (native, ar, copy) by one, got {other:?}"
        )),
    }
}

/// One request of a trial, finished.
#[derive(Clone, Debug)]
pub(super) struct RequestRecord {
    pub trial: usize,
    pub lane: usize,
    pub seed: u32,
    /// The request's `TOKENS` ids, EOS padded.
    pub row: Vec<u32>,
    pub windows: Vec<WindowFact>,
    /// The 1-based window after which the lane's floor retired it.
    pub retired_at: Option<usize>,
}

/// `rows`'s lane-minor assembly: trial by trial, request by request.
pub(super) fn assemble_rows(trials: &[Vec<RequestRecord>]) -> Vec<Vec<u32>> {
    trials
        .iter()
        .flat_map(|trial| trial.iter().map(|record| record.row.clone()))
        .collect()
}

// ---------------------------------------------------------------- timing

/// `first_trial_s` (it pays the kernel loads), then the mean / median / max
/// of the remaining trials: the figure shard sizing wants.
pub(super) fn trial_timing(seconds: &[f64]) -> String {
    let Some((first, rest)) = seconds.split_first() else {
        return "timing: no trials".to_string();
    };
    if rest.is_empty() {
        return format!("timing: first_trial_s={first:.3} (no later trial)");
    }
    let mut sorted = rest.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = rest.iter().sum::<f64>() / rest.len() as f64;
    format!(
        "timing: first_trial_s={first:.3} rest_trials={} rest_mean_s={mean:.3} rest_median_s={:.3} rest_max_s={:.3}",
        rest.len(),
        sorted[sorted.len() / 2],
        sorted[sorted.len() - 1]
    )
}

// -------------------------------------------------------------- evidence

/// Sums of the mesh steps' `MeshStepFacts` (cross-checks of the per-window
/// classification).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct MeshTally {
    pub steps: usize,
    /// Steps by the number of lanes they carried.
    pub by_lanes: BTreeMap<usize, usize>,
    pub lanes: usize,
    pub native: usize,
    pub copy: usize,
    pub ar: usize,
    pub offered: usize,
    pub accepted: usize,
    pub emitted: usize,
}

impl MeshTally {
    pub(super) fn note(&mut self, facts: &MeshStepFacts) {
        self.steps += 1;
        *self.by_lanes.entry(facts.lanes).or_default() += 1;
        self.lanes += facts.lanes;
        self.native += facts.native_windows;
        self.copy += facts.copy_windows;
        self.ar += facts.ar_windows;
        self.offered += facts.offered;
        self.accepted += facts.accepted;
        self.emitted += facts.emitted;
    }
}

/// Whether a request ran both a native and an AR window: it crossed the
/// product native<->AR switch (or, for the reference arm, never could).
pub(super) fn crossed(windows: &[WindowFact]) -> bool {
    windows.iter().any(|w| w.kind == WindowKind::Native)
        && windows.iter().any(|w| w.kind == WindowKind::Ar)
}

/// The counters of one lane arm shard (the schema of
/// [`lane_suite::lane_coverage`]).
#[derive(Debug, Default)]
pub(super) struct LaneEvidence {
    trials: usize,
    requests: usize,
    /// Native windows that verified at least one draft.
    mtp_windows: usize,
    /// Native windows with no draft (the calibration token, a one-token window).
    zero_draft_windows: usize,
    /// Head-free windows (retired lanes; every AR row of the reference arm).
    ar_windows: usize,
    copy_windows: usize,
    copy_point_mass_windows: usize,
    retired_trials: usize,
    calibration_ok_trials: usize,
    no_window_trials: usize,
    /// Requests that ran both a native and an AR window ([`crossed`]).
    crossing_requests: usize,
    /// The most drafts any one window offered.
    max_offered: usize,
    /// Requests by the drafts their first window offered (native first windows).
    first_window_offered: BTreeMap<usize, usize>,
    drafted: usize,
    accepted: usize,
    retire_steps: Vec<usize>,
    per_trial: Vec<Value>,
    texts: String,
    sample_limit: usize,
    pub(super) mesh: MeshTally,
}

impl LaneEvidence {
    pub(super) fn with_samples(sample_limit: usize) -> Self {
        Self {
            sample_limit,
            ..Self::default()
        }
    }

    /// Folds one finished trial: counters from every request's windows, the
    /// per-request rows and decoded text of the first `sample_limit` trials
    /// of the shard.
    pub(super) fn observe_trial(
        &mut self,
        first_trial: usize,
        records: &[RequestRecord],
        decode: &dyn Fn(&[u32]) -> String,
    ) {
        self.trials += 1;
        let sampled = records
            .first()
            .is_some_and(|record| record.trial < first_trial + self.sample_limit);
        for record in records {
            self.requests += 1;
            self.no_window_trials += usize::from(record.windows.is_empty());
            self.calibration_ok_trials += usize::from(record.windows.first().is_some_and(|w| {
                w.kind == WindowKind::Native && w.offered == 0 && w.accepted == 0
            }));
            self.crossing_requests += usize::from(crossed(&record.windows));
            if let Some(first) = record
                .windows
                .first()
                .filter(|w| w.kind == WindowKind::Native)
            {
                *self.first_window_offered.entry(first.offered).or_default() += 1;
            }
            if let Some(step) = record.retired_at {
                self.retired_trials += 1;
                self.retire_steps.push(step);
            }
            for window in &record.windows {
                match window.kind {
                    WindowKind::Native if window.offered > 0 => self.mtp_windows += 1,
                    WindowKind::Native => self.zero_draft_windows += 1,
                    WindowKind::Ar => self.ar_windows += 1,
                    WindowKind::Copy => {
                        // A copy window verifies the one-hot q of its candidates
                        // (`sampled_lane`: `ProposalSource::Copy` -> point
                        // mass), and every lane of this suite is a Leviathan
                        // lane: each copy window is a point-mass window.
                        self.copy_windows += 1;
                        self.copy_point_mass_windows += 1;
                    }
                }
                self.drafted += window.offered;
                self.accepted += window.accepted;
                self.max_offered = self.max_offered.max(window.offered);
            }
            if sampled {
                self.per_trial.push(json!({
                    "trial": record.trial,
                    "lane": record.lane,
                    "seed": record.seed,
                    "ids": record.row,
                    "retired_at_window": record.retired_at,
                    "windows": record.windows.iter().map(WindowFact::json).collect::<Vec<_>>(),
                }));
                self.texts.push_str(&format!(
                    "trial {} lane {} seed {}: {:?}\n",
                    record.trial,
                    record.lane,
                    record.seed,
                    decode(&record.row)
                ));
            }
        }
    }

    fn native_windows(&self) -> usize {
        self.mtp_windows + self.zero_draft_windows
    }

    fn windows(&self) -> usize {
        self.native_windows() + self.copy_windows + self.ar_windows
    }

    /// The windows classified from the lanes' counters against the mesh
    /// steps' own tally (mesh arms only). The mesh facts label a draft-less
    /// non-retired window `ar`; the lane counters label it native, so only
    /// the window total, the drafts and the emitted tokens are compared.
    pub(super) fn check_mesh(&self, emitted: usize) -> Result<(), String> {
        let tally = &self.mesh;
        if tally.lanes != self.windows() {
            return Err(format!(
                "evidence cross-check: the mesh carried {} lane-windows but the lane counters classified {}",
                tally.lanes,
                self.windows()
            ));
        }
        if tally.offered != self.drafted || tally.accepted != self.accepted {
            return Err(format!(
                "evidence cross-check: mesh offered/accepted {}/{} but the lane counters recorded {}/{}",
                tally.offered, tally.accepted, self.drafted, self.accepted
            ));
        }
        if tally.copy != self.copy_windows {
            return Err(format!(
                "evidence cross-check: mesh copy windows {} but the lane counters recorded {}",
                tally.copy, self.copy_windows
            ));
        }
        if tally.emitted != emitted {
            return Err(format!(
                "evidence cross-check: mesh emitted {} tokens but the requests hold {emitted} window tokens",
                tally.emitted
            ));
        }
        Ok(())
    }

    pub(super) fn texts(&self) -> &str {
        &self.texts
    }

    pub(super) fn to_json(
        &self,
        arm: &str,
        placement: lane_suite::Placement,
        range: (usize, usize),
        probe_budget: usize,
    ) -> Value {
        let by_lanes: BTreeMap<String, usize> = self
            .mesh
            .by_lanes
            .iter()
            .map(|(lanes, steps)| (lanes.to_string(), *steps))
            .collect();
        let first_window_offered: BTreeMap<String, usize> = self
            .first_window_offered
            .iter()
            .map(|(offered, requests)| (offered.to_string(), *requests))
            .collect();
        json!({
            "kind": "lane-evidence",
            "arm": arm,
            "executor": placement.executor.as_str(),
            "topology": placement.topology.as_str(),
            "lanes": placement.lanes,
            "lane_k": placement.lane_k,
            "range": [range.0, range.1],
            "probe_budget": probe_budget,
            "trials": self.trials,
            "requests": self.requests,
            "mtp_windows": self.mtp_windows,
            "zero_draft_windows": self.zero_draft_windows,
            "ar_windows": self.ar_windows,
            "retired_trials": self.retired_trials,
            "calibration_ok_trials": self.calibration_ok_trials,
            "no_window_trials": self.no_window_trials,
            "crossing_requests": self.crossing_requests,
            "max_offered": self.max_offered,
            "first_window_offered": first_window_offered,
            "copy_windows": self.copy_windows,
            "copy_point_mass_windows": self.copy_point_mass_windows,
            "steps": self.windows(),
            "drafted": self.drafted,
            "accepted": self.accepted,
            "retire_steps": self.retire_steps,
            "mesh": {
                "steps": self.mesh.steps,
                "steps_by_lanes": by_lanes,
                "lane_windows": self.mesh.lanes,
                "native": self.mesh.native,
                "copy": self.mesh.copy,
                "ar": self.mesh.ar,
                "offered": self.mesh.offered,
                "accepted": self.mesh.accepted,
                "emitted": self.mesh.emitted,
            },
            "sources": {
                "window": "classified per window from the lane's LaneSpecStats delta (lane_spec_stats_for), cross-checked against MeshStepFacts",
                "copy_point_mass": "every copy window of a Leviathan lane verifies the one-hot q (ProposalSource::Copy)",
            },
            "per_trial": self.per_trial,
        })
    }
}

// -------------------------------------------------------------- lane plans

/// One request's rows in a step.
struct RowSpec {
    slot: usize,
    epoch: RequestEpoch,
    tokens: Vec<u32>,
    /// Absolute position of the first row.
    start: usize,
    kind: RequestStepKind,
}

/// One step's plan: per-slot rows for every slot `0..max_lanes` (idle slots
/// empty), requests in slot order, absolute positions.
fn build_plan(max_lanes: usize, mut rows: Vec<RowSpec>) -> BatchStepPlan {
    rows.sort_by_key(|r| r.slot);
    let mut per: Vec<(SlotId, Vec<u32>, usize)> =
        (0..max_lanes).map(|s| (SlotId(s), Vec::new(), 0)).collect();
    for r in &rows {
        per[r.slot] = (SlotId(r.slot), r.tokens.clone(), r.start);
    }
    let triples: Vec<(SlotId, &[u32], usize)> =
        per.iter().map(|(s, t, p)| (*s, t.as_slice(), *p)).collect();
    let mut plan = BatchStepPlan {
        batch: SlotBatch::build(&triples),
        ..Default::default()
    };
    let mut begin = 0;
    for r in &rows {
        plan.requests.push(RequestRows {
            epoch: r.epoch,
            rows: RowRange {
                begin,
                len: r.tokens.len(),
            },
            kind: r.kind,
        });
        match r.kind {
            RequestStepKind::Ar => plan.decode_rows += r.tokens.len(),
            RequestStepKind::Prefill => plan.prefill_rows += r.tokens.len(),
            RequestStepKind::Verify { .. } => plan.verify_rows += r.tokens.len(),
            RequestStepKind::Forced => plan.forced_rows += r.tokens.len(),
        }
        begin += r.tokens.len();
    }
    plan
}

/// provision -> forward -> commit of one executor plan; a failure after
/// provisioning aborts the step.
fn exec_step(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    plan: &BatchStepPlan,
) -> Result<Vec<RequestAdvance>, String> {
    let mut ex = bundle.lane_executor().map_err(|e| e.to_string())?;
    ex.provision_step(gpu, plan)
        .map_err(|m| format!("provision_step failed: {m}"))?;
    let out = match ex.forward_step(gpu, plan) {
        Ok(out) => out,
        Err(m) => {
            ex.abort_step(plan);
            return Err(format!("forward_step failed: {m}"));
        }
    };
    match ex.commit_step(gpu, plan, out) {
        Ok(advances) => Ok(advances),
        Err(m) => {
            ex.abort_step(plan);
            Err(format!("commit_step failed: {m}"))
        }
    }
}

// --------------------------------------------------------------- the load

/// The EP2 target of one lane arm: the product-loaded model, its daemon-side
/// `Gpu`, the staged lane store and the request plumbing.
struct EpLanes {
    model: hipfire_loader::LoadedModel,
    gpu: Gpu,
    tokenizer: Tokenizer,
    template: String,
    kind: ArmKind,
    lanes: usize,
    store_lanes: usize,
    next_tag: u64,
    state: String,
}

/// `load_model`'s admitted route (the oracle's, `fn_mesh_lane_oracle`) with
/// the serve's per-load lane reservation: `lanes` lanes at `row_budget` rows,
/// native-MTP lanes of depth `spec_k` or AR
/// lanes (`spec_k: None`), so the loader's EP expert placement reserves what
/// the staging below will commit. Planner capacity only; numerics are the
/// plain load's.
fn load_ep(
    model: &Path,
    kind: ArmKind,
    store_lanes: usize,
    spec_k: usize,
    gpu: &mut Gpu,
) -> Result<hipfire_loader::LoadedModel, String> {
    use hipfire_loader::admission::{
        admit_source_with_options, legacy_warning, qwen4_kv_adaptive_requested,
        qwen_default_q8_enabled, BatchLoadRequest, KvBackendHints, KvBackendRequest,
        SourceAdmissionOptions,
    };
    let path = model
        .to_str()
        .ok_or_else(|| format!("{} is not valid UTF-8", model.display()))?;
    let (state_request, _) = hw_identity::gdn_test_state_quant()?;
    let spec = hipfire_runtime::loader_api::SpecLoadCfg {
        // The native head is attached only for the arms that decode with it.
        mtp: kind.mesh().then_some(true),
        ..Default::default()
    };
    let mut admission = admit_source_with_options(
        path,
        1,
        1,
        KvBackendRequest::Automatic,
        None,
        gpu.arch.as_str(),
        None,
        "auto",
        None,
        MAX_SEQ,
        KvBackendHints {
            kv_mode: None,
            kv_k: None,
            kv_v: None,
            qwen_default_q8: qwen_default_q8_enabled(),
            kv_adaptive: None,
            deepseek4_heterogeneous: false,
            vmm_runtime_refusal: gpu.vmm_runtime_refusal().as_deref(),
            free_vram_bytes: gpu.device_mem_info().ok().map(|(free, _)| free),
        },
        SourceAdmissionOptions {
            spec,
            kv_adaptive: qwen4_kv_adaptive_requested(None),
            ..Default::default()
        },
    )?;
    admission.batch_request = BatchLoadRequest {
        vmm_batch: true,
        continuous_batch_size: store_lanes,
        row_budget: store_lanes,
        spec_k: kind.mesh().then_some(spec_k),
        // The reservation the serve makes for the copy lane it stages by
        // default; the other arms stage none, so reserve none.
        spec_copy: kind == ArmKind::Copy,
    };
    let warning = legacy_warning(admission.kv_backend, admission.kv_backend_reason.as_deref());
    let loaded = hipfire_loader::load_admitted_with_gemma4_drafter(
        admission,
        path,
        MAX_SEQ,
        None,
        hipfire_config::Deepseek4ComputePlacement::Single,
        None,
        None,
        hipfire_loader::GEMMA4_EAGLE_DRAFT_LEN,
        None,
        None,
        None,
        None,
        Some(state_request),
        1,
        spec,
        gpu,
    );
    if loaded.is_ok() {
        if let Some(warning) = warning {
            eprintln!("{warning}");
        }
    }
    loaded
}

impl EpLanes {
    fn load(model: &Path, arm: &str, lanes: usize, lane_k: usize) -> Result<Self, String> {
        let kind = ArmKind::of(arm)?;
        let store_lanes = store_lanes(kind, lanes);
        // The daemon's host-memory residency preamble (pinned experts stay out
        // of reclaim); before the device initialises.
        if let Some(devices) = hipfire_config::devices::startup_devices() {
            let archs = devices
                .iter()
                .map(|device| device.arch.as_str())
                .collect::<Vec<_>>();
            if residency::keeps_host_memory_out_of_reclaim(hipfire_arch_qwen4::ARCH_ID, &archs) {
                hip_bridge::keep_host_memory_out_of_reclaim(&format!(
                    "Qwen4 model on {}",
                    archs.join(",")
                ));
            }
        }
        let spec_k = hipfire_arch_qwen4::ep::ep_lane_spec_k(MTP_K);
        let ep_degree = hipfire_config::topology::qwen4_ep()?;
        check_ep_env(
            kind,
            lanes,
            hipfire_arch_qwen4::lane_switch::max_active(),
            ep_degree,
        )?;
        check_lane_k(kind, spec_k, lane_k)?;
        let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
        let mut loaded = load_ep(model, kind, store_lanes, spec_k, &mut gpu)?;
        let tokenizer = loaded.tokenizer.take().ok_or("the load has no tokenizer")?;
        let template = loaded
            .chat_template
            .take()
            .ok_or("the artifact has no chat template")?;
        let (state_request, _) = hw_identity::gdn_test_state_quant()?;
        let state = format!("state_quant={state_request} (product loader, ep={EP_DEGREE})");
        let mut this = Self {
            model: loaded,
            gpu,
            tokenizer,
            template,
            kind,
            lanes,
            store_lanes,
            next_tag: 0,
            state,
        };
        this.stage(spec_k)?;
        Ok(this)
    }

    /// Stage the lane store (once per process) and, for the MTP arms, arm the
    /// mesh lane transaction; mark the store sampled.
    fn stage(&mut self, spec_k: usize) -> Result<(), String> {
        let (kind, store_lanes, lanes) = (self.kind, self.store_lanes, self.lanes);
        self.gpu.bind_thread().map_err(|e| e.to_string())?;
        let Self { model, gpu, .. } = self;
        let bundle = model
            .qwen4_mut()
            .ok_or("the loaded model is not a Qwen4 bundle")?;
        let ranks = bundle.ranks();
        if bundle.ep_mesh().is_none() || ranks != EP_DEGREE {
            return Err(format!(
                "the load is not an EP{EP_DEGREE} mesh (ranks {ranks}): set HIPFIRE_TOPOLOGY_EP={EP_DEGREE} and two visible cards"
            ));
        }
        let spec = kind.mesh().then_some(LaneSpecConfig {
            k: spec_k,
            end_of_turn: None,
            spec_lanes: None,
        });
        // The copy arm stages exactly what the serve stages by default; a
        // policy that resolves to no copy lane (or fewer wide lanes than the
        // arm runs) is an error, never a silent copy-off arm.
        let copy = if kind == ArmKind::Copy {
            let copy = hipfire_arch_qwen4::ep::ep_default_copy_config(gpu.arch.as_str())
                .ok_or("lane_copy: the default EP copy lane resolves to off (HIPFIRE_EP_MTP_LANE_COPY, HIPFIRE_MTP_NGRAM_COPY or HIPFIRE_QWEN4_LANE_COPY_LANES=0)")?;
            if copy.wide_lanes < lanes {
                return Err(format!(
                    "lane_copy: {} lanes but only {} copy-wide lanes (HIPFIRE_QWEN4_LANE_COPY_LANES)",
                    lanes, copy.wide_lanes
                ));
            }
            Some(copy)
        } else {
            None
        };
        let before = gpu.device_mem_info().map(|(free, _)| free).unwrap_or(0);
        bundle
            .stage_lanes_with_copy(gpu, store_lanes, store_lanes, spec, copy)
            .map_err(|e| e.to_string())?;
        let after = gpu.device_mem_info().map(|(free, _)| free).unwrap_or(0);
        eprintln!(
            "[lane-suite] staged {store_lanes} lanes ({kind:?}, k={spec_k}) consuming {} MiB on rank 0 ({} MiB free after)",
            before.saturating_sub(after) >> 20,
            after >> 20
        );
        if kind.mesh() {
            bundle.ep_mesh_lanes_arm(gpu, 1)?;
        }
        bundle.set_lanes_sampled(true);
        Ok(())
    }

    /// One trial: `seeds.len()` concurrent requests of `prompt`, in lockstep.
    fn run_trial(
        &mut self,
        trial: usize,
        arm: &str,
        prompt: &[u32],
        sampler: &LaneSamplerSpec,
        tally: &mut MeshTally,
    ) -> Result<Vec<RequestRecord>, String> {
        let (kind, lanes, store_lanes) = (self.kind, self.lanes, self.store_lanes);
        let seeds = (0..lanes)
            .map(|lane| lane_seed(arm, lanes, trial, lane))
            .collect::<Result<Vec<_>, _>>()?;
        let Self {
            model,
            gpu,
            next_tag,
            ..
        } = self;
        let bundle = model
            .qwen4_mut()
            .ok_or("the loaded model is not a Qwen4 bundle")?;
        let eos = bundle.config.eos_token_id;
        let mut requests: Vec<LaneReq> = Vec::with_capacity(lanes);
        let driven = drive_trial(
            bundle,
            gpu,
            kind,
            store_lanes,
            prompt,
            sampler,
            eos,
            trial,
            &seeds,
            &lane_suite::admit_schedule(arm, lanes),
            next_tag,
            &mut requests,
            tally,
        );
        let mut release_error = None;
        for request in requests.iter_mut().filter(|r| !r.released) {
            request.released = true;
            if let Err(error) = bundle.release_lane(&request.epoch) {
                release_error.get_or_insert(format!("release lane {:?}: {error}", request.epoch));
            }
        }
        match (driven, release_error) {
            (Err(error), _) | (Ok(()), Some(error)) => Err(error),
            (Ok(()), None) => Ok(requests
                .into_iter()
                .enumerate()
                .map(|(lane, request)| RequestRecord {
                    trial,
                    lane,
                    seed: request.seed,
                    row: finish_row(&request.ids, eos),
                    windows: request.windows,
                    retired_at: request.retired_at,
                })
                .collect()),
        }
    }
}

/// The request sampler of a lane suite run (neutral penalties only).
#[derive(Clone, Copy, Debug)]
pub(super) struct LaneSamplerSpec {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
}

impl LaneSamplerSpec {
    pub(super) fn from_settings(settings: &DistSettings) -> Result<Self, String> {
        if !(settings.temp.is_finite() && settings.temp > 0.0) {
            return Err(format!(
                "the lane suite samples: {TEMP_ENV}={} must be finite and > 0 (a greedy lane carries no sampler)",
                settings.temp
            ));
        }
        Ok(Self {
            temperature: settings.temp,
            top_p: settings.top_p,
            top_k: settings.top_k,
            min_p: (settings.min_p > 0.0).then_some(settings.min_p),
        })
    }

    pub(super) fn lane(&self, seed: u32) -> LaneSampling {
        LaneSampling {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            seed,
            // `HIPFIRE_SAMPLED_MTP_UNSEEDED=1`: the lane routes as an unseeded sampled
            // request (wide-verify eligible); the RNG seed itself stays fixed.
            seeded: std::env::var("HIPFIRE_SAMPLED_MTP_UNSEEDED").as_deref() != Ok("1"),
        }
    }
}

/// One request's progress inside a trial.
struct LaneReq {
    epoch: RequestEpoch,
    slot: usize,
    seed: u32,
    prefilled: usize,
    ids: Vec<u32>,
    windows: Vec<WindowFact>,
    stats: LaneSpecStats,
    retired_at: Option<usize>,
    done: bool,
    released: bool,
}

/// Admits `seeds` as one group of new requests (lane `i` of the trial is
/// `reqs.len()` at its admission), arming each lane's sampler.
fn admit_group(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    prompt: &[u32],
    sampler: &LaneSamplerSpec,
    seeds: &[u32],
    next_tag: &mut u64,
    reqs: &mut Vec<LaneReq>,
) -> Result<(), String> {
    for &seed in seeds {
        *next_tag += 1;
        let epoch = RequestEpoch {
            request_tag: *next_tag,
            owner_generation: 1,
        };
        let slot = bundle
            .admit_lane(gpu, epoch, prompt.to_vec())
            .map_err(|e| e.to_string())?;
        reqs.push(LaneReq {
            epoch,
            slot,
            seed,
            prefilled: 0,
            ids: Vec::new(),
            windows: Vec::new(),
            stats: LaneSpecStats::default(),
            retired_at: None,
            done: false,
            released: false,
        });
        bundle
            .set_lane_sampling(&epoch, sampler.lane(seed))
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Admit the trial's `seeds` in the groups of `schedule`
/// (`lane_suite::admit_schedule`: every arm but `lane_cross` admits all at
/// once; the first group is admitted at tick 0 and each later one on the tick
/// after the previous groups have forwarded their whole prompt), then tick
/// until every one is done: each tick runs the executor half (every prefilling
/// lane's next tile; the AR arm also one AR row per decoding lane) and, for the
/// MTP arms, one mesh step carrying every lane that was decoding at the tick's
/// start. A request is released the moment it is done.
#[allow(clippy::too_many_arguments)]
fn drive_trial(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    kind: ArmKind,
    store_lanes: usize,
    prompt: &[u32],
    sampler: &LaneSamplerSpec,
    eos: u32,
    trial: usize,
    seeds: &[u32],
    schedule: &[usize],
    next_tag: &mut u64,
    reqs: &mut Vec<LaneReq>,
    tally: &mut MeshTally,
) -> Result<(), String> {
    if schedule.iter().sum::<usize>() != seeds.len() {
        return Err(format!(
            "admission schedule {schedule:?} does not cover the trial's {} seeds",
            seeds.len()
        ));
    }
    let mut admitted_groups = 0;
    let tick_cap = prompt.len() + 16 * TOKENS + 64;
    for _tick in 0..tick_cap {
        while admitted_groups < schedule.len() {
            let prefilled: Vec<usize> = reqs.iter().map(|r| r.prefilled).collect();
            if !lane_suite::next_group_due(&prefilled, prompt.len()) {
                break;
            }
            let first = reqs.len();
            let group = &seeds[first..first + schedule[admitted_groups]];
            admit_group(bundle, gpu, prompt, sampler, group, next_tag, reqs)?;
            admitted_groups += 1;
        }
        if admitted_groups == schedule.len() && reqs.iter().all(|r| r.done) {
            return Ok(());
        }
        // Lanes decoding at the tick's start (a lane finishing its prompt this
        // tick decodes from the next one, as the serve scheduler plans).
        let decoding: Vec<usize> = (0..reqs.len())
            .filter(|&i| !reqs[i].done && reqs[i].prefilled == prompt.len())
            .collect();
        let mut specs = Vec::new();
        for (i, req) in reqs.iter().enumerate().filter(|(_, r)| !r.done) {
            if req.prefilled < prompt.len() {
                let len = bundle
                    .lane_executor()
                    .map_err(|e| e.to_string())?
                    .exact_prefill_chunk_len(&req.epoch)?;
                if len == 0 || req.prefilled + len > prompt.len() {
                    return Err(format!(
                        "request {i}: prefill tile of {len} rows at {}/{}",
                        req.prefilled,
                        prompt.len()
                    ));
                }
                specs.push(RowSpec {
                    slot: req.slot,
                    epoch: req.epoch,
                    tokens: prompt[req.prefilled..req.prefilled + len].to_vec(),
                    start: req.prefilled,
                    kind: RequestStepKind::Prefill,
                });
            } else if kind == ArmKind::Ar {
                let seed = *req
                    .ids
                    .last()
                    .ok_or_else(|| format!("request {i} decodes without a pending seed"))?;
                specs.push(RowSpec {
                    slot: req.slot,
                    epoch: req.epoch,
                    tokens: vec![seed],
                    start: prompt.len() + req.ids.len() - 1,
                    kind: RequestStepKind::Ar,
                });
            }
        }
        if !specs.is_empty() {
            let plan = build_plan(store_lanes, specs);
            let advances = exec_step(bundle, gpu, &plan)?;
            if advances.len() != plan.requests.len() {
                return Err(format!(
                    "{} advances for {} plan requests",
                    advances.len(),
                    plan.requests.len()
                ));
            }
            for (row, advance) in plan.requests.iter().zip(&advances) {
                if advance.epoch != row.epoch {
                    return Err("advance order differs from the plan".into());
                }
                let req = reqs
                    .iter_mut()
                    .find(|r| r.epoch == row.epoch)
                    .ok_or("plan epoch is not one of this trial's requests")?;
                apply_executor_advance(req, row, advance, prompt.len(), eos)?;
            }
        }
        if kind.mesh() && !decoding.is_empty() {
            let requests: Vec<MeshDecodeRequest> = decoding
                .iter()
                .map(|&i| MeshDecodeRequest {
                    epoch: reqs[i].epoch,
                    tokens_left: tokens_left(reqs[i].ids.len()),
                })
                .collect();
            for &i in &decoding {
                reqs[i].stats = bundle
                    .lane_spec_stats_for(&reqs[i].epoch)
                    .ok_or_else(|| format!("request {i} has no MTP lane counters"))?;
            }
            let (decoded, facts) = bundle.ep_mesh_decode_step(gpu, &requests)?;
            tally.note(&facts);
            if decoded.len() != requests.len() {
                return Err(format!(
                    "{} mesh windows for {} requests",
                    decoded.len(),
                    requests.len()
                ));
            }
            for (&i, window) in decoding.iter().zip(&decoded) {
                let advance = &window.advance;
                if advance.epoch != reqs[i].epoch {
                    return Err("mesh window order differs from the request order".into());
                }
                let after = bundle
                    .lane_spec_stats_for(&advance.epoch)
                    .ok_or_else(|| format!("request {i} lost its MTP lane counters"))?;
                let req = &mut reqs[i];
                let fact = classify_window(
                    &req.stats,
                    &after,
                    advance.committed_ids.len(),
                    facts.native_mode,
                )?;
                let expected = prompt.len() + req.ids.len() + advance.committed_ids.len() - 1;
                if advance.committed_position != expected {
                    return Err(format!(
                        "request {i}: mesh window committed position {} expected {expected}",
                        advance.committed_position
                    ));
                }
                req.windows.push(fact);
                if after.retired && req.retired_at.is_none() {
                    req.retired_at = Some(req.windows.len());
                }
                req.stats = after;
                req.ids.extend_from_slice(&advance.committed_ids);
                req.done = request_done(&req.ids, eos);
            }
        }
        for (i, req) in reqs.iter_mut().enumerate() {
            if req.done && !req.released {
                req.released = true;
                bundle
                    .release_lane(&req.epoch)
                    .map_err(|e| format!("release request {i}: {e}"))?;
            }
        }
    }
    Err(format!(
        "trial {trial}: requests still running after {tick_cap} ticks"
    ))
}

/// A prefill tile's or AR row's advance applied to its request.
fn apply_executor_advance(
    req: &mut LaneReq,
    row: &RequestRows,
    advance: &RequestAdvance,
    prompt_len: usize,
    eos: u32,
) -> Result<(), String> {
    match row.kind {
        RequestStepKind::Prefill => {
            req.prefilled += row.rows.len;
            if req.prefilled < prompt_len {
                if !advance.committed_ids.is_empty() {
                    return Err(format!(
                        "a non-final prefill tile committed {:?}",
                        advance.committed_ids
                    ));
                }
                return Ok(());
            }
            let [id] = advance.committed_ids.as_slice() else {
                return Err(format!(
                    "the prompt-completing tile committed {:?}, expected one id",
                    advance.committed_ids
                ));
            };
            req.ids.push(*id);
        }
        RequestStepKind::Ar => {
            let [id] = advance.committed_ids.as_slice() else {
                return Err(format!(
                    "an AR row committed {:?}, expected one id",
                    advance.committed_ids
                ));
            };
            let expected = prompt_len + req.ids.len();
            if advance.committed_position != expected {
                return Err(format!(
                    "AR row committed position {} expected {expected}",
                    advance.committed_position
                ));
            }
            req.ids.push(*id);
            req.windows.push(WindowFact {
                kind: WindowKind::Ar,
                offered: 0,
                accepted: 0,
                emitted: 1,
            });
        }
        other => return Err(format!("unexpected plan row kind {other:?}")),
    }
    req.done = request_done(&req.ids, eos);
    Ok(())
}

// ------------------------------------------------------------------ draw route

/// The sampled-draw route this process resolves, exactly as the product does
/// (`lane::lane_device_sample`, which integration tests cannot see): `device`
/// when the flag is on and the device supports the lane-wide draw launch,
/// `host` otherwise.
fn lane_draw_route(gpu: &Gpu) -> &'static str {
    if gpu.flags.qwen4_lane_device_sample_enabled()
        && hipfire_runtime::lane_draw::lane_draws_supported(gpu)
    {
        "device"
    } else {
        "host"
    }
}

// ----------------------------------------------------------------- the arm

/// One EP2 lane arm shard: load, stage once, run `range`'s trials, write the
/// shard's ids / config / receipt / evidence / text. `config` and `ids_path`
/// were validated by the caller.
pub(super) fn run_ep_arm(
    arm: &str,
    model: &Path,
    dir: &Path,
    placement: lane_suite::Placement,
    settings: &DistSettings,
    range: (usize, usize),
    config: &Value,
    stem: &str,
) -> Result<(), String> {
    let ids_path = dir.join(format!("{stem}.ids.json"));
    let sampler = LaneSamplerSpec::from_settings(settings)?;
    let lanes = placement.lanes;
    let mut run = EpLanes::load(model, arm, lanes, placement.lane_k)?;
    // The route is resolved from the loaded GPU, so it joins the config here;
    // ids already on disk must have been produced under the same route.
    let route = lane_draw_route(&run.gpu);
    let config = lane_suite::with_draw_route(config.clone(), route)?;
    if ids_path.exists() {
        check_config(&ids_path, &config)?;
    }
    let rendered = render_prompt(
        &run.tokenizer,
        &run.template,
        &settings.fixture.text(),
        false,
    )?;
    // The copy fixture primes the assistant turn so the copy pool proposes on
    // the first window (its CPU pool check is in `lane_copy_prompt`); every arm
    // of the cohort, the AR reference included, runs the same prompt.
    let prompt = if settings.fixture == Fixture::Copy {
        lane_copy_prompt(&run.tokenizer, rendered)?
    } else {
        settings
            .fixture
            .check_prompt(rendered.len(), &settings.pen)?;
        rendered
    };
    let env = own_env(&lane_run::LANE_CONTROL_ENV);
    println!(
        "ARM {arm} arch={} lane_draw_route={route} prompt_tokens={} trials {}..{} lanes={lanes} temp={} top_p={} top_k={:?} min_p={} fixture={} state={} route_class={}",
        run.gpu.arch,
        prompt.len(),
        range.0,
        range.1,
        settings.temp,
        settings.top_p,
        settings.top_k,
        settings.min_p,
        settings.fixture.name(),
        run.state,
        hw_identity::route_class(&env)
    );
    let budget = floor_probe_budget()?;
    let mut evidence = LaneEvidence::with_samples(EVIDENCE_SAMPLES);
    let mut trials: Vec<Vec<RequestRecord>> = Vec::with_capacity(range.1 - range.0);
    let started = std::time::Instant::now();
    let mut trial_secs = Vec::with_capacity(range.1 - range.0);
    for trial in range.0..range.1 {
        let trial_started = std::time::Instant::now();
        let records = run.run_trial(trial, arm, &prompt, &sampler, &mut evidence.mesh)?;
        trial_secs.push(trial_started.elapsed().as_secs_f64());
        let tokenizer = &run.tokenizer;
        evidence.observe_trial(range.0, &records, &|ids| tokenizer.decode(ids));
        trials.push(records);
    }
    let wall = started.elapsed().as_secs_f64();
    if run.kind.mesh() {
        let emitted = trials
            .iter()
            .flatten()
            .flat_map(|record| record.windows.iter())
            .map(|window| window.emitted)
            .sum();
        evidence.check_mesh(emitted)?;
    }
    // Only a forced floor-off route replays byte for byte; the adaptive
    // route's choices depend on wall-clock prices, and the AR reference draws
    // from the lane RNG alone.
    if run.kind.mesh() && hw_identity::forced_floor_off(&env) {
        let replays = REPLAYS.min(range.1 - range.0);
        let mut scratch = MeshTally::default();
        for local in 0..replays {
            let trial = range.0 + local;
            let replay = run.run_trial(trial, arm, &prompt, &sampler, &mut scratch)?;
            for (first, again) in trials[local].iter().zip(&replay) {
                if first.row != again.row {
                    return Err(format!(
                        "trial {trial} lane {} (seed {}) replayed {:?}, first run {:?}",
                        first.lane, first.seed, again.row, first.row
                    ));
                }
            }
        }
        println!("ARM {arm} replayed {replays} trials identically");
    } else {
        println!(
            "ARM {arm} replay not asserted (route class {})",
            hw_identity::route_class(&env)
        );
    }
    let rows = assemble_rows(&trials);
    if rows.len() != (range.1 - range.0) * lanes {
        return Err(format!(
            "{} rows assembled for {} trials x {lanes} lanes",
            rows.len(),
            range.1 - range.0
        ));
    }
    println!(
        "ARM {arm} trials {}..{} lanes={lanes} in {wall:.1}s ({:.3} s/trial); first sample: {:?}",
        range.0,
        range.1,
        wall / (range.1 - range.0) as f64,
        run.tokenizer.decode(&rows[0])
    );
    println!("ARM {arm} {}", trial_timing(&trial_secs));
    println!(
        "ARM {arm} mesh: steps={} by_lanes={:?} lane_windows={} native={} copy={} ar={} offered={} accepted={}",
        evidence.mesh.steps,
        evidence.mesh.by_lanes,
        evidence.mesh.lanes,
        evidence.mesh.native,
        evidence.mesh.copy,
        evidence.mesh.ar,
        evidence.mesh.offered,
        evidence.mesh.accepted
    );
    println!(
        "ARM {arm} lane_k={} max_offered={} first_window_offered={:?} crossing_requests={} of {} requests (a request that ran both a native and an AR window)",
        placement.lane_k,
        evidence.max_offered,
        evidence.first_window_offered,
        evidence.crossing_requests,
        trials.iter().map(Vec::len).sum::<usize>()
    );
    debug_assert!(rows.iter().all(|row| row.len() == TOKENS));
    fs::write(
        &ids_path,
        serde_json::to_vec(&rows).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    write_config(&ids_path, &config)?;
    write_receipt(&ids_path, &hw_identity::receipt(&run.gpu, model))?;
    write_json(
        &dir.join(format!("{stem}.evidence.json")),
        &evidence.to_json(arm, placement, range, budget),
    )?;
    let text_path = dir.join(format!("{stem}.text.txt"));
    fs::write(&text_path, evidence.texts()).map_err(|e| format!("{}: {e}", text_path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn stats(
        windows: u64,
        ar: u64,
        copy: u64,
        drafts: u64,
        accepted: u64,
        copy_drafts: u64,
        copy_accepted: u64,
    ) -> LaneSpecStats {
        LaneSpecStats {
            windows,
            ar_windows: ar,
            drafts,
            accepted,
            copy_windows: copy,
            copy_drafts,
            copy_accepted,
            ..LaneSpecStats::default()
        }
    }

    fn placement(lanes: usize) -> lane_suite::Placement {
        lane_suite::Placement {
            executor: lane_suite::Executor::Lanes,
            topology: lane_suite::Topology::Ep2,
            lanes,
            lane_k: 2,
        }
    }

    fn native(offered: usize, accepted: usize, emitted: usize) -> WindowFact {
        WindowFact {
            kind: WindowKind::Native,
            offered,
            accepted,
            emitted,
        }
    }

    fn record(trial: usize, lane: usize, windows: Vec<WindowFact>) -> RequestRecord {
        RequestRecord {
            trial,
            lane,
            seed: 7,
            row: vec![1, 2, 3, 4],
            windows,
            retired_at: None,
        }
    }

    #[test]
    fn arm_kinds_cover_the_declared_arms() {
        for arm in lane_suite::EP2_ARMS {
            assert!(ArmKind::of(arm).is_ok(), "{arm}");
        }
        assert_eq!(ArmKind::of("lane_ar"), Ok(ArmKind::Ar));
        assert_eq!(ArmKind::of("lane_adaptive"), Ok(ArmKind::Mtp));
        assert_eq!(ArmKind::of("lane_cross"), Ok(ArmKind::Cross));
        assert!(ArmKind::Cross.mesh() && ArmKind::Mtp.mesh() && !ArmKind::Ar.mesh());
        assert_eq!(ArmKind::of("lane_copy"), Ok(ArmKind::Copy));
        assert!(ArmKind::Copy.mesh());
        for arm in lane_suite::EP2_COPY_ARMS {
            assert!(ArmKind::of(arm).is_ok(), "{arm}");
        }
        // The cohort's lane bound is the serve's copy-lane step cap.
        assert_eq!(
            lane_suite::EP2_COPY_LANES,
            hipfire_arch_qwen4::ep::COPY_WIDE_LANES
        );
        assert!(ArmKind::of("ar").is_err());
        assert!(ArmKind::of("lane_interleaved").is_err());
    }

    #[test]
    fn the_copy_arm_runs_at_the_serves_copy_step_cap_or_fewer_lanes() {
        for lanes in 1..=lane_suite::EP2_COPY_LANES {
            assert!(check_ep_env(ArmKind::Copy, lanes, 8, 2).is_ok(), "{lanes}");
        }
        let wide = check_ep_env(ArmKind::Copy, lane_suite::EP2_COPY_LANES + 1, 8, 2).unwrap_err();
        assert!(wide.contains("lane_copy"), "{wide}");
        assert!(check_ep_env(ArmKind::Copy, 2, 1, 2).is_err());
        assert!(check_ep_env(ArmKind::Copy, 2, 8, 1).is_err());
        assert!(check_lane_k(ArmKind::Copy, 3, 2).is_err());
        assert!(check_lane_k(ArmKind::Copy, 2, 2).is_ok());
    }

    #[test]
    fn a_one_request_arm_still_runs_on_a_two_slot_store() {
        assert_eq!(store_lanes(ArmKind::Mtp, 1), 2);
        assert_eq!(store_lanes(ArmKind::Mtp, 8), 8);
        // The copy arm's store holds the serve's eight lanes at any lane count.
        assert_eq!(store_lanes(ArmKind::Copy, 1), 8);
        assert_eq!(store_lanes(ArmKind::Copy, 2), 8);
    }

    #[test]
    fn the_env_check_pins_the_topology_and_native_windows() {
        let mtp = ArmKind::Mtp;
        assert!(check_ep_env(mtp, 8, 8, 2).is_ok());
        assert!(check_ep_env(mtp, 1, 8, 2).is_ok());
        assert!(
            check_ep_env(ArmKind::Ar, 8, 1, 2).is_ok(),
            "the AR store never switches windows"
        );
        let low = check_ep_env(mtp, 8, 4, 2).unwrap_err();
        assert!(low.contains("HIPFIRE_EP_MTP_LANE_MAX_ACTIVE=4"), "{low}");
        assert!(check_ep_env(mtp, 8, 8, 1)
            .unwrap_err()
            .contains("topology.ep"));
    }

    #[test]
    fn the_lane_depth_must_be_the_cohorts_for_the_native_arms_only() {
        for lane_k in lane_suite::LANE_K_VALUES {
            for kind in [ArmKind::Mtp, ArmKind::Cross] {
                assert!(check_lane_k(kind, lane_k, lane_k).is_ok());
                let other = if lane_k == 2 { 3 } else { 2 };
                let error = check_lane_k(kind, other, lane_k).unwrap_err();
                assert!(
                    error.contains("HIPFIRE_EP_MTP_LANE_K")
                        && error.contains("HIPFIRE_SAMPLED_MTP_LANE_K"),
                    "{error}"
                );
            }
            // The AR reference drafts nothing and pins nothing.
            assert!(check_lane_k(ArmKind::Ar, 1, lane_k).is_ok());
        }
        assert!(check_lane_k(ArmKind::Mtp, 1, 2).is_err());
    }

    #[test]
    fn every_suite_lane_count_passes_the_fixed_count_env_check() {
        for lanes in lane_suite::LANE_COUNTS {
            // The fixed-count pin is the largest lane count: native windows at every N.
            assert!(check_ep_env(ArmKind::Mtp, lanes, 8, 2).is_ok(), "{lanes}");
            // A store is never smaller than the serve's two-lane minimum.
            assert_eq!(store_lanes(ArmKind::Mtp, lanes), lanes.max(2));
            // Native windows need at least as many active slots as lanes.
            let below = check_ep_env(ArmKind::Mtp, lanes, lanes - 1, 2);
            assert!(below.is_err(), "{lanes}");
        }
    }

    #[test]
    fn the_crossing_env_check_pins_the_crossing_max_active_and_eight_lanes() {
        use hipfire_arch_qwen4::lane_switch::DEFAULT_MAX_ACTIVE;
        let cross = ArmKind::Cross;
        // The crossing arm pins the previous shipped switch (3), below the native-C8 default,
        // so eight lanes still cross it.
        assert!(lane_suite::CROSS_MAX_ACTIVE_N < DEFAULT_MAX_ACTIVE);
        assert!(check_ep_env(cross, 8, lane_suite::CROSS_MAX_ACTIVE_N, 2).is_ok());
        // The fixed-count pin (8) would never switch: refused for the crossing arm.
        let wide = check_ep_env(cross, 8, 8, 2).unwrap_err();
        assert!(wide.contains("HIPFIRE_EP_MTP_LANE_MAX_ACTIVE=8"), "{wide}");
        for max_active in [1, 2, 4, 5, 6, 7] {
            assert!(
                check_ep_env(cross, 8, max_active, 2).is_err(),
                "{max_active}"
            );
        }
        for lanes in 1..=7 {
            let error = check_ep_env(cross, lanes, lane_suite::CROSS_MAX_ACTIVE_N, 2).unwrap_err();
            assert!(
                error.contains("lane_cross runs 8 lanes"),
                "{lanes}: {error}"
            );
        }
        assert!(check_ep_env(cross, 8, lane_suite::CROSS_MAX_ACTIVE_N, 1)
            .unwrap_err()
            .contains("topology.ep"));
    }

    #[test]
    fn lane_seeds_are_request_functions_distinct_per_arm_trial_and_lane() {
        let mut seen = BTreeSet::new();
        for arm in lane_suite::EP2_ARMS {
            for trial in 0..50 {
                for lane in 0..8 {
                    let seed = lane_seed(arm, 8, trial, lane).unwrap();
                    assert!(seen.insert((arm, seed)), "{arm} trial {trial} lane {lane}");
                }
            }
        }
        // The seed is the global request index's SplitMix64, truncated.
        assert_eq!(
            lane_seed("lane_batched", 8, 3, 5).unwrap(),
            trial_seed(lane_suite::lane_salt("lane_batched").unwrap(), 3 * 8 + 5) as u32
        );
        // Request 3*8+5 of an 8-lane run is request 29 of a 1-lane run of the
        // same trial index sequence: only the global request index matters.
        assert_eq!(
            lane_seed("lane_ar", 8, 3, 5).unwrap(),
            lane_seed("lane_ar", 1, 29, 0).unwrap()
        );
        assert!(lane_seed("ar", 8, 0, 0).is_err());
    }

    #[test]
    fn rows_cut_after_eos_and_pad_to_the_token_count() {
        let eos = 9;
        assert_eq!(finish_row(&[1, 2, 3, 4], eos), vec![1, 2, 3, 4]);
        assert_eq!(finish_row(&[1, 2, 3, 4, 5], eos), vec![1, 2, 3, 4]);
        assert_eq!(finish_row(&[1, 9, 3, 4], eos), vec![1, 9, 9, 9]);
        assert_eq!(finish_row(&[9], eos), vec![9, 9, 9, 9]);
        assert_eq!(finish_row(&[1, 2], eos), vec![1, 2, 9, 9]);
        assert_eq!(finish_row(&[1, 2, 3, 4, 9], eos), vec![1, 2, 3, 4]);
    }

    #[test]
    fn the_mesh_gets_the_tokens_it_may_still_emit_with_the_pending_seed() {
        assert_eq!(tokens_left(1), TOKENS);
        assert_eq!(tokens_left(TOKENS - 1), 2);
        assert_eq!(tokens_left(TOKENS), 1);
        assert_eq!(tokens_left(TOKENS + 3), 1);
    }

    #[test]
    fn trial_timing_separates_the_first_trial_from_the_steady_state() {
        assert_eq!(trial_timing(&[]), "timing: no trials");
        assert!(trial_timing(&[2.0]).contains("first_trial_s=2.000 (no later trial)"));
        let line = trial_timing(&[9.0, 1.0, 3.0, 2.0]);
        assert!(line.contains("first_trial_s=9.000"), "{line}");
        assert!(
            line.contains("rest_trials=3 rest_mean_s=2.000 rest_median_s=2.000 rest_max_s=3.000"),
            "{line}"
        );
    }

    #[test]
    fn a_request_is_done_at_the_token_count_or_after_eos() {
        assert!(!request_done(&[1, 2, 3], 9));
        assert!(request_done(&[1, 2, 3, 4], 9));
        assert!(request_done(&[1, 2, 3, 4, 5], 9));
        assert!(request_done(&[1, 9], 9));
        assert!(
            request_done(&[1, 9, 3], 9),
            "an EOS inside a window ends the request"
        );
        assert!(!request_done(&[], 9));
    }

    #[test]
    fn windows_classify_from_the_lane_counter_deltas() {
        let zero = stats(0, 0, 0, 0, 0, 0, 0);
        assert_eq!(
            classify_window(&zero, &stats(1, 0, 0, 3, 2, 0, 0), 3, true).unwrap(),
            native(3, 2, 3)
        );
        // A draft-less window of a native step: native, no drafts.
        assert_eq!(
            classify_window(&zero, &stats(1, 0, 0, 0, 0, 0, 0), 1, true).unwrap(),
            native(0, 0, 1)
        );
        assert_eq!(
            classify_window(&zero, &stats(0, 0, 1, 0, 0, 3, 3), 4, true).unwrap(),
            WindowFact {
                kind: WindowKind::Copy,
                offered: 3,
                accepted: 3,
                emitted: 4
            }
        );
        let ar = WindowFact {
            kind: WindowKind::Ar,
            offered: 0,
            accepted: 0,
            emitted: 1,
        };
        let before = stats(2, 0, 0, 5, 3, 0, 0);
        assert_eq!(
            classify_window(&before, &stats(2, 1, 0, 5, 3, 0, 0), 1, true).unwrap(),
            ar
        );
        // The product switch above max_active (`native_mode` false): the
        // lane counters book the one-token `ArSynced` window as a draft-less
        // native window; it is an AR window, as is a retired lane's.
        assert_eq!(
            classify_window(&before, &stats(3, 0, 0, 5, 3, 0, 0), 1, false).unwrap(),
            ar
        );
        assert_eq!(
            classify_window(&before, &stats(2, 1, 0, 5, 3, 0, 0), 1, false).unwrap(),
            ar
        );
        // A switched step never drafts or copies.
        assert!(classify_window(&before, &stats(3, 0, 0, 7, 4, 0, 0), 3, false).is_err());
        assert!(classify_window(&before, &stats(2, 0, 1, 5, 3, 2, 1), 2, false).is_err());
    }

    #[test]
    fn a_window_that_is_not_exactly_one_kind_is_an_error() {
        let zero = stats(0, 0, 0, 0, 0, 0, 0);
        for native_mode in [true, false] {
            let c = |before: &LaneSpecStats, after: &LaneSpecStats| {
                classify_window(before, after, 1, native_mode)
            };
            assert!(c(&zero, &zero).is_err(), "no window moved");
            assert!(c(&zero, &stats(1, 1, 0, 0, 0, 0, 0)).is_err());
            assert!(c(&zero, &stats(2, 0, 0, 0, 0, 0, 0)).is_err());
            assert!(c(&stats(1, 0, 0, 0, 0, 0, 0), &zero).is_err(), "backwards");
            // A native window may not move the copy counters, nor the reverse.
            assert!(c(&zero, &stats(1, 0, 0, 0, 0, 1, 1)).is_err());
            assert!(c(&zero, &stats(0, 0, 1, 1, 0, 1, 0)).is_err());
            assert!(c(&zero, &stats(0, 1, 0, 1, 0, 0, 0)).is_err());
            let mut forced = zero;
            forced.forced_windows = 1;
            forced.windows = 1;
            assert!(c(&zero, &forced).unwrap_err().contains("forced"));
        }
    }

    #[test]
    fn rows_assemble_lane_minor_trial_by_trial() {
        let mut trials = Vec::new();
        for trial in 0..3 {
            trials.push(
                (0..2)
                    .map(|lane| RequestRecord {
                        row: vec![trial as u32, lane as u32],
                        ..record(trial, lane, Vec::new())
                    })
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(
            assemble_rows(&trials),
            vec![
                vec![0, 0],
                vec![0, 1],
                vec![1, 0],
                vec![1, 1],
                vec![2, 0],
                vec![2, 1]
            ]
        );
        // The shard planner's per-lane split recovers each lane's stream.
        let lanes = lane_suite::per_lane_rows(&assemble_rows(&trials), 2).unwrap();
        assert_eq!(lanes[1], vec![vec![0, 1], vec![1, 1], vec![2, 1]]);
    }

    fn adaptive_records() -> Vec<RequestRecord> {
        (0..2)
            .map(|lane| record(0, lane, vec![native(0, 0, 1), native(3, 2, 3)]))
            .collect()
    }

    #[test]
    fn evidence_counts_windows_by_kind_and_meets_the_coverage_schema() {
        let mut evidence = LaneEvidence::with_samples(1);
        evidence.observe_trial(0, &adaptive_records(), &|ids| format!("{ids:?}"));
        let json = evidence.to_json("lane_adaptive", placement(2), (0, 1), 3);
        assert_eq!(json["trials"], 1);
        assert_eq!(json["requests"], 2);
        assert_eq!(json["mtp_windows"], 2);
        assert_eq!(json["zero_draft_windows"], 2);
        assert_eq!(json["calibration_ok_trials"], 2);
        assert_eq!(json["drafted"], 6);
        assert_eq!(json["accepted"], 4);
        assert_eq!(json["steps"], 4);
        assert_eq!(json["probe_budget"], 3);
        assert_eq!(json["per_trial"].as_array().unwrap().len(), 2);
        assert!(evidence.texts().contains("trial 0 lane 1"));
        assert!(
            lane_suite::lane_coverage(lane_suite::Topology::Ep2, "lane_adaptive", 2, 1, &json)
                .is_empty()
        );
        // The batched arm passes on the same counters; the AR reference and the
        // copy arm do not.
        assert!(
            lane_suite::lane_coverage(lane_suite::Topology::Ep2, "lane_batched", 2, 1, &json)
                .is_empty()
        );
        assert!(
            !lane_suite::lane_coverage(lane_suite::Topology::Ep2, "lane_ar", 2, 1, &json)
                .is_empty()
        );
        assert!(
            !lane_suite::lane_coverage(lane_suite::Topology::Ep2, "lane_copy", 2, 1, &json)
                .is_empty()
        );
    }

    #[test]
    fn the_reference_arm_records_only_ar_windows() {
        let ar = WindowFact {
            kind: WindowKind::Ar,
            offered: 0,
            accepted: 0,
            emitted: 1,
        };
        let records: Vec<_> = (0..2).map(|lane| record(0, lane, vec![ar; 3])).collect();
        let mut evidence = LaneEvidence::with_samples(0);
        evidence.observe_trial(0, &records, &|_| String::new());
        let json = evidence.to_json("lane_ar", placement(2), (0, 1), 3);
        assert_eq!(json["ar_windows"], 6);
        assert_eq!(json["mtp_windows"], 0);
        assert!(
            lane_suite::lane_coverage(lane_suite::Topology::Ep2, "lane_ar", 2, 1, &json).is_empty()
        );
        assert!(
            !lane_suite::lane_coverage(lane_suite::Topology::Ep2, "lane_batched", 2, 1, &json)
                .is_empty()
        );
        assert!(
            json["per_trial"].as_array().unwrap().is_empty(),
            "no samples requested"
        );
    }

    #[test]
    fn copy_windows_and_retirement_reach_the_evidence() {
        let copy = WindowFact {
            kind: WindowKind::Copy,
            offered: 3,
            accepted: 2,
            emitted: 3,
        };
        let mut request = record(0, 0, vec![native(0, 0, 1), copy, copy]);
        request.retired_at = Some(2);
        let mut evidence = LaneEvidence::with_samples(0);
        evidence.observe_trial(0, &[request], &|_| String::new());
        let json = evidence.to_json("lane_copy", placement(1), (0, 1), 3);
        assert_eq!(json["copy_windows"], 2);
        assert_eq!(json["copy_point_mass_windows"], 2);
        assert_eq!(json["retired_trials"], 1);
        assert_eq!(json["retire_steps"], serde_json::json!([2]));
        assert!(
            lane_suite::lane_coverage(lane_suite::Topology::Ep2, "lane_copy", 1, 1, &json)
                .is_empty()
        );
    }

    #[test]
    fn a_request_without_windows_is_a_no_window_trial() {
        let mut evidence = LaneEvidence::with_samples(0);
        evidence.observe_trial(0, &[record(0, 0, Vec::new())], &|_| String::new());
        let json = evidence.to_json("lane_adaptive", placement(1), (0, 1), 3);
        assert_eq!(json["no_window_trials"], 1);
        assert_eq!(json["calibration_ok_trials"], 0);
        // calibrated + no_window == requests: the adaptive rule is met.
        assert!(!lane_suite::lane_coverage(
            lane_suite::Topology::Ep2,
            "lane_adaptive",
            1,
            1,
            &json
        )
        .iter()
        .any(|line| line.contains("calibration token")));
    }

    #[test]
    fn evidence_shards_merge_under_the_suite_rules() {
        let shard = |first: usize| {
            let mut evidence = LaneEvidence::with_samples(1);
            let records: Vec<_> = adaptive_records()
                .into_iter()
                .map(|r| RequestRecord { trial: first, ..r })
                .collect();
            evidence.observe_trial(first, &records, &|ids| format!("{ids:?}"));
            evidence.to_json("lane_adaptive", placement(2), (first, first + 1), 3)
        };
        let merged = lane_suite::merge_evidence(&[shard(0), shard(1)]).unwrap();
        assert_eq!(merged["trials"], 2);
        assert_eq!(merged["requests"], 4);
        assert_eq!(merged["per_trial"].as_array().unwrap().len(), 4);
        assert!(lane_suite::lane_coverage(
            lane_suite::Topology::Ep2,
            "lane_adaptive",
            2,
            2,
            &merged
        )
        .is_empty());
    }

    #[test]
    fn the_mesh_cross_check_compares_windows_drafts_copy_and_emitted() {
        let mut evidence = LaneEvidence::with_samples(0);
        evidence.observe_trial(0, &adaptive_records(), &|_| String::new());
        // Two lanes, two steps: the first (calibration) step is `ar` in the
        // mesh facts, the second native.
        evidence.mesh.note(&MeshStepFacts {
            lanes: 2,
            native_windows: 0,
            copy_windows: 0,
            ar_windows: 2,
            offered: 0,
            accepted: 0,
            emitted: 2,
            native_mode: true,
        });
        evidence.mesh.note(&MeshStepFacts {
            lanes: 2,
            native_windows: 2,
            copy_windows: 0,
            ar_windows: 0,
            offered: 6,
            accepted: 4,
            emitted: 6,
            native_mode: true,
        });
        assert_eq!(evidence.check_mesh(8), Ok(()));
        assert!(evidence.check_mesh(9).unwrap_err().contains("emitted"));
        evidence.mesh.offered += 1;
        assert!(evidence.check_mesh(8).unwrap_err().contains("offered"));
        evidence.mesh.offered -= 1;
        evidence.mesh.lanes += 1;
        assert!(evidence.check_mesh(8).unwrap_err().contains("lane-windows"));
        evidence.mesh.lanes -= 1;
        evidence.mesh.copy += 1;
        assert!(evidence.check_mesh(8).unwrap_err().contains("copy"));
    }

    #[test]
    fn the_mesh_tally_histograms_the_lanes_of_each_step() {
        let mut tally = MeshTally::default();
        for lanes in [8, 8, 5, 1] {
            tally.note(&MeshStepFacts {
                lanes,
                ..MeshStepFacts::default()
            });
        }
        assert_eq!(tally.steps, 4);
        assert_eq!(tally.lanes, 22);
        assert_eq!(tally.by_lanes[&8], 2);
        assert_eq!(tally.by_lanes[&1], 1);
    }

    fn ar_window() -> WindowFact {
        WindowFact {
            kind: WindowKind::Ar,
            offered: 0,
            accepted: 0,
            emitted: 1,
        }
    }

    #[test]
    fn a_request_crosses_when_it_ran_both_a_native_and_an_ar_window() {
        assert!(crossed(&[native(3, 2, 3), ar_window()]));
        assert!(crossed(&[ar_window(), native(0, 0, 1)]));
        assert!(!crossed(&[native(3, 2, 3), native(3, 1, 2)]));
        assert!(!crossed(&[ar_window(), ar_window()]));
        assert!(!crossed(&[]));
    }

    /// One 8-lane crossing trial: lanes 0..crossed_lanes ran a native window
    /// then an AR one, the others native windows only.
    fn cross_trial(trial: usize, crossed_lanes: usize) -> Vec<RequestRecord> {
        (0..8)
            .map(|lane| {
                let windows = if lane < crossed_lanes {
                    vec![native(3, 2, 3), ar_window()]
                } else {
                    vec![native(3, 3, 4)]
                };
                record(trial, lane, windows)
            })
            .collect()
    }

    fn note_steps(evidence: &mut LaneEvidence, occupancies: &[usize]) {
        for &lanes in occupancies {
            evidence.mesh.note(&MeshStepFacts {
                lanes,
                ..MeshStepFacts::default()
            });
        }
    }

    #[test]
    fn crossing_evidence_counts_crossed_requests_and_the_occupancy_histogram() {
        let mut evidence = LaneEvidence::with_samples(0);
        evidence.observe_trial(0, &cross_trial(0, 5), &|_| String::new());
        note_steps(&mut evidence, &[3, 8, 8, 4, 5]);
        let json = evidence.to_json("lane_cross", placement(8), (0, 1), 3);
        assert_eq!(json["crossing_requests"], 5);
        assert_eq!(json["requests"], 8);
        assert_eq!(
            json["mesh"]["steps_by_lanes"],
            serde_json::json!({"3": 1, "4": 1, "5": 1, "8": 2})
        );
        let failures =
            lane_suite::lane_coverage(lane_suite::Topology::Ep2, "lane_cross", 8, 1, &json);
        assert!(failures.is_empty(), "{failures:?}");
        // Every other arm records the counter too (the reference never crosses).
        let mut reference = LaneEvidence::with_samples(0);
        let records: Vec<_> = (0..2)
            .map(|lane| record(0, lane, vec![ar_window(); 3]))
            .collect();
        reference.observe_trial(0, &records, &|_| String::new());
        assert_eq!(
            reference.to_json("lane_ar", placement(2), (0, 1), 3)["crossing_requests"],
            0
        );
    }

    #[test]
    fn a_run_that_never_crossed_has_no_crossing_verdict() {
        // No request ran both kinds of window.
        let mut evidence = LaneEvidence::with_samples(0);
        evidence.observe_trial(0, &cross_trial(0, 0), &|_| String::new());
        // AR windows exist elsewhere, so only the crossing rule is violated.
        evidence.ar_windows += 1;
        note_steps(&mut evidence, &[3, 8]);
        let json = evidence.to_json("lane_cross", placement(8), (0, 1), 3);
        assert_eq!(json["crossing_requests"], 0);
        let failures =
            lane_suite::lane_coverage(lane_suite::Topology::Ep2, "lane_cross", 8, 1, &json)
                .concat();
        assert!(failures.contains("crossing_requests=0"), "{failures}");

        // Steps that never had occupancy <= 3 / >= 5 (4 is the hold case).
        for (occupancies, needle) in [
            (&[8, 8][..], "<= 3"),
            (&[3, 4][..], ">= 5"),
            (&[][..], "<= 3"),
        ] {
            let mut evidence = LaneEvidence::with_samples(0);
            evidence.observe_trial(0, &cross_trial(0, 5), &|_| String::new());
            note_steps(&mut evidence, occupancies);
            let json = evidence.to_json("lane_cross", placement(8), (0, 1), 3);
            let failures =
                lane_suite::lane_coverage(lane_suite::Topology::Ep2, "lane_cross", 8, 1, &json)
                    .concat();
            assert!(failures.contains(needle), "{occupancies:?}: {failures}");
        }
    }

    #[test]
    fn crossing_evidence_shards_merge_and_still_cover() {
        let shard = |first: usize| {
            let mut evidence = LaneEvidence::with_samples(0);
            evidence.observe_trial(first, &cross_trial(first, 3), &|_| String::new());
            // Shard 0 sees only the native side, shard 1 only the AR side: the
            // merged histogram covers both.
            note_steps(&mut evidence, if first == 0 { &[3, 3] } else { &[8, 5] });
            evidence.to_json("lane_cross", placement(8), (first, first + 1), 3)
        };
        let (a, b) = (shard(0), shard(1));
        assert!(
            !lane_suite::lane_coverage(lane_suite::Topology::Ep2, "lane_cross", 8, 1, &a)
                .is_empty(),
            "one shard alone never reached the AR side"
        );
        let merged = lane_suite::merge_evidence(&[a, b]).unwrap();
        assert_eq!(merged["crossing_requests"], 6);
        assert_eq!(
            merged["mesh"]["steps_by_lanes"],
            serde_json::json!({"3": 2, "5": 1, "8": 1})
        );
        let failures =
            lane_suite::lane_coverage(lane_suite::Topology::Ep2, "lane_cross", 8, 2, &merged);
        assert!(failures.is_empty(), "{failures:?}");
    }

    fn depth_evidence(offered: &[usize], lane_k: usize) -> Value {
        let records: Vec<_> = offered
            .iter()
            .enumerate()
            .map(|(lane, &drafts)| record(0, lane, vec![native(drafts, drafts, drafts + 1)]))
            .collect();
        let mut evidence = LaneEvidence::with_samples(0);
        evidence.observe_trial(0, &records, &|_| String::new());
        let placement = lane_suite::Placement {
            lane_k,
            ..placement(offered.len())
        };
        evidence.to_json("lane_batched", placement, (0, 1), 3)
    }

    #[test]
    fn the_evidence_records_the_lane_depth_and_what_the_windows_offered() {
        let json = depth_evidence(&[3, 3, 1], 3);
        assert_eq!(json["lane_k"], 3);
        assert_eq!(json["max_offered"], 3);
        assert_eq!(
            json["first_window_offered"],
            serde_json::json!({"1": 1, "3": 2})
        );
        assert!(lane_suite::lane_k_coverage("lane_batched", 3, &json).is_empty());
        // The same windows under a K=2 cohort: a window above K, and no first
        // window at K.
        let json = depth_evidence(&[3, 3, 1], 2);
        let failures = lane_suite::lane_k_coverage("lane_batched", 2, &json).concat();
        assert!(failures.contains("above the pinned lane K=2"), "{failures}");
        assert!(failures.contains("first window offered K=2"), "{failures}");
        // K=2 evidence read under a K=3 cohort is the wrong cohort.
        let json = depth_evidence(&[2, 2, 1], 2);
        assert!(lane_suite::lane_k_coverage("lane_batched", 2, &json).is_empty());
        let failures = lane_suite::lane_k_coverage("lane_batched", 3, &json).concat();
        assert!(failures.contains("records lane_k"), "{failures}");
        // The first windows never reached K.
        let json = depth_evidence(&[1, 1, 0], 2);
        let failures = lane_suite::lane_k_coverage("lane_batched", 2, &json).concat();
        assert!(failures.contains("never exercised"), "{failures}");
        // The reference arm has no depth rule; the crossing arm has it.
        assert!(lane_suite::lane_k_coverage("lane_ar", 2, &json).is_empty());
        assert!(!lane_suite::lane_k_coverage("lane_cross", 2, &json).is_empty());
    }

    #[test]
    fn depth_evidence_shards_merge_and_disagreeing_depths_do_not() {
        let a = depth_evidence(&[3, 1], 3);
        let b = depth_evidence(&[2, 2], 3);
        let merged = lane_suite::merge_evidence(&[a.clone(), b]).unwrap();
        assert_eq!(merged["max_offered"], 3);
        assert_eq!(
            merged["first_window_offered"],
            serde_json::json!({"1": 1, "2": 2, "3": 1})
        );
        assert_eq!(merged["lane_k"], 3);
        let k2 = depth_evidence(&[2, 1], 2);
        assert!(lane_suite::merge_evidence(&[a, k2])
            .unwrap_err()
            .contains("lane_k"));
    }

    #[test]
    fn plans_carry_requests_in_slot_order_with_absolute_positions() {
        let epoch = |tag| RequestEpoch {
            request_tag: tag,
            owner_generation: 1,
        };
        let plan = build_plan(
            4,
            vec![
                RowSpec {
                    slot: 2,
                    epoch: epoch(2),
                    tokens: vec![5, 6, 7],
                    start: 0,
                    kind: RequestStepKind::Prefill,
                },
                RowSpec {
                    slot: 0,
                    epoch: epoch(1),
                    tokens: vec![9],
                    start: 11,
                    kind: RequestStepKind::Ar,
                },
            ],
        );
        assert_eq!(plan.requests.len(), 2);
        assert_eq!(plan.requests[0].epoch, epoch(1));
        assert_eq!(plan.requests[0].kind, RequestStepKind::Ar);
        assert_eq!(plan.requests[1].rows, RowRange { begin: 1, len: 3 });
        assert_eq!((plan.decode_rows, plan.prefill_rows), (1, 3));
        assert_eq!(plan.batch.tokens, vec![9, 5, 6, 7]);
        assert_eq!(plan.batch.positions, vec![11, 0, 1, 2]);
    }

    #[test]
    fn the_sampler_spec_refuses_greedy_and_keeps_the_request_sampler() {
        let settings = |temp: f32, min_p: f32| DistSettings {
            trials: 1,
            temp,
            top_p: 0.95,
            top_k: Some(20),
            min_p,
            pen: Penalties {
                repeat_penalty: 1.0,
                repeat_window: 0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
            },
            fixture: Fixture::Prose,
        };
        assert!(LaneSamplerSpec::from_settings(&settings(0.0, 0.0)).is_err());
        assert!(LaneSamplerSpec::from_settings(&settings(f32::NAN, 0.0)).is_err());
        let spec = LaneSamplerSpec::from_settings(&settings(0.7, 0.05)).unwrap();
        let lane = spec.lane(42);
        assert_eq!(
            lane,
            LaneSampling {
                temperature: 0.7,
                top_p: 0.95,
                top_k: Some(20),
                min_p: Some(0.05),
                seed: 42,
                seeded: std::env::var("HIPFIRE_SAMPLED_MTP_UNSEEDED").as_deref() != Ok("1"),
            }
        );
        assert_eq!(
            LaneSamplerSpec::from_settings(&settings(1.0, 0.0))
                .unwrap()
                .min_p,
            None
        );
    }
}
