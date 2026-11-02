// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Micro-batch group count (`G`) selection for the PP decode wavefront.
//!
//! Pure host policy: no device, no clock. A step's lanes are split into `G`
//! contiguous whole-lane groups (`G` in `1..=min(lanes, 2 * stages)`); the
//! partition itself lives with the wavefront in `bundle/pp_lanes.rs`.
//!
//! Selection, in priority order:
//!
//! 1. A developer override (`HIPFIRE_PP_LANE_GROUPS=<G>`, any positive
//!    integer) wins over every policy: the effective count is
//!    `min(G, lanes, 2 * stages)` for this step, so one fixed request stays
//!    valid while the live lane count changes.
//! 2. A stage policy with no row-local sharing (every stage per lane or
//!    per class) keeps the maximal overlap `min(lanes, 2 * stages)`.
//! 3. A policy that shares row-local stages makes the best split
//!    shape-dependent: each `G` is sampled [`EXPLORE_SAMPLES`] times
//!    (round-robin, larger `G` first) from complete successful steps' walls,
//!    then the `G` with the smallest observed mean wall runs; exact ties go
//!    to the larger `G`. Statistics are keyed by the exact stage policy,
//!    stage count and per-lane row shape, so one never leaks into another.
//!
//! No arithmetic model or constant of a device is assumed: only measured
//! walls are compared.

use crate::gpu_forward_lanes::{LaneStagePolicy, StageName, StageShare};
use std::num::NonZeroUsize;

/// Developer override of the decode micro-batch count (positive integer).
pub(crate) const GROUPS_ENV: &str = "HIPFIRE_PP_LANE_GROUPS";
/// Developer override of the lane stage policy, parsed by
/// [`LaneStagePolicy::parse`], applied at every lane store's construction.
pub(crate) const POLICY_ENV: &str = "HIPFIRE_FN_LANE_POLICY";

/// Valid wall samples each `G` of a shape collects before the choice is made
/// from measurements.
pub(crate) const EXPLORE_SAMPLES: u32 = 2;
/// Attempts (valid or rejected observations) after which a `G` stops being
/// probed even without [`EXPLORE_SAMPLES`] valid samples.
const EXPLORE_ATTEMPTS: u32 = 2 * EXPLORE_SAMPLES;
/// Valid samples the running mean spans; beyond it the mean is exponential
/// with weight `1 / SAMPLE_WINDOW`, so a changed regime is eventually
/// followed without re-exploring.
pub(crate) const SAMPLE_WINDOW: u32 = 32;
/// Distinct (policy, stages, row shape) entries one planner tracks (an AR
/// store has at most one shape per lane count). Beyond it new shapes use the
/// maximal overlap and record nothing, so the storage is bounded.
pub(crate) const MAX_SHAPES: usize = 64;

/// Most groups `lanes` lanes can form over `stages` bands: one whole lane per
/// group and at most two groups per stage.
pub(crate) fn max_groups(lanes: usize, stages: usize) -> usize {
    lanes.min(stages.saturating_mul(2))
}

/// The count a request resolves to for the live `lanes` and `stages`.
pub(crate) fn clamp_groups(requested: NonZeroUsize, lanes: usize, stages: usize) -> usize {
    requested.get().min(max_groups(lanes, stages))
}

/// Parse `HIPFIRE_PP_LANE_GROUPS`. Unset is `Ok(None)`; any positive decimal
/// integer is accepted (larger than the legal maximum clamps at use, an
/// out-of-`usize` value saturates); zero, empty, signs, fractions and any
/// non-digit text are refused.
pub(crate) fn parse_groups_override(value: Option<&str>) -> Result<Option<NonZeroUsize>, String> {
    let Some(raw) = value else {
        return Ok(None);
    };
    let text = raw.trim();
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!(
            "{GROUPS_ENV}={raw:?} is not a positive integer micro-batch count"
        ));
    }
    let parsed = text.parse::<usize>().unwrap_or(usize::MAX);
    NonZeroUsize::new(parsed)
        .map(Some)
        .ok_or_else(|| format!("{GROUPS_ENV}={raw:?} must be at least 1"))
}

/// [`parse_groups_override`] over the developer snapshot.
pub(crate) fn groups_override_from_env() -> Result<Option<NonZeroUsize>, String> {
    parse_groups_override(hipfire_config::developer_var(GROUPS_ENV).ok().as_deref())
}

/// Parse `HIPFIRE_FN_LANE_POLICY`: unset is `Ok(None)`; anything else must be
/// a valid [`LaneStagePolicy::parse`] spec (an error refuses, never falls
/// back).
pub(crate) fn parse_policy_override(
    value: Option<&str>,
) -> Result<Option<LaneStagePolicy>, String> {
    let Some(raw) = value else {
        return Ok(None);
    };
    let policy = LaneStagePolicy::parse(raw).map_err(|e| format!("{POLICY_ENV}={raw:?}: {e}"))?;
    policy
        .validate()
        .map_err(|e| format!("{POLICY_ENV}={raw:?}: {e}"))?;
    Ok(Some(policy))
}

/// [`parse_policy_override`] over the developer snapshot.
pub(crate) fn policy_override_from_env() -> Result<Option<LaneStagePolicy>, String> {
    parse_policy_override(hipfire_config::developer_var(POLICY_ENV).ok().as_deref())
}

/// Whether the policy runs any stage once over all rows. `PerClass` executes
/// as per lane, so only `OnceAllRows` counts.
pub(crate) fn shares_row_local_stages(policy: &LaneStagePolicy) -> bool {
    StageName::ALL
        .iter()
        .any(|&stage| policy.share(stage) == StageShare::OnceAllRows)
}

/// Observed complete-step wall of one `G`.
#[derive(Clone, Copy, Debug, Default)]
struct WallStat {
    /// Valid samples, saturating at [`SAMPLE_WINDOW`].
    samples: u32,
    /// Observations offered (valid or not).
    attempts: u32,
    mean_us: f64,
}

impl WallStat {
    /// Fold in one wall; a non-finite or non-positive wall is rejected and
    /// does not count as a sample.
    fn observe(&mut self, wall_us: f64) -> bool {
        self.attempts = self.attempts.saturating_add(1);
        if !wall_us.is_finite() || wall_us <= 0.0 {
            return false;
        }
        self.samples = (self.samples + 1).min(SAMPLE_WINDOW);
        self.mean_us += (wall_us - self.mean_us) / f64::from(self.samples);
        true
    }

    fn needs_probe(&self) -> bool {
        self.samples < EXPLORE_SAMPLES && self.attempts < EXPLORE_ATTEMPTS
    }
}

/// Statistics of one exact (policy, stages, row shape); `walls[g - 1]` is `G`.
#[derive(Debug)]
struct ShapeStats {
    policy: LaneStagePolicy,
    stages: usize,
    rows: Vec<usize>,
    walls: Vec<WallStat>,
}

impl ShapeStats {
    /// Next `G` to run: the least-sampled one still probing (larger `G` on a
    /// tie), else the smallest observed mean (larger `G` on a tie), else the
    /// maximal overlap when nothing was ever measured.
    fn choose(&self) -> usize {
        let mut probe: Option<(usize, u32)> = None;
        for (index, stat) in self.walls.iter().enumerate().rev() {
            if stat.needs_probe() && probe.is_none_or(|(_, samples)| stat.samples < samples) {
                probe = Some((index + 1, stat.samples));
            }
        }
        if let Some((groups, _)) = probe {
            return groups;
        }
        let mut best: Option<(usize, f64)> = None;
        for (index, stat) in self.walls.iter().enumerate().rev() {
            if stat.samples > 0 && best.is_none_or(|(_, mean)| stat.mean_us < mean) {
                best = Some((index + 1, stat.mean_us));
            }
        }
        best.map_or(self.walls.len(), |(groups, _)| groups)
    }
}

/// The group-count policy of one root lane store.
#[derive(Debug)]
pub(crate) struct GroupPlanner {
    fixed: Option<NonZeroUsize>,
    entries: Vec<ShapeStats>,
}

impl GroupPlanner {
    pub(crate) fn new(fixed: Option<NonZeroUsize>) -> Self {
        Self {
            fixed,
            entries: Vec::new(),
        }
    }

    #[cfg(test)]
    /// The developer-requested count, before clamping.
    pub(crate) fn fixed(&self) -> Option<NonZeroUsize> {
        self.fixed
    }

    #[cfg(test)]
    /// Distinct shapes tracked.
    pub(crate) fn tracked_shapes(&self) -> usize {
        self.entries.len()
    }

    /// Forget every observation (a policy change).
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }

    fn position<I>(&self, policy: &LaneStagePolicy, stages: usize, rows: I) -> Option<usize>
    where
        I: Iterator<Item = usize> + Clone,
    {
        self.entries.iter().position(|entry| {
            entry.stages == stages
                && entry.policy == *policy
                && entry.rows.iter().copied().eq(rows.clone())
        })
    }

    /// The group count for a step over `rows` (rows per lane, plan order),
    /// in `1..=min(lanes, 2 * stages)`; `0` only for no lanes. Allocates only
    /// to record a first-seen shape.
    pub(crate) fn select<I>(&mut self, policy: &LaneStagePolicy, stages: usize, rows: I) -> usize
    where
        I: ExactSizeIterator<Item = usize> + Clone,
    {
        let max = max_groups(rows.len(), stages);
        if max == 0 {
            return 0;
        }
        if let Some(requested) = self.fixed {
            return clamp_groups(requested, rows.len(), stages);
        }
        if !shares_row_local_stages(policy) {
            return max;
        }
        let index = match self.position(policy, stages, rows.clone()) {
            Some(index) => index,
            None => {
                if self.entries.len() >= MAX_SHAPES {
                    return max;
                }
                self.entries.push(ShapeStats {
                    policy: *policy,
                    stages,
                    rows: rows.collect(),
                    walls: vec![WallStat::default(); max],
                });
                self.entries.len() - 1
            }
        };
        self.entries[index].choose()
    }

    /// Record the wall (microseconds) of one complete successful step that
    /// ran `groups` groups over `rows`. Only shapes [`Self::select`] already
    /// tracks record; an unknown shape, an out-of-range `groups` and an
    /// invalid wall are ignored. Returns whether the wall was recorded.
    pub(crate) fn observe<I>(
        &mut self,
        policy: &LaneStagePolicy,
        stages: usize,
        rows: I,
        groups: usize,
        wall_us: f64,
    ) -> bool
    where
        I: Iterator<Item = usize> + Clone,
    {
        let Some(index) = self.position(policy, stages, rows) else {
            return false;
        };
        let walls = &mut self.entries[index].walls;
        if groups == 0 || groups > walls.len() {
            return false;
        }
        walls[groups - 1].observe(wall_us)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shared() -> LaneStagePolicy {
        LaneStagePolicy::production()
    }

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    /// Run `steps` select/observe rounds with `wall(g)`; the chosen groups.
    fn drive(
        planner: &mut GroupPlanner,
        policy: &LaneStagePolicy,
        stages: usize,
        rows: &[usize],
        steps: usize,
        wall: impl Fn(usize) -> f64,
    ) -> Vec<usize> {
        (0..steps)
            .map(|_| {
                let g = planner.select(policy, stages, rows.iter().copied());
                assert!(planner.observe(policy, stages, rows.iter().copied(), g, wall(g)));
                g
            })
            .collect()
    }

    #[test]
    fn groups_override_parse() {
        assert_eq!(parse_groups_override(None), Ok(None));
        assert_eq!(parse_groups_override(Some("4")), Ok(Some(nz(4))));
        assert_eq!(parse_groups_override(Some(" 4 ")), Ok(Some(nz(4))));
        assert_eq!(parse_groups_override(Some("007")), Ok(Some(nz(7))));
        assert_eq!(
            parse_groups_override(Some("99999999999999999999999")),
            Ok(Some(nz(usize::MAX)))
        );
        for bad in [
            "0", "00", "", " ", "abc", "1.5", "-1", "+4", "4x", "1e2", "0x4",
        ] {
            let error = parse_groups_override(Some(bad)).unwrap_err();
            assert!(error.contains(GROUPS_ENV), "{bad:?}: {error}");
        }
    }

    #[test]
    fn policy_override_parse_refuses_errors() {
        assert_eq!(parse_policy_override(None), Ok(None));
        assert_eq!(
            parse_policy_override(Some("production")),
            Ok(Some(LaneStagePolicy::production()))
        );
        assert_eq!(
            parse_policy_override(Some("per-lane")),
            Ok(Some(LaneStagePolicy::all_per_lane()))
        );
        let tuned = parse_policy_override(Some("per-lane:moe=once"))
            .unwrap()
            .unwrap();
        assert_eq!(tuned.share(StageName::Moe), StageShare::OnceAllRows);
        for bad in [
            "",
            "bogus",
            "production:moe=sometimes",
            "production:nope=once",
            "production:gdn_core=once",
            "production:moe",
        ] {
            let error = parse_policy_override(Some(bad)).unwrap_err();
            assert!(error.contains(POLICY_ENV), "{bad:?}: {error}");
        }
    }

    #[test]
    fn shared_policy_detection() {
        assert!(shares_row_local_stages(&LaneStagePolicy::production()));
        assert!(!shares_row_local_stages(&LaneStagePolicy::all_per_lane()));
        let class = LaneStagePolicy::all_per_lane()
            .with(StageName::Moe, StageShare::PerClass)
            .unwrap();
        assert!(!shares_row_local_stages(&class));
    }

    #[test]
    fn override_clamps_to_live_lanes_and_stages() {
        for policy in [shared(), LaneStagePolicy::all_per_lane()] {
            let mut planner = GroupPlanner::new(Some(nz(4)));
            for lanes in 1..=8 {
                for stages in 1..=4 {
                    let rows = vec![1; lanes];
                    let g = planner.select(&policy, stages, rows.iter().copied());
                    assert_eq!(g, 4.min(lanes).min(2 * stages));
                    assert!(g >= 1);
                }
            }
            // K1/K2/K3 under a fixed G4 become 1/2/3.
            for lanes in 1..=3 {
                assert_eq!(
                    planner.select(&policy, 2, vec![1; lanes].into_iter()),
                    lanes
                );
            }
            // The override alone never records statistics.
            assert_eq!(planner.tracked_shapes(), 0);
        }
        // A request above the stage bound clamps rather than refuses.
        let mut planner = GroupPlanner::new(Some(nz(usize::MAX)));
        assert_eq!(planner.select(&shared(), 2, vec![1; 16].into_iter()), 4);
        assert_eq!(planner.select(&shared(), 3, vec![1; 16].into_iter()), 6);
        assert_eq!(planner.select(&shared(), 2, std::iter::empty()), 0);
        assert_eq!(planner.fixed(), Some(nz(usize::MAX)));
    }

    #[test]
    fn per_lane_policy_keeps_maximal_overlap_and_records_nothing() {
        let policy = LaneStagePolicy::all_per_lane();
        let mut planner = GroupPlanner::new(None);
        for stages in 1..=5 {
            for lanes in 1..=32 {
                let rows = vec![1; lanes];
                let g = planner.select(&policy, stages, rows.iter().copied());
                assert_eq!(g, lanes.min(2 * stages));
                assert!(!planner.observe(&policy, stages, rows.iter().copied(), g, 10.0));
            }
        }
        assert_eq!(planner.tracked_shapes(), 0);
        assert_eq!(planner.select(&policy, 2, std::iter::empty()), 0);
    }

    #[test]
    fn exploration_is_bounded_then_measured() {
        let mut planner = GroupPlanner::new(None);
        let rows = [1, 1, 1, 1];
        // G1 is cheapest, G4 dearest.
        let wall = |g: usize| [1000.0, 1500.0, 1400.0, 2000.0][g - 1];
        let steps = drive(&mut planner, &shared(), 2, &rows, 8, wall);
        // Round-robin, larger G first, EXPLORE_SAMPLES each.
        assert_eq!(steps, [4, 3, 2, 1, 4, 3, 2, 1]);
        for g in 1..=4 {
            assert_eq!(
                steps.iter().filter(|&&s| s == g).count(),
                EXPLORE_SAMPLES as usize
            );
        }
        // Exploration is over: the cheapest measured G runs from now on.
        let after = drive(&mut planner, &shared(), 2, &rows, 20, wall);
        assert!(after.iter().all(|&g| g == 1));
        assert_eq!(planner.tracked_shapes(), 1);
    }

    #[test]
    fn measured_choice_follows_the_walls_both_ways() {
        let rows = [1, 1, 1, 1];
        let mut g1 = GroupPlanner::new(None);
        drive(&mut g1, &shared(), 2, &rows, 8, |g| 100.0 * g as f64);
        assert_eq!(g1.select(&shared(), 2, rows.iter().copied()), 1);

        let mut g4 = GroupPlanner::new(None);
        drive(&mut g4, &shared(), 2, &rows, 8, |g| 100.0 * (5 - g) as f64);
        assert_eq!(g4.select(&shared(), 2, rows.iter().copied()), 4);

        // An interior minimum is found too.
        let mut g3 = GroupPlanner::new(None);
        drive(&mut g3, &shared(), 2, &rows, 8, |g| {
            [90.0, 80.0, 70.0, 95.0][g - 1]
        });
        assert_eq!(g3.select(&shared(), 2, rows.iter().copied()), 3);
    }

    #[test]
    fn exact_ties_prefer_the_larger_group_count() {
        let rows = [1, 1, 1, 1];
        let mut planner = GroupPlanner::new(None);
        drive(&mut planner, &shared(), 2, &rows, 8, |_| 500.0);
        assert_eq!(planner.select(&shared(), 2, rows.iter().copied()), 4);
        let mut planner = GroupPlanner::new(None);
        drive(&mut planner, &shared(), 2, &rows, 8, |g| {
            if g >= 3 {
                400.0
            } else {
                900.0
            }
        });
        assert_eq!(planner.select(&shared(), 2, rows.iter().copied()), 4);
    }

    #[test]
    fn shapes_stages_and_policies_are_isolated() {
        let rows4 = [1, 1, 1, 1];
        let mut planner = GroupPlanner::new(None);
        drive(&mut planner, &shared(), 2, &rows4, 8, |g| 100.0 * g as f64);
        assert_eq!(planner.select(&shared(), 2, rows4.iter().copied()), 1);

        // Other lane count, other row shape, other stage count and another
        // sharing policy each start from scratch: first probe is their max.
        assert_eq!(planner.select(&shared(), 2, [1, 1, 1].into_iter()), 3);
        assert_eq!(planner.select(&shared(), 2, [1, 1, 2, 1].into_iter()), 4);
        assert_eq!(planner.select(&shared(), 3, rows4.into_iter()), 4);
        let other = shared().with(StageName::Moe, StageShare::PerLane).unwrap();
        assert_ne!(other, shared());
        assert!(shares_row_local_stages(&other));
        assert_eq!(planner.select(&other, 2, rows4.into_iter()), 4);
        assert_eq!(planner.tracked_shapes(), 5);

        // An observation for one key never lands in another.
        assert!(!planner.observe(&shared(), 4, rows4.into_iter(), 1, 1.0));
        assert!(planner.observe(&other, 2, rows4.into_iter(), 4, 1.0));
        assert_eq!(planner.select(&shared(), 2, rows4.into_iter()), 1);
        // The original key's statistics are untouched by the others'.
        drive(&mut planner, &other, 2, &rows4, 7, |g| {
            10.0 * (5 - g) as f64
        });
        assert_eq!(planner.select(&other, 2, rows4.into_iter()), 4);
        assert_eq!(planner.select(&shared(), 2, rows4.into_iter()), 1);
    }

    #[test]
    fn selection_is_deterministic() {
        let run = || {
            let mut planner = GroupPlanner::new(None);
            let mut out = Vec::new();
            for step in 0..40usize {
                let rows: Vec<usize> = vec![1; 1 + step % 6];
                let g = planner.select(&shared(), 3, rows.iter().copied());
                let wall = 50.0 + ((g * 37 + rows.len() * 11) % 23) as f64;
                planner.observe(&shared(), 3, rows.iter().copied(), g, wall);
                out.push(g);
            }
            out
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn invalid_observations_are_dropped_and_probing_stays_bounded() {
        let rows = [1, 1];
        let mut planner = GroupPlanner::new(None);
        assert_eq!(planner.select(&shared(), 2, rows.iter().copied()), 2);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -5.0] {
            assert!(!planner.observe(&shared(), 2, rows.iter().copied(), 2, bad));
        }
        // Out-of-range counts and unknown shapes are ignored.
        assert!(!planner.observe(&shared(), 2, rows.iter().copied(), 0, 5.0));
        assert!(!planner.observe(&shared(), 2, rows.iter().copied(), 3, 5.0));
        assert!(!planner.observe(&shared(), 2, [1, 1, 1].into_iter(), 1, 5.0));
        // G2 burned its attempts without a sample: it is no longer probed
        // (five attempts spent above), so G1 is probed instead.
        assert_eq!(planner.select(&shared(), 2, rows.iter().copied()), 1);
        assert!(planner.observe(&shared(), 2, rows.iter().copied(), 1, 40.0));
        assert!(planner.observe(&shared(), 2, rows.iter().copied(), 1, 60.0));
        // G2 has no measurement, so G1 is the only candidate.
        assert_eq!(planner.select(&shared(), 2, rows.iter().copied()), 1);
    }

    #[test]
    fn nothing_measured_falls_back_to_maximal_overlap() {
        let rows = [1, 1, 1];
        let mut planner = GroupPlanner::new(None);
        planner.select(&shared(), 2, rows.iter().copied());
        for g in 1..=3 {
            for _ in 0..EXPLORE_ATTEMPTS {
                planner.observe(&shared(), 2, rows.iter().copied(), g, f64::NAN);
            }
        }
        assert_eq!(planner.select(&shared(), 2, rows.iter().copied()), 3);
    }

    #[test]
    fn mean_follows_a_changed_regime() {
        let rows = [1, 1];
        let mut planner = GroupPlanner::new(None);
        drive(&mut planner, &shared(), 2, &rows, 4, |g| 1000.0 * g as f64);
        for _ in 0..(SAMPLE_WINDOW * 2) {
            planner.observe(&shared(), 2, rows.iter().copied(), 1, 1000.0);
            planner.observe(&shared(), 2, rows.iter().copied(), 2, 2000.0);
        }
        assert_eq!(planner.select(&shared(), 2, rows.iter().copied()), 1);
        for _ in 0..(SAMPLE_WINDOW * 8) {
            planner.observe(&shared(), 2, rows.iter().copied(), 2, 1.0);
        }
        assert_eq!(planner.select(&shared(), 2, rows.iter().copied()), 2);
    }

    #[test]
    fn tracked_shapes_are_bounded() {
        let mut planner = GroupPlanner::new(None);
        for r in 1..=(MAX_SHAPES + 8) {
            let g = planner.select(&shared(), 2, [r, 1].into_iter());
            assert_eq!(g, 2);
        }
        assert_eq!(planner.tracked_shapes(), MAX_SHAPES);
        assert!(!planner.observe(&shared(), 2, [MAX_SHAPES + 3, 1].into_iter(), 2, 1.0));
    }

    #[test]
    fn clear_forgets_statistics() {
        let rows = [1, 1, 1, 1];
        let mut planner = GroupPlanner::new(None);
        drive(&mut planner, &shared(), 2, &rows, 8, |g| 100.0 * g as f64);
        assert_eq!(planner.select(&shared(), 2, rows.iter().copied()), 1);
        planner.clear();
        assert_eq!(planner.tracked_shapes(), 0);
        assert_eq!(planner.select(&shared(), 2, rows.iter().copied()), 4);
    }

    #[test]
    fn choice_is_always_a_legal_group_count() {
        let mut planner = GroupPlanner::new(None);
        for stages in 1..=5 {
            for lanes in 1..=16 {
                let rows = vec![1; lanes];
                for step in 0..(3 * 2 * stages) {
                    let g = planner.select(&shared(), stages, rows.iter().copied());
                    assert!((1..=max_groups(lanes, stages)).contains(&g));
                    planner.observe(
                        &shared(),
                        stages,
                        rows.iter().copied(),
                        g,
                        10.0 + step as f64,
                    );
                }
            }
        }
    }
}
