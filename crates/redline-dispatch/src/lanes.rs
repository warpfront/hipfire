// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! How many independent queue lanes to submit PM4 work on.
//!
//! This is a distinct axis from [`crate::partition`], which carves *compute
//! units* for green contexts. Lanes are *queues*: separate hardware submission
//! paths that let independent work progress concurrently.
//!
//! # Why this needs a policy at all
//!
//! The useful lane count is device-specific, is not derivable from any published
//! device property, and overshooting it is catastrophic rather than merely
//! suboptimal. Measured on ROCm 10.0 with N=512 no-op dispatches, host latency
//! per dispatch in microseconds:
//!
//! | lanes | gfx1030 | gfx1100 | gfx1151 | gfx1201 |
//! | ----: | ------: | ------: | ------: | ------: |
//! |     1 |  0.2009 |  0.2239 |  0.2379 |  0.1472 |
//! |     2 |  0.1325 |  0.1331 |  0.1482 |  0.0908 |
//! |     4 |  0.0952 |  0.1005 |  0.0786 |  0.1212 |
//! |     5 |  0.1537 |  0.1568 |  0.1024 |       — |
//! |     8 |       — |       — |       — | 11.7072 |
//!
//! gfx1201 at 8 lanes is 129x worse than the same part at its own 2-lane
//! optimum. That is why [`LaneWidth::Measured`] falls back to
//! [`CONSERVATIVE_LANES`] for devices it does not recognize instead of guessing
//! high: the downside of undershooting is a fraction of the available speedup,
//! and the downside of overshooting is two orders of magnitude.
//!
//! All four parts advertise 128 maximum queues, so the advertised maximum says
//! nothing about the useful width and is deliberately not consulted here.
//!
//! # What these numbers do and do not cover
//!
//! The table is submission cost for empty kernels at one work shape: N=512
//! dispatches of a 1x1x1 grid. It measures the dispatch path, not occupancy.
//!
//! That caveat is not hypothetical. Measured afterwards on gfx1201 with the
//! `hipfire-6409` suite, three repeats per point, HIP independent-throughput
//! us/op, comparing 4 lanes against this table's 2:
//!
//! | dispatches | grid groups | 4 lanes | 2 lanes | better |
//! | ---: | ---: | ---: | ---: | --- |
//! | 50 | 1 | 9.5 | 4.3 | 2 lanes, 2.21x |
//! | 200 | 1 | 8.0 | 3.3 | 2 lanes, 2.40x |
//! | 941 | 1 | 7.6 | 14.9 | **4 lanes, 1.96x** |
//! | 941 | 128 | 8.6 | 15.5 | **4 lanes, 1.79x** |
//! | 941 | 1024 | 17.5 | 12.2 | 2 lanes, 1.44x |
//! | 941 | 8192 | 65 | 51 | 2 lanes, 1.26x |
//!
//! The preferred width **reverses with work shape on a single device**, by up to
//! 2.4x either way, and the direction is stable across repeats rather than
//! noise. Across the suite's eleven kernel families, six preferred 4 lanes and
//! four preferred 2 with a consistent sign over three runs.
//!
//! So [`measured_lanes`] returning one integer per device is the wrong shape for
//! the real problem, and is kept only because it is an honest record of one
//! measured point: submission-bound work, small grids, at the stated N. Treating
//! it as a device-wide optimum will cost roughly 2x on the shapes that want the
//! other width. A caller with a real workload MUST measure its own shapes and
//! pass [`LaneWidth::Explicit`], or drive [`LaneWidth::Probe`]; that is not
//! defensive boilerplate, it is the measured conclusion.
//!
//! # Runtime probing
//!
//! The table covers only the parts that were in the lab when it was built. For
//! any other device [`LaneWidth::Measured`] must honestly fall back to
//! [`CONSERVATIVE_LANES`], but a caller can instead ask the runtime to discover
//! the width via [`LaneWidth::Probe`]. Probing is out-of-band and
//! caller-driven: the caller obtains the candidate list from
//! [`probe_plan`], times each lane count on the actual device, selects the
//! winner with [`best_from_samples`], records it in a [`LaneProfile`], and then
//! consults that profile on subsequent resolves. No file IO, no serde, no
//! global mutable state is involved — the profile is an in-memory value the
//! caller owns and threads through [`resolve_with_profile`].

use std::num::NonZeroUsize;

/// Lane count used for devices with no measured entry.
///
/// Two, not one: two lanes beat one on every part measured so far, and two was
/// also the optimum on the part that collapsed hardest past its width. It is
/// the widest value not observed to fall off a cliff anywhere.
pub const CONSERVATIVE_LANES: usize = 2;

/// Refuse to resolve a lane count above this.
///
/// Not a hardware limit — the parts advertise 128 queues. It is a guard against
/// a caller or config passing a number in the range that was measured to cost
/// 129x, since no measured optimum is anywhere near it.
pub const MAX_LANES: usize = 16;

/// How to choose the number of queue lanes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LaneWidth {
    /// Submit on a single queue. Always valid; forgoes the queue-width win.
    Single,
    /// Exactly this many lanes, chosen by the caller — normally because the
    /// caller measured its own workload rather than trusting the no-op table.
    Explicit(NonZeroUsize),
    /// The measured optimum for the running device, or [`CONSERVATIVE_LANES`]
    /// when the device is not in the table.
    Measured,
    /// Discover the width at runtime by timing `candidates` lane counts.
    ///
    /// # Why this shape
    ///
    /// `candidates` is the ordered set of lane counts the caller wants timed
    /// (e.g. `[1, 2, 4, 8]`). `dispatches` records how many dispatches the
    /// probe workload will contain, so the plan can be reasoned about without
    /// threading a separate count through every call. At resolve time the
    /// *actual* `dispatch_count` passed to [`resolve`] or [`probe_plan`] is
    /// authoritative for clamping — a probe that was constructed for 512
    /// dispatches but is asked to resolve for 3 dispatches will not create a
    /// fourth lane with no work. The stored `dispatches` is informational and
    /// does not override the caller's dispatch count.
    ///
    /// A `Probe` that has not yet been timed (no [`LaneProfile`] has been
    /// produced from its samples) resolves to [`CONSERVATIVE_LANES`], not to a
    /// guess. This keeps an unmeasured device from silently running wide and
    /// hitting the 129x cliff observed on gfx1201.
    Probe {
        /// Ordered lane counts to time. Duplicates are de-duplicated by
        /// [`probe_plan`]; values above [`MAX_LANES`] are rejected.
        candidates: Vec<usize>,
        /// Number of dispatches the probe workload will contain. Informational;
        /// the `dispatch_count` argument to [`resolve`]/[`probe_plan`] governs
        /// clamping.
        dispatches: usize,
    },
}

/// Why a lane count could not be resolved.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum LaneError {
    /// The requested lane count exceeds [`MAX_LANES`].
    #[error("requested {requested} lanes exceeds the {MAX_LANES}-lane guard")]
    TooWide { requested: usize },
    /// There is no work to distribute.
    #[error("cannot resolve lanes for zero dispatches")]
    NoWork,
}

/// The measured best lane count for a device, if it has been measured.
///
/// Matching is on the `gfx` family prefix reported by the device, so
/// `gfx1201`-class parts resolve regardless of any suffix. Returns `None`
/// rather than a guess for anything unmeasured; that distinction is what keeps
/// [`LaneWidth::Measured`] honest about which devices it actually knows.
pub fn measured_lanes(device_name: &str) -> Option<usize> {
    // Measured on ROCm 10.0, N=512 no-op dispatches, dependency-ordered within
    // each lane. See the module table for the full sweeps these came from.
    let entry = match device_name {
        n if n.starts_with("gfx1030") => 4,
        n if n.starts_with("gfx1100") => 4,
        n if n.starts_with("gfx1151") => 4,
        n if n.starts_with("gfx1201") => 2,
        _ => return None,
    };
    Some(entry)
}

/// Pick the lane count with the lowest measured cost.
///
/// Pure decision logic — no GPU work, no IO. `samples` is a slice of
/// `(lane_count, cost)` pairs where `cost` is typically microseconds per
/// dispatch (lower is better). Returns `None` for an empty slice, otherwise
/// the lane count whose cost is minimal. On an exact tie the **fewer** lanes
/// wins, because fewer queues is less resource for equal gain. Non-finite costs
/// (`NaN`, `+inf`, `-inf`) are ignored; if every sample is non-finite the
/// result is `None`. Lane counts of zero are ignored as invalid.
///
/// This does not claim optimality for any workload — it simply returns the
/// minimum of what it was given. The honesty of the no-op table applies here
/// too: if the caller timed real kernels, the winner reflects real kernels; if
/// the caller timed no-ops, the winner reflects no-ops.
pub fn best_from_samples(samples: &[(usize, f64)]) -> Option<usize> {
    let mut best_lane: Option<usize> = None;
    let mut best_cost = f64::INFINITY;
    for &(lane, cost) in samples {
        if lane == 0 {
            continue;
        }
        if !cost.is_finite() {
            continue;
        }
        match best_lane {
            None => {
                best_lane = Some(lane);
                best_cost = cost;
            }
            Some(current_best) => {
                if cost < best_cost {
                    best_lane = Some(lane);
                    best_cost = cost;
                } else if cost == best_cost && lane < current_best {
                    // Exact tie: prefer fewer lanes.
                    best_lane = Some(lane);
                    // best_cost unchanged
                }
            }
        }
    }
    best_lane
}

/// Ordered candidate lane counts a caller should time for `policy`.
///
/// For [`LaneWidth::Single`], [`LaneWidth::Explicit`], and
/// [`LaneWidth::Measured`] this is the single resolved lane count (clamped to
/// `dispatch_count`). For [`LaneWidth::Probe`] it is the deduplicated,
/// dispatch-clamped candidate list in the order the candidates were supplied.
///
/// Each candidate is checked against [`MAX_LANES`] *before* clamping: a
/// candidate that exceeds the guard is an error even if `dispatch_count` would
/// have clamped it down. After the guard, each value is clamped to
/// `dispatch_count` (a lane with no work is a queue created and fenced for
/// nothing) and to at least 1, then deduplicated preserving first-seen order.
///
/// Returns `Err(LaneError::NoWork)` if `dispatch_count == 0` or if a `Probe`
/// carries an empty candidate list. Returns `Err(LaneError::TooWide)` if any
/// candidate (or the single resolved value) exceeds [`MAX_LANES`].
pub fn probe_plan(
    policy: &LaneWidth,
    device_name: &str,
    dispatch_count: usize,
) -> Result<Vec<usize>, LaneError> {
    if dispatch_count == 0 {
        return Err(LaneError::NoWork);
    }
    match policy {
        LaneWidth::Single => {
            let requested = 1;
            if requested > MAX_LANES {
                return Err(LaneError::TooWide { requested });
            }
            Ok(vec![requested.min(dispatch_count).max(1)])
        }
        LaneWidth::Explicit(n) => {
            let requested = n.get();
            if requested > MAX_LANES {
                return Err(LaneError::TooWide { requested });
            }
            Ok(vec![requested.min(dispatch_count).max(1)])
        }
        LaneWidth::Measured => {
            let requested = measured_lanes(device_name).unwrap_or(CONSERVATIVE_LANES);
            if requested > MAX_LANES {
                return Err(LaneError::TooWide { requested });
            }
            Ok(vec![requested.min(dispatch_count).max(1)])
        }
        LaneWidth::Probe {
            candidates,
            dispatches: _,
        } => {
            if candidates.is_empty() {
                return Err(LaneError::NoWork);
            }
            let mut out = Vec::new();
            for &c in candidates {
                if c == 0 {
                    // Zero is not a valid lane count; treat as an error
                    // rather than silently mapping to 1, so the caller
                    // notices the malformed candidate list.
                    return Err(LaneError::NoWork);
                }
                if c > MAX_LANES {
                    return Err(LaneError::TooWide { requested: c });
                }
                let clamped = c.min(dispatch_count).max(1);
                if !out.contains(&clamped) {
                    out.push(clamped);
                }
            }
            if out.is_empty() {
                return Err(LaneError::NoWork);
            }
            Ok(out)
        }
    }
}

/// Resolve `policy` into a concrete lane count for `dispatch_count` dispatches
/// on `device_name`.
///
/// The result never exceeds `dispatch_count`: a lane with no work is a queue
/// created and fenced for nothing. It is always at least 1.
///
/// For [`LaneWidth::Probe`] this function has no samples to consult, so it
/// conservatively returns [`CONSERVATIVE_LANES`] (clamped to `dispatch_count`).
/// A `Probe` that has not run yet must not be treated as a measurement — the
/// downside of guessing wide is the 129x cliff measured on gfx1201. Callers
/// that have timed a probe should construct a [`LaneProfile`] via
/// [`LaneProfile::from_samples`] and then resolve through
/// [`resolve_with_profile`] or [`LaneProfile::resolve`].
pub fn resolve(
    policy: &LaneWidth,
    device_name: &str,
    dispatch_count: usize,
) -> Result<usize, LaneError> {
    if dispatch_count == 0 {
        return Err(LaneError::NoWork);
    }
    let requested = match policy {
        LaneWidth::Single => 1,
        LaneWidth::Explicit(n) => n.get(),
        LaneWidth::Measured => measured_lanes(device_name).unwrap_or(CONSERVATIVE_LANES),
        LaneWidth::Probe { .. } => {
            // No samples have been supplied, so this is an unresolved probe.
            // Fall back conservatively rather than guessing from the candidate
            // list — the candidate list may contain 8 or 16, which would be
            // catastrophic on gfx1201.
            CONSERVATIVE_LANES
        }
    };
    if requested > MAX_LANES {
        return Err(LaneError::TooWide { requested });
    }
    Ok(requested.min(dispatch_count).max(1))
}

/// An in-memory recording of a lane-width probe.
///
/// Created from timing samples via [`LaneProfile::from_samples`], which picks
/// the winner with [`best_from_samples`]. The profile is a plain owned value:
/// no file IO, no serde, no global mutable state. The caller keeps it and
/// passes it to [`resolve_with_profile`] or calls [`LaneProfile::resolve`]
/// when it wants the probed width to be used.
#[derive(Clone, Debug, PartialEq)]
pub struct LaneProfile {
    /// Device name the samples were taken on (e.g. `"gfx1201"`).
    pub device: String,
    /// Winning lane count selected by [`best_from_samples`].
    pub lanes: usize,
    /// The samples the decision was made from, retained for inspection.
    pub samples: Vec<(usize, f64)>,
}

impl LaneProfile {
    /// Build a profile from `samples` taken on `device`.
    ///
    /// Returns `None` if `samples` is empty, contains only non-finite costs,
    /// or otherwise has no winner per [`best_from_samples`], or if the winner
    /// exceeds [`MAX_LANES`] or is zero.
    pub fn from_samples(device: impl Into<String>, samples: &[(usize, f64)]) -> Option<Self> {
        let lanes = best_from_samples(samples)?;
        if lanes == 0 || lanes > MAX_LANES {
            return None;
        }
        Some(Self {
            device: device.into(),
            lanes,
            samples: samples.to_vec(),
        })
    }

    /// Resolve a lane count for `dispatch_count` dispatches using this profile
    /// when it matches `device_name`, otherwise fall back to `CONSERVATIVE_LANES`.
    ///
    /// This is a convenience for callers that own a single profile. For
    /// policy-aware resolution that also respects [`LaneWidth`] and an optional
    /// profile, see [`resolve_with_profile`].
    pub fn resolve(&self, device_name: &str, dispatch_count: usize) -> Result<usize, LaneError> {
        if dispatch_count == 0 {
            return Err(LaneError::NoWork);
        }
        let requested = if self.device == device_name {
            self.lanes
        } else {
            CONSERVATIVE_LANES
        };
        if requested > MAX_LANES {
            return Err(LaneError::TooWide { requested });
        }
        Ok(requested.min(dispatch_count).max(1))
    }

    /// Whether this profile was taken on `device_name`.
    pub fn matches_device(&self, device_name: &str) -> bool {
        self.device == device_name
    }
}

/// Resolve `policy` into a concrete lane count, consulting `profile` when it
/// matches `device_name`.
///
/// If `policy` is [`LaneWidth::Probe`] and `profile` is `Some` with a matching
/// device, the profile's winning lane count is used (clamped to
/// `dispatch_count`). If the profile is `None` or for a different device, a
/// `Probe` falls back to [`CONSERVATIVE_LANES`] — exactly as [`resolve`] does
/// for an unresolved probe. For non-`Probe` policies the profile is ignored and
/// resolution is identical to [`resolve`].
pub fn resolve_with_profile(
    policy: &LaneWidth,
    device_name: &str,
    dispatch_count: usize,
    profile: Option<&LaneProfile>,
) -> Result<usize, LaneError> {
    if dispatch_count == 0 {
        return Err(LaneError::NoWork);
    }
    let requested = match policy {
        LaneWidth::Single => 1,
        LaneWidth::Explicit(n) => n.get(),
        LaneWidth::Measured => measured_lanes(device_name).unwrap_or(CONSERVATIVE_LANES),
        LaneWidth::Probe { .. } => {
            if let Some(p) = profile {
                if p.device == device_name {
                    p.lanes
                } else {
                    CONSERVATIVE_LANES
                }
            } else {
                CONSERVATIVE_LANES
            }
        }
    };
    if requested > MAX_LANES {
        return Err(LaneError::TooWide { requested });
    }
    Ok(requested.min(dispatch_count).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("nonzero")
    }

    #[test]
    fn measured_table_covers_the_parts_that_were_measured() {
        assert_eq!(measured_lanes("gfx1030"), Some(4));
        assert_eq!(measured_lanes("gfx1100"), Some(4));
        assert_eq!(measured_lanes("gfx1151"), Some(4));
        // The part that collapses at 8 lanes must not inherit a wider default.
        assert_eq!(measured_lanes("gfx1201"), Some(2));
    }

    #[test]
    fn unmeasured_devices_are_not_guessed() {
        assert_eq!(measured_lanes("gfx942"), None);
        assert_eq!(measured_lanes("gfx90a"), None);
    }

    #[test]
    fn unknown_device_falls_back_conservatively_rather_than_wide() {
        let lanes = resolve(&LaneWidth::Measured, "gfx942", 512).expect("resolves");
        assert_eq!(lanes, CONSERVATIVE_LANES);
    }

    #[test]
    fn measured_policy_uses_the_table_when_present() {
        assert_eq!(resolve(&LaneWidth::Measured, "gfx1201", 512), Ok(2));
        assert_eq!(resolve(&LaneWidth::Measured, "gfx1100", 512), Ok(4));
    }

    #[test]
    fn lane_count_never_exceeds_the_work_available() {
        // Four lanes asked for, three dispatches to spread: a fourth lane would
        // be a queue created and fenced for no work.
        assert_eq!(resolve(&LaneWidth::Measured, "gfx1100", 3), Ok(3));
        assert_eq!(resolve(&LaneWidth::Explicit(nz(8)), "gfx1100", 1), Ok(1));
    }

    #[test]
    fn zero_work_is_an_error_not_a_silent_single_lane() {
        assert_eq!(resolve(&LaneWidth::Measured, "gfx1100", 0), Err(LaneError::NoWork));
    }

    #[test]
    fn absurd_widths_are_refused() {
        assert_eq!(
            resolve(&LaneWidth::Explicit(nz(64)), "gfx1100", 512),
            Err(LaneError::TooWide { requested: 64 })
        );
    }

    #[test]
    fn single_is_always_one_lane() {
        assert_eq!(resolve(&LaneWidth::Single, "gfx1201", 512), Ok(1));
        assert_eq!(resolve(&LaneWidth::Single, "unknown", 512), Ok(1));
    }

    // --- best_from_samples ---

    #[test]
    fn best_from_samples_empty_is_none() {
        assert_eq!(best_from_samples(&[]), None);
    }

    #[test]
    fn best_from_samples_single_sample_wins() {
        assert_eq!(best_from_samples(&[(4, 0.1)]), Some(4));
    }

    #[test]
    fn best_from_samples_picks_minimum() {
        let samples = vec![(1, 0.20), (2, 0.13), (4, 0.09), (8, 0.15)];
        assert_eq!(best_from_samples(&samples), Some(4));
    }

    #[test]
    fn best_from_samples_tie_prefers_fewer_lanes() {
        // Two lane counts with identical cost: fewer must win.
        let samples = vec![(4, 0.10), (2, 0.10)];
        assert_eq!(best_from_samples(&samples), Some(2));
        // Order should not matter.
        let samples_rev = vec![(2, 0.10), (4, 0.10)];
        assert_eq!(best_from_samples(&samples_rev), Some(2));
        // Three-way tie.
        let samples3 = vec![(8, 0.10), (4, 0.10), (2, 0.10)];
        assert_eq!(best_from_samples(&samples3), Some(2));
    }

    #[test]
    fn best_from_samples_ignores_non_finite_costs() {
        let samples = vec![(1, f64::NAN), (2, f64::INFINITY), (4, 0.09)];
        assert_eq!(best_from_samples(&samples), Some(4));
        // All non-finite => None
        let all_bad = vec![(1, f64::NAN), (2, f64::INFINITY)];
        assert_eq!(best_from_samples(&all_bad), None);
    }

    #[test]
    fn best_from_samples_ignores_zero_lane_counts() {
        let samples = vec![(0, 0.01), (2, 0.10)];
        assert_eq!(best_from_samples(&samples), Some(2));
        // Only zero lanes => None
        assert_eq!(best_from_samples(&[(0, 0.01)]), None);
    }

    #[test]
    fn best_from_samples_cliff_gfx1201_shape() {
        // gfx1201 shape: 1:0.1472, 2:0.0908 (optimum), 4:0.1212, 8:11.7072
        // 8 lanes is ~129x worse than optimum and must never win.
        let samples = vec![(1, 0.1472), (2, 0.0908), (4, 0.1212), (8, 11.7072)];
        assert_eq!(best_from_samples(&samples), Some(2));
        // Even if only the cliff and the optimum are present, the cliff loses.
        let cliff_only = vec![(2, 0.0908), (8, 11.7072)];
        assert_eq!(best_from_samples(&cliff_only), Some(2));
    }

    #[test]
    fn best_from_samples_does_not_prefer_wide_on_close_costs() {
        // Slightly lower cost at wider width should still win — the function
        // is honest about what it was given. If the caller wants a bias toward
        // fewer lanes they must encode it in cost.
        let samples = vec![(2, 0.100), (4, 0.099)];
        assert_eq!(best_from_samples(&samples), Some(4));
    }

    // --- probe_plan ---

    #[test]
    fn probe_plan_for_single_is_single_lane() {
        assert_eq!(probe_plan(&LaneWidth::Single, "gfx1100", 512), Ok(vec![1]));
        // Clamped to dispatch count when less than 1? 1 is always <= dispatch
        // except dispatch==1 where it equals.
        assert_eq!(probe_plan(&LaneWidth::Single, "gfx1100", 1), Ok(vec![1]));
    }

    #[test]
    fn probe_plan_for_explicit_is_that_value() {
        assert_eq!(
            probe_plan(&LaneWidth::Explicit(nz(4)), "gfx1100", 512),
            Ok(vec![4])
        );
    }

    #[test]
    fn probe_plan_for_measured_uses_table() {
        assert_eq!(probe_plan(&LaneWidth::Measured, "gfx1201", 512), Ok(vec![2]));
        assert_eq!(
            probe_plan(&LaneWidth::Measured, "gfx942", 512),
            Ok(vec![CONSERVATIVE_LANES])
        );
    }

    #[test]
    fn probe_plan_for_measured_clamps_to_work() {
        assert_eq!(probe_plan(&LaneWidth::Measured, "gfx1100", 2), Ok(vec![2]));
    }

    #[test]
    fn probe_plan_for_probe_returns_deduped_clamped_candidates() {
        let policy = LaneWidth::Probe {
            candidates: vec![1, 2, 4, 8],
            dispatches: 512,
        };
        assert_eq!(probe_plan(&policy, "gfx942", 512), Ok(vec![1, 2, 4, 8]));
        // Deduplication
        let dup = LaneWidth::Probe {
            candidates: vec![2, 4, 2, 4, 8],
            dispatches: 512,
        };
        assert_eq!(probe_plan(&dup, "gfx942", 512), Ok(vec![2, 4, 8]));
        // Clamping deduplicates
        let clamp_dup = LaneWidth::Probe {
            candidates: vec![4, 8],
            dispatches: 512,
        };
        // dispatch_count=3 clamps both 4 and 8 to 3 => dedup to [3]
        assert_eq!(probe_plan(&clamp_dup, "gfx942", 3), Ok(vec![3]));
        // Mixed clamping preserves order of first appearance
        let mixed = LaneWidth::Probe {
            candidates: vec![1, 4, 8],
            dispatches: 512,
        };
        assert_eq!(probe_plan(&mixed, "gfx942", 2), Ok(vec![1, 2]));
    }

    #[test]
    fn probe_plan_rejects_candidate_above_max_lanes() {
        let policy = LaneWidth::Probe {
            candidates: vec![1, 2, 64],
            dispatches: 512,
        };
        assert_eq!(
            probe_plan(&policy, "gfx942", 512),
            Err(LaneError::TooWide { requested: 64 })
        );
        // Explicit above MAX_LANES also errors
        assert_eq!(
            probe_plan(&LaneWidth::Explicit(nz(32)), "gfx942", 512),
            Err(LaneError::TooWide { requested: 32 })
        );
    }

    #[test]
    fn probe_plan_zero_dispatch_is_no_work() {
        assert_eq!(
            probe_plan(&LaneWidth::Measured, "gfx1100", 0),
            Err(LaneError::NoWork)
        );
        let probe = LaneWidth::Probe {
            candidates: vec![1, 2],
            dispatches: 10,
        };
        assert_eq!(probe_plan(&probe, "gfx942", 0), Err(LaneError::NoWork));
    }

    #[test]
    fn probe_plan_empty_candidates_is_no_work() {
        let probe = LaneWidth::Probe {
            candidates: vec![],
            dispatches: 512,
        };
        assert_eq!(probe_plan(&probe, "gfx942", 512), Err(LaneError::NoWork));
    }

    #[test]
    fn probe_plan_zero_candidate_is_no_work() {
        let probe = LaneWidth::Probe {
            candidates: vec![0, 2],
            dispatches: 512,
        };
        assert_eq!(probe_plan(&probe, "gfx942", 512), Err(LaneError::NoWork));
    }

    // --- resolve with Probe ---

    #[test]
    fn probe_without_profile_resolves_conservatively() {
        let probe = LaneWidth::Probe {
            candidates: vec![1, 2, 4, 8],
            dispatches: 512,
        };
        // Unmeasured device => conservative
        assert_eq!(resolve(&probe, "gfx942", 512), Ok(CONSERVATIVE_LANES));
        // Measured device too — an unresolved probe does not inherit the table.
        // This is intentional: Probe means "I will measure", not "use the table".
        assert_eq!(resolve(&probe, "gfx1100", 512), Ok(CONSERVATIVE_LANES));
        // Still clamped to dispatch_count
        assert_eq!(resolve(&probe, "gfx942", 1), Ok(1));
    }

    #[test]
    fn probe_without_profile_zero_work_is_error() {
        let probe = LaneWidth::Probe {
            candidates: vec![1, 2],
            dispatches: 512,
        };
        assert_eq!(resolve(&probe, "gfx942", 0), Err(LaneError::NoWork));
    }

    // --- LaneProfile ---

    #[test]
    fn lane_profile_from_samples_picks_winner() {
        let samples = vec![(1, 0.1472), (2, 0.0908), (4, 0.1212), (8, 11.7072)];
        let profile = LaneProfile::from_samples("gfx1201", &samples).expect("profile");
        assert_eq!(profile.lanes, 2);
        assert_eq!(profile.device, "gfx1201");
        assert_eq!(profile.samples, samples);
    }

    #[test]
    fn lane_profile_from_empty_samples_is_none() {
        assert_eq!(LaneProfile::from_samples("gfx942", &[]), None);
    }

    #[test]
    fn lane_profile_from_all_non_finite_is_none() {
        let bad = vec![(1, f64::NAN), (2, f64::INFINITY)];
        assert_eq!(LaneProfile::from_samples("gfx942", &bad), None);
    }

    #[test]
    fn lane_profile_resolve_uses_profile_when_device_matches() {
        let samples = vec![(2, 0.09), (4, 0.12)];
        let profile = LaneProfile::from_samples("gfx942", &samples).unwrap();
        // Matching device => profile lanes, clamped
        assert_eq!(profile.resolve("gfx942", 512), Ok(2));
        assert_eq!(profile.resolve("gfx942", 1), Ok(1));
        // Non-matching device => conservative
        assert_eq!(profile.resolve("other", 512), Ok(CONSERVATIVE_LANES));
    }

    #[test]
    fn lane_profile_resolve_zero_work_is_error() {
        let profile = LaneProfile::from_samples("gfx942", &[(2, 0.09)]).unwrap();
        assert_eq!(profile.resolve("gfx942", 0), Err(LaneError::NoWork));
    }

    #[test]
    fn lane_profile_matches_device() {
        let profile = LaneProfile::from_samples("gfx1201", &[(2, 0.09)]).unwrap();
        assert!(profile.matches_device("gfx1201"));
        assert!(!profile.matches_device("gfx1100"));
    }

    // --- resolve_with_profile ---

    #[test]
    fn resolve_with_profile_probe_uses_matching_profile() {
        let probe = LaneWidth::Probe {
            candidates: vec![1, 2, 4, 8],
            dispatches: 512,
        };
        let samples = vec![(1, 0.20), (2, 0.13), (4, 0.09), (8, 11.0)];
        let profile = LaneProfile::from_samples("gfx942", &samples).unwrap();
        // Probe + matching profile => profile wins (4)
        assert_eq!(
            resolve_with_profile(&probe, "gfx942", 512, Some(&profile)),
            Ok(4)
        );
        // Clamped
        assert_eq!(
            resolve_with_profile(&probe, "gfx942", 1, Some(&profile)),
            Ok(1)
        );
    }

    #[test]
    fn resolve_with_profile_probe_falls_back_when_profile_mismatched() {
        let probe = LaneWidth::Probe {
            candidates: vec![1, 2, 4],
            dispatches: 512,
        };
        let profile = LaneProfile::from_samples("gfx1201", &[(2, 0.09)]).unwrap();
        assert_eq!(
            resolve_with_profile(&probe, "gfx942", 512, Some(&profile)),
            Ok(CONSERVATIVE_LANES)
        );
    }

    #[test]
    fn resolve_with_profile_probe_falls_back_when_no_profile() {
        let probe = LaneWidth::Probe {
            candidates: vec![1, 2, 4],
            dispatches: 512,
        };
        assert_eq!(
            resolve_with_profile(&probe, "gfx942", 512, None),
            Ok(CONSERVATIVE_LANES)
        );
    }

    #[test]
    fn resolve_with_profile_non_probe_ignores_profile() {
        let profile = LaneProfile::from_samples("gfx942", &[(4, 0.09)]).unwrap();
        // Measured ignores profile and uses table/conservative
        assert_eq!(
            resolve_with_profile(&LaneWidth::Measured, "gfx1100", 512, Some(&profile)),
            Ok(4)
        );
        assert_eq!(
            resolve_with_profile(&LaneWidth::Measured, "gfx942", 512, Some(&profile)),
            Ok(CONSERVATIVE_LANES)
        );
        // Explicit ignores profile
        assert_eq!(
            resolve_with_profile(&LaneWidth::Explicit(nz(8)), "gfx942", 512, Some(&profile)),
            Ok(8)
        );
        // Single ignores profile
        assert_eq!(
            resolve_with_profile(&LaneWidth::Single, "gfx942", 512, Some(&profile)),
            Ok(1)
        );
    }

    #[test]
    fn resolve_with_profile_zero_work_is_error() {
        let probe = LaneWidth::Probe {
            candidates: vec![1, 2],
            dispatches: 512,
        };
        let profile = LaneProfile::from_samples("gfx942", &[(2, 0.09)]).unwrap();
        assert_eq!(
            resolve_with_profile(&probe, "gfx942", 0, Some(&profile)),
            Err(LaneError::NoWork)
        );
    }

    #[test]
    fn resolve_with_profile_too_wide_profile_is_error() {
        // Construct a profile with lanes above MAX_LANES manually (bypassing
        // from_samples guard) to exercise the TooWide path in
        // resolve_with_profile. This can happen if a caller constructs a
        // profile by hand.
        let profile = LaneProfile {
            device: "gfx942".to_string(),
            lanes: 64,
            samples: vec![(64, 0.01)],
        };
        let probe = LaneWidth::Probe {
            candidates: vec![1, 2],
            dispatches: 512,
        };
        assert_eq!(
            resolve_with_profile(&probe, "gfx942", 512, Some(&profile)),
            Err(LaneError::TooWide { requested: 64 })
        );
    }
}
