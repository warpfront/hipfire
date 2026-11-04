// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The step-level concurrency switch of multi-card native-MTP lanes, one
//! topology-neutral policy shared by EP ([`crate::ep_mesh_serve`]) and PP.
//!
//! Native MTP windows win at low concurrency (measured EP2: C1-C4) and lose
//! to plain AR rows once many lanes already fill the verify rows. Each step
//! the driver decides from the number of decoding lanes it carries:
//! native while `active <= max_active`, AR from `active >= max_active + 2`,
//! and in between the previous decision stands (hysteresis, so a count
//! hovering around the edge does not flap). An AR window of a spec lane is
//! the protocol's one-token window (`remaining = 1`: `ArSynced`, the head
//! stays aligned), so a lane crossing in either direction loses no state.
//!
//! The same count also sizes the mixed lane store: only the first
//! [`spec_lane_count`] slots carry MTP state (target ring, head); the rest are
//! AR-only lanes, so the reservation (and the resident-expert cost) is that
//! of `max_active` MTP lanes, not of every lane.

/// Default `max_active`: native windows up to 8 lanes in a step (native C8; `8` is the lane store
/// size of the serve route, so every slot carries MTP state). `HIPFIRE_EP_MTP_LANE_MAX_ACTIVE=3`
/// restores the previous switch (3 MTP lanes at K=2 keep 24 resident expert layers at 8 lanes;
/// measured), which forced AR windows from 5 active lanes up.
pub const DEFAULT_MAX_ACTIVE: usize = 8;

/// `HIPFIRE_EP_MTP_LANE_NATIVE_ALL=1` (developer; default off): every lane slot
/// carries MTP state and no step is forced to AR windows by the lane count.
/// The per-lane floor (`mtp_lanes::LaneSpecPolicy`, priced by the store's
/// `LaneCostModel` in AR-row units) then decides per window whether a native
/// window beats an AR row, so the step-level switch above stops overriding it.
/// It reserves MTP state (target ring, head) for every lane, which costs
/// resident experts, and it makes slot 7 a native lane, which the 8-lane
/// sampled oracle currently flags (lane 7, `PARITY.md` 4.9 (j)): not a default
/// until that is explained and the wide verify has landed.
pub const NATIVE_ALL_ENV: &str = "HIPFIRE_EP_MTP_LANE_NATIVE_ALL";

/// The `max_active` [`NATIVE_ALL_ENV`] stands for: above any lane count a store
/// can have, and small enough that `max_active + 2` cannot overflow.
pub const NATIVE_ALL_MAX_ACTIVE: usize = 1 << 16;

/// Lane draft-depth cap when [`NATIVE_ALL_ENV`] is on and `HIPFIRE_EP_MTP_LANE_K`
/// is unset: the per-lane depth chooser (`batched_depth` over each lane's
/// measured per-depth agreement and priced windows) picks the depth up to this,
/// so a code lane (per-position acceptance 1.00/.955/.914) runs deep and a
/// story lane (.67/.57/.62) stops at 2-3 inside the same cap.
pub const NATIVE_ALL_LANE_K: usize = 5;

/// The shipped lane draft-depth cap (`HIPFIRE_EP_MTP_LANE_K` unset, flag off).
/// 5 with the lane cost profile of `mtp_lanes` (gfx1201: per-request fit, even
/// split of one 8-row group over the step's lanes, so deeper K never spills at
/// C2..C4); `HIPFIRE_EP_MTP_LANE_K=2` restores the previous cap (use
/// `HIPFIRE_QWEN4_LANE_COST=0` with it for the previous lane floor).
pub const DEFAULT_LANE_K: usize = 5;

/// [`NATIVE_ALL_ENV`] on.
pub fn native_all() -> bool {
    hipfire_config::developer_bool(NATIVE_ALL_ENV, false)
}

/// `max_active` for a flag state and an explicit `HIPFIRE_EP_MTP_LANE_MAX_ACTIVE`
/// value: [`NATIVE_ALL_MAX_ACTIVE`] when the flag is on (the explicit value is
/// then ignored), else the explicit value or [`DEFAULT_MAX_ACTIVE`], at least 1.
pub fn max_active_for(native_all: bool, explicit: Option<usize>) -> usize {
    if native_all {
        return NATIVE_ALL_MAX_ACTIVE;
    }
    explicit.unwrap_or(DEFAULT_MAX_ACTIVE).max(1)
}

/// The lane draft-depth cap for a flag state: [`NATIVE_ALL_LANE_K`] on,
/// [`DEFAULT_LANE_K`] off.
pub fn default_lane_k_for(native_all: bool) -> usize {
    if native_all {
        NATIVE_ALL_LANE_K
    } else {
        DEFAULT_LANE_K
    }
}

/// [`default_lane_k_for`] under the process environment.
pub fn default_lane_k() -> usize {
    default_lane_k_for(native_all())
}

/// `HIPFIRE_EP_MTP_LANE_MAX_ACTIVE` (developer; the name predates PP), else
/// [`DEFAULT_MAX_ACTIVE`]; unbounded under [`NATIVE_ALL_ENV`]. At least 1.
pub fn max_active() -> usize {
    max_active_for(
        native_all(),
        hipfire_config::developer_var("HIPFIRE_EP_MTP_LANE_MAX_ACTIVE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok()),
    )
}

/// The window mode of a step carrying `active` decoding lanes after a step
/// whose mode was `prev_native`: `true` = native windows, `false` = AR.
pub fn native_mode(active: usize, prev_native: bool, max_active: usize) -> bool {
    if active <= max_active {
        true
    } else if active >= max_active + 2 {
        false
    } else {
        prev_native
    }
}

/// MTP-carrying slots of a store of `max_lanes` lanes: `max_active`, never
/// more than the store has. Slots at or above it are AR-only lanes.
pub fn spec_lane_count(max_lanes: usize, max_active: usize) -> usize {
    max_active.min(max_lanes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_up_to_max_active_ar_from_two_above_and_held_between() {
        assert!(native_mode(1, false, 4));
        assert!(native_mode(4, false, 4));
        assert!(!native_mode(6, true, 4));
        assert!(!native_mode(8, true, 4));
        // 5 lanes: the previous decision stands (no flapping).
        assert!(native_mode(5, true, 4));
        assert!(!native_mode(5, false, 4));
    }

    #[test]
    fn a_count_hovering_at_the_edge_does_not_flap() {
        let mut mode = true;
        let mut flips = 0;
        for active in [4, 5, 4, 5, 5, 4, 5, 4] {
            let next = native_mode(active, mode, 4);
            flips += usize::from(next != mode);
            mode = next;
        }
        assert_eq!(flips, 0);
    }

    #[test]
    fn spec_lanes_are_capped_by_the_store() {
        assert_eq!(spec_lane_count(8, 4), 4);
        assert_eq!(spec_lane_count(2, 4), 2);
    }

    #[test]
    fn native_all_lifts_max_active_and_raises_the_depth_cap() {
        assert_eq!(max_active_for(false, None), DEFAULT_MAX_ACTIVE);
        assert_eq!(max_active_for(false, Some(0)), 1);
        assert_eq!(max_active_for(false, Some(6)), 6);
        // The flag wins over an explicit cap.
        assert_eq!(max_active_for(true, Some(2)), NATIVE_ALL_MAX_ACTIVE);
        assert_eq!(default_lane_k_for(false), DEFAULT_LANE_K);
        assert_eq!(default_lane_k_for(true), NATIVE_ALL_LANE_K);
        assert_eq!(DEFAULT_LANE_K, 5, "the shipped default");
    }

    /// Native C8 is the default: a step of up to 8 lanes runs native windows whatever the previous
    /// mode was, and every slot of an 8-lane store carries MTP state; `HIPFIRE_EP_MTP_LANE_MAX_ACTIVE=3`
    /// restores the previous switch (AR from 5 lanes up).
    #[test]
    fn native_c8_is_the_default_and_the_explicit_value_restores_the_old_switch() {
        assert_eq!(DEFAULT_MAX_ACTIVE, 8);
        let cap = max_active_for(false, None);
        for lanes in 1..=8 {
            assert!(native_mode(lanes, false, cap), "{lanes}");
            assert!(native_mode(lanes, true, cap), "{lanes}");
        }
        assert_eq!(spec_lane_count(8, cap), 8);
        assert!(!native_mode(10, true, cap));
        let previous = max_active_for(false, Some(3));
        assert_eq!(previous, 3);
        assert!(native_mode(3, false, previous));
        assert!(!native_mode(5, false, previous));
        assert!(!native_mode(8, true, previous));
        assert_eq!(spec_lane_count(8, previous), 3);
    }

    #[test]
    fn native_all_runs_native_windows_at_every_lane_count() {
        let cap = max_active_for(true, None);
        for lanes in 1..=64 {
            assert!(native_mode(lanes, false, cap), "{lanes}");
            assert!(native_mode(lanes, true, cap), "{lanes}");
            assert_eq!(
                spec_lane_count(lanes, cap),
                lanes,
                "every slot is an MTP slot"
            );
        }
        // `max_active + 2` cannot overflow
        assert!(native_mode(cap, false, cap));
    }
}
