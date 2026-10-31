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

/// Default `max_active`: native windows up to 3 lanes in a step (3 MTP lanes
/// at K=2 keep 24 resident expert layers at 8 lanes; measured).
pub const DEFAULT_MAX_ACTIVE: usize = 3;

/// `HIPFIRE_EP_MTP_LANE_MAX_ACTIVE` (developer; the name predates PP), else
/// [`DEFAULT_MAX_ACTIVE`]. At least 1.
pub fn max_active() -> usize {
    hipfire_config::developer_var("HIPFIRE_EP_MTP_LANE_MAX_ACTIVE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_ACTIVE)
        .max(1)
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
}
