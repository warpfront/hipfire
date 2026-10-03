// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fail-closed launch-grid guard.
//!
//! gfx1201 (RDNA4) presents 16-bit y/z workgroup IDs. Raw module-launch
//! measurements show that 65536 workgroups execute all IDs 0..65535, while
//! larger launches succeed but wrap those IDs, leaving higher rows unwritten.
//! Reject unsafe geometry before graph capture or replay recording; do not
//! rely on the module-launch API to reject it. This policy does not bound x.
//!
//! The limit is applied only to architectures with measured evidence
//! ([`grid_yz_limit_for_arch`]); every other architecture keeps its historic
//! behaviour. A geometry over the limit is an error, never clamped or
//! truncated: silently shrinking a grid would drop rows.
//!
//! One policy table and one checker live here; every ingress (raw HIP launch,
//! dispatch recording funnels, PM4/AQL replay preparation) calls them.

use crate::error::{HipError, HipResult};

/// Largest y/z workgroup count that preserves every workgroup ID on gfx1201.
pub const GFX1201_MAX_GRID_YZ: u32 = 65_536;

/// `hipErrorInvalidValue`.
const HIP_ERROR_INVALID_VALUE: u32 = 1;

/// Per-architecture `gridDim.y`/`gridDim.z` ceiling, or `None` when the
/// architecture has no recorded ceiling and launches are left to HIP.
pub fn grid_yz_limit_for_arch(arch: &str) -> Option<u32> {
    arch.eq_ignore_ascii_case("gfx1201")
        .then_some(GFX1201_MAX_GRID_YZ)
}

/// Reject a launch geometry whose `y` or `z` axis exceeds `limit`.
///
/// `limit == None` accepts everything (no guard for this architecture).
pub fn check_launch_grid(grid: [u32; 3], limit: Option<u32>) -> HipResult<()> {
    let Some(limit) = limit else {
        return Ok(());
    };
    for (axis, name) in [(1usize, "y"), (2, "z")] {
        if grid[axis] > limit {
            return Err(HipError::new(
                HIP_ERROR_INVALID_VALUE,
                &format!(
                    "launch grid {grid:?}: grid.{name}={} exceeds the device grid.y/grid.z limit \
                     {limit}; refusing to record or launch (never truncated) — chunk the launch \
                     or fold the axis into grid.x in the caller",
                    grid[axis]
                ),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gfx1201_accepts_boundary_and_rejects_next() {
        let limit = grid_yz_limit_for_arch("gfx1201");
        assert!(check_launch_grid([1, 65_536, 1], limit).is_ok());
        assert!(check_launch_grid([1, 1, 65_536], limit).is_ok());
        assert!(check_launch_grid([u32::MAX, 65_536, 65_536], limit).is_ok());
        let y = check_launch_grid([8, 65_537, 1], limit).unwrap_err();
        assert_eq!(y.code, HIP_ERROR_INVALID_VALUE);
        let z = check_launch_grid([8, 1, 65_537], limit).unwrap_err();
        assert_eq!(z.code, HIP_ERROR_INVALID_VALUE);
    }

    #[test]
    fn other_architectures_are_unguarded() {
        for arch in ["gfx1100", "gfx1200", "gfx1151", "gfx942", "gfx1010"] {
            assert!(check_launch_grid([1, 70_000, 70_000], grid_yz_limit_for_arch(arch)).is_ok());
        }
    }
}
