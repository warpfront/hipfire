// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1100 PeaceMaker twin of the grouped MoE GEMM `gemm_mq4g256v2_moe_grouped_wmma_k2`
//! (`HIPFIRE_MOE_GROUPED_K2_PM`, developer, default on; `=0` restores the hipcc kernel).
//!
//! `kernels/gemm_mq4g256v2_moe_grouped_k2_pm_gfx1100.hxaco` is the card-judged, bit-identical
//! hipcc candidate `R1Moe-04-xlds-w4-t2` sealed by `scripts/pm-hip-twin.sh` (route A: lift
//! identity, hazards, SGPR >= 16, kernarg ABI equal to the incumbent): four waves (64 weight
//! rows) per 128-lane block share the expert run's X tile through double-buffered static LDS
//! (9216 B), and a workgroup at an even expert-local slot-tile index also computes the expert's
//! next tile (`tile_run` 2; the odd one returns). Same ten kernargs (`experts`, `tile_ids`,
//! `sorted_slot_index`, `X`, `Y`, `M`, `K`, `x_row_div`, `m_total`, `n_experts`), grid
//! `[ceil(M / 64), ceil(m_total / 16), 1]`, block 128, dynamic LDS 0.
//!
//! Selection is inside `gemm_mq4g256v2_moe_grouped_wmma_k2_impl`: the F32 entry only (the BF16
//! output entries keep hipcc), exact gfx1100, `M % 64 == 0`, `K % 256 == 0`. Profile timers keep
//! the incumbent kernel name.

use crate::dispatch::Gpu;
use std::sync::LazyLock;

pub const MODULE: &str = "gemm_mq4g256v2_moe_grouped_k2_pm_gfx1100";
pub const SYMBOL: &str = "gemm_mq4g256v2_moe_grouped_wmma_k2_pm_gfx1100";
pub const IMAGE: &[u8] = crate::kernels::GEMM_MQ4G256V2_MOE_GROUPED_K2_PM_GFX1100;
pub const IMAGE_SHA256: &str = crate::kernels::GEMM_MQ4G256V2_MOE_GROUPED_K2_PM_GFX1100_SHA256;
/// Weight rows per workgroup (four waves of one 16-row tile each).
pub const ROWS_PER_BLOCK: usize = 64;
/// Lanes per workgroup.
pub const BLOCK: u32 = 128;
/// Slot rows per tile.
pub const SLOT_TILE: usize = 16;
/// MQ4G256V2 group width: `K` must be a whole number of groups.
pub const K_GROUP: usize = 256;

static FLAG: LazyLock<bool> =
    LazyLock::new(|| hipfire_config::developer_bool("HIPFIRE_MOE_GROUPED_K2_PM", true));

/// Why a grouped MoE launch keeps the hipcc kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeGroupedK2PmRefusal {
    Disabled,
    /// Not exact gfx1100 (the default-on flag is not a refusal worth logging there).
    Arch,
    /// A BF16-output (`gfx11_entry`) or non-default row-tile entry: hipcc only.
    Entry,
    Rows,
    K,
}

impl std::fmt::Display for MoeGroupedK2PmRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Disabled => "HIPFIRE_MOE_GROUPED_K2_PM is off",
            Self::Arch => "exact gfx1100 only",
            Self::Entry => "the F32 16-row entry only",
            Self::Rows => "needs M % 64 == 0",
            Self::K => "needs K % 256 == 0",
        })
    }
}

impl MoeGroupedK2PmRefusal {
    fn log_once(self) {
        let index = match self {
            Self::Disabled | Self::Arch | Self::Entry => return,
            Self::Rows => 0,
            Self::K => 1,
        };
        static REFUSALS: [std::sync::Once; 2] = [std::sync::Once::new(), std::sync::Once::new()];
        REFUSALS[index].call_once(|| {
            eprintln!("[moe-grouped-k2-pm] refused: {self}; keeping the hipcc kernel");
        });
    }
}

/// Pure admission: flag on, exact gfx1100, the F32 entry (`gfx11_entry` none, 16-row tile
/// launcher), `M % 64 == 0` and `K % 256 == 0`.
pub fn admission(
    flag: bool,
    arch: &str,
    has_gfx11_entry: bool,
    rows_per_block: usize,
    m: usize,
    k: usize,
) -> Result<(), MoeGroupedK2PmRefusal> {
    if !flag {
        Err(MoeGroupedK2PmRefusal::Disabled)
    } else if arch != "gfx1100" {
        Err(MoeGroupedK2PmRefusal::Arch)
    } else if has_gfx11_entry || rows_per_block != SLOT_TILE {
        Err(MoeGroupedK2PmRefusal::Entry)
    } else if m == 0 || !m.is_multiple_of(ROWS_PER_BLOCK) {
        Err(MoeGroupedK2PmRefusal::Rows)
    } else if k == 0 || !k.is_multiple_of(K_GROUP) {
        Err(MoeGroupedK2PmRefusal::K)
    } else {
        Ok(())
    }
}

/// Twin grid: 64 weight rows by one 16-slot tile per workgroup.
pub fn grid(m: usize, m_total: usize) -> [u32; 3] {
    [
        m.div_ceil(ROWS_PER_BLOCK) as u32,
        m_total.div_ceil(SLOT_TILE) as u32,
        1,
    ]
}

impl Gpu {
    pub(crate) fn moe_grouped_k2_pm_admitted(
        &self,
        has_gfx11_entry: bool,
        rows_per_block: usize,
        m: usize,
        k: usize,
    ) -> bool {
        match admission(*FLAG, &self.arch, has_gfx11_entry, rows_per_block, m, k) {
            Ok(()) => true,
            Err(reason) => {
                reason.log_once();
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn image_matches_pinned_sha256() {
        assert_eq!(format!("{:x}", Sha256::digest(IMAGE)), IMAGE_SHA256.trim());
        assert_eq!(IMAGE_SHA256, IMAGE_SHA256.trim(), "sha file must be bare");
        assert!(IMAGE.windows(SYMBOL.len()).any(|w| w == SYMBOL.as_bytes()));
    }

    #[test]
    fn admission_arch_entry_and_shape() {
        assert_eq!(admission(true, "gfx1100", false, 16, 512, 2048), Ok(()));
        assert_eq!(admission(true, "gfx1100", false, 16, 64, 256), Ok(()));
        assert_eq!(
            admission(false, "gfx1100", false, 16, 512, 2048),
            Err(MoeGroupedK2PmRefusal::Disabled)
        );
        for arch in ["gfx1101", "gfx1151", "gfx1201", "gfx942", ""] {
            assert_eq!(
                admission(true, arch, false, 16, 512, 2048),
                Err(MoeGroupedK2PmRefusal::Arch),
                "{arch}"
            );
        }
        assert_eq!(
            admission(true, "gfx1100", true, 16, 512, 2048),
            Err(MoeGroupedK2PmRefusal::Entry)
        );
        assert_eq!(
            admission(true, "gfx1100", false, 64, 512, 2048),
            Err(MoeGroupedK2PmRefusal::Entry)
        );
        for m in [0, 16, 32, 96, 513] {
            assert_eq!(
                admission(true, "gfx1100", false, 16, m, 2048),
                Err(MoeGroupedK2PmRefusal::Rows),
                "M {m}"
            );
        }
        for k in [0, 128, 255, 257, 1000] {
            assert_eq!(
                admission(true, "gfx1100", false, 16, 512, k),
                Err(MoeGroupedK2PmRefusal::K),
                "K {k}"
            );
        }
    }

    #[test]
    fn grid_matches_the_judged_geometry() {
        // gu512 / dn512 style shapes: 64 rows per block, 16 slots per tile.
        assert_eq!(grid(512, 3072), [8, 192, 1]);
        assert_eq!(grid(2048, 16), [32, 1, 1]);
        assert_eq!(grid(64, 17), [1, 2, 1]);
    }
}
