// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1100 PeaceMaker twins of the generic gated-delta ops in [`crate::gated_delta_vector`]
//! (`HIPFIRE_GATED_DELTA_VECTOR_PM`, developer, default on; `=0` restores the hipcc kernels).
//!
//! One embedded module (`kernels/gated_delta_vector_pm_gfx1100.hxaco`, sealed by
//! `scripts/pm-hip-twin.sh`, route A: lift identity, hazards, SGPR >= 16, kernarg ABI equal to
//! the incumbent) holds two card-judged, bit-identical hipcc candidates:
//!
//! * [`RDA_SYMBOL`] replaces `gated_delta_net_vector_serial_f32` (`R2Rda-05-rs-tv8-prio`,
//!   DPP lockstep row reductions, raised issue priority on the pred butterfly). Same kernargs, grid `[n_heads, 128 / 8, 1]`, block 32, no LDS.
//! * [`CONV_SYMBOL`] replaces `causal_conv_depthwise_silu_rows_f32` (`R1Conv-04-rpw32-k4`,
//!   32 rows per workgroup). Same kernargs, grid `[ceil(C / 128), ceil(N / 32), 1]`, block 128, no
//!   LDS. Row slice 0 alone reads `hist_in` and writes `hist_out`, so it is correct both in place
//!   and not; the `rows_per_chunk` kernarg is ignored.
//!
//! Selection is made inside the generic launchers, so the arch crates are unchanged. The twins
//! admit the exact `gfx1100` arch only; every shape the incumbent accepts is admitted.
//! Profile timers keep the incumbent kernel names, so attribution buckets are unchanged.

use crate::dispatch::Gpu;
use std::sync::LazyLock;

pub const MODULE: &str = "gated_delta_vector_pm_gfx1100";
pub const RDA_SYMBOL: &str = "gated_delta_net_vector_serial_f32_pm_gfx1100";
pub const CONV_SYMBOL: &str = "causal_conv_depthwise_silu_rows_f32_pm_gfx1100";
pub const IMAGE: &[u8] = crate::kernels::GATED_DELTA_VECTOR_PM_GFX1100;
pub const IMAGE_SHA256: &str = crate::kernels::GATED_DELTA_VECTOR_PM_GFX1100_SHA256;
/// Head dimension (K == V) the RDA twin is built for.
pub const HEAD_DIM: usize = 128;
/// Value rows per RDA workgroup (`GDV_TV`).
pub const RDA_ROWS_PER_WG: usize = 8;
/// Conv channels per workgroup (block size).
pub const CONV_BLOCK: u32 = 128;
/// Conv rows per workgroup (`CCD_RPW`); only row slice 0 reads the history.
pub const CONV_ROWS_PER_WG: usize = 32;

static FLAG: LazyLock<bool> =
    LazyLock::new(|| hipfire_config::developer_bool("HIPFIRE_GATED_DELTA_VECTOR_PM", true));

/// Why a launch keeps the hipcc kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatedDeltaVectorPmRefusal {
    Disabled,
    /// Not exact gfx1100 (the default-on flag is not a refusal worth logging there).
    Arch,
    HeadDim,
    ConvKernel,
}

impl std::fmt::Display for GatedDeltaVectorPmRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Disabled => "HIPFIRE_GATED_DELTA_VECTOR_PM is off",
            Self::Arch => "exact gfx1100 only",
            Self::HeadDim => "RDA twin needs head_dim 128",
            Self::ConvKernel => "conv twin needs 2 <= ksize and ksize - 1 <= 32 history rows",
        })
    }
}

impl GatedDeltaVectorPmRefusal {
    fn log_once(self) {
        let index = match self {
            Self::Disabled | Self::Arch => return,
            Self::HeadDim => 0,
            Self::ConvKernel => 1,
        };
        static REFUSALS: [std::sync::Once; 2] = [std::sync::Once::new(), std::sync::Once::new()];
        REFUSALS[index].call_once(|| {
            eprintln!("[gated-delta-vector-pm] refused: {self}; keeping the hipcc kernel");
        });
    }
}

fn common(flag: bool, arch: &str) -> Result<(), GatedDeltaVectorPmRefusal> {
    if !flag {
        Err(GatedDeltaVectorPmRefusal::Disabled)
    } else if arch != "gfx1100" {
        Err(GatedDeltaVectorPmRefusal::Arch)
    } else {
        Ok(())
    }
}

/// Pure admission of the RDA twin: flag on, exact gfx1100, `head_dim == 128`.
pub fn rda_admission(
    flag: bool,
    arch: &str,
    head_dim: usize,
) -> Result<(), GatedDeltaVectorPmRefusal> {
    common(flag, arch)?;
    if head_dim != HEAD_DIM {
        return Err(GatedDeltaVectorPmRefusal::HeadDim);
    }
    Ok(())
}

/// Pure admission of the conv twin: flag on, exact gfx1100, and the `ksize - 1` history rows fit
/// in row slice 0 (always true for the launcher's `2..=8`).
pub fn conv_admission(
    flag: bool,
    arch: &str,
    ksize: usize,
) -> Result<(), GatedDeltaVectorPmRefusal> {
    common(flag, arch)?;
    if ksize < 2 || ksize - 1 > CONV_ROWS_PER_WG {
        return Err(GatedDeltaVectorPmRefusal::ConvKernel);
    }
    Ok(())
}

/// RDA twin grid: one workgroup per (head, 8 value rows).
pub fn rda_grid(n_heads: usize) -> [u32; 3] {
    [n_heads as u32, (HEAD_DIM / RDA_ROWS_PER_WG) as u32, 1]
}

/// Conv twin grid: 128 channels by 32 rows per workgroup, independent of `rows_per_chunk`.
pub fn conv_grid(n_channels: usize, n_rows: usize) -> [u32; 3] {
    [
        n_channels.div_ceil(CONV_BLOCK as usize) as u32,
        n_rows.div_ceil(CONV_ROWS_PER_WG) as u32,
        1,
    ]
}

impl Gpu {
    pub(crate) fn gated_delta_vector_pm_rda_admitted(&self, head_dim: usize) -> bool {
        match rda_admission(*FLAG, &self.arch, head_dim) {
            Ok(()) => true,
            Err(reason) => {
                reason.log_once();
                false
            }
        }
    }

    pub(crate) fn gated_delta_vector_pm_conv_admitted(&self, ksize: usize) -> bool {
        match conv_admission(*FLAG, &self.arch, ksize) {
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
    }

    #[test]
    fn image_carries_both_symbols() {
        for sym in [RDA_SYMBOL, CONV_SYMBOL] {
            assert!(
                IMAGE.windows(sym.len()).any(|w| w == sym.as_bytes()),
                "{sym} missing from the image"
            );
        }
    }

    #[test]
    fn rda_admission_arch_flag_and_head_dim() {
        assert_eq!(rda_admission(true, "gfx1100", 128), Ok(()));
        assert_eq!(
            rda_admission(false, "gfx1100", 128),
            Err(GatedDeltaVectorPmRefusal::Disabled)
        );
        for arch in ["gfx1101", "gfx1151", "gfx1201", "gfx942", "gfx1100x", ""] {
            assert_eq!(
                rda_admission(true, arch, 128),
                Err(GatedDeltaVectorPmRefusal::Arch),
                "{arch}"
            );
        }
        for hd in [0, 64, 127, 129, 256] {
            assert_eq!(
                rda_admission(true, "gfx1100", hd),
                Err(GatedDeltaVectorPmRefusal::HeadDim),
                "{hd}"
            );
        }
    }

    #[test]
    fn conv_admission_arch_flag_and_ksize() {
        for k in 2..=8 {
            assert_eq!(conv_admission(true, "gfx1100", k), Ok(()), "{k}");
        }
        assert_eq!(conv_admission(true, "gfx1100", 33), Ok(()));
        for k in [0, 1, 34, 1000] {
            assert_eq!(
                conv_admission(true, "gfx1100", k),
                Err(GatedDeltaVectorPmRefusal::ConvKernel),
                "{k}"
            );
        }
        assert_eq!(
            conv_admission(false, "gfx1100", 4),
            Err(GatedDeltaVectorPmRefusal::Disabled)
        );
        assert_eq!(
            conv_admission(true, "gfx1151", 4),
            Err(GatedDeltaVectorPmRefusal::Arch)
        );
    }

    #[test]
    fn grids_match_the_judged_geometry() {
        assert_eq!(rda_grid(48), [48, 16, 1]);
        assert_eq!(conv_grid(128, 32), [1, 1, 1]);
        assert_eq!(conv_grid(129, 33), [2, 2, 1]);
        assert_eq!(conv_grid(8192, 512), [64, 16, 1]);
    }
}
