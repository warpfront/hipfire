// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1100 / gfx1201 PeaceMaker twins of the generic gated-delta ops in [`crate::gated_delta_vector`]
//! (`HIPFIRE_GATED_DELTA_VECTOR_PM`, developer, default on; `=0` restores the hipcc kernels).
//!
//! One embedded module per arch (`kernels/gated_delta_vector_pm_gfx1100.hxaco`,
//! `kernels/gated_delta_vector_pm_gfx1201.hxaco`, sealed by `scripts/pm-hip-twin.sh`, route A: lift
//! identity, hazards, SGPR >= 16, kernarg ABI equal to the incumbent, `.text` equal to the card-judged
//! candidate) holds two card-judged, bit-identical hipcc candidates:
//!
//! * `rda_symbol` replaces `gated_delta_net_vector_serial_f32` (`R2Rda-05-rs-tv8-prio`,
//!   DPP lockstep row reductions, raised issue priority on the pred butterfly). Same kernargs, grid `[n_heads, 128 / 8, 1]`, block 32, no LDS.
//! * `conv_symbol` replaces `causal_conv_depthwise_silu_rows_f32` (`R1Conv-04-rpw32-k4`,
//!   32 rows per workgroup). Same kernargs, grid `[ceil(C / 128), ceil(N / 32), 1]`, block 128, no
//!   LDS. Row slice 0 alone reads `hist_in` and writes `hist_out`, so it is correct both in place
//!   and not; the `rows_per_chunk` kernarg is ignored.
//!
//! Selection is made inside the generic launchers, so the arch crates are unchanged. Admission is
//! the exact arch plus the shapes measured on the card:
//!
//! * `gfx1100` (7900 XTX judge): every shape the incumbent accepts (RDA `head_dim == 128`, conv
//!   `2 <= ksize`).
//! * `gfx1201` (R9700 judge): RDA `head_dim == 128` (every head count, token count, decay axis,
//!   `qk_group`, rounding, baked or device slot / token count measured bit-identical and faster);
//!   conv `2 <= ksize <= 8` with the in-place history (Raon's single-chunk launch). The conv twin
//!   on a separate `hist_out` measured no faster than the incumbent (ratio 1.00 at N >= 2048,
//!   0.3 to 0.5% slower), so that launch keeps hipcc.
//!
//! Profile timers keep the incumbent kernel names, so attribution buckets are unchanged.

use crate::dispatch::Gpu;
use std::sync::LazyLock;

/// One arch's sealed twin module.
#[derive(Debug, PartialEq, Eq)]
pub struct Twin {
    /// Exact `Gpu::arch` the module is built and judged for.
    pub arch: &'static str,
    pub module: &'static str,
    pub image: &'static [u8],
    pub image_sha256: &'static str,
    pub rda_symbol: &'static str,
    pub conv_symbol: &'static str,
}

pub const GFX1100: Twin = Twin {
    arch: "gfx1100",
    module: "gated_delta_vector_pm_gfx1100",
    image: crate::kernels::GATED_DELTA_VECTOR_PM_GFX1100,
    image_sha256: crate::kernels::GATED_DELTA_VECTOR_PM_GFX1100_SHA256,
    rda_symbol: "gated_delta_net_vector_serial_f32_pm_gfx1100",
    conv_symbol: "causal_conv_depthwise_silu_rows_f32_pm_gfx1100",
};

pub const GFX1201: Twin = Twin {
    arch: "gfx1201",
    module: "gated_delta_vector_pm_gfx1201",
    image: crate::kernels::GATED_DELTA_VECTOR_PM_GFX1201,
    image_sha256: crate::kernels::GATED_DELTA_VECTOR_PM_GFX1201_SHA256,
    rda_symbol: "gated_delta_net_vector_serial_f32_pm_gfx1201",
    conv_symbol: "causal_conv_depthwise_silu_rows_f32_pm_gfx1201",
};

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

/// The twin module of an exact arch, if one is sealed.
pub fn twin_for_arch(arch: &str) -> Option<&'static Twin> {
    match arch {
        "gfx1100" => Some(&GFX1100),
        "gfx1201" => Some(&GFX1201),
        _ => None,
    }
}

/// The launch parameters the conv admission looks at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvShape {
    pub ksize: usize,
    /// `hist_in == hist_out` (one row chunk at the incumbent).
    pub in_place: bool,
}

/// Why a launch keeps the hipcc kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatedDeltaVectorPmRefusal {
    Disabled,
    /// No twin is sealed for this arch (the default-on flag is not a refusal worth logging there).
    Arch,
    HeadDim,
    ConvKernel,
    /// gfx1201 conv twin: only the in-place history launch (a separate `hist_out` is no faster).
    ConvNotInPlace,
}

impl std::fmt::Display for GatedDeltaVectorPmRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Disabled => "HIPFIRE_GATED_DELTA_VECTOR_PM is off",
            Self::Arch => "exact gfx1100 / gfx1201 only",
            Self::HeadDim => "RDA twin needs head_dim 128",
            Self::ConvKernel => "conv twin needs 2 <= ksize and ksize - 1 <= 32 history rows",
            Self::ConvNotInPlace => "gfx1201 conv twin needs the history in place",
        })
    }
}

impl GatedDeltaVectorPmRefusal {
    fn log_once(self) {
        let index = match self {
            Self::Disabled | Self::Arch => return,
            Self::HeadDim => 0,
            Self::ConvKernel => 1,
            Self::ConvNotInPlace => 2,
        };
        static REFUSALS: [std::sync::Once; 3] = [
            std::sync::Once::new(),
            std::sync::Once::new(),
            std::sync::Once::new(),
        ];
        REFUSALS[index].call_once(|| {
            eprintln!("[gated-delta-vector-pm] refused: {self}; keeping the hipcc kernel");
        });
    }
}

fn common(flag: bool, arch: &str) -> Result<&'static Twin, GatedDeltaVectorPmRefusal> {
    if !flag {
        return Err(GatedDeltaVectorPmRefusal::Disabled);
    }
    twin_for_arch(arch).ok_or(GatedDeltaVectorPmRefusal::Arch)
}

/// Pure admission of the RDA twin: flag on, an exact-arch twin, `head_dim == 128`.
pub fn rda_admission(
    flag: bool,
    arch: &str,
    head_dim: usize,
) -> Result<&'static Twin, GatedDeltaVectorPmRefusal> {
    let twin = common(flag, arch)?;
    if head_dim != HEAD_DIM {
        return Err(GatedDeltaVectorPmRefusal::HeadDim);
    }
    Ok(twin)
}

/// Pure admission of the conv twin: flag on, an exact-arch twin, and the `ksize - 1` history rows
/// fit in row slice 0 (always true for the launcher's `2..=8`); on gfx1201 also the in-place history.
pub fn conv_admission(
    flag: bool,
    arch: &str,
    shape: ConvShape,
) -> Result<&'static Twin, GatedDeltaVectorPmRefusal> {
    let twin = common(flag, arch)?;
    if shape.ksize < 2 || shape.ksize - 1 > CONV_ROWS_PER_WG {
        return Err(GatedDeltaVectorPmRefusal::ConvKernel);
    }
    if twin.arch == "gfx1201" && !shape.in_place {
        return Err(GatedDeltaVectorPmRefusal::ConvNotInPlace);
    }
    Ok(twin)
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
    pub(crate) fn gated_delta_vector_pm_rda_admitted(
        &self,
        head_dim: usize,
    ) -> Option<&'static Twin> {
        match rda_admission(*FLAG, &self.arch, head_dim) {
            Ok(twin) => Some(twin),
            Err(reason) => {
                reason.log_once();
                None
            }
        }
    }

    pub(crate) fn gated_delta_vector_pm_conv_admitted(
        &self,
        shape: ConvShape,
    ) -> Option<&'static Twin> {
        match conv_admission(*FLAG, &self.arch, shape) {
            Ok(twin) => Some(twin),
            Err(reason) => {
                reason.log_once();
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    const TWINS: [&Twin; 2] = [&GFX1100, &GFX1201];

    fn conv(ksize: usize, in_place: bool) -> ConvShape {
        ConvShape { ksize, in_place }
    }

    #[test]
    fn images_match_pinned_sha256() {
        for twin in TWINS {
            assert_eq!(
                format!("{:x}", Sha256::digest(twin.image)),
                twin.image_sha256.trim(),
                "{}",
                twin.arch
            );
            assert_eq!(
                twin.image_sha256,
                twin.image_sha256.trim(),
                "sha file must be bare"
            );
        }
    }

    #[test]
    fn images_carry_both_symbols() {
        for twin in TWINS {
            for sym in [twin.rda_symbol, twin.conv_symbol] {
                assert!(
                    twin.image.windows(sym.len()).any(|w| w == sym.as_bytes()),
                    "{sym} missing from the {} image",
                    twin.arch
                );
            }
        }
    }

    #[test]
    fn twin_table_is_exact_arch() {
        for twin in TWINS {
            assert_eq!(twin_for_arch(twin.arch), Some(twin));
            assert!(twin.module.ends_with(twin.arch));
            assert!(twin.rda_symbol.ends_with(twin.arch));
            assert!(twin.conv_symbol.ends_with(twin.arch));
        }
        for arch in ["gfx1101", "gfx1151", "gfx942", "gfx1100x", "gfx12", ""] {
            assert_eq!(twin_for_arch(arch), None, "{arch}");
        }
        assert_ne!(GFX1100.image_sha256, GFX1201.image_sha256);
    }

    #[test]
    fn rda_admission_arch_flag_and_head_dim() {
        assert_eq!(rda_admission(true, "gfx1100", 128), Ok(&GFX1100));
        assert_eq!(rda_admission(true, "gfx1201", 128), Ok(&GFX1201));
        for arch in ["gfx1100", "gfx1201"] {
            assert_eq!(
                rda_admission(false, arch, 128),
                Err(GatedDeltaVectorPmRefusal::Disabled)
            );
            for hd in [0, 64, 127, 129, 256] {
                assert_eq!(
                    rda_admission(true, arch, hd),
                    Err(GatedDeltaVectorPmRefusal::HeadDim),
                    "{arch} {hd}"
                );
            }
        }
        for arch in ["gfx1101", "gfx1151", "gfx942", "gfx1100x", ""] {
            assert_eq!(
                rda_admission(true, arch, 128),
                Err(GatedDeltaVectorPmRefusal::Arch),
                "{arch}"
            );
        }
    }

    #[test]
    fn conv_admission_arch_flag_and_ksize() {
        for k in 2..=8 {
            assert_eq!(
                conv_admission(true, "gfx1100", conv(k, true)),
                Ok(&GFX1100),
                "{k}"
            );
            assert_eq!(
                conv_admission(true, "gfx1100", conv(k, false)),
                Ok(&GFX1100),
                "{k}"
            );
        }
        assert_eq!(
            conv_admission(true, "gfx1100", conv(33, true)),
            Ok(&GFX1100)
        );
        for arch in ["gfx1100", "gfx1201"] {
            for k in [0, 1, 34, 1000] {
                assert_eq!(
                    conv_admission(true, arch, conv(k, true)),
                    Err(GatedDeltaVectorPmRefusal::ConvKernel),
                    "{arch} {k}"
                );
            }
            assert_eq!(
                conv_admission(false, arch, conv(4, true)),
                Err(GatedDeltaVectorPmRefusal::Disabled)
            );
        }
        assert_eq!(
            conv_admission(true, "gfx1151", conv(4, true)),
            Err(GatedDeltaVectorPmRefusal::Arch)
        );
    }

    #[test]
    fn conv_gfx1201_admits_in_place_only() {
        for k in 2..=8 {
            assert_eq!(
                conv_admission(true, "gfx1201", conv(k, true)),
                Ok(&GFX1201),
                "{k}"
            );
            assert_eq!(
                conv_admission(true, "gfx1201", conv(k, false)),
                Err(GatedDeltaVectorPmRefusal::ConvNotInPlace),
                "{k}"
            );
        }
    }

    #[test]
    fn grids_match_the_judged_geometry() {
        assert_eq!(rda_grid(48), [48, 16, 1]);
        assert_eq!(rda_grid(40), [40, 16, 1]);
        assert_eq!(conv_grid(128, 32), [1, 1, 1]);
        assert_eq!(conv_grid(129, 33), [2, 2, 1]);
        assert_eq!(conv_grid(8192, 512), [64, 16, 1]);
        assert_eq!(conv_grid(5120, 512), [40, 16, 1]);
    }
}
