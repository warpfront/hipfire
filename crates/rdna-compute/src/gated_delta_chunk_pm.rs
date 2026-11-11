// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PeaceMaker twins of the chunked-GDN prefill recurrence + gated RMSNorm
//! (`gated_delta_chunk_gate_wmma`, the NPU-arena `..._arena` entry and the opt-in inline-Q8
//! `gated_delta_chunk_gate_q8_wmma` pair) on exact gfx1151 and exact gfx1100
//! (`HIPFIRE_GDN_CHUNK_PM`, developer, default on; `=0` restores the hipcc kernels).
//!
//! Each twin is the shipped hipcc kernel itself: `kernels/gated_delta_chunk_*_pm_gfx11*.hip`
//! `#include`s the shared source under a per-arch symbol rename and nothing else, and
//! `scripts/pm-hip-twin.sh` (route A: lift identity through `emit::checked`, byte-identical re-emit,
//! static hazards, kernarg ABI equal to the incumbent) seals it into an embedded `.hxaco`. The
//! script aborts unless each twin's `.text` sha256 equals the `.text` of the incumbent compiled
//! with the identical hipcc argv, so the twin runs the same machine code: same kernargs, grid
//! `[value_heads]`, block 256, same LDS.
//!
//! Admission is the exact arch and the module that arch ships: gfx1151 all four (F32, NPU arena,
//! inline-Q8, inline-Q8 arena), gfx1100 the F32 kernel only (the arena and inline-Q8 arms are
//! gfx1151-only routes). The route that reaches this launcher excludes recorders and graph
//! capture (`tensor_ops::gated_delta_chunk_route`), so no recorded launch carries a twin symbol.

use crate::dispatch::Gpu;
use crate::tensor_ops::{
    gdn_chunk_kernel_name, gdn_chunk_module_name, gdn_chunk_source, GdnStateFormat,
};
use hip_bridge::{HipError, HipResult};
use std::sync::LazyLock;

/// One sealed twin module (one kernel).
#[derive(Debug, PartialEq, Eq)]
pub struct Twin {
    /// Exact `Gpu::arch` the module is built for.
    pub arch: &'static str,
    pub variant: Variant,
    pub module: &'static str,
    pub image: &'static [u8],
    pub image_sha256: &'static str,
    pub symbol: &'static str,
}

/// The four chunk kernels of the product (`tensor_ops::gdn_chunk_*_name`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// `gated_delta_chunk_gate_wmma` (F32 state, or a Q8-family state run on its F32 scratch).
    F32,
    /// `gated_delta_chunk_gate_wmma_arena` (NPU trunk BF16 arena `z`).
    F32Arena,
    /// `gated_delta_chunk_gate_q8_wmma` (inline plain-Q8 state).
    Q8,
    /// `gated_delta_chunk_gate_q8_wmma_arena`.
    Q8Arena,
}

impl Variant {
    /// Every variant.
    pub const ALL: [Self; 4] = [Self::F32, Self::F32Arena, Self::Q8, Self::Q8Arena];

    /// The `(format, inline, arena)` of the product launch that runs this variant.
    pub const fn product_args(self) -> (GdnStateFormat, bool, bool) {
        match self {
            Self::F32 => (GdnStateFormat::F32, false, false),
            Self::F32Arena => (GdnStateFormat::F32, false, true),
            Self::Q8 => (GdnStateFormat::Q8, true, false),
            Self::Q8Arena => (GdnStateFormat::Q8, true, true),
        }
    }

    /// The variant a product launch of `(format, inline, arena)` runs
    /// (the same split as [`gdn_chunk_module_name`]).
    pub const fn of(format: GdnStateFormat, inline: bool, arena: bool) -> Self {
        match (format, inline, arena) {
            (GdnStateFormat::Q8, true, false) => Self::Q8,
            (GdnStateFormat::Q8, true, true) => Self::Q8Arena,
            (_, _, false) => Self::F32,
            (_, _, true) => Self::F32Arena,
        }
    }
}

pub const GFX1151_F32: Twin = Twin {
    arch: "gfx1151",
    variant: Variant::F32,
    module: "gated_delta_chunk_pm_gfx1151",
    image: crate::kernels::GATED_DELTA_CHUNK_PM_GFX1151,
    image_sha256: crate::kernels::GATED_DELTA_CHUNK_PM_GFX1151_SHA256,
    symbol: "gated_delta_chunk_gate_wmma_pm_gfx1151",
};
pub const GFX1151_F32_ARENA: Twin = Twin {
    arch: "gfx1151",
    variant: Variant::F32Arena,
    module: "gated_delta_chunk_arena_pm_gfx1151",
    image: crate::kernels::GATED_DELTA_CHUNK_ARENA_PM_GFX1151,
    image_sha256: crate::kernels::GATED_DELTA_CHUNK_ARENA_PM_GFX1151_SHA256,
    symbol: "gated_delta_chunk_gate_wmma_arena_pm_gfx1151",
};
pub const GFX1151_Q8: Twin = Twin {
    arch: "gfx1151",
    variant: Variant::Q8,
    module: "gated_delta_chunk_q8_pm_gfx1151",
    image: crate::kernels::GATED_DELTA_CHUNK_Q8_PM_GFX1151,
    image_sha256: crate::kernels::GATED_DELTA_CHUNK_Q8_PM_GFX1151_SHA256,
    symbol: "gated_delta_chunk_gate_q8_wmma_pm_gfx1151",
};
pub const GFX1151_Q8_ARENA: Twin = Twin {
    arch: "gfx1151",
    variant: Variant::Q8Arena,
    module: "gated_delta_chunk_q8_arena_pm_gfx1151",
    image: crate::kernels::GATED_DELTA_CHUNK_Q8_ARENA_PM_GFX1151,
    image_sha256: crate::kernels::GATED_DELTA_CHUNK_Q8_ARENA_PM_GFX1151_SHA256,
    symbol: "gated_delta_chunk_gate_q8_wmma_arena_pm_gfx1151",
};
pub const GFX1100_F32: Twin = Twin {
    arch: "gfx1100",
    variant: Variant::F32,
    module: "gated_delta_chunk_pm_gfx1100",
    image: crate::kernels::GATED_DELTA_CHUNK_PM_GFX1100,
    image_sha256: crate::kernels::GATED_DELTA_CHUNK_PM_GFX1100_SHA256,
    symbol: "gated_delta_chunk_gate_wmma_pm_gfx1100",
};

/// Every sealed twin.
pub const TWINS: [&Twin; 5] = [
    &GFX1151_F32,
    &GFX1151_F32_ARENA,
    &GFX1151_Q8,
    &GFX1151_Q8_ARENA,
    &GFX1100_F32,
];

static FLAG: LazyLock<bool> =
    LazyLock::new(|| hipfire_config::developer_bool("HIPFIRE_GDN_CHUNK_PM", true));

/// The twin of an exact arch and variant, if one is sealed.
pub fn twin_for(arch: &str, variant: Variant) -> Option<&'static Twin> {
    TWINS
        .into_iter()
        .find(|t| t.arch == arch && t.variant == variant)
}

/// Why a launch keeps the hipcc kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnChunkPmRefusal {
    Disabled,
    /// No twin is sealed for this arch / variant (the default-on flag is not a refusal worth logging).
    NoTwin,
}

/// Pure admission: flag on and an exact-arch twin of the variant.
pub fn admission(
    flag: bool,
    arch: &str,
    variant: Variant,
) -> Result<&'static Twin, GdnChunkPmRefusal> {
    if !flag {
        return Err(GdnChunkPmRefusal::Disabled);
    }
    twin_for(arch, variant).ok_or(GdnChunkPmRefusal::NoTwin)
}

impl Gpu {
    /// Load the chunk kernel of a launch and return the symbol to launch: the sealed PM twin of this
    /// exact arch under `HIPFIRE_GDN_CHUNK_PM` (default on), else the hipcc incumbent
    /// (`gdn_chunk_{module,source,kernel}_name`). Kernargs, grid and block are the same either way.
    pub(crate) fn gdn_chunk_ensure(
        &mut self,
        format: GdnStateFormat,
        inline: bool,
        arena: bool,
    ) -> HipResult<&'static str> {
        let variant = Variant::of(format, inline, arena);
        let pm = admission(*FLAG, &self.arch, variant).is_ok();
        self.gdn_chunk_load(variant, pm)
    }

    /// Lab oracle entry: load the sealed twin (`pm`) or the hipcc incumbent of `variant`, whatever
    /// `HIPFIRE_GDN_CHUNK_PM` says, and return the symbol to launch (an error when `pm` and this arch
    /// has no twin of the variant).
    #[cfg(feature = "lab")]
    pub fn lab_gdn_chunk_load(&mut self, variant: Variant, pm: bool) -> HipResult<&'static str> {
        self.gdn_chunk_load(variant, pm)
    }

    fn gdn_chunk_load(&mut self, variant: Variant, pm: bool) -> HipResult<&'static str> {
        if pm {
            let twin = twin_for(&self.arch, variant).ok_or_else(|| {
                HipError::new(
                    0,
                    &format!(
                        "no gated_delta_chunk PM twin of {variant:?} on {}",
                        self.arch
                    ),
                )
            })?;
            self.ensure_embedded_kernel(twin.module, twin.image, twin.symbol)?;
            return Ok(twin.symbol);
        }
        let (format, inline, arena) = variant.product_args();
        self.ensure_kernel_public(
            gdn_chunk_module_name(format, inline, arena),
            gdn_chunk_source(format, inline, arena),
            gdn_chunk_kernel_name(format, inline, arena),
        )?;
        Ok(gdn_chunk_kernel_name(format, inline, arena))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn images_match_pinned_sha256() {
        for twin in TWINS {
            assert_eq!(
                format!("{:x}", Sha256::digest(twin.image)),
                twin.image_sha256,
                "{}",
                twin.module
            );
            assert_eq!(twin.image_sha256.len(), 64, "sha file must be bare");
        }
    }

    #[test]
    fn images_carry_their_symbol_and_no_other() {
        for twin in TWINS {
            let has = |sym: &str| twin.image.windows(sym.len()).any(|w| w == sym.as_bytes());
            assert!(
                has(twin.symbol),
                "{} missing from {}",
                twin.symbol,
                twin.module
            );
            for other in TWINS.iter().filter(|o| o.symbol != twin.symbol) {
                // Whole NUL-delimited names (a symbol is a substring of its `.kd` descriptor name).
                let pat: Vec<u8> = [&[0u8][..], other.symbol.as_bytes(), &[0u8][..]].concat();
                assert!(
                    !twin.image.windows(pat.len()).any(|w| w == pat.as_slice()),
                    "{} carries {}",
                    twin.module,
                    other.symbol
                );
            }
        }
    }

    #[test]
    fn twin_table_is_exact_arch_and_variant() {
        for twin in TWINS {
            assert_eq!(twin_for(twin.arch, twin.variant), Some(twin));
            assert!(twin.module.ends_with(twin.arch));
            assert!(twin.symbol.ends_with(twin.arch));
        }
        for arch in ["gfx1101", "gfx1150", "gfx1201", "gfx942", "gfx1151x", ""] {
            for variant in [
                Variant::F32,
                Variant::F32Arena,
                Variant::Q8,
                Variant::Q8Arena,
            ] {
                assert_eq!(twin_for(arch, variant), None, "{arch}");
            }
        }
        // gfx1100 only ships the F32 kernel.
        for variant in [Variant::F32Arena, Variant::Q8, Variant::Q8Arena] {
            assert_eq!(twin_for("gfx1100", variant), None);
        }
        let mut shas: Vec<&str> = TWINS.iter().map(|t| t.image_sha256).collect();
        shas.sort_unstable();
        shas.dedup();
        assert_eq!(shas.len(), TWINS.len());
    }

    #[test]
    fn admission_flag_and_arch() {
        assert_eq!(admission(true, "gfx1151", Variant::F32), Ok(&GFX1151_F32));
        assert_eq!(admission(true, "gfx1100", Variant::F32), Ok(&GFX1100_F32));
        assert_eq!(
            admission(false, "gfx1151", Variant::F32),
            Err(GdnChunkPmRefusal::Disabled)
        );
        assert_eq!(
            admission(true, "gfx1201", Variant::F32),
            Err(GdnChunkPmRefusal::NoTwin)
        );
        assert_eq!(
            admission(true, "gfx1100", Variant::Q8),
            Err(GdnChunkPmRefusal::NoTwin)
        );
    }

    /// The variant split mirrors the product's module / kernel names: each twin's symbol is the
    /// incumbent kernel name plus the `_pm_<arch>` suffix.
    #[test]
    fn variants_follow_the_product_selection() {
        for format in [
            GdnStateFormat::F32,
            GdnStateFormat::Q8,
            GdnStateFormat::Q8Ef,
        ] {
            for inline in [false, true] {
                for arena in [false, true] {
                    let want = format!(
                        "{}_pm_gfx1151",
                        gdn_chunk_kernel_name(format, inline, arena)
                    );
                    let twin = twin_for("gfx1151", Variant::of(format, inline, arena)).unwrap();
                    assert_eq!(
                        twin.symbol, want,
                        "{format:?} inline {inline} arena {arena}"
                    );
                }
            }
        }
    }
}
