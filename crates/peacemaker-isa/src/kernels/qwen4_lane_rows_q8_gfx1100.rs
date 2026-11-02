// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Actual-object-derived gfx1100 Q8 fixed8 row tiles for total R=1..64.
//!
//! Reference ELFs under `row-exact-wide/xtx-singleton-reference/`:
//!
//! * `q8-scalar.co`, `gemv_q8_0`, SHA256
//!   `5462cc8b601c3d3f227046fefeeb2a4ac5c638df8516f9b7c341157e2637f762`.
//!   One +0 chain, ascending 32-element blocks, `fma(x,dq,acc)` in both
//!   full tiles and tails. The k2560-named export covers every positive
//!   K>1536 divisible by32, not just2560; x2 is restricted to K=2560.
//! * `q8-hc.co`, `gemv_q8_0_k8`, SHA256
//!   `c9343949ec0225b8206d82e0b69a80253f5de5e75c14074d72a16c70a9a8f7d5`.
//!   Eight fixed contiguous K/8 partitions, K>0 and K%256=0. Their lane0
//!   partials fold `((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7))`.
//! * `q8-wide-acquisition/gemv_q8_0_wide.gfx1100.co`, `gemv_q8_0_wide`,
//!   SHA256 `4de9f1a0a0c6f7ab9e7030ae8cfc920da48c2b4e70b9344c8267b855291bfe0e`.
//!   Source SHA256
//!   `3691845ced9c39fb2aae345b81c49566d499239533bea447944e80815da7af0f`;
//!   compiler flags/header pins are in that directory's `acquisition.json`
//!   and `include-sha256.json`. Main quads use four +0 chains and
//!   `fma(dq,x,acc)`; remaining blocks use separate `mul(x,dq)` then
//!   `add(acc,product)` into chain `block&3`. Fold `(a3+a2)+(a1+a0)`.
//!   K320 and K640 use this same recipe, never Halo's staged special cases.
//!
//! All three objects dequantize with `fma_mix(d16,float(int8),-0)` and reduce
//! with swizzle16 followed by own-clamped bpermute8/4/2/1, own+other each step.
//! HC activation transcribes q8-hc.co c834..ca94: bit-exact BF16 round trip,
//! `mul(scale,value)`, BF16 round trip, the actual VALU exp/div expansion,
//! then BF16 round trip. The shared `bf16_round`/`silu` helpers express those
//! same nodes; no approximate exp, reciprocal-only divide or Halo SALU
//! assumption is substituted. The division carry remains live through
//! `div_fmas` without an intervening row-mask comparison.
//!
//! Only integer scheduling, load staging and proven identical arithmetic
//! helpers are shared with the Halo module. This module explicitly selects
//! architecture, arithmetic recipe, K domain and symbol; it does not retag
//! Halo's K320/K640 arithmetic. Existing Halo entry generation takes the
//! unchanged default branch of the shared emitter.
//!
//! Ordinary ABI36: A@0,x@8,y@16,M@24,K@28,R@32. x2 ABI56:
//! A0@0,A1@8,x@16,y0@24,y1@32,M0@40,M1@44,K@48,R@52. HC ABI44:
//! A@0,x@8,y@16,M@24,K@28,R@32,scale@36,hc_act@40.
//! Grid y is ceil(R/8); ordinary grid x is ceil(M/8), x2 sums both matrix
//! ceilings, and HC uses ceil(M/2). Blocks are256 ordinary/512 HC; static
//! LDS is8192/16384 bytes respectively, dynamic LDS0. Storage is always eight
//! token rows, with80-VGPR/56-SGPR plan ceilings, never R-sized accumulators.
//! Grid-y rebases x/y once while preserving full K/M row strides. Dead output
//! waves still stage live x rows and reach every barrier; inactive token rows
//! have no floating arithmetic, loads or stores. Invalid R/K domains and
//! wholly dead groups refuse uniformly before producer/barrier phases.
//!
//! These are source/object recipe pins, not execution or capability evidence.
//! Independent object-vs-tile CPU and GPU byte gates, HC exceptional inputs,
//! object descriptor FP modes, runtime registration and admission remain
//! integration/qualification responsibilities.

use super::qwen4_lane_rows_q8::{emit_rowtiles_for, RowtileDomain, RowtileRecipe, RowtileTarget};
use crate::{Arch, Emitted};

const TARGETS: [RowtileTarget; 5] = [
    RowtileTarget {
        arch: Arch::Gfx1100,
        symbol: "gemv_q8_0_k8_rows_t8_pm_gfx1100",
        recipe: RowtileRecipe::Hc,
        domain: RowtileDomain::EightPartitions,
    },
    RowtileTarget {
        arch: Arch::Gfx1100,
        symbol: "gemv_q8_0_k320_rows_t8_pm_gfx1100",
        recipe: RowtileRecipe::Small,
        domain: RowtileDomain::Exact(320),
    },
    RowtileTarget {
        arch: Arch::Gfx1100,
        symbol: "gemv_q8_0_k2560_rows_t8_pm_gfx1100",
        recipe: RowtileRecipe::Plain,
        domain: RowtileDomain::Large,
    },
    RowtileTarget {
        arch: Arch::Gfx1100,
        symbol: "gemv_q8_0_k2560_rows_x2_t8_pm_gfx1100",
        recipe: RowtileRecipe::PlainX2,
        domain: RowtileDomain::Exact(2560),
    },
    RowtileTarget {
        arch: Arch::Gfx1100,
        symbol: "gemv_q8_0_wide_rows_t8_pm_gfx1100",
        recipe: RowtileRecipe::Small,
        domain: RowtileDomain::Small,
    },
];

/// Emit HC k8, k320, generic large-K, K2560 x2, then small-K wide.
///
/// Each entry has fixed8 physical storage and accepts total logical R=1..64.
/// Small-K wide accepts positive K<=1536 divisible by32; the other domains
/// are pinned alongside their explicit recipe selections above.
pub fn build_rowtiles_gfx1100() -> Result<Vec<Emitted>, String> {
    TARGETS
        .into_iter()
        .map(|target| emit_rowtiles_for(target).map_err(|e| format!("{}: {e}", target.symbol)))
        .collect()
}
