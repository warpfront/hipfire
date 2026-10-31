// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Prefill activation-precision tiers (gfx1201) and the tiered Q8_0 prefill GEMM.
//!
//! A caller picks a [`PrefillActTier`] per GEMM *site*; every quantized tier is an opt-in that the
//! runtime admission ([`prefill_quant_admitted`]) can refuse, in which case the F16 route runs
//! (never an error). The tier that actually ran is reported back where a call returns one.

use crate::types::DispatchError;
use rdna_compute::{Gpu, GpuTensor};

/// Activation precision of one prefill GEMM site.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PrefillActTier {
    /// F16 activations (the architecture's default F16 WMMA route).
    #[default]
    F16,
    /// Symmetric int4 activations (`wmma_i32_16x16x32_iu4`, `block_i4_128`). qt44 weights only.
    A4,
    /// Int8 activations (`wmma_i32_16x16x16_iu8`; `block_i8_128` for qt44 weights, per-32 int8
    /// blocks for Q8_0 weights).
    A8,
    /// e4m3 activations (`wmma_f32_16x16x16_fp8_fp8_w32_gfx12`).
    Fp8,
}

impl PrefillActTier {
    /// `f16|a4|a8|fp8` (ASCII case-insensitive, surrounding whitespace ignored).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f16" => Some(Self::F16),
            "a4" => Some(Self::A4),
            "a8" => Some(Self::A8),
            "fp8" => Some(Self::Fp8),
            _ => None,
        }
    }

    /// The name [`Self::parse`] accepts.
    pub const fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::A4 => "a4",
            Self::A8 => "a8",
            Self::Fp8 => "fp8",
        }
    }
}

/// Row floor of every quantized prefill tier: below it the weight stream dominates each route and
/// the F16 route stays.
pub const PREFILL_QUANT_MIN_ROWS: usize = 64;

/// Arch half of the quantized prefill tiers (pure): the grouped symmetric qt44 cores
/// (routed/shared/dense FFN) run W4A8 on gfx1201 and on the gfx11 twin (gfx1100, gfx1151), W4A4 and
/// FP8 on gfx1201 only; the tiered Q8_0 GEMM is gfx1201 only ([`q8_0_prefill_runs`]).
pub fn grouped_sym_tier_admits(arch: &str, tier: PrefillActTier) -> bool {
    use rdna_compute::moe_grouped_qt44_iu4 as g;
    match tier {
        PrefillActTier::F16 => true,
        PrefillActTier::A8 => g::a8_supported(arch),
        PrefillActTier::A4 | PrefillActTier::Fp8 => g::a4_fp8_supported(arch),
    }
}

/// Runtime admission shared by every quantized prefill tier: at least
/// [`PREFILL_QUANT_MIN_ROWS`] rows, and neither graph capture nor Redline recording (the
/// quantizers allocate through the MMQ scratch). The arch, shape and weight-format conditions are
/// the call site's ([`grouped_sym_tier_admits`], [`q8_0_prefill_runs`]).
pub fn prefill_quant_admitted(gpu: &Gpu, rows: usize) -> bool {
    rows >= PREFILL_QUANT_MIN_ROWS && !gpu.graphs.capture_mode && !gpu.replay.is_recording()
}

/// Whether [`q8_0_prefill_gemm`] with `tier` runs a quantized route for this `k` and `rows`
/// (`false`: it would run the F16 batched GEMM).
pub fn q8_0_prefill_runs(gpu: &Gpu, tier: PrefillActTier, k: usize, rows: usize) -> bool {
    gpu.arch == "gfx1201"
        && prefill_quant_admitted(gpu, rows)
        && match tier {
            PrefillActTier::A8 => gpu.q8_a8_supported(k),
            PrefillActTier::Fp8 => gpu.q8_fp8_supported(k),
            // Q8_0 has no int4-activation route.
            PrefillActTier::F16 | PrefillActTier::A4 => false,
        }
}

/// `y[rows, m] = x[rows, k] @ W_q8[m, k]^T` for a Q8_0 weight `w` (F32 `x`/`y`), at the activation
/// precision `tier`: `A8` -> `gemm_q8_0_a8_batched` (int8 activations per 32-K block), `Fp8` ->
/// `gemm_q8_0_fp8_batched`, each when the device and `k` support it and the runtime admission
/// ([`prefill_quant_admitted`]) holds; anything else (including `F16`, `A4`, decode-sized `rows`,
/// graph capture) runs `gemm_q8_0_batched_chunked`. Returns the tier that ran.
#[allow(clippy::too_many_arguments)]
pub fn q8_0_prefill_gemm(
    gpu: &mut Gpu,
    tier: PrefillActTier,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    rows: usize,
) -> Result<PrefillActTier, DispatchError> {
    if q8_0_prefill_runs(gpu, tier, k, rows) {
        match tier {
            PrefillActTier::A8 => hip!(gpu.gemm_q8_0_a8_batched(w, x, y, m, k, rows))?,
            PrefillActTier::Fp8 => hip!(gpu.gemm_q8_0_fp8_batched(w, x, y, m, k, rows))?,
            PrefillActTier::F16 | PrefillActTier::A4 => unreachable!("not admitted above"),
        }
        return Ok(tier);
    }
    hip!(gpu.gemm_q8_0_batched_chunked(w, x, y, m, k, rows))?;
    Ok(PrefillActTier::F16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_names_round_trip() {
        for t in [
            PrefillActTier::F16,
            PrefillActTier::A4,
            PrefillActTier::A8,
            PrefillActTier::Fp8,
        ] {
            assert_eq!(PrefillActTier::parse(t.name()), Some(t));
        }
        assert_eq!(PrefillActTier::parse(" FP8 "), Some(PrefillActTier::Fp8));
        assert_eq!(PrefillActTier::parse("a5"), None);
        assert_eq!(PrefillActTier::parse(""), None);
        assert_eq!(PrefillActTier::default(), PrefillActTier::F16);
    }
}
