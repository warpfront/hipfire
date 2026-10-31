// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Prefill activation-precision tiers (gfx1201, gfx1100, gfx1151) and the tiered Q8_0 prefill GEMM.
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
/// FP8 on gfx1201 only; the tiered Q8_0 GEMM is W8A8 on gfx1201 and on gfx1100 / gfx1151 (i8-MMQ 64x64
/// kernel), FP8 on gfx1201 only ([`q8_0_weight_admits`]).
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

/// Arch half of the tiered Q8_0 GEMM (pure): the weight-shape part of [`q8_0_prefill_runs`] for
/// a Q8_0 weight `[m, k]`. gfx1201 runs W8A8 (and FP8) on any `k % 32 == 0`; gfx1100 and gfx1151
/// run W8A8 on the i8-MMQ 64x64 kernel (`gemm_q8_0_mmq_4w_gfx1151`), which needs `k % 128 == 0`
/// (M and N tails are bounds-checked in the kernel). Every other arch, and the A4 tier, has no Q8_0 route. `Err` is the reason
/// the weight stays on the F16 route.
pub fn q8_0_weight_admits(
    arch: &str,
    tier: PrefillActTier,
    m: usize,
    k: usize,
) -> Result<(), &'static str> {
    match (arch, tier) {
        (_, PrefillActTier::F16) => Ok(()),
        (_, PrefillActTier::A4) => Err("Q8_0 has no A4 route"),
        ("gfx1201", _) => {
            if k > 0 && k % 32 == 0 {
                Ok(())
            } else {
                Err("Q8_0 K is not a multiple of 32")
            }
        }
        ("gfx1100" | "gfx1151", PrefillActTier::A8) => {
            if m > 0 && k > 0 && k % 128 == 0 {
                Ok(())
            } else {
                Err("Q8_0 gfx11 int8 route needs K % 128 == 0")
            }
        }
        ("gfx1100" | "gfx1151", PrefillActTier::Fp8) => Err("FP8 prefill tier is gfx1201-only"),
        _ => Err("Q8_0 has no int8 route on this arch"),
    }
}

/// Whether [`q8_0_prefill_gemm`] with `tier` runs a quantized route for a Q8_0 weight `[m, k]` at
/// this many `rows` (`false`: it would run the F16 batched GEMM).
pub fn q8_0_prefill_runs(gpu: &Gpu, tier: PrefillActTier, m: usize, k: usize, rows: usize) -> bool {
    if !prefill_quant_admitted(gpu, rows) || q8_0_weight_admits(&gpu.arch, tier, m, k).is_err() {
        return false;
    }
    match gpu.arch.as_str() {
        "gfx1201" => match tier {
            PrefillActTier::A8 => gpu.q8_a8_supported(k),
            PrefillActTier::Fp8 => gpu.q8_fp8_supported(k),
            PrefillActTier::F16 | PrefillActTier::A4 => false,
        },
        "gfx1100" | "gfx1151" => tier == PrefillActTier::A8,
        _ => false,
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
    if q8_0_prefill_runs(gpu, tier, m, k, rows) {
        match tier {
            PrefillActTier::A8 if gpu.arch != "gfx1201" => {
                hip!(gpu.gemm_q8_0_mmq_4w_gfx1151(w, x, y, m, k, rows))?
            }
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

    #[test]
    fn q8_0_weight_admission_per_arch() {
        use PrefillActTier::*;
        // gfx1201: any K % 32, A8 and FP8, never A4
        assert!(q8_0_weight_admits("gfx1201", A8, 40, 2560).is_ok());
        assert!(q8_0_weight_admits("gfx1201", Fp8, 5120, 576).is_ok());
        assert_eq!(
            q8_0_weight_admits("gfx1201", A8, 64, 100),
            Err("Q8_0 K is not a multiple of 32")
        );
        assert_eq!(
            q8_0_weight_admits("gfx1201", A4, 64, 128),
            Err("Q8_0 has no A4 route")
        );
        // gfx1100 / gfx1151: A8 on the i8-MMQ 64x64 kernel only (K % 128)
        for arch in ["gfx1100", "gfx1151"] {
            assert!(q8_0_weight_admits(arch, A8, 5120, 2560).is_ok(), "{arch}");
            assert!(q8_0_weight_admits(arch, A8, 576, 2560).is_ok(), "{arch}");
            assert!(q8_0_weight_admits(arch, A8, 40, 2560).is_ok(), "{arch}");
            assert!(q8_0_weight_admits(arch, A8, 2560, 12032).is_ok(), "{arch}");
            for (m, k) in [(7680, 576), (5120, 32), (5120, 100)] {
                assert_eq!(
                    q8_0_weight_admits(arch, A8, m, k),
                    Err("Q8_0 gfx11 int8 route needs K % 128 == 0"),
                    "{arch} {m}x{k}"
                );
            }
            assert_eq!(
                q8_0_weight_admits(arch, Fp8, 5120, 2560),
                Err("FP8 prefill tier is gfx1201-only")
            );
            assert_eq!(
                q8_0_weight_admits(arch, A4, 5120, 2560),
                Err("Q8_0 has no A4 route")
            );
        }
        // other arches: no route; F16 is always fine
        assert!(q8_0_weight_admits("gfx942", A8, 5120, 2560).is_err());
        assert!(q8_0_weight_admits("gfx942", F16, 5120, 2560).is_ok());
    }
}
