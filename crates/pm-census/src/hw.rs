// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-arch roofline peaks (dense, no sparsity).
//!
//! Units: every `wmma_*` / `valu_f32` field is **operations per second**
//! counting a multiply-accumulate as 2 (FLOP/s, TOPS), so the arithmetic floor
//! of `macs` MACs is `2 * macs / peak(pipe)`; `bw` is bytes per second. A pipe
//! the hardware does not have is `0.0` (callers must not divide by it).

use hipfire_kernel_abi::work::Pipe;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peaks {
    /// gfx target this row describes.
    pub arch: &'static str,
    /// Marketing name.
    pub name: &'static str,
    /// Where the numbers come from / which are inferred.
    pub source: &'static str,
    pub wmma_f16: f64,
    pub wmma_bf16: f64,
    pub wmma_fp8: f64,
    pub wmma_iu8: f64,
    pub wmma_iu4: f64,
    pub valu_f32: f64,
    pub bw: f64,
}

/// Peaks of a gfx target (a `:feature` suffix such as `gfx1201:xnack-` is
/// ignored); `None` for an arch without a table row.
pub fn peaks(arch: &str) -> Option<Peaks> {
    let arch = arch.split(':').next().unwrap_or(arch);
    match arch {
        // R9700 (Radeon AI PRO R9700, gfx1201): AMD dense peaks as quoted in
        // /home/kaden/qcal/release-0.4.3/raon/pfx/PHASE1.md §1 ("FP16/BF16 WMMA
        // 191 TFLOP/s, INT8 383 TOPS, INT4 766 TOPS, 640 GB/s") and the spec
        // /home/kaden/qcal/release-0.4.3/pmrip/S1-SPEC.md (FP8 383, VALU F32
        // 47.8 TFLOP/s). [INFERENCE] FP8 = INT8 rate (RDNA4 WMMA FP8 and IU8
        // issue at the same rate). Attained HBM copy bandwidth is lower: 619 GB/s
        // (pm-krsi-1100/iu4-roofline/calc.py `G1201.bw`); nominal here.
        "gfx1201" => Some(Peaks {
            arch: "gfx1201",
            name: "AMD Radeon AI PRO R9700",
            source: "PHASE1.md §1 + S1-SPEC.md (AMD dense peaks; BW nominal 640 GB/s, attained 619)",
            wmma_f16: 191.0e12,
            wmma_bf16: 191.0e12,
            wmma_fp8: 383.0e12,
            wmma_iu8: 383.0e12,
            wmma_iu4: 766.0e12,
            valu_f32: 47.8e12,
            bw: 640.0e9,
        }),
        // RX 7900 XTX (gfx1100): AMD spec 122.8 TFLOP/s FP16/BF16 WMMA (96 CU x
        // 2.5 GHz x 512 FLOP/clk/CU), IU8 same, IU4 2x, VALU F32 61.4 TFLOP/s
        // (96 CU x 2.5 GHz x 256, dual-issue), 960 GB/s GDDR6 (S1-SPEC.md).
        // No FP8 WMMA on gfx11: 0.0.
        "gfx1100" => Some(Peaks {
            arch: "gfx1100",
            name: "AMD Radeon RX 7900 XTX",
            source: "S1-SPEC.md / AMD spec (122.8 TFLOP/s F16 WMMA, 61.4 TFLOP/s F32, 960 GB/s); no FP8 WMMA",
            wmma_f16: 122.8e12,
            wmma_bf16: 122.8e12,
            wmma_fp8: 0.0,
            wmma_iu8: 122.8e12,
            wmma_iu4: 245.6e12,
            valu_f32: 61.4e12,
            bw: 960.0e9,
        }),
        // Strix Halo (Radeon 8060S, gfx1151): 40 CU. AMD quotes 59.4 TFLOP/s FP16
        // for the 8060S, which equals 40 CU x 2.9 GHz x 512 FLOP/clk/CU, the same
        // per-CU WMMA rate as gfx1100 [INFERENCE: IU8 = F16 rate, IU4 = 2x as on
        // gfx1100]. VALU F32 = 40 CU x 2.9 GHz x 256 = 29.7 TFLOP/s [INFERENCE,
        // gfx1100 per-CU rate]. Bandwidth 215 GB/s is the best attained cold copy
        // in /home/kaden/qcal/release-0.4.3/pm-krsi-1100/iu4-roofline/calc.py
        // (`G1151.bw`; nominal LPDDR5X-8000 256-bit = 256 GB/s). No FP8 WMMA: 0.0.
        "gfx1151" => Some(Peaks {
            arch: "gfx1151",
            name: "AMD Radeon 8060S (Strix Halo)",
            source: "AMD 59.4 TFLOP/s FP16 spec; IU/VALU rates [INFERENCE] from the gfx1100 per-CU rate at 2.9 GHz; BW 215 GB/s attained (calc.py)",
            wmma_f16: 59.4e12,
            wmma_bf16: 59.4e12,
            wmma_fp8: 0.0,
            wmma_iu8: 59.4e12,
            wmma_iu4: 118.8e12,
            valu_f32: 29.7e12,
            bw: 215.0e9,
        }),
        _ => None,
    }
}

/// Peak ops/s of an arithmetic pipe; `0.0` for [`Pipe::Memory`] (no
/// arithmetic) and for pipes the arch lacks.
pub fn peak(p: &Peaks, pipe: Pipe) -> f64 {
    match pipe {
        Pipe::WmmaF16 => p.wmma_f16,
        Pipe::WmmaBf16 => p.wmma_bf16,
        Pipe::WmmaFp8 => p.wmma_fp8,
        Pipe::WmmaIu8 => p.wmma_iu8,
        Pipe::WmmaIu4 => p.wmma_iu4,
        Pipe::ValuF32 => p.valu_f32,
        Pipe::Memory => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r9700_matches_phase1_ceiling_numbers() {
        let p = peaks("gfx1201").unwrap();
        assert_eq!(peak(&p, Pipe::WmmaF16), 191.0e12);
        assert_eq!(peak(&p, Pipe::WmmaBf16), 191.0e12);
        assert_eq!(peak(&p, Pipe::WmmaFp8), 383.0e12);
        assert_eq!(peak(&p, Pipe::WmmaIu8), 383.0e12);
        assert_eq!(peak(&p, Pipe::WmmaIu4), 766.0e12);
        assert_eq!(peak(&p, Pipe::ValuF32), 47.8e12);
        assert_eq!(peak(&p, Pipe::Memory), 0.0);
        assert_eq!(p.bw, 640.0e9);
        // PHASE1 §1: 5.97 GFLOP/tok linear on all-F16 WMMA => 191e12 / 5.97e9 = 32.0k
        // tok/s linear-only; with attention + recurrence 30.1k.
        let linear: f64 = 2.0 * 2_986e6;
        let attn = 0.37e9;
        let rec = 94e6;
        let ceiling = 191.0e12 / (linear + attn + rec);
        assert!((ceiling - 30.1e3).abs() / 30.1e3 < 0.05, "{ceiling}");
    }

    #[test]
    fn rdna3_and_strix_rows_and_unknown() {
        let x = peaks("gfx1100").unwrap();
        assert_eq!(x.wmma_f16, 122.8e12);
        assert_eq!(x.wmma_iu4, 2.0 * x.wmma_iu8);
        assert_eq!(x.valu_f32, 61.4e12);
        assert_eq!(x.bw, 960.0e9);
        assert_eq!(peak(&x, Pipe::WmmaFp8), 0.0);
        let s = peaks("gfx1151:sramecc+:xnack-").unwrap();
        assert_eq!(s.arch, "gfx1151");
        assert_eq!(s.bw, 215.0e9);
        assert!(peaks("gfx900").is_none());
        assert!(peaks("gfx1201").unwrap().source.contains("PHASE1"));
    }
}
