// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity test: `gemm_f16_x_f16_wmma_lds` (LDS-staged 128×128 macro-tile,
//! fused bias) vs `gemm_f16_x_f16_wmma` (LDS 0, 16×16 per wave) + a separate
//! `bias_add_f32` pass. The baseline kernel is the reference.
//!
//! Why the tolerance is tight: both kernels walk K in ascending steps of 16
//! and issue one `wmma_f32_16x16x16_f16_w32` per step into an F32 accumulator.
//! The LDS kernel stages 64 K-elements at a time but still consumes them
//! kt = 0,1,2,3 in order, so the **accumulation order along K is identical**.
//! Only the bias add differs (fused in the epilogue vs a separate f32 pass).
//! Results should therefore agree to near bit-exactness. A large delta means
//! a real indexing or staging bug — fix the kernel, do NOT widen the bound.
//!
//! Suites:
//!   1. Aligned shapes — everything a multiple of the 128×128 tile and of the
//!      64-element K stage. Covers the FLUX hot path.
//!   2. Tail shapes — M and/or B not a multiple of 128, including the
//!      degenerate B = 1 (FLUX modulation GEMMs) and M = 64
//!      (FLUX `final_layer.linear`). These exercise the clamp-and-discard path.
//!   3. Bias — same shapes with a non-zero bias, to check the fused epilogue
//!      against the separate `bias_add_f32`.
//!
//! Run: cargo run --release --example test_gemm_f16_x_f16_wmma_lds_parity -p rdna-compute
//! Exits 0 on pass, 1 on any failure.

use rdna_compute::{DType, Gpu, GpuTensor};

/// Max relative error accepted. The two kernels share K-accumulation order, so
/// anything above this is a bug, not float noise.
const TOL_REL: f32 = 1e-5;

/// Deterministic values in roughly [-1, 1). An LCG keeps the test reproducible
/// without pulling in a rand dependency.
fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

/// Upload f32 host data and convert it to an F16 device tensor.
fn upload_f16(gpu: &mut Gpu, host: &[f32], rows: usize, cols: usize) -> GpuTensor {
    let src = gpu.upload_f32(host, &[rows, cols]).expect("upload f32");
    let dst = gpu.zeros(&[rows, cols], DType::F16).expect("alloc f16");
    gpu.cast_f32_to_f16(&src, &dst).expect("cast f32->f16");
    gpu.free_tensor(src).expect("free f32 src");
    dst
}

struct Case {
    label: &'static str,
    m: usize,
    k: usize,
    b: usize,
}

const CASES: &[Case] = &[
    // Suite 1 — aligned.
    Case {
        label: "aligned/small",
        m: 256,
        k: 256,
        b: 256,
    },
    Case {
        label: "aligned/square-tile",
        m: 128,
        k: 64,
        b: 128,
    },
    Case {
        label: "aligned/flux-txt-qkv",
        m: 3072,
        k: 3072,
        b: 512,
    },
    Case {
        label: "aligned/flux-mlp-in",
        m: 12288,
        k: 3072,
        b: 128,
    },
    // The three shapes that carry 67 % of a FLUX.1-dev step's GEMM FLOPs, at
    // their real B = n_img + n_txt = 4608. These are the shapes the A/B bench
    // times, so the gate must cover them at full size, not a reduced B.
    Case {
        label: "hot/single-qkv",
        m: 3072,
        k: 3072,
        b: 4608,
    },
    Case {
        label: "hot/single-mlp-in",
        m: 12288,
        k: 3072,
        b: 4608,
    },
    Case {
        label: "hot/single-linear2",
        m: 3072,
        k: 15360,
        b: 4608,
    },
    // Suite 2 — tails.
    Case {
        label: "tail/M",
        m: 200,
        k: 128,
        b: 256,
    },
    Case {
        label: "tail/B",
        m: 256,
        k: 128,
        b: 100,
    },
    Case {
        label: "tail/both",
        m: 70,
        k: 64,
        b: 13,
    },
    Case {
        label: "tail/B=1 (flux mod)",
        m: 256,
        k: 128,
        b: 1,
    },
    Case {
        label: "tail/M=64 (flux final)",
        m: 64,
        k: 3072,
        b: 512,
    },
    Case {
        label: "tail/M<tile B<tile",
        m: 16,
        k: 64,
        b: 16,
    },
];

/// Run one shape through both kernels and compare. Returns the max relative
/// error, or None if the case failed to run.
fn run_case(gpu: &mut Gpu, c: &Case, with_bias: bool) -> f32 {
    let a_host = pseudo_random(c.m * c.k, 0xA5A5_0000 ^ c.m as u64);
    let x_host = pseudo_random(c.b * c.k, 0x5A5A_0000 ^ c.b as u64);
    let a = upload_f16(gpu, &a_host, c.m, c.k);
    let x = upload_f16(gpu, &x_host, c.b, c.k);

    let bias_host = pseudo_random(c.m, 0xB1A5_0000 ^ c.k as u64);
    let bias = gpu.upload_f32(&bias_host, &[c.m]).expect("upload bias");

    // Reference: baseline kernel, then the separate bias pass the caller
    // currently performs.
    let y_ref = gpu.zeros(&[c.b, c.m], DType::F32).expect("alloc y_ref");
    gpu.gemm_f16_x_f16_wmma(&a, &x, &y_ref, c.m, c.k, c.b)
        .expect("baseline gemm");
    if with_bias {
        gpu.bias_add_f32(&y_ref, &bias, c.b, c.m).expect("bias_add");
    }

    // Candidate: LDS kernel with the bias fused into the epilogue.
    let y_new = gpu.zeros(&[c.b, c.m], DType::F32).expect("alloc y_new");
    let bias_arg = if with_bias { Some(&bias) } else { None };
    gpu.gemm_f16_x_f16_wmma_lds(&a, &x, &y_new, bias_arg, c.m, c.k, c.b)
        .expect("lds gemm");

    gpu.hip.device_synchronize().expect("sync");
    let r = gpu.download_f32(&y_ref).expect("dl ref");
    let n = gpu.download_f32(&y_new).expect("dl new");

    let mut max_rel = 0.0f32;
    let mut max_at = 0usize;
    for i in 0..r.len() {
        let d = (r[i] - n[i]).abs();
        // Guard the denominator so a near-zero reference cannot manufacture a
        // huge ratio out of a tiny absolute difference.
        let rel = d / r[i].abs().max(1e-3);
        if rel > max_rel {
            max_rel = rel;
            max_at = i;
        }
    }
    if max_rel > TOL_REL {
        eprintln!(
            "      first-worst at flat {max_at}: ref={} new={}",
            r[max_at], n[max_at]
        );
    }

    for t in [a, x, bias, y_ref, y_new] {
        let _ = gpu.free_tensor(t);
    }
    max_rel
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== test_gemm_f16_x_f16_wmma_lds_parity ===");
    eprintln!("  arch = {arch}");
    eprintln!("  tol  = {TOL_REL:e} relative");
    if !arch.starts_with("gfx11") {
        eprintln!("  SKIPPED: gfx11 wave32 WMMA layout required, got {arch}");
        std::process::exit(0);
    }

    let mut fails = 0usize;
    for with_bias in [false, true] {
        eprintln!("\n--- bias = {with_bias} ---");
        for c in CASES {
            let max_rel = run_case(&mut gpu, c, with_bias);
            let verdict = if max_rel <= TOL_REL { "PASS" } else { "FAIL" };
            if max_rel > TOL_REL {
                fails += 1;
            }
            eprintln!(
                "  {verdict}  {:<24} M={:<6} K={:<6} B={:<5} max_rel={max_rel:.3e}",
                c.label, c.m, c.k, c.b
            );
        }
    }

    eprintln!();
    if fails == 0 {
        eprintln!("ALL PASS ({} cases)", CASES.len() * 2);
        std::process::exit(0);
    }
    eprintln!("{fails} FAILED");
    std::process::exit(1);
}
