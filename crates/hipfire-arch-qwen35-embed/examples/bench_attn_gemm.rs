// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Microbenchmark: is the trunk's GEMM fast at *attention's* shape?
//!
//! Attention as two batched GEMMs plus a row softmax is only worth building if
//! the production `attention_gemm_ld` route holds its throughput when the
//! contraction (`k`) is the head dimension rather than a projection width.
//! It uses optimized split1 LDS on gfx1201 and the existing gfx11 tile selection.
//! QK^T for one head is `[n_q, n_k, hd]` — M = n_keys, K = hd (256 here),
//! B = n_q — and a small K means few K-tiles per block, which is exactly where
//! a macro-tile kernel loses efficiency. This measures it.
//!
//! Usage: cargo run --release -p hipfire-arch-qwen35-embed \
//!          --example bench_attn_gemm --features lab -- [n] [hd] [iters]

use hipfire_arch_qwen35_embed::forward::attention_gemm_ld;
use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(4096);
    let hd: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(256);
    let iters: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(20);

    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    println!("gpu {} | shape [n={n}, k={hd}, batch={n}]", gpu.arch);

    // Deterministic pseudo-random f16 operands.
    let mut s: u32 = 12345;
    let mut rnd = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (half::f16::from_f32(((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5)).to_bits()
    };
    let w_words: Vec<u16> = (0..n * hd).map(|_| rnd()).collect();
    let x_words: Vec<u16> = (0..n * hd).map(|_| rnd()).collect();
    let w = gpu
        .upload_f16_bits(&w_words, &[n, hd])
        .map_err(|e| format!("w: {e:?}"))?;
    let x = gpu
        .upload_f16_bits(&x_words, &[n, hd])
        .map_err(|e| format!("x: {e:?}"))?;
    let y = gpu
        .alloc_tensor(&[n, n], DType::F32)
        .map_err(|e| format!("y: {e:?}"))?;

    // ml_dtypes/lldm: out[M=n, k=hd] times x[batch=n, k=hd] -> y[n, n].
    let run = |gpu: &mut Gpu| -> Result<(), String> {
        attention_gemm_ld(gpu, &w, &x, &y, n, hd, n, hd, hd, true)
            .map_err(|e| format!("gemm: {e:?}"))
    };
    for _ in 0..3 {
        run(&mut gpu)?;
    }
    let _ = gpu.download_f32(&y).map_err(|e| format!("sync: {e:?}"))?;

    let t0 = Instant::now();
    for _ in 0..iters {
        run(&mut gpu)?;
    }
    let _ = gpu.download_f32(&y).map_err(|e| format!("sync: {e:?}"))?;
    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let flops = 2.0 * (n as f64) * (n as f64) * (hd as f64);
    println!(
        "  attention_gemm_ld (production QK): {ms:.3} ms/call  ->  {:.2} TFLOP/s",
        flops / (ms * 1e-3) / 1e12
    );
    // The per-layer attention budget this must beat to be worth building.
    println!(
        "  (per-layer attention today: {:.1} ms for 8 heads; 8x this call = {:.1} ms)",
        66.7,
        8.0 * ms
    );
    Ok(())
}
