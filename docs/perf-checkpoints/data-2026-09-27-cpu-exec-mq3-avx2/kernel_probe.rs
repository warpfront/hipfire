// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Throwaway microbench: scalar vs AVX2 row dot per quant format and shape.
//!
//! `cargo run --release -p hipfire-cpu --example kernel_probe`
//!
//! Not committed (delete after use): the numbers it prints are a measurement
//! aid, not evidence on their own.

use std::time::Instant;

use hipfire_cpu::gemv::{gemv_with_simd, row_bytes};
use hipfire_cpu::quant::CpuQuant;

/// Synthetic weight bytes with a sane per-group header and a byte-diverse
/// payload, mirroring the fixtures the parity test uses.
fn weights(q: CpuQuant, m: usize, k: usize) -> Vec<u8> {
    let (ge, gb) = (q.group_elems(), q.group_bytes());
    let mut out = Vec::with_capacity(m * (k / ge) * gb);
    let groups = k / ge;
    for row in 0..m {
        for g in 0..groups {
            let salt = (row * groups + g) * 7;
            let mut bytes: Vec<u8> = (0..gb).map(|i| ((i + salt) * 37 + 11) as u8).collect();
            match q {
                // [f32 scale][f32 zero]
                CpuQuant::Mq4G256 => {
                    bytes[..4].copy_from_slice(&0.0313f32.to_le_bytes());
                    bytes[4..8].copy_from_slice(&(-0.4921f32).to_le_bytes());
                }
                // [s0 z0 s1 z1] as fp16, non-power-of-two like real weights
                CpuQuant::Mq3G256V2 => {
                    for (i, bits) in [0x2802u16, 0xafdf, 0x3456, 0xadad].iter().enumerate() {
                        bytes[2 * i..2 * i + 2].copy_from_slice(&bits.to_le_bytes());
                    }
                }
                _ => unreachable!("only the two kernel formats are probed"),
            }
            out.extend_from_slice(&bytes);
        }
    }
    out
}

fn x_of(k: usize) -> Vec<f32> {
    (0..k)
        .map(|i| ((i as u64 * 2654435761) % 8192) as f32 * 0.000_244_140_625 - 1.0)
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time(
    q: CpuQuant,
    packed: &[u8],
    m: usize,
    k: usize,
    x: &[f32],
    simd: bool,
    reps: usize,
) -> (f64, Vec<f32>) {
    let mut y = vec![0.0f32; m];
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        gemv_with_simd(q, packed, m, k, x, &mut y, Some(simd));
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    (median(times), y)
}

fn main() {
    println!(
        "avx2={} f16c-path={} threads={}",
        hipfire_cpu::simd::avx2_available(),
        hipfire_cpu::simd::avx2_f16c_available(),
        rayon::current_num_threads(),
    );
    // Exactly the step shapes `HIPFIRE_CPU_EXEC_TRACE=1` reports for
    // qwen3.8-27b.mq3-xt at budget 56 (8 spilled layers), one call each per
    // layer per token.
    let shapes: [(usize, usize); 8] = [
        (10240, 5120),
        (6144, 5120),
        (17408, 5120),
        (12288, 5120),
        (1024, 5120),
        (48, 5120),
        (5120, 6144),
        (5120, 17408),
    ];
    for q in [CpuQuant::Mq3G256V2, CpuQuant::Mq4G256] {
        let mut token_bytes = 0.0f64;
        let mut token_ms = 0.0f64;
        for (m, k) in shapes {
            let packed = weights(q, m, k);
            let rb = row_bytes(q, k) * m;
            let x = x_of(k);
            let reps = if m > 4096 { 5 } else { 20 };
            let (t_scalar, ys) = time(q, &packed, m, k, &x, false, reps);
            let (t_simd, yv) = time(q, &packed, m, k, &x, true, reps);
            let rel = ys
                .iter()
                .zip(&yv)
                .map(|(a, b)| (a - b).abs() / a.abs().max(1e-6))
                .fold(0.0f32, f32::max);
            let absdiff = ys
                .iter()
                .zip(&yv)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let mmac = (m * k) as f64 / 1e6;
            token_bytes += rb as f64 * 8.0;
            token_ms += t_simd * 8.0;
            println!(
                "{q:?} {m:5}x{k:5} {rb:>9} B  scalar {t_scalar:8.3} ms ({:6.2} GMAC/s)  \
                 avx2 {t_simd:8.3} ms ({:6.2} GMAC/s, {:5.2} GB/s)  speedup {:5.1}x  \
                 rel {rel:.1e} abs {absdiff:.3e} y0 {} / {}",
                mmac / t_scalar,
                mmac / t_simd,
                rb as f64 / 1e6 / t_simd,
                t_scalar / t_simd,
                ys[0],
                yv[0],
            );
        }
        // 8 spilled layers, one call per shape per layer per token.
        println!(
            "{q:?} per-token spilled bytes {:.1} MB -> {token_ms:.1} ms of GEMV ({:.1} GB/s)",
            token_bytes / 1e6,
            token_bytes / 1e9 / (token_ms / 1e3),
        );
    }
}
