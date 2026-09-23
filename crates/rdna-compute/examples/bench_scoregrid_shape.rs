// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — synthetic shape bench for the DeepSeek V4 SWA+topK attention
// kernel, anchored to the shape recorded in the gfx1151 MQ2R retained route.
//
// The retained ds4 tape spends 2510 us/token (7.32%, fifth-largest block) in
// `deepseek4_attn_swa_topk_scoregrid_f32_buf` across 41 calls — 61.2 us per
// call — at grid [n_heads=64] x block [512]. That is 1024 waves against
// gfx1151's 1280 resident wave slots: it cannot fill the machine once. Its
// code object also carries 281 wait instructions against 41 global loads
// (6.9 waits/load), the second-worst ratio in the tape.
//
// `head_dim` and the live position count are properties of the loaded model
// and the decode position, so this bench does not assume them: it sweeps the
// plausible grid and reports which cell reproduces the recorded 61.2 us/call.
// That cell is the anchor; decomposition experiments start from there.
//
// Variant is chosen by the same env the production dispatch reads, so this
// measures the exact kernels the tape uses:
//   HIPFIRE_DEEPSEEK4_ATTN_SCOREGRID=1                 scoregrid (shipped)
//   HIPFIRE_DEEPSEEK4_ATTN_SCOREGRID_XLANE=1           cross-lane variant
//   HIPFIRE_DEEPSEEK4_ATTN_SCOREGRID_LARGE_SERIAL=1    large-serial variant
//   (unset)                                            non-scoregrid baseline

use rdna_compute::{DType, Gpu};
use std::time::Instant;

/// Recorded cost of one scoregrid launch in the ds4 gfx1151 retained route:
/// 2510.0 us over 41 calls.
const RECORDED_US_PER_CALL: f64 = 2510.0 / 41.0;
/// gfx1151: 40 CU x 2 SIMD32 x 16 waves resident.
const WAVE_SLOTS: usize = 1280;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("=== scoregrid synthetic shape bench ===");
    eprintln!("  arch={}", gpu.arch);
    eprintln!(
        "  anchor: recorded {:.1} us/call (2510.0 us / 41 calls), grid[64] x block[512]",
        RECORDED_US_PER_CALL
    );
    eprintln!(
        "  variant env: SCOREGRID={:?} XLANE={:?} LARGE_SERIAL={:?}",
        std::env::var("HIPFIRE_DEEPSEEK4_ATTN_SCOREGRID").ok(),
        std::env::var("HIPFIRE_DEEPSEEK4_ATTN_SCOREGRID_XLANE").ok(),
        std::env::var("HIPFIRE_DEEPSEEK4_ATTN_SCOREGRID_LARGE_SERIAL").ok(),
    );
    eprintln!();

    let n_heads = 64usize; // grid.x in the recorded launch
    let swa_window = 128usize; // DS4 Flash sliding window
    let topk_window = 512usize; // index_topk

    let warmup = 50usize;
    let trials = 500usize;

    eprintln!(
        "  {:>8} {:>8} {:>8} {:>7} {:>6} {:>11} {:>10}",
        "head_dim", "n_swa", "n_topk", "n_total", "waves", "fills", "us/call"
    );
    eprintln!("  {}", "-".repeat(70));

    for head_dim in [64usize, 128, 192, 256] {
        for (n_swa, n_topk) in [
            (swa_window, 0usize),
            (swa_window, 128),
            (swa_window, 256),
            (swa_window, topk_window),
        ] {
            let n_total = n_swa + n_topk;
            if n_total > 1024 {
                continue; // MAX_TOTAL in the kernel
            }

            // MLA shares the latent KV across heads: the kernel indexes
            // k_col[d * stride + col] with no head offset, so K/V are
            // [head_dim][window], not per-head.
            let q = gpu.alloc_tensor(&[n_heads * head_dim], DType::F32).unwrap();
            let swa_k = gpu.alloc_tensor(&[head_dim * swa_window], DType::F32).unwrap();
            let swa_v = gpu.alloc_tensor(&[head_dim * swa_window], DType::F32).unwrap();
            let tk = gpu.alloc_tensor(&[head_dim * topk_window], DType::F32).unwrap();
            let sink = gpu.alloc_tensor(&[n_heads], DType::F32).unwrap();
            let out = gpu.alloc_tensor(&[n_heads * head_dim], DType::F32).unwrap();

            let qh = (0..n_heads * head_dim)
                .map(|i| ((i % 97) as f32 - 48.0) / 64.0)
                .collect::<Vec<f32>>();
            gpu.hip.memcpy_htod(&q.buf, bytes_of(&qh)).unwrap();
            let kh = (0..head_dim * topk_window)
                .map(|i| ((i % 89) as f32 - 44.0) / 64.0)
                .collect::<Vec<f32>>();
            gpu.hip
                .memcpy_htod(&swa_k.buf, bytes_of(&kh[..head_dim * swa_window]))
                .unwrap();
            gpu.hip
                .memcpy_htod(&swa_v.buf, bytes_of(&kh[..head_dim * swa_window]))
                .unwrap();
            gpu.hip.memcpy_htod(&tk.buf, bytes_of(&kh)).unwrap();

            // Live counts live in device buffers; the kernel reads [0] of each.
            // Production takes these as 4-byte sub-views of the F32 attention
            // scratch and the kernel reads them as `const int*`; mirror that
            // rather than inventing an integer tensor type.
            let n_valid = gpu.alloc_tensor(&[1], DType::F32).unwrap();
            let k_active = gpu.alloc_tensor(&[1], DType::F32).unwrap();
            gpu.hip
                .memcpy_htod(&n_valid.buf, bytes_of(&[n_swa as i32]))
                .unwrap();
            gpu.hip
                .memcpy_htod(&k_active.buf, bytes_of(&[n_topk as i32]))
                .unwrap();

            let mut call = |gpu: &mut Gpu| {
                gpu.deepseek4_attn_swa_topk_f32_buf(
                    &q,
                    &swa_k,
                    &swa_v,
                    &tk,
                    &tk,
                    &sink,
                    &out,
                    &n_valid,
                    &k_active,
                    n_heads as i32,
                    head_dim as i32,
                    swa_window as i32,
                    topk_window as i32,
                )
                .expect("scoregrid launch")
            };

            for _ in 0..warmup {
                call(&mut gpu);
            }
            gpu.hip.device_synchronize().unwrap();
            let t = Instant::now();
            for _ in 0..trials {
                call(&mut gpu);
            }
            gpu.hip.device_synchronize().unwrap();
            let us = t.elapsed().as_secs_f64() * 1e6 / trials as f64;

            // 512 threads = 16 waves per workgroup, one workgroup per head.
            let waves = n_heads * 16;
            let mark = if (us - RECORDED_US_PER_CALL).abs() / RECORDED_US_PER_CALL < 0.15 {
                "  <== matches recorded"
            } else {
                ""
            };
            eprintln!(
                "  {:>8} {:>8} {:>8} {:>7} {:>6} {:>10.2}x {:>10.2}{}",
                head_dim,
                n_swa,
                n_topk,
                n_total,
                waves,
                waves as f64 / WAVE_SLOTS as f64,
                us,
                mark
            );
        }
    }

    eprintln!();
    eprintln!("  The cell within 15% of {:.1} us is the recorded shape. Its wave count is",
        RECORDED_US_PER_CALL);
    eprintln!("  fixed at n_heads*16 = 1024 (0.80 fills) regardless of head_dim or n_total,");
    eprintln!("  because grid.x is n_heads and nothing else supplies parallelism. If us/call");
    eprintln!("  is flat across n_total, the kernel is launch/occupancy bound rather than");
    eprintln!("  work bound, and only a decomposition that adds blocks can help.");
}

fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: reading a Copy slice as bytes for an upload.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}
