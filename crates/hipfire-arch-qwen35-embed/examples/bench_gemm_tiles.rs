// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sweep architecture-supported LDS macro-tiles for the attention-shaped GEMM.
//!
//! On gfx11, the baseline is `gemm_f16_x_f16_wmma_lds_auto`, followed by every
//! `Gpu::LDS_TILE_VARIANTS` tile. On gfx1201, the baseline is the production
//! `attention_gemm_ld` QK route, followed by the four optimized split1 LDS tiles.
//! QK^T has K = head_dim = 256, where the K-loop is only a handful of stages
//! and a projection-shaped default may not be the best choice. This reports
//! throughput per tile without admitting gfx11-only kernels on gfx1201.
//!
//! Usage: cargo run --release -p hipfire-arch-qwen35-embed \
//!          --example bench_gemm_tiles --features lab -- [n] [hd] [iters]

use hipfire_arch_qwen35_embed::forward::attention_gemm_ld;
use rdna_compute::gemm::LdsTileSplitK;
use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    // M = the A operand's row count (out_dim), K = the contraction, B = rows of
    // X. QK^T is [M=n_k, K=hd, B=n_q]; PV is [M=hd, K=n_k, B=n_q].
    let n: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(4096);
    let hd: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(256);
    let iters: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(20);
    let batch: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(n);

    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    println!("gpu {} | [M={n}, K={hd}, B={batch}]", gpu.arch);
    let gfx1201 = gpu.arch == "gfx1201";

    let mut s: u32 = 12345;
    let mut rnd = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        half::f16::from_f32(((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5).to_bits()
    };
    let w_words: Vec<u16> = (0..n * hd).map(|_| rnd()).collect();
    let x_words: Vec<u16> = (0..batch * hd).map(|_| rnd()).collect();
    let w = gpu
        .upload_f16_bits(&w_words, &[n, hd])
        .map_err(|e| format!("w: {e:?}"))?;
    let x = gpu
        .upload_f16_bits(&x_words, &[batch, hd])
        .map_err(|e| format!("x: {e:?}"))?;
    let y = gpu
        .alloc_tensor(&[batch, n], DType::F32)
        .map_err(|e| format!("y: {e:?}"))?;
    let flops = 2.0 * (n as f64) * (hd as f64) * (batch as f64);

    let mut best: Option<(String, f64)> = None;
    // The default first, as the baseline.
    let mut results: Vec<(String, f64)> = Vec::new();
    {
        let baseline_name = if gfx1201 {
            "attention_gemm_ld (production QK)"
        } else {
            "auto (default)"
        };
        let run = |gpu: &mut Gpu| {
            if gfx1201 {
                attention_gemm_ld(gpu, &w, &x, &y, n, hd, batch, hd, hd, true)
            } else {
                gpu.gemm_f16_x_f16_wmma_lds_auto(&w, &x, &y, None, n, hd, batch)
            }
            .map_err(|e| format!("{baseline_name}: {e:?}"))
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
        if gfx1201 {
            println!(
                "  {baseline_name:<34} {ms:7.3} ms  {:6.2} TFLOP/s",
                flops / (ms * 1e-3) / 1e12
            );
        } else {
            println!(
                "  {:<34} {ms:7.3} ms  {:6.2} TFLOP/s  (grid_m={} grid_n={})",
                baseline_name,
                flops / (ms * 1e-3) / 1e12,
                n.div_ceil(128),
                batch.div_ceil(256)
            );
        }
        results.push((baseline_name.into(), ms));
    }

    if gfx1201 {
        for tile in [
            LdsTileSplitK::new(64, 64, 32, 64, true, 1),
            LdsTileSplitK::new(64, 128, 32, 64, true, 1),
            LdsTileSplitK::new(128, 128, 32, 64, false, 1),
            LdsTileSplitK::new(128, 256, 32, 64, false, 1),
        ] {
            if tile.bm > n || tile.bn > batch {
                continue;
            }
            let name = format!("{}", tile.entry());
            let mut ok = true;
            for _ in 0..2 {
                if gpu
                    .gemm_f16_x_f16_wmma_lds_splitk(&w, &x, &y, n, hd, batch, tile)
                    .is_err()
                {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            let _ = gpu.download_f32(&y).map_err(|e| format!("sync: {e:?}"))?;
            let t0 = Instant::now();
            for _ in 0..iters {
                gpu.gemm_f16_x_f16_wmma_lds_splitk(&w, &x, &y, n, hd, batch, tile)
                    .map_err(|e| format!("split1: {e:?}"))?;
            }
            let _ = gpu.download_f32(&y).map_err(|e| format!("sync: {e:?}"))?;
            let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
            println!(
                "  {name:<34} {ms:7.3} ms  {:6.2} TFLOP/s",
                flops / (ms * 1e-3) / 1e12
            );
            if best.as_ref().is_none_or(|(_, b)| ms < *b) {
                best = Some((name.clone(), ms));
            }
            results.push((name, ms));
        }
    } else {
        for tile in rdna_compute::Gpu::LDS_TILE_VARIANTS {
            if tile.bm > n || tile.bn > batch {
                continue;
            }
            let name = format!("{}", tile.entry());
            let mut ok = true;
            for _ in 0..2 {
                if gpu
                    .gemm_f16_x_f16_wmma_lds_tiled(&w, &x, &y, None, n, hd, batch, *tile)
                    .is_err()
                {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            let _ = gpu.download_f32(&y).map_err(|e| format!("sync: {e:?}"))?;
            let t0 = Instant::now();
            for _ in 0..iters {
                gpu.gemm_f16_x_f16_wmma_lds_tiled(&w, &x, &y, None, n, hd, batch, *tile)
                    .map_err(|e| format!("tiled: {e:?}"))?;
            }
            let _ = gpu.download_f32(&y).map_err(|e| format!("sync: {e:?}"))?;
            let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
            println!(
                "  {name:<34} {ms:7.3} ms  {:6.2} TFLOP/s",
                flops / (ms * 1e-3) / 1e12
            );
            if best.as_ref().is_none_or(|(_, b)| ms < *b) {
                best = Some((name.clone(), ms));
            }
            results.push((name, ms));
        }
    }

    if let Some((name, ms)) = best {
        println!("\n  best: {name}  {ms:.3} ms  (8 heads = {:.1} ms; today's attention is 66.7 ms/layer)", 8.0 * ms);
    }
    Ok(())
}
