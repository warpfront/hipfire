// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! ViT attention A/B at the qwen3.5-4b vision-tower shapes
//! (hidden=1024, heads=16, head_dim=64, depth=24; the tower runs
//! bidirectional attention over N = grid_h*grid_w patches).
//!
//! Motivation: `vision_forward` was measured at 25.5 s for N=4624 (68x68
//! grid) on gfx1101. The naive `vit_attention_f32` launches one block per
//! (head, query) and each block re-reads all of K and V from DRAM —
//! ~220 GB of traffic per layer at N=4624. This bench times it against
//! `vit_attention_opt` (QPB=2) and checks parity against a CPU reference
//! (small n) or cross-kernel (large n) so a replacement kernel can be
//! judged on evidence.
//!
//! Usage:
//!   cargo build --release -p rdna-compute --example bench_vit_attention_shapes
//!   (run inside a ROCm container: needs /dev/kfd + /opt/rocm/lib)

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const HIDDEN: usize = 1024;
const HEADS: usize = 16;
const HEAD_DIM: usize = 64;
const LAYERS: usize = 24;

fn cpu_reference(qkv: &[f32], n: usize) -> Vec<f32> {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let stride = 3 * HIDDEN;
    let mut out = vec![0.0f32; n * HIDDEN];
    for qi in 0..n {
        for h in 0..HEADS {
            let q_off = h * HEAD_DIM;
            let k_off = HIDDEN + h * HEAD_DIM;
            let v_off = 2 * HIDDEN + h * HEAD_DIM;
            let mut scores = vec![0.0f32; n];
            let mut m = f32::MIN;
            for j in 0..n {
                let mut dot = 0.0f32;
                for d in 0..HEAD_DIM {
                    dot += qkv[qi * stride + q_off + d] * qkv[j * stride + k_off + d];
                }
                scores[j] = dot * scale;
                m = m.max(scores[j]);
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                sum += *s;
            }
            for d in 0..HEAD_DIM {
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc += scores[j] * qkv[j * stride + v_off + d];
                }
                out[qi * HIDDEN + h * HEAD_DIM + d] = acc / sum;
            }
        }
    }
    out
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn time_kernel<F>(gpu: &mut Gpu, mut f: F, warmup: usize, trials: usize) -> f64
where
    F: FnMut(&mut Gpu) -> rdna_compute::HipResult<()>,
{
    for _ in 0..warmup {
        f(gpu).expect("kernel launch");
    }
    gpu.hip.device_synchronize().expect("sync");
    let t0 = Instant::now();
    for _ in 0..trials {
        f(gpu).expect("kernel launch");
    }
    gpu.hip.device_synchronize().expect("sync");
    t0.elapsed().as_secs_f64() / trials as f64 * 1e3
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("Arch: {}", gpu.arch);

    let mut seed: u64 = 0x9e3779b97f4a7c15;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f32 / (1u64 << 53) as f32 - 1.0
    };

    let ns: [usize; 3] = [676, 2704, 4624];
    for &n in &ns {
        let stride = 3 * HIDDEN;
        let qkv_h: Vec<f32> = (0..n * stride).map(|_| rng()).collect();
        let qkv = gpu.upload_f32(&qkv_h, &[n * stride]).expect("upload");
        let out = gpu.alloc_tensor(&[n * HIDDEN], DType::F32).expect("alloc");

        let trials = if n > 700 { 3 } else { 10 };
        let t_naive = time_kernel(
            &mut gpu,
            |g| g.vit_attention_f32(&qkv, &out, n, HIDDEN, HEADS, HEAD_DIM),
            2,
            trials,
        );
        let t_opt = time_kernel(
            &mut gpu,
            |g| g.vit_attention_opt(&qkv, &out, n, HIDDEN, HEADS, HEAD_DIM),
            2,
            trials,
        );
        let t_qtiled = time_kernel(
            &mut gpu,
            |g| g.vit_attention_qtiled_f32(&qkv, &out, n, HIDDEN, HEADS, HEAD_DIM),
            2,
            trials,
        );

        let (err_naive, err_opt, err_qtiled) = if n <= 676 {
            let host_ref = cpu_reference(&qkv_h, n);
            gpu.vit_attention_f32(&qkv, &out, n, HIDDEN, HEADS, HEAD_DIM)
                .expect("naive");
            let naive_out = gpu.download_f32(&out).expect("download");
            gpu.vit_attention_opt(&qkv, &out, n, HIDDEN, HEADS, HEAD_DIM)
                .expect("opt");
            let opt_out = gpu.download_f32(&out).expect("download");
            gpu.vit_attention_qtiled_f32(&qkv, &out, n, HIDDEN, HEADS, HEAD_DIM)
                .expect("qtiled");
            let qtiled_out = gpu.download_f32(&out).expect("download");
            (
                max_abs_err(&naive_out, &host_ref),
                max_abs_err(&opt_out, &host_ref),
                max_abs_err(&qtiled_out, &host_ref),
            )
        } else {
            gpu.vit_attention_f32(&qkv, &out, n, HIDDEN, HEADS, HEAD_DIM)
                .expect("naive");
            let naive_out = gpu.download_f32(&out).expect("download");
            gpu.vit_attention_opt(&qkv, &out, n, HIDDEN, HEADS, HEAD_DIM)
                .expect("opt");
            let opt_out = gpu.download_f32(&out).expect("download");
            gpu.vit_attention_qtiled_f32(&qkv, &out, n, HIDDEN, HEADS, HEAD_DIM)
                .expect("qtiled");
            let qtiled_out = gpu.download_f32(&out).expect("download");
            (
                max_abs_err(&opt_out, &naive_out),
                max_abs_err(&naive_out, &opt_out),
                max_abs_err(&qtiled_out, &naive_out),
            )
        };

        let traffic_gb =
            (LAYERS as f64) * (HEADS as f64) * (n as f64) * (n as f64) * (HEAD_DIM as f64) * 8.0
                / 1e9;
        println!(
            "N={n:5}  naive {t_naive:8.1} ms   opt {t_opt:8.1} ms   qtiled {t_qtiled:8.1} ms   \
             speedup {:6.2}x   |err| n/a-opt {err_naive:.2e}/{err_opt:.2e}/q {err_qtiled:.2e}   \
             est naive traffic {traffic_gb:6.0} GB/tower",
            t_naive / t_qtiled
        );
    }
}
