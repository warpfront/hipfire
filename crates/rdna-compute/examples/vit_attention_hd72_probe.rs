// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity + timing probe for the ViT attention kernels at the Qwen3.5/3.8
//! vision-tower shape (hidden 1152, 16 heads, head_dim 72) and at head_dim 64
//! (hidden 1024, proves the Q-tiled kernel is unchanged for `% 16 == 0`).
//! Runs naive, qtiled and wmma on identical seeded N(0, 1.5) QKV and prints
//! max_abs / rel-L2 vs naive plus median-of-5 time (2 warmups). Exits non-zero
//! if qtiled rel-L2 > 1e-5 or wmma rel-L2 > 5e-3.
//!
//! Run inside a ROCm container: /dev/kfd + /opt/rocm/lib.

use rdna_compute::{DType, Gpu, GpuTensor, HipResult};
use std::time::Instant;

fn time_median<F>(gpu: &mut Gpu, mut f: F) -> f64
where
    F: FnMut(&mut Gpu) -> HipResult<()>,
{
    for _ in 0..2 {
        f(gpu).expect("launch");
    }
    gpu.hip.device_synchronize().expect("sync");
    let mut ts = Vec::new();
    for _ in 0..5 {
        let t0 = Instant::now();
        f(gpu).expect("launch");
        gpu.hip.device_synchronize().expect("sync");
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts[2]
}

fn errs(a: &[f32], reference: &[f32]) -> (f32, f64) {
    let mut maxe = 0.0f32;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(reference) {
        maxe = maxe.max((x - y).abs());
        let d = (*x - *y) as f64;
        num += d * d;
        den += (*y as f64) * (*y as f64);
    }
    (maxe, (num / den.max(1e-30)).sqrt())
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let has_wmma = gpu.arch_caps.has_wmma_w32() || gpu.arch_caps.has_wmma_w32_gfx12();
    println!("Arch: {}  wmma={has_wmma}", gpu.arch);

    let mut seed: u64 = 0x9e3779b97f4a7c15;
    let mut uni = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    };
    let mut normal = move || -> f32 {
        let (u1, u2) = (uni(), uni());
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * 1.5) as f32
    };

    let mut failed = false;
    let heads = 16usize;
    for &(hidden, head_dim) in &[(1152usize, 72usize), (1024, 64)] {
        assert_eq!(heads * head_dim, hidden);
        for &n in &[784usize, 4096, 5000] {
            println!("== hidden={hidden} head_dim={head_dim} N={n}");
            let qkv_h: Vec<f32> = (0..n * 3 * hidden).map(|_| normal()).collect();
            let qkv = gpu.upload_f32(&qkv_h, &[n * 3 * hidden]).expect("upload");
            let fresh = |gpu: &mut Gpu| -> GpuTensor {
                gpu.alloc_tensor(&[n * hidden], DType::F32).expect("alloc")
            };

            let out_n = fresh(&mut gpu);
            let ms_n = time_median(&mut gpu, |g| {
                g.vit_attention_f32(&qkv, &out_n, n, hidden, heads, head_dim)
            });
            let naive = gpu.download_f32(&out_n).expect("download");
            println!("  naive : {ms_n:9.3} ms");

            let out_q = gpu
                .upload_f32(&vec![0.0f32; n * hidden], &[n * hidden])
                .expect("upload");
            let ms_q = time_median(&mut gpu, |g| {
                g.vit_attention_qtiled_f32(&qkv, &out_q, n, hidden, heads, head_dim)
            });
            let q = gpu.download_f32(&out_q).expect("download");
            let (me, rl2) = errs(&q, &naive);
            println!("  qtiled: {ms_q:9.3} ms  max_abs={me:.3e} rel_l2={rl2:.3e}");
            if !(rl2 <= 1e-5) {
                println!("  FAIL qtiled rel_l2 > 1e-5");
                failed = true;
            }

            if has_wmma {
                let out_w = gpu
                    .upload_f32(&vec![0.0f32; n * hidden], &[n * hidden])
                    .expect("upload");
                let ms_w = time_median(&mut gpu, |g| {
                    g.vit_attention_wmma_f32(&qkv, &out_w, n, hidden, heads, head_dim)
                });
                let w = gpu.download_f32(&out_w).expect("download");
                let (me, rl2) = errs(&w, &naive);
                println!("  wmma  : {ms_w:9.3} ms  max_abs={me:.3e} rel_l2={rl2:.3e}");
                if !(rl2 <= 5e-3) {
                    println!("  FAIL wmma rel_l2 > 5e-3");
                    failed = true;
                }
                gpu.free_tensor(out_w).expect("free");
            } else {
                println!("  wmma  : skipped (no wave32 WMMA on this arch)");
            }
            gpu.free_tensor(out_n).expect("free");
            gpu.free_tensor(out_q).expect("free");
            gpu.free_tensor(qkv).expect("free");
        }
    }
    if failed {
        std::process::exit(1);
    }
}
