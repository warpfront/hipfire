// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! ViT-tower linear shapes: `gemm_f16_wmma_mb8` + `bias_add_f32` (the old
//! `linear_f16` path, `HIPFIRE_VIT_GEMM=mb8`) against the LDS-tiled route the
//! qwen35-vl tower now takes on gfx1201 / gfx11 (F32 X cast to F16, LDS WMMA
//! GEMM, bias fused on gfx11 and a separate `bias_add_f32` on gfx1201).
//!
//! Shapes are the 27B tower's (hidden 1152, mlp 4304, out_hidden 5120, merger
//! 4608) at n = 784 and 4096 patches; the two merger GEMMs run at n / 4. fc2
//! (K = 4304, not a multiple of 64) runs the same way the tower does: weight
//! zero-padded to K_pad = 4352 per row, X cast into a `[n, K_pad]` buffer with
//! zero pad columns, K = K_pad and pitch = K_pad on both operands.
//!
//! Per case: median of 5 timed runs after 2 warmups for each route, and the
//! relative L2 error of the LDS output against the mb8 output. Exits 1 if any
//! case exceeds 5e-3 (mb8 also rounds X to F16, so the expected value is ~0).
//!
//! Run (needs a gfx1201 or gfx11 GPU):
//!   cargo run --release -p rdna-compute --features lab --example bench_vit_gemm_shapes

use half::f16;
use rdna_compute::gemm::LdsTileSplitK;
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};
use std::time::Instant;

const REL_L2_LIMIT: f64 = 5e-3;
const K_ALIGN: usize = 64;
const WARMUP: usize = 2;
const RUNS: usize = 5;

struct Shape {
    name: &'static str,
    out: usize,
    k: usize,
    /// Merger GEMMs see n / 4 rows (2x2 spatial merge).
    merged: bool,
}

const SHAPES: [Shape; 7] = [
    Shape {
        name: "patch_embed",
        out: 1152,
        k: 1536,
        merged: false,
    },
    Shape {
        name: "qkv",
        out: 3456,
        k: 1152,
        merged: false,
    },
    Shape {
        name: "proj",
        out: 1152,
        k: 1152,
        merged: false,
    },
    Shape {
        name: "fc1",
        out: 4304,
        k: 1152,
        merged: false,
    },
    Shape {
        name: "fc2",
        out: 1152,
        k: 4304,
        merged: false,
    },
    Shape {
        name: "merger_fc1",
        out: 4608,
        k: 4608,
        merged: true,
    },
    Shape {
        name: "merger_fc2",
        out: 5120,
        k: 4608,
        merged: true,
    },
];

/// Same tile policy as `vit_gfx1201_tile` in qwen35_vl.rs.
fn gfx1201_tile(out: usize) -> LdsTileSplitK {
    match out {
        0..512 => LdsTileSplitK::new(64, 128, 32, 64, true, 1),
        512..1024 => LdsTileSplitK::new(128, 128, 32, 64, false, 1),
        _ => LdsTileSplitK::new(128, 256, 32, 64, false, 1),
    }
}

/// The tower's LDS route (`linear_f16_ld`), minus the allocations.
#[allow(clippy::too_many_arguments)]
fn lds_route(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    x16: &GpuTensor,
    bias: &GpuTensor,
    y: &GpuTensor,
    out: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    let k_pad = k.next_multiple_of(K_ALIGN);
    if k_pad == k {
        gpu.cast_f32_to_f16(x, x16)?;
    } else {
        gpu.cast_f32_to_f16_pitched(x, x16, n, k, k_pad)?;
    }
    if gpu.arch == "gfx1201" {
        let tile = gfx1201_tile(out);
        if k_pad == k {
            gpu.gemm_f16_x_f16_wmma_lds_splitk(w, x16, y, out, k_pad, n, tile)?;
        } else {
            gpu.gemm_f16_x_f16_wmma_lds_splitk_ld(w, x16, y, out, k_pad, n, tile, k_pad, k_pad)?;
        }
        gpu.bias_add_f32(y, bias, n, out)
    } else {
        gpu.gemm_f16_x_f16_wmma_lds_auto_ld(w, x16, y, Some(bias), out, k_pad, n, k_pad, k_pad)
    }
}

fn median_ms<F>(gpu: &mut Gpu, mut f: F) -> f64
where
    F: FnMut(&mut Gpu) -> HipResult<()>,
{
    for _ in 0..WARMUP {
        f(gpu).expect("warmup launch");
    }
    gpu.hip.device_synchronize().expect("sync");
    let mut t = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t0 = Instant::now();
        f(gpu).expect("timed launch");
        gpu.hip.device_synchronize().expect("sync");
        t.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[RUNS / 2]
}

fn rel_l2(test: &[f32], reference: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&a, &b) in test.iter().zip(reference) {
        let d = a as f64 - b as f64;
        num += d * d;
        den += b as f64 * b as f64;
    }
    (num / den.max(1e-30)).sqrt()
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("Arch: {}", gpu.arch);
    if !(gpu.arch == "gfx1201" || gpu.arch_caps.has_wmma_w32()) {
        println!("no LDS-GEMM route on this arch; nothing to compare");
        return;
    }

    let mut seed: u64 = 0x9e3779b97f4a7c15;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
    };

    println!(
        "{:<12} {:>5} {:>5} {:>6} {:>10} {:>10} {:>8} {:>9} {:>10}",
        "shape", "out", "K", "n", "mb8 ms", "lds ms", "speedup", "lds TF/s", "rel-L2"
    );
    let mut failed = false;
    for &n_patches in &[784usize, 4096] {
        for s in &SHAPES {
            let n = if s.merged { n_patches / 4 } else { n_patches };
            let (out, k) = (s.out, s.k);
            let k_pad = k.next_multiple_of(K_ALIGN);

            // F16 weights, packed for mb8 and zero-padded to K_pad for the LDS route.
            let w_h: Vec<f16> = (0..out * k)
                .map(|_| f16::from_f32(rng() / (k as f32).sqrt()))
                .collect();
            let as_bytes =
                |v: &[f16]| -> Vec<u8> { v.iter().flat_map(|h| h.to_le_bytes()).collect() };
            let w_packed = gpu
                .upload_raw(&as_bytes(&w_h), &[out * k])
                .expect("upload W");
            let w_lds = if k_pad == k {
                None
            } else {
                let mut padded = vec![f16::ZERO; out * k_pad];
                for r in 0..out {
                    padded[r * k_pad..r * k_pad + k].copy_from_slice(&w_h[r * k..(r + 1) * k]);
                }
                let mut t = gpu
                    .upload_raw(&as_bytes(&padded), &[out * k_pad])
                    .expect("upload padded W");
                t.dtype = DType::F16;
                Some(t)
            };
            let x_h: Vec<f32> = (0..n * k).map(|_| rng()).collect();
            let b_h: Vec<f32> = (0..out).map(|_| rng()).collect();
            let x = gpu.upload_f32(&x_h, &[n * k]).expect("upload X");
            let bias = gpu.upload_f32(&b_h, &[out]).expect("upload bias");
            let x16 = gpu
                .alloc_tensor(&[n * k_pad], DType::F16)
                .expect("alloc x16");
            let y_mb8 = gpu.alloc_tensor(&[n * out], DType::F32).expect("alloc y");
            let y_lds = gpu.alloc_tensor(&[n * out], DType::F32).expect("alloc y");

            let t_mb8 = median_ms(&mut gpu, |g| {
                g.gemm_f16_wmma_mb8(&w_packed, &x, &y_mb8, out, k, n)?;
                g.bias_add_f32(&y_mb8, &bias, n, out)
            });
            let w_for_lds = w_lds.as_ref().unwrap_or(&w_packed);
            let t_lds = median_ms(&mut gpu, |g| {
                lds_route(g, w_for_lds, &x, &x16, &bias, &y_lds, out, k, n)
            });

            let a = gpu.download_f32(&y_lds).expect("download lds");
            let b = gpu.download_f32(&y_mb8).expect("download mb8");
            let err = rel_l2(&a, &b);
            let tflops = 2.0 * n as f64 * out as f64 * k as f64 / (t_lds * 1e-3) / 1e12;
            let flag = if err > REL_L2_LIMIT || !err.is_finite() {
                failed = true;
                "  FAIL"
            } else {
                ""
            };
            println!(
                "{:<12} {:>5} {:>5} {:>6} {:>10.3} {:>10.3} {:>7.2}x {:>9.2} {:>10.2e}{}",
                s.name,
                out,
                if k_pad == k {
                    k.to_string()
                } else {
                    format!("{k}>{k_pad}")
                },
                n,
                t_mb8,
                t_lds,
                t_mb8 / t_lds,
                tflops,
                err,
                flag
            );

            for t in [x, bias, x16, y_mb8, y_lds, w_packed] {
                gpu.free_tensor(t).expect("free");
            }
            if let Some(t) = w_lds {
                gpu.free_tensor(t).expect("free");
            }
        }
    }
    if failed {
        eprintln!("FAIL: rel-L2 of LDS vs mb8 above {REL_L2_LIMIT:e}");
        std::process::exit(1);
    }
}
