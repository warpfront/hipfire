// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Stage timing of the latent-attention prefill routes at one (rows, pos)
// operating point: decompressed flash prefill, and the absorbed route's three
// stages (q_abs gemm, flash over the 576/512 latent view, value expansion).
// Developer instrument: wall time over repeated launches, synchronized. A
// shared-card run is indicative only, never a throughput claim.
//
// usage: latent_attention_absorb_bench [rows=512] [pos=12000] [heads=40] [iters=10]

use std::time::Instant;

use rdna_compute::latent_attention_absorb::latent_attention_absorb_packed_elems;
use rdna_compute::latent_attention_flash::LATENT_ATTENTION_FLASH_SRC;
use rdna_compute::latent_attention_tiled::{LatentRows, TiledAttention};
use rdna_compute::{DType, Gpu, GpuTensor};

fn fill_bf16(gpu: &mut Gpu, n: usize, seed: u64, amp: f32) -> GpuTensor {
    let t = gpu.alloc_tensor(&[n], DType::BF16).expect("alloc");
    let mut x = seed | 1;
    let v: Vec<u16> = (0..n)
        .map(|_| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            let u = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32;
            ((((u * 2.0 - 1.0) * amp).to_bits()) >> 16) as u16
        })
        .collect();
    gpu.hip
        .memcpy_htod(&t.buf, unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2)
        })
        .expect("htod");
    t
}

fn fill_f32(gpu: &mut Gpu, n: usize, amp: f32) -> GpuTensor {
    let v: Vec<f32> = (0..n)
        .map(|i| (((i * 2654435761usize) >> 7) % 2001) as f32 / 1000.0 - 1.0)
        .map(|f| f * amp)
        .collect();
    gpu.upload_f32(&v, &[n]).expect("upload")
}

fn time<F: FnMut(&mut Gpu)>(gpu: &mut Gpu, iters: usize, mut f: F) -> f64 {
    f(gpu);
    gpu.hip.device_synchronize().expect("sync");
    let t0 = Instant::now();
    for _ in 0..iters {
        f(gpu);
    }
    gpu.hip.device_synchronize().expect("sync");
    t0.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn main() {
    let a: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("number"))
        .collect();
    let rows = a.first().copied().unwrap_or(512);
    let pos = a.get(1).copied().unwrap_or(12000);
    let heads = a.get(2).copied().unwrap_or(40);
    let iters = a.get(3).copied().unwrap_or(10);
    let (nope, tail, v, latent) = (128usize, 64usize, 128usize, 512usize);
    let total = pos + rows;
    let width = latent + tail;
    let mut gpu = Gpu::init().expect("gpu");

    let cache = fill_bf16(&mut gpu, total * width, 11, 1.0);
    let knope = fill_bf16(&mut gpu, total * heads * nope, 12, 0.5);
    let vmir = fill_bf16(&mut gpu, total * heads * v, 13, 1.0);
    let q = fill_f32(&mut gpu, rows * heads * (nope + tail), 3.0);
    let y = gpu.zeros(&[rows * heads * v], DType::F32).expect("y");
    let q_abs = gpu
        .zeros(&[rows * heads * width], DType::F32)
        .expect("q_abs");
    let z = gpu.zeros(&[rows * heads * latent], DType::F32).expect("z");
    let packed = fill_bf16(
        &mut gpu,
        latent_attention_absorb_packed_elems(heads, nope, v, latent),
        14,
        0.05,
    );
    let posb = gpu.alloc_tensor(&[1], DType::F32).expect("pos");
    gpu.hip
        .memcpy_htod(&posb.buf, &(pos as i32).to_le_bytes())
        .expect("pos");
    let scale = (192f32).powf(-0.5);

    let dec = TiledAttention {
        q: &q,
        y: &y,
        k_a: LatentRows {
            buf: &knope,
            offset: 0,
            tok_stride: heads * nope,
            head_stride: nope,
            width: nope,
        },
        k_b: LatentRows {
            buf: &cache,
            offset: latent,
            tok_stride: width,
            head_stride: 0,
            width: tail,
        },
        v: LatentRows {
            buf: &vmir,
            offset: 0,
            tok_stride: heads * v,
            head_stride: v,
            width: v,
        },
        pos: &posb,
        pos_offset: 0,
        n_rows: rows,
        n_heads: heads,
        scale,
        max_rows: total,
        round_y_bf16: true,
    };
    let abs = TiledAttention {
        q: &q_abs,
        y: &z,
        k_a: LatentRows {
            buf: &cache,
            offset: 0,
            tok_stride: width,
            head_stride: 0,
            width: latent,
        },
        k_b: LatentRows {
            buf: &cache,
            offset: latent,
            tok_stride: width,
            head_stride: 0,
            width: tail,
        },
        v: LatentRows {
            buf: &cache,
            offset: 0,
            tok_stride: width,
            head_stride: 0,
            width: latent,
        },
        pos: &posb,
        pos_offset: 0,
        n_rows: rows,
        n_heads: heads,
        scale,
        max_rows: total,
        round_y_bf16: false,
    };

    let avg_keys = pos as f64 + rows as f64 / 2.0;
    let flop =
        |dq: usize, dv: usize| rows as f64 * avg_keys * heads as f64 * (dq + dv) as f64 * 2.0;
    let t_dec = time(&mut gpu, iters, |g| {
        g.latent_attention_flash_prefill(&dec).expect("flash dec")
    });
    let t_q = time(&mut gpu, iters, |g| {
        g.latent_attention_absorb_q(&q, &packed, &q_abs, rows, heads, nope, tail, v, latent)
            .expect("absorb_q")
    });
    let t_abs = time(&mut gpu, iters, |g| {
        g.latent_attention_flash_prefill(&abs).expect("flash abs")
    });
    let t_v = time(&mut gpu, iters, |g| {
        g.latent_attention_absorb_v(&z, &packed, &y, rows, heads, nope, v, latent, true)
            .expect("absorb_v")
    });
    // Optional kernel-source variants for tuning: LAW_DEFS="LAW_VPF=2,LAW_VPF=8" (comma list of
    // `-D`-style definitions, each timed on the absorbed shape).
    let mut variants: Vec<(String, f64)> = Vec::new();
    if let Ok(list) = std::env::var("LAW_DEFS") {
        for (vi, def) in list.split(',').filter(|d| !d.is_empty()).enumerate() {
            let (k, val) = def.split_once('=').expect("DEF=VALUE");
            let func = format!("latent_attention_flash_prefill_var{vi}");
            let src = format!(
                "#define {k} {val}\n#define latent_attention_flash_prefill_wide {func}_wide\n{LATENT_ATTENTION_FLASH_SRC}"
            );
            let module = format!("latent_attention_flash_var{vi}");
            let t = time(&mut gpu, iters, |g| {
                g.latent_attention_flash_prefill_with_source(&abs, &module, &func, &src)
                    .expect("variant")
            });
            variants.push((def.to_string(), t));
        }
    }
    println!("rows={rows} pos={pos} heads={heads} iters={iters}");
    for (d, t) in &variants {
        println!("variant {d:<24}: {t:8.3} ms");
    }
    println!(
        "decompressed flash : {t_dec:8.3} ms  {:6.1} TFLOP/s (nominal 192/128)",
        flop(192, 128) / t_dec / 1e9
    );
    println!("absorb_q           : {t_q:8.3} ms");
    println!(
        "absorbed flash     : {t_abs:8.3} ms  {:6.1} TFLOP/s (nominal 576/512)",
        flop(576, 512) / t_abs / 1e9
    );
    println!("absorb_v           : {t_v:8.3} ms");
    println!(
        "absorbed total     : {:8.3} ms  ({:.2}x decompressed)",
        t_q + t_abs + t_v,
        (t_q + t_abs + t_v) / t_dec
    );
}
