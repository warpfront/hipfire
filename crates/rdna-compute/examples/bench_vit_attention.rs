// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Focused microbench + parity probe for the qwen3.5-VL vision-tower
//! self-attention kernels (`vit_attention_f32` naive, `vit_attention_qtiled_f32`
//! production).
//!
//! Default shape is the real one measured on the OCR serve: N=7564 patches
//! (62x122 grid, 2 MP page), hidden=1024, num_heads=16, head_dim=64.
//!
//! The naive kernel is the numeric reference (two-pass softmax). The Q-tiled
//! kernel is compared against it (max abs / max rel error) and timed.
//!
//! Run:
//!   cargo run --release --example bench_vit_attention -p rdna-compute --features lab -- \
//!       [--n N] [--heads H] [--hd D] [--iters K]

use rdna_compute::{DType, Gpu};

fn lcg_data(seed: u32, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let u = (s >> 16) & 0x7fff;
            (u as f32 / 32_768.0 - 0.5) * scale
        })
        .collect()
}

fn arg_u32(args: &[String], name: &str, default: u32) -> u32 {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|v| v.parse().expect("flag value must be an integer"))
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n = arg_u32(&args, "--n", 7564) as usize;
    let num_heads = arg_u32(&args, "--heads", 16) as usize;
    let head_dim = arg_u32(&args, "--hd", 64) as usize;
    let iters = arg_u32(&args, "--iters", 5).max(1) as usize;
    let hidden = num_heads * head_dim;

    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!(
        "GPU: {}  shape: N={n} hidden={hidden} heads={num_heads} hd={head_dim}",
        gpu.arch
    );

    let qkv = lcg_data(0xa5a5_a5a5, n * 3 * hidden, 0.2);
    let d_qkv = gpu.upload_f32(&qkv, &[n * 3 * hidden]).expect("upload qkv");
    let d_ref = gpu.zeros(&[n * hidden], DType::F32).expect("alloc ref");
    let d_out = gpu.zeros(&[n * hidden], DType::F32).expect("alloc out");

    // Numeric reference: naive two-pass softmax.
    let t = std::time::Instant::now();
    gpu.vit_attention_f32(&d_qkv, &d_ref, n, hidden, num_heads, head_dim)
        .expect("naive attention");
    gpu.hip.device_synchronize().expect("sync");
    eprintln!(
        "naive   vit_attention_f32        {:8.1} ms",
        t.elapsed().as_secs_f64() * 1e3
    );

    // Production path: warm once, then time.
    gpu.vit_attention_qtiled_f32(&d_qkv, &d_out, n, hidden, num_heads, head_dim)
        .expect("qtiled attention");
    gpu.hip.device_synchronize().expect("sync");
    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.vit_attention_qtiled_f32(&d_qkv, &d_out, n, hidden, num_heads, head_dim)
            .expect("qtiled attention");
    }
    gpu.hip.device_synchronize().expect("sync");
    eprintln!(
        "qtiled  vit_attention_qtiled_f32 {:8.1} ms  ({iters} iters)",
        t.elapsed().as_secs_f64() * 1e3 / iters as f64
    );

    let a = gpu.download_f32(&d_ref).expect("dl ref");
    let b = gpu.download_f32(&d_out).expect("dl out");
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut ref_abs_max = 0f32;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        max_abs = max_abs.max(d);
        ref_abs_max = ref_abs_max.max(a[i].abs());
        if a[i].abs() > 1e-6 {
            max_rel = max_rel.max(d / a[i].abs());
        }
    }
    eprintln!(
        "parity  ref_abs_max={ref_abs_max:.6e}  max_abs={max_abs:.6e}  max_rel={max_rel:.3e}  {}",
        if max_abs <= 5e-4 { "OK" } else { "CHECK" }
    );

    // The ViT WMMA flash kernel is what `HIPFIRE_VIT_ATTN=auto` picks on
    // gfx11/gfx12 for head_dim <= 80, so it is the arm the Q-tiled path has
    // to beat to change what the production tower runs. It packs Q/K/V to
    // F16 internally, so parity against the F32 naive reference is a
    // round-trip, not an arithmetic identity.
    if head_dim <= 80 {
        let d_wmma = gpu.zeros(&[n * hidden], DType::F32).expect("alloc wmma out");
        gpu.vit_attention_wmma_f32(&d_qkv, &d_wmma, n, hidden, num_heads, head_dim)
            .expect("wmma vit attention");
        gpu.hip.device_synchronize().expect("sync");
        let c = gpu.download_f32(&d_wmma).expect("dl wmma");
        let mut w_abs = 0f32;
        for i in 0..a.len() {
            w_abs = w_abs.max((a[i] - c[i]).abs());
        }
        eprintln!(
            "wmma    vit_attention_wmma_f32  dynamic  parity vs naive max_abs={w_abs:.3e}  {}",
            if w_abs <= 5e-3 { "OK" } else { "CHECK" }
        );
        let t = std::time::Instant::now();
        for _ in 0..iters {
            gpu.vit_attention_wmma_f32(&d_qkv, &d_wmma, n, hidden, num_heads, head_dim)
                .expect("wmma vit attention");
        }
        gpu.hip.device_synchronize().expect("sync");
        eprintln!(
            "wmma    vit_attention_wmma_f32  {:8.1} ms  ({iters} iters)",
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        );
        gpu.free_tensor(d_wmma).ok();
    }

    // The dots.ocr family routes vision attention through split Q/K/V +
    // WMMA flash attention. Measure the same shapes through it: split cost
    // is included (the real tower would pay it per layer).
    let d_q = gpu.zeros(&[n * hidden], DType::F32).expect("alloc q");
    let d_k = gpu.zeros(&[n * hidden], DType::F32).expect("alloc k");
    let d_v = gpu.zeros(&[n * hidden], DType::F32).expect("alloc v");
    let d_wout = gpu.zeros(&[n * hidden], DType::F32).expect("alloc wout");

    gpu.qkv_split_interleaved_f32(&d_qkv, &d_q, &d_k, &d_v, n, hidden)
        .expect("qkv split");
    gpu.attention_dflash_wmma_m32_f32(
        &d_q, &d_k, &d_v, &d_wout, n, n, num_heads, num_heads, head_dim,
    )
    .expect("wmma m32 attention");
    gpu.hip.device_synchronize().expect("sync");

    let c = gpu.download_f32(&d_wout).expect("dl wout");
    let mut w_abs = 0f32;
    let mut w_rel = 0f32;
    for i in 0..a.len() {
        let d = (a[i] - c[i]).abs();
        w_abs = w_abs.max(d);
        if a[i].abs() > 1e-6 {
            w_rel = w_rel.max(d / a[i].abs());
        }
    }
    eprintln!(
        "wmma m32 parity  max_abs={w_abs:.6e}  max_rel={w_rel:.3e}  {}",
        if w_abs <= 5e-4 { "OK" } else { "CHECK" }
    );

    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.qkv_split_interleaved_f32(&d_qkv, &d_q, &d_k, &d_v, n, hidden)
            .expect("qkv split");
        gpu.attention_dflash_wmma_m32_f32(
            &d_q, &d_k, &d_v, &d_wout, n, n, num_heads, num_heads, head_dim,
        )
        .expect("wmma m32 attention");
    }
    gpu.hip.device_synchronize().expect("sync");
    eprintln!(
        "wmma    split+attention_m32_f32  {:8.1} ms  ({iters} iters)",
        t.elapsed().as_secs_f64() * 1e3 / iters as f64
    );

    gpu.qkv_split_interleaved_f32(&d_qkv, &d_q, &d_k, &d_v, n, hidden)
        .expect("qkv split");
    gpu.attention_dflash_wmma_f32(
        &d_q, &d_k, &d_v, &d_wout, n, n, num_heads, num_heads, head_dim,
    )
    .expect("wmma m16 attention");
    gpu.hip.device_synchronize().expect("sync");
    let c = gpu.download_f32(&d_wout).expect("dl wout");
    let mut w_abs = 0f32;
    for i in 0..a.len() {
        w_abs = w_abs.max((a[i] - c[i]).abs());
    }
    eprintln!(
        "wmma m16 parity  max_abs={w_abs:.6e}  {}",
        if w_abs <= 5e-4 { "OK" } else { "CHECK" }
    );
    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.qkv_split_interleaved_f32(&d_qkv, &d_q, &d_k, &d_v, n, hidden)
            .expect("qkv split");
        gpu.attention_dflash_wmma_f32(
            &d_q, &d_k, &d_v, &d_wout, n, n, num_heads, num_heads, head_dim,
        )
        .expect("wmma m16 attention");
    }
    gpu.hip.device_synchronize().expect("sync");
    eprintln!(
        "wmma    split+attention_m16_f32  {:8.1} ms  ({iters} iters)",
        t.elapsed().as_secs_f64() * 1e3 / iters as f64
    );

    if head_dim == 128 {
        gpu.qkv_split_interleaved_f32(&d_qkv, &d_q, &d_k, &d_v, n, hidden)
            .expect("qkv split");
        gpu.attention_dflash_wmma_n64_f32(
            &d_q, &d_k, &d_v, &d_wout, n, n, num_heads, num_heads, head_dim,
        )
        .expect("wmma n64 attention");
        gpu.hip.device_synchronize().expect("sync");
        let t = std::time::Instant::now();
        for _ in 0..iters {
            gpu.qkv_split_interleaved_f32(&d_qkv, &d_q, &d_k, &d_v, n, hidden)
                .expect("qkv split");
            gpu.attention_dflash_wmma_n64_f32(
                &d_q, &d_k, &d_v, &d_wout, n, n, num_heads, num_heads, head_dim,
            )
            .expect("wmma n64 attention");
        }
        gpu.hip.device_synchronize().expect("sync");
        eprintln!(
            "wmma    split+attention_n64_f32  {:8.1} ms  ({iters} iters)",
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        );
    }

    gpu.free_tensor(d_qkv).ok();
    gpu.free_tensor(d_ref).ok();
    gpu.free_tensor(d_out).ok();
    gpu.free_tensor(d_q).ok();
    gpu.free_tensor(d_k).ok();
    gpu.free_tensor(d_v).ok();
    gpu.free_tensor(d_wout).ok();
}
