// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.8-27B verify shape (24 heads, hd 256, Q8 KV, 8 rows): multi-row vs
//! the existing batched route, with achieved KV bandwidth. `--check` compares the two outputs
//! and fails above a 1e-3 relative envelope (same reduce, different scan).

use rdna_compute::{DType, Gpu};

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) & 0x7fff) as f32 / 32_768.0 - 0.5
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let argval = |k: &str, d: usize| {
        args.iter()
            .position(|a| a == k)
            .map(|i| args[i + 1].parse().unwrap())
            .unwrap_or(d)
    };
    let seq_len = argval("--seq", 33014);
    let batch = argval("--rows", 8);
    let iters = argval("--iters", 50);
    let n_heads = argval("--heads", 24);
    let n_kv_heads = argval("--kv-heads", 4);
    let head_dim = argval("--head-dim", 256);

    let max_seq = 65536usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    let mut gpu = Gpu::init().expect("GPU init");
    let tile = gpu.attn_tile_size();
    eprintln!(
        "GPU: {}  seq={seq_len} rows={batch} heads={n_heads}/{n_kv_heads} hd={head_dim} tile={tile}",
        gpu.arch
    );

    let d_q = gpu
        .upload_f32(&lcg(0xa5a5, batch * q_dim), &[batch * q_dim])
        .unwrap();
    let d_kf = gpu
        .upload_f32(&lcg(0xc3c3, max_seq * kv_dim), &[max_seq * kv_dim])
        .unwrap();
    let d_vf = gpu
        .upload_f32(&lcg(0x9696, max_seq * kv_dim), &[max_seq * kv_dim])
        .unwrap();
    let kv_bytes_total = max_seq * n_kv_heads * (head_dim / 32) * 34;
    let d_k = gpu.alloc_tensor(&[kv_bytes_total], DType::Q8_0).unwrap();
    let d_v = gpu.alloc_tensor(&[kv_bytes_total], DType::Q8_0).unwrap();
    let all_pos: Vec<u8> = (0..max_seq as i32).flat_map(|p| p.to_ne_bytes()).collect();
    let d_all_pos = gpu.alloc_tensor(&[max_seq], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&d_all_pos.buf, &all_pos).unwrap();
    let mut written = 0usize;
    while written < max_seq {
        let chunk = (max_seq - written).min(8192);
        let pos_view = d_all_pos.sub_offset(written, chunk);
        let kf_view = d_kf.sub_offset(written * kv_dim, chunk * kv_dim);
        let vf_view = d_vf.sub_offset(written * kv_dim, chunk * kv_dim);
        gpu.kv_cache_write_q8_0_batched(&d_k, &kf_view, &pos_view, n_kv_heads, head_dim, chunk)
            .unwrap();
        gpu.kv_cache_write_q8_0_batched(&d_v, &vf_view, &pos_view, n_kv_heads, head_dim, chunk)
            .unwrap();
        written += chunk;
    }
    gpu.hip.device_synchronize().unwrap();

    let check = args.iter().any(|a| a == "--check");
    let d_out = gpu.zeros(&[batch * q_dim], DType::F32).unwrap();
    let d_out_batched = gpu.zeros(&[batch * q_dim], DType::F32).unwrap();
    let max_tiles = seq_len.div_ceil(tile);
    let d_part = gpu
        .zeros(
            &[16 * n_heads * (max_seq / tile) * (2 + head_dim)],
            DType::F32,
        )
        .unwrap();

    let positions: Vec<u8> = (0..batch)
        .flat_map(|i| ((seq_len - batch + i) as i32).to_ne_bytes())
        .collect();
    let d_pos = gpu.alloc_tensor(&[batch], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&d_pos.buf, &positions).unwrap();

    let kv_bytes = (seq_len * n_kv_heads * (head_dim / 32) * 34 * 2) as f64;
    eprintln!(
        "KV footprint per layer-call: {:.1} MB   partials {:.1} MB",
        kv_bytes / 1e6,
        (max_tiles * n_heads * batch * (2 + head_dim) * 4) as f64 / 1e6
    );

    let run_rows = |gpu: &mut Gpu| {
        gpu.attention_flash_q8_0_rows_masked(
            &d_q, &d_k, &d_v, &d_out, &d_pos, n_heads, n_kv_heads, head_dim, seq_len, batch,
            &d_part,
        )
        .unwrap()
    };
    if !run_rows(&mut gpu) {
        eprintln!("multi-row kernel refused this shape");
        return;
    }
    gpu.hip.device_synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..iters {
        run_rows(&mut gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    eprintln!(
        "rows kernel:   {us:8.1} us/call   {:.0} GB/s of KV",
        kv_bytes / (us * 1e3)
    );

    let run_batched = |gpu: &mut Gpu| {
        gpu.attention_flash_q8_0_batched_masked(
            &d_q,
            &d_k,
            &d_v,
            &d_out_batched,
            &d_pos,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            seq_len,
            batch,
            &d_part,
            None,
            0,
            0,
        )
        .unwrap()
    };
    run_batched(&mut gpu);
    gpu.hip.device_synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..iters {
        run_batched(&mut gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    let us_b = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    eprintln!(
        "batched (ROW): {us_b:8.1} us/call   {:.0} GB/s of KV×rows",
        kv_bytes * batch as f64 / (us_b * 1e3)
    );
    eprintln!("speedup: {:.2}x", us_b / us);

    let rows_out = gpu.download_f32(&d_out).unwrap();
    let batched_out = gpu.download_f32(&d_out_batched).unwrap();
    let scale = batched_out
        .iter()
        .fold(0.0f32, |m, x| m.max(x.abs()))
        .max(1e-12);
    let max_abs = rows_out
        .iter()
        .zip(&batched_out)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    let rel = max_abs / scale;
    eprintln!("parity vs batched: max_abs={max_abs:.3e} rel={rel:.3e}");
    if check && !(rel < 1e-3) {
        eprintln!("FAIL: multi-row output diverges from the batched kernel");
        std::process::exit(1);
    }
}
