// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Row-invariance microtest for the batched Q8_0 prefill attention kernels
//! (split-prefill investigation). Synthetic Q (f32) and Q8_0 K/V cache for 64
//! positions; the same query rows are attended as different batches:
//!
//!   R      rows 0..64 (n=64)             reference
//!   T(a,b) rows a..b  (positions a..b)   same rows, different batch window
//!
//! and the per-row outputs are compared with R bit-for-bit. A row-invariant
//! kernel gives 0 for every window. Exercises the gfx11 default Q8 prefill
//! attention kernel, `attention_q8_0_flash_prefill_wmma` (f16 WMMA). The
//! legacy f32 kernel (`HIPFIRE_FLASH_PREFILL=0`) is covered end to end by
//! `split_prefill_probe` instead.
//!
//! Usage: cargo run --release -p hipfire-generate --example attn_wmma_row_probe

fn lcg(s: &mut u64) -> u32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 33) as u32
}

fn f16_bits(x: f32) -> u16 {
    // Round-to-nearest f32 -> f16 for normal range values used here.
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7fffff;
    if exp <= 0 {
        return sign;
    }
    let mut h = sign | ((exp as u16) << 10) | ((man >> 13) as u16);
    if (man >> 12) & 1 == 1 {
        h += 1;
    }
    h
}

fn run_case(gpu: &mut rdna_compute::Gpu, n_heads: usize, n_kv_heads: usize, head_dim: usize) {
    const NP: usize = 64;
    let mut seed = 0x1234_5678u64 ^ (head_dim as u64) << 8;
    let q_stride = n_heads * head_dim;
    let q: Vec<f32> = (0..NP * q_stride)
        .map(|_| (lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0)
        .collect();
    let bph = head_dim / 32;
    let row_bytes = n_kv_heads * bph * 34;
    let mut mk_cache = |seed: &mut u64| {
        let mut c = vec![0u8; NP * row_bytes];
        for p in 0..NP {
            for b in 0..n_kv_heads * bph {
                let off = p * row_bytes + b * 34;
                let sc = 0.02 + (lcg(seed) % 100) as f32 * 0.0005;
                let hb = f16_bits(sc).to_le_bytes();
                c[off] = hb[0];
                c[off + 1] = hb[1];
                for j in 0..32 {
                    c[off + 2 + j] = (lcg(seed) % 255) as u8;
                }
            }
        }
        c
    };
    let kc = mk_cache(&mut seed);
    let vc = mk_cache(&mut seed);
    let kt = gpu.upload_raw(&kc, &[kc.len()]).unwrap();
    let vt = gpu.upload_raw(&vc, &[vc.len()]).unwrap();

    let run = |gpu: &mut rdna_compute::Gpu, a: usize, b: usize| -> Vec<f32> {
        let n = b - a;
        let qt = gpu
            .upload_f32(&q[a * q_stride..b * q_stride], &[n * q_stride])
            .unwrap();
        let pos: Vec<i32> = (a..b).map(|p| p as i32).collect();
        let pb: Vec<u8> = pos.iter().flat_map(|p| p.to_le_bytes()).collect();
        let pt = gpu.upload_raw(&pb, &[n]).unwrap();
        let ot = gpu
            .upload_f32(&vec![0.0f32; n * q_stride], &[n * q_stride])
            .unwrap();
        gpu.attention_q8_0_flash_prefill_wmma(
            &qt, &kt, &vt, &ot, &pt, n_heads, n_kv_heads, head_dim, b, n,
        )
        .unwrap();
        gpu.hip.device_synchronize().unwrap();
        let o = gpu.download_f32(&ot).unwrap();
        gpu.free_tensor(qt).unwrap();
        gpu.free_tensor(pt).unwrap();
        gpu.free_tensor(ot).unwrap();
        o
    };

    let r = run(gpu, 0, NP);
    println!("shape nh={n_heads} nkv={n_kv_heads} hd={head_dim}: WMMA f16 prefill kernel");
    for &(a, b) in &[
        (0usize, 64usize),
        (0, 63),
        (0, 62),
        (0, 49),
        (0, 48),
        (1, 64),
        (2, 64),
        (3, 64),
        (16, 64),
        (17, 64),
        (32, 64),
    ] {
        let o = run(gpu, a, b);
        let mut max_d = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut rows_diff = Vec::new();
        for i in 0..(b - a) {
            let row = a + i;
            let mut rd = 0.0f32;
            for j in 0..q_stride {
                let x = o[i * q_stride + j];
                let y = r[row * q_stride + j];
                rd = rd.max((x - y).abs());
                max_rel = max_rel.max((x - y).abs() / y.abs().max(1e-3));
            }
            if rd > 0.0 {
                rows_diff.push(row);
            }
            max_d = max_d.max(rd);
        }
        let shown: Vec<_> = rows_diff.iter().take(12).collect();
        println!(
            "  window rows {a:2}..{b:2} (n={:2}): max|d|={max_d:.3e} max_rel={max_rel:.2e} rows_differing={} {:?}{}",
            b - a,
            rows_diff.len(),
            shown,
            if rows_diff.len() > 12 { " ..." } else { "" }
        );
    }
    gpu.free_tensor(kt).unwrap();
    gpu.free_tensor(vt).unwrap();
}

fn main() {
    let mut gpu = rdna_compute::Gpu::init().expect("gpu");
    // qwen3-0.6b attention shape, and the Qwen3.5-4B full-attention shape.
    run_case(&mut gpu, 16, 8, 128);
    run_case(&mut gpu, 16, 4, 256);
}
