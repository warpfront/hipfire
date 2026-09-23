// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — decode-GEMV perf comparison: mfp4-E8 vs MQ4G256-Lloyd on gfx1151.
//
// Measures raw decode throughput at batch=1 (N=1, single row of activations)
// for matched weight shapes. Reports µs/call + effective GB/s + ratio.
//
// mfp4-E8 weight bytes: M * (16 + (K/32)*17)  [header 16B + 17B/block]
// MQ4G256-Lloyd weight bytes: M * (K/256) * 160  [160B/group of 256 weights]
//
// PEAK BW: gfx1151 (Strix Halo) LPDDR5X ~256 GB/s

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const PEAK_GBPS: f64 = 256.0;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== E8 vs MQ4G256-Lloyd decode-GEMV bench ===");
    eprintln!("  arch={arch}  peak_bw_gbps={PEAK_GBPS}  (gfx1151 Strix Halo)");
    eprintln!();

    let shapes: Vec<(usize, usize, &str)> = vec![
        (2048,  2048,  "qkv-q      M=2048  K=2048 "),
        (512,   2048,  "qkv-kv     M=512   K=2048 "),
        (11008, 2048,  "gate_up    M=11008 K=2048 "),
        (2048,  11008, "w_down     M=2048  K=11008"),
        (4096,  2048,  "med        M=4096  K=2048 "),
        (1024,  2048,  "small      M=1024  K=2048 "),
    ];

    let warmup = 20usize;
    let trials = 200usize;

    eprintln!(
        "  {:<42}  {:>24}  {:>24}  {:>10}",
        "shape", "mfp4-E8", "MQ4G256-Lloyd", "e8/mq4"
    );
    eprintln!(
        "  {:<42}  {:>24}  {:>24}  {:>10}",
        "-".repeat(42), "-".repeat(24), "-".repeat(24), "-".repeat(10)
    );

    for (m, k, label) in &shapes {
        let (m, k) = (*m, *k);

        // ---- E8 ----
        let e8_row_bytes = 16 + (k / 32) * 17;
        let e8_total = m * e8_row_bytes;
        let e8_w = gpu.upload_raw(&synth_e8(m, k, 0x1234 ^ m as u64 ^ k as u64), &[e8_total]).unwrap();
        let x_e8 = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        let y_e8 = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        {
            let xh = make_x(k, 0xABCD);
            gpu.hip.memcpy_htod(&x_e8.buf, bytes_of(&xh)).unwrap();
        }

        // warmup E8
        for _ in 0..warmup {
            gpu.gemv_mfp4g32_e8(&e8_w, &x_e8, &y_e8, m, k).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();

        let t0 = Instant::now();
        for _ in 0..trials {
            gpu.gemv_mfp4g32_e8(&e8_w, &x_e8, &y_e8, m, k).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let e8_us = t0.elapsed().as_secs_f64() * 1e6 / trials as f64;
        let e8_gbps = e8_total as f64 / (e8_us * 1e-6) / 1e9;
        let e8_pct = e8_gbps / PEAK_GBPS * 100.0;

        // ---- MQ4G256-Lloyd ----
        // MQ4G256-Lloyd weight bytes: M * (K/256) * 160
        // Layout: 160 bytes per group-of-256:
        //   32 bytes = 16 x f16 codebook entries (16 * 2B)
        //   128 bytes = 32 x u32 nibble-packed 4-bit codes (32 * 4B)
        let groups = k / 256;
        let mq4_total = m * groups * 160;
        let mq4_w = gpu.upload_raw(&synth_mq4(m, k, 0x5678 ^ m as u64 ^ k as u64), &[mq4_total]).unwrap();
        let x_mq4 = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        let y_mq4 = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        {
            let xh = make_x(k, 0xABCD);
            gpu.hip.memcpy_htod(&x_mq4.buf, bytes_of(&xh)).unwrap();
        }

        // warmup MQ4
        for _ in 0..warmup {
            gpu.gemv_mq4g256_lloyd(&mq4_w, &x_mq4, &y_mq4, m, k).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();

        let t1 = Instant::now();
        for _ in 0..trials {
            gpu.gemv_mq4g256_lloyd(&mq4_w, &x_mq4, &y_mq4, m, k).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let mq4_us = t1.elapsed().as_secs_f64() * 1e6 / trials as f64;
        let mq4_gbps = mq4_total as f64 / (mq4_us * 1e-6) / 1e9;
        let mq4_pct = mq4_gbps / PEAK_GBPS * 100.0;

        let ratio = e8_us / mq4_us;

        eprintln!(
            "  {:<42}  {:6.2} µs  {:5.1} GB/s ({:4.1}%)  {:6.2} µs  {:5.1} GB/s ({:4.1}%)  {:6.3}x",
            label,
            e8_us, e8_gbps, e8_pct,
            mq4_us, mq4_gbps, mq4_pct,
            ratio
        );
    }

    eprintln!();
    eprintln!("  ratio = e8_time / mq4_time  (ratio > 1.0 means E8 is slower, < 1.0 means E8 is faster)");
}

fn make_x(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as f32) * 2.3e-10 - 0.5
    }).collect()
}

/// Synthesize a plausible mfp4-E8 weight buffer (perf bench — values irrelevant).
/// Row layout: 16B header [f16 row_scale, pad, u16 n_blocks, u8 fmt_tag, pad] + n_blocks * 17B
/// Block layout: 1B E4M3 block_scale + 4 x u32 E8 codewords
fn synth_e8(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let blocks_per_row = k / 32;
    let row_bytes = 16 + blocks_per_row * 17;
    let mut out = vec![0u8; m * row_bytes];
    let mut state = seed;
    let mut rng = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };

    for row in 0..m {
        let roff = row * row_bytes;
        // f16 row scale ~0.01 (bytes 0..2)
        let rs_f16: u16 = 0x2400; // ~0.015625 in f16
        out[roff..roff + 2].copy_from_slice(&rs_f16.to_le_bytes());
        // n_blocks (bytes 4..6)
        out[roff + 4..roff + 6].copy_from_slice(&(blocks_per_row as u16).to_le_bytes());
        // fmt_tag (byte 6)
        out[roff + 6] = 0x05;

        for b in 0..blocks_per_row {
            let bp = roff + 16 + b * 17;
            // E4M3 block scale byte (non-zero, valid range)
            out[bp] = 120u8.wrapping_add((rng() & 0x3F) as u8);
            // 4 x u32 E8 codewords
            for w in 0..4 {
                let codeword = rng();
                out[bp + 1 + w * 4..bp + 1 + w * 4 + 4].copy_from_slice(&codeword.to_le_bytes());
            }
        }
    }
    out
}

/// Synthesize MQ4G256-Lloyd weight buffer (perf bench — values irrelevant).
/// Layout per 160-byte group: 16 x f16 codebook (32B) + 32 x u32 nibble codes (128B)
fn synth_mq4(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let groups = k / 256;
    let mut out = vec![0u8; m * groups * 160];
    let mut state = seed;
    let mut rng = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };

    for row in 0..m {
        for g in 0..groups {
            let goff = (row * groups + g) * 160;
            // 16 x f16 codebook entries (32 bytes)
            for i in 0..16 {
                let v_f16: u16 = 0x3800u16.wrapping_add((rng() & 0xFF) as u16); // small positive f16
                out[goff + i * 2..goff + i * 2 + 2].copy_from_slice(&v_f16.to_le_bytes());
            }
            // 32 x u32 nibble-packed codes (128 bytes)
            for p in 0..32 {
                let code = rng();
                out[goff + 32 + p * 4..goff + 32 + p * 4 + 4].copy_from_slice(&code.to_le_bytes());
            }
        }
    }
    out
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
