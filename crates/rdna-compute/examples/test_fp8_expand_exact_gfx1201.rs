// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Bit-exactness dump for the gfx1201 FP8-WMMA MQ4v2 uniform kernels
//! (gate_up / residual / qkv / qkvza, all compiled from
//! `gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip`).
//!
//! Runs the four production (uniform, scale_mode=1) entries at
//! N in {64, 192, 512, 1024} (covers BT4/BT12/S2BT8 batch tiles,
//! SLABS=1 and SLABS=2) over deterministic synthetic MQ4v2 weights
//! that cycle all 16 nibble values and sprinkle zero-scale (dead)
//! halves plus zero/nonzero zero-points, then prints one FNV-1a-64
//! per output and optionally dumps raw bytes for `cmp`.
//!
//! Usage: test_fp8_expand_exact_gfx1201 [--dump-dir DIR]
//! Compare across builds: run on baseline, save stdout; run on the
//! candidate; `diff`. Any difference = exactness violation.
//! Skips silently with PASS on non-gfx1201 archs.

use rdna_compute::kv_slots::half_from_f32;
use rdna_compute::{DType, Gpu};

const K: usize = 512;
const GPR: usize = K / 256;
const GB: usize = 136; // 8 header + 128 payload per 256-K group row
const NS: [usize; 4] = [64, 192, 512, 1024];

fn prng(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Synthetic uniform MQ4v2 rows: per 256-group, fp16 (sc,zp) headers per
/// 128-half + 128 payload bytes (LSB-first nibbles). Cycles every nibble
/// 0..15; every 5th half has scale 0 (dead half); zero-points vary.
fn synth_weights(m: usize, seed: u64) -> Vec<u8> {
    let mut st = seed;
    let mut out = vec![0u8; m * GPR * GB];
    for r in 0..m {
        for g in 0..GPR {
            let dst = (r * GPR + g) * GB;
            for h in 0..2 {
                let dead = (r + g * 2 + h) % 5 == 4;
                let sc: f32 = if dead {
                    0.0
                } else {
                    0.01 * (1 + ((prng(&mut st) >> 11) % 200) as u32) as f32
                };
                let zp: f32 = if (r + h) % 3 == 2 {
                    0.0
                } else {
                    -4.0 + 0.05 * ((prng(&mut st) >> 11) % 160) as f32
                };
                out[dst + h * 4..dst + h * 4 + 2]
                    .copy_from_slice(&half_from_f32(sc).to_le_bytes());
                out[dst + h * 4 + 2..dst + h * 4 + 4]
                    .copy_from_slice(&half_from_f32(zp).to_le_bytes());
            }
            for i in 0..128 {
                let q0 = ((r * 131 + g * 17 + i * 7 + 3) % 16) as u8;
                let q1 = ((r * 131 + g * 17 + i * 7 + 11) % 16) as u8;
                out[dst + 8 + i] = q0 | (q1 << 4);
            }
        }
    }
    out
}

fn synth_x(nk: usize, seed: u64) -> Vec<f32> {
    let mut st = seed;
    (0..nk)
        .map(|_| {
            let u = (prng(&mut st) >> 11) as f32 / (1u64 << 53) as f32;
            (u - 0.5) * 4.0
        })
        .collect()
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1201" {
        eprintln!("=== SKIP === exact gfx1201 only (arch={})", gpu.arch);
        return;
    }
    let dump_dir = std::env::args()
        .skip_while(|a| a != "--dump-dir")
        .nth(1);
    if let Some(d) = &dump_dir {
        std::fs::create_dir_all(d).unwrap();
    }

    let max_n = *NS.iter().max().unwrap();
    let x_host = synth_x(max_n * K, 0xC0FFEE);
    let x_full = gpu.alloc_tensor(&[max_n * K], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&x_full.buf, bytes_of(&x_host)).unwrap();

    // Weight blobs (shared across N).
    let w_gate = gpu.upload_raw(&synth_weights(256, 0xA1), &[256 * GPR * GB]).unwrap();
    let w_up = gpu.upload_raw(&synth_weights(256, 0xA2), &[256 * GPR * GB]).unwrap();
    let w_q = gpu.upload_raw(&synth_weights(256, 0xB1), &[256 * GPR * GB]).unwrap();
    let w_k = gpu.upload_raw(&synth_weights(128, 0xB2), &[128 * GPR * GB]).unwrap();
    let w_v = gpu.upload_raw(&synth_weights(128, 0xB3), &[128 * GPR * GB]).unwrap();
    let w_qkv = gpu.upload_raw(&synth_weights(256, 0xC1), &[256 * GPR * GB]).unwrap();
    let w_z = gpu.upload_raw(&synth_weights(64, 0xC2), &[64 * GPR * GB]).unwrap();
    let w_b = gpu.upload_raw(&synth_weights(64, 0xC3), &[64 * GPR * GB]).unwrap();
    let w_a = gpu.upload_raw(&synth_weights(64, 0xC4), &[64 * GPR * GB]).unwrap();
    let w_res = gpu.upload_raw(&synth_weights(512, 0xD1), &[512 * GPR * GB]).unwrap();

    // Output buffers at max N.
    let y_gate = gpu.alloc_tensor(&[max_n * 256], DType::F32).unwrap();
    let y_up = gpu.alloc_tensor(&[max_n * 256], DType::F32).unwrap();
    let y_q = gpu.alloc_tensor(&[max_n * 256], DType::F32).unwrap();
    let y_k = gpu.alloc_tensor(&[max_n * 128], DType::F32).unwrap();
    let y_v = gpu.alloc_tensor(&[max_n * 128], DType::F32).unwrap();
    let y_qkv = gpu.alloc_tensor(&[max_n * 256], DType::F32).unwrap();
    let y_z = gpu.alloc_tensor(&[max_n * 64], DType::F32).unwrap();
    let y_beta = gpu.alloc_tensor(&[max_n * 64], DType::F32).unwrap();
    let y_alpha = gpu.alloc_tensor(&[max_n * 64], DType::F32).unwrap();
    let y_res = gpu.alloc_tensor(&[max_n * 512], DType::F32).unwrap();

    // Deterministic residual accumulate seed.
    let res_host: Vec<f32> = synth_x(max_n * 512, 0x5EED);

    for &n in &NS {
        let x = x_full.sub_offset(0, n * K);
        let mut outs: Vec<(&str, Vec<u8>)> = Vec::new();

        // gate_up (256+256).
        {
            let yg = y_gate.sub_offset(0, n * 256);
            let yu = y_up.sub_offset(0, n * 256);
            gpu.gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2_fp8_bt12(
                &w_gate, &w_up, &x, &yg, &yu, 256, 256, K, n, 1,
            )
            .unwrap();
            gpu.hip.device_synchronize().unwrap();
            let mut hg = vec![0u8; n * 256 * 4];
            let mut hu = vec![0u8; n * 256 * 4];
            gpu.hip.memcpy_dtoh(&mut hg, &yg.buf).unwrap();
            gpu.hip.memcpy_dtoh(&mut hu, &yu.buf).unwrap();
            outs.push(("gate", hg));
            outs.push(("up", hu));
        }
        // qkv (256/128/128).
        {
            let yq = y_q.sub_offset(0, n * 256);
            let yk = y_k.sub_offset(0, n * 128);
            let yv = y_v.sub_offset(0, n * 128);
            gpu.gemm_qkv_hfq4g256_wmma_gfx12_mq4v2_fp8(
                &w_q, &w_k, &w_v, &x, &yq, &yk, &yv, 256, 128, 128, K, n, 1,
            )
            .unwrap();
            gpu.hip.device_synchronize().unwrap();
            for (name, t, m) in [("qq", &yq, 256), ("kk", &yk, 128), ("vv", &yv, 128)] {
                let mut h = vec![0u8; n * m * 4];
                gpu.hip.memcpy_dtoh(&mut h, &t.buf).unwrap();
                outs.push((name, h));
            }
        }
        // qkvza (256/64/64/64).
        {
            let yq = y_qkv.sub_offset(0, n * 256);
            let yz = y_z.sub_offset(0, n * 64);
            let yb = y_beta.sub_offset(0, n * 64);
            let ya = y_alpha.sub_offset(0, n * 64);
            gpu.gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2_fp8(
                &w_qkv, &w_z, &w_b, &w_a, &x, &yq, &yz, &yb, &ya, 256, 64, 64, 64,
                K, n, 1,
            )
            .unwrap();
            gpu.hip.device_synchronize().unwrap();
            for (name, t, m) in [
                ("zqkv", &yq, 256),
                ("zz", &yz, 64),
                ("zb", &yb, 64),
                ("za", &ya, 64),
            ] {
                let mut h = vec![0u8; n * m * 4];
                gpu.hip.memcpy_dtoh(&mut h, &t.buf).unwrap();
                outs.push((name, h));
            }
        }
        // residual (512, += over deterministic seed).
        {
            let yr = y_res.sub_offset(0, n * 512);
            gpu.hip
                .memcpy_htod(&yr.buf, bytes_of(&res_host[..n * 512]))
                .unwrap();
            gpu.gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8(&w_res, &x, &yr, 512, K, n, 1)
                .unwrap();
            gpu.hip.device_synchronize().unwrap();
            let mut h = vec![0u8; n * 512 * 4];
            gpu.hip.memcpy_dtoh(&mut h, &yr.buf).unwrap();
            outs.push(("res", h));
        }

        for (name, v) in &outs {
            let h = fnv1a(v);
            println!("fp8exact N={n} {name} len={} fnv={h:016x}", v.len() / 4);
            if let Some(d) = &dump_dir {
                std::fs::write(format!("{d}/N{n}_{name}.bin"), v).unwrap();
            }
        }
    }
    eprintln!("=== DONE === fp8exact");
}
