// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — deterministic dense mfp3-E8 GPU/CPU GEMV parity gate.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const QUANT_STEP: f32 = 1.8;

fn e4m3_decode(b: u8) -> f32 {
    let exp = ((b >> 3) & 0xf) as i32;
    let mant = (b & 0x7) as u32;
    if exp == 0 {
        return 0.015625 * mant as f32 * 0.125;
    }
    if exp == 0xf && mant == 7 {
        return 448.0;
    }
    2.0f32.powi(exp - 7) * (1.0 + mant as f32 * 0.125)
}

fn decode_index(idx: u32) -> [f32; 8] {
    let coset = (idx >> 23) & 1;
    let mut e = [0u32; 8];
    let mut sum = 0u32;
    for (i, slot) in e.iter_mut().take(7).enumerate() {
        *slot = (idx >> (3 * i)) & 0x7;
        sum += *slot;
    }
    let p7 = ((idx >> 21) & 0x3) << 1;
    e[7] = p7 | ((sum + p7) & 1);
    let mut out = [0.0f32; 8];
    for i in 0..8 {
        let c = e[i] as f32 - 4.0;
        out[i] = if coset != 0 { c + 0.5 } else { c };
    }
    out
}

fn next_u32(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

fn build_weights(m: usize, k: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(k % 256, 0);
    let blocks = k / 32;
    let row_bytes = 16 + blocks * 13;
    let mut packed = vec![0u8; m * row_bytes];
    let mut decoded = vec![0.0f32; m * k];
    let mut rng = seed;

    for row in 0..m {
        let row_off = row * row_bytes;
        packed[row_off..row_off + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
        packed[row_off + 4..row_off + 6].copy_from_slice(&(blocks as u16).to_le_bytes());
        packed[row_off + 6] = 0x07;
        for block in 0..blocks {
            let block_off = row_off + 16 + block * 13;
            let scale_code = [0x30, 0x38, 0x3c, 0x40][(row + block) & 3];
            packed[block_off] = scale_code;
            let scale = e4m3_decode(scale_code) * QUANT_STEP;
            for cw in 0..4 {
                let idx = next_u32(&mut rng) & 0x00ff_ffff;
                let bytes = idx.to_le_bytes();
                let code_off = block_off + 1 + cw * 3;
                packed[code_off..code_off + 3].copy_from_slice(&bytes[..3]);
                let coords = decode_index(idx);
                for i in 0..8 {
                    decoded[row * k + block * 32 + cw * 8 + i] = scale * coords[i];
                }
            }
        }
    }
    (packed, decoded)
}

fn run_shape(gpu: &mut Gpu, m: usize, k: usize, seed: u32) -> bool {
    let (packed, decoded) = build_weights(m, k, seed);
    let mut rng = seed ^ 0xa5a5_5a5a;
    let x: Vec<f32> = (0..k)
        .map(|_| {
            let bits = next_u32(&mut rng);
            ((bits >> 8) as i32 % 2001 - 1000) as f32 * 0.0005
        })
        .collect();
    let mut reference = vec![0.0f32; m];
    for row in 0..m {
        let mut sum = 0.0f64;
        for col in 0..k {
            sum += decoded[row * k + col] as f64 * x[col] as f64;
        }
        reference[row] = sum as f32;
    }

    let mut weights = gpu.upload_raw(&packed, &[packed.len()]).unwrap();
    weights.dtype = DType::MFP3G32E8;
    let x_gpu = gpu.upload_f32(&x, &[k]).unwrap();
    let y_gpu = gpu.zeros(&[m], DType::F32).unwrap();
    gpu.gemv_mfp3g32_e8(&weights, &x_gpu, &y_gpu, m, k)
        .unwrap();
    let actual = gpu.download_f32(&y_gpu).unwrap();

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for row in 0..m {
        let abs = (actual[row] - reference[row]).abs();
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / reference[row].abs().max(1e-3));
    }
    let finite = actual.iter().all(|v| v.is_finite());
    let pass = finite && max_rel < 2.0e-3 && max_abs < 2.0e-2;
    println!(
        "{} M={m} K={k} seed={seed:#010x} bytes={} max_abs={max_abs:.6e} max_rel={max_rel:.6e}",
        if pass { "PASS" } else { "FAIL" },
        packed.len()
    );
    pass
}

fn bench_shape(gpu: &mut Gpu, m: usize, k: usize, occurrences: usize, seed: u32) -> f64 {
    let (packed, _) = build_weights(m, k, seed);
    let mut rng = seed ^ 0xa5a5_5a5a;
    let x: Vec<f32> = (0..k)
        .map(|_| {
            let bits = next_u32(&mut rng);
            ((bits >> 8) as i32 % 2001 - 1000) as f32 * 0.0005
        })
        .collect();
    let mut weights = gpu.upload_raw(&packed, &[packed.len()]).unwrap();
    weights.dtype = DType::MFP3G32E8;
    let x_gpu = gpu.upload_f32(&x, &[k]).unwrap();
    let y_gpu = gpu.zeros(&[m], DType::F32).unwrap();
    gpu.gemv_mfp3g32_e8(&weights, &x_gpu, &y_gpu, m, k)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();

    let trials = 200usize;
    let started = Instant::now();
    for _ in 0..trials {
        gpu.gemv_mfp3g32_e8(&weights, &x_gpu, &y_gpu, m, k)
            .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let us = started.elapsed().as_secs_f64() * 1.0e6 / trials as f64;
    println!(
        "BENCH M={m} K={k} occurrences={occurrences} us={us:.6} route_ms={:.6}",
        us * occurrences as f64 / 1.0e3
    );
    us * occurrences as f64 / 1.0e3
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch={}", gpu.arch);
    let mut all_pass = true;
    if std::env::var("HIPFIRE_MFP3_SKIP_ORACLE").ok().as_deref() != Some("1") {
        let shapes = [(33usize, 1024usize), (65, 4096), (17, 8192)];
        let seeds = [0x3141_5926u32, 0x2718_2818, 0xdead_beef];
        for (m, k) in shapes {
            for seed in seeds {
                all_pass &= run_shape(&mut gpu, m, k, seed);
            }
        }
    }
    if !all_pass {
        std::process::exit(1);
    }
    if std::env::var("HIPFIRE_MFP3_PROD_BENCH").ok().as_deref() == Some("1") {
        let route_ms = bench_shape(&mut gpu, 1024, 4096, 176, 0x1024_4096)
            + bench_shape(&mut gpu, 4096, 8192, 43, 0x4096_8192)
            + bench_shape(&mut gpu, 32768, 1024, 43, 0x3276_8102);
        println!("BENCH_ROUTE ms={route_ms:.6}");
    }
}
