// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1151 E8 small-M K-split screening harness.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    let iters = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1000)
        .max(1);
    let mut gpu = Gpu::init().expect("GPU init");
    assert_eq!(gpu.arch, "gfx1151");

    for m in [64usize, 256, 512] {
        let k = 4096usize;
        let aos = synth_e8_aos(m, k, 0x1234_5678 ^ m as u64);
        let soa = aos_to_soa_full(&aos, m, k);
        let weight = gpu.upload_raw(&soa, &[soa.len()]).expect("upload weight");
        let x = gpu.alloc_tensor(&[k], DType::F32).expect("alloc x");
        let baseline = gpu.alloc_tensor(&[m], DType::F32).expect("alloc baseline");
        let candidate = gpu.alloc_tensor(&[m], DType::F32).expect("alloc candidate");
        let host_x = make_x(k, 0x9abc_def0);
        gpu.hip.memcpy_htod(&x.buf, bytes_of(&host_x)).unwrap();

        for _ in 0..50 {
            gpu.gemv_mfp4g32_e8_soa_u4(&weight, &x, &baseline, m, k)
                .unwrap();
            gpu.gemv_mfp4g32_e8_soa_u4_ksplit4_gfx1151(
                &weight, &x, &candidate, m, k,
            )
            .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();

        let mut baseline_host = vec![0.0f32; m];
        let mut candidate_host = vec![0.0f32; m];
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut baseline_host), &baseline.buf)
            .unwrap();
        gpu.hip
            .memcpy_dtoh(bytes_of_mut(&mut candidate_host), &candidate.buf)
            .unwrap();
        let bit_exact = baseline_host
            .iter()
            .zip(&candidate_host)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits());

        let baseline_us = timed(iters, &mut gpu, |gpu| {
            gpu.gemv_mfp4g32_e8_soa_u4(&weight, &x, &baseline, m, k)
                .unwrap();
        });
        let candidate_us = timed(iters, &mut gpu, |gpu| {
            gpu.gemv_mfp4g32_e8_soa_u4_ksplit4_gfx1151(
                &weight, &x, &candidate, m, k,
            )
            .unwrap();
        });
        let bytes = soa.len() as f64;
        println!(
            "M={m} K={k} bytes={} baseline_us={baseline_us:.3} candidate_us={candidate_us:.3} \
             baseline_gbps={:.3} candidate_gbps={:.3} speedup={:.6} bit_exact={bit_exact}",
            soa.len(),
            bytes / (baseline_us * 1e-6) / 1e9,
            bytes / (candidate_us * 1e-6) / 1e9,
            baseline_us / candidate_us,
        );
        assert!(bit_exact, "K-split output differs at M={m}");
    }
}

fn timed<F: FnMut(&mut Gpu)>(iters: usize, gpu: &mut Gpu, mut launch: F) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        launch(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    start.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn aos_to_soa_row(aos_row: &[u8], n_blocks: usize) -> Vec<u8> {
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let mut out = vec![0u8; 16 + scale_padded + n_blocks * 16];
    out[..16].copy_from_slice(&aos_row[..16]);
    out[6] = 0x06;
    for block in 0..n_blocks {
        out[16 + block] = aos_row[16 + block * 17];
    }
    let codewords = 16 + scale_padded;
    for block in 0..n_blocks {
        let source = 16 + block * 17 + 1;
        let destination = codewords + block * 16;
        out[destination..destination + 16]
            .copy_from_slice(&aos_row[source..source + 16]);
    }
    out
}

fn aos_to_soa_full(aos: &[u8], m: usize, k: usize) -> Vec<u8> {
    let n_blocks = k / 32;
    let aos_row_bytes = 16 + n_blocks * 17;
    let mut out = Vec::new();
    for row in 0..m {
        out.extend_from_slice(&aos_to_soa_row(
            &aos[row * aos_row_bytes..(row + 1) * aos_row_bytes],
            n_blocks,
        ));
    }
    out
}

fn synth_e8_aos(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let blocks_per_row = k / 32;
    let row_bytes = 16 + blocks_per_row * 17;
    let mut out = vec![0u8; m * row_bytes];
    let mut state = seed;
    let mut rng = || -> u32 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        let offset = row * row_bytes;
        out[offset..offset + 2].copy_from_slice(&0x2400u16.to_le_bytes());
        out[offset + 4..offset + 6]
            .copy_from_slice(&(blocks_per_row as u16).to_le_bytes());
        out[offset + 6] = 0x05;
        for block in 0..blocks_per_row {
            let block_offset = offset + 16 + block * 17;
            out[block_offset] = 120u8.wrapping_add((rng() & 0x3f) as u8);
            for word in 0..4 {
                out[block_offset + 1 + word * 4..block_offset + 5 + word * 4]
                    .copy_from_slice(&rng().to_le_bytes());
            }
        }
    }
    out
}

fn make_x(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) * 2.3e-10 - 0.5
        })
        .collect()
}

fn bytes_of(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

fn bytes_of_mut(values: &mut [f32]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}
