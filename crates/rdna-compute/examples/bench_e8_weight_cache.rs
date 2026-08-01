// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1151 E8 dense weight-cache policy screen.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    let iters = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(100)
        .max(1);
    let mut gpu = Gpu::init().expect("GPU init");
    assert_eq!(gpu.arch, "gfx1151");

    for (m, k, label) in [
        (512usize, 4096usize, "small-m"),
        (4096, 8192, "wo-b"),
        (32768, 1024, "wq-b"),
    ] {
        let aos = synth_e8_aos(m, k, 0x57a9_11e5 ^ m as u64 ^ k as u64);
        let soa = aos_to_soa_full(&aos, m, k);
        let weight = gpu.upload_raw(&soa, &[soa.len()]).expect("upload weight");
        let x = gpu.alloc_tensor(&[k], DType::F32).expect("alloc x");
        let incumbent = gpu.alloc_tensor(&[m], DType::F32).expect("alloc incumbent");
        let cpol0 = gpu.alloc_tensor(&[m], DType::F32).expect("alloc cpol0");
        let cpol20 = gpu.alloc_tensor(&[m], DType::F32).expect("alloc cpol20");
        let cpol22 = gpu.alloc_tensor(&[m], DType::F32).expect("alloc cpol22");
        let host_x = make_x(k, 0x1020_3040);
        gpu.hip.memcpy_htod(&x.buf, bytes_of(&host_x)).unwrap();

        for _ in 0..10 {
            launch(&mut gpu, None, &weight, &x, &incumbent, m, k);
            launch(&mut gpu, Some(0), &weight, &x, &cpol0, m, k);
            launch(&mut gpu, Some(20), &weight, &x, &cpol20, m, k);
            launch(&mut gpu, Some(22), &weight, &x, &cpol22, m, k);
        }
        gpu.hip.device_synchronize().unwrap();

        let reference = download(&gpu, &incumbent, m);
        let exact0 = reference == download(&gpu, &cpol0, m);
        let exact20 = reference == download(&gpu, &cpol20, m);
        let exact22 = reference == download(&gpu, &cpol22, m);
        assert!(exact0 && exact20 && exact22, "cache-policy output mismatch");

        let incumbent_us = timed(iters, &mut gpu, |gpu| {
            launch(gpu, None, &weight, &x, &incumbent, m, k)
        });
        let cpol0_us = timed(iters, &mut gpu, |gpu| {
            launch(gpu, Some(0), &weight, &x, &cpol0, m, k)
        });
        let cpol20_us = timed(iters, &mut gpu, |gpu| {
            launch(gpu, Some(20), &weight, &x, &cpol20, m, k)
        });
        let cpol22_us = timed(iters, &mut gpu, |gpu| {
            launch(gpu, Some(22), &weight, &x, &cpol22, m, k)
        });
        let gib = soa.len() as f64 / 1e9;
        println!(
            "shape={label} M={m} K={k} bytes={} incumbent_us={incumbent_us:.3} \
             cpol0_us={cpol0_us:.3} cpol20_us={cpol20_us:.3} cpol22_us={cpol22_us:.3} \
             incumbent_gbps={:.3} cpol0_gbps={:.3} cpol20_gbps={:.3} cpol22_gbps={:.3} \
             exact0={exact0} exact20={exact20} exact22={exact22}",
            soa.len(),
            gib / (incumbent_us * 1e-6),
            gib / (cpol0_us * 1e-6),
            gib / (cpol20_us * 1e-6),
            gib / (cpol22_us * 1e-6),
        );
    }
}

fn launch(
    gpu: &mut Gpu,
    policy: Option<u32>,
    weight: &rdna_compute::GpuTensor,
    x: &rdna_compute::GpuTensor,
    y: &rdna_compute::GpuTensor,
    m: usize,
    k: usize,
) {
    match policy {
        None => gpu.gemv_mfp4g32_e8_soa_u4(weight, x, y, m, k).unwrap(),
        Some(policy) => gpu
            .gemv_mfp4g32_e8_soa_u4_buffer_cpol_gfx1151(policy, weight, x, y, m, k)
            .unwrap(),
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

fn download(gpu: &Gpu, tensor: &rdna_compute::GpuTensor, len: usize) -> Vec<u8> {
    let mut values = vec![0u8; len * 4];
    gpu.hip.memcpy_dtoh(&mut values, &tensor.buf).unwrap();
    values
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
        out[destination..destination + 16].copy_from_slice(&aos_row[source..source + 16]);
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
        out[offset + 4..offset + 6].copy_from_slice(&(blocks_per_row as u16).to_le_bytes());
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
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}
