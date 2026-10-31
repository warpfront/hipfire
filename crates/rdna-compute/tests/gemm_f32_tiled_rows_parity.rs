// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `gemm_f32_tiled_rows` must be bit-identical to `gemm_f32_batched` for every
//! output (same per-lane K order and shuffle tree). Covers the mHC selector
//! (M=17, K=10240), the MoE router (M=128, K=2560) and odd partial tiles.
//!
//! `#[ignore]`d: needs a HIP GPU. Run explicitly:
//!
//!   cargo test -p rdna-compute --release --test gemm_f32_tiled_rows_parity -- --ignored

use rdna_compute::Gpu;

fn next(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 24
}

fn rand_f32(seed: &mut u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| (next(seed) as u32 & 0xff_ffff) as f32 / (1u32 << 24) as f32 * 4.0 - 2.0)
        .collect()
}

#[test]
#[ignore = "needs a HIP GPU"]
fn gemm_f32_tiled_rows_bit_identical_to_batched() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut seed = 0x7143_0001u64;
    let mut shapes: Vec<(usize, usize, usize)> = Vec::new();
    for n in [1usize, 7, 512, 4096] {
        shapes.push((17, 10240, n));
    }
    for n in [1usize, 5, 512, 4096] {
        shapes.push((128, 2560, n));
    }
    shapes.push((33, 1000, 9));
    shapes.push((5, 31, 3));
    shapes.push((300, 700, 6));
    for (m, k, n) in shapes {
        let a = gpu
            .upload_f32(&rand_f32(&mut seed, m * k), &[m * k])
            .expect("upload a");
        let x = gpu
            .upload_f32(&rand_f32(&mut seed, n * k), &[n * k])
            .expect("upload x");
        let want = gpu
            .alloc_tensor(&[n * m], rdna_compute::DType::F32)
            .expect("alloc");
        let got = gpu
            .alloc_tensor(&[n * m], rdna_compute::DType::F32)
            .expect("alloc");
        gpu.gemm_f32_batched(&a, &x, &want, m, k, n)
            .expect("batched");
        gpu.gemm_f32_tiled_rows(&a, &x, &got, m, k, n)
            .expect("tiled");
        let w: Vec<u32> = gpu
            .download_f32(&want)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let g: Vec<u32> = gpu
            .download_f32(&got)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(w, g, "(M,K,N)=({m},{k},{n})");
    }
}
