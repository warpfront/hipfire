// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The BF16-weight x F32-activation GEMMs (`gemm_bf16_xf32_batched`,
//! `gemm_bf16_xf32_tiled_rows`) must be bit-identical, for every output, to
//! `gemm_f32_batched` run on the exactly widened weight (same lane-strided FMA
//! order, same width-32 shuffle tree). Covers the Raon mHC selector (M=17,
//! K=10240), the MoE router (M=128, K=2560) at decode / verify / prefill row
//! counts and odd partial tiles, with weights spanning normal and subnormal
//! BF16 magnitudes and both signs.
//!
//! `#[ignore]`d: needs a HIP GPU. Run explicitly:
//!
//!   cargo test -p rdna-compute --release --test gemm_bf16_xf32_parity -- --ignored

use rdna_compute::{DType, Gpu, GpuTensor};

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

/// Random BF16 words: mostly checkpoint-like magnitudes, with a share of
/// subnormals, signed zeros and large exponents (finite only).
fn rand_bf16_bits(seed: &mut u64, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| {
            let r = next(seed);
            let sign = ((r >> 20) as u16 & 1) << 15;
            match r % 16 {
                0 => sign | (r >> 8) as u16 & 0x007f,            // subnormal
                1 => sign,                                       // +-0
                2 => sign | (0x4000 + (r >> 8) as u16 % 0x1000), // large
                _ => {
                    // small checkpoint-like magnitude, exponent 0x38..0x3f
                    let e = 0x3800u16 + ((r >> 8) as u16 % 0x0800);
                    sign | (e & 0x7f80) | ((r >> 16) as u16 & 0x007f)
                }
            }
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn upload_bf16(gpu: &mut Gpu, w: &[u16], m: usize, k: usize) -> GpuTensor {
    let bytes: Vec<u8> = w.iter().flat_map(|b| b.to_le_bytes()).collect();
    let mut t = gpu.upload_raw(&bytes, &[m, k]).expect("upload bf16");
    t.dtype = DType::BF16;
    t
}

#[test]
#[ignore = "needs a HIP GPU"]
fn gemm_bf16_xf32_bit_identical_to_f32_batched() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut seed = 0x7143_0002u64;
    let mut shapes: Vec<(usize, usize, usize)> = Vec::new();
    for n in [1usize, 2, 7, 16, 512, 4096] {
        shapes.push((17, 10240, n));
        shapes.push((128, 2560, n));
    }
    shapes.push((33, 1000, 9));
    shapes.push((5, 31, 3));
    shapes.push((300, 700, 6));
    shapes.push((129, 2593, 513));
    shapes.push((1, 33, 1));
    for (m, k, n) in shapes {
        let w_bits = rand_bf16_bits(&mut seed, m * k);
        let w_f32: Vec<f32> = w_bits
            .iter()
            .map(|&b| f32::from_bits((b as u32) << 16))
            .collect();
        let w_ref = gpu.upload_f32(&w_f32, &[m, k]).expect("upload f32 w");
        let w_bf = upload_bf16(&mut gpu, &w_bits, m, k);
        let x = gpu
            .upload_f32(&rand_f32(&mut seed, n * k), &[n, k])
            .expect("upload x");
        let alloc = |gpu: &mut Gpu| gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y");
        let want = alloc(&mut gpu);
        let got_batched = alloc(&mut gpu);
        let got_tiled = alloc(&mut gpu);
        gpu.gemm_f32_batched(&w_ref, &x, &want, m, k, n)
            .expect("f32 batched");
        gpu.gemm_bf16_xf32_batched(&w_bf, &x, &got_batched, m, k, n)
            .expect("bf16 batched");
        gpu.gemm_bf16_xf32_tiled_rows(&w_bf, &x, &got_tiled, m, k, n)
            .expect("bf16 tiled");
        let want_bits = bits(&gpu.download_f32(&want).unwrap());
        assert_eq!(
            want_bits,
            bits(&gpu.download_f32(&got_batched).unwrap()),
            "batched (M,K,N)=({m},{k},{n})"
        );
        assert_eq!(
            want_bits,
            bits(&gpu.download_f32(&got_tiled).unwrap()),
            "tiled (M,K,N)=({m},{k},{n})"
        );
    }
}

#[test]
#[ignore = "needs a HIP GPU"]
fn gemm_bf16_xf32_rejects_wrong_views() {
    let mut gpu = Gpu::init().expect("gpu init");
    let (m, k, n) = (4usize, 64usize, 2usize);
    let w_f32 = gpu.upload_f32(&vec![0.5; m * k], &[m, k]).unwrap();
    let w_bf = upload_bf16(&mut gpu, &vec![0x3f00u16; m * k], m, k);
    let short_bf = upload_bf16(&mut gpu, &vec![0x3f00u16; m * k - 1], 1, m * k - 1);
    let x = gpu.upload_f32(&vec![1.0; n * k], &[n, k]).unwrap();
    let y = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
    // An F32-typed weight is refused: the loader's F32 fallback takes gemm_f32_batched.
    assert!(gpu.gemm_bf16_xf32_batched(&w_f32, &x, &y, m, k, n).is_err());
    assert!(gpu
        .gemm_bf16_xf32_tiled_rows(&w_f32, &x, &y, m, k, n)
        .is_err());
    // A weight window shorter than [m, k] is refused.
    assert!(gpu
        .gemm_bf16_xf32_batched(&short_bf, &x, &y, m, k, n)
        .is_err());
    assert!(gpu
        .gemm_bf16_xf32_tiled_rows(&short_bf, &x, &y, m, k, n)
        .is_err());
    // The valid call succeeds.
    gpu.gemm_bf16_xf32_batched(&w_bf, &x, &y, m, k, n).unwrap();
}
