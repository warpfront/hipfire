// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The BF16-weight GEMMs (`gemm_bf16_xf32_batched`, `gemm_bf16_xf32_tiled_rows`
//! and the register-tiled `gemm_bf16_small_rows` profiles with F32 and BF16
//! activations) must be bit-identical, for every output, to
//! `gemm_f32_batched` run on the exactly widened weight (same lane-strided FMA
//! order, same width-32 shuffle tree). Covers the Raon mHC selector (M=17,
//! K=10240), the MoE router (M=128, K=2560) at decode / verify / prefill row
//! counts and odd partial tiles, with weights spanning normal and subnormal
//! BF16 magnitudes and both signs.
//!
//! `#[ignore]`d: needs a HIP GPU. Run explicitly:
//!
//!   cargo test -p rdna-compute --release --test gemm_bf16_xf32_parity -- --ignored

use rdna_compute::gemm::Bf16SmallRowsProfile;
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

/// Checkpoint-like BF16 activation words: the high halves of F32 values in [-2, 2], so a BF16 `x`
/// widens exactly (`bits << 16`) to the F32 the reference kernel reads.
fn rand_bf16_activations(seed: &mut u64, n: usize) -> Vec<u16> {
    rand_f32(seed, n)
        .iter()
        .map(|v| (v.to_bits() >> 16) as u16)
        .collect()
}

fn widen_bf16(w: &[u16]) -> Vec<f32> {
    w.iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect()
}

/// `gemm_bf16_small_rows`: every profile with an F32 activation (router shape) and a BF16 activation
/// (selector shape) is bit-identical to `gemm_f32_batched` on the exactly widened weight and
/// activation, at router / selector / odd M, K a multiple of 512 (the kernels have no k tail: 512 is
/// one unrolled group, 2560 / 10240 the Raon router / selector K), and row counts around every
/// profile boundary (partial row groups, one and several full groups, 513 rows).
#[test]
#[ignore = "needs a HIP GPU"]
fn gemm_bf16_small_rows_bit_identical_to_f32_batched() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut seed = 0x7143_0003u64;
    let profiles = [
        Bf16SmallRowsProfile::Rows1,
        Bf16SmallRowsProfile::Rows4,
        Bf16SmallRowsProfile::Rows8,
    ];
    let ns = [1usize, 2, 5, 8, 9, 16, 33, 64, 255, 257, 512, 513];
    for m in [17usize, 128, 200, 3] {
        for k in [512usize, 2560, 5120, 10240] {
            let w_bits = rand_bf16_bits(&mut seed, m * k);
            let w_ref = gpu
                .upload_f32(&widen_bf16(&w_bits), &[m, k])
                .expect("upload f32 w");
            let w_bf = upload_bf16(&mut gpu, &w_bits, m, k);
            for n in ns {
                let x_f32 = rand_f32(&mut seed, n * k);
                let x_bits = rand_bf16_activations(&mut seed, n * k);
                let x_f32_t = gpu.upload_f32(&x_f32, &[n, k]).expect("upload x f32");
                let x_bf_t = upload_bf16(&mut gpu, &x_bits, n, k);
                let x_bf_ref = gpu
                    .upload_f32(&widen_bf16(&x_bits), &[n, k])
                    .expect("upload widened x");
                let want_f32 = gpu.zeros(&[n * m], DType::F32).expect("alloc y");
                let want_bf = gpu.zeros(&[n * m], DType::F32).expect("alloc y");
                gpu.gemm_f32_batched(&w_ref, &x_f32_t, &want_f32, m, k, n)
                    .expect("f32 batched (f32 x)");
                gpu.gemm_f32_batched(&w_ref, &x_bf_ref, &want_bf, m, k, n)
                    .expect("f32 batched (widened bf16 x)");
                let want_f32_bits = bits(&gpu.download_f32(&want_f32).unwrap());
                let want_bf_bits = bits(&gpu.download_f32(&want_bf).unwrap());
                for profile in profiles {
                    let got_f32 = gpu.zeros(&[n * m], DType::F32).expect("alloc y");
                    gpu.gemm_bf16_small_rows(&w_bf, &x_f32_t, &got_f32, m, k, n, profile)
                        .expect("small rows (f32 x)");
                    assert_eq!(
                        want_f32_bits,
                        bits(&gpu.download_f32(&got_f32).unwrap()),
                        "small rows {profile:?} F32 x (M,K,N)=({m},{k},{n})"
                    );
                    let got_bf = gpu.zeros(&[n * m], DType::F32).expect("alloc y");
                    gpu.gemm_bf16_small_rows(&w_bf, &x_bf_t, &got_bf, m, k, n, profile)
                        .expect("small rows (bf16 x)");
                    assert_eq!(
                        want_bf_bits,
                        bits(&gpu.download_f32(&got_bf).unwrap()),
                        "small rows {profile:?} BF16 x (M,K,N)=({m},{k},{n})"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "needs a HIP GPU"]
fn gemm_bf16_small_rows_rejects_wrong_views() {
    let mut gpu = Gpu::init().expect("gpu init");
    let (m, k, n) = (4usize, 512usize, 2usize);
    let profile = Bf16SmallRowsProfile::Rows1;
    let w_f32 = gpu.upload_f32(&vec![0.5; m * k], &[m, k]).unwrap();
    let w_bf = upload_bf16(&mut gpu, &vec![0x3f00u16; m * k], m, k);
    let short_bf = upload_bf16(&mut gpu, &vec![0x3f00u16; m * k - 1], 1, m * k - 1);
    let x = gpu.upload_f32(&vec![1.0; n * k], &[n, k]).unwrap();
    let short_x_bf = upload_bf16(&mut gpu, &vec![0x3f80u16; n * k - 1], 1, n * k - 1);
    let y = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
    // An F32-typed weight is refused: the loader's F32 fallback takes gemm_f32_batched.
    assert!(gpu
        .gemm_bf16_small_rows(&w_f32, &x, &y, m, k, n, profile)
        .is_err());
    // A weight window shorter than [m, k] is refused.
    assert!(gpu
        .gemm_bf16_small_rows(&short_bf, &x, &y, m, k, n, profile)
        .is_err());
    // A BF16 activation window shorter than [n, k] is refused.
    assert!(gpu
        .gemm_bf16_small_rows(&w_bf, &short_x_bf, &y, m, k, n, profile)
        .is_err());
    // The valid call succeeds.
    gpu.gemm_bf16_small_rows(&w_bf, &x, &y, m, k, n, profile)
        .unwrap();
    // K must be a multiple of 512 (the kernels have no k tail): ragged K is refused even when every
    // view is large enough for it.
    for bad_k in [2600usize, 40, 511, 513] {
        let w = upload_bf16(&mut gpu, &vec![0x3f00u16; m * bad_k], m, bad_k);
        let xf = gpu.upload_f32(&vec![1.0; n * bad_k], &[n, bad_k]).unwrap();
        let xb = upload_bf16(&mut gpu, &vec![0x3f80u16; n * bad_k], n, bad_k);
        for p in [
            Bf16SmallRowsProfile::Rows1,
            Bf16SmallRowsProfile::Rows4,
            Bf16SmallRowsProfile::Rows8,
        ] {
            assert!(
                gpu.gemm_bf16_small_rows(&w, &xf, &y, m, bad_k, n, p)
                    .is_err(),
                "F32 x, K = {bad_k}, {p:?} must be refused"
            );
            assert!(
                gpu.gemm_bf16_small_rows(&w, &xb, &y, m, bad_k, n, p)
                    .is_err(),
                "BF16 x, K = {bad_k}, {p:?} must be refused"
            );
        }
    }
}
