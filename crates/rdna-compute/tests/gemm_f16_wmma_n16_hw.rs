// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Hardware gate for `Gpu::gemm_f16_wmma_n16_gfx12` (module `gemm_f16_wmma_mb8_gfx12`), the
//! one-column-tile sister of `gemm_f16_wmma_mb8` that `gemm_f16_batched_lmhead` (the F16 GEMV
//! route) takes on gfx12 for 1..=16 rows:
//!
//! * bit-identical to `gemm_f16_wmma_mb8` for every row count 1..=16 over the F16 GEMV shapes
//!   (Inkling-2 router M 128 x K 2304, ragged M, K below / not a multiple of the unroll, wide M);
//! * `gemm_f16_batched_lmhead` routes there (bitwise the same), and the output row past N stays
//!   untouched;
//! * sanity against an f64 reference on the F16-rounded operands;
//! * information: median ms of both kernels at the router shape.
//!
//! `#[ignore]`d: needs a gfx12 GPU. Run explicitly (through the GPU fabric):
//!
//!   cargo test -p rdna-compute --release --test gemm_f16_wmma_n16_hw -- --ignored --nocapture

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

const SENTINEL: u32 = 0x7fc0_dead;

fn upload_f16(gpu: &Gpu, w: &[f32]) -> (GpuTensor, Vec<f32>) {
    let bits: Vec<u16> = w
        .iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect();
    let back = bits
        .iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect();
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    (
        gpu.upload_raw(&bytes, &[bytes.len()]).expect("upload w"),
        back,
    )
}

fn median_ms(gpu: &mut Gpu, mut f: impl FnMut(&mut Gpu)) -> f64 {
    let mut samples = Vec::new();
    for round in 0..14 {
        let t = Instant::now();
        for _ in 0..20 {
            f(gpu);
        }
        gpu.hip.device_synchronize().unwrap();
        if round >= 2 {
            samples.push(t.elapsed().as_secs_f64() * 1e3 / 20.0);
        }
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn run_shape(gpu: &mut Gpu, g: &mut Lcg, m: usize, k: usize, time: bool) {
    let w: Vec<f32> = (0..m * k).map(|_| g.unit() * 0.08).collect();
    let (wt, w_rt) = upload_f16(gpu, &w);
    // Activations span magnitudes so the F32 -> F16 rounding of X matters (and a few hit
    // F16 subnormals / large values).
    let x_all: Vec<f32> = (0..16 * k)
        .map(|i| {
            let v = g.unit();
            match i % 97 {
                0 => v * 3.0e-6,
                1 => v * 900.0,
                _ => v * 1.7,
            }
        })
        .collect();
    let xt = gpu.upload_f32(&x_all, &[x_all.len()]).expect("upload x");
    for n in 1..=16usize {
        let sentinel = vec![f32::from_bits(SENTINEL); n * m + m];
        let y_ref = gpu.upload_f32(&sentinel, &[sentinel.len()]).expect("y_ref");
        let y_new = gpu.upload_f32(&sentinel, &[sentinel.len()]).expect("y_new");
        let y_lm = gpu.upload_f32(&sentinel, &[sentinel.len()]).expect("y_lm");
        let x_view = xt.sub_offset(0, n * k);
        assert!(gpu.gemm_f16_wmma_n16_admitted(&wt, &x_view, m, k, n));
        gpu.gemm_f16_wmma_mb8(&wt, &x_view, &y_ref, m, k, n)
            .expect("mb8");
        gpu.gemm_f16_wmma_n16_gfx12(&wt, &x_view, &y_new, m, k, n)
            .expect("n16");
        gpu.gemm_f16_batched_lmhead(&wt, &x_view, &y_lm, m, k, n)
            .expect("lmhead");
        gpu.hip.device_synchronize().expect("sync");
        let a = gpu.download_f32(&y_ref).expect("download");
        let b = gpu.download_f32(&y_new).expect("download");
        let c = gpu.download_f32(&y_lm).expect("download");
        for i in 0..n * m {
            assert_eq!(
                a[i].to_bits(),
                b[i].to_bits(),
                "m {m} k {k} n {n}: n16 differs from mb8 at {i}: {} vs {}",
                b[i],
                a[i]
            );
            assert_eq!(
                a[i].to_bits(),
                c[i].to_bits(),
                "m {m} k {k} n {n}: lmhead route differs from mb8 at {i}"
            );
        }
        for i in n * m..n * m + m {
            assert_eq!(
                b[i].to_bits(),
                SENTINEL,
                "m {m} k {k} n {n}: wrote past the output at {i}"
            );
        }
        if n == 1 || n == 16 {
            let mut worst = 0f64;
            for r in 0..n {
                for row in 0..m {
                    let dot: f64 = (0..k)
                        .map(|j| {
                            let xh = half::f16::from_f32(x_all[r * k + j]).to_f32();
                            f64::from(w_rt[row * k + j]) * f64::from(xh)
                        })
                        .sum();
                    let got = f64::from(b[r * m + row]);
                    assert!(
                        (got - dot).abs() <= 1e-3 * (1.0 + dot.abs()),
                        "m {m} k {k} n {n} row {r} out {row}: {got} vs reference {dot}"
                    );
                    worst = worst.max((got - dot).abs());
                }
            }
            eprintln!("m {m:5} k {k:5} n {n:2}: bitwise == mb8; max |n16 - f64| {worst:.2e}");
        }
        if time && matches!(n, 1 | 4 | 16) {
            let mb8 = median_ms(gpu, |g| {
                g.gemm_f16_wmma_mb8(&wt, &x_view, &y_ref, m, k, n).unwrap()
            });
            let n16 = median_ms(gpu, |g| {
                g.gemm_f16_wmma_n16_gfx12(&wt, &x_view, &y_new, m, k, n)
                    .unwrap()
            });
            eprintln!("    m {m} k {k} n {n}: ms mb8 {mb8:.4}  n16 {n16:.4}");
        }
        for t in [y_ref, y_new, y_lm] {
            gpu.free_tensor(t).unwrap();
        }
    }
    gpu.free_tensor(xt).unwrap();
    gpu.free_tensor(wt).unwrap();
}

#[test]
#[ignore = "needs a gfx12 GPU"]
fn gemm_f16_wmma_n16_matches_mb8_bitwise_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    if !gpu.arch_caps.has_wmma_w32_gfx12() {
        eprintln!("skip: {} has no gfx12 WMMA", gpu.arch);
        return;
    }
    let mut g = Lcg(0x5eed_f16a);
    // (m, k): router shape (timed), ragged M, K a multiple of 16 below / off the unroll, wide M.
    run_shape(&mut gpu, &mut g, 128, 2304, true);
    run_shape(&mut gpu, &mut g, 130, 2304, false);
    run_shape(&mut gpu, &mut g, 16, 16, false);
    run_shape(&mut gpu, &mut g, 40, 272, false);
    run_shape(&mut gpu, &mut g, 2304, 3072, true);
    // Unadmitted shapes are refused, never mis-launched.
    let wt = gpu.alloc_tensor(&[64 * 24], DType::F32).unwrap();
    let xt = gpu.alloc_tensor(&[17 * 24], DType::F32).unwrap();
    let refused = |m, k, n| !gpu.gemm_f16_wmma_n16_admitted(&wt, &xt, m, k, n);
    assert!(refused(64, 24, 1), "k % 16 != 0");
    assert!(refused(16, 16, 17), "n > 16");
    assert!(refused(16, 16, 0), "n == 0");
    gpu.free_tensor(wt).unwrap();
    gpu.free_tensor(xt).unwrap();
}

/// `gemm_f16_wmma_cols_gfx12` (the same kernel over ceil(n / 16) column tiles) against mb8
/// at prefill row counts: bitwise, no write past `n` rows, and (router shape) timed.
fn run_cols(gpu: &mut Gpu, g: &mut Lcg, m: usize, k: usize, ns: &[usize], time: bool) {
    let w: Vec<f32> = (0..m * k).map(|_| g.unit() * 0.08).collect();
    let (wt, _) = upload_f16(gpu, &w);
    let n_max = *ns.iter().max().unwrap();
    let x_all: Vec<f32> = (0..n_max * k)
        .map(|i| {
            let v = g.unit();
            match i % 97 {
                0 => v * 3.0e-6,
                1 => v * 900.0,
                _ => v * 1.7,
            }
        })
        .collect();
    let xt = gpu.upload_f32(&x_all, &[x_all.len()]).expect("upload x");
    for &n in ns {
        let sentinel = vec![f32::from_bits(SENTINEL); n * m + m];
        let y_ref = gpu.upload_f32(&sentinel, &[sentinel.len()]).expect("y_ref");
        let y_new = gpu.upload_f32(&sentinel, &[sentinel.len()]).expect("y_new");
        let x_view = xt.sub_offset(0, n * k);
        assert!(gpu.gemm_f16_wmma_cols_admitted(&wt, &x_view, m, k, n));
        gpu.gemm_f16_wmma_mb8(&wt, &x_view, &y_ref, m, k, n)
            .expect("mb8");
        gpu.gemm_f16_wmma_cols_gfx12(&wt, &x_view, &y_new, m, k, n)
            .expect("cols");
        gpu.hip.device_synchronize().expect("sync");
        let a = gpu.download_f32(&y_ref).expect("download");
        let b = gpu.download_f32(&y_new).expect("download");
        for i in 0..n * m {
            assert_eq!(
                a[i].to_bits(),
                b[i].to_bits(),
                "m {m} k {k} n {n}: cols differs from mb8 at {i}: {} vs {}",
                b[i],
                a[i]
            );
        }
        for (i, v) in b.iter().enumerate().skip(n * m) {
            assert_eq!(
                v.to_bits(),
                SENTINEL,
                "m {m} k {k} n {n}: wrote past the output at {i}"
            );
        }
        eprintln!("m {m:5} k {k:5} n {n:4}: cols bitwise == mb8");
        if time && n == n_max {
            let mb8 = median_ms(gpu, |g| {
                g.gemm_f16_wmma_mb8(&wt, &x_view, &y_ref, m, k, n).unwrap()
            });
            let cols = median_ms(gpu, |g| {
                g.gemm_f16_wmma_cols_gfx12(&wt, &x_view, &y_new, m, k, n)
                    .unwrap()
            });
            eprintln!("    m {m} k {k} n {n}: ms mb8 {mb8:.4}  cols {cols:.4}");
        }
        for t in [y_ref, y_new] {
            gpu.free_tensor(t).unwrap();
        }
    }
    gpu.free_tensor(xt).unwrap();
    gpu.free_tensor(wt).unwrap();
}

#[test]
#[ignore = "needs a gfx12 GPU"]
fn gemm_f16_wmma_cols_matches_mb8_bitwise_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    if !gpu.arch_caps.has_wmma_w32_gfx12() {
        eprintln!("skip: {} has no gfx12 WMMA", gpu.arch);
        return;
    }
    let mut g = Lcg(0x5eed_c015);
    // Inkling-2 router (blocked prefill: 512-row blocks and ragged tails), ragged M, small K.
    let ns = [1, 16, 17, 31, 100, 129, 511, 512];
    run_cols(&mut gpu, &mut g, 128, 2304, &ns, true);
    run_cols(&mut gpu, &mut g, 130, 2304, &[17, 200], false);
    run_cols(&mut gpu, &mut g, 40, 272, &[33, 300], false);
    let wt = gpu.alloc_tensor(&[64 * 24], DType::F32).unwrap();
    let xt = gpu.alloc_tensor(&[17 * 24], DType::F32).unwrap();
    assert!(
        !gpu.gemm_f16_wmma_cols_admitted(&wt, &xt, 64, 24, 1),
        "k % 16 != 0"
    );
    assert!(
        !gpu.gemm_f16_wmma_cols_admitted(&wt, &xt, 16, 16, 0),
        "n == 0"
    );
    gpu.free_tensor(wt).unwrap();
    gpu.free_tensor(xt).unwrap();
}
