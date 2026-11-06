// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Gpu::gemm_f16_skinny_xf32` (few-row F16 x F32, weights read once; the DFlash2
//! draft's staged `fc` projection) against an f64 CPU reference, for 1..=16 rows
//! at the Raon draft's `fc` shape (M 2560, K 51200) and a ragged M / small K.
//!
//! * accuracy: every output within `1e-3 * (1 + |ref|)` of the f64 reference
//!   (F32 accumulation of ~51k products; the inputs are random, zero-mean);
//! * determinism: a second launch is bit-identical to the first;
//! * the unwritten guard row past the output stays untouched;
//! * information: max |skinny - WMMA GEMM| (`gemm_f16_batched_lmhead`, which
//!   rounds X to F16) and the ms of both at rows 1, 2, 4, 8, 16.
//!
//! `#[ignore]`d: needs a HIP GPU. Run explicitly (through the GPU fabric):
//!
//!   cargo test -p rdna-compute --release --test gemm_f16_skinny_gpu -- --ignored --nocapture

use rdna_compute::{Gpu, GpuTensor};
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

fn f16_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

fn upload_f16(gpu: &Gpu, w: &[f32]) -> (GpuTensor, Vec<f32>) {
    let bits: Vec<u16> = w.iter().map(|&v| f16_bits(v)).collect();
    let back: Vec<f32> = bits
        .iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect();
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    (
        gpu.upload_raw(&bytes, &[bytes.len()]).expect("upload w"),
        back,
    )
}

const SENTINEL: u32 = 0x7fc0_dead;

fn run_shape(gpu: &mut Gpu, g: &mut Lcg, m: usize, k: usize, time: bool) {
    let w: Vec<f32> = (0..m * k).map(|_| g.unit() * 0.05).collect();
    let (wt, w_rt) = upload_f16(gpu, &w);
    let x_all: Vec<f32> = (0..16 * k).map(|_| g.unit()).collect();
    let xt = gpu.upload_f32(&x_all, &[x_all.len()]).expect("upload x");
    for n in [1usize, 2, 3, 4, 7, 8, 15, 16] {
        let sentinel = vec![f32::from_bits(SENTINEL); n * m + m];
        let y0 = gpu.upload_f32(&sentinel, &[sentinel.len()]).expect("y0");
        let y1 = gpu.upload_f32(&sentinel, &[sentinel.len()]).expect("y1");
        let x_view = xt.sub_offset(0, n * k);
        assert!(gpu.gemm_f16_skinny_admitted(&wt, &x_view, m, k, n));
        gpu.gemm_f16_skinny_xf32(&wt, &x_view, &y0, m, k, n)
            .expect("skinny");
        gpu.gemm_f16_skinny_xf32(&wt, &x_view, &y1, m, k, n)
            .expect("skinny again");
        gpu.hip.device_synchronize().expect("sync");
        let a = gpu.download_f32(&y0).expect("download");
        let b = gpu.download_f32(&y1).expect("download");
        for i in 0..n * m {
            assert_eq!(
                a[i].to_bits(),
                b[i].to_bits(),
                "m {m} k {k} n {n}: non-deterministic at {i}"
            );
        }
        for i in n * m..n * m + m {
            assert_eq!(
                a[i].to_bits(),
                SENTINEL,
                "m {m} k {k} n {n}: wrote past the output at {i}"
            );
        }
        // f64 reference on the F16-rounded weights (X is exact F32).
        let mut worst = 0f64;
        for r in 0..n {
            for row in 0..m {
                let dot: f64 = (0..k)
                    .map(|j| f64::from(w_rt[row * k + j]) * f64::from(x_all[r * k + j]))
                    .sum();
                let got = f64::from(a[r * m + row]);
                let tol = 1e-3 * (1.0 + dot.abs());
                assert!(
                    (got - dot).abs() <= tol,
                    "m {m} k {k} n {n} row {r} out {row}: {got} vs reference {dot}"
                );
                worst = worst.max((got - dot).abs());
            }
        }
        // Information: the WMMA GEMM this replaces.
        let yw = gpu.upload_f32(&sentinel, &[sentinel.len()]).expect("yw");
        gpu.gemm_f16_batched_lmhead(&wt, &x_view, &yw, m, k, n)
            .expect("wmma");
        gpu.hip.device_synchronize().expect("sync");
        let c = gpu.download_f32(&yw).expect("download");
        let diff = (0..n * m)
            .map(|i| f64::from((a[i] - c[i]).abs()))
            .fold(0.0, f64::max);
        eprintln!(
            "m {m:5} k {k:6} n {n:2}: max |skinny - f64| {worst:.2e}, max |skinny - WMMA| {diff:.2e}"
        );
        if time && matches!(n, 1 | 2 | 4 | 8 | 16) {
            let mut ms = [0f64; 2];
            for (slot, skinny) in [(0, true), (1, false)] {
                let mut samples = Vec::new();
                for round in 0..12 {
                    let t = Instant::now();
                    for _ in 0..10 {
                        if skinny {
                            gpu.gemm_f16_skinny_xf32(&wt, &x_view, &y0, m, k, n)
                                .unwrap();
                        } else {
                            gpu.gemm_f16_batched_lmhead(&wt, &x_view, &yw, m, k, n)
                                .unwrap();
                        }
                    }
                    gpu.hip.device_synchronize().unwrap();
                    if round >= 2 {
                        samples.push(t.elapsed().as_secs_f64() * 1e3 / 10.0);
                    }
                }
                samples.sort_by(|p, q| p.partial_cmp(q).unwrap());
                ms[slot] = samples[samples.len() / 2];
            }
            eprintln!("    ms: skinny {:.3}  WMMA {:.3}", ms[0], ms[1]);
        }
        for t in [y0, y1, yw] {
            let _ = gpu.free_tensor(t);
        }
    }
    let _ = gpu.free_tensor(wt);
    let _ = gpu.free_tensor(xt);
}

#[test]
#[ignore = "needs a GPU: run through the GPU fabric"]
fn skinny_f16_gemm_matches_the_f64_reference_and_is_deterministic() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut g = Lcg(0xfc);
    run_shape(&mut gpu, &mut g, 2557, 1536, false);
    run_shape(&mut gpu, &mut g, 2560, 51_200, true);
}
