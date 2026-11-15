// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `vit_attention_qtiled_f32` (the non-WMMA vision-tower fallback) against a
//! double-precision CPU oracle and against the naive two-pass kernel.
//!
//! The kernel's only parity coverage before this test was lab examples, which
//! no CI runs; the LDS carve-up and the query-tile sizing are the pair that
//! corrupts *silently* when one side drifts, so they need a numeric check the
//! suite can actually run (`--ignored` on a GPU host).
//!
//! Sweep: head_dim 1..=128 (the guarded lane tail at 70/72, the float4 score
//! path at % 4 == 0, the scalar path otherwise) x query tails around every
//! tile boundary, with the output buffer poisoned to NaN first so a query the
//! kernel never writes shows up as non-finite rather than as a stale value.
//! head_dim 128 additionally runs the qpb=32 tile that wave64 arches take.
//!
//! `#[ignore]`d: needs a GPU and a working HIP toolchain.
//!
//!   cargo test -p rdna-compute --release --test vit_attention_qtiled_gpu -- --ignored

use rdna_compute::{DType, Gpu};

/// Both kernels accumulate in f32 with different association; the f64 oracle
/// differs by rounding only. A layout, tile or tail mistake is >= 1e-3 here.
const REL_L2_TOL: f64 = 1e-5;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() & 0xFFFF) as f32 / 32768.0 - 1.0
    }
}

/// `out[qi, h] = softmax(Q[qi] @ K^T / sqrt(d)) @ V`, f64 accumulation,
/// packed-QKV in and planar-head out — the kernel's contract exactly.
fn oracle(qkv: &[f32], n: usize, hidden: usize, heads: usize, hd: usize) -> Vec<f64> {
    let scale = 1.0 / (hd as f64).sqrt();
    let stride = 3 * hidden;
    let mut out = vec![0.0f64; n * hidden];
    for qi in 0..n {
        for h in 0..heads {
            let q_off = h * hd;
            let k_off = hidden + h * hd;
            let v_off = 2 * hidden + h * hd;
            let mut s = vec![0.0f64; n];
            let mut m = f64::NEG_INFINITY;
            for (j, sj) in s.iter_mut().enumerate() {
                let mut dot = 0.0f64;
                for d in 0..hd {
                    dot += qkv[qi * stride + q_off + d] as f64 * qkv[j * stride + k_off + d] as f64;
                }
                *sj = dot * scale;
                m = m.max(*sj);
            }
            let w: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = w.iter().sum();
            for (j, wj) in w.iter().enumerate() {
                for d in 0..hd {
                    out[qi * hidden + h * hd + d] += wj / z * qkv[j * stride + v_off + d] as f64;
                }
            }
        }
    }
    out
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = *x as f64 - *y as f64;
        num += d * d;
        den += (*y as f64) * (*y as f64);
    }
    (num / den.max(1e-30)).sqrt()
}

/// Run the Q-tiled kernel on `qkv` into a NaN-poisoned buffer.
fn run(gpu: &mut Gpu, qkv_h: &[f32], n: usize, hidden: usize, heads: usize, hd: usize) -> Vec<f32> {
    let qkv = gpu
        .upload_f32(qkv_h, &[n * 3 * hidden])
        .expect("upload qkv");
    let out = gpu
        .upload_f32(&vec![f32::NAN; n * hidden], &[n * hidden])
        .expect("poisoned out");
    gpu.vit_attention_qtiled_f32(&qkv, &out, n, hidden, heads, hd)
        .expect("qtiled launch");
    let got = gpu.download_f32(&out).expect("download");
    gpu.free_tensor(qkv).expect("free");
    gpu.free_tensor(out).expect("free");
    got
}

#[test]
#[ignore = "needs a GPU"]
fn qtiled_matches_a_f64_oracle_across_head_dims_and_tails() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut rng = Rng(0x7174_696c_6564);
    let heads = 2usize;
    let tails = [
        1usize, 2, 7, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 127, 128, 129,
    ];
    for hd in [1usize, 4, 16, 20, 32, 64, 70, 72, 80, 96, 112, 128] {
        let hidden = heads * hd;
        for &n in &tails {
            let qkv_h: Vec<f32> = (0..n * 3 * hidden).map(|_| rng.unit()).collect();
            let got = run(&mut gpu, &qkv_h, n, hidden, heads, hd);
            assert!(
                got.iter().all(|v| v.is_finite()),
                "head_dim {hd} n {n}: the kernel left an output cell unwritten (non-finite)"
            );
            let want = oracle(&qkv_h, n, hidden, heads, hd);
            let want32: Vec<f32> = want.iter().map(|v| *v as f32).collect();
            let e = rel_l2(&got, &want32);
            assert!(
                e <= REL_L2_TOL,
                "head_dim {hd} n {n}: rel_l2 {e:.3e} vs the f64 oracle"
            );
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn qtiled_matches_naive_at_production_shapes() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut rng = Rng(0x7669_742d_6174_746e);
    // The OCR serve shape (2 MP page -> 7564 patches), the Qwen3.5/3.8 tower
    // (hd 72), and hd 128 which is the qpb=32 tile wave64 arches take.
    for &(hidden, heads, hd, n) in &[
        (1024usize, 16usize, 64usize, 7564usize),
        (1152, 16, 72, 4096),
        (2048, 16, 128, 4096),
    ] {
        let qkv_h: Vec<f32> = (0..n * 3 * hidden).map(|_| rng.unit()).collect();
        let got = run(&mut gpu, &qkv_h, n, hidden, heads, hd);
        assert!(
            got.iter().all(|v| v.is_finite()),
            "hidden {hidden} hd {hd} n {n}: unwritten output cell"
        );

        let qkv = gpu.upload_f32(&qkv_h, &[n * 3 * hidden]).expect("upload");
        let out = gpu.zeros(&[n * hidden], DType::F32).expect("out");
        gpu.vit_attention_f32(&qkv, &out, n, hidden, heads, hd)
            .expect("naive launch");
        let naive = gpu.download_f32(&out).expect("download");
        gpu.free_tensor(qkv).expect("free");
        gpu.free_tensor(out).expect("free");

        let e = rel_l2(&got, &naive);
        assert!(
            e <= REL_L2_TOL,
            "hidden {hidden} hd {hd} n {n}: rel_l2 {e:.3e} vs the naive kernel"
        );
    }
}
