// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU regression for GitHub #1136 (Qwen3.5 trunk garbage on gfx942 / MI300X):
//! `Gpu::fused_rmsnorm_rotate_mq` / `Gpu::fused_rmsnorm_rotate_mq_batched`
//! against an f64 CPU reference of the whole op.
//!
//! The gfx94x split route (`kernels/src/rotate_with_rms.gfx942.hip`, default on
//! when `gpu.flags.gfx942_rmsnorm_split`) indexed the RMSNorm weight with the
//! in-group offset instead of the row offset, so every 256-group of a row used
//! group 0's weights: exact for K = 256, garbage for every K > 256 trunk. The
//! sweep covers K = 256 / 1024 / 5120 and batch 1 / 3 with a distinct weight
//! per element, so a repeated-group weight index cannot pass. A negative
//! control proves the comparison itself catches that defect.
//!
//! Op semantics (all routes), per row `b` of `x [batch, K]`, `K % 256 == 0`:
//!   rms   = 1 / sqrt(sum(x^2) / K + eps)
//!   y[i]  = ((x[i] * w[i]) * rms) * signs1[i % 256]
//!   per 256-group: unnormalized Walsh-Hadamard (strides 1..128, (a,b)->(a+b,a-b))
//!   out[i] = y_fwht[i] * (1/16) * signs2[i % 256]
//! with `signs1 = gen_fwht_signs(42, 256)`, `signs2 = gen_fwht_signs(1042, 256)`
//! (the tables `ensure_mq_signs` uploads).
//!
//! Outputs are NaN-poisoned first, so an unwritten element fails. Runs on
//! whatever GPU is present and prints its arch.
//!
//! `#[ignore]`d: needs a GPU, no model files.
//!
//!   cargo test -p rdna-compute --release --test mq_rmsnorm_rotate_ref \
//!       -- --ignored --nocapture

use rdna_compute::{gen_fwht_signs, DType, Gpu, GpuTensor};

const EPS: f32 = 1e-6;
const GROUP: usize = 256;

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform in [0, 1).
    fn f32_unit(&mut self) -> f32 {
        let a = self.next_u32() & 0x7F_FFFF;
        f32::from_bits(0x3F80_0000 | a) - 1.0
    }
    /// Irwin-Hall(12) - 6: approximately N(0, 1).
    fn f32_normalish(&mut self) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..12 {
            s += self.f32_unit();
        }
        s - 6.0
    }
}

fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len()], DType::F32)
        .expect("alloc f32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod f32");
    t
}

/// f64 reference. `buggy` indexes the weight with `i % 256` (the #1136 defect).
fn reference(x: &[f32], w: &[f32], k: usize, batch: usize, buggy: bool) -> Vec<f64> {
    let signs1 = gen_fwht_signs(42, GROUP);
    let signs2 = gen_fwht_signs(1042, GROUP);
    let eps = EPS as f64;
    let mut out = vec![0.0f64; batch * k];
    for b in 0..batch {
        let row = &x[b * k..(b + 1) * k];
        let sumsq: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let rms = 1.0 / (sumsq / k as f64 + eps).sqrt();
        for g in 0..k / GROUP {
            let mut y = [0.0f64; GROUP];
            for j in 0..GROUP {
                let i = g * GROUP + j;
                let wi = if buggy { w[j] } else { w[i] } as f64;
                y[j] = ((row[i] as f64 * wi) * rms) * signs1[j] as f64;
            }
            let mut stride = 1;
            while stride < GROUP {
                for base in (0..GROUP).step_by(stride * 2) {
                    for j in base..base + stride {
                        let (a, c) = (y[j], y[j + stride]);
                        y[j] = a + c;
                        y[j + stride] = a - c;
                    }
                }
                stride *= 2;
            }
            for j in 0..GROUP {
                out[b * k + g * GROUP + j] = y[j] * (1.0 / 16.0) * signs2[j] as f64;
            }
        }
    }
    out
}

fn rel_l2(got: &[f64], want: &[f64]) -> f64 {
    let num: f64 = got.iter().zip(want).map(|(g, r)| (g - r) * (g - r)).sum();
    let den: f64 = want.iter().map(|r| r * r).sum();
    (num / den).sqrt()
}

#[test]
#[ignore]
fn mq_rmsnorm_rotate_matches_cpu_reference() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    println!(
        "arch={} gfx942_rmsnorm_split={}",
        gpu.arch, gpu.flags.gfx942_rmsnorm_split
    );

    for &k in &[256usize, 1024, 5120] {
        for &batch in &[1usize, 3] {
            let mut rng = Lcg(0x1136_0000 + (k as u64) * 16 + batch as u64);
            let mut x: Vec<f32> = (0..batch * k).map(|_| rng.f32_normalish()).collect();
            // One outlier in the last row, mid-row (shrinks rms, stresses scaling).
            x[(batch - 1) * k + k / 2 + 3] = 50.0;
            // Distinct weight per element, in [0.5, 1.5).
            let w: Vec<f32> = (0..k).map(|_| 0.5 + rng.f32_unit()).collect();

            let want = reference(&x, &w, k, batch, false);

            let x_t = upload_f32(&mut gpu, &x);
            let w_t = upload_f32(&mut gpu, &w);
            let x_rot = upload_f32(&mut gpu, &vec![f32::NAN; batch * k]);
            if batch == 1 {
                gpu.fused_rmsnorm_rotate_mq(&x_t, &w_t, &x_rot, k, EPS)
                    .expect("fused_rmsnorm_rotate_mq");
            } else {
                gpu.fused_rmsnorm_rotate_mq_batched(&x_t, &w_t, &x_rot, k, EPS, batch)
                    .expect("fused_rmsnorm_rotate_mq_batched");
            }
            let got: Vec<f64> = gpu
                .download_f32(&x_rot)
                .expect("download x_rot")
                .into_iter()
                .map(|v| v as f64)
                .collect();
            assert_eq!(got.len(), batch * k, "K={k} batch={batch}: output length");

            let ref_max = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let tol = 1e-5 * ref_max.max(1.0);
            let mut max_err = 0.0f64;
            let mut first_bad: Option<usize> = None;
            for (i, (g, r)) in got.iter().zip(&want).enumerate() {
                let err = (g - r).abs();
                let bad = !g.is_finite() || err > tol;
                if bad && first_bad.is_none() {
                    first_bad = Some(i);
                }
                if g.is_finite() {
                    max_err = max_err.max(err);
                }
            }
            let l2 = rel_l2(&got, &want);
            println!(
                "K={k:5} batch={batch}: max_abs_err={max_err:.3e} (tol {tol:.3e}) rel_l2={l2:.3e}"
            );
            if let Some(i) = first_bad {
                panic!(
                    "K={k} batch={batch}: first bad index {i} (row {}, group {}, lane {}): \
                     gpu={} ref={} tol={tol:.3e}",
                    i / k,
                    (i % k) / GROUP,
                    i % GROUP,
                    got[i],
                    want[i]
                );
            }

            // Negative control: the #1136 defect (group-0 weights in every
            // group) must be far from the correct reference.
            if k == 5120 {
                let buggy = reference(&x, &w, k, batch, true);
                let bug_l2 = rel_l2(&buggy, &want);
                let bug_max = buggy
                    .iter()
                    .zip(&want)
                    .fold(0.0f64, |m, (b, r)| m.max((b - r).abs()));
                println!(
                    "K={k:5} batch={batch}: negative control (i%256 weight) \
                     rel_l2={bug_l2:.3e} max_abs_diff={bug_max:.3e}"
                );
                assert!(
                    bug_l2 > 1e-2 && bug_max > 100.0 * tol,
                    "K={k} batch={batch}: buggy-weight reference is indistinguishable from the \
                     correct one (rel_l2={bug_l2:.3e}, max_abs_diff={bug_max:.3e}, tol={tol:.3e}); \
                     the comparison would not catch #1136"
                );
                // And the GPU result must be on the correct side of that gap.
                let gpu_vs_bug = rel_l2(&got, &buggy);
                assert!(
                    gpu_vs_bug > 1e-2,
                    "K={k} batch={batch}: GPU output matches the buggy-weight reference \
                     (rel_l2 vs buggy {gpu_vs_bug:.3e})"
                );
            }

            for t in [x_t, w_t, x_rot] {
                gpu.free_tensor(t).expect("free_tensor");
            }
        }
    }
}
