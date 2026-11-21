// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Inkling-2 Q/K (+relative R) L2 clip (gfx1100 first).
//!
//! Spec (`docs/plans/local/inkling2-kernel-inventory.md` §5, oracle
//! `hipfire-arch-inkling2/src/oracle.rs::clip`): optional per-channel scale,
//! then per-head-vector row L2 (eps 1e-6 added to the norm), then
//! `out = row · (threshold / norm).clamp(0, 1)`.
//!
//! Assignment: Q clips with the `attn_q_clip` scale, K is scaleless, R clips
//! with the `attn_rel_clip` scale. Model geometry: Q/K rows are `[n_heads,
//! 128]` (24Q / 8KV heads), R rows are `[n_heads, d_rel=16]`. Clips are
//! 22.627 / 8.0. NaN in = NaN out (fail closed).
//!
//! [`inkling2_clip_f32`] is the CPU reference; [`INKLING2_CLIP_HIP_SRC`] is
//! the gfx1100 kernel (one thread-block per row: shared-memory norm
//! reduction, then the in-place scale).

/// Clip norm epsilon (added to the row L2 before dividing).
pub const INKLING2_CLIP_EPS: f32 = 1e-6;

/// gfx1100 kernel: one thread-block per row.
///
/// `blockDim.x` must equal `dim` (128 for Q/K, 16 for R — both ≤ 1024).
/// Phase 1 reduces the scaled row's sum-of-squares in shared memory;
/// phase 2 multiplies the row by `min(threshold / (sqrt + eps), 1)`.
/// `has_scale=0` selects the scaleless (K) path.
pub const INKLING2_CLIP_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
// rows are [n_rows, dim] row-major; scale is [n_rows, dim] or null
extern "C" __global__ void inkling2_l2_clip_rows_f32(
    float* __restrict__ rows, // in-place
    const float* __restrict__ scale, // may be null when has_scale == 0
    int n_rows,
    int dim,
    float threshold,
    int has_scale) {
    const int r = blockIdx.x; // one block per row
    if (r >= n_rows) return;
    const int t = threadIdx.x;
    extern __shared__ float sh[]; // dim floats
    float v = rows[r * dim + t];
    if (has_scale) v *= scale[r * dim + t];
    sh[t] = v * v;
    __syncthreads();
    // tree reduction over dim (dim is a power of two: 128 or 16)
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    const float norm = sqrtf(sh[0]) + 1e-6f;
    float factor = threshold / norm;
    if (factor > 1.0f) factor = 1.0f;
    // NaN factor (NaN row) propagates: NaN in = NaN out, fail closed.
    rows[r * dim + t] = v * factor;
}
"#;

/// L2-clip CPU reference (oracle formula): per-head-vector rows of `dim`,
/// optional per-channel `scale` (`[rows, dim]`), factor
/// `threshold / norm` clamped to `[0, 1]`, eps 1e-6.
pub fn inkling2_clip_f32(
    cur: &[f32],
    scale: Option<&[f32]>,
    threshold: f32,
    dim: usize,
) -> Vec<f32> {
    assert!(!cur.is_empty() && dim > 0 && cur.len().is_multiple_of(dim));
    if let Some(s) = scale {
        assert_eq!(s.len(), cur.len());
    }
    let rows = cur.len() / dim;
    let mut out = cur.to_vec();
    for r in 0..rows {
        let base = r * dim;
        if let Some(s) = scale {
            for i in 0..dim {
                out[base + i] = cur[base + i] * s[base + i];
            }
        }
        let mut sum = 0.0f32;
        for i in 0..dim {
            sum += out[base + i] * out[base + i];
        }
        let norm = sum.sqrt() + INKLING2_CLIP_EPS;
        let factor = (threshold / norm).clamp(0.0, 1.0);
        for i in 0..dim {
            out[base + i] *= factor;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_next(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn rng_f32(state: &mut u64, lo: f32, hi: f32) -> f32 {
        let u = (rng_next(state) >> 11) as f32 / (1u64 << 53) as f32;
        lo + u * (hi - lo)
    }

    fn rand_vec(state: &mut u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| rng_f32(state, lo, hi)).collect()
    }

    /// Naive transcription of the oracle formula (iterator chains, exactly
    /// the oracle's evaluation order).
    fn naive_clip(cur: &[f32], scale: Option<&[f32]>, threshold: f32, dim: usize) -> Vec<f32> {
        let rows = cur.len() / dim;
        let mut out = cur.to_vec();
        for r in 0..rows {
            let base = r * dim;
            for i in 0..dim {
                if let Some(s) = scale {
                    out[base + i] = cur[base + i] * s[base + i];
                }
            }
            let norm = (0..dim)
                .map(|i| out[base + i] * out[base + i])
                .sum::<f32>()
                .sqrt()
                + 1e-6;
            let factor = (threshold / norm).clamp(0.0, 1.0);
            for i in 0..dim {
                out[base + i] *= factor;
            }
        }
        out
    }

    fn row_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn oracle_goldens() {
        // threshold 2.0 on [3,4] (norm 5) scales to [1.2,1.6].
        let out = inkling2_clip_f32(&[3.0, 4.0], None, 2.0, 2);
        assert!((out[0] - 1.2).abs() < 1e-6 && (out[1] - 1.6).abs() < 1e-6);
        // Below-threshold rows pass through.
        let out = inkling2_clip_f32(&[3.0, 4.0], None, 10.0, 2);
        assert_eq!(out, vec![3.0, 4.0]);
        // Optional channel scale applies before the norm.
        let out = inkling2_clip_f32(&[3.0, 4.0], Some(&[2.0, 0.5]), 10.0, 2);
        assert_eq!(out, vec![6.0, 2.0]);
        // Per-head rows: head 0 [3,4] clips, head 1 [0.3,0.4] is untouched.
        let out = inkling2_clip_f32(&[3.0, 4.0, 0.3, 0.4], None, 2.0, 2);
        assert!((out[0] - 1.2).abs() < 1e-6 && (out[1] - 1.6).abs() < 1e-6);
        assert!((out[2] - 0.3).abs() < 1e-9 && (out[3] - 0.4).abs() < 1e-9);
    }

    #[test]
    fn q_heads_with_scale_at_model_dims() {
        // Q: 24 heads × d128 with attn_q_clip-style scale, threshold 22.627.
        let (heads, dim, threshold) = (24usize, 128usize, 22.627f32);
        let mut s = 0x2468_ace0_1357_9bdfu64;
        let cur = rand_vec(&mut s, heads * dim, -3.0, 3.0);
        let scale = rand_vec(&mut s, heads * dim, 0.5, 1.5);
        let got = inkling2_clip_f32(&cur, Some(&scale), threshold, dim);
        let want = naive_clip(&cur, Some(&scale), threshold, dim);
        assert_eq!(got.len(), heads * dim);
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "i={i}: {a} vs {b}");
        }
        // Clipped rows sit exactly at the threshold; small rows are untouched.
        for r in 0..heads {
            let n = row_norm(&got[r * dim..(r + 1) * dim]);
            assert!(n <= threshold + 1e-3, "row {r}: norm {n}");
        }
    }

    #[test]
    fn k_heads_scaleless_at_model_dims() {
        // K: 8 KV heads × d128, no scale, threshold 8.0.
        let (heads, dim, threshold) = (8usize, 128usize, 8.0f32);
        let mut s = 0x0bad_f00d_1234_5678u64;
        let cur = rand_vec(&mut s, heads * dim, -3.0, 3.0);
        let got = inkling2_clip_f32(&cur, None, threshold, dim);
        let want = naive_clip(&cur, None, threshold, dim);
        assert_eq!(got, want);
        for r in 0..heads {
            let n = row_norm(&got[r * dim..(r + 1) * dim]);
            assert!(n <= threshold + 1e-3, "row {r}: norm {n}");
        }
    }

    #[test]
    fn r_heads_with_scale_at_model_dims() {
        // R: 8 heads × d_rel 16 with attn_rel_clip-style scale.
        let (heads, dim, threshold) = (8usize, 16usize, 8.0f32);
        let mut s = 0xfeed_face_0000_1111u64;
        let cur = rand_vec(&mut s, heads * dim, -2.0, 2.0);
        let scale = rand_vec(&mut s, heads * dim, 0.5, 2.0);
        let got = inkling2_clip_f32(&cur, Some(&scale), threshold, dim);
        let want = naive_clip(&cur, Some(&scale), threshold, dim);
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "i={i}: {a} vs {b}");
        }
    }

    #[test]
    fn downscale_only_never_amplifies() {
        // factor ∈ [0,1]: no output row may exceed its scaled input norm.
        let (heads, dim) = (6usize, 32usize);
        let mut s = 0xaaaa_bbbb_cccc_ddddu64;
        let cur = rand_vec(&mut s, heads * dim, -5.0, 5.0);
        let scale = rand_vec(&mut s, heads * dim, 0.1, 3.0);
        let got = inkling2_clip_f32(&cur, Some(&scale), 1.0, dim);
        for r in 0..heads {
            let scaled_norm = row_norm(
                &(0..dim)
                    .map(|i| cur[r * dim + i] * scale[r * dim + i])
                    .collect::<Vec<_>>(),
            );
            let n = row_norm(&got[r * dim..(r + 1) * dim]);
            assert!(n <= scaled_norm + 1e-4, "row {r}: {n} > {scaled_norm}");
        }
    }

    #[test]
    fn nan_in_is_nan_out() {
        let out = inkling2_clip_f32(&[f32::NAN, 1.0], None, 2.0, 2);
        assert!(out[0].is_nan() && out[1].is_nan());
    }
}

// --- GPU dispatch (gfx1100) -------------------------------------------------

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor};
use hip_bridge::HipResult;

impl Gpu {
    /// In-place per-head-row L2 clip: `rows[r] *= min(threshold /
    /// (|scale[r] .* rows[r]| + eps), 1)` over `n_rows` rows of `dim`.
    ///
    /// `scale = None` (the scaleless K path) passes a null scale pointer
    /// with `has_scale = 0`. One block per row with `blockDim.x = dim`
    /// (128 for Q/K, 16 for R) and `dim * 4` bytes of shared memory — the
    /// tree reduction needs a power-of-two `dim <= 1024`, refused otherwise.
    /// Launch shape mirrors the elementwise kernels in `norm.rs`:
    /// `launch_maybe_blob` (graph-safe kernargs).
    pub fn inkling2_l2_clip_rows_f32(
        &mut self,
        rows: &GpuTensor,
        scale: Option<&GpuTensor>,
        n_rows: usize,
        dim: usize,
        threshold: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            dim.is_power_of_two() && dim <= 1024,
            "dim {dim} must be a power of two <= 1024"
        );
        assert_eq!(rows.shape, vec![n_rows, dim]);
        let has_scale = scale.is_some() as i32;
        if let Some(s) = scale {
            assert_eq!(s.shape, vec![n_rows, dim]);
        }
        self.ensure_kernel(
            "inkling2_clip",
            INKLING2_CLIP_HIP_SRC,
            "inkling2_l2_clip_rows_f32",
        )?;
        let rows_ptr = rows.buf.as_ptr();
        let scale_ptr = scale
            .map(|s| s.buf.as_ptr())
            .unwrap_or(std::ptr::null_mut());
        let n_rows_val = n_rows as i32;
        let dim_val = dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &rows_ptr as *const _ as *mut c_void,
            &scale_ptr as *const _ as *mut c_void,
            &n_rows_val as *const _ as *mut c_void,
            &dim_val as *const _ as *mut c_void,
            &threshold as *const _ as *mut c_void,
            &has_scale as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "inkling2_l2_clip_rows_f32",
            [n_rows as u32, 1, 1],
            [dim as u32, 1, 1],
            (dim * 4) as u32,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(rows_ptr);
                b.push_ptr(scale_ptr);
                b.push_i32(n_rows_val);
                b.push_i32(dim_val);
                b.push_f32(threshold);
                b.push_i32(has_scale);
                b
            },
        )
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;

    fn rng_next(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn rng_f32(state: &mut u64, lo: f32, hi: f32) -> f32 {
        let u = (rng_next(state) >> 11) as f32 / (1u64 << 53) as f32;
        lo + u * (hi - lo)
    }

    fn run_case(
        gpu: &mut Gpu,
        cur_h: &[f32],
        scale_h: Option<&[f32]>,
        threshold: f32,
        dim: usize,
    ) -> f32 {
        let n_rows = cur_h.len() / dim;
        let rows = gpu.upload_f32(cur_h, &[n_rows, dim]).expect("rows upload");
        let scale = scale_h.map(|s| gpu.upload_f32(s, &[n_rows, dim]).expect("scale upload"));
        gpu.inkling2_l2_clip_rows_f32(&rows, scale.as_ref(), n_rows, dim, threshold)
            .expect("clip launch");
        let got = gpu.download_f32(&rows).expect("download");
        let want = inkling2_clip_f32(cur_h, scale_h, threshold, dim);
        assert_eq!(got.len(), want.len());
        got.iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    /// Gate: device output matches the crate F32 reference at model dims
    /// (Q 24x128 scaled, K 8x128 scaleless, R 8x16 scaled) and on a
    /// randomized case. Bar: <= 1e-4 abs.
    #[test]
    #[ignore = "requires gfx1100 and the HIP toolchain"]
    fn gpu_matches_cpu_reference() {
        let mut gpu = match crate::gpu_test::init() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skip: no usable GPU ({e:?})");
                return;
            }
        };
        assert_eq!(gpu.arch, "gfx1100");
        let mut s = 0x2468_ace0_1357_9bdfu64;
        // (a) model dims: Q heads with scale, K heads scaleless, R scaled.
        let cur: Vec<f32> = (0..24 * 128).map(|_| rng_f32(&mut s, -3.0, 3.0)).collect();
        let scale: Vec<f32> = (0..24 * 128).map(|_| rng_f32(&mut s, 0.5, 1.5)).collect();
        let d = run_case(&mut gpu, &cur, Some(&scale), 22.627, 128);
        eprintln!("clip Q model-dims max abs diff: {d}");
        assert!(d <= 1e-4, "Q diff {d}");
        let cur: Vec<f32> = (0..8 * 128).map(|_| rng_f32(&mut s, -3.0, 3.0)).collect();
        let d = run_case(&mut gpu, &cur, None, 8.0, 128);
        eprintln!("clip K model-dims (scaleless) max abs diff: {d}");
        assert!(d <= 1e-4, "K diff {d}");
        let cur: Vec<f32> = (0..8 * 16).map(|_| rng_f32(&mut s, -2.0, 2.0)).collect();
        let scale: Vec<f32> = (0..8 * 16).map(|_| rng_f32(&mut s, 0.5, 2.0)).collect();
        let d = run_case(&mut gpu, &cur, Some(&scale), 8.0, 16);
        eprintln!("clip R model-dims max abs diff: {d}");
        assert!(d <= 1e-4, "R diff {d}");
        // (b) randomized case (dim 32, mixed above/below threshold).
        let cur: Vec<f32> = (0..6 * 32).map(|_| rng_f32(&mut s, -5.0, 5.0)).collect();
        let scale: Vec<f32> = (0..6 * 32).map(|_| rng_f32(&mut s, 0.1, 3.0)).collect();
        let d = run_case(&mut gpu, &cur, Some(&scale), 1.0, 32);
        eprintln!("clip randomized max abs diff: {d}");
        assert!(d <= 1e-4, "randomized diff {d}");
    }
}
