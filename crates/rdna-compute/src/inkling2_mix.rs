// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Inkling-2 `mix()` over block outputs plus current (gfx1100 first).
//!
//! Spec (`docs/plans/local/inkling2-kernel-inventory.md` §3, oracle
//! `hipfire-arch-inkling2/src/oracle.rs::mix`): `concat([blocks…, cur])`
//! → RMSNorm 1e-6 per state → scores against each learned query row at
//! `1/sqrt(n_embd)` → softmax over states (F32) → weighted sum. The bank is
//! the previous pass's `blocks[prelude_len+1..]` plus its output; the prelude
//! is restored each pass (skip-first-core-block detail). Model: `n_embd`
//! 2304, one row out per query row.
//!
//! [`inkling2_mix_f32`] is the CPU reference the GPU path must match.
//! The HIP envelope ([`INKLING2_MIX_HIP_SRC`]) splits the op the way a
//! single-token decode kernel does: pass 1 RMSNorms the bank in-shader via
//! the shared gamma-less norm (one block per state row), pass 2 runs one
//! query row per block with a cooperative dot/reduce, a `tid == 0`
//! softmax over the <= 6 weights, and a cooperative weighted sum.

/// mix() RMSNorm epsilon.
pub const INKLING2_MIX_EPS: f32 = 1e-6;

/// gfx1100 decode envelope for `mix()`.
///
/// Pass 1 normalizes the `[n_states, n_embd]` bank with the shared
/// gamma-less RMSNorm from `inkling2_control.rs` (one block per state row)
/// — there is no `row_inv_scale` input and no `scratch[n_states]` global.
/// Pass 2 (`inkling2_mix_query_row_f32`): one block per query row; all 256
/// threads cooperatively dot/reduce (`i = tid, tid+256, ... < n_embd`) per
/// state with a shared-memory reduction, `tid == 0` soft-maxes the <= 6
/// weights and broadcasts them, then all threads cooperatively accumulate
/// the weighted sum.
pub const INKLING2_MIX_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
// out[q * n_embd + i] = sum_s softmax_s(dot(normed_s, query_q)/sqrt(n_embd)) * normed_s[i]
// normed is [n_states, n_embd] (already RMSNormed), query is [query_rows, n_embd].
// One block per query row, blockDim.x == 256 fixed. n_states <= 6.
extern "C" __global__ void inkling2_mix_query_row_f32(
    const float* __restrict__ normed, // [n_states, n_embd], F32
    const float* __restrict__ query,  // [query_rows, n_embd]
    float* __restrict__ out,          // [query_rows, n_embd]
    int n_states,
    int n_embd,
    float inv_sqrt_embd) {
    const int q = blockIdx.x; // one block per query row
    const int t = threadIdx.x;
    __shared__ float sh[256];
    __shared__ float w[6]; // n_states <= 6: one softmax weight per state
    // 1. scores[s] = dot(normed_s, query_q) * inv_sqrt_embd, cooperatively.
    for (int s = 0; s < n_states; ++s) {
        float acc = 0.0f;
        const float* ns = normed + (size_t)s * n_embd;
        const float* qq = query + (size_t)q * n_embd;
        for (int i = t; i < n_embd; i += blockDim.x) {
            acc += ns[i] * qq[i];
        }
        sh[t] = acc;
        __syncthreads();
        for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
            if (t < st) sh[t] += sh[t + st];
            __syncthreads();
        }
        if (t == 0) w[s] = sh[0] * inv_sqrt_embd;
        __syncthreads();
    }
    // 2. stable softmax over the <= 6 weights on thread 0, then broadcast.
    if (t == 0) {
        float mx = w[0];
        for (int s = 1; s < n_states; ++s) {
            if (w[s] > mx) mx = w[s];
        }
        float sum = 0.0f;
        for (int s = 0; s < n_states; ++s) {
            w[s] = expf(w[s] - mx);
            sum += w[s];
        }
        for (int s = 0; s < n_states; ++s) w[s] /= sum;
    }
    __syncthreads();
    // 3. out[q, i] = sum_s w[s] * normed[s, i], cooperatively.
    float* oq = out + (size_t)q * n_embd;
    for (int i = t; i < n_embd; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_states; ++s) {
            acc += w[s] * normed[(size_t)s * n_embd + i];
        }
        oq[i] = acc;
    }
}
"#;

fn rmsnorm_row(x: &[f32]) -> Vec<f32> {
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    let scale = 1.0 / ((sum / x.len() as f32) + INKLING2_MIX_EPS).sqrt();
    x.iter().map(|v| v * scale).collect()
}

/// `mix()` CPU reference: `blocks` (oldest first) plus `cur` form the state
/// bank; `query` is `[query_rows, n_embd]`; returns `[query_rows, n_embd]`.
///
/// Structured state-outer: each state's weight is applied to every query row
/// before moving on (the order a banked kernel accumulates in).
pub fn inkling2_mix_f32(
    blocks: &[Vec<f32>],
    cur: &[f32],
    query: &[f32],
    n_embd: usize,
    query_rows: usize,
) -> Vec<f32> {
    assert_eq!(query.len(), query_rows * n_embd);
    let mut states: Vec<Vec<f32>> = blocks
        .iter()
        .map(|b| {
            assert_eq!(b.len(), n_embd);
            rmsnorm_row(b)
        })
        .collect();
    assert_eq!(cur.len(), n_embd);
    states.push(rmsnorm_row(cur));
    let ns = states.len();
    let inv = 1.0 / (n_embd as f32).sqrt();
    // weights[q][s]: stable softmax per query row over state scores.
    let mut weights = vec![0.0f32; query_rows * ns];
    for q in 0..query_rows {
        let mut max = f32::NEG_INFINITY;
        for s in 0..ns {
            let dot: f32 = (0..n_embd)
                .map(|i| states[s][i] * query[q * n_embd + i])
                .sum();
            let sc = dot * inv;
            weights[q * ns + s] = sc;
            if sc > max {
                max = sc;
            }
        }
        let mut sum = 0.0f32;
        for s in 0..ns {
            let e = (weights[q * ns + s] - max).exp();
            weights[q * ns + s] = e;
            sum += e;
        }
        for s in 0..ns {
            weights[q * ns + s] /= sum;
        }
    }
    // State-outer accumulation into all query rows.
    let mut out = vec![0.0f32; query_rows * n_embd];
    for (s, st) in states.iter().enumerate() {
        for q in 0..query_rows {
            let p = weights[q * ns + s];
            for i in 0..n_embd {
                out[q * n_embd + i] += p * st[i];
            }
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

    /// Naive transcription of the oracle formula (query-outer accumulation).
    fn naive_mix(
        blocks: &[Vec<f32>],
        cur: &[f32],
        query: &[f32],
        n_embd: usize,
        query_rows: usize,
    ) -> Vec<f32> {
        let norm = |x: &[f32]| -> Vec<f32> {
            let m = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            x.iter().map(|v| v / (m + 1e-6).sqrt()).collect()
        };
        let mut states: Vec<Vec<f32>> = blocks.iter().map(|b| norm(b)).collect();
        states.push(norm(cur));
        let ns = states.len();
        let scale = 1.0 / (n_embd as f32).sqrt();
        let mut out = vec![0.0f32; query_rows * n_embd];
        for b in 0..query_rows {
            let mut scores = Vec::with_capacity(ns);
            for s in &states {
                let dot: f32 = (0..n_embd).map(|i| s[i] * query[b * n_embd + i]).sum();
                scores.push(dot * scale);
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (idx, e) in exps.iter().enumerate() {
                let p = e / sum;
                for i in 0..n_embd {
                    out[b * n_embd + i] += states[idx][i] * p;
                }
            }
        }
        out
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn single_state_is_its_normalized_value() {
        // Oracle golden: no blocks => the single state passes through normalized.
        let out = inkling2_mix_f32(&[], &[3.0, 4.0], &[1.0, 0.0], 2, 1);
        let m = (9.0f32 + 16.0) / 2.0;
        let s = 1.0 / (m + 1e-6).sqrt();
        assert!((out[0] - 3.0 * s).abs() < 1e-6);
        assert!((out[1] - 4.0 * s).abs() < 1e-6);
    }

    #[test]
    fn equal_states_split_evenly() {
        // Oracle golden: identical states average to themselves.
        let out = inkling2_mix_f32(&[vec![1.0, 1.0]], &[1.0, 1.0], &[1.0, 1.0], 2, 1);
        assert!(
            (out[0] - 1.0).abs() < 1e-6 && (out[1] - 1.0).abs() < 1e-6,
            "{out:?}"
        );
    }

    #[test]
    fn model_dims_match_oracle_formula() {
        // Model geometry, synthetic data: n_embd 2304, 5 bank blocks + cur.
        let (n_embd, n_blocks, query_rows) = (2304usize, 5usize, 2usize);
        let mut s = 0x51ab_3c9d_2ef0_1123u64;
        let blocks: Vec<Vec<f32>> = (0..n_blocks)
            .map(|_| rand_vec(&mut s, n_embd, -2.0, 2.0))
            .collect();
        let cur = rand_vec(&mut s, n_embd, -2.0, 2.0);
        let query = rand_vec(&mut s, query_rows * n_embd, -1.0, 1.0);
        let got = inkling2_mix_f32(&blocks, &cur, &query, n_embd, query_rows);
        let want = naive_mix(&blocks, &cur, &query, n_embd, query_rows);
        assert_eq!(got.len(), query_rows * n_embd);
        let d = max_abs_diff(&got, &want);
        assert!(d < 1e-4, "max abs diff {d}");
    }

    #[test]
    fn output_is_convex_combination_of_states() {
        // Softmax weights are non-negative and sum to 1, so every output
        // element lies within the per-element min/max of the normalized bank.
        let (n_embd, n_blocks) = (64usize, 3usize);
        let mut s = 0x77aa_bbcc_ddee_ff01u64;
        let blocks: Vec<Vec<f32>> = (0..n_blocks)
            .map(|_| rand_vec(&mut s, n_embd, -3.0, 3.0))
            .collect();
        let cur = rand_vec(&mut s, n_embd, -3.0, 3.0);
        let query = rand_vec(&mut s, n_embd, -1.0, 1.0);
        let out = inkling2_mix_f32(&blocks, &cur, &query, n_embd, 1);
        let norm = |x: &[f32]| -> Vec<f32> {
            let m = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            x.iter().map(|v| v / (m + 1e-6).sqrt()).collect()
        };
        let mut states: Vec<Vec<f32>> = blocks.iter().map(|b| norm(b)).collect();
        states.push(norm(&cur));
        for i in 0..n_embd {
            let lo = states.iter().map(|st| st[i]).fold(f32::INFINITY, f32::min);
            let hi = states
                .iter()
                .map(|st| st[i])
                .fold(f32::NEG_INFINITY, f32::max);
            assert!(out[i] >= lo - 1e-5 && out[i] <= hi + 1e-5, "i={i}");
        }
    }

    #[test]
    fn per_query_rows_are_independent() {
        // Each query row only reads its own query vector: appending a row
        // must not move the earlier rows.
        let (n_embd, query_rows) = (128usize, 2usize);
        let mut s = 0x0102_0304_0506_0708u64;
        let blocks = vec![rand_vec(&mut s, n_embd, -2.0, 2.0)];
        let cur = rand_vec(&mut s, n_embd, -2.0, 2.0);
        let query = rand_vec(&mut s, query_rows * n_embd, -1.0, 1.0);
        let both = inkling2_mix_f32(&blocks, &cur, &query, n_embd, query_rows);
        let first = inkling2_mix_f32(&blocks, &cur, &query[..n_embd], n_embd, 1);
        assert_eq!(&both[..n_embd], &first[..]);
    }
}

// --- GPU dispatch (gfx1100) -------------------------------------------------

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor};
use hip_bridge::HipResult;

impl Gpu {
    /// One `mix()` query block: cooperative dot/reduce + softmax + weighted
    /// sum over the already-normalized `normed` bank. `normed` is
    /// `[n_states, n_embd]` F32 (normalize the bank first with
    /// [`Self::inkling2_rmsnorm_nogamma_f32`]), `query` is
    /// `[query_rows, n_embd]`, `out` is `[query_rows, n_embd]`.
    /// `n_states <= 6` (the model bank plus `cur`); the softmax weights live
    /// in shared memory, so no `scratch` global crosses the launch.
    pub fn inkling2_mix_query_row_f32(
        &mut self,
        normed: &GpuTensor,
        query: &GpuTensor,
        out: &GpuTensor,
        n_states: usize,
        n_embd: usize,
        query_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            (1..=6).contains(&n_states),
            "n_states {n_states} exceeds the 6-weight shared design"
        );
        assert_eq!(normed.shape, vec![n_states, n_embd]);
        assert_eq!(query.shape, vec![query_rows, n_embd]);
        assert_eq!(out.shape, vec![query_rows, n_embd]);
        self.ensure_kernel(
            "inkling2_mix_query",
            INKLING2_MIX_HIP_SRC,
            "inkling2_mix_query_row_f32",
        )?;
        let normed_ptr = normed.buf.as_ptr();
        let query_ptr = query.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let n_states_val = n_states as i32;
        let n_embd_val = n_embd as i32;
        let inv_sqrt_embd = 1.0 / (n_embd as f32).sqrt();
        let mut params: Vec<*mut c_void> = vec![
            &normed_ptr as *const _ as *mut c_void,
            &query_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &n_states_val as *const _ as *mut c_void,
            &n_embd_val as *const _ as *mut c_void,
            &inv_sqrt_embd as *const _ as *mut c_void,
        ];
        // 256-wide shared reduction + 6 softmax weights: (256 + 6) * 4 B.
        self.launch_maybe_blob(
            "inkling2_mix_query_row_f32",
            [query_rows as u32, 1, 1],
            [256, 1, 1],
            (256 + 6) * 4,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(normed_ptr);
                b.push_ptr(query_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(n_states_val);
                b.push_i32(n_embd_val);
                b.push_f32(inv_sqrt_embd);
                b
            },
        )
    }

    /// Full `mix()`: RMSNorm the `[n_states, n_embd]` `bank` in-shader into
    /// caller-owned `normed`, then run all `query_rows` against it. The
    /// per-call path joins nothing and allocates nothing: both buffers are
    /// caller-owned (brief 03 stages them in its state).
    pub fn inkling2_mix_f32_gpu(
        &mut self,
        bank: &GpuTensor,
        query: &GpuTensor,
        normed: &GpuTensor,
        out: &GpuTensor,
        n_states: usize,
        n_embd: usize,
        query_rows: usize,
        eps: f32,
    ) -> HipResult<()> {
        assert_eq!(bank.shape, vec![n_states, n_embd]);
        self.inkling2_rmsnorm_nogamma_f32(bank, normed, n_states, n_embd, eps)?;
        self.inkling2_mix_query_row_f32(normed, query, out, n_states, n_embd, query_rows)
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use crate::dispatch::DType;
    use crate::inkling2_control::INKLING2_CONTROL_EPS;

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
        blocks_h: &[Vec<f32>],
        cur_h: &[f32],
        query_h: &[f32],
        n_embd: usize,
        query_rows: usize,
    ) -> f32 {
        let n_states = blocks_h.len() + 1;
        let mut bank_h: Vec<f32> = Vec::with_capacity(n_states * n_embd);
        for b in blocks_h {
            bank_h.extend_from_slice(b);
        }
        bank_h.extend_from_slice(cur_h);
        let bank = gpu
            .upload_f32(&bank_h, &[n_states, n_embd])
            .expect("bank upload");
        let query = gpu
            .upload_f32(query_h, &[query_rows, n_embd])
            .expect("query upload");
        let normed = gpu
            .alloc_tensor(&[n_states, n_embd], DType::F32)
            .expect("normed alloc");
        let out = gpu
            .alloc_tensor(&[query_rows, n_embd], DType::F32)
            .expect("out alloc");
        gpu.inkling2_mix_f32_gpu(
            &bank,
            &query,
            &normed,
            &out,
            n_states,
            n_embd,
            query_rows,
            INKLING2_MIX_EPS,
        )
        .expect("mix launch");
        let got = gpu.download_f32(&out).expect("download");
        let want = inkling2_mix_f32(blocks_h, cur_h, query_h, n_embd, query_rows);
        assert_eq!(got.len(), want.len());
        got.iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    /// Gate: device output matches the crate F32 reference at model dims
    /// (n_embd 2304, 5 bank blocks + cur, 2 query rows) and on a randomized
    /// case. Bar: <= 1e-5 abs.
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
        assert_eq!(INKLING2_MIX_EPS, INKLING2_CONTROL_EPS);
        // (a) model dims.
        let (n_embd, n_blocks, query_rows) = (2304usize, 5usize, 2usize);
        let mut s = 0x51ab_3c9d_2ef0_1123u64;
        let blocks: Vec<Vec<f32>> = (0..n_blocks)
            .map(|_| (0..n_embd).map(|_| rng_f32(&mut s, -2.0, 2.0)).collect())
            .collect();
        let cur: Vec<f32> = (0..n_embd).map(|_| rng_f32(&mut s, -2.0, 2.0)).collect();
        let query: Vec<f32> = (0..query_rows * n_embd)
            .map(|_| rng_f32(&mut s, -1.0, 1.0))
            .collect();
        let d = run_case(&mut gpu, &blocks, &cur, &query, n_embd, query_rows);
        eprintln!("mix model-dims max abs diff: {d}");
        assert!(d <= 1e-5, "model dims diff {d}");
        // (b) randomized small case (single state + two-state bank).
        let (n_embd, query_rows) = (64usize, 3usize);
        let mut s = 0x77aa_bbcc_ddee_ff03u64;
        for n_blocks in [0usize, 2usize] {
            let blocks: Vec<Vec<f32>> = (0..n_blocks)
                .map(|_| (0..n_embd).map(|_| rng_f32(&mut s, -3.0, 3.0)).collect())
                .collect();
            let cur: Vec<f32> = (0..n_embd).map(|_| rng_f32(&mut s, -3.0, 3.0)).collect();
            let query: Vec<f32> = (0..query_rows * n_embd)
                .map(|_| rng_f32(&mut s, -1.0, 1.0))
                .collect();
            let d = run_case(&mut gpu, &blocks, &cur, &query, n_embd, query_rows);
            eprintln!("mix randomized ({} blocks) max abs diff: {d}", n_blocks);
            assert!(d <= 1e-5, "randomized diff {d}");
        }
    }
}
