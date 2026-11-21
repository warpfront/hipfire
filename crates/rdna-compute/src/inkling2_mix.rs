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
//! single-token decode kernel does: pass 1 normalizes the bank into scratch,
//! pass 2 runs one query row per block (F32 stable softmax + weighted sum).

/// mix() RMSNorm epsilon.
pub const INKLING2_MIX_EPS: f32 = 1e-6;

/// gfx1100 decode envelope for `mix()`.
///
/// Pass 1 (`inkling2_mix_norm_states_f32`): one thread per `(state, i)`
/// element, grid-stride over the flat `[n_states, n_embd]` bank.
/// Pass 2 (`inkling2_mix_query_row_f32`): one block per query row;
/// `blockDim.x` (256) threads cooperatively reduce the `n_states` scores
/// (shared-memory max + sum, F32 stable softmax), then accumulate the
/// weighted sum grid-stride over `n_embd`.
pub const INKLING2_MIX_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
// normed[s * n_embd + i] = bank[s * n_embd + i] / sqrt(mean_sq + 1e-6)
extern "C" __global__ void inkling2_mix_norm_states_f32(
    const float* __restrict__ bank,
    float* __restrict__ normed,
    const float* __restrict__ row_inv_scale, // per-state 1/sqrt(mean_sq+eps)
    int n_states,
    int n_embd) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total = n_states * n_embd;
    for (int t = tid; t < total; t += stride) {
        normed[t] = bank[t] * row_inv_scale[t / n_embd];
    }
}
// out[q * n_embd + i] = sum_s softmax_s(dot(normed_s, query_q)/sqrt(n_embd)) * normed_s[i]
extern "C" __global__ void inkling2_mix_query_row_f32(
    const float* __restrict__ normed, // [n_states, n_embd], F32
    const float* __restrict__ query,  // [query_rows, n_embd]
    float* __restrict__ out,          // [query_rows, n_embd]
    float* __restrict__ scratch,      // [n_states] scores for this row
    int n_states,
    int n_embd,
    float inv_sqrt_embd) {
    const int q = blockIdx.x; // one block per query row
    const int tid = threadIdx.x;
    // scores[s] = dot(normed_s, query_q) * inv_sqrt_embd (grid-stride over s)
    for (int s = tid; s < n_states; s += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < n_embd; ++i) {
            acc += normed[s * n_embd + i] * query[q * n_embd + i];
        }
        scratch[s] = acc * inv_sqrt_embd;
    }
    __syncthreads();
    // stable softmax over scratch[0..n_states] in shared pattern (block 256)
    __shared__ float sh[256];
    float mx = -3.4028235e38f;
    for (int s = tid; s < n_states; s += blockDim.x) {
        if (scratch[s] > mx) mx = scratch[s];
    }
    sh[tid] = mx;
    __syncthreads();
    if (tid == 0) {
        float m = sh[0];
        for (int t = 1; t < blockDim.x; ++t) {
            if (sh[t] > m) m = sh[t];
        }
        sh[0] = m;
    }
    __syncthreads();
    mx = sh[0];
    float sum = 0.0f;
    for (int s = tid; s < n_states; s += blockDim.x) {
        scratch[s] = expf(scratch[s] - mx);
        sum += scratch[s];
    }
    sh[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int k = 0; k < blockDim.x; ++k) t += sh[k];
        sh[0] = t;
    }
    __syncthreads();
    sum = sh[0];
    // out[q, i] = sum_s scratch[s]/sum * normed[s, i] (grid-stride over i)
    for (int i = tid; i < n_embd; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_states; ++s) {
            acc += (scratch[s] / sum) * normed[s * n_embd + i];
        }
        out[q * n_embd + i] = acc;
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
    let mut states: Vec<Vec<f32>> = blocks.iter().map(|b| {
        assert_eq!(b.len(), n_embd);
        rmsnorm_row(b)
    }).collect();
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
        assert!((out[0] - 1.0).abs() < 1e-6 && (out[1] - 1.0).abs() < 1e-6, "{out:?}");
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
            let hi = states.iter().map(|st| st[i]).fold(f32::NEG_INFINITY, f32::max);
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
