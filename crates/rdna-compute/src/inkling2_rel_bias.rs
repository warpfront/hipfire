// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Inkling-2 relative-position bias gather (gfx1100 first).
//!
//! Spec (`docs/plans/local/inkling2-kernel-inventory.md` §4, oracle
//! `hipfire-arch-inkling2/src/oracle.rs::relative_bias`): R is projected to
//! `n_rel` dims per head (`d_rel` 16); scores add the `[n_rel, n_extent=1024]`
//! table (`attn_rel_bias`) gathered by distance; **the per-head sink logit is
//! subtracted before the result enters attention**; then distance `get_rows`.
//! Validate the `loop_msg_query [n_embd, 4]` packing
//! `[q2i, kv2i, kv2i+1, q2i+1]` at ingest, fail closed.
//!
//! Layouts (row-major, F32): `rel: [n_rel, n_heads]`, `table: [n_rel,
//! n_extent]`, `sink: [n_heads]`, `distances: [n_kv]` with every entry <
//! `n_extent`. Returns `[n_heads, n_kv]`.
//!
//! Order note: the sink term is distance-independent, so subtracting it after
//! the distance gather is exactly the oracle's sink-subtract-before-gather
//! (the subtraction must happen before the scores enter the sink-softmax,
//! not after it — that is the load-bearing part).
//!
//! [`inkling2_rel_bias_f32`] is the CPU reference; [`INKLING2_REL_BIAS_HIP_SRC`]
//! is the gfx1100 kernel (one thread per `(head, kv)` pair).

/// gfx1100 kernel: one thread per `(head, kv)` output element.
///
/// Each thread walks the `n_rel` (16) table entries for its distance,
/// accumulates `table[r, d] * rel[r, h]`, then subtracts `sink[h]`.
pub const INKLING2_REL_BIAS_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
// out[h * n_kv + k] = sum_r table[r * n_extent + dist[k]] * rel[r * n_heads + h] - sink[h]
extern "C" __global__ void inkling2_rel_bias_gather_f32(
    const float* __restrict__ rel,   // [n_rel, n_heads]
    const float* __restrict__ table, // [n_rel, n_extent]
    const float* __restrict__ sink,  // [n_heads]
    const int* __restrict__ dist,    // [n_kv], each < n_extent
    float* __restrict__ out,         // [n_heads, n_kv]
    int n_rel,
    int n_heads,
    int n_extent,
    int n_kv) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total = n_heads * n_kv;
    for (int t = tid; t < total; t += stride) {
        const int h = t / n_kv;
        const int k = t % n_kv;
        const int d = dist[k];
        float v = 0.0f;
        for (int r = 0; r < n_rel; ++r) {
            v += table[r * n_extent + d] * rel[r * n_heads + h];
        }
        out[t] = v - sink[h];
    }
}
"#;

/// Relative-bias gather CPU reference (oracle formula).
///
/// Panics on an out-of-range distance (fail closed: a bad distance must
/// refuse the load, never wrap into the table).
pub fn inkling2_rel_bias_f32(
    rel: &[f32],
    table: &[f32],
    sink: &[f32],
    distances: &[usize],
    n_rel: usize,
    n_heads: usize,
    n_extent: usize,
) -> Vec<f32> {
    assert_eq!(rel.len(), n_rel * n_heads);
    assert_eq!(table.len(), n_rel * n_extent);
    assert_eq!(sink.len(), n_heads);
    let n_kv = distances.len();
    // Head-outer loop order (the HIP kernel is (h, k) element-parallel;
    // the test transcription below is kv-outer).
    let mut out = vec![0.0f32; n_heads * n_kv];
    for h in 0..n_heads {
        for (k, &d) in distances.iter().enumerate() {
            assert!(d < n_extent, "relative distance {d} out of range");
            let mut v = 0.0f32;
            for r in 0..n_rel {
                v += table[r * n_extent + d] * rel[r * n_heads + h];
            }
            out[h * n_kv + k] = v - sink[h];
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

    /// Naive transcription of the oracle formula (kv-outer loop order).
    fn naive_rel_bias(
        rel: &[f32],
        table: &[f32],
        sink: &[f32],
        distances: &[usize],
        n_rel: usize,
        n_heads: usize,
        n_extent: usize,
    ) -> Vec<f32> {
        let n_kv = distances.len();
        let mut out = vec![0.0f32; n_heads * n_kv];
        for (k, &d) in distances.iter().enumerate() {
            assert!(d < n_extent);
            for h in 0..n_heads {
                let mut v = 0.0f32;
                for r in 0..n_rel {
                    v += table[r * n_extent + d] * rel[r * n_heads + h];
                }
                out[h * n_kv + k] = v - sink[h];
            }
        }
        out
    }

    #[test]
    fn oracle_golden_subtracts_sink() {
        // Oracle golden: table rows [1,2,3,4],[5,6,7,8]; R=[2,3]; sink=1.
        // d=0: 1*2+5*3-1=16; d=2: 3*2+7*3-1=26.
        let table = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let out = inkling2_rel_bias_f32(&[2.0, 3.0], &table, &[1.0], &[0, 2], 2, 1, 4);
        assert_eq!(out, vec![16.0, 26.0]);
    }

    #[test]
    fn model_dims_match_oracle_formula() {
        // Model geometry, synthetic data: d_rel 16, 8 KV heads, extent 1024.
        let (n_rel, n_heads, n_extent, n_kv) = (16usize, 8usize, 1024usize, 37usize);
        let mut s = 0xabcd_ef01_2345_6789u64;
        let rel = rand_vec(&mut s, n_rel * n_heads, -1.0, 1.0);
        let table = rand_vec(&mut s, n_rel * n_extent, -0.5, 0.5);
        let sink = rand_vec(&mut s, n_heads, -1.0, 1.0);
        let distances: Vec<usize> = (0..n_kv).map(|_| (rng_next(&mut s) % n_extent as u64) as usize).collect();
        let got = inkling2_rel_bias_f32(&rel, &table, &sink, &distances, n_rel, n_heads, n_extent);
        let want = naive_rel_bias(&rel, &table, &sink, &distances, n_rel, n_heads, n_extent);
        assert_eq!(got.len(), n_heads * n_kv);
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "i={i}: {a} vs {b}");
        }
    }

    #[test]
    fn sink_subtract_commutes_with_distance_gather() {
        // The sink is distance-independent: folding it into the table column
        // first must give the identical row.
        let (n_rel, n_heads, n_extent, n_kv) = (16usize, 8usize, 1024usize, 11usize);
        let mut s = 0x1357_9bdf_0246_8aceu64;
        let rel = rand_vec(&mut s, n_rel * n_heads, -1.0, 1.0);
        let table = rand_vec(&mut s, n_rel * n_extent, -0.5, 0.5);
        let sink = rand_vec(&mut s, n_heads, -1.0, 1.0);
        let distances: Vec<usize> = (0..n_kv).map(|_| (rng_next(&mut s) % n_extent as u64) as usize).collect();
        let got = inkling2_rel_bias_f32(&rel, &table, &sink, &distances, n_rel, n_heads, n_extent);
        // Gathering with a zero sink, then subtracting sink[h] per head,
        // must reproduce the sink-subtracted output.
        let zero_sink = vec![0.0f32; n_heads];
        let gathered = inkling2_rel_bias_f32(&rel, &table, &zero_sink, &distances, n_rel, n_heads, n_extent);
        for h in 0..n_heads {
            for k in 0..n_kv {
                assert!((gathered[h * n_kv + k] - sink[h] - got[h * n_kv + k]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn heads_are_independent() {
        // Head h's output must not depend on any other head's rel/sink.
        let (n_rel, n_heads, n_extent, n_kv) = (16usize, 4usize, 64usize, 7usize);
        let mut s = 0xf00d_d00d_feed_beefu64;
        let rel = rand_vec(&mut s, n_rel * n_heads, -1.0, 1.0);
        let table = rand_vec(&mut s, n_rel * n_extent, -0.5, 0.5);
        let sink = rand_vec(&mut s, n_heads, -1.0, 1.0);
        let distances: Vec<usize> = (0..n_kv).map(|_| (rng_next(&mut s) % n_extent as u64) as usize).collect();
        let full = inkling2_rel_bias_f32(&rel, &table, &sink, &distances, n_rel, n_heads, n_extent);
        // Single-head slice: rel rows restricted to head 1.
        let rel1: Vec<f32> = (0..n_rel).map(|r| rel[r * n_heads + 1]).collect();
        let one = inkling2_rel_bias_f32(&rel1, &table, &sink[1..2], &distances, n_rel, 1, n_extent);
        assert_eq!(&full[n_kv..2 * n_kv], &one[..]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn out_of_range_distance_fails_closed() {
        let table = vec![0.0f32; 2 * 4];
        inkling2_rel_bias_f32(&[1.0, 2.0], &table, &[0.0], &[4], 2, 1, 4);
    }
}
