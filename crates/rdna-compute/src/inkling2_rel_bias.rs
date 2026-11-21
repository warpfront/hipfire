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
//! Layouts (row-major, F32): `rel: [n_heads, n_rel]` (head-major: row `h`
//! is Q head `h`'s `n_rel` dims, exactly the `rc[h * n_rel..]` rows
//! `forward.rs` builds from `wr_du` and the `inkling2_l2_clip_rows_f32`
//! output — no transpose), `table: [n_rel, n_extent]`, `sink: [n_heads]`,
//! `distances: [n_kv]` with every entry `< n_extent` on the CPU path.
//! Returns `[n_heads, n_kv]`.
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
/// Each thread gathers its distance column across the `n_rel` (16) table
/// rows against its head's `rel` row, then subtracts `sink[h]`. `rel` is
/// head-major (`rel[h * n_rel + r]`, matching `forward.rs`'s `rc` rows).
/// Out-of-extent distances clamp to the last table column in-shader
/// (`d >= n_extent` only happens at `pos >= 1024`); upstream zeroes that
/// term instead (see brief 04) — the clamp-vs-zero choice is documented
/// here because briefs 03/05 need `n_kv > 1` from day one. The CPU
/// reference below fails closed (panics) instead: shader and reference
/// agree on every in-extent distance.
pub const INKLING2_REL_BIAS_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
// out[h * n_kv + k] = sum_r table[r * n_extent + d] * rel[h * n_rel + r] - sink[h]
extern "C" __global__ void inkling2_rel_bias_gather_f32(
    const float* __restrict__ rel,   // [n_heads, n_rel]
    const float* __restrict__ table, // [n_rel, n_extent]
    const float* __restrict__ sink,  // [n_heads]
    const int* __restrict__ dist,    // [n_kv], each < n_extent on the fast path
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
        const int d0 = dist[k];
        const int d = d0 < n_extent ? d0 : n_extent - 1;
        float v = 0.0f;
        for (int r = 0; r < n_rel; ++r) {
            v += table[r * n_extent + d] * rel[h * n_rel + r];
        }
        out[t] = v - sink[h];
    }
}
"#;

/// Relative-bias gather CPU reference (oracle formula, head-major `rel`).
///
/// `rel` is `[n_heads, n_rel]` row-major (row `h` = head `h`'s dims).
///
/// Panics on an out-of-range distance (fail closed: a bad distance must
/// refuse the load, never wrap into the table). The HIP kernel clamps
/// instead — see [`INKLING2_REL_BIAS_HIP_SRC`].
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
                v += table[r * n_extent + d] * rel[h * n_rel + r];
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
                    v += table[r * n_extent + d] * rel[h * n_rel + r];
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
    fn head_major_matches_wr_du_output_layout() {
        // forward.rs reads `rc[h * n_rel..(h+1) * n_rel]` per Q head: the
        // reference must index head-major, not rel-major. Two heads, n_rel 2,
        // extent 2: head 0 row [1,0] picks table column d's row0, head 1
        // row [0,1] picks row1.
        let table = [10.0f32, 20.0, 30.0, 40.0]; // row0 [10,20], row1 [30,40]
        let rel = [1.0f32, 0.0, 0.0, 1.0]; // head0 [1,0], head1 [0,1]
        let out = inkling2_rel_bias_f32(&rel, &table, &[0.0, 0.0], &[0, 1], 2, 2, 2);
        // head0: d0 -> 10, d1 -> 20; head1: d0 -> 30, d1 -> 40.
        assert_eq!(out, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn model_dims_match_oracle_formula() {
        // Model geometry at the Q-head count: 24 heads × d_rel 16.
        let (n_rel, n_heads, n_extent, n_kv) = (16usize, 24usize, 1024usize, 37usize);
        let mut s = 0xabcd_ef01_2345_6789u64;
        let rel = rand_vec(&mut s, n_rel * n_heads, -1.0, 1.0);
        let table = rand_vec(&mut s, n_rel * n_extent, -0.5, 0.5);
        let sink = rand_vec(&mut s, n_heads, -1.0, 1.0);
        let distances: Vec<usize> = (0..n_kv)
            .map(|_| (rng_next(&mut s) % n_extent as u64) as usize)
            .collect();
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
        let distances: Vec<usize> = (0..n_kv)
            .map(|_| (rng_next(&mut s) % n_extent as u64) as usize)
            .collect();
        let got = inkling2_rel_bias_f32(&rel, &table, &sink, &distances, n_rel, n_heads, n_extent);
        // Gathering with a zero sink, then subtracting sink[h] per head,
        // must reproduce the sink-subtracted output.
        let zero_sink = vec![0.0f32; n_heads];
        let gathered = inkling2_rel_bias_f32(
            &rel, &table, &zero_sink, &distances, n_rel, n_heads, n_extent,
        );
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
        let distances: Vec<usize> = (0..n_kv)
            .map(|_| (rng_next(&mut s) % n_extent as u64) as usize)
            .collect();
        let full = inkling2_rel_bias_f32(&rel, &table, &sink, &distances, n_rel, n_heads, n_extent);
        // Single-head slice: head 1's row (head-major: rel[1*n_rel..2*n_rel]).
        let rel1: Vec<f32> = rel[n_rel..2 * n_rel].to_vec();
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

// --- GPU dispatch (gfx1100) -------------------------------------------------

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor};
use hip_bridge::HipResult;

impl Gpu {
    /// `out[h, k] = sum_r table[r, dist[k]] * rel[h, r] - sink[h]`.
    ///
    /// `rel` is head-major `[n_heads, n_rel]` (the `rc` rows `forward.rs`
    /// builds), `table` is `[n_rel, n_extent]`, `sink` is `[n_heads]`,
    /// `dist` holds `n_kv` little-endian `i32` distances (`dist[k] = pos -
    /// key_pos`, any `pos >= 0`), `out` is `[n_heads, n_kv]`.
    /// Launch shape mirrors the element-parallel tail kernels in `norm.rs`:
    /// `launch_maybe_blob` (graph-safe kernargs) on a 1-D grid.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_rel_bias_gather_f32(
        &mut self,
        rel: &GpuTensor,
        table: &GpuTensor,
        sink: &GpuTensor,
        dist: &GpuTensor,
        out: &GpuTensor,
        n_rel: usize,
        n_heads: usize,
        n_extent: usize,
        n_kv: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            rel.shape,
            vec![n_heads, n_rel],
            "rel must be [n_heads, n_rel]"
        );
        assert_eq!(table.shape, vec![n_rel, n_extent]);
        assert_eq!(sink.shape, vec![n_heads]);
        assert_eq!(out.shape, vec![n_heads, n_kv]);
        self.ensure_kernel(
            "inkling2_rel_bias",
            INKLING2_REL_BIAS_HIP_SRC,
            "inkling2_rel_bias_gather_f32",
        )?;
        let rel_ptr = rel.buf.as_ptr();
        let table_ptr = table.buf.as_ptr();
        let sink_ptr = sink.buf.as_ptr();
        let dist_ptr = dist.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let n_rel_val = n_rel as i32;
        let n_heads_val = n_heads as i32;
        let n_extent_val = n_extent as i32;
        let n_kv_val = n_kv as i32;
        let mut params: Vec<*mut c_void> = vec![
            &rel_ptr as *const _ as *mut c_void,
            &table_ptr as *const _ as *mut c_void,
            &sink_ptr as *const _ as *mut c_void,
            &dist_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &n_rel_val as *const _ as *mut c_void,
            &n_heads_val as *const _ as *mut c_void,
            &n_extent_val as *const _ as *mut c_void,
            &n_kv_val as *const _ as *mut c_void,
        ];
        let total = n_heads * n_kv;
        let grid = [total.div_ceil(256) as u32, 1, 1];
        self.launch_maybe_blob(
            "inkling2_rel_bias_gather_f32",
            grid,
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(rel_ptr);
                b.push_ptr(table_ptr);
                b.push_ptr(sink_ptr);
                b.push_ptr(dist_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(n_rel_val);
                b.push_i32(n_heads_val);
                b.push_i32(n_extent_val);
                b.push_i32(n_kv_val);
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
        rel_h: &[f32],
        table_h: &[f32],
        sink_h: &[f32],
        distances: &[usize],
        n_rel: usize,
        n_heads: usize,
        n_extent: usize,
    ) -> f32 {
        let n_kv = distances.len();
        let rel = gpu
            .upload_f32(rel_h, &[n_heads, n_rel])
            .expect("rel upload");
        let table = gpu
            .upload_f32(table_h, &[n_rel, n_extent])
            .expect("table upload");
        let sink = gpu.upload_f32(sink_h, &[n_heads]).expect("sink upload");
        let dist_bytes: Vec<u8> = distances
            .iter()
            .flat_map(|&d| (d as i32).to_le_bytes())
            .collect();
        let dist = gpu
            .upload_raw(&dist_bytes, &[dist_bytes.len()])
            .expect("dist upload");
        let out = gpu
            .alloc_tensor(&[n_heads, n_kv], crate::dispatch::DType::F32)
            .expect("out alloc");
        gpu.inkling2_rel_bias_gather_f32(
            &rel, &table, &sink, &dist, &out, n_rel, n_heads, n_extent, n_kv,
        )
        .expect("rel_bias launch");
        let got = gpu.download_f32(&out).expect("download");
        let want =
            inkling2_rel_bias_f32(rel_h, table_h, sink_h, distances, n_rel, n_heads, n_extent);
        assert_eq!(got.len(), want.len());
        got.iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    /// Gate: device output matches the crate F32 reference at model dims
    /// (24 Q heads) and on a randomized case, both with n_kv > 1 varying
    /// distances. Bar: max abs diff <= 1e-5.
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
        // (a) model dims: 24 Q heads x d_rel 16, extent 1024, 37 keys.
        let (n_rel, n_heads, n_extent, n_kv) = (16usize, 24usize, 1024usize, 37usize);
        let mut s = 0x51ab_3c9d_2ef0_1123u64;
        let rel: Vec<f32> = (0..n_rel * n_heads)
            .map(|_| rng_f32(&mut s, -1.0, 1.0))
            .collect();
        let table: Vec<f32> = (0..n_rel * n_extent)
            .map(|_| rng_f32(&mut s, -0.5, 0.5))
            .collect();
        let sink: Vec<f32> = (0..n_heads).map(|_| rng_f32(&mut s, -1.0, 1.0)).collect();
        let distances: Vec<usize> = (0..n_kv)
            .map(|_| (rng_next(&mut s) % n_extent as u64) as usize)
            .collect();
        let d = run_case(
            &mut gpu, &rel, &table, &sink, &distances, n_rel, n_heads, n_extent,
        );
        eprintln!("rel_bias model-dims max abs diff: {d}");
        assert!(d <= 1e-5, "model dims diff {d}");
        // (b) randomized small case, n_kv > 1, varying distances incl. 0.
        let (n_rel, n_heads, n_extent, n_kv) = (4usize, 3usize, 32usize, 9usize);
        let mut s = 0x77aa_bbcc_ddee_ff02u64;
        let rel: Vec<f32> = (0..n_rel * n_heads)
            .map(|_| rng_f32(&mut s, -2.0, 2.0))
            .collect();
        let table: Vec<f32> = (0..n_rel * n_extent)
            .map(|_| rng_f32(&mut s, -1.0, 1.0))
            .collect();
        let sink: Vec<f32> = (0..n_heads).map(|_| rng_f32(&mut s, -1.0, 1.0)).collect();
        let mut distances: Vec<usize> = (0..n_kv)
            .map(|_| (rng_next(&mut s) % n_extent as u64) as usize)
            .collect();
        distances[0] = 0;
        distances[1] = n_extent - 1;
        let d = run_case(
            &mut gpu, &rel, &table, &sink, &distances, n_rel, n_heads, n_extent,
        );
        eprintln!("rel_bias randomized max abs diff: {d}");
        assert!(d <= 1e-5, "randomized diff {d}");
    }
}
