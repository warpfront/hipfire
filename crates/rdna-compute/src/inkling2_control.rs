// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Inkling-2 loop-controller fuse (gfx1100 first).
//!
//! Spec (`docs/plans/local/inkling2-kernel-inventory.md` §2, oracle
//! `hipfire-arch-inkling2/src/oracle.rs::control`): per core layer ×4
//! (Q, K, V, FFN inputs)
//! ```text
//! out = cur − tanh(F·g)·cur + tanh(G·g)·msg,
//! g   = down · (RMSNorm(cur,1e-6) ‖ RMSNorm(msg,1e-6))
//! ```
//! with `down: [rank, 2·n_embd]`, `gain G / forget F: [n_embd, rank]`
//! (model: `n_embd` 2304, `rank` 64). F32 throughout (keep-listed, never
//! requanted); tanh in F32; bitwise-vs-oracle is the gate.
//!
//! Phase-1 decomposition (inventory): the two RMSNorms reuse
//! `rmsnorm_batched`, the three projections reuse the dense GEMV paths, and
//! the tail blend is the NEW elementwise kernel below
//! ([`INKLING2_CONTROL_HIP_SRC`]). [`inkling2_control_f32`] is the fused CPU
//! reference the GPU path must match; [`inkling2_control_blend_f32`] is the
//! exact CPU image of the HIP blend kernel.

/// Controller RMSNorm epsilon (both norms).
pub const INKLING2_CONTROL_EPS: f32 = 1e-6;

/// gfx1100 elementwise tail of the controller fuse.
///
/// `gain`/`forget` are the precomputed `G·g` / `F·g` rows (`[n_embd]`);
/// norms and GEMVs arrive via the reused paths. One thread per element,
/// grid-stride loop, no shared memory — same shape as
/// `kernels/src/logit_softcap.hip`.
pub const INKLING2_CONTROL_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
// out[i] = cur[i] - tanhf(forget[i]) * cur[i] + tanhf(gain[i]) * msg[i]
extern "C" __global__ void inkling2_control_blend_f32(
    const float* __restrict__ cur,
    const float* __restrict__ msg,
    const float* __restrict__ gain,
    const float* __restrict__ forget,
    float* __restrict__ out,
    int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        const float f = tanhf(forget[i]);
        const float a = tanhf(gain[i]);
        out[i] = cur[i] - f * cur[i] + a * msg[i];
    }
}
"#;

/// Row-major `w [rows, cols]` times `x [cols]`, accumulated in K-tiles of 512
/// (the order a tiled GEMV kernel accumulates in).
fn matvec_tiled(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    const TILE: usize = 512;
    let mut y = vec![0.0f32; rows];
    let mut k0 = 0;
    while k0 < cols {
        let k1 = (k0 + TILE).min(cols);
        for (r, yv) in y.iter_mut().enumerate() {
            let base = r * cols;
            let mut acc = 0.0f32;
            for c in k0..k1 {
                acc += w[base + c] * x[c];
            }
            *yv += acc;
        }
        k0 = k1;
    }
    y
}

fn rmsnorm_row(x: &[f32]) -> Vec<f32> {
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    let scale = 1.0 / ((sum / x.len() as f32) + INKLING2_CONTROL_EPS).sqrt();
    x.iter().map(|v| v * scale).collect()
}

/// Exact CPU image of [`INKLING2_CONTROL_HIP_SRC`]: the elementwise tail.
pub fn inkling2_control_blend_f32(
    cur: &[f32],
    msg: &[f32],
    gain: &[f32],
    forget: &[f32],
) -> Vec<f32> {
    assert_eq!(cur.len(), msg.len());
    assert_eq!(cur.len(), gain.len());
    assert_eq!(cur.len(), forget.len());
    cur.iter()
        .zip(msg.iter())
        .zip(gain.iter())
        .zip(forget.iter())
        .map(|(((c, m), g), f)| c - f.tanh() * c + g.tanh() * m)
        .collect()
}

/// Fused controller step: `out = cur − tanh(F·g)·cur + tanh(G·g)·msg`.
///
/// `down_w: [rank, 2·n_embd]`, `gain_w` / `forget_w`: `[n_embd, rank]`.
pub fn inkling2_control_f32(
    cur: &[f32],
    msg: &[f32],
    down_w: &[f32],
    gain_w: &[f32],
    forget_w: &[f32],
    n_embd: usize,
    rank: usize,
) -> Vec<f32> {
    assert_eq!(cur.len(), n_embd);
    assert_eq!(msg.len(), n_embd);
    assert_eq!(down_w.len(), rank * 2 * n_embd);
    assert_eq!(gain_w.len(), n_embd * rank);
    assert_eq!(forget_w.len(), n_embd * rank);
    let cur_n = rmsnorm_row(cur);
    let msg_n = rmsnorm_row(msg);
    let mut gate = Vec::with_capacity(2 * n_embd);
    gate.extend_from_slice(&cur_n);
    gate.extend_from_slice(&msg_n);
    let g = matvec_tiled(down_w, &gate, rank, 2 * n_embd);
    let gain = matvec_tiled(gain_w, &g, n_embd, rank);
    let forget = matvec_tiled(forget_w, &g, n_embd, rank);
    inkling2_control_blend_f32(cur, msg, &gain, &forget)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic xorshift64* — no new dependencies for synthetic data.
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

    /// Naive transcription of the oracle formula (row-outer matvec, direct
    /// loops) — deliberately different accumulation order from
    /// [`matvec_tiled`].
    fn naive_control(
        cur: &[f32],
        msg: &[f32],
        down_w: &[f32],
        gain_w: &[f32],
        forget_w: &[f32],
        n_embd: usize,
        rank: usize,
    ) -> Vec<f32> {
        let norm = |x: &[f32]| -> Vec<f32> {
            let m: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let s = 1.0 / (m + 1e-6).sqrt();
            x.iter().map(|v| v * s).collect()
        };
        let mv = |w: &[f32], x: &[f32], rows: usize, cols: usize| -> Vec<f32> {
            (0..rows)
                .map(|r| (0..cols).map(|c| w[r * cols + c] * x[c]).sum())
                .collect()
        };
        let cn = norm(cur);
        let mn = norm(msg);
        let gate: Vec<f32> = cn.into_iter().chain(mn.into_iter()).collect();
        let g = mv(down_w, &gate, rank, 2 * n_embd);
        let gain = mv(gain_w, &g, n_embd, rank);
        let forget = mv(forget_w, &g, n_embd, rank);
        (0..n_embd)
            .map(|i| cur[i] - forget[i].tanh() * cur[i] + gain[i].tanh() * msg[i])
            .collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn zero_gates_are_identity() {
        // Oracle golden: zero down/gain/forget rows => tanh(0)=0 => out=cur.
        let cur = [3.0f32, 4.0];
        let out = inkling2_control_f32(&cur, &[9.0, 9.0], &[0.0; 4], &[0.0, 0.0], &[0.0, 0.0], 2, 1);
        assert_eq!(out, cur.to_vec());
    }

    #[test]
    fn gain_blends_message_and_forget_drops_current() {
        // Oracle golden: rank 1, down reads RMSNorm(cur)[0]; gain[0]/forget[1]
        // read g[0].
        let cur = [3.0f32, 4.0];
        let msg = [4.0f32, 3.0];
        let out = inkling2_control_f32(
            &cur,
            &msg,
            &[1.0, 0.0, 0.0, 0.0],
            &[1.0, 0.0],
            &[0.0, 1.0],
            2,
            1,
        );
        let rms = |x: &[f32]| -> Vec<f32> {
            let m = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            x.iter().map(|v| v / (m + 1e-6).sqrt()).collect()
        };
        let g0 = rms(&cur)[0];
        let e0 = cur[0] + g0.tanh() * msg[0];
        let e1 = cur[1] - g0.tanh() * cur[1];
        assert!((out[0] - e0).abs() < 1e-6, "{} vs {}", out[0], e0);
        assert!((out[1] - e1).abs() < 1e-6, "{} vs {}", out[1], e1);
    }

    #[test]
    fn blend_matches_hip_kernel_formula_elementwise() {
        let mut s = 0x1234_5678_9abc_def1u64;
        let n = 1024usize;
        let cur = rand_vec(&mut s, n, -3.0, 3.0);
        let msg = rand_vec(&mut s, n, -3.0, 3.0);
        let gain = rand_vec(&mut s, n, -2.0, 2.0);
        let forget = rand_vec(&mut s, n, -2.0, 2.0);
        let out = inkling2_control_blend_f32(&cur, &msg, &gain, &forget);
        for i in 0..n {
            let want = cur[i] - forget[i].tanh() * cur[i] + gain[i].tanh() * msg[i];
            assert!((out[i] - want).abs() < 1e-6, "i={i}");
        }
    }

    #[test]
    fn model_dims_match_oracle_formula() {
        // Model geometry, synthetic data: n_embd 2304 block, rank 64.
        let (n_embd, rank) = (2304usize, 64usize);
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let cur = rand_vec(&mut s, n_embd, -2.0, 2.0);
        let msg = rand_vec(&mut s, n_embd, -2.0, 2.0);
        let down_w = rand_vec(&mut s, rank * 2 * n_embd, -0.1, 0.1);
        let gain_w = rand_vec(&mut s, n_embd * rank, -0.1, 0.1);
        let forget_w = rand_vec(&mut s, n_embd * rank, -0.1, 0.1);
        let got = inkling2_control_f32(&cur, &msg, &down_w, &gain_w, &forget_w, n_embd, rank);
        let want = naive_control(&cur, &msg, &down_w, &gain_w, &forget_w, n_embd, rank);
        assert_eq!(got.len(), n_embd);
        let d = max_abs_diff(&got, &want);
        assert!(d < 1e-4, "max abs diff {d}");
    }

    #[test]
    fn fused_matches_separate_blend_tail() {
        // The fused reference must equal norms+GEMVs feeding the blend tail.
        let (n_embd, rank) = (256usize, 8usize);
        let mut s = 0xdead_beef_cafe_f00du64;
        let cur = rand_vec(&mut s, n_embd, -2.0, 2.0);
        let msg = rand_vec(&mut s, n_embd, -2.0, 2.0);
        let down_w = rand_vec(&mut s, rank * 2 * n_embd, -0.2, 0.2);
        let gain_w = rand_vec(&mut s, n_embd * rank, -0.2, 0.2);
        let forget_w = rand_vec(&mut s, n_embd * rank, -0.2, 0.2);
        let fused = inkling2_control_f32(&cur, &msg, &down_w, &gain_w, &forget_w, n_embd, rank);
        let cn = rmsnorm_row(&cur);
        let mn = rmsnorm_row(&msg);
        let gate: Vec<f32> = cn.into_iter().chain(mn.into_iter()).collect();
        let g = matvec_tiled(down_w.as_slice(), &gate, rank, 2 * n_embd);
        let gain = matvec_tiled(gain_w.as_slice(), &g, n_embd, rank);
        let forget = matvec_tiled(forget_w.as_slice(), &g, n_embd, rank);
        let tail = inkling2_control_blend_f32(&cur, &msg, &gain, &forget);
        assert_eq!(fused, tail);
    }

    #[test]
    fn nan_in_is_nan_out() {
        // Fail closed: NaN propagates through tanh and the blend.
        let out = inkling2_control_blend_f32(&[f32::NAN], &[1.0], &[0.5], &[0.5]);
        assert!(out[0].is_nan());
    }
}
