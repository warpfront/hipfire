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

/// Gamma-less batched RMSNorm shared by the controller input norms, the
/// `mix()` state-bank norm and the attention V-head norm (all Inkling-2
/// norms are gamma-less, so no dummy ones-tensor through `rmsnorm_batched`).
/// One block per row, 256 threads, grid-stride over `n`; `x` and `out` may
/// be distinct buffers (the controller norms into `gate`, `mix` into its
/// normed bank).
pub const INKLING2_RMSNORM_NOGAMMA_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
// out[r * n + i] = x[r * n + i] / sqrt(mean_sq + eps)
extern "C" __global__ void inkling2_rmsnorm_nogamma_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int batch,
    int n,
    float eps) {
    const int r = blockIdx.x;
    if (r >= batch) return;
    const float* xr = x + (size_t)r * n;
    float* orow = out + (size_t)r * n;
    const int t = threadIdx.x; // blockDim.x == 256, fixed
    __shared__ float sh[256];
    float acc = 0.0f;
    for (int i = t; i < n; i += blockDim.x) {
        const float v = xr[i];
        acc += v * v;
    }
    sh[t] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(sh[0] / (float)n + eps);
    for (int i = t; i < n; i += blockDim.x) {
        orow[i] = xr[i] * inv;
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
        let out =
            inkling2_control_f32(&cur, &[9.0, 9.0], &[0.0; 4], &[0.0, 0.0], &[0.0, 0.0], 2, 1);
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

// --- GPU dispatch (gfx1100) -------------------------------------------------

use std::ffi::c_void;

use crate::dispatch::{DType, Gpu, GpuTensor};
use hip_bridge::{DeviceBuffer, HipResult};

/// Byte-offset view into `buf` (mirrors `rows_buffer` in `norm.rs`):
/// the controller carves `gate`, `g`, `gain_out`, `forget_out` out of one
/// caller-owned scratch allocation, so the per-call path joins nothing and
/// allocates nothing.
fn scratch_view(buf: &DeviceBuffer, float_off: usize, floats: usize, dtype: DType) -> GpuTensor {
    let byte_off = float_off * 4;
    let len = floats * 4;
    assert!(
        byte_off + len <= buf.size(),
        "scratch view [{byte_off}, {}) exceeds buffer {}",
        byte_off + len,
        buf.size()
    );
    // SAFETY: `[byte_off, byte_off + len)` lies inside `buf`; the view is
    // non-owning (`Borrowed`), so dropping it never frees the parent.
    unsafe {
        GpuTensor {
            buf: DeviceBuffer::from_raw(
                (buf.as_ptr() as *mut u8).add(byte_off) as *mut c_void,
                len,
            ),
            shape: vec![floats],
            dtype,
        }
    }
}

impl Gpu {
    /// Gamma-less batched RMSNorm: `out[r] = rmsnorm(x[r], eps)` for
    /// `batch` rows of `n` (shared by the controller input norms, the
    /// `mix()` state-bank norm and the attention V-head norm).
    pub fn inkling2_rmsnorm_nogamma_f32(
        &mut self,
        x: &GpuTensor,
        out: &GpuTensor,
        batch: usize,
        n: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "inkling2_rmsnorm_nogamma",
            INKLING2_RMSNORM_NOGAMMA_HIP_SRC,
            "inkling2_rmsnorm_nogamma_f32",
        )?;
        let x_ptr = x.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let batch_val = batch as i32;
        let n_val = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &batch_val as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "inkling2_rmsnorm_nogamma_f32",
            [batch as u32, 1, 1],
            [256, 1, 1],
            256 * 4,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(x_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(batch_val);
                b.push_i32(n_val);
                b.push_f32(eps);
                b
            },
        )
    }

    /// Elementwise controller tail:
    /// `out[i] = cur[i] - tanh(forget[i]) * cur[i] + tanh(gain[i]) * msg[i]`.
    pub fn inkling2_control_blend_gpu_f32(
        &mut self,
        cur: &GpuTensor,
        msg: &GpuTensor,
        gain: &GpuTensor,
        forget: &GpuTensor,
        out: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "inkling2_control_blend",
            INKLING2_CONTROL_HIP_SRC,
            "inkling2_control_blend_f32",
        )?;
        let cur_ptr = cur.buf.as_ptr();
        let msg_ptr = msg.buf.as_ptr();
        let gain_ptr = gain.buf.as_ptr();
        let forget_ptr = forget.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let n_val = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &cur_ptr as *const _ as *mut c_void,
            &msg_ptr as *const _ as *mut c_void,
            &gain_ptr as *const _ as *mut c_void,
            &forget_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "inkling2_control_blend_f32",
            [n.div_ceil(256) as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cur_ptr);
                b.push_ptr(msg_ptr);
                b.push_ptr(gain_ptr);
                b.push_ptr(forget_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(n_val);
                b
            },
        )
    }

    /// One fused controller step: `out = cur - tanh(F·g)·cur + tanh(G·g)·msg`.
    ///
    /// `down_w` is F32 `[rank, 2*n_embd]`, `gain_w`/`forget_w` are F32
    /// `[n_embd, rank]` (brief-01 F32 keep-list weights arrive as device
    /// buffers). `scratch` is caller-owned F32 space for `gate[2*n_embd]`,
    /// `g[rank]`, `gain_out[n_embd]`, `forget_out[n_embd]` —
    /// `4*n_embd + rank` floats. The three projections reuse `gemv_f32`;
    /// the two input norms use [`Self::inkling2_rmsnorm_nogamma_f32`].
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_control_f32_gpu(
        &mut self,
        cur: &GpuTensor,
        msg: &GpuTensor,
        down_w: &GpuTensor,
        gain_w: &GpuTensor,
        forget_w: &GpuTensor,
        out: &GpuTensor,
        scratch: &GpuTensor,
        n_embd: usize,
        rank: usize,
    ) -> HipResult<()> {
        assert_eq!(down_w.shape, vec![rank, 2 * n_embd]);
        assert_eq!(gain_w.shape, vec![n_embd, rank]);
        assert_eq!(forget_w.shape, vec![n_embd, rank]);
        assert_eq!(scratch.shape, vec![4 * n_embd + rank]);
        let gate = scratch_view(&scratch.buf, 0, 2 * n_embd, DType::F32);
        let gate_cur = scratch_view(&scratch.buf, 0, n_embd, DType::F32);
        let gate_msg = scratch_view(&scratch.buf, n_embd, n_embd, DType::F32);
        let g = scratch_view(&scratch.buf, 2 * n_embd, rank, DType::F32);
        let gain_out = scratch_view(&scratch.buf, 2 * n_embd + rank, n_embd, DType::F32);
        let forget_out = scratch_view(&scratch.buf, 3 * n_embd + rank, n_embd, DType::F32);
        self.inkling2_rmsnorm_nogamma_f32(cur, &gate_cur, 1, n_embd, INKLING2_CONTROL_EPS)?;
        self.inkling2_rmsnorm_nogamma_f32(msg, &gate_msg, 1, n_embd, INKLING2_CONTROL_EPS)?;
        self.gemv_f32(down_w, &gate, &g)?;
        self.gemv_f32(gain_w, &g, &gain_out)?;
        self.gemv_f32(forget_w, &g, &forget_out)?;
        self.inkling2_control_blend_gpu_f32(cur, msg, &gain_out, &forget_out, out, n_embd)
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
        msg_h: &[f32],
        down_h: &[f32],
        gain_h: &[f32],
        forget_h: &[f32],
        n_embd: usize,
        rank: usize,
    ) -> f32 {
        let cur = gpu.upload_f32(cur_h, &[n_embd]).expect("cur upload");
        let msg = gpu.upload_f32(msg_h, &[n_embd]).expect("msg upload");
        let down = gpu
            .upload_f32(down_h, &[rank, 2 * n_embd])
            .expect("down upload");
        let gain = gpu
            .upload_f32(gain_h, &[n_embd, rank])
            .expect("gain upload");
        let forget = gpu
            .upload_f32(forget_h, &[n_embd, rank])
            .expect("forget upload");
        let out = gpu.alloc_tensor(&[n_embd], DType::F32).expect("out alloc");
        let scratch = gpu
            .alloc_tensor(&[4 * n_embd + rank], DType::F32)
            .expect("scratch alloc");
        gpu.inkling2_control_f32_gpu(
            &cur, &msg, &down, &gain, &forget, &out, &scratch, n_embd, rank,
        )
        .expect("control launch");
        let got = gpu.download_f32(&out).expect("download");
        let want = inkling2_control_f32(cur_h, msg_h, down_h, gain_h, forget_h, n_embd, rank);
        assert_eq!(got.len(), want.len());
        got.iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    /// Gate: device output matches the crate F32 reference at model dims
    /// (n_embd 2304, rank 64) and on a randomized case. Bar: <= 1e-4 abs.
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
        // (a) model dims, synthetic data.
        let (n_embd, rank) = (2304usize, 64usize);
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let cur: Vec<f32> = (0..n_embd).map(|_| rng_f32(&mut s, -2.0, 2.0)).collect();
        let msg: Vec<f32> = (0..n_embd).map(|_| rng_f32(&mut s, -2.0, 2.0)).collect();
        let down: Vec<f32> = (0..rank * 2 * n_embd)
            .map(|_| rng_f32(&mut s, -0.1, 0.1))
            .collect();
        let gain: Vec<f32> = (0..n_embd * rank)
            .map(|_| rng_f32(&mut s, -0.1, 0.1))
            .collect();
        let forget: Vec<f32> = (0..n_embd * rank)
            .map(|_| rng_f32(&mut s, -0.1, 0.1))
            .collect();
        let d = run_case(&mut gpu, &cur, &msg, &down, &gain, &forget, n_embd, rank);
        eprintln!("control model-dims max abs diff: {d}");
        assert!(d <= 1e-4, "model dims diff {d}");
        // (b) randomized small case.
        let (n_embd, rank) = (256usize, 8usize);
        let mut s = 0xdead_beef_cafe_f01du64;
        let cur: Vec<f32> = (0..n_embd).map(|_| rng_f32(&mut s, -2.0, 2.0)).collect();
        let msg: Vec<f32> = (0..n_embd).map(|_| rng_f32(&mut s, -2.0, 2.0)).collect();
        let down: Vec<f32> = (0..rank * 2 * n_embd)
            .map(|_| rng_f32(&mut s, -0.2, 0.2))
            .collect();
        let gain: Vec<f32> = (0..n_embd * rank)
            .map(|_| rng_f32(&mut s, -0.2, 0.2))
            .collect();
        let forget: Vec<f32> = (0..n_embd * rank)
            .map(|_| rng_f32(&mut s, -0.2, 0.2))
            .collect();
        let d = run_case(&mut gpu, &cur, &msg, &down, &gain, &forget, n_embd, rank);
        eprintln!("control randomized max abs diff: {d}");
        assert!(d <= 1e-4, "randomized diff {d}");
    }
}
