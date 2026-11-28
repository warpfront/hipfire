// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Inkling-2 blocked-body glue fusion, attention and residual side (gfx1201
//! decode and prefill rows). As in [`crate::inkling2_fuse`], each kernel's
//! per-element arithmetic is that of the chain it replaces, in the same order:
//!
//! * [`INKLING2_FZ_ATTN_POST`]: the K/V short convs, K L2 clip, V gamma-less
//!   RMSNorm, K/V F16 stores (staging planes or the one-row cache slot), the
//!   Q/Rel scaled L2 clips and, for one row, the K/V history shift.
//! * [`INKLING2_FZ_GATE_ROT`]: the output gate's sigmoid·mul feeding the `wo`
//!   AWQ+FWHT producer.
//! * [`INKLING2_FZ_CONV_RES`]: the post-attention / post-MoE short conv plus
//!   its residual add (and, for one row, the history shift).
//! * [`INKLING2_FZ_CONV_STATE`]: the history update of up to four conv streams
//!   of a multi-row visit, once every read of the old history is done.

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor};
use crate::inkling2_fuse::fz_awq_fwht_hip;
use hip_bridge::{HipError, HipResult, KernargBlob};

/// Module of the attention/residual-side fused glue kernels.
pub const INKLING2_FUSE_POST_MODULE: &str = "inkling2_fuse_post";

/// Attention-producer epilogue ([`Gpu::inkling2_fz_attn_post`]).
pub const INKLING2_FZ_ATTN_POST: &str = "inkling2_fz_attn_post_f32";

/// Output-gate sigmoid·mul + AWQ/FWHT producer ([`Gpu::inkling2_fz_gate_rot`]).
pub const INKLING2_FZ_GATE_ROT: &str = "inkling2_fz_gate_rot_f32";

/// Short conv + residual ([`Gpu::inkling2_fz_conv_res`]).
pub const INKLING2_FZ_CONV_RES: &str = "inkling2_fz_conv_res_f32";

/// Batched conv-history update ([`Gpu::inkling2_fz_conv_state`]).
pub const INKLING2_FZ_CONV_STATE: &str = "inkling2_fz_conv_state4_f32";

/// Attention head width the fused epilogue is built for.
pub const FZ_HEAD_DIM: usize = 128;
/// Relative-logit width per head the fused epilogue is built for.
pub const FZ_N_REL: usize = 16;

pub const INKLING2_FUSE_POST_HIP_SRC: &str = concat!(
    "#include <hip/hip_runtime.h>\n",
    fz_awq_fwht_hip!(),
    r#"

// `ink_pf_conv_rows_f32` of (row r, channel ch): x_{r-3..r} from `cur` and the
// visit history, static taps (+ dynamic per-head taps when `dyn_w`), then
// (acc + xc) * inv_scale. Hands back x1, x2, xc for the one-row history shift.
__device__ __forceinline__ float fz_conv(
    const float* __restrict__ cur, int cur_ld, const float* __restrict__ sw,
    const float* __restrict__ dyn_w, int dyn_ld, const float* state,
    int state_stride, int col_off, int channels, int head_dim, float inv_scale,
    int r, int ch, float& x1o, float& x2o, float& xco)
{
    const float x0 = r >= 3 ? cur[(size_t)(r - 3) * cur_ld + ch]
                            : state[(size_t)r * state_stride + col_off + ch];
    const float x1 = r >= 2 ? cur[(size_t)(r - 2) * cur_ld + ch]
                            : state[(size_t)(r + 1) * state_stride + col_off + ch];
    const float x2 = r >= 1 ? cur[(size_t)(r - 1) * cur_ld + ch]
                            : state[(size_t)(r + 2) * state_stride + col_off + ch];
    const float xc = cur[(size_t)r * cur_ld + ch];
    float acc;
    if (dyn_w == nullptr) {
        acc = x0 * sw[ch] + x1 * sw[channels + ch]
            + x2 * sw[channels * 2 + ch] + xc * sw[channels * 3 + ch];
    } else {
        const int h = ch / head_dim;
        const float* drow = dyn_w + (size_t)r * dyn_ld + (size_t)h * 4;
        acc = x0 * sw[ch] + x1 * sw[channels + ch]
            + x2 * sw[channels * 2 + ch] + xc * sw[channels * 3 + ch]
            + xc * drow[0] + x2 * drow[1] + x1 * drow[2] + x0 * drow[3];
    }
    x1o = x1;
    x2o = x2;
    xco = xc;
    return (acc + xc) * inv_scale;
}

// One-row history shift (`ink_pf_conv_state_f32` at rows == 1), by the
// thread that just read the history of this channel.
__device__ __forceinline__ void fz_conv_shift(
    float* state, int state_stride, int col_off, int ch, float x1, float x2, float xc)
{
    state[col_off + ch] = x1;
    state[(size_t)state_stride + col_off + ch] = x2;
    state[(size_t)state_stride * 2 + col_off + ch] = xc;
}

struct FzPlanes8 { _Float16* p[8]; };

// Attention producers' epilogue of one visit, block (part, row b), 128 threads
// (head_dim 128, n_rel 16). Parts [0, nkv): K head (conv, L2 clip at qk_clip);
// [nkv, 2nkv): V head (conv, gamma-less RMSNorm); then nh Q heads (scaled L2
// clip in place) and nh / 8 Rel blocks (8 heads of 16, scaled L2 clip in
// place). K/V land as F32 rows (`k_f32`/`v_f32`, nullable) and/or F16 at
// `k16.p[h] + b * kv_ld` (nullable: staging planes, or the cache slot of a
// one-row block). `state_update` (rows == 1 only) shifts the K/V history.
// Per element the arithmetic of ink_pf_conv_rows_f32, ink_pf_l2_clip_rows_f32,
// inkling2_rmsnorm_nogamma_f32 (blockDim 256 over 128: the upper half adds 0)
// and ink_pf_kv_stage_f16.
extern "C" __global__ __launch_bounds__(128)
void inkling2_fz_attn_post_f32(
    float* __restrict__ q, const float* __restrict__ q_scale, float qk_clip,
    float* __restrict__ rel, const float* __restrict__ rel_scale, float rel_clip,
    const float* __restrict__ k_raw, const float* __restrict__ v_raw,
    const float* __restrict__ k_sw, const float* __restrict__ v_sw,
    const float* __restrict__ dyn_k, const float* __restrict__ dyn_v, int dyn_ld,
    float conv_scale, float* state, int state_stride, int k_off, int v_off,
    int state_update, float v_eps,
    float* __restrict__ k_f32, float* __restrict__ v_f32,
    FzPlanes8 k16, FzPlanes8 v16, int kv_ld,
    int rows, int nh, int nkv)
{
    const int blk = (int)blockIdx.x;
    const int b = (int)blockIdx.y;
    const int t = (int)threadIdx.x;
    if (b >= rows) return;
    __shared__ float sh[128];
    const int ekv = nkv * 128;
    if (blk < 2 * nkv) {
        const bool is_v = blk >= nkv;
        const int h = is_v ? blk - nkv : blk;
        const int ch = h * 128 + t;
        const int off = is_v ? v_off : k_off;
        float x1, x2, xc;
        const float c = fz_conv(is_v ? v_raw : k_raw, ekv, is_v ? v_sw : k_sw,
                                is_v ? dyn_v : dyn_k, dyn_ld, state, state_stride, off,
                                ekv, 128, conv_scale, b, ch, x1, x2, xc);
        sh[t] = c * c;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (t < s) sh[t] += sh[t + s];
            __syncthreads();
        }
        float o;
        if (is_v) {
            const float inv = 1.0f / sqrtf(sh[0] / (float)128 + v_eps);
            o = c * inv;
        } else {
            const float norm = sqrtf(sh[0]) + 1e-6f;
            float factor = qk_clip / norm;
            if (factor > 1.0f) factor = 1.0f;
            o = c * factor;
        }
        float* f = is_v ? v_f32 : k_f32;
        if (f) f[(size_t)b * ekv + ch] = o;
        _Float16* d = is_v ? v16.p[h] : k16.p[h];
        if (d) d[(size_t)b * kv_ld + t] = (_Float16)o;
        if (state_update) fz_conv_shift(state, state_stride, off, ch, x1, x2, xc);
        return;
    }
    if (blk < 2 * nkv + nh) {
        const int h = blk - 2 * nkv;
        const size_t idx = ((size_t)b * nh + h) * 128 + t;
        float v = q[idx];
        v *= q_scale[h * 128 + t];
        sh[t] = v * v;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (t < s) sh[t] += sh[t + s];
            __syncthreads();
        }
        const float norm = sqrtf(sh[0]) + 1e-6f;
        float factor = qk_clip / norm;
        if (factor > 1.0f) factor = 1.0f;
        q[idx] = v * factor;
        return;
    }
    const int g = t >> 4;
    const int l = t & 15;
    const int hh = (blk - 2 * nkv - nh) * 8 + g;
    const size_t idx = ((size_t)b * nh + hh) * 16 + l;
    float v = rel[idx];
    v *= rel_scale[hh * 16 + l];
    sh[t] = v * v;
    __syncthreads();
    for (int s = 8; s > 0; s >>= 1) {
        if (l < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    const float norm = sqrtf(sh[g * 16]) + 1e-6f;
    float factor = rel_clip / norm;
    if (factor > 1.0f) factor = 1.0f;
    rel[idx] = v * factor;
}

// Output-gate producer: x_out = AWQ+FWHT256 of attn * sigmoid(gate) per row
// (`sigmoid_f32`, `mul_f32`, then `rotate_x_mq_awq`), folded 1-D grid
// [(K / 256) * rows], 32 threads.
extern "C" __launch_bounds__(32, 16)
__global__ void inkling2_fz_gate_rot_f32(
    const float* __restrict__ attn, const float* __restrict__ gate,
    const float* __restrict__ awq_scale, float* __restrict__ x_out,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int K)
{
    const int tid = threadIdx.x;
    const int groups = K / 256;
    if (groups <= 0) return;
    const int group = (int)(blockIdx.x % (unsigned)groups);
    const long long row = (long long)(blockIdx.x / (unsigned)groups);
    const float* a = attn + row * K;
    const float* gt = gate + row * K;
    const int base = group * 256 + tid * 8;
    float x[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float s = 1.0f / (1.0f + expf(-gt[base + k]));
        x[k] = a[base + k] * s;
    }
    fz_awq_fwht(x, awq_scale, x_out + row * K, signs1, signs2, group, tid);
}

// Static short conv of `cur [rows, channels]` (history at col_off) plus the
// residual: out = res + conv (`add_f32`), or conv when `res` is null.
// `state_update` (rows == 1 only) shifts the history in place.
extern "C" __global__ void inkling2_fz_conv_res_f32(
    const float* __restrict__ cur, const float* __restrict__ sw,
    float* state, int state_stride, int col_off, int channels, float inv_scale,
    const float* __restrict__ res, float* __restrict__ out, int rows, int state_update)
{
    const int ch = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int r = (int)blockIdx.y;
    if (ch >= channels || r >= rows) return;
    float x1, x2, xc;
    const float c = fz_conv(cur, channels, sw, nullptr, 0, state, state_stride, col_off,
                            channels, 1, inv_scale, r, ch, x1, x2, xc);
    const size_t i = (size_t)r * channels + ch;
    out[i] = res ? res[i] + c : c;
    if (state_update) fz_conv_shift(state, state_stride, col_off, ch, x1, x2, xc);
}

// History after `rows` more raw inputs for up to 4 streams (blockIdx.y):
// `ink_pf_conv_state_f32` per stream.
struct FzStreams4 { const float* cur[4]; int off[4]; int ch[4]; };
extern "C" __global__ void inkling2_fz_conv_state4_f32(
    FzStreams4 st, float* __restrict__ state, int state_stride, int rows)
{
    const int k = (int)blockIdx.y;
    const int ch = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int channels = st.ch[k];
    if (ch >= channels) return;
    const float* cur = st.cur[k];
    const int off = st.off[k];
    float seq[3];
    for (int i = 0; i < 3; ++i) {
        const int j = rows + i;
        seq[i] = j >= 3 ? cur[(size_t)(j - 3) * channels + ch]
                        : state[(size_t)j * state_stride + off + ch];
    }
    state[off + ch] = seq[0];
    state[(size_t)state_stride + off + ch] = seq[1];
    state[(size_t)state_stride * 2 + off + ch] = seq[2];
}
"#
);

/// F16 K/V destinations of [`Gpu::inkling2_fz_attn_post`]: row `b` of head
/// `h` lands at `k[h] + b * row_ld` (F16 elements): staging planes
/// `[nkv, rows, 128]` (`row_ld` 128) or, for one row, the cache ring slot.
pub struct FzKv16 {
    pub k: [*const c_void; 8],
    pub v: [*const c_void; 8],
    pub row_ld: usize,
}

/// [`Gpu::inkling2_fz_attn_post`] operands (head_dim 128, n_rel 16).
pub struct Inkling2FzAttnPost<'a> {
    /// `[rows, nh, 128]`, clipped in place.
    pub q: &'a GpuTensor,
    pub q_scale: &'a GpuTensor,
    pub qk_clip: f32,
    /// `[rows, nh, 16]`, clipped in place.
    pub rel: &'a GpuTensor,
    pub rel_scale: &'a GpuTensor,
    pub rel_clip: f32,
    /// Raw K/V projections `[rows, nkv * 128]`.
    pub k_raw: &'a GpuTensor,
    pub v_raw: &'a GpuTensor,
    /// Static taps `[4, nkv * 128]`.
    pub k_sw: &'a GpuTensor,
    pub v_sw: &'a GpuTensor,
    /// Dynamic per-head taps `(k [rows, ld], v [rows, ld], ld)` (full layers).
    pub dyn_kv: Option<(&'a GpuTensor, &'a GpuTensor, usize)>,
    pub conv_scale: f32,
    /// Conv history `[3, state_stride]`; K at column `k_off`, V at `v_off`.
    pub state: &'a GpuTensor,
    pub state_stride: usize,
    pub k_off: usize,
    pub v_off: usize,
    /// Shift the K/V history in the launch (one row only).
    pub state_update: bool,
    pub v_eps: f32,
    /// F32 K / V rows `[rows, nkv * 128]` (row-exact attention).
    pub k_f32: Option<&'a GpuTensor>,
    pub v_f32: Option<&'a GpuTensor>,
    pub kv16: Option<FzKv16>,
    pub rows: usize,
    pub nh: usize,
    pub nkv: usize,
}

impl Gpu {
    fn fz_post_ensure(&mut self, sym: &str) -> HipResult<()> {
        self.ensure_kernel(INKLING2_FUSE_POST_MODULE, INKLING2_FUSE_POST_HIP_SRC, sym)
    }

    fn fz_signs(&mut self) -> HipResult<(*const c_void, *const c_void)> {
        self.ensure_mq_signs()?;
        match (
            self.scratch.mq_signs1.as_ref(),
            self.scratch.mq_signs2.as_ref(),
        ) {
            (Some(a), Some(b)) => Ok((
                a.buf.as_ptr() as *const c_void,
                b.buf.as_ptr() as *const c_void,
            )),
            _ => Err(HipError::new(0, "inkling2_fuse: no mq signs")),
        }
    }

    /// One visit's attention-producer epilogue over `rows` rows in one launch:
    /// K/V short convs (static, plus dynamic taps on full layers), K L2 clip,
    /// V gamma-less RMSNorm, K/V F32 rows and/or F16 stores, Q and Rel scaled
    /// L2 clips in place, and (one row) the K/V history shift. Bitwise the
    /// unfused conv / clip / norm / F16-stage chain.
    pub fn inkling2_fz_attn_post(&mut self, a: &Inkling2FzAttnPost<'_>) -> HipResult<()> {
        self.bind_thread()?;
        let bad = |m: String| Err(HipError::new(0, &format!("inkling2_fz_attn_post: {m}")));
        let (rows, nh, nkv) = (a.rows, a.nh, a.nkv);
        let (hd, nrel) = (FZ_HEAD_DIM, FZ_N_REL);
        if rows == 0 || nkv == 0 || nkv > 8 || nh == 0 || nh % 8 != 0 {
            return bad(format!("rows {rows} nh {nh} nkv {nkv}"));
        }
        if a.state_update && rows != 1 {
            return bad("in-kernel history shift needs one row".into());
        }
        let f32_len = |t: &GpuTensor| t.buf.size() / 4;
        let ekv = nkv * hd;
        let dyn_ok = a.dyn_kv.is_none_or(|(k, v, ld)| {
            ld >= nkv * 4 && f32_len(k) >= rows * ld && f32_len(v) >= rows * ld
        });
        if f32_len(a.q) < rows * nh * hd
            || f32_len(a.q_scale) < nh * hd
            || f32_len(a.rel) < rows * nh * nrel
            || f32_len(a.rel_scale) < nh * nrel
            || f32_len(a.k_raw) < rows * ekv
            || f32_len(a.v_raw) < rows * ekv
            || f32_len(a.k_sw) < 4 * ekv
            || f32_len(a.v_sw) < 4 * ekv
            || f32_len(a.state) < 2 * a.state_stride + a.k_off.max(a.v_off) + ekv
            || a.k_f32.is_some_and(|t| f32_len(t) < rows * ekv)
            || a.v_f32.is_some_and(|t| f32_len(t) < rows * ekv)
            || !dyn_ok
        {
            return bad("operand too small".into());
        }
        self.fz_post_ensure(INKLING2_FZ_ATTN_POST)?;
        let p = |t: &GpuTensor| t.buf.as_ptr() as *const c_void;
        let null = std::ptr::null::<c_void>();
        let mut b = KernargBlob::new();
        b.push_ptr(p(a.q));
        b.push_ptr(p(a.q_scale));
        b.push_f32(a.qk_clip);
        b.push_ptr(p(a.rel));
        b.push_ptr(p(a.rel_scale));
        b.push_f32(a.rel_clip);
        b.push_ptr(p(a.k_raw));
        b.push_ptr(p(a.v_raw));
        b.push_ptr(p(a.k_sw));
        b.push_ptr(p(a.v_sw));
        b.push_ptr(a.dyn_kv.map_or(null, |(k, _, _)| p(k)));
        b.push_ptr(a.dyn_kv.map_or(null, |(_, v, _)| p(v)));
        b.push_i32(a.dyn_kv.map_or(0, |(_, _, ld)| ld as i32));
        b.push_f32(a.conv_scale);
        b.push_ptr(p(a.state));
        b.push_i32(a.state_stride as i32);
        b.push_i32(a.k_off as i32);
        b.push_i32(a.v_off as i32);
        b.push_i32(a.state_update as i32);
        b.push_f32(a.v_eps);
        b.push_ptr(a.k_f32.map_or(null, p));
        b.push_ptr(a.v_f32.map_or(null, p));
        for h in 0..8 {
            b.push_ptr(
                a.kv16
                    .as_ref()
                    .filter(|_| h < nkv)
                    .map_or(null, |kv| kv.k[h]),
            );
        }
        for h in 0..8 {
            b.push_ptr(
                a.kv16
                    .as_ref()
                    .filter(|_| h < nkv)
                    .map_or(null, |kv| kv.v[h]),
            );
        }
        b.push_i32(a.kv16.as_ref().map_or(0, |kv| kv.row_ld as i32));
        b.push_i32(rows as i32);
        b.push_i32(nh as i32);
        b.push_i32(nkv as i32);
        b.pad_to(16);
        let parts = (2 * nkv + nh + nh / 8) as u32;
        let rows_u = u32::try_from(rows).map_err(|_| HipError::new(0, "rows exceed u32"))?;
        self.launch_kernel_blob(
            INKLING2_FZ_ATTN_POST,
            [parts, rows_u, 1],
            [hd as u32, 1, 1],
            0,
            b.as_mut_slice(),
        )
    }

    /// `x_rot [rows, k]` = AWQ+FWHT256 of `attn * sigmoid(gate)` per row, one
    /// launch, bitwise `sigmoid_f32` + `mul_f32` + `rotate_x_mq_awq`.
    pub fn inkling2_fz_gate_rot(
        &mut self,
        attn: &GpuTensor,
        gate: &GpuTensor,
        awq_scale: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let f32_len = |t: &GpuTensor| t.buf.size() / 4;
        if k == 0
            || k % 256 != 0
            || rows == 0
            || f32_len(attn) < rows * k
            || f32_len(gate) < rows * k
            || f32_len(x_rot) < rows * k
            || f32_len(awq_scale) < k
        {
            return Err(HipError::new(0, "inkling2_fz_gate_rot: operand shape"));
        }
        let (s1, s2) = self.fz_signs()?;
        self.fz_post_ensure(INKLING2_FZ_GATE_ROT)?;
        let mut b = KernargBlob::new();
        b.push_ptr(attn.buf.as_ptr() as *const c_void);
        b.push_ptr(gate.buf.as_ptr() as *const c_void);
        b.push_ptr(awq_scale.buf.as_ptr() as *const c_void);
        b.push_ptr(x_rot.buf.as_ptr() as *const c_void);
        b.push_ptr(s1);
        b.push_ptr(s2);
        b.push_i32(k as i32);
        b.pad_to(16);
        let grid = u32::try_from((k / 256) * rows)
            .map_err(|_| HipError::new(0, "inkling2_fz_gate_rot: grid exceeds u32"))?;
        self.launch_kernel_blob(
            INKLING2_FZ_GATE_ROT,
            [grid, 1, 1],
            [32, 1, 1],
            0,
            b.as_mut_slice(),
        )?;
        self.scratch.invalidate_x_caches_for(x_rot.buf.as_ptr());
        Ok(())
    }

    /// `out [rows, channels]` = `res` + static short conv of `cur` (history
    /// at `col_off` of `state`, row stride `state_stride`), or the conv alone
    /// without `res`; `state_update` (one row) shifts the history in the same
    /// launch. Bitwise `ink_pf_conv_rows_f32` (+ `add_f32`).
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_fz_conv_res(
        &mut self,
        cur: &GpuTensor,
        static_w: &GpuTensor,
        state: &GpuTensor,
        state_stride: usize,
        col_off: usize,
        channels: usize,
        inv_scale: f32,
        res: Option<&GpuTensor>,
        out: &GpuTensor,
        rows: usize,
        state_update: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let f32_len = |t: &GpuTensor| t.buf.size() / 4;
        if rows == 0
            || (state_update && rows != 1)
            || f32_len(cur) < rows * channels
            || f32_len(out) < rows * channels
            || res.is_some_and(|r| f32_len(r) < rows * channels)
            || f32_len(static_w) < 4 * channels
            || f32_len(state) < 2 * state_stride + col_off + channels
        {
            return Err(HipError::new(0, "inkling2_fz_conv_res: operand shape"));
        }
        self.fz_post_ensure(INKLING2_FZ_CONV_RES)?;
        let mut b = KernargBlob::new();
        b.push_ptr(cur.buf.as_ptr() as *const c_void);
        b.push_ptr(static_w.buf.as_ptr() as *const c_void);
        b.push_ptr(state.buf.as_ptr() as *const c_void);
        b.push_i32(state_stride as i32);
        b.push_i32(col_off as i32);
        b.push_i32(channels as i32);
        b.push_f32(inv_scale);
        b.push_ptr(res.map_or(std::ptr::null(), |r| r.buf.as_ptr() as *const c_void));
        b.push_ptr(out.buf.as_ptr() as *const c_void);
        b.push_i32(rows as i32);
        b.push_i32(state_update as i32);
        b.pad_to(16);
        let rows_u = u32::try_from(rows).map_err(|_| HipError::new(0, "rows exceed u32"))?;
        self.launch_kernel_blob(
            INKLING2_FZ_CONV_RES,
            [channels.div_ceil(256) as u32, rows_u, 1],
            [256, 1, 1],
            0,
            b.as_mut_slice(),
        )
    }

    /// Conv histories after `rows` more raw inputs for up to 4 streams
    /// `(cur [rows, channels], col_off, channels)` of one `state`, one launch
    /// (`ink_pf_conv_state_f32` per stream). Run after every read of the old
    /// history.
    pub fn inkling2_fz_conv_state(
        &mut self,
        streams: &[(&GpuTensor, usize, usize)],
        state: &GpuTensor,
        state_stride: usize,
        rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let f32_len = |t: &GpuTensor| t.buf.size() / 4;
        if streams.is_empty()
            || streams.len() > 4
            || rows == 0
            || streams.iter().any(|&(c, off, ch)| {
                f32_len(c) < rows * ch || f32_len(state) < 2 * state_stride + off + ch
            })
        {
            return Err(HipError::new(0, "inkling2_fz_conv_state: operand shape"));
        }
        self.fz_post_ensure(INKLING2_FZ_CONV_STATE)?;
        let mut b = KernargBlob::new();
        for i in 0..4 {
            b.push_ptr(
                streams
                    .get(i)
                    .map_or(std::ptr::null(), |s| s.0.buf.as_ptr() as *const c_void),
            );
        }
        for i in 0..4 {
            b.push_i32(streams.get(i).map_or(0, |s| s.1 as i32));
        }
        for i in 0..4 {
            b.push_i32(streams.get(i).map_or(0, |s| s.2 as i32));
        }
        b.push_ptr(state.buf.as_ptr() as *const c_void);
        b.push_i32(state_stride as i32);
        b.push_i32(rows as i32);
        b.pad_to(16);
        let max_ch = streams.iter().map(|s| s.2).max().unwrap_or(0);
        self.launch_kernel_blob(
            INKLING2_FZ_CONV_STATE,
            [max_ch.div_ceil(256) as u32, streams.len() as u32, 1],
            [256, 1, 1],
            0,
            b.as_mut_slice(),
        )
    }
}
