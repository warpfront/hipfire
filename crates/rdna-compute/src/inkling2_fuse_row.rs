// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Inkling-2 one-row (decode) fusions of the blocked body (gfx1201). As in
//! [`crate::inkling2_fuse`], each kernel's arithmetic is that of the launches
//! it replaces, in the same order, so outputs are bitwise theirs:
//!
//! * [`INKLING2_FZ_GEMV_SEGS`]: up to [`FZ_GEMV_SEGS`] one-row MQ4-G256-v2
//!   GEMVs over pre-rotated inputs (the visit's q, k, v, rel and gate
//!   projections), one launch. `grid.z` selects the segment; each segment runs
//!   the body of the lane-batched twin of `gemv_mq4g256v2_multirow_r2`
//!   (`kernels/src/gemv_mq4g256v2_multirow_r2_lanes.hip`), byte-identical to
//!   the singleton GEMV the one-row `weight_gemv_prerotated` launches.
//! * [`INKLING2_FZ_GEMV_CONV_RES`]: the decode visit's o projection (the same
//!   lanes body, so `y` is bitwise the singleton GEMV) with the attention post
//!   conv + residual of `inkling2_fz_conv_res_f32` (one row, history shift)
//!   done by the lane that stored each pair of channels: one launch for two.
//! * [`INKLING2_FZ_ROUTE_ROW`]: the routing of one row: sigmoid + bias top-k
//!   with renormalized weights (`ffn_moe_router_topk_f32`, kind 1), the
//!   grouped scatter tables (`moe_scatter_fused_k8`; one row routes to
//!   `top_k` distinct experts, so every bucket holds one slot and the tables
//!   are deterministic), and the per-slot AWQ+FWHT producer
//!   (`ink_pf_rotate_awq_slots`) quantized into the A8 MMQ sidecar
//!   (`quantize_int8_mmq_ds128`) the grouped gate/up GEMM reads.

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor, ReplayLaunchBindings};
use crate::inkling2_fuse::fz_awq_fwht_hip;
use crate::inkling2_fuse_post::fz_conv_hip;
use crate::scratch::{Int8MmqPrepared, Int8MmqReservation};
use hip_bridge::{HipError, HipResult, KernargBlob};

/// Module of the one-row fused kernels.
pub const INKLING2_FUSE_ROW_MODULE: &str = "inkling2_fuse_row";

/// Multi-segment one-row GEMV ([`Gpu::inkling2_fz_gemv_segs`]).
pub const INKLING2_FZ_GEMV_SEGS: &str = "inkling2_fz_gemv_mq4v2_segs";

/// One-row routing + slot producer + A8 sidecar ([`Gpu::inkling2_fz_route_row`]).
pub const INKLING2_FZ_ROUTE_ROW: &str = "inkling2_fz_route_row_f32";

/// One-row o projection + attention post conv + residual
/// ([`Gpu::inkling2_fz_gemv_conv_res`]).
pub const INKLING2_FZ_GEMV_CONV_RES: &str = "inkling2_fz_gemv_conv_res";

/// Segments per [`Gpu::inkling2_fz_gemv_segs`] launch.
pub const FZ_GEMV_SEGS: usize = 8;
/// Experts bound of [`Gpu::inkling2_fz_route_row`] (`ffn_moe_router_topk_f32`'s).
pub const FZ_ROUTE_MAX_E: usize = 1024;
/// Top-k bound of [`Gpu::inkling2_fz_route_row`] (`ffn_moe_router_topk_f32`'s).
pub const FZ_ROUTE_MAX_K: usize = 8;

pub const INKLING2_FUSE_ROW_HIP_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_mq4g256v2_multirow_r2_lanes.hip"),
    "#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1\n",
    include_str!("../../../kernels/src/block_i8_128_quant.hip"),
    fz_awq_fwht_hip!(),
    fz_conv_hip!(),
    r#"

struct FzGemvSegs {
    const char* a[8];
    const float* x[8];
    float* y[8];
    int m[8];
};

// Grid [max ceil(m/2), 1, segments], block 32: segment z is the lanes twin's
// body on (a[z], x[z], y[z], m[z]); blocks past a short segment's rows return.
__launch_bounds__(32, 18)
extern "C" __global__ void inkling2_fz_gemv_mq4v2_segs(FzGemvSegs s, int K)
{
    const int z = (int)blockIdx.z;
    gemv_mq4g256v2_multirow_r2_lanes_body(s.a[z], s.x[z], s.y[z], s.m[z], K);
}

// Grid [ceil(m/2)], block 32: the o projection (the lanes twin's body, so
// `y` is bitwise the singleton GEMV's) and, by the lane that stored them, the
// attention post conv + residual of its two channels: `inkling2_fz_conv_res_f32`
// at one row with the history shift, the same `fz_conv` text.
__launch_bounds__(32, 18)
extern "C" __global__ void inkling2_fz_gemv_conv_res(
    const char* __restrict__ A, const float* __restrict__ x, float* __restrict__ y,
    int M, int K, const float* __restrict__ sw, float* state, int state_stride,
    int col_off, float inv_scale, const float* __restrict__ res, float* __restrict__ out)
{
    gemv_mq4g256v2_multirow_r2_lanes_body(A, x, y, M, K);
    const int row0 = (int)blockIdx.x << 1;
    if (threadIdx.x != 0 || row0 >= M) return;
    for (int ch = row0; ch < row0 + 2 && ch < M; ++ch) {
        float x1, x2, xc;
        const float c = fz_conv(y, M, sw, nullptr, 0, state, state_stride, col_off,
                                M, 1, inv_scale, 0, ch, x1, x2, xc);
        out[ch] = res ? res[ch] + c : c;
        fz_conv_shift(state, state_stride, col_off, ch, x1, x2, xc);
    }
}

#define FZR_THREADS 1024
#define FZR_WAVES (FZR_THREADS / 32)
#define FZR_MAX_E 1024
#define FZR_MAX_K 8
#define FZR_LOWEST (-3.4028234663852886e38f)  /* -FLT_MAX */

// (v, i) beats (mv, mi): higher selection score, the lower expert id on a tie.
// A total order, so any reduction tree picks `ffn_moe_router_topk_f32`'s winner.
__device__ __forceinline__ void fzr_best(float& v, int& i, float ov, int oi)
{
    if (ov > v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
    }
}

__device__ __forceinline__ void fzr_wave_sync()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// One row, block 1024 (32 waves). Phases:
//   1. `ffn_moe_router_topk_f32` kind 1: the same scores, per-thread strided
//      maxima and rank-order weights; each round's argmax is reduced per wave
//      then across waves (same winner, the order is total).
//   2. `moe_scatter_fused_k8` tables for top_k distinct experts: expert e's
//      padded count is block_m when chosen, its offset block_m times the number
//      of chosen experts below it; slot r lands at its expert's offset.
//   3. task (slot s, 256-group g), one wave each: the AWQ+FWHT group into the
//      wave's LDS row, then its two 128-blocks quantized to
//      `xq[(2g + b) * top_k + s]`.
extern "C" __global__ __launch_bounds__(FZR_THREADS) void inkling2_fz_route_row_f32(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int* __restrict__ idx_out,
    float* __restrict__ w_out,
    int* __restrict__ counts,
    int* __restrict__ offsets,
    int* __restrict__ sorted_slot,
    int* __restrict__ tile_ids,
    int* __restrict__ inverse_perm,
    const float* __restrict__ x_in,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    block_i8_128* __restrict__ xq,
    int n_experts,
    int top_k,
    float eps,
    float scale,
    int K,
    int m_total_max,
    int block_m)
{
    __shared__ float scores[FZR_MAX_E];
    __shared__ float sel[FZR_MAX_E];
    __shared__ float red_val[FZR_WAVES];
    __shared__ int red_idx[FZR_WAVES];
    __shared__ int chosen_idx[FZR_MAX_K];
    __shared__ float chosen_score[FZR_MAX_K];
    __shared__ float chosen_w[FZR_MAX_K];
    __shared__ float rot[FZR_WAVES][256];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    // ---- 1. top-k (ffn_moe_router_topk_f32, kind 1, row 0) ----
    for (int e = tid; e < n_experts; e += FZR_THREADS) {
        const float sg = 1.0f / (1.0f + expf(-logits[e]));
        scores[e] = sg;
        sel[e] = (bias != nullptr) ? (sg + bias[e]) : sg;
    }
    __syncthreads();
    for (int e = tid; e < n_experts; e += FZR_THREADS) {
        const float v = sel[e];
        sel[e] = (v >= FZR_LOWEST) ? v : FZR_LOWEST;
    }
    __syncthreads();
    for (int r = 0; r < top_k; ++r) {
        float best_v = -INFINITY;
        int best_i = 0x7fffffff;
        for (int e = tid; e < n_experts; e += FZR_THREADS) {
            const float v = sel[e];
            if (v > best_v) {
                best_v = v;
                best_i = e;
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor(best_v, off, 32);
            const int oi = __shfl_xor(best_i, off, 32);
            fzr_best(best_v, best_i, ov, oi);
        }
        if (lane == 0) {
            red_val[wave] = best_v;
            red_idx[wave] = best_i;
        }
        __syncthreads();
        if (wave == 0) {
            best_v = red_val[lane];
            best_i = red_idx[lane];
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_xor(best_v, off, 32);
                const int oi = __shfl_xor(best_i, off, 32);
                fzr_best(best_v, best_i, ov, oi);
            }
            if (lane == 0) {
                chosen_idx[r] = best_i;
                chosen_score[r] = scores[best_i];
                sel[best_i] = -INFINITY;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        float sum = 0.0f;
        for (int r = 0; r < top_k; ++r) sum += chosen_score[r];
        const float denom = sum + eps;
        for (int r = 0; r < top_k; ++r) chosen_w[r] = (chosen_score[r] / denom) * scale;
    }
    __syncthreads();
    // Grid [z]: every block computes the same top-k (same operations, same
    // winner); block 0 alone writes the routing outputs and tables, and the
    // phase-3 tasks are dealt over the blocks' waves (z == 1: the old order).
    const int zb = (int)blockIdx.x;
    const int zn = (int)gridDim.x;
    if (zb == 0 && tid < top_k) {
        idx_out[tid] = chosen_idx[tid];
        w_out[tid] = chosen_w[tid];
    }

    // ---- 2. scatter tables ----
    if (zb == 0) {
    for (int i = tid; i < m_total_max; i += FZR_THREADS) sorted_slot[i] = -1;
    for (int i = tid; i < m_total_max / block_m; i += FZR_THREADS) tile_ids[i] = -1;
    for (int e = tid; e < n_experts; e += FZR_THREADS) {
        int below = 0;
        bool hit = false;
        for (int r = 0; r < top_k; ++r) {
            below += chosen_idx[r] < e ? 1 : 0;
            hit = hit || chosen_idx[r] == e;
        }
        offsets[e] = below * block_m;
        counts[e] = hit ? block_m : 0;
    }
    if (tid == 0) offsets[n_experts] = top_k * block_m;
    __syncthreads();
    if (tid < top_k) {
        const int e = chosen_idx[tid];
        int rank = 0;
        for (int r = 0; r < top_k; ++r) rank += chosen_idx[r] < e ? 1 : 0;
        sorted_slot[rank * block_m] = tid;
        inverse_perm[tid] = rank * block_m;
        tile_ids[rank] = e;
    }
    }

    // ---- 3. slot producer + A8 sidecar ----
    const int groups = K / 256;
    for (int t = wave * zn + zb; t < top_k * groups; t += FZR_WAVES * zn) {
        const int s = t / groups;
        const int g = t - s * groups;
        const float* __restrict__ awq =
            reinterpret_cast<const float*>(scale_ptrs[chosen_idx[s]]);
        const float* xr = x_in + g * 256 + lane * 8;
        const float xv[8] = {xr[0], xr[1], xr[2], xr[3], xr[4], xr[5], xr[6], xr[7]};
        fz_awq_fwht(xv, awq + g * 256, rot[wave], signs1, signs2, 0, lane);
        fzr_wave_sync();
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const float* src = rot[wave] + b * 128 + lane * 4;
            const float qv[4] = {src[0], src[1], src[2], src[3]};
            quantize_block_i8_128_wave(qv, xq + (size_t)(2 * g + b) * top_k + s, lane);
        }
        fzr_wave_sync();
    }
}
"#
);

/// One segment of [`Gpu::inkling2_fz_gemv_segs`]: `y [m] = W [m, k] . x_rot [k]`
/// for an MQ4-G256-v2 `w` and an already AWQ+FWHT-rotated `x_rot`.
pub struct FzGemvSeg<'a> {
    pub w: &'a GpuTensor,
    pub x_rot: &'a GpuTensor,
    pub y: &'a GpuTensor,
    pub m: usize,
}

/// Operands of [`Gpu::inkling2_fz_route_row`]. Integer outputs are i32.
pub struct Inkling2FzRouteRow<'a> {
    /// Router logits `[n_experts]`.
    pub logits: &'a GpuTensor,
    pub correction_bias: Option<&'a GpuTensor>,
    pub eps: f32,
    pub scale: f32,
    /// `[top_k]` expert ids / weights.
    pub topk_idx: &'a GpuTensor,
    pub topk_w: &'a GpuTensor,
    /// Scatter tables: counts `[n_experts]`, offsets `[n_experts + 1]`,
    /// sorted slots `[m_total_max]`, tile ids `[m_total_max / block_m]`,
    /// inverse permutation `[top_k]`.
    pub counts: &'a GpuTensor,
    pub offsets: &'a GpuTensor,
    pub sorted: &'a GpuTensor,
    pub tiles: &'a GpuTensor,
    pub inverse: &'a GpuTensor,
    /// Normed FFN input row `[k]`.
    pub x: &'a GpuTensor,
    /// `[n_experts]` u64 device pointers to the per-expert gate/up AWQ scales.
    pub awq_ptrs: &'a GpuTensor,
    pub n_experts: usize,
    pub top_k: usize,
    pub k: usize,
    pub m_total_max: usize,
    pub block_m: usize,
}

impl Gpu {
    fn fz_row_ensure(&mut self, sym: &str) -> HipResult<()> {
        self.ensure_kernel(INKLING2_FUSE_ROW_MODULE, INKLING2_FUSE_ROW_HIP_SRC, sym)
    }

    /// The one-row GEMVs `segs` (common `k`, a multiple of 256) in one launch,
    /// each bitwise the singleton `gemv_mq4g256v2_multirow_r2` on its input.
    pub fn inkling2_fz_gemv_segs(&mut self, segs: &[FzGemvSeg<'_>], k: usize) -> HipResult<()> {
        self.bind_thread()?;
        let shape_ok = |s: &FzGemvSeg<'_>| {
            s.m > 0
                && s.w.buf.size() >= s.m * (k / 256) * 136
                && s.m * (k / 256) * 136 <= u32::MAX as usize
                && s.x_rot.buf.size() >= k * 4
                && s.y.buf.size() >= s.m * 4
        };
        if segs.is_empty()
            || segs.len() > FZ_GEMV_SEGS
            || k == 0
            || k % 256 != 0
            || k > i32::MAX as usize
            || !segs.iter().all(shape_ok)
        {
            return Err(HipError::new(0, "inkling2_fz_gemv_segs: operand shape"));
        }
        self.fz_row_ensure(INKLING2_FZ_GEMV_SEGS)?;
        let null = std::ptr::null::<c_void>();
        let mut b = KernargBlob::new();
        for i in 0..FZ_GEMV_SEGS {
            b.push_ptr(
                segs.get(i)
                    .map_or(null, |s| s.w.buf.as_ptr() as *const c_void),
            );
        }
        for i in 0..FZ_GEMV_SEGS {
            b.push_ptr(
                segs.get(i)
                    .map_or(null, |s| s.x_rot.buf.as_ptr() as *const c_void),
            );
        }
        for i in 0..FZ_GEMV_SEGS {
            b.push_ptr(
                segs.get(i)
                    .map_or(null, |s| s.y.buf.as_ptr() as *const c_void),
            );
        }
        for i in 0..FZ_GEMV_SEGS {
            b.push_i32(segs.get(i).map_or(0, |s| s.m as i32));
        }
        b.push_i32(k as i32);
        b.pad_to(16);
        let blocks = segs.iter().map(|s| s.m.div_ceil(2)).max().unwrap_or(0);
        let blocks =
            u32::try_from(blocks).map_err(|_| HipError::new(0, "inkling2_fz_gemv_segs: grid"))?;
        self.launch_blob_recorded(
            INKLING2_FZ_GEMV_SEGS,
            [blocks, 1, segs.len() as u32],
            [32, 1, 1],
            0,
            b.as_mut_slice(),
            ReplayLaunchBindings::NONE,
        )
    }

    /// One row: `y [m] = W . x_rot` (bitwise the singleton
    /// `gemv_mq4g256v2_multirow_r2`) and, in the same launch,
    /// `out = res + conv(y)` with the history shift, bitwise
    /// [`Gpu::inkling2_fz_conv_res`] at `rows == 1, state_update`.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_fz_gemv_conv_res(
        &mut self,
        w: &GpuTensor,
        m: usize,
        k: usize,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        static_w: &GpuTensor,
        state: &GpuTensor,
        state_stride: usize,
        col_off: usize,
        inv_scale: f32,
        res: Option<&GpuTensor>,
        out: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let f32_len = |t: &GpuTensor| t.buf.size() / 4;
        if m == 0
            || m > i32::MAX as usize
            || k == 0
            || k % 256 != 0
            || k > i32::MAX as usize
            || w.buf.size() < m * (k / 256) * 136
            || m * (k / 256) * 136 > u32::MAX as usize
            || f32_len(x_rot) < k
            || f32_len(y) < m
            || f32_len(out) < m
            || res.is_some_and(|r| f32_len(r) < m)
            || f32_len(static_w) < 4 * m
            || f32_len(state) < 2 * state_stride + col_off + m
        {
            return Err(HipError::new(0, "inkling2_fz_gemv_conv_res: operand shape"));
        }
        self.fz_row_ensure(INKLING2_FZ_GEMV_CONV_RES)?;
        let p = |t: &GpuTensor| t.buf.as_ptr() as *const c_void;
        let mut b = KernargBlob::new();
        b.push_ptr(p(w));
        b.push_ptr(p(x_rot));
        b.push_ptr(p(y));
        b.push_i32(m as i32);
        b.push_i32(k as i32);
        b.push_ptr(p(static_w));
        b.push_ptr(p(state));
        b.push_i32(state_stride as i32);
        b.push_i32(col_off as i32);
        b.push_f32(inv_scale);
        b.push_ptr(res.map_or(std::ptr::null(), p));
        b.push_ptr(p(out));
        b.pad_to(16);
        self.launch_blob_recorded(
            INKLING2_FZ_GEMV_CONV_RES,
            [m.div_ceil(2) as u32, 1, 1],
            [32, 1, 1],
            0,
            b.as_mut_slice(),
            ReplayLaunchBindings::NONE,
        )
    }

    /// One row's routing, scatter tables and gate/up A8 sidecar in one launch
    /// (module doc). Reserves the int8 MMQ sidecar for `top_k` rows of width
    /// `k` and returns it prepared; it stays valid until the next int8
    /// reservation or quantization.
    pub fn inkling2_fz_route_row(
        &mut self,
        a: &Inkling2FzRouteRow<'_>,
    ) -> HipResult<Int8MmqPrepared> {
        self.bind_thread()?;
        let f32_len = |t: &GpuTensor| t.buf.size() / 4;
        let (ne, tk, k) = (a.n_experts, a.top_k, a.k);
        if !self.arch_caps.is_gfx1201()
            || ne == 0
            || ne > FZ_ROUTE_MAX_E
            || tk == 0
            || tk > FZ_ROUTE_MAX_K
            || tk > ne
            || k == 0
            || k % 256 != 0
            || a.block_m == 0
            || a.m_total_max < tk * a.block_m
            || a.m_total_max % a.block_m != 0
            || f32_len(a.logits) < ne
            || a.correction_bias.is_some_and(|b| f32_len(b) < ne)
            || f32_len(a.topk_idx) < tk
            || f32_len(a.topk_w) < tk
            || f32_len(a.counts) < ne
            || f32_len(a.offsets) < ne + 1
            || f32_len(a.sorted) < a.m_total_max
            || f32_len(a.tiles) < a.m_total_max / a.block_m
            || f32_len(a.inverse) < tk
            || f32_len(a.x) < k
            || a.awq_ptrs.buf.size() < ne * 8
        {
            return Err(HipError::new(0, "inkling2_fz_route_row: operand shape"));
        }
        self.ensure_mq_signs()?;
        let (s1, s2) = match (
            self.scratch.mq_signs1.as_ref(),
            self.scratch.mq_signs2.as_ref(),
        ) {
            (Some(x), Some(y)) => (
                x.buf.as_ptr() as *const c_void,
                y.buf.as_ptr() as *const c_void,
            ),
            _ => return Err(HipError::new(0, "inkling2_fz_route_row: no mq signs")),
        };
        let res: Int8MmqReservation = self.reserve_int8_mmq(k, tk)?;
        self.fz_row_ensure(INKLING2_FZ_ROUTE_ROW)?;
        let p = |t: &GpuTensor| t.buf.as_ptr() as *const c_void;
        let mut b = KernargBlob::new();
        b.push_ptr(p(a.logits));
        b.push_ptr(a.correction_bias.map_or(std::ptr::null(), p));
        b.push_ptr(p(a.topk_idx));
        b.push_ptr(p(a.topk_w));
        b.push_ptr(p(a.counts));
        b.push_ptr(p(a.offsets));
        b.push_ptr(p(a.sorted));
        b.push_ptr(p(a.tiles));
        b.push_ptr(p(a.inverse));
        b.push_ptr(p(a.x));
        b.push_ptr(p(a.awq_ptrs));
        b.push_ptr(s1);
        b.push_ptr(s2);
        b.push_ptr(res.ptr() as *const c_void);
        b.push_i32(ne as i32);
        b.push_i32(tk as i32);
        b.push_f32(a.eps);
        b.push_f32(a.scale);
        b.push_i32(k as i32);
        b.push_i32(a.m_total_max as i32);
        b.push_i32(a.block_m as i32);
        b.pad_to(16);
        // One phase-3 task per wave: ceil(top_k * groups / 32) blocks.
        let blocks = (tk * (k / 256)).div_ceil(32) as u32;
        self.launch_blob_recorded(
            INKLING2_FZ_ROUTE_ROW,
            [blocks, 1, 1],
            [1024, 1, 1],
            0,
            b.as_mut_slice(),
            ReplayLaunchBindings::NONE,
        )?;
        Ok(Int8MmqPrepared::from_reservation(res))
    }
}
