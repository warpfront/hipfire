// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Blocked (multi-row) prompt prefill for Inkling-2-Mini (arch 24).
//!
//! [`forward_prefill_gpu`] runs `B` consecutive prompt rows through all 48
//! visits per launch sequence instead of one row at a time
//! ([`crate::gpu_forward::forward_token_gpu`]). Per visit, in the scalar order:
//! AttnRes mixes over the row banks, loop controllers (pass > 0), norms, the
//! dense projections as one GEMM over the block, the four short-conv streams
//! with causal history across rows, clips, sinks, KV write + causal attention,
//! the gate, the output projection and its conv, the residual, the FFN mix,
//! router, top-8 and the routed experts as grouped GEMMs, the combine and the
//! MoE conv. The LM head runs on the rows the caller asks for
//! ([`PrefillLogits`]); the product asks for the final row only.
//!
//! **What is row-exact.** Every row-independent op runs on kernels whose
//! per-row arithmetic equals the singleton decode kernels (mix, controllers,
//! norms, clips, conv, sinks, router F16 WMMA, top-k, SiTU, AWQ producers,
//! combine), so [`PrefillMode::ROW_EXACT`] (per-row GEMV linears, the indexed
//! expert chain and the decode attention per row) reproduces the scalar loop
//! bit for bit, state included. [`PrefillMode::BLOCKED`] changes three named
//! arithmetic sites: the dense linears (weight-only WMMA GEMM, X rounded to
//! F16), the routed experts (grouped weight-only WMMA) and the attention
//! (blocked causal FA2, hi/lo Q/P, no partition merge).
//!
//! **KV ring.** A block's keys/values are staged as F16 and the attention
//! reads pre-block keys from the cache and in-block keys from the staging
//! planes; the block is committed to the cache only after the attention, so
//! no early query loses a pre-block key to a later row's ring write.
//!
//! **Routes.** Stage 1 executes the F16 dense route and the grouped
//! weight-only experts; [`PrefillRoutes`] names the activation tier of each
//! site. Typed-XTS loads on gfx1201 only; anything else keeps the scalar loop.

use crate::arch_model::Inkling2Bundle;
use crate::config::Inkling2Config;
use crate::gpu::{
    swa_gather_map, swa_slot, Inkling2Controller, Inkling2GpuLayer, Inkling2GpuState,
    Inkling2GpuWeights, INKLING2_CONV_WIDTH,
};
use crate::gpu_forward::{ensure_kernels, launch_attn};
use hip_bridge::KernargBlob;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemm::{GemmFamily, GemmParams};
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::moe::{grouped_moe_m_total_bound, MOE_GROUPED_BLOCK_M};
use hipfire_dispatch::pipeline::{
    grouped_qt44_gemm_tier, run_uniform_moe_down_expanded, GroupedQt44Tables, PrefillActTier,
};
use hipfire_dispatch::types::KernelKey;
use hipfire_runtime::llama::{weight_gemm, weight_gemv, WeightTensor};
use rdna_compute::ffn_activation::{
    CombineOrder, FfnActivation, FfnActivationGrouped, FfnActivationRows, FfnAwq, MoeCombine,
    MoeRouterTopK, RouterKind,
};
use rdna_compute::inkling2_attention::inkling2_attn_fa2_splits;
use rdna_compute::inkling2_attention_prefill::{Inkling2PrefillAttn, INKLING2_ATTN_PREFILL_REL};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;
use std::sync::{Arc, LazyLock};

// ---------------------------------------------------------------------------
// Dedicated HIP (row-batched twins of the singleton kernels)
// ---------------------------------------------------------------------------

const INKLING2_PF_MODULE: &str = "inkling2_gpu_prefill";

const INKLING2_PF_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Short conv over `rows` consecutive rows, one thread per (row, channel).
// Row r convolves x = [x_{r-3}, x_{r-2}, x_{r-1}, x_r] where x_{<0} come from
// the visit history (state row 0 oldest). Per-row arithmetic is the singleton
// `inkling2_gf_conv_f32`. The history is read only here; ink_pf_conv_state_f32
// advances it afterwards.
extern "C" __global__ void ink_pf_conv_rows_f32(
    const float* __restrict__ cur, int cur_ld,
    const float* __restrict__ sw,
    const float* __restrict__ dyn_w, int dyn_ld,
    const float* __restrict__ state,
    float* __restrict__ out, int out_ld,
    int state_stride, int col_off, int channels,
    int dyn_heads, int head_dim, float inv_scale, int rows)
{
    const int ch = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int r = (int)blockIdx.y;
    if (ch >= channels || r >= rows) return;
    const float x0 = r >= 3 ? cur[(size_t)(r - 3) * cur_ld + ch]
                            : state[(size_t)r * state_stride + col_off + ch];
    const float x1 = r >= 2 ? cur[(size_t)(r - 2) * cur_ld + ch]
                            : state[(size_t)(r + 1) * state_stride + col_off + ch];
    const float x2 = r >= 1 ? cur[(size_t)(r - 1) * cur_ld + ch]
                            : state[(size_t)(r + 2) * state_stride + col_off + ch];
    const float xc = cur[(size_t)r * cur_ld + ch];
    float acc;
    if (dyn_heads == 0) {
        acc = x0 * sw[ch] + x1 * sw[channels + ch]
            + x2 * sw[channels * 2 + ch] + xc * sw[channels * 3 + ch];
    } else {
        const int h = ch / head_dim;
        const float* drow = dyn_w + (size_t)r * dyn_ld + (size_t)h * 4;
        acc = x0 * sw[ch] + x1 * sw[channels + ch]
            + x2 * sw[channels * 2 + ch] + xc * sw[channels * 3 + ch]
            + xc * drow[0] + x2 * drow[1] + x1 * drow[2] + x0 * drow[3];
    }
    out[(size_t)r * out_ld + ch] = (acc + xc) * inv_scale;
}

// History after `rows` more raw inputs: the last three of [hist0, hist1,
// hist2, cur_0 .. cur_{rows-1}].
extern "C" __global__ void ink_pf_conv_state_f32(
    const float* __restrict__ cur, int cur_ld,
    float* __restrict__ state,
    int state_stride, int col_off, int channels, int rows)
{
    const int ch = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (ch >= channels) return;
    float seq[3];
    for (int i = 0; i < 3; ++i) {
        const int j = rows + i;  // index into [hist0, hist1, hist2, cur...]
        seq[i] = j >= 3 ? cur[(size_t)(j - 3) * cur_ld + ch]
                        : state[(size_t)j * state_stride + col_off + ch];
    }
    state[col_off + ch] = seq[0];
    state[state_stride + col_off + ch] = seq[1];
    state[state_stride * 2 + col_off + ch] = seq[2];
}

// mix() over `rows` rows: block (row b, query q). States s are separate
// [rows, n_embd] tensors. Per row the arithmetic is the singleton pair
// `inkling2_rmsnorm_nogamma_f32` + `inkling2_mix_query_row_f32` (blockDim 256).
// out[(q * rows + b) * n_embd + i].
struct Ptr6 { const float* p[6]; };
extern "C" __global__ void ink_pf_mix_rows_f32(
    Ptr6 st, int n_states,
    const float* __restrict__ query,
    float* __restrict__ out,
    int rows, int n_embd, float inv_sqrt_embd, float eps)
{
    const int b = (int)blockIdx.x;
    const int q = (int)blockIdx.y;
    const int t = threadIdx.x;
    if (b >= rows) return;
    __shared__ float sh[256];
    __shared__ float w[6];
    float inv[6];
#pragma unroll
    for (int s = 0; s < 6; ++s) {
        inv[s] = 0.0f;
        if (s < n_states) {
            const float* xr = st.p[s] + (size_t)b * n_embd;
            float acc = 0.0f;
            for (int i = t; i < n_embd; i += blockDim.x) {
                const float v = xr[i];
                acc += v * v;
            }
            sh[t] = acc;
            __syncthreads();
            for (int s2 = blockDim.x >> 1; s2 > 0; s2 >>= 1) {
                if (t < s2) sh[t] += sh[t + s2];
                __syncthreads();
            }
            inv[s] = 1.0f / sqrtf(sh[0] / (float)n_embd + eps);
            __syncthreads();
        }
    }
    const float* qq = query + (size_t)q * n_embd;
#pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < n_states) {
            const float* xr = st.p[s] + (size_t)b * n_embd;
            float acc = 0.0f;
            for (int i = t; i < n_embd; i += blockDim.x) {
                const float ns = xr[i] * inv[s];
                acc += ns * qq[i];
            }
            sh[t] = acc;
            __syncthreads();
            for (int s2 = blockDim.x >> 1; s2 > 0; s2 >>= 1) {
                if (t < s2) sh[t] += sh[t + s2];
                __syncthreads();
            }
            if (t == 0) w[s] = sh[0] * inv_sqrt_embd;
            __syncthreads();
        }
    }
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
    float* oq = out + ((size_t)q * rows + b) * n_embd;
    for (int i = t; i < n_embd; i += blockDim.x) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < 6; ++s) {
            if (s < n_states) {
                const float ns = st.p[s][(size_t)b * n_embd + i] * inv[s];
                acc += w[s] * ns;
            }
        }
        oq[i] = acc;
    }
}

// `inkling2_l2_clip_rows_f32` with the per-channel scale indexed by
// `r % scale_rows` (one [heads, dim] scale shared by every block row).
extern "C" __global__ void ink_pf_l2_clip_rows_f32(
    float* __restrict__ rows_buf,
    const float* __restrict__ scale,
    int n_rows, int dim, int scale_rows,
    float threshold, int has_scale)
{
    const int r = blockIdx.x;
    if (r >= n_rows) return;
    const int t = threadIdx.x;
    extern __shared__ float sh[];
    float v = rows_buf[r * dim + t];
    if (has_scale) v *= scale[(r % scale_rows) * dim + t];
    sh[t] = v * v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    const float norm = sqrtf(sh[0]) + 1e-6f;
    float factor = threshold / norm;
    if (factor > 1.0f) factor = 1.0f;
    rows_buf[r * dim + t] = v * factor;
}

// `inkling2_rmsnorm_nogamma_f32` with row strides (controller gate halves).
extern "C" __global__ void ink_pf_rmsnorm_nogamma_ld_f32(
    const float* __restrict__ x, int x_ld,
    float* __restrict__ out, int out_ld,
    int batch, int n, float eps)
{
    const int r = blockIdx.x;
    if (r >= batch) return;
    const float* xr = x + (size_t)r * x_ld;
    float* orow = out + (size_t)r * out_ld;
    const int t = threadIdx.x;
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

// `gemv_f32` (y = A x, A [M, K] F32) for up to 8 activation rows per block:
// block (m, row group); every row keeps the singleton lane-strided K loop and
// shared-memory tree of blockDim = min(256, K), so each y is bitwise the
// singleton's. A is read once per 8 rows.
#define PF_GR 8
extern "C" __global__ void ink_pf_gemv_f32_rows(
    const float* __restrict__ A,
    const float* __restrict__ x, int x_ld,
    float* __restrict__ y, int y_ld,
    int M, int K, int rows)
{
    extern __shared__ float sdata[];
    const int m = blockIdx.x;
    const int r0 = blockIdx.y * PF_GR;
    const int t = threadIdx.x;
    const int bs = blockDim.x;
    if (m >= M) return;
    float sum[PF_GR];
#pragma unroll
    for (int rr = 0; rr < PF_GR; ++rr) sum[rr] = 0.0f;
    for (int k = t; k < K; k += bs) {
        const float a = A[m * K + k];
#pragma unroll
        for (int rr = 0; rr < PF_GR; ++rr) {
            if (r0 + rr < rows) sum[rr] += a * x[(size_t)(r0 + rr) * x_ld + k];
        }
    }
#pragma unroll
    for (int rr = 0; rr < PF_GR; ++rr) sdata[rr * bs + t] = sum[rr];
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (t < s) {
#pragma unroll
            for (int rr = 0; rr < PF_GR; ++rr) sdata[rr * bs + t] += sdata[rr * bs + t + s];
        }
        __syncthreads();
    }
    if (t < PF_GR && r0 + t < rows) y[(size_t)(r0 + t) * y_ld + m] = sdata[t * bs];
}

// Block K/V F32 [rows, 8, 128] -> F16 staging planes [8, rows, 128]
// (the singleton cache write's `(_Float16)` conversion).
extern "C" __global__ void ink_pf_kv_stage_f16(
    const float* __restrict__ k, const float* __restrict__ v,
    _Float16* __restrict__ ks, _Float16* __restrict__ vs, int rows)
{
    const int idx = (int)(blockIdx.x * 256 + threadIdx.x);
    if (idx >= rows * 1024) return;
    const int r = idx >> 10;
    const int h = (idx >> 7) & 7;
    const int d = idx & 127;
    const size_t o = ((size_t)h * rows + r) * 128 + d;
    ks[o] = (_Float16)k[idx];
    vs[o] = (_Float16)v[idx];
}

// Commit staged rows [first_row, rows) to the cache ring (slot = pos % ring).
struct Planes8 { unsigned short* p[8]; };
extern "C" __global__ void ink_pf_kv_commit_f16(
    const unsigned short* __restrict__ ks, const unsigned short* __restrict__ vs,
    Planes8 kc, Planes8 vc,
    int rows, int first_row, int pos0, int ring_rows)
{
    const int idx = (int)(blockIdx.x * 256 + threadIdx.x);
    if (idx >= (rows - first_row) * 1024) return;
    const int r = first_row + (idx >> 10);
    const int h = (idx >> 7) & 7;
    const int d = idx & 127;
    const size_t src = ((size_t)h * rows + r) * 128 + d;
    const size_t dst = (size_t)((pos0 + r) % ring_rows) * 128 + d;
    kc.p[h][dst] = ks[src];
    vc.p[h][dst] = vs[src];
}

// `rotate_x_mq_awq_topk` over `slots` (token, rank) rows: row s reads token
// row s / top_k of x_in and expert ids[s]. Same arithmetic per row.
extern "C" __launch_bounds__(32, 16)
__global__ void ink_pf_rotate_awq_slots(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const unsigned long long* __restrict__ scale_ptrs,
    const int* __restrict__ expert_ids,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int K, int n_exp, int top_k)
{
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    if (groups_total <= 0) return;
    const unsigned bx = blockIdx.x;
    const int slot = (int)(bx / (unsigned)groups_total);
    const int group = (int)(bx - (unsigned)slot * (unsigned)groups_total);
    const int expert_id = expert_ids[slot];
    const float* __restrict__ xr = x_in + (long long)(slot / top_k) * K;
    float* x_out_b = x_out + (long long)slot * K;
    const int d0 = tid * 8;
    const int base = group * 256 + d0;
    if (expert_id < 0 || expert_id >= n_exp) {
#pragma unroll
        for (int e = 0; e < 8; ++e) x_out_b[base + e] = 0.0f;
        return;
    }
    const float* __restrict__ awq_scale =
        reinterpret_cast<const float*>(scale_ptrs[expert_id]);
    float v0 = (xr[base    ] / awq_scale[base    ]) * signs1[d0    ];
    float v1 = (xr[base + 1] / awq_scale[base + 1]) * signs1[d0 + 1];
    float v2 = (xr[base + 2] / awq_scale[base + 2]) * signs1[d0 + 2];
    float v3 = (xr[base + 3] / awq_scale[base + 3]) * signs1[d0 + 3];
    float v4 = (xr[base + 4] / awq_scale[base + 4]) * signs1[d0 + 4];
    float v5 = (xr[base + 5] / awq_scale[base + 5]) * signs1[d0 + 5];
    float v6 = (xr[base + 6] / awq_scale[base + 6]) * signs1[d0 + 6];
    float v7 = (xr[base + 7] / awq_scale[base + 7]) * signs1[d0 + 7];
    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;
    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;
    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;
    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while(0)
    #define HBFLY8(pat, str) \
        HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
        HBFLY(v4,pat,str); HBFLY(v5,pat,str); HBFLY(v6,pat,str); HBFLY(v7,pat,str)
    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8
    #undef HBFLY
    const float s = 0.0625f;
    x_out_b[base    ] = v0 * s * signs2[d0    ];
    x_out_b[base + 1] = v1 * s * signs2[d0 + 1];
    x_out_b[base + 2] = v2 * s * signs2[d0 + 2];
    x_out_b[base + 3] = v3 * s * signs2[d0 + 3];
    x_out_b[base + 4] = v4 * s * signs2[d0 + 4];
    x_out_b[base + 5] = v5 * s * signs2[d0 + 5];
    x_out_b[base + 6] = v6 * s * signs2[d0 + 6];
    x_out_b[base + 7] = v7 * s * signs2[d0 + 7];
}
"#;

const PF_SYMBOLS: [&str; 9] = [
    "ink_pf_conv_rows_f32",
    "ink_pf_conv_state_f32",
    "ink_pf_mix_rows_f32",
    "ink_pf_l2_clip_rows_f32",
    "ink_pf_rmsnorm_nogamma_ld_f32",
    "ink_pf_gemv_f32_rows",
    "ink_pf_kv_stage_f16",
    "ink_pf_kv_commit_f16",
    "ink_pf_rotate_awq_slots",
];

fn ensure_pf_kernels(gpu: &mut Gpu) -> Result<(), String> {
    for sym in PF_SYMBOLS {
        gpu.ensure_kernel_public(INKLING2_PF_MODULE, INKLING2_PF_HIP_SRC, sym)
            .map_err(|e| format!("inkling2 prefill: kernel {sym}: {e:?}"))?;
    }
    Ok(())
}

fn launch(
    gpu: &mut Gpu,
    sym: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    mut blob: KernargBlob,
) -> Result<(), String> {
    blob.pad_to(16);
    gpu.launch_kernel_blob(sym, grid, block, shared, blob.as_mut_slice())
        .map_err(|e| format!("inkling2 prefill: {sym}: {e:?}"))
}

fn hip<T>(r: Result<T, hip_bridge::HipError>, what: &str) -> Result<T, String> {
    r.map_err(|e| format!("inkling2 prefill: {what}: {e:?}"))
}

fn ptr(t: &GpuTensor) -> *const c_void {
    t.buf.as_ptr() as *const c_void
}

fn grid_u32(n: usize, what: &str) -> Result<u32, String> {
    u32::try_from(n).map_err(|_| format!("inkling2 prefill: {what} grid {n} exceeds u32"))
}

// ---------------------------------------------------------------------------
// Public contract
// ---------------------------------------------------------------------------

/// Default rows per prefill block (one launch sequence through the 48 visits).
pub const PREFILL_BLOCK_ROWS: usize = 512;

/// Which rows of a prefill call produce logits.
pub enum PrefillLogits<'a> {
    /// None (intermediate chunks).
    None,
    /// The final row only (the product; same head arithmetic as decode).
    Last(&'a mut Vec<f32>),
    /// Every row at position `>= from`, delivered in order as
    /// `(position, unpadded logits)`. Evidence only (scoring).
    Rows {
        from: usize,
        sink: &'a mut dyn FnMut(usize, &[f32]) -> Result<(), String>,
    },
}

/// Arithmetic of a prefill call, per site. Every other op is row-exact in both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillMode {
    /// Dense projections and the multi-row LM head: one GEMM per block, or the
    /// singleton GEMV per row.
    pub gemm: bool,
    /// Routed experts: grouped GEMMs, or the indexed decode chain over rows.
    pub grouped: bool,
    /// Attention: blocked causal FA2, or the decode attention per row.
    pub blocked_attn: bool,
}

impl PrefillMode {
    /// The product: GEMM linears, grouped experts, blocked FA2 attention.
    pub const BLOCKED: Self = Self {
        gemm: true,
        grouped: true,
        blocked_attn: true,
    };
    /// Row-batched glue with per-row GEMV linears, the indexed expert chain and
    /// the decode attention per row: bitwise the scalar loop (oracle mode).
    pub const ROW_EXACT: Self = Self {
        gemm: false,
        grouped: false,
        blocked_attn: false,
    };
}

/// Activation tier of each prefill GEMM site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillRoutes {
    /// Q/K/V/Rel/gate/O projections.
    pub dense: PrefillActTier,
    /// Routed expert gate_up and down.
    pub experts: PrefillActTier,
}

impl PrefillRoutes {
    /// Stage 1: weight-only F16 everywhere.
    pub const F16: Self = Self {
        dense: PrefillActTier::F16,
        experts: PrefillActTier::F16,
    };
}

/// Whether [`forward_prefill_gpu`] runs the blocked path for this load: gfx1201,
/// typed-XTS layout (symmetric qt44 + AWQ sidecars, F16 router), Mini geometry.
pub fn prefill_supported(bundle: &Inkling2Bundle, gpu: &Gpu) -> bool {
    let cfg = &bundle.config;
    let Some(gw) = bundle.gpu_weights.as_ref() else {
        return false;
    };
    gpu.arch == "gfx1201"
        && bundle.gpu_state.is_some()
        && gw.xts_routes.is_some()
        && !gw.layers.is_empty()
        && gw
            .layers
            .iter()
            .all(|l| l.xts.is_some() && l.router.gpu_dtype == DType::F16)
        && cfg.n_head == 24
        && cfg.n_head_kv == 8
        && cfg.head_dim == 128
        && cfg.n_rel == INKLING2_ATTN_PREFILL_REL
        && cfg.conv_kernel == 4
        && cfg.n_expert_used == 8
        && cfg.n_embd % 256 == 0
}

/// Prefill `ids` at positions `start_pos..start_pos + ids.len()` (must equal
/// the session cursor). Success advances every visit's KV/conv state and the
/// cursor by exactly `ids.len()`. Blocked path when [`prefill_supported`],
/// otherwise the scalar loop (one [`crate::forward_token_gpu`] per row).
pub fn forward_prefill_gpu(
    bundle: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    ids: &[u32],
    start_pos: usize,
    logits: PrefillLogits<'_>,
) -> Result<(), String> {
    if !prefill_supported(bundle, gpu) {
        return scalar_prefill(bundle, gpu, ids, start_pos, logits);
    }
    let routes = bundle
        .gpu_state
        .as_ref()
        .map(|s| s.prefill_routes)
        .unwrap_or(PrefillRoutes::F16);
    forward_prefill_gpu_with(
        bundle,
        gpu,
        ids,
        start_pos,
        logits,
        PrefillMode::BLOCKED,
        routes,
        PREFILL_BLOCK_ROWS,
        None,
    )
}

/// The scalar loop behind the same contract.
fn scalar_prefill(
    bundle: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    ids: &[u32],
    start_pos: usize,
    mut logits: PrefillLogits<'_>,
) -> Result<(), String> {
    for (i, &tok) in ids.iter().enumerate() {
        let pos = start_pos + i;
        let row = crate::gpu_forward::forward_token_gpu(bundle, gpu, tok, pos)?;
        match &mut logits {
            PrefillLogits::None => {}
            PrefillLogits::Last(out) => {
                if i + 1 == ids.len() {
                    **out = row;
                }
            }
            PrefillLogits::Rows { from, sink } => {
                if pos >= *from {
                    sink(pos, &row)?;
                }
            }
        }
    }
    Ok(())
}

/// Evidence entry: explicit mode, routes and block size; `trace` receives
/// every visit's `visit_out` rows (`[visit][row * n_embd]`, all rows of the call).
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_gpu_with(
    bundle: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    ids: &[u32],
    start_pos: usize,
    mut logits: PrefillLogits<'_>,
    mode: PrefillMode,
    routes: PrefillRoutes,
    block_rows: usize,
    mut trace: Option<&mut Vec<Vec<f32>>>,
) -> Result<(), String> {
    if ids.is_empty() {
        return Ok(());
    }
    if block_rows == 0 {
        return Err("inkling2 prefill: block_rows must be > 0".into());
    }
    if (mode.gemm || mode.grouped || mode.blocked_attn) && gpu.arch != "gfx1201" {
        return Err(format!("inkling2 prefill: blocked sites are gfx1201 only, not {}", gpu.arch));
    }
    let cfg = bundle.config.clone();
    let gw = bundle
        .gpu_weights
        .as_ref()
        .ok_or("inkling2 prefill: no gpu weights")?;
    let gs = bundle
        .gpu_state
        .as_mut()
        .ok_or("inkling2 prefill: no gpu state")?;
    if gw.layers.iter().any(|l| l.xts.is_none()) {
        return Err("inkling2 prefill: needs the typed-XTS layout".into());
    }
    if start_pos != gs.seq_pos {
        return Err(format!(
            "inkling2 prefill: start_pos {start_pos} != seq_pos {} (out-of-order)",
            gs.seq_pos
        ));
    }
    if let Some(t) = trace.as_mut() {
        t.clear();
        t.resize(cfg.loop_cfg.n_slot(), Vec::new());
    }
    ensure_kernels(gpu)?;
    ensure_pf_kernels(gpu)?;
    let saved = pin_dense_route(gpu, routes.dense);
    let r = prefill_blocks(gpu, &cfg, gw, gs, ids, start_pos, &mut logits, mode, routes, block_rows, trace);
    gpu.flags = saved;
    r
}

#[allow(clippy::too_many_arguments)]
fn prefill_blocks(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    gw: &Inkling2GpuWeights,
    gs: &mut Inkling2GpuState,
    ids: &[u32],
    start_pos: usize,
    logits: &mut PrefillLogits<'_>,
    mode: PrefillMode,
    routes: PrefillRoutes,
    block_rows: usize,
    mut trace: Option<&mut Vec<Vec<f32>>>,
) -> Result<(), String> {
    let n = ids.len();
    let mut done = 0usize;
    while done < n {
        let take = block_rows.min(n - done);
        let block = &ids[done..done + take];
        let p = start_pos + done;
        let last_block = done + take == n;
        let want = match &*logits {
            PrefillLogits::None => HeadRows::None,
            PrefillLogits::Last(_) if last_block => HeadRows::Last,
            PrefillLogits::Last(_) => HeadRows::None,
            PrefillLogits::Rows { from, .. } => {
                if p + take > *from {
                    HeadRows::From(from.saturating_sub(p))
                } else {
                    HeadRows::None
                }
            }
        };
        let ctx = BlockCtx {
            cfg,
            gw,
            mode,
            routes,
            rows: take,
            p,
        };
        let mut o = Owned::default();
        let r = drive_block(gpu, &ctx, gs, block, &mut o, want, trace.as_deref_mut());
        o.free(gpu);
        let heads = r?;
        match logits {
            PrefillLogits::None => {}
            PrefillLogits::Last(out) => {
                if let Some((_, row)) = heads.into_iter().last() {
                    **out = row;
                }
            }
            PrefillLogits::Rows { sink, .. } => {
                for (pos, row) in heads {
                    sink(pos, &row)?;
                }
            }
        }
        done += take;
    }
    Ok(())
}

/// Pin the dense activation arm of `dense` for the prefill: the gfx12 qt44
/// GEMM ladder picks it from `gpu.flags` (A8 > IU4 > FP8 > F16). Returns the
/// caller's flags, which the prefill restores on every exit.
fn pin_dense_route(
    gpu: &mut Gpu,
    dense: PrefillActTier,
) -> Arc<rdna_compute::feature_flags::FeatureFlags> {
    let saved = gpu.flags.clone();
    let mut f = (*saved).clone();
    f.a8_prefill = dense == PrefillActTier::A8;
    f.gfx11_a8_prefill = false;
    f.iu4_prefill = Some(dense == PrefillActTier::A4);
    f.gfx12_mq4v2_fp8_resid = dense == PrefillActTier::Fp8;
    gpu.flags = Arc::new(f);
    saved
}

// ---------------------------------------------------------------------------
// Arena: visit temporaries, freed on every path
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct T(usize);

#[derive(Default)]
struct Arena {
    ts: Vec<Option<GpuTensor>>,
}

impl Arena {
    fn alloc(&mut self, gpu: &mut Gpu, shape: &[usize], dtype: DType, what: &str) -> Result<T, String> {
        let t = gpu
            .alloc_tensor(shape, dtype)
            .map_err(|e| format!("inkling2 prefill: alloc {what}: {e:?}"))?;
        self.ts.push(Some(t));
        Ok(T(self.ts.len() - 1))
    }
    fn f32(&mut self, gpu: &mut Gpu, n: usize, what: &str) -> Result<T, String> {
        self.alloc(gpu, &[n], DType::F32, what)
    }
    fn push(&mut self, t: GpuTensor) -> T {
        self.ts.push(Some(t));
        T(self.ts.len() - 1)
    }
    fn get(&self, i: T) -> &GpuTensor {
        self.ts[i.0].as_ref().expect("inkling2 prefill: arena slot taken")
    }
    fn take(&mut self, i: T) -> GpuTensor {
        self.ts[i.0].take().expect("inkling2 prefill: arena slot taken")
    }
    fn free(self, gpu: &mut Gpu) {
        for t in self.ts.into_iter().flatten() {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Non-owning F32 view `[off, off + len)` (elements) with `shape`.
fn view(t: &GpuTensor, off: usize, shape: &[usize]) -> Result<GpuTensor, String> {
    let len: usize = shape.iter().product();
    let (b0, bl) = (off * 4, len * 4);
    if b0 + bl > t.buf.size() {
        return Err(format!(
            "inkling2 prefill: view [{b0},{}) of {} bytes",
            b0 + bl,
            t.buf.size()
        ));
    }
    Ok(GpuTensor {
        buf: t.buf.byte_view(b0, bl),
        shape: shape.to_vec(),
        dtype: t.dtype,
    })
}

/// Fresh `[n]` copy of the first `n` words of `src`.
fn dup(gpu: &mut Gpu, src: &GpuTensor, n: usize, what: &str) -> Result<GpuTensor, String> {
    if src.buf.size() < n * 4 {
        return Err(format!("inkling2 prefill: dup {what}: short source"));
    }
    let out = hip(gpu.alloc_tensor(&[n], DType::F32), what)?;
    if let Err(e) = gpu.copy_rows_strided_f32(&view(src, 0, &[n])?, &out, 1, n, n, n, 0, None) {
        let _ = gpu.free_tensor(out);
        return Err(format!("inkling2 prefill: dup {what}: {e:?}"));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Row-batched primitives
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn conv_rows(
    gpu: &mut Gpu,
    cur: &GpuTensor,
    static_w: &GpuTensor,
    dyn_w: Option<(&GpuTensor, usize)>,
    state: &GpuTensor,
    out: &GpuTensor,
    col_off: usize,
    channels: usize,
    dyn_heads: usize,
    head_dim: usize,
    inv_scale: f32,
    rows: usize,
) -> Result<(), String> {
    let mut b = KernargBlob::new();
    b.push_ptr(ptr(cur));
    b.push_i32(channels as i32);
    b.push_ptr(ptr(static_w));
    match dyn_w {
        Some((t, ld)) => {
            b.push_ptr(ptr(t));
            b.push_i32(ld as i32);
        }
        None => {
            b.push_ptr(std::ptr::null());
            b.push_i32(0);
        }
    }
    b.push_ptr(ptr(state));
    b.push_ptr(ptr(out));
    b.push_i32(channels as i32);
    b.push_i32(INKLING2_CONV_WIDTH as i32);
    b.push_i32(col_off as i32);
    b.push_i32(channels as i32);
    b.push_i32(dyn_heads as i32);
    b.push_i32(head_dim as i32);
    b.push_f32(inv_scale);
    b.push_i32(rows as i32);
    let gx = grid_u32(channels.div_ceil(256), "conv")?;
    launch(gpu, "ink_pf_conv_rows_f32", [gx, grid_u32(rows, "conv")?, 1], [256, 1, 1], 0, b)?;
    let mut s = KernargBlob::new();
    s.push_ptr(ptr(cur));
    s.push_i32(channels as i32);
    s.push_ptr(ptr(state));
    s.push_i32(INKLING2_CONV_WIDTH as i32);
    s.push_i32(col_off as i32);
    s.push_i32(channels as i32);
    s.push_i32(rows as i32);
    launch(gpu, "ink_pf_conv_state_f32", [gx, 1, 1], [256, 1, 1], 0, s)
}

/// mix() of `rows` rows: `states` (oldest first, `cur` last) are `[rows, e]`;
/// `query [qr, e]`; `out [qr, rows, e]`.
#[allow(clippy::too_many_arguments)]
fn mix_rows(
    gpu: &mut Gpu,
    states: &[&GpuTensor],
    query: &GpuTensor,
    out: &GpuTensor,
    rows: usize,
    e: usize,
    qr: usize,
    eps: f32,
) -> Result<(), String> {
    if !(1..=6).contains(&states.len()) {
        return Err(format!("inkling2 prefill: mix states {} exceeds 6", states.len()));
    }
    if query.buf.size() < qr * e * 4 || out.buf.size() < qr * rows * e * 4 {
        return Err("inkling2 prefill: mix operand too small".into());
    }
    let mut b = KernargBlob::new();
    for i in 0..6 {
        b.push_ptr(states.get(i).map(|t| ptr(t)).unwrap_or(std::ptr::null()));
    }
    b.push_i32(states.len() as i32);
    b.push_ptr(ptr(query));
    b.push_ptr(ptr(out));
    b.push_i32(rows as i32);
    b.push_i32(e as i32);
    b.push_f32(1.0 / (e as f32).sqrt());
    b.push_f32(eps);
    launch(
        gpu,
        "ink_pf_mix_rows_f32",
        [grid_u32(rows, "mix")?, qr as u32, 1],
        [256, 1, 1],
        0,
        b,
    )
}

/// `y [rows, M] = x [rows, K] . A^T`, singleton `gemv_f32` arithmetic per row.
#[allow(clippy::too_many_arguments)]
fn gemv_rows(
    gpu: &mut Gpu,
    a: &GpuTensor,
    x: &GpuTensor,
    x_ld: usize,
    y: &GpuTensor,
    y_ld: usize,
    m: usize,
    k: usize,
    rows: usize,
) -> Result<(), String> {
    if a.buf.size() < m * k * 4 {
        return Err("inkling2 prefill: gemv weight too small".into());
    }
    let bs = 256usize.min(k);
    let mut b = KernargBlob::new();
    b.push_ptr(ptr(a));
    b.push_ptr(ptr(x));
    b.push_i32(x_ld as i32);
    b.push_ptr(ptr(y));
    b.push_i32(y_ld as i32);
    b.push_i32(m as i32);
    b.push_i32(k as i32);
    b.push_i32(rows as i32);
    launch(
        gpu,
        "ink_pf_gemv_f32_rows",
        [grid_u32(m, "gemv")?, grid_u32(rows.div_ceil(8), "gemv")?, 1],
        [bs as u32, 1, 1],
        (8 * bs * 4) as u32,
        b,
    )
}

#[allow(clippy::too_many_arguments)]
fn clip_rows(
    gpu: &mut Gpu,
    rows_buf: &GpuTensor,
    scale: Option<&GpuTensor>,
    n_rows: usize,
    dim: usize,
    scale_rows: usize,
    threshold: f32,
) -> Result<(), String> {
    let mut b = KernargBlob::new();
    b.push_ptr(ptr(rows_buf));
    b.push_ptr(scale.map(ptr).unwrap_or(std::ptr::null()));
    b.push_i32(n_rows as i32);
    b.push_i32(dim as i32);
    b.push_i32(scale_rows.max(1) as i32);
    b.push_f32(threshold);
    b.push_i32(scale.is_some() as i32);
    launch(
        gpu,
        "ink_pf_l2_clip_rows_f32",
        [grid_u32(n_rows, "clip")?, 1, 1],
        [dim as u32, 1, 1],
        (dim * 4) as u32,
        b,
    )
}

/// One fused controller step over `rows` rows (singleton
/// `inkling2_control_f32_gpu` arithmetic per row). `out [rows, e]`.
#[allow(clippy::too_many_arguments)]
fn control_rows(
    gpu: &mut Gpu,
    ar: &mut Arena,
    cur: &GpuTensor,
    msg: &GpuTensor,
    ctrl: &Inkling2Controller,
    e: usize,
    rank: usize,
    rows: usize,
) -> Result<T, String> {
    let eps = rdna_compute::inkling2_control::INKLING2_CONTROL_EPS;
    let gate = ar.f32(gpu, rows * 2 * e, "ctrl_gate")?;
    let g = ar.f32(gpu, rows * rank, "ctrl_g")?;
    let gain = ar.f32(gpu, rows * e, "ctrl_gain")?;
    let forget = ar.f32(gpu, rows * e, "ctrl_forget")?;
    let out = ar.f32(gpu, rows * e, "ctrl_out")?;
    for (half, x) in [(0usize, cur), (1, msg)] {
        let mut b = KernargBlob::new();
        b.push_ptr(ptr(x));
        b.push_i32(e as i32);
        b.push_ptr(ar.get(gate).buf.as_ptr().wrapping_byte_add(half * e * 4) as *const c_void);
        b.push_i32((2 * e) as i32);
        b.push_i32(rows as i32);
        b.push_i32(e as i32);
        b.push_f32(eps);
        launch(
            gpu,
            "ink_pf_rmsnorm_nogamma_ld_f32",
            [grid_u32(rows, "ctrl norm")?, 1, 1],
            [256, 1, 1],
            0,
            b,
        )?;
    }
    gemv_rows(gpu, &ctrl.down, ar.get(gate), 2 * e, ar.get(g), rank, rank, 2 * e, rows)?;
    gemv_rows(gpu, &ctrl.gain, ar.get(g), rank, ar.get(gain), e, e, rank, rows)?;
    gemv_rows(gpu, &ctrl.forget, ar.get(g), rank, ar.get(forget), e, e, rank, rows)?;
    hip(
        gpu.inkling2_control_blend_gpu_f32(
            cur,
            msg,
            ar.get(gain),
            ar.get(forget),
            ar.get(out),
            rows * e,
        ),
        "control blend",
    )?;
    Ok(out)
}

static GEMM_FAMILY: LazyLock<GemmFamily> = LazyLock::new(GemmFamily::new);

/// Dense qt44 projection over `rows` rows: one GEMM (blocked) or the singleton
/// GEMV per row (row-exact).
fn linear(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    gemm: bool,
) -> Result<(), String> {
    if gemm && rows > 1 {
        return hip(weight_gemm(gpu, w, x, y, rows), "gemm");
    }
    for r in 0..rows {
        let xr = view(x, r * w.k, &[w.k])?;
        let yr = view(y, r * w.m, &[w.m])?;
        hip(weight_gemv(gpu, w, &xr, &yr), "gemv")?;
    }
    Ok(())
}

/// F16 router `[n_experts, e]` over `rows` rows: the gfx12 F16 WMMA (`mb8`)
/// chain the singleton router runs, one column per row (bitwise per row).
fn router_rows(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    gemm: bool,
) -> Result<(), String> {
    if !gemm || w.gpu_dtype != DType::F16 {
        return linear(gpu, w, x, y, rows, false);
    }
    let ctx = DispatchCtx::new(gpu);
    let wr = WeightRef {
        buf: &w.buf,
        dtype: w.gpu_dtype,
        m: w.m,
        k: w.k,
        row_stride: 0,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    GEMM_FAMILY
        .run_key(
            KernelKey::GemmF16WmmaMb8,
            &ctx,
            gpu,
            &GemmParams {
                w: &wr,
                x,
                y,
                batch_size: rows,
            },
        )
        .map_err(|e| format!("inkling2 prefill: router gemm: {e}"))
}

// ---------------------------------------------------------------------------
// One visit over a block of rows
// ---------------------------------------------------------------------------

struct BlockCtx<'a> {
    cfg: &'a Inkling2Config,
    gw: &'a Inkling2GpuWeights,
    mode: PrefillMode,
    routes: PrefillRoutes,
    rows: usize,
    p: usize,
}

struct VisitOut {
    visit_out: GpuTensor,
    push: Option<GpuTensor>,
}

type Msgs<'a> = (
    &'a [GpuTensor],
    &'a GpuTensor,
    &'a GpuTensor,
    &'a [Inkling2Controller; 4],
);

#[allow(clippy::too_many_arguments)]
fn run_visit_rows(
    gpu: &mut Gpu,
    c: &BlockCtx<'_>,
    layer: &Inkling2GpuLayer,
    x: &GpuTensor,
    blocks: &[GpuTensor],
    visit: usize,
    phys: usize,
    msgs: Option<Msgs<'_>>,
    gs: &mut Inkling2GpuState,
) -> Result<VisitOut, String> {
    let mut ar = Arena::default();
    match visit_body(gpu, c, layer, x, blocks, visit, phys, msgs, gs, &mut ar) {
        Ok((vo, push)) => {
            let visit_out = ar.take(vo);
            let push = push.map(|p| ar.take(p));
            ar.free(gpu);
            Ok(VisitOut { visit_out, push })
        }
        Err(e) => {
            ar.free(gpu);
            Err(e)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn visit_body(
    gpu: &mut Gpu,
    c: &BlockCtx<'_>,
    layer: &Inkling2GpuLayer,
    x: &GpuTensor,
    blocks: &[GpuTensor],
    visit: usize,
    phys: usize,
    msgs: Option<Msgs<'_>>,
    gs: &mut Inkling2GpuState,
    ar: &mut Arena,
) -> Result<(T, Option<T>), String> {
    let cfg = c.cfg;
    let n = c.rows;
    let p = c.p;
    let e = cfg.n_embd;
    let nh = cfg.n_head;
    let nkv = cfg.n_head_kv;
    let hd = cfg.head_dim;
    let ekv = nkv * hd;
    let eattn = nh * hd;
    let erel = nh * cfg.n_rel;
    let eps = cfg.rms_norm_eps;
    let swa = cfg.swa_layers[phys];
    let ne = n * e;

    // --- attention input: AttnRes mix (visit 0 / phys 0 takes x itself) ---
    let bases: Option<[GpuTensor; 3]> = if phys == 0 {
        None
    } else {
        let aq = layer
            .attn_query
            .as_ref()
            .ok_or_else(|| format!("inkling2 prefill: visit {visit} phys {phys} missing attn_query"))?;
        let total = aq.buf.size() / 4;
        if total % e != 0 {
            return Err(format!("inkling2 prefill: attn_query len {total} not a multiple of {e}"));
        }
        let qr = if total / e >= 3 { 3 } else { 1 };
        let mixed = ar.f32(gpu, qr * ne, "attn_mix")?;
        let states: Vec<&GpuTensor> = blocks.iter().chain(std::iter::once(x)).collect();
        mix_rows(gpu, &states, aq, ar.get(mixed), n, e, qr, eps)?;
        let m = ar.get(mixed);
        Some(if qr >= 3 {
            [view(m, 0, &[n, e])?, view(m, ne, &[n, e])?, view(m, 2 * ne, &[n, e])?]
        } else {
            [view(m, 0, &[n, e])?, view(m, 0, &[n, e])?, view(m, 0, &[n, e])?]
        })
    };
    // Loop messages (pass > 0): control each input.
    let mut mids: Option<T> = None;
    let (qc0, kc0, vc0): (GpuTensor, GpuTensor, GpuTensor) = match (&msgs, &bases) {
        (Some((bank, prev_out, cross, ctrls)), Some([bq, bk, bv])) => {
            if cross.buf.size() / 4 < 4 * e {
                return Err(format!("inkling2 prefill: cross len {} < 4*{e}", cross.buf.size() / 4));
            }
            let msg = ar.f32(gpu, 4 * ne, "msg_mix")?;
            let states: Vec<&GpuTensor> = bank.iter().chain(std::iter::once(*prev_out)).collect();
            mix_rows(gpu, &states, cross, ar.get(msg), n, e, 4, eps)?;
            mids = Some(msg);
            let rank = cfg.loop_cfg.gate_rank;
            let mut outs = Vec::with_capacity(3);
            for (j, base) in [bq, bk, bv].into_iter().enumerate() {
                let mj = view(ar.get(msg), j * ne, &[n, e])?;
                let o = control_rows(gpu, ar, base, &mj, &ctrls[j], e, rank, n)?;
                outs.push(view(ar.get(o), 0, &[n, e])?);
            }
            let v = outs.pop().unwrap();
            let k = outs.pop().unwrap();
            let q = outs.pop().unwrap();
            (q, k, v)
        }
        (Some(_), None) => return Err("inkling2 prefill: phys 0 has no base".into()),
        (None, None) => (view(x, 0, &[n, e])?, view(x, 0, &[n, e])?, view(x, 0, &[n, e])?),
        (None, Some([bq, bk, bv])) => (bq.clone_view(), bk.clone_view(), bv.clone_view()),
    };
    // Norms: SWA applies attn_norm to all three inputs; full layers the three
    // input norms. Identical (input, gamma) pairs are normalized once.
    let (wq_n, wk_n, wv_n) = if swa {
        let an = layer
            .attn_norm
            .as_ref()
            .ok_or("inkling2 prefill: SWA layer missing attn_norm")?;
        (an, an, an)
    } else {
        match (&layer.q_input_norm, &layer.k_input_norm, &layer.v_input_norm) {
            (Some(a), Some(b), Some(cc)) => (a, b, cc),
            _ => return Err("inkling2 prefill: full layer missing input norms".into()),
        }
    };
    let mut normed: Vec<(*mut c_void, *mut c_void, T)> = Vec::new();
    let mut norm_of = |gpu: &mut Gpu, ar: &mut Arena, src: &GpuTensor, w: &GpuTensor| -> Result<T, String> {
        let key = (src.buf.as_ptr(), w.buf.as_ptr());
        if let Some(&(_, _, t)) = normed.iter().find(|(a, b, _)| (*a, *b) == key) {
            return Ok(t);
        }
        let out = ar.f32(gpu, ne, "inp_norm")?;
        hip(gpu.rmsnorm_batched(src, w, ar.get(out), n, e, eps), "input norm")?;
        normed.push((key.0, key.1, out));
        Ok(out)
    };
    let qi = norm_of(gpu, ar, &qc0, wq_n)?;
    let ki = norm_of(gpu, ar, &kc0, wk_n)?;
    let vi = norm_of(gpu, ar, &vc0, wv_n)?;
    // Projections.
    let q = ar.f32(gpu, n * eattn, "q")?;
    let k_raw = ar.f32(gpu, n * ekv, "k_raw")?;
    let v_raw = ar.f32(gpu, n * ekv, "v_raw")?;
    let r = ar.f32(gpu, n * erel, "r")?;
    let gemm = c.mode.gemm;
    linear(gpu, &layer.wq, ar.get(qi), ar.get(q), n, gemm)?;
    linear(gpu, &layer.wk, ar.get(ki), ar.get(k_raw), n, gemm)?;
    linear(gpu, &layer.wv, ar.get(vi), ar.get(v_raw), n, gemm)?;
    linear(gpu, &layer.wr, ar.get(qi), ar.get(r), n, gemm)?;
    // Convs (static on SWA; dynamic taps on full layers).
    let k_conv = ar.f32(gpu, n * ekv, "k_conv")?;
    let v_conv = ar.f32(gpu, n * ekv, "v_conv")?;
    let conv_state = &gs.conv[visit];
    if swa {
        let s = 1.0 / 5.0f32.sqrt();
        conv_rows(gpu, ar.get(k_raw), &layer.k_conv, None, conv_state, ar.get(k_conv), 0, ekv, 0, hd, s, n)?;
        conv_rows(gpu, ar.get(v_raw), &layer.v_conv, None, conv_state, ar.get(v_conv), ekv, ekv, 0, hd, s, n)?;
    } else {
        let (kp, vp) = match (&layer.k_proj, &layer.v_proj) {
            (Some(a), Some(b)) => (a, b),
            _ => return Err("inkling2 prefill: full layer missing k/v proj".into()),
        };
        let dw = nkv * cfg.conv_kernel;
        let dk = ar.f32(gpu, n * dw, "dyn_k")?;
        let dv = ar.f32(gpu, n * dw, "dyn_v")?;
        gemv_rows(gpu, kp, ar.get(ki), e, ar.get(dk), dw, dw, e, n)?;
        gemv_rows(gpu, vp, ar.get(vi), e, ar.get(dv), dw, dw, e, n)?;
        let s = 1.0 / 3.0;
        conv_rows(gpu, ar.get(k_raw), &layer.k_conv, Some((ar.get(dk), dw)), conv_state, ar.get(k_conv), 0, ekv, nkv, hd, s, n)?;
        conv_rows(gpu, ar.get(v_raw), &layer.v_conv, Some((ar.get(dv), dw)), conv_state, ar.get(v_conv), ekv, ekv, nkv, hd, s, n)?;
    }
    // Clips + V head-norm.
    clip_rows(gpu, ar.get(q), Some(&layer.q_clip), n * nh, hd, nh, cfg.qk_clip)?;
    clip_rows(gpu, ar.get(k_conv), None, n * nkv, hd, 1, cfg.qk_clip)?;
    clip_rows(gpu, ar.get(r), Some(&layer.rel_clip), n * nh, cfg.n_rel, nh, cfg.rel_clip)?;
    let vc = ar.f32(gpu, n * ekv, "vc")?;
    hip(
        gpu.inkling2_rmsnorm_nogamma_f32(ar.get(v_conv), ar.get(vc), n * nkv, hd, eps),
        "v norm",
    )?;
    // Full-layer tau: exactly 1.0 below `log_floor`; fail closed beyond.
    if !swa && !cfg.gpu_full_tau_supported(p + n - 1) {
        return Err(format!(
            "inkling2 prefill: full-layer tau {} at pos {} unimplemented",
            cfg.full_attention_tau(p + n - 1),
            p + n - 1
        ));
    }
    // Sinks `[n, nh]`.
    let sinks = ar.f32(gpu, n * nh, "sinks")?;
    gemv_rows(gpu, &layer.sink, ar.get(qi), e, ar.get(sinks), nh, nh, e, n)?;
    // KV + attention.
    let ring = gs.visit_rows[visit];
    if !swa && p + n > ring {
        return Err(format!(
            "inkling2 prefill: visit {visit} KV overflow at pos {} (rows {ring})",
            p + n - 1
        ));
    }
    let attn_heads = ar.f32(gpu, n * eattn, "attn_heads")?;
    if c.mode.blocked_attn {
        attn_blocked(gpu, cfg, layer, gs, ar, visit, swa, p, n, q, r, sinks, k_conv, vc, attn_heads)?;
    } else {
        attn_row_exact(gpu, cfg, layer, gs, ar, visit, swa, p, n, q, r, sinks, k_conv, vc, attn_heads)?;
    }
    // Output gate + projection + post conv.
    let gate = ar.f32(gpu, n * eattn, "gate")?;
    linear(gpu, &layer.wgate, ar.get(qi), ar.get(gate), n, gemm)?;
    hip(gpu.sigmoid_f32(ar.get(gate)), "gate sigmoid")?;
    hip(gpu.mul_f32(ar.get(attn_heads), ar.get(gate), ar.get(attn_heads)), "gate mul")?;
    let wo_out = ar.f32(gpu, ne, "wo_out")?;
    linear(gpu, &layer.wo, ar.get(attn_heads), ar.get(wo_out), n, gemm)?;
    let attn_res = ar.f32(gpu, ne, "attn_res")?;
    conv_rows(
        gpu,
        ar.get(wo_out),
        &layer.attn_post_conv,
        None,
        &gs.conv[visit],
        ar.get(attn_res),
        2 * ekv,
        e,
        0,
        hd,
        1.0 / 5.0f32.sqrt(),
        n,
    )?;
    // AttnRes bookkeeping: first visit of a block pushes x, others add.
    let has_push = phys % cfg.attn_res_block == 0;
    let push = if has_push {
        let t = dup(gpu, x, ne, "block push")?;
        Some(ar.push(t))
    } else {
        None
    };
    let cur = if has_push {
        attn_res
    } else {
        let cur = ar.f32(gpu, ne, "cur")?;
        hip(gpu.add_f32(&view(x, 0, &[ne])?, ar.get(attn_res), ar.get(cur)), "attn residual")?;
        cur
    };
    // --- MoE: mix over blocks + push + cur, control (pass > 0), norm ---
    if layer.mlp_query.buf.size() / 4 != e {
        return Err(format!("inkling2 prefill: mlp_query len {} != {e}", layer.mlp_query.buf.size() / 4));
    }
    let ffn_mixed = ar.f32(gpu, ne, "ffn_mix")?;
    {
        let mut states: Vec<&GpuTensor> = blocks.iter().collect();
        if let Some(pi) = push {
            states.push(ar.get(pi));
        }
        states.push(ar.get(cur));
        mix_rows(gpu, &states, &layer.mlp_query, ar.get(ffn_mixed), n, e, 1, eps)?;
    }
    let ffn_in = match (&msgs, mids) {
        (Some((_, _, _, ctrls)), Some(m)) => {
            let m3 = view(ar.get(m), 3 * ne, &[n, e])?;
            let fm = view(ar.get(ffn_mixed), 0, &[n, e])?;
            control_rows(gpu, ar, &fm, &m3, &ctrls[3], e, cfg.loop_cfg.gate_rank, n)?
        }
        _ => ffn_mixed,
    };
    let ffn_normed = ar.f32(gpu, ne, "ffn_normed")?;
    hip(
        gpu.rmsnorm_batched(&view(ar.get(ffn_in), 0, &[n, e])?, &layer.mlp_norm, ar.get(ffn_normed), n, e, eps),
        "ffn norm",
    )?;
    // Router + top-8.
    let n_exp = cfg.n_experts;
    let k_top = cfg.n_expert_used;
    let router_logits = ar.f32(gpu, n * n_exp, "router_logits")?;
    router_rows(gpu, &layer.router, &view(ar.get(ffn_normed), 0, &[n, e])?, ar.get(router_logits), n, gemm)?;
    let topk_idx = ar.f32(gpu, n * k_top, "topk_idx")?;
    let topk_w = ar.f32(gpu, n * k_top, "topk_w")?;
    hip(
        gpu.ffn_moe_router_topk_f32(&MoeRouterTopK {
            kind: RouterKind::SigmoidBiasRenorm {
                eps: 1e-20,
                scale: cfg.route_scale,
            },
            logits: ar.get(router_logits),
            correction_bias: layer.router_bias.as_ref(),
            topk_indices: ar.get(topk_idx),
            topk_weights: ar.get(topk_w),
            rows: n,
            n_experts: n_exp,
            top_k: k_top,
        }),
        "router topk",
    )?;
    let moe_raw = ar.f32(gpu, ne, "moe_raw")?;
    experts_rows(gpu, c, layer, ar, ffn_normed, topk_idx, topk_w, moe_raw)?;
    let moe_conv = ar.f32(gpu, ne, "moe_conv")?;
    conv_rows(
        gpu,
        ar.get(moe_raw),
        &layer.mlp_post_conv,
        None,
        &gs.conv[visit],
        ar.get(moe_conv),
        2 * ekv + e,
        e,
        0,
        hd,
        1.0 / 5.0f32.sqrt(),
        n,
    )?;
    let visit_out = ar.f32(gpu, ne, "visit_out")?;
    hip(gpu.add_f32(ar.get(cur), ar.get(moe_conv), ar.get(visit_out)), "moe residual")?;
    Ok((visit_out, push))
}

trait CloneView {
    fn clone_view(&self) -> GpuTensor;
}

impl CloneView for GpuTensor {
    fn clone_view(&self) -> GpuTensor {
        GpuTensor {
            buf: self.buf.byte_view(0, self.buf.size()),
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }
}

/// Row-exact attention: the singleton KV write, gather map, relative bias and
/// decode attention, one row at a time in position order.
#[allow(clippy::too_many_arguments)]
fn attn_row_exact(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    layer: &Inkling2GpuLayer,
    gs: &mut Inkling2GpuState,
    ar: &mut Arena,
    visit: usize,
    swa: bool,
    p: usize,
    n: usize,
    q: T,
    r: T,
    sinks: T,
    k_conv: T,
    vc: T,
    attn_heads: T,
) -> Result<(), String> {
    let nh = cfg.n_head;
    let nkv = cfg.n_head_kv;
    let hd = cfg.head_dim;
    let ekv = nkv * hd;
    let eattn = nh * hd;
    let erel = nh * cfg.n_rel;
    let rows = gs.visit_rows[visit];
    for i in 0..n {
        let pos = p + i;
        let wpos = if swa { swa_slot(pos, rows) } else { pos };
        gs.visit_pos_tags[visit][wpos] = pos as u32;
        let (slots, dists) = swa_gather_map(&gs.visit_pos_tags[visit], pos, rows)?;
        let n_kv = slots.len();
        let dist_bytes: Vec<u8> = dists.iter().flat_map(|d| d.to_le_bytes()).collect();
        let dist = hip(gpu.upload_raw(&dist_bytes, &[dist_bytes.len()]), "dist")?;
        let di = ar.push(dist);
        let slot_bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
        let slot_buf = hip(gpu.upload_raw(&slot_bytes, &[slot_bytes.len()]), "slots")?;
        let si = ar.push(slot_buf);
        let rel_bias = ar.alloc(gpu, &[nh, n_kv], DType::F32, "rel_bias")?;
        let rv = view(ar.get(r), i * erel, &[nh, cfg.n_rel])?;
        let sv = view(ar.get(sinks), i * nh, &[nh])?;
        hip(
            gpu.inkling2_rel_bias_gather_f32(
                &rv,
                &layer.rel_table,
                &sv,
                ar.get(di),
                ar.get(rel_bias),
                cfg.n_rel,
                nh,
                cfg.n_extent,
                n_kv,
            ),
            "rel_bias",
        )?;
        for h in 0..nkv {
            let src_k = view(ar.get(k_conv), i * ekv + h * hd, &[hd])?;
            let dst_k = gs.k_cache[visit][h].sub_offset(wpos * hd, hd);
            hip(gpu.convert_f32_to_f16_into(&src_k, &dst_k, hd), "k cache write")?;
            let src_v = view(ar.get(vc), i * ekv + h * hd, &[hd])?;
            let dst_v = gs.v_cache[visit][h].sub_offset(wpos * hd, hd);
            hip(gpu.convert_f32_to_f16_into(&src_v, &dst_v, hd), "v cache write")?;
        }
        gs.visit_cap[visit] = pos + 1;
        let qv = view(ar.get(q), i * eattn, &[nh, hd])?;
        let out = view(ar.get(attn_heads), i * eattn, &[eattn])?;
        if gs.attn_fa2 {
            let splits = inkling2_attn_fa2_splits(n_kv);
            let lse = ar.f32(gpu, splits * nh, "fa2_lse")?;
            let oacc = ar.f32(gpu, splits * nh * hd, "fa2_oacc")?;
            hip(
                gpu.inkling2_attn_decode_fa2(
                    &qv,
                    &gs.k_cache[visit],
                    &gs.v_cache[visit],
                    ar.get(rel_bias),
                    ar.get(si),
                    ar.get(lse),
                    ar.get(oacc),
                    &out,
                    n_kv,
                    splits,
                ),
                "attn fa2",
            )?;
        } else {
            launch_attn(
                gpu,
                &qv,
                &gs.k_cache[visit],
                &gs.v_cache[visit],
                ar.get(rel_bias),
                ar.get(si),
                &out,
                n_kv,
                nh,
                hd,
            )?;
        }
    }
    Ok(())
}

/// Blocked attention: stage the block's K/V as F16, run the causal FA2 prefill
/// over cache (pre-block) + staging (in-block) keys, then commit the block to
/// the cache ring and its position tags.
#[allow(clippy::too_many_arguments)]
fn attn_blocked(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    layer: &Inkling2GpuLayer,
    gs: &mut Inkling2GpuState,
    ar: &mut Arena,
    visit: usize,
    swa: bool,
    p: usize,
    n: usize,
    q: T,
    r: T,
    sinks: T,
    k_conv: T,
    vc: T,
    attn_heads: T,
) -> Result<(), String> {
    let nkv = cfg.n_head_kv;
    let hd = cfg.head_dim;
    let ring = gs.visit_rows[visit];
    let window = if swa { ring } else { 0 };
    // Pre-block keys the first query reads must carry their true positions.
    let lo = if swa { (p + 1).saturating_sub(ring) } else { 0 };
    let tags = &gs.visit_pos_tags[visit];
    for j in lo..p {
        let slot = swa_slot(j, ring);
        if tags[slot] as usize != j {
            return Err(format!(
                "inkling2 prefill: visit {visit} tag mismatch at slot {slot}: tag {} != key_pos {j}",
                tags[slot]
            ));
        }
    }
    let plane = nkv * n * hd;
    let ks = ar.alloc(gpu, &[plane], DType::F16, "k_stage")?;
    let vs = ar.alloc(gpu, &[plane], DType::F16, "v_stage")?;
    let mut b = KernargBlob::new();
    b.push_ptr(ptr(ar.get(k_conv)));
    b.push_ptr(ptr(ar.get(vc)));
    b.push_ptr(ptr(ar.get(ks)));
    b.push_ptr(ptr(ar.get(vs)));
    b.push_i32(n as i32);
    launch(
        gpu,
        "ink_pf_kv_stage_f16",
        [grid_u32((n * nkv * hd).div_ceil(256), "kv stage")?, 1, 1],
        [256, 1, 1],
        0,
        b,
    )?;
    hip(
        gpu.inkling2_attn_prefill_fa2(&Inkling2PrefillAttn {
            q: ar.get(q),
            rel: ar.get(r),
            sinks: ar.get(sinks),
            table: &layer.rel_table,
            k_cache: &gs.k_cache[visit],
            v_cache: &gs.v_cache[visit],
            k_stage: ar.get(ks),
            v_stage: ar.get(vs),
            out: ar.get(attn_heads),
            rows: n,
            pos0: p,
            window,
            ring_rows: ring,
            n_extent: cfg.n_extent,
        }),
        "attn prefill",
    )?;
    // Commit: only the trailing `ring` rows survive a block wider than the ring.
    let first = n.saturating_sub(ring);
    let mut b = KernargBlob::new();
    b.push_ptr(ptr(ar.get(ks)));
    b.push_ptr(ptr(ar.get(vs)));
    for h in 0..nkv {
        b.push_ptr(ptr(&gs.k_cache[visit][h]));
    }
    for h in 0..nkv {
        b.push_ptr(ptr(&gs.v_cache[visit][h]));
    }
    b.push_i32(n as i32);
    b.push_i32(first as i32);
    b.push_i32(p as i32);
    b.push_i32(ring as i32);
    launch(
        gpu,
        "ink_pf_kv_commit_f16",
        [grid_u32(((n - first) * nkv * hd).div_ceil(256), "kv commit")?, 1, 1],
        [256, 1, 1],
        0,
        b,
    )?;
    for i in first..n {
        let pos = p + i;
        gs.visit_pos_tags[visit][swa_slot(pos, ring)] = pos as u32;
    }
    gs.visit_cap[visit] = p + n;
    Ok(())
}

/// Routed experts over the block: per-slot AWQ+FWHT producer, then the grouped
/// GEMMs at the route's tier (blocked) or the indexed decode chain batched over
/// rows (row-exact), SiTU with the per-expert down AWQ, rank-order combine.
#[allow(clippy::too_many_arguments)]
fn experts_rows(
    gpu: &mut Gpu,
    c: &BlockCtx<'_>,
    layer: &Inkling2GpuLayer,
    ar: &mut Arena,
    ffn_normed: T,
    topk_idx: T,
    topk_w: T,
    moe_raw: T,
) -> Result<(), String> {
    let cfg = c.cfg;
    let xl = layer
        .xts
        .as_ref()
        .ok_or("inkling2 prefill: layer without the typed-XTS tables")?;
    let n = c.rows;
    let e = cfg.n_embd;
    let inter = cfg.inter;
    let phys = crate::weights::xts_down_dims(cfg).1;
    let k_top = cfg.n_expert_used;
    let n_exp = cfg.n_experts;
    let slots = n * k_top;
    // Per-slot producer: row s = FWHT256(ffn_normed[s / 8] / S_gu[ids[s]]).
    let x_rot = ar.alloc(gpu, &[slots, e], DType::F32, "xts_x_rot")?;
    hip(gpu.ensure_mq_signs(), "mq signs")?;
    let s1 = gpu.scratch.mq_signs1.as_ref().ok_or("inkling2 prefill: no mq signs")?.buf.as_ptr();
    let s2 = gpu.scratch.mq_signs2.as_ref().ok_or("inkling2 prefill: no mq signs")?.buf.as_ptr();
    let mut b = KernargBlob::new();
    b.push_ptr(ptr(ar.get(ffn_normed)));
    b.push_ptr(ptr(ar.get(x_rot)));
    b.push_ptr(ptr(&xl.gu_awq_ptrs));
    b.push_ptr(ptr(ar.get(topk_idx)));
    b.push_ptr(s1 as *const c_void);
    b.push_ptr(s2 as *const c_void);
    b.push_i32(e as i32);
    b.push_i32(n_exp as i32);
    b.push_i32(k_top as i32);
    launch(
        gpu,
        "ink_pf_rotate_awq_slots",
        [grid_u32((e / 256) * slots, "awq slots")?, 1, 1],
        [32, 1, 1],
        0,
        b,
    )?;
    gpu.scratch.invalidate_x_caches_for(ar.get(x_rot).buf.as_ptr());
    let situ = FfnActivation::Situ {
        gate_beta: cfg.situ_beta,
        value_beta: cfg.situ_linear_beta,
    };
    match c.mode.grouped {
        false => {
            let gu = ar.f32(gpu, slots * 2 * inter, "xts_gate_up")?;
            run_uniform_moe_down_expanded(
                gpu,
                DType::MQ4G256V2,
                &xl.gu_ptrs,
                ar.get(topk_idx),
                ar.get(x_rot),
                ar.get(gu),
                2 * inter,
                e,
                k_top,
                n,
            )
            .map_err(|e2| format!("inkling2 prefill: indexed gate_up: {e2:?}"))?;
            let act = ar.f32(gpu, slots * phys, "xts_act")?;
            hip(
                gpu.ffn_activation_rows_f32(&FfnActivationRows {
                    activation: situ,
                    gate: ar.get(gu),
                    gate_offset: 0,
                    gate_ld: 2 * inter,
                    up: ar.get(gu),
                    up_offset: inter,
                    up_ld: 2 * inter,
                    out: ar.get(act),
                    rows: slots,
                    logical_n: inter,
                    phys_n: phys,
                    round_inputs: false,
                    round_output: false,
                    awq: FfnAwq::PerExpert {
                        table: &xl.down_awq_ptrs,
                        row_expert: ar.get(topk_idx),
                        n_experts: n_exp,
                    },
                }),
                "situ",
            )?;
            let rot = ar.f32(gpu, slots * phys, "xts_rot")?;
            hip(gpu.rotate_x_mq_batched(ar.get(act), ar.get(rot), phys, slots), "act rotate")?;
            let down = ar.f32(gpu, slots * e, "xts_down")?;
            run_uniform_moe_down_expanded(
                gpu,
                DType::MQ4G256V2,
                &xl.down_ptrs,
                ar.get(topk_idx),
                ar.get(rot),
                ar.get(down),
                e,
                phys,
                k_top,
                n,
            )
            .map_err(|e2| format!("inkling2 prefill: indexed down: {e2:?}"))?;
            combine(gpu, cfg, ar, down, None, topk_idx, topk_w, moe_raw, n)
        }
        true => {
            let tier = c.routes.experts;
            let m_total = grouped_moe_m_total_bound(slots, n_exp);
            let counts = ar.f32(gpu, n_exp, "moe_counts")?;
            let offsets = ar.f32(gpu, n_exp + 1, "moe_offsets")?;
            let sorted = ar.f32(gpu, m_total, "moe_sorted")?;
            let tiles = ar.f32(gpu, m_total / MOE_GROUPED_BLOCK_M, "moe_tiles")?;
            let inv = ar.f32(gpu, slots, "moe_inv")?;
            hip(
                gpu.moe_scatter_fused_k8(
                    ar.get(topk_idx),
                    ar.get(counts),
                    ar.get(offsets),
                    ar.get(sorted),
                    ar.get(tiles),
                    ar.get(inv),
                    slots,
                    n_exp,
                    m_total,
                    MOE_GROUPED_BLOCK_M,
                ),
                "moe scatter",
            )?;
            let y_gu = ar.f32(gpu, m_total * 2 * inter, "moe_y_gu")?;
            let gu_tables = GroupedQt44Tables {
                expert_weight_ptrs: &xl.gu_ptrs,
                expert_tile_ids: ar.get(tiles),
                expert_offsets: ar.get(offsets),
                sorted_slot_index: ar.get(sorted),
            };
            grouped_qt44_gemm_tier(
                gpu,
                tier,
                &gu_tables,
                ar.get(x_rot),
                ar.get(y_gu),
                2 * inter,
                e,
                1,
                m_total,
                slots,
            )
            .map_err(|e2| format!("inkling2 prefill: grouped gate_up: {e2}"))?;
            let act = ar.alloc(gpu, &[slots, phys], DType::F32, "moe_act")?;
            hip(
                gpu.ffn_activation_grouped_f32(&FfnActivationGrouped {
                    activation: situ,
                    y_grouped: ar.get(y_gu),
                    sorted_slot_index: ar.get(sorted),
                    expert_tile_ids: ar.get(tiles),
                    out: ar.get(act),
                    m_total,
                    block_m: MOE_GROUPED_BLOCK_M,
                    logical_n: inter,
                    phys_n: phys,
                    round_inputs: false,
                    round_output: false,
                    awq_table: Some(&xl.down_awq_ptrs),
                    n_experts: n_exp,
                }),
                "grouped situ",
            )?;
            let rot = ar.alloc(gpu, &[slots, phys], DType::F32, "moe_rot")?;
            hip(gpu.rotate_x_mq_batched(ar.get(act), ar.get(rot), phys, slots), "act rotate")?;
            let y_dn = ar.f32(gpu, m_total * e, "moe_y_dn")?;
            let dn_tables = GroupedQt44Tables {
                expert_weight_ptrs: &xl.down_ptrs,
                expert_tile_ids: ar.get(tiles),
                expert_offsets: ar.get(offsets),
                sorted_slot_index: ar.get(sorted),
            };
            grouped_qt44_gemm_tier(
                gpu,
                tier,
                &dn_tables,
                ar.get(rot),
                ar.get(y_dn),
                e,
                phys,
                1,
                m_total,
                slots,
            )
            .map_err(|e2| format!("inkling2 prefill: grouped down: {e2}"))?;
            combine(gpu, cfg, ar, y_dn, Some(inv), topk_idx, topk_w, moe_raw, n)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn combine(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    ar: &mut Arena,
    down_rows: T,
    row_of_slot: Option<T>,
    topk_idx: T,
    topk_w: T,
    out: T,
    rows: usize,
) -> Result<(), String> {
    hip(
        gpu.ffn_moe_combine_f32(&MoeCombine {
            order: CombineOrder::Rank,
            round_expert_outputs: false,
            round_routed_sum: false,
            round_final: false,
            down_rows: ar.get(down_rows),
            row_of_slot: row_of_slot.map(|t| ar.get(t)),
            topk_indices: ar.get(topk_idx),
            topk_weights: ar.get(topk_w),
            shared: None,
            routed_out: None,
            out: ar.get(out),
            rows,
            hidden: cfg.n_embd,
            top_k: cfg.n_expert_used,
            n_experts: cfg.n_experts,
        }),
        "combine",
    )
}

// ---------------------------------------------------------------------------
// Block driver (mirrors gpu_forward::drive_body over [rows, e] streams)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Owned {
    x: Option<GpuTensor>,
    blocks: Vec<GpuTensor>,
    prelude_x: Option<GpuTensor>,
    prelude_blocks: Vec<GpuTensor>,
    prev_blocks: Vec<GpuTensor>,
    prev_out: Option<GpuTensor>,
}

impl Owned {
    fn free(self, gpu: &mut Gpu) {
        for t in self
            .x
            .into_iter()
            .chain(self.blocks)
            .chain(self.prelude_x)
            .chain(self.prelude_blocks)
            .chain(self.prev_blocks)
            .chain(self.prev_out)
        {
            let _ = gpu.free_tensor(t);
        }
    }
}

#[derive(Clone, Copy)]
enum HeadRows {
    None,
    Last,
    /// Rows `>= from` (block-relative).
    From(usize),
}

#[allow(clippy::too_many_arguments)]
fn step(
    gpu: &mut Gpu,
    c: &BlockCtx<'_>,
    gs: &mut Inkling2GpuState,
    o: &mut Owned,
    visit: usize,
    core_i: Option<usize>,
    trace: &mut Option<&mut Vec<Vec<f32>>>,
) -> Result<(), String> {
    let cfg = c.cfg;
    let gw = c.gw;
    let phys = cfg.loop_cfg.layer_phys(visit);
    let x = o.x.take().ok_or("inkling2 prefill: missing x")?;
    let msgs = match core_i {
        Some(i) => {
            let skip = o.prelude_blocks.len() + 1;
            if o.prev_blocks.len() < skip {
                let _ = gpu.free_tensor(x);
                return Err(format!(
                    "inkling2 prefill: pass prev_blocks {} < skip {skip}",
                    o.prev_blocks.len()
                ));
            }
            let Some(ctrls) = gw.layers[phys].controllers.as_ref() else {
                let _ = gpu.free_tensor(x);
                return Err(format!("inkling2 prefill: core layer {phys} missing controllers"));
            };
            Some((
                &o.prev_blocks[skip..],
                o.prev_out.as_ref().expect("prev_out"),
                &gw.cross_pass[i],
                ctrls,
            ))
        }
        None => None,
    };
    let vo = match run_visit_rows(gpu, c, &gw.layers[phys], &x, &o.blocks, visit, phys, msgs, gs) {
        Ok(v) => v,
        Err(e) => {
            let _ = gpu.free_tensor(x);
            return Err(e);
        }
    };
    let _ = gpu.free_tensor(x);
    if let Some(p) = vo.push {
        o.blocks.push(p);
    }
    if let Some(t) = trace.as_mut() {
        let row = hip(gpu.download_f32(&view(&vo.visit_out, 0, &[c.rows * cfg.n_embd])?), "trace")?;
        t[visit].extend_from_slice(&row);
    }
    o.x = Some(vo.visit_out);
    Ok(())
}

/// One block: embed, 48 visits, then the head on the requested rows. Returns
/// `(position, unpadded logits)` per head row; advances `gs.seq_pos`.
fn drive_block(
    gpu: &mut Gpu,
    c: &BlockCtx<'_>,
    gs: &mut Inkling2GpuState,
    ids: &[u32],
    o: &mut Owned,
    head: HeadRows,
    mut trace: Option<&mut Vec<Vec<f32>>>,
) -> Result<Vec<(usize, Vec<f32>)>, String> {
    let cfg = c.cfg;
    let gw = c.gw;
    let n = c.rows;
    let e = cfg.n_embd;
    let ne = n * e;
    if gw.layers.len() != cfg.n_phys {
        return Err(format!("inkling2 prefill: {} layers != n_phys {}", gw.layers.len(), cfg.n_phys));
    }
    let core_len = cfg.loop_cfg.core_end - cfg.loop_cfg.core_start;
    if gw.cross_pass.len() != core_len {
        return Err(format!("inkling2 prefill: {} cross_pass != core_len {core_len}", gw.cross_pass.len()));
    }
    // Embed: host row decode, one upload, batched token RMSNorm.
    let mut rows_h = Vec::with_capacity(ne);
    for &id in ids {
        rows_h.extend_from_slice(&gw.embed_packed.row_f32(id)?);
    }
    let emb = hip(gpu.upload_f32(&rows_h, &[n, e]), "embed upload")?;
    let tok = match gpu.alloc_tensor(&[ne], DType::F32) {
        Ok(t) => t,
        Err(e2) => {
            let _ = gpu.free_tensor(emb);
            return Err(format!("inkling2 prefill: tok alloc: {e2:?}"));
        }
    };
    let r = gpu.rmsnorm_batched(&emb, &gw.embed_norm, &tok, n, e, cfg.rms_norm_eps);
    let _ = gpu.free_tensor(emb);
    o.x = Some(tok);
    hip(r, "embed norm")?;
    let mut visit = 0usize;
    for _ in 0..cfg.loop_cfg.core_start {
        step(gpu, c, gs, o, visit, None, &mut trace)?;
        visit += 1;
    }
    o.prelude_x = Some(dup(gpu, o.x.as_ref().expect("x"), ne, "prelude_x")?);
    for b in std::mem::take(&mut o.blocks) {
        let d = dup(gpu, &b, ne, "prelude_block");
        let _ = gpu.free_tensor(b);
        o.prelude_blocks.push(d?);
    }
    for i in 0..o.prelude_blocks.len() {
        let d = dup(gpu, &o.prelude_blocks[i], ne, "prev_block")?;
        o.prev_blocks.push(d);
    }
    o.prev_out = Some(dup(gpu, o.prelude_x.as_ref().expect("x"), ne, "prev_out")?);
    for pass in 0..cfg.loop_cfg.num_passes {
        if let Some(x) = o.x.take() {
            let _ = gpu.free_tensor(x);
        }
        for b in o.blocks.drain(..) {
            let _ = gpu.free_tensor(b);
        }
        o.x = Some(dup(gpu, o.prelude_x.as_ref().expect("x"), ne, "pass x")?);
        for i in 0..o.prelude_blocks.len() {
            let d = dup(gpu, &o.prelude_blocks[i], ne, "pass block")?;
            o.blocks.push(d);
        }
        for i in 0..core_len {
            let core_i = if pass > 0 { Some(i) } else { None };
            step(gpu, c, gs, o, visit, core_i, &mut trace)?;
            visit += 1;
        }
        if pass + 1 < cfg.loop_cfg.num_passes {
            for b in o.prev_blocks.drain(..) {
                let _ = gpu.free_tensor(b);
            }
            for i in 0..o.blocks.len() {
                let d = dup(gpu, &o.blocks[i], ne, "prev block")?;
                o.prev_blocks.push(d);
            }
            if let Some(p) = o.prev_out.take() {
                let _ = gpu.free_tensor(p);
            }
            o.prev_out = Some(dup(gpu, o.x.as_ref().expect("x"), ne, "prev out")?);
        }
    }
    for _ in cfg.loop_cfg.core_end..cfg.n_phys {
        step(gpu, c, gs, o, visit, None, &mut trace)?;
        visit += 1;
    }
    if visit != cfg.loop_cfg.n_slot() {
        return Err(format!("inkling2 prefill: visit count {visit} != {}", cfg.loop_cfg.n_slot()));
    }
    let heads = head_rows(gpu, c, o, head)?;
    gs.seq_pos = c.p + n;
    Ok(heads)
}

/// Final mix over the block streams, final norm, LM head + softcap on the
/// requested rows. `Last` runs the singleton head arithmetic on the last row.
fn head_rows(
    gpu: &mut Gpu,
    c: &BlockCtx<'_>,
    o: &Owned,
    head: HeadRows,
) -> Result<Vec<(usize, Vec<f32>)>, String> {
    let cfg = c.cfg;
    let gw = c.gw;
    let n = c.rows;
    let e = cfg.n_embd;
    let r0 = match head {
        HeadRows::None => return Ok(Vec::new()),
        HeadRows::Last => n - 1,
        HeadRows::From(f) => f.min(n),
    };
    let rows = n - r0;
    if rows == 0 {
        return Ok(Vec::new());
    }
    if gw.output_query.buf.size() / 4 != e {
        return Err(format!("inkling2 prefill: output_query len {} != {e}", gw.output_query.buf.size() / 4));
    }
    let x = o.x.as_ref().ok_or("inkling2 prefill: missing x")?;
    let mut ar = Arena::default();
    let r = (|| -> Result<Vec<(usize, Vec<f32>)>, String> {
        let mut states = Vec::with_capacity(o.blocks.len() + 1);
        for b in o.blocks.iter().chain(std::iter::once(x)) {
            states.push(view(b, r0 * e, &[rows, e])?);
        }
        let refs: Vec<&GpuTensor> = states.iter().collect();
        let mixed = ar.f32(gpu, rows * e, "final_mix")?;
        mix_rows(gpu, &refs, &gw.output_query, ar.get(mixed), rows, e, 1, cfg.rms_norm_eps)?;
        let hidden = ar.f32(gpu, rows * e, "hidden")?;
        hip(
            gpu.rmsnorm_batched(&view(ar.get(mixed), 0, &[rows, e])?, &gw.norm, ar.get(hidden), rows, e, cfg.rms_norm_eps),
            "final norm",
        )?;
        let vocab = cfg.vocab_size;
        let mut out = Vec::with_capacity(rows);
        const HEAD_ROWS: usize = 64;
        let mut done = 0;
        while done < rows {
            let take = HEAD_ROWS.min(rows - done);
            let logits = ar.f32(gpu, take * vocab, "logits")?;
            let hv = view(ar.get(hidden), done * e, &[take, e])?;
            linear(gpu, &gw.unembed, &hv, ar.get(logits), take, c.mode.gemm)?;
            hip(gpu.logit_softcap_f32(ar.get(logits), take * vocab, cfg.softcap), "softcap")?;
            let host = hip(gpu.download_f32(&view(ar.get(logits), 0, &[take * vocab])?), "logits download")?;
            for i in 0..take {
                let row = &host[i * vocab..i * vocab + cfg.unpadded_vocab_size];
                out.push((c.p + r0 + done + i, row.to_vec()));
            }
            done += take;
        }
        Ok(out)
    })();
    ar.free(gpu);
    r
}
