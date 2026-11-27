// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU forward for Inkling-2-Mini (arch 24): the 48-visit decode loop.
//!
//! Brief 03 owns this file. It consumes brief 01's bundle/state
//! (`Inkling2GpuWeights`, `Inkling2GpuLayer`, `Inkling2GpuState` in
//! `gpu.rs` — read-only) and brief 02's four dispatches
//! (`inkling2_l2_clip_rows_f32`, `inkling2_rmsnorm_nogamma_f32`,
//! `inkling2_control_blend_f32` / `inkling2_control_f32_gpu`,
//! `inkling2_rel_bias_gather_f32`, `inkling2_mix_query_row_f32` /
//! `inkling2_mix_f32_gpu`). It mirrors `forward.rs::forward_token` exactly:
//! prelude visits 0..5, three core passes over 6..17, post visits 18..23
//! (physical), i.e. 48 visits keyed by visit index, never by physical layer.
//!
//! Binding kernel choices (from review; do not improvise):
//! * No `attention_dflash_sliding_f32`, no `dynamic_causal_conv_f32`, no
//!   `conv1d_silu_split_f32`, no `seal_decode` — all four are wrong here
//!   (SwiGLU vs SiTU, missing sink/past/scale). The two kernels below are
//!   the only HIP in this file.
//! * MoE calls the building blocks directly in the minimax pattern:
//!   router GEMV, `ffn_moe_router_topk_f32` (SigmoidBiasRenorm, eps 1e-20,
//!   scale 8.0), `rotate_x_mq`, the `_batched` gate_up (NOT the unbatched
//!   one: K=2304 drops group 9 there), `ffn_activation_rows_f32`
//!   (Situ(4.0, 25.0), rows 8, logical 1152, phys 1280), the
//!   `_batched_expanded` down (2304, 1280, 8, 1), `ffn_moe_combine_f32`
//!   (Rank).
//!
//! Kernel contract of the two dedicated kernels:
//! * `inkling2_gf_conv_f32`: short conv with history in place.
//!   Static path: `out[ch] = (cur[ch] + sum_l x[l]*W[l,C]) / sqrt(5)`.
//!   Dynamic path (full-layer K/V): torch-view static taps plus
//!   `dyn_w[h,lag]` over reversed history, `/ sqrt(9)`.
//!   Weights arrive pre-transformed from 01 (`[kernel, channels]`, row 0 =
//!   oldest tap), matching `oracle::convolve_static` / `convolve_dynamic`
//!   tap conventions. The unified per-visit state is `[3, 6656]` with
//!   segments K(0..1024) / V(1024..2048) / attn-out(2048..4352) /
//!   ffn-out(4352..6656).
//! * `inkling2_gf_attn_decode_f32`: 1-query decode attention, 1 block of 128
//!   threads per Q head (h in 0..24, kv_h = h/3). Online softmax over
//!   `0..n_kv` with the sink folded into `bias` (02's gather subtracts it):
//!   `m = 0, l = 1`, `score = dot/sqrt(d) + bias[h,k]`.
//!   Position-correct for any `pos >= 0`: logical row `k` reads physical
//!   slot `slots[k]` (SWA ring; identity below the ring size), dist from
//!   the position tags. K/V caches stay F16 (`INKLING2_KV_DTYPE`);
//!   conversion happens on load.

use crate::arch_model::Inkling2Bundle;
use crate::config::Inkling2Config;
use crate::gpu::{
    swa_gather_map, swa_slot, Inkling2Controller, Inkling2GpuLayer, Inkling2GpuState,
    Inkling2GpuWeights, Inkling2VisitScratch, Inkling2XtsLayer, INKLING2_CONV_WIDTH,
};
use hip_bridge::{DeviceBuffer, KernargBlob};
use hipfire_dispatch::pipeline::run_uniform_moe_down_expanded;
use hipfire_runtime::llama::{rotate_x_mq_batched_for, rotate_x_mq_for, weight_gemv, WeightTensor};
use rdna_compute::ffn_activation::{
    CombineOrder, FfnActivation, FfnActivationRows, FfnAwq, MoeCombine, MoeRouterTopK, RouterKind,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::HashMap;
use std::ffi::c_void;

// ---------------------------------------------------------------------------
// Dedicated HIP (conv + decode attention + D1 zero-copy decode kernels)
// ---------------------------------------------------------------------------

const INKLING2_GF_MODULE: &str = "inkling2_gpu_forward";

const INKLING2_GF_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// --- D1 zero-copy decode kernels --------------------------------------
// Device pos word: advanced once per token by a 1-thread kernel, so the
// steady-state step performs zero H2Ds. 0.0f32 and 0u32 share bits, so
// reset writes the word with the existing F32 fill path.
extern "C" __global__ void inkling2_gf_advance_pos(unsigned int* __restrict__ pos) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *pos += 1u;
}

// Device slots/dists map: slots[k] = (start+k) % rows,
// dists[k] = pos-(start+k), n_kv = min(pos+1, rows), all from the device
// pos word. One thread per k; grid covers the largest visit. Matches
// swa_gather_map() exactly (same arithmetic, same order); the host keeps
// validating tags fail-closed with zero H2D.
extern "C" __global__ void inkling2_gf_build_slots_dists(
    const unsigned int* __restrict__ pos,
    int* __restrict__ slots,
    int* __restrict__ dists,
    int rows) {
    int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int p = *pos;
    int n_kv = (int)p + 1;
    if (n_kv > rows) n_kv = rows;
    if (k >= n_kv) return;
    int kp = (int)p + 1 - n_kv + k;
    slots[k] = kp % rows;
    dists[k] = (int)p - kp;
}

// Batched KV write, F16 path. Grid (8, 2): block (h, kv) converts one
// 128-wide F32 head row into cache row wpos. Bitwise identical to 16
// separate convert_f32_to_f16_into launches (same (_Float16) cast).
extern "C" __global__ void inkling2_gf_kv_write_f16_batched(
    const float* __restrict__ k_conv, // [8, 128] contiguous
    const float* __restrict__ vc,     // [8, 128] contiguous
    unsigned short* __restrict__ k0, unsigned short* __restrict__ k1,
    unsigned short* __restrict__ k2, unsigned short* __restrict__ k3,
    unsigned short* __restrict__ k4, unsigned short* __restrict__ k5,
    unsigned short* __restrict__ k6, unsigned short* __restrict__ k7,
    unsigned short* __restrict__ v0, unsigned short* __restrict__ v1,
    unsigned short* __restrict__ v2, unsigned short* __restrict__ v3,
    unsigned short* __restrict__ v4, unsigned short* __restrict__ v5,
    unsigned short* __restrict__ v6, unsigned short* __restrict__ v7,
    int wpos, int hd) {
    int h = (int)blockIdx.x;   // 0..7
    int kv = (int)blockIdx.y;  // 0 = K, 1 = V
    int d = (int)threadIdx.x;  // 0..127
    if (h >= 8 || d >= hd) return;
    unsigned short* ks[8] = {k0, k1, k2, k3, k4, k5, k6, k7};
    unsigned short* vs[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
    const float* src = (kv == 0 ? k_conv : vc) + (size_t)h * (size_t)hd;
    unsigned short* row = (kv == 0 ? ks[h] : vs[h]) + (size_t)wpos * (size_t)hd;
    row[d] = __half_as_ushort((__half)(src[d]));
}

// Short conv, one thread per channel. Static path taps W[l,C] over
// x = [past0, past1, past2, cur] (01 uploads [K,C] with row 0 oldest).
// Dynamic path (full-layer K/V): static taps in torch view static_w[ch*4+lag]
// over history[lag] plus dyn_w[h*4+lag] over history[3-lag]; /sqrt(9).
// History updates in place; out is separate (no aliasing).
extern "C" __global__ void inkling2_gf_conv_f32(
    const float* __restrict__ cur,
    const float* __restrict__ sw,
    const float* __restrict__ dyn_w,
    float* __restrict__ state,
    float* __restrict__ out,
    int state_stride, int col_off, int channels,
    int dyn_heads, int head_dim, float inv_scale)
{
    int ch = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (ch >= channels) return;
    float x0 = state[col_off + ch];
    float x1 = state[state_stride + col_off + ch];
    float x2 = state[state_stride * 2 + col_off + ch];
    float xc = cur[ch];
    float acc;
    if (dyn_heads == 0) {
        acc = x0 * sw[ch] + x1 * sw[channels + ch]
            + x2 * sw[channels * 2 + ch] + xc * sw[channels * 3 + ch];
    } else {
        int h = ch / head_dim;
        const float* drow = dyn_w + (size_t)h * 4;
        // Static taps share 01's [K,C] layout (row = tap, oldest first),
        // identical to the static path above; only the dynamic taps (torch
        // view dyn_w[h,lag] over reversed history) differ.
        acc = x0 * sw[ch] + x1 * sw[channels + ch]
            + x2 * sw[channels * 2 + ch] + xc * sw[channels * 3 + ch]
            + xc * drow[0] + x2 * drow[1] + x1 * drow[2] + x0 * drow[3];
    }
    out[ch] = (acc + xc) * inv_scale;
    state[col_off + ch] = x1;
    state[state_stride + col_off + ch] = x2;
    state[state_stride * 2 + col_off + ch] = xc;
}

// 1-query decode attention: 1 block of 128 threads per Q head.
// Online softmax (m = 0, l = 1: the sink is already folded into bias by
// inkling2_rel_bias_gather_f32, whose zero sink value contributes nothing).
// K/V are F16 cache rows, converted on load.
extern "C" __global__ void inkling2_gf_attn_decode_f32(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k0, const unsigned short* __restrict__ k1,
    const unsigned short* __restrict__ k2, const unsigned short* __restrict__ k3,
    const unsigned short* __restrict__ k4, const unsigned short* __restrict__ k5,
    const unsigned short* __restrict__ k6, const unsigned short* __restrict__ k7,
    const unsigned short* __restrict__ v0, const unsigned short* __restrict__ v1,
    const unsigned short* __restrict__ v2, const unsigned short* __restrict__ v3,
    const unsigned short* __restrict__ v4, const unsigned short* __restrict__ v5,
    const unsigned short* __restrict__ v6, const unsigned short* __restrict__ v7,
    const float* __restrict__ bias,
    const int* __restrict__ slots,
    float* __restrict__ out,
    int n_kv, float inv_sqrt_d)
{
    int h = (int)blockIdx.x;    // 0..24
    int d = (int)threadIdx.x;   // 0..127
    const unsigned short* ks[8] = {k0, k1, k2, k3, k4, k5, k6, k7};
    const unsigned short* vs[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
    const unsigned short* kb = ks[h / 3];
    const unsigned short* vb = vs[h / 3];
    const float* qh = q + (size_t)h * 128;
    const float* bh = bias + (size_t)h * (size_t)n_kv;
    float qd = qh[d];
    float m = 0.0f, l = 1.0f, o = 0.0f;
    __shared__ float sh[128];
    for (int k = 0; k < n_kv; ++k) {
        // Ring indirection: logical row k lives in physical slot slots[k].
        // Below the ring size slots[k] == k, identical to direct indexing.
        int kk = slots[k];
        float kd = __half2float(__ushort_as_half(kb[(size_t)kk * 128 + d]));
        sh[d] = qd * kd;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (d < s) sh[d] += sh[d + s];
            __syncthreads();
        }
        float s = sh[0] * inv_sqrt_d + bh[k];
        float m_new = fmaxf(m, s);
        float a = expf(m - m_new);
        float e = expf(s - m_new);
        l = l * a + e;
        float vd = __half2float(__ushort_as_half(vb[(size_t)kk * 128 + d]));
        o = o * a + e * vd;
        m = m_new;
        __syncthreads();
    }
    out[(size_t)h * 128 + d] = o / l;
}
"#;

fn ensure_kernels(gpu: &mut Gpu) -> Result<(), String> {
    gpu.ensure_kernel_public(
        INKLING2_GF_MODULE,
        INKLING2_GF_HIP_SRC,
        "inkling2_gf_conv_f32",
    )
    .map_err(|e| format!("inkling2 gpu fwd: conv kernel: {e:?}"))?;
    gpu.ensure_kernel_public(
        INKLING2_GF_MODULE,
        INKLING2_GF_HIP_SRC,
        "inkling2_gf_attn_decode_f32",
    )
    .map_err(|e| format!("inkling2 gpu fwd: attn kernel: {e:?}"))?;
    gpu.ensure_kernel_public(
        INKLING2_GF_MODULE,
        INKLING2_GF_HIP_SRC,
        "inkling2_gf_advance_pos",
    )
    .map_err(|e| format!("inkling2 gpu fwd: advance_pos kernel: {e:?}"))?;
    gpu.ensure_kernel_public(
        INKLING2_GF_MODULE,
        INKLING2_GF_HIP_SRC,
        "inkling2_gf_build_slots_dists",
    )
    .map_err(|e| format!("inkling2 gpu fwd: slots kernel: {e:?}"))?;
    gpu.ensure_kernel_public(
        INKLING2_GF_MODULE,
        INKLING2_GF_HIP_SRC,
        "inkling2_gf_kv_write_f16_batched",
    )
    .map_err(|e| format!("inkling2 gpu fwd: kv_write kernel: {e:?}"))?;
    Ok(())
}

/// D1: advance the device pos word by one (1 thread, 0 H2D). Launched once
/// per completed token; the slots kernel of the next token reads the word.
fn launch_advance_pos(gpu: &mut Gpu, pos_dev: &GpuTensor) -> Result<(), String> {
    let mut blob = KernargBlob::new();
    blob.push_ptr(pos_dev.buf.as_ptr());
    blob.pad_to(16);
    gpu.launch_kernel_blob(
        "inkling2_gf_advance_pos",
        [1, 1, 1],
        [1, 1, 1],
        0,
        blob.as_mut_slice(),
    )
    .map_err(|e| format!("inkling2 gpu fwd: advance pos launch: {e:?}"))
}

/// D1: build slots/dists maps on-device from the pos word (grid covers
/// `max_rows`; threads past n_kv exit). Replaces 2 H2D uploads per visit.
fn launch_build_slots_dists(
    gpu: &mut Gpu,
    pos_dev: &GpuTensor,
    slots_dev: &GpuTensor,
    dists_dev: &GpuTensor,
    max_rows: usize,
) -> Result<(), String> {
    let mut blob = KernargBlob::new();
    blob.push_ptr(pos_dev.buf.as_ptr());
    blob.push_ptr(slots_dev.buf.as_ptr());
    blob.push_ptr(dists_dev.buf.as_ptr());
    blob.push_i32(max_rows as i32);
    blob.pad_to(16);
    let grid = [(max_rows as u32).div_ceil(256).max(1), 1, 1];
    gpu.launch_kernel_blob(
        "inkling2_gf_build_slots_dists",
        grid,
        [256, 1, 1],
        0,
        blob.as_mut_slice(),
    )
    .map_err(|e| format!("inkling2 gpu fwd: slots launch: {e:?}"))
}

/// D1: single batched F16 KV-write launch per visit, grid (8, 2). `k_conv`
/// and `vc` are the contiguous `[8, 128]` visit temps; each cache head
/// buffer receives row `wpos`. Bitwise identical to 16 converts.
#[allow(clippy::too_many_arguments)]
fn launch_kv_write_f16_batched(
    gpu: &mut Gpu,
    k_conv: &GpuTensor,
    vc: &GpuTensor,
    k_cache: &[GpuTensor],
    v_cache: &[GpuTensor],
    wpos: usize,
    head_dim: usize,
) -> Result<(), String> {
    if k_cache.len() != 8 || v_cache.len() != 8 {
        return Err("inkling2 gpu fwd: batched kv write needs 8 K and 8 V caches".to_string());
    }
    if head_dim != 128 {
        return Err("inkling2 gpu fwd: batched kv write is head_dim=128".to_string());
    }
    let mut blob = KernargBlob::new();
    blob.push_ptr(k_conv.buf.as_ptr());
    blob.push_ptr(vc.buf.as_ptr());
    for t in k_cache {
        blob.push_ptr(t.buf.as_ptr());
    }
    for t in v_cache {
        blob.push_ptr(t.buf.as_ptr());
    }
    blob.push_i32(wpos as i32);
    blob.push_i32(head_dim as i32);
    blob.pad_to(16);
    gpu.launch_kernel_blob(
        "inkling2_gf_kv_write_f16_batched",
        [8, 2, 1],
        [128, 1, 1],
        0,
        blob.as_mut_slice(),
    )
    .map_err(|e| format!("inkling2 gpu fwd: kv_write launch: {e:?}"))
}

#[allow(clippy::too_many_arguments)]
fn launch_conv(
    gpu: &mut Gpu,
    cur: &GpuTensor,
    static_w: &GpuTensor,
    dyn_w: Option<&GpuTensor>,
    state: &GpuTensor,
    out: &GpuTensor,
    col_off: usize,
    channels: usize,
    dyn_heads: usize,
    head_dim: usize,
    inv_scale: f32,
) -> Result<(), String> {
    let null: *mut c_void = std::ptr::null_mut();
    let mut blob = KernargBlob::new();
    blob.push_ptr(cur.buf.as_ptr());
    blob.push_ptr(static_w.buf.as_ptr());
    blob.push_ptr(dyn_w.map(|t| t.buf.as_ptr()).unwrap_or(null));
    blob.push_ptr(state.buf.as_ptr());
    blob.push_ptr(out.buf.as_ptr());
    blob.push_i32(INKLING2_CONV_WIDTH as i32);
    blob.push_i32(col_off as i32);
    blob.push_i32(channels as i32);
    blob.push_i32(dyn_heads as i32);
    blob.push_i32(head_dim as i32);
    blob.push_f32(inv_scale);
    blob.pad_to(16);
    let grid = [(channels as u32).div_ceil(256), 1, 1];
    gpu.launch_kernel_blob(
        "inkling2_gf_conv_f32",
        grid,
        [256, 1, 1],
        0,
        blob.as_mut_slice(),
    )
    .map_err(|e| format!("inkling2 gpu fwd: conv launch: {e:?}"))
}

#[allow(clippy::too_many_arguments)]
fn launch_attn(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k_cache: &[GpuTensor],
    v_cache: &[GpuTensor],
    bias: &GpuTensor,
    slots: &GpuTensor,
    out: &GpuTensor,
    n_kv: usize,
    n_head: usize,
    head_dim: usize,
) -> Result<(), String> {
    if k_cache.len() != 8 || v_cache.len() != 8 {
        return Err("inkling2 gpu fwd: need exactly 8 K and 8 V caches".to_string());
    }
    if head_dim != 128 {
        return Err("inkling2 gpu fwd: attn kernel is head_dim=128".to_string());
    }
    let mut blob = KernargBlob::new();
    blob.push_ptr(q.buf.as_ptr());
    for t in k_cache {
        blob.push_ptr(t.buf.as_ptr());
    }
    for t in v_cache {
        blob.push_ptr(t.buf.as_ptr());
    }
    blob.push_ptr(bias.buf.as_ptr());
    blob.push_ptr(slots.buf.as_ptr());
    blob.push_ptr(out.buf.as_ptr());
    blob.push_i32(n_kv as i32);
    blob.push_f32(1.0 / (head_dim as f32).sqrt());
    blob.pad_to(16);
    gpu.launch_kernel_blob(
        "inkling2_gf_attn_decode_f32",
        [n_head as u32, 1, 1],
        [head_dim as u32, 1, 1],
        (head_dim * 4) as u32,
        blob.as_mut_slice(),
    )
    .map_err(|e| format!("inkling2 gpu fwd: attn launch: {e:?}"))
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn hip<T>(r: Result<T, hip_bridge::HipError>, what: &str) -> Result<T, String> {
    r.map_err(|e| format!("inkling2 gpu fwd: {what}: {e:?}"))
}

/// Allocate a visit temp. Temps die at visit end; results never live here.
fn at(
    gpu: &mut Gpu,
    tmp: &mut Vec<GpuTensor>,
    shape: &[usize],
    what: &str,
) -> Result<usize, String> {
    let t = gpu
        .alloc_tensor(shape, DType::F32)
        .map_err(|e| format!("inkling2 gpu fwd: alloc {what}: {e:?}"))?;
    tmp.push(t);
    Ok(tmp.len() - 1)
}

fn free_all(gpu: &mut Gpu, ts: Vec<GpuTensor>) {
    for t in ts {
        let _ = gpu.free_tensor(t);
    }
}

/// Checked non-owning byte-range view into a pool-served buffer.
///
/// The reuse pool best-fit may hand a larger buffer to an exact-size request,
/// so a bigger buffer is normal: the view takes the logical
/// `[byte_offset, byte_offset + byte_len)` prefix (same elements a fresh
/// exact buffer would hold — `>=`, not `==`). A short buffer fails closed
/// with `Err` (surfaces as 503) instead of an `assert!` panic that would
/// kill the daemon when the pool retires under OOM pressure. Never clamps:
/// short is refused, never silently truncated.
fn sub_buf(
    t: &GpuTensor,
    byte_offset: usize,
    byte_len: usize,
    what: &str,
) -> Result<DeviceBuffer, String> {
    let end = byte_offset.checked_add(byte_len).ok_or_else(|| {
        format!("inkling2 gpu fwd: {what}: offset {byte_offset}+len {byte_len} overflows")
    })?;
    let have = t.buf.size();
    if end > have {
        return Err(format!(
            "inkling2 gpu fwd: {what}: need [{byte_offset},{end}) of {have} buffer bytes"
        ));
    }
    Ok(t.buf.byte_view(byte_offset, byte_len))
}

/// Non-owning `[rows, cols]` F32 view of a whole buffer. The pool best-fit may
/// hand a larger buffer to an exact-size request; the view takes the logical
/// prefix (same elements a fresh exact buffer would hold), so `>=` is the
/// correct gate, not `==`. Short buffers and wrong dtypes fail closed with
/// `Err`, never a daemon-killing `assert!`.
fn view2(t: &GpuTensor, rows: usize, cols: usize) -> Result<GpuTensor, String> {
    if t.dtype != DType::F32 {
        return Err(format!(
            "inkling2 gpu fwd: view2 of {:?}, need F32",
            t.dtype
        ));
    }
    let len = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| format!("inkling2 gpu fwd: view2 [{rows},{cols}] extent overflows"))?;
    Ok(GpuTensor {
        buf: sub_buf(t, 0, len, "view2")?,
        shape: vec![rows, cols],
        dtype: DType::F32,
    })
}

/// Copy row `row` of a `[rows, e]` buffer into a flat `[e]` buffer.
fn copy_row_from(
    gpu: &mut Gpu,
    src_2d: &GpuTensor,
    row: usize,
    dst: &GpuTensor,
    e: usize,
    what: &str,
) -> Result<(), String> {
    let need = e
        .checked_mul(4)
        .ok_or_else(|| format!("inkling2 gpu fwd: copy_row_from [{e}] extent overflows"))?;
    let off = row
        .checked_mul(need)
        .ok_or_else(|| format!("inkling2 gpu fwd: copy_row_from row {row}*{e} overflows"))?;
    let view = GpuTensor {
        buf: sub_buf(src_2d, off, need, "copy_row_from src")?,
        shape: vec![e],
        dtype: DType::F32,
    };
    if dst.buf.size() < need {
        return Err(format!(
            "inkling2 gpu fwd: copy_row_from dst: need {need} of {} buffer bytes",
            dst.buf.size()
        ));
    }
    hip(
        gpu.copy_rows_strided_f32(&view, dst, 1, e, e, e, 0, None),
        what,
    )
}

fn dup(gpu: &mut Gpu, src: &GpuTensor, what: &str) -> Result<GpuTensor, String> {
    // Logical element count, NOT buffer bytes: the pool best-fit may hand a
    // larger buffer to an [e] request, and sizing the copy from the buffer
    // would clone pool slack into the shape (a [3072]-shaped "copy" of an
    // [e] row — silent under copy-with-explicit-counts, fatal to D1's
    // pointer banks). The buffer must still hold all logical elements.
    let n: usize = src.shape.iter().product();
    if src.buf.size() / 4 < n {
        return Err(format!(
            "inkling2 gpu fwd: dup {what}: buffer {} words < shape {n}",
            src.buf.size() / 4
        ));
    }
    let out = gpu
        .alloc_tensor(&[n], DType::F32)
        .map_err(|e| format!("inkling2 gpu fwd: dup {what}: {e:?}"))?;
    hip(
        gpu.copy_rows_strided_f32(src, &out, 1, n, n, n, 0, None),
        "dup",
    )?;
    Ok(out)
}

fn dl(gpu: &Gpu, t: &GpuTensor) -> Result<Vec<f32>, String> {
    gpu.download_f32(t)
        .map_err(|e| format!("inkling2 gpu fwd: download: {e:?}"))
}

fn cap_put(
    cap: &mut Option<HashMap<String, Vec<f32>>>,
    gpu: &Gpu,
    key: &str,
    t: &GpuTensor,
) -> Result<(), String> {
    if let Some(m) = cap.as_mut() {
        m.insert(key.to_string(), dl(gpu, t)?);
    }
    Ok(())
}

/// Routed-expert stage of a typed-XTS (`qt44-padded-awq-v1`) layer for one row
/// (scalar decode): returns the `tmp` index of `down_exp [top_k, n_embd]`.
///
/// 1. `rotate_x_mq_awq_topk`: `rows[r] = FWHT256(ffn_normed / S_gu[e_r])` (exact f32
///    divide; with `S == 1` bitwise `rotate_x_mq` of the shared row);
/// 2. gate_up through the qt44 expanded GEMV with per-rank rows (`m = 2*inter`,
///    `k = n_embd`) -> `[top_k, 2*inter]` rows `[gate | up]`;
/// 3. SiTU (gate offset 0, up offset `inter`, ld `2*inter`, logical `inter`, physical
///    `down_k_physical`) with the per-expert down AWQ divide (`row_expert` = route ids);
/// 4. `rotate_x_mq_batched` over the physical width;
/// 5. down through the same expanded GEMV (`m = n_embd`, `k = down_k_physical`).
///
/// No D2H of route ids, no host expert loop, no per-token table rebuild: the
/// pointer tables live on the layer. Failure leaves `tmp` for the caller to free.
pub(crate) fn xts_expert_stage(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    xl: &Inkling2XtsLayer,
    tmp: &mut Vec<GpuTensor>,
    ffn_normed: &GpuTensor,
    topk_idx: &GpuTensor,
    cap: &mut Option<HashMap<String, Vec<f32>>>,
) -> Result<usize, String> {
    let e = cfg.n_embd;
    let inter = cfg.inter;
    let phys = crate::weights::xts_down_dims(cfg).1;
    let k_top = cfg.n_expert_used;
    let n_exp = cfg.n_experts;
    let rows = at(gpu, tmp, &[k_top, e], "xts_x_rot")?;
    gpu.rotate_x_mq_awq_topk(
        ffn_normed,
        &xl.gu_awq_ptrs,
        topk_idx,
        &tmp[rows],
        e,
        k_top,
        n_exp,
    )
    .map_err(|e2| format!("inkling2 gpu fwd: xts producer: {e2:?}"))?;
    let gu = at(gpu, tmp, &[k_top, 2 * inter], "xts_gate_up")?;
    run_uniform_moe_down_expanded(
        gpu,
        DType::MQ4G256V2,
        &xl.gu_ptrs,
        topk_idx,
        &tmp[rows],
        &tmp[gu],
        2 * inter,
        e,
        k_top,
        1,
    )
    .map_err(|e2| format!("inkling2 gpu fwd: xts gate_up: {e2:?}"))?;
    cap_put(cap, gpu, "gate_up", &tmp[gu])?;
    let act = at(gpu, tmp, &[k_top, phys], "xts_act")?;
    gpu.ffn_activation_rows_f32(&FfnActivationRows {
        activation: FfnActivation::Situ {
            gate_beta: cfg.situ_beta,
            value_beta: cfg.situ_linear_beta,
        },
        gate: &tmp[gu],
        gate_offset: 0,
        gate_ld: 2 * inter,
        up: &tmp[gu],
        up_offset: inter,
        up_ld: 2 * inter,
        out: &tmp[act],
        rows: k_top,
        logical_n: inter,
        phys_n: phys,
        round_inputs: false,
        round_output: false,
        awq: FfnAwq::PerExpert {
            table: &xl.down_awq_ptrs,
            row_expert: topk_idx,
            n_experts: n_exp,
        },
    })
    .map_err(|e2| format!("inkling2 gpu fwd: xts situ: {e2:?}"))?;
    cap_put(cap, gpu, "act", &tmp[act])?;
    let rot = at(gpu, tmp, &[k_top, phys], "xts_rot")?;
    gpu.rotate_x_mq_batched(&tmp[act], &tmp[rot], phys, k_top)
        .map_err(|e2| format!("inkling2 gpu fwd: xts act rotate: {e2:?}"))?;
    let down_exp = at(gpu, tmp, &[k_top, e], "xts_down_exp")?;
    run_uniform_moe_down_expanded(
        gpu,
        DType::MQ4G256V2,
        &xl.down_ptrs,
        topk_idx,
        &tmp[rot],
        &tmp[down_exp],
        e,
        phys,
        k_top,
        1,
    )
    .map_err(|e2| format!("inkling2 gpu fwd: xts down: {e2:?}"))?;
    cap_put(cap, gpu, "down_exp", &tmp[down_exp])?;
    cap_put(cap, gpu, "rot_b", &tmp[rot])?;
    Ok(down_exp)
}

/// AttnRes-style mix over norm-once `normed_bank` + `cur` with a `[qr, e]` query.
/// D1 zero-copy path: bank rows are already RMSNormed (norm-once at push
/// with the same tree reduction as the copying path, so bitwise identical),
/// `cur` is normalized in-kernel into `scratch.mix_cur_normed`. No bank
/// allocation, no `copy_rows`, no per-call bank re-norm. Only the `[qr, e]`
/// output escapes (pool-owned, as before).
fn gpu_mix(
    gpu: &mut Gpu,
    scratch: &Inkling2VisitScratch,
    normed_bank: &[&GpuTensor],
    cur: &GpuTensor,
    query: &GpuTensor,
    e: usize,
    qr: usize,
    eps: f32,
) -> Result<GpuTensor, String> {
    let n = normed_bank.len() + 1;
    if !(1..=6).contains(&n) {
        return Err(format!("inkling2 gpu fwd: mix states {n} exceeds 6"));
    }
    if normed_bank.len() > 5 {
        return Err(format!(
            "inkling2 gpu fwd: mix bank {} exceeds 5",
            normed_bank.len()
        ));
    }
    let mut tmp = Vec::new();
    let fail = |gpu: &mut Gpu, tmp: Vec<GpuTensor>, msg: String| -> Result<GpuTensor, String> {
        free_all(gpu, tmp);
        Err(msg)
    };
    // Only the output allocates; every input is read in place.
    let out = match at(gpu, &mut tmp, &[qr, e], "mix_out") {
        Ok(i) => i,
        Err(msg) => return fail(gpu, tmp, msg),
    };
    let qv = match view2(query, qr, e) {
        Ok(v) => v,
        Err(msg) => return fail(gpu, tmp, msg),
    };
    // Norm cur into the persistent slot inside the wrapper (one independent
    // batch row — bitwise identical to norming it as part of a contiguous
    // bank with the same eps).
    if let Err(e2) = gpu.inkling2_mix_ptr_f32_gpu(
        normed_bank,
        cur,
        &scratch.mix_cur_normed,
        &qv,
        &tmp[out],
        e,
        qr,
        eps,
    ) {
        return fail(gpu, tmp, format!("inkling2 gpu fwd: mix: {e2:?}"));
    }
    let got = tmp.pop().expect("mix out is last");
    free_all(gpu, tmp);
    Ok(got)
}

/// One fused controller step. Internal scratch is the persistent
/// `scratch.ctrl_scratch` slot (dead after each call); only the output
/// escapes (pool-owned, as before — the four message outputs stay live
/// simultaneously, so they cannot share one slot).
fn gpu_control(
    gpu: &mut Gpu,
    scratch: &Inkling2VisitScratch,
    cur: &GpuTensor,
    msg: &GpuTensor,
    ctrl: &Inkling2Controller,
    e: usize,
    rank: usize,
) -> Result<GpuTensor, String> {
    let mut tmp = Vec::new();
    let fail = |gpu: &mut Gpu, tmp: Vec<GpuTensor>, msg: String| -> Result<GpuTensor, String> {
        free_all(gpu, tmp);
        Err(msg)
    };
    let out = match at(gpu, &mut tmp, &[e], "ctrl_out") {
        Ok(i) => i,
        Err(msg) => return fail(gpu, tmp, msg),
    };
    if let Err(e2) = gpu.inkling2_control_f32_gpu(
        cur,
        msg,
        &ctrl.down,
        &ctrl.gain,
        &ctrl.forget,
        &tmp[out],
        &scratch.ctrl_scratch,
        e,
        rank,
    ) {
        return fail(gpu, tmp, format!("inkling2 gpu fwd: control: {e2:?}"));
    }
    let got = tmp.pop().expect("ctrl out is last");
    free_all(gpu, tmp);
    Ok(got)
}

// ---------------------------------------------------------------------------
/// Full-layer tau decision for one GPU visit (`oracle::tau` parity).
/// `None` means t is 1 within epsilon and no scaling launch is needed.
fn full_layer_tau(pos: usize, floor: usize, alpha: f32) -> Result<Option<f32>, String> {
    let t = crate::oracle::tau(pos, floor, alpha);
    if (t - 1.0).abs() <= crate::config::GPU_UNIT_TAU_TOLERANCE {
        return Ok(None);
    }
    if !t.is_finite() || t <= 0.0 || t > crate::config::GPU_MAX_FULL_TAU {
        return Err(format!(
            "inkling2 gpu fwd: full-layer tau {t} at pos {pos} out of range"
        ));
    }
    Ok(Some(t))
}

// One attention+MoE visit on the GPU
// ---------------------------------------------------------------------------

struct VisitOut {
    visit_out: GpuTensor,
    push: Option<GpuTensor>,
    /// D1 norm-once twin of `push`: RMSNormed once at push time, rides in
    /// `Owned.blocks_n` and feeds every later mix via pointers.
    push_normed: Option<GpuTensor>,
}

/// `msgs`: pass > 0 core visits carry `(msg_bank, prev_out, cross[4,e], ctrls)`.
/// `gs` carries the visit's conv slice, KV caches and cursor (updated here).
#[allow(clippy::too_many_arguments)]
fn run_visit(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    layer: &Inkling2GpuLayer,
    x: &GpuTensor,
    blocks: &[GpuTensor],
    blocks_normed: &[GpuTensor],
    visit: usize,
    phys: usize,
    pos: usize,
    msgs: Option<(
        &[GpuTensor],
        &[GpuTensor],
        &GpuTensor,
        &GpuTensor,
        &[Inkling2Controller; 4],
    )>,
    gs: &mut Inkling2GpuState,
    cap: &mut Option<HashMap<String, Vec<f32>>>,
) -> Result<VisitOut, String> {
    let e = cfg.n_embd;
    let nh = cfg.n_head;
    let nkv = cfg.n_head_kv;
    let hd = cfg.head_dim;
    let ekv = nkv * hd;
    let eattn = nh * hd;
    let erel = nh * cfg.n_rel;
    let mut tmp: Vec<GpuTensor> = Vec::new();
    // Block-push made before the MoE half; freed with tmp on error.
    let mut push: Option<GpuTensor> = None;
    let mut push_normed: Option<GpuTensor> = None;
    // Cross-piece rows live in a caller-owned `rows` vec (never in `tmp`),
    // so the shared pieces can take `&rows[i]` refs alongside `&mut tmp`
    // without borrow conflicts. The seam owns pool temps here; the blocked
    // driver passes stack-row views in the same layout. Fixed layout:
    // 0 qi, 1 ki, 2 vi (head outputs), 3 q, 4 k_raw, 5 v_raw, 6 r
    // (projections), 7 sinks (pre output), 8 attn_heads (attn output),
    // 9 ffn_normed (tail output), 10 ffn_rot (rotated MoE input),
    // 11 router_logits, 12 topk_idx, 13 topk_w, 14 moe_out (MoE output),
    // 15 wo_out (post-attention projection output).
    // Alloc order shifts vs the old inline body (pool addresses may differ);
    // values are identical. Declared before `step!` (macro hygiene: the
    // macro body must see `rows`).
    let mut rows: Vec<GpuTensor> = Vec::new();
    for (n, what) in [
        (e, "q_inp"),
        (e, "k_inp"),
        (e, "v_inp"),
        (eattn, "q"),
        (ekv, "k_raw"),
        (ekv, "v_raw"),
        (erel, "r_raw"),
        (nh, "sinks"),
        (eattn, "attn_heads"),
        (e, "ffn_normed"),
        (e, "ffn_x_rot"),
        (cfg.n_experts, "router_logits"),
        (cfg.n_expert_used, "topk_idx"),
        (cfg.n_expert_used, "topk_w"),
        (e, "moe_out"),
        (e, "wo_out"),
    ] {
        let t = match gpu.alloc_tensor(&[n], DType::F32) {
            Ok(t) => t,
            Err(e2) => {
                free_all(gpu, rows);
                free_all(gpu, tmp);
                return Err(format!("inkling2 gpu fwd: alloc {what}: {e2:?}"));
            }
        };
        rows.push(t);
    }
    macro_rules! step {
        ($expr:expr, $what:literal) => {
            match $expr {
                Ok(v) => v,
                Err(e) => {
                    if let Some(p) = push.take() {
                        let _ = gpu.free_tensor(p);
                    }
                    if let Some(p) = push_normed.take() {
                        let _ = gpu.free_tensor(p);
                    }
                    free_all(gpu, rows);
                    free_all(gpu, tmp);
                    return Err(format!("inkling2 gpu fwd: {}: {e:?}", $what));
                }
            }
        };
    }
    // No bare `?` below this point: every failure must free tmp/push/rows.

    if cfg.conv_kernel != 4 {
        free_all(gpu, rows);
        free_all(gpu, tmp);
        return Err(format!(
            "inkling2 gpu fwd: conv_kernel {} != 4",
            cfg.conv_kernel
        ));
    }
    // Visit head: mix bases, messages/control, input norms (shared with the
    // blocked driver).
    let mids = match visit_head(
        gpu,
        cfg,
        layer,
        x,
        blocks,
        blocks_normed,
        visit,
        phys,
        msgs.clone(),
        &gs.scratch,
        cap,
        &mut tmp,
        &rows[0],
        &rows[1],
        &rows[2],
    ) {
        Ok(m) => m,
        Err(e) => {
            free_all(gpu, rows);
            free_all(gpu, tmp);
            return Err(e);
        }
    };
    // Projections (seam GEMVs; the blocked driver reads q32 stacks instead).
    step!(weight_gemv(gpu, &layer.wq, &rows[0], &rows[3]), "wq");
    step!(weight_gemv(gpu, &layer.wk, &rows[1], &rows[4]), "wk");
    step!(weight_gemv(gpu, &layer.wv, &rows[2], &rows[5]), "wv");
    step!(weight_gemv(gpu, &layer.wr, &rows[0], &rows[6]), "wr");
    // Pre-attention half (shared with the blocked driver): convs, clips,
    // norms, sinks, tags, KV write. Reads rows[0..7], writes rows[7].
    // `step!` frees rows+tmp on error (no push yet at this point).
    let n_kv = step!(
        visit_pre(
            gpu, cfg, layer, gs, visit, phys, pos, &rows[0], &rows[1], &rows[2], &rows[3],
            &rows[4], &rows[5], &rows[6], &rows[7], cap, &mut tmp,
        ),
        "pre"
    );
    // Single-row attention (shared with the blocked fallback arm; the
    // batched arm uses the qtile instead). Writes rows[8].
    {
        let qv = step!(view2(&rows[3], nh, hd), "view");
        step!(
            visit_attn_single(
                gpu, cfg, layer, gs, visit, &qv, &rows[6], &rows[7], n_kv, &rows[8], cap, &mut tmp,
            ),
            "attn"
        );
    }
    step!(cap_put(cap, gpu, "attn_heads", &rows[8]), "capture");
    // Output gate + projection + post conv (seam GEMVs; the blocked
    // driver uses blocked_gate_values/blocked_wo_values instead).
    let gate = step!(at(gpu, &mut tmp, &[eattn], "gate"), "alloc");
    step!(
        weight_gemv(gpu, &layer.wgate, &rows[0], &tmp[gate]),
        "wgate"
    );
    step!(gpu.sigmoid_f32(&tmp[gate]), "gate sigmoid");
    step!(gpu.mul_f32(&rows[8], &tmp[gate], &rows[8]), "gate mul");
    step!(weight_gemv(gpu, &layer.wo, &rows[8], &rows[15]), "wo");
    step!(cap_put(cap, gpu, "wo_out", &rows[15]), "capture");
    // Post-attention tail through the FFN norm (shared with the blocked
    // driver): post-conv, cur/push, MoE mix/control/norm into rows[9].
    // `step!` frees rows+tmp on error.
    let (cur, p1, p2) = step!(
        tail_conv_norm(
            gpu,
            cfg,
            layer,
            gs,
            visit,
            phys,
            msgs,
            mids,
            cap,
            &mut tmp,
            x,
            &rows[15],
            &rows[9],
            blocks_normed,
        ),
        "tail"
    );
    push = p1;
    push_normed = p2;
    // Router: logits, top-8 (SigmoidBiasRenorm eps 1e-20, scale route_scale).
    step!(
        weight_gemv(gpu, &layer.router, &rows[9], &rows[11]),
        "router"
    );
    step!(cap_put(cap, gpu, "router_logits", &rows[11]), "capture");
    step!(
        gpu.ffn_moe_router_topk_f32(&MoeRouterTopK {
            kind: RouterKind::SigmoidBiasRenorm {
                eps: 1e-20,
                scale: cfg.route_scale
            },
            logits: &rows[11],
            correction_bias: layer.router_bias.as_ref(),
            topk_indices: &rows[12],
            topk_weights: &rows[13],
            rows: 1,
            n_experts: cfg.n_experts,
            top_k: cfg.n_expert_used,
        }),
        "router topk"
    );
    step!(cap_put(cap, gpu, "topk_idx", &rows[12]), "capture");
    step!(cap_put(cap, gpu, "topk_w", &rows[13]), "capture");
    // Typed XTS layers keep their per-expert AWQ producer and down scale.
    // Legacy layers use the load-resident uniform pointer tables.
    let down = if let Some(xl) = layer.xts.as_ref() {
        step!(
            xts_expert_stage(gpu, cfg, xl, &mut tmp, &rows[9], &rows[12], cap),
            "xts experts"
        )
    } else {
        step!(
            rotate_x_mq_for(gpu, &layer.experts[0].gate_up, &rows[9], &rows[10], e),
            "ffn rotate"
        );
        step!(cap_put(cap, gpu, "ffn_x_rot", &rows[10]), "capture");
        step!(
            moe_middle_single(gpu, cfg, layer, cap, &mut tmp, &rows[10], &rows[12]),
            "middle"
        )
    };
    let vo = match moe_postfix(
        gpu,
        cfg,
        layer,
        gs,
        visit,
        cap,
        &mut tmp,
        cur,
        down,
        &rows[12],
        &rows[13],
        &rows[14],
        push,
        push_normed,
    ) {
        Ok(v) => v,
        Err(e) => {
            free_all(gpu, rows);
            free_all(gpu, tmp);
            return Err(e);
        }
    };
    free_all(gpu, rows);
    Ok(vo)
}

// ---------------------------------------------------------------------------
// Token driver (mirrors forward.rs::forward_token)
// ---------------------------------------------------------------------------

/// Token-level owned tensors, freed on every path by [`drive`].
#[derive(Default)]
struct Owned {
    x: Option<GpuTensor>,
    blocks: Vec<GpuTensor>,
    /// D1 norm-once twins: `blocks_n[i]` is the RMSNorm of `blocks[i]`
    /// (same index, freed in lockstep). Mix calls read these via pointers.
    blocks_n: Vec<GpuTensor>,
    prelude_x: Option<GpuTensor>,
    prelude_blocks: Vec<GpuTensor>,
    prelude_blocks_n: Vec<GpuTensor>,
    prev_blocks: Vec<GpuTensor>,
    prev_blocks_n: Vec<GpuTensor>,
    prev_out: Option<GpuTensor>,
}

impl Owned {
    fn free(self, gpu: &mut Gpu) {
        if let Some(t) = self.x {
            let _ = gpu.free_tensor(t);
        }
        for t in self.blocks {
            let _ = gpu.free_tensor(t);
        }
        for t in self.blocks_n {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.prelude_x {
            let _ = gpu.free_tensor(t);
        }
        for t in self.prelude_blocks {
            let _ = gpu.free_tensor(t);
        }
        for t in self.prelude_blocks_n {
            let _ = gpu.free_tensor(t);
        }
        for t in self.prev_blocks {
            let _ = gpu.free_tensor(t);
        }
        for t in self.prev_blocks_n {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.prev_out {
            let _ = gpu.free_tensor(t);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn drive(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    gw: &Inkling2GpuWeights,
    gs: &mut Inkling2GpuState,
    token_id: u32,
    pos: usize,
    capture_visit: Option<usize>,
    cap: &mut Option<HashMap<String, Vec<f32>>>,
    trace: &mut Option<Vec<Vec<f32>>>,
) -> Result<Vec<f32>, String> {
    if gw.layers.len() != cfg.n_phys {
        return Err(format!(
            "inkling2 gpu fwd: {} layers != n_phys {}",
            gw.layers.len(),
            cfg.n_phys
        ));
    }
    if pos != gs.seq_pos {
        return Err(format!(
            "inkling2 gpu fwd: pos {pos} != seq_pos {} (out-of-order)",
            gs.seq_pos
        ));
    }
    ensure_kernels(gpu)?;
    let mut o = Owned::default();
    let r = drive_body(
        gpu,
        cfg,
        gw,
        gs,
        token_id,
        pos,
        capture_visit,
        cap,
        trace,
        &mut o,
    );
    // D1: advance the device pos word once per completed token (1-thread
    // kernel, 0 H2D). The slots kernel of the next token reads the new word.
    // A failed token leaves the word behind — but its KV writes are also
    // partial, so the session is already torn down by the caller.
    if r.is_ok() {
        if let Err(e) = launch_advance_pos(gpu, &gs.pos_dev) {
            o.free(gpu);
            return Err(format!("inkling2 gpu fwd: advance pos: {e}"));
        }
    }
    o.free(gpu);
    r
}

/// Run one visit, pushing `visit_out` to `trace` and moving the stream.
/// `take_msgs` builds the pass > 0 message tuple (or None).
/// `cap` is `Some` iff this is the captured visit (run_visit fills it).
#[allow(clippy::too_many_arguments)]
fn step_visit(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    gw: &Inkling2GpuWeights,
    gs: &mut Inkling2GpuState,
    o: &mut Owned,
    visit: usize,
    pos: usize,
    take_msgs: Option<(usize, usize)>,
    cap: &mut Option<HashMap<String, Vec<f32>>>,
    trace: &mut Option<Vec<Vec<f32>>>,
) -> Result<(), String> {
    let phys = cfg.loop_cfg.layer_phys(visit);
    let x = o.x.as_ref().expect("inkling2 gpu fwd: missing x");
    let msgs = match take_msgs {
        Some((core_i, _)) => {
            let skip = o.prelude_blocks.len() + 1;
            if o.prev_blocks.len() < skip {
                return Err(format!(
                    "inkling2 gpu fwd: pass prev_blocks {} < skip {skip}",
                    o.prev_blocks.len()
                ));
            }
            // D1: normed twins slice in lockstep with the blocks.
            assert_eq!(
                o.prev_blocks.len(),
                o.prev_blocks_n.len(),
                "inkling2 gpu fwd: prev blocks/normed split"
            );
            let ctrls = match &gw.layers[phys].controllers {
                Some(c) => c,
                None => {
                    return Err(format!(
                        "inkling2 gpu fwd: core layer {phys} missing controllers"
                    ))
                }
            };
            Some((
                // The bank is the previous pass's blocks after the prelude and
                // the first core block (`blocks[prelude_len + 1..]`), as in the
                // CPU reference; the mix appends `prev_out`.
                &o.prev_blocks[skip..],
                &o.prev_blocks_n[skip..],
                o.prev_out.as_ref().expect("prev_out"),
                &gw.cross_pass[core_i],
                ctrls,
            ))
        }
        None => None,
    };
    let vo = run_visit(
        gpu,
        cfg,
        &gw.layers[phys],
        x,
        &o.blocks,
        &o.blocks_n,
        visit,
        phys,
        pos,
        msgs,
        gs,
        cap,
    )?;
    if let Some(t) = trace.as_mut() {
        match dl(gpu, &vo.visit_out) {
            Ok(v) => t.push(v),
            Err(e) => {
                let _ = gpu.free_tensor(vo.visit_out);
                if let Some(p) = vo.push {
                    let _ = gpu.free_tensor(p);
                }
                if let Some(p) = vo.push_normed {
                    let _ = gpu.free_tensor(p);
                }
                return Err(e);
            }
        }
    }
    if let Some(p) = vo.push {
        o.blocks.push(p);
    }
    if let Some(p) = vo.push_normed {
        o.blocks_n.push(p);
    }
    let x = o.x.take().expect("inkling2 gpu fwd: missing x");
    if let Err(e) = gpu.free_tensor(x) {
        let _ = gpu.free_tensor(vo.visit_out);
        return Err(format!("inkling2 gpu fwd: x free: {e:?}"));
    }
    o.x = Some(vo.visit_out);
    Ok(())
}

/// Capture-swap helper: run one loop iteration capturing only when
/// `capture_visit == visit`, merging the map into `cap`.
#[allow(clippy::too_many_arguments)]
fn step_capture(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    gw: &Inkling2GpuWeights,
    gs: &mut Inkling2GpuState,
    o: &mut Owned,
    visit: usize,
    pos: usize,
    take_msgs: Option<(usize, usize)>,
    capture_visit: Option<usize>,
    cap: &mut Option<HashMap<String, Vec<f32>>>,
    trace: &mut Option<Vec<Vec<f32>>>,
) -> Result<(), String> {
    let mut one: Option<HashMap<String, Vec<f32>>> = if capture_visit == Some(visit) {
        Some(HashMap::new())
    } else {
        None
    };
    step_visit(gpu, cfg, gw, gs, o, visit, pos, take_msgs, &mut one, trace)?;
    if let Some(m) = one {
        *cap = Some(m);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn drive_body(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    gw: &Inkling2GpuWeights,
    gs: &mut Inkling2GpuState,
    token_id: u32,
    pos: usize,
    capture_visit: Option<usize>,
    cap: &mut Option<HashMap<String, Vec<f32>>>,
    trace: &mut Option<Vec<Vec<f32>>>,
    o: &mut Owned,
) -> Result<Vec<f32>, String> {
    let e = cfg.n_embd;
    let core_len = cfg.loop_cfg.core_end - cfg.loop_cfg.core_start;
    if gw.cross_pass.len() != core_len {
        return Err(format!(
            "inkling2 gpu fwd: {} cross_pass != core_len {core_len}",
            gw.cross_pass.len()
        ));
    }
    // Embed: host row decode (packed qt44 stays on the host by 01 design),
    // then the token RMSNorm on the device.
    let row = gw.embed_packed.row_f32(token_id)?;
    let emb = hip(gpu.upload_f32(&row, &[e]), "embed upload")?;
    let tok = match gpu.alloc_tensor(&[e], DType::F32) {
        Ok(t) => t,
        Err(e2) => {
            let _ = gpu.free_tensor(emb);
            return Err(format!("inkling2 gpu fwd: tok: {e2:?}"));
        }
    };
    if let Err(e2) = gpu.rmsnorm_f32(&emb, &gw.embed_norm, &tok, cfg.rms_norm_eps) {
        let _ = gpu.free_tensor(emb);
        let _ = gpu.free_tensor(tok);
        return Err(format!("inkling2 gpu fwd: embed norm: {e2:?}"));
    }
    if let Err(e2) = gpu.free_tensor(emb) {
        let _ = gpu.free_tensor(tok);
        return Err(format!("inkling2 gpu fwd: embed free: {e2:?}"));
    }
    o.x = Some(tok);
    let mut visit = 0usize;
    // Prelude visits 0..core_start.
    for _ in 0..cfg.loop_cfg.core_start {
        step_capture(
            gpu,
            cfg,
            gw,
            gs,
            o,
            visit,
            pos,
            None,
            capture_visit,
            cap,
            trace,
        )?;
        visit += 1;
    }
    // Snapshot prelude stream for the pass resets.
    o.prelude_x = Some(dup(gpu, o.x.as_ref().expect("x"), "prelude_x")?);
    // D1: drain blocks and their normed twins in lockstep (same order).
    assert_eq!(
        o.blocks.len(),
        o.blocks_n.len(),
        "inkling2 gpu fwd: prelude drain split"
    );
    for (b, bn) in o.blocks.drain(..).zip(o.blocks_n.drain(..)) {
        let d = dup(gpu, &b, "prelude_block")?;
        o.prelude_blocks.push(d);
        let dn = dup(gpu, &bn, "prelude_block_n")?;
        o.prelude_blocks_n.push(dn);
        if let Err(e) = gpu.free_tensor(b) {
            let _ = gpu.free_tensor(bn);
            return Err(format!("inkling2 gpu fwd: block free: {e:?}"));
        }
        if let Err(e) = gpu.free_tensor(bn) {
            return Err(format!("inkling2 gpu fwd: block_n free: {e:?}"));
        }
    }
    // prev_* seeds pass 1's bank (pass 0 itself takes no messages).
    for (b, bn) in o.prelude_blocks.iter().zip(o.prelude_blocks_n.iter()) {
        o.prev_blocks.push(dup(gpu, b, "prev_block")?);
        o.prev_blocks_n.push(dup(gpu, bn, "prev_block_n")?);
    }
    o.prev_out = Some(dup(gpu, o.prelude_x.as_ref().expect("x"), "prev_out")?);
    // Loop passes.
    for pass in 0..cfg.loop_cfg.num_passes {
        // Reset the stream to the prelude output.
        if let Some(x) = o.x.take() {
            hip(gpu.free_tensor(x), "x free")?;
        }
        assert_eq!(
            o.blocks.len(),
            o.blocks_n.len(),
            "inkling2 gpu fwd: pass drain split"
        );
        for (b, bn) in o.blocks.drain(..).zip(o.blocks_n.drain(..)) {
            hip(gpu.free_tensor(b), "block free")?;
            hip(gpu.free_tensor(bn), "block_n free")?;
        }
        o.x = Some(dup(gpu, o.prelude_x.as_ref().expect("x"), "pass x")?);
        for (b, bn) in o.prelude_blocks.iter().zip(o.prelude_blocks_n.iter()) {
            o.blocks.push(dup(gpu, b, "pass block")?);
            o.blocks_n.push(dup(gpu, bn, "pass block_n")?);
        }
        for i in 0..core_len {
            let take = if pass > 0 { Some((i, 0)) } else { None };
            step_capture(
                gpu,
                cfg,
                gw,
                gs,
                o,
                visit,
                pos,
                take,
                capture_visit,
                cap,
                trace,
            )?;
            visit += 1;
        }
        // prev_* for the next pass (dead after the last one — skip it).
        if pass + 1 < cfg.loop_cfg.num_passes {
            assert_eq!(
                o.prev_blocks.len(),
                o.prev_blocks_n.len(),
                "inkling2 gpu fwd: prev drain split"
            );
            for (b, bn) in o.prev_blocks.drain(..).zip(o.prev_blocks_n.drain(..)) {
                hip(gpu.free_tensor(b), "prev block free")?;
                hip(gpu.free_tensor(bn), "prev block_n free")?;
            }
            for (b, bn) in o.blocks.iter().zip(o.blocks_n.iter()) {
                o.prev_blocks.push(dup(gpu, b, "prev block")?);
                o.prev_blocks_n.push(dup(gpu, bn, "prev block_n")?);
            }
            if let Some(p) = o.prev_out.take() {
                hip(gpu.free_tensor(p), "prev out free")?;
            }
            o.prev_out = Some(dup(gpu, o.x.as_ref().expect("x"), "prev out")?);
        }
    }
    // Post-loop layers core_end..n_phys once (stream continues).
    for _ in cfg.loop_cfg.core_end..cfg.n_phys {
        step_capture(
            gpu,
            cfg,
            gw,
            gs,
            o,
            visit,
            pos,
            None,
            capture_visit,
            cap,
            trace,
        )?;
        visit += 1;
    }
    assert_eq!(
        visit,
        cfg.loop_cfg.n_slot(),
        "inkling2 gpu fwd: visit count"
    );
    // Final: mix the collected blocks with the output query, final RMSNorm.
    drive_final(gpu, cfg, gw, gs, o, pos)
}

/// Output tail for one finished stream: final mix, norm, LM head, softcap.
/// Shared by the single-row loop and the blocked driver (which runs it only
/// for the prompt-final row). Consumes `o.x`; advances the session cursor.
fn drive_final(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    gw: &Inkling2GpuWeights,
    gs: &mut Inkling2GpuState,
    o: &mut Owned,
    pos: usize,
) -> Result<Vec<f32>, String> {
    let e = cfg.n_embd;
    let mut ftmp = Vec::new();
    let ffail = |gpu: &mut Gpu, ftmp: Vec<GpuTensor>, msg: String| -> Result<Vec<f32>, String> {
        free_all(gpu, ftmp);
        Err(msg)
    };
    if gw.output_query.buf.size() / 4 != e {
        return Err(format!(
            "inkling2 gpu fwd: output_query len {} != {e}",
            gw.output_query.buf.size() / 4
        ));
    }
    let x = o.x.take().expect("x");
    assert_eq!(
        o.blocks.len(),
        o.blocks_n.len(),
        "inkling2 gpu fwd: final blocks/normed split"
    );
    let block_n: Vec<&GpuTensor> = o.blocks_n.iter().collect();
    let mixed = match gpu_mix(
        gpu,
        &gs.scratch,
        &block_n,
        &x,
        &gw.output_query,
        e,
        1,
        cfg.rms_norm_eps,
    ) {
        Ok(t) => t,
        Err(msg) => {
            let _ = gpu.free_tensor(x);
            return ffail(gpu, ftmp, msg);
        }
    };
    if let Err(e2) = gpu.free_tensor(x) {
        let _ = gpu.free_tensor(mixed);
        return ffail(gpu, ftmp, format!("inkling2 gpu fwd: final x free: {e2:?}"));
    }
    ftmp.push(mixed);
    let mi = ftmp.len() - 1;
    let final_row = match at(gpu, &mut ftmp, &[e], "final") {
        Ok(i) => i,
        Err(msg) => return ffail(gpu, ftmp, msg),
    };
    if let Err(e2) = gpu.copy_rows_strided_f32(&ftmp[mi], &ftmp[final_row], 1, e, e, e, 0, None) {
        return ffail(gpu, ftmp, format!("inkling2 gpu fwd: final row: {e2:?}"));
    }
    let hidden = match at(gpu, &mut ftmp, &[e], "hidden") {
        Ok(i) => i,
        Err(msg) => return ffail(gpu, ftmp, msg),
    };
    if let Err(e2) = gpu.rmsnorm_f32(&ftmp[final_row], &gw.norm, &ftmp[hidden], cfg.rms_norm_eps) {
        return ffail(gpu, ftmp, format!("inkling2 gpu fwd: final norm: {e2:?}"));
    }
    let logits_full = match gpu.alloc_tensor(&[cfg.vocab_size], DType::F32) {
        Ok(t) => t,
        Err(e2) => return ffail(gpu, ftmp, format!("inkling2 gpu fwd: logits alloc: {e2:?}")),
    };
    if let Err(e2) = weight_gemv(gpu, &gw.unembed, &ftmp[hidden], &logits_full) {
        let _ = gpu.free_tensor(logits_full);
        return ffail(gpu, ftmp, format!("inkling2 gpu fwd: unembed: {e2:?}"));
    }
    if let Err(e2) = gpu.logit_softcap_f32(&logits_full, cfg.vocab_size, cfg.softcap) {
        let _ = gpu.free_tensor(logits_full);
        return ffail(gpu, ftmp, format!("inkling2 gpu fwd: softcap: {e2:?}"));
    }
    let full = match dl(gpu, &logits_full) {
        Ok(v) => v,
        Err(msg) => {
            let _ = gpu.free_tensor(logits_full);
            return ffail(gpu, ftmp, msg);
        }
    };
    let _ = gpu.free_tensor(logits_full);
    free_all(gpu, ftmp);
    gs.seq_pos = pos + 1;
    Ok(full[..cfg.unpadded_vocab_size].to_vec())
}

// ---------------------------------------------------------------------------
// Public entry points (brief 05 calls forward_token_gpu)
// ---------------------------------------------------------------------------

/// One full token on the GPU: 48 visits, final mix + norm, LM head, softcap.
/// Returns host logits of length `unpadded_vocab_size`. `pos` must equal the
/// session cursor (out-of-order calls fail closed); the cursor advances by 1.
pub fn forward_token_gpu(
    bundle: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    token_id: u32,
    pos: usize,
) -> Result<Vec<f32>, String> {
    let cfg = bundle.config.clone();
    let gw = bundle
        .gpu_weights
        .as_ref()
        .ok_or("inkling2 gpu fwd: no gpu weights")?;
    let gs = bundle
        .gpu_state
        .as_mut()
        .ok_or("inkling2 gpu fwd: no gpu state")?;
    let mut cap: Option<HashMap<String, Vec<f32>>> = None;
    let mut trace: Option<Vec<Vec<f32>>> = None;
    drive(gpu, &cfg, gw, gs, token_id, pos, None, &mut cap, &mut trace)
}

/// One full token, downloading every visit's `visit_out` (48 × n_embd).
/// Gate/debug helper; same numerics as [`forward_token_gpu`].
pub fn forward_token_gpu_trace(
    bundle: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    token_id: u32,
    pos: usize,
) -> Result<(Vec<f32>, Vec<Vec<f32>>), String> {
    let cfg = bundle.config.clone();
    let gw = bundle
        .gpu_weights
        .as_ref()
        .ok_or("inkling2 gpu fwd: no gpu weights")?;
    let gs = bundle
        .gpu_state
        .as_mut()
        .ok_or("inkling2 gpu fwd: no gpu state")?;
    let mut cap: Option<HashMap<String, Vec<f32>>> = None;
    let mut trace: Option<Vec<Vec<f32>>> = Some(Vec::new());
    let logits = drive(gpu, &cfg, gw, gs, token_id, pos, None, &mut cap, &mut trace)?;
    Ok((logits, trace.expect("trace")))
}

/// One full token, downloading one visit's intermediates (CPU-debug keys).
/// Gate/debug helper; the full token still runs (state evolves per visit).
pub fn forward_token_gpu_capture(
    bundle: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    token_id: u32,
    pos: usize,
    visit: usize,
) -> Result<HashMap<String, Vec<f32>>, String> {
    let cfg = bundle.config.clone();
    let gw = bundle
        .gpu_weights
        .as_ref()
        .ok_or("inkling2 gpu fwd: no gpu weights")?;
    let gs = bundle
        .gpu_state
        .as_mut()
        .ok_or("inkling2 gpu fwd: no gpu state")?;
    let mut cap: Option<HashMap<String, Vec<f32>>> = Some(HashMap::new());
    let mut trace: Option<Vec<Vec<f32>>> = None;
    drive(
        gpu,
        &cfg,
        gw,
        gs,
        token_id,
        pos,
        Some(visit),
        &mut cap,
        &mut trace,
    )?;
    Ok(cap.expect("cap"))
}

// ---------------------------------------------------------------------------
// F1 blocked prefill (beta port): chunked multi-row visit driver.
// ---------------------------------------------------------------------------
//
// One block of B prompt tokens runs each of the 48 visits width-first:
// heads write normed inputs into stack rows, one batched GEMM serves each
// projection (QKV/R via kernel #1, F16-out + widen-step), one qtile launch
// serves attention (kernel #3), one batched GEMM each serves wgate/wo, the
// router stays per-row F32 (routing-identical) with a batched topk. Every
// other op runs per row in position order (conv/state/KV/tags/pos, MoE
// experts, combine), exactly like the single-row loop. B == 1 never enters
// this path, so the fallback stays bit-identical to the eager loop by
// construction. B > 1 gates on the agreement bar (argmax + KL <= 0.005),
// not 1e-5: F16 rounding in the widen-step is inherent, not kernel error.
//
// Beta adaptations vs the stack's F1: F16 KV only (no Q8 twin — beta has no
// Q8 KV path), no grouped MoE (per-row experts — grouped never fires on the
// uniform-qt44/qt13 fixture anyway), beta clamp rel-bias (NOT the stack's
// zero-term rule — that belongs to the separate rel-bias PR), pool temps
// (no L7 scratch wiring), direct launches (no blob keep-alive store).

/// Chunk size for the blocked entry point (matches the stack's gate fixture).
pub const INKLING2_PREFILL_CHUNK: usize = 512;

/// Block schedule: `n` prompt tokens in `chunk`-token blocks (last one
/// short). Pure helper, host-tested below.
pub fn prefill_blocks(n: usize, chunk: usize) -> Vec<(usize, usize)> {
    let chunk = if chunk == 0 {
        INKLING2_PREFILL_CHUNK
    } else {
        chunk
    };
    let mut out = Vec::new();
    let mut start = 0;
    while start < n {
        let len = (n - start).min(chunk);
        out.push((start, len));
        start += len;
    }
    out
}

/// Master switch for the batched QKV/R projections (kernel #1).
const BATCHED_QKVR_GATED: bool = true;

/// Master switch for the batched qtile attention (kernel #3).
const BATCHED_ATTN_GATED: bool = true;

/// Router batching stays OFF: the router GEMV is ~0.6% of tail bytes and
/// batching it risks routing flips for negligible gain. Per-row F32 router
/// is routing-identical to the seam; topk still batches over rows.
const BATCHED_ROUTER_GEMM: bool = false;

/// LUT for one blocked projection: the weight's own Lloyd codebook, or the
/// identity codebook for uniform MQ4G256V2 (this artifact's dense dtype --
/// a bit-exact twin of the uniform kernel). Anything else fails closed.
fn blocked_proj_lut(w: &WeightTensor) -> Result<[u32; 8], String> {
    match w.gpu_dtype {
        DType::MQ4G256V2Lloyd => w
            .lloyd_lut_f16
            .ok_or_else(|| "inkling2 gpu fwd: blocked prefill needs a Lloyd LUT".to_string()),
        DType::MQ4G256V2 => Ok(rdna_compute::inkling2_prefill_gemm::inkling2_identity_lut()),
        other => Err(format!(
            "inkling2 gpu fwd: blocked prefill unsupported dtype {other:?}"
        )),
    }
}

/// Per-op eligibility for the batched tail GEMMs (wgate/wo/router).
/// Same admission as the QKV/R path: Lloyd+LUT or uniform-identity;
/// anything else fails closed to the single-row GEMV. `m`/`k` pin the
/// expected weight shape.
fn tail_gemm_ok(w: &WeightTensor, m: usize, k: usize) -> Option<[u32; 8]> {
    if w.m != m || w.k != k {
        return None;
    }
    blocked_proj_lut(w).ok()
}

/// Non-owning F32 row view of a `[rows, n]` stack (never freed; the driver
/// frees the stack once per block). Short stacks fail closed with `Err`.
fn stack_row(stack: &GpuTensor, row: usize, n: usize) -> Result<GpuTensor, String> {
    let stride = n
        .checked_mul(4)
        .ok_or_else(|| "inkling2 gpu fwd: stack_row extent overflows".to_string())?;
    let off = row
        .checked_mul(stride)
        .ok_or_else(|| format!("inkling2 gpu fwd: stack_row row {row}*{n} overflows"))?;
    if stack.dtype != DType::F32 {
        return Err("inkling2 gpu fwd: stack_row of non-F32".to_string());
    }
    if stack.buf.size() < off + stride {
        return Err(format!(
            "inkling2 gpu fwd: stack_row [{off}, {}) exceeds {} B",
            off + stride,
            stack.buf.size()
        ));
    }
    Ok(GpuTensor {
        buf: stack.buf.byte_view(off, stride),
        shape: vec![n],
        dtype: DType::F32,
    })
}

/// Non-owning F16 view of one head's `[b, 128]` slab of a `[b, 8*128]`
/// snapshot stack (never freed; the driver frees the stack once per block).
fn snap_head(stack: &GpuTensor, kh: usize, b: usize) -> Result<GpuTensor, String> {
    let slab = b
        .checked_mul(128 * 2)
        .ok_or_else(|| "inkling2 gpu fwd: snap_head extent overflows".to_string())?;
    let off = kh
        .checked_mul(slab)
        .ok_or_else(|| format!("inkling2 gpu fwd: snap_head head {kh} overflows"))?;
    if stack.dtype != DType::F16 {
        return Err("inkling2 gpu fwd: snap_head of non-F16".to_string());
    }
    if stack.buf.size() < off + slab {
        return Err(format!(
            "inkling2 gpu fwd: snap_head [{off}, {}) exceeds {} B",
            off + slab,
            stack.buf.size()
        ));
    }
    Ok(GpuTensor {
        buf: stack.buf.byte_view(off, slab),
        shape: vec![b, 128],
        dtype: DType::F16,
    })
}

/// All eight F16 snapshot head views of a `[b, 8*128]` stack, head-major.
/// One fail-closed cut instead of eight panicking ones at every call site.
fn snap_heads(stack: &GpuTensor, b: usize) -> Result<[GpuTensor; 8], String> {
    Ok([
        snap_head(stack, 0, b)?,
        snap_head(stack, 1, b)?,
        snap_head(stack, 2, b)?,
        snap_head(stack, 3, b)?,
        snap_head(stack, 4, b)?,
        snap_head(stack, 5, b)?,
        snap_head(stack, 6, b)?,
        snap_head(stack, 7, b)?,
    ])
}

/// Pre-overwrite evict range for an SWA-wrap block: the block
/// `[start, start+b)` evicts ring rows whose positions fall in
/// `EV = [start-rows, start+b-rows)`; a batched launch running after the
/// block's KV writes must read those keys from a snapshot taken before the
/// writes. Returns `None` when the block cannot batch (multi-wrap `b > rows`
/// or a logic error `ev_hi > start`): the caller keeps the single-row path.
/// The caller only invokes this when the block wraps (`start + b > rows`);
/// on a non-wrapping block the range is empty and `None` is returned.
fn swa_snapshot_range(start: usize, b: usize, rows: usize) -> Option<(usize, usize)> {
    if b == 0 || b > rows {
        return None;
    }
    let ev_lo = start.saturating_sub(rows);
    let ev_hi = start.saturating_add(b).saturating_sub(rows);
    if ev_hi <= ev_lo || ev_hi > start {
        return None;
    }
    Some((ev_lo, ev_hi))
}

/// Blocked-visit projection stacks, allocated once per block and shared by
/// every visit (shapes are config-fixed). QKV/R half: norm/rotation stacks
/// plus the F16 GEMM outputs and their F32 widened twins. MoE half: FFN
/// rotation rows, router outputs, topk tables and the MoE output rows.
struct BlockStacks {
    qi: GpuTensor,
    ki: GpuTensor,
    vi: GpuTensor,
    rot_q: GpuTensor,
    rot_k: GpuTensor,
    rot_v: GpuTensor,
    q16: GpuTensor,
    k16: GpuTensor,
    v16: GpuTensor,
    r16: GpuTensor,
    q32: GpuTensor,
    k32: GpuTensor,
    v32: GpuTensor,
    r32: GpuTensor,
    sinks: GpuTensor,
    attn_out: GpuTensor,
    gate16: GpuTensor,
    gate32: GpuTensor,
    gate_rot: GpuTensor,
    wo_rot: GpuTensor,
    router_rot: GpuTensor,
    wo16: GpuTensor,
    wo32: GpuTensor,
    fnorm: GpuTensor,
    sk_snap: GpuTensor,
    sv_snap: GpuTensor,
    ffn_rot: GpuTensor,
    rout16: GpuTensor,
    rout32: GpuTensor,
    topk_idx: GpuTensor,
    topk_w: GpuTensor,
    mmoe: GpuTensor,
}

impl BlockStacks {
    #[allow(clippy::too_many_arguments)]
    fn alloc(
        gpu: &mut Gpu,
        b: usize,
        e: usize,
        m_q: usize,
        m_kv: usize,
        m_r: usize,
        n_head: usize,
        n_exp: usize,
        n_topk: usize,
    ) -> Result<Self, String> {
        let e2 = e;
        let specs: [(Vec<usize>, DType, &str); 32] = [
            (vec![b, e2], DType::F32, "qi"),
            (vec![b, e2], DType::F32, "ki"),
            (vec![b, e2], DType::F32, "vi"),
            (vec![b, e2], DType::F32, "rot_q"),
            (vec![b, e2], DType::F32, "rot_k"),
            (vec![b, e2], DType::F32, "rot_v"),
            (vec![b, m_q], DType::F16, "q16"),
            (vec![b, m_kv], DType::F16, "k16"),
            (vec![b, m_kv], DType::F16, "v16"),
            (vec![b, m_r], DType::F16, "r16"),
            (vec![b, m_q], DType::F32, "q32"),
            (vec![b, m_kv], DType::F32, "k32"),
            (vec![b, m_kv], DType::F32, "v32"),
            (vec![b, m_r], DType::F32, "r32"),
            (vec![b, n_head], DType::F32, "sinks"),
            (vec![b, m_q], DType::F32, "attn_out"),
            (vec![b, m_q], DType::F16, "gate16"),
            (vec![b, m_q], DType::F32, "gate32"),
            (vec![b, e2], DType::F32, "gate_rot"),
            (vec![b, m_q], DType::F32, "wo_rot"),
            (vec![b, e2], DType::F32, "router_rot"),
            (vec![b, e2], DType::F16, "wo16"),
            (vec![b, e2], DType::F32, "wo32"),
            (vec![b, e2], DType::F32, "fnorm"),
            (vec![b, 8 * 128], DType::F16, "sk_snap"),
            (vec![b, 8 * 128], DType::F16, "sv_snap"),
            (vec![b, e2], DType::F32, "ffn_rot"),
            (vec![b, n_exp], DType::F16, "rout16"),
            (vec![b, n_exp], DType::F32, "rout32"),
            (vec![b, n_topk], DType::F32, "topk_idx"),
            (vec![b, n_topk], DType::F32, "topk_w"),
            (vec![b, e2], DType::F32, "mmoe"),
        ];
        // Topk tables carry int bits in F32 buffers (same as the seam's
        // `at()` F32 allocs); kernels read raw words.
        let mut ts: Vec<GpuTensor> = Vec::with_capacity(specs.len());
        for (shape, dt, what) in specs {
            match gpu.alloc_tensor(&shape, dt) {
                Ok(t) => ts.push(t),
                Err(e2) => {
                    free_all(gpu, ts);
                    return Err(format!("inkling2 gpu fwd: block stack {what}: {e2:?}"));
                }
            }
        }
        let mut it = ts.into_iter();
        let next = |it: &mut std::vec::IntoIter<GpuTensor>| it.next().expect("stacks");
        Ok(Self {
            qi: next(&mut it),
            ki: next(&mut it),
            vi: next(&mut it),
            rot_q: next(&mut it),
            rot_k: next(&mut it),
            rot_v: next(&mut it),
            q16: next(&mut it),
            k16: next(&mut it),
            v16: next(&mut it),
            r16: next(&mut it),
            q32: next(&mut it),
            k32: next(&mut it),
            v32: next(&mut it),
            r32: next(&mut it),
            sinks: next(&mut it),
            attn_out: next(&mut it),
            gate16: next(&mut it),
            gate32: next(&mut it),
            gate_rot: next(&mut it),
            wo_rot: next(&mut it),
            router_rot: next(&mut it),
            wo16: next(&mut it),
            wo32: next(&mut it),
            fnorm: next(&mut it),
            sk_snap: next(&mut it),
            sv_snap: next(&mut it),
            ffn_rot: next(&mut it),
            rout16: next(&mut it),
            rout32: next(&mut it),
            topk_idx: next(&mut it),
            topk_w: next(&mut it),
            mmoe: next(&mut it),
        })
    }

    fn free(self, gpu: &mut Gpu) {
        let Self {
            qi,
            ki,
            vi,
            rot_q,
            rot_k,
            rot_v,
            q16,
            k16,
            v16,
            r16,
            q32,
            k32,
            v32,
            r32,
            sinks,
            attn_out,
            gate16,
            gate32,
            gate_rot,
            wo_rot,
            router_rot,
            wo16,
            wo32,
            fnorm,
            sk_snap,
            sv_snap,
            ffn_rot,
            rout16,
            rout32,
            topk_idx,
            topk_w,
            mmoe,
        } = self;
        for t in [
            qi, ki, vi, rot_q, rot_k, rot_v, q16, k16, v16, r16, q32, k32, v32, r32, sinks,
            attn_out, gate16, gate32, gate_rot, wo_rot, router_rot, wo16, wo32, fnorm, sk_snap,
            sv_snap, ffn_rot, rout16, rout32, topk_idx, topk_w, mmoe,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// The four QKV/R projections over a whole block: one batched rotation per
/// norm stack (`wr` shares `qi`'s input and rotation with `wq`), one
/// batched F16 GEMM per weight, one exact widen per output. Downstream
/// stays F32 (widen-step).
fn blocked_project(
    gpu: &mut Gpu,
    layer: &Inkling2GpuLayer,
    e: usize,
    b: usize,
    stacks: &BlockStacks,
) -> Result<(), String> {
    // Fail closed on ragged K before touching the stacks (wr shares qi's
    // input and rotation with wq; three rotations total below).
    for (w, what) in [&layer.wq, &layer.wk, &layer.wv, &layer.wr]
        .iter()
        .zip(["wq", "wk", "wv", "wr"])
    {
        if w.k != e {
            return Err(format!("inkling2 gpu fwd: block {what} k {} != {e}", w.k));
        }
    }
    hip(
        rotate_x_mq_batched_for(gpu, &layer.wq, &stacks.qi, &stacks.rot_q, e, b),
        "block rotate q",
    )?;
    hip(
        rotate_x_mq_batched_for(gpu, &layer.wk, &stacks.ki, &stacks.rot_k, e, b),
        "block rotate k",
    )?;
    hip(
        rotate_x_mq_batched_for(gpu, &layer.wv, &stacks.vi, &stacks.rot_v, e, b),
        "block rotate v",
    )?;
    let luts = [
        blocked_proj_lut(&layer.wq)?,
        blocked_proj_lut(&layer.wk)?,
        blocked_proj_lut(&layer.wv)?,
        blocked_proj_lut(&layer.wr)?,
    ];
    let projs = [
        (&layer.wq, &stacks.rot_q, &stacks.q16, &stacks.q32, "wq"),
        (&layer.wk, &stacks.rot_k, &stacks.k16, &stacks.k32, "wk"),
        (&layer.wv, &stacks.rot_v, &stacks.v16, &stacks.v32, "wv"),
        (&layer.wr, &stacks.rot_q, &stacks.r16, &stacks.r32, "wr"),
    ];
    for (i, (w, xr, y16, y32, what)) in projs.iter().enumerate() {
        if w.k != e {
            return Err(format!("inkling2 gpu fwd: block {what} k {} != {e}", w.k));
        }
        hip(
            gpu.inkling2_prefill_gemm_mq4v2_lloyd_f16out(&w.buf, xr, y16, w.m, e, b, luts[i]),
            *what,
        )?;
        hip(gpu.inkling2_widen_f16_f32(y16, y32, b * w.m), "block widen")?;
    }
    Ok(())
}

/// Batched output-gate projection over the whole block. `qi` rows are ready
/// (blocked_project ran first). Eligible weights take one batched GEMM +
/// widen into the `gate32` stack; anything else falls back to per-row GEMVs
/// into the same stack rows (uniform downstream either way).
fn blocked_gate_values(
    gpu: &mut Gpu,
    layer: &Inkling2GpuLayer,
    stacks: &BlockStacks,
    b: usize,
    e: usize,
    eattn: usize,
) -> Result<(), String> {
    match tail_gemm_ok(&layer.wgate, eattn, e) {
        Some(lut) => {
            // FWHT-rotate first: the batch GEMM reads rotated rows (same
            // contract as the QKV path; the seam's weight_gemv rotates
            // internally, so skipping this is silent corruption).
            hip(
                rotate_x_mq_batched_for(gpu, &layer.wgate, &stacks.qi, &stacks.gate_rot, e, b),
                "block rotate gate",
            )?;
            hip(
                gpu.inkling2_prefill_gemm_mq4v2_lloyd_f16out(
                    &layer.wgate.buf,
                    &stacks.gate_rot,
                    &stacks.gate16,
                    eattn,
                    e,
                    b,
                    lut,
                ),
                "block wgate",
            )?;
            hip(
                gpu.inkling2_widen_f16_f32(&stacks.gate16, &stacks.gate32, b * eattn),
                "block widen gate",
            )?;
        }
        None => {
            for r in 0..b {
                let qi_v = stack_row(&stacks.qi, r, e)?;
                let g_v = stack_row(&stacks.gate32, r, eattn)?;
                hip(
                    weight_gemv(gpu, &layer.wgate, &qi_v, &g_v),
                    "block wgate row",
                )?;
            }
        }
    }
    Ok(())
}

/// Batched output projection over the whole block. Reads the gated
/// attention rows (the gate loop ran first, in place over `attn_out`) and
/// writes the `wo32` stack (F16 GEMM + widen, or per-row GEMV fallback).
fn blocked_wo_values(
    gpu: &mut Gpu,
    layer: &Inkling2GpuLayer,
    stacks: &BlockStacks,
    b: usize,
    e: usize,
    eattn: usize,
) -> Result<(), String> {
    match tail_gemm_ok(&layer.wo, e, eattn) {
        Some(lut) => {
            hip(
                rotate_x_mq_batched_for(gpu, &layer.wo, &stacks.attn_out, &stacks.wo_rot, eattn, b),
                "block rotate wo",
            )?;
            hip(
                gpu.inkling2_prefill_gemm_mq4v2_lloyd_f16out(
                    &layer.wo.buf,
                    &stacks.wo_rot,
                    &stacks.wo16,
                    e,
                    eattn,
                    b,
                    lut,
                ),
                "block wo",
            )?;
            hip(
                gpu.inkling2_widen_f16_f32(&stacks.wo16, &stacks.wo32, b * e),
                "block widen wo",
            )?;
        }
        None => {
            for r in 0..b {
                let a_v = stack_row(&stacks.attn_out, r, eattn)?;
                let o_v = stack_row(&stacks.wo32, r, e)?;
                hip(weight_gemv(gpu, &layer.wo, &a_v, &o_v), "block wo row")?;
            }
        }
    }
    Ok(())
}

/// Batched router over the whole block. With `BATCHED_ROUTER_GEMM` off (the
/// shipped setting) every row takes the per-row F32 GEMV into its `rout32`
/// row plus a per-row topk — routing-identical to the seam at negligible
/// cost (the router GEMV is ~0.6% of tail bytes).
fn blocked_router_values(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    layer: &Inkling2GpuLayer,
    stacks: &BlockStacks,
    b: usize,
    e: usize,
) -> Result<(), String> {
    let nexp = cfg.n_experts;
    let ntopk = cfg.n_expert_used;
    // See BATCHED_ROUTER_GEMM: the F32 fallback is routing-identical.
    let lut_opt = if BATCHED_ROUTER_GEMM {
        tail_gemm_ok(&layer.router, nexp, e)
    } else {
        None
    };
    match lut_opt {
        Some(lut) => {
            hip(
                rotate_x_mq_batched_for(
                    gpu,
                    &layer.router,
                    &stacks.fnorm,
                    &stacks.router_rot,
                    e,
                    b,
                ),
                "block rotate router",
            )?;
            hip(
                gpu.inkling2_prefill_gemm_mq4v2_lloyd_f16out(
                    &layer.router.buf,
                    &stacks.router_rot,
                    &stacks.rout16,
                    nexp,
                    e,
                    b,
                    lut,
                ),
                "block router",
            )?;
            hip(
                gpu.inkling2_widen_f16_f32(&stacks.rout16, &stacks.rout32, b * nexp),
                "block widen router",
            )?;
            hip(
                gpu.ffn_moe_router_topk_f32(&MoeRouterTopK {
                    kind: RouterKind::SigmoidBiasRenorm {
                        eps: 1e-20,
                        scale: cfg.route_scale,
                    },
                    logits: &stacks.rout32,
                    correction_bias: layer.router_bias.as_ref(),
                    topk_indices: &stacks.topk_idx,
                    topk_weights: &stacks.topk_w,
                    rows: b,
                    n_experts: nexp,
                    top_k: ntopk,
                }),
                "block router topk",
            )?;
        }
        None => {
            for r in 0..b {
                let n_v = stack_row(&stacks.fnorm, r, e)?;
                let idx_v = stack_row(&stacks.topk_idx, r, ntopk)?;
                let w_v = stack_row(&stacks.topk_w, r, ntopk)?;
                // Per-row logits still need a home: reuse the rout32 row.
                let l_v = stack_row(&stacks.rout32, r, nexp)?;
                hip(
                    weight_gemv(gpu, &layer.router, &n_v, &l_v),
                    "block router row",
                )?;
                hip(
                    gpu.ffn_moe_router_topk_f32(&MoeRouterTopK {
                        kind: RouterKind::SigmoidBiasRenorm {
                            eps: 1e-20,
                            scale: cfg.route_scale,
                        },
                        logits: &l_v,
                        correction_bias: layer.router_bias.as_ref(),
                        topk_indices: &idx_v,
                        topk_weights: &w_v,
                        rows: 1,
                        n_experts: nexp,
                        top_k: ntopk,
                    }),
                    "block router topk row",
                )?;
            }
        }
    }
    Ok(())
}

/// Free one blocked visit's in-flight rows (fail-closed cleanup; the stacks
/// and `Owned`s are freed by the caller).
fn free_block_rows(gpu: &mut Gpu, tmps: Vec<Vec<GpuTensor>>, xs: Vec<GpuTensor>) {
    for t in tmps {
        free_all(gpu, t);
    }
    for x in xs {
        let _ = gpu.free_tensor(x);
    }
}

fn free_push_pair(
    gpu: &mut Gpu,
    push: &mut Option<GpuTensor>,
    push_normed: &mut Option<GpuTensor>,
) {
    if let Some(p) = push.take() {
        let _ = gpu.free_tensor(p);
    }
    if let Some(p) = push_normed.take() {
        let _ = gpu.free_tensor(p);
    }
}

fn free_push_pairs(
    gpu: &mut Gpu,
    pushes: &mut Vec<Option<GpuTensor>>,
    push_ns: &mut Vec<Option<GpuTensor>>,
) {
    for p in pushes.drain(..).chain(push_ns.drain(..)).flatten() {
        let _ = gpu.free_tensor(p);
    }
}

/// One batched qtile attention launch over a whole block (kernel #3).
/// Reads the clipped q/rel/sinks stacks plus the visit's F16 K/V caches
/// (fully written by the prequel loop) and writes the attn_out stack.
/// `swa_ev` carries a wrapping SWA block's pre-overwrite snapshot range
/// (snapshotted before the prequel writes by the caller); `None` disables
/// the redirect (ev 0,0). Fail-closed on any shape/dtype mismatch.
#[allow(clippy::too_many_arguments)]
fn blocked_attn_launch(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    layer: &Inkling2GpuLayer,
    gs: &mut Inkling2GpuState,
    stacks: &BlockStacks,
    visit: usize,
    b: usize,
    start: usize,
    swa_ev: Option<(usize, usize)>,
) -> Result<(), String> {
    let nh = cfg.n_head;
    let hd = cfg.head_dim;
    if hd != 128 {
        return Err(format!(
            "inkling2 gpu fwd: blocked attn needs head_dim 128, got {hd}"
        ));
    }
    let eattn = nh * hd;
    let erel = nh * cfg.n_rel;
    if stacks.q32.shape != vec![b, eattn]
        || stacks.q32.dtype != DType::F32
        || stacks.attn_out.shape != vec![b, eattn]
        || stacks.attn_out.dtype != DType::F32
    {
        return Err(format!(
            "inkling2 gpu fwd: blocked attn q32/attn_out must be F32 [B,{eattn}]=[{b},{eattn}]"
        ));
    }
    if stacks.r32.shape != vec![b, erel] || stacks.r32.dtype != DType::F32 {
        return Err(format!(
            "inkling2 gpu fwd: blocked attn r32 must be F32 [B,{erel}]=[{b},{erel}]"
        ));
    }
    if stacks.sinks.shape != vec![b, nh] || stacks.sinks.dtype != DType::F32 {
        return Err(format!(
            "inkling2 gpu fwd: blocked attn sinks must be F32 [B,{nh}]=[{b},{nh}]"
        ));
    }
    let kc = match gs.k_cache.get(visit) {
        Some(k) => k,
        None => return Err(format!("inkling2 gpu fwd: no k_cache for visit {visit}")),
    };
    let vc = match gs.v_cache.get(visit) {
        Some(v) => v,
        None => return Err(format!("inkling2 gpu fwd: no v_cache for visit {visit}")),
    };
    if kc.len() != 8 || vc.len() != 8 {
        return Err(format!(
            "inkling2 gpu fwd: blocked attn needs 8 K/V heads, got {}/{}",
            kc.len(),
            vc.len()
        ));
    }
    let k = [
        &kc[0], &kc[1], &kc[2], &kc[3], &kc[4], &kc[5], &kc[6], &kc[7],
    ];
    let v = [
        &vc[0], &vc[1], &vc[2], &vc[3], &vc[4], &vc[5], &vc[6], &vc[7],
    ];
    let sk_views = snap_heads(&stacks.sk_snap, b)?;
    let sv_views = snap_heads(&stacks.sv_snap, b)?;
    let sk = [
        &sk_views[0],
        &sk_views[1],
        &sk_views[2],
        &sk_views[3],
        &sk_views[4],
        &sk_views[5],
        &sk_views[6],
        &sk_views[7],
    ];
    let sv = [
        &sv_views[0],
        &sv_views[1],
        &sv_views[2],
        &sv_views[3],
        &sv_views[4],
        &sv_views[5],
        &sv_views[6],
        &sv_views[7],
    ];
    let rows = gs.visit_rows[visit];
    let inv = 1.0 / (hd as f32).sqrt();
    let (ev_lo, ev_hi) = swa_ev.unwrap_or((0, 0));
    gpu.inkling2_attn_prefill_qtile(
        &stacks.q32,
        &stacks.r32,
        &layer.rel_table,
        &stacks.sinks,
        &k,
        &v,
        &sk,
        &sv,
        ev_lo,
        ev_hi,
        &stacks.attn_out,
        b,
        start,
        rows,
        cfg.n_extent,
        inv,
    )
    .map_err(|e| format!("inkling2 gpu fwd: blocked attn launch: {e:?}"))
}

/// One row's visit head: mix bases, messages/control, input norms.
/// Writes normed q/k/v into caller-owned `qi`/`ki`/`vi` (pool temps in
/// single-row mode, stack-row views in blocked mode). Returns the message
/// indices the tail's MoE half needs. Shared by the seam and the blocked
/// driver (same ops, same order, same labels).
#[allow(clippy::too_many_arguments)]
fn visit_head(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    layer: &Inkling2GpuLayer,
    x: &GpuTensor,
    blocks: &[GpuTensor],
    blocks_normed: &[GpuTensor],
    visit: usize,
    phys: usize,
    msgs: Option<(
        &[GpuTensor],
        &[GpuTensor],
        &GpuTensor,
        &GpuTensor,
        &[Inkling2Controller; 4],
    )>,
    scratch: &Inkling2VisitScratch,
    cap: &mut Option<HashMap<String, Vec<f32>>>,
    tmp: &mut Vec<GpuTensor>,
    qi: &GpuTensor,
    ki: &GpuTensor,
    vi: &GpuTensor,
) -> Result<Option<[usize; 4]>, String> {
    let e = cfg.n_embd;
    let eps = cfg.rms_norm_eps;
    let swa = cfg.swa_layers[phys];
    macro_rules! step {
        ($expr:expr, $what:literal) => {
            match $expr {
                Ok(v) => v,
                Err(e) => {
                    free_all(gpu, std::mem::take(tmp));
                    return Err(format!("inkling2 gpu fwd: {}: {e:?}", $what));
                }
            }
        };
    }
    // No bare `?` below this point: every failure must free tmp.
    step!(cap_put(cap, gpu, "x_in", x), "capture");
    // --- attention input: AttnRes mix (no query feeds visit 0) ---
    // The triple holds temp indices; single-query layers share one buffer
    // for q/k/v (read-only until the norms write into fresh buffers).
    let (bq, bk, bv): (Option<usize>, Option<usize>, Option<usize>) = if phys == 0 {
        (None, None, None)
    } else {
        let aq = match layer.attn_query.as_ref() {
            Some(t) => t,
            None => {
                return Err(format!(
                    "inkling2 gpu fwd: visit {visit} phys {phys} missing attn_query"
                ));
            }
        };
        let total = aq.buf.size() / 4;
        if total % e != 0 {
            return Err(format!(
                "inkling2 gpu fwd: attn_query len {total} not a multiple of {e}"
            ));
        }
        // Mirrors forward.rs: 3 rows when the query carries them, else 1.
        let qr = if total / e >= 3 { 3 } else { 1 };
        // D1: the normed twins ride alongside `blocks` (norm-once at push).
        assert_eq!(
            blocks.len(),
            blocks_normed.len(),
            "inkling2 gpu fwd: blocks/normed length split"
        );
        let bank_n: Vec<&GpuTensor> = blocks_normed.iter().collect();
        let mixed = step!(
            gpu_mix(gpu, scratch, &bank_n, x, aq, e, qr, eps),
            "attn mix"
        );
        tmp.push(mixed);
        let mi = tmp.len() - 1;
        if qr >= 3 {
            let r0 = step!(at(gpu, tmp, &[e], "base_q"), "alloc");
            let r1 = step!(at(gpu, tmp, &[e], "base_k"), "alloc");
            let r2 = step!(at(gpu, tmp, &[e], "base_v"), "alloc");
            step!(
                copy_row_from(gpu, &tmp[mi], 0, &tmp[r0], e, "copy"),
                "base_q"
            );
            step!(
                copy_row_from(gpu, &tmp[mi], 1, &tmp[r1], e, "copy"),
                "base_k"
            );
            step!(
                copy_row_from(gpu, &tmp[mi], 2, &tmp[r2], e, "copy"),
                "base_v"
            );
            (Some(r0), Some(r1), Some(r2))
        } else {
            let r0 = step!(at(gpu, tmp, &[e], "base_qkv"), "alloc");
            step!(
                gpu.copy_rows_strided_f32(&tmp[mi], &tmp[r0], 1, e, e, e, 0, None),
                "base"
            );
            (Some(r0), Some(r0), Some(r0))
        }
    };
    // Loop messages (pass > 0): control each input; pass 0: bases as-is.
    // Visit 0 (phys 0, no mix): all three inputs are x itself.
    // mids survive in tmp for the MoE half (FFN message + controller 3).
    let mut mids: Option<[usize; 4]> = None;
    let (qc0, kc0, vc0) = match &msgs {
        Some((bank, bank_normed, prev_out, cross, ctrls)) => {
            assert_eq!(
                bank.len(),
                bank_normed.len(),
                "inkling2 gpu fwd: msg bank/normed split"
            );
            let (bqi, bki, bvi) = match (bq, bk, bv) {
                (Some(a), Some(b), Some(c)) => (a, b, c),
                _ => {
                    return Err("inkling2 gpu fwd: phys 0 has no base".to_string());
                }
            };
            if cross.buf.size() / 4 != 4 * e {
                return Err(format!(
                    "inkling2 gpu fwd: cross len {} != 4*{e}",
                    cross.buf.size() / 4
                ));
            }
            let mut m = [0usize; 4];
            for j in 0..4 {
                let qj = GpuTensor {
                    buf: cross.buf.byte_view(j * e * 4, e * 4),
                    shape: vec![1, e],
                    dtype: DType::F32,
                };
                let bank_refs: Vec<&GpuTensor> = bank_normed.iter().collect();
                let msg = step!(
                    gpu_mix(gpu, scratch, &bank_refs, prev_out, &qj, e, 1, eps),
                    "msg mix"
                );
                tmp.push(msg);
                let mi = tmp.len() - 1;
                let row = step!(at(gpu, tmp, &[e], "msg"), "alloc");
                step!(
                    gpu.copy_rows_strided_f32(&tmp[mi], &tmp[row], 1, e, e, e, 0, None),
                    "msg row"
                );
                m[j] = row;
            }
            let q = step!(
                gpu_control(
                    gpu,
                    scratch,
                    &tmp[bqi],
                    &tmp[m[0]],
                    &ctrls[0],
                    e,
                    cfg.loop_cfg.gate_rank
                ),
                "q control"
            );
            tmp.push(q);
            let qci = tmp.len() - 1;
            let k = step!(
                gpu_control(
                    gpu,
                    scratch,
                    &tmp[bki],
                    &tmp[m[1]],
                    &ctrls[1],
                    e,
                    cfg.loop_cfg.gate_rank
                ),
                "k control"
            );
            tmp.push(k);
            let kci = tmp.len() - 1;
            let v = step!(
                gpu_control(
                    gpu,
                    scratch,
                    &tmp[bvi],
                    &tmp[m[2]],
                    &ctrls[2],
                    e,
                    cfg.loop_cfg.gate_rank
                ),
                "v control"
            );
            tmp.push(v);
            let vci = tmp.len() - 1;
            step!(cap_put(cap, gpu, "base_q", &tmp[bqi]), "capture");
            step!(cap_put(cap, gpu, "q_ctrl", &tmp[qci]), "capture");
            mids = Some(m);
            (qci, kci, vci)
        }
        None if phys == 0 => {
            let xi = step!(at(gpu, tmp, &[e], "x_dup"), "alloc");
            step!(
                gpu.copy_rows_strided_f32(x, &tmp[xi], 1, e, e, e, 0, None),
                "x dup"
            );
            (xi, xi, xi)
        }
        None => match (bq, bk, bv) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => {
                return Err("inkling2 gpu fwd: phys 0 has no base".to_string());
            }
        },
    };
    // Norms: SWA applies attn_norm to all three inputs (pass > 0 SWA has
    // three different controllers, so q/k/v are already distinct); full
    // layers use the three input norms. Outputs land in the caller-owned
    // qi/ki/vi (pool temps single-row, stack rows blocked).
    if swa {
        let an = match layer.attn_norm.as_ref() {
            Some(t) => t,
            None => {
                return Err("inkling2 gpu fwd: SWA layer missing attn_norm".to_string());
            }
        };
        step!(gpu.rmsnorm_f32(&tmp[qc0], an, qi, eps), "q norm");
        step!(gpu.rmsnorm_f32(&tmp[kc0], an, ki, eps), "k norm");
        step!(gpu.rmsnorm_f32(&tmp[vc0], an, vi, eps), "v norm");
    } else {
        let (qn, kn, vn) = match (
            &layer.q_input_norm,
            &layer.k_input_norm,
            &layer.v_input_norm,
        ) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => {
                return Err("inkling2 gpu fwd: full layer missing input norms".to_string());
            }
        };
        step!(gpu.rmsnorm_f32(&tmp[qc0], qn, qi, eps), "q norm");
        step!(gpu.rmsnorm_f32(&tmp[kc0], kn, ki, eps), "k norm");
        step!(gpu.rmsnorm_f32(&tmp[vc0], vn, vi, eps), "v norm");
    };
    step!(cap_put(cap, gpu, "q_inp", qi), "capture");
    step!(cap_put(cap, gpu, "k_inp", ki), "capture");
    step!(cap_put(cap, gpu, "v_inp", vi), "capture");
    Ok(mids)
}

/// One row's pre-attention half: convs, clips, norms, sinks, tags, KV
/// write. `q`/`k_raw`/`v_raw`/`r` are the projection rows (pool temps in
/// single-row mode, q32/k32/v32/r32 stack rows in blocked mode, already
/// widened to F32 there); `sinks_out` is caller-owned (pool temp or sinks
/// stack row). Clips run in place on the caller's rows. Shared by the seam
/// and the blocked driver (same ops, same order, same labels). Returns the
/// window length `n_kv` the attention step reads.
#[allow(clippy::too_many_arguments)]
fn visit_pre(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    layer: &Inkling2GpuLayer,
    gs: &mut Inkling2GpuState,
    visit: usize,
    phys: usize,
    pos: usize,
    qi: &GpuTensor,
    ki: &GpuTensor,
    vi: &GpuTensor,
    q: &GpuTensor,
    k_raw: &GpuTensor,
    v_raw: &GpuTensor,
    r: &GpuTensor,
    sinks_out: &GpuTensor,
    cap: &mut Option<HashMap<String, Vec<f32>>>,
    tmp: &mut Vec<GpuTensor>,
) -> Result<usize, String> {
    let nh = cfg.n_head;
    let nkv = cfg.n_head_kv;
    let hd = cfg.head_dim;
    let ekv = nkv * hd;
    let eattn = nh * hd;
    let erel = nh * cfg.n_rel;
    let eps = cfg.rms_norm_eps;
    let swa = cfg.swa_layers[phys];
    macro_rules! step {
        ($expr:expr, $what:literal) => {
            match $expr {
                Ok(v) => v,
                Err(e) => {
                    free_all(gpu, std::mem::take(tmp));
                    return Err(format!("inkling2 gpu fwd: {}: {e:?}", $what));
                }
            }
        };
    }
    // No bare `?` below this point: every failure must free tmp.
    step!(cap_put(cap, gpu, "k_raw", k_raw), "capture");
    step!(cap_put(cap, gpu, "v_raw", v_raw), "capture");
    // Convs (static on SWA; dynamic taps on full layers).
    let k_conv = step!(at(gpu, tmp, &[ekv], "k_conv"), "alloc");
    let v_conv = step!(at(gpu, tmp, &[ekv], "v_conv"), "alloc");
    if swa {
        step!(
            launch_conv(
                gpu,
                k_raw,
                &layer.k_conv,
                None,
                &gs.conv[visit],
                &tmp[k_conv],
                0,
                ekv,
                0,
                hd,
                1.0 / 5.0f32.sqrt()
            ),
            "k conv"
        );
        step!(
            launch_conv(
                gpu,
                v_raw,
                &layer.v_conv,
                None,
                &gs.conv[visit],
                &tmp[v_conv],
                ekv,
                ekv,
                0,
                hd,
                1.0 / 5.0f32.sqrt()
            ),
            "v conv"
        );
    } else {
        let (kp, vp) = match (&layer.k_proj, &layer.v_proj) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                return Err("inkling2 gpu fwd: full layer missing k/v proj".to_string());
            }
        };
        let dk = step!(at(gpu, tmp, &[nkv * cfg.conv_kernel], "dyn_k"), "alloc");
        let dv = step!(at(gpu, tmp, &[nkv * cfg.conv_kernel], "dyn_v"), "alloc");
        step!(gpu.gemv_f32(kp, ki, &tmp[dk]), "k dyn proj");
        step!(gpu.gemv_f32(vp, vi, &tmp[dv]), "v dyn proj");
        step!(
            launch_conv(
                gpu,
                k_raw,
                &layer.k_conv,
                Some(&tmp[dk]),
                &gs.conv[visit],
                &tmp[k_conv],
                0,
                ekv,
                nkv,
                hd,
                1.0 / 3.0
            ),
            "k conv"
        );
        step!(
            launch_conv(
                gpu,
                v_raw,
                &layer.v_conv,
                Some(&tmp[dv]),
                &gs.conv[visit],
                &tmp[v_conv],
                ekv,
                ekv,
                nkv,
                hd,
                1.0 / 3.0
            ),
            "v conv"
        );
    }
    step!(cap_put(cap, gpu, "k_conv", &tmp[k_conv]), "capture");
    step!(cap_put(cap, gpu, "v_conv", &tmp[v_conv]), "capture");
    // Clips + V head-norm (clips run in place on the caller's rows).
    {
        let qv = step!(view2(q, nh, hd), "view");
        step!(
            gpu.inkling2_l2_clip_rows_f32(&qv, Some(&layer.q_clip), nh, hd, cfg.qk_clip),
            "q clip"
        );
        let kv = step!(view2(&tmp[k_conv], nkv, hd), "view");
        step!(
            gpu.inkling2_l2_clip_rows_f32(&kv, None, nkv, hd, cfg.qk_clip),
            "k clip"
        );
        let rv = step!(view2(r, nh, cfg.n_rel), "view");
        step!(
            gpu.inkling2_l2_clip_rows_f32(&rv, Some(&layer.rel_clip), nh, cfg.n_rel, cfg.rel_clip),
            "r clip"
        );
    }
    let vc = step!(at(gpu, tmp, &[ekv], "vc"), "alloc");
    {
        let vin = step!(view2(&tmp[v_conv], nkv, hd), "view");
        step!(
            gpu.inkling2_rmsnorm_nogamma_f32(&vin, &tmp[vc], nkv, hd, eps),
            "v norm"
        );
    }
    // Capture the clipped, unscaled values so CPU/GPU component parity keeps
    // the same meaning at every position. Scaled values get separate keys.
    step!(cap_put(cap, gpu, "qc", q), "capture");
    step!(cap_put(cap, gpu, "rc", r), "capture");
    // Full-layer tau multiplies clipped Q and R by oracle::tau. SWA is
    // untouched; below the floor no launch fires. The documented kill switch
    // is for the GPU-vs-GPU gate arm only.
    if !swa && hipfire_config::developer_var("HIPFIRE_INKLING2_TAU_OFF").as_deref() != Ok("1") {
        match full_layer_tau(pos, cfg.log_floor, cfg.log_alpha) {
            Ok(None) => {}
            Ok(Some(t)) => {
                let scq = step!(
                    hip(gpu.upload_f32(&vec![t; eattn], &[eattn]), "tau scale"),
                    "tau scale"
                );
                tmp.push(scq);
                let siq = tmp.len() - 1;
                step!(gpu.mul_f32(q, &tmp[siq], q), "q tau");
                let scr = step!(
                    hip(gpu.upload_f32(&vec![t; erel], &[erel]), "tau scale"),
                    "tau scale"
                );
                tmp.push(scr);
                let sir = tmp.len() - 1;
                step!(gpu.mul_f32(r, &tmp[sir], r), "r tau");
                step!(cap_put(cap, gpu, "qc_scaled", q), "capture");
                step!(cap_put(cap, gpu, "rc_scaled", r), "capture");
            }
            Err(e) => return Err(e),
        }
    }
    // Sinks.
    step!(gpu.gemv_f32(&layer.sink, qi, sinks_out), "sink");
    // KV write + gather map: full visits grow to max_seq and fail closed
    // past it; SWA visits are an evict-on-write ring (slot `pos % rows`)
    // holding the trailing `min(pos+1, rows)` keys with true positions in
    // the per-slot tags. Below the ring size both reduce to the old path
    // (slot == pos, tag == pos) bit-identically.
    let rows = gs.visit_rows[visit];
    if !swa && pos >= rows {
        return Err(format!(
            "inkling2 gpu fwd: visit {visit} KV overflow at pos {pos} (rows {rows})"
        ));
    }
    let wpos = if swa { swa_slot(pos, rows) } else { pos };
    gs.visit_pos_tags[visit][wpos] = pos as u32;
    gs.visit_cap[visit] = pos + 1;
    // D1: tags stay host-side (fail-closed validation, zero H2D). The
    // slots/dists maps are built on-device from the pos word advanced once
    // per token; host recomputation below only validates the window.
    let n_kv = match swa_gather_map(&gs.visit_pos_tags[visit], pos, rows) {
        Ok((slots, _)) => slots.len(),
        Err(e) => {
            return Err(format!("inkling2 gpu fwd: {e}"));
        }
    };
    step!(
        launch_build_slots_dists(gpu, &gs.pos_dev, &gs.slots_dev, &gs.dists_dev, rows),
        "slots kernel"
    );
    step!(cap_put(cap, gpu, "kc", &tmp[k_conv]), "capture");
    step!(cap_put(cap, gpu, "vc", &tmp[vc]), "capture");
    step!(cap_put(cap, gpu, "sinks", sinks_out), "capture");
    // D1: one batched KV-write launch per visit (grid (8, 2)) replaces 16
    // per-head launches. F16 keeps the (_Float16) cast bit-identically.
    step!(
        launch_kv_write_f16_batched(
            gpu,
            &tmp[k_conv],
            &tmp[vc],
            &gs.k_cache[visit],
            &gs.v_cache[visit],
            wpos,
            hd
        ),
        "k/v cache write"
    );
    Ok(n_kv)
}

/// One row's single-row attention: rel-bias gather into a tmp-owned buffer,
/// then the legacy decode kernel. Shared by the seam and the blocked
/// fallback arm (the batched arm uses the qtile instead). `rel_in` is the
/// clipped r row, `sinks` its sink row, `out` the caller-owned attention
/// output (pool temp or attn_out stack row).
#[allow(clippy::too_many_arguments)]
fn visit_attn_single(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    layer: &Inkling2GpuLayer,
    gs: &mut Inkling2GpuState,
    visit: usize,
    qv: &GpuTensor,
    rel_in: &GpuTensor,
    sinks: &GpuTensor,
    n_kv: usize,
    out: &GpuTensor,
    cap: &mut Option<HashMap<String, Vec<f32>>>,
    tmp: &mut Vec<GpuTensor>,
) -> Result<(), String> {
    let nh = cfg.n_head;
    let hd = cfg.head_dim;
    macro_rules! step {
        ($expr:expr, $what:literal) => {
            match $expr {
                Ok(v) => v,
                Err(e) => {
                    free_all(gpu, std::mem::take(tmp));
                    return Err(format!("inkling2 gpu fwd: {}: {e:?}", $what));
                }
            }
        };
    }
    // Device-map views (first n_kv words; kernels only read the prefix).
    let dists_v = GpuTensor {
        buf: step!(sub_buf(&gs.dists_dev, 0, n_kv * 4, "dists view"), "view"),
        shape: vec![n_kv],
        dtype: DType::F32,
    };
    let slots_v = GpuTensor {
        buf: step!(sub_buf(&gs.slots_dev, 0, n_kv * 4, "slots view"), "view"),
        shape: vec![n_kv],
        dtype: DType::F32,
    };
    // Persistent rel_bias staging: prefix [nh, n_kv] of the largest-visit
    // buffer (same prefix bytes the per-visit alloc produced).
    let rel_bias_v = GpuTensor {
        buf: step!(
            sub_buf(&gs.scratch.rel_bias, 0, nh * n_kv * 4, "rel_bias view"),
            "view"
        ),
        shape: vec![nh, n_kv],
        dtype: DType::F32,
    };
    {
        let rv = step!(view2(rel_in, nh, cfg.n_rel), "view");
        step!(
            gpu.inkling2_rel_bias_gather_f32(
                &rv,
                &layer.rel_table,
                sinks,
                &dists_v,
                &rel_bias_v,
                cfg.n_rel,
                nh,
                cfg.n_extent,
                n_kv
            ),
            "rel_bias"
        );
    }
    step!(cap_put(cap, gpu, "rel_bias", &rel_bias_v), "capture");
    step!(
        launch_attn(
            gpu,
            qv,
            &gs.k_cache[visit],
            &gs.v_cache[visit],
            &rel_bias_v,
            &slots_v,
            out,
            n_kv,
            nh,
            hd
        ),
        "attn"
    );
    Ok(())
}

/// In-place gate over one attention row: sigmoid then multiply. No
/// allocation, so both the seam (pool rows) and the blocked driver (stack
/// rows) share it directly.
fn gate_apply(gpu: &mut Gpu, gate_row: &GpuTensor, attn_row: &GpuTensor) -> Result<(), String> {
    gpu.sigmoid_f32(gate_row)
        .map_err(|e| format!("inkling2 gpu fwd: gate sigmoid: {e:?}"))?;
    gpu.mul_f32(attn_row, gate_row, attn_row)
        .map_err(|e| format!("inkling2 gpu fwd: gate mul: {e:?}"))?;
    Ok(())
}

/// One row's post-attention tail through the FFN norm: post-conv over the
/// batched/single `wo` row, cur/push bookkeeping, MoE mix/control/norm into
/// the caller-owned `fnorm_row`. Moved code (conv through norm); the seam
/// keeps calling it with pool rows, the blocked driver with stack rows.
/// Returns the `cur` index and the push pair (`tmp` stays with the caller).
#[allow(clippy::too_many_arguments)]
fn tail_conv_norm(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    layer: &Inkling2GpuLayer,
    gs: &mut Inkling2GpuState,
    visit: usize,
    phys: usize,
    msgs: Option<(
        &[GpuTensor],
        &[GpuTensor],
        &GpuTensor,
        &GpuTensor,
        &[Inkling2Controller; 4],
    )>,
    mids: Option<[usize; 4]>,
    cap: &mut Option<HashMap<String, Vec<f32>>>,
    tmp: &mut Vec<GpuTensor>,
    x: &GpuTensor,
    wo_row: &GpuTensor,
    fnorm_row: &GpuTensor,
    blocks_normed: &[GpuTensor],
) -> Result<(usize, Option<GpuTensor>, Option<GpuTensor>), String> {
    let e = cfg.n_embd;
    let nkv = cfg.n_head_kv;
    let hd = cfg.head_dim;
    let ekv = nkv * hd;
    let eps = cfg.rms_norm_eps;
    let mut push: Option<GpuTensor> = None;
    let mut push_normed: Option<GpuTensor> = None;
    macro_rules! step {
        ($expr:expr, $what:literal) => {
            match $expr {
                Ok(v) => v,
                Err(e) => {
                    if let Some(p) = push.take() {
                        let _ = gpu.free_tensor(p);
                    }
                    if let Some(p) = push_normed.take() {
                        let _ = gpu.free_tensor(p);
                    }
                    free_all(gpu, std::mem::take(tmp));
                    return Err(format!("inkling2 gpu fwd: {}: {e:?}", $what));
                }
            }
        };
    }
    // No bare `?` below this point: every failure must free tmp/push.
    let attn_res = step!(at(gpu, tmp, &[e], "attn_res"), "alloc");
    step!(
        launch_conv(
            gpu,
            wo_row,
            &layer.attn_post_conv,
            None,
            &gs.conv[visit],
            &tmp[attn_res],
            2 * ekv,
            e,
            0,
            hd,
            1.0 / 5.0f32.sqrt()
        ),
        "attn post"
    );
    // AttnRes bookkeeping: first visit of a block pushes x, others add.
    let has_push = phys % cfg.attn_res_block == 0;
    if has_push {
        let p = step!(dup(gpu, x, "block"), "dup");
        // D1 norm-once: RMSNorm the pushed block once (same tree reduction
        // as the copying path, so bitwise identical); every later mix reads
        // the twin via pointers and never re-norms it.
        let pn = match gpu.alloc_tensor(&[e], DType::F32) {
            Ok(t) => t,
            Err(e2) => {
                let _ = gpu.free_tensor(p);
                if let Some(pp) = push.take() {
                    let _ = gpu.free_tensor(pp);
                }
                free_all(gpu, std::mem::take(tmp));
                return Err(format!("inkling2 gpu fwd: alloc block_normed: {e2:?}"));
            }
        };
        if let Err(e2) = gpu.inkling2_rmsnorm_nogamma_f32(&p, &pn, 1, e, eps) {
            let _ = gpu.free_tensor(p);
            let _ = gpu.free_tensor(pn);
            if let Some(pp) = push.take() {
                let _ = gpu.free_tensor(pp);
            }
            free_all(gpu, std::mem::take(tmp));
            return Err(format!("inkling2 gpu fwd: block norm: {e2:?}"));
        }
        push = Some(p);
        push_normed = Some(pn);
    }
    let cur = step!(at(gpu, tmp, &[e], "cur"), "alloc");
    if has_push {
        step!(
            gpu.copy_rows_strided_f32(&tmp[attn_res], &tmp[cur], 1, e, e, e, 0, None),
            "cur copy"
        );
    } else {
        step!(gpu.add_f32(x, &tmp[attn_res], &tmp[cur]), "attn residual");
    }
    step!(cap_put(cap, gpu, "cur_attn", &tmp[cur]), "capture");
    step!(cap_put(cap, gpu, "cur", &tmp[cur]), "capture");
    // --- MoE: mix over blocks+cur. The just-pushed block IS included:
    // CPU pushes during attention bookkeeping, before the MoE half.
    // Control (pass > 0), norm, routed MoE follow.
    if layer.mlp_query.buf.size() / 4 != e {
        if let Some(p) = push.take() {
            let _ = gpu.free_tensor(p);
        }
        if let Some(p) = push_normed.take() {
            let _ = gpu.free_tensor(p);
        }
        free_all(gpu, std::mem::take(tmp));
        return Err(format!(
            "inkling2 gpu fwd: mlp_query len {} != {e}",
            layer.mlp_query.buf.size() / 4
        ));
    }
    let block_n: Vec<&GpuTensor> = blocks_normed.iter().chain(push_normed.iter()).collect();
    let ffn_mixed = step!(
        gpu_mix(
            gpu,
            &gs.scratch,
            &block_n,
            &tmp[cur],
            &layer.mlp_query,
            e,
            1,
            eps
        ),
        "ffn mix"
    );
    tmp.push(ffn_mixed);
    let fmi = tmp.len() - 1;
    let ffn_in = match (&msgs, mids) {
        (Some((_, _, _, _, ctrls)), Some(m)) => {
            let f = step!(
                gpu_control(
                    gpu,
                    &gs.scratch,
                    &tmp[fmi],
                    &tmp[m[3]],
                    &ctrls[3],
                    e,
                    cfg.loop_cfg.gate_rank
                ),
                "ffn control"
            );
            tmp.push(f);
            tmp.len() - 1
        }
        _ => fmi,
    };
    step!(
        gpu.rmsnorm_f32(&tmp[ffn_in], &layer.mlp_norm, fnorm_row, eps),
        "ffn norm"
    );
    step!(cap_put(cap, gpu, "ffn_mix", &tmp[ffn_in]), "capture");
    step!(cap_put(cap, gpu, "ffn_normed", fnorm_row), "capture");
    Ok((cur, push, push_normed))
}

/// One row's MoE middle: rotate, gate_up (indexed), SiTU, act-rotate, down
/// (indexed expanded). Moved code; the seam and the blocked driver share it
/// (grouped MoE never fires on the fixture's qt13 experts, so the middle is
/// always per row). `rot_row`/`idx_row` are caller-owned (pool temp or stack
/// row); the intermediates live in `tmp`. Returns `tmp` and the `down_exp`
/// index.
#[allow(clippy::too_many_arguments)]
fn moe_middle_single(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    layer: &Inkling2GpuLayer,
    cap: &mut Option<HashMap<String, Vec<f32>>>,
    tmp: &mut Vec<GpuTensor>,
    rot_row: &GpuTensor,
    idx_row: &GpuTensor,
) -> Result<usize, String> {
    let e = cfg.n_embd;
    let inter = cfg.inter;
    macro_rules! step {
        ($expr:expr, $what:literal) => {
            match $expr {
                Ok(v) => v,
                Err(e) => {
                    free_all(gpu, std::mem::take(tmp));
                    return Err(format!("inkling2 gpu fwd: {}: {e:?}", $what));
                }
            }
        };
    }
    let gate_b = step!(at(gpu, tmp, &[cfg.n_expert_used, inter], "gate_b"), "alloc");
    let up_b = step!(at(gpu, tmp, &[cfg.n_expert_used, inter], "up_b"), "alloc");
    step!(
        gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
            &layer.ptr_gu,
            idx_row,
            rot_row,
            &tmp[gate_b],
            &tmp[up_b],
            2 * inter,
            e,
            cfg.n_expert_used,
            1
        ),
        "gate_up"
    );
    step!(cap_put(cap, gpu, "gate_b", &tmp[gate_b]), "capture");
    step!(cap_put(cap, gpu, "up_b", &tmp[up_b]), "capture");
    let act = step!(at(gpu, tmp, &[cfg.n_expert_used, 1280], "act"), "alloc");
    step!(
        gpu.ffn_activation_rows_f32(&FfnActivationRows {
            activation: FfnActivation::Situ {
                gate_beta: cfg.situ_beta,
                value_beta: cfg.situ_linear_beta
            },
            gate: &tmp[gate_b],
            gate_offset: 0,
            gate_ld: inter,
            up: &tmp[up_b],
            up_offset: 0,
            up_ld: inter,
            out: &tmp[act],
            rows: cfg.n_expert_used,
            logical_n: inter,
            phys_n: 1280,
            round_inputs: false,
            round_output: false,
            awq: FfnAwq::None,
        }),
        "situ"
    );
    step!(cap_put(cap, gpu, "act", &tmp[act]), "capture");
    let rot = step!(at(gpu, tmp, &[cfg.n_expert_used, 1280], "rot"), "alloc");
    step!(
        gpu.rotate_x_mq_batched(&tmp[act], &tmp[rot], 1280, cfg.n_expert_used),
        "act rotate"
    );
    let down_exp = step!(at(gpu, tmp, &[cfg.n_expert_used, e], "down_exp"), "alloc");
    step!(
        gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
            &layer.ptr_dn,
            idx_row,
            &tmp[rot],
            &tmp[down_exp],
            e,
            1280,
            cfg.n_expert_used,
            1
        ),
        "down"
    );
    step!(cap_put(cap, gpu, "down_exp", &tmp[down_exp]), "capture");
    step!(cap_put(cap, gpu, "rot_b", &tmp[rot]), "capture");
    Ok(down_exp)
}

/// One row's MoE postfix: combine, post-conv, residual into the visit
/// output. Moved code; shared by the seam and the blocked driver.
/// `moe_out_row` is caller-owned (pool temp or mmoe stack row).
#[allow(clippy::too_many_arguments)]
fn moe_postfix(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    layer: &Inkling2GpuLayer,
    gs: &mut Inkling2GpuState,
    visit: usize,
    cap: &mut Option<HashMap<String, Vec<f32>>>,
    tmp: &mut Vec<GpuTensor>,
    cur: usize,
    down: usize,
    idx_row: &GpuTensor,
    w_row: &GpuTensor,
    moe_out_row: &GpuTensor,
    push: Option<GpuTensor>,
    push_normed: Option<GpuTensor>,
) -> Result<VisitOut, String> {
    let e = cfg.n_embd;
    let nkv = cfg.n_head_kv;
    let hd = cfg.head_dim;
    let ekv = nkv * hd;
    let mut push = push;
    let mut push_normed = push_normed;
    macro_rules! step {
        ($expr:expr, $what:literal) => {
            match $expr {
                Ok(v) => v,
                Err(e) => {
                    if let Some(p) = push.take() {
                        let _ = gpu.free_tensor(p);
                    }
                    if let Some(p) = push_normed.take() {
                        let _ = gpu.free_tensor(p);
                    }
                    free_all(gpu, std::mem::take(tmp));
                    return Err(format!("inkling2 gpu fwd: {}: {e:?}", $what));
                }
            }
        };
    }
    // No bare `?` below this point: every failure must free tmp/push.
    step!(
        gpu.ffn_moe_combine_f32(&MoeCombine {
            order: CombineOrder::Rank,
            round_expert_outputs: false,
            round_routed_sum: false,
            round_final: false,
            down_rows: &tmp[down],
            row_of_slot: None,
            topk_indices: idx_row,
            topk_weights: w_row,
            shared: None,
            routed_out: None,
            out: moe_out_row,
            rows: 1,
            hidden: e,
            top_k: cfg.n_expert_used,
            n_experts: cfg.n_experts,
        }),
        "combine"
    );
    step!(cap_put(cap, gpu, "moe_raw", moe_out_row), "capture");
    let moe_conv = step!(at(gpu, tmp, &[e], "moe_conv"), "alloc");
    step!(
        launch_conv(
            gpu,
            moe_out_row,
            &layer.mlp_post_conv,
            None,
            &gs.conv[visit],
            &tmp[moe_conv],
            // ffn-out segment starts after K, V and attn-out: 2*ekv + e
            // (was 4 * ekv, which overlapped the last 256 attn-out channels).
            2 * ekv + e,
            e,
            0,
            hd,
            1.0 / 5.0f32.sqrt()
        ),
        "moe post"
    );
    step!(cap_put(cap, gpu, "moe_conv", &tmp[moe_conv]), "capture");
    // Tail: visit_out escapes; everything else dies here. Only add/capture
    // can fail past this point — free the survivors explicitly.
    let visit_out = match gpu.alloc_tensor(&[e], DType::F32) {
        Ok(t) => t,
        Err(e2) => {
            if let Some(p) = push.take() {
                let _ = gpu.free_tensor(p);
            }
            if let Some(p) = push_normed.take() {
                let _ = gpu.free_tensor(p);
            }
            free_all(gpu, std::mem::take(tmp));
            return Err(format!("inkling2 gpu fwd: alloc visit_out: {e2:?}"));
        }
    };
    if let Err(e2) = gpu.add_f32(&tmp[cur], &tmp[moe_conv], &visit_out) {
        let _ = gpu.free_tensor(visit_out);
        if let Some(p) = push.take() {
            let _ = gpu.free_tensor(p);
        }
        if let Some(p) = push_normed.take() {
            let _ = gpu.free_tensor(p);
        }
        free_all(gpu, std::mem::take(tmp));
        return Err(format!("inkling2 gpu fwd: moe residual: {e2:?}"));
    }
    if let Err(e2) = cap_put(cap, gpu, "visit_out", &visit_out) {
        let _ = gpu.free_tensor(visit_out);
        if let Some(p) = push.take() {
            let _ = gpu.free_tensor(p);
        }
        if let Some(p) = push_normed.take() {
            let _ = gpu.free_tensor(p);
        }
        free_all(gpu, std::mem::take(tmp));
        return Err(format!("inkling2 gpu fwd: capture: {e2:?}"));
    }
    free_all(gpu, std::mem::take(tmp));
    Ok(VisitOut {
        visit_out,
        push,
        push_normed,
    })
}

/// Message tuple for one row's visit (same shape as `step_visit` builds).
/// Rebuilt per row so borrows never cross the stream mutation.
type VisitMsgs<'a> = Option<(
    &'a [GpuTensor],
    &'a [GpuTensor],
    &'a GpuTensor,
    &'a GpuTensor,
    &'a [Inkling2Controller; 4],
)>;

/// Rebuild one row's message tuple (mirrors `step_visit`; cheap pointer
/// bundle).
fn block_msgs<'a>(
    o: &'a Owned,
    gw: &'a Inkling2GpuWeights,
    phys: usize,
    take_msgs: Option<(usize, usize)>,
) -> Result<VisitMsgs<'a>, String> {
    match take_msgs {
        Some((core_i, _)) => {
            let skip = o.prelude_blocks.len() + 1;
            if o.prev_blocks.len() < skip {
                return Err(format!(
                    "inkling2 gpu fwd: pass prev_blocks {} < skip {skip}",
                    o.prev_blocks.len()
                ));
            }
            assert_eq!(
                o.prev_blocks.len(),
                o.prev_blocks_n.len(),
                "inkling2 gpu fwd: prev blocks/normed split"
            );
            let ctrls = match &gw.layers[phys].controllers {
                Some(c) => c,
                None => {
                    return Err(format!(
                        "inkling2 gpu fwd: core layer {phys} missing controllers"
                    ))
                }
            };
            Ok(Some((
                // Same [skip..] bank as `step_visit` (CPU reference parity).
                &o.prev_blocks[skip..],
                &o.prev_blocks_n[skip..],
                o.prev_out.as_ref().expect("prev_out"),
                &gw.cross_pass[core_i],
                ctrls,
            )))
        }
        None => Ok(None),
    }
}

/// One block through the blocked visit: kernel #1 serves the four QKV/R
/// projections width-first (widen-step); every other op runs per row in
/// position order. B <= 1 never reaches here (see the entry point).
#[allow(clippy::too_many_arguments)]
fn blocked_step_visit(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    gw: &Inkling2GpuWeights,
    gs: &mut Inkling2GpuState,
    owned: &mut [Owned],
    start: usize,
    visit: usize,
    take_msgs: Option<(usize, usize)>,
    stacks: &mut BlockStacks,
) -> Result<(), String> {
    let b = owned.len();
    if b == 0 {
        return Err("inkling2 gpu fwd: blocked visit on empty block".to_string());
    }
    let phys = cfg.loop_cfg.layer_phys(visit);
    let layer = &gw.layers[phys];
    let e = cfg.n_embd;
    let eattn = cfg.n_head * cfg.head_dim;
    let ekv = cfg.n_head_kv * cfg.head_dim;
    let erel = cfg.n_head * cfg.n_rel;
    let ntopk = cfg.n_expert_used;
    let mut no_cap: Option<HashMap<String, Vec<f32>>> = None;
    // Heads: norms write straight into stack rows, so there is no gather.
    let mut tmps: Vec<Vec<GpuTensor>> = Vec::with_capacity(b);
    let mut xs: Vec<GpuTensor> = Vec::with_capacity(b);
    let mut mids: Vec<Option<[usize; 4]>> = Vec::with_capacity(b);
    for (r, o) in owned.iter_mut().enumerate() {
        let x = o.x.take().expect("inkling2 gpu fwd: missing x");
        let msgs = match block_msgs(o, gw, phys, take_msgs) {
            Ok(m) => m,
            Err(e2) => {
                free_block_rows(gpu, tmps, xs);
                let _ = gpu.free_tensor(x);
                return Err(e2);
            }
        };
        let (qi_v, ki_v, vi_v) = match (
            stack_row(&stacks.qi, r, e),
            stack_row(&stacks.ki, r, e),
            stack_row(&stacks.vi, r, e),
        ) {
            (Ok(a), Ok(b), Ok(c)) => (a, b, c),
            _ => {
                free_block_rows(gpu, tmps, xs);
                let _ = gpu.free_tensor(x);
                return Err("inkling2 gpu fwd: head row view short".to_string());
            }
        };
        let mut tmp = Vec::new();
        let mid = match visit_head(
            gpu,
            cfg,
            layer,
            &x,
            &o.blocks,
            &o.blocks_n,
            visit,
            phys,
            msgs,
            &gs.scratch,
            &mut no_cap,
            &mut tmp,
            &qi_v,
            &ki_v,
            &vi_v,
        ) {
            Ok(m) => m,
            Err(e2) => {
                free_block_rows(gpu, tmps, xs);
                let _ = gpu.free_tensor(x);
                return Err(e2);
            }
        };
        tmps.push(tmp);
        xs.push(x);
        mids.push(mid);
    }
    // Batched projections (or the per-row seam fallback below the gate).
    if BATCHED_QKVR_GATED {
        if let Err(e2) = blocked_project(gpu, layer, e, b, stacks) {
            free_block_rows(gpu, tmps, xs);
            return Err(e2);
        }
    } else {
        for (r, _) in owned.iter_mut().enumerate() {
            let (q_v, k_v, v_v, r_v) = match (
                stack_row(&stacks.q32, r, eattn),
                stack_row(&stacks.k32, r, ekv),
                stack_row(&stacks.v32, r, ekv),
                stack_row(&stacks.r32, r, erel),
            ) {
                (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
                _ => {
                    free_block_rows(gpu, tmps, xs);
                    return Err("inkling2 gpu fwd: proj row view short".to_string());
                }
            };
            let (qi_v, ki_v, vi_v) = match (
                stack_row(&stacks.qi, r, e),
                stack_row(&stacks.ki, r, e),
                stack_row(&stacks.vi, r, e),
            ) {
                (Ok(a), Ok(b), Ok(c)) => (a, b, c),
                _ => {
                    free_block_rows(gpu, tmps, xs);
                    return Err("inkling2 gpu fwd: head row view short".to_string());
                }
            };
            if let Err(e2) = weight_gemv(gpu, &layer.wq, &qi_v, &q_v)
                .and_then(|_| weight_gemv(gpu, &layer.wk, &ki_v, &k_v))
                .and_then(|_| weight_gemv(gpu, &layer.wv, &vi_v, &v_v))
                .and_then(|_| weight_gemv(gpu, &layer.wr, &qi_v, &r_v))
                .map_err(|e| format!("inkling2 gpu fwd: block proj row: {e:?}"))
            {
                free_block_rows(gpu, tmps, xs);
                return Err(e2);
            }
        }
    }
    // Attention + MoE-prefix per row in position order (the same
    // persistent-state order as the single-row loop: conv, tags, KV, pos).
    // Kernel #3 (BATCHED_ATTN_GATED): prequel per row, one qtile launch,
    // sequel per row below. SWA-wrap blocks batch via pre-overwrite
    // snapshot instead of falling back to single-row. Un-snapshotable
    // (multi-wrap, logic error) keeps the per-row path.
    let rows_v = gs.visit_rows[visit];
    let swa_wrap = cfg.swa_layers[phys] && start + b > rows_v;
    let swa_ev = if swa_wrap {
        swa_snapshot_range(start, b, rows_v)
    } else {
        None
    };
    let attn_batched = BATCHED_ATTN_GATED && (!swa_wrap || swa_ev.is_some());
    if let Err(e2) = gpu.inkling2_gf_set_pos(&gs.pos_dev, start) {
        free_block_rows(gpu, tmps, xs);
        return Err(format!("inkling2 gpu fwd: block set pos: {e2:?}"));
    }
    // Snapshot only on the batched path, in the visit's own dtype.
    if let Some((ev_lo, ev_hi)) = swa_ev.filter(|_| attn_batched) {
        // Snapshot BEFORE the block's KV writes: the values preserved are
        // exactly what those writes evict (proof at swa_snapshot_range).
        // Empty ranges cannot occur under swa_wrap (guarded above); skip
        // the launch defensively rather than snapshotting nothing.
        if ev_hi > ev_lo {
            let kc = match gs.k_cache.get(visit) {
                Some(k) => k,
                None => {
                    free_block_rows(gpu, tmps, xs);
                    return Err(format!("inkling2 gpu fwd: no k_cache for visit {visit}"));
                }
            };
            let vc = match gs.v_cache.get(visit) {
                Some(v) => v,
                None => {
                    free_block_rows(gpu, tmps, xs);
                    return Err(format!("inkling2 gpu fwd: no v_cache for visit {visit}"));
                }
            };
            if kc.len() != 8 || vc.len() != 8 {
                free_block_rows(gpu, tmps, xs);
                return Err(format!(
                    "inkling2 gpu fwd: blocked snap needs 8 K/V heads, got {}/{}",
                    kc.len(),
                    vc.len()
                ));
            }
            let kk = [
                &kc[0], &kc[1], &kc[2], &kc[3], &kc[4], &kc[5], &kc[6], &kc[7],
            ];
            let vv = [
                &vc[0], &vc[1], &vc[2], &vc[3], &vc[4], &vc[5], &vc[6], &vc[7],
            ];
            let sk_views = match snap_heads(&stacks.sk_snap, b) {
                Ok(v) => v,
                Err(e2) => {
                    free_block_rows(gpu, tmps, xs);
                    return Err(e2);
                }
            };
            let sv_views = match snap_heads(&stacks.sv_snap, b) {
                Ok(v) => v,
                Err(e2) => {
                    free_block_rows(gpu, tmps, xs);
                    return Err(e2);
                }
            };
            let ska = [
                &sk_views[0],
                &sk_views[1],
                &sk_views[2],
                &sk_views[3],
                &sk_views[4],
                &sk_views[5],
                &sk_views[6],
                &sk_views[7],
            ];
            let sva = [
                &sv_views[0],
                &sv_views[1],
                &sv_views[2],
                &sv_views[3],
                &sv_views[4],
                &sv_views[5],
                &sv_views[6],
                &sv_views[7],
            ];
            if let Err(e2) =
                gpu.inkling2_swa_snapshot_f16(&kk, &vv, &ska, &sva, ev_lo, ev_hi - ev_lo, rows_v)
            {
                free_block_rows(gpu, tmps, xs);
                return Err(format!("inkling2 gpu fwd: swa snapshot: {e2:?}"));
            }
        }
    }
    let mut curs: Vec<usize> = Vec::with_capacity(b);
    let mut pushes: Vec<Option<GpuTensor>> = Vec::with_capacity(b);
    let mut push_ns: Vec<Option<GpuTensor>> = Vec::with_capacity(b);
    // Row inputs ride the block in `xs` (one per row from the heads loop);
    // both attention arms forward them into `xs_kept` for the unified tail.
    let mut xs_iter = xs.into_iter();
    let mut xs_kept: Vec<GpuTensor> = Vec::with_capacity(b);
    if attn_batched {
        for (r, _) in owned.iter_mut().enumerate() {
            let pos = start + r;
            let tmp = std::mem::take(&mut tmps[r]);
            let x = xs_iter.next().expect("xs");
            let (q_v, k_v, v_v, r_v, s_v) = match (|| -> Result<_, String> {
                Ok((
                    stack_row(&stacks.q32, r, eattn)?,
                    stack_row(&stacks.k32, r, ekv)?,
                    stack_row(&stacks.v32, r, ekv)?,
                    stack_row(&stacks.r32, r, erel)?,
                    stack_row(&stacks.sinks, r, cfg.n_head)?,
                ))
            })() {
                Ok(v) => v,
                Err(e2) => {
                    free_all(gpu, tmp);
                    for t in tmps.drain(r + 1..) {
                        free_all(gpu, t);
                    }
                    for x in xs_iter {
                        let _ = gpu.free_tensor(x);
                    }
                    for x in xs_kept.drain(..) {
                        let _ = gpu.free_tensor(x);
                    }
                    let _ = gpu.free_tensor(x);
                    return Err(e2);
                }
            };
            // Per-row prequel into a private tmp; the qtile runs after the
            // whole block's KV writes. qi/ki/vi come from the head stacks.
            let (qi_v, ki_v, vi_v) = match (
                stack_row(&stacks.qi, r, e),
                stack_row(&stacks.ki, r, e),
                stack_row(&stacks.vi, r, e),
            ) {
                (Ok(a), Ok(b), Ok(c)) => (a, b, c),
                _ => {
                    free_all(gpu, tmp);
                    for t in tmps.drain(r + 1..) {
                        free_all(gpu, t);
                    }
                    for x in xs_iter {
                        let _ = gpu.free_tensor(x);
                    }
                    for x in xs_kept.drain(..) {
                        let _ = gpu.free_tensor(x);
                    }
                    let _ = gpu.free_tensor(x);
                    return Err("inkling2 gpu fwd: head row view short".to_string());
                }
            };
            let mut rtmp = tmp;
            let n_kv = match visit_pre(
                gpu,
                cfg,
                layer,
                gs,
                visit,
                phys,
                pos,
                &qi_v,
                &ki_v,
                &vi_v,
                &q_v,
                &k_v,
                &v_v,
                &r_v,
                &s_v,
                &mut no_cap,
                &mut rtmp,
            ) {
                Ok(n) => n,
                Err(e2) => {
                    free_all(gpu, rtmp);
                    for t in tmps.drain(r + 1..) {
                        free_all(gpu, t);
                    }
                    for x in xs_iter {
                        let _ = gpu.free_tensor(x);
                    }
                    for x in xs_kept.drain(..) {
                        let _ = gpu.free_tensor(x);
                    }
                    let _ = gpu.free_tensor(x);
                    return Err(e2);
                }
            };
            let _ = n_kv;
            tmps[r] = rtmp;
            xs_kept.push(x);
            if let Err(e2) = launch_advance_pos(gpu, &gs.pos_dev) {
                for t in tmps.drain(r..) {
                    free_all(gpu, t);
                }
                for x in xs_iter {
                    let _ = gpu.free_tensor(x);
                }
                for x in xs_kept.drain(..) {
                    let _ = gpu.free_tensor(x);
                }
                return Err(format!("inkling2 gpu fwd: advance pos: {e2}"));
            }
        }
        if let Err(e2) = blocked_attn_launch(gpu, cfg, layer, gs, stacks, visit, b, start, swa_ev) {
            for t in tmps.drain(..) {
                free_all(gpu, t);
            }
            for x in xs_kept.drain(..) {
                let _ = gpu.free_tensor(x);
            }
            return Err(e2);
        }
        // Unified tail below (shared with the fallback arm) consumes
        // xs_kept with the per-row tmps; nothing else runs here.
    } else {
        // Fallback attention: per-row prequel + single-row legacy attention
        // into the attn_out stack rows — same launches and persistent-state
        // order as the seam. The unified tail below closes every row
        // (batched GEMMs, no per-row weight streams).
        for (r, _) in owned.iter_mut().enumerate() {
            let pos = start + r;
            let tmp = std::mem::take(&mut tmps[r]);
            let x = xs_iter.next().expect("xs");
            let (qi_v, ki_v, vi_v, q_v, k_v, v_v, r_v, s_v, a_v) =
                match (|| -> Result<_, String> {
                    Ok((
                        stack_row(&stacks.qi, r, e)?,
                        stack_row(&stacks.ki, r, e)?,
                        stack_row(&stacks.vi, r, e)?,
                        stack_row(&stacks.q32, r, eattn)?,
                        stack_row(&stacks.k32, r, ekv)?,
                        stack_row(&stacks.v32, r, ekv)?,
                        stack_row(&stacks.r32, r, erel)?,
                        stack_row(&stacks.sinks, r, cfg.n_head)?,
                        stack_row(&stacks.attn_out, r, eattn)?,
                    ))
                })() {
                    Ok(v) => v,
                    Err(e2) => {
                        free_all(gpu, tmp);
                        for t in tmps.drain(r + 1..) {
                            free_all(gpu, t);
                        }
                        for x in xs_iter {
                            let _ = gpu.free_tensor(x);
                        }
                        for x in xs_kept.drain(..) {
                            let _ = gpu.free_tensor(x);
                        }
                        let _ = gpu.free_tensor(x);
                        return Err(e2);
                    }
                };
            let mut rtmp = tmp;
            let n_kv = match visit_pre(
                gpu,
                cfg,
                layer,
                gs,
                visit,
                phys,
                pos,
                &qi_v,
                &ki_v,
                &vi_v,
                &q_v,
                &k_v,
                &v_v,
                &r_v,
                &s_v,
                &mut no_cap,
                &mut rtmp,
            ) {
                Ok(n) => n,
                Err(e2) => {
                    free_all(gpu, rtmp);
                    for t in tmps.drain(r + 1..) {
                        free_all(gpu, t);
                    }
                    for x in xs_iter {
                        let _ = gpu.free_tensor(x);
                    }
                    for x in xs_kept.drain(..) {
                        let _ = gpu.free_tensor(x);
                    }
                    let _ = gpu.free_tensor(x);
                    return Err(e2);
                }
            };
            {
                let qv = match view2(&q_v, cfg.n_head, cfg.head_dim) {
                    Ok(v) => v,
                    Err(e2) => {
                        free_all(gpu, rtmp);
                        for t in tmps.drain(r + 1..) {
                            free_all(gpu, t);
                        }
                        for x in xs_iter {
                            let _ = gpu.free_tensor(x);
                        }
                        for x in xs_kept.drain(..) {
                            let _ = gpu.free_tensor(x);
                        }
                        let _ = gpu.free_tensor(x);
                        return Err(e2);
                    }
                };
                if let Err(e2) = visit_attn_single(
                    gpu,
                    cfg,
                    layer,
                    gs,
                    visit,
                    &qv,
                    &r_v,
                    &s_v,
                    n_kv,
                    &a_v,
                    &mut no_cap,
                    &mut rtmp,
                ) {
                    free_all(gpu, rtmp);
                    for t in tmps.drain(r + 1..) {
                        free_all(gpu, t);
                    }
                    for x in xs_iter {
                        let _ = gpu.free_tensor(x);
                    }
                    for x in xs_kept.drain(..) {
                        let _ = gpu.free_tensor(x);
                    }
                    let _ = gpu.free_tensor(x);
                    return Err(e2);
                }
            }
            tmps[r] = rtmp;
            xs_kept.push(x);
            if let Err(e2) = launch_advance_pos(gpu, &gs.pos_dev) {
                for t in tmps.drain(r..) {
                    free_all(gpu, t);
                }
                for x in xs_iter {
                    let _ = gpu.free_tensor(x);
                }
                for x in xs_kept.drain(..) {
                    let _ = gpu.free_tensor(x);
                }
                return Err(format!("inkling2 gpu fwd: advance pos: {e2}"));
            }
        }
    }
    // ---- unified tail (both attention arms): batched GEMMs, no
    // per-row weight streams. attn_out rows are filled (qtile or the
    // single-row loop); tmps/xs_kept carry the per-row state in order.
    if let Err(e2) = blocked_gate_values(gpu, layer, stacks, b, e, eattn) {
        for t in tmps.drain(..) {
            free_all(gpu, t);
        }
        for x in xs_kept.drain(..) {
            let _ = gpu.free_tensor(x);
        }
        return Err(e2);
    }
    // Gate loop: sigmoid + in-place mul over attn_out (tiny, stays per row).
    for r in 0..b {
        let (g_v, a_v) = match (|| -> Result<_, String> {
            Ok((
                stack_row(&stacks.gate32, r, eattn)?,
                stack_row(&stacks.attn_out, r, eattn)?,
            ))
        })() {
            Ok(v) => v,
            Err(e2) => {
                for t in tmps.drain(..) {
                    free_all(gpu, t);
                }
                for x in xs_kept.drain(..) {
                    let _ = gpu.free_tensor(x);
                }
                return Err(e2);
            }
        };
        if let Err(e2) = gate_apply(gpu, &g_v, &a_v) {
            for t in tmps.drain(..) {
                free_all(gpu, t);
            }
            for x in xs_kept.drain(..) {
                let _ = gpu.free_tensor(x);
            }
            return Err(format!("inkling2 gpu fwd: gate apply: {e2:?}"));
        }
    }
    if let Err(e2) = blocked_wo_values(gpu, layer, stacks, b, e, eattn) {
        for t in tmps.drain(..) {
            free_all(gpu, t);
        }
        for x in xs_kept.drain(..) {
            let _ = gpu.free_tensor(x);
        }
        return Err(e2);
    }
    // Tail loop: post-conv + cur/push, then mix/control/norm per row, then
    // the rotation into ffn_rot rows. Router/topk batch after all norms.
    let mut xs_tail = xs_kept.into_iter();
    for (r, o) in owned.iter_mut().enumerate() {
        let msgs = match block_msgs(o, gw, phys, take_msgs) {
            Ok(m) => m,
            Err(e) => {
                for t in tmps.drain(r..) {
                    free_all(gpu, t);
                }
                free_push_pairs(gpu, &mut pushes, &mut push_ns);
                for x in xs_tail {
                    let _ = gpu.free_tensor(x);
                }
                return Err(e);
            }
        };
        let tmp = std::mem::take(&mut tmps[r]);
        let x = xs_tail.next().expect("xs");
        let (wo_v, fn_v) = match (|| -> Result<_, String> {
            Ok((
                stack_row(&stacks.wo32, r, e)?,
                stack_row(&stacks.fnorm, r, e)?,
            ))
        })() {
            Ok(v) => v,
            Err(e2) => {
                free_all(gpu, tmp);
                for t in tmps.drain(r..) {
                    free_all(gpu, t);
                }
                free_push_pairs(gpu, &mut pushes, &mut push_ns);
                for x in xs_tail {
                    let _ = gpu.free_tensor(x);
                }
                let _ = gpu.free_tensor(x);
                return Err(e2);
            }
        };
        let mut rtmp = tmp;
        let (cur, mut push, mut push_normed) = match tail_conv_norm(
            gpu,
            cfg,
            layer,
            gs,
            visit,
            phys,
            msgs,
            mids[r],
            &mut no_cap,
            &mut rtmp,
            &x,
            &wo_v,
            &fn_v,
            &o.blocks_n,
        ) {
            Ok(v) => v,
            Err(e2) => {
                free_all(gpu, rtmp);
                for t in tmps.drain(r + 1..) {
                    free_all(gpu, t);
                }
                free_push_pairs(gpu, &mut pushes, &mut push_ns);
                for x in xs_tail {
                    let _ = gpu.free_tensor(x);
                }
                let _ = gpu.free_tensor(x);
                return Err(e2);
            }
        };
        if layer.xts.is_none() {
            let rot_v = match stack_row(&stacks.ffn_rot, r, e) {
                Ok(v) => v,
                Err(e2) => {
                    free_all(gpu, rtmp);
                    free_push_pair(gpu, &mut push, &mut push_normed);
                    free_push_pairs(gpu, &mut pushes, &mut push_ns);
                    for t in tmps.drain(r + 1..) {
                        free_all(gpu, t);
                    }
                    for x in xs_tail {
                        let _ = gpu.free_tensor(x);
                    }
                    let _ = gpu.free_tensor(x);
                    return Err(e2);
                }
            };
            if let Err(e2) = rotate_x_mq_for(gpu, &layer.experts[0].gate_up, &fn_v, &rot_v, e) {
                free_all(gpu, rtmp);
                free_push_pair(gpu, &mut push, &mut push_normed);
                free_push_pairs(gpu, &mut pushes, &mut push_ns);
                for t in tmps.drain(r + 1..) {
                    free_all(gpu, t);
                }
                for x in xs_tail {
                    let _ = gpu.free_tensor(x);
                }
                let _ = gpu.free_tensor(x);
                return Err(format!("inkling2 gpu fwd: ffn rotate: {e2:?}"));
            }
        }
        tmps[r] = rtmp;
        curs.push(cur);
        pushes.push(push);
        push_ns.push(push_normed);
        if let Err(e2) = gpu.free_tensor(x) {
            for t in tmps.drain(..) {
                free_all(gpu, t);
            }
            free_push_pairs(gpu, &mut pushes, &mut push_ns);
            for x in xs_tail {
                let _ = gpu.free_tensor(x);
            }
            return Err(format!("inkling2 gpu fwd: x free: {e2:?}"));
        }
    }
    if let Err(e2) = blocked_router_values(gpu, cfg, layer, stacks, b, e) {
        for t in tmps.drain(..) {
            free_all(gpu, t);
        }
        free_push_pairs(gpu, &mut pushes, &mut push_ns);
        return Err(e2);
    }
    // MoE middle: per-row single-row middle (grouped MoE never fires on the
    // fixture's qt13 experts — same code the seam runs). `downs[r]` is the
    // down_exp index into tmps[r].
    let mut downs: Vec<usize> = Vec::with_capacity(b);
    for r in 0..b {
        let (fn_v, idx_v, rot_v) = match (|| -> Result<_, String> {
            let fn_v = stack_row(&stacks.fnorm, r, e)?;
            let idx_v = stack_row(&stacks.topk_idx, r, ntopk)?;
            let rot_v = if layer.xts.is_some() {
                None
            } else {
                Some(stack_row(&stacks.ffn_rot, r, e)?)
            };
            Ok((fn_v, idx_v, rot_v))
        })() {
            Ok(v) => v,
            Err(e2) => {
                for t in tmps.drain(..) {
                    free_all(gpu, t);
                }
                free_push_pairs(gpu, &mut pushes, &mut push_ns);
                return Err(e2);
            }
        };
        let mut rtmp = std::mem::take(&mut tmps[r]);
        let result = if let Some(xl) = layer.xts.as_ref() {
            xts_expert_stage(gpu, cfg, xl, &mut rtmp, &fn_v, &idx_v, &mut no_cap)
        } else {
            moe_middle_single(
                gpu,
                cfg,
                layer,
                &mut no_cap,
                &mut rtmp,
                rot_v.as_ref().expect("legacy ffn rotation"),
                &idx_v,
            )
        };
        match result {
            Ok(d) => {
                tmps[r] = rtmp;
                downs.push(d);
            }
            Err(e2) => {
                free_all(gpu, rtmp);
                for t in tmps.drain(..) {
                    free_all(gpu, t);
                }
                free_push_pairs(gpu, &mut pushes, &mut push_ns);
                return Err(e2);
            }
        }
    }
    for (r, o) in owned.iter_mut().enumerate() {
        let (idx_v, w_v, moe_v) = match (|| -> Result<_, String> {
            Ok((
                stack_row(&stacks.topk_idx, r, ntopk)?,
                stack_row(&stacks.topk_w, r, ntopk)?,
                stack_row(&stacks.mmoe, r, e)?,
            ))
        })() {
            Ok(v) => v,
            Err(e2) => {
                for t in tmps.drain(r..) {
                    free_all(gpu, t);
                }
                free_push_pairs(gpu, &mut pushes, &mut push_ns);
                return Err(e2);
            }
        };
        let mut rtmp = std::mem::take(&mut tmps[r]);
        let vo = match moe_postfix(
            gpu,
            cfg,
            layer,
            gs,
            visit,
            &mut no_cap,
            &mut rtmp,
            curs[r],
            downs[r],
            &idx_v,
            &w_v,
            &moe_v,
            pushes[r].take(),
            push_ns[r].take(),
        ) {
            Ok(v) => v,
            Err(e2) => {
                free_all(gpu, rtmp);
                for t in tmps.drain(r + 1..) {
                    free_all(gpu, t);
                }
                free_push_pairs(gpu, &mut pushes, &mut push_ns);
                return Err(e2);
            }
        };
        if let Some(p) = vo.push {
            o.blocks.push(p);
        }
        if let Some(p) = vo.push_normed {
            o.blocks_n.push(p);
        }
        o.x = Some(vo.visit_out);
    }
    Ok(())
}

/// Blocked prefill body: the `drive_body` pass skeleton over one `Owned`
/// per row, with visits executed width-first. Intermediate rows never
/// consume the output tail; the last row runs `drive_final` when it is the
/// prompt-final token.
#[allow(clippy::too_many_arguments)]
fn drive_body_blocked(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    gw: &Inkling2GpuWeights,
    gs: &mut Inkling2GpuState,
    token_ids: &[u32],
    start: usize,
    prompt_len: usize,
) -> Result<Vec<f32>, String> {
    let b = token_ids.len();
    if b <= 1 {
        return Err("inkling2 gpu fwd: blocked body needs B >= 2".to_string());
    }
    if start + b > prompt_len {
        return Err(format!(
            "inkling2 gpu fwd: block [{start}, {}) exceeds prompt {prompt_len}",
            start + b
        ));
    }
    if gw.layers.len() != cfg.n_phys {
        return Err(format!(
            "inkling2 gpu fwd: {} layers != n_phys {}",
            gw.layers.len(),
            cfg.n_phys
        ));
    }
    if start != gs.seq_pos {
        return Err(format!(
            "inkling2 gpu fwd: block start {start} != seq_pos {} (out-of-order)",
            gs.seq_pos
        ));
    }
    ensure_kernels(gpu)?;
    let e = cfg.n_embd;
    let core_len = cfg.loop_cfg.core_end - cfg.loop_cfg.core_start;
    if gw.cross_pass.len() != core_len {
        return Err(format!(
            "inkling2 gpu fwd: {} cross_pass != core_len {core_len}",
            gw.cross_pass.len()
        ));
    }
    let cleanup = |gpu: &mut Gpu, owned: Vec<Owned>| {
        for o in owned {
            o.free(gpu);
        }
    };
    // Embed + norm per row. The inner scope makes every failure clean up
    // earlier rows; per-row failures also release their own partial pair.
    let mut owned: Vec<Owned> = Vec::with_capacity(b);
    let embed_result = (|| -> Result<(), String> {
        for tid in token_ids {
            let row = gw.embed_packed.row_f32(*tid)?;
            let emb = hip(gpu.upload_f32(&row, &[e]), "embed upload")?;
            let tok = match gpu.alloc_tensor(&[e], DType::F32) {
                Ok(t) => t,
                Err(e2) => {
                    let _ = gpu.free_tensor(emb);
                    return Err(format!("inkling2 gpu fwd: tok: {e2:?}"));
                }
            };
            if let Err(e2) = gpu.rmsnorm_f32(&emb, &gw.embed_norm, &tok, cfg.rms_norm_eps) {
                let _ = gpu.free_tensor(emb);
                let _ = gpu.free_tensor(tok);
                return Err(format!("inkling2 gpu fwd: embed norm: {e2:?}"));
            }
            if let Err(e2) = gpu.free_tensor(emb) {
                let _ = gpu.free_tensor(tok);
                return Err(format!("inkling2 gpu fwd: embed free: {e2:?}"));
            }
            let mut o = Owned::default();
            o.x = Some(tok);
            owned.push(o);
        }
        Ok(())
    })();
    if let Err(e2) = embed_result {
        cleanup(gpu, owned);
        return Err(e2);
    }
    // Projection stacks, shared by every visit of the block.
    let m_q = gw.layers[0].wq.m;
    let m_kv = gw.layers[0].wk.m;
    let m_r = gw.layers[0].wr.m;
    let mut stacks = match BlockStacks::alloc(
        gpu,
        b,
        e,
        m_q,
        m_kv,
        m_r,
        cfg.n_head,
        cfg.n_experts,
        cfg.n_expert_used,
    ) {
        Ok(s) => s,
        Err(e2) => {
            cleanup(gpu, owned);
            return Err(e2);
        }
    };
    let result = (|| -> Result<Vec<f32>, String> {
        macro_rules! step_block {
            ($expr:expr) => {
                match $expr {
                    Ok(v) => v,
                    Err(e2) => return Err(e2),
                }
            };
        }
        let mut visit = 0usize;
        // Prelude visits 0..core_start (no messages).
        for _ in 0..cfg.loop_cfg.core_start {
            if let Err(e2) = blocked_step_visit(
                gpu,
                cfg,
                gw,
                gs,
                &mut owned,
                start,
                visit,
                None,
                &mut stacks,
            ) {
                return Err(e2);
            }
            visit += 1;
        }
        // Snapshot prelude stream for the pass resets, per row.
        for o in owned.iter_mut() {
            o.prelude_x = Some(step_block!(dup(gpu, o.x.as_ref().expect("x"), "prelude_x")));
            assert_eq!(
                o.blocks.len(),
                o.blocks_n.len(),
                "inkling2 gpu fwd: prelude drain split"
            );
            for (blk, bn) in o.blocks.drain(..).zip(o.blocks_n.drain(..)) {
                let d = step_block!(dup(gpu, &blk, "prelude_block"));
                let dn = step_block!(dup(gpu, &bn, "prelude_block_n"));
                o.prelude_blocks.push(d);
                o.prelude_blocks_n.push(dn);
                step_block!(hip(gpu.free_tensor(blk), "block free"));
                step_block!(hip(gpu.free_tensor(bn), "block_n free"));
            }
            for (blk, bn) in o.prelude_blocks.iter().zip(o.prelude_blocks_n.iter()) {
                o.prev_blocks.push(step_block!(dup(gpu, blk, "prev_block")));
                o.prev_blocks_n
                    .push(step_block!(dup(gpu, bn, "prev_block_n")));
            }
            o.prev_out = Some(step_block!(dup(
                gpu,
                o.prelude_x.as_ref().expect("x"),
                "prev_out"
            )));
        }
        // Loop passes.
        for pass in 0..cfg.loop_cfg.num_passes {
            for o in owned.iter_mut() {
                if let Some(x) = o.x.take() {
                    step_block!(hip(gpu.free_tensor(x), "x free"));
                }
                assert_eq!(
                    o.blocks.len(),
                    o.blocks_n.len(),
                    "inkling2 gpu fwd: pass drain split"
                );
                for (blk, bn) in o.blocks.drain(..).zip(o.blocks_n.drain(..)) {
                    step_block!(hip(gpu.free_tensor(blk), "block free"));
                    step_block!(hip(gpu.free_tensor(bn), "block_n free"));
                }
                o.x = Some(step_block!(dup(
                    gpu,
                    o.prelude_x.as_ref().expect("x"),
                    "pass x"
                )));
                for (blk, bn) in o.prelude_blocks.iter().zip(o.prelude_blocks_n.iter()) {
                    o.blocks.push(step_block!(dup(gpu, blk, "pass block")));
                    o.blocks_n.push(step_block!(dup(gpu, bn, "pass block_n")));
                }
            }
            for i in 0..core_len {
                let take = if pass > 0 { Some((i, 0)) } else { None };
                if let Err(e2) = blocked_step_visit(
                    gpu,
                    cfg,
                    gw,
                    gs,
                    &mut owned,
                    start,
                    visit,
                    take,
                    &mut stacks,
                ) {
                    return Err(e2);
                }
                visit += 1;
            }
            if pass + 1 < cfg.loop_cfg.num_passes {
                for o in owned.iter_mut() {
                    assert_eq!(
                        o.prev_blocks.len(),
                        o.prev_blocks_n.len(),
                        "inkling2 gpu fwd: prev drain split"
                    );
                    for (blk, bn) in o.prev_blocks.drain(..).zip(o.prev_blocks_n.drain(..)) {
                        step_block!(hip(gpu.free_tensor(blk), "prev block free"));
                        step_block!(hip(gpu.free_tensor(bn), "prev block_n free"));
                    }
                    for (blk, bn) in o.blocks.iter().zip(o.blocks_n.iter()) {
                        o.prev_blocks.push(step_block!(dup(gpu, blk, "prev block")));
                        o.prev_blocks_n
                            .push(step_block!(dup(gpu, bn, "prev block_n")));
                    }
                    if let Some(p) = o.prev_out.take() {
                        step_block!(hip(gpu.free_tensor(p), "prev out free"));
                    }
                    o.prev_out = Some(step_block!(dup(gpu, o.x.as_ref().expect("x"), "prev out")));
                }
            }
        }
        // Post-loop layers core_end..n_phys once (stream continues).
        for _ in cfg.loop_cfg.core_end..cfg.n_phys {
            if let Err(e2) = blocked_step_visit(
                gpu,
                cfg,
                gw,
                gs,
                &mut owned,
                start,
                visit,
                None,
                &mut stacks,
            ) {
                return Err(e2);
            }
            visit += 1;
        }
        assert_eq!(
            visit,
            cfg.loop_cfg.n_slot(),
            "inkling2 gpu fwd: visit count"
        );
        // Intermediate rows never consume logits; the last row runs the output
        // tail only when it is the prompt-final token.
        if start + b != prompt_len {
            gs.seq_pos = start + b;
            return Ok(Vec::new());
        }
        let last = b - 1;
        let logits = step_block!(drive_final(
            gpu,
            cfg,
            gw,
            gs,
            &mut owned[last],
            start + last
        ));
        Ok(logits)
    })();
    stacks.free(gpu);
    cleanup(gpu, owned);
    result
}

/// Prefill a whole prompt with multi-row blocks and return the last token's
/// logits.
///
/// Block driver over the single-row loop: `reset` once, then one block per
/// `prefill_blocks` partition. A block wider than one row whose width fits
/// the gated batched path takes `drive_body_blocked` (kernel #1 serves the
/// four QKV/R projections width-first, widen-step); every other block falls
/// back to the single-row loop. B == 1 blocks always take the seam, so the
/// fallback stays bit-identical to the eager loop by construction.
pub fn prefill_prompt_gpu_blocked(
    bundle: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    token_ids: &[u32],
    chunk_tokens: usize,
) -> Result<Vec<f32>, String> {
    if token_ids.is_empty() {
        return Err("inkling2 gpu fwd: batched prefill of empty prompt".to_string());
    }
    bundle
        .gpu_state
        .as_mut()
        .ok_or("inkling2 gpu fwd: no gpu state")?
        .reset_session_state(gpu)?;
    let cfg = bundle.config.clone();
    let gw = bundle
        .gpu_weights
        .as_ref()
        .ok_or("inkling2 gpu fwd: no gpu weights")?;
    let gs = bundle
        .gpu_state
        .as_mut()
        .ok_or("inkling2 gpu fwd: no gpu state")?;
    let mut last: Vec<f32> = Vec::new();
    for (start, len) in prefill_blocks(token_ids.len(), chunk_tokens) {
        if BATCHED_QKVR_GATED && len > 1 {
            last = drive_body_blocked(
                gpu,
                &cfg,
                gw,
                gs,
                &token_ids[start..start + len],
                start,
                token_ids.len(),
            )?;
            continue;
        }
        // Single-row fallback: beta's own token loop (the eager path the
        // parity gate compares against).
        for i in start..start + len {
            last = drive(
                gpu,
                &cfg,
                gw,
                gs,
                token_ids[i],
                i,
                None,
                &mut None,
                &mut None,
            )?;
        }
    }
    Ok(last)
}

#[cfg(test)]
mod tests {
    use crate::gpu::{swa_gather_map, swa_slot, swa_window_len, SWA_WINDOW_ROWS};

    /// Simulate `n` sequential ring writes into a tag array, as `run_visit`
    /// does (`tags[slot] = pos`), then check the gather map at `pos`.
    fn written_tags(rows: usize, n: usize) -> Vec<u32> {
        let mut tags = vec![0u32; rows];
        for pos in 0..n {
            tags[swa_slot(pos, rows)] = pos as u32;
        }
        tags
    }

    #[test]
    fn ring_is_identity_below_window() {
        // Bit-identity hard requirement: pos < 1024 keeps slot == pos and
        // dist == pos - k exactly as the pre-ring path computed them.
        for pos in [0usize, 1, 7, 512, 1023] {
            let tags = written_tags(SWA_WINDOW_ROWS, pos + 1);
            let (slots, dists) = swa_gather_map(&tags, pos, SWA_WINDOW_ROWS).unwrap();
            assert_eq!(slots.len(), pos + 1);
            for (k, (&s, &d)) in slots.iter().zip(dists.iter()).enumerate() {
                assert_eq!(s as usize, k, "pos {pos} slot {k}");
                assert_eq!(d as usize, pos - k, "pos {pos} dist {k}");
            }
        }
    }

    #[test]
    fn ring_wraps_with_true_positions() {
        // First wrap: pos 1024 evicts key 0; window is keys 1..=1024.
        let tags = written_tags(SWA_WINDOW_ROWS, 1025);
        let (slots, dists) = swa_gather_map(&tags, 1024, SWA_WINDOW_ROWS).unwrap();
        assert_eq!(slots.len(), SWA_WINDOW_ROWS);
        assert_eq!(slots[0] as usize, 1);
        assert_eq!(slots[SWA_WINDOW_ROWS - 1] as usize, 0);
        assert_eq!(dists[0] as usize, 1023);
        assert_eq!(dists[SWA_WINDOW_ROWS - 1], 0);
        // Order preserved across the physical seam: consecutive logical rows
        // are consecutive true positions.
        for w in slots.windows(2) {
            let (a, b) = (w[0] as usize, w[1] as usize);
            assert_eq!(tags[b] - tags[a], 1);
        }
        // Deep wrap: pos 2500 holds keys 1477..=2500.
        let tags = written_tags(SWA_WINDOW_ROWS, 2501);
        let (slots, dists) = swa_gather_map(&tags, 2500, SWA_WINDOW_ROWS).unwrap();
        assert_eq!(slots.len(), SWA_WINDOW_ROWS);
        assert_eq!(tags[slots[0] as usize], 1477);
        assert_eq!(tags[*slots.last().unwrap() as usize], 2500);
        assert_eq!(dists[0] as usize, 1023);
        assert_eq!(dists[dists.len() - 1], 0);
    }

    #[test]
    fn ring_window_len_caps_at_rows() {
        assert_eq!(swa_window_len(0, SWA_WINDOW_ROWS), 1);
        assert_eq!(swa_window_len(1023, SWA_WINDOW_ROWS), 1024);
        assert_eq!(swa_window_len(1024, SWA_WINDOW_ROWS), 1024);
        assert_eq!(swa_window_len(5000, SWA_WINDOW_ROWS), 1024);
        // Small rings (e.g. max_seq=64 test loads) wrap at their own size.
        assert_eq!(swa_window_len(63, 64), 64);
        assert_eq!(swa_window_len(64, 64), 64);
        assert_eq!(swa_slot(64, 64), 0);
        assert_eq!(swa_slot(65, 64), 1);
    }

    #[test]
    fn ring_tag_mismatch_fails_closed() {
        // Stale tag (skipped write) is an error, never a silent wrong gather.
        let mut tags = written_tags(SWA_WINDOW_ROWS, 1025);
        tags[swa_slot(500, SWA_WINDOW_ROWS)] = 0;
        assert!(swa_gather_map(&tags, 1024, SWA_WINDOW_ROWS).is_err());
        // Shape mismatches fail closed too.
        assert!(swa_gather_map(&tags[..100], 50, SWA_WINDOW_ROWS).is_err());
        assert!(swa_gather_map(&tags, 50, 0).is_err());
    }

    /// D1 host model of `inkling2_gf_build_slots_dists`: one "thread" per k
    /// over the device pos word. Must equal `swa_gather_map` (the H2D path
    /// it replaces) on every (pos, rows), including ring wrap.
    fn slots_dists_kernel_model(pos: usize, rows: usize) -> (Vec<i32>, Vec<i32>) {
        assert!(rows > 0);
        let n_kv = (pos + 1).min(rows);
        let start = pos + 1 - n_kv;
        let mut slots = vec![0i32; n_kv];
        let mut dists = vec![0i32; n_kv];
        // One thread per k, same arithmetic and order as the kernel.
        for k in 0..n_kv {
            let kp = start + k;
            slots[k] = (kp % rows) as i32;
            dists[k] = (pos - kp) as i32;
        }
        (slots, dists)
    }

    /// Drive-faithful tags: slot s holds the last key_pos written there.
    fn tags_for(pos: usize, rows: usize) -> Vec<u32> {
        let n_kv = (pos + 1).min(rows);
        let start = pos + 1 - n_kv;
        let mut tags = vec![0u32; rows];
        for kp in start..=pos {
            tags[swa_slot(kp, rows)] = kp as u32;
        }
        tags
    }

    #[test]
    fn slots_kernel_model_matches_gather_map() {
        // Full-visit rows (max_seq-like), SWA rows, degenerate rows.
        for rows in [1usize, 3, 1024, 4096, 32768] {
            for pos in [
                0usize, 1, 2, 7, 63, 255, 1023, 1024, 1025, 2500, 5000, 100000,
            ] {
                let tags = tags_for(pos, rows);
                let (want_slots, want_dists) =
                    swa_gather_map(&tags, pos, rows).expect("gather map");
                let (got_slots, got_dists) = slots_dists_kernel_model(pos, rows);
                assert_eq!(got_slots, want_slots, "pos={pos} rows={rows}: slots");
                assert_eq!(got_dists, want_dists, "pos={pos} rows={rows}: dists");
                // Window invariants the attention kernels rely on.
                assert_eq!(got_slots.len(), (pos + 1).min(rows));
                assert_eq!(*got_dists.first().unwrap(), (got_slots.len() - 1) as i32);
                assert_eq!(*got_dists.last().unwrap(), 0);
            }
        }
        // Ring wrap: oldest key_pos is 1477 in physical slot 453.
        let (slots, dists) = slots_dists_kernel_model(2500, 1024);
        assert_eq!(slots.len(), 1024);
        assert_eq!(slots[0], 1477 % 1024);
        assert_eq!(*slots.last().unwrap(), 2500 % 1024);
        assert_eq!(dists[0], 1023);
        assert_eq!(*dists.last().unwrap(), 0);
    }

    // View-cut bounds gates (upstream #1192): the reuse pool best-fit may
    // hand a LARGER buffer to an exact-size request. The view takes the
    // logical prefix; a short buffer fails closed instead of panicking.
    use super::{full_layer_tau, sub_buf, view2};
    use hip_bridge::DeviceBuffer;
    use rdna_compute::{DType, GpuTensor};

    fn fake_tensor(backing: &[u8], shape: Vec<usize>, dtype: DType) -> GpuTensor {
        GpuTensor {
            buf: unsafe {
                DeviceBuffer::from_raw(backing.as_ptr() as *mut std::ffi::c_void, backing.len())
            },
            shape,
            dtype,
        }
    }

    #[test]
    fn view2_oversized_pool_buffer_takes_logical_prefix() {
        let backing = vec![0xABu8; 15704];
        let t = fake_tensor(&backing, vec![3926], DType::F32);
        let v = view2(&t, 1, 3072).expect("oversized buffer must view");
        assert_eq!(v.shape, vec![1, 3072]);
        assert_eq!(v.dtype, DType::F32);
        assert_eq!(v.buf.size(), 12288);
        assert_eq!(v.buf.as_ptr(), t.buf.as_ptr());
    }

    #[test]
    fn view2_exact_size_unchanged() {
        let backing = vec![0u8; 12288];
        let t = fake_tensor(&backing, vec![3072], DType::F32);
        let v = view2(&t, 1, 3072).expect("exact buffer must view");
        assert_eq!(v.shape, vec![1, 3072]);
        assert_eq!(v.buf.size(), 12288);
        assert_eq!(v.buf.as_ptr(), t.buf.as_ptr());
    }

    #[test]
    fn view2_short_buffer_fails_closed() {
        let backing = vec![0u8; 12284];
        let t = fake_tensor(&backing, vec![3071], DType::F32);
        assert!(view2(&t, 1, 3072).is_err());
    }

    #[test]
    fn view2_wrong_dtype_fails_closed() {
        let backing = vec![0u8; 12288];
        let t = fake_tensor(&backing, vec![3072], DType::F16);
        assert!(view2(&t, 1, 3072).is_err());
    }

    #[test]
    fn view2_extent_overflow_fails_closed() {
        let backing = vec![0u8; 16];
        let t = fake_tensor(&backing, vec![4], DType::F32);
        assert!(view2(&t, usize::MAX, usize::MAX).is_err());
    }

    #[test]
    fn sub_buf_never_clamps() {
        let backing = vec![0u8; 100];
        let t = fake_tensor(&backing, vec![25], DType::F32);
        assert_eq!(sub_buf(&t, 0, 100, "t").unwrap().size(), 100);
        assert_eq!(sub_buf(&t, 0, 64, "t").unwrap().size(), 64);
        assert_eq!(sub_buf(&t, 36, 64, "t").unwrap().size(), 64);
        assert!(sub_buf(&t, 0, 101, "t").is_err());
        assert!(sub_buf(&t, 37, 64, "t").is_err());
        assert!(sub_buf(&t, usize::MAX, 1, "t").is_err());
    }

    #[test]
    fn tau_skips_scaling_below_floor() {
        for pos in [0, 1, 4096, 8191] {
            assert_eq!(full_layer_tau(pos, 8192, 0.1), Ok(None), "pos {pos}");
        }
    }

    #[test]
    fn tau_matches_oracle_past_floor() {
        for pos in [8192, 8193, 16383, 24800, 100_000] {
            let got = full_layer_tau(pos, 8192, 0.1)
                .expect("scales")
                .expect("some");
            let want = 1.0 + 0.1 * (((pos + 1) as f64 / 8192.0).max(1.0)).ln();
            assert!(
                (got as f64 - want).abs() < 1e-6,
                "pos {pos}: {got} vs {want}"
            );
        }
        let first = full_layer_tau(8192, 8192, 0.1).unwrap().unwrap();
        assert!((first - 1.0000122).abs() < 1e-6, "{first}");
    }

    #[test]
    fn tau_absurd_stays_fail_closed() {
        assert!(full_layer_tau(9000, 8192, f32::NAN).is_err());
        assert!(full_layer_tau(200_000_000, 8192, 0.1).is_err());
        assert!(full_layer_tau(200_000_000, 8192, -1.0).is_err());
    }
}
