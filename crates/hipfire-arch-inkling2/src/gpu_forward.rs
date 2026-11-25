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
    Inkling2GpuWeights, INKLING2_CONV_WIDTH,
};
use hip_bridge::KernargBlob;
use hipfire_runtime::llama::{rotate_x_mq_for, weight_gemv};
use rdna_compute::ffn_activation::{
    CombineOrder, FfnActivation, FfnActivationRows, FfnAwq, MoeCombine, MoeRouterTopK, RouterKind,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::HashMap;
use std::ffi::c_void;

// ---------------------------------------------------------------------------
// Dedicated HIP (the only two kernels in this file)
// ---------------------------------------------------------------------------

const INKLING2_GF_MODULE: &str = "inkling2_gpu_forward";

const INKLING2_GF_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

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
    Ok(())
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

/// Non-owning `[rows, cols]` F32 view of a whole buffer.
fn view2(t: &GpuTensor, rows: usize, cols: usize) -> GpuTensor {
    assert_eq!(t.dtype, DType::F32, "inkling2 gpu fwd: view of non-F32");
    assert_eq!(
        t.buf.size(),
        rows * cols * 4,
        "inkling2 gpu fwd: view extent"
    );
    GpuTensor {
        buf: t.buf.byte_view(0, rows * cols * 4),
        shape: vec![rows, cols],
        dtype: DType::F32,
    }
}

/// Non-owning `[hd]` F32 view of one head row inside a `[heads*hd]` buffer.
fn head_view(t: &GpuTensor, h: usize, hd: usize) -> GpuTensor {
    assert_eq!(
        t.dtype,
        DType::F32,
        "inkling2 gpu fwd: head view of non-F32"
    );
    GpuTensor {
        buf: t.buf.byte_view(h * hd * 4, hd * 4),
        shape: vec![hd],
        dtype: DType::F32,
    }
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
    let view = GpuTensor {
        buf: src_2d.buf.byte_view(row * e * 4, e * 4),
        shape: vec![e],
        dtype: DType::F32,
    };
    hip(
        gpu.copy_rows_strided_f32(&view, dst, 1, e, e, e, 0, None),
        what,
    )
}

fn dup(gpu: &mut Gpu, src: &GpuTensor, what: &str) -> Result<GpuTensor, String> {
    let n = src.buf.size() / 4;
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

/// AttnRes-style mix over `bank_rows + cur` with a `[qr, e]` query.
/// Self-contained scratch: only the `[qr, e]` output escapes.
/// Peak AttnRes state count is 5 (post-loop visits, final output mix),
/// inside 02's 6-state kernel cap.
fn gpu_mix(
    gpu: &mut Gpu,
    bank_rows: &[&GpuTensor],
    cur: &GpuTensor,
    query: &GpuTensor,
    e: usize,
    qr: usize,
    eps: f32,
) -> Result<GpuTensor, String> {
    let n = bank_rows.len() + 1;
    if !(1..=6).contains(&n) {
        return Err(format!("inkling2 gpu fwd: mix states {n} exceeds 6"));
    }
    let mut tmp = Vec::new();
    let fail = |gpu: &mut Gpu, tmp: Vec<GpuTensor>, msg: String| -> Result<GpuTensor, String> {
        free_all(gpu, tmp);
        Err(msg)
    };
    let b = match at(gpu, &mut tmp, &[n, e], "mix_bank") {
        Ok(i) => i,
        Err(msg) => return fail(gpu, tmp, msg),
    };
    for (i, r) in bank_rows.iter().chain(std::iter::once(&cur)).enumerate() {
        if let Err(e2) = gpu.copy_rows_strided_f32(r, &tmp[b], 1, e, e, e, i * e, None) {
            return fail(gpu, tmp, format!("inkling2 gpu fwd: mix bank row: {e2:?}"));
        }
    }
    let normed = match at(gpu, &mut tmp, &[n, e], "mix_normed") {
        Ok(i) => i,
        Err(msg) => return fail(gpu, tmp, msg),
    };
    let out = match at(gpu, &mut tmp, &[qr, e], "mix_out") {
        Ok(i) => i,
        Err(msg) => return fail(gpu, tmp, msg),
    };
    let qv = view2(query, qr, e);
    if let Err(e2) = gpu.inkling2_mix_f32_gpu(&tmp[b], &qv, &tmp[normed], &tmp[out], n, e, qr, eps)
    {
        return fail(gpu, tmp, format!("inkling2 gpu fwd: mix: {e2:?}"));
    }
    let got = tmp.pop().expect("mix out is last");
    free_all(gpu, tmp);
    Ok(got)
}

/// One fused controller step. Self-contained scratch; only the output escapes.
fn gpu_control(
    gpu: &mut Gpu,
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
    let scratch = match at(gpu, &mut tmp, &[4 * e + rank], "ctrl_scratch") {
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
        &tmp[scratch],
        e,
        rank,
    ) {
        return fail(gpu, tmp, format!("inkling2 gpu fwd: control: {e2:?}"));
    }
    let _ = tmp.remove(scratch);
    let got = tmp.remove(out);
    free_all(gpu, tmp);
    Ok(got)
}

// ---------------------------------------------------------------------------
// One attention+MoE visit on the GPU
// ---------------------------------------------------------------------------

struct VisitOut {
    visit_out: GpuTensor,
    push: Option<GpuTensor>,
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
    visit: usize,
    phys: usize,
    pos: usize,
    msgs: Option<(
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
    let eps = cfg.rms_norm_eps;
    let swa = cfg.swa_layers[phys];
    let mut tmp: Vec<GpuTensor> = Vec::new();
    // Block-push made before the MoE half; freed with tmp on error.
    let mut push: Option<GpuTensor> = None;
    macro_rules! step {
        ($expr:expr, $what:literal) => {
            match $expr {
                Ok(v) => v,
                Err(e) => {
                    if let Some(p) = push.take() {
                        let _ = gpu.free_tensor(p);
                    }
                    free_all(gpu, tmp);
                    return Err(format!("inkling2 gpu fwd: {}: {e:?}", $what));
                }
            }
        };
    }
    // No bare `?` below this point: every failure must free tmp/push.

    if cfg.conv_kernel != 4 {
        free_all(gpu, tmp);
        return Err(format!(
            "inkling2 gpu fwd: conv_kernel {} != 4",
            cfg.conv_kernel
        ));
    }
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
                free_all(gpu, tmp);
                return Err(format!(
                    "inkling2 gpu fwd: visit {visit} phys {phys} missing attn_query"
                ));
            }
        };
        let total = aq.buf.size() / 4;
        if total % e != 0 {
            free_all(gpu, tmp);
            return Err(format!(
                "inkling2 gpu fwd: attn_query len {total} not a multiple of {e}"
            ));
        }
        // Mirrors forward.rs: 3 rows when the query carries them, else 1.
        let qr = if total / e >= 3 { 3 } else { 1 };
        let bank_rows: Vec<&GpuTensor> = blocks.iter().collect();
        let mixed = step!(gpu_mix(gpu, &bank_rows, x, aq, e, qr, eps), "attn mix");
        tmp.push(mixed);
        let mi = tmp.len() - 1;
        if qr >= 3 {
            let r0 = step!(at(gpu, &mut tmp, &[e], "base_q"), "alloc");
            let r1 = step!(at(gpu, &mut tmp, &[e], "base_k"), "alloc");
            let r2 = step!(at(gpu, &mut tmp, &[e], "base_v"), "alloc");
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
            let r0 = step!(at(gpu, &mut tmp, &[e], "base_qkv"), "alloc");
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
        Some((bank, prev_out, cross, ctrls)) => {
            let (bqi, bki, bvi) = match (bq, bk, bv) {
                (Some(a), Some(b), Some(c)) => (a, b, c),
                _ => {
                    free_all(gpu, tmp);
                    return Err("inkling2 gpu fwd: phys 0 has no base".to_string());
                }
            };
            if cross.buf.size() / 4 != 4 * e {
                free_all(gpu, tmp);
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
                let bank_refs: Vec<&GpuTensor> = bank.iter().collect();
                let msg = step!(
                    gpu_mix(gpu, &bank_refs, prev_out, &qj, e, 1, eps),
                    "msg mix"
                );
                tmp.push(msg);
                let mi = tmp.len() - 1;
                let row = step!(at(gpu, &mut tmp, &[e], "msg"), "alloc");
                step!(
                    gpu.copy_rows_strided_f32(&tmp[mi], &tmp[row], 1, e, e, e, 0, None),
                    "msg row"
                );
                m[j] = row;
            }
            let q = step!(
                gpu_control(
                    gpu,
                    &tmp[bqi],
                    &tmp[m[0]],
                    &ctrls[0],
                    e,
                    cfg.loop_cfg.gate_rank
                ),
                "q control"
            );
            tmp.push(q);
            let qi = tmp.len() - 1;
            let k = step!(
                gpu_control(
                    gpu,
                    &tmp[bki],
                    &tmp[m[1]],
                    &ctrls[1],
                    e,
                    cfg.loop_cfg.gate_rank
                ),
                "k control"
            );
            tmp.push(k);
            let ki = tmp.len() - 1;
            let v = step!(
                gpu_control(
                    gpu,
                    &tmp[bvi],
                    &tmp[m[2]],
                    &ctrls[2],
                    e,
                    cfg.loop_cfg.gate_rank
                ),
                "v control"
            );
            tmp.push(v);
            let vi = tmp.len() - 1;
            step!(cap_put(cap, gpu, "base_q", &tmp[bqi]), "capture");
            step!(cap_put(cap, gpu, "q_ctrl", &tmp[qi]), "capture");
            mids = Some(m);
            (qi, ki, vi)
        }
        None if phys == 0 => {
            let xi = step!(at(gpu, &mut tmp, &[e], "x_dup"), "alloc");
            step!(
                gpu.copy_rows_strided_f32(x, &tmp[xi], 1, e, e, e, 0, None),
                "x dup"
            );
            (xi, xi, xi)
        }
        None => match (bq, bk, bv) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => {
                free_all(gpu, tmp);
                return Err("inkling2 gpu fwd: phys 0 has no base".to_string());
            }
        },
    };
    // Norms: SWA applies attn_norm to all three inputs (pass > 0 SWA has
    // three different controllers, so q/k/v are already distinct); full
    // layers use the three input norms.
    let (qi, ki, vi) = if swa {
        let an = match layer.attn_norm.as_ref() {
            Some(t) => t,
            None => {
                free_all(gpu, tmp);
                return Err("inkling2 gpu fwd: SWA layer missing attn_norm".to_string());
            }
        };
        let q = step!(at(gpu, &mut tmp, &[e], "q_inp"), "alloc");
        let k = step!(at(gpu, &mut tmp, &[e], "k_inp"), "alloc");
        let v = step!(at(gpu, &mut tmp, &[e], "v_inp"), "alloc");
        step!(gpu.rmsnorm_f32(&tmp[qc0], an, &tmp[q], eps), "q norm");
        step!(gpu.rmsnorm_f32(&tmp[kc0], an, &tmp[k], eps), "k norm");
        step!(gpu.rmsnorm_f32(&tmp[vc0], an, &tmp[v], eps), "v norm");
        (q, k, v)
    } else {
        let (qn, kn, vn) = match (
            &layer.q_input_norm,
            &layer.k_input_norm,
            &layer.v_input_norm,
        ) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => {
                free_all(gpu, tmp);
                return Err("inkling2 gpu fwd: full layer missing input norms".to_string());
            }
        };
        let q = step!(at(gpu, &mut tmp, &[e], "q_inp"), "alloc");
        let k = step!(at(gpu, &mut tmp, &[e], "k_inp"), "alloc");
        let v = step!(at(gpu, &mut tmp, &[e], "v_inp"), "alloc");
        step!(gpu.rmsnorm_f32(&tmp[qc0], qn, &tmp[q], eps), "q norm");
        step!(gpu.rmsnorm_f32(&tmp[kc0], kn, &tmp[k], eps), "k norm");
        step!(gpu.rmsnorm_f32(&tmp[vc0], vn, &tmp[v], eps), "v norm");
        (q, k, v)
    };
    step!(cap_put(cap, gpu, "q_inp", &tmp[qi]), "capture");
    step!(cap_put(cap, gpu, "k_inp", &tmp[ki]), "capture");
    step!(cap_put(cap, gpu, "v_inp", &tmp[vi]), "capture");
    // Projections.
    let q = step!(at(gpu, &mut tmp, &[eattn], "q"), "alloc");
    let k_raw = step!(at(gpu, &mut tmp, &[ekv], "k_raw"), "alloc");
    let v_raw = step!(at(gpu, &mut tmp, &[ekv], "v_raw"), "alloc");
    let r = step!(at(gpu, &mut tmp, &[erel], "r_raw"), "alloc");
    step!(weight_gemv(gpu, &layer.wq, &tmp[qi], &tmp[q]), "wq");
    step!(weight_gemv(gpu, &layer.wk, &tmp[ki], &tmp[k_raw]), "wk");
    step!(weight_gemv(gpu, &layer.wv, &tmp[vi], &tmp[v_raw]), "wv");
    step!(weight_gemv(gpu, &layer.wr, &tmp[qi], &tmp[r]), "wr");
    step!(cap_put(cap, gpu, "k_raw", &tmp[k_raw]), "capture");
    step!(cap_put(cap, gpu, "v_raw", &tmp[v_raw]), "capture");
    // Convs (static on SWA; dynamic taps on full layers).
    let k_conv = step!(at(gpu, &mut tmp, &[ekv], "k_conv"), "alloc");
    let v_conv = step!(at(gpu, &mut tmp, &[ekv], "v_conv"), "alloc");
    if swa {
        step!(
            launch_conv(
                gpu,
                &tmp[k_raw],
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
                &tmp[v_raw],
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
                free_all(gpu, tmp);
                return Err("inkling2 gpu fwd: full layer missing k/v proj".to_string());
            }
        };
        let dk = step!(
            at(gpu, &mut tmp, &[nkv * cfg.conv_kernel], "dyn_k"),
            "alloc"
        );
        let dv = step!(
            at(gpu, &mut tmp, &[nkv * cfg.conv_kernel], "dyn_v"),
            "alloc"
        );
        step!(gpu.gemv_f32(kp, &tmp[ki], &tmp[dk]), "k dyn proj");
        step!(gpu.gemv_f32(vp, &tmp[vi], &tmp[dv]), "v dyn proj");
        step!(
            launch_conv(
                gpu,
                &tmp[k_raw],
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
                &tmp[v_raw],
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
    // Clips + V head-norm.
    {
        let qv = view2(&tmp[q], nh, hd);
        step!(
            gpu.inkling2_l2_clip_rows_f32(&qv, Some(&layer.q_clip), nh, hd, cfg.qk_clip),
            "q clip"
        );
        let kv = view2(&tmp[k_conv], nkv, hd);
        step!(
            gpu.inkling2_l2_clip_rows_f32(&kv, None, nkv, hd, cfg.qk_clip),
            "k clip"
        );
        let rv = view2(&tmp[r], nh, cfg.n_rel);
        step!(
            gpu.inkling2_l2_clip_rows_f32(&rv, Some(&layer.rel_clip), nh, cfg.n_rel, cfg.rel_clip),
            "r clip"
        );
    }
    let vc = step!(at(gpu, &mut tmp, &[ekv], "vc"), "alloc");
    {
        let vin = view2(&tmp[v_conv], nkv, hd);
        step!(
            gpu.inkling2_rmsnorm_nogamma_f32(&vin, &tmp[vc], nkv, hd, eps),
            "v norm"
        );
    }
    // Full-layer tau: exactly 1.0 below `log_floor`; fail closed beyond
    // instead of silently dropping it. Generate admission refuses requests
    // that would reach this through `Inkling2Config::gpu_position_limit`.
    if !swa && !cfg.gpu_full_tau_supported(pos) {
        free_all(gpu, tmp);
        return Err(format!(
            "inkling2 gpu fwd: full-layer tau {} at pos {pos} unimplemented",
            cfg.full_attention_tau(pos)
        ));
    }
    // Sinks + relative bias (sink folded in by the gather).
    let sinks = step!(at(gpu, &mut tmp, &[nh], "sinks"), "alloc");
    step!(gpu.gemv_f32(&layer.sink, &tmp[qi], &tmp[sinks]), "sink");
    // KV write + gather map: full visits grow to max_seq and fail closed
    // past it; SWA visits are an evict-on-write ring (slot `pos % rows`)
    // holding the trailing `min(pos+1, rows)` keys with true positions in
    // the per-slot tags. Below the ring size both reduce to the old path
    // (slot == pos, tag == pos) bit-identically.
    let rows = gs.visit_rows[visit];
    if !swa && pos >= rows {
        free_all(gpu, tmp);
        return Err(format!(
            "inkling2 gpu fwd: visit {visit} KV overflow at pos {pos} (rows {rows})"
        ));
    }
    let wpos = if swa { swa_slot(pos, rows) } else { pos };
    gs.visit_pos_tags[visit][wpos] = pos as u32;
    let (slots, dists) = match swa_gather_map(&gs.visit_pos_tags[visit], pos, rows) {
        Ok(m) => m,
        Err(e) => {
            free_all(gpu, tmp);
            return Err(format!("inkling2 gpu fwd: {e}"));
        }
    };
    let n_kv = slots.len();
    let dist_bytes: Vec<u8> = dists.iter().flat_map(|d| d.to_le_bytes()).collect();
    let dist = step!(
        hip(gpu.upload_raw(&dist_bytes, &[dist_bytes.len()]), "dist"),
        "dist"
    );
    tmp.push(dist);
    let di = tmp.len() - 1;
    let slot_bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
    let slot_buf = step!(
        hip(gpu.upload_raw(&slot_bytes, &[slot_bytes.len()]), "slots"),
        "slots"
    );
    tmp.push(slot_buf);
    let si = tmp.len() - 1;
    let rel_bias = step!(at(gpu, &mut tmp, &[nh, n_kv], "rel_bias"), "alloc");
    {
        let rv = view2(&tmp[r], nh, cfg.n_rel);
        step!(
            gpu.inkling2_rel_bias_gather_f32(
                &rv,
                &layer.rel_table,
                &tmp[sinks],
                &tmp[di],
                &tmp[rel_bias],
                cfg.n_rel,
                nh,
                cfg.n_extent,
                n_kv
            ),
            "rel_bias"
        );
    }
    step!(cap_put(cap, gpu, "qc", &tmp[q]), "capture");
    step!(cap_put(cap, gpu, "kc", &tmp[k_conv]), "capture");
    step!(cap_put(cap, gpu, "vc", &tmp[vc]), "capture");
    step!(cap_put(cap, gpu, "rc", &tmp[r]), "capture");
    step!(cap_put(cap, gpu, "sinks", &tmp[sinks]), "capture");
    step!(cap_put(cap, gpu, "rel_bias", &tmp[rel_bias]), "capture");
    // KV write (F32 -> F16 rows), then decode attention over 0..n_kv.
    for h in 0..nkv {
        let src_k = head_view(&tmp[k_conv], h, hd);
        let dst_k = gs.k_cache[visit][h].sub_offset(wpos * hd, hd);
        step!(
            gpu.convert_f32_to_f16_into(&src_k, &dst_k, hd),
            "k cache write"
        );
        let src_v = head_view(&tmp[vc], h, hd);
        let dst_v = gs.v_cache[visit][h].sub_offset(wpos * hd, hd);
        step!(
            gpu.convert_f32_to_f16_into(&src_v, &dst_v, hd),
            "v cache write"
        );
    }
    gs.visit_cap[visit] = pos + 1;
    let attn_heads = step!(at(gpu, &mut tmp, &[eattn], "attn_heads"), "alloc");
    {
        let qv = view2(&tmp[q], nh, hd);
        step!(
            launch_attn(
                gpu,
                &qv,
                &gs.k_cache[visit],
                &gs.v_cache[visit],
                &tmp[rel_bias],
                &tmp[si],
                &tmp[attn_heads],
                n_kv,
                nh,
                hd
            ),
            "attn"
        );
    }
    step!(cap_put(cap, gpu, "attn_heads", &tmp[attn_heads]), "capture");
    // Output gate + projection + post conv.
    let gate = step!(at(gpu, &mut tmp, &[eattn], "gate"), "alloc");
    step!(
        weight_gemv(gpu, &layer.wgate, &tmp[qi], &tmp[gate]),
        "wgate"
    );
    step!(gpu.sigmoid_f32(&tmp[gate]), "gate sigmoid");
    step!(
        gpu.mul_f32(&tmp[attn_heads], &tmp[gate], &tmp[attn_heads]),
        "gate mul"
    );
    let wo_out = step!(at(gpu, &mut tmp, &[e], "wo_out"), "alloc");
    step!(
        weight_gemv(gpu, &layer.wo, &tmp[attn_heads], &tmp[wo_out]),
        "wo"
    );
    step!(cap_put(cap, gpu, "wo_out", &tmp[wo_out]), "capture");
    let attn_res = step!(at(gpu, &mut tmp, &[e], "attn_res"), "alloc");
    step!(
        launch_conv(
            gpu,
            &tmp[wo_out],
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
        push = Some(step!(dup(gpu, x, "block"), "dup"));
    }
    let cur = step!(at(gpu, &mut tmp, &[e], "cur"), "alloc");
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
        free_all(gpu, tmp);
        if let Some(p) = push.take() {
            let _ = gpu.free_tensor(p);
        }
        return Err(format!(
            "inkling2 gpu fwd: mlp_query len {} != {e}",
            layer.mlp_query.buf.size() / 4
        ));
    }
    let block_rows: Vec<&GpuTensor> = blocks.iter().chain(push.iter()).collect();
    let ffn_mixed = step!(
        gpu_mix(gpu, &block_rows, &tmp[cur], &layer.mlp_query, e, 1, eps),
        "ffn mix"
    );
    tmp.push(ffn_mixed);
    let fmi = tmp.len() - 1;
    let ffn_in = match (&msgs, mids) {
        (Some((_, _, _, ctrls)), Some(m)) => {
            let f = step!(
                gpu_control(
                    gpu,
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
    let ffn_normed = step!(at(gpu, &mut tmp, &[e], "ffn_normed"), "alloc");
    step!(
        gpu.rmsnorm_f32(&tmp[ffn_in], &layer.mlp_norm, &tmp[ffn_normed], eps),
        "ffn norm"
    );
    step!(cap_put(cap, gpu, "ffn_mix", &tmp[ffn_in]), "capture");
    step!(cap_put(cap, gpu, "ffn_normed", &tmp[ffn_normed]), "capture");
    // Router: logits, top-8 (SigmoidBiasRenorm eps 1e-20, scale route_scale).
    let router_logits = step!(
        at(gpu, &mut tmp, &[cfg.n_experts], "router_logits"),
        "alloc"
    );
    step!(
        weight_gemv(gpu, &layer.router, &tmp[ffn_normed], &tmp[router_logits]),
        "router"
    );
    step!(
        cap_put(cap, gpu, "router_logits", &tmp[router_logits]),
        "capture"
    );
    let topk_idx = step!(at(gpu, &mut tmp, &[cfg.n_expert_used], "topk_idx"), "alloc");
    let topk_w = step!(at(gpu, &mut tmp, &[cfg.n_expert_used], "topk_w"), "alloc");
    step!(
        gpu.ffn_moe_router_topk_f32(&MoeRouterTopK {
            kind: RouterKind::SigmoidBiasRenorm {
                eps: 1e-20,
                scale: cfg.route_scale
            },
            logits: &tmp[router_logits],
            correction_bias: layer.router_bias.as_ref(),
            topk_indices: &tmp[topk_idx],
            topk_weights: &tmp[topk_w],
            rows: 1,
            n_experts: cfg.n_experts,
            top_k: cfg.n_expert_used,
        }),
        "router topk"
    );
    step!(cap_put(cap, gpu, "topk_idx", &tmp[topk_idx]), "capture");
    step!(cap_put(cap, gpu, "topk_w", &tmp[topk_w]), "capture");
    // Expert pointer tables (per-expert buffers; rebuilt per visit —
    // 2 KiB of H2D, no lifetime questions).
    let gu_bytes: Vec<u8> = layer
        .experts
        .iter()
        .flat_map(|x| (x.gate_up.buf.buf.as_ptr() as u64).to_ne_bytes())
        .collect();
    let dn_bytes: Vec<u8> = layer
        .experts
        .iter()
        .flat_map(|x| (x.down.buf.buf.as_ptr() as u64).to_ne_bytes())
        .collect();
    let ptr_gu = step!(at(gpu, &mut tmp, &[2 * cfg.n_experts], "ptr_gu"), "alloc");
    let ptr_dn = step!(at(gpu, &mut tmp, &[2 * cfg.n_experts], "ptr_dn"), "alloc");
    step!(
        gpu.hip.memcpy_htod(&tmp[ptr_gu].buf, &gu_bytes),
        "gu ptr table"
    );
    step!(
        gpu.hip.memcpy_htod(&tmp[ptr_dn].buf, &dn_bytes),
        "dn ptr table"
    );
    // Rotate, gate_up (batched), SiTU, rotate, down (batched expanded).
    let ffn_x_rot = step!(at(gpu, &mut tmp, &[e], "ffn_x_rot"), "alloc");
    step!(
        rotate_x_mq_for(
            gpu,
            &layer.experts[0].gate_up,
            &tmp[ffn_normed],
            &tmp[ffn_x_rot],
            e
        ),
        "ffn rotate"
    );
    step!(cap_put(cap, gpu, "ffn_x_rot", &tmp[ffn_x_rot]), "capture");
    let inter = cfg.inter;
    let gate_b = step!(
        at(gpu, &mut tmp, &[cfg.n_expert_used, inter], "gate_b"),
        "alloc"
    );
    let up_b = step!(
        at(gpu, &mut tmp, &[cfg.n_expert_used, inter], "up_b"),
        "alloc"
    );
    step!(
        gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
            &tmp[ptr_gu],
            &tmp[topk_idx],
            &tmp[ffn_x_rot],
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
    let act = step!(
        at(gpu, &mut tmp, &[cfg.n_expert_used, 1280], "act"),
        "alloc"
    );
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
    let rot = step!(
        at(gpu, &mut tmp, &[cfg.n_expert_used, 1280], "rot"),
        "alloc"
    );
    step!(
        gpu.rotate_x_mq_batched(&tmp[act], &tmp[rot], 1280, cfg.n_expert_used),
        "act rotate"
    );
    let down_exp = step!(
        at(gpu, &mut tmp, &[cfg.n_expert_used, e], "down_exp"),
        "alloc"
    );
    step!(
        gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
            &tmp[ptr_dn],
            &tmp[topk_idx],
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
    let moe_raw = step!(at(gpu, &mut tmp, &[e], "moe_raw"), "alloc");
    step!(
        gpu.ffn_moe_combine_f32(&MoeCombine {
            order: CombineOrder::Rank,
            round_expert_outputs: false,
            round_routed_sum: false,
            round_final: false,
            down_rows: &tmp[down_exp],
            row_of_slot: None,
            topk_indices: &tmp[topk_idx],
            topk_weights: &tmp[topk_w],
            shared: None,
            routed_out: None,
            out: &tmp[moe_raw],
            rows: 1,
            hidden: e,
            top_k: cfg.n_expert_used,
            n_experts: cfg.n_experts,
        }),
        "combine"
    );
    step!(cap_put(cap, gpu, "moe_raw", &tmp[moe_raw]), "capture");
    let moe_conv = step!(at(gpu, &mut tmp, &[e], "moe_conv"), "alloc");
    step!(
        launch_conv(
            gpu,
            &tmp[moe_raw],
            &layer.mlp_post_conv,
            None,
            &gs.conv[visit],
            &tmp[moe_conv],
            4 * ekv,
            e,
            0,
            hd,
            1.0 / 5.0f32.sqrt()
        ),
        "moe post"
    );
    step!(cap_put(cap, gpu, "moe_conv", &tmp[moe_conv]), "capture");
    // Tail: visit_out escapes; everything else dies here. Only add/capture
    // can fail past this point — free the three survivors explicitly.
    let visit_out = match gpu.alloc_tensor(&[e], DType::F32) {
        Ok(t) => t,
        Err(e2) => {
            if let Some(p) = push.take() {
                let _ = gpu.free_tensor(p);
            }
            free_all(gpu, tmp);
            return Err(format!("inkling2 gpu fwd: alloc visit_out: {e2:?}"));
        }
    };
    if let Err(e2) = gpu.add_f32(&tmp[cur], &tmp[moe_conv], &visit_out) {
        let _ = gpu.free_tensor(visit_out);
        if let Some(p) = push.take() {
            let _ = gpu.free_tensor(p);
        }
        free_all(gpu, tmp);
        return Err(format!("inkling2 gpu fwd: moe residual: {e2:?}"));
    }
    if let Err(e2) = cap_put(cap, gpu, "visit_out", &visit_out) {
        let _ = gpu.free_tensor(visit_out);
        if let Some(p) = push.take() {
            let _ = gpu.free_tensor(p);
        }
        free_all(gpu, tmp);
        return Err(format!("inkling2 gpu fwd: capture: {e2:?}"));
    }
    free_all(gpu, tmp);
    Ok(VisitOut { visit_out, push })
}

// ---------------------------------------------------------------------------
// Token driver (mirrors forward.rs::forward_token)
// ---------------------------------------------------------------------------

/// Token-level owned tensors, freed on every path by [`drive`].
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
        if let Some(t) = self.x {
            let _ = gpu.free_tensor(t);
        }
        for t in self.blocks {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.prelude_x {
            let _ = gpu.free_tensor(t);
        }
        for t in self.prelude_blocks {
            let _ = gpu.free_tensor(t);
        }
        for t in self.prev_blocks {
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
    let x = o.x.take().expect("inkling2 gpu fwd: missing x");
    let msgs = match take_msgs {
        Some((core_i, _)) => {
            let skip = o.prelude_blocks.len() + 1;
            if o.prev_blocks.len() < skip {
                return Err(format!(
                    "inkling2 gpu fwd: pass prev_blocks {} < skip {skip}",
                    o.prev_blocks.len()
                ));
            }
            let ctrls = match &gw.layers[phys].controllers {
                Some(c) => c,
                None => {
                    return Err(format!(
                        "inkling2 gpu fwd: core layer {phys} missing controllers"
                    ))
                }
            };
            Some((
                o.prev_blocks.as_slice(),
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
        &x,
        &o.blocks,
        visit,
        phys,
        pos,
        msgs,
        gs,
        cap,
    )?;
    if let Some(t) = trace.as_mut() {
        t.push(dl(gpu, &vo.visit_out)?);
    }
    if let Some(p) = vo.push {
        o.blocks.push(p);
    }
    hip(gpu.free_tensor(x), "x free")?;
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
    let tok = hip(gpu.alloc_tensor(&[e], DType::F32), "tok")?;
    hip(
        gpu.rmsnorm_f32(&emb, &gw.embed_norm, &tok, cfg.rms_norm_eps),
        "embed norm",
    )?;
    hip(gpu.free_tensor(emb), "embed free")?;
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
    for b in o.blocks.drain(..) {
        let d = dup(gpu, &b, "prelude_block")?;
        o.prelude_blocks.push(d);
        hip(gpu.free_tensor(b), "block free")?;
    }
    // prev_* seeds pass 1's bank (pass 0 itself takes no messages).
    for b in &o.prelude_blocks {
        o.prev_blocks.push(dup(gpu, b, "prev_block")?);
    }
    o.prev_out = Some(dup(gpu, o.prelude_x.as_ref().expect("x"), "prev_out")?);
    // Loop passes.
    for pass in 0..cfg.loop_cfg.num_passes {
        // Reset the stream to the prelude output.
        if let Some(x) = o.x.take() {
            hip(gpu.free_tensor(x), "x free")?;
        }
        for b in o.blocks.drain(..) {
            hip(gpu.free_tensor(b), "block free")?;
        }
        o.x = Some(dup(gpu, o.prelude_x.as_ref().expect("x"), "pass x")?);
        for b in &o.prelude_blocks {
            o.blocks.push(dup(gpu, b, "pass block")?);
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
            for b in o.prev_blocks.drain(..) {
                hip(gpu.free_tensor(b), "prev block free")?;
            }
            for b in &o.blocks {
                o.prev_blocks.push(dup(gpu, b, "prev block")?);
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
    let block_refs: Vec<&GpuTensor> = o.blocks.iter().collect();
    let mixed = match gpu_mix(
        gpu,
        &block_refs,
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
    let _ = gpu.free_tensor(x);
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
}
