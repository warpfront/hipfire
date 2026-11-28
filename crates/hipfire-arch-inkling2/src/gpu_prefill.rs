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
//! bit for bit, state included. [`PrefillMode::BLOCKED`] changes four named
//! arithmetic sites: the dense linears (weight-only WMMA GEMM, X rounded to
//! F16), the F32 controller / sink / dynamic-tap projections (hi/lo-F16 WMMA
//! GEMMs, `rdna_compute::inkling2_hilo`), the routed experts (grouped
//! weight-only WMMA) and the attention (blocked causal FA2, hi/lo Q/P, no
//! partition merge). The router stays bitwise (the `mb8` chain per tile).
//! A one-row block (decode, [`forward_decode_gpu`]) commits its key/value
//! first and runs the split-KV FA2 decode attention of the scalar visit: the
//! prefill kernel tiles queries only, so one row would be one tile per head.
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
    swa_gather_map, swa_slot, swa_window_len, Inkling2Controller, Inkling2GpuLayer,
    Inkling2GpuState, Inkling2GpuWeights, INKLING2_CONV_WIDTH,
};
use crate::gpu_forward::{ensure_kernels, launch_attn};
use hip_bridge::KernargBlob;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemm::{GemmFamily, GemmParams};
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::moe::{grouped_moe_m_total_bound, MOE_GROUPED_BLOCK_M};
use hipfire_dispatch::pipeline::{
    grouped_qt44_gemm_prepared, grouped_qt44_gemm_tier, run_uniform_moe_down_expanded,
    GroupedQt44Tables, PrefillActTier,
};
use hipfire_dispatch::types::KernelKey;
use hipfire_runtime::llama::{weight_gemm, weight_gemv, weight_gemv_prerotated, WeightTensor};
use rdna_compute::ffn_activation::{
    CombineOrder, FfnActivation, FfnActivationGrouped, FfnActivationGroupedI8, FfnActivationRows,
    FfnAwq, MoeCombine, MoeRouterTopK, RouterKind,
};
use rdna_compute::inkling2_attention::inkling2_attn_fa2_splits;
use rdna_compute::inkling2_attention_prefill::{Inkling2PrefillAttn, INKLING2_ATTN_PREFILL_REL};
use rdna_compute::inkling2_fuse::{FzRow, FzTarget, Inkling2FzMixNormRot};
use rdna_compute::inkling2_fuse_post::{FzKv16, Inkling2FzAttnPost};
use rdna_compute::inkling2_hilo::{Inkling2HiloBlend, Inkling2HiloGemm};
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

const PF_SYMBOLS: [&str; 3] = [
    "ink_pf_gemv_f32_rows",
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

    /// Evidence spelling `dense=<tier>,experts=<tier>` with tiers `f16|fp8|a8|a4`
    /// (developer tools only; the product resolves routes from the Qwen knobs).
    pub fn parse(s: &str) -> Result<Self, String> {
        let tier = |v: &str| match v {
            "f16" => Ok(PrefillActTier::F16),
            "fp8" => Ok(PrefillActTier::Fp8),
            "a8" => Ok(PrefillActTier::A8),
            "a4" => Ok(PrefillActTier::A4),
            other => Err(format!(
                "prefill route tier {other}: expected f16|fp8|a8|a4"
            )),
        };
        let mut r = Self::F16;
        for kv in s.split(',').filter(|kv| !kv.is_empty()) {
            match kv.split_once('=') {
                Some(("dense", v)) => r.dense = tier(v)?,
                Some(("experts", v)) => r.experts = tier(v)?,
                _ => return Err(format!("prefill routes: bad entry {kv}")),
            }
        }
        Ok(r)
    }
}

/// Routes the blocked prefill runs when no Qwen activation knob is set
/// ([`crate::gpu::resolve_prefill_routes`]).
pub const PREFILL_ROUTES_DEFAULT: PrefillRoutes = PrefillRoutes {
    dense: PrefillActTier::A8,
    experts: PrefillActTier::A8,
};

/// Whether [`forward_prefill_gpu`] runs the blocked path for this load: gfx1201,
/// typed-XTS layout (symmetric qt44 + AWQ sidecars, F16 router), Mini geometry.
pub fn prefill_supported(bundle: &Inkling2Bundle, gpu: &Gpu) -> bool {
    let cfg = &bundle.config;
    let Some(gw) = bundle.gpu_weights.as_ref() else {
        return false;
    };
    gpu.arch_caps.is_gfx1201()
        && bundle.gpu_state.is_some()
        && gw.xts_routes.as_ref().is_some_and(|c| c.prefill.is_some())
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

/// One decode step: `token_id` at `pos` (the session cursor), host logits of
/// the unpadded vocab. Runs the blocked body at one row when
/// [`prefill_supported`] (the load's prefill routes: dense GEMV at M=1, routed
/// experts as grouped GEMMs, blocked FA2 attention), otherwise
/// [`crate::forward_token_gpu`]. Row-exact or replay-identity callers use
/// [`crate::forward_token_gpu`] directly.
pub fn forward_decode_gpu(
    bundle: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    token_id: u32,
    pos: usize,
) -> Result<Vec<f32>, String> {
    let mut logits = Vec::new();
    forward_prefill_gpu(
        bundle,
        gpu,
        &[token_id],
        pos,
        PrefillLogits::Last(&mut logits),
    )?;
    Ok(logits)
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
    if (mode.gemm || mode.grouped || mode.blocked_attn) && !gpu.arch_caps.is_gfx1201() {
        return Err(format!(
            "inkling2 prefill: blocked sites are gfx1201 only, not {}",
            gpu.arch
        ));
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
    // Ordinary prefill intent: the quantized dense arms admit every block row
    // count (a 1..63-row tail keeps its route instead of falling to F16).
    let ordinary = gpu.ordinary_prefill_enter();
    let r = prefill_blocks(
        gpu,
        &cfg,
        gw,
        gs,
        ids,
        start_pos,
        &mut logits,
        mode,
        routes,
        block_rows,
        trace,
    );
    gpu.ordinary_prefill_exit(ordinary);
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
        // A one-row blocked call (decode) attends through the decode kernels.
        let row_maps = if take == 1 && mode.blocked_attn {
            Some(RowMaps::upload(gpu, gs, p)?)
        } else {
            None
        };
        let ctx = BlockCtx {
            cfg,
            gw,
            mode,
            routes,
            rows: take,
            p,
            row_maps: row_maps.as_ref(),
        };
        let mut o = Owned::default();
        let r = drive_block(gpu, &ctx, gs, block, &mut o, want, trace.as_deref_mut());
        o.free(gpu);
        if let Some(m) = row_maps {
            m.free(gpu);
        }
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
    fn alloc(
        &mut self,
        gpu: &mut Gpu,
        shape: &[usize],
        dtype: DType,
        what: &str,
    ) -> Result<T, String> {
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
        self.ts[i.0]
            .as_ref()
            .expect("inkling2 prefill: arena slot taken")
    }
    fn take(&mut self, i: T) -> GpuTensor {
        self.ts[i.0]
            .take()
            .expect("inkling2 prefill: arena slot taken")
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

/// `y [rows, m] = x [rows, k] . A^T` for every `(A, x, y)`: one hi/lo-F16
/// WMMA GEMM launch over all of them (blocked), or the singleton `gemv_f32`
/// arithmetic per row (row-exact).
fn proj_rows(
    gpu: &mut Gpu,
    ps: &[(&GpuTensor, &GpuTensor, &GpuTensor)],
    m: usize,
    k: usize,
    rows: usize,
    gemm: bool,
) -> Result<(), String> {
    if gemm {
        let probs: Vec<Inkling2HiloGemm<'_>> = ps
            .iter()
            .map(|&(w, x, y)| Inkling2HiloGemm { w, x, y })
            .collect();
        return hip(gpu.inkling2_gemm_hilo_f32(&probs, m, k, rows), "hilo gemm");
    }
    for &(a, x, y) in ps {
        gemv_rows(gpu, a, x, k, y, m, m, k, rows)?;
    }
    Ok(())
}

/// One controller input of [`control_rows`]: `(cur, msg, controller)`, rows `[rows, e]`.
type CtrlIn<'a> = (&'a GpuTensor, &'a GpuTensor, &'a Inkling2Controller);

/// Fused controller steps over `rows` rows, one `out [rows, e]` per input.
/// `gates[i]` is input i's gate input `[rows, 2e]`: the gamma-less RMSNorms
/// of cur | msg (`INKLING2_CONTROL_EPS`), written by the mix launch that
/// produces cur or msg ([`Gpu::inkling2_fz_mix_norm_rot`]).
/// Row-exact: the singleton `inkling2_control_f32_gpu` arithmetic per row.
/// Blocked (`gemm`): the down projections of every input in one hi/lo-F16
/// WMMA GEMM launch, then their gain/forget projections and blend in another.
#[allow(clippy::too_many_arguments)]
fn control_rows(
    gpu: &mut Gpu,
    ar: &mut Arena,
    ins: &[CtrlIn<'_>],
    gates: &[T],
    e: usize,
    rank: usize,
    rows: usize,
    gemm: bool,
) -> Result<Vec<T>, String> {
    if gates.len() != ins.len() {
        return Err("inkling2 prefill: one controller gate per input".into());
    }
    let mut gs = Vec::with_capacity(ins.len());
    let mut outs = Vec::with_capacity(ins.len());
    for _ in ins {
        gs.push(ar.f32(gpu, rows * rank, "ctrl_g")?);
        outs.push(ar.f32(gpu, rows * e, "ctrl_out")?);
    }
    let downs: Vec<_> = ins
        .iter()
        .zip(gates.iter().zip(&gs))
        .map(|(&(_, _, c), (&x, &y))| (&c.down, ar.get(x), ar.get(y)))
        .collect();
    proj_rows(gpu, &downs, rank, 2 * e, rows, gemm)?;
    if gemm {
        let blends: Vec<_> = ins
            .iter()
            .zip(gs.iter().zip(&outs))
            .map(|(&(cur, msg, c), (&g, &out))| Inkling2HiloBlend {
                gain_w: &c.gain,
                forget_w: &c.forget,
                g: ar.get(g),
                cur,
                msg,
                out: ar.get(out),
            })
            .collect();
        hip(
            gpu.inkling2_control_up_blend_hilo(&blends, e, rank, rows),
            "control up+blend",
        )?;
        return Ok(outs);
    }
    for (&(cur, msg, c), (&g, &out)) in ins.iter().zip(gs.iter().zip(&outs)) {
        let gain = ar.f32(gpu, rows * e, "ctrl_gain")?;
        let forget = ar.f32(gpu, rows * e, "ctrl_forget")?;
        gemv_rows(
            gpu,
            &c.gain,
            ar.get(g),
            rank,
            ar.get(gain),
            e,
            e,
            rank,
            rows,
        )?;
        gemv_rows(
            gpu,
            &c.forget,
            ar.get(g),
            rank,
            ar.get(forget),
            e,
            e,
            rank,
            rows,
        )?;
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
    }
    Ok(outs)
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

/// [`linear`] of a typed-XTS (MQ4G256V2 + AWQ) weight over rows whose AWQ
/// divide + FWHT256 producer already ran (`x_rot [rows, K]`): the GEMM /
/// GEMV `weight_gemm` / `weight_gemv` run after their own producer.
fn linear_rot(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x_rot: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    gemm: bool,
) -> Result<(), String> {
    if w.gpu_dtype != DType::MQ4G256V2 || w.awq_scale.is_none() {
        return Err(format!(
            "inkling2 prefill: pre-rotated linear needs MQ4G256V2 + AWQ, got {:?}",
            w.gpu_dtype
        ));
    }
    if gemm && rows > 1 {
        hip(gpu.ensure_mq_signs(), "mq signs")?;
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
        return GEMM_FAMILY
            .run_key(
                KernelKey::GemmMq4G256V2BatchedLmhead,
                &ctx,
                gpu,
                &GemmParams {
                    w: &wr,
                    x: x_rot,
                    y,
                    batch_size: rows,
                },
            )
            .map_err(|e| format!("inkling2 prefill: pre-rotated gemm: {e}"));
    }
    for r in 0..rows {
        let xr = view(x_rot, r * w.k, &[w.k])?;
        let yr = view(y, r * w.m, &[w.m])?;
        hip(weight_gemv_prerotated(gpu, w, &xr, Some(&xr), &yr), "gemv")?;
    }
    Ok(())
}

/// [`Gpu::inkling2_fz_mix_norm_rot`] over `n` rows of width `e`.
fn fz(
    gpu: &mut Gpu,
    states: &[&GpuTensor],
    query: Option<&GpuTensor>,
    out: &[FzRow<'_>],
    n: usize,
    e: usize,
    mix_eps: f32,
) -> Result<(), String> {
    hip(
        gpu.inkling2_fz_mix_norm_rot(&Inkling2FzMixNormRot {
            states,
            query,
            out,
            rows: n,
            n: e,
            mix_eps,
        }),
        "mix/norm/producer",
    )
}

/// One attention input (q, k or v): its source row among the launch's rows,
/// norm gamma, normed output (if read later) and producers.
struct Role<'a> {
    src: usize,
    gamma: &'a GpuTensor,
    out: Option<&'a GpuTensor>,
    rots: Vec<(&'a GpuTensor, &'a GpuTensor)>,
}

/// Gamma-less norm target written at row stride `ld` (controller gate halves).
fn nogamma_target(out: &GpuTensor, eps: f32, ld: usize) -> FzTarget<'_> {
    FzTarget {
        gamma: None,
        eps,
        out: Some((out, ld)),
        rots: Vec::new(),
    }
}

/// Norm targets of `roles` over `n_src` launch rows: roles on the same row
/// with the same gamma share one norm (and so one normed output).
fn role_rows(n_src: usize, roles: Vec<Role<'_>>, e: usize, eps: f32) -> Vec<FzRow<'_>> {
    let mut rows: Vec<FzRow<'_>> = (0..n_src).map(|_| FzRow::default()).collect();
    for r in roles {
        let row = &mut rows[r.src];
        let gp = r.gamma.buf.as_ptr();
        let hit = row.targets.iter_mut().find(|t| {
            t.gamma.map(|g| g.buf.as_ptr()) == Some(gp) && (t.out.is_none() || r.out.is_none())
        });
        match hit {
            Some(t) => {
                if t.out.is_none() {
                    t.out = r.out.map(|o| (o, e));
                }
                t.rots.extend(r.rots);
            }
            None => row.targets.push(FzTarget {
                gamma: Some(r.gamma),
                eps,
                out: r.out.map(|o| (o, e)),
                rots: r.rots,
            }),
        }
    }
    rows
}

/// Attention-input buffers of one visit: normed inputs and producer outputs.
struct AttnIn<'a> {
    swa: bool,
    /// q, k, v input-norm gammas.
    gammas: [&'a GpuTensor; 3],
    qi: T,
    ki: T,
    vi: T,
    rot_q: T,
    rot_r: T,
    rot_g: T,
    rot_k: T,
    rot_v: T,
}

impl<'a> AttnIn<'a> {
    /// The q, k, v roles with source rows `src`.
    fn roles<'b>(
        &self,
        ar: &'b Arena,
        layer: &'b Inkling2GpuLayer,
        src: [usize; 3],
    ) -> Result<Vec<Role<'b>>, String>
    where
        'a: 'b,
    {
        let awq = |w: &'b WeightTensor, what: &str| {
            w.awq_scale
                .as_ref()
                .ok_or_else(|| format!("inkling2 prefill: {what} without its AWQ sidecar"))
        };
        let mut rots_q = vec![(awq(&layer.wq, "wq")?, ar.get(self.rot_q))];
        if !layer.qrg_awq_shared {
            rots_q.push((awq(&layer.wr, "wr")?, ar.get(self.rot_r)));
            rots_q.push((awq(&layer.wgate, "wgate")?, ar.get(self.rot_g)));
        }
        Ok(vec![
            Role {
                src: src[0],
                gamma: self.gammas[0],
                out: Some(ar.get(self.qi)),
                rots: rots_q,
            },
            Role {
                src: src[1],
                gamma: self.gammas[1],
                out: (!self.swa).then(|| ar.get(self.ki)),
                rots: vec![(awq(&layer.wk, "wk")?, ar.get(self.rot_k))],
            },
            Role {
                src: src[2],
                gamma: self.gammas[2],
                out: (!self.swa).then(|| ar.get(self.vi)),
                rots: vec![(awq(&layer.wv, "wv")?, ar.get(self.rot_v))],
            },
        ])
    }
}

/// F16 router `[n_experts, e]` over `rows` rows: the gfx12 F16 WMMA (`mb8`)
/// chain the singleton router runs, one column per row (bitwise per row), at
/// one wave per 16 x 16 tile (`GemmF16WmmaColsGfx12`: 256 waves at 512 rows).
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
            KernelKey::GemmF16WmmaColsGfx12,
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
    /// One-row call: the decode gather maps its attention runs over.
    row_maps: Option<&'a RowMaps>,
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

    // --- attention inputs: AttnRes mix (visit 0 / phys 0 takes x itself),
    // loop controllers (pass > 0), input norms and the dense AWQ+FWHT
    // producers. Each norm/producer stage is one `inkling2_fz_mix_norm_rot`
    // launch, bitwise the unfused mix / norm / `rotate_x_mq_awq` chain.
    let gemm = c.mode.gemm;
    let ctrl_eps = rdna_compute::inkling2_control::INKLING2_CONTROL_EPS;
    // Norms: SWA applies attn_norm to all three inputs; full layers the three
    // input norms. Identical (input, gamma) pairs are normalized once.
    let (wq_n, wk_n, wv_n) = if swa {
        let an = layer
            .attn_norm
            .as_ref()
            .ok_or("inkling2 prefill: SWA layer missing attn_norm")?;
        (an, an, an)
    } else {
        match (
            &layer.q_input_norm,
            &layer.k_input_norm,
            &layer.v_input_norm,
        ) {
            (Some(a), Some(b), Some(cc)) => (a, b, cc),
            _ => return Err("inkling2 prefill: full layer missing input norms".into()),
        }
    };
    // Producer outputs: Q, shared with Rel and gate when their AWQ sidecars
    // match (`qrg_awq_shared`), K and V. Normed inputs: Q's feeds the sinks;
    // full layers' dynamic K/V taps read the normed K/V inputs.
    let shared = layer.qrg_awq_shared;
    let rot_q = ar.f32(gpu, ne, "rot_q")?;
    let qi = ar.f32(gpu, ne, "q_in")?;
    let ai = AttnIn {
        swa,
        gammas: [wq_n, wk_n, wv_n],
        qi,
        ki: if swa { qi } else { ar.f32(gpu, ne, "k_in")? },
        vi: if swa { qi } else { ar.f32(gpu, ne, "v_in")? },
        rot_q,
        rot_r: if shared {
            rot_q
        } else {
            ar.f32(gpu, ne, "rot_r")?
        },
        rot_g: if shared {
            rot_q
        } else {
            ar.f32(gpu, ne, "rot_gate")?
        },
        rot_k: ar.f32(gpu, ne, "rot_k")?,
        rot_v: ar.f32(gpu, ne, "rot_v")?,
    };
    let (ki, vi, rot_r, rot_g, rot_k, rot_v) =
        (ai.ki, ai.vi, ai.rot_r, ai.rot_g, ai.rot_k, ai.rot_v);
    // Pass > 0: controller gate inputs `[n, 2e]` (gamma-less normed cur | msg)
    // of the q, k, v and FFN controllers, written by the mixes below.
    let gates: Vec<T> = if msgs.is_some() {
        (0..4)
            .map(|_| ar.f32(gpu, n * 2 * e, "ctrl_gate"))
            .collect::<Result<_, _>>()?
    } else {
        Vec::new()
    };
    let (msg_buf, attn_mixed) = match (&msgs, phys) {
        (Some(_), 0) => return Err("inkling2 prefill: phys 0 has no base".into()),
        (Some(_), _) => (
            Some(ar.f32(gpu, 4 * ne, "msg_mix")?),
            Some(ar.f32(gpu, 3 * ne, "attn_mix")?),
        ),
        (None, _) => (None, None),
    };
    let attn_query = if phys == 0 {
        None
    } else {
        let aq = layer.attn_query.as_ref().ok_or_else(|| {
            format!("inkling2 prefill: visit {visit} phys {phys} missing attn_query")
        })?;
        let total = aq.buf.size() / 4;
        if total % e != 0 {
            return Err(format!(
                "inkling2 prefill: attn_query len {total} not a multiple of {e}"
            ));
        }
        Some((aq, if total / e >= 3 { 3 } else { 1 }))
    };
    let states: Vec<&GpuTensor> = blocks.iter().chain(std::iter::once(x)).collect();
    match (&msgs, attn_query) {
        (None, None) => {
            let rows = role_rows(1, ai.roles(ar, layer, [0, 0, 0])?, e, eps);
            fz(gpu, &[x], None, &rows, n, e, eps)?
        }
        (None, Some((aq, qr))) => {
            let src = if qr >= 3 { [0, 1, 2] } else { [0, 0, 0] };
            let rows = role_rows(qr, ai.roles(ar, layer, src)?, e, eps);
            fz(gpu, &states, Some(aq), &rows, n, e, eps)?
        }
        (Some((bank, prev_out, cross, ctrls)), Some((aq, qr))) => {
            if cross.buf.size() / 4 < 4 * e {
                return Err(format!(
                    "inkling2 prefill: cross len {} < 4*{e}",
                    cross.buf.size() / 4
                ));
            }
            let (msg_buf, attn_mixed) = (msg_buf.unwrap(), attn_mixed.unwrap());
            let gate_half =
                |j: usize, half: usize| view(ar.get(gates[j]), half * e, &[n * 2 * e - e]);
            // Attention mix: raw bases (the blends' `cur`) + gamma-less norms
            // into the controllers' cur halves.
            let cur_halves = (0..3)
                .map(|j| gate_half(j, 0))
                .collect::<Result<Vec<_>, _>>()?;
            let bases = (0..qr)
                .map(|j| view(ar.get(attn_mixed), j * ne, &[n, e]))
                .collect::<Result<Vec<_>, _>>()?;
            let rows_a: Vec<FzRow<'_>> = (0..qr)
                .map(|j| FzRow {
                    raw: Some((&bases[j], e)),
                    targets: if qr >= 3 {
                        vec![nogamma_target(&cur_halves[j], ctrl_eps, 2 * e)]
                    } else {
                        cur_halves
                            .iter()
                            .map(|h| nogamma_target(h, ctrl_eps, 2 * e))
                            .collect()
                    },
                })
                .collect();
            fz(gpu, &states, Some(aq), &rows_a, n, e, eps)?;
            // Message mix: raw messages + norms into the msg halves (FFN's included).
            let msg_halves = (0..4)
                .map(|j| gate_half(j, 1))
                .collect::<Result<Vec<_>, _>>()?;
            let msgs_v = (0..4)
                .map(|j| view(ar.get(msg_buf), j * ne, &[n, e]))
                .collect::<Result<Vec<_>, _>>()?;
            let rows_m: Vec<FzRow<'_>> = (0..4)
                .map(|j| FzRow {
                    raw: Some((&msgs_v[j], e)),
                    targets: vec![nogamma_target(&msg_halves[j], ctrl_eps, 2 * e)],
                })
                .collect();
            let mstates: Vec<&GpuTensor> = bank.iter().chain(std::iter::once(*prev_out)).collect();
            fz(gpu, &mstates, Some(cross), &rows_m, n, e, eps)?;
            let rank = cfg.loop_cfg.gate_rank;
            let ins: Vec<CtrlIn<'_>> = (0..3)
                .map(|j| (&bases[if qr >= 3 { j } else { 0 }], &msgs_v[j], &ctrls[j]))
                .collect();
            let outs = control_rows(gpu, ar, &ins, &gates[..3], e, rank, n, gemm)?;
            let outs_v = outs
                .iter()
                .map(|&o| view(ar.get(o), 0, &[n, e]))
                .collect::<Result<Vec<_>, _>>()?;
            let srcs: Vec<&GpuTensor> = outs_v.iter().collect();
            let rows = role_rows(3, ai.roles(ar, layer, [0, 1, 2])?, e, eps);
            fz(gpu, &srcs, None, &rows, n, e, eps)?;
        }
        (Some(_), None) => return Err("inkling2 prefill: phys 0 has no base".into()),
    }
    // Projections.
    let q = ar.f32(gpu, n * eattn, "q")?;
    let k_raw = ar.f32(gpu, n * ekv, "k_raw")?;
    let v_raw = ar.f32(gpu, n * ekv, "v_raw")?;
    let r = ar.f32(gpu, n * erel, "r")?;
    linear_rot(gpu, &layer.wq, ar.get(rot_q), ar.get(q), n, gemm)?;
    linear_rot(gpu, &layer.wk, ar.get(rot_k), ar.get(k_raw), n, gemm)?;
    linear_rot(gpu, &layer.wv, ar.get(rot_v), ar.get(v_raw), n, gemm)?;
    linear_rot(gpu, &layer.wr, ar.get(rot_r), ar.get(r), n, gemm)?;
    // Dynamic conv taps (full layers).
    let dyn_kv = if swa {
        None
    } else {
        let (kp, vp) = match (&layer.k_proj, &layer.v_proj) {
            (Some(a), Some(b)) => (a, b),
            _ => return Err("inkling2 prefill: full layer missing k/v proj".into()),
        };
        let dw = nkv * cfg.conv_kernel;
        let dk = ar.f32(gpu, n * dw, "dyn_k")?;
        let dv = ar.f32(gpu, n * dw, "dyn_v")?;
        proj_rows(
            gpu,
            &[(kp, ar.get(ki), ar.get(dk)), (vp, ar.get(vi), ar.get(dv))],
            dw,
            e,
            n,
            gemm,
        )?;
        Some((dk, dv, dw))
    };
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
    proj_rows(
        gpu,
        &[(&layer.sink, ar.get(qi), ar.get(sinks))],
        nh,
        e,
        n,
        gemm,
    )?;
    let ring = gs.visit_rows[visit];
    if !swa && p + n > ring {
        return Err(format!(
            "inkling2 prefill: visit {visit} KV overflow at pos {} (rows {ring})",
            p + n - 1
        ));
    }
    // K/V convs (static on SWA; plus the dynamic taps on full layers), K clip,
    // V head-norm, the K/V stores and the Q / Rel clips: one launch. Where the
    // K/V land: the cache slot (one-row block, committed before its decode
    // attention), F16 staging planes (blocked attention, committed after it)
    // or F32 rows (row-exact attention writes them per row).
    let one_row = c.mode.blocked_attn && c.row_maps.is_some();
    if c.mode.blocked_attn {
        check_pre_block_tags(gs, visit, swa, p)?;
    }
    let plane = nkv * n * hd;
    let staged = if c.mode.blocked_attn && !one_row {
        Some((
            ar.alloc(gpu, &[plane], DType::F16, "k_stage")?,
            ar.alloc(gpu, &[plane], DType::F16, "v_stage")?,
        ))
    } else {
        None
    };
    let kv_f32 = if c.mode.blocked_attn {
        None
    } else {
        Some((ar.f32(gpu, n * ekv, "k_conv")?, ar.f32(gpu, n * ekv, "vc")?))
    };
    let kv16 = if one_row {
        let slot = swa_slot(p, ring);
        let at = |t: &GpuTensor| t.buf.as_ptr().wrapping_byte_add(slot * hd * 2) as *const c_void;
        Some(FzKv16 {
            k: std::array::from_fn(|h| gs.k_cache[visit].get(h).map_or(std::ptr::null(), at)),
            v: std::array::from_fn(|h| gs.v_cache[visit].get(h).map_or(std::ptr::null(), at)),
            row_ld: 0,
        })
    } else {
        staged.map(|(ks, vs)| {
            let at = |t: T, h: usize| {
                ar.get(t).buf.as_ptr().wrapping_byte_add(h * n * hd * 2) as *const c_void
            };
            FzKv16 {
                k: std::array::from_fn(|h| if h < nkv { at(ks, h) } else { std::ptr::null() }),
                v: std::array::from_fn(|h| if h < nkv { at(vs, h) } else { std::ptr::null() }),
                row_ld: hd,
            }
        })
    };
    if one_row
        && (0..nkv).any(|h| {
            [&gs.k_cache[visit], &gs.v_cache[visit]]
                .iter()
                .any(|c| c.get(h).is_none_or(|t| t.buf.size() < ring * hd * 2))
        })
    {
        return Err(format!("inkling2 prefill: visit {visit} KV cache planes"));
    }
    hip(
        gpu.inkling2_fz_attn_post(&Inkling2FzAttnPost {
            q: ar.get(q),
            q_scale: &layer.q_clip,
            qk_clip: cfg.qk_clip,
            rel: ar.get(r),
            rel_scale: &layer.rel_clip,
            rel_clip: cfg.rel_clip,
            k_raw: ar.get(k_raw),
            v_raw: ar.get(v_raw),
            k_sw: &layer.k_conv,
            v_sw: &layer.v_conv,
            dyn_kv: dyn_kv.map(|(dk, dv, dw)| (ar.get(dk), ar.get(dv), dw)),
            conv_scale: if swa { 1.0 / 5.0f32.sqrt() } else { 1.0 / 3.0 },
            state: &gs.conv[visit],
            state_stride: INKLING2_CONV_WIDTH,
            k_off: 0,
            v_off: ekv,
            state_update: n == 1,
            v_eps: eps,
            k_f32: kv_f32.map(|(k, _)| ar.get(k)),
            v_f32: kv_f32.map(|(_, v)| ar.get(v)),
            kv16,
            rows: n,
            nh,
            nkv,
        }),
        "attention epilogue",
    )?;
    if one_row {
        kv_tags(gs, visit, p, 1);
    }
    // Attention.
    let attn_heads = ar.f32(gpu, n * eattn, "attn_heads")?;
    match (c.row_maps, staged, kv_f32) {
        (Some(maps), _, _) if one_row => attn_one_row(
            gpu, cfg, layer, gs, ar, maps, visit, q, r, sinks, attn_heads,
        )?,
        (_, Some(st), _) => attn_blocked(
            gpu, cfg, layer, gs, ar, visit, swa, p, n, q, r, sinks, st, attn_heads,
        )?,
        (_, _, Some((k_conv, vc))) => attn_row_exact(
            gpu, cfg, layer, gs, ar, visit, swa, p, n, q, r, sinks, k_conv, vc, attn_heads,
        )?,
        _ => return Err("inkling2 prefill: no attention route".into()),
    }
    // Output gate (sigmoid · mul folded into the `wo` producer), projection,
    // post conv + residual.
    let gate = ar.f32(gpu, n * eattn, "gate")?;
    linear_rot(gpu, &layer.wgate, ar.get(rot_g), ar.get(gate), n, gemm)?;
    let rot_o = ar.f32(gpu, n * eattn, "rot_o")?;
    let wo_awq = layer
        .wo
        .awq_scale
        .as_ref()
        .ok_or("inkling2 prefill: wo without its AWQ sidecar")?;
    hip(
        gpu.inkling2_fz_gate_rot(
            ar.get(attn_heads),
            ar.get(gate),
            wo_awq,
            ar.get(rot_o),
            eattn,
            n,
        ),
        "gate producer",
    )?;
    let wo_out = ar.f32(gpu, ne, "wo_out")?;
    linear_rot(gpu, &layer.wo, ar.get(rot_o), ar.get(wo_out), n, gemm)?;
    // AttnRes bookkeeping: first visit of a block pushes x and takes the
    // conv output as cur; the others add it to x.
    let has_push = phys % cfg.attn_res_block == 0;
    let push = if has_push {
        let t = dup(gpu, x, ne, "block push")?;
        Some(ar.push(t))
    } else {
        None
    };
    let cur = ar.f32(gpu, ne, "cur")?;
    hip(
        gpu.inkling2_fz_conv_res(
            ar.get(wo_out),
            &layer.attn_post_conv,
            &gs.conv[visit],
            INKLING2_CONV_WIDTH,
            2 * ekv,
            e,
            1.0 / 5.0f32.sqrt(),
            if has_push { None } else { Some(x) },
            ar.get(cur),
            n,
            n == 1,
        ),
        "attn post conv",
    )?;
    // --- MoE: mix over blocks + push + cur, control (pass > 0), norm ---
    if layer.mlp_query.buf.size() / 4 != e {
        return Err(format!(
            "inkling2 prefill: mlp_query len {} != {e}",
            layer.mlp_query.buf.size() / 4
        ));
    }
    // Mix, then the FFN norm (no controller) or the gamma-less controller
    // input norm into the FFN controller's cur half (pass > 0), one launch.
    let ffn_normed = ar.f32(gpu, ne, "ffn_normed")?;
    let ffn_mixed = match msgs {
        Some(_) => Some(ar.f32(gpu, ne, "ffn_mix")?),
        None => None,
    };
    {
        let mut states: Vec<&GpuTensor> = blocks.iter().collect();
        if let Some(pi) = push {
            states.push(ar.get(pi));
        }
        states.push(ar.get(cur));
        let cur_half = match &ffn_mixed {
            Some(_) => Some(view(ar.get(gates[3]), 0, &[n * 2 * e - e])?),
            None => None,
        };
        let row = match (ffn_mixed, &cur_half) {
            (Some(fm), Some(h)) => FzRow {
                raw: Some((ar.get(fm), e)),
                targets: vec![FzTarget {
                    gamma: None,
                    eps: ctrl_eps,
                    out: Some((h, 2 * e)),
                    rots: Vec::new(),
                }],
            },
            _ => FzRow {
                raw: None,
                targets: vec![FzTarget {
                    gamma: Some(&layer.mlp_norm),
                    eps,
                    out: Some((ar.get(ffn_normed), e)),
                    rots: Vec::new(),
                }],
            },
        };
        fz(gpu, &states, Some(&layer.mlp_query), &[row], n, e, eps)?;
    }
    if let (Some((_, _, _, ctrls)), Some(m), Some(fm)) = (&msgs, msg_buf, ffn_mixed) {
        let m3 = view(ar.get(m), 3 * ne, &[n, e])?;
        let fmv = view(ar.get(fm), 0, &[n, e])?;
        let ins: [CtrlIn<'_>; 1] = [(&fmv, &m3, &ctrls[3])];
        let ffn_in = control_rows(
            gpu,
            ar,
            &ins,
            &gates[3..],
            e,
            cfg.loop_cfg.gate_rank,
            n,
            gemm,
        )?[0];
        let inp = view(ar.get(ffn_in), 0, &[n, e])?;
        let row = FzRow {
            raw: None,
            targets: vec![FzTarget {
                gamma: Some(&layer.mlp_norm),
                eps,
                out: Some((ar.get(ffn_normed), e)),
                rots: Vec::new(),
            }],
        };
        fz(gpu, &[&inp], None, &[row], n, e, eps)?;
    }
    // Router + top-8.
    let n_exp = cfg.n_experts;
    let k_top = cfg.n_expert_used;
    let router_logits = ar.f32(gpu, n * n_exp, "router_logits")?;
    router_rows(
        gpu,
        &layer.router,
        &view(ar.get(ffn_normed), 0, &[n, e])?,
        ar.get(router_logits),
        n,
        gemm,
    )?;
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
    let visit_out = ar.f32(gpu, ne, "visit_out")?;
    hip(
        gpu.inkling2_fz_conv_res(
            ar.get(moe_raw),
            &layer.mlp_post_conv,
            &gs.conv[visit],
            INKLING2_CONV_WIDTH,
            2 * ekv + e,
            e,
            1.0 / 5.0f32.sqrt(),
            Some(ar.get(cur)),
            ar.get(visit_out),
            n,
            n == 1,
        ),
        "moe post conv",
    )?;
    // A multi-row visit advances its four conv histories once every conv has
    // read the old ones (a one-row visit shifted them in its conv launches).
    if n > 1 {
        hip(
            gpu.inkling2_fz_conv_state(
                &[
                    (ar.get(k_raw), 0, ekv),
                    (ar.get(v_raw), ekv, ekv),
                    (ar.get(wo_out), 2 * ekv, e),
                    (ar.get(moe_raw), 2 * ekv + e, e),
                ],
                &gs.conv[visit],
                INKLING2_CONV_WIDTH,
                n,
            ),
            "conv history",
        )?;
    }
    Ok((visit_out, push))
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
            hip(
                gpu.convert_f32_to_f16_into(&src_k, &dst_k, hd),
                "k cache write",
            )?;
            let src_v = view(ar.get(vc), i * ekv + h * hd, &[hd])?;
            let dst_v = gs.v_cache[visit][h].sub_offset(wpos * hd, hd);
            hip(
                gpu.convert_f32_to_f16_into(&src_v, &dst_v, hd),
                "v cache write",
            )?;
        }
        gs.visit_cap[visit] = pos + 1;
        let qv = view(ar.get(q), i * eattn, &[nh, hd])?;
        let out = view(ar.get(attn_heads), i * eattn, &[eattn])?;
        if gs.attn_fa2 {
            let splits = inkling2_attn_fa2_splits(n_kv);
            // The state's persistent partition scratch, as the decode visit uses it.
            hip(
                gpu.inkling2_attn_decode_fa2(
                    &qv,
                    &gs.k_cache[visit],
                    &gs.v_cache[visit],
                    ar.get(rel_bias),
                    ar.get(si),
                    &gs.fa2_lse,
                    &gs.fa2_oacc,
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

/// Fail closed unless every pre-block key the first query of a block at `p`
/// reads carries its true position tag.
fn check_pre_block_tags(
    gs: &Inkling2GpuState,
    visit: usize,
    swa: bool,
    p: usize,
) -> Result<(), String> {
    let ring = gs.visit_rows[visit];
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
    Ok(())
}

/// Commit the staged block to the cache ring and its position tags: only the
/// trailing `ring` rows survive a block wider than the ring.
#[allow(clippy::too_many_arguments)]
fn kv_commit(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    gs: &mut Inkling2GpuState,
    ar: &Arena,
    visit: usize,
    p: usize,
    n: usize,
    (ks, vs): (T, T),
) -> Result<(), String> {
    let nkv = cfg.n_head_kv;
    let ring = gs.visit_rows[visit];
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
        [
            grid_u32(
                ((n - first) * nkv * cfg.head_dim).div_ceil(256),
                "kv commit",
            )?,
            1,
            1,
        ],
        [256, 1, 1],
        0,
        b,
    )?;
    kv_tags(gs, visit, p, n);
    Ok(())
}

/// Position tags and cursor of `n` rows from `p` stored in the cache ring of
/// `visit` (only the trailing `ring` rows survive a block wider than the ring).
fn kv_tags(gs: &mut Inkling2GpuState, visit: usize, p: usize, n: usize) {
    let ring = gs.visit_rows[visit];
    for i in n.saturating_sub(ring)..n {
        let pos = p + i;
        gs.visit_pos_tags[visit][swa_slot(pos, ring)] = pos as u32;
    }
    gs.visit_cap[visit] = p + n;
}

/// Blocked attention over the block's F16-staged K/V (`[nkv, n, hd]`): the
/// causal FA2 prefill over cache (pre-block) + staging (in-block) keys, then
/// the block's commit to the cache ring and its position tags. The caller
/// checked the pre-block tags before staging.
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
    staged: (T, T),
    attn_heads: T,
) -> Result<(), String> {
    let ring = gs.visit_rows[visit];
    let window = if swa { ring } else { 0 };
    hip(
        gpu.inkling2_attn_prefill_fa2(&Inkling2PrefillAttn {
            q: ar.get(q),
            rel: ar.get(r),
            sinks: ar.get(sinks),
            table: &layer.rel_table,
            k_cache: &gs.k_cache[visit],
            v_cache: &gs.v_cache[visit],
            k_stage: ar.get(staged.0),
            v_stage: ar.get(staged.1),
            out: ar.get(attn_heads),
            rows: n,
            pos0: p,
            window,
            ring_rows: ring,
            n_extent: cfg.n_extent,
        }),
        "attn prefill",
    )?;
    kv_commit(gpu, cfg, gs, ar, visit, p, n, staged)
}

/// Decode gather maps of a one-row call at `pos`, one per distinct ring size
/// (SWA ring, full-layer rows): key slots oldest to newest, then relative
/// distances, the [`swa_gather_map`] of a ring whose tags hold their true
/// positions (each visit checks its tags before use). One pooled buffer and one
/// upload per call, before the call's first launch.
struct RowMaps {
    buf: GpuTensor,
    /// `(ring rows, n_kv, slots byte offset, dists byte offset)`.
    rings: Vec<(usize, usize, usize, usize)>,
}

impl RowMaps {
    fn upload(gpu: &mut Gpu, gs: &Inkling2GpuState, pos: usize) -> Result<Self, String> {
        let mut rings: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut bytes: Vec<u8> = Vec::new();
        for &ring in &gs.visit_rows {
            if rings.iter().any(|r| r.0 == ring) {
                continue;
            }
            let n_kv = swa_window_len(pos, ring);
            let start = pos + 1 - n_kv;
            let slots_off = bytes.len();
            for j in start..=pos {
                bytes.extend_from_slice(&(swa_slot(j, ring) as i32).to_le_bytes());
            }
            let dists_off = bytes.len();
            for j in start..=pos {
                bytes.extend_from_slice(&((pos - j) as i32).to_le_bytes());
            }
            rings.push((ring, n_kv, slots_off, dists_off));
        }
        let buf = hip(gpu.alloc_tensor(&[bytes.len()], DType::Raw), "row maps")?;
        if let Err(e) = gpu.memcpy_htod_auto(&buf.buf, &bytes) {
            let _ = gpu.free_tensor(buf);
            return Err(format!("inkling2 prefill: row maps upload: {e:?}"));
        }
        Ok(Self { buf, rings })
    }

    /// `(n_kv, slots, dists)` of the ring of `ring` rows.
    fn get(&self, ring: usize) -> Result<(usize, GpuTensor, GpuTensor), String> {
        let &(_, n_kv, s, d) = self
            .rings
            .iter()
            .find(|r| r.0 == ring)
            .ok_or_else(|| format!("inkling2 prefill: no row map for ring {ring}"))?;
        let part = |off: usize| GpuTensor {
            buf: self.buf.buf.byte_view(off, n_kv * 4),
            shape: vec![n_kv * 4],
            dtype: DType::Raw,
        };
        Ok((n_kv, part(s), part(d)))
    }

    fn free(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.buf);
    }
}

/// Attention of a one-row block (decode), whose K/V the visit's attention
/// epilogue already stored in its cache slot: the split-KV FA2 decode
/// attention of the scalar decode visit over the call's [`RowMaps`]. The
/// blocked FA2 prefill kernel tiles queries only, so one row would be one tile
/// per head over every key (8.2 ms per token at 2K context against 1.2 ms for
/// the decode kernels, `decode1-runs/runs/j1`).
#[allow(clippy::too_many_arguments)]
fn attn_one_row(
    gpu: &mut Gpu,
    cfg: &Inkling2Config,
    layer: &Inkling2GpuLayer,
    gs: &mut Inkling2GpuState,
    ar: &mut Arena,
    maps: &RowMaps,
    visit: usize,
    q: T,
    r: T,
    sinks: T,
    attn_heads: T,
) -> Result<(), String> {
    let nh = cfg.n_head;
    let hd = cfg.head_dim;
    let (n_kv, slots, dists) = maps.get(gs.visit_rows[visit])?;
    let rel_bias = ar.alloc(gpu, &[nh, n_kv], DType::F32, "rel_bias")?;
    hip(
        gpu.inkling2_rel_bias_gather_f32(
            &view(ar.get(r), 0, &[nh, cfg.n_rel])?,
            &layer.rel_table,
            &view(ar.get(sinks), 0, &[nh])?,
            &dists,
            ar.get(rel_bias),
            cfg.n_rel,
            nh,
            cfg.n_extent,
            n_kv,
        ),
        "rel_bias",
    )?;
    let qv = view(ar.get(q), 0, &[nh, hd])?;
    let out = view(ar.get(attn_heads), 0, &[nh * hd])?;
    if gs.attn_fa2 {
        hip(
            gpu.inkling2_attn_decode_fa2(
                &qv,
                &gs.k_cache[visit],
                &gs.v_cache[visit],
                ar.get(rel_bias),
                &slots,
                &gs.fa2_lse,
                &gs.fa2_oacc,
                &out,
                n_kv,
                inkling2_attn_fa2_splits(n_kv),
            ),
            "attn fa2",
        )
    } else {
        launch_attn(
            gpu,
            &qv,
            &gs.k_cache[visit],
            &gs.v_cache[visit],
            ar.get(rel_bias),
            &slots,
            &out,
            n_kv,
            nh,
            hd,
        )
    }
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
    let s1 = gpu
        .scratch
        .mq_signs1
        .as_ref()
        .ok_or("inkling2 prefill: no mq signs")?
        .buf
        .as_ptr();
    let s2 = gpu
        .scratch
        .mq_signs2
        .as_ref()
        .ok_or("inkling2 prefill: no mq signs")?
        .buf
        .as_ptr();
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
    gpu.scratch
        .invalidate_x_caches_for(ar.get(x_rot).buf.as_ptr());
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
            hip(
                gpu.rotate_x_mq_batched(ar.get(act), ar.get(rot), phys, slots),
                "act rotate",
            )?;
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
            let y_dn = ar.f32(gpu, m_total * e, "moe_y_dn")?;
            let fused =
                tier == PrefillActTier::A8 && gpu.flags.gfx12_producer_quant_fused_enabled();
            let planes = if fused {
                None
            } else {
                let act = ar.alloc(gpu, &[slots, phys], DType::F32, "moe_act")?;
                let rot = ar.alloc(gpu, &[slots, phys], DType::F32, "moe_rot")?;
                Some((act, rot))
            };
            let dn_tables = GroupedQt44Tables {
                expert_weight_ptrs: &xl.down_ptrs,
                expert_tile_ids: ar.get(tiles),
                expert_offsets: ar.get(offsets),
                sorted_slot_index: ar.get(sorted),
            };
            match planes {
                None => {
                    // SiTU + down AWQ + pad + FWHT + block_i8_128 in one launch (bitwise the
                    // three-stage chain + `ensure_int8_mmq_x`), straight into the A8 sidecar.
                    let reservation = hip(gpu.reserve_int8_mmq(phys, slots), "int8 reserve")?;
                    let prepared = hip(
                        gpu.ffn_activation_grouped_rotate_i8_f32(
                            &FfnActivationGroupedI8 {
                                activation: situ,
                                y_grouped: ar.get(y_gu),
                                sorted_slot_index: ar.get(sorted),
                                expert_tile_ids: ar.get(tiles),
                                x_out: None,
                                m_total,
                                block_m: MOE_GROUPED_BLOCK_M,
                                logical_n: inter,
                                phys_n: phys,
                                total_slots: slots,
                                round_inputs: false,
                                round_output: false,
                                awq_table: Some(&xl.down_awq_ptrs),
                                n_experts: n_exp,
                            },
                            reservation,
                        ),
                        "grouped situ i8",
                    )?;
                    let xq = hip(
                        gpu.int8_mmq_prepared_ptr(&prepared, phys, slots),
                        "int8 prepared",
                    )?;
                    grouped_qt44_gemm_prepared(
                        gpu,
                        tier,
                        &dn_tables,
                        xq,
                        ar.get(y_dn),
                        e,
                        phys,
                        1,
                        m_total,
                        slots,
                    )
                    .map_err(|e2| format!("inkling2 prefill: grouped down: {e2}"))?;
                }
                Some((act, rot)) => {
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
                    // rot allocated above
                    hip(
                        gpu.rotate_x_mq_batched(ar.get(act), ar.get(rot), phys, slots),
                        "act rotate",
                    )?;
                    gpu.scratch
                        .invalidate_x_caches_for(ar.get(rot).buf.as_ptr());
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
                }
            }
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
                return Err(format!(
                    "inkling2 prefill: core layer {phys} missing controllers"
                ));
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
    let vo = match run_visit_rows(
        gpu,
        c,
        &gw.layers[phys],
        &x,
        &o.blocks,
        visit,
        phys,
        msgs,
        gs,
    ) {
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
        let row = hip(
            gpu.download_f32(&view(&vo.visit_out, 0, &[c.rows * cfg.n_embd])?),
            "trace",
        )?;
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
        return Err(format!(
            "inkling2 prefill: {} layers != n_phys {}",
            gw.layers.len(),
            cfg.n_phys
        ));
    }
    let core_len = cfg.loop_cfg.core_end - cfg.loop_cfg.core_start;
    if gw.cross_pass.len() != core_len {
        return Err(format!(
            "inkling2 prefill: {} cross_pass != core_len {core_len}",
            gw.cross_pass.len()
        ));
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
        return Err(format!(
            "inkling2 prefill: visit count {visit} != {}",
            cfg.loop_cfg.n_slot()
        ));
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
        return Err(format!(
            "inkling2 prefill: output_query len {} != {e}",
            gw.output_query.buf.size() / 4
        ));
    }
    let x = o.x.as_ref().ok_or("inkling2 prefill: missing x")?;
    let mut ar = Arena::default();
    let r = (|| -> Result<Vec<(usize, Vec<f32>)>, String> {
        let mut states = Vec::with_capacity(o.blocks.len() + 1);
        for b in o.blocks.iter().chain(std::iter::once(x)) {
            states.push(view(b, r0 * e, &[rows, e])?);
        }
        let refs: Vec<&GpuTensor> = states.iter().collect();
        let hidden = ar.f32(gpu, rows * e, "hidden")?;
        let row = FzRow {
            raw: None,
            targets: vec![FzTarget {
                gamma: Some(&gw.norm),
                eps: cfg.rms_norm_eps,
                out: Some((ar.get(hidden), e)),
                rots: Vec::new(),
            }],
        };
        fz(
            gpu,
            &refs,
            Some(&gw.output_query),
            &[row],
            rows,
            e,
            cfg.rms_norm_eps,
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
            hip(
                gpu.logit_softcap_f32(ar.get(logits), take * vocab, cfg.softcap),
                "softcap",
            )?;
            if take == 1 {
                // One row (decode, `Last`): download the unpadded row itself.
                let row = hip(
                    gpu.download_f32(&view(ar.get(logits), 0, &[cfg.unpadded_vocab_size])?),
                    "logits download",
                )?;
                out.push((c.p + r0 + done, row));
            } else {
                let host = hip(
                    gpu.download_f32(&view(ar.get(logits), 0, &[take * vocab])?),
                    "logits download",
                )?;
                for i in 0..take {
                    let row = &host[i * vocab..i * vocab + cfg.unpadded_vocab_size];
                    out.push((c.p + r0 + done + i, row.to_vec()));
                }
            }
            done += take;
        }
        Ok(out)
    })();
    ar.free(gpu);
    r
}
