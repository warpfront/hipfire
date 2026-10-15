// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MiniMax-M2 forward pass (free functions — hot-path static dispatch).
//!
//! Per-layer pipeline (validated vs the HF `MiniMaxM2` modeling oracle to
//! cosine 0.9996):
//!   h += o_proj · attn( qk_norm(q/k/v_proj(rmsnorm(h))) + partial-RoPE )   [GQA, Q8 KV]
//!   h += combine( experts( sigmoid+bias top-8 route( rmsnorm(h) ) ) )       [MoE]
//! then logits = lm_head( rmsnorm(h) ).
//!
//! Single-lane decode runs each layer as the typed step program in
//! [`crate::program`]: Q8 attention projections, flat QK-norm, prefix
//! half-split RoPE over the first `rotary_dim` channels, the Q8 KV attention,
//! then one sealed MoE call (sigmoid router with the selection bias, top-8
//! FWHT-basis routed experts, route_scale = 1.0: MiniMax-M2 applies no
//! routed-scaling factor). The batched prefill and EP MoE below still issue
//! their kernels by hand.
//!
//! Decode has two entry points: `decode_step` (eager, used for prefill +
//! warmup) and `decode_step_with_graph` (hipGraph capture/replay of the
//! 62-layer body + lm_head, recovering the ~9% per-token launch-latency gap on
//! gfx11/gfx12 — see the gfx1151 perfmaxx characterization). Both share
//! `decode_step_body`; the only per-token-varying GPU input is the device
//! position scalar (`pos_buf`), staged from the heap-stable `state.pos_host`
//! so the captured memcpy re-reads it on each replay. The embedding lookup is
//! kept OUTSIDE the captured region (token_id is baked into its kernarg).

use crate::minimax::{MiniMaxConfig, MiniMaxLayerWeights, MiniMaxState, MiniMaxWeights};
use crate::program::{head_steps, LayerRefs};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::superop::{
    self, ForwardBindings, OpBinding, OpFlavor, SuperOp, SuperOpKind,
};
use hipfire_dispatch::pipeline::{execute_steps, BoundMoeExperts};
use hipfire_dispatch::types::DispatchError;
use hipfire_runtime::llama::{
    fused_silu_mul_rotate_mq_batched_for, rotate_x_mq_batched_for, rotate_x_mq_for, weight_gemv,
};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Decode one token (eager); returns the full logits vector. Used for prefill,
/// the warm pass, and as the `HIPFIRE_MINIMAX_GRAPH=0` fallback.
pub fn decode_step(
    cfg: &MiniMaxConfig,
    weights: &MiniMaxWeights,
    state: &mut MiniMaxState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    gpu.embedding_lookup_q8(&weights.embed, &state.h, token_id, cfg.hidden_size)
        .map_err(|e| format!("minimax: embed lookup: {e:?}"))?;
    decode_step_body(cfg, weights, state, gpu, position, None)?;
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("minimax: download logits: {e:?}"))
}

/// Decode one token, appending each layer's post-residual hidden state
/// (pre final-norm) to `capture[layer]` — used by the oracle dumper. Set
/// `HIPFIRE_MINIMAX_CAPTURE_POSTATTN` to capture the post-attention residual
/// (pre-MoE) instead, for attention-vs-MoE divergence localization. Eager
/// only (the per-layer D2H downloads are incompatible with graph capture).
pub fn decode_step_capture(
    cfg: &MiniMaxConfig,
    weights: &MiniMaxWeights,
    state: &mut MiniMaxState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
    capture: &mut [Vec<f32>],
) -> Result<(), String> {
    gpu.embedding_lookup_q8(&weights.embed, &state.h, token_id, cfg.hidden_size)
        .map_err(|e| format!("minimax: embed lookup: {e:?}"))?;
    decode_step_body(cfg, weights, state, gpu, position, Some(capture))
}

/// Decode one token via hipGraph capture/replay. **Opt-in, default OFF**
/// (`HIPFIRE_MINIMAX_GRAPH=1` to enable). The 62-layer body + lm_head are
/// captured once and replayed per token.
///
/// Output is byte-for-byte identical to eager `decode_step` (validated over 96
/// greedy tokens). But the perf payoff is marginal: on gfx1151 (Strix Halo —
/// the only arch MiniMax's 86 GB footprint fits) it measured **+1.0%**
/// (27.68 → 27.95 tok/s, tight variance), NOT the ~9% the inter-kernel-gap
/// analysis predicted. Root cause: the 9.7% decode launch/idle gap is GPU
/// command-processor inter-kernel dispatch latency, not host-launch overhead —
/// the host thread already runs ahead of the 90%-busy iGPU, so removing the
/// host launch API cost (all hipGraph does) recovers ~nothing. This matches the
/// DeepSeek-V4 "hipGraph dead on gfx1151 decode" finding. Kept as a validated
/// opt-in (may help on a faster CP, e.g. a gfx12 dGPU, if MiniMax ever fits one).
///
/// Capture-safety invariants (mirrors the proven DeepSeek-V4 path):
///   - token_id is per-token → embedding runs OUTSIDE the capture.
///   - position is per-token → staged via `state.pos_host` (stable `Box`); the
///     captured `memcpy_htod_auto` re-reads it on every replay.
///   - attention launch geometry is sized for `state.max_seq` (constant), not
///     the live `seq_len`, so the baked grid/shared-mem stays valid as the KV
///     length grows (the kernel reads the true length from `pos_buf[0]+1`).
pub fn decode_step_with_graph(
    cfg: &MiniMaxConfig,
    weights: &MiniMaxWeights,
    state: &mut MiniMaxState,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    use std::sync::OnceLock;
    static GRAPH_ENV: OnceLock<Option<bool>> = OnceLock::new();
    let env_override = *GRAPH_ENV.get_or_init(|| {
        match hipfire_config::developer_var("HIPFIRE_MINIMAX_GRAPH")
            .ok()
            .as_deref()
        {
            Some("1") => Some(true),
            Some("0") => Some(false),
            _ => None,
        }
    });
    // Default OFF — measured only +1.0% on gfx1151 (the sole arch MiniMax fits);
    // the decode gap is GPU-CP dispatch latency, not host-launch overhead, so
    // hipGraph recovers ~nothing here. Opt in with HIPFIRE_MINIMAX_GRAPH=1.
    let graph_on = env_override.unwrap_or(false);
    if !graph_on {
        return decode_step(cfg, weights, state, gpu, token_id, position);
    }

    // Warmup: first decode after a fresh load runs eager (JITs kernels + settles
    // DPM) and drops any stale graph from a previously-loaded model so the next
    // call captures fresh for THIS model's weight pointers.
    if !state.ar_warmed_up {
        state.ar_warmed_up = true;
        gpu.graphs.graph_exec = None;
        return decode_step(cfg, weights, state, gpu, token_id, position);
    }

    // Capture + replay both need an explicit (non-null) stream.
    if gpu.active_stream.is_none() {
        let s = gpu
            .hip
            .stream_create()
            .map_err(|e| format!("minimax graph: stream_create: {e:?}"))?;
        gpu.active_stream = Some(s);
    }

    // Embedding lookup OUTSIDE the captured region — token_id is baked into the
    // embedding kernarg. Runs on the active stream, ordered before the captured
    // body that reads `state.h`.
    gpu.embedding_lookup_q8(&weights.embed, &state.h, token_id, cfg.hidden_size)
        .map_err(|e| format!("minimax graph: embed lookup: {e:?}"))?;

    if gpu.graphs.graph_exec.is_none() {
        // ── Capture phase ──────────────────────────────────────────────
        // decode_step_body stages pos_host → pos_buf via memcpy_htod_auto
        // INSIDE the capture, so the recorded memcpy node re-reads pos_host
        // on each replay.
        //
        // API drift (integration/dispatch-migration): the hipGraph capture
        // helpers moved into the `gpu.graphs` substruct and now take
        // (&hip, device_id, &stream) — same shape as the LFM2.5-MoE +
        // DeepSeek-V4 graph paths on this branch.
        gpu.graphs
            .begin_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("minimax begin_graph_capture: {e:?}"))?;
        decode_step_body(cfg, weights, state, gpu, position, None)?;
        gpu.graphs
            .end_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("minimax end_graph_capture: {e:?}"))?;
        // Captured kernels were RECORDED, not run — launch once so this token's
        // logits actually get produced.
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("minimax graph_launch (capture): {e:?}"))?;
        eprintln!(
            "[MiniMax hipGraph] captured decode forward — {} kernarg blobs retained",
            gpu.graphs.capture_blobs.len()
        );
    } else {
        // ── Replay phase ───────────────────────────────────────────────
        // Host-only update of the stable position source; the captured memcpy
        // re-reads it and propagates to pos_buf (read by rope / kv-write /
        // attention).
        state.pos_host[0] = position as i32;
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("minimax graph_launch (replay): {e:?}"))?;
    }
    state.n_tokens = position as usize + 1;

    // Logits download is outside the captured region (sync dtoh completes after
    // the captured kernels, which the device observes on the active stream).
    gpu.download_f32(&state.logits)
        .map_err(|e| format!("minimax graph: download logits: {e:?}"))
}

/// The capturable core: stage the device position scalar, run each layer's
/// typed step program (attention, then the sealed MoE), then final-norm +
/// lm_head. Does NOT do the embedding lookup (the caller stages `state.h`).
/// Under graph capture, `capture` is `None` (no D2H); the oracle dumper passes
/// `Some(..)` and runs eager only.
fn decode_step_body(
    cfg: &MiniMaxConfig,
    weights: &MiniMaxWeights,
    state: &mut MiniMaxState,
    gpu: &mut Gpu,
    position: u32,
    mut capture: Option<&mut [Vec<f32>]>,
) -> Result<(), String> {
    let seq_len = position as usize + 1;
    let capture_postattn =
        hipfire_config::developer_var_os("HIPFIRE_MINIMAX_CAPTURE_POSTATTN").is_some();

    // Device position scalar (i32) for rope / kv-write / attention. Staged from
    // the heap-stable `state.pos_host` so the captured memcpy re-reads it on
    // replay (memcpy_htod_auto → async on the capture stream when capturing).
    state.pos_host[0] = position as i32;
    {
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(state.pos_host.as_ptr() as *const u8, 4) };
        gpu.memcpy_htod_auto(&state.pos_buf, pos_bytes)
            .map_err(|e| format!("minimax: htod pos: {e:?}"))?;
    }

    {
        let state = &*state;
        let ctx = DispatchCtx::new(gpu);
        for (l, layer) in weights.layers.iter().enumerate() {
            let refs = LayerRefs::new(layer);
            let attention = refs
                .attention_steps(cfg, state, l, position as usize)
                .map_err(|e| format!("minimax L{l}: {e}"))?;
            execute_steps(gpu, &ctx, &attention)
                .map_err(|e| format!("minimax L{l}: attention: {e:?}"))?;
            if capture_postattn {
                capture_hidden(gpu, state, capture.as_deref_mut(), l)?;
            }
            let (routed, sealed) = layer
                .packed_experts()
                .ok_or_else(|| format!("minimax L{l}: experts are not sealed (EP shard)"))?;
            let bound = BoundMoeExperts::from_cache(&sealed.table, &sealed.cache)
                .map_err(|e| format!("minimax L{l}: bind experts: {e:?}"))?;
            let moe = refs
                .moe_step(cfg, &ctx, state, l, &routed, bound)
                .map_err(|e| format!("minimax L{l}: {e}"))?;
            execute_steps(gpu, &ctx, std::slice::from_ref(&moe))
                .map_err(|e| format!("minimax L{l}: moe: {e:?}"))?;
            if !capture_postattn {
                capture_hidden(gpu, state, capture.as_deref_mut(), l)?;
            }
        }
        let lm_head = weights.lm_head.dispatch_ref();
        execute_steps(gpu, &ctx, &head_steps(cfg, weights, &lm_head, state))
            .map_err(|e| format!("minimax: head: {e:?}"))?;
    }
    state.n_tokens = seq_len;
    Ok(())
}

/// Oracle dumper: append layer `l`'s residual stream to `capture[l]`.
fn capture_hidden(
    gpu: &mut Gpu,
    state: &MiniMaxState,
    capture: Option<&mut [Vec<f32>]>,
    l: usize,
) -> Result<(), String> {
    if let Some(cap) = capture {
        let h = gpu
            .download_f32(&state.h)
            .map_err(|e| format!("minimax L{l}: capture download: {e:?}"))?;
        cap[l].extend_from_slice(&h);
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// EP decode (Ship 6 substrate-EP). Each rank runs the [Attend, Moe] layer
// program through `hipfire_runtime::ep::run_layer_program_ep`: Attend is the
// Single attention step program (replicated); Moe is the EP-only hand block
// below, which accumulates the rank's routed experts into the zeroed partial
// the EP executor all-reduces. Single decode runs the sealed MoE step instead.
// ─────────────────────────────────────────────────────────────────────────

/// EP MoE block (ffn-norm folded in): the routed down/combine accumulates
/// into the zeroed `out_target` partial (the EP driver all-reduces it and adds
/// it into each rank's `state.h`). MiniMax has NO shared expert, so the
/// entire MoE output is routed.
#[allow(clippy::too_many_arguments)]
fn minimax_moe_block_ep(
    gpu: &mut Gpu,
    cfg: &MiniMaxConfig,
    layer: &MiniMaxLayerWeights,
    state: &MiniMaxState,
    l: usize,
    out_target: &GpuTensor,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let n_exp = cfg.num_local_experts;
    let k_top = cfg.num_experts_per_tok;
    let eps = cfg.rms_norm_eps;
    gpu.rmsnorm_f32(&state.h, &layer.ffn_norm, &state.ffn_tmp, eps)
        .map_err(|e| format!("minimax L{l}: ffn rmsnorm: {e:?}"))?;
    rotate_x_mq_for(
        gpu,
        &layer.experts[0].gate_up,
        &state.ffn_tmp,
        &state.ffn_x_rot,
        hidden,
    )
    .map_err(|e| format!("minimax L{l}: ffn rotate: {e:?}"))?;
    weight_gemv(gpu, &layer.router, &state.ffn_tmp, &state.router_logits)
        .map_err(|e| format!("minimax L{l}: router: {e}"))?;
    gpu.sigmoid_f32(&state.router_logits)
        .map_err(|e| format!("minimax L{l}: sigmoid: {e:?}"))?;
    gpu.deepseek4_moe_topk_bias_aware_f32(
        &state.router_logits,
        &layer.routing_bias,
        &state.topk_indices,
        &state.topk_weights,
        n_exp as i32,
        k_top as i32,
        1.0,
    )
    .map_err(|e| format!("minimax L{l}: topk: {e:?}"))?;
    let edt = layer.experts[0].gate_up.gpu_dtype;
    match edt {
        DType::MQ4G256 | DType::HFQ4G256 => gpu
            .gemv_hfq4g256_moe_gate_up_k8_indexed(
                &layer.expert_gate_up_ptrs,
                &state.topk_indices,
                &state.ffn_x_rot,
                &state.gate_batch,
                &state.up_batch,
                2 * inter,
                hidden,
                k_top,
            )
            .map_err(|e| format!("minimax L{l}: gate_up hfq4: {e:?}"))?,
        DType::MQ6G256 | DType::HFQ6G256 => gpu
            .gemv_hfq6g256_moe_gate_up_k8_indexed(
                &layer.expert_gate_up_ptrs,
                &state.topk_indices,
                &state.ffn_x_rot,
                &state.gate_batch,
                &state.up_batch,
                2 * inter,
                hidden,
                k_top,
            )
            .map_err(|e| format!("minimax L{l}: gate_up hfq6: {e:?}"))?,
        DType::MQ2G256Lloyd => gpu
            .deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
                &layer.expert_gate_up_ptrs,
                &state.topk_indices,
                &state.ffn_x_rot,
                &state.gate_batch,
                &state.up_batch,
                2 * inter,
                hidden,
                k_top,
            )
            .map_err(|e| format!("minimax L{l}: gate_up mq2l: {e:?}"))?,
        DType::MQ3G256Lloyd => gpu
            .deepseek4_gemv_mq3g256_lloyd_moe_gate_up_indexed(
                &layer.expert_gate_up_ptrs,
                &state.topk_indices,
                &state.ffn_x_rot,
                &state.gate_batch,
                &state.up_batch,
                2 * inter,
                hidden,
                k_top,
            )
            .map_err(|e| format!("minimax L{l}: gate_up mq3l: {e:?}"))?,
        other => return Err(format!("minimax L{l}: unsupported expert dtype {other:?}")),
    }
    fused_silu_mul_rotate_mq_batched_for(
        gpu,
        &layer.experts[0].down,
        &state.gate_batch,
        &state.up_batch,
        &state.rot_batch,
        inter,
        k_top,
    )
    .map_err(|e| format!("minimax L{l}: silu_mul_rotate: {e:?}"))?;
    let ddt = layer.experts[0].down.gpu_dtype;
    match ddt {
        DType::MQ4G256 | DType::HFQ4G256 => {
            gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                &layer.expert_down_ptrs,
                &state.topk_indices,
                &state.rot_batch,
                &state.down_expanded,
                hidden,
                inter,
                k_top,
                1,
            )
            .map_err(|e| format!("minimax L{l}: down hfq4: {e:?}"))?;
            gpu.moe_down_combine_k8_batched(
                &state.down_expanded,
                &state.topk_weights,
                out_target,
                hidden,
                k_top,
                1,
            )
            .map_err(|e| format!("minimax L{l}: combine: {e:?}"))?;
        }
        DType::MQ6G256 | DType::HFQ6G256 => {
            gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                &layer.expert_down_ptrs,
                &state.topk_indices,
                &state.rot_batch,
                &state.down_expanded,
                hidden,
                inter,
                k_top,
                1,
            )
            .map_err(|e| format!("minimax L{l}: down hfq6: {e:?}"))?;
            gpu.moe_down_combine_k8_batched(
                &state.down_expanded,
                &state.topk_weights,
                out_target,
                hidden,
                k_top,
                1,
            )
            .map_err(|e| format!("minimax L{l}: combine: {e:?}"))?;
        }
        DType::MQ2G256Lloyd => {
            gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
                &layer.expert_down_ptrs,
                &state.topk_indices,
                &state.topk_weights,
                &state.rot_batch,
                out_target,
                hidden,
                inter,
                k_top,
                false,
            )
            .map_err(|e| format!("minimax L{l}: down mq2l: {e:?}"))?;
        }
        DType::MQ3G256Lloyd => {
            gpu.deepseek4_gemv_mq3g256_lloyd_moe_down_residual_scaled_indexed(
                &layer.expert_down_ptrs,
                &state.topk_indices,
                &state.topk_weights,
                &state.rot_batch,
                out_target,
                hidden,
                inter,
                k_top,
            )
            .map_err(|e| format!("minimax L{l}: down mq3l: {e:?}"))?;
        }
        other => return Err(format!("minimax L{l}: unsupported expert dtype {other:?}")),
    }
    Ok(())
}

/// Per-layer EP execution context (rebuilt each layer).
struct MinimaxBindings<'a> {
    cfg: &'a MiniMaxConfig,
    layer: &'a MiniMaxLayerWeights,
    state: &'a MiniMaxState,
    l: usize,
}

impl<'a> ForwardBindings for MinimaxBindings<'a> {
    fn run_attend(
        &mut self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let refs = LayerRefs::new(self.layer);
        let position = self.state.pos_host[0] as usize;
        let steps = refs
            .attention_steps(self.cfg, self.state, self.l, position)
            .map_err(|e| DispatchError::Hip(format!("minimax L{}: {e}", self.l)))?;
        execute_steps(gpu, ctx, &steps)
    }
    fn run_moe(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(
            "minimax EP bindings run the MoE only as an EP partial".into(),
        ))
    }
    fn run_moe_ep(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        routed_out: &GpuTensor,
        _skip_shared: bool,
    ) -> Result<(), DispatchError> {
        // MiniMax has no shared expert → the entire MoE output is routed, so the
        // whole block redirects into `routed_out` (zeroed by the EP executor);
        // `state.h` (the replicated attention residual) is added after all-reduce
        // via ep_add_into_residual. `skip_shared` is irrelevant (no shared expert).
        minimax_moe_block_ep(gpu, self.cfg, self.layer, self.state, self.l, routed_out)
            .map_err(DispatchError::Hip)
    }
    fn ep_add_into_residual(
        &mut self,
        gpu: &mut Gpu,
        partial: &GpuTensor,
    ) -> Result<(), DispatchError> {
        gpu.add_inplace_f32(&self.state.h, partial)
            .map_err(|e| DispatchError::Hip(e.to_string()))
    }
    fn run_proj(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("minimax has no Proj super-op".into()))
    }
    fn run_residual_gemv(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(
            "minimax has no ResidualGemv super-op".into(),
        ))
    }
    fn run_norm(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("minimax has no Norm super-op".into()))
    }
    fn run_conv(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("minimax has no Conv super-op".into()))
    }
    fn run_recurrent(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(
            "minimax has no Recurrent super-op".into(),
        ))
    }
    fn run_escape(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        kind: superop::EscapeKind,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(format!(
            "minimax has no Escape super-op ({kind:?})"
        )))
    }
}

#[inline]
fn mm_superop(kind: SuperOpKind) -> SuperOp {
    SuperOp {
        kind,
        binding: OpBinding {
            key: None,
            weights: Vec::new(),
            scratch: Vec::new(),
            flavor: OpFlavor::None,
        },
    }
}

/// MiniMax has ONE layer shape (all layers Attn+MoE) → the same 2-op EP
/// program for every layer.
fn minimax_lower_program() -> superop::LayerProgram {
    vec![
        mm_superop(SuperOpKind::Attend),
        mm_superop(SuperOpKind::Moe),
    ]
}

/// True iff every layer's expert gate_up + down dtypes have batched kernels, so
/// `forward_batch` won't `Err` partway through a pass. Pre-check this before
/// enabling batched prefill: unsupported tiers (MQ3-Lloyd, HFQ6-gate_up) then
/// cleanly take the sequential path instead of corrupting state on a mid-layer
/// `Err`. Mirrors the dtype match arms in `forward_batch`.
pub fn forward_batch_supported(weights: &MiniMaxWeights) -> bool {
    weights.layers.iter().all(|layer| {
        let gate_up_ok = matches!(
            layer.experts[0].gate_up.gpu_dtype,
            DType::MQ4G256 | DType::HFQ4G256 | DType::MQ2G256Lloyd
        );
        let down_ok = matches!(
            layer.experts[0].down.gpu_dtype,
            DType::MQ4G256
                | DType::HFQ4G256
                | DType::MQ6G256
                | DType::HFQ6G256
                | DType::MQ2G256Lloyd
                | DType::MQ3G256Lloyd
        );
        gate_up_ok && down_ok
    })
}

fn grouped_moe_dtypes_supported(gate_up: DType, down: DType) -> bool {
    matches!(gate_up, DType::MQ2G256Lloyd)
        && matches!(down, DType::MQ2G256Lloyd | DType::MQ3G256Lloyd)
}

fn grouped_moe_topology_supported(n_experts: usize, experts_per_token: usize) -> bool {
    n_experts == 256 && experts_per_token == 8
}

fn large_batch_topology_supported(cfg: &MiniMaxConfig) -> bool {
    cfg.hidden_size == 3072
        && cfg.intermediate_size == 1536
        && cfg.num_attention_heads == 48
        && cfg.num_key_value_heads == 8
        && cfg.head_dim == 128
        && cfg.rotary_dim == 64
        && grouped_moe_topology_supported(cfg.num_local_experts, cfg.num_experts_per_tok)
}

/// Batched forward over `B` tokens in ONE pass — the spec-decode VERIFY forward
/// and fast-prefill keystone. Fills the KV cache for all B positions and returns
/// the LAST token's logits. Reads each weight matrix ONCE for all B tokens
/// (bandwidth-amortized — verifying B tokens costs ~1× the 6.2 GB/token weight
/// read, not B×), which is the basis of the 2-5× spec-decode / fast-TTFT win.
///
/// `tokens`: B token ids. `start_pos`: absolute position of `tokens[0]` (the KV
/// cache must already hold positions `[0, start_pos)`). `B` must be 1..=64
/// (`gemm_q8_0_batched` kernel cap); the caller chunks longer prompts.
///
/// Batched twin of `decode_step_body`: every op uses its batched kernel variant
/// (audited present in rdna-compute), dense Q8 projections go through
/// `gemm_q8_0_batched` directly (the `weight_gemm` helper falls back to per-row
/// GEMV for Q8). Per-row causal masking + the growing KV length are handled
/// inside `attention_q8_0_kv_batched` via the `positions[B]` array.
///
/// Supported expert dtypes (batched kernels that exist today): gate_up ∈
/// {HFQ4/MQ4, MQ2-Lloyd}; down ∈ {HFQ4/MQ4, HFQ6, MQ2-Lloyd}. HFQ6-gate_up and
/// MQ3-Lloyd have no batched kernel yet → Err (caller falls back to sequential).
#[allow(clippy::too_many_arguments)]
pub fn forward_batch(
    cfg: &MiniMaxConfig,
    weights: &MiniMaxWeights,
    state: &mut MiniMaxState,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: usize,
) -> Result<Vec<f32>, String> {
    let b = tokens.len();
    if b == 0 {
        return Err("minimax forward_batch: empty token slice".to_string());
    }
    // The attention/MoE batched kernels take B as a grid dimension and scale
    // freely; the dense projections go through `gemm_q8_0_batched_chunked`
    // (which internally tiles to the GEMM kernel's MAX_BATCH=64). So large
    // prefill chunks are supported — the only ceiling is the grid-Z limit
    // (65535) and the non-flash attention LDS ctx bound (~12k). Bigger chunks
    // amortize the 79 GB expert-weight read across more tokens (the dominant
    // prefill cost at 256 experts / top-8).
    if b > 4096 {
        return Err(format!(
            "minimax forward_batch: B={b} exceeds supported prefill chunk 4096"
        ));
    }
    if b > 64 && !large_batch_topology_supported(cfg) {
        return Err(format!(
            "minimax forward_batch: B={b} requires the validated MiniMax-M2 production topology"
        ));
    }
    let hidden = cfg.hidden_size;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let inter = cfg.intermediate_size;
    let n_exp = cfg.num_local_experts;
    let k_top = cfg.num_experts_per_tok;
    let eps = cfg.rms_norm_eps;
    let max_ctx = start_pos + b; // largest seq_len across the B rows (geometry)
    let max_seq = state.kv.physical_cap; // KV cache stride

    // ── Batched scratch (allocated per call; prefill/verify is not the hot
    //    per-kernel inner loop, and this keeps MiniMaxState unchanged). ──
    let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
        g.alloc_tensor(&[n], DType::F32)
            .map_err(|e| format!("forward_batch alloc {label}: {e:?}"))
    };
    let x = alloc(gpu, b * hidden, "x")?;
    let tmp = alloc(gpu, b * hidden, "tmp")?;
    let fq = alloc(gpu, b * q_dim, "fq")?;
    let fk = alloc(gpu, b * kv_dim, "fk")?;
    let fv = alloc(gpu, b * kv_dim, "fv")?;
    let attn_out = alloc(gpu, b * q_dim, "attn_out")?;
    let o = alloc(gpu, b * hidden, "o")?;
    let ffn_tmp = alloc(gpu, b * hidden, "ffn_tmp")?;
    let ffn_x_rot = alloc(gpu, b * hidden, "ffn_x_rot")?;
    let router_logits = alloc(gpu, b * n_exp, "router_logits")?;
    let topk_idx = alloc(gpu, b * k_top, "topk_idx")?;
    let topk_w = alloc(gpu, b * k_top, "topk_w")?;
    let gate = alloc(gpu, b * k_top * inter, "gate")?;
    let up = alloc(gpu, b * k_top * inter, "up")?;
    let rot = alloc(gpu, b * k_top * inter, "rot")?;
    let down_exp = alloc(gpu, b * k_top * hidden, "down_exp")?;

    // positions [B] i32 (stored in an f32-sized buffer; kernels read it as i32).
    let pos_data: Vec<i32> = (0..b).map(|i| (start_pos + i) as i32).collect();
    let pos_bytes: Vec<u8> = pos_data.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let pos_array = alloc(gpu, b, "pos_array")?;
    gpu.hip
        .memcpy_htod(&pos_array.buf, &pos_bytes)
        .map_err(|e| format!("forward_batch htod pos: {e:?}"))?;

    // Embedding: per-token lookup into x[B, hidden] (token_id is a scalar arg).
    {
        let x_single = alloc(gpu, hidden, "x_single")?;
        for (i, &tok) in tokens.iter().enumerate() {
            gpu.embedding_lookup_q8(&weights.embed, &x_single, tok, hidden)
                .map_err(|e| format!("forward_batch embed lookup: {e:?}"))?;
            gpu.hip
                .memcpy_dtod_at(&x.buf, i * hidden * 4, &x_single.buf, 0, hidden * 4)
                .map_err(|e| format!("forward_batch embed copy: {e:?}"))?;
        }
        gpu.free_tensor(x_single).ok();
    }

    // ── Grouped-MoE prefill decision + scratch ─────────────────────────────
    // The indexed-batched GEMV re-reads each expert weight ONCE PER ROUTED
    // TOKEN — at 256 experts/top-8 that dominates prefill (rocprofv3: 93%).
    // The scatter-grouped path reads each expert weight ONCE PER CHUNK and
    // runs WMMA, but needs enough rows/expert to be worth the BLOCK_M padding.
    // Gate on chunk size: below MOE_GROUPED_GATE rows the 256 experts get too
    // few rows each to fill a BLOCK_M tile, so the per-token indexed path wins;
    // at/above it the grouped path is faster + coherent for the dtype pairs
    // with grouped kernels (validated by coherence-gate-minimax.sh). Require
    // every layer to be eligible before allocating grouped scratch: MiniMax
    // k-maps can use MQ4 for down even when gate_up is MQ2-Lloyd, and that
    // combination must retain the indexed path rather than failing mid-pass.
    const MOE_BLOCK_M: usize = 16;
    const MOE_GROUPED_GATE: usize = 256;
    let moe_grouped = b >= MOE_GROUPED_GATE
        && gpu.arch_caps.has_wmma()
        && grouped_moe_topology_supported(n_exp, k_top)
        && weights.layers.iter().all(|layer| {
            grouped_moe_dtypes_supported(
                layer.experts[0].gate_up.gpu_dtype,
                layer.experts[0].down.gpu_dtype,
            )
        });
    // Round the padded-scatter bound UP to a whole number of BLOCK_M tiles.
    // The grouped kernels' grid is ceil(m_total/16) tiles and each tile reads
    // expert_tile_ids[tile_y]; sizing that buffer at `m_total_max / 16`
    // (integer-div) must therefore not truncate. b*k_top is only 16-aligned
    // when b is even (k_top=8), so an odd-length last prefill chunk left the
    // grid one tile longer than the buffer → OOB read → GPU page fault. Caught
    // by the coherence battery's odd-b prompts; would also hit a production
    // prompt whose final 512-chunk has odd length.
    let m_total_max = (b * k_top + n_exp * MOE_BLOCK_M).next_multiple_of(MOE_BLOCK_M);
    // i32 scratch lives in F32-sized buffers (kernels read raw bytes as i32,
    // same convention as topk_idx above). `None` when the indexed path is used.
    let alloc_opt =
        |g: &mut Gpu, want: bool, n: usize, label: &str| -> Result<Option<GpuTensor>, String> {
            if want {
                Ok(Some(alloc(g, n, label)?))
            } else {
                Ok(None)
            }
        };
    let g_counts = alloc_opt(gpu, moe_grouped, n_exp, "moe_g_counts")?;
    let g_offsets = alloc_opt(gpu, moe_grouped, n_exp + 1, "moe_g_offsets")?;
    let g_sorted = alloc_opt(gpu, moe_grouped, m_total_max, "moe_g_sorted")?;
    let g_tiles = alloc_opt(gpu, moe_grouped, m_total_max / MOE_BLOCK_M, "moe_g_tiles")?;
    let g_inv = alloc_opt(gpu, moe_grouped, b * k_top, "moe_g_inv")?;
    let g_y_gu = alloc_opt(gpu, moe_grouped, m_total_max * 2 * inter, "moe_g_y_gu")?;
    let g_y_dn = alloc_opt(gpu, moe_grouped, m_total_max * hidden, "moe_g_y_dn")?;

    for (l, layer) in weights.layers.iter().enumerate() {
        // ── Attention (batched, per-row causal via positions) ──────────────
        gpu.rmsnorm_batched(&x, &layer.attn_norm, &tmp, b, hidden, eps)
            .map_err(|e| format!("minimax L{l} batch attn rmsnorm: {e:?}"))?;
        gpu.gemm_q8_0_batched_chunked(&layer.wq.buf, &tmp, &fq, q_dim, hidden, b)
            .map_err(|e| format!("minimax L{l} batch q_proj: {e:?}"))?;
        gpu.gemm_q8_0_batched_chunked(&layer.wk.buf, &tmp, &fk, kv_dim, hidden, b)
            .map_err(|e| format!("minimax L{l} batch k_proj: {e:?}"))?;
        gpu.gemm_q8_0_batched_chunked(&layer.wv.buf, &tmp, &fv, kv_dim, hidden, b)
            .map_err(|e| format!("minimax L{l} batch v_proj: {e:?}"))?;
        if cfg.use_qk_norm {
            // Per-row RMSNorm over the full flat q/k vector (MiniMax convention).
            gpu.rmsnorm_batched(&fq, &layer.q_norm, &fq, b, q_dim, eps)
                .map_err(|e| format!("minimax L{l} batch q_norm: {e:?}"))?;
            gpu.rmsnorm_batched(&fk, &layer.k_norm, &fk, b, kv_dim, eps)
                .map_err(|e| format!("minimax L{l} batch k_norm: {e:?}"))?;
        }
        gpu.rope_partial_interleaved_f32_batched(
            &fq,
            &fk,
            &pos_array,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.rotary_dim,
            cfg.rope_theta,
            b,
            // pos_offset (API drift on integration): added to positions[b] for
            // the RoPE angle only. MiniMax prefill does no KV compaction, so 0
            // is the behavior-preserving no-op offset.
            0,
        )
        .map_err(|e| format!("minimax L{l} batch rope: {e:?}"))?;
        gpu.kv_cache_write_q8_0_batched(
            &state.kv.k_gpu[l],
            &fk,
            &pos_array,
            cfg.num_key_value_heads,
            cfg.head_dim,
            b,
        )
        .map_err(|e| format!("minimax L{l} batch kv write k: {e:?}"))?;
        gpu.kv_cache_write_q8_0_batched(
            &state.kv.v_gpu[l],
            &fv,
            &pos_array,
            cfg.num_key_value_heads,
            cfg.head_dim,
            b,
        )
        .map_err(|e| format!("minimax L{l} batch kv write v: {e:?}"))?;
        gpu.attention_q8_0_kv_batched(
            &fq,
            &state.kv.k_gpu[l],
            &state.kv.v_gpu[l],
            &attn_out,
            &pos_array,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            max_seq,
            max_ctx,
            b,
        )
        .map_err(|e| format!("minimax L{l} batch attention: {e:?}"))?;
        gpu.gemm_q8_0_batched_chunked(&layer.wo.buf, &attn_out, &o, hidden, q_dim, b)
            .map_err(|e| format!("minimax L{l} batch o_proj: {e:?}"))?;
        gpu.add_inplace_f32(&x, &o)
            .map_err(|e| format!("minimax L{l} batch o residual: {e:?}"))?;

        // ── MoE (batched; no shared expert) ────────────────────────────────
        gpu.rmsnorm_batched(&x, &layer.ffn_norm, &ffn_tmp, b, hidden, eps)
            .map_err(|e| format!("minimax L{l} batch ffn rmsnorm: {e:?}"))?;
        // AWQ-aware FWHT rotate (gate_up may carry an AWQ activation scale —
        // MQ2-Lloyd+AWQ); the raw rotate_x_mq_batched would drop it.
        rotate_x_mq_batched_for(
            gpu,
            &layer.experts[0].gate_up,
            &ffn_tmp,
            &ffn_x_rot,
            hidden,
            b,
        )
        .map_err(|e| format!("minimax L{l} batch ffn rotate: {e}"))?;
        gpu.gemm_q8_0_batched_chunked(
            &layer.router.buf,
            &ffn_tmp,
            &router_logits,
            n_exp,
            hidden,
            b,
        )
        .map_err(|e| format!("minimax L{l} batch router: {e:?}"))?;
        gpu.sigmoid_f32(&router_logits)
            .map_err(|e| format!("minimax L{l} batch sigmoid: {e:?}"))?;
        gpu.deepseek4_moe_topk_bias_aware_batched_f32(
            &router_logits,
            &layer.routing_bias,
            &topk_idx,
            &topk_w,
            n_exp as i32,
            k_top as i32,
            1.0,
            b as i32,
        )
        .map_err(|e| format!("minimax L{l} batch topk: {e:?}"))?;

        if moe_grouped {
            // ── Scatter-grouped MoE (large chunks): read each expert weight
            //    ONCE per chunk via WMMA grouped GEMM, vs once-per-routed-token
            //    in the indexed path. Mirrors the deepseek4 SGLang-style
            //    pipeline (scatter → grouped gate_up → unscatter → AWQ
            //    silu·mul·rotate → grouped down → weighted combine into x). ──
            let g_counts = g_counts.as_ref().unwrap();
            let g_offsets = g_offsets.as_ref().unwrap();
            let g_sorted = g_sorted.as_ref().unwrap();
            let g_tiles = g_tiles.as_ref().unwrap();
            let g_inv = g_inv.as_ref().unwrap();
            let g_y_gu = g_y_gu.as_ref().unwrap();
            let g_y_dn = g_y_dn.as_ref().unwrap();

            gpu.moe_scatter_fused_k8(
                &topk_idx,
                g_counts,
                g_offsets,
                g_sorted,
                g_tiles,
                g_inv,
                b * k_top,
                n_exp,
                m_total_max,
                MOE_BLOCK_M,
            )
            .map_err(|e| format!("minimax L{l} grouped scatter: {e:?}"))?;

            // Grouped gate_up GEMM (MQ2-Lloyd): gathers ffn_x_rot rows by token
            // (x_row_div=k_top) → y_gate_up_grouped [m_total, 2*inter]. The
            // dispatcher picks i8 MMQ (gfx1151) vs FP16 WMMA by arch.
            gpu.gemm_mq2g256_lloyd_moe_grouped(
                &layer.expert_gate_up_ptrs,
                g_tiles,
                g_sorted,
                &ffn_x_rot,
                g_y_gu,
                2 * inter,
                hidden,
                k_top,
                m_total_max,
                b,
            )
            .map_err(|e| format!("minimax L{l} grouped gate_up: {e:?}"))?;

            // Unscatter grouped → natural [B*k_top, inter] gate/up.
            gpu.moe_gate_up_unscatter_k8(g_y_gu, g_sorted, &gate, &up, inter, k_top, m_total_max)
                .map_err(|e| format!("minimax L{l} grouped unscatter: {e:?}"))?;

            // AWQ-aware silu·mul·rotate (down weight) → rot [B*k_top, inter].
            fused_silu_mul_rotate_mq_batched_for(
                gpu,
                &layer.experts[0].down,
                &gate,
                &up,
                &rot,
                inter,
                b * k_top,
            )
            .map_err(|e| format!("minimax L{l} grouped silu_mul_rotate: {e}"))?;

            // Grouped down GEMM: gathers rot rows (x_row_div=1) → y_down_grouped.
            // The per-dtype dispatcher picks i8 MMQ (gfx1151) vs FP16 WMMA.
            let ddt = layer.experts[0].down.gpu_dtype;
            match ddt {
                DType::MQ3G256Lloyd => gpu
                    .gemm_mq3g256_lloyd_moe_grouped(
                        &layer.expert_down_ptrs,
                        g_tiles,
                        g_sorted,
                        &rot,
                        g_y_dn,
                        hidden,
                        inter,
                        1,
                        m_total_max,
                        b * k_top,
                    )
                    .map_err(|e| format!("minimax L{l} grouped down mq3l: {e:?}"))?,
                DType::MQ2G256Lloyd => gpu
                    .gemm_mq2g256_lloyd_moe_grouped(
                        &layer.expert_down_ptrs,
                        g_tiles,
                        g_sorted,
                        &rot,
                        g_y_dn,
                        hidden,
                        inter,
                        1,
                        m_total_max,
                        b * k_top,
                    )
                    .map_err(|e| format!("minimax L{l} grouped down mq2l: {e:?}"))?,
                other => {
                    return Err(format!(
                        "minimax L{l} grouped down dtype {other:?} unsupported"
                    ));
                }
            }

            // Weighted combine (inverse_perm + topk_w) → x in-place (residual).
            gpu.moe_down_combine_grouped_k8(g_y_dn, g_inv, &topk_w, &x, hidden, k_top, b)
                .map_err(|e| format!("minimax L{l} grouped combine: {e:?}"))?;
            continue;
        }

        let edt = layer.experts[0].gate_up.gpu_dtype;
        match edt {
            DType::MQ4G256 | DType::HFQ4G256 => gpu
                .gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
                    &layer.expert_gate_up_ptrs,
                    &topk_idx,
                    &ffn_x_rot,
                    &gate,
                    &up,
                    2 * inter,
                    hidden,
                    k_top,
                    b,
                )
                .map_err(|e| format!("minimax L{l} batch gate_up hfq4: {e:?}"))?,
            DType::MQ2G256Lloyd => gpu
                .deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4(
                    &layer.expert_gate_up_ptrs,
                    &topk_idx,
                    &ffn_x_rot,
                    &gate,
                    &up,
                    2 * inter,
                    hidden,
                    k_top,
                    b,
                )
                .map_err(|e| format!("minimax L{l} batch gate_up mq2l: {e:?}"))?,
            other => {
                return Err(format!(
                    "minimax L{l} forward_batch: gate_up dtype {other:?} has no batched kernel yet"
                ));
            }
        }

        // AWQ-aware silu·mul·rotate (down weight; b*k_top expert streams).
        fused_silu_mul_rotate_mq_batched_for(
            gpu,
            &layer.experts[0].down,
            &gate,
            &up,
            &rot,
            inter,
            b * k_top,
        )
        .map_err(|e| format!("minimax L{l} batch silu_mul_rotate: {e}"))?;

        let ddt = layer.experts[0].down.gpu_dtype;
        match ddt {
            DType::MQ4G256 | DType::HFQ4G256 => {
                gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                    &layer.expert_down_ptrs,
                    &topk_idx,
                    &rot,
                    &down_exp,
                    hidden,
                    inter,
                    k_top,
                    b,
                )
                .map_err(|e| format!("minimax L{l} batch down hfq4: {e:?}"))?;
                gpu.moe_down_combine_k8_batched(&down_exp, &topk_w, &x, hidden, k_top, b)
                    .map_err(|e| format!("minimax L{l} batch combine: {e:?}"))?;
            }
            DType::MQ6G256 | DType::HFQ6G256 => {
                gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                    &layer.expert_down_ptrs,
                    &topk_idx,
                    &rot,
                    &down_exp,
                    hidden,
                    inter,
                    k_top,
                    b,
                )
                .map_err(|e| format!("minimax L{l} batch down hfq6: {e:?}"))?;
                gpu.moe_down_combine_k8_batched(&down_exp, &topk_w, &x, hidden, k_top, b)
                    .map_err(|e| format!("minimax L{l} batch combine: {e:?}"))?;
            }
            DType::MQ2G256Lloyd => gpu
                .deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched_k4(
                    &layer.expert_down_ptrs,
                    &topk_idx,
                    &topk_w,
                    &rot,
                    &x,
                    hidden,
                    inter,
                    k_top,
                    b,
                )
                .map_err(|e| format!("minimax L{l} batch down mq2l: {e:?}"))?,
            DType::MQ3G256Lloyd => gpu
                .deepseek4_gemv_mq3g256_lloyd_moe_down_residual_scaled_indexed_batched_k4(
                    &layer.expert_down_ptrs,
                    &topk_idx,
                    &topk_w,
                    &rot,
                    &x,
                    hidden,
                    inter,
                    k_top,
                    b,
                )
                .map_err(|e| format!("minimax L{l} batch down mq3l: {e:?}"))?,
            other => {
                return Err(format!(
                    "minimax L{l} forward_batch: down dtype {other:?} has no batched kernel yet"
                ));
            }
        }
    }
    state.n_tokens = start_pos + b;

    // ── Final RMSNorm + lm_head on the LAST row only (verify/prefill need the
    //    last position's logits; per-position logits = a future all-rows head). ──
    let x_last = alloc(gpu, hidden, "x_last")?;
    gpu.hip
        .memcpy_dtod_at(&x_last.buf, 0, &x.buf, (b - 1) * hidden * 4, hidden * 4)
        .map_err(|e| format!("forward_batch last copy: {e:?}"))?;
    gpu.rmsnorm_f32(&x_last, &weights.final_norm, &state.final_norm_buf, eps)
        .map_err(|e| format!("minimax batch final rmsnorm: {e:?}"))?;
    weight_gemv(gpu, &weights.lm_head, &state.final_norm_buf, &state.logits)
        .map_err(|e| format!("minimax batch lm_head: {e}"))?;
    let logits = gpu
        .download_f32(&state.logits)
        .map_err(|e| format!("forward_batch download logits: {e:?}"))?;

    for t in [
        x,
        tmp,
        fq,
        fk,
        fv,
        attn_out,
        o,
        ffn_tmp,
        ffn_x_rot,
        router_logits,
        topk_idx,
        topk_w,
        gate,
        up,
        rot,
        down_exp,
        pos_array,
        x_last,
    ] {
        gpu.free_tensor(t).ok();
    }
    // Grouped-MoE scratch (only allocated when the grouped path ran). GpuTensor
    // has no Drop, so free explicitly.
    for t in [
        g_counts, g_offsets, g_sorted, g_tiles, g_inv, g_y_gu, g_y_dn,
    ]
    .into_iter()
    .flatten()
    {
        gpu.free_tensor(t).ok();
    }
    Ok(logits)
}

// ───────────────────────── Ship 6 substrate-EP (MiniMax) ─────────────────────
//
// Mirror of the qwen35 EP wiring. MiniMax packs all experts into ONE blob per
// projection (too big to load-then-free on a 32 GB card), so sharding is done at
// LOAD time: `MiniMaxWeights::load(.., Some((shard, rank)))` uploads only the
// rank-owned experts (non-owned → zeroed gate_up dummy). MiniMax has NO shared
// expert, so the entire MoE output is routed → the whole MoE block redirects
// into the per-rank partial. Attention (Q8 KV) is replicated; only the MoE
// routed sum crosses ranks (peer-direct all-reduce).

/// EP (Ship 6 substrate-EP) replicated N-rank decode forward for ONE token.
/// Mirror of qwen35::forward_ep: every rank holds full replicated weights /
/// state / KV EXCEPT MoE experts (sharded at load). Embeds + stages pos per
/// rank, runs each layer's 2-op program (Attend replicated, Moe all-reduce-EP'd)
/// via [`hipfire_runtime::ep::run_layer_program_ep`], then final norm + lm_head
/// on rank 0 → `state_per_rank[0].logits`. Every device must have an
/// `active_stream` ([`hipfire_runtime::ep::ensure_rank_streams`]); peer access
/// enabled for the fast peer-direct all-reduce.
#[allow(clippy::too_many_arguments)]
pub fn forward_ep(
    gpus: &mut hipfire_runtime::multi_gpu::Gpus,
    weights_per_rank: &[MiniMaxWeights],
    cfg: &MiniMaxConfig,
    state_per_rank: &mut [MiniMaxState],
    partials: &[GpuTensor],
    token: u32,
    position: u32,
) -> Result<(), String> {
    let n = gpus.devices.len();
    assert_eq!(
        weights_per_rank.len(),
        n,
        "forward_ep: weights_per_rank len"
    );
    assert_eq!(state_per_rank.len(), n, "forward_ep: state_per_rank len");
    assert_eq!(partials.len(), n, "forward_ep: partials len");
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;

    // 1. Embed + stage pos per rank (replicated, deterministic).
    for r in 0..n {
        gpus.devices[r]
            .bind_thread()
            .map_err(|e| format!("forward_ep bind {r}: {e:?}"))?;
        gpus.devices[r]
            .embedding_lookup_q8(
                &weights_per_rank[r].embed,
                &state_per_rank[r].h,
                token,
                hidden,
            )
            .map_err(|e| format!("forward_ep embed {r}: {e:?}"))?;
        state_per_rank[r].pos_host[0] = position as i32;
        let pos_bytes = unsafe {
            std::slice::from_raw_parts(state_per_rank[r].pos_host.as_ptr() as *const u8, 4)
        };
        gpus.devices[r]
            .memcpy_htod_auto(&state_per_rank[r].pos_buf, pos_bytes)
            .map_err(|e| format!("forward_ep pos {r}: {e:?}"))?;
    }

    // 2. Per-layer EP program (Attend replicated; Moe all-reduce-EP'd).
    let timing = hipfire_config::developer_var("HIPFIRE_EP_DECODE_TIMING").is_ok();
    let t_layers = std::time::Instant::now();
    let program = minimax_lower_program();
    let n_layers = weights_per_rank[0].layers.len();
    for l in 0..n_layers {
        let mut binds: Vec<MinimaxBindings> = Vec::with_capacity(n);
        for r in 0..n {
            binds.push(MinimaxBindings {
                cfg,
                layer: &weights_per_rank[r].layers[l],
                state: &state_per_rank[r],
                l,
            });
        }
        hipfire_runtime::ep::run_layer_program_ep(
            gpus,
            binds.as_mut_slice(),
            partials,
            &program,
            hidden,
            None,
        )
        .map_err(|e| format!("forward_ep run_layer_program_ep L{l}: {e}"))?;
    }

    // 3. Final norm + lm_head on rank 0 → state_per_rank[0].logits.
    {
        gpus.devices[0]
            .bind_thread()
            .map_err(|e| format!("forward_ep bind0: {e:?}"))?;
        let w = &weights_per_rank[0];
        let s = &state_per_rank[0];
        let gpu = &mut gpus.devices[0];
        gpu.rmsnorm_f32(&s.h, &w.final_norm, &s.final_norm_buf, eps)
            .map_err(|e| format!("forward_ep final norm: {e:?}"))?;
        weight_gemv(gpu, &w.lm_head, &s.final_norm_buf, &s.logits)
            .map_err(|e| format!("forward_ep lm_head: {e}"))?;
    }

    let layers_ms = t_layers.elapsed().as_secs_f64() * 1000.0;
    // 4. Sync every rank (work ran on active_streams; host logits read races otherwise).
    let t_sync = std::time::Instant::now();
    for r in 0..n {
        gpus.devices[r]
            .bind_thread()
            .map_err(|e| format!("forward_ep sync bind {r}: {e:?}"))?;
        gpus.devices[r]
            .hip
            .device_synchronize()
            .map_err(|e| format!("forward_ep sync {r}: {e:?}"))?;
    }
    if timing {
        // layers_ms = host enqueue + any blocking (RCCL/backpressure); sync_ms =
        // GPU drain remaining at the barrier. host-launch-bound ⇒ layers_ms is
        // the bulk and sync_ms is small; GPU-bound ⇒ sync_ms is the bulk.
        eprintln!(
            "EP-DECODE-TIMING: layers(host)={layers_ms:.2} ms  final-sync(gpu)={:.2} ms",
            t_sync.elapsed().as_secs_f64() * 1000.0,
        );
    }
    for s in state_per_rank.iter_mut() {
        s.n_tokens = position as usize + 1;
    }
    Ok(())
}

#[cfg(test)]
mod ship6_lower_tests {
    use super::*;
    use superop::SuperOpKind::{Attend, Moe};

    // #397 Ship 6 — minimax is one variant (every layer Attn+MoE).
    #[test]
    fn minimax_program_is_attend_then_moe() {
        let kinds: Vec<_> = minimax_lower_program().iter().map(|o| o.kind).collect();
        assert_eq!(kinds, vec![Attend, Moe]);
    }

    #[test]
    fn grouped_moe_accepts_only_implemented_lloyd_pairs() {
        assert!(grouped_moe_dtypes_supported(
            DType::MQ2G256Lloyd,
            DType::MQ2G256Lloyd
        ));
        assert!(grouped_moe_dtypes_supported(
            DType::MQ2G256Lloyd,
            DType::MQ3G256Lloyd
        ));
        assert!(!grouped_moe_dtypes_supported(
            DType::MQ2G256Lloyd,
            DType::MQ4G256
        ));
        assert!(!grouped_moe_dtypes_supported(
            DType::MQ4G256,
            DType::MQ3G256Lloyd
        ));
    }

    #[test]
    fn grouped_moe_accepts_only_minimax_m2_topology() {
        assert!(grouped_moe_topology_supported(256, 8));
        assert!(!grouped_moe_topology_supported(16, 8));
        assert!(!grouped_moe_topology_supported(256, 4));
    }

    #[test]
    fn large_batch_accepts_only_minimax_m2_production_topology() {
        let mut cfg = MiniMaxConfig {
            vocab_size: 200064,
            hidden_size: 3072,
            num_hidden_layers: 62,
            num_attention_heads: 48,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 1536,
            num_local_experts: 256,
            num_experts_per_tok: 8,
            rotary_dim: 64,
            rope_theta: 5_000_000.0,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 204800,
            use_qk_norm: true,
            use_routing_bias: true,
            scoring_func: "sigmoid".to_string(),
            num_mtp_modules: 3,
            reap_keep: None,
        };
        assert!(large_batch_topology_supported(&cfg));
        cfg.num_local_experts = 16;
        assert!(!large_batch_topology_supported(&cfg));
    }
}
