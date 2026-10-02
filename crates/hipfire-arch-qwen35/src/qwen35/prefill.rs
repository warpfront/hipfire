// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 batched prefill: eligibility gates, dispatch-routed GEMM helpers,
//! `forward_prefill_batch*`, and the per-layer batched chunk bodies.

use super::batch::for_each_active_span;
use super::batch::partial_lane_mask;
use super::batch::valid_lane_mask;
use super::batch::BatchSemantics;
use super::batch::PrefillBatchScratch;
use super::config::DflashFusionCtx;
use super::config::LayerType;
use super::config::MaskEmbedOverride;
use super::config::Qwen35Config;
use super::config::TreeVerifyCtx;
use super::forward::checked_kv_end;
use super::forward::forward_scratch;
use super::forward::forward_scratch_with_hidden;
use super::forward::kv_cache_attention_dispatch;
use super::forward::moe_ffn_has_mq3_experts_uniform;
use super::forward::moe_ffn_has_mq3_structural;
use super::forward::moe_ffn_has_unsupported_mq3_experts_uniform;
use super::forward::Qwen35Scratch;
use super::weights::DeltaNetLayerWeights;
use super::weights::DeltaNetMoeLayerWeights;
use super::weights::DeltaNetState;
use super::weights::FullAttnLayerWeights;
use super::weights::FullAttnMoeLayerWeights;
use super::weights::LayerWeights;
use super::weights::MoeFfnWeights;
use super::weights::Qwen35Weights;
use super::weights::StateQuant;
use crate::speculative::HiddenStateRingBuffer;
use hip_bridge::HipError;
use hip_bridge::HipResult;
use hipfire_dispatch::pipeline::sealed_moe::PrefillRouteMode;

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::context::DispatchWorkload;
use hipfire_dispatch::families::moe::codebook_batched_admit_enabled;
#[cfg(test)]
use hipfire_dispatch::families::moe::{
    codebook_batched_admit_enabled_from_env, mq6_batched_admit_enabled_from_env,
};
pub(crate) use hipfire_dispatch::pipeline::batched::q8_prefill_wmma_enabled;
#[cfg(test)]
use hipfire_dispatch::pipeline::batched::q8_prefill_wmma_enabled_from_env;
pub(crate) use hipfire_dispatch::pipeline::batched::BatchEpilogue;
pub(crate) use hipfire_dispatch::pipeline::batched::{
    run_fused_gate_up_key, run_plain_gemm_key, run_residual_gemm_key,
};
pub(crate) use hipfire_dispatch::pipeline::batched_attention::run_fused_qkv_key;
use hipfire_dispatch::pipeline::batched_attention::{
    attention_attend_batched, attention_input_projection_batched,
    attention_moe_output_projection_batched, attention_moe_prepare_batched,
    attention_output_projection_batched, attention_prepare_batched, execute_fa_attend_step,
    execute_gated_attention_batched, AttentionBatchScratch, AttentionLayerWeights, FlashScratch,
    KvView,
};
pub(crate) use hipfire_dispatch::pipeline::batched_deltanet::run_fused_qkvza_key;
use hipfire_dispatch::pipeline::batched_deltanet::{
    execute_deltanet_batched, execute_deltanet_moe_layer_batched, DeltaNetBatchScratch,
    DeltaNetLayerView, DeltaNetStateView, GdnTapeView,
};
use hipfire_dispatch::pipeline::dump_hidden_localize;
use hipfire_dispatch::pipeline::execute_steps;
use hipfire_dispatch::pipeline::GemvInput;
use hipfire_dispatch::pipeline::Step;
use hipfire_runtime::llama;
use hipfire_runtime::llama::fused_rmsnorm_rotate_for_mq;

use hipfire_runtime::llama::weight_gemv_prerotated;
use hipfire_runtime::llama::weight_gemv_swiglu_residual;
use hipfire_runtime::llama::EmbeddingFormat;
use hipfire_runtime::llama::KvCacheExt;
use hipfire_runtime::llama::WeightTensor;
use rdna_compute::DType;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;
/// Attention weights of a dense full-attention layer, bound for the shared
/// batched attention executor.
fn fa_attention_weights(layer: &FullAttnLayerWeights) -> AttentionLayerWeights<'_> {
    AttentionLayerWeights {
        attn_norm: &layer.attn_norm,
        wq: layer.wq.dispatch_ref(),
        wk: layer.wk.dispatch_ref(),
        wv: layer.wv.dispatch_ref(),
        q_norm: &layer.q_norm,
        k_norm: &layer.k_norm,
        wo: layer.wo.dispatch_ref(),
        w_gate: Some(layer.w_gate.dispatch_ref()),
    }
}

/// Attention weights of a full-attention MoE layer (routed FFN: no fold).
fn fa_moe_attention_weights(layer: &FullAttnMoeLayerWeights) -> AttentionLayerWeights<'_> {
    AttentionLayerWeights {
        attn_norm: &layer.attn_norm,
        wq: layer.wq.dispatch_ref(),
        wk: layer.wk.dispatch_ref(),
        wv: layer.wv.dispatch_ref(),
        q_norm: &layer.q_norm,
        k_norm: &layer.k_norm,
        wo: layer.wo.dispatch_ref(),
        w_gate: None,
    }
}

fn attention_scratch(pbs: &PrefillBatchScratch) -> AttentionBatchScratch<'_> {
    AttentionBatchScratch {
        x_batch: &pbs.x_batch,
        x_rot_batch: &pbs.x_rot_batch,
        x_rot_f16_batch: &pbs.x_rot_f16_batch,
        fa_q_full_batch: &pbs.fa_q_full_batch,
        fa_q_batch: &pbs.fa_q_batch,
        fa_gate_batch: &pbs.fa_gate_batch,
        fa_k_batch: &pbs.fa_k_batch,
        fa_v_batch: &pbs.fa_v_batch,
        fa_attn_out_batch: &pbs.fa_attn_out_batch,
        fa_attn_out_rot_batch: &pbs.fa_attn_out_rot_batch,
        fa_attn_out_rot_f16_batch: &pbs.fa_attn_out_rot_f16_batch,
        x_norm_batch: &pbs.x_norm_batch,
        gate_ffn_batch: &pbs.gate_ffn_batch,
        positions: &pbs.positions,
        rope_positions: &pbs.rope_positions,
    }
}

fn flash_scratch(s: &Qwen35Scratch) -> FlashScratch<'_> {
    FlashScratch {
        flash_partials: &s.flash_partials,
        pos_buf: &s.pos_buf,
        flash_mode: s.flash_mode,
    }
}

pub(crate) fn kv_view(kv_cache: &llama::KvCache) -> KvView<'_> {
    KvView {
        k_gpu: &kv_cache.k_gpu,
        v_gpu: &kv_cache.v_gpu,
        givens_cos: &kv_cache.givens_cos,
        givens_sin: &kv_cache.givens_sin,
        compact_offset: kv_cache.compact_offset,
        physical_cap: kv_cache.physical_cap,
        quant_q8: kv_cache.quant_q8,
        quant_fp8: kv_cache.quant_fp8,
        tier: kv_cache.tier_inputs(),
        vmm_backend: kv_cache.uses_vmm_backend(),
    }
}

/// TriAttention pre-RoPE tap over `rows` Q (and optionally K) rows.
pub(crate) fn triattn_tap_rows(
    gpu: &mut Gpu,
    layer_idx: usize,
    q: &GpuTensor,
    k: &GpuTensor,
    rows: usize,
    config: &Qwen35Config,
) -> HipResult<()> {
    // Try the GPU reduction first (zero PCIe transfer; only when
    // install_tap_gpu() was used), else record on the CPU.
    let gpu_handled = hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
        gpu,
        layer_idx,
        &q.buf,
        rows,
        config.n_heads,
        config.head_dim,
    )?;
    if !gpu_handled {
        let n_q = config.n_heads * config.head_dim;
        let q_cpu = gpu.download_f32(q)?;
        if hipfire_runtime::triattn::tap_needs_k() {
            let n_k = config.n_kv_heads * config.head_dim;
            let k_cpu = gpu.download_f32(k)?;
            for b in 0..rows {
                hipfire_runtime::triattn::record_prerope_qk(
                    layer_idx,
                    &q_cpu[b * n_q..(b + 1) * n_q],
                    Some(&k_cpu[b * n_k..(b + 1) * n_k]),
                );
            }
        } else {
            for b in 0..rows {
                hipfire_runtime::triattn::record_prerope_q(
                    layer_idx,
                    &q_cpu[b * n_q..(b + 1) * n_q],
                );
            }
        }
    }
    Ok(())
}

/// The TriAttention pre-RoPE tap for one layer, when the tap is enabled.
#[allow(clippy::type_complexity)]
pub(crate) fn attention_tap<'a>(
    layer_idx: usize,
    config: &'a Qwen35Config,
) -> Option<Box<dyn Fn(&mut Gpu, &GpuTensor, &GpuTensor, usize) -> HipResult<()> + 'a>> {
    hipfire_runtime::triattn::tap_enabled().then(|| {
        Box::new(
            move |gpu: &mut Gpu, q: &GpuTensor, k: &GpuTensor, rows: usize| {
                triattn_tap_rows(gpu, layer_idx, q, k, rows, config)
            },
        ) as Box<dyn Fn(&mut Gpu, &GpuTensor, &GpuTensor, usize) -> HipResult<()> + 'a>
    })
}

/// Batched gated full attention of a dense layer through the shared executor.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_chunk_full_attn_attn(
    gpu: &mut Gpu,
    fa_attn_multirow: bool,
    layer: &FullAttnLayerWeights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    kv_cache: &llama::KvCache,
    n: usize,
    dim: usize,
    start_pos: usize,
    max_ctx_len: usize,
    ctx: &DispatchCtx,
    batch_semantics: BatchSemantics<'_>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    q8_wmma_arch: bool,
    arch_has_wmma: bool,
    kv_layer_idx: usize,
    layer_idx: usize,
    epilogue: BatchEpilogue<'_>,
    fusion: DflashFusionCtx,
    commit_stride: Option<usize>,
) -> HipResult<()> {
    let tap = attention_tap(layer_idx, config);
    execute_gated_attention_batched(
        gpu,
        fa_attn_multirow,
        &fa_attention_weights(layer),
        &super::program::hybrid_dims(config),
        &attention_scratch(pbs),
        &flash_scratch(s),
        &kv_view(kv_cache),
        n,
        dim,
        start_pos,
        max_ctx_len,
        ctx,
        batch_semantics,
        tree_verify,
        q8_wmma_arch,
        arch_has_wmma,
        kv_layer_idx,
        layer_idx,
        epilogue,
        fusion == DflashFusionCtx::ChainVerify,
        commit_stride,
        tap.as_deref(),
    )
}

/// Weights of a dense DeltaNet layer for the shared batched executor.
fn deltanet_layer_view(layer: &DeltaNetLayerWeights) -> DeltaNetLayerView<'_> {
    DeltaNetLayerView {
        attn_norm: &layer.attn_norm,
        wqkv: layer.wqkv.dispatch_ref(),
        wz: layer.wz.dispatch_ref(),
        w_beta: layer.w_beta.dispatch_ref(),
        w_alpha: layer.w_alpha.dispatch_ref(),
        dt_bias: &layer.dt_bias,
        a_log: &layer.a_log,
        conv_weight: &layer.conv_weight,
        norm_weight: &layer.norm_weight,
        wo: layer.wo.dispatch_ref(),
        w_gate: Some(layer.w_gate.dispatch_ref()),
    }
}

/// Weights of a DeltaNet MoE layer (routed FFN: no fold).
fn deltanet_moe_layer_view(layer: &DeltaNetMoeLayerWeights) -> DeltaNetLayerView<'_> {
    DeltaNetLayerView {
        attn_norm: &layer.attn_norm,
        wqkv: layer.wqkv.dispatch_ref(),
        wz: layer.wz.dispatch_ref(),
        w_beta: layer.w_beta.dispatch_ref(),
        w_alpha: layer.w_alpha.dispatch_ref(),
        dt_bias: &layer.dt_bias,
        a_log: &layer.a_log,
        conv_weight: &layer.conv_weight,
        norm_weight: &layer.norm_weight,
        wo: layer.wo.dispatch_ref(),
        w_gate: None,
    }
}

fn deltanet_scratch(pbs: &PrefillBatchScratch) -> DeltaNetBatchScratch<'_> {
    DeltaNetBatchScratch {
        x_batch: &pbs.x_batch,
        x_rot_batch: &pbs.x_rot_batch,
        x_rot_f16_batch: &pbs.x_rot_f16_batch,
        x_norm_batch: &pbs.x_norm_batch,
        gate_ffn_batch: &pbs.gate_ffn_batch,
        dn_qkv_batch: &pbs.dn_qkv_batch,
        dn_z_batch: &pbs.dn_z_batch,
        dn_z_fold_batch: &pbs.dn_z_fold_batch,
        dn_beta_batch: &pbs.dn_beta_batch,
        dn_alpha_batch: &pbs.dn_alpha_batch,
        dn_q_raw_batch: &pbs.dn_q_raw_batch,
        dn_k_raw_batch: &pbs.dn_k_raw_batch,
        dn_v_batch: &pbs.dn_v_batch,
        dn_q_batch: &pbs.dn_q_batch,
        dn_k_batch: &pbs.dn_k_batch,
        dn_attn_out_batch: &pbs.dn_attn_out_batch,
        dn_normed_batch: &pbs.dn_normed_batch,
        dn_normed_rot_batch: &pbs.dn_normed_rot_batch,
        dn_normed_rot_f16_batch: &pbs.dn_normed_rot_f16_batch,
        dn_s_tape_q8: &pbs.dn_s_tape_q8,
        dn_s_tape_scales: &pbs.dn_s_tape_scales,
        dn_s_tape_f32: &pbs.dn_s_tape_f32,
    }
}

pub(crate) fn deltanet_state_view(state: &DeltaNetState) -> DeltaNetStateView<'_> {
    DeltaNetStateView {
        s_matrices: &state.s_matrices,
        s_scales: &state.s_scales,
        conv_states: &state.conv_states,
        s_ef_residual: &state.s_ef_residual,
        quant: state.quant,
    }
}

fn gdn_tape_view(tape: &crate::speculative::GdnTape) -> GdnTapeView<'_> {
    GdnTapeView {
        max_n: tape.max_n,
        qkv_dim: tape.qkv_dim,
        qkv_bufs: &tape.qkv_bufs,
        alpha_bufs: &tape.alpha_bufs,
        beta_bufs: &tape.beta_bufs,
    }
}

/// Batched gated DeltaNet of a dense layer through the shared executor.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_chunk_delta_net_attn(
    gpu: &mut Gpu,
    layer: &DeltaNetLayerWeights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    dn_state: &mut DeltaNetState,
    n: usize,
    dim: usize,
    k_dim: usize,
    v_dim: usize,
    n_v_heads: usize,
    hd: usize,
    batch_semantics: BatchSemantics<'_>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    gdn_tape: Option<&crate::speculative::GdnTape>,
    tape_offset: usize,
    delta_layer_idx: usize,
    q8_wmma_arch: bool,
    arch_has_wmma: bool,
    epilogue: BatchEpilogue<'_>,
    fusion: DflashFusionCtx,
    commit_stride: Option<usize>,
    gdn_chunk_scan_admitted: bool,
) -> HipResult<()> {
    let tape = gdn_tape.map(gdn_tape_view);
    execute_deltanet_batched(
        gpu,
        &deltanet_layer_view(layer),
        &super::program::hybrid_dims(config),
        &deltanet_scratch(pbs),
        &deltanet_state_view(dn_state),
        n,
        dim,
        k_dim,
        v_dim,
        n_v_heads,
        hd,
        batch_semantics,
        tree_verify,
        tape.as_ref(),
        tape_offset,
        delta_layer_idx,
        q8_wmma_arch,
        arch_has_wmma,
        epilogue,
        fusion == DflashFusionCtx::ChainVerify,
        commit_stride,
        gdn_chunk_scan_admitted,
    )
}

/// Dense SwiGLU FFN weights of one decoder layer.
pub(crate) struct DenseFfnWeights<'a> {
    pub norm: &'a GpuTensor,
    pub gate: &'a WeightTensor,
    pub up: &'a WeightTensor,
    pub down: &'a WeightTensor,
}

/// Batched dense FFN through the shared SwiGLU FFN op.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_chunk_dense_ffn(
    gpu: &mut Gpu,
    ffn: DenseFfnWeights<'_>,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    n: usize,
    dim: usize,
    hidden_dim: usize,
    q8_wmma_arch: bool,
    epilogue: BatchEpilogue<'_>,
    fusion: DflashFusionCtx,
) -> HipResult<()> {
    let ctx = DispatchCtx::new(gpu);
    let op = hipfire_dispatch::pipeline::hybrid::SwigluFfnOp {
        rows: n,
        eps: config.norm_eps,
        x: &pbs.x_batch,
        norm: ffn.norm,
        w_gate: ffn.gate.dispatch_ref(),
        w_up: ffn.up.dispatch_ref(),
        w_down: ffn.down.dispatch_ref(),
        plain: &pbs.x_rot_batch,
        x_rot: &pbs.x_rot_batch,
        gate: &pbs.gate_ffn_batch,
        up: &pbs.up_batch,
        hidden: &pbs.ffn_hidden_batch,
        batch: Some(hipfire_dispatch::pipeline::batched::SwigluFfnBatch {
            dim,
            hidden_dim,
            x_rot_f16: &pbs.x_rot_f16_batch,
            hidden_f16: &pbs.ffn_hidden_f16_batch,
            epilogue,
            chain_verify: fusion == DflashFusionCtx::ChainVerify,
            q8_wmma_arch,
            packed_mq4: !pbs.lean,
        }),
    };
    execute_steps(gpu, &ctx, &[Step::SwigluFfn(op)]).map_err(|e| HipError::new(0, &e.to_string()))
}

/// Batched prefill entry point: processes N prompt tokens in one call,
/// writing the last token's logits into `scratch.logits` and leaving
/// the KV cache + DeltaNet state advanced by N positions.
///
/// Takes the batched kernel path when ALL linear-attention layer weights
/// are MQ4G256 (the batched element-wise kernels are MQ-specific).
/// Otherwise falls back to a per-token loop over `forward_scratch` that's
/// byte-identical to decode. FA layers always use a per-token gather/scatter
/// fallback — the FA causal attention kernel can't yet be batched (task #71).
///
/// `gated_delta_net_q8_batch_seq` runs one launch per LA layer; the kernel
/// loops over the N tokens internally and requants the Q8 state once at
/// launch end (default fast kernel; `HIPFIRE_DN_REQUANT_PER_TOKEN=1` opts
/// into the per-token round-trip slow kernel). Chunk-boundary requants are
/// part of the trajectory: two 512-row launches are NOT numerically equal
/// to one 1024-row launch (dropped mid Q8 round-trip plus a
/// `frame + lane * n_tokens` reseed of the final requant), so the F2 pair
/// path keeps the DeltaNet sequential kernels per-half — see its envelope.
///
/// `tokens`: slice of prompt tokens to prefill in order.
/// `start_pos`: first KV cache / DeltaNet position to write. Positions
/// `start_pos .. start_pos + tokens.len()` get populated.
/// On return, `scratch.logits` holds the logits for the *last* token
/// (position `start_pos + tokens.len() - 1`).
///
/// `hidden_rb`: if `Some`, post-layer residual hidden states are captured
/// into the ring buffer for the configured extract layers. Used by the
/// DFlash target-side verify path to batch `verify_dflash_block` into a
/// single forward launch (MVP does B per-token forwards — 88 ms on 4B;
/// this path drops it to ~40 ms with batched forward, further improvement
/// possible with batched lm_head). The per-token fallback also honors it,
/// so the fast-path eligibility doesn't change behavior.
///
/// `per_token_hidden_out`: if `Some`, writes post-output-norm hidden state
/// for each of the N tokens into the provided [N × dim] buffer. The caller
/// then loops `weight_gemv(weights.output, hidden_row, logits)` to recover
/// per-token logits. Required for DFlash verify (needs all B positions'
/// logits, not just the last). `None` preserves the existing "last token
/// only" semantics where logits land in `scratch.logits`.
///
/// `gdn_tape`: if `Some`, captures the post-processed `(q, k, v, α, β)` for
/// every DN (LinearAttention) layer and block position BEFORE the batched
/// `gated_delta_net_q8_batch_seq` call. Enables the DFlash rollback path
/// to replay GDN recurrence from a pre-verify S-state snapshot for
/// `accept_len + 1` steps — no full-target re-run needed.
#[allow(clippy::too_many_arguments)]
/// Conservative cross-arch upper bound on `forward_prefill_batch`'s per-chunk
/// size. Adaptive-KV outer boundaries, eviction hard-caps, and callers that
/// need a fixed staging ceiling still use this constant. Production default
/// chunking is arch-aware via [`prefill_max_batch`] (measured 512 on exact
/// gfx1100, 384 on exact gfx1201; 256 elsewhere). Exposed so callers sizing
/// `HiddenStateRingBuffer` staging can match a safe chunk ceiling (staging
/// smaller than a chunk will assert-fail on prompt seeding of long prompts).
pub const PREFILL_MAX_BATCH: usize = 256;
/// Minimum number of rows for the batched prefill kernels.
const MIN_BATCH: usize = 2;

/// gfx1100-measured default prefill chunk size (Qwen3.8 / MQ4V2 gate-up BT path).
/// Exact `gfx1100` only — not gfx1101/1102/1151 or other gfx11 variants.
const PREFILL_DEFAULT_BATCH_GFX1100: usize = 512;

/// gfx1201-measured FP8 chunk size (Qwen3.8 FP8-WMMA MQ4v2 prefill path).
/// Applies only when all three FP8 projection flags are set (see
/// [`fp8_chunk512_for_gpu`]); explicit `HIPFIRE_PREFILL_MAX_BATCH` still wins.
const PREFILL_DEFAULT_BATCH_GFX1201_FP8: usize = 512;

/// True when the FP8 prefill path is fully admitted on exact gfx1201: all
/// three FP8 projection flags set. The chunk default then rises 384 -> 512
/// (measured sweet spot under FP8-WMMA). The 256-row hidden-ring staging,
/// adaptive-KV outer-chunk, and eviction internal-chunk caps are enforced
/// downstream (`prefill_effective_chunk_batch` mins with PBS/ring staging;
/// `forward_prefill_batch_capped` mins with the caller cap), exactly as they
/// already absorb the 384 default — a larger default only widens the
/// configured ceiling, never a staging write.
#[inline]
fn fp8_chunk512_for_gpu(gpu: &Gpu) -> bool {
    gpu.arch == "gfx1201"
        && gpu.flags.gfx12_mq4v2_fp8_gateup
        && gpu.flags.gfx12_mq4v2_fp8_resid
        && gpu.flags.gfx12_mq4v2_fp8_qkvza
}

/// gfx1201-measured default prefill chunk size (Qwen3.8 prefill sweet spot).
/// Exact `gfx1201` only — not gfx1200 or other gfx12 variants.
const PREFILL_DEFAULT_BATCH_GFX1201: usize = 384;

/// Architecture default for prefill chunk size when
/// `HIPFIRE_PREFILL_MAX_BATCH` is unset or invalid. `fp8_chunk512` lifts
/// exact gfx1201 384 -> 512 when the full FP8 path is admitted (see
/// [`fp8_chunk512_for_gpu`]); every other arch ignores it, so the F16
/// default path is unchanged.
#[inline]
fn prefill_max_batch_for_arch(arch: &str, fp8_chunk512: bool) -> usize {
    if arch == "gfx1100" {
        PREFILL_DEFAULT_BATCH_GFX1100
    } else if arch == "gfx1201" {
        if fp8_chunk512 {
            PREFILL_DEFAULT_BATCH_GFX1201_FP8
        } else {
            PREFILL_DEFAULT_BATCH_GFX1201
        }
    } else {
        PREFILL_MAX_BATCH
    }
}

fn explicit_prefill_max_batch() -> Option<usize> {
    hipfire_config::developer_var("HIPFIRE_PREFILL_MAX_BATCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v >= MIN_BATCH)
}

/// Architecture default for the widened ordinary-prefill chunk ceiling
/// (`prefill.chunk_rows`): 8192 on gfx1100/gfx1151/gfx1201, 512 elsewhere.
#[inline]
fn prefill_chunk_rows_default(arch: &str) -> usize {
    if matches!(arch, "gfx1100" | "gfx1151" | "gfx1201") {
        8192
    } else {
        512
    }
}

/// Requested widened chunk ceiling: explicit `HIPFIRE_PREFILL_MAX_BATCH`
/// wins, then `prefill.chunk_rows` (config file or
/// `HIPFIRE_PREFILL_CHUNK_ROWS`), then the arch default above. Values below
/// `WIDENED_COMMIT_ROWS` keep legacy behavior downstream.
fn prefill_chunk_rows_requested(gpu: &Gpu) -> usize {
    if let Some(explicit) = explicit_prefill_max_batch() {
        return explicit;
    }
    if let Ok(s) = hipfire_config::developer_var("HIPFIRE_PREFILL_CHUNK_ROWS") {
        if let Ok(v) = s.parse::<usize>() {
            if v >= MIN_BATCH {
                return v;
            }
        }
    }
    prefill_chunk_rows_default(gpu.arch.as_str())
}

fn dense_layers_are_all_mq4v2(weights: &Qwen35Weights) -> bool {
    dense_layers_have_projection_dtype(weights, DType::MQ4G256V2)
}

fn dense_layers_have_projection_dtype(weights: &Qwen35Weights, dtype: DType) -> bool {
    !weights.layers.is_empty()
        && weights.layers.iter().all(|layer| match layer {
            LayerWeights::DeltaNet(layer) => [
                &layer.wqkv,
                &layer.wz,
                &layer.w_alpha,
                &layer.w_beta,
                &layer.wo,
                &layer.w_gate,
                &layer.w_up,
                &layer.w_down,
            ]
            .iter()
            .all(|weight| weight.gpu_dtype == dtype),
            LayerWeights::FullAttn(layer) => [
                &layer.wq,
                &layer.wk,
                &layer.wv,
                &layer.wo,
                &layer.w_gate,
                &layer.w_up,
                &layer.w_down,
            ]
            .iter()
            .all(|weight| weight.gpu_dtype == dtype),
            LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_) => false,
        })
}

fn native_mq4_widened_requested() -> bool {
    hipfire_config::developer_var("HIPFIRE_GFX1100_MQ4_WIDE_PREFILL")
        .ok().as_deref() == Some("1")
}

/// Separately admitted native MQ4 route: no V2 producer or lean PBS contract.
/// Large projections use default MMQ or opt-in packed FFN (the same 144-byte
/// activation scratch slot); small tails keep bounded FP16/ksplit fallback.
fn native_mq4_widened_route(gpu: &Gpu, weights: &Qwen35Weights) -> bool {
    native_mq4_widened_requested()
        && dense_layers_have_projection_dtype(weights, DType::MQ4G256)
        && native_mq4_widened_flags_admitted(&gpu.arch, &gpu.flags, gpu.mmq_screen.enabled)
}

fn native_mq4_widened_flags_admitted(
    arch: &str, flags: &rdna_compute::FeatureFlags, screen: bool,
) -> bool {
    arch == "gfx1100"
        && !flags.fp16_disabled
        && !flags.rocblas_all_archs
        && !flags.gemv_dp4a.unwrap_or(false)
        && flags.mmq_override.is_none()
        && flags.mmq_min_batch.is_none()
        && !screen
        && !flags.qkvza_split_tail
        && !flags.mw16
        && flags.wo_wmma_variant.is_none()
}

fn prefill_max_batch_for_model(gpu: &Gpu, weights: &Qwen35Weights) -> usize {
    explicit_prefill_max_batch().unwrap_or_else(|| {
        if gpu.arch == "gfx1151" && dense_layers_are_all_mq4v2(weights) {
            512
        } else {
            prefill_max_batch_for_arch(gpu.arch.as_str(), fp8_chunk512_for_gpu(gpu))
        }
    })
}

/// Resolve the prefill chunk upper bound for `gpu`.
///
/// Honors explicit `HIPFIRE_PREFILL_MAX_BATCH` when it parses as an integer
/// `>= MIN_BATCH` (2); otherwise returns the arch default — 512 on exact
/// gfx1100, 384 on exact gfx1201 (512 when the full FP8 path is admitted,
/// see [`fp8_chunk512_for_gpu`]), [`PREFILL_MAX_BATCH`] (256) on every other
/// arch string. Capped entry points further min with an explicit caller
/// ceiling via `prefill_max_batch(gpu).min(max_batch_cap)`.
pub fn prefill_max_batch(gpu: &Gpu) -> usize {
    explicit_prefill_max_batch()
        .unwrap_or_else(|| prefill_max_batch_for_arch(gpu.arch.as_str(), fp8_chunk512_for_gpu(gpu)))
}

/// Ceiling on the TP compensation below; beyond it the prefill kernels stop
/// gaining and the per-chunk scratch keeps growing.
const PREFILL_TP_BATCH_CAP: usize = 2048;

/// Prefill chunk for one rank of a `tp`-way split.
///
/// The prefill attention grid is `local_heads x chunk / M_TILE`, and TP divides
/// the heads, so a rank running the arch default launches `tp` times fewer
/// workgroups than a single card and starves an already latency-bound kernel.
/// Scaling the chunk by `tp` restores the single-card workgroup count.
/// Measured on gfx1100, Qwen3.8-27B, 33k prompt, tp=2: 649 -> 773 tok/s with
/// byte-identical output. An explicit `HIPFIRE_PREFILL_MAX_BATCH` wins.
pub fn prefill_max_batch_tp(gpu: &Gpu, tp: usize) -> usize {
    if let Some(explicit) = explicit_prefill_max_batch() {
        return explicit;
    }
    let base = prefill_max_batch_for_arch(gpu.arch.as_str(), fp8_chunk512_for_gpu(gpu));
    base.saturating_mul(tp.max(1)).min(PREFILL_TP_BATCH_CAP)
}

/// Prefill chunk ceiling for EP sequential load/serve staging.
///
/// Honors explicit `HIPFIRE_PREFILL_MAX_BATCH` when set (`>= MIN_BATCH`);
/// otherwise the EP shared default of 512 (not arch-scaled TP compensation).
pub fn prefill_max_batch_ep() -> usize {
    explicit_prefill_max_batch().unwrap_or(512)
}
/// Widened ordinary prefill: larger contiguous GEMM/PBS chunks with the
/// DeltaNet kernel still launched once per 512-row commit on views and flash
/// attention on 512-row tile views. No kernel source changes.
///
/// gfx1201 keeps its FP8 projection route. gfx1100/gfx1151 use their existing
/// IU4 K16 projection route and Q8 FA2 cache route; no gfx12 FP8 kernel is
/// admitted on gfx11.
/// Only `None` (legacy cadence) and `Some(512)` are legal commit strides.
/// `n` continues to mean chunk rows, never commit stride.
pub(crate) use hipfire_dispatch::pipeline::batched_attention::WIDENED_COMMIT_ROWS;
///
/// Staged GEMM-width rungs. Merging powers of two limits new GEMM widths to
/// these five; a requested ceiling snaps down to the largest rung it covers.
const WIDENED_RUNGS: [usize; 5] = [512, 1024, 2048, 4096, 8192];
///
/// Performance ceiling for the staged rungs above: `prefill.chunk_rows`
/// (config file or `HIPFIRE_PREFILL_CHUNK_ROWS`, default 8192 on
/// gfx1100/gfx1151/gfx1201 and 512 elsewhere; explicit
/// `HIPFIRE_PREFILL_MAX_BATCH` wins). See [`prefill_chunk_rows_requested`].
/// Memory admission may still select a smaller rung per device.
///
/// Uncommitted-VRAM headroom reserved by the per-device capacity admission
/// (plan §4.2). Conservative policy margin covering remaining state, output,
/// graph/JIT and ordinary transients — not permission for unaccounted KV.
const WIDENED_VRAM_HEADROOM_BYTES: usize = 1 << 30;
///
/// Arch gate for the widened ordinary route. gfx1201 uses FP8 projections;
/// gfx1100/gfx1151 use the existing IU4 K16 projections and Q8 FA2 path.
#[inline]
fn widened_arch_admitted(arch: &str) -> bool {
    matches!(arch, "gfx1100" | "gfx1151" | "gfx1201")
}
///
/// Exact dense 27B shape the widened route admits: the shared 64-layer
/// predicate plus the 17408 hidden_dim that fixes the allocation envelope.
#[inline]
fn widened_dense_shape_admitted(config: &Qwen35Config) -> bool {
    config.hidden_dim == 17_408
        && super::config::qwen36_27b_dense_shape(config, config.linear_num_value_heads)
}
///
/// Verbatim truthy predicate of rdna-compute's private `dn_requant_per_token`
/// (norm.rs): non-empty and non-"0" means per-token requant. Read here
/// directly so the widened route needs no GDN-wrapper/TU change; per-token
/// requant inserts 511 extra Q8/EF boundaries per 512 rows and can never
/// equal the 512-single-end trajectory, so it stays on the legacy path.
#[inline]
fn dn_requant_per_token_env() -> bool {
    hipfire_config::developer_var("HIPFIRE_DN_REQUANT_PER_TOKEN")
        .map(|v| {
            let v = v.trim();
            !v.is_empty() && v != "0"
        })
        .unwrap_or(false)
}
///
/// Static + perf part of the widened ordinary admission. Pure except for env
/// reads: no GPU work, no allocation. Returns `None` when the request is not
/// statically eligible (caller keeps its legacy ceiling unchanged); else the
/// performance-admitted chunk ceiling in rows.
///
/// A requested ceiling below 512 is returned as-is (explicit small values
/// retain existing behavior); otherwise the ceiling snaps down to the
/// largest staged rung the request covers. The request is
/// [`prefill_chunk_rows_requested`] (explicit override, then
/// `prefill.chunk_rows` config, then the 8192/512 arch default).
fn ordinary_prefill_static_ceiling(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    dn_state: &DeltaNetState,
) -> Option<usize> {
    if !widened_arch_admitted(gpu.arch.as_str()) {
        return None;
    }
    if !widened_dense_shape_admitted(config) {
        return None;
    }
    let native_mq4 = native_mq4_widened_route(gpu, weights);
    if !dense_layers_are_all_mq4v2(weights) && !native_mq4 {
        return None;
    }
    // gfx1201 keeps all four FP8 projection routes. gfx11 substitutes the
    // existing IU4 K16 projection route; its full-attention layers remain on
    // Q8 FA2 because the packet kernel and FP8 KV format are gfx1201-only.
    let projection_route_admitted = if native_mq4 {
        true
    } else if gpu.arch == "gfx1201" {
        gpu.flags.gfx12_mq4v2_fp8_gateup
            && gpu.flags.gfx12_mq4v2_fp8_resid
            && gpu.flags.gfx12_mq4v2_fp8_qkvza
            && gpu.flags.gfx12_mq4v2_fp8_qkv
    } else {
        gpu.flags.iu4_prefill_enabled()
    };
    if !projection_route_admitted {
        return None;
    }
    // Actual Q8 state with a complete EF owner for every LA layer, single
    // lane, and single-end requant. EF-off changes the global frame schedule
    // under layer-major grouping (frame enters the arithmetic without EF),
    // so it resolves to the legacy 512 path even under a larger ceiling.
    let n_la = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::LinearAttention)
        .count();
    if dn_state.quant != StateQuant::Q8 || n_la == 0 {
        return None;
    }
    if dn_state.s_ef_residual.len() != n_la || dn_state.s_ef_residual.iter().any(|t| t.numel() == 0)
    {
        return None;
    }
    let single_lane_s =
        config.linear_num_value_heads * config.linear_value_head_dim * config.linear_value_head_dim;
    if dn_state.s_matrices.len() != n_la
        || dn_state
            .s_matrices
            .iter()
            .any(|t| t.numel() != single_lane_s)
    {
        return None;
    }
    if dn_requant_per_token_env() {
        return None;
    }
    let requested = prefill_chunk_rows_requested(gpu);
    if requested < WIDENED_COMMIT_ROWS {
        return Some(requested);
    }
    // Largest staged rung the request covers.
    let mut rung = WIDENED_COMMIT_ROWS;
    for &r in WIDENED_RUNGS.iter() {
        if r <= requested {
            rung = r;
        }
    }
    Some(rung)
}
/// Only the retained widened PBS owner can use this representation: ordinary
/// sequential prefill runs eager, with no tree/chain verify or MoE/TP/EP.
/// Check the exact same producer predicates used by the fallback decisions.
fn lean_pbs_route(gpu: &Gpu, weights: &Qwen35Weights, config: &Qwen35Config, n: usize) -> bool {
    matches!(gpu.arch.as_str(), "gfx1100" | "gfx1151")
        && config.num_experts == 0
        && config.linear_value_head_dim == 128
        && dense_layers_are_all_mq4v2(weights)
        && gpu.iu4_producer_sidecar_active(n, config.dim)
        && gpu.iu4_producer_sidecar_active(n, config.hidden_dim)
        && gpu.iu4_gfx11_producer_quant_fused_active(
            n,
            config.linear_num_value_heads * config.linear_value_head_dim,
        )
        && gpu.iu4_gfx11_producer_quant_fused_active(n, config.n_heads * config.head_dim)
        && weights.layers.iter().all(|layer| match layer {
            LayerWeights::DeltaNet(layer) => {
                layer.wo.k == config.linear_num_value_heads * config.linear_value_head_dim
            }
            LayerWeights::FullAttn(layer) => {
                let k = config.n_heads * config.head_dim;
                layer.wo.k == k
                    && layer
                        .wo
                        .awq_scale
                        .as_ref()
                        .is_some_and(|awq| awq.numel() >= k)
            }
            _ => false,
        })
}

fn lean_pbs_requested() -> bool {
    hipfire_config::developer_var("HIPFIRE_GFX11_LEAN_PBS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true)
}

fn lean_dense_prefill_allocation_bytes(config: &Qwen35Config, rows: usize) -> Option<usize> {
    let full = dense_prefill_allocation_bytes(config, rows)?;
    let v = config
        .linear_num_value_heads
        .checked_mul(config.linear_value_head_dim)?;
    let q = config.n_heads.checked_mul(config.head_dim)?;
    let fallback_f32 = config
        .dim
        .checked_add(v.checked_mul(2)?)?
        .checked_add(config.hidden_dim)?
        .checked_add(q)?
        .checked_mul(4)?;
    let verify_f16 = config
        .dim
        .checked_add(v)?
        .checked_add(config.hidden_dim)?
        .checked_add(q)?
        .checked_mul(2)?
        .checked_add(4)?;
    full.checked_sub(rows.saturating_sub(64).checked_mul(fallback_f32)?)
        .and_then(|bytes| bytes.checked_sub(rows.checked_mul(verify_f16)?))
}

///
/// Steady-state bytes of one dense tape-free [`PrefillBatchScratch`] at
/// `rows` rows: every unconditional allocation including positions, rope,
/// tokens, the four F16 sidecars and the 256-byte prologue control plane.
/// `None` for MoE configs (out of scope) or on arithmetic overflow.
///
/// Term-by-term mirror of `PrefillBatchScratch::new_opt(..., false)` for
/// dense configs: F32 `3D + 4K + 8V + 2H + 3I + 6Q + 2W + 3` elems/row, F16
/// `D + V + I + Q` elems/row, plus 256 fixed bytes.
fn dense_prefill_allocation_bytes(config: &Qwen35Config, rows: usize) -> Option<usize> {
    if config.num_experts != 0 {
        return None;
    }
    let d = config.dim;
    let i = config.hidden_dim;
    let k = config
        .linear_num_key_heads
        .checked_mul(config.linear_key_head_dim)?;
    let v = config
        .linear_num_value_heads
        .checked_mul(config.linear_value_head_dim)?;
    let h = config.linear_num_value_heads;
    let q = config.n_heads.checked_mul(config.head_dim)?;
    let w = config.n_kv_heads.checked_mul(config.head_dim)?;
    let f32_per_row = 3usize
        .checked_mul(d)?
        .checked_add(4usize.checked_mul(k)?)?
        .checked_add(8usize.checked_mul(v)?)?
        .checked_add(2usize.checked_mul(h)?)?
        .checked_add(3usize.checked_mul(i)?)?
        .checked_add(6usize.checked_mul(q)?)?
        .checked_add(2usize.checked_mul(w)?)?
        .checked_add(3)?;
    let f16_per_row = d.checked_add(v)?.checked_add(i)?.checked_add(q)?;
    rows.checked_mul(f32_per_row)?
        .checked_mul(4)?
        .checked_add(rows.checked_mul(f16_per_row)?.checked_mul(2)?)?
        .checked_add(256)
}
///
/// FP8 pre-pass bytes/row at the conservative `Kmax = hidden_dim` envelope:
/// E4M3 codes plus `8 * (I/256)` half-sum bytes plus one f32 row scale.
/// Mirrors `mq4v2_fp8_needed(n, I)` summed over its three slots.
fn fp8_row_bytes_wide(config: &Qwen35Config) -> Option<usize> {
    let groups = config.hidden_dim.checked_div(256)?;
    config
        .hidden_dim
        .checked_add(groups.checked_mul(8)?)?
        .checked_add(4)
}
///
/// IU4 K16 prelude bytes/row at the conservative `Kmax = hidden_dim`
/// envelope: one 72-byte `block_i4_128` per K/128 block.
fn iu4_row_bytes_wide(config: &Qwen35Config) -> Option<usize> {
    config
        .hidden_dim
        .checked_add(127)?
        .checked_div(128)?
        .checked_mul(72)
}
/// Per-token physical K+V stride for the selected cache encoding. Only
/// full-attention layers own KV; DeltaNet layers carry no token rows.
pub fn vmm_kv_token_bytes(
    config: &Qwen35Config,
    pair: hipfire_runtime::kv_mode::KvPair,
    adaptive: bool,
) -> Option<usize> {
    use hipfire_runtime::kv_mode::{KvMode, KvPair, VMode};
    let head = config.head_dim;
    let k_head = if adaptive {
        head / 2 + 4 // FWHT4 start, not a future adaptive floor
    } else {
        match pair.k() {
            KvMode::Fp8 => head + 2,
            KvMode::Bf16 | KvMode::F16 => head.checked_mul(2)?,
            KvMode::Q8 => head / 32 * 34,
            KvMode::Asym2 | KvMode::Fwht2 => head / 4 + 4,
            KvMode::Asym3 | KvMode::Fwht3 => head * 3 / 8 + 4,
            KvMode::Asym4 | KvMode::Fwht4 => head / 2 + 4,
        }
    };
    let v_head = match pair {
        KvPair::Native(_) => k_head,
        KvPair::Split(_, VMode::Q8) => head / 32 * 34,
        KvPair::Split(_, v) => 4 + head.checked_mul(v.bits() as usize)? / 8,
    };
    config
        .layer_types
        .iter()
        .filter(|layer| **layer == LayerType::FullAttention)
        .count()
        .checked_mul(config.n_kv_heads)?
        .checked_mul(k_head.checked_add(v_head)?)
        .filter(|bytes| *bytes > 0)
}

/// Charge only the minimum viable 512-row prefill scratch when sizing KV.
/// Larger PBS buffers are allocated lazily and admitted per request from free
/// VRAM *after mapped KV*, not subtracted from the lifetime context bound.
pub fn minimum_prefill_reservation_bytes(config: &Qwen35Config, arch: &str) -> Option<usize> {
    let base = dense_prefill_reservation_bytes(config, WIDENED_COMMIT_ROWS, arch)?;
    // The loader has not classified all projection tensors yet. Under the
    // explicit experimental MQ4 opt-ins, conservatively reserve their prelude
    // too; the default V2/other-format reservation remains unchanged.
    let packed = hipfire_config::developer_var("HIPFIRE_GFX1100_PACKED_MQ4_PREFILL")
        .ok().as_deref() == Some("1");
    if arch == "gfx1100" && (native_mq4_widened_requested() || packed) {
        base.checked_add(native_mq4_projection_deficit(config, WIDENED_COMMIT_ROWS, 0, 0, 0)?)
    } else {
        Some(base)
    }
}

/// Large rows use the shared 144-byte Q8_1 MMQ slot. A non-power-of-two
/// request may leave a 2..127 row FP16 tail even under a wide ceiling;
/// account for that slot and the four deterministic residual partials too.
fn native_mq4_projection_deficit(
    config: &Qwen35Config,
    rows: usize,
    live_mmq: usize,
    live_f16: usize,
    live_partials: usize,
) -> Option<usize> {
    let mmq = config.hidden_dim.checked_add(127)?.checked_div(128)?
        .checked_mul(144)?.checked_mul(rows)?;
    let tail = rows.min(127);
    let f16 = tail.checked_mul(config.hidden_dim)?.checked_mul(2)?;
    let partials = tail.checked_mul(config.dim)?.checked_mul(4)?.checked_mul(4)?;
    mmq.saturating_sub(live_mmq)
        .checked_add(f16.saturating_sub(live_f16))?
        .checked_add(partials.saturating_sub(live_partials))
}

fn dense_prefill_reservation_bytes(
    config: &Qwen35Config,
    rows: usize,
    arch: &str,
) -> Option<usize> {
    let projection = if arch == "gfx1201" {
        fp8_row_bytes_wide(config)?
    } else {
        iu4_row_bytes_wide(config)?
    };
    let q16 = WIDENED_COMMIT_ROWS
        .checked_mul(config.n_heads)?
        .checked_mul(config.head_dim)?
        .checked_mul(2)?;
    dense_prefill_allocation_bytes(config, rows)?
        .checked_add(rows.checked_mul(projection)?)?
        .checked_add(q16)?
        .checked_add(WIDENED_VRAM_HEADROOM_BYTES)
}
/// Retain a widened ordinary PBS while there is room for the minimum prefill
/// and ordinary headroom. On a discrete gfx1201 card, future VMM KV growth
/// reclaims the cache just before mapping if that growth needs its memory.
/// Reserving *all* unmapped KV here made mixed-size prefills repeatedly free
/// and reallocate the same multi-gigabyte PBS despite no KV growth.
fn can_retain_widened_pbs(gpu: &Gpu, kv: &llama::KvCache, config: &Qwen35Config) -> bool {
    if !kv.uses_vmm_backend() {
        return true;
    }
    let Some(minimum) = minimum_prefill_reservation_bytes(config, &gpu.arch) else {
        return false;
    };
    let grow_only = widened_pbs_grow_only(gpu);
    let mut unmapped = 0usize;
    if !grow_only {
        for tensor in kv.k_gpu.iter().chain(kv.v_gpu.iter()) {
            if tensor.buf.is_vmm_owner() {
                let Some(mapped) = gpu.vmm_mapped_bytes(tensor) else {
                    return false;
                };
                unmapped = unmapped.saturating_add(tensor.byte_size().saturating_sub(mapped));
            }
        }
    }
    gpu.hip
        .get_vram_info()
        .is_ok_and(|(free, _)| free >= unmapped.saturating_add(minimum).saturating_add(128 << 20))
}

/// gfx1100 stays off the grow-only route on purpose: there, later pp8192
/// requests fall back to 2×4096 chunks (the freed PBS sits in the process
/// pool, which admission cannot see), and a same-binary e2e A/B measured that
/// fallback 0.16–0.21% faster than retaining the 8192-row PBS (gfx11-1).
fn widened_pbs_grow_only(gpu: &Gpu) -> bool {
    gpu.arch == "gfx1201"
        && !gpu.is_uma()
        && hipfire_config::developer_var("HIPFIRE_WIDENED_PBS_GROW_ONLY")
            .ok()
            .as_deref()
            != Some("0")
}

/// Give a growing VMM KV mapping priority over optional cached PBS. This
/// check is called before mapping, not when merely changing prompt size.
/// The 1 GiB reservation in `minimum` covers VMM chunk/granularity rounding
/// and other transient allocations; overflow or unknown capacity evicts.
pub(crate) fn release_widened_pbs_for_kv_growth(
    gpu: &mut Gpu,
    kv: &llama::KvCache,
    config: &Qwen35Config,
    scratch: &Qwen35Scratch,
    required_tokens: usize,
) -> HipResult<()> {
    if scratch.widened_prefill_batch.borrow().is_none() {
        return Ok(());
    }
    if !kv.needs_mapped_growth(required_tokens)? || !widened_pbs_grow_only(gpu) {
        return Ok(());
    }
    let keep = kv
        .planned_mapped_growth_bytes(gpu, required_tokens)
        .ok()
        .and_then(|growth| {
            minimum_prefill_reservation_bytes(config, &gpu.arch)
                .and_then(|minimum| growth.checked_add(minimum))
        })
        .and_then(|needed| needed.checked_add(128 << 20))
        .is_some_and(|needed| {
            gpu.hip
                .get_vram_info()
                .is_ok_and(|(free, _)| free >= needed)
        });
    if !keep {
        if let Some(old) = scratch.widened_prefill_batch.borrow_mut().take() {
            old.free_gpu(gpu)?;
        }
    }
    Ok(())
}
///
/// Per-device capacity admission: largest performance-admitted rung
/// `<= perf_rows` whose projected PBS bytes, projection-prelude deficit,
/// missing FA Q16 bytes and 1 GiB headroom fit in current free device bytes.
/// The retained PBS is credited only on the grow-only route: a fitting
/// owner needs no allocation, and a smaller owner is freed before replacement.
/// The query follows model+KV mapping and request-state allocation.
/// Falls back to 512 when no enlarged rung fits.
fn memory_admitted_rung(
    gpu: &Gpu,
    config: &Qwen35Config,
    kv_cache: &llama::KvCache,
    perf_rows: usize,
    lean: bool,
    native_mq4: bool,
    cached: Option<&PrefillBatchScratch>,
) -> HipResult<usize> {
    if perf_rows <= WIDENED_COMMIT_ROWS {
        return Ok(WIDENED_COMMIT_ROWS.min(perf_rows));
    }
    let fp8_projection = gpu.arch == "gfx1201";
    if (if lean {
        lean_dense_prefill_allocation_bytes(config, perf_rows)
    } else {
        dense_prefill_allocation_bytes(config, perf_rows)
    })
    .is_none()
        || (fp8_projection && fp8_row_bytes_wide(config).is_none())
        || (!fp8_projection && iu4_row_bytes_wide(config).is_none())
    {
        return Ok(WIDENED_COMMIT_ROWS);
    }
    let (free_bytes, _) = gpu.hip.get_vram_info()?;
    let q_dim = config.n_heads.checked_mul(config.head_dim).unwrap_or(0);
    // Only a selectable gfx11 whole-chunk FA2 route needs more than the
    // incumbent 512-row Q16 scratch. Charge each candidate rung separately.
    let wide_q16_selectable = gpu.flags.gfx11_q8_fa2_wide
        && gpu.flags.gfx11_fa2_prefill
        && matches!(gpu.arch.as_str(), "gfx1100" | "gfx1151")
        && kv_cache.uses_vmm_backend()
        && kv_cache.quant_q8
        && !kv_cache.quant_fp8
        && kv_cache.tier_inputs().v_mode_bits == 8
        && config.n_heads == 24
        && config.n_kv_heads == 4
        && config.head_dim == 256;
    let live_fp8_x = gpu.scratch.mq4v2_fp8_x_scratch_bytes;
    let live_fp8_sums = gpu.scratch.mq4v2_fp8_half_sums_scratch_bytes;
    let live_fp8_scales = gpu.scratch.mq4v2_fp8_row_scales_scratch_bytes;
    let live_iu4 = gpu.scratch.int4_mmq_x_scratch_bytes;
    let fp8_groups = config.hidden_dim / 256;
    let iu4_bytes_per_row = iu4_row_bytes_wide(config).unwrap_or(usize::MAX);
    for &rung in WIDENED_RUNGS.iter().rev() {
        if rung > perf_rows {
            continue;
        }
        if rung <= WIDENED_COMMIT_ROWS {
            return Ok(WIDENED_COMMIT_ROWS);
        }
        let need_pbs = if lean {
            lean_dense_prefill_allocation_bytes(config, rung)
        } else {
            dense_prefill_allocation_bytes(config, rung)
        }
        .unwrap_or(usize::MAX);
        let projection_deficit = if native_mq4 {
            native_mq4_projection_deficit(
                config, rung,
                gpu.scratch.q8_1_mmq_x_scratch_bytes,
                gpu.scratch.fp16_x_scratch_bytes,
                gpu.scratch.ksplit_det_partials_bytes,
            ).unwrap_or(usize::MAX)
        } else if fp8_projection {
            let x_need = rung.checked_mul(config.hidden_dim).unwrap_or(usize::MAX);
            let sums_need = rung
                .checked_mul(fp8_groups)
                .and_then(|v| v.checked_mul(8))
                .unwrap_or(usize::MAX);
            let scales_need = rung.checked_mul(4).unwrap_or(usize::MAX);
            x_need
                .saturating_sub(live_fp8_x)
                .saturating_add(sums_need.saturating_sub(live_fp8_sums))
                .saturating_add(scales_need.saturating_sub(live_fp8_scales))
        } else {
            rung.checked_mul(iu4_bytes_per_row)
                .unwrap_or(usize::MAX)
                .saturating_sub(live_iu4)
        };
        let q16_rows = if wide_q16_selectable {
            rung
        } else {
            WIDENED_COMMIT_ROWS
        };
        let q16_need = q16_rows
            .checked_mul(q_dim)
            .and_then(|v| v.checked_mul(2))
            .unwrap_or(usize::MAX);
        let q16_missing = q16_need.saturating_sub(gpu.scratch.fa2_q16_scratch_bytes);
        let pbs_credit = cached
            .filter(|p| p.lean == lean)
            .and_then(|p| {
                if lean {
                    lean_dense_prefill_allocation_bytes(config, p.max_batch)
                } else {
                    dense_prefill_allocation_bytes(config, p.max_batch)
                }
            })
            .unwrap_or(0);
        let total = need_pbs
            .saturating_sub(pbs_credit)
            .saturating_add(projection_deficit)
            .saturating_add(q16_missing)
            .saturating_add(WIDENED_VRAM_HEADROOM_BYTES);
        if total <= free_bytes {
            return Ok(rung);
        }
    }
    Ok(WIDENED_COMMIT_ROWS)
}
///
/// Effective admitted ordinary chunk ceiling for this request: static
/// eligibility, highest performance-admitted rung, explicit requested
/// ceiling (`HIPFIRE_PREFILL_MAX_BATCH` over `prefill.chunk_rows`), and
/// per-device byte-based capacity.
///
/// Public only because hipfire-generate's ordinary AR caller needs the same
/// decision. Existing generic/TP/EP getters are unchanged; `pbs` limits the
/// ceiling when a caller owns a smaller staging buffer.
pub fn ordinary_prefill_chunk_limit(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    dn_state: &DeltaNetState,
    kv_cache: &llama::KvCache,
    pbs: Option<&PrefillBatchScratch>,
) -> HipResult<usize> {
    ordinary_prefill_chunk_limit_with_cache(gpu, weights, config, dn_state, kv_cache, pbs, None)
}

fn ordinary_prefill_chunk_limit_with_cache(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    dn_state: &DeltaNetState,
    kv_cache: &llama::KvCache,
    pbs: Option<&PrefillBatchScratch>,
    cached: Option<&PrefillBatchScratch>,
) -> HipResult<usize> {
    let legacy = prefill_max_batch_for_model(gpu, weights);
    let Some(perf) = ordinary_prefill_static_ceiling(gpu, weights, config, dn_state) else {
        return Ok(legacy);
    };
    if perf <= WIDENED_COMMIT_ROWS {
        return Ok(perf.min(legacy));
    }
    let lean = lean_pbs_requested() && lean_pbs_route(gpu, weights, config, perf);
    let mut admitted = memory_admitted_rung(
        gpu, config, kv_cache, perf, lean, native_mq4_widened_route(gpu, weights), cached,
    )?;
    if let Some(p) = pbs {
        admitted = admitted.min(p.max_batch);
    }
    // Admission only narrows the requested ceiling: perf already snaps the
    // request to rungs, and the sub-512 explicit path returns above.
    Ok(admitted.min(prefill_chunk_rows_requested(gpu).max(WIDENED_COMMIT_ROWS)))
}
///
/// Next direct-ordinary chunk under a widened ceiling. Keep all valid rows in
/// the largest admitted chunk so non-rung tails ride the padded multi-row GEMM
/// path instead of becoming a separate small-M request. Do not leave a
/// singleton after a full ceiling: shorten that chunk by one row so the
/// 512-row recurrent segments end in a 511-row partial and the final chunk has
/// two rows. The same rule maps a final in-ceiling 513 rows to 511 + 2.
pub(crate) fn next_exact_prefill_chunk_len(remaining: usize, ceiling: usize) -> Option<usize> {
    if remaining < MIN_BATCH || ceiling < MIN_BATCH {
        return None;
    }
    if ceiling <= WIDENED_COMMIT_ROWS {
        return next_prefill_chunk_len(remaining, ceiling);
    }
    let mut chunk = remaining.min(ceiling);
    if remaining - chunk == 1 && chunk > WIDENED_COMMIT_ROWS {
        chunk -= 1;
    } else if chunk > WIDENED_COMMIT_ROWS && chunk % WIDENED_COMMIT_ROWS == 1 {
        chunk -= MIN_BATCH;
    }
    Some(chunk)
}

/// Coalesce only complete legacy 512-row chunks for packed FFN. Preserve
/// the legacy irregular tail (including 511+2 for a singleton remainder),
/// so changing the ceiling does not change which rows use packed quantization.
fn next_packed_prefill_chunk_len(remaining: usize, ceiling: usize) -> Option<usize> {
    if ceiling <= WIDENED_COMMIT_ROWS {
        return next_prefill_chunk_len(remaining, ceiling);
    }
    let mut full = remaining / WIDENED_COMMIT_ROWS;
    if remaining % WIDENED_COMMIT_ROWS == 1 {
        full = full.saturating_sub(1);
    }
    let merged = full.min(ceiling / WIDENED_COMMIT_ROWS) * WIDENED_COMMIT_ROWS;
    if merged > 0 {
        Some(merged)
    } else {
        next_prefill_chunk_len(remaining, WIDENED_COMMIT_ROWS)
    }
}
///
/// Serve-caller outer chunk under a widened ceiling. Keep a short final region
/// attached to the preceding rows; the inner planner preserves exact 512-row
/// state/KV commits and handles the singleton exception.
pub fn ordinary_serve_prefill_chunk_len(remaining: usize, ceiling: usize) -> Option<usize> {
    if remaining == 0 || ceiling < MIN_BATCH {
        return None;
    }
    Some(remaining.min(ceiling))
}
///
/// First-prefill receipt: proves requested-vs-executed rows for the evidence
/// log. Emitted on the first chunked (multi-token) prefill of the process and
/// again whenever the admitted ceiling changes, so a later memory-admission
/// fallback is visible in the log.
fn emit_prefill_chunk_receipt(requested: usize, admitted: usize, commit_stride: Option<usize>) {
    static LAST_ADMITTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    if LAST_ADMITTED.swap(admitted, std::sync::atomic::Ordering::Relaxed) == admitted {
        return;
    }
    match commit_stride {
        Some(s) => {
            eprintln!("prefill_chunk: requested={requested} admitted={admitted} commit_stride={s}")
        }
        None => {
            eprintln!("prefill_chunk: requested={requested} admitted={admitted} commit_stride=none")
        }
    }
}
///
/// Never form a chunk larger than the configured/capped max, the PBS
/// staging owner, or (when present) the hidden ring's cap: its staging, so a
/// `HiddenStateRingBuffer` staged at 256 cannot receive a 384/512-row staging
/// write, or on the eager ring-seed route the ring itself, whose wider chunks
/// skip staging.
#[inline]
fn prefill_effective_chunk_batch(
    configured_max_batch: usize,
    pbs_max_batch: usize,
    hidden_rb_max_batch: Option<usize>,
) -> usize {
    let mut cap = configured_max_batch.min(pbs_max_batch);
    if let Some(hb) = hidden_rb_max_batch {
        cap = cap.min(hb);
    }
    cap
}

pub(crate) const MOE_GROUPED_BLOCK_M: usize = 16;

#[inline]
fn prefill_should_emit_last_token_logits(
    has_per_token_hidden_out: bool,
    needs_last_token_logits: bool,
) -> bool {
    !has_per_token_hidden_out || needs_last_token_logits
}

#[inline]
fn align_up_usize(x: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (x + align - 1) & !(align - 1)
}

#[inline]
pub(crate) fn moe_grouped_m_total_max(max_batch: usize, k_top: usize, n_exp: usize) -> usize {
    // Every grouped-GEMM tile consumes 16 sorted slots. The scatter kernel
    // initializes sentinel tile ids up to this bound, so the bound itself must
    // be tile-aligned; otherwise the final launched tile can read an
    // uninitialized expert id.
    align_up_usize(
        max_batch * k_top + n_exp * (MOE_GROUPED_BLOCK_M - 1),
        MOE_GROUPED_BLOCK_M,
    )
}

#[inline]
fn moe_grouped_m_total_bound(total_slots: usize, n_exp: usize) -> usize {
    // Actual grouped rows are sum_e align_up(count_e, BLOCK_M). Only experts
    // that receive at least one slot can contribute padding, so small verify
    // batches do not need to launch the full all-experts worst case.
    let live_expert_bound = total_slots.min(n_exp);
    align_up_usize(
        total_slots + live_expert_bound * (MOE_GROUPED_BLOCK_M - 1),
        MOE_GROUPED_BLOCK_M,
    )
}

/// Host-side helper: upload token ids and positions to a `PrefillBatchScratch`
/// via sync `memcpy_htod`. Call this BEFORE entering a hipGraph capture to
/// pre-populate `pbs.tokens` and `pbs.positions`, then pass `pre_uploaded:
/// true` (or use `forward_prefill_chunk_captured_safe`) so the forward
/// does not issue any additional uploads inside the captured region.
pub fn upload_prefill_batch_inputs(
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    tokens: &[u32],
    start_pos: usize,
) -> HipResult<()> {
    let n = tokens.len();
    let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let tokens_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
    let positions_host: Vec<i32> = (0..n).map(|i| (start_pos + i) as i32).collect();
    let positions_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
    Ok(())
}

/// Capture-friendly entry point that runs the batched forward against a
/// SINGLE chunk (`tokens.len() <= pbs.max_batch`), skipping the internal
/// token/position upload and assuming the caller has already populated
/// `pbs.tokens` / `pbs.positions` via `upload_prefill_batch_inputs`.
///
/// This exists so `hipStreamBeginCapture` can wrap the forward without
/// the per-call `memcpy_htod` sync operations (which would either error
/// under capture or bake stale host data into the captured graph nodes).
///
/// Callers still must handle `hidden_rb.commit_staging_to_ring(gpu, n)`
/// AFTER the forward returns (outside any captured region) to scatter
/// staging writes to the ring at the current head.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_single_chunk_captured(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
    hidden_rb: Option<&HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
) -> HipResult<()> {
    forward_prefill_batch_single_chunk_captured_opts(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        pbs,
        hidden_rb,
        per_token_hidden_out,
        gdn_tape,
        tree_verify,
        true,
        DflashFusionCtx::Off,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_single_chunk_captured_opts(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
    hidden_rb: Option<&HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    needs_last_token_logits: bool,
    fusion: DflashFusionCtx,
) -> HipResult<()> {
    let _ = fusion;
    let n = tokens.len();
    debug_assert!(
        n > 0 && n <= pbs.max_batch,
        "single_chunk_captured: n={} but pbs.max_batch={}",
        n,
        pbs.max_batch
    );
    let required_tokens = checked_kv_end(start_pos, n, "captured prefill")?;
    // This entry may already be inside graph capture, so it must only validate
    // capacity preflighted by its caller.
    kv_cache.require_mapped_capacity(required_tokens)?;

    // Defense-in-depth: this entry point bypasses the eligibility check
    // in `forward_prefill_batch_with_pbs`, so the caller is responsible
    // for ensuring the batched fast-path is valid. Two structural bypasses
    // could land here:
    //   1. MQ3-weighted model on an arch that lacks the gfx11 wave32 WMMA
    //      builtin (gfx12, gfx10, gfx906, gfx94x).
    //   2. MQ3 weights inside a MoE/A3B layer (DeltaNetMoe/FullAttnMoe) —
    //      the MoE batched branches dispatch through HFQ4-layout kernels
    //      and would memory-fault on the 104-vs-136 byte stride.
    // In production, `daemon.rs`'s DFlash refusal guard blocks both, but
    // dflash_spec_demo and other example callers go through ModelSlot::load
    // directly. We cross-check here so any caller is protected.
    let arch = gpu.arch.as_str();
    let mut mq3_in_dense = false;
    let mut mq3_in_moe = false;
    // The Lloyd dtype is treated identically to plain MQ3 in this guard:
    // both use 112-vs-104-byte stride that the MoE batched branches'
    // HFQ4-layout dispatch would corrupt, and both depend on the gfx11/12
    // WMMA family that other archs lack. Add Lloyd alongside MQ3 so the
    // refusal fires symmetrically and a future MQ3-Lloyd MoE model can't
    // silently land here without explicit MoE-Lloyd kernels.
    let is_mq3_any = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
    for lw in &weights.layers {
        match lw {
            LayerWeights::DeltaNet(l) => {
                if is_mq3_any(l.wqkv.gpu_dtype)
                    || is_mq3_any(l.wz.gpu_dtype)
                    || is_mq3_any(l.w_beta.gpu_dtype)
                    || is_mq3_any(l.w_alpha.gpu_dtype)
                    || is_mq3_any(l.wo.gpu_dtype)
                    || is_mq3_any(l.w_gate.gpu_dtype)
                    || is_mq3_any(l.w_up.gpu_dtype)
                    || is_mq3_any(l.w_down.gpu_dtype)
                {
                    mq3_in_dense = true;
                }
            }
            LayerWeights::FullAttn(l) => {
                if is_mq3_any(l.wq.gpu_dtype)
                    || is_mq3_any(l.wk.gpu_dtype)
                    || is_mq3_any(l.wv.gpu_dtype)
                    || is_mq3_any(l.wo.gpu_dtype)
                    || is_mq3_any(l.w_gate.gpu_dtype)
                    || is_mq3_any(l.w_up.gpu_dtype)
                    || is_mq3_any(l.w_down.gpu_dtype)
                {
                    mq3_in_dense = true;
                }
            }
            LayerWeights::DeltaNetMoe(l) => {
                if is_mq3_any(l.wqkv.gpu_dtype)
                    || is_mq3_any(l.wz.gpu_dtype)
                    || is_mq3_any(l.w_beta.gpu_dtype)
                    || is_mq3_any(l.w_alpha.gpu_dtype)
                    || is_mq3_any(l.wo.gpu_dtype)
                    || moe_ffn_has_mq3_structural(&l.ffn)
                    || moe_ffn_has_mq3_experts_uniform(&l.ffn)
                {
                    mq3_in_moe = true;
                }
            }
            LayerWeights::FullAttnMoe(l) => {
                if is_mq3_any(l.wq.gpu_dtype)
                    || is_mq3_any(l.wk.gpu_dtype)
                    || is_mq3_any(l.wv.gpu_dtype)
                    || is_mq3_any(l.wo.gpu_dtype)
                    || moe_ffn_has_mq3_structural(&l.ffn)
                    || moe_ffn_has_mq3_experts_uniform(&l.ffn)
                {
                    mq3_in_moe = true;
                }
            }
        }
    }
    // ANTIBLEED admit-vs-select fix: this guard rejects MQ3-in-dense when the
    // arch lacks the WMMA builtin. The old ad-hoc string list OMITTED gfx1103
    // (Phoenix APU) and gfx1152, yet both ARE wave32-WMMA archs (is_rdna3) and
    // are ADMITTED by is_batchable_la's mq3_uniform_with_wmma — so a gfx1103 /
    // gfx1152 box would be wrongly rejected here. Derive from the has_wmma
    // capability molecule instead (rdna3 incl 1103/1152, + rdna4), matching the
    // sibling `arch_has_wmma = gpu.arch_caps.has_wmma()` in forward_prefill_chunk.
    let arch_has_wmma = gpu.arch_caps.has_wmma();
    if mq3_in_moe {
        return Err(hip_bridge::HipError::new(
            0,
            "forward_prefill_batch_single_chunk_captured: model has MQ3G256 / \
             MQ3G256Lloyd weights inside a MoE/A3B layer (DeltaNetMoe or \
             FullAttnMoe). The MoE batched prefill branches dispatch through \
             HFQ4-layout kernels and would memory-fault on the 104/112-vs-136 \
             byte stride. Use an MQ4 quantization for MoE/A3B targets, or wait \
             for the MQ3 MoE branches to land.",
        ));
    }
    if mq3_in_dense && !arch_has_wmma {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "forward_prefill_batch_single_chunk_captured: model contains MQ3G256 \
             weights but arch {arch} lacks the gfx11 wave32 WMMA builtin. The MQ3 \
             prefill kernels (gemm_*_hfq3g256_wmma) only compile on the wave32-WMMA \
             archs (rdna3: gfx1100/1101/1102/1103/1150/1151/1152, + rdna4 gfx12). \
             Caller must use the non-captured \
             forward_prefill_batch path (which falls back to per-token \
             forward_scratch on this arch). gfx12 K4 variant for MQ3 is \
             a planned follow-up."
            ),
        ));
    }

    // Q8 KV at any physical_cap is capture-safe: forward_prefill_chunk
    // dispatches through the unified DispatchCtx → AttnQ8_0KvBatchedMasked,
    // which routes max_ctx_len > 8192 to the tiled attention_flash_q8_0_tile_batched
    // (O(1) LDS, no per-position malloc). The former physical_cap > 15000 guard
    // predated that crossover (landed 2026-06-09) and is now obsolete.
    #[cfg(feature = "moe-oracle")]
    crate::qwen35::oracle::set_prefill_start(start_pos).map_err(|e| HipError::new(0, &e))?;
    forward_prefill_chunk(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        pbs,
        hidden_rb,
        per_token_hidden_out.map(|t| (t, 0, HiddenCapture::Verify)),
        gdn_tape,
        0,
        tree_verify,
        true, // pre_uploaded: caller must have run upload_prefill_batch_inputs
        None, // band: full-stack single-GPU path
        None, // mask_override: captured-prefill caller does not use the MTP probe hook
        needs_last_token_logits,
        None, // max_layer: single-chunk captured path always runs the full stack
        None, // routed_out: non-EP single-GPU path
        fusion,
        None, // commit_stride: captured path keeps legacy cadence
    )
}

/// Batched prefill entry point. Chunk ceiling is arch-aware via
/// [`prefill_max_batch`] (512 on exact gfx1100, 384 on exact gfx1201). Use
/// [`forward_prefill_batch_capped`] when internal owned-PBS chunks must stay
/// at a hard staging budget (e.g. eviction ≤ 256), and pass a hidden ring
/// whose `max_batch` will also bound actual chunks.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_rb: Option<&mut HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
) -> HipResult<()> {
    // Widened ordinary route: omit the implicit ≤512 legacy cache so a fully
    // admitted request allocates one larger owned PBS in the inner entry.
    // Explicit small ceilings keep the cache; capped/explicit-PBS callers
    // never bypass (see `forward_prefill_batch_capped` and the inner entry).
    let widen = ordinary_prefill_static_ceiling(gpu, weights, config, dn_state)
        .is_some_and(|c| c > WIDENED_COMMIT_ROWS);
    let pbs_for_call = match scratch.prefill_batch.as_ref() {
        Some(_) if widen => None,
        other => other,
    };
    forward_prefill_batch_with_pbs(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        hidden_rb,
        per_token_hidden_out,
        gdn_tape,
        tree_verify,
        pbs_for_call,
        None, // mask_override: MTP probe is the only consumer; default callers don't override
        None, // max_layer: pflash uses this; non-pflash default is full stack
    )
}

/// Ordinary prompt prefill that also writes every row's post-output-norm
/// hidden state into `hidden_out` (`tokens.len() × dim` f32, row `i` = token
/// `i`), for a draft head that consumes the prompt's hidden rows (MTP).
///
/// Chunk planning, widened admission, the GDN chunk scan and the dispatch
/// workload are exactly those of [`forward_prefill_batch`] with no capture,
/// so KV, DeltaNet state and `scratch.logits` (last-token logits) match a
/// plain AR prefill of the same call. Verify callers that capture hidden
/// rows keep [`forward_prefill_batch`]'s `per_token_hidden_out`, which
/// selects the speculative-verify kernels.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_capture_hidden(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_out: &GpuTensor,
) -> HipResult<()> {
    // Same implicit-cache omission as `forward_prefill_batch`.
    let widen = ordinary_prefill_static_ceiling(gpu, weights, config, dn_state)
        .is_some_and(|c| c > WIDENED_COMMIT_ROWS);
    let pbs_for_call = match scratch.prefill_batch.as_ref() {
        Some(_) if widen => None,
        other => other,
    };
    forward_prefill_batch_with_pbs_opts_inner(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        None,
        Some(hidden_out),
        HiddenCapture::PromptFill,
        None,
        None,
        pbs_for_call,
        None,
        None,
        true,
        None,
        DflashFusionCtx::Off,
    )
}

/// Like [`forward_prefill_batch`], but forces the configured chunk ceiling
/// through `prefill_max_batch(gpu).min(max_batch_cap)` before owned-PBS
/// planning and chunking.
///
/// Use when the caller keeps a larger outer window (eviction cadence,
/// adaptive maybe_evict) but must not allocate or form internal chunks above
/// a hard staging budget (typically [`PREFILL_MAX_BATCH`] = 256). Preserves
/// ordinary defaults: `scratch.prefill_batch`, no mask override, full stack
/// (`max_layer = None`), last-token logits enabled.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_capped(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_rb: Option<&mut HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    max_batch_cap: usize,
) -> HipResult<()> {
    forward_prefill_batch_with_pbs_opts_inner(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        hidden_rb,
        per_token_hidden_out,
        HiddenCapture::Verify,
        gdn_tape,
        tree_verify,
        scratch.prefill_batch.as_ref(),
        None,
        None,
        true,
        Some(max_batch_cap),
        DflashFusionCtx::Off,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_with_pbs(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_rb: Option<&mut HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    pbs_in: Option<&PrefillBatchScratch>,
    mask_override: Option<MaskEmbedOverride<'_>>,
    max_layer: Option<usize>,
) -> HipResult<()> {
    forward_prefill_batch_with_pbs_opts(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        hidden_rb,
        per_token_hidden_out,
        gdn_tape,
        tree_verify,
        pbs_in,
        mask_override,
        max_layer,
        true, // preserve legacy post-condition: scratch.logits is last-token logits
        DflashFusionCtx::Off,
    )
}

/// Like `forward_prefill_batch`, but accepts a caller-owned `PrefillBatchScratch`
/// so the ~25 per-cycle tensor allocations can be amortized across many calls.
///
/// `pbs = None` allocates and frees a right-sized scratch per call;
/// `pbs = Some(&pbs)` reuses the provided scratch. Chunk size is the minimum of
/// the configured/arch max ([`prefill_max_batch`]), `pbs.max_batch`, and the
/// ring's staging `hidden_rb.max_batch` when a hidden ring is supplied — never
/// larger than any staging owner. The exception is an eager ring-only prompt
/// seed (no per-token hidden, tape or tree), whose chunks wider than staging
/// write straight to the ring. Callers driving DFlash verify should size `pbs`
/// (and hidden staging) to the maximum block they will request so everything
/// fits in one chunk, or accept multi-chunk commits.
///
/// `needs_last_token_logits = false` is only for callers that pass
/// `per_token_hidden_out` and compute their own logits from those hidden rows.
/// The default wrapper keeps this true to protect existing callers that rely on
/// `scratch.logits` being populated with the last token's logits.
///
/// For an explicit hard ceiling on owned-PBS planning (eviction path), use
/// [`forward_prefill_batch_capped`] instead of threading a private cap here.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_with_pbs_opts(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_rb: Option<&mut HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    pbs_in: Option<&PrefillBatchScratch>,
    mask_override: Option<MaskEmbedOverride<'_>>,
    max_layer: Option<usize>,
    needs_last_token_logits: bool,
    fusion: DflashFusionCtx,
) -> HipResult<()> {
    forward_prefill_batch_with_pbs_opts_inner(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        hidden_rb,
        per_token_hidden_out,
        HiddenCapture::Verify,
        gdn_tape,
        tree_verify,
        pbs_in,
        mask_override,
        max_layer,
        needs_last_token_logits,
        None,
        fusion,
    )
}

#[allow(clippy::too_many_arguments)]
fn forward_prefill_batch_with_pbs_opts_inner(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    mut hidden_rb: Option<&mut HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    hidden_capture: HiddenCapture,
    mut gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    pbs_in: Option<&PrefillBatchScratch>,
    mask_override: Option<MaskEmbedOverride<'_>>,
    max_layer: Option<usize>,
    needs_last_token_logits: bool,
    max_batch_cap: Option<usize>,
    fusion: DflashFusionCtx,
) -> HipResult<()> {
    // Plain single-token AR decode? Only then is the per-token `forward_scratch`
    // call below eligible for the AR-forward hipGraph (capture/replay). Any spec
    // marker (tree_verify / gdn_tape / per-token-hidden extraction / hidden ring)
    // or a multi-token batch means this is prefill or a spec/MTP verify forward,
    // which must NOT replay the plain-AR graph. See `forward_scratch`'s
    // `ar_graph_eligible` one-shot signal.
    let plain_ar_graph_eligible = tree_verify.is_none()
        && gdn_tape.is_none()
        && per_token_hidden_out.is_none()
        && hidden_rb.is_none()
        && tokens.len() == 1;
    // Upper bound on the PrefillBatchScratch — large prompts get split
    // into chunks of this size and processed in a loop.
    //
    // Tuning note: each extra chunk pays full dispatch-overhead for the LA
    // preamble (rmsnorm, rotate, 4-way fused GEMM) and FFN (gate_up + down).
    // The default is arch-aware via `prefill_max_batch`: measured 512 on
    // exact gfx1100, measured 384 on exact gfx1201 (Qwen3.8 sweet spots),
    // conservative `PREFILL_MAX_BATCH` (256) elsewhere. Override with
    // `HIPFIRE_PREFILL_MAX_BATCH>=2`. An explicit `max_batch_cap` (from
    // [`forward_prefill_batch_capped`]) mins on top. Actual chunks are further
    // limited by PBS and hidden-ring staging so a 256-row ring can never
    // receive a larger arch-default write. 256 costs ~80 MB of scratch on 9B
    // vs 20 MB at 64 — trivial on modern cards — and drops chunk count for
    // pp2048 from 32 → 8. The inner gated_delta_net_q8_batch_seq loop is
    // still sequential per token, so the per-chunk DeltaNet cost is linear
    // in N either way; raising the batch just amortizes the NON-DeltaNet
    // kernels more.
    let max_batch: usize = {
        let configured = prefill_max_batch_for_model(gpu, weights);
        match max_batch_cap {
            Some(cap) => configured.min(cap),
            None => configured,
        }
    };

    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }
    let required_tokens = checked_kv_end(start_pos, n, "forward_prefill_batch")?;
    release_widened_pbs_for_kv_growth(gpu, kv_cache, config, scratch, required_tokens)?;
    kv_cache.ensure_mapped_capacity(gpu, required_tokens)?;

    // Cross-path safety: refuse MQ3 / MQ3-Lloyd weights inside any MoE
    // layer (attention OR FFN), mirroring the captured-path guard at
    // `forward_prefill_batch_single_chunk_captured` (line 3367+). Without
    // this, the eligibility check below would admit a hybrid model with
    // (e.g.) MQ3 attention + MQ4 MoE FFN onto the batched path, where the
    // MoE-batched LA/FA bodies would misroute: the QKV matcher drops MQ3
    // and the wo path is hardcoded to `gemm_hfq4g256_residual` regardless
    // of `layer.wo.gpu_dtype`. The result is a 104/112 vs 136 byte stride
    // mismatch and silent-corruption fluent-looking output. Issue #179
    // documents the matcher half of this; the wo half was uncovered in
    // review. Wiring both correctly (plus Lloyd) is tracked separately
    // (see followup issue) — until then we hard-error here so all three
    // entry points (daemon-DFlash setup, captured prefill, non-captured
    // prefill) reject MQ3+MoE consistently.
    let is_mq3_any = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
    // Routed-expert clause uses the NARROWED predicate: a uniform-per-projection
    // codebook pair (MQ2-Lloyd gate_up / MQ3-Lloyd down) now has its own
    // grouped-GEMM arms, so the 112-vs-136 stride hazard the refusal exists for
    // does not apply to it. Everything else — MQ3 in the attention projections,
    // in the router, or in the shared expert — still refuses, because those
    // paths really are hardcoded to the HFQ4 layout. The captured entry point
    // keeps the unnarrowed predicate (see that predicate's doc comment).
    let codebook_admit = codebook_batched_admit_enabled(gpu.arch.as_str());
    let mq3_in_moe = weights.layers.iter().any(|lw| match lw {
        LayerWeights::DeltaNetMoe(l) => {
            is_mq3_any(l.wqkv.gpu_dtype)
                || is_mq3_any(l.wz.gpu_dtype)
                || is_mq3_any(l.w_beta.gpu_dtype)
                || is_mq3_any(l.w_alpha.gpu_dtype)
                || is_mq3_any(l.wo.gpu_dtype)
                || moe_ffn_has_mq3_structural(&l.ffn)
                || moe_ffn_has_unsupported_mq3_experts_uniform(&l.ffn, codebook_admit)
        }
        LayerWeights::FullAttnMoe(l) => {
            is_mq3_any(l.wq.gpu_dtype)
                || is_mq3_any(l.wk.gpu_dtype)
                || is_mq3_any(l.wv.gpu_dtype)
                || is_mq3_any(l.wo.gpu_dtype)
                || moe_ffn_has_mq3_structural(&l.ffn)
                || moe_ffn_has_unsupported_mq3_experts_uniform(&l.ffn, codebook_admit)
        }
        _ => false,
    });
    // NOTE: the refusal this predicate drives is deliberately deferred until
    // after `eligible` is computed below — see the `mq3_in_moe && (eligible ||
    // gdn_tape.is_some())` guard. Refusing here would pre-empt the per-token
    // fallback, which handles MQ3/Lloyd correctly.

    // Tree-verify mode sanity checks — the downstream path can't silently
    // fall back to per-token FA (that's always causal and would ignore the
    // tree mask), and the positions/bias shapes must match the token count.
    if let Some(ctx) = tree_verify.as_ref() {
        assert_eq!(
            ctx.positions.len(),
            n,
            "TreeVerifyCtx.positions length {} must equal tokens.len() {}",
            ctx.positions.len(),
            n,
        );
        assert_eq!(
            ctx.attn_bias.numel(),
            n * n,
            "TreeVerifyCtx.attn_bias must be [{} × {}] f32 ({}), got numel {}",
            n,
            n,
            n * n,
            ctx.attn_bias.numel(),
        );
    }

    // Fast path requires (a) every LA layer's weights to be either MQ4G256
    // or HFQ4G256 (the batched GEMM kernels are dtype-agnostic but the LA
    // preamble's rmsnorm+rotate and SwiGLU+rotate kernels differ per dtype),
    // and (b) Q8 S-state for the GDN recurrence. Mixed-dtype layers are
    // allowed; each layer is routed to its own path. HFQ6/others fall back.
    let arch = gpu.arch.as_str();
    // Whether the tape-capturing batched (PBS) path runs for this call — the
    // single source of truth shared with spec-decode callers that later replay a
    // captured GDN tape. On `false` the forward drops to the tape-less per-token
    // loop below, leaving any passed tape stale (see `prefill_batch_pbs_eligible`).
    let moe_router_logits_present = pbs_in
        .map(|p| p.moe_router_logits_batch.is_some())
        .unwrap_or(true);
    let eligible = prefill_batch_pbs_eligible(
        weights,
        config,
        dn_state,
        n,
        arch,
        moe_router_logits_present,
    );
    // F4 guard: reject batched prefill when KV tier has no batched keys.
    // F32 KV has only BatchEq(1) → MissingImpl at resolve. asym2 + tree-verify
    // has no _batched_masked variant → UnsupportedTreeTier. Force per-token
    // fallback for these cases.
    let kv_f32 = !kv_cache.quantized && !kv_cache.quant_q8 && !kv_cache.quant_hfq4;
    let kv_asym2_tree = kv_cache.quant_asym2 && tree_verify.is_some();
    let eligible = eligible && !kv_f32 && !kv_asym2_tree;

    // MQ3-in-MoE refusal (predicate computed above). This protects the batched
    // MoE LA/FA bodies: their QKV matcher drops MQ3 and the wo path is
    // hardcoded to `gemm_hfq4g256_residual`, so an MQ3/Lloyd model would take a
    // 104/112-vs-136 byte stride mismatch and emit fluent-looking corruption.
    //
    // It is gated on `eligible` because when the batched path does NOT run,
    // those bodies never execute: the `!eligible` branch below is a per-token
    // `forward_scratch` loop, byte-identical to decode, which dispatches every
    // weight by its own dtype and supports MQ3/Lloyd fully. Refusing before the
    // eligibility check pre-empted that correct path — routed codebook models
    // (MQ2/MQ3 Lloyd/GL experts) are already inadmissible to the batched MoE
    // bodies via `moe_ffn_batched_admissible_for_dtypes`, so the refusal was
    // protecting nothing and only blocked a working per-token prefill.
    //
    // The `gdn_tape.is_some()` clause is load-bearing: the `!eligible` fallback
    // leaves a passed GDN tape untouched/stale rather than erroring, so a
    // spec-decode caller must still get the loud refusal instead of a silent
    // stale-tape DeltaNet corruption.
    //
    // The sibling guard in `forward_prefill_batch_single_chunk_captured_opts`
    // is intentionally NOT gated this way — that entry point has no eligibility
    // check and no per-token fallback, so its refusal is the only protection.
    if mq3_in_moe && (eligible || gdn_tape.is_some()) {
        return Err(hip_bridge::HipError::new(
            0,
            "forward_prefill_batch: model has MQ3G256 / MQ3G256Lloyd weights \
             inside a MoE/A3B layer (DeltaNetMoe or FullAttnMoe). The MoE \
             batched prefill branches dispatch through HFQ4-layout kernels \
             (QKV matcher drops MQ3; wo path is hardcoded MQ4) and would \
             produce silent corruption from the 104/112-vs-136 byte stride \
             mismatch. Use an MQ4 quantization for MoE/A3B targets, or wait \
             for the MQ3 MoE branches to land (see followup issue).",
        ));
    }

    if !eligible {
        assert!(
            tree_verify.is_none(),
            "tree-verify mode requires the batched-FA-eligible prefill path; \
             kv quant + FA weight dtypes do not match on this model",
        );
        // mask_override has nowhere to land on the per-token forward_scratch
        // fallback (it operates on `scratch.x`, not the batched `pbs.x_batch`,
        // and there's no shared "post-embed, pre-layer" hook). The MTP probe
        // is the only consumer today and runs on MQ4-quantized models that
        // always satisfy `eligible`, so hard-error rather than silently
        // ignoring the override.
        assert!(
            mask_override.is_none(),
            "MaskEmbedOverride requires the batched prefill path, but this \
             model fell through to the per-token fallback (likely non-MQ4 \
             weights, dn_state quant != Q8, or HIPFIRE_PREFILL_BATCHED=0).",
        );
        // Fallback: per-token loop, byte-identical to decode. If hidden
        // extraction is requested, use the with_hidden variant so the ring
        // buffer still gets populated correctly (each call advances head by 1).
        // When per-token hidden output is also requested, extract post-norm
        // hidden row-by-row into the caller's buffer.
        let dim = config.dim;
        for (i, &tok) in tokens.iter().enumerate() {
            if let Some(rb) = hidden_rb.as_mut() {
                forward_scratch_with_hidden(
                    gpu,
                    weights,
                    config,
                    tok,
                    start_pos + i,
                    kv_cache,
                    dn_state,
                    scratch,
                    rb,
                )?;
            } else {
                // One-shot: mark this forward AR-graph-eligible iff it's plain
                // single-token decode (consumed inside forward_scratch).
                gpu.graphs.ar_graph_eligible = plain_ar_graph_eligible;
                forward_scratch(
                    gpu,
                    weights,
                    config,
                    tok,
                    start_pos + i,
                    kv_cache,
                    dn_state,
                    scratch,
                )?;
            }
            if let Some(dst) = per_token_hidden_out {
                // scratch.tmp holds post-output-norm hidden after
                // forward_scratch_{with_hidden,layers} — it's the same buffer
                // lm_head reads from. Copy into the caller's output.
                gpu.hip
                    .memcpy_dtod_at(&dst.buf, i * dim * 4, &scratch.tmp.buf, 0, dim * 4)?;
            }
        }
        return Ok(());
    }

    // Tree-verify mode runs as a single chunk (tree is small, O(16) nodes);
    // chunk splitting would require slicing the mask by chunk rows which
    // is extra work for a case we don't need.
    if tree_verify.is_some() {
        assert!(
            n <= max_batch,
            "tree-verify tokens {} exceeds max_batch {}; tree budget must fit",
            n,
            max_batch,
        );
    }
    // A ring-only forward (no per-token hidden rows, tape or tree) is a DFlash
    // prompt seed. Eager, it rides the ordinary route below exactly as AR
    // prefills the same tokens; chunks wider than ring staging write their
    // hidden rows straight to the ring (`HiddenStateRingBuffer::writes_direct`).
    // Verify and captured/recorded forwards keep the staged ring route.
    let ring_seed = match hidden_rb.as_deref() {
        Some(rb) => {
            per_token_hidden_out.is_none()
                && gdn_tape.is_none()
                && tree_verify.is_none()
                && rb.rides_ordinary_prefill(gpu)
        }
        None => false,
    };
    // Widened ordinary admission. Decided here, after required KV mapping, so
    // the per-device byte admission charges mapped KV. Only the whole-stack
    // sequential ordinary entry with no caller PBS/cap can widen: captured,
    // TP/EP (separate callers), tree/tape, staged hidden-ring, band-limited,
    // fused, or caller-capped requests resolve to the legacy path before any
    // wider allocation or launch. The ordinary wrappers omit their *implicit*
    // ≤512 legacy cache above when a wider ceiling is statically admitted, so
    // `pbs_in.is_none()` here means the owner decision is ours; an explicit
    // smaller caller PBS always wins and keeps legacy cadence.
    let wide_candidate = pbs_in.is_none()
        && max_batch_cap.is_none()
        && tree_verify.is_none()
        && gdn_tape.is_none()
        && (hidden_rb.is_none() || ring_seed)
        && max_layer.is_none()
        && matches!(fusion, DflashFusionCtx::Off)
        && !gpu.graphs.capture_mode
        && !gpu.replay.is_recording();
    let limit = if wide_candidate {
        let cached = scratch.widened_prefill_batch.borrow();
        ordinary_prefill_chunk_limit_with_cache(
            gpu,
            weights,
            config,
            dn_state,
            kv_cache,
            None,
            cached.as_ref().filter(|_| widened_pbs_grow_only(gpu)),
        )?
    } else {
        max_batch
    };
    let wide_admitted = wide_candidate && limit > WIDENED_COMMIT_ROWS;
    emit_prefill_chunk_receipt(
        if wide_admitted {
            prefill_chunk_rows_requested(gpu)
        } else {
            explicit_prefill_max_batch().unwrap_or(max_batch)
        },
        limit,
        wide_admitted.then_some(WIDENED_COMMIT_ROWS),
    );
    // Allocate the batch scratch once per call (or reuse a caller-owned one).
    // When `pbs_in` is Some, we neither allocate nor free — the caller retains
    // ownership across DFlash cycles to avoid ~25 per-cycle tensor alloc/free
    // pairs on the hot verify path. When None, size the allocation to this
    // call's largest possible chunk and allocate the DeltaNet S-state tape only
    // for tree verify.
    // Plain prefill never consumes that tape, and short prompts do not pay
    // the full configured footprint. Actual chunk length is
    // min(configured/capped max, pbs.max_batch, hidden_rb staging) so no
    // write exceeds a staging owner.
    let mut own_pbs: Option<PrefillBatchScratch> = None;
    // True when `own_pbs` was taken from the retained widened cache below
    // (not freshly allocated): it is moved back on return instead of freed.
    let mut widened_from_cache = false;
    // F2 pair scratch (second 512-row PBS + 1024-row FA staging), allocated
    // lazily on the first pair and shared by all pairs of this call.
    let mut pair_scratch: Option<(PrefillBatchScratch, FaPairStage)> = None;
    let result = (|| -> HipResult<()> {
        // On the fully admitted path the implicit undersized cache was already
        // omitted by the wrapper, so `None` here allocates one larger
        // tape-free owned PBS sized to the actual largest chunk — short calls
        // never pay the full ceiling. All small/excluded paths keep their
        // prior capacity and cadence.
        let wide = wide_candidate && limit > WIDENED_COMMIT_ROWS;
        let pbs: &PrefillBatchScratch = match pbs_in {
            Some(p) => p,
            None if wide => {
                let owned_rows = limit.min(n).max(MIN_BATCH);
                // Retained widened PBS: reuse across requests when it fits,
                // else replace. Saves the ~25 ms per-request alloc on
                // 6k-token prefills. Bit-identical: every PBS tensor is
                // overwritten before it is read (same reuse contract as the
                // legacy `prefill_batch` cache).
                let mut cached = scratch.widened_prefill_batch.borrow_mut();
                let lean = lean_pbs_requested() && lean_pbs_route(gpu, weights, config, owned_rows);
                if !cached
                    .as_ref()
                    .is_some_and(|p| p.max_batch >= owned_rows && p.lean == lean)
                {
                    if let Some(old) = cached.take() {
                        let _ = old.free_gpu(gpu);
                    }
                    *cached = Some(if lean {
                        PrefillBatchScratch::new_opt_lean(gpu, config, owned_rows)?
                    } else {
                        PrefillBatchScratch::new_opt(gpu, config, owned_rows, false)?
                    });
                }
                own_pbs = cached.take();
                widened_from_cache = true;
                own_pbs.as_ref().unwrap()
            }
            None => {
                // `limit == max_batch` on every legacy path; on the admitted
                // but memory-narrowed path it sizes the owned scratch to the
                // executed 512 cadence instead of the unadmitted request.
                let (owned_max_batch, cap_gdn_tape) =
                    owned_prefill_scratch_plan(n, max_batch.min(limit), tree_verify.is_some());
                own_pbs = Some(PrefillBatchScratch::new_opt(
                    gpu,
                    config,
                    owned_max_batch,
                    cap_gdn_tape,
                )?);
                own_pbs.as_ref().unwrap()
            }
        };
        // A ring seed chunks exactly as AR does; chunks wider than staging
        // land straight in the ring, so the ring (not its staging) bounds
        // them. Every other ring caller stays capped at staging.
        let chunk_batch = prefill_effective_chunk_batch(
            limit,
            pbs.max_batch,
            hidden_rb.as_ref().map(|rb| {
                if ring_seed {
                    rb.max_positions.max(rb.max_batch)
                } else {
                    rb.max_batch
                }
            }),
        );
        // F2 pair envelope, loop-invariant half: ordinary sequential prefill
        // with FA layers batch-admissible, no tree/tape/max_layer machinery,
        // no hidden ring (its staging commit is chunk-sequential), plain
        // DFlash fusion. Per-pair checks (exact 512+512, eager, shapes, KV
        // tier, max_ctx window) happen at each step below.
        let fa_pair_common_admitted = tree_verify.is_none()
            && gdn_tape.is_none()
            && max_layer.is_none()
            && hidden_rb.is_none()
            && matches!(fusion, DflashFusionCtx::Off)
            && (kv_cache.quant_q8
                || kv_cache.quant_asym4
                || kv_cache.quant_asym3
                || kv_cache.quant_asym2)
            && weights.layers.iter().all(|lw| match lw {
                LayerWeights::FullAttn(_) | LayerWeights::FullAttnMoe(_) => {
                    qwen35_layer_batch_admissible(lw, config, gpu.arch.as_str()).is_ok()
                }
                _ => true,
            });
        let mut chunk_start = 0usize;
        while chunk_start < n {
            let remaining = n - chunk_start;
            // Widened grouping keeps an odd tail attached to the largest
            // admitted chunk so its GEMMs use the padded multi-row route.
            let chunk_n = if wide
                && gpu.arch == "gfx1100"
                && gpu.flags.packed_mq4_prefill
                && dense_layers_have_projection_dtype(weights, DType::MQ4G256)
            {
                next_packed_prefill_chunk_len(remaining, chunk_batch)
            } else if wide {
                next_exact_prefill_chunk_len(remaining, chunk_batch)
            } else {
                next_prefill_chunk_len(remaining, chunk_batch)
            }
            .ok_or_else(|| {
                hip_bridge::HipError::new(
                    0,
                    "forward_prefill_batch: chunk plan cannot satisfy the two-token minimum",
                )
            })?;
            if pbs.lean {
                if !matches!(fusion, DflashFusionCtx::Off) || tree_verify.is_some() {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "lean PBS belongs to ordinary prefill; verify route requires full scratch",
                    ));
                }
                if chunk_n > 64 && !lean_pbs_route(gpu, weights, config, chunk_n) {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "lean PBS fused producer unavailable beyond 64 fallback rows",
                    ));
                }
                for (name, capacity) in [
                    ("x_norm", pbs.x_norm_batch.numel() / config.dim),
                    (
                        "dn_normed",
                        pbs.dn_normed_batch.numel()
                            / (config.linear_num_value_heads * config.linear_value_head_dim),
                    ),
                    (
                        "ffn_hidden",
                        pbs.ffn_hidden_batch.numel() / config.hidden_dim,
                    ),
                    (
                        "dn_normed_rot",
                        pbs.dn_normed_rot_batch.numel()
                            / (config.linear_num_value_heads * config.linear_value_head_dim),
                    ),
                    (
                        "fa_attn_out_rot",
                        pbs.fa_attn_out_rot_batch.numel() / (config.n_heads * config.head_dim),
                    ),
                ] {
                    if chunk_n <= 64 && chunk_n > capacity {
                        return Err(hip_bridge::HipError::new(
                            0,
                            &format!(
                                "lean PBS {name} fallback capacity {capacity} < {chunk_n} rows"
                            ),
                        ));
                    }
                }
            }
            // F2 pair peek is only relevant to the legacy 512-row schedule;
            // widened chunks carry their tail directly.
            let pair_b = if chunk_n == FA_PAIR_ROWS
                && fa_pair_common_admitted
                && remaining - chunk_n >= MIN_BATCH
            {
                next_prefill_chunk_len(remaining - chunk_n, chunk_batch)
            } else {
                None
            };
            if pair_b == Some(FA_PAIR_ROWS)
                && fa_pair_merge_admitted(
                    chunk_n,
                    FA_PAIR_ROWS,
                    gpu.arch.as_str(),
                    gpu.graphs.capture_mode,
                    gpu.replay.is_recording(),
                )
                && fa_pair_env_admitted(
                    gpu,
                    config,
                    kv_cache,
                    start_pos + chunk_start + 2 * FA_PAIR_ROWS,
                    fusion,
                )
            {
                // Lazily allocate the second PBS + pair staging; both are
                // shared by the rest of this call. Transactional: the stage
                // is freed when the PBS allocation fails, so no leak.
                if pair_scratch.is_none() {
                    let stage = FaPairStage::alloc(gpu, config)?;
                    match PrefillBatchScratch::new_opt(gpu, config, FA_PAIR_ROWS, false) {
                        Ok(pbs2) => pair_scratch = Some((pbs2, stage)),
                        Err(e) => {
                            let _ = stage.free_gpu(gpu);
                            return Err(e);
                        }
                    }
                }
                let (pbs2, stage) = pair_scratch.as_ref().unwrap();
                forward_prefill_chunk_pair(
                    gpu,
                    weights,
                    config,
                    &tokens[chunk_start..chunk_start + FA_PAIR_ROWS],
                    &tokens[chunk_start + FA_PAIR_ROWS..chunk_start + 2 * FA_PAIR_ROWS],
                    start_pos,
                    chunk_start,
                    kv_cache,
                    dn_state,
                    scratch,
                    pbs,
                    pbs2,
                    stage,
                    per_token_hidden_out.map(|t| (t, hidden_capture)),
                    mask_override,
                    needs_last_token_logits,
                    fusion,
                )?;
                chunk_start += 2 * FA_PAIR_ROWS;
                continue;
            }
            let chunk_end = chunk_start + chunk_n;
            let chunk = &tokens[chunk_start..chunk_end];
            // The chunk only reads the ring buffer's head/dims to place its
            // writes. We advance the head AFTER the chunk returns, here in
            // the caller, to keep the mutable borrow scope tight.
            let pth_slot = per_token_hidden_out.map(|t| (t, chunk_start, hidden_capture));
            // Reborrow the tape for this chunk so we keep the outer mut
            // after the chunk returns.
            let tape_for_chunk: Option<&mut crate::speculative::GdnTape> =
                gdn_tape.as_mut().map(|t| &mut **t);
            // Tree-verify was asserted to fit in one chunk above, so passing
            // the whole ctx through unconditionally is safe.
            let tv_for_chunk = tree_verify.as_ref().copied();
            // Apply mask_override only to the chunk that actually contains
            // its target slot, and rebase the slot index to chunk-local
            // coordinates. Out-of-range slots panic (caller error).
            let mo_for_chunk = mask_override.and_then(|ovr| {
                if ovr.slot >= chunk_start && ovr.slot < chunk_end {
                    Some(MaskEmbedOverride {
                        slot: ovr.slot - chunk_start,
                        embed: ovr.embed,
                    })
                } else {
                    None
                }
            });
            // Sanity: if caller provided an override, it MUST land in some
            // chunk. Detect "fell off the end" at the last chunk boundary.
            if mask_override.is_some() && chunk_end == n {
                let landed_anywhere = mask_override.unwrap().slot < n;
                assert!(
                    landed_anywhere,
                    "MaskEmbedOverride.slot ({}) is out of range for tokens.len() ({})",
                    mask_override.unwrap().slot,
                    n,
                );
            }
            #[cfg(feature = "moe-oracle")]
            crate::qwen35::oracle::set_prefill_start(start_pos + chunk_start)
                .map_err(|e| hip_bridge::HipError::new(0, &e))?;
            // Only the admitted wide path produces `Some(512)`; every other
            // caller (captured, TP/EP, independent, pair, Halo) passes `None`.
            let commit_stride = if wide && chunk_n > WIDENED_COMMIT_ROWS {
                Some(WIDENED_COMMIT_ROWS)
            } else {
                None
            };
            forward_prefill_chunk(
                gpu,
                weights,
                config,
                chunk,
                start_pos + chunk_start,
                kv_cache,
                dn_state,
                scratch,
                pbs,
                hidden_rb.as_deref(),
                pth_slot,
                tape_for_chunk,
                chunk_start,
                tv_for_chunk,
                false, // pre_uploaded: default path uploads inside
                None,  // band: full-stack single-GPU path
                mo_for_chunk,
                needs_last_token_logits,
                max_layer,
                None, // routed_out: non-EP single-GPU path
                fusion,
                commit_stride,
            )?;
            if let Some(rb) = hidden_rb.as_mut() {
                // Place the chunk's hidden rows at the ring's current head,
                // then advance head by n. Staged chunks scatter their
                // fixed-offset staging writes here, the out-of-capture step:
                // graph-captured writes went to staging[0..n*h], and head is
                // read from CPU state at call time (not baked into a captured
                // graph node). Eager chunks wider than staging already wrote
                // at head inside the chunk and only advance it.
                rb.finish_prefill_chunk(gpu, chunk_n)?;
            }
            chunk_start = chunk_end;
        }
        Ok(())
    })();
    if widened_from_cache {
        // Return the retained PBS to the cache (also on error: it is pure
        // scratch, and the next use overwrites before reading).
        if let Some(owned) = own_pbs {
            if can_retain_widened_pbs(gpu, kv_cache, config) {
                *scratch.widened_prefill_batch.borrow_mut() = Some(owned);
            } else {
                let _ = owned.free_gpu(gpu);
            }
        }
    } else if let Some(owned) = own_pbs {
        owned.free_gpu(gpu);
    }
    if let Some((pbs2, stage)) = pair_scratch {
        let _ = stage.free_gpu(gpu);
        let _ = pbs2.free_gpu(gpu);
    }
    result
}

/// Accepts the dtypes the batched prefill path can handle (shared by the
/// eligibility check in `forward_prefill_batch` and the per-layer dtype
/// branches in `forward_prefill_chunk`).
#[inline]
// IMPORTANT: This allowlist is paired with the `is_mq*` matchers in
// forward_prefill_chunk (lines 4063+, 4360+, 4768, 4919) and with the
// MoE FFN gate `moe_ffn_batched_admissible`. They MUST be updated together when
// adding a new batchable dtype. Updating one without the others either
// produces dead code (safe but useless) or silent prefill corruption
// (HFQ4-stride GEMM reading a different-stride weight block). See
// docs/plans/mq-lloyd-batched-prefill-followup.md for the full
// checklist + rationale.
//
// As of this PR (issue #116 Phase 5 + gfx12 validation): MQ3G256Lloyd is
// wired through the gemm_*_mq3g256_lloyd_wmma family on gfx11 and gfx12
// (always-on; validated on gfx1201 since PR #195). MQ4G256Lloyd is wired
// through the gemm_*_mq4g256_lloyd_wmma family on gfx11 and gfx12
// (always-on). MQ2G256Lloyd remains unwired — MQ2-Lloyd lands separately.
pub(crate) fn is_batchable_la(dt: DType, arch: &str) -> bool {
    let always_ok = matches!(
        dt,
        DType::MQ4G256 | DType::HFQ4G256
        | DType::MQ6G256 | DType::HFQ6G256
        | DType::Q8_0
        // TQ2G128/BQ1G128 (PrismML Bonsai ternary/binary). Unrotated, so they
        // take the plain-rmsnorm activation path and dispatch through
        // plain_gemm_key_for to the tiled prefill GEMMs. Admitting them here is
        // only safe because every is_q8 unfused branch was widened to accept
        // them in the same change -- see the all-together rule in
        // docs/plans/mq-lloyd-batched-prefill-followup.md.
        | DType::TQ2G128 | DType::BQ1G128
        // Phase 1.5 (PARO): wqkv/wz/wo are ParoQ4G128, w_alpha/w_beta are F32
        // on shisa-Qwen3.6-A3B-PARO. Dispatch in the DeltaNetMoe LA matcher
        // routes these through gemm_hfq4g128 (with per-weight Givens
        // rotation pre-pass) and gemm_f32_batched respectively. Eligibility
        // is gated downstream by the env-keyed moe_ffn_batched_admissible
        // (HIPFIRE_PARO_BATCHED=1) — admitting them here keeps non-PARO
        // models unaffected because no production checkpoint sets
        // wqkv.gpu_dtype = ParoQ4G128 outside the shisa-PARO codepath.
        | DType::ParoQ4G128 | DType::F32
    );
    if always_ok {
        return true;
    }
    // MQ3 (uniform / HFQ3 family) is batchable on archs with a WMMA
    // family ported. As of this commit:
    //   - gfx11 (gfx1100/1101/1102/1150/1151): wave32 WMMA via the
    //     `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` builtin.
    //   - gfx12 (gfx1200/1201): wave32 WMMA via the `_w32_gfx12` builtin
    //     with K4 unroll + half8_t lane-split, runtime-validated through
    //     the existing HFQ3 dispatch fork (gemm_*_hfq3g256_wmma_gfx12).
    // gfx906 GCN5 / gfx94x CDNA3 lack a ported MQ3 WMMA kernel; they
    // stay on the per-token forward_scratch fallback (correct, just
    // slower). gfx10 RDNA1/2 gains batched-prefill support via the
    // scalar HFQ3 GEMM family below (Phase 1 of
    // docs/plans/gfx10_mq3_prefill.md).
    let mq3_uniform_with_wmma = matches!(dt, DType::MQ3G256)
        && matches!(
            arch,
            "gfx1100"
                | "gfx1101"
                | "gfx1102"
                | "gfx1103"
                | "gfx1150"
                | "gfx1151"
                | "gfx1152"
                | "gfx1200"
                | "gfx1201"
        );

    // gfx10 RDNA1/2 scalar HFQ3 batched-prefill family (Phase 1).
    // Routes the four LA + FA matchers below to the new non-WMMA kernels
    // (gemm_qkv_hfq3g256, gemm_qkvza_hfq3g256, gemm_gate_up_hfq3g256,
    // gemm_hfq3g256_residual). Lloyd-MQ3 stays gated on gfx11+ — no
    // gfx10 Lloyd port (separate larger project).
    let mq3_uniform_with_gfx10_scalar = matches!(dt, DType::MQ3G256)
        && matches!(
            arch,
            "gfx1010" | "gfx1011" | "gfx1012" | "gfx1013" | "gfx1030" | "gfx1031" | "gfx1032"
        );

    // HFP4G32 / MFP4G32 (v2 #2 batched WMMA prefill): same arch gate as
    // MQ3. The 4 fused kernels (gemm_qkv/qkvza/gate_up/residual_hfp4g32_wmma)
    // ship in pairs for gfx11 + gfx12; identical eligibility to llama.rs
    // (see hipfire_runtime::llama::is_batchable_la).
    let fp4_with_wmma = matches!(dt, DType::HFP4G32 | DType::MFP4G32)
        && matches!(
            arch,
            "gfx1100"
                | "gfx1101"
                | "gfx1102"
                | "gfx1103"
                | "gfx1150"
                | "gfx1151"
                | "gfx1152"
                | "gfx1200"
                | "gfx1201"
        );

    // Lloyd-MQ3 (MQ3G256Lloyd): Phase 5 of issue #116 ships the
    // gemm_*_mq3g256_lloyd_wmma family alongside the existing HFQ3 WMMA
    // path; group stride differs (112 B Lloyd vs 104 B HFQ3) so dispatch
    // must route to the Lloyd-specific arms (handled by the LA/FA
    // matchers downstream — see followup-checklist condition 3).
    // gfx12 (RDNA4) siblings are always-on after gfx1201 validation (PR #195).
    let lloyd_mq3_with_wmma = matches!(dt, DType::MQ3G256Lloyd)
        && matches!(
            arch,
            "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" | "gfx1200" | "gfx1201"
        );

    // Lloyd-MQ4 (MQ4G256Lloyd): shipped as part of issue #182.
    // Uses the gemm_*_mq4g256_lloyd_wmma family; group stride differs
    // (160 B Lloyd vs 136 B HFQ4) so dispatch routes through the
    // Lloyd-specific arms in forward_prefill_chunk.
    // ANTIBLEED admit-vs-select fix: the MQ4-Lloyd batched-prefill GEMM source
    // selectors (gemm_*_mq4g256_lloyd_wmma_for_arch in rdna-compute/kernels.rs)
    // ship a kernel only for gfx1100/1101/1102/1151 (+ gfx12 siblings) and
    // PANIC on any other arch (160 B Lloyd stride mismatches the default).
    // gfx1150 has no MQ4-Lloyd source (intentionally excluded to stay
    // symmetric with the MQ4-Lloyd GEMV/fused-decode path — see
    // kernels.rs:195), so a gfx1150 box doing MQ4-Lloyd batched prefill would
    // crash at source lookup. Drop gfx1150 from the admit set so admit ==
    // select. (MQ3-Lloyd DOES ship a gfx1150 source, hence its admit set
    // keeps gfx1150.) gfx12 always-on after gfx1201 validation (PR #195).
    let lloyd_mq4_with_wmma = matches!(dt, DType::MQ4G256Lloyd)
        && matches!(
            arch,
            "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" | "gfx1200" | "gfx1201"
        );

    // MFP4G32E8 on gfx11/gfx1151/gfx12: the mfp4-E8 A3B model takes the
    // batched-prefill path (FWHT-rotated activations + dequant→F16 GEMM for
    // the shared expert, indexed E8 kernels for the routed experts). Admission
    // is behind the HIPFIRE_E8_GFX12 gate because the shared-expert dequant
    // path is validated on gfx1151 only; other arches are opt-in for now.
    // The LA matchers (wqkv/wz/wo/etc.) for an MFP4G32E8 A3B model are
    // still MQ4/Q8 (only the FFN expert weights are E8), so reaching here
    // with DType::MFP4G32E8 means a weight was quantized to E8 dtype at the
    // LA level — admitting it keeps the eligibility gate from rejecting the
    // whole model when an attention tensor is E8 (unlikely today, but correct
    // defensively). The real admission gate for the FFN body is
    // `moe_ffn_batched_admissible`.
    let e8_with_wmma = matches!(dt, DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8)
        && matches!(
            arch,
            "gfx1100"
                | "gfx1101"
                | "gfx1102"
                | "gfx1150"
                | "gfx1151"
                | "gfx1152"
                | "gfx1200"
                | "gfx1201"
        )
        && hipfire_config::developer_var("HIPFIRE_E8_GFX12")
            .ok()
            .as_deref()
            == Some("1");
    // MQ4G256V2 (qt44) and MQ6/5/3/2G256V2 (qt47-50) batched prefill GEMM.
    // Dedicated WMMA sources exist for BOTH gfx11
    // (gfx1100/1101/1102/1150/1151, wave32 WMMA) and gfx12 (gfx1200/1201) —
    // parity-proven on gfx1100/gfx1151 at rel-RMS 2.5e-4–4.0e-4 across
    // residual/qkv/qkvza/gate_up K=256/512 N=16/64. Admit on HasWmma
    // (gfx1100/1101/1102/1150/1151 + gfx1200/1201) but gate the gfx11 half
    // behind HIPFIRE_MQV2_GFX11_WMMA != "0" — setting
    // HIPFIRE_MQV2_GFX11_WMMA=0 restores the per-token fallback ONLY on
    // gfx11, leaving gfx12 untouched. Delegates to the shared
    // `hipfire_runtime::llama::mqv2_wmma_batchable` rule (shared home for
    // the dtype/arch/kill-switch set). NOTE: `llama::is_batchable_la` does
    // NOT delegate to it — the llama chunk path has no V2 arms, so llama
    // refuses V2 everywhere; only this qwen35 caller admits V2. Lockstep with
    // the HasWmma predicate on GemmMq*G256V2* keys and with
    // gemm_mq*g256v2's has_wmma() guard.
    // MQ4CG256 (qt45) remains gfx12-only until its gfx11 sibling lands.
    let mqv2_with_wmma = llama::mqv2_wmma_batchable(
        dt,
        hipfire_config::developer_var("HIPFIRE_MQV2_GFX11_WMMA")
            .ok()
            .as_deref(),
        arch,
    );
    let mq_other_gfx12 = matches!(dt, DType::MQ4CG256) && matches!(arch, "gfx1200" | "gfx1201");

    // BF16 calibration teacher (qt=16) — native BF16 GEMM on gfx942 (CDNA3
    // MFMA v_mfma_f32_16x16x16bf16_1k). Gated on arch == gfx942 so the
    // eligibility check correctly rejects BF16 on non-gfx942 and falls back
    // to per-token forward_scratch, rather than admitting and then failing
    // at dispatch with UnsupportedVariant.
    let bf16_with_gfx942 =
        matches!(dt, DType::BF16) && arch == "gfx942" && rdna_compute::calib_force_bf16();

    mq3_uniform_with_wmma
        || mq3_uniform_with_gfx10_scalar
        || lloyd_mq3_with_wmma
        || lloyd_mq4_with_wmma
        || fp4_with_wmma
        || e8_with_wmma
        || mqv2_with_wmma
        || mq_other_gfx12
        || bf16_with_gfx942
}

/// Single source of truth for per-layer batchability and checked geometry.
/// Called by `validate_ep_batch_compatibility`, `prefill_batch_pbs_eligible`,
/// `fa_batched_ok` guard, and later EP state preflight. Validates every
/// projection/norm shape via checked arithmetic, rejects mismatched variant,
/// and enforces environment-sensitive dispatch predicates via
/// `is_batchable_la` and `moe_ffn_batched_admissible`.
pub fn qwen35_layer_batch_admissible(
    layer: &LayerWeights,
    config: &Qwen35Config,
    arch: &str,
) -> HipResult<()> {
    let dim = config.dim;
    // Derived dimensions with checked arithmetic.
    let k_dim = config
        .linear_num_key_heads
        .checked_mul(config.linear_key_head_dim)
        .ok_or_else(|| HipError::new(0, "qwen35_layer_batch_admissible: k_dim overflow"))?;
    let v_dim = config
        .linear_num_value_heads
        .checked_mul(config.linear_value_head_dim)
        .ok_or_else(|| HipError::new(0, "qwen35_layer_batch_admissible: v_dim overflow"))?;
    let qkv_dim = k_dim
        .checked_mul(2)
        .and_then(|v| v.checked_add(v_dim))
        .ok_or_else(|| HipError::new(0, "qwen35_layer_batch_admissible: qkv_dim overflow"))?;
    let d_inner = v_dim;
    let conv_elems = qkv_dim
        .checked_mul(config.conv_kernel_dim)
        .ok_or_else(|| HipError::new(0, "qwen35_layer_batch_admissible: conv_elems overflow"))?;
    let q_out_dim = config
        .n_heads
        .checked_mul(config.head_dim)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| HipError::new(0, "qwen35_layer_batch_admissible: q_out_dim overflow"))?;
    let kv_dim = config
        .n_kv_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| HipError::new(0, "qwen35_layer_batch_admissible: kv_dim overflow"))?;
    let o_in = config
        .n_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| HipError::new(0, "qwen35_layer_batch_admissible: o_in overflow"))?;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    match layer {
        LayerWeights::DeltaNet(l) => {
            if l.attn_norm.shape != vec![dim] {
                return Err(HipError::new(0, "DeltaNet attn_norm shape mismatch"));
            }
            if l.wqkv.m != qkv_dim || l.wqkv.k != dim {
                return Err(HipError::new(0, "DeltaNet wqkv shape mismatch"));
            }
            if l.wz.m != d_inner || l.wz.k != dim {
                return Err(HipError::new(0, "DeltaNet wz shape mismatch"));
            }
            if l.w_alpha.m != config.linear_num_value_heads || l.w_alpha.k != dim {
                return Err(HipError::new(0, "DeltaNet w_alpha shape mismatch"));
            }
            if l.w_beta.m != config.linear_num_value_heads || l.w_beta.k != dim {
                return Err(HipError::new(0, "DeltaNet w_beta shape mismatch"));
            }
            if l.a_log.shape != vec![config.linear_num_value_heads] {
                return Err(HipError::new(0, "DeltaNet a_log shape mismatch"));
            }
            if l.dt_bias.shape != vec![config.linear_num_value_heads] {
                return Err(HipError::new(0, "DeltaNet dt_bias shape mismatch"));
            }
            if l.conv_weight.shape != vec![conv_elems] {
                return Err(HipError::new(0, "DeltaNet conv_weight shape mismatch"));
            }
            if l.norm_weight.shape != vec![config.linear_value_head_dim] {
                return Err(HipError::new(0, "DeltaNet norm_weight shape mismatch"));
            }
            if l.wo.m != dim || l.wo.k != d_inner {
                return Err(HipError::new(0, "DeltaNet wo shape mismatch"));
            }
            if l.ffn_norm.shape != vec![dim] {
                return Err(HipError::new(0, "DeltaNet ffn_norm shape mismatch"));
            }
            if l.w_gate.m != config.hidden_dim || l.w_gate.k != dim {
                return Err(HipError::new(0, "DeltaNet w_gate shape mismatch"));
            }
            if l.w_up.m != config.hidden_dim || l.w_up.k != dim {
                return Err(HipError::new(0, "DeltaNet w_up shape mismatch"));
            }
            if l.w_down.m != dim || l.w_down.k != config.hidden_dim {
                return Err(HipError::new(0, "DeltaNet w_down shape mismatch"));
            }
            for (name, dt) in [
                ("wqkv", l.wqkv.gpu_dtype),
                ("wz", l.wz.gpu_dtype),
                ("w_beta", l.w_beta.gpu_dtype),
                ("w_alpha", l.w_alpha.gpu_dtype),
                ("wo", l.wo.gpu_dtype),
                ("w_gate", l.w_gate.gpu_dtype),
                ("w_up", l.w_up.gpu_dtype),
                ("w_down", l.w_down.gpu_dtype),
            ] {
                if !is_batchable_la(dt, arch) {
                    return Err(HipError::new(
                        0,
                        &format!("DeltaNet {name} dtype {dt:?} not batchable on {arch}"),
                    ));
                }
            }
            Ok(())
        }
        LayerWeights::FullAttn(l) => {
            if l.attn_norm.shape != vec![dim] {
                return Err(HipError::new(0, "FullAttn attn_norm shape mismatch"));
            }
            if l.wq.m != q_out_dim || l.wq.k != dim {
                return Err(HipError::new(0, "FullAttn wq shape mismatch"));
            }
            if l.wk.m != kv_dim || l.wk.k != dim {
                return Err(HipError::new(0, "FullAttn wk shape mismatch"));
            }
            if l.wv.m != kv_dim || l.wv.k != dim {
                return Err(HipError::new(0, "FullAttn wv shape mismatch"));
            }
            if l.wo.m != dim || l.wo.k != o_in {
                return Err(HipError::new(0, "FullAttn wo shape mismatch"));
            }
            if l.q_norm.shape != vec![config.head_dim] {
                return Err(HipError::new(0, "FullAttn q_norm shape mismatch"));
            }
            if l.k_norm.shape != vec![config.head_dim] {
                return Err(HipError::new(0, "FullAttn k_norm shape mismatch"));
            }
            if l.ffn_norm.shape != vec![dim] {
                return Err(HipError::new(0, "FullAttn ffn_norm shape mismatch"));
            }
            if l.w_gate.m != config.hidden_dim || l.w_gate.k != dim {
                return Err(HipError::new(0, "FullAttn w_gate shape mismatch"));
            }
            if l.w_up.m != config.hidden_dim || l.w_up.k != dim {
                return Err(HipError::new(0, "FullAttn w_up shape mismatch"));
            }
            if l.w_down.m != dim || l.w_down.k != config.hidden_dim {
                return Err(HipError::new(0, "FullAttn w_down shape mismatch"));
            }
            for (name, dt) in [
                ("wq", l.wq.gpu_dtype),
                ("wk", l.wk.gpu_dtype),
                ("wv", l.wv.gpu_dtype),
                ("wo", l.wo.gpu_dtype),
                ("w_gate", l.w_gate.gpu_dtype),
                ("w_up", l.w_up.gpu_dtype),
                ("w_down", l.w_down.gpu_dtype),
            ] {
                if !is_batchable_la(dt, arch) {
                    return Err(HipError::new(
                        0,
                        &format!("FullAttn {name} dtype {dt:?} not batchable on {arch}"),
                    ));
                }
            }
            Ok(())
        }
        LayerWeights::DeltaNetMoe(l) => {
            if l.attn_norm.shape != vec![dim] {
                return Err(HipError::new(0, "DeltaNetMoe attn_norm shape mismatch"));
            }
            if l.wqkv.m != qkv_dim || l.wqkv.k != dim {
                return Err(HipError::new(0, "DeltaNetMoe wqkv shape mismatch"));
            }
            if l.wz.m != d_inner || l.wz.k != dim {
                return Err(HipError::new(0, "DeltaNetMoe wz shape mismatch"));
            }
            if l.w_alpha.m != config.linear_num_value_heads || l.w_alpha.k != dim {
                return Err(HipError::new(0, "DeltaNetMoe w_alpha shape mismatch"));
            }
            if l.w_beta.m != config.linear_num_value_heads || l.w_beta.k != dim {
                return Err(HipError::new(0, "DeltaNetMoe w_beta shape mismatch"));
            }
            if l.a_log.shape != vec![config.linear_num_value_heads] {
                return Err(HipError::new(0, "DeltaNetMoe a_log shape mismatch"));
            }
            if l.dt_bias.shape != vec![config.linear_num_value_heads] {
                return Err(HipError::new(0, "DeltaNetMoe dt_bias shape mismatch"));
            }
            if l.conv_weight.shape != vec![conv_elems] {
                return Err(HipError::new(0, "DeltaNetMoe conv_weight shape mismatch"));
            }
            if l.norm_weight.shape != vec![config.linear_value_head_dim] {
                return Err(HipError::new(0, "DeltaNetMoe norm_weight shape mismatch"));
            }
            if l.wo.m != dim || l.wo.k != d_inner {
                return Err(HipError::new(0, "DeltaNetMoe wo shape mismatch"));
            }
            if l.ffn_norm.shape != vec![dim] {
                return Err(HipError::new(0, "DeltaNetMoe ffn_norm shape mismatch"));
            }
            for (name, dt) in [
                ("wqkv", l.wqkv.gpu_dtype),
                ("wz", l.wz.gpu_dtype),
                ("w_beta", l.w_beta.gpu_dtype),
                ("w_alpha", l.w_alpha.gpu_dtype),
                ("wo", l.wo.gpu_dtype),
            ] {
                if !is_batchable_la(dt, arch) {
                    return Err(HipError::new(
                        0,
                        &format!("DeltaNetMoe {name} dtype {dt:?} not batchable on {arch}"),
                    ));
                }
            }
            // MoE geometry.
            if l.ffn.router.m != config.num_experts || l.ffn.router.k != dim {
                return Err(HipError::new(0, "DeltaNetMoe router shape mismatch"));
            }
            if l.ffn.shared_expert.gate.m != smi || l.ffn.shared_expert.gate.k != dim {
                return Err(HipError::new(0, "DeltaNetMoe shared gate shape mismatch"));
            }
            if l.ffn.shared_expert.up.m != smi || l.ffn.shared_expert.up.k != dim {
                return Err(HipError::new(0, "DeltaNetMoe shared up shape mismatch"));
            }
            if l.ffn.shared_expert.down.m != dim || l.ffn.shared_expert.down.k != smi {
                return Err(HipError::new(0, "DeltaNetMoe shared down shape mismatch"));
            }
            if l.ffn.shared_expert_gate.m != 1 || l.ffn.shared_expert_gate.k != dim {
                return Err(HipError::new(
                    0,
                    "DeltaNetMoe shared_expert_gate shape mismatch",
                ));
            }
            // TopK gate is environment-sensitive.
            if !hipfire_dispatch::families::moe::moe_prefill_topk_shape_supported(
                config.num_experts_per_tok,
                config.num_experts,
            ) {
                return Err(HipError::new(0, "DeltaNetMoe topk shape unsupported"));
            }
            let admit_mq6 = hipfire_dispatch::families::moe::mq6_batched_admit_enabled_from_env(
                hipfire_config::developer_var("HIPFIRE_MOE_MQ6_ADMIT")
                    .ok()
                    .as_deref(),
                arch,
            );
            let dtypes = moe_prefill_dtypes(&l.ffn)
                .ok_or_else(|| HipError::new(0, "DeltaNetMoe MoE dtype metadata unavailable"))?;
            if !hipfire_dispatch::families::moe::gated_moe_prefill_admissible(
                &dtypes, admit_mq6, arch,
            ) {
                return Err(HipError::new(0, "DeltaNetMoe moe_ffn not batch-admissible"));
            }
            Ok(())
        }
        LayerWeights::FullAttnMoe(l) => {
            if l.attn_norm.shape != vec![dim] {
                return Err(HipError::new(0, "FullAttnMoe attn_norm shape mismatch"));
            }
            if l.wq.m != q_out_dim || l.wq.k != dim {
                return Err(HipError::new(0, "FullAttnMoe wq shape mismatch"));
            }
            if l.wk.m != kv_dim || l.wk.k != dim {
                return Err(HipError::new(0, "FullAttnMoe wk shape mismatch"));
            }
            if l.wv.m != kv_dim || l.wv.k != dim {
                return Err(HipError::new(0, "FullAttnMoe wv shape mismatch"));
            }
            if l.wo.m != dim || l.wo.k != o_in {
                return Err(HipError::new(0, "FullAttnMoe wo shape mismatch"));
            }
            if l.q_norm.shape != vec![config.head_dim] {
                return Err(HipError::new(0, "FullAttnMoe q_norm shape mismatch"));
            }
            if l.k_norm.shape != vec![config.head_dim] {
                return Err(HipError::new(0, "FullAttnMoe k_norm shape mismatch"));
            }
            if l.ffn_norm.shape != vec![dim] {
                return Err(HipError::new(0, "FullAttnMoe ffn_norm shape mismatch"));
            }
            for (name, dt) in [
                ("wq", l.wq.gpu_dtype),
                ("wk", l.wk.gpu_dtype),
                ("wv", l.wv.gpu_dtype),
                ("wo", l.wo.gpu_dtype),
            ] {
                if !is_batchable_la(dt, arch) {
                    return Err(HipError::new(
                        0,
                        &format!("FullAttnMoe {name} dtype {dt:?} not batchable on {arch}"),
                    ));
                }
            }
            if l.ffn.router.m != config.num_experts || l.ffn.router.k != dim {
                return Err(HipError::new(0, "FullAttnMoe router shape mismatch"));
            }
            if l.ffn.shared_expert.gate.m != smi || l.ffn.shared_expert.gate.k != dim {
                return Err(HipError::new(0, "FullAttnMoe shared gate shape mismatch"));
            }
            if l.ffn.shared_expert.up.m != smi || l.ffn.shared_expert.up.k != dim {
                return Err(HipError::new(0, "FullAttnMoe shared up shape mismatch"));
            }
            if l.ffn.shared_expert.down.m != dim || l.ffn.shared_expert.down.k != smi {
                return Err(HipError::new(0, "FullAttnMoe shared down shape mismatch"));
            }
            if l.ffn.shared_expert_gate.m != 1 || l.ffn.shared_expert_gate.k != dim {
                return Err(HipError::new(
                    0,
                    "FullAttnMoe shared_expert_gate shape mismatch",
                ));
            }
            if !hipfire_dispatch::families::moe::moe_prefill_topk_shape_supported(
                config.num_experts_per_tok,
                config.num_experts,
            ) {
                return Err(HipError::new(0, "FullAttnMoe topk shape unsupported"));
            }
            let admit_mq6 = hipfire_dispatch::families::moe::mq6_batched_admit_enabled_from_env(
                hipfire_config::developer_var("HIPFIRE_MOE_MQ6_ADMIT")
                    .ok()
                    .as_deref(),
                arch,
            );
            let dtypes = moe_prefill_dtypes(&l.ffn)
                .ok_or_else(|| HipError::new(0, "FullAttnMoe MoE dtype metadata unavailable"))?;
            if !hipfire_dispatch::families::moe::gated_moe_prefill_admissible(
                &dtypes, admit_mq6, arch,
            ) {
                return Err(HipError::new(0, "FullAttnMoe moe_ffn not batch-admissible"));
            }
            Ok(())
        }
    }
}
/// Choose the next prefill chunk without leaving an invalid singleton tail.
///
/// Batched kernels require at least `MIN_BATCH` rows. When a full chunk would
/// leave one row, move one row into the tail (for example, 257 → 255 + 2).
#[inline]
fn next_prefill_chunk_len(remaining: usize, max_batch: usize) -> Option<usize> {
    if remaining < MIN_BATCH || max_batch < MIN_BATCH {
        return None;
    }
    if max_batch == MIN_BATCH && remaining % MIN_BATCH != 0 {
        return None;
    }

    let chunk = remaining.min(max_batch);
    if remaining - chunk == 1 {
        (chunk > MIN_BATCH).then_some(chunk - 1)
    } else {
        Some(chunk)
    }
}

/// Plan scratch owned by one prefill call.
///
/// Large prompts retain the configured chunk size, while a short prompt gets
/// exactly one right-sized chunk. The DeltaNet S-state tape is consumed only by
/// tree verify; ordinary prefill advances recurrent state in place.
#[inline]
fn owned_prefill_scratch_plan(
    n: usize,
    configured_max_batch: usize,
    tree_verify: bool,
) -> (usize, bool) {
    debug_assert!(n > 0);
    debug_assert!(configured_max_batch >= MIN_BATCH);
    (configured_max_batch.min(n.max(MIN_BATCH)), tree_verify)
}

/// Whether `forward_prefill_batch_with_pbs` will take the tape-capturing
/// batched (PBS) path for an `n`-token call.
///
/// This is the single source of truth for the eligibility decision shared by
/// the forward and speculative-decode callers.
pub fn prefill_batch_pbs_eligible(
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    // Kept for API stability and future state-aware gating.
    _dn_state: &DeltaNetState,
    n: usize,
    arch: &str,
    moe_router_logits_present: bool,
) -> bool {
    let decouple_env = hipfire_config::developer_var("HIPFIRE_MTP_VERIFY_DECOUPLE").ok();
    let is_rdna3_decouple = arch.starts_with("gfx11");
    let verify_decouple = n <= 32
        && decouple_env.as_deref() != Some("0")
        && (is_rdna3_decouple || decouple_env.as_deref() == Some("1"));
    let force_fallback = !verify_decouple && !hipfire_runtime::config::get().prefill_batched;
    let has_dn = weights
        .layers
        .iter()
        .any(|lw| matches!(lw, LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_)));
    let all_layers_ok = weights.layers.iter().all(|lw| {
        if matches!(
            lw,
            LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_)
        ) && !moe_router_logits_present
        {
            return false;
        }
        qwen35_layer_batch_admissible(lw, config, arch).is_ok()
    });
    let result = !force_fallback && n >= MIN_BATCH && has_dn && all_layers_ok;
    if hipfire_config::developer_var("HIPFIRE_DEBUG_BATCH")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "[hipfire::batch_eligible] result={result} \
             arch={arch} n={n} n>={MIN_BATCH}={} \
             force_fallback={force_fallback} \
             has_dn={has_dn} \
             moe_router_logits_present={moe_router_logits_present} \
             all_layers_ok={all_layers_ok}",
            n >= MIN_BATCH,
        );
    }
    result
}

pub(crate) fn trace_finite_if_enabled(gpu: &Gpu, label: &str, tensor: &GpuTensor) -> HipResult<()> {
    if hipfire_config::developer_var_os("HIPFIRE_QWEN35_FINITE_TRACE").is_none() {
        return Ok(());
    }
    let vals = gpu.download_f32(tensor)?;
    let mut n_nan = 0usize;
    let mut n_inf = 0usize;
    let mut n_finite = 0usize;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &v in &vals {
        if v.is_nan() {
            n_nan += 1;
        } else if v.is_infinite() {
            n_inf += 1;
        } else {
            n_finite += 1;
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    eprintln!(
        "[qwen35 finite] {label}: finite={n_finite}/{} nan={n_nan} inf={n_inf} range=[{min_v:.6e}, {max_v:.6e}]",
        vals.len(),
    );
    Ok(())
}

/// Process one chunk of up to `pbs.max_batch` tokens through the batched
/// prefill path. All LA layers go through batched kernels; all FA layers
/// go through a per-token gather/scatter loop with the inline FA body.
///
/// `hidden_rb`: if `Some`, post-layer residual hidden states for configured
/// extract layers get written into the ring buffer at its current head. The
/// caller (forward_prefill_batch) advances the head by N after this chunk
/// completes so writes from the next chunk don't overwrite.
///
/// `per_token_hidden_out`: if `Some((dst, offset_rows))`, writes post-output
/// RMSNorm hidden for each of the N tokens into `dst[offset_rows..offset_rows+N]`
/// in row-major order. Required for DFlash verify to compute per-position
/// logits via B sequential `weight_gemv` calls on the caller side.
///
/// `gdn_tape` + `tape_offset`: if `Some`, captures the post-processed
/// `(q, k, v, α, β)` tensors per DN layer at rows
/// `[tape_offset .. tape_offset+N]` right before the batched GDN kernel
/// runs. Used by the DFlash rollback path.
/// Does the MoE FFN admit the batched prefill fast path?
///
/// Router + shared_expert_gate may be Q8_0 (the engine's default — these
/// small tensors are never quantized to MQ4 to preserve routing
/// accuracy). They get a separate `gemm_q8_0_batched_chunked` dispatch
/// against the *un-rotated* `x_norm_batch` inside
/// `prefill_moe_ffn_body_batched`. All other weights (shared expert
/// gate/up/down + every expert gate_up/down) must be MQ4G256 — these are
/// the ones consumed by the FWHT-rotated `_k8_indexed_batched` and
/// `gemm_hfq4g256` family, which is stride-136 only.
///
/// Pre-fix this required ALL weights to be MQ4G256, which made every
/// A3B model fall back to per-token prefill because router is universally
/// Q8_0. Widening to accept Q8 router + Q8 shared_expert_gate unlocks
/// uniform-MQ4 A3B variants (Qwen3.5-A3B, qwen3.6-35b-a3b-uniform.mq4).
/// Mixed-precision Qwen3.6-A3B (MQ6 in 16/40 layers) still falls back —
/// needs an MQ6 sibling for `_k8_indexed_batched`, follow-up work.
/// MoE FFN admit predicate for the batched prefill body
/// `prefill_moe_ffn_body_batched`. Per-projection MQ4 OR MQ6 admit:
///
/// - router, shared_expert_gate: MQ4 or Q8 (small scalars; dispatched
///   inline below).
/// - shared_expert.gate AND .up: same dtype, MQ4 or MQ6 (fused gate+up
///   kernel handles one storage layout per call).
/// - shared_expert.down: MQ4 or MQ6 (independent dtype).
/// - experts.gate_up: uniform across all experts in this layer, MQ4 or MQ6.
/// - experts.down: uniform across all experts in this layer, MQ4 or MQ6.
///
/// AWQ A3B dtype dump 2026-05-19 confirms experts are uniform per
/// projection per layer. The 4 grouped/fused dispatch sites in
/// `prefill_moe_ffn_body_batched` branch on the actual dtype, so a
/// layer admitted here is dispatchable end-to-end.
///
/// Assemble the shared, dtype-only prefill admission record from Qwen-owned
/// weights. The policy itself lives in dispatch; this adapter only translates
/// the load-time metadata, including global EP expert tiers.
pub(crate) fn moe_prefill_dtypes(
    ffn: &MoeFfnWeights,
) -> Option<hipfire_dispatch::families::moe::MoePrefillDtypes> {
    if let Some(global) = ffn.global_expert_dtypes.as_ref() {
        let first = global.first()?;
        return Some(hipfire_dispatch::families::moe::MoePrefillDtypes {
            router: ffn.router.gpu_dtype,
            shared_expert_scalar_gate: ffn.shared_expert_gate.gpu_dtype,
            shared_expert_gate: ffn.shared_expert.gate.gpu_dtype,
            shared_expert_up: ffn.shared_expert.up.gpu_dtype,
            shared_expert_down: ffn.shared_expert.down.gpu_dtype,
            expert_gate_up: first.0,
            expert_down: first.1,
            expert_gate_up_uniform: global.iter().all(|(g, _)| *g == first.0),
            expert_down_uniform: global.iter().all(|(_, d)| *d == first.1),
            routed_mixed_merged: ffn.expert_dtype_tags.is_some(),
        });
    }
    let first = ffn.experts.first()?;
    Some(hipfire_dispatch::families::moe::MoePrefillDtypes {
        router: ffn.router.gpu_dtype,
        shared_expert_scalar_gate: ffn.shared_expert_gate.gpu_dtype,
        shared_expert_gate: ffn.shared_expert.gate.gpu_dtype,
        shared_expert_up: ffn.shared_expert.up.gpu_dtype,
        shared_expert_down: ffn.shared_expert.down.gpu_dtype,
        expert_gate_up: first.gate_up.gpu_dtype,
        expert_down: first.down.gpu_dtype,
        expert_gate_up_uniform: ffn
            .experts
            .iter()
            .all(|e| e.gate_up.gpu_dtype == first.gate_up.gpu_dtype),
        expert_down_uniform: ffn
            .experts
            .iter()
            .all(|e| e.down.gpu_dtype == first.down.gpu_dtype),
        routed_mixed_merged: ffn.expert_dtype_tags.is_some(),
    })
}

/// Batched MoE FFN for `forward_prefill_chunk`. Takes the post-attention
/// residual stream in `pbs.x_batch` ([N × dim]) and writes the FFN output
/// residual back into the same buffer in-place.
///
/// Preconditions (caller must guarantee):
/// - `moe_ffn_batched_admissible(ffn)` returns true: router + shared_expert_gate may
///   be MQ4G256 *or* Q8_0; all other MoE weights must be MQ4G256
/// - `pbs.moe_*_batch` tensors are allocated (num_experts > 0 at scratch
///   construction time) and sized to max_batch ≥ N
/// - `config.num_experts_per_tok == 8` and `config.num_experts <= 1024`
///   (hard limits of the batched top-K kernel)
///
/// Sequence mirrors `moe_ffn_decode_impl`'s GPU fast path, with every
/// per-token launch replaced by its N-batched equivalent. Byte-exact
/// except for atomicAdd nondeterminism in the routed-down accumulation
/// (same as the single-token indexed kernel it replaces).
// pub(crate): also called directly by forward_slots.rs (MoE slots port) —
// the MoE FFN body is stateless per row (no kv_cache, no dn_state, no
// positions), so the flat N-row batch this function already expects is
// exactly what a multi-slot step produces; no slot-aware variant is needed.
// Visibility change only — behavior and existing callers are unchanged.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_moe_ffn_body_batched(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    n: usize,
    ctx: &DispatchCtx,
    model_has_mq6_moe: bool,
    // EP (Ship 6 substrate-EP prefill): when `Some`, the routed combine writes
    // into this zeroed `[n × dim]` partial instead of `pbs.x_batch` (the EP
    // driver all-reduce-sums it across ranks and adds into x_batch). The shared
    // expert stays in `pbs.x_batch` (replicated per rank — added once to each
    // rank's own copy). `None` = byte-identical single-GPU behavior.
    routed_out: Option<&GpuTensor>,
) -> HipResult<()> {
    // Historical entry point: replicated local routing (Single seal path).
    // Byte-identical to before; the slot-aware forward_slots.rs callers and
    // all single-GPU paths preserve the fixed-width batch semantics.
    prefill_moe_ffn_body_batched_with_route(
        gpu,
        ffn,
        ffn_norm,
        config,
        pbs,
        n,
        ctx,
        model_has_mq6_moe,
        routed_out,
        PrefillRouteMode::Replicated,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_moe_prefill_params<'a>(
    gpu: &Gpu,
    ffn: &'a MoeFfnWeights,
    ffn_norm: &'a GpuTensor,
    config: &Qwen35Config,
    pbs: &'a PrefillBatchScratch,
    n: usize,
    model_has_mq6_moe: bool,
    routed_out: Option<&'a GpuTensor>,
    route: PrefillRouteMode<'a>,
) -> HipResult<(
    hipfire_dispatch::pipeline::sealed_moe::BoundMoeExperts<'a>,
    hipfire_dispatch::families::moe::MoePrefillParams<'a>,
)> {
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let k_top = config.num_experts_per_tok;
    let n_exp = config.num_experts;

    let router_logits = pbs.moe_router_logits_batch.as_ref().expect("moe scratch");
    let router_scores = pbs
        .moe_router_score_views_batch
        .as_ref()
        .and_then(|views| n.checked_sub(1).and_then(|index| views.get(index)))
        .ok_or_else(|| HipError::new(0, "moe router score scratch view is unavailable"))?;
    let topk_indices = pbs.moe_topk_indices_batch.as_ref().expect("moe scratch");
    let topk_weights = pbs.moe_topk_weights_batch.as_ref().expect("moe scratch");
    let gate_batch = pbs.moe_gate_batch.as_ref().expect("moe scratch");
    let up_batch = pbs.moe_up_batch.as_ref().expect("moe scratch");
    let rot_batch = pbs.moe_rot_batch.as_ref().expect("moe scratch");
    let down_expanded = pbs.moe_down_expanded_batch.as_ref().expect("moe scratch");
    let shared_scalar = pbs.moe_shared_scalar_batch.as_ref().expect("moe scratch");
    let shared_gate = pbs.moe_shared_gate_batch.as_ref().expect("moe scratch");
    let shared_up = pbs.moe_shared_up_batch.as_ref().expect("moe scratch");
    let shared_rot = pbs.moe_shared_rot_batch.as_ref().expect("moe scratch");

    let bound = ffn.bound_experts()?;
    let (per_expert_gate_up, per_expert_down) = ffn.per_expert_tier_tables();
    let moe_dtypes = hipfire_dispatch::families::moe::MoeDtypes {
        router: ffn.router.gpu_dtype,
        shared: Some(hipfire_dispatch::families::moe::MoeSharedDtypes {
            selector: ffn.shared_expert_gate.gpu_dtype,
            gate: ffn.shared_expert.gate.gpu_dtype,
            up: ffn.shared_expert.up.gpu_dtype,
            down: ffn.shared_expert.down.gpu_dtype,
        }),
        experts_all_gate_up_mq4: if let Some(global) = ffn.global_expert_dtypes.as_ref() {
            global
                .iter()
                .all(|(g, _)| matches!(*g, DType::MQ4G256 | DType::MQ4G256V2))
        } else {
            ffn.experts
                .iter()
                .all(|e| matches!(e.gate_up.gpu_dtype, DType::MQ4G256 | DType::MQ4G256V2))
        },
        routed_gate_up: if let Some(global) = ffn.global_expert_dtypes.as_ref() {
            global[0].0
        } else {
            ffn.experts[0].gate_up.gpu_dtype
        },
        routed_down: if let Some(global) = ffn.global_expert_dtypes.as_ref() {
            global[0].1
        } else {
            ffn.experts[0].down.gpu_dtype
        },
        routed_has_mixed_experts: per_expert_gate_up.is_some() || per_expert_down.is_some(),
        has_paro_shared: ffn.paro_shared.is_some(),
        per_expert_gate_up,
        per_expert_down,
    };

    let paro_gate_up =
        ffn.paro_shared
            .as_ref()
            .map(|paro| hipfire_dispatch::families::gemv::GivensRef {
                pairs: &paro.gate_up_pairs,
                theta: &paro.gate_up_theta,
                scales: &paro.gate_up_channel_scales,
                krot: paro.krot as usize,
            });
    let paro_down =
        ffn.paro_shared
            .as_ref()
            .map(|paro| hipfire_dispatch::families::gemv::GivensRef {
                pairs: &paro.down_pairs,
                theta: &paro.down_theta,
                scales: &paro.down_channel_scales,
                krot: paro.krot as usize,
            });

    let prelude = hipfire_dispatch::families::moe::MoePrefillPrelude {
        normalization: hipfire_dispatch::families::moe::MoeNormalization::RmsNorm {
            weight: ffn_norm,
            plain_out: &pbs.x_norm_batch,
            eps: config.norm_eps,
        },
        router: ffn.router.dispatch_ref(),
        router_logits,
        router_scores,
        norm_topk_prob: config.norm_topk_prob,
        route,
        shared: Some(hipfire_dispatch::families::moe::MoeSharedPrefill {
            weights: hipfire_dispatch::families::moe::MoeSharedWeights {
                selector: ffn.shared_expert_gate.dispatch_ref(),
                gate: ffn.shared_expert.gate.dispatch_ref(),
                up: ffn.shared_expert.up.dispatch_ref(),
                down: ffn.shared_expert.down.dispatch_ref(),
            },
            intermediate: smi,
            scalar: shared_scalar,
            gate_out: shared_gate,
            up_out: shared_up,
            rotated: shared_rot,
        }),
        q8_router_policy: hipfire_dispatch::families::moe::MoeQ8RouterPolicy::DispatcherEntry,
    };

    let moe_prefill_params = hipfire_dispatch::families::moe::MoePrefillParams {
        dtypes: moe_dtypes,
        recipe: hipfire_dispatch::families::moe::MoeRecipe::SoftmaxGatedShared {
            bf16_round_trip: false,
            shared_after_combine: false,
        },
        route_policy: None,
        prelude,
        batch_size: n,
        mi,
        down_m: ffn.experts[0].down.m,
        down_k: ffn.experts[0].down.k,
        gate_up_k: ffn.experts[0].gate_up.k,
        k_top,
        n_exp,
        m_total_max: moe_grouped_m_total_bound(n * k_top, n_exp),
        force_mq4_grouped_fp16: model_has_mq6_moe
            && gpu.arch_caps.is_gfx1151()
            && gpu.flags.moe_grouped_i8.is_none(),
        topk_indices,
        topk_weights,
        x_batch: &pbs.x_batch,
        x_norm_batch: &pbs.x_norm_batch,
        x_rot_batch: &pbs.x_rot_batch,
        expert_gate_up_ptrs: &ffn.expert_gate_up_ptrs,
        expert_down_ptrs: &ffn.expert_down_ptrs,
        expert_stage_ptrs: None,
        routed_experts: ffn,
        expert_down_awq_ptrs: ffn.expert_down_awq_ptrs.as_ref(),
        expert_dtype_tags: ffn.expert_dtype_tags.as_ref(),
        gate_batch,
        up_batch,
        rot_batch,
        down_expanded,
        expert_token_counts: pbs.moe_expert_token_counts.as_ref().expect("moe scratch"),
        expert_offsets: pbs.moe_expert_offsets.as_ref().expect("moe scratch"),
        sorted_slot_index: pbs.moe_sorted_slot_index.as_ref().expect("moe scratch"),
        expert_tile_ids: pbs.moe_expert_tile_ids.as_ref().expect("moe scratch"),
        inverse_perm: pbs.moe_inverse_perm.as_ref().expect("moe scratch"),
        y_gate_up_grouped: pbs.moe_y_gate_up_grouped.as_ref().expect("moe scratch"),
        y_down_grouped: pbs.moe_y_down_grouped.as_ref().expect("moe scratch"),
        paro_gate_up,
        paro_down,
        down_awq_scale: None,
        routed_out,
    };
    Ok((bound, moe_prefill_params))
}

/// Pure grouped-prefill EP preflight shared by the runtime mesh schedule.
///
/// This reuses the exact live-operand parameter builder and sealer used by the
/// executing body. It performs no norm, routing, or expert launch; the actual
/// root route proof is still minted only after the root produces route bytes.
pub(crate) fn preflight_moe_ffn_batched_ep(
    gpu: &Gpu,
    ffn: &MoeFfnWeights,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    n: usize,
    ctx: &DispatchCtx,
    model_has_mq6_moe: bool,
    routed_out: &GpuTensor,
) -> HipResult<()> {
    let proof_slot = std::cell::Cell::new(None);
    let route = PrefillRouteMode::ProduceRoot { slot: &proof_slot };
    let (bound, params) = build_moe_prefill_params(
        gpu,
        ffn,
        ffn_norm,
        config,
        pbs,
        n,
        model_has_mq6_moe,
        Some(routed_out),
        route,
    )?;
    hipfire_dispatch::pipeline::sealed_moe::seal_prefill_ep(bound, ctx, params)
        .map(|_| ())
        .map_err(HipError::from)
}

/// Return the canonical rank-local expert output geometry that compact EP
/// gathers to root before the ordinary slot-order combine.
pub(crate) fn moe_ffn_batched_ep_slot_geometry(
    gpu: &Gpu,
    ffn: &MoeFfnWeights,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    n: usize,
    ctx: &DispatchCtx,
    model_has_mq6_moe: bool,
    routed_out: &GpuTensor,
) -> HipResult<usize> {
    let proof_slot = std::cell::Cell::new(None);
    let (bound, params) = build_moe_prefill_params(
        gpu,
        ffn,
        ffn_norm,
        config,
        pbs,
        n,
        model_has_mq6_moe,
        Some(routed_out),
        PrefillRouteMode::ProduceRoot { slot: &proof_slot },
    )?;
    let resolution = hipfire_dispatch::families::moe::MoePrefillResolution::resolve(
        &params.dtypes,
        &ctx.arch,
        &ctx.flags,
    );
    if resolution.down_path0 {
        return Err(HipError::new(
            0,
            "compact EP prefill requires expanded expert outputs",
        ));
    }
    let contribution_count = params
        .batch_size
        .checked_mul(params.k_top)
        .and_then(|slots| slots.checked_mul(params.down_m))
        .ok_or_else(|| HipError::new(0, "compact EP output count overflow"))?;
    hipfire_dispatch::pipeline::sealed_moe::seal_prefill_ep(bound, ctx, params)
        .map_err(HipError::from)?;
    Ok(contribution_count)
}

/// Fold root's gathered expert rows with the ordinary single-device combine.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_moe_ffn_batched_ep_slot_order(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    n: usize,
    ctx: &DispatchCtx,
    model_has_mq6_moe: bool,
    routed_out: &GpuTensor,
) -> HipResult<()> {
    let proof_slot = std::cell::Cell::new(None);
    let (bound, params) = build_moe_prefill_params(
        gpu,
        ffn,
        ffn_norm,
        config,
        pbs,
        n,
        model_has_mq6_moe,
        Some(routed_out),
        PrefillRouteMode::ProduceRoot { slot: &proof_slot },
    )?;
    let sealed = hipfire_dispatch::pipeline::sealed_moe::seal_prefill_ep(bound, ctx, params)
        .map_err(HipError::from)?;
    sealed.execute_ep_slot_combine(gpu).map_err(HipError::from)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_moe_ffn_body_batched_with_route<'a>(
    gpu: &mut Gpu,
    ffn: &'a MoeFfnWeights,
    ffn_norm: &'a GpuTensor,
    config: &Qwen35Config,
    pbs: &'a PrefillBatchScratch,
    n: usize,
    ctx: &'a DispatchCtx,
    model_has_mq6_moe: bool,
    routed_out: Option<&'a GpuTensor>,
    route: PrefillRouteMode<'a>,
) -> HipResult<()> {
    let (bound, params) = build_moe_prefill_params(
        gpu,
        ffn,
        ffn_norm,
        config,
        pbs,
        n,
        model_has_mq6_moe,
        routed_out,
        route,
    )?;
    let compact_ep = matches!(
        &params.prelude.route,
        PrefillRouteMode::ProduceRoot { .. } | PrefillRouteMode::AdoptRoot { .. }
    ) && bound.rank_count() > 1;
    let sealed = if compact_ep {
        hipfire_dispatch::pipeline::sealed_moe::seal_prefill_ep(bound, ctx, params)
    } else {
        hipfire_dispatch::pipeline::sealed_moe::seal_prefill(bound, ctx, params)
    }
    .map_err(HipError::from)?;

    let body_result = (|| -> HipResult<()> {
        #[cfg(feature = "moe-oracle")]
        crate::qwen35::oracle::prefill_before(
            gpu,
            ffn,
            config,
            &pbs.x_batch,
            n,
            crate::qwen35::oracle::prefill_start().map_err(|e| hip_bridge::HipError::new(0, &e))?,
        )
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;

        execute_steps(gpu, ctx, &[Step::Moe(sealed)])
            .map_err(|e| HipError::new(0, &e.to_string()))?;

        #[cfg(feature = "moe-oracle")]
        {
            let router_logits = pbs.moe_router_logits_batch.as_ref().expect("moe scratch");
            let oracle_start = crate::qwen35::oracle::prefill_start()
                .map_err(|e| hip_bridge::HipError::new(0, &e))?;
            crate::qwen35::oracle::prefill_after(
                gpu,
                ffn,
                config,
                &pbs.x_batch,
                router_logits,
                pbs.moe_topk_indices_batch.as_ref().expect("moe scratch"),
                pbs.moe_topk_weights_batch.as_ref().expect("moe scratch"),
                pbs.moe_gate_batch.as_ref().expect("moe scratch"),
                pbs.moe_up_batch.as_ref().expect("moe scratch"),
                pbs.moe_rot_batch.as_ref().expect("moe scratch"),
                pbs.moe_down_expanded_batch.as_ref().expect("moe scratch"),
                n,
                oracle_start,
            )
            .map_err(|e| hip_bridge::HipError::new(0, &e))?;
        }

        Ok(())
    })();

    body_result
}

/// Band view for `forward_prefill_chunk`. `None` (the default) means the
/// chunk processes the whole stack: embedding → all layers → final norm
/// + lm_head. `Some(b)` restricts the chunk to layers `b.layer_start..
/// b.layer_end`, skips the embedding when `!b.is_first_band` (input is
/// already in `pbs.x_batch` from a prior peer-copy), and skips the final
/// norm + lm_head when `!b.is_last_band` (output activation stays in
/// `pbs.x_batch` for the next band's peer-copy).
///
/// Counter offsets seed the running per-LA / per-KV / per-FA counters so
/// the band's first DeltaNet/FullAttn layer indexes the correct
/// `dn_state.s_matrices[i]` / `kv_cache.k_caches[i]` slot.
pub(crate) struct PrefillBandCtx<'a> {
    pub layer_start: usize,
    pub layer_end: usize,
    pub delta_layer_offset: usize,
    pub kv_layer_offset: usize,
    pub is_first_band: bool,
    pub is_last_band: bool,
    /// Per-device asym{2,3,4} givens replicas. When `Some`, the chunk's
    /// FA-layer batched KV writers use these instead of `kv_cache.givens_*`
    /// (which is `None` in multi-GPU mode by design — each device needs its
    /// own copy of the rotation tables).
    pub givens_cos: Option<&'a GpuTensor>,
    pub givens_sin: Option<&'a GpuTensor>,
    /// EP prefill route authority for single-layer MoE bands. `Replicated`
    /// everywhere else (whole-stack and multi-layer bands, decode ticks, PP).
    pub route: PrefillRouteMode<'a>,
}

/// Why a batched forward writes per-token post-output-norm hidden rows.
///
/// `Verify` is a speculative verify (DFlash, MTP, the MTP probe): it keeps
/// the `SpeculativeVerify` dispatch workload and the sequential GDN
/// recurrence. `PromptFill` is an ordinary prompt prefill whose hidden rows
/// also seed a draft head (MTP prompt fill): it selects exactly the kernels a
/// plain AR prefill of the same chunk selects, so the capture adds output
/// rows and changes nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HiddenCapture {
    Verify,
    PromptFill,
}

/// Per-token hidden destination: buffer, first destination row, capture kind.
pub(crate) type HiddenRowsOut<'a> = (&'a GpuTensor, usize, HiddenCapture);

#[inline]
fn captures_verify_hidden(per_token_hidden_out: Option<HiddenRowsOut<'_>>) -> bool {
    per_token_hidden_out.is_some_and(|(_, _, capture)| capture == HiddenCapture::Verify)
}

#[inline]
fn prefill_dispatch_workload(
    captures_verify_hidden: bool,
    captures_rollback_tape: bool,
    is_tree_verify: bool,
) -> DispatchWorkload {
    if captures_verify_hidden || captures_rollback_tape || is_tree_verify {
        DispatchWorkload::SpeculativeVerify
    } else {
        DispatchWorkload::Standard
    }
}

pub(crate) fn forward_prefill_chunk(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    s: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
    hidden_rb: Option<&HiddenStateRingBuffer>,
    per_token_hidden_out: Option<HiddenRowsOut<'_>>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tape_offset: usize,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    pre_uploaded: bool,
    band: Option<&PrefillBandCtx<'_>>,
    mask_override: Option<MaskEmbedOverride<'_>>,
    needs_last_token_logits: bool,
    max_layer: Option<usize>,
    routed_out: Option<&GpuTensor>,
    fusion: DflashFusionCtx,
    commit_stride: Option<usize>,
) -> HipResult<()> {
    debug_assert!(
        commit_stride.is_none() || commit_stride == Some(WIDENED_COMMIT_ROWS),
        "commit_stride admits only None and Some(512)"
    );
    forward_batch_chunk_impl(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        s,
        pbs,
        hidden_rb,
        per_token_hidden_out,
        gdn_tape,
        tape_offset,
        tree_verify,
        pre_uploaded,
        false,
        band,
        mask_override,
        needs_last_token_logits,
        max_layer,
        routed_out,
        BatchSemantics::Sequential,
        fusion,
        commit_stride,
    )
}
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn batch_chunk_validate_independent(
    n: usize,
    batch_semantics: BatchSemantics<'_>,
    dn_state: &DeltaNetState,
    kv_cache: &llama::KvCache,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    gdn_tape: Option<&crate::speculative::GdnTape>,
) -> HipResult<()> {
    if let BatchSemantics::Independent {
        positions,
        lane_capacity,
        active_mask,
    } = batch_semantics
    {
        if positions.len() != n {
            return Err(HipError::new(
                0,
                "independent decode positions length must equal token batch length",
            ));
        }
        if positions.iter().any(|&p| p >= lane_capacity) {
            return Err(HipError::new(
                0,
                "independent decode position exceeds lane KV capacity",
            ));
        }
        // Fixed-slot positions are range-checked before any mutation, including inactive lanes,
        // so masked kernels cannot read outside the lane slice.
        let valid_n = valid_lane_mask(n)?;
        if active_mask & !valid_n != 0 {
            return Err(HipError::new(
                0,
                "independent decode active_mask has bits beyond batch size",
            ));
        }
        if dn_state.quant != StateQuant::Q8 || !kv_cache.quant_q8 {
            return Err(HipError::new(
                0,
                "independent decode currently requires Q8 DeltaNet state and Q8 KV",
            ));
        }
        if tree_verify.is_some() || gdn_tape.is_some() {
            return Err(HipError::new(
                0,
                "independent decode does not support tree or GDN tape modes",
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_chunk_embed_tokens(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    tokens: &[u32],
    s: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
    n: usize,
    dim: usize,
    dim_row_bytes: usize,
    do_embed: bool,
    pre_embedded: bool,
    pre_uploaded: bool,
    active_mask: Option<u64>,
    mask_override: Option<MaskEmbedOverride<'_>>,
) -> HipResult<()> {
    // ── 1. Embed tokens into pbs.x_batch ─────────────────────────────────
    //
    // Fast path for HFQ4G256 (all MQ4-quantized Qwen3.5 models + friends):
    // upload token ids to a device buffer and dispatch one batched kernel
    // that dequantizes N rows directly into `pbs.x_batch`. This collapses
    // 2N launches (N embed + N memcpy_dtod_at) into 1 upload + 1 launch
    // AND is hipGraph-captureable — the kernel reads token ids from a
    // device pointer instead of taking them as a baked-in scalar arg.
    //
    // Independent decode can leave lanes inactive. Restrict embedding to
    // active contiguous spans so an inactive lane's scratch activation is
    // not overwritten by the dummy token in the fixed-width input array.
    //
    // Other formats fall back to the per-token loop (kept for correctness
    // breadth; the MQ4-quantized hot path doesn't hit them).
    //
    // Multi-GPU band-mode: skip embedding when this is not the first band.
    // The activation already lives in `pbs.x_batch` from a peer-copy of
    // the previous band's `pbs.x_batch`.
    let embed_mask = partial_lane_mask(active_mask, n)?;
    if do_embed
        && !pre_embedded
        && matches!(
            weights.embd_format,
            EmbeddingFormat::HFQ4G256 | EmbeddingFormat::Q8_0
        )
    {
        if !pre_uploaded {
            if let Some(embed_mask) = embed_mask {
                for_each_active_span(embed_mask, n, |start, len| {
                    let tokens_host: Vec<i32> = tokens[start..start + len]
                        .iter()
                        .map(|&t| t as i32)
                        .collect();
                    let tokens_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, len * 4)
                    };
                    gpu.hip
                        .memcpy_htod_offset(&pbs.tokens.buf, start * 4, tokens_bytes)
                })?;
            } else {
                let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
                let tokens_bytes: &[u8] =
                    unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
                gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
            }
        }
        if let Some(embed_mask) = embed_mask {
            for_each_active_span(embed_mask, n, |start, len| {
                let output = pbs.x_batch.sub_offset(start * dim, len * dim);
                let token_ids = pbs.tokens.sub_offset(start, len);
                match weights.embd_format {
                    EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256_batched(
                        &weights.token_embd,
                        &output,
                        &token_ids,
                        len,
                        dim,
                    ),
                    EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8_batched(
                        &weights.token_embd,
                        &output,
                        &token_ids,
                        len,
                        dim,
                    ),
                    _ => unreachable!(),
                }
            })?;
        } else {
            match weights.embd_format {
                EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256_batched(
                    &weights.token_embd,
                    &pbs.x_batch,
                    &pbs.tokens,
                    n,
                    dim,
                )?,
                EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8_batched(
                    &weights.token_embd,
                    &pbs.x_batch,
                    &pbs.tokens,
                    n,
                    dim,
                )?,
                _ => unreachable!(),
            }
        }
    } else if do_embed && !pre_embedded {
        for (i, &tok) in tokens.iter().enumerate() {
            if embed_mask.is_some_and(|mask| (mask >> i) & 1 == 0) {
                continue;
            }
            match weights.embd_format {
                EmbeddingFormat::HFQ4G256 => unreachable!(),
                EmbeddingFormat::HFQ4G128 => {
                    gpu.embedding_lookup_hfq4g128(&weights.token_embd, &s.x, tok, dim)?
                }
                EmbeddingFormat::Q8_0 => {
                    gpu.embedding_lookup_q8(&weights.token_embd, &s.x, tok, dim)?
                }
                EmbeddingFormat::F32 => {
                    gpu.embedding_lookup(&weights.token_embd, &s.x, tok, dim)?
                }
                _ => panic!("unsupported embedding format"),
            }
            gpu.hip.memcpy_dtod_at(
                &pbs.x_batch.buf,
                i * dim_row_bytes,
                &s.x.buf,
                0,
                dim_row_bytes,
            )?;
        }
    }

    // ── 1a. Apply MaskEmbedOverride (MTP probe hook) ─────────────────────
    //
    // Overwrite a single batch slot's embedding row in `pbs.x_batch` after
    // the embedding-lookup kernel populated it but BEFORE the layer loop
    // (or any subsequent kernel) reads it. The Qualcomm MTP probe uses this
    // to replace the embedding-table value at a "mask token" position with
    // a prompt-mean vector. Default callers pass `None` → zero overhead.
    //
    // Multi-GPU band-mode: skip on non-first bands; pbs.x_batch already
    // holds the peer-copied activation from the previous band's
    // `pbs.x_batch`. Re-applying here would clobber the partial forward.
    if do_embed {
        if let Some(ovr) = mask_override {
            assert!(
                ovr.slot < n,
                "MaskEmbedOverride.slot ({}) must be < n ({})",
                ovr.slot,
                n,
            );
            assert_eq!(
                ovr.embed.len(),
                dim,
                "MaskEmbedOverride.embed.len() ({}) must equal config.dim ({})",
                ovr.embed.len(),
                dim,
            );
            if embed_mask.is_none_or(|mask| (mask >> ovr.slot) & 1 != 0) {
                let bytes: &[u8] =
                    unsafe { std::slice::from_raw_parts(ovr.embed.as_ptr() as *const u8, dim * 4) };
                let offset = ovr.slot * dim_row_bytes;
                gpu.hip
                    .memcpy_htod_offset(&pbs.x_batch.buf, offset, bytes)?;
            }
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_chunk_upload_positions(
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    batch_semantics: BatchSemantics<'_>,
    start_pos: usize,
    n: usize,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    pre_uploaded: bool,
) -> HipResult<()> {
    // ── 1b. Upload positions array ────────────────────────────────────────
    //
    // Positions is the per-row RoPE angle AND the physical KV cache slot (the
    // batched kv_write kernels use the same index for both). We always use
    // flat linear `start_pos .. start_pos + n`. Siblings in DDTree mode get
    // DISTINCT slots this way — no write race — and the stored K carries a
    // RoPE angle that matches the physical slot, which keeps subsequent
    // cycles' attention reads consistent.
    //
    // Semantic trade vs. the original depth-based scheme (paper): tree
    // siblings that represent "alternative futures at the same time step"
    // now see a RoPE distance of 1 (or more) instead of 0. Empirically that
    // slight distance shift costs little — the attn_bias mask still gates
    // ancestor visibility exactly, and the Q·K dot products stay consistent
    // across the whole cache (prompt + tree block). In exchange we get
    // DDTree correctness for topk>1 without needing a tree-local KV scratch
    // or a scatter-kernel for commit. `ctx.positions` is accepted for API
    // compatibility but ignored — the DdNode depths it carries are only
    // used by `linearize_tree` to build the attn_bias mask.
    //
    // 39aa358 DECOUPLING (2026-07-28): the "costs little" trade was wrong
    // — it was the gfx1100 DDTree regression. Sibling logits computed with
    // slot-based phases depress non-rank-0 acceptance, collapsing τ toward
    // the chain and making tree strictly slower than linear. Fix: upload
    // `ctx.positions` (depth-based, `base_pos + depth`) to
    // `pbs.rope_positions`; FA RoPE reads THAT (correct sibling phases),
    // while KV writes + attention seq_len keep the flat physical slots
    // (no sibling write race, contiguous-cache invariants intact).
    if !pre_uploaded {
        let (positions_host, active_mask) = match batch_semantics {
            BatchSemantics::Sequential => (
                (0..n).map(|i| (start_pos + i) as i32).collect::<Vec<_>>(),
                None,
            ),
            BatchSemantics::Independent {
                positions,
                active_mask,
                ..
            } => {
                debug_assert_eq!(positions.len(), n);
                (
                    positions.iter().map(|&p| p as i32).collect::<Vec<_>>(),
                    Some(active_mask),
                )
            }
        };
        if let Some(active_mask) = partial_lane_mask(active_mask, n)? {
            for_each_active_span(active_mask, n, |start, len| {
                let positions_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        positions_host[start..start + len].as_ptr() as *const u8,
                        len * 4,
                    )
                };
                gpu.hip
                    .memcpy_htod_offset(&pbs.positions.buf, start * 4, positions_bytes)?;
                if let Some(tv) = tree_verify.as_ref() {
                    debug_assert_eq!(tv.positions.len(), n, "tree RoPE positions length");
                    let rope_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            tv.positions[start..start + len].as_ptr() as *const u8,
                            len * 4,
                        )
                    };
                    gpu.hip
                        .memcpy_htod_offset(&pbs.rope_positions.buf, start * 4, rope_bytes)?;
                }
                Ok(())
            })?;
        } else {
            let positions_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
            gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
            if let Some(tv) = tree_verify.as_ref() {
                debug_assert_eq!(tv.positions.len(), n, "tree RoPE positions length");
                let rope_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(tv.positions.as_ptr() as *const u8, n * 4)
                };
                gpu.hip.memcpy_htod(&pbs.rope_positions.buf, rope_bytes)?;
            }
        }
    }

    Ok(())
}

/// Context length past which an admitted Q8 small-batch attend step leaves
/// the batched masked FA kernel for the multi-row tile. Measured with the
/// Qwen3.8-27B verify shape (`bench_flash_rows`, tile 128); gfx1100 and
/// gfx1201 keep the conservative 4k boundary by default. gfx1151 is opt-in:
/// it takes the route only when `HIPFIRE_FA_PERTOKEN_MIN_CTX` is set.
/// `HIPFIRE_FA_PERTOKEN_MIN_CTX` overrides; `0` disables the route.
pub(crate) fn fa_pertoken_min_ctx(arch: &str) -> Option<usize> {
    static EXPLICIT: std::sync::LazyLock<Option<usize>> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_FA_PERTOKEN_MIN_CTX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
    });
    fa_pertoken_min_ctx_for(arch, *EXPLICIT)
}

fn fa_pertoken_min_ctx_for(arch: &str, explicit: Option<usize>) -> Option<usize> {
    match explicit {
        Some(v) => (v > 0).then_some(v),
        None if arch == "gfx1151" => None,
        None => Some(4_096),
    }
}

#[allow(clippy::too_many_arguments)]
fn q8_multirow_attn_admitted(
    arch: &str,
    quant_q8: bool,
    head_dim: usize,
    n: usize,
    logical_ctx: usize,
    min_ctx: Option<usize>,
    is_tree: bool,
    is_independent: bool,
    capture_mode: bool,
    replay_recording: bool,
) -> bool {
    matches!(arch, "gfx1100" | "gfx1151" | "gfx1201")
        && quant_q8
        && matches!(head_dim, 128 | 256)
        && (4..=32).contains(&n)
        && min_ctx.is_some_and(|threshold| logical_ctx > threshold)
        && !is_tree
        && !is_independent
        && !capture_mode
        && !replay_recording
}

#[allow(clippy::too_many_arguments)]
fn batch_chunk_full_attn_fallback(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    layer_idx: usize,
    kv_layer_idx: usize,
    start_pos: usize,
    n: usize,
    dim_row_bytes: usize,
    kv_cache: &mut llama::KvCache,
    s: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
) -> HipResult<()> {
    // Per-token gather/scatter fallback for FA layers that don't
    // qualify for batched FA (non-MQ4 weights, non-Q8_0 KV, etc).
    for i in 0..n {
        let pos = start_pos + i;
        gpu.hip.memcpy_dtod_at(
            &s.x.buf,
            0,
            &pbs.x_batch.buf,
            i * dim_row_bytes,
            dim_row_bytes,
        )?;
        let pos_i32 = pos as i32;
        gpu.memcpy_htod_auto(&s.pos_buf, &pos_i32.to_ne_bytes())?;
        run_fa_layer_body(
            gpu,
            weights,
            config,
            layer_idx,
            kv_layer_idx,
            pos,
            kv_cache,
            s,
        )?;
        gpu.hip.memcpy_dtod_at(
            &pbs.x_batch.buf,
            i * dim_row_bytes,
            &s.x.buf,
            0,
            dim_row_bytes,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn batch_chunk_delta_net_moe(
    gpu: &mut Gpu,
    layer: &DeltaNetMoeLayerWeights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    dn_state: &mut DeltaNetState,
    n: usize,
    dim: usize,
    hidden_dim: usize,
    k_dim: usize,
    v_dim: usize,
    n_v_heads: usize,
    hd: usize,
    batch_semantics: BatchSemantics<'_>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    gdn_tape: Option<&crate::speculative::GdnTape>,
    tape_offset: usize,
    delta_layer_idx: usize,
    start_pos: usize,
    layer_idx: usize,
    ctx: &DispatchCtx,
    weights: &Qwen35Weights,
    routed_out: Option<&GpuTensor>,
    route: PrefillRouteMode<'_>,
) -> HipResult<()> {
    let tape = gdn_tape.map(gdn_tape_view);
    execute_deltanet_moe_layer_batched(
        gpu,
        &deltanet_moe_layer_view(layer),
        &super::program::hybrid_dims(config),
        &deltanet_scratch(pbs),
        &deltanet_state_view(dn_state),
        n,
        dim,
        k_dim,
        v_dim,
        n_v_heads,
        hd,
        batch_semantics,
        tree_verify,
        tape.as_ref(),
        tape_offset,
        delta_layer_idx,
        start_pos,
        layer_idx,
    )?;
    // Batched MoE FFN replaces the dense (rmsnorm + gate+up +
    // silu_mul + w_down) block. Takes pbs.x_batch as input AND
    // accumulates the FFN output residual back into it via the
    // batched indexed down kernel's atomicAdd path.
    prefill_moe_ffn_body_batched_with_route(
        gpu,
        &layer.ffn,
        &layer.ffn_norm,
        config,
        pbs,
        n,
        &ctx,
        weights.moe_has_mq6,
        routed_out,
        route,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn batch_chunk_full_attn_moe(
    gpu: &mut Gpu,
    fa_attn_multirow: bool,
    layer: &FullAttnMoeLayerWeights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    kv_cache: &llama::KvCache,
    n: usize,
    dim: usize,
    start_pos: usize,
    max_ctx_len: usize,
    ctx: &DispatchCtx,
    batch_semantics: BatchSemantics<'_>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    q8_wmma_arch: bool,
    arch_has_wmma: bool,
    kv_layer_idx: usize,
    layer_idx: usize,
    routed_out: Option<&GpuTensor>,
    weights: &Qwen35Weights,
    route: PrefillRouteMode<'_>,
) -> HipResult<()> {
    // F2 split: QKV projection + norms + RoPE (everything above the old
    // attend call) live in `batch_chunk_full_attn_moe_prep`; called here
    // for the single-chunk path in the original position.
    batch_chunk_full_attn_moe_prep(
        gpu,
        layer,
        config,
        pbs,
        n,
        dim,
        kv_cache,
        tree_verify,
        layer_idx,
    )?;
    // F2 split: KV-write + flash attention, sigmoid/wo and the MoE FFN now
    // live in `batch_chunk_full_attn_moe_finish` so a chunk pair can share
    // one merged attend step; called here for the single-chunk path in the
    // original position.
    batch_chunk_full_attn_moe_finish(
        gpu,
        fa_attn_multirow,
        layer,
        config,
        pbs,
        s,
        kv_cache,
        n,
        start_pos,
        max_ctx_len,
        ctx,
        batch_semantics,
        tree_verify,
        q8_wmma_arch,
        layer_idx,
        weights,
        routed_out,
        route,
    )?;
    Ok(())
}

/// F2 split of `batch_chunk_full_attn_moe`: QKV projection (all dtype arms)
/// + deinterleave/Q-norm/K-norm + triattn tap + RoPE. Same statements, same
/// order, same launches as the inlined head. The pair path calls this per
/// half, then runs one merged attend step, then the finish per half.
#[allow(clippy::too_many_arguments)]
fn batch_chunk_full_attn_moe_prep(
    gpu: &mut Gpu,
    layer: &FullAttnMoeLayerWeights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    n: usize,
    dim: usize,
    kv_cache: &llama::KvCache,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    layer_idx: usize,
) -> HipResult<()> {
    let tap = attention_tap(layer_idx, config);
    attention_moe_prepare_batched(
        gpu,
        &fa_moe_attention_weights(layer),
        &super::program::hybrid_dims(config),
        &attention_scratch(pbs),
        n,
        dim,
        kv_cache.compact_offset,
        tree_verify,
        layer_idx,
        tap.as_deref(),
    )
}

/// F2 split of `batch_chunk_full_attn_moe`: KV-write + flash attention,
/// sigmoid/wo residual and the batched MoE FFN. Same statements, same order,
/// same launches as the inlined tail. The pair path calls the prep head
/// (everything above the old attend call) for both halves, runs one merged
/// attend step, then this finish per half.
#[allow(clippy::too_many_arguments)]
fn batch_chunk_full_attn_moe_finish(
    gpu: &mut Gpu,
    fa_attn_multirow: bool,
    layer: &FullAttnMoeLayerWeights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    kv_cache: &llama::KvCache,
    n: usize,
    start_pos: usize,
    max_ctx_len: usize,
    ctx: &DispatchCtx,
    batch_semantics: BatchSemantics<'_>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    q8_wmma_arch: bool,
    layer_idx: usize,
    weights: &Qwen35Weights,
    routed_out: Option<&GpuTensor>,
    route: PrefillRouteMode<'_>,
) -> HipResult<()> {
    // Batched KV write + flash attention (via dispatch).
    attention_attend_batched(
        gpu,
        &super::program::hybrid_dims(config),
        &attention_scratch(pbs),
        &flash_scratch(s),
        &kv_view(kv_cache),
        n,
        start_pos,
        max_ctx_len,
        ctx,
        batch_semantics,
        tree_verify,
        layer_idx,
        fa_attn_multirow,
        None,
        // commit_stride: MoE finish keeps legacy cadence
        false,
    )?;
    attention_moe_output_projection_batched(
        gpu,
        &fa_moe_attention_weights(layer),
        &attention_scratch(pbs),
        n,
        q8_wmma_arch,
        layer_idx,
    )?;

    // Batched MoE FFN.
    prefill_moe_ffn_body_batched_with_route(
        gpu,
        &layer.ffn,
        &layer.ffn_norm,
        config,
        pbs,
        n,
        &ctx,
        weights.moe_has_mq6,
        routed_out,
        route,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn batch_chunk_final_logits(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
    n: usize,
    dim: usize,
    dim_row_bytes: usize,
    per_token_hidden_out: Option<HiddenRowsOut<'_>>,
    needs_last_token_logits: bool,
    do_lm_head: bool,
    ctx: &DispatchCtx,
) -> HipResult<()> {
    // ── 3. Final output norm + logits ───────────────────────────────────
    // Multi-GPU band-mode: skip when this is not the last band — the
    // running activation in `pbs.x_batch` is what the next band's
    // peer-copy reads. `weights.output_norm` and `weights.output` only
    // live on the last band's device anyway.
    if do_lm_head {
        // If the caller requested per-token hidden output (DFlash verify path),
        // run rmsnorm over all N rows into their buffer. Otherwise use the
        // legacy last-token-only path.
        if let Some((dst, offset_rows, _)) = per_token_hidden_out {
            let dst_view = dst.sub_offset(offset_rows * dim, n * dim);
            gpu.rmsnorm_batched(
                &pbs.x_batch,
                &weights.output_norm,
                &dst_view,
                n,
                dim,
                config.norm_eps,
            )?;
            if prefill_should_emit_last_token_logits(true, needs_last_token_logits) {
                // Still populate s.logits with the last-token logits for
                // callers that rely on it (the legacy prefill post-condition).
                let last = n - 1;
                let last_view = dst.sub_offset((offset_rows + last) * dim, dim);
                {
                    let wr = weights.output.dispatch_ref();
                    let step = Step::Gemv {
                        w: &wr,
                        input: GemvInput::Raw(&last_view),
                        out: &s.logits,
                    };
                    execute_steps(gpu, &ctx, &[step])
                        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                }
            }
        } else {
            // Legacy path: only last-token logits.
            // Use _auto so the D→D copy routes through the active stream
            // during hipGraph capture (bare memcpy_dtod_at uses the legacy
            // null stream and breaks capture: HIP error 906).
            let last = n - 1;
            gpu.memcpy_dtod_at_auto(
                &s.x.buf,
                0,
                &pbs.x_batch.buf,
                last * dim_row_bytes,
                dim_row_bytes,
            )?;
            gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
            {
                let wr = weights.output.dispatch_ref();
                let step = Step::Gemv {
                    w: &wr,
                    input: GemvInput::Raw(&s.tmp),
                    out: &s.logits,
                };
                execute_steps(gpu, &ctx, &[step])
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
            }
        }
    }

    Ok(())
}

fn restore_inactive_residual_rows(
    gpu: &Gpu,
    pbs: &PrefillBatchScratch,
    backup: &GpuTensor,
    inactive_mask: u64,
    n: usize,
    dim: usize,
) -> HipResult<()> {
    for_each_active_span(inactive_mask, n, |start, len| {
        let start_elems = start
            .checked_mul(dim)
            .ok_or_else(|| HipError::new(0, "inactive residual offset overflow"))?;
        let bytes = len
            .checked_mul(dim)
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| HipError::new(0, "inactive residual span overflow"))?;
        let offset = start_elems
            .checked_mul(4)
            .ok_or_else(|| HipError::new(0, "inactive residual byte offset overflow"))?;
        gpu.memcpy_dtod_at_auto(&pbs.x_batch.buf, offset, &backup.buf, offset, bytes)
    })
}

// ── F2 N1024 pair envelope ──────────────────────────────────────────────
// On exact gfx1151, two complete consecutive 512-row prefill chunks for the
// same request share ONE FA2 launch over 1024 query rows instead of two 512
// launches (F1-oracle-proven bit-exact over identical complete KV, 6–8%
// faster at L8192/L32768). Everything except the FA layers' attention call
// stays on the per-512 chunk pipeline: GDN layers keep their per-512 state
// cadence (half C fully before half N per layer), projections/norms/FFN run
// per 512-row half out of two 512-row PBS instances, and only the FA2
// kernel sees 1024 rows (contiguous staged Q/K/V/positions plus staged out,
// split back into the halves afterwards).
//
// Bit-identity argument: the merged `Step::Attend` derives the same tier
// plan as the halves (derivation is batch-size agnostic beyond batch > 1)
// and dispatch routes it to the identical FA2 kernel (the widened
// exactly-1024 gfx1151 gates) with identical inputs — the same Q/K/V bytes
// the halves' own prep produced, the same positions, the same caches — so
// F1's one-launch-vs-two-halves proof applies verbatim. The dispatch write
// runs first inside the same step on the same stream, so both halves' KV
// rows (keys up to start + 1023) are fully written before the FA2 body
// reads them; positions for rows 512..1024 refer to those just-written
// keys. The F16 Q pre-convert scratch is Gpu-owned and scales with the
// launch batch (`batch * 24 * 256 * 2` bytes), so 1024 rows are covered.
// The merged path is eager-only: changing the launch count would
// invalidate graph capture/replay.
//
// Envelope (anything outside falls back to sequential 512 chunks): exact
// gfx1151, eager, sequential single-request prefill, both chunks exactly
// 512 rows, H24/KV4/D256, KV tier Q8 or fwht3-Asym3, no tree/tape/band/
// max_layer, no hidden ring (its per-chunk staging commit is
// chunk-sequential), DFlash fusion Off.
// DeltaNet (dense LA + MoE-LA) sequential kernels stay 2x512 inside the
// pair — deliberately NOT merged (pair-width analysis, 2026-09):
// - `gated_delta_net_q8_batch_seq` (default fast kernel): single-end
//   requant with a `frame + lane * n_tokens` stochastic seed (lane = 0 on
//   the sequential path, grid.z = 1). One 1024-row launch would drop the
//   token-512 Q8 round-trip (including the EF residual commit) and reseed
//   the final requant (F vs F+512), so rows 512..1023 and the final
//   S_q8/scales/EF diverge from 2x512 — an eval-md5 break. No 512 cap
//   exists (any n_tokens runs; LDS is n-independent); the blocker is
//   requant semantics, fixable only by mid-boundary requant kernel
//   surgery, which is declined without GPU verification. (Under
//   HIPFIRE_DN_REQUANT_PER_TOKEN=1 the slow kernel's per-token
//   `frame + bt` seeds read pair-continuous at lane 0, so a merged launch
//   would be identical there — but that path is ~1.8x slower and off-gate;
//   still unmerged.)
// - `conv1d_silu_split_f32_n`: generic n_tokens loop, so 1x1024 over a
//   staged concatenation IS trajectory-identical (same per-channel op
//   order, continuous ring) — but staging costs ~50 MB/layer (1024x6144x4 B
//   qkv in + q/k/v raws out at 16/16x128) to save a single launch, which
//   is net-negative against launch overhead. Stays split on perf.
// - sigmoid / QK-norm / gated_norm glue: row-parallel, batch-agnostic per
//   row; merging saves only launch overhead at staging-copy cost. Stay.
// MoE-LA expert FFN (row-parallel grouped GEMM) is out of scope: not a
// sequential-recurrence kernel, same copy-trap economics as the glue.

/// Rows per F2 pair half. Only two complete halves ever merge — never a
/// partial chunk.
pub(crate) const FA_PAIR_ROWS: usize = 512;
/// Merged FA2 batch for one F2 pair.
pub(crate) const FA_PAIR_BATCH: usize = 1024;

/// Pure F2 pairing decision: which consecutive chunk lengths merge, on which
/// arch, under which capture/replay state. CPU-only, unit-tested.
fn fa_pair_merge_admitted(
    chunk_a: usize,
    chunk_b: usize,
    arch: &str,
    capture_mode: bool,
    replay_recording: bool,
) -> bool {
    chunk_a == FA_PAIR_ROWS
        && chunk_b == FA_PAIR_ROWS
        && arch == "gfx1151"
        && !capture_mode
        && !replay_recording
}

/// Runtime half of the F2 guard: everything the pure pairing decision cannot
/// see. Each predicate mirrors the FA2 route the two halves would take
/// through dispatch, so the merged 1024-row launch selects the identical
/// kernel (see `fa2_gfx11_batch_admitted`).
fn fa_pair_env_admitted(
    gpu: &Gpu,
    config: &Qwen35Config,
    kv_cache: &llama::KvCache,
    max_ctx_end: usize,
    fusion: DflashFusionCtx,
) -> bool {
    if gpu.arch.as_str() != "gfx1151" {
        return false;
    }
    if config.n_heads != 24 || config.n_kv_heads != 4 || config.head_dim != 256 {
        return false;
    }
    if !matches!(fusion, DflashFusionCtx::Off) {
        return false;
    }
    // Same window the second half would see through ingress/dispatch.
    if !gpu.fa2_gfx11_ctx_admitted(max_ctx_end) {
        return false;
    }
    if !gpu.flags.gfx11_fa2_prefill {
        return false;
    }
    // No 1024-vs-2x512 bit-identity proof exists for CK tiling.
    if gpu.flash_attn_ck_loaded() {
        return false;
    }
    let tier_inputs = kv_cache.tier_inputs();
    match hipfire_dispatch::families::kv_tier::classify(
        tier_inputs.quant_q8,
        tier_inputs.quant_asym4,
        tier_inputs.quant_asym3,
        tier_inputs.quant_asym2,
        tier_inputs.quant_hfq4,
        tier_inputs.quant_q4,
        tier_inputs.quant_int8,
        tier_inputs.quant_hfq8,
        tier_inputs.quant_fwht,
        tier_inputs.quant_bf16,
        tier_inputs.quant_fp8,
    ) {
        hipfire_dispatch::families::kv_tier::KTier::Q8 => {
            // The halves take the WMMA/flash-prefill route into ingress;
            // require exactly that (same env read as dispatch, default-on
            // for gfx11). Windowed Q8 (non-Qwen) has no FA2 arm.
            if tier_inputs.q8_windowed {
                return false;
            }
            let flash_optin = match hipfire_config::developer_var("HIPFIRE_FLASH_PREFILL")
                .ok()
                .as_deref()
            {
                Some("0") | Some("off") | Some("false") => false,
                Some("1") | Some("on") | Some("true") => true,
                _ => gpu.arch.starts_with("gfx11"),
            };
            if !flash_optin {
                return false;
            }
            gpu.arch_caps.has_wmma_w32() || gpu.arch_caps.has_wmma_w32_gfx12()
        }
        hipfire_dispatch::families::kv_tier::KTier::Asym3 { fwht: true } => {
            // fwht3-K FA2 arm: V must be Q8_0 and the givens tables present.
            tier_inputs.v_mode_bits == 8
                && kv_cache.givens_cos.is_some()
                && kv_cache.givens_sin.is_some()
        }
        _ => false,
    }
}

/// Device-side staging for one F2 pair's merged FA2 step: contiguous
/// 1024-row Q/K/V/positions inputs plus the 1024-row output. The halves'
/// own prep writes into their own PBS tensors; the merged step reads the
/// staged concatenation and the halves' outputs are copied back afterwards.
/// Allocated once per prefill call that forms at least one pair and shared
/// by every FA layer of every pair (never per-layer).
struct FaPairStage {
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    pos: GpuTensor,
    out: GpuTensor,
}

impl FaPairStage {
    fn alloc(gpu: &mut Gpu, config: &Qwen35Config) -> HipResult<Self> {
        let pair = FA_PAIR_BATCH;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;
        // Transactional: free whatever was allocated if a later alloc fails.
        let mut done: Vec<GpuTensor> = Vec::with_capacity(5);
        macro_rules! stage {
            ($shape:expr) => {{
                match gpu.alloc_tensor($shape, DType::F32) {
                    Ok(t) => {
                        done.push(t);
                    }
                    Err(e) => {
                        for t in done.drain(..) {
                            let _ = gpu.free_tensor(t);
                        }
                        return Err(e);
                    }
                }
            }};
        }
        stage!(&[pair * q_dim]);
        stage!(&[pair * kv_dim]);
        stage!(&[pair * kv_dim]);
        stage!(&[pair]);
        stage!(&[pair * q_dim]);
        let mut it = done.into_iter();
        let mut take = || it.next().expect("FaPairStage slot");
        Ok(Self {
            q: take(),
            k: take(),
            v: take(),
            pos: take(),
            out: take(),
        })
    }

    fn free_gpu(self, gpu: &mut Gpu) -> HipResult<()> {
        let mut first_err: Option<HipError> = None;
        for t in [self.q, self.k, self.v, self.pos, self.out] {
            if let Err(e) = gpu.free_tensor(t) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// One merged FA2 step for an F2 pair: stage both halves' Q/K/V/positions
/// contiguously, run the shared write-then-attend dispatch once at batch
/// 1024, split the outputs back into the halves' own PBS tensors.
/// `start_c` is the first half's absolute start; the merged max_ctx is
/// `start_c + 1024` (the value the second half would see).
#[allow(clippy::too_many_arguments)]
fn batch_chunk_fa_attend_merged(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    pbs_c: &PrefillBatchScratch,
    pbs_n: &PrefillBatchScratch,
    stage: &FaPairStage,
    s: &Qwen35Scratch,
    kv_cache: &llama::KvCache,
    start_c: usize,
    max_ctx_len: usize,
    ctx: &DispatchCtx,
    layer_idx: usize,
) -> HipResult<()> {
    let q_dim = config.n_heads * config.head_dim;
    let kv_dim = config.n_kv_heads * config.head_dim;
    let half_q_bytes = FA_PAIR_ROWS * q_dim * 4;
    let half_kv_bytes = FA_PAIR_ROWS * kv_dim * 4;
    let half_pos_bytes = FA_PAIR_ROWS * 4;
    // Same stream throughout: no sync is needed between the staging copies
    // and the attend step, or between the attend step and the split copies.
    gpu.hip
        .memcpy_dtod_at(&stage.q.buf, 0, &pbs_c.fa_q_batch.buf, 0, half_q_bytes)?;
    gpu.hip.memcpy_dtod_at(
        &stage.q.buf,
        half_q_bytes,
        &pbs_n.fa_q_batch.buf,
        0,
        half_q_bytes,
    )?;
    gpu.hip
        .memcpy_dtod_at(&stage.k.buf, 0, &pbs_c.fa_k_batch.buf, 0, half_kv_bytes)?;
    gpu.hip.memcpy_dtod_at(
        &stage.k.buf,
        half_kv_bytes,
        &pbs_n.fa_k_batch.buf,
        0,
        half_kv_bytes,
    )?;
    gpu.hip
        .memcpy_dtod_at(&stage.v.buf, 0, &pbs_c.fa_v_batch.buf, 0, half_kv_bytes)?;
    gpu.hip.memcpy_dtod_at(
        &stage.v.buf,
        half_kv_bytes,
        &pbs_n.fa_v_batch.buf,
        0,
        half_kv_bytes,
    )?;
    gpu.hip
        .memcpy_dtod_at(&stage.pos.buf, 0, &pbs_c.positions.buf, 0, half_pos_bytes)?;
    gpu.hip.memcpy_dtod_at(
        &stage.pos.buf,
        half_pos_bytes,
        &pbs_n.positions.buf,
        0,
        half_pos_bytes,
    )?;
    execute_fa_attend_step(
        gpu,
        &super::program::hybrid_dims(config),
        &stage.q,
        &stage.k,
        &stage.v,
        &stage.pos,
        &stage.out,
        &flash_scratch(s),
        &kv_view(kv_cache),
        FA_PAIR_BATCH,
        start_c,
        max_ctx_len,
        ctx,
        None,
        layer_idx,
        None,
    )?;
    gpu.hip.memcpy_dtod_at(
        &pbs_c.fa_attn_out_batch.buf,
        0,
        &stage.out.buf,
        0,
        half_q_bytes,
    )?;
    gpu.hip.memcpy_dtod_at(
        &pbs_n.fa_attn_out_batch.buf,
        0,
        &stage.out.buf,
        half_q_bytes,
        half_q_bytes,
    )?;
    Ok(())
}

/// F2 pair chunk: run two complete consecutive 512-row chunks of one request
/// through the layer stack lockstep, merging only the FA layers' attention
/// call into one 1024-row FA2 launch per FA layer.
///
/// Per-layer order matches two sequential chunks exactly: for GDN/MoE-LA
/// layers half C runs fully before half N (recurrence cadence deliberately
/// unmerged — see the F2 envelope above for the per-kernel analysis);
/// for FA layers both halves' QKV prep (projection + norms + RoPE) runs
/// first, then one merged write-then-attend step, then each half's output
/// projection/FFN. `dn_state`, the KV cache and `per_token_hidden_out` see
/// the same row order as sequential chunks; each half keeps its own PBS
/// (no cross-half scratch aliasing).
///
/// Callers guarantee the F2 envelope (see `fa_pair_env_admitted`); in
/// particular `hidden_rb` is None (its per-chunk staging commit is
/// chunk-sequential), `gdn_tape`/`tree_verify`/band are absent, and both
/// halves are exactly `FA_PAIR_ROWS`.
#[allow(clippy::too_many_arguments)]
fn forward_prefill_chunk_pair(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens_c: &[u32],
    tokens_n: &[u32],
    start_pos: usize,
    chunk_start_c: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    s: &Qwen35Scratch,
    pbs_c: &PrefillBatchScratch,
    pbs_n: &PrefillBatchScratch,
    stage: &FaPairStage,
    per_token_hidden_out: Option<(&GpuTensor, HiddenCapture)>,
    mask_override: Option<MaskEmbedOverride<'_>>,
    needs_last_token_logits: bool,
    fusion: DflashFusionCtx,
) -> HipResult<()> {
    let n = FA_PAIR_ROWS;
    debug_assert!(tokens_c.len() == n && tokens_n.len() == n);
    debug_assert!(pbs_c.max_batch >= n && pbs_n.max_batch >= n);
    let start_c = start_pos + chunk_start_c;
    let start_n = start_c + n;
    let max_ctx_c = start_c + n;
    let max_ctx_n = start_n + n;
    let max_ctx_merged = start_c + 2 * n;
    kv_cache.require_mapped_capacity(checked_kv_end(
        start_c,
        2 * n,
        "forward_prefill_chunk_pair",
    )?)?;

    let dim = config.dim;
    let hidden_dim = config.hidden_dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    let dim_row_bytes = dim * 4;
    let dispatch_workload = prefill_dispatch_workload(
        per_token_hidden_out.is_some_and(|(_, capture)| capture == HiddenCapture::Verify),
        false,
        false,
    );
    let ctx = DispatchCtx::new(gpu).with_workload(dispatch_workload);

    // Mask slot is call-relative: rebase per half.
    let mo_c = mask_override.as_ref().and_then(|ovr| {
        if ovr.slot >= chunk_start_c && ovr.slot < chunk_start_c + n {
            Some(MaskEmbedOverride {
                slot: ovr.slot - chunk_start_c,
                embed: ovr.embed,
            })
        } else {
            None
        }
    });
    let mo_n = mask_override.as_ref().and_then(|ovr| {
        if ovr.slot >= chunk_start_c + n && ovr.slot < chunk_start_c + 2 * n {
            Some(MaskEmbedOverride {
                slot: ovr.slot - chunk_start_c - n,
                embed: ovr.embed,
            })
        } else {
            None
        }
    });

    batch_chunk_embed_tokens(
        gpu,
        weights,
        tokens_c,
        s,
        pbs_c,
        n,
        dim,
        dim_row_bytes,
        true,
        false,
        false,
        None,
        mo_c,
    )?;
    batch_chunk_upload_positions(
        gpu,
        pbs_c,
        BatchSemantics::Sequential,
        start_c,
        n,
        None,
        false,
    )?;
    batch_chunk_embed_tokens(
        gpu,
        weights,
        tokens_n,
        s,
        pbs_n,
        n,
        dim,
        dim_row_bytes,
        true,
        false,
        false,
        None,
        mo_n,
    )?;
    batch_chunk_upload_positions(
        gpu,
        pbs_n,
        BatchSemantics::Sequential,
        start_n,
        n,
        None,
        false,
    )?;

    #[cfg(feature = "moe-oracle")]
    crate::qwen35::oracle::set_prefill_start(start_c)
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    #[cfg(feature = "moe-oracle")]
    crate::qwen35::oracle::set_prefill_start(start_n)
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    // Once per half, exactly as two sequential chunks would validate.
    batch_chunk_validate_independent(
        n,
        BatchSemantics::Sequential,
        dn_state,
        kv_cache,
        None,
        None,
    )?;
    batch_chunk_validate_independent(
        n,
        BatchSemantics::Sequential,
        dn_state,
        kv_cache,
        None,
        None,
    )?;

    let fa_arch = gpu.arch.as_str();
    let q8_wmma_arch = q8_prefill_wmma_enabled(gpu);
    let arch_has_wmma = q8_wmma_arch;
    // Slice-B (FA2 fp8 fill) admission: native fp8 KV has complete batched
    // write/attend keys (KvWriteFp8E4m3Batched/AttnFp8E4m3KvBatchedMasked),
    // so it takes the batched FA path like the other tiers. BF16 stays out.
    let fa_batched_ok = (kv_cache.quant_q8
        || kv_cache.quant_asym4
        || kv_cache.quant_asym3
        || kv_cache.quant_asym2
        || kv_cache.quant_fp8)
        && weights.layers.iter().all(|lw| match lw {
            LayerWeights::FullAttn(_) | LayerWeights::FullAttnMoe(_) => {
                qwen35_layer_batch_admissible(lw, config, fa_arch).is_ok()
            }
            _ => true,
        });
    // Merge only when both halves take the plain dispatch route. Multirow
    // never admits a 512-row half (its window is 4..=32 rows); the per-half
    // fallback below keeps this fail-closed if that ever changes.
    let multirow_common = (
        gpu.arch_caps.arch(),
        kv_cache.quant_q8,
        config.head_dim,
        fa_pertoken_min_ctx(gpu.arch_caps.arch()),
        gpu.graphs.capture_mode,
        gpu.replay.is_recording(),
    );
    let multirow_admitted = |max_ctx: usize| {
        q8_multirow_attn_admitted(
            multirow_common.0,
            multirow_common.1,
            multirow_common.2,
            n,
            max_ctx,
            multirow_common.3,
            false,
            false,
            multirow_common.4,
            multirow_common.5,
        )
    };
    let multirow_c = multirow_admitted(max_ctx_c);
    let multirow_n = multirow_admitted(max_ctx_n);
    let merge_fa = !multirow_c && !multirow_n;

    let mut delta_layer_idx = 0usize;
    let mut kv_layer_idx = 0usize;
    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                for (half_pbs, half_start, half_tape) in [
                    (pbs_c, start_c, chunk_start_c),
                    (pbs_n, start_n, chunk_start_c + n),
                ] {
                    batch_chunk_delta_net_attn(
                        gpu,
                        layer,
                        config,
                        half_pbs,
                        dn_state,
                        n,
                        dim,
                        k_dim,
                        v_dim,
                        n_v_heads,
                        hd,
                        BatchSemantics::Sequential,
                        None,
                        None,
                        half_tape,
                        delta_layer_idx,
                        q8_wmma_arch,
                        arch_has_wmma,
                        BatchEpilogue::Residual,
                        fusion,
                        None,  // commit_stride: pair halves keep legacy cadence
                        false, // Paired halves retain the incumbent route
                    )?;
                    batch_chunk_dense_ffn(
                        gpu,
                        layer.dense_ffn(),
                        config,
                        half_pbs,
                        n,
                        dim,
                        hidden_dim,
                        q8_wmma_arch,
                        BatchEpilogue::Residual,
                        fusion,
                    )?;
                    dump_hidden_localize(
                        gpu,
                        &half_pbs.x_batch,
                        n,
                        half_start,
                        dim,
                        layer_idx,
                        "batched",
                    );
                }
                delta_layer_idx += 1;
            }
            (LayerWeights::FullAttn(layer), LayerType::FullAttention) if fa_batched_ok => {
                if merge_fa {
                    attention_input_projection_batched(
                        gpu,
                        &fa_attention_weights(layer),
                        &super::program::hybrid_dims(config),
                        &attention_scratch(pbs_c),
                        n,
                        dim,
                        q8_wmma_arch,
                        fusion == DflashFusionCtx::ChainVerify,
                    )?;
                    attention_prepare_batched(
                        gpu,
                        false,
                        &fa_attention_weights(layer),
                        &super::program::hybrid_dims(config),
                        &attention_scratch(pbs_c),
                        &flash_scratch(s),
                        &kv_view(kv_cache),
                        n,
                        start_c,
                        max_ctx_c,
                        &ctx,
                        BatchSemantics::Sequential,
                        None,
                        kv_layer_idx,
                        layer_idx,
                        fusion == DflashFusionCtx::ChainVerify,
                        false,
                        false,
                        false,
                        attention_tap(layer_idx, config).as_deref(),
                    )?;
                    attention_input_projection_batched(
                        gpu,
                        &fa_attention_weights(layer),
                        &super::program::hybrid_dims(config),
                        &attention_scratch(pbs_n),
                        n,
                        dim,
                        q8_wmma_arch,
                        fusion == DflashFusionCtx::ChainVerify,
                    )?;
                    attention_prepare_batched(
                        gpu,
                        false,
                        &fa_attention_weights(layer),
                        &super::program::hybrid_dims(config),
                        &attention_scratch(pbs_n),
                        &flash_scratch(s),
                        &kv_view(kv_cache),
                        n,
                        start_n,
                        max_ctx_n,
                        &ctx,
                        BatchSemantics::Sequential,
                        None,
                        kv_layer_idx,
                        layer_idx,
                        fusion == DflashFusionCtx::ChainVerify,
                        false,
                        false,
                        false,
                        attention_tap(layer_idx, config).as_deref(),
                    )?;
                    batch_chunk_fa_attend_merged(
                        gpu,
                        config,
                        pbs_c,
                        pbs_n,
                        stage,
                        s,
                        kv_cache,
                        start_c,
                        max_ctx_merged,
                        &ctx,
                        layer_idx,
                    )?;
                    attention_output_projection_batched(
                        gpu,
                        &fa_attention_weights(layer),
                        &attention_scratch(pbs_c),
                        n,
                        q8_wmma_arch,
                        arch_has_wmma,
                        BatchEpilogue::Residual,
                        fusion == DflashFusionCtx::ChainVerify,
                        false,
                        None,
                    )?;
                    batch_chunk_dense_ffn(
                        gpu,
                        layer.dense_ffn(),
                        config,
                        pbs_c,
                        n,
                        dim,
                        hidden_dim,
                        q8_wmma_arch,
                        BatchEpilogue::Residual,
                        fusion,
                    )?;
                    attention_output_projection_batched(
                        gpu,
                        &fa_attention_weights(layer),
                        &attention_scratch(pbs_n),
                        n,
                        q8_wmma_arch,
                        arch_has_wmma,
                        BatchEpilogue::Residual,
                        fusion == DflashFusionCtx::ChainVerify,
                        false,
                        None,
                    )?;
                    batch_chunk_dense_ffn(
                        gpu,
                        layer.dense_ffn(),
                        config,
                        pbs_n,
                        n,
                        dim,
                        hidden_dim,
                        q8_wmma_arch,
                        BatchEpilogue::Residual,
                        fusion,
                    )?;
                } else {
                    batch_chunk_full_attn_attn(
                        gpu,
                        multirow_c,
                        layer,
                        config,
                        pbs_c,
                        s,
                        kv_cache,
                        n,
                        dim,
                        start_c,
                        max_ctx_c,
                        &ctx,
                        BatchSemantics::Sequential,
                        None,
                        q8_wmma_arch,
                        arch_has_wmma,
                        kv_layer_idx,
                        layer_idx,
                        BatchEpilogue::Residual,
                        fusion,
                        None, // commit_stride: pair halves keep legacy cadence
                    )?;
                    batch_chunk_dense_ffn(
                        gpu,
                        layer.dense_ffn(),
                        config,
                        pbs_c,
                        n,
                        dim,
                        hidden_dim,
                        q8_wmma_arch,
                        BatchEpilogue::Residual,
                        fusion,
                    )?;
                    batch_chunk_full_attn_attn(
                        gpu,
                        multirow_n,
                        layer,
                        config,
                        pbs_n,
                        s,
                        kv_cache,
                        n,
                        dim,
                        start_n,
                        max_ctx_n,
                        &ctx,
                        BatchSemantics::Sequential,
                        None,
                        q8_wmma_arch,
                        arch_has_wmma,
                        kv_layer_idx,
                        layer_idx,
                        BatchEpilogue::Residual,
                        fusion,
                        None, // commit_stride: pair halves keep legacy cadence
                    )?;
                    batch_chunk_dense_ffn(
                        gpu,
                        layer.dense_ffn(),
                        config,
                        pbs_n,
                        n,
                        dim,
                        hidden_dim,
                        q8_wmma_arch,
                        BatchEpilogue::Residual,
                        fusion,
                    )?;
                }
                kv_layer_idx += 1;
                dump_hidden_localize(gpu, &pbs_c.x_batch, n, start_c, dim, layer_idx, "batched");
                dump_hidden_localize(gpu, &pbs_n.x_batch, n, start_n, dim, layer_idx, "batched");
            }
            (LayerWeights::FullAttn(_layer), LayerType::FullAttention) => {
                batch_chunk_full_attn_fallback(
                    gpu,
                    weights,
                    config,
                    layer_idx,
                    kv_layer_idx,
                    start_c,
                    n,
                    dim_row_bytes,
                    kv_cache,
                    s,
                    pbs_c,
                )?;
                batch_chunk_full_attn_fallback(
                    gpu,
                    weights,
                    config,
                    layer_idx,
                    kv_layer_idx,
                    start_n,
                    n,
                    dim_row_bytes,
                    kv_cache,
                    s,
                    pbs_n,
                )?;
                kv_layer_idx += 1;
                dump_hidden_localize(gpu, &pbs_c.x_batch, n, start_c, dim, layer_idx, "batched");
                dump_hidden_localize(gpu, &pbs_n.x_batch, n, start_n, dim, layer_idx, "batched");
            }
            (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                for (half_pbs, half_start, half_tape) in [
                    (pbs_c, start_c, chunk_start_c),
                    (pbs_n, start_n, chunk_start_c + n),
                ] {
                    batch_chunk_delta_net_moe(
                        gpu,
                        layer,
                        config,
                        half_pbs,
                        dn_state,
                        n,
                        dim,
                        hidden_dim,
                        k_dim,
                        v_dim,
                        n_v_heads,
                        hd,
                        BatchSemantics::Sequential,
                        None,
                        None,
                        half_tape,
                        delta_layer_idx,
                        half_start,
                        layer_idx,
                        &ctx,
                        weights,
                        None,
                        PrefillRouteMode::Replicated,
                    )?;
                    dump_hidden_localize(
                        gpu,
                        &half_pbs.x_batch,
                        n,
                        half_start,
                        dim,
                        layer_idx,
                        "batched",
                    );
                }
                delta_layer_idx += 1;
            }
            (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) if fa_batched_ok => {
                if merge_fa {
                    batch_chunk_full_attn_moe_prep(
                        gpu, layer, config, pbs_c, n, dim, kv_cache, None, layer_idx,
                    )?;
                    batch_chunk_full_attn_moe_prep(
                        gpu, layer, config, pbs_n, n, dim, kv_cache, None, layer_idx,
                    )?;
                    batch_chunk_fa_attend_merged(
                        gpu,
                        config,
                        pbs_c,
                        pbs_n,
                        stage,
                        s,
                        kv_cache,
                        start_c,
                        max_ctx_merged,
                        &ctx,
                        layer_idx,
                    )?;
                    batch_chunk_full_attn_moe_finish(
                        gpu,
                        false,
                        layer,
                        config,
                        pbs_c,
                        s,
                        kv_cache,
                        n,
                        start_c,
                        max_ctx_c,
                        &ctx,
                        BatchSemantics::Sequential,
                        None,
                        q8_wmma_arch,
                        layer_idx,
                        weights,
                        None,
                        PrefillRouteMode::Replicated,
                    )?;
                    batch_chunk_full_attn_moe_finish(
                        gpu,
                        false,
                        layer,
                        config,
                        pbs_n,
                        s,
                        kv_cache,
                        n,
                        start_n,
                        max_ctx_n,
                        &ctx,
                        BatchSemantics::Sequential,
                        None,
                        q8_wmma_arch,
                        layer_idx,
                        weights,
                        None,
                        PrefillRouteMode::Replicated,
                    )?;
                } else {
                    batch_chunk_full_attn_moe(
                        gpu,
                        multirow_c,
                        layer,
                        config,
                        pbs_c,
                        s,
                        kv_cache,
                        n,
                        dim,
                        start_c,
                        max_ctx_c,
                        &ctx,
                        BatchSemantics::Sequential,
                        None,
                        q8_wmma_arch,
                        arch_has_wmma,
                        kv_layer_idx,
                        layer_idx,
                        None,
                        weights,
                        PrefillRouteMode::Replicated,
                    )?;
                    batch_chunk_full_attn_moe(
                        gpu,
                        multirow_n,
                        layer,
                        config,
                        pbs_n,
                        s,
                        kv_cache,
                        n,
                        dim,
                        start_n,
                        max_ctx_n,
                        &ctx,
                        BatchSemantics::Sequential,
                        None,
                        q8_wmma_arch,
                        arch_has_wmma,
                        kv_layer_idx,
                        layer_idx,
                        None,
                        weights,
                        PrefillRouteMode::Replicated,
                    )?;
                }
                kv_layer_idx += 1;
                dump_hidden_localize(gpu, &pbs_c.x_batch, n, start_c, dim, layer_idx, "batched");
                dump_hidden_localize(gpu, &pbs_n.x_batch, n, start_n, dim, layer_idx, "batched");
            }
            _ => panic!("layer type mismatch at layer {layer_idx}"),
        }
    }

    // Tail logits exactly as two sequential chunks would: each half norms
    // its own rows into the caller's per-token buffer (when present) with
    // its own chunk-relative offset, and the legacy last-token path runs
    // per half in order so `s.logits` ends on the pair's last token.
    let pth_c = per_token_hidden_out.map(|(t, capture)| (t, chunk_start_c, capture));
    let pth_n = per_token_hidden_out.map(|(t, capture)| (t, chunk_start_c + n, capture));
    batch_chunk_final_logits(
        gpu,
        weights,
        config,
        s,
        pbs_c,
        n,
        dim,
        dim_row_bytes,
        pth_c,
        needs_last_token_logits,
        true,
        &ctx,
    )?;
    batch_chunk_final_logits(
        gpu,
        weights,
        config,
        s,
        pbs_n,
        n,
        dim,
        dim_row_bytes,
        pth_n,
        needs_last_token_logits,
        true,
        &ctx,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_batch_chunk_impl(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    s: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
    hidden_rb: Option<&HiddenStateRingBuffer>,
    per_token_hidden_out: Option<HiddenRowsOut<'_>>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tape_offset: usize,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    pre_uploaded: bool,
    pre_embedded: bool,
    band: Option<&PrefillBandCtx<'_>>,
    mask_override: Option<MaskEmbedOverride<'_>>,
    needs_last_token_logits: bool,
    max_layer: Option<usize>,
    routed_out: Option<&GpuTensor>,
    batch_semantics: BatchSemantics<'_>,
    fusion: DflashFusionCtx,
    commit_stride: Option<usize>,
) -> HipResult<()> {
    let n = tokens.len();
    debug_assert!(n > 0);
    debug_assert!(n <= pbs.max_batch);
    debug_assert!(
        commit_stride.is_none() || commit_stride == Some(WIDENED_COMMIT_ROWS),
        "commit_stride admits only None and Some(512)"
    );
    // Segmentation applies only above the stride; small tails execute their
    // old branch. Authority comes from this parameter alone — never inferred
    // from arch or `n` inside a generic wrapper.
    let commit_stride = match commit_stride {
        Some(s) if n > s => Some(s),
        _ => None,
    };
    batch_chunk_validate_independent(
        n,
        batch_semantics,
        dn_state,
        kv_cache,
        tree_verify,
        gdn_tape.as_deref(),
    )?;
    let inactive_mask = match batch_semantics.active_mask() {
        Some(active_mask) => {
            let full_mask = valid_lane_mask(n)?;
            if active_mask == 0 || active_mask & !full_mask != 0 {
                return Err(HipError::new(
                    0,
                    "forward_batch_chunk: active mask out of range",
                ));
            }
            let inactive = full_mask & !active_mask;
            (inactive != 0).then_some(inactive)
        }
        None => None,
    };
    let inactive_backup = if inactive_mask.is_some() {
        let backup = pbs
            .moe_inactive_backup
            .as_ref()
            .ok_or_else(|| HipError::new(0, "inactive residual backup is unavailable"))?;
        let bytes = n
            .checked_mul(config.dim)
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| HipError::new(0, "inactive residual backup size overflow"))?;
        gpu.memcpy_dtod_at_auto(&backup.buf, 0, &pbs.x_batch.buf, 0, bytes)?;
        Some(backup)
    } else {
        None
    };

    let dispatch_workload = prefill_dispatch_workload(
        captures_verify_hidden(per_token_hidden_out),
        gdn_tape.is_some(),
        tree_verify.is_some(),
    );
    let required_tokens = checked_kv_end(start_pos, n, "forward_prefill_chunk")?;
    kv_cache.require_mapped_capacity(required_tokens)?;
    debug_assert!(
        routed_out.is_none()
            || band
                .map(|b| b.layer_end - b.layer_start <= 1)
                .unwrap_or(false),
        "forward_prefill_chunk: routed_out requires a single-layer band (EP driver invariant)",
    );

    let dim = config.dim;
    let hidden_dim = config.hidden_dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    let dim_row_bytes = dim * 4;
    let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);

    let do_embed = band.map(|b| b.is_first_band).unwrap_or(true);
    let layer_start = band.map(|b| b.layer_start).unwrap_or(0);
    let layer_end = band
        .map(|b| b.layer_end)
        .unwrap_or(config.n_layers)
        .min(max_layer.unwrap_or(usize::MAX));
    let do_lm_head = band.map(|b| b.is_last_band).unwrap_or(true) && max_layer.is_none();
    // EP prefill route authority for single-layer MoE bands. `None` (whole
    // stack) and every non-EP caller resolve to `Replicated`: the historical
    // seal_prefill + local produce path, unchanged.
    let route = band
        .map(|b| b.route)
        .unwrap_or(PrefillRouteMode::Replicated);
    macro_rules! givens_cos_view {
        () => {
            band.and_then(|b| b.givens_cos)
                .or(kv_cache.givens_cos.as_ref())
        };
    }
    macro_rules! givens_sin_view {
        () => {
            band.and_then(|b| b.givens_sin)
                .or(kv_cache.givens_sin.as_ref())
        };
    }

    batch_chunk_embed_tokens(
        gpu,
        weights,
        tokens,
        s,
        pbs,
        n,
        dim,
        dim_row_bytes,
        do_embed,
        pre_embedded,
        pre_uploaded,
        batch_semantics.active_mask(),
        mask_override,
    )?;
    batch_chunk_upload_positions(
        gpu,
        pbs,
        batch_semantics,
        start_pos,
        n,
        tree_verify,
        pre_uploaded,
    )?;

    let fa_arch = gpu.arch.as_str();
    let q8_wmma_arch = q8_prefill_wmma_enabled(gpu);
    let arch_has_wmma = q8_wmma_arch;
    // Slice-B admission (see above): native fp8 KV takes the batched FA path.
    let fa_batched_ok = (kv_cache.quant_q8
        || kv_cache.quant_asym4
        || kv_cache.quant_asym3
        || kv_cache.quant_asym2
        || kv_cache.quant_fp8)
        && weights.layers.iter().all(|lw| match lw {
            LayerWeights::FullAttn(_) | LayerWeights::FullAttnMoe(_) => {
                qwen35_layer_batch_admissible(lw, config, fa_arch).is_ok()
            }
            _ => true,
        });
    // Attention only: the batched masked FA kernel grids [n_heads, tiles, ROW]
    // and re-scans the whole KV once per row, so a small verify block over a
    // long context pays the scan n times (202 vs 103 ms at 33k). The layer's
    // GEMMs stay batched either way — only the attend step switches to the
    // multi-row tile. Its tile grid is sized from the live logical context on
    // the host, so a captured replay would keep the first cycle's tile count.
    let fa_attn_multirow = q8_multirow_attn_admitted(
        gpu.arch_caps.arch(),
        kv_cache.quant_q8,
        config.head_dim,
        n,
        start_pos + n,
        fa_pertoken_min_ctx(gpu.arch_caps.arch()),
        tree_verify.is_some(),
        batch_semantics.is_independent(),
        gpu.graphs.capture_mode,
        gpu.replay.is_recording(),
    );
    let logical_max_ctx = match batch_semantics {
        BatchSemantics::Sequential => start_pos + n,
        BatchSemantics::Independent { positions, .. } => {
            positions.iter().copied().max().unwrap_or(0) + 1
        }
    };
    if batch_semantics.is_independent() && !fa_batched_ok {
        return Err(HipError::new(
            0,
            "independent decode requires the fully batched FullAttention weight path",
        ));
    }
    let max_ctx_len = if gpu.graphs.capture_mode {
        kv_cache.physical_cap
    } else {
        logical_max_ctx
    };

    let mut delta_layer_idx = band.map(|b| b.delta_layer_offset).unwrap_or(0);
    let mut kv_layer_idx = band.map(|b| b.kv_layer_offset).unwrap_or(0);
    let gdn_chunk_scan_common_admitted = gpu.flags.gfx12_gdn_chunk_scan
        && matches!(gpu.arch.as_str(), "gfx1100" | "gfx1151" | "gfx1201")
        && config.num_experts == 0
        && config.linear_num_key_heads == 16
        && config.linear_num_value_heads == 48
        && config.linear_key_head_dim == 128
        && config.linear_value_head_dim == 128
        && config.conv_kernel_dim == 4
        && dense_layers_are_all_mq4v2(weights)
        // gfx1201 admits both MQ4V2 projection routes: iu4-direct (default) or
        // FP8-WMMA qkvza (HIPFIRE_IU4_PREFILL=0 arm). Every qkvza dispatch arm
        // writes the same f32 dn_qkv/dn_z/dn_beta/dn_alpha scratch the scan
        // consumes, so the scan is route-agnostic. gfx11 keeps the iu4 gate.
        && (gpu.flags.iu4_prefill_enabled()
            || (gpu.arch == "gfx1201" && gpu.flags.gfx12_mq4v2_fp8_qkvza))
        // Native FP8 KV is gfx1201-only; gfx11 runs this scan with its supported Q8 KV tier.
        && (kv_cache.quant_fp8 || (gpu.arch != "gfx1201" && kv_cache.quant_q8))
        && dn_state.quant == StateQuant::Q8
        && !dn_state.s_matrices.is_empty()
        && dn_state.s_matrices.len() == dn_state.s_scales.len()
        && dn_state.s_matrices.len() == dn_state.conv_states.len()
        && dn_state.s_matrices.len() == dn_state.s_ef_residual.len()
        && matches!(batch_semantics, BatchSemantics::Sequential)
        && tree_verify.is_none()
        && gdn_tape.is_none()
        && fusion == DflashFusionCtx::Off
        && !gpu.graphs.capture_mode
        // Only a forward being recorded into a Redline tape must stay off the
        // chunk scan; prefill never records, so a Redline-enabled process
        // prefills exactly like the HIP-graph default.
        && !gpu.replay.is_recording()
        // A DFlash prompt seed's ring copies each layer's finished residual
        // rows, whichever recurrence kernel produced them.
        && hidden_rb.map_or(true, |rb| rb.rides_ordinary_prefill(gpu))
        // A prompt fill that hands its hidden rows to a draft head runs the
        // same scan plain AR prefill runs; verify captures stay sequential.
        && !captures_verify_hidden(per_token_hidden_out)
        && band.is_none()
        && max_layer.is_none()
        && routed_out.is_none()
        && mask_override.is_none()
        && n >= 64
        && (n <= 512
            || (commit_stride == Some(512)
                && (n % 512 == 0 || (64..512).contains(&(n % 512)))));
    let ctx = DispatchCtx::new(gpu).with_workload(dispatch_workload);

    for layer_idx in layer_start..layer_end {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                let gdn_chunk_scan_admitted = gdn_chunk_scan_common_admitted
                    && layer.wqkv.gpu_dtype == DType::MQ4G256V2
                    && layer.wz.gpu_dtype == DType::MQ4G256V2
                    && layer.w_beta.gpu_dtype == DType::MQ4G256V2
                    && layer.w_alpha.gpu_dtype == DType::MQ4G256V2
                    && delta_layer_idx < dn_state.s_ef_residual.len();
                batch_chunk_delta_net_attn(
                    gpu,
                    layer,
                    config,
                    pbs,
                    dn_state,
                    n,
                    dim,
                    k_dim,
                    v_dim,
                    n_v_heads,
                    hd,
                    batch_semantics,
                    tree_verify,
                    gdn_tape.as_deref(),
                    tape_offset,
                    delta_layer_idx,
                    q8_wmma_arch,
                    arch_has_wmma,
                    BatchEpilogue::Residual,
                    fusion,
                    commit_stride,
                    gdn_chunk_scan_admitted,
                )?;
                batch_chunk_dense_ffn(
                    gpu,
                    layer.dense_ffn(),
                    config,
                    pbs,
                    n,
                    dim,
                    hidden_dim,
                    q8_wmma_arch,
                    BatchEpilogue::Residual,
                    fusion,
                )?;
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_chunk_rows(gpu, slot, &pbs.x_batch, n)?;
                    }
                }
                delta_layer_idx += 1;
                dump_hidden_localize(gpu, &pbs.x_batch, n, start_pos, dim, layer_idx, "batched");
            }
            (LayerWeights::FullAttn(layer), LayerType::FullAttention) if fa_batched_ok => {
                batch_chunk_full_attn_attn(
                    gpu,
                    fa_attn_multirow,
                    layer,
                    config,
                    pbs,
                    s,
                    kv_cache,
                    n,
                    dim,
                    start_pos,
                    max_ctx_len,
                    &ctx,
                    batch_semantics,
                    tree_verify,
                    q8_wmma_arch,
                    arch_has_wmma,
                    kv_layer_idx,
                    layer_idx,
                    BatchEpilogue::Residual,
                    fusion,
                    commit_stride,
                )?;
                batch_chunk_dense_ffn(
                    gpu,
                    layer.dense_ffn(),
                    config,
                    pbs,
                    n,
                    dim,
                    hidden_dim,
                    q8_wmma_arch,
                    BatchEpilogue::Residual,
                    fusion,
                )?;
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_chunk_rows(gpu, slot, &pbs.x_batch, n)?;
                    }
                }
                kv_layer_idx += 1;
                dump_hidden_localize(gpu, &pbs.x_batch, n, start_pos, dim, layer_idx, "batched");
            }
            (LayerWeights::FullAttn(_layer), LayerType::FullAttention) => {
                batch_chunk_full_attn_fallback(
                    gpu,
                    weights,
                    config,
                    layer_idx,
                    kv_layer_idx,
                    start_pos,
                    n,
                    dim_row_bytes,
                    kv_cache,
                    s,
                    pbs,
                )?;
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_chunk_rows(gpu, slot, &pbs.x_batch, n)?;
                    }
                }
                kv_layer_idx += 1;
                dump_hidden_localize(gpu, &pbs.x_batch, n, start_pos, dim, layer_idx, "batched");
            }
            (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                batch_chunk_delta_net_moe(
                    gpu,
                    layer,
                    config,
                    pbs,
                    dn_state,
                    n,
                    dim,
                    hidden_dim,
                    k_dim,
                    v_dim,
                    n_v_heads,
                    hd,
                    batch_semantics,
                    tree_verify,
                    gdn_tape.as_deref(),
                    tape_offset,
                    delta_layer_idx,
                    start_pos,
                    layer_idx,
                    &ctx,
                    weights,
                    routed_out,
                    route,
                )?;
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_chunk_rows(gpu, slot, &pbs.x_batch, n)?;
                    }
                }
                delta_layer_idx += 1;
                dump_hidden_localize(gpu, &pbs.x_batch, n, start_pos, dim, layer_idx, "batched");
            }
            (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) if fa_batched_ok => {
                batch_chunk_full_attn_moe(
                    gpu,
                    fa_attn_multirow,
                    layer,
                    config,
                    pbs,
                    s,
                    kv_cache,
                    n,
                    dim,
                    start_pos,
                    max_ctx_len,
                    &ctx,
                    batch_semantics,
                    tree_verify,
                    q8_wmma_arch,
                    arch_has_wmma,
                    kv_layer_idx,
                    layer_idx,
                    routed_out,
                    weights,
                    route,
                )?;
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_chunk_rows(gpu, slot, &pbs.x_batch, n)?;
                    }
                }
                kv_layer_idx += 1;
                dump_hidden_localize(gpu, &pbs.x_batch, n, start_pos, dim, layer_idx, "batched");
            }
            _ => panic!("layer type mismatch at layer {layer_idx}"),
        }
        if let (Some(inactive_mask), Some(backup)) = (inactive_mask, inactive_backup.as_ref()) {
            restore_inactive_residual_rows(gpu, pbs, backup, inactive_mask, n, dim)?;
        }
    }

    batch_chunk_final_logits(
        gpu,
        weights,
        config,
        s,
        pbs,
        n,
        dim,
        dim_row_bytes,
        per_token_hidden_out,
        needs_last_token_logits,
        do_lm_head,
        &ctx,
    )?;

    Ok(())
}

/// Run a single FullAttn layer body on s.x at position `pos`. Extracted
/// for use from the batched prefill path's FA-layer fallback. Byte-exact
/// with the FA branch of forward_scratch_layers.
#[allow(clippy::too_many_arguments)]
fn run_fa_layer_body(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    layer_idx: usize,
    _kv_layer_idx: usize,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    s: &Qwen35Scratch,
) -> HipResult<()> {
    let layer = match &weights.layers[layer_idx] {
        LayerWeights::FullAttn(l) => l,
        _ => unreachable!(),
    };

    // Fused rmsnorm + FWHT rotation for wq/wk/wv (MQ-family).
    let x_rot = fused_rmsnorm_rotate_for_mq(
        gpu,
        &layer.wq,
        &s.x,
        &layer.attn_norm,
        &s.tmp,
        &s.x_rot,
        config.norm_eps,
    )?;
    // Cross-arch fast path: fused 3-way projection for wq+wk+wv.
    let dt = layer.wq.gpu_dtype;
    let fa3_same_dtype = layer.wk.gpu_dtype == dt && layer.wv.gpu_dtype == dt;
    let fused_fa3_mq4 = fa3_same_dtype
        && (matches!(
            dt,
            DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ4CG256 | DType::HFQ4G256
        ));
    let fused_fa3_lloyd_mq3 = fa3_same_dtype && dt == DType::MQ3G256Lloyd;
    let fused_fa3_lloyd_mq4 = fa3_same_dtype && dt == DType::MQ4G256Lloyd;
    // Phase A.1c (gfx906): fused dp4a path for HFQ6/MQ6 weights.
    let fused_fa3_hfq6 = fa3_same_dtype
        && (dt == DType::MQ6G256 || dt == DType::HFQ6G256)
        && gpu.arch_caps.gemv_dp4a_enabled();
    if fused_fa3_mq4 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        if dt == DType::MQ4CG256 || dt == DType::MQ4G256V2 {
            let key = hipfire_dispatch::families::fused_qkv::fused_qkv_key_for(dt);
            let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);
            let params = hipfire_dispatch::families::fused_qkv::FusedQkvParams {
                kind: key,
                weights: &[&layer.wq.buf, &layer.wk.buf, &layer.wv.buf],
                x: eff_x,
                outputs: &[&s.fa_q_full, &s.fa_k, &s.fa_v],
                m: &[layer.wq.m, layer.wk.m, layer.wv.m],
                k: layer.wq.k,
                rot_scratch: &[],
                batch_size: None,
            };
            hipfire_runtime::llama::fused_qkv_family()
                .run(&ctx, gpu, &params)
                .map_err(hip_bridge::HipError::from)?;
        } else {
            gpu.fused_qkv_hfq4g256(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                eff_x,
                &s.fa_q_full,
                &s.fa_k,
                &s.fa_v,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
            )?;
        }
    } else if fused_fa3_lloyd_mq3 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_qkv_mq3g256_lloyd(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            eff_x,
            &s.fa_q_full,
            &s.fa_k,
            &s.fa_v,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
        )?;
    } else if fused_fa3_lloyd_mq4 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_qkv_mq4g256_lloyd(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            eff_x,
            &s.fa_q_full,
            &s.fa_k,
            &s.fa_v,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
        )?;
    } else if fused_fa3_hfq6 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_qkv_hfq6g256_dp4a(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            eff_x,
            &s.fa_q_full,
            &s.fa_k,
            &s.fa_v,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
        )?;
    } else {
        weight_gemv_prerotated(gpu, &layer.wq, &s.tmp, x_rot, &s.fa_q_full)?;
        weight_gemv_prerotated(gpu, &layer.wk, &s.tmp, x_rot, &s.fa_k)?;
        weight_gemv_prerotated(gpu, &layer.wv, &s.tmp, x_rot, &s.fa_v)?;
    }

    gpu.deinterleave_f32(
        &s.fa_q_full,
        &s.fa_q,
        &s.fa_gate,
        config.n_heads,
        config.head_dim,
    )?;
    gpu.rmsnorm_batched(
        &s.fa_q,
        &layer.q_norm,
        &s.fa_q,
        config.n_heads,
        config.head_dim,
        config.norm_eps,
    )?;
    let kv_dim = config.n_kv_heads * config.head_dim;
    gpu.rmsnorm_batched(
        &s.fa_k,
        &layer.k_norm,
        &s.fa_k,
        config.n_kv_heads,
        config.head_dim,
        config.norm_eps,
    )?;

    if hipfire_runtime::triattn::tap_enabled() {
        // Try GPU path first (matches the batched FA tap at line ~3499 in
        // forward_prefill_batch). When the calibration tap is GPU-resident
        // (CalibrateGpu) we MUST dispatch the kernel here — falling
        // through to record_prerope_qk would either silently drop the
        // sample (pre-Phase-2) or panic (post-Phase-2).
        let gpu_handled = hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
            gpu,
            layer_idx,
            &s.fa_q.buf,
            1,
            config.n_heads,
            config.head_dim,
        )?;
        if !gpu_handled {
            let n_q = config.n_heads * config.head_dim;
            let q_cpu = gpu.download_f32(&s.fa_q)?;
            if hipfire_runtime::triattn::tap_needs_k() {
                let n_k = config.n_kv_heads * config.head_dim;
                let k_cpu = gpu.download_f32(&s.fa_k)?;
                hipfire_runtime::triattn::record_prerope_qk(
                    layer_idx,
                    &q_cpu[..n_q],
                    Some(&k_cpu[..n_k]),
                );
            } else {
                hipfire_runtime::triattn::record_prerope_q(layer_idx, &q_cpu[..n_q]);
            }
        }
    }

    // If TriAttention has compacted the cache, absolute RoPE phase diverges
    // from the physical cache index. Temporarily load the absolute position
    // into pos_buf for the rope call, then restore the physical position
    // for kv_cache_write + flash attention (which both want the write slot).
    if kv_cache.compact_offset > 0 {
        let abs = (pos + kv_cache.compact_offset) as i32;
        gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
    }
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
    gpu.rope_partial_interleaved_f32(
        &s.fa_q,
        &s.fa_k,
        &s.pos_buf,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        n_rot,
        config.rope_theta,
    )?;
    if kv_cache.compact_offset > 0 {
        let phys = pos as i32;
        gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
    }
    let ctx = DispatchCtx::new(gpu);
    let fused_epilogue =
        kv_cache_attention_dispatch(&ctx, gpu, kv_cache, s, config, &layer.wo, layer_idx, pos)?;

    if !fused_epilogue {
        gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
    }
    {
        let wr = layer.wo.dispatch_ref();
        let input = if fused_epilogue {
            GemvInput::Prerotated(&s.fa_attn_out)
        } else {
            GemvInput::Raw(&s.fa_attn_out)
        };
        execute_steps(
            gpu,
            &ctx,
            &[Step::GemvResidual {
                w: &wr,
                input,
                residual: &s.x,
                out: &s.x,
            }],
        )
        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    }

    // FFN: fused rmsnorm + rotate for w_gate/w_up.
    let x_rot = fused_rmsnorm_rotate_for_mq(
        gpu,
        &layer.w_gate,
        &s.x,
        &layer.ffn_norm,
        &s.tmp,
        &s.x_rot,
        config.norm_eps,
    )?;
    let dt_g = layer.w_gate.gpu_dtype;
    let same_dtype = layer.w_up.gpu_dtype == dt_g;
    let fused_gu_mq4 = same_dtype
        && (matches!(
            dt_g,
            DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ4CG256 | DType::HFQ4G256
        ));
    let fused_gu_lloyd_mq3 = same_dtype && dt_g == DType::MQ3G256Lloyd;
    let fused_gu_lloyd_mq4 = same_dtype && dt_g == DType::MQ4G256Lloyd;
    // Phase A.1c (gfx906): fused dp4a path for HFQ6/MQ6 weights.
    let fused_gu_hfq6 = same_dtype
        && (dt_g == DType::MQ6G256 || dt_g == DType::HFQ6G256)
        && gpu.arch_caps.gemv_dp4a_enabled();
    if fused_gu_mq4 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        if dt_g == DType::MQ4CG256 || dt_g == DType::MQ4G256V2 {
            let key = hipfire_dispatch::families::fused_qkv::fused_gate_up_key_for(dt_g);
            let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);
            let params = hipfire_dispatch::families::fused_qkv::FusedQkvParams {
                kind: key,
                weights: &[&layer.w_gate.buf, &layer.w_up.buf],
                x: eff_x,
                outputs: &[&s.gate_ffn, &s.up],
                m: &[layer.w_gate.m, layer.w_up.m],
                k: layer.w_gate.k,
                rot_scratch: &[],
                batch_size: None,
            };
            hipfire_runtime::llama::fused_qkv_family()
                .run(&ctx, gpu, &params)
                .map_err(hip_bridge::HipError::from)?;
        } else {
            gpu.fused_gate_up_hfq4g256(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                eff_x,
                &s.gate_ffn,
                &s.up,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
            )?;
        }
    } else if fused_gu_lloyd_mq3 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_gate_up_mq3g256_lloyd(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            eff_x,
            &s.gate_ffn,
            &s.up,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
        )?;
    } else if fused_gu_lloyd_mq4 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_gate_up_mq4g256_lloyd(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            eff_x,
            &s.gate_ffn,
            &s.up,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
        )?;
    } else if fused_gu_hfq6 {
        let eff_x = match x_rot {
            Some(xr) => xr,
            None => &s.tmp,
        };
        gpu.fused_gate_up_hfq6g256_dp4a(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            eff_x,
            &s.gate_ffn,
            &s.up,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
        )?;
    } else {
        weight_gemv_prerotated(gpu, &layer.w_gate, &s.tmp, x_rot, &s.gate_ffn)?;
        weight_gemv_prerotated(gpu, &layer.w_up, &s.tmp, x_rot, &s.up)?;
    }
    weight_gemv_swiglu_residual(gpu, &layer.w_down, &s.gate_ffn, &s.up, &s.ffn_hidden, &s.x)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::forward::unsupported_mq3_experts_uniform_from_dtypes;
    use super::*;
    use hipfire_dispatch::context::DispatchWorkload;
    use hipfire_dispatch::families::fused_qkv::fused_gate_up_key_for;
    use hipfire_dispatch::families::gemm::residual_gemm_key_for;
    use hipfire_dispatch::families::moe::{
        gated_moe_prefill_admissible_for_dtypes as moe_ffn_batched_admissible_for_dtypes,
        moe_prefill_topk_shape_supported, paro_batched_admit_enabled_from_env,
        routed_codebook_grouped_supported, routed_uniform_mqv2_grouped_supported, MoePrefillDtypes,
    };
    use rdna_compute::DType;

    #[test]
    fn q8_multirow_attn_admits_only_explicit_arch_shapes() {
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            for head_dim in [128, 256] {
                for n in [4, 8, 32] {
                    assert!(q8_multirow_attn_admitted(
                        arch,
                        true,
                        head_dim,
                        n,
                        4097,
                        Some(4096),
                        false,
                        false,
                        false,
                        false,
                    ));
                }
            }
        }
    }

    #[test]
    fn q8_multirow_attn_rejects_unmeasured_or_unsupported_routes() {
        let admitted = |arch,
                        quant_q8,
                        head_dim,
                        n,
                        logical_ctx,
                        min_ctx,
                        is_tree,
                        is_independent,
                        capture_mode,
                        replay_recording| {
            q8_multirow_attn_admitted(
                arch,
                quant_q8,
                head_dim,
                n,
                logical_ctx,
                min_ctx,
                is_tree,
                is_independent,
                capture_mode,
                replay_recording,
            )
        };
        assert!(!admitted(
            "gfx1200",
            true,
            256,
            8,
            8192,
            Some(4096),
            false,
            false,
            false,
            false,
        ));
        assert!(!admitted(
            "gfx1100",
            false,
            256,
            8,
            8192,
            Some(4096),
            false,
            false,
            false,
            false,
        ));
        for head_dim in [64, 320] {
            assert!(!admitted(
                "gfx1100",
                true,
                head_dim,
                8,
                8192,
                Some(4096),
                false,
                false,
                false,
                false,
            ));
        }
        for n in [1, 3, 33] {
            assert!(!admitted(
                "gfx1100",
                true,
                256,
                n,
                8192,
                Some(4096),
                false,
                false,
                false,
                false,
            ));
        }
        assert!(!admitted(
            "gfx1100",
            true,
            256,
            8,
            4096,
            Some(4096),
            false,
            false,
            false,
            false,
        ));
        assert!(!admitted(
            "gfx1100", true, 256, 8, 8192, None, false, false, false, false,
        ));
        assert!(!admitted(
            "gfx1100",
            true,
            256,
            8,
            8192,
            Some(4096),
            true,
            false,
            false,
            false,
        ));
        assert!(!admitted(
            "gfx1100",
            true,
            256,
            8,
            8192,
            Some(4096),
            false,
            true,
            false,
            false,
        ));
        assert!(!admitted(
            "gfx1100",
            true,
            256,
            8,
            8192,
            Some(4096),
            false,
            false,
            true,
            false,
        ));
    }

    #[test]
    fn q8_multirow_attn_rejects_replay_recording_on_supported_arches() {
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            assert!(q8_multirow_attn_admitted(
                arch, true, 256, 8, 8192, Some(4096), false, false, false, false,
            ));
            assert!(!q8_multirow_attn_admitted(
                arch, true, 256, 8, 8192, Some(4096), false, false, false, true,
            ));
        }
    }

    #[test]
    fn fa_pertoken_min_ctx_is_opt_in_on_gfx1151_only() {
        for arch in ["gfx1100", "gfx1201"] {
            assert_eq!(fa_pertoken_min_ctx_for(arch, None), Some(4096));
        }
        assert_eq!(fa_pertoken_min_ctx_for("gfx1151", None), None);
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            assert_eq!(fa_pertoken_min_ctx_for(arch, Some(8192)), Some(8192));
            assert_eq!(fa_pertoken_min_ctx_for(arch, Some(0)), None);
        }
    }

    #[test]
    fn paro_batched_admit_defaults_off_and_allows_opt_in() {
        // PARO batched prefill is default-OFF (the path has a coherence/echo bug;
        // per-token fallback is correct) — opt in via HIPFIRE_PARO_BATCHED=1.
        // `paro_batched_admit_enabled_from_env` is `value == Some("1")`, so only
        // the exact string "1" enables it; everything else (incl. None) is off.
        assert!(!paro_batched_admit_enabled_from_env(None));
        assert!(paro_batched_admit_enabled_from_env(Some("1")));
        assert!(!paro_batched_admit_enabled_from_env(Some("surprise")));
        assert!(!paro_batched_admit_enabled_from_env(Some("0")));
    }

    #[test]
    fn prefill_max_batch_arch_defaults() {
        // Exact gfx1100 / gfx1201 alone get the measured defaults; every other
        // string keeps the conservative PREFILL_MAX_BATCH=256 ceiling.
        // Pure helper — no process env mutation.
        assert_eq!(prefill_max_batch_for_arch("gfx1100", false), 512);
        assert_eq!(
            prefill_max_batch_for_arch("gfx1100", true),
            PREFILL_DEFAULT_BATCH_GFX1100,
            "fp8 arm must not move the gfx1100 default"
        );
        assert_eq!(prefill_max_batch_for_arch("gfx1201", false), 384);
        assert_eq!(
            prefill_max_batch_for_arch("gfx1201", false),
            PREFILL_DEFAULT_BATCH_GFX1201
        );
        assert_eq!(
            prefill_max_batch_for_arch("gfx1201", true),
            512,
            "full-FP8 gfx1201 default must be 512"
        );
        assert_eq!(
            prefill_max_batch_for_arch("gfx1201", true),
            PREFILL_DEFAULT_BATCH_GFX1201_FP8
        );
        for arch in ["gfx1200", "gfx1151", "gfx942", "unknown"] {
            assert_eq!(
                prefill_max_batch_for_arch(arch, false),
                PREFILL_MAX_BATCH,
                "arch default must stay {PREFILL_MAX_BATCH} on {arch}"
            );
            assert_eq!(
                prefill_max_batch_for_arch(arch, true),
                PREFILL_MAX_BATCH,
                "fp8 arm must not move non-gfx1201 defaults ({arch})"
            );
        }
    }

    #[test]
    fn prefill_effective_chunk_hidden_ring_caps_over_configured_and_pbs() {
        // DFlash / prompt-seed staging is commonly 256 while gfx1201 default is 384.
        assert_eq!(
            prefill_effective_chunk_batch(384, 384, Some(256)),
            256,
            "hidden_rb=256 must never receive a 384-row chunk"
        );
        assert_eq!(prefill_effective_chunk_batch(384, 512, Some(256)), 256);
        assert_eq!(prefill_effective_chunk_batch(256, 384, Some(128)), 128);
    }

    #[test]
    fn prefill_effective_chunk_explicit_cap_wins_over_larger_pbs() {
        // Eviction capped path: configured/capped max is already min'd to 256.
        assert_eq!(prefill_effective_chunk_batch(256, 384, None), 256);
        assert_eq!(prefill_effective_chunk_batch(256, 256, None), 256);
        // Ordinary no-hidden/no-cap gfx1201 keeps the 384 win when PBS matches.
        assert_eq!(prefill_effective_chunk_batch(384, 384, None), 384);
        // Caller-owned PBS smaller than configured still bounds the chunk.
        assert_eq!(prefill_effective_chunk_batch(384, 128, None), 128);
    }

    // ── Qwen3.5 dispatch: is_batchable_la ────────────────────────

    /// The Qwen3.5-specific copy admits more dtypes than the runtime copy
    /// (ParoQ4G128, F32, Lloyd variants).

    const BATCHABLE_ARCHS: &[&str] = &[
        "gfx900", "gfx906", "gfx908", "gfx940", "gfx941", "gfx942", "gfx1010", "gfx1011",
        "gfx1012", "gfx1013", "gfx1030", "gfx1031", "gfx1032", "gfx1100", "gfx1101", "gfx1102",
        "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200", "gfx1201",
    ];

    const WMMA_ARCHS: &[&str] = &[
        "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200",
        "gfx1201",
    ];

    const GFX10_SCALAR_ARCHS: &[&str] = &[
        "gfx1010", "gfx1011", "gfx1012", "gfx1013", "gfx1030", "gfx1031", "gfx1032",
    ];

    const NO_WMMA_ARCHS: &[&str] = &["gfx900", "gfx906", "gfx908", "gfx940", "gfx941", "gfx942"];

    #[test]
    fn qwen35_is_batchable_la_always_ok() {
        for &arch in BATCHABLE_ARCHS {
            assert!(
                is_batchable_la(DType::MQ4G256, arch),
                "MQ4G256 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::HFQ4G256, arch),
                "HFQ4G256 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::MQ6G256, arch),
                "MQ6G256 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::HFQ6G256, arch),
                "HFQ6G256 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::Q8_0, arch),
                "Q8_0 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::ParoQ4G128, arch),
                "ParoQ4G128 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::F32, arch),
                "F32 should batch on {arch}"
            );
        }
    }

    #[test]
    fn qwen35_is_batchable_la_mq4_v2_gfx11_and_gfx12() {
        // MQ4G256V2 (qt44) now batches on gfx11 (gfx1100/1101/1102/1150/1151)
        // and gfx12 via WMMA; MQ4CG256 remains gfx12-only. Proves the
        // parity-proven WMMA kernels are admitted on HasWmma arches.
        for arch in [
            "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
        ] {
            assert!(
                is_batchable_la(DType::MQ4G256V2, arch),
                "MQ4G256V2 should batch on {arch}"
            );
        }
        // non-WMMA must still fall back
        for arch in ["gfx1010", "gfx1030", "gfx942", "gfx906"] {
            assert!(
                !is_batchable_la(DType::MQ4G256V2, arch),
                "MQ4G256V2 must fall back on {arch}"
            );
        }
        // gfx12 unchanged already proven above, but explicitly re-prove
        // that the admit set includes both gfx12 variants
        assert!(is_batchable_la(DType::MQ4G256V2, "gfx1200"));
        assert!(is_batchable_la(DType::MQ4G256V2, "gfx1201"));
        // gfx1100/gfx1151 true (the two parity-proven parts)
        assert!(is_batchable_la(DType::MQ4G256V2, "gfx1100"));
        assert!(is_batchable_la(DType::MQ4G256V2, "gfx1151"));
        // MQ4CG256 remains gfx12-only (must NOT widen)
        for arch in ["gfx1200", "gfx1201"] {
            assert!(
                is_batchable_la(DType::MQ4CG256, arch),
                "MQ4CG256 should batch on {arch}"
            );
        }
        for arch in ["gfx1010", "gfx1100", "gfx1151", "gfx942"] {
            assert!(
                !is_batchable_la(DType::MQ4CG256, arch),
                "MQ4CG256 must fall back on {arch}"
            );
        }
    }

    #[test]
    fn qwen35_is_batchable_la_mq4_v2_env_escape() {
        // HIPFIRE_MQV2_GFX11_WMMA=0 restores fallback ONLY on gfx11; gfx12
        // remains admitted. Use the shared helper directly to avoid global env
        // mutation flakiness in parallel tests — is_batchable_la delegates
        // to `llama::mqv2_wmma_batchable`, which calls this helper verbatim.
        for arch in ["gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151"] {
            assert!(
                !llama::mqv2_gfx11_wmma_enabled_from_env(Some("0"), arch),
                "env=0 should disable {arch}"
            );
            assert!(
                llama::mqv2_gfx11_wmma_enabled_from_env(None, arch),
                "unset should enable {arch}"
            );
            assert!(
                llama::mqv2_gfx11_wmma_enabled_from_env(Some("1"), arch),
                "env=1 should enable {arch}"
            );
        }
        for arch in ["gfx1200", "gfx1201"] {
            assert!(
                llama::mqv2_gfx11_wmma_enabled_from_env(Some("0"), arch),
                "gfx12 unaffected by env=0 on {arch}"
            );
            assert!(
                llama::mqv2_gfx11_wmma_enabled_from_env(None, arch),
                "gfx12 enabled without env on {arch}"
            );
        }
        for arch in ["gfx1010", "gfx942", "gfx1030", "gfx1103", "gfx1152"] {
            assert!(
                !llama::mqv2_gfx11_wmma_enabled_from_env(None, arch),
                "non-WMMA {arch} must never admit"
            );
            assert!(
                !llama::mqv2_gfx11_wmma_enabled_from_env(Some("0"), arch),
                "non-WMMA {arch} with env=0"
            );
            assert!(
                !llama::mqv2_gfx11_wmma_enabled_from_env(Some("1"), arch),
                "non-WMMA {arch} with env=1"
            );
        }
        // Prove gfx12 unchanged via is_batchable_la even with env=0 — the
        // helper above shows helper-level, but also confirm the public gate:
        // We cannot set env globally here without serializing tests, but
        // helper's gfx12=true with env=0 proves the delegate will keep gfx12
        // true when is_batchable_la reads HIPFIRE_MQV2_GFX11_WMMA=0.
    }

    #[test]
    fn qwen35_is_batchable_la_v2_family_gfx11_and_gfx12() {
        for arch in [
            "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
        ] {
            assert!(is_batchable_la(DType::MQ6G256V2, arch), "MQ6V2 on {arch}");
            assert!(is_batchable_la(DType::MQ5G256V2, arch), "MQ5V2 on {arch}");
            assert!(is_batchable_la(DType::MQ3G256V2, arch), "MQ3V2 on {arch}");
            assert!(is_batchable_la(DType::MQ2G256V2, arch), "MQ2V2 on {arch}");
        }
        for arch in ["gfx942", "gfx1010", "gfx1030", "gfx1103", "gfx1152"] {
            assert!(
                !is_batchable_la(DType::MQ6G256V2, arch),
                "MQ6V2 not on {arch}"
            );
            assert!(
                !is_batchable_la(DType::MQ5G256V2, arch),
                "MQ5V2 not on {arch}"
            );
            assert!(
                !is_batchable_la(DType::MQ3G256V2, arch),
                "MQ3V2 not on {arch}"
            );
            assert!(
                !is_batchable_la(DType::MQ2G256V2, arch),
                "MQ2V2 not on {arch}"
            );
        }
        // Distinguish V2 from legacy: same group bytes but different DType
        assert_ne!(DType::MQ6G256, DType::MQ6G256V2);
        assert_ne!(DType::MQ3G256, DType::MQ3G256V2);
        assert_ne!(DType::MQ4G256, DType::MQ4G256V2);
        assert_ne!(DType::MQ4G256, DType::MQ6G256V2);
        // Byte counts per contract
        assert_eq!(rdna_compute::MQ6G256V2_GROUP_BYTES, 200);
        assert_eq!(rdna_compute::MQ5G256V2_GROUP_BYTES, 168);
        assert_eq!(rdna_compute::MQ3G256V2_GROUP_BYTES, 104);
        assert_eq!(rdna_compute::MQ2G256V2_GROUP_BYTES, 72);
        assert_eq!(rdna_compute::MQ4V2_GROUP_BYTES, 136);
    }

    #[test]
    fn mqv2_admit_llama_qwen35_lockstep() {
        // True contract (PR #690 hw-gate regression): llama and qwen35 agree
        // on every NON-V2 dtype, but for the V2 family they deliberately
        // diverge — qwen35's `forward_prefill_chunk` has V2 dispatch arms
        // (206 hits) so it admits V2 via the shared
        // `llama::mqv2_wmma_batchable` rule, while llama's chunk path has no
        // V2 arms (`qkv_is_mq`/`wo_is_mq`/`ffn_is_mq`/`w_down_is_mq` list
        // only V1 dtypes) so `llama::is_batchable_la` refuses V2 everywhere
        // and stays on per-token decode. Admitting V2 to the llama path
        // would skip the FWHT rotate and run V1 `hfq4g256` launchers on V2
        // blobs — silently incoherent prefill.
        // Non-V2 agreement across the 5-arch sample.
        let non_v2 = [
            DType::MQ4G256,
            DType::HFQ4G256,
            DType::MQ6G256,
            DType::MQ3G256,
            DType::MFP4G32,
            DType::Q8_0,
        ];
        for dt in non_v2 {
            for arch in ["gfx1100", "gfx1151", "gfx1201", "gfx1030", "gfx1010"] {
                assert_eq!(
                    llama::is_batchable_la(dt, arch),
                    is_batchable_la(dt, arch),
                    "lockstep drift for {dt:?} on {arch}"
                );
            }
        }
        // V2 divergence: qwen35 admits on gfx11/gfx12 (kill-switch at its
        // default ON here — both gates read `HIPFIRE_MQV2_GFX11_WMMA`
        // identically, so with the var unset gfx11 admits), refuses
        // pre-WMMA; llama refuses on all 5 arches.
        let v2 = [
            DType::MQ4G256V2,
            DType::MQ6G256V2,
            DType::MQ5G256V2,
            DType::MQ3G256V2,
            DType::MQ2G256V2,
            DType::MQ4CG256,
        ];
        for dt in v2 {
            for arch in ["gfx1100", "gfx1151"] {
                // MQ4CG256 is gfx12-only by intent in BOTH callers.
                if dt == DType::MQ4CG256 {
                    assert!(
                        !is_batchable_la(dt, arch),
                        "qwen35 must refuse {dt:?} on {arch}"
                    );
                } else {
                    assert!(
                        is_batchable_la(dt, arch),
                        "qwen35 should admit {dt:?} on {arch}"
                    );
                }
                assert!(
                    !llama::is_batchable_la(dt, arch),
                    "llama must refuse {dt:?} on {arch}"
                );
            }
            assert!(
                is_batchable_la(dt, "gfx1201"),
                "qwen35 should admit {dt:?} on gfx1201"
            );
            assert!(
                !llama::is_batchable_la(dt, "gfx1201"),
                "llama must refuse {dt:?} on gfx1201"
            );
            for arch in ["gfx1030", "gfx1010"] {
                assert!(
                    !is_batchable_la(dt, arch),
                    "qwen35 must refuse {dt:?} on {arch}"
                );
                assert!(
                    !llama::is_batchable_la(dt, arch),
                    "llama must refuse {dt:?} on {arch}"
                );
            }
        }
        // Absolute pins so the test also fails if the shared rule itself
        // regresses, not just on caller drift.
        assert!(is_batchable_la(DType::MQ4G256V2, "gfx1201"));
        assert!(!is_batchable_la(DType::MQ4G256V2, "gfx1030"));
        assert!(!is_batchable_la(DType::MQ4CG256, "gfx1100"));
        assert!(!llama::is_batchable_la(DType::MQ4G256V2, "gfx1201"));
    }

    #[test]
    fn qwen35_v2_dense_keys_are_exact_no_hfq4_default() {
        // Contract: every admitted V2 dtype maps 1:1 to its exact V2 kernel
        // in every dense operation (plain, residual, QKV, QKVZA, gate_up).
        // All V2 widths admit on both gfx11 and gfx12 (HasWmma); no qt47-50 falls into HFQ4/default/wildcard.
        use hipfire_dispatch::families::fused_qkv::{fused_qkv_key_for, fused_qkvza_key_for};
        use hipfire_dispatch::types::KernelKey;
        use rdna_compute::DType;
        let cases: &[(DType, KernelKey, KernelKey, KernelKey, KernelKey, &str)] = &[
            (
                DType::MQ6G256V2,
                KernelKey::GemmMq6G256V2,
                KernelKey::GemmMq6G256V2Residual,
                KernelKey::FusedQkvMq6G256V2,
                KernelKey::FusedQkvzaMq6G256V2,
                "qt47",
            ),
            (
                DType::MQ5G256V2,
                KernelKey::GemmMq5G256V2,
                KernelKey::GemmMq5G256V2Residual,
                KernelKey::FusedQkvMq5G256V2,
                KernelKey::FusedQkvzaMq5G256V2,
                "qt48",
            ),
            (
                DType::MQ3G256V2,
                KernelKey::GemmMq3G256V2,
                KernelKey::GemmMq3G256V2Residual,
                KernelKey::FusedQkvMq3G256V2,
                KernelKey::FusedQkvzaMq3G256V2,
                "qt49",
            ),
            (
                DType::MQ2G256V2,
                KernelKey::GemmMq2G256V2,
                KernelKey::GemmMq2G256V2Residual,
                KernelKey::FusedQkvMq2G256V2,
                KernelKey::FusedQkvzaMq2G256V2,
                "qt50",
            ),
        ];
        for (dt, exp_plain, exp_resid, exp_qkv, exp_qkvza, qt) in cases {
            // plain GEMM via GemmFamily::resolve through direct key
            // (plain keys are gfx12-only, but must be exact, not HFQ4)
            assert_ne!(
                *exp_plain,
                KernelKey::GemmHfq4G256,
                "{} plain must not be HFQ4",
                qt
            );
            assert_ne!(
                *exp_resid,
                KernelKey::GemmHfq4G256Residual,
                "{} residual must not be HFQ4",
                qt
            );
            // fused helpers
            assert_eq!(fused_qkv_key_for(*dt), *exp_qkv, "{} qkv", qt);
            assert_eq!(fused_qkvza_key_for(*dt), *exp_qkvza, "{} qkvza", qt);
            assert_eq!(
                fused_gate_up_key_for(*dt),
                match dt {
                    DType::MQ6G256V2 => KernelKey::FusedGateUpMq6G256V2,
                    DType::MQ5G256V2 => KernelKey::FusedGateUpMq5G256V2,
                    DType::MQ3G256V2 => KernelKey::FusedGateUpMq3G256V2,
                    DType::MQ2G256V2 => KernelKey::FusedGateUpMq2G256V2,
                    _ => unreachable!(),
                },
                "{} gate_up",
                qt
            );
            assert_eq!(
                residual_gemm_key_for(*dt),
                *exp_resid,
                "{} resid helper",
                qt
            );
            // Ensure helpers never return HFQ4 for V2
            assert_ne!(fused_qkv_key_for(*dt), KernelKey::FusedQkvHfq4G256);
            assert_ne!(fused_qkvza_key_for(*dt), KernelKey::FusedQkvzaHfq4G256);
            assert_ne!(fused_gate_up_key_for(*dt), KernelKey::FusedGateUpHfq4G256);
            assert_ne!(residual_gemm_key_for(*dt), KernelKey::GemmHfq4G256Residual);
            // batchable on both gfx11 and gfx12 (HasWmma)
            assert!(is_batchable_la(*dt, "gfx1201"));
            assert!(is_batchable_la(*dt, "gfx1100"));
        }
        // Legacy must stay on HFQ4 path
        assert_eq!(
            hipfire_dispatch::families::fused_qkv::fused_qkv_key_for(DType::HFQ4G256),
            KernelKey::FusedQkvHfq4G256
        );
    }

    #[test]
    fn qwen35_is_batchable_la_mq3_wmma_and_gfx10_scalar() {
        for &arch in WMMA_ARCHS {
            assert!(
                is_batchable_la(DType::MQ3G256, arch),
                "MQ3G256 should batch on {arch} (WMMA)"
            );
        }
        for &arch in GFX10_SCALAR_ARCHS {
            assert!(
                is_batchable_la(DType::MQ3G256, arch),
                "MQ3G256 should batch on {arch} (scalar)"
            );
        }
        for &arch in NO_WMMA_ARCHS {
            assert!(
                !is_batchable_la(DType::MQ3G256, arch),
                "MQ3G256 must fall back on {arch}"
            );
        }
    }

    #[test]
    fn qwen35_is_batchable_la_fp4_only_on_wmma() {
        for &arch in WMMA_ARCHS {
            assert!(
                is_batchable_la(DType::HFP4G32, arch),
                "HFP4G32 should batch on {arch}"
            );
            assert!(
                is_batchable_la(DType::MFP4G32, arch),
                "MFP4G32 should batch on {arch}"
            );
        }
        for &arch in NO_WMMA_ARCHS {
            assert!(
                !is_batchable_la(DType::HFP4G32, arch),
                "HFP4G32 must fall back on {arch}"
            );
            assert!(
                !is_batchable_la(DType::MFP4G32, arch),
                "MFP4G32 must fall back on {arch}"
            );
        }
    }

    #[test]
    fn qwen35_is_batchable_la_lloyd_mq3_mq4_on_gfx11_and_gfx12() {
        // MQ3-Lloyd admits gfx1100/1101/1102/1150/1151 — the MQ3-Lloyd GEMM
        // source selectors DO ship a gfx1150 kernel.
        for &arch in &["gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151"] {
            assert!(
                is_batchable_la(DType::MQ3G256Lloyd, arch),
                "MQ3G256Lloyd should batch on {arch}"
            );
        }
        // MQ4-Lloyd admits gfx1100/1101/1102/1151 ONLY (NOT gfx1150). ANTIBLEED
        // admit-vs-select fix: the MQ4-Lloyd GEMM source selectors panic on
        // gfx1150 (no kernel), so admitting it upstream would crash at lookup.
        for &arch in &["gfx1100", "gfx1101", "gfx1102", "gfx1151"] {
            assert!(
                is_batchable_la(DType::MQ4G256Lloyd, arch),
                "MQ4G256Lloyd should batch on {arch}"
            );
        }
        assert!(
            !is_batchable_la(DType::MQ4G256Lloyd, "gfx1150"),
            "gfx1150 must NOT admit Lloyd MQ4 (no MQ4-Lloyd kernel source → panic)"
        );
        // gfx1152 not in either admit list
        assert!(
            !is_batchable_la(DType::MQ3G256Lloyd, "gfx1152"),
            "gfx1152 should NOT admit Lloyd MQ3"
        );
        assert!(
            !is_batchable_la(DType::MQ4G256Lloyd, "gfx1152"),
            "gfx1152 should NOT admit Lloyd MQ4"
        );
        // gfx12 always-on after gfx1201 validation (PR #195)
        assert!(
            is_batchable_la(DType::MQ3G256Lloyd, "gfx1200"),
            "gfx1200 MQ3G256Lloyd should batch"
        );
        assert!(
            is_batchable_la(DType::MQ4G256Lloyd, "gfx1200"),
            "gfx1200 MQ4G256Lloyd should batch"
        );
    }

    #[test]
    fn qwen35_is_batchable_la_unsupported_dtypes() {
        for &arch in WMMA_ARCHS {
            assert!(!is_batchable_la(DType::Q4K, arch), "Q4K must fall back");
            assert!(!is_batchable_la(DType::Q6K, arch), "Q6K must fall back");
            assert!(
                !is_batchable_la(DType::Q4F16G64, arch),
                "Q4F16G64 must fall back"
            );
            assert!(
                !is_batchable_la(DType::Q4F16G32, arch),
                "Q4F16G32 must fall back"
            );
            assert!(
                !is_batchable_la(DType::MQ2G256, arch),
                "MQ2G256 must fall back"
            );
            assert!(
                !is_batchable_la(DType::MQ8G256, arch),
                "MQ8G256 must fall back"
            );
            assert!(
                !is_batchable_la(DType::HFQ2G256, arch),
                "HFQ2G256 must fall back"
            );
            assert!(
                !is_batchable_la(DType::BF16, arch),
                "BF16 must fall back until the batched BF16 dispatch family is wired"
            );
        }
    }

    // ── Qwen3.5 MoE dispatch predicates ──────────────────────────

    #[test]
    fn moe_ffn_has_mq3_detects_mq3_in_experts() {
        // Smoke-test the renamed split predicates to ensure they compile and
        // the DType-level logic is preserved.
        // MoeFfnWeights requires GPU-backed tensors; the real DType dispatch
        // is tested via moe_prefill_rejects_mq3_before_admission_work below.
        let _mq3_dt = DType::MQ3G256;
        let _mq3l_dt = DType::MQ3G256Lloyd;
        let _mq4_dt = DType::MQ4G256;
        // Verify the predicates are callable (the MoeFfnWeights tensor
        // requirement prevents constructing a real fixture here; logic
        // coverage is via the admission tests below that use MoePrefillDtypes).
    }

    #[test]
    fn moe_prefill_topk_shape_requires_k8_and_bounded_experts() {
        assert!(moe_prefill_topk_shape_supported(8, 256));
        assert!(moe_prefill_topk_shape_supported(8, 1024));
        assert!(!moe_prefill_topk_shape_supported(4, 256));
        assert!(!moe_prefill_topk_shape_supported(8, 1025));
    }

    #[test]
    fn moe_prefill_admits_mq4_as_known_good_control() {
        let dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, false, false, false
        ));
    }

    #[test]
    fn moe_prefill_admits_graded_mixed_experts_via_merged_kernel() {
        // Graded T3-3L: routed experts dtype-mixed (hot MQ6 / mid MQ4 / cold
        // MQ3-Lloyd), shared expert + router MQ4. The merged grouped-WMMA prefill
        // kernel serves the routed experts, so this MUST be batched-admissible —
        // otherwise it silently drops to the per-token prefill fallback at
        // ~decode speed and the merged kernel never fires.
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.routed_mixed_merged = true;
        dtypes.expert_gate_up_uniform = false;
        dtypes.expert_down_uniform = false;
        dtypes.expert_down = DType::MQ3G256Lloyd; // representative cold-tier dtype
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false, false
        ));
        // The same mixed file WITHOUT the merged-kernel tag table is NOT admissible.
        dtypes.routed_mixed_merged = false;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false, false
        ));
    }

    /// Plain (affine) MQ3G256 is NOT the Lloyd codebook dtype and has no
    /// grouped-GEMM arm — it must stay rejected even with the codebook admit on.
    /// Same for MQ3 anywhere STRUCTURAL (router / shared expert), which the
    /// codebook arm never covers.
    #[test]
    fn moe_prefill_rejects_mq3_before_admission_work() {
        for admit_codebook in [false, true] {
            let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
            dtypes.expert_gate_up = DType::MQ3G256;
            assert!(!moe_ffn_batched_admissible_for_dtypes(
                &dtypes,
                true,
                false,
                false,
                admit_codebook
            ));

            let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
            dtypes.shared_expert_down = DType::MQ3G256;
            assert!(!moe_ffn_batched_admissible_for_dtypes(
                &dtypes,
                true,
                false,
                false,
                admit_codebook
            ));

            // MQ3-Lloyd routed pair with an MQ3-Lloyd SHARED expert: routed side
            // is fine, shared side is not — the codebook arm validates the shared
            // expert exactly like the MQ4 arm, so this must still reject.
            let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
            dtypes.expert_gate_up = DType::MQ2G256Lloyd;
            dtypes.expert_down = DType::MQ3G256Lloyd;
            dtypes.shared_expert_down = DType::MQ3G256Lloyd;
            assert!(!moe_ffn_batched_admissible_for_dtypes(
                &dtypes,
                true,
                false,
                false,
                admit_codebook
            ));
        }
    }

    /// MQ2/MQ3-G256-GL routed experts have DECODE kernels only — the four
    /// indexed MoE GEMVs. There is no grouped-WMMA GEMM and no batched indexed
    /// GEMV for the GL layouts, so the batched-prefill gate must reject them and
    /// let the model prefill through the per-token path (correct, just slower).
    ///
    /// Admitting GL here would send a `[M*gpr*64 B idx][M*gpr*2 B scale]` SoA
    /// blob into an HFQ4-layout (136 B/group interleaved) GEMM — out-of-bounds
    /// reads and garbage, exactly the failure the MQ3 guard above exists for.
    #[test]
    fn moe_prefill_rejects_gl_routed_experts() {
        for gl in [DType::MQ2G256GL, DType::MQ3G256GL] {
            for admit_mq6 in [false, true] {
                let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
                dtypes.expert_gate_up = gl;
                dtypes.expert_down = gl;
                assert!(
                    !moe_ffn_batched_admissible_for_dtypes(&dtypes, admit_mq6, true, true, true),
                    "{gl:?} must not be batched-prefill admissible (admit_mq6={admit_mq6})"
                );
            }
        }
        // Per-projection GL mix (the target SKU: gate_up MQ2-GL, down MQ3-GL).
        // Must reject with the codebook admit BOTH off and on — GL is excluded
        // from `routed_codebook_grouped_supported` on purpose.
        for admit_codebook in [false, true] {
            let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
            dtypes.expert_gate_up = DType::MQ2G256GL;
            dtypes.expert_down = DType::MQ3G256GL;
            assert!(!moe_ffn_batched_admissible_for_dtypes(
                &dtypes,
                true,
                true,
                true,
                admit_codebook
            ));
            // Half-GL pairs must not sneak through the codebook arm either.
            let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
            dtypes.expert_gate_up = DType::MQ2G256GL;
            dtypes.expert_down = DType::MQ3G256Lloyd;
            assert!(!moe_ffn_batched_admissible_for_dtypes(
                &dtypes,
                true,
                true,
                true,
                admit_codebook
            ));
            let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
            dtypes.expert_gate_up = DType::MQ2G256Lloyd;
            dtypes.expert_down = DType::MQ3G256GL;
            assert!(!moe_ffn_batched_admissible_for_dtypes(
                &dtypes,
                true,
                true,
                true,
                admit_codebook
            ));
        }
    }

    #[test]
    fn moe_prefill_mq6_requires_explicit_admission() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.shared_expert_scalar_gate = DType::Q8_0;
        dtypes.shared_expert_gate = DType::MQ6G256;
        dtypes.shared_expert_up = DType::MQ6G256;
        dtypes.shared_expert_down = DType::MQ6G256;
        dtypes.expert_gate_up = DType::MQ6G256;
        dtypes.expert_down = DType::MQ6G256;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, false, false, false
        ));
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false, false
        ));
    }

    /// Uniform MQ4G256V2 shared+routed: always admissible (no MQ6 gate). Path-2
    /// uses `gemm_mq4g256v2_moe_grouped_wmma_k2` / `_gfx12` after dispatch —
    /// never HFQ4 V1. Router/scalar-gate stay MQ4V2 (admitted like MQ4).
    #[test]
    fn moe_prefill_admits_uniform_mq4v2_without_mq6_gate() {
        let dtypes = MoePrefillDtypes::uniform(DType::MQ4G256V2);
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, false, false, false
        ));
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false, false
        ));
        assert!(routed_uniform_mqv2_grouped_supported(DType::MQ4G256V2));
        assert!(!routed_codebook_grouped_supported(DType::MQ4G256V2));
    }

    /// Uniform MQ6G256V2 shared+routed: requires `admit_mq6` (same gate as V1
    /// MQ6). Router stays MQ4 so router_ok holds; shared/routed projections
    /// are exact MQ6V2 — Path-2 `gemm_mq6g256v2_moe_grouped_wmma_k2`.
    #[test]
    fn moe_prefill_admits_uniform_mq6v2_only_with_mq6_gate() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.shared_expert_scalar_gate = DType::Q8_0;
        dtypes.shared_expert_gate = DType::MQ6G256V2;
        dtypes.shared_expert_up = DType::MQ6G256V2;
        dtypes.shared_expert_down = DType::MQ6G256V2;
        dtypes.expert_gate_up = DType::MQ6G256V2;
        dtypes.expert_down = DType::MQ6G256V2;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, false, false, false
        ));
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false, false
        ));
        assert!(routed_uniform_mqv2_grouped_supported(DType::MQ6G256V2));
        assert!(!routed_codebook_grouped_supported(DType::MQ6G256V2));
    }

    /// MQ4V2 shared + MQ6V2 routed (and the reverse) under admit_mq6 — exact
    /// dual-half dtypes, no V1 collapse. Cross-family pairs still go through
    /// the main admit arm once MQ6 is gated on.
    #[test]
    fn moe_prefill_admits_mq4v2_mq6v2_cross_with_mq6_gate() {
        // MQ4V2 shared, MQ6V2 routed.
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256V2);
        dtypes.expert_gate_up = DType::MQ6G256V2;
        dtypes.expert_down = DType::MQ6G256V2;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, false, false, false
        ));
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false, false
        ));

        // MQ6V2 shared, MQ4V2 routed.
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.shared_expert_scalar_gate = DType::Q8_0;
        dtypes.shared_expert_gate = DType::MQ6G256V2;
        dtypes.shared_expert_up = DType::MQ6G256V2;
        dtypes.shared_expert_down = DType::MQ6G256V2;
        dtypes.expert_gate_up = DType::MQ4G256V2;
        dtypes.expert_down = DType::MQ4G256V2;
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false, false
        ));
    }

    /// Graded mixed-tag files (tags 7..18) only need the SHARED expert batchable;
    /// MQ4V2 shared always works, MQ6V2 shared needs admit_mq6.
    #[test]
    fn moe_prefill_graded_mixed_admits_mqv2_shared() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256V2);
        dtypes.routed_mixed_merged = true;
        dtypes.expert_gate_up_uniform = false;
        dtypes.expert_down_uniform = false;
        dtypes.expert_down = DType::MQ3G256Lloyd;
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, false, false, false
        ));

        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.shared_expert_scalar_gate = DType::Q8_0;
        dtypes.shared_expert_gate = DType::MQ6G256V2;
        dtypes.shared_expert_up = DType::MQ6G256V2;
        dtypes.shared_expert_down = DType::MQ6G256V2;
        dtypes.routed_mixed_merged = true;
        dtypes.expert_gate_up_uniform = false;
        dtypes.expert_down_uniform = false;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, false, false, false
        ));
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false, false
        ));
    }

    /// MQ2/3/5V2 are dense-only V2 — never MoE grouped prefill admissible.
    #[test]
    fn moe_prefill_rejects_mq235v2_routed() {
        for v2 in [DType::MQ2G256V2, DType::MQ3G256V2, DType::MQ5G256V2] {
            let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
            dtypes.expert_gate_up = v2;
            dtypes.expert_down = v2;
            assert!(
                !moe_ffn_batched_admissible_for_dtypes(&dtypes, true, true, true, true),
                "{v2:?} must not be MoE batched-prefill admissible"
            );
            assert!(!routed_uniform_mqv2_grouped_supported(v2));
            assert!(!routed_codebook_grouped_supported(v2));
        }
    }

    /// Dense residual / fused-gate-up keys for shared-expert V2 stay exact —
    /// never HFQ4/HFQ6 V1 residual_sigmoid aliases.
    #[test]
    fn moe_shared_mqv2_dense_keys_never_collapse_to_v1() {
        use hipfire_dispatch::types::KernelKey;

        assert_eq!(
            fused_gate_up_key_for(DType::MQ4G256V2),
            KernelKey::FusedGateUpMq4G256V2
        );
        assert_eq!(
            fused_gate_up_key_for(DType::MQ6G256V2),
            KernelKey::FusedGateUpMq6G256V2
        );
        assert_ne!(
            fused_gate_up_key_for(DType::MQ4G256V2),
            KernelKey::FusedGateUpHfq4G256
        );
        assert_ne!(
            fused_gate_up_key_for(DType::MQ6G256V2),
            KernelKey::FusedGateUpHfq6G256
        );

        assert_eq!(
            residual_gemm_key_for(DType::MQ4G256V2),
            KernelKey::GemmMq4G256V2Residual
        );
        assert_eq!(
            residual_gemm_key_for(DType::MQ6G256V2),
            KernelKey::GemmMq6G256V2Residual
        );
        assert_ne!(
            residual_gemm_key_for(DType::MQ4G256V2),
            KernelKey::GemmHfq4G256Residual
        );
        assert_ne!(
            residual_gemm_key_for(DType::MQ6G256V2),
            KernelKey::GemmHfq6G256Residual
        );

        // Grouped-supported lockstep with Path-2 launchers (gfx11+gfx12).
        assert!(routed_uniform_mqv2_grouped_supported(DType::MQ4G256V2));
        assert!(routed_uniform_mqv2_grouped_supported(DType::MQ6G256V2));
        // V1 stays on the codebook/default arms, not the V2 helper.
        assert!(!routed_uniform_mqv2_grouped_supported(DType::MQ4G256));
        assert!(!routed_uniform_mqv2_grouped_supported(DType::MQ6G256));
    }

    #[test]
    fn moe_prefill_rejects_nonuniform_expert_projections() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.expert_gate_up_uniform = false;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false, false
        ));

        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.expert_down_uniform = false;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false, false
        ));
    }

    #[test]
    fn moe_prefill_shared_gate_up_must_be_one_dtype() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.shared_expert_up = DType::MQ6G256;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false, false
        ));
    }

    #[test]
    fn moe_prefill_rejects_cross_version_shared_gate_mq4_up_mq4v2() {
        // Fused shared gate+up handles one layout per launch (MQ4 vs MQ4V2
        // dual-half headers differ → stride/decoder mismatch). Exact equality
        // required in every fused admission arm (uniform, graded merged,
        // codebook). Both orderings reject; uniform V2 still admits.
        for (gate, up) in [
            (DType::MQ4G256, DType::MQ4G256V2),
            (DType::MQ4G256V2, DType::MQ4G256),
        ] {
            // Uniform arm (no mixed tags, no codebook).
            let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
            dtypes.shared_expert_gate = gate;
            dtypes.shared_expert_up = up;
            assert!(
                !moe_ffn_batched_admissible_for_dtypes(&dtypes, false, false, false, false),
                "uniform gate={gate:?} up={up:?} must reject without mq6 gate"
            );
            assert!(
                !moe_ffn_batched_admissible_for_dtypes(&dtypes, true, false, false, false),
                "uniform gate={gate:?} up={up:?} must reject with mq6 gate"
            );
            // Graded merged arm (routed_mixed_merged = true).
            let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
            dtypes.routed_mixed_merged = true;
            dtypes.expert_gate_up_uniform = false;
            dtypes.expert_down_uniform = false;
            dtypes.shared_expert_gate = gate;
            dtypes.shared_expert_up = up;
            assert!(
                !moe_ffn_batched_admissible_for_dtypes(&dtypes, false, false, false, false),
                "graded gate={gate:?} up={up:?} must reject"
            );
            assert!(
                !moe_ffn_batched_admissible_for_dtypes(&dtypes, true, false, false, false),
                "graded gate={gate:?} up={up:?} must reject with mq6"
            );
            // Codebook arm (uniform codebook pair admitted via grouped GEMM).
            let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
            dtypes.expert_gate_up = DType::MQ2G256Lloyd;
            dtypes.expert_down = DType::MQ3G256Lloyd;
            dtypes.shared_expert_gate = gate;
            dtypes.shared_expert_up = up;
            assert!(
                !moe_ffn_batched_admissible_for_dtypes(&dtypes, false, false, false, true),
                "codebook gate={gate:?} up={up:?} must reject"
            );
        }
        // Exact V2 support preserved: uniform MQ4V2 gate/up admits.
        let dtypes = MoePrefillDtypes::uniform(DType::MQ4G256V2);
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, false, false, false
        ));
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false, false
        ));
    }

    #[test]
    fn moe_prefill_rejects_cross_version_shared_gate_mq6_up_mq6v2() {
        // Same exact-equality rule for MQ6 vs MQ6V2 (qt46 vs qt47): different
        // G256 packing (HFQ6 vs dual-half MQ6V2) → never collapse.
        for (gate, up) in [
            (DType::MQ6G256, DType::MQ6G256V2),
            (DType::MQ6G256V2, DType::MQ6G256),
        ] {
            let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
            dtypes.shared_expert_scalar_gate = DType::Q8_0;
            dtypes.shared_expert_gate = gate;
            dtypes.shared_expert_up = up;
            dtypes.shared_expert_down = gate;
            dtypes.expert_gate_up = gate;
            dtypes.expert_down = gate;
            assert!(
                !moe_ffn_batched_admissible_for_dtypes(&dtypes, true, false, false, false),
                "mq6 uniform gate={gate:?} up={up:?} must reject"
            );
            // Graded merged arm with MQ6 cross-version shared expert.
            let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
            dtypes.shared_expert_scalar_gate = DType::Q8_0;
            dtypes.shared_expert_gate = gate;
            dtypes.shared_expert_up = up;
            dtypes.shared_expert_down = gate;
            dtypes.routed_mixed_merged = true;
            dtypes.expert_gate_up_uniform = false;
            dtypes.expert_down_uniform = false;
            assert!(
                !moe_ffn_batched_admissible_for_dtypes(&dtypes, true, false, false, false),
                "mq6 graded gate={gate:?} up={up:?} must reject"
            );
        }
        // Exact V2 support preserved: uniform MQ6V2 gate/up admits with mq6 gate.
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.shared_expert_scalar_gate = DType::Q8_0;
        dtypes.shared_expert_gate = DType::MQ6G256V2;
        dtypes.shared_expert_up = DType::MQ6G256V2;
        dtypes.shared_expert_down = DType::MQ6G256V2;
        dtypes.expert_gate_up = DType::MQ6G256V2;
        dtypes.expert_down = DType::MQ6G256V2;
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false, false
        ));
    }

    #[test]
    fn moe_prefill_admits_paro_when_enabled() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::ParoQ4G128);
        dtypes.router = DType::F32;
        dtypes.shared_expert_scalar_gate = DType::F32;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false, false
        ));
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, true, false, false
        ));
    }

    #[test]
    fn moe_prefill_admits_e8_only_with_arch_gate() {
        // A3B mfp4-E8: Q8 router/scalar-gate/shared-expert + E8 routed experts.
        let mut dtypes = MoePrefillDtypes::uniform(DType::Q8_0);
        dtypes.expert_gate_up = DType::MFP4G32E8;
        dtypes.expert_down = DType::MFP4G32E8;
        // Without the arch gate (non-gfx1151), E8 is rejected.
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, false, false, false
        ));
        // With the gfx1151 arch gate, the Q8-shared + E8-routed layer admits.
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, false, true, false
        ));
    }

    /// The antirez asymmetric routed pair (gate_up MQ2-Lloyd / down MQ3-Lloyd)
    /// on an MQ4 shared expert + MQ4 router — the shape every current low-bit
    /// a3b SKU ships. Rejected without the codebook admit (today's behavior:
    /// per-token fallback), admitted with it.
    #[test]
    fn moe_prefill_admits_uniform_codebook_pair_only_with_gate() {
        let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
        dtypes.expert_gate_up = DType::MQ2G256Lloyd;
        dtypes.expert_down = DType::MQ3G256Lloyd;
        assert!(!moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false, false
        ));
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, true, false, false, true
        ));
        // Also with the MQ6 admit off — the codebook arm does not depend on it
        // when the shared expert is plain MQ4.
        assert!(moe_ffn_batched_admissible_for_dtypes(
            &dtypes, false, false, false, true
        ));
    }

    /// Every uniform gate_up/down permutation over the supported codebook set,
    /// plus the negative cases. Guards `routed_codebook_grouped_supported`
    /// against silent widening.
    #[test]
    fn moe_prefill_codebook_pair_permutations() {
        use DType::{MQ2G256Lloyd, MQ3G256Lloyd, MQ4G256};
        // (gate_up, down, admissible_with_codebook_gate)
        let cases = [
            (MQ2G256Lloyd, MQ3G256Lloyd, true),
            (MQ3G256Lloyd, MQ3G256Lloyd, true),
            (MQ2G256Lloyd, MQ2G256Lloyd, true),
            (MQ2G256Lloyd, MQ4G256, true),
            (MQ4G256, MQ3G256Lloyd, true),
            // Pure MQ4/MQ4 admits via the pre-existing default arm, not this one.
            (MQ4G256, MQ4G256, true),
            // Not in the supported set: plain affine MQ3, MQ5, GL, E8.
            (DType::MQ3G256, MQ3G256Lloyd, false),
            (MQ2G256Lloyd, DType::MQ3G256, false),
            (DType::MQ5G256, MQ3G256Lloyd, false),
            (DType::MQ2G256GL, MQ3G256Lloyd, false),
            (MQ2G256Lloyd, DType::MQ3G256GL, false),
            (DType::MFP4G32E8, MQ3G256Lloyd, false),
        ];
        for (gu, dn, want) in cases {
            let mut dtypes = MoePrefillDtypes::uniform(MQ4G256);
            dtypes.expert_gate_up = gu;
            dtypes.expert_down = dn;
            let got = moe_ffn_batched_admissible_for_dtypes(&dtypes, false, false, false, true);
            assert_eq!(got, want, "gate_up={gu:?} down={dn:?}");
        }
    }

    /// Non-uniform routed experts without a tag table stay rejected even for a
    /// supported codebook pair — `dispatch_grouped_gemm` would apply experts[0]'s
    /// group stride to every expert.
    #[test]
    fn moe_prefill_codebook_pair_requires_uniform_projections() {
        for (gu_uniform, dn_uniform) in [(false, true), (true, false), (false, false)] {
            let mut dtypes = MoePrefillDtypes::uniform(DType::MQ4G256);
            dtypes.expert_gate_up = DType::MQ2G256Lloyd;
            dtypes.expert_down = DType::MQ3G256Lloyd;
            dtypes.expert_gate_up_uniform = gu_uniform;
            dtypes.expert_down_uniform = dn_uniform;
            assert!(!moe_ffn_batched_admissible_for_dtypes(
                &dtypes, true, false, false, true
            ));
        }
    }

    /// LOCKSTEP INVARIANT: anything the codebook admission arm accepts must NOT
    /// trip the MQ3-in-MoE refusal, or `forward_prefill_batch_with_pbs_opts`
    /// hard-errors on a model it just declared eligible. Walks the same routed
    /// pairs through both predicates and asserts they never disagree that way.
    #[test]
    fn mq3_refusal_never_fires_on_an_admitted_codebook_pair() {
        use DType::{MQ2G256Lloyd, MQ3G256Lloyd, MQ2G256GL, MQ3G256, MQ3G256GL, MQ4G256, MQ6G256};
        let pairs = [
            (MQ2G256Lloyd, MQ3G256Lloyd),
            (MQ3G256Lloyd, MQ3G256Lloyd),
            (MQ2G256Lloyd, MQ2G256Lloyd),
            (MQ4G256, MQ3G256Lloyd),
            (MQ2G256Lloyd, MQ4G256),
            (MQ4G256, MQ4G256),
            (MQ3G256, MQ3G256Lloyd),
            (MQ2G256GL, MQ3G256GL),
            (MQ2G256Lloyd, MQ3G256GL),
            (MQ6G256, MQ3G256Lloyd),
        ];
        for admit_codebook in [false, true] {
            for (gu, dn) in pairs {
                let mut dtypes = MoePrefillDtypes::uniform(MQ4G256);
                dtypes.expert_gate_up = gu;
                dtypes.expert_down = dn;
                let admitted = moe_ffn_batched_admissible_for_dtypes(
                    &dtypes,
                    true,
                    false,
                    false,
                    admit_codebook,
                );
                let refused = unsupported_mq3_experts_uniform_from_dtypes(
                    false,
                    [(gu, dn); 4],
                    admit_codebook,
                );
                assert!(
                    !(admitted && refused),
                    "gate_up={gu:?} down={dn:?} admit_codebook={admit_codebook}: \
                     admitted by the batched gate but refused by the MQ3-in-MoE guard"
                );
            }
        }
    }

    /// The narrowed refusal only relaxes for a UNIFORM supported pair: a graded
    /// tag table is out of scope (merged kernel), a mixed no-tags file stays
    /// refused, and MQ3 stays refused when the admit is off.
    #[test]
    fn unsupported_mq3_predicate_shape() {
        use DType::{MQ2G256Lloyd, MQ3G256Lloyd, MQ4G256};
        let pair = (MQ2G256Lloyd, MQ3G256Lloyd);
        // Uniform supported pair, admit on -> no refusal.
        assert!(!unsupported_mq3_experts_uniform_from_dtypes(
            false, [pair; 8], true
        ));
        // Same file, admit off -> refusal stands (today's behavior).
        assert!(unsupported_mq3_experts_uniform_from_dtypes(
            false, [pair; 8], false
        ));
        // Mixed routed dtypes with NO tag table -> refusal stands even with the
        // admit on; the grouped GEMM would use experts[0]'s stride for all.
        assert!(unsupported_mq3_experts_uniform_from_dtypes(
            false,
            [pair, (MQ4G256, MQ3G256Lloyd)],
            true
        ));
        // Graded file (tag table present) -> never this predicate's business.
        assert!(!unsupported_mq3_experts_uniform_from_dtypes(
            true,
            [pair, (MQ4G256, MQ4G256)],
            true
        ));
        // No MQ3 anywhere in the routed experts -> nothing to refuse.
        assert!(!unsupported_mq3_experts_uniform_from_dtypes(
            false,
            [(MQ4G256, MQ4G256); 4],
            false
        ));
        // Empty expert list (paged mode) -> nothing to refuse.
        assert!(!unsupported_mq3_experts_uniform_from_dtypes(
            false,
            std::iter::empty(),
            true
        ));
    }

    /// The codebook admit is arch-gated (WMMA only) and hard-gated on Path 2
    /// being enabled — Path 0/1 have no MQ2/MQ3-Lloyd indexed-GEMV arm, so
    /// admitting with `HIPFIRE_MOE_GROUPED_GEMM=0` would turn a working slow
    /// prefill into an `UnsupportedVariant` hard error.
    #[test]
    fn codebook_batched_admit_arch_and_flag_gates() {
        for arch in [
            "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200",
            "gfx1201",
        ] {
            // DEFAULT OFF until the grouped-WMMA codebook kernels have run on
            // hardware — they had never executed when this landed.
            assert!(
                !codebook_batched_admit_enabled_from_env(None, None, arch),
                "{arch} must NOT default-admit: the grouped-WMMA codebook \
                 kernels are unvalidated (opt in with HIPFIRE_MOE_CODEBOOK_BATCHED=1)"
            );
            // Opt-in works on every WMMA arch.
            assert!(
                codebook_batched_admit_enabled_from_env(Some("1"), None, arch),
                "{arch} should admit under an explicit opt-in"
            );
            // Path 2 disabled => never admit, whatever the codebook var says.
            assert!(!codebook_batched_admit_enabled_from_env(
                None,
                Some("0"),
                arch
            ));
            assert!(!codebook_batched_admit_enabled_from_env(
                Some("1"),
                Some("off"),
                arch
            ));
            // Explicit opt-out.
            assert!(!codebook_batched_admit_enabled_from_env(
                Some("0"),
                None,
                arch
            ));
        }
        // Non-WMMA archs have no grouped-WMMA sister kernel at all.
        for arch in ["gfx1010", "gfx1030", "gfx906", "gfx942"] {
            assert!(
                !codebook_batched_admit_enabled_from_env(None, None, arch),
                "{arch} has no grouped-WMMA kernel — must not default-admit"
            );
            // ...but an explicit =1 is still honored for research / bring-up.
            assert!(codebook_batched_admit_enabled_from_env(
                Some("1"),
                None,
                arch
            ));
        }
    }

    #[test]
    fn mq6_batched_admit_defaults_to_gfx11_and_gfx12() {
        // Post-merge resolved default (gfx11 widen 8d555fc6 ∪ master's gfx1151
        // fast-path): every WMMA arch — all gfx11 (RDNA3/3.5, incl. gfx1100 and
        // gfx1151) and all gfx12 (RDNA4) — default-admits MQ6 batched prefill.
        // Non-WMMA archs (gfx942 CDNA, gfx1030 RDNA2) stay default-off.
        assert!(mq6_batched_admit_enabled_from_env(None, "gfx1201"));
        assert!(mq6_batched_admit_enabled_from_env(None, "gfx1200"));
        assert!(mq6_batched_admit_enabled_from_env(None, "gfx1151"));
        // gfx1100 is now ADMITTED by default (the gfx11 widen), where master
        // had it default-off pending channel testing.
        assert!(mq6_batched_admit_enabled_from_env(None, "gfx1100"));
        assert!(!mq6_batched_admit_enabled_from_env(None, "gfx942"));
        assert!(!mq6_batched_admit_enabled_from_env(None, "gfx1030"));
        // Explicit env overrides still win on every arch.
        assert!(mq6_batched_admit_enabled_from_env(Some("1"), "gfx1151"));
        assert!(mq6_batched_admit_enabled_from_env(Some("1"), "gfx1100"));
        assert!(!mq6_batched_admit_enabled_from_env(Some("0"), "gfx1201"));
        assert!(!mq6_batched_admit_enabled_from_env(Some("0"), "gfx1100"));
    }

    #[test]
    fn q8_prefill_wmma_defaults_on_for_wave32_wmma_arches() {
        assert!(q8_prefill_wmma_enabled_from_env(None, "gfx1201", true));
        assert!(q8_prefill_wmma_enabled_from_env(None, "gfx1100", true));
        assert!(q8_prefill_wmma_enabled_from_env(None, "gfx1151", true));
        assert!(!q8_prefill_wmma_enabled_from_env(None, "gfx1030", false));
        assert!(q8_prefill_wmma_enabled_from_env(Some("1"), "gfx1151", true));
        assert!(!q8_prefill_wmma_enabled_from_env(
            Some("0"),
            "gfx1201",
            true
        ));
        assert!(!q8_prefill_wmma_enabled_from_env(
            Some("1"),
            "gfx1030",
            false
        ));
    }

    #[test]
    fn speculative_verify_has_an_explicit_dispatch_workload() {
        assert_eq!(
            prefill_dispatch_workload(false, false, false),
            DispatchWorkload::Standard
        );
        // DFlash and DSpark/MTP verify request per-token target hidden.
        assert_eq!(
            prefill_dispatch_workload(true, false, false),
            DispatchWorkload::SpeculativeVerify
        );
        assert_eq!(
            prefill_dispatch_workload(false, true, false),
            DispatchWorkload::SpeculativeVerify
        );
        assert_eq!(
            prefill_dispatch_workload(false, false, true),
            DispatchWorkload::SpeculativeVerify
        );
    }

    #[test]
    fn prefill_last_token_logits_policy_requires_explicit_opt_out() {
        assert!(prefill_should_emit_last_token_logits(false, true));
        assert!(prefill_should_emit_last_token_logits(true, true));
        assert!(prefill_should_emit_last_token_logits(false, false));
        assert!(!prefill_should_emit_last_token_logits(true, false));
    }

    #[test]
    fn owned_prefill_scratch_is_right_sized_without_tree_tape() {
        assert_eq!(owned_prefill_scratch_plan(1, 256, false), (2, false));
        assert_eq!(owned_prefill_scratch_plan(2, 256, false), (2, false));
        assert_eq!(owned_prefill_scratch_plan(32, 256, false), (32, false));
        assert_eq!(owned_prefill_scratch_plan(256, 256, false), (256, false));
        assert_eq!(owned_prefill_scratch_plan(1024, 256, false), (256, false));
        // Capped-256 path: internal owned PBS splits at 256 even if arch default is 384.
        assert_eq!(owned_prefill_scratch_plan(1024, 256, false), (256, false));
        // Ordinary gfx1201 owned path may plan 384-row scratch.
        assert_eq!(owned_prefill_scratch_plan(1024, 384, false), (384, false));
        assert_eq!(owned_prefill_scratch_plan(200, 384, false), (200, false));
    }

    #[test]
    fn prefill_chunk_plan_rebalances_singleton_tail() {
        fn plan(mut remaining: usize, max_batch: usize) -> Vec<usize> {
            let mut chunks = Vec::new();
            while remaining > 0 {
                let n = next_prefill_chunk_len(remaining, max_batch)
                    .expect("valid partition should exist");
                chunks.push(n);
                remaining -= n;
            }
            chunks
        }

        assert_eq!(plan(256, 256), vec![256]);
        assert_eq!(plan(257, 256), vec![255, 2]);
        assert_eq!(plan(258, 256), vec![256, 2]);
        assert_eq!(plan(513, 256), vec![256, 255, 2]);
        assert_eq!(plan(129, 128), vec![127, 2]);
        // 384-row ceiling (gfx1201 ordinary path).
        assert_eq!(plan(384, 384), vec![384]);
        assert_eq!(plan(385, 384), vec![383, 2]);
        assert_eq!(plan(768, 384), vec![384, 384]);
    }

    #[test]
    fn prefill_chunk_plan_refuses_unpartitionable_minimum_batch() {
        assert_eq!(next_prefill_chunk_len(3, 2), None);
    }
    #[test]
    fn widened_direct_grouping_keeps_non_rung_tail_attached() {
        assert_eq!(next_exact_prefill_chunk_len(277, 4096), Some(277));
        assert_eq!(next_exact_prefill_chunk_len(513, 4096), Some(511));
        assert_eq!(next_exact_prefill_chunk_len(1025, 4096), Some(1023));
        assert_eq!(next_exact_prefill_chunk_len(1813, 4096), Some(1813));
        assert_eq!(next_exact_prefill_chunk_len(4097, 4096), Some(4095));
        assert_eq!(next_exact_prefill_chunk_len(5909, 4096), Some(4096));
        assert_eq!(next_exact_prefill_chunk_len(5909, 8192), Some(5909));
        assert_eq!(next_exact_prefill_chunk_len(8192, 8192), Some(8192));
        // Sub-512 explicit ceilings keep the legacy schedule exactly.
        assert_eq!(next_exact_prefill_chunk_len(513, 256), Some(256));
        assert_eq!(next_exact_prefill_chunk_len(385, 384), Some(383));
        assert_eq!(next_exact_prefill_chunk_len(1, 4096), None);
        assert_eq!(next_exact_prefill_chunk_len(3, 2), None);
    }

    #[test]
    fn packed_widening_preserves_legacy_commits_and_quant_routes() {
        fn plan(mut rows: usize, ceiling: usize) -> Vec<(usize, bool)> {
            let mut out = Vec::new();
            while rows > 0 {
                let n = next_packed_prefill_chunk_len(rows, ceiling).unwrap();
                assert!(n >= 2 && n <= ceiling);
                let packed = n % 256 == 0;
                let mut left = n;
                while left > 0 {
                    let segment = left.min(512);
                    out.push((segment, packed));
                    left -= segment;
                }
                rows -= n;
            }
            out
        }
        assert_eq!(next_packed_prefill_chunk_len(4097, 4096), Some(3584));
        assert_eq!(next_packed_prefill_chunk_len(513, 4096), Some(511));
        assert_eq!(next_packed_prefill_chunk_len(1, 4096), None);
        for rows in 2..=16385 {
            let base = plan(rows, 512);
            for ceiling in [1024, 2048, 4096, 8192] {
                assert_eq!(plan(rows, ceiling), base, "rows={rows} ceiling={ceiling}");
            }
        }
    }

    #[test]
    fn widened_direct_flattened_commits_equal_legacy_schedule() {
        fn schedule(mut remaining: usize, ceiling: usize, exact: bool) -> Vec<usize> {
            let mut chunks = Vec::new();
            while remaining > 0 {
                let c = if exact {
                    next_exact_prefill_chunk_len(remaining, ceiling)
                } else {
                    next_prefill_chunk_len(remaining, ceiling.min(512))
                }
                .expect("valid partition should exist");
                chunks.push(c);
                remaining -= c;
            }
            chunks
        }
        fn commits(schedule: &[usize]) -> Vec<usize> {
            let mut out = Vec::new();
            for &c in schedule {
                let full = c / WIDENED_COMMIT_ROWS;
                out.extend(std::iter::repeat_n(WIDENED_COMMIT_ROWS, full));
                let tail = c % WIDENED_COMMIT_ROWS;
                if tail != 0 {
                    out.push(tail);
                }
            }
            out
        }
        for &ceiling in &[512usize, 1024, 2048, 4096, 8192] {
            for len in 2..=16385usize {
                let wide = schedule(len, ceiling, true);
                assert!(
                    wide.iter().all(|&c| c <= ceiling),
                    "len {len} ceiling {ceiling}: chunk exceeds ceiling in {wide:?}"
                );
                assert_eq!(
                    commits(&wide),
                    schedule(len, 512, false),
                    "len {len} ceiling {ceiling}: commit sequence diverged from legacy"
                );
            }
        }
    }

    #[test]
    fn widened_serve_grouping_keeps_tail_in_outer_call() {
        fn plan(mut remaining: usize, ceiling: usize) -> Vec<usize> {
            let mut chunks = Vec::new();
            while remaining > 0 {
                let c = ordinary_serve_prefill_chunk_len(remaining, ceiling)
                    .expect("valid partition should exist");
                chunks.push(c);
                remaining -= c;
            }
            chunks
        }
        assert_eq!(plan(1025, 4096), vec![1025]);
        assert_eq!(plan(1813, 4096), vec![1813]);
        assert_eq!(plan(5909, 4096), vec![4096, 1813]);
        assert_eq!(plan(5909, 8192), vec![5909]);
        // At the 512 ceiling the serve split is the legacy min-split.
        for len in 1..=3000usize {
            let mut legacy = Vec::new();
            let mut rem = len;
            while rem > 0 {
                let c = rem.min(512);
                legacy.push(c);
                rem -= c;
            }
            assert_eq!(plan(len, 512), legacy, "len {len}: 512 serve split changed");
        }
        assert_eq!(ordinary_serve_prefill_chunk_len(0, 4096), None);
    }

    /// Dense 27B fixture matching the widened admission shape (48 LA + 16 FA
    /// layers, D5120/I17408, LA K16/V48/D128, FA H24/KV4/D256).
    fn widened_test_config() -> Qwen35Config {
        Qwen35Config {
            dim: 5120,
            n_layers: 64,
            i_gpu_start: 0,
            vocab_size: 152064,
            norm_eps: 1e-6,
            eos_token: 2,
            n_heads: 24,
            n_kv_heads: 4,
            head_dim: 256,
            rope_theta: 500000.0,
            partial_rotary_factor: 0.25,
            is_vl_text: false,
            mrope_interleaved: false,
            mrope_section: [0, 0, 0],
            linear_num_key_heads: 16,
            linear_num_value_heads: 48,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            conv_kernel_dim: 4,
            hidden_dim: 17408,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            has_shared_expert: false,
            norm_topk_prob: false,
            layer_types: (0..48)
                .map(|_| LayerType::LinearAttention)
                .chain((0..16).map(|_| LayerType::FullAttention))
                .collect(),
            paged_experts: false,
            vram_budget_bytes: u64::MAX,
            reap_keep: None,
        }
    }

    #[test]
    fn vmm_stride_uses_resolved_native_or_split_layout() {
        use hipfire_runtime::kv_mode::{KvMode, KvPair, VMode};
        let config = widened_test_config();
        assert_eq!(
            vmm_kv_token_bytes(&config, KvPair::Native(KvMode::Fp8), false),
            Some(33_024)
        );
        assert_eq!(
            vmm_kv_token_bytes(&config, KvPair::Split(KvMode::Q8, VMode::Q8), false),
            Some(34_816)
        );
        assert_eq!(
            vmm_kv_token_bytes(&config, KvPair::Split(KvMode::Fwht3, VMode::Lloyd3), false),
            Some(12_800)
        );
    }

    #[test]
    fn native_mq4_wide_rejects_unaudited_dispatch_overrides() {
        let make = || rdna_compute::FeatureFlags::for_test("gfx1100");
        assert!(native_mq4_widened_flags_admitted("gfx1100", &make(), false));
        assert!(!native_mq4_widened_flags_admitted("gfx1201", &make(), false));
        assert!(!native_mq4_widened_flags_admitted("gfx1100", &make(), true));
        let mut variants = Vec::new();
        let mut f = make(); f.fp16_disabled = true; variants.push(f);
        let mut f = make(); f.rocblas_all_archs = true; variants.push(f);
        let mut f = make(); f.gemv_dp4a = Some(true); variants.push(f);
        let mut f = make(); f.mmq_override = Some(false); variants.push(f);
        let mut f = make(); f.mmq_min_batch = Some(2048); variants.push(f);
        let mut f = make(); f.qkvza_split_tail = true; variants.push(f);
        let mut f = make(); f.mw16 = true; variants.push(f);
        let mut f = make(); f.wo_wmma_variant = Some("k4".into()); variants.push(f);
        // Packed FFN shares the budgeted MMQ slot and keeps native tails.
        let mut packed = make();
        packed.packed_mq4_prefill = true;
        assert!(native_mq4_widened_flags_admitted("gfx1100", &packed, false));
        packed.mmq_override = Some(false);
        assert!(!native_mq4_widened_flags_admitted("gfx1100", &packed, false));
        for f in variants {
            assert!(!native_mq4_widened_flags_admitted("gfx1100", &f, false));
        }
    }

    #[test]
    fn native_mq4_wide_scratch_counts_mmq_and_small_tail() {
        let config = widened_test_config();
        let mmq = 8192 * 136 * 144;
        let f16 = 127 * 17408 * 2;
        let partials = 127 * 5120 * 4 * 4;
        assert_eq!(mmq, 153 * 1024 * 1024);
        assert_eq!(native_mq4_projection_deficit(&config, 8192, 0, 0, 0),
                   Some(mmq + f16 + partials));
        assert_eq!(native_mq4_projection_deficit(&config, 8192, mmq, f16, partials), Some(0));
        assert_eq!(native_mq4_projection_deficit(&config, 8192, mmq + 1, 0, partials), Some(f16));
        assert_eq!(native_mq4_projection_deficit(&config, usize::MAX, 0, 0, 0), None);
        for rows in [512, 1024, 2048, 4096, 8192] {
            assert_eq!(native_mq4_projection_deficit(&config, rows, 0, 0, 0),
                       Some(rows * 19584 + f16 + partials));
        }
    }

    #[test]
    fn widened_dense_bytes_match_ledger() {
        // Plan §4.1 steady-state envelope: 725,388 B/row + 256 fixed, FP8
        // 17,956 B/row, Q16 fixed 6,291,456 B for 24×256×2×512.
        let config = widened_test_config();
        assert!(widened_dense_shape_admitted(&config));
        assert_eq!(fp8_row_bytes_wide(&config), Some(17_956));
        assert_eq!(
            dense_prefill_allocation_bytes(&config, 1),
            Some(725_388 + 256)
        );
        assert_eq!(
            dense_prefill_allocation_bytes(&config, 512),
            Some(371_398_912)
        );
        assert_eq!(
            dense_prefill_allocation_bytes(&config, 1024),
            Some(742_797_568)
        );
        assert_eq!(
            dense_prefill_allocation_bytes(&config, 2048),
            Some(1_485_594_880)
        );
        assert_eq!(
            dense_prefill_allocation_bytes(&config, 4096),
            Some(2_971_189_504)
        );
        assert_eq!(
            dense_prefill_allocation_bytes(&config, 8192),
            Some(5_942_378_752)
        );
        // Deltas vs 512: dense PBS delta plus the FP8-slot growth at Kmax
        // sum to the ledger's incremental VRAM cost (2,664,144,896 B).
        let base = dense_prefill_allocation_bytes(&config, 512).unwrap();
        let dense_delta = dense_prefill_allocation_bytes(&config, 4096).unwrap() - base;
        assert_eq!(dense_delta, 2_599_790_592);
        let fp8_delta = (4096 - 512) * fp8_row_bytes_wide(&config).unwrap();
        assert_eq!(fp8_delta, 64_354_304);
        assert_eq!(dense_delta + fp8_delta, 2_664_144_896);
        // MoE configs are out of scope for the byte ledger.
        let mut moe = widened_test_config();
        moe.num_experts = 256;
        assert_eq!(dense_prefill_allocation_bytes(&moe, 512), None);
        // Wrong hidden_dim breaks the allocation envelope.
        let mut narrow = widened_test_config();
        narrow.hidden_dim = 3584;
        assert!(!widened_dense_shape_admitted(&narrow));
    }

    #[test]
    fn fa_pair_merge_decision() {
        // Only two complete 512-row chunks merge, on exact gfx1151, eager.
        assert!(fa_pair_merge_admitted(512, 512, "gfx1151", false, false));
        // Partial chunks never merge (tails keep today's dispatch); a 1024
        // chunk never merges either (the merged launch is exactly one pair).
        for partial in [2, 64, 255, 256, 384, 511, 513, 1024] {
            assert!(!fa_pair_merge_admitted(
                512, partial, "gfx1151", false, false
            ));
            assert!(!fa_pair_merge_admitted(
                partial, 512, "gfx1151", false, false
            ));
        }
        assert!(!fa_pair_merge_admitted(1024, 1024, "gfx1151", false, false));
        // Every other arch is byte-identical: no pairs anywhere.
        for arch in [
            "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1200", "gfx1201", "gfx942",
        ] {
            assert!(!fa_pair_merge_admitted(512, 512, arch, false, false));
        }
        // Capture/replay change the launch count: the merged path is
        // eager-only.
        assert!(!fa_pair_merge_admitted(512, 512, "gfx1151", true, false));
        assert!(!fa_pair_merge_admitted(512, 512, "gfx1151", false, true));
        assert!(!fa_pair_merge_admitted(512, 512, "gfx1151", true, true));
        // Geometry consts stay consistent with the merged launch.
        assert_eq!(FA_PAIR_ROWS, 512);
        assert_eq!(FA_PAIR_BATCH, 2 * FA_PAIR_ROWS);
    }

    #[test]
    fn owned_prefill_scratch_preserves_tree_verify_tape() {
        assert_eq!(owned_prefill_scratch_plan(22, 256, true), (22, true));
        assert_eq!(owned_prefill_scratch_plan(64, 64, true), (64, true));
        assert_eq!(owned_prefill_scratch_plan(22, 384, true), (22, true));
    }

    #[test]
    fn moe_grouped_m_total_max_is_tile_aligned() {
        let small_verify = moe_grouped_m_total_max(3, 8, 256);
        assert_eq!(small_verify % MOE_GROUPED_BLOCK_M, 0);
        assert_eq!(small_verify, 3872);

        let prompt_prefill = moe_grouped_m_total_max(27, 8, 256);
        assert_eq!(prompt_prefill % MOE_GROUPED_BLOCK_M, 0);
        assert_eq!(prompt_prefill, 4064);

        let full_chunk = moe_grouped_m_total_max(256, 8, 256);
        assert_eq!(full_chunk, 5888);
    }

    #[test]
    fn moe_grouped_m_total_bound_is_tight_for_small_batches() {
        let small_verify = moe_grouped_m_total_bound(24, 256);
        assert_eq!(small_verify % MOE_GROUPED_BLOCK_M, 0);
        assert_eq!(small_verify, 384);

        let prompt_prefill = moe_grouped_m_total_bound(216, 256);
        assert_eq!(prompt_prefill % MOE_GROUPED_BLOCK_M, 0);
        assert_eq!(prompt_prefill, 3456);

        let full_chunk = moe_grouped_m_total_bound(2048, 256);
        assert_eq!(full_chunk, 5888);
    }
    /// gfx1201 slice-1 gate: fused silu+quant producer vs the unfused
    /// silu → f32 → standalone-quantizer chain on REAL layer-0 down-proj
    /// inputs. Runs one layer-0-only prefill chunk (N=128) on the mq4-xt
    /// fixture, snapshots the live gate/up activations, replays both paths
    /// on device, and demands byte-identical `block_i4_128` streams.
    ///
    /// Requires a real HIP GPU + the fixture
    /// `$HIPFIRE_MODELS_DIR/qwen3.8-27b.mq4-xt` (default dir
    /// `/home/kaden/.hipfire/models`). Ignored by default; run under the GPU
    /// flock with `--test-threads=1`:
    /// `cargo test -p hipfire-arch-qwen35 --lib -- --ignored
    /// gfx12_silu_quant_fused_oracle_matches_standalone --test-threads=1 --nocapture`
    #[test]
    #[ignore = "requires real HIP GPU + qwen3.8-27b.mq4-xt fixture"]
    fn gfx12_silu_quant_fused_oracle_matches_standalone() {
        use crate::qwen35::batch::PrefillBatchScratch;
        use crate::qwen35::forward::Qwen35Scratch;
        use crate::qwen35::load::{HfqSource, Layout};
        use crate::qwen35::{
            config_from_hfq, load_weights, DeltaNetState, LayerWeights, Qwen35Config,
        };
        use hip_bridge::DeviceBuffer;
        use hipfire_runtime::hfq::HfqFile;
        use hipfire_runtime::llama::{self, KvCache};
        use rdna_compute::Gpu;
        use std::time::Duration;
        let mut gpu = match Gpu::init() {
            Ok(gpu) => gpu,
            Err(_) => {
                eprintln!("skip: no GPU");
                return;
            }
        };
        // Fixture path from the bootstrap-exempt HIPFIRE_MODELS_DIR only;
        // production HIPFIRE_* reads must stay config-owned (check-env-docs).
        let models_dir = std::env::var("HIPFIRE_MODELS_DIR")
            .unwrap_or_else(|_| "/home/kaden/.hipfire/models".to_string());
        let model = format!("{models_dir}/qwen3.8-27b.mq4-xt");
        if !std::path::Path::new(&model).exists() {
            eprintln!("skip: fixture {model} missing");
            return;
        }
        let mut hfq = HfqFile::open(std::path::Path::new(&model)).expect("open model");
        let config: Qwen35Config = config_from_hfq(&hfq).expect("read config");
        assert_eq!(gpu.arch, "gfx1201", "oracle is gfx1201-only");
        let weights = {
            let mut src = HfqSource::new(&mut hfq, &config);
            let layout = Layout::single(config.n_layers);
            load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
        }
        .expect("load weights");
        let w_down = match &weights.layers[0] {
            LayerWeights::DeltaNet(l) => &l.w_down,
            LayerWeights::FullAttn(l) => &l.w_down,
            _ => panic!("oracle needs a dense layer 0 (DeltaNet/FullAttn)"),
        };
        assert_eq!(
            w_down.gpu_dtype,
            rdna_compute::DType::MQ4G256V2,
            "oracle needs uniform MQ4G256V2 w_down"
        );
        let k = w_down.k;
        assert!(k > 0 && k % 256 == 0, "w_down.k={k} must be 256-divisible");

        // One layer-0-only chunk of N=128 real tokens through the real stack.
        const N: usize = 128;
        let tokens: Vec<u32> = (0..N as u32).collect();
        let mut kv_cache = KvCache::new_gpu_q8(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            N + 32,
        )
        .expect("kv cache");
        let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn state");
        let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).expect("scratch");
        // Explicit PBS: the widened route would otherwise use an internal
        // owned scratch that is dropped after the call.
        let pbs = PrefillBatchScratch::new_opt(&mut gpu, &config, N, false).expect("pbs");
        forward_prefill_batch_with_pbs(
            &mut gpu,
            &weights,
            &config,
            &tokens,
            0,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
            None,
            None,
            None,
            None,
            Some(&pbs),
            None,
            Some(1),
        )
        .expect("layer-0 prefill");
        // Snapshot the REAL layer-0 gate/up rows (first N*K f32 each).
        let gate_all = gpu.download_f32(&pbs.gate_ffn_batch).expect("dl gate");
        let up_all = gpu.download_f32(&pbs.up_batch).expect("dl up");
        assert!(gate_all.len() >= N * k && up_all.len() >= N * k);
        let gate_host = gate_all[..N * k].to_vec();
        let up_host = up_all[..N * k].to_vec();

        // Unfused chain on device: silu+rotate → f32 → standalone quantizer.
        let gate_t = gpu.upload_f32(&gate_host, &[N * k]).expect("up gate");
        let up_t = gpu.upload_f32(&up_host, &[N * k]).expect("up up");
        let x_rot = gpu
            .alloc_tensor(&[N * k], rdna_compute::DType::F32)
            .expect("alloc x_rot");
        llama::fused_silu_mul_rotate_mq_batched_for(&mut gpu, w_down, &gate_t, &up_t, &x_rot, k, N)
            .expect("unfused silu");
        gpu.sync_with_deadline(Duration::from_secs(60))
            .expect("sync silu");
        let xq_unfused = gpu
            .ensure_int4_mmq_x(&x_rot, N, k)
            .expect("standalone quant");
        gpu.sync_with_deadline(Duration::from_secs(60))
            .expect("sync quant");
        let nbytes = (k / 128) * N * 72;
        let mut bytes_unfused = vec![0u8; nbytes];
        let view_unfused = unsafe { DeviceBuffer::from_raw(xq_unfused, nbytes) };
        gpu.hip
            .memcpy_dtoh(&mut bytes_unfused, &view_unfused)
            .expect("dl unfused blocks");
        std::mem::forget(view_unfused);

        // Fused producer on the same inputs (no f32 store).
        let res = gpu.reserve_int4_mmq(k, N).expect("reserve");
        let prep = gpu
            .fused_silu_mul_rotate_mq_i4_gfx12_batched(
                &gate_t,
                &up_t,
                w_down.awq_scale.as_ref(),
                None,
                res,
                k,
                N,
            )
            .expect("fused silu+quant");
        gpu.sync_with_deadline(Duration::from_secs(60))
            .expect("sync fused");
        let ptr_fused = gpu.int4_mmq_prepared_ptr(&prep, k, N).expect("prep ptr");
        let mut bytes_fused = vec![0u8; nbytes];
        let view_fused = unsafe { DeviceBuffer::from_raw(ptr_fused, nbytes) };
        gpu.hip
            .memcpy_dtoh(&mut bytes_fused, &view_fused)
            .expect("dl fused blocks");
        std::mem::forget(view_fused);

        assert_eq!(
            bytes_unfused.len(),
            bytes_fused.len(),
            "block stream length mismatch"
        );
        let mut first_diff = None;
        for (i, (a, b)) in bytes_unfused.iter().zip(bytes_fused.iter()).enumerate() {
            if a != b && first_diff.is_none() {
                first_diff = Some(i);
                break;
            }
        }
        assert!(
            first_diff.is_none(),
            "fused blocks differ at byte {:?} of {nbytes} (awq={})",
            first_diff,
            w_down.awq_scale.is_some()
        );
        eprintln!(
            "oracle PASS: {nbytes} block bytes identical (K={k} N={N} awq={})",
            w_down.awq_scale.is_some()
        );
    }

    /// Token-order `block_i4_128` records of a slab-layout IU4 sidecar
    /// (`HIPFIRE_A4_SLAB` producer twins: per K128 block of `n` tokens the
    /// planes `[d 4n][s 4n][qs 0..31: 32n][qs 32..63: 32n]`).
    fn slab_to_token_i4(slab: &[u8], n: usize) -> Vec<u8> {
        let mut out = vec![0u8; slab.len()];
        for (src, dst) in slab.chunks_exact(n * 72).zip(out.chunks_exact_mut(n * 72)) {
            for t in 0..n {
                let r = &mut dst[t * 72..t * 72 + 72];
                r[..4].copy_from_slice(&src[t * 4..t * 4 + 4]);
                r[4..8].copy_from_slice(&src[4 * n + t * 4..4 * n + t * 4 + 4]);
                r[8..40].copy_from_slice(&src[8 * n + t * 32..8 * n + t * 32 + 32]);
                r[40..].copy_from_slice(&src[40 * n + t * 32..40 * n + t * 32 + 32]);
            }
        }
        out
    }

    /// gfx1201 slices-2 gate: fused rmsnorm+quant producer vs the unfused
    /// rmsnorm+rotate → f32 → standalone-quantizer chain on REAL layer-0
    /// qkvza/gate_up inputs. Runs one layer-0-only prefill chunk (N=128) on
    /// the mq4-xt fixture, snapshots the live residual rows, replays both
    /// paths on device, and demands byte-identical f32 outputs and
    /// `block_i4_128` streams.
    ///
    /// Requires a real HIP GPU + the fixture
    /// `$HIPFIRE_MODELS_DIR/qwen3.8-27b.mq4-xt` (default dir
    /// `/home/kaden/.hipfire/models`). Ignored by default; run under the GPU
    /// flock with `--test-threads=1`:
    /// `cargo test -p hipfire-arch-qwen35 --lib -- --ignored
    /// gfx12_rmsnorm_quant_fused_oracle_matches_standalone --test-threads=1 --nocapture`
    #[test]
    #[ignore = "requires real HIP GPU + qwen3.8-27b.mq4-xt fixture"]
    fn gfx12_rmsnorm_quant_fused_oracle_matches_standalone() {
        use crate::qwen35::batch::PrefillBatchScratch;
        use crate::qwen35::forward::Qwen35Scratch;
        use crate::qwen35::load::{HfqSource, Layout};
        use crate::qwen35::{
            config_from_hfq, load_weights, DeltaNetState, LayerWeights, Qwen35Config,
        };
        use hip_bridge::DeviceBuffer;
        use hipfire_runtime::hfq::HfqFile;
        use hipfire_runtime::llama::{self, KvCache};
        use rdna_compute::Gpu;
        use std::time::Duration;
        let mut gpu = match Gpu::init() {
            Ok(gpu) => gpu,
            Err(_) => {
                eprintln!("skip: no GPU");
                return;
            }
        };
        // Fixture path from the bootstrap-exempt HIPFIRE_MODELS_DIR only;
        // production HIPFIRE_* reads must stay config-owned (check-env-docs).
        let models_dir = std::env::var("HIPFIRE_MODELS_DIR")
            .unwrap_or_else(|_| "/home/kaden/.hipfire/models".to_string());
        let model = format!("{models_dir}/qwen3.8-27b.mq4-xt");
        if !std::path::Path::new(&model).exists() {
            eprintln!("skip: fixture {model} missing");
            return;
        }
        let mut hfq = HfqFile::open(std::path::Path::new(&model)).expect("open model");
        let config: Qwen35Config = config_from_hfq(&hfq).expect("read config");
        assert_eq!(gpu.arch, "gfx1201", "oracle is gfx1201-only");
        let weights = {
            let mut src = HfqSource::new(&mut hfq, &config);
            let layout = Layout::single(config.n_layers);
            load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
        }
        .expect("load weights");
        let (norm_weight, next_linear) = match &weights.layers[0] {
            LayerWeights::DeltaNet(l) => (&l.attn_norm, &l.wqkv),
            _ => panic!("oracle needs a DeltaNet layer 0"),
        };
        assert_eq!(
            next_linear.gpu_dtype,
            rdna_compute::DType::MQ4G256V2,
            "oracle needs uniform MQ4G256V2 wqkv"
        );
        let k = next_linear.k;
        assert!(k > 0 && k % 256 == 0, "wqkv.k={k} must be 256-divisible");

        // One layer-0-only chunk of N=128 real tokens through the real stack.
        const N: usize = 128;
        let tokens: Vec<u32> = (0..N as u32).collect();
        let mut kv_cache = KvCache::new_gpu_q8(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            N + 32,
        )
        .expect("kv cache");
        let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn state");
        let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).expect("scratch");
        // Explicit PBS: the widened route would otherwise use an internal
        // owned scratch that is dropped after the call.
        let pbs = PrefillBatchScratch::new_opt(&mut gpu, &config, N, false).expect("pbs");
        forward_prefill_batch_with_pbs(
            &mut gpu,
            &weights,
            &config,
            &tokens,
            0,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
            None,
            None,
            None,
            None,
            Some(&pbs),
            None,
            Some(1),
        )
        .expect("layer-0 prefill");
        // Snapshot the REAL layer-0 residual rows (first N*K f32).
        let x_all = gpu.download_f32(&pbs.x_batch).expect("dl x");
        assert!(x_all.len() >= N * k);
        let x_host = x_all[..N * k].to_vec();

        // Unfused chain on device: rmsnorm+rotate → f32 → standalone quantizer.
        let x_t = gpu.upload_f32(&x_host, &[N * k]).expect("up x");
        let x_rot = gpu
            .alloc_tensor(&[N * k], rdna_compute::DType::F32)
            .expect("alloc x_rot");
        llama::fused_rmsnorm_rotate_mq_batched_for(
            &mut gpu,
            &x_t,
            norm_weight,
            next_linear,
            &x_rot,
            k,
            config.norm_eps,
            N,
        )
        .expect("unfused rmsnorm+rotate");
        gpu.sync_with_deadline(Duration::from_secs(60))
            .expect("sync rmsnorm");
        let f32_unfused = gpu.download_f32(&x_rot).expect("dl unfused f32");
        let xq_unfused = gpu
            .ensure_int4_mmq_x(&x_rot, N, k)
            .expect("standalone quant");
        gpu.sync_with_deadline(Duration::from_secs(60))
            .expect("sync quant");
        let nbytes = (k / 128) * N * 72;
        let mut bytes_unfused = vec![0u8; nbytes];
        let view_unfused = unsafe { DeviceBuffer::from_raw(xq_unfused, nbytes) };
        gpu.hip
            .memcpy_dtoh(&mut bytes_unfused, &view_unfused)
            .expect("dl unfused blocks");
        std::mem::forget(view_unfused);

        // Fused producer on the same inputs (with f32 store for comparison).
        let x_rot_fused = gpu
            .alloc_tensor(&[N * k], rdna_compute::DType::F32)
            .expect("alloc x_rot fused");
        let res = gpu.reserve_int4_mmq(k, N).expect("reserve");
        let prep = gpu
            .fused_rmsnorm_rotate_mq_i4_gfx12_batched(
                &x_t,
                norm_weight,
                next_linear.awq_scale.as_ref(),
                Some(&x_rot_fused),
                res,
                k,
                config.norm_eps,
                N,
            )
            .expect("fused rmsnorm+quant");
        gpu.sync_with_deadline(Duration::from_secs(60))
            .expect("sync fused");
        let f32_fused = gpu.download_f32(&x_rot_fused).expect("dl fused f32");
        let ptr_fused = gpu.int4_mmq_prepared_ptr(&prep, k, N).expect("prep ptr");
        let mut bytes_fused = vec![0u8; nbytes];
        let view_fused = unsafe { DeviceBuffer::from_raw(ptr_fused, nbytes) };
        gpu.hip
            .memcpy_dtoh(&mut bytes_fused, &view_fused)
            .expect("dl fused blocks");
        std::mem::forget(view_fused);
        if gpu.scratch.int4_mmq_slab_at(ptr_fused) {
            bytes_fused = slab_to_token_i4(&bytes_fused, N);
        }

        assert_eq!(f32_unfused.len(), f32_fused.len(), "f32 length mismatch");
        let mut first_f32_diff = None;
        for (i, (a, b)) in f32_unfused.iter().zip(f32_fused.iter()).enumerate() {
            if a.to_bits() != b.to_bits() && first_f32_diff.is_none() {
                first_f32_diff = Some(i);
                break;
            }
        }
        assert!(
            first_f32_diff.is_none(),
            "fused f32 differs at element {:?} of {} (awq={})",
            first_f32_diff,
            f32_unfused.len(),
            next_linear.awq_scale.is_some()
        );
        assert_eq!(
            bytes_unfused.len(),
            bytes_fused.len(),
            "block stream length mismatch"
        );
        let mut first_diff = None;
        for (i, (a, b)) in bytes_unfused.iter().zip(bytes_fused.iter()).enumerate() {
            if a != b && first_diff.is_none() {
                first_diff = Some(i);
                break;
            }
        }
        assert!(
            first_diff.is_none(),
            "fused blocks differ at byte {:?} of {nbytes} (awq={})",
            first_diff,
            next_linear.awq_scale.is_some()
        );
        eprintln!(
            "oracle PASS: {} f32 + {nbytes} block bytes identical (K={k} N={N} awq={})",
            f32_unfused.len(),
            next_linear.awq_scale.is_some()
        );
    }

    /// gfx1201 slices-3 gate: fused rotate+quant producer vs the unfused
    /// rotate → f32 → standalone-quantizer chain on REAL layer-0 GDN
    /// post-gated-norm rows (the attention out-proj input family, K=6144 in
    /// flight). Runs one layer-0-only prefill chunk (N=128) on the mq4-xt
    /// fixture, snapshots the live gated-norm rows, replays both paths on
    /// device, and demands byte-identical f32 outputs and `block_i4_128`
    /// streams.
    ///
    /// Requires a real HIP GPU + the fixture
    /// `$HIPFIRE_MODELS_DIR/qwen3.8-27b.mq4-xt` (default dir
    /// `/home/kaden/.hipfire/models`). Ignored by default; run under the GPU
    /// flock with `--test-threads=1`:
    /// `cargo test -p hipfire-arch-qwen35 --lib -- --ignored
    /// gfx12_rotate_quant_fused_oracle_matches_standalone --test-threads=1 --nocapture`
    #[test]
    #[ignore = "requires real HIP GPU + qwen3.8-27b.mq4-xt fixture"]
    fn gfx12_rotate_quant_fused_oracle_matches_standalone() {
        use crate::qwen35::batch::PrefillBatchScratch;
        use crate::qwen35::forward::Qwen35Scratch;
        use crate::qwen35::load::{HfqSource, Layout};
        use crate::qwen35::{
            config_from_hfq, load_weights, DeltaNetState, LayerWeights, Qwen35Config,
        };
        use hip_bridge::DeviceBuffer;
        use hipfire_runtime::hfq::HfqFile;
        use hipfire_runtime::llama::{self, KvCache};
        use rdna_compute::Gpu;
        use std::time::Duration;
        let mut gpu = match Gpu::init() {
            Ok(gpu) => gpu,
            Err(_) => {
                eprintln!("skip: no GPU");
                return;
            }
        };
        // Fixture path from the bootstrap-exempt HIPFIRE_MODELS_DIR only;
        // production HIPFIRE_* reads must stay config-owned (check-env-docs).
        let models_dir = std::env::var("HIPFIRE_MODELS_DIR")
            .unwrap_or_else(|_| "/home/kaden/.hipfire/models".to_string());
        let model = format!("{models_dir}/qwen3.8-27b.mq4-xt");
        if !std::path::Path::new(&model).exists() {
            eprintln!("skip: fixture {model} missing");
            return;
        }
        let mut hfq = HfqFile::open(std::path::Path::new(&model)).expect("open model");
        let config: Qwen35Config = config_from_hfq(&hfq).expect("read config");
        assert_eq!(gpu.arch, "gfx1201", "oracle is gfx1201-only");
        let weights = {
            let mut src = HfqSource::new(&mut hfq, &config);
            let layout = Layout::single(config.n_layers);
            load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
        }
        .expect("load weights");
        let wo = match &weights.layers[0] {
            LayerWeights::DeltaNet(l) => &l.wo,
            _ => panic!("oracle needs a DeltaNet layer 0"),
        };
        assert_eq!(
            wo.gpu_dtype,
            rdna_compute::DType::MQ4G256V2,
            "oracle needs uniform MQ4G256V2 wo"
        );
        let k = wo.k;
        assert!(k > 0 && k % 256 == 0, "wo.k={k} must be 256-divisible");

        // One layer-0-only chunk of N=128 real tokens through the real stack.
        const N: usize = 128;
        let tokens: Vec<u32> = (0..N as u32).collect();
        let mut kv_cache = KvCache::new_gpu_q8(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            N + 32,
        )
        .expect("kv cache");
        let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn state");
        let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).expect("scratch");
        // Explicit PBS: the widened route would otherwise use an internal
        // owned scratch that is dropped after the call.
        let pbs = PrefillBatchScratch::new_opt(&mut gpu, &config, N, false).expect("pbs");
        forward_prefill_batch_with_pbs(
            &mut gpu,
            &weights,
            &config,
            &tokens,
            0,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
            None,
            None,
            None,
            None,
            Some(&pbs),
            None,
            Some(1),
        )
        .expect("layer-0 prefill");
        // Snapshot the REAL layer-0 gated-norm rows (first N*K f32). This is
        // the exact buffer the wo rotate consumes on the iu4 route.
        let normed_all = gpu.download_f32(&pbs.dn_normed_batch).expect("dl normed");
        assert!(normed_all.len() >= N * k);
        let normed_host = normed_all[..N * k].to_vec();

        // Unfused chain on device: rotate → f32 → standalone quantizer.
        let x_t = gpu.upload_f32(&normed_host, &[N * k]).expect("up x");
        let x_rot = gpu
            .alloc_tensor(&[N * k], rdna_compute::DType::F32)
            .expect("alloc x_rot");
        llama::rotate_x_mq_batched_for(&mut gpu, wo, &x_t, &x_rot, k, N).expect("unfused rotate");
        gpu.sync_with_deadline(Duration::from_secs(60))
            .expect("sync rotate");
        let f32_unfused = gpu.download_f32(&x_rot).expect("dl unfused f32");
        let xq_unfused = gpu
            .ensure_int4_mmq_x(&x_rot, N, k)
            .expect("standalone quant");
        gpu.sync_with_deadline(Duration::from_secs(60))
            .expect("sync quant");
        let nbytes = (k / 128) * N * 72;
        let mut bytes_unfused = vec![0u8; nbytes];
        let view_unfused = unsafe { DeviceBuffer::from_raw(xq_unfused, nbytes) };
        gpu.hip
            .memcpy_dtoh(&mut bytes_unfused, &view_unfused)
            .expect("dl unfused blocks");
        std::mem::forget(view_unfused);

        // Fused producer on the same inputs (f32 store always written).
        let x_rot_fused = gpu
            .alloc_tensor(&[N * k], rdna_compute::DType::F32)
            .expect("alloc x_rot fused");
        let res = gpu.reserve_int4_mmq(k, N).expect("reserve");
        let prep = gpu
            .rotate_x_mq_i4_gfx12_batched(
                &x_t,
                wo.awq_scale.as_ref(),
                Some(&x_rot_fused),
                res,
                k,
                N,
            )
            .expect("fused rotate+quant");
        gpu.sync_with_deadline(Duration::from_secs(60))
            .expect("sync fused");
        let f32_fused = gpu.download_f32(&x_rot_fused).expect("dl fused f32");
        let ptr_fused = gpu.int4_mmq_prepared_ptr(&prep, k, N).expect("prep ptr");
        let mut bytes_fused = vec![0u8; nbytes];
        let view_fused = unsafe { DeviceBuffer::from_raw(ptr_fused, nbytes) };
        gpu.hip
            .memcpy_dtoh(&mut bytes_fused, &view_fused)
            .expect("dl fused blocks");
        std::mem::forget(view_fused);

        assert_eq!(f32_unfused.len(), f32_fused.len(), "f32 length mismatch");
        let mut first_f32_diff = None;
        for (i, (a, b)) in f32_unfused.iter().zip(f32_fused.iter()).enumerate() {
            if a.to_bits() != b.to_bits() && first_f32_diff.is_none() {
                first_f32_diff = Some(i);
                break;
            }
        }
        assert!(
            first_f32_diff.is_none(),
            "fused f32 differs at element {:?} of {} (awq={})",
            first_f32_diff,
            f32_unfused.len(),
            wo.awq_scale.is_some()
        );
        assert_eq!(
            bytes_unfused.len(),
            bytes_fused.len(),
            "block stream length mismatch"
        );
        let mut first_diff = None;
        for (i, (a, b)) in bytes_unfused.iter().zip(bytes_fused.iter()).enumerate() {
            if a != b && first_diff.is_none() {
                first_diff = Some(i);
                break;
            }
        }
        assert!(
            first_diff.is_none(),
            "fused blocks differ at byte {:?} of {nbytes} (awq={})",
            first_diff,
            wo.awq_scale.is_some()
        );
        eprintln!(
            "oracle PASS: {} f32 + {nbytes} block bytes identical (K={k} N={N} awq={})",
            f32_unfused.len(),
            wo.awq_scale.is_some()
        );
    }

    /// gfx1201 slices-4 gate: fused gated_norm+rotate+quant producer vs the
    /// unfused gated_norm_f32 → rotate → standalone-quantizer chain on REAL
    /// layer-0 GDN attention/gate rows. Runs one layer-0-only prefill chunk
    /// (N=128) on the mq4-xt fixture, snapshots the live GDN outputs,
    /// replays both paths on device, and demands byte-identical f32 outputs
    /// and `block_i4_128` streams.
    ///
    /// Requires a real HIP GPU + the fixture
    /// `$HIPFIRE_MODELS_DIR/qwen3.8-27b.mq4-xt` (default dir
    /// `/home/kaden/.hipfire/models`). Ignored by default; run under the GPU
    /// flock with `--test-threads=1`:
    /// `cargo test -p hipfire-arch-qwen35 --lib -- --ignored
    /// gfx12_gdn_quant_fused_oracle_matches_standalone --test-threads=1 --nocapture`
    #[test]
    #[ignore = "requires real HIP GPU + qwen3.8-27b.mq4-xt fixture"]
    fn gfx12_gdn_quant_fused_oracle_matches_standalone() {
        use crate::qwen35::batch::PrefillBatchScratch;
        use crate::qwen35::forward::Qwen35Scratch;
        use crate::qwen35::load::{HfqSource, Layout};
        use crate::qwen35::{
            config_from_hfq, load_weights, DeltaNetState, LayerWeights, Qwen35Config,
        };
        use hip_bridge::DeviceBuffer;
        use hipfire_runtime::hfq::HfqFile;
        use hipfire_runtime::llama::{self, KvCache};
        use rdna_compute::Gpu;
        use std::time::Duration;
        let mut gpu = match Gpu::init() {
            Ok(gpu) => gpu,
            Err(_) => {
                eprintln!("skip: no GPU");
                return;
            }
        };
        // Fixture path from the bootstrap-exempt HIPFIRE_MODELS_DIR only;
        // production HIPFIRE_* reads must stay config-owned (check-env-docs).
        let models_dir = std::env::var("HIPFIRE_MODELS_DIR")
            .unwrap_or_else(|_| "/home/kaden/.hipfire/models".to_string());
        let model = format!("{models_dir}/qwen3.8-27b.mq4-xt");
        if !std::path::Path::new(&model).exists() {
            eprintln!("skip: fixture {model} missing");
            return;
        }
        let mut hfq = HfqFile::open(std::path::Path::new(&model)).expect("open model");
        let config: Qwen35Config = config_from_hfq(&hfq).expect("read config");
        assert_eq!(gpu.arch, "gfx1201", "oracle is gfx1201-only");
        let weights = {
            let mut src = HfqSource::new(&mut hfq, &config);
            let layout = Layout::single(config.n_layers);
            load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
        }
        .expect("load weights");
        let (norm_weight, wo) = match &weights.layers[0] {
            LayerWeights::DeltaNet(l) => (&l.norm_weight, &l.wo),
            _ => panic!("oracle needs a DeltaNet layer 0"),
        };
        assert_eq!(
            wo.gpu_dtype,
            rdna_compute::DType::MQ4G256V2,
            "oracle needs uniform MQ4G256V2 wo"
        );
        let k = wo.k;
        let head_dim = config.linear_value_head_dim;
        assert_eq!(head_dim, 128, "oracle needs head_dim == 128");
        assert!(
            k > 0 && k % 256 == 0 && k % head_dim == 0,
            "wo.k={k} must be 256-divisible with head_dim=128"
        );
        let n_heads = k / head_dim;

        // One layer-0-only chunk of N=128 real tokens through the real stack.
        const N: usize = 128;
        let tokens: Vec<u32> = (0..N as u32).collect();
        let mut kv_cache = KvCache::new_gpu_q8(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            N + 32,
        )
        .expect("kv cache");
        let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn state");
        let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).expect("scratch");
        // Explicit PBS: the widened route would otherwise use an internal
        // owned scratch that is dropped after the call.
        let pbs = PrefillBatchScratch::new_opt(&mut gpu, &config, N, false).expect("pbs");
        forward_prefill_batch_with_pbs(
            &mut gpu,
            &weights,
            &config,
            &tokens,
            0,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
            None,
            None,
            None,
            None,
            Some(&pbs),
            None,
            Some(1),
        )
        .expect("layer-0 prefill");
        // Snapshot the REAL layer-0 GDN attention/gate rows (first N*K f32
        // each). These are the exact buffers the wo chain consumes.
        let attn_all = gpu.download_f32(&pbs.dn_attn_out_batch).expect("dl attn");
        let z_all = gpu.download_f32(&pbs.dn_z_batch).expect("dl z");
        assert!(attn_all.len() >= N * k && z_all.len() >= N * k);
        let attn_host = attn_all[..N * k].to_vec();
        let z_host = z_all[..N * k].to_vec();

        // Unfused chain on device: gated_norm → rotate → standalone quantizer.
        let x_t = gpu.upload_f32(&attn_host, &[N * k]).expect("up x");
        let z_t = gpu.upload_f32(&z_host, &[N * k]).expect("up z");
        let normed = gpu
            .alloc_tensor(&[N * k], rdna_compute::DType::F32)
            .expect("alloc normed");
        gpu.gated_norm_f32_batched(
            &x_t,
            &z_t,
            norm_weight,
            &normed,
            n_heads,
            head_dim,
            config.norm_eps,
            N,
        )
        .expect("unfused gated_norm");
        let x_rot = gpu
            .alloc_tensor(&[N * k], rdna_compute::DType::F32)
            .expect("alloc x_rot");
        llama::rotate_x_mq_batched_for(&mut gpu, wo, &normed, &x_rot, k, N)
            .expect("unfused rotate");
        gpu.sync_with_deadline(Duration::from_secs(60))
            .expect("sync chain");
        let f32_unfused = gpu.download_f32(&x_rot).expect("dl unfused f32");
        let xq_unfused = gpu
            .ensure_int4_mmq_x(&x_rot, N, k)
            .expect("standalone quant");
        gpu.sync_with_deadline(Duration::from_secs(60))
            .expect("sync quant");
        let nbytes = (k / 128) * N * 72;
        let mut bytes_unfused = vec![0u8; nbytes];
        let view_unfused = unsafe { DeviceBuffer::from_raw(xq_unfused, nbytes) };
        gpu.hip
            .memcpy_dtoh(&mut bytes_unfused, &view_unfused)
            .expect("dl unfused blocks");
        std::mem::forget(view_unfused);

        // Fused producer on the same inputs (f32 store always written).
        let x_rot_fused = gpu
            .alloc_tensor(&[N * k], rdna_compute::DType::F32)
            .expect("alloc x_rot fused");
        let res = gpu.reserve_int4_mmq(k, N).expect("reserve");
        let prep = gpu
            .gated_norm_rotate_mq_i4_gfx12_batched(
                &x_t,
                rdna_compute::norm::GdnScanOut::F32,
                &z_t,
                norm_weight,
                wo.awq_scale.as_ref(),
                Some(&x_rot_fused),
                res,
                n_heads,
                head_dim,
                config.norm_eps,
                k,
                N,
            )
            .expect("fused gdn+quant");
        gpu.sync_with_deadline(Duration::from_secs(60))
            .expect("sync fused");
        let f32_fused = gpu.download_f32(&x_rot_fused).expect("dl fused f32");
        let ptr_fused = gpu.int4_mmq_prepared_ptr(&prep, k, N).expect("prep ptr");
        let mut bytes_fused = vec![0u8; nbytes];
        let view_fused = unsafe { DeviceBuffer::from_raw(ptr_fused, nbytes) };
        gpu.hip
            .memcpy_dtoh(&mut bytes_fused, &view_fused)
            .expect("dl fused blocks");
        std::mem::forget(view_fused);
        if gpu.scratch.int4_mmq_slab_at(ptr_fused) {
            bytes_fused = slab_to_token_i4(&bytes_fused, N);
        }

        assert_eq!(f32_unfused.len(), f32_fused.len(), "f32 length mismatch");
        let mut first_f32_diff = None;
        for (i, (a, b)) in f32_unfused.iter().zip(f32_fused.iter()).enumerate() {
            if a.to_bits() != b.to_bits() && first_f32_diff.is_none() {
                first_f32_diff = Some(i);
                break;
            }
        }
        assert!(
            first_f32_diff.is_none(),
            "fused f32 differs at element {:?} of {} (awq={})",
            first_f32_diff,
            f32_unfused.len(),
            wo.awq_scale.is_some()
        );
        assert_eq!(
            bytes_unfused.len(),
            bytes_fused.len(),
            "block stream length mismatch"
        );
        let mut first_diff = None;
        for (i, (a, b)) in bytes_unfused.iter().zip(bytes_fused.iter()).enumerate() {
            if a != b && first_diff.is_none() {
                first_diff = Some(i);
                break;
            }
        }
        assert!(
            first_diff.is_none(),
            "fused blocks differ at byte {:?} of {nbytes} (awq={})",
            first_diff,
            wo.awq_scale.is_some()
        );
        eprintln!(
            "oracle PASS: {} f32 + {nbytes} block bytes identical (K={k} N={N} heads={n_heads} awq={})",
            f32_unfused.len(),
            wo.awq_scale.is_some()
        );
    }
    /// gfx11 slices-4 gate: fused gated_norm+rotate+quant producer vs the
    /// unfused gated_norm_f32 → rotate → standalone-quantizer chain on REAL
    /// layer-0 GDN attention/gate rows, under the `_gfx11` entry symbols.
    /// Port of `gfx12_gdn_quant_fused_oracle_matches_standalone`.
    ///
    /// Requires a real HIP GPU (gfx1100/gfx1151) + the fixture
    /// `$HIPFIRE_MODELS_DIR/qwen3.8-27b.mq4-xt` (default dir
    /// `/home/kaden/.hipfire/models`). Ignored by default; run under the GPU
    /// flock with `--test-threads=1`:
    /// `cargo test -p hipfire-arch-qwen35 --lib -- --ignored
    /// gfx11_gdn_quant_fused_oracle_matches_standalone --test-threads=1 --nocapture`
    #[test]
    #[ignore = "requires real HIP GPU + qwen3.8-27b.mq4-xt fixture"]
    fn gfx11_gdn_quant_fused_oracle_matches_standalone() {
        use crate::qwen35::batch::PrefillBatchScratch;
        use crate::qwen35::forward::Qwen35Scratch;
        use crate::qwen35::load::{HfqSource, Layout};
        use crate::qwen35::{
            config_from_hfq, load_weights, DeltaNetState, LayerWeights, Qwen35Config,
        };
        use hip_bridge::DeviceBuffer;
        use hipfire_runtime::hfq::HfqFile;
        use hipfire_runtime::llama::{self, KvCache};
        use rdna_compute::Gpu;
        use std::time::Duration;
        let mut gpu = match Gpu::init() {
            Ok(gpu) => gpu,
            Err(_) => {
                eprintln!("skip: no GPU");
                return;
            }
        };
        let models_dir = std::env::var("HIPFIRE_MODELS_DIR")
            .unwrap_or_else(|_| "/home/kaden/.hipfire/models".to_string());
        let model = format!("{models_dir}/qwen3.8-27b.mq4-xt");
        if !std::path::Path::new(&model).exists() {
            eprintln!("skip: fixture {model} missing");
            return;
        }
        let mut hfq = HfqFile::open(std::path::Path::new(&model)).expect("open model");
        let config: Qwen35Config = config_from_hfq(&hfq).expect("read config");
        assert!(
            matches!(gpu.arch.as_str(), "gfx1100" | "gfx1151"),
            "oracle is gfx11-only (arch={})",
            gpu.arch
        );
        let weights = {
            let mut src = HfqSource::new(&mut hfq, &config);
            let layout = Layout::single(config.n_layers);
            load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
        }
        .expect("load weights");
        let (norm_weight, wo) = match &weights.layers[0] {
            LayerWeights::DeltaNet(l) => (&l.norm_weight, &l.wo),
            _ => panic!("oracle needs a DeltaNet layer 0"),
        };
        assert_eq!(
            wo.gpu_dtype,
            rdna_compute::DType::MQ4G256V2,
            "oracle needs uniform MQ4G256V2 wo"
        );
        let k = wo.k;
        let head_dim = config.linear_value_head_dim;
        assert_eq!(head_dim, 128, "oracle needs head_dim == 128");
        assert!(
            k > 0 && k % 256 == 0 && k % head_dim == 0,
            "wo.k={k} must be 256-divisible with head_dim=128"
        );
        let n_heads = k / head_dim;

        // One layer-0-only chunk of N=128 real tokens through the real stack.
        const N: usize = 128;
        let tokens: Vec<u32> = (0..N as u32).collect();
        let mut kv_cache = KvCache::new_gpu_q8(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            N + 32,
        )
        .expect("kv cache");
        let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn state");
        let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).expect("scratch");
        let pbs = PrefillBatchScratch::new_opt(&mut gpu, &config, N, false).expect("pbs");
        forward_prefill_batch_with_pbs(
            &mut gpu,
            &weights,
            &config,
            &tokens,
            0,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
            None,
            None,
            None,
            None,
            Some(&pbs),
            None,
            Some(1),
        )
        .expect("layer-0 prefill");
        // Snapshot the REAL layer-0 GDN attention/gate rows (first N*K f32
        // each). These are the exact buffers the wo chain consumes.
        let attn_all = gpu.download_f32(&pbs.dn_attn_out_batch).expect("dl attn");
        let z_all = gpu.download_f32(&pbs.dn_z_batch).expect("dl z");
        assert!(attn_all.len() >= N * k && z_all.len() >= N * k);
        let attn_host = attn_all[..N * k].to_vec();
        let z_host = z_all[..N * k].to_vec();

        // Unfused chain on device: gated_norm → rotate → standalone quantizer.
        let x_t = gpu.upload_f32(&attn_host, &[N * k]).expect("up x");
        let z_t = gpu.upload_f32(&z_host, &[N * k]).expect("up z");
        let normed = gpu
            .alloc_tensor(&[N * k], rdna_compute::DType::F32)
            .expect("alloc normed");
        gpu.gated_norm_f32_batched(
            &x_t,
            &z_t,
            norm_weight,
            &normed,
            n_heads,
            head_dim,
            config.norm_eps,
            N,
        )
        .expect("unfused gated_norm");
        let x_rot = gpu
            .alloc_tensor(&[N * k], rdna_compute::DType::F32)
            .expect("alloc x_rot");
        llama::rotate_x_mq_batched_for(&mut gpu, wo, &normed, &x_rot, k, N)
            .expect("unfused rotate");
        gpu.sync_with_deadline(Duration::from_secs(60))
            .expect("sync chain");
        let f32_unfused = gpu.download_f32(&x_rot).expect("dl unfused f32");
        let xq_unfused = gpu
            .ensure_int4_mmq_x(&x_rot, N, k)
            .expect("standalone quant");
        gpu.sync_with_deadline(Duration::from_secs(60))
            .expect("sync quant");
        let nbytes = (k / 128) * N * 72;
        let mut bytes_unfused = vec![0u8; nbytes];
        let view_unfused = unsafe { DeviceBuffer::from_raw(xq_unfused, nbytes) };
        gpu.hip
            .memcpy_dtoh(&mut bytes_unfused, &view_unfused)
            .expect("dl unfused blocks");
        std::mem::forget(view_unfused);

        // Fused `_gfx11` producer on the same inputs (f32 store written for
        // comparison; production passes None).
        let x_rot_fused = gpu
            .alloc_tensor(&[N * k], rdna_compute::DType::F32)
            .expect("alloc x_rot fused");
        let res = gpu.reserve_int4_mmq(k, N).expect("reserve");
        let prep = gpu
            .gated_norm_rotate_mq_i4_gfx11_batched(
                &x_t,
                &z_t,
                norm_weight,
                wo.awq_scale.as_ref(),
                Some(&x_rot_fused),
                res,
                n_heads,
                head_dim,
                config.norm_eps,
                k,
                N,
            )
            .expect("fused gdn+quant");
        gpu.sync_with_deadline(Duration::from_secs(60))
            .expect("sync fused");
        let f32_fused = gpu.download_f32(&x_rot_fused).expect("dl fused f32");
        let ptr_fused = gpu.int4_mmq_prepared_ptr(&prep, k, N).expect("prep ptr");
        let mut bytes_fused = vec![0u8; nbytes];
        let view_fused = unsafe { DeviceBuffer::from_raw(ptr_fused, nbytes) };
        gpu.hip
            .memcpy_dtoh(&mut bytes_fused, &view_fused)
            .expect("dl fused blocks");
        std::mem::forget(view_fused);

        assert_eq!(f32_unfused.len(), f32_fused.len(), "f32 length mismatch");
        let mut first_f32_diff = None;
        for (i, (a, b)) in f32_unfused.iter().zip(f32_fused.iter()).enumerate() {
            if a.to_bits() != b.to_bits() && first_f32_diff.is_none() {
                first_f32_diff = Some(i);
                break;
            }
        }
        assert!(
            first_f32_diff.is_none(),
            "fused f32 differs at element {:?} of {} (awq={})",
            first_f32_diff,
            f32_unfused.len(),
            wo.awq_scale.is_some()
        );
        assert_eq!(
            bytes_unfused.len(),
            bytes_fused.len(),
            "block stream length mismatch"
        );
        let mut first_diff = None;
        for (i, (a, b)) in bytes_unfused.iter().zip(bytes_fused.iter()).enumerate() {
            if a != b && first_diff.is_none() {
                first_diff = Some(i);
                break;
            }
        }
        assert!(
            first_diff.is_none(),
            "fused blocks differ at byte {:?} of {nbytes} (awq={})",
            first_diff,
            wo.awq_scale.is_some()
        );
        eprintln!(
            "oracle PASS: {} f32 + {nbytes} block bytes identical (K={k} N={N} heads={n_heads} awq={})",
            f32_unfused.len(),
            wo.awq_scale.is_some()
        );
    }

    /// gfx11 FA out-proj gate: fused sigmoid+rotate+quant producer vs the
    /// unfused sigmoid_mul_f32 → rotate → standalone-quantizer chain on REAL
    /// FA attention/gate rows, under the `_gfx11` entry symbol. The fused
    /// entry is AWQ-only like its `_gfx12` twin; the test vacates (skip)
    /// when the first FA wo carries no AWQ sidecar.
    ///
    /// Requires a real HIP GPU (gfx1100/gfx1151) + the fixture
    /// `$HIPFIRE_MODELS_DIR/qwen3.8-27b.mq4-xt` (default dir
    /// `/home/kaden/.hipfire/models`). Ignored by default; run under the GPU
    /// flock with `--test-threads=1`:
    /// `cargo test -p hipfire-arch-qwen35 --lib -- --ignored
    /// gfx11_sigmoid_rotate_quant_fused_oracle_matches_standalone --test-threads=1 --nocapture`
    #[test]
    #[ignore = "requires real HIP GPU + qwen3.8-27b.mq4-xt fixture"]
    fn gfx11_sigmoid_rotate_quant_fused_oracle_matches_standalone() {
        use crate::qwen35::batch::PrefillBatchScratch;
        use crate::qwen35::config::LayerType;
        use crate::qwen35::forward::Qwen35Scratch;
        use crate::qwen35::load::{HfqSource, Layout};
        use crate::qwen35::{
            config_from_hfq, load_weights, DeltaNetState, LayerWeights, Qwen35Config,
        };
        use hip_bridge::DeviceBuffer;
        use hipfire_runtime::hfq::HfqFile;
        use hipfire_runtime::llama::{self, KvCache};
        use rdna_compute::Gpu;
        use std::time::Duration;
        let mut gpu = match Gpu::init() {
            Ok(gpu) => gpu,
            Err(_) => {
                eprintln!("skip: no GPU");
                return;
            }
        };
        let models_dir = std::env::var("HIPFIRE_MODELS_DIR")
            .unwrap_or_else(|_| "/home/kaden/.hipfire/models".to_string());
        let model = format!("{models_dir}/qwen3.8-27b.mq4-xt");
        if !std::path::Path::new(&model).exists() {
            eprintln!("skip: fixture {model} missing");
            return;
        }
        let mut hfq = HfqFile::open(std::path::Path::new(&model)).expect("open model");
        let config: Qwen35Config = config_from_hfq(&hfq).expect("read config");
        assert!(
            matches!(gpu.arch.as_str(), "gfx1100" | "gfx1151"),
            "oracle is gfx11-only (arch={})",
            gpu.arch
        );
        let weights = {
            let mut src = HfqSource::new(&mut hfq, &config);
            let layout = Layout::single(config.n_layers);
            load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
        }
        .expect("load weights");
        let fa_idx = config
            .layer_types
            .iter()
            .position(|t| *t == LayerType::FullAttention)
            .expect("oracle needs a FullAttention layer");
        let wo = match &weights.layers[fa_idx] {
            LayerWeights::FullAttn(l) => &l.wo,
            _ => panic!("oracle needs FullAttn weights at layer {fa_idx}"),
        };
        assert_eq!(
            wo.gpu_dtype,
            rdna_compute::DType::MQ4G256V2,
            "oracle needs uniform MQ4G256V2 wo"
        );
        let Some(awq) = wo.awq_scale.as_ref() else {
            eprintln!("skip: first FA wo carries no AWQ sidecar (fusion AWQ-only)");
            return;
        };
        let k = wo.k;
        assert!(k > 0 && k % 256 == 0, "wo.k={k} must be 256-divisible");

        // Prefill through the first FA layer to populate the REAL FA
        // attention/gate rows the wo chain consumes.
        const N: usize = 128;
        let tokens: Vec<u32> = (0..N as u32).collect();
        let mut kv_cache = KvCache::new_gpu_q8(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            N + 32,
        )
        .expect("kv cache");
        let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn state");
        let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).expect("scratch");
        let pbs = PrefillBatchScratch::new_opt(&mut gpu, &config, N, false).expect("pbs");
        forward_prefill_batch_with_pbs(
            &mut gpu,
            &weights,
            &config,
            &tokens,
            0,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
            None,
            None,
            None,
            None,
            Some(&pbs),
            None,
            Some(fa_idx + 1),
        )
        .expect("FA prefill");
        let attn_all = gpu.download_f32(&pbs.fa_attn_out_batch).expect("dl attn");
        let gate_all = gpu.download_f32(&pbs.fa_gate_batch).expect("dl gate");
        assert!(attn_all.len() >= N * k && gate_all.len() >= N * k);
        let attn_host = attn_all[..N * k].to_vec();
        let gate_host = gate_all[..N * k].to_vec();

        // Unfused chain on device: sigmoid (in-place on a copy) → rotate →
        // standalone quantizer.
        let attn_t = gpu.upload_f32(&attn_host, &[N * k]).expect("up attn");
        let gate_t = gpu.upload_f32(&gate_host, &[N * k]).expect("up gate");
        gpu.sigmoid_mul_f32(&attn_t, &gate_t)
            .expect("unfused sigmoid");
        let x_rot = gpu
            .alloc_tensor(&[N * k], rdna_compute::DType::F32)
            .expect("alloc x_rot");
        llama::rotate_x_mq_batched_for(&mut gpu, wo, &attn_t, &x_rot, k, N)
            .expect("unfused rotate");
        gpu.sync_with_deadline(Duration::from_secs(60))
            .expect("sync chain");
        let f32_unfused = gpu.download_f32(&x_rot).expect("dl unfused f32");
        let xq_unfused = gpu
            .ensure_int4_mmq_x(&x_rot, N, k)
            .expect("standalone quant");
        gpu.sync_with_deadline(Duration::from_secs(60))
            .expect("sync quant");
        let nbytes = (k / 128) * N * 72;
        let mut bytes_unfused = vec![0u8; nbytes];
        let view_unfused = unsafe { DeviceBuffer::from_raw(xq_unfused, nbytes) };
        gpu.hip
            .memcpy_dtoh(&mut bytes_unfused, &view_unfused)
            .expect("dl unfused blocks");
        std::mem::forget(view_unfused);

        // Fused `_gfx11` producer on the same inputs (f32 store written for
        // comparison; production passes None).
        let attn_t2 = gpu.upload_f32(&attn_host, &[N * k]).expect("up attn2");
        let gate_t2 = gpu.upload_f32(&gate_host, &[N * k]).expect("up gate2");
        let x_rot_fused = gpu
            .alloc_tensor(&[N * k], rdna_compute::DType::F32)
            .expect("alloc x_rot fused");
        let res = gpu.reserve_int4_mmq(k, N).expect("reserve");
        let prep = gpu
            .sigmoid_mul_rotate_x_mq_awq_i4_gfx11_batched(
                &attn_t2,
                &gate_t2,
                awq,
                Some(&x_rot_fused),
                res,
                k,
                N,
            )
            .expect("fused sigmoid+quant");
        gpu.sync_with_deadline(Duration::from_secs(60))
            .expect("sync fused");
        let f32_fused = gpu.download_f32(&x_rot_fused).expect("dl fused f32");
        let ptr_fused = gpu.int4_mmq_prepared_ptr(&prep, k, N).expect("prep ptr");
        let mut bytes_fused = vec![0u8; nbytes];
        let view_fused = unsafe { DeviceBuffer::from_raw(ptr_fused, nbytes) };
        gpu.hip
            .memcpy_dtoh(&mut bytes_fused, &view_fused)
            .expect("dl fused blocks");
        std::mem::forget(view_fused);

        assert_eq!(f32_unfused.len(), f32_fused.len(), "f32 length mismatch");
        let mut first_f32_diff = None;
        for (i, (a, b)) in f32_unfused.iter().zip(f32_fused.iter()).enumerate() {
            if a.to_bits() != b.to_bits() && first_f32_diff.is_none() {
                first_f32_diff = Some(i);
                break;
            }
        }
        assert!(
            first_f32_diff.is_none(),
            "fused f32 differs at element {:?} of {}",
            first_f32_diff,
            f32_unfused.len(),
        );
        assert_eq!(
            bytes_unfused.len(),
            bytes_fused.len(),
            "block stream length mismatch"
        );
        let mut first_diff = None;
        for (i, (a, b)) in bytes_unfused.iter().zip(bytes_fused.iter()).enumerate() {
            if a != b && first_diff.is_none() {
                first_diff = Some(i);
                break;
            }
        }
        assert!(
            first_diff.is_none(),
            "fused blocks differ at byte {:?} of {nbytes}",
            first_diff,
        );
        eprintln!(
            "oracle PASS: {} f32 + {nbytes} block bytes identical (K={k} N={N} layer={fa_idx})",
            f32_unfused.len(),
        );
    }
}
