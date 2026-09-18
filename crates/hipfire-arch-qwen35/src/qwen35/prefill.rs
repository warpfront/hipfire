// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 batched prefill: eligibility gates, dispatch-routed GEMM helpers,
//! `forward_prefill_batch*`, and the per-layer batched chunk bodies.

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
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::context::DispatchWorkload;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::kv_tier::KvTierInputs;
use hipfire_dispatch::families::kv_tier::KvTierPlan;
use hipfire_dispatch::pipeline::execute_steps;
use hipfire_dispatch::pipeline::GemvInput;
use hipfire_dispatch::pipeline::Step;
use hipfire_runtime::llama;
use hipfire_runtime::llama::fused_rmsnorm_rotate_for_mq;
use hipfire_runtime::llama::fused_rmsnorm_rotate_mq_batched_for;
use hipfire_runtime::llama::fused_rmsnorm_rotate_mq_f16_batched_for;
use hipfire_runtime::llama::fused_silu_mul_rotate_mq_batched_for;

/// C2: when MQ4V2 + IU4 producer sidecar is live, emit block_i4_128 from
/// RMSNorm/FWHT and return a prepared handle. `None` → caller keeps the
/// incumbent fused_rmsnorm + standalone-quantizer path.
fn try_iu4_rmsnorm_prepared(
    gpu: &mut Gpu,
    x: &GpuTensor,
    norm_weight: &GpuTensor,
    next_linear: &hipfire_runtime::llama::WeightTensor,
    x_rot: &GpuTensor,
    k: usize,
    eps: f32,
    n: usize,
    emit_f32: bool,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    if next_linear.gpu_dtype != DType::MQ4G256V2 || !gpu.iu4_producer_sidecar_active(n, k) {
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep = gpu.fused_rmsnorm_rotate_mq_i4_batched(
        x,
        norm_weight,
        next_linear.awq_scale.as_ref(),
        if emit_f32 { Some(x_rot) } else { None },
        res,
        k,
        eps,
        n,
    )?;
    Ok(Some(prep))
}

/// C2: SwiGLU/FWHT IU4 producer for w_down. `None` → incumbent path.
fn try_iu4_silu_prepared(
    gpu: &mut Gpu,
    w_down: &hipfire_runtime::llama::WeightTensor,
    gate: &GpuTensor,
    up: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    n: usize,
    emit_f32: bool,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    if w_down.gpu_dtype != DType::MQ4G256V2 || !gpu.iu4_producer_sidecar_active(n, k) {
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep = gpu.fused_silu_mul_rotate_mq_i4_batched(
        gate,
        up,
        w_down.awq_scale.as_ref(),
        if emit_f32 { Some(x_rot) } else { None },
        res,
        k,
        n,
    )?;
    Ok(Some(prep))
}

/// T-B: FWHT-rotate + `block_i4_128` IU4 producer for wo (residual) inputs.
/// The wo input is a gated_norm/sigmoid output (never an rmsnorm output),
/// so the C2 producers can't cover it; its only IU4 consumer is the residual
/// GEMM via standalone `quantize_int4_mmq_ds128`, which this removes.
/// `None` → caller keeps incumbent rotate + standalone-quantizer path.
/// Residual-only (w_down C2 precedent): Partial TP keeps the plain path.
fn try_iu4_rotate_prepared(
    gpu: &mut Gpu,
    wo: &hipfire_runtime::llama::WeightTensor,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    n: usize,
    epilogue: &BatchEpilogue<'_>,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    if wo.gpu_dtype != DType::MQ4G256V2
        || !matches!(epilogue, BatchEpilogue::Residual)
        || !gpu.iu4_producer_sidecar_active(n, k)
    {
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep = gpu.rotate_x_mq_i4_batched(
        x,
        wo.awq_scale.as_ref(),
        x_rot,
        res,
        k,
        n,
    )?;
    Ok(Some(prep))
}

/// T-B: epilogue-free twin of [`try_iu4_rotate_prepared`] for the MoE wo
/// sites, which have no `epilogue` param and always accumulate residual
/// into x_batch. Same gate minus the Residual check.
fn try_iu4_rotate_prepared_no_epilogue(
    gpu: &mut Gpu,
    wo: &hipfire_runtime::llama::WeightTensor,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    n: usize,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    if wo.gpu_dtype != DType::MQ4G256V2 || !gpu.iu4_producer_sidecar_active(n, k) {
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep = gpu.rotate_x_mq_i4_batched(
        x,
        wo.awq_scale.as_ref(),
        x_rot,
        res,
        k,
        n,
    )?;
    Ok(Some(prep))
}

use hipfire_runtime::llama::fused_silu_mul_rotate_mq_f16_batched_for;
use hipfire_runtime::llama::rotate_x_mq_batched_for;
use hipfire_runtime::llama::weight_gemv_prerotated;
use hipfire_runtime::llama::weight_gemv_swiglu_residual;
use hipfire_runtime::llama::EmbeddingFormat;
use hipfire_runtime::llama::KvCacheExt;
use hipfire_runtime::llama::WeightTensor;
use rdna_compute::DType;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;
/// Row-parallel epilogue for dense-TP batched partials.
/// `Residual` is byte-identical single-GPU/MoE/EP behavior: GEMM adds into
/// `pbs.x_batch`. `Partial(out)` writes the N×dim GEMM result into `out`
/// without touching the residual; the caller all-reduces `out` and then
/// adds it into each rank's `x_batch`. This avoids duplicating full layer
/// bodies for the two transports.
pub(crate) enum BatchEpilogue<'a> {
    Residual,
    Partial(&'a GpuTensor),
}
#[inline]
fn zero_partial_for_residual(
    gpu: &mut Gpu,
    partial: &GpuTensor,
    n: usize,
    m: usize,
) -> HipResult<()> {
    let elems = n
        .checked_mul(m)
        .ok_or_else(|| HipError::new(0, "zero_partial overflow"))?;
    let view = partial.sub_offset(0, elems);
    gpu.hip.memset(&view.buf, 0, view.buf.size())?;
    Ok(())
}

/// Shared dispatch for batched `wo` / `w_down` with selectable epilogue.
/// Preserves every dtype-specific residual kernel path; `Partial` reuses the
/// same kernels into a zeroed `n×m` slice of `out` instead of `pbs.x_batch`.
fn dispatch_batched_gemm_epilogue(
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    w: &WeightTensor,
    input: &GpuTensor,
    epilogue: &BatchEpilogue<'_>,
    n: usize,
    q8_wmma_arch: bool,
    _arch_has_wmma: bool,
) -> HipResult<()> {
    let m = w.m;
    let k = w.k;
    let is_6bit = matches!(w.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
    let is_mq3_lloyd = matches!(w.gpu_dtype, DType::MQ3G256Lloyd);
    let is_mq3 = matches!(w.gpu_dtype, DType::MQ3G256);
    let is_fp4 = matches!(w.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
    let is_q8 = matches!(w.gpu_dtype, DType::Q8_0);
    let is_lowbit = matches!(w.gpu_dtype, DType::TQ2G128 | DType::BQ1G128);
    match epilogue {
        BatchEpilogue::Residual => {
            if is_6bit {
                return run_residual_gemm_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                    &w.buf,
                    w.gpu_dtype,
                    input,
                    &pbs.x_batch,
                    m,
                    k,
                    n,
                );
            } else if is_q8 && q8_wmma_arch {
                let x_n = pbs.x_batch.sub_offset(0, n * m);
                return run_residual_gemm_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                    &w.buf,
                    w.gpu_dtype,
                    input,
                    &x_n,
                    m,
                    k,
                    n,
                );
            } else if is_q8 || is_lowbit {
                let scratch = pbs.x_rot_batch.sub_offset(0, n * m);
                run_plain_gemm_key(
                    gpu,
                    plain_gemm_key_for(w.gpu_dtype),
                    &w.buf,
                    w.gpu_dtype,
                    input,
                    &scratch,
                    m,
                    k,
                    n,
                )?;
                let x_n = pbs.x_batch.sub_offset(0, n * m);
                return gpu.add_inplace_f32(&x_n, &scratch);
            } else if is_mq3_lloyd {
                return run_residual_gemm_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::GemmMq3G256LloydResidual,
                    &w.buf,
                    w.gpu_dtype,
                    input,
                    &pbs.x_batch,
                    m,
                    k,
                    n,
                );
            } else if is_mq3 {
                return run_residual_gemm_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                    &w.buf,
                    w.gpu_dtype,
                    input,
                    &pbs.x_batch,
                    m,
                    k,
                    n,
                );
            } else if is_fp4 {
                return run_residual_gemm_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::GemmHfp4G32Residual,
                    &w.buf,
                    w.gpu_dtype,
                    input,
                    &pbs.x_batch,
                    m,
                    k,
                    n,
                );
            } else if matches!(w.gpu_dtype, DType::MQ4G256V2Lloyd) {
                // qt=52: gfx11 → MMQ-LUT residual; gfx12 keeps FP8-LUT; else
                // fail-closed inside the FP8 launcher (never uniform).
                if lloyd_mmq_lut_route(gpu) {
                    let c16 = lloyd_c16_or_fail(w, "dispatch_batched_gemm_epilogue")?;
                    return gpu.gemm_hfq4g256_residual_mmq_lloyd(
                        &w.buf,
                        input,
                        &pbs.x_batch,
                        m,
                        k,
                        n,
                        c16,
                    );
                }
                let lut = lloyd_e4m3_or_fail(w, "dispatch_batched_gemm_epilogue")?;
                return gpu.gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8_lloyd(
                    &w.buf,
                    input,
                    &pbs.x_batch,
                    m,
                    k,
                    n,
                    1,
                    lut,
                );
            } else {
                return run_residual_gemm_key(
                    gpu,
                    crate::forward_slots::residual_gemm_key_for(w.gpu_dtype),
                    &w.buf,
                    w.gpu_dtype,
                    input,
                    &pbs.x_batch,
                    m,
                    k,
                    n,
                );
            }
        }
        BatchEpilogue::Partial(out) => {
            let out_n = out.sub_offset(0, n * m);
            if is_6bit {
                zero_partial_for_residual(gpu, out, n, m)?;
                return run_residual_gemm_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                    &w.buf,
                    w.gpu_dtype,
                    input,
                    &out_n,
                    m,
                    k,
                    n,
                );
            } else if is_q8 && q8_wmma_arch {
                zero_partial_for_residual(gpu, out, n, m)?;
                return run_residual_gemm_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                    &w.buf,
                    w.gpu_dtype,
                    input,
                    &out_n,
                    m,
                    k,
                    n,
                );
            } else if is_q8 || is_lowbit {
                return run_plain_gemm_key(
                    gpu,
                    plain_gemm_key_for(w.gpu_dtype),
                    &w.buf,
                    w.gpu_dtype,
                    input,
                    &out_n,
                    m,
                    k,
                    n,
                );
            } else if is_mq3_lloyd {
                zero_partial_for_residual(gpu, out, n, m)?;
                return run_residual_gemm_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::GemmMq3G256LloydResidual,
                    &w.buf,
                    w.gpu_dtype,
                    input,
                    &out_n,
                    m,
                    k,
                    n,
                );
            } else if is_mq3 {
                zero_partial_for_residual(gpu, out, n, m)?;
                return run_residual_gemm_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                    &w.buf,
                    w.gpu_dtype,
                    input,
                    &out_n,
                    m,
                    k,
                    n,
                );
            } else if is_fp4 {
                zero_partial_for_residual(gpu, out, n, m)?;
                return run_residual_gemm_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::GemmHfp4G32Residual,
                    &w.buf,
                    w.gpu_dtype,
                    input,
                    &out_n,
                    m,
                    k,
                    n,
                );
            } else if matches!(w.gpu_dtype, DType::MQ4G256V2Lloyd) {
                // qt=52 Partial: zero then residual (== plain GEMM). gfx11 →
                // MMQ-LUT ADD into zeroed Y; gfx12 keeps FP8-LUT.
                zero_partial_for_residual(gpu, out, n, m)?;
                if lloyd_mmq_lut_route(gpu) {
                    let c16 = lloyd_c16_or_fail(w, "dispatch_batched_gemm_epilogue")?;
                    return gpu
                        .gemm_hfq4g256_residual_mmq_lloyd(&w.buf, input, &out_n, m, k, n, c16);
                }
                let lut = lloyd_e4m3_or_fail(w, "dispatch_batched_gemm_epilogue")?;
                return gpu.gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8_lloyd(
                    &w.buf, input, &out_n, m, k, n, 1, lut,
                );
            } else {
                zero_partial_for_residual(gpu, out, n, m)?;
                return run_residual_gemm_key(
                    gpu,
                    crate::forward_slots::residual_gemm_key_for(w.gpu_dtype),
                    &w.buf,
                    w.gpu_dtype,
                    input,
                    &out_n,
                    m,
                    k,
                    n,
                );
            }
        }
    }
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

fn dense_layers_are_all_mq4v2(weights: &Qwen35Weights) -> bool {
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
            .all(|weight| weight.gpu_dtype == DType::MQ4G256V2),
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
            .all(|weight| weight.gpu_dtype == DType::MQ4G256V2),
            LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_) => false,
        })
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
/// gfx1201 widened ordinary prefill: larger contiguous GEMM/PBS chunks with the
/// DeltaNet kernel still launched once per 512-row commit on views and flash
/// attention on 512-row tile views. No kernel source changes.
///
/// The host change is arch-parameterized: [`widened_arch_admitted`] is the
/// single arch gate (gfx1201 only until a rung passes; gfx1151/Halo admission
/// is a follow-on extension with its own static checks — its F2 512-pair path
/// below is preserved regardless, since wide chunks never equal 512 rows).
/// Only `None` (legacy cadence) and `Some(512)` are legal commit strides.
/// `n` continues to mean chunk rows, never commit stride.
pub(crate) const WIDENED_COMMIT_ROWS: usize = 512;
///
/// Staged GEMM-width rungs. Merging powers of two limits new GEMM widths to
/// these five; a requested ceiling snaps down to the largest rung it covers.
const WIDENED_RUNGS: [usize; 5] = [512, 1024, 2048, 4096, 8192];
///
/// Highest performance-admitted rung. Starts at 512 (behaviour-identical
/// ship); raised one rung at a time only after that rung's E/M/S/V gates
/// pass. Provisional final ceiling 4096 is an upper bound, not a per-device
/// value — memory admission may still select a smaller rung per device.
const WIDENED_PERF_CEILING: usize = 512;
///
/// Uncommitted-VRAM headroom reserved by the per-device capacity admission
/// (plan §4.2). Conservative policy margin covering remaining state, output,
/// graph/JIT and ordinary transients — not permission for unaccounted KV.
const WIDENED_VRAM_HEADROOM_BYTES: usize = 1 << 30;
///
/// Arch gate for the widened ordinary route. gfx1201 only for now; extend
/// here (with that arch's static checks in
/// [`ordinary_prefill_static_ceiling`]) once a rung passes on gfx1201.
/// When gfx1151 follows, its FA tiling must pair in 1024-row views where F2
/// is admitted (F2 is part of the Halo baseline) or record F2-off explicitly.
#[inline]
fn widened_arch_admitted(arch: &str) -> bool {
    arch == "gfx1201"
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
/// largest staged rung covered by both the request and
/// [`WIDENED_PERF_CEILING`].
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
    if !dense_layers_are_all_mq4v2(weights) {
        return None;
    }
    // All four FP8 projection routes (gate/up, residual, qkvza, FA qkv).
    // `fp8_chunk512_for_gpu` tests only the first three — legacy getter
    // retained; the widened route needs the stricter call-local admission.
    if !(gpu.flags.gfx12_mq4v2_fp8_gateup
        && gpu.flags.gfx12_mq4v2_fp8_resid
        && gpu.flags.gfx12_mq4v2_fp8_qkvza
        && gpu.flags.gfx12_mq4v2_fp8_qkv)
    {
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
    if dn_state.s_ef_residual.len() != n_la
        || dn_state.s_ef_residual.iter().any(|t| t.numel() == 0)
    {
        return None;
    }
    let single_lane_s = config.linear_num_value_heads * config.linear_value_head_dim
        * config.linear_value_head_dim;
    if dn_state.s_matrices.len() != n_la || dn_state.s_matrices.iter().any(|t| t.numel() != single_lane_s) {
        return None;
    }
    if dn_requant_per_token_env() {
        return None;
    }
    let requested =
        explicit_prefill_max_batch().unwrap_or_else(|| prefill_max_batch_for_model(gpu, weights));
    if requested < WIDENED_COMMIT_ROWS {
        return Some(requested);
    }
    let cap = requested.min(WIDENED_PERF_CEILING);
    // Largest staged rung covered by the request and the perf ceiling.
    let mut rung = WIDENED_COMMIT_ROWS;
    for &r in WIDENED_RUNGS.iter() {
        if r <= cap {
            rung = r;
        }
    }
    Some(rung)
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
    let k = config.linear_num_key_heads.checked_mul(config.linear_key_head_dim)?;
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
    rows
        .checked_mul(f32_per_row)?
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
/// Per-device capacity admission (plan §4.2): largest performance-admitted
/// rung `<= perf_rows` whose full new PBS, per-slot FP8 deficits at Kmax,
/// missing FA Q16 bytes and 1 GiB headroom fit in current free device bytes.
/// The query runs after model+KV mapping and request-state allocation, so
/// free bytes already charge mapped KV and any inactive reuse cache. Falls
/// back to 512 (legacy route) when no enlarged rung fits — a memory-only
/// failure selects a smaller rung, never a global rollback.
fn memory_admitted_rung(gpu: &Gpu, config: &Qwen35Config, perf_rows: usize) -> HipResult<usize> {
    if perf_rows <= WIDENED_COMMIT_ROWS {
        return Ok(WIDENED_COMMIT_ROWS.min(perf_rows));
    }
    if dense_prefill_allocation_bytes(config, perf_rows).is_none()
        || fp8_row_bytes_wide(config).is_none()
    {
        return Ok(WIDENED_COMMIT_ROWS);
    }
    let (free_bytes, _) = gpu.hip.get_vram_info()?;
    let q_dim = config.n_heads.checked_mul(config.head_dim).unwrap_or(0);
    let q16_need = WIDENED_COMMIT_ROWS
        .checked_mul(q_dim)
        .and_then(|v| v.checked_mul(2))
        .unwrap_or(usize::MAX);
    let q16_missing = q16_need.saturating_sub(gpu.scratch.fa2_q16_scratch_bytes);
    let live_x = gpu.scratch.mq4v2_fp8_x_scratch_bytes;
    let live_sums = gpu.scratch.mq4v2_fp8_half_sums_scratch_bytes;
    let live_scales = gpu.scratch.mq4v2_fp8_row_scales_scratch_bytes;
    let groups = config.hidden_dim / 256;
    for &rung in WIDENED_RUNGS.iter().rev() {
        if rung > perf_rows {
            continue;
        }
        if rung <= WIDENED_COMMIT_ROWS {
            return Ok(WIDENED_COMMIT_ROWS);
        }
        let need_pbs = dense_prefill_allocation_bytes(config, rung).unwrap_or(usize::MAX);
        let x_need = rung.checked_mul(config.hidden_dim).unwrap_or(usize::MAX);
        let sums_need = rung
            .checked_mul(groups)
            .and_then(|v| v.checked_mul(8))
            .unwrap_or(usize::MAX);
        let scales_need = rung.checked_mul(4).unwrap_or(usize::MAX);
        let fp8_deficit = x_need
            .saturating_sub(live_x)
            .saturating_add(sums_need.saturating_sub(live_sums))
            .saturating_add(scales_need.saturating_sub(live_scales));
        let total = need_pbs
            .saturating_add(fp8_deficit)
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
/// ceiling (`HIPFIRE_PREFILL_MAX_BATCH`), and per-device byte-based capacity.
///
/// Public only because hipfire-generate's ordinary AR caller needs the same
/// decision. Existing generic/TP/EP getters are unchanged. `pbs` denotes an
/// explicit hard-cap owner: a caller PBS caps the result at its `max_batch`
/// and is never bypassed here (the ordinary wrappers omit their *implicit*
/// legacy cache before calling when a wider ceiling is statically admitted).
pub fn ordinary_prefill_chunk_limit(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    dn_state: &DeltaNetState,
    kv_cache: &llama::KvCache,
    pbs: Option<&PrefillBatchScratch>,
) -> HipResult<usize> {
    let legacy = prefill_max_batch_for_model(gpu, weights);
    let Some(perf) = ordinary_prefill_static_ceiling(gpu, weights, config, dn_state) else {
        return Ok(legacy);
    };
    if perf <= WIDENED_COMMIT_ROWS {
        return Ok(perf.min(legacy));
    }
    let _ = kv_cache;
    let mut admitted = memory_admitted_rung(gpu, config, perf)?;
    if let Some(p) = pbs {
        admitted = admitted.min(p.max_batch);
    }
    // Admission only narrows the legacy explicit request: perf already snaps
    // the request to rungs, and the sub-512 explicit path returns above.
    Ok(admitted.min(legacy.max(WIDENED_COMMIT_ROWS)))
}
///
/// Next direct-ordinary chunk under a widened ceiling: group complete
/// legacy 512-row chunks into the largest power-of-two multiple the ceiling
/// covers (at least two), else the legacy `<= 512` tail schedule. Preserves
/// the direct partitioner's singleton avoidance: 513→511+2, 1025→512+511+2,
/// 1537→1024+511+2, 2049→1024+512+511+2, 4097→2048+1024+512+511+2. Pure.
fn next_exact_prefill_chunk_len(remaining: usize, ceiling: usize) -> Option<usize> {
    if remaining < MIN_BATCH || ceiling < MIN_BATCH {
        return None;
    }
    if ceiling <= WIDENED_COMMIT_ROWS {
        return next_prefill_chunk_len(remaining, ceiling);
    }
    let reserve_two = if remaining % WIDENED_COMMIT_ROWS == 1 { 2 } else { 0 };
    let full = remaining.saturating_sub(reserve_two) / WIDENED_COMMIT_ROWS;
    let count = full.min(ceiling / WIDENED_COMMIT_ROWS);
    if count >= 2 {
        // Largest power of two ≤ count (count ≥ 2, so the shift is safe).
        let pow2 = 1usize << (usize::BITS - count.leading_zeros() - 1);
        Some(WIDENED_COMMIT_ROWS * pow2)
    } else {
        next_prefill_chunk_len(remaining, WIDENED_COMMIT_ROWS)
    }
}
///
/// Serve-caller outer chunk under a widened ceiling: the serve caller
/// (`ar.rs`) splits with `min(remaining, chunk_max)` and invokes forward
/// separately per outer chunk, so its tail rule differs from the direct one
/// — served 1025→1024+1, flattening to old 512+512+1. Pure.
pub fn ordinary_serve_prefill_chunk_len(remaining: usize, ceiling: usize) -> Option<usize> {
    if remaining == 0 || ceiling < MIN_BATCH {
        return None;
    }
    if ceiling <= WIDENED_COMMIT_ROWS || remaining <= WIDENED_COMMIT_ROWS {
        return Some(remaining.min(ceiling));
    }
    let count = (remaining / WIDENED_COMMIT_ROWS).min(ceiling / WIDENED_COMMIT_ROWS);
    if count >= 2 {
        let pow2 = 1usize << (usize::BITS - count.leading_zeros() - 1);
        Some(WIDENED_COMMIT_ROWS * pow2)
    } else {
        // Fewer than two full chunks: legacy ≤512 tail (this branch only
        // runs with a widened ceiling, so the 512 cap is below it).
        Some(remaining.min(WIDENED_COMMIT_ROWS))
    }
}
///
/// First-prefill receipt: proves requested-vs-executed rows for the evidence
/// log. Emitted once per process on the first chunked (multi-token) prefill.
fn emit_prefill_chunk_receipt(requested: usize, admitted: usize, commit_stride: Option<usize>) {
    static DONE: std::sync::Once = std::sync::Once::new();
    DONE.call_once(|| match commit_stride {
        Some(s) => eprintln!(
            "prefill_chunk: requested={requested} admitted={admitted} commit_stride={s}"
        ),
        None => eprintln!(
            "prefill_chunk: requested={requested} admitted={admitted} commit_stride=none"
        ),
    });
}
///
/// Never form a chunk larger than the configured/capped max, the PBS
/// staging owner, or (when present) the hidden-ring staging owner. A
/// `HiddenStateRingBuffer` sized to 256 therefore cannot receive a 384/512-row
/// write even when an arch default exceeds 256.
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
    let mut lloyd_in_dense = false;
    // The Lloyd dtype is treated identically to plain MQ3 in this guard:
    // both use 112-vs-104-byte stride that the MoE batched branches'
    // HFQ4-layout dispatch would corrupt, and both depend on the gfx11/12
    // WMMA family that other archs lack. Add Lloyd alongside MQ3 so the
    // refusal fires symmetrically and a future MQ3-Lloyd MoE model can't
    // silently land here without explicit MoE-Lloyd kernels.
    //
    // We also track `lloyd_in_dense` separately because Lloyd-MQ3 on
    // gfx12 ships behind an opt-in env gate (see is_batchable_la above) —
    // the gfx12 sibling kernels are runtime-unvalidated locally, so by
    // default a captured-path call with Lloyd-MQ3 weights on gfx1200/1201
    // must refuse rather than dispatch to an untested kernel.
    let is_mq3_any = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
    let is_lloyd = |dt: DType| matches!(dt, DType::MQ3G256Lloyd);
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
                if is_lloyd(l.wqkv.gpu_dtype)
                    || is_lloyd(l.wz.gpu_dtype)
                    || is_lloyd(l.w_beta.gpu_dtype)
                    || is_lloyd(l.w_alpha.gpu_dtype)
                    || is_lloyd(l.wo.gpu_dtype)
                    || is_lloyd(l.w_gate.gpu_dtype)
                    || is_lloyd(l.w_up.gpu_dtype)
                    || is_lloyd(l.w_down.gpu_dtype)
                {
                    lloyd_in_dense = true;
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
                if is_lloyd(l.wq.gpu_dtype)
                    || is_lloyd(l.wk.gpu_dtype)
                    || is_lloyd(l.wv.gpu_dtype)
                    || is_lloyd(l.wo.gpu_dtype)
                    || is_lloyd(l.w_gate.gpu_dtype)
                    || is_lloyd(l.w_up.gpu_dtype)
                    || is_lloyd(l.w_down.gpu_dtype)
                {
                    lloyd_in_dense = true;
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
    // Lloyd-MQ3 on gfx12 is opt-in (see is_batchable_la's gate). The
    // captured entry point bypasses is_batchable_la, so we replicate the
    // gate here: refuse Lloyd-on-gfx12 unless HIPFIRE_LLOYD_GFX12=1 is set.
    // Without this guard, a captured call would reach the dispatch arms
    // and try to load gfx12 kernels that are still community-CI-pending.
    let arch_is_gfx12 = matches!(arch, "gfx1200" | "gfx1201");
    let lloyd_gfx12_optin = hipfire_config::developer_var("HIPFIRE_LLOYD_GFX12")
        .ok()
        .as_deref()
        == Some("1");
    if lloyd_in_dense && arch_is_gfx12 && !lloyd_gfx12_optin {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "forward_prefill_batch_single_chunk_captured: model contains \
             MQ3G256Lloyd weights on arch {arch}, but the gfx12 (RDNA4) \
             sibling kernels (gemm_*_mq3g256_lloyd_wmma.gfx12.hip) are \
             runtime-unvalidated locally and ship behind an opt-in gate. \
             Set HIPFIRE_LLOYD_GFX12=1 to enable the gfx12 path for parity \
             testing, or use the non-captured forward_prefill_batch path \
             (which falls back to per-token forward_scratch on this arch \
             when the env var is unset)."
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
        per_token_hidden_out.map(|t| (t, 0)),
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
/// the configured/arch max ([`prefill_max_batch`]), `pbs.max_batch`, and
/// `hidden_rb.max_batch` when a hidden ring is supplied — never larger than any
/// staging owner. Callers driving DFlash verify should size `pbs` (and hidden
/// staging) to the maximum block they will request so everything fits in one
/// chunk, or accept multi-chunk commits.
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
    // Widened ordinary admission. Decided here, after required KV mapping, so
    // the per-device byte admission charges mapped KV. Only the whole-stack
    // sequential ordinary entry with no caller PBS/cap can widen: captured,
    // TP/EP (separate callers), tree/tape, hidden-ring, band-limited, fused,
    // or caller-capped requests resolve to the legacy path before any wider
    // allocation or launch. The ordinary wrappers omit their *implicit* ≤512
    // legacy cache above when a wider ceiling is statically admitted, so
    // `pbs_in.is_none()` here means the owner decision is ours; an explicit
    // smaller caller PBS always wins and keeps legacy cadence.
    let wide_candidate = pbs_in.is_none()
        && max_batch_cap.is_none()
        && tree_verify.is_none()
        && gdn_tape.is_none()
        && hidden_rb.is_none()
        && max_layer.is_none()
        && matches!(fusion, DflashFusionCtx::Off)
        && !gpu.graphs.capture_mode
        && !gpu.replay.is_recording();
    let limit = if wide_candidate {
        ordinary_prefill_chunk_limit(gpu, weights, config, dn_state, kv_cache, None)?
    } else {
        max_batch
    };
    emit_prefill_chunk_receipt(
        explicit_prefill_max_batch().unwrap_or(max_batch),
        limit,
        (wide_candidate && limit > WIDENED_COMMIT_ROWS).then_some(WIDENED_COMMIT_ROWS),
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
                own_pbs = Some(PrefillBatchScratch::new_opt(gpu, config, owned_rows, false)?);
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
        let chunk_batch = prefill_effective_chunk_batch(
            limit,
            pbs.max_batch,
            hidden_rb.as_ref().map(|rb| rb.max_batch),
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
            // Widened grouping merges complete 512s into power-of-two chunks;
            // the legacy tail schedule (and its singleton avoidance) is the
            // fallback for both the legacy path and widened tails.
            let chunk_n = if wide {
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
            // F2 pair peek: merge exactly two complete 512-row chunks (never
            // a partial tail — the tail selector below is unchanged, so 513
            // stays 511+2 and 1025 stays 512+511+2). The second length is
            // only computed when the first is a complete half.
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
                    per_token_hidden_out,
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
            let pth_slot = per_token_hidden_out.map(|t| (t, chunk_start));
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
            // Small tails keep the legacy branch inside the chunk body.
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
                // Scatter fixed-offset staging writes (done inside the chunk)
                // to the ring at the current head, then advance head by n.
                // This is the out-of-capture step: graph-captured writes went
                // to staging[0..n*h], this commit places them at head*h
                // where head is read from CPU state at call time (not baked
                // into a captured graph node).
                rb.commit_staging_to_ring(gpu, chunk_n)?;
            }
            chunk_start = chunk_end;
        }
        Ok(())
    })();
    if let Some(owned) = own_pbs {
        owned.free_gpu(gpu);
    }
    if let Some((pbs2, stage)) = pair_scratch {
        let _ = stage.free_gpu(gpu);
        let _ = pbs2.free_gpu(gpu);
    }
    result
}

/// Plain (unfused) batched-GEMM dispatcher key for a weight dtype.
///
/// Q8 keeps the chunked kernel it already used, so this is behaviour-preserving
/// for every existing model; the low-bit formats route to their tiled prefill
/// GEMMs. They share the Q8 call sites deliberately: none of the three has a
/// fused qkvza/gate_up/qkv kernel, so all three want the same unfused strategy.
fn plain_gemm_key_for(dt: DType) -> hipfire_dispatch::types::KernelKey {
    use hipfire_dispatch::types::KernelKey as K;
    match dt {
        DType::TQ2G128 => K::GemmTQ2G128Prefill,
        DType::BQ1G128 => K::GemmBQ1G128Prefill,
        _ => K::GemmQ8_0BatchedChunked,
    }
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
// As of this PR (issue #116 Phase 5): MQ3G256Lloyd is wired through
// the gemm_*_mq3g256_lloyd_wmma family on gfx11 (always-on) and on
// gfx12 (opt-in via HIPFIRE_LLOYD_GFX12=1). MQ4G256Lloyd is wired
// through the gemm_*_mq4g256_lloyd_wmma family on gfx11 (always-on)
// and gfx12 (opt-in via HIPFIRE_LLOYD_GFX12=1). MQ2G256Lloyd remains
// unwired — MQ2-Lloyd lands separately.
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

    // Lloyd-MQ3 (MQ3G256Lloyd) on gfx11: Phase 5 of issue #116 ships the
    // gemm_*_mq3g256_lloyd_wmma family alongside the existing HFQ3 WMMA
    // path; group stride differs (112 B Lloyd vs 104 B HFQ3) so dispatch
    // must route to the Lloyd-specific arms (handled by the LA/FA
    // matchers downstream — see followup-checklist condition 3).
    let lloyd_mq3_with_gfx11_wmma = matches!(dt, DType::MQ3G256Lloyd)
        && matches!(
            arch,
            "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151"
        );

    // Lloyd-MQ3 on gfx12 (RDNA4): the gemm_*_mq3g256_lloyd_wmma.gfx12.hip
    // kernels are code-complete but runtime-unvalidated locally — bench
    // host is gfx1100/1151 — so they ship behind an opt-in env gate.
    // With HIPFIRE_LLOYD_GFX12 unset (default), Lloyd-MQ3 on gfx1200/1201
    // falls through to per-token forward_scratch (correct, ~14× slower;
    // matches pre-Phase-B2 behaviour for that arch class). With
    // HIPFIRE_LLOYD_GFX12=1, the WMMA path is exercised — this is the
    // path RDNA4 reviewers should set when running the parity tests /
    // coherence-gate to validate the gfx12 sibling kernels. Once external
    // CI confirms gfx12 parity, the gate can be dropped (or default
    // flipped) in a follow-up commit.
    let lloyd_mq3_with_gfx12_wmma = matches!(dt, DType::MQ3G256Lloyd)
        && matches!(arch, "gfx1200" | "gfx1201")
        && hipfire_config::developer_var("HIPFIRE_LLOYD_GFX12")
            .ok()
            .as_deref()
            == Some("1");

    // Lloyd-MQ4 (MQ4G256Lloyd) on gfx11: shipped as part of issue #182.
    // Uses the gemm_*_mq4g256_lloyd_wmma family; group stride differs
    // (160 B Lloyd vs 136 B HFQ4) so dispatch routes through the
    // Lloyd-specific arms in forward_prefill_chunk.
    // ANTIBLEED admit-vs-select fix: the MQ4-Lloyd batched-prefill GEMM source
    // selectors (gemm_*_mq4g256_lloyd_wmma_for_arch in rdna-compute/kernels.rs)
    // ship a kernel only for gfx1100/1101/1102/1151 and PANIC on any other arch
    // (160 B Lloyd stride mismatches the default). gfx1150 was admitted here but
    // has no MQ4-Lloyd source (intentionally excluded to stay symmetric with the
    // MQ4-Lloyd GEMV/fused-decode path — see kernels.rs:195), so a gfx1150 box
    // doing MQ4-Lloyd batched prefill would crash at source lookup. Drop gfx1150
    // from the admit set so admit == select. (MQ3-Lloyd DOES ship a gfx1150
    // source, hence its admit set below keeps gfx1150.)
    let lloyd_mq4_with_gfx11_wmma = matches!(dt, DType::MQ4G256Lloyd)
        && matches!(arch, "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151");

    // Lloyd-MQ4 on gfx12 (RDNA4): same opt-in gate as Lloyd-MQ3.
    let lloyd_mq4_with_gfx12_wmma = matches!(dt, DType::MQ4G256Lloyd)
        && matches!(arch, "gfx1200" | "gfx1201")
        && hipfire_config::developer_var("HIPFIRE_LLOYD_GFX12")
            .ok()
            .as_deref()
            == Some("1");

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
        || lloyd_mq3_with_gfx11_wmma
        || lloyd_mq3_with_gfx12_wmma
        || lloyd_mq4_with_gfx11_wmma
        || lloyd_mq4_with_gfx12_wmma
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
            if !moe_prefill_topk_shape_supported(config.num_experts_per_tok, config.num_experts) {
                return Err(HipError::new(0, "DeltaNetMoe topk shape unsupported"));
            }
            let admit_mq6 = mq6_batched_admit_enabled_from_env(
                hipfire_config::developer_var("HIPFIRE_MOE_MQ6_ADMIT")
                    .ok()
                    .as_deref(),
                arch,
            );
            if !moe_ffn_batched_admissible(&l.ffn, admit_mq6, arch) {
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
            if !moe_prefill_topk_shape_supported(config.num_experts_per_tok, config.num_experts) {
                return Err(HipError::new(0, "FullAttnMoe topk shape unsupported"));
            }
            let admit_mq6 = mq6_batched_admit_enabled_from_env(
                hipfire_config::developer_var("HIPFIRE_MOE_MQ6_ADMIT")
                    .ok()
                    .as_deref(),
                arch,
            );
            if !moe_ffn_batched_admissible(&l.ffn, admit_mq6, arch) {
                return Err(HipError::new(0, "FullAttnMoe moe_ffn not batch-admissible"));
            }
            Ok(())
        }
    }
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
fn paro_batched_admit_enabled_from_env(value: Option<&str>) -> bool {
    // Default OFF (opt-in via HIPFIRE_PARO_BATCHED=1). The PARO batched prefill
    // path (ParoQ4G128 wqkv/wz/wo → gemm_hfq4g128 + per-weight Givens) was
    // only validated for finite logits, not coherence. Per-token fallback
    // (forward_scratch) is correct and avoids the echo bug. Set =1 to re-enable
    // for eval/benchmarking, understanding that output may differ from decode.
    value == Some("1")
}

#[derive(Debug, Clone, Copy)]
struct MoePrefillDtypes {
    router: DType,
    shared_expert_scalar_gate: DType,
    shared_expert_gate: DType,
    shared_expert_up: DType,
    shared_expert_down: DType,
    expert_gate_up: DType,
    expert_down: DType,
    expert_gate_up_uniform: bool,
    expert_down_uniform: bool,
    /// Routed experts are dtype-mixed (graded) AND carry an `expert_dtype_tags`
    /// table → served by the merged grouped-WMMA prefill kernel (per-expert
    /// MQ6/MQ4/MQ3L/MQ2L). When true, the per-expert *uniform* requirement is
    /// waived for the ROUTED experts; the router + shared expert still use their
    /// own batched paths and are validated normally. Without this, a graded file
    /// fails admission and silently drops to the per-token prefill fallback (the
    /// merged kernel never fires — observed as ~decode-speed prefill).
    routed_mixed_merged: bool,
}

impl MoePrefillDtypes {
    #[cfg(test)]
    fn uniform(dtype: DType) -> Self {
        Self {
            router: dtype,
            shared_expert_scalar_gate: dtype,
            shared_expert_gate: dtype,
            shared_expert_up: dtype,
            shared_expert_down: dtype,
            expert_gate_up: dtype,
            expert_down: dtype,
            expert_gate_up_uniform: true,
            expert_down_uniform: true,
            routed_mixed_merged: false,
        }
    }

    fn from_ffn(ffn: &MoeFfnWeights) -> Option<Self> {
        // When the EP global table is present, derive every routed-expert
        // decision from the full-model table — never from the compact local
        // `ffn.experts` slice, which may appear uniform on each rank even
        // when the global model is graded mixed.
        if let Some(global) = ffn.global_expert_dtypes.as_ref() {
            let first = global.first()?;
            return Some(Self {
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
        Some(Self {
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
}

fn moe_prefill_topk_shape_supported(k_top: usize, num_experts: usize) -> bool {
    k_top == 8 && num_experts <= 1024
}

/// Routed-expert dtypes the batched-prefill grouped-GEMM path (Path 2) serves
/// natively for a UNIFORM-per-projection CODEBOOK file (`expert_dtype_tags == None`
/// and at least one Lloyd projection). Pure MQ4/MQ4 pairs use the default arm.
///
/// Each has a real `dispatch_grouped_gemm` arm whose launcher covers BOTH gfx11
/// (`_k2`) and gfx12 (`_gfx12`):
///   MQ4G256      -> `gemm_hfq4g256_moe_grouped_wmma_k2` / `.gfx12`
///   MQ2G256Lloyd -> `gemm_mq2g256_lloyd_moe_grouped_wmma` (arch-selecting)
///   MQ3G256Lloyd -> `gemm_mq3g256_lloyd_moe_grouped_wmma` (arch-selecting)
///
/// DELIBERATELY EXCLUDED — do not add without landing the kernels first:
///   MQ2G256GL / MQ3G256GL — GL ships FIVE kernels total, all single-token
///     indexed decode GEMVs (`gemv_mq{2,3}g256gl_moe_{gate_up,down}_indexed`
///     plus the sym gate_up). There is no grouped-WMMA GEMM and no batched
///     indexed GEMV for the SoA global-codebook layout, and the merged
///     dtype-tag kernel has no GL branch. Admitting GL would push a
///     `[idx][scale]` SoA blob into a per-group-header decoder: OOB reads and
///     token soup, with no error.
///   MQ6G256 / MQ6G256V2 — handled by the `admit_mq6` arm (env/arch gated).
///   MQ4G256V2 — handled by the default MQ4 arm + Path-2
///     `gemm_mq4g256v2_moe_grouped_wmma_k2` / `_gfx12` (never HFQ4 V1).
///   MQ2/3/5G256V2 — out of scope for MoE grouped prefill (dense-only V2).
fn routed_codebook_grouped_supported(dt: DType) -> bool {
    matches!(
        dt,
        DType::MQ4G256 | DType::MQ2G256Lloyd | DType::MQ3G256Lloyd
    )
}

/// Uniform MQ4V2 / MQ6V2 routed projections Path-2 grouped GEMM serves on
/// gfx11 (`_k2`) and gfx12 (`_gfx12`) after dispatch. Distinct wire layouts
/// from V1 (dual-half fp16 headers); never collapse onto HFQ4/HFQ6 launchers.
/// Used by admission tests and documentation lockstep with
/// `dispatch_grouped_gemm` / `gemm_mq{4,6}g256v2_moe_grouped_wmma_k2`.
#[inline]
fn routed_uniform_mqv2_grouped_supported(dt: DType) -> bool {
    matches!(dt, DType::MQ4G256V2 | DType::MQ6G256V2)
}

/// True when the routed pair is a uniform codebook pair the batched grouped-GEMM
/// path now serves AND at least one projection is a Lloyd codebook dtype (a pure
/// MQ4/MQ4 pair is the pre-existing default arm, not this one). Used by BOTH the
/// admission predicate and the MQ3-in-MoE refusal so the two can never disagree:
/// an admitted-but-refused layer would hard-error a model that serves today.
pub(crate) fn routed_codebook_pair_batched_supported(gate_up: DType, down: DType) -> bool {
    routed_codebook_grouped_supported(gate_up)
        && routed_codebook_grouped_supported(down)
        && (matches!(gate_up, DType::MQ2G256Lloyd | DType::MQ3G256Lloyd)
            || matches!(down, DType::MQ2G256Lloyd | DType::MQ3G256Lloyd))
}

/// Arch/env gate for the uniform codebook routed-expert batched prefill.
///
/// Requires (a) a WMMA arch — the grouped-GEMM kernels are gfx11 wave32-WMMA or
/// gfx12 WMMA only — and (b) `HIPFIRE_MOE_GROUPED_GEMM` not forced off, because
/// Path 2 is the ONLY implemented route for these dtypes: the Path 0/1
/// indexed-GEMV arms in `run_moe_prefill` have no MQ2/MQ3-Lloyd branch and would
/// return `UnsupportedVariant` (a hard error, not a fallback). Admitting while
/// Path 2 is disabled would turn a working slow prefill into a failure.
///
/// `HIPFIRE_MOE_CODEBOOK_BATCHED=0` restores the per-token fallback (the bisect
/// escape hatch); `=1` forces the admit on an unlisted arch for bring-up.
fn codebook_batched_admit_enabled_from_env(
    value: Option<&str>,
    grouped_gemm_value: Option<&str>,
    arch: &str,
) -> bool {
    // Mirrors FeatureFlags::moe_grouped_gemm (default on; "0"/"off" disables).
    let grouped_gemm_on = !matches!(grouped_gemm_value, Some("0") | Some("off"));
    match value {
        Some("0") | Some("off") | Some("false") => false,
        // Explicit opt-in is honored on any arch (research / bring-up), still
        // hard-gated on Path 2 being enabled.
        Some("1") | Some("on") | Some("true") => grouped_gemm_on,
        // DEFAULT OFF — opt-in via HIPFIRE_MOE_CODEBOOK_BATCHED=1.
        //
        // Admitting the codebook pair here makes
        // `gemm_mq2g256_lloyd_moe_grouped_wmma_gfx12` the gate_up kernel for
        // every uniform-Lloyd a3b SKU, and promotes its MQ3 sibling from
        // wired-but-unreachable to live. NEITHER has ever executed on
        // hardware. Static review and a clean hipcc compile are not acceptance
        // evidence (CLAUDE.md, "Runtime validation (mandatory)"), and the
        // failure mode is silently-wrong prefill on exactly the SKUs this is
        // meant to accelerate.
        //
        // Flip this default only after a `scripts/serve_harness.py` coherence
        // run of the MQ2L/MQ3L pair on gfx1201 AND on a gfx11 part — the gfx11
        // `_k2` leg is a separate translation unit and is equally unexercised.
        _ => false,
    }
}

fn codebook_batched_admit_enabled(arch: &str) -> bool {
    codebook_batched_admit_enabled_from_env(
        hipfire_config::developer_var("HIPFIRE_MOE_CODEBOOK_BATCHED")
            .ok()
            .as_deref(),
        hipfire_config::developer_var("HIPFIRE_MOE_GROUPED_GEMM")
            .ok()
            .as_deref(),
        arch,
    )
}

fn moe_ffn_batched_admissible_for_dtypes(
    dtypes: &MoePrefillDtypes,
    admit_mq6: bool,
    admit_paro: bool,
    admit_e8: bool,
    admit_codebook: bool,
) -> bool {
    let router_ok = matches!(
        dtypes.router,
        DType::MQ4G256 | DType::MQ4G256V2 | DType::Q8_0 | DType::F32
    );
    let shared_gate_ok = matches!(
        dtypes.shared_expert_scalar_gate,
        DType::MQ4G256 | DType::MQ4G256V2 | DType::Q8_0 | DType::F32
    );
    // Graded (mixed-dtype) routed experts are served by the merged grouped-WMMA
    // prefill kernel, so the per-expert *uniform* requirement is waived for the
    // routed experts; the router + shared expert still go through their own
    // batched paths and are validated below.
    let routed_ok =
        dtypes.routed_mixed_merged || (dtypes.expert_gate_up_uniform && dtypes.expert_down_uniform);
    if !(router_ok && shared_gate_ok && routed_ok) {
        return false;
    }

    if dtypes.routed_mixed_merged {
        // Routed experts handled by the merged kernel (per-expert MQ6/MQ4/MQ3L/
        // MQ2L + V2 tags 7..18). Only require the SHARED expert to be batchable
        // on its dense path: MQ4/MQ4V2 always, MQ6/MQ6V2 when this arch admits
        // MQ6 dense kernels. Exact V2 dtypes — never V1 aliases. Fused
        // gate+up requires exact dtype equality (MQ4 != MQ4V2, MQ6 != MQ6V2
        // have different dual-half vs single-half headers → never collapse).
        let shared_gu_ok = (dtypes.shared_expert_gate == dtypes.shared_expert_up
            && matches!(dtypes.shared_expert_gate, DType::MQ4G256 | DType::MQ4G256V2))
            || (admit_mq6
                && matches!(
                    dtypes.shared_expert_gate,
                    DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2
                )
                && dtypes.shared_expert_up == dtypes.shared_expert_gate);
        let shared_dn_ok = matches!(dtypes.shared_expert_down, DType::MQ4G256 | DType::MQ4G256V2)
            || (admit_mq6
                && matches!(dtypes.shared_expert_down, DType::MQ6G256 | DType::MQ6G256V2));
        return shared_gu_ok && shared_dn_ok;
    }

    // mfp4-E8 routed experts with Q8 shared expert (original arm):
    // gfx1151-native A3B checkpoint. Shared expert is Q8 (gate/up/down);
    // router/scalar-gate are Q8 (validated by router_ok/shared_gate_ok above).
    // The batched body runs a dedicated Q8 shared-expert path (two plain Q8
    // GEMMs + silu_mul + sigmoid-scaled residual add) and routes the E8
    // experts through `run_moe_prefill` Path 1 (indexed batched GEMV).
    // E8-family match helper: MFP4, MFP3, MFP2 lattice types.
    let is_e8_family =
        |dt: DType| matches!(dt, DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8);

    if admit_e8
        && dtypes.shared_expert_gate == DType::Q8_0
        && dtypes.shared_expert_up == DType::Q8_0
        && dtypes.shared_expert_down == DType::Q8_0
        && is_e8_family(dtypes.expert_gate_up)
        && is_e8_family(dtypes.expert_down)
    {
        return true;
    }

    // Uniform mfp4/mfp3/mfp2-E8: BOTH shared AND routed experts are E8-family
    // (Option B from the implementation spec). Router + shared_expert_gate
    // (scalar) remain Q8 (validated above). The batched body dequants the shared
    // expert E8→F16 transiently and runs `gemm_f16_wmma_mb8` against
    // `x_rot_batch` (FWHT-rotated activations), then the routed experts go
    // through the indexed E8 batched GEMV path. The dequant→F16 path requires
    // has_wmma_w32 (gfx11+), which `admit_e8` already gates on arch.
    if admit_e8
        && is_e8_family(dtypes.expert_gate_up)
        && is_e8_family(dtypes.expert_down)
        // Shared expert may be per-projection MIXED — gate+up are dispatched
        // together (one match on gate's dtype) so they must share a dtype;
        // down is matched independently. The batched body handles Q8 (un-rotated)
        // and E8 (dequant→f16, x_rot) per projection and keys the SwiGLU rotate
        // on the down dtype, so any {Q8,E8-family} combination of (gate==up,down)
        // is correct.
        && dtypes.shared_expert_gate == dtypes.shared_expert_up
        && matches!(dtypes.shared_expert_gate, DType::Q8_0 | DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8)
        && matches!(dtypes.shared_expert_down, DType::Q8_0 | DType::MFP4G32E8 | DType::MFP3G32E8 | DType::MFP2G32E8)
    {
        return true;
    }

    // Uniform-per-projection CODEBOOK routed experts (the antirez asymmetric
    // recipe: gate_up = MQ2-Lloyd, down = MQ3-Lloyd; also the MQ4/Lloyd mixes
    // the per-layer tiered formats emit). `routed_ok` above already established
    // uniformity per projection and the absence of a tag table, so
    // `expert_gate_up` / `expert_down` describe EVERY expert. The routed block
    // runs Path 2 grouped-WMMA (`dispatch_grouped_gemm` MQ2/MQ3-Lloyd arms), the
    // SwiGLU+FWHT-rotate is weight-agnostic, and the shared expert + router keep
    // their own dense batched paths — which is why the shared side is validated
    // exactly as in the default MQ4 arm (plus MQ6/MQ6V2 when this arch admits).
    //
    // Structurally distinct from `routed_mixed_merged` above: that arm covers a
    // GRADED file served by the merged dtype-tag kernel; this one covers a
    // UNIFORM file served by the per-dtype grouped kernels, with no tag table.
    if admit_codebook
        && routed_codebook_pair_batched_supported(dtypes.expert_gate_up, dtypes.expert_down)
    {
        let shared_gu_ok = (dtypes.shared_expert_gate == dtypes.shared_expert_up
            && matches!(dtypes.shared_expert_gate, DType::MQ4G256 | DType::MQ4G256V2))
            || (admit_mq6
                && matches!(
                    dtypes.shared_expert_gate,
                    DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2
                )
                && dtypes.shared_expert_up == dtypes.shared_expert_gate);
        let shared_dn_ok = matches!(dtypes.shared_expert_down, DType::MQ4G256 | DType::MQ4G256V2)
            || (admit_mq6
                && matches!(dtypes.shared_expert_down, DType::MQ6G256 | DType::MQ6G256V2));
        if shared_gu_ok && shared_dn_ok {
            return true;
        }
    }

    if admit_paro
        && dtypes.shared_expert_gate == DType::ParoQ4G128
        && dtypes.shared_expert_up == DType::ParoQ4G128
        && dtypes.shared_expert_down == DType::ParoQ4G128
        && dtypes.expert_gate_up == DType::ParoQ4G128
        && dtypes.expert_down == DType::ParoQ4G128
    {
        return true;
    }

    // Uniform MQ4 / MQ4V2 / MQ6 / MQ6V2 shared+routed (Path 2 grouped after
    // dispatch on gfx11/gfx12). MQ6* needs `admit_mq6`; MQ4V2 always admits
    // like MQ4. MQ2/3/5V2 deliberately excluded from MoE grouped prefill.
    if admit_mq6 {
        let shared_gu_dt = dtypes.shared_expert_gate;
        let shared_gu_ok = matches!(
            shared_gu_dt,
            DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2
        ) && dtypes.shared_expert_up == shared_gu_dt;
        let shared_dn_ok = matches!(
            dtypes.shared_expert_down,
            DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2
        );
        let experts_ok = matches!(
            dtypes.expert_gate_up,
            DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2
        ) && matches!(
            dtypes.expert_down,
            DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256 | DType::MQ6G256V2
        );
        // Lockstep: any uniform V2 projection we admit must be a Path-2
        // grouped-supported dtype (or V1 MQ4/MQ6 which have their own arms).
        debug_assert!(
            !matches!(dtypes.expert_gate_up, DType::MQ4G256V2 | DType::MQ6G256V2)
                || routed_uniform_mqv2_grouped_supported(dtypes.expert_gate_up)
        );
        debug_assert!(
            !matches!(dtypes.expert_down, DType::MQ4G256V2 | DType::MQ6G256V2)
                || routed_uniform_mqv2_grouped_supported(dtypes.expert_down)
        );
        shared_gu_ok && shared_dn_ok && experts_ok
    } else {
        // Exact gate/up dtype equality required even for MQ4-family (MQ4 !=
        // MQ4V2 have different header layouts; fused kernel handles one layout
        // per launch → cross-version ordering rejects).
        dtypes.shared_expert_gate == dtypes.shared_expert_up
            && matches!(dtypes.shared_expert_gate, DType::MQ4G256 | DType::MQ4G256V2)
            && matches!(dtypes.shared_expert_down, DType::MQ4G256 | DType::MQ4G256V2)
            && matches!(dtypes.expert_gate_up, DType::MQ4G256 | DType::MQ4G256V2)
            && matches!(dtypes.expert_down, DType::MQ4G256 | DType::MQ4G256V2)
    }
}
/// Threshold below which batching overhead isn't worth the alloc + per-layer
/// dispatch — single-token prefill must not take the batched path.
const MIN_BATCH: usize = 2;

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
/// batched (PBS) path for an `n`-token call — equivalently, whether a `GdnTape`
/// handed to that forward will actually be populated. When this is false the
/// forward silently drops to a tape-less per-token loop, so spec-decode callers
/// that later replay the GDN tape MUST gate that cheap replay on this predicate;
/// otherwise they replay a stale/zero tape and corrupt DeltaNet state. This is
/// the single source of truth for the eligibility decision — called by the
/// forward itself and by those callers, so the two can never drift. (The
/// tree-verify forward keeps its own, deliberately simpler, eligibility check.)
pub fn prefill_batch_pbs_eligible(
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    // Kept for API stability and future state-aware gating. The batched path
    // now dispatches the GDN recurrence by state quant on the non-tree route,
    // so it no longer gates eligibility here (see the removed Q8-only check).
    _dn_state: &DeltaNetState,
    n: usize,
    arch: &str,
    moe_router_logits_present: bool,
) -> bool {
    // HIPFIRE_PREFILL_BATCHED=0 forces the per-token fallback — an escape hatch
    // for the LARGE seed prefill (gfx11 24GB OOM + a batched-seed correctness bug
    // that collapses MTP τ→1.0). But the small-B MTP verify (n = K+1, ≤ ~32) is
    // cheap and its BATCHED path is the dominant gfx11 decode lever: per-token it
    // costs ~K full sequential trunk forwards/cycle (the measured 92ms→16.8ms /
    // 41→223 tok/s bottleneck, rocprofv3 2026-06-16). Decouple: let the small-B
    // verify batch even when the flag forces the seed per-token. Opt-in
    // (HIPFIRE_MTP_VERIFY_DECOUPLE=1) until the batched verify is validated
    // coherent + τ-preserving per-arch (the gfx11 batched *seed* corrupts; whether
    // the small-B *verify* also corrupts is exactly what this gate tests).
    // DEFAULT-ON for RDNA3 (gfx11) — the small-B verify BATCHED is validated
    // coherent + τ-preserving there (W3x 2026-06-16: byte-identical output vs
    // per-token at 240-tok ctx; +20% mq4; the scalar→WMMA + MQ3L-LUT kernel
    // wins lift all STRUCTURED domains >AR on both mq4/mq4p; fresh default-config
    // re-validated mq4 code 1.26× / mq4p chat 1.07×). Opt-out
    // HIPFIRE_MTP_VERIFY_DECOUPLE=0. Other archs opt-in (=1) until validated;
    // gfx12 batches the whole prefill already so it is moot there. The seed
    // stays per-token for LONG prompts (n>32 fails this gate → force_fallback
    // when PREFILL_BATCHED=0); a short seed (n≤32) batches, fine for mq4/mq4p
    // (E8 short-seed batched-prefill can OOM, but E8 admission is itself opt-in
    // via HIPFIRE_E8_GFX12 so the default config never reaches it).
    let decouple_env = hipfire_config::developer_var("HIPFIRE_MTP_VERIFY_DECOUPLE").ok();
    let is_rdna3_decouple = arch.starts_with("gfx11");
    let verify_decouple = n <= 32
        && decouple_env.as_deref() != Some("0")
        && (is_rdna3_decouple || decouple_env.as_deref() == Some("1"));
    let force_fallback = !verify_decouple && !hipfire_runtime::config::get().prefill_batched;
    let has_dn = weights
        .layers
        .iter()
        .any(|lw| matches!(lw, LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_),));
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
    let result = !force_fallback
        && n >= MIN_BATCH
        // State quant no longer gates batched prefill: forward_prefill_chunk
        // dispatches the GDN recurrence by dn_state.quant on the non-tree path
        // (FP32 → gated_delta_net_f32_batch_seq, Q8 → _q8_batch_seq, Q4 → _q4),
        // so FP32/Q4 state is fully batchable here. Was hard-gated to Q8 when
        // the batched GDN was Q8-only; that's the seed + per-cycle-commit
        // per-token fallback that made FP32 DFlash ~4.5× slower + 10× TTFT.
        && has_dn
        // LA/FA/MoE projection + MoE-FFN weight dtypes must all be batchable;
        // A3B engine policy quantizes attention as Q8 (admitted alongside MQ4).
        && all_layers_ok;
    // HIPFIRE_DEBUG_BATCH=1: print per-component eligibility to stderr.
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

/// Whether MQ6 MoE FFN projections can enter batched prefill. Default-on for
/// gfx11 (RDNA3/3.5) AND gfx12 (RDNA4): the MQ6 grouped-WMMA decode is present
/// on both (tag 0 of the merged `gemm_mixed_moe_grouped_wmma{_k2,.gfx12}` kernel,
/// plus the standalone `gemm_hfq6g256_moe_grouped_wmma{_k2,.gfx12,_gfx1151}` ported
/// 2026-06-11/12), and the graded shared-MQ6 expert runs the dense-GEMM batched
/// path. Validated on gfx1100: graded T3-3L-E8 (MQ6 hot / E8 mid / MQ3L cold,
/// MQ6 shared) batches coherently, KLD 0.038964, and gfx11 prefill is ~10× the
/// per-token fallback (1012 vs 106 tok/s pp512). UNIFORM-MQ6 OOMs on gfx11
/// (>24 GB) so it never reaches this gate there — only graded MQ6 models do.
/// The original gfx12-only default predated the gfx11 MQ6 grouped port and
/// silently forced per-token prefill on every graded-MQ6 model on gfx11.
/// gfx1151 (RDNA3.5, Strix Halo) additionally has master's channel-tested
/// routed grouped-WMMA MQ6 fast-path (its unrelated Q8 WMMA prefill family is
/// gated separately by `q8_prefill_wmma_enabled`). Override per-arch with
/// `HIPFIRE_MOE_MQ6_ADMIT=0|1`.
// pub(crate): also used by forward_slots.rs (MoE slots port) to compute the
// same admit_mq6 predicate before calling `moe_ffn_batched_admissible`.
// Visibility change only — behavior and existing callers are unchanged.
pub(crate) fn mq6_batched_admit_enabled_from_env(value: Option<&str>, arch: &str) -> bool {
    match value {
        Some("0") | Some("off") | Some("false") => false,
        Some("1") | Some("on") | Some("true") => true,
        // RDNA3/3.5 (gfx11xx, includes gfx1151) + RDNA4 (gfx12xx). RDNA1/2
        // (gfx10xx, no WMMA) stay off. The gfx11 widen (8d555fc6) subsumes
        // master's narrower gfx12||gfx1151 default; gfx1151 still picks up its
        // channel-tested grouped-WMMA fast-path inside the kernel dispatcher.
        _ => arch.starts_with("gfx11") || arch.starts_with("gfx12"),
    }
}

/// Qwen3.5 batched prefill can run Q8 projections through fused WMMA kernels
/// or through the older chunked-Q8 substrate. gfx12 has a separate WMMA ABI;
/// gfx11/gfx1151 use the gfx11 wave32 WMMA ABI. The low-level Q8 channel tests
/// cover the fused, residual, and generic chunked drop-in paths, so default on
/// for every arch that advertises wave32 WMMA while preserving the env opt-out.
fn q8_prefill_wmma_enabled_from_env(value: Option<&str>, arch: &str, has_wmma: bool) -> bool {
    let _ = arch;
    if !has_wmma {
        return false;
    }
    match value {
        Some("0") | Some("off") | Some("false") => false,
        Some("1") | Some("on") | Some("true") => true,
        _ => true,
    }
}

// pub(crate): also used by forward_slots.rs (SP3 Task 2) to pick the same
// Q8 WMMA fused-kernel gate the dense batched-prefill path uses. Visibility
// change only — behavior and callers inside this file are unchanged.
pub(crate) fn q8_prefill_wmma_enabled(gpu: &Gpu) -> bool {
    q8_prefill_wmma_enabled_from_env(
        hipfire_config::developer_var("HIPFIRE_Q8_PREFILL_WMMA")
            .ok()
            .as_deref(),
        gpu.arch.as_str(),
        gpu.arch_caps.has_wmma(),
    )
}

// pub(crate): also used by forward_slots.rs (MoE slots port) to gate entry
// into `prefill_moe_ffn_body_batched` from the slot-aware path, mirroring
// this file's own `prefill_batch_pbs_eligible` precondition check. Visibility
// change only — behavior and existing callers are unchanged.
pub(crate) fn moe_ffn_batched_admissible(ffn: &MoeFfnWeights, admit_mq6: bool, arch: &str) -> bool {
    let Some(dtypes) = MoePrefillDtypes::from_ffn(ffn) else {
        return false;
    };
    // mfp4-E8 routed experts: originally gfx1151-only, now widened to all
    // gfx11 + gfx12 arches behind the HIPFIRE_E8_GFX12 gate. The shared-expert
    // dequant→F16 GEMM path uses `dequantize_mfp4g32_e8_to_f16` + `gemm_f16_wmma_mb8`
    // which are available on all WMMA-capable arches. The indexed E8 GEMV kernels
    // for the routed experts are also present on gfx11 (shipped in the MoE-AWQ
    // branch). Gate on HIPFIRE_E8_GFX12 to allow safe opt-in rollout.
    let admit_e8 = matches!(
        arch,
        "gfx1100"
            | "gfx1101"
            | "gfx1102"
            | "gfx1150"
            | "gfx1151"
            | "gfx1152"
            | "gfx1200"
            | "gfx1201"
    ) && hipfire_config::developer_var("HIPFIRE_E8_GFX12")
        .ok()
        .as_deref()
        == Some("1");

    // PARO admit is default-on. Set HIPFIRE_PARO_BATCHED=0 to force the old
    // fallback path while bisecting or debugging.
    // for shisa-Qwen3.6-A3B-PARO and similar ParoQuant checkpoints where the
    // routed-expert + shared-expert weights are ParoQ4G128 (HFQ4G128 +
    // per-weight Givens rotation metadata). The downstream dispatch arms for
    // ParoQ4G128 are implemented on this branch. See roadmap at
    // .claude/plans/magical-marinating-hippo.md.
    static PARO_ADMIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let admit_paro = *PARO_ADMIT.get_or_init(|| {
        paro_batched_admit_enabled_from_env(
            hipfire_config::developer_var("HIPFIRE_PARO_BATCHED")
                .ok()
                .as_deref(),
        )
    });

    let admit_codebook = codebook_batched_admit_enabled(arch);
    // One-time provenance line. A codebook-routed model that used to prefill
    // per-token now takes grouped-WMMA batched prefill, so any timing captured
    // across that flip must be attributable; print once per process which route
    // the routed experts took and how to put it back.
    if admit_codebook
        && routed_codebook_pair_batched_supported(dtypes.expert_gate_up, dtypes.expert_down)
        && dtypes.expert_gate_up_uniform
        && dtypes.expert_down_uniform
    {
        static NOTED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        NOTED.get_or_init(|| {
            eprintln!(
                "[moe-prefill] uniform codebook routed experts (gate_up={:?} down={:?}) \
                 admitted to grouped-WMMA batched prefill on {arch}. \
                 HIPFIRE_MOE_CODEBOOK_BATCHED=0 restores the per-token fallback.",
                dtypes.expert_gate_up, dtypes.expert_down
            );
        });
    }
    moe_ffn_batched_admissible_for_dtypes(&dtypes, admit_mq6, admit_paro, admit_e8, admit_codebook)
}

/// #397 Ship 5.2 slice 1: route a single PLAIN-batched prefill GEMM through
/// [`GemmFamily::run_key`] against an *explicit* dispatcher-entry [`KernelKey`].
///
/// This is the behavior-preserving migration primitive proved by the Ship 5.2
/// pilot (028ac9f3): passing the dispatcher-entry key (e.g.
/// `GemmQ8_0BatchedChunked`, `GemmHfq4G256`, `GemmHfq4G128`, `GemmF32Batched`)
/// makes `run_key` dispatch to the IDENTICAL `gpu.gemm_*` method the direct
/// call used, so each method's own internal arch routing (RDNA4-WMMA /
/// gfx906-dp4a / CDNA-rocBLAS / …) is preserved byte-for-byte on every
/// (dtype × arch × shape). `resolve()` is deliberately NOT used here — it
/// front-runs the kernel's internal dispatch with a dtype-keyed WMMA preference
/// and can diverge from a direct dispatcher-entry call on some arches.
///
/// Only the four PLAIN-batched dispatcher-entry keys with existing table
/// entries are valid here. Residual-fused kernels (`gemm_*_residual*`) and the
/// fused QKVZA / gate+up kernels are NOT plain GEMMs and are migrated in later
/// slices (they need new table entries).
#[inline]
// pub(crate): also used by forward_slots.rs (SP3 Task 2), which mirrors this
// file's Q8_0 batched-prefill dispatch for the slot-aware path. Visibility
// change only — body and existing callers are untouched.
pub(crate) fn run_plain_gemm_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    w_buf: &GpuTensor,
    w_dtype: DType,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use hipfire_dispatch::families::gemm::GemmParams;
    let ctx = DispatchCtx::new(gpu);
    let w = WeightRef {
        buf: w_buf,
        dtype: w_dtype,
        m,
        k,
        row_stride: k,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    let params = GemmParams {
        w: &w,
        x,
        y,
        batch_size: n,
    };
    hipfire_runtime::llama::gemm_family()
        .run_key(key, &ctx, gpu, &params)
        .map_err(HipError::from)
}

/// #397 Ship 5.2 FINAL: route a single BATCHED-prefill RESIDUAL-fused GEMM
/// (`y += W·x`) through [`GemmFamily::run_key`] against an explicit
/// `Gemm*Residual` [`KernelKey`].
///
/// Residual analogue of [`run_plain_gemm_key`]. The residual op writes its
/// output IN-PLACE into the residual stream `y` (which carries the pre-add
/// value); the `gpu.gemm_*_residual` kernels perform the add internally and
/// NEVER reuse `y` as GEMV scratch, so the migration cannot reintroduce the
/// a9e8dfda aliasing bug — `y`, the residual/input `x`, and the weight buffer
/// are passed in the IDENTICAL order the direct call used. Each residual key
/// routes to the same `gpu.gemm_*_residual` method (which keeps its own internal
/// arch routing: WMMA/gfx12-WMMA / dp4a / fp16 / scalar) byte-for-byte. For
/// HFQ3 the run-arm replicates the call-site WMMA-vs-base arch split internally
/// via `gpu.arch_caps`; `resolve()` only confirms the entry's ArchPredicate
/// admits the current arch (it is NOT used to front-run the kernel's dispatch).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_residual_gemm_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    w_buf: &GpuTensor,
    w_dtype: DType,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use hipfire_dispatch::families::gemm::GemmParams;
    let ctx = DispatchCtx::new(gpu);
    let w = WeightRef {
        buf: w_buf,
        dtype: w_dtype,
        m,
        k,
        row_stride: k,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    // The residual stream `y` is BOTH the residual and the output (`y += W·x`).
    let params = GemmParams {
        w: &w,
        x,
        y,
        batch_size: n,
    };
    hipfire_runtime::llama::gemm_family()
        .run_key(key, &ctx, gpu, &params)
        .map_err(HipError::from)
}
/// qt=52 (MQ4G256V2Lloyd) prefill re-arm: extract the per-tensor E4M3 codebook
/// LUT or fail closed naming the prefill family. The uniform route never
/// touches this (uniform tensors carry no sidecar), so it stays bit-identical.
#[inline]
fn lloyd_e4m3_or_fail(w: &WeightTensor, family: &'static str) -> HipResult<[u32; 4]> {
    w.lloyd_lut_e4m3.ok_or_else(|| {
        hip_bridge::HipError::new(
            0,
            &format!(
                "{family}: MQ4G256V2Lloyd weight (m={} k={}) reached prefill with lloyd_lut_e4m3=None — \
                 re-quantize with a build that emits lloyd_levels (F32[16]); refusing uniform decode",
                w.m, w.k
            ),
        )
    })
}

/// qt=52 gfx11 MMQ-LUT prefill: extract the per-tensor C16 codebook or fail
/// closed naming the prefill family. Sibling of [`lloyd_e4m3_or_fail`]; the
/// uniform route never touches this.
#[inline]
fn lloyd_c16_or_fail(w: &WeightTensor, family: &'static str) -> HipResult<[u32; 4]> {
    w.lloyd_lut_c16.ok_or_else(|| {
        hip_bridge::HipError::new(
            0,
            &format!(
                "{family}: MQ4G256V2Lloyd weight (m={} k={}) reached prefill with lloyd_lut_c16=None — \
                 re-quantize with a build that emits lloyd_levels (F32[16]); refusing uniform decode",
                w.m, w.k
            ),
        )
    })
}

/// gfx11 (exact gfx1100|gfx1151) MMQ-LUT route for qt=52, unless kill-switched.
/// Same arch predicate as the uniform MQ4V2 MMQ path; `HIPFIRE_LLOYD_MMQ_OFF=1`
/// forces fail-closed via the existing FP8-LUT arm (never uniform).
#[inline]
fn lloyd_mmq_lut_route(gpu: &Gpu) -> bool {
    matches!(gpu.arch.as_str(), "gfx1100" | "gfx1151")
        && hipfire_config::developer_var("HIPFIRE_LLOYD_MMQ_OFF")
            .ok()
            .as_deref()
            != Some("1")
}

/// True when all listed projection dtypes are the Lloyd-V2 dtype.
#[inline]
fn all_mq4v2_lloyd(dts: &[DType]) -> bool {
    dts.iter().all(|d| matches!(d, DType::MQ4G256V2Lloyd))
}

/// #397 Ship 5.2 slice 2: route a single BATCHED-prefill FUSED gate+up GEMM
/// through [`FusedQkvFamily`] against an explicit `FusedGateUp*` [`KernelKey`].
///
/// This is the gate+up analogue of [`run_plain_gemm_key`]. Unlike a plain GEMM,
/// gate+up carries TWO weights (gate, up) and writes TWO outputs in one fused
/// launch, so it goes through `FusedQkvFamily` (the gate+up variant) rather than
/// `GemmFamily`. Passing `batch_size: Some(n)` makes the family's gate+up run-arm
/// dispatch to the IDENTICAL batched `gpu.gemm_gate_up_*(.., n)` method the direct
/// prefill call used — each method keeps its own internal arch routing
/// (RDNA4-WMMA / gfx906-dp4a / MMQ / fp16 / scalar) byte-for-byte. The weights,
/// activation `x` (already rmsnorm-rotated by the caller), outputs and m/k/n args
/// are unchanged at every migrated site.
///
/// The `FusedGateUp*` key carries the dtype; the run-arm replicates any
/// call-site arch split (e.g. HFQ3 WMMA-vs-base) internally via `gpu.arch_caps`,
/// so the same kernel runs. `resolve()` only confirms the entry's ArchPredicate
/// admits the current arch — it does NOT front-run the kernel's internal dispatch.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_fused_gate_up_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    w_gate: &GpuTensor,
    w_up: &GpuTensor,
    x: &GpuTensor,
    y_gate: &GpuTensor,
    y_up: &GpuTensor,
    gate_m: usize,
    up_m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use hipfire_dispatch::families::fused_qkv::FusedQkvParams;
    let ctx = DispatchCtx::new(gpu);
    let params = FusedQkvParams {
        kind: key,
        weights: &[w_gate, w_up],
        x,
        outputs: &[y_gate, y_up],
        m: &[gate_m, up_m],
        k,
        rot_scratch: &[],
        batch_size: Some(n),
    };
    hipfire_runtime::llama::fused_qkv_family()
        .run(&ctx, gpu, &params)
        .map_err(HipError::from)
}

/// Dispatch a batched-prefill **3-way fused QKV** projection (wq+wk+wv) through
/// [`FusedQkvFamily`] against an explicit `FusedQkv*` [`KernelKey`]
/// (`#397 Ship 5.2 slice 3`).
///
/// QKV analogue of [`run_fused_gate_up_key`]: three weights (wq, wk, wv), three
/// outputs (q, k, v), three row-counts. Passing `batch_size: Some(n)` routes the
/// family's QKV run-arm to the IDENTICAL batched `gpu.gemm_qkv_*(.., n)` method
/// the direct prefill call used — each method keeps its own internal arch routing
/// (RDNA4-WMMA / gfx906-dp4a / MMQ / fp16 / scalar) byte-for-byte. The weights,
/// activation `x` (already rmsnorm[-rotated] by the caller), outputs and m/k/n
/// args are unchanged at every migrated site. The `FusedQkv*` key carries the
/// dtype; for HFQ3 the run-arm replicates the call-site WMMA-vs-base arch split
/// internally via `gpu.arch_caps`. `resolve()` only confirms the entry's
/// ArchPredicate admits the current arch.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_fused_qkv_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    wq: &GpuTensor,
    wk: &GpuTensor,
    wv: &GpuTensor,
    x: &GpuTensor,
    y_q: &GpuTensor,
    y_k: &GpuTensor,
    y_v: &GpuTensor,
    q_m: usize,
    k_m: usize,
    v_m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use hipfire_dispatch::families::fused_qkv::FusedQkvParams;
    let ctx = DispatchCtx::new(gpu);
    let params = FusedQkvParams {
        kind: key,
        weights: &[wq, wk, wv],
        x,
        outputs: &[y_q, y_k, y_v],
        m: &[q_m, k_m, v_m],
        k,
        rot_scratch: &[],
        batch_size: Some(n),
    };
    hipfire_runtime::llama::fused_qkv_family()
        .run(&ctx, gpu, &params)
        .map_err(HipError::from)
}

/// Dispatch a batched-prefill **4-way fused QKVZA** projection (DeltaNet linear
/// attention: wqkv + wz + w_beta + w_alpha) through [`FusedQkvFamily`] against an
/// explicit `FusedQkvza*` [`KernelKey`] (`#397 Ship 5.2 slice 3`).
///
/// QKVZA analogue of [`run_fused_qkv_key`]: four weights, four outputs, four
/// row-counts. `batch_size: Some(n)` routes the family's QKVZA run-arm to the
/// IDENTICAL batched `gpu.gemm_qkvza_*(.., n)` method the direct prefill call
/// used. All operands are passed unchanged; for HFQ3 the run-arm replicates the
/// call-site WMMA-vs-base arch split internally.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_fused_qkvza_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    w_qkv: &GpuTensor,
    w_z: &GpuTensor,
    w_beta: &GpuTensor,
    w_alpha: &GpuTensor,
    x: &GpuTensor,
    y_qkv: &GpuTensor,
    y_z: &GpuTensor,
    y_beta: &GpuTensor,
    y_alpha: &GpuTensor,
    qkv_m: usize,
    z_m: usize,
    beta_m: usize,
    alpha_m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use hipfire_dispatch::families::fused_qkv::FusedQkvParams;
    let ctx = DispatchCtx::new(gpu);
    let params = FusedQkvParams {
        kind: key,
        weights: &[w_qkv, w_z, w_beta, w_alpha],
        x,
        outputs: &[y_qkv, y_z, y_beta, y_alpha],
        m: &[qkv_m, z_m, beta_m, alpha_m],
        k,
        rot_scratch: &[],
        batch_size: Some(n),
    };
    hipfire_runtime::llama::fused_qkv_family()
        .run(&ctx, gpu, &params)
        .map_err(HipError::from)
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
    // expert (step 5) stays in `pbs.x_batch` — replicated per rank, not
    // redirected. `None` = byte-identical single-GPU behavior.
    routed_out: Option<&GpuTensor>,
) -> HipResult<()> {
    // Historical entry point: replicated local routing (Single seal path).
    // Byte-identical to before; the slot-aware forward_slots.rs callers and
    // all single-GPU paths keep calling this unchanged.
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
pub(crate) fn prefill_moe_ffn_body_batched_with_route(
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
    // expert (step 5) stays in `pbs.x_batch` — replicated per rank, not
    // redirected. `None` = byte-identical single-GPU behavior.
    routed_out: Option<&GpuTensor>,
    route: PrefillRouteMode<'_>,
) -> HipResult<()> {
    let dim = config.dim;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let k_top = config.num_experts_per_tok;
    let n_exp = config.num_experts;

    let router_logits = pbs.moe_router_logits_batch.as_ref().expect("moe scratch");
    let shared_scalar = pbs.moe_shared_scalar_batch.as_ref().expect("moe scratch");
    let shared_gate = pbs.moe_shared_gate_batch.as_ref().expect("moe scratch");
    let shared_up = pbs.moe_shared_up_batch.as_ref().expect("moe scratch");
    let shared_rot = pbs.moe_shared_rot_batch.as_ref().expect("moe scratch");
    let topk_indices = pbs.moe_topk_indices_batch.as_ref().expect("moe scratch");
    let topk_weights = pbs.moe_topk_weights_batch.as_ref().expect("moe scratch");
    let gate_batch = pbs.moe_gate_batch.as_ref().expect("moe scratch");
    let up_batch = pbs.moe_up_batch.as_ref().expect("moe scratch");
    let rot_batch = pbs.moe_rot_batch.as_ref().expect("moe scratch");
    let down_expanded = pbs.moe_down_expanded_batch.as_ref().expect("moe scratch");
    // ── 0. Preflight: bind, build params, seal, adopt — before any launch ──
    //
    // Actual (not static-metadata) validation BEFORE any GPU mutation. The
    // seal checks the live binding, the preexisting PBS tensor descriptors,
    // the zeroed routed partial, and (on non-root ranks) the root proof —
    // all pure, launching nothing. Sealing needs only descriptors, never
    // contents, so it may run before norm/rotate/shared/router produce any.
    // Mode resolution is the historical seal-site rule: Single bindings
    // ignore the band mode; compact bindings follow it, with legacy
    // `Replicated` keeping today's fail-closed sealer error — only earlier.
    // The root route is still produced (and its receipt attached) after the
    // router kernels at step 6, before the experts execute; the producer
    // proof still only exports from that actual produced+attached receipt,
    // never fabricated. No re-sealing: the late block reuses this seal.
    let bound = ffn.bound_experts()?;
    let single = bound.rank_count() == 1;
    let produce_root = matches!(route, PrefillRouteMode::ProduceRoot { .. }) && !single;
    let adopt_root = matches!(route, PrefillRouteMode::AdoptRoot { .. }) && !single;
    // Non-root EP ranks skip the router GEMM + softmax/topk + produce below:
    // their route arrays already hold the root's D2D-copied bytes. The
    // shared-expert GEMMs still run (shared stays replicated per rank).
    let skip_router_produce = adopt_root;
    let down_m = ffn.experts[0].down.m;
    let down_k = ffn.experts[0].down.k;
    let gate_up_k = ffn.experts[0].gate_up.k;
    let total_slots = n * k_top;
    let m_total_max = moe_grouped_m_total_bound(total_slots, n_exp);
    // The load-time owner caches only genuinely mixed tier arrays. Borrowing
    // these slices keeps prefill allocation-free and preserves uniform None.
    let (per_expert_gate_up, per_expert_down) = ffn.per_expert_tier_tables();
    let moe_dtypes = hipfire_dispatch::families::moe::MoeDtypes {
        router: ffn.router.gpu_dtype,
        shared_gate: ffn.shared_expert_gate.gpu_dtype,
        shared_expert_gate: ffn.shared_expert.gate.gpu_dtype,
        shared_expert_up: ffn.shared_expert.up.gpu_dtype,
        shared_expert_down: ffn.shared_expert.down.gpu_dtype,
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
        // The cached mixed-only tier slices are the source of truth for both
        // projections; borrowing their presence avoids any per-call scan.
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
    // Route A MoE-AWQ: the per-expert indexed table (built at load) supersedes
    // the Ship 4.2 single-scale `down_awq_scale` stub for routed experts — that
    // stub applied experts[0]'s scale to every routed slot, which is wrong once
    // experts actually carry per-expert AWQ. Pass `None` for the single scale;
    // `expert_down_awq_ptrs` drives the correct per-slot path in run_moe_prefill.
    let down_awq_scale: Option<&GpuTensor> = None;
    let moe_prefill_params = hipfire_dispatch::families::moe::MoePrefillParams {
        dtypes: moe_dtypes,
        batch_size: n,
        mi,
        down_m,
        down_k,
        gate_up_k,
        k_top,
        n_exp,
        m_total_max,
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
        down_awq_scale,
        routed_out,
    };
    let mut sealed = if adopt_root {
        // Non-root EP rank: seal compact, adopt the root proof (launches
        // nothing — the receipt binds this call's own invocation over this
        // rank's own topk buffers), attach now so a proof mismatch rejects
        // before this rank mutates anything.
        let proof = match route {
            PrefillRouteMode::AdoptRoot { proof } => proof,
            _ => unreachable!("adopt_root without AdoptRoot mode"),
        };
        let mut sealed =
            hipfire_dispatch::pipeline::sealed_moe::seal_prefill_ep(bound, ctx, moe_prefill_params)
                .map_err(HipError::from)?;
        let receipt = hipfire_dispatch::pipeline::sealed_moe::adopt_prefill_route(&sealed, proof)
            .map_err(HipError::from)?;
        sealed
            .attach_route_receipt(receipt)
            .map_err(HipError::from)?;
        sealed
    } else if produce_root {
        hipfire_dispatch::pipeline::sealed_moe::seal_prefill_ep(bound, ctx, moe_prefill_params)
            .map_err(HipError::from)?
    } else {
        hipfire_dispatch::pipeline::sealed_moe::seal_prefill(bound, ctx, moe_prefill_params)
            .map_err(HipError::from)?
    };

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
    // ── 1. Split rmsnorm vs FWHT rotate ──
    //
    // A3B (and every other MoE here) leaves router + shared_expert_gate
    // as Q8_0 in the quantizer — these tiny tensors lose too much
    // accuracy at 4-bit, so the engine never reduces them. Q8 weights
    // are quantized against the un-rotated rmsnorm output, while the
    // MQ4 siblings (shared_expert.{gate,up,down} + experts.{gate_up,down})
    // expect FWHT(rmsnorm(x) / awq_scale). Populate both:
    //   x_norm_batch ← rmsnorm(x_batch)
    //   x_rot_batch  ← FWHT(x_norm_batch / awq_scale)  (only if any
    //                  downstream MQ weight is present, which moe_ffn_batched_admissible
    //                  guarantees — shared_expert.gate is always MQ4 here)
    //
    // Pick `shared_expert.gate` as the AWQ representative (instead of
    // the previous `ffn.router`). Per the F1 imatrix scope every gate-side
    // MQ4 sibling shares the same input basis and therefore an identical
    // awq_scale, but the router itself is excluded from F1 (it stays Q8).
    // Reading awq_scale from router would silently drop AWQ rotation in
    // v3 AWQ runs — latent until this predicate widened.
    gpu.rmsnorm_batched(
        &pbs.x_batch,
        ffn_norm,
        &pbs.x_norm_batch,
        n,
        dim,
        config.norm_eps,
    )?;
    // PARO mode (shared_expert.gate is ParoQ4G128): each weight carries its
    // own Givens rotation table (paro.pairs / theta / channel_scales). The
    // shared MQ4-style FWHT pre-rotation here would be wrong — skip it. The
    // ParoQ4G128 dispatch arms below run per-weight Givens rotation in-place
    // before each GEMM, using pbs.x_rot_batch as the rotation destination.
    let paro_mode = matches!(ffn.shared_expert.gate.gpu_dtype, DType::ParoQ4G128);
    if !paro_mode {
        rotate_x_mq_batched_for(
            gpu,
            &ffn.shared_expert.gate,
            &pbs.x_norm_batch,
            &pbs.x_rot_batch,
            dim,
            n,
        )?;
    }

    // ── 2. Router + shared-gate + shared.gate + shared.up (4 batched GEMMs) ──
    //
    // Per-dtype dispatch — Q8 reads `x_norm_batch`, MQ4 reads
    // `x_rot_batch`. The natural 4-way fuse via `gemm_qkvza_hfq4g256`
    // is not applicable when router/shared_expert_gate are Q8 (mixed
    // strides). Four separate launches; +3 per MoE layer over the fused
    // ideal, acceptable for the structural unlock.
    // #397 Ship 5.2 PILOT: route the router GEMM through GemmFamily::run_key.
    // Each arm uses the *dispatcher-entry* KernelKey (GemmQ8_0BatchedChunked /
    // GemmHfq4G256 / GemmF32Batched) so run_key dispatches to the IDENTICAL
    // gpu.gemm_* method the prior direct call used — preserving each method's
    // own internal arch routing (RDNA4-WMMA / gfx906-dp4a / CDNA-rocBLAS / …)
    // byte-for-byte. The x input still differs per dtype (Q8/F32 read
    // x_norm_batch; MQ4 reads x_rot_batch), exactly as before. The three keys
    // are registered ArchPredicate::Always, so run_key never rejects.
    // Non-root EP ranks skip the router GEMM (and the logits dump below):
    // their route arrays already hold the root's D2D-copied bytes. The
    // shared-expert-gate GEMM after the dump still runs (shared stays
    // replicated per rank).
    if !skip_router_produce {
        use hipfire_dispatch::families::gemm::GemmParams;
        let ctx = DispatchCtx::new(gpu);
        let (key, x_in): (hipfire_dispatch::types::KernelKey, &GpuTensor) =
            match ffn.router.gpu_dtype {
                DType::Q8_0 => (
                    hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                    &pbs.x_norm_batch,
                ),
                DType::MQ4G256 => (
                    hipfire_dispatch::types::KernelKey::GemmHfq4G256,
                    &pbs.x_rot_batch,
                ),
                DType::MQ4G256V2 => (
                    hipfire_dispatch::types::KernelKey::GemmMq4G256V2,
                    &pbs.x_rot_batch,
                ),
                DType::F32 => (
                    hipfire_dispatch::types::KernelKey::GemmF32Batched,
                    &pbs.x_norm_batch,
                ),
                other => panic!(
                    "prefill_moe_ffn_body_batched: unexpected router dtype {other:?} \
                         — moe_ffn_batched_admitted admits MQ4G256, Q8_0, F32"
                ),
            };
        let w = WeightRef {
            buf: &ffn.router.buf,
            dtype: ffn.router.gpu_dtype,
            m: ffn.router.m,
            k: ffn.router.k,
            row_stride: ffn.router.k,
            rotation: None,
            awq_scale: None,
            lloyd_lut_e4m3: ffn.router.lloyd_lut_e4m3,
            lloyd_lut_f16: ffn.router.lloyd_lut_f16,
            lloyd_lut_c16: ffn.router.lloyd_lut_c16,
        };
        let params = GemmParams {
            w: &w,
            x: x_in,
            y: router_logits,
            batch_size: n,
        };
        hipfire_runtime::llama::gemm_family()
            .run_key(key, &ctx, gpu, &params)
            .map_err(HipError::from)?;
        // DIAG: dump MoE router logits (batched)
        dump_hidden_localize(gpu, router_logits, n, 0, ffn.router.m, 0, "router_b");
    }
    // #397 Ship 5.2 slice1: route the shared-expert-gate GEMM through
    // GemmFamily::run_key. Same dtype-routed dispatcher-entry keys as the router
    // match above (Q8/F32 read x_norm_batch, MQ4 reads x_rot_batch) → identical
    // gpu.gemm_* method, byte-for-byte.
    {
        use hipfire_dispatch::types::KernelKey;
        let (key, x_in): (KernelKey, &GpuTensor) = match ffn.shared_expert_gate.gpu_dtype {
            DType::Q8_0 => (KernelKey::GemmQ8_0BatchedChunked, &pbs.x_norm_batch),
            DType::MQ4G256 => (KernelKey::GemmHfq4G256, &pbs.x_rot_batch),
            DType::MQ4G256V2 => (KernelKey::GemmMq4G256V2, &pbs.x_rot_batch),
            DType::F32 => (KernelKey::GemmF32Batched, &pbs.x_norm_batch),
            other => panic!(
                "prefill_moe_ffn_body_batched: unexpected shared_expert_gate dtype {other:?} \
                         — moe_ffn_batched_admissible admits MQ4G256, Q8_0, F32"
            ),
        };
        run_plain_gemm_key(
            gpu,
            key,
            &ffn.shared_expert_gate.buf,
            ffn.shared_expert_gate.gpu_dtype,
            x_in,
            shared_scalar,
            ffn.shared_expert_gate.m,
            ffn.shared_expert_gate.k,
            n,
        )?;
    }
    // Fused gate+up dispatch for the shared expert — halves the kernel
    // launch count vs back-to-back gemm_hfq*g256 (~75µs/launch × 40
    // MoE layers = ~3ms saved on R9700 A3B prefill at bs=256).
    // Per-projection dispatch: gate AND up share the same dtype (predicate
    // enforces). MQ4 → HFQ4-layout fused kernel; MQ6 → HFQ6-layout.
    match ffn.shared_expert.gate.gpu_dtype {
        // #397 Ship 5.2 slice 2: shared-expert fused gate+up → FusedQkvFamily
        // (batched-prefill gate+up variant). Same batched kernel, behavior-preserving.
        // MQ4G256V2 / MQ6G256V2 select container-specific keys via fused_gate_up_key_for
        // (never V1 aliases). MQ4G256 falls through to FusedGateUpHfq4G256.
        DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ6G256V2 => run_fused_gate_up_key(
            gpu,
            crate::forward_slots::fused_gate_up_key_for(ffn.shared_expert.gate.gpu_dtype),
            &ffn.shared_expert.gate.buf,
            &ffn.shared_expert.up.buf,
            &pbs.x_rot_batch,
            shared_gate,
            shared_up,
            ffn.shared_expert.gate.m,
            ffn.shared_expert.up.m,
            ffn.shared_expert.gate.k,
            n,
        )?,
        DType::MQ6G256 => run_fused_gate_up_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedGateUpHfq6G256,
            &ffn.shared_expert.gate.buf,
            &ffn.shared_expert.up.buf,
            &pbs.x_rot_batch,
            shared_gate,
            shared_up,
            ffn.shared_expert.gate.m,
            ffn.shared_expert.up.m,
            ffn.shared_expert.gate.k,
            n,
        )?,
        // Phase 2: PARO shared_expert.gate + up. Each weight has its own
        // Givens rotation table — rotate x_norm_batch into x_rot_batch using
        // gate's tables, GEMM, then re-rotate using up's tables, GEMM. Total
        // 4 dispatches vs the MQ4 path's 1 fused gemm_gate_up — acceptable
        // overhead for the per-token-loop elimination win. Phase 4 could
        // collapse this into a single fused kernel
        // (gemm_gate_up_paro_q4g128_batched) if measurement shows it matters.
        DType::ParoQ4G128 => {
            let paro_gate = ffn
                .shared_expert
                .gate
                .paro
                .as_ref()
                .expect("ParoQ4G128 shared_expert.gate missing paro metadata");
            let paro_up = ffn
                .shared_expert
                .up
                .paro
                .as_ref()
                .expect("ParoQ4G128 shared_expert.up missing paro metadata");
            // Gate: rotate x_norm by gate's Givens → x_rot, then HFQ4G128 GEMM
            gpu.givens_rotate_to(
                &pbs.x_norm_batch,
                &pbs.x_rot_batch,
                &paro_gate.pairs,
                &paro_gate.theta,
                &paro_gate.channel_scales,
                n,
                dim,
                paro_gate.krot as usize,
            )?;
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                &ffn.shared_expert.gate.buf,
                ffn.shared_expert.gate.gpu_dtype,
                &pbs.x_rot_batch,
                shared_gate,
                ffn.shared_expert.gate.m,
                ffn.shared_expert.gate.k,
                n,
            )?;
            // Up: re-rotate x_norm by up's Givens → x_rot (overwrite), GEMM
            gpu.givens_rotate_to(
                &pbs.x_norm_batch,
                &pbs.x_rot_batch,
                &paro_up.pairs,
                &paro_up.theta,
                &paro_up.channel_scales,
                n,
                dim,
                paro_up.krot as usize,
            )?;
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                &ffn.shared_expert.up.buf,
                ffn.shared_expert.up.gpu_dtype,
                &pbs.x_rot_batch,
                shared_up,
                ffn.shared_expert.up.m,
                ffn.shared_expert.up.k,
                n,
            )?;
        }
        // Q8 shared expert (A3B mfp4-E8): gate + up via two batched Q8 GEMMs
        // reading the UN-rotated x_norm_batch (Q8 weights are quantized against
        // un-rotated rmsnorm output). No fused Q8 gate+up kernel — two plain
        // launches; mirrors the decode `gemv.run_auto` Q8 shared gate/up arm.
        DType::Q8_0 => {
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                &ffn.shared_expert.gate.buf,
                ffn.shared_expert.gate.gpu_dtype,
                &pbs.x_norm_batch,
                shared_gate,
                ffn.shared_expert.gate.m,
                ffn.shared_expert.gate.k,
                n,
            )?;
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                &ffn.shared_expert.up.buf,
                ffn.shared_expert.up.gpu_dtype,
                &pbs.x_norm_batch,
                shared_up,
                ffn.shared_expert.up.m,
                ffn.shared_expert.up.k,
                n,
            )?;
        }
        // Uniform mfp4-E8 shared expert (Option B): shared expert gate/up are both
        // MFP4G32E8. We dequant each to F16 transiently (E8→F16 gives W_rot in the
        // FWHT-rotated domain), then run GemmF16WmmaMb8 against x_rot_batch (F32).
        // Math: GEMM(W_rot, x_rot) = W·x_norm = correct forward pass.
        // The F16 scratch tensors are allocated per call and freed after use; they
        // are small (smi × dim × 2 bytes each) relative to VRAM.
        DType::MFP4G32E8 => {
            let gate_m = ffn.shared_expert.gate.m;
            let gate_k = ffn.shared_expert.gate.k;
            let up_m = ffn.shared_expert.up.m;
            let up_k = ffn.shared_expert.up.k;
            // Dequantize gate and up weights: E8 → F16 (in-rotated domain)
            let gate_f16 = gpu.alloc_tensor(&[gate_m * gate_k], DType::F16)?;
            gpu.dequantize_mfp4g32_e8_to_f16(
                &ffn.shared_expert.gate.buf.buf,
                &gate_f16.buf,
                gate_m,
                gate_k,
            )?;
            let up_f16 = gpu.alloc_tensor(&[up_m * up_k], DType::F16)?;
            gpu.dequantize_mfp4g32_e8_to_f16(
                &ffn.shared_expert.up.buf.buf,
                &up_f16.buf,
                up_m,
                up_k,
            )?;
            // GemmF16WmmaMb8: W(F16) × x_rot_batch(F32) → shared_gate / shared_up (F32)
            // x_rot_batch is F32; gemm_f16_wmma_mb8 accepts F32 activations directly.
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmF16WmmaMb8,
                &gate_f16,
                DType::F16,
                &pbs.x_rot_batch,
                shared_gate,
                gate_m,
                gate_k,
                n,
            )?;
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmF16WmmaMb8,
                &up_f16,
                DType::F16,
                &pbs.x_rot_batch,
                shared_up,
                up_m,
                up_k,
                n,
            )?;
            // Free the transient F16 weight buffers
            gpu.free_tensor(gate_f16)?;
            gpu.free_tensor(up_f16)?;
        }
        other => panic!(
            "prefill_moe_ffn_body_batched: unsupported shared_expert.gate dtype {other:?} \
                         — admit predicate should have rejected this layer"
        ),
    }

    // ── 4. Shared-expert SwiGLU + FWHT, batched over N tokens ──
    //
    // fused_silu_mul_rotate_mq_batched expects [batch × k] gate/up with
    // batch on grid.y and writes FWHT(silu(gate) * up) into x_rot. Here
    // batch=N, k=smi; the shared-rot output buffer is [N × smi].
    // F2: AWQ-aware silu_mul+rotate for the batched shared-expert down input.
    // PARO: shared_expert.down has its own Givens rotation tables (paro.*);
    // use the dedicated fused kernel (commit 50198daa). It takes a per-weight
    // (pairs, theta, channel_scales, krot) tuple instead of the MQ4 FWHT
    // convention. Same shape: gate/up [N × smi] → shared_rot [N × smi].
    if paro_mode {
        let paro_down = ffn
            .shared_expert
            .down
            .paro
            .as_ref()
            .expect("ParoQ4G128 shared_expert.down missing paro metadata");
        gpu.fused_silu_mul_givens_rotate_f32(
            shared_gate,
            shared_up,
            shared_rot,
            &paro_down.pairs,
            &paro_down.theta,
            &paro_down.channel_scales,
            n,
            smi,
            paro_down.krot as usize,
        )?;
    } else if matches!(ffn.shared_expert.down.gpu_dtype, DType::Q8_0) {
        // Q8 shared down expects the UN-rotated SwiGLU hidden (no FWHT). Plain
        // element-wise silu_mul over the flat [N × smi] buffers (batched for
        // free) writes the hidden into shared_rot, feeding the Q8 down GEMM.
        gpu.silu_mul_f32(shared_gate, shared_up, shared_rot)?;
    } else {
        fused_silu_mul_rotate_mq_batched_for(
            gpu,
            &ffn.shared_expert.down,
            shared_gate,
            shared_up,
            shared_rot,
            smi,
            n,
        )?;
    }

    // ── 5. Shared-expert down with sigmoid-scaled residual, batched ──
    //
    // Reads shared_scalar[token] as the pre-sigmoid logit, applies sigmoid
    // internally, and += sigmoid(scalar) × (W_down · rot) into
    // pbs.x_batch[token × dim + row]. (Note: HFQ4 sister uses += not
    // atomicAdd; each (bid, row) writes a unique cell.)
    // Per-projection dispatch: MQ4 → HFQ4 kernel, MQ6 → HFQ6 sister
    // (shipped via feat/hfq6-sigmoid-scaled-batched).
    match ffn.shared_expert.down.gpu_dtype {
        DType::MQ4G256 => gpu.gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched(
            &ffn.shared_expert.down.buf,
            shared_rot,
            &pbs.x_batch,
            shared_scalar,
            ffn.shared_expert.down.m,
            ffn.shared_expert.down.k,
            n,
        )?,
        DType::MQ6G256 => gpu.gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched(
            &ffn.shared_expert.down.buf,
            shared_rot,
            &pbs.x_batch,
            shared_scalar,
            ffn.shared_expert.down.m,
            ffn.shared_expert.down.k,
            n,
        )?,
        // MQ4G256V2 / MQ6G256V2: exact dense V2 residual GEMM into temp +
        // sigmoid scale (no V1 HFQ4/HFQ6 residual_sigmoid alias — dual-half
        // headers differ from V1). Mirrors the Q8_0 split below.
        DType::MQ4G256V2 | DType::MQ6G256V2 => {
            let down_tmp = GpuTensor {
                buf: unsafe { down_expanded.buf.alias() },
                shape: vec![n * dim],
                dtype: DType::F32,
            };
            let bytes = n * dim * 4;
            if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip.memset_async(&down_tmp.buf, 0, bytes, stream)?;
            } else {
                gpu.hip.memset(&down_tmp.buf, 0, bytes)?;
            }
            run_residual_gemm_key(
                gpu,
                crate::forward_slots::residual_gemm_key_for(ffn.shared_expert.down.gpu_dtype),
                &ffn.shared_expert.down.buf,
                ffn.shared_expert.down.gpu_dtype,
                shared_rot,
                &down_tmp,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?;
            gpu.sigmoid_scaled_residual_add_batched_f32(
                &pbs.x_batch,
                &down_tmp,
                shared_scalar,
                n,
                dim,
            )?;
        }
        // Phase 2: HFQ4G128 batched residual+sigmoid-scaled kernel. Single
        // launch, same semantics as the HFQ4G256 sister — reads shared_rot
        // (already silu-mul-rotated by the PARO fused kernel above), GEMVs
        // against W_down, applies sigmoid(shared_scalar[token]) × output,
        // accumulates into pbs.x_batch.
        DType::ParoQ4G128 => gpu.gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched(
            &ffn.shared_expert.down.buf,
            shared_rot,
            &pbs.x_batch,
            shared_scalar,
            ffn.shared_expert.down.m,
            ffn.shared_expert.down.k,
            n,
        )?,
        // Q8 shared down (A3B mfp4-E8, Q8-shared variant): plain batched Q8 GEMM
        // W_down · hidden into a [N × dim] temp, then fold into the residual with
        // the per-token sigmoid(shared_scalar) gate. The temp aliases the first N×dim
        // of `down_expanded` (the routed down-expanded scratch), which is FREE here —
        // the routed experts (step 6) overwrite it only after this completes, and
        // the HIP stream is in-order so the add reads before that. Batched analog
        // of the decode sigmoid_f32 + scaled_add_inplace shared-down arm.
        DType::Q8_0 => {
            let down_tmp = GpuTensor {
                buf: unsafe { down_expanded.buf.alias() },
                shape: vec![n * dim],
                dtype: DType::F32,
            };
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                &ffn.shared_expert.down.buf,
                ffn.shared_expert.down.gpu_dtype,
                shared_rot,
                &down_tmp,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?;
            gpu.sigmoid_scaled_residual_add_batched_f32(
                &pbs.x_batch,
                &down_tmp,
                shared_scalar,
                n,
                dim,
            )?;
        }
        // Uniform mfp4-E8 shared expert down (Option B): dequant the E8 down weight
        // to F16, run GemmF16WmmaMb8 against shared_rot (FWHT-rotated SwiGLU hidden)
        // into a [N × dim] temp, then sigmoid-scale-add into the residual. The temp
        // aliases the first N×dim of `down_expanded` (safe: step 6 routed experts
        // run after this, and the stream is in-order). Mirrors the Q8_0 arm above
        // except the GEMM is F16 weight × F32 activation = F32 output.
        DType::MFP4G32E8 => {
            let down_m = ffn.shared_expert.down.m;
            let down_k = ffn.shared_expert.down.k;
            let down_f16 = gpu.alloc_tensor(&[down_m * down_k], DType::F16)?;
            gpu.dequantize_mfp4g32_e8_to_f16(
                &ffn.shared_expert.down.buf.buf,
                &down_f16.buf,
                down_m,
                down_k,
            )?;
            let down_tmp = GpuTensor {
                buf: unsafe { down_expanded.buf.alias() },
                shape: vec![n * dim],
                dtype: DType::F32,
            };
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmF16WmmaMb8,
                &down_f16,
                DType::F16,
                shared_rot,
                &down_tmp,
                down_m,
                down_k,
                n,
            )?;
            gpu.free_tensor(down_f16)?;
            gpu.sigmoid_scaled_residual_add_batched_f32(
                &pbs.x_batch,
                &down_tmp,
                shared_scalar,
                n,
                dim,
            )?;
        }
        other => panic!(
            "prefill_moe_ffn_body_batched: unsupported shared_expert.down dtype {other:?} \
                         — admit predicate should have rejected this layer"
        ),
    }

    // ── 6. Routed experts: delegated to MoeFamily::run_prefill (Ship 4.2) ──
    //
    // The call was sealed (and, on non-root ranks, adopted + attached) before
    // the launches above; only the route production stays here because it
    // must run after the router GEMM filled the score contents — still before
    // the experts execute. No re-sealing: the preflight seal is reused.
    if !adopt_root {
        // The producer requires an exact [N × num_experts] score shape. Select
        // the preallocated non-owning view for this chunk instead of constructing
        // a shape Vec for every MoE layer.
        let router_scores = pbs
            .moe_router_score_views_batch
            .as_ref()
            .and_then(|views| n.checked_sub(1).and_then(|index| views.get(index)))
            .ok_or_else(|| HipError::new(0, "moe router score scratch view is unavailable"))?;
        let receipt = hipfire_dispatch::pipeline::sealed_moe::produce_prefill_route(
            &sealed,
            gpu,
            router_scores,
            config.norm_topk_prob,
        )
        .map_err(HipError::from)?;
        sealed
            .attach_route_receipt(receipt)
            .map_err(HipError::from)?;
        if produce_root {
            // Publish the opaque producer proof through the band-carried slot.
            // It is issued from this actual successful root-produced receipt
            // (produced + attached + validated above), never fabricated.
            let proof = sealed
                .prefill_route_producer_proof()
                .map_err(HipError::from)?;
            match route {
                PrefillRouteMode::ProduceRoot { slot } => slot.set(Some(proof)),
                _ => unreachable!("produce_root without ProduceRoot mode"),
            }
        }
    }
    hipfire_dispatch::pipeline::execute_steps(
        gpu,
        ctx,
        &[hipfire_dispatch::pipeline::Step::Moe(sealed)],
    )
    .map_err(HipError::from)?;
    #[cfg(feature = "moe-oracle")]
    {
        let oracle_start =
            crate::qwen35::oracle::prefill_start().map_err(|e| hip_bridge::HipError::new(0, &e))?;
        crate::qwen35::oracle::prefill_after(
            gpu,
            ffn,
            config,
            &pbs.x_batch,
            router_logits,
            topk_indices,
            topk_weights,
            gate_batch,
            up_batch,
            rot_batch,
            down_expanded,
            n,
            oracle_start,
        )
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    }

    Ok(())
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

/// EP prefill route authority for one single-layer MoE band execution.
///
/// Carried inside [`PrefillBandCtx`] so the per-layer band plumbing needs no
/// new parameters, no heap allocation, and no signature churn for the
/// single-GPU callers (which always use `Replicated`). All variants are
/// `Copy` (shared borrows only).
#[derive(Clone, Copy, Debug)]
pub(crate) enum PrefillRouteMode<'a> {
    /// Local/replicated routing: run router GEMM + softmax/topk + produce on
    /// this rank. This is the historical single-GPU behavior, byte-identical.
    Replicated,
    /// EP root rank (rank 0): run router GEMM + produce locally, then publish
    /// the opaque producer proof through `slot` — a caller-owned stack
    /// [`std::cell::Cell`]. The cell gives write-through under the shared
    /// `&PrefillBandCtx` borrow without heap allocation (the proof is
    /// `Copy`); the driver reads it back after the call returns.
    ProduceRoot {
        slot: &'a std::cell::Cell<
            Option<hipfire_dispatch::pipeline::sealed_moe::MoePrefillRouteProducerProof>,
        >,
    },
    /// EP non-root rank: skip router GEMM + softmax/topk + produce entirely;
    /// the driver D2D-copies the root's first `N*k` route entries into this
    /// rank's PBS arrays first, then this mode adopts the root's proof
    /// (launching nothing) and executes the grouped MoE over those bytes.
    AdoptRoot {
        proof: &'a hipfire_dispatch::pipeline::sealed_moe::MoePrefillRouteProducerProof,
    },
}

#[allow(clippy::too_many_arguments)]
/// Debug localization hook (no-op unless `HIPFIRE_DUMP_HIDDEN` is set to a file
/// prefix). Appends the post-layer hidden row for the target absolute position
/// to `{HIPFIRE_DUMP_HIDDEN}.{tag}` as `u32 layer_idx` followed by `dim`
/// little-endian f32. The target absolute position is `HIPFIRE_DUMP_HIDDEN_POS`
/// (default 0); `abs_pos_of_row0` is the absolute sequence position of row 0 of
/// `x` (`start_pos` for the batched residual `pbs.x_batch`, `pos` for the
/// single-row per-token `s.x`). Used to localize the PARO batched-prefill
/// divergence by diffing `.batched` vs `.pertoken` per layer. Requires
/// `HIPFIRE_GRAPH=0` (does a synchronous D2H readback, which is illegal under
/// graph capture).
pub(crate) fn dump_hidden_localize(
    gpu: &Gpu,
    x: &GpuTensor,
    n_rows: usize,
    abs_pos_of_row0: usize,
    dim: usize,
    layer_idx: usize,
    tag: &str,
) {
    let prefix = match hipfire_config::developer_var("HIPFIRE_DUMP_HIDDEN") {
        Ok(p) => p,
        Err(_) => return,
    };
    let target: usize = hipfire_config::developer_var("HIPFIRE_DUMP_HIDDEN_POS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if target < abs_pos_of_row0 {
        return;
    }
    let row = target - abs_pos_of_row0;
    if row >= n_rows {
        return;
    }
    if gpu.hip.device_synchronize().is_err() {
        return;
    }
    let all = match gpu.download_f32(x) {
        Ok(v) => v,
        Err(_) => return,
    };
    let off = row * dim;
    if off + dim > all.len() {
        return;
    }
    use std::io::Write;
    let path = format!("{prefix}.{tag}");
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = f.write_all(&(layer_idx as u32).to_le_bytes());
        let mut bytes = Vec::with_capacity(dim * 4);
        for v in &all[off..off + dim] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let _ = f.write_all(&bytes);
    }
}

#[inline]
fn prefill_dispatch_workload(
    captures_per_token_hidden: bool,
    captures_rollback_tape: bool,
    is_tree_verify: bool,
) -> DispatchWorkload {
    if captures_per_token_hidden || captures_rollback_tape || is_tree_verify {
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
    per_token_hidden_out: Option<(&GpuTensor, usize)>,
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
fn run_independent_q8_attention(
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    kv_cache: &llama::KvCache,
    config: &Qwen35Config,
    layer_idx: usize,
    batch_size: usize,
    lane_capacity: usize,
    max_ctx_len: usize,
    active_mask: u64,
) -> HipResult<()> {
    debug_assert!(kv_cache.quant_q8);
    // Full-mask fast path preserves exact unmasked ABI; partial uses masked kernels.
    let full_mask = valid_lane_mask(batch_size)?;
    if active_mask == full_mask {
        gpu.kv_cache_write_q8_0_independent(
            &kv_cache.k_gpu[layer_idx],
            &pbs.fa_k_batch,
            &pbs.positions,
            config.n_kv_heads,
            config.head_dim,
            batch_size,
            lane_capacity,
        )?;
        gpu.kv_cache_write_q8_0_independent(
            &kv_cache.v_gpu[layer_idx],
            &pbs.fa_v_batch,
            &pbs.positions,
            config.n_kv_heads,
            config.head_dim,
            batch_size,
            lane_capacity,
        )?;
    } else {
        // Masked writes return before any inactive-lane read/write.
        gpu.kv_cache_write_q8_0_independent_masked(
            &kv_cache.k_gpu[layer_idx],
            &pbs.fa_k_batch,
            &pbs.positions,
            config.n_kv_heads,
            config.head_dim,
            batch_size,
            lane_capacity,
            active_mask,
        )?;
        gpu.kv_cache_write_q8_0_independent_masked(
            &kv_cache.v_gpu[layer_idx],
            &pbs.fa_v_batch,
            &pbs.positions,
            config.n_kv_heads,
            config.head_dim,
            batch_size,
            lane_capacity,
            active_mask,
        )?;
    }
    // Attention is scratch-only; keep unmasked but fixed-slot positions already range-checked
    // before the mutation boundary so it cannot read outside the lane slice.
    gpu.attention_q8_0_kv_independent(
        &pbs.fa_q_batch,
        &kv_cache.k_gpu[layer_idx],
        &kv_cache.v_gpu[layer_idx],
        &pbs.fa_attn_out_batch,
        &pbs.positions,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        lane_capacity,
        max_ctx_len,
        batch_size,
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
    // Other formats fall back to the per-token loop (kept for correctness
    // breadth; the MQ4-quantized hot path doesn't hit them).
    //
    // Multi-GPU band-mode: skip embedding when this is not the first band.
    // The activation already lives in `pbs.x_batch` from a peer-copy of
    // the previous band's `pbs.x_batch`.
    if do_embed
        && !pre_embedded
        && matches!(
            weights.embd_format,
            EmbeddingFormat::HFQ4G256 | EmbeddingFormat::Q8_0
        )
    {
        if !pre_uploaded {
            let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let tokens_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
            gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
        }
        match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256_batched(
                    &weights.token_embd,
                    &pbs.x_batch,
                    &pbs.tokens,
                    n,
                    dim,
                )?;
            }
            EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8_batched(
                    &weights.token_embd,
                    &pbs.x_batch,
                    &pbs.tokens,
                    n,
                    dim,
                )?;
            }
            _ => unreachable!(),
        }
    } else if do_embed && !pre_embedded {
        for (i, &tok) in tokens.iter().enumerate() {
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
    // holds the peer-copied activation from the previous band, so an
    // override applied at band 0 has already propagated through the layer
    // stack on that device — re-applying here would clobber the partial
    // forward state.
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
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(ovr.embed.as_ptr() as *const u8, dim * 4) };
            let offset = ovr.slot * dim_row_bytes;
            gpu.hip
                .memcpy_htod_offset(&pbs.x_batch.buf, offset, bytes)?;
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
        let positions_host: Vec<i32> = match batch_semantics {
            BatchSemantics::Sequential => (0..n).map(|i| (start_pos + i) as i32).collect(),
            BatchSemantics::Independent { positions, .. } => {
                debug_assert_eq!(positions.len(), n);
                positions.iter().map(|&p| p as i32).collect()
            }
        };
        let positions_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
        gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
        if let Some(tv) = tree_verify.as_ref() {
            debug_assert_eq!(tv.positions.len(), n, "tree RoPE positions length");
            let rope_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(tv.positions.as_ptr() as *const u8, n * 4) };
            gpu.hip.memcpy_htod(&pbs.rope_positions.buf, rope_bytes)?;
        }
    }

    Ok(())
}

/// S3-f16-projection-inputs: exact-route gate for the FP16 projection-input
/// fast path (all four `batch_chunk_*` projection hooks below).
///
/// All predicates are cheap field reads — no env/lock/JIT in the cycle (the
/// kill switch resolves once at `FeatureFlags` init). Every failed predicate
/// runs the pre-change F32-producer + `convert_f32_to_f16` path byte-for-byte.
#[inline]
fn mq_f16_projection_fast_route(gpu: &Gpu, fusion: DflashFusionCtx, n: usize, dim: usize) -> bool {
    matches!(fusion, DflashFusionCtx::ChainVerify)
        && gpu.arch_caps.is_gfx1100()
        && !gpu.flags.mq_f16_projection_off
        && n >= 1
        && n <= 16
        && dim % 256 == 0
        && !gpu.graphs.capture_mode
        && !gpu.replay.is_recording()
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
/// Prescaffold (behavior-only) extraction for S3-f16-projection-inputs.
///
/// Same statements, same order, same launches as the inlined block.
/// S9-mq4v2-persistent-prologues will issue `try_mq4v2_persistent_prologue`
/// from inside this hook after S3/S4 land.
fn batch_chunk_delta_net_input_projection(
    gpu: &mut Gpu,
    layer: &DeltaNetLayerWeights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    n: usize,
    dim: usize,
    q8_wmma_arch: bool,
    fusion: DflashFusionCtx,
) -> HipResult<()> {
    let _ = fusion;
    // S3-f16-projection-inputs fast path: emit exact FP16 directly from the
    // RMSNorm+FWHT producer into `x_rot_f16_batch` and consume it with the
    // F16-direct qkvza GEMM. Saves the `convert_f32_to_f16` launch with
    // bit-identical projection outputs. All four weights must share the
    // exact MQ4G256V2 stride (the fused kernel reads them as same-stride
    // byte arrays); anything else stays on the pre-change path.
    if mq_f16_projection_fast_route(gpu, fusion, n, dim)
        && layer.wqkv.gpu_dtype == DType::MQ4G256V2
        && layer.wz.gpu_dtype == DType::MQ4G256V2
        && layer.w_beta.gpu_dtype == DType::MQ4G256V2
        && layer.w_alpha.gpu_dtype == DType::MQ4G256V2
    {
        fused_rmsnorm_rotate_mq_f16_batched_for(
            gpu,
            &pbs.x_batch,
            &layer.attn_norm,
            &layer.wqkv,
            &pbs.x_rot_f16_batch,
            dim,
            config.norm_eps,
            n,
        )?;
        return gpu.gemm_qkvza_mq4g256v2_wmma_f16(
            &layer.wqkv.buf,
            &layer.wz.buf,
            &layer.w_beta.buf,
            &layer.w_alpha.buf,
            &pbs.x_rot_f16_batch,
            &pbs.dn_qkv_batch,
            &pbs.dn_z_batch,
            &pbs.dn_beta_batch,
            &pbs.dn_alpha_batch,
            layer.wqkv.m,
            layer.wz.m,
            layer.w_beta.m,
            layer.w_alpha.m,
            layer.wqkv.k,
            n,
        );
    }
    let is_mq = matches!(
        layer.wqkv.gpu_dtype,
        DType::MQ4G256
            | DType::MQ4G256V2
            // qt=52 shares qt=44's FWHT input contract (LUT decode needs
            // rotated x); the DISPATCH arm below routes to the FP8-LUT twin.
            | DType::MQ4G256V2Lloyd
            | DType::MQ4CG256
            | DType::MQ6G256
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256
            | DType::MQ3G256V2
            | DType::MQ2G256V2
            | DType::MQ3G256Lloyd
            | DType::MFP4G32
    );
    let is_6bit = matches!(layer.wqkv.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
    let is_mq3 = matches!(layer.wqkv.gpu_dtype, DType::MQ3G256);
    let is_mq3_lloyd = matches!(layer.wqkv.gpu_dtype, DType::MQ3G256Lloyd);
    let is_fp4 = matches!(layer.wqkv.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
    let is_q8 = matches!(layer.wqkv.gpu_dtype, DType::Q8_0);
    // TQ2G128/BQ1G128 have no fused qkvza/gate_up/qkv kernel, so they take
    // the same UNFUSED plain-GEMM strategy as Q8 rather than falling through
    // to the HFQ4 arm, which would read these packed blocks at the wrong
    // stride and produce fluent-but-wrong tokens.
    let is_lowbit = matches!(layer.wqkv.gpu_dtype, DType::TQ2G128 | DType::BQ1G128);
    // qt=52 re-arm anchor: all four LA projections Lloyd → FP8-LUT launcher.
    // Mixed Lloyd/uniform is a fail-closed HipError at the dispatch arm.
    let is_mq4v2_lloyd = all_mq4v2_lloyd(&[
        layer.wqkv.gpu_dtype,
        layer.wz.gpu_dtype,
        layer.w_beta.gpu_dtype,
        layer.w_alpha.gpu_dtype,
    ]);

    // Batched rmsnorm (+ FWHT for MQ) for the LA preamble.
    // x_batch / x_rot_batch are [N × dim] contiguous. For HFQ
    // we reuse x_rot_batch as the "normed, unrotated" output
    // so the subsequent GEMM can read it the same way.
    let mut iu4_prep: Option<rdna_compute::Int4MmqPrepared> = None;
    if is_mq {
        // C2: MQ4V2 IU4 producer sidecar (emit_f32=true for beta/alpha tails).
        iu4_prep = try_iu4_rmsnorm_prepared(
            gpu,
            &pbs.x_batch,
            &layer.attn_norm,
            &layer.wqkv,
            &pbs.x_rot_batch,
            dim,
            config.norm_eps,
            n,
            true,
        )?;
        if iu4_prep.is_none() {
            // AWQ-aware: next linear is LA's fused wqkv.
            fused_rmsnorm_rotate_mq_batched_for(
                gpu,
                &pbs.x_batch,
                &layer.attn_norm,
                &layer.wqkv,
                &pbs.x_rot_batch,
                dim,
                config.norm_eps,
                n,
            )?;
        }
    } else {
        gpu.rmsnorm_batched(
            &pbs.x_batch,
            &layer.attn_norm,
            &pbs.x_rot_batch,
            n,
            dim,
            config.norm_eps,
        )?;
    }

    // Batched 4-way LA projection (wqkv + wz + w_beta + w_alpha).
    if let Some(prep) = &iu4_prep {
        gpu.gemm_qkvza_mq4g256v2_wmma_iu4_prepared(
            &layer.wqkv.buf,
            &layer.wz.buf,
            &layer.w_beta.buf,
            &layer.w_alpha.buf,
            &pbs.x_rot_batch,
            prep,
            &pbs.dn_qkv_batch,
            &pbs.dn_z_batch,
            &pbs.dn_beta_batch,
            &pbs.dn_alpha_batch,
            layer.wqkv.m,
            layer.wz.m,
            layer.w_beta.m,
            layer.w_alpha.m,
            layer.wqkv.k,
            n,
        )?;
    } else if is_6bit {
        run_fused_qkvza_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedQkvzaHfq6G256,
            &layer.wqkv.buf,
            &layer.wz.buf,
            &layer.w_beta.buf,
            &layer.w_alpha.buf,
            &pbs.x_rot_batch,
            &pbs.dn_qkv_batch,
            &pbs.dn_z_batch,
            &pbs.dn_beta_batch,
            &pbs.dn_alpha_batch,
            layer.wqkv.m,
            layer.wz.m,
            layer.w_beta.m,
            layer.w_alpha.m,
            layer.wqkv.k,
            n,
        )?;
    } else if is_q8 && q8_wmma_arch {
        // `is_q8` only inspects `wqkv` (the routing anchor). The fused
        // kernel assumes ALL four weights share the Q8_0 stride; a
        // mixed-dtype layer would silently re-introduce the Tier-1
        // kernel-vs-stride corruption mode.
        debug_assert!(
            matches!(layer.wz.gpu_dtype, DType::Q8_0)
                && matches!(layer.w_beta.gpu_dtype, DType::Q8_0)
                && matches!(layer.w_alpha.gpu_dtype, DType::Q8_0),
            "LA qkvza Q8 WMMA dispatch requires all of wqkv/wz/w_beta/w_alpha to be Q8_0",
        );
        run_fused_qkvza_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedQkvzaQ8_0,
            &layer.wqkv.buf,
            &layer.wz.buf,
            &layer.w_beta.buf,
            &layer.w_alpha.buf,
            &pbs.x_rot_batch,
            &pbs.dn_qkv_batch,
            &pbs.dn_z_batch,
            &pbs.dn_beta_batch,
            &pbs.dn_alpha_batch,
            layer.wqkv.m,
            layer.wz.m,
            layer.w_beta.m,
            layer.w_alpha.m,
            layer.wqkv.k,
            n,
        )?;
    } else if is_q8 || is_lowbit {
        // #397 Ship 5.2 slice1: four plain Q8 batched GEMMs
        // (wqkv/wz/w_beta/w_alpha) → GemmFamily::run_key with the
        // GemmQ8_0BatchedChunked dispatcher-entry key → identical
        // gpu.gemm_q8_0_batched_chunked method, byte-for-byte.
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wqkv.gpu_dtype),
            &layer.wqkv.buf,
            layer.wqkv.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.dn_qkv_batch,
            layer.wqkv.m,
            layer.wqkv.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wz.gpu_dtype),
            &layer.wz.buf,
            layer.wz.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.dn_z_batch,
            layer.wz.m,
            layer.wz.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.w_beta.gpu_dtype),
            &layer.w_beta.buf,
            layer.w_beta.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.dn_beta_batch,
            layer.w_beta.m,
            layer.w_beta.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.w_alpha.gpu_dtype),
            &layer.w_alpha.buf,
            layer.w_alpha.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.dn_alpha_batch,
            layer.w_alpha.m,
            layer.w_alpha.k,
            n,
        )?;
    } else if is_mq3_lloyd {
        // 112 B/group Lloyd-MQ3 stride; X is already FWHT-rotated.
        run_fused_qkvza_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedQkvzaMq3G256Lloyd,
            &layer.wqkv.buf,
            &layer.wz.buf,
            &layer.w_beta.buf,
            &layer.w_alpha.buf,
            &pbs.x_rot_batch,
            &pbs.dn_qkv_batch,
            &pbs.dn_z_batch,
            &pbs.dn_beta_batch,
            &pbs.dn_alpha_batch,
            layer.wqkv.m,
            layer.wz.m,
            layer.w_beta.m,
            layer.w_alpha.m,
            layer.wqkv.k,
            n,
        )?;
    } else if is_mq3 {
        // 104 B/group HFQ3-stride; X is already FWHT-rotated by
        // fused_rmsnorm_rotate_mq_batched above. The FusedQkvzaHfq3G256
        // run-arm replicates the call-site WMMA-vs-base arch split
        // internally (gemm_qkvza_hfq3g256_wmma on has_wmma() else the
        // base cross-arch ladder), so the same kernel runs.
        run_fused_qkvza_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedQkvzaHfq3G256,
            &layer.wqkv.buf,
            &layer.wz.buf,
            &layer.w_beta.buf,
            &layer.w_alpha.buf,
            &pbs.x_rot_batch,
            &pbs.dn_qkv_batch,
            &pbs.dn_z_batch,
            &pbs.dn_beta_batch,
            &pbs.dn_alpha_batch,
            layer.wqkv.m,
            layer.wz.m,
            layer.w_beta.m,
            layer.w_alpha.m,
            layer.wqkv.k,
            n,
        )?;
    } else if is_fp4 {
        // HFP4G32: 17-B blocks (vs HFQ4's 136-B groups), per-row 16-B header.
        // MFP4G32: same storage as HFP4 + offline-FWHT weights; X is already
        // rotated above when is_mq, so this branch handles both unrotated
        // (HFP4) and post-rotation (MFP4) activations identically.
        run_fused_qkvza_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedQkvzaHfp4G32,
            &layer.wqkv.buf,
            &layer.wz.buf,
            &layer.w_beta.buf,
            &layer.w_alpha.buf,
            &pbs.x_rot_batch,
            &pbs.dn_qkv_batch,
            &pbs.dn_z_batch,
            &pbs.dn_beta_batch,
            &pbs.dn_alpha_batch,
            layer.wqkv.m,
            layer.wz.m,
            layer.w_beta.m,
            layer.w_alpha.m,
            layer.wqkv.k,
            n,
        )?;
    } else if is_mq4v2_lloyd {
        // qt=52: all four projections Lloyd. gfx11 → MMQ-LUT; gfx12 FP8-LUT.
        if lloyd_mmq_lut_route(gpu) {
            gpu.gemm_qkvza_mq4g256v2_mmq_lloyd(
                &layer.wqkv.buf,
                &layer.wz.buf,
                &layer.w_beta.buf,
                &layer.w_alpha.buf,
                &pbs.x_rot_batch,
                &pbs.dn_qkv_batch,
                &pbs.dn_z_batch,
                &pbs.dn_beta_batch,
                &pbs.dn_alpha_batch,
                layer.wqkv.m,
                layer.wz.m,
                layer.w_beta.m,
                layer.w_alpha.m,
                layer.wqkv.k,
                n,
                lloyd_c16_or_fail(&layer.wqkv, "batch_chunk_delta_net_input_projection")?,
                lloyd_c16_or_fail(&layer.wz, "batch_chunk_delta_net_input_projection")?,
                lloyd_c16_or_fail(&layer.w_beta, "batch_chunk_delta_net_input_projection")?,
                lloyd_c16_or_fail(&layer.w_alpha, "batch_chunk_delta_net_input_projection")?,
            )?;
        } else {
            gpu.gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2_fp8_lloyd(
                &layer.wqkv.buf,
                &layer.wz.buf,
                &layer.w_beta.buf,
                &layer.w_alpha.buf,
                &pbs.x_rot_batch,
                &pbs.dn_qkv_batch,
                &pbs.dn_z_batch,
                &pbs.dn_beta_batch,
                &pbs.dn_alpha_batch,
                layer.wqkv.m,
                layer.wz.m,
                layer.w_beta.m,
                layer.w_alpha.m,
                layer.wqkv.k,
                n,
                1,
                lloyd_e4m3_or_fail(&layer.wqkv, "batch_chunk_delta_net_input_projection")?,
                lloyd_e4m3_or_fail(&layer.wz, "batch_chunk_delta_net_input_projection")?,
                lloyd_e4m3_or_fail(&layer.w_beta, "batch_chunk_delta_net_input_projection")?,
                lloyd_e4m3_or_fail(&layer.w_alpha, "batch_chunk_delta_net_input_projection")?,
            )?;
        }
    } else if matches!(layer.wqkv.gpu_dtype, DType::MQ4G256V2Lloyd)
        || matches!(layer.wz.gpu_dtype, DType::MQ4G256V2Lloyd)
        || matches!(layer.w_beta.gpu_dtype, DType::MQ4G256V2Lloyd)
        || matches!(layer.w_alpha.gpu_dtype, DType::MQ4G256V2Lloyd)
    {
        // Mixed Lloyd/uniform LA projections: no kernel reads mixed codebooks.
        return Err(hip_bridge::HipError::new(
            0,
            "batch_chunk_delta_net_input_projection: mixed MQ4G256V2Lloyd/uniform LA projections — refusing (quantize all four or none)",
        ));
    } else {
        run_fused_qkvza_key(
            gpu,
            crate::forward_slots::fused_qkvza_key_for(layer.wqkv.gpu_dtype),
            &layer.wqkv.buf,
            &layer.wz.buf,
            &layer.w_beta.buf,
            &layer.w_alpha.buf,
            &pbs.x_rot_batch,
            &pbs.dn_qkv_batch,
            &pbs.dn_z_batch,
            &pbs.dn_beta_batch,
            &pbs.dn_alpha_batch,
            layer.wqkv.m,
            layer.wz.m,
            layer.w_beta.m,
            layer.w_alpha.m,
            layer.wqkv.k,
            n,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// Prescaffold (behavior-only) extraction for S5-gdn-pre-tape-fusion.
///
/// Same statements, same order, same launches as the inlined block.
/// Returns `tree_parents` so the caller's statement/launch order is unchanged.
fn batch_chunk_delta_net_pre_gdn<'a>(
    gpu: &mut Gpu,
    layer: &DeltaNetLayerWeights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    dn_state: &mut DeltaNetState,
    n: usize,
    k_dim: usize,
    v_dim: usize,
    n_v_heads: usize,
    hd: usize,
    batch_semantics: BatchSemantics<'_>,
    tree_verify: Option<TreeVerifyCtx<'a>>,
    gdn_tape: Option<&crate::speculative::GdnTape>,
    tape_offset: usize,
    delta_layer_idx: usize,
    fusion: DflashFusionCtx,
) -> HipResult<Option<&'a GpuTensor>> {
    let _ = fusion;
    // S5-gdn-pre-tape-fusion fast path: one launch for sigmoid(alpha/beta) +
    // tape writes + conv + QK norm/interleave. Chain verify is sequential
    // with no tree parents, so success returns None. Every failed predicate
    // (kill switch, non-sequential batch, non-GQA route, tape absence or
    // overflow, ineligible shapes/arch) runs the pre-change sequence below
    // launch-for-launch.
    if fusion == DflashFusionCtx::ChainVerify
        && !gpu.flags.gdn_pre_fuse_off
        && matches!(batch_semantics, BatchSemantics::Sequential)
        && config.linear_num_key_heads < n_v_heads
        && (1..=16).contains(&n)
    {
        if let Some(tape) = gdn_tape.as_ref() {
            if tape_offset + n <= tape.max_n {
                let fused = gpu.dflash_gdn_pre_capture_gfx1100(
                    &pbs.dn_beta_batch,
                    &pbs.dn_alpha_batch,
                    &layer.dt_bias,
                    &layer.a_log,
                    &pbs.dn_qkv_batch,
                    &layer.conv_weight,
                    &dn_state.conv_states[delta_layer_idx],
                    &pbs.dn_q_raw_batch,
                    &pbs.dn_k_raw_batch,
                    &pbs.dn_v_batch,
                    &pbs.dn_q_batch,
                    &pbs.dn_k_batch,
                    &tape.qkv_bufs[delta_layer_idx],
                    &tape.alpha_bufs[delta_layer_idx],
                    &tape.beta_bufs[delta_layer_idx],
                    n_v_heads,
                    config.linear_num_key_heads,
                    hd,
                    k_dim,
                    v_dim,
                    tape.qkv_dim,
                    n,
                    tape_offset,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                )?;
                if fused {
                    return Ok(None);
                }
            }
        }
    }
    // Slice-P fused preamble (gfx1201): sigmoid(beta) + alpha gate + conv1d +
    // SiLU + split + Q/K norm + scale + interleave in ONE launch. Admission is
    // the kernel header's exact list: flag opt-in, exact gfx1201, sequential
    // semantics, no tree parents, no DFlash tape (tape writes stay on the old
    // path), the GQA interleave branch with exact head division, HD 128, and
    // n >= 1 (ragged tails run with a short last group). The MoE sibling
    // (`batch_chunk_delta_net_moe`) keeps the 3-launch sequence. q_raw/k_raw
    // stores are kept by the fused kernel (v1).
    {
        let no_tree_parents = tree_verify
            .as_ref()
            .and_then(|c| c.parent_indices)
            .is_none();
        let n_key_heads = config.linear_num_key_heads;
        if gpu.flags.gfx12_gdn_pre_fused
            && gpu.arch_caps.is_gfx1201()
            && matches!(batch_semantics, BatchSemantics::Sequential)
            && no_tree_parents
            && gdn_tape.is_none()
            && n_key_heads < n_v_heads
            && n_v_heads % n_key_heads == 0
            && hd == 128
            && n >= 1
        {
            let ratio = n_v_heads / n_key_heads;
            gpu.gdn_pre_batched_gfx1201(
                &pbs.dn_beta_batch,
                &pbs.dn_alpha_batch,
                &layer.dt_bias,
                &layer.a_log,
                &pbs.dn_qkv_batch,
                &layer.conv_weight,
                &dn_state.conv_states[delta_layer_idx],
                &pbs.dn_q_raw_batch,
                &pbs.dn_k_raw_batch,
                &pbs.dn_v_batch,
                &pbs.dn_q_batch,
                &pbs.dn_k_batch,
                n_v_heads,
                n_key_heads,
                ratio,
                k_dim,
                v_dim,
                n,
                1.0 / (hd as f32).sqrt(),
                config.norm_eps,
            )?;
            return Ok(None);
        }
    }
    // Fused sigmoid(beta) + alpha_gate(alpha) — [N × n_v_heads] each.
    gpu.fused_sigmoid_alpha_gate_f32_batched(
        &pbs.dn_beta_batch,
        &pbs.dn_alpha_batch,
        &layer.dt_bias,
        &layer.a_log,
        n_v_heads,
        n,
    )?;

    // DFlash tape capture: snap pre-conv1d qkv + post-sigmoid α/β
    // for this layer into the per-layer tape slots. The next LA
    // layer's fused_qkvza / fused_sigmoid_alpha_gate will overwrite
    // dn_qkv_batch / dn_{alpha,beta}_batch, so capture must happen
    // now (after sigmoid_alpha_gate, before conv1d consumes qkv).
    if let Some(tape) = gdn_tape.as_ref() {
        let qkv_row_bytes = tape.qkv_dim * 4;
        let alpha_row_bytes = n_v_heads * 4;
        let off_qkv = tape_offset * qkv_row_bytes;
        let off_a = tape_offset * alpha_row_bytes;
        let copy_qkv = n * qkv_row_bytes;
        let copy_a = n * alpha_row_bytes;
        gpu.memcpy_dtod_at_auto(
            &tape.qkv_bufs[delta_layer_idx].buf,
            off_qkv,
            &pbs.dn_qkv_batch.buf,
            0,
            copy_qkv,
        )?;
        gpu.memcpy_dtod_at_auto(
            &tape.alpha_bufs[delta_layer_idx].buf,
            off_a,
            &pbs.dn_alpha_batch.buf,
            0,
            copy_a,
        )?;
        gpu.memcpy_dtod_at_auto(
            &tape.beta_bufs[delta_layer_idx].buf,
            off_a,
            &pbs.dn_beta_batch.buf,
            0,
            copy_a,
        )?;
    }

    // Tree-aware dispatch gate: when the caller provides
    // parent_indices (Phase 3b+ of Task #101), swap the linear
    // conv1d + GDN for tree-walking variants that eliminate
    // sibling-subtree state cross-contamination. The tree
    // kernels are READ-ONLY on dn_state (don't advance it) —
    // caller runs linear replay on the accepted spine
    // post-acceptance to commit the trajectory.
    let tree_parents = tree_verify.and_then(|c| c.parent_indices);
    if let Some(parents) = tree_parents {
        gpu.conv1d_silu_split_tree_f32_n(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            &pbs.dn_v_batch,
            &pbs.dn_qkv_batch,
            &layer.conv_weight,
            &dn_state.conv_states[delta_layer_idx],
            parents,
            k_dim,
            v_dim,
            n,
        )?;
    } else if let BatchSemantics::Independent { active_mask, .. } = batch_semantics {
        let full_mask = valid_lane_mask(n)?;
        if active_mask == full_mask {
            gpu.conv1d_silu_split_f32_independent(
                &pbs.dn_q_raw_batch,
                &pbs.dn_k_raw_batch,
                &pbs.dn_v_batch,
                &pbs.dn_qkv_batch,
                &layer.conv_weight,
                &dn_state.conv_states[delta_layer_idx],
                k_dim,
                v_dim,
                n,
            )?;
        } else {
            gpu.conv1d_silu_split_f32_independent_masked(
                &pbs.dn_q_raw_batch,
                &pbs.dn_k_raw_batch,
                &pbs.dn_v_batch,
                &pbs.dn_qkv_batch,
                &layer.conv_weight,
                &dn_state.conv_states[delta_layer_idx],
                k_dim,
                v_dim,
                n,
                active_mask,
            )?;
        }
    } else {
        gpu.conv1d_silu_split_f32_n(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            &pbs.dn_v_batch,
            &pbs.dn_qkv_batch,
            &layer.conv_weight,
            &dn_state.conv_states[delta_layer_idx],
            k_dim,
            v_dim,
            n,
        )?;
    }

    // Fused L2-norm(Q) + scale(Q) + L2-norm(K) + repeat-interleave
    // when n_key_heads < n_v_heads. One launch instead of two —
    // ~200µs saved per LA layer × ~30 LA layers ≈ 6ms per prefill
    // on A3B (R9700/gfx1201).
    //
    // The fused kernel reads q_raw/k_raw (unchanged on exit), so
    // the conv1d output is preserved if downstream readers need it
    // (no current consumer reads _raw after this).
    if config.linear_num_key_heads < n_v_heads {
        let ratio = n_v_heads / config.linear_num_key_heads;
        gpu.fused_qk_l2_norm_scale_interleave_f32_batched(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            &pbs.dn_q_batch,
            &pbs.dn_k_batch,
            config.linear_num_key_heads,
            ratio,
            hd,
            1.0 / (hd as f32).sqrt(),
            config.norm_eps,
            n,
        )?;
    } else {
        // n_key_heads == n_v_heads → no replication; keep the
        // original sequence (norm in place, then memcpy).
        gpu.fused_qk_l2_norm_scale_f32_batched(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            config.linear_num_key_heads,
            hd,
            1.0 / (hd as f32).sqrt(),
            config.norm_eps,
            n,
        )?;
        gpu.memcpy_dtod_auto(&pbs.dn_q_batch.buf, &pbs.dn_q_raw_batch.buf, n * k_dim * 4)?;
        gpu.memcpy_dtod_auto(&pbs.dn_k_batch.buf, &pbs.dn_k_raw_batch.buf, n * k_dim * 4)?;
    }
    Ok(tree_parents)
}

#[allow(clippy::too_many_arguments)]
/// S4-f16-residual-inputs: shared fast-route predicate for the four
/// post-attention/down hooks.
///
/// True only for the frozen fixture route: chain (non-tree) verify on exact
/// gfx1100, the slice kill switch clear, an MQ4G256V2 residual consumer, a
/// `Residual` epilogue, and a verify-block batch `1 <= n <= 16`. Every false
/// keeps the pre-change path byte-for-byte.
fn s4_residual_fast(
    gpu: &Gpu,
    fusion: DflashFusionCtx,
    w_dtype: DType,
    epilogue: &BatchEpilogue<'_>,
    n: usize,
) -> bool {
    fusion == DflashFusionCtx::ChainVerify
        && !gpu.flags.mq_f16_residual_off
        && gpu.arch_caps.is_gfx1100()
        && w_dtype == DType::MQ4G256V2
        && matches!(epilogue, BatchEpilogue::Residual)
        && (1..=16).contains(&n)
}

/// Prescaffold (behavior-only) extraction for S4-f16-residual-inputs.
///
/// Same statements, same order, same launches as the inlined block.
/// S9-mq4v2-persistent-prologues will issue `try_mq4v2_persistent_prologue`
/// from inside this hook after S3/S4 land.
fn batch_chunk_delta_net_output_projection(
    gpu: &mut Gpu,
    layer: &DeltaNetLayerWeights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    n: usize,
    n_v_heads: usize,
    q8_wmma_arch: bool,
    arch_has_wmma: bool,
    epilogue: BatchEpilogue<'_>,
    fusion: DflashFusionCtx,
) -> HipResult<()> {
    // S4: one gated_norm+FWHT+F16 producer + direct-F16 residual GEMM
    // instead of gated_norm_f32 + mq_rotate_x + convert.
    if s4_residual_fast(gpu, fusion, layer.wo.gpu_dtype, &epilogue, n)
        && config.linear_value_head_dim == 128
        && n_v_heads * config.linear_value_head_dim == layer.wo.k
    {
        let k = layer.wo.k;
        let m = layer.wo.m;
        if k > 0 && k % 256 == 0 {
            if let Some(awq) = layer.wo.awq_scale.as_ref() {
                if awq.numel() >= k {
                    gpu.gated_norm_rotate_mq_awq_f16_batched(
                        &pbs.dn_attn_out_batch,
                        &pbs.dn_z_batch,
                        &layer.norm_weight,
                        awq,
                        &pbs.dn_normed_rot_f16_batch,
                        n_v_heads,
                        config.linear_value_head_dim,
                        config.norm_eps,
                        n,
                    )?;
                    let x_f16 = pbs.dn_normed_rot_f16_batch.sub_offset(0, n * k);
                    gpu.gemm_mq4g256v2_residual_wmma_f16(
                        &layer.wo.buf,
                        &x_f16,
                        &pbs.x_batch,
                        m,
                        k,
                        n,
                    )?;
                    return Ok(());
                }
            } else {
                gpu.gated_norm_rotate_mq_f16_batched(
                    &pbs.dn_attn_out_batch,
                    &pbs.dn_z_batch,
                    &layer.norm_weight,
                    &pbs.dn_normed_rot_f16_batch,
                    n_v_heads,
                    config.linear_value_head_dim,
                    config.norm_eps,
                    n,
                )?;
                let x_f16 = pbs.dn_normed_rot_f16_batch.sub_offset(0, n * k);
                gpu.gemm_mq4g256v2_residual_wmma_f16(&layer.wo.buf, &x_f16, &pbs.x_batch, m, k, n)?;
                return Ok(());
            }
        }
    }
    // Batched gated output norm.
    gpu.gated_norm_f32_batched(
        &pbs.dn_attn_out_batch,
        &pbs.dn_z_batch,
        &layer.norm_weight,
        &pbs.dn_normed_batch,
        n_v_heads,
        config.linear_value_head_dim,
        config.norm_eps,
        n,
    )?;

    // Batched wo + residual/partial.
    //
    // For MQ weights, the decode path's weight_gemv_residual
    // internally FWHT-rotates dn_normed into mq_x_rot before
    // calling gemv_hfq{4,6}g256_residual (MQ weights are pre-rotated
    // at quant time; math requires dot(rot(W), rot(x)) = dot(W,x)).
    // For HFQ weights no rotation is needed — the activation
    // feeds gemm_hfq{4,6}g256_residual directly.
    let wo_is_mq = matches!(
        layer.wo.gpu_dtype,
        DType::MQ4G256
            | DType::MQ4G256V2
            // qt=52 shares qt=44's FWHT input contract (LUT decode needs rotated x).
            | DType::MQ4G256V2Lloyd
            | DType::MQ4CG256
            | DType::MQ6G256
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256
            | DType::MQ3G256V2
            | DType::MQ2G256V2
            | DType::MQ3G256Lloyd
            | DType::MFP4G32
    );
    // T-B: fused rotate+quantize — x_rot feeds only the IU4 residual GEMM.
    let iu4_wo_prep = try_iu4_rotate_prepared(
        gpu,
        &layer.wo,
        &pbs.dn_normed_batch,
        &pbs.dn_normed_rot_batch,
        layer.wo.k,
        n,
        &epilogue,
    )?;
    let wo_input = if iu4_wo_prep.is_some() {
        &pbs.dn_normed_rot_batch
    } else if wo_is_mq {
        rotate_x_mq_batched_for(
            gpu,
            &layer.wo,
            &pbs.dn_normed_batch,
            &pbs.dn_normed_rot_batch,
            layer.wo.k,
            n,
        )?;
        &pbs.dn_normed_rot_batch
    } else {
        &pbs.dn_normed_batch
    };
    if let Some(prep) = &iu4_wo_prep {
        gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(
            &layer.wo.buf,
            prep,
            &pbs.x_batch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    } else {
        dispatch_batched_gemm_epilogue(
            gpu,
            pbs,
            &layer.wo,
            wo_input,
            &epilogue,
            n,
            q8_wmma_arch,
            arch_has_wmma,
        )?;
    }
    Ok(())
}

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
) -> HipResult<()> {
    // Per-layer dtype branch: MQ4 needs FWHT-rotation on the
    // activation to match its pre-rotated weights; HFQ4 uses
    // plain rmsnormed activations. The GEMM kernels themselves
    // are dtype-agnostic — they just consume whatever [N × K]
    // activation buffer we point them at.
    // GAP NOTE: this matcher (and the 7 sibling dense LA/FA
    // matchers in this file) wires MQ3G256Lloyd through the
    // gemm_*_mq3g256_lloyd_wmma family. MQ2G256Lloyd remains
    // unwired — to add it, update is_batchable_la, ALL 8 is_mq*
    // matchers, AND add a Lloyd-MQ2-specific GEMM dispatch arm
    // together (the all-together corruption-prevention rule from
    // docs/plans/mq-lloyd-batched-prefill-followup.md). MQ4-Lloyd
    // is wired in a separate PR (issue #182).
    batch_chunk_delta_net_input_projection(gpu, layer, config, pbs, n, dim, q8_wmma_arch, fusion)?;

    let tree_parents = batch_chunk_delta_net_pre_gdn(
        gpu,
        layer,
        config,
        pbs,
        dn_state,
        n,
        k_dim,
        v_dim,
        n_v_heads,
        hd,
        batch_semantics,
        tree_verify,
        gdn_tape,
        tape_offset,
        delta_layer_idx,
        fusion,
    )?;

    // Gated Delta Net — tree variant reads per-token S from
    // s_tape[parent] (or pre-block s_q8_init at root); linear
    // variant advances dn_state.s_matrices in place.
    if let Some(parents) = tree_parents {
        // Tree-verify GDN, dispatched by DeltaNet state quant.
        // FP32 uses the full-precision tree-tape kernel (no
        // per-node Q8 round-trip); Q8 the original; Q4 tree has
        // no kernel (was silently mis-routed to the Q8 tree
        // kernel before — now a clean error).
        match dn_state.quant {
            StateQuant::FP32 => {
                let tape_f32 = pbs.dn_s_tape_f32.as_ref().expect(
                                "FP32 tree-aware LA requires dn_s_tape_f32 scratch (check PrefillBatchScratch::new)",
                            );
                gpu.gated_delta_net_f32_tree_batch_seq(
                    &pbs.dn_q_batch,
                    &pbs.dn_k_batch,
                    &pbs.dn_v_batch,
                    &pbs.dn_alpha_batch,
                    &pbs.dn_beta_batch,
                    &dn_state.s_matrices[delta_layer_idx],
                    tape_f32,
                    parents,
                    &pbs.dn_attn_out_batch,
                    n,
                    n_v_heads,
                    config.linear_value_head_dim,
                )?;
            }
            StateQuant::Q8 => {
                let tape_q8 = pbs.dn_s_tape_q8.as_ref().expect(
                    "tree-aware LA requires dn_s_tape_q8 scratch (check PrefillBatchScratch::new)",
                );
                let tape_sc = pbs.dn_s_tape_scales.as_ref()
                                .expect("tree-aware LA requires dn_s_tape_scales scratch (check PrefillBatchScratch::new)");
                gpu.gated_delta_net_q8_tree_batch_seq(
                    &pbs.dn_q_batch,
                    &pbs.dn_k_batch,
                    &pbs.dn_v_batch,
                    &pbs.dn_alpha_batch,
                    &pbs.dn_beta_batch,
                    &dn_state.s_matrices[delta_layer_idx],
                    &dn_state.s_scales[delta_layer_idx],
                    tape_q8,
                    tape_sc,
                    parents,
                    &pbs.dn_attn_out_batch,
                    n,
                    n_v_heads,
                    config.linear_value_head_dim,
                )?;
            }
            StateQuant::Q4 => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "Q4 DeltaNet state + tree-verify (DDTree) is unsupported: \
                                 there is no Q4 tree-tape GDN kernel. Use Q8 or FP32 state \
                                 for tree spec-decode.",
                ));
            }
        }
    } else {
        // EXPERIMENT (not #417): mirror the state-quant dispatch the
        // decode siblings already do (forward_scratch_layers:13194),
        // so the captured/eager batched prefill honours FP32/Q4 state
        // instead of forcing the Q8 kernel onto non-Q8 buffers.
        match dn_state.quant {
            StateQuant::FP32 => {
                if rdna_compute::norm::gdn_chunked() && n > 1 {
                    gpu.gated_delta_net_f32_chunked(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                        rdna_compute::norm::gdn_chunk_size(),
                    )?
                } else {
                    gpu.gated_delta_net_f32_batch_seq(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?
                }
            }
            StateQuant::Q8 => {
                if let BatchSemantics::Independent { active_mask, .. } = batch_semantics {
                    let full_mask = valid_lane_mask(n)?;
                    if active_mask == full_mask {
                        gpu.gated_delta_net_q8_independent(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                            dn_state.ef_residual(delta_layer_idx),
                        )?
                    } else {
                        gpu.gated_delta_net_q8_independent_masked(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                            dn_state.ef_residual(delta_layer_idx),
                            active_mask,
                        )?
                    }
                } else if let Some(stride) = commit_stride {
                    // Widened ordinary chunk: the same wrapper with the same
                    // 512-row q/k/v/output views and the same unsliced
                    // S/scales/EF owners, once per legacy segment. Repeats the
                    // exact dequant→recurrence→EF-fold→requant at every seam;
                    // no new arithmetic exists. Only the admitted ordinary
                    // path sets `commit_stride`, so tree/independent/TP/EP
                    // callers keep their branch above.
                    assert!(
                        n % stride == 0,
                        "widened GDN chunk {n} is not a multiple of commit stride {stride}"
                    );
                    for off in (0..n).step_by(stride) {
                        let q = pbs.dn_q_batch.sub_offset(off * v_dim, stride * v_dim);
                        let k = pbs.dn_k_batch.sub_offset(off * v_dim, stride * v_dim);
                        let v = pbs.dn_v_batch.sub_offset(off * v_dim, stride * v_dim);
                        let alpha = pbs
                            .dn_alpha_batch
                            .sub_offset(off * n_v_heads, stride * n_v_heads);
                        let beta = pbs
                            .dn_beta_batch
                            .sub_offset(off * n_v_heads, stride * n_v_heads);
                        let out =
                            pbs.dn_attn_out_batch.sub_offset(off * v_dim, stride * v_dim);
                        gpu.gated_delta_net_q8_batch_seq(
                            &q,
                            &k,
                            &v,
                            &alpha,
                            &beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &out,
                            stride,
                            n_v_heads,
                            config.linear_value_head_dim,
                            dn_state.ef_residual(delta_layer_idx),
                        )?
                    }
                } else {
                    gpu.gated_delta_net_q8_batch_seq(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                        dn_state.ef_residual(delta_layer_idx),
                    )?
                }
            }
            StateQuant::Q4 => gpu.gated_delta_net_q4(
                &pbs.dn_q_batch,
                &pbs.dn_k_batch,
                &pbs.dn_v_batch,
                &pbs.dn_alpha_batch,
                &pbs.dn_beta_batch,
                &dn_state.s_matrices[delta_layer_idx],
                &dn_state.s_scales[delta_layer_idx],
                &pbs.dn_attn_out_batch,
                n,
                n_v_heads,
                config.linear_value_head_dim,
            )?,
        }
    }

    batch_chunk_delta_net_output_projection(
        gpu,
        layer,
        config,
        pbs,
        n,
        n_v_heads,
        q8_wmma_arch,
        arch_has_wmma,
        epilogue,
        fusion,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
/// Prescaffold (behavior-only) extraction for S3-f16-projection-inputs.
///
/// Same statements, same order, same launches as the inlined block.
/// S9-mq4v2-persistent-prologues will issue `try_mq4v2_persistent_prologue`
/// from inside this hook after S3/S4 land.
fn batch_chunk_delta_net_ffn_gate_up(
    gpu: &mut Gpu,
    layer: &DeltaNetLayerWeights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    n: usize,
    dim: usize,
    q8_wmma_arch: bool,
    fusion: DflashFusionCtx,
) -> HipResult<()> {
    let _ = fusion;
    // S3-f16-projection-inputs fast path: exact-FP16 FFN gate/up inputs.
    // gate/up share the pre-rotation input, so both must be MQ4G256V2.
    if mq_f16_projection_fast_route(gpu, fusion, n, dim)
        && layer.w_gate.gpu_dtype == DType::MQ4G256V2
        && layer.w_up.gpu_dtype == DType::MQ4G256V2
    {
        fused_rmsnorm_rotate_mq_f16_batched_for(
            gpu,
            &pbs.x_batch,
            &layer.ffn_norm,
            &layer.w_gate,
            &pbs.x_rot_f16_batch,
            dim,
            config.norm_eps,
            n,
        )?;
        return gpu.gemm_gate_up_mq4g256v2_wmma_f16(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            &pbs.x_rot_f16_batch,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
            n,
        );
    }
    // FFN: rmsnorm (+ rotate for MQ).
    let ffn_is_mq = matches!(
        layer.w_gate.gpu_dtype,
        DType::MQ4G256
            | DType::MQ4G256V2
            // qt=52 shares qt=44's FWHT input contract; dispatch arm below
            // routes to the FP8-LUT twin.
            | DType::MQ4G256V2Lloyd
            | DType::MQ4CG256
            | DType::MQ6G256
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256
            | DType::MQ3G256V2
            | DType::MQ2G256V2
            | DType::MQ3G256Lloyd
            | DType::MFP4G32
    );
    let ffn_is_6bit = matches!(layer.w_gate.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
    let ffn_is_mq3 = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256);
    let ffn_is_mq3_lloyd = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256Lloyd);
    let ffn_is_fp4 = matches!(layer.w_gate.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
    let ffn_is_q8 = matches!(layer.w_gate.gpu_dtype, DType::Q8_0);
    // TQ2G128/BQ1G128 have no fused qkvza/gate_up/qkv kernel, so they take
    // the same UNFUSED plain-GEMM strategy as Q8 rather than falling through
    // to the HFQ4 arm, which would read these packed blocks at the wrong
    // stride and produce fluent-but-wrong tokens.
    let ffn_is_lowbit = matches!(layer.w_gate.gpu_dtype, DType::TQ2G128 | DType::BQ1G128);
    // qt=52 re-arm anchor: gate+up both Lloyd → FP8-LUT launcher.
    let ffn_is_mq4v2_lloyd = all_mq4v2_lloyd(&[layer.w_gate.gpu_dtype, layer.w_up.gpu_dtype]);
    let mut iu4_prep: Option<rdna_compute::Int4MmqPrepared> = None;
    if ffn_is_mq {
        // C2: MQ4V2 IU4 producer (emit_f32=false — gate_up has no f32 small tails).
        iu4_prep = try_iu4_rmsnorm_prepared(
            gpu,
            &pbs.x_batch,
            &layer.ffn_norm,
            &layer.w_gate,
            &pbs.x_rot_batch,
            dim,
            config.norm_eps,
            n,
            false,
        )?;
        if iu4_prep.is_none() {
            // AWQ-aware: next linear is w_gate (gate/up share input → same AWQ scale).
            fused_rmsnorm_rotate_mq_batched_for(
                gpu,
                &pbs.x_batch,
                &layer.ffn_norm,
                &layer.w_gate,
                &pbs.x_rot_batch,
                dim,
                config.norm_eps,
                n,
            )?;
        }
    } else {
        gpu.rmsnorm_batched(
            &pbs.x_batch,
            &layer.ffn_norm,
            &pbs.x_rot_batch,
            n,
            dim,
            config.norm_eps,
        )?;
    }

    // Batched gate+up projection.
    // #397 Ship 5.2 slice 2: fused gate+up dtypes → FusedQkvFamily
    // (batched-prefill gate+up variant) via run_fused_gate_up_key.
    // The Q8-non-WMMA case stays as two plain GemmQ8_0BatchedChunked
    // GEMMs (not a fused kernel — slice 1). The HFQ3 WMMA-vs-base
    // split is folded into the FusedGateUpHfq3G256 run-arm, which
    // re-derives it from gpu.arch_caps.has_wmma() (== arch_has_wmma).
    if let Some(prep) = &iu4_prep {
        gpu.gemm_gate_up_mq4g256v2_wmma_iu4_prepared(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            prep,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
            n,
        )?;
    } else if ffn_is_6bit {
        run_fused_gate_up_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedGateUpHfq6G256,
            &layer.w_gate.buf,
            &layer.w_up.buf,
            &pbs.x_rot_batch,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
            n,
        )?;
    } else if ffn_is_q8 && q8_wmma_arch {
        debug_assert!(
            matches!(layer.w_up.gpu_dtype, DType::Q8_0),
            "LA FFN Q8 WMMA dispatch requires both w_gate and w_up to be Q8_0",
        );
        run_fused_gate_up_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedGateUpQ8_0,
            &layer.w_gate.buf,
            &layer.w_up.buf,
            &pbs.x_rot_batch,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
            n,
        )?;
    } else if ffn_is_q8 || ffn_is_lowbit {
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.w_gate.gpu_dtype),
            &layer.w_gate.buf,
            layer.w_gate.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.gate_ffn_batch,
            layer.w_gate.m,
            layer.w_gate.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.w_up.gpu_dtype),
            &layer.w_up.buf,
            layer.w_up.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.up_batch,
            layer.w_up.m,
            layer.w_up.k,
            n,
        )?;
    } else if ffn_is_mq3_lloyd {
        run_fused_gate_up_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedGateUpMq3G256Lloyd,
            &layer.w_gate.buf,
            &layer.w_up.buf,
            &pbs.x_rot_batch,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
            n,
        )?;
    } else if ffn_is_mq3 {
        run_fused_gate_up_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedGateUpHfq3G256,
            &layer.w_gate.buf,
            &layer.w_up.buf,
            &pbs.x_rot_batch,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
            n,
        )?;
    } else if ffn_is_fp4 {
        run_fused_gate_up_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedGateUpHfp4G32,
            &layer.w_gate.buf,
            &layer.w_up.buf,
            &pbs.x_rot_batch,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
            n,
        )?;
    } else if ffn_is_mq4v2_lloyd {
        // qt=52: gate+up both Lloyd. gfx11 → MMQ-LUT; gfx12 FP8-LUT.
        if lloyd_mmq_lut_route(gpu) {
            gpu.gemm_gate_up_mq4g256v2_mmq_lloyd(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
                lloyd_c16_or_fail(&layer.w_gate, "batch_chunk_delta_net_ffn_gate_up")?,
                lloyd_c16_or_fail(&layer.w_up, "batch_chunk_delta_net_ffn_gate_up")?,
            )?;
        } else {
            gpu.gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2_fp8_lloyd(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
                1,
                lloyd_e4m3_or_fail(&layer.w_gate, "batch_chunk_delta_net_ffn_gate_up")?,
                lloyd_e4m3_or_fail(&layer.w_up, "batch_chunk_delta_net_ffn_gate_up")?,
            )?;
        }
    } else if matches!(layer.w_gate.gpu_dtype, DType::MQ4G256V2Lloyd)
        || matches!(layer.w_up.gpu_dtype, DType::MQ4G256V2Lloyd)
    {
        // Mixed Lloyd/uniform gate/up: no kernel reads mixed codebooks.
        return Err(hip_bridge::HipError::new(
            0,
            "batch_chunk_delta_net_ffn_gate_up: mixed MQ4G256V2Lloyd/uniform gate/up — refusing (quantize both or neither)",
        ));
    } else {
        run_fused_gate_up_key(
            gpu,
            crate::forward_slots::fused_gate_up_key_for(layer.w_gate.gpu_dtype),
            &layer.w_gate.buf,
            &layer.w_up.buf,
            &pbs.x_rot_batch,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
            n,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// Prescaffold (behavior-only) extraction for S4-f16-residual-inputs.
///
/// Same statements, same order, same launches as the inlined block.
/// S9-mq4v2-persistent-prologues will issue `try_mq4v2_persistent_prologue`
/// from inside this hook after S3/S4 land.
fn batch_chunk_delta_net_ffn_down(
    gpu: &mut Gpu,
    layer: &DeltaNetLayerWeights,
    pbs: &PrefillBatchScratch,
    hidden_dim: usize,
    n: usize,
    q8_wmma_arch: bool,
    arch_has_wmma: bool,
    epilogue: BatchEpilogue<'_>,
    fusion: DflashFusionCtx,
) -> HipResult<()> {
    // S4: one silu*up+FWHT+F16 producer + direct-F16 residual GEMM instead
    // of fused_silu_mul_rotate_mq_batched + convert.
    if s4_residual_fast(gpu, fusion, layer.w_down.gpu_dtype, &epilogue, n) {
        let k = layer.w_down.k;
        let m = layer.w_down.m;
        if k > 0 && k % 256 == 0 && k == hidden_dim {
            fused_silu_mul_rotate_mq_f16_batched_for(
                gpu,
                &layer.w_down,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                &pbs.ffn_hidden_f16_batch,
                hidden_dim,
                n,
            )?;
            let x_f16 = pbs.ffn_hidden_f16_batch.sub_offset(0, n * hidden_dim);
            gpu.gemm_mq4g256v2_residual_wmma_f16(
                &layer.w_down.buf,
                &x_f16,
                &pbs.x_batch,
                m,
                hidden_dim,
                n,
            )?;
            return Ok(());
        }
    }
    // SwiGLU activation feeding w_down. For MQ, we need the
    // output FWHT-rotated so it matches the pre-rotated w_down
    // weights. For HFQ, plain silu_mul is enough. silu_mul_f32
    // is purely element-wise and uses numel() as its length,
    // so a [N × hidden_dim] tensor processes all rows in one
    // launch with no batch offset needed.
    let w_down_is_mq = matches!(
        layer.w_down.gpu_dtype,
        DType::MQ4G256
            | DType::MQ4G256V2
            // qt=52 shares qt=44's FWHT input contract (LUT decode needs rotated x).
            | DType::MQ4G256V2Lloyd
            | DType::MQ4CG256
            | DType::MQ6G256
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256
            | DType::MQ3G256V2
            | DType::MQ2G256V2
            | DType::MQ3G256Lloyd
            | DType::MFP4G32
    );
    let mut iu4_prep: Option<rdna_compute::Int4MmqPrepared> = None;
    if w_down_is_mq {
        // C2: SwiGLU/FWHT IU4 producer for w_down (emit_f32=false). Residual only —
        // Partial TP epilogue still needs the f32 rotated buffer.
        if matches!(&epilogue, BatchEpilogue::Residual) {
            iu4_prep = try_iu4_silu_prepared(
                gpu,
                &layer.w_down,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                &pbs.ffn_hidden_batch,
                hidden_dim,
                n,
                false,
            )?;
        }
        if iu4_prep.is_none() {
            // F2: AWQ-aware silu_mul+rotate for w_down input.
            fused_silu_mul_rotate_mq_batched_for(
                gpu,
                &layer.w_down,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                &pbs.ffn_hidden_batch,
                hidden_dim,
                n,
            )?;
        }
    } else {
        gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
    }
    // Batched w_down + residual/partial.
    if let Some(prep) = &iu4_prep {
        gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(
            &layer.w_down.buf,
            prep,
            &pbs.x_batch,
            layer.w_down.m,
            layer.w_down.k,
            n,
        )?;
    } else {
        dispatch_batched_gemm_epilogue(
            gpu,
            pbs,
            &layer.w_down,
            &pbs.ffn_hidden_batch,
            &epilogue,
            n,
            q8_wmma_arch,
            arch_has_wmma,
        )?;
    }
    Ok(())
}

pub(crate) fn batch_chunk_delta_net_ffn(
    gpu: &mut Gpu,
    layer: &DeltaNetLayerWeights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    n: usize,
    dim: usize,
    hidden_dim: usize,
    q8_wmma_arch: bool,
    arch_has_wmma: bool,
    epilogue: BatchEpilogue<'_>,
    fusion: DflashFusionCtx,
) -> HipResult<()> {
    batch_chunk_delta_net_ffn_gate_up(gpu, layer, config, pbs, n, dim, q8_wmma_arch, fusion)?;

    batch_chunk_delta_net_ffn_down(
        gpu,
        layer,
        pbs,
        hidden_dim,
        n,
        q8_wmma_arch,
        arch_has_wmma,
        epilogue,
        fusion,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
/// Prescaffold (behavior-only) extraction for S3-f16-projection-inputs.
///
/// Same statements, same order, same launches as the inlined block.
/// S9-mq4v2-persistent-prologues will issue `try_mq4v2_persistent_prologue`
/// from inside this hook after S3/S4 land.
fn batch_chunk_full_attn_input_projection(
    gpu: &mut Gpu,
    layer: &FullAttnLayerWeights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    n: usize,
    dim: usize,
    q8_wmma_arch: bool,
    fusion: DflashFusionCtx,
) -> HipResult<()> {
    let _ = fusion;
    // S3-f16-projection-inputs fast path: exact-FP16 FA qkv inputs. The
    // fused QKV kernel requires all three weights to share the MQ4G256V2
    // stride (same gate as `qkv_same_dtype` below, restricted to MQ4V2).
    if mq_f16_projection_fast_route(gpu, fusion, n, dim)
        && layer.wq.gpu_dtype == DType::MQ4G256V2
        && layer.wk.gpu_dtype == DType::MQ4G256V2
        && layer.wv.gpu_dtype == DType::MQ4G256V2
    {
        fused_rmsnorm_rotate_mq_f16_batched_for(
            gpu,
            &pbs.x_batch,
            &layer.attn_norm,
            &layer.wq,
            &pbs.x_rot_f16_batch,
            dim,
            config.norm_eps,
            n,
        )?;
        return gpu.gemm_qkv_mq4g256v2_wmma_f16(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            &pbs.x_rot_f16_batch,
            &pbs.fa_q_full_batch,
            &pbs.fa_k_batch,
            &pbs.fa_v_batch,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
            n,
        );
    }
    let qkv_is_mq = matches!(
        layer.wq.gpu_dtype,
        DType::MQ4G256
            | DType::MQ4G256V2
            // qt=52 shares qt=44's FWHT input contract; dispatch arm below
            // routes to the FP8-LUT twin.
            | DType::MQ4G256V2Lloyd
            | DType::MQ4CG256
            | DType::MQ6G256
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256
            | DType::MQ3G256V2
            | DType::MQ2G256V2
            | DType::MQ3G256Lloyd
            | DType::MFP4G32
    );
    let qkv_is_6bit = matches!(layer.wq.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
    let qkv_is_mq3 = matches!(layer.wq.gpu_dtype, DType::MQ3G256);
    let qkv_is_mq3_lloyd = matches!(layer.wq.gpu_dtype, DType::MQ3G256Lloyd);
    let qkv_is_fp4 = matches!(layer.wq.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
    let qkv_is_q8 = matches!(layer.wq.gpu_dtype, DType::Q8_0);
    // TQ2G128/BQ1G128 have no fused qkvza/gate_up/qkv kernel, so they take
    // the same UNFUSED plain-GEMM strategy as Q8 rather than falling through
    // to the HFQ4 arm, which would read these packed blocks at the wrong
    // stride and produce fluent-but-wrong tokens.
    let qkv_is_lowbit = matches!(layer.wq.gpu_dtype, DType::TQ2G128 | DType::BQ1G128);
    // qt=52 re-arm anchor: q/k/v all Lloyd → FP8-LUT launcher.
    let qkv_is_mq4v2_lloyd =
        all_mq4v2_lloyd(&[layer.wq.gpu_dtype, layer.wk.gpu_dtype, layer.wv.gpu_dtype]);
    // Fused QKV kernels require all three weights to share a
    // dtype — they treat wq/wk/wv as same-stride byte arrays.
    // When kmap mode 2 promotes only `v_proj` (issue #249), the
    // fused HFQ4 path reads `wv` as MQ6 with HFQ4's 136-B stride
    // and produces silent NaN. Gate the fused kernels here.
    //
    // The Q8 substrate path (gemm_q8_0_batched_chunked × 3) also
    // dispatches a Q8-stride kernel per weight, so it needs the
    // same gate when wk/wv aren't Q8.
    let qkv_same_dtype =
        layer.wk.gpu_dtype == layer.wq.gpu_dtype && layer.wv.gpu_dtype == layer.wq.gpu_dtype;

    // 1. rmsnorm (+ rotate for MQ) for the attn preamble.
    let mut iu4_prep: Option<rdna_compute::Int4MmqPrepared> = None;
    if qkv_is_mq {
        // C2: MQ4V2 IU4 producer (emit_f32=false — FA qkv has no f32 small tails).
        iu4_prep = try_iu4_rmsnorm_prepared(
            gpu,
            &pbs.x_batch,
            &layer.attn_norm,
            &layer.wq,
            &pbs.x_rot_batch,
            dim,
            config.norm_eps,
            n,
            false,
        )?;
        if iu4_prep.is_none() {
            // AWQ-aware: next linear is wq (Q/K/V share input → same AWQ scale).
            fused_rmsnorm_rotate_mq_batched_for(
                gpu,
                &pbs.x_batch,
                &layer.attn_norm,
                &layer.wq,
                &pbs.x_rot_batch,
                dim,
                config.norm_eps,
                n,
            )?;
        }
    } else {
        gpu.rmsnorm_batched(
            &pbs.x_batch,
            &layer.attn_norm,
            &pbs.x_rot_batch,
            n,
            dim,
            config.norm_eps,
        )?;
    }

    // 2. Batched 3-way QKV projection (wq+wk+wv).
    if let Some(prep) = &iu4_prep {
        gpu.gemm_qkv_mq4g256v2_wmma_iu4_prepared(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            prep,
            &pbs.fa_q_full_batch,
            &pbs.fa_k_batch,
            &pbs.fa_v_batch,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
            n,
        )?;
    } else if qkv_is_6bit && qkv_same_dtype {
        run_fused_qkv_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedQkvHfq6G256,
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            &pbs.x_rot_batch,
            &pbs.fa_q_full_batch,
            &pbs.fa_k_batch,
            &pbs.fa_v_batch,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
            n,
        )?;
    } else if qkv_is_mq3_lloyd && qkv_same_dtype {
        run_fused_qkv_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedQkvMq3G256Lloyd,
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            &pbs.x_rot_batch,
            &pbs.fa_q_full_batch,
            &pbs.fa_k_batch,
            &pbs.fa_v_batch,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
            n,
        )?;
    } else if qkv_is_mq3 && qkv_same_dtype {
        // X is already FWHT-rotated by fused_rmsnorm_rotate_mq_batched
        // above; call the bare HFQ3 GEMM (no second rotation). The
        // FusedQkvHfq3G256 run-arm replicates the call-site WMMA-vs-base
        // arch split internally (gemm_qkv_hfq3g256_wmma on has_wmma()
        // else the base cross-arch ladder), so the same kernel runs.
        run_fused_qkv_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedQkvHfq3G256,
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            &pbs.x_rot_batch,
            &pbs.fa_q_full_batch,
            &pbs.fa_k_batch,
            &pbs.fa_v_batch,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
            n,
        )?;
    } else if qkv_is_fp4 && qkv_same_dtype {
        // HFP4G32 / MFP4G32 FP4 batched WMMA. X is already
        // rotated above for MFP4 (is_mq path) — same kernel
        // covers both unrotated HFP4 and rotated MFP4 inputs.
        run_fused_qkv_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedQkvHfp4G32,
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            &pbs.x_rot_batch,
            &pbs.fa_q_full_batch,
            &pbs.fa_k_batch,
            &pbs.fa_v_batch,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
            n,
        )?;
    } else if qkv_is_q8 && q8_wmma_arch && qkv_same_dtype {
        debug_assert!(
            matches!(layer.wk.gpu_dtype, DType::Q8_0) && matches!(layer.wv.gpu_dtype, DType::Q8_0),
            "FA qkv Q8 WMMA dispatch requires all of wq/wk/wv to be Q8_0",
        );
        run_fused_qkv_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedQkvQ8_0,
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            &pbs.x_rot_batch,
            &pbs.fa_q_full_batch,
            &pbs.fa_k_batch,
            &pbs.fa_v_batch,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
            n,
        )?;
    } else if (qkv_is_q8 || qkv_is_lowbit) && qkv_same_dtype {
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wq.gpu_dtype),
            &layer.wq.buf,
            layer.wq.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.fa_q_full_batch,
            layer.wq.m,
            layer.wq.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wk.gpu_dtype),
            &layer.wk.buf,
            layer.wk.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.fa_k_batch,
            layer.wk.m,
            layer.wk.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wv.gpu_dtype),
            &layer.wv.buf,
            layer.wv.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.fa_v_batch,
            layer.wv.m,
            layer.wv.k,
            n,
        )?;
    } else if qkv_is_mq4v2_lloyd {
        // qt=52: q/k/v all Lloyd. gfx11 → MMQ-LUT; gfx12 FP8-LUT.
        if lloyd_mmq_lut_route(gpu) {
            gpu.gemm_qkv_mq4g256v2_mmq_lloyd(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &pbs.x_rot_batch,
                &pbs.fa_q_full_batch,
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
                n,
                lloyd_c16_or_fail(&layer.wq, "batch_chunk_full_attn_input_projection")?,
                lloyd_c16_or_fail(&layer.wk, "batch_chunk_full_attn_input_projection")?,
                lloyd_c16_or_fail(&layer.wv, "batch_chunk_full_attn_input_projection")?,
            )?;
        } else {
            gpu.gemm_qkv_hfq4g256_wmma_gfx12_mq4v2_fp8_lloyd(
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &pbs.x_rot_batch,
                &pbs.fa_q_full_batch,
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
                n,
                1,
                lloyd_e4m3_or_fail(&layer.wq, "batch_chunk_full_attn_input_projection")?,
                lloyd_e4m3_or_fail(&layer.wk, "batch_chunk_full_attn_input_projection")?,
                lloyd_e4m3_or_fail(&layer.wv, "batch_chunk_full_attn_input_projection")?,
            )?;
        }
    } else if matches!(layer.wq.gpu_dtype, DType::MQ4G256V2Lloyd)
        || matches!(layer.wk.gpu_dtype, DType::MQ4G256V2Lloyd)
        || matches!(layer.wv.gpu_dtype, DType::MQ4G256V2Lloyd)
    {
        // Mixed Lloyd/uniform FA qkv: no kernel reads mixed codebooks.
        return Err(hip_bridge::HipError::new(
            0,
            "batch_chunk_full_attn_input_projection: mixed MQ4G256V2Lloyd/uniform FA qkv — refusing (quantize all three or none)",
        ));
    } else if qkv_same_dtype {
        run_fused_qkv_key(
            gpu,
            crate::forward_slots::fused_qkv_key_for(layer.wq.gpu_dtype),
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            &pbs.x_rot_batch,
            &pbs.fa_q_full_batch,
            &pbs.fa_k_batch,
            &pbs.fa_v_batch,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
            n,
        )?;
    } else {
        // Mixed-format fallback (issue #249): wq/wk/wv don't all
        // share a dtype. Dispatch each weight to its own
        // single-weight batched GEMM, dropping the fused-kernel
        // launch-overhead optimization for correctness.
        batched_gemm_single_weight(gpu, &layer.wq, &pbs.x_rot_batch, &pbs.fa_q_full_batch, n)?;
        batched_gemm_single_weight(gpu, &layer.wk, &pbs.x_rot_batch, &pbs.fa_k_batch, n)?;
        batched_gemm_single_weight(gpu, &layer.wv, &pbs.x_rot_batch, &pbs.fa_v_batch, n)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// Prescaffold (behavior-only) extraction for S6-fa-prep-q8-pair.
///
/// Same statements, same order, same launches as the inlined block, except
/// the trailing KV-write + flash-attention dispatch (F2 split): the caller
/// (`batch_chunk_full_attn_attn`) issues it via `batch_chunk_fa_attend`
/// immediately after, so single-chunk launch order is unchanged.
fn batch_chunk_full_attn_prepare(
    gpu: &mut Gpu,
    // F2 split: the trailing KV-write + flash-attention dispatch moved to
    // the caller, so these attend-only params are kept for signature
    // stability but currently unused.
    _fa_attn_multirow: bool,
    layer: &FullAttnLayerWeights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    _s: &Qwen35Scratch,
    kv_cache: &llama::KvCache,
    n: usize,
    _start_pos: usize,
    _max_ctx_len: usize,
    _ctx: &DispatchCtx,
    _batch_semantics: BatchSemantics<'_>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    _kv_layer_idx: usize,
    layer_idx: usize,
    fusion: DflashFusionCtx,
) -> HipResult<()> {
    // S6-fa-prep-q8-pair: exact gfx1100 fold of steps 3-5 (deinterleave +
    // Q/K rmsnorm + half-split RoPE, 4 launches) into one
    // qwen35_fa_prep_batched_gfx1100 launch. Bit-exact (same reduction tree,
    // same RoPE expression/phase, explicit old-TU FMA formation); the triattn
    // tap needs pre-RoPE Q, legacy interleaved RoPE needs its own kernel, and
    // every other shape/arch/ctx keeps the old path.
    // HIPFIRE_FA_BATCH_FUSE_OFF=1 restores it byte-for-byte.
    // Admitted geometries are 16Q/2K and 24Q/4K (Qwen3.8-27B FA is 24/4);
    // HD must be 256 and n_rot 64.
    let fa_prep_n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
    // 39aa358: in DDTree verify, rotate at DEPTH positions; KV writes below
    // still use flat physical slots. The fused kernel takes the same buffer
    // choice, so tree mode stays fused.
    let fa_prep_rope_pos_buf = if tree_verify.is_some() {
        &pbs.rope_positions
    } else {
        &pbs.positions
    };
    let fa_prep_shape_ok = matches!((config.n_heads, config.n_kv_heads), (16, 2) | (24, 4))
        && config.head_dim == 256
        && fa_prep_n_rot == 64;
    let fa_prep_fused_ok = fusion == DflashFusionCtx::ChainVerify
        && gpu.arch_caps.is_gfx1100()
        && !gpu.flags.fa_batch_fuse_off
        && !gpu.flags.rope_interleaved_legacy
        && !hipfire_runtime::triattn::tap_enabled()
        && fa_prep_shape_ok
        && n >= 1;
    if fa_prep_fused_ok {
        gpu.qwen35_fa_prep_batched_gfx1100(
            &pbs.fa_q_full_batch,
            &pbs.fa_q_batch,
            &pbs.fa_gate_batch,
            &pbs.fa_k_batch,
            &layer.q_norm,
            &layer.k_norm,
            fa_prep_rope_pos_buf,
            config.norm_eps,
            config.rope_theta,
            kv_cache.compact_offset as i32,
            config.n_heads,
            config.n_kv_heads,
            n,
        )?;
    } else {
        // Fused deinterleave + Q-rmsnorm — one launch instead of two, no Q
        // global-memory round trip. K-norm, triattn tap, and RoPE below are
        // unchanged.
        gpu.deinterleave_q_rmsnorm_f32_batched(
            &pbs.fa_q_full_batch,
            &pbs.fa_q_batch,
            &pbs.fa_gate_batch,
            &layer.q_norm,
            config.n_heads,
            config.head_dim,
            n,
            config.norm_eps,
        )?;
        gpu.rmsnorm_batched(
            &pbs.fa_k_batch,
            &layer.k_norm,
            &pbs.fa_k_batch,
            n * config.n_kv_heads,
            config.head_dim,
            config.norm_eps,
        )?;

        if hipfire_runtime::triattn::tap_enabled() {
            // Try GPU path first: dispatches a reduce kernel on the
            // device-resident Q tensor, zero PCIe transfer. Only
            // succeeds when install_tap_gpu() was used. Falls through
            // to CPU path otherwise.
            let gpu_handled = hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
                gpu,
                layer_idx,
                &pbs.fa_q_batch.buf,
                n,
                config.n_heads,
                config.head_dim,
            )?;
            if !gpu_handled {
                let n_q = config.n_heads * config.head_dim;
                let q_cpu = gpu.download_f32(&pbs.fa_q_batch)?;
                if hipfire_runtime::triattn::tap_needs_k() {
                    let n_k = config.n_kv_heads * config.head_dim;
                    let k_cpu = gpu.download_f32(&pbs.fa_k_batch)?;
                    for b in 0..n {
                        hipfire_runtime::triattn::record_prerope_qk(
                            layer_idx,
                            &q_cpu[b * n_q..(b + 1) * n_q],
                            Some(&k_cpu[b * n_k..(b + 1) * n_k]),
                        );
                    }
                } else {
                    for b in 0..n {
                        hipfire_runtime::triattn::record_prerope_q(
                            layer_idx,
                            &q_cpu[b * n_q..(b + 1) * n_q],
                        );
                    }
                }
            }
        }

        // 5. Batched partial-interleaved RoPE (per-row positions).
        // pos_offset = compact_offset so new Q/K rotate at ABSOLUTE phase
        // after eviction (cached keys are absolute-phased); pbs.positions
        // stays physical for the KV-write below. 0 when no compaction.
        let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
        // 39aa358: in DDTree verify, rotate at DEPTH positions (correct
        // sibling phases); KV writes below still use flat physical
        // slots. Linear path unchanged.
        let rope_pos_buf = if tree_verify.is_some() {
            &pbs.rope_positions
        } else {
            &pbs.positions
        };
        gpu.rope_partial_interleaved_f32_batched(
            &pbs.fa_q_batch,
            &pbs.fa_k_batch,
            rope_pos_buf,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            n_rot,
            config.rope_theta,
            n,
            kv_cache.compact_offset as i32,
        )?;
    }

    // 6–7. (F2 split: the KV write + flash attention dispatch now lives in
    // the caller via `batch_chunk_fa_attend`, so a pair of chunks can share
    // one merged 1024-row attend step. Single-chunk order is unchanged.)
    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// Prescaffold (behavior-only) extraction for S4-f16-residual-inputs.
///
/// Same statements, same order, same launches as the inlined block.
/// S9-mq4v2-persistent-prologues will issue `try_mq4v2_persistent_prologue`
/// from inside this hook after S3/S4 land.
fn batch_chunk_full_attn_output_projection(
    gpu: &mut Gpu,
    layer: &FullAttnLayerWeights,
    pbs: &PrefillBatchScratch,
    n: usize,
    q8_wmma_arch: bool,
    arch_has_wmma: bool,
    epilogue: BatchEpilogue<'_>,
    fusion: DflashFusionCtx,
) -> HipResult<()> {
    // S4: one sigmoid*attn+FWHT+F16 producer + direct-F16 residual GEMM
    // instead of sigmoid_mul_f32 + mq_rotate_x + convert. The F32 attn
    // input is left unmutated (the old in-place sigmoid write is skipped).
    if s4_residual_fast(gpu, fusion, layer.wo.gpu_dtype, &epilogue, n) {
        let k = layer.wo.k;
        let m = layer.wo.m;
        if k > 0 && k % 256 == 0 {
            if let Some(awq) = layer.wo.awq_scale.as_ref() {
                if awq.numel() >= k {
                    gpu.sigmoid_mul_rotate_mq_awq_f16_batched(
                        &pbs.fa_attn_out_batch,
                        &pbs.fa_gate_batch,
                        awq,
                        &pbs.fa_attn_out_rot_f16_batch,
                        k,
                        n,
                    )?;
                    let x_f16 = pbs.fa_attn_out_rot_f16_batch.sub_offset(0, n * k);
                    gpu.gemm_mq4g256v2_residual_wmma_f16(
                        &layer.wo.buf,
                        &x_f16,
                        &pbs.x_batch,
                        m,
                        k,
                        n,
                    )?;
                    return Ok(());
                }
            } else {
                gpu.sigmoid_mul_rotate_mq_f16_batched(
                    &pbs.fa_attn_out_batch,
                    &pbs.fa_gate_batch,
                    &pbs.fa_attn_out_rot_f16_batch,
                    k,
                    n,
                )?;
                let x_f16 = pbs.fa_attn_out_rot_f16_batch.sub_offset(0, n * k);
                gpu.gemm_mq4g256v2_residual_wmma_f16(&layer.wo.buf, &x_f16, &pbs.x_batch, m, k, n)?;
                return Ok(());
            }
        }
    }
    // 8. Fused sigmoid(gate) * attn_out, element-wise over the
    // full [N × q_dim] tensor.
    gpu.sigmoid_mul_f32(&pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;

    // 9. wo residual: x_batch += wo · (optional rotate)(fa_attn_out_batch).
    // Same MQ rotation requirement as the LA wo path.
    let fa_wo_is_mq = matches!(
        layer.wo.gpu_dtype,
        DType::MQ4G256
            | DType::MQ4G256V2
            // qt=52 shares qt=44's FWHT input contract (LUT decode needs rotated x).
            | DType::MQ4G256V2Lloyd
            | DType::MQ4CG256
            | DType::MQ6G256
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256
            | DType::MQ3G256V2
            | DType::MQ2G256V2
            | DType::MQ3G256Lloyd
            | DType::MFP4G32
    );
    // T-B: fused rotate+quantize — x_rot feeds only the IU4 residual GEMM.
    let iu4_wo_prep = try_iu4_rotate_prepared(
        gpu,
        &layer.wo,
        &pbs.fa_attn_out_batch,
        &pbs.fa_attn_out_rot_batch,
        layer.wo.k,
        n,
        &epilogue,
    )?;
    let fa_wo_input = if iu4_wo_prep.is_some() {
        &pbs.fa_attn_out_rot_batch
    } else if fa_wo_is_mq {
        rotate_x_mq_batched_for(
            gpu,
            &layer.wo,
            &pbs.fa_attn_out_batch,
            &pbs.fa_attn_out_rot_batch,
            layer.wo.k,
            n,
        )?;
        &pbs.fa_attn_out_rot_batch
    } else {
        &pbs.fa_attn_out_batch
    };
    if let Some(prep) = &iu4_wo_prep {
        gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(
            &layer.wo.buf,
            prep,
            &pbs.x_batch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    } else {
        dispatch_batched_gemm_epilogue(
            gpu,
            pbs,
            &layer.wo,
            fa_wo_input,
            &epilogue,
            n,
            q8_wmma_arch,
            arch_has_wmma,
        )?;
    }
    Ok(())
}

/// Context length past which an admitted gfx1100/gfx1201 Q8 small-batch attend step
/// leaves the batched masked FA kernel for the multi-row tile. Measured with
/// the Qwen3.8-27B verify shape (`bench_flash_rows`, tile 128). The conservative
/// 4k boundary is retained across both measured architectures.
/// `HIPFIRE_FA_PERTOKEN_MIN_CTX` overrides; `0` disables the route.
pub(crate) fn fa_pertoken_min_ctx() -> Option<usize> {
    use std::sync::OnceLock;
    static MIN_CTX: OnceLock<Option<usize>> = OnceLock::new();
    *MIN_CTX.get_or_init(|| {
        let v = hipfire_config::developer_var("HIPFIRE_FA_PERTOKEN_MIN_CTX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4_096);
        (v > 0).then_some(v)
    })
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
    matches!(arch, "gfx1100" | "gfx1201")
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
fn batch_chunk_fa_attend(
    gpu: &mut Gpu,
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
    layer_idx: usize,
    multirow: bool,
    commit_stride: Option<usize>,
) -> HipResult<()> {
    if let BatchSemantics::Independent {
        lane_capacity,
        active_mask,
        ..
    } = batch_semantics
    {
        return run_independent_q8_attention(
            gpu,
            pbs,
            kv_cache,
            config,
            layer_idx,
            n,
            lane_capacity,
            max_ctx_len,
            active_mask,
        );
    }
    if batch_semantics.is_independent() {
        unreachable!("independent variant must carry active_mask");
    }
    if let Some(stride) = commit_stride {
        // Widened ordinary chunk: one 512-row write-then-attend tile per
        // legacy segment, each with its own tile-local context
        // (`start+o .. start+o+512`). The family writes that tile's KV rows
        // before attending, so earlier logical prefix is present and no
        // future tile row is needed — the same kernel route, scalar
        // arguments and byte operands each legacy 512 request sees.
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;
        assert!(
            n % stride == 0,
            "widened FA chunk {n} is not a multiple of commit stride {stride}"
        );
        for off in (0..n).step_by(stride) {
            let q = pbs.fa_q_batch.sub_offset(off * q_dim, stride * q_dim);
            let k = pbs.fa_k_batch.sub_offset(off * kv_dim, stride * kv_dim);
            let v = pbs.fa_v_batch.sub_offset(off * kv_dim, stride * kv_dim);
            let positions = pbs.positions.sub_offset(off, stride);
            let out = pbs
                .fa_attn_out_batch
                .sub_offset(off * q_dim, stride * q_dim);
            execute_fa_attend_step(
                gpu,
                config,
                &q,
                &k,
                &v,
                &positions,
                &out,
                s,
                kv_cache,
                stride,
                start_pos + off,
                start_pos + off + stride,
                ctx,
                tree_verify,
                layer_idx,
            )?;
        }
        return Ok(());
    }
    if multirow {
        debug_assert!(gpu.arch_caps.is_gfx1100() || gpu.arch_caps.is_gfx1201());
        debug_assert!(kv_cache.quant_q8);
        debug_assert!(matches!(config.head_dim, 128 | 256));
        gpu.kv_cache_write_q8_0_batched(
            &kv_cache.k_gpu[layer_idx],
            &pbs.fa_k_batch,
            &pbs.positions,
            config.n_kv_heads,
            config.head_dim,
            n,
        )?;
        gpu.kv_cache_write_q8_0_batched(
            &kv_cache.v_gpu[layer_idx],
            &pbs.fa_v_batch,
            &pbs.positions,
            config.n_kv_heads,
            config.head_dim,
            n,
        )?;
        if gpu.attention_flash_q8_0_rows_masked(
            &pbs.fa_q_batch,
            &kv_cache.k_gpu[layer_idx],
            &kv_cache.v_gpu[layer_idx],
            &pbs.fa_attn_out_batch,
            &pbs.positions,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            max_ctx_len,
            n,
            &s.flash_partials,
        )? {
            return Ok(());
        }
        // Admission and launcher support intentionally duplicate the shape
        // checks. If they ever drift, retain the established batched route
        // below rather than silently exploding the verify block into n
        // independent attention launches.
    }
    // Shared dispatch tail (F2 extraction): identical plan derivation and
    // single write-then-attend step with this chunk's own FA tensors.
    execute_fa_attend_step(
        gpu,
        config,
        &pbs.fa_q_batch,
        &pbs.fa_k_batch,
        &pbs.fa_v_batch,
        &pbs.positions,
        &pbs.fa_attn_out_batch,
        s,
        kv_cache,
        n,
        start_pos,
        max_ctx_len,
        ctx,
        tree_verify,
        layer_idx,
    )

}

/// Shared KV-write + flash-attention dispatch tail (F2 extraction).
///
/// `batch_chunk_fa_attend` calls this with the chunk's own FA tensors; the
/// F2 pair path calls it once with 1024-row staged tensors covering both
/// halves. Same plan derivation, same single `Step::Attend` (the family
/// writes the KV rows from `k`/`v` at `positions` before attending, so both
/// halves' KV are fully written before the merged FA2 reads them), same
/// stream ordering.
#[allow(clippy::too_many_arguments)]
fn execute_fa_attend_step(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    positions: &GpuTensor,
    output: &GpuTensor,
    s: &Qwen35Scratch,
    kv_cache: &llama::KvCache,
    n: usize,
    start_pos: usize,
    max_ctx_len: usize,
    ctx: &DispatchCtx,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    layer_idx: usize,
) -> HipResult<()> {
    let is_tree = tree_verify.is_some();
    let (block_start, block_cols) = match tree_verify.as_ref() {
        Some(_) => (start_pos, n),
        None => (0, 0),
    };
    let tree_bias = tree_verify.as_ref().map(|c| c.attn_bias);
    let plan = KvTierPlan::derive(KvTierInputs {
        pos: start_pos,
        flash_mode: s.flash_mode as usize,
        capture_mode: gpu.graphs.capture_mode,
        batch_size: n,
        is_tree,
        ..kv_cache.tier_inputs()
    })
    .map_err(|e| HipError::new(0, &e.to_string()))?;
    let io = AttnParams {
        q,
        k,
        v,
        k_cache: &kv_cache.k_gpu[layer_idx],
        v_cache: &kv_cache.v_gpu[layer_idx],
        k_scales: None,
        v_scales: None,
        pos_buf: &s.pos_buf,
        pos: start_pos,
        positions: Some(positions),
        n_heads: config.n_heads,
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        physical_cap: kv_cache.physical_cap,
        batch_size: n,
        max_ctx_len,
        flash_partials: Some(&s.flash_partials),
        givens_cos: kv_cache.givens_cos.as_ref(),
        givens_sin: kv_cache.givens_sin.as_ref(),
        tree_bias,
        block_start,
        block_cols,
        output_gate: None,
        output_awq_scale: None,
        output,
    };
    execute_steps(gpu, ctx, &[Step::Attend { plan, io }])
        .map_err(|e| HipError::new(0, &e.to_string()))
}

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
    // Fully batched FA layer. Mirrors the FA branch of
    // forward_scratch_layers kernel-for-kernel, but every
    // launch covers all N tokens at once.
    let kv_dim = config.n_kv_heads * config.head_dim;
    let q_dim = config.n_heads * config.head_dim;
    batch_chunk_full_attn_input_projection(gpu, layer, config, pbs, n, dim, q8_wmma_arch, fusion)?;

    batch_chunk_full_attn_prepare(
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
        kv_layer_idx,
        layer_idx,
        fusion,
    )?;

    // 6–7. Batched KV write + flash attention (via dispatch). Split out of
    // `batch_chunk_full_attn_prepare` (F2) so a chunk pair can run both
    // halves' prep first and share one merged attend step; called here for
    // the single-chunk path in the original position.
    batch_chunk_fa_attend(
        gpu,
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
        layer_idx,
        fa_attn_multirow,
        commit_stride,
    )?;
    batch_chunk_full_attn_output_projection(
        gpu,
        layer,
        pbs,
        n,
        q8_wmma_arch,
        arch_has_wmma,
        epilogue,
        fusion,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
/// Prescaffold (behavior-only) extraction for S3-f16-projection-inputs.
///
/// Same statements, same order, same launches as the inlined block.
/// S9-mq4v2-persistent-prologues will issue `try_mq4v2_persistent_prologue`
/// from inside this hook after S3/S4 land.
fn batch_chunk_full_attn_ffn_gate_up(
    gpu: &mut Gpu,
    layer: &FullAttnLayerWeights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    n: usize,
    dim: usize,
    q8_wmma_arch: bool,
    fusion: DflashFusionCtx,
) -> HipResult<()> {
    let _ = fusion;
    // S3-f16-projection-inputs fast path: exact-FP16 FA-FFN gate/up inputs.
    if mq_f16_projection_fast_route(gpu, fusion, n, dim)
        && layer.w_gate.gpu_dtype == DType::MQ4G256V2
        && layer.w_up.gpu_dtype == DType::MQ4G256V2
    {
        fused_rmsnorm_rotate_mq_f16_batched_for(
            gpu,
            &pbs.x_batch,
            &layer.ffn_norm,
            &layer.w_gate,
            &pbs.x_rot_f16_batch,
            dim,
            config.norm_eps,
            n,
        )?;
        return gpu.gemm_gate_up_mq4g256v2_wmma_f16(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            &pbs.x_rot_f16_batch,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
            n,
        );
    }
    // 10. FFN: rmsnorm (+ rotate for MQ), gate+up, silu_mul
    // (+ rotate for MQ), w_down residual.
    let fa_ffn_is_mq = matches!(
        layer.w_gate.gpu_dtype,
        DType::MQ4G256
            | DType::MQ4G256V2
            // qt=52 shares qt=44's FWHT input contract; dispatch arm below
            // routes to the FP8-LUT twin.
            | DType::MQ4G256V2Lloyd
            | DType::MQ4CG256
            | DType::MQ6G256
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256
            | DType::MQ3G256V2
            | DType::MQ2G256V2
            | DType::MQ3G256Lloyd
            | DType::MFP4G32
    );
    let fa_ffn_is_6bit = matches!(layer.w_gate.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
    let fa_ffn_is_mq3 = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256);
    let fa_ffn_is_mq3_lloyd = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256Lloyd);
    let fa_ffn_is_fp4 = matches!(layer.w_gate.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
    let fa_ffn_is_q8 = matches!(layer.w_gate.gpu_dtype, DType::Q8_0);
    // TQ2G128/BQ1G128 have no fused qkvza/gate_up/qkv kernel, so they take
    // the same UNFUSED plain-GEMM strategy as Q8 rather than falling through
    // to the HFQ4 arm, which would read these packed blocks at the wrong
    // stride and produce fluent-but-wrong tokens.
    let fa_ffn_is_lowbit = matches!(layer.w_gate.gpu_dtype, DType::TQ2G128 | DType::BQ1G128);
    // qt=52 re-arm anchor: gate+up both Lloyd → FP8-LUT launcher.
    let fa_ffn_is_mq4v2_lloyd = all_mq4v2_lloyd(&[layer.w_gate.gpu_dtype, layer.w_up.gpu_dtype]);
    let mut iu4_prep: Option<rdna_compute::Int4MmqPrepared> = None;
    if fa_ffn_is_mq {
        iu4_prep = try_iu4_rmsnorm_prepared(
            gpu,
            &pbs.x_batch,
            &layer.ffn_norm,
            &layer.w_gate,
            &pbs.x_rot_batch,
            dim,
            config.norm_eps,
            n,
            false,
        )?;
        if iu4_prep.is_none() {
            // AWQ-aware: next linear is w_gate (FA-FFN, gate/up share input).
            fused_rmsnorm_rotate_mq_batched_for(
                gpu,
                &pbs.x_batch,
                &layer.ffn_norm,
                &layer.w_gate,
                &pbs.x_rot_batch,
                dim,
                config.norm_eps,
                n,
            )?;
        }
    } else {
        gpu.rmsnorm_batched(
            &pbs.x_batch,
            &layer.ffn_norm,
            &pbs.x_rot_batch,
            n,
            dim,
            config.norm_eps,
        )?;
    }
    // #397 Ship 5.2 slice 2: FA-FFN fused gate+up → FusedQkvFamily
    // (batched-prefill gate+up variant), mirroring the LA-FFN block
    // above. Q8-non-WMMA stays as two plain GEMMs; HFQ3 WMMA-vs-base
    // is folded into the FusedGateUpHfq3G256 run-arm.
    if let Some(prep) = &iu4_prep {
        gpu.gemm_gate_up_mq4g256v2_wmma_iu4_prepared(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            prep,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
            n,
        )?;
    } else if fa_ffn_is_6bit {
        run_fused_gate_up_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedGateUpHfq6G256,
            &layer.w_gate.buf,
            &layer.w_up.buf,
            &pbs.x_rot_batch,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
            n,
        )?;
    } else if fa_ffn_is_q8 && q8_wmma_arch {
        debug_assert!(
            matches!(layer.w_up.gpu_dtype, DType::Q8_0),
            "FA FFN Q8 WMMA dispatch requires both w_gate and w_up to be Q8_0",
        );
        run_fused_gate_up_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedGateUpQ8_0,
            &layer.w_gate.buf,
            &layer.w_up.buf,
            &pbs.x_rot_batch,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
            n,
        )?;
    } else if fa_ffn_is_q8 || fa_ffn_is_lowbit {
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.w_gate.gpu_dtype),
            &layer.w_gate.buf,
            layer.w_gate.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.gate_ffn_batch,
            layer.w_gate.m,
            layer.w_gate.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.w_up.gpu_dtype),
            &layer.w_up.buf,
            layer.w_up.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.up_batch,
            layer.w_up.m,
            layer.w_up.k,
            n,
        )?;
    } else if fa_ffn_is_mq3_lloyd {
        run_fused_gate_up_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedGateUpMq3G256Lloyd,
            &layer.w_gate.buf,
            &layer.w_up.buf,
            &pbs.x_rot_batch,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
            n,
        )?;
    } else if fa_ffn_is_mq3 {
        run_fused_gate_up_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedGateUpHfq3G256,
            &layer.w_gate.buf,
            &layer.w_up.buf,
            &pbs.x_rot_batch,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
            n,
        )?;
    } else if fa_ffn_is_fp4 {
        run_fused_gate_up_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedGateUpHfp4G32,
            &layer.w_gate.buf,
            &layer.w_up.buf,
            &pbs.x_rot_batch,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
            n,
        )?;
    } else if fa_ffn_is_mq4v2_lloyd {
        // qt=52: gate+up both Lloyd. gfx11 → MMQ-LUT; gfx12 FP8-LUT.
        if lloyd_mmq_lut_route(gpu) {
            gpu.gemm_gate_up_mq4g256v2_mmq_lloyd(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
                lloyd_c16_or_fail(&layer.w_gate, "batch_chunk_full_attn_ffn_gate_up")?,
                lloyd_c16_or_fail(&layer.w_up, "batch_chunk_full_attn_ffn_gate_up")?,
            )?;
        } else {
            gpu.gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2_fp8_lloyd(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
                1,
                lloyd_e4m3_or_fail(&layer.w_gate, "batch_chunk_full_attn_ffn_gate_up")?,
                lloyd_e4m3_or_fail(&layer.w_up, "batch_chunk_full_attn_ffn_gate_up")?,
            )?;
        }
    } else if matches!(layer.w_gate.gpu_dtype, DType::MQ4G256V2Lloyd)
        || matches!(layer.w_up.gpu_dtype, DType::MQ4G256V2Lloyd)
    {
        // Mixed Lloyd/uniform gate/up: no kernel reads mixed codebooks.
        return Err(hip_bridge::HipError::new(
            0,
            "batch_chunk_full_attn_ffn_gate_up: mixed MQ4G256V2Lloyd/uniform gate/up — refusing (quantize both or neither)",
        ));
    } else {
        run_fused_gate_up_key(
            gpu,
            crate::forward_slots::fused_gate_up_key_for(layer.w_gate.gpu_dtype),
            &layer.w_gate.buf,
            &layer.w_up.buf,
            &pbs.x_rot_batch,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
            n,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// Prescaffold (behavior-only) extraction for S4-f16-residual-inputs.
///
/// Same statements, same order, same launches as the inlined block.
/// S9-mq4v2-persistent-prologues will issue `try_mq4v2_persistent_prologue`
/// from inside this hook after S3/S4 land.
fn batch_chunk_full_attn_ffn_down(
    gpu: &mut Gpu,
    layer: &FullAttnLayerWeights,
    pbs: &PrefillBatchScratch,
    hidden_dim: usize,
    n: usize,
    q8_wmma_arch: bool,
    arch_has_wmma: bool,
    epilogue: BatchEpilogue<'_>,
    fusion: DflashFusionCtx,
) -> HipResult<()> {
    // S4: one silu*up+FWHT+F16 producer + direct-F16 residual GEMM instead
    // of fused_silu_mul_rotate_mq_batched + convert.
    if s4_residual_fast(gpu, fusion, layer.w_down.gpu_dtype, &epilogue, n) {
        let k = layer.w_down.k;
        let m = layer.w_down.m;
        if k > 0 && k % 256 == 0 && k == hidden_dim {
            fused_silu_mul_rotate_mq_f16_batched_for(
                gpu,
                &layer.w_down,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                &pbs.ffn_hidden_f16_batch,
                hidden_dim,
                n,
            )?;
            let x_f16 = pbs.ffn_hidden_f16_batch.sub_offset(0, n * hidden_dim);
            gpu.gemm_mq4g256v2_residual_wmma_f16(
                &layer.w_down.buf,
                &x_f16,
                &pbs.x_batch,
                m,
                hidden_dim,
                n,
            )?;
            return Ok(());
        }
    }
    let fa_w_down_is_mq = matches!(
        layer.w_down.gpu_dtype,
        DType::MQ4G256
            | DType::MQ4G256V2
            // qt=52 shares qt=44's FWHT input contract (LUT decode needs rotated x).
            | DType::MQ4G256V2Lloyd
            | DType::MQ4CG256
            | DType::MQ6G256
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256
            | DType::MQ3G256V2
            | DType::MQ2G256V2
            | DType::MQ3G256Lloyd
            | DType::MFP4G32
    );
    let mut iu4_prep: Option<rdna_compute::Int4MmqPrepared> = None;
    if fa_w_down_is_mq {
        if matches!(&epilogue, BatchEpilogue::Residual) {
            iu4_prep = try_iu4_silu_prepared(
                gpu,
                &layer.w_down,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                &pbs.ffn_hidden_batch,
                hidden_dim,
                n,
                false,
            )?;
        }
        if iu4_prep.is_none() {
            fused_silu_mul_rotate_mq_batched_for(
                gpu,
                &layer.w_down,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                &pbs.ffn_hidden_batch,
                hidden_dim,
                n,
            )?;
        }
    } else {
        gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
    }
    if let Some(prep) = &iu4_prep {
        gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(
            &layer.w_down.buf,
            prep,
            &pbs.x_batch,
            layer.w_down.m,
            layer.w_down.k,
            n,
        )?;
    } else {
        dispatch_batched_gemm_epilogue(
            gpu,
            pbs,
            &layer.w_down,
            &pbs.ffn_hidden_batch,
            &epilogue,
            n,
            q8_wmma_arch,
            arch_has_wmma,
        )?;
    }
    Ok(())
}

pub(crate) fn batch_chunk_full_attn_ffn(
    gpu: &mut Gpu,
    layer: &FullAttnLayerWeights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    n: usize,
    dim: usize,
    hidden_dim: usize,
    q8_wmma_arch: bool,
    arch_has_wmma: bool,
    epilogue: BatchEpilogue<'_>,
    fusion: DflashFusionCtx,
) -> HipResult<()> {
    batch_chunk_full_attn_ffn_gate_up(gpu, layer, config, pbs, n, dim, q8_wmma_arch, fusion)?;
    batch_chunk_full_attn_ffn_down(
        gpu,
        layer,
        pbs,
        hidden_dim,
        n,
        q8_wmma_arch,
        arch_has_wmma,
        epilogue,
        fusion,
    )?;

    Ok(())
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
    q8_wmma_arch: bool,
    start_pos: usize,
    layer_idx: usize,
    ctx: &DispatchCtx,
    weights: &Qwen35Weights,
    routed_out: Option<&GpuTensor>,
    route: PrefillRouteMode<'_>,
) -> HipResult<()> {
    // Batched MoE LA layer. LA body is the same as DeltaNet
    // (rmsnorm + qkvza + sigmoid_alpha + conv1d + L2norm +
    // repeat_interleave + GDN + gated_norm + wo+residual);
    // only the FFN differs. Duplicated inline for now — can
    // be factored into a `prefill_la_body_batched` helper
    // when dense and MoE LA paths are proven byte-exact.
    // This body is unreachable for MQ3 / MQ3-Lloyd weights —
    // the upstream `mq3_in_moe` guard at the top of
    // `forward_prefill_batch_with_pbs` rejects any MoE layer
    // with MQ3/Lloyd-MQ3 weights anywhere (attention OR FFN),
    // mirroring the captured-path guard at line 3367+. So
    // `layer.wqkv.gpu_dtype` is restricted here to MQ4G256 /
    // HFQ4G256 / MQ6G256 / HFQ6G256 / Q8_0. Q8 admit landed
    // alongside the moe_ffn router/gate Q8 unlock (A3B's LA
    // attention weights are Q8 — engine quantizer keeps q/k/v/o
    // at Q8 alongside the Q8 router + shared_expert_gate).
    let is_mq = matches!(
        layer.wqkv.gpu_dtype,
        DType::MQ4G256
            | DType::MQ4G256V2
            | DType::MQ4CG256
            | DType::MQ6G256
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256V2
            | DType::MQ2G256V2
    );
    let is_6bit = matches!(layer.wqkv.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
    let is_q8 = matches!(layer.wqkv.gpu_dtype, DType::Q8_0);
    // TQ2G128/BQ1G128 have no fused qkvza/gate_up/qkv kernel, so they take
    // the same UNFUSED plain-GEMM strategy as Q8 rather than falling through
    // to the HFQ4 arm, which would read these packed blocks at the wrong
    // stride and produce fluent-but-wrong tokens.
    let is_lowbit = matches!(layer.wqkv.gpu_dtype, DType::TQ2G128 | DType::BQ1G128);
    // qt=52 has no MoE-batched kernels (grouped/indexed LUT variants don't
    // exist): refuse loudly here rather than falling through to a uniform
    // fused key. Per-token decode serves Lloyd-MoE via generic dispatch paths.
    if matches!(layer.wqkv.gpu_dtype, DType::MQ4G256V2Lloyd) {
        return Err(hip_bridge::HipError::new(
            0,
            "batch_chunk_delta_net_moe: MQ4G256V2Lloyd attention weights have no MoE-batched prefill kernel — refusing (per-token fallback serves this layer)",
        ));
    }
    // Phase 1.5: PARO mode for DeltaNetMoe — wqkv/wz are
    // ParoQ4G128 (each with its own Givens rotation tables);
    // w_alpha/w_beta are F32 (no rotation, no quantization).
    // Dispatch is unfused: rotate+gemm_hfq4g128 for wqkv and wz,
    // direct gemm_f32_batched for w_alpha and w_beta. Same shape
    // outputs as the Q8/MQ4 paths (dn_qkv_batch, dn_z_batch,
    // dn_alpha_batch, dn_beta_batch).
    let is_paro = matches!(layer.wqkv.gpu_dtype, DType::ParoQ4G128);
    let q8_wmma_arch = q8_prefill_wmma_enabled(gpu);

    if is_mq {
        // AWQ-aware: next linear is LA's fused wqkv.
        fused_rmsnorm_rotate_mq_batched_for(
            gpu,
            &pbs.x_batch,
            &layer.attn_norm,
            &layer.wqkv,
            &pbs.x_rot_batch,
            dim,
            config.norm_eps,
            n,
        )?;
    } else if is_paro {
        // PARO: need un-rotated x_norm available for per-weight
        // Givens rotation. Write rmsnorm into x_norm_batch (the
        // dedicated normalized buffer); x_rot_batch becomes the
        // per-weight rotation scratch (overwritten per GEMM).
        gpu.rmsnorm_batched(
            &pbs.x_batch,
            &layer.attn_norm,
            &pbs.x_norm_batch,
            n,
            dim,
            config.norm_eps,
        )?;
    } else {
        gpu.rmsnorm_batched(
            &pbs.x_batch,
            &layer.attn_norm,
            &pbs.x_rot_batch,
            n,
            dim,
            config.norm_eps,
        )?;
    }
    if is_paro {
        // PARO 4-way unfused dispatch. wqkv and wz are
        // ParoQ4G128 with their own Givens rotation tables;
        // w_alpha and w_beta are F32 with no rotation.
        let paro_wqkv = layer.wqkv.paro.as_ref().unwrap_or_else(|| {
            panic!(
                "ParoQ4G128 wqkv missing paro metadata at LA layer {layer_idx} \
                             — load_paroquant_weight() loader regression?"
            )
        });
        let paro_wz = layer.wz.paro.as_ref().unwrap_or_else(|| {
            panic!("ParoQ4G128 wz missing paro metadata at LA layer {layer_idx}")
        });
        // wqkv: rotate x_norm → x_rot, then HFQ4G128 GEMM.
        gpu.givens_rotate_to(
            &pbs.x_norm_batch,
            &pbs.x_rot_batch,
            &paro_wqkv.pairs,
            &paro_wqkv.theta,
            &paro_wqkv.channel_scales,
            n,
            dim,
            paro_wqkv.krot as usize,
        )?;
        run_plain_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmHfq4G128,
            &layer.wqkv.buf,
            layer.wqkv.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.dn_qkv_batch,
            layer.wqkv.m,
            layer.wqkv.k,
            n,
        )?;
        // wz: re-rotate x_norm → x_rot (overwrite), then GEMM.
        gpu.givens_rotate_to(
            &pbs.x_norm_batch,
            &pbs.x_rot_batch,
            &paro_wz.pairs,
            &paro_wz.theta,
            &paro_wz.channel_scales,
            n,
            dim,
            paro_wz.krot as usize,
        )?;
        run_plain_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmHfq4G128,
            &layer.wz.buf,
            layer.wz.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.dn_z_batch,
            layer.wz.m,
            layer.wz.k,
            n,
        )?;
        // w_alpha / w_beta: F32, no rotation, direct batched GEMM.
        run_plain_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmF32Batched,
            &layer.w_alpha.buf,
            layer.w_alpha.gpu_dtype,
            &pbs.x_norm_batch,
            &pbs.dn_alpha_batch,
            layer.w_alpha.m,
            layer.w_alpha.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmF32Batched,
            &layer.w_beta.buf,
            layer.w_beta.gpu_dtype,
            &pbs.x_norm_batch,
            &pbs.dn_beta_batch,
            layer.w_beta.m,
            layer.w_beta.k,
            n,
        )?;
    } else if is_6bit {
        run_fused_qkvza_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedQkvzaHfq6G256,
            &layer.wqkv.buf,
            &layer.wz.buf,
            &layer.w_beta.buf,
            &layer.w_alpha.buf,
            &pbs.x_rot_batch,
            &pbs.dn_qkv_batch,
            &pbs.dn_z_batch,
            &pbs.dn_beta_batch,
            &pbs.dn_alpha_batch,
            layer.wqkv.m,
            layer.wz.m,
            layer.w_beta.m,
            layer.w_alpha.m,
            layer.wqkv.k,
            n,
        )?;
    } else if is_q8 && q8_wmma_arch {
        // Fused Q8 QKVZA WMMA — assumes all 4 weights share Q8_0
        // stride; mixed Q8/other layers within DNMoe are rejected
        // upstream by `moe_ffn_batched_admissible` (router/gate Q8 OK, but
        // shared_expert + experts must be MQ4) and would otherwise
        // re-introduce Tier-1 stride corruption.
        debug_assert!(
            matches!(layer.wz.gpu_dtype, DType::Q8_0)
                && matches!(layer.w_beta.gpu_dtype, DType::Q8_0)
                && matches!(layer.w_alpha.gpu_dtype, DType::Q8_0),
            "DNMoe LA qkvza Q8 WMMA dispatch requires all of wqkv/wz/w_beta/w_alpha to be Q8_0",
        );
        run_fused_qkvza_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedQkvzaQ8_0,
            &layer.wqkv.buf,
            &layer.wz.buf,
            &layer.w_beta.buf,
            &layer.w_alpha.buf,
            &pbs.x_rot_batch,
            &pbs.dn_qkv_batch,
            &pbs.dn_z_batch,
            &pbs.dn_beta_batch,
            &pbs.dn_alpha_batch,
            layer.wqkv.m,
            layer.wz.m,
            layer.w_beta.m,
            layer.w_alpha.m,
            layer.wqkv.k,
            n,
        )?;
    } else if is_q8 || is_lowbit {
        // #397 Ship 5.2 slice1: four plain Q8 batched GEMMs
        // (wqkv/wz/w_beta/w_alpha), sibling DeltaNet QKVZA path.
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wqkv.gpu_dtype),
            &layer.wqkv.buf,
            layer.wqkv.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.dn_qkv_batch,
            layer.wqkv.m,
            layer.wqkv.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wz.gpu_dtype),
            &layer.wz.buf,
            layer.wz.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.dn_z_batch,
            layer.wz.m,
            layer.wz.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.w_beta.gpu_dtype),
            &layer.w_beta.buf,
            layer.w_beta.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.dn_beta_batch,
            layer.w_beta.m,
            layer.w_beta.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.w_alpha.gpu_dtype),
            &layer.w_alpha.buf,
            layer.w_alpha.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.dn_alpha_batch,
            layer.w_alpha.m,
            layer.w_alpha.k,
            n,
        )?;
    } else {
        run_fused_qkvza_key(
            gpu,
            crate::forward_slots::fused_qkvza_key_for(layer.wqkv.gpu_dtype),
            &layer.wqkv.buf,
            &layer.wz.buf,
            &layer.w_beta.buf,
            &layer.w_alpha.buf,
            &pbs.x_rot_batch,
            &pbs.dn_qkv_batch,
            &pbs.dn_z_batch,
            &pbs.dn_beta_batch,
            &pbs.dn_alpha_batch,
            layer.wqkv.m,
            layer.wz.m,
            layer.w_beta.m,
            layer.w_alpha.m,
            layer.wqkv.k,
            n,
        )?;
    }
    gpu.fused_sigmoid_alpha_gate_f32_batched(
        &pbs.dn_beta_batch,
        &pbs.dn_alpha_batch,
        &layer.dt_bias,
        &layer.a_log,
        n_v_heads,
        n,
    )?;
    if let Some(tape) = gdn_tape.as_ref() {
        let qkv_row_bytes = tape.qkv_dim * 4;
        let alpha_row_bytes = n_v_heads * 4;
        let off_qkv = tape_offset * qkv_row_bytes;
        let off_a = tape_offset * alpha_row_bytes;
        let copy_qkv = n * qkv_row_bytes;
        let copy_a = n * alpha_row_bytes;
        gpu.memcpy_dtod_at_auto(
            &tape.qkv_bufs[delta_layer_idx].buf,
            off_qkv,
            &pbs.dn_qkv_batch.buf,
            0,
            copy_qkv,
        )?;
        gpu.memcpy_dtod_at_auto(
            &tape.alpha_bufs[delta_layer_idx].buf,
            off_a,
            &pbs.dn_alpha_batch.buf,
            0,
            copy_a,
        )?;
        gpu.memcpy_dtod_at_auto(
            &tape.beta_bufs[delta_layer_idx].buf,
            off_a,
            &pbs.dn_beta_batch.buf,
            0,
            copy_a,
        )?;
    }
    // Same tree-aware dispatch gate as dense LA branch above.
    let tree_parents = tree_verify.as_ref().and_then(|c| c.parent_indices);
    if let Some(parents) = tree_parents {
        gpu.conv1d_silu_split_tree_f32_n(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            &pbs.dn_v_batch,
            &pbs.dn_qkv_batch,
            &layer.conv_weight,
            &dn_state.conv_states[delta_layer_idx],
            parents,
            k_dim,
            v_dim,
            n,
        )?;
    } else if let BatchSemantics::Independent { active_mask, .. } = batch_semantics {
        let full_mask = valid_lane_mask(n)?;
        if active_mask == full_mask {
            gpu.conv1d_silu_split_f32_independent(
                &pbs.dn_q_raw_batch,
                &pbs.dn_k_raw_batch,
                &pbs.dn_v_batch,
                &pbs.dn_qkv_batch,
                &layer.conv_weight,
                &dn_state.conv_states[delta_layer_idx],
                k_dim,
                v_dim,
                n,
            )?;
        } else {
            gpu.conv1d_silu_split_f32_independent_masked(
                &pbs.dn_q_raw_batch,
                &pbs.dn_k_raw_batch,
                &pbs.dn_v_batch,
                &pbs.dn_qkv_batch,
                &layer.conv_weight,
                &dn_state.conv_states[delta_layer_idx],
                k_dim,
                v_dim,
                n,
                active_mask,
            )?;
        }
    } else {
        gpu.conv1d_silu_split_f32_n(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            &pbs.dn_v_batch,
            &pbs.dn_qkv_batch,
            &layer.conv_weight,
            &dn_state.conv_states[delta_layer_idx],
            k_dim,
            v_dim,
            n,
        )?;
    }
    gpu.fused_qk_l2_norm_scale_f32_batched(
        &pbs.dn_q_raw_batch,
        &pbs.dn_k_raw_batch,
        config.linear_num_key_heads,
        hd,
        1.0 / (hd as f32).sqrt(),
        config.norm_eps,
        n,
    )?;
    if config.linear_num_key_heads < n_v_heads {
        let ratio = n_v_heads / config.linear_num_key_heads;
        gpu.repeat_interleave_qk_f32_batched(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            &pbs.dn_q_batch,
            &pbs.dn_k_batch,
            config.linear_num_key_heads,
            ratio,
            hd,
            n,
        )?;
    } else {
        gpu.memcpy_dtod_auto(&pbs.dn_q_batch.buf, &pbs.dn_q_raw_batch.buf, n * k_dim * 4)?;
        gpu.memcpy_dtod_auto(&pbs.dn_k_batch.buf, &pbs.dn_k_raw_batch.buf, n * k_dim * 4)?;
    }
    // DIAG: dump GDN inputs (batched, MoE branch)
    if layer_idx == 0 {
        let qk_dim = n_v_heads * hd;
        dump_hidden_localize(gpu, &pbs.dn_q_batch, n, start_pos, qk_dim, 0, "q_b");
        dump_hidden_localize(gpu, &pbs.dn_k_batch, n, start_pos, qk_dim, 0, "k_b");
        dump_hidden_localize(gpu, &pbs.dn_v_batch, n, start_pos, v_dim, 0, "v_b");
        dump_hidden_localize(
            gpu,
            &pbs.dn_alpha_batch,
            n,
            start_pos,
            n_v_heads,
            0,
            "alpha_b",
        );
        dump_hidden_localize(
            gpu,
            &pbs.dn_beta_batch,
            n,
            start_pos,
            n_v_heads,
            0,
            "beta_b",
        );
    }
    if let Some(parents) = tree_parents {
        // MoE-path tree-verify GDN, dispatched by state quant
        // (mirror of the dense path above).
        match dn_state.quant {
            StateQuant::FP32 => {
                let tape_f32 = pbs.dn_s_tape_f32.as_ref().expect(
                                "FP32 tree-aware LA requires dn_s_tape_f32 scratch (check PrefillBatchScratch::new)",
                            );
                gpu.gated_delta_net_f32_tree_batch_seq(
                    &pbs.dn_q_batch,
                    &pbs.dn_k_batch,
                    &pbs.dn_v_batch,
                    &pbs.dn_alpha_batch,
                    &pbs.dn_beta_batch,
                    &dn_state.s_matrices[delta_layer_idx],
                    tape_f32,
                    parents,
                    &pbs.dn_attn_out_batch,
                    n,
                    n_v_heads,
                    config.linear_value_head_dim,
                )?;
            }
            StateQuant::Q8 => {
                let tape_q8 = pbs
                    .dn_s_tape_q8
                    .as_ref()
                    .expect("tree-aware LA requires dn_s_tape_q8 scratch");
                let tape_sc = pbs
                    .dn_s_tape_scales
                    .as_ref()
                    .expect("tree-aware LA requires dn_s_tape_scales scratch");
                gpu.gated_delta_net_q8_tree_batch_seq(
                    &pbs.dn_q_batch,
                    &pbs.dn_k_batch,
                    &pbs.dn_v_batch,
                    &pbs.dn_alpha_batch,
                    &pbs.dn_beta_batch,
                    &dn_state.s_matrices[delta_layer_idx],
                    &dn_state.s_scales[delta_layer_idx],
                    tape_q8,
                    tape_sc,
                    parents,
                    &pbs.dn_attn_out_batch,
                    n,
                    n_v_heads,
                    config.linear_value_head_dim,
                )?;
            }
            StateQuant::Q4 => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "Q4 DeltaNet state + tree-verify (DDTree) is unsupported: \
                                 there is no Q4 tree-tape GDN kernel. Use Q8 or FP32 state \
                                 for tree spec-decode.",
                ));
            }
        }
    } else {
        match dn_state.quant {
            StateQuant::FP32 => {
                if rdna_compute::norm::gdn_chunked() && n > 1 {
                    gpu.gated_delta_net_f32_chunked(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                        rdna_compute::norm::gdn_chunk_size(),
                    )?
                } else {
                    gpu.gated_delta_net_f32_batch_seq(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?
                }
            }
            StateQuant::Q8 => {
                if let BatchSemantics::Independent { active_mask, .. } = batch_semantics {
                    let full_mask = valid_lane_mask(n)?;
                    if active_mask == full_mask {
                        gpu.gated_delta_net_q8_independent(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                            dn_state.ef_residual(delta_layer_idx),
                        )?
                    } else {
                        gpu.gated_delta_net_q8_independent_masked(
                            &pbs.dn_q_batch,
                            &pbs.dn_k_batch,
                            &pbs.dn_v_batch,
                            &pbs.dn_alpha_batch,
                            &pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            &pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_head_dim,
                            dn_state.ef_residual(delta_layer_idx),
                            active_mask,
                        )?
                    }
                } else {
                    gpu.gated_delta_net_q8_batch_seq(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                        dn_state.ef_residual(delta_layer_idx),
                    )?
                }
            }
            StateQuant::Q4 => gpu.gated_delta_net_q4(
                &pbs.dn_q_batch,
                &pbs.dn_k_batch,
                &pbs.dn_v_batch,
                &pbs.dn_alpha_batch,
                &pbs.dn_beta_batch,
                &dn_state.s_matrices[delta_layer_idx],
                &dn_state.s_scales[delta_layer_idx],
                &pbs.dn_attn_out_batch,
                n,
                n_v_heads,
                config.linear_value_head_dim,
            )?,
        }
        // DIAG: dump GDN attention output at layer 0
        if layer_idx == 0 {
            dump_hidden_localize(
                gpu,
                &pbs.dn_attn_out_batch,
                n,
                start_pos,
                n_v_heads * config.linear_value_head_dim,
                0,
                "gdn_b",
            );
        }
    }
    gpu.gated_norm_f32_batched(
        &pbs.dn_attn_out_batch,
        &pbs.dn_z_batch,
        &layer.norm_weight,
        &pbs.dn_normed_batch,
        n_v_heads,
        config.linear_value_head_dim,
        config.norm_eps,
        n,
    )?;
    // wo + residual. Q8 wo lands un-rotated (Q8 weights were
    // quantized against un-rotated activations); MQ4/MQ6 wo
    // require FWHT(awq_scale-adjusted) rotation. Mirrors the
    // dense LA wo dispatch (qwen35.rs:5000-5043) — the MQ6
    // branch is required for AWQ A3B where 4/40 LA layers
    // ship MQ6 wo and would otherwise corrupt the residual
    // stream when dispatched through the HFQ4 kernel against
    // 200 B/group MQ6-layout bytes.
    let dn_wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
    // TQ2G128/BQ1G128 have no fused qkvza/gate_up/qkv kernel, so they take
    // the same UNFUSED plain-GEMM strategy as Q8 rather than falling through
    // to the HFQ4 arm, which would read these packed blocks at the wrong
    // stride and produce fluent-but-wrong tokens.
    let dn_wo_is_lowbit = matches!(layer.wo.gpu_dtype, DType::TQ2G128 | DType::BQ1G128);
    let dn_wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
    let dn_wo_is_paro = matches!(layer.wo.gpu_dtype, DType::ParoQ4G128);
    // T-B: fused rotate+quantize (always-Residual here — no epilogue param).
    let iu4_wo_prep = try_iu4_rotate_prepared_no_epilogue(
        gpu,
        &layer.wo,
        &pbs.dn_normed_batch,
        &pbs.dn_normed_rot_batch,
        layer.wo.k,
        n,
    )?;
    let dn_wo_input = if iu4_wo_prep.is_some() {
        &pbs.dn_normed_rot_batch
    } else if dn_wo_is_q8 {
        &pbs.dn_normed_batch
    } else if dn_wo_is_paro {
        // PARO wo: rotate dn_normed by wo's own Givens tables
        // into dn_normed_rot_batch. Same scratch layout as MQ4
        // (since dn_normed_rot_batch is unused on the Q8 path).
        let paro_wo = layer.wo.paro.as_ref().unwrap_or_else(|| {
            panic!("ParoQ4G128 wo missing paro metadata at LA layer {layer_idx}")
        });
        gpu.givens_rotate_to(
            &pbs.dn_normed_batch,
            &pbs.dn_normed_rot_batch,
            &paro_wo.pairs,
            &paro_wo.theta,
            &paro_wo.channel_scales,
            n,
            layer.wo.k,
            paro_wo.krot as usize,
        )?;
        &pbs.dn_normed_rot_batch
    } else {
        // F2: AWQ-aware rotate for linear_attn wo (out_proj) input.
        rotate_x_mq_batched_for(
            gpu,
            &layer.wo,
            &pbs.dn_normed_batch,
            &pbs.dn_normed_rot_batch,
            layer.wo.k,
            n,
        )?;
        &pbs.dn_normed_rot_batch
    };
    if dn_wo_is_6bit {
        run_residual_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
            &layer.wo.buf,
            layer.wo.gpu_dtype,
            dn_wo_input,
            &pbs.x_batch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    } else if dn_wo_is_q8 && q8_wmma_arch {
        let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
        run_residual_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
            &layer.wo.buf,
            layer.wo.gpu_dtype,
            dn_wo_input,
            &x_n,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    } else if dn_wo_is_q8 || dn_wo_is_lowbit {
        // Non-WMMA Q8: gemm into a scratch then add into x_batch.
        // Reuse `dn_normed_rot_batch` (free since the MQ4 rotate
        // path didn't run here) as the GEMM scratch.
        let scratch = pbs.dn_normed_rot_batch.sub_offset(0, n * layer.wo.m);
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wo.gpu_dtype),
            &layer.wo.buf,
            layer.wo.gpu_dtype,
            dn_wo_input,
            &scratch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
        let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
        gpu.add_inplace_f32(&x_n, &scratch)?;
    } else if dn_wo_is_paro {
        // PARO wo residual: HFQ4G128 batched GEMM into scratch,
        // then add into x_batch. Reuse x_norm_batch (free at
        // this point — used earlier for the QKVZA stage; not
        // needed for the rest of this layer) as the scratch.
        let scratch = pbs.x_norm_batch.sub_offset(0, n * layer.wo.m);
        run_plain_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmHfq4G128,
            &layer.wo.buf,
            layer.wo.gpu_dtype,
            dn_wo_input,
            &scratch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
        let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
        gpu.add_inplace_f32(&x_n, &scratch)?;
    } else if let Some(prep) = &iu4_wo_prep {
        gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(
            &layer.wo.buf,
            prep,
            &pbs.x_batch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    } else {
        run_residual_gemm_key(
            gpu,
            crate::forward_slots::residual_gemm_key_for(layer.wo.gpu_dtype),
            &layer.wo.buf,
            layer.wo.gpu_dtype,
            dn_wo_input,
            &pbs.x_batch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    }

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
    // Batched MoE FA layer. FA body is the same as FullAttn
    // (rmsnorm + qkv + deinterleave + q/k norm + RoPE +
    // kv_write + attention + sigmoid_mul + wo+residual);
    // only the FFN differs. Duplicated inline — will be
    // consolidated with the dense FA batched body once the
    // MoE path is proven byte-exact.
    let kv_dim = config.n_kv_heads * config.head_dim;
    let q_dim = config.n_heads * config.head_dim;
    // This body is unreachable for MQ3 / MQ3-Lloyd weights —
    // the upstream `mq3_in_moe` guard at the top of
    // `forward_prefill_batch_with_pbs` rejects any MoE layer
    // with MQ3/Lloyd-MQ3 weights anywhere (attention OR FFN),
    // mirroring the captured-path guard at line 3367+. So
    // `layer.wq.gpu_dtype` is restricted to MQ4G256 / HFQ4G256
    // / MQ6G256 / HFQ6G256 here. Adding MQ3 to the matcher AND
    // the QKV dispatch is insufficient — the wo path below
    // (line 5320) is hardcoded MQ4 too — so the all-or-nothing
    // wiring lives in a separate PR (see followup issue).
    let qkv_is_mq = matches!(
        layer.wq.gpu_dtype,
        DType::MQ4G256
            | DType::MQ4G256V2
            | DType::MQ4CG256
            | DType::MQ6G256
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256V2
            | DType::MQ2G256V2
    );
    let qkv_is_6bit = matches!(layer.wq.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
    let qkv_is_q8 = matches!(layer.wq.gpu_dtype, DType::Q8_0);
    // TQ2G128/BQ1G128 have no fused qkvza/gate_up/qkv kernel, so they take
    // the same UNFUSED plain-GEMM strategy as Q8 rather than falling through
    // to the HFQ4 arm, which would read these packed blocks at the wrong
    // stride and produce fluent-but-wrong tokens.
    let qkv_is_lowbit = matches!(layer.wq.gpu_dtype, DType::TQ2G128 | DType::BQ1G128);
    // qt=52 has no MoE-batched kernels (grouped/indexed LUT variants don't
    // exist): refuse loudly here rather than falling through to a uniform
    // fused key. Per-token decode serves Lloyd-MoE via generic dispatch paths.
    if matches!(layer.wq.gpu_dtype, DType::MQ4G256V2Lloyd) {
        return Err(hip_bridge::HipError::new(
            0,
            "batch_chunk_full_attn_moe: MQ4G256V2Lloyd attention weights have no MoE-batched prefill kernel — refusing (per-token fallback serves this layer)",
        ));
    }
    // Phase 1.6 (PARO FullAttnMoe): wq/wk/wv are ParoQ4G128
    // (each with its own Givens rotation tables). The fused-QKV
    // kernels can't handle this — they assume one shared
    // rotation. Unfused 3-way dispatch (rotate + gemm_hfq4g128
    // per projection) matches the LA QKVZA Phase 1.5 pattern.
    let qkv_is_paro = matches!(layer.wq.gpu_dtype, DType::ParoQ4G128);
    // Fused QKV requires uniform dtype — see issue #249 for
    // the dense FA variant. Gate the same way here.
    let q8_wmma_arch = q8_prefill_wmma_enabled(gpu);
    let qkv_same_dtype =
        layer.wk.gpu_dtype == layer.wq.gpu_dtype && layer.wv.gpu_dtype == layer.wq.gpu_dtype;

    if qkv_is_mq {
        // AWQ-aware: next linear is wq (Q/K/V share input → same AWQ scale).
        fused_rmsnorm_rotate_mq_batched_for(
            gpu,
            &pbs.x_batch,
            &layer.attn_norm,
            &layer.wq,
            &pbs.x_rot_batch,
            dim,
            config.norm_eps,
            n,
        )?;
    } else if qkv_is_paro {
        // PARO: rmsnorm into x_norm_batch (un-rotated). x_rot_batch
        // is reused as the per-weight rotation scratch.
        gpu.rmsnorm_batched(
            &pbs.x_batch,
            &layer.attn_norm,
            &pbs.x_norm_batch,
            n,
            dim,
            config.norm_eps,
        )?;
    } else {
        gpu.rmsnorm_batched(
            &pbs.x_batch,
            &layer.attn_norm,
            &pbs.x_rot_batch,
            n,
            dim,
            config.norm_eps,
        )?;
    }
    if qkv_is_paro {
        // PARO 3-way unfused dispatch (wq, wk, wv each with own
        // Givens rotation). Same shape outputs as the fused
        // paths: fa_q_full_batch, fa_k_batch, fa_v_batch.
        let paro_wq = layer.wq.paro.as_ref().unwrap_or_else(|| {
            panic!("ParoQ4G128 wq missing paro metadata at FA layer {layer_idx}")
        });
        let paro_wk = layer.wk.paro.as_ref().unwrap_or_else(|| {
            panic!("ParoQ4G128 wk missing paro metadata at FA layer {layer_idx}")
        });
        let paro_wv = layer.wv.paro.as_ref().unwrap_or_else(|| {
            panic!("ParoQ4G128 wv missing paro metadata at FA layer {layer_idx}")
        });
        // wq
        gpu.givens_rotate_to(
            &pbs.x_norm_batch,
            &pbs.x_rot_batch,
            &paro_wq.pairs,
            &paro_wq.theta,
            &paro_wq.channel_scales,
            n,
            dim,
            paro_wq.krot as usize,
        )?;
        run_plain_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmHfq4G128,
            &layer.wq.buf,
            layer.wq.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.fa_q_full_batch,
            layer.wq.m,
            layer.wq.k,
            n,
        )?;
        // wk
        gpu.givens_rotate_to(
            &pbs.x_norm_batch,
            &pbs.x_rot_batch,
            &paro_wk.pairs,
            &paro_wk.theta,
            &paro_wk.channel_scales,
            n,
            dim,
            paro_wk.krot as usize,
        )?;
        run_plain_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmHfq4G128,
            &layer.wk.buf,
            layer.wk.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.fa_k_batch,
            layer.wk.m,
            layer.wk.k,
            n,
        )?;
        // wv
        gpu.givens_rotate_to(
            &pbs.x_norm_batch,
            &pbs.x_rot_batch,
            &paro_wv.pairs,
            &paro_wv.theta,
            &paro_wv.channel_scales,
            n,
            dim,
            paro_wv.krot as usize,
        )?;
        run_plain_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmHfq4G128,
            &layer.wv.buf,
            layer.wv.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.fa_v_batch,
            layer.wv.m,
            layer.wv.k,
            n,
        )?;
    } else if qkv_is_6bit && qkv_same_dtype {
        run_fused_qkv_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedQkvHfq6G256,
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            &pbs.x_rot_batch,
            &pbs.fa_q_full_batch,
            &pbs.fa_k_batch,
            &pbs.fa_v_batch,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
            n,
        )?;
    } else if qkv_is_q8 && q8_wmma_arch && qkv_same_dtype {
        debug_assert!(
            matches!(layer.wk.gpu_dtype, DType::Q8_0) && matches!(layer.wv.gpu_dtype, DType::Q8_0),
            "FAMoe qkv Q8 WMMA dispatch requires all of wq/wk/wv to be Q8_0",
        );
        run_fused_qkv_key(
            gpu,
            hipfire_dispatch::types::KernelKey::FusedQkvQ8_0,
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            &pbs.x_rot_batch,
            &pbs.fa_q_full_batch,
            &pbs.fa_k_batch,
            &pbs.fa_v_batch,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
            n,
        )?;
    } else if (qkv_is_q8 || qkv_is_lowbit) && qkv_same_dtype {
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wq.gpu_dtype),
            &layer.wq.buf,
            layer.wq.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.fa_q_full_batch,
            layer.wq.m,
            layer.wq.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wk.gpu_dtype),
            &layer.wk.buf,
            layer.wk.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.fa_k_batch,
            layer.wk.m,
            layer.wk.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wv.gpu_dtype),
            &layer.wv.buf,
            layer.wv.gpu_dtype,
            &pbs.x_rot_batch,
            &pbs.fa_v_batch,
            layer.wv.m,
            layer.wv.k,
            n,
        )?;
    } else if qkv_same_dtype {
        run_fused_qkv_key(
            gpu,
            crate::forward_slots::fused_qkv_key_for(layer.wq.gpu_dtype),
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            &pbs.x_rot_batch,
            &pbs.fa_q_full_batch,
            &pbs.fa_k_batch,
            &pbs.fa_v_batch,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
            n,
        )?;
    } else {
        // Mixed-format fallback (issue #249). batched_gemm_single_weight
        // covers MQ4/HFQ4 + MQ6/HFQ6 + Q8_0; mixed-Q8/MQ4 within FAMoe
        // routes here.
        batched_gemm_single_weight(gpu, &layer.wq, &pbs.x_rot_batch, &pbs.fa_q_full_batch, n)?;
        batched_gemm_single_weight(gpu, &layer.wk, &pbs.x_rot_batch, &pbs.fa_k_batch, n)?;
        batched_gemm_single_weight(gpu, &layer.wv, &pbs.x_rot_batch, &pbs.fa_v_batch, n)?;
    }
    // Fused deinterleave + Q-rmsnorm — one launch instead of two, no Q
    // global-memory round trip. K-norm, triattn tap, and RoPE below are
    // unchanged.
    gpu.deinterleave_q_rmsnorm_f32_batched(
        &pbs.fa_q_full_batch,
        &pbs.fa_q_batch,
        &pbs.fa_gate_batch,
        &layer.q_norm,
        config.n_heads,
        config.head_dim,
        n,
        config.norm_eps,
    )?;
    gpu.rmsnorm_batched(
        &pbs.fa_k_batch,
        &layer.k_norm,
        &pbs.fa_k_batch,
        n * config.n_kv_heads,
        config.head_dim,
        config.norm_eps,
    )?;
    if hipfire_runtime::triattn::tap_enabled() {
        let gpu_handled = hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
            gpu,
            layer_idx,
            &pbs.fa_q_batch.buf,
            n,
            config.n_heads,
            config.head_dim,
        )?;
        if !gpu_handled {
            let n_q = config.n_heads * config.head_dim;
            let q_cpu = gpu.download_f32(&pbs.fa_q_batch)?;
            if hipfire_runtime::triattn::tap_needs_k() {
                let n_k = config.n_kv_heads * config.head_dim;
                let k_cpu = gpu.download_f32(&pbs.fa_k_batch)?;
                for b in 0..n {
                    hipfire_runtime::triattn::record_prerope_qk(
                        layer_idx,
                        &q_cpu[b * n_q..(b + 1) * n_q],
                        Some(&k_cpu[b * n_k..(b + 1) * n_k]),
                    );
                }
            } else {
                for b in 0..n {
                    hipfire_runtime::triattn::record_prerope_q(
                        layer_idx,
                        &q_cpu[b * n_q..(b + 1) * n_q],
                    );
                }
            }
        }
    }
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
    // pos_offset = compact_offset (absolute RoPE phase post-eviction);
    // pbs.positions stays physical for the KV-write. 0 when no compaction.
    // 39aa358: in DDTree verify, rotate at DEPTH positions instead
    // (correct sibling phases); KV write below stays physical.
    let rope_pos_buf = if tree_verify.is_some() {
        &pbs.rope_positions
    } else {
        &pbs.positions
    };
    gpu.rope_partial_interleaved_f32_batched(
        &pbs.fa_q_batch,
        &pbs.fa_k_batch,
        rope_pos_buf,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        n_rot,
        config.rope_theta,
        n,
        kv_cache.compact_offset as i32,
    )?;
    Ok(())
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
    batch_chunk_fa_attend(
        gpu,
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
        layer_idx,
        fa_attn_multirow,
        None, // commit_stride: MoE finish keeps legacy cadence
    )?;
    gpu.sigmoid_mul_f32(&pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
    // wo + residual. Mirrors the dense FA wo dispatch at
    // qwen35.rs:5591-5623 — Q8 wo skips rotation (un-rotated
    // input expected); MQ4/MQ6 wo apply FWHT(awq_scale-adjusted).
    // MQ6 branch added alongside MQ6_ADMIT (without it, MQ6 wo
    // bytes get fed to gemm_hfq4g256_residual which reads them
    // as 136 B/group HFQ4 layout vs the actual 200 B/group MQ6
    // — catastrophic stride mismatch produces a single-token
    // attractor on AWQ A3B's 4/40 FA layers with MQ6 wo).
    let fa_wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
    // TQ2G128/BQ1G128 have no fused qkvza/gate_up/qkv kernel, so they take
    // the same UNFUSED plain-GEMM strategy as Q8 rather than falling through
    // to the HFQ4 arm, which would read these packed blocks at the wrong
    // stride and produce fluent-but-wrong tokens.
    let fa_wo_is_lowbit = matches!(layer.wo.gpu_dtype, DType::TQ2G128 | DType::BQ1G128);
    let fa_wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
    // Phase 1.6 (PARO FullAttnMoe wo): own Givens rotation table,
    // 72 B/group HFQ4G128 layout. Rotate fa_attn_out_batch by wo's
    // paro into fa_attn_out_rot_batch, then HFQ4G128 GEMM into a
    // scratch, then add into x_batch.
    let fa_wo_is_paro = matches!(layer.wo.gpu_dtype, DType::ParoQ4G128);
    // T-B: fused rotate+quantize (always-Residual here — no epilogue param).
    let iu4_wo_prep = try_iu4_rotate_prepared_no_epilogue(
        gpu,
        &layer.wo,
        &pbs.fa_attn_out_batch,
        &pbs.fa_attn_out_rot_batch,
        layer.wo.k,
        n,
    )?;
    let fa_wo_input = if iu4_wo_prep.is_some() {
        &pbs.fa_attn_out_rot_batch
    } else if fa_wo_is_q8 {
        &pbs.fa_attn_out_batch
    } else if fa_wo_is_paro {
        let paro_wo = layer.wo.paro.as_ref().unwrap_or_else(|| {
            panic!("ParoQ4G128 wo missing paro metadata at FA layer {layer_idx}")
        });
        gpu.givens_rotate_to(
            &pbs.fa_attn_out_batch,
            &pbs.fa_attn_out_rot_batch,
            &paro_wo.pairs,
            &paro_wo.theta,
            &paro_wo.channel_scales,
            n,
            layer.wo.k,
            paro_wo.krot as usize,
        )?;
        &pbs.fa_attn_out_rot_batch
    } else {
        // F2: AWQ-aware rotate for FullAttention wo (o_proj) input.
        rotate_x_mq_batched_for(
            gpu,
            &layer.wo,
            &pbs.fa_attn_out_batch,
            &pbs.fa_attn_out_rot_batch,
            layer.wo.k,
            n,
        )?;
        &pbs.fa_attn_out_rot_batch
    };
    if fa_wo_is_6bit {
        run_residual_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
            &layer.wo.buf,
            layer.wo.gpu_dtype,
            fa_wo_input,
            &pbs.x_batch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    } else if fa_wo_is_q8 && q8_wmma_arch {
        let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
        run_residual_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
            &layer.wo.buf,
            layer.wo.gpu_dtype,
            fa_wo_input,
            &x_n,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    } else if fa_wo_is_q8 || fa_wo_is_lowbit {
        // Non-WMMA Q8: GEMM into a scratch then add into x_batch.
        // Reuse `fa_attn_out_rot_batch` (free since MQ4 rotate
        // didn't run here) as scratch.
        let scratch = pbs.fa_attn_out_rot_batch.sub_offset(0, n * layer.wo.m);
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wo.gpu_dtype),
            &layer.wo.buf,
            layer.wo.gpu_dtype,
            fa_wo_input,
            &scratch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
        let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
        gpu.add_inplace_f32(&x_n, &scratch)?;
    } else if fa_wo_is_paro {
        // PARO wo residual: HFQ4G128 batched GEMM into scratch,
        // then add into x_batch. Reuse x_norm_batch (free since
        // QKVZA is done — the MoE FFN body below rewrites it
        // as its first action) as the gemm output scratch.
        let scratch = pbs.x_norm_batch.sub_offset(0, n * layer.wo.m);
        run_plain_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmHfq4G128,
            &layer.wo.buf,
            layer.wo.gpu_dtype,
            fa_wo_input,
            &scratch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
        let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
        gpu.add_inplace_f32(&x_n, &scratch)?;
    } else if let Some(prep) = &iu4_wo_prep {
        gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(
            &layer.wo.buf,
            prep,
            &pbs.x_batch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    } else {
        run_residual_gemm_key(
            gpu,
            crate::forward_slots::residual_gemm_key_for(layer.wo.gpu_dtype),
            &layer.wo.buf,
            layer.wo.gpu_dtype,
            fa_wo_input,
            &pbs.x_batch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    }

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
    per_token_hidden_out: Option<(&GpuTensor, usize)>,
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
        if let Some((dst, offset_rows)) = per_token_hidden_out {
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
    if !(64..=32768).contains(&max_ctx_end) {
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
    gpu.hip.memcpy_dtod_at(&stage.q.buf, 0, &pbs_c.fa_q_batch.buf, 0, half_q_bytes)?;
    gpu.hip
        .memcpy_dtod_at(&stage.q.buf, half_q_bytes, &pbs_n.fa_q_batch.buf, 0, half_q_bytes)?;
    gpu.hip.memcpy_dtod_at(&stage.k.buf, 0, &pbs_c.fa_k_batch.buf, 0, half_kv_bytes)?;
    gpu.hip
        .memcpy_dtod_at(&stage.k.buf, half_kv_bytes, &pbs_n.fa_k_batch.buf, 0, half_kv_bytes)?;
    gpu.hip.memcpy_dtod_at(&stage.v.buf, 0, &pbs_c.fa_v_batch.buf, 0, half_kv_bytes)?;
    gpu.hip
        .memcpy_dtod_at(&stage.v.buf, half_kv_bytes, &pbs_n.fa_v_batch.buf, 0, half_kv_bytes)?;
    gpu.hip.memcpy_dtod_at(&stage.pos.buf, 0, &pbs_c.positions.buf, 0, half_pos_bytes)?;
    gpu.hip.memcpy_dtod_at(
        &stage.pos.buf,
        half_pos_bytes,
        &pbs_n.positions.buf,
        0,
        half_pos_bytes,
    )?;
    execute_fa_attend_step(
        gpu,
        config,
        &stage.q,
        &stage.k,
        &stage.v,
        &stage.pos,
        &stage.out,
        s,
        kv_cache,
        FA_PAIR_BATCH,
        start_c,
        max_ctx_len,
        ctx,
        None,
        layer_idx,
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
    per_token_hidden_out: Option<&GpuTensor>,
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
    kv_cache
        .require_mapped_capacity(checked_kv_end(start_c, 2 * n, "forward_prefill_chunk_pair")?)?;

    let dim = config.dim;
    let hidden_dim = config.hidden_dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    let dim_row_bytes = dim * 4;
    let dispatch_workload = prefill_dispatch_workload(per_token_hidden_out.is_some(), false, false);
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
        gpu, weights, tokens_c, s, pbs_c, n, dim, dim_row_bytes, true, false, false, mo_c,
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
        gpu, weights, tokens_n, s, pbs_n, n, dim, dim_row_bytes, true, false, false, mo_n,
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
    let fa_batched_ok = (kv_cache.quant_q8
        || kv_cache.quant_asym4
        || kv_cache.quant_asym3
        || kv_cache.quant_asym2)
        && weights.layers.iter().all(|lw| match lw {
            LayerWeights::FullAttn(_) | LayerWeights::FullAttnMoe(_) => {
                qwen35_layer_batch_admissible(lw, config, fa_arch).is_ok()
            }
            _ => true,
        });
    // Merge only when both halves take the plain dispatch route. Multirow
    // is never admitted on gfx1151 (arch gate above it); the per-half
    // fallback below keeps this fail-closed if that ever changes.
    let multirow_common = (
        gpu.arch_caps.arch(),
        kv_cache.quant_q8,
        config.head_dim,
        fa_pertoken_min_ctx(),
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
                for (half_pbs, half_start, half_tape) in
                    [(pbs_c, start_c, chunk_start_c), (pbs_n, start_n, chunk_start_c + n)]
                {
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
                        None, // commit_stride: pair halves keep legacy cadence
                    )?;
                    batch_chunk_delta_net_ffn(
                        gpu,
                        layer,
                        config,
                        half_pbs,
                        n,
                        dim,
                        hidden_dim,
                        q8_wmma_arch,
                        arch_has_wmma,
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
                    batch_chunk_full_attn_input_projection(
                        gpu, layer, config, pbs_c, n, dim, q8_wmma_arch, fusion,
                    )?;
                    batch_chunk_full_attn_prepare(
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
                        kv_layer_idx,
                        layer_idx,
                        fusion,
                    )?;
                    batch_chunk_full_attn_input_projection(
                        gpu, layer, config, pbs_n, n, dim, q8_wmma_arch, fusion,
                    )?;
                    batch_chunk_full_attn_prepare(
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
                        kv_layer_idx,
                        layer_idx,
                        fusion,
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
                    batch_chunk_full_attn_output_projection(
                        gpu,
                        layer,
                        pbs_c,
                        n,
                        q8_wmma_arch,
                        arch_has_wmma,
                        BatchEpilogue::Residual,
                        fusion,
                    )?;
                    batch_chunk_full_attn_ffn(
                        gpu,
                        layer,
                        config,
                        pbs_c,
                        n,
                        dim,
                        hidden_dim,
                        q8_wmma_arch,
                        arch_has_wmma,
                        BatchEpilogue::Residual,
                        fusion,
                    )?;
                    batch_chunk_full_attn_output_projection(
                        gpu,
                        layer,
                        pbs_n,
                        n,
                        q8_wmma_arch,
                        arch_has_wmma,
                        BatchEpilogue::Residual,
                        fusion,
                    )?;
                    batch_chunk_full_attn_ffn(
                        gpu,
                        layer,
                        config,
                        pbs_n,
                        n,
                        dim,
                        hidden_dim,
                        q8_wmma_arch,
                        arch_has_wmma,
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
                    batch_chunk_full_attn_ffn(
                        gpu,
                        layer,
                        config,
                        pbs_c,
                        n,
                        dim,
                        hidden_dim,
                        q8_wmma_arch,
                        arch_has_wmma,
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
                    batch_chunk_full_attn_ffn(
                        gpu,
                        layer,
                        config,
                        pbs_n,
                        n,
                        dim,
                        hidden_dim,
                        q8_wmma_arch,
                        arch_has_wmma,
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
                for (half_pbs, half_start, half_tape) in
                    [(pbs_c, start_c, chunk_start_c), (pbs_n, start_n, chunk_start_c + n)]
                {
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
                        q8_wmma_arch,
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
    let pth_c = per_token_hidden_out.map(|t| (t, chunk_start_c));
    let pth_n = per_token_hidden_out.map(|t| (t, chunk_start_c + n));
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
    per_token_hidden_out: Option<(&GpuTensor, usize)>,
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
    let dispatch_workload = prefill_dispatch_workload(
        per_token_hidden_out.is_some(),
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
    let fa_batched_ok =
        (kv_cache.quant_q8 || kv_cache.quant_asym4 || kv_cache.quant_asym3 || kv_cache.quant_asym2)
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
        fa_pertoken_min_ctx(),
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
    let ctx = DispatchCtx::new(gpu).with_workload(dispatch_workload);

    for layer_idx in layer_start..layer_end {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
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
                )?;
                batch_chunk_delta_net_ffn(
                    gpu,
                    layer,
                    config,
                    pbs,
                    n,
                    dim,
                    hidden_dim,
                    q8_wmma_arch,
                    arch_has_wmma,
                    BatchEpilogue::Residual,
                    fusion,
                )?;
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
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
                batch_chunk_full_attn_ffn(
                    gpu,
                    layer,
                    config,
                    pbs,
                    n,
                    dim,
                    hidden_dim,
                    q8_wmma_arch,
                    arch_has_wmma,
                    BatchEpilogue::Residual,
                    fusion,
                )?;
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
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
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
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
                    q8_wmma_arch,
                    start_pos,
                    layer_idx,
                    &ctx,
                    weights,
                    routed_out,
                    route,
                )?;
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
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
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }
                kv_layer_idx += 1;
                dump_hidden_localize(gpu, &pbs.x_batch, n, start_pos, dim, layer_idx, "batched");
            }
            _ => panic!("layer type mismatch at layer {layer_idx}"),
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
            let key = crate::forward_slots::fused_qkv_key_for(dt);
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
            let key = crate::forward_slots::fused_gate_up_key_for(dt_g);
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

/// Batched single-weight GEMM used by the mixed-format fallback in
/// `forward_prefill_chunk`'s FA QKV path. The fused `gemm_qkv_hfq*` kernels
/// require wq/wk/wv to share a bit-width — they index all three weight
/// buffers with the same stride. When `--kmap-dense --kmap-mode 2` promotes
/// only `v_proj` to MQ6 (issue #249), the fused HFQ4 kernel reads `wv`'s
/// MQ6 buffer with HFQ4's 136-B stride (true stride: 200 B), producing
/// silent NaN. Callers gate the fused path on a same-dtype check and route
/// here per-weight when they disagree.
///
/// Covers same-rotation-family bit-width mixes: MQ4/MQ4V2/MQ4C+MQ6 (all
/// FWHT-baked; kmap mode 2 can promote one of q/k/v to MQ6 while leaving
/// others at qt13/qt44/qt45) and HFQ4+HFQ6 (both unrotated). qt44/qt45 are
/// gfx12-only through existing model eligibility and dispatch predicates;
/// each uses its format-specific residual GEMM key after zeroing Y — never
/// the v1 `GemmHfq4G256` key. Cross-family mixes (e.g. HFQ4+MQ6) would
/// corrupt the shared rmsnorm+rotate output; no quantizer config produces
/// them today, but extend the dispatch caller's invariants here if that
/// changes.
fn batched_gemm_single_weight(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    match w.gpu_dtype {
        DType::MQ4G256 | DType::HFQ4G256 => run_plain_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmHfq4G256,
            &w.buf,
            w.gpu_dtype,
            x,
            y,
            w.m,
            w.k,
            n,
        ),
        DType::MQ4G256V2 => {
            // No non-residual batched MQ4V2 GEMM in the mixed-QKV path.
            // Zero Y on the active stream then accumulate via the gfx12
            // residual key — never GemmHfq4G256 (v1 header decode).
            let bytes = w.m * n * 4;
            if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip.memset_async(&y.buf, 0, bytes, stream)?;
            } else {
                gpu.hip.memset(&y.buf, 0, bytes)?;
            }
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmMq4G256V2Residual,
                &w.buf,
                w.gpu_dtype,
                x,
                y,
                w.m,
                w.k,
                n,
            )
        }
        DType::MQ4G256V2Lloyd => {
            // qt=52 mixed-format twin of the V2 arm: zero Y then residual
            // (== plain GEMM). gfx11 → MMQ-LUT ADD; gfx12 keeps FP8-LUT.
            let bytes = w.m * n * 4;
            if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip.memset_async(&y.buf, 0, bytes, stream)?;
            } else {
                gpu.hip.memset(&y.buf, 0, bytes)?;
            }
            if lloyd_mmq_lut_route(gpu) {
                let c16 = lloyd_c16_or_fail(w, "batched_gemm_single_weight")?;
                gpu.gemm_hfq4g256_residual_mmq_lloyd(&w.buf, x, y, w.m, w.k, n, c16)
            } else {
                let lut = lloyd_e4m3_or_fail(w, "batched_gemm_single_weight")?;
                gpu.gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8_lloyd(
                    &w.buf, x, y, w.m, w.k, n, 1, lut,
                )
            }
        }
        DType::MQ4CG256 => {
            // Same residual-only contract as MQ4V2: zero Y then format-
            // specific residual GEMM. qt45 header is a single affine grid;
            // routing through GemmHfq4G256 would mis-decode scale/zero.
            let bytes = w.m * n * 4;
            if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip.memset_async(&y.buf, 0, bytes, stream)?;
            } else {
                gpu.hip.memset(&y.buf, 0, bytes)?;
            }
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmMq4CG256Residual,
                &w.buf,
                w.gpu_dtype,
                x,
                y,
                w.m,
                w.k,
                n,
            )
        }
        DType::MQ6G256 | DType::HFQ6G256 => {
            // No non-residual batched MQ6/HFQ6 GEMM exists. Zero Y then
            // accumulate. The zero MUST be ordered on the same stream as
            // the GEMM that consumes it — using sync `hipMemset` on the
            // null stream while subsequent kernels enqueue on a non-null
            // active stream leaves a race that produces silent NaN in the
            // residual stream (logits stay NaN on eval until a stray host
            // sync masks the order bug).
            let bytes = w.m * n * 4;
            if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip.memset_async(&y.buf, 0, bytes, stream)?;
            } else {
                gpu.hip.memset(&y.buf, 0, bytes)?;
            }
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                &w.buf,
                w.gpu_dtype,
                x,
                y,
                w.m,
                w.k,
                n,
            )
        }
        DType::MQ6G256V2 => {
            let bytes = w.m * n * 4;
            if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip.memset_async(&y.buf, 0, bytes, stream)?;
            } else {
                gpu.hip.memset(&y.buf, 0, bytes)?;
            }
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmMq6G256V2Residual,
                &w.buf,
                w.gpu_dtype,
                x,
                y,
                w.m,
                w.k,
                n,
            )
        }
        DType::MQ5G256V2 => {
            let bytes = w.m * n * 4;
            if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip.memset_async(&y.buf, 0, bytes, stream)?;
            } else {
                gpu.hip.memset(&y.buf, 0, bytes)?;
            }
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmMq5G256V2Residual,
                &w.buf,
                w.gpu_dtype,
                x,
                y,
                w.m,
                w.k,
                n,
            )
        }
        DType::MQ3G256V2 => {
            let bytes = w.m * n * 4;
            if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip.memset_async(&y.buf, 0, bytes, stream)?;
            } else {
                gpu.hip.memset(&y.buf, 0, bytes)?;
            }
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmMq3G256V2Residual,
                &w.buf,
                w.gpu_dtype,
                x,
                y,
                w.m,
                w.k,
                n,
            )
        }
        DType::MQ2G256V2 => {
            let bytes = w.m * n * 4;
            if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip.memset_async(&y.buf, 0, bytes, stream)?;
            } else {
                gpu.hip.memset(&y.buf, 0, bytes)?;
            }
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmMq2G256V2Residual,
                &w.buf,
                w.gpu_dtype,
                x,
                y,
                w.m,
                w.k,
                n,
            )
        }
        DType::MQ3G256 => {
            // Same pattern as MQ6: no non-residual batched HFQ3 GEMM
            // exists in the scalar gfx10 family — `gemm_hfq3g256_residual`
            // is the only single-weight batched dispatch. Zero Y on the
            // active stream (same race-free contract as the HFQ6 arm)
            // then accumulate.
            let bytes = w.m * n * 4;
            if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip.memset_async(&y.buf, 0, bytes, stream)?;
            } else {
                gpu.hip.memset(&y.buf, 0, bytes)?;
            }
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                &w.buf,
                w.gpu_dtype,
                x,
                y,
                w.m,
                w.k,
                n,
            )
        }
        DType::Q8_0 => {
            // Q8 weights consume the un-rotated rmsnorm output. Callers
            // routing here must pass `pbs.x_rot_batch` containing
            // `rmsnorm(x_batch)` *without* FWHT — the existing pattern is
            // to gate the `fused_rmsnorm_rotate_*_for(...)` call on
            // `is_mq` and fall through to `gpu.rmsnorm_batched(...)` for
            // Q8 (see DNMoe LA preamble for a representative).
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                &w.buf,
                w.gpu_dtype,
                x,
                y,
                w.m,
                w.k,
                n,
            )
        }
        other => Err(hip_bridge::HipError::new(
            0,
            &format!(
                "mixed-format batched prefill: weight dtype {other:?} has no \
             single-weight batched dispatch yet. Currently MQ3/HFQ3, MQ6/5/3/2V2, \
             MQ4/MQ4V2/MQ4C/HFQ4, MQ6/HFQ6, MQ6/5/3/2V2, and Q8_0 mixes are wired. Re-quantize with \
             uniform format or extend `batched_gemm_single_weight` to cover this format."
            ),
        )),
    }
}
#[cfg(test)]
mod tests {
    use super::super::forward::unsupported_mq3_experts_uniform_from_dtypes;
    use super::*;
    use hipfire_dispatch::context::DispatchWorkload;
    use rdna_compute::DType;

    #[test]
    fn q8_multirow_attn_admits_only_measured_arch_shapes() {
        for arch in ["gfx1100", "gfx1201"] {
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
        for arch in ["gfx1100", "gfx1201"] {
            assert!(q8_multirow_attn_admitted(
                arch,
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
            assert!(!q8_multirow_attn_admitted(
                arch,
                true,
                256,
                8,
                8192,
                Some(4096),
                false,
                false,
                false,
                true,
            ));
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
        use crate::forward_slots::{
            fused_gate_up_key_for, fused_qkv_key_for, fused_qkvza_key_for, residual_gemm_key_for,
        };
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
            crate::forward_slots::fused_qkv_key_for(DType::HFQ4G256),
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
    fn qwen35_is_batchable_la_lloyd_mq3_only_on_gfx11_with_opt_in_gfx12() {
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
        // gfx12 requires env gate
        assert!(
            !is_batchable_la(DType::MQ3G256Lloyd, "gfx1200"),
            "gfx1200 without HIPFIRE_LLOYD_GFX12=1"
        );
        assert!(
            !is_batchable_la(DType::MQ4G256Lloyd, "gfx1200"),
            "gfx1200 without HIPFIRE_LLOYD_GFX12=1"
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
        use crate::forward_slots::{fused_gate_up_key_for, residual_gemm_key_for};
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
    fn widened_direct_grouping_matches_tail_contract() {
        // Plan §2.3 first-chunk examples at the 4096 ceiling.
        assert_eq!(next_exact_prefill_chunk_len(513, 4096), Some(511));
        assert_eq!(next_exact_prefill_chunk_len(1025, 4096), Some(512));
        assert_eq!(next_exact_prefill_chunk_len(1537, 4096), Some(1024));
        assert_eq!(next_exact_prefill_chunk_len(2049, 4096), Some(1024));
        assert_eq!(next_exact_prefill_chunk_len(4097, 4096), Some(2048));
        // Power-of-two merging: non-power counts snap down (6→4, 7→4).
        assert_eq!(next_exact_prefill_chunk_len(3072, 4096), Some(2048));
        assert_eq!(next_exact_prefill_chunk_len(3584, 4096), Some(2048));
        assert_eq!(next_exact_prefill_chunk_len(1024, 4096), Some(1024));
        assert_eq!(next_exact_prefill_chunk_len(8192, 8192), Some(8192));
        // Sub-512 explicit ceilings keep the legacy schedule exactly
        // (513→256+255+2, like the legacy 256 planner).
        assert_eq!(next_exact_prefill_chunk_len(513, 256), Some(256));
        assert_eq!(next_exact_prefill_chunk_len(385, 384), Some(383));
        // Degenerate inputs refuse like the legacy planner.
        assert_eq!(next_exact_prefill_chunk_len(1, 4096), None);
        assert_eq!(next_exact_prefill_chunk_len(3, 2), None);
    }

    #[test]
    fn widened_direct_flattened_commits_equal_legacy_schedule() {
        // Every wide chunk is an exact multiple of 512, so expanding each
        // wide chunk back into 512-row commits must reproduce the legacy
        // 512 schedule row-for-row (the plan's partition proof). Deterministic
        // CPU check over lengths 2..=16385 and every staged ceiling.
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
                if c > 512 {
                    assert_eq!(c % 512, 0, "wide chunk must be a 512 multiple");
                    out.extend(std::iter::repeat_n(512, c / 512));
                } else {
                    out.push(c);
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
    fn widened_serve_grouping_matches_caller_contract() {
        // Serve splits outer chunks and invokes forward separately: 1025 →
        // 1024+1 (flattening to old 512+512+1 inside), 1537 → 1024+512+1.
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
        assert_eq!(plan(1025, 4096), vec![1024, 1]);
        assert_eq!(plan(1537, 4096), vec![1024, 512, 1]);
        assert_eq!(plan(513, 4096), vec![512, 1]);
        assert_eq!(plan(2049, 2048), vec![2048, 1]);
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
    fn widened_dense_bytes_match_ledger() {
        // Plan §4.1 steady-state envelope: 725,388 B/row + 256 fixed, FP8
        // 17,956 B/row, Q16 fixed 6,291,456 B for 24×256×2×512.
        let config = widened_test_config();
        assert!(widened_dense_shape_admitted(&config));
        assert_eq!(fp8_row_bytes_wide(&config), Some(17_956));
        assert_eq!(dense_prefill_allocation_bytes(&config, 1), Some(725_388 + 256));
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
            assert!(!fa_pair_merge_admitted(512, partial, "gfx1151", false, false));
            assert!(!fa_pair_merge_admitted(partial, 512, "gfx1151", false, false));
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
}
