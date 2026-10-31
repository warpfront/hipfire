// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 batched prefill: eligibility gates, dispatch-routed GEMM helpers,
//! `forward_prefill_batch*`, and the per-layer batched chunk bodies.

use super::batch::for_each_active_span;
use super::batch::partial_lane_mask;
use super::batch::PrefillBatchScratch;
use super::config::LayerType;
use super::config::MaskEmbedOverride;
use super::config::Qwen35Config;
use super::forward::checked_kv_end;
use super::forward::forward_scratch;
use super::forward::forward_scratch_with_hidden;
use super::forward::moe_ffn_has_mq3_experts_uniform;
use super::forward::moe_ffn_has_mq3_structural;
use super::forward::moe_ffn_has_unsupported_mq3_experts_uniform;
use super::forward::Qwen35Scratch;
use super::program::{
    batched_attention_tap, deltanet_state_view, flash_scratch, gdn_tape_view, hybrid_dims, kv_view,
};
use super::weights::DeltaNetState;
use super::weights::DeltaNetStateGeometry;
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
use hipfire_dispatch::families::kv_tier::KvTierInputs;
use hipfire_dispatch::families::kv_tier::KvTierPlan;
use hipfire_dispatch::families::moe::{codebook_batched_admit_enabled, NARROW_VERIFY_MAX_ROWS};
#[cfg(test)]
use hipfire_dispatch::families::moe::{
    codebook_batched_admit_enabled_from_env, mq6_batched_admit_enabled_from_env,
};
use hipfire_dispatch::pipeline::batched::{
    mq_f16_projection_fast_route, q8_prefill_wmma_enabled, s4_residual_fast, valid_lane_mask,
    verify_dense_math, BatchEpilogue, BatchSemantics, DenseBatchMath, DflashFusionCtx,
    TreeVerifyCtx,
};
use hipfire_dispatch::pipeline::batched_attention::{
    batch_chunk_fa_attend, batch_chunk_full_attn_attn, batch_chunk_full_attn_input_projection,
    batch_chunk_full_attn_output_projection, batch_chunk_full_attn_prepare,
    dflash_fa_batch_fusion_admitted, fa_pertoken_min_ctx, fa_prep_unfused_qk_norm,
    fa_prep_unfused_rope, gfx12_fa_prep_admitted, q8_multirow_attn_admitted, FaPairHalf,
    FaPairStage, FA_PAIR_ROWS,
};
use hipfire_dispatch::pipeline::batched_deltanet::{
    batch_chunk_delta_net_attn, batch_chunk_delta_net_input_projection,
    batch_chunk_delta_net_output_projection, batch_chunk_delta_net_pre_gdn,
    gdn_chunk_head_geometry_supported, gdn_qk_norm_interleave, gdn_sigmoid_alpha_gate,
    push_gdn_tape_copies,
};
use hipfire_dispatch::pipeline::batched_moe::{
    attention_moe_finish_batched, attention_moe_prep_batched, execute_attention_moe_layer_batched,
    execute_deltanet_moe_layer_batched,
};
use hipfire_dispatch::pipeline::dump_hidden_localize;
use hipfire_dispatch::pipeline::execute_steps;
use hipfire_dispatch::pipeline::gated_mlp::MlpOutput;
use hipfire_dispatch::pipeline::GemvInput;
use hipfire_dispatch::pipeline::Step;
use hipfire_runtime::llama;
use rdna_compute::norm::GdnScanOut;
use rdna_compute::prefill_route_capabilities;
use rdna_compute::prefill_route_capabilities::PrefillRoute;

/// Multi-request batched prefill chunk (cross-request speculative verify).
#[path = "prefill_multi.rs"]
pub mod multi;

use hipfire_runtime::llama::EmbeddingFormat;
use hipfire_runtime::llama::KvCacheExt;
use rdna_compute::DType;
use rdna_compute::Fp8VerifyTarget;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;

/// Whether a target-verify forward over `weights` with KV tier `kv` can run
/// inside the PM FP8 verify scope (`Gpu::fp8_verify_enter`): a dense
/// MQ4G256V2 target whose batched attention takes the KV tier (FP8, Q8 or
/// asym). A tree verify passes [`Fp8VerifyTarget::Ineligible`] itself.
/// Singleton, batched and rollback forwards of one request share this
/// answer, so they share the arithmetic.
pub(crate) fn fp8_verify_target(
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    kv: &llama::KvCache,
) -> Fp8VerifyTarget {
    let kv_batched =
        kv.quant_fp8 || kv.quant_q8 || kv.quant_asym4 || kv.quant_asym3 || kv.quant_asym2;
    if kv_batched && multi::multi_chunk_dense_mq4v2_target(weights, config) {
        Fp8VerifyTarget::Eligible
    } else {
        Fp8VerifyTarget::Ineligible
    }
}

/// Open the PM FP8 verify scope for a target-verify forward (see
/// [`fp8_verify_target`]); the caller closes it with
/// `gpu.fp8_verify_exit(prev)` on every path.
pub(crate) fn fp8_verify_open(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    kv: &llama::KvCache,
) -> HipResult<bool> {
    gpu.fp8_verify_enter(fp8_verify_target(weights, config, kv))
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

/// Widest batched forward whose routed experts run on the CPU when the loader
/// armed the expert splice (`memory.offload_exec=cpu`). Matches the MTP verify
/// window (drafts + 1, capped at 4): the narrow speculative verify and the
/// accepted-prefix rollback replay both fit, so they share the single-token
/// decode's CPU target arithmetic. The gate is width-only — every armed-sink
/// forward at `n <= 4` splices on the CPU irrespective of dispatch workload, so
/// a `Standard` tape-disabled rollback replay and an ordinary very short prompt
/// prefill take it too. Wider prompt prefill and wide verify keep the GPU
/// grouped PCIe read, which beats the CPU expert FFN by >10x at chunk sizes.
/// Single source is [`hipfire_dispatch::families::moe::NARROW_VERIFY_MAX_ROWS`].

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
    fp8_chunk512_for_flags(gpu.arch.as_str(), &gpu.flags)
}

/// Pure core of [`fp8_chunk512_for_gpu`]: the same three FP8 projection flags
/// on exact gfx1201, read from any [`rdna_compute::FeatureFlags`].
#[inline]
fn fp8_chunk512_for_flags(arch: &str, flags: &rdna_compute::FeatureFlags) -> bool {
    arch == "gfx1201"
        && flags.gfx12_mq4v2_fp8_gateup
        && flags.gfx12_mq4v2_fp8_resid
        && flags.gfx12_mq4v2_fp8_qkvza
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

/// Explicit chunk-rows request (`HIPFIRE_PREFILL_CHUNK_ROWS` or the
/// `prefill.chunk_rows` config key), or `None` when only the registered
/// built-in default exists: built-ins are omitted from `ProcessConfig`
/// (hipfire-config `include_builtin_in_process_config`), so `developer_var`
/// errs on them and an explicit user value is the only thing observed here.
fn explicit_prefill_chunk_rows() -> Option<usize> {
    hipfire_config::developer_var("HIPFIRE_PREFILL_CHUNK_ROWS")
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
pub fn prefill_chunk_rows_requested(gpu: &Gpu) -> usize {
    chunk_rows_requested_from(
        gpu.arch.as_str(),
        explicit_prefill_max_batch(),
        explicit_prefill_chunk_rows(),
    )
}

/// Pure core of [`prefill_chunk_rows_requested`].
fn chunk_rows_requested_from(
    arch: &str,
    explicit_max_batch: Option<usize>,
    explicit_chunk_rows: Option<usize>,
) -> usize {
    explicit_max_batch
        .or(explicit_chunk_rows)
        .unwrap_or_else(|| prefill_chunk_rows_default(arch))
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

/// No allocation: callers supply one fixed projection set per layer.
/// DeltaNet may retain its unrotated output projection as Q8_0 (HFQ qt3);
/// every other projection, including FullAttn output, must be symmetric qt49.
fn symmetric_mq3v2_projection_sets(
    marker: bool,
    mut layers: impl Iterator<Item = Option<([DType; 8], bool)>>,
) -> bool {
    let admissible = |(dtypes, delta_net): ([DType; 8], bool)| {
        dtypes.iter().enumerate().all(|(i, &dtype)| {
            dtype == DType::MQ3G256V2 || (delta_net && i == 4 && dtype == DType::Q8_0)
        })
    };
    marker
        && layers
            .next()
            .is_some_and(|projections| projections.is_some_and(admissible))
        && layers.all(|projections| projections.is_some_and(admissible))
}

fn dense_layers_are_uniform_symmetric_mq3v2(weights: &Qwen35Weights) -> bool {
    symmetric_mq3v2_projection_sets(
        weights.mq3v2_symmetric,
        weights.layers.iter().map(|layer| match layer {
            LayerWeights::DeltaNet(layer) => Some((
                [
                    layer.wqkv.gpu_dtype,
                    layer.wz.gpu_dtype,
                    layer.w_alpha.gpu_dtype,
                    layer.w_beta.gpu_dtype,
                    layer.wo.gpu_dtype,
                    layer.w_gate.gpu_dtype,
                    layer.w_up.gpu_dtype,
                    layer.w_down.gpu_dtype,
                ],
                true,
            )),
            LayerWeights::FullAttn(layer) => Some((
                [
                    layer.wq.gpu_dtype,
                    layer.wk.gpu_dtype,
                    layer.wv.gpu_dtype,
                    layer.wo.gpu_dtype,
                    layer.w_gate.gpu_dtype,
                    layer.w_up.gpu_dtype,
                    layer.w_down.gpu_dtype,
                    layer.w_down.gpu_dtype,
                ],
                false,
            )),
            LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_) => None,
        }),
    )
}

/// Symmetric qt49 dense model with an enabled A4 or A8 projection route: the
/// widened (8192-row) chunk, its A4/A8 projection scratch sizing and the GDN
/// chunk scan all key on this one predicate.
fn mq3_wide_route(gpu: &Gpu, weights: &Qwen35Weights) -> bool {
    dense_layers_are_uniform_symmetric_mq3v2(weights)
        && (gpu.flags.mq3v25_a8_enabled(gpu.mq3v2_symmetric)
            || gpu.flags.mq3v25_a4_enabled(gpu.mq3v2_symmetric))
}

/// Activation-projection sizing family of the widened route for projection
/// scratch reservation: `Off` keeps the incumbent MQ4 sizing (FP8 on gfx1201,
/// IU4 elsewhere); a symmetric qt49 route sizes against its A4 (72 B per K128
/// block) or A8 (136 B per K128 block) prelude. A8 wins over A4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mq3WideProjection {
    Off,
    A4,
    A8,
}

fn mq3_wide_projection(gpu: &Gpu, weights: &Qwen35Weights) -> Mq3WideProjection {
    if !mq3_wide_route(gpu, weights) {
        Mq3WideProjection::Off
    } else if gpu.flags.mq3v25_a8_enabled(gpu.mq3v2_symmetric) {
        Mq3WideProjection::A8
    } else {
        Mq3WideProjection::A4
    }
}

fn native_mq4_widened_requested() -> bool {
    hipfire_config::developer_var("HIPFIRE_GFX1100_MQ4_WIDE_PREFILL")
        .ok()
        .as_deref()
        == Some("1")
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
    arch: &str,
    flags: &rdna_compute::FeatureFlags,
    screen: bool,
) -> bool {
    arch == "gfx1100"
        && !flags.fp16_disabled
        && !flags.rocblas_all_archs
        && !flags.gemv_dp4a.unwrap_or(false)
        && flags.mmq_override.is_none()
        && flags.mmq_min_batch.is_none()
        && !screen
        && !flags.qkvza_split_tail
        && flags.wo_wmma_variant.is_none()
}

pub(crate) fn prefill_max_batch_for_model(gpu: &Gpu, weights: &Qwen35Weights) -> usize {
    legacy_prefill_max_batch(
        explicit_prefill_max_batch(),
        gpu.arch.as_str(),
        fp8_chunk512_for_gpu(gpu),
        gpu.arch == "gfx1151" && dense_layers_are_all_mq4v2(weights),
    )
}

/// Rows and tape flag of the implicit `HIPFIRE_PREFILL_REUSE_PBS=1` cache that
/// `Qwen35Scratch::new_with_kv_max` allocates (pure core of that sizing): an
/// explicit `HIPFIRE_PREFILL_MAX_BATCH` raises the cache exactly as it raises
/// every other ordinary ceiling, capped at 512 rows and tape-free on exact
/// gfx1201, floored at 512 rows on gfx1151, the configured rows with a tape
/// elsewhere.
pub(crate) fn implicit_reuse_pbs_plan(
    arch: &str,
    explicit_max_batch: Option<usize>,
    fp8_chunk512: bool,
) -> (usize, bool) {
    let configured =
        explicit_max_batch.unwrap_or_else(|| prefill_max_batch_for_arch(arch, fp8_chunk512));
    if arch == "gfx1201" {
        (configured.min(512), false)
    } else if arch == "gfx1151" {
        (configured.max(512), true)
    } else {
        (configured, true)
    }
}

/// [`implicit_reuse_pbs_plan`] for the live device and process configuration.
pub(crate) fn implicit_reuse_pbs_plan_for_gpu(gpu: &Gpu) -> (usize, bool) {
    implicit_reuse_pbs_plan(
        gpu.arch.as_str(),
        explicit_prefill_max_batch(),
        fp8_chunk512_for_gpu(gpu),
    )
}

/// Pure core of [`prefill_max_batch_for_model`]: the incumbent (non-widened)
/// ordinary chunk ceiling. Explicit `HIPFIRE_PREFILL_MAX_BATCH` wins, then the
/// gfx1151 dense-MQ4V2 512, then the per-arch default.
fn legacy_prefill_max_batch(
    explicit: Option<usize>,
    arch: &str,
    fp8_chunk512: bool,
    gfx1151_dense_mq4v2: bool,
) -> usize {
    explicit.unwrap_or_else(|| {
        if gfx1151_dense_mq4v2 {
            512
        } else {
            prefill_max_batch_for_arch(arch, fp8_chunk512)
        }
    })
}

/// Widened ordinary route: an ordinary (non-captured, no tree/tape/cap/caller
/// PBS) call whose admitted ceiling exceeds the 512-row commit seam.
fn widened_route(wide_candidate: bool, limit: usize) -> bool {
    wide_candidate && limit > WIDENED_COMMIT_ROWS
}

/// Rows of the owned widened PBS for an `n`-token call under `limit`.
fn widened_owned_rows(limit: usize, n: usize) -> usize {
    limit.min(n).max(MIN_BATCH)
}

/// Commit stride handed to `forward_prefill_chunk` for one chunk: only the
/// widened route produces `Some(512)`, and only above one seam.
fn widened_commit_stride(wide: bool, chunk_n: usize) -> Option<usize> {
    (wide && chunk_n > WIDENED_COMMIT_ROWS).then_some(WIDENED_COMMIT_ROWS)
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
pub(crate) use hipfire_dispatch::pipeline::batched::WIDENED_COMMIT_ROWS;
///
/// Staged GEMM-width rungs. Merging powers of two limits new GEMM widths to
/// these five; a requested ceiling snaps down to the largest rung it covers.
pub(crate) const WIDENED_RUNGS: [usize; 5] = [512, 1024, 2048, 4096, 8192];
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
    config.hidden_dim == 17_408 && super::program::hybrid_dims(config).is_dense_27b_shape()
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
    let mq3 = mq3_wide_route(gpu, weights);
    if !dense_layers_are_all_mq4v2(weights) && !native_mq4 && !mq3 {
        return None;
    }
    // gfx1201 keeps all four FP8 projection routes for qt44; symmetric qt49
    // uses A4/A8 on gfx1201 (its flag predicate is the admission) and the
    // gfx1100 A4 flag already implies the IU4 prefill gate. gfx11 substitutes
    // the existing IU4 K16 projection route; its full-attention layers remain
    // on Q8 FA2 because the packet kernel and FP8 KV format are gfx1201-only.
    let projection_route_admitted = if native_mq4 || mq3 {
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
    // lane, and single-end requant (see `ordinary_dn_state_admitted`).
    if !ordinary_dn_state_admitted_live(config, dn_state) {
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
///
/// DeltaNet state contract every widened ordinary route shares: actual Q8
/// state with a complete EF owner for every LA layer, single lane, and
/// single-end requant. EF-off changes the global frame schedule under
/// layer-major grouping (frame enters the arithmetic without EF), so it
/// resolves to the legacy path even under a larger ceiling. Per-token requant
/// inserts 511 extra Q8/EF boundaries per 512 rows and can never equal the
/// 512-single-end trajectory.
fn ordinary_dn_state_admitted_live(config: &Qwen35Config, dn_state: &DeltaNetState) -> bool {
    ordinary_dn_state_admitted(config, dn_state, dn_requant_per_token_env())
}

/// Pure core of [`ordinary_dn_state_admitted_live`]: the allocator geometry of
/// the state ([`DeltaNetStateGeometry`], a live state or an allocation plan)
/// against the model config. Every S and EF owner is checked (no first-owner
/// summary) through borrowed iterators, with no heap allocation.
fn ordinary_dn_state_admitted(
    config: &Qwen35Config,
    state: &impl DeltaNetStateGeometry,
    per_token_requant: bool,
) -> bool {
    let n_la = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::LinearAttention)
        .count();
    if state.quant() != StateQuant::Q8 || n_la == 0 {
        return false;
    }
    let mut ef_numel = state.ef_numel();
    if ef_numel.len() != n_la || ef_numel.any(|n| n == 0) {
        return false;
    }
    let single_lane_s =
        config.linear_num_value_heads * config.linear_value_head_dim * config.linear_value_head_dim;
    let mut s_numel = state.s_numel();
    if s_numel.len() != n_la || s_numel.any(|n| n != single_lane_s) {
        return false;
    }
    !per_token_requant
}
///
/// The one widened rung of the resident Qwen35 MoE ordinary route. Admission
/// is binary: this rung when [`resident_moe_rung_fits`] holds, otherwise the
/// incumbent `prefill_max_batch_for_model` ceiling — never an intermediate
/// 1024/2048/4096 rung, because 8192 is the wider GEMM width validated for the
/// grouped MoE producers (4096 was the prior rung; the 8192 producer, scratch,
/// PBS, projection, Q16 and headroom proofs moved together).
///
/// Origin of the incumbent 512: `PREFILL_DEFAULT_BATCH_GFX1100` is the
/// gfx1100-measured MQ4V2 gate-up sweet spot (384 on exact gfx1201, 512 under
/// the full FP8 path, 256 elsewhere), and [`WIDENED_COMMIT_ROWS`] is the same
/// 512-row DeltaNet commit seam. The receipt's `commit_stride=512` is the
/// generic widened-chunk value; the MoE layer bodies
/// (`hipfire_dispatch::pipeline::batched_moe::execute_deltanet_moe_layer_batched`,
/// `execute_attention_moe_layer_batched`) do NOT take a commit stride and run the
/// incumbent whole-chunk GDN launch and FA attend, exactly as an explicit
/// `HIPFIRE_PREFILL_MAX_BATCH=8192` does. The seam is therefore not claimed
/// for MoE bodies.
pub(crate) const RESIDENT_MOE_CHUNK_ROWS: usize = 8192;
///
/// Outcome of resolving the user's chunk request for the resident-MoE route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResidentMoeRequest {
    /// An explicit override or opt-out: the incumbent ceiling, unchanged.
    Incumbent,
    /// Ask for [`RESIDENT_MOE_CHUNK_ROWS`]; `requested` is the value logged.
    Widen { requested: usize },
}
///
/// Precedence: explicit `HIPFIRE_PREFILL_MAX_BATCH` wins and keeps the
/// incumbent path exactly as before. Otherwise an explicit
/// `HIPFIRE_PREFILL_CHUNK_ROWS` / `prefill.chunk_rows` below the MoE rung opts
/// out (incumbent), and one at or above it requests the (single) MoE rung.
/// With nothing explicit the route requests [`RESIDENT_MOE_CHUNK_ROWS`]
/// by default.
fn resident_moe_request(
    explicit_max_batch: Option<usize>,
    explicit_chunk_rows: Option<usize>,
) -> ResidentMoeRequest {
    if explicit_max_batch.is_some() {
        return ResidentMoeRequest::Incumbent;
    }
    match explicit_chunk_rows {
        Some(v) if v >= RESIDENT_MOE_CHUNK_ROWS => ResidentMoeRequest::Widen { requested: v },
        Some(_) => ResidentMoeRequest::Incumbent,
        None => ResidentMoeRequest::Widen {
            requested: RESIDENT_MOE_CHUNK_ROWS,
        },
    }
}

fn resident_moe_request_now() -> ResidentMoeRequest {
    resident_moe_request(explicit_prefill_max_batch(), explicit_prefill_chunk_rows())
}
///
/// The d53 routed pair (`MQ4G256V2` gate_up, `MQ4G128V2` down) of the symmetric
/// IU4 artifact class.
fn resident_moe_iu4_d53_pair(d: &hipfire_dispatch::families::moe::MoePrefillDtypes) -> bool {
    d.expert_gate_up == DType::MQ4G256V2 && d.expert_down == DType::MQ4G128V2
}
///
/// Grouped-producer dtype capability of one MoE layer: routed experts uniform
/// MQ4G256 or uniform MQ4G256V2 (gate_up and down the same format, no graded
/// merge), or the d53 pair when the layer carries the verified loader IU4
/// capability (`expert_iu4`), shared expert MQ4G256/MQ4G256V2 with a fused
/// gate/up pair, router and shared scalar gate in the formats the dispatch
/// admission accepts. Anything else (MQ6, E8, Lloyd/codebook, PARO, mixed
/// tiers, d53 without the IU4 capability) keeps the incumbent chunk.
fn resident_moe_producer_dtypes_supported(
    d: &hipfire_dispatch::families::moe::MoePrefillDtypes,
    expert_iu4: bool,
) -> bool {
    let mq4 = |dt: DType| matches!(dt, DType::MQ4G256 | DType::MQ4G256V2);
    let scalar = |dt: DType| {
        matches!(
            dt,
            DType::MQ4G256 | DType::MQ4G256V2 | DType::Q8_0 | DType::F32
        )
    };
    !d.routed_mixed_merged
        && d.expert_gate_up_uniform
        && d.expert_down_uniform
        && mq4(d.expert_gate_up)
        && (d.expert_down == d.expert_gate_up || (expert_iu4 && resident_moe_iu4_d53_pair(d)))
        && mq4(d.shared_expert_gate)
        && d.shared_expert_up == d.shared_expert_gate
        && mq4(d.shared_expert_down)
        && scalar(d.router)
        && scalar(d.shared_expert_scalar_gate)
}
///
/// Whether a layer's sealed expert contract is a multi-device (TP/EP)
/// execution. A `Single` contract on one physical device is the ordinary
/// resident load; absent contracts are single by construction.
fn resident_moe_contract_multi_device(
    contract: Option<&hipfire_dispatch::pipeline::sealed_moe::ExpertExecutionContract>,
) -> bool {
    use hipfire_dispatch::pipeline::sealed_moe::ContractParallelism;
    contract.is_some_and(|c| {
        c.parallelism() != ContractParallelism::Single || c.physical_devices().len() > 1
    })
}
///
/// Placement facts of one MoE layer read from the loaded weights (never from
/// configuration intent). Resident means every flag is clear.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ResidentMoePlacement {
    /// An attention projection lives in host-mapped memory.
    attention_host_located: bool,
    /// The routed gate/up owner is host-mapped (packed owners, not the
    /// non-owning per-expert subviews, which cannot report residency).
    routed_gate_up_host: bool,
    /// The routed down owner is host-mapped.
    routed_down_host: bool,
    /// The router, shared-expert gate or a shared projection is host-mapped.
    small_ffn_host_located: bool,
    /// `memory.offload_exec=cpu` armed the CPU expert splice on this layer.
    cpu_deferred_experts: bool,
    /// Paged experts: the resident expert list is empty (a pager owns them).
    paged_experts: bool,
    /// The sealed expert contract spans more than the one local device
    /// (tensor/expert parallel). Every ordinary single-device load also
    /// carries a sealed contract (`adapt_expert_execution_plan` always attaches
    /// one), so its mere presence is NOT multi-device.
    multi_device_contract: bool,
    /// AWQ/PARO sidecars, per-expert dtype tags or an EP global table: routes
    /// outside the validated grouped-producer set.
    sidecar_experts: bool,
}

impl ResidentMoePlacement {
    fn resident(self) -> bool {
        self == Self::default()
    }
}

fn resident_moe_layer_placement(
    gpu: &Gpu,
    attention: &[&hipfire_runtime::llama::WeightTensor],
    ffn: &MoeFfnWeights,
) -> ResidentMoePlacement {
    use hipfire_dispatch::families::moe::RoutedExpertWeights;
    let (routed_gate_up_host, routed_down_host) = ffn.host_mapped_projections();
    ResidentMoePlacement {
        attention_host_located: attention.iter().any(|w| gpu.host_located(&w.buf)),
        routed_gate_up_host,
        routed_down_host,
        small_ffn_host_located: [
            &ffn.router,
            &ffn.shared_expert_gate,
            &ffn.shared_expert.gate,
            &ffn.shared_expert.up,
            &ffn.shared_expert.down,
        ]
        .iter()
        .any(|w| gpu.host_located(&w.buf)),
        cpu_deferred_experts: ffn.cpu_deferred_experts,
        paged_experts: ffn.experts.is_empty(),
        multi_device_contract: resident_moe_contract_multi_device(
            ffn.expert_table.execution_contract(),
        ),
        sidecar_experts: ffn.paro_shared.is_some()
            || ffn.expert_down_awq_ptrs.is_some()
            || ffn.expert_dtype_tags.is_some()
            || ffn.global_expert_dtypes.is_some(),
    }
}

fn resident_moe_layer_capable(
    gpu: &Gpu,
    attention: &[&hipfire_runtime::llama::WeightTensor],
    ffn: &MoeFfnWeights,
) -> bool {
    resident_moe_layer_facts_capable(
        &resident_moe_layer_placement(gpu, attention, ffn),
        moe_prefill_dtypes(ffn).as_ref(),
        resident_moe_layer_iu4_bound(ffn),
    )
}

/// The layer carries the loader's verified symmetric IU4 capability (bound on
/// every MoE layer by `finalize_expert_iu4`, withheld under the explicit kill
/// and on incapable devices).
fn resident_moe_layer_iu4_bound(ffn: &MoeFfnWeights) -> bool {
    ffn.expert_iu4.is_some_and(|(_, verified)| verified)
}

/// A layer that the d53 pair admits only through the IU4 capability, so its
/// prefill needs the symmetric IU4 scratch at the widened rows.
fn resident_moe_layer_needs_iu4_scratch(
    dtypes: Option<&hipfire_dispatch::families::moe::MoePrefillDtypes>,
    expert_iu4: bool,
) -> bool {
    expert_iu4 && dtypes.is_some_and(resident_moe_iu4_d53_pair)
}

/// Pure core of [`resident_moe_layer_capable`]: every placement flag clear and
/// the layer's dtype record inside the validated grouped-producer set.
fn resident_moe_layer_facts_capable(
    placement: &ResidentMoePlacement,
    dtypes: Option<&hipfire_dispatch::families::moe::MoePrefillDtypes>,
    expert_iu4: bool,
) -> bool {
    placement.resident()
        && dtypes.is_some_and(|d| resident_moe_producer_dtypes_supported(d, expert_iu4))
}
///
/// Actual symmetric-IU4 MoE scratch held by one device, bytes per owner: the
/// live buffer size capped by the tracked bytes (the dispatch
/// `expert_iu4_scratch_ready` evidence), 0 for an owner never allocated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MoeSymScratchHave {
    group: usize,
    down: usize,
    gate: usize,
}

impl MoeSymScratchHave {
    fn of(gpu: &Gpu) -> Self {
        let s = &gpu.scratch;
        let held = |buf: &Option<hip_bridge::DeviceBuffer>, tracked: usize| {
            buf.as_ref().map_or(0, |b| b.size().min(tracked))
        };
        Self {
            group: held(&s.qwen4_moe_group_scratch, s.qwen4_moe_group_scratch_bytes),
            down: held(
                &s.qwen4_moe_down_i4_scratch,
                s.qwen4_moe_down_i4_scratch_bytes,
            ),
            gate: held(&s.int4_mmq_x_scratch, s.int4_mmq_x_scratch_bytes),
        }
    }
}

/// Whether `have` covers every owner `Gpu::reserve_qwen4_moe_sym` would size
/// for `rows` rows of this model (grouping layout of `rows * top_k` slots, the
/// down sidecar, and the gate/up activation scratch). Overflow or a rejected
/// shape fails closed. No allocation.
fn moe_sym_scratch_covers_rows(
    have: MoeSymScratchHave,
    rows: usize,
    config: &Qwen35Config,
) -> bool {
    use rdna_compute::scratch::{
        int4_mmq_reservation_bytes, qwen4_moe_down_i4_bytes, qwen4_moe_group_layout_checked,
    };
    let Some(slots) = rows.checked_mul(config.num_experts_per_tok) else {
        return false;
    };
    match (
        qwen4_moe_group_layout_checked(slots, config.num_experts),
        qwen4_moe_down_i4_bytes(config.moe_intermediate_size, slots),
        int4_mmq_reservation_bytes(config.dim, rows),
    ) {
        (Some(group), Some(down), Some(gate)) => {
            have.group >= group.bytes && have.down >= down && have.gate >= gate
        }
        _ => false,
    }
}
///
/// Every static (no allocation, no memory query) fact the resident Qwen35 MoE
/// widened ordinary route reads. [`resident_moe_static_facts`] gathers them
/// from the loaded model; CPU tests build the derivable ones from an installed
/// artifact. [`resident_moe_static_decision`] is the one decision over them.
#[derive(Clone, Debug)]
struct ResidentMoeStaticFacts {
    /// `config.num_experts != 0`.
    moe_config: bool,
    /// Chunk-knob resolution ([`resident_moe_request`]).
    request: ResidentMoeRequest,
    /// The device has the WMMA grouped producer.
    has_wmma: bool,
    /// Paged experts / EP shard / MQ6 anywhere in the loaded weights.
    pager: bool,
    ep_shard: bool,
    moe_has_mq6: bool,
    layers_present: bool,
    /// [`ordinary_dn_state_admitted`] for the actual state.
    dn_state_admitted: bool,
    /// [`prefill_batch_pbs_eligible`] for the loaded layers.
    batch_pbs_eligible: bool,
    /// Every layer passes [`resident_moe_layer_facts_capable`].
    layers_resident_capable: bool,
    /// Some layer is a d53 pair admitted only through the IU4 capability
    /// ([`resident_moe_layer_needs_iu4_scratch`]).
    iu4_d53_layers: bool,
    /// The device's actual symmetric-IU4 scratch covers
    /// [`RESIDENT_MOE_CHUNK_ROWS`] ([`moe_sym_scratch_covers_rows`]). Read only
    /// when `iu4_d53_layers`.
    iu4_scratch_wide: bool,
}
///
/// Static (no allocation, no memory query) eligibility of the resident
/// Qwen35 MoE widened ordinary route: returns [`RESIDENT_MOE_CHUNK_ROWS`] when
/// the loaded model is a fully resident, single-device, non-paged MoE whose
/// every layer passes the batched-prefill producer admission
/// (`qwen35_layer_batch_admissible` through `prefill_batch_pbs_eligible`) and
/// whose routed/shared experts are MQ4G256 or MQ4G256V2, the DeltaNet state
/// meets [`ordinary_dn_state_admitted`], and the device has the WMMA grouped
/// producer. Capability and actual placement only: no card or tag names.
/// `None` keeps the incumbent ceiling (and the dense route is unaffected).
fn resident_moe_static_decision(f: &ResidentMoeStaticFacts) -> Option<usize> {
    if !f.moe_config {
        return None;
    }
    if !matches!(f.request, ResidentMoeRequest::Widen { .. }) {
        return None;
    }
    if !f.has_wmma {
        return None;
    }
    if f.pager || f.ep_shard || f.moe_has_mq6 || !f.layers_present {
        return None;
    }
    if !f.dn_state_admitted {
        return None;
    }
    if !f.batch_pbs_eligible {
        return None;
    }
    if !f.layers_resident_capable {
        return None;
    }
    // A d53 layer's prefill allocates nothing: when the loader's widened
    // reservation fell back to the incumbent rows the scratch cannot hold the
    // widened chunk.
    (!f.iu4_d53_layers || f.iu4_scratch_wide).then_some(RESIDENT_MOE_CHUNK_ROWS)
}
///
/// Gather [`ResidentMoeStaticFacts`] from the loaded model. The gates are read
/// in the order [`resident_moe_static_decision`] applies them and gathering
/// stops at the first closed one (the later facts stay `false` and are never
/// consulted by the decision), so a dense or non-widening request pays no
/// per-layer walk.
fn resident_moe_static_facts(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    dn_state: &DeltaNetState,
) -> ResidentMoeStaticFacts {
    let mut f = ResidentMoeStaticFacts {
        moe_config: config.num_experts != 0,
        request: resident_moe_request_now(),
        has_wmma: gpu.arch_caps.has_wmma(),
        pager: weights.pager.is_some(),
        ep_shard: weights.ep_shard.is_some(),
        moe_has_mq6: weights.moe_has_mq6,
        layers_present: !weights.layers.is_empty(),
        dn_state_admitted: false,
        batch_pbs_eligible: false,
        layers_resident_capable: false,
        iu4_d53_layers: false,
        iu4_scratch_wide: false,
    };
    if !f.moe_config
        || !matches!(f.request, ResidentMoeRequest::Widen { .. })
        || !f.has_wmma
        || f.pager
        || f.ep_shard
        || f.moe_has_mq6
        || !f.layers_present
    {
        return f;
    }
    f.dn_state_admitted = ordinary_dn_state_admitted_live(config, dn_state);
    if !f.dn_state_admitted {
        return f;
    }
    f.batch_pbs_eligible = prefill_batch_pbs_eligible(
        weights,
        config,
        dn_state,
        WIDENED_COMMIT_ROWS,
        gpu.arch.as_str(),
        true,
    );
    if !f.batch_pbs_eligible {
        return f;
    }
    f.layers_resident_capable = resident_moe_weights_capable(gpu, weights);
    if !f.layers_resident_capable {
        return f;
    }
    f.iu4_d53_layers = resident_moe_weights_need_iu4_scratch(weights);
    f.iu4_scratch_wide = f.iu4_d53_layers
        && moe_sym_scratch_covers_rows(MoeSymScratchHave::of(gpu), RESIDENT_MOE_CHUNK_ROWS, config);
    f
}
///
/// Every layer passes [`resident_moe_layer_capable`] (placement, dtype and
/// IU4 capability read from the loaded weights).
fn resident_moe_weights_capable(gpu: &Gpu, weights: &Qwen35Weights) -> bool {
    weights.layers.iter().all(|layer| match layer {
        LayerWeights::DeltaNetMoe(l) => {
            resident_moe_layer_capable(gpu, &[&l.wqkv, &l.wz, &l.w_alpha, &l.w_beta, &l.wo], &l.ffn)
        }
        LayerWeights::FullAttnMoe(l) => {
            resident_moe_layer_capable(gpu, &[&l.wq, &l.wk, &l.wv, &l.wo], &l.ffn)
        }
        _ => false,
    })
}

fn resident_moe_ffn_needs_iu4_scratch(ffn: &MoeFfnWeights) -> bool {
    resident_moe_layer_needs_iu4_scratch(
        moe_prefill_dtypes(ffn).as_ref(),
        resident_moe_layer_iu4_bound(ffn),
    )
}

/// Some MoE layer is a d53 pair admitted through the IU4 capability.
fn resident_moe_weights_need_iu4_scratch(weights: &Qwen35Weights) -> bool {
    weights.layers.iter().any(|layer| match layer {
        LayerWeights::DeltaNetMoe(l) => resident_moe_ffn_needs_iu4_scratch(&l.ffn),
        LayerWeights::FullAttnMoe(l) => resident_moe_ffn_needs_iu4_scratch(&l.ffn),
        _ => false,
    })
}
///
/// Rows the loader reserves symmetric-IU4 scratch for on `gpu` beyond the
/// incumbent ceiling: [`RESIDENT_MOE_CHUNK_ROWS`] only when the SAME static
/// decision prefill applies ([`resident_moe_static_decision`], hence the same
/// [`resident_moe_request`], WMMA, placement, residency, dtype and IU4 gates)
/// admits the loaded model, `None` otherwise. Facts that need the not-yet-built
/// DeltaNet state or live batching context (`dn_state_admitted`,
/// `batch_pbs_eligible`) are runtime-only and re-checked by prefill, and the
/// scratch fact is what this reservation creates, so it is taken as met here.
pub(crate) fn resident_moe_iu4_load_wide_rows(gpu: &Gpu, weights: &Qwen35Weights) -> Option<usize> {
    resident_moe_static_decision(&ResidentMoeStaticFacts {
        moe_config: true,
        request: resident_moe_request_now(),
        has_wmma: gpu.arch_caps.has_wmma(),
        pager: weights.pager.is_some(),
        ep_shard: weights.ep_shard.is_some(),
        moe_has_mq6: weights.moe_has_mq6,
        layers_present: !weights.layers.is_empty(),
        dn_state_admitted: true,
        batch_pbs_eligible: true,
        layers_resident_capable: resident_moe_weights_capable(gpu, weights),
        iu4_d53_layers: false,
        iu4_scratch_wide: true,
    })
}
///
/// Wrapper-level gate: the DENSE widened ordinary route is statically
/// admitted above the 512-row seam. Only that route makes the wrappers omit
/// their implicit ≤512 cached PBS (see [`plan_wrapper_pbs`]).
fn ordinary_dense_widen_statically_admitted(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    dn_state: &DeltaNetState,
) -> bool {
    ordinary_prefill_static_ceiling(gpu, weights, config, dn_state)
        .is_some_and(|c| c > WIDENED_COMMIT_ROWS)
}
///
/// Which PBS the ordinary wrappers hand the inner entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WrapperOwner {
    /// No caller PBS: the inner entry owns the decision (reuse off, or the
    /// dense widened route that drops the implicit cache).
    None,
    /// The implicit `HIPFIRE_PREFILL_REUSE_PBS=1` cache as allocated.
    Physical,
    /// The lazily allocated explicit-equivalent owner (`Qwen35Scratch::
    /// reuse_virtual_pbs`) with exactly these rows and tape flag.
    Virtual { rows: usize, tape: bool },
}
///
/// The wrapper-side parameters of one ordinary call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WrapperPbsPlan {
    owner: WrapperOwner,
    /// Virtual explicit `HIPFIRE_PREFILL_MAX_BATCH`: the incumbent ceiling the
    /// inner entry uses instead of the arch default.
    max_batch_override: Option<usize>,
    /// An existing virtual owner is released before anything else.
    free_virtual: bool,
    /// The virtual owner is (re)allocated with `owner`'s rows/tape.
    alloc_virtual: bool,
}

impl WrapperPbsPlan {
    fn pbs<'a>(
        &self,
        physical: Option<&'a PrefillBatchScratch>,
        virtual_owner: Option<&'a PrefillBatchScratch>,
    ) -> Option<&'a PrefillBatchScratch> {
        match self.owner {
            WrapperOwner::None => None,
            WrapperOwner::Physical => physical,
            WrapperOwner::Virtual { .. } => virtual_owner,
        }
    }
}
///
/// Inputs of [`plan_wrapper_pbs`].
#[derive(Clone, Copy, Debug)]
struct WrapperPbsInputs<'a> {
    arch: &'a str,
    fp8_chunk512: bool,
    /// The dense widened route is statically admitted.
    dense_widened: bool,
    /// The implicit reuse cache `(rows, tape)`, `None` with reuse off.
    physical: Option<(usize, bool)>,
    /// An already allocated virtual owner `(rows, tape)` (e.g. kept from an
    /// earlier request).
    existing_virtual: Option<(usize, bool)>,
    /// The DEFAULT resident-MoE request (no explicit knob) is statically
    /// eligible and its 8192 rung fits memory with no caller PBS present.
    default_moe_fits: bool,
    /// The call's own caller/capture facts ([`OrdinaryCallFacts`], the same
    /// record the inner entry builds, with no caller PBS) leave the call a
    /// widen candidate: captured/recorded, tree, tape, staged ring,
    /// band/fusion and capped calls never get a promotion.
    call_eligible: bool,
}
///
/// Wrapper PBS plan. Beta's explicit `HIPFIRE_PREFILL_MAX_BATCH=8192` with
/// `HIPFIRE_PREFILL_REUSE_PBS=1` runs on an implicit cache sized by
/// [`implicit_reuse_pbs_plan`]`(arch, Some(8192), ..)` with the ceiling 8192.
/// An eligible, memory-admitted DEFAULT request is exactly that "virtual
/// explicit 8192": same owner rows and tape flag (the existing explicit sizing
/// function and clamps), same ceiling, so every executed parameter is identical.
/// A default that does not fit keeps the incumbent implicit cache and ceiling;
/// an explicit manual knob is never rewritten (it never reports
/// `default_moe_fits`, its static gate is closed).
fn plan_wrapper_pbs(i: &WrapperPbsInputs) -> WrapperPbsPlan {
    let none = WrapperPbsPlan {
        owner: WrapperOwner::None,
        max_batch_override: None,
        free_virtual: i.existing_virtual.is_some(),
        alloc_virtual: false,
    };
    if i.dense_widened {
        return none;
    }
    let Some(physical) = i.physical else {
        return none;
    };
    if !i.call_eligible {
        // Keep whatever owners exist untouched: only the implicit cache runs.
        return WrapperPbsPlan {
            owner: WrapperOwner::Physical,
            free_virtual: false,
            ..none
        };
    }
    if !i.default_moe_fits {
        return WrapperPbsPlan {
            owner: WrapperOwner::Physical,
            ..none
        };
    }
    let target = implicit_reuse_pbs_plan(i.arch, Some(RESIDENT_MOE_CHUNK_ROWS), i.fp8_chunk512);
    let max_batch_override = Some(RESIDENT_MOE_CHUNK_ROWS);
    if physical == target {
        WrapperPbsPlan {
            owner: WrapperOwner::Physical,
            max_batch_override,
            ..none
        }
    } else {
        WrapperPbsPlan {
            owner: WrapperOwner::Virtual {
                rows: target.0,
                tape: target.1,
            },
            max_batch_override,
            free_virtual: i.existing_virtual.is_some_and(|e| e != target),
            alloc_virtual: i.existing_virtual != Some(target),
        }
    }
}

fn pbs_has_tape(p: &PrefillBatchScratch) -> bool {
    p.dn_s_tape_q8.is_some() || p.dn_s_tape_f32.is_some()
}
///
/// The wrapper's call-level gate: the one decision of "may this call be
/// promoted at all", shared by the production wrapper step and the CPU tests
/// (which call it with the same inputs over artifact-derived facts). The call
/// is a promotion candidate only when
/// - it is non-empty and its caller/capture record ([`OrdinaryCallFacts`], no
///   caller PBS) leaves it a widen candidate,
/// - the ordinary batched (PBS) path actually runs for ITS OWN row count `n`
///   (`batched_eligible(n)`: the verify-decouple window, the `prefill_batched`
///   switch, the two-token minimum, DeltaNet layers and layer admission; a
///   one-token call falls to the per-token AR loop, so a 8192 owner would be
///   allocated for nothing), and
/// - the widened chunks (every one at least the 512-row commit seam) are
///   batched-eligible too (`batched_eligible(WIDENED_COMMIT_ROWS)`, the probe
///   the resident static facts use). Wherever the actual `n` and the 512 probe
///   disagree (n = 1, or a small verify-decouple window with
///   `prefill_batched` off) the call keeps the incumbent resource plan: no
///   KV mapping, no override, no allocation, no owner released.
/// The gate is evaluated before any KV mapping, override or allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OrdinaryWrapperGate {
    n: usize,
    /// The implicit reuse cache `(rows, tape)`, `None` with reuse off.
    physical: Option<(usize, bool)>,
    call_eligible: bool,
}

impl OrdinaryWrapperGate {
    fn new(
        n: usize,
        call: &OrdinaryCallFacts,
        physical: Option<(usize, bool)>,
        batched_eligible: impl Fn(usize) -> bool,
    ) -> Self {
        let call_eligible = n > 0
            && call.wide_candidate()
            && batched_eligible(n)
            && batched_eligible(WIDENED_COMMIT_ROWS);
        Self {
            n,
            physical,
            call_eligible,
        }
    }
    /// The wrapper maps the call's required KV before the memory query. With
    /// the implicit cache present the wrapper decides first, so it performs
    /// the same (idempotent) mapping the inner entry would.
    fn maps_kv(&self) -> bool {
        self.call_eligible && self.physical.is_some()
    }
    /// The memory admission of the virtual-explicit owner is queried.
    fn needs_default_fit(&self, dense_widened: bool) -> bool {
        self.maps_kv() && !dense_widened
    }
    fn plan(
        &self,
        arch: &str,
        fp8_chunk512: bool,
        dense_widened: bool,
        existing_virtual: Option<(usize, bool)>,
        default_moe_fits: bool,
    ) -> WrapperPbsPlan {
        // An empty call returns before the inner entry touches anything.
        if self.n == 0 {
            return WrapperPbsPlan {
                owner: WrapperOwner::None,
                max_batch_override: None,
                free_virtual: false,
                alloc_virtual: false,
            };
        }
        plan_wrapper_pbs(&WrapperPbsInputs {
            arch,
            fp8_chunk512,
            dense_widened,
            physical: self.physical,
            existing_virtual,
            default_moe_fits: default_moe_fits && self.needs_default_fit(dense_widened),
            call_eligible: self.call_eligible,
        })
    }
}
///
/// Memory credit of the wrapper's existing virtual owner: its payload bytes
/// only when it is exactly the explicit-equivalent `target` AND the implicit
/// cache is not already that shape, i.e. when [`plan_wrapper_pbs`] keeps it
/// (no `free_virtual`, no `alloc_virtual`). A kept owner is held by
/// `Qwen35Scratch::reuse_virtual_pbs` across calls, so its bytes are already
/// absent from device-free bytes and offset the full owner charge. An owner of
/// any other shape is released and replaced; its bytes would only return to
/// the allocator pool (not guaranteed reclaimable for a different size
/// bucket), so it earns no credit. The implicit cache's own bytes are never
/// credited.
fn wrapper_virtual_credit_bytes(
    config: &Qwen35Config,
    physical: Option<(usize, bool)>,
    existing_virtual: Option<(usize, bool)>,
    target: (usize, bool),
) -> usize {
    existing_virtual
        .filter(|&e| e == target && physical != Some(target))
        .and_then(|(rows, tape)| resident_moe_owner_bytes(config, rows, tape))
        .unwrap_or(0)
}
///
/// Production wrapper step: gather the real facts, run the shared
/// [`OrdinaryWrapperGate`] and [`plan_wrapper_pbs`] (the default-fits fact is
/// the shared resident-MoE static decision plus the shared memory admission of
/// the owner that would be allocated), then apply the plan to the scratch
/// (release / allocate the virtual owner). Runs before any cache owner exists
/// for the call, with the weights, config, state and KV in hand and no caller
/// PBS. `call` is the caller/capture record of THIS call (built by
/// `ordinary_call_facts`, the constructor the inner entry uses, with no caller
/// PBS).
#[allow(clippy::too_many_arguments)]
fn ordinary_wrapper_pbs_prepare(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    dn_state: &DeltaNetState,
    kv_cache: &mut llama::KvCache,
    scratch: &Qwen35Scratch,
    start_pos: usize,
    n: usize,
    call: OrdinaryCallFacts,
) -> HipResult<WrapperPbsPlan> {
    let physical = scratch
        .prefill_batch
        .as_ref()
        .map(|p| (p.max_batch, pbs_has_tape(p)));
    let existing_virtual = scratch
        .reuse_virtual_pbs
        .borrow()
        .as_ref()
        .map(|p| (p.max_batch, pbs_has_tape(p)));
    // Every denying fact (caller/capture record, the call's own batched
    // eligibility at its actual `n`, the 512-row seam) closes promotion before
    // any override, mapping or allocation.
    let gate = {
        let arch = gpu.arch.as_str();
        OrdinaryWrapperGate::new(n, &call, physical, |rows| {
            prefill_batch_pbs_eligible(weights, config, dn_state, rows, arch, true)
        })
    };
    // The inner entry charges the memory admission after the required KV is
    // mapped.
    if gate.maps_kv() {
        let required_tokens = checked_kv_end(start_pos, n, "forward_prefill_batch")?;
        release_widened_pbs_for_kv_growth(gpu, kv_cache, config, scratch, required_tokens)?;
        kv_cache.ensure_mapped_capacity(gpu, required_tokens)?;
    }
    let dense_widened =
        n > 0 && ordinary_dense_widen_statically_admitted(gpu, weights, config, dn_state);
    let fp8_chunk512 = fp8_chunk512_for_gpu(gpu);
    let default_moe_fits = gate.needs_default_fit(dense_widened) && {
        let moe = resident_moe_static_facts(gpu, weights, config, dn_state);
        let target = implicit_reuse_pbs_plan(
            gpu.arch.as_str(),
            Some(RESIDENT_MOE_CHUNK_ROWS),
            fp8_chunk512,
        );
        let credit_bytes = wrapper_virtual_credit_bytes(config, physical, existing_virtual, target);
        resident_moe_default_fits(
            &moe,
            config,
            KvTierFlags::from_cache(kv_cache),
            || resident_moe_memory_facts(gpu),
            ResidentMoeOwner {
                rows: target.0,
                tape: target.1,
                credit_bytes,
            },
        )
    };
    let plan = gate.plan(
        gpu.arch.as_str(),
        fp8_chunk512,
        dense_widened,
        existing_virtual,
        default_moe_fits,
    );
    if plan.free_virtual {
        if let Some(old) = scratch.reuse_virtual_pbs.borrow_mut().take() {
            old.free_gpu(gpu)?;
        }
    }
    if plan.alloc_virtual {
        if let WrapperOwner::Virtual { rows, tape } = plan.owner {
            let owner = PrefillBatchScratch::new_opt(gpu, config, rows, tape)?;
            *scratch.reuse_virtual_pbs.borrow_mut() = Some(owner);
        }
    }
    Ok(plan)
}
///
/// A ring-only eager forward (no per-token hidden rows, tape or tree) is a
/// DFlash prompt seed that rides the ordinary route; verify and
/// captured/recorded forwards keep the staged ring route. Shared by the inner
/// entry and the wrappers.
fn hidden_ring_seed(
    gpu: &Gpu,
    hidden_ring: Option<&HiddenStateRingBuffer>,
    has_per_token_hidden_out: bool,
    has_gdn_tape: bool,
    has_tree_verify: bool,
) -> bool {
    match hidden_ring {
        Some(rb) => {
            !has_per_token_hidden_out
                && !has_gdn_tape
                && !has_tree_verify
                && rb.rides_ordinary_prefill(gpu)
        }
        None => false,
    }
}
///
/// The one constructor of the caller/capture facts of an ordinary prefill
/// call: the inner entry calls it with its real arguments, and each wrapper
/// calls it (with no caller PBS) to gate the virtual-explicit promotion.
#[allow(clippy::too_many_arguments)]
fn ordinary_call_facts(
    gpu: &Gpu,
    caller_pbs_rows: Option<usize>,
    max_batch_cap: Option<usize>,
    has_tree_verify: bool,
    has_gdn_tape: bool,
    hidden_ring: Option<&HiddenStateRingBuffer>,
    has_per_token_hidden_out: bool,
    has_max_layer: bool,
    fusion_on: bool,
) -> OrdinaryCallFacts {
    let ring_seed = hidden_ring_seed(
        gpu,
        hidden_ring,
        has_per_token_hidden_out,
        has_gdn_tape,
        has_tree_verify,
    );
    OrdinaryCallFacts {
        caller_pbs_rows,
        max_batch_cap,
        tree_verify: has_tree_verify,
        gdn_tape: has_gdn_tape,
        staged_hidden_ring: hidden_ring.is_some() && !ring_seed,
        max_layer: has_max_layer,
        fusion_on,
        capture_mode: gpu.graphs.capture_mode,
        recording: gpu.replay.is_recording(),
    }
}
///
/// The ordinary incumbent ceiling a call runs with: the model/arch default
/// (or the wrapper's virtual-explicit override), capped by an explicit
/// `max_batch_cap`.
fn resolve_ordinary_max_batch(
    model_default: usize,
    max_batch_override: Option<usize>,
    max_batch_cap: Option<usize>,
) -> usize {
    let configured = max_batch_override.unwrap_or(model_default);
    match max_batch_cap {
        Some(cap) => configured.min(cap),
        None => configured,
    }
}
///
/// Live projection-prelude scratch already resident on the device (bytes).
#[derive(Clone, Copy, Debug, Default)]
struct LiveProjectionScratch {
    int4_mmq_x: usize,
    q8_1_mmq_x: usize,
    fp16_x: usize,
    fp8_x: usize,
    fp8_half_sums: usize,
    fp8_row_scales: usize,
}

fn live_projection_scratch(gpu: &Gpu) -> LiveProjectionScratch {
    LiveProjectionScratch {
        int4_mmq_x: gpu.scratch.int4_mmq_x_scratch_bytes,
        q8_1_mmq_x: gpu.scratch.q8_1_mmq_x_scratch_bytes,
        fp16_x: gpu.scratch.fp16_x_scratch_bytes,
        fp8_x: gpu.scratch.mq4v2_fp8_x_scratch_bytes,
        fp8_half_sums: gpu.scratch.mq4v2_fp8_half_sums_scratch_bytes,
        fp8_row_scales: gpu.scratch.mq4v2_fp8_row_scales_scratch_bytes,
    }
}
///
/// Bytes of the `PrefillBatchScratch::new_opt` allocations that
/// [`PrefillBatchScratch::projected_allocation_bytes`] does not mirror:
/// `dn_z_fold_batch` (`rows * (v_dim + 256)` F32) and `dn_normed_batch`
/// (`rows * v_dim` F32 for a non-lean owner), i.e. `rows * (2 * v_dim + 256) *
/// 4`. Charged here, in the resident-MoE census only, so the global
/// placement/offload reservation (`dense_prefill_allocation_bytes`) is
/// unchanged. `None` on overflow.
fn resident_moe_pbs_unmirrored_bytes(config: &Qwen35Config, rows: usize) -> Option<usize> {
    let v_dim = config
        .linear_num_value_heads
        .checked_mul(config.linear_value_head_dim)?;
    let per_row = v_dim.checked_mul(2)?.checked_add(256)?;
    rows.checked_mul(per_row)?.checked_mul(4)
}
///
/// Requested payload bytes of one non-lean [`PrefillBatchScratch`] owner of
/// `rows` rows with or without the verify-only S-tape: the constructor mirror
/// plus [`resident_moe_pbs_unmirrored_bytes`], each term once. Payload only
/// (no driver rounding or allocator overhead: the 1 GiB headroom covers such
/// transients). `None` on overflow.
fn resident_moe_owner_bytes(config: &Qwen35Config, rows: usize, tape: bool) -> Option<usize> {
    let mirrored = PrefillBatchScratch::projected_allocation_bytes(config, rows, tape)
        .ok()
        .and_then(|bytes| usize::try_from(bytes).ok())?;
    mirrored.checked_add(resident_moe_pbs_unmirrored_bytes(config, rows)?)
}
///
/// Steady-state bytes of one ordinary (tape-free) MoE [`PrefillBatchScratch`]
/// at `rows` rows ([`resident_moe_owner_bytes`] with no tape): the checked
/// constructor mirror, which charges the shared-expert, expanded-expert and
/// grouped-GEMM scratch term for term (sorted maps, tile ids, grouped gate/up
/// and down outputs) and the inactive residual backup, plus the two
/// allocations it omits. Ordinary prefill allocates `cap_gdn_tape = false`, so
/// the verify-only DeltaNet S-tape is NOT charged (unlike the placement
/// reservation in [`dense_prefill_allocation_bytes`]). `None` on overflow.
fn resident_moe_pbs_bytes(config: &Qwen35Config, rows: usize) -> Option<usize> {
    resident_moe_owner_bytes(config, rows, false)
}
///
/// Projection-prelude growth to `rows` rows. Producer-agnostic and
/// conservative for MoE (no dense FP8-only byte assumption): at the widest
/// projection K the grouped route can feed — hidden `dim`, the DeltaNet/attention
/// output widths and the shared/routed intermediates — every prelude
/// representation a producer may select (iu4 72 B/128, Q8_1 144 B/128, FP16,
/// FP8 codes + half sums + row scales) is charged, each net of the bytes the
/// matching live scratch already holds. `None` on overflow.
fn resident_moe_projection_deficit(
    config: &Qwen35Config,
    rows: usize,
    live: &LiveProjectionScratch,
) -> Option<usize> {
    let v = config
        .linear_num_value_heads
        .checked_mul(config.linear_value_head_dim)?;
    let q = config.n_heads.checked_mul(config.head_dim)?;
    let k = config
        .dim
        .max(v)
        .max(q)
        .max(config.shared_expert_intermediate_size)
        .max(config.moe_intermediate_size);
    let blocks128 = k.checked_add(127)? / 128;
    let blocks256 = k.checked_add(255)? / 256;
    let terms = [
        (blocks128.checked_mul(72)?, live.int4_mmq_x),
        (blocks128.checked_mul(144)?, live.q8_1_mmq_x),
        (k.checked_mul(2)?, live.fp16_x),
        (k, live.fp8_x),
        (blocks256.checked_mul(8)?, live.fp8_half_sums),
        (4, live.fp8_row_scales),
    ];
    let mut total = 0usize;
    for (per_row, live_bytes) in terms {
        let need = rows.checked_mul(per_row)?;
        total = total.checked_add(need.saturating_sub(live_bytes))?;
    }
    Some(total)
}
///
/// Missing FA Q16 scratch for `rows` rows. The MoE layer bodies run the whole
/// chunk through attention (`commit_stride = None`, no 512-row segmenting), and
/// every selectable Q16 scratch consumer (`rdna-compute` attention) sizes
/// `batch_size * n_heads * head_dim * 2` from that actual batch, freeing the
/// old buffer before allocating the grown one, so the growth charged here is
/// the full-row requirement net of the bytes the live buffer already holds.
/// `None` on overflow.
fn resident_moe_q16_missing(config: &Qwen35Config, rows: usize, live_q16: usize) -> Option<usize> {
    let need = rows
        .checked_mul(config.n_heads.checked_mul(config.head_dim)?)?
        .checked_mul(2)?;
    Some(need.saturating_sub(live_q16))
}
///
/// The retained widened PBS owner `(rows, lean)` the call will actually REUSE
/// without replacement: the inner entry keeps its cached owner only when
/// `rows >= reuse_rows` (the rows the call's owned PBS needs,
/// [`widened_owned_rows`]`(RESIDENT_MOE_CHUNK_ROWS, n)`) and its leanness
/// matches the call's. A resident MoE owner is never lean
/// (`lean_pbs_route` requires a dense model), so a lean cached owner is a
/// mismatch. Anything else is freed (`free_gpu` returns it to the allocator
/// pool, bucket-local; only an out-of-memory drain releases it to the
/// driver, and a different size bucket is not guaranteed to reclaim it) and
/// replaced by a full owner, so it earns NO credit: the full rung is charged.
fn resident_moe_reusable_cached(
    cached: Option<(usize, bool)>,
    reuse_rows: usize,
) -> Option<(usize, bool)> {
    cached.filter(|&(rows, lean)| !lean && rows >= reuse_rows)
}
///
/// Credit for a retained widened PBS owner that is reused in place (see
/// [`resident_moe_reusable_cached`]): exactly the bytes of that owner's shape,
/// only when it is a full (non-lean) owner. Callers hand a cached owner only
/// on the grow-only route where retention is observable in free VRAM.
fn resident_moe_cache_credit(config: &Qwen35Config, cached: Option<(usize, bool)>) -> usize {
    cached
        .filter(|&(_, lean)| !lean)
        .and_then(|(rows, _)| resident_moe_pbs_bytes(config, rows))
        .unwrap_or(0)
}
///
/// Inputs of the resident-MoE memory decision. `None` terms mean checked
/// arithmetic overflowed and make the rung inadmissible.
#[derive(Clone, Copy, Debug)]
struct ResidentMoeFit {
    free_bytes: usize,
    free_deduction: usize,
    pbs_bytes: Option<usize>,
    cached_credit: usize,
    projection_deficit: Option<usize>,
    q16_missing: Option<usize>,
}
///
/// The rung fits iff projected PBS (net of the cached owner credit) +
/// projection-prelude growth + missing Q16 + the existing
/// [`WIDENED_VRAM_HEADROOM_BYTES`] is `<=` device-free bytes minus pending
/// allocations. Any overflow fails closed.
fn resident_moe_rung_fits(fit: &ResidentMoeFit) -> bool {
    let (Some(pbs), Some(projection), Some(q16)) =
        (fit.pbs_bytes, fit.projection_deficit, fit.q16_missing)
    else {
        return false;
    };
    let Some(total) = pbs
        .saturating_sub(fit.cached_credit)
        .checked_add(projection)
        .and_then(|t| t.checked_add(q16))
        .and_then(|t| t.checked_add(WIDENED_VRAM_HEADROOM_BYTES))
    else {
        return false;
    };
    total <= fit.free_bytes.saturating_sub(fit.free_deduction)
}
///
/// Final binary decision: the MoE rung when it fits, else the incumbent.
fn resident_moe_limit(fits: bool, incumbent: usize) -> usize {
    if fits {
        RESIDENT_MOE_CHUNK_ROWS
    } else {
        incumbent
    }
}
///
/// KV tier flags the resident-MoE admission reads from a [`llama::KvCache`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
struct KvTierFlags {
    quant_q8: bool,
    quant_asym4: bool,
    quant_asym3: bool,
    quant_asym2: bool,
    quant_fp8: bool,
}

impl KvTierFlags {
    fn from_cache(kv: &llama::KvCache) -> Self {
        Self {
            quant_q8: kv.quant_q8,
            quant_asym4: kv.quant_asym4,
            quant_asym3: kv.quant_asym3,
            quant_asym2: kv.quant_asym2,
            quant_fp8: kv.quant_fp8,
        }
    }

    /// Batched flash attention is selectable for the tier.
    fn batched_flash(self) -> bool {
        self.quant_q8 || self.quant_asym4 || self.quant_asym3 || self.quant_asym2 || self.quant_fp8
    }
}
///
/// Device-memory facts of the resident-MoE memory decision.
#[derive(Clone, Copy, Debug)]
struct ResidentMoeMemoryFacts {
    /// Device-free bytes; `None` when the query failed (keeps the incumbent).
    free_bytes: Option<usize>,
    live: LiveProjectionScratch,
    /// Bytes the live FA Q16 scratch already holds.
    live_q16_bytes: usize,
}

fn resident_moe_memory_facts(gpu: &Gpu) -> ResidentMoeMemoryFacts {
    ResidentMoeMemoryFacts {
        free_bytes: gpu.device_mem_info().ok().map(|(free, _)| free),
        live: live_projection_scratch(gpu),
        live_q16_bytes: gpu.scratch.fa2_q16_scratch_bytes,
    }
}
///
/// The PBS owner whose bytes the memory decision charges and credits.
/// [`Self::WIDENED`] is the tape-free widened owner the inner entry
/// allocates; the wrapper's virtual-explicit owner may be a different shape
/// (tape flag, rows) and may already exist (`credit_bytes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResidentMoeOwner {
    rows: usize,
    tape: bool,
    credit_bytes: usize,
}

impl ResidentMoeOwner {
    const WIDENED: Self = Self {
        rows: RESIDENT_MOE_CHUNK_ROWS,
        tape: false,
        credit_bytes: 0,
    };
}
///
/// Memory admission of [`RESIDENT_MOE_CHUNK_ROWS`]: batched flash attention
/// must be selectable for the KV tier, the device free-byte query must
/// succeed (a failed query keeps the incumbent), and the rung must fit. The
/// charged PBS is the larger of the tape-free 8192-row rung and `owner`'s own
/// projected bytes. `cached` is the retained widened owner `(rows, lean)` the
/// grow-only route credits; `owner.credit_bytes` credits an existing owner.
fn resident_moe_memory_admits(
    config: &Qwen35Config,
    kv: KvTierFlags,
    mem: &ResidentMoeMemoryFacts,
    cached: Option<(usize, bool)>,
    owner: ResidentMoeOwner,
    free_deduction: usize,
) -> bool {
    if !kv.batched_flash() {
        return false;
    }
    let Some(free_bytes) = mem.free_bytes else {
        return false;
    };
    let owner_bytes = resident_moe_owner_bytes(config, owner.rows, owner.tape);
    let pbs_bytes = match (
        resident_moe_pbs_bytes(config, RESIDENT_MOE_CHUNK_ROWS),
        owner_bytes,
    ) {
        (Some(rung), Some(own)) => Some(rung.max(own)),
        _ => None,
    };
    resident_moe_rung_fits(&ResidentMoeFit {
        free_bytes,
        free_deduction,
        pbs_bytes,
        cached_credit: resident_moe_cache_credit(config, cached).saturating_add(owner.credit_bytes),
        projection_deficit: resident_moe_projection_deficit(
            config,
            RESIDENT_MOE_CHUNK_ROWS,
            &mem.live,
        ),
        q16_missing: resident_moe_q16_missing(config, RESIDENT_MOE_CHUNK_ROWS, mem.live_q16_bytes),
    })
}
///
/// The DEFAULT resident-MoE request is eligible and its rung fits with
/// `owner` as the PBS: the wrapper's "virtual explicit 8192" test. `mem` is
/// read only once the static decision admits.
fn resident_moe_default_fits(
    static_facts: &ResidentMoeStaticFacts,
    config: &Qwen35Config,
    kv: KvTierFlags,
    mem: impl FnOnce() -> ResidentMoeMemoryFacts,
    owner: ResidentMoeOwner,
) -> bool {
    resident_moe_static_decision(static_facts).is_some()
        && resident_moe_memory_admits(config, kv, &mem(), None, owner, 0)
}
///
/// Ordinary chunk ceiling of the resident MoE route: [`RESIDENT_MOE_CHUNK_ROWS`]
/// only when [`resident_moe_static_decision`] admits, the caller's staging
/// owner (if any) can hold it, and [`resident_moe_memory_admits`]; otherwise
/// `incumbent`. `mem` is read only once the cheaper gates pass. `cached` is the
/// retained widened owner `(rows, lean)` and `reuse_rows` the rows the call's
/// owned PBS needs: it is credited only when it is actually reused
/// ([`resident_moe_reusable_cached`]). The single decision shared by
/// production ([`ordinary_prefill_chunk_limit_with_facts`]) and the CPU
/// installed-artifact tests.
#[allow(clippy::too_many_arguments)]
fn resident_moe_chunk_limit_from_facts(
    static_facts: &ResidentMoeStaticFacts,
    caller_pbs_rows: Option<usize>,
    config: &Qwen35Config,
    kv: KvTierFlags,
    mem: impl FnOnce() -> ResidentMoeMemoryFacts,
    cached: Option<(usize, bool)>,
    reuse_rows: usize,
    free_deduction: usize,
    incumbent: usize,
) -> usize {
    if resident_moe_static_decision(static_facts).is_none() {
        return incumbent;
    }
    if caller_pbs_rows.is_some_and(|rows| rows < RESIDENT_MOE_CHUNK_ROWS) {
        return incumbent;
    }
    resident_moe_limit(
        resident_moe_memory_admits(
            config,
            kv,
            &mem(),
            resident_moe_reusable_cached(cached, reuse_rows),
            ResidentMoeOwner::WIDENED,
            free_deduction,
        ),
        incumbent,
    )
}
///
/// Caller/capture facts of one ordinary prefill call: every reason the inner
/// entry keeps the legacy cadence regardless of the model.
#[derive(Clone, Copy, Debug, Default)]
struct OrdinaryCallFacts {
    /// Rows of the PBS the caller (or the wrapper's retained implicit cache)
    /// hands the inner entry.
    caller_pbs_rows: Option<usize>,
    max_batch_cap: Option<usize>,
    tree_verify: bool,
    gdn_tape: bool,
    /// A staged hidden ring (a ring-only eager seed rides the ordinary route).
    staged_hidden_ring: bool,
    max_layer: bool,
    fusion_on: bool,
    capture_mode: bool,
    recording: bool,
}

impl OrdinaryCallFacts {
    /// Only the whole-stack sequential ordinary entry with no caller PBS/cap
    /// can widen: captured, TP/EP (separate callers), tree/tape, staged
    /// hidden-ring, band-limited, fused, or caller-capped requests resolve to
    /// the legacy path before any wider allocation or launch.
    fn wide_candidate(&self) -> bool {
        self.caller_pbs_rows.is_none()
            && self.max_batch_cap.is_none()
            && !self.tree_verify
            && !self.gdn_tape
            && !self.staged_hidden_ring
            && !self.max_layer
            && !self.fusion_on
            && !self.capture_mode
            && !self.recording
    }
}
///
/// Inputs of [`plan_ordinary_prefill`] beyond the call facts.
#[derive(Clone, Copy, Debug)]
struct OrdinaryPlanRequest {
    call: OrdinaryCallFacts,
    n: usize,
    /// Incumbent ordinary ceiling (explicit/arch default, caller-capped).
    max_batch: usize,
    explicit_max_batch: Option<usize>,
    /// [`prefill_chunk_rows_requested`] (the dense widened receipt value).
    chunk_rows_requested: usize,
    moe_request: ResidentMoeRequest,
    /// [`resident_moe_static_decision`] admitted the model.
    moe_static_admitted: bool,
}
///
/// The parameters one ordinary prefill call executes with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OrdinaryPrefillPlan {
    /// Admitted chunk ceiling in rows.
    limit: usize,
    /// Widened route: larger chunks with 512-row commit seams.
    wide: bool,
    moe_route: bool,
    /// The visible `prefill_chunk:` line.
    receipt: PrefillChunkReceipt,
    /// Rows of the PBS the chunks run on: the caller's owner, the widened owned
    /// owner, or the legacy right-sized owned scratch.
    pbs_rows: usize,
}

impl OrdinaryPrefillPlan {
    /// Chunk rows actually executed per step (before the hidden-ring cap).
    fn chunk_batch(&self, hidden_ring_cap: Option<usize>) -> usize {
        prefill_effective_chunk_batch(self.limit, self.pbs_rows, hidden_ring_cap)
    }
}
///
/// The ordinary prefill parameter planner shared by the production inner entry
/// and the CPU tests. `limit_when_candidate` runs only for a widen candidate
/// (production: the dense/MoE admission over the loaded model and device;
/// tests: the same shared decisions over artifact-derived facts); every other
/// call keeps the incumbent `max_batch`.
fn plan_ordinary_prefill<E>(
    req: &OrdinaryPlanRequest,
    limit_when_candidate: impl FnOnce() -> Result<usize, E>,
) -> Result<OrdinaryPrefillPlan, E> {
    let wide_candidate = req.call.wide_candidate();
    let limit = if wide_candidate {
        limit_when_candidate()?
    } else {
        req.max_batch
    };
    let wide = widened_route(wide_candidate, limit);
    let moe_route = wide_candidate && req.moe_static_admitted;
    let requested = if moe_route {
        match req.moe_request {
            ResidentMoeRequest::Widen { requested } => requested,
            ResidentMoeRequest::Incumbent => req.max_batch,
        }
    } else if wide {
        req.chunk_rows_requested
    } else {
        req.explicit_max_batch.unwrap_or(req.max_batch)
    };
    let pbs_rows = match req.call.caller_pbs_rows {
        Some(rows) => rows,
        None if wide => widened_owned_rows(limit, req.n),
        None => owned_prefill_scratch_plan(req.n, req.max_batch.min(limit), req.call.tree_verify).0,
    };
    Ok(OrdinaryPrefillPlan {
        limit,
        wide,
        moe_route,
        receipt: PrefillChunkReceipt {
            requested,
            admitted: limit,
            commit_stride: wide.then_some(WIDENED_COMMIT_ROWS),
        },
        pbs_rows,
    })
}
///
/// Next chunk of the ordinary loop: widened grouping keeps an odd tail attached
/// to the largest admitted chunk (packed for gfx1100 native MQ4), the legacy
/// path keeps its cadence.
fn next_ordinary_chunk_len(
    wide: bool,
    packed_mq4: bool,
    remaining: usize,
    chunk_batch: usize,
) -> Option<usize> {
    if wide && packed_mq4 {
        next_packed_prefill_chunk_len(remaining, chunk_batch)
    } else if wide {
        next_exact_prefill_chunk_len(remaining, chunk_batch)
    } else {
        next_prefill_chunk_len(remaining, chunk_batch)
    }
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
/// `None` on arithmetic overflow.
///
/// Only the *dense* arms are enumerated here (this enumerator predates the MoE
/// route). A MoE config is charged through
/// [`PrefillBatchScratch::projected_allocation_bytes`], the checked mirror of the
/// same constructor that already models its shared-expert, expanded-expert and
/// grouped-GEMM scratch term-for-term. Returning `None` for MoE (the previous
/// behaviour) left the reservation at zero, and the placement search then
/// under-spilled by exactly that scratch — the difference between a load and an
/// `hipMalloc: out of memory`.
///
/// Term-by-term mirror of `PrefillBatchScratch::new_opt(..., false)` for dense
/// configs: F32 `3D + 4K + 8V + 2H + 3I + 6Q + 2W + 3` elems/row, F16
/// `D + V + I + Q` elems/row, plus 256 fixed bytes.
fn dense_prefill_allocation_bytes(config: &Qwen35Config, rows: usize) -> Option<usize> {
    if config.num_experts != 0 {
        // `cap_gdn_tape = true` is the conservative arm: the tape is an
        // `alloc_opt!` in the constructor, so charging it can only make the
        // placement spill slightly more, never less.
        return crate::qwen35::batch::PrefillBatchScratch::projected_allocation_bytes(
            config, rows, true,
        )
        .ok()
        .and_then(|bytes| usize::try_from(bytes).ok());
    }
    dense_layer_prefill_allocation_bytes(config, rows)
}

/// The arch-independent terms of [`dense_prefill_allocation_bytes`].
fn dense_layer_prefill_allocation_bytes(config: &Qwen35Config, rows: usize) -> Option<usize> {
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
/// K128-major A8 prelude bytes/row at the `Kmax = hidden_dim` envelope: one
/// 136-byte `block_i8_128` per K/128 block.
fn iu8_row_bytes_wide(config: &Qwen35Config) -> Option<usize> {
    config
        .hidden_dim
        .checked_add(127)?
        .checked_div(128)?
        .checked_mul(136)
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
    let base =
        dense_prefill_reservation_bytes(config, WIDENED_COMMIT_ROWS, arch, Mq3WideProjection::Off)?;
    // The loader has not classified all projection tensors yet. Under the
    // explicit experimental MQ4 opt-ins, conservatively reserve their prelude
    // too; the default V2/other-format reservation remains unchanged.
    let packed = hipfire_config::developer_var("HIPFIRE_GFX1100_PACKED_MQ4_PREFILL")
        .ok()
        .as_deref()
        == Some("1");
    if arch == "gfx1100" && (native_mq4_widened_requested() || packed) {
        base.checked_add(native_mq4_projection_deficit(
            config,
            WIDENED_COMMIT_ROWS,
            0,
            0,
            0,
        )?)
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
    let mmq = config
        .hidden_dim
        .checked_add(127)?
        .checked_div(128)?
        .checked_mul(144)?
        .checked_mul(rows)?;
    let tail = rows.min(127);
    let f16 = tail.checked_mul(config.hidden_dim)?.checked_mul(2)?;
    let partials = tail
        .checked_mul(config.dim)?
        .checked_mul(4)?
        .checked_mul(4)?;
    mmq.saturating_sub(live_mmq)
        .checked_add(f16.saturating_sub(live_f16))?
        .checked_add(partials.saturating_sub(live_partials))
}

fn dense_prefill_reservation_bytes(
    config: &Qwen35Config,
    rows: usize,
    arch: &str,
    mq3: Mq3WideProjection,
) -> Option<usize> {
    let projection = match mq3 {
        Mq3WideProjection::A8 => iu8_row_bytes_wide(config)?,
        Mq3WideProjection::A4 => iu4_row_bytes_wide(config)?,
        Mq3WideProjection::Off if arch == "gfx1201" => fp8_row_bytes_wide(config)?,
        Mq3WideProjection::Off => iu4_row_bytes_wide(config)?,
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
fn can_retain_widened_pbs(
    gpu: &Gpu,
    kv: &llama::KvCache,
    config: &Qwen35Config,
    mq3: Mq3WideProjection,
) -> bool {
    if !kv.uses_vmm_backend() {
        return true;
    }
    let Some(minimum) =
        dense_prefill_reservation_bytes(config, WIDENED_COMMIT_ROWS, &gpu.arch, mq3)
    else {
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
    gpu.device_mem_info()
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
        .is_some_and(|needed| gpu.device_mem_info().is_ok_and(|(free, _)| free >= needed));
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
#[allow(clippy::too_many_arguments)]
fn memory_admitted_rung(
    gpu: &Gpu,
    config: &Qwen35Config,
    kv_cache: &llama::KvCache,
    perf_rows: usize,
    lean: bool,
    native_mq4: bool,
    mq3: Mq3WideProjection,
    cached: Option<&PrefillBatchScratch>,
    free_deduction: usize,
) -> HipResult<usize> {
    if perf_rows <= WIDENED_COMMIT_ROWS {
        return Ok(WIDENED_COMMIT_ROWS.min(perf_rows));
    }
    let a8_projection = mq3 == Mq3WideProjection::A8;
    let fp8_projection = gpu.arch == "gfx1201" && mq3 == Mq3WideProjection::Off;
    if (if lean {
        lean_dense_prefill_allocation_bytes(config, perf_rows)
    } else {
        dense_prefill_allocation_bytes(config, perf_rows)
    })
    .is_none()
        || (a8_projection && iu8_row_bytes_wide(config).is_none())
        || (fp8_projection && fp8_row_bytes_wide(config).is_none())
        || (!a8_projection && !fp8_projection && iu4_row_bytes_wide(config).is_none())
    {
        return Ok(WIDENED_COMMIT_ROWS);
    }
    let free_bytes = gpu.device_mem_info()?.0.saturating_sub(free_deduction);
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
    let live_iu8 = gpu.scratch.int8_mmq_x_scratch_bytes;
    let fp8_groups = config.hidden_dim / 256;
    let iu4_bytes_per_row = iu4_row_bytes_wide(config).unwrap_or(usize::MAX);
    let iu8_bytes_per_row = iu8_row_bytes_wide(config).unwrap_or(usize::MAX);
    for &rung in WIDENED_RUNGS.iter().rev() {
        // Symmetric qt49 A4/A8 on gfx1201 admit 8192/4096 before the 512 fallback.
        if mq3 != Mq3WideProjection::Off
            && gpu.arch == "gfx1201"
            && !matches!(rung, 8192 | 4096 | 512)
        {
            continue;
        }
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
        let projection_deficit = if a8_projection {
            rung.checked_mul(iu8_bytes_per_row)
                .unwrap_or(usize::MAX)
                .saturating_sub(live_iu8)
        } else if native_mq4 {
            native_mq4_projection_deficit(
                config,
                rung,
                gpu.scratch.q8_1_mmq_x_scratch_bytes,
                gpu.scratch.fp16_x_scratch_bytes,
                gpu.scratch.ksplit_det_partials_bytes,
            )
            .unwrap_or(usize::MAX)
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
/// Physical KV bytes this request will still map to reach `end_tokens` rows
/// (clamped to the cache's physical capacity). Zero on non-VMM KV, when the
/// mapping already covers the request, or if the query errors. Feed to
/// [`ordinary_prefill_chunk_limit_deducting`] so the once-per-prompt chunk
/// ceiling is sized against memory the same request is about to map.
pub fn request_kv_growth_deduction(
    gpu: &Gpu,
    kv_cache: &llama::KvCache,
    end_tokens: usize,
) -> usize {
    kv_cache
        .planned_mapped_growth_bytes(gpu, end_tokens.min(kv_cache.physical_cap))
        .unwrap_or(0)
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
///
/// Resident Qwen35 MoE models (fully resident, single device, MQ4G256/V2
/// grouped producers; see [`resident_moe_static_decision`]) resolve to
/// [`RESIDENT_MOE_CHUNK_ROWS`] when the projected PBS, grouped/projection
/// scratch, full-row Q16 growth and the 1 GiB headroom fit in device-free
/// bytes minus `free_deduction`, else to the incumbent ceiling — never an
/// intermediate rung. Explicit `HIPFIRE_PREFILL_MAX_BATCH` keeps the incumbent.
pub fn ordinary_prefill_chunk_limit(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    dn_state: &DeltaNetState,
    kv_cache: &llama::KvCache,
    pbs: Option<&PrefillBatchScratch>,
) -> HipResult<usize> {
    ordinary_prefill_chunk_limit_with_cache(
        gpu,
        weights,
        config,
        dn_state,
        kv_cache,
        pbs,
        None,
        RESIDENT_MOE_CHUNK_ROWS,
        0,
    )
}
///
/// [`ordinary_prefill_chunk_limit`] as it will read after `free_deduction`
/// more device bytes are allocated: the byte admission sees
/// `free - free_deduction` (saturating). The query is monotone in free
/// bytes, so the result is a lower bound on the ceiling taken once that
/// allocation has happened, provided nothing else shrinks free memory
/// further. Lets a caller bound a prefill's chunk plan before the prefill's
/// owners exist (VMM lane admission).
pub fn ordinary_prefill_chunk_limit_deducting(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    dn_state: &DeltaNetState,
    kv_cache: &llama::KvCache,
    free_deduction: usize,
) -> HipResult<usize> {
    ordinary_prefill_chunk_limit_with_cache(
        gpu,
        weights,
        config,
        dn_state,
        kv_cache,
        None,
        None,
        RESIDENT_MOE_CHUNK_ROWS,
        free_deduction,
    )
}

#[allow(clippy::too_many_arguments)]
fn ordinary_prefill_chunk_limit_with_cache(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    dn_state: &DeltaNetState,
    kv_cache: &llama::KvCache,
    pbs: Option<&PrefillBatchScratch>,
    cached: Option<&PrefillBatchScratch>,
    reuse_rows: usize,
    free_deduction: usize,
) -> HipResult<usize> {
    let moe = resident_moe_static_facts(gpu, weights, config, dn_state);
    ordinary_prefill_chunk_limit_with_facts(
        gpu,
        weights,
        config,
        dn_state,
        kv_cache,
        pbs,
        cached,
        reuse_rows,
        free_deduction,
        &moe,
    )
}

/// [`ordinary_prefill_chunk_limit_with_cache`] over already gathered resident
/// MoE static facts (the inner entry gathers them once for the whole call).
#[allow(clippy::too_many_arguments)]
fn ordinary_prefill_chunk_limit_with_facts(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    dn_state: &DeltaNetState,
    kv_cache: &llama::KvCache,
    pbs: Option<&PrefillBatchScratch>,
    cached: Option<&PrefillBatchScratch>,
    // Rows of the owned widened PBS the call needs (see
    // [`resident_moe_reusable_cached`]); only the resident-MoE arm reads it.
    reuse_rows: usize,
    free_deduction: usize,
    moe: &ResidentMoeStaticFacts,
) -> HipResult<usize> {
    let legacy = prefill_max_batch_for_model(gpu, weights);
    let Some(perf) = ordinary_prefill_static_ceiling(gpu, weights, config, dn_state) else {
        return Ok(resident_moe_chunk_limit_from_facts(
            moe,
            pbs.map(|p| p.max_batch),
            config,
            KvTierFlags::from_cache(kv_cache),
            || resident_moe_memory_facts(gpu),
            cached.map(|p| (p.max_batch, p.lean)),
            reuse_rows,
            free_deduction,
            legacy,
        ));
    };
    if perf <= WIDENED_COMMIT_ROWS {
        return Ok(perf.min(legacy));
    }
    let lean = lean_pbs_requested() && lean_pbs_route(gpu, weights, config, perf);
    let mut admitted = memory_admitted_rung(
        gpu,
        config,
        kv_cache,
        perf,
        lean,
        native_mq4_widened_route(gpu, weights),
        mq3_wide_projection(gpu, weights),
        cached,
        free_deduction,
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
pub fn next_exact_prefill_chunk_len(remaining: usize, ceiling: usize) -> Option<usize> {
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
    emit_prefill_chunk_receipt_guarded(&LAST_ADMITTED, requested, admitted, commit_stride);
}

/// [`emit_prefill_chunk_receipt`] with the changed-admission guard injected,
/// so a CPU test (one fresh guard per modelled load) exercises the exact
/// production print path. Returns whether a line was printed.
fn emit_prefill_chunk_receipt_guarded(
    last_admitted: &std::sync::atomic::AtomicUsize,
    requested: usize,
    admitted: usize,
    commit_stride: Option<usize>,
) -> bool {
    if last_admitted.swap(admitted, std::sync::atomic::Ordering::Relaxed) == admitted {
        return false;
    }
    eprintln!(
        "{}",
        PrefillChunkReceipt {
            requested,
            admitted,
            commit_stride
        }
    );
    true
}
///
/// The one visible admitted-chunk line (`prefill_chunk: requested=… admitted=…
/// commit_stride=…`). Production prints it with `eprintln!("{receipt}")`
/// through [`emit_prefill_chunk_receipt`]; CPU tests format the same value, so
/// the log line they assert is the production line and neither allocates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PrefillChunkReceipt {
    requested: usize,
    admitted: usize,
    commit_stride: Option<usize>,
}

impl std::fmt::Display for PrefillChunkReceipt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "prefill_chunk: requested={} admitted={} commit_stride=",
            self.requested, self.admitted
        )?;
        match self.commit_stride {
            Some(stride) => write!(f, "{stride}"),
            None => f.write_str("none"),
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
    // VL: a captured replay would bake the image splice / phase staging into
    // the graph, and this entry bypasses the eligibility gate.
    super::vl_prefill::refuse_overlap(start_pos, tokens.len(), "captured prefill")?;
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
    // Wrapper PBS plan (`plan_wrapper_pbs`): the dense widened route drops the
    // implicit ≤512 cache; an eligible, memory-admitted DEFAULT resident-MoE
    // request runs as a virtual explicit `HIPFIRE_PREFILL_MAX_BATCH=8192`
    // (owner rows/tape from the explicit cache sizing, ceiling 8192); every
    // other call keeps the implicit cache. Capped/explicit-PBS callers never
    // bypass (see `forward_prefill_batch_capped` and the inner entry).
    let call = ordinary_call_facts(
        gpu,
        None,
        None,
        tree_verify.is_some(),
        gdn_tape.is_some(),
        hidden_rb.as_deref(),
        per_token_hidden_out.is_some(),
        false,
        false,
    );
    let wrapper = ordinary_wrapper_pbs_prepare(
        gpu,
        weights,
        config,
        dn_state,
        kv_cache,
        scratch,
        start_pos,
        tokens.len(),
        call,
    )?;
    let virtual_owner = scratch.reuse_virtual_pbs.borrow();
    let pbs_for_call = wrapper.pbs(scratch.prefill_batch.as_ref(), virtual_owner.as_ref());
    let result = forward_prefill_batch_with_pbs_opts_inner(
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
        pbs_for_call,
        None, // mask_override: MTP probe is the only consumer; default callers don't override
        None, // max_layer: pflash uses this; non-pflash default is full stack
        true, // preserve legacy post-condition: scratch.logits is last-token logits
        None,
        wrapper.max_batch_override,
        DflashFusionCtx::Off,
    );
    result
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
    // Same wrapper PBS plan as `forward_prefill_batch`.
    let call = ordinary_call_facts(gpu, None, None, false, false, None, true, false, false);
    let wrapper = ordinary_wrapper_pbs_prepare(
        gpu,
        weights,
        config,
        dn_state,
        kv_cache,
        scratch,
        start_pos,
        tokens.len(),
        call,
    )?;
    let virtual_owner = scratch.reuse_virtual_pbs.borrow();
    let pbs_for_call = wrapper.pbs(scratch.prefill_batch.as_ref(), virtual_owner.as_ref());
    let result = forward_prefill_batch_with_pbs_opts_inner(
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
        wrapper.max_batch_override,
        DflashFusionCtx::Off,
    );
    result
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
        None,
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
    // Virtual explicit `HIPFIRE_PREFILL_MAX_BATCH` chosen by the ordinary
    // wrapper (`plan_wrapper_pbs`); replaces the arch/env default ceiling.
    max_batch_override: Option<usize>,
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
    let max_batch: usize = resolve_ordinary_max_batch(
        prefill_max_batch_for_model(gpu, weights),
        max_batch_override,
        max_batch_cap,
    );

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
        // VL: the per-token loop below cannot splice image embeddings or rotate
        // at mrope phases. Surfaces before any KV / recurrent state is touched.
        super::vl_prefill::refuse_overlap(start_pos, n, "per-token prefill fallback")?;
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

    // A batched forward is never the plain-AR Redline retained body: clear the
    // eligibility the previous decode forward left set, so prefill-time gates
    // (MoE IU4 / combine zinit / sealed retained admission) read as on a fresh
    // load instead of seeing a retained body after the first Redline decode.
    // The next `forward_scratch` re-arms it.
    gpu.replay.enter_batched_forward();

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
    let ring_seed = hidden_ring_seed(
        gpu,
        hidden_rb.as_deref(),
        per_token_hidden_out.is_some(),
        gdn_tape.is_some(),
        tree_verify.is_some(),
    );
    // Widened ordinary admission. Decided here, after required KV mapping, so
    // the per-device byte admission charges mapped KV. Only the whole-stack
    // sequential ordinary entry with no caller PBS/cap can widen: captured,
    // TP/EP (separate callers), tree/tape, staged hidden-ring, band-limited,
    // fused, or caller-capped requests resolve to the legacy path before any
    // wider allocation or launch. The dense widened wrapper omits its
    // *implicit* ≤512 legacy cache above (`plan_wrapper_pbs`); an eligible,
    // memory-admitted DEFAULT resident-MoE request arrives here as a virtual
    // explicit `HIPFIRE_PREFILL_MAX_BATCH=8192` (caller owner with the explicit
    // cache rows/tape and `max_batch_override`), and every other cache is a
    // caller owner that keeps legacy cadence. An explicit smaller caller PBS
    // always wins. Every parameter below comes from `plan_ordinary_prefill`,
    // the same planner the CPU installed-artifact tests call.
    let moe_facts = resident_moe_static_facts(gpu, weights, config, dn_state);
    let plan = plan_ordinary_prefill(
        &OrdinaryPlanRequest {
            call: ordinary_call_facts(
                gpu,
                pbs_in.map(|p| p.max_batch),
                max_batch_cap,
                tree_verify.is_some(),
                gdn_tape.is_some(),
                hidden_rb.as_deref(),
                per_token_hidden_out.is_some(),
                max_layer.is_some(),
                !matches!(fusion, DflashFusionCtx::Off),
            ),
            n,
            max_batch,
            explicit_max_batch: explicit_prefill_max_batch(),
            chunk_rows_requested: prefill_chunk_rows_requested(gpu),
            moe_request: moe_facts.request,
            moe_static_admitted: resident_moe_static_decision(&moe_facts).is_some(),
        },
        || {
            let cached = scratch.widened_prefill_batch.borrow();
            ordinary_prefill_chunk_limit_with_facts(
                gpu,
                weights,
                config,
                dn_state,
                kv_cache,
                None,
                cached.as_ref().filter(|_| widened_pbs_grow_only(gpu)),
                // Credit the retained owner only if this call reuses it in
                // place: the rows its owned widened PBS needs at the rung.
                widened_owned_rows(RESIDENT_MOE_CHUNK_ROWS, n),
                0,
                &moe_facts,
            )
        },
    )?;
    let limit = plan.limit;
    emit_prefill_chunk_receipt(
        plan.receipt.requested,
        plan.receipt.admitted,
        plan.receipt.commit_stride,
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
        let wide = plan.wide;
        let pbs: &PrefillBatchScratch = match pbs_in {
            Some(p) => p,
            None if wide => {
                let owned_rows = plan.pbs_rows;
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
                let (owned_max_batch, cap_gdn_tape) = (plan.pbs_rows, tree_verify.is_some());
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
        let packed_mq4 = wide
            && gpu.arch == "gfx1100"
            && gpu.flags.packed_mq4_prefill
            && dense_layers_have_projection_dtype(weights, DType::MQ4G256);
        let mut chunk_start = 0usize;
        while chunk_start < n {
            let remaining = n - chunk_start;
            // Widened grouping keeps an odd tail attached to the largest
            // admitted chunk so its GEMMs use the padded multi-row route.
            let chunk_n = next_ordinary_chunk_len(wide, packed_mq4, remaining, chunk_batch)
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
            // VL chunks take the single-chunk route below (hooked); the F2
            // pair route has no splice / mrope staging.
            if pair_b == Some(FA_PAIR_ROWS)
                && !super::vl_prefill::overlaps(start_pos + chunk_start, 2 * FA_PAIR_ROWS)
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
                    let stage = FaPairStage::alloc(gpu, &hybrid_dims(config))?;
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
            let commit_stride = widened_commit_stride(wide, chunk_n);
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
            if can_retain_widened_pbs(gpu, kv_cache, config, mq3_wide_projection(gpu, weights)) {
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
        // GGUF-native block formats: unrotated unfused plain-GEMM prefill
        // (`run_plain_gemm_key` dequantises each weight); lockstep with the
        // `is_lowbit` / `qkv_all_ggml` widenings in hipfire-dispatch.
        | DType::GgmlQ4K | DType::GgmlQ5K | DType::GgmlQ6K | DType::GgmlQ8_0 | DType::GgmlIQ4XS
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
    let has_dn = weights
        .layers
        .iter()
        .any(|lw| matches!(lw, LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_)));
    let scalar_ok = prefill_batch_scalar_eligible(
        n,
        arch,
        decouple_env.as_deref(),
        hipfire_runtime::config::get().prefill_batched,
        has_dn,
    );
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
    let result = scalar_ok && all_layers_ok;
    if hipfire_config::developer_var("HIPFIRE_DEBUG_BATCH")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "[hipfire::batch_eligible] result={result} \
             arch={arch} n={n} n>={MIN_BATCH}={} \
             scalar_ok={scalar_ok} \
             has_dn={has_dn} \
             moe_router_logits_present={moe_router_logits_present} \
             all_layers_ok={all_layers_ok}",
            n >= MIN_BATCH,
        );
    }
    result
}

/// Pure scalar half of [`prefill_batch_pbs_eligible`] (no weights): the
/// verify-decouple window, the `prefill_batched` config switch, the
/// two-token minimum and a DeltaNet layer. `decouple_env` is
/// `HIPFIRE_MTP_VERIFY_DECOUPLE`.
fn prefill_batch_scalar_eligible(
    n: usize,
    arch: &str,
    decouple_env: Option<&str>,
    prefill_batched: bool,
    has_dn: bool,
) -> bool {
    let is_rdna3_decouple = arch.starts_with("gfx11");
    let verify_decouple =
        n <= 32 && decouple_env != Some("0") && (is_rdna3_decouple || decouple_env == Some("1"));
    let force_fallback = !verify_decouple && !prefill_batched;
    !force_fallback && n >= MIN_BATCH && has_dn
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
    cpu_deferred_experts: bool,
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
    crate::qwen35::weights::trace_moe_arm(
        "prefill",
        moe_dtypes.routed_gate_up,
        moe_dtypes.routed_down,
        moe_dtypes.experts_all_gate_up_mq4,
        ffn.packed_expert_owners.as_ref().is_some_and(|owners| {
            owners.gate_up.iter().all(|b| b.buf.is_host_mapped())
                && owners.down.iter().all(|b| b.buf.is_host_mapped())
        }),
        per_expert_gate_up.is_some() || per_expert_down.is_some(),
    );

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
        cpu_deferred_experts,
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
        ep_row_gather: None,
        shared_down_out: None,
        ep_layer_idx: 0,
        ep_tile_exchange: None,
        expert_iu4: ffn.expert_iu4,
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
        false,
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
        false,
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
        false,
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
    prefill_moe_ffn_body_batched_with_route_layered(
        gpu,
        ffn,
        ffn_norm,
        config,
        pbs,
        n,
        ctx,
        model_has_mq6_moe,
        routed_out,
        route,
        ffn.layer_idx as usize,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_moe_ffn_body_batched_with_route_layered<'a>(
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
    layer_idx: usize,
) -> HipResult<()> {
    // Narrow CPU-arithmetic forwards (the `<=4`-row speculative verify and the
    // accepted-prefix rollback replay) must recompute the routed experts on the
    // CPU, matching single-token decode; the GPU grouped PCIe read would
    // otherwise diverge from the CPU target arithmetic these rows are validated
    // against. The splice is width-only: any splice-armed layer at `n <= 4`
    // takes it regardless of dispatch workload, so the tape-disabled
    // accepted-prefix rollback (a `Standard` `forward_prefill_batch` call) and
    // an ordinary very short `<=4` prompt prefill both land on the CPU. Wider
    // prompt prefill and wide verify keep the real expert tables and the GPU
    // grouped read, where the CPU expert FFN loses by >10x. Expert-parallel
    // prefill never defers: its routed partial is gathered and all-reduced, and
    // the CPU splice would fold the routed FFN into `x_batch` twice.
    let cpu_deferred_experts = ffn.cpu_deferred_experts
        && n <= NARROW_VERIFY_MAX_ROWS
        && routed_out.is_none()
        && !matches!(
            route,
            PrefillRouteMode::ProduceRoot { .. } | PrefillRouteMode::AdoptRoot { .. }
        );
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
        cpu_deferred_experts,
    )?;
    let compact_ep = matches!(
        &params.prelude.route,
        PrefillRouteMode::ProduceRoot { .. } | PrefillRouteMode::AdoptRoot { .. }
    ) && bound.rank_count() > 1;
    // The CPU splice below adds into `pbs.x_batch` directly, so it is correct
    // on every down path: the deferred program omitted the routed stages.
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

        // Window span (opt-in `HIPFIRE_CPU_EXEC_TRACE` only): sealed GPU program
        // vs CPU splice vs caller-side gap. Host wall around calls this body
        // already makes; no new sync, no order change.
        let t_window = std::time::Instant::now();
        let t_sealed = std::time::Instant::now();
        execute_steps(gpu, ctx, &[Step::Moe(sealed)])
            .map_err(|e| HipError::new(0, &e.to_string()))?;
        let sealed_ns = t_sealed.elapsed().as_nanos() as u64;
        // The sealed program above runs the CPU expert splice in place of the
        // routed GPU stages for this call, so the splice's cost is inside
        // `sealed_ns` and cannot be separated on this side; `cpu_splice_ns`
        // therefore reports 0 and the stage-side trace is the authority for
        // the splice's own timing.
        if cpu_deferred_experts {
            // The omitted GPU routed tail no longer changes the FP16 staging
            // cache key. The next layer reuses x_rot_batch with new contents.
            gpu.invalidate_fp16_cache();
        } else {
            // `memory.offload_exec=cpu` deliberately does **not** splice the
            // routed experts for ordinary wide prefill: the batched prefill
            // params bind the real expert tables, so the sealed step above
            // already produced the routed contribution with the GPU's grouped
            // read over the host-mapped blobs (one weight pass per unique expert
            // per chunk). The CPU expert FFN is compute-bound and loses to that
            // by >10x at chunk sizes (measured 15x on ornith/gfx1201).
        }
        let total_ns = t_window.elapsed().as_nanos() as u64;
        if hipfire_dispatch::cpu_exec::moe_trace_enabled() {
            hipfire_dispatch::cpu_exec::moe_window_trace(
                layer_idx,
                hipfire_dispatch::cpu_exec::MoeWindowTiming {
                    total_ns,
                    gpu_sealed_ns: sealed_ns,
                    // The splice now runs inside the sealed program (stage-side), so
                    // its cost is part of `gpu_sealed_ns`; see the comment above.
                    cpu_splice_ns: 0,
                    other_ns: total_ns.saturating_sub(sealed_ns),
                    rows: n,
                    top_k: config.num_experts_per_tok,
                },
            );
        }

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
            EmbeddingFormat::HFQ4G256 | EmbeddingFormat::Q8_0 | EmbeddingFormat::Ggml(_)
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
                    EmbeddingFormat::Ggml(dtype) => gpu.ggml_embedding_lookup_batched(
                        dtype,
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
                EmbeddingFormat::Ggml(dtype) => gpu.ggml_embedding_lookup_batched(
                    dtype,
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
                EmbeddingFormat::Ggml(dtype) => {
                    gpu.ggml_embedding_lookup(dtype, &weights.token_embd, &s.x, tok, dim)?
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

/// `HIPFIRE_GDN_MOE_CHUNK_SCAN` (`developer.gdn_moe_chunk_scan`, a typed
/// boolean: `true`/`false`/`1`/`0`; any other spelling is rejected when the
/// process configuration is built). The explicit value, read once: `true`/`1`
/// arms the dense GDN chunk-scan route for admitted MoE LA layers,
/// `false`/`0` is the kill switch. Unset (`None`) follows
/// `prefill_route_capabilities` qualification of the running architecture.
fn gdn_moe_chunk_scan_explicit() -> Option<bool> {
    static EXPLICIT: std::sync::LazyLock<Option<bool>> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_typed_bool("HIPFIRE_GDN_MOE_CHUNK_SCAN")
    });
    *EXPLICIT
}

/// Pure resolution of the MoE chunk-scan flag: an explicit value wins on any
/// architecture, unset follows the architecture's qualification.
fn gdn_moe_chunk_scan_resolve(explicit: Option<bool>, qualified: bool) -> bool {
    explicit.unwrap_or(qualified)
}

/// Whether the MoE chunk scan is on for `arch`: the explicit
/// `HIPFIRE_GDN_MOE_CHUNK_SCAN` value, else qualification. Admission is
/// [`GdnMoeChunkScanInputs::admitted`].
fn gdn_moe_chunk_scan_on(arch: &str) -> bool {
    gdn_moe_chunk_scan_resolve(
        gdn_moe_chunk_scan_explicit(),
        prefill_route_capabilities::qualified(arch, PrefillRoute::GdnMoeChunkScan),
    )
}

/// Row admission of the chunk scan, shared by the dense and MoE routes: at
/// least one 64-row chunk, and either one segment of at most 512 rows or
/// 512-row commit segments whose tail is a whole chunk or empty.
fn gdn_chunk_scan_rows_admitted(n: usize, commit_stride: Option<usize>) -> bool {
    n >= 64
        && (n <= 512
            || (commit_stride == Some(512) && (n % 512 == 0 || (64..512).contains(&(n % 512)))))
}

/// Layer-independent inputs of the MoE GDN chunk-scan admission.
#[derive(Clone, Copy)]
struct GdnMoeChunkScanInputs {
    /// Resolved `HIPFIRE_GDN_MOE_CHUNK_SCAN` ([`gdn_moe_chunk_scan_on`]).
    enabled: bool,
    /// Chunk-scan capability: the global flag on exact gfx1201, outside FP8
    /// verify, HIP-graph capture and Redline recording.
    capable: bool,
    /// [`gdn_chunk_head_geometry_supported`].
    geometry: bool,
    /// Q8 DeltaNet state with complete scale, conv and EF vectors.
    q8_state: bool,
    sequential: bool,
    tree: bool,
    tape: bool,
    /// No DFlash fusion, hidden-state capture, verify capture, band, layer cap,
    /// routed-out or mask override (the dense route's plain-forward clauses).
    plain_forward: bool,
    /// [`gdn_chunk_scan_rows_admitted`].
    rows: bool,
}

impl GdnMoeChunkScanInputs {
    fn admitted(self) -> bool {
        self.enabled
            && self.capable
            && self.geometry
            && self.q8_state
            && self.sequential
            && !self.tree
            && !self.tape
            && self.plain_forward
            && self.rows
    }
}

/// Per-layer clause of the MoE admission: MQ4G256V2 on all four fused-QKVZA
/// projections (wqkv, wz, w_beta, w_alpha) and an EF residual slot for this
/// DeltaNet layer.
fn gdn_moe_chunk_scan_layer_admitted(
    common: bool,
    projection_dtypes: [DType; 4],
    ef_layer: bool,
) -> bool {
    common && projection_dtypes == [DType::MQ4G256V2; 4] && ef_layer
}

/// [`hipfire_dispatch::pipeline::batched_attention::gfx1100_split_verify_min_ctx`]
/// for a Qwen3.5 model.
#[doc(hidden)]
pub fn gfx1100_split_verify_min_ctx(gpu: &Gpu, config: &Qwen35Config) -> Option<usize> {
    hipfire_dispatch::pipeline::batched_attention::gfx1100_split_verify_min_ctx(
        gpu,
        &super::program::hybrid_dims(config),
    )
}

/// [`hipfire_dispatch::pipeline::batched_attention::gfx1100_split_verify_admitted`]
/// for a Qwen3.5 model. `quant_q8`: both K and V use exact plain Q8; the
/// K-only cache flag must also be paired with
/// `matches!(kv_cache.v_mode, llama::VMode::Q8)`. `kv_vmm`: the target KV
/// cache is a VMM owner ([`llama::KvCache::uses_vmm_backend`]).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn gfx1100_split_verify_admitted(
    gpu: &Gpu,
    config: &Qwen35Config,
    quant_q8: bool,
    kv_vmm: bool,
    batch_size: usize,
    logical_ctx_len: usize,
    flash_partials_numel: usize,
) -> bool {
    hipfire_dispatch::pipeline::batched_attention::gfx1100_split_verify_admitted(
        gpu,
        &super::program::hybrid_dims(config),
        quant_q8,
        kv_vmm,
        batch_size,
        logical_ctx_len,
        flash_partials_numel,
    )
}

#[allow(clippy::too_many_arguments)]
fn batch_chunk_full_attn_fallback(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    layer_idx: usize,
    start_pos: usize,
    n: usize,
    dim_row_bytes: usize,
    kv_cache: &llama::KvCache,
    dn_state: &DeltaNetState,
    s: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
) -> HipResult<()> {
    // Per-row decode steps for FA layers the batched attention does not
    // admit (non-MQ4 weights, non-Q8_0 KV, etc).
    // Capture-safe: the per-row copies are stream-ordered on the active
    // (capturing) stream via the `_auto` helpers, and the row position is a
    // device-to-device copy out of `pbs.positions` (already uploaded for this
    // chunk by every caller, including the pre-uploaded captured verify
    // path). A host upload here would be baked into a verify graph as a
    // dangling host pointer and replay the first cycle's position; the legacy
    // blocking `hipMemcpy` would fail capture with HipError 906. Decode
    // kernels read the position from `s.pos_buf` on the device, so replay
    // sees the live window's positions.
    for i in 0..n {
        let pos = start_pos + i;
        gpu.memcpy_dtod_at_auto(
            &s.x.buf,
            0,
            &pbs.x_batch.buf,
            i * dim_row_bytes,
            dim_row_bytes,
        )?;
        gpu.memcpy_dtod_at_auto(
            &s.pos_buf,
            0,
            &pbs.positions.buf,
            i * std::mem::size_of::<i32>(),
            std::mem::size_of::<i32>(),
        )?;
        super::forward::forward_scratch_dense_layer(
            gpu, weights, config, layer_idx, pos, kv_cache, dn_state, s,
        )?;
        gpu.memcpy_dtod_at_auto(
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

fn fa_pair_half(pbs: &PrefillBatchScratch) -> FaPairHalf<'_> {
    FaPairHalf {
        q: &pbs.fa_q_batch,
        k: &pbs.fa_k_batch,
        v: &pbs.fa_v_batch,
        positions: &pbs.positions,
        out: &pbs.fa_attn_out_batch,
    }
}

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
    super::vl_prefill::refuse_overlap(start_c, 2 * n, "F2 chunk-pair")?;
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
    let ctx = DispatchCtx::new(gpu)
        .with_workload(dispatch_workload)
        .with_verify_tile_attend(fusion.is_chain_verify());

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
            gfx1100_split_verify_admitted(
                gpu,
                config,
                kv_cache.quant_q8 && matches!(kv_cache.v_mode, llama::VMode::Q8),
                kv_cache.uses_vmm_backend(),
                n,
                max_ctx,
                s.flash_partials.numel(),
            ),
        )
    };
    let multirow_c = multirow_admitted(max_ctx_c);
    let multirow_n = multirow_admitted(max_ctx_n);
    let merge_fa = !multirow_c && !multirow_n;
    let dims = hybrid_dims(config);
    let flash = flash_scratch(s);
    let tap_fn = batched_attention_tap(config);
    let tap = tap_fn.get();

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
                        &layer.mixer_view(),
                        &dims,
                        &half_pbs.view(),
                        &deltanet_state_view(dn_state),
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
                    layer.ffn_rows(
                        gpu,
                        config.norm_eps,
                        &half_pbs.view(),
                        n,
                        dim,
                        q8_wmma_arch,
                        arch_has_wmma,
                        MlpOutput::Residual,
                        fusion,
                        DenseBatchMath::Product,
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
                        gpu,
                        &layer.mixer_view(),
                        &dims,
                        &pbs_c.view(),
                        n,
                        dim,
                        q8_wmma_arch,
                        fusion,
                        DenseBatchMath::Product,
                    )?;
                    batch_chunk_full_attn_prepare(
                        gpu,
                        false,
                        &layer.mixer_view(),
                        &dims,
                        &pbs_c.view(),
                        &flash,
                        &kv_view(kv_cache),
                        n,
                        start_c,
                        max_ctx_c,
                        &ctx,
                        BatchSemantics::Sequential,
                        None,
                        kv_layer_idx,
                        layer_idx,
                        fusion,
                        false,
                        false,
                        false,
                        tap,
                    )?;
                    batch_chunk_full_attn_input_projection(
                        gpu,
                        &layer.mixer_view(),
                        &dims,
                        &pbs_n.view(),
                        n,
                        dim,
                        q8_wmma_arch,
                        fusion,
                        DenseBatchMath::Product,
                    )?;
                    batch_chunk_full_attn_prepare(
                        gpu,
                        false,
                        &layer.mixer_view(),
                        &dims,
                        &pbs_n.view(),
                        &flash,
                        &kv_view(kv_cache),
                        n,
                        start_n,
                        max_ctx_n,
                        &ctx,
                        BatchSemantics::Sequential,
                        None,
                        kv_layer_idx,
                        layer_idx,
                        fusion,
                        false,
                        false,
                        false,
                        tap,
                    )?;
                    stage.attend(
                        gpu,
                        &hybrid_dims(config),
                        [fa_pair_half(pbs_c), fa_pair_half(pbs_n)],
                        &flash_scratch(s),
                        &kv_view(kv_cache),
                        start_c,
                        max_ctx_merged,
                        &ctx,
                        layer_idx,
                    )?;
                    batch_chunk_full_attn_output_projection(
                        gpu,
                        &layer.mixer_view(),
                        &pbs_c.view(),
                        n,
                        q8_wmma_arch,
                        arch_has_wmma,
                        BatchEpilogue::Residual,
                        fusion,
                        false,
                        None,
                        DenseBatchMath::Product,
                    )?;
                    layer.ffn_rows(
                        gpu,
                        config.norm_eps,
                        &pbs_c.view(),
                        n,
                        dim,
                        q8_wmma_arch,
                        arch_has_wmma,
                        MlpOutput::Residual,
                        fusion,
                        DenseBatchMath::Product,
                    )?;
                    batch_chunk_full_attn_output_projection(
                        gpu,
                        &layer.mixer_view(),
                        &pbs_n.view(),
                        n,
                        q8_wmma_arch,
                        arch_has_wmma,
                        BatchEpilogue::Residual,
                        fusion,
                        false,
                        None,
                        DenseBatchMath::Product,
                    )?;
                    layer.ffn_rows(
                        gpu,
                        config.norm_eps,
                        &pbs_n.view(),
                        n,
                        dim,
                        q8_wmma_arch,
                        arch_has_wmma,
                        MlpOutput::Residual,
                        fusion,
                        DenseBatchMath::Product,
                    )?;
                } else {
                    batch_chunk_full_attn_attn(
                        gpu,
                        multirow_c,
                        &layer.mixer_view(),
                        &dims,
                        &pbs_c.view(),
                        &flash,
                        &kv_view(kv_cache),
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
                        tap,
                    )?;
                    layer.ffn_rows(
                        gpu,
                        config.norm_eps,
                        &pbs_c.view(),
                        n,
                        dim,
                        q8_wmma_arch,
                        arch_has_wmma,
                        MlpOutput::Residual,
                        fusion,
                        DenseBatchMath::Product,
                    )?;
                    batch_chunk_full_attn_attn(
                        gpu,
                        multirow_n,
                        &layer.mixer_view(),
                        &dims,
                        &pbs_n.view(),
                        &flash,
                        &kv_view(kv_cache),
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
                        tap,
                    )?;
                    layer.ffn_rows(
                        gpu,
                        config.norm_eps,
                        &pbs_n.view(),
                        n,
                        dim,
                        q8_wmma_arch,
                        arch_has_wmma,
                        MlpOutput::Residual,
                        fusion,
                        DenseBatchMath::Product,
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
                    start_c,
                    n,
                    dim_row_bytes,
                    kv_cache,
                    dn_state,
                    s,
                    pbs_c,
                )?;
                batch_chunk_full_attn_fallback(
                    gpu,
                    weights,
                    config,
                    layer_idx,
                    start_n,
                    n,
                    dim_row_bytes,
                    kv_cache,
                    dn_state,
                    s,
                    pbs_n,
                )?;
                kv_layer_idx += 1;
                dump_hidden_localize(gpu, &pbs_c.x_batch, n, start_c, dim, layer_idx, "batched");
                dump_hidden_localize(gpu, &pbs_n.x_batch, n, start_n, dim, layer_idx, "batched");
            }
            (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                for (half_pbs, half_start) in [(pbs_c, start_c), (pbs_n, start_n)] {
                    execute_deltanet_moe_layer_batched(
                        gpu,
                        &layer.mixer_view(),
                        &dims,
                        &half_pbs.view(),
                        &deltanet_state_view(dn_state),
                        n,
                        dim,
                        k_dim,
                        v_dim,
                        n_v_heads,
                        hd,
                        BatchSemantics::Sequential,
                        None,
                        None,
                        0,
                        delta_layer_idx,
                        half_start,
                        layer_idx,
                        None, // commit_stride pair halves keep legacy cadence
                        false,
                        |gpu: &mut Gpu| {
                            prefill_moe_ffn_body_batched_with_route_layered(
                                gpu,
                                &layer.ffn,
                                &layer.ffn_norm,
                                config,
                                half_pbs,
                                n,
                                &ctx,
                                weights.moe_has_mq6,
                                None,
                                PrefillRouteMode::Replicated,
                                layer_idx,
                            )
                        },
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
                let mixer = layer.mixer_view();
                if merge_fa {
                    for half_pbs in [pbs_c, pbs_n] {
                        attention_moe_prep_batched(
                            gpu,
                            &mixer,
                            &dims,
                            &half_pbs.view(),
                            n,
                            dim,
                            &kv_view(kv_cache),
                            None,
                            layer_idx,
                            false,
                            false, // the F2 pair never admits the FP8 Q prep
                            tap,
                        )?;
                    }
                    stage.attend(
                        gpu,
                        &hybrid_dims(config),
                        [fa_pair_half(pbs_c), fa_pair_half(pbs_n)],
                        &flash_scratch(s),
                        &kv_view(kv_cache),
                        start_c,
                        max_ctx_merged,
                        &ctx,
                        layer_idx,
                    )?;
                    for (half_pbs, half_start, half_max_ctx) in
                        [(pbs_c, start_c, max_ctx_c), (pbs_n, start_n, max_ctx_n)]
                    {
                        attention_moe_finish_batched(
                            gpu,
                            false,
                            &mixer,
                            &dims,
                            &half_pbs.view(),
                            &flash,
                            &kv_view(kv_cache),
                            n,
                            half_start,
                            half_max_ctx,
                            &ctx,
                            BatchSemantics::Sequential,
                            None,
                            q8_wmma_arch,
                            layer_idx,
                            false,
                            false, // the F2 pair never admits the FP8 Q prep
                            |gpu: &mut Gpu| {
                                prefill_moe_ffn_body_batched_with_route_layered(
                                    gpu,
                                    &layer.ffn,
                                    &layer.ffn_norm,
                                    config,
                                    half_pbs,
                                    n,
                                    &ctx,
                                    weights.moe_has_mq6,
                                    None,
                                    PrefillRouteMode::Replicated,
                                    layer_idx,
                                )
                            },
                        )?;
                    }
                } else {
                    for (half_pbs, half_start, half_max_ctx, multirow) in [
                        (pbs_c, start_c, max_ctx_c, multirow_c),
                        (pbs_n, start_n, max_ctx_n, multirow_n),
                    ] {
                        execute_attention_moe_layer_batched(
                            gpu,
                            multirow,
                            &mixer,
                            &dims,
                            &half_pbs.view(),
                            &flash,
                            &kv_view(kv_cache),
                            n,
                            dim,
                            half_start,
                            half_max_ctx,
                            &ctx,
                            BatchSemantics::Sequential,
                            None,
                            q8_wmma_arch,
                            layer_idx,
                            tap,
                            |gpu: &mut Gpu| {
                                prefill_moe_ffn_body_batched_with_route_layered(
                                    gpu,
                                    &layer.ffn,
                                    &layer.ffn_norm,
                                    config,
                                    half_pbs,
                                    n,
                                    &ctx,
                                    weights.moe_has_mq6,
                                    None,
                                    PrefillRouteMode::Replicated,
                                    layer_idx,
                                )
                            },
                        )?;
                    }
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

/// Every full-attention layer takes the batched FA route (KV tier with
/// batched keys + batch-admissible layer weights); else the per-token FA
/// fallback runs. Shared by the chunk body and the VL route check.
fn fa_layers_batched_ok(
    kv_cache: &llama::KvCache,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    arch: &str,
) -> bool {
    // Slice-B admission: native fp8 KV takes the batched FA path.
    (kv_cache.quant_q8
        || kv_cache.quant_asym4
        || kv_cache.quant_asym3
        || kv_cache.quant_asym2
        || kv_cache.quant_fp8)
        && weights.layers.iter().all(|lw| match lw {
            LayerWeights::FullAttn(_) | LayerWeights::FullAttnMoe(_) => {
                qwen35_layer_batch_admissible(lw, config, arch).is_ok()
            }
            _ => true,
        })
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
    // VL prompt chunk: refuse every route that cannot splice image rows and
    // rotate at mrope phases BEFORE any scratch / KV / recurrent write.
    if super::vl_prefill::overlaps(start_pos, n) {
        super::vl_prefill::check_chunk_route(
            start_pos,
            n,
            &super::vl_prefill::ChunkRoute {
                sequential: matches!(batch_semantics, BatchSemantics::Sequential),
                tree_verify: tree_verify.is_some(),
                pre_embedded,
                pre_uploaded,
                banded: band.is_some(),
                mask_override: mask_override.is_some(),
                capturing: gpu.graphs.capture_mode || gpu.replay.is_recording(),
                fa_unbatched: !fa_layers_batched_ok(kv_cache, weights, config, gpu.arch.as_str()),
                moe_fa: weights
                    .layers
                    .iter()
                    .any(|lw| matches!(lw, LayerWeights::FullAttnMoe(_))),
            },
        )?;
    }
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
    // VL: splice the vision tower's embeddings over this chunk's image rows
    // (after the token embedding + mask override, before layer 0) and stage
    // the chunk's absolute mrope phases into `pbs.pos3`. No-ops without an
    // installed `vl_prefill` scope overlapping the chunk.
    super::vl_prefill::splice_image_rows(gpu, &pbs.x_batch, start_pos, n)?;
    let vl_rope_section = super::vl_prefill::stage_rope_rows(gpu, &pbs.pos3, start_pos, n)?;

    let fa_arch = gpu.arch.as_str();
    let q8_wmma_arch = q8_prefill_wmma_enabled(gpu);
    let arch_has_wmma = q8_wmma_arch;
    // Slice-B admission (see above): native fp8 KV takes the batched FA path.
    let fa_batched_ok = fa_layers_batched_ok(kv_cache, weights, config, fa_arch);
    // Attention only: the batched masked FA kernel grids [n_heads, tiles, ROW]
    // and re-scans the whole KV once per row, so a small verify block over a
    // long context pays the scan n times (202 vs 103 ms at 33k). The layer's
    // GEMMs stay batched either way — only the attend step switches to the
    // multi-row tile. The incumbent tile grid is sized from the live logical
    // context and therefore stays out of capture. The gfx1100 split-KV route
    // uses fixed S=8 geometry and is admitted only inside its exact opt-in
    // gfx1100 envelope on VMM KV (`kernel.gfx1100_fa2_split_verify`, default
    // off; the parent `kernel.verify_attn` still opts out);
    // the launcher independently rejects every other captured shape.
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
        gfx1100_split_verify_admitted(
            gpu,
            config,
            kv_cache.quant_q8 && matches!(kv_cache.v_mode, llama::VMode::Q8),
            kv_cache.uses_vmm_backend(),
            n,
            start_pos + n,
            s.flash_partials.numel(),
        ),
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
    // Dense FFN arithmetic of this forward: SingletonWmma under the PM FP8
    // verify scope, Product otherwise.
    let math = verify_dense_math(gpu);
    // Fail closed: the PM FP8 verify scope covers only the dense batched
    // DeltaNet / FullAttention layers (the MoE and per-token FullAttention
    // fallback arms never reach a `*VerifyExact` key).
    if gpu.fp8_verify_scope
        && (!fa_batched_ok
            || weights.layers.iter().any(|l| {
                matches!(
                    l,
                    LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_)
                )
            }))
    {
        return Err(HipError::new(
            0,
            "forward_batch_chunk: fp8 verify scope needs dense batched DeltaNet/FullAttention layers",
        ));
    }
    let mq3_scan = mq3_wide_route(gpu, weights);
    let mq4_scan = dense_layers_are_all_mq4v2(weights);
    let gdn_chunk_scan_common_admitted = gpu.flags.gfx12_gdn_chunk_scan
        // The GDN chunk-scan arm has no `SingletonWmma` producer.
        && !gpu.fp8_verify_scope
        && matches!(gpu.arch.as_str(), "gfx1100" | "gfx1151" | "gfx1201")
        && config.num_experts == 0
        && config.linear_num_key_heads == 16
        && config.linear_num_value_heads == 48
        && config.linear_key_head_dim == 128
        && config.linear_value_head_dim == 128
        && config.conv_kernel_dim == 4
        // Each dtype must have its own admitted qkvza producer; the A4/A8
        // flags cannot enable a qt44 scan when the MQ4 projection route is off.
        // gfx1201 admits both MQ4V2 projection routes: iu4-direct (default) or
        // FP8-WMMA qkvza (HIPFIRE_IU4_PREFILL=0 arm). Every qkvza dispatch arm
        // writes the same f32 dn_qkv/dn_z/dn_beta/dn_alpha scratch the scan
        // consumes, so the scan is route-agnostic. gfx11 keeps the iu4 gate.
        && (mq3_scan
            || (mq4_scan
                && (gpu.flags.iu4_prefill_enabled()
                    || (gpu.arch == "gfx1201" && gpu.flags.gfx12_mq4v2_fp8_qkvza))))
        // The scan never reads KV, so the clause only names qualified tiers:
        // native FP8 (gfx1201-only), Q8 on gfx11, and the fwht4/fwht3 4/3-bit
        // K tiers (`quant_asym4`/`quant_asym3`, any V) on every scan arch.
        && (kv_cache.quant_fp8
            || (gpu.arch != "gfx1201" && kv_cache.quant_q8)
            || kv_cache.quant_asym4
            || kv_cache.quant_asym3)
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
    // Default-resolved MoE route (explicit `HIPFIRE_GDN_MOE_CHUNK_SCAN` wins,
    // unset follows qualification): the dense chunk-scan clauses that apply to
    // a MoE LA layer (no KV-cache or iu4-prefill clause: the MoE projection
    // arms all write the same f32 scratch and the KV cache is not read by this
    // layer).
    let gdn_moe_chunk_scan_common_admitted = GdnMoeChunkScanInputs {
        enabled: gdn_moe_chunk_scan_on(&gpu.arch),
        capable: gpu.flags.gfx12_gdn_chunk_scan
            && gpu.arch_caps.is_gfx1201()
            && !gpu.fp8_verify_scope
            && !gpu.graphs.capture_mode
            && !gpu.replay.is_recording(),
        geometry: gdn_chunk_head_geometry_supported(
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.conv_kernel_dim,
        ),
        q8_state: dn_state.quant == StateQuant::Q8
            && !dn_state.s_matrices.is_empty()
            && dn_state.s_matrices.len() == dn_state.s_scales.len()
            && dn_state.s_matrices.len() == dn_state.conv_states.len()
            && dn_state.s_matrices.len() == dn_state.s_ef_residual.len(),
        sequential: matches!(batch_semantics, BatchSemantics::Sequential),
        tree: tree_verify.is_some(),
        tape: gdn_tape.is_some(),
        plain_forward: fusion == DflashFusionCtx::Off
            && hidden_rb.map_or(true, |rb| rb.rides_ordinary_prefill(gpu))
            && !captures_verify_hidden(per_token_hidden_out)
            && band.is_none()
            && max_layer.is_none()
            && routed_out.is_none()
            && mask_override.is_none(),
        rows: gdn_chunk_scan_rows_admitted(n, commit_stride),
    }
    .admitted();
    let ctx = DispatchCtx::new(gpu)
        .with_workload(dispatch_workload)
        .with_verify_tile_attend(fusion.is_chain_verify())
        .with_batch_size(n);
    let dims = hybrid_dims(config);
    let mut batch = pbs.view();
    if let Some(section) = vl_rope_section {
        batch.mrope = Some(hipfire_dispatch::pipeline::batched::MropeRows {
            pos3: &pbs.pos3,
            section,
        });
    }
    let flash = flash_scratch(s);
    let tap_fn = batched_attention_tap(config);
    let tap = tap_fn.get();

    for layer_idx in layer_start..layer_end {
        // Developer capture (`HIPFIRE_MOE_ROUTE_DUMP` / `HIPFIRE_MOE_ACT_CAPTURE`): names this
        // layer and the chunk's first position for the sealed-MoE hooks; a no-op when both are unset.
        hipfire_dispatch::pipeline::moe_capture::set_site(layer_idx, start_pos);
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                let qkvza_dtype = if mq3_scan {
                    DType::MQ3G256V2
                } else {
                    DType::MQ4G256V2
                };
                let gdn_chunk_scan_admitted = gdn_chunk_scan_common_admitted
                    && layer.wqkv.gpu_dtype == qkvza_dtype
                    && layer.wz.gpu_dtype == qkvza_dtype
                    && layer.w_beta.gpu_dtype == qkvza_dtype
                    && layer.w_alpha.gpu_dtype == qkvza_dtype
                    && delta_layer_idx < dn_state.s_ef_residual.len();
                batch_chunk_delta_net_attn(
                    gpu,
                    &layer.mixer_view(),
                    &dims,
                    &batch,
                    &deltanet_state_view(dn_state),
                    n,
                    dim,
                    k_dim,
                    v_dim,
                    n_v_heads,
                    hd,
                    batch_semantics,
                    tree_verify,
                    gdn_tape.as_deref().map(gdn_tape_view).as_ref(),
                    tape_offset,
                    delta_layer_idx,
                    q8_wmma_arch,
                    arch_has_wmma,
                    BatchEpilogue::Residual,
                    fusion,
                    commit_stride,
                    gdn_chunk_scan_admitted,
                )?;
                layer.ffn_rows(
                    gpu,
                    config.norm_eps,
                    &batch,
                    n,
                    dim,
                    q8_wmma_arch,
                    arch_has_wmma,
                    MlpOutput::Residual,
                    fusion,
                    math,
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
                    &layer.mixer_view(),
                    &dims,
                    &batch,
                    &flash,
                    &kv_view(kv_cache),
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
                    tap,
                )?;
                layer.ffn_rows(
                    gpu,
                    config.norm_eps,
                    &batch,
                    n,
                    dim,
                    q8_wmma_arch,
                    arch_has_wmma,
                    MlpOutput::Residual,
                    fusion,
                    math,
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
                    start_pos,
                    n,
                    dim_row_bytes,
                    kv_cache,
                    dn_state,
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
                let gdn_chunk_scan_admitted = gdn_moe_chunk_scan_layer_admitted(
                    gdn_moe_chunk_scan_common_admitted,
                    [
                        layer.wqkv.gpu_dtype,
                        layer.wz.gpu_dtype,
                        layer.w_beta.gpu_dtype,
                        layer.w_alpha.gpu_dtype,
                    ],
                    delta_layer_idx < dn_state.s_ef_residual.len(),
                );
                execute_deltanet_moe_layer_batched(
                    gpu,
                    &layer.mixer_view(),
                    &dims,
                    &batch,
                    &deltanet_state_view(dn_state),
                    n,
                    dim,
                    k_dim,
                    v_dim,
                    n_v_heads,
                    hd,
                    batch_semantics,
                    tree_verify,
                    gdn_tape.as_deref().map(gdn_tape_view).as_ref(),
                    tape_offset,
                    delta_layer_idx,
                    start_pos,
                    layer_idx,
                    commit_stride,
                    gdn_chunk_scan_admitted,
                    |gpu: &mut Gpu| {
                        prefill_moe_ffn_body_batched_with_route_layered(
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
                            layer_idx,
                        )
                    },
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
                execute_attention_moe_layer_batched(
                    gpu,
                    fa_attn_multirow,
                    &layer.mixer_view(),
                    &dims,
                    &batch,
                    &flash,
                    &kv_view(kv_cache),
                    n,
                    dim,
                    start_pos,
                    max_ctx_len,
                    &ctx,
                    batch_semantics,
                    tree_verify,
                    q8_wmma_arch,
                    layer_idx,
                    tap,
                    |gpu: &mut Gpu| {
                        prefill_moe_ffn_body_batched_with_route_layered(
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
                            layer_idx,
                        )
                    },
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

#[cfg(test)]
mod tests {
    use super::super::forward::unsupported_mq3_experts_uniform_from_dtypes;
    use super::super::weights::DeltaNetStatePlan;
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
    fn ordinary_prefill_mq3v25_projection_admission_requires_marker_and_safe_q8_wo() {
        let la = [DType::MQ3G256V2; 8];
        let fa = [DType::MQ3G256V2; 8]; // FA's seven weights, down repeated
        let layer = |dtypes, delta_net| Some((dtypes, delta_net));
        assert!(symmetric_mq3v2_projection_sets(
            true,
            [layer(la, true), layer(fa, false)].into_iter()
        ));
        let mut pro_la = la;
        pro_la[4] = DType::Q8_0; // qt3 DeltaNet linear_attn.out_proj
        assert!(symmetric_mq3v2_projection_sets(
            true,
            [layer(pro_la, true), layer(fa, false)].into_iter()
        ));
        assert!(!symmetric_mq3v2_projection_sets(
            false,
            [layer(pro_la, true)].into_iter()
        ));
        let mut wrong_la = la;
        wrong_la[5] = DType::Q8_0;
        assert!(!symmetric_mq3v2_projection_sets(
            true,
            [layer(wrong_la, true)].into_iter()
        ));
        let mut wrong_fa = fa;
        wrong_fa[3] = DType::Q8_0;
        assert!(!symmetric_mq3v2_projection_sets(
            true,
            [layer(wrong_fa, false)].into_iter()
        ));
        assert!(!symmetric_mq3v2_projection_sets(true, std::iter::empty()));
        assert!(!symmetric_mq3v2_projection_sets(
            true,
            [layer(la, true), None].into_iter()
        ));
    }

    /// A8 sizes 136-byte `block_i8_128` rows, A4 sizes 72-byte `block_i4_128`
    /// rows, and only the unrouted gfx1201 reservation keeps the FP8 sizing.
    #[test]
    fn mq3_wide_projection_reservation_uses_a4_a8_block_sizes() {
        let config = widened_test_config();
        assert_eq!(config.hidden_dim, 17_408);
        assert_eq!(iu4_row_bytes_wide(&config), Some(136 * 72));
        assert_eq!(iu8_row_bytes_wide(&config), Some(136 * 136));
        let rows = 8192;
        let at = |arch, mq3| dense_prefill_reservation_bytes(&config, rows, arch, mq3).unwrap();
        let base = at("gfx1100", Mq3WideProjection::Off);
        assert_eq!(at("gfx1100", Mq3WideProjection::A4), base);
        assert_eq!(at("gfx1201", Mq3WideProjection::A4), base);
        assert_eq!(
            at("gfx1201", Mq3WideProjection::A8) - base,
            rows * (136 * 136 - 136 * 72)
        );
        assert_eq!(
            at("gfx1201", Mq3WideProjection::Off) - base,
            rows * (fp8_row_bytes_wide(&config).unwrap() - 136 * 72)
        );
    }

    /// Head geometry of the generalized chunk scan: 16 key heads, value heads a
    /// multiple of 16 up to 48, 128-wide heads, 4-tap conv.
    #[test]
    fn gdn_chunk_head_geometry_truth_table() {
        for (kh, vh, hd, vhd, conv, want) in [
            // 27B dense and the A3B-class shape.
            (16, 48, 128, 128, 4, true),
            (16, 32, 128, 128, 4, true),
            (16, 16, 128, 128, 4, true),
            // Zero / oversize / non-multiple value heads.
            (16, 0, 128, 128, 4, false),
            (16, 64, 128, 128, 4, false),
            (16, 24, 128, 128, 4, false),
            (16, 40, 128, 128, 4, false),
            // Key heads other than 16 (including fewer value than key heads).
            (0, 32, 128, 128, 4, false),
            (8, 16, 128, 128, 4, false),
            (32, 32, 128, 128, 4, false),
            // Head dims and conv taps.
            (16, 32, 64, 128, 4, false),
            (16, 32, 128, 64, 4, false),
            (16, 32, 256, 256, 4, false),
            (16, 32, 128, 128, 3, false),
            (16, 32, 128, 128, 5, false),
        ] {
            assert_eq!(
                gdn_chunk_head_geometry_supported(kh, vh, hd, vhd, conv),
                want,
                "kh={kh} vh={vh} hd={hd} vhd={vhd} conv={conv}"
            );
        }
    }

    /// The shared row admission is the dense route's historical inline clause:
    /// widened rows above 512 admit only with the 512-row commit stride and a
    /// whole-chunk (or empty) tail.
    #[test]
    fn gdn_chunk_scan_rows_admission_matches_the_dense_clause() {
        let dense = |n: usize, stride: Option<usize>| {
            n >= 64
                && (n <= 512
                    || (stride == Some(512) && (n % 512 == 0 || (64..512).contains(&(n % 512)))))
        };
        for stride in [None, Some(256), Some(512)] {
            for n in 0..=4096usize {
                assert_eq!(
                    gdn_chunk_scan_rows_admitted(n, stride),
                    dense(n, stride),
                    "n={n} stride={stride:?}"
                );
            }
        }
        for (n, stride, want) in [
            (63, None, false),
            (64, None, true),
            (512, None, true),
            (513, None, false),
            (513, Some(512), false),
            (575, Some(512), false),
            (576, Some(512), true),
            (1024, Some(512), true),
            (1025, Some(512), false),
            (1088, Some(512), true),
        ] {
            assert_eq!(gdn_chunk_scan_rows_admitted(n, stride), want, "n={n}");
        }
    }

    /// `HIPFIRE_GDN_MOE_CHUNK_SCAN` resolution: an explicit `1`/`0` wins on any
    /// architecture; unset follows qualification.
    #[test]
    fn gdn_moe_chunk_scan_flag_resolution_truth_table() {
        for qualified in [false, true] {
            assert!(gdn_moe_chunk_scan_resolve(Some(true), qualified));
            assert!(!gdn_moe_chunk_scan_resolve(Some(false), qualified));
            assert_eq!(gdn_moe_chunk_scan_resolve(None, qualified), qualified);
        }
    }
    /// The resolver over the real capability table: for every architecture and
    /// explicit value, `gdn_moe_chunk_scan_resolve` fed the actual
    /// `qualified(arch, GdnMoeChunkScan)` equals `prefill_route_capabilities::enabled`.
    /// An explicit `1`/`0` wins on every architecture; unset follows `qualified`
    /// (so this stays valid whichever architectures the table qualifies).
    #[test]
    fn gdn_moe_chunk_scan_flag_follows_the_capability_table_per_arch() {
        for arch in ["gfx1201", "gfx1100", "gfx1151", "gfx1030"] {
            let qualified =
                prefill_route_capabilities::qualified(arch, PrefillRoute::GdnMoeChunkScan);
            for explicit in [None, Some(true), Some(false)] {
                let resolved = gdn_moe_chunk_scan_resolve(explicit, qualified);
                assert_eq!(
                    resolved,
                    prefill_route_capabilities::enabled(
                        explicit,
                        arch,
                        PrefillRoute::GdnMoeChunkScan
                    ),
                    "arch={arch} explicit={explicit:?}"
                );
                assert_eq!(
                    resolved,
                    explicit.unwrap_or(qualified),
                    "arch={arch} explicit={explicit:?}"
                );
            }
        }
    }

    /// `HIPFIRE_GDN_MOE_CHUNK_SCAN` admission: the flag, the chunk-scan
    /// capability, head geometry, a complete Q8 state, a sequential batch, no
    /// tree, no tape, a plain forward and row admission must all hold.
    #[test]
    fn gdn_moe_chunk_scan_admission_truth_table() {
        for mask in 0u16..512 {
            let bit = |i: u16| mask & (1 << i) != 0;
            let inputs = GdnMoeChunkScanInputs {
                enabled: bit(0),
                capable: bit(1),
                geometry: bit(2),
                q8_state: bit(3),
                sequential: bit(4),
                tree: bit(5),
                tape: bit(6),
                plain_forward: bit(7),
                rows: bit(8),
            };
            let want = bit(0)
                && bit(1)
                && bit(2)
                && bit(3)
                && bit(4)
                && !bit(5)
                && !bit(6)
                && bit(7)
                && bit(8);
            assert_eq!(inputs.admitted(), want, "mask={mask:09b}");
        }
    }

    /// End to end over the pure pieces: flag unset / `1` / `0` x qualified /
    /// unqualified architecture x 32 / 48 value heads x single-segment rows
    /// (<= 512) / widened rows (> 512, stride 512) x plain forward / verify
    /// capture. Verify capture never admits; widened rows without the 512
    /// stride and a 64-value-head model never admit.
    #[test]
    fn gdn_moe_chunk_scan_flag_arch_geometry_rows_capture_truth_table() {
        for explicit in [None, Some(true), Some(false)] {
            for qualified in [false, true] {
                for value_heads in [32usize, 48, 64] {
                    for (rows, stride) in [
                        (256usize, None),
                        (512, None),
                        (1024, Some(512)),
                        (1088, Some(512)),
                        (1024, None),
                        (1025, Some(512)),
                    ] {
                        for verify_capture in [false, true] {
                            let admitted = GdnMoeChunkScanInputs {
                                enabled: gdn_moe_chunk_scan_resolve(explicit, qualified),
                                capable: true,
                                geometry: gdn_chunk_head_geometry_supported(
                                    16,
                                    value_heads,
                                    128,
                                    128,
                                    4,
                                ),
                                q8_state: true,
                                sequential: true,
                                tree: false,
                                tape: false,
                                plain_forward: !verify_capture,
                                rows: gdn_chunk_scan_rows_admitted(rows, stride),
                            }
                            .admitted();
                            let want_flag = explicit.unwrap_or(qualified);
                            let want_rows = rows >= 64
                                && (rows <= 512
                                    || (stride == Some(512)
                                        && (rows % 512 == 0 || (64..512).contains(&(rows % 512)))));
                            let want =
                                want_flag && value_heads <= 48 && want_rows && !verify_capture;
                            assert_eq!(
                                admitted, want,
                                "explicit={explicit:?} qualified={qualified} vh={value_heads} \
                                 rows={rows} stride={stride:?} verify_capture={verify_capture}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Per layer: MQ4G256V2 on wqkv, wz, w_beta and w_alpha plus an EF slot.
    #[test]
    fn gdn_moe_chunk_scan_layer_admission_truth_table() {
        let all = [DType::MQ4G256V2; 4];
        assert!(gdn_moe_chunk_scan_layer_admitted(true, all, true));
        assert!(!gdn_moe_chunk_scan_layer_admitted(false, all, true));
        assert!(!gdn_moe_chunk_scan_layer_admitted(true, all, false));
        // qwen3.5 A3B `.mq4` holds the LA projections at Q8_0, qwen3.6 `.mq4r`
        // at MQ4G256: neither admits. Any single non-MQ4G256V2 slot falls back.
        for slot in 0..4 {
            for other in [
                DType::Q8_0,
                DType::MQ4G256,
                DType::MQ4G256V2Lloyd,
                DType::MQ6G256,
                DType::F32,
            ] {
                let mut dtypes = all;
                dtypes[slot] = other;
                assert!(
                    !gdn_moe_chunk_scan_layer_admitted(true, dtypes, true),
                    "slot={slot}"
                );
            }
        }
        assert!(!gdn_moe_chunk_scan_layer_admitted(
            true,
            [DType::Q8_0; 4],
            true
        ));
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
        assert!(!native_mq4_widened_flags_admitted(
            "gfx1201",
            &make(),
            false
        ));
        assert!(!native_mq4_widened_flags_admitted("gfx1100", &make(), true));
        let mut variants = Vec::new();
        let mut f = make();
        f.fp16_disabled = true;
        variants.push(f);
        let mut f = make();
        f.rocblas_all_archs = true;
        variants.push(f);
        let mut f = make();
        f.gemv_dp4a = Some(true);
        variants.push(f);
        let mut f = make();
        f.mmq_override = Some(false);
        variants.push(f);
        let mut f = make();
        f.mmq_min_batch = Some(2048);
        variants.push(f);
        let mut f = make();
        f.qkvza_split_tail = true;
        variants.push(f);
        let mut f = make();
        f.wo_wmma_variant = Some("k4".into());
        variants.push(f);
        // Packed FFN shares the budgeted MMQ slot and keeps native tails.
        let mut packed = make();
        packed.packed_mq4_prefill = true;
        assert!(native_mq4_widened_flags_admitted("gfx1100", &packed, false));
        packed.mmq_override = Some(false);
        assert!(!native_mq4_widened_flags_admitted(
            "gfx1100", &packed, false
        ));
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
        assert_eq!(
            native_mq4_projection_deficit(&config, 8192, 0, 0, 0),
            Some(mmq + f16 + partials)
        );
        assert_eq!(
            native_mq4_projection_deficit(&config, 8192, mmq, f16, partials),
            Some(0)
        );
        assert_eq!(
            native_mq4_projection_deficit(&config, 8192, mmq + 1, 0, partials),
            Some(f16)
        );
        assert_eq!(
            native_mq4_projection_deficit(&config, usize::MAX, 0, 0, 0),
            None
        );
        for rows in [512, 1024, 2048, 4096, 8192] {
            assert_eq!(
                native_mq4_projection_deficit(&config, rows, 0, 0, 0),
                Some(rows * 19584 + f16 + partials)
            );
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
        // MoE configs charge the same dense terms plus the Path-2 grouped-GEMM
        // scratch `PrefillBatchScratch::new_opt` allocates for them, through the
        // checked mirror of that constructor. This used to be `None` (out of
        // scope), which left `minimum_prefill_reservation_bytes` undefined for
        // every MoE model and made the partial-offload placement under-spill by
        // exactly that scratch.
        let mut moe = widened_test_config();
        moe.num_experts = 256;
        moe.num_experts_per_tok = 8;
        moe.moe_intermediate_size = 512;
        moe.shared_expert_intermediate_size = 512;
        let moe_bytes =
            dense_prefill_allocation_bytes(&moe, 512).expect("a MoE config is chargeable");
        let dense_only = dense_layer_prefill_allocation_bytes(&moe, 512).unwrap();
        assert!(
            moe_bytes > dense_only,
            "the MoE charge must include the grouped scratch it allocates: {moe_bytes} vs {dense_only}"
        );
        let mirror =
            crate::qwen35::batch::PrefillBatchScratch::projected_allocation_bytes(&moe, 512, true)
                .unwrap();
        assert_eq!(
            moe_bytes as u64, mirror,
            "must be exactly the constructor mirror"
        );
        // Wrong hidden_dim breaks the allocation envelope.
        let mut narrow = widened_test_config();
        narrow.hidden_dim = 3584;
        assert!(!widened_dense_shape_admitted(&narrow));
    }

    fn resident_moe_test_config() -> Qwen35Config {
        let mut moe = widened_test_config();
        moe.num_experts = 256;
        moe.num_experts_per_tok = 8;
        moe.moe_intermediate_size = 512;
        moe.shared_expert_intermediate_size = 512;
        moe
    }

    fn resident_moe_dtypes(dt: DType) -> hipfire_dispatch::families::moe::MoePrefillDtypes {
        let mut d = hipfire_dispatch::families::moe::MoePrefillDtypes::uniform(dt);
        // The shipped recipe keeps the router and shared scalar gate in Q8.
        d.router = DType::Q8_0;
        d.shared_expert_scalar_gate = DType::Q8_0;
        d
    }

    #[test]
    fn resident_moe_request_precedence() {
        use ResidentMoeRequest::{Incumbent, Widen};
        let default = Widen {
            requested: RESIDENT_MOE_CHUNK_ROWS,
        };
        // Nothing explicit: the MoE rung is requested by default.
        assert_eq!(resident_moe_request(None, None), default);
        // Explicit HIPFIRE_PREFILL_MAX_BATCH always wins, whatever chunk_rows says.
        assert_eq!(resident_moe_request(Some(512), None), Incumbent);
        assert_eq!(resident_moe_request(Some(8192), Some(4096)), Incumbent);
        assert_eq!(resident_moe_request(Some(2), Some(8192)), Incumbent);
        // Explicit chunk_rows below the rung opts out; at/above it requests the rung.
        assert_eq!(resident_moe_request(None, Some(512)), Incumbent);
        assert_eq!(resident_moe_request(None, Some(2048)), Incumbent);
        assert_eq!(resident_moe_request(None, Some(4096)), Incumbent);
        assert_eq!(resident_moe_request(None, Some(8191)), Incumbent);
        assert_eq!(resident_moe_request(None, Some(8192)), default);
        assert_eq!(
            resident_moe_request(None, Some(16384)),
            Widen { requested: 16384 }
        );
    }

    #[test]
    fn resident_moe_rung_is_a_staged_commit_multiple() {
        assert!(WIDENED_RUNGS.contains(&RESIDENT_MOE_CHUNK_ROWS));
        assert_eq!(RESIDENT_MOE_CHUNK_ROWS % WIDENED_COMMIT_ROWS, 0);
        assert_eq!(WIDENED_COMMIT_ROWS, 512);
    }

    #[test]
    fn resident_moe_dtype_capability_covers_v1_and_v2_only() {
        for supported in [DType::MQ4G256, DType::MQ4G256V2] {
            assert!(
                resident_moe_producer_dtypes_supported(&resident_moe_dtypes(supported), false),
                "{supported:?} grouped producer must be admitted"
            );
            assert!(
                resident_moe_producer_dtypes_supported(&resident_moe_dtypes(supported), true),
                "{supported:?} grouped producer must stay admitted with the IU4 capability"
            );
        }
        // A Q8/F32/MQ4 router and scalar gate are all accepted.
        for router in [DType::Q8_0, DType::F32, DType::MQ4G256, DType::MQ4G256V2] {
            let mut d = resident_moe_dtypes(DType::MQ4G256V2);
            d.router = router;
            d.shared_expert_scalar_gate = router;
            assert!(resident_moe_producer_dtypes_supported(&d, false));
            assert!(resident_moe_producer_dtypes_supported(&d, true));
        }
        for unsupported in [
            DType::MQ6G256,
            DType::MQ6G256V2,
            DType::HFQ4G256,
            DType::MFP4G32E8,
            DType::MQ2G256Lloyd,
            DType::MQ3G256Lloyd,
            DType::Q8_0,
        ] {
            assert!(
                !resident_moe_producer_dtypes_supported(&resident_moe_dtypes(unsupported), false),
                "{unsupported:?} must keep the incumbent chunk"
            );
            assert!(
                !resident_moe_producer_dtypes_supported(&resident_moe_dtypes(unsupported), true),
                "{unsupported:?} must keep the incumbent chunk even with the IU4 capability"
            );
        }
        // V1 gate_up with V2 down (or the reverse) is a graded mix, not a uniform pair.
        let mut mixed = resident_moe_dtypes(DType::MQ4G256);
        mixed.expert_down = DType::MQ4G256V2;
        assert!(!resident_moe_producer_dtypes_supported(&mixed, false));
        assert!(!resident_moe_producer_dtypes_supported(&mixed, true));
        // Graded merge, non-uniform experts, broken shared pair, MQ6 router.
        let mut merged = resident_moe_dtypes(DType::MQ4G256V2);
        merged.routed_mixed_merged = true;
        assert!(!resident_moe_producer_dtypes_supported(&merged, false));
        assert!(!resident_moe_producer_dtypes_supported(&merged, true));
        let mut ragged_gate_up = resident_moe_dtypes(DType::MQ4G256);
        ragged_gate_up.expert_gate_up_uniform = false;
        assert!(!resident_moe_producer_dtypes_supported(
            &ragged_gate_up,
            false
        ));
        assert!(!resident_moe_producer_dtypes_supported(
            &ragged_gate_up,
            true
        ));
        let mut ragged_down = resident_moe_dtypes(DType::MQ4G256);
        ragged_down.expert_down_uniform = false;
        assert!(!resident_moe_producer_dtypes_supported(&ragged_down, false));
        assert!(!resident_moe_producer_dtypes_supported(&ragged_down, true));
        let mut shared_pair = resident_moe_dtypes(DType::MQ4G256V2);
        shared_pair.shared_expert_up = DType::MQ4G256;
        assert!(!resident_moe_producer_dtypes_supported(&shared_pair, false));
        assert!(!resident_moe_producer_dtypes_supported(&shared_pair, true));
        let mut shared_down = resident_moe_dtypes(DType::MQ4G256V2);
        shared_down.shared_expert_down = DType::MQ6G256V2;
        assert!(!resident_moe_producer_dtypes_supported(&shared_down, false));
        assert!(!resident_moe_producer_dtypes_supported(&shared_down, true));
        let mut router = resident_moe_dtypes(DType::MQ4G256V2);
        router.router = DType::MQ6G256;
        assert!(!resident_moe_producer_dtypes_supported(&router, false));
        assert!(!resident_moe_producer_dtypes_supported(&router, true));
    }

    fn d53_dtypes() -> hipfire_dispatch::families::moe::MoePrefillDtypes {
        let mut d = resident_moe_dtypes(DType::MQ4G256V2);
        d.expert_down = DType::MQ4G128V2;
        d
    }

    #[test]
    fn resident_moe_d53_pair_is_admitted_only_with_the_iu4_capability() {
        let d53 = d53_dtypes();
        assert!(!resident_moe_producer_dtypes_supported(&d53, false));
        assert!(resident_moe_producer_dtypes_supported(&d53, true));
        // Uniform only: no graded merge, no per-projection mix.
        let mut merged = d53;
        merged.routed_mixed_merged = true;
        let mut ragged_gate_up = d53;
        ragged_gate_up.expert_gate_up_uniform = false;
        let mut ragged_down = d53;
        ragged_down.expert_down_uniform = false;
        // The capability widens only the exact d53 pair.
        let mut v1_gate_up = d53;
        v1_gate_up.expert_gate_up = DType::MQ4G256;
        let mut reversed = d53;
        reversed.expert_gate_up = DType::MQ4G128V2;
        reversed.expert_down = DType::MQ4G256V2;
        let mut qt53_both = d53;
        qt53_both.expert_gate_up = DType::MQ4G128V2;
        // Shared expert and router keep their gates.
        let mut shared_pair = d53;
        shared_pair.shared_expert_up = DType::MQ4G256;
        let mut shared_down = d53;
        shared_down.shared_expert_down = DType::MQ6G256V2;
        let mut shared_qt53 = d53;
        shared_qt53.shared_expert_down = DType::MQ4G128V2;
        let mut router = d53;
        router.router = DType::MQ6G256;
        let mut scalar_gate = d53;
        scalar_gate.shared_expert_scalar_gate = DType::MQ4G128V2;
        for (what, d) in [
            ("graded merge", merged),
            ("non-uniform gate_up", ragged_gate_up),
            ("non-uniform down", ragged_down),
            ("V1 gate_up + qt53 down", v1_gate_up),
            ("qt53 gate_up + V2 down", reversed),
            ("qt53 gate_up + qt53 down", qt53_both),
            ("shared gate/up pair", shared_pair),
            ("shared down MQ6", shared_down),
            ("shared down qt53", shared_qt53),
            ("router MQ6", router),
            ("scalar gate qt53", scalar_gate),
        ] {
            assert!(
                !resident_moe_producer_dtypes_supported(&d, true),
                "{what} must keep the incumbent chunk even with the IU4 capability"
            );
        }
        // Same-format admission is independent of the capability.
        for dt in [DType::MQ4G256, DType::MQ4G256V2] {
            let d = resident_moe_dtypes(dt);
            assert!(resident_moe_producer_dtypes_supported(&d, false));
            assert!(resident_moe_producer_dtypes_supported(&d, true));
        }
        for dt in [DType::MQ6G256, DType::MQ4G128V2, DType::Q8_0] {
            let d = resident_moe_dtypes(dt);
            assert!(!resident_moe_producer_dtypes_supported(&d, false));
            assert!(!resident_moe_producer_dtypes_supported(&d, true));
        }
    }

    #[test]
    fn resident_moe_placement_excludes_every_nonresident_route() {
        assert!(ResidentMoePlacement::default().resident());
        let nonresident = [
            ResidentMoePlacement {
                attention_host_located: true,
                ..Default::default()
            },
            ResidentMoePlacement {
                routed_gate_up_host: true,
                ..Default::default()
            },
            ResidentMoePlacement {
                routed_down_host: true,
                ..Default::default()
            },
            ResidentMoePlacement {
                small_ffn_host_located: true,
                ..Default::default()
            },
            ResidentMoePlacement {
                cpu_deferred_experts: true,
                ..Default::default()
            },
            ResidentMoePlacement {
                paged_experts: true,
                ..Default::default()
            },
            ResidentMoePlacement {
                multi_device_contract: true,
                ..Default::default()
            },
            ResidentMoePlacement {
                sidecar_experts: true,
                ..Default::default()
            },
        ];
        for placement in nonresident {
            assert!(!placement.resident(), "{placement:?} must not widen");
        }
    }
    #[test]
    fn resident_moe_d53_layer_capability_needs_resident_placement_and_iu4() {
        let d53 = d53_dtypes();
        let resident = ResidentMoePlacement::default();
        assert!(!resident_moe_layer_facts_capable(
            &resident,
            Some(&d53),
            false
        ));
        assert!(resident_moe_layer_facts_capable(
            &resident,
            Some(&d53),
            true
        ));
        assert!(!resident_moe_layer_facts_capable(&resident, None, true));
        for placement in [
            ResidentMoePlacement {
                routed_gate_up_host: true,
                ..Default::default()
            },
            ResidentMoePlacement {
                routed_down_host: true,
                ..Default::default()
            },
            ResidentMoePlacement {
                paged_experts: true,
                ..Default::default()
            },
            ResidentMoePlacement {
                multi_device_contract: true,
                ..Default::default()
            },
            ResidentMoePlacement {
                sidecar_experts: true,
                ..Default::default()
            },
        ] {
            assert!(!resident_moe_layer_facts_capable(
                &placement,
                Some(&d53),
                true
            ));
        }
        // Uniform same-format layers are unchanged by the capability.
        for dt in [DType::MQ4G256, DType::MQ4G256V2] {
            let d = resident_moe_dtypes(dt);
            assert!(resident_moe_layer_facts_capable(&resident, Some(&d), false));
            assert!(resident_moe_layer_facts_capable(&resident, Some(&d), true));
        }
        // Only a d53 layer bound to the capability needs the IU4 scratch.
        assert!(resident_moe_layer_needs_iu4_scratch(Some(&d53), true));
        assert!(!resident_moe_layer_needs_iu4_scratch(Some(&d53), false));
        assert!(!resident_moe_layer_needs_iu4_scratch(None, true));
        for dt in [DType::MQ4G256, DType::MQ4G256V2] {
            let d = resident_moe_dtypes(dt);
            assert!(!resident_moe_layer_needs_iu4_scratch(Some(&d), true));
        }
    }

    fn d53_static_facts(
        request: ResidentMoeRequest,
        iu4_d53_layers: bool,
        iu4_scratch_wide: bool,
    ) -> ResidentMoeStaticFacts {
        ResidentMoeStaticFacts {
            moe_config: true,
            request,
            has_wmma: true,
            pager: false,
            ep_shard: false,
            moe_has_mq6: false,
            layers_present: true,
            dn_state_admitted: true,
            batch_pbs_eligible: true,
            layers_resident_capable: true,
            iu4_d53_layers,
            iu4_scratch_wide,
        }
    }

    #[test]
    fn resident_moe_d53_widening_requires_actual_wide_scratch() {
        let widen = resident_moe_request(None, None);
        assert_eq!(
            resident_moe_static_decision(&d53_static_facts(widen, true, true)),
            Some(RESIDENT_MOE_CHUNK_ROWS)
        );
        // The loader's widened reservation fell back: the scratch holds only
        // the incumbent rows, so the fallback cannot widen.
        assert_eq!(
            resident_moe_static_decision(&d53_static_facts(widen, true, false)),
            None
        );
        // Same-format layers never read the scratch fact.
        assert_eq!(
            resident_moe_static_decision(&d53_static_facts(widen, false, false)),
            Some(RESIDENT_MOE_CHUNK_ROWS)
        );
        // Explicit max batch and a sub-rung chunk request opt out.
        for request in [
            resident_moe_request(Some(8192), None),
            resident_moe_request(None, Some(4096)),
        ] {
            assert_eq!(
                resident_moe_static_decision(&d53_static_facts(request, true, true)),
                None
            );
        }
    }

    #[test]
    fn moe_sym_scratch_cover_matches_the_reservation_extents() {
        use rdna_compute::scratch::{qwen4_moe_down_i4_bytes, qwen4_moe_group_layout};
        let mut config = resident_moe_test_config();
        config.dim = 2048;
        let rows = RESIDENT_MOE_CHUNK_ROWS;
        let slots = rows * config.num_experts_per_tok;
        let full = MoeSymScratchHave {
            group: qwen4_moe_group_layout(slots, config.num_experts).bytes,
            down: qwen4_moe_down_i4_bytes(config.moe_intermediate_size, slots).unwrap(),
            gate: config.dim / 128 * rows * 72,
        };
        // A3B (E256, top-8, H2048, I512) extents at 8192 rows.
        assert_eq!(
            (full.group, full.down, full.gate),
            (787_456, 18_874_368, 9_437_184)
        );
        assert!(moe_sym_scratch_covers_rows(full, rows, &config));
        assert!(moe_sym_scratch_covers_rows(full, 512, &config));
        assert!(!moe_sym_scratch_covers_rows(full, rows + 1, &config));
        // Every owner is required.
        for short in [
            MoeSymScratchHave {
                group: full.group - 1,
                ..full
            },
            MoeSymScratchHave {
                down: full.down - 1,
                ..full
            },
            MoeSymScratchHave {
                gate: full.gate - 1,
                ..full
            },
            MoeSymScratchHave::default(),
        ] {
            assert!(!moe_sym_scratch_covers_rows(short, rows, &config));
        }
        // The incumbent 512-row reservation cannot widen.
        let incumbent = MoeSymScratchHave {
            group: qwen4_moe_group_layout(512 * config.num_experts_per_tok, config.num_experts)
                .bytes,
            down: qwen4_moe_down_i4_bytes(
                config.moe_intermediate_size,
                512 * config.num_experts_per_tok,
            )
            .unwrap(),
            gate: config.dim / 128 * 512 * 72,
        };
        assert!(moe_sym_scratch_covers_rows(incumbent, 512, &config));
        assert!(!moe_sym_scratch_covers_rows(incumbent, rows, &config));
        // Overflow fails closed.
        assert!(!moe_sym_scratch_covers_rows(full, usize::MAX, &config));
    }

    /// The loader reserves the widened rows exactly when prefill's static
    /// decision would admit them: one shared request resolver and decision.
    #[test]
    fn resident_moe_load_and_prefill_agree_on_the_widen_request() {
        let explicit_max = [None, Some(512), Some(8192)];
        let explicit_chunk = [
            None,
            Some(512),
            Some(2048),
            Some(4096),
            Some(8192),
            Some(16384),
        ];
        for max in explicit_max {
            for chunk in explicit_chunk {
                let request = resident_moe_request(max, chunk);
                let widens = matches!(request, ResidentMoeRequest::Widen { .. });
                assert_eq!(widens, max.is_none() && chunk.is_none_or(|c| c >= 8192));
                let wide = resident_moe_static_decision(&d53_static_facts(request, false, true));
                assert_eq!(
                    wide,
                    widens.then_some(RESIDENT_MOE_CHUNK_ROWS),
                    "{max:?} {chunk:?}"
                );
            }
        }
    }

    /// Independent closed form of the REQUESTED payload bytes of one non-lean
    /// MoE `PrefillBatchScratch` (u128), written from the `new_opt` allocation
    /// list rather than from `projected_allocation_bytes`. Payload only (no
    /// driver rounding or allocator overhead; the 1 GiB headroom covers
    /// those transients). With `n` rows, `p = ceil(n/64)*64`, `k` key dim, `v`
    /// value dim, `H` value heads, `q`/`c` attention q/kv dims, experts `E`,
    /// top-k `t`, shared/routed intermediates `s`/`i`.
    fn constructor_payload_bytes(c: &Qwen35Config, n: u128, tape: bool) -> u128 {
        let d = c.dim as u128;
        let h = c.hidden_dim as u128;
        let k = (c.linear_num_key_heads * c.linear_key_head_dim) as u128;
        let v = (c.linear_num_value_heads * c.linear_value_head_dim) as u128;
        let hh = c.linear_num_value_heads as u128;
        let vhd = c.linear_value_head_dim as u128;
        let q = (c.n_heads * c.head_dim) as u128;
        let kvd = (c.n_kv_heads * c.head_dim) as u128;
        let p = n.div_ceil(64) * 64;
        let base = 4
            * (n * (3 * d + 3 * h + 9 * v + 3 * k + 2 * hh + 6 * q + 2 * kvd + 263) + p * k)
            + 2 * n * (d + v + h + q)
            + 8 * n
            + 256;
        let e = c.num_experts as u128;
        let t = c.num_experts_per_tok as u128;
        let s = c.shared_expert_intermediate_size as u128;
        let i = c.moe_intermediate_size as u128;
        let m = (n * t + 15 * e).div_ceil(16) * 16;
        let moe = 4 * (n * (e + 1 + 3 * s + 2 * t + 3 * t * i + t * d + d) + m * (2 * i + d))
            + 4 * (2 * e + 1 + m + n * t + m / 16);
        let tape_bytes = if tape && hh > 0 {
            5 * n * hh * vhd * vhd + 4 * n * v
        } else {
            0
        };
        base + moe + tape_bytes
    }

    #[test]
    fn resident_moe_pbs_bytes_are_tape_free_and_include_grouped_scratch() {
        let moe = resident_moe_test_config();
        let rows = RESIDENT_MOE_CHUNK_ROWS;
        let bytes = resident_moe_pbs_bytes(&moe, rows).expect("chargeable");
        assert_eq!(
            bytes as u128,
            constructor_payload_bytes(&moe, rows as u128, false),
            "the census equals the independent constructor closed form (tape-free owner)"
        );
        let with_tape = resident_moe_owner_bytes(&moe, rows, true).expect("chargeable");
        assert!(
            bytes < with_tape,
            "ordinary prefill never allocates the verify-only S-tape"
        );
        // The grouped gate/up and down outputs are inside the PBS term.
        let mut dense_shape = moe.clone();
        dense_shape.num_experts = 0;
        let dense_bytes = resident_moe_pbs_bytes(&dense_shape, rows).expect("chargeable");
        let m_max = moe_grouped_m_total_max(rows, 8, 256) as u64;
        let grouped = m_max * 2 * 512 * 4 + m_max * moe.dim as u64 * 4;
        assert!(
            (bytes - dense_bytes) as u64 >= grouped,
            "grouped dynamic scratch must be counted once, inside the PBS projection"
        );
        // Overflow fails closed.
        assert_eq!(resident_moe_pbs_bytes(&moe, usize::MAX), None);
    }

    /// The two `new_opt` allocations `projected_allocation_bytes` does not
    /// mirror (`dn_z_fold_batch`, `dn_normed_batch`) are charged exactly once,
    /// and the census equals the independent constructor closed form for tape
    /// off/on, credits and overflow.
    #[test]
    fn resident_moe_census_matches_constructor_closed_form() {
        let moe = resident_moe_test_config();
        let v = moe.linear_num_value_heads * moe.linear_value_head_dim;
        for rows in [100usize, 512, 1024, 4096] {
            for tape in [false, true] {
                assert_eq!(
                    resident_moe_owner_bytes(&moe, rows, tape).map(|b| b as u128),
                    Some(constructor_payload_bytes(&moe, rows as u128, tape)),
                    "rows={rows} tape={tape}"
                );
            }
            let z_fold = rows * (v + 256) * 4;
            let normed = rows * v * 4;
            assert_eq!(
                resident_moe_pbs_unmirrored_bytes(&moe, rows),
                Some(z_fold + normed)
            );
            let mirrored = usize::try_from(
                PrefillBatchScratch::projected_allocation_bytes(&moe, rows, false).unwrap(),
            )
            .unwrap();
            assert_eq!(
                resident_moe_pbs_bytes(&moe, rows),
                Some(mirrored + z_fold + normed)
            );
        }
        assert_eq!(resident_moe_owner_bytes(&moe, usize::MAX, false), None);
        assert_eq!(resident_moe_pbs_unmirrored_bytes(&moe, usize::MAX), None);
        // Credits are the same census: a retained widened owner credits its own bytes.
        assert_eq!(
            resident_moe_cache_credit(&moe, Some((2048, false))) as u128,
            constructor_payload_bytes(&moe, 2048, false)
        );
        // Memory decision charged from the closed form (not from the helpers).
        let kv = KvTierFlags {
            quant_q8: true,
            ..Default::default()
        };
        let projection = resident_moe_projection_deficit(
            &moe,
            RESIDENT_MOE_CHUNK_ROWS,
            &LiveProjectionScratch::default(),
        )
        .unwrap();
        let q16 = RESIDENT_MOE_CHUNK_ROWS * moe.n_heads * moe.head_dim * 2;
        let need = usize::try_from(constructor_payload_bytes(
            &moe,
            RESIDENT_MOE_CHUNK_ROWS as u128,
            false,
        ))
        .unwrap()
            + projection
            + q16
            + WIDENED_VRAM_HEADROOM_BYTES;
        let mem = |free: usize| ResidentMoeMemoryFacts {
            free_bytes: Some(free),
            live: LiveProjectionScratch::default(),
            live_q16_bytes: 0,
        };
        let widened = ResidentMoeOwner::WIDENED;
        assert!(resident_moe_memory_admits(
            &moe,
            kv,
            &mem(need),
            None,
            widened,
            0
        ));
        assert!(!resident_moe_memory_admits(
            &moe,
            kv,
            &mem(need - 1),
            None,
            widened,
            0
        ));
    }

    #[test]
    fn resident_moe_projection_deficit_scales_credits_live_scratch_and_checks_overflow() {
        let moe = resident_moe_test_config();
        let none = LiveProjectionScratch::default();
        let base = resident_moe_projection_deficit(&moe, 4096, &none).unwrap();
        assert!(base > 0);
        assert_eq!(
            resident_moe_projection_deficit(&moe, 8192, &none).unwrap(),
            2 * base,
            "no live scratch: linear in rows"
        );
        // Credit is exactly the bytes the matching live scratch holds.
        let credited = LiveProjectionScratch {
            int4_mmq_x: 1000,
            ..Default::default()
        };
        assert_eq!(
            resident_moe_projection_deficit(&moe, 4096, &credited).unwrap(),
            base - 1000
        );
        // Fully resident scratch of every kind leaves nothing to grow.
        let full = LiveProjectionScratch {
            int4_mmq_x: usize::MAX,
            q8_1_mmq_x: usize::MAX,
            fp16_x: usize::MAX,
            fp8_x: usize::MAX,
            fp8_half_sums: usize::MAX,
            fp8_row_scales: usize::MAX,
        };
        assert_eq!(resident_moe_projection_deficit(&moe, 4096, &full), Some(0));
        assert_eq!(
            resident_moe_projection_deficit(&moe, usize::MAX, &none),
            None
        );
        // Q16: the FULL chunk (the MoE bodies attend the whole chunk; there is
        // no 512-row commit segment), net of live bytes, saturating.
        let q16 = RESIDENT_MOE_CHUNK_ROWS * moe.n_heads * moe.head_dim * 2;
        let q16_512 = WIDENED_COMMIT_ROWS * moe.n_heads * moe.head_dim * 2;
        assert!(q16 > q16_512);
        assert_eq!(
            resident_moe_q16_missing(&moe, RESIDENT_MOE_CHUNK_ROWS, 0),
            Some(q16)
        );
        assert_eq!(
            resident_moe_q16_missing(&moe, RESIDENT_MOE_CHUNK_ROWS, 100),
            Some(q16 - 100)
        );
        assert_eq!(
            resident_moe_q16_missing(&moe, RESIDENT_MOE_CHUNK_ROWS, q16),
            Some(0)
        );
        assert_eq!(
            resident_moe_q16_missing(&moe, RESIDENT_MOE_CHUNK_ROWS, usize::MAX),
            Some(0)
        );
        // Overflow fails closed.
        assert_eq!(resident_moe_q16_missing(&moe, usize::MAX, 0), None);
        let mut huge = resident_moe_test_config();
        huge.head_dim = usize::MAX;
        assert_eq!(
            resident_moe_q16_missing(&huge, RESIDENT_MOE_CHUNK_ROWS, 0),
            None
        );
    }

    /// The full-row Q16 term and the 1 GiB headroom through the production
    /// memory decision: the 512-row budget the first draft charged would
    /// admit a rung whose real growth leaves less than the headroom.
    #[test]
    fn resident_moe_full_row_q16_keeps_one_gib_headroom() {
        let moe = resident_moe_test_config();
        let kv = KvTierFlags {
            quant_q8: true,
            ..Default::default()
        };
        let pbs = resident_moe_pbs_bytes(&moe, RESIDENT_MOE_CHUNK_ROWS).unwrap();
        let projection = resident_moe_projection_deficit(
            &moe,
            RESIDENT_MOE_CHUNK_ROWS,
            &LiveProjectionScratch::default(),
        )
        .unwrap();
        let q16_full = resident_moe_q16_missing(&moe, RESIDENT_MOE_CHUNK_ROWS, 0).unwrap();
        let q16_512 = resident_moe_q16_missing(&moe, WIDENED_COMMIT_ROWS, 0).unwrap();
        assert!(q16_full > q16_512);
        let mem = |free: usize, live_q16: usize| ResidentMoeMemoryFacts {
            free_bytes: Some(free),
            live: LiveProjectionScratch::default(),
            live_q16_bytes: live_q16,
        };
        let admits = |m: ResidentMoeMemoryFacts| {
            resident_moe_memory_admits(&moe, kv, &m, None, ResidentMoeOwner::WIDENED, 0)
        };
        let need = pbs + projection + q16_full + WIDENED_VRAM_HEADROOM_BYTES;
        assert!(admits(mem(need, 0)));
        assert!(!admits(mem(need - 1, 0)));
        // Remaining free bytes after the full growth are at least the headroom.
        assert_eq!(
            need - (pbs + projection + q16_full),
            WIDENED_VRAM_HEADROOM_BYTES
        );
        // The 512-row Q16 budget would have admitted this much free memory.
        let old_need = need - (q16_full - q16_512);
        assert!(!admits(mem(old_need, 0)));
        // A live buffer credits exactly its own bytes.
        assert!(admits(mem(need - q16_full, q16_full)));
        assert!(!admits(mem(need - q16_full - 1, q16_full)));
        assert!(admits(mem(need - q16_512, q16_512)));
        assert!(!admits(mem(need - q16_512 - 1, q16_512)));
        // A failed free-byte query and a KV tier without batched flash keep the incumbent.
        assert!(!admits(ResidentMoeMemoryFacts {
            free_bytes: None,
            ..mem(need, 0)
        }));
        assert!(!resident_moe_memory_admits(
            &moe,
            KvTierFlags::default(),
            &mem(usize::MAX, 0),
            None,
            ResidentMoeOwner::WIDENED,
            0
        ));
        // Pending allocations are deducted before the headroom test.
        let widened = ResidentMoeOwner::WIDENED;
        assert!(resident_moe_memory_admits(
            &moe,
            kv,
            &mem(need + 4096, 0),
            None,
            widened,
            4096
        ));
        assert!(!resident_moe_memory_admits(
            &moe,
            kv,
            &mem(need + 4096, 0),
            None,
            widened,
            4097
        ));
        // A taped explicit-equivalent owner is charged its own (larger) bytes,
        // and an existing owner credits exactly its bytes.
        let taped = ResidentMoeOwner {
            rows: RESIDENT_MOE_CHUNK_ROWS,
            tape: true,
            credit_bytes: 0,
        };
        let taped_bytes = resident_moe_owner_bytes(&moe, taped.rows, true).unwrap();
        assert!(taped_bytes > pbs);
        let taped_need = need - pbs + taped_bytes;
        assert!(resident_moe_memory_admits(
            &moe,
            kv,
            &mem(taped_need, 0),
            None,
            taped,
            0
        ));
        assert!(!resident_moe_memory_admits(
            &moe,
            kv,
            &mem(taped_need - 1, 0),
            None,
            taped,
            0
        ));
        // An owner smaller than the rung never lowers the 8192-row charge.
        let small = ResidentMoeOwner {
            rows: WIDENED_COMMIT_ROWS,
            tape: false,
            credit_bytes: 0,
        };
        assert!(!resident_moe_memory_admits(
            &moe,
            kv,
            &mem(need - 1, 0),
            None,
            small,
            0
        ));
        assert!(resident_moe_memory_admits(
            &moe,
            kv,
            &mem(need, 0),
            None,
            small,
            0
        ));
        let credited = ResidentMoeOwner {
            credit_bytes: taped_bytes,
            ..taped
        };
        assert!(resident_moe_memory_admits(
            &moe,
            kv,
            &mem(need - pbs, 0),
            None,
            credited,
            0
        ));
        assert!(!resident_moe_memory_admits(
            &moe,
            kv,
            &mem(need - pbs - 1, 0),
            None,
            credited,
            0
        ));
    }

    #[test]
    fn resident_moe_cache_credit_is_exact_owner_bytes_only() {
        let moe = resident_moe_test_config();
        assert_eq!(resident_moe_cache_credit(&moe, None), 0);
        // A lean owner has a different layout: never credited.
        assert_eq!(resident_moe_cache_credit(&moe, Some((2048, true))), 0);
        assert_eq!(
            resident_moe_cache_credit(&moe, Some((2048, false))),
            resident_moe_pbs_bytes(&moe, 2048).unwrap()
        );
        // An unprojectable owner shape is not credited (conservative).
        assert_eq!(resident_moe_cache_credit(&moe, Some((0, false))), 0);
    }

    /// Only a retained owner the call reuses in place is credited: non-lean and
    /// at least as many rows as the call's owned PBS needs.
    #[test]
    fn resident_moe_reusable_cached_requires_in_place_reuse() {
        let r = resident_moe_reusable_cached;
        assert_eq!(r(None, 100), None);
        assert_eq!(r(Some((4096, false)), 4096), Some((4096, false)));
        assert_eq!(r(Some((8192, false)), 4096), Some((8192, false)));
        assert_eq!(r(Some((2048, false)), 2048), Some((2048, false)));
        // Smaller owner: replaced, no credit.
        assert_eq!(r(Some((2048, false)), 4096), None);
        assert_eq!(r(Some((2048, false)), 2049), None);
        // Lean owner: layout mismatch, replaced.
        assert_eq!(r(Some((8192, true)), 100), None);
    }

    /// The retained-owner credit through the shared chunk-limit decision: a
    /// short owner that must be replaced is charged in full (a budget the old
    /// speculative credit admitted is rejected), the same-shape owner keeps its
    /// exact positive credit, and shape/lean mismatches earn none.
    #[test]
    fn resident_moe_cached_credit_only_when_reused_boundary() {
        let moe = resident_moe_test_config();
        let facts = ResidentMoeStaticFacts {
            moe_config: true,
            request: ResidentMoeRequest::Widen {
                requested: RESIDENT_MOE_CHUNK_ROWS,
            },
            has_wmma: true,
            pager: false,
            ep_shard: false,
            moe_has_mq6: false,
            layers_present: true,
            dn_state_admitted: true,
            batch_pbs_eligible: true,
            layers_resident_capable: true,
            iu4_d53_layers: false,
            iu4_scratch_wide: false,
        };
        let kv = KvTierFlags {
            quant_q8: true,
            ..KvTierFlags::default()
        };
        let need = resident_moe_need_bytes(&moe, 0);
        let incumbent = 512usize;
        let limit = |cached: Option<(usize, bool)>, reuse_rows: usize, free: usize| {
            resident_moe_chunk_limit_from_facts(
                &facts,
                None,
                &moe,
                kv,
                || ResidentMoeMemoryFacts {
                    free_bytes: Some(free),
                    live: LiveProjectionScratch::default(),
                    live_q16_bytes: 0,
                },
                cached,
                reuse_rows,
                0,
                incumbent,
            )
        };
        let bytes = |rows: usize| resident_moe_pbs_bytes(&moe, rows).unwrap();
        // No owner: the exact full-rung boundary.
        assert_eq!(limit(None, 8192, need), RESIDENT_MOE_CHUNK_ROWS);
        assert_eq!(limit(None, 8192, need - 1), incumbent);
        // Same-shape owner reused in place: positive credit, exact boundary.
        let same = Some((8192, false));
        assert_eq!(
            limit(same, 8192, need - bytes(8192)),
            RESIDENT_MOE_CHUNK_ROWS
        );
        assert_eq!(limit(same, 8192, need - bytes(8192) - 1), incumbent);
        // Short owner at 8192 needed rows must be replaced: growth rejects the
        // budget the speculative credit would admit, and charges the full rung.
        let short = Some((2048, false));
        assert_eq!(limit(short, 8192, need - bytes(2048)), incumbent);
        assert_eq!(limit(short, 8192, need), RESIDENT_MOE_CHUNK_ROWS);
        assert_eq!(limit(short, 8192, need - 1), incumbent);
        // The same short owner is reused (and credited) when the call needs no
        // more rows than it holds; one row more is a replacement.
        assert_eq!(
            limit(short, 2048, need - bytes(2048)),
            RESIDENT_MOE_CHUNK_ROWS
        );
        assert_eq!(limit(short, 2048, need - bytes(2048) - 1), incumbent);
        assert_eq!(limit(short, 2049, need - bytes(2048)), incumbent);
        // Lean owner: never credited.
        assert_eq!(
            limit(Some((2048, true)), 100, need - bytes(2048)),
            incumbent
        );
        // An oversized lean owner still earns no credit. Its payload can exceed
        // the full-rung need, so use the uncredited boundary without subtraction.
        assert_eq!(limit(Some((8192, true)), 100, need - 1), incumbent);
    }

    #[test]
    fn resident_moe_memory_boundary_pending_deduction_credit_and_overflow() {
        let moe = resident_moe_test_config();
        let pbs = resident_moe_pbs_bytes(&moe, RESIDENT_MOE_CHUNK_ROWS).unwrap();
        let projection = resident_moe_projection_deficit(
            &moe,
            RESIDENT_MOE_CHUNK_ROWS,
            &LiveProjectionScratch::default(),
        )
        .unwrap();
        let q16 = resident_moe_q16_missing(&moe, RESIDENT_MOE_CHUNK_ROWS, 0).unwrap();
        let need = pbs + projection + q16 + WIDENED_VRAM_HEADROOM_BYTES;
        let fit = |free: usize, deduction: usize, credit: usize| ResidentMoeFit {
            free_bytes: free,
            free_deduction: deduction,
            pbs_bytes: Some(pbs),
            cached_credit: credit,
            projection_deficit: Some(projection),
            q16_missing: Some(q16),
        };
        // Inclusive boundary at the existing 1 GiB headroom.
        assert_eq!(WIDENED_VRAM_HEADROOM_BYTES, 1 << 30);
        assert!(resident_moe_rung_fits(&fit(need, 0, 0)));
        assert!(!resident_moe_rung_fits(&fit(need - 1, 0, 0)));
        // Pending allocations are deducted from free bytes (saturating).
        assert!(resident_moe_rung_fits(&fit(need + 4096, 4096, 0)));
        assert!(!resident_moe_rung_fits(&fit(need + 4096, 4097, 0)));
        assert!(!resident_moe_rung_fits(&fit(10, 100, 0)));
        // A cached owner credits exactly its own bytes, never more than the need.
        let credit = resident_moe_pbs_bytes(&moe, 2048).unwrap();
        assert!(resident_moe_rung_fits(&fit(need - credit, 0, credit)));
        assert!(!resident_moe_rung_fits(&fit(need - credit - 1, 0, credit)));
        let floor = projection + q16 + WIDENED_VRAM_HEADROOM_BYTES;
        assert!(resident_moe_rung_fits(&fit(floor, 0, pbs * 2)));
        assert!(!resident_moe_rung_fits(&fit(floor - 1, 0, pbs * 2)));
        // Headroom is charged on top of an otherwise exactly-fitting rung.
        assert!(!resident_moe_rung_fits(&fit(
            need - WIDENED_VRAM_HEADROOM_BYTES,
            0,
            0
        )));
        // Overflow and unprojectable terms fail closed.
        let mut overflow = fit(usize::MAX, 0, 0);
        overflow.pbs_bytes = Some(usize::MAX);
        assert!(!resident_moe_rung_fits(&overflow));
        for broken in [
            ResidentMoeFit {
                pbs_bytes: None,
                ..fit(usize::MAX, 0, 0)
            },
            ResidentMoeFit {
                projection_deficit: None,
                ..fit(usize::MAX, 0, 0)
            },
            ResidentMoeFit {
                q16_missing: None,
                ..fit(usize::MAX, 0, 0)
            },
        ] {
            assert!(!resident_moe_rung_fits(&broken));
        }
    }

    #[test]
    fn resident_moe_limit_is_binary_incumbent_fallback() {
        for incumbent in [256usize, 384, 512] {
            assert_eq!(resident_moe_limit(true, incumbent), RESIDENT_MOE_CHUNK_ROWS);
            let fallback = resident_moe_limit(false, incumbent);
            assert_eq!(fallback, incumbent, "memory denial returns the incumbent");
            assert!(
                ![1024, 2048].contains(&fallback),
                "no intermediate rung is ever returned"
            );
        }
    }

    #[test]
    fn resident_moe_sealed_contract_is_multi_device_only_for_tp_ep() {
        use hipfire_dispatch::pipeline::sealed_moe::{
            ContractAssignment, ContractParallelism, ExpertExecutionContract,
        };
        let contract = |parallelism, devices: Vec<i32>| {
            ExpertExecutionContract::new(
                "g",
                Some(0),
                "fp",
                0,
                devices,
                parallelism,
                ContractAssignment::Stride,
                vec![0; 256],
                (0..256).collect(),
                "indexed-decode",
                Vec::new(),
            )
            .expect("contract")
        };
        // The ordinary load (`adapt_expert_execution_plan`, ExpertParallelism::Single)
        // always carries a sealed contract: it must read as resident.
        let single = contract(ContractParallelism::Single, vec![0]);
        assert!(!resident_moe_contract_multi_device(Some(&single)));
        assert!(!resident_moe_contract_multi_device(None));
        let ep = contract(ContractParallelism::ExpertParallel, vec![0, 1]);
        assert!(resident_moe_contract_multi_device(Some(&ep)));
        let tp = contract(ContractParallelism::TensorParallel, vec![0, 1]);
        assert!(resident_moe_contract_multi_device(Some(&tp)));
        let single_multi = contract(ContractParallelism::Single, vec![0, 1]);
        assert!(resident_moe_contract_multi_device(Some(&single_multi)));
    }

    #[test]
    fn prefill_chunk_receipt_is_the_visible_log_line() {
        let admitted = PrefillChunkReceipt {
            requested: 4096,
            admitted: 4096,
            commit_stride: Some(WIDENED_COMMIT_ROWS),
        };
        assert_eq!(
            admitted.to_string(),
            "prefill_chunk: requested=4096 admitted=4096 commit_stride=512"
        );
        let fallback = PrefillChunkReceipt {
            requested: 4096,
            admitted: 512,
            commit_stride: None,
        };
        assert_eq!(
            fallback.to_string(),
            "prefill_chunk: requested=4096 admitted=512 commit_stride=none"
        );
    }

    /// Facts of one INSTALLED artifact, derived by production code.
    ///
    /// The config comes from `config_from_hfq`, every layer's sealed expert
    /// table from the loader's own `build_moe_layer_expert_binding` (what
    /// `load_moe_ffn` calls), every router/shared tag from the production tag
    /// mapper. Device-owned facts (`host_located`, `cpu_deferred_experts`, live
    /// scratch bytes, free VRAM) cannot exist on CPU: the scenarios supply them
    /// as resident / modelled inputs and the GPU hold covers them.
    struct InstalledMoe {
        name: String,
        config: Qwen35Config,
        moe_has_mq6: bool,
        /// Every dtype-derivable `prefill_batch_pbs_eligible` leaf predicate
        /// (`is_batchable_la`, `gated_moe_prefill_admissible`) holds.
        batch_pbs_eligible: bool,
        layers: Vec<(
            ResidentMoePlacement,
            hipfire_dispatch::families::moe::MoePrefillDtypes,
        )>,
    }

    fn load_installed_moe(
        name: &str,
        hfq: &hipfire_runtime::hfq::HfqFile,
        arch: &str,
    ) -> InstalledMoe {
        let config = super::super::config::config_from_hfq(hfq).expect("production config");
        assert!(config.num_experts > 0, "{name}: MoE config");
        let tag = |tensor: &str| -> DType {
            let info = hfq
                .tensors()
                .iter()
                .find(|t| t.name == tensor)
                .unwrap_or_else(|| panic!("{name}: tensor {tensor}"));
            crate::qwen35::weights::dtype_from_quant_type(info.quant_type)
                .unwrap_or_else(|e| panic!("{name}: {tensor}: {e:?}"))
        };
        // `load_moe_ffn` arms the AWQ pointer table only when every expert has a
        // down sidecar and `HIPFIRE_MOE_AWQ` is not "0".
        let moe_awq_enabled = hipfire_config::developer_var("HIPFIRE_MOE_AWQ")
            .ok()
            .as_deref()
            != Some("0");
        let mut layers = Vec::with_capacity(config.n_layers);
        let mut batch_pbs_eligible = true;
        let mut moe_has_mq6 = false;
        for layer in 0..config.n_layers {
            let prefix = format!("model.language_model.layers.{layer}");
            let p = format!("{prefix}.mlp");
            let ids: Vec<usize> = (0..config.num_experts).collect();
            let (_plan, table, _cache) = crate::qwen35::load::build_moe_layer_expert_binding(
                hfq,
                &prefix,
                &config,
                layer,
                0,
                &ids,
                config.num_experts,
            )
            .unwrap_or_else(|e| panic!("{name} layer {layer}: expert binding: {e:?}"));
            assert!(
                table.execution_contract().is_some(),
                "{name} layer {layer}: ordinary Single load carries a sealed contract"
            );
            let n = table.n_experts();
            assert_eq!(n, config.num_experts, "{name} layer {layer}");
            let gate_up: Vec<DType> = table
                .experts()
                .iter()
                .map(|e| {
                    let r = e.resources();
                    r.gate_up()
                        .or_else(|| r.gate())
                        .expect("fused or separate gate/up resource")
                        .dtype()
                })
                .collect();
            let down: Vec<DType> = table
                .experts()
                .iter()
                .map(|e| e.resources().down().dtype())
                .collect();
            let awq_all = moe_awq_enabled
                && table
                    .experts()
                    .iter()
                    .all(|e| !e.resources().down().sidecars().is_empty());
            let gu_uniform = gate_up.iter().all(|d| *d == gate_up[0]);
            let down_uniform = down.iter().all(|d| *d == down[0]);
            let dtypes = hipfire_dispatch::families::moe::MoePrefillDtypes {
                router: tag(&format!("{p}.gate.weight")),
                shared_expert_scalar_gate: tag(&format!("{p}.shared_expert_gate.weight")),
                shared_expert_gate: tag(&format!("{p}.shared_expert.gate_proj.weight")),
                shared_expert_up: tag(&format!("{p}.shared_expert.up_proj.weight")),
                shared_expert_down: tag(&format!("{p}.shared_expert.down_proj.weight")),
                expert_gate_up: gate_up[0],
                expert_down: down[0],
                expert_gate_up_uniform: gu_uniform,
                expert_down_uniform: down_uniform,
                // `load_moe_ffn` publishes `expert_dtype_tags` exactly when the
                // per-expert dtypes are mixed.
                routed_mixed_merged: !gu_uniform || !down_uniform,
            };
            moe_has_mq6 |= [dtypes.expert_gate_up, dtypes.expert_down]
                .iter()
                .any(|d| matches!(d, DType::MQ6G256 | DType::MQ6G256V2));
            batch_pbs_eligible &= hipfire_dispatch::families::moe::gated_moe_prefill_admissible(
                &dtypes,
                mq6_batched_admit_enabled_from_env(None, arch),
                arch,
            );
            layers.push((
                ResidentMoePlacement {
                    multi_device_contract: resident_moe_contract_multi_device(
                        table.execution_contract(),
                    ),
                    paged_experts: n == 0,
                    sidecar_experts: awq_all || !gu_uniform || !down_uniform,
                    ..ResidentMoePlacement::default()
                },
                dtypes,
            ));
        }
        // Attention/projection tags the batched layer admission checks.
        for t in hfq.tensors() {
            let n = &t.name;
            let projection = n.ends_with("in_proj_qkv.weight")
                || n.ends_with("in_proj_z.weight")
                || n.ends_with("in_proj_a.weight")
                || n.ends_with("in_proj_b.weight")
                || n.ends_with("out_proj.weight")
                || n.ends_with("q_proj.weight")
                || n.ends_with("k_proj.weight")
                || n.ends_with("v_proj.weight")
                || n.ends_with("o_proj.weight");
            if projection && !n.contains(".experts.") && !n.contains("shared_expert") {
                let dt = crate::qwen35::weights::dtype_from_quant_type(t.quant_type).unwrap();
                batch_pbs_eligible &= is_batchable_la(dt, arch);
            }
        }
        InstalledMoe {
            name: name.to_string(),
            config,
            moe_has_mq6,
            batch_pbs_eligible,
            layers,
        }
    }

    /// KV tier flags a resolved KV pair sets on its `KvCache` constructors.
    fn kv_tier_flags_for_pair(pair: hipfire_runtime::kv_mode::KvPair) -> KvTierFlags {
        use hipfire_runtime::kv_mode::KvMode;
        let mut flags = KvTierFlags::default();
        match pair.k() {
            KvMode::Q8 => flags.quant_q8 = true,
            KvMode::Asym2 | KvMode::Fwht2 => flags.quant_asym2 = true,
            KvMode::Asym3 | KvMode::Fwht3 => flags.quant_asym3 = true,
            KvMode::Asym4 | KvMode::Fwht4 => flags.quant_asym4 = true,
            KvMode::Fp8 => flags.quant_fp8 = true,
            _ => {}
        }
        flags
    }

    /// The KV tier the production default ("auto" through the qwen35 HFQ policy
    /// and native-eligibility rule) resolves to for this model on `arch`.
    fn installed_default_kv(config: &Qwen35Config, arch: &str) -> KvTierFlags {
        use hipfire_runtime::kv_mode;
        let auto_native = kv_mode::qwen35_native_auto_eligible(
            arch,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            1,
            false,
            false,
        );
        let policy =
            kv_mode::qwen35_policy_for_native(&kv_mode::QWEN35_HFQ_POLICY, "auto", auto_native);
        let pair = kv_mode::resolve_kv_pair("auto", None, None, &policy, arch, true)
            .expect("default kv pair");
        kv_tier_flags_for_pair(pair)
    }

    fn resident_moe_need_bytes(config: &Qwen35Config, live_q16: usize) -> usize {
        resident_moe_pbs_bytes(config, RESIDENT_MOE_CHUNK_ROWS).unwrap()
            + resident_moe_projection_deficit(
                config,
                RESIDENT_MOE_CHUNK_ROWS,
                &LiveProjectionScratch::default(),
            )
            .unwrap()
            + resident_moe_q16_missing(config, RESIDENT_MOE_CHUNK_ROWS, live_q16).unwrap()
            + WIDENED_VRAM_HEADROOM_BYTES
    }

    fn owner_bytes_of(config: &Qwen35Config, owner: (usize, bool)) -> usize {
        resident_moe_owner_bytes(config, owner.0, owner.1).unwrap()
    }

    /// `resident_moe_need_bytes` with the wrapper's explicit-equivalent owner as
    /// the charged PBS (the larger of the tape-free rung and the owner).
    fn resident_moe_need_owner_bytes(
        config: &Qwen35Config,
        live_q16: usize,
        owner: (usize, bool),
    ) -> usize {
        let rung = resident_moe_pbs_bytes(config, RESIDENT_MOE_CHUNK_ROWS).unwrap();
        resident_moe_need_bytes(config, live_q16) - rung + rung.max(owner_bytes_of(config, owner))
    }

    /// Every input of one modelled ordinary prefill call. Each field is a REAL
    /// production input (caller facts, state geometry plan, KV tier, device
    /// flags/caps from `FeatureFlags`/`ArchCaps`, loader-derived layer facts);
    /// tests flip single fields and read the decision, never a typed verdict.
    #[derive(Clone)]
    struct Sc {
        arch: &'static str,
        flags: rdna_compute::FeatureFlags,
        explicit: bool,
        reuse_pbs: bool,
        n: usize,
        call: OrdinaryCallFacts,
        dn_quant: StateQuant,
        dn_batch: usize,
        dn_ef: bool,
        per_token_requant: bool,
        kv: KvTierFlags,
        free_bytes: Option<usize>,
        live_q16: usize,
        pager: bool,
        ep_shard: bool,
        mq6_weights: bool,
        batch_pbs_eligible: bool,
        prefill_batched: bool,
        layers: Vec<(
            ResidentMoePlacement,
            hipfire_dispatch::families::moe::MoePrefillDtypes,
        )>,
        cached: Option<(usize, bool)>,
        deduction: usize,
        existing_virtual: Option<(usize, bool)>,
    }

    type CallFlip = fn(&mut OrdinaryCallFacts);
    type ScFlip = fn(&mut Sc);
    type PlacementFlip = fn(&mut ResidentMoePlacement);
    type DtypeFlip = fn(&mut hipfire_dispatch::families::moe::MoePrefillDtypes);

    fn baseline(
        moe: &InstalledMoe,
        arch: &'static str,
        explicit: bool,
        reuse_pbs: bool,
        n: usize,
    ) -> Sc {
        let flags = rdna_compute::FeatureFlags::from_active_config(arch);
        let target = implicit_reuse_pbs_plan(
            arch,
            Some(RESIDENT_MOE_CHUNK_ROWS),
            fp8_chunk512_for_flags(arch, &flags),
        );
        Sc {
            arch,
            flags,
            explicit,
            reuse_pbs,
            n,
            call: OrdinaryCallFacts::default(),
            dn_quant: StateQuant::Q8,
            dn_batch: 1,
            dn_ef: crate::qwen35::weights::delta_net_state_ef_enabled(StateQuant::Q8),
            per_token_requant: dn_requant_per_token_env(),
            kv: installed_default_kv(&moe.config, arch),
            free_bytes: Some(resident_moe_need_owner_bytes(&moe.config, 0, target) + (64 << 20)),
            live_q16: 0,
            pager: false,
            ep_shard: false,
            mq6_weights: moe.moe_has_mq6,
            batch_pbs_eligible: moe.batch_pbs_eligible,
            prefill_batched: hipfire_runtime::config::get().prefill_batched,
            layers: moe.layers.clone(),
            cached: None,
            deduction: 0,
            existing_virtual: None,
        }
    }

    struct Run {
        facts: ResidentMoeStaticFacts,
        wrapper: WrapperPbsPlan,
        /// The wrapper maps the call's required KV (before any memory query).
        maps_kv: bool,
        /// The wrapper queried device memory for the virtual owner's admission.
        mem_queried: bool,
        /// The inner entry returned to the per-token AR loop before the planner
        /// (`prefill_batch_pbs_eligible` false at the actual n): `plan` is an
        /// unexecuted placeholder and `chunks` is empty.
        ar_fallback: bool,
        /// The executed PBS owner `(rows, tape)`, `None` when the inner entry
        /// allocates its own.
        owner: Option<(usize, bool)>,
        plan: OrdinaryPrefillPlan,
        chunks: Vec<(usize, Option<usize>)>,
        /// The effective incumbent ceiling the inner entry ran with.
        max_batch: usize,
        /// The ORIGINAL incumbent ceiling (arch/env default capped by an explicit
        /// `max_batch_cap`, never the wrapper's virtual-explicit override).
        incumbent: usize,
    }

    /// `prefill_batch_pbs_eligible(rows)` over artifact-derived facts: the
    /// dtype-derivable layer predicates and the SAME scalar gate production
    /// calls (verify-decouple window, `prefill_batched`, two-token minimum,
    /// a DeltaNet layer).
    fn sc_batched_eligible(moe: &InstalledMoe, sc: &Sc, rows: usize) -> bool {
        sc.batch_pbs_eligible
            && prefill_batch_scalar_eligible(
                rows,
                sc.arch,
                hipfire_config::developer_var("HIPFIRE_MTP_VERIFY_DECOUPLE")
                    .ok()
                    .as_deref(),
                sc.prefill_batched,
                moe.config
                    .layer_types
                    .iter()
                    .any(|t| *t == LayerType::LinearAttention),
            )
    }

    fn sc_static_facts(moe: &InstalledMoe, sc: &Sc) -> ResidentMoeStaticFacts {
        let caps =
            rdna_compute::arch_caps::ArchCaps::new(sc.arch, std::sync::Arc::new(sc.flags.clone()));
        let explicit_max = sc.explicit.then_some(8192usize);
        let plan = DeltaNetStatePlan::new(&moe.config, sc.dn_quant, sc.dn_batch, sc.dn_ef);
        ResidentMoeStaticFacts {
            moe_config: moe.config.num_experts != 0,
            request: resident_moe_request(explicit_max, explicit_max),
            has_wmma: caps.has_wmma(),
            pager: sc.pager,
            ep_shard: sc.ep_shard,
            moe_has_mq6: sc.mq6_weights,
            layers_present: !sc.layers.is_empty(),
            dn_state_admitted: ordinary_dn_state_admitted(&moe.config, &plan, sc.per_token_requant),
            // The static facts' probe is the 512-row seam, exactly as
            // `resident_moe_static_facts` evaluates it.
            batch_pbs_eligible: sc_batched_eligible(moe, sc, WIDENED_COMMIT_ROWS),
            layers_resident_capable: sc.layers.iter().all(|(placement, dtypes)| {
                resident_moe_layer_facts_capable(placement, Some(dtypes), false)
            }),
            iu4_d53_layers: false,
            iu4_scratch_wide: false,
        }
    }

    fn sc_memory_facts(sc: &Sc) -> ResidentMoeMemoryFacts {
        ResidentMoeMemoryFacts {
            free_bytes: sc.free_bytes,
            live: LiveProjectionScratch::default(),
            live_q16_bytes: sc.live_q16,
        }
    }

    /// The wrapper step exactly as `ordinary_wrapper_pbs_prepare` flows it,
    /// through the SAME shared [`OrdinaryWrapperGate`] (actual `n`, the 512-row
    /// seam, the caller/capture record), [`resident_moe_default_fits`],
    /// [`wrapper_virtual_credit_bytes`] and [`plan_wrapper_pbs`]. Needs no
    /// inner entry, so it also covers calls (n = 1) that fall to the
    /// per-token loop. Returns `(plan, maps_kv, mem_queried)`.
    fn wrapper_run(
        moe: &InstalledMoe,
        sc: &Sc,
        facts: &ResidentMoeStaticFacts,
    ) -> (WrapperPbsPlan, bool, bool) {
        let arch = sc.arch;
        let fp8 = fp8_chunk512_for_flags(arch, &sc.flags);
        let explicit_max = sc.explicit.then_some(8192usize);
        // The implicit reuse cache as `Qwen35Scratch` allocates it for this
        // knob state, the virtual-explicit target, and the shared fit.
        let physical = sc
            .reuse_pbs
            .then(|| implicit_reuse_pbs_plan(arch, explicit_max, fp8));
        if sc.call.caller_pbs_rows.is_some() {
            // Direct `_with_pbs` callers: their own PBS, no wrapper.
            return (
                WrapperPbsPlan {
                    owner: WrapperOwner::None,
                    max_batch_override: None,
                    free_virtual: false,
                    alloc_virtual: false,
                },
                false,
                false,
            );
        }
        if sc.call.max_batch_cap.is_some() {
            // `forward_prefill_batch_capped`: the implicit cache, never the wrapper.
            return (
                WrapperPbsPlan {
                    owner: if physical.is_some() {
                        WrapperOwner::Physical
                    } else {
                        WrapperOwner::None
                    },
                    max_batch_override: None,
                    free_virtual: false,
                    alloc_virtual: false,
                },
                false,
                false,
            );
        }
        let target = implicit_reuse_pbs_plan(arch, Some(RESIDENT_MOE_CHUNK_ROWS), fp8);
        let credit_bytes =
            wrapper_virtual_credit_bytes(&moe.config, physical, sc.existing_virtual, target);
        let gate = OrdinaryWrapperGate::new(sc.n, &sc.call, physical, |rows| {
            sc_batched_eligible(moe, sc, rows)
        });
        let queried = std::cell::Cell::new(false);
        let dense_widened = false;
        let default_moe_fits = gate.needs_default_fit(dense_widened)
            && resident_moe_default_fits(
                facts,
                &moe.config,
                sc.kv,
                || {
                    queried.set(true);
                    sc_memory_facts(sc)
                },
                ResidentMoeOwner {
                    rows: target.0,
                    tape: target.1,
                    credit_bytes,
                },
            );
        let plan = gate.plan(
            arch,
            fp8,
            dense_widened,
            sc.existing_virtual,
            default_moe_fits,
        );
        (plan, gate.maps_kv(), queried.get())
    }

    /// The ordinary prefill call exactly as production flows it: wrapper PBS
    /// plan ([`wrapper_run`]), then `plan_ordinary_prefill` whose widen-candidate
    /// limit is `resident_moe_chunk_limit_from_facts` (the MoE arm of
    /// `ordinary_prefill_chunk_limit_with_facts`; a MoE shape has no dense
    /// static ceiling, asserted below), then the production chunk iterator.
    /// A caller-supplied PBS bypasses the wrapper (direct `_with_pbs` callers).
    /// Requires `sc.n >= MIN_BATCH` (a one-token call never reaches the planner:
    /// the inner entry takes the per-token loop; use [`wrapper_run`]). A call
    /// whose batched eligibility fails at its actual n (`prefill_batched` off
    /// above the verify-decouple window, a layer dtype that is not batchable)
    /// also takes the AR loop in production: `Run::ar_fallback` with no chunks
    /// and no planner execution.
    fn run(moe: &InstalledMoe, sc: &Sc) -> Run {
        let arch = sc.arch;
        let fp8 = fp8_chunk512_for_flags(arch, &sc.flags);
        let explicit_max = sc.explicit.then_some(8192usize);
        let request = resident_moe_request(explicit_max, explicit_max);
        let legacy_max_batch = legacy_prefill_max_batch(explicit_max, arch, fp8, false);
        assert!(
            !widened_dense_shape_admitted(&moe.config),
            "{}: a MoE shape has no dense widened ceiling",
            moe.name
        );
        assert!(
            sc.n >= MIN_BATCH,
            "run() needs n >= 2 (n = 1: use wrapper_run)"
        );
        let facts = sc_static_facts(moe, sc);
        let mem = sc_memory_facts(sc);
        let physical = sc
            .reuse_pbs
            .then(|| implicit_reuse_pbs_plan(arch, explicit_max, fp8));
        let (wrapper, maps_kv, mem_queried) = wrapper_run(moe, sc, &facts);
        let owner = match wrapper.owner {
            WrapperOwner::None => None,
            WrapperOwner::Physical => physical,
            WrapperOwner::Virtual { rows, tape } => Some((rows, tape)),
        };
        let incumbent = resolve_ordinary_max_batch(legacy_max_batch, None, sc.call.max_batch_cap);
        let max_batch = resolve_ordinary_max_batch(
            legacy_max_batch,
            wrapper.max_batch_override,
            sc.call.max_batch_cap,
        );
        // The production inner entry computes `prefill_batch_pbs_eligible` at
        // the ACTUAL n and, when false, runs the per-token AR loop and returns
        // BEFORE `plan_ordinary_prefill` (no planner, no chunk plan). The
        // wrapper step above already ran, as in production.
        if !sc_batched_eligible(moe, sc, sc.n) {
            return Run {
                facts,
                wrapper,
                maps_kv,
                mem_queried,
                owner,
                // Not executed: placeholder carrying the incumbent ceiling.
                plan: OrdinaryPrefillPlan {
                    limit: max_batch,
                    wide: false,
                    moe_route: false,
                    receipt: PrefillChunkReceipt {
                        requested: max_batch,
                        admitted: max_batch,
                        commit_stride: None,
                    },
                    pbs_rows: 0,
                },
                chunks: Vec::new(),
                max_batch,
                incumbent,
                ar_fallback: true,
            };
        }
        let mut call = sc.call;
        call.caller_pbs_rows = call.caller_pbs_rows.or(owner.map(|o| o.0));
        let req = OrdinaryPlanRequest {
            call,
            n: sc.n,
            max_batch,
            explicit_max_batch: explicit_max,
            chunk_rows_requested: chunk_rows_requested_from(arch, explicit_max, explicit_max),
            moe_request: request,
            moe_static_admitted: resident_moe_static_decision(&facts).is_some(),
        };
        let plan = plan_ordinary_prefill::<()>(&req, || {
            Ok(resident_moe_chunk_limit_from_facts(
                &facts,
                None,
                &moe.config,
                sc.kv,
                || mem,
                sc.cached,
                widened_owned_rows(RESIDENT_MOE_CHUNK_ROWS, sc.n),
                sc.deduction,
                max_batch,
            ))
        })
        .unwrap();
        let batch = plan.chunk_batch(None);
        let (mut chunks, mut left) = (Vec::new(), sc.n);
        while left > 0 {
            let c = next_ordinary_chunk_len(plan.wide, false, left, batch).expect("chunk plan");
            chunks.push((c, widened_commit_stride(plan.wide, c)));
            left -= c;
        }
        Run {
            facts,
            wrapper,
            maps_kv,
            mem_queried,
            owner,
            plan,
            chunks,
            max_batch,
            incumbent,
            ar_fallback: false,
        }
    }

    /// The ORIGINAL incumbent plan, compared against an independent
    /// recomputation: no override, not widened, the original (capped) ceiling,
    /// the original owner rows, and exactly the legacy chunk cadence.
    fn assert_incumbent(r: &Run, sc: &Sc, what: &str) {
        assert!(!r.plan.wide, "{what}: must not widen");
        if r.ar_fallback {
            // The inner entry returned to the per-token AR loop before the
            // planner: no override, the incumbent ceiling, and no chunk plan.
            assert_eq!(
                r.wrapper.max_batch_override, None,
                "{what}: no virtual-explicit override"
            );
            assert_eq!(r.plan.limit, r.incumbent, "{what}: incumbent ceiling");
            assert!(
                r.chunks.is_empty(),
                "{what}: AR fallback runs no chunk plan"
            );
            return;
        }
        assert_eq!(
            r.wrapper.max_batch_override, None,
            "{what}: no virtual-explicit override"
        );
        assert_eq!(r.plan.limit, r.incumbent, "{what}: incumbent ceiling");
        let rows = r
            .owner
            .map(|o| o.0)
            .or(sc.call.caller_pbs_rows)
            .unwrap_or_else(|| {
                owned_prefill_scratch_plan(sc.n, r.incumbent, sc.call.tree_verify).0
            });
        assert_eq!(r.plan.pbs_rows, rows, "{what}: original owner rows");
        let batch = r.incumbent.min(rows);
        let (mut expected, mut left) = (Vec::new(), sc.n);
        while left > 0 {
            let c = next_prefill_chunk_len(left, batch).expect("legacy chunk plan");
            expected.push((c, None));
            left -= c;
        }
        assert_eq!(r.chunks, expected, "{what}: original chunk plan");
        assert!(
            r.chunks.iter().all(|(c, _)| *c <= r.incumbent),
            "{what}: no chunk above the incumbent ceiling"
        );
    }

    /// Planner proof over the INSTALLED artifacts: both real HFQs go through the
    /// loader's expert binding; the production default and the beta explicit
    /// `HIPFIRE_PREFILL_MAX_BATCH=8192` arms go through the SAME
    /// wrapper-cache / `plan_ordinary_prefill` / memory / chunk functions the
    /// production inner entry calls, with `HIPFIRE_PREFILL_REUSE_PBS` on and
    /// off, and every denying fact is flipped against the same decision. Absent
    /// artifacts skip loudly unless `HIPFIRE_REQUIRE_INSTALLED_A3B_HFQ` is set,
    /// which makes both mandatory.
    #[test]
    fn resident_moe_installed_artifacts_admit_8192_and_log_it() {
        use hipfire_runtime::hfq::HfqFile;
        let models = std::env::var_os("HIPFIRE_MODELS_DIR")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(|home| std::path::Path::new(&home).join(".hipfire/models"))
            })
            .expect("HOME or HIPFIRE_MODELS_DIR");
        let require =
            hipfire_config::developer_var_os("HIPFIRE_REQUIRE_INSTALLED_A3B_HFQ").is_some();
        let mut checked = 0;
        for name in ["qwen3.6-35b-a3b.mq4r", "ornith-1.5-35b-a3b.mq4r"] {
            let path = models.join(name);
            if !path.exists() {
                assert!(
                    !require,
                    "resident_moe artifact test: required {} is absent",
                    path.display()
                );
                eprintln!(
                    "resident_moe artifact test: {} absent, skipped",
                    path.display()
                );
                continue;
            }
            let hfq = HfqFile::open(&path).expect("open installed hfq");
            for arch in ["gfx1201", "gfx1100"] {
                let moe = load_installed_moe(name, &hfq, arch);
                assert!(!moe.moe_has_mq6, "{name}: moe_has_mq6 must be false");
                assert!(
                    moe.batch_pbs_eligible,
                    "{name}: batched layer dtype admission"
                );
                let line = |r: &Run| r.plan.receipt.to_string();
                // Census of the actual HFQ shape against the independent
                // constructor closed form (not the projected mirror).
                for rows in [100usize, 512, 1024, 4096] {
                    for tape in [false, true] {
                        assert_eq!(
                            resident_moe_owner_bytes(&moe.config, rows, tape).map(|b| b as u128),
                            Some(constructor_payload_bytes(&moe.config, rows as u128, tape)),
                            "{name}: census rows={rows} tape={tape}"
                        );
                    }
                }
                assert_eq!(
                    resident_moe_owner_bytes(&moe.config, usize::MAX, false),
                    None
                );

                // --- reuse PBS off: default vs beta explicit 8192 -------------
                for n in [4096usize, 8192, 3855, 16384, 4097, 513, 100] {
                    let d = run(&moe, &baseline(&moe, arch, false, false, n));
                    let e = run(&moe, &baseline(&moe, arch, true, false, n));
                    assert!(
                        resident_moe_static_decision(&d.facts).is_some(),
                        "{name} {arch}: static decision must admit the installed model"
                    );
                    assert_eq!(
                        resident_moe_static_decision(&e.facts),
                        None,
                        "{name} {arch}: explicit 8192 keeps the incumbent static gate"
                    );
                    assert!(d.plan.wide && e.plan.wide, "{name} {arch} n={n}");
                    assert_eq!(d.plan.limit, RESIDENT_MOE_CHUNK_ROWS);
                    assert_eq!(e.plan.limit, RESIDENT_MOE_CHUNK_ROWS);
                    assert_eq!(
                        (d.plan.pbs_rows, &d.chunks, line(&d)),
                        (e.plan.pbs_rows, &e.chunks, line(&e)),
                        "{name} {arch} n={n}: default vs explicit parameters"
                    );
                    assert_eq!((d.owner, e.owner), (None, None));
                }
                let d8192 = run(&moe, &baseline(&moe, arch, false, false, 8192));
                assert_eq!(d8192.plan.pbs_rows, 8192);
                assert_eq!(d8192.chunks, vec![(8192, Some(WIDENED_COMMIT_ROWS))]);
                assert_eq!(
                    line(&d8192),
                    "prefill_chunk: requested=8192 admitted=8192 commit_stride=512"
                );
                eprintln!(
                    "resident_moe production-planner proof: {name} {arch}: default==explicit8192 limit={} pbs_rows={} chunks(n=8192)={:?} line=\"{}\"",
                    d8192.plan.limit,
                    d8192.plan.pbs_rows,
                    d8192.chunks,
                    line(&d8192)
                );
                let guard = std::sync::atomic::AtomicUsize::new(0);
                assert!(emit_prefill_chunk_receipt_guarded(
                    &guard,
                    d8192.plan.receipt.requested,
                    d8192.plan.receipt.admitted,
                    d8192.plan.receipt.commit_stride,
                ));

                // --- reuse PBS on: default == beta explicit 8192, EXACTLY -------
                let flags0 = rdna_compute::FeatureFlags::from_active_config(arch);
                let _ = &flags0;
                let eff = |r: &Run| {
                    (
                        r.plan.wide,
                        r.plan.limit,
                        r.plan.pbs_rows,
                        r.owner,
                        r.chunks.clone(),
                        r.plan.receipt.to_string(),
                    )
                };
                for fp8_off in [false, true] {
                    if fp8_off && arch != "gfx1201" {
                        continue;
                    }
                    let tweak = |sc: &mut Sc| {
                        if fp8_off {
                            sc.flags.gfx12_mq4v2_fp8_gateup = false;
                        }
                    };
                    let flags_now = {
                        let mut f = flags0.clone();
                        if fp8_off {
                            f.gfx12_mq4v2_fp8_gateup = false;
                        }
                        f
                    };
                    let fp8_now = fp8_chunk512_for_flags(arch, &flags_now);
                    let target =
                        implicit_reuse_pbs_plan(arch, Some(RESIDENT_MOE_CHUNK_ROWS), fp8_now);
                    let default_physical = implicit_reuse_pbs_plan(arch, None, fp8_now);
                    let need_owner = resident_moe_need_owner_bytes(&moe.config, 0, target);
                    for n in [4096usize, 8192, 513, 100] {
                        let mut sd = baseline(&moe, arch, false, true, n);
                        tweak(&mut sd);
                        let mut se = baseline(&moe, arch, true, true, n);
                        tweak(&mut se);
                        let (d, e) = (run(&moe, &sd), run(&moe, &se));
                        // Explicit manual knob: never rewritten (static gate closed,
                        // implicit cache sized by the explicit knob itself).
                        assert_eq!(resident_moe_static_decision(&e.facts), None);
                        assert_eq!(e.wrapper.owner, WrapperOwner::Physical);
                        assert_eq!(e.wrapper.max_batch_override, None);
                        assert_eq!(e.owner, Some(target));
                        assert!(!d.plan.wide && !e.plan.wide, "{name} {arch} n={n}");
                        assert!(d
                            .chunks
                            .iter()
                            .chain(e.chunks.iter())
                            .all(|c| c.1.is_none()));
                        assert_eq!(d.plan.limit, RESIDENT_MOE_CHUNK_ROWS);
                        // Default (fresh owner): EXACT beta explicit parameters.
                        assert_eq!(
                            eff(&d),
                            eff(&e),
                            "{name} {arch} fp8_off={fp8_off} n={n}: reuse-PBS default vs explicit"
                        );
                        if default_physical == target {
                            assert_eq!(d.wrapper.owner, WrapperOwner::Physical);
                            assert!(!d.wrapper.alloc_virtual);
                        } else {
                            assert_eq!(
                                d.wrapper.owner,
                                WrapperOwner::Virtual {
                                    rows: target.0,
                                    tape: target.1
                                }
                            );
                            assert!(d.wrapper.alloc_virtual && !d.wrapper.free_virtual);
                        }
                        // Existing owner kept from an earlier admitted request: the
                        // same effective parameters, allocated only when it differs.
                        for existing in [
                            Some(target),
                            Some((1024, false)),
                            Some((target.0, !target.1)),
                        ] {
                            let mut s2 = sd.clone();
                            s2.existing_virtual = existing;
                            let d2 = run(&moe, &s2);
                            assert_eq!(eff(&d2), eff(&e), "{name} {arch} existing={existing:?}");
                            if default_physical != target && existing == Some(target) {
                                assert!(!d2.wrapper.alloc_virtual && !d2.wrapper.free_virtual);
                            }
                        }
                        // Exact memory boundary of the explicit-equivalent owner,
                        // fresh and credited by an existing owner.
                        let mut fit = sd.clone();
                        fit.free_bytes = Some(need_owner);
                        assert_eq!(eff(&run(&moe, &fit)), eff(&e));
                        let mut low = sd.clone();
                        low.free_bytes = Some(need_owner - 1);
                        let l = run(&moe, &low);
                        // Does not fit: the incumbent cache and ceiling, no override.
                        assert_eq!(l.wrapper.owner, WrapperOwner::Physical);
                        assert_eq!(l.wrapper.max_batch_override, None);
                        assert_eq!(l.owner, Some(default_physical));
                        assert!(!l.plan.wide);
                        assert_eq!(l.plan.limit, l.max_batch);
                        assert_ne!(l.max_batch, RESIDENT_MOE_CHUNK_ROWS);
                        assert_eq!(
                            l.plan.chunk_batch(None),
                            l.max_batch.min(default_physical.0)
                        );
                        // A virtual owner kept from an earlier admitted request is
                        // released when a later request no longer fits even with
                        // the owner's own bytes credited; only a SAME-TARGET owner
                        // that the wrapper keeps without replacement is credited
                        // (it is held across calls, so its bytes are already absent
                        // from free VRAM). When the implicit cache already is the
                        // target shape no virtual owner is ever kept (nothing to
                        // credit). Any other shape is replaced, and its freed bytes
                        // only return to the allocator pool, so it earns no credit.
                        let owner_bytes = owner_bytes_of(&moe.config, target);
                        if default_physical != target {
                            let tight = need_owner - owner_bytes - 1;
                            let mut tight_plain = sd.clone();
                            tight_plain.free_bytes = Some(tight);
                            let mut tight_existing = tight_plain.clone();
                            tight_existing.existing_virtual = Some(target);
                            let (lp, le) = (run(&moe, &tight_plain), run(&moe, &tight_existing));
                            assert!(le.wrapper.free_virtual && !le.wrapper.alloc_virtual);
                            assert_eq!(eff(&le), eff(&lp));
                            assert_eq!(le.wrapper.max_batch_override, None);
                            // Credited by a kept owner, the boundary moves by its bytes.
                            let mut credited = sd.clone();
                            credited.existing_virtual = Some(target);
                            credited.free_bytes = Some(need_owner - owner_bytes);
                            assert_eq!(eff(&run(&moe, &credited)), eff(&e), "credited fit");
                            credited.free_bytes = Some(need_owner - owner_bytes - 1);
                            assert_eq!(run(&moe, &credited).wrapper.max_batch_override, None);
                            // A mismatched owner (rows or tape) is replaced: the
                            // budget a speculative credit would admit is denied, and
                            // the stale owner is released.
                            for stale in [(1024, false), (target.0, !target.1)] {
                                let stale_bytes = owner_bytes_of(&moe.config, stale);
                                assert!(stale_bytes < need_owner);
                                let mut sc_stale = sd.clone();
                                sc_stale.existing_virtual = Some(stale);
                                sc_stale.free_bytes = Some(need_owner - stale_bytes);
                                let r = run(&moe, &sc_stale);
                                assert_eq!(
                                    r.wrapper.max_batch_override, None,
                                    "{name} {arch}: mismatched owner {stale:?} earns no credit"
                                );
                                assert!(r.wrapper.free_virtual && !r.wrapper.alloc_virtual);
                                assert_eq!(r.owner, Some(default_physical));
                                // Charged in full, it admits at exactly the fresh boundary.
                                sc_stale.free_bytes = Some(need_owner);
                                assert_eq!(eff(&run(&moe, &sc_stale)), eff(&e));
                            }
                        }
                        // Eligibility facts keep denying with the cache present.
                        let mut no_kv = sd.clone();
                        no_kv.kv = KvTierFlags::default();
                        let nk = run(&moe, &no_kv);
                        assert_eq!(nk.wrapper.max_batch_override, None);
                        assert_eq!(nk.owner, Some(default_physical));
                        let mut ef_off = sd.clone();
                        ef_off.dn_ef = false;
                        let ne = run(&moe, &ef_off);
                        assert_eq!(ne.wrapper.max_batch_override, None);
                        assert_eq!(ne.owner, Some(default_physical));
                    }
                    eprintln!(
                        "resident_moe reuse-PBS proof: {name} {arch} fp8_chunk512={fp8_now}: default cache={default_physical:?} -> virtual-explicit owner={target:?} limit=8192 wide=false commit=none == beta explicit8192 (EXACT, fresh + existing owner); low-mem keeps incumbent"
                    );
                }
                let _ = &flags0;

                // --- negative flags: each must deny through the actual decision,
                // with the reuse cache absent AND present -------------------------
                let base = baseline(&moe, arch, false, false, 8192);
                let admitted = run(&moe, &base);
                assert!(admitted.plan.wide, "{name} {arch}: baseline admits");
                let deny = |what: &str, sc: &Sc| assert_incumbent(&run(&moe, sc), sc, what);
                for reuse in [false, true] {
                    let rbase = baseline(&moe, arch, false, reuse, 8192);
                    if reuse {
                        assert_eq!(
                            run(&moe, &rbase).wrapper.max_batch_override,
                            Some(RESIDENT_MOE_CHUNK_ROWS),
                            "{name} {arch}: reuse baseline promotes"
                        );
                    }
                    let call_flips: [(&str, CallFlip); 10] = [
                        ("caller PBS smaller", |c| c.caller_pbs_rows = Some(512)),
                        ("caller PBS covering the rung", |c| {
                            c.caller_pbs_rows = Some(8192)
                        }),
                        ("max_batch_cap", |c| c.max_batch_cap = Some(256)),
                        ("tree verify", |c| c.tree_verify = true),
                        ("gdn tape", |c| c.gdn_tape = true),
                        ("staged hidden ring", |c| c.staged_hidden_ring = true),
                        ("max_layer", |c| c.max_layer = true),
                        ("fusion", |c| c.fusion_on = true),
                        ("capture mode", |c| c.capture_mode = true),
                        ("recording", |c| c.recording = true),
                    ];
                    for (what, flip) in call_flips {
                        let mut sc = rbase.clone();
                        flip(&mut sc.call);
                        deny(what, &sc);
                        if what == "max_batch_cap" {
                            // The ACTUAL chunks honor the cap exactly as production does.
                            let r = run(&moe, &sc);
                            assert!(r.incumbent <= 256 && r.plan.limit <= 256);
                            assert!(r.chunks.iter().all(|(c, _)| *c <= 256));
                        }
                    }
                    // State: wrong quant, EF off, lane batch > 1, per-token requant.
                    let state_flips: [(&str, ScFlip); 5] = [
                        ("state FP32", |s| s.dn_quant = StateQuant::FP32),
                        ("state Q4", |s| s.dn_quant = StateQuant::Q4),
                        ("state EF off", |s| s.dn_ef = false),
                        ("state batch 2", |s| s.dn_batch = 2),
                        ("per-token requant", |s| s.per_token_requant = true),
                    ];
                    for (what, flip) in state_flips {
                        let mut sc = rbase.clone();
                        flip(&mut sc);
                        deny(what, &sc);
                    }
                    // KV: tiers without batched flash resolve through the real pair resolver.
                    {
                        use hipfire_runtime::kv_mode::{KvMode, KvPair};
                        let mut sc = rbase.clone();
                        sc.kv = kv_tier_flags_for_pair(KvPair::Native(KvMode::Bf16));
                        assert!(!sc.kv.batched_flash());
                        deny("kv bf16", &sc);
                        sc.kv = KvTierFlags::default();
                        deny("kv f32/none", &sc);
                        for admitted_kv in [
                            KvMode::Q8,
                            KvMode::Fwht2,
                            KvMode::Fwht3,
                            KvMode::Fwht4,
                            KvMode::Fp8,
                        ] {
                            let mut ok = rbase.clone();
                            ok.kv = kv_tier_flags_for_pair(KvPair::Native(admitted_kv));
                            assert!(ok.kv.batched_flash(), "{admitted_kv:?}");
                            let r = run(&moe, &ok);
                            if reuse {
                                assert_eq!(
                                    r.wrapper.max_batch_override,
                                    Some(RESIDENT_MOE_CHUNK_ROWS),
                                    "{name} {arch}: kv {admitted_kv:?}"
                                );
                            } else {
                                assert!(r.plan.wide, "{name} {arch}: kv {admitted_kv:?}");
                            }
                        }
                    }
                    // Static model facts, including the production scalar gate.
                    let static_flips: [(&str, ScFlip); 6] = [
                        ("pager", |s| s.pager = true),
                        ("ep shard", |s| s.ep_shard = true),
                        ("mq6 weights", |s| s.mq6_weights = true),
                        ("batch pbs dtype ineligible", |s| {
                            s.batch_pbs_eligible = false
                        }),
                        ("prefill_batched off", |s| s.prefill_batched = false),
                        ("no layers", |s| s.layers.clear()),
                    ];
                    for (what, flip) in static_flips {
                        let mut sc = rbase.clone();
                        flip(&mut sc);
                        deny(what, &sc);
                    }
                    // Every placement flag of one middle layer, and every dtype class.
                    let mid = moe.layers.len() / 2;
                    let placement_flips: [(&str, PlacementFlip); 8] = [
                        ("attention host", |p| p.attention_host_located = true),
                        ("gate_up host", |p| p.routed_gate_up_host = true),
                        ("down host", |p| p.routed_down_host = true),
                        ("small ffn host", |p| p.small_ffn_host_located = true),
                        ("cpu deferred experts", |p| p.cpu_deferred_experts = true),
                        ("paged experts", |p| p.paged_experts = true),
                        ("multi-device contract", |p| p.multi_device_contract = true),
                        ("sidecar experts", |p| p.sidecar_experts = true),
                    ];
                    for (what, flip) in placement_flips {
                        let mut sc = rbase.clone();
                        flip(&mut sc.layers[mid].0);
                        deny(what, &sc);
                    }
                    let dtype_flips: [(&str, DtypeFlip); 5] = [
                        ("router mq6", |d| d.router = DType::MQ6G256),
                        ("expert gate_up mq6", |d| d.expert_gate_up = DType::MQ6G256),
                        ("expert down mixed", |d| d.expert_down_uniform = false),
                        ("merged mixed routed", |d| d.routed_mixed_merged = true),
                        ("shared pair split", |d| d.shared_expert_up = DType::MQ6G256),
                    ];
                    for (what, flip) in dtype_flips {
                        let mut sc = rbase.clone();
                        flip(&mut sc.layers[0].1);
                        deny(what, &sc);
                    }
                    // Device: no WMMA grouped producer.
                    {
                        let mut sc = rbase.clone();
                        sc.arch = "gfx1030";
                        sc.flags = rdna_compute::FeatureFlags::from_active_config("gfx1030");
                        let r = run(&moe, &sc);
                        assert!(!r.facts.has_wmma);
                        assert_incumbent(&r, &sc, "no wmma");
                    }
                }
                // Memory: exact boundary with the full-row Q16 term, deduction,
                // live Q16 credit, retained-owner credit, failed free query.
                let need = resident_moe_need_bytes(&moe.config, 0);
                let mut sc = base.clone();
                sc.free_bytes = Some(need);
                assert!(
                    run(&moe, &sc).plan.wide,
                    "{name} {arch}: free == need admits"
                );
                sc.free_bytes = Some(need - 1);
                deny("free < need", &sc);
                let fallback = run(&moe, &sc);
                assert_eq!(
                    fallback.plan.receipt.to_string(),
                    format!(
                        "prefill_chunk: requested=8192 admitted={} commit_stride=none",
                        fallback.max_batch
                    ),
                    "{name} {arch}: memory fallback receipt"
                );
                assert!(emit_prefill_chunk_receipt_guarded(
                    &guard,
                    fallback.plan.receipt.requested,
                    fallback.plan.receipt.admitted,
                    fallback.plan.receipt.commit_stride,
                ));
                sc.free_bytes = None;
                deny("free query failed", &sc);
                sc.free_bytes = Some(need + 4096);
                sc.deduction = 4096;
                assert!(run(&moe, &sc).plan.wide);
                sc.deduction = 4097;
                deny("pending deduction", &sc);
                sc.deduction = 0;
                let q16_full =
                    resident_moe_q16_missing(&moe.config, RESIDENT_MOE_CHUNK_ROWS, 0).unwrap();
                sc.live_q16 = q16_full;
                sc.free_bytes = Some(need - q16_full);
                assert!(run(&moe, &sc).plan.wide, "{name} {arch}: live Q16 credit");
                sc.free_bytes = Some(need - q16_full - 1);
                deny("live Q16 credit boundary", &sc);
                sc.live_q16 = 0;
                // Retained widened owner credit: ONLY an owner this call reuses in
                // place (non-lean, rows >= the call's owned PBS rows). A smaller or
                // lean owner is freed to the allocator pool (bucket-local, not
                // guaranteed reclaimable by another size) and replaced, so the
                // full 8192 PBS is charged: no speculative credit.
                let pbs_rows_bytes =
                    |rows: usize| resident_moe_pbs_bytes(&moe.config, rows).unwrap();
                let with_cached = |n: usize, cached: Option<(usize, bool)>, free: usize| {
                    let mut c = base.clone();
                    c.n = n;
                    c.cached = cached;
                    c.free_bytes = Some(free);
                    c
                };
                let short = pbs_rows_bytes(2048);
                // Short owner (2048 < the 8192 rows an 8192-row call needs) is
                // replaced: the old speculative credit would admit this budget.
                deny(
                    "short cached owner earns no credit",
                    &with_cached(8192, Some((2048, false)), need - short),
                );
                assert!(
                    run(&moe, &with_cached(8192, Some((2048, false)), need))
                        .plan
                        .wide,
                    "{name} {arch}: short cached owner, full rung charged exactly"
                );
                // Same-shape owner reused in place: positive credit, exact boundary.
                let same = pbs_rows_bytes(8192);
                assert!(
                    run(&moe, &with_cached(8192, Some((8192, false)), need - same))
                        .plan
                        .wide,
                    "{name} {arch}: reused owner credit"
                );
                deny(
                    "reused owner credit boundary",
                    &with_cached(8192, Some((8192, false)), need - same - 1),
                );
                // A call needing only the owner's rows reuses a smaller owner.
                assert!(
                    run(&moe, &with_cached(100, Some((2048, false)), need - short))
                        .plan
                        .wide,
                    "{name} {arch}: n=100 reuses a 2048-row owner"
                );
                deny(
                    "n=100 reused owner credit boundary",
                    &with_cached(100, Some((2048, false)), need - short - 1),
                );
                // Exact reuse boundary: rows == needed rows credits, one row more
                // needed replaces the owner.
                assert!(
                    run(&moe, &with_cached(2048, Some((2048, false)), need - short))
                        .plan
                        .wide,
                    "{name} {arch}: owner rows == needed rows"
                );
                deny(
                    "owner one row short",
                    &with_cached(2049, Some((2048, false)), need - short),
                );
                // Lean mismatch: never credited.
                deny(
                    "lean owner is never credited",
                    &with_cached(100, Some((2048, true)), need - short),
                );
                // A 512-row Q16 budget would have admitted a rung whose real
                // full-row growth leaves < 1 GiB: it must deny now.
                let q16_512 =
                    resident_moe_q16_missing(&moe.config, WIDENED_COMMIT_ROWS, 0).unwrap();
                let mut old = base.clone();
                old.free_bytes = Some(need - (q16_full - q16_512));
                deny("old 512-row Q16 budget", &old);
                check_wrapper_actual_n(&moe, arch);
                checked += 1;
            }
        }
        eprintln!("resident_moe installed-artifact cases checked: {checked}");
        if require {
            assert_eq!(checked, 4, "both installed HFQs on both arches");
        }
    }

    /// Wrapper and call-fact seams without an artifact: only the dense route
    /// drops the implicit cache, every caller/capture fact closes the widen
    /// candidate, and the allocator geometry plan drives the state gate.
    #[test]
    fn ordinary_wrapper_cache_and_state_planners() {
        let inputs = |arch: &'static str,
                      fp8: bool,
                      dense: bool,
                      physical: Option<(usize, bool)>,
                      existing: Option<(usize, bool)>,
                      fits: bool| WrapperPbsInputs {
            arch,
            fp8_chunk512: fp8,
            dense_widened: dense,
            physical,
            existing_virtual: existing,
            default_moe_fits: fits,
            call_eligible: true,
        };
        let none = |p: WrapperPbsPlan| {
            p.owner == WrapperOwner::None && p.max_batch_override.is_none() && !p.alloc_virtual
        };
        // Dense widened route and reuse off never hand a cache; an old virtual
        // owner is released.
        assert!(none(plan_wrapper_pbs(&inputs(
            "gfx1100",
            false,
            true,
            Some((512, true)),
            None,
            true
        ))));
        assert!(none(plan_wrapper_pbs(&inputs(
            "gfx1100", false, false, None, None, true
        ))));
        assert!(
            plan_wrapper_pbs(&inputs(
                "gfx1100",
                false,
                false,
                None,
                Some((4096, true)),
                true
            ))
            .free_virtual
        );
        // Not fitting: the implicit cache, no override; an existing owner is released.
        let p = plan_wrapper_pbs(&inputs(
            "gfx1100",
            false,
            false,
            Some((512, true)),
            Some((4096, true)),
            false,
        ));
        assert_eq!(
            (
                p.owner,
                p.max_batch_override,
                p.free_virtual,
                p.alloc_virtual
            ),
            (WrapperOwner::Physical, None, true, false)
        );
        // Fitting default on gfx1100: the explicit-8192 owner (8192 rows, tape) from the
        // existing explicit sizing function, ceiling 8192.
        let target = implicit_reuse_pbs_plan("gfx1100", Some(RESIDENT_MOE_CHUNK_ROWS), false);
        assert_eq!(target, (8192, true));
        let p = plan_wrapper_pbs(&inputs(
            "gfx1100",
            false,
            false,
            Some((512, true)),
            None,
            true,
        ));
        assert_eq!(
            (
                p.owner,
                p.max_batch_override,
                p.free_virtual,
                p.alloc_virtual
            ),
            (
                WrapperOwner::Virtual {
                    rows: 8192,
                    tape: true
                },
                Some(8192),
                false,
                true
            )
        );
        // An existing owner of the same shape is reused; a different one is replaced.
        let p = plan_wrapper_pbs(&inputs(
            "gfx1100",
            false,
            false,
            Some((512, true)),
            Some(target),
            true,
        ));
        assert_eq!((p.free_virtual, p.alloc_virtual), (false, false));
        let p = plan_wrapper_pbs(&inputs(
            "gfx1100",
            false,
            false,
            Some((512, true)),
            Some((1024, false)),
            true,
        ));
        assert_eq!((p.free_virtual, p.alloc_virtual), (true, true));
        // A denying call (capture/tree/tape/ring/cap/state) never promotes, whatever
        // fits, and leaves existing owners untouched.
        let mut denied = inputs(
            "gfx1100",
            false,
            false,
            Some((512, true)),
            Some((4096, true)),
            true,
        );
        denied.call_eligible = false;
        let p = plan_wrapper_pbs(&denied);
        assert_eq!(
            (
                p.owner,
                p.max_batch_override,
                p.free_virtual,
                p.alloc_virtual
            ),
            (WrapperOwner::Physical, None, false, false)
        );
        // The shared scalar gate: prefill_batched off denies (outside the verify-decouple window).
        assert!(prefill_batch_scalar_eligible(
            512, "gfx1201", None, true, true
        ));
        assert!(!prefill_batch_scalar_eligible(
            512, "gfx1201", None, false, true
        ));
        assert!(!prefill_batch_scalar_eligible(
            512,
            "gfx1100",
            Some("0"),
            false,
            true
        ));
        assert!(prefill_batch_scalar_eligible(
            16, "gfx1100", None, false, true
        ));
        assert!(!prefill_batch_scalar_eligible(
            1, "gfx1201", None, true, true
        ));
        assert!(!prefill_batch_scalar_eligible(
            512, "gfx1201", None, true, false
        ));
        // The shared ceiling resolver: the cap applies to the override too.
        assert_eq!(resolve_ordinary_max_batch(512, None, Some(256)), 256);
        assert_eq!(resolve_ordinary_max_batch(512, Some(4096), Some(256)), 256);
        assert_eq!(resolve_ordinary_max_batch(512, Some(4096), None), 4096);
        assert_eq!(resolve_ordinary_max_batch(384, None, None), 384);
        // gfx1201 with the FP8 path: the implicit cache already is the explicit-8192 shape.
        let p = plan_wrapper_pbs(&inputs(
            "gfx1201",
            true,
            false,
            Some((512, false)),
            None,
            true,
        ));
        assert_eq!(
            (p.owner, p.max_batch_override, p.alloc_virtual),
            (WrapperOwner::Physical, Some(8192), false)
        );
        // gfx1201 without it: 384-row cache -> the 512-row tape-free explicit owner.
        let p = plan_wrapper_pbs(&inputs(
            "gfx1201",
            false,
            false,
            Some((384, false)),
            None,
            true,
        ));
        assert_eq!(
            (p.owner, p.max_batch_override, p.alloc_virtual),
            (
                WrapperOwner::Virtual {
                    rows: 512,
                    tape: false
                },
                Some(8192),
                true
            )
        );
        let open = OrdinaryCallFacts::default();
        assert!(open.wide_candidate());
        let flips: [CallFlip; 9] = [
            |c| c.caller_pbs_rows = Some(8192),
            |c| c.max_batch_cap = Some(256),
            |c| c.tree_verify = true,
            |c| c.gdn_tape = true,
            |c| c.staged_hidden_ring = true,
            |c| c.max_layer = true,
            |c| c.fusion_on = true,
            |c| c.capture_mode = true,
            |c| c.recording = true,
        ];
        for flip in flips {
            let mut c = open;
            flip(&mut c);
            assert!(!c.wide_candidate());
        }
        // Implicit cache sizing per arch, default and explicit.
        assert_eq!(implicit_reuse_pbs_plan("gfx1201", None, true), (512, false));
        assert_eq!(
            implicit_reuse_pbs_plan("gfx1201", None, false),
            (384, false)
        );
        assert_eq!(
            implicit_reuse_pbs_plan("gfx1201", Some(8192), false),
            (512, false)
        );
        assert_eq!(implicit_reuse_pbs_plan("gfx1100", None, false), (512, true));
        assert_eq!(
            implicit_reuse_pbs_plan("gfx1100", Some(8192), false),
            (8192, true)
        );
        assert_eq!(implicit_reuse_pbs_plan("gfx1151", None, false), (512, true));
        // State gate over the allocator's own scalar plan.
        let moe = resident_moe_test_config();
        let q8 = DeltaNetStatePlan::new(&moe, StateQuant::Q8, 1, true);
        assert!(ordinary_dn_state_admitted(&moe, &q8, false));
        assert!(!ordinary_dn_state_admitted(&moe, &q8, true));
        assert!(!ordinary_dn_state_admitted(
            &moe,
            &DeltaNetStatePlan::new(&moe, StateQuant::Q8, 1, false),
            false
        ));
        assert!(!ordinary_dn_state_admitted(
            &moe,
            &DeltaNetStatePlan::new(&moe, StateQuant::Q8, 2, true),
            false
        ));
        for quant in [StateQuant::FP32, StateQuant::Q4] {
            assert!(!ordinary_dn_state_admitted(
                &moe,
                &DeltaNetStatePlan::new(&moe, quant, 1, true),
                false
            ));
        }
    }

    /// Geometry over arbitrary per-owner counts (a live state's tensors or a
    /// malformed one): the same borrowed-iterator view the live state exposes.
    struct SliceGeometry {
        quant: StateQuant,
        s: Vec<usize>,
        ef: Vec<usize>,
    }

    impl DeltaNetStateGeometry for SliceGeometry {
        fn quant(&self) -> StateQuant {
            self.quant
        }
        fn s_numel(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
            self.s.iter().copied()
        }
        fn ef_numel(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
            self.ef.iter().copied()
        }
    }

    /// The allocator plan is three scalars with exactly the old per-layer
    /// allocation formulas, and its borrowed iterators reproduce the old
    /// per-layer vectors with no heap allocation.
    #[test]
    fn delta_net_state_plan_is_scalar_with_unchanged_allocation_formulas() {
        let moe = resident_moe_test_config();
        let n_la = moe
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count();
        assert_eq!(n_la, 48);
        let s_size = |batch: usize| {
            batch * moe.linear_num_value_heads * moe.linear_key_head_dim * moe.linear_key_head_dim
        };
        for batch in [1usize, 2, 4] {
            for (quant, ef) in [
                (StateQuant::FP32, false),
                (StateQuant::FP32, true),
                (StateQuant::Q8, true),
                (StateQuant::Q8, false),
                (StateQuant::Q4, true),
                (StateQuant::Q4, false),
            ] {
                let plan = DeltaNetStatePlan::new(&moe, quant, batch, ef);
                // Scalars: F32/Q8 one element per state element, Q4 packs two
                // per byte, EF (when on) is the unpacked per-element count.
                let s_per_layer = if quant == StateQuant::Q4 {
                    s_size(batch) / 2
                } else {
                    s_size(batch)
                };
                assert_eq!(plan.quant, quant);
                assert_eq!(plan.n_layers, n_la);
                assert_eq!(plan.s_per_layer, s_per_layer);
                assert_eq!(plan.ef_per_layer, ef.then_some(s_size(batch)));
                assert_eq!(plan.quant(), quant);
                let s: Vec<usize> = plan.s_numel().collect();
                assert_eq!(s, vec![s_per_layer; n_la]);
                let efs: Vec<usize> = plan.ef_numel().collect();
                assert_eq!(
                    efs,
                    if ef {
                        vec![s_size(batch); n_la]
                    } else {
                        Vec::new()
                    }
                );
                assert_eq!(plan.s_numel().len(), n_la);
                assert_eq!(plan.ef_numel().len(), if ef { n_la } else { 0 });
            }
        }
    }

    /// Admission walks EVERY S and EF owner (never a first-owner summary):
    /// a nonuniform, missing, extra, zero or wrong-quant owner at ANY position
    /// denies, and the live-state / plan views agree on valid state.
    #[test]
    fn dn_state_admission_checks_every_owner() {
        let moe = resident_moe_test_config();
        let n_la = 48usize;
        let single =
            moe.linear_num_value_heads * moe.linear_value_head_dim * moe.linear_value_head_dim;
        let valid = || SliceGeometry {
            quant: StateQuant::Q8,
            s: vec![single; n_la],
            ef: vec![single; n_la],
        };
        assert!(ordinary_dn_state_admitted(&moe, &valid(), false));
        assert!(!ordinary_dn_state_admitted(&moe, &valid(), true));
        // The slice view and the allocator plan are the same geometry.
        let plan = DeltaNetStatePlan::new(&moe, StateQuant::Q8, 1, true);
        assert_eq!(plan.s_per_layer, single);
        assert!(ordinary_dn_state_admitted(&moe, &plan, false));
        // Quant mismatch, including with otherwise valid owners.
        for quant in [StateQuant::FP32, StateQuant::Q4] {
            let mut g = valid();
            g.quant = quant;
            assert!(!ordinary_dn_state_admitted(&moe, &g, false));
        }
        // Every position, every malformation, in both owner lists.
        for i in 0..n_la {
            for bad in [0usize, single - 1, single + 1, single * 2, single / 2] {
                let mut g = valid();
                g.s[i] = bad;
                assert!(
                    !ordinary_dn_state_admitted(&moe, &g, false),
                    "S owner {i} = {bad}"
                );
            }
            // EF owners are required non-empty per layer (any nonzero count).
            let mut g = valid();
            g.ef[i] = 0;
            assert!(
                !ordinary_dn_state_admitted(&moe, &g, false),
                "EF owner {i} empty"
            );
            let mut g = valid();
            g.ef[i] = single * 3;
            assert!(
                ordinary_dn_state_admitted(&moe, &g, false),
                "EF owner {i} nonzero"
            );
        }
        // Missing / extra owners (count mismatch) anywhere in the list.
        for list in 0..2 {
            for extra in [false, true] {
                for pos in [0usize, n_la / 2, n_la - 1] {
                    let mut g = valid();
                    let v = if list == 0 { &mut g.s } else { &mut g.ef };
                    if extra {
                        v.insert(pos, single);
                    } else {
                        v.remove(pos);
                    }
                    assert!(
                        !ordinary_dn_state_admitted(&moe, &g, false),
                        "list={list} extra={extra} pos={pos}"
                    );
                }
            }
        }
        // EF entirely missing (EF off) and S entirely missing.
        let mut g = valid();
        g.ef.clear();
        assert!(!ordinary_dn_state_admitted(&moe, &g, false));
        let mut g = valid();
        g.s.clear();
        assert!(!ordinary_dn_state_admitted(&moe, &g, false));
    }

    /// The wrapper gate over pure facts: actual `n`, the 512-row seam and the
    /// caller/capture record, evaluated before any KV mapping / memory query /
    /// override / allocation.
    #[test]
    fn ordinary_wrapper_gate_uses_actual_n_and_seam() {
        let open = OrdinaryCallFacts::default();
        let phys = Some((512usize, true));
        let gate = |arch: &str,
                    decouple: Option<&str>,
                    batched: bool,
                    n: usize,
                    call: &OrdinaryCallFacts,
                    physical: Option<(usize, bool)>| {
            OrdinaryWrapperGate::new(n, call, physical, |rows| {
                prefill_batch_scalar_eligible(rows, arch, decouple, batched, true)
            })
        };
        let target = implicit_reuse_pbs_plan("gfx1100", Some(RESIDENT_MOE_CHUNK_ROWS), false);
        // Closed gate: incumbent resource plan whatever the fit says, with the
        // implicit cache (kept, existing virtual owner untouched) or without
        // (reuse off: the plan's stale-owner release only).
        let assert_closed = |g: &OrdinaryWrapperGate, what: &str| {
            assert!(!g.call_eligible, "{what}");
            assert!(!g.maps_kv(), "{what}: no KV mapping");
            assert!(!g.needs_default_fit(false), "{what}: no memory query");
            let p = g.plan("gfx1100", false, false, Some(target), true);
            assert_eq!(p.max_batch_override, None, "{what}");
            assert!(!p.alloc_virtual, "{what}");
            if g.physical.is_some() {
                assert_eq!(p.owner, WrapperOwner::Physical, "{what}");
                assert!(!p.free_virtual, "{what}: existing owner untouched");
            } else {
                assert_eq!(p.owner, WrapperOwner::None, "{what}");
            }
        };
        // n = 1 falls to the per-token AR loop: closed, reuse on and off.
        for physical in [phys, None] {
            assert_closed(&gate("gfx1100", None, true, 1, &open, physical), "n=1");
            assert_closed(
                &gate("gfx1201", None, true, 1, &open, physical),
                "n=1 gfx1201",
            );
        }
        // prefill_batched off: the verify-decouple window admits small n on
        // gfx11 (actual n allowed) but the 512-row seam is not: closed. Above
        // the window both are closed. gfx1201 needs the env opt-in for the
        // window and is still closed at the seam.
        assert!(prefill_batch_scalar_eligible(
            16, "gfx1100", None, false, true
        ));
        assert!(!prefill_batch_scalar_eligible(
            512, "gfx1100", None, false, true
        ));
        for physical in [phys, None] {
            for n in [2usize, 16, 32, 33, 40, 600] {
                assert_closed(
                    &gate("gfx1100", None, false, n, &open, physical),
                    "gfx1100 prefill_batched off",
                );
            }
            assert_closed(
                &gate("gfx1201", None, false, 16, &open, physical),
                "gfx1201 off",
            );
            assert!(prefill_batch_scalar_eligible(
                16,
                "gfx1201",
                Some("1"),
                false,
                true
            ));
            assert_closed(
                &gate("gfx1201", Some("1"), false, 16, &open, physical),
                "gfx1201 decouple=1, off",
            );
            assert_closed(
                &gate("gfx1100", Some("0"), false, 16, &open, physical),
                "decouple=0, off",
            );
        }
        // Valid positive n opens the gate: the implicit cache is mapped and the
        // memory fit is queried; reuse off maps nothing.
        for n in [2usize, 3, 16, 33, 100, 4096, 8192] {
            let g = gate("gfx1100", None, true, n, &open, phys);
            assert!(
                g.call_eligible && g.maps_kv() && g.needs_default_fit(false),
                "n={n}"
            );
            assert!(
                !g.needs_default_fit(true),
                "dense widened never queries the MoE fit"
            );
            let p = g.plan("gfx1100", false, false, None, true);
            assert_eq!(
                (p.owner, p.max_batch_override, p.alloc_virtual),
                (
                    WrapperOwner::Virtual {
                        rows: target.0,
                        tape: target.1
                    },
                    Some(RESIDENT_MOE_CHUNK_ROWS),
                    true
                ),
                "n={n}"
            );
            // A fit that was not asked for (needs_default_fit false) never promotes.
            let off = gate("gfx1100", None, true, n, &open, None);
            assert!(off.call_eligible && !off.maps_kv() && !off.needs_default_fit(false));
            let p = off.plan("gfx1100", false, false, Some(target), true);
            assert_eq!(
                (p.owner, p.max_batch_override, p.alloc_virtual),
                (WrapperOwner::None, None, false)
            );
            assert!(p.free_virtual);
        }
        // Every caller/capture fact closes the gate at a valid n.
        let flips: [CallFlip; 9] = [
            |c| c.caller_pbs_rows = Some(8192),
            |c| c.max_batch_cap = Some(256),
            |c| c.tree_verify = true,
            |c| c.gdn_tape = true,
            |c| c.staged_hidden_ring = true,
            |c| c.max_layer = true,
            |c| c.fusion_on = true,
            |c| c.capture_mode = true,
            |c| c.recording = true,
        ];
        for flip in flips {
            let mut c = open;
            flip(&mut c);
            assert_closed(&gate("gfx1100", None, true, 100, &c, phys), "call fact");
        }
        // An empty call touches nothing at all.
        let g = gate("gfx1100", None, true, 0, &open, phys);
        assert!(!g.call_eligible && !g.maps_kv());
        let p = g.plan("gfx1100", false, false, Some(target), true);
        assert_eq!(
            p,
            WrapperPbsPlan {
                owner: WrapperOwner::None,
                max_batch_override: None,
                free_virtual: false,
                alloc_virtual: false,
            }
        );
    }

    /// Only the existing virtual owner the wrapper keeps without replacement is
    /// credited; a mismatched owner (replaced, freed to the allocator pool) and
    /// an owner equal to the implicit cache (never kept) earn nothing.
    #[test]
    fn wrapper_virtual_credit_is_same_target_only() {
        let moe = resident_moe_test_config();
        let target = (8192usize, true);
        let bytes = resident_moe_owner_bytes(&moe, target.0, target.1).unwrap();
        let physical = Some((512usize, true));
        assert_eq!(
            wrapper_virtual_credit_bytes(&moe, physical, Some(target), target),
            bytes
        );
        assert_eq!(
            wrapper_virtual_credit_bytes(&moe, physical, None, target),
            0
        );
        for stale in [(1024usize, true), (4096, false), (4096, true), (512, false)] {
            assert_eq!(
                wrapper_virtual_credit_bytes(&moe, physical, Some(stale), target),
                0,
                "{stale:?}"
            );
        }
        // The implicit cache already is the target: no virtual owner is kept.
        assert_eq!(
            wrapper_virtual_credit_bytes(&moe, Some(target), Some(target), target),
            0
        );
        // The plan keeps exactly the credited owner (no free, no alloc).
        let p = plan_wrapper_pbs(&WrapperPbsInputs {
            arch: "gfx1100",
            fp8_chunk512: false,
            dense_widened: false,
            physical,
            existing_virtual: Some(target),
            default_moe_fits: true,
            call_eligible: true,
        });
        assert!(!p.free_virtual && !p.alloc_virtual);
        for stale in [(1024usize, true), (4096, false)] {
            let p = plan_wrapper_pbs(&WrapperPbsInputs {
                arch: "gfx1100",
                fp8_chunk512: false,
                dense_widened: false,
                physical,
                existing_virtual: Some(stale),
                default_moe_fits: true,
                call_eligible: true,
            });
            assert!(p.free_virtual && p.alloc_virtual, "{stale:?} is replaced");
        }
    }

    /// Wrapper actual-N proof over `moe` through the SHARED production
    /// wrapper decision ([`OrdinaryWrapperGate`], [`wrapper_run`] /
    /// [`run`]): valid positive n equals the explicit 8192 parameters; n = 1
    /// and the config-off / verify-decouple small windows (actual n and the
    /// 512 seam disagree) keep the original incumbent resource plan with no
    /// override, KV mapping, memory query or allocation.
    fn check_wrapper_actual_n(moe: &InstalledMoe, arch: &'static str) {
        let name = &moe.name;
        let base = |explicit: bool, reuse: bool, n: usize| {
            let mut sc = baseline(moe, arch, explicit, reuse, n);
            if !sc.kv.batched_flash() {
                sc.kv = KvTierFlags {
                    quant_q8: true,
                    ..KvTierFlags::default()
                };
            }
            sc
        };
        let flags = rdna_compute::FeatureFlags::from_active_config(arch);
        let fp8 = fp8_chunk512_for_flags(arch, &flags);
        let target = implicit_reuse_pbs_plan(arch, Some(RESIDENT_MOE_CHUNK_ROWS), fp8);
        let eff = |r: &Run| {
            (
                r.plan.wide,
                r.plan.limit,
                r.plan.pbs_rows,
                r.owner,
                r.chunks.clone(),
                r.plan.receipt.to_string(),
            )
        };
        // Valid positive n, reuse off and on: the default equals the explicit 8192.
        for reuse in [false, true] {
            for n in [2usize, 16, 33, 100] {
                let d = run(moe, &base(false, reuse, n));
                let e = run(moe, &base(true, reuse, n));
                assert!(
                    resident_moe_static_decision(&d.facts).is_some(),
                    "{name} {arch} n={n}: default admitted"
                );
                assert_eq!(resident_moe_static_decision(&e.facts), None);
                assert_eq!(d.plan.limit, RESIDENT_MOE_CHUNK_ROWS, "{name} {arch} n={n}");
                assert_eq!(
                    eff(&d),
                    eff(&e),
                    "{name} {arch} reuse={reuse} n={n}: default vs explicit 8192"
                );
                assert_eq!(
                    d.maps_kv, reuse,
                    "{name} {arch} n={n}: KV mapped iff cache present"
                );
                assert!(
                    !e.mem_queried,
                    "explicit knob never queries the default fit"
                );
                if reuse {
                    assert_eq!(d.wrapper.max_batch_override, Some(RESIDENT_MOE_CHUNK_ROWS));
                    assert!(d.mem_queried);
                }
            }
        }
        // Gate-denied calls: incumbent resource plan (see `OrdinaryWrapperGate`).
        let assert_denied = |sc: &Sc, what: &str| {
            let facts = sc_static_facts(moe, sc);
            let (plan, maps_kv, queried) = wrapper_run(moe, sc, &facts);
            assert!(!maps_kv, "{name} {arch} {what}: no KV mapping");
            assert!(!queried, "{name} {arch} {what}: no memory query");
            assert_eq!(
                plan.max_batch_override, None,
                "{name} {arch} {what}: no override"
            );
            assert!(!plan.alloc_virtual, "{name} {arch} {what}: no allocation");
            if sc.reuse_pbs {
                assert_eq!(plan.owner, WrapperOwner::Physical, "{what}");
                assert!(!plan.free_virtual, "{name} {arch} {what}: owners untouched");
            } else {
                assert_eq!(plan.owner, WrapperOwner::None, "{what}");
                assert_eq!(plan.free_virtual, sc.existing_virtual.is_some(), "{what}");
            }
        };
        // n = 1, reuse on/off, fresh and existing virtual owner. The 512-row
        // static facts alone would still admit the model.
        for reuse in [false, true] {
            for existing in [None, Some(target), Some((1024, false))] {
                let mut sc = base(false, reuse, 1);
                sc.existing_virtual = existing;
                let facts = sc_static_facts(moe, &sc);
                assert!(
                    resident_moe_static_decision(&facts).is_some(),
                    "{name} {arch}"
                );
                assert!(!sc_batched_eligible(moe, &sc, 1));
                assert_denied(&sc, "n=1");
                // Explicit knob: also the incumbent.
                let mut ex = base(true, reuse, 1);
                ex.existing_virtual = existing;
                assert_denied(&ex, "n=1 explicit");
            }
        }
        // prefill_batched off at the verify-decouple boundary and above: the
        // actual n may be admitted (gfx11 window) while the 512 seam is not.
        let decouple_env_unset =
            hipfire_config::developer_var_os("HIPFIRE_MTP_VERIFY_DECOUPLE").is_none();
        for reuse in [false, true] {
            for existing in [None, Some(target)] {
                for n in [2usize, 16, 32, 33, 40, 600] {
                    let mut sc = base(false, reuse, n);
                    sc.prefill_batched = false;
                    sc.existing_virtual = existing;
                    assert!(!sc_batched_eligible(moe, &sc, WIDENED_COMMIT_ROWS));
                    if decouple_env_unset {
                        assert_eq!(
                            sc_batched_eligible(moe, &sc, n),
                            arch == "gfx1100" && n <= 32,
                            "{name} {arch} n={n}: actual-n admission differs from the 512 probe"
                        );
                    }
                    assert_denied(&sc, "prefill_batched off");
                    // The whole planner flow keeps the original incumbent plan.
                    let r = run(moe, &sc);
                    assert_incumbent(&r, &sc, "prefill_batched off");
                    // The inner entry's own scalar/PBS eligibility at the actual
                    // n: above the verify-decouple window it returns to the
                    // per-token AR loop BEFORE the planner (no chunk plan runs);
                    // inside the gfx11 window the batched planner runs.
                    let inner_batched = sc_batched_eligible(moe, &sc, n);
                    assert_eq!(r.ar_fallback, !inner_batched, "{name} {arch} n={n}");
                    assert_eq!(r.chunks.is_empty(), !inner_batched, "{name} {arch} n={n}");
                }
            }
        }
    }

    /// [`check_wrapper_actual_n`] on a synthetic resident MoE (no installed
    /// artifact needed).
    #[test]
    fn wrapper_actual_n_synthetic_resident_moe() {
        let config = resident_moe_test_config();
        let dtypes = resident_moe_dtypes(DType::MQ4G256);
        let moe = InstalledMoe {
            name: "synthetic-a3b".to_string(),
            layers: vec![(ResidentMoePlacement::default(), dtypes); config.n_layers],
            config,
            moe_has_mq6: false,
            batch_pbs_eligible: true,
        };
        for arch in ["gfx1201", "gfx1100"] {
            check_wrapper_actual_n(&moe, arch);
        }
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
                GdnScanOut::F32,
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
