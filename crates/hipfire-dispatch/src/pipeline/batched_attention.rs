// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched (prefill / multi-row verify) gated full attention: input
//! projection, Q/K preparation, KV write + flash attention, and the gated
//! output projection with its producer routes. Moved verbatim from the
//! Qwen3.5 prefill path; the views below carry the same field names as the
//! architecture-owned weights, scratch and KV cache they borrow from.

#![allow(clippy::too_many_arguments)]

use crate::context::DispatchCtx;
use crate::families::attention::AttnParams;
use crate::families::gemv::WeightRef;
use crate::families::kv_tier::{KvTierInputs, KvTierPlan};
use crate::pipeline::batched::*;
use crate::pipeline::hybrid::{AttentionTap, HybridDims};
use crate::pipeline::steps::{execute_steps, Step};
use hip_bridge::{HipError, HipResult};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Row cadence of the widened ordinary prefill: larger contiguous GEMM/PBS
/// chunks still commit DeltaNet and flash attention per 512-row view. Only
/// `None` (legacy cadence) and `Some(512)` are legal commit strides.
pub const WIDENED_COMMIT_ROWS: usize = 512;

/// Central lane helpers — single source for max_batch bounds and shifts.
pub fn valid_lane_mask(max_batch: usize) -> HipResult<u64> {
    if max_batch == 0 || max_batch > 64 {
        return Err(HipError::new(0, "valid_lane_mask: max_batch must be 1..64"));
    }
    if max_batch >= 64 {
        Ok(u64::MAX)
    } else {
        Ok((1u64 << max_batch) - 1)
    }
}

/// Row semantics of one batched forward chunk.
#[derive(Clone, Copy)]
pub enum BatchSemantics<'a> {
    Sequential,
    Independent {
        positions: &'a [usize],
        lane_capacity: usize,
        active_mask: u64,
    },
}

impl BatchSemantics<'_> {
    #[inline]
    pub fn is_independent(self) -> bool {
        matches!(self, Self::Independent { .. })
    }
    #[inline]
    pub fn active_mask(self) -> Option<u64> {
        match self {
            Self::Independent { active_mask, .. } => Some(active_mask),
            Self::Sequential => None,
        }
    }
}

/// DDTree verify context: depth positions, ancestor bias and parent slots.
#[derive(Clone, Copy)]
pub struct TreeVerifyCtx<'a> {
    pub positions: &'a [i32],
    pub attn_bias: &'a GpuTensor,
    /// `[N]` i32 — for each linearized slot, the slot index of its parent
    /// in the same linearization (or -1 for the root / seed). Produced by
    /// `hipfire_runtime::ddtree::linearize_tree_with_parents`. When `Some`, LA layers
    /// use tree-aware kernels that read parent state from the per-layer
    /// s_tape scratch in `PrefillBatchScratch`.
    pub parent_indices: Option<&'a GpuTensor>,
}

/// Resident weights of one gated full-attention sublayer.
#[derive(Clone, Copy)]
pub struct AttentionLayerWeights<'a> {
    pub attn_norm: &'a GpuTensor,
    pub wq: WeightRef<'a>,
    pub wk: WeightRef<'a>,
    pub wv: WeightRef<'a>,
    pub q_norm: &'a GpuTensor,
    pub k_norm: &'a GpuTensor,
    pub wo: WeightRef<'a>,
    /// Gate projection of the dense FFN that follows, if any; decides whether
    /// the output projection may fold its residual into that FFN's norm
    /// producer. A routed MoE FFN never consumes the fold.
    pub w_gate: Option<WeightRef<'a>>,
}

/// Batched attention scratch rows, `[rows, …]` per tensor.
#[derive(Clone, Copy)]
pub struct AttentionBatchScratch<'a> {
    pub x_batch: &'a GpuTensor,
    pub x_rot_batch: &'a GpuTensor,
    pub x_rot_f16_batch: &'a GpuTensor,
    pub fa_q_full_batch: &'a GpuTensor,
    pub fa_q_batch: &'a GpuTensor,
    pub fa_gate_batch: &'a GpuTensor,
    pub fa_k_batch: &'a GpuTensor,
    pub fa_v_batch: &'a GpuTensor,
    pub fa_attn_out_batch: &'a GpuTensor,
    pub fa_attn_out_rot_batch: &'a GpuTensor,
    pub fa_attn_out_rot_f16_batch: &'a GpuTensor,
    /// PARO projection-input / output-projection scratch.
    pub x_norm_batch: &'a GpuTensor,
    /// Residual-fold delta (the following FFN's gate buffer, dead here).
    pub gate_ffn_batch: &'a GpuTensor,
    pub positions: &'a GpuTensor,
    pub rope_positions: &'a GpuTensor,
}

/// Flash-attention scratch shared with the decode path.
#[derive(Clone, Copy)]
pub struct FlashScratch<'a> {
    pub flash_partials: &'a GpuTensor,
    pub pos_buf: &'a hip_bridge::DeviceBuffer,
    pub flash_mode: u8,
}

/// Read-only KV cache view (per-layer K/V planes and tier facts).
#[derive(Clone, Copy)]
pub struct KvView<'a> {
    pub k_gpu: &'a [GpuTensor],
    pub v_gpu: &'a [GpuTensor],
    pub givens_cos: &'a Option<GpuTensor>,
    pub givens_sin: &'a Option<GpuTensor>,
    pub compact_offset: usize,
    pub physical_cap: usize,
    pub quant_q8: bool,
    pub quant_fp8: bool,
    pub tier: KvTierInputs,
    pub vmm_backend: bool,
}

impl KvView<'_> {
    pub fn tier_inputs(&self) -> KvTierInputs {
        self.tier
    }
    pub fn uses_vmm_backend(&self) -> bool {
        self.vmm_backend
    }
}

/// A4 (IU4) FA gate in place: `HIPFIRE_A4_FA_GATE_IL=0` keeps the fp8q prep's
/// gate copy and the compact-gate IU4 sigmoid producer (default on).
pub fn a4_fa_gate_in_place_enabled() -> bool {
    hipfire_config::developer_bool("HIPFIRE_A4_FA_GATE_IL", true)
}

/// A4 (IU4) attention epilogue: `HIPFIRE_A4_ATTN_EPI=0` keeps the fp8q
/// Q-resident attention's f32 output and the separate `_gil_` slab sigmoid
/// producer instead of the attention writing that producer's slab (default
/// on).
pub fn a4_attn_epilogue_enabled() -> bool {
    hipfire_config::developer_bool("HIPFIRE_A4_ATTN_EPI", true)
}

/// Whether [`try_gfx12_sigmoid_rotate_quant_fused_prepared`] admits: the
/// gfx1201 IU4 AWQ sigmoid producer is the FA output projection's producer.
pub fn gfx12_sigmoid_rotate_quant_admitted(
    gpu: &Gpu,
    wo: &WeightRef<'_>,
    k: usize,
    n: usize,
    epilogue: &BatchEpilogue<'_>,
) -> bool {
    wo.dtype == DType::MQ4G256V2
        && matches!(epilogue, BatchEpilogue::Residual)
        && wo.awq_scale.is_some_and(|awq| awq.numel() >= k)
        && gpu.iu4_producer_quant_fused_active(n, k)
}

/// ADD-epilogue fold (gfx1151 V2B, `HIPFIRE_V2B_ADDEPI`): whether the FFN
/// norm after an out-projection will take the IU4 RMSNorm route, the fold's
/// only consumer. When it will, the out-projection arms `pbs.gate_ffn_batch`
/// as the delta for its residual GEMM: that buffer is dead from the
/// out-projection until the gate/up GEMM, which runs after the norm.
pub fn residual_fold_consumer_ready(
    gpu: &Gpu,
    w_gate: Option<&WeightRef<'_>>,
    chain_verify: bool,
    n: usize,
    dim: usize,
) -> bool {
    w_gate.is_some_and(|w| w.dtype == DType::MQ4G256V2)
        && gpu.iu4_producer_sidecar_active(n, dim)
        && !mq_f16_projection_fast_route(gpu, chain_verify, n, dim)
}

/// Residual GEMM of an out-projection whose FFN norm follows directly:
/// offers the fold delta when [`residual_fold_consumer_ready`].
#[allow(clippy::too_many_arguments)]
pub fn out_proj_residual_iu4_prepared(
    gpu: &mut Gpu,
    x_batch: &GpuTensor,
    fold_delta: &GpuTensor,
    wo: &WeightRef<'_>,
    w_gate: Option<&WeightRef<'_>>,
    prep: &rdna_compute::Int4MmqPrepared,
    chain_verify: bool,
    n: usize,
) -> HipResult<()> {
    let fold = residual_fold_consumer_ready(gpu, w_gate, chain_verify, n, wo.m);
    if fold {
        gpu.arm_residual_fold(fold_delta)?;
    }
    let result =
        gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(wo.buf, prep, x_batch, wo.m, wo.k, n);
    if fold {
        gpu.disarm_residual_fold();
    }
    result
}

pub fn out_proj_residual_i8_prepared(
    gpu: &mut Gpu,
    x_batch: &GpuTensor,
    fold_delta: &GpuTensor,
    wo: &WeightRef<'_>,
    w_gate: Option<&WeightRef<'_>>,
    prep: &rdna_compute::Int8MmqPrepared,
    chain_verify: bool,
    n: usize,
) -> HipResult<()> {
    let fold = residual_fold_consumer_ready(gpu, w_gate, chain_verify, n, wo.m);
    if fold {
        gpu.arm_residual_fold(fold_delta)?;
    }
    let result = gpu.gemm_mq4g256v2_residual_wmma_i8_prepared(wo.buf, prep, x_batch, wo.m, wo.k, n);
    if fold {
        gpu.disarm_residual_fold();
    }
    result
}

pub fn try_a8_sigmoid_prepared(
    gpu: &mut Gpu,
    wo: &WeightRef<'_>,
    attn: &GpuTensor,
    gate: &GpuTensor,
    k: usize,
    n: usize,
    epilogue: &BatchEpilogue<'_>,
) -> HipResult<Option<rdna_compute::Int8MmqPrepared>> {
    if !gpu.flags.a8_fused_prod
        || !gpu.a8_prefill_active(n, k)
        || wo.dtype != DType::MQ4G256V2
        || !matches!(epilogue, BatchEpilogue::Residual)
    {
        return Ok(None);
    }
    let Some(awq) = wo.awq_scale else {
        return Ok(None);
    };
    let reservation = gpu.reserve_int8_mmq(k, n)?;
    gpu.sigmoid_mul_rotate_x_mq_awq_i8_gfx12_batched(attn, gate, awq, None, reservation, k, n)
        .map(Some)
}

/// gfx1201 slices-3: FWHT-rotate + `block_i4_128` IU4 producer for the
/// attention out-proj input. `None` → caller keeps the incumbent rotate +
/// standalone-quantizer path. Same contract as [`try_iu4_rotate_prepared`]
/// (uniform MQ4G256V2 only, Residual-only) but gated by
/// `HIPFIRE_GFX12_PRODUCER_QUANT_FUSED` on exact gfx1201. The f32 `x_rot`
/// store is always written, so every downstream reader is preserved
/// byte-for-byte.
pub fn try_gfx12_rotate_quant_fused_prepared(
    gpu: &mut Gpu,
    wo: &WeightRef<'_>,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    n: usize,
    epilogue: &BatchEpilogue<'_>,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    if wo.dtype != DType::MQ4G256V2
        || !matches!(epilogue, BatchEpilogue::Residual)
        || !gpu.iu4_producer_quant_fused_active(n, k)
    {
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep = gpu.rotate_x_mq_i4_gfx12_batched(x, wo.awq_scale, None, res, k, n)?;
    Ok(Some(prep))
}

/// gfx1201 FA out-proj producer: fuse the still-standalone
/// `sigmoid_mul_f32` into the AWQ rotate+IU4 sidecar. This is deliberately
/// AWQ-only: the arm of record has AWQ sidecars, while every failed predicate
/// keeps the established sigmoid → rotate/quant chain byte-for-byte.
/// `gate` is the compact gate copy or, when the fp8q FA prep skipped the
/// copy, the gate half of `fa_q_full_batch` read in place.
pub fn try_gfx12_sigmoid_rotate_quant_fused_prepared(
    gpu: &mut Gpu,
    wo: &WeightRef<'_>,
    attn: &GpuTensor,
    gate: rdna_compute::gemv::SigmoidGate<'_>,
    k: usize,
    n: usize,
    epilogue: &BatchEpilogue<'_>,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    if !gfx12_sigmoid_rotate_quant_admitted(gpu, wo, k, n, epilogue) {
        return Ok(None);
    }
    let Some(awq) = wo.awq_scale else {
        return Ok(None);
    };
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep =
        gpu.sigmoid_mul_rotate_x_mq_awq_i4_gfx12_batched(attn, gate, awq, None, res, k, n)?;
    Ok(Some(prep))
}

/// gfx1201 FP8-stream FA output producer.  Folds sigmoid, optional AWQ,
/// FWHT rotation and the scale_mode=1 pack into one row-wide launch;
/// `HIPFIRE_FP8_PROD_INREG=1` keeps the row in registers and skips `x_rot`.
/// `gate` is the compact gate copy or, when the fp8q FA prep skipped the
/// copy, the gate half of `fa_q_full_batch` read in place.
pub fn try_gfx12_fp8_stream_sigmoid_prepared(
    gpu: &mut Gpu,
    wo: &WeightRef<'_>,
    attn: &GpuTensor,
    gate: rdna_compute::gemv::SigmoidGate<'_>,
    x_rot: &GpuTensor,
    k: usize,
    n: usize,
) -> HipResult<Option<rdna_compute::Mq4v2Fp8Prepared>> {
    if !matches!(wo.dtype, DType::MQ4G256V2 | DType::MQ4G256V2Lloyd) || !gpu.fp8_stream_active(n, k)
    {
        return Ok(None);
    }
    let prep = gpu.rotate_x_mq_fp8_gfx12_batched(attn, Some(gate), wo.awq_scale, x_rot, k, n)?;
    Ok(Some(prep))
}

/// gfx11 FA out-proj producer: fuse the still-standalone
/// `sigmoid_mul_f32` into the AWQ rotate+IU4 sidecar, under the `_gfx11`
/// entry symbol. Deliberately AWQ-only like the `_gfx12` twin: the arm of
/// record has AWQ sidecars, while every failed predicate keeps the
/// established sigmoid → rotate/quant chain byte-for-byte. Gated by
/// `HIPFIRE_GFX11_PRODUCER_QUANT_FUSED` on gfx1100/gfx1151.
pub fn try_gfx11_sigmoid_rotate_quant_fused_prepared(
    gpu: &mut Gpu,
    wo: &WeightRef<'_>,
    attn: &GpuTensor,
    gate: &GpuTensor,
    k: usize,
    n: usize,
    epilogue: &BatchEpilogue<'_>,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    let Some(awq) = wo.awq_scale else {
        return Ok(None);
    };
    if wo.dtype != DType::MQ4G256V2
        || !matches!(epilogue, BatchEpilogue::Residual)
        || awq.numel() < k
        || !gpu.iu4_gfx11_producer_quant_fused_active(n, k)
    {
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep =
        gpu.sigmoid_mul_rotate_x_mq_awq_i4_gfx11_batched(attn, gate, awq, None, res, k, n)?;
    Ok(Some(prep))
}

/// T-B: FWHT-rotate + `block_i4_128` IU4 producer for wo (residual) inputs.
/// The wo input is a gated_norm/sigmoid output (never an rmsnorm output),
/// so the C2 producers can't cover it; its only IU4 consumer is the residual
/// GEMM via standalone `quantize_int4_mmq_ds128`, which this removes.
/// `None` → caller keeps incumbent rotate + standalone-quantizer path.
/// Residual-only (w_down C2 precedent): Partial TP keeps the plain path.
/// The f32 `x_rot` store is always skipped: on the admitted path the
/// prepared IU4 GEMM is the only consumer, and every fallback writes the
/// buffer itself before reading it.
pub fn try_iu4_rotate_prepared(
    gpu: &mut Gpu,
    wo: &WeightRef<'_>,
    x: &GpuTensor,
    k: usize,
    n: usize,
    epilogue: &BatchEpilogue<'_>,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    if wo.dtype != DType::MQ4G256V2
        || !matches!(epilogue, BatchEpilogue::Residual)
        || !gpu.iu4_producer_sidecar_active(n, k)
    {
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep = gpu.rotate_x_mq_i4_batched(x, wo.awq_scale, None, res, k, n)?;
    Ok(Some(prep))
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
pub fn run_fused_qkv_key(
    gpu: &mut Gpu,
    key: crate::types::KernelKey,
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
    use crate::families::fused_qkv::FusedQkvParams;
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
    fused_qkv_family()
        .run(&ctx, gpu, &params)
        .map_err(|e| HipError::new(0, &e.to_string()))
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
pub fn batched_gemm_single_weight(
    gpu: &mut Gpu,
    w: &WeightRef<'_>,
    x: &GpuTensor,
    y: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    match w.dtype {
        DType::MQ4G256 | DType::HFQ4G256 => run_plain_gemm_key(
            gpu,
            crate::types::KernelKey::GemmHfq4G256,
            w.buf,
            w.dtype,
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
                crate::types::KernelKey::GemmMq4G256V2Residual,
                w.buf,
                w.dtype,
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
                let c16 = lloyd_c16_or_fail(&w, "batched_gemm_single_weight")?;
                gpu.gemm_hfq4g256_residual_mmq_lloyd(w.buf, x, y, w.m, w.k, n, c16)
            } else {
                let lut = lloyd_e4m3_or_fail(&w, "batched_gemm_single_weight")?;
                gpu.gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8_lloyd(
                    w.buf, x, y, w.m, w.k, n, 1, lut,
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
                crate::types::KernelKey::GemmMq4CG256Residual,
                w.buf,
                w.dtype,
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
                crate::types::KernelKey::GemmHfq6G256Residual,
                w.buf,
                w.dtype,
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
                crate::types::KernelKey::GemmMq6G256V2Residual,
                w.buf,
                w.dtype,
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
                crate::types::KernelKey::GemmMq5G256V2Residual,
                w.buf,
                w.dtype,
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
                crate::types::KernelKey::GemmMq3G256V2Residual,
                w.buf,
                w.dtype,
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
                crate::types::KernelKey::GemmMq2G256V2Residual,
                w.buf,
                w.dtype,
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
                crate::types::KernelKey::GemmHfq3G256Residual,
                w.buf,
                w.dtype,
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
                crate::types::KernelKey::GemmQ8_0BatchedChunked,
                w.buf,
                w.dtype,
                x,
                y,
                w.m,
                w.k,
                n,
            )
        }
        other => Err(HipError::new(
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

#[allow(clippy::too_many_arguments)]
pub fn run_independent_q8_attention(
    gpu: &mut Gpu,
    pbs: &AttentionBatchScratch<'_>,
    kv_cache: &KvView<'_>,
    config: &HybridDims,
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
/// Prescaffold (behavior-only) extraction for S3-f16-projection-inputs.
///
/// Same statements, same order, same launches as the inlined block.
/// S9-mq4v2-persistent-prologues will issue `try_mq4v2_persistent_prologue`
/// from inside this hook after S3/S4 land.
pub fn attention_input_projection_batched(
    gpu: &mut Gpu,
    layer: &AttentionLayerWeights<'_>,
    config: &HybridDims,
    pbs: &AttentionBatchScratch<'_>,
    n: usize,
    dim: usize,
    q8_wmma_arch: bool,
    chain_verify: bool,
) -> HipResult<()> {
    let _ = chain_verify;
    // S3-f16-projection-inputs fast path: exact-FP16 FA qkv inputs. The
    // fused QKV kernel requires all three weights to share the MQ4G256V2
    // stride (same gate as `qkv_same_dtype` below, restricted to MQ4V2).
    if mq_f16_projection_fast_route(gpu, chain_verify, n, dim)
        && layer.wq.dtype == DType::MQ4G256V2
        && layer.wk.dtype == DType::MQ4G256V2
        && layer.wv.dtype == DType::MQ4G256V2
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
        layer.wq.dtype,
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
    let qkv_is_6bit = matches!(layer.wq.dtype, DType::MQ6G256 | DType::HFQ6G256);
    let qkv_is_mq3 = matches!(layer.wq.dtype, DType::MQ3G256);
    let qkv_is_mq3_lloyd = matches!(layer.wq.dtype, DType::MQ3G256Lloyd);
    let qkv_is_fp4 = matches!(layer.wq.dtype, DType::HFP4G32 | DType::MFP4G32);
    let qkv_is_q8 = matches!(layer.wq.dtype, DType::Q8_0);
    // TQ2G128/BQ1G128 have no fused qkvza/gate_up/qkv kernel, so they take
    // the same UNFUSED plain-GEMM strategy as Q8 rather than falling through
    // to the HFQ4 arm, which would read these packed blocks at the wrong
    // stride and produce fluent-but-wrong tokens.
    let qkv_is_lowbit = matches!(layer.wq.dtype, DType::TQ2G128 | DType::BQ1G128);
    // qt=52 re-arm anchor: q/k/v all Lloyd → FP8-LUT launcher.
    let qkv_is_mq4v2_lloyd = all_mq4v2_lloyd(&[layer.wq.dtype, layer.wk.dtype, layer.wv.dtype]);
    // Fused QKV kernels require all three weights to share a
    // dtype — they treat wq/wk/wv as same-stride byte arrays.
    // When kmap mode 2 promotes only `v_proj` (issue #249), the
    // fused HFQ4 path reads `wv` as MQ6 with HFQ4's 136-B stride
    // and produces silent NaN. Gate the fused kernels here.
    //
    // The Q8 substrate path (gemm_q8_0_batched_chunked × 3) also
    // dispatches a Q8-stride kernel per weight, so it needs the
    // same gate when wk/wv aren't Q8.
    let qkv_same_dtype = layer.wk.dtype == layer.wq.dtype && layer.wv.dtype == layer.wq.dtype;

    // 1. rmsnorm (+ rotate for MQ) for the attn preamble.
    let a8_prep = if qkv_same_dtype && layer.wq.dtype == DType::MQ4G256V2 {
        try_a8_rmsnorm_prepared(
            gpu,
            &pbs.x_batch,
            &layer.attn_norm,
            &layer.wq,
            dim,
            config.norm_eps,
            n,
        )?
    } else {
        None
    };
    let mut iu4_prep: Option<rdna_compute::Int4MmqPrepared> = None;
    let mut fp8_prep: Option<rdna_compute::Mq4v2Fp8Prepared> = None;
    if qkv_is_mq && a8_prep.is_none() {
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
            // gfx1201 slices-2: fused rmsnorm+quant producer (bit-identical).
            iu4_prep = try_gfx12_rmsnorm_quant_fused_prepared(
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
        }
        if iu4_prep.is_none() {
            // gfx1201 FP8-stream: RMSNorm/FWHT producer emits the fp8
            // pre-pass planes directly (byte-identical prepare outputs);
            // the standalone pack launch disappears. Lloyd weights only.
            fp8_prep = try_gfx12_fp8_stream_rmsnorm_prepared(
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
        if iu4_prep.is_none() && fp8_prep.is_none() {
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
    } else if !qkv_is_mq {
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
    if let Some(prep) = &a8_prep {
        let xq = gpu.int8_mmq_prepared_ptr(prep, layer.wq.k, n)?;
        gpu.gemm_mq4g256v2_mmq_set_prequant_i8(
            &layer.wq.buf,
            xq,
            &pbs.fa_q_full_batch,
            layer.wq.m,
            layer.wq.k,
            n,
        )?;
        gpu.gemm_mq4g256v2_mmq_set_prequant_i8(
            &layer.wk.buf,
            xq,
            &pbs.fa_k_batch,
            layer.wk.m,
            layer.wk.k,
            n,
        )?;
        gpu.gemm_mq4g256v2_mmq_set_prequant_i8(
            &layer.wv.buf,
            xq,
            &pbs.fa_v_batch,
            layer.wv.m,
            layer.wv.k,
            n,
        )?;
    } else if let Some(prep) = &iu4_prep {
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
            crate::types::KernelKey::FusedQkvHfq6G256,
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
            crate::types::KernelKey::FusedQkvMq3G256Lloyd,
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
            crate::types::KernelKey::FusedQkvHfq3G256,
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
            crate::types::KernelKey::FusedQkvHfp4G32,
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
            matches!(layer.wk.dtype, DType::Q8_0) && matches!(layer.wv.dtype, DType::Q8_0),
            "FA qkv Q8 WMMA dispatch requires all of wq/wk/wv to be Q8_0",
        );
        run_fused_qkv_key(
            gpu,
            crate::types::KernelKey::FusedQkvQ8_0,
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
            plain_gemm_key_for(layer.wq.dtype),
            &layer.wq.buf,
            layer.wq.dtype,
            &pbs.x_rot_batch,
            &pbs.fa_q_full_batch,
            layer.wq.m,
            layer.wq.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wk.dtype),
            &layer.wk.buf,
            layer.wk.dtype,
            &pbs.x_rot_batch,
            &pbs.fa_k_batch,
            layer.wk.m,
            layer.wk.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wv.dtype),
            &layer.wv.buf,
            layer.wv.dtype,
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
                lloyd_c16_or_fail(&layer.wq, "attention_input_projection_batched")?,
                lloyd_c16_or_fail(&layer.wk, "attention_input_projection_batched")?,
                lloyd_c16_or_fail(&layer.wv, "attention_input_projection_batched")?,
            )?;
        } else if let Some(prep) = &fp8_prep {
            // gfx1201 FP8-stream: the producer already emitted the fp8
            // pre-pass planes; consume them directly, no pack launch.
            gpu.gemm_qkv_hfq4g256_wmma_gfx12_mq4v2_fp8_prepared_lloyd(
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
                lloyd_e4m3_or_fail(&layer.wq, "attention_input_projection_batched")?,
                lloyd_e4m3_or_fail(&layer.wk, "attention_input_projection_batched")?,
                lloyd_e4m3_or_fail(&layer.wv, "attention_input_projection_batched")?,
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
                lloyd_e4m3_or_fail(&layer.wq, "attention_input_projection_batched")?,
                lloyd_e4m3_or_fail(&layer.wk, "attention_input_projection_batched")?,
                lloyd_e4m3_or_fail(&layer.wv, "attention_input_projection_batched")?,
            )?;
        }
    } else if matches!(layer.wq.dtype, DType::MQ4G256V2Lloyd)
        || matches!(layer.wk.dtype, DType::MQ4G256V2Lloyd)
        || matches!(layer.wv.dtype, DType::MQ4G256V2Lloyd)
    {
        // Mixed Lloyd/uniform FA qkv: no kernel reads mixed codebooks.
        return Err(HipError::new(
            0,
            "attention_input_projection_batched: mixed MQ4G256V2Lloyd/uniform FA qkv — refusing (quantize all three or none)",
        ));
    } else if layer.wq.dtype == DType::MQ4G256V2
        && layer.wk.dtype == DType::MQ4G256V2
        && layer.wv.dtype == DType::MQ4G256V2
        && gpu.flags.gfx12_mq4v2_fp8_qkv
        && fp8_prep.is_some()
    {
        // gfx1201 FP8-stream (uniform): the producer already emitted the
        // fp8 pre-pass planes; consume them directly with the launch twin
        // of the family's fp8 route — no pack launch. Same fp8 intercept
        // conditions as the uniform router (iu4 divergence excluded by
        // producer-side ordering: fp8_prep implies iu4_prep is None).
        gpu.gemm_qkv_hfq4g256_wmma_gfx12_mq4v2_fp8_prepared(
            &layer.wq.buf,
            &layer.wk.buf,
            &layer.wv.buf,
            fp8_prep.as_ref().unwrap(),
            &pbs.fa_q_full_batch,
            &pbs.fa_k_batch,
            &pbs.fa_v_batch,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
            n,
        )?;
    } else if qkv_same_dtype {
        run_fused_qkv_key(
            gpu,
            crate::families::fused_qkv::fused_qkv_key_for(layer.wq.dtype),
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
/// (`execute_gated_attention_batched`) issues it via `attention_attend_batched`
/// immediately after, so single-chunk launch order is unchanged.
pub fn attention_prepare_batched(
    gpu: &mut Gpu,
    // F2 split: the trailing KV-write + flash-attention dispatch moved to
    // the caller, so these attend-only params are kept for signature
    // stability but currently unused.
    _fa_attn_multirow: bool,
    layer: &AttentionLayerWeights<'_>,
    config: &HybridDims,
    pbs: &AttentionBatchScratch<'_>,
    _s: &FlashScratch<'_>,
    kv_cache: &KvView<'_>,
    n: usize,
    _start_pos: usize,
    _max_ctx_len: usize,
    _ctx: &DispatchCtx,
    _batch_semantics: BatchSemantics<'_>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    _kv_layer_idx: usize,
    layer_idx: usize,
    chain_verify: bool,
    gfx12_fa_prep: bool,
    gfx12_fa_prep_fp8q: bool,
    // The fp8q prep leaves the gate in `fa_q_full_batch` (no gate copy); the
    // output projection must then read it in place (`fa_gate_in_place`) with
    // the IU4 or FP8-stream sigmoid producer.
    fa_gate_in_place: bool,
    tap: Option<AttentionTap<'_>>,
) -> HipResult<()> {
    // S6-fa-prep-q8-pair: exact gfx1100 fold of steps 3-5 (deinterleave +
    // Q/K rmsnorm + half-split RoPE, 4 launches) into one
    // qwen35_fa_prep_batched_gfx1100 launch. Bit-exact (same reduction tree,
    // same RoPE expression/phase, explicit old-TU FMA formation); the triattn
    // tap needs pre-RoPE Q, legacy interleaved RoPE needs its own kernel, and
    // every other shape/arch/ctx keeps the old path. Admitted for DFlash
    // chain verify and, since gfx11-1, ordinary prefill (DflashFusionCtx::Off:
    // −325 µs per N4096 call vs the three-launch chain) unless
    // HIPFIRE_GFX1100_FA_PREP=0.
    // HIPFIRE_FA_BATCH_FUSE_OFF=1 restores it byte-for-byte.
    // Admitted geometries are 16Q/2K and 24Q/4K (Qwen3.8-27B FA is 24/4);
    // HD must be 256 and n_rot 64.
    let fa_prep_n_rot = config.n_rot;
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
    let fa_prep_fused_ok = (chain_verify
        || (!chain_verify && hipfire_config::developer_bool("HIPFIRE_GFX1100_FA_PREP", true)))
        && gpu.arch_caps.is_gfx1100()
        && !gpu.flags.fa_batch_fuse_off
        && !gpu.flags.rope_interleaved_legacy
        && !tap.is_some()
        && fa_prep_shape_ok
        && n >= 1;
    // gfx1151 twin (`qwen35_fa_prep_batched_gfx1151`, same source) in
    // ordinary prefill, bit-exact against the three-launch chain on the
    // Halo; HIPFIRE_GFX1151_FA_PREP=0 restores the chain.
    let fa_prep_fused_gfx1151_ok = !chain_verify
        && hipfire_config::developer_bool("HIPFIRE_GFX1151_FA_PREP", true)
        && gpu.arch_caps.is_gfx1151()
        && !gpu.flags.fa_batch_fuse_off
        && !gpu.flags.rope_interleaved_legacy
        && !tap.is_some()
        && fa_prep_shape_ok
        && n >= 1;
    if gfx12_fa_prep {
        if gfx12_fa_prep_fp8q {
            let bytes = n * config.n_heads * (config.head_dim + 4);
            let q_codes = GpuTensor {
                buf: unsafe {
                    hip_bridge::DeviceBuffer::from_raw(pbs.fa_q_batch.buf.as_ptr(), bytes)
                },
                shape: vec![bytes],
                dtype: DType::Raw,
            };
            gpu.qwen35_fa_prep_batched_gfx1201(
                &pbs.fa_q_full_batch,
                rdna_compute::qwen35_fa_batch::FaPrepQOut::Fp8Codes(&q_codes),
                (!fa_gate_in_place).then_some(&pbs.fa_gate_batch),
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
            gpu.qwen35_fa_prep_batched_gfx1201(
                &pbs.fa_q_full_batch,
                rdna_compute::qwen35_fa_batch::FaPrepQOut::F32(&pbs.fa_q_batch),
                Some(&pbs.fa_gate_batch),
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
        }
    } else if fa_prep_fused_ok {
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
    } else if fa_prep_fused_gfx1151_ok {
        gpu.qwen35_fa_prep_batched_gfx1151(
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

        if let Some(tap) = tap {
            tap(gpu, &pbs.fa_q_batch, &pbs.fa_k_batch, n)?;
        }

        // 5. Batched partial-interleaved RoPE (per-row positions).
        // pos_offset = compact_offset so new Q/K rotate at ABSOLUTE phase
        // after eviction (cached keys are absolute-phased); pbs.positions
        // stays physical for the KV-write below. 0 when no compaction.
        let n_rot = config.n_rot;
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
    // the caller via `attention_attend_batched`, so a pair of chunks can share
    // one merged 1024-row attend step. Single-chunk order is unchanged.)
    Ok(())
}

/// Shared KV-write + flash-attention dispatch tail (F2 extraction).
///
/// `attention_attend_batched` calls this with the chunk's own FA tensors; the
/// F2 pair path calls it once with 1024-row staged tensors covering both
/// halves. Same plan derivation, same single `Step::Attend` (the family
/// writes the KV rows from `k`/`v` at `positions` before attending, so both
/// halves' KV are fully written before the merged FA2 reads them), same
/// stream ordering. `a4_epilogue` = (Q/gate projection rows, wo AWQ scales)
/// selects the A4 epilogue attention on the fp8q route; `output` is then the
/// out-projection's A4 slab ([`attention_attend_a4_batched`]).
#[allow(clippy::too_many_arguments)]
pub fn execute_fa_attend_step(
    gpu: &mut Gpu,
    config: &HybridDims,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    positions: &GpuTensor,
    output: &GpuTensor,
    s: &FlashScratch<'_>,
    kv_cache: &KvView<'_>,
    n: usize,
    start_pos: usize,
    max_ctx_len: usize,
    ctx: &DispatchCtx,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    layer_idx: usize,
    a4_epilogue: Option<(&GpuTensor, &GpuTensor)>,
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
        output_gate: a4_epilogue.map(|(gate, _)| gate),
        output_awq_scale: a4_epilogue.map(|(_, awq)| awq),
        output,
    };
    execute_steps(gpu, ctx, &[Step::Attend { plan, io }])
        .map_err(|e| HipError::new(0, &e.to_string()))
}

#[allow(clippy::too_many_arguments)]
pub fn attention_attend_batched(
    gpu: &mut Gpu,
    config: &HybridDims,
    pbs: &AttentionBatchScratch<'_>,
    s: &FlashScratch<'_>,
    kv_cache: &KvView<'_>,
    n: usize,
    start_pos: usize,
    max_ctx_len: usize,
    ctx: &DispatchCtx,
    batch_semantics: BatchSemantics<'_>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    layer_idx: usize,
    multirow: bool,
    commit_stride: Option<usize>,
    gfx12_fa_prep_fp8q: bool,
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
        // Exact gfx1201 native-fp8 packet path: write the whole widened
        // chunk once, then let packet grid.z enumerate the equal-length
        // legacy runs in one launch. Every row keeps its absolute position;
        // pre-writing later K/V rows is unobservable under the causal mask,
        // while x restarts per run so each workgroup sees the same query set
        // and max/min bounds as the former per-step launch. Only whole-run
        // chunks qualify; an odd tail takes the per-segment loop below.
        let packet_runs = gpu.arch == "gfx1201"
            && (gpu.flags.gfx12_fa_packet || gpu.flags.attn_qresident)
            && kv_cache.quant_fp8
            && config.n_heads == 24
            && config.n_kv_heads == 4
            && config.head_dim == 256
            && stride == WIDENED_COMMIT_ROWS
            && n % WIDENED_COMMIT_ROWS == 0
            && n <= 32768
            && max_ctx_len <= 262_144
            && tree_verify.is_none();
        let q_codes = gfx12_fa_prep_fp8q.then(|| {
            let bytes = n * config.n_heads * (config.head_dim + 4);
            GpuTensor {
                buf: unsafe {
                    hip_bridge::DeviceBuffer::from_raw(pbs.fa_q_batch.buf.as_ptr(), bytes)
                },
                shape: vec![bytes],
                dtype: DType::Raw,
            }
        });
        if packet_runs {
            execute_fa_attend_step(
                gpu,
                config,
                q_codes.as_ref().unwrap_or(&pbs.fa_q_batch),
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
                None,
                layer_idx,
                None,
            )?;
            return Ok(());
        }
        // Whole-chunk Q8/Q8 writes all positions before one causal FA2
        // launch. Keep all exceptions on the unchanged segmented path: an
        // alternate backend or variant cannot safely receive B>512 here.
        let q8_wide_runs = gpu.flags.gfx11_q8_fa2_wide
            && gpu.flags.gfx11_fa2_prefill
            && matches!(gpu.arch.as_str(), "gfx1100" | "gfx1151")
            && kv_cache.quant_q8
            && !kv_cache.quant_fp8
            && kv_cache.tier_inputs().v_mode_bits == 8
            && kv_cache.uses_vmm_backend()
            && !gpu.flash_attn_ck_loaded()
            && !matches!(
                hipfire_config::developer_var("HIPFIRE_FLASH_PREFILL")
                    .ok()
                    .as_deref(),
                Some("0") | Some("off") | Some("false")
            )
            && !matches!(
                hipfire_config::developer_var("HIPFIRE_FLASH_PREFILL_KERNEL")
                    .ok()
                    .as_deref(),
                Some("scalar") | Some("batched")
            )
            && ctx.workload == crate::context::DispatchWorkload::Standard
            && config.n_heads == 24
            && config.n_kv_heads == 4
            && config.head_dim == 256
            && stride == WIDENED_COMMIT_ROWS
            && (64..=8192).contains(&n)
            && (n <= WIDENED_COMMIT_ROWS || n % WIDENED_COMMIT_ROWS == 0)
            && gpu.fa2_gfx11_ctx_admitted(max_ctx_len)
            && start_pos.checked_add(n) == Some(max_ctx_len)
            && max_ctx_len <= kv_cache.physical_cap
            && tree_verify.is_none();
        if q8_wide_runs {
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
                None,
                layer_idx,
                None,
            )?;
            return Ok(());
        }
        // Other tiers, and odd-length chunks, retain one write-then-attend
        // step per legacy segment, then the final valid partial tile. Padded
        // projection rows never enter KV; each context ends at the last valid
        // row in its segment.
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;
        for off in (0..n).step_by(stride) {
            let seg_n = (n - off).min(stride);
            let q = pbs.fa_q_batch.sub_offset(off * q_dim, seg_n * q_dim);
            let k = pbs.fa_k_batch.sub_offset(off * kv_dim, seg_n * kv_dim);
            let v = pbs.fa_v_batch.sub_offset(off * kv_dim, seg_n * kv_dim);
            let positions = pbs.positions.sub_offset(off, seg_n);
            let out = pbs.fa_attn_out_batch.sub_offset(off * q_dim, seg_n * q_dim);
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
                seg_n,
                start_pos + off,
                start_pos + off + seg_n,
                ctx,
                tree_verify,
                layer_idx,
                None,
            )?;
        }
        return Ok(());
    }
    if gfx12_fa_prep_fp8q {
        let bytes = n * config.n_heads * (config.head_dim + 4);
        let q_codes = GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(pbs.fa_q_batch.buf.as_ptr(), bytes) },
            shape: vec![bytes],
            dtype: DType::Raw,
        };
        return execute_fa_attend_step(
            gpu,
            config,
            &q_codes,
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
            None,
        );
    }
    if multirow {
        debug_assert!(
            gpu.arch_caps.is_gfx1100() || gpu.arch_caps.is_gfx1151() || gpu.arch_caps.is_gfx1201()
        );
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
        None,
    )
}

/// The fp8q route's single write-then-attend step (what `attention_attend_batched`
/// reduces to with `gfx12_fa_prep_fp8q`) with the A4 attention epilogue: the
/// attention writes the out-projection's A4 slab into a fresh IU4
/// reservation of `(k, n)` instead of `fa_attn_out_batch`; the gate is read
/// in place from `fa_q_full_batch`.
#[allow(clippy::too_many_arguments)]
pub fn attention_attend_a4_batched(
    gpu: &mut Gpu,
    config: &HybridDims,
    pbs: &AttentionBatchScratch<'_>,
    s: &FlashScratch<'_>,
    kv_cache: &KvView<'_>,
    n: usize,
    start_pos: usize,
    max_ctx_len: usize,
    ctx: &DispatchCtx,
    layer_idx: usize,
    k: usize,
    awq: &GpuTensor,
) -> HipResult<rdna_compute::Int4MmqPrepared> {
    let res = gpu.reserve_int4_mmq(k, n)?;
    let slab_bytes = (k / 128) * n * 72;
    let slab = GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(res.ptr(), slab_bytes) },
        shape: vec![slab_bytes],
        dtype: DType::Raw,
    };
    let q_bytes = n * config.n_heads * (config.head_dim + 4);
    let q_codes = GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(pbs.fa_q_batch.buf.as_ptr(), q_bytes) },
        shape: vec![q_bytes],
        dtype: DType::Raw,
    };
    execute_fa_attend_step(
        gpu,
        config,
        &q_codes,
        &pbs.fa_k_batch,
        &pbs.fa_v_batch,
        &pbs.positions,
        &slab,
        s,
        kv_cache,
        n,
        start_pos,
        max_ctx_len,
        ctx,
        None,
        layer_idx,
        Some((&pbs.fa_q_full_batch, awq)),
    )?;
    let prep = rdna_compute::Int4MmqPrepared::from_reservation(res);
    gpu.scratch.mark_int4_mmq_slab(&prep)?;
    Ok(prep)
}

#[allow(clippy::too_many_arguments)]
/// Prescaffold (behavior-only) extraction for S4-f16-residual-inputs.
///
/// Same statements, same order, same launches as the inlined block.
/// S9-mq4v2-persistent-prologues will issue `try_mq4v2_persistent_prologue`
/// from inside this hook after S3/S4 land.
pub fn attention_output_projection_batched(
    gpu: &mut Gpu,
    layer: &AttentionLayerWeights<'_>,
    pbs: &AttentionBatchScratch<'_>,
    n: usize,
    q8_wmma_arch: bool,
    arch_has_wmma: bool,
    epilogue: BatchEpilogue<'_>,
    chain_verify: bool,
    // Set with the matching `attention_prepare_batched` flag: the gate
    // was left in `fa_q_full_batch` and only the gfx1201 IU4 or FP8-stream
    // sigmoid producer reads it.
    fa_gate_in_place: bool,
    // The out-projection's A4 slab, already written by the A4 attention
    // epilogue ([`attention_attend_a4_batched`]); `fa_attn_out_batch` was not.
    a4_epi: Option<rdna_compute::Int4MmqPrepared>,
) -> HipResult<()> {
    if let Some(prep) = &a4_epi {
        return out_proj_residual_iu4_prepared(
            gpu,
            &pbs.x_batch,
            &pbs.gate_ffn_batch,
            &layer.wo,
            layer.w_gate.as_ref(),
            prep,
            chain_verify,
            n,
        );
    }
    // S4: one sigmoid*attn+FWHT+F16 producer + direct-F16 residual GEMM
    // instead of sigmoid_mul_f32 + mq_rotate_x + convert. The F32 attn
    // input is left unmutated (the old in-place sigmoid write is skipped).
    if s4_residual_fast(gpu, chain_verify, layer.wo.dtype, &epilogue, n) {
        let k = layer.wo.k;
        let m = layer.wo.m;
        if k > 0 && k % 256 == 0 {
            if let Some(awq) = layer.wo.awq_scale {
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
    // The gfx1201 IU4 AWQ fast route below folds this sigmoid multiply into
    // its rotate+quant producer. Every fallback still executes the established
    // in-place launch before selecting its rotate/GEMM route.

    // 9. wo residual: x_batch += wo · (optional rotate)(fa_attn_out_batch).
    // Same MQ rotation requirement as the LA wo path.
    let fa_wo_is_mq = matches!(
        layer.wo.dtype,
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
    let a8_wo_prep = if fa_gate_in_place {
        None
    } else {
        try_a8_sigmoid_prepared(
            gpu,
            &layer.wo,
            &pbs.fa_attn_out_batch,
            &pbs.fa_gate_batch,
            layer.wo.k,
            n,
            &epilogue,
        )?
    };
    let fa_gate = if fa_gate_in_place {
        rdna_compute::gemv::SigmoidGate::QGateInterleaved(&pbs.fa_q_full_batch)
    } else {
        rdna_compute::gemv::SigmoidGate::Rows(&pbs.fa_gate_batch)
    };
    let mut iu4_wo_prep = if a8_wo_prep.is_none() {
        try_gfx12_sigmoid_rotate_quant_fused_prepared(
            gpu,
            &layer.wo,
            &pbs.fa_attn_out_batch,
            fa_gate,
            layer.wo.k,
            n,
            &epilogue,
        )?
    } else {
        None
    };
    if iu4_wo_prep.is_none() && !fa_gate_in_place {
        // gfx11 twin of the slices-5 sigmoid chain_verify: skip the standalone
        // sigmoid store when the `_gfx11` producer admits (prepared GEMM is
        // the only consumer, `attn` left unmodified).
        iu4_wo_prep = try_gfx11_sigmoid_rotate_quant_fused_prepared(
            gpu,
            &layer.wo,
            &pbs.fa_attn_out_batch,
            &pbs.fa_gate_batch,
            layer.wo.k,
            n,
            &epilogue,
        )?;
    }
    let mut fp8_wo_prep: Option<rdna_compute::Mq4v2Fp8Prepared> = None;
    if a8_wo_prep.is_none() && iu4_wo_prep.is_none() {
        fp8_wo_prep = try_gfx12_fp8_stream_sigmoid_prepared(
            gpu,
            &layer.wo,
            &pbs.fa_attn_out_batch,
            fa_gate,
            &pbs.fa_attn_out_rot_batch,
            layer.wo.k,
            n,
        )?;
    }
    if fa_gate_in_place && iu4_wo_prep.is_none() && fp8_wo_prep.is_none() {
        return Err(HipError::new(
            0,
            "FA output: gate left in fa_q_full_batch but neither the IU4 nor the FP8-stream sigmoid producer was admitted",
        ));
    }
    if a8_wo_prep.is_none() && iu4_wo_prep.is_none() && fp8_wo_prep.is_none() {
        gpu.sigmoid_mul_f32(&pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
        // T-B / slices-3: rotate+quantize once the standalone sigmoid has
        // produced the f32 input. All non-admitted routes remain unchanged.
        iu4_wo_prep = try_iu4_rotate_prepared(
            gpu,
            &layer.wo,
            &pbs.fa_attn_out_batch,
            layer.wo.k,
            n,
            &epilogue,
        )?;
        if iu4_wo_prep.is_none() {
            iu4_wo_prep = try_gfx12_rotate_quant_fused_prepared(
                gpu,
                &layer.wo,
                &pbs.fa_attn_out_batch,
                &pbs.fa_attn_out_rot_batch,
                layer.wo.k,
                n,
                &epilogue,
            )?;
        }
    }
    let fa_wo_input = if a8_wo_prep.is_some() || iu4_wo_prep.is_some() || fp8_wo_prep.is_some() {
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
    if let Some(prep) = &a8_wo_prep {
        out_proj_residual_i8_prepared(
            gpu,
            &pbs.x_batch,
            &pbs.gate_ffn_batch,
            &layer.wo,
            layer.w_gate.as_ref(),
            prep,
            chain_verify,
            n,
        )?;
    } else if let Some(prep) = &iu4_wo_prep {
        out_proj_residual_iu4_prepared(
            gpu,
            &pbs.x_batch,
            &pbs.gate_ffn_batch,
            &layer.wo,
            layer.w_gate.as_ref(),
            prep,
            chain_verify,
            n,
        )?;
    } else if let Some(prep) = &fp8_wo_prep {
        dispatch_batched_fp8_lloyd_epilogue(gpu, &pbs.x_batch, &layer.wo, prep, &epilogue, n)?;
    } else {
        dispatch_batched_gemm_epilogue(
            gpu,
            &pbs.x_batch,
            &pbs.x_rot_batch,
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

pub fn execute_gated_attention_batched(
    gpu: &mut Gpu,
    fa_attn_multirow: bool,
    layer: &AttentionLayerWeights<'_>,
    config: &HybridDims,
    pbs: &AttentionBatchScratch<'_>,
    s: &FlashScratch<'_>,
    kv_cache: &KvView<'_>,
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
    chain_verify: bool,
    commit_stride: Option<usize>,
    tap: Option<AttentionTap<'_>>,
) -> HipResult<()> {
    // Fully batched FA layer. Mirrors the FA branch of
    // forward_scratch_layers kernel-for-kernel, but every
    // launch covers all N tokens at once.
    let kv_dim = config.n_kv_heads * config.head_dim;
    let q_dim = config.n_heads * config.head_dim;
    let gfx12_fa_prep = gpu.arch == "gfx1201"
        && gpu.flags.gfx12_fa_prep_fused
        && !chain_verify
        && !gpu.flags.rope_interleaved_legacy
        && !tap.is_some()
        && config.head_dim == 256
        && (config.n_heads, config.n_kv_heads) == (24, 4)
        && config.n_rot == 64
        && n > 0;
    let gfx12_fa_prep_fp8q = gfx12_fa_prep
        && gpu.flags.gfx12_fa_prep_fp8q
        && gpu.flags.attn_qresident
        && gpu.flags.attn_qresident_v2
        && !fa_attn_multirow
        && !batch_semantics.is_independent()
        && kv_cache.quant_fp8
        && config.n_heads == 24
        && config.n_kv_heads == 4
        && (64..=32768).contains(&n)
        && (n <= 512 || n % 512 == 0)
        && (64..=262_144).contains(&max_ctx_len)
        && tree_verify.is_none()
        && (commit_stride.is_none()
            || (commit_stride == Some(WIDENED_COMMIT_ROWS) && n % WIDENED_COMMIT_ROWS == 0));
    // The fp8q prep can leave the sigmoid gate in `fa_q_full_batch` (no
    // 4·n·q_dim-byte copy, and its gate half is never read by the prep) when
    // the output projection will select a sigmoid producer that reads the
    // gate in place: the gfx1201 IU4 AWQ producer (A4, unless
    // `HIPFIRE_A4_FA_GATE_IL=0`) or the FP8-stream producer. Mirrors their
    // admission: A8 is excluded by `iu4_producer_quant_fused_active` /
    // `fp8_stream_active`, the IU4 producer is tried first and the gfx11
    // chain_verify only after it; gfx1100 S4 needs `gfx12_fa_prep`'s gfx1201 false.
    let fa_gate_in_place = gfx12_fa_prep_fp8q
        && layer.wo.k == q_dim
        && if gfx12_sigmoid_rotate_quant_admitted(gpu, &layer.wo, layer.wo.k, n, &epilogue) {
            a4_fa_gate_in_place_enabled()
        } else {
            matches!(layer.wo.dtype, DType::MQ4G256V2 | DType::MQ4G256V2Lloyd)
                && gpu.fp8_stream_active(n, layer.wo.k)
                && !gpu.iu4_producer_quant_fused_active(n, layer.wo.k)
        };
    // A4 attention epilogue: when the gate stays in place for the gfx1201 IU4
    // AWQ sigmoid producer on the slab route, the fp8q attention itself
    // writes that producer's A4 slab (`HIPFIRE_A4_ATTN_EPI=0` keeps the
    // attention + producer pair). Implies the fp8q single-step attend.
    let a4_epi_awq = if fa_gate_in_place
        && gpu.a4_slab_active()
        && gfx12_sigmoid_rotate_quant_admitted(gpu, &layer.wo, layer.wo.k, n, &epilogue)
        && a4_attn_epilogue_enabled()
    {
        layer.wo.awq_scale
    } else {
        None
    };
    attention_input_projection_batched(
        gpu,
        layer,
        config,
        pbs,
        n,
        dim,
        q8_wmma_arch,
        chain_verify,
    )?;

    attention_prepare_batched(
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
        chain_verify,
        gfx12_fa_prep,
        gfx12_fa_prep_fp8q,
        fa_gate_in_place,
        tap,
    )?;

    // 6–7. Batched KV write + flash attention (via dispatch). Split out of
    // `attention_prepare_batched` (F2) so a chunk pair can run both
    // halves' prep first and share one merged attend step; called here for
    // the single-chunk path in the original position.
    let a4_epi = match a4_epi_awq {
        Some(awq) => Some(attention_attend_a4_batched(
            gpu,
            config,
            pbs,
            s,
            kv_cache,
            n,
            start_pos,
            max_ctx_len,
            ctx,
            layer_idx,
            layer.wo.k,
            awq,
        )?),
        None => {
            attention_attend_batched(
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
                gfx12_fa_prep_fp8q,
            )?;
            None
        }
    };
    attention_output_projection_batched(
        gpu,
        layer,
        pbs,
        n,
        q8_wmma_arch,
        arch_has_wmma,
        epilogue,
        chain_verify,
        fa_gate_in_place,
        a4_epi,
    )?;

    Ok(())
}

/// QKV projection (all dtype arms, including PARO's per-weight Givens
/// rotation) + deinterleave/Q-norm/K-norm + TriAttention tap + RoPE of a
/// full-attention layer whose FFN is a routed MoE. The paired-chunk path runs
/// this per half, one merged attend step, then
/// [`attention_moe_output_projection_batched`] per half.
#[allow(clippy::too_many_arguments)]
pub fn attention_moe_prepare_batched(
    gpu: &mut Gpu,
    layer: &AttentionLayerWeights<'_>,
    config: &HybridDims,
    pbs: &AttentionBatchScratch<'_>,
    n: usize,
    dim: usize,
    compact_offset: usize,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    layer_idx: usize,
    tap: Option<AttentionTap<'_>>,
) -> HipResult<()> {
    // Batched MoE FA layer. FA body is the same as FullAttn
    // (rmsnorm + qkv + deinterleave + q/k norm + RoPE +
    // kv_write + attention + sigmoid_mul + wo+residual);
    // only the FFN differs. Duplicated inline — will be
    // consolidated with the dense FA batched body once the
    // MoE path is proven byte-exact.
    // This body is unreachable for MQ3 / MQ3-Lloyd weights —
    // the upstream `mq3_in_moe` guard at the top of
    // `forward_prefill_batch_with_pbs` rejects any MoE layer
    // with MQ3/Lloyd-MQ3 weights anywhere (attention OR FFN),
    // mirroring the captured-path guard at line 3367+. So
    // `layer.wq.dtype` is restricted to MQ4G256 / HFQ4G256
    // / MQ6G256 / HFQ6G256 here. Adding MQ3 to the matcher AND
    // the QKV dispatch is insufficient — the wo path below
    // (line 5320) is hardcoded MQ4 too — so the all-or-nothing
    // wiring lives in a separate PR (see followup issue).
    let qkv_is_mq = matches!(
        layer.wq.dtype,
        DType::MQ4G256
            | DType::MQ4G256V2
            | DType::MQ4CG256
            | DType::MQ6G256
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256V2
            | DType::MQ2G256V2
    );
    let qkv_is_6bit = matches!(layer.wq.dtype, DType::MQ6G256 | DType::HFQ6G256);
    let qkv_is_q8 = matches!(layer.wq.dtype, DType::Q8_0);
    // TQ2G128/BQ1G128 have no fused qkvza/gate_up/qkv kernel, so they take
    // the same UNFUSED plain-GEMM strategy as Q8 rather than falling through
    // to the HFQ4 arm, which would read these packed blocks at the wrong
    // stride and produce fluent-but-wrong tokens.
    let qkv_is_lowbit = matches!(layer.wq.dtype, DType::TQ2G128 | DType::BQ1G128);
    // qt=52 has no MoE-batched kernels (grouped/indexed LUT variants don't
    // exist): refuse loudly here rather than falling through to a uniform
    // fused key. Per-token decode serves Lloyd-MoE via generic dispatch paths.
    if matches!(layer.wq.dtype, DType::MQ4G256V2Lloyd) {
        return Err(HipError::new(
            0,
            "batch_chunk_full_attn_moe: MQ4G256V2Lloyd attention weights have no MoE-batched prefill kernel — refusing (per-token fallback serves this layer)",
        ));
    }
    // Phase 1.6 (PARO FullAttnMoe): wq/wk/wv are ParoQ4G128
    // (each with its own Givens rotation tables). The fused-QKV
    // kernels can't handle this — they assume one shared
    // rotation. Unfused 3-way dispatch (rotate + gemm_hfq4g128
    // per projection) matches the LA QKVZA Phase 1.5 pattern.
    let qkv_is_paro = matches!(layer.wq.dtype, DType::ParoQ4G128);
    // Fused QKV requires uniform dtype — see issue #249 for
    // the dense FA variant. Gate the same way here.
    let q8_wmma_arch = q8_prefill_wmma_enabled(gpu);
    let qkv_same_dtype = layer.wk.dtype == layer.wq.dtype && layer.wv.dtype == layer.wq.dtype;

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
        let paro_wq = layer.wq.rotation.as_ref().unwrap_or_else(|| {
            panic!("ParoQ4G128 wq missing paro metadata at FA layer {layer_idx}")
        });
        let paro_wk = layer.wk.rotation.as_ref().unwrap_or_else(|| {
            panic!("ParoQ4G128 wk missing paro metadata at FA layer {layer_idx}")
        });
        let paro_wv = layer.wv.rotation.as_ref().unwrap_or_else(|| {
            panic!("ParoQ4G128 wv missing paro metadata at FA layer {layer_idx}")
        });
        // wq
        gpu.givens_rotate_to(
            &pbs.x_norm_batch,
            &pbs.x_rot_batch,
            &paro_wq.pairs,
            &paro_wq.theta,
            &paro_wq.scales,
            n,
            dim,
            paro_wq.krot,
        )?;
        run_plain_gemm_key(
            gpu,
            crate::types::KernelKey::GemmHfq4G128,
            &layer.wq.buf,
            layer.wq.dtype,
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
            &paro_wk.scales,
            n,
            dim,
            paro_wk.krot,
        )?;
        run_plain_gemm_key(
            gpu,
            crate::types::KernelKey::GemmHfq4G128,
            &layer.wk.buf,
            layer.wk.dtype,
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
            &paro_wv.scales,
            n,
            dim,
            paro_wv.krot,
        )?;
        run_plain_gemm_key(
            gpu,
            crate::types::KernelKey::GemmHfq4G128,
            &layer.wv.buf,
            layer.wv.dtype,
            &pbs.x_rot_batch,
            &pbs.fa_v_batch,
            layer.wv.m,
            layer.wv.k,
            n,
        )?;
    } else if qkv_is_6bit && qkv_same_dtype {
        run_fused_qkv_key(
            gpu,
            crate::types::KernelKey::FusedQkvHfq6G256,
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
            matches!(layer.wk.dtype, DType::Q8_0) && matches!(layer.wv.dtype, DType::Q8_0),
            "FAMoe qkv Q8 WMMA dispatch requires all of wq/wk/wv to be Q8_0",
        );
        run_fused_qkv_key(
            gpu,
            crate::types::KernelKey::FusedQkvQ8_0,
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
            plain_gemm_key_for(layer.wq.dtype),
            &layer.wq.buf,
            layer.wq.dtype,
            &pbs.x_rot_batch,
            &pbs.fa_q_full_batch,
            layer.wq.m,
            layer.wq.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wk.dtype),
            &layer.wk.buf,
            layer.wk.dtype,
            &pbs.x_rot_batch,
            &pbs.fa_k_batch,
            layer.wk.m,
            layer.wk.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wv.dtype),
            &layer.wv.buf,
            layer.wv.dtype,
            &pbs.x_rot_batch,
            &pbs.fa_v_batch,
            layer.wv.m,
            layer.wv.k,
            n,
        )?;
    } else if qkv_same_dtype {
        run_fused_qkv_key(
            gpu,
            crate::families::fused_qkv::fused_qkv_key_for(layer.wq.dtype),
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
    if let Some(tap) = tap {
        tap(gpu, &pbs.fa_q_batch, &pbs.fa_k_batch, n)?;
    }
    let n_rot = config.n_rot;
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
        compact_offset as i32,
    )?;
    Ok(())
}

/// Gate sigmoid + output projection + residual of a full-attention layer
/// whose FFN is a routed MoE (always the residual epilogue, no fold).
pub fn attention_moe_output_projection_batched(
    gpu: &mut Gpu,
    layer: &AttentionLayerWeights<'_>,
    pbs: &AttentionBatchScratch<'_>,
    n: usize,
    q8_wmma_arch: bool,
    layer_idx: usize,
) -> HipResult<()> {
    gpu.sigmoid_mul_f32(&pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
    // wo + residual. Mirrors the dense FA wo dispatch at
    // qwen35.rs:5591-5623 — Q8 wo skips rotation (un-rotated
    // input expected); MQ4/MQ6 wo apply FWHT(awq_scale-adjusted).
    // MQ6 branch added alongside MQ6_ADMIT (without it, MQ6 wo
    // bytes get fed to gemm_hfq4g256_residual which reads them
    // as 136 B/group HFQ4 layout vs the actual 200 B/group MQ6
    // — catastrophic stride mismatch produces a single-token
    // attractor on AWQ A3B's 4/40 FA layers with MQ6 wo).
    let fa_wo_is_q8 = matches!(layer.wo.dtype, DType::Q8_0);
    // TQ2G128/BQ1G128 have no fused qkvza/gate_up/qkv kernel, so they take
    // the same UNFUSED plain-GEMM strategy as Q8 rather than falling through
    // to the HFQ4 arm, which would read these packed blocks at the wrong
    // stride and produce fluent-but-wrong tokens.
    let fa_wo_is_lowbit = matches!(layer.wo.dtype, DType::TQ2G128 | DType::BQ1G128);
    let fa_wo_is_6bit = matches!(layer.wo.dtype, DType::MQ6G256 | DType::HFQ6G256);
    // Phase 1.6 (PARO FullAttnMoe wo): own Givens rotation table,
    // 72 B/group HFQ4G128 layout. Rotate fa_attn_out_batch by wo's
    // paro into fa_attn_out_rot_batch, then HFQ4G128 GEMM into a
    // scratch, then add into x_batch.
    let fa_wo_is_paro = matches!(layer.wo.dtype, DType::ParoQ4G128);
    // T-B: fused rotate+quantize (always-Residual here — no epilogue param).
    let iu4_wo_prep =
        try_iu4_rotate_prepared_no_epilogue(gpu, &layer.wo, &pbs.fa_attn_out_batch, layer.wo.k, n)?;
    let fa_wo_input = if iu4_wo_prep.is_some() {
        &pbs.fa_attn_out_rot_batch
    } else if fa_wo_is_q8 {
        &pbs.fa_attn_out_batch
    } else if fa_wo_is_paro {
        let paro_wo = layer.wo.rotation.as_ref().unwrap_or_else(|| {
            panic!("ParoQ4G128 wo missing paro metadata at FA layer {layer_idx}")
        });
        gpu.givens_rotate_to(
            &pbs.fa_attn_out_batch,
            &pbs.fa_attn_out_rot_batch,
            &paro_wo.pairs,
            &paro_wo.theta,
            &paro_wo.scales,
            n,
            layer.wo.k,
            paro_wo.krot,
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
            crate::types::KernelKey::GemmHfq6G256Residual,
            &layer.wo.buf,
            layer.wo.dtype,
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
            crate::types::KernelKey::GemmQ8_0ResidualWmma,
            &layer.wo.buf,
            layer.wo.dtype,
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
            plain_gemm_key_for(layer.wo.dtype),
            &layer.wo.buf,
            layer.wo.dtype,
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
            crate::types::KernelKey::GemmHfq4G128,
            &layer.wo.buf,
            layer.wo.dtype,
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
            crate::families::gemm::residual_gemm_key_for(layer.wo.dtype),
            &layer.wo.buf,
            layer.wo.dtype,
            fa_wo_input,
            &pbs.x_batch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    }
    Ok(())
}
