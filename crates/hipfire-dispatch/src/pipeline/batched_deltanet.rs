// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched (prefill / multi-row verify) gated DeltaNet linear attention:
//! qkv/z/beta/alpha projection, causal conv + gates, the gated delta
//! recurrence (sequential, chunk-scan, tree and tape variants), gated norm and
//! the output projection. Moved verbatim from the Qwen3.5 prefill path; the
//! views carry the field names of the architecture-owned values they borrow.

#![allow(clippy::too_many_arguments)]

use crate::families::gemv::WeightRef;
use crate::ops::delta_net::StateQuant;
use crate::pipeline::batched::*;
use crate::pipeline::hybrid::HybridDims;
use hip_bridge::{HipError, HipResult};
use rdna_compute::gdn_stream::{
    GdnAppendSchedule, GdnCarryGeometry, GdnCarryView as ScanCarryView, GdnLiveState,
    GdnPrefillPlan, GdnQ8Planes, GdnScanAlgorithm, GdnSeqInputs, GdnStreamConsts, GdnStreamInput,
    GdnStreamMeta, GdnStreamRaw, GdnStreamScratch, GdnStreamWorkspace, GdnTensorFormat,
};
use rdna_compute::norm::GdnScanOut;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::cell::{Cell, RefCell};

/// Resident weights of one gated DeltaNet sublayer.
#[derive(Clone, Copy)]
pub struct DeltaNetLayerView<'a> {
    pub attn_norm: &'a GpuTensor,
    pub wqkv: WeightRef<'a>,
    pub wz: WeightRef<'a>,
    pub w_alpha: WeightRef<'a>,
    pub w_beta: WeightRef<'a>,
    pub a_log: &'a GpuTensor,
    pub dt_bias: &'a GpuTensor,
    pub conv_weight: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub wo: WeightRef<'a>,
    /// Gate projection of the following dense FFN: the output projection may
    /// fold its residual into that FFN's norm producer.
    pub w_gate: WeightRef<'a>,
}

/// Borrowed per-layer GDN stream carry buffers (CHUNK-CANONICAL §3.2): FP32
/// tile-entry state, tile-entry conv history, and the raw pre-conv QKV and
/// raw alpha/beta of the incomplete microtile. One tensor per DeltaNet layer,
/// each lane-major (`lanes` × per-lane extent of `geometry`).
#[derive(Clone, Copy)]
pub struct GdnCarryView<'a> {
    pub geometry: GdnCarryGeometry,
    pub tile_state: &'a [GpuTensor],
    pub tile_conv: &'a [GpuTensor],
    pub raw_qkv: &'a [GpuTensor],
    pub raw_ab: &'a [GpuTensor],
}

/// Borrowed absolute-position stream state of every DeltaNet layer: recipe,
/// owner generation, per-layer and per-lane host cursors
/// (`meta[layer * lanes + lane]`), and the carry buffers when the resolved scan
/// algorithm needs a microtile larger than one row (`None` otherwise).
#[derive(Clone, Copy)]
pub struct GdnStreamView<'a> {
    pub recipe_id: u64,
    pub generation: u64,
    pub lanes: usize,
    pub meta: &'a [GdnStreamMeta],
    pub carry: Option<GdnCarryView<'a>>,
    /// Resolved canonical prefill plan of every DeltaNet layer (`plans[layer]`);
    /// empty while the owner is unprovisioned (F32/Q4 state, or never prefilled).
    pub plans: &'a [GdnPrefillPlan],
    /// Fixed 64-row edge workspaces of the Scan64 executor, one per scan-output format.
    pub workspaces: &'a [GdnStreamWorkspace],
}

/// Recurrent state of every DeltaNet layer.
#[derive(Clone, Copy)]
pub struct DeltaNetStateView<'a> {
    pub s_matrices: &'a [GpuTensor],
    pub s_scales: &'a [GpuTensor],
    pub conv_states: &'a [GpuTensor],
    pub s_ef_residual: &'a [GpuTensor],
    pub quant: StateQuant,
    pub stream: GdnStreamView<'a>,
}

impl DeltaNetStateView<'_> {
    /// Error-feedback residual of DeltaNet layer `idx`, if Q8 error feedback
    /// is active.
    pub fn ef_residual(&self, idx: usize) -> Option<&GpuTensor> {
        self.s_ef_residual.get(idx)
    }
}

/// DFlash GDN tape: per-layer recurrence inputs captured for rollback replay.
#[derive(Clone, Copy)]
pub struct GdnTapeView<'a> {
    pub max_n: usize,
    pub qkv_dim: usize,
    pub qkv_bufs: &'a [GpuTensor],
    pub alpha_bufs: &'a [GpuTensor],
    pub beta_bufs: &'a [GpuTensor],
}

fn a8_gdn_admitted(
    gpu: &Gpu,
    wo: &WeightRef<'_>,
    n_heads: usize,
    head_dim: usize,
    k: usize,
    n: usize,
    epilogue: &BatchEpilogue<'_>,
) -> bool {
    // Lloyd (qt=52) rides the A4 predicate in place of the symmetric-file
    // requirement (the producer is weight-independent; the twin consumer
    // enforces the same A4 predicate).
    let lloyd_wo = wo.dtype == DType::MQ4G256V2Lloyd && lloyd_mmq_a4_route(gpu, n, k);
    gpu.flags.a8_fused_prod
        && (gpu.a8_prefill_active(n, k) || lloyd_wo)
        && (wo.dtype == DType::MQ4G256V2 || lloyd_wo)
        && matches!(epilogue, BatchEpilogue::Residual)
        && head_dim == 128
        && n_heads * head_dim == k
}

#[allow(clippy::too_many_arguments)]
fn try_a8_gdn_prepared(
    gpu: &mut Gpu,
    wo: &WeightRef<'_>,
    x: &GpuTensor,
    z: &GpuTensor,
    weight: &GpuTensor,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
    k: usize,
    n: usize,
    epilogue: &BatchEpilogue<'_>,
) -> HipResult<Option<rdna_compute::Int8MmqPrepared>> {
    if !a8_gdn_admitted(gpu, wo, n_heads, head_dim, k, n, epilogue) {
        return Ok(None);
    }
    let reservation = gpu.reserve_int8_mmq(k, n)?;
    gpu.gated_norm_rotate_mq_i8_batched(
        x,
        z,
        weight,
        wo.awq_scale,
        None,
        reservation,
        n_heads,
        head_dim,
        eps,
        k,
        n,
    )
    .map(Some)
}

pub fn gfx12_gdn_quant_fused_admitted(
    gpu: &Gpu,
    wo: &WeightRef<'_>,
    n_heads: usize,
    head_dim: usize,
    k: usize,
    n: usize,
    epilogue: &BatchEpilogue<'_>,
) -> bool {
    wo.dtype == DType::MQ4G256V2
        && matches!(epilogue, BatchEpilogue::Residual)
        && head_dim == 128
        && n_heads * head_dim == k
        && gpu.iu4_producer_quant_fused_active(n, k)
}

/// gfx1201 slices-4: gated RMSNorm + FWHT + `block_i4_128` IU4 producer for
/// the LA post-GDN `wo` input. `None` → caller keeps the incumbent
/// gated_norm_f32 + rotate + standalone-quantizer path. Admission is
/// uniform MQ4G256V2 only, Residual-only, `head_dim == 128`,
/// `n_heads * head_dim == k`, `K % 256 == 0`, plus the shared
/// `HIPFIRE_GFX12_PRODUCER_QUANT_FUSED` gate on exact gfx1201. When live,
/// the caller must skip the standalone `gated_norm_f32_batched` store
/// (`dn_normed_batch` has no other reader on the admitted path); the f32
/// `x_rot` store is always written, so every downstream reader is preserved
/// byte-for-byte. `x_fmt` is the storage of `x` (the GDN chunk-scan plane).
#[allow(clippy::too_many_arguments)]
pub fn try_gfx12_gdn_quant_fused_prepared(
    gpu: &mut Gpu,
    wo: &WeightRef<'_>,
    x: &GpuTensor,
    x_fmt: GdnScanOut,
    z: &GpuTensor,
    norm_weight: &GpuTensor,
    x_rot: &GpuTensor,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
    k: usize,
    n: usize,
    epilogue: &BatchEpilogue<'_>,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    if !gfx12_gdn_quant_fused_admitted(gpu, wo, n_heads, head_dim, k, n, epilogue) {
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep = gpu.gated_norm_rotate_mq_i4_gfx12_batched(
        x,
        x_fmt,
        z,
        norm_weight,
        wo.awq_scale,
        None,
        res,
        n_heads,
        head_dim,
        eps,
        k,
        n,
    )?;
    Ok(Some(prep))
}

fn gfx12_fp8_stream_gdn_admitted(
    gpu: &Gpu,
    wo: &WeightRef<'_>,
    n_heads: usize,
    head_dim: usize,
    k: usize,
    n: usize,
) -> bool {
    head_dim == 128
        && n_heads * head_dim == k
        && gfx12_fp8_stream_admitted(gpu, Fp8StreamConsumer::Residual, &[wo.dtype], n, k)
}

/// gfx1201 FP8-stream LA output producer.  Replaces gated_norm + AWQ/FWHT
/// rotate + standalone pack for Lloyd residual consumers. With
/// `HIPFIRE_FP8_PROD_INREG=1`, the FP8 prepared consumer does not read `x_rot`.
/// `x_fmt` is the storage of `x` (the GDN chunk-scan plane).
#[allow(clippy::too_many_arguments)]
fn try_gfx12_fp8_stream_gdn_prepared(
    gpu: &mut Gpu,
    wo: &WeightRef<'_>,
    x: &GpuTensor,
    x_fmt: GdnScanOut,
    z: &GpuTensor,
    norm_weight: &GpuTensor,
    x_rot: &GpuTensor,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
    k: usize,
    n: usize,
) -> HipResult<Option<rdna_compute::Mq4v2Fp8Prepared>> {
    if !gfx12_fp8_stream_gdn_admitted(gpu, wo, n_heads, head_dim, k, n) {
        return Ok(None);
    }
    let prep = gpu.gated_norm_rotate_mq_fp8_gfx12_batched(
        x,
        x_fmt,
        z,
        norm_weight,
        wo.awq_scale,
        x_rot,
        n_heads,
        head_dim,
        eps,
        k,
        n,
    )?;
    Ok(Some(prep))
}

fn gfx11_gdn_quant_fused_admitted(
    gpu: &Gpu,
    wo: &WeightRef<'_>,
    n_heads: usize,
    head_dim: usize,
    k: usize,
    n: usize,
    epilogue: &BatchEpilogue<'_>,
) -> bool {
    wo.dtype == DType::MQ4G256V2
        && matches!(epilogue, BatchEpilogue::Residual)
        && head_dim == 128
        && n_heads * head_dim == k
        && gpu.iu4_gfx11_producer_quant_fused_active(n, k)
}

/// gfx11 slices-4: gated RMSNorm + FWHT + `block_i4_128` IU4 producer for
/// the LA post-GDN `wo` input, under the `_gfx11` entry symbols. `None` →
/// caller keeps the incumbent gated_norm_f32 + rotate +
/// standalone-quantizer path. Admission is uniform MQ4G256V2 only,
/// `head_dim == 128`, `n_heads * head_dim == k`, `K % 256 == 0`, plus the
/// shared `HIPFIRE_GFX11_PRODUCER_QUANT_FUSED` gate on gfx1100/gfx1151.
/// When live, the caller must skip the standalone `gated_norm_f32_batched`
/// store (`dn_normed_batch` has no other reader on the admitted path);
/// the f32 `x_rot` store is always skipped (prepared GEMM is the only
/// consumer), so every surviving downstream reader is preserved
/// byte-for-byte.
#[allow(clippy::too_many_arguments)]
fn try_gfx11_gdn_quant_fused_prepared(
    gpu: &mut Gpu,
    wo: &WeightRef<'_>,
    x: &GpuTensor,
    z: &GpuTensor,
    norm_weight: &GpuTensor,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
    k: usize,
    n: usize,
    epilogue: &BatchEpilogue<'_>,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    if !gfx11_gdn_quant_fused_admitted(gpu, wo, n_heads, head_dim, k, n, epilogue) {
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep = gpu.gated_norm_rotate_mq_i4_gfx11_batched(
        x,
        z,
        norm_weight,
        wo.awq_scale,
        None,
        res,
        n_heads,
        head_dim,
        eps,
        k,
        n,
    )?;
    Ok(Some(prep))
}

/// gfx11 slices-4 epilogue-free twin of [`try_gfx11_gdn_quant_fused_prepared`]
/// for the MoE LA wo site, which has no `epilogue` param and always
/// accumulates residual into x_batch. Same gate minus the Residual check.
#[allow(clippy::too_many_arguments)]
pub fn try_gfx11_gdn_quant_fused_prepared_no_epilogue(
    gpu: &mut Gpu,
    wo: &WeightRef<'_>,
    x: &GpuTensor,
    z: &GpuTensor,
    norm_weight: &GpuTensor,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
    k: usize,
    n: usize,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    if wo.dtype != DType::MQ4G256V2
        || head_dim != 128
        || n_heads * head_dim != k
        || !gpu.iu4_gfx11_producer_quant_fused_active(n, k)
    {
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep = gpu.gated_norm_rotate_mq_i4_gfx11_batched(
        x,
        z,
        norm_weight,
        wo.awq_scale,
        None,
        res,
        n_heads,
        head_dim,
        eps,
        k,
        n,
    )?;
    Ok(Some(prep))
}

/// Prescaffold (behavior-only) extraction for S3-f16-projection-inputs.
///
/// Same statements, same order, same launches as the inlined block.
/// S9-mq4v2-persistent-prologues will issue `try_mq4v2_persistent_prologue`
/// from inside this hook after S3/S4 land.
/// With `gdn` (the admitted chunk-scan route), the gfx1201 FP8 F2 QKVZA or
/// the A4 `_b1` fused QKVZ+beta/alpha projection may also run
/// `gdn_chunk_prep` in its epilogue; the return value says whether it did, in
/// which case the caller runs `gdn_chunk_prep_fixup` instead of the prep.
#[allow(clippy::too_many_arguments)]
pub fn batch_chunk_delta_net_input_projection(
    gpu: &mut Gpu,
    layer: &DeltaNetLayerView<'_>,
    config: &HybridDims,
    pbs: &BatchScratch<'_>,
    n: usize,
    dim: usize,
    q8_wmma_arch: bool,
    fusion: DflashFusionCtx,
    gdn: Option<&rdna_compute::F2GdnTargets<'_>>,
    math: DenseBatchMath,
) -> HipResult<bool> {
    let _ = fusion;
    let math = math.resolve(gpu, "batch_chunk_delta_net_input_projection")?;
    // IU4 verify keeps the plain SET consumers: the fused QKVZ + GDN-prep
    // epilogue was not row-count checked, and the caller runs the prep.
    let gdn = if math == DenseBatchMath::Iu4Verify {
        None
    } else {
        gdn
    };
    if math == DenseBatchMath::SingletonWmma {
        require_singleton_wmma_mq4v2(
            "batch_chunk_delta_net_input_projection",
            &[
                layer.wqkv.dtype,
                layer.wz.dtype,
                layer.w_beta.dtype,
                layer.w_alpha.dtype,
            ],
        )?;
        if gdn.is_some() {
            return Err(HipError::new(
                0,
                "batch_chunk_delta_net_input_projection: SingletonWmma verify math has no GDN chunk-scan targets",
            ));
        }
        gpu.flush_residual_fold()?;
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
        run_fused_qkvza_key(
            gpu,
            crate::types::KernelKey::FusedQkvzaMq4G256V2VerifyExact,
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
        return Ok(false);
    }
    // S3-f16-projection-inputs fast path: emit exact FP16 directly from the
    // RMSNorm+FWHT producer into `x_rot_f16_batch` and consume it with the
    // F16-direct qkvza GEMM. Saves the `convert_f32_to_f16` launch with
    // bit-identical projection outputs. All four weights must share the
    // exact MQ4G256V2 stride (the fused kernel reads them as same-stride
    // byte arrays); anything else stays on the pre-change path.
    if mq_f16_projection_fast_route(gpu, fusion, n, dim)
        && layer.wqkv.dtype == DType::MQ4G256V2
        && layer.wz.dtype == DType::MQ4G256V2
        && layer.w_beta.dtype == DType::MQ4G256V2
        && layer.w_alpha.dtype == DType::MQ4G256V2
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
        gpu.gemm_qkvza_mq4g256v2_wmma_f16(
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
        )?;
        return Ok(false);
    }
    let is_mq = matches!(
        layer.wqkv.dtype,
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
    let is_6bit = matches!(layer.wqkv.dtype, DType::MQ6G256 | DType::HFQ6G256);
    let is_mq3 = matches!(layer.wqkv.dtype, DType::MQ3G256);
    let is_mq3_lloyd = matches!(layer.wqkv.dtype, DType::MQ3G256Lloyd);
    let is_fp4 = matches!(layer.wqkv.dtype, DType::HFP4G32 | DType::MFP4G32);
    let is_q8 = matches!(layer.wqkv.dtype, DType::Q8_0);
    // TQ2G128/BQ1G128 have no fused qkvza/gate_up/qkv kernel, so they take
    // the same UNFUSED plain-GEMM strategy as Q8 rather than falling through
    // to the HFQ4 arm, which would read these packed blocks at the wrong
    // stride and produce fluent-but-wrong tokens.
    // GGUF-native weights (`DType::Ggml*`) take the same unfused plain-GEMM
    // strategy (dequantise + F16 / F32 GEMM inside `run_plain_gemm_key`).
    let is_lowbit =
        matches!(layer.wqkv.dtype, DType::TQ2G128 | DType::BQ1G128) || layer.wqkv.dtype.is_ggml();
    // qt=52 re-arm anchor: all four LA projections Lloyd → FP8-LUT launcher.
    // Mixed Lloyd/uniform is a fail-closed HipError at the dispatch arm.
    let is_mq4v2_lloyd = all_mq4v2_lloyd(&[
        layer.wqkv.dtype,
        layer.wz.dtype,
        layer.w_beta.dtype,
        layer.w_alpha.dtype,
    ]);

    // Batched rmsnorm (+ FWHT for MQ) for the LA preamble.
    // x_batch / x_rot_batch are [N × dim] contiguous. For HFQ
    // we reuse x_rot_batch as the "normed, unrotated" output
    // so the subsequent GEMM can read it the same way.
    // MQ4V2 beta/alpha can be appended to Z at load time. The producer
    // emits IU4 only when its SET consumes all three projections; the F32 X
    // is otherwise still required by the small-M MW4 tail.
    let fold_betaalpha = [
        layer.wqkv.dtype,
        layer.wz.dtype,
        layer.w_beta.dtype,
        layer.w_alpha.dtype,
    ] == [DType::MQ4G256V2; 4]
        && layer.wz.buf.byte_size()
            == gpu.mq4v2_fold_betaalpha_padded_m(layer.wz.m)
                * (layer.wz.k / 256)
                * rdna_compute::MQ4V2_GROUP_BYTES
        && gpu.mq4v2_fold_betaalpha_active(
            layer.wqkv.m,
            layer.wz.m,
            layer.w_beta.m,
            layer.w_alpha.m,
            layer.wqkv.k,
            n,
        );
    let a8_prep = if [
        layer.wqkv.dtype,
        layer.wz.dtype,
        layer.w_beta.dtype,
        layer.w_alpha.dtype,
    ] == [DType::MQ4G256V2; 4]
        || (is_mq4v2_lloyd && lloyd_mmq_a4_route(gpu, n, layer.wqkv.k))
    {
        try_a8_rmsnorm_prepared(
            gpu,
            &pbs.x_batch,
            &layer.attn_norm,
            &layer.wqkv,
            dim,
            config.norm_eps,
            n,
        )?
    } else {
        None
    };
    let mut iu4_prep: Option<rdna_compute::Int4MmqPrepared> = None;
    let mut fp8_prep: Option<rdna_compute::Mq4v2Fp8Prepared> = None;
    if is_mq && a8_prep.is_none() {
        // Folded Z no longer needs the F32 sidecar for MW4 or its f16 convert.
        iu4_prep = try_iu4_rmsnorm_prepared(
            gpu,
            &pbs.x_batch,
            &layer.attn_norm,
            &layer.wqkv,
            &pbs.x_rot_batch,
            dim,
            config.norm_eps,
            n,
            !fold_betaalpha,
        )?;
        if iu4_prep.is_none()
            && mq3v25_a4_consumers_agree(
                gpu,
                layer.wqkv.dtype,
                &[layer.wz.dtype, layer.w_beta.dtype, layer.w_alpha.dtype],
            )
        {
            // gfx1201 slices-2: fused rmsnorm+quant producer (bit-identical).
            iu4_prep = try_gfx12_rmsnorm_quant_fused_prepared(
                gpu,
                &pbs.x_batch,
                &layer.attn_norm,
                &layer.wqkv,
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
                &layer.wqkv,
                Fp8StreamConsumer::Qkvza,
                &[
                    layer.wqkv.dtype,
                    layer.wz.dtype,
                    layer.w_beta.dtype,
                    layer.w_alpha.dtype,
                ],
                &pbs.x_rot_batch,
                dim,
                config.norm_eps,
                n,
            )?;
        }
        if iu4_prep.is_none() && fp8_prep.is_none() {
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
    } else if !is_mq {
        gpu.rmsnorm_batched(
            &pbs.x_batch,
            &layer.attn_norm,
            &pbs.x_rot_batch,
            n,
            dim,
            config.norm_eps,
        )?;
    }

    require_iu4_verify_prep(
        math,
        "batch_chunk_delta_net_input_projection",
        iu4_prep.is_some(),
    )?;
    // Batched 4-way LA projection (wqkv + wz + w_beta + w_alpha).
    if let Some(prep) = &a8_prep {
        let xq = gpu.int8_mmq_prepared_ptr(prep, layer.wqkv.k, n)?;
        if is_mq4v2_lloyd {
            // Lloyd twin: same prepared int8 X, per-tensor codebooks.
            gpu.gemm_mq4g256v2_mmq_set_prequant_i8_lloyd(
                &layer.wqkv.buf,
                xq,
                &pbs.dn_qkv_batch,
                layer.wqkv.m,
                layer.wqkv.k,
                n,
                lloyd_c16_or_fail(&layer.wqkv, "batch_chunk_delta_net_input_projection")?,
            )?;
            gpu.gemm_mq4g256v2_mmq_set_prequant_i8_lloyd(
                &layer.wz.buf,
                xq,
                &pbs.dn_z_batch,
                layer.wz.m,
                layer.wz.k,
                n,
                lloyd_c16_or_fail(&layer.wz, "batch_chunk_delta_net_input_projection")?,
            )?;
            gpu.gemm_mq4g256v2_mmq_set_prequant_i8_lloyd(
                &layer.w_beta.buf,
                xq,
                &pbs.dn_beta_batch,
                layer.w_beta.m,
                layer.w_beta.k,
                n,
                lloyd_c16_or_fail(&layer.w_beta, "batch_chunk_delta_net_input_projection")?,
            )?;
            gpu.gemm_mq4g256v2_mmq_set_prequant_i8_lloyd(
                &layer.w_alpha.buf,
                xq,
                &pbs.dn_alpha_batch,
                layer.w_alpha.m,
                layer.w_alpha.k,
                n,
                lloyd_c16_or_fail(&layer.w_alpha, "batch_chunk_delta_net_input_projection")?,
            )?;
        } else {
            gpu.gemm_mq4g256v2_mmq_set_prequant_i8(
                &layer.wqkv.buf,
                xq,
                &pbs.dn_qkv_batch,
                layer.wqkv.m,
                layer.wqkv.k,
                n,
            )?;
            gpu.gemm_mq4g256v2_mmq_set_prequant_i8(
                &layer.wz.buf,
                xq,
                &pbs.dn_z_batch,
                layer.wz.m,
                layer.wz.k,
                n,
            )?;
            gpu.gemm_mq4g256v2_mmq_set_prequant_i8(
                &layer.w_beta.buf,
                xq,
                &pbs.dn_beta_batch,
                layer.w_beta.m,
                layer.w_beta.k,
                n,
            )?;
            gpu.gemm_mq4g256v2_mmq_set_prequant_i8(
                &layer.w_alpha.buf,
                xq,
                &pbs.dn_alpha_batch,
                layer.w_alpha.m,
                layer.w_alpha.k,
                n,
            )?;
        }
    } else if let Some(prep) = iu4_prep.as_ref().filter(|_| {
        mq3v25_a4_weight(gpu, layer.wqkv.dtype)
            && [layer.wz.dtype, layer.w_beta.dtype, layer.w_alpha.dtype]
                .iter()
                .all(|&dtype| dtype == layer.wqkv.dtype)
    }) {
        // Symmetric qt49 A4 (gfx1201): four SETs over one producer. It
        // bypasses the MQ4-only beta/alpha fold, the zba SET and the fused
        // QKVZ + `gdn_chunk_prep` epilogue (the caller runs the ordinary prep).
        gpu.gemm_qkvza_mq3v25_iu4_prepared(
            &layer.wqkv.buf,
            &layer.wz.buf,
            &layer.w_beta.buf,
            &layer.w_alpha.buf,
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
    } else if let Some(prep) = iu4_prep
        .as_ref()
        .filter(|_| !mq3v25_a4_weight(gpu, layer.wqkv.dtype))
    {
        if fold_betaalpha {
            gpu.gemm_qkvza_mq4g256v2_wmma_iu4_fold_prepared(
                &layer.wqkv.buf,
                &layer.wz.buf,
                prep,
                &pbs.dn_qkv_batch,
                &pbs.dn_z_fold_batch,
                &pbs.dn_z_batch,
                &pbs.dn_beta_batch,
                &pbs.dn_alpha_batch,
                layer.wqkv.m,
                layer.wz.m,
                layer.wqkv.k,
                n,
            )?;
        } else {
            // gfx1201 A4: one `_b1` launch over QKV and the load-time Z fold
            // whose q/k/v tiles also run `gdn_chunk_prep` in the epilogue.
            if let Some(targets) = gdn {
                if gpu.gemm_qkvza_mq4g256v2_iu4_gdn_prepared(
                    &layer.wqkv.buf,
                    &layer.wz.buf,
                    prep,
                    &pbs.dn_qkv_batch,
                    &pbs.dn_z_batch,
                    &pbs.dn_beta_batch,
                    &pbs.dn_alpha_batch,
                    [layer.wqkv.m, layer.wz.m, layer.w_beta.m, layer.w_alpha.m],
                    layer.wqkv.k,
                    n,
                    targets,
                )? {
                    return Ok(true);
                }
            }
            // gfx1201 verify (n <= 128): one SET over the Z fold + a copy
            // kernel instead of separate Z / beta / alpha SETs; other shapes
            // and archs run the unchanged four-GEMM sequence.
            gpu.gemm_qkvza_mq4g256v2_wmma_iu4_prepared_zba(
                &layer.wqkv.buf,
                &layer.wz.buf,
                &layer.w_beta.buf,
                &layer.w_alpha.buf,
                &pbs.x_rot_batch,
                prep,
                &pbs.dn_qkv_batch,
                &pbs.dn_z_fold_batch,
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
    } else if is_6bit {
        run_fused_qkvza_key(
            gpu,
            crate::types::KernelKey::FusedQkvzaHfq6G256,
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
            matches!(layer.wz.dtype, DType::Q8_0)
                && matches!(layer.w_beta.dtype, DType::Q8_0)
                && matches!(layer.w_alpha.dtype, DType::Q8_0),
            "LA qkvza Q8 WMMA dispatch requires all of wqkv/wz/w_beta/w_alpha to be Q8_0",
        );
        run_fused_qkvza_key(
            gpu,
            crate::types::KernelKey::FusedQkvzaQ8_0,
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
            plain_gemm_key_for(layer.wqkv.dtype),
            &layer.wqkv.buf,
            layer.wqkv.dtype,
            &pbs.x_rot_batch,
            &pbs.dn_qkv_batch,
            layer.wqkv.m,
            layer.wqkv.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wz.dtype),
            &layer.wz.buf,
            layer.wz.dtype,
            &pbs.x_rot_batch,
            &pbs.dn_z_batch,
            layer.wz.m,
            layer.wz.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.w_beta.dtype),
            &layer.w_beta.buf,
            layer.w_beta.dtype,
            &pbs.x_rot_batch,
            &pbs.dn_beta_batch,
            layer.w_beta.m,
            layer.w_beta.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.w_alpha.dtype),
            &layer.w_alpha.buf,
            layer.w_alpha.dtype,
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
            crate::types::KernelKey::FusedQkvzaMq3G256Lloyd,
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
            crate::types::KernelKey::FusedQkvzaHfq3G256,
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
            crate::types::KernelKey::FusedQkvzaHfp4G32,
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
        // qt=52: all four projections Lloyd. Small-N gfx1100 → F16-BT-LUT;
        // N>=128 or other gfx11 → MMQ-LUT; gfx12 FP8-LUT.
        if lloyd_f16_bt_route(gpu, n) {
            gpu.gemm_qkvza_mq4g256v2_wmma_gfx11_lloyd(
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
                [
                    lloyd_f16_or_fail(&layer.wqkv, "batch_chunk_delta_net_input_projection")?,
                    lloyd_f16_or_fail(&layer.wz, "batch_chunk_delta_net_input_projection")?,
                    lloyd_f16_or_fail(&layer.w_beta, "batch_chunk_delta_net_input_projection")?,
                    lloyd_f16_or_fail(&layer.w_alpha, "batch_chunk_delta_net_input_projection")?,
                ],
            )?;
        } else if lloyd_mmq_lut_route(gpu) {
            if lloyd_mmq_a4_route(gpu, n, layer.wqkv.k) {
                // Staged i8 twin: one int8 quantize of X, then the existing
                // single-matrix SET per projection — no pad-128 allocs.
                // w_beta/w_alpha (M=48) run as two masked SETs for now.
                let xq = gpu.ensure_int8_mmq_x(&pbs.x_rot_batch, n, layer.wqkv.k)?;
                gpu.gemm_mq4g256v2_mmq_set_prequant_i8_lloyd(
                    &layer.wqkv.buf,
                    xq,
                    &pbs.dn_qkv_batch,
                    layer.wqkv.m,
                    layer.wqkv.k,
                    n,
                    lloyd_c16_or_fail(&layer.wqkv, "batch_chunk_delta_net_input_projection")?,
                )?;
                gpu.gemm_mq4g256v2_mmq_set_prequant_i8_lloyd(
                    &layer.wz.buf,
                    xq,
                    &pbs.dn_z_batch,
                    layer.wz.m,
                    layer.wz.k,
                    n,
                    lloyd_c16_or_fail(&layer.wz, "batch_chunk_delta_net_input_projection")?,
                )?;
                gpu.gemm_mq4g256v2_mmq_set_prequant_i8_lloyd(
                    &layer.w_beta.buf,
                    xq,
                    &pbs.dn_beta_batch,
                    layer.w_beta.m,
                    layer.w_beta.k,
                    n,
                    lloyd_c16_or_fail(&layer.w_beta, "batch_chunk_delta_net_input_projection")?,
                )?;
                gpu.gemm_mq4g256v2_mmq_set_prequant_i8_lloyd(
                    &layer.w_alpha.buf,
                    xq,
                    &pbs.dn_alpha_batch,
                    layer.w_alpha.m,
                    layer.w_alpha.k,
                    n,
                    lloyd_c16_or_fail(&layer.w_alpha, "batch_chunk_delta_net_input_projection")?,
                )?;
            } else {
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
            }
        } else if let Some(prep) = &fp8_prep {
            // gfx1201 FP8-stream: the producer already emitted the fp8
            // pre-pass planes; consume them directly, no pack launch.
            gpu.gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2_fp8_prepared_lloyd(
                &layer.wqkv.buf,
                &layer.wz.buf,
                &layer.w_beta.buf,
                &layer.w_alpha.buf,
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
                lloyd_e4m3_or_fail(&layer.wqkv, "batch_chunk_delta_net_input_projection")?,
                lloyd_e4m3_or_fail(&layer.wz, "batch_chunk_delta_net_input_projection")?,
                lloyd_e4m3_or_fail(&layer.w_beta, "batch_chunk_delta_net_input_projection")?,
                lloyd_e4m3_or_fail(&layer.w_alpha, "batch_chunk_delta_net_input_projection")?,
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
    } else if matches!(layer.wqkv.dtype, DType::MQ4G256V2Lloyd)
        || matches!(layer.wz.dtype, DType::MQ4G256V2Lloyd)
        || matches!(layer.w_beta.dtype, DType::MQ4G256V2Lloyd)
        || matches!(layer.w_alpha.dtype, DType::MQ4G256V2Lloyd)
    {
        // Mixed Lloyd/uniform LA projections: no kernel reads mixed codebooks.
        return Err(hip_bridge::HipError::new(
            0,
            "batch_chunk_delta_net_input_projection: mixed MQ4G256V2Lloyd/uniform LA projections — refusing (quantize all four or none)",
        ));
    } else if layer.wqkv.dtype == DType::MQ4G256V2
        && layer.wz.dtype == DType::MQ4G256V2
        && layer.w_beta.dtype == DType::MQ4G256V2
        && layer.w_alpha.dtype == DType::MQ4G256V2
        // Producer admission already required the QKVZA family to take FP8
        // (`gfx12_fp8_stream_admitted`); fp8_prep must be consumed here.
        && fp8_prep.is_some()
    {
        // gfx1201 FP8-stream (uniform): the producer already emitted the
        // fp8 pre-pass planes; consume them directly with the launch twin
        // of the family's fp8 route — no pack launch. Same fp8 intercept
        // conditions as the uniform router (iu4 divergence excluded by
        // producer-side ordering: fp8_prep implies iu4_prep is None).
        if let Some(targets) = gdn {
            if gpu.gemm_qkvza_mq4g256v2_fp8_gdn_prepared(
                &layer.wqkv.buf,
                &layer.wz.buf,
                &layer.w_beta.buf,
                &layer.w_alpha.buf,
                fp8_prep.as_ref().unwrap(),
                &pbs.dn_qkv_batch,
                &pbs.dn_z_batch,
                &pbs.dn_beta_batch,
                &pbs.dn_alpha_batch,
                [layer.wqkv.m, layer.wz.m, layer.w_beta.m, layer.w_alpha.m],
                layer.wqkv.k,
                n,
                targets,
            )? {
                return Ok(true);
            }
        }
        gpu.gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2_fp8_prepared(
            &layer.wqkv.buf,
            &layer.wz.buf,
            &layer.w_beta.buf,
            &layer.w_alpha.buf,
            fp8_prep.as_ref().unwrap(),
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
    } else {
        run_fused_qkvza_key(
            gpu,
            crate::families::fused_qkv::fused_qkvza_key_for(layer.wqkv.dtype),
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
    Ok(false)
}

/// Default-on switch for the singleton chain-verify batched tape copy
/// (`HIPFIRE_VERIFY_TAPE_COPY_BATCH=0` restores three `memcpy_dtod_at_auto`).
fn verify_tape_copy_batch_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_VERIFY_TAPE_COPY_BATCH")
            .map_or(true, |v| v.trim() != "0")
    });
    *ENABLED
}

/// One DeltaNet layer's DFlash tape-capture copies (pre-conv1d qkv rows,
/// post-sigmoid alpha and beta rows of `pbs`'s rows `[0, n)`, into the layer's
/// tape slots at `tape_offset`) as three descriptors for
/// `Gpu::tape_copy_batched`. `pbs` may be a row view of a shared scratch: the
/// source is its first row.
fn gdn_tape_copy_descriptors(
    tape: &GdnTapeView<'_>,
    pbs: &BatchScratch<'_>,
    n_v_heads: usize,
    tape_offset: usize,
    n: usize,
    delta_layer_idx: usize,
) -> [rdna_compute::tape_copy::TapeCopy; 3] {
    use rdna_compute::tape_copy::TapeCopy;
    let qkv_row_bytes = tape.qkv_dim * 4;
    let alpha_row_bytes = n_v_heads * 4;
    let off_qkv = tape_offset * qkv_row_bytes;
    let off_a = tape_offset * alpha_row_bytes;
    let copy_qkv = n * qkv_row_bytes;
    let copy_a = n * alpha_row_bytes;
    [
        TapeCopy::new(
            &tape.qkv_bufs[delta_layer_idx].buf,
            off_qkv,
            &pbs.dn_qkv_batch.buf,
            0,
            copy_qkv,
        ),
        TapeCopy::new(
            &tape.alpha_bufs[delta_layer_idx].buf,
            off_a,
            &pbs.dn_alpha_batch.buf,
            0,
            copy_a,
        ),
        TapeCopy::new(
            &tape.beta_bufs[delta_layer_idx].buf,
            off_a,
            &pbs.dn_beta_batch.buf,
            0,
            copy_a,
        ),
    ]
}

/// Queue the three descriptors of [`gdn_tape_copy_descriptors`] on `sink`.
/// Shared by the per-request pre-GDN sink branch and the lane-batched
/// multi-request route, so both queue identical descriptors.
pub fn push_gdn_tape_copies(
    sink: &mut Vec<rdna_compute::tape_copy::TapeCopy>,
    tape: &GdnTapeView<'_>,
    pbs: &BatchScratch<'_>,
    n_v_heads: usize,
    tape_offset: usize,
    n: usize,
    delta_layer_idx: usize,
) {
    sink.extend(gdn_tape_copy_descriptors(
        tape,
        pbs,
        n_v_heads,
        tape_offset,
        n,
        delta_layer_idx,
    ));
}

/// The pre-GDN fused sigmoid(beta) + alpha gate over rows `[0, n)` of `pbs`
/// (row-local: each row's bytes are independent of `n`). Shared by the
/// per-request pre-GDN and the lane-batched multi-request route.
pub fn gdn_sigmoid_alpha_gate(
    gpu: &mut Gpu,
    layer: &DeltaNetLayerView<'_>,
    pbs: &BatchScratch<'_>,
    n_v_heads: usize,
    n: usize,
) -> HipResult<()> {
    gpu.fused_sigmoid_alpha_gate_f32_batched(
        &pbs.dn_beta_batch,
        &pbs.dn_alpha_batch,
        &layer.dt_bias,
        &layer.a_log,
        n_v_heads,
        n,
    )
}

/// The GQA (`n_key_heads < n_v_heads`) fused L2-norm(Q) + scale + L2-norm(K)
/// + repeat-interleave over rows `[0, n)` of `pbs` (row-local). Shared by the
/// per-request pre-GDN and the lane-batched multi-request route.
pub fn gdn_qk_norm_interleave(
    gpu: &mut Gpu,
    config: &HybridDims,
    pbs: &BatchScratch<'_>,
    n_v_heads: usize,
    hd: usize,
    n: usize,
) -> HipResult<()> {
    let ratio = n_v_heads / config.linear_key_heads;
    gpu.fused_qk_l2_norm_scale_interleave_f32_batched(
        &pbs.dn_q_raw_batch,
        &pbs.dn_k_raw_batch,
        &pbs.dn_q_batch,
        &pbs.dn_k_batch,
        config.linear_key_heads,
        ratio,
        hd,
        1.0 / (hd as f32).sqrt(),
        config.norm_eps,
        n,
    )
}

#[allow(clippy::too_many_arguments)]
/// Prescaffold (behavior-only) extraction for S5-gdn-pre-tape-fusion.
///
/// Same statements, same order, same launches as the inlined block.
/// Returns `tree_parents` so the caller's statement/launch order is unchanged.
pub fn batch_chunk_delta_net_pre_gdn<'a>(
    gpu: &mut Gpu,
    layer: &DeltaNetLayerView<'_>,
    config: &HybridDims,
    pbs: &BatchScratch<'_>,
    dn_state: &DeltaNetStateView<'_>,
    n: usize,
    k_dim: usize,
    v_dim: usize,
    n_v_heads: usize,
    hd: usize,
    batch_semantics: BatchSemantics<'_>,
    tree_verify: Option<TreeVerifyCtx<'a>>,
    gdn_tape: Option<&GdnTapeView<'_>>,
    tape_offset: usize,
    delta_layer_idx: usize,
    fusion: DflashFusionCtx,
    tape_sink: Option<&mut Vec<rdna_compute::tape_copy::TapeCopy>>,
) -> HipResult<Option<&'a GpuTensor>> {
    let _ = fusion;
    // S5-gdn-pre-tape-fusion fast path: one launch for sigmoid(alpha/beta) +
    // tape writes + conv + QK norm/interleave. Chain verify is sequential
    // with no tree parents, so success returns None. Every failed predicate
    // (kill switch, non-sequential batch, non-GQA route, tape absence or
    // overflow, ineligible shapes/arch) runs the pre-change sequence below
    // launch-for-launch.
    if fusion.is_chain_verify()
        && !gpu.flags.gdn_pre_fuse_off
        && matches!(batch_semantics, BatchSemantics::Sequential)
        && config.linear_key_heads < n_v_heads
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
                    config.linear_key_heads,
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
    // (`batched_moe::execute_deltanet_moe_layer_batched`) keeps the 3-launch
    // sequence. q_raw/k_raw stores are kept by the fused kernel (v1).
    {
        let no_tree_parents = tree_verify
            .as_ref()
            .and_then(|c| c.parent_indices)
            .is_none();
        let n_key_heads = config.linear_key_heads;
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
    gdn_sigmoid_alpha_gate(gpu, layer, pbs, n_v_heads, n)?;

    // DFlash tape capture: snap pre-conv1d qkv + post-sigmoid α/β
    // for this layer into the per-layer tape slots. The next LA
    // layer's fused_qkvza / fused_sigmoid_alpha_gate will overwrite
    // dn_qkv_batch / dn_{alpha,beta}_batch, so capture must happen
    // now (after sigmoid_alpha_gate, before conv1d consumes qkv) —
    // or, with a `tape_sink` (the multi-request verify trunk), be
    // deferred: the three copies are queued and the caller issues
    // one batched launch for every lane of the layer before the
    // next layer's producers run. Nothing on this path writes the
    // source rows after the sigmoid, so the tape bytes are equal.
    if let Some(tape) = gdn_tape.as_ref() {
        if let Some(sink) = tape_sink {
            push_gdn_tape_copies(sink, tape, pbs, n_v_heads, tape_offset, n, delta_layer_idx);
        } else if fusion.is_chain_verify()
            && matches!(batch_semantics, BatchSemantics::Sequential)
            && tree_verify
                .as_ref()
                .and_then(|c| c.parent_indices)
                .is_none()
            && verify_tape_copy_batch_enabled()
        {
            // Singleton chain verify: one batched launch for the same three
            // descriptors, issued at the original copy point (before conv).
            let copies =
                gdn_tape_copy_descriptors(tape, pbs, n_v_heads, tape_offset, n, delta_layer_idx);
            gpu.tape_copy_batched(&copies)?;
        } else {
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
    if config.linear_key_heads < n_v_heads {
        gdn_qk_norm_interleave(gpu, config, pbs, n_v_heads, hd, n)?;
    } else {
        // n_key_heads == n_v_heads → no replication; keep the
        // original sequence (norm in place, then memcpy).
        gpu.fused_qk_l2_norm_scale_f32_batched(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            config.linear_key_heads,
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

/// Producer that [`batch_chunk_delta_net_output_projection`] selects for the
/// LA `out` plane (same selection order). Only the gfx1201 FP8-stream and A4
/// IU4 producers have `_xbf16` twins.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GdnOutReader {
    Fp8Stream,
    Gfx12Iu4,
    F32Only,
}

fn gdn_out_reader(
    gpu: &Gpu,
    layer: &DeltaNetLayerView<'_>,
    config: &HybridDims,
    n: usize,
    n_v_heads: usize,
    epilogue: &BatchEpilogue<'_>,
    fusion: DflashFusionCtx,
) -> GdnOutReader {
    let (wo, hd, k) = (&layer.wo, config.linear_value_dim, layer.wo.k);
    if s4_residual_fast(gpu, fusion, wo.dtype, epilogue, n)
        || a8_gdn_admitted(gpu, wo, n_v_heads, hd, k, n, epilogue)
    {
        GdnOutReader::F32Only
    } else if gfx12_gdn_quant_fused_admitted(gpu, wo, n_v_heads, hd, k, n, epilogue) {
        GdnOutReader::Gfx12Iu4
    } else if gfx11_gdn_quant_fused_admitted(gpu, wo, n_v_heads, hd, k, n, epilogue) {
        GdnOutReader::F32Only
    } else if gfx12_fp8_stream_gdn_admitted(gpu, wo, n_v_heads, hd, k, n) {
        GdnOutReader::Fp8Stream
    } else {
        GdnOutReader::F32Only
    }
}

/// Prescaffold (behavior-only) extraction for S4-f16-residual-inputs.
///
/// Same statements, same order, same launches as the inlined block.
/// S9-mq4v2-persistent-prologues will issue `try_mq4v2_persistent_prologue`
/// from inside this hook after S3/S4 land.
/// `x_fmt` is the storage of `dn_attn_out_batch` ([`gdn_scan_out_fmt`]).
pub fn batch_chunk_delta_net_output_projection(
    gpu: &mut Gpu,
    layer: &DeltaNetLayerView<'_>,
    config: &HybridDims,
    pbs: &BatchScratch<'_>,
    n: usize,
    n_v_heads: usize,
    q8_wmma_arch: bool,
    arch_has_wmma: bool,
    epilogue: BatchEpilogue<'_>,
    fusion: DflashFusionCtx,
    x_fmt: GdnScanOut,
    math: DenseBatchMath,
) -> HipResult<()> {
    let math = math.resolve(gpu, "batch_chunk_delta_net_output_projection")?;
    if math == DenseBatchMath::SingletonWmma {
        require_singleton_wmma_mq4v2("batch_chunk_delta_net_output_projection", &[layer.wo.dtype])?;
        if x_fmt != GdnScanOut::F32 || !matches!(&epilogue, BatchEpilogue::Residual) {
            return Err(HipError::new(
                0,
                "batch_chunk_delta_net_output_projection: SingletonWmma verify math requires an F32 GDN scan plane and the Residual epilogue",
            ));
        }
        gpu.gated_norm_f32_batched(
            &pbs.dn_attn_out_batch,
            &pbs.dn_z_batch,
            &layer.norm_weight,
            &pbs.dn_normed_batch,
            n_v_heads,
            config.linear_value_dim,
            config.norm_eps,
            n,
        )?;
        rotate_x_mq_batched_for(
            gpu,
            &layer.wo,
            &pbs.dn_normed_batch,
            &pbs.dn_normed_rot_batch,
            layer.wo.k,
            n,
        )?;
        return dispatch_batched_gemm_epilogue(
            gpu,
            pbs,
            &layer.wo,
            &pbs.dn_normed_rot_batch,
            &epilogue,
            n,
            q8_wmma_arch,
            arch_has_wmma,
            math,
        );
    }
    if x_fmt != GdnScanOut::F32
        && gdn_out_reader(gpu, layer, config, n, n_v_heads, &epilogue, fusion)
            == GdnOutReader::F32Only
    {
        return Err(HipError::new(
            0,
            "bf16 GDN scan plane without an _xbf16 producer",
        ));
    }
    // S4: one gated_norm+FWHT+F16 producer + direct-F16 residual GEMM
    // instead of gated_norm_f32 + mq_rotate_x + convert.
    if s4_residual_fast(gpu, fusion, layer.wo.dtype, &epilogue, n)
        && config.linear_value_dim == 128
        && n_v_heads * config.linear_value_dim == layer.wo.k
    {
        let k = layer.wo.k;
        let m = layer.wo.m;
        if k > 0 && k % 256 == 0 {
            if let Some(awq) = layer.wo.awq_scale {
                if awq.numel() >= k {
                    gpu.gated_norm_rotate_mq_awq_f16_batched(
                        &pbs.dn_attn_out_batch,
                        &pbs.dn_z_batch,
                        &layer.norm_weight,
                        awq,
                        &pbs.dn_normed_rot_f16_batch,
                        n_v_heads,
                        config.linear_value_dim,
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
                    config.linear_value_dim,
                    config.norm_eps,
                    n,
                )?;
                let x_f16 = pbs.dn_normed_rot_f16_batch.sub_offset(0, n * k);
                gpu.gemm_mq4g256v2_residual_wmma_f16(&layer.wo.buf, &x_f16, &pbs.x_batch, m, k, n)?;
                return Ok(());
            }
        }
    }
    // slices-4: whole gated_norm+rotate+quant chain in one producer
    // (`_gfx12` entries on gfx1201, `_gfx11` entries on gfx1100/gfx1151).
    // When live, the standalone gated_norm_f32 store is skipped
    // (`dn_normed_batch` has no other reader on the admitted
    // uniform-MQ4G256V2/Residual/head_dim-128 path); the f32 `x_rot` store
    // is skipped and only the iu4 sidecar is emitted by the fused producer.
    let a8_gdn_prep = try_a8_gdn_prepared(
        gpu,
        &layer.wo,
        &pbs.dn_attn_out_batch,
        &pbs.dn_z_batch,
        &layer.norm_weight,
        n_v_heads,
        config.linear_value_dim,
        config.norm_eps,
        layer.wo.k,
        n,
        &epilogue,
    )?;
    let mut gdn_fused_prep = if a8_gdn_prep.is_none() {
        try_gfx12_gdn_quant_fused_prepared(
            gpu,
            &layer.wo,
            &pbs.dn_attn_out_batch,
            x_fmt,
            &pbs.dn_z_batch,
            &layer.norm_weight,
            &pbs.dn_normed_rot_batch,
            n_v_heads,
            config.linear_value_dim,
            config.norm_eps,
            layer.wo.k,
            n,
            &epilogue,
        )?
    } else {
        None
    };
    if a8_gdn_prep.is_none() && gdn_fused_prep.is_none() {
        gdn_fused_prep = try_gfx11_gdn_quant_fused_prepared(
            gpu,
            &layer.wo,
            &pbs.dn_attn_out_batch,
            &pbs.dn_z_batch,
            &layer.norm_weight,
            n_v_heads,
            config.linear_value_dim,
            config.norm_eps,
            layer.wo.k,
            n,
            &epilogue,
        )?;
    }
    let mut fp8_gdn_prep: Option<rdna_compute::Mq4v2Fp8Prepared> = None;
    if a8_gdn_prep.is_none() && gdn_fused_prep.is_none() {
        fp8_gdn_prep = try_gfx12_fp8_stream_gdn_prepared(
            gpu,
            &layer.wo,
            &pbs.dn_attn_out_batch,
            x_fmt,
            &pbs.dn_z_batch,
            &layer.norm_weight,
            &pbs.dn_normed_rot_batch,
            n_v_heads,
            config.linear_value_dim,
            config.norm_eps,
            layer.wo.k,
            n,
        )?;
    }
    if a8_gdn_prep.is_none() && gdn_fused_prep.is_none() && fp8_gdn_prep.is_none() {
        // Batched gated output norm.
        gpu.gated_norm_f32_batched(
            &pbs.dn_attn_out_batch,
            &pbs.dn_z_batch,
            &layer.norm_weight,
            &pbs.dn_normed_batch,
            n_v_heads,
            config.linear_value_dim,
            config.norm_eps,
            n,
        )?;
    }

    // Batched wo + residual/partial.
    //
    // For MQ weights, the decode path's weight_gemv_residual
    // internally FWHT-rotates dn_normed into mq_x_rot before
    // calling gemv_hfq{4,6}g256_residual (MQ weights are pre-rotated
    // at quant time; math requires dot(rot(W), rot(x)) = dot(W,x)).
    // For HFQ weights no rotation is needed — the activation
    // feeds gemm_hfq{4,6}g256_residual directly.
    let wo_is_mq = matches!(
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
    // T-B: fused rotate+quantize — x_rot feeds only the IU4 residual GEMM.
    // Skipped when the slices-4 GDN producer already emitted both outputs
    // (`dn_normed_batch` is stale there by construction).
    let mut iu4_wo_prep: Option<rdna_compute::Int4MmqPrepared> = None;
    if a8_gdn_prep.is_none() && gdn_fused_prep.is_none() && fp8_gdn_prep.is_none() {
        iu4_wo_prep = try_iu4_rotate_prepared(
            gpu,
            &layer.wo,
            &pbs.dn_normed_batch,
            layer.wo.k,
            n,
            &epilogue,
        )?;
        if iu4_wo_prep.is_none() {
            // gfx1201 slices-3: fused rotate+quant producer (bit-identical).
            iu4_wo_prep = try_gfx12_rotate_quant_fused_prepared(
                gpu,
                &layer.wo,
                &pbs.dn_normed_batch,
                &pbs.dn_normed_rot_batch,
                layer.wo.k,
                n,
                &epilogue,
            )?;
        }
    }
    let wo_input = if a8_gdn_prep.is_some()
        || gdn_fused_prep.is_some()
        || fp8_gdn_prep.is_some()
        || iu4_wo_prep.is_some()
    {
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
    if let Some(prep) = &a8_gdn_prep {
        if layer.wo.dtype == DType::MQ4G256V2Lloyd {
            out_proj_residual_i8_prepared_lloyd(
                gpu,
                pbs,
                &layer.wo,
                &layer.w_gate,
                prep,
                fusion,
                n,
            )?;
        } else {
            out_proj_residual_i8_prepared(gpu, pbs, &layer.wo, &layer.w_gate, prep, fusion, n)?;
        }
    } else if let Some(prep) = gdn_fused_prep.as_ref().or(iu4_wo_prep.as_ref()) {
        out_proj_residual_iu4_prepared(gpu, pbs, &layer.wo, &layer.w_gate, prep, fusion, n)?;
    } else if let Some(prep) = &fp8_gdn_prep {
        dispatch_batched_fp8_lloyd_epilogue(gpu, pbs, &layer.wo, prep, &epilogue, n)?;
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
            math,
        )?;
    }
    Ok(())
}

/// Key/value head width of the GDN chunk-scan kernels.
pub const GDN_CHUNK_HEAD_DIM: usize = 128;
/// Key heads of every chunk-scan layer (the 27B dense route's count).
pub const GDN_CHUNK_KEY_HEADS: usize = 16;
/// Value-head maximum of the chunk-scan kernels (the 27B dense route's count).
pub const GDN_CHUNK_MAX_VALUE_HEADS: usize = 48;

/// Head geometry the generalized GDN chunk-scan kernels run: 16 key heads,
/// 128-wide key and value heads, a 4-tap conv, and 1..=48 value heads that are
/// a multiple of the key heads (so value heads >= key heads).
pub fn gdn_chunk_head_geometry_supported(
    key_heads: usize,
    value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    conv_kernel_dim: usize,
) -> bool {
    key_heads == GDN_CHUNK_KEY_HEADS
        && value_heads > 0
        && value_heads <= GDN_CHUNK_MAX_VALUE_HEADS
        && value_heads % key_heads == 0
        && key_head_dim == GDN_CHUNK_HEAD_DIM
        && value_head_dim == GDN_CHUNK_HEAD_DIM
        && conv_kernel_dim == 4
}

/// Per-forward handle of the canonical ordinary-prefill GDN route
/// (CHUNK-CANONICAL-DESIGN §3.3). Every DeltaNet layer appends its rows at
/// absolute `start_pos` through the layer's frozen plan; outputs are published
/// for the new rows only.
#[derive(Clone, Copy)]
pub struct GdnStreamCall<'a> {
    /// Absolute position of the forward's first row.
    pub start_pos: u64,
    /// Candidate cursor of every DeltaNet layer (`[delta_layer]`), filled by each
    /// layer's append. The owner publishes them only after the whole forward
    /// succeeded on every layer.
    pub candidates: &'a RefCell<Vec<Option<GdnStreamMeta>>>,
    /// Set just before the first device-mutating append so a failure can fail closed.
    pub attempted: &'a Cell<bool>,
}

/// How a batched DeltaNet layer advances its recurrence.
#[derive(Clone, Copy)]
pub enum GdnPrefillRoute<'a> {
    /// Per-call recurrence of decode-shaped, verify, tape and banded forwards.
    Legacy,
    /// Canonical ordinary prefill: absolute-position stream append.
    Stream(GdnStreamCall<'a>),
}

/// Plan, current cursor and carry of one DeltaNet layer of a single-lane stream.
/// With a `call`, a candidate recorded earlier in the same forward (the two halves of a
/// chunk pair append to the same layer in order) is the layer's current cursor.
fn gdn_stream_layer<'a>(
    dn_state: &DeltaNetStateView<'a>,
    layer: usize,
    call: Option<&GdnStreamCall<'_>>,
) -> HipResult<(&'a GdnPrefillPlan, GdnStreamMeta, GdnCarryView<'a>)> {
    let s = dn_state.stream;
    let fail = |m: &str| HipError::new(0, &format!("GDN stream route: {m}"));
    if s.lanes != 1 {
        return Err(fail("ordinary prefill needs a single-lane state"));
    }
    let plan = s
        .plans
        .get(layer)
        .ok_or_else(|| fail("no plan bound for this layer"))?;
    let committed = *s
        .meta
        .get(layer)
        .ok_or_else(|| fail("no cursor for this layer"))?;
    let meta = call
        .and_then(|c| c.candidates.borrow().get(layer).copied().flatten())
        .unwrap_or(committed);
    let carry = s.carry.ok_or_else(|| fail("state owns no carry"))?;
    Ok((plan, meta, carry))
}

/// Scan algorithm the frozen plan of DeltaNet layer `layer` selects.
pub fn gdn_stream_algorithm(
    dn_state: &DeltaNetStateView<'_>,
    layer: usize,
) -> HipResult<GdnScanAlgorithm> {
    gdn_stream_layer(dn_state, layer, None).map(|(plan, _, _)| plan.algorithm)
}

fn gdn_stream_record(call: &GdnStreamCall<'_>, layer: usize, next: GdnStreamMeta) -> HipResult<()> {
    let mut c = call.candidates.borrow_mut();
    match c.get_mut(layer) {
        Some(slot) => {
            *slot = Some(next);
            Ok(())
        }
        None => Err(HipError::new(
            0,
            "GDN stream route: candidate table too small",
        )),
    }
}

/// Byte views of the stream executor's scratch planes over `pbs` for an `n`-row
/// append: q/k (`[n][n_k_heads*128]`), v (`[n][n_v_heads*128]`) f16 planes and
/// the 64-row A blocks. The single construction both the executor's scratch and
/// the fused producer's q/k/v targets use, so they cannot drift.
struct GdnStreamViews {
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    a: GpuTensor,
}

fn gdn_stream_scratch_views(
    pbs: &BatchScratch<'_>,
    n: usize,
    n_k_heads: usize,
    n_v_heads: usize,
) -> HipResult<GdnStreamViews> {
    let fail = |m: String| HipError::new(0, &format!("GDN stream scan: {m}"));
    let raw_view = |tensor: &GpuTensor, need: usize, name: &str| -> HipResult<GpuTensor> {
        if tensor.buf.size() < need {
            return Err(fail(format!(
                "{name} scratch undersized: have {}, need {need}",
                tensor.buf.size()
            )));
        }
        Ok(GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(tensor.buf.as_ptr(), need) },
            shape: vec![need],
            dtype: DType::Raw,
        })
    };
    let q_bytes = n * n_k_heads * GDN_CHUNK_HEAD_DIM * 2;
    let v_bytes = n * n_v_heads * GDN_CHUNK_HEAD_DIM * 2;
    let a_bytes = n.div_ceil(64) * 64 * n_v_heads * 64 * 2;
    Ok(GdnStreamViews {
        q: raw_view(&pbs.dn_q_batch, q_bytes, "q")?,
        k: raw_view(&pbs.dn_k_batch, q_bytes, "k")?,
        v: raw_view(&pbs.dn_v_batch, v_bytes, "v")?,
        a: raw_view(&pbs.dn_q_raw_batch, a_bytes, "A")?,
    })
}

/// Whether the fused A4 producer (+ `gdn_chunk_prep_fixup`) may run for this
/// layer append: Scan64 plan with the BF16 scan plane and the IU4 gfx1201
/// projection recipe, fused prep enabled, supported head geometry, and an
/// append schedule (the exact one `gdn_stream_append` computes) with no head
/// replay and no incomplete final tile. Any other state returns `false` and the
/// raw-row path runs unchanged; errors are left for that path to report.
fn gdn_stream_fused_prep_admitted(
    gpu: &Gpu,
    config: &HybridDims,
    dn_state: &DeltaNetStateView<'_>,
    layer: usize,
    call: &GdnStreamCall<'_>,
    n: usize,
    n_k_heads: usize,
    n_v_heads: usize,
) -> bool {
    let Ok((plan, meta, _)) = gdn_stream_layer(dn_state, layer, Some(call)) else {
        return false;
    };
    if plan.algorithm != GdnScanAlgorithm::Scan64
        || plan.scan_output_format != GdnTensorFormat::Bf16
        || plan.projection_recipe.qkv != rdna_compute::prefill_invariant::gdn_arith::IU4_G12
        || n == 0
        || !gpu.gdn_prep_fused_enabled()
        || !gdn_chunk_head_geometry_supported(
            n_k_heads,
            n_v_heads,
            config.linear_key_dim,
            config.linear_value_dim,
            config.conv_kernel,
        )
    {
        return false;
    }
    matches!(
        GdnAppendSchedule::new(&meta, call.start_pos, n, 64, 512),
        Ok(s) if !s.head && s.bulk_tail == 0
    )
}

/// One Scan64 layer: append the layer's raw projected rows (post-projection,
/// pre-convolution QKV and raw alpha/beta in `pbs`) at absolute position through
/// the plan's edge/bulk executor; the scan output lands in `pbs.dn_attn_out_batch`
/// in the plan's scan-output format (returned). The candidate cursor is recorded
/// in the call; nothing is published here. With `input == FusedPrepped` the
/// projection was the fused A4 producer (see
/// `gdn_stream_fused_prep_admitted`) and the executor runs only the fixup.
pub fn gdn_stream_scan_layer(
    gpu: &mut Gpu,
    call: &GdnStreamCall<'_>,
    config: &HybridDims,
    pbs: &BatchScratch<'_>,
    dn_state: &DeltaNetStateView<'_>,
    delta_layer_idx: usize,
    conv_weight: &GpuTensor,
    dt_bias: &GpuTensor,
    a_log: &GpuTensor,
    n: usize,
    q_scale: f32,
    n_k_heads: usize,
    n_v_heads: usize,
    input: GdnStreamInput,
) -> HipResult<GdnScanOut> {
    let fail = |m: String| HipError::new(0, &format!("GDN stream scan: {m}"));
    if !gdn_chunk_head_geometry_supported(
        n_k_heads,
        n_v_heads,
        config.linear_key_dim,
        config.linear_value_dim,
        config.conv_kernel,
    ) {
        return Err(fail("unsupported head geometry".into()));
    }
    let (plan, meta, carry) = gdn_stream_layer(dn_state, delta_layer_idx, Some(call))?;
    if plan.algorithm != GdnScanAlgorithm::Scan64 {
        return Err(fail(format!("plan algorithm is {:?}", plan.algorithm)));
    }
    let out_fmt = match plan.scan_output_format {
        GdnTensorFormat::Bf16 => GdnScanOut::Bf16,
        GdnTensorFormat::F32 => GdnScanOut::F32,
        GdnTensorFormat::F16 => return Err(fail("the scan has no f16 output plane".into())),
    };
    let ws = dn_state
        .stream
        .workspaces
        .iter()
        .find(|w| w.out_fmt == out_fmt && w.hk == n_k_heads && w.hv == n_v_heads)
        .ok_or_else(|| fail("no edge workspace for this geometry/output format".into()))?;
    let micro = carry.geometry.micro_rows;
    if micro < 64 {
        return Err(fail(format!("carry microtile {micro} < 64")));
    }
    let l = delta_layer_idx;
    let alpha = carry.raw_ab[l].sub_offset(0, micro * n_v_heads);
    let beta = carry.raw_ab[l].sub_offset(micro * n_v_heads, micro * n_v_heads);
    let scan_carry = ScanCarryView {
        tile_entry_f32: &carry.tile_state[l],
        tile_entry_conv: &carry.tile_conv[l],
        pending_qkv: &carry.raw_qkv[l],
        pending_alpha: &alpha,
        pending_beta: &beta,
    };
    let ef = dn_state.ef_residual(l);
    let live = GdnLiveState {
        planes: GdnQ8Planes {
            codes: &dn_state.s_matrices[l],
            scales: &dn_state.s_scales[l],
            ef,
        },
        conv: &dn_state.conv_states[l],
    };
    let GdnStreamViews { q, k, v, a } = gdn_stream_scratch_views(pbs, n, n_k_heads, n_v_heads)?;
    let consts = GdnStreamConsts {
        conv_weight,
        dt_bias,
        a_log,
        q_scale,
        eps: config.norm_eps,
        n_key_heads: n_k_heads,
        n_value_heads: n_v_heads,
    };
    let raw = GdnStreamRaw {
        qkv: &pbs.dn_qkv_batch,
        alpha: &pbs.dn_alpha_batch,
        beta: &pbs.dn_beta_batch,
        input,
    };
    let scratch = GdnStreamScratch {
        q: &q,
        k: &k,
        v: &v,
        a: &a,
    };
    call.attempted.set(true);
    let next = gpu.gdn_stream_append(
        plan,
        &meta,
        call.start_pos,
        n,
        &consts,
        &raw,
        &scratch,
        &pbs.dn_attn_out_batch,
        &live,
        &scan_carry,
        ws,
    )?;
    gdn_stream_record(call, l, next)?;
    Ok(out_fmt)
}

/// One Sequential-plan layer: the caller's producer already ran the convolution,
/// gates and Q/K normalization into `pbs` (`dn_q_batch`/`dn_k_batch` value-head
/// planes, `dn_v_batch`, gate `dn_alpha_batch`, beta `dn_beta_batch`); the
/// recurrence continues from the FP32 accumulator and quantizes only at absolute
/// 512-row seams. Output F32 in `pbs.dn_attn_out_batch`.
pub fn gdn_stream_sequential_layer(
    gpu: &mut Gpu,
    call: &GdnStreamCall<'_>,
    pbs: &BatchScratch<'_>,
    dn_state: &DeltaNetStateView<'_>,
    delta_layer_idx: usize,
    n: usize,
    n_v_heads: usize,
) -> HipResult<()> {
    let (plan, meta, carry) = gdn_stream_layer(dn_state, delta_layer_idx, Some(call))?;
    if plan.algorithm != GdnScanAlgorithm::Sequential {
        return Err(HipError::new(
            0,
            &format!(
                "GDN stream sequential: plan algorithm is {:?}",
                plan.algorithm
            ),
        ));
    }
    let l = delta_layer_idx;
    let planes = GdnQ8Planes {
        codes: &dn_state.s_matrices[l],
        scales: &dn_state.s_scales[l],
        ef: dn_state.ef_residual(l),
    };
    let inp = GdnSeqInputs {
        q: &pbs.dn_q_batch,
        k: &pbs.dn_k_batch,
        v: &pbs.dn_v_batch,
        gate: &pbs.dn_alpha_batch,
        beta: &pbs.dn_beta_batch,
        qk_head_ratio: 1,
    };
    call.attempted.set(true);
    let next = gpu.gdn_stream_append_sequential(
        plan,
        &meta,
        call.start_pos,
        n,
        &inp,
        &pbs.dn_attn_out_batch,
        &planes,
        &carry.tile_state[l],
        n_v_heads,
    )?;
    gdn_stream_record(call, l, next)
}

pub fn batch_chunk_delta_net_attn(
    gpu: &mut Gpu,
    layer: &DeltaNetLayerView<'_>,
    config: &HybridDims,
    pbs: &BatchScratch<'_>,
    dn_state: &DeltaNetStateView<'_>,
    n: usize,
    dim: usize,
    k_dim: usize,
    v_dim: usize,
    n_v_heads: usize,
    hd: usize,
    batch_semantics: BatchSemantics<'_>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    gdn_tape: Option<&GdnTapeView<'_>>,
    tape_offset: usize,
    delta_layer_idx: usize,
    q8_wmma_arch: bool,
    arch_has_wmma: bool,
    epilogue: BatchEpilogue<'_>,
    fusion: DflashFusionCtx,
    route: GdnPrefillRoute<'_>,
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
    let math = verify_dense_math(gpu);
    // Canonical ordinary prefill: the projection stores the raw rows and the
    // stream owns convolution, gates and scan from the absolute position, so
    // the numerical recipe never depends on the outer call's extent.
    //
    // Admission of the fused A4 producer inside the stream (gfx1201, 27B): the
    // fused `qkvzagdn_b1s` launch (QKV+Z+beta+alpha SET whose epilogue also runs
    // the q/k/v part of `gdn_chunk_prep`) followed by `gdn_chunk_prep_fixup` is
    // the same per-row arithmetic as four SETs plus the plain prep
    // (`gdn_arith::IU4_G12`: "one arithmetic"; the fixup carries the same causal
    // C64 gate prefix; `gdn_canonical_producer_oracle` `fused_vs_plain`). The
    // only difference is that the fused GEMM stores just the raw QKV rows the
    // fixup reads, while the stream needs full raw rows for a head replay and
    // for the carried raw prefix of an incomplete final tile. So it is admitted
    // only for an append whose schedule has NO head replay and NO incomplete
    // final tile (start and end on absolute 64-row tile boundaries): the C64
    // gate chunks and 128-row producer tiles then start on the absolute tile
    // grid and no later append reads this call's raw rows. Every other append
    // keeps the four SETs + full plain prep.
    let q_scale = 1.0 / (hd as f32).sqrt();
    let fused_views = match &route {
        GdnPrefillRoute::Stream(call)
            if gdn_stream_fused_prep_admitted(
                gpu,
                config,
                dn_state,
                delta_layer_idx,
                call,
                n,
                config.linear_key_heads,
                n_v_heads,
            ) =>
        {
            // Resolve the completion module before the projection mutates anything.
            gpu.gdn_chunk_prep_fixup_prepare()?;
            Some(gdn_stream_scratch_views(
                pbs,
                n,
                config.linear_key_heads,
                n_v_heads,
            )?)
        }
        _ => None,
    };
    let gdn_targets = fused_views
        .as_ref()
        .map(|views| rdna_compute::F2GdnTargets {
            conv_weight: &layer.conv_weight,
            conv_state: &dn_state.conv_states[delta_layer_idx],
            q: &views.q,
            k: &views.k,
            v: &views.v,
            q_scale,
            eps: config.norm_eps,
        });
    let prep_fused = batch_chunk_delta_net_input_projection(
        gpu,
        layer,
        config,
        pbs,
        n,
        dim,
        q8_wmma_arch,
        fusion,
        gdn_targets.as_ref(),
        math,
    )?;
    if let GdnPrefillRoute::Stream(call) = route {
        let (plan, _, _) = gdn_stream_layer(dn_state, delta_layer_idx, Some(&call))?;
        let x_fmt = match plan.algorithm {
            GdnScanAlgorithm::Scan64 => gdn_stream_scan_layer(
                gpu,
                &call,
                config,
                pbs,
                dn_state,
                delta_layer_idx,
                &layer.conv_weight,
                &layer.dt_bias,
                &layer.a_log,
                n,
                q_scale,
                config.linear_key_heads,
                n_v_heads,
                if prep_fused {
                    GdnStreamInput::FusedPrepped
                } else {
                    GdnStreamInput::Raw
                },
            )?,
            GdnScanAlgorithm::Sequential => {
                batch_chunk_delta_net_pre_gdn(
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
                    None,
                    None,
                    tape_offset,
                    delta_layer_idx,
                    fusion,
                    None,
                )?;
                gdn_stream_sequential_layer(
                    gpu,
                    &call,
                    pbs,
                    dn_state,
                    delta_layer_idx,
                    n,
                    n_v_heads,
                )?;
                GdnScanOut::F32
            }
            GdnScanAlgorithm::Wy16 => {
                return Err(HipError::new(
                    0,
                    "GDN stream route: Wy16 plans belong to Qwen4",
                ));
            }
        };
        return batch_chunk_delta_net_output_projection(
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
            x_fmt,
            math,
        );
    }

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
        None,
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
                    config.linear_value_dim,
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
                    config.linear_value_dim,
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
                        config.linear_value_dim,
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
                        config.linear_value_dim,
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
                            config.linear_value_dim,
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
                            config.linear_value_dim,
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
                        config.linear_value_dim,
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
                config.linear_value_dim,
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
        GdnScanOut::F32,
        math,
    )?;

    Ok(())
}
