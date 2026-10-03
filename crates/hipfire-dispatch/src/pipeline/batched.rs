// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched (prefill / multi-row) projection library shared by hybrid layer
//! operations: activation producers (IU4, A8, FP8-stream), key-selected GEMM
//! launches and the residual/partial epilogues. Moved verbatim from the
//! Qwen3.5 prefill path; every route issues the same launches as before.

use crate::context::DispatchCtx;
use crate::families::fused_qkv::FusedQkvFamily;
use crate::families::gemm::GemmFamily;
use crate::families::gemv::WeightRef;
#[cfg(feature = "deltanet")]
use crate::pipeline::hybrid::SwigluFfnOp;
use hip_bridge::{HipError, HipResult};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::LazyLock;

static GEMM: LazyLock<GemmFamily> = LazyLock::new(GemmFamily::new);
static FUSED_QKV: LazyLock<FusedQkvFamily> = LazyLock::new(FusedQkvFamily::new);

/// Process-wide GEMM family (shared kernel-selection state).
pub fn gemm_family() -> &'static GemmFamily {
    &GEMM
}

/// Process-wide fused QKV / gate-up family.
pub fn fused_qkv_family() -> &'static FusedQkvFamily {
    &FUSED_QKV
}

fn reject_mq4g128v2(dtype: DType) -> HipResult<()> {
    if dtype == DType::MQ4G128V2 {
        return Err(HipError::new(
            0,
            "batched MQ projection does not consume MQ4G128V2 (qt=53)",
        ));
    }
    Ok(())
}

/// RMS norm + FWHT rotation of `x` for the MQ weight `next_linear`.
#[allow(clippy::too_many_arguments)]
pub fn fused_rmsnorm_rotate_mq_batched_for(
    gpu: &mut Gpu,
    x: &GpuTensor,
    norm_weight: &GpuTensor,
    next_linear: &WeightRef<'_>,
    x_rot: &GpuTensor,
    k: usize,
    eps: f32,
    batch_size: usize,
) -> HipResult<()> {
    reject_mq4g128v2(next_linear.dtype)?;
    if let Some(awq) = next_linear.awq_scale {
        gpu.fused_rmsnorm_rotate_mq_awq_batched(x, norm_weight, awq, x_rot, k, eps, batch_size)
    } else {
        gpu.fused_rmsnorm_rotate_mq_batched(x, norm_weight, x_rot, k, eps, batch_size)
    }
}

/// F16-output twin of [`fused_rmsnorm_rotate_mq_batched_for`].
#[allow(clippy::too_many_arguments)]
pub fn fused_rmsnorm_rotate_mq_f16_batched_for(
    gpu: &mut Gpu,
    x: &GpuTensor,
    norm_weight: &GpuTensor,
    next_linear: &WeightRef<'_>,
    x_rot_f16: &GpuTensor,
    k: usize,
    eps: f32,
    batch_size: usize,
) -> HipResult<()> {
    reject_mq4g128v2(next_linear.dtype)?;
    if let Some(awq) = next_linear.awq_scale {
        gpu.fused_rmsnorm_rotate_mq_awq_f16_batched(
            x,
            norm_weight,
            awq,
            x_rot_f16,
            k,
            eps,
            batch_size,
        )
    } else {
        gpu.fused_rmsnorm_rotate_mq_f16_batched(x, norm_weight, x_rot_f16, k, eps, batch_size)
    }
}

/// FWHT rotation of `x` for the MQ weight `next_linear`.
pub fn rotate_x_mq_batched_for(
    gpu: &mut Gpu,
    next_linear: &WeightRef<'_>,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    batch_size: usize,
) -> HipResult<()> {
    reject_mq4g128v2(next_linear.dtype)?;
    if let Some(awq) = next_linear.awq_scale {
        gpu.rotate_x_mq_awq_batched(x, awq, x_rot, k, batch_size)
    } else {
        gpu.rotate_x_mq_batched(x, x_rot, k, batch_size)
    }
}

/// SiLU(gate)·up + FWHT rotation for the MQ down projection.
pub fn fused_silu_mul_rotate_mq_batched_for(
    gpu: &mut Gpu,
    down_proj_weight: &WeightRef<'_>,
    gate: &GpuTensor,
    up: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    batch_size: usize,
) -> HipResult<()> {
    reject_mq4g128v2(down_proj_weight.dtype)?;
    if let Some(awq) = down_proj_weight.awq_scale {
        gpu.fused_silu_mul_rotate_mq_awq_batched(gate, up, awq, x_rot, k, batch_size)
    } else {
        gpu.fused_silu_mul_rotate_mq_batched(gate, up, x_rot, k, batch_size)
    }
}

/// F16-output twin of [`fused_silu_mul_rotate_mq_batched_for`].
pub fn fused_silu_mul_rotate_mq_f16_batched_for(
    gpu: &mut Gpu,
    down_proj_weight: &WeightRef<'_>,
    gate: &GpuTensor,
    up: &GpuTensor,
    x_rot_f16: &GpuTensor,
    k: usize,
    batch_size: usize,
) -> HipResult<()> {
    reject_mq4g128v2(down_proj_weight.dtype)?;
    if let Some(awq) = down_proj_weight.awq_scale {
        gpu.fused_silu_mul_rotate_mq_awq_f16_batched(gate, up, awq, x_rot_f16, k, batch_size)
    } else {
        gpu.fused_silu_mul_rotate_mq_f16_batched(gate, up, x_rot_f16, k, batch_size)
    }
}

/// Row-parallel epilogue for dense-TP batched partials.
/// `Residual` is byte-identical single-GPU/MoE/EP behavior: GEMM adds into
/// `pbs.x_batch`. `Partial(out)` writes the N×dim GEMM result into `out`
/// without touching the residual; the caller all-reduces `out` and then
/// adds it into each rank's `x_batch`. This avoids duplicating full layer
/// bodies for the two transports.
pub enum BatchEpilogue<'a> {
    Residual,
    Partial(&'a GpuTensor),
}

#[inline]
pub fn zero_partial_for_residual(
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

/// Plain (unfused) batched-GEMM dispatcher key for a weight dtype.
///
/// Q8 keeps the chunked kernel it already used, so this is behaviour-preserving
/// for every existing model; the low-bit formats route to their tiled prefill
/// GEMMs. They share the Q8 call sites deliberately: none of the three has a
/// fused qkvza/gate_up/qkv kernel, so all three want the same unfused strategy.
pub fn plain_gemm_key_for(dt: DType) -> crate::types::KernelKey {
    use crate::types::KernelKey as K;
    match dt {
        DType::TQ2G128 => K::GemmTQ2G128Prefill,
        DType::BQ1G128 => K::GemmBQ1G128Prefill,
        _ => K::GemmQ8_0BatchedChunked,
    }
}

/// Route one plain batched GEMM through [`GemmFamily::run_key`] with an
/// explicit dispatcher-entry key, so the kernel's own arch routing is kept.
#[allow(clippy::too_many_arguments)]
pub fn run_plain_gemm_key(
    gpu: &mut Gpu,
    key: crate::types::KernelKey,
    w_buf: &GpuTensor,
    w_dtype: DType,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use crate::families::gemm::GemmParams;
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
    gemm_family()
        .run_key(key, &ctx, gpu, &params)
        .map_err(|e| HipError::new(0, &e.to_string()))
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
pub fn run_residual_gemm_key(
    gpu: &mut Gpu,
    key: crate::types::KernelKey,
    w_buf: &GpuTensor,
    w_dtype: DType,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use crate::families::gemm::GemmParams;
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
    gemm_family()
        .run_key(key, &ctx, gpu, &params)
        .map_err(|e| HipError::new(0, &e.to_string()))
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
pub fn run_fused_gate_up_key(
    gpu: &mut Gpu,
    key: crate::types::KernelKey,
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
    use crate::families::fused_qkv::FusedQkvParams;
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
    fused_qkv_family()
        .run(&ctx, gpu, &params)
        .map_err(|e| HipError::new(0, &e.to_string()))
}

/// qt=52 (MQ4G256V2Lloyd) prefill re-arm: extract the per-tensor E4M3 codebook
/// LUT or fail closed naming the prefill family. The uniform route never
/// touches this (uniform tensors carry no sidecar), so it stays bit-identical.
#[inline]
pub fn lloyd_e4m3_or_fail(w: &WeightRef<'_>, family: &'static str) -> HipResult<[u32; 4]> {
    w.lloyd_lut_e4m3.ok_or_else(|| {
        HipError::new(
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
pub fn lloyd_c16_or_fail(w: &WeightRef<'_>, family: &'static str) -> HipResult<[u32; 4]> {
    w.lloyd_lut_c16.ok_or_else(|| {
        HipError::new(
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
pub fn lloyd_mmq_lut_route(gpu: &Gpu) -> bool {
    matches!(gpu.arch.as_str(), "gfx1100" | "gfx1151")
        && hipfire_config::developer_var("HIPFIRE_LLOYD_MMQ_OFF")
            .ok()
            .as_deref()
            != Some("1")
}

/// True when all listed projection dtypes are the Lloyd-V2 dtype.
#[inline]
pub fn all_mq4v2_lloyd(dts: &[DType]) -> bool {
    dts.iter().all(|d| matches!(d, DType::MQ4G256V2Lloyd))
}

/// S3-f16-projection-inputs: exact-route gate for the FP16 projection-input
/// fast path (all four `batch_chunk_*` projection hooks below).
///
/// All predicates are cheap field reads — no env/lock/JIT in the cycle (the
/// kill switch resolves once at `FeatureFlags` init). Every failed predicate
/// runs the pre-change F32-producer + `convert_f32_to_f16` path byte-for-byte.
#[inline]
pub fn mq_f16_projection_fast_route(gpu: &Gpu, chain_verify: bool, n: usize, dim: usize) -> bool {
    chain_verify
        && gpu.arch_caps.is_gfx1100()
        && !gpu.flags.mq_f16_projection_off
        && n >= 1
        && n <= 16
        && dim % 256 == 0
        && !gpu.graphs.capture_mode
        && !gpu.replay.is_recording()
}

#[allow(clippy::too_many_arguments)]
/// S4-f16-residual-inputs: shared fast-route predicate for the four
/// post-attention/down hooks.
///
/// True only for the frozen fixture route: chain (non-tree) verify on exact
/// gfx1100 or exact gfx1201, the slice kill switch clear, an MQ4G256V2 residual consumer, a
/// `Residual` epilogue, and a verify-block batch `1 <= n <= 16`. Every false
/// keeps the pre-change path byte-for-byte.
pub fn s4_residual_fast(
    gpu: &Gpu,
    chain_verify: bool,
    w_dtype: DType,
    epilogue: &BatchEpilogue<'_>,
    n: usize,
) -> bool {
    chain_verify
        && !gpu.flags.mq_f16_residual_off
        && gpu.arch_caps.supports_dflash_f16_residual_fusions()
        && w_dtype == DType::MQ4G256V2
        && matches!(epilogue, BatchEpilogue::Residual)
        && (1..=16).contains(&n)
}

/// Producer-emitted A8 RMSNorm/FWHT, selected only for uniform MQ4v2 weights.
#[allow(clippy::too_many_arguments)]
pub fn try_a8_rmsnorm_prepared(
    gpu: &mut Gpu,
    x: &GpuTensor,
    weight: &GpuTensor,
    next: &WeightRef<'_>,
    k: usize,
    eps: f32,
    n: usize,
) -> HipResult<Option<rdna_compute::Int8MmqPrepared>> {
    if !gpu.flags.a8_fused_prod || !gpu.a8_prefill_active(n, k) || next.dtype != DType::MQ4G256V2 {
        return Ok(None);
    }
    let reservation = gpu.reserve_int8_mmq(k, n)?;
    gpu.fused_rmsnorm_rotate_mq_i8_gfx12_batched(
        x,
        weight,
        next.awq_scale,
        None,
        reservation,
        k,
        eps,
        n,
    )
    .map(Some)
}

pub fn try_a8_hin_prepared(
    gpu: &mut Gpu,
    down: &WeightRef<'_>,
    h: &GpuTensor,
    k: usize,
    n: usize,
) -> HipResult<Option<rdna_compute::Int8MmqPrepared>> {
    if !gpu.flags.a8_fused_prod || !gpu.a8_prefill_active(n, k) || down.dtype != DType::MQ4G256V2 {
        return Ok(None);
    }
    let Some(awq) = down.awq_scale else {
        return Ok(None);
    };
    let reservation = gpu.reserve_int8_mmq(k, n)?;
    gpu.fused_silu_hin_rotate_mq_i8_gfx12_batched(h, awq, reservation, k, n)
        .map(Some)
}

/// C2: when MQ4V2 + IU4 producer sidecar is live, emit block_i4_128 from
/// RMSNorm/FWHT and return a prepared handle. `None` → caller keeps the
/// incumbent fused_rmsnorm + standalone-quantizer path.
pub fn try_iu4_rmsnorm_prepared(
    gpu: &mut Gpu,
    x: &GpuTensor,
    norm_weight: &GpuTensor,
    next_linear: &WeightRef<'_>,
    x_rot: &GpuTensor,
    k: usize,
    eps: f32,
    n: usize,
    emit_f32: bool,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    if next_linear.dtype != DType::MQ4G256V2 || !gpu.iu4_producer_sidecar_active(n, k) {
        // The caller's fallback norms read `x` directly: land any residual
        // add a folded GEMM left owed to it first.
        gpu.flush_residual_fold()?;
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep = gpu.fused_rmsnorm_rotate_mq_i4_batched(
        x,
        norm_weight,
        next_linear.awq_scale,
        if emit_f32 { Some(x_rot) } else { None },
        res,
        k,
        eps,
        n,
    )?;
    Ok(Some(prep))
}

/// C2: SwiGLU/FWHT IU4 producer for w_down. `None` → incumbent path.
pub fn try_iu4_silu_prepared(
    gpu: &mut Gpu,
    w_down: &WeightRef<'_>,
    gate: &GpuTensor,
    up: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    n: usize,
    emit_f32: bool,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    if w_down.dtype != DType::MQ4G256V2 || !gpu.iu4_producer_sidecar_active(n, k) {
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep = gpu.fused_silu_mul_rotate_mq_i4_batched(
        gate,
        up,
        w_down.awq_scale,
        if emit_f32 { Some(x_rot) } else { None },
        res,
        k,
        n,
    )?;
    Ok(Some(prep))
}

/// F1-lite: AWQ IU4 producer for w_down reading h = silu(gate)*up, already
/// formed by the gate/up GEMM epilogue in `h`. Byte-identical sidecar to
/// [`try_iu4_silu_prepared`]'s.
pub fn iu4_hin_prepared(
    gpu: &mut Gpu,
    w_down: &WeightRef<'_>,
    h: &GpuTensor,
    k: usize,
    n: usize,
) -> HipResult<rdna_compute::Int4MmqPrepared> {
    let awq = w_down
        .awq_scale
        .as_ref()
        .expect("F1-lite h requires the w_down AWQ scale (f1lite_ffn_eligible)");
    let res = gpu.reserve_int4_mmq(k, n)?;
    gpu.fused_silu_hin_rotate_mq_i4_batched(h, awq, res, k, n)
}

/// gfx1201 slice-1: SwiGLU/FWHT + in-register `block_i4_128` producer for
/// w_down. `None` → caller keeps the incumbent silu + standalone-quantizer
/// path. Same contract as [`try_iu4_silu_prepared`] (uniform MQ4G256V2 only,
/// Residual-only at the call sites) but routed by `iu4_silu_quant_fused_active`
/// (default on for exact-gfx1201 IU4; `HIPFIRE_GFX12_SILU_QUANT_FUSED=0` opts
/// out) instead of the portable gfx11 sidecar gate. Every other caller keeps
/// `quantize_int4_mmq_ds128` path untouched.
pub fn try_gfx12_silu_quant_fused_prepared(
    gpu: &mut Gpu,
    w_down: &WeightRef<'_>,
    gate: &GpuTensor,
    up: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    n: usize,
    emit_f32: bool,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    if w_down.dtype != DType::MQ4G256V2 || !gpu.iu4_silu_quant_fused_active(n, k) {
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep = gpu.fused_silu_mul_rotate_mq_i4_gfx12_batched(
        gate,
        up,
        w_down.awq_scale,
        if emit_f32 { Some(x_rot) } else { None },
        res,
        k,
        n,
    )?;
    Ok(Some(prep))
}

/// gfx1201 slices-2: RMSNorm/FWHT + in-register `block_i4_128` producer for
/// the qkvza/gate_up/qkv inputs. `None` → caller keeps the incumbent
/// rmsnorm+rotate + standalone-quantizer path. Same contract as
/// [`try_iu4_rmsnorm_prepared`] (uniform MQ4G256V2 only) but gated by
/// `HIPFIRE_GFX12_PRODUCER_QUANT_FUSED` on exact gfx1201 instead of the
/// portable gfx11 sidecar gate.
pub fn try_gfx12_rmsnorm_quant_fused_prepared(
    gpu: &mut Gpu,
    x: &GpuTensor,
    norm_weight: &GpuTensor,
    next_linear: &WeightRef<'_>,
    x_rot: &GpuTensor,
    k: usize,
    eps: f32,
    n: usize,
    emit_f32: bool,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    if next_linear.dtype != DType::MQ4G256V2 || !gpu.iu4_producer_quant_fused_active(n, k) {
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep = gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(
        x,
        norm_weight,
        next_linear.awq_scale,
        if emit_f32 { Some(x_rot) } else { None },
        res,
        k,
        eps,
        n,
    )?;
    Ok(Some(prep))
}

/// gfx1201 FP8-stream: RMSNorm/FWHT producer that emits byte-identical
/// `prepare_mq4v2_fp8_x_f32` (scale_mode=1) planes for Lloyd-weight fp8
/// GEMMs. `None` → caller keeps the incumbent producer + standalone-pack
/// path. Admission is uniform `MQ4G256V2Lloyd` next-linear plus the shared
/// `fp8_stream_active` gate (exact gfx1201, eager, batch >= 64,
/// K % 256 == 0; partial N tiles are masked in-kernel, so odd tails run
/// unpadded). On `HIPFIRE_FP8_PROD_INREG=1` the producer leaves `x_rot`
/// untouched; the admitted consumer reads only the prepared FP8 planes.
pub fn try_gfx12_fp8_stream_rmsnorm_prepared(
    gpu: &mut Gpu,
    x: &GpuTensor,
    norm_weight: &GpuTensor,
    next_linear: &WeightRef<'_>,
    x_rot: &GpuTensor,
    k: usize,
    eps: f32,
    n: usize,
) -> HipResult<Option<rdna_compute::Mq4v2Fp8Prepared>> {
    if !matches!(next_linear.dtype, DType::MQ4G256V2 | DType::MQ4G256V2Lloyd)
        || !gpu.fp8_stream_active(n, k)
    {
        return Ok(None);
    }
    let prep = gpu.fused_rmsnorm_rotate_mq_fp8_gfx12_batched(
        x,
        norm_weight,
        next_linear.awq_scale,
        x_rot,
        k,
        eps,
        n,
    )?;
    Ok(Some(prep))
}

/// gfx1201 FP8-stream down-projection producer. The admitted Lloyd route
/// writes the exact scale_mode=1 FP8 planes for the residual GEMM.
/// `HIPFIRE_FP8_PROD_INREG=1` retains the FWHT row in registers.
/// With the paired-row gate/up epilogue, `gate` already contains h and the
/// h-first in-register producer consumes it without reading `up` or `x_rot`.
pub fn try_gfx12_fp8_stream_silu_prepared(
    gpu: &mut Gpu,
    w_down: &WeightRef<'_>,
    gate: &GpuTensor,
    up: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    n: usize,
    h_ready: bool,
) -> HipResult<Option<rdna_compute::Mq4v2Fp8Prepared>> {
    if !matches!(w_down.dtype, DType::MQ4G256V2 | DType::MQ4G256V2Lloyd)
        || !gpu.fp8_stream_active(n, k)
    {
        return Ok(None);
    }
    let prep = if h_ready {
        gpu.fused_silu_hin_rotate_mq_fp8_gfx12_batched(gate, w_down.awq_scale, k, n)?
    } else {
        gpu.fused_silu_mul_rotate_mq_fp8_gfx12_batched(gate, up, w_down.awq_scale, x_rot, k, n)?
    };
    Ok(Some(prep))
}

/// The gate/up output format determines which down-projection producer may
/// consume it. An h tensor is not interchangeable with separate gate/up.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FfnGateOutput {
    Separate,
    Iu4H,
    Fp8H,
    A8H,
}

/// F1-lite: true when this FFN may fold SwiGLU into the gate/up GEMM, i.e.
/// the down projection would take the AWQ IU4 SwiGLU producer
/// ([`try_iu4_silu_prepared`] with an AWQ scale: uniform MQ4G256V2, `Residual`
/// epilogue, no S4 f16 route, portable gfx11 sidecar live; or on exact
/// gfx1201 [`try_gfx12_silu_quant_fused_prepared`]'s AWQ twin) and gate/up are
/// MQ4G256V2 of equal shape. Decided once per FFN, before gate/up; the gate/up
/// hook then reports whether it actually emitted h
/// (`Gpu::gemm_gate_up_silu_mq4g256v2_iu4_prepared` adds the per-arch tile
/// rule and `HIPFIRE_F1LITE`).
#[allow(clippy::too_many_arguments)]
pub fn f1lite_ffn_eligible(
    gpu: &Gpu,
    w_gate: &WeightRef<'_>,
    w_up: &WeightRef<'_>,
    w_down: &WeightRef<'_>,
    epilogue: &BatchEpilogue<'_>,
    chain_verify: bool,
    hidden_dim: usize,
    n: usize,
) -> bool {
    w_gate.dtype == DType::MQ4G256V2
        && w_up.dtype == DType::MQ4G256V2
        && w_gate.m == hidden_dim
        && w_up.m == hidden_dim
        && w_gate.k == w_up.k
        && w_down.dtype == DType::MQ4G256V2
        && w_down.awq_scale.is_some()
        && w_down.k == hidden_dim
        && matches!(epilogue, BatchEpilogue::Residual)
        && !s4_residual_fast(gpu, chain_verify, w_down.dtype, epilogue, n)
        && (gpu.a8_prefill_active(n, w_gate.k) && hidden_dim % 128 == 0
            || gpu.iu4_producer_sidecar_active(n, hidden_dim)
            || gpu.iu4_silu_quant_fused_active(n, hidden_dim))
}

/// Shared dispatch for batched `wo` / `w_down` with selectable epilogue.
/// Preserves every dtype-specific residual kernel path; `Partial` reuses the
/// same kernels into a zeroed `n×m` slice of `out` instead of `x_batch`.
pub fn dispatch_batched_gemm_epilogue(
    gpu: &mut Gpu,
    x_batch: &GpuTensor,
    scratch: &GpuTensor,
    w: &WeightRef<'_>,
    input: &GpuTensor,
    epilogue: &BatchEpilogue<'_>,
    n: usize,
    q8_wmma_arch: bool,
    _arch_has_wmma: bool,
) -> HipResult<()> {
    let m = w.m;
    let k = w.k;
    let is_6bit = matches!(w.dtype, DType::MQ6G256 | DType::HFQ6G256);
    let is_mq3_lloyd = matches!(w.dtype, DType::MQ3G256Lloyd);
    let is_mq3 = matches!(w.dtype, DType::MQ3G256);
    let is_fp4 = matches!(w.dtype, DType::HFP4G32 | DType::MFP4G32);
    let is_q8 = matches!(w.dtype, DType::Q8_0);
    let is_lowbit = matches!(w.dtype, DType::TQ2G128 | DType::BQ1G128);
    match epilogue {
        BatchEpilogue::Residual => {
            if is_6bit {
                return run_residual_gemm_key(
                    gpu,
                    crate::types::KernelKey::GemmHfq6G256Residual,
                    w.buf,
                    w.dtype,
                    input,
                    x_batch,
                    m,
                    k,
                    n,
                );
            } else if is_q8 && q8_wmma_arch {
                let x_n = x_batch.sub_offset(0, n * m);
                return run_residual_gemm_key(
                    gpu,
                    crate::types::KernelKey::GemmQ8_0ResidualWmma,
                    w.buf,
                    w.dtype,
                    input,
                    &x_n,
                    m,
                    k,
                    n,
                );
            } else if is_q8 || is_lowbit {
                let scratch = scratch.sub_offset(0, n * m);
                run_plain_gemm_key(
                    gpu,
                    plain_gemm_key_for(w.dtype),
                    w.buf,
                    w.dtype,
                    input,
                    &scratch,
                    m,
                    k,
                    n,
                )?;
                let x_n = x_batch.sub_offset(0, n * m);
                return gpu.add_inplace_f32(&x_n, &scratch);
            } else if is_mq3_lloyd {
                return run_residual_gemm_key(
                    gpu,
                    crate::types::KernelKey::GemmMq3G256LloydResidual,
                    w.buf,
                    w.dtype,
                    input,
                    x_batch,
                    m,
                    k,
                    n,
                );
            } else if is_mq3 {
                return run_residual_gemm_key(
                    gpu,
                    crate::types::KernelKey::GemmHfq3G256Residual,
                    w.buf,
                    w.dtype,
                    input,
                    x_batch,
                    m,
                    k,
                    n,
                );
            } else if is_fp4 {
                return run_residual_gemm_key(
                    gpu,
                    crate::types::KernelKey::GemmHfp4G32Residual,
                    w.buf,
                    w.dtype,
                    input,
                    x_batch,
                    m,
                    k,
                    n,
                );
            } else if matches!(w.dtype, DType::MQ4G256V2Lloyd) {
                // qt=52: gfx11 → MMQ-LUT residual; gfx12 keeps FP8-LUT; else
                // fail-closed inside the FP8 launcher (never uniform).
                if lloyd_mmq_lut_route(gpu) {
                    let c16 = lloyd_c16_or_fail(w, "dispatch_batched_gemm_epilogue")?;
                    return gpu
                        .gemm_hfq4g256_residual_mmq_lloyd(w.buf, input, x_batch, m, k, n, c16);
                }
                let lut = lloyd_e4m3_or_fail(w, "dispatch_batched_gemm_epilogue")?;
                return gpu.gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8_lloyd(
                    w.buf, input, x_batch, m, k, n, 1, lut,
                );
            } else {
                return run_residual_gemm_key(
                    gpu,
                    crate::families::gemm::residual_gemm_key_for(w.dtype),
                    w.buf,
                    w.dtype,
                    input,
                    x_batch,
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
                    crate::types::KernelKey::GemmHfq6G256Residual,
                    w.buf,
                    w.dtype,
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
                    crate::types::KernelKey::GemmQ8_0ResidualWmma,
                    w.buf,
                    w.dtype,
                    input,
                    &out_n,
                    m,
                    k,
                    n,
                );
            } else if is_q8 || is_lowbit {
                return run_plain_gemm_key(
                    gpu,
                    plain_gemm_key_for(w.dtype),
                    w.buf,
                    w.dtype,
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
                    crate::types::KernelKey::GemmMq3G256LloydResidual,
                    w.buf,
                    w.dtype,
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
                    crate::types::KernelKey::GemmHfq3G256Residual,
                    w.buf,
                    w.dtype,
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
                    crate::types::KernelKey::GemmHfp4G32Residual,
                    w.buf,
                    w.dtype,
                    input,
                    &out_n,
                    m,
                    k,
                    n,
                );
            } else if matches!(w.dtype, DType::MQ4G256V2Lloyd) {
                // qt=52 Partial: zero then residual (== plain GEMM). gfx11 →
                // MMQ-LUT ADD into zeroed Y; gfx12 keeps FP8-LUT.
                zero_partial_for_residual(gpu, out, n, m)?;
                if lloyd_mmq_lut_route(gpu) {
                    let c16 = lloyd_c16_or_fail(w, "dispatch_batched_gemm_epilogue")?;
                    return gpu
                        .gemm_hfq4g256_residual_mmq_lloyd(w.buf, input, &out_n, m, k, n, c16);
                }
                let lut = lloyd_e4m3_or_fail(w, "dispatch_batched_gemm_epilogue")?;
                return gpu.gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8_lloyd(
                    w.buf, input, &out_n, m, k, n, 1, lut,
                );
            } else {
                zero_partial_for_residual(gpu, out, n, m)?;
                return run_residual_gemm_key(
                    gpu,
                    crate::families::gemm::residual_gemm_key_for(w.dtype),
                    w.buf,
                    w.dtype,
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

/// Consume producer-emitted MQ4v2 FP8 planes for a Lloyd residual/partial
/// projection.  This is the prepared twin of `dispatch_batched_gemm_epilogue`
/// for the exact gfx1201 route; no activation pack is launched here.
pub fn dispatch_batched_fp8_lloyd_epilogue(
    gpu: &mut Gpu,
    x_batch: &GpuTensor,
    w: &WeightRef<'_>,
    prepared: &rdna_compute::Mq4v2Fp8Prepared,
    epilogue: &BatchEpilogue<'_>,
    n: usize,
) -> HipResult<()> {
    let out = match epilogue {
        BatchEpilogue::Residual => x_batch,
        BatchEpilogue::Partial(out) => {
            zero_partial_for_residual(gpu, out, n, w.m)?;
            out
        }
    };
    match w.dtype {
        DType::MQ4G256V2 => gpu.gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8_prepared(
            w.buf, prepared, out, w.m, w.k, n,
        ),
        DType::MQ4G256V2Lloyd => gpu.gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8_prepared_lloyd(
            w.buf,
            prepared,
            out,
            w.m,
            w.k,
            n,
            lloyd_e4m3_or_fail(w, "dispatch_batched_fp8_lloyd_epilogue")?,
        ),
        _ => unreachable!("fp8 producer admission accepts only MQ4G256V2 weights"),
    }
}

/// Batched-row operands of a [`SwigluFfnOp`] (`rows > 1`).
pub struct SwigluFfnBatch<'a> {
    /// Model width (gate/up input columns).
    pub dim: usize,
    /// FFN intermediate width.
    pub hidden_dim: usize,
    pub x_rot_f16: &'a GpuTensor,
    pub hidden_f16: &'a GpuTensor,
    pub epilogue: BatchEpilogue<'a>,
    /// DFlash chain-verify fusion admits the exact-FP16 input routes.
    pub chain_verify: bool,
    pub q8_wmma_arch: bool,
    /// The operand buffers hold every row (not the lean gfx11 prefill owner):
    /// the opt-in packed MQ4 route may run.
    pub packed_mq4: bool,
}

/// Uniform MQ4G256 (v1) only; the explicit packed opt-in leaves all native
/// routes unchanged when disabled. MQ4V2 (qt44, e.g. mq4-xts) keeps the landed
/// gfx1100 builder V2C / IU4 routes: packed was not measured faster there.
/// Lloyd and mixed wire formats are excluded.
#[cfg_attr(not(feature = "deltanet"), allow(dead_code))]
fn packed_mq4_ffn_gate_up_admitted(
    packed_admitted: bool,
    gate: (DType, usize, usize),
    up: (DType, usize, usize),
    n: usize,
) -> bool {
    packed_admitted
        && gate.0 == DType::MQ4G256
        && (gate.1, gate.2) == (17_408, 5_120)
        && up == gate
        && n > 0
        && n % 256 == 0
}

/// Runs before native prepared producers: explicitly emit AWQ-aware rotated
/// F32 input, never reuse a potentially unwritten F32 plane from an A8/IU4/FP8
/// producer. Output remains separate F32 planes with no residual epilogue.
#[cfg(feature = "deltanet")]
fn try_packed_mq4_ffn_gate_up(
    gpu: &mut Gpu,
    op: &SwigluFfnOp<'_>,
    b: &SwigluFfnBatch<'_>,
    n: usize,
) -> HipResult<bool> {
    let (gate, up) = (&op.w_gate, &op.w_up);
    if !packed_mq4_ffn_gate_up_admitted(
        b.packed_mq4 && gpu.packed_mq4_admitted(gate.m, gate.k, n),
        (gate.dtype, gate.m, gate.k),
        (up.dtype, up.m, up.k),
        n,
    ) {
        return Ok(false);
    }
    gpu.flush_residual_fold()?;
    let separate_awq_inputs = gate.awq_scale.is_some() || up.awq_scale.is_some();
    for (index, (weight, output)) in [(gate, op.gate), (up, op.up)].into_iter().enumerate() {
        // Sidecars are per-weight and may differ (or exist on only one side).
        // Reuse the rotated input only when neither projection carries AWQ.
        if index == 0 || separate_awq_inputs {
            fused_rmsnorm_rotate_mq_batched_for(
                gpu, op.x, op.norm, weight, op.x_rot, weight.k, op.eps, n,
            )?;
        }
        run_plain_gemm_key(
            gpu,
            crate::types::KernelKey::GemmMq4Packed,
            weight.buf,
            weight.dtype,
            op.x_rot,
            output,
            weight.m,
            weight.k,
            n,
        )?;
    }
    Ok(true)
}

#[cfg_attr(not(feature = "deltanet"), allow(dead_code))]
fn packed_mq4_down_admitted(
    packed_admitted: bool,
    separate: bool,
    residual: bool,
    down: (DType, usize, usize),
    hidden_dim: usize,
) -> bool {
    packed_admitted
        && separate
        && residual
        && down.0 == DType::MQ4G256
        && (down.1, down.2) == (5120, 17408)
        && hidden_dim == down.2
}

/// Only consume separate F32 gate/up planes; never reinterpret a prepared H
/// plane or add a residual to a tensor-parallel partial output.
#[cfg(feature = "deltanet")]
fn try_packed_mq4_down(
    gpu: &mut Gpu,
    op: &SwigluFfnOp<'_>,
    b: &SwigluFfnBatch<'_>,
    n: usize,
    h_source: FfnGateOutput,
) -> HipResult<bool> {
    let down = &op.w_down;
    if !packed_mq4_down_admitted(
        b.packed_mq4 && gpu.packed_mq4_admitted(down.m, down.k, n),
        h_source == FfnGateOutput::Separate,
        matches!(b.epilogue, BatchEpilogue::Residual),
        (down.dtype, down.m, down.k),
        b.hidden_dim,
    ) {
        return Ok(false);
    }
    gpu.flush_residual_fold()?;
    fused_silu_mul_rotate_mq_batched_for(gpu, down, op.gate, op.up, op.hidden, b.hidden_dim, n)?;
    run_residual_gemm_key(
        gpu,
        crate::types::KernelKey::GemmMq4PackedResidual,
        down.buf,
        down.dtype,
        op.hidden,
        op.x,
        down.m,
        down.k,
        n,
    )?;
    Ok(true)
}

#[cfg(feature = "deltanet")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
/// Prescaffold (behavior-only) extraction for S3-f16-projection-inputs.
///
/// Same statements, same order, same launches as the inlined block.
/// S9-mq4v2-persistent-prologues will issue `try_mq4v2_persistent_prologue`
/// from inside this hook after S3/S4 land.
///
/// F1-lite: with `f1lite` (see [`f1lite_ffn_eligible`]) the IU4 gate/up pair
/// may instead emit h = silu(gate)*up into `gate_ffn_batch`; returns whether
/// it did, i.e. whether the down hook must read h.
fn swiglu_ffn_gate_up_batched(
    gpu: &mut Gpu,
    op: &SwigluFfnOp<'_>,
    b: &SwigluFfnBatch<'_>,
    n: usize,
    f1lite: bool,
) -> HipResult<FfnGateOutput> {
    let dim = b.dim;
    if try_packed_mq4_ffn_gate_up(gpu, op, b, n)? {
        return Ok(FfnGateOutput::Separate);
    }
    // S3-f16-projection-inputs fast path: exact-FP16 FFN gate/up inputs.
    // gate/up share the pre-rotation input, so both must be MQ4G256V2.
    if mq_f16_projection_fast_route(gpu, b.chain_verify, n, dim)
        && op.w_gate.dtype == DType::MQ4G256V2
        && op.w_up.dtype == DType::MQ4G256V2
    {
        gpu.flush_residual_fold()?;
        fused_rmsnorm_rotate_mq_f16_batched_for(
            gpu,
            op.x,
            &op.norm,
            &op.w_gate,
            b.x_rot_f16,
            dim,
            op.eps,
            n,
        )?;
        return gpu
            .gemm_gate_up_mq4g256v2_wmma_f16(
                op.w_gate.buf,
                op.w_up.buf,
                b.x_rot_f16,
                op.gate,
                op.up,
                op.w_gate.m,
                op.w_up.m,
                op.w_gate.k,
                n,
            )
            .map(|()| FfnGateOutput::Separate);
    }
    // FFN: rmsnorm (+ rotate for MQ).
    let ffn_is_mq = matches!(
        op.w_gate.dtype,
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
    let ffn_is_6bit = matches!(op.w_gate.dtype, DType::MQ6G256 | DType::HFQ6G256);
    let ffn_is_mq3 = matches!(op.w_gate.dtype, DType::MQ3G256);
    let ffn_is_mq3_lloyd = matches!(op.w_gate.dtype, DType::MQ3G256Lloyd);
    let ffn_is_fp4 = matches!(op.w_gate.dtype, DType::HFP4G32 | DType::MFP4G32);
    let ffn_is_q8 = matches!(op.w_gate.dtype, DType::Q8_0);
    // TQ2G128/BQ1G128 have no fused qkvza/gate_up/qkv kernel, so they take
    // the same UNFUSED plain-GEMM strategy as Q8 rather than falling through
    // to the HFQ4 arm, which would read these packed blocks at the wrong
    // stride and produce fluent-but-wrong tokens.
    let ffn_is_lowbit = matches!(op.w_gate.dtype, DType::TQ2G128 | DType::BQ1G128);
    // qt=52 re-arm anchor: gate+up both Lloyd → FP8-LUT launcher.
    let ffn_is_mq4v2_lloyd = all_mq4v2_lloyd(&[op.w_gate.dtype, op.w_up.dtype]);
    let a8_prep = if op.w_gate.dtype == DType::MQ4G256V2 && op.w_up.dtype == DType::MQ4G256V2 {
        try_a8_rmsnorm_prepared(gpu, op.x, &op.norm, &op.w_gate, dim, op.eps, n)?
    } else {
        None
    };
    let mut iu4_prep: Option<rdna_compute::Int4MmqPrepared> = None;
    let mut fp8_prep: Option<rdna_compute::Mq4v2Fp8Prepared> = None;
    if ffn_is_mq && a8_prep.is_none() {
        // C2: MQ4V2 IU4 producer (emit_f32=false — gate_up has no f32 small tails).
        iu4_prep = try_iu4_rmsnorm_prepared(
            gpu, op.x, &op.norm, &op.w_gate, op.x_rot, dim, op.eps, n, false,
        )?;
        if iu4_prep.is_none() {
            // gfx1201 slices-2: fused rmsnorm+quant producer (bit-identical).
            iu4_prep = try_gfx12_rmsnorm_quant_fused_prepared(
                gpu, op.x, &op.norm, &op.w_gate, op.x_rot, dim, op.eps, n, false,
            )?;
        }
        if iu4_prep.is_none() {
            // gfx1201 FP8-stream: RMSNorm/FWHT producer emits the fp8
            // pre-pass planes directly (byte-identical prepare outputs);
            // the standalone pack launch disappears. Lloyd weights only.
            fp8_prep = try_gfx12_fp8_stream_rmsnorm_prepared(
                gpu, op.x, &op.norm, &op.w_gate, op.x_rot, dim, op.eps, n,
            )?;
        }
        if iu4_prep.is_none() && fp8_prep.is_none() {
            // AWQ-aware: next linear is w_gate (gate/up share input → same AWQ scale).
            fused_rmsnorm_rotate_mq_batched_for(
                gpu, op.x, &op.norm, &op.w_gate, op.x_rot, dim, op.eps, n,
            )?;
        }
    } else if !ffn_is_mq {
        gpu.rmsnorm_batched(op.x, &op.norm, op.x_rot, n, dim, op.eps)?;
    }

    // Batched gate+up projection.
    // #397 Ship 5.2 slice 2: fused gate+up dtypes → FusedQkvFamily
    // (batched-prefill gate+up variant) via run_fused_gate_up_key.
    // The Q8-non-WMMA case stays as two plain GemmQ8_0BatchedChunked
    // GEMMs (not a fused kernel — slice 1). The HFQ3 WMMA-vs-base
    // split is folded into the FusedGateUpHfq3G256 run-arm, which
    // re-derives it from gpu.arch_caps.has_wmma() (== arch_has_wmma).
    if let Some(prep) = &a8_prep {
        let xq = gpu.int8_mmq_prepared_ptr(prep, op.w_gate.k, n)?;
        if f1lite {
            gpu.gemm_gate_up_silu_mq4g256v2_i8_prequant(
                op.w_gate.buf,
                op.w_up.buf,
                xq,
                op.gate,
                op.w_gate.m,
                op.w_gate.k,
                n,
            )?;
            return Ok(FfnGateOutput::A8H);
        }
        gpu.gemm_mq4g256v2_mmq_set_prequant_i8(
            op.w_gate.buf,
            xq,
            op.gate,
            op.w_gate.m,
            op.w_gate.k,
            n,
        )?;
        gpu.gemm_mq4g256v2_mmq_set_prequant_i8(op.w_up.buf, xq, op.up, op.w_up.m, op.w_up.k, n)?;
    } else if f1lite && gpu.a8_prefill_active(n, op.w_gate.k) {
        let xq = gpu.ensure_int8_mmq_x(op.x_rot, n, op.w_gate.k)?;
        gpu.gemm_gate_up_silu_mq4g256v2_i8_prequant(
            op.w_gate.buf,
            op.w_up.buf,
            xq,
            op.gate,
            op.w_gate.m,
            op.w_gate.k,
            n,
        )?;
        return Ok(FfnGateOutput::A8H);
    } else if let Some(prep) = &iu4_prep {
        // F1-lite: one GEMM v2 launch emits h; else the SET pair below.
        if f1lite
            && gpu.gemm_gate_up_silu_mq4g256v2_iu4_prepared(
                op.w_gate.buf,
                op.w_up.buf,
                prep,
                op.gate,
                op.w_gate.m,
                op.w_up.m,
                op.w_gate.k,
                n,
            )?
        {
            return Ok(FfnGateOutput::Iu4H);
        }
        gpu.gemm_gate_up_mq4g256v2_wmma_iu4_prepared(
            op.w_gate.buf,
            op.w_up.buf,
            prep,
            op.gate,
            op.up,
            op.w_gate.m,
            op.w_up.m,
            op.w_gate.k,
            n,
        )?;
    } else if ffn_is_6bit {
        run_fused_gate_up_key(
            gpu,
            crate::types::KernelKey::FusedGateUpHfq6G256,
            op.w_gate.buf,
            op.w_up.buf,
            op.x_rot,
            op.gate,
            op.up,
            op.w_gate.m,
            op.w_up.m,
            op.w_gate.k,
            n,
        )?;
    } else if ffn_is_q8 && b.q8_wmma_arch {
        debug_assert!(
            matches!(op.w_up.dtype, DType::Q8_0),
            "LA FFN Q8 WMMA dispatch requires both w_gate and w_up to be Q8_0",
        );
        run_fused_gate_up_key(
            gpu,
            crate::types::KernelKey::FusedGateUpQ8_0,
            op.w_gate.buf,
            op.w_up.buf,
            op.x_rot,
            op.gate,
            op.up,
            op.w_gate.m,
            op.w_up.m,
            op.w_gate.k,
            n,
        )?;
    } else if ffn_is_q8 || ffn_is_lowbit {
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(op.w_gate.dtype),
            op.w_gate.buf,
            op.w_gate.dtype,
            op.x_rot,
            op.gate,
            op.w_gate.m,
            op.w_gate.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(op.w_up.dtype),
            op.w_up.buf,
            op.w_up.dtype,
            op.x_rot,
            op.up,
            op.w_up.m,
            op.w_up.k,
            n,
        )?;
    } else if ffn_is_mq3_lloyd {
        run_fused_gate_up_key(
            gpu,
            crate::types::KernelKey::FusedGateUpMq3G256Lloyd,
            op.w_gate.buf,
            op.w_up.buf,
            op.x_rot,
            op.gate,
            op.up,
            op.w_gate.m,
            op.w_up.m,
            op.w_gate.k,
            n,
        )?;
    } else if ffn_is_mq3 {
        run_fused_gate_up_key(
            gpu,
            crate::types::KernelKey::FusedGateUpHfq3G256,
            op.w_gate.buf,
            op.w_up.buf,
            op.x_rot,
            op.gate,
            op.up,
            op.w_gate.m,
            op.w_up.m,
            op.w_gate.k,
            n,
        )?;
    } else if ffn_is_fp4 {
        run_fused_gate_up_key(
            gpu,
            crate::types::KernelKey::FusedGateUpHfp4G32,
            op.w_gate.buf,
            op.w_up.buf,
            op.x_rot,
            op.gate,
            op.up,
            op.w_gate.m,
            op.w_up.m,
            op.w_gate.k,
            n,
        )?;
    } else if ffn_is_mq4v2_lloyd {
        // qt=52: gate+up both Lloyd. gfx11 → MMQ-LUT; gfx12 FP8-LUT.
        if lloyd_mmq_lut_route(gpu) {
            gpu.gemm_gate_up_mq4g256v2_mmq_lloyd(
                op.w_gate.buf,
                op.w_up.buf,
                op.x_rot,
                op.gate,
                op.up,
                op.w_gate.m,
                op.w_up.m,
                op.w_gate.k,
                n,
                lloyd_c16_or_fail(&op.w_gate, "swiglu_ffn_gate_up_batched")?,
                lloyd_c16_or_fail(&op.w_up, "swiglu_ffn_gate_up_batched")?,
            )?;
        } else if let Some(prep) = &fp8_prep {
            // gfx1201 FP8-stream: the producer already emitted the fp8
            // pre-pass planes; consume them directly, no pack launch.
            gpu.gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2_fp8_bt12_prepared_lloyd(
                op.w_gate.buf,
                op.w_up.buf,
                prep,
                op.gate,
                op.up,
                op.w_gate.m,
                op.w_up.m,
                op.w_gate.k,
                n,
                lloyd_e4m3_or_fail(&op.w_gate, "swiglu_ffn_gate_up_batched")?,
                lloyd_e4m3_or_fail(&op.w_up, "swiglu_ffn_gate_up_batched")?,
            )?;
        } else {
            gpu.gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2_fp8_lloyd(
                op.w_gate.buf,
                op.w_up.buf,
                op.x_rot,
                op.gate,
                op.up,
                op.w_gate.m,
                op.w_up.m,
                op.w_gate.k,
                n,
                1,
                lloyd_e4m3_or_fail(&op.w_gate, "swiglu_ffn_gate_up_batched")?,
                lloyd_e4m3_or_fail(&op.w_up, "swiglu_ffn_gate_up_batched")?,
            )?;
        }
    } else if matches!(op.w_gate.dtype, DType::MQ4G256V2Lloyd)
        || matches!(op.w_up.dtype, DType::MQ4G256V2Lloyd)
    {
        // Mixed Lloyd/uniform gate/up: no kernel reads mixed codebooks.
        return Err(HipError::new(
            0,
            "swiglu_ffn_gate_up_batched: mixed MQ4G256V2Lloyd/uniform gate/up — refusing (quantize both or neither)",
        ));
    } else if op.w_gate.dtype == DType::MQ4G256V2
        && op.w_up.dtype == DType::MQ4G256V2
        && gpu.flags.gfx12_mq4v2_fp8_gateup
        && !gpu.flags.hfq4g256_ldsstage_wmma
        && fp8_prep.is_some()
    {
        // gfx1201 FP8-stream (uniform): the producer already emitted the
        // fp8 pre-pass planes; consume them directly with the launch twin
        // of the family's fp8 route — no pack launch. Same fp8 intercept
        // conditions as the uniform router (iu4 divergence excluded by
        // producer-side ordering: fp8_prep implies iu4_prep is None).
        let silu_h = gpu.fp8_silu_h_active(n, op.w_gate.k, op.w_gate.m, op.w_up.m)
            && op.w_down.dtype == DType::MQ4G256V2
            && gpu.fp8_stream_active(n, op.w_down.k);
        gpu.gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2_fp8_bt12_prepared(
            op.w_gate.buf,
            op.w_up.buf,
            fp8_prep.as_ref().unwrap(),
            op.gate,
            op.up,
            op.w_gate.m,
            op.w_up.m,
            op.w_gate.k,
            n,
            silu_h,
        )?;
        return Ok(if silu_h {
            FfnGateOutput::Fp8H
        } else {
            FfnGateOutput::Separate
        });
    } else {
        run_fused_gate_up_key(
            gpu,
            crate::families::fused_qkv::fused_gate_up_key_for(op.w_gate.dtype),
            op.w_gate.buf,
            op.w_up.buf,
            op.x_rot,
            op.gate,
            op.up,
            op.w_gate.m,
            op.w_up.m,
            op.w_gate.k,
            n,
        )?;
    }
    Ok(FfnGateOutput::Separate)
}

#[cfg(feature = "deltanet")]
#[allow(clippy::too_many_arguments)]
/// Prescaffold (behavior-only) extraction for S4-f16-residual-inputs.
///
/// Same statements, same order, same launches as the inlined block.
/// S9-mq4v2-persistent-prologues will issue `try_mq4v2_persistent_prologue`
/// from inside this hook after S3/S4 land.
///
/// `h_source` tells whether gate/up emitted separate planes, IU4 h, or FP8 h;
/// h is stored in `gate_ffn_batch` and needs its matching down producer.
fn swiglu_ffn_down_batched(
    gpu: &mut Gpu,
    op: &SwigluFfnOp<'_>,
    b: &SwigluFfnBatch<'_>,
    n: usize,
    h_source: FfnGateOutput,
) -> HipResult<()> {
    let hidden_dim = b.hidden_dim;
    if try_packed_mq4_down(gpu, op, b, n, h_source)? {
        return Ok(());
    }
    // S4: one silu*up+FWHT+F16 producer + direct-F16 residual GEMM instead
    // of fused_silu_mul_rotate_mq_batched + convert.
    if h_source == FfnGateOutput::Separate
        && s4_residual_fast(gpu, b.chain_verify, op.w_down.dtype, &b.epilogue, n)
    {
        let k = op.w_down.k;
        let m = op.w_down.m;
        if k > 0 && k % 256 == 0 && k == hidden_dim {
            fused_silu_mul_rotate_mq_f16_batched_for(
                gpu,
                &op.w_down,
                op.gate,
                op.up,
                b.hidden_f16,
                hidden_dim,
                n,
            )?;
            let x_f16 = b.hidden_f16.sub_offset(0, n * hidden_dim);
            gpu.gemm_mq4g256v2_residual_wmma_f16(op.w_down.buf, &x_f16, op.x, m, hidden_dim, n)?;
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
        op.w_down.dtype,
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
    if h_source == FfnGateOutput::A8H
        && gpu.a8_prefill_active(n, hidden_dim)
        && op.w_down.dtype == DType::MQ4G256V2
        && matches!(&b.epilogue, BatchEpilogue::Residual)
    {
        let xq = if let Some(prep) = try_a8_hin_prepared(gpu, &op.w_down, op.gate, hidden_dim, n)? {
            gpu.int8_mmq_prepared_ptr(&prep, hidden_dim, n)?
        } else {
            rotate_x_mq_batched_for(gpu, &op.w_down, op.gate, op.hidden, hidden_dim, n)?;
            gpu.ensure_int8_mmq_x(op.hidden, n, hidden_dim)?
        };
        return gpu.gemm_mq4g256v2_mmq_add_prequant_i8(
            op.w_down.buf,
            xq,
            op.x,
            op.w_down.m,
            hidden_dim,
            n,
        );
    }
    let mut iu4_prep: Option<rdna_compute::Int4MmqPrepared> = None;
    let mut fp8_prep: Option<rdna_compute::Mq4v2Fp8Prepared> = None;
    if w_down_is_mq {
        // C2: SwiGLU/FWHT IU4 producer for w_down (emit_f32=false). Residual only —
        // Partial TP epilogue still needs the f32 rotated buffer.
        if matches!(&b.epilogue, BatchEpilogue::Residual) && h_source != FfnGateOutput::Fp8H {
            iu4_prep = if h_source == FfnGateOutput::Iu4H {
                Some(iu4_hin_prepared(gpu, &op.w_down, op.gate, hidden_dim, n)?)
            } else {
                try_iu4_silu_prepared(
                    gpu, &op.w_down, op.gate, op.up, op.hidden, hidden_dim, n, false,
                )?
            };
            if iu4_prep.is_none() {
                // gfx1201 slice-1: fused silu+quant producer (bit-identical).
                iu4_prep = try_gfx12_silu_quant_fused_prepared(
                    gpu, &op.w_down, op.gate, op.up, op.hidden, hidden_dim, n, false,
                )?;
            }
        }
        if iu4_prep.is_none() {
            fp8_prep = try_gfx12_fp8_stream_silu_prepared(
                gpu,
                &op.w_down,
                op.gate,
                op.up,
                op.hidden,
                hidden_dim,
                n,
                h_source == FfnGateOutput::Fp8H,
            )?;
        }
        if iu4_prep.is_none() && fp8_prep.is_none() {
            // F2: AWQ-aware silu_mul+rotate for w_down input.
            fused_silu_mul_rotate_mq_batched_for(
                gpu, &op.w_down, op.gate, op.up, op.hidden, hidden_dim, n,
            )?;
        }
    } else {
        gpu.silu_mul_f32(op.gate, op.up, op.hidden)?;
    }
    // Batched w_down + residual/partial.
    if let Some(prep) = &iu4_prep {
        gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(
            op.w_down.buf,
            prep,
            op.x,
            op.w_down.m,
            op.w_down.k,
            n,
        )?;
    } else if let Some(prep) = &fp8_prep {
        dispatch_batched_fp8_lloyd_epilogue(gpu, op.x, &op.w_down, prep, &b.epilogue, n)?;
    } else {
        dispatch_batched_gemm_epilogue(
            gpu,
            op.x,
            op.x_rot,
            &op.w_down,
            op.hidden,
            &b.epilogue,
            n,
            b.q8_wmma_arch,
            false,
        )?;
    }
    Ok(())
}

#[cfg(feature = "deltanet")]
/// Batched dense SwiGLU FFN: norm + gate/up (optionally emitting SwiGLU
/// directly), then the down projection into the residual or TP partial.
pub fn execute_swiglu_ffn_batched(
    gpu: &mut Gpu,
    op: &SwigluFfnOp<'_>,
    b: &SwigluFfnBatch<'_>,
) -> HipResult<()> {
    let n = op.rows;
    let f1lite = f1lite_ffn_eligible(
        gpu,
        &op.w_gate,
        &op.w_up,
        &op.w_down,
        &b.epilogue,
        b.chain_verify,
        b.hidden_dim,
        n,
    );
    let h_source = swiglu_ffn_gate_up_batched(gpu, op, b, n, f1lite)?;
    swiglu_ffn_down_batched(gpu, op, b, n, h_source)
}

pub fn q8_prefill_wmma_enabled_from_env(value: Option<&str>, _arch: &str, has_wmma: bool) -> bool {
    if !has_wmma {
        return false;
    }
    match value {
        Some("0") | Some("off") | Some("false") => false,
        Some("1") | Some("on") | Some("true") => true,
        _ => true,
    }
}

pub fn q8_prefill_wmma_enabled(gpu: &Gpu) -> bool {
    q8_prefill_wmma_enabled_from_env(
        hipfire_config::developer_var("HIPFIRE_Q8_PREFILL_WMMA")
            .ok()
            .as_deref(),
        gpu.arch.as_str(),
        gpu.arch_caps.has_wmma(),
    )
}

/// T-B: epilogue-free twin of [`try_iu4_rotate_prepared`] for the MoE wo
/// sites, which have no `epilogue` param and always accumulate residual
/// into x_batch. Same gate minus the Residual check. The f32 `x_rot` store
/// is always skipped for the same reason as the Residual twin.
pub fn try_iu4_rotate_prepared_no_epilogue(
    gpu: &mut Gpu,
    wo: &WeightRef<'_>,
    x: &GpuTensor,
    k: usize,
    n: usize,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    if wo.dtype != DType::MQ4G256V2 || !gpu.iu4_producer_sidecar_active(n, k) {
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep = gpu.rotate_x_mq_i4_batched(x, wo.awq_scale, None, res, k, n)?;
    Ok(Some(prep))
}

#[cfg(test)]
mod packed_mq4_tests {
    use super::{packed_mq4_down_admitted, packed_mq4_ffn_gate_up_admitted};
    use rdna_compute::DType;

    #[test]
    fn packed_mq4_ffn_gate_up_route_is_exact_and_opt_in() {
        let shape = (DType::MQ4G256, 17_408, 5_120);
        let v2 = (DType::MQ4G256V2, 17_408, 5_120);
        for n in [256, 512, 768] {
            assert!(packed_mq4_ffn_gate_up_admitted(true, shape, shape, n));
            // MQ4V2 keeps the landed V2C / IU4 routes.
            assert!(!packed_mq4_ffn_gate_up_admitted(true, v2, v2, n));
            assert!(!packed_mq4_ffn_gate_up_admitted(false, shape, shape, n));
            assert!(!packed_mq4_ffn_gate_up_admitted(false, v2, v2, n));
            assert!(!packed_mq4_ffn_gate_up_admitted(true, shape, v2, n));
            assert!(!packed_mq4_ffn_gate_up_admitted(true, v2, shape, n));
        }
        for n in [0, 1, 128, 255, 257] {
            assert!(!packed_mq4_ffn_gate_up_admitted(true, shape, shape, n));
            assert!(!packed_mq4_ffn_gate_up_admitted(true, v2, v2, n));
        }
        for rejected in [
            (DType::MQ4G256V2Lloyd, 17_408, 5_120),
            (DType::MQ4G256Lloyd, 17_408, 5_120),
            (DType::HFQ4G256, 17_408, 5_120),
            (DType::Q8_0, 17_408, 5_120),
            (DType::MQ4G256, 8_704, 5_120),
            (DType::MQ4G256, 17_408, 2_560),
        ] {
            assert!(!packed_mq4_ffn_gate_up_admitted(true, rejected, shape, 256));
            assert!(!packed_mq4_ffn_gate_up_admitted(true, shape, rejected, 256));
            assert!(!packed_mq4_ffn_gate_up_admitted(
                true, rejected, rejected, 256
            ));
        }
    }

    #[test]
    fn packed_mq4_down_refuses_prepared_partial_and_nonuniform_inputs() {
        assert!(!packed_mq4_down_admitted(
            true,
            true,
            true,
            (DType::MQ4G256V2, 5120, 17408),
            17408
        ));
        for dtype in [DType::MQ4G256] {
            let shape = (dtype, 5120, 17408);
            assert!(packed_mq4_down_admitted(true, true, true, shape, 17408));
            assert!(!packed_mq4_down_admitted(false, true, true, shape, 17408));
            assert!(!packed_mq4_down_admitted(true, false, true, shape, 17408));
            assert!(!packed_mq4_down_admitted(true, true, false, shape, 17408));
            assert!(!packed_mq4_down_admitted(true, true, true, shape, 8704));
            assert!(!packed_mq4_down_admitted(
                true,
                true,
                true,
                (dtype, 2560, 17408),
                17408
            ));
        }
        for dtype in [
            DType::MQ4G256V2Lloyd,
            DType::MQ4G256Lloyd,
            DType::HFQ4G256,
            DType::Q8_0,
        ] {
            assert!(!packed_mq4_down_admitted(
                true,
                true,
                true,
                (dtype, 5120, 17408),
                17408
            ));
        }
    }
}
