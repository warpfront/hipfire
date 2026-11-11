// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched (prefill / multi-row verify) projection library and the rows > 1
//! executor of the SwiGLU FFN sublayer: IU4/A8/FP8 activation producers,
//! key-selected GEMM runners, residual/partial epilogues and their arch- and
//! dtype-gated fast routes. Moved verbatim from the Qwen3.5 prefill path; the
//! views carry the field names of the architecture-owned weights and scratch
//! they borrow, so every route issues the same launches in the same order.

#![allow(clippy::too_many_arguments)]

use crate::context::DispatchCtx;
use crate::families::fused_qkv::FusedQkvFamily;
use crate::families::gemm::GemmFamily;
use crate::families::gemv::WeightRef;
use crate::types::DispatchError;
use hip_bridge::{HipError, HipResult};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::LazyLock;

static GEMM: LazyLock<GemmFamily> = LazyLock::new(GemmFamily::new);
pub(crate) static FUSED_QKV: LazyLock<FusedQkvFamily> = LazyLock::new(FusedQkvFamily::new);

pub(crate) fn hip_err(e: DispatchError) -> HipError {
    HipError::new(0, &e.to_string())
}

/// Frozen AR/verify discriminator for the DFlash launch-fusion project.
///
/// `Off` is the behavior-preserving default: every hook takes the pre-change
/// path. `ChainVerify` arms the exact-shape fast routes (linear chain verify
/// only — tree verify stays `Off`). Computed once in
/// `verify_dflash_block_inner` (`ChainVerify` iff `tree_verify` is `None`)
/// and threaded through the verify forwards; every other caller passes `Off`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DflashFusionCtx {
    Off,
    ChainVerify,
    /// Linear-chain verify whose exact model/KV/context envelope admits the
    /// gfx1100 FA2 split-KV route. This only widens capture-safe auxiliary
    /// kernels; ordinary eager ChainVerify behavior is otherwise unchanged.
    ChainVerifySplit,
}

impl DflashFusionCtx {
    #[inline]
    pub const fn is_chain_verify(self) -> bool {
        matches!(self, Self::ChainVerify | Self::ChainVerifySplit)
    }

    #[inline]
    pub const fn split_verify_active(self) -> bool {
        matches!(self, Self::ChainVerifySplit)
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
    /// s_tape scratch of the batch scratch.
    pub parent_indices: Option<&'a GpuTensor>,
}

/// Row semantics of one batched forward chunk: a sequential prompt/verify
/// window, or independent decode lanes at their own positions.
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

/// Row cadence of the widened ordinary prefill: larger contiguous GEMM/PBS
/// chunks still commit DeltaNet and flash attention per 512-row view. Only
/// `None` (legacy cadence) and `Some(512)` are legal commit strides.
pub const WIDENED_COMMIT_ROWS: usize = 512;

/// Batched scratch rows of one hybrid-decoder chunk, `[rows, …]` per tensor.
#[derive(Clone, Copy)]
pub struct BatchScratch<'a> {
    /// Small-arena scratch: the packed MQ4 routes need full-size planes.
    pub lean: bool,
    pub x_batch: &'a GpuTensor,
    /// Un-rotated RMSNorm rows: the PARO Givens projections rotate from it.
    pub x_norm_batch: &'a GpuTensor,
    pub x_rot_batch: &'a GpuTensor,
    pub x_rot_f16_batch: &'a GpuTensor,
    pub gate_ffn_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub ffn_hidden_batch: &'a GpuTensor,
    pub ffn_hidden_f16_batch: &'a GpuTensor,
    pub dn_qkv_batch: &'a GpuTensor,
    pub dn_z_batch: &'a GpuTensor,
    pub dn_z_fold_batch: &'a GpuTensor,
    pub dn_alpha_batch: &'a GpuTensor,
    pub dn_beta_batch: &'a GpuTensor,
    pub dn_q_raw_batch: &'a GpuTensor,
    pub dn_k_raw_batch: &'a GpuTensor,
    pub dn_v_batch: &'a GpuTensor,
    pub dn_q_batch: &'a GpuTensor,
    pub dn_k_batch: &'a GpuTensor,
    pub dn_attn_out_batch: &'a GpuTensor,
    pub dn_normed_batch: &'a GpuTensor,
    pub dn_normed_rot_batch: &'a GpuTensor,
    pub dn_normed_rot_f16_batch: &'a GpuTensor,
    pub dn_s_tape_q8: Option<&'a GpuTensor>,
    pub dn_s_tape_scales: Option<&'a GpuTensor>,
    pub dn_s_tape_f32: Option<&'a GpuTensor>,
    pub fa_q_full_batch: &'a GpuTensor,
    pub fa_q_batch: &'a GpuTensor,
    pub fa_gate_batch: &'a GpuTensor,
    pub fa_k_batch: &'a GpuTensor,
    pub fa_v_batch: &'a GpuTensor,
    pub fa_attn_out_batch: &'a GpuTensor,
    pub fa_attn_out_rot_batch: &'a GpuTensor,
    pub fa_attn_out_rot_f16_batch: &'a GpuTensor,
    /// Physical KV slot per row.
    pub positions: &'a GpuTensor,
    /// RoPE phase per row (DDTree depth positions; else unused).
    pub rope_positions: &'a GpuTensor,
    /// Explicit per-row M-RoPE phases of a VL prompt chunk. `Some` makes the
    /// batched full-attention prep rotate with the batched M-RoPE kernel at
    /// these phases (and bypass every fused prep that bakes 1D rope); `None`
    /// (every other caller) keeps the 1D routes.
    pub mrope: Option<MropeRows<'a>>,
}

/// `[rows][3]` absolute `(t, h, w)` phases (i32 stored in an F32 tensor) of a
/// chunk plus the checkpoint's `mrope_section`.
#[derive(Clone, Copy)]
pub struct MropeRows<'a> {
    pub pos3: &'a GpuTensor,
    pub section: [usize; 3],
}

/// Weights of one dense SwiGLU FFN sublayer, as the rows route of
/// [`crate::pipeline::gated_mlp::execute_product_rows`] binds them.
#[derive(Clone, Copy)]
pub(crate) struct FfnLayerView<'a> {
    pub ffn_norm: &'a GpuTensor,
    pub w_gate: WeightRef<'a>,
    pub w_up: WeightRef<'a>,
    pub w_down: WeightRef<'a>,
    /// Mixer kind of the owning layer (`"LA"` / `"FA"`), for error messages.
    pub kind: &'static str,
}

/// Refuse the typed-only weight formats (and the not-yet-runnable ggml block
/// formats) every generic batched route rejects.
pub fn reject_typed_only_format(dtype: DType) -> HipResult<()> {
    if dtype == DType::MQ4G128V2 {
        Err(HipError::new(
            0,
            "this model uses the MQ4G128V2 weight format (qt=53), which only Qwen4 (Flash-Next) models can run; load a Qwen4 model or re-quantize this one to a different format",
        ))
    } else if let Some(name) = dtype.mq_i4_name() {
        Err(HipError::new(
            0,
            &format!(
                "this model uses the {name} weight format (MQ-I4 / E8P-I4, qt=131..=136), an expert-slab format only Qwen4 (Flash-Next) models can run, and it has no generic GPU kernel; re-quantize it to a different format"
            ),
        ))
    } else if let Some(refusal) = dtype.ggml_refusal() {
        // GGUF block formats (`DType::Ggml*`): parsed and planned, but no ggml
        // kernel is wired, so no route may run or reinterpret them.
        Err(HipError::new(0, &refusal))
    } else {
        Ok(())
    }
}

/// Batched AWQ-aware RMSNorm + FWHT rotation. `next_linear` is the FIRST
/// linear after the rotation; its AWQ scale sidecar (if any) selects the
/// `_awq_batched` kernel, which divides activations by `awq_scale[i]` before
/// the FWHT. Q/K/V and gate/up share one input, hence one scale.
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
    reject_typed_only_format(next_linear.dtype)?;
    if let Some(awq) = next_linear.awq_scale {
        gpu.fused_rmsnorm_rotate_mq_awq_batched(x, norm_weight, awq, x_rot, k, eps, batch_size)
    } else {
        gpu.fused_rmsnorm_rotate_mq_batched(x, norm_weight, x_rot, k, eps, batch_size)
    }
}

/// Batched AWQ-aware FWHT rotation; `next_linear` consumes `x_rot`.
pub fn rotate_x_mq_batched_for(
    gpu: &mut Gpu,
    next_linear: &WeightRef<'_>,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    batch_size: usize,
) -> HipResult<()> {
    reject_typed_only_format(next_linear.dtype)?;
    if let Some(awq) = next_linear.awq_scale {
        gpu.rotate_x_mq_awq_batched(x, awq, x_rot, k, batch_size)
    } else {
        gpu.rotate_x_mq_batched(x, x_rot, k, batch_size)
    }
}

/// [`fused_rmsnorm_rotate_mq_batched_for`] storing exact FP16 into
/// `x_rot_f16` (bit-identical to the F32 producer followed by
/// `convert_f32_to_f16`) for the `*_wmma_f16` GEMM entries.
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
    reject_typed_only_format(next_linear.dtype)?;
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

/// Batched AWQ-aware `silu(gate) * up` + FWHT rotation; `down_proj_weight`
/// consumes `x_rot`.
pub fn fused_silu_mul_rotate_mq_batched_for(
    gpu: &mut Gpu,
    down_proj_weight: &WeightRef<'_>,
    gate: &GpuTensor,
    up: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    batch_size: usize,
) -> HipResult<()> {
    reject_typed_only_format(down_proj_weight.dtype)?;
    if let Some(awq) = down_proj_weight.awq_scale {
        gpu.fused_silu_mul_rotate_mq_awq_batched(gate, up, awq, x_rot, k, batch_size)
    } else {
        gpu.fused_silu_mul_rotate_mq_batched(gate, up, x_rot, k, batch_size)
    }
}

/// [`fused_silu_mul_rotate_mq_batched_for`] writing the F16 sidecar directly
/// (byte-identical to the F32 producer followed by `convert_f32_to_f16`).
pub fn fused_silu_mul_rotate_mq_f16_batched_for(
    gpu: &mut Gpu,
    down_proj_weight: &WeightRef<'_>,
    gate: &GpuTensor,
    up: &GpuTensor,
    x_rot_f16: &GpuTensor,
    k: usize,
    batch_size: usize,
) -> HipResult<()> {
    reject_typed_only_format(down_proj_weight.dtype)?;
    if let Some(awq) = down_proj_weight.awq_scale {
        gpu.fused_silu_mul_rotate_mq_awq_f16_batched(gate, up, awq, x_rot_f16, k, batch_size)
    } else {
        gpu.fused_silu_mul_rotate_mq_f16_batched(gate, up, x_rot_f16, k, batch_size)
    }
}

/// Producer-emitted A8 RMSNorm/FWHT, selected only for uniform MQ4v2 weights.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_a8_rmsnorm_prepared(
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
    gpu.fused_rmsnorm_rotate_mq_i8_batched(x, weight, next.awq_scale, None, reservation, k, eps, n)
        .map(Some)
}

fn try_a8_hin_prepared(
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
    gpu.fused_silu_hin_rotate_mq_i8_batched(h, awq, reservation, k, n)
        .map(Some)
}

/// Symmetric qt49 (MQ3G256V2) weight that enters the gfx1201 A4 producer/consumer
/// pair: validated provenance, the A4 flag, exact gfx1201, and no active
/// higher-priority A8 route. Every producer admitted here has a prepared
/// MQ3V25 consumer at its call site (no producer runs without its consumer).
pub(crate) fn mq3v25_a4_weight(gpu: &Gpu, dtype: DType) -> bool {
    dtype == DType::MQ3G256V2
        && gpu.arch == "gfx1201"
        && !gpu.flags.mq3v25_a8_enabled(gpu.mq3v2_symmetric)
        && gpu.flags.mq3v25_a4_enabled(gpu.mq3v2_symmetric)
}

/// The shared-input producer of a projection group anchored at `anchor` may run
/// for a qt49 A4 route only when every sibling projection has the same dtype
/// (the one prepared consumer reads all of them); any other anchor is
/// unaffected. Keeps the rule "no producer runs without its consumer".
pub(crate) fn mq3v25_a4_consumers_agree(gpu: &Gpu, anchor: DType, siblings: &[DType]) -> bool {
    !mq3v25_a4_weight(gpu, anchor) || siblings.iter().all(|&dtype| dtype == anchor)
}

/// C2: when MQ4V2 + IU4 producer sidecar is live, emit block_i4_128 from
/// RMSNorm/FWHT and return a prepared handle. `None` → caller keeps the
/// incumbent fused_rmsnorm + standalone-quantizer path.
pub(crate) fn try_iu4_rmsnorm_prepared(
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

/// ADD-epilogue fold (gfx1151 V2B, `HIPFIRE_V2B_ADDEPI`): whether the FFN
/// norm after an out-projection will take the IU4 RMSNorm route, the fold's
/// only consumer. When it will, the out-projection arms `pbs.gate_ffn_batch`
/// as the delta for its residual GEMM: that buffer is dead from the
/// out-projection until the gate/up GEMM, which runs after the norm.
fn residual_fold_consumer_ready(
    gpu: &Gpu,
    w_gate: &WeightRef<'_>,
    fusion: DflashFusionCtx,
    n: usize,
    dim: usize,
) -> bool {
    w_gate.dtype == DType::MQ4G256V2
        && gpu.iu4_producer_sidecar_active(n, dim)
        && !mq_f16_projection_fast_route(gpu, fusion, n, dim)
}

/// Residual GEMM of an out-projection whose FFN norm follows directly:
/// offers the fold delta when [`residual_fold_consumer_ready`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn out_proj_residual_iu4_prepared(
    gpu: &mut Gpu,
    pbs: &BatchScratch<'_>,
    wo: &WeightRef<'_>,
    w_gate: &WeightRef<'_>,
    prep: &rdna_compute::Int4MmqPrepared,
    fusion: DflashFusionCtx,
    n: usize,
) -> HipResult<()> {
    let fold = residual_fold_consumer_ready(gpu, w_gate, fusion, n, wo.m);
    if fold {
        gpu.arm_residual_fold(&pbs.gate_ffn_batch)?;
    }
    let result =
        gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(&wo.buf, prep, &pbs.x_batch, wo.m, wo.k, n);
    if fold {
        gpu.disarm_residual_fold();
    }
    result
}

pub(crate) fn out_proj_residual_i8_prepared(
    gpu: &mut Gpu,
    pbs: &BatchScratch<'_>,
    wo: &WeightRef<'_>,
    w_gate: &WeightRef<'_>,
    prep: &rdna_compute::Int8MmqPrepared,
    fusion: DflashFusionCtx,
    n: usize,
) -> HipResult<()> {
    let fold = residual_fold_consumer_ready(gpu, w_gate, fusion, n, wo.m);
    if fold {
        gpu.arm_residual_fold(&pbs.gate_ffn_batch)?;
    }
    let result =
        gpu.gemm_mq4g256v2_residual_wmma_i8_prepared(&wo.buf, prep, &pbs.x_batch, wo.m, wo.k, n);
    if fold {
        gpu.disarm_residual_fold();
    }
    result
}

/// C2: SwiGLU/FWHT IU4 producer for w_down. `None` → incumbent path.
fn try_iu4_silu_prepared(
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

/// The gate/up output format determines which down-projection producer may
/// consume it. An h tensor is not interchangeable with separate gate/up.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FfnGateOutput {
    Separate,
    /// A4 fusion (`HIPFIRE_V2B_A4_EPI`, default on gfx1151; `=0` opts out): the gate/up epilogue wrote the w_down A4
    /// sidecar itself; `gate_ffn_batch` (h) was not written.
    Iu4A4(rdna_compute::Int4MmqDownPrepared),
    Iu4H,
    Fp8H,
    A8H,
}

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
fn try_packed_mq4_down(
    gpu: &mut Gpu,
    down: &WeightRef<'_>,
    pbs: &BatchScratch<'_>,
    hidden_dim: usize,
    n: usize,
    epilogue: &BatchEpilogue<'_>,
    h_source: FfnGateOutput,
) -> HipResult<bool> {
    if !packed_mq4_down_admitted(
        !pbs.lean && gpu.packed_mq4_admitted(down.m, down.k, n),
        h_source == FfnGateOutput::Separate,
        matches!(epilogue, BatchEpilogue::Residual),
        (down.dtype, down.m, down.k),
        hidden_dim,
    ) {
        return Ok(false);
    }
    gpu.flush_residual_fold()?;
    fused_silu_mul_rotate_mq_batched_for(
        gpu,
        down,
        &pbs.gate_ffn_batch,
        &pbs.up_batch,
        &pbs.ffn_hidden_batch,
        hidden_dim,
        n,
    )?;
    run_residual_gemm_key(
        gpu,
        crate::types::KernelKey::GemmMq4PackedResidual,
        &down.buf,
        down.dtype,
        &pbs.ffn_hidden_batch,
        &pbs.x_batch,
        down.m,
        down.k,
        n,
    )?;
    Ok(true)
}

/// Uniform MQ4G256 (v1) only; the explicit packed opt-in leaves all native
/// routes unchanged when disabled. MQ4V2 (qt44, e.g. mq4-xts) keeps the landed
/// gfx1100 builder V2C / IU4 routes: packed was not measured faster there.
/// Lloyd and mixed wire formats are excluded.
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
fn try_packed_mq4_ffn_gate_up(
    gpu: &mut Gpu,
    gate: &WeightRef<'_>,
    up: &WeightRef<'_>,
    norm: &GpuTensor,
    norm_eps: f32,
    pbs: &BatchScratch<'_>,
    n: usize,
) -> HipResult<bool> {
    if !packed_mq4_ffn_gate_up_admitted(
        !pbs.lean && gpu.packed_mq4_admitted(gate.m, gate.k, n),
        (gate.dtype, gate.m, gate.k),
        (up.dtype, up.m, up.k),
        n,
    ) {
        return Ok(false);
    }
    gpu.flush_residual_fold()?;
    let separate_awq_inputs = gate.awq_scale.is_some() || up.awq_scale.is_some();
    for (index, (weight, output)) in [(gate, &pbs.gate_ffn_batch), (up, &pbs.up_batch)]
        .into_iter()
        .enumerate()
    {
        // Sidecars are per-weight and may differ (or exist on only one side).
        // Reuse the rotated input only when neither projection carries AWQ.
        if index == 0 || separate_awq_inputs {
            fused_rmsnorm_rotate_mq_batched_for(
                gpu,
                &pbs.x_batch,
                norm,
                weight,
                &pbs.x_rot_batch,
                weight.k,
                norm_eps,
                n,
            )?;
        }
        run_plain_gemm_key(
            gpu,
            crate::types::KernelKey::GemmMq4Packed,
            &weight.buf,
            weight.dtype,
            &pbs.x_rot_batch,
            output,
            weight.m,
            weight.k,
            n,
        )?;
    }
    Ok(true)
}

/// F1-lite: true when this FFN may fold SwiGLU into the gate/up GEMM, i.e.
/// the down projection would take the AWQ IU4 SwiGLU producer
/// ([`try_iu4_silu_prepared`] with an AWQ scale: uniform MQ4G256V2, `Residual`
/// epilogue, no S4 f16 route, portable gfx11 sidecar live; or on exact
/// gfx1201 [`try_gfx12_silu_quant_fused_prepared`]'s AWQ twin) and gate/up are
/// MQ4G256V2 of equal shape (or, when [`mq3v25_a4_weight`] admits, equal-dtype
/// symmetric qt49 gate/up/down feeding the same AWQ h producer). Decided once
/// per FFN, before gate/up; the gate/up hook then reports whether it actually
/// emitted h (`Gpu::gemm_gate_up_silu_mq4g256v2_iu4_prepared` /
/// `gemm_gate_up_silu_mq3v25_iu4_prepared` add the per-arch tile rule and
/// `HIPFIRE_F1LITE`).
#[allow(clippy::too_many_arguments)]
fn f1lite_ffn_eligible(
    gpu: &Gpu,
    w_gate: &WeightRef<'_>,
    w_up: &WeightRef<'_>,
    w_down: &WeightRef<'_>,
    epilogue: &BatchEpilogue<'_>,
    fusion: DflashFusionCtx,
    hidden_dim: usize,
    n: usize,
) -> bool {
    (w_gate.dtype == DType::MQ4G256V2 || mq3v25_a4_weight(gpu, w_gate.dtype))
        && w_up.dtype == w_gate.dtype
        && w_gate.m == hidden_dim
        && w_up.m == hidden_dim
        && w_gate.k == w_up.k
        && w_down.dtype == w_gate.dtype
        && w_down.awq_scale.is_some()
        && w_down.k == hidden_dim
        && matches!(epilogue, BatchEpilogue::Residual)
        && !s4_residual_fast(gpu, fusion, w_down.dtype, epilogue, n)
        && if gpu.a8_prefill_active(n, w_gate.k) {
            // A8 gate/up SiLU entry tiles 128 rows; a sidecar-capable gfx11
            // must not admit f1lite for a hidden_dim the A8 entry rejects.
            hidden_dim % 128 == 0
        } else {
            gpu.iu4_producer_sidecar_active(n, hidden_dim)
                || gpu.iu4_silu_quant_fused_active(n, hidden_dim)
        }
}

/// F1-lite: AWQ IU4 producer for w_down reading h = silu(gate)*up, already
/// formed by the gate/up GEMM epilogue in `h`. Byte-identical sidecar to
/// [`try_iu4_silu_prepared`]'s.
fn iu4_hin_prepared(
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
fn try_gfx12_silu_quant_fused_prepared(
    gpu: &mut Gpu,
    w_down: &WeightRef<'_>,
    gate: &GpuTensor,
    up: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    n: usize,
    emit_f32: bool,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    if !(w_down.dtype == DType::MQ4G256V2 || mq3v25_a4_weight(gpu, w_down.dtype))
        || !gpu.iu4_silu_quant_fused_active(n, k)
    {
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep = gpu.fused_silu_mul_rotate_mq_i4_gfx12_batched(
        gate,
        up,
        w_down.awq_scale,
        // The MQ3 consumer's small-tail and fallback readers need the f32 plane.
        if emit_f32 || w_down.dtype == DType::MQ3G256V2 {
            Some(x_rot)
        } else {
            None
        },
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
pub(crate) fn try_gfx12_rmsnorm_quant_fused_prepared(
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
    if !(next_linear.dtype == DType::MQ4G256V2 || mq3v25_a4_weight(gpu, next_linear.dtype))
        || !gpu.iu4_producer_quant_fused_active(n, k)
    {
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep = gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(
        x,
        norm_weight,
        next_linear.awq_scale,
        // The MQ3 consumer's small-tail and fallback readers need the f32 plane.
        if emit_f32 || next_linear.dtype == DType::MQ3G256V2 {
            Some(x_rot)
        } else {
            None
        },
        res,
        k,
        eps,
        n,
    )?;
    Ok(Some(prep))
}

/// gfx1201 MQ4v2 FP8-WMMA consumer family fed by an FP8-stream producer.
/// The family's `HIPFIRE_GFX12_MQ4V2_FP8_*` opt-out decides the activation
/// format of both halves of the producer/consumer pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fp8StreamConsumer {
    /// LA 4-way qkvza (`HIPFIRE_GFX12_MQ4V2_FP8_QKVZA`).
    Qkvza,
    /// FFN gate/up (`HIPFIRE_GFX12_MQ4V2_FP8_GATEUP`).
    GateUp,
    /// FA 3-way qkv (`HIPFIRE_GFX12_MQ4V2_FP8_QKV`).
    Qkv,
    /// Down / out-proj residual (`HIPFIRE_GFX12_MQ4V2_FP8_RESID`).
    Residual,
}

/// Whether `consumer` over weights `dtypes` takes the FP8 prepared route.
/// An admitted FP8-stream producer emits only the FP8 planes (in-register
/// producers leave the f32 `x_rot` unwritten), so it may run only when this
/// holds. Uniform MQ4G256V2 consumers have an F16 route behind their family
/// flag (gate/up also behind the `HIPFIRE_HFQ4G256_LDSSTAGE` WMMA); Lloyd
/// consumers are FP8-only, so the family flags never route them elsewhere.
/// Mixed or non-MQ4v2 sets have no FP8 prepared consumer.
fn fp8_stream_consumer_takes_fp8(
    flags: &rdna_compute::FeatureFlags,
    consumer: Fp8StreamConsumer,
    dtypes: &[DType],
) -> bool {
    if dtypes.is_empty() {
        return false;
    }
    if all_mq4v2_lloyd(dtypes) {
        return true;
    }
    if !dtypes.iter().all(|d| *d == DType::MQ4G256V2) {
        return false;
    }
    match consumer {
        Fp8StreamConsumer::Qkvza => flags.gfx12_mq4v2_fp8_qkvza,
        Fp8StreamConsumer::GateUp => flags.gfx12_mq4v2_fp8_gateup && !flags.hfq4g256_ldsstage_wmma,
        Fp8StreamConsumer::Qkv => flags.gfx12_mq4v2_fp8_qkv,
        Fp8StreamConsumer::Residual => flags.gfx12_mq4v2_fp8_resid,
    }
}

/// Single admission for every gfx1201 FP8-stream producer and for each
/// decision that anticipates one (gate/up h epilogue, FA gate in place, GDN
/// scan-out format): the consumer takes FP8 and the per-call
/// `fp8_stream_active` gate holds.
pub(crate) fn gfx12_fp8_stream_admitted(
    gpu: &Gpu,
    consumer: Fp8StreamConsumer,
    dtypes: &[DType],
    n: usize,
    k: usize,
) -> bool {
    fp8_stream_consumer_takes_fp8(&gpu.flags, consumer, dtypes) && gpu.fp8_stream_active(n, k)
}

/// gfx1201 FP8-stream: RMSNorm/FWHT producer that emits byte-identical
/// `prepare_mq4v2_fp8_x_f32` (scale_mode=1) planes for the `consumer`
/// projection over `consumer_dtypes`. `None` → caller keeps the incumbent
/// producer + standalone-pack path. Admission is
/// [`gfx12_fp8_stream_admitted`] (consumer takes FP8; exact gfx1201, eager,
/// batch >= 64, K % 256 == 0; partial N tiles are masked in-kernel, so odd
/// tails run unpadded). On `HIPFIRE_FP8_PROD_INREG=1` the producer leaves
/// `x_rot` untouched; the admitted consumer reads only the prepared FP8
/// planes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_gfx12_fp8_stream_rmsnorm_prepared(
    gpu: &mut Gpu,
    x: &GpuTensor,
    norm_weight: &GpuTensor,
    next_linear: &WeightRef<'_>,
    consumer: Fp8StreamConsumer,
    consumer_dtypes: &[DType],
    x_rot: &GpuTensor,
    k: usize,
    eps: f32,
    n: usize,
) -> HipResult<Option<rdna_compute::Mq4v2Fp8Prepared>> {
    if !gfx12_fp8_stream_admitted(gpu, consumer, consumer_dtypes, n, k) {
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
fn try_gfx12_fp8_stream_silu_prepared(
    gpu: &mut Gpu,
    w_down: &WeightRef<'_>,
    gate: &GpuTensor,
    up: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    n: usize,
    h_ready: bool,
) -> HipResult<Option<rdna_compute::Mq4v2Fp8Prepared>> {
    if !gfx12_fp8_stream_admitted(gpu, Fp8StreamConsumer::Residual, &[w_down.dtype], n, k) {
        return Ok(None);
    }
    let prep = if h_ready {
        gpu.fused_silu_hin_rotate_mq_fp8_gfx12_batched(gate, w_down.awq_scale, k, n)?
    } else {
        gpu.fused_silu_mul_rotate_mq_fp8_gfx12_batched(gate, up, w_down.awq_scale, x_rot, k, n)?
    };
    Ok(Some(prep))
}

/// gfx1201 slices-3: FWHT-rotate + `block_i4_128` IU4 producer for the
/// attention out-proj input. `None` → caller keeps the incumbent rotate +
/// standalone-quantizer path. Same contract as [`try_iu4_rotate_prepared`]
/// (uniform MQ4G256V2 only, Residual-only) but gated by
/// `HIPFIRE_GFX12_PRODUCER_QUANT_FUSED` on exact gfx1201. The f32 `x_rot`
/// store is always written, so every downstream reader is preserved
/// byte-for-byte.
pub(crate) fn try_gfx12_rotate_quant_fused_prepared(
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

/// T-B: FWHT-rotate + `block_i4_128` IU4 producer for wo (residual) inputs.
/// The wo input is a gated_norm/sigmoid output (never an rmsnorm output),
/// so the C2 producers can't cover it; its only IU4 consumer is the residual
/// GEMM via standalone `quantize_int4_mmq_ds128`, which this removes.
/// `None` → caller keeps incumbent rotate + standalone-quantizer path.
/// Residual-only (w_down C2 precedent): Partial TP keeps the plain path.
/// The f32 `x_rot` store is always skipped: on the admitted path the
/// prepared IU4 GEMM is the only consumer, and every fallback writes the
/// buffer itself before reading it.
pub(crate) fn try_iu4_rotate_prepared(
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

/// Arithmetic policy of the dense batched projection / FFN helpers.
///
/// `Product` is every pre-existing route: the prepared A8/IU4/FP8 producers,
/// packed/fused quantized FFN, F1-lite and the in-place quantized FA gate and
/// output arms stay admitted exactly as before. `SingletonWmma` runs only the
/// F32 producers (AWQ + RMSNorm + FWHT, SwiGLU, sigmoid gate) and the explicit
/// exact verify GEMMs (`Gpu::gemm_*_mq4g256v2_verify_exact`), whose per-output
/// arithmetic is the singleton F16-input / f32-WMMA chain at any row count.
/// It is MQ4G256V2-only: any other weight dtype, a `Partial` epilogue or a
/// prepared/in-place producer request is a pre-mutation error, never a
/// fallback to the quantized product arithmetic.
///
/// `Iu4Verify` (DFlash IU4 verify: default on at `dflash_block` 8,
/// `HIPFIRE_CB_VERIFY_IU4`; numerics differ from F16) is the
/// DFlash verify inside `Gpu::iu4_verify_scope`, for the singleton verify
/// and the multi-request verify trunk at any combined row count alike: every
/// projection takes a gfx1201 A4 producer (AWQ RMSNorm/FWHT, gated-norm,
/// sigmoid, SwiGLU or F1-lite h, slab layout) and its IU4 GEMM (`_b1s` SET /
/// ADD, the F1-lite gate/up+SwiGLU), which the scope admits at any row count
/// and whose output rows do not depend on it. An F16 projection fallback
/// inside it is an error, never a silent mixed route. Stages reached with
/// `Product` inside the scope run as `Iu4Verify` ([`Self::resolve`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseBatchMath {
    Product,
    SingletonWmma,
    Iu4Verify,
}

impl DenseBatchMath {
    /// The math a stage runs: `Product` inside the IU4 verify scope is
    /// `Iu4Verify`. `SingletonWmma` inside the scope and `Iu4Verify` outside
    /// it are caller errors.
    pub(crate) fn resolve(self, gpu: &Gpu, site: &str) -> HipResult<Self> {
        match (self, gpu.iu4_verify_scope) {
            (Self::Product, true) | (Self::Iu4Verify, true) => Ok(Self::Iu4Verify),
            (Self::Product, false) => Ok(Self::Product),
            (Self::SingletonWmma, false) => Ok(Self::SingletonWmma),
            (Self::SingletonWmma, true) => Err(HipError::new(
                0,
                &format!("{site}: SingletonWmma verify math inside the IU4 verify scope"),
            )),
            (Self::Iu4Verify, false) => Err(HipError::new(
                0,
                &format!("{site}: Iu4Verify math outside the IU4 verify scope"),
            )),
        }
    }
}

/// `Iu4Verify`: the A4 producer for this projection must have run.
pub(crate) fn require_iu4_verify_prep(
    math: DenseBatchMath,
    site: &str,
    produced: bool,
) -> HipResult<()> {
    if math == DenseBatchMath::Iu4Verify && !produced {
        return Err(HipError::new(
            0,
            &format!(
                "{site}: IU4 verify math requires the gfx1201 A4 producer, which was not admitted"
            ),
        ));
    }
    Ok(())
}

pub(crate) fn require_singleton_wmma_mq4v2(site: &str, dtypes: &[DType]) -> HipResult<()> {
    if dtypes.iter().all(|d| *d == DType::MQ4G256V2) {
        Ok(())
    } else {
        Err(HipError::new(
            0,
            &format!(
                "{site}: SingletonWmma verify math requires dense MQ4G256V2 weights, got {dtypes:?}"
            ),
        ))
    }
}

/// Dense-batch arithmetic of a target-verify forward: [`DenseBatchMath::SingletonWmma`]
/// at every row count while the PM FP8 verify scope is open
/// (`HIPFIRE_CB_VERIFY_FP8`, see `Gpu::fp8_verify_enter`) so every dense stage
/// is an F32 producer + `*VerifyExact` key, which rdna routes to the FP8 verify
/// GEMMs (batched == singleton by construction, no F16 fusion fast route);
/// [`DenseBatchMath::Product`] (byte-identical to the pre-FP8 route) otherwise.
#[inline]
pub fn verify_dense_math(gpu: &Gpu) -> DenseBatchMath {
    if gpu.fp8_verify_scope {
        DenseBatchMath::SingletonWmma
    } else {
        DenseBatchMath::Product
    }
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
pub(crate) fn dispatch_batched_gemm_epilogue(
    gpu: &mut Gpu,
    pbs: &BatchScratch<'_>,
    w: &WeightRef<'_>,
    input: &GpuTensor,
    epilogue: &BatchEpilogue<'_>,
    n: usize,
    q8_wmma_arch: bool,
    _arch_has_wmma: bool,
    math: DenseBatchMath,
) -> HipResult<()> {
    let m = w.m;
    let k = w.k;
    let math = math.resolve(gpu, "dispatch_batched_gemm_epilogue")?;
    // Every IU4 verify residual GEMM consumes its A4 producer's prepared
    // activations; reaching this F32-input dispatch means none ran.
    require_iu4_verify_prep(math, "dispatch_batched_gemm_epilogue", false)?;
    if math == DenseBatchMath::SingletonWmma {
        require_singleton_wmma_mq4v2("dispatch_batched_gemm_epilogue", &[w.dtype])?;
        if !matches!(epilogue, BatchEpilogue::Residual) {
            return Err(HipError::new(
                0,
                "dispatch_batched_gemm_epilogue: SingletonWmma verify math requires the Residual epilogue",
            ));
        }
        return run_residual_gemm_key(
            gpu,
            crate::types::KernelKey::GemmMq4G256V2ResidualVerifyExact,
            &w.buf,
            w.dtype,
            input,
            &pbs.x_batch,
            m,
            k,
            n,
        );
    }
    let is_6bit = matches!(w.dtype, DType::MQ6G256 | DType::HFQ6G256);
    let is_mq3_lloyd = matches!(w.dtype, DType::MQ3G256Lloyd);
    let is_mq3 = matches!(w.dtype, DType::MQ3G256);
    let is_fp4 = matches!(w.dtype, DType::HFP4G32 | DType::MFP4G32);
    let is_q8 = matches!(w.dtype, DType::Q8_0);
    // GGUF-native (`DType::Ggml*`) weights take the same unfused plain-GEMM
    // strategy: `run_plain_gemm_key` dequantises the weight and runs the
    // engine's F16 / F32 GEMM on it.
    let is_lowbit = matches!(w.dtype, DType::TQ2G128 | DType::BQ1G128) || w.dtype.is_ggml();
    match epilogue {
        BatchEpilogue::Residual => {
            if is_6bit {
                return run_residual_gemm_key(
                    gpu,
                    crate::types::KernelKey::GemmHfq6G256Residual,
                    &w.buf,
                    w.dtype,
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
                    crate::types::KernelKey::GemmQ8_0ResidualWmma,
                    &w.buf,
                    w.dtype,
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
                    plain_gemm_key_for(w.dtype),
                    &w.buf,
                    w.dtype,
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
                    crate::types::KernelKey::GemmMq3G256LloydResidual,
                    &w.buf,
                    w.dtype,
                    input,
                    &pbs.x_batch,
                    m,
                    k,
                    n,
                );
            } else if is_mq3 {
                return run_residual_gemm_key(
                    gpu,
                    crate::types::KernelKey::GemmHfq3G256Residual,
                    &w.buf,
                    w.dtype,
                    input,
                    &pbs.x_batch,
                    m,
                    k,
                    n,
                );
            } else if is_fp4 {
                return run_residual_gemm_key(
                    gpu,
                    crate::types::KernelKey::GemmHfp4G32Residual,
                    &w.buf,
                    w.dtype,
                    input,
                    &pbs.x_batch,
                    m,
                    k,
                    n,
                );
            } else if matches!(w.dtype, DType::MQ4G256V2Lloyd) {
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
                    crate::families::gemm::residual_gemm_key_for(w.dtype),
                    &w.buf,
                    w.dtype,
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
                    crate::types::KernelKey::GemmHfq6G256Residual,
                    &w.buf,
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
                    &w.buf,
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
                    &w.buf,
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
                    &w.buf,
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
                    &w.buf,
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
                    &w.buf,
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
                    crate::families::gemm::residual_gemm_key_for(w.dtype),
                    &w.buf,
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
pub(crate) fn dispatch_batched_fp8_lloyd_epilogue(
    gpu: &mut Gpu,
    pbs: &BatchScratch<'_>,
    w: &WeightRef<'_>,
    prepared: &rdna_compute::Mq4v2Fp8Prepared,
    epilogue: &BatchEpilogue<'_>,
    n: usize,
) -> HipResult<()> {
    let out = match epilogue {
        BatchEpilogue::Residual => &pbs.x_batch,
        BatchEpilogue::Partial(out) => {
            zero_partial_for_residual(gpu, out, n, w.m)?;
            out
        }
    };
    match w.dtype {
        DType::MQ4G256V2 => gpu.gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8_prepared(
            &w.buf, prepared, out, w.m, w.k, n,
        ),
        DType::MQ4G256V2Lloyd => gpu.gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8_prepared_lloyd(
            &w.buf,
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
        // Float weights kept F32 (the GDN beta/alpha matrices of a GGUF file).
        DType::F32 => K::GemmF32RegisterTiled,
        _ => K::GemmQ8_0BatchedChunked,
    }
}

fn q8_prefill_wmma_enabled_from_env(value: Option<&str>, _arch: &str, has_wmma: bool) -> bool {
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
    // GGUF-native weight: dequantise into the persistent prefill scratch (F16,
    // or BF16-rounded F32 for the codec-only tier) and run the engine's GEMM
    // on it; `HIPFIRE_GGUF_A8_EMU=1` first rounds the activations through q8_1.
    if w_dtype.is_ggml() {
        return gpu.ggml_prefill_gemm(w_dtype, w_buf, m, k, x, y, n);
    }
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
    GEMM.run_key(key, &ctx, gpu, &params).map_err(hip_err)
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
    GEMM.run_key(key, &ctx, gpu, &params).map_err(hip_err)
}

/// qt=52 (MQ4G256V2Lloyd) prefill re-arm: extract the per-tensor E4M3 codebook
/// LUT or fail closed naming the prefill family. The uniform route never
/// touches this (uniform tensors carry no sidecar), so it stays bit-identical.
#[inline]
pub(crate) fn lloyd_e4m3_or_fail(w: &WeightRef<'_>, family: &'static str) -> HipResult<[u32; 4]> {
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
pub(crate) fn lloyd_c16_or_fail(w: &WeightRef<'_>, family: &'static str) -> HipResult<[u32; 4]> {
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
pub(crate) fn lloyd_mmq_lut_route(gpu: &Gpu) -> bool {
    matches!(gpu.arch.as_str(), "gfx1100" | "gfx1151")
        && hipfire_config::developer_var("HIPFIRE_LLOYD_MMQ_OFF")
            .ok()
            .as_deref()
            != Some("1")
}

/// True when all listed projection dtypes are the Lloyd-V2 dtype.
#[inline]
pub(crate) fn all_mq4v2_lloyd(dts: &[DType]) -> bool {
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
    FUSED_QKV.run(&ctx, gpu, &params).map_err(hip_err)
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
    FUSED_QKV.run(&ctx, gpu, &params).map_err(hip_err)
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
pub fn run_fused_qkvza_key(
    gpu: &mut Gpu,
    key: crate::types::KernelKey,
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
    use crate::families::fused_qkv::FusedQkvParams;
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
    FUSED_QKV.run(&ctx, gpu, &params).map_err(hip_err)
}

/// S3-f16-projection-inputs: exact-route gate for the FP16 projection-input
/// fast path (all four `batch_chunk_*` projection hooks below).
///
/// All ordinary-path predicates are cheap field reads. Capture/recording is
/// admitted only together with the exact-gfx1100 split-verifier experiment:
/// that route owns fixed, preallocated F16 sidecars and records every producer
/// and consumer through the blob funnel. Every failed predicate runs the
/// pre-change F32-producer + `convert_f32_to_f16` path byte-for-byte.
#[inline]
pub fn mq_f16_projection_fast_route(
    gpu: &Gpu,
    fusion: DflashFusionCtx,
    n: usize,
    dim: usize,
) -> bool {
    let recording = gpu.graphs.capture_mode || gpu.replay.is_recording();
    let recording_supported = !recording
        || (fusion.split_verify_active() && gpu.gfx1100_q8_fa2_split_capture_assets_ready(n));
    fusion.is_chain_verify()
        // The PM FP8 verify scope never takes an F16-input projection route.
        && !gpu.fp8_verify_scope
        && gpu.arch_caps.supports_dflash_f16_projection_fusions()
        && !gpu.flags.mq_f16_projection_off
        && n >= 1
        && n <= 16
        && dim % 256 == 0
        && recording_supported
        // IU4 verify (gfx1201): every projection takes the A4 producer + IU4
        // GEMM at every row count; an F16 projection there is a mixed route.
        && !gpu.iu4_verify_scope
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
    fusion: DflashFusionCtx,
    w_dtype: DType,
    epilogue: &BatchEpilogue<'_>,
    n: usize,
) -> bool {
    fusion.is_chain_verify()
        // The PM FP8 verify scope keeps the F32 producer + FP8 residual GEMM
        // (batched == singleton by construction; no F16-input residual GEMM).
        && !gpu.fp8_verify_scope
        && !gpu.flags.mq_f16_residual_off
        && gpu.arch_caps.supports_dflash_f16_residual_fusions()
        && w_dtype == DType::MQ4G256V2
        && matches!(epilogue, BatchEpilogue::Residual)
        && (1..=16).contains(&n)
        // IU4 verify: the out-projections take the A4 producer + IU4 GEMM
        // at every row count, never this 1..=16-row F16 arm.
        && !gpu.iu4_verify_scope
}

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
    layer: &FfnLayerView<'_>,
    norm_eps: f32,
    pbs: &BatchScratch<'_>,
    n: usize,
    dim: usize,
    q8_wmma_arch: bool,
    fusion: DflashFusionCtx,
    f1lite: bool,
    math: DenseBatchMath,
) -> HipResult<FfnGateOutput> {
    let _ = fusion;
    let math = math.resolve(gpu, "swiglu_ffn_gate_up_batched")?;
    if math == DenseBatchMath::SingletonWmma {
        require_singleton_wmma_mq4v2(
            "swiglu_ffn_gate_up_batched",
            &[layer.w_gate.dtype, layer.w_up.dtype],
        )?;
        gpu.flush_residual_fold()?;
        fused_rmsnorm_rotate_mq_batched_for(
            gpu,
            &pbs.x_batch,
            &layer.ffn_norm,
            &layer.w_gate,
            &pbs.x_rot_batch,
            dim,
            norm_eps,
            n,
        )?;
        run_fused_gate_up_key(
            gpu,
            crate::types::KernelKey::FusedGateUpMq4G256V2VerifyExact,
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
        return Ok(FfnGateOutput::Separate);
    }
    if try_packed_mq4_ffn_gate_up(
        gpu,
        &layer.w_gate,
        &layer.w_up,
        &layer.ffn_norm,
        norm_eps,
        pbs,
        n,
    )? {
        return Ok(FfnGateOutput::Separate);
    }
    // S3-f16-projection-inputs fast path: exact-FP16 FFN gate/up inputs.
    // gate/up share the pre-rotation input, so both must be MQ4G256V2.
    if mq_f16_projection_fast_route(gpu, fusion, n, dim)
        && layer.w_gate.dtype == DType::MQ4G256V2
        && layer.w_up.dtype == DType::MQ4G256V2
    {
        gpu.flush_residual_fold()?;
        fused_rmsnorm_rotate_mq_f16_batched_for(
            gpu,
            &pbs.x_batch,
            &layer.ffn_norm,
            &layer.w_gate,
            &pbs.x_rot_f16_batch,
            dim,
            norm_eps,
            n,
        )?;
        return gpu
            .gemm_gate_up_mq4g256v2_wmma_f16(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_f16_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
                fusion.split_verify_active() && gpu.gfx1100_q8_fa2_split_capture_assets_ready(n),
            )
            .map(|()| FfnGateOutput::Separate);
    }
    // FFN: rmsnorm (+ rotate for MQ).
    let ffn_is_mq = matches!(
        layer.w_gate.dtype,
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
    let ffn_is_6bit = matches!(layer.w_gate.dtype, DType::MQ6G256 | DType::HFQ6G256);
    let ffn_is_mq3 = matches!(layer.w_gate.dtype, DType::MQ3G256);
    let ffn_is_mq3_lloyd = matches!(layer.w_gate.dtype, DType::MQ3G256Lloyd);
    let ffn_is_fp4 = matches!(layer.w_gate.dtype, DType::HFP4G32 | DType::MFP4G32);
    let ffn_is_q8 = matches!(layer.w_gate.dtype, DType::Q8_0);
    // TQ2G128/BQ1G128 have no fused qkvza/gate_up/qkv kernel, so they take
    // the same UNFUSED plain-GEMM strategy as Q8 rather than falling through
    // to the HFQ4 arm, which would read these packed blocks at the wrong
    // stride and produce fluent-but-wrong tokens.
    let ffn_is_lowbit = matches!(layer.w_gate.dtype, DType::TQ2G128 | DType::BQ1G128)
        || layer.w_gate.dtype.is_ggml();
    // qt=52 re-arm anchor: gate+up both Lloyd → FP8-LUT launcher.
    let ffn_is_mq4v2_lloyd = all_mq4v2_lloyd(&[layer.w_gate.dtype, layer.w_up.dtype]);
    let a8_prep = if layer.w_gate.dtype == DType::MQ4G256V2 && layer.w_up.dtype == DType::MQ4G256V2
    {
        try_a8_rmsnorm_prepared(
            gpu,
            &pbs.x_batch,
            &layer.ffn_norm,
            &layer.w_gate,
            dim,
            norm_eps,
            n,
        )?
    } else {
        None
    };
    let mut iu4_prep: Option<rdna_compute::Int4MmqPrepared> = None;
    let mut fp8_prep: Option<rdna_compute::Mq4v2Fp8Prepared> = None;
    if ffn_is_mq && a8_prep.is_none() {
        // C2: MQ4V2 IU4 producer (emit_f32=false — gate_up has no f32 small tails).
        iu4_prep = try_iu4_rmsnorm_prepared(
            gpu,
            &pbs.x_batch,
            &layer.ffn_norm,
            &layer.w_gate,
            &pbs.x_rot_batch,
            dim,
            norm_eps,
            n,
            false,
        )?;
        if iu4_prep.is_none()
            && mq3v25_a4_consumers_agree(gpu, layer.w_gate.dtype, &[layer.w_up.dtype])
        {
            // gfx1201 slices-2: fused rmsnorm+quant producer (bit-identical).
            iu4_prep = try_gfx12_rmsnorm_quant_fused_prepared(
                gpu,
                &pbs.x_batch,
                &layer.ffn_norm,
                &layer.w_gate,
                &pbs.x_rot_batch,
                dim,
                norm_eps,
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
                &layer.ffn_norm,
                &layer.w_gate,
                Fp8StreamConsumer::GateUp,
                &[layer.w_gate.dtype, layer.w_up.dtype],
                &pbs.x_rot_batch,
                dim,
                norm_eps,
                n,
            )?;
        }
        if iu4_prep.is_none() && fp8_prep.is_none() {
            // AWQ-aware: next linear is w_gate (gate/up share input → same AWQ scale).
            fused_rmsnorm_rotate_mq_batched_for(
                gpu,
                &pbs.x_batch,
                &layer.ffn_norm,
                &layer.w_gate,
                &pbs.x_rot_batch,
                dim,
                norm_eps,
                n,
            )?;
        }
    } else if !ffn_is_mq {
        gpu.rmsnorm_batched(
            &pbs.x_batch,
            &layer.ffn_norm,
            &pbs.x_rot_batch,
            n,
            dim,
            norm_eps,
        )?;
    }

    require_iu4_verify_prep(math, "swiglu_ffn_gate_up_batched", iu4_prep.is_some())?;
    // Batched gate+up projection.
    // #397 Ship 5.2 slice 2: fused gate+up dtypes → FusedQkvFamily
    // (batched-prefill gate+up variant) via run_fused_gate_up_key.
    // The Q8-non-WMMA case stays as two plain GemmQ8_0BatchedChunked
    // GEMMs (not a fused kernel — slice 1). The HFQ3 WMMA-vs-base
    // split is folded into the FusedGateUpHfq3G256 run-arm, which
    // re-derives it from gpu.arch_caps.has_wmma() (== arch_has_wmma).
    if let Some(prep) = &a8_prep {
        let xq = gpu.int8_mmq_prepared_ptr(prep, layer.w_gate.k, n)?;
        if f1lite {
            gpu.gemm_gate_up_silu_mq4g256v2_i8_prequant(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                xq,
                &pbs.gate_ffn_batch,
                layer.w_gate.m,
                layer.w_gate.k,
                n,
            )?;
            return Ok(FfnGateOutput::A8H);
        }
        gpu.gemm_mq4g256v2_mmq_set_prequant_i8(
            &layer.w_gate.buf,
            xq,
            &pbs.gate_ffn_batch,
            layer.w_gate.m,
            layer.w_gate.k,
            n,
        )?;
        gpu.gemm_mq4g256v2_mmq_set_prequant_i8(
            &layer.w_up.buf,
            xq,
            &pbs.up_batch,
            layer.w_up.m,
            layer.w_up.k,
            n,
        )?;
    } else if f1lite && iu4_prep.is_none() && gpu.a8_prefill_active(n, layer.w_gate.k) {
        let xq = gpu.ensure_int8_mmq_x(&pbs.x_rot_batch, n, layer.w_gate.k)?;
        gpu.gemm_gate_up_silu_mq4g256v2_i8_prequant(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            xq,
            &pbs.gate_ffn_batch,
            layer.w_gate.m,
            layer.w_gate.k,
            n,
        )?;
        return Ok(FfnGateOutput::A8H);
    } else if let Some(prep) = iu4_prep.as_ref().filter(|_| {
        mq3v25_a4_weight(gpu, layer.w_gate.dtype) && layer.w_up.dtype == layer.w_gate.dtype
    }) {
        // Symmetric qt49 A4 (gfx1201): F1-lite single launch emits h when
        // admitted (the down hook then runs the AWQ h producer exactly as
        // for MQ4); else the SET pair. No MQ4-only epilogue (A4 down
        // sidecar) applies to qt49.
        if f1lite
            && gpu.gemm_gate_up_silu_mq3v25_iu4_prepared(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                prep,
                &pbs.gate_ffn_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?
        {
            return Ok(FfnGateOutput::Iu4H);
        }
        gpu.gemm_gate_up_mq3v25_iu4_prepared(
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
    } else if let Some(prep) = iu4_prep
        .as_ref()
        .filter(|_| !mq3v25_a4_weight(gpu, layer.w_gate.dtype))
    {
        // F1-lite: one GEMM v2 launch emits h; else the SET pair below.
        // A4 fusion (gfx1151 V2B only, default on; `HIPFIRE_V2B_A4_EPI=0` opts out): the epilogue
        // also writes the w_down sidecar, so no h and no hin producer.
        if f1lite {
            if let Some(awq) = layer.w_down.awq_scale {
                if let Some(down_prep) = gpu.gemm_gate_up_silu_a4_mq4g256v2_iu4_prepared(
                    &layer.w_gate.buf,
                    &layer.w_up.buf,
                    prep,
                    awq,
                    layer.w_gate.m,
                    layer.w_up.m,
                    layer.w_gate.k,
                    n,
                )? {
                    return Ok(FfnGateOutput::Iu4A4(down_prep));
                }
            }
        }
        if f1lite
            && gpu.gemm_gate_up_silu_mq4g256v2_iu4_prepared(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                prep,
                &pbs.gate_ffn_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?
        {
            return Ok(FfnGateOutput::Iu4H);
        }
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
            crate::types::KernelKey::FusedGateUpHfq6G256,
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
            matches!(layer.w_up.dtype, DType::Q8_0),
            "{} FFN Q8 WMMA dispatch requires both w_gate and w_up to be Q8_0",
            layer.kind,
        );
        run_fused_gate_up_key(
            gpu,
            crate::types::KernelKey::FusedGateUpQ8_0,
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
            plain_gemm_key_for(layer.w_gate.dtype),
            &layer.w_gate.buf,
            layer.w_gate.dtype,
            &pbs.x_rot_batch,
            &pbs.gate_ffn_batch,
            layer.w_gate.m,
            layer.w_gate.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.w_up.dtype),
            &layer.w_up.buf,
            layer.w_up.dtype,
            &pbs.x_rot_batch,
            &pbs.up_batch,
            layer.w_up.m,
            layer.w_up.k,
            n,
        )?;
    } else if ffn_is_mq3_lloyd {
        run_fused_gate_up_key(
            gpu,
            crate::types::KernelKey::FusedGateUpMq3G256Lloyd,
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
            crate::types::KernelKey::FusedGateUpHfq3G256,
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
            crate::types::KernelKey::FusedGateUpHfp4G32,
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
                lloyd_c16_or_fail(&layer.w_gate, "swiglu_ffn_gate_up_batched")?,
                lloyd_c16_or_fail(&layer.w_up, "swiglu_ffn_gate_up_batched")?,
            )?;
        } else if let Some(prep) = &fp8_prep {
            // gfx1201 FP8-stream: the producer already emitted the fp8
            // pre-pass planes; consume them directly, no pack launch.
            gpu.gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2_fp8_bt12_prepared_lloyd(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                prep,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
                lloyd_e4m3_or_fail(&layer.w_gate, "swiglu_ffn_gate_up_batched")?,
                lloyd_e4m3_or_fail(&layer.w_up, "swiglu_ffn_gate_up_batched")?,
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
                lloyd_e4m3_or_fail(&layer.w_gate, "swiglu_ffn_gate_up_batched")?,
                lloyd_e4m3_or_fail(&layer.w_up, "swiglu_ffn_gate_up_batched")?,
            )?;
        }
    } else if matches!(layer.w_gate.dtype, DType::MQ4G256V2Lloyd)
        || matches!(layer.w_up.dtype, DType::MQ4G256V2Lloyd)
    {
        // Mixed Lloyd/uniform gate/up: no kernel reads mixed codebooks.
        return Err(hip_bridge::HipError::new(
            0,
            "swiglu_ffn_gate_up_batched: mixed MQ4G256V2Lloyd/uniform gate/up — refusing (quantize both or neither)",
        ));
    } else if layer.w_gate.dtype == DType::MQ4G256V2
        && layer.w_up.dtype == DType::MQ4G256V2
        // Producer admission already required the gate/up family to take
        // FP8 (`gfx12_fp8_stream_admitted`); fp8_prep must be consumed here.
        && fp8_prep.is_some()
    {
        // gfx1201 FP8-stream (uniform): the producer already emitted the
        // fp8 pre-pass planes; consume them directly with the launch twin
        // of the family's fp8 route — no pack launch. Same fp8 intercept
        // conditions as the uniform router (iu4 divergence excluded by
        // producer-side ordering: fp8_prep implies iu4_prep is None).
        let silu_h = gpu.fp8_silu_h_active(n, layer.w_gate.k, layer.w_gate.m, layer.w_up.m)
            && layer.w_down.dtype == DType::MQ4G256V2
            && gfx12_fp8_stream_admitted(
                gpu,
                Fp8StreamConsumer::Residual,
                &[layer.w_down.dtype],
                n,
                layer.w_down.k,
            );
        gpu.gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2_fp8_bt12_prepared(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            fp8_prep.as_ref().unwrap(),
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
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
            crate::families::fused_qkv::fused_gate_up_key_for(layer.w_gate.dtype),
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
    Ok(FfnGateOutput::Separate)
}

#[allow(clippy::too_many_arguments)]
/// Prescaffold (behavior-only) extraction for S4-f16-residual-inputs.
///
/// Same statements, same order, same launches as the inlined block.
/// S9-mq4v2-persistent-prologues will issue `try_mq4v2_persistent_prologue`
/// from inside this hook after S3/S4 land.
///
/// `h_source` tells whether gate/up emitted separate planes, IU4 h, FP8 h, or
/// (`Iu4A4`) the finished w_down A4 sidecar with no h at all; h is stored in
/// `gate_ffn_batch` and needs its matching down producer.
fn swiglu_ffn_down_batched(
    gpu: &mut Gpu,
    layer: &FfnLayerView<'_>,
    pbs: &BatchScratch<'_>,
    hidden_dim: usize,
    n: usize,
    q8_wmma_arch: bool,
    arch_has_wmma: bool,
    epilogue: BatchEpilogue<'_>,
    fusion: DflashFusionCtx,
    h_source: FfnGateOutput,
    math: DenseBatchMath,
) -> HipResult<()> {
    let math = math.resolve(gpu, "swiglu_ffn_down_batched")?;
    if math == DenseBatchMath::SingletonWmma {
        require_singleton_wmma_mq4v2("swiglu_ffn_down_batched", &[layer.w_down.dtype])?;
        if h_source != FfnGateOutput::Separate
            || !matches!(&epilogue, BatchEpilogue::Residual)
            || layer.w_down.k != hidden_dim
        {
            return Err(HipError::new(
                0,
                "swiglu_ffn_down_batched: SingletonWmma verify math requires separate gate/up planes, the Residual epilogue and w_down.k == hidden_dim",
            ));
        }
        fused_silu_mul_rotate_mq_batched_for(
            gpu,
            &layer.w_down,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            &pbs.ffn_hidden_batch,
            hidden_dim,
            n,
        )?;
        return dispatch_batched_gemm_epilogue(
            gpu,
            pbs,
            &layer.w_down,
            &pbs.ffn_hidden_batch,
            &epilogue,
            n,
            q8_wmma_arch,
            arch_has_wmma,
            math,
        );
    }
    if try_packed_mq4_down(gpu, &layer.w_down, pbs, hidden_dim, n, &epilogue, h_source)? {
        return Ok(());
    }
    // S4: one silu*up+FWHT+F16 producer + direct-F16 residual GEMM instead
    // of fused_silu_mul_rotate_mq_batched + convert.
    if h_source == FfnGateOutput::Separate
        && s4_residual_fast(gpu, fusion, layer.w_down.dtype, &epilogue, n)
    {
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
        layer.w_down.dtype,
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
        && layer.w_down.dtype == DType::MQ4G256V2
        && matches!(&epilogue, BatchEpilogue::Residual)
    {
        let xq = if let Some(prep) =
            try_a8_hin_prepared(gpu, &layer.w_down, &pbs.gate_ffn_batch, hidden_dim, n)?
        {
            gpu.int8_mmq_prepared_ptr(&prep, hidden_dim, n)?
        } else {
            rotate_x_mq_batched_for(
                gpu,
                &layer.w_down,
                &pbs.gate_ffn_batch,
                &pbs.ffn_hidden_batch,
                hidden_dim,
                n,
            )?;
            gpu.ensure_int8_mmq_x(&pbs.ffn_hidden_batch, n, hidden_dim)?
        };
        return gpu.gemm_mq4g256v2_mmq_add_prequant_i8(
            &layer.w_down.buf,
            xq,
            &pbs.x_batch,
            layer.w_down.m,
            hidden_dim,
            n,
        );
    }
    if let FfnGateOutput::Iu4A4(prep) = h_source {
        gpu.gemm_mq4g256v2_residual_wmma_iu4_down_prepared(
            &layer.w_down.buf,
            &prep,
            &pbs.x_batch,
            layer.w_down.m,
            layer.w_down.k,
            n,
        )?;
        return Ok(());
    }
    let mut iu4_prep: Option<rdna_compute::Int4MmqPrepared> = None;
    let mut fp8_prep: Option<rdna_compute::Mq4v2Fp8Prepared> = None;
    if w_down_is_mq {
        // C2: SwiGLU/FWHT IU4 producer for w_down (emit_f32=false). Residual only —
        // Partial TP epilogue still needs the f32 rotated buffer.
        if matches!(&epilogue, BatchEpilogue::Residual) && h_source != FfnGateOutput::Fp8H {
            iu4_prep = if h_source == FfnGateOutput::Iu4H {
                Some(iu4_hin_prepared(
                    gpu,
                    &layer.w_down,
                    &pbs.gate_ffn_batch,
                    hidden_dim,
                    n,
                )?)
            } else {
                try_iu4_silu_prepared(
                    gpu,
                    &layer.w_down,
                    &pbs.gate_ffn_batch,
                    &pbs.up_batch,
                    &pbs.ffn_hidden_batch,
                    hidden_dim,
                    n,
                    false,
                )?
            };
            if iu4_prep.is_none() {
                // gfx1201 slice-1: fused silu+quant producer (bit-identical).
                iu4_prep = try_gfx12_silu_quant_fused_prepared(
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
        }
        if iu4_prep.is_none() {
            fp8_prep = try_gfx12_fp8_stream_silu_prepared(
                gpu,
                &layer.w_down,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                &pbs.ffn_hidden_batch,
                hidden_dim,
                n,
                h_source == FfnGateOutput::Fp8H,
            )?;
        }
        if iu4_prep.is_none() && fp8_prep.is_none() {
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
    if let (Some(prep), true) = (&iu4_prep, mq3v25_a4_weight(gpu, layer.w_down.dtype)) {
        gpu.gemm_mq3v25_residual_iu4_prepared(
            &layer.w_down.buf,
            prep,
            &pbs.x_batch,
            layer.w_down.m,
            layer.w_down.k,
            n,
            true,
        )?;
    } else if let Some(prep) = &iu4_prep {
        gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(
            &layer.w_down.buf,
            prep,
            &pbs.x_batch,
            layer.w_down.m,
            layer.w_down.k,
            n,
        )?;
    } else if let Some(prep) = &fp8_prep {
        dispatch_batched_fp8_lloyd_epilogue(gpu, pbs, &layer.w_down, prep, &epilogue, n)?;
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
            math,
        )?;
    }
    Ok(())
}

pub(crate) fn execute_swiglu_ffn_batched(
    gpu: &mut Gpu,
    layer: &FfnLayerView<'_>,
    norm_eps: f32,
    pbs: &BatchScratch<'_>,
    n: usize,
    dim: usize,
    hidden_dim: usize,
    q8_wmma_arch: bool,
    arch_has_wmma: bool,
    epilogue: BatchEpilogue<'_>,
    fusion: DflashFusionCtx,
    math: DenseBatchMath,
) -> HipResult<()> {
    let f1lite = f1lite_ffn_eligible(
        gpu,
        &layer.w_gate,
        &layer.w_up,
        &layer.w_down,
        &epilogue,
        fusion,
        hidden_dim,
        n,
    );
    let h_source = swiglu_ffn_gate_up_batched(
        gpu,
        layer,
        norm_eps,
        pbs,
        n,
        dim,
        q8_wmma_arch,
        fusion,
        f1lite,
        math,
    )?;

    swiglu_ffn_down_batched(
        gpu,
        layer,
        pbs,
        hidden_dim,
        n,
        q8_wmma_arch,
        arch_has_wmma,
        epilogue,
        fusion,
        h_source,
        math,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ECSQ-5 §6 regression: `HIPFIRE_IU4_PREFILL=0` plus
    /// `HIPFIRE_GFX12_MQ4V2_FP8_{GATEUP,RESID,QKVZA,QKV}=0` gave WT2 KLD 11.74
    /// because the FP8-stream producers (gated only on
    /// `HIPFIRE_GFX12_FP8_STREAM`) still emitted FP8-only planes for consumers
    /// that had moved to F16 and read the unwritten `x_rot`. Every subset of
    /// family opt-outs must disable exactly its own producers.
    #[test]
    fn fp8_stream_producer_follows_consumer_family_opt_out() {
        use Fp8StreamConsumer::*;
        const U: DType = DType::MQ4G256V2;
        const L: DType = DType::MQ4G256V2Lloyd;
        let families: [(Fp8StreamConsumer, &[DType]); 4] = [
            (Qkvza, &[U, U, U, U]),
            (GateUp, &[U, U]),
            (Qkv, &[U, U, U]),
            (Residual, &[U]),
        ];
        for mask in 0u8..16 {
            for v2 in [true, false] {
                let mut f = rdna_compute::FeatureFlags::for_test("gfx1201");
                f.gfx12_fp8_stream = true;
                f.gfx12_mq4v2_fp8_v2 = v2;
                f.gfx12_mq4v2_fp8_qkvza = mask & 1 != 0;
                f.gfx12_mq4v2_fp8_gateup = mask & 2 != 0;
                f.gfx12_mq4v2_fp8_qkv = mask & 4 != 0;
                f.gfx12_mq4v2_fp8_resid = mask & 8 != 0;
                for (consumer, dtypes) in families {
                    let consumer_fp8 = match consumer {
                        Qkvza => f.gfx12_mq4v2_fp8_qkvza,
                        GateUp => f.gfx12_mq4v2_fp8_gateup,
                        Qkv => f.gfx12_mq4v2_fp8_qkv,
                        Residual => f.gfx12_mq4v2_fp8_resid,
                    };
                    assert_eq!(
                        fp8_stream_consumer_takes_fp8(&f, consumer, dtypes),
                        consumer_fp8,
                        "mask={mask:04b} v2={v2} consumer={consumer:?}"
                    );
                    // Lloyd consumers are FP8-only: the family flags never
                    // move them to F16, so the producer stays admitted.
                    let lloyd = vec![L; dtypes.len()];
                    assert!(fp8_stream_consumer_takes_fp8(&f, consumer, &lloyd));
                }
            }
        }
        // Default gfx1201 route (all families on) is unchanged.
        let mut on = rdna_compute::FeatureFlags::for_test("gfx1201");
        on.gfx12_mq4v2_fp8_qkvza = true;
        on.gfx12_mq4v2_fp8_gateup = true;
        on.gfx12_mq4v2_fp8_qkv = true;
        on.gfx12_mq4v2_fp8_resid = true;
        for (consumer, dtypes) in families {
            assert!(
                fp8_stream_consumer_takes_fp8(&on, consumer, dtypes),
                "{consumer:?}"
            );
        }
        // gate/up under the LDS-stage WMMA has no FP8 prepared consumer.
        on.hfq4g256_ldsstage_wmma = true;
        assert!(!fp8_stream_consumer_takes_fp8(&on, GateUp, &[U, U]));
        assert!(fp8_stream_consumer_takes_fp8(&on, Residual, &[U]));
        // Mixed, non-MQ4v2, or empty weight sets never admit a producer.
        assert!(!fp8_stream_consumer_takes_fp8(
            &on,
            Qkvza,
            &[U, U, U, DType::Q8_0]
        ));
        assert!(!fp8_stream_consumer_takes_fp8(&on, GateUp, &[U, L]));
        assert!(!fp8_stream_consumer_takes_fp8(
            &on,
            Residual,
            &[DType::MQ4G256]
        ));
        assert!(!fp8_stream_consumer_takes_fp8(&on, Residual, &[]));
    }

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
}
