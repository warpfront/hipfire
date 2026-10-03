// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Hybrid linear/full-attention decoder layer operations.
//!
//! A hybrid decoder interleaves gated-DeltaNet linear attention with gated
//! full attention, each followed by a dense SwiGLU or MoE FFN. Architecture
//! crates declare one operation per sublayer and bind their resident weights,
//! recurrent state and fixed scratch; every arch-, shape- and dtype-gated route
//! choice lives here. Executors are row-parameterised; decode binds `rows == 1`.
//!
//! The executor bodies are the former Qwen3.5 decode super-op handlers moved
//! verbatim, so every route issues the same launches in the same order.

use crate::context::DispatchCtx;
use crate::families::attention::AttnParams;
use crate::families::gemv::{GemvFamily, GemvParams, WeightRef};
use crate::families::kv_tier::{KvTierInputs, KvTierPlan};
use crate::pipeline::steps::{execute_steps, GemvInput, Step};
use crate::types::{
    dtype_needs_rotation, dtype_rotation_plan, DispatchError, GemvVariant, KernelKey, RotationPlan,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::LazyLock;

fn hip(e: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(e.to_string())
}

fn developer_var(name: &str) -> Option<String> {
    hipfire_config::developer_var(name).ok()
}

/// Model-level geometry of a hybrid decoder. Route selection reads only these
/// values, never a model name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridDims {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Rotated channels per head (partial rotary).
    pub n_rot: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub linear_key_heads: usize,
    pub linear_value_heads: usize,
    pub linear_key_dim: usize,
    pub linear_value_dim: usize,
    /// DeltaNet causal-conv kernel width.
    pub conv_kernel: usize,
    pub num_experts: usize,
}

impl HybridDims {
    fn key_width(&self) -> usize {
        self.linear_key_heads * self.linear_key_dim
    }
    fn value_width(&self) -> usize {
        self.linear_value_heads * self.linear_value_dim
    }
    /// The certified 5120-wide, 64-layer dense shape (Qwen3.6-27B class).
    pub fn is_dense_27b_shape(&self) -> bool {
        self.dim == 5_120
            && self.n_layers == 64
            && self.n_heads == 24
            && self.n_kv_heads == 4
            && self.head_dim == 256
            && self.linear_key_heads == 16
            && self.linear_key_dim == 128
            && self.linear_value_heads == 48
            && self.linear_value_dim == 128
            && self.num_experts == 0
    }
    /// The certified 2048-wide A3B shape the gfx1201 state fusions admit.
    pub fn is_a3b_state_fusion_shape(&self) -> bool {
        self.dim == 2_048
            && self.n_heads == 16
            && self.n_kv_heads == 2
            && self.head_dim == 256
            && self.linear_key_heads == 16
            && self.linear_value_heads == 32
            && self.linear_key_dim == 128
            && self.linear_value_dim == 128
    }
}

/// Recurrent-state storage for one gated DeltaNet layer.
#[derive(Clone, Copy)]
pub enum GdnState<'a> {
    F32 {
        state: &'a GpuTensor,
    },
    Q8 {
        state: &'a GpuTensor,
        scales: &'a GpuTensor,
        error_feedback: Option<&'a GpuTensor>,
    },
    Q4 {
        state: &'a GpuTensor,
        scales: &'a GpuTensor,
    },
}

impl GdnState<'_> {
    fn is_q8(&self) -> bool {
        matches!(self, Self::Q8 { .. })
    }
}

// ── Route predicates (arch-, shape- and dtype-gated fusions) ──────────────

/// Exact gfx1151 admission for its certified Radiowave decode bundle. Kept
/// separate from broad RDNA3 capability checks so no neighboring architecture
/// inherits the gfx1151 schedules.
pub fn gfx1151_radiowave(gpu: &Gpu) -> bool {
    gpu.arch_caps.is_gfx1151()
}

/// Exact gfx1201 admission for the ported decode state fusions.
fn gfx1201_state_fusions(gpu: &Gpu) -> bool {
    gpu.arch_caps.is_gfx1201()
}

fn is_mq4(dtype: DType) -> bool {
    matches!(dtype, DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ4CG256)
}

/// Compact DeltaNet QK: each pair (or triple) of value/state heads reuses one
/// Q/K head instead of materialising repeated Q/K. `Some(ratio)` when admitted.
pub fn gdn_compact_qk_div(gpu: &Gpu, dims: &HybridDims, q8_state: bool) -> Option<usize> {
    static COMPACT3: LazyLock<bool> =
        LazyLock::new(|| developer_var("HIPFIRE_GDN_COMPACT3").as_deref() != Some("0"));
    let compact2 =
        (gpu.arch_caps.is_gfx1201() || gpu.arch_caps.arch() == "gfx1100" || gfx1151_radiowave(gpu))
            && developer_var("HIPFIRE_GDN_COMPACT2").as_deref() != Some("0")
            && q8_state
            && dims.linear_key_heads * 2 == dims.linear_value_heads;
    if compact2 {
        return Some(2);
    }
    // 3:1 route for the dense 27B shape: +0.45% at 512 tokens, 853 -> 805
    // dispatches/token on gfx1100; exact gfx1201 runs the same object for the
    // same shape. `HIPFIRE_GDN_COMPACT3=0` restores explicit Q/K.
    let compact3 = *COMPACT3
        && (gpu.arch_caps.is_gfx1100() || gfx1201_state_fusions(gpu))
        && q8_state
        && dims.linear_key_heads * 3 == dims.linear_value_heads
        && dims.is_dense_27b_shape();
    compact3.then_some(3)
}

/// Keep DeltaNet's normalized output on chip and feed the exact MQ rotation
/// from LDS; each pair of 128-value heads forms one 256-value MQ group.
/// `HIPFIRE_GATED_NORM_MQ_ROTATE=0` restores both explicit operations.
pub fn gated_norm_mq_rotate(gpu: &Gpu, dims: &HybridDims, wo: &WeightRef<'_>) -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| developer_var("HIPFIRE_GATED_NORM_MQ_ROTATE").as_deref() != Some("0"));
    let enabled = *ENABLED;
    let n_v_heads = dims.linear_value_heads;
    let admitted_arch_shape =
        ((gpu.arch_caps.is_gfx1100() || gfx1151_radiowave(gpu) || gfx1201_state_fusions(gpu))
            && dims.dim == 2_048
            && n_v_heads == 32)
            || ((gpu.arch_caps.is_gfx1100() || gfx1201_state_fusions(gpu))
                && dims.is_dense_27b_shape());
    enabled
        && admitted_arch_shape
        && dims.linear_value_dim == 128
        && wo.k == n_v_heads * dims.linear_value_dim
        && is_mq4(wo.dtype)
}

/// One head-local launch for Q/gate deinterleave, Q/K RMS norm and partial
/// half-split RoPE on the certified shapes. `HIPFIRE_QWEN35_FA_PREP_FUSE=0`
/// retains the multi-dispatch path.
pub fn fused_attention_prep(gpu: &Gpu, dims: &HybridDims) -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| developer_var("HIPFIRE_QWEN35_FA_PREP_FUSE").as_deref() != Some("0"));
    let enabled = *ENABLED;
    let admitted_arch_shape = ((gpu.arch_caps.is_gfx1100() || gfx1151_radiowave(gpu))
        && dims.n_heads == 16
        && dims.n_kv_heads == 2)
        || (gfx1201_state_fusions(gpu) && dims.is_a3b_state_fusion_shape())
        || ((gpu.arch_caps.is_gfx1100() || gfx1201_state_fusions(gpu))
            && dims.is_dense_27b_shape()
            && dims.n_heads == 24
            && dims.n_kv_heads == 4);
    enabled
        && admitted_arch_shape
        && !gpu.flags.rope_interleaved_legacy
        && dims.head_dim == 256
        && dims.n_rot == 64
}

/// Fold the output gate plus MQ rotation into the flash-attention reduce
/// epilogue on certified MQ4 shapes. `HIPFIRE_QWEN35_FA_EPILOGUE_FUSE=0`
/// retains the separate gate.
fn fused_attention_epilogue(gpu: &Gpu, dims: &HybridDims, wo: &WeightRef<'_>) -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| developer_var("HIPFIRE_QWEN35_FA_EPILOGUE_FUSE").as_deref() != Some("0"));
    let enabled = *ENABLED;
    let admitted_arch_shape = ((gpu.arch_caps.is_gfx1100() || gfx1151_radiowave(gpu))
        && dims.n_heads == 16
        && dims.n_kv_heads == 2)
        || (gfx1201_state_fusions(gpu) && dims.is_a3b_state_fusion_shape())
        || (gpu.arch_caps.is_gfx1100()
            && dims.is_dense_27b_shape()
            && dims.n_heads == 24
            && dims.n_kv_heads == 4);
    enabled && admitted_arch_shape && dims.head_dim == 256 && is_mq4(wo.dtype)
}

/// fp8 is admitted only for the flash-tile attend key whose `[2+head_dim]` f32
/// partials the shared q8 gated reducer consumes unmodified; the scalar fp8
/// reader writes final output directly and never takes the fused epilogue.
pub fn attention_epilogue_route_supported(
    is_gfx1201: bool,
    q8_route: bool,
    asym3_route: bool,
    fp8_route: bool,
) -> bool {
    q8_route || fp8_route || (!is_gfx1201 && asym3_route)
}

/// Experimental: fold the DeltaNet beta/alpha scalar preparation into the
/// fixed-K QKVZA producer (`HIPFIRE_QKVZA_SCALAR_PREP=1`).
pub fn qkvza_scalar_prep(
    gpu: &Gpu,
    dims: &HybridDims,
    q8_state: bool,
    wqkv: &WeightRef<'_>,
    wz: &WeightRef<'_>,
    w_beta: &WeightRef<'_>,
    w_alpha: &WeightRef<'_>,
) -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| developer_var("HIPFIRE_QKVZA_SCALAR_PREP").as_deref() == Some("1"));
    let enabled = *ENABLED;
    let dtype = wqkv.dtype;
    let n_v_heads = dims.linear_value_heads;
    enabled
        && gpu.arch_caps.is_gfx1100()
        && gdn_compact_qk_div(gpu, dims, q8_state) == Some(2)
        && wqkv.k == 2_048
        && w_beta.m == n_v_heads
        && w_alpha.m == n_v_heads
        && wz.dtype == dtype
        && w_beta.dtype == dtype
        && w_alpha.dtype == dtype
        && matches!(
            dtype,
            DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ4CG256 | DType::HFQ4G256
        )
}

fn conv_qknorm(gpu: &Gpu, dims: &HybridDims, q8_state: bool) -> bool {
    let mode = developer_var("HIPFIRE_CONV_QKNORM");
    let arch_enabled =
        (gpu.arch_caps.is_gfx1201() || gpu.arch_caps.arch() == "gfx1100" || gfx1151_radiowave(gpu))
            && mode.as_deref() != Some("0");
    arch_enabled && q8_state && dims.linear_key_dim == 128
}

/// Schedule the beta/alpha transforms as one extra workgroup of the following
/// conv/QK-normalization dispatch.
fn conv_scalar_prep(gpu: &Gpu, dims: &HybridDims, q8_state: bool) -> bool {
    static MODE: LazyLock<Option<String>> =
        LazyLock::new(|| developer_var("HIPFIRE_CONV_SCALAR_PREP"));
    let enabled = match MODE.as_deref() {
        Some("0") => false,
        Some("1") => true,
        // Dense 27B on gfx1100: +0.79% at 512 tokens, 48 fewer dispatches.
        _ => dims.is_dense_27b_shape(),
    };
    let shape = developer_var("HIPFIRE_CONV_QKNORM_SHAPE");
    enabled
        && gpu.arch_caps.is_gfx1100()
        && dims.linear_value_heads <= 256
        && shape.as_deref().is_none_or(|v| v == "b256")
        && conv_qknorm(gpu, dims, q8_state)
}

// ── Shared stages ─────────────────────────────────────────────────────────

/// RMS norm (plus the input basis the first weight needs) followed by one
/// projection per `(weight, output)` pair sharing that normalized input. The
/// fusion table rewrites the projections into fused qkv/qkvza/gate-up kernels.
#[allow(clippy::too_many_arguments)]
pub fn normed_projection(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    x: &GpuTensor,
    norm: &GpuTensor,
    plain: &GpuTensor,
    normed: &GpuTensor,
    eps: f32,
    projections: &[(WeightRef<'_>, &GpuTensor)],
) -> Result<(), DispatchError> {
    let first = &projections
        .first()
        .ok_or_else(|| DispatchError::Hip("normed projection needs a weight".into()))?
        .0;
    let rotation = dtype_rotation_plan(first.dtype);
    let givens = rotation == RotationPlan::Givens;
    // Projection weights carry their Givens rotation only on the Givens route;
    // the AWQ scale is consumed by the norm, never by the GEMV.
    let weights: Vec<WeightRef<'_>> = projections
        .iter()
        .map(|(w, _)| WeightRef {
            row_stride: 0,
            rotation: if givens { w.rotation } else { None },
            awq_scale: None,
            ..*w
        })
        .collect();
    let mut steps = Vec::with_capacity(projections.len() + 1);
    steps.push(Step::RmsnormAutomatic {
        x,
        norm_weight: norm,
        x_plain: plain,
        out: normed,
        awq_scale: first.awq_scale,
        k: first.k,
        eps,
        rotation: if givens { RotationPlan::None } else { rotation },
    });
    for (w, (_, out)) in weights.iter().zip(projections) {
        steps.push(Step::Gemv {
            w,
            input: if givens {
                GemvInput::Raw(normed)
            } else {
                GemvInput::Prerotated(normed)
            },
            out,
        });
    }
    execute_steps(gpu, ctx, &steps)
}

static GEMV: LazyLock<GemvFamily> = LazyLock::new(GemvFamily::new);

fn mq_rotation_scratch(gpu: &mut Gpu) -> Result<GpuTensor, DispatchError> {
    let scratch = gpu
        .scratch
        .mq_x_rot
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("MQ rotation scratch is not allocated".into()))?;
    Ok(GpuTensor {
        buf: unsafe { scratch.buf.alias() },
        shape: vec![scratch.buf.size() / 4],
        dtype: DType::F32,
    })
}

/// `x += W_down · (silu(gate) ⊙ up)`. MQ weights fuse SiLU·mul with the input
/// rotation and run the SwiGLU-residual GEMV; HFQ weights use the residual
/// GEMV; everything else is a plain GEMV plus an in-place add.
pub fn swiglu_down_residual(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w_down: &WeightRef<'_>,
    gate: &GpuTensor,
    up: &GpuTensor,
    hidden: &GpuTensor,
    x: &GpuTensor,
) -> Result<(), DispatchError> {
    super::reject_mq4g128v2(w_down.dtype, "swiglu_down_residual")?;
    // Calibration tap: down_proj input, captured pre-rotation.
    gpu.maybe_capture_activation(w_down.buf, hidden, 1, w_down.k);
    let wr = WeightRef {
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        ..*w_down
    };
    match w_down.dtype {
        DType::MQ4G256
        | DType::MQ4G256V2
        | DType::MQ4G256V2Lloyd
        | DType::MQ4CG256
        | DType::MQ6G256
        | DType::MQ6G256V2
        | DType::MQ5G256V2
        | DType::MQ3G256
        | DType::MQ3G256V2
        | DType::MQ2G256V2
        | DType::MQ3G256Lloyd
        | DType::MQ4G256Lloyd => {
            gpu.ensure_mq_signs().map_err(hip)?;
            let xr = mq_rotation_scratch(gpu)?;
            match w_down.awq_scale {
                Some(awq) => gpu.fused_silu_mul_rotate_mq_awq(gate, up, awq, &xr, w_down.k),
                None => gpu.fused_silu_mul_rotate_mq(gate, up, &xr, w_down.k),
            }
            .map_err(hip)?;
            GEMV.run(
                ctx,
                gpu,
                &GemvParams {
                    w: &wr,
                    x: &xr,
                    y: x,
                    variant: GemvVariant::WithSwiGLUResidual,
                    residual: Some(x),
                    gate: Some(&xr),
                    up: None,
                },
            )
        }
        DType::HFQ4G256 | DType::HFQ3G256 | DType::HFQ6G256 => {
            gpu.silu_mul_f32(gate, up, hidden).map_err(hip)?;
            gpu.maybe_capture_activation(w_down.buf, hidden, 1, w_down.k);
            GEMV.run(
                ctx,
                gpu,
                &GemvParams {
                    w: &wr,
                    x: hidden,
                    y: x,
                    variant: GemvVariant::WithResidual,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
        }
        dtype if !dtype_needs_rotation(dtype) => {
            gpu.silu_mul_f32(gate, up, hidden).map_err(hip)?;
            gpu.maybe_capture_activation(w_down.buf, hidden, 1, w_down.k);
            let tmp = gpu.alloc_tensor(&[w_down.m], DType::F32).map_err(hip)?;
            let run = GEMV.run_auto(ctx, gpu, &wr, hidden, &tmp);
            let add = run.and_then(|_| gpu.add_inplace_f32(x, &tmp).map_err(hip));
            let free = gpu.free_tensor(tmp).map_err(hip);
            add.and(free)
        }
        other => Err(DispatchError::UnsupportedVariant {
            family: "hybrid",
            variant: "swiglu_down_residual",
            arch: "",
            quant: super::dtype_name(other),
        }),
    }
}

// ── Gated DeltaNet mixer ──────────────────────────────────────────────────

/// Linear-attention sublayer: norm → fused qkv/z/beta/alpha projection →
/// causal conv + QK norm + gates → gated delta recurrence → gated norm →
/// output projection added into `x`.
pub struct DeltaNetMixerOp<'a> {
    pub dims: HybridDims,
    pub rows: usize,
    /// Residual stream; the output projection accumulates into it.
    pub x: &'a GpuTensor,
    /// `true` when the previous op already wrote the normalized, rotated input
    /// into `x_rot` (cross-layer MoE combine + next-norm transition).
    pub prerotated_input: bool,
    pub attn_norm: &'a GpuTensor,
    pub wqkv: WeightRef<'a>,
    pub wz: WeightRef<'a>,
    pub w_beta: WeightRef<'a>,
    pub w_alpha: WeightRef<'a>,
    pub dt_bias: &'a GpuTensor,
    pub a_log: &'a GpuTensor,
    pub conv_weight: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub wo: WeightRef<'a>,
    pub conv_state: &'a GpuTensor,
    pub state: GdnState<'a>,
    pub plain: &'a GpuTensor,
    pub x_rot: &'a GpuTensor,
    pub qkv: &'a GpuTensor,
    pub z: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    pub alpha: &'a GpuTensor,
    pub q_raw: &'a GpuTensor,
    pub k_raw: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub recurrent_out: &'a GpuTensor,
    pub normed: &'a GpuTensor,
}

impl DeltaNetMixerOp<'_> {
    fn require_decode(&self) -> Result<(), DispatchError> {
        if self.rows != 1 {
            return Err(DispatchError::Hip(format!(
                "DeltaNet mixer rows={} has no batched executor yet",
                self.rows
            )));
        }
        Ok(())
    }

    fn scalar_prep(&self, gpu: &Gpu) -> bool {
        qkvza_scalar_prep(
            gpu,
            &self.dims,
            self.state.is_q8(),
            &self.wqkv,
            &self.wz,
            &self.w_beta,
            &self.w_alpha,
        )
    }

    /// Norm + fused qkv/z/beta/alpha projection.
    pub fn project(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<(), DispatchError> {
        self.require_decode()?;
        let (wqkv, wz, wb, wa) = (&self.wqkv, &self.wz, &self.w_beta, &self.w_alpha);
        if self.prerotated_input {
            return gpu
                .fused_qkvza_hfq4g256(
                    wqkv.buf, wz.buf, wb.buf, wa.buf, self.x_rot, self.qkv, self.z, self.beta,
                    self.alpha, wqkv.m, wz.m, wb.m, wa.m, wqkv.k,
                )
                .map_err(hip);
        }
        if self.scalar_prep(gpu) {
            // Admitted dtypes are MQ4 (fused norm + rotation) or HFQ4 (plain norm).
            let input = if is_mq4(wqkv.dtype) {
                match wqkv.awq_scale {
                    Some(awq) => gpu.fused_rmsnorm_rotate_mq_awq(
                        self.x,
                        self.attn_norm,
                        awq,
                        self.x_rot,
                        wqkv.k,
                        self.dims.norm_eps,
                    ),
                    None => gpu.fused_rmsnorm_rotate_mq(
                        self.x,
                        self.attn_norm,
                        self.x_rot,
                        wqkv.k,
                        self.dims.norm_eps,
                    ),
                }
                .map_err(hip)?;
                self.x_rot
            } else {
                gpu.rmsnorm_f32(self.x, self.attn_norm, self.plain, self.dims.norm_eps)
                    .map_err(hip)?;
                self.plain
            };
            return gpu
                .fused_qkvza_hfq4g256_scalar_prep_gfx1100(
                    wqkv.buf,
                    wz.buf,
                    wb.buf,
                    wa.buf,
                    input,
                    self.qkv,
                    self.z,
                    self.beta,
                    self.alpha,
                    self.dt_bias,
                    self.a_log,
                    wqkv.m,
                    wz.m,
                    wb.m,
                    wa.m,
                    wqkv.k,
                )
                .map_err(hip);
        }
        normed_projection(
            gpu,
            ctx,
            self.x,
            self.attn_norm,
            self.plain,
            self.x_rot,
            self.dims.norm_eps,
            &[
                (*wqkv, self.qkv),
                (*wz, self.z),
                (*wb, self.beta),
                (*wa, self.alpha),
            ],
        )
    }

    /// Gates, causal conv with SiLU split, QK L2 norm and Q/K head expansion.
    pub fn prepare(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        self.require_decode()?;
        let dims = &self.dims;
        let q8 = self.state.is_q8();
        let n_v_heads = dims.linear_value_heads;
        let hd = dims.linear_key_dim;
        let (k_dim, v_dim) = (dims.key_width(), dims.value_width());
        let qkvza_scalar_prep = self.scalar_prep(gpu);
        let conv_scalar = !qkvza_scalar_prep && conv_scalar_prep(gpu, dims, q8);
        if !qkvza_scalar_prep && !conv_scalar {
            gpu.fused_sigmoid_alpha_gate_f32(
                self.beta,
                self.alpha,
                self.dt_bias,
                self.a_log,
                n_v_heads,
            )
            .map_err(hip)?;
        }
        let scale = 1.0 / (hd as f32).sqrt();
        if conv_qknorm(gpu, dims, q8) {
            if conv_scalar {
                gpu.conv1d_silu_split_qknorm_scalar_prep_gfx1100(
                    self.q_raw,
                    self.k_raw,
                    self.v,
                    self.qkv,
                    self.conv_weight,
                    self.conv_state,
                    self.beta,
                    self.alpha,
                    self.dt_bias,
                    self.a_log,
                    k_dim,
                    v_dim,
                    dims.linear_key_heads,
                    hd,
                    scale,
                    dims.norm_eps,
                    n_v_heads,
                )
            } else {
                gpu.conv1d_silu_split_qknorm(
                    self.q_raw,
                    self.k_raw,
                    self.v,
                    self.qkv,
                    self.conv_weight,
                    self.conv_state,
                    k_dim,
                    v_dim,
                    dims.linear_key_heads,
                    hd,
                    scale,
                    dims.norm_eps,
                )
            }
            .map_err(hip)?;
        } else {
            gpu.conv1d_silu_split_f32(
                self.q_raw,
                self.k_raw,
                self.v,
                self.qkv,
                self.conv_weight,
                self.conv_state,
                k_dim,
                v_dim,
            )
            .map_err(hip)?;
            gpu.fused_qk_l2_norm_scale_f32(
                self.q_raw,
                self.k_raw,
                dims.linear_key_heads,
                hd,
                scale,
                dims.norm_eps,
            )
            .map_err(hip)?;
        }
        if gdn_compact_qk_div(gpu, dims, q8).is_none() {
            // Ratio 1 stays on the same kernel surface as ratio > 1: one
            // deterministic Q+K kernel instead of two runtime memcpy nodes
            // keeps the whole decode body recordable by Redline AQL.
            let ratio = if dims.linear_key_heads < n_v_heads {
                n_v_heads / dims.linear_key_heads
            } else {
                1
            };
            gpu.repeat_interleave_qk_f32(
                self.q_raw,
                self.k_raw,
                self.q,
                self.k,
                dims.linear_key_heads,
                ratio,
                hd,
            )
            .map_err(hip)?;
        }
        Ok(())
    }

    /// Gated delta recurrence over the layer's state.
    pub fn recur(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        self.require_decode()?;
        let dims = &self.dims;
        let (heads, vd) = (dims.linear_value_heads, dims.linear_value_dim);
        match self.state {
            GdnState::F32 { state } => gpu.gated_delta_net_f32(
                self.q,
                self.k,
                self.v,
                self.alpha,
                self.beta,
                state,
                self.recurrent_out,
                1,
                heads,
                vd,
            ),
            GdnState::Q8 {
                state,
                scales,
                error_feedback,
            } => match gdn_compact_qk_div(gpu, dims, true) {
                Some(qk_head_div) => gpu.gated_delta_net_q8_compact(
                    self.q_raw,
                    self.k_raw,
                    self.v,
                    self.alpha,
                    self.beta,
                    state,
                    scales,
                    self.recurrent_out,
                    1,
                    heads,
                    vd,
                    qk_head_div,
                    error_feedback,
                ),
                None => gpu.gated_delta_net_q8(
                    self.q,
                    self.k,
                    self.v,
                    self.alpha,
                    self.beta,
                    state,
                    scales,
                    self.recurrent_out,
                    1,
                    heads,
                    vd,
                    error_feedback,
                ),
            },
            GdnState::Q4 { state, scales } => gpu.gated_delta_net_q4(
                self.q,
                self.k,
                self.v,
                self.alpha,
                self.beta,
                state,
                scales,
                self.recurrent_out,
                1,
                heads,
                vd,
            ),
        }
        .map_err(hip)
    }

    fn norm_rotates(&self, gpu: &Gpu) -> bool {
        gated_norm_mq_rotate(gpu, &self.dims, &self.wo)
    }

    /// Gated RMS norm of the recurrence output (optionally fused with the
    /// output projection's MQ rotation).
    pub fn gated_norm(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        self.require_decode()?;
        let dims = &self.dims;
        if self.norm_rotates(gpu) {
            gpu.gated_norm_rotate_mq_gfx1100(
                self.recurrent_out,
                self.z,
                self.norm_weight,
                self.wo.awq_scale,
                self.x_rot,
                dims.linear_value_heads,
                dims.linear_value_dim,
                dims.norm_eps,
            )
        } else {
            gpu.gated_norm_f32(
                self.recurrent_out,
                self.z,
                self.norm_weight,
                self.normed,
                dims.linear_value_heads,
                dims.linear_value_dim,
                dims.norm_eps,
            )
        }
        .map_err(hip)
    }

    /// Output projection accumulated into the residual stream.
    pub fn output(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<(), DispatchError> {
        let input = if self.norm_rotates(gpu) {
            GemvInput::Prerotated(self.x_rot)
        } else {
            GemvInput::Raw(self.normed)
        };
        execute_steps(
            gpu,
            ctx,
            &[Step::GemvResidual {
                w: &self.wo,
                input,
                residual: self.x,
                out: self.x,
            }],
        )
    }
}

pub fn execute_deltanet_mixer(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &DeltaNetMixerOp<'_>,
) -> Result<(), DispatchError> {
    op.project(gpu, ctx)?;
    op.prepare(gpu)?;
    op.recur(gpu)?;
    op.gated_norm(gpu)?;
    op.output(gpu, ctx)
}

// ── Gated full-attention mixer ────────────────────────────────────────────

/// Observer of the pre-RoPE Q/K of one attention layer (diagnostic tap).
/// Arguments: pre-RoPE Q rows, K rows, row count.
pub type AttentionTap<'a> =
    &'a dyn Fn(&mut Gpu, &GpuTensor, &GpuTensor, usize) -> hip_bridge::HipResult<()>;

/// KV storage one attention layer writes and reads.
pub struct AttentionKv<'a> {
    pub tier: KvTierInputs,
    pub k_cache: &'a GpuTensor,
    pub v_cache: &'a GpuTensor,
    pub physical_cap: usize,
    pub givens_cos: Option<&'a GpuTensor>,
    pub givens_sin: Option<&'a GpuTensor>,
    /// Logical positions evicted ahead of the physical slot (KV compaction).
    pub compact_offset: usize,
}

/// Full-attention sublayer: norm → q(+gate)/k/v projection → Q/gate split,
/// Q/K norm, partial RoPE → KV write + flash attention → sigmoid output gate →
/// output projection added into `x`.
pub struct GatedAttentionOp<'a> {
    pub dims: HybridDims,
    pub rows: usize,
    pub position: usize,
    pub x: &'a GpuTensor,
    /// See [`DeltaNetMixerOp::prerotated_input`].
    pub prerotated_input: bool,
    pub attn_norm: &'a GpuTensor,
    pub wq: WeightRef<'a>,
    pub wk: WeightRef<'a>,
    pub wv: WeightRef<'a>,
    pub q_norm: &'a GpuTensor,
    pub k_norm: &'a GpuTensor,
    pub wo: WeightRef<'a>,
    pub kv: AttentionKv<'a>,
    pub pos_buf: &'a hip_bridge::DeviceBuffer,
    pub plain: &'a GpuTensor,
    pub x_rot: &'a GpuTensor,
    /// Interleaved Q and gate, `[n_heads, 2 * head_dim]`.
    pub q_gate: &'a GpuTensor,
    pub q: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub flash_partials: &'a GpuTensor,
    pub attn_out: &'a GpuTensor,
    pub tap: Option<AttentionTap<'a>>,
    /// Multimodal (t, h, w) RoPE; `None` rotates at `pos_buf`.
    pub mrope: Option<MropeRope<'a>>,
}

/// Three-section RoPE positions of one token (vision-language prompts).
#[derive(Clone, Copy)]
pub struct MropeRope<'a> {
    /// `[3]` i32 (t, h, w) rotary phases, already offset by KV compaction.
    pub positions: &'a hip_bridge::DeviceBuffer,
    /// Frequency counts of the t, h and w sections.
    pub section: [usize; 3],
}

impl GatedAttentionOp<'_> {
    fn require_decode(&self) -> Result<(), DispatchError> {
        if self.rows != 1 {
            return Err(DispatchError::Hip(format!(
                "gated attention rows={} has no batched executor yet",
                self.rows
            )));
        }
        Ok(())
    }

    pub fn project(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<(), DispatchError> {
        self.require_decode()?;
        let (wq, wk, wv) = (&self.wq, &self.wk, &self.wv);
        if self.prerotated_input {
            return gpu
                .fused_qkv_hfq4g256(
                    wq.buf,
                    wk.buf,
                    wv.buf,
                    self.x_rot,
                    self.q_gate,
                    self.k,
                    self.v,
                    wq.m,
                    wk.m,
                    wv.m,
                    wq.k,
                )
                .map_err(hip);
        }
        normed_projection(
            gpu,
            ctx,
            self.x,
            self.attn_norm,
            self.plain,
            self.x_rot,
            self.dims.norm_eps,
            &[(*wq, self.q_gate), (*wk, self.k), (*wv, self.v)],
        )
    }

    /// Q/K preparation, KV write and attention. Returns whether the output
    /// gate (and the output projection's rotation) ran inside the epilogue.
    pub fn attend(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<bool, DispatchError> {
        self.require_decode()?;
        let dims = &self.dims;
        let fused_prep =
            fused_attention_prep(gpu, dims) && self.tap.is_none() && self.mrope.is_none();
        if !fused_prep {
            gpu.deinterleave_f32(self.q_gate, self.q, self.gate, dims.n_heads, dims.head_dim)
                .map_err(hip)?;
            gpu.rmsnorm_batched(
                self.q,
                self.q_norm,
                self.q,
                dims.n_heads,
                dims.head_dim,
                dims.norm_eps,
            )
            .map_err(hip)?;
            gpu.rmsnorm_batched(
                self.k,
                self.k_norm,
                self.k,
                dims.n_kv_heads,
                dims.head_dim,
                dims.norm_eps,
            )
            .map_err(hip)?;
        }
        if let Some(tap) = self.tap {
            tap(gpu, self.q, self.k, 1).map_err(hip)?;
        }
        // RoPE uses the logical position; the KV slot stays physical.
        let compacted = self.kv.compact_offset > 0;
        if compacted {
            let logical = (self.position + self.kv.compact_offset) as i32;
            gpu.memcpy_htod_auto(self.pos_buf, &logical.to_ne_bytes())
                .map_err(hip)?;
        }
        if let Some(mrope) = self.mrope {
            gpu.rope_mrope_halfsplit_f32(
                self.q,
                self.k,
                mrope.positions,
                dims.n_heads,
                dims.n_kv_heads,
                dims.head_dim,
                dims.n_rot,
                dims.rope_theta,
                mrope.section,
            )
        } else if fused_prep {
            gpu.qwen35_fa_prep_gfx1100(
                self.q_gate,
                self.q,
                self.gate,
                self.k,
                self.q_norm,
                self.k_norm,
                self.pos_buf,
                dims.norm_eps,
                dims.rope_theta,
                dims.n_heads,
                dims.n_kv_heads,
            )
        } else {
            gpu.rope_partial_interleaved_f32(
                self.q,
                self.k,
                self.pos_buf,
                dims.n_heads,
                dims.n_kv_heads,
                dims.head_dim,
                dims.n_rot,
                dims.rope_theta,
            )
        }
        .map_err(hip)?;
        if compacted {
            let physical = self.position as i32;
            gpu.memcpy_htod_auto(self.pos_buf, &physical.to_ne_bytes())
                .map_err(hip)?;
        }
        let plan = KvTierPlan::derive(KvTierInputs {
            pos: self.position,
            capture_mode: gpu.graphs.capture_mode,
            ..self.kv.tier
        })
        .map_err(hip)?;
        let route = |write, attend| plan.write_key == write && plan.attend_key == attend;
        let epilogue_route = attention_epilogue_route_supported(
            gpu.arch_caps.is_gfx1201(),
            route(KernelKey::KvWriteQ8_0, KernelKey::AttnFlashQ8_0),
            route(KernelKey::KvWriteAsym3, KernelKey::AttnFlashAsym3),
            route(KernelKey::KvWriteFp8E4m3, KernelKey::AttnFlashFp8E4m3),
        );
        let fused_epilogue = fused_attention_epilogue(gpu, dims, &self.wo) && epilogue_route;
        let io = AttnParams {
            q: self.q,
            k: self.k,
            v: self.v,
            k_cache: self.kv.k_cache,
            v_cache: self.kv.v_cache,
            k_scales: None,
            v_scales: None,
            pos_buf: self.pos_buf,
            pos: self.position,
            positions: None,
            n_heads: dims.n_heads,
            n_kv_heads: dims.n_kv_heads,
            head_dim: dims.head_dim,
            physical_cap: self.kv.physical_cap,
            batch_size: 1,
            max_ctx_len: 0,
            flash_partials: Some(self.flash_partials),
            givens_cos: self.kv.givens_cos,
            givens_sin: self.kv.givens_sin,
            tree_bias: None,
            block_start: 0,
            block_cols: 0,
            output_gate: fused_epilogue.then_some(self.gate),
            output_awq_scale: if fused_epilogue {
                self.wo.awq_scale
            } else {
                None
            },
            output: self.attn_out,
        };
        execute_steps(gpu, ctx, &[Step::Attend { plan, io }])?;
        if !fused_epilogue {
            gpu.sigmoid_mul_f32(self.attn_out, self.gate).map_err(hip)?;
        }
        Ok(fused_epilogue)
    }

    pub fn output(
        &self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        prerotated: bool,
    ) -> Result<(), DispatchError> {
        let input = if prerotated {
            GemvInput::Prerotated(self.attn_out)
        } else {
            GemvInput::Raw(self.attn_out)
        };
        execute_steps(
            gpu,
            ctx,
            &[Step::GemvResidual {
                w: &self.wo,
                input,
                residual: self.x,
                out: self.x,
            }],
        )
    }
}

pub fn execute_gated_attention(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &GatedAttentionOp<'_>,
) -> Result<(), DispatchError> {
    op.project(gpu, ctx)?;
    let prerotated = op.attend(gpu, ctx)?;
    op.output(gpu, ctx, prerotated)
}

// ── Dense SwiGLU FFN ──────────────────────────────────────────────────────

/// Dense FFN sublayer: norm → gate/up projection → SwiGLU down projection
/// added into `x`. With `rows > 1` the tensors are `[rows, …]` batches and
/// `batch` carries the batched-only operands.
pub struct SwigluFfnOp<'a> {
    pub rows: usize,
    pub eps: f32,
    pub x: &'a GpuTensor,
    pub norm: &'a GpuTensor,
    pub w_gate: WeightRef<'a>,
    pub w_up: WeightRef<'a>,
    pub w_down: WeightRef<'a>,
    pub plain: &'a GpuTensor,
    pub x_rot: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub hidden: &'a GpuTensor,
    pub batch: Option<crate::pipeline::batched::SwigluFfnBatch<'a>>,
}

impl SwigluFfnOp<'_> {
    pub fn project(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<(), DispatchError> {
        if self.rows != 1 {
            return Err(DispatchError::Hip(format!(
                "SwiGLU FFN rows={} has no batched executor yet",
                self.rows
            )));
        }
        normed_projection(
            gpu,
            ctx,
            self.x,
            self.norm,
            self.plain,
            self.x_rot,
            self.eps,
            &[(self.w_gate, self.gate), (self.w_up, self.up)],
        )
    }

    pub fn down(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<(), DispatchError> {
        swiglu_down_residual(
            gpu,
            ctx,
            &self.w_down,
            self.gate,
            self.up,
            self.hidden,
            self.x,
        )
    }
}

pub fn execute_swiglu_ffn(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &SwigluFfnOp<'_>,
) -> Result<(), DispatchError> {
    if let Some(batch) = &op.batch {
        return crate::pipeline::batched::execute_swiglu_ffn_batched(gpu, op, batch).map_err(hip);
    }
    op.project(gpu, ctx)?;
    op.down(gpu, ctx)
}
