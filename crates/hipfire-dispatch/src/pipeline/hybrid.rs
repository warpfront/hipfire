// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Hybrid linear/full-attention decoder layer operations.
//!
//! A hybrid decoder interleaves gated-DeltaNet linear attention with gated
//! full attention, each followed by a dense SwiGLU or MoE FFN. Architecture
//! crates declare one operation per sublayer and bind their resident weights,
//! recurrent state, KV storage and fixed scratch; the arch-, shape- and
//! dtype-gated route choices of these sublayers live here, once. (MoE routing
//! and the cross-layer MoE combine gate stay with the sealed MoE program and
//! its caller.)
//!
//! The executor bodies are the former Qwen3.5 single-token decode super-op
//! handlers moved verbatim, so every route issues the same launches in the
//! same order.

use crate::context::DispatchCtx;
use crate::families::attention::{AttentionFamily, AttnParams};
use crate::families::gemv::WeightRef;
use crate::families::kv_tier::{KvTierInputs, KvTierPlan};
use crate::pipeline::steps::{execute_steps, GemvInput, Step};
use crate::types::{dtype_rotation_plan, DispatchError, KernelKey, RotationPlan};
use hip_bridge::DeviceBuffer;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::LazyLock;

fn hip(e: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(e.to_string())
}

fn developer_var(name: &str) -> Option<String> {
    hipfire_config::developer_var(name).ok()
}

static ATTENTION: LazyLock<AttentionFamily> = LazyLock::new(AttentionFamily::new);

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

    /// Exact dense Qwen3.6-27B shape shared by its gfx1100 decode selectors.
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

    fn is_a3b_state_fusion_shape(&self) -> bool {
        self.dim == 2_048
            && self.n_heads == 16
            && self.n_kv_heads == 2
            && self.head_dim == 256
            && self.linear_key_heads == 16
            && self.linear_value_heads == 32
            && self.linear_key_dim == 128
            && self.linear_value_dim == 128
    }

    /// The gfx1100 campaign adds defaults only for the measured Qwen3.5-4B
    /// geometry. Existing A3B/27B admissions and the gfx1101 experiment are
    /// separate gates.
    fn is_gfx1100_4b_campaign_shape(&self, arch: &str) -> bool {
        arch == "gfx1100"
            && self.dim == 2_560
            && self.n_heads == 16
            && self.n_kv_heads == 4
            && self.head_dim == 256
            && self.n_rot == 64
            && self.linear_key_heads == 16
            && self.linear_key_dim == 128
            && self.linear_value_heads == 32
            && self.linear_value_dim == 128
            && self.conv_kernel == 4
            && self.num_experts == 0
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

/// Exact gfx1151 admission gate for its certified Radiowave decode bundle.
/// Keep this separate from broad RDNA3 capability checks so no neighboring
/// architecture can inherit the gfx1151 schedules.
pub fn gfx1151_radiowave(gpu: &Gpu) -> bool {
    gpu.arch_caps.is_gfx1151()
}

/// Exact gfx1201 admission gate for the ported decode state fusions
/// (gated-norm/MQ rotation, full-attention prep, gated MQ-rotate FA epilogue).
pub fn gfx1201_state_fusions(gpu: &Gpu) -> bool {
    gpu.arch_caps.is_gfx1201()
}

/// Experimental gate: run the gfx1100-certified campaign fusions on a gfx1101
/// device (same RDNA3 ISA, same wave32 WMMA; kernels compile for the runtime
/// arch via --offload-arch). Set HIPFIRE_GFX1101_GFX1100_CAMPAIGN=1. NOT a
/// certified path — diagnostic A/B only.
fn gfx1101_campaign(gpu: &Gpu) -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| developer_var("HIPFIRE_GFX1101_GFX1100_CAMPAIGN").as_deref() == Some("1"));
    gpu.arch_caps.is_gfx1101() && *ENABLED
}

fn is_mq4(dtype: DType) -> bool {
    matches!(dtype, DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ4CG256)
}

/// Decode path that keeps DeltaNet Q/K at their native head count and lets
/// each pair of value/state heads reuse one Q/K head. The compact B2 schedule
/// is the certified default on gfx1100 and gfx1201; `HIPFIRE_GDN_COMPACT2=0`
/// is a fail-closed escape hatch.
fn gdn_compact2(gpu: &Gpu, dims: &HybridDims, q8_state: bool) -> bool {
    let mode = developer_var("HIPFIRE_GDN_COMPACT2");
    let arch_enabled =
        (gpu.arch_caps.is_gfx1201() || gpu.arch_caps.arch() == "gfx1100" || gfx1151_radiowave(gpu))
            && mode.as_deref() != Some("0");
    arch_enabled && q8_state && dims.linear_key_heads * 2 == dims.linear_value_heads
}

/// 3:1 compact-QK route for Qwen3.6-27B dense. A 512-token graph-on A/B/B/A
/// measured +0.45%, while profiling confirmed 853 -> 805 dispatches/token and
/// a net 64.35 us/token reduction in GPU compute. Exact gfx1201 runs the same
/// `gated_delta_net_q8_compact3_b2` object for the same shape (its state,
/// output and EF bytes equal the materialized repeat-interleave route). Set
/// `HIPFIRE_GDN_COMPACT3=0` to restore explicit Q/K materialization.
fn gdn_compact3(gpu: &Gpu, dims: &HybridDims, q8_state: bool) -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| developer_var("HIPFIRE_GDN_COMPACT3").as_deref() != Some("0"));
    *ENABLED
        && (gpu.arch_caps.is_gfx1100() || gfx1201_state_fusions(gpu))
        && q8_state
        && dims.linear_key_heads * 3 == dims.linear_value_heads
        && dims.is_dense_27b_shape()
}

/// Compact DeltaNet QK: `Some(ratio)` when each `ratio` value/state heads
/// share one Q/K head and the expanded Q/K is never materialized.
pub fn gdn_compact_qk_div(gpu: &Gpu, dims: &HybridDims, q8_state: bool) -> Option<usize> {
    if gdn_compact2(gpu, dims, q8_state) {
        Some(2)
    } else if gdn_compact3(gpu, dims, q8_state) {
        Some(3)
    } else {
        None
    }
}

/// Keep DeltaNet's normalized output on chip and feed the exact MQ rotation
/// directly from LDS. Each pair of 128-value heads forms one 256-value MQ
/// group without changing either operation's arithmetic order. The 32-head
/// A3B route remains isolated from the 48-head Qwen3.6-27B route (gfx1100 and
/// exact gfx1201, each with its own k6144 entry symbol); on gfx1100 the latter
/// measured +0.45% over a 512-token A/B/B/A and removes 48 dispatches/token.
/// Set `HIPFIRE_GATED_NORM_MQ_ROTATE=0` to restore both explicit operations.
pub fn gated_norm_mq_rotate(gpu: &Gpu, dims: &HybridDims, wo: &WeightRef<'_>) -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| developer_var("HIPFIRE_GATED_NORM_MQ_ROTATE").as_deref() != Some("0"));
    let n_v_heads = dims.linear_value_heads;
    let admitted_arch_shape = ((gpu.arch_caps.is_gfx1100()
        || gfx1151_radiowave(gpu)
        || gfx1201_state_fusions(gpu))
        && dims.dim == 2_048
        && n_v_heads == 32)
        // New gfx1100 default: only the measured 4B geometry.
        || dims.is_gfx1100_4b_campaign_shape(gpu.arch_caps.arch())
        // Preserve the separately opted-in gfx1101 experiment.
        || (gfx1101_campaign(gpu) && dims.dim == 2_560 && n_v_heads == 32)
        || ((gpu.arch_caps.is_gfx1100() || gfx1201_state_fusions(gpu))
            && dims.is_dense_27b_shape());
    *ENABLED
        && admitted_arch_shape
        && dims.linear_value_dim == 128
        && wo.k == n_v_heads * dims.linear_value_dim
        && is_mq4(wo.dtype)
}

/// Fold the single-token Q8_0 KV-cache write into the FA-prep epilogue on the
/// certified gfx1100 Qwen3.5-4B shape (16Q/4K, head_dim 256). Removes one
/// `kv_cache_write_q8_0_pair` launch per full-attention layer; cache bytes are
/// bit-identical to the pair writer. Default OFF — set
/// `HIPFIRE_FA_KVWRITE_FOLD=1` to enable.
fn fused_attention_kv_write(gpu: &Gpu, dims: &HybridDims) -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| developer_var("HIPFIRE_FA_KVWRITE_FOLD").as_deref() == Some("1"));
    *ENABLED
        && (gpu.arch_caps.is_gfx1100() || gfx1101_campaign(gpu))
        && dims.n_heads == 16
        && dims.n_kv_heads == 4
        && dims.head_dim == 256
}

/// The folded prep writes Q8_0 K/V rows itself, so the plan must be the
/// single-token Q8_0 flash write/attend pair.
fn kv_write_fold_route_supported(plan: &KvTierPlan) -> bool {
    plan.write_key == KernelKey::KvWriteQ8_0
        && plan.attend_key == KernelKey::AttnFlashQ8_0
        && plan.batch_size == 1
        && plan.v_mode_bits == 8
}

/// Collapse full-attention Q/gate deinterleave, Q/K RMS normalization, and
/// partial half-split RoPE into one head-local launch on the certified gfx1100
/// shapes (and their exact-gfx1201 twins). The Qwen3.6-27B q24k4 route
/// measured +0.64% over a 512-token A/B/B/A on gfx1100 and reduced 757 -> 709
/// dispatches/token. Set `HIPFIRE_QWEN35_FA_PREP_FUSE=0` to retain the legacy
/// path. The legacy interleaved-RoPE compatibility mode and diagnostic tap
/// retain the established multi-dispatch path.
pub fn fused_attention_prep(gpu: &Gpu, dims: &HybridDims) -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| developer_var("HIPFIRE_QWEN35_FA_PREP_FUSE").as_deref() != Some("0"));
    let admitted_arch_shape = ((gpu.arch_caps.is_gfx1100() || gfx1151_radiowave(gpu))
        && dims.n_heads == 16
        && dims.n_kv_heads == 2)
        || (gfx1201_state_fusions(gpu) && dims.is_a3b_state_fusion_shape())
        || dims.is_gfx1100_4b_campaign_shape(gpu.arch_caps.arch())
        || (gfx1101_campaign(gpu) && dims.n_heads == 16 && dims.n_kv_heads == 4)
        || ((gpu.arch_caps.is_gfx1100() || gfx1201_state_fusions(gpu))
            && dims.is_dense_27b_shape()
            && dims.n_heads == 24
            && dims.n_kv_heads == 4);
    *ENABLED
        && admitted_arch_shape
        && !gpu.flags.rope_interleaved_legacy
        && dims.head_dim == 256
        && dims.n_rot == 64
}

/// Fold the output gate plus MQ rotation into the flash-attention reduce
/// epilogue on certified gfx1100/MQ4 shapes. Extending the existing Q8 reducer
/// to Qwen3.6-27B's asym3 route measured +0.37% over a 512-token A/B/B/A and
/// reduced 709 -> 677 dispatches/token; a 1025-token replay remained exact.
/// Set `HIPFIRE_QWEN35_FA_EPILOGUE_FUSE=0` to retain the legacy path.
pub fn fused_attention_epilogue(gpu: &Gpu, dims: &HybridDims, wo: &WeightRef<'_>) -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| developer_var("HIPFIRE_QWEN35_FA_EPILOGUE_FUSE").as_deref() != Some("0"));
    let admitted_arch_shape = ((gpu.arch_caps.is_gfx1100() || gfx1151_radiowave(gpu))
        && dims.n_heads == 16
        && dims.n_kv_heads == 2)
        || (gfx1201_state_fusions(gpu) && dims.is_a3b_state_fusion_shape())
        || dims.is_gfx1100_4b_campaign_shape(gpu.arch_caps.arch())
        || (gfx1101_campaign(gpu) && dims.n_heads == 16 && dims.n_kv_heads == 4)
        || (gpu.arch_caps.is_gfx1100()
            && dims.is_dense_27b_shape()
            && dims.n_heads == 24
            && dims.n_kv_heads == 4);
    *ENABLED && admitted_arch_shape && dims.head_dim == 256 && is_mq4(wo.dtype)
}

/// fp8: admitted only for the flash-tile attend key whose [2+head_dim] f32
/// partials the shared q8 gated reducer consumes unmodified (StageB S6). The
/// scalar fp8 reader writes final output directly and has no fused path, so
/// it must never set output_gate.
fn attention_epilogue_route_supported(
    is_gfx1201: bool,
    q8_route: bool,
    asym3_route: bool,
    fp8_route: bool,
) -> bool {
    q8_route || fp8_route || (!is_gfx1201 && asym3_route)
}

/// Pure core of [`conv_scalar_prep`]: the arch, env `mode` and shape decision.
fn conv_scalar_prep_requested(
    arch: &str,
    dims: &HybridDims,
    mode: Option<&str>,
    gfx1101_campaign: bool,
) -> bool {
    if arch != "gfx1100" && !(arch == "gfx1101" && gfx1101_campaign) {
        return false;
    }
    match mode {
        Some("0") => false,
        Some("1") => true,
        _ => {
            dims.is_dense_27b_shape()
            || dims.is_gfx1100_4b_campaign_shape(arch)
            // The gfx1101 gate remains experimental; no default arch admission.
            || (arch == "gfx1101" && gfx1101_campaign)
        }
    }
}

/// Radiowave experiment: fold the DeltaNet beta/alpha scalar preparation into
/// the fixed-K QKVZA producer. Keep the gate deliberately narrow until exact
/// replay and stationary product certification justify a default flip.
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
    let dtype = wqkv.dtype;
    let n_v_heads = dims.linear_value_heads;
    *ENABLED
        && (gpu.arch_caps.is_gfx1100() || gfx1101_campaign(gpu))
        && gdn_compact2(gpu, dims, q8_state)
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

/// Schedule the independent beta/alpha transforms as one extra workgroup of
/// the following conv/QK-normalization dispatch. This keeps the hot QKVZA
/// projection unchanged while deleting the same boundary.
pub fn conv_scalar_prep(gpu: &Gpu, dims: &HybridDims, q8_state: bool) -> bool {
    static MODE: LazyLock<Option<String>> =
        LazyLock::new(|| developer_var("HIPFIRE_CONV_SCALAR_PREP"));
    // Keep the exact 27B default and add only the measured gfx1100 4B
    // geometry. Explicit value 1 retains the structurally supported experiment.
    let enabled = conv_scalar_prep_requested(
        gpu.arch_caps.arch(),
        dims,
        MODE.as_deref(),
        gfx1101_campaign(gpu),
    );
    let shape = developer_var("HIPFIRE_CONV_QKNORM_SHAPE");
    enabled
        && (gpu.arch_caps.is_gfx1100() || gfx1101_campaign(gpu))
        && dims.linear_value_heads <= 256
        && shape.as_deref().is_none_or(|v| v == "b256")
        && conv_qknorm(gpu, dims, q8_state)
}

pub fn conv_qknorm(gpu: &Gpu, dims: &HybridDims, q8_state: bool) -> bool {
    let mode = developer_var("HIPFIRE_CONV_QKNORM");
    let arch_enabled = (gpu.arch_caps.is_gfx1201()
        || gpu.arch_caps.arch() == "gfx1100"
        || gfx1101_campaign(gpu)
        || gfx1151_radiowave(gpu))
        && mode.as_deref() != Some("0");
    arch_enabled && q8_state && dims.linear_key_dim == 128
}

// ── Shared stages ─────────────────────────────────────────────────────────

/// RMS norm (plus the input basis the first weight needs) followed by one
/// projection per `(weight, output)` pair sharing that normalized input. The
/// fusion table rewrites the projections into fused qkv/qkvza/gate-up kernels.
/// Givens (ParoQuant) weights take a plain norm and rotate per GEMV; every
/// other format rotates once in the norm and the GEMVs read it prerotated.
#[allow(clippy::too_many_arguments)]
pub fn normed_projection<const N: usize>(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    x: &GpuTensor,
    norm: &GpuTensor,
    plain: &GpuTensor,
    normed: &GpuTensor,
    eps: f32,
    projections: [(WeightRef<'_>, &GpuTensor); N],
) -> Result<(), DispatchError> {
    const { assert!(N >= 1 && N <= 4) };
    let first = &projections[0].0;
    if projections.iter().any(|(w, _)| w.dtype.is_ggml()) {
        // GGUF-native: plain RMSNorm into `normed`, ONE q8_1 producer launch
        // shared by every ggml projection of this window, one mmvq per
        // projection (numeric contract `ggml.a8q81.<type>`). Float
        // projections of the same window (the GDN beta/alpha matrices are
        // stored F32 in GGUF files) read the same normed row through the
        // plain GEMV. No fusion table row exists for the ggml dtypes.
        if projections.iter().any(|(w, _)| w.k != first.k) {
            return Err(DispatchError::Hip(
                "normed projection: ggml window projections must share one input width".into(),
            ));
        }
        gpu.rmsnorm_f32(x, norm, normed, eps).map_err(hip)?;
        let q81 = gpu
            .ggml_quantize_activation(normed, first.k, None)
            .map_err(hip)?;
        for (w, out) in &projections {
            if w.dtype.is_ggml() {
                gpu.ggml_gemv_quantized(w.dtype, w.buf, w.m, w.k, &q81, out, false)
                    .map_err(hip)?;
            } else {
                let plain_w = WeightRef {
                    row_stride: 0,
                    rotation: None,
                    awq_scale: None,
                    ..*w
                };
                execute_steps(
                    gpu,
                    ctx,
                    &[Step::Gemv {
                        w: &plain_w,
                        input: GemvInput::Prerotated(normed),
                        out,
                    }],
                )?;
            }
        }
        return Ok(());
    }
    let rotation = dtype_rotation_plan(first.dtype);
    let givens = rotation == RotationPlan::Givens;
    // The AWQ scale is consumed by the norm, never by the GEMV.
    let weights = projections.map(|(w, _)| WeightRef {
        row_stride: 0,
        rotation: if givens { w.rotation } else { None },
        awq_scale: None,
        ..w
    });
    let input = if givens {
        GemvInput::Raw(normed)
    } else {
        GemvInput::Prerotated(normed)
    };
    let gemv = |i: usize| {
        let i = i.min(N - 1);
        Step::Gemv {
            w: &weights[i],
            input,
            out: projections[i].1,
        }
    };
    // Fixed-capacity program; only the first N + 1 steps run.
    let steps = [
        Step::RmsnormAutomatic {
            x,
            norm_weight: norm,
            x_plain: plain,
            out: normed,
            awq_scale: first.awq_scale,
            k: first.k,
            eps,
            rotation: if givens { RotationPlan::None } else { rotation },
            rows: 1,
        },
        gemv(0),
        gemv(1),
        gemv(2),
        gemv(3),
    ];
    execute_steps(gpu, ctx, &steps[..=N])
}

// ── Mixer output ──────────────────────────────────────────────────────────

/// Where a mixer's output projection lands.
#[derive(Clone, Copy)]
pub enum MixerOutput<'a> {
    /// `x += W_o · mix(x)` through the fused single-device decode body.
    Residual,
    /// `out = W_o · mix(x)` over this rank's heads: the tensor-parallel
    /// shard body (`SplitRule::HeadMixer`), whose caller reduces `out` across
    /// ranks. It runs the plain chain (no arch fusions, no compacted KV), so
    /// a rank's partial does not depend on which decode fusions its shard
    /// geometry would admit.
    Write(&'a GpuTensor),
}

// ── Gated DeltaNet mixer ──────────────────────────────────────────────────

/// Linear-attention sublayer: norm → fused qkv/z/beta/alpha projection →
/// causal conv + QK norm + gates → gated delta recurrence → gated norm →
/// output projection, added into `x` or written ([`MixerOutput`]).
pub struct DeltaNetMixerOp<'a> {
    pub dims: HybridDims,
    /// Residual stream; the output projection accumulates into it.
    pub x: &'a GpuTensor,
    /// `true` when the previous op already wrote this layer's normalized,
    /// rotated input into `x_rot` (cross-layer MoE combine + next-norm).
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
    pub output: MixerOutput<'a>,
}

impl DeltaNetMixerOp<'_> {
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
            [
                (*wqkv, self.qkv),
                (*wz, self.z),
                (*wb, self.beta),
                (*wa, self.alpha),
            ],
        )
    }

    /// Gates, causal conv with SiLU split, QK L2 norm and Q/K head expansion.
    pub fn prepare(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
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
        // The compact Q8 recurrence maps each state head directly to its
        // shared Q/K head, so the normalized tensors stay compact. Otherwise
        // ratio 1 stays on the same kernel as ratio > 1: one deterministic Q+K
        // kernel instead of two runtime memcpy nodes keeps the whole decode
        // body recordable by Redline AQL.
        if gdn_compact_qk_div(gpu, dims, q8).is_none() {
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

    /// Gated RMS norm of the recurrence output, fused with the output
    /// projection's MQ rotation where admitted.
    pub fn gated_norm(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
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
    if let MixerOutput::Write(out) = op.output {
        return deltanet_shard(gpu, ctx, op, out);
    }
    op.project(gpu, ctx)?;
    op.prepare(gpu)?;
    op.recur(gpu)?;
    op.gated_norm(gpu)?;
    op.output(gpu, ctx)
}

/// [`MixerOutput::Write`]: the dense tensor-parallel shard body over this
/// rank's key/value heads, its output projection written to `out`.
fn deltanet_shard(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &DeltaNetMixerOp<'_>,
    out: &GpuTensor,
) -> Result<(), DispatchError> {
    if op.prerotated_input {
        return Err(DispatchError::Hip(
            "deltanet shard: a prerotated input is not admitted".into(),
        ));
    }
    let dims = &op.dims;
    let (k_dim, v_dim) = (dims.key_width(), dims.value_width());
    let n_v_heads = dims.linear_value_heads;
    let hd = dims.linear_key_dim;
    normed_projection(
        gpu,
        ctx,
        op.x,
        op.attn_norm,
        op.plain,
        op.x_rot,
        dims.norm_eps,
        [
            (op.wqkv, op.qkv),
            (op.wz, op.z),
            (op.w_beta, op.beta),
            (op.w_alpha, op.alpha),
        ],
    )?;
    gpu.fused_sigmoid_alpha_gate_f32(op.beta, op.alpha, op.dt_bias, op.a_log, n_v_heads)
        .map_err(hip)?;
    gpu.conv1d_silu_split_f32(
        op.q_raw,
        op.k_raw,
        op.v,
        op.qkv,
        op.conv_weight,
        op.conv_state,
        k_dim,
        v_dim,
    )
    .map_err(hip)?;
    let scale = 1.0 / (hd as f32).sqrt();
    gpu.fused_qk_l2_norm_scale_f32(
        op.q_raw,
        op.k_raw,
        dims.linear_key_heads,
        hd,
        scale,
        dims.norm_eps,
    )
    .map_err(hip)?;
    if dims.linear_key_heads < n_v_heads {
        gpu.repeat_interleave_qk_f32(
            op.q_raw,
            op.k_raw,
            op.q,
            op.k,
            dims.linear_key_heads,
            n_v_heads / dims.linear_key_heads,
            hd,
        )
        .map_err(hip)?;
    } else {
        gpu.memcpy_dtod_auto(&op.q.buf, &op.q_raw.buf, k_dim * 4)
            .map_err(hip)?;
        gpu.memcpy_dtod_auto(&op.k.buf, &op.k_raw.buf, k_dim * 4)
            .map_err(hip)?;
    }
    let vd = dims.linear_value_dim;
    match op.state {
        GdnState::F32 { state } => gpu.gated_delta_net_f32(
            op.q,
            op.k,
            op.v,
            op.alpha,
            op.beta,
            state,
            op.recurrent_out,
            1,
            n_v_heads,
            vd,
        ),
        GdnState::Q8 {
            state,
            scales,
            error_feedback,
        } => gpu.gated_delta_net_q8(
            op.q,
            op.k,
            op.v,
            op.alpha,
            op.beta,
            state,
            scales,
            op.recurrent_out,
            1,
            n_v_heads,
            vd,
            error_feedback,
        ),
        GdnState::Q4 { state, scales } => gpu.gated_delta_net_q4(
            op.q,
            op.k,
            op.v,
            op.alpha,
            op.beta,
            state,
            scales,
            op.recurrent_out,
            1,
            n_v_heads,
            vd,
        ),
    }
    .map_err(hip)?;
    gpu.gated_norm_f32(
        op.recurrent_out,
        op.z,
        op.norm_weight,
        op.normed,
        n_v_heads,
        vd,
        dims.norm_eps,
    )
    .map_err(hip)?;
    execute_steps(
        gpu,
        ctx,
        &[Step::Gemv {
            w: &op.wo,
            input: GemvInput::Raw(op.normed),
            out,
        }],
    )
}

// ── Gated full-attention mixer ────────────────────────────────────────────

/// Observer of one attention layer's pre-RoPE Q/K (diagnostic tap). Its
/// presence keeps the unfused prep so Q/K exist in memory before RoPE.
pub type AttentionTap<'a> = &'a dyn Fn(&mut Gpu) -> Result<(), DispatchError>;

/// KV storage one attention layer writes and reads.
pub struct AttentionKv<'a> {
    /// Tier inputs of the cache (flash mode included); position and capture
    /// mode are filled in per launch.
    pub tier: KvTierInputs,
    pub k_cache: &'a GpuTensor,
    pub v_cache: &'a GpuTensor,
    pub physical_cap: usize,
    pub givens_cos: Option<&'a GpuTensor>,
    pub givens_sin: Option<&'a GpuTensor>,
    /// Logical positions evicted ahead of the physical slot (KV compaction).
    pub compact_offset: usize,
}

/// The decode row one attention layer reads and writes.
pub struct AttentionRow<'a> {
    /// Physical KV position of this token.
    pub position: usize,
    pub pos_buf: &'a DeviceBuffer,
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    /// Output gate, `[n_heads * head_dim]`.
    pub gate: &'a GpuTensor,
    pub flash_partials: &'a GpuTensor,
    pub out: &'a GpuTensor,
}

impl AttentionKv<'_> {
    fn tier_plan(&self, gpu: &Gpu, position: usize) -> Result<KvTierPlan, DispatchError> {
        KvTierPlan::derive(KvTierInputs {
            pos: position,
            capture_mode: gpu.graphs.capture_mode,
            ..self.tier
        })
        .map_err(hip)
    }
}

/// KV write + attention of one decode row. `kv_write_prewritten` marks that
/// the fused prep already wrote this token's Q8_0 K/V rows. Returns whether
/// the output gate (and `wo`'s input rotation) ran inside the epilogue.
pub fn kv_attention(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    dims: &HybridDims,
    kv: &AttentionKv<'_>,
    row: &AttentionRow<'_>,
    wo: &WeightRef<'_>,
    kv_write_prewritten: bool,
) -> Result<bool, DispatchError> {
    let plan = kv.tier_plan(gpu, row.position)?;
    let route = |write, attend| plan.write_key == write && plan.attend_key == attend;
    let fused_epilogue_route = attention_epilogue_route_supported(
        gpu.arch_caps.is_gfx1201(),
        route(KernelKey::KvWriteQ8_0, KernelKey::AttnFlashQ8_0),
        route(KernelKey::KvWriteAsym3, KernelKey::AttnFlashAsym3),
        route(KernelKey::KvWriteFp8E4m3, KernelKey::AttnFlashFp8E4m3),
    );
    let fused_epilogue = fused_attention_epilogue(gpu, dims, wo) && fused_epilogue_route;
    let io = AttnParams {
        q: row.q,
        k: row.k,
        v: row.v,
        k_cache: kv.k_cache,
        v_cache: kv.v_cache,
        k_scales: None,
        v_scales: None,
        pos_buf: row.pos_buf,
        pos: row.position,
        positions: None,
        n_heads: dims.n_heads,
        n_kv_heads: dims.n_kv_heads,
        head_dim: dims.head_dim,
        physical_cap: kv.physical_cap,
        batch_size: 1,
        max_ctx_len: 0,
        flash_partials: Some(row.flash_partials),
        givens_cos: kv.givens_cos,
        givens_sin: kv.givens_sin,
        tree_bias: None,
        block_start: 0,
        block_cols: 0,
        output_gate: fused_epilogue.then_some(row.gate),
        output_awq_scale: if fused_epilogue { wo.awq_scale } else { None },
        output: row.out,
    };
    if kv_write_prewritten {
        // The prep kernel's folded epilogue already wrote the Q8_0 K/V cache
        // rows for this token; dispatch attention only. The fold gate
        // guarantees the q8 write/attend route and a single decode token.
        if !kv_write_fold_route_supported(&plan) {
            return Err(DispatchError::Hip(
                "folded FA prep requires single-token Q8_0 flash KV".into(),
            ));
        }
        ATTENTION.run_attend_only(ctx, gpu, &plan, &io)?;
        return Ok(fused_epilogue);
    }
    execute_steps(gpu, ctx, &[Step::Attend { plan, io }])?;
    Ok(fused_epilogue)
}

/// Full-attention sublayer: norm → q(+gate)/k/v projection → Q/gate split,
/// Q/K norm, partial RoPE → KV write + flash attention → sigmoid output gate →
/// output projection added into `x`.
pub struct GatedAttentionOp<'a> {
    pub dims: HybridDims,
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
    pub row: AttentionRow<'a>,
    pub plain: &'a GpuTensor,
    pub x_rot: &'a GpuTensor,
    /// Interleaved Q and gate, `[n_heads, 2 * head_dim]`.
    pub q_gate: &'a GpuTensor,
    pub tap: Option<AttentionTap<'a>>,
    pub output: MixerOutput<'a>,
    /// Multimodal (t, h, w) RoPE; `None` rotates at `row.pos_buf`.
    pub mrope: Option<MropeRope<'a>>,
}

/// Three-section RoPE positions of one token (vision-language prompts).
#[derive(Clone, Copy)]
pub struct MropeRope<'a> {
    /// `[3]` i32 (t, h, w) rotary phases, already offset by KV compaction.
    pub positions: &'a DeviceBuffer,
    /// Frequency counts of the t, h and w sections.
    pub section: [usize; 3],
}

impl GatedAttentionOp<'_> {
    pub fn project(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<(), DispatchError> {
        let (wq, wk, wv) = (&self.wq, &self.wk, &self.wv);
        let (q, k, v) = (self.q_gate, self.row.k, self.row.v);
        if self.prerotated_input {
            return gpu
                .fused_qkv_hfq4g256(
                    wq.buf, wk.buf, wv.buf, self.x_rot, q, k, v, wq.m, wk.m, wv.m, wq.k,
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
            [(*wq, q), (*wk, k), (*wv, v)],
        )
    }

    /// Q/K preparation, KV write and attention. Returns whether the output
    /// gate (and the output projection's rotation) ran inside the epilogue.
    pub fn attend(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<bool, DispatchError> {
        let dims = &self.dims;
        let row = &self.row;
        let fused_prep =
            fused_attention_prep(gpu, dims) && self.tap.is_none() && self.mrope.is_none();
        // Fold the Q8_0 KV write into the prep epilogue. Requires the
        // non-compact decode route: pos_buf must hold the physical position
        // when the folded writer reads it.
        let kv_write_fold =
            if fused_prep && self.kv.compact_offset == 0 && fused_attention_kv_write(gpu, dims) {
                kv_write_fold_route_supported(&self.kv.tier_plan(gpu, row.position)?)
            } else {
                false
            };
        if !fused_prep {
            gpu.deinterleave_f32(self.q_gate, row.q, row.gate, dims.n_heads, dims.head_dim)
                .map_err(hip)?;
            gpu.rmsnorm_batched(
                row.q,
                self.q_norm,
                row.q,
                dims.n_heads,
                dims.head_dim,
                dims.norm_eps,
            )
            .map_err(hip)?;
            gpu.rmsnorm_batched(
                row.k,
                self.k_norm,
                row.k,
                dims.n_kv_heads,
                dims.head_dim,
                dims.norm_eps,
            )
            .map_err(hip)?;
        }
        if let Some(tap) = self.tap {
            tap(gpu)?;
        }
        // RoPE uses the logical position; the KV slot stays physical.
        let compacted = self.kv.compact_offset > 0;
        if compacted {
            let logical = (row.position + self.kv.compact_offset) as i32;
            gpu.memcpy_htod_auto(row.pos_buf, &logical.to_ne_bytes())
                .map_err(hip)?;
        }
        if let Some(mrope) = self.mrope {
            gpu.rope_mrope_halfsplit_f32(
                row.q,
                row.k,
                mrope.positions,
                dims.n_heads,
                dims.n_kv_heads,
                dims.head_dim,
                dims.n_rot,
                dims.rope_theta,
                mrope.section,
            )
        } else if fused_prep && kv_write_fold {
            gpu.qwen35_fa_prep_kvwrite_gfx1100(
                self.q_gate,
                row.q,
                row.gate,
                row.k,
                row.v,
                self.kv.k_cache,
                self.kv.v_cache,
                self.q_norm,
                self.k_norm,
                row.pos_buf,
                dims.norm_eps,
                dims.rope_theta,
                dims.n_heads,
                dims.n_kv_heads,
            )
        } else if fused_prep {
            gpu.qwen35_fa_prep_gfx1100(
                self.q_gate,
                row.q,
                row.gate,
                row.k,
                self.q_norm,
                self.k_norm,
                row.pos_buf,
                dims.norm_eps,
                dims.rope_theta,
                dims.n_heads,
                dims.n_kv_heads,
            )
        } else {
            gpu.rope_partial_interleaved_f32(
                row.q,
                row.k,
                row.pos_buf,
                dims.n_heads,
                dims.n_kv_heads,
                dims.head_dim,
                dims.n_rot,
                dims.rope_theta,
            )
        }
        .map_err(hip)?;
        if compacted {
            let physical = row.position as i32;
            gpu.memcpy_htod_auto(row.pos_buf, &physical.to_ne_bytes())
                .map_err(hip)?;
        }
        let fused_epilogue = kv_attention(gpu, ctx, dims, &self.kv, row, &self.wo, kv_write_fold)?;
        if !fused_epilogue {
            gpu.sigmoid_mul_f32(row.out, row.gate).map_err(hip)?;
        }
        Ok(fused_epilogue)
    }

    /// Output projection accumulated into the residual stream; `prerotated`
    /// is [`Self::attend`]'s result.
    pub fn output(
        &self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        prerotated: bool,
    ) -> Result<(), DispatchError> {
        let input = if prerotated {
            GemvInput::Prerotated(self.row.out)
        } else {
            GemvInput::Raw(self.row.out)
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
    if let MixerOutput::Write(out) = op.output {
        return attention_shard(gpu, ctx, op, out);
    }
    op.project(gpu, ctx)?;
    let prerotated = op.attend(gpu, ctx)?;
    op.output(gpu, ctx, prerotated)
}

/// [`MixerOutput::Write`]: the dense tensor-parallel shard body over this
/// rank's query/KV heads, its output projection written to `out`.
fn attention_shard(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &GatedAttentionOp<'_>,
    out: &GpuTensor,
) -> Result<(), DispatchError> {
    if op.kv.compact_offset > 0 || op.prerotated_input {
        return Err(DispatchError::Hip(
            "gated attention shard: compacted KV and a prerotated input are not admitted".into(),
        ));
    }
    let (dims, row) = (&op.dims, &op.row);
    normed_projection(
        gpu,
        ctx,
        op.x,
        op.attn_norm,
        op.plain,
        op.x_rot,
        dims.norm_eps,
        [(op.wq, op.q_gate), (op.wk, row.k), (op.wv, row.v)],
    )?;
    gpu.deinterleave_f32(op.q_gate, row.q, row.gate, dims.n_heads, dims.head_dim)
        .map_err(hip)?;
    let eps = dims.norm_eps;
    gpu.rmsnorm_batched(row.q, op.q_norm, row.q, dims.n_heads, dims.head_dim, eps)
        .map_err(hip)?;
    gpu.rmsnorm_batched(row.k, op.k_norm, row.k, dims.n_kv_heads, dims.head_dim, eps)
        .map_err(hip)?;
    if let Some(tap) = op.tap {
        tap(gpu)?;
    }
    gpu.rope_partial_interleaved_f32(
        row.q,
        row.k,
        row.pos_buf,
        dims.n_heads,
        dims.n_kv_heads,
        dims.head_dim,
        dims.n_rot,
        dims.rope_theta,
    )
    .map_err(hip)?;
    let fused_epilogue = kv_attention(gpu, ctx, dims, &op.kv, row, &op.wo, false)?;
    if !fused_epilogue {
        gpu.sigmoid_mul_f32(row.out, row.gate).map_err(hip)?;
    }
    let input = if fused_epilogue {
        GemvInput::Prerotated(row.out)
    } else {
        GemvInput::Raw(row.out)
    };
    execute_steps(
        gpu,
        ctx,
        &[Step::Gemv {
            w: &op.wo,
            input,
            out,
        }],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn campaign_4b() -> HybridDims {
        HybridDims {
            dim: 2_560,
            n_layers: 32,
            n_heads: 16,
            n_kv_heads: 4,
            head_dim: 256,
            n_rot: 64,
            rope_theta: 10_000_000.0,
            norm_eps: 1e-6,
            linear_key_heads: 16,
            linear_value_heads: 32,
            linear_key_dim: 128,
            linear_value_dim: 128,
            conv_kernel: 4,
            num_experts: 0,
        }
    }

    #[test]
    fn gfx1100_campaign_default_is_exact_4b_geometry() {
        let dims = campaign_4b();
        assert!(dims.is_gfx1100_4b_campaign_shape("gfx1100"));
        for arch in ["gfx1101", "gfx1102", "gfx1151", "gfx1201", "gfx942"] {
            assert!(!dims.is_gfx1100_4b_campaign_shape(arch), "{arch}");
            assert!(
                !conv_scalar_prep_requested(arch, &dims, None, false),
                "{arch}"
            );
        }
        let mutations: [fn(&mut HybridDims); 14] = [
            |d| d.dim = 4_096, // 9B: same 16Q/4K is not evidence.
            |d| {
                // 0.8B fixture geometry
                d.dim = 1_024;
                d.n_layers = 24;
                d.n_heads = 8;
                d.n_kv_heads = 2;
                d.linear_value_heads = 16;
            },
            |d| {
                // 2B fixture geometry (existing A3B gates stay separate).
                d.dim = 2_048;
                d.n_layers = 24;
                d.n_heads = 8;
                d.n_kv_heads = 2;
                d.linear_value_heads = 16;
            },
            |d| d.n_heads = 8,
            |d| d.n_kv_heads = 2,
            |d| d.head_dim = 128,
            |d| d.n_rot = 128,
            |d| d.linear_key_heads = 8,
            |d| d.linear_key_dim = 64,
            |d| d.linear_value_heads = 16,
            |d| d.linear_value_heads = 31,
            |d| d.linear_value_dim = 64,
            |d| d.conv_kernel = 3,
            |d| d.num_experts = 8,
        ];
        for mutate in mutations {
            let mut other = dims;
            mutate(&mut other);
            assert!(!other.is_gfx1100_4b_campaign_shape("gfx1100"));
            assert!(!conv_scalar_prep_requested("gfx1100", &other, None, false));
        }
    }

    #[test]
    fn conv_scalar_prep_keeps_27b_and_explicit_experiments() {
        let dims = campaign_4b();
        for mode in [None, Some("unrecognized"), Some("1")] {
            assert!(conv_scalar_prep_requested("gfx1100", &dims, mode, false));
        }
        assert!(!conv_scalar_prep_requested(
            "gfx1100",
            &dims,
            Some("0"),
            false
        ));
        let dims27 = HybridDims {
            dim: 5_120,
            n_layers: 64,
            n_heads: 24,
            linear_value_heads: 48,
            ..dims
        };
        assert!(dims27.is_dense_27b_shape());
        assert!(conv_scalar_prep_requested("gfx1100", &dims27, None, false));
        assert!(!dims27.is_gfx1100_4b_campaign_shape("gfx1100"));
        let dims9 = HybridDims { dim: 4_096, ..dims };
        assert!(!conv_scalar_prep_requested("gfx1100", &dims9, None, false));
        assert!(conv_scalar_prep_requested(
            "gfx1100",
            &dims9,
            Some("1"),
            false
        ));
        for arch in ["gfx1101", "gfx1151", "gfx1201"] {
            assert!(!conv_scalar_prep_requested(arch, &dims27, None, false));
            assert!(!conv_scalar_prep_requested(arch, &dims9, Some("1"), false));
        }
        assert!(conv_scalar_prep_requested("gfx1101", &dims9, None, true));
        assert!(conv_scalar_prep_requested(
            "gfx1101",
            &dims9,
            Some("1"),
            true
        ));
        assert!(!conv_scalar_prep_requested(
            "gfx1101",
            &dims9,
            Some("0"),
            true
        ));
    }

    #[test]
    fn dense_27b_shape_is_exact() {
        let dims27 = HybridDims {
            dim: 5_120,
            n_layers: 64,
            n_heads: 24,
            linear_value_heads: 48,
            ..campaign_4b()
        };
        assert!(dims27.is_dense_27b_shape());
        for other in [
            HybridDims {
                linear_value_heads: 47,
                ..dims27
            },
            HybridDims {
                num_experts: 8,
                ..dims27
            },
            HybridDims {
                dim: 2_048,
                ..dims27
            },
        ] {
            assert!(!other.is_dense_27b_shape());
        }
    }

    #[test]
    fn fa_kvwrite_fold_requires_single_token_q8_flash_plan() {
        let plan = KvTierPlan {
            write_key: KernelKey::KvWriteQ8_0,
            attend_key: KernelKey::AttnFlashQ8_0,
            v_mode_bits: 8,
            uses_givens: false,
            batch_size: 1,
            window: 0,
            gemma_tiles: false,
        };
        assert!(kv_write_fold_route_supported(&plan));
        assert!(!kv_write_fold_route_supported(&KvTierPlan {
            batch_size: 2,
            ..plan
        }));
        for v_mode_bits in [2, 3, 4] {
            assert!(
                !kv_write_fold_route_supported(&KvTierPlan {
                    v_mode_bits,
                    ..plan
                }),
                "Lloyd V bits={v_mode_bits}"
            );
        }
        for (write_key, attend_key) in [
            (KernelKey::KvWriteQ8_0, KernelKey::AttnQ8_0Kv),
            (KernelKey::KvWriteQ8_0, KernelKey::AttnFlashQ8_0Windowed),
            (KernelKey::KvWriteBf16, KernelKey::AttnFlashBf16),
            (KernelKey::KvWriteF32, KernelKey::AttnFlash),
            (KernelKey::KvWriteFp8E4m3, KernelKey::AttnFlashFp8E4m3),
            (KernelKey::KvWriteAsym3Fwht, KernelKey::AttnFlashAsym3Fwht),
            (
                KernelKey::KvWriteQ8_0Batched,
                KernelKey::AttnQ8_0KvBatchedMasked,
            ),
            (KernelKey::KvWriteBf16, KernelKey::AttnFlashQ8_0),
            (KernelKey::KvWriteQ8_0, KernelKey::AttnFlashBf16),
        ] {
            assert!(
                !kv_write_fold_route_supported(&KvTierPlan {
                    write_key,
                    attend_key,
                    ..plan
                }),
                "{write_key:?}/{attend_key:?}"
            );
        }
    }

    #[test]
    fn gfx1201_fa_epilogue_admits_q8_and_fp8_tile() {
        assert!(attention_epilogue_route_supported(true, true, false, false));
        assert!(attention_epilogue_route_supported(true, false, false, true));
        assert!(!attention_epilogue_route_supported(
            true, false, true, false
        ));
        assert!(attention_epilogue_route_supported(
            false, false, true, false
        ));
        assert!(!attention_epilogue_route_supported(
            true, false, false, false
        ));
    }
}
