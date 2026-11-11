// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Dense gated MLP: `out = down(act(gate(n)) * up(n))` with `n = pre_norm(x)`
//! (or `n = x` without a pre-norm), then the output epilogue.
//!
//! One op for the dense gated FFN forms. The stage order is fixed: optional
//! pre-norm, gate/up, activation, down, output. Which kernels run each stage is
//! the declared [`MlpLadder`]: every arm is the body of the op it replaced,
//! moved verbatim, so a family keeps its launches. Admission (ladder ×
//! pre-norm × activation × storage × output) is checked before the first
//! launch; any other combination is refused.
//!
//! `rows == 1` is single-token decode, `rows > 1` a batch of consecutive
//! positions (prefill or speculative verify) laid out row-major.

use super::layer_ops::{
    checked_mul, project_weight, project_weights, require_tensor, require_weight, rotation_basis,
    view,
};
use super::sandwich::{
    calibrating, fused_projection, gemm_rows_wide, gemv, hip, q8_fused_prefill, rms_norm,
    SandwichRoute, SandwichStream,
};
use crate::context::DispatchCtx;
use crate::families::gemv::WeightRef;
use crate::pipeline::RowsRoute;
use crate::types::{DispatchError, KernelKey};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Gated MLP activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    /// `gelu_pytorch_tanh(gate) * up`.
    GeluTanh,
    /// `silu(gate) * up`.
    Silu,
}

/// Storage boundaries of the source equation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlpStorage {
    /// F32 between every stage.
    F32,
    /// BF16 storage after the gate projection, the up projection, the
    /// activation, the product and the down projection.
    Bf16Source,
    /// [`Self::Bf16Source`] for a tensor-parallel rank whose output is a
    /// partial sum: every boundary but the down projection. The rank hands
    /// its F32 partial to the reduction, so the sum is rounded once (by its
    /// consumer), as the unsharded down projection is; rounding each partial
    /// would round twice. Set only by a TP rank's sharded FFN.
    Bf16SourcePartial,
}

/// RMS pre-norm of `x` into `normed` (F32 `[rows, hidden]`).
#[derive(Clone, Copy)]
pub struct RmsNorm<'a> {
    pub weight: &'a GpuTensor,
    pub eps: f32,
    pub normed: &'a GpuTensor,
}

/// Where the down projection lands.
#[derive(Clone, Copy)]
pub enum MlpOutput<'a> {
    /// `out = mlp(x)`; the caller combines (Qwen4 HyperWrite, Gemma's parallel
    /// dense + routed-expert block, a tensor-parallel partial sum).
    Write(&'a GpuTensor),
    /// `x += mlp(x)` through the ladder's residual-accumulating down kernel.
    Residual,
    /// Sandwich epilogue `x = residual + norm(out)` with `out = mlp(x)`. The op
    /// saves `x` into `residual` first; the post-norm uses the pre-norm's
    /// `normed` as scratch.
    PostNorm {
        norm: &'a GpuTensor,
        eps: f32,
        out: &'a GpuTensor,
        residual: &'a GpuTensor,
    },
}

/// The kernel ladder the stages run on. The ladders give different bits for
/// the same weight format, so a family names the one it was validated on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlpLadder {
    /// The shared stateful-op projections (`project_weights`: shared F16
    /// rotation, few-row verify GEMVs, gfx942 BF16 MFMA). Qwen4.
    Trunk,
    /// The sandwich ladder: one-row fused MQ4 norm + gate/up, `gemm_rows` for
    /// rows, fused post-norm residual on one row; greedy-EAGLE strict kernels
    /// inside. Gemma 4.
    SandwichWide,
    /// The sandwich steps ladder of [`super::sandwich::SandwichMlpOp`]
    /// (Glimmer). Named here only so a split rule can declare it; a
    /// [`GatedMlpOp`] refuses it.
    SandwichSteps,
    /// The hybrid-decoder ladder (Qwen3.5/3.6/3.8 dense layers).
    /// One row: norm + gate/up through the fusion table (`normed_projection`),
    /// then the fused SiLU·mul + rotation and the SwiGLU-residual GEMV
    /// (`Residual`), or SiLU·mul + a plain GEMV (`Write`, the dense-TP
    /// partial). Rows: the batched producer/GEMM ladder
    /// ([`execute_product_rows`]), whose `Write` is the TP `Partial` epilogue.
    #[cfg(feature = "deltanet")]
    Product,
}

impl MlpLadder {
    /// Whether the ladder runs unchanged on an intermediate shard of its
    /// weights (gate/up rows, down columns) and writes a partial sum, which
    /// its consumer reduces: every ladder through its `Write` form. The split
    /// rule does not record the output form, so a TP route must bind the
    /// `Write` (`Product`: never its `Residual`; the sandwich ladders: never
    /// the `PostNorm`, whose norm cannot take a partial sum, but the reduced
    /// output's separate epilogue). A new ladder is refused until it opts in
    /// here.
    pub fn tp_shardable(self) -> bool {
        match self {
            Self::Trunk | Self::SandwichWide | Self::SandwichSteps => true,
            #[cfg(feature = "deltanet")]
            Self::Product => true,
        }
    }
}

/// Fixed scratch of one op.
#[derive(Clone, Copy)]
pub struct GatedMlpScratch<'a> {
    /// F32 `[rows, intermediate]` each. `gate` and `up` must be distinct
    /// buffers, and `act` must not alias `up` (the activation writes `act`
    /// before the product reads `up`); `act` may alias `gate`.
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub act: &'a GpuTensor,
    /// FWHT basis scratch. `Trunk`: `[rows, max(hidden, intermediate)]`, read
    /// only for FWHT payloads. `SandwichWide`: `[hidden]` on one row,
    /// `[rows, max(hidden, intermediate)]` on rows.
    pub x_rot: &'a GpuTensor,
}

/// One dense gated MLP over `rows` rows; see the module docs.
#[derive(Clone, Copy)]
pub struct GatedMlpOp<'a> {
    pub rows: usize,
    pub hidden: usize,
    pub intermediate: usize,
    /// F32 `[rows, hidden]`: the FFN input, or the residual stream that a
    /// `PostNorm` output updates in place.
    pub x: &'a GpuTensor,
    /// `None`: `x` is the gate/up input.
    pub pre_norm: Option<RmsNorm<'a>>,
    pub gate: WeightRef<'a>,
    pub up: WeightRef<'a>,
    pub down: WeightRef<'a>,
    pub activation: Activation,
    pub storage: MlpStorage,
    pub output: MlpOutput<'a>,
    pub ladder: MlpLadder,
    pub scratch: GatedMlpScratch<'a>,
}

impl GatedMlpOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        self.validate_layout()
    }

    fn admit(&self) -> Result<(), DispatchError> {
        let admitted = match self.ladder {
            MlpLadder::Trunk => {
                self.pre_norm.is_none()
                    && self.activation == Activation::Silu
                    && matches!(
                        self.storage,
                        MlpStorage::Bf16Source | MlpStorage::Bf16SourcePartial
                    )
                    && matches!(self.output, MlpOutput::Write(_))
            }
            MlpLadder::SandwichWide => {
                self.pre_norm.is_some()
                    && self.activation == Activation::GeluTanh
                    && self.storage == MlpStorage::F32
                    && !matches!(self.output, MlpOutput::Residual)
            }
            MlpLadder::SandwichSteps => false,
            #[cfg(feature = "deltanet")]
            MlpLadder::Product => {
                self.rows >= 1
                    && self.pre_norm.is_some()
                    && self.activation == Activation::Silu
                    && self.storage == MlpStorage::F32
                    && matches!(self.output, MlpOutput::Write(_) | MlpOutput::Residual)
            }
        };
        if admitted {
            return Ok(());
        }
        let output = match self.output {
            MlpOutput::Write(_) => "write",
            MlpOutput::Residual => "residual",
            MlpOutput::PostNorm { .. } => "post-norm",
        };
        Err(DispatchError::Hip(format!(
            "gated MLP: ladder {:?} does not admit rows={} pre_norm={} activation={:?} storage={:?} output={output}",
            self.ladder,
            self.rows,
            self.pre_norm.is_some(),
            self.activation,
            self.storage,
        )))
    }

    pub(super) fn validate_layout(&self) -> Result<(), DispatchError> {
        self.validate_with_act_rows(self.rows)
    }

    /// [`Self::validate_layout`] with the `act` scratch required to hold
    /// `act_rows` rows; every other tensor is checked over `rows`.
    fn validate_with_act_rows(&self, act_rows: usize) -> Result<(), DispatchError> {
        let (rows, hidden, inner_width) = (self.rows, self.hidden, self.intermediate);
        if rows == 0 || hidden == 0 || inner_width == 0 {
            return Err(DispatchError::Hip("gated MLP has empty geometry".into()));
        }
        self.admit()?;
        for (weight, m, k, label) in [
            (&self.gate, inner_width, hidden, "gated MLP gate"),
            (&self.up, inner_width, hidden, "gated MLP up"),
            (&self.down, hidden, inner_width, "gated MLP down"),
        ] {
            match self.ladder {
                MlpLadder::Trunk => require_weight(weight, m, k, label)?,
                // Every format the GEMV family serves; formats without a
                // batched kernel take one GEMV per row.
                _ if weight.m != m || weight.k != k => {
                    return Err(DispatchError::Hip(format!(
                        "{label} has incompatible shape"
                    )));
                }
                _ => {}
            }
        }
        let wide = checked_mul(rows, hidden, "gated MLP hidden rows")?;
        let inner = checked_mul(rows, inner_width, "gated MLP intermediate rows")?;
        let sc = &self.scratch;
        require_tensor(self.x, wide, DType::F32, "gated MLP input")?;
        if let Some(norm) = &self.pre_norm {
            require_tensor(norm.weight, hidden, DType::F32, "gated MLP pre-norm weight")?;
            require_tensor(norm.normed, wide, DType::F32, "gated MLP normed")?;
        }
        require_tensor(sc.gate, inner, DType::F32, "gated MLP gate scratch")?;
        require_tensor(sc.up, inner, DType::F32, "gated MLP up scratch")?;
        let act = checked_mul(act_rows, inner_width, "gated MLP act rows")?;
        require_tensor(sc.act, act, DType::F32, "gated MLP act scratch")?;
        match self.output {
            MlpOutput::Write(out) => require_tensor(out, wide, DType::F32, "gated MLP output")?,
            MlpOutput::Residual => {}
            MlpOutput::PostNorm {
                norm,
                out,
                residual,
                ..
            } => {
                require_tensor(norm, hidden, DType::F32, "gated MLP post-norm weight")?;
                require_tensor(out, wide, DType::F32, "gated MLP output")?;
                require_tensor(residual, wide, DType::F32, "gated MLP residual")?;
            }
        }
        let widest = checked_mul(rows, hidden.max(inner_width), "gated MLP rotation")?;
        let rotation = match self.ladder {
            MlpLadder::Trunk => [&self.gate, &self.up, &self.down]
                .iter()
                .any(|weight| rotation_basis(weight.dtype).is_some())
                .then_some(widest),
            MlpLadder::SandwichWide if rows == 1 => Some(hidden),
            MlpLadder::SandwichWide => Some(widest),
            MlpLadder::SandwichSteps => None,
            // The pre-norm writes its (rotated) gate/up input here.
            #[cfg(feature = "deltanet")]
            MlpLadder::Product => Some(rows * hidden),
        };
        if let Some(elements) = rotation {
            require_tensor(sc.x_rot, elements, DType::F32, "gated MLP rotation")?;
        }
        Ok(())
    }
}

/// Validate `op`, then run its ladder.
pub fn execute_gated_mlp(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &GatedMlpOp<'_>,
) -> Result<(), DispatchError> {
    op.validate_layout()?;
    match (op.ladder, op.pre_norm, op.output) {
        (MlpLadder::Trunk, None, MlpOutput::Write(out)) => trunk(gpu, op, out),
        (MlpLadder::SandwichWide, Some(norm), _) => sandwich_wide(gpu, ctx, op, &norm),
        #[cfg(feature = "deltanet")]
        (MlpLadder::Product, Some(_), _) if op.rows == 1 => {
            op.project(gpu, ctx)?;
            op.down(gpu, ctx)
        }
        #[cfg(feature = "deltanet")]
        (MlpLadder::Product, Some(_), _) => Err(DispatchError::Hip(
            "gated MLP: Product rows run through execute_product_rows".into(),
        )),
        _ => unreachable!("validate_layout admitted an unknown ladder combination"),
    }
}

/// `Trunk` with `Bf16Source` storage, each step on exact-length views, no
/// allocation:
///
/// 1. gate and up projections of `x` (`project_weights`);
/// 2. BF16 round trip of gate, then of up;
/// 3. `act = silu(gate)`, BF16 round trip;
/// 4. `act = act * up`, BF16 round trip;
/// 5. down projection of `act` into `out`, BF16 round trip (none for
///    `Bf16SourcePartial`: the F32 partial is reduced first).
///
/// The element-wise kernels read each element before writing it, so `act`
/// may alias `gate`.
fn trunk(gpu: &mut Gpu, op: &GatedMlpOp<'_>, out: &GpuTensor) -> Result<(), DispatchError> {
    let wide = checked_mul(op.rows, op.hidden, "gated MLP hidden rows")?;
    let inner = checked_mul(op.rows, op.intermediate, "gated MLP intermediate rows")?;
    let sc = &op.scratch;
    let gate = view(sc.gate, 0, inner);
    let up = view(sc.up, 0, inner);
    let act = view(sc.act, 0, inner);
    let out = view(out, 0, wide);
    let rotation = Some(sc.x_rot);
    project_weights(
        gpu,
        op.x,
        op.rows,
        rotation,
        &[(&op.gate, &gate), (&op.up, &up)],
    )?;
    gpu.bf16_round_trip_f32(&gate).map_err(hip)?;
    gpu.bf16_round_trip_f32(&up).map_err(hip)?;
    gpu.silu_f32(&gate, &act).map_err(hip)?;
    gpu.bf16_round_trip_f32(&act).map_err(hip)?;
    gpu.mul_f32(&act, &up, &act).map_err(hip)?;
    gpu.bf16_round_trip_f32(&act).map_err(hip)?;
    project_weight(gpu, &op.down, &act, &out, op.rows, rotation)?;
    if op.storage == MlpStorage::Bf16SourcePartial {
        return Ok(());
    }
    gpu.bf16_round_trip_f32(&out).map_err(hip)
}

/// `SandwichWide`: the former Gemma 4 sandwich MLP, decode and batched arms.
fn sandwich_wide(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &GatedMlpOp<'_>,
    norm: &RmsNorm<'_>,
) -> Result<(), DispatchError> {
    let (rows, hidden, sc) = (op.rows, op.hidden, &op.scratch);
    let (out, post) = match op.output {
        MlpOutput::Write(out) => (out, None),
        MlpOutput::Residual => unreachable!("SandwichWide refuses the residual output"),
        MlpOutput::PostNorm {
            norm: post_norm,
            eps,
            out,
            residual,
        } => {
            let stream = SandwichStream {
                rows,
                hidden,
                eps,
                post_eps: eps,
                x: op.x,
                residual,
                normed: norm.normed,
                out: None,
                route: if rows == 1 {
                    RowsRoute::Decode
                } else {
                    RowsRoute::Prefill
                },
                kernels: SandwichRoute::Wide,
                bf16: None,
            };
            stream.save_residual(gpu)?;
            (out, Some((stream, post_norm)))
        }
    };
    if rows > 1 {
        rms_norm(gpu, rows, hidden, norm.eps, op.x, norm.weight, norm.normed)?;
        if q8_fused_prefill(gpu, &[&op.gate, &op.up]) {
            fused_projection(
                gpu,
                ctx,
                KernelKey::FusedGateUpQ8_0,
                &[&op.gate, &op.up],
                norm.normed,
                &[sc.gate, sc.up],
                rows,
            )?;
        } else {
            gemm_rows_wide(gpu, ctx, &op.gate, norm.normed, sc.gate, sc.x_rot, rows)?;
            gemm_rows_wide(gpu, ctx, &op.up, norm.normed, sc.up, sc.x_rot, rows)?;
        }
    } else if op.gate.dtype == DType::MQ4G256 && op.up.dtype == DType::MQ4G256 && !calibrating(gpu)
    {
        gpu.fused_rmsnorm_rotate_mq(op.x, norm.weight, sc.x_rot, hidden, norm.eps)
            .map_err(hip)?;
        gpu.fused_gate_up_hfq4g256(
            op.gate.buf,
            op.up.buf,
            sc.x_rot,
            sc.gate,
            sc.up,
            op.gate.m,
            op.up.m,
            op.gate.k,
        )
        .map_err(hip)?;
    } else {
        rms_norm(gpu, rows, hidden, norm.eps, op.x, norm.weight, norm.normed)?;
        gemv(gpu, ctx, &op.gate, norm.normed, sc.gate)?;
        gemv(gpu, ctx, &op.up, norm.normed, sc.up)?;
    }
    gpu.gelu_tanh_f32(sc.gate, sc.act, rows * op.intermediate)
        .map_err(hip)?;
    gpu.mul_f32(sc.act, sc.up, sc.act).map_err(hip)?;
    if rows == 1 {
        gemv(gpu, ctx, &op.down, sc.act, out)?;
    } else {
        gemm_rows_wide(gpu, ctx, &op.down, sc.act, out, sc.x_rot, rows)?;
    }
    match post {
        Some((stream, post_norm)) => stream.post_norm_residual(gpu, out, Some(post_norm)),
        None => Ok(()),
    }
}

/// The `Product` ladder's two halves. The Qwen3.5 decode super-op
/// (`ForwardBindings`) issues them as separate opcodes; every other caller
/// runs the whole op through [`execute_gated_mlp`].
#[cfg(feature = "deltanet")]
impl GatedMlpOp<'_> {
    /// Pre-norm (plus the gate weight's input basis) and the gate/up GEMVs,
    /// which the fusion table rewrites into one fused gate/up kernel.
    pub fn project(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<(), DispatchError> {
        self.require_product()?;
        let norm = self.pre_norm.expect("Product admits only a pre-norm");
        super::hybrid::normed_projection(
            gpu,
            ctx,
            self.x,
            norm.weight,
            norm.normed,
            self.scratch.x_rot,
            norm.eps,
            [(self.gate, self.scratch.gate), (self.up, self.scratch.up)],
        )
    }

    /// SiLU·mul of the projected gate/up, then the down projection into the
    /// output.
    pub fn down(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<(), DispatchError> {
        self.require_product()?;
        let sc = &self.scratch;
        match self.output {
            MlpOutput::Residual => {
                swiglu_down_residual(gpu, ctx, &self.down, sc.gate, sc.up, sc.act, self.x)
            }
            MlpOutput::Write(_) => self.write_partial(gpu, ctx).map_err(hip),
            MlpOutput::PostNorm { .. } => unreachable!("Product refuses the post-norm output"),
        }
    }

    /// The `Write` down half: SiLU·mul, then a plain GEMV into the output
    /// (the dense-TP FFN partial). HIP errors keep their codes; dispatch
    /// errors become code 0, as the hand route's `map_err` did.
    pub fn write_partial(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> hip_bridge::HipResult<()> {
        let code0 = |e: DispatchError| hip_bridge::HipError::new(0, &e.to_string());
        self.require_product().map_err(code0)?;
        let MlpOutput::Write(out) = self.output else {
            return Err(code0(DispatchError::Hip(
                "gated MLP: write_partial needs a Write output".into(),
            )));
        };
        let sc = &self.scratch;
        gpu.silu_mul_f32(sc.gate, sc.up, sc.act)?;
        super::steps::execute_steps(
            gpu,
            ctx,
            &[super::steps::Step::Gemv {
                w: &self.down,
                input: super::steps::GemvInput::Raw(sc.act),
                out,
            }],
        )
        .map_err(code0)
    }

    fn require_product(&self) -> Result<(), DispatchError> {
        if self.ladder != MlpLadder::Product || self.rows != 1 {
            return Err(DispatchError::Hip(format!(
                "gated MLP: {:?} over {} rows has no separate project/down halves",
                self.ladder, self.rows
            )));
        }
        self.validate_layout()
    }
}

/// Per-call context of the `Product` ladder's batched route (prefill, verify,
/// TP prefill; any row count, one included): the chunk's batch scratch
/// planes and the route policy the batched producers and GEMMs read.
#[cfg(feature = "deltanet")]
#[derive(Clone, Copy)]
pub struct ProductRows<'a> {
    /// The op's `x`, gate/up/act scratch and normed/rotation planes must be
    /// this scratch's `x_batch`, `gate_ffn_batch`, `up_batch`,
    /// `ffn_hidden_batch` and `x_rot_batch`.
    pub batch: super::batched::BatchScratch<'a>,
    pub q8_wmma_arch: bool,
    pub arch_has_wmma: bool,
    pub fusion: super::batched::DflashFusionCtx,
    pub math: super::batched::DenseBatchMath,
    /// Mixer kind of the owning layer (`"LA"` / `"FA"`), for error messages.
    pub kind: &'static str,
}

/// Validate `op` (a `Product` op) against `rows`, then run the batched
/// ladder over `op.rows` rows: `Residual` adds into `x_batch`,
/// `Write(partial)` is the tensor-parallel partial epilogue. HIP errors keep
/// their codes. On a lean batch scratch the `act` plane holds only the
/// fallback rows: wider chunks run the fused producers, which never read it,
/// and the caller admits them (`lean_pbs_route`), so its capacity is not
/// checked against `op.rows`.
#[cfg(feature = "deltanet")]
pub fn execute_product_rows(
    gpu: &mut Gpu,
    op: &GatedMlpOp<'_>,
    rows: &ProductRows<'_>,
) -> hip_bridge::HipResult<()> {
    use super::batched::{execute_swiglu_ffn_batched, BatchEpilogue, FfnLayerView};
    let refuse = |msg: String| hip_bridge::HipError::new(0, &msg);
    let act_rows = if rows.batch.lean { 0 } else { op.rows };
    op.validate_with_act_rows(act_rows)
        .map_err(|e| refuse(e.to_string()))?;
    let norm = match (op.ladder, op.pre_norm) {
        (MlpLadder::Product, Some(norm)) => norm,
        _ => {
            return Err(refuse(format!(
                "gated MLP: the batched route needs a Product op, got {:?}",
                op.ladder
            )))
        }
    };
    let (b, sc) = (&rows.batch, &op.scratch);
    let same = |a: &GpuTensor, b: &GpuTensor| a.buf.as_ptr() == b.buf.as_ptr();
    if !(same(op.x, b.x_batch)
        && same(norm.normed, b.x_rot_batch)
        && same(sc.x_rot, b.x_rot_batch)
        && same(sc.gate, b.gate_ffn_batch)
        && same(sc.up, b.up_batch)
        && same(sc.act, b.ffn_hidden_batch))
    {
        return Err(refuse(
            "gated MLP: Product rows op is not bound to its batch scratch".into(),
        ));
    }
    let epilogue = match op.output {
        MlpOutput::Residual => BatchEpilogue::Residual,
        MlpOutput::Write(partial) => BatchEpilogue::Partial(partial),
        MlpOutput::PostNorm { .. } => unreachable!("Product refuses the post-norm output"),
    };
    let layer = FfnLayerView {
        ffn_norm: norm.weight,
        w_gate: op.gate,
        w_up: op.up,
        w_down: op.down,
        kind: rows.kind,
    };
    execute_swiglu_ffn_batched(
        gpu,
        &layer,
        norm.eps,
        b,
        op.rows,
        op.hidden,
        op.intermediate,
        rows.q8_wmma_arch,
        rows.arch_has_wmma,
        epilogue,
        rows.fusion,
        rows.math,
    )
}

#[cfg(feature = "deltanet")]
fn mq_rotation_scratch(gpu: &Gpu) -> Result<GpuTensor, DispatchError> {
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
/// rotation and run the SwiGLU-residual GEMV; HFQ weights run the residual
/// GEMV; every other format is a plain GEMV (with its own input rotation and
/// AWQ sidecar) plus an in-place add.
#[cfg(feature = "deltanet")]
#[allow(clippy::too_many_arguments)]
fn swiglu_down_residual(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w_down: &WeightRef<'_>,
    gate: &GpuTensor,
    up: &GpuTensor,
    hidden: &GpuTensor,
    x: &GpuTensor,
) -> Result<(), DispatchError> {
    use super::steps::{execute_steps, GemvInput, Step};
    use crate::families::gemv::{GemvFamily, GemvParams};
    use crate::types::{dtype_needs_rotation, GemvVariant};
    static GEMV: std::sync::LazyLock<GemvFamily> = std::sync::LazyLock::new(GemvFamily::new);

    super::reject_mq4g128v2(w_down.dtype, "swiglu_down_residual")?;
    // Calibration tap: down_proj input is `hidden` (post-SiLU), K = w_down.k.
    // For MQ4 this will be rotated inside; we capture PRE-rotation.
    gpu.maybe_capture_activation(w_down.buf, hidden, 1, w_down.k);
    let wr = WeightRef {
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        ..*w_down
    };
    match w_down.dtype {
        // Host-executed offload (`memory.offload_exec=cpu|passback`): the SiLU
        // stays on the GPU; the down projection and its residual accumulate are
        // one `Step::GemvResidual` over the host-mapped weight bytes, which the
        // generic seam runs on the CPU (or splits by rows for pass-back).
        _ if crate::host_mapped_cpu_capable(gpu, &wr) => {
            gpu.silu_mul_f32(gate, up, hidden).map_err(hip)?;
            // The CPU path divides the AWQ sidecar itself; the activation here
            // is the raw SiLU output the fused GPU arm would divide.
            let wr_cpu = WeightRef {
                awq_scale: w_down.awq_scale,
                ..wr
            };
            execute_steps(
                gpu,
                ctx,
                &[Step::GemvResidual {
                    w: &wr_cpu,
                    input: GemvInput::Raw(hidden),
                    residual: x,
                    out: x,
                }],
            )
        }
        DType::MQ4G256
        | DType::MQ4G256V2
        // qt=52: same FWHT input contract as qt=44; the residual GEMV is the
        // LUT twin (family branches on dtype).
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
        // GGUF-native: SiLU·mul into `hidden`, then the mmvq accumulate export
        // adds `W_down · hidden` into the residual stream (q8_1 producer + one
        // GEMV launch, no temporary).
        dtype if dtype.is_ggml() => {
            gpu.silu_mul_f32(gate, up, hidden).map_err(hip)?;
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
        dtype => {
            gpu.silu_mul_f32(gate, up, hidden).map_err(hip)?;
            // Two taps, as the runtime twin's residual-then-GEMV fallback
            // (`weight_gemv_residual` + `weight_gemv`) takes: calibration
            // weighting must not depend on which decode path ran.
            gpu.maybe_capture_activation(w_down.buf, hidden, 1, w_down.k);
            gpu.maybe_capture_activation(w_down.buf, hidden, 1, w_down.k);
            // Rotated formats (MQ8, G128, Givens, FP4, ...) keep their rotation
            // and AWQ sidecar so `run_auto` rotates the input first; it also
            // prepares that rotation's own sign tables and scratch.
            let w = if dtype_needs_rotation(dtype) { w_down } else { &wr };
            let tmp = gpu.alloc_tensor(&[w_down.m], DType::F32).map_err(hip)?;
            let run = GEMV.run_auto(ctx, gpu, w, hidden, &tmp);
            let add = run.and_then(|_| gpu.add_inplace_f32(x, &tmp).map_err(hip));
            let free = gpu.free_tensor(tmp).map_err(hip);
            add.and(free)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(elements: usize, dtype: DType) -> GpuTensor {
        let mut tensor = GpuTensor::null_for_test();
        tensor.shape = vec![elements];
        tensor.dtype = dtype;
        tensor
    }

    fn weight(buf: &GpuTensor, dtype: DType, m: usize, k: usize) -> WeightRef<'_> {
        WeightRef {
            buf,
            dtype,
            m,
            k,
            row_stride: k,
            rotation: None,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        }
    }

    const ROWS: usize = 3;
    const HIDDEN: usize = 16;
    const INNER: usize = 8;

    /// Weights `[gate, up, down]`, then `x`, `gate`, `up`, `out`, the norm
    /// weight and `normed`, all exactly sized for `ROWS`.
    fn fixture() -> [GpuTensor; 9] {
        [
            tensor(HIDDEN * INNER, DType::BF16),
            tensor(HIDDEN * INNER, DType::BF16),
            tensor(HIDDEN * INNER, DType::BF16),
            tensor(ROWS * HIDDEN, DType::F32),
            tensor(ROWS * INNER, DType::F32),
            tensor(ROWS * INNER, DType::F32),
            tensor(ROWS * HIDDEN, DType::F32),
            tensor(HIDDEN, DType::F32),
            tensor(ROWS * HIDDEN, DType::F32),
        ]
    }

    /// The Qwen4 trunk declaration: no pre-norm, SiLU, BF16 storage,
    /// `act = gate`.
    fn trunk<'a>(t: &'a [GpuTensor; 9], x_rot: &'a GpuTensor) -> GatedMlpOp<'a> {
        GatedMlpOp {
            rows: ROWS,
            hidden: HIDDEN,
            intermediate: INNER,
            x: &t[3],
            pre_norm: None,
            gate: weight(&t[0], DType::BF16, INNER, HIDDEN),
            up: weight(&t[1], DType::BF16, INNER, HIDDEN),
            down: weight(&t[2], DType::BF16, HIDDEN, INNER),
            activation: Activation::Silu,
            storage: MlpStorage::Bf16Source,
            output: MlpOutput::Write(&t[6]),
            ladder: MlpLadder::Trunk,
            scratch: GatedMlpScratch {
                gate: &t[4],
                up: &t[5],
                act: &t[4],
                x_rot,
            },
        }
    }

    /// The Gemma 4 dense-layer declaration.
    fn sandwich<'a>(t: &'a [GpuTensor; 9], x_rot: &'a GpuTensor) -> GatedMlpOp<'a> {
        GatedMlpOp {
            pre_norm: Some(RmsNorm {
                weight: &t[7],
                eps: 1e-6,
                normed: &t[8],
            }),
            activation: Activation::GeluTanh,
            storage: MlpStorage::F32,
            output: MlpOutput::PostNorm {
                norm: &t[7],
                eps: 1e-6,
                out: &t[6],
                residual: &t[8],
            },
            ladder: MlpLadder::SandwichWide,
            ..trunk(t, x_rot)
        }
    }

    fn refusal(op: &GatedMlpOp<'_>) -> String {
        op.validate_layout()
            .expect_err("gated MLP op must be refused")
            .to_string()
    }

    #[test]
    fn ladder_admission() {
        let t = fixture();
        let x_rot = tensor(ROWS * HIDDEN, DType::F32);
        assert!(trunk(&t, &x_rot).validate_layout().is_ok());
        assert!(sandwich(&t, &x_rot).validate_layout().is_ok());
        let parallel_dense = GatedMlpOp {
            output: MlpOutput::Write(&t[6]),
            ..sandwich(&t, &x_rot)
        };
        assert!(parallel_dense.validate_layout().is_ok());

        let norm = sandwich(&t, &x_rot).pre_norm;
        let post_norm = sandwich(&t, &x_rot).output;
        let refused = [
            GatedMlpOp {
                pre_norm: norm,
                ..trunk(&t, &x_rot)
            },
            GatedMlpOp {
                activation: Activation::GeluTanh,
                ..trunk(&t, &x_rot)
            },
            GatedMlpOp {
                storage: MlpStorage::F32,
                ..trunk(&t, &x_rot)
            },
            GatedMlpOp {
                output: post_norm,
                ..trunk(&t, &x_rot)
            },
            GatedMlpOp {
                pre_norm: None,
                ..sandwich(&t, &x_rot)
            },
            GatedMlpOp {
                activation: Activation::Silu,
                ..sandwich(&t, &x_rot)
            },
            GatedMlpOp {
                storage: MlpStorage::Bf16Source,
                ..sandwich(&t, &x_rot)
            },
            GatedMlpOp {
                output: MlpOutput::Residual,
                ..sandwich(&t, &x_rot)
            },
            GatedMlpOp {
                output: MlpOutput::Residual,
                ..trunk(&t, &x_rot)
            },
        ];
        for op in &refused {
            assert!(refusal(op).contains("does not admit"), "{}", refusal(op));
        }
    }

    /// The Qwen3.5 decode declaration: one row, pre-norm, SiLU, F32, the
    /// residual GEMV on one device or a written TP partial.
    #[cfg(feature = "deltanet")]
    fn product<'a>(t: &'a [GpuTensor; 9], x_rot: &'a GpuTensor) -> GatedMlpOp<'a> {
        GatedMlpOp {
            rows: 1,
            activation: Activation::Silu,
            output: MlpOutput::Residual,
            ladder: MlpLadder::Product,
            ..sandwich(t, x_rot)
        }
    }

    #[cfg(feature = "deltanet")]
    #[test]
    fn product_admission() {
        let t = fixture();
        let x_rot = tensor(HIDDEN, DType::F32);
        assert!(product(&t, &x_rot).validate_layout().is_ok());
        let partial = GatedMlpOp {
            output: MlpOutput::Write(&t[6]),
            ..product(&t, &x_rot)
        };
        assert!(partial.validate_layout().is_ok());
        assert!(MlpLadder::Product.tp_shardable());
        // The batched route: rows share the op; its rotation plane covers
        // every row.
        let rows_rot = tensor(ROWS * HIDDEN, DType::F32);
        let rows = GatedMlpOp {
            rows: ROWS,
            ..product(&t, &rows_rot)
        };
        assert!(rows.validate_layout().is_ok());
        let short_rows = GatedMlpOp {
            rows: ROWS,
            ..product(&t, &x_rot)
        };
        assert!(refusal(&short_rows).contains("gated MLP rotation"));

        let post_norm = sandwich(&t, &x_rot).output;
        let refused = [
            GatedMlpOp {
                pre_norm: None,
                ..product(&t, &x_rot)
            },
            GatedMlpOp {
                activation: Activation::GeluTanh,
                ..product(&t, &x_rot)
            },
            GatedMlpOp {
                storage: MlpStorage::Bf16Source,
                ..product(&t, &x_rot)
            },
            GatedMlpOp {
                output: post_norm,
                ..product(&t, &x_rot)
            },
        ];
        for op in &refused {
            assert!(refusal(op).contains("does not admit"), "{}", refusal(op));
        }
        let short_rot = tensor(HIDDEN - 1, DType::F32);
        assert!(refusal(&product(&t, &short_rot)).contains("gated MLP rotation"));
    }

    #[test]
    fn geometry_validation() {
        let t = fixture();
        let x_rot = tensor(1, DType::F32);
        let op = || trunk(&t, &x_rot);
        for bad in [
            GatedMlpOp { rows: 0, ..op() },
            GatedMlpOp { hidden: 0, ..op() },
            GatedMlpOp {
                intermediate: 0,
                ..op()
            },
        ] {
            assert!(refusal(&bad).contains("empty geometry"));
        }

        // Transposed or mis-sized projection geometry, on both ladders.
        let gate_t = weight(&t[0], DType::BF16, HIDDEN, INNER);
        let up_t = weight(&t[1], DType::BF16, INNER, INNER);
        let down_t = weight(&t[2], DType::BF16, INNER, HIDDEN);
        let wide_rot = tensor(ROWS * HIDDEN, DType::F32);
        for base in [op(), sandwich(&t, &wide_rot)] {
            let bad = GatedMlpOp {
                gate: gate_t,
                ..base
            };
            assert!(refusal(&bad).contains("gated MLP gate"));
            let bad = GatedMlpOp { up: up_t, ..base };
            assert!(refusal(&bad).contains("gated MLP up"));
            let bad = GatedMlpOp {
                down: down_t,
                ..base
            };
            assert!(refusal(&bad).contains("gated MLP down"));
        }

        // Trunk: a BF16 buffer smaller than its geometry, and a payload the
        // stateful projections cannot read. The sandwich ladder reads any
        // GEMV format.
        let short = tensor(HIDDEN * INNER - 1, DType::BF16);
        let bad = GatedMlpOp {
            down: weight(&short, DType::BF16, HIDDEN, INNER),
            ..op()
        };
        assert!(refusal(&bad).contains("gated MLP down"));
        let mq4 = weight(&t[0], DType::MQ4G256, INNER, HIDDEN);
        let bad = GatedMlpOp { gate: mq4, ..op() };
        assert!(refusal(&bad).contains("gated MLP gate"));
        let gemma = GatedMlpOp {
            gate: mq4,
            ..sandwich(&t, &wide_rot)
        };
        assert!(gemma.validate_layout().is_ok());
    }

    #[test]
    fn scratch_validation() {
        let t = fixture();
        let x_rot = tensor(1, DType::F32);
        let small_wide = tensor(ROWS * HIDDEN - 1, DType::F32);
        let small_inner = tensor(ROWS * INNER - 1, DType::F32);
        let wrong_dtype = tensor(ROWS * HIDDEN, DType::BF16);
        let op = || trunk(&t, &x_rot);
        let scratch = op().scratch;

        assert!(refusal(&GatedMlpOp {
            x: &small_wide,
            ..op()
        })
        .contains("gated MLP input"));
        assert!(refusal(&GatedMlpOp {
            x: &wrong_dtype,
            ..op()
        })
        .contains("gated MLP input"));
        for (bad, label) in [
            (
                GatedMlpScratch {
                    gate: &small_inner,
                    ..scratch
                },
                "gate scratch",
            ),
            (
                GatedMlpScratch {
                    up: &small_inner,
                    ..scratch
                },
                "up scratch",
            ),
            (
                GatedMlpScratch {
                    act: &small_inner,
                    ..scratch
                },
                "act scratch",
            ),
        ] {
            assert!(refusal(&GatedMlpOp {
                scratch: bad,
                ..op()
            })
            .contains(label));
        }
        let bad = GatedMlpOp {
            output: MlpOutput::Write(&small_wide),
            ..op()
        };
        assert!(refusal(&bad).contains("gated MLP output"));

        // Trunk owes rotation scratch only to FWHT payloads, and then
        // rows * max(hidden, intermediate).
        let fwht_down = weight(&t[2], DType::MQ4G256V2, HIDDEN, INNER);
        let bad = GatedMlpOp {
            down: fwht_down,
            ..op()
        };
        assert!(refusal(&bad).contains("gated MLP rotation"));
        let enough = tensor(ROWS * HIDDEN, DType::F32);
        let fwht = GatedMlpOp {
            down: fwht_down,
            ..trunk(&t, &enough)
        };
        assert!(fwht.validate_layout().is_ok());

        // The sandwich ladder always rotates through `x_rot`: `[hidden]` on
        // one row, rows * max(hidden, intermediate) on rows.
        let small_rot = tensor(ROWS * HIDDEN - 1, DType::F32);
        assert!(refusal(&sandwich(&t, &small_rot)).contains("gated MLP rotation"));
        let row_rot = tensor(HIDDEN, DType::F32);
        let decode = GatedMlpOp {
            rows: 1,
            ..sandwich(&t, &row_rot)
        };
        assert!(decode.validate_layout().is_ok());
        let short_rot = tensor(HIDDEN - 1, DType::F32);
        let decode = GatedMlpOp {
            rows: 1,
            ..sandwich(&t, &short_rot)
        };
        assert!(refusal(&decode).contains("gated MLP rotation"));

        // The sandwich epilogue's tensors and the pre-norm's.
        let base = sandwich(&t, &enough);
        let MlpOutput::PostNorm { norm, eps, out, .. } = base.output else {
            unreachable!()
        };
        let bad = GatedMlpOp {
            output: MlpOutput::PostNorm {
                norm,
                eps,
                out,
                residual: &small_wide,
            },
            ..base
        };
        assert!(refusal(&bad).contains("gated MLP residual"));
        let pre = base.pre_norm.unwrap();
        let bad = GatedMlpOp {
            pre_norm: Some(RmsNorm {
                normed: &small_wide,
                ..pre
            }),
            ..base
        };
        assert!(refusal(&bad).contains("gated MLP normed"));
    }
}
