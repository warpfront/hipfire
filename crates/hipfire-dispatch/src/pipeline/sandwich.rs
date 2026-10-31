// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Sandwich-norm decoder layer operations.
//!
//! A sandwich-norm decoder normalizes each sublayer's input AND its output
//! before the residual add (`x = residual + post_norm(f(pre_norm(x)))`).
//! Architecture crates declare one attention and one MLP operation per layer
//! (plus an optional per-layer-input branch and layer scale) and bind their
//! resident weights, KV storage and fixed scratch; every dtype-, tier-, arch-
//! and fusion-gated route choice lives here.
//!
//! Executors take `rows` consecutive positions laid out row-major and one
//! [`RowsRoute`]: [`RowsRoute::Decode`] is single-token decode, a batched
//! route (prefill or speculative verify) takes the batched kernels at any row
//! count. The bodies are the former Gemma 4 eager decode and batched verify
//! arms moved verbatim, so every route issues the same launches.

use crate::context::DispatchCtx;
use crate::families::attention::{AttentionFamily, AttnParams, FullAttnParams};
use crate::families::fused_qkv::{FusedQkvFamily, FusedQkvParams};
use crate::families::gemm::{GemmFamily, GemmParams};
use crate::families::gemv::{GemvFamily, GemvParams, WeightRef};
use crate::families::kv_tier::{KvTierInputs, KvTierPlan};
use crate::families::rotation::{RotationFamily, RotationParams};
use crate::pipeline::gated_mlp::{execute_gated_mlp, GatedMlpOp, MlpOutput};
use crate::pipeline::steps::{execute_steps, GemvInput, Step};
use crate::pipeline::RowsRoute;
use crate::types::{
    dtype_rotation_plan, DispatchError, GemvVariant, KernelKey, RotationPlan, RotationVariant,
    TileImpl,
};
use hip_bridge::DeviceBuffer;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::LazyLock;

pub(super) fn hip(e: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(e.to_string())
}

fn hip_dbg(e: impl std::fmt::Debug) -> DispatchError {
    DispatchError::Hip(format!("{e:?}"))
}

fn invalid(message: String) -> DispatchError {
    DispatchError::Hip(message)
}

/// `t` holds at least `elems` elements.
fn need(t: &GpuTensor, elems: usize, op: &str, what: &str) -> Result<(), DispatchError> {
    if t.numel() < elems {
        return Err(invalid(format!(
            "{op}: {what} holds {} elements, needs {elems}",
            t.numel()
        )));
    }
    Ok(())
}

/// The rotation and BF16 staging scratch the projections of `weights` read:
/// decode's fused norm + FWHT rotates one `hidden` row; a batch rotates
/// (FWHT formats) or stages (BF16) each weight's `[rows, k]` input.
fn need_projection_scratch(
    s: &SandwichStream<'_>,
    x_rot: &GpuTensor,
    weights: &[&WeightRef<'_>],
    op: &str,
) -> Result<(), DispatchError> {
    if !s.batched() {
        return need(x_rot, s.hidden, op, "rotation scratch");
    }
    let widest = |keep: fn(DType) -> bool| {
        weights
            .iter()
            .filter(|w| keep(w.dtype))
            .map(|w| w.k)
            .max()
            .unwrap_or(0)
    };
    need(
        x_rot,
        s.rows * widest(rotated_rows_format),
        op,
        "rotation scratch",
    )?;
    if let Some(bf16) = s.bf16 {
        need(
            bf16,
            s.rows * widest(|d| d == DType::BF16),
            op,
            "BF16 staging",
        )?;
    }
    Ok(())
}

/// Greedy EAGLE (`HIPFIRE_GEMMA4_EAGLE=1`) turns off the fusions that are not
/// byte-identical to their unfused form and keeps batched Q8 GEMMs on decode's
/// `gemv_q8_0` arithmetic (the few-row `gemv_q8_0_rows` on gfx1151, else the
/// unchunked scalar kernel), so batched verify and single-row decode share one
/// arithmetic.
static EAGLE_STRICT: LazyLock<bool> = LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_GEMMA4_EAGLE")
        .ok()
        .as_deref()
        == Some("1")
});

static GEMV: LazyLock<GemvFamily> = LazyLock::new(GemvFamily::new);
static GEMM: LazyLock<GemmFamily> = LazyLock::new(GemmFamily::new);
static FUSED_QKV: LazyLock<FusedQkvFamily> = LazyLock::new(FusedQkvFamily::new);
static ATTENTION: LazyLock<AttentionFamily> = LazyLock::new(AttentionFamily::new);
static ROTATION: LazyLock<RotationFamily> = LazyLock::new(RotationFamily::new);

/// Arches whose batched sandwich prefill admits the exact fused and wide
/// routes below. Elsewhere batched rows take the per-row fallbacks.
fn batched_fusion_arch(arch: &str) -> bool {
    matches!(arch, "gfx1100" | "gfx1201")
}

/// True while a calibration collector is armed. Routes that never
/// materialize a projection's unrotated input (fused norm+FWHT) step aside so
/// every projection reaches its `maybe_capture_activation` tap.
pub(super) fn calibrating(gpu: &Gpu) -> bool {
    gpu.active_capture.is_some()
}

/// `y = W x` for one row. Taps the calibration collector with the unrotated
/// input.
pub(super) fn gemv(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w: &WeightRef,
    x: &GpuTensor,
    y: &GpuTensor,
) -> Result<(), DispatchError> {
    if w.dtype == DType::BF16 {
        return gemm_rows_wide(gpu, ctx, w, x, y, x, 1);
    }
    gpu.maybe_capture_activation(w.buf, x, 1, w.k);
    GEMV.run_auto(ctx, gpu, w, x, y)
}

fn gemv_prerotated(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w: &WeightRef,
    x_rot: &GpuTensor,
    y: &GpuTensor,
) -> Result<(), DispatchError> {
    GEMV.run(
        ctx,
        gpu,
        &GemvParams {
            w,
            x: x_rot,
            y,
            variant: GemvVariant::Prerotated,
            residual: None,
            gate: None,
            up: None,
        },
    )
}

/// FWHT rotation of `rows` rows of `x` for the MQ weight `w` (AWQ-aware).
fn rotate_x_mq_rows(
    gpu: &mut Gpu,
    w: &WeightRef,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    rows: usize,
) -> Result<(), DispatchError> {
    match w.awq_scale {
        Some(awq) => gpu.rotate_x_mq_awq_batched(x, awq, x_rot, w.k, rows),
        None => gpu.rotate_x_mq_batched(x, x_rot, w.k, rows),
    }
    .map_err(hip)
}

/// Kernel ladder of [`gemm_rows`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ladder {
    /// gfx1151 Q8 takes the 4-warp 64x64 WMMA tile from 64 rows (the scalar
    /// kernel under greedy EAGLE); every arch's other formats take their
    /// batched GEMMs.
    Wide,
    /// Q8 takes the chunked GEMM at every row count; on gfx12 a prefill chunk
    /// of more than one row takes the residual WMMA GEMMs for MQ4/HFQ4. A
    /// decode row's `Route(Decode)` is one row of that ladder.
    Route(RowsRoute),
}

/// gfx12 Muse residual GEMM: bit-identical to the shared residual kernel with
/// more occupancy, which wins only where the grid is not yet full. Measured on
/// gfx1201 at 384 rows: M=6656 +12..18 %, M=8704 -20.6 %, M=19968 -16.6 %.
const MUSE_RESIDUAL_MAX_M: usize = 8192;

fn gfx12_residual(gpu: &Gpu, ladder: Ladder, dtype: DType, rows: usize) -> bool {
    ladder == Ladder::Route(RowsRoute::Prefill)
        && rows > 1
        && gpu.arch_caps.has_wmma_w32_gfx12()
        && matches!(dtype, DType::MQ4G256 | DType::HFQ4G256)
}

/// Formats whose batched GEMMs read an FWHT-G256-rotated input.
fn rotated_rows_format(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::MQ4G256 | DType::HFQ4G256 | DType::MQ6G256 | DType::MQ4G256V2
    )
}

/// `y[rows, m] = x[rows, k] · Wᵀ` on `ladder`. A `Raw` input of a MagnumQuant
/// weight is rotated into `x_rot` (`[rows, k]`) first; a `Prerotated` input
/// already is. BF16 weights stage the input to BF16 (`bf16`, or a per-call
/// buffer). Formats without a batched kernel run one GEMV per row. Taps the
/// calibration collector with the unrotated input.
#[allow(clippy::too_many_arguments)]
pub fn gemm_rows(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w: &WeightRef,
    input: GemvInput<'_>,
    y: &GpuTensor,
    x_rot: &GpuTensor,
    bf16: Option<&GpuTensor>,
    rows: usize,
    ladder: Ladder,
) -> Result<(), DispatchError> {
    let x = match input {
        GemvInput::Raw(x) => x,
        GemvInput::Prerotated(x) => {
            if !rotated_rows_format(w.dtype) {
                return Err(DispatchError::Hip(format!(
                    "gemm rows: a prerotated input cannot feed a {:?} weight",
                    w.dtype
                )));
            }
            x
        }
    };
    if !supports_batched_projection(w.dtype) {
        for row in 0..rows {
            let x_row = x.sub_offset(row * w.k, w.k);
            let y_row = y.sub_offset(row * w.m, w.m);
            gemv(gpu, ctx, w, &x_row, &y_row)?;
        }
        return Ok(());
    }
    let x = match input {
        GemvInput::Raw(x) => {
            gpu.maybe_capture_activation(w.buf, x, rows, w.k);
            if rotated_rows_format(w.dtype) {
                rotate_x_mq_rows(gpu, w, x, x_rot, rows)?;
                x_rot
            } else {
                x
            }
        }
        GemvInput::Prerotated(x) => x,
    };
    let strict = ladder == Ladder::Wide && *EAGLE_STRICT;
    match w.dtype {
        DType::F32 => gpu
            .gemm_f32_batched(w.buf, x, y, w.m, w.k, rows)
            .map_err(hip),
        // gfx1151 few-row verify: decode's gemv_q8_0 over every row, the
        // weights read once (the scalar GEMM reaches ~30 GB/s here).
        DType::Q8_0 if strict && gpu.gemv_q8_0_rows_supported(w.k, rows) => {
            gpu.gemv_q8_0_rows(w.buf, x, y, w.m, w.k, rows).map_err(hip)
        }
        DType::Q8_0 if strict => gpu
            .gemm_q8_0_batched(w.buf, x, y, w.m, w.k, rows)
            .map_err(hip),
        // gfx1151: the 4-warp 64x64 tile reads each weight tile once per 64
        // rows (the 16x16 kernel re-reads it per 16 rows); a sub-64 row tail
        // takes the 16x16 kernel.
        DType::Q8_0
            if ladder == Ladder::Wide
                && gpu.arch_caps.is_gfx1151()
                && w.m % 64 == 0
                && rows >= 64 =>
        {
            let full = rows / 64 * 64;
            gpu.gemm_q8_0_wmma_4w(w.buf, &x.sub_offset(0, full * w.k), y, w.m, w.k, full)
                .map_err(hip)?;
            if full < rows {
                let tail = rows - full;
                gpu.gemm_q8_0_wmma(
                    w.buf,
                    &x.sub_offset(full * w.k, tail * w.k),
                    &y.sub_offset(full * w.m, tail * w.m),
                    w.m,
                    w.k,
                    tail,
                )
                .map_err(hip)?;
            }
            Ok(())
        }
        DType::Q8_0 => gpu
            .gemm_q8_0_batched_chunked(w.buf, x, y, w.m, w.k, rows)
            .map_err(hip),
        DType::MQ4G256 | DType::HFQ4G256 if gfx12_residual(gpu, ladder, w.dtype, rows) => {
            // The F16 activation cache keys on the source pointer alone, and
            // the rotated scratch is reused with new contents.
            gpu.invalidate_fp16_cache();
            gpu.hip.memset(&y.buf, 0, rows * w.m * 4).map_err(hip)?;
            if w.m < MUSE_RESIDUAL_MAX_M
                && rows.is_multiple_of(192)
                && gpu
                    .gemm_hfq4g256_residual_muse(w.buf, x, y, w.m, w.k, rows, 12)
                    .map_err(hip)?
            {
                return Ok(());
            }
            gpu.gemm_hfq4g256_residual(w.buf, x, y, w.m, w.k, rows)
                .map_err(hip)
        }
        DType::MQ4G256 | DType::HFQ4G256 => gpu
            .gemm_hfq4g256_batched_lmhead(w.buf, x, y, w.m, w.k, rows)
            .map_err(hip),
        DType::MQ6G256 => gpu
            .gemm_mq6g256_batched_lmhead(w.buf, x, y, w.m, w.k, rows)
            .map_err(hip),
        DType::MQ4G256V2 => gpu.gemm_mq4g256v2(w.buf, x, y, w.m, w.k, rows).map_err(hip),
        // BF16 teachers (calibration): stage the F32 input to BF16 for the
        // MFMA GEMM, which resolves only where the arch has one.
        DType::BF16 => {
            let n = rows * w.k;
            // ponytail: per-call staging buffer when the caller holds none;
            // hold one in scratch if BF16 calibration becomes throughput-bound.
            let owned = match bf16 {
                Some(_) => None,
                None => Some(gpu.alloc_tensor(&[n], DType::BF16).map_err(hip)?),
            };
            let staged = match (bf16, owned.as_ref()) {
                (Some(b), _) => b.sub_offset(0, n),
                (None, Some(o)) => o.sub_offset(0, n),
                (None, None) => unreachable!(),
            };
            let run = gpu
                .convert_f32_to_bf16(x, &staged, n)
                .map_err(hip)
                .and_then(|_| {
                    GEMM.run_key(
                        KernelKey::GemmBf16Mfma,
                        ctx,
                        gpu,
                        &GemmParams {
                            w,
                            x: &staged,
                            y,
                            batch_size: rows,
                        },
                    )
                });
            if let Some(o) = owned {
                gpu.free_tensor(o).map_err(hip)?;
            }
            run
        }
        other => unreachable!("{other:?} is not a batched projection format"),
    }
}

/// [`gemm_rows`] of an unrotated input on the [`Ladder::Wide`] ladder.
pub(super) fn gemm_rows_wide(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w: &WeightRef,
    x: &GpuTensor,
    y: &GpuTensor,
    x_rot: &GpuTensor,
    rows: usize,
) -> Result<(), DispatchError> {
    gemm_rows(
        gpu,
        ctx,
        w,
        GemvInput::Raw(x),
        y,
        x_rot,
        None,
        rows,
        Ladder::Wide,
    )
}

/// Weight formats [`gemm_rows`] can project with `rows > 1`.
pub fn supports_batched_projection(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F32
            | DType::BF16
            | DType::Q8_0
            | DType::MQ4G256
            | DType::HFQ4G256
            | DType::MQ6G256
            | DType::MQ4G256V2
    )
}

pub(super) fn fused_projection(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    key: KernelKey,
    weights: &[&WeightRef],
    x: &GpuTensor,
    outputs: &[&GpuTensor],
    rows: usize,
) -> Result<(), DispatchError> {
    let bufs: Vec<&GpuTensor> = weights.iter().map(|w| w.buf).collect();
    let m: Vec<usize> = weights.iter().map(|w| w.m).collect();
    FUSED_QKV.run(
        ctx,
        gpu,
        &FusedQkvParams {
            kind: key,
            weights: &bufs,
            x,
            outputs,
            m: &m,
            k: weights[0].k,
            rot_scratch: &[],
            batch_size: (rows > 1).then_some(rows),
        },
    )
}

/// gfx1100 opt-in: Q8 projections sharing one input run as one fused WMMA GEMM.
pub(super) fn q8_fused_prefill(gpu: &Gpu, weights: &[&WeightRef]) -> bool {
    gpu.arch == "gfx1100"
        && gpu.flags.gemma4_q8_fused_prefill
        && weights
            .iter()
            .all(|w| w.dtype == DType::Q8_0 && w.k == weights[0].k)
        && weights[0].k.is_multiple_of(32)
}

fn scale(gpu: &mut Gpu, x: &GpuTensor, factor: f32) -> Result<(), DispatchError> {
    rdna_compute::tensor_ops::scale_f32(
        gpu,
        &rdna_compute::tensor_ops::ScaleF32 {
            values: x,
            scale: factor,
        },
    )
    .map_err(hip)
}

/// How a sandwich model's sublayers choose kernels: the ladder and the
/// fusions. Which rows run (decode or a batch) is the stream's [`RowsRoute`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SandwichRoute {
    /// Batches take the [`Ladder::Wide`] ladder. Decode fuses the Q8 Q/K and
    /// MQ4 gate/up projections and the norm, RoPE and post-norm residual of
    /// one row.
    Wide,
    /// Decode projections of one pre-norm run as generic `Gemv` steps, so the
    /// dispatch fusion table groups them; batches take their route's ladder.
    Steps(StepsRoute),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepsRoute {
    /// FWHT-G256 projections of one pre-norm read one rotated activation
    /// (fused norm + FWHT, AWQ of the first projection); otherwise each
    /// projection rotates the plain norm for itself.
    pub shared_rotation: bool,
    /// MQ4G256/MQ6G256 down projections read the SwiGLU product already
    /// FWHT-rotated (fused SiLU-mul + FWHT).
    pub fused_silu_rotate: bool,
}

/// Residual stream and the fixed scratch every sandwich sublayer shares.
/// Each tensor holds `rows` row-major rows.
#[derive(Clone, Copy)]
pub struct SandwichStream<'a> {
    pub rows: usize,
    pub hidden: usize,
    pub eps: f32,
    /// RMS norm eps of the post-norms.
    pub post_eps: f32,
    /// Residual stream; updated in place unless `out` is set.
    pub x: &'a GpuTensor,
    /// Holds `x` at sublayer entry.
    pub residual: &'a GpuTensor,
    /// Normed activation / sublayer output scratch.
    pub normed: &'a GpuTensor,
    /// Output stream: the sublayer writes `out = x + post_norm(f(x))` and
    /// leaves `x` as it was, so a layer alternating two streams needs no
    /// residual copies (`residual` is then unused). `None` updates `x`.
    pub out: Option<&'a GpuTensor>,
    /// Decode (exactly one row) or a batch of `rows` rows.
    pub route: RowsRoute,
    pub kernels: SandwichRoute,
    /// BF16 staging (`[rows, k]`) of batched BF16 projections; `None`
    /// allocates one per call.
    pub bf16: Option<&'a GpuTensor>,
}

impl SandwichStream<'_> {
    fn len(&self) -> usize {
        self.rows * self.hidden
    }

    /// Whether the rows take the batched kernels.
    fn batched(&self) -> bool {
        self.route.batched()
    }

    fn ladder(&self) -> Ladder {
        match self.kernels {
            SandwichRoute::Wide => Ladder::Wide,
            SandwichRoute::Steps(_) => Ladder::Route(self.route),
        }
    }

    /// Greedy EAGLE's byte-identical routes (Gemma 4, [`SandwichRoute::Wide`]).
    fn strict(&self) -> bool {
        self.kernels == SandwichRoute::Wide && *EAGLE_STRICT
    }

    fn steps_route(&self) -> Option<StepsRoute> {
        match self.kernels {
            SandwichRoute::Wide => None,
            SandwichRoute::Steps(r) => Some(r),
        }
    }

    /// Route/rows consistency and stream extents.
    fn validate(&self, label: &str) -> Result<(), DispatchError> {
        if self.rows == 0 || self.hidden == 0 {
            return Err(invalid(format!("{label}: empty geometry")));
        }
        if self.route == RowsRoute::Decode && self.rows != 1 {
            return Err(invalid(format!(
                "{label}: {} rows on the decode route",
                self.rows
            )));
        }
        let n = self.len();
        need(self.x, n, label, "stream x")?;
        need(self.normed, n, label, "normed scratch")?;
        match self.out {
            Some(out) => need(out, n, label, "output stream"),
            None => need(self.residual, n, label, "residual"),
        }
    }

    /// Copies are kernel launches so hipGraph capture and the retained-replay
    /// recorder both see them (a D2D memcpy is invisible to Redline PM4).
    pub(super) fn save_residual(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        gpu.copy_f32_buffer(self.residual, self.x, self.len())
            .map_err(hip)
    }

    /// Saves `x` when [`Self::post_norm_residual`] with `post_norm` reads the
    /// saved copy.
    fn save_residual_for(
        &self,
        gpu: &mut Gpu,
        post_norm: Option<&GpuTensor>,
    ) -> Result<(), DispatchError> {
        if self.out.is_none() && post_norm.is_some() {
            self.save_residual(gpu)?;
        }
        Ok(())
    }

    fn restore_residual(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        gpu.copy_f32_buffer(self.x, self.residual, self.len())
            .map_err(hip)
    }

    fn norm(
        &self,
        gpu: &mut Gpu,
        x: &GpuTensor,
        w: &GpuTensor,
        out: &GpuTensor,
    ) -> Result<(), DispatchError> {
        if self.batched() {
            gpu.rmsnorm_batched(x, w, out, self.rows, self.hidden, self.eps)
        } else {
            gpu.rmsnorm_f32(x, w, out, self.eps)
        }
        .map_err(hip)
    }

    /// `x = x + post_norm(out)` (into [`Self::out`] when set), or the plain
    /// residual add without a post-norm; `out` may alias `normed`.
    pub(super) fn post_norm_residual(
        &self,
        gpu: &mut Gpu,
        out: &GpuTensor,
        post_norm: Option<&GpuTensor>,
    ) -> Result<(), DispatchError> {
        let n = self.len();
        let (x, out_rows) = (self.x.sub_offset(0, n), out.sub_offset(0, n));
        match (post_norm, self.out) {
            (Some(post_norm), None) => {
                if !self.batched() && !self.strict() {
                    return gpu
                        .rmsnorm_residual_add_f32(
                            out,
                            post_norm,
                            self.residual,
                            self.x,
                            self.post_eps,
                        )
                        .map_err(hip);
                }
                self.post_norm(gpu, out, post_norm, self.normed)?;
                self.restore_residual(gpu)?;
                gpu.add_inplace_f32(self.x, self.normed).map_err(hip)
            }
            (Some(post_norm), Some(dst)) if !self.batched() => gpu
                .rmsnorm_residual_add_f32(out, post_norm, self.x, dst, self.post_eps)
                .map_err(hip),
            (Some(post_norm), Some(dst)) => {
                self.post_norm(gpu, out, post_norm, out)?;
                gpu.add_f32(&x, &out_rows, &dst.sub_offset(0, n))
                    .map_err(hip)
            }
            (None, None) => gpu.add_inplace_f32(&x, &out_rows).map_err(hip),
            (None, Some(dst)) => gpu
                .add_f32(&x, &out_rows, &dst.sub_offset(0, n))
                .map_err(hip),
        }
    }

    fn post_norm(
        &self,
        gpu: &mut Gpu,
        x: &GpuTensor,
        w: &GpuTensor,
        out: &GpuTensor,
    ) -> Result<(), DispatchError> {
        if self.batched() {
            gpu.rmsnorm_batched(x, w, out, self.rows, self.hidden, self.post_eps)
        } else {
            gpu.rmsnorm_f32(x, w, out, self.post_eps)
        }
        .map_err(hip)
    }
}

/// RMS norm of `rows` rows of `hidden`: the one-row kernel for decode, the
/// batched kernel otherwise.
pub(super) fn rms_norm(
    gpu: &mut Gpu,
    rows: usize,
    hidden: usize,
    eps: f32,
    x: &GpuTensor,
    w: &GpuTensor,
    out: &GpuTensor,
) -> Result<(), DispatchError> {
    if rows == 1 {
        gpu.rmsnorm_f32(x, w, out, eps)
    } else {
        gpu.rmsnorm_batched(x, w, out, rows, hidden, eps)
    }
    .map_err(hip)
}

/// Rotary position embedding applied to Q and K.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopeKind {
    /// Every channel pair of the head rotates.
    RotateHalf,
    /// The first `rot_pairs` pairs of each half rotate; the rest pass through.
    PartialHalved { rot_pairs: usize },
    /// Rotate-half over the first `rot_dim` channels of each head (pair
    /// `(i, i + rot_dim / 2)`); channels `rot_dim..` pass through.
    PrefixHalfSplit { rot_dim: usize },
    /// No rotation (NoPE layers).
    NoPe,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rope {
    pub kind: RopeKind,
    pub theta: f32,
}

impl Rope {
    /// Rotated pairs in the whole-head rotate-half pairing; `None` when the
    /// pairing is a prefix of the head.
    fn halved_pairs(&self, head_dim: usize) -> Option<usize> {
        match self.kind {
            RopeKind::RotateHalf => Some(head_dim / 2),
            RopeKind::PartialHalved { rot_pairs } => Some(rot_pairs),
            RopeKind::PrefixHalfSplit { .. } => None,
            RopeKind::NoPe => Some(0),
        }
    }

    /// RoPE of one row at the position in `pos_buf`; `n_kv_heads == 0` leaves
    /// K untouched.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_row(
        &self,
        gpu: &mut Gpu,
        q: &GpuTensor,
        k: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(), DispatchError> {
        let theta = self.theta;
        match self.kind {
            RopeKind::NoPe => Ok(()),
            RopeKind::RotateHalf => {
                gpu.rope_f32(q, k, pos_buf, n_heads, n_kv_heads, head_dim, theta)
            }
            RopeKind::PartialHalved { rot_pairs } => gpu.rope_partial_halved_f32(
                q, k, pos_buf, n_heads, n_kv_heads, head_dim, rot_pairs, theta,
            ),
            RopeKind::PrefixHalfSplit { rot_dim } => {
                // The launcher takes the half-split kernel unless the legacy
                // interleaved probe is on, which would change the pairing.
                if gpu.flags.rope_interleaved_legacy {
                    return Err(DispatchError::Hip(
                        "prefix half-split RoPE refuses HIPFIRE_ROPE_INTERLEAVED_LEGACY".into(),
                    ));
                }
                #[cfg(feature = "deltanet")]
                {
                    gpu.rope_partial_interleaved_f32(
                        q, k, pos_buf, n_heads, n_kv_heads, head_dim, rot_dim, theta,
                    )
                }
                #[cfg(not(feature = "deltanet"))]
                {
                    let _ = rot_dim;
                    return Err(DispatchError::UnsupportedVariant {
                        family: "rope",
                        variant: "prefix-half-split-requires-deltanet",
                        arch: "",
                        quant: "",
                    });
                }
            }
        }
        .map_err(hip)
    }

    /// RoPE of `rows` rows, row `r` at the I32 position `positions[r]`;
    /// `n_kv_heads == 0` leaves K untouched.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_rows(
        &self,
        gpu: &mut Gpu,
        q: &GpuTensor,
        k: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rows: usize,
    ) -> Result<(), DispatchError> {
        let theta = self.theta;
        match self.kind {
            RopeKind::RotateHalf => {
                gpu.rope_batched_f32(q, k, positions, n_heads, n_kv_heads, head_dim, theta, rows)
            }
            RopeKind::PartialHalved { rot_pairs } => gpu.rope_partial_halved_f32_batched(
                q, k, positions, n_heads, n_kv_heads, head_dim, rot_pairs, theta, rows,
            ),
            RopeKind::PrefixHalfSplit { .. } => {
                return Err(DispatchError::Hip(
                    "prefix half-split RoPE is single-row only".into(),
                ))
            }
            RopeKind::NoPe => return Ok(()),
        }
        .map_err(hip)
    }
}

/// The K/V one attention sublayer attends. A sublayer with its own
/// [`KvProjection`] writes the current rows first; a query-only sublayer reads
/// another layer's populated cache (KV sharing, draft heads).
pub struct SandwichKv<'a> {
    pub k_cache: &'a GpuTensor,
    pub v_cache: &'a GpuTensor,
    pub physical_cap: usize,
    /// Sliding window; `0` = full causal.
    pub window: usize,
    pub route: KvRoute<'a>,
}

/// Cache layout and attention kernels of a [`SandwichKv`].
pub enum KvRoute<'a> {
    /// The KV-tier plan's decode attention (Gemma's Q8 tile/reduce pair);
    /// batches take the native causal or windowed kernels over a Q8 cache,
    /// the GQA-shared WMMA flash kernel where admitted.
    Tier {
        tier: KvTierInputs,
        givens_cos: Option<&'a GpuTensor>,
        givens_sin: Option<&'a GpuTensor>,
        flash_partials: &'a GpuTensor,
    },
    /// Absolute-position Q8 cache (row `t` holds position `t`). Decode takes
    /// the one-workgroup-per-head kernel below [`WINDOWED_Q8_TILED_MIN_SEQ`]
    /// and the key-tiled flash kernel from there; batches take the query-tiled
    /// WMMA flash kernels per [`RowsRoute`] (see [`windowed_q8_rows_flash`]),
    /// or with `wmma_flash == false` the F32 scalar kernels.
    WindowedQ8 {
        tiled: Option<WindowedQ8Tiled<'a>>,
        rows_flash: Option<WindowedQ8RowsFlash<'a>>,
        wmma_flash: bool,
    },
    /// Block-diffusion draft: the rows attend `kv_rows` uncached F32 K/V rows
    /// (`k_cache`/`v_cache`), the context ahead of them then the rows'
    /// own, without a causal mask.
    Draft {
        kv_rows: usize,
        context: Option<DraftContext<'a>>,
    },
}

/// Context rows of a [`KvRoute::Draft`] attention: `delta` rows new since
/// the last window are projected from `input` (K normed, before RoPE) into
/// the cache, the cached `window` rows are copied to the front of the
/// attended K/V and rotated at `positions`.
pub struct DraftContext<'a> {
    pub delta: usize,
    pub input: &'a GpuTensor,
    /// FWHT scratch of the `delta` input rows.
    pub rot: &'a GpuTensor,
    pub k_new: &'a GpuTensor,
    pub v_new: &'a GpuTensor,
    pub k_window: &'a GpuTensor,
    pub v_window: &'a GpuTensor,
    /// Front of the attended K/V the window is copied to.
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub rows: usize,
    pub positions: &'a GpuTensor,
}

/// From this context length on, [`KvRoute::WindowedQ8`] decode takes the
/// key-tiled flash kernel instead of the one-workgroup-per-head kernel. The
/// per-head kernel leaves most of the GPU idle and grows with the context; the
/// tiled one spreads tiles over the grid but pays a fixed reduction. Measured
/// on gfx1201, Muse Glimmer decode (32 heads, head_dim 128), `hipfire bench
/// --matrix --tg 64`, tok/s, medians of 3:
///
/// ```text
///     ctx        768   1024   1536   2048   4096   8192
///     per-head  29.99  29.08  27.53  26.22  24.94  22.66
///     tiled     28.65  28.55  28.38  28.20  27.86  27.20
/// ```
///
/// Both kernels read the same cache with the same mask; the output is
/// byte-identical on either side of the threshold.
pub const WINDOWED_Q8_TILED_MIN_SEQ: usize = 1280;

/// F32 elements of the tiled kernel's partials for a cache of `cap` rows.
pub fn windowed_q8_partials_elems(n_heads: usize, head_dim: usize, cap: usize) -> usize {
    n_heads * cap.div_ceil(128) * (2 + head_dim) * 64
}

/// Decode scratch of the key-tiled windowed Q8 attention kernel.
#[derive(Clone, Copy)]
pub struct WindowedQ8Tiled<'a> {
    /// `[1]` copy of the position (i32 bits).
    pub positions: &'a GpuTensor,
    /// [`windowed_q8_partials_elems`] floats.
    pub partials: &'a GpuTensor,
}

/// Prefill scratch of the key-tiled flash kernel over a chunk of rows.
#[derive(Clone, Copy)]
pub struct WindowedQ8RowsFlash<'a> {
    /// `rows` i32: each row's position minus [`windowed_q8_rows_origin`].
    pub shifted_positions: &'a GpuTensor,
    /// [`windowed_q8_partials_elems`] floats.
    pub partials: &'a GpuTensor,
}

/// Context length past which a full (`window == 0`) layer's prefill takes the
/// flash kernels: `attention_q8_0_kv_batched` sizes its LDS by the sequence
/// and fails past about 16K, while below this the flash route is neutral.
const WINDOWED_Q8_FULL_FLASH_MIN_SEQ: usize = 2048;

/// Whether a prefill chunk of `rows` rows at `position` takes the flash
/// kernels on a layer of `window` (and so reads [`WindowedQ8RowsFlash`]).
pub fn windowed_q8_rows_flash(position: usize, rows: usize, window: usize) -> bool {
    let seq_len = position + rows;
    if window == 0 {
        seq_len > WINDOWED_Q8_FULL_FLASH_MIN_SEQ
    } else {
        seq_len > window
    }
}

/// Tile-aligned key origin of a sliding layer's prefill flash launch.
///
/// The flash launcher sizes its grid by the whole context, one block per 128
/// keys, though only `window` of them can be in range. Rebasing keys,
/// positions and the sequence length by one origin keeps every comparison the
/// kernel makes (causal and `seq_len - window`) and launches only the live
/// tiles. The origin is tile-aligned and at most the first row's lowest
/// in-window key; `0` while the chunk ends inside the window.
pub fn windowed_q8_rows_origin(position: usize, rows: usize, window: usize) -> usize {
    if window == 0 || position + rows <= window {
        return 0;
    }
    ((position + 1).saturating_sub(window) / 128) * 128
}

/// K/V projections of an attention sublayer that writes its own cache rows.
pub struct KvProjection<'a> {
    pub wk: WeightRef<'a>,
    /// `None` takes V from the pre-norm K (K=V).
    pub wv: Option<WeightRef<'a>>,
    pub k_norm: &'a GpuTensor,
    /// Weight-less V RMSNorm, bound as a ones vector of `head_dim`.
    pub v_norm: Option<&'a GpuTensor>,
}

/// Output gate of an attention sublayer: `attn_out *= sigmoid(W · n)` over
/// the sublayer's pre-norm `n`, before the output projection.
pub struct AttnGate<'a> {
    pub w: WeightRef<'a>,
    pub out: &'a GpuTensor,
}

/// Attention sublayer:
/// `x = residual + post_norm(o(attend(rope(q_norm(q)), rope(k_norm(k)), v_norm(v))))`
/// over `input_norm(x)`; without a post-norm the plain pre-norm residual.
///
/// `kv_proj == None` projects only Q and attends a cache it does not write.
/// Row `r` sits at absolute position `position + r`.
pub struct SandwichAttentionOp<'a> {
    pub stream: SandwichStream<'a>,
    pub position: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub input_norm: &'a GpuTensor,
    pub wq: WeightRef<'a>,
    pub q_norm: &'a GpuTensor,
    pub kv_proj: Option<KvProjection<'a>>,
    /// Q multiplier applied after q_norm (sets the softmax scale against the
    /// kernel's `1/sqrt(head_dim)`).
    pub q_scale: f32,
    pub rope: Rope,
    pub kv: SandwichKv<'a>,
    /// Device position scalar (decode).
    pub pos_buf: &'a DeviceBuffer,
    /// Last key position and its device scalar for a single query-only row
    /// whose own row is not cached (a draft head querying at the pending
    /// token's position). `None` attends through `position`.
    pub attend_through: Option<(usize, &'a DeviceBuffer)>,
    /// Per-row i32 positions (batched rows).
    pub positions: Option<&'a GpuTensor>,
    /// Output gate ([`SandwichRoute::Steps`] only).
    pub gate: Option<AttnGate<'a>>,
    pub wo: WeightRef<'a>,
    pub post_norm: Option<&'a GpuTensor>,
    /// FWHT rotation scratch, `[rows, max(hidden, n_heads*head_dim)]`.
    pub x_rot: &'a GpuTensor,
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub attn_out: &'a GpuTensor,
}

impl SandwichAttentionOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        self.validate()
    }

    /// Checks route/rows consistency, the cache bound and the scratch the
    /// chosen kernels read; launches nothing.
    fn validate(&self) -> Result<(), DispatchError> {
        const L: &str = "sandwich attention";
        let s = &self.stream;
        s.validate(L)?;
        let (rows, hd) = (s.rows, self.head_dim);
        if self.n_kv_heads == 0 || hd == 0 || !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(invalid(format!(
                "{L}: geometry heads={} kv_heads={} head_dim={hd}",
                self.n_heads, self.n_kv_heads
            )));
        }
        let q_len = rows * self.n_heads * hd;
        let kv_len = rows * self.n_kv_heads * hd;
        need(self.q, q_len, L, "q")?;
        need(self.attn_out, q_len, L, "attention output")?;
        if self.kv_proj.is_some() {
            need(self.k, kv_len, L, "k")?;
            need(self.v, kv_len, L, "v")?;
        }
        if s.batched() {
            let positions = self
                .positions
                .ok_or_else(|| invalid(format!("{L}: batched rows need per-row positions")))?;
            need(positions, rows, L, "positions")?;
        }
        if let Some((through, _)) = self.attend_through {
            let tier = matches!(self.kv.route, KvRoute::Tier { .. });
            if s.batched() || self.kv_proj.is_some() || !tier || through >= self.position {
                return Err(invalid(format!(
                    "{L}: attending through {through} needs one query-only Tier row past it (position {})",
                    self.position
                )));
            }
        }
        let mut weights = vec![&self.wq, &self.wo];
        weights.extend(self.wk());
        weights.extend(self.wv());
        if let Some(gate) = &self.gate {
            if s.steps_route().is_none() {
                return Err(invalid(format!(
                    "{L}: an output gate needs the steps route"
                )));
            }
            need(gate.out, q_len, L, "gate output")?;
            weights.push(&gate.w);
        }
        need_projection_scratch(s, self.x_rot, &weights, L)?;
        let kv = &self.kv;
        // Tier and windowed caches are position-indexed (row `t` holds
        // position `t`); a draft attends its own uncached rows.
        if !matches!(kv.route, KvRoute::Draft { .. }) && self.position + rows > kv.physical_cap {
            return Err(invalid(format!(
                "{L}: rows {}..{} outside a cache of {} rows",
                self.position,
                self.position + rows,
                kv.physical_cap
            )));
        }
        match &kv.route {
            KvRoute::Tier { .. } => Ok(()),
            KvRoute::WindowedQ8 {
                tiled, rows_flash, ..
            } => {
                if s.steps_route().is_none() {
                    return Err(invalid(format!(
                        "{L}: windowed Q8 KV needs the steps route"
                    )));
                }
                if self.kv_proj.is_none() {
                    return Err(invalid(format!("{L}: windowed Q8 KV writes its own rows")));
                }
                let partials = windowed_q8_partials_elems(self.n_heads, hd, kv.physical_cap);
                match s.route {
                    RowsRoute::Decode if self.position + 1 >= WINDOWED_Q8_TILED_MIN_SEQ => {
                        let t = tiled.ok_or_else(|| {
                            invalid(format!(
                                "{L}: position {} needs tiled scratch",
                                self.position
                            ))
                        })?;
                        need(t.positions, 1, L, "tiled positions")?;
                        need(t.partials, partials, L, "tiled partials")
                    }
                    RowsRoute::Prefill
                        if windowed_q8_rows_flash(self.position, rows, kv.window) =>
                    {
                        let f = rows_flash.ok_or_else(|| {
                            invalid(format!(
                                "{L}: prefill at {}+{rows} needs flash scratch",
                                self.position
                            ))
                        })?;
                        need(f.shifted_positions, rows, L, "flash shifted positions")?;
                        need(f.partials, partials, L, "flash partials")
                    }
                    _ => Ok(()),
                }
            }
            KvRoute::Draft { kv_rows, context } => {
                if s.steps_route().is_none() || !s.batched() {
                    return Err(invalid(format!(
                        "{L}: draft attention needs batched rows on the steps route"
                    )));
                }
                if !matches!(&self.kv_proj, Some(KvProjection { wv: Some(_), .. })) {
                    return Err(invalid(format!(
                        "{L}: draft attention needs K and V projections"
                    )));
                }
                let ctx_rows = context.as_ref().map_or(0, |c| c.rows);
                if ctx_rows + rows != *kv_rows || *kv_rows > kv.physical_cap {
                    return Err(invalid(format!(
                        "{L}: draft K/V of {kv_rows} rows != context {ctx_rows} + rows {rows} (cap {})",
                        kv.physical_cap
                    )));
                }
                let kv_all = kv_rows * self.n_kv_heads * hd;
                need(kv.k_cache, kv_all, L, "draft K")?;
                need(kv.v_cache, kv_all, L, "draft V")?;
                if let Some(c) = context {
                    need(c.positions, c.rows, L, "context positions")?;
                }
                Ok(())
            }
        }
    }
}

impl SandwichAttentionOp<'_> {
    fn wk(&self) -> Option<&WeightRef<'_>> {
        self.kv_proj.as_ref().map(|p| &p.wk)
    }

    fn wv(&self) -> Option<&WeightRef<'_>> {
        self.kv_proj.as_ref().and_then(|p| p.wv.as_ref())
    }

    fn projections(&self) -> [Option<(&WeightRef<'_>, &GpuTensor)>; 3] {
        [
            Some((&self.wq, self.q)),
            self.wk().map(|w| (w, self.k)),
            self.wv().map(|w| (w, self.v)),
        ]
    }

    fn kv_len(&self) -> usize {
        self.stream.rows * self.n_kv_heads * self.head_dim
    }

    /// `input_norm(x)` → Q, K, V (and gate) projections (V from K when K=V).
    fn project(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<(), DispatchError> {
        let s = &self.stream;
        if let Some(route) = s.steps_route() {
            let mut group = vec![(&self.wq, self.q)];
            group.extend(self.wk().map(|w| (w, self.k)));
            group.extend(self.wv().map(|w| (w, self.v)));
            group.extend(self.gate.as_ref().map(|g| (&g.w, g.out)));
            project_group(gpu, ctx, s, route, self.input_norm, self.x_rot, &group)?;
            if self.wk().is_some() && self.wv().is_none() {
                gpu.copy_f32_buffer(self.v, self.k, self.kv_len())
                    .map_err(hip)?;
            }
            return Ok(());
        }
        if self.gate.is_some() {
            return Err(DispatchError::Hip(
                "sandwich attention: an output gate needs the steps route".into(),
            ));
        }
        if s.batched() {
            return self.project_rows(gpu, ctx);
        }
        let all_mq4 = self
            .projections()
            .iter()
            .flatten()
            .all(|(w, _)| w.dtype == DType::MQ4G256 && w.awq_scale.is_none());
        if all_mq4 && !calibrating(gpu) {
            gpu.fused_rmsnorm_rotate_mq(s.x, self.input_norm, self.x_rot, s.hidden, s.eps)
                .map_err(hip)?;
            for (w, out) in self.projections().iter().flatten() {
                gemv_prerotated(gpu, ctx, w, self.x_rot, out)?;
            }
        } else {
            s.norm(gpu, s.x, self.input_norm, s.normed)?;
            match self.wk() {
                Some(wk) if self.wq.dtype == DType::Q8_0 && wk.dtype == DType::Q8_0 => {
                    fused_projection(
                        gpu,
                        ctx,
                        KernelKey::FusedGateUpQ8_0,
                        &[&self.wq, wk],
                        s.normed,
                        &[self.q, self.k],
                        1,
                    )?;
                }
                Some(wk) => {
                    gemv(gpu, ctx, &self.wq, s.normed, self.q)?;
                    gemv(gpu, ctx, wk, s.normed, self.k)?;
                }
                None => gemv(gpu, ctx, &self.wq, s.normed, self.q)?,
            }
            if let Some(wv) = self.wv() {
                gemv(gpu, ctx, wv, s.normed, self.v)?;
            }
        }
        if self.wk().is_some() && self.wv().is_none() {
            gpu.copy_f32_buffer(self.v, self.k, self.kv_len())
                .map_err(hip)?;
        }
        Ok(())
    }

    fn project_rows(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<(), DispatchError> {
        let s = &self.stream;
        let rows = s.rows;
        s.norm(gpu, s.x, self.input_norm, s.normed)?;
        let mut v_projected = false;
        match (self.wk(), self.wv()) {
            (Some(wk), Some(wv)) if q8_fused_prefill(gpu, &[&self.wq, wk, wv]) => {
                fused_projection(
                    gpu,
                    ctx,
                    KernelKey::FusedQkvQ8_0,
                    &[&self.wq, wk, wv],
                    s.normed,
                    &[self.q, self.k, self.v],
                    rows,
                )?;
                v_projected = true;
            }
            (Some(wk), None) if q8_fused_prefill(gpu, &[&self.wq, wk]) => {
                fused_projection(
                    gpu,
                    ctx,
                    KernelKey::FusedGateUpQ8_0,
                    &[&self.wq, wk],
                    s.normed,
                    &[self.q, self.k],
                    rows,
                )?;
            }
            (wk, wv) => {
                gemm_rows_wide(gpu, ctx, &self.wq, s.normed, self.q, self.x_rot, rows)?;
                if let Some(wk) = wk {
                    gemm_rows_wide(gpu, ctx, wk, s.normed, self.k, self.x_rot, rows)?;
                }
                if let (Some(_), Some(wv)) = (wk, wv) {
                    gemm_rows_wide(gpu, ctx, wv, s.normed, self.v, self.x_rot, rows)?;
                    v_projected = true;
                }
            }
        }
        if self.wk().is_some() && !v_projected {
            gpu.copy_f32_buffer(self.v, self.k, self.kv_len())
                .map_err(hip)?;
        }
        Ok(())
    }

    /// Per-head q_norm/k_norm, weight-less v_norm, Q prescale, RoPE.
    fn position_encode(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let s = &self.stream;
        let eps = s.eps;
        let rows = s.rows;
        let (n_heads, n_kv, hd) = (self.n_heads, self.n_kv_heads, self.head_dim);
        if let Some(ones) = self.kv_proj.as_ref().and_then(|p| p.v_norm) {
            gpu.rmsnorm_batched(self.v, ones, self.v, rows * n_kv, hd, eps)
                .map_err(hip)?;
        }
        let fused_pairs = self.rope.halved_pairs(hd).filter(|_| !s.strict());
        if let (false, Some(p), Some(rot_pairs)) = (s.batched(), &self.kv_proj, fused_pairs) {
            return gpu
                .fused_gemma4_qk_norm_rope_f32(
                    self.q,
                    self.k,
                    self.q_norm,
                    p.k_norm,
                    self.pos_buf,
                    n_heads,
                    n_kv,
                    hd,
                    rot_pairs,
                    self.q_scale,
                    self.rope.theta,
                    eps,
                )
                .map_err(hip);
        }
        gpu.rmsnorm_batched(self.q, self.q_norm, self.q, rows * n_heads, hd, eps)
            .map_err(hip)?;
        if let Some(p) = &self.kv_proj {
            gpu.rmsnorm_batched(self.k, p.k_norm, self.k, rows * n_kv, hd, eps)
                .map_err(hip)?;
        }
        scale(gpu, self.q, self.q_scale)?;
        let rope_kv = if self.kv_proj.is_some() { n_kv } else { 0 };
        if !s.batched() {
            return self
                .rope
                .apply_row(gpu, self.q, self.k, self.pos_buf, n_heads, rope_kv, hd);
        }
        self.rope.apply_rows(
            gpu,
            self.q,
            self.k,
            self.positions()?,
            n_heads,
            rope_kv,
            hd,
            rows,
        )
    }

    fn positions(&self) -> Result<&GpuTensor, DispatchError> {
        self.positions.ok_or_else(|| {
            DispatchError::Hip("sandwich attention: batched rows need per-row positions".into())
        })
    }

    fn attend(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<(), DispatchError> {
        match &self.kv.route {
            KvRoute::Tier {
                tier,
                givens_cos,
                givens_sin,
                flash_partials,
            } => {
                if self.stream.batched() {
                    self.attend_rows(gpu, tier, flash_partials)
                } else {
                    self.attend_tier(gpu, ctx, tier, *givens_cos, *givens_sin, flash_partials)
                }
            }
            KvRoute::WindowedQ8 {
                tiled,
                rows_flash,
                wmma_flash,
            } => match self.stream.route {
                RowsRoute::Decode => self.attend_windowed_q8(gpu, *tiled),
                route => self.attend_windowed_q8_rows(gpu, route, *rows_flash, *wmma_flash),
            },
            KvRoute::Draft { kv_rows, context } => {
                self.attend_draft(gpu, ctx, *kv_rows, context.as_ref())
            }
        }
    }

    fn attend_tier(
        &self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        tier: &KvTierInputs,
        givens_cos: Option<&GpuTensor>,
        givens_sin: Option<&GpuTensor>,
        flash_partials: &GpuTensor,
    ) -> Result<(), DispatchError> {
        let kv = &self.kv;
        let (pos, pos_buf) = self.attend_through.unwrap_or((self.position, self.pos_buf));
        let mut plan = KvTierPlan::derive(KvTierInputs {
            pos,
            batch_size: 1,
            q8_windowed: true,
            window: kv.window as i32,
            ..*tier
        })
        .map_err(hip)?;
        // Gemma 4 decode runs its own Q8 tile/reduce pair; the shared
        // modules stay exactly as every other model runs them.
        plan.gemma_tiles = true;
        let io = AttnParams {
            q: self.q,
            k: self.k,
            v: self.v,
            k_cache: kv.k_cache,
            v_cache: kv.v_cache,
            k_scales: None,
            v_scales: None,
            pos_buf,
            pos,
            positions: None,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            physical_cap: kv.physical_cap,
            batch_size: 1,
            max_ctx_len: 0,
            flash_partials: Some(flash_partials),
            givens_cos,
            givens_sin,
            tree_bias: None,
            block_start: 0,
            block_cols: 0,
            output_gate: None,
            output_awq_scale: None,
            output: self.attn_out,
        };
        if self.kv_proj.is_some() {
            ATTENTION.run_attention(ctx, gpu, &plan, &io)
        } else {
            ATTENTION.run_attend_only(ctx, gpu, &plan, &io)
        }
    }

    /// Causal attention for `rows` consecutive positions over a Q8 cache.
    fn attend_rows(
        &self,
        gpu: &mut Gpu,
        tier: &KvTierInputs,
        flash_partials: &GpuTensor,
    ) -> Result<(), DispatchError> {
        let kv = &self.kv;
        if !tier.quant_q8 {
            return Err(DispatchError::Hip(
                "sandwich attention: batched rows need a Q8 KV cache".into(),
            ));
        }
        let rows = self.stream.rows;
        let positions = self.positions()?;
        let (n_heads, n_kv, hd) = (self.n_heads, self.n_kv_heads, self.head_dim);
        let seq_len = self.position + rows;
        if self.kv_proj.is_some() {
            gpu.kv_cache_write_q8_0_batched(kv.k_cache, self.k, positions, n_kv, hd, rows)
                .map_err(hip)?;
            gpu.kv_cache_write_q8_0_batched(kv.v_cache, self.v, positions, n_kv, hd, rows)
                .map_err(hip)?;
        }
        // Prefill is not tree verification: the native causal kernels, and the
        // windowed tile kernel for sliding layers. gfx1151 takes the
        // GQA-shared WMMA flash kernel for every layer.
        if gpu.attention_q8_0_prefill_gqa_wmma_admitted(n_heads, n_kv, hd) {
            gpu.attention_q8_0_prefill_gqa_wmma(
                self.q,
                kv.k_cache,
                kv.v_cache,
                self.attn_out,
                positions,
                n_heads,
                n_kv,
                hd,
                rows,
                kv.window as i32,
            )
        } else if kv.window > 0 {
            gpu.attention_flash_q8_0_batched_masked_windowed(
                self.q,
                kv.k_cache,
                kv.v_cache,
                self.attn_out,
                positions,
                n_heads,
                n_kv,
                hd,
                kv.physical_cap,
                seq_len,
                rows,
                flash_partials,
                None,
                0,
                0,
                kv.window as i32,
            )
        } else if seq_len > 8_192 {
            gpu.attention_flash_q8_0_batched_masked(
                self.q,
                kv.k_cache,
                kv.v_cache,
                self.attn_out,
                positions,
                n_heads,
                n_kv,
                hd,
                kv.physical_cap,
                seq_len,
                rows,
                flash_partials,
                None,
                0,
                0,
            )
        } else {
            gpu.attention_q8_0_kv_batched_masked(
                self.q,
                kv.k_cache,
                kv.v_cache,
                self.attn_out,
                positions,
                n_heads,
                n_kv,
                hd,
                kv.physical_cap,
                seq_len,
                rows,
                None,
                0,
                0,
            )
        }
        .map_err(hip)
    }

    /// Decode row over an absolute-position Q8 cache: K/V write, then the
    /// per-head kernel or, from [`WINDOWED_Q8_TILED_MIN_SEQ`], the key-tiled
    /// flash kernel.
    fn attend_windowed_q8(
        &self,
        gpu: &mut Gpu,
        tiled: Option<WindowedQ8Tiled<'_>>,
    ) -> Result<(), DispatchError> {
        let kv = &self.kv;
        let (n_heads, n_kv, hd) = (self.n_heads, self.n_kv_heads, self.head_dim);
        gpu.kv_cache_write_q8_0(kv.k_cache, self.k, self.pos_buf, n_kv, hd)
            .map_err(hip)?;
        gpu.kv_cache_write_q8_0(kv.v_cache, self.v, self.pos_buf, n_kv, hd)
            .map_err(hip)?;
        let seq_len = self.position + 1;
        if seq_len < WINDOWED_Q8_TILED_MIN_SEQ {
            // `seq_len` sizes the launch's score LDS to the live context, not
            // the cache: the kernel indexes scores by absolute position.
            return gpu
                .attention_q8_0_kv_swa(
                    self.q,
                    kv.k_cache,
                    kv.v_cache,
                    self.attn_out,
                    self.pos_buf,
                    seq_len,
                    n_heads,
                    n_kv,
                    hd,
                    kv.physical_cap,
                    kv.window,
                )
                .map_err(hip);
        }
        let t = tiled.ok_or_else(|| {
            DispatchError::Hip(format!(
                "windowed q8 attention at position {} needs tiled scratch",
                self.position
            ))
        })?;
        gpu.attention_flash_q8_0_batched_masked_windowed(
            self.q,
            kv.k_cache,
            kv.v_cache,
            self.attn_out,
            t.positions,
            n_heads,
            n_kv,
            hd,
            kv.physical_cap,
            seq_len,
            1,
            t.partials,
            None,
            0,
            0,
            kv.window as i32,
        )
        .map_err(hip)
    }

    /// Batched rows over an absolute-position Q8 cache.
    ///
    /// From two rows (unless `wmma_flash` is off) both routes take the
    /// query-tiled Q8 WMMA flash kernels (one key fragment reused across 16
    /// query rows), full layers the causal one and sliding layers the windowed
    /// sibling. Verify takes them for every block. Prefill takes them only
    /// once the chunk crosses [`windowed_q8_rows_flash`]; before that it keeps
    /// the per-row causal kernel, and where no WMMA kernel serves, the scalar
    /// key-tiled flash kernel on the rebased origin. A verify block without
    /// WMMA keeps the causal kernel.
    fn attend_windowed_q8_rows(
        &self,
        gpu: &mut Gpu,
        route: RowsRoute,
        rows_flash: Option<WindowedQ8RowsFlash<'_>>,
        wmma_flash: bool,
    ) -> Result<(), DispatchError> {
        let kv = &self.kv;
        let (n_heads, n_kv, hd, rows) = (
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.stream.rows,
        );
        let (q, kc, vc, out) = (self.q, kv.k_cache, kv.v_cache, self.attn_out);
        let positions = self.positions()?;
        let seq_len = self.position + rows;
        gpu.kv_cache_write_q8_0_batched(kc, self.k, positions, n_kv, hd, rows)
            .map_err(hip)?;
        gpu.kv_cache_write_q8_0_batched(vc, self.v, positions, n_kv, hd, rows)
            .map_err(hip)?;
        let flash = match route {
            RowsRoute::Verify => true,
            // Decode never reaches the rows kernels.
            RowsRoute::Prefill | RowsRoute::Decode => {
                windowed_q8_rows_flash(self.position, rows, kv.window)
            }
        };
        if flash && rows > 1 && wmma_flash {
            if kv.window == 0 {
                return gpu
                    .attention_q8_0_flash_prefill_wmma(
                        q, kc, vc, out, positions, n_heads, n_kv, hd, seq_len, rows,
                    )
                    .map_err(hip);
            }
            if gpu
                .attention_q8_0_flash_prefill_wmma_swa(
                    q, kc, vc, out, positions, n_heads, n_kv, hd, rows, kv.window,
                )
                .map_err(hip)?
            {
                return Ok(());
            }
        }
        if !flash || route == RowsRoute::Verify {
            return gpu
                .attention_q8_0_kv_batched(
                    q,
                    kc,
                    vc,
                    out,
                    positions,
                    n_heads,
                    n_kv,
                    hd,
                    kv.physical_cap,
                    seq_len,
                    rows,
                )
                .map_err(hip);
        }
        let f = rows_flash.ok_or_else(|| {
            DispatchError::Hip("windowed q8 prefill attention needs flash scratch".into())
        })?;
        let origin = windowed_q8_rows_origin(self.position, rows, kv.window);
        let (k_view, v_view);
        let (k, v, pos) = if origin == 0 {
            (kc, vc, positions)
        } else {
            // Q8_0 positions of `n_kv * head_dim / 32` blocks of 34 bytes; a
            // 128-aligned origin starts on an F32 element. The views end at
            // the live extent: VMM caches are accessible only up to the
            // mapped prefix.
            let row_bytes = n_kv * (hd / 32) * 34;
            let off = origin * row_bytes / 4;
            let live = (seq_len - origin) * row_bytes / 4;
            k_view = kc.sub_offset(off, live.min(kc.numel().saturating_sub(off)));
            v_view = vc.sub_offset(off, live.min(vc.numel().saturating_sub(off)));
            (&k_view, &v_view, f.shifted_positions)
        };
        gpu.attention_flash_q8_0_batched_masked_windowed(
            q,
            k,
            v,
            out,
            pos,
            n_heads,
            n_kv,
            hd,
            kv.physical_cap,
            seq_len - origin,
            rows,
            f.partials,
            None,
            0,
            0,
            kv.window as i32,
        )
        .map_err(hip)
    }

    /// Block-diffusion draft attention: the context rows (projected,
    /// cached, copied ahead of the rows and rotated), then every row attends
    /// all `kv_rows` K/V rows on the DFlash family's scalar tile.
    fn attend_draft(
        &self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        kv_rows: usize,
        context: Option<&DraftContext<'_>>,
    ) -> Result<(), DispatchError> {
        let s = &self.stream;
        let (n_heads, n_kv, hd) = (self.n_heads, self.n_kv_heads, self.head_dim);
        let p = self.kv_proj.as_ref().ok_or_else(|| {
            DispatchError::Hip("sandwich draft attention needs its K/V projections".into())
        })?;
        if let Some(c) = context {
            if c.delta > 0 {
                let wv = p.wv.as_ref().ok_or_else(|| {
                    DispatchError::Hip("sandwich draft attention needs a V projection".into())
                })?;
                let ladder = s.ladder();
                let input = GemvInput::Raw(c.input);
                gemm_rows(
                    gpu, ctx, &p.wk, input, c.k_new, c.rot, s.bf16, c.delta, ladder,
                )?;
                gemm_rows(gpu, ctx, wv, input, c.v_new, c.rot, s.bf16, c.delta, ladder)?;
                gpu.rmsnorm_batched(c.k_new, p.k_norm, c.k_new, c.delta * n_kv, hd, s.eps)
                    .map_err(hip)?;
            }
            let n = c.rows * n_kv * hd;
            gpu.copy_f32_buffer(c.k, c.k_window, n).map_err(hip)?;
            gpu.copy_f32_buffer(c.v, c.v_window, n).map_err(hip)?;
            // `n_heads == 0`: K only; Q is never touched.
            if self.rope.kind != RopeKind::NoPe {
                gpu.rope_batched_f32(
                    self.q,
                    c.k,
                    c.positions,
                    0,
                    n_kv,
                    hd,
                    self.rope.theta,
                    c.rows,
                )
                .map_err(hip)?;
            }
        }
        ATTENTION.run_full_attention_on(
            ctx,
            gpu,
            &FullAttnParams {
                key: KernelKey::AttnFullF32,
                q: self.q,
                k: self.kv.k_cache,
                v: self.kv.v_cache,
                out: self.attn_out,
                n: s.rows,
                seq_len: kv_rows,
                n_heads,
                n_kv_heads: n_kv,
                head_dim: hd,
            },
            TileImpl::DflashScalar,
        )
    }
}

pub fn execute_sandwich_attention(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &SandwichAttentionOp<'_>,
) -> Result<(), DispatchError> {
    let s = op.stream;
    s.save_residual_for(gpu, op.post_norm)?;
    execute_sandwich_attention_partial(gpu, ctx, op)?;
    s.post_norm_residual(gpu, s.normed, op.post_norm)
}

/// [`execute_sandwich_attention`] up to the output projection: `Wo·a` lands
/// in `op.stream.normed` and the stream is untouched. A tensor-parallel rank
/// runs this over its heads, so `normed` holds its partial sum; once reduced,
/// [`execute_sandwich_epilogue`] applies the post-norm residual. The post-norm
/// cannot take a partial sum, so this is the only TP split point.
pub fn execute_sandwich_attention_partial(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &SandwichAttentionOp<'_>,
) -> Result<(), DispatchError> {
    let s = op.stream;
    op.project(gpu, ctx)?;
    op.position_encode(gpu)?;
    op.attend(gpu, ctx)?;
    if let Some(gate) = &op.gate {
        gpu.sigmoid_mul_f32(op.attn_out, gate.out).map_err(hip)?;
    }
    if s.batched() {
        let input = GemvInput::Raw(op.attn_out);
        gemm_rows(
            gpu,
            ctx,
            &op.wo,
            input,
            s.normed,
            op.x_rot,
            s.bf16,
            s.rows,
            s.ladder(),
        )
    } else {
        gemv(gpu, ctx, &op.wo, op.attn_out, s.normed)
    }
}

/// The sandwich sublayer epilogue over a reduced sublayer output `out`:
/// `x + post_norm(out)` into the stream ([`SandwichStream::out`] when set),
/// the same kernels the whole sublayer ends with. Only a stream with an
/// output stream is admitted: the in-place form reads the residual the
/// sublayer saved first.
pub fn execute_sandwich_epilogue(
    gpu: &mut Gpu,
    stream: &SandwichStream<'_>,
    out: &GpuTensor,
    post_norm: Option<&GpuTensor>,
) -> Result<(), DispatchError> {
    const L: &str = "sandwich epilogue";
    stream.validate(L)?;
    need(out, stream.len(), L, "sublayer output")?;
    if stream.out.is_none() {
        return Err(invalid(format!("{L}: needs an output stream")));
    }
    stream.post_norm_residual(gpu, out, post_norm)
}

/// Gated MLP activation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    /// `gelu_pytorch_tanh(gate) * up`.
    GeluTanh,
    /// `silu(gate) * up` (SwiGLU).
    Silu,
}

/// MLP sublayer: `x = residual + post_norm(down(act(gate(n)) * up(n)))`,
/// `n = pre_norm(x)`; without a post-norm the plain pre-norm residual.
pub struct SandwichMlpOp<'a> {
    pub stream: SandwichStream<'a>,
    pub pre_norm: &'a GpuTensor,
    pub w_gate: WeightRef<'a>,
    pub w_up: WeightRef<'a>,
    pub w_down: WeightRef<'a>,
    pub activation: Activation,
    pub hidden_dim: usize,
    pub post_norm: Option<&'a GpuTensor>,
    /// FWHT rotation scratch, `[rows, max(hidden, hidden_dim)]`.
    pub x_rot: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub act: &'a GpuTensor,
    pub out: &'a GpuTensor,
}

impl SandwichMlpOp<'_> {
    /// Checks route/rows consistency and extents; launches nothing.
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        const L: &str = "sandwich mlp";
        let s = &self.stream;
        s.validate(L)?;
        let inner = s.rows * self.hidden_dim;
        need(self.gate, inner, L, "gate")?;
        need(self.up, inner, L, "up")?;
        need(self.act, inner, L, "activation")?;
        need(self.out, s.len(), L, "output")?;
        need_projection_scratch(s, self.x_rot, &[&self.w_gate, &self.w_up, &self.w_down], L)
    }
}

pub fn execute_sandwich_mlp(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &SandwichMlpOp<'_>,
) -> Result<(), DispatchError> {
    op.stream.save_residual_for(gpu, op.post_norm)?;
    dense_mlp(gpu, ctx, op)?;
    op.stream.post_norm_residual(gpu, op.out, op.post_norm)
}

/// [`execute_sandwich_mlp`] up to the down projection: `out = down(..)`, the
/// stream untouched. A tensor-parallel rank runs this over its intermediate
/// slice (a partial sum in `op.out`), reduced before
/// [`execute_sandwich_epilogue`].
pub fn execute_sandwich_mlp_partial(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &SandwichMlpOp<'_>,
) -> Result<(), DispatchError> {
    dense_mlp(gpu, ctx, op)
}

/// `out = down(act(gate(n)) * up(n))`, `n = pre_norm(x)`.
fn dense_mlp(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &SandwichMlpOp<'_>,
) -> Result<(), DispatchError> {
    let s = op.stream;
    let rows = s.rows;
    if let Some(route) = s.steps_route() {
        return steps_mlp(gpu, ctx, op, route);
    }
    if s.batched() {
        s.norm(gpu, s.x, op.pre_norm, s.normed)?;
        if q8_fused_prefill(gpu, &[&op.w_gate, &op.w_up]) {
            fused_projection(
                gpu,
                ctx,
                KernelKey::FusedGateUpQ8_0,
                &[&op.w_gate, &op.w_up],
                s.normed,
                &[op.gate, op.up],
                rows,
            )?;
        } else {
            gemm_rows_wide(gpu, ctx, &op.w_gate, s.normed, op.gate, op.x_rot, rows)?;
            gemm_rows_wide(gpu, ctx, &op.w_up, s.normed, op.up, op.x_rot, rows)?;
        }
    } else if op.w_gate.dtype == DType::MQ4G256
        && op.w_up.dtype == DType::MQ4G256
        && !calibrating(gpu)
    {
        gpu.fused_rmsnorm_rotate_mq(s.x, op.pre_norm, op.x_rot, s.hidden, s.eps)
            .map_err(hip)?;
        gpu.fused_gate_up_hfq4g256(
            op.w_gate.buf,
            op.w_up.buf,
            op.x_rot,
            op.gate,
            op.up,
            op.w_gate.m,
            op.w_up.m,
            op.w_gate.k,
        )
        .map_err(hip)?;
    } else {
        s.norm(gpu, s.x, op.pre_norm, s.normed)?;
        gemv(gpu, ctx, &op.w_gate, s.normed, op.gate)?;
        gemv(gpu, ctx, &op.w_up, s.normed, op.up)?;
    }
    let act = activate(gpu, ctx, op, false)?;
    if !s.batched() {
        gemv(gpu, ctx, &op.w_down, act, op.out)?;
    } else {
        gemm_rows_wide(gpu, ctx, &op.w_down, act, op.out, op.x_rot, rows)?;
    }
    Ok(())
}

/// [`dense_mlp`] on the steps route.
fn steps_mlp(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &SandwichMlpOp<'_>,
    route: StepsRoute,
) -> Result<(), DispatchError> {
    let s = op.stream;
    let group = [(&op.w_gate, op.gate), (&op.w_up, op.up)];
    project_group(gpu, ctx, &s, route, op.pre_norm, op.x_rot, &group)?;
    let down = &op.w_down;
    let rotate = route.fused_silu_rotate
        && matches!(down.dtype, DType::MQ4G256 | DType::MQ6G256)
        && down.k.is_multiple_of(256);
    let act = activate(gpu, ctx, op, rotate)?;
    let input = if rotate {
        GemvInput::Prerotated(act)
    } else {
        GemvInput::Raw(act)
    };
    match (s.batched(), input) {
        (true, _) => gemm_rows(
            gpu,
            ctx,
            down,
            input,
            op.out,
            op.x_rot,
            s.bf16,
            s.rows,
            s.ladder(),
        ),
        (false, GemvInput::Raw(x)) => gemv(gpu, ctx, down, x, op.out),
        (false, GemvInput::Prerotated(x)) => gemv_prerotated(gpu, ctx, down, x, op.out),
    }
}

/// `act = act(gate) * up`; with `rotate`, FWHT-rotated (and AWQ-divided) for
/// the down projection.
fn activate<'a>(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &SandwichMlpOp<'a>,
    rotate: bool,
) -> Result<&'a GpuTensor, DispatchError> {
    let rows = op.stream.rows;
    match op.activation {
        Activation::GeluTanh => {
            gpu.gelu_tanh_f32(op.gate, op.act, rows * op.hidden_dim)
                .map_err(hip)?;
            gpu.mul_f32(op.act, op.up, op.act).map_err(hip)?;
        }
        Activation::Silu if rotate => ROTATION
            .run(
                ctx,
                gpu,
                RotationParams {
                    x: op.gate,
                    x_up: Some(op.up),
                    w_norm: None,
                    x_plain: op.act,
                    x_rot: op.act,
                    awq_scale: op.w_down.awq_scale,
                    k: op.w_down.k,
                    eps: 0.0,
                    batch_size: if op.stream.batched() { rows } else { 1 },
                    variant: RotationVariant::WithSwiGLU,
                    givens_pairs: None,
                    givens_theta: None,
                    givens_scales: None,
                    givens_krot: None,
                    givens_group_size: None,
                },
            )
            .map_err(hip_dbg)?,
        Activation::Silu => gpu.silu_mul_f32(op.gate, op.up, op.act).map_err(hip)?,
    }
    Ok(op.act)
}

/// `pre_norm(x)` into every projection of `group` on the steps route.
///
/// Decode runs the norm and the projections as one generic step window, so
/// the dispatch fusion table groups same-format projections of one rotated
/// input (QKVZA, gate/up). Batched rows norm once (fused with the shared
/// FWHT rotation when it applies) and project each weight on the route's
/// ladder; on gfx12 a prefill chunk's four MQ4/HFQ4 projections of one
/// rotated input run as one fused QKVZA WMMA GEMM (2.65x the four GEMMs at
/// 256 rows on gfx1201).
fn project_group(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    s: &SandwichStream<'_>,
    route: StepsRoute,
    norm: &GpuTensor,
    x_rot: &GpuTensor,
    group: &[(&WeightRef<'_>, &GpuTensor)],
) -> Result<(), DispatchError> {
    let first = group[0].0;
    let shared =
        route.shared_rotation && dtype_rotation_plan(first.dtype) == RotationPlan::FwhtG256;
    if !s.batched() {
        let (out, input, rotation) = if shared {
            (x_rot, GemvInput::Prerotated(x_rot), RotationPlan::FwhtG256)
        } else {
            (s.normed, GemvInput::Raw(s.normed), RotationPlan::None)
        };
        let mut steps = vec![Step::RmsnormAutomatic {
            x: s.x,
            norm_weight: norm,
            x_plain: s.normed,
            out,
            awq_scale: if shared { first.awq_scale } else { None },
            k: s.hidden,
            eps: s.eps,
            rotation,
            rows: 1,
        }];
        steps.extend(group.iter().map(|&(w, out)| Step::Gemv { w, input, out }));
        return execute_steps(gpu, ctx, &steps);
    }
    let rows = s.rows;
    let input = if shared {
        match first.awq_scale {
            None => gpu.fused_rmsnorm_rotate_mq_batched(s.x, norm, x_rot, s.hidden, s.eps, rows),
            Some(awq) => gpu
                .fused_rmsnorm_rotate_mq_awq_batched(s.x, norm, awq, x_rot, s.hidden, s.eps, rows),
        }
        .map_err(hip)?;
        GemvInput::Prerotated(x_rot)
    } else {
        s.norm(gpu, s.x, norm, s.normed)?;
        GemvInput::Raw(s.normed)
    };
    if let ([q, k, v, g], GemvInput::Prerotated(x)) = (group, input) {
        let ws = [q.0, k.0, v.0, g.0];
        if gfx12_residual(gpu, s.ladder(), q.0.dtype, rows)
            && ws.iter().all(|w| w.dtype == q.0.dtype && w.k == q.0.k)
        {
            gpu.invalidate_fp16_cache();
            return FUSED_QKV.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: KernelKey::FusedQkvzaHfq4G256,
                    weights: &ws.map(|w| w.buf),
                    x,
                    outputs: &[q.1, k.1, v.1, g.1],
                    m: &ws.map(|w| w.m),
                    k: q.0.k,
                    rot_scratch: &[],
                    batch_size: Some(rows),
                },
            );
        }
    }
    for &(w, out) in group {
        gemm_rows(gpu, ctx, w, input, out, x_rot, s.bf16, rows, s.ladder())?;
    }
    Ok(())
}

/// Routed-expert half of a parallel dense+MoE block.
pub struct RoutedExperts<'a> {
    /// Norm of the residual stream feeding the experts.
    pub pre_norm: &'a GpuTensor,
    /// Router input norm weight; the normed input is scaled by
    /// `router_input_scale` before the router projection.
    pub router_norm: &'a GpuTensor,
    pub router_input_scale: f32,
    pub router: WeightRef<'a>,
    pub n_experts: usize,
    pub top_k: usize,
    /// Per-expert FFN width.
    pub hidden_dim: usize,
    pub gate_up_dtype: DType,
    pub down_dtype: DType,
    /// `[n_experts]` device pointers to each expert's fused gate/up weight.
    pub gate_up_ptrs: &'a GpuTensor,
    /// `[n_experts]` device pointers to each expert's down weight.
    pub down_ptrs: &'a GpuTensor,
    /// `[n_experts]` learned scale applied to each expert's output.
    pub per_expert_scale: &'a GpuTensor,
    /// Norm of the combined expert output.
    pub post_norm: &'a GpuTensor,
}

impl RoutedExperts<'_> {
    /// Expert formats with an indexed device-resident kernel pair.
    pub fn supports(gate_up: DType, down: DType) -> bool {
        matches!(
            gate_up,
            DType::MQ4G256
                | DType::MQ4G256V2
                | DType::MQ6G256
                | DType::HFQ4G256
                | DType::HFQ6G256
                | DType::Q8_0
        ) && matches!(down, DType::Q8_0 | DType::HFQ4G128)
    }

    /// Formats whose batched rows take the grouped (sorted-by-expert) gate/up
    /// GEMM: each expert's weights are read once per batch instead of once
    /// per routed row.
    pub fn grouped_rows(gate_up: DType, down: DType) -> bool {
        matches!(gate_up, DType::MQ4G256V2 | DType::MQ6G256) && down == DType::HFQ4G128
    }
}

/// Batched-rows scratch of the grouped routed-expert route (`rows` rows,
/// `slots = rows * top_k`, `m_total = slots + n_experts * MOE_GROUPED_BLOCK_M`).
/// Index buffers hold i32.
#[derive(Clone, Copy)]
pub struct RoutedBatch<'a> {
    /// `[rows, hidden]`: router input, then `pre_norm(residual)`.
    pub input: &'a GpuTensor,
    /// `[rows, hidden]` FWHT rotation (router `x_rot` scratch first).
    pub input_rot: &'a GpuTensor,
    /// `[rows, n_experts]`.
    pub router_logits: &'a GpuTensor,
    /// `[slots]` each.
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub inverse_perm: &'a GpuTensor,
    /// `[n_experts]`, `[n_experts + 1]`.
    pub counts: &'a GpuTensor,
    pub offsets: &'a GpuTensor,
    /// `[m_total]`, `[m_total / MOE_GROUPED_BLOCK_M]`.
    pub sorted_slots: &'a GpuTensor,
    pub tile_ids: &'a GpuTensor,
    /// `[m_total, max(2 * hidden_dim, hidden)]`: grouped gate/up rows, then
    /// grouped down rows.
    pub gate_up_grouped: &'a GpuTensor,
    /// `[slots, hidden_dim]` each; `gate` ends as the activation.
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    /// `[rows, hidden]` combined expert output.
    pub out: &'a GpuTensor,
}

impl RoutedBatch<'_> {
    /// Grouped rows (with padding) for `rows` rows: a whole number of
    /// 16-slot tiles, because the grouped kernels launch `ceil(m_total / 16)`
    /// tiles and read one tile id each.
    pub fn m_total(rows: usize, top_k: usize, n_experts: usize) -> usize {
        let block = crate::families::moe::MOE_GROUPED_BLOCK_M;
        (rows * top_k).next_multiple_of(block) + n_experts * block
    }
}

/// Scratch of one routed-expert pass.
#[derive(Clone, Copy)]
pub struct RoutedScratch<'a> {
    /// `pre_norm(residual)`, and its FWHT rotation for MagnumQuant experts.
    pub input: &'a GpuTensor,
    pub input_rot: &'a GpuTensor,
    pub router_in: &'a GpuTensor,
    pub router_logits: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    /// `[top_k, hidden_dim]` each.
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub act: &'a GpuTensor,
    /// Weighted sum of the selected experts.
    pub out: &'a GpuTensor,
    /// `dense_post_norm(dense MLP output)`.
    pub dense_normed: &'a GpuTensor,
    /// Grouped-route scratch for batched rows; `None` routes row by row.
    pub batch: Option<RoutedBatch<'a>>,
}

/// Parallel dense+MoE MLP sublayer:
/// `x = residual + post_norm(dense_post_norm(mlp(x)) + experts.post_norm(moe(r)))`
/// with `r = experts.pre_norm(x)`; the router reads
/// `router_norm(x) * router_input_scale`. `mlp` is the dense half: it reads
/// the block's residual stream (`mlp.x` is `stream.x`, its pre-norm writes
/// `stream.normed`) and writes its own output (`MlpOutput::Write`).
pub struct ParallelMoeMlpOp<'a> {
    pub stream: SandwichStream<'a>,
    pub mlp: GatedMlpOp<'a>,
    pub dense_post_norm: &'a GpuTensor,
    /// The block's final post-norm.
    pub post_norm: &'a GpuTensor,
    pub experts: RoutedExperts<'a>,
    pub scratch: RoutedScratch<'a>,
}

impl ParallelMoeMlpOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        self.validate_layout()
    }

    /// Everything the block refuses, checked before its first launch.
    fn validate_layout(&self) -> Result<(), DispatchError> {
        let (s, mlp, e) = (&self.stream, &self.mlp, &self.experts);
        if !matches!(mlp.output, MlpOutput::Write(_)) {
            return Err(DispatchError::Hip(
                "parallel MoE MLP: the dense half must write its own output".into(),
            ));
        }
        // One residual stream, described twice: the dense half must read the
        // tensor the experts read through `stream.residual`.
        let same = |a: &GpuTensor, b: &GpuTensor| {
            a.buf.as_ptr() == b.buf.as_ptr() && a.numel() == b.numel()
        };
        if mlp.rows != s.rows
            || mlp.hidden != s.hidden
            || !same(mlp.x, s.x)
            || !mlp.pre_norm.is_some_and(|n| same(n.normed, s.normed))
        {
            return Err(DispatchError::Hip(
                "parallel MoE MLP: the dense half must read the block's stream (x, normed, rows, hidden)"
                    .into(),
            ));
        }
        if e.top_k != 8 {
            return Err(DispatchError::Hip(format!(
                "parallel MoE MLP: top_k={} unsupported (indexed kernels are k=8)",
                e.top_k
            )));
        }
        if !RoutedExperts::supports(e.gate_up_dtype, e.down_dtype) {
            return Err(DispatchError::Hip(format!(
                "parallel MoE MLP: expert formats gate_up={:?} down={:?} have no indexed kernel",
                e.gate_up_dtype, e.down_dtype
            )));
        }
        mlp.validate_layout()
    }
}

pub fn execute_parallel_moe_mlp(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &ParallelMoeMlpOp<'_>,
) -> Result<(), DispatchError> {
    let s = op.stream;
    let e = &op.experts;
    let r = &op.scratch;
    op.validate_layout()?;
    let MlpOutput::Write(dense_out) = op.mlp.output else {
        unreachable!("validate_layout admits only a written dense output")
    };
    s.save_residual(gpu)?;
    execute_gated_mlp(gpu, ctx, &op.mlp)?;
    if !s.batched() {
        s.norm(gpu, dense_out, op.dense_post_norm, r.dense_normed)?;
        routed_row(gpu, ctx, e, r, s.residual, s.hidden, s.eps)?;
        gpu.add_f32(r.dense_normed, r.out, s.normed).map_err(hip)?;
        s.norm(gpu, s.normed, op.post_norm, s.normed)?;
    } else {
        s.norm(gpu, dense_out, op.dense_post_norm, dense_out)?;
        match r.batch.filter(|_| {
            gpu.arch_caps.is_gfx1151() && RoutedExperts::grouped_rows(e.gate_up_dtype, e.down_dtype)
        }) {
            Some(b) => {
                routed_rows_grouped(gpu, ctx, e, &b, s)?;
                gpu.add_inplace_f32(dense_out, b.out).map_err(hip)?;
            }
            // The indexed expert kernels are single-row: route row by row
            // and accumulate into the normed dense output.
            None => {
                for row in 0..s.rows {
                    let residual = s.residual.sub_offset(row * s.hidden, s.hidden);
                    routed_row(gpu, ctx, e, r, &residual, s.hidden, s.eps)?;
                    let out = dense_out.sub_offset(row * s.hidden, s.hidden);
                    gpu.add_inplace_f32(&out, r.out).map_err(hip)?;
                }
            }
        }
        s.norm(gpu, dense_out, op.post_norm, s.normed)?;
    }
    s.restore_residual(gpu)?;
    gpu.add_inplace_f32(s.x, s.normed).map_err(hip)
}

/// `b.out = experts.post_norm(moe(pre_norm(residual)))` for all `s.rows`
/// rows: batched router, then the routed slots sorted by expert so each
/// expert's gate/up and down weights are read once per 16-slot tile.
fn routed_rows_grouped(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    e: &RoutedExperts<'_>,
    b: &RoutedBatch<'_>,
    s: SandwichStream<'_>,
) -> Result<(), DispatchError> {
    use crate::families::moe::MOE_GROUPED_BLOCK_M;
    let (rows, hidden, mi, k) = (s.rows, s.hidden, e.hidden_dim, e.top_k);
    gpu.rmsnorm_batched(s.residual, e.router_norm, b.input, rows, hidden, s.eps)
        .map_err(hip)?;
    scale(gpu, b.input, e.router_input_scale)?;
    // Expert selection is tie-sensitive: the F16-input Q8 WMMA GEMM moves
    // router logits by ~1e-3, which flips near-tied experts and costs
    // prefill KL 0.16 -> 1.19 on 26B-A4B. The batched Q8 GEMV keeps F32
    // activations and shares each weight load across eight rows.
    if e.router.dtype == DType::Q8_0 {
        gpu.gemm_q8_0_batched_wide_exact(
            e.router.buf,
            b.input,
            b.router_logits,
            e.router.m,
            e.router.k,
            rows,
        )
        .map_err(hip)?;
    } else {
        gemm_rows_wide(
            gpu,
            ctx,
            &e.router,
            b.input,
            b.router_logits,
            b.input_rot,
            rows,
        )?;
    }
    gpu.moe_softmax_topk_renorm_k8_batched(
        b.router_logits,
        b.topk_indices,
        b.topk_weights,
        e.n_experts,
        true,
        rows,
    )
    .map_err(hip)?;

    gpu.rmsnorm_batched(s.residual, e.pre_norm, b.input, rows, hidden, s.eps)
        .map_err(hip)?;
    gpu.rotate_x_mq_batched(b.input, b.input_rot, hidden, rows)
        .map_err(hip)?;
    let m_total = RoutedBatch::m_total(rows, k, e.n_experts);
    gpu.moe_scatter_fused_k8(
        b.topk_indices,
        b.counts,
        b.offsets,
        b.sorted_slots,
        b.tile_ids,
        b.inverse_perm,
        rows * k,
        e.n_experts,
        m_total,
        MOE_GROUPED_BLOCK_M,
    )
    .map_err(hip)?;
    super::dispatch_grouped_gemm(
        gpu,
        e.gate_up_dtype,
        None,
        e.gate_up_ptrs,
        b.tile_ids,
        b.sorted_slots,
        b.input_rot,
        b.gate_up_grouped,
        2 * mi,
        hidden,
        k,
        m_total,
        rows,
        false,
        false,
        false,
        // Gemma 4 experts are device-resident: no host-mapped coalesced read.
        false,
    )?;
    gpu.moe_gate_up_unscatter_k8(
        b.gate_up_grouped,
        b.sorted_slots,
        b.gate,
        b.up,
        mi,
        k,
        m_total,
    )
    .map_err(hip)?;
    gpu.gelu_tanh_f32(b.gate, b.gate, rows * k * mi)
        .map_err(hip)?;
    gpu.mul_f32(b.gate, b.up, b.gate).map_err(hip)?;
    gpu.gemv_hfq4g128_moe_down_grouped(
        e.down_ptrs,
        b.tile_ids,
        b.sorted_slots,
        e.per_expert_scale,
        b.gate,
        b.gate_up_grouped,
        hidden,
        mi,
        m_total,
    )
    .map_err(hip)?;
    gpu.zero_f32(b.out).map_err(hip)?;
    gpu.moe_down_combine_grouped_k8(
        b.gate_up_grouped,
        b.inverse_perm,
        b.topk_weights,
        b.out,
        hidden,
        k,
        rows,
    )
    .map_err(hip)?;
    gpu.rmsnorm_batched(b.out, e.post_norm, b.out, rows, hidden, s.eps)
        .map_err(hip)
}

/// `r.out = experts.post_norm(moe(pre_norm(residual)))` for one row; the
/// router reads `router_norm(residual) * router_input_scale`.
fn routed_row(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    e: &RoutedExperts<'_>,
    r: &RoutedScratch<'_>,
    residual: &GpuTensor,
    hidden: usize,
    eps: f32,
) -> Result<(), DispatchError> {
    gpu.rmsnorm_f32(residual, e.pre_norm, r.input, eps)
        .map_err(hip)?;
    gpu.rmsnorm_f32(residual, e.router_norm, r.router_in, eps)
        .map_err(hip)?;
    scale(gpu, r.router_in, e.router_input_scale)?;
    gemv(gpu, ctx, &e.router, r.router_in, r.router_logits)?;
    gpu.moe_softmax_topk_renorm_k8(
        r.router_logits,
        r.topk_indices,
        r.topk_weights,
        e.n_experts,
        true,
    )
    .map_err(hip)?;

    // Q8 down accumulates atomically; HFQ4G128 down assigns.
    gpu.zero_f32(r.out).map_err(hip)?;
    let (mi, k) = (e.hidden_dim, e.top_k);
    match e.gate_up_dtype {
        DType::MQ4G256 => {
            gpu.rotate_x_mq(r.input, r.input_rot, hidden).map_err(hip)?;
            gpu.gemv_mq4g256_moe_gate_up_k8_indexed(
                e.gate_up_ptrs,
                r.topk_indices,
                r.input_rot,
                r.gate,
                r.up,
                2 * mi,
                hidden,
            )
            .map_err(hip)?;
        }
        // FWHT-rotated input; MQ6G256 reads it with the HFQ6G256 kernel.
        DType::MQ4G256V2 | DType::MQ6G256 => {
            gpu.rotate_x_mq(r.input, r.input_rot, hidden).map_err(hip)?;
            crate::pipeline::run_uniform_moe_gate_up(
                gpu,
                e.gate_up_dtype,
                e.gate_up_ptrs,
                r.topk_indices,
                r.input_rot,
                r.gate,
                r.up,
                2 * mi,
                hidden,
                k,
            )?
        }
        DType::HFQ4G256 | DType::HFQ6G256 => crate::pipeline::run_uniform_moe_gate_up(
            gpu,
            e.gate_up_dtype,
            e.gate_up_ptrs,
            r.topk_indices,
            r.input,
            r.gate,
            r.up,
            2 * mi,
            hidden,
            k,
        )?,
        _ => gpu
            .gemv_q8_0_moe_gate_up_k8_indexed(
                e.gate_up_ptrs,
                r.topk_indices,
                r.input,
                r.gate,
                r.up,
                2 * mi,
                hidden,
            )
            .map_err(hip)?,
    }
    gpu.gelu_tanh_f32(r.gate, r.act, k * mi).map_err(hip)?;
    gpu.mul_f32(r.act, r.up, r.act).map_err(hip)?;
    if e.down_dtype == DType::Q8_0 {
        gpu.gemv_q8_0_moe_down_residual_scaled_k8_indexed(
            e.down_ptrs,
            r.topk_indices,
            r.topk_weights,
            e.per_expert_scale,
            r.act,
            r.out,
            hidden,
            mi,
        )
    } else {
        gpu.gemv_hfq4g128_moe_down_residual_scaled_k8_indexed(
            e.down_ptrs,
            r.topk_indices,
            r.topk_weights,
            e.per_expert_scale,
            r.act,
            r.out,
            hidden,
            mi,
        )
    }
    .map_err(hip)?;
    gpu.rmsnorm_f32(r.out, e.post_norm, r.out, eps).map_err(hip)
}

/// Per-layer-input branch:
/// `x = residual + post_norm(proj(gelu(gate(x)) * inputs[row, layer]))`.
/// `inputs` holds `layer_width`-wide slices for `n_layers` layers per row.
pub struct PerLayerInputOp<'a> {
    pub stream: SandwichStream<'a>,
    pub layer: usize,
    pub layer_width: usize,
    pub n_layers: usize,
    pub inputs: &'a GpuTensor,
    pub w_gate: WeightRef<'a>,
    pub w_proj: WeightRef<'a>,
    pub post_norm: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub act: &'a GpuTensor,
    pub out: &'a GpuTensor,
}

/// Q8 projections whose batched GEMM reproduces the single-row GEMV exactly.
fn exact_wide_q8_gemm(dtype: DType, k: usize) -> bool {
    dtype == DType::Q8_0 && k > 0 && k <= 1536 && k.is_multiple_of(32)
}

pub fn execute_per_layer_input(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &PerLayerInputOp<'_>,
) -> Result<(), DispatchError> {
    let s = op.stream;
    let (rows, width, hidden) = (s.rows, op.layer_width, s.hidden);
    let packed = op.n_layers * width;
    s.save_residual(gpu)?;
    if !s.batched() {
        let layer_input = op.inputs.sub_offset(op.layer * width, width);
        gemv(gpu, ctx, &op.w_gate, s.x, op.gate)?;
        gpu.gelu_tanh_f32(op.gate, op.act, width).map_err(hip)?;
        gpu.mul_f32(op.act, &layer_input, op.act).map_err(hip)?;
        gemv(gpu, ctx, &op.w_proj, op.act, op.out)?;
    } else {
        let exact = batched_fusion_arch(&gpu.arch) && gpu.flags.gemma4_ple_branch_batched_prefill;
        let wide = |gpu: &mut Gpu, w: &WeightRef, x: &GpuTensor, y: &GpuTensor| {
            GEMM.run_key(
                KernelKey::GemmQ8_0BatchedWideExact,
                ctx,
                gpu,
                &GemmParams {
                    w,
                    x,
                    y,
                    batch_size: rows,
                },
            )
        };
        if exact && exact_wide_q8_gemm(op.w_gate.dtype, op.w_gate.k) {
            wide(gpu, &op.w_gate, s.x, op.gate)?;
        } else {
            for row in 0..rows {
                let x_row = s.x.sub_offset(row * hidden, hidden);
                let gate_row = op.gate.sub_offset(row * width, width);
                gemv(gpu, ctx, &op.w_gate, &x_row, &gate_row)?;
            }
        }
        if batched_fusion_arch(&gpu.arch) && gpu.flags.gemma4_ple_activation_fused_prefill {
            gpu.gemma4_ple_gelu_mul_strided_f32(
                op.gate, op.inputs, op.act, rows, width, packed, op.layer,
            )
            .map_err(hip_dbg)?;
        } else {
            gpu.gelu_tanh_f32(op.gate, op.act, rows * width)
                .map_err(hip)?;
            for row in 0..rows {
                let act_row = op.act.sub_offset(row * width, width);
                let layer_input = op.inputs.sub_offset(row * packed + op.layer * width, width);
                gpu.mul_f32(&act_row, &layer_input, &act_row).map_err(hip)?;
            }
        }
        if exact && exact_wide_q8_gemm(op.w_proj.dtype, op.w_proj.k) {
            wide(gpu, &op.w_proj, op.act, op.out)?;
        } else {
            for row in 0..rows {
                let act_row = op.act.sub_offset(row * width, width);
                let out_row = op.out.sub_offset(row * hidden, hidden);
                gemv(gpu, ctx, &op.w_proj, &act_row, &out_row)?;
            }
        }
    }
    s.norm(gpu, op.out, op.post_norm, s.normed)?;
    s.restore_residual(gpu)?;
    gpu.add_inplace_f32(s.x, s.normed).map_err(hip)
}

/// In-place `x *= factor`.
pub struct ScaleOp<'a> {
    pub x: &'a GpuTensor,
    pub factor: f32,
}

pub fn execute_scale(gpu: &mut Gpu, op: &ScaleOp<'_>) -> Result<(), DispatchError> {
    scale(gpu, op.x, op.factor)
}

/// In-place final-logit soft cap: `x = tanh(x / cap) * cap`.
pub struct SoftcapOp<'a> {
    pub logits: &'a GpuTensor,
    pub n: usize,
    pub cap: f32,
}

pub fn execute_softcap(gpu: &mut Gpu, op: &SoftcapOp<'_>) -> Result<(), DispatchError> {
    gpu.logit_softcap_f32(op.logits, op.n, op.cap).map_err(hip)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::gated_mlp::{Activation, GatedMlpScratch, MlpLadder, MlpStorage, RmsNorm};

    /// A launch-free F32 tensor of `elements` at `addr` (never dereferenced).
    fn tensor(elements: usize, addr: usize) -> GpuTensor {
        let mut t = GpuTensor::null_for_test();
        t.buf = unsafe { DeviceBuffer::from_raw(addr as *mut std::ffi::c_void, elements * 4) };
        t.shape = vec![elements];
        t
    }

    fn weight(buf: &GpuTensor, m: usize, k: usize) -> WeightRef<'_> {
        WeightRef {
            buf,
            dtype: DType::Q8_0,
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

    /// The Gemma 26B-A4B binding shape: every operand is `t` except the
    /// stream `x` and `normed`, which the dense half shares with the block.
    fn op<'a>(t: &'a GpuTensor, x: &'a GpuTensor, normed: &'a GpuTensor) -> ParallelMoeMlpOp<'a> {
        ParallelMoeMlpOp {
            stream: SandwichStream {
                rows: 2,
                hidden: 16,
                eps: 1e-6,
                post_eps: 1e-6,
                x,
                residual: t,
                normed,
                out: None,
                route: RowsRoute::Prefill,
                kernels: SandwichRoute::Wide,
                bf16: None,
            },
            mlp: GatedMlpOp {
                rows: 2,
                hidden: 16,
                intermediate: 8,
                x,
                pre_norm: Some(RmsNorm {
                    weight: t,
                    eps: 1e-6,
                    normed,
                }),
                gate: weight(t, 8, 16),
                up: weight(t, 8, 16),
                down: weight(t, 16, 8),
                activation: Activation::GeluTanh,
                storage: MlpStorage::F32,
                output: MlpOutput::Write(t),
                ladder: MlpLadder::SandwichWide,
                scratch: GatedMlpScratch {
                    gate: t,
                    up: t,
                    act: t,
                    x_rot: t,
                },
            },
            dense_post_norm: t,
            post_norm: t,
            experts: RoutedExperts {
                pre_norm: t,
                router_norm: t,
                router_input_scale: 1.0,
                router: weight(t, 4, 16),
                n_experts: 4,
                top_k: 8,
                hidden_dim: 8,
                gate_up_dtype: DType::Q8_0,
                down_dtype: DType::Q8_0,
                gate_up_ptrs: t,
                down_ptrs: t,
                per_expert_scale: t,
                post_norm: t,
            },
            scratch: RoutedScratch {
                input: t,
                input_rot: t,
                router_in: t,
                router_logits: t,
                topk_indices: t,
                topk_weights: t,
                gate: t,
                up: t,
                act: t,
                out: t,
                dense_normed: t,
                batch: None,
            },
        }
    }

    #[test]
    fn parallel_moe_preflight_covers_the_dense_half() {
        let (t, x, normed) = (tensor(1024, 0x1000), tensor(32, 0x2000), tensor(32, 0x3000));
        assert!(op(&t, &x, &normed).validate_layout().is_ok());
        let refusal = |o: &ParallelMoeMlpOp<'_>| o.validate_layout().unwrap_err().to_string();

        // A dense half the GatedMlp preflight refuses is refused by the block,
        // before `execute_parallel_moe_mlp` saves the residual.
        let mut bad = op(&t, &x, &normed);
        bad.mlp.down = weight(&t, 17, 8);
        assert!(
            refusal(&bad).contains("gated MLP down"),
            "{}",
            refusal(&bad)
        );

        // The dense half must read the block's own stream.
        let other = tensor(32, 0x4000);
        let short = tensor(31, 0x2000);
        let other_norm = Some(RmsNorm {
            weight: &t,
            eps: 1e-6,
            normed: &other,
        });
        let base = || op(&t, &x, &normed);
        let cases = [
            GatedMlpOp {
                x: &other,
                ..base().mlp
            },
            GatedMlpOp {
                x: &short,
                ..base().mlp
            },
            GatedMlpOp {
                pre_norm: other_norm,
                ..base().mlp
            },
            GatedMlpOp {
                pre_norm: None,
                ..base().mlp
            },
            GatedMlpOp {
                rows: 1,
                ..base().mlp
            },
            GatedMlpOp {
                hidden: 8,
                ..base().mlp
            },
        ];
        for mlp in cases {
            let message = refusal(&ParallelMoeMlpOp { mlp, ..base() });
            assert!(message.contains("block's stream"), "{message}");
        }

        let mut bad = op(&t, &x, &normed);
        bad.mlp.output = MlpOutput::PostNorm {
            norm: &t,
            eps: 1e-6,
            out: &t,
            residual: &t,
        };
        assert!(refusal(&bad).contains("write its own output"));
    }

    #[test]
    fn batched_projection_formats_are_explicit() {
        for dtype in [
            DType::F32,
            DType::Q8_0,
            DType::MQ4G256,
            DType::HFQ4G256,
            DType::MQ6G256,
            DType::MQ4G256V2,
        ] {
            assert!(supports_batched_projection(dtype));
        }
        for dtype in [
            DType::HFQ4G128,
            DType::HFQ6G256,
            DType::HFQ2G256,
            DType::HFQ3G256,
            DType::MQ3G256,
        ] {
            assert!(!supports_batched_projection(dtype));
        }
    }

    #[test]
    fn exact_wide_q8_gemm_matches_the_wide_gemv_boundary() {
        assert!(exact_wide_q8_gemm(DType::Q8_0, 1536));
        assert!(!exact_wide_q8_gemm(DType::Q8_0, 1535));
        assert!(!exact_wide_q8_gemm(DType::Q8_0, 1537));
        assert!(!exact_wide_q8_gemm(DType::Q8_0, 0));
        assert!(!exact_wide_q8_gemm(DType::F32, 256));
    }
}

#[cfg(test)]
mod windowed_q8_rows_tests {
    use super::{windowed_q8_rows_flash, windowed_q8_rows_origin};

    #[test]
    fn prefill_flash_starts_past_the_window_and_origin_keeps_every_in_window_key() {
        // Sliding layers: flash once the chunk ends past the window; full
        // layers past 2048 (the batched causal kernel's LDS ceiling).
        assert!(!windowed_q8_rows_flash(1856, 192, 2048));
        assert!(windowed_q8_rows_flash(1857, 192, 2048));
        assert!(!windowed_q8_rows_flash(2047, 1, 0));
        assert!(windowed_q8_rows_flash(2048, 1, 0));
        assert_eq!(windowed_q8_rows_origin(1857, 192, 2048), 0);
        assert_eq!(windowed_q8_rows_origin(4000, 256, 0), 0);
        for (position, rows) in [(2048, 1), (2259, 1), (2251, 192), (5000, 512), (8000, 7)] {
            let origin = windowed_q8_rows_origin(position, rows, 2048);
            // Tile-aligned, and at or below the first row's lowest live key.
            assert_eq!(origin % 128, 0);
            assert!(origin <= position + 1 - 2048, "{position}+{rows}: {origin}");
            assert!(origin + 128 > position + 1 - 2048);
        }
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    const DIM: usize = 256;
    const CAP: usize = 64;

    /// A host-only tensor view: validation reads extents, never memory.
    fn tensor(elems: usize) -> GpuTensor {
        GpuTensor {
            buf: unsafe { DeviceBuffer::from_raw(std::ptr::dangling_mut(), elems * 4) },
            shape: vec![elems],
            dtype: DType::F32,
        }
    }

    fn weight(t: &GpuTensor) -> WeightRef<'_> {
        WeightRef {
            buf: t,
            dtype: DType::Q8_0,
            m: DIM,
            k: DIM,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        }
    }

    /// Validates a one-head windowed-Q8 attention of `rows` rows at
    /// `position` on `route` / `kernels`, with scratch for 16 rows.
    fn check(
        route: RowsRoute,
        kernels: SandwichRoute,
        position: usize,
        rows: usize,
    ) -> Result<(), DispatchError> {
        let big = tensor(16 * DIM * CAP);
        let pos_buf = unsafe { DeviceBuffer::from_raw(std::ptr::dangling_mut(), 4) };
        let op = SandwichAttentionOp {
            stream: SandwichStream {
                rows,
                hidden: DIM,
                eps: 1e-6,
                post_eps: 1e-6,
                x: &big,
                residual: &big,
                normed: &big,
                out: None,
                route,
                kernels,
                bf16: None,
            },
            position,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: DIM,
            input_norm: &big,
            wq: weight(&big),
            q_norm: &big,
            kv_proj: Some(KvProjection {
                wk: weight(&big),
                wv: Some(weight(&big)),
                k_norm: &big,
                v_norm: None,
            }),
            q_scale: 1.0,
            rope: Rope {
                kind: RopeKind::NoPe,
                theta: 1.0,
            },
            kv: SandwichKv {
                k_cache: &big,
                v_cache: &big,
                physical_cap: CAP,
                window: 0,
                route: KvRoute::WindowedQ8 {
                    tiled: None,
                    rows_flash: None,
                    wmma_flash: false,
                },
            },
            pos_buf: &pos_buf,
            attend_through: None,
            positions: Some(&big),
            gate: None,
            wo: weight(&big),
            post_norm: None,
            x_rot: &big,
            q: &big,
            k: &big,
            v: &big,
            attn_out: &big,
        };
        op.validate()
    }

    const STEPS: SandwichRoute = SandwichRoute::Steps(StepsRoute {
        shared_rotation: false,
        fused_silu_rotate: false,
    });

    fn refused(r: Result<(), DispatchError>, why: &str) {
        match r {
            Err(DispatchError::Hip(m)) => assert!(m.contains(why), "{m}"),
            other => panic!("expected a refusal containing {why:?}, got {other:?}"),
        }
    }

    #[test]
    fn decode_route_refuses_more_than_one_row() {
        assert!(check(RowsRoute::Decode, STEPS, 8, 1).is_ok());
        refused(
            check(RowsRoute::Decode, STEPS, 8, 4),
            "4 rows on the decode route",
        );
    }

    #[test]
    fn rows_past_the_cache_are_refused() {
        assert!(check(RowsRoute::Prefill, STEPS, CAP - 4, 4).is_ok());
        refused(
            check(RowsRoute::Prefill, STEPS, CAP - 4, 5),
            "outside a cache of 64 rows",
        );
        refused(check(RowsRoute::Decode, STEPS, CAP, 1), "outside a cache");
    }

    #[test]
    fn windowed_q8_needs_the_steps_route() {
        refused(
            check(RowsRoute::Decode, SandwichRoute::Wide, 8, 1),
            "windowed Q8 KV needs the steps route",
        );
    }
}
