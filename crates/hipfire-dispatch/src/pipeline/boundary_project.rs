// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Projections at a BF16 output boundary (`Step::BoundaryProject`, #666 G18).
//!
//! A [`BoundaryProjectOp`] runs the projections of one F32 input plane
//! `[rows, k]`, each output optionally rounded to BF16 (kept as an F32 plane,
//! [`OutRound::Bf16`]) by the exact bit-level round trip
//! (`Gpu::bf16_round_trip_f32_strided`) or a GEMM epilogue bit-identical to it.
//! Raon-1-mini's reference model is BF16, so every GEMM output of its layer
//! program crosses this boundary.
//!
//! The explicit [`RowsRoute`] selects the kernels; the row count never does,
//! except inside a route (the MQ8 verify GEMV over 1..=2 rows):
//!
//! * `Decode` (one row): the decode GEMV (`gemv_q8_0` / `gemv_f32` /
//!   `gemv_mq8g256_b_x` / FWHT + prerotated qt44 GEMV).
//! * `Prefill`: the batched GEMMs (Q8_0 at the site's activation tier through
//!   [`q8_0_prefill_gemm_prepared`], qt44 rotate + `gemm_mq4g256v2_batched_lmhead`,
//!   `gemm_f32_batched`, MQ8 prefill GEMM).
//! * `Verify`: every row bit-identical to the decode step on that row (Q8_0
//!   `gemv_q8_0_rows_exact`, MQ8 row-exact GEMVs, qt44 / F32 the decode GEMV per row).
//!
//! [`ProjectShare`] lets the projections of one node share their input
//! preparation: one A8 (or MQ8 rotated int8) quantization, or one F16 copy for
//! the F16-WMMA Q8_0 GEMM. Both are bit-identical to each projection preparing
//! its own input.
//!
//! Supported weight dtypes: `MQ4G256V2` (qt44, optional AWQ sidecar), `Q8_0`,
//! `MQ8G256` (device layout B) and `F32`; anything else is a named refusal.
//! `HIPFIRE_RAON_A8_REUSE` / `HIPFIRE_RAON_A8_FUSE_ROUND` switch the shared
//! quantization and the fused round off (bit-identical either way).

use std::sync::LazyLock;

use crate::context::DispatchCtx;
use crate::families::gemv::{GemvFamily, GemvParams, WeightRef};
use crate::pipeline::batched::rotate_x_mq_batched_for;
use crate::pipeline::{
    mq8_acts_current, mq8_gemv_fused, mq8_gemv_rows, mq8_prefill_gemm, mq8_quantize,
    q8_0_f16_x_shareable, q8_0_prefill_gemm_f16x, q8_0_prefill_gemm_prepared, q8_0_prefill_prepare,
    PrefillActTier, PreparedA8, RowsRoute,
};
use crate::types::{DispatchError, GemvVariant};
use rdna_compute::{mq8g256_b_bytes, DType, Gpu, GpuTensor, Mq8Activations};

/// Output rounding applied after a projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutRound {
    /// Leave the F32 result as produced (FP32-boundary consumers).
    Exact,
    /// Round every output element to BF16.
    Bf16,
}

/// How the projections of one op share their input plane.
pub enum ProjectShare {
    /// Each projection prepares its own input.
    Own,
    /// One A8 Q8_0 quantization (or, for MQ8G256 weights, one rotated int8
    /// plane) at `tier`, prepared for the first projection's weight, serves
    /// every projection that can consume it.
    A8 { tier: PrefillActTier },
    /// One F16 copy of the input in `plane` serves every projection, when all
    /// of them take the F16-WMMA Q8_0 GEMM at `tier`; otherwise each prepares
    /// its own. `plane` is a non-owning view of at least `rows * k` F16.
    F16 {
        plane: GpuTensor,
        tier: PrefillActTier,
    },
}

/// One weight of a [`BoundaryProjectOp`] and where its `[rows, m]` output goes.
#[derive(Clone, Copy)]
pub struct BoundaryProjection<'a> {
    pub weight: WeightRef<'a>,
    pub output: &'a GpuTensor,
    pub round: OutRound,
    /// The site's resolved prefill activation tier (Q8_0 batched GEMM only).
    pub tier: PrefillActTier,
}

/// The projections of one F32 input plane `[rows, k]`, in order.
pub struct BoundaryProjectOp<'a> {
    pub input: &'a GpuTensor,
    pub rows: usize,
    pub route: RowsRoute,
    pub share: ProjectShare,
    pub projections: Vec<BoundaryProjection<'a>>,
}

fn err(msg: String) -> DispatchError {
    DispatchError::Hip(msg)
}

static GEMV: LazyLock<GemvFamily> = LazyLock::new(GemvFamily::new);

fn gemv() -> &'static GemvFamily {
    &GEMV
}

/// The GEMV form of `w`: no PARO rotation, AWQ or row stride (the activation
/// is rotated and AWQ-divided before a prerotated GEMV, never inside it).
fn plain<'a>(w: &WeightRef<'a>) -> WeightRef<'a> {
    WeightRef {
        buf: w.buf,
        dtype: w.dtype,
        m: w.m,
        k: w.k,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: w.lloyd_lut_e4m3,
        lloyd_lut_f16: w.lloyd_lut_f16,
        lloyd_lut_c16: w.lloyd_lut_c16,
    }
}

fn check(
    w: &WeightRef<'_>,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    what: &str,
) -> Result<(), DispatchError> {
    if rows == 0 {
        return Err(err(format!("raon projection {what}: zero rows")));
    }
    if x.dtype != DType::F32 || y.dtype != DType::F32 {
        return Err(err(format!(
            "raon projection {what}: activations must be F32 planes"
        )));
    }
    let need_x = rows
        .checked_mul(w.k)
        .ok_or_else(|| err(format!("raon projection {what}: input extent overflows")))?;
    let need_y = rows
        .checked_mul(w.m)
        .ok_or_else(|| err(format!("raon projection {what}: output extent overflows")))?;
    if x.numel() < need_x || y.numel() < need_y {
        return Err(err(format!(
            "raon projection {what}: buffers too small (x {} < {need_x} or y {} < {need_y}) for [{rows} x {}] -> [{rows} x {}]",
            x.numel(),
            y.numel(),
            w.k,
            w.m
        )));
    }
    match w.dtype {
        DType::MQ4G256V2 | DType::Q8_0 | DType::F32 => Ok(()),
        DType::MQ8G256 => {
            if w.k == 0 || w.k % 256 != 0 || w.buf.buf.size() < mq8g256_b_bytes(w.m, w.k) {
                return Err(err(format!(
                    "raon projection {what}: MQ8G256 weight [{} x {}] needs K % 256 == 0 and {} layout-B bytes (buffer {})",
                    w.m,
                    w.k,
                    mq8g256_b_bytes(w.m, w.k),
                    w.buf.buf.size()
                )));
            }
            Ok(())
        }
        other => Err(err(format!(
            "raon projection {what}: weight dtype {other:?} is not routed (expected qt44 / Q8F16 / MQ8G256 / F32)"
        ))),
    }
}

impl BoundaryProjectOp<'_> {
    /// [`Self::validate`] as the step-program preflight.
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        self.validate()
    }

    /// Route/row consistency and every projection's operands; launch-free.
    pub fn validate(&self) -> Result<(), DispatchError> {
        let what = match self.route {
            RowsRoute::Decode if self.rows != 1 => {
                return Err(err(format!(
                    "raon projection: a decode route over {} rows",
                    self.rows
                )))
            }
            RowsRoute::Prefill | RowsRoute::Verify if self.rows < 2 => {
                return Err(err(format!(
                    "raon projection: a batched route over {} rows",
                    self.rows
                )))
            }
            RowsRoute::Decode => "row",
            _ => "rows",
        };
        let Some(first) = self.projections.first() else {
            return Err(err("raon projection: no projections".into()));
        };
        for p in &self.projections {
            check(&p.weight, self.input, p.output, self.rows, what)?;
        }
        if !matches!(self.share, ProjectShare::Own)
            && self
                .projections
                .iter()
                .any(|p| p.weight.k != first.weight.k)
        {
            return Err(err(
                "raon projection: a shared input needs one K across its projections".into(),
            ));
        }
        if let ProjectShare::F16 { plane, .. } = &self.share {
            let need = self.rows * first.weight.k;
            if plane.dtype != DType::F16 || plane.numel() < need {
                return Err(err(format!(
                    "raon projection: shared F16 plane holds {} {:?}, {need} F16 needed",
                    plane.numel(),
                    plane.dtype
                )));
            }
        }
        Ok(())
    }
}

/// Validate `op`, then run its projections in order on its route.
pub fn execute_boundary_project(
    gpu: &mut Gpu,
    op: &BoundaryProjectOp<'_>,
) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    let (x, rows, route) = (op.input, op.rows, op.route);
    match &op.share {
        ProjectShare::Own => {
            for p in &op.projections {
                project(gpu, &p.weight, x, p.output, rows, route, p.round, p.tier)?;
            }
        }
        ProjectShare::A8 { tier } => {
            let prepared = prepare_a8(gpu, &op.projections[0].weight, x, rows, route, *tier)?;
            for p in &op.projections {
                project_prepared(
                    gpu,
                    &p.weight,
                    x,
                    p.output,
                    rows,
                    route,
                    p.round,
                    p.tier,
                    prepared.as_ref(),
                )?;
            }
        }
        ProjectShare::F16 { plane, tier } => {
            let ws: Vec<&WeightRef<'_>> = op.projections.iter().map(|p| &p.weight).collect();
            let x16 = share_f16_x(gpu, x, plane, rows, route, *tier, &ws)?;
            for p in &op.projections {
                project_shared_f16(
                    gpu, &p.weight, x, x16, p.output, rows, route, p.round, p.tier,
                )?;
            }
        }
    }
    Ok(())
}

// ───────────────────────── decode / batched GEMV and GEMM ─────────────────────────
//
// Copies of the arms of `hipfire_runtime::llama::{weight_gemv, weight_gemv_rows_exact,
// weight_gemm}` Raon reaches (dispatch cannot call hipfire-runtime): a change to those
// launch sequences must land here too, or Raon's projections drift from the shared ones.

/// The decode GEMV `y[m] = W x[k]` of one row (no rounding). Safe inside graph
/// capture: device-resident operands, the Gpu-owned rotation scratch.
/// Mirrors `hipfire_runtime::llama::weight_gemv` (non-rotated and qt44 arms).
fn weight_gemv(
    gpu: &mut Gpu,
    w: &WeightRef<'_>,
    x: &GpuTensor,
    y: &GpuTensor,
) -> Result<(), DispatchError> {
    // Calibration tap: PRE-rotation input x (n=1, k=w.k). No-op when unarmed.
    gpu.maybe_capture_activation(w.buf, x, 1, w.k);
    let ctx = DispatchCtx::new(gpu);
    let wr = plain(w);
    match w.dtype {
        DType::MQ4G256V2 => {
            gpu.ensure_mq_signs().map_err(|e| err(e.to_string()))?;
            let scratch = gpu.scratch.mq_x_rot.as_ref().expect("mq signs ensured");
            let xr = GpuTensor {
                buf: unsafe { scratch.buf.alias() },
                shape: vec![scratch.buf.size() / 4],
                dtype: DType::F32,
            };
            match w.awq_scale {
                Some(awq) => gpu.rotate_x_mq_awq(x, awq, &xr, w.k),
                None => gpu.rotate_x_mq(x, &xr, w.k),
            }
            .map_err(|e| err(e.to_string()))?;
            // `xr` is already FWHT-rotated: the Prerotated GEMV, never `run_auto`
            // (which would rotate a second time).
            gemv().run(
                &ctx,
                gpu,
                &GemvParams {
                    w: &wr,
                    x: &xr,
                    y,
                    variant: GemvVariant::Prerotated,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
        }
        _ => gemv().run_auto(&ctx, gpu, &wr, x, y),
    }
}

/// `rows` decode GEMVs on row views: exact by construction, the weight re-read per row.
fn per_row_decode(
    gpu: &mut Gpu,
    w: &WeightRef<'_>,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
) -> Result<(), DispatchError> {
    for r in 0..rows {
        let x_row = x.sub_offset(r * w.k, w.k);
        let y_row = y.sub_offset(r * w.m, w.m);
        weight_gemv(gpu, w, &x_row, &y_row)?;
    }
    Ok(())
}

/// The batched qt44 GEMM: FWHT-rotate (AWQ-aware) every row into a pooled
/// plane, then the INT4-G256 v2 WMMA GEMM. Allocates: eager prefill only.
fn mq4v2_gemm(
    gpu: &mut Gpu,
    w: &WeightRef<'_>,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
) -> Result<(), DispatchError> {
    gpu.maybe_capture_activation(w.buf, x, rows, w.k);
    let hip = |e: hip_bridge::HipError| err(format!("{e:?}"));
    gpu.ensure_mq_signs().map_err(hip)?;
    let x_rot = gpu.alloc_tensor(&[rows, w.k], DType::F32).map_err(hip)?;
    let r = rotate_x_mq_batched_for(gpu, w, x, &x_rot, w.k, rows)
        .and_then(|()| gpu.gemm_mq4g256v2_batched_lmhead(w.buf, &x_rot, y, w.m, w.k, rows));
    let freed = gpu.free_tensor(x_rot);
    r.map_err(hip)?;
    freed.map_err(hip)
}

fn round_bf16_rows(
    gpu: &mut Gpu,
    t: &GpuTensor,
    rows: usize,
    width: usize,
) -> Result<(), DispatchError> {
    gpu.bf16_round_trip_f32_strided(t, rows, 0, width, width)
        .map_err(|e| err(format!("raon bf16 round: {e:?}")))
}

fn is_mq8(w: &WeightRef<'_>) -> bool {
    w.dtype == DType::MQ8G256
}

/// `HIPFIRE_RAON_A8_REUSE=0` restores one activation quantization per projection (the
/// pre-reuse behavior). Bit-identical either way. Read once.
fn a8_reuse_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| hipfire_config::developer_bool("HIPFIRE_RAON_A8_REUSE", true));
    *ON
}

/// `HIPFIRE_RAON_A8_FUSE_ROUND=0` runs the projection's BF16 output round as its own pass
/// (`bf16_round_trip_f32_strided`) instead of the GEMM epilogue (Q8_0: the W8A8 i8-MMQ `_bf16`
/// entries on gfx1201 / gfx1100 and the F16 wmma `gemm_q8_0_wmma_bf16` on gfx1100; MQ8G256: the
/// prefill GEMM's BF16 epilogue); bit-identical either way. Read once.
fn a8_fuse_round_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_bool("HIPFIRE_RAON_A8_FUSE_ROUND", true)
    });
    *ON
}

/// An activation plane quantized once for the projections of one node.
#[derive(Clone, Copy, Debug)]
enum PreparedAct {
    Q8(PreparedA8),
    /// `x` is the source plane's device address (a different plane is never served).
    Mq8 {
        acts: Mq8Activations,
        x: *const std::ffi::c_void,
    },
}

// ───────────────────────── routes ─────────────────────────

/// One projection on its route, preparing its own input.
#[allow(clippy::too_many_arguments)]
fn project(
    gpu: &mut Gpu,
    w: &WeightRef<'_>,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    route: RowsRoute,
    round: OutRound,
    tier: PrefillActTier,
) -> Result<(), DispatchError> {
    if route == RowsRoute::Decode {
        return project_row(gpu, w, x, y, round);
    }
    if is_mq8(w) {
        return mq8_project(gpu, w, x, y, rows, round, None, route == RowsRoute::Verify);
    }
    if route == RowsRoute::Prefill {
        return project_batched(gpu, w, x, y, rows, round, tier, None);
    }
    match w.dtype {
        // Row-exact verify: qt44 and F32 have no multi-row kernel with the decode
        // reduction order (WMMA F16 x for qt44, a 32-lane tree for F32 vs the 256-thread
        // `gemv_f32`), so each row runs the decode projection itself.
        DType::MQ4G256V2 | DType::F32 => per_row_decode(gpu, w, x, y, rows),
        DType::Q8_0 => {
            let ctx = DispatchCtx::new(gpu);
            gemv().run_rows_exact(&ctx, gpu, &plain(w), x, y, rows)
        }
        other => unreachable!("dtype {other:?} rejected by check() or routed to MQ8"),
    }
    .map_err(|e| err(format!("raon projection (rows={rows}): {e}")))?;
    if round == OutRound::Bf16 {
        round_bf16_rows(gpu, y, rows, w.m)?;
    }
    Ok(())
}

/// The decode route of one row.
fn project_row(
    gpu: &mut Gpu,
    w: &WeightRef<'_>,
    x: &GpuTensor,
    y: &GpuTensor,
    round: OutRound,
) -> Result<(), DispatchError> {
    if is_mq8(w) {
        return mq8_project(gpu, w, x, y, 1, round, None, true);
    }
    weight_gemv(gpu, w, x, y).map_err(|e| err(format!("raon projection (row): {e:?}")))?;
    if round == OutRound::Bf16 {
        round_bf16_rows(gpu, y, 1, w.m)?;
    }
    Ok(())
}

/// Quantize `x[rows, w.k]` once for the projections of a node that read it (`w` is any one
/// consumer: its `k` and dtype are the shared contract). Q8_0: the A8 quantization on a
/// batched route. MQ8G256: the rotated int8 plane on any route but decode and the 2-row
/// verify (which fuse the rotation into the GEMV), whatever `tier` is. `None` (nothing
/// launched) when the reuse switch is off, the weight is neither, or the Q8_0 quantized route
/// is not admitted at this `tier`/`rows`; every consumer then takes its ordinary route. A stale
/// result is detected and ignored by the consumers.
fn prepare_a8(
    gpu: &mut Gpu,
    w: &WeightRef<'_>,
    x: &GpuTensor,
    rows: usize,
    route: RowsRoute,
    tier: PrefillActTier,
) -> Result<Option<PreparedAct>, DispatchError> {
    if !a8_reuse_enabled() {
        return Ok(None);
    }
    match w.dtype {
        DType::MQ8G256 => {
            if route == RowsRoute::Decode || (rows == 2 && route == RowsRoute::Verify) {
                return Ok(None);
            }
            let acts = mq8_quantize(gpu, x, rows, w.k)
                .map_err(|e| err(format!("raon MQ8 prepare (rows={rows}): {e}")))?;
            Ok(Some(PreparedAct::Mq8 {
                acts,
                x: x.buf.as_ptr() as *const std::ffi::c_void,
            }))
        }
        DType::Q8_0 if route != RowsRoute::Decode => {
            q8_0_prefill_prepare(gpu, tier, x, w.m, w.k, rows)
                .map(|p| p.map(PreparedAct::Q8))
                .map_err(|e| err(format!("raon A8 prepare (rows={rows}): {e}")))
        }
        _ => Ok(None),
    }
}

/// The batched route, consuming `prepared` on the A8 Q8_0 GEMM instead of quantizing `x`
/// again and, for `OutRound::Bf16`, rounding in the GEMM epilogue. Bit-identical to the
/// unprepared, unfused route.
#[allow(clippy::too_many_arguments)]
fn project_batched(
    gpu: &mut Gpu,
    w: &WeightRef<'_>,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    round: OutRound,
    tier: PrefillActTier,
    prepared: Option<&PreparedAct>,
) -> Result<(), DispatchError> {
    if is_mq8(w) {
        return mq8_project(gpu, w, x, y, rows, round, prepared, false);
    }
    let prepared = match prepared {
        Some(PreparedAct::Q8(p)) => Some(p),
        _ => None,
    };
    let mut rounded = false;
    match w.dtype {
        DType::MQ4G256V2 => mq4v2_gemm(gpu, w, x, y, rows).map_err(|e| e.to_string()),
        DType::Q8_0 => q8_0_prefill_gemm_prepared(
            gpu,
            tier,
            prepared,
            w.buf,
            x,
            y,
            w.m,
            w.k,
            rows,
            round == OutRound::Bf16 && a8_fuse_round_enabled(),
        )
        .map(|run| rounded = run.bf16_fused)
        .map_err(|e| e.to_string()),
        DType::F32 => gpu
            .gemm_f32_batched(w.buf, x, y, w.m, w.k, rows)
            .map_err(|e| format!("{e:?}")),
        other => unreachable!("dtype {other:?} rejected by check()"),
    }
    .map_err(|e| err(format!("raon projection (rows={rows}): {e}")))?;
    if round == OutRound::Bf16 && !rounded {
        round_bf16_rows(gpu, y, rows, w.m)?;
    }
    Ok(())
}

/// A consumer of a node whose activation was prepared once ([`prepare_a8`]). Verify blocks
/// keep the row-exact route (the A8 tier is never admitted there).
#[allow(clippy::too_many_arguments)]
fn project_prepared(
    gpu: &mut Gpu,
    w: &WeightRef<'_>,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    route: RowsRoute,
    round: OutRound,
    tier: PrefillActTier,
    prepared: Option<&PreparedAct>,
) -> Result<(), DispatchError> {
    if is_mq8(w) {
        return mq8_project(
            gpu,
            w,
            x,
            y,
            rows,
            round,
            prepared,
            route != RowsRoute::Prefill,
        );
    }
    if route == RowsRoute::Verify {
        return project(gpu, w, x, y, rows, route, round, tier);
    }
    if route == RowsRoute::Decode {
        return project_row(gpu, w, x, y, round);
    }
    project_batched(gpu, w, x, y, rows, round, tier, prepared)
}

/// MQ8G256 projection. The activation plane is `prepared` when it still describes `x` (same
/// plane, rows, k, no quantizer ran since), else quantized here. `exact` (decode and verify):
/// the GEMV routes, each row bit-identical to the decode step on that row; otherwise the
/// prefill GEMM, which rounds to BF16 in its epilogue when asked.
#[allow(clippy::too_many_arguments)]
fn mq8_project(
    gpu: &mut Gpu,
    w: &WeightRef<'_>,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    round: OutRound,
    prepared: Option<&PreparedAct>,
    exact: bool,
) -> Result<(), DispatchError> {
    if exact && rows <= 2 {
        // Rotate + quantize fused into the GEMV (one launch, no shared plane).
        mq8_gemv_fused(gpu, w.buf, x, rows, y, w.m, w.k)
            .map_err(|e| err(format!("raon projection (MQ8 fused, rows={rows}): {e}")))?;
        if round == OutRound::Bf16 {
            round_bf16_rows(gpu, y, rows, w.m)?;
        }
        return Ok(());
    }
    let acts = match prepared {
        Some(PreparedAct::Mq8 { acts, x: px })
            if *px == x.buf.as_ptr() as *const std::ffi::c_void
                && acts.rows() == rows
                && acts.k() == w.k
                && mq8_acts_current(gpu, acts) =>
        {
            *acts
        }
        _ => mq8_quantize(gpu, x, rows, w.k)
            .map_err(|e| err(format!("raon projection (MQ8 quantize, rows={rows}): {e}")))?,
    };
    let mut rounded = false;
    if exact {
        mq8_gemv_rows(gpu, w.buf, &acts, y, w.m)
    } else {
        rounded = round == OutRound::Bf16 && a8_fuse_round_enabled();
        mq8_prefill_gemm(gpu, w.buf, &acts, y, w.m, rounded)
    }
    .map_err(|e| err(format!("raon projection (MQ8, rows={rows}): {e}")))?;
    if round == OutRound::Bf16 && !rounded {
        round_bf16_rows(gpu, y, rows, w.m)?;
    }
    Ok(())
}

/// Whether the batched route of `w` is the F16-WMMA Q8_0 GEMM that reads an F16 X it is handed
/// without converting.
fn f16_x_route(
    gpu: &Gpu,
    w: &WeightRef<'_>,
    rows: usize,
    route: RowsRoute,
    tier: PrefillActTier,
) -> bool {
    route == RowsRoute::Prefill
        && w.dtype == DType::Q8_0
        && q8_0_f16_x_shareable(gpu, tier, w.m, w.k, rows)
}

/// Convert `x[rows, k]` to F16 once into `x16` for the F16-tier Q8_0 projections `ws`, and
/// return the plane. `None` (nothing launched) unless every `ws` entry takes the F16-WMMA
/// route with the same `k`. The convert is the launcher's own `convert_f32_to_f16` kernel
/// writing the words the launcher would stage, so every output is bit-identical to the
/// unshared route.
fn share_f16_x<'p>(
    gpu: &mut Gpu,
    x: &GpuTensor,
    x16: &'p GpuTensor,
    rows: usize,
    route: RowsRoute,
    tier: PrefillActTier,
    ws: &[&WeightRef<'_>],
) -> Result<Option<&'p GpuTensor>, DispatchError> {
    let Some(first) = ws.first() else {
        return Ok(None);
    };
    let k = first.k;
    if ws
        .iter()
        .any(|w| w.k != k || !f16_x_route(gpu, w, rows, route, tier))
    {
        return Ok(None);
    }
    let n = rows
        .checked_mul(k)
        .ok_or_else(|| err("raon projection: shared F16 input extent overflows".into()))?;
    if x.dtype != DType::F32 || x16.dtype != DType::F16 || x.numel() < n || x16.numel() < n {
        return Ok(None);
    }
    gpu.convert_f32_to_f16_into(x, x16, n).map_err(|e| {
        err(format!(
            "raon projection (rows={rows}): shared F16 convert: {e:?}"
        ))
    })?;
    Ok(Some(x16))
}

/// A consumer of [`share_f16_x`]: with `Some(x16)` the F16 plane feeds the F16-WMMA GEMM
/// directly (no convert of its own) and the output is rounded exactly as [`project`] rounds
/// it; with `None` this is [`project`].
#[allow(clippy::too_many_arguments)]
fn project_shared_f16(
    gpu: &mut Gpu,
    w: &WeightRef<'_>,
    x: &GpuTensor,
    x16: Option<&GpuTensor>,
    y: &GpuTensor,
    rows: usize,
    route: RowsRoute,
    round: OutRound,
    tier: PrefillActTier,
) -> Result<(), DispatchError> {
    let Some(x16) = x16 else {
        return project(gpu, w, x, y, rows, route, round, tier);
    };
    if !f16_x_route(gpu, w, rows, route, tier) {
        return Err(err(
            "raon projection: a shared F16 input was passed to a projection off the F16-WMMA route"
                .into(),
        ));
    }
    let fused = q8_0_prefill_gemm_f16x(
        gpu,
        w.buf,
        x16,
        y,
        w.m,
        w.k,
        rows,
        round == OutRound::Bf16 && a8_fuse_round_enabled(),
    )
    .map_err(|e| err(format!("raon projection (rows={rows}): {e}")))?;
    if round == OutRound::Bf16 && !fused {
        round_bf16_rows(gpu, y, rows, w.m)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(elements: usize, dtype: DType) -> GpuTensor {
        let mut t = GpuTensor::null_for_test();
        t.shape = vec![elements];
        t.dtype = dtype;
        t
    }

    fn weight(buf: &GpuTensor, dtype: DType, m: usize, k: usize) -> WeightRef<'_> {
        WeightRef {
            buf,
            dtype,
            m,
            k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        }
    }

    fn op<'a>(
        x: &'a GpuTensor,
        rows: usize,
        route: RowsRoute,
        share: ProjectShare,
        projections: Vec<BoundaryProjection<'a>>,
    ) -> BoundaryProjectOp<'a> {
        BoundaryProjectOp {
            input: x,
            rows,
            route,
            share,
            projections,
        }
    }

    fn out<'a>(w: WeightRef<'a>, y: &'a GpuTensor) -> BoundaryProjection<'a> {
        BoundaryProjection {
            weight: w,
            output: y,
            round: OutRound::Bf16,
            tier: PrefillActTier::F16,
        }
    }

    #[test]
    fn route_must_match_the_row_count() {
        let (wb, x, y) = (
            tensor(0, DType::Q8_0),
            tensor(4 * 64, DType::F32),
            tensor(4 * 32, DType::F32),
        );
        let w = weight(&wb, DType::Q8_0, 32, 64);
        let own = || ProjectShare::Own;
        assert!(op(&x, 2, RowsRoute::Decode, own(), vec![out(w, &y)])
            .validate()
            .is_err());
        assert!(op(&x, 1, RowsRoute::Prefill, own(), vec![out(w, &y)])
            .validate()
            .is_err());
        assert!(op(&x, 1, RowsRoute::Verify, own(), vec![out(w, &y)])
            .validate()
            .is_err());
        assert!(op(&x, 1, RowsRoute::Decode, own(), vec![out(w, &y)])
            .validate()
            .is_ok());
        assert!(op(&x, 4, RowsRoute::Prefill, own(), vec![out(w, &y)])
            .validate()
            .is_ok());
        // Four rows of output do not fit five rows.
        assert!(op(&x, 5, RowsRoute::Prefill, own(), vec![out(w, &y)])
            .validate()
            .is_err());
    }

    #[test]
    fn refuses_unrouted_dtypes_and_mixed_shared_k() {
        let (wb, x, y) = (
            tensor(0, DType::BF16),
            tensor(2 * 128, DType::F32),
            tensor(2 * 32, DType::F32),
        );
        let bf16 = weight(&wb, DType::BF16, 32, 64);
        assert!(op(
            &x,
            2,
            RowsRoute::Prefill,
            ProjectShare::Own,
            vec![out(bf16, &y)]
        )
        .validate()
        .is_err());
        let (a, b) = (
            weight(&wb, DType::Q8_0, 32, 64),
            weight(&wb, DType::Q8_0, 32, 128),
        );
        let share = ProjectShare::A8 {
            tier: PrefillActTier::A8,
        };
        assert!(op(
            &x,
            2,
            RowsRoute::Prefill,
            share,
            vec![out(a, &y), out(b, &y)]
        )
        .validate()
        .is_err());
        // Each projection preparing its own input may differ in K.
        assert!(op(
            &x,
            2,
            RowsRoute::Prefill,
            ProjectShare::Own,
            vec![out(a, &y), out(b, &y)]
        )
        .validate()
        .is_ok());
    }

    #[test]
    fn a_shared_f16_plane_must_hold_the_input() {
        let (wb, x, y) = (
            tensor(0, DType::Q8_0),
            tensor(2 * 64, DType::F32),
            tensor(2 * 32, DType::F32),
        );
        let w = weight(&wb, DType::Q8_0, 32, 64);
        let f16 = |n, dtype| ProjectShare::F16 {
            plane: tensor(n, dtype),
            tier: PrefillActTier::F16,
        };
        let run = |share| op(&x, 2, RowsRoute::Prefill, share, vec![out(w, &y)]).validate();
        assert!(run(f16(2 * 64, DType::F16)).is_ok());
        assert!(run(f16(2 * 64 - 1, DType::F16)).is_err());
        assert!(run(f16(2 * 64, DType::F32)).is_err());
    }
}
