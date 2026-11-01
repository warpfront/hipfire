// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Thin Raon projection helpers over the existing dispatch projection
//! machinery. Owner: slice A.
//!
//! No arch-local GEMV/GEMM format dispatch lives here: a one-row call goes
//! through `hipfire_runtime::llama::weight_gemv` (GemvFamily + the
//! AWQ-aware FWHT rotation for qt44), an N-row call through the batched
//! family entry of the weight's dtype. What this module adds is the Raon
//! precision boundary: the reference model is BF16, so every GEMM output is
//! rounded to BF16 (kept as an F32 plane) — [`OutRound::Bf16`] — via the
//! existing exact bit-level round-trip kernel (`Gpu::bf16_round_trip_f32_strided`).
//!
//! Supported weight dtypes: `MQ4G256V2` (qt44, optional AWQ sidecar),
//! `Q8_0` (Q8F16) and `F32`. The loader widens F16/BF16 small tensors to F32,
//! so those never reach this layer; anything else is a named error, never a
//! silent per-token loop.

use hipfire_dispatch::pipeline::{
    q8_0_f16_x_shareable, q8_0_prefill_gemm_f16x, q8_0_prefill_gemm_prepared, q8_0_prefill_prepare,
    PrefillActTier, PreparedA8,
};
use hipfire_runtime::llama::{weight_gemm, weight_gemv, weight_gemv_rows_exact, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Output rounding applied after the projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutRound {
    /// Leave the F32 result as produced (FP32 oracle mode / FP32-boundary consumers).
    Exact,
    /// Round every output element to BF16 (PLAN §1.3 GEMM-output boundary).
    Bf16,
}

fn check(
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    what: &str,
) -> Result<(), String> {
    if rows == 0 {
        return Err(format!("raon projection {what}: zero rows"));
    }
    if x.dtype != DType::F32 || y.dtype != DType::F32 {
        return Err(format!(
            "raon projection {what}: activations must be F32 planes"
        ));
    }
    let need_x = rows
        .checked_mul(w.k)
        .ok_or_else(|| format!("raon projection {what}: input extent overflows"))?;
    let need_y = rows
        .checked_mul(w.m)
        .ok_or_else(|| format!("raon projection {what}: output extent overflows"))?;
    if x.numel() < need_x || y.numel() < need_y {
        return Err(format!(
            "raon projection {what}: buffers too small (x {} < {need_x} or y {} < {need_y}) for [{rows} x {}] -> [{rows} x {}]",
            x.numel(),
            y.numel(),
            w.k,
            w.m
        ));
    }
    match w.gpu_dtype {
        DType::MQ4G256V2 | DType::Q8_0 | DType::F32 => Ok(()),
        other => Err(format!(
            "raon projection {what}: weight dtype {other:?} is not routed (expected qt44 / Q8F16 / F32)"
        )),
    }
}

/// Round `rows * width` contiguous F32 elements of `t` to BF16, in place.
pub fn round_bf16_rows(
    gpu: &mut Gpu,
    t: &GpuTensor,
    rows: usize,
    width: usize,
) -> Result<(), String> {
    gpu.bf16_round_trip_f32_strided(t, rows, 0, width, width)
        .map_err(|e| format!("raon bf16 round: {e:?}"))
}

/// `y[m] = W x[k]` for one row (decode). Safe inside graph capture: no
/// allocation, device-resident operands only.
pub fn project_row(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    round: OutRound,
) -> Result<(), String> {
    check(w, x, y, 1, "row")?;
    weight_gemv(gpu, w, x, y).map_err(|e| format!("raon projection (row): {e:?}"))?;
    if round == OutRound::Bf16 {
        round_bf16_rows(gpu, y, 1, w.m)?;
    }
    Ok(())
}

/// `Y[rows, m] = X[rows, k] W^T`: one batched GEMM (prefill). `rows == 1`
/// takes the decode route so the two paths agree bit for bit on a single row.
///
/// `tier` is the site's resolved activation precision (`precision::RaonPrecisionPlan`): Q8_0
/// weights take [`q8_0_prefill_gemm`] (which runs the F16 batched GEMM whenever the quantized
/// tier is not admitted: rows < 64, graph capture, an arch without a W8A8 Q8_0 route); qt44 and F32 weights ignore it
/// and keep their F16/F32 routes. Pass `F16` for sites that are never tiered.
pub fn project_rows(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    round: OutRound,
    tier: PrefillActTier,
) -> Result<(), String> {
    check(w, x, y, rows, "rows")?;
    if rows == 1 {
        return project_row(gpu, w, x, y, round);
    }
    if !rows_exact(&gpu.arch, rows) {
        return project_rows_batched(gpu, w, x, y, rows, round, tier);
    }
    match w.gpu_dtype {
        // Row-exact verify: qt44 and F32 have no multi-row kernel with the decode
        // reduction order (WMMA F16 x for qt44, a 32-lane tree for F32 vs the 256-thread
        // `gemv_f32`), so each row runs the decode projection itself.
        DType::MQ4G256V2 | DType::F32 => per_row_decode(gpu, w, x, y, rows),
        DType::Q8_0 => weight_gemv_rows_exact(gpu, w, x, y, rows).map_err(|e| e.to_string()),
        other => unreachable!("dtype {other:?} rejected by check()"),
    }
    .map_err(|e| format!("raon projection (rows={rows}): {e}"))?;
    if round == OutRound::Bf16 {
        round_bf16_rows(gpu, y, rows, w.m)?;
    }
    Ok(())
}

/// [`project_rows`] on the batched GEMM routes whatever the row-exact knob says:
/// for rows no decode step ever produces (the drafter's head over its draft
/// rows), where the multi-row GEMM is the cheap route and exactness buys nothing.
pub fn project_rows_batched(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    round: OutRound,
    tier: PrefillActTier,
) -> Result<(), String> {
    project_batched(gpu, w, x, y, rows, round, tier, None)
}

/// `HIPFIRE_RAON_A8_REUSE=0` restores one activation quantization per A8 projection call (the
/// pre-reuse behavior); anything else, or unset, shares one quantization per node. Read once.
fn a8_reuse_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| hipfire_config::developer_bool("HIPFIRE_RAON_A8_REUSE", true));
    *ON
}

/// Quantize the activation plane `x[rows, w.k]` once for every A8 Q8_0 projection of a node that
/// reads it (RDA q/k/v/g, MLA q_a/kv_a/g); `w` is any one consumer (its `k` and dtype are the
/// shared contract). `None` (nothing launched) when the reuse switch is off, the weight is not
/// Q8_0, or the quantized route is not admitted at this `tier`/`rows`, in which case every consumer
/// takes its ordinary route. The result is valid for one node: pass it to [`project_prepared`] for
/// consumers of the same `x` before anything else quantizes (a stale value is detected and ignored).
pub fn prepare_a8(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    rows: usize,
    tier: PrefillActTier,
) -> Result<Option<PreparedA8>, String> {
    if rows <= 1 || w.gpu_dtype != DType::Q8_0 || !a8_reuse_enabled() {
        return Ok(None);
    }
    q8_0_prefill_prepare(gpu, tier, x, w.m, w.k, rows)
        .map_err(|e| format!("raon A8 prepare (rows={rows}): {e}"))
}

/// [`project_rows_batched`] whose A8 Q8_0 route consumes `prepared` (see [`prepare_a8`]) instead of
/// quantizing `x` again and, for `OutRound::Bf16`, rounds in the GEMM epilogue. Bit-identical to
/// [`project_rows`]: the same quantizer output feeds the same GEMM, the same RN-even round.
#[allow(clippy::too_many_arguments)]
fn project_batched(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    round: OutRound,
    tier: PrefillActTier,
    prepared: Option<&PreparedA8>,
) -> Result<(), String> {
    check(w, x, y, rows, "rows")?;
    if rows == 1 {
        return project_row(gpu, w, x, y, round);
    }
    let mut rounded = false;
    match w.gpu_dtype {
        DType::MQ4G256V2 => weight_gemm(gpu, w, x, y, rows).map_err(|e| format!("{e:?}")),
        DType::Q8_0 => q8_0_prefill_gemm_prepared(
            gpu,
            tier,
            prepared,
            &w.buf,
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
            .gemm_f32_batched(&w.buf, x, y, w.m, w.k, rows)
            .map_err(|e| format!("{e:?}")),
        other => unreachable!("dtype {other:?} rejected by check()"),
    }
    .map_err(|e| format!("raon projection (rows={rows}): {e}"))?;
    if round == OutRound::Bf16 && !rounded {
        round_bf16_rows(gpu, y, rows, w.m)?;
    }
    Ok(())
}

/// `HIPFIRE_RAON_A8_FUSE_ROUND=0` runs the Q8_0 projection's BF16 output round as its own pass
/// (`bf16_round_trip_f32_strided`) instead of the GEMM epilogue (the W8A8 i8-MMQ `_bf16` entries on
/// gfx1201 / gfx1100, the F16 wmma `gemm_q8_0_wmma_bf16` on gfx1100); bit-identical either way. Read once.
fn a8_fuse_round_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_bool("HIPFIRE_RAON_A8_FUSE_ROUND", true)
    });
    *ON
}

/// [`project_rows`] for a consumer of a node whose activation was prepared once ([`prepare_a8`]).
/// Row-exact verify blocks keep [`project_rows`]' exact route (the A8 tier is never admitted there).
#[allow(clippy::too_many_arguments)]
pub fn project_prepared(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    round: OutRound,
    tier: PrefillActTier,
    prepared: Option<&PreparedA8>,
) -> Result<(), String> {
    if rows_exact(&gpu.arch, rows) {
        return project_rows(gpu, w, x, y, rows, round, tier);
    }
    project_batched(gpu, w, x, y, rows, round, tier, prepared)
}

/// `rows` launches of the decode projection (`weight_gemv`) on row views: exact by
/// construction, the weight re-read per row.
fn per_row_decode(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
) -> Result<(), String> {
    for r in 0..rows {
        let x_row = x.sub_offset(r * w.k, w.k);
        let y_row = y.sub_offset(r * w.m, w.m);
        weight_gemv(gpu, w, &x_row, &y_row).map_err(|e| format!("{e:?}"))?;
    }
    Ok(())
}

/// Widest block the row-exact verify route takes (a DFlash2 window).
pub const ROWS_EXACT_MAX: usize = 16;

/// Blocks up to these rows take the row-exact route by default (`auto`): the
/// widest row count at which it is no slower than the batched route, from the
/// verify cost tables (`raon_dflash_oracle costs`, graph decode, beta
/// 365862651c; receipts `raon/dflash2/evidence/b37` R9700, `b34x` / `b38x`
/// 7900 XTX). R9700 2/3/4/5 rows: 11.68/13.24/15.08/17.12 ms row-exact vs
/// 13.97/14.99/15.99/16.87 batched; 7900 XTX 2/9/10 rows: 10.90/22.47/24.65 vs
/// 18.36/23.87/24.38. Unlisted arches are unmeasured: batched.
const ROWS_EXACT_AUTO: &[(&str, usize)] = &[("gfx1201", 4), ("gfx1100", 9)];

/// `HIPFIRE_RAON_VERIFY_ROWEXACT`: unset / `auto` = [`ROWS_EXACT_AUTO`] for this
/// arch; `1` = every block of `2..=ROWS_EXACT_MAX` rows; `0` = never.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowsExactMode {
    Auto,
    On,
    Off,
}

static ROWS_EXACT_MODE: std::sync::LazyLock<RowsExactMode> = std::sync::LazyLock::new(|| {
    match hipfire_config::developer_var("HIPFIRE_RAON_VERIFY_ROWEXACT")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .as_deref()
    {
        None | Some("") | Some("auto") => RowsExactMode::Auto,
        Some("1" | "on" | "true") => RowsExactMode::On,
        Some("0" | "off" | "false") => RowsExactMode::Off,
        Some(other) => {
            eprintln!(
                "raon1_mini: HIPFIRE_RAON_VERIFY_ROWEXACT={other} is not auto|0|1; using auto"
            );
            RowsExactMode::Auto
        }
    }
});

/// Widest block that takes the row-exact route on `arch` (0: none).
pub fn rows_exact_max(arch: &str) -> usize {
    match *ROWS_EXACT_MODE {
        RowsExactMode::Off => 0,
        RowsExactMode::On => ROWS_EXACT_MAX,
        RowsExactMode::Auto => ROWS_EXACT_AUTO
            .iter()
            .find(|(a, _)| *a == arch)
            .map_or(0, |&(_, max)| max),
    }
}

/// Whether a forward of `rows` rows on `arch` runs the row-exact multi-row
/// route: every row bit-identical to the decode step's single row (Q8_0:
/// `gemv_q8_0_rows_exact`, MoE `RowsExact`, MLA `DecodeExact`, dense layer 0
/// rows-exact, F32/qt44 projections per row). The decision depends on `rows`
/// and the arch only, so every op of one forward agrees.
pub fn rows_exact(arch: &str, rows: usize) -> bool {
    rows >= 2 && rows <= rows_exact_max(arch)
}

/// Dispatch to [`project_row`] / [`project_rows`] by row count.
pub fn project(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    round: OutRound,
    tier: PrefillActTier,
) -> Result<(), String> {
    project_rows(gpu, w, x, y, rows, round, tier)
}

/// Whether [`project`] on `w` takes the batched F16-WMMA Q8_0 route that reads an F16 X it is
/// handed without converting (see [`share_f16_x`]).
fn f16_x_route(gpu: &Gpu, w: &WeightTensor, rows: usize, tier: PrefillActTier) -> bool {
    rows >= 2
        && !rows_exact(&gpu.arch, rows)
        && w.gpu_dtype == DType::Q8_0
        && q8_0_f16_x_shareable(gpu, tier, w.m, w.k, rows)
}

/// Convert the F32 activation plane `x[rows, k]` to F16 once into `x16` for the F16-tier Q8_0
/// projections `ws` that read it, and return the plane to pass to [`project_shared_f16`].
/// `None` (nothing launched) unless *every* `ws` entry takes the batched F16-WMMA route
/// ([`project`] would run `gemm_q8_0_batched_chunked` for each) with the same `k`, so a `None`
/// leaves each projection on its ordinary route. The convert is the launcher's own
/// `convert_f32_to_f16` kernel writing the words the launcher would stage, so every output is
/// bit-identical to the unshared route; `x16` is overwritten by the next convert into it and
/// holds `x` only until then (one node).
pub fn share_f16_x<'a>(
    gpu: &mut Gpu,
    x: &GpuTensor,
    x16: Option<&'a GpuTensor>,
    rows: usize,
    tier: PrefillActTier,
    ws: &[&WeightTensor],
) -> Result<Option<&'a GpuTensor>, String> {
    let (Some(first), Some(x16)) = (ws.first(), x16) else {
        return Ok(None);
    };
    let k = first.k;
    if ws
        .iter()
        .any(|w| w.k != k || !f16_x_route(gpu, w, rows, tier))
    {
        return Ok(None);
    }
    let n = rows
        .checked_mul(k)
        .ok_or_else(|| "raon projection: shared F16 input extent overflows".to_string())?;
    if x.dtype != DType::F32 || x16.dtype != DType::F16 || x.numel() < n || x16.numel() < n {
        return Ok(None);
    }
    gpu.convert_f32_to_f16_into(x, x16, n)
        .map_err(|e| format!("raon projection (rows={rows}): shared F16 convert: {e:?}"))?;
    Ok(Some(x16))
}

/// [`project`] for a consumer of [`share_f16_x`]: with `Some(x16)` the F16 plane feeds the
/// F16-WMMA GEMM directly (no convert of its own) and the output is rounded exactly as
/// [`project`] rounds it; with `None` this is [`project`] on `x`.
#[allow(clippy::too_many_arguments)]
pub fn project_shared_f16(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    x16: Option<&GpuTensor>,
    y: &GpuTensor,
    rows: usize,
    round: OutRound,
    tier: PrefillActTier,
) -> Result<(), String> {
    let Some(x16) = x16 else {
        return project(gpu, w, x, y, rows, round, tier);
    };
    check(w, x, y, rows, "rows")?;
    if !f16_x_route(gpu, w, rows, tier) {
        return Err(
            "raon projection: a shared F16 input was passed to a projection off the F16-WMMA route"
                .into(),
        );
    }
    let fused = q8_0_prefill_gemm_f16x(
        gpu,
        &w.buf,
        x16,
        y,
        w.m,
        w.k,
        rows,
        round == OutRound::Bf16 && a8_fuse_round_enabled(),
    )
    .map_err(|e| format!("raon projection (rows={rows}): {e}"))?;
    if round == OutRound::Bf16 && !fused {
        round_bf16_rows(gpu, y, rows, w.m)?;
    }
    Ok(())
}
