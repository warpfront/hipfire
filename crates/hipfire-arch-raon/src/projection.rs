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

use hipfire_runtime::llama::{weight_gemm, weight_gemv, WeightTensor};
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
pub fn project_rows(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    round: OutRound,
) -> Result<(), String> {
    check(w, x, y, rows, "rows")?;
    if rows == 1 {
        return project_row(gpu, w, x, y, round);
    }
    match w.gpu_dtype {
        DType::MQ4G256V2 => weight_gemm(gpu, w, x, y, rows),
        DType::Q8_0 => gpu.gemm_q8_0_batched_chunked(&w.buf, x, y, w.m, w.k, rows),
        DType::F32 => gpu.gemm_f32_batched(&w.buf, x, y, w.m, w.k, rows),
        other => unreachable!("dtype {other:?} rejected by check()"),
    }
    .map_err(|e| format!("raon projection (rows={rows}): {e:?}"))?;
    if round == OutRound::Bf16 {
        round_bf16_rows(gpu, y, rows, w.m)?;
    }
    Ok(())
}

/// Dispatch to [`project_row`] / [`project_rows`] by row count.
pub fn project(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    round: OutRound,
) -> Result<(), String> {
    project_rows(gpu, w, x, y, rows, round)
}
