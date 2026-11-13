// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MQ8G256 (qt14, device layout B) projection routes: the one place an arch crate reaches the
//! MQ8G256 launchers (Raon attention class).
//!
//! All routes read the same layout-B weight (`rdna_compute::mq8g256_repack_b`) and the same
//! FWHT256-rotated int8 activation planes ([`Mq8Activations`], written once per node by
//! [`mq8_quantize`]):
//!
//! * [`mq8_gemv_fused`]: decode (`n == 1`) and 2-row row-exact verify; the GEMV rotates and
//!   quantizes `x` inside its own launch, bit-identical to producer + [`mq8_gemv_rows`].
//! * [`mq8_gemv_rows`]: `n == 1` the decode GEMV, `n >= 2` the row-exact GEMV whose row `r` is
//!   bit-identical to the decode GEMV on row `r`.
//! * [`mq8_prefill_gemm`]: the batched int8 GEMM, optionally rounding every output to BF16 in its
//!   epilogue (bit-identical to a separate round pass).
//!
//! No route reads an environment variable; the weight's format selects it. Every launch is a plain
//! kernel launch (no allocation, no host sync), so each is capture-safe after one eager call.

use crate::types::DispatchError;
use rdna_compute::{Gpu, GpuTensor, Mq8Activations, Mq8WLayout};

/// Rotate + quantize `x` `[rows, k]` once into the Gpu's MQ8 activation scratch. The result is
/// valid until the next quantizer launch ([`mq8_acts_current`]).
pub fn mq8_quantize(
    gpu: &mut Gpu,
    x: &GpuTensor,
    rows: usize,
    k: usize,
) -> Result<Mq8Activations, DispatchError> {
    hip!(gpu.mq8_quantize_x(x, rows, k))
}

/// Whether `acts` still describes the scratch contents (no quantizer ran since).
pub fn mq8_acts_current(gpu: &Gpu, acts: &Mq8Activations) -> bool {
    gpu.mq8_acts_current(acts)
}

/// `y[n, m] = W x^T` for `n` in {1, 2}: the fused-rotate GEMV (one launch, no producer).
pub fn mq8_gemv_fused(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    n: usize,
    y: &GpuTensor,
    m: usize,
    k: usize,
) -> Result<(), DispatchError> {
    hip!(gpu.gemv_mq8g256_b_x(w, x, n, y, m, k))
}

/// `y[n, m] = W X^T` on prepared activations, every row bit-identical to the decode GEMV.
pub fn mq8_gemv_rows(
    gpu: &mut Gpu,
    w: &GpuTensor,
    acts: &Mq8Activations,
    y: &GpuTensor,
    m: usize,
) -> Result<(), DispatchError> {
    hip!(gpu.gemv_mq8g256_b_rows(w, acts, y, m))
}

/// `y[n, m] = X W^T` on prepared activations, F32 out; `bf16_out` rounds every output to BF16
/// (F32 plane holding BF16 values) in the GEMM epilogue.
pub fn mq8_prefill_gemm(
    gpu: &mut Gpu,
    w: &GpuTensor,
    acts: &Mq8Activations,
    y: &GpuTensor,
    m: usize,
    bf16_out: bool,
) -> Result<(), DispatchError> {
    if bf16_out {
        hip!(gpu.gemm_mq8g256_gemm_bf16(w, Mq8WLayout::B, acts, y, m))
    } else {
        hip!(gpu.gemm_mq8g256_gemm(w, Mq8WLayout::B, acts, y, m))
    }
}
