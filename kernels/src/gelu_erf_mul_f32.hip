// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Exact-erf gated GELU: `out[i] = gelu_erf(gate[i]) * up[i]` with
// `gelu_erf(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`.
//
// This is HuggingFace `ACT2FN["gelu"]`, the MLP activation Spark-X2.5 uses.
// It is NOT the tanh approximation in gelu_tanh.hip (which matches HF
// `gelu_pytorch_tanh` for Gemma-family models): the two differ by up to ~1e-3
// in the mid-range, so substituting one for the other silently diverges the
// FFN. A separate kernel keeps the formulas from ever being confused.
//
// Portable: plain `erff`, no arch-specific intrinsics — the same source JITs
// for gfx1010/1030/11/12.
//
// `out` may alias `gate` or `up` (each element is read once before written).
extern "C" __global__ void gelu_erf_mul_f32(
    const float* gate,
    const float* up,
    float* out,
    int n) {

    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < 0 || i >= n) return;
    const float x = gate[i];
    const float g = 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
    out[i] = g * up[i];
}
