// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Exact (erf) GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
//
// LFM3-VL's `multi_modal_projector` uses `ACT2FN["gelu"]` = nn.GELU() with
// the default `approximate="none"` (erf), NOT the tanh approximation covered
// by `gelu_tanh_f32`. The projector is a small GELU MLP (4096->4096->2048);
// using the tanh approx here measurably shifts the projected visual tokens
// (rel-L1 ~1e-3 vs the HF reference), so ship the exact erf form. In-place
// capable if x == out. One thread per element.

extern "C" __global__ void gelu_erf_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    // 1/sqrt(2) = 0.7071067811865476
    out[i] = 0.5f * v * (1.0f + erff(v * 0.7071067812f));
}
