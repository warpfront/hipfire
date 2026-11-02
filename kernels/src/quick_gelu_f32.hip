// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// OpenAI CLIP's default MLP activation: `x * sigmoid(1.702 x)`.
// Mirrors the `quick_gelu` branch of `hipfire_arch_diffusion::clip::encode`.
//
// The FLUX `clip_l` config ships `hidden_act: "quick_gelu"`; substituting the
// exact-erf GELU does not fail, it silently diverges the pooled vector. The
// erf variant therefore stays on the CPU reference only — this kernel is
// quick-GELU and nothing else, so a config that asks for erf falls back to
// the host path rather than being quietly approximated here.
//
// In-place capable (`out` may alias `x`).
extern "C" __global__ void quick_gelu_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int n) {

    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < 0 || i >= n) return;
    const float v = x[i];
    out[i] = v * (1.0f / (1.0f + expf(-1.702f * v)));
}
