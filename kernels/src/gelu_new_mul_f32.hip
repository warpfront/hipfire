// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// T5 v1.1 gated-GELU FFN activation: `out = gelu_new(a) * b`, i.e. the
// `wo(gelu(wi_0 x) * wi_1 x)` middle term. Mirrors
// `hipfire_arch_diffusion::nn::gelu_tanh` term for term.
//
// The inner constant is sqrt(2/pi) = 0.7978845608 — NOT sqrt(2)*sqrt(2/pi).
// A stray sqrt(2) over-activates every FFN lane (~9% at x=1, ~46% at x=-1)
// and silently diverges the T5 conditioning from every reference; the CPU
// side has a pinned unit test for exactly that, and this kernel must agree.
//
// `out` may alias `a` or `b` (each element is read once before it is written).
extern "C" __global__ void gelu_new_mul_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n) {

    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < 0 || i >= n) return;
    const float x = a[i];
    const float g = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    out[i] = g * b[i];
}
