// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

// Output gate evaluated in the BF16 activation dtype:
//     out = bf16( y * bf16( sigmoid(g) ) )
// Both the sigmoid result and the product are rounded to BF16 (the sigmoid is
// computed in FP32 from the BF16 gate value, as a BF16 tensor op does). y and
// g are BF16-valued F32 planes; out may alias y.

static __device__ __forceinline__ uint16_t la_f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static __device__ __forceinline__ float la_round_bf16(float f) {
    return __uint_as_float(((uint32_t)la_f32_to_bf16(f)) << 16);
}

extern "C" __global__ void latent_attention_gate_bf16(
    const float* y,                // [n]
    const float* __restrict__ g,   // [n]
    float* out,                    // [n]
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = la_round_bf16(1.0f / (1.0f + expf(-g[i])));
    out[i] = la_round_bf16(y[i] * s);
}
