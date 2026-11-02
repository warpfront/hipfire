// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Qwen4 BF16-recipe siblings of silu_mul_f32 (silu_mul.hip).

// bf16_round_trip_f32's rounding (RNE; non-finite values unchanged).
static __device__ __forceinline__ float silu_mul_bf16_rt(float v) {
    if (!isfinite(v)) return v;
    const unsigned int u = __float_as_uint(v);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

// silu_mul_f32 with gate and up round-tripped through BF16 first and the
// product round-tripped after: the round trip / silu_mul / round trip
// sequence in one pass, the same value per element.
extern "C" __global__ void silu_mul_bf16_rt_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = silu_mul_bf16_rt(gate[i]);
        out[i] = silu_mul_bf16_rt((v / (1.0f + expf(-v))) * silu_mul_bf16_rt(up[i]));
    }
}

// The BF16 shared-expert activation of `selectors` tokens in one launch:
// `out` is silu_mul_bf16_rt_f32(gate, up) and threads below `selectors` also
// replace their token's selector scalar with sigmoid(its BF16 round trip) —
// bitwise the separate round-trip, sigmoid_f32 and silu_mul_bf16_rt_f32
// launches (n >= selectors).
extern "C" __global__ void shared_expert_activation_bf16_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    float* __restrict__ selector,
    int n,
    int selectors
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < selectors) {
        selector[i] = 1.0f / (1.0f + expf(-silu_mul_bf16_rt(selector[i])));
    }
    if (i < n) {
        float v = silu_mul_bf16_rt(gate[i]);
        out[i] = silu_mul_bf16_rt((v / (1.0f + expf(-v))) * silu_mul_bf16_rt(up[i]));
    }
}
