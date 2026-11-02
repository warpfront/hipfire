// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// FP8B128 shared-expert BF16 SwiGLU plus shared-expert selector sigmoid
// (gfx942).
//
// Numerical contract: the det teacher, fastfwd.py:476-479, rounds every op of
// gate/up/silu/mul to BF16:
//
//   g = bf16_rne(gate_acc)
//   u = bf16_rne(up_acc)
//   s = bf16_rne(g / (1 + expf(-g)))
//   h = bf16_rne(s * u)               // stored as F32 holding the BF16 value
//
// Selector (one scalar per row, in place, applied exactly once):
//
//   selector[r] = bf16_rne(1 / (1 + expf(-bf16_rne(selector[r]))))
//
// i.e. a BF16 sigmoid of the BF16 logit. expf is the precise libm expf: no
// __expf, no fast-math.
//
//   gate, up, activation: F32 [rows, mi] (distinct buffers)
//   selector:             F32 [rows], read-modify-write by one thread per row
//
// Every output element (activation[i], selector[r]) has exactly one writer:
// plain stores, no atomics. Thread i < rows owns selector[i]; thread i < n
// owns activation[i], n = rows*mi (mi >= 1, so rows <= n).
//
// Geometry: grid [ceil(rows*mi / 256)], block [256]; one thread per element.

#include <hip/hip_runtime.h>

// bf16 round-to-nearest-even of an F32 (non-finite values pass through).
static __device__ __forceinline__ float fp8act_bf16_rne(float v) {
    if (!isfinite(v)) return v;
    const unsigned int u = __float_as_uint(v);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

extern "C" __global__ __launch_bounds__(256) void fp8b128_moe_shared_activation_gfx942(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ activation,
    float* __restrict__ selector,
    int rows,
    int mi
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        const float l = fp8act_bf16_rne(selector[i]);
        selector[i] = fp8act_bf16_rne(1.0f / (1.0f + expf(-l)));
    }
    if (i < (long long)rows * mi) {
        const float g = fp8act_bf16_rne(gate[i]);
        const float u = fp8act_bf16_rne(up[i]);
        const float s = fp8act_bf16_rne(g / (1.0f + expf(-g)));
        activation[i] = fp8act_bf16_rne(s * u);
    }
}
