// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Exact Hugging Face YaRN half-split RoPE for Qwen DSpark.
// Host uploads inverse-frequency table freqs[n_rot/2]; kernel only reads it
// (no per-thread pow/log). Pairs are HF rotate_half: (i, i + n_rot/2).
// mscale multiplies both cos and sin.
//
// Grid: [ceil(n_rot/2 / block), batch_size, 1]
// Q/K are [batch_size × n_heads × head_dim] row-major.
// Inactive dimensions outside n_rot remain untouched.

extern "C" __global__ void rope_yarn_halfsplit_batched_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ positions,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int n_rot,
    const float* __restrict__ freqs,  // [n_rot/2] host-computed inv-freqs
    float mscale,
    int batch_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    int half = n_rot / 2;
    if (i >= half || b >= batch_size) return;

    int pos = positions[b];
    float angle = (float)pos * freqs[i];
    float cos_a = cosf(angle) * mscale;
    float sin_a = sinf(angle) * mscale;

    int q_stride = n_heads_q * head_dim;
    int k_stride = n_heads_k * head_dim;

    for (int h = 0; h < n_heads_q; h++) {
        int base = b * q_stride + h * head_dim;
        float v0 = q[base + i];
        float v1 = q[base + i + half];
        q[base + i]        = v0 * cos_a - v1 * sin_a;
        q[base + i + half] = v0 * sin_a + v1 * cos_a;
    }

    for (int h = 0; h < n_heads_k; h++) {
        int base = b * k_stride + h * head_dim;
        float v0 = k[base + i];
        float v1 = k[base + i + half];
        k[base + i]        = v0 * cos_a - v1 * sin_a;
        k[base + i + half] = v0 * sin_a + v1 * cos_a;
    }
}
