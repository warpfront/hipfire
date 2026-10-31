// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void rope_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ pos_buf,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    float freq_base,
    float l3_inv_factor,
    float l3_smooth_mul,
    float l3_smooth_add
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    if (i >= half) return;

    int pos = pos_buf[0];
    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
    // Llama-3 remap (rdna_compute::rope::RopeLlama3); inv_factor 1 = plain RoPE.
    if (l3_inv_factor != 1.0f) {
        float s = fminf(fmaxf(freq * l3_smooth_mul + l3_smooth_add, 0.0f), 1.0f);
        freq *= s + (1.0f - s) * l3_inv_factor;
    }
    float val = (float)pos * freq;
    float cos_val = cosf(val);
    float sin_val = sinf(val);

    for (int h = 0; h < n_heads_q; h++) {
        int base = h * head_dim;
        float q0 = q[base + i];
        float q1 = q[base + i + half];
        q[base + i]        = q0 * cos_val - q1 * sin_val;
        q[base + i + half] = q0 * sin_val + q1 * cos_val;
    }

    for (int h = 0; h < n_heads_k; h++) {
        int base = h * head_dim;
        float k0 = k[base + i];
        float k1 = k[base + i + half];
        k[base + i]        = k0 * cos_val - k1 * sin_val;
        k[base + i + half] = k0 * sin_val + k1 * cos_val;
    }
}
