// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1201 decode (`HIPFIRE_G12_DEC_NORM`): the same rotation with one
// workgroup per head, grid (ceil(half / blockDim.x), n_heads_q + n_heads_k),
// instead of one wave looping over every head. Each workgroup recomputes the
// identical per-dim frequency, angle, cos and sin, and touches only its own
// head, so the in-place q/k updates stay race-free and bit-identical.
extern "C" __global__ void rope_partial_halfsplit_f32_headgrid(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ pos_buf,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int n_rot,
    float freq_base
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = n_rot / 2;
    if (i >= half) return;

    int pos = *pos_buf;
    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)n_rot);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    const int h = blockIdx.y;
    float* v = (h < n_heads_q) ? q + h * head_dim : k + (h - n_heads_q) * head_dim;
    if (h >= n_heads_q + n_heads_k) return;
    float v0 = v[i];
    float v1 = v[i + half];
    v[i]        = v0 * cos_a - v1 * sin_a;
    v[i + half] = v0 * sin_a + v1 * cos_a;
}
