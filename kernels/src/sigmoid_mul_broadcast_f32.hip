// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Headwise broadcast sigmoid gate, applied in place:
// `x[h * head_dim + j] *= sigmoid(gate[h])`.
//
// Spark-X2.5's attention output gate (`g_proj`, one scalar per head — 16
// scalars, not q_dim elements) broadcasts over each contiguous head_dim-wide
// head of the attention output before `out_proj`. This differs from
// sigmoid_mul.hip, whose gate is elementwise (one value per x lane); the
// per-lane math here is bit-identical to that kernel, only the gate indexing
// differs, so the two agree exactly when the gate is explicitly expanded.
//
// head_dim is a runtime argument (256 for Spark-X2.5) so the same source
// serves any head width. Portable: scalar indexing only, no arch-specific
// intrinsics — the same source JITs for gfx1010/1030/11/12.
//
// `x` is read-modify-written in place; `gate` is read-only.
extern "C" __global__ void sigmoid_mul_broadcast_f32(
    float* __restrict__ x,          // in/out: [n_heads * head_dim]
    const float* __restrict__ gate, // read-only: [n_heads]
    int n,                          // == n_heads * head_dim
    int head_dim
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < 0 || i >= n) return;
    float g = gate[i / head_dim];
    float s = 1.0f / (1.0f + expf(-g));
    x[i] = x[i] * s;
}
