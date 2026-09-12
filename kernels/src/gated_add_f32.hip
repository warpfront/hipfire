// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// FLUX gated residual accumulation, broadcast over rows, in-place on `acc`.
// Seed for the MMDiT forward. The CPU double/single blocks gate
// a residual with a `d`-wide vector: `acc[r,i] += gate[i] * x[r,i]` — the
// double-block's `g1` (attn-proj) and `g2` (mlp) gates and the single-block's
// `g1`. `gate` is a `d` row vector shared by every row.
//
// Layouts: acc [n_rows, d], gate [d], x [n_rows, d], row-major.
extern "C" __global__ void gated_add_f32(
    float* __restrict__ acc,
    const float* __restrict__ gate,
    const float* __restrict__ x,
    int n_rows, int d) {

    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    long total = (long)n_rows * d;
    if (idx < 0 || (long)idx >= total) return;

    int i = idx % d;
    acc[idx] += gate[i] * x[idx];
}
