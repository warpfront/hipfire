// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// c[r][i] = a[r][i] + b[r / group][i] over `total` = rows * width elements:
// each `b` row is shared by `group` consecutive rows of `a` (bitwise the
// per-row add_f32 of the same operands).
extern "C" __global__ void add_broadcast_rows_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int width,
    int group,
    int total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int row = i / width;
    c[i] = a[i] + b[(row / group) * width + (i - row * width)];
}
