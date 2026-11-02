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

// c[i] = a[i] + b[(i / (branches * hidden)) * hidden + i % hidden] over
// n * branches * hidden elements: each of the `n` rows of `b` (width `hidden`)
// is added to all `branches` branch rows of the matching row of `a` (what
// `branches` add_f32 launches over one row do). One IEEE add per element.
extern "C" __global__ void broadcast_add_rows_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int branches,
    int hidden,
    int n
) {
    const long long total = (long long)n * branches * hidden;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long long wide = (long long)branches * hidden;
    c[i] = a[i] + b[(i / wide) * hidden + i % hidden];
}
