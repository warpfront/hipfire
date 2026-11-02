// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Qwen4 sibling of bf16_round_trip_f32 (bf16_round_trip.hip).

// Round-trip only strided rows in-place. This is the exact same bit-level
// RN-even operation as bf16_round_trip_f32, with row addressing kept outside
// the per-element conversion so QSA index-K retains its source boundary.
extern "C" __global__ void bf16_round_trip_f32_strided(
    float* __restrict__ x,
    int rows,
    int row_offset,
    int row_stride,
    int width
) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * width;
    if (linear >= total) return;
    int row = linear / width;
    int column = linear - row * width;
    int index = row * row_stride + row_offset + column;
    float v = x[index];
    if (!isfinite(v)) return;
    union { float f; unsigned int u; } box;
    box.f = v;
    unsigned int lsb = (box.u >> 16) & 1u;
    box.u = (box.u + 0x7FFFu + lsb) & 0xFFFF0000u;
    x[index] = box.f;
}
