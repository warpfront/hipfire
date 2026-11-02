// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

// HFQ4-G64 GEMV (ParoQ4G64): 40-byte groups = f32 scale, f32 zero, 32 B of
// nibbles (low nibble = even element). One warp per row; the two half-warps
// take one group each, so every lane reads 2 bytes = 4 weights per step like
// gemv_hfq4g128. Plain HIP, no arch intrinsics (gfx11/gfx12 wave32).

#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq4g64(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int half = tid >> 4;
    const int lane = tid & 15;

    const int groups_per_row = (K + 63) / 64;
    const char* row_ptr = A + (long long)row * groups_per_row * 40;

    float acc = 0.0f;

    for (int g = half; g < groups_per_row; g += 2) {
        const char* gptr = row_ptr + g * 40;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        int base_idx = g * 64 + lane * 4;
        unsigned char b0 = nibbles[lane * 2];
        unsigned char b1 = nibbles[lane * 2 + 1];

        float v0 = scale * (float)(b0 & 0xF) + zero;
        float v1 = scale * (float)(b0 >> 4)  + zero;
        float v2 = scale * (float)(b1 & 0xF) + zero;
        float v3 = scale * (float)(b1 >> 4)  + zero;

        if (base_idx + 3 < K) {
            acc += v0 * x[base_idx]     + v1 * x[base_idx + 1]
                 + v2 * x[base_idx + 2] + v3 * x[base_idx + 3];
        } else {
            if (base_idx     < K) acc += v0 * x[base_idx];
            if (base_idx + 1 < K) acc += v1 * x[base_idx + 1];
            if (base_idx + 2 < K) acc += v2 * x[base_idx + 2];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
