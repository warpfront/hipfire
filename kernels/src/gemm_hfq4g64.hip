// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

// Batched HFQ4-G64 GEMM (ParoQ4G64 prefill): y[b, row] = A[row] . x[b] for an
// 8-row batch tile per grid.y. Same 40-byte group layout and half-warp group
// split as gemv_hfq4g64; the dequantized weights are reused across the tile.

#include <hip/hip_runtime.h>

#define BATCH_TILE 8

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_hfq4g64(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int batch_start = blockIdx.y * BATCH_TILE;
    const int tid = threadIdx.x;
    const int half = tid >> 4;
    const int lane = tid & 15;

    const int groups_per_row = (K + 63) / 64;
    const char* row_ptr = A + (long long)row * groups_per_row * 40;

    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    float acc[BATCH_TILE];
    for (int b = 0; b < BATCH_TILE; b++) acc[b] = 0.0f;

    for (int g = half; g < groups_per_row; g += 2) {
        const char* gptr = row_ptr + g * 40;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        int base_k = g * 64 + lane * 4;
        unsigned char b0 = nibbles[lane * 2];
        unsigned char b1 = nibbles[lane * 2 + 1];

        float w0 = scale * (float)(b0 & 0xF) + zero;
        float w1 = scale * (float)(b0 >> 4)  + zero;
        float w2 = scale * (float)(b1 & 0xF) + zero;
        float w3 = scale * (float)(b1 >> 4)  + zero;

        for (int b = 0; b < local_bs; b++) {
            const float* xb = x + (long long)(batch_start + b) * K;
            if (base_k + 3 < K) {
                acc[b] += w0 * xb[base_k]     + w1 * xb[base_k + 1]
                        + w2 * xb[base_k + 2] + w3 * xb[base_k + 3];
            } else {
                if (base_k     < K) acc[b] += w0 * xb[base_k];
                if (base_k + 1 < K) acc[b] += w1 * xb[base_k + 1];
                if (base_k + 2 < K) acc[b] += w2 * xb[base_k + 2];
            }
        }
    }

    for (int b = 0; b < local_bs; b++) {
        float val = acc[b];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
        if (tid == 0) y[(long long)(batch_start + b) * M + row] = val;
    }
}
