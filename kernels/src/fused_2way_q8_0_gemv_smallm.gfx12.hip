// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx12 (RDNA4) 2-way fused Q8_0 GEMV for small-M cases (LA layer
// w_beta + w_alpha after Q8 conversion). 256-thread block with stride-K
// access pattern; 1 launch instead of 2.

extern "C" __global__ void fused_2way_q8_0_gemv_smallm(
    const char*  __restrict__ A_a,    // [M_a × (K/32) × 34] Q8_0
    const char*  __restrict__ A_b,    // [M_b × (K/32) × 34] Q8_0
    const float* __restrict__ x,      // [K] F32 input
    float*       __restrict__ y_a,    // [M_a]
    float*       __restrict__ y_b,    // [M_b]
    int M_a, int M_b, int K
) {
    const int gid = blockIdx.x;
    const int total = M_a + M_b;
    if (gid >= total) return;

    const char* A;
    float* y;
    int row;
    if (gid < M_a) {
        A = A_a; y = y_a; row = gid;
    } else {
        A = A_b; y = y_b; row = gid - M_a;
    }

    const int tid = threadIdx.x;
    const int num_blocks = K / 32;
    const char* row_base = A + (long long)row * num_blocks * 34;

    float sum = 0.0f;
    for (int k = tid; k < K; k += 256) {
        const int b = k >> 5;
        const int e = k & 31;
        const char* block_base = row_base + b * 34;
        const _Float16 d = *reinterpret_cast<const _Float16*>(block_base);
        const float scale = (float)d;
        const int8_t v = *reinterpret_cast<const int8_t*>(block_base + 2 + e);
        const float w = scale * (float)v;
        sum += w * x[k];
    }

    __shared__ float sdata[256];
    sdata[tid] = sum;
    __syncthreads();
    #pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        y[row] = sdata[0];
    }
}
