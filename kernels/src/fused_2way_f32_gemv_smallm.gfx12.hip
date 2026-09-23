// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx12 (RDNA4) 2-way fused F32 GEMV optimized for small-M cases
// (LA layer w_beta + w_alpha, M=n_v_heads=16 each).
//
// Why a separate small-M variant from the 4-way fused gate-side kernel:
//   The 4-way kernel uses wave32 (32 threads/WG) which is fine when
//   total M is large (router=256, gate=512, up=512 → 1280 WGs in flight).
//   For tiny M=16 + M=16 = 32 WGs total, wave32 under-saturates.
//   Empirically the 4-way kernel was -3% on alpha+beta fusion.
//
// This kernel uses 256-thread blocks (matching baseline gemv_f32) for
// proper BW saturation per WG. Combines 2 F32 GEMV launches into 1
// with shared X. WGs 0..M_a-1 produce y_a; WGs M_a..M_a+M_b-1 produce y_b.
//
// Per LA layer: 2 launches → 1 = 1 launch saved.
// Per token (32 LA layers): 32 launches saved × ~12 µs = ~0.4 ms wall savings.

extern "C" __global__ void fused_2way_f32_gemv_smallm(
    const float* __restrict__ A_a,
    const float* __restrict__ A_b,
    const float* __restrict__ x,
    float*       __restrict__ y_a,
    float*       __restrict__ y_b,
    int M_a, int M_b, int K,
    float alpha, float beta
) {
    const int gid = blockIdx.x;
    const int total = M_a + M_b;
    if (gid >= total) return;

    // Route this WG to its target matrix + output row.
    const float* A;
    float* y;
    int row;
    if (gid < M_a) {
        A = A_a; y = y_a; row = gid;
    } else {
        A = A_b; y = y_b; row = gid - M_a;
    }

    const int tid = threadIdx.x;
    const float* a_row = A + (long long)row * K;

    // 256-thread strided dot-product.
    float sum = 0.0f;
    for (int k = tid; k < K; k += 256) {
        sum += a_row[k] * x[k];
    }

    // 256-thread LDS reduction.
    __shared__ float sdata[256];
    sdata[tid] = sum;
    __syncthreads();
    #pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        y[row] = alpha * sdata[0] + beta * y[row];
    }
}
