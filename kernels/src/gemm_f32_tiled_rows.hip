// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Row-tiled F32 GEMM, bit-identical to gemm_f32_batched (gemm_f32.hip):
//   Y[n, m] = sum_k A[m, k] * X[n, k]
// Block = 256 threads = 8 groups of 32 lanes. A block owns NT=4 rows n; the
// X[NT, KT] tile is staged in LDS (KT=1024) and shared by all 8 groups.
// Group g owns outputs m = m0 + g + 8*i, i < MPW (m0 = blockIdx.x * 8*MPW).
// Per (m, n) the arithmetic is exactly gemm_f32_batched: lane l accumulates
// `sum += a * x` over k = l, l+32, ... in increasing k (KT is a multiple of 32,
// so k = kbase + kk keeps k == l mod 32 across tiles), then the shfl_down tree
// 16..1 (width 32); lane 0 stores. A rows are read from global (L2).
// Grid: (ceil(M/(8*MPW)), ceil(N/NT)). Accumulators MPW*NT <= 64 per lane.

#include <hip/hip_runtime.h>

#define TR_NT 4
#define TR_KT 1024

template <int MPW>
__device__ __forceinline__ void gemm_f32_tiled_rows_body(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ Y,
    int M, int K, int N
) {
    __shared__ float xs[TR_NT][TR_KT];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int grp = tid >> 5;
    const int ngrp = blockDim.x >> 5;  // 8
    const int m0 = blockIdx.x * ngrp * MPW;
    const int n0 = blockIdx.y * TR_NT;
    const int nc = (N - n0) < TR_NT ? (N - n0) : TR_NT;  // >= 1 by grid

    const float* b_rows = B + (size_t)n0 * K;

    float sum[MPW][TR_NT];
    #pragma unroll
    for (int i = 0; i < MPW; ++i) {
        #pragma unroll
        for (int j = 0; j < TR_NT; ++j) sum[i][j] = 0.0f;
    }

    for (int kbase = 0; kbase < K; kbase += TR_KT) {
        const int kt = (K - kbase) < TR_KT ? (K - kbase) : TR_KT;
        #pragma unroll
        for (int j = 0; j < TR_NT; ++j) {
            for (int kk = tid; kk < kt; kk += 256)
                xs[j][kk] = (j < nc) ? b_rows[(size_t)j * K + kbase + kk] : 0.0f;
        }
        __syncthreads();
        for (int kk = lane; kk < kt; kk += 32) {
            float x[TR_NT];
            #pragma unroll
            for (int j = 0; j < TR_NT; ++j) x[j] = xs[j][kk];
            #pragma unroll
            for (int i = 0; i < MPW; ++i) {
                const int m = m0 + grp + ngrp * i;
                if (m < M) {
                    const float a = A[(size_t)m * K + kbase + kk];
                    #pragma unroll
                    for (int j = 0; j < TR_NT; ++j)
                        sum[i][j] += a * x[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < MPW; ++i) {
        const int m = m0 + grp + ngrp * i;
        #pragma unroll
        for (int j = 0; j < TR_NT; ++j) {
            float v = sum[i][j];
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down(v, offset, 32);
            if (lane == 0 && m < M && j < nc)
                Y[(size_t)(n0 + j) * M + m] = v;
        }
    }
}

// MPW=16: any M (grid.x chunks M by 128). Router M=128 -> 16 x 4 = 64 acc.
extern "C" __launch_bounds__(256)
__global__ void gemm_f32_tiled_rows(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ Y, int M, int K, int N
) {
    gemm_f32_tiled_rows_body<16>(A, B, Y, M, K, N);
}

// MPW=4: M <= 32 (selector M=17 -> 3 outputs per group).
extern "C" __launch_bounds__(256)
__global__ void gemm_f32_tiled_rows_mp4(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ Y, int M, int K, int N
) {
    gemm_f32_tiled_rows_body<4>(A, B, Y, M, K, N);
}
