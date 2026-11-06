// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// BF16-weight x BF16-activation GEMM for the Raon mHC selector when the streams
// live as BF16 bits. Bit-identical per output to gemm_bf16_xf32_batched /
// gemm_bf16_xf32_tiled_rows (and so to gemm_f32_batched on the widened weight)
// run on the F32 plane holding the same BF16 values: the activation widening is
// the exact high-half shift, and the per-output arithmetic (lane-strided FMA
// `sum += a * x` over k = l, l+32, ... in increasing k, then the width-32
// shfl_down tree 16..1) is copied unchanged.
//   Y[n, m] = sum_k float(A[m, k]) * float(B[n, k])
//
// gemm_bf16_xbf16_batched:    1 wave per (m, n); decode / verify row counts.
// gemm_bf16_xbf16_tiled_rows: gemm_bf16_xf32_tiled_rows with the X tile staged
//                             from BF16 (block = 4 rows, 8 waves over outputs).
// Grids as the xf32 kernels: batched (M, N); tiled (ceil(M/(8*MPW)), ceil(N/4)).

#include <hip/hip_runtime.h>

#define TR_NT 4
#define TR_KT 1024

__device__ __forceinline__ float b2f(unsigned short v) {
    return __uint_as_float(((unsigned int)v) << 16);
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_bf16_xbf16_batched(
    const unsigned short* __restrict__ W,
    const unsigned short* __restrict__ X,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int row = blockIdx.x;
    const int batch = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= M || batch >= N || lane >= 32) return;

    const unsigned short* w_row = W + (size_t)row * K;
    const unsigned short* x_row = X + (size_t)batch * K;
    float sum = 0.0f;
    for (int k = lane; k < K; k += 32) sum += b2f(w_row[k]) * b2f(x_row[k]);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset, 32);
    if (lane == 0)
        Y[(size_t)batch * M + row] = sum;
}

template <int MPW>
__device__ __forceinline__ void gemm_bf16_xbf16_tiled_rows_body(
    const unsigned short* __restrict__ A,
    const unsigned short* __restrict__ B,
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

    const unsigned short* b_rows = B + (size_t)n0 * K;

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
                xs[j][kk] = (j < nc) ? b2f(b_rows[(size_t)j * K + kbase + kk]) : 0.0f;
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
                    const float a = b2f(A[(size_t)m * K + kbase + kk]);
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

// MPW=16: any M (grid.x chunks M by 128).
extern "C" __launch_bounds__(256)
__global__ void gemm_bf16_xbf16_tiled_rows(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ B,
    float* __restrict__ Y, int M, int K, int N
) {
    gemm_bf16_xbf16_tiled_rows_body<16>(A, B, Y, M, K, N);
}

// MPW=4: M <= 32 (selector M=17 -> 3 outputs per group).
extern "C" __launch_bounds__(256)
__global__ void gemm_bf16_xbf16_tiled_rows_mp4(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ B,
    float* __restrict__ Y, int M, int K, int N
) {
    gemm_bf16_xbf16_tiled_rows_body<4>(A, B, Y, M, K, N);
}
