// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Few-row F16-weight x F32-activation GEMM with the weights read ONCE:
//   Y[n, m] = sum_k W[m, k] * X[n, k]      W: [M, K] F16, X: [N, K] F32, N <= 16.
//
// The DFlash2 draft's `fc` projection of the target's feature rows is the
// shape this exists for: K = num_extract x streams x hidden (51,200 on Raon),
// M = hidden (2,560), N = the window's new rows (accepted + 1 <= 16). The
// batched WMMA GEMM it replaces runs one wave per 16 output rows with scalar
// F16 loads (160 waves for ~262 MB of weights, ~106 GB/s); this kernel spreads
// the K reduction over 256 threads and streams the weights with 16-byte loads.
//
// One block per MR = 4 output rows. Thread t owns k = 8 t + 2048 j: it loads
// the 4 rows' 8-element weight vectors once, then for every activation row
// accumulates 8 F32 FMAs per output (activations stay F32, no F16 rounding of
// X). Reduction: wave butterflies, then the waves in ascending order. The
// order is fixed, so the result is deterministic run to run (it is not the
// WMMA kernel's summation order).
//
// Contract: K % 8 == 0, W and X 16-byte aligned, 1 <= N <= 16.

#define MR 4
#define NMAX 16
#define NT 256
#define MAX_WAVES (NT / 32)

extern "C" __launch_bounds__(NT)
__global__ void gemm_f16_skinny_xf32(
    const _Float16* __restrict__ W,  // [M, K]
    const float* __restrict__ X,     // [N, K]
    float* __restrict__ Y,           // [N, M]
    int M,
    int K,
    int N) {
    __shared__ float red[MAX_WAVES][MR * NMAX];

    const int m0 = blockIdx.x * MR;
    const int tid = threadIdx.x;

    float acc[MR][NMAX];
#pragma unroll
    for (int r = 0; r < MR; r++) {
#pragma unroll
        for (int b = 0; b < NMAX; b++) acc[r][b] = 0.0f;
    }

    // Rows past M re-read the last row (their sums are discarded).
    const _Float16* wrow[MR];
#pragma unroll
    for (int r = 0; r < MR; r++) {
        const int row = (m0 + r < M) ? (m0 + r) : (M - 1);
        wrow[r] = W + (size_t)row * K;
    }

    for (int k = tid * 8; k < K; k += NT * 8) {
        uint4 wv[MR];
#pragma unroll
        for (int r = 0; r < MR; r++) {
            wv[r] = *reinterpret_cast<const uint4*>(wrow[r] + k);
        }
#pragma unroll
        for (int b = 0; b < NMAX; b++) {
            if (b < N) {
                const float4 x0 = *reinterpret_cast<const float4*>(X + (size_t)b * K + k);
                const float4 x1 = *reinterpret_cast<const float4*>(X + (size_t)b * K + k + 4);
#pragma unroll
                for (int r = 0; r < MR; r++) {
                    const _Float16* hp = reinterpret_cast<const _Float16*>(&wv[r]);
                    float a = acc[r][b];
                    a = fmaf((float)hp[0], x0.x, a);
                    a = fmaf((float)hp[1], x0.y, a);
                    a = fmaf((float)hp[2], x0.z, a);
                    a = fmaf((float)hp[3], x0.w, a);
                    a = fmaf((float)hp[4], x1.x, a);
                    a = fmaf((float)hp[5], x1.y, a);
                    a = fmaf((float)hp[6], x1.z, a);
                    a = fmaf((float)hp[7], x1.w, a);
                    acc[r][b] = a;
                }
            }
        }
    }

    // Wave butterflies (fixed order), then the waves in ascending order.
    const int lane = tid % warpSize;
    const int wave = tid / warpSize;
#pragma unroll
    for (int r = 0; r < MR; r++) {
#pragma unroll
        for (int b = 0; b < NMAX; b++) {
            float v = acc[r][b];
            for (int o = warpSize >> 1; o > 0; o >>= 1) v += __shfl_xor(v, o);
            acc[r][b] = v;
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < MR; r++) {
#pragma unroll
            for (int b = 0; b < NMAX; b++) red[wave][r * NMAX + b] = acc[r][b];
        }
    }
    __syncthreads();
    const int waves = NT / warpSize;
    for (int idx = tid; idx < MR * NMAX; idx += NT) {
        const int r = idx / NMAX;
        const int b = idx % NMAX;
        if (b < N && m0 + r < M) {
            float s = 0.0f;
            for (int w = 0; w < waves; w++) s += red[w][idx];
            Y[(size_t)b * M + m0 + r] = s;
        }
    }
}
