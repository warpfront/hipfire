// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Register-tiled exact-order BF16-weight small-M GEMMs for the Raon mHC selector
// (M = 17, K = 10240, BF16 activation rows) and MoE router (M = 128, K = 2560, F32
// activation rows):
//   Y[n, m] = sum_k float(A[m, k]) * float(X[n, k])
// Per output the arithmetic is exactly gemm_f32_batched run on the widened weight
// (and so gemm_bf16_xf32_batched / gemm_bf16_xf32_tiled_rows / gemm_bf16_xbf16_*): lane l
// accumulates `sum += a * x` over k = l, l+32, ... in increasing k, then the width-32
// shfl_down tree 16..1; lane 0 stores. Only the register layout differs:
//   * a CTA is NW waves and owns NT activation rows; wave g owns the OS outputs
//     m = m0 + g + NW*i (i < OS), m0 = blockIdx.x * NW * OS;
//   * every wave keeps its OS*NT accumulators in registers, reads its NT activation values and OS
//     weights of each k straight from global memory (no LDS, no barrier) and issues UR consecutive
//     k steps' loads before the FMAs, so one weight fetch feeds NT rows and one activation fetch
//     feeds OS outputs;
//   * K must be a multiple of 32*UR (every profile: a multiple of 512): each lane's UR k steps per
//     loop trip are all in range, so the loop needs neither a tail nor a lane-dependent bound (a
//     remainder split measured 1.4x slower on the selector profile with the same arithmetic). The
//     launcher and `hipfire-dispatch` admit only such K and keep the batched kernels otherwise.
// Grid: (ceil(M / (NW*OS)), ceil(N / NT)); block NW*32 threads. Rows past N and outputs past M are
// clamped on load (their lanes duplicate a valid output) and never stored. There is no early exit
// for waves past M: it measured up to 1.6x slower on the same arithmetic, and the Raon shapes (M = 17
// and 128) leave at most 3 idle output slots per CTA.
//
// Profiles (rows per CTA NT; the dispatch picks by N, see stream_mix.rs `rows_gemm`):
//   xf32  _r1/_r4/_r8: NT 1/4/8, OS 1/4/8, NW 8, UR 16/8/8 — 8/32/64 outputs per CTA (router, M = 128)
//   xbf16 _s1/_s4/_s8: NT 1/4/8, OS 1/3/5, NW 17/6/4, UR 16/8/8 — 17/18/20 outputs per CTA (selector, M = 17)

#include <hip/hip_runtime.h>

__device__ __forceinline__ float sr_b2f(unsigned short v) {
    return __uint_as_float(((unsigned int)v) << 16);
}
__device__ __forceinline__ float sr_ld(const float* p, size_t i) { return p[i]; }
__device__ __forceinline__ float sr_ld(const unsigned short* p, size_t i) { return sr_b2f(p[i]); }

template <typename XT, int NT, int OS, int NW, int UR>
__device__ __forceinline__ void gemm_bf16_small_rows_body(
    const unsigned short* __restrict__ A,
    const XT* __restrict__ B,
    float* __restrict__ Y,
    int M, int K, int N
) {
    const int lane = threadIdx.x & 31;
    const int grp = threadIdx.x >> 5;
    const int m0 = blockIdx.x * (NW * OS);
    const int n0 = blockIdx.y * NT;

    const unsigned short* wp[OS];
    #pragma unroll
    for (int i = 0; i < OS; ++i) {
        int m = m0 + grp + NW * i;
        m = m < M ? m : M - 1;
        wp[i] = A + (size_t)m * K;
    }
    size_t xo[NT];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        int n = n0 + j;
        n = n < N ? n : N - 1;
        xo[j] = (size_t)n * K;
    }

    float sum[OS][NT];
    #pragma unroll
    for (int i = 0; i < OS; ++i) {
        #pragma unroll
        for (int j = 0; j < NT; ++j) sum[i][j] = 0.0f;
    }

    for (int k0 = lane; k0 < K; k0 += 32 * UR) {
        float x[UR][NT];
        unsigned short w[UR][OS];
        #pragma unroll
        for (int u = 0; u < UR; ++u) {
            #pragma unroll
            for (int j = 0; j < NT; ++j) x[u][j] = sr_ld(B, xo[j] + k0 + 32 * u);
            #pragma unroll
            for (int i = 0; i < OS; ++i) w[u][i] = wp[i][k0 + 32 * u];
        }
        #pragma unroll
        for (int u = 0; u < UR; ++u) {
            #pragma unroll
            for (int i = 0; i < OS; ++i) {
                const float a = sr_b2f(w[u][i]);
                #pragma unroll
                for (int j = 0; j < NT; ++j) sum[i][j] += a * x[u][j];
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < OS; ++i) {
        const int m = m0 + grp + NW * i;
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            float v = sum[i][j];
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down(v, offset, 32);
            if (lane == 0 && m < M && n0 + j < N)
                Y[(size_t)(n0 + j) * M + m] = v;
        }
    }
}

#define SMALL_ROWS_KERNEL(NAME, XT, NT, OS, NW, UR)                                              \
    extern "C" __launch_bounds__((NW) * 32) __global__ void NAME(                               \
        const unsigned short* __restrict__ A, const XT* __restrict__ B, float* __restrict__ Y,   \
        int M, int K, int N) {                                                                   \
        gemm_bf16_small_rows_body<XT, NT, OS, NW, UR>(A, B, Y, M, K, N);                        \
    }

// F32 activation rows (router, M > 32).
SMALL_ROWS_KERNEL(gemm_bf16_xf32_small_rows_r1, float, 1, 1, 8, 16)
SMALL_ROWS_KERNEL(gemm_bf16_xf32_small_rows_r4, float, 4, 4, 8, 8)
SMALL_ROWS_KERNEL(gemm_bf16_xf32_small_rows_r8, float, 8, 8, 8, 8)
// BF16 activation rows (mHC selector, M <= 32).
SMALL_ROWS_KERNEL(gemm_bf16_xbf16_small_rows_s1, unsigned short, 1, 1, 17, 16)
SMALL_ROWS_KERNEL(gemm_bf16_xbf16_small_rows_s4, unsigned short, 4, 3, 6, 8)
SMALL_ROWS_KERNEL(gemm_bf16_xbf16_small_rows_s8, unsigned short, 8, 5, 4, 8)
