// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12/RDNA4 sister of gemm_f16_wmma_mb8.hip.
//
// gfx12 WMMA differs from gfx11 in two load-bearing ways:
// - half8_t operands and the _w32_gfx12 builtin
// - output rows map contiguously per lane group:
//     acc[j] = C[8 * (tid >> 4) + j][tid & 15]

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

#define NB 8

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_f16_wmma_mb8_gfx12(
    const _Float16* __restrict__ W,   // [M, K] row-major F16 (weights)
    const float*    __restrict__ X,   // [N, K] row-major F32 (input)
    float*          __restrict__ Y,   // [N, M] row-major F32 (transposed output)
    int M, int K, int N
) {
    const int row_start = blockIdx.x * 16;
    const int col_start = blockIdx.y * (16 * NB);
    const int tid = threadIdx.x;
    const int lane_m = tid & 15;
    const int k_grp = tid >> 4;

    if (row_start >= M || col_start >= N) return;

    const int my_a_row = row_start + lane_m;
    const bool a_in_bounds = my_a_row < M;

    float8_t acc[NB];
    #pragma unroll
    for (int nb = 0; nb < NB; nb++) {
        acc[nb] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    for (int k0 = 0; k0 < K; k0 += 16) {
        half8_t a_reg;
        if (a_in_bounds) {
            const _Float16* src = W + (long long)my_a_row * K + k0;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int kk = k0 + k_grp * 8 + j;
                a_reg[j] = kk < K ? src[k_grp * 8 + j] : (_Float16)0.0f;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) a_reg[j] = (_Float16)0.0f;
        }

        #pragma unroll
        for (int nb = 0; nb < NB; nb++) {
            const int my_b_row = col_start + nb * 16 + lane_m;
            half8_t b_reg;
            if (my_b_row < N) {
                const float* src = X + (long long)my_b_row * K + k0;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int kk = k0 + k_grp * 8 + j;
                    b_reg[j] = kk < K ? (_Float16)src[k_grp * 8 + j] : (_Float16)0.0f;
                }
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) b_reg[j] = (_Float16)0.0f;
            }
            acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_reg, acc[nb]);
        }
    }

    #pragma unroll
    for (int nb = 0; nb < NB; nb++) {
        const int out_col = col_start + nb * 16 + lane_m;
        if (out_col < N) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int out_row = row_start + 8 * k_grp + j;
                if (out_row < M) {
                    Y[(long long)out_col * M + out_row] = acc[nb][j];
                }
            }
        }
    }
}

// One-column-tile sister of gemm_f16_wmma_mb8_gfx12: block (x, y) computes the
// 16 x 16 output tile at rows 16x, columns 16y. Launched with one column tile
// for N <= 16 (the F16 GEMV / small-batch lm_head shape behind
// `Gpu::gemm_f16_batched_lmhead`) and with ceil(N / 16) of them for prefill
// row blocks (`Gpu::gemm_f16_wmma_cols_gfx12`, the blocked Inkling-2 router).
//
// Bit-identical to gemm_f16_wmma_mb8_gfx12 at every N: mb8 keeps column tile
// t = 8 * blockIdx.y + nb in accumulator acc[nb] and stores it; this kernel
// issues the SAME wmma_f32_16x16x16_f16 chain into its one accumulator for
// tile blockIdx.y: same per-lane operand mapping (lane_m = A row / B column,
// k_grp = K half), A taken bit for bit from W, B rounded F32 -> F16 by the
// same conversion, zero operands for rows >= M / columns >= N, ascending k0,
// zero-initialised, same stores.
//
// It differs only in scheduling: one wave per 16 x 16 tile (mb8 runs one wave
// per eight tiles: 32 waves for the M=128, K=2304, N=512 router), and each
// lane fetches its operands as 16-byte vectors N16_UNROLL steps ahead of the
// chain (the mb8 body issues 8 scalar F16 loads per step and waits on them
// before each WMMA: ~93 us for M=128, K=2304, N=1 on gfx1201).
//
// Requires K % 16 == 0 and 16-byte aligned W / X (launcher-checked).
// Grid: (ceil(M / 16), ceil(N / 16), 1). Block: 32 threads.

#ifndef N16_UNROLL
#define N16_UNROLL 16
#endif

typedef unsigned int u32x4_t __attribute__((ext_vector_type(4)));
typedef float        f32x4_t __attribute__((ext_vector_type(4)));

__launch_bounds__(32)
extern "C" __global__ void gemm_f16_wmma_n16_gfx12(
    const _Float16* __restrict__ W,   // [M, K] row-major F16 (weights)
    const float*    __restrict__ X,   // [N, K] row-major F32 (input)
    float*          __restrict__ Y,   // [N, M] row-major F32 (transposed output)
    int M, int K, int N
) {
    const int row_start = blockIdx.x * 16;
    const int tid = threadIdx.x;
    const int lane_m = tid & 15;
    const int k_grp = tid >> 4;

    if (row_start >= M) return;

    // Column tile blockIdx.y: its 16 X rows / Y rows, N counted from there.
    const int col_start = blockIdx.y * 16;
    X += (long long)col_start * K;
    Y += (long long)col_start * M;
    N -= col_start;

    const bool a_in = row_start + lane_m < M;
    const bool b_in = lane_m < N;
    // One 16-byte A vector (8 F16) and two 16-byte X vectors (8 F32) per lane
    // per 16-wide K step; step s sits at +2*s / +4*s vectors.
    const u32x4_t* __restrict__ wa = reinterpret_cast<const u32x4_t*>(
        W + (long long)(a_in ? row_start + lane_m : row_start) * K + k_grp * 8);
    const f32x4_t* __restrict__ xb = reinterpret_cast<const f32x4_t*>(
        X + (long long)(b_in ? lane_m : 0) * K + k_grp * 8);

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

#define N16_STEP(av, xlo, xhi) do { \
        const half8_t a_reg = __builtin_bit_cast(half8_t, (av)); \
        half8_t b_reg; \
        b_reg[0] = (_Float16)(xlo).x; b_reg[1] = (_Float16)(xlo).y; \
        b_reg[2] = (_Float16)(xlo).z; b_reg[3] = (_Float16)(xlo).w; \
        b_reg[4] = (_Float16)(xhi).x; b_reg[5] = (_Float16)(xhi).y; \
        b_reg[6] = (_Float16)(xhi).z; b_reg[7] = (_Float16)(xhi).w; \
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_reg, acc); \
    } while (0)

    const u32x4_t z4 = {0u, 0u, 0u, 0u};
    const f32x4_t zf = {0.0f, 0.0f, 0.0f, 0.0f};
    const int steps = K >> 4;
    int s = 0;
    for (; s + N16_UNROLL <= steps; s += N16_UNROLL) {
        u32x4_t av[N16_UNROLL];
        f32x4_t xlo[N16_UNROLL], xhi[N16_UNROLL];
        #pragma unroll
        for (int u = 0; u < N16_UNROLL; u++) {
            av[u] = a_in ? wa[2 * (s + u)] : z4;
            xlo[u] = b_in ? xb[4 * (s + u)] : zf;
            xhi[u] = b_in ? xb[4 * (s + u) + 1] : zf;
        }
        #pragma unroll
        for (int u = 0; u < N16_UNROLL; u++) {
            N16_STEP(av[u], xlo[u], xhi[u]);
        }
    }
    for (; s < steps; s++) {
        const u32x4_t av = a_in ? wa[2 * s] : z4;
        const f32x4_t xlo = b_in ? xb[4 * s] : zf;
        const f32x4_t xhi = b_in ? xb[4 * s + 1] : zf;
        N16_STEP(av, xlo, xhi);
    }
#undef N16_STEP

    if (b_in) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 8 * k_grp + j;
            if (out_row < M) {
                Y[(long long)lane_m * M + out_row] = acc[j];
            }
        }
    }
}
