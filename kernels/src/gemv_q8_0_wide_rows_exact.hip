// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Row-exact multi-row Q8_0 GEMV, small-K (K <= 1536) arm: `gemv_q8_0_wide`
// (gemv_q8_0_wide.hip) over R activation rows in one launch, weights read once.
//
// Contract: output row r (y[r*M + row]) is BIT-IDENTICAL to what the rows == 1
// decode singleton `gemv_q8_0_wide` writes for x row r, for every R in 1..=16.
// Everything that determines a bit is the singleton's, per row:
//   * thread -> output: 64 threads / block, one weight row per wave
//     (row = blockIdx.x * 2 + wave), lane = element within a 32-wide Q8_0 block;
//   * four accumulator chains over 4-block quads (block bi feeds chain bi & 3),
//     each starting at +0.0f; quad blocks accumulate `fma(dq, x, acc)` with
//     dq = f16 scale * (float)int8 (exact in f32);
//   * the K % 4 tail blocks add into chain 0, 1, 2 as an UNFUSED `mul` then
//     `add` (the singleton's tail compiles that way on gfx1100 and gfx1201:
//     `contrib` has three conditional consumers, so the backend cannot fuse it);
//   * fold `(acc0 + acc1) + (acc2 + acc3)`, then the wave `__shfl_down` tree
//     16, 8, 4, 2, 1; lane 0 stores.
// The two contraction points are written explicitly (`__builtin_fmaf` for the
// quad blocks, `fp contract(off)` for the tail) so the choice does not depend on
// what the optimiser decides at each R. Dequantised scale/quant are loaded once
// per block and shared by the R rows; only x loads and accumulators are per row.
//
// One kernel per R (`..._r1` .. `..._r16`) so a small R does not inherit the
// register footprint of R = 16.
//
// Instruction-level evidence (hipcc `-S`, gfx1100 and gfx1201, 2026-10-31): the
// singleton's quad body is v_cvt_f32_i32 + v_fma_mix_f32(d16, q, -0) + v_fmac_f32
// per chain, its tail is v_fma_mix_f32 + v_mul_f32 + v_add_f32, its epilogue is the
// three v_add_f32 folds + ds_swizzle/ds_bpermute tree; the same sequences appear
// per row in these kernels.

template <int R>
static __device__ __forceinline__ void gemv_q8_0_wide_rows_exact_body(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    const int row = blockIdx.x * 2 + warp_id;
    if (row >= M) return;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q8 + (size_t)row * blocks_per_row * 34;

    float acc0[R], acc1[R], acc2[R], acc3[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc0[r] = acc1[r] = acc2[r] = acc3[r] = 0.0f;
    const int quads = blocks_per_row >> 2;
    const int tail = blocks_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int bi = q << 2;
        const unsigned char* bp0 = row_data + (bi + 0) * 34;
        const unsigned char* bp1 = row_data + (bi + 1) * 34;
        const unsigned char* bp2 = row_data + (bi + 2) * 34;
        const unsigned char* bp3 = row_data + (bi + 3) * 34;

        const float d0 = (float)*((const _Float16*)bp0);
        const float d1 = (float)*((const _Float16*)bp1);
        const float d2 = (float)*((const _Float16*)bp2);
        const float d3 = (float)*((const _Float16*)bp3);

        const signed char q0 = (signed char)bp0[2 + lane];
        const signed char q1 = (signed char)bp1[2 + lane];
        const signed char q2 = (signed char)bp2[2 + lane];
        const signed char q3 = (signed char)bp3[2 + lane];

        const float dq0 = d0 * (float)q0;
        const float dq1 = d1 * (float)q1;
        const float dq2 = d2 * (float)q2;
        const float dq3 = d3 * (float)q3;

#pragma unroll
        for (int r = 0; r < R; r++) {
            const float* xr = x + (size_t)r * K;
            acc0[r] = __builtin_fmaf(dq0, xr[(bi + 0) * 32 + lane], acc0[r]);
            acc1[r] = __builtin_fmaf(dq1, xr[(bi + 1) * 32 + lane], acc1[r]);
            acc2[r] = __builtin_fmaf(dq2, xr[(bi + 2) * 32 + lane], acc2[r]);
            acc3[r] = __builtin_fmaf(dq3, xr[(bi + 3) * 32 + lane], acc3[r]);
        }
    }

    // Tail blocks accumulate into acc[bi % 4]: mul then add, never fused.
    if (tail > 0) {
#pragma clang fp contract(off)
        const int bi = quads << 2;
        const unsigned char* bp = row_data + bi * 34;
        const float d = (float)*((const _Float16*)bp);
        const signed char qv = (signed char)bp[2 + lane];
        const float dq = d * (float)qv;
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float contrib = dq * x[(size_t)r * K + bi * 32 + lane];
            acc0[r] = acc0[r] + contrib;
        }
    }
    if (tail > 1) {
#pragma clang fp contract(off)
        const int bi = (quads << 2) + 1;
        const unsigned char* bp = row_data + bi * 34;
        const float d = (float)*((const _Float16*)bp);
        const signed char qv = (signed char)bp[2 + lane];
        const float dq = d * (float)qv;
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float contrib = dq * x[(size_t)r * K + bi * 32 + lane];
            acc1[r] = acc1[r] + contrib;
        }
    }
    if (tail > 2) {
#pragma clang fp contract(off)
        const int bi = (quads << 2) + 2;
        const unsigned char* bp = row_data + bi * 34;
        const float d = (float)*((const _Float16*)bp);
        const signed char qv = (signed char)bp[2 + lane];
        const float dq = d * (float)qv;
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float contrib = dq * x[(size_t)r * K + bi * 32 + lane];
            acc2[r] = acc2[r] + contrib;
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++) {
        float sum = (acc0[r] + acc1[r]) + (acc2[r] + acc3[r]);
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down(sum, offset);
        if (lane == 0) y[(size_t)r * M + row] = sum;
    }
}

#define GEMV_Q8_0_WIDE_ROWS_EXACT(R)                                          \
    extern "C" __global__ void gemv_q8_0_wide_rows_exact_r##R(                \
        const unsigned char* __restrict__ A_q8,                               \
        const float* __restrict__ x,                                          \
        float* __restrict__ y,                                                \
        int M, int K                                                          \
    ) {                                                                       \
        gemv_q8_0_wide_rows_exact_body<R>(A_q8, x, y, M, K);                  \
    }

GEMV_Q8_0_WIDE_ROWS_EXACT(1)
GEMV_Q8_0_WIDE_ROWS_EXACT(2)
GEMV_Q8_0_WIDE_ROWS_EXACT(3)
GEMV_Q8_0_WIDE_ROWS_EXACT(4)
GEMV_Q8_0_WIDE_ROWS_EXACT(5)
GEMV_Q8_0_WIDE_ROWS_EXACT(6)
GEMV_Q8_0_WIDE_ROWS_EXACT(7)
GEMV_Q8_0_WIDE_ROWS_EXACT(8)
GEMV_Q8_0_WIDE_ROWS_EXACT(9)
GEMV_Q8_0_WIDE_ROWS_EXACT(10)
GEMV_Q8_0_WIDE_ROWS_EXACT(11)
GEMV_Q8_0_WIDE_ROWS_EXACT(12)
GEMV_Q8_0_WIDE_ROWS_EXACT(13)
GEMV_Q8_0_WIDE_ROWS_EXACT(14)
GEMV_Q8_0_WIDE_ROWS_EXACT(15)
GEMV_Q8_0_WIDE_ROWS_EXACT(16)
