// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Row-exact multi-row Q8_0 GEMV, large-K (K > 1536) arm: `gemv_q8_0`
// (gemv_q8_0.hip) over R activation rows in one launch, weights read once.
//
// Contract: output row r (y[r*M + row]) is BIT-IDENTICAL to what the rows == 1
// decode singleton `gemv_q8_0` writes for x row r, for every R in 1..=16.
// Everything that determines a bit is the singleton's, per row:
//   * thread -> output: one 32-thread wave per weight row (row = blockIdx.x),
//     lane = element within a 32-wide Q8_0 block;
//   * ONE accumulator chain per lane, starting at +0.0f, over blocks in
//     ascending order, `fma(dq, x, acc)` with dq = f16 scale * (float)int8
//     (exact in f32) in the 8-block unrolled body and in the remainder alike;
//   * the wave `__shfl_down` tree 16, 8, 4, 2, 1; lane 0 stores.
// The fma is written explicitly (`__builtin_fmaf`) so the contraction does not
// depend on what the optimiser decides at each R. The scale/quant of a block are
// loaded once and shared by the R rows; only x loads and accumulators are per row.
//
// One kernel per R (`..._r1` .. `..._r16`) so a small R does not inherit the
// register footprint of R = 16. The singleton's `__launch_bounds__(32, 20)` only
// caps registers, it never changes a value, so it is not repeated.
//
// Instruction-level evidence (hipcc `-S`, gfx1100 and gfx1201, 2026-10-31): the
// singleton's body is v_cvt_f32_i32 + v_fma_mix_f32(d16, q, -0) + one v_fmac_f32
// chain, tail and unrolled body alike, then the ds_swizzle/ds_bpermute v_add_f32
// tree; the same sequences appear per row in these kernels.

template <int R>
static __device__ __forceinline__ void gemv_q8_0_rows_exact_body(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q8 + (size_t)row * blocks_per_row * 34;

    float sum[R];
#pragma unroll
    for (int r = 0; r < R; r++) sum[r] = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 34;
        const float d = (float)*((const _Float16*)block);
        const signed char qval = (signed char)block[2 + tid];
        const float dq = d * (float)qval;
#pragma unroll
        for (int r = 0; r < R; r++)
            sum[r] = __builtin_fmaf(dq, x[(size_t)r * K + bi * 32 + tid], sum[r]);
    }

#pragma unroll
    for (int r = 0; r < R; r++) {
        float s = sum[r];
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down(s, offset);
        if (tid == 0) y[(size_t)r * M + row] = s;
    }
}

#define GEMV_Q8_0_ROWS_EXACT(R)                                               \
    __launch_bounds__(32)                                                     \
    extern "C" __global__ void gemv_q8_0_rows_exact_r##R(                     \
        const unsigned char* __restrict__ A_q8,                               \
        const float* __restrict__ x,                                          \
        float* __restrict__ y,                                                \
        int M, int K                                                          \
    ) {                                                                       \
        gemv_q8_0_rows_exact_body<R>(A_q8, x, y, M, K);                       \
    }

GEMV_Q8_0_ROWS_EXACT(1)
GEMV_Q8_0_ROWS_EXACT(2)
GEMV_Q8_0_ROWS_EXACT(3)
GEMV_Q8_0_ROWS_EXACT(4)
GEMV_Q8_0_ROWS_EXACT(5)
GEMV_Q8_0_ROWS_EXACT(6)
GEMV_Q8_0_ROWS_EXACT(7)
GEMV_Q8_0_ROWS_EXACT(8)
GEMV_Q8_0_ROWS_EXACT(9)
GEMV_Q8_0_ROWS_EXACT(10)
GEMV_Q8_0_ROWS_EXACT(11)
GEMV_Q8_0_ROWS_EXACT(12)
GEMV_Q8_0_ROWS_EXACT(13)
GEMV_Q8_0_ROWS_EXACT(14)
GEMV_Q8_0_ROWS_EXACT(15)
GEMV_Q8_0_ROWS_EXACT(16)
