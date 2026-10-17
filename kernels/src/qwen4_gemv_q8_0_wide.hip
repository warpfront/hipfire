// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Qwen4 decode/verify siblings of gemv_q8_0_wide (gemv_q8_0_wide.hip).

// gfx1151, K = 640 (680-byte rows, Qwen4 decode shared-expert down): the same
// four-accumulator math on a row staged through LDS with dword loads issued
// before the x loads, x (20 values per lane) held in registers, four rows
// (waves) per block. Bitwise gemv_q8_0_wide; 10.5 -> 9.5 us per call.
template <bool kScaledAdd>
static __device__ __forceinline__ void gemv_q8_0_wide_k640_staged_body(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ residual,
    const float* __restrict__ scalar,
    int M
) {
    __shared__ __align__(16) unsigned int staged[4][170];
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + wave;
    if (row >= M) return;
    const unsigned int* src = reinterpret_cast<const unsigned int*>(A_q8 + (size_t)row * 680);
    unsigned int v[6];
#pragma unroll
    for (int u = 0; u < 5; ++u) v[u] = src[lane + u * 32];
    v[5] = lane < 10 ? src[lane + 160] : 0u;
    float xr[20];
#pragma unroll
    for (int b = 0; b < 20; ++b) xr[b] = x[b * 32 + lane];
#pragma unroll
    for (int u = 0; u < 5; ++u) staged[wave][lane + u * 32] = v[u];
    if (lane < 10) staged[wave][lane + 160] = v[5];
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
    const unsigned char* data = reinterpret_cast<const unsigned char*>(staged[wave]);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
#pragma unroll
    for (int q = 0; q < 5; ++q) {
        const unsigned char* bp = data + q * 4 * 34;
        acc0 += (float)*((const _Float16*)bp) * (float)(signed char)bp[2 + lane] * xr[q * 4 + 0];
        acc1 += (float)*((const _Float16*)(bp + 34)) * (float)(signed char)bp[36 + lane] * xr[q * 4 + 1];
        acc2 += (float)*((const _Float16*)(bp + 68)) * (float)(signed char)bp[70 + lane] * xr[q * 4 + 2];
        acc3 += (float)*((const _Float16*)(bp + 102)) * (float)(signed char)bp[104 + lane] * xr[q * 4 + 3];
    }
    float sum = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (lane != 0) return;
    y[row] = sum;
    if (kScaledAdd) {
        // bf16_scaled_add_f32 on this row (same expression and roundings).
        const hip_bfloat16 lhs = static_cast<hip_bfloat16>(residual[row]);
        const hip_bfloat16 rhs = static_cast<hip_bfloat16>(sum);
        const hip_bfloat16 gate = static_cast<hip_bfloat16>(scalar[0]);
        const hip_bfloat16 product = static_cast<hip_bfloat16>(
            static_cast<float>(gate) * static_cast<float>(rhs));
        residual[row] = static_cast<float>(static_cast<hip_bfloat16>(
            static_cast<float>(lhs) + static_cast<float>(product)));
    }
}

__launch_bounds__(128)
extern "C" __global__ void gemv_q8_0_wide_k640_staged(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M
) {
    gemv_q8_0_wide_k640_staged_body<false>(A_q8, x, y, nullptr, nullptr, M);
}

// gemv_q8_0_wide_k640_staged followed by bf16_scaled_add_f32(residual, y,
// scalar) in one launch (Qwen4 decode shared-expert down + source BF16 fold);
// y and residual are bitwise the two launches.
__launch_bounds__(128)
extern "C" __global__ void gemv_q8_0_wide_k640_staged_bf16_scaled_add(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ residual,
    const float* __restrict__ scalar,
    int M
) {
    gemv_q8_0_wide_k640_staged_body<true>(A_q8, x, y, residual, scalar, M);
}

// gemv_q8_0_wide over R activation rows (x row stride K, y row stride M):
// each weight row's blocks are read once and folded into every row's four
// accumulators in the single-row kernel's order, so each output is bitwise
// gemv_q8_0_wide's (speculative verify must match single-token decode).
template <int R>
static __device__ __forceinline__ void gemv_q8_0_wide_rows_body(
    const unsigned char* __restrict__ A_q8, const float* __restrict__ x,
    float* __restrict__ y, int M, int K, int row, int lane) {
    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q8 + (size_t)row * blocks_per_row * 34;
    float acc0[R], acc1[R], acc2[R], acc3[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc0[r] = acc1[r] = acc2[r] = acc3[r] = 0.0f;
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
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float* xr = x + (size_t)r * K;
            acc0[r] += d0 * (float)q0 * xr[(bi + 0) * 32 + lane];
            acc1[r] += d1 * (float)q1 * xr[(bi + 1) * 32 + lane];
            acc2[r] += d2 * (float)q2 * xr[(bi + 2) * 32 + lane];
            acc3[r] += d3 * (float)q3 * xr[(bi + 3) * 32 + lane];
        }
    }
    for (int t = 0; t < tail; t++) {
        const int bi = (quads << 2) + t;
        const unsigned char* bp = row_data + bi * 34;
        const float d = (float)*((const _Float16*)bp);
        const signed char qv = (signed char)bp[2 + lane];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float contrib = d * (float)qv * x[(size_t)r * K + bi * 32 + lane];
            if (t == 0) acc0[r] += contrib;
            else if (t == 1) acc1[r] += contrib;
            else if (t == 2) acc2[r] += contrib;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float sum = (acc0[r] + acc1[r]) + (acc2[r] + acc3[r]);
        for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset);
        if (lane == 0) y[(size_t)r * M + row] = sum;
    }
}

extern "C" __global__ void gemv_q8_0_wide_rows(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int rows
) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 2 + (threadIdx.x >> 5);
    if (row >= M) return;
    switch (rows) {
#define WIDE_ROWS(R) \
    case R: gemv_q8_0_wide_rows_body<R>(A_q8, x, y, M, K, row, lane); return;
    WIDE_ROWS(1) WIDE_ROWS(2) WIDE_ROWS(3) WIDE_ROWS(4) WIDE_ROWS(5) WIDE_ROWS(6) WIDE_ROWS(7) WIDE_ROWS(8)
#undef WIDE_ROWS
    }
}

// ===========================================================================
// Exact wide twin of the four-chain small-K recipe: `gemv_q8_0_wide_rows_wide_
// r16` / `_r32` (A, X, Y, M, K, R): pointers at 0, 8, 16, ints at 24, 28, 32;
// grid ceil(M/8), block 256, K % 32 == 0, R in 1..=RCAP, x row stride K, y row
// stride M. Every output is bitwise gemv_q8_0_wide (and so
// gemv_q8_0_wide_k640_staged on gfx1151, gemv_q8_0_wide_rows at R <= 8):
// four sums per lane, block b into sum b&3 in increasing b, the blocks >= 4*
// floor(nblk/4) (tail, up to three) added unfused, then
// (a0 + a1) + (a2 + a3) and the shuffle-down 16..1 tree.
//
// Schedule as qwen4_gemv_q8_0.hip's single-chain wide twins (design
// WideTwinGemvSol section 2.2): one wave32 per column, 8 columns per block,
// all 4R accumulators in the wave, X staged 8 rows x 256 elements (8 KiB, Q8
// lane map g*256 + 32*j + lane), weights loaded once per K tile as
// dq[j] = RN(float(scale_j) * float(q_j)) (exact product). Tiles start on
// multiples of 8 blocks, so tile block j is global block b0 + j with chain
// (b0 + j) & 3 == j & 3. Contraction is pinned: quad-path blocks
// __builtin_fmaf(dq, x, acc) (v_fmac_f32 in the incumbent on gfx1151/gfx1201),
// tail blocks an unfused RN(dq*x) then add (the incumbent's multi-use
// `contrib` compiles to v_mul_f32 + v_add_f32).
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void q8w_fill_x(
    float* xs, const float* __restrict__ x, int K, int R, int rb, int k0, int nb,
    int wave, int lane
) {
    const int row = rb + wave;
    if (row < R) {
        const float* xr = x + (size_t)row * K + k0 + lane;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (j < nb) xs[(wave * 8 + j) * 32 + lane] = xr[32 * j];
    }
}

static __device__ __forceinline__ void q8w_load_dq(
    float (&dq)[8], const unsigned char* wrow, int b0, int nb, int lane
) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        dq[j] = 0.0f;
        if (j < nb) {
            const unsigned char* bp = wrow + (size_t)(b0 + j) * 34;
            const float d = (float)*((const _Float16*)bp);
            const signed char q = (signed char)bp[2 + lane];
            dq[j] = d * (float)q;
        }
    }
}

// RN(RN(d*q) * x) then a separate add: no contraction.
static __device__ __forceinline__ float q8w_unfused_add(float acc, float dq, float xv) {
#pragma clang fp contract(off)
    const float contrib = dq * xv;
    return acc + contrib;
}

// One K tile: tile blocks j < quad_left are quad-path (fused), quad_left <= j
// < nb are tail (unfused). FULL: nb == quad_left == 8.
template <int RCAP, bool FULL>
static __device__ __forceinline__ void q8w_four_tile(
    float (&acc)[4][RCAP], const float (&dq)[8], float* xs, const float* __restrict__ x,
    int K, int R, int k0, int nb, int quad_left, bool live, int wave, int lane
) {
#pragma unroll
    for (int rbi = 0; rbi < RCAP / 8; ++rbi) {
        const int rb = rbi * 8;
        if (rb < R) {
            __syncthreads();
            q8w_fill_x(xs, x, K, R, rb, k0, FULL ? 8 : nb, wave, lane);
            __syncthreads();
            if (live) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    if (rb + r < R) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            if (FULL) {
                                acc[j & 3][rb + r] = __builtin_fmaf(
                                    dq[j], xs[(r * 8 + j) * 32 + lane], acc[j & 3][rb + r]);
                            } else {
                                // Last (partial) tile only. Both candidates are exact
                                // operations on identical operands; the wave-uniform
                                // select keeps the live accumulator set branch-free
                                // (structured branches here spilled at R32). Blocks
                                // j >= nb do not exist and leave the chain untouched.
                                const float xv = xs[(r * 8 + j) * 32 + lane];
                                const float a = acc[j & 3][rb + r];
                                const float quad = __builtin_fmaf(dq[j], xv, a);
                                const float tail = q8w_unfused_add(a, dq[j], xv);
                                acc[j & 3][rb + r] = j < quad_left ? quad : (j < nb ? tail : a);
                            }
                        }
                        // Scheduling hint only (no arithmetic effect): bound the
                        // row's LDS prefetch to keep the 4R accumulators resident.
                        __builtin_amdgcn_sched_barrier(0);
                    }
                }
            }
        }
    }
}

template <int RCAP>
static __device__ __forceinline__ void q8w_four_chain_body(
    const unsigned char* __restrict__ A_q8, const float* __restrict__ x, float* __restrict__ y,
    int M, int K, int R, float* xs
) {
    const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 5));
    const int lane = threadIdx.x & 31;
    const int m = (int)blockIdx.x * 8 + wave;
    const bool live = m < M;
    const int nblk = K >> 5;
    const unsigned char* wrow = A_q8 + (size_t)(live ? m : 0) * nblk * 34;
    float acc[4][RCAP];
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int r = 0; r < RCAP; ++r) acc[c][r] = 0.0f;
    // Full 8-block tiles in the loop (single code path over the accumulators),
    // then at most one partial tile peeled after it. For a full tile
    // b0 + 8 <= nblk, hence <= quad_end: all eight blocks are quad-path.
    const int nfull = nblk >> 3;
    for (int t = 0; t < nfull; ++t) {
        const int b0 = t * 8;
        float dq[8];
        if (live) {
            q8w_load_dq(dq, wrow, b0, 8, lane);
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) dq[j] = 0.0f;
        }
        q8w_four_tile<RCAP, true>(acc, dq, xs, x, K, R, b0 * 32, 8, 8, live, wave, lane);
    }
    const int ntail = nblk & 7;
    if (ntail != 0) {
        // nb in 1..7: blocks 0..3 of the tile are quad-path iff nb >= 4 (quad_end
        // is a multiple of 4), the rest are the (up to three) tail blocks.
        const int b0 = nfull * 8;
        float dq[8];
        if (live) {
            q8w_load_dq(dq, wrow, b0, ntail, lane);
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) dq[j] = 0.0f;
        }
        q8w_four_tile<RCAP, false>(acc, dq, xs, x, K, R, b0 * 32, ntail, ntail & 4, live, wave, lane);
    }
    if (live) {
#pragma unroll
        for (int r = 0; r < RCAP; ++r) {
            if (r < R) {
                float sum = (acc[0][r] + acc[1][r]) + (acc[2][r] + acc[3][r]);
                for (int offset = 16; offset > 0; offset >>= 1)
                    sum += __shfl_down(sum, offset);
                if (lane == 0) y[(size_t)r * M + m] = sum;
            }
        }
    }
}

#define GEMV_Q8_0_WIDE_ROWS_WIDE(NAME, RCAP)                                          \
    __launch_bounds__(256) extern "C" __global__ void NAME(                           \
        const unsigned char* __restrict__ A_q8, const float* __restrict__ x,          \
        float* __restrict__ y, int M, int K, int R) {                                 \
        __shared__ __align__(16) float xs[8 * 8 * 32];                                \
        q8w_four_chain_body<RCAP>(A_q8, x, y, M, K, R, xs);                           \
    }
GEMV_Q8_0_WIDE_ROWS_WIDE(gemv_q8_0_wide_rows_wide_r16, 16)
GEMV_Q8_0_WIDE_ROWS_WIDE(gemv_q8_0_wide_rows_wide_r32, 32)
#undef GEMV_Q8_0_WIDE_ROWS_WIDE
