// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx1201 all-rows BF16 GEMV (HIPFIRE_QWEN4_BF16_ALLROWS): up to four BF16 matrices reading the
// same [rows, K] F32 x (the Flash-Next MoE call's router + shared selector + shared gate/up, or
// the shared down), 9..=64 rows, in ONE launch with every weight loaded once per block.
//
// Per output element the arithmetic is `gemv_bf16_xf32`'s, verbatim: lane `l` walks the K
// positions `l * 8 + 256 * j` (j ascending, K % 8 == 0; the scalar tail otherwise), folds each
// eight-value step into four accumulators in the singleton's order (`a0 += w0 * x.x` ...
// `a3 += w7 * xb.w`), takes `(a0 + a1) + (a2 + a3)` and the wave-shuffle tree. Every (output row,
// token row) pair owns its own accumulators, so a token row's value never depends on the other
// token rows, the output tile it sits in, the row count or its slot: it is bitwise the single-row
// `gemv_bf16_xf32` / `gemv_bf16_xf32_x4` value, and the same one `gemm_bf16_xf32_multirow` and
// `gemv_bf16_xf32_x4_rows_r<R>` give it.
//
// Work split. Grid [tiles]: a tile is TM consecutive output rows of one part (tiles of the four
// parts concatenated, a part's last tile clamped to its last row; the clamped duplicates are
// never stored). Block = ceil(rows / TN) waves: wave w owns token rows [w * TN, w * TN + TN),
// the last wave's slots past `rows` alias its last valid row (no activation row past `rows` is
// read) and are never stored. A wave loads each weight uint4 of its TM rows once and applies it
// to its TN token rows; the waves of a block read the same TM weight rows, so the block reads
// each weight from memory once and every other wave hits the CU's cache. TM output rows share
// every activation load, so the activation traffic per FMA is 1 / TM of the one-output-row wave
// the incumbent runs.

#include <hip/hip_runtime.h>

#define HF_BF16_LO(u) __uint_as_float((u) << 16)
#define HF_BF16_HI(u) __uint_as_float((u) & 0xFFFF0000u)

// Lane 0 of `v += __shfl_down(v, off, 32)` for off = 16, 8, 4, 2, 1 (the incumbent's tree; the
// verbatim helper of gemm_bf16_xf32_multirow.hip). Only lane 0's result is defined.
static __device__ __forceinline__ float hf_lane0_tree_sum(float v) {
    v += __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x108, 0xf, 0xf, true));  // row_shl:8
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x104, 0xf, 0xf, true));  // row_shl:4
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x102, 0xf, 0xf, true));  // row_shl:2
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x101, 0xf, 0xf, true));  // row_shl:1
    return v;
}

template <int TM, int TN, int UNROLL>
static __device__ __forceinline__ void bf16_allrows_body(
    const unsigned short* __restrict__ W0,
    const unsigned short* __restrict__ W1,
    const unsigned short* __restrict__ W2,
    const unsigned short* __restrict__ W3,
    const float*          __restrict__ x,
    float* __restrict__ y0,
    float* __restrict__ y1,
    float* __restrict__ y2,
    float* __restrict__ y3,
    int M0, int M1, int M2, int M3, int K, int rows
) {
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n0 = wave * TN;
    if (n0 >= rows) return;

    // Tile -> (part, first output row).
    int tile = blockIdx.x;
    const unsigned short* W = W0;
    float* y = y0;
    int M = M0;
    {
        const int t0 = (M0 + TM - 1) / TM;
        const int t1 = (M1 + TM - 1) / TM;
        const int t2 = (M2 + TM - 1) / TM;
        if (tile >= t0) {
            tile -= t0; W = W1; y = y1; M = M1;
            if (tile >= t1) {
                tile -= t1; W = W2; y = y2; M = M2;
                if (tile >= t2) {
                    tile -= t2; W = W3; y = y3; M = M3;
                }
            }
        }
    }
    const int row0 = tile * TM;
    if (row0 >= M) return;

    const unsigned short* w_row[TM];
#pragma unroll
    for (int t = 0; t < TM; ++t) {
        const int r = (row0 + t < M) ? row0 + t : M - 1;
        w_row[t] = W + (size_t)r * K;
    }
    const float* xr[TN];
#pragma unroll
    for (int n = 0; n < TN; ++n) {
        const int b = (n0 + n < rows) ? n0 + n : rows - 1;
        xr[n] = x + (size_t)b * K;
    }

    float a0[TM][TN], a1[TM][TN], a2[TM][TN], a3[TM][TN];
#pragma unroll
    for (int t = 0; t < TM; ++t)
#pragma unroll
        for (int n = 0; n < TN; ++n) a0[t][n] = a1[t][n] = a2[t][n] = a3[t][n] = 0.0f;

    if ((K & 7) == 0) {
#pragma unroll UNROLL
        for (int i = lane * 8; i + 8 <= K; i += 32 * 8) {
            uint4 wv[TM];
#pragma unroll
            for (int t = 0; t < TM; ++t) wv[t] = *(const uint4*)(w_row[t] + i);
#pragma unroll
            for (int n = 0; n < TN; ++n) {
                const float4 xa = *(const float4*)(xr[n] + i);
                const float4 xb = *(const float4*)(xr[n] + i + 4);
#pragma unroll
                for (int t = 0; t < TM; ++t) {
                    a0[t][n] += HF_BF16_LO(wv[t].x) * xa.x;
                    a1[t][n] += HF_BF16_HI(wv[t].x) * xa.y;
                    a2[t][n] += HF_BF16_LO(wv[t].y) * xa.z;
                    a3[t][n] += HF_BF16_HI(wv[t].y) * xa.w;
                    a0[t][n] += HF_BF16_LO(wv[t].z) * xb.x;
                    a1[t][n] += HF_BF16_HI(wv[t].z) * xb.y;
                    a2[t][n] += HF_BF16_LO(wv[t].w) * xb.z;
                    a3[t][n] += HF_BF16_HI(wv[t].w) * xb.w;
                }
            }
        }
    }
    // Scalar tail: the final K % 8 elements (all of K when K % 8 != 0), the singleton's.
    for (int i = ((K & 7) == 0 ? (K & ~7) : 0) + lane; i < K; i += 32) {
#pragma unroll
        for (int t = 0; t < TM; ++t) {
            const float w = __uint_as_float(((unsigned int)w_row[t][i]) << 16);
#pragma unroll
            for (int n = 0; n < TN; ++n) a0[t][n] += w * xr[n][i];
        }
    }

#pragma unroll
    for (int t = 0; t < TM; ++t) {
#pragma unroll
        for (int n = 0; n < TN; ++n) {
            const float s = hf_lane0_tree_sum((a0[t][n] + a1[t][n]) + (a2[t][n] + a3[t][n]));
            if (lane == 0 && row0 + t < M && n0 + n < rows) y[(size_t)(n0 + n) * M + row0 + t] = s;
        }
    }
}

// Entry ABI is gemv_bf16_xf32_x4_rows_r<R>'s plus the runtime row count: W0..W3, x, y0..y3,
// M0..M3, K, rows. Unused parts have M = 0.
#define BF16_ALLROWS_ENTRY(TM, TN, U, SUF, BOUNDS)                                           \
    __launch_bounds__(BOUNDS) extern "C" __global__ void                                      \
    gemv_bf16_xf32_x4_rows_allrows_m##TM##t##TN##SUF(                                         \
        const unsigned short* __restrict__ W0, const unsigned short* __restrict__ W1,        \
        const unsigned short* __restrict__ W2, const unsigned short* __restrict__ W3,        \
        const float* __restrict__ x, float* __restrict__ y0, float* __restrict__ y1,         \
        float* __restrict__ y2, float* __restrict__ y3, int M0, int M1, int M2, int M3,      \
        int K, int rows) {                                                                   \
        bf16_allrows_body<TM, TN, U>(W0, W1, W2, W3, x, y0, y1, y2, y3, M0, M1, M2, M3, K,   \
                                     rows);                                                  \
    }

BF16_ALLROWS_ENTRY(1, 4, 1, , 512)
BF16_ALLROWS_ENTRY(2, 4, 10, u10, 512)
BF16_ALLROWS_ENTRY(4, 4, 1, , 512)
BF16_ALLROWS_ENTRY(4, 4, 5, u5, 512)
