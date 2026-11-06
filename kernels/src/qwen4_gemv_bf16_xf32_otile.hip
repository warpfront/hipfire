// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Re-scheduled twins of the Qwen4 decode BF16 rows GEMVs
// (`gemv_bf16_xf32_k4_rows_r<R>` and `gemv_bf16_xf32_x4_rows_r<R>` of
// qwen4_gemv_bf16_xf32.hip; module `qwen4_gemv_bf16_xf32_otile`,
// `HIPFIRE_QWEN4_BF16_ROWS_OTILE`, default off).
//
// The per-(row, output) arithmetic is the incumbent's, verbatim: per
// 256-element group lane l owns bf16 [8l, 8l + 8), chains a0..a3 take
// `a0 += w0 * x0; a1 += w1 * x1; a2 += w2 * x2; a3 += w3 * x3; a0 += w4 * x4;
// ...` in group order, the fold is (a0 + a1) + (a2 + a3), then the shfl_down
// 16, 8, 4, 2, 1 tree (k4: quarter fold (q0 + q1) + (q2 + q3) and the
// `hc_act` epilogue). The expressions are the incumbent's text, so the
// compiler contracts them the same way; which wave computes which output and
// when its loads issue is free. `otile_bf16_rows_oracle` proves every output
// word equal on real weights.
//
// What changes is the load schedule and the tiling:
//  * NO outputs per wave share every X load (k4 ships NO = 2: X traffic, ~16x
//    the weight bytes in the incumbent, halves);
//  * FG > 0 software-pipelines the first FG full 256-element groups of a wave's
//    K slice: weights WP groups ahead, X one group ahead, the issue points
//    pinned with scheduling barriers (the compiler otherwise interleaves each
//    group's loads with their waits and keeps few loads in flight). The groups
//    after FG, and K slices shorter than FG groups, run the plain loop.
//
// Both entries require K % 8 == 0 (the incumbent's scalar tail is not carried;
// the launcher falls back to the incumbent for other K) and 2 <= R <= 8.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#define HF_BF16_LO(u) __uint_as_float((u) << 16)
#define HF_BF16_HI(u) __uint_as_float((u) & 0xFFFF0000u)

// One wave's K slice [0, K) of NO weight rows against R activation rows
// (x row stride `x_stride`). The partial sums stay per lane:
// acc[o][r] = (a0 + a1) + (a2 + a3), before the shuffle tree.
template <int R, int NO, int FG, int WP>
static __device__ __forceinline__ void otile_dot(
    const unsigned short* const (&w)[NO],
    const float* __restrict__ x,
    int x_stride,
    int K,
    int tid,
    float (&acc)[NO][R]
) {
    constexpr int FGE = FG > 0 ? FG : 1;
    float a0[NO][R], a1[NO][R], a2[NO][R], a3[NO][R];
#pragma unroll
    for (int o = 0; o < NO; ++o) {
#pragma unroll
        for (int r = 0; r < R; ++r) a0[o][r] = a1[o][r] = a2[o][r] = a3[o][r] = 0.0f;
    }
    int i = tid * 8;
    if (FG > 0 && K >= FG * 256) {
        uint4 wv[NO][FGE];
        float4 xa[FGE][R], xb[FGE][R];
#pragma unroll
        for (int g = 0; g < (WP < FG ? WP : FG); ++g) {
#pragma unroll
            for (int o = 0; o < NO; ++o) wv[o][g] = *(const uint4*)(w[o] + i + g * 256);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            xa[0][r] = *(const float4*)(x + (size_t)r * x_stride + i);
            xb[0][r] = *(const float4*)(x + (size_t)r * x_stride + i + 4);
        }
#pragma unroll
        for (int g = 0; g < FG; ++g) {
            __builtin_amdgcn_sched_barrier(0);
            if (g + WP < FG) {
#pragma unroll
                for (int o = 0; o < NO; ++o) wv[o][g + WP < FG ? g + WP : 0] = *(const uint4*)(w[o] + i + (g + WP) * 256);
            }
            if (g + 1 < FG) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    xa[g + 1 < FG ? g + 1 : 0][r] = *(const float4*)(x + (size_t)r * x_stride + i + (g + 1) * 256);
                    xb[g + 1 < FG ? g + 1 : 0][r] = *(const float4*)(x + (size_t)r * x_stride + i + (g + 1) * 256 + 4);
                }
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int o = 0; o < NO; ++o) {
                    a0[o][r] += HF_BF16_LO(wv[o][g].x) * xa[g][r].x;
                    a1[o][r] += HF_BF16_HI(wv[o][g].x) * xa[g][r].y;
                    a2[o][r] += HF_BF16_LO(wv[o][g].y) * xa[g][r].z;
                    a3[o][r] += HF_BF16_HI(wv[o][g].y) * xa[g][r].w;
                    a0[o][r] += HF_BF16_LO(wv[o][g].z) * xb[g][r].x;
                    a1[o][r] += HF_BF16_HI(wv[o][g].z) * xb[g][r].y;
                    a2[o][r] += HF_BF16_LO(wv[o][g].w) * xb[g][r].z;
                    a3[o][r] += HF_BF16_HI(wv[o][g].w) * xb[g][r].w;
                }
            }
        }
        i += FG * 256;
    }
    for (; i + 8 <= K; i += 256) {
        uint4 wv[NO];
#pragma unroll
        for (int o = 0; o < NO; ++o) wv[o] = *(const uint4*)(w[o] + i);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float4 xa = *(const float4*)(x + (size_t)r * x_stride + i);
            const float4 xb = *(const float4*)(x + (size_t)r * x_stride + i + 4);
#pragma unroll
            for (int o = 0; o < NO; ++o) {
                a0[o][r] += HF_BF16_LO(wv[o].x) * xa.x;
                a1[o][r] += HF_BF16_HI(wv[o].x) * xa.y;
                a2[o][r] += HF_BF16_LO(wv[o].y) * xa.z;
                a3[o][r] += HF_BF16_HI(wv[o].y) * xa.w;
                a0[o][r] += HF_BF16_LO(wv[o].z) * xb.x;
                a1[o][r] += HF_BF16_HI(wv[o].z) * xb.y;
                a2[o][r] += HF_BF16_LO(wv[o].w) * xb.z;
                a3[o][r] += HF_BF16_HI(wv[o].w) * xb.w;
            }
        }
    }
#pragma unroll
    for (int o = 0; o < NO; ++o) {
#pragma unroll
        for (int r = 0; r < R; ++r) acc[o][r] = (a0[o][r] + a1[o][r]) + (a2[o][r] + a3[o][r]);
    }
}

// gemv_bf16_xf32_k4_rows_body<R> over NO output rows per workgroup: grid
// [ceil(M / NO)], block 128 (wave = K quarter), K % 32 == 0.
template <int R, int NO, int FG, int WP>
static __device__ __forceinline__ void gemv_bf16_xf32_k4_rows_otile_body(
    const unsigned short* __restrict__ W,
    const float*          __restrict__ x,
    float*                __restrict__ y,
    int M, int K,
    float scale,
    int hc_act
) {
    __shared__ float quarter[NO][R][4];
    const int row0 = blockIdx.x * NO;
    const int wave = threadIdx.x >> 5;
    const int part = K / 4;
    const unsigned short* w[NO];
#pragma unroll
    for (int o = 0; o < NO; ++o) {
        const int row = row0 + o < M ? row0 + o : M - 1;
        w[o] = W + (size_t)row * K + wave * part;
    }
    float s[NO][R];
    otile_dot<R, NO, FG, WP>(w, x + wave * part, K, part, threadIdx.x & 31, s);
#pragma unroll
    for (int o = 0; o < NO; ++o) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float v = s[o][r];
            for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down(v, offset);
            s[o][r] = v;
        }
    }
    if ((threadIdx.x & 31) == 0) {
#pragma unroll
        for (int o = 0; o < NO; ++o) {
#pragma unroll
            for (int r = 0; r < R; ++r) quarter[o][r][wave] = s[o][r];
        }
    }
    __syncthreads();
    const int t = threadIdx.x;
    if (t >= NO * R) return;
    const int o = t / R;
    const int r = t - o * R;
    if (row0 + o >= M) return;
    float value = (quarter[o][r][0] + quarter[o][r][1]) + (quarter[o][r][2] + quarter[o][r][3]);
    if (hc_act) {
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
        value *= scale;
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
        value = value / (1.0f + expf(-value));
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
    }
    y[(size_t)r * M + row0 + o] = value;
}

// gemv_bf16_xf32_x4_rows over NO output rows per wave: a wave's tile is NO
// consecutive rows of one of the four matrices (ceil(M_p / NO) tiles each,
// laid out part after part), `tile` = wave index. K % 8 == 0.
template <int R, int NO, int FG, int WP>
static __device__ __forceinline__ void gemv_bf16_xf32_x4_rows_otile_body(
    const unsigned short* __restrict__ W0, const unsigned short* __restrict__ W1,
    const unsigned short* __restrict__ W2, const unsigned short* __restrict__ W3,
    const float* __restrict__ x, float* __restrict__ y0, float* __restrict__ y1,
    float* __restrict__ y2, float* __restrict__ y3, int M0, int M1, int M2, int M3,
    int K, int tile
) {
    const unsigned short* W = W0;
    float* y = y0;
    int M = M0;
    const int t0 = (M0 + NO - 1) / NO;
    const int t1 = (M1 + NO - 1) / NO;
    const int t2 = (M2 + NO - 1) / NO;
    const int t3 = (M3 + NO - 1) / NO;
    if (tile >= t0) {
        tile -= t0; W = W1; y = y1; M = M1;
        if (tile >= t1) {
            tile -= t1; W = W2; y = y2; M = M2;
            if (tile >= t2) {
                tile -= t2; W = W3; y = y3; M = M3;
                if (tile >= t3) return;
            }
        }
    }
    const int row0 = tile * NO;
    const int tid = threadIdx.x & 31;
    const unsigned short* w[NO];
#pragma unroll
    for (int o = 0; o < NO; ++o) {
        const int row = row0 + o < M ? row0 + o : M - 1;
        w[o] = W + (size_t)row * K;
    }
    float s[NO][R];
    otile_dot<R, NO, FG, WP>(w, x, K, K, tid, s);
#pragma unroll
    for (int o = 0; o < NO; ++o) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float v = s[o][r];
            for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down(v, offset);
            if (tid == 0 && row0 + o < M) y[(size_t)r * M + row0 + o] = v;
        }
    }
}

// ---- entries ---------------------------------------------------------------
// k4: two output rows per workgroup (host grid ceil(M / 2),
// BF16_K4_OTILE_ROWS_PER_BLOCK in gemv.rs), the ten 256-element groups of a
// quarter pipelined, weights two groups ahead.
#define GEMV_BF16_XF32_K4_ROWS_OTILE(R)                                                  \
    __launch_bounds__(128) extern "C" __global__ void gemv_bf16_xf32_k4_rows_otile_r##R( \
        const unsigned short* __restrict__ W, const float* __restrict__ x,               \
        float* __restrict__ y, int M, int K, float scale, int hc_act) {                  \
        gemv_bf16_xf32_k4_rows_otile_body<R, 2, 10, 2>(W, x, y, M, K, scale, hc_act);    \
    }
GEMV_BF16_XF32_K4_ROWS_OTILE(2)
GEMV_BF16_XF32_K4_ROWS_OTILE(3)
GEMV_BF16_XF32_K4_ROWS_OTILE(4)
GEMV_BF16_XF32_K4_ROWS_OTILE(5)
GEMV_BF16_XF32_K4_ROWS_OTILE(6)
GEMV_BF16_XF32_K4_ROWS_OTILE(7)
GEMV_BF16_XF32_K4_ROWS_OTILE(8)
#undef GEMV_BF16_XF32_K4_ROWS_OTILE

// x4: one output row per wave (grid = sum of M, as the incumbent; the win is the
// load schedule): R <= 7 the plain loop, R == 8 the first ten groups pipelined
// (the plain loop is slower there: 26 vs 22 us on the gate-side call).
#define GEMV_BF16_XF32_X4_ROWS_OTILE(R, FG, WP)                                            \
    __launch_bounds__(32) extern "C" __global__ void gemv_bf16_xf32_x4_rows_otile_r##R(    \
        const unsigned short* __restrict__ W0, const unsigned short* __restrict__ W1,      \
        const unsigned short* __restrict__ W2, const unsigned short* __restrict__ W3,      \
        const float* __restrict__ x, float* __restrict__ y0, float* __restrict__ y1,       \
        float* __restrict__ y2, float* __restrict__ y3, int M0, int M1, int M2, int M3,    \
        int K) {                                                                           \
        gemv_bf16_xf32_x4_rows_otile_body<R, 1, FG, WP>(                                   \
            W0, W1, W2, W3, x, y0, y1, y2, y3, M0, M1, M2, M3, K, (int)blockIdx.x);        \
    }
GEMV_BF16_XF32_X4_ROWS_OTILE(2, 0, 0)
GEMV_BF16_XF32_X4_ROWS_OTILE(3, 0, 0)
GEMV_BF16_XF32_X4_ROWS_OTILE(4, 0, 0)
GEMV_BF16_XF32_X4_ROWS_OTILE(5, 0, 0)
GEMV_BF16_XF32_X4_ROWS_OTILE(6, 0, 0)
GEMV_BF16_XF32_X4_ROWS_OTILE(7, 0, 0)
GEMV_BF16_XF32_X4_ROWS_OTILE(8, 10, 2)
#undef GEMV_BF16_XF32_X4_ROWS_OTILE
