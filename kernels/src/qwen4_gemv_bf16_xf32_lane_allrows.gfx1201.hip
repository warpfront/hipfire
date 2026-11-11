// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx1201 lane-step HC all-rows k4 GEMV (HIPFIRE_QWEN4_LANE_HC_ALLROWS).
//
// `gemv_bf16_xf32_k4_rows_lane_t8` is `gemv_bf16_xf32_k4_rows_tiled_r8`
// (qwen4_gemv_bf16_xf32.hip) with the partial final tile folded into the same
// launch: grid [M, ceil(rows / 8)], block 128, blockIdx.y selects an eight-row
// tile whose activation rows (x row stride K) and output rows (y row stride M)
// start 8 * blockIdx.y rows in; `rows` is the runtime row count. A full tile
// runs `gemv_bf16_xf32_row_multi<8>`; the last tile, when it has V < 8 valid
// rows, runs the same body instantiated at R = V (exactly what the
// `_k4_rows_tiled_r<V>` tail launch ran), so no lane ever loads an activation
// row past `rows` or stores an output row past it.
//
// The device arithmetic is the incumbent's, verbatim: `gemv_bf16_xf32_row`'s
// per-chain order, `gemv_bf16_xf32_row_multi<R>` and the
// `gemv_bf16_xf32_k4_rows_body<R>` quarter fold and `hc_act` epilogue of
// qwen4_gemv_bf16_xf32.hip. Every row's accumulators are its own (weights are
// shared, activations are per row), so a row's value does not depend on the
// tile it sits in, on its slot or on the row count: it is bitwise the
// single-row `gemv_bf16_xf32_k4` value, the same one the <= 8-row
// `_k4_rows_r<R>` / `_otile_r<R>` launches give it.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#define HF_BF16_LO(u) __uint_as_float((u) << 16)
#define HF_BF16_HI(u) __uint_as_float((u) & 0xFFFF0000u)

// verbatim: qwen4_gemv_bf16_xf32.hip `gemv_bf16_xf32_row_multi` (R rows).
template <int R>
static __device__ __forceinline__ void lane_gemv_bf16_xf32_row_multi(
    const unsigned short* __restrict__ w_row,
    const float*          __restrict__ x,
    int x_stride,
    int K,
    int tid,
    float (&sum)[R]
) {
    float a0[R], a1[R], a2[R], a3[R];
#pragma unroll
    for (int r = 0; r < R; ++r) a0[r] = a1[r] = a2[r] = a3[r] = 0.0f;
    if ((K & 7) == 0) {
        for (int i = tid * 8; i + 8 <= K; i += 32 * 8) {
            const uint4 wv = *(const uint4*)(w_row + i);
            const float w0 = HF_BF16_LO(wv.x), w1 = HF_BF16_HI(wv.x);
            const float w2 = HF_BF16_LO(wv.y), w3 = HF_BF16_HI(wv.y);
            const float w4 = HF_BF16_LO(wv.z), w5 = HF_BF16_HI(wv.z);
            const float w6 = HF_BF16_LO(wv.w), w7 = HF_BF16_HI(wv.w);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float4 xa = *(const float4*)(x + (size_t)r * x_stride + i);
                const float4 xb = *(const float4*)(x + (size_t)r * x_stride + i + 4);
                a0[r] += w0 * xa.x;
                a1[r] += w1 * xa.y;
                a2[r] += w2 * xa.z;
                a3[r] += w3 * xa.w;
                a0[r] += w4 * xb.x;
                a1[r] += w5 * xb.y;
                a2[r] += w6 * xb.z;
                a3[r] += w7 * xb.w;
            }
        }
    }
    for (int i = ((K & 7) == 0 ? (K & ~7) : 0) + tid; i < K; i += 32) {
        const float w = __uint_as_float(((unsigned int)w_row[i]) << 16);
#pragma unroll
        for (int r = 0; r < R; ++r) a0[r] += w * x[(size_t)r * x_stride + i];
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = (a0[r] + a1[r]) + (a2[r] + a3[r]);
        for (int offset = 16; offset > 0; offset >>= 1) {
            s += __shfl_down(s, offset);
        }
        sum[r] = s;
    }
}

// `gemv_bf16_xf32_k4_rows_body` of one tile of `valid` (1..=8) real rows.
static __device__ __forceinline__ void lane_gemv_bf16_xf32_k4_rows_tile(
    const unsigned short* __restrict__ W,
    const float*          __restrict__ x,   // first row of this tile
    float*                __restrict__ y,   // first output row of this tile
    int M, int K,
    float scale,
    int hc_act,
    int valid
) {
    constexpr int R = 8;
    __shared__ float quarter[R][4];
    const int row = blockIdx.x;
    if (row >= M) return;
    const int wave = threadIdx.x >> 5;
    const int part = K / 4;
    float s[R];
    if (valid == R) {
        lane_gemv_bf16_xf32_row_multi<R>(
            W + (size_t)row * K + wave * part, x + wave * part, K, part, threadIdx.x & 31, s);
    } else {
        // Partial tile: run the same body over the valid rows only (template
        // arms 1..7), so no slot ever loads a row past `valid`.
        switch (valid) {
#define LANE_PARTIAL_ARM(V)                                                              \
    case V: {                                                                           \
        float sv[V];                                                                    \
        lane_gemv_bf16_xf32_row_multi<V>(                                               \
            W + (size_t)row * K + wave * part, x + wave * part, K, part,                \
            threadIdx.x & 31, sv);                                                      \
        for (int r = 0; r < V; ++r) s[r] = sv[r];                                       \
        for (int r = V; r < R; ++r) s[r] = 0.0f;                                        \
        break;                                                                          \
    }
            LANE_PARTIAL_ARM(1)
            LANE_PARTIAL_ARM(2)
            LANE_PARTIAL_ARM(3)
            LANE_PARTIAL_ARM(4)
            LANE_PARTIAL_ARM(5)
            LANE_PARTIAL_ARM(6)
            LANE_PARTIAL_ARM(7)
#undef LANE_PARTIAL_ARM
            default:
                break;
        }
    }
    if ((threadIdx.x & 31) == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) quarter[r][wave] = s[r];
    }
    __syncthreads();
    const int r = threadIdx.x;
    if (r >= valid) return;
    float value = (quarter[r][0] + quarter[r][1]) + (quarter[r][2] + quarter[r][3]);
    if (hc_act) {
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
        value *= scale;
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
        value = value / (1.0f + expf(-value));
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
    }
    y[(size_t)r * M + row] = value;
}

__launch_bounds__(128) extern "C" __global__ void gemv_bf16_xf32_k4_rows_lane_t8(
    const unsigned short* __restrict__ W,
    const float*          __restrict__ x,
    float*                __restrict__ y,
    int M, int K,
    float scale,
    int hc_act,
    int rows
) {
    const int first = blockIdx.y * 8;
    const int left = rows - first;
    const int valid = left < 8 ? left : 8;
    lane_gemv_bf16_xf32_k4_rows_tile(
        W, x + (size_t)first * K, y + (size_t)first * M, M, K, scale, hc_act, valid);
}
