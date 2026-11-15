// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// qwen4_g942_hc_gates: the exact-gfx942 batched k4 HC write-gate entries
// `gemv_bf16_xf32_k4_rows_tiled_r2` .. `_r8`, as their own module (gfx942 only).
//
// A gfx942 lane step's all-rows HC segment (and the ordinary prefill chunk's k4 gates, where
// one runs) computes each write's four gates with these entries: per branch the four-quarter
// `gemv_bf16_xf32_row`-order dot folded `(q0 + q1) + (q2 + q3)`, every row bitwise the
// single-row `gemv_bf16_xf32_k4` kernel's. `qwen4_gemv_bf16_xf32.hip` is JIT-compiled by every
// arch and is not edited for this route, so the blocks between the BEGIN / END markers are
// verbatim copies of it: the bf16 widen macros, the multi-row dot, the k4 rows body and the
// tiled entry macro with its seven instantiations. A unit test in rdna-compute
// (`g942_hc_gates_source_is_qwen4_gemv_bf16_xf32_verbatim`) asserts each block occurs byte
// for byte in QWEN4_GEMV_BF16_XF32_SRC, so an edit of the production body that is not
// mirrored here fails the build. Grid [M, tiles], block 128, K % 32 == 0.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// BEGIN verbatim qwen4_gemv_bf16_xf32.hip
#define HF_BF16_LO(u) __uint_as_float((u) << 16)
#define HF_BF16_HI(u) __uint_as_float((u) & 0xFFFF0000u)
// END verbatim qwen4_gemv_bf16_xf32.hip

// BEGIN verbatim qwen4_gemv_bf16_xf32.hip
template <int R>
static __device__ __forceinline__ void gemv_bf16_xf32_row_multi(
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
// END verbatim qwen4_gemv_bf16_xf32.hip

// BEGIN verbatim qwen4_gemv_bf16_xf32.hip
template <int R>
static __device__ __forceinline__ void gemv_bf16_xf32_k4_rows_body(
    const unsigned short* __restrict__ W,
    const float*          __restrict__ x,
    float*                __restrict__ y,
    int M, int K,
    float scale,
    int hc_act
) {
    __shared__ float quarter[R][4];
    const int row = blockIdx.x;
    if (row >= M) return;
    const int wave = threadIdx.x >> 5;
    const int part = K / 4;
    float s[R];
    gemv_bf16_xf32_row_multi<R>(
        W + (size_t)row * K + wave * part, x + wave * part, K, part, threadIdx.x & 31, s);
    if ((threadIdx.x & 31) == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) quarter[r][wave] = s[r];
    }
    __syncthreads();
    const int r = threadIdx.x;
    if (r >= R) return;
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
// END verbatim qwen4_gemv_bf16_xf32.hip

// BEGIN verbatim qwen4_gemv_bf16_xf32.hip
#define GEMV_BF16_XF32_K4_ROWS_TILED(R)                                             \
    __launch_bounds__(128) extern "C" __global__ void                               \
    gemv_bf16_xf32_k4_rows_tiled_r##R(                                              \
        const unsigned short* __restrict__ W, const float* __restrict__ x,          \
        float* __restrict__ y, int M, int K, float scale, int hc_act) {             \
        gemv_bf16_xf32_k4_rows_body<R>(                                             \
            W, x + (size_t)blockIdx.y * R * K, y + (size_t)blockIdx.y * R * M,      \
            M, K, scale, hc_act);                                                   \
    }
GEMV_BF16_XF32_K4_ROWS_TILED(2)
GEMV_BF16_XF32_K4_ROWS_TILED(3)
GEMV_BF16_XF32_K4_ROWS_TILED(4)
GEMV_BF16_XF32_K4_ROWS_TILED(5)
GEMV_BF16_XF32_K4_ROWS_TILED(6)
GEMV_BF16_XF32_K4_ROWS_TILED(7)
GEMV_BF16_XF32_K4_ROWS_TILED(8)
#undef GEMV_BF16_XF32_K4_ROWS_TILED
// END verbatim qwen4_gemv_bf16_xf32.hip
