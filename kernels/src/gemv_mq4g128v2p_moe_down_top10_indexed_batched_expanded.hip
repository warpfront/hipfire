// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — QT53 top-10 indexed down projection with expanded output, drop-zero planar
// twin (DESIGN F1, MQ4G128V2P, qt=138) of
// gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded.hip.
//
// Expert tensor [M, K] (K % 128 == 0), planar:
//   [codes plane: M*K/2 bytes][scales plane: M*(K/128) LE f16]
//   row r codes  : base + r*(K/2)            (64 nibble bytes per 128-column group)
//   row r scales : base + M*(K/2) + r*(K/128)*2   (one f16 per group; only 2-byte aligned)
// zp = -8.0f * sc (exactly the shipped f16 zero widened, see the gate/up twin). Everything
// else is the shipped kernel line for line, so every expanded row is bit-identical.
// Whole groups only (the Qwen4 K = 640 shape); any other K returns without writing.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef QT53P_DOWN_ENTRY
#define QT53P_DOWN_ENTRY gemv_mq4g128v2p_moe_down_top10_indexed_batched_expanded
#endif
__launch_bounds__(32, 8)
extern "C" __global__ void QT53P_DOWN_ENTRY(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ rot_batch,
    float* __restrict__ expert_outputs,
    int M,
    int K,
    int N,
    int n_exp,
    int k_top
) {
    const int row = blockIdx.x;
    const int rank = blockIdx.y;
    const int token = blockIdx.z;
    const int tid = threadIdx.x;
    if (row >= M || k_top <= 0 || rank >= k_top || token >= N || K <= 0 || (K & 127) != 0) return;

    const int flat = token * k_top + rank;
    const int expert = topk_indices[flat];
    float acc = 0.0f;
    if (expert >= 0 && expert < n_exp) {
        const char* A = reinterpret_cast<const char*>(expert_ptrs[expert]);
        const int groups = K >> 7;
        const unsigned char* codes_row = (const unsigned char*)(A + (long long)row * (K >> 1));
        const unsigned short* scales_row =
            (const unsigned short*)(A + (long long)M * (K >> 1) + (long long)row * groups * 2);
        const float* x = rot_batch + (long long)flat * K;
#pragma unroll 5
        for (int group = 0; group < groups; ++group) {
            const float scale = __half2float(__ushort_as_half(scales_row[group]));
            const float zero = -8.0f * scale;
            const unsigned char* payload = codes_row + (long long)group * 64;
            const unsigned char b0 = payload[tid * 2];
            const unsigned char b1 = payload[tid * 2 + 1];
            const float* xg = x + group * 128 + tid * 4;
            const float x0 = xg[0];
            const float x1 = xg[1];
            const float x2 = xg[2];
            const float x3 = xg[3];
            acc += (scale * (float)(b0 & 0xF) + zero) * x0
                + (scale * (float)(b0 >> 4) + zero) * x1
                + (scale * (float)(b1 & 0xF) + zero) * x2
                + (scale * (float)(b1 >> 4) + zero) * x3;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if (tid == 0)
        expert_outputs[(long long)flat * M + row] = acc;
}

// K = 640 twin with R weight rows per wave (x loaded once, reused for R rows); every row keeps
// the scalar kernel's per-lane expression, group order and shuffle reduction.
// Grid [ceil(M / R), k_top, N].
template <int R>
static __device__ __forceinline__ void mq4g128v2p_down_k640_rows(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ rot_batch,
    float* __restrict__ expert_outputs,
    int M, int K, int N, int n_exp, int k_top
) {
    constexpr int GROUPS = 5;
    const int row0 = blockIdx.x * R;
    const int rank = blockIdx.y;
    const int token = blockIdx.z;
    const int tid = threadIdx.x;
    if (row0 >= M || k_top <= 0 || rank >= k_top || token >= N || K != GROUPS * 128) return;
    const int flat = token * k_top + rank;
    const int expert = topk_indices[flat];
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    if (expert >= 0 && expert < n_exp) {
        const char* A = reinterpret_cast<const char*>(expert_ptrs[expert]);
        const float* x = rot_batch + (long long)flat * K + tid * 4;
        float xs[GROUPS][4];
#pragma unroll
        for (int g = 0; g < GROUPS; ++g)
#pragma unroll
            for (int i = 0; i < 4; ++i) xs[g][i] = x[g * 128 + i];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r < M ? row0 + r : M - 1;
            const unsigned char* codes_row = (const unsigned char*)(A + (long long)row * (GROUPS * 64));
            const unsigned short* scales_row =
                (const unsigned short*)(A + (long long)M * (GROUPS * 64) + (long long)row * GROUPS * 2);
#pragma unroll
            for (int g = 0; g < GROUPS; ++g) {
                const float scale = __half2float(__ushort_as_half(scales_row[g]));
                const float zero = -8.0f * scale;
                const unsigned char* payload = codes_row + g * 64;
                const unsigned char b0 = payload[tid * 2];
                const unsigned char b1 = payload[tid * 2 + 1];
                acc[r] += (scale * (float)(b0 & 0xF) + zero) * xs[g][0]
                    + (scale * (float)(b0 >> 4) + zero) * xs[g][1]
                    + (scale * (float)(b1 & 0xF) + zero) * xs[g][2]
                    + (scale * (float)(b1 >> 4) + zero) * xs[g][3];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float sum = acc[r];
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down(sum, offset);
        if (tid == 0 && row0 + r < M)
            expert_outputs[(long long)flat * M + row0 + r] = sum;
    }
}

__launch_bounds__(32, 8)
extern "C" __global__ void gemv_mq4g128v2p_moe_down_top10_indexed_batched_expanded_k640_r4(
    const unsigned long long* __restrict__ expert_ptrs, const int* __restrict__ topk_indices,
    const float* __restrict__ rot_batch, float* __restrict__ expert_outputs, int M, int K, int N, int n_exp,
    int k_top) {
    mq4g128v2p_down_k640_rows<4>(expert_ptrs, topk_indices, rot_batch, expert_outputs, M, K, N, n_exp, k_top);
}
