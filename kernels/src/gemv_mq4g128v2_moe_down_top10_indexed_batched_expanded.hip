// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — QT53 top-10 indexed down projection with expanded output.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// The kernel writes unweighted expanded rows.  moe_down_combine_top10_batched
// is the sole owner of route weighting.  This keeps the ten route products
// deterministic and prevents a second weighting in the grouped path.
//
// Wire rows use exactly 68*ceil(K/128) bytes.  This is deliberately not the
// legacy 72-byte HFQ4G128/Paro row stride.
// The exact K640 module uses this same arithmetic with a constant trip
// count. The generic entry remains the independent dynamic-K launch oracle.
#ifndef QT53_DOWN_ENTRY
#define QT53_DOWN_ENTRY gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded
#endif
__launch_bounds__(32, 8)
extern "C" __global__ void QT53_DOWN_ENTRY(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ rot_batch,
    float* __restrict__ expert_outputs,
    int M,
    int K_arg,
    int N,
    int n_exp
#ifndef QT53_DOWN_K640
    , int k_top
#endif
) {
#ifdef QT53_DOWN_K640
    constexpr int K = 640;
    constexpr int k_top = 10;
#else
    const int K = K_arg;
#endif
    const int row = blockIdx.x;
    const int rank = blockIdx.y;
    const int token = blockIdx.z;
    const int tid = threadIdx.x;
    if (row >= M || k_top <= 0 || rank >= k_top || token >= N || K <= 0) return;

    const int flat = token * k_top + rank;
    const int expert = topk_indices[flat];
    float acc = 0.0f;
    if (expert >= 0 && expert < n_exp) {
        const char* A = reinterpret_cast<const char*>(expert_ptrs[expert]);
        const int groups = (K + 127) / 128;
        const long long row_stride = (long long)groups * 68;
        const char* row_ptr = A + (long long)row * row_stride;
        const float* x = rot_batch + (long long)flat * K;
        // Whole groups (the Qwen4 K = 640 shape): unguarded x loads, so the
        // unrolled groups' loads issue together instead of one guarded load
        // (branch + wait) at a time. Same arithmetic.
        if ((K & 127) == 0) {
#pragma unroll 5
            for (int group = 0; group < groups; ++group) {
                const char* gp = row_ptr + (long long)group * 68;
                const float scale = __half2float(__ushort_as_half(*(const unsigned short*)gp));
                const float zero = __half2float(__ushort_as_half(*(const unsigned short*)(gp + 2)));
                const unsigned char* payload = (const unsigned char*)(gp + 4);
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
        } else for (int group = 0; group < groups; ++group) {
            const char* gp = row_ptr + (long long)group * 68;
            const unsigned short scale_bits = *(const unsigned short*)(gp + 0);
            const unsigned short zero_bits = *(const unsigned short*)(gp + 2);
            const float scale = __half2float(__ushort_as_half(scale_bits));
            const float zero = __half2float(__ushort_as_half(zero_bits));
            const unsigned char* payload = (const unsigned char*)(gp + 4);
            const int base = group * 128 + tid * 4;
            const unsigned char b0 = payload[tid * 2];
            const unsigned char b1 = payload[tid * 2 + 1];
            const int i0 = base;
            const int i1 = base + 1;
            const int i2 = base + 2;
            const int i3 = base + 3;
            const float x0 = i0 < K ? x[i0] : 0.0f;
            const float x1 = i1 < K ? x[i1] : 0.0f;
            const float x2 = i2 < K ? x[i2] : 0.0f;
            const float x3 = i3 < K ? x[i3] : 0.0f;
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

// K = 640 twin of the kernel above with R weight rows per wave: the pick's
// x slice (20 floats per lane) is loaded once and reused for R rows; each
// row keeps the scalar kernel's per-lane expression, group order and
// shuffle reduction, so every expanded row is bitwise the kernel above.
// Grid [ceil(M / R), k_top, N].
template <int R>
static __device__ __forceinline__ void mq4g128v2_down_k640_rows(
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
            const char* row_ptr = A + (long long)row * GROUPS * 68;
#pragma unroll
            for (int g = 0; g < GROUPS; ++g) {
                const char* gp = row_ptr + g * 68;
                const float scale = __half2float(__ushort_as_half(*(const unsigned short*)gp));
                const float zero = __half2float(__ushort_as_half(*(const unsigned short*)(gp + 2)));
                const unsigned char* payload = (const unsigned char*)(gp + 4);
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

#ifndef QT53_DOWN_K640
__launch_bounds__(32, 8)
extern "C" __global__ void gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_k640_r4(
    const unsigned long long* __restrict__ expert_ptrs, const int* __restrict__ topk_indices,
    const float* __restrict__ rot_batch, float* __restrict__ expert_outputs, int M, int K, int N, int n_exp,
    int k_top) {
    mq4g128v2_down_k640_rows<4>(expert_ptrs, topk_indices, rot_batch, expert_outputs, M, K, N, n_exp, k_top);
}
#endif
