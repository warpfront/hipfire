// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64

#include <hip/hip_runtime.h>

// gfx12 wave64 router: one wave owns one token, keeps the router row in
// registers, performs zero-LDS softmax, and selects the eight winners with a
// packed full-precision key.  This is intentionally a separate translation
// unit: per-function wave-size attributes miscompile HIP lane/shuffle builtins
// with ROCm 7.2, while -mwavefrontsize64 makes the whole TU coherent.

#ifndef ROUTER_PER_LANE
#define ROUTER_PER_LANE 4
#endif
#ifndef ROUTER_MIN_BLOCKS
#define ROUTER_MIN_BLOCKS 8
#endif
#ifndef ROUTER_KERNEL_NAME
#define ROUTER_KERNEL_NAME moe_router_softmax_topk_k8_wave64
#endif
#define ROUTER_K 8

__device__ __forceinline__ unsigned router_sortable_u32(float f) {
    unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ unsigned long long router_allmax_u64(unsigned long long x) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        unsigned lo = (unsigned)x;
        unsigned hi = (unsigned)(x >> 32);
        unsigned olo = __shfl_xor(lo, off);
        unsigned ohi = __shfl_xor(hi, off);
        unsigned long long y = ((unsigned long long)ohi << 32) | olo;
        x = y > x ? y : x;
    }
    return x;
}

__device__ __forceinline__ float router_allmax_f32(float x) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1)
        x = fmaxf(x, __shfl_xor(x, off));
    return x;
}

__device__ __forceinline__ float router_alladd_f32(float x) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1)
        x += __shfl_xor(x, off);
    return x;
}

extern "C" __launch_bounds__(64, ROUTER_MIN_BLOCKS)
__global__ void ROUTER_KERNEL_NAME(
    const float* __restrict__ logits,
    int* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int n_exp,
    int norm_topk)
{
    const int lane = threadIdx.x;
    float prob[ROUTER_PER_LANE];
    int ids[ROUTER_PER_LANE];
    float row_max = -INFINITY;

    #pragma unroll
    for (int i = 0; i < ROUTER_PER_LANE; ++i) {
        int e = lane + i * 64;
        float v = e < n_exp ? logits[e] : -INFINITY;
        ids[i] = e;
        prob[i] = v;
        row_max = fmaxf(row_max, v);
    }
    row_max = router_allmax_f32(row_max);

    float local_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < ROUTER_PER_LANE; ++i) {
        float e = ids[i] < n_exp ? expf(prob[i] - row_max) : 0.0f;
        local_sum += e;
        prob[i] = ids[i] < n_exp ? e : -INFINITY;
    }
    float inv_sum = 1.0f / router_alladd_f32(local_sum);
    #pragma unroll
    for (int i = 0; i < ROUTER_PER_LANE; ++i)
        if (ids[i] < n_exp) prob[i] *= inv_sum;

    float picked[ROUTER_K];
    #pragma unroll 1
    for (int r = 0; r < ROUTER_K; ++r) {
        float best_v = -INFINITY;
        int best_i = 0;
        #pragma unroll
        for (int i = 0; i < ROUTER_PER_LANE; ++i) {
            if (prob[i] > best_v) {
                best_v = prob[i];
                best_i = ids[i];
            }
        }
        unsigned long long key =
            ((unsigned long long)router_sortable_u32(best_v) << 32) |
            (unsigned long long)(255u - (unsigned)(best_i & 255));
        unsigned long long winner = router_allmax_u64(key);
        int win_i = 255 - (int)(winner & 255ull);
        float candidate = -INFINITY;
        #pragma unroll
        for (int i = 0; i < ROUTER_PER_LANE; ++i)
            if (ids[i] == win_i) candidate = prob[i];
        float win_v = router_allmax_f32(candidate);
        if (lane == 0) topk_idx[r] = win_i;
        picked[r] = win_v;
        #pragma unroll
        for (int i = 0; i < ROUTER_PER_LANE; ++i)
            if (ids[i] == win_i) prob[i] = -INFINITY;
    }

    if (lane == 0) {
        float s = 0.0f;
        #pragma unroll
        for (int r = 0; r < ROUTER_K; ++r) s += picked[r];
        #pragma unroll
        for (int r = 0; r < ROUTER_K; ++r) {
            int idx = topk_idx[r];
            float w = norm_topk && s > 0.0f ? picked[r] / s : picked[r];
            if (idx < 0 || idx >= n_exp) { idx = 0; w = 0.0f; }
            topk_idx[r] = idx;
            topk_w[r] = w;
        }
    }
}
