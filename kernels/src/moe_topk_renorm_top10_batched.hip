// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched top-10 + renorm given pre-softmaxed probabilities. This is kept as
// a dedicated symbol: Qwen4's fixed top-10 route must not silently enter the
// incumbent k=8 producer.
#define BLOCK_SIZE 256
#define MAX_NEXP   1024
#define K_TOP      10

extern "C" __launch_bounds__(BLOCK_SIZE, 2)
__global__ void moe_topk_renorm_top10_batched(
    const float* __restrict__ probs_in,
    int*         __restrict__ topk_idx,
    float*       __restrict__ topk_w,
    int n_exp,
    int norm_topk
) {
    __shared__ float smem_v[MAX_NEXP];
    __shared__ int   smem_i[MAX_NEXP];
    __shared__ float reduce_v[BLOCK_SIZE];
    __shared__ int   reduce_i[BLOCK_SIZE];
    __shared__ int   picked_idx[K_TOP];
    __shared__ float picked_w[K_TOP];

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const float* __restrict__ row_probs = probs_in + (long long)bid * n_exp;
    int* __restrict__ row_idx = topk_idx + bid * K_TOP;
    float* __restrict__ row_w = topk_w + bid * K_TOP;

    for (int i = tid; i < MAX_NEXP; i += BLOCK_SIZE) {
        smem_v[i] = (i < n_exp) ? row_probs[i] : -INFINITY;
        smem_i[i] = i;
    }
    __syncthreads();

    for (int k = 0; k < K_TOP; ++k) {
        float best_v = -INFINITY;
        int best_i = -1;
        for (int i = tid; i < n_exp; i += BLOCK_SIZE) {
            float value = smem_v[i];
            // The strict comparison intentionally retains the lower-index
            // winner on ties, matching the CPU route reference.
            if (value > best_v) {
                best_v = value;
                best_i = smem_i[i];
            }
        }
        reduce_v[tid] = best_v;
        reduce_i[tid] = best_i;
        __syncthreads();
        for (int stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float lhs = reduce_v[tid];
                float rhs = reduce_v[tid + stride];
                if (rhs > lhs) {
                    reduce_v[tid] = rhs;
                    reduce_i[tid] = reduce_i[tid + stride];
                }
            }
            __syncthreads();
        }
        const int winner = reduce_i[0];
        if (tid == 0) {
            picked_idx[k] = winner;
            picked_w[k] = reduce_v[0];
        }
        if (winner >= 0 && winner < n_exp && ((winner & (BLOCK_SIZE - 1)) == tid)) {
            smem_v[winner] = -INFINITY;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float sum = 0.0f;
        for (int k = 0; k < K_TOP; ++k) sum += picked_w[k];
        if (norm_topk && sum > 0.0f) {
            for (int k = 0; k < K_TOP; ++k) picked_w[k] /= sum;
        }
    }
    __syncthreads();

    if (tid < K_TOP) {
        int safe_idx = picked_idx[tid];
        float safe_w = picked_w[tid];
        if (safe_idx < 0 || safe_idx >= n_exp) {
            safe_idx = 0;
            safe_w = 0.0f;
        }
        row_idx[tid] = safe_idx;
        row_w[tid] = safe_w;
    }
}
