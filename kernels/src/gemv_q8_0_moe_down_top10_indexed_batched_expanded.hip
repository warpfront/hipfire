// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — Qwen4 Q8F16/qt3 top-10 indexed down projection.

#include <hip/hip_runtime.h>
#include <math.h>

// Q8F16 uses the project's Q8_0 wire representation: one FP16 scale followed
// by 32 signed int8 values (34 bytes per 32-element block).  The output is
// intentionally unweighted [N * 10 * M]; weighted routing belongs solely to
// the top-10 combine kernel.
__launch_bounds__(32, 8)
extern "C" __global__ void gemv_q8_0_moe_down_top10_indexed_batched_expanded(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,                // [N * 10]
    const float* __restrict__ hidden_batch,              // [N * 10 * K]
    float* __restrict__ expert_outputs,                  // [N * 10 * M]
    int M,
    int K,
    int N,
    int n_exp
) {
    const int row = blockIdx.x;
    const int rank = blockIdx.y;
    const int token = blockIdx.z;
    const int lane = threadIdx.x;
    if (row >= M || rank >= 10 || token >= N || lane >= 32) return;

    const int flat = token * 10 + rank;
    const int expert = topk_indices[flat];
    float sum = 0.0f;
    if (expert >= 0 && expert < n_exp && K > 0 && (K & 31) == 0) {
        const unsigned char* __restrict__ weights =
            reinterpret_cast<const unsigned char*>(expert_ptrs[expert]);
        const int blocks = K >> 5;
        const size_t row_base = (size_t)row * blocks * 34;
        const float* x = hidden_batch + (size_t)flat * K;
        for (int k = lane; k < K; k += 32) {
            const int block = k >> 5;
            const unsigned char* qblock = weights + row_base + (size_t)block * 34;
            const _Float16 scale_half = *reinterpret_cast<const _Float16*>(qblock);
            const float scale = (float)scale_half;
            const signed char q = reinterpret_cast<const signed char*>(qblock + 2)[k & 31];
            sum += scale * (float)q * x[k];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (lane == 0) {
        expert_outputs[(size_t)flat * M + row] = sum;
    }
}
