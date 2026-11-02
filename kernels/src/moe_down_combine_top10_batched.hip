// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — fixed top-10 BF16 expanded-down combine.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ __hip_bfloat16 moe_top10_bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

static __device__ __forceinline__ float moe_top10_bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

// `expert_outputs` is deliberately unweighted [N * 10 * H].  Each route
// product and each index-add step is rounded to the source BF16 dtype before
// the next rank is accumulated.
__launch_bounds__(256, 4)
extern "C" __global__ void moe_down_combine_top10_batched(
    const float* __restrict__ expert_outputs, // [N * 10 * H]
    const int* __restrict__ topk_indices,     // [N * 10]
    const float* __restrict__ topk_weights,    // [N * 10]
    float* __restrict__ residual,              // [N * H], in-place +=
    int hidden,
    int tokens,
    int n_experts
) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    if (column >= hidden || token >= tokens) return;
    const size_t token_slot = (size_t)token * 10;
    // Every operand loads up front (one memory round trip); the ascending-
    // expert order is then chosen in registers, accumulating as before.
    int experts[10];
    float weights[10], outputs[10];
    // A rank whose id is outside [0, n_experts) (the router's -1 marker for a
    // non-finite row) is unrouted: it is never selected, so neither its weight
    // nor its (possibly uninitialised) output row contributes.
    int used_ranks = 0;
#pragma unroll
    for (int rank = 0; rank < 10; ++rank) {
        experts[rank] = topk_indices[token_slot + rank];
        const bool live = experts[rank] >= 0 && experts[rank] < n_experts;
        weights[rank] = topk_weights[token_slot + rank];
        outputs[rank] = live ? expert_outputs[(token_slot + rank) * hidden + column] : 0.0f;
        if (!live) used_ranks |= 1 << rank;
    }
    __hip_bfloat16 accumulated = moe_top10_bf16_round(
        residual[(size_t)token * hidden + column]);
    for (int count = 0; count < 10; ++count) {
        int selected_rank = -1;
        int selected_expert = 0x7fffffff;
        float selected_weight = 0.0f, selected_output = 0.0f;
#pragma unroll
        for (int rank = 0; rank < 10; ++rank) {
            if ((used_ranks & (1 << rank)) == 0 && experts[rank] < selected_expert) {
                selected_expert = experts[rank];
                selected_rank = rank;
                selected_weight = weights[rank];
                selected_output = outputs[rank];
            }
        }
        if (selected_rank < 0) break;
        used_ranks |= 1 << selected_rank;
        const __hip_bfloat16 weight = moe_top10_bf16_round(selected_weight);
        const __hip_bfloat16 output = moe_top10_bf16_round(selected_output);
        const __hip_bfloat16 weighted = moe_top10_bf16_round(
            moe_top10_bf16_to_f32(output) * moe_top10_bf16_to_f32(weight));
        accumulated = moe_top10_bf16_round(
            moe_top10_bf16_to_f32(accumulated) + moe_top10_bf16_to_f32(weighted));
    }
    residual[(size_t)token * hidden + column] =
        moe_top10_bf16_to_f32(accumulated);
}
