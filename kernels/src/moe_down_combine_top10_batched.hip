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
    int tokens
) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    if (column >= hidden || token >= tokens) return;
    const size_t token_slot = (size_t)token * 10;
    __hip_bfloat16 accumulated = moe_top10_bf16_round(
        residual[(size_t)token * hidden + column]);
    int used_ranks = 0;
    for (int count = 0; count < 10; ++count) {
        int selected_rank = -1;
        int selected_expert = 0x7fffffff;
        for (int rank = 0; rank < 10; ++rank) {
            if ((used_ranks & (1 << rank)) != 0) continue;
            const int expert = topk_indices[token_slot + rank];
            if (expert < selected_expert) {
                selected_expert = expert;
                selected_rank = rank;
            }
        }
        if (selected_rank < 0) break;
        used_ranks |= 1 << selected_rank;
        const __hip_bfloat16 weight =
            moe_top10_bf16_round(topk_weights[token_slot + selected_rank]);
        const __hip_bfloat16 output = moe_top10_bf16_round(
            expert_outputs[(token_slot + selected_rank) * hidden + column]);
        const __hip_bfloat16 weighted = moe_top10_bf16_round(
            moe_top10_bf16_to_f32(output) * moe_top10_bf16_to_f32(weight));
        accumulated = moe_top10_bf16_round(
            moe_top10_bf16_to_f32(accumulated) + moe_top10_bf16_to_f32(weighted));
    }
    residual[(size_t)token * hidden + column] =
        moe_top10_bf16_to_f32(accumulated);
}
