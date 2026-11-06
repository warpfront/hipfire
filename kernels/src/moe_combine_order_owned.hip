// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — owner-masked rank order of the fixed top-10 BF16 grouped-down combine.
//
// A module of its own (as slowlink_prefill), so the legacy
// moe_down_combine_grouped_top10 source keeps its recorded identity. The two
// BF16 helpers are verbatim copies of that source's.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ __hip_bfloat16 moe_top10_bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

static __device__ __forceinline__ float moe_top10_bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

// moe_combine_order_top10 (moe_down_combine_grouped_top10.hip) under the EP
// partial tile mask: the ten route ranks in ascending expert ID as (grouped
// row or -1, BF16-rounded weight bits), where a rank whose expert another rank
// owns (`owner[expert] != rank`) keeps its place and its BF16 weight but gets
// row -1. Its grouped rows were never written (the masked GEMMs skip its
// tiles), and every consumer of the order (the BF16-row combine, the HC row
// folds) substitutes a +0 BF16 row for -1, the bits the cleared buffer held,
// so the multiply, add and rounding of that zero term still run in the
// ten-term order. `owner`: an i32 owner rank per expert (the tile mask's
// table). Grid [ceil(tokens / 64)], block 64.
extern "C" __global__ __launch_bounds__(64) void moe_combine_order_top10_owned(
    const int* __restrict__ inverse_perm, const int* __restrict__ topk_indices,
    const float* __restrict__ topk_weights, int2* __restrict__ order, int grouped_rows,
    int tokens, int n_experts, const int* __restrict__ owner, int rank) {
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= tokens) return;
    const size_t route_base = (size_t)token * 10;
    int expert[10];
    int used_ranks = 0;
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        expert[r] = topk_indices[route_base + r];
        // Id outside [0, n_experts): unrouted, never selected (order slot stays (-1, 0)).
        if (expert[r] < 0 || expert[r] >= n_experts) used_ranks |= 1 << r;
    }
#pragma unroll
    for (int c = 0; c < 10; ++c) {
        int selected_rank = -1, selected_expert = 0x7fffffff;
#pragma unroll
        for (int r = 0; r < 10; ++r) {
            if (!(used_ranks & (1 << r)) && expert[r] < selected_expert) {
                selected_expert = expert[r];
                selected_rank = r;
            }
        }
        if (selected_rank < 0) {  // no rank left: adds a zero (the sum is unchanged)
            order[route_base + c] = make_int2(-1, 0);
            continue;
        }
        used_ranks |= 1 << selected_rank;
        const int grouped_row = inverse_perm[route_base + selected_rank];
        const float weight =
            moe_top10_bf16_to_f32(moe_top10_bf16_round(topk_weights[route_base + selected_rank]));
        const bool live = grouped_row >= 0 && grouped_row < grouped_rows &&
                          owner[selected_expert] == rank;
        order[route_base + c] = make_int2(live ? grouped_row : -1, __float_as_int(weight));
    }
}
