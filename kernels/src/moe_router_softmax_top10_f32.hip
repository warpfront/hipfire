// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — fixed E512/top-10 softmax router.

#include <hip/hip_runtime.h>
#include <math.h>

// One 256-thread workgroup handles one token.  Each wave selects its best ten
// from 64 logits, then wave 0 selects the global best ten from those 80
// candidates.  This keeps stable ordering while avoiding ten serial 512-entry
// scans and the barrier-heavy full sort.
//
// topk_idx is an i32 bit-pattern stored in the project's F32-typed scratch
// convention.  `normalize_topk_prob != 0` applies the Qwen4 selected-probability
// renormalization after the float32 softmax.
__device__ __forceinline__ bool router_better(
    float lhs_value,
    int lhs_index,
    float rhs_value,
    int rhs_index
) {
    return lhs_value > rhs_value ||
        (lhs_value == rhs_value && lhs_index < rhs_index);
}

__launch_bounds__(256, 1)
extern "C" __global__ void moe_router_softmax_top10_f32(
    const float* __restrict__ logits, // [tokens, n_exp]
    int* __restrict__ topk_idx,       // [tokens, 10]
    float* __restrict__ topk_w,       // [tokens, 10]
    int n_exp,
    int top_k,
    int normalize_topk_prob
) {
    __shared__ float wave_values[80];
    __shared__ int wave_indices[80];
    __shared__ float selected_values[10];
    __shared__ float probabilities[10];
    const int thread = threadIdx.x;
    const int lane = thread & 31;
    const int wave = thread >> 5;
    const int token = blockIdx.x;

    // The sealed E512/top-10 route checks this geometry before launch. Keep the guard uniform
    // across the workgroup so no lane can return across the block barrier.
    if (n_exp != 512 || top_k != 10) return;

    const size_t token_base = (size_t)token * 512;
    const float first_logit = logits[token_base + thread];
    const float second_logit = logits[token_base + thread + 256];
    float first_value = first_logit;
    float second_value = second_logit;
    const int first_index = thread;
    const int second_index = thread + 256;

    // The global top ten must be among each wave's local top ten.  Select those
    // with register state and wave shuffles; no block barrier is needed here.
    for (int rank = 0; rank < 10; ++rank) {
        float best_value = first_value;
        int best_index = first_index;
        if (router_better(second_value, second_index, best_value, best_index)) {
            best_value = second_value;
            best_index = second_index;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_value = __shfl_down(best_value, offset, 32);
            const int other_index = __shfl_down(best_index, offset, 32);
            if (router_better(other_value, other_index, best_value, best_index)) {
                best_value = other_value;
                best_index = other_index;
            }
        }
        best_value = __shfl(best_value, 0, 32);
        best_index = __shfl(best_index, 0, 32);
        if (lane == 0) {
            wave_values[wave * 10 + rank] = best_value;
            wave_indices[wave * 10 + rank] = best_index;
        }
        if (first_index == best_index) first_value = -INFINITY;
        if (second_index == best_index) second_value = -INFINITY;
    }
    __syncthreads();

    // Wave 0 merges the 80 wave-local candidates.  Each lane owns at most
    // three values and participates in the same stable wave reduction.
    if (wave == 0) {
        const int slot0 = lane;
        const int slot1 = lane + 32;
        const int slot2 = lane + 64;
        float value0 = wave_values[slot0];
        float value1 = wave_values[slot1];
        float value2 = slot2 < 80 ? wave_values[slot2] : -INFINITY;
        const int index0 = wave_indices[slot0];
        const int index1 = wave_indices[slot1];
        const int index2 = slot2 < 80 ? wave_indices[slot2] : 0x7fffffff;

        for (int rank = 0; rank < top_k; ++rank) {
            float best_value = value0;
            int best_index = index0;
            if (router_better(value1, index1, best_value, best_index)) {
                best_value = value1;
                best_index = index1;
            }
            if (router_better(value2, index2, best_value, best_index)) {
                best_value = value2;
                best_index = index2;
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other_value = __shfl_down(best_value, offset, 32);
                const int other_index = __shfl_down(best_index, offset, 32);
                if (router_better(other_value, other_index, best_value, best_index)) {
                    best_value = other_value;
                    best_index = other_index;
                }
            }
            best_value = __shfl(best_value, 0, 32);
            best_index = __shfl(best_index, 0, 32);
            if (lane == 0) {
                topk_idx[(size_t)token * 10 + rank] = best_index;
                selected_values[rank] = best_value;
            }
            if (index0 == best_index) value0 = -INFINITY;
            if (index1 == best_index) value1 = -INFINITY;
            if (index2 == best_index) value2 = -INFINITY;
        }
    }

    // Preserve the reference's float32 operation order: full softmax in
    // expert-index order, followed by selected-probability renormalization.
    // The 512 exponentials are computed in parallel into LDS; one serial pass
    // then sums them in expert order, keeping routed-output numerics.
    __shared__ float exps[512];
    __syncthreads();
    {
        const float max_logit = selected_values[0];
        exps[thread] = expf(first_logit - max_logit);
        exps[thread + 256] = expf(second_logit - max_logit);
    }
    __syncthreads();
    if (thread == 0) {
        const size_t output_base = (size_t)token * 10;
        const float max_logit = selected_values[0];
        float denom = 0.0f;
        for (int expert = 0; expert < 512; ++expert)
            denom += exps[expert];
        if (!(denom > 0.0f) || !isfinite(denom)) return;

        const float inv_denom = 1.0f / denom;
        for (int rank = 0; rank < top_k; ++rank)
            probabilities[rank] = expf(selected_values[rank] - max_logit) * inv_denom;

        if (normalize_topk_prob != 0) {
            float selected_sum = 0.0f;
            for (int rank = 0; rank < top_k; ++rank)
                selected_sum += probabilities[rank];
            if (!(selected_sum > 0.0f) || !isfinite(selected_sum)) return;
            const float inv_selected = 1.0f / selected_sum;
            for (int rank = 0; rank < top_k; ++rank)
                probabilities[rank] *= inv_selected;
        }

        for (int rank = 0; rank < top_k; ++rank)
            topk_w[output_base + rank] = probabilities[rank];
    }
}
