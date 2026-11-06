// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — fixed top-10 BF16 expanded-down combine, fixed reduction partition.
//
// Identical to `moe_down_combine_top10_batched` (same ascending-expert-id
// ten-term fold, same BF16 rounding of output / weight / product / each
// accumulation, same invalid-id skip) with ONE numerical difference: a live
// slot whose expert `e % ranks != rank` contributes output +0.0f instead of
// its expanded row.  The slot is still selected in its ascending-expert
// position and its weight keeps its place; the BF16 multiply/add/round of the
// zero term still runs in order.
//
// The partition is the FIXED reduction owner `e % ranks`, independent of which
// rank computed the expanded row: whatever a peer or this rank wrote into a
// foreign slot's row is never read.  `expert_outputs` is never written (peers
// may be reading it concurrently).
//
// Kept as a module of its own (precedent: moe_combine_order_owned.hip) so
// `moe_down_combine_top10_batched.hip` keeps its recorded identity; the two
// BF16 helpers are verbatim copies.

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
// the next slot is accumulated.
__launch_bounds__(256, 4)
extern "C" __global__ void moe_down_combine_top10_batched_partitioned(
    const float* __restrict__ expert_outputs, // [N * 10 * H]
    const int* __restrict__ topk_indices,     // [N * 10]
    const float* __restrict__ topk_weights,    // [N * 10]
    float* __restrict__ residual,              // [N * H], in-place +=
    int hidden,
    int tokens,
    int n_experts,
    int rank,
    int ranks
) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    if (column >= hidden || token >= tokens) return;
    const size_t token_slot = (size_t)token * 10;
    int experts[10];
    float weights[10], outputs[10];
    // A slot whose id is outside [0, n_experts) (the router's -1 marker for a
    // non-finite row) is unrouted: it is never selected, so neither its weight
    // nor its (possibly uninitialised) output row contributes.
    int used_slots = 0;
#pragma unroll
    for (int slot = 0; slot < 10; ++slot) {
        experts[slot] = topk_indices[token_slot + slot];
        const bool live = experts[slot] >= 0 && experts[slot] < n_experts;
        weights[slot] = topk_weights[token_slot + slot];
        outputs[slot] = (live && (experts[slot] % ranks) == rank)
            ? expert_outputs[(token_slot + slot) * hidden + column]
            : 0.0f;
        if (!live) used_slots |= 1 << slot;
    }
    __hip_bfloat16 accumulated = moe_top10_bf16_round(
        residual[(size_t)token * hidden + column]);
    for (int count = 0; count < 10; ++count) {
        int selected_slot = -1;
        int selected_expert = 0x7fffffff;
        float selected_weight = 0.0f, selected_output = 0.0f;
#pragma unroll
        for (int slot = 0; slot < 10; ++slot) {
            if ((used_slots & (1 << slot)) == 0 && experts[slot] < selected_expert) {
                selected_expert = experts[slot];
                selected_slot = slot;
                selected_weight = weights[slot];
                selected_output = outputs[slot];
            }
        }
        if (selected_slot < 0) break;
        used_slots |= 1 << selected_slot;
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
