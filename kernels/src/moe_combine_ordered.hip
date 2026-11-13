// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — owner-masked fixed top-10 BF16 grouped-down combine with the rank
// order resolved inline (HIPFIRE_QWEN4_MOE_COMBINE_INLINE_ORDER).
//
// A module of its own (as moe_combine_order_owned / slowlink_prefill), so the legacy
// moe_down_combine_grouped_top10 and moe_combine_order_owned sources keep their recorded
// identity. The two BF16 helpers and the per-column accumulation are verbatim copies of
// moe_down_combine_grouped_top10.hip's `moe_down_combine_grouped_bf16in`.
//
// Replaces the two launches `moe_combine_order_top10_owned` (one thread per token, the ten
// route ranks selected by ascending expert id into an int2 (grouped row or -1, BF16-rounded
// weight bits) order) -> `moe_down_combine_grouped_top10_bf16in[_zinit]` (reads that order).
// The order buffer has no reader on the wide-verify route (the next consumer of its scratch,
// the shared down, overwrites it), so it is not written.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ __hip_bfloat16 moe_top10_bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

static __device__ __forceinline__ float moe_top10_bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

// Grid [ceil(hidden / 512), tokens], block 64, hidden % 8 == 0.
//
// The order, per token, in the first ten lanes of every wave: lane r holds route rank r's
// expert (a key of INT_MAX when the id is outside [0, n_experts): unrouted, never selected,
// its slot stays (-1, +0.0)), its grouped row (-1 when out of range or when `owner[expert]`
// is another rank: the foreign row keeps its place and BF16 weight but is read as a +0 row,
// exactly moe_combine_order_top10_owned) and its BF16-rounded weight. moe_combine_order_top10's
// selection loop takes, ten times, the unused rank with the smallest expert id, the lowest
// rank on a tie; that rank's slot is therefore the number of ranks with a smaller
// (expert, rank) key, computed here with a shuffle round instead of the serial selection.
// Slot c is then fetched from the lane whose slot is c; a slot with no rank left is
// (-1, +0.0): adds a zero and leaves the sum unchanged.
//
// `initial_zero`: as moe_down_combine_grouped_top10_bf16in_zinit (the accumulator starts as a
// register +0.0, the destination is write-only).
template <bool initial_zero>
static __device__ __forceinline__ void moe_down_combine_grouped_bf16in_ordered(
    const unsigned short* __restrict__ grouped_down, const int* __restrict__ inverse_perm,
    const int* __restrict__ topk_indices, const float* __restrict__ topk_weights,
    float* __restrict__ residual, const int* __restrict__ owner, int hidden, int grouped_rows,
    int tokens, int n_experts, int rank) {
    const int token = blockIdx.y;
    if (token >= tokens) return;
    const size_t route_base = (size_t)token * 10;
    const int lane = (int)__lane_id();

    int key = 0x7fffffff;
    int row = -1;
    float weight = 0.0f;
    if (lane < 10) {
        const int expert = topk_indices[route_base + lane];
        if (expert >= 0 && expert < n_experts) {
            key = expert;
            const int grouped_row = inverse_perm[route_base + lane];
            weight = moe_top10_bf16_to_f32(
                moe_top10_bf16_round(topk_weights[route_base + lane]));
            const bool live = grouped_row >= 0 && grouped_row < grouped_rows &&
                              owner[expert] == rank;
            row = live ? grouped_row : -1;
        }
    }
    int pos = 0;
#pragma unroll
    for (int j = 0; j < 10; ++j) {
        const int key_j = __shfl(key, j);
        pos += (key_j < key || (key_j == key && j < lane)) ? 1 : 0;
    }
    const bool selectable = lane < 10 && key != 0x7fffffff;

    int slot_row[10];
    float wts[10];
#pragma unroll
    for (int c = 0; c < 10; ++c) {
        const unsigned long long who = __ballot(selectable && pos == c);
        const int src = who ? (int)__ffsll((long long)who) - 1 : 0;
        const int r = __shfl(row, src);
        const float w = __shfl(weight, src);
        slot_row[c] = who ? r : -1;
        wts[c] = who ? w : 0.0f;
    }

    const int col0 = (blockIdx.x * 64 + threadIdx.x) * 8;
    if (col0 >= hidden) return;
    uint4 rows[10];
#pragma unroll
    for (int c = 0; c < 10; ++c) {
        rows[c] = slot_row[c] >= 0
                      ? *(const uint4*)(grouped_down + (size_t)slot_row[c] * hidden + col0)
                      : make_uint4(0, 0, 0, 0);
    }
    float* res = residual + (size_t)token * hidden + col0;
    float acc[8];
    if (initial_zero) {
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
    } else {
        const float4 r0 = *(const float4*)res, r1 = *(const float4*)(res + 4);
        acc[0] = r0.x; acc[1] = r0.y; acc[2] = r0.z; acc[3] = r0.w;
        acc[4] = r1.x; acc[5] = r1.y; acc[6] = r1.z; acc[7] = r1.w;
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        __hip_bfloat16 accumulated = moe_top10_bf16_round(acc[e]);
#pragma unroll
        for (int c = 0; c < 10; ++c) {
            const unsigned int w = (&rows[c].x)[e >> 1];
            const float raw_output = __uint_as_float((e & 1) ? (w & 0xFFFF0000u) : (w << 16));
            const __hip_bfloat16 output = moe_top10_bf16_round(raw_output);
            const __hip_bfloat16 weighted = moe_top10_bf16_round(
                moe_top10_bf16_to_f32(output) * wts[c]);
            accumulated = moe_top10_bf16_round(
                moe_top10_bf16_to_f32(accumulated) + moe_top10_bf16_to_f32(weighted));
        }
        acc[e] = moe_top10_bf16_to_f32(accumulated);
    }
    *(float4*)res = make_float4(acc[0], acc[1], acc[2], acc[3]);
    *(float4*)(res + 4) = make_float4(acc[4], acc[5], acc[6], acc[7]);
}

extern "C" __global__ __launch_bounds__(64) void moe_down_combine_grouped_top10_bf16in_ordered(
    const unsigned short* __restrict__ grouped_down, const int* __restrict__ inverse_perm,
    const int* __restrict__ topk_indices, const float* __restrict__ topk_weights,
    float* __restrict__ residual, const int* __restrict__ owner, int hidden, int grouped_rows,
    int tokens, int n_experts, int rank) {
    moe_down_combine_grouped_bf16in_ordered<false>(grouped_down, inverse_perm, topk_indices,
                                                   topk_weights, residual, owner, hidden,
                                                   grouped_rows, tokens, n_experts, rank);
}

// Same arguments and launch as the symbol above; `residual` is write-only.
extern "C" __global__ __launch_bounds__(64) void
moe_down_combine_grouped_top10_bf16in_ordered_zinit(
    const unsigned short* __restrict__ grouped_down, const int* __restrict__ inverse_perm,
    const int* __restrict__ topk_indices, const float* __restrict__ topk_weights,
    float* __restrict__ residual, const int* __restrict__ owner, int hidden, int grouped_rows,
    int tokens, int n_experts, int rank) {
    moe_down_combine_grouped_bf16in_ordered<true>(grouped_down, inverse_perm, topk_indices,
                                                  topk_weights, residual, owner, hidden,
                                                  grouped_rows, tokens, n_experts, rank);
}
