// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — fixed top-10 BF16 grouped-down combine.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ __hip_bfloat16 moe_top10_bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

static __device__ __forceinline__ float moe_top10_bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

// `inverse_perm[token * 10 + rank]` maps a route slot to the sorted grouped
// row.  The grouped path intentionally mirrors
// moe_down_combine_top10_batched: route ranks are selected by ascending expert
// ID, and residual, weight, output, product, and every accumulation are each
// rounded through the source BF16 contract before the next rank.
static __device__ __forceinline__ float moe_down_widen(float v) { return v; }
static __device__ __forceinline__ float moe_down_widen(unsigned short b) {
    return __uint_as_float((unsigned int)b << 16);
}

template <typename T>
__device__ __forceinline__ void moe_down_combine_grouped(
    const T* __restrict__ grouped_down,     // [grouped_rows * H], unweighted
    const int* __restrict__ inverse_perm,    // [N * 10], flat -> grouped row
    const int* __restrict__ topk_indices,     // [N * 10], expert IDs
    const float* __restrict__ topk_weights,   // [N * 10]
    float* __restrict__ residual,             // [N * H], in-place +=
    int hidden,
    int grouped_rows,
    int tokens
) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    if (token >= tokens || column >= hidden) return;

    const size_t route_base = (size_t)token * 10;
    __hip_bfloat16 accumulated = moe_top10_bf16_round(
        residual[(size_t)token * hidden + column]);
    int used_ranks = 0;
    for (int count = 0; count < 10; ++count) {
        int selected_rank = -1;
        int selected_expert = 0x7fffffff;
        for (int rank = 0; rank < 10; ++rank) {
            if ((used_ranks & (1 << rank)) != 0) continue;
            const int expert = topk_indices[route_base + rank];
            if (expert < selected_expert) {
                selected_expert = expert;
                selected_rank = rank;
            }
        }
        if (selected_rank < 0) break;
        used_ranks |= 1 << selected_rank;

        const __hip_bfloat16 weight =
            moe_top10_bf16_round(topk_weights[route_base + selected_rank]);
        const int grouped_row = inverse_perm[route_base + selected_rank];
        const float raw_output =
            (grouped_row >= 0 && grouped_row < grouped_rows)
                ? moe_down_widen(grouped_down[(size_t)grouped_row * hidden + column])
                : 0.0f;
        const __hip_bfloat16 output = moe_top10_bf16_round(raw_output);
        const __hip_bfloat16 weighted = moe_top10_bf16_round(
            moe_top10_bf16_to_f32(output) * moe_top10_bf16_to_f32(weight));
        accumulated = moe_top10_bf16_round(
            moe_top10_bf16_to_f32(accumulated) + moe_top10_bf16_to_f32(weighted));
    }
    residual[(size_t)token * hidden + column] =
        moe_top10_bf16_to_f32(accumulated);
}

__launch_bounds__(256, 4)
extern "C" __global__ void moe_down_combine_grouped_top10(
    const float* __restrict__ grouped_down, const int* __restrict__ inverse_perm,
    const int* __restrict__ topk_indices, const float* __restrict__ topk_weights,
    float* __restrict__ residual, int hidden, int grouped_rows, int tokens) {
    moe_down_combine_grouped(grouped_down, inverse_perm, topk_indices, topk_weights,
                             residual, hidden, grouped_rows, tokens);
}

// The BF16-input combine's rank order, once per token: the ten route ranks
// in ascending expert ID (moe_down_combine_grouped's selection) as (grouped
// row or -1, BF16-rounded weight bits).  Grid [ceil(tokens / 64)], block 64.
extern "C" __global__ __launch_bounds__(64) void moe_combine_order_top10(
    const int* __restrict__ inverse_perm, const int* __restrict__ topk_indices,
    const float* __restrict__ topk_weights, int2* __restrict__ order, int grouped_rows,
    int tokens) {
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= tokens) return;
    const size_t route_base = (size_t)token * 10;
    int expert[10];
#pragma unroll
    for (int rank = 0; rank < 10; ++rank) expert[rank] = topk_indices[route_base + rank];
    int used_ranks = 0;
#pragma unroll
    for (int c = 0; c < 10; ++c) {
        int selected_rank = -1, selected_expert = 0x7fffffff;
#pragma unroll
        for (int rank = 0; rank < 10; ++rank) {
            if (!(used_ranks & (1 << rank)) && expert[rank] < selected_expert) {
                selected_expert = expert[rank];
                selected_rank = rank;
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
        order[route_base + c] = make_int2(
            (grouped_row >= 0 && grouped_row < grouped_rows) ? grouped_row : -1,
            __float_as_int(weight));
    }
}

// Grouped down stored as BF16 bits (the values the BF16 round above yields),
// eight columns per thread, in moe_combine_order_top10's order: all ten
// ranks' 16-byte slices are loaded before the in-order accumulation, and each
// column runs the exact moe_down_combine_grouped arithmetic.
// Grid [ceil(hidden / 512), tokens], block 64, hidden % 8 == 0.
//
// `initial_zero`: the destination was just cleared to +0.0 by a ClearOp and
// carries no residual, so the accumulator starts as a register +0.0 and the
// destination is write-only (no residual load; the clear is then redundant).
// +0.0 is exactly what the cleared buffer would have read back, and
// bf16_round(+0.0) = +0.0, so every round that follows sees identical bits
// and the stores are bytewise those of clear + the `false` instantiation.
// Only the `_zinit` symbol instantiates it; the original symbol is `false`.
template <bool initial_zero>
static __device__ __forceinline__ void moe_down_combine_grouped_bf16in(
    const unsigned short* __restrict__ grouped_down, const int2* __restrict__ order,
    float* __restrict__ residual, int hidden, int tokens) {
    const int token = blockIdx.y;
    const int col0 = (blockIdx.x * 64 + threadIdx.x) * 8;
    if (token >= tokens || col0 >= hidden) return;
    const size_t route_base = (size_t)token * 10;
    uint4 rows[10];
    float wts[10];
#pragma unroll
    for (int c = 0; c < 10; ++c) {
        const int2 o = order[route_base + c];
        wts[c] = __int_as_float(o.y);
        rows[c] = o.x >= 0 ? *(const uint4*)(grouped_down + (size_t)o.x * hidden + col0)
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

extern "C" __global__ __launch_bounds__(64) void moe_down_combine_grouped_top10_bf16in(
    const unsigned short* __restrict__ grouped_down, const int2* __restrict__ order,
    float* __restrict__ residual, int hidden, int tokens) {
    moe_down_combine_grouped_bf16in<false>(grouped_down, order, residual, hidden, tokens);
}

// Same arguments and launch as the symbol above; `residual` is write-only.
extern "C" __global__ __launch_bounds__(64) void moe_down_combine_grouped_top10_bf16in_zinit(
    const unsigned short* __restrict__ grouped_down, const int2* __restrict__ order,
    float* __restrict__ residual, int hidden, int tokens) {
    moe_down_combine_grouped_bf16in<true>(grouped_down, order, residual, hidden, tokens);
}
