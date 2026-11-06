// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — fused Qwen4 E512/top-10 router + grouped scatter, one launch
// (`HIPFIRE_QWEN4_MOE_ROUTE_SCATTER`, default off; exact gfx1201, 1..=8 rows).
//
// This unit is compiled AFTER moe_router_softmax_top10_f32_fast.hip and
// qwen4_moe_scatter_fused_k8.hip (the module source is the concatenation, see
// `MOE_ROUTE_SCATTER_TOP10_F32_SRC`): it adds no arithmetic of its own.
//
//   phase 1  wave w (w < tokens) runs `router_fast_token` for token w: the very text of
//            moe_router_softmax_top10_f32_fast (selection, expf, left-to-right denominator,
//            fail-closed ids -1 / weight 0 / fault word).  With <= 8 tokens the router is one
//            workgroup, so the whole route fits the one CTA below.
//   barrier  __syncthreads(): every wave's topk_idx stores are visible to the whole CTA.
//   phase 2  `qwen4_moe_scatter_body<scan, ORDERED = true>` over the `tokens * 10` slots: the body
//            of moe_scatter_fused_top10 (single CTA, 256 threads, LDS histogram of 512) with the
//            bucket fill on one wave in ascending slot order.
//
// Outputs are byte for byte those of the two-launch chain (router then scatter): topk_idx,
// topk_w, expert_token_counts, expert_offsets, expert_tile_ids and the router fault word; and
// sorted_slot_index / inverse_perm up to the order of the slots INSIDE one expert's bucket, which
// the chain's atomic fill leaves to wave arrival order (a race: it is not ascending in a large
// share of the cases and not even repeatable). The ORDERED fill pins it to ascending slot order.  Shape generic in rows
// 1..8; the caller declines anything else (> 8 rows needs more than one router workgroup, hence a grid barrier).

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "moe_route_scatter_top10_f32.gfx1201.hip is gfx1201-only"
#endif

__launch_bounds__(256)
extern "C" __global__ void moe_route_scatter_top10_f32(
    const float* __restrict__ logits,  // [tokens, 512]
    int* topk_idx,                     // [tokens, 10]   (router output, scatter input)
    float* __restrict__ topk_w,        // [tokens, 10]
    int normalize_topk_prob,
    int round_logits,
    int tokens,                        // 1..=8
    unsigned long long* __restrict__ fault, // host-mapped router fault word, or null
    int layer,                              // sealed call layer index
    int row_base,                           // global row of this launch's row 0
    int* __restrict__ expert_token_counts,  // [512]
    int* __restrict__ expert_offsets,       // [513]
    int* __restrict__ sorted_slot_index,    // [m_total_max] (-1 padding)
    int* __restrict__ expert_tile_ids,      // [m_total_max / block_m]
    int* __restrict__ inverse_perm,         // [tokens * 10]
    int m_total_max,
    int block_m
) {
    __shared__ float s_exps[ROUTER_TOKENS_PER_BLOCK][512];

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    // Wave-uniform: waves past `tokens` have no row.  No wave leaves before the barrier.
    if (wave < tokens) {
        router_fast_token(logits, topk_idx, topk_w, s_exps[wave], normalize_topk_prob,
                          round_logits, wave, lane, fault, layer, row_base);
    }
    __syncthreads();
    qwen4_moe_scatter_body<QWEN4_SCATTER_DEFAULT_SCAN, true>(
        topk_idx, expert_token_counts, expert_offsets, sorted_slot_index, expert_tile_ids,
        inverse_perm, tokens * 10, 512, m_total_max, block_m);
}
