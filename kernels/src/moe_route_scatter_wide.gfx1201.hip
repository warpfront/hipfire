// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — wave-3 C8 MoE glue folds (`HIPFIRE_QWEN4_MOE_GLUE_FOLD`, default off; exact gfx1201):
//
//   moe_scatter_fused_top10_pmask      X8a  scatter + owner tile mask, one 256-thread CTA
//   moe_route_scatter_top10_wide_f32   X8b  router + scatter (+ mask), 9..=64 rows, ceil(rows / 8)
//                                           workgroups of 256 threads; the last one to finish its
//                                           router rows runs the scatter
//
// This unit is compiled AFTER moe_router_softmax_top10_f32_fast.hip and
// qwen4_moe_scatter_fused_k8.hip (the module source is the concatenation, see
// `MOE_ROUTE_SCATTER_WIDE_SRC`): it adds no arithmetic of its own.
//
//   router   workgroup b, wave w routes token 8 b + w with `router_fast_token` (the very text of
//            moe_router_softmax_top10_f32_fast, the launch shape of that kernel), so topk_idx /
//            topk_w / the fault word are the multi-workgroup router's.
//   arrival  every thread fences its own stores, a workgroup barrier, thread 0 fences again and
//            bumps the device counter `sync[0]` (zero between launches); the workgroup that sees
//            the count reach gridDim.x - 1 is the LAST one and resets the counter; the others exit.
//            Same discipline as hyper_norm_gate_k4_outputs_f32's row counters.
//   scatter  the last workgroup, after an acquire fence, runs `qwen4_moe_scatter_body<
//            QWEN4_SCATTER_DEFAULT_SCAN>` UNCHANGED over the `rows * 10` slots (the atomic bucket
//            fill: the order of slots inside one expert's bucket is the same arrival race the
//            shipped scatter has; the tables are equal as sets).
//   mask     (optional) `tile_ids_owned[t] = owner_table[id] == rank && 0 <= id < n_experts ? id : -1`,
//            the law of `ep_tile_mask_pm_gfx1201`, over the scatter's final `expert_tile_ids`
//            behind one more barrier.  A null `owner_table` writes nothing.
//
// Outputs are byte for byte those of the unfused chain (router, scatter, ep_tile_mask): topk_idx,
// topk_w, expert_token_counts, expert_offsets, expert_tile_ids, tile_ids_owned and the router fault
// word; sorted_slot_index / inverse_perm up to the order inside one expert's bucket.  Not for
// recording / graph capture (the counter is one persistent buffer per Gpu; the host guards it).

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "moe_route_scatter_wide.gfx1201.hip is gfx1201-only"
#endif

// The tile mask of `ep_tile_mask_pm_gfx1201` over `tiles` tiles, the CTA's threads striding.
__device__ __forceinline__ void qwen4_moe_tile_mask_pass(
    const int* __restrict__ tile_ids,
    int* __restrict__ tile_ids_owned,
    const int* __restrict__ owner_table,
    int tiles,
    int n_experts,
    int rank
) {
    for (int t = threadIdx.x; t < tiles; t += blockDim.x) {
        const int id = tile_ids[t];
        const bool take = (unsigned)id < (unsigned)n_experts && owner_table[id] == rank;
        tile_ids_owned[t] = take ? id : -1;
    }
}

// X8a: the shipped scatter launch (same ABI prefix as moe_scatter_fused_top10) followed by the mask.
// Single CTA of 256 threads, dynamic LDS = num_experts * 4.
__launch_bounds__(256)
extern "C" __global__ void moe_scatter_fused_top10_pmask(
    const int* __restrict__ topk_indices,        // [total_slots]
    int*       __restrict__ expert_token_counts, // [E]
    int*       __restrict__ expert_offsets,      // [E+1]
    int*       __restrict__ sorted_slot_index,   // [m_total_max] (-1 padding)
    int*       __restrict__ expert_tile_ids,     // [m_total_max / block_m]
    int*       __restrict__ inverse_perm,        // [total_slots]
    int        total_slots,
    int        num_experts,
    int        m_total_max,
    int        block_m,
    int*       __restrict__ tile_ids_owned,      // [tiles] (never aliases expert_tile_ids)
    const int* __restrict__ owner_table,         // [num_experts]
    int        tiles,
    int        rank
) {
    qwen4_moe_scatter_body<QWEN4_SCATTER_DEFAULT_SCAN>(
        topk_indices, expert_token_counts, expert_offsets, sorted_slot_index, expert_tile_ids,
        inverse_perm, total_slots, num_experts, m_total_max, block_m);
    __syncthreads();
    qwen4_moe_tile_mask_pass(expert_tile_ids, tile_ids_owned, owner_table, tiles, num_experts, rank);
}

// X8b: router (grid = ceil(tokens / 8), 8 tokens per workgroup) + last-arriving-workgroup scatter
// (+ mask).  E512 / top-10.  Dynamic LDS = 512 * 4 bytes (the scatter histogram); the router rows
// are static.
__launch_bounds__(256)
extern "C" __global__ void moe_route_scatter_top10_wide_f32(
    const float* __restrict__ logits,  // [tokens, 512]
    int* topk_idx,                     // [tokens, 10]   (router output, scatter input)
    float* __restrict__ topk_w,        // [tokens, 10]
    int normalize_topk_prob,
    int round_logits,
    int tokens,                        // 1..=64
    unsigned long long* __restrict__ fault, // host-mapped router fault word, or null
    int layer,                              // sealed call layer index
    int row_base,                           // global row of this launch's row 0
    int* __restrict__ expert_token_counts,  // [512]
    int* __restrict__ expert_offsets,       // [513]
    int* __restrict__ sorted_slot_index,    // [m_total_max] (-1 padding)
    int* __restrict__ expert_tile_ids,      // [m_total_max / block_m]
    int* __restrict__ inverse_perm,         // [tokens * 10]
    int m_total_max,
    int block_m,
    int* __restrict__ tile_ids_owned,       // [tiles] or null
    const int* __restrict__ owner_table,    // [512] or null (no mask)
    int tiles,
    int rank,
    unsigned int* sync                      // [1] arrival counter, zero between launches
) {
    __shared__ float s_exps[ROUTER_TOKENS_PER_BLOCK][512];
    __shared__ int s_last;

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int token = blockIdx.x * ROUTER_TOKENS_PER_BLOCK + wave;
    if (token < tokens) {
        router_fast_token(logits, topk_idx, topk_w, s_exps[wave], normalize_topk_prob,
                          round_logits, token, lane, fault, layer, row_base);
    }
    // Publish this workgroup's router rows, then count the arrival.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned int arrived = atomicAdd(sync, 1u);
        const bool last = arrived == gridDim.x - 1u;
        if (last) atomicExch(sync, 0u);
        s_last = last;
    }
    __syncthreads();
    if (!s_last) return;
    // Acquire every other workgroup's stores before the scatter reads topk_idx.
    __threadfence();
    qwen4_moe_scatter_body<QWEN4_SCATTER_DEFAULT_SCAN>(
        topk_idx, expert_token_counts, expert_offsets, sorted_slot_index, expert_tile_ids,
        inverse_perm, tokens * 10, 512, m_total_max, block_m);
    if (owner_table != nullptr) {
        __syncthreads();
        qwen4_moe_tile_mask_pass(expert_tile_ids, tile_ids_owned, owner_table, tiles, 512, rank);
    }
}
