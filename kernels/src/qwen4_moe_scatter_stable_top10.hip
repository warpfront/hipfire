// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Deterministic grouping for the opt-in Qwen4 symmetric IU4 MoE route
// (fn-moe-sym): slots ordered by (expert_id, flat_slot), each expert run
// padded to 16, no atomics. Same six outputs as moe_scatter_fused_top10
// (counts, pad16 offsets[E+1], sorted_slot_index with -1 padding/tail,
// expert_tile_ids with -1 tail, inverse_perm), which stays the default route's
// scatter.
//   1. qwen4_moe_group_ranks   grid C = ceil(L/256), block 256: per chunk
//      local rank of every slot among same-expert slots of the chunk, and the
//      chunk's per-expert count (published by the first occurrence; the
//      block clears its own partial column first).
//   2. qwen4_moe_group_prefix  grid 1, block E (<= 1024): per-expert chunk
//      prefix, total count (stored in chunk_prefix column C, so chunk_prefix
//      is [E, C+1]), expert_token_counts, pad16 exclusive offsets[E+1].
//   3. qwen4_moe_group_scatter grid ceil(max(L,Pmax)/256), block 256: every
//      valid slot writes sorted/inverse once; every padding/tail position
//      writes -1; tile ids written per 16-position tile. Each output word is
//      written exactly once per invocation, so replay needs no pre-clear.
// Complexity O(256 L + E C).

#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__) && !defined(__gfx1201__)
#error "qwen4_moe_scatter_stable_top10.hip is gfx1151/gfx1201-only"
#endif

extern "C" __global__ __launch_bounds__(256) void qwen4_moe_group_ranks(
    const int* __restrict__ topk, int* __restrict__ rank, int* __restrict__ partial,
    int L, int E, int C) {
    __shared__ int s_e[256];
    const int c = blockIdx.x, t = threadIdx.x;
    const int flat = c * 256 + t;
    for (int e = t; e < E; e += 256) partial[(size_t)e * C + c] = 0;
    const int n = min(256, L - c * 256);
    const int my = t < n ? topk[flat] : -1;
    s_e[t] = my;
    __syncthreads();
    if (t < n) {
        int r = 0;
        for (int j = 0; j < t; ++j) r += s_e[j] == my;
        rank[flat] = r;
        if (r == 0) {
            int cnt = 1;
            for (int j = t + 1; j < n; ++j) cnt += s_e[j] == my;
            partial[(size_t)my * C + c] = cnt;
        }
    }
}

extern "C" __global__ __launch_bounds__(1024) void qwen4_moe_group_prefix(
    const int* __restrict__ partial, int* __restrict__ chunk_prefix,
    int* __restrict__ counts, int* __restrict__ offsets, int E, int C) {
    __shared__ int s_pad[1024];
    const int e = threadIdx.x;
    int total = 0;
    if (e < E) {
        for (int c = 0; c < C; ++c) {
            chunk_prefix[(size_t)e * (C + 1) + c] = total;
            total += partial[(size_t)e * C + c];
        }
        chunk_prefix[(size_t)e * (C + 1) + C] = total;
        // Padded count, as moe_scatter_fused_top10 stores it.
        counts[e] = (total + 15) / 16 * 16;
    }
    s_pad[e] = e < E ? (total + 15) / 16 * 16 : 0;
    __syncthreads();
    // Inclusive Hillis-Steele scan over blockDim entries.
    for (int off = 1; off < (int)blockDim.x; off <<= 1) {
        const int v = e >= off ? s_pad[e - off] : 0;
        __syncthreads();
        s_pad[e] += v;
        __syncthreads();
    }
    if (e < E) offsets[e + 1] = s_pad[e];
    if (e == 0) offsets[0] = 0;
}

extern "C" __global__ __launch_bounds__(256) void qwen4_moe_group_scatter(
    const int* __restrict__ topk, const int* __restrict__ rank,
    const int* __restrict__ chunk_prefix, const int* __restrict__ offsets,
    int* __restrict__ sorted, int* __restrict__ tiles, int* __restrict__ inverse,
    int L, int E, int Pmax) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    const int C = (L + 255) / 256;
    if (i < L) {
        const int e = topk[i];
        const int pos = offsets[e] + chunk_prefix[(size_t)e * (C + 1) + i / 256] + rank[i];
        sorted[pos] = i;
        inverse[i] = pos;
    }
    if (i < Pmax) {
        const int used = offsets[E];
        int e = -1;
        if (i < used) {
            int lo = 0, hi = E;  // offsets[lo] <= i < offsets[hi]
            while (hi - lo > 1) {
                const int mid = (lo + hi) >> 1;
                if (offsets[mid] <= i) lo = mid; else hi = mid;
            }
            e = lo;
            const int local = i - offsets[e];
            if (local >= chunk_prefix[(size_t)e * (C + 1) + C]) sorted[i] = -1;
        } else {
            sorted[i] = -1;
        }
        if ((i & 15) == 0) tiles[i >> 4] = e;
    }
}
