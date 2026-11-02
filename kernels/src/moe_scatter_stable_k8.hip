// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Multi-workgroup, deterministic twin of moe_scatter_fused_k8 (one workgroup,
// LDS-atomic placement). Same six outputs, same padding/sentinel contract:
//   expert_token_counts[E]  padded (block_m) counts
//   expert_offsets[E+1]     exclusive prefix of the padded counts, last = M_total
//   sorted_slot_index[m_total_max]  flat slot, -1 for padding and the tail
//   expert_tile_ids[m_total_max / block_m]  expert per block_m tile, -1 tail
//   inverse_perm[total_slots]  flat slot -> sorted position, -1 when unrouted
// An id outside [0, E) is not routed (not counted, inverse_perm = -1).
//
// Order within an expert: ascending flat slot (stable). The incumbent's order
// is whatever the LDS atomics of its eight waves produced (the bucket SET is
// the contract), so any consumer already tolerates every permutation of a
// bucket; this one is merely a fixed member of that family.
//
// Three launches, no global atomics, no pre-clear, no scratch argument:
// the per-chunk expert histograms live in inverse_perm, which is only
// written (to its final values) by the last launch.
//   1. moe_scatter_stable_hist_k8   grid G, block 256. Block g counts the
//      experts of its chunk of ceil(S / G) consecutive slots into
//      inverse_perm[g * E + e] (G * E <= S, enforced by the launcher) and,
//      grid-stride, fills sorted_slot_index and expert_tile_ids with -1.
//   2. moe_scatter_stable_place_k8  grid G, block 256. Every block rebuilds
//      the padded expert offsets from the G * E histogram cells (each block
//      is its own copy; block 0 publishes counts/offsets, block g publishes
//      the tile ids of experts e % G == g), then places its chunk: 256-slot
//      sub-tiles, rank = (earlier same-expert lanes of the wave, by readlane
//      compares) + (same-expert slots of earlier waves, from a per-wave LDS
//      count table filled by LDS atomic increments, so only counts, never
//      order, depend on arrival), cursor[e] += sub-tile count after a barrier.
//   3. moe_scatter_stable_inverse_k8 grid ceil(max(S, m_total_max) / 1024),
//      block 256. sorted_slot_index[p] = s >= 0 -> inverse_perm[s] = p;
//      unrouted slots get -1. Every inverse_perm word is written exactly once.
//
// Any wave size (warpSize lanes per wave; the rank uses readlane, the rest goes
// through LDS behind barriers). LDS: 41 KiB static (E <= 1024).

#define SC_THREADS 256
#define SC_MAX_E   1024

extern "C" __global__ __launch_bounds__(SC_THREADS) void moe_scatter_stable_hist_k8(
    const int* __restrict__ topk_indices,   // [total_slots]
    int*       __restrict__ part,           // [G * E] histogram cells (inverse_perm used as scratch)
    int*       __restrict__ sorted_slot_index, // [m_total_max]
    int*       __restrict__ expert_tile_ids,   // [m_total_max / block_m]
    int        total_slots,
    int        num_experts,
    int        m_total_max,
    int        block_m
) {
    __shared__ int s_hist[SC_MAX_E];
    const int tid = threadIdx.x;
    const int g = blockIdx.x;
    const int gtid = g * SC_THREADS + tid;
    const int stride = gridDim.x * SC_THREADS;

    for (int e = tid; e < num_experts; e += SC_THREADS) s_hist[e] = 0;
    for (int i = gtid; i < m_total_max; i += stride) sorted_slot_index[i] = -1;
    const int num_tiles_max = m_total_max / block_m;
    for (int i = gtid; i < num_tiles_max; i += stride) expert_tile_ids[i] = -1;
    __syncthreads();

    const int chunk = (total_slots + gridDim.x - 1) / gridDim.x;
    const int lo = g * chunk;
    const int hi = min(total_slots, lo + chunk);
    for (int i = lo + tid; i < hi; i += SC_THREADS) {
        const int e = topk_indices[i];
        if (e >= 0 && e < num_experts) atomicAdd(&s_hist[e], 1);
    }
    __syncthreads();
    for (int e = tid; e < num_experts; e += SC_THREADS) part[g * num_experts + e] = s_hist[e];
}

extern "C" __global__ __launch_bounds__(SC_THREADS) void moe_scatter_stable_place_k8(
    const int* __restrict__ topk_indices,       // [total_slots]
    const int* __restrict__ part,               // [G * E] from the hist launch
    int*       __restrict__ expert_token_counts,// [E]
    int*       __restrict__ expert_offsets,     // [E+1]
    int*       __restrict__ sorted_slot_index,  // [m_total_max] (pre-filled with -1)
    int*       __restrict__ expert_tile_ids,    // [m_total_max / block_m] (pre-filled with -1)
    int        total_slots,
    int        num_experts,
    int        block_m
) {
    __shared__ int s_cursor[SC_MAX_E];  // next sorted position per expert (this chunk)
    __shared__ int s_pad[SC_MAX_E];     // padded count, then exclusive offset
    __shared__ int s_wc[SC_THREADS / 32][SC_MAX_E]; // per-wave, per-expert sub-tile counts
    __shared__ int s_scan[SC_THREADS];
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int wave = tid / warpSize;
    const int nwaves = SC_THREADS / warpSize;
    const int g = blockIdx.x;
    const int G = gridDim.x;

    // Offsets: thread t owns the consecutive experts [t * per, (t + 1) * per).
    const int per = (num_experts + SC_THREADS - 1) / SC_THREADS;
    const int e0 = min(num_experts, tid * per);
    const int e1 = min(num_experts, e0 + per);
    int local_total = 0;
    for (int e = e0; e < e1; e++) {
        int before = 0;
        int total = 0;
        #pragma unroll 8
        for (int c = 0; c < G; c++) {
            const int v = part[c * num_experts + e];
            before += (c < g) ? v : 0;
            total += v;
        }
        const int padded = ((total + block_m - 1) / block_m) * block_m;
        s_cursor[e] = before;
        s_pad[e] = padded;
        local_total += padded;
    }
    s_scan[tid] = local_total;
    __syncthreads();
    for (int d = 1; d < SC_THREADS; d <<= 1) {
        int v = s_scan[tid];
        if (tid >= d) v += s_scan[tid - d];
        __syncthreads();
        s_scan[tid] = v;
        __syncthreads();
    }
    int run = s_scan[tid] - local_total;
    for (int e = e0; e < e1; e++) {
        const int padded = s_pad[e];
        s_cursor[e] += run;
        if (g == 0) {
            expert_token_counts[e] = padded;
            expert_offsets[e] = run;
        }
        if (e % G == g) {
            const int tile_end = (run + padded) / block_m;
            for (int t = run / block_m; t < tile_end; t++) expert_tile_ids[t] = e;
        }
        run += padded;
    }
    if (g == 0 && tid == 0) expert_offsets[num_experts] = s_scan[SC_THREADS - 1];
    __syncthreads();

    // Placement of this block's chunk, 256 slots at a time.
    const int chunk = (total_slots + G - 1) / G;
    const int lo = g * chunk;
    const int hi = min(total_slots, lo + chunk);
    for (int base = lo; base < hi; base += SC_THREADS) {
        const int i = base + tid;
        int my = -1;
        if (i < hi) {
            const int e = topk_indices[i];
            if (e >= 0 && e < num_experts) my = e;
        }
        for (int w = 0; w < nwaves; w++) {
            for (int e = tid; e < num_experts; e += SC_THREADS) s_wc[w][e] = 0;
        }
        __syncthreads();
        // Earlier same-expert lanes of this wave.
        int rank = 0;
        for (int j = 0; j < warpSize; j++) {
            const int other = __builtin_amdgcn_readlane(my, j);
            rank += (j < lane && other == my && my >= 0) ? 1 : 0;
        }
        if (my >= 0) atomicAdd(&s_wc[wave][my], 1);
        __syncthreads();
        if (my >= 0) {
            for (int w = 0; w < wave; w++) rank += s_wc[w][my];
            sorted_slot_index[s_cursor[my] + rank] = i;
        }
        __syncthreads();
        for (int e = tid; e < num_experts; e += SC_THREADS) {
            int add = 0;
            for (int w = 0; w < nwaves; w++) add += s_wc[w][e];
            s_cursor[e] += add;
        }
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(SC_THREADS) void moe_scatter_stable_inverse_k8(
    const int* __restrict__ topk_indices,       // [total_slots]
    const int* __restrict__ sorted_slot_index,  // [m_total_max]
    int*       __restrict__ inverse_perm,       // [total_slots]
    int        total_slots,
    int        num_experts,
    int        m_total_max
) {
    const int gtid = blockIdx.x * SC_THREADS + threadIdx.x;
    const int stride = gridDim.x * SC_THREADS;
    for (int p = gtid; p < m_total_max; p += stride) {
        const int s = sorted_slot_index[p];
        if (s >= 0) inverse_perm[s] = p;
    }
    for (int i = gtid; i < total_slots; i += stride) {
        const int e = topk_indices[i];
        if (!(e >= 0 && e < num_experts)) inverse_perm[i] = -1;
    }
}
