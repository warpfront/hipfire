// Qwen4 copy of moe_scatter_fused_k8.hip as #774 left it (block-wide scan on gfx1151). The
// top-10 alias compiles this on every arch and the shipped k8 module on gfx1151 only;
// moe_scatter_fused_k8.hip keeps the shipped text for every other arch.

#include <hip/hip_runtime.h>

// Fused SGLang-style MoE scatter pipeline: histogram + padded prefix-sum
// + permutation in a single launch. Saves ~2 kernel launches per MoE
// layer (40 layers × ~50µs = 2ms per prefill on R9700) and a couple of
// __syncthreads-only cycles that the split kernels paid.
//
// LDS layout (interleaved across phases):
//   [0 .. num_experts)              — raw histogram (Phase 1)
//                                  — padded count + bucket-fill (Phase 3)
//
// Writes:
//   expert_token_counts[E]   — final padded counts
//   expert_offsets[E+1]      — exclusive prefix-sum, last cell = M_total
//   sorted_slot_index[m_total_max] — flat slot or -1 for padding (only
//                                    [0..m_total] are written; caller is
//                                    responsible for sizing m_total_max
//                                    appropriately or memset'ing first)
//   expert_tile_ids[m_total/block_m]
//   inverse_perm[total_slots] — flat → sorted_pos
//
// Single workgroup, 256 threads. LDS = num_experts * 4 bytes.

extern "C" __global__ void moe_scatter_fused_k8(
    const int* __restrict__ topk_indices,        // [total_slots]
    int*       __restrict__ expert_token_counts, // [E]
    int*       __restrict__ expert_offsets,      // [E+1]
    int*       __restrict__ sorted_slot_index,   // [m_total_max] (-1 padding)
    int*       __restrict__ expert_tile_ids,     // [m_total_max / block_m]
    int*       __restrict__ inverse_perm,        // [total_slots]
    int        total_slots,
    int        num_experts,
    int        m_total_max,                      // upper bound for -1 init
    int        block_m
) {
    extern __shared__ int s_workspace[];
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    // ── Phase 1: histogram (LDS atomicAdd) ──
    for (int i = tid; i < num_experts; i += bdim) {
        s_workspace[i] = 0;
    }
    // Init sorted_slot_index to -1 across the upper bound so padding
    // slots within each expert range carry the sentinel even when
    // m_total < m_total_max. The grouped-GEMM tile reader uses -1 for
    // the X-gather sentinel.
    for (int i = tid; i < m_total_max; i += bdim) {
        sorted_slot_index[i] = -1;
    }
    // Init expert_tile_ids upper bound to -1 sentinel. The grouped-GEMM
    // dispatcher launches with slot_tiles = m_total_max / block_m to
    // avoid the m_total dtoh sync; tiles beyond the live m_total / block_m
    // have expert_id == -1 → kernel early-returns. Saves ~50µs sync per
    // MoE layer.
    const int num_tiles_max = m_total_max / block_m;
    for (int i = tid; i < num_tiles_max; i += bdim) {
        expert_tile_ids[i] = -1;
    }
    __syncthreads();

    for (int i = tid; i < total_slots; i += bdim) {
        const int e = topk_indices[i];
        if (e >= 0 && e < num_experts) {
            atomicAdd(&s_workspace[e], 1);
        }
    }
    __syncthreads();

    // ── Phase 2: pad + exclusive prefix-sum ──
#if defined(__gfx1151__)
    // gfx1151: a block-wide scan over contiguous per-thread expert ranges:
    // integer sums, so the offsets equal the serial walk's exactly (512-expert
    // Qwen4 routing). Other archs keep the serial walk below unchanged.
    __shared__ int wave_totals[32];
    const int per = (num_experts + bdim - 1) / bdim;
    const int first = tid * per;
    const int last = min(first + per, num_experts);
    int local = 0;
    for (int e = first; e < last; ++e) {
        local += ((s_workspace[e] + block_m - 1) / block_m) * block_m;
    }
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int waves = (bdim + 31) >> 5;
    int inclusive = local;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int up = __shfl_up(inclusive, offset, 32);
        if (lane >= offset) inclusive += up;
    }
    if (lane == 31 || tid == bdim - 1) wave_totals[wave] = inclusive;
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < waves; ++w) wave_totals[w] += wave_totals[w - 1];
        expert_offsets[num_experts] = wave_totals[waves - 1]; // M_total
    }
    __syncthreads();
    int acc = inclusive - local + (wave > 0 ? wave_totals[wave - 1] : 0);
    for (int e = first; e < last; ++e) {
        const int padded = ((s_workspace[e] + block_m - 1) / block_m) * block_m;
        expert_offsets[e] = acc;
        expert_token_counts[e] = padded;
        acc += padded;
    }
#else
    // Serial in thread 0 — E ≤ 256.
    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < num_experts; i++) {
            const int raw = s_workspace[i];
            const int padded = ((raw + block_m - 1) / block_m) * block_m;
            expert_offsets[i] = acc;
            expert_token_counts[i] = padded;
            acc += padded;
        }
        expert_offsets[num_experts] = acc; // M_total
    }
#endif
    __syncthreads();

    // ── Phase 3a: reset s_workspace for bucket-fill counters ──
    for (int i = tid; i < num_experts; i += bdim) {
        s_workspace[i] = 0;
    }
    __syncthreads();

    // ── Phase 3b: scatter ──
    for (int i = tid; i < total_slots; i += bdim) {
        const int e = topk_indices[i];
        if (e >= 0 && e < num_experts) {
            const int pos_in_bucket = atomicAdd(&s_workspace[e], 1);
            const int base = expert_offsets[e];
            const int sorted_pos = base + pos_in_bucket;
            sorted_slot_index[sorted_pos] = i;
            inverse_perm[i] = sorted_pos;
        }
    }
    __syncthreads();

    // ── Phase 3c: tile_ids — walk by expert ──
    for (int e = tid; e < num_experts; e += bdim) {
        const int start_slot = expert_offsets[e];
        const int end_slot   = expert_offsets[e + 1];
        const int tile_start = start_slot / block_m;
        const int tile_end   = end_slot / block_m;
        for (int t = tile_start; t < tile_end; t++) {
            expert_tile_ids[t] = e;
        }
    }
}
