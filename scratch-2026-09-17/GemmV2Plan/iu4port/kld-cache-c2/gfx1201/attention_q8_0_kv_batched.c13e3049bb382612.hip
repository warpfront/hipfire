// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Multi-slot KV descriptor — the single point of KV address translation for
// every batched attention kernel.
//
// A "slot" is one independent sequence in a batch. Today each slot owns a
// contiguous slab of `cap` tokens inside a per-layer arena, so translation is
// base + pos*stride. When this moves to paged block tables, ONLY
// kv_offset_for() changes — no kernel is touched. That is the entire reason
// this indirection exists; do not inline it away.
//
// Layout must stay byte-identical to the Rust mirror in
// crates/rdna-compute/src/kv_slots.rs. 24 bytes, 8-byte aligned.

#pragma once

#include <hip/hip_runtime.h>

struct KvSlotDesc {
    unsigned long long k_base;  // byte offset of this slot's K slab in the arena
    unsigned long long v_base;  // byte offset of this slot's V slab in the arena
    int seq_len;                // logical KV length; kernel reads [0, seq_len)
    int cap;                    // physical slab capacity in tokens; seq_len <= cap
};

// Byte offset of position `pos` within slot `s`'s K slab.
// `per_pos_bytes` is the per-position stride in bytes, uniform across slots
// (n_kv_heads * (head_dim/32) * 34 for Q8_0).
__device__ __forceinline__ unsigned long long kv_offset_for_k(
    const KvSlotDesc& s, int pos, int per_pos_bytes)
{
    return s.k_base + (unsigned long long)pos * (unsigned long long)per_pos_bytes;
}

__device__ __forceinline__ unsigned long long kv_offset_for_v(
    const KvSlotDesc& s, int pos, int per_pos_bytes)
{
    return s.v_base + (unsigned long long)pos * (unsigned long long)per_pos_bytes;
}

// Single-slot fallback used when the descriptor pointer is null. Keeps the
// ported kernels on ONE code path: callers build this on the stack from the
// legacy scalar args, so there is no `if (descs) ... else ...` around every
// KV read. Behaviour is then bitwise identical to the pre-SP1 kernel.
__device__ __forceinline__ KvSlotDesc kv_slot_legacy(int seq_len, int max_seq)
{
    KvSlotDesc s;
    s.k_base = 0ULL;
    s.v_base = 0ULL;
    s.seq_len = seq_len;
    s.cap = max_seq;
    return s;
}

// Legacy fallback for kernels that also honour the pre-descriptor
// independent-sequence contract: a NEGATIVE `max_seq` whose magnitude is one
// lane's token capacity, with batch row `row` reading only its own
// `[row * cap, (row + 1) * cap)` slice of a lane-major arena. Folding that
// base into the synthesised descriptor keeps those kernels on the single
// kv_offset_for_*() address path instead of branching at every KV read.
// A positive `max_seq` reproduces `kv_slot_legacy` exactly (base 0, shared
// cache), so sequential prefill stays byte-for-byte unchanged.
__device__ __forceinline__ KvSlotDesc kv_slot_legacy_lane(
    int seq_len, int max_seq, int row, int per_pos_bytes)
{
    const bool independent = max_seq < 0;
    const int lane_capacity = independent ? -max_seq : max_seq;
    const unsigned long long lane_bytes =
        (unsigned long long)lane_capacity * (unsigned long long)per_pos_bytes;
    const unsigned long long base =
        independent ? (unsigned long long)row * lane_bytes : 0ULL;

    KvSlotDesc s;
    s.k_base = base;
    s.v_base = base;
    s.seq_len = seq_len;
    s.cap = lane_capacity;
    return s;
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <assert.h>


// Batched causal attention with Q8_0 quantized KV cache. Counterpart of
// attention_q8_0_kv for prefill batching — processes N queries in one launch,
// each with its own causal window read from positions[] array.
//
// Grid: [n_heads, batch_size, 1]
// Block processes (head h, batch row b). The query comes from
// q[b * n_heads * head_dim + h * head_dim]; the causal window extends
// from position 0 through positions[b] inclusive. Output is written to
// out[b * n_heads * head_dim + h * head_dim].
//
// Shared memory layout (host must pass `(max_seq + nthreads + head_dim)*4`
// as the dynamic LDS size, where max_seq is the MAX positions[b]+1 in
// this launch):
//   scores[seq_len] + workspace[nthreads] + q_shared[head_dim]
//
// Byte-exact with attention_q8_0_kv at batch_size=1: same math, same
// reduction order, same shared-memory layout (the workspace and q_shared
// pointers are derived relative to seq_len so different batch rows with
// different seq_len each have their own valid slice within the per-block
// allocation).
//
// ─── Tree-attention mode (DDTree batched verify) ──────────────────────────
// When `tree_bias != nullptr` the kernel runs in tree mode:
//   - seq_len = block_start + block_cols (full cache up to and including
//     the tree block, ignoring `positions[b]` for the causal cutoff).
//   - For keys in [block_start, block_start + block_cols), the score is
//     augmented by `tree_bias[b * block_cols + (t - block_start)]`. The
//     caller uses `+0.0` on ancestor-or-self positions and `-inf` on
//     non-ancestors; this gates within-block attention to the tree's
//     visibility matrix without changing the prompt path.
//   - Keys in [0, block_start) are unmasked (prompt is always visible).
//   - Rope-encoded K entries in the block may have LOGICAL positions that
//     differ from their cache slot (siblings share a depth). That's
//     handled host-side at KV-write time — this kernel just does dot
//     products on whatever is in the cache.
// When `tree_bias == nullptr` the kernel falls back to standard causal
// (byte-identical to the pre-tree path).
extern "C" __global__ void attention_q8_0_kv_batched(
    const float* __restrict__ q,            // [batch_size × n_heads × head_dim]
    const unsigned char* __restrict__ k_cache,  // [max_seq × n_kv_heads × head_dim Q8_0 blocks]
    const unsigned char* __restrict__ v_cache,  // same layout
    float* __restrict__ out,                // [batch_size × n_heads × head_dim]
    const int* __restrict__ positions,      // [batch_size]
    const float* __restrict__ tree_bias,    // optional [batch_size × block_cols] or nullptr
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int block_start,                        // ignored when tree_bias == nullptr
    int block_cols,                         // ignored when tree_bias == nullptr
    const KvSlotDesc* __restrict__ slot_descs,  // [n_slots] or nullptr = legacy
    const int* __restrict__ row_slot            // [batch_size] or nullptr = legacy
) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    if (h >= n_heads) return;

    const bool tree_mode = (tree_bias != nullptr);
    // hipGraph-safe block_start: under captured tree-verify replay the
    // `block_start` scalar kernarg is baked at capture time and goes stale as
    // the committed prefix grows. positions[] is re-uploaded (uncaptured) every
    // cycle and the linearized tree root sits at positions[0] == start_pos ==
    // block_start, so derive it from the device buffer instead of the scalar.
    // Eager (non-capture) value is identical, so this is a no-op off-graph.
    // A negative block_start selects a full-context bias: callers provide
    // [batch, seq_len] and the bias begins at cache slot zero.  This is
    // distinct from tree verify, whose block starts at positions[0] and whose
    // scalar start must remain graph-replay safe.
    const bool full_context_bias = tree_mode && block_start < 0;
    const int eff_block_start = tree_mode
        ? (full_context_bias ? 0 : positions[0])
        : block_start;
    // Tree mode: iterate over the full cache including the block.
    // Causal mode: attend to 0..positions[b] inclusive (unchanged).
    const int seq_len = tree_mode ? (eff_block_start + block_cols) : (positions[b] + 1);

    // One code path for both modes: in legacy mode we synthesise a descriptor
    // with zero bases, so the address arithmetic below is unchanged and the
    // output is bitwise identical to the pre-SP1 kernel.
    const int slot = (row_slot != nullptr) ? row_slot[b] : 0;
    const int per_pos_bytes = n_kv_heads * (head_dim / 32) * 34;
    // Legacy here also covers the pre-descriptor independent-sequence contract
    // (negative `max_seq` = per-lane capacity, row `b` reads its own slice).
    // kv_slot_legacy_lane folds that base into the descriptor, so the address
    // arithmetic below is the same one expression in all three modes.
    const KvSlotDesc desc = (slot_descs != nullptr)
        ? slot_descs[slot]
        : kv_slot_legacy_lane(seq_len, max_seq, b, per_pos_bytes);
    // `positions[row] + 1` (== `seq_len` above) is the PER-ROW CAUSAL BOUND;
    // `desc.seq_len` is the SLOT's logical KV length (allocation/capacity
    // metadata). They coincide only when a slot has exactly one query row —
    // a slot verifying M>1 draft tokens has rows at different positions that
    // must not all share one bound. `desc` supplies the K/V slab BASE
    // ADDRESS (via kv_offset_for_k/v below); it must never override the
    // row's own causal bound. See kv_slot_desc.h and
    // attention_flash_q8_0_tile_batched.hip for the full analysis (this
    // kernel has no separate reduce pass, so a wrong eff_seq_len here would
    // read live neighbor-slot KV bytes into softmax rather than fold in an
    // unwritten partial — a different corruption mode, same root cause).
    const int eff_seq_len = seq_len;

    // Sanity invariant, not a correctness mechanism: a row may never attend
    // past its own slot's logical KV length.
    // NOTE: a debug `assert(seq_len <= desc.seq_len)` used to sit here. It was
    // removed: crates/rdna-compute/src/compiler.rs never passes -DNDEBUG, so the
    // assert shipped in RELEASE and cost 64 bytes/lane of scratch on every one of
    // these kernels. The paired `__builtin_assume` was worse than useless — it
    // would only take effect under -DNDEBUG, where it becomes a UB licence on a
    // fact the kernel never otherwise uses, buying no optimisation. The invariant
    // is documented on KvSlotDesc and checked host-side by the builders instead.

    extern __shared__ float sdata[];

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int q_stride = n_heads * head_dim;
    const float* q_head = q + (long long)b * q_stride + h * head_dim;

    const int blocks_per_head = head_dim / 32;
    const int kv_head_block_start = kv_h * blocks_per_head;

    float* scores = sdata;
    float* workspace = sdata + seq_len;
    float* q_shared = workspace + nthreads;

    // Preload this batch row's Q head into shared memory.
    for (int d = tid; d < head_dim; d += nthreads)
        q_shared[d] = q_head[d];
    __syncthreads();

    // Phase 1: Q @ K^T with Q8_0 dequant.
    float local_max = -1e30f;
    for (int t = tid; t < eff_seq_len; t += nthreads) {
        float dot = 0.0f;
        for (int bi = 0; bi < blocks_per_head; bi++) {
            const unsigned char* blk = k_cache
                                      + kv_offset_for_k(desc, t, per_pos_bytes)
                                      + (kv_head_block_start + bi) * 34;
            float d = (float)*((const _Float16*)blk);
            const float* qb = q_shared + bi * 32;
            for (int j = 0; j < 32; j++) {
                dot += qb[j] * (d * (float)((signed char)blk[2 + j]));
            }
        }
        float s = dot * scale_attn;
        // Tree-attention bias: add per-row mask for keys inside the block.
        // Prompt keys (t < block_start) stay unmasked. A -inf bias on a
        // non-ancestor key makes expf(s - max) collapse to 0, so that key
        // contributes nothing to the numerator or denominator. Correctness
        // of the running-max reduction requires that we NOT mix -inf into
        // the max when every non-ancestor gets -inf — but the ancestor set
        // always includes the root (slot 0 in the block) so at least one
        // in-block score is finite on every row.
        if (tree_mode && t >= eff_block_start) {
            s += tree_bias[b * block_cols + (t - eff_block_start)];
        }
        scores[t] = s;
        local_max = fmaxf(local_max, s);
    }

    workspace[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] = fmaxf(workspace[tid], workspace[tid + s]);
        __syncthreads();
    }
    float max_val = workspace[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int t = tid; t < eff_seq_len; t += nthreads) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    workspace[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] += workspace[tid + s];
        __syncthreads();
    }
    float sum_val = workspace[0];
    __syncthreads();

    for (int t = tid; t < eff_seq_len; t += nthreads) {
        scores[t] /= sum_val;
    }
    __syncthreads();

    // Phase 2: weighted V sum with Q8_0 dequant.
    float* out_head = out + (long long)b * q_stride + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        int bi = d / 32;
        int bj = d % 32;
        for (int t = 0; t < eff_seq_len; t++) {
            const unsigned char* blk = v_cache
                                      + kv_offset_for_v(desc, t, per_pos_bytes)
                                      + (kv_head_block_start + bi) * 34;
            float vd = (float)*((const _Float16*)blk) * (float)(signed char)blk[2 + bj];
            val += scores[t] * vd;
        }
        out_head[d] = val;
    }
}

// Independent-sequence decode with active-mask + sliding window.
// Additive sibling of attention_q8_0_kv_batched's independent contract
// (negative max_seq). Existing symbol/ABI above is untouched.
//
// Grid: [n_heads, batch_size, 1]  — physical lane = blockIdx.y
// - active_mask bit b selects lane b (max 64 lanes; u64 mask).
// - Inactive lanes return before any Q/KV pointer dereference or write.
// - lane_capacity is positive; each lane reads only its private
//   [lane * capacity, (lane + 1) * capacity) Q8 cache slice.
// - window == 0 -> full causal over [0, positions[b]+1)
// - window  > 0 -> t_lo = max(0, positions[b]+1-window); attend [t_lo, seq_len)
// Scalar Q8 reduction order matches the independent path above (and SWA).
extern "C" __global__ void attention_q8_0_kv_independent_masked_windowed(
    const float* __restrict__ q,                // [batch_size × n_heads × head_dim]
    const unsigned char* __restrict__ k_cache,  // lane-major [batch × capacity × ...]
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,                    // [batch_size × n_heads × head_dim]
    const int* __restrict__ positions,          // [batch_size]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int lane_capacity,                          // positive capacity per lane
    float scale_attn,
    int batch_size,
    unsigned long long active_mask,
    int window
) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    // Early-exit inactive / OOB lanes before any Q/KV/positions dereference.
    if (b >= batch_size) return;
    if (((active_mask >> b) & 1ULL) == 0) return;
    if (h >= n_heads) return;

    const int seq_len = positions[b] + 1;
    // Sliding-window lookback. window==0 is full causal.
    const int t_lo = (window > 0 && seq_len > window) ? (seq_len - window) : 0;
    extern __shared__ float sdata[];

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int q_stride = n_heads * head_dim;
    const float* q_head = q + (long long)b * q_stride + h * head_dim;

    const int blocks_per_head = head_dim / 32;
    const int total_blocks_per_pos = n_kv_heads * blocks_per_head;
    const int kv_head_block_start = kv_h * blocks_per_head;
    const size_t lane_bytes = (size_t)lane_capacity * total_blocks_per_pos * 34;
    const unsigned char* k_lane = k_cache + (size_t)b * lane_bytes;
    const unsigned char* v_lane = v_cache + (size_t)b * lane_bytes;

    float* scores = sdata;
    float* workspace = sdata + seq_len;
    float* q_shared = workspace + nthreads;

    // Preload this batch row's Q head into shared memory.
    for (int d = tid; d < head_dim; d += nthreads)
        q_shared[d] = q_head[d];
    __syncthreads();

    // Phase 1: Q @ K^T with Q8_0 dequant over [t_lo, seq_len).
    float local_max = -1e30f;
    for (int t = t_lo + tid; t < seq_len; t += nthreads) {
        float dot = 0.0f;
        for (int bi = 0; bi < blocks_per_head; bi++) {
            const unsigned char* blk = k_lane + (size_t)t * total_blocks_per_pos * 34
                                      + (kv_head_block_start + bi) * 34;
            float d = (float)*((const _Float16*)blk);
            const float* qb = q_shared + bi * 32;
            for (int j = 0; j < 32; j++) {
                dot += qb[j] * (d * (float)((signed char)blk[2 + j]));
            }
        }
        float s = dot * scale_attn;
        scores[t] = s;
        local_max = fmaxf(local_max, s);
    }

    workspace[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] = fmaxf(workspace[tid], workspace[tid + s]);
        __syncthreads();
    }
    float max_val = workspace[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int t = t_lo + tid; t < seq_len; t += nthreads) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    workspace[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] += workspace[tid + s];
        __syncthreads();
    }
    float sum_val = workspace[0];
    __syncthreads();

    for (int t = t_lo + tid; t < seq_len; t += nthreads) {
        scores[t] /= sum_val;
    }
    __syncthreads();

    // Phase 2: weighted V sum with Q8_0 dequant over the window only.
    float* out_head = out + (long long)b * q_stride + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        int bi = d / 32;
        int bj = d % 32;
        for (int t = t_lo; t < seq_len; t++) {
            const unsigned char* blk = v_lane + (size_t)t * total_blocks_per_pos * 34
                                      + (kv_head_block_start + bi) * 34;
            float vd = (float)*((const _Float16*)blk) * (float)(signed char)blk[2 + bj];
            val += scores[t] * vd;
        }
        out_head[d] = val;
    }
}
