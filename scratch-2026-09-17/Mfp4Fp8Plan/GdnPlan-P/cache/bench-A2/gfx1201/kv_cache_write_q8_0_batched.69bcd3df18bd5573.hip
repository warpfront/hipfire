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


extern "C" __global__ void kv_cache_write_q8_0_batched(
    unsigned char* __restrict__ dst,
    const float* __restrict__ src,        // [batch_size × kv_dim]
    const int* __restrict__ positions,    // [batch_size]
    int n_kv_heads,
    int head_dim,
    int batch_size,
    const KvSlotDesc* __restrict__ slot_descs,  // [n_slots] or nullptr = legacy
    const int* __restrict__ row_slot            // [batch_size] or nullptr = legacy
) {
    const int gid = blockIdx.x;           // Q8_0 block index within one position
    const int bid = blockIdx.y;           // batch index
    if (bid >= batch_size) return;
    const int tid = threadIdx.x;          // 0..31

    const int blocks_per_head = head_dim / 32;
    const int total_blocks = n_kv_heads * blocks_per_head;
    if (gid >= total_blocks) return;

    // One code path for both modes: in legacy mode we synthesise a zero-base
    // descriptor, so the address arithmetic below is unchanged and the
    // output is bitwise identical to the pre-SP2 kernel.
    const int slot = (row_slot != nullptr) ? row_slot[bid] : 0;
    const KvSlotDesc desc = (slot_descs != nullptr)
        ? slot_descs[slot]
        : kv_slot_legacy(0, 0);
    const int per_pos_bytes = total_blocks * 34;

    const int pos = positions[bid];
    const int head_idx = gid / blocks_per_head;
    const int block_idx = gid % blocks_per_head;
    const int kv_dim = n_kv_heads * head_dim;
    const int elem_offset = bid * kv_dim + head_idx * head_dim + block_idx * 32 + tid;

    float val = src[elem_offset];

    // Warp max absolute value
    float amax = fabsf(val);
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset));

    float scale = amax / 127.0f;
    float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int q = __float2int_rn(val * inv_scale);
    q = max(-127, min(127, q));

    unsigned char* out = dst + kv_offset_for_k(desc, pos, per_pos_bytes) + gid * 34;
    if (tid == 0) *((_Float16*)(out)) = (_Float16)scale;
    out[2 + tid] = (unsigned char)(signed char)q;
}

// Independent-sequence sibling.  `positions` are logical per-lane positions;
// lane-major addressing prevents one request from overwriting another.
extern "C" __global__ void kv_cache_write_q8_0_independent(
    unsigned char* __restrict__ dst,
    const float* __restrict__ src,
    const int* __restrict__ positions,
    int n_kv_heads,
    int head_dim,
    int batch_size,
    int lane_capacity
) {
    const int gid = blockIdx.x;
    const int bid = blockIdx.y;
    if (bid >= batch_size) return;
    const int tid = threadIdx.x;

    const int blocks_per_head = head_dim / 32;
    const int total_blocks = n_kv_heads * blocks_per_head;
    if (gid >= total_blocks) return;

    const int pos = positions[bid];
    const int head_idx = gid / blocks_per_head;
    const int block_idx = gid % blocks_per_head;
    const int kv_dim = n_kv_heads * head_dim;
    const int elem_offset = bid * kv_dim + head_idx * head_dim + block_idx * 32 + tid;
    float val = src[elem_offset];

    float amax = fabsf(val);
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset));

    float scale = amax / 127.0f;
    float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int q = __float2int_rn(val * inv_scale);
    q = max(-127, min(127, q));

    const size_t flat_pos = (size_t)bid * lane_capacity + pos;
    unsigned char* out = dst + flat_pos * total_blocks * 34 + gid * 34;
    if (tid == 0) *((_Float16*)(out)) = (_Float16)scale;
    out[2 + tid] = (unsigned char)(signed char)q;
}

// Masked independent-sequence sibling. `active_mask` bit i selects lane i.
// Derives physical lane from grid y (blockIdx.y) and returns before any
// inactive-lane read/write. Full-mask callers route through the unmasked
// `kv_cache_write_q8_0_independent` wrapper for exact ABI parity.
extern "C" __global__ void kv_cache_write_q8_0_independent_masked(
    unsigned char* __restrict__ dst,
    const float* __restrict__ src,
    const int* __restrict__ positions,
    int n_kv_heads,
    int head_dim,
    int batch_size,
    int lane_capacity,
    unsigned long long active_mask
) {
    const int gid = blockIdx.x;
    const int bid = blockIdx.y;
    // Derive lane from existing grid and early-exit inactive lanes before any
    // state/input read or quantized write. Safe shift: bid < 64, active_mask
    // is u64 with max_batch 64.
    if (((active_mask >> bid) & 1ULL) == 0) return;
    if (bid >= batch_size) return;
    const int tid = threadIdx.x;

    const int blocks_per_head = head_dim / 32;
    const int total_blocks = n_kv_heads * blocks_per_head;
    if (gid >= total_blocks) return;

    const int pos = positions[bid];
    const int head_idx = gid / blocks_per_head;
    const int block_idx = gid % blocks_per_head;
    const int kv_dim = n_kv_heads * head_dim;
    const int elem_offset = bid * kv_dim + head_idx * head_dim + block_idx * 32 + tid;
    float val = src[elem_offset];

    float amax = fabsf(val);
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset));

    float scale = amax / 127.0f;
    float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int q = __float2int_rn(val * inv_scale);
    q = max(-127, min(127, q));

    const size_t flat_pos = (size_t)bid * lane_capacity + pos;
    unsigned char* out = dst + flat_pos * total_blocks * 34 + gid * 34;
    if (tid == 0) *((_Float16*)(out)) = (_Float16)scale;
    out[2 + tid] = (unsigned char)(signed char)q;
}