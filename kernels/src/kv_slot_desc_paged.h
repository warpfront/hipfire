// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Paged multi-slot KV descriptor — the KV address translation the multi-slot
// (SCS) engine's `*_paged` kernel modules compile against.
//
// Descriptor-less launches (every default route) compile the SAME kernel
// bodies against `kv_slot_desc.h` instead, which is the unchanged 24-byte
// legacy descriptor, so their code objects are exactly the pre-paging ones.
// Only modules the slot engine launches with real descriptors prepend this
// header; it defines HIPFIRE_KV_SLOT_PAGED so a body can carry paged-only
// kernel parameters (e.g. `use_v_base`, `slot_descs`/`row_slot`) behind
// `#ifdef HIPFIRE_KV_SLOT_PAGED`.
//
// A "slot" is one independent sequence in a batch. Each slot's KV is stored
// in a paged block table: `block_table[logical_page]` gives the physical
// page index, and the byte offset is:
//
//   phys_page * page_tokens * per_pos_bytes + page_off * per_pos_bytes
//
// with the stride passed per call — K and V may (and for asym3 DO) differ,
// but the physical page a logical page maps to is shared, so both arenas
// resolve through the one block table.
//
// When `block_table` is null the descriptor is in legacy contiguous mode and
// translation degenerates to `legacy_k/v_base + pos * stride`, byte-identical
// to the pre-paged kernel. This keeps every kernel on ONE code path: callers
// either pass a real paged descriptor or a legacy fallback built on the stack.
//
// Legacy K and V bases are SEPARATE fields. They differ whenever the K and V
// per-position strides differ — asym3 (3-bit rotated K against Q8_0 V) is the
// standing case, and its arena builders pack K and V slabs at different
// offsets. Q8_0 / BF16 pools keep them equal (their strides match, and the
// Q8 flash-prefill kernel stages K and V through ONE shared slab offset —
// it reads only legacy_k_base, so carrying a second base costs it nothing).
//
// Layout must stay byte-identical to the Rust mirror in
// crates/rdna-compute/src/kv_slots.rs (`KvSlotDesc`). 32 bytes, 8-byte aligned.
//
// KERNEL-AUTHORING INVARIANT (MTP stale-read safety): every batched attention
// kernel MUST bound its causal sweep by the per-row `positions[]` array the
// host uploads, never by `desc.seq_len`. The slot engine's MTP verify writes
// k+1 rows ahead of the committed frontier; after a partial accept the rows
// between the frontier and the draft tip hold REJECTED tokens' KV until the
// next cycle overwrites them. Bounding by positions[] masks those rows off
// (they sit past every live query's position); bounding by seq_len would
// sweep them back into the softmax as silent garbage.

#pragma once

#include <hip/hip_runtime.h>

#define HIPFIRE_KV_SLOT_PAGED 1

struct KvSlotDesc {
    // GPU pointer to array of physical page indices, or nullptr for legacy
    // contiguous mode. In paged mode the array has `ceil(seq_len / page_tokens)`
    // valid entries; the kernel only reads entries for positions < seq_len.
    const int* __restrict__ block_table;

    // Byte offset of this slot's K slab in the K arena (legacy mode only).
    unsigned long long legacy_k_base;

    // Byte offset of this slot's V slab in the V arena (legacy mode only).
    // Equal to legacy_k_base whenever K and V share a per-position stride.
    unsigned long long legacy_v_base;

    int seq_len;        // logical KV length; kernel reads [0, seq_len)
    int page_tokens;    // tokens per page (e.g. 128); 0 = legacy contiguous mode
};

// Byte offset of position `pos` within the K arena.
// `per_pos_bytes` is the per-position stride in bytes, uniform across slots
// (n_kv_heads * (head_dim/32) * 34 for Q8_0).
__device__ __forceinline__ unsigned long long kv_offset_for_k(
    const KvSlotDesc& s, int pos, int per_pos_bytes)
{
    if (s.block_table != nullptr) {
        const int page_idx = pos / s.page_tokens;
        const int page_off = pos % s.page_tokens;
        const int phys_page = s.block_table[page_idx];
        return (unsigned long long)phys_page
             * (unsigned long long)s.page_tokens
             * (unsigned long long)per_pos_bytes
           + (unsigned long long)page_off
             * (unsigned long long)per_pos_bytes;
    }
    return s.legacy_k_base + (unsigned long long)pos * (unsigned long long)per_pos_bytes;
}

__device__ __forceinline__ unsigned long long kv_offset_for_v(
    const KvSlotDesc& s, int pos, int per_pos_bytes)
{
    // In paged mode K and V share the same block table (same page layout);
    // the caller passes each arena's own stride.
    if (s.block_table != nullptr) {
        const int page_idx = pos / s.page_tokens;
        const int page_off = pos % s.page_tokens;
        const int phys_page = s.block_table[page_idx];
        return (unsigned long long)phys_page
             * (unsigned long long)s.page_tokens
             * (unsigned long long)per_pos_bytes
           + (unsigned long long)page_off
             * (unsigned long long)per_pos_bytes;
    }
    return s.legacy_v_base + (unsigned long long)pos * (unsigned long long)per_pos_bytes;
}

// Single-slot fallback used when the descriptor pointer is null. Keeps the
// ported kernels on ONE code path: callers build this on the stack from the
// legacy scalar args, so there is no `if (descs) ... else ...` around every
// KV read. Behaviour is then bitwise identical to the pre-paged kernel.
__device__ __forceinline__ KvSlotDesc kv_slot_legacy(int seq_len, int max_seq)
{
    KvSlotDesc s;
    s.block_table = nullptr;
    s.legacy_k_base = 0ULL;
    s.legacy_v_base = 0ULL;
    s.seq_len = seq_len;
    s.page_tokens = 0;
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
//
// K and V share the lane buffer at the same offset here (single flat cache),
// so both legacy bases get the lane base.
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
    s.block_table = nullptr;
    s.legacy_k_base = base;
    s.legacy_v_base = base;
    s.seq_len = seq_len;
    s.page_tokens = 0;
    return s;
}
