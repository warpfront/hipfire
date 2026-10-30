// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// VMM-backed shared paged KV descriptor — the address translation the
// `*_vmm_paged` kernel variants compile against.
//
// The logical page stays PAGE_TOKENS positions and the block table stays the
// same i32 array as `kv_slot_desc_paged.h`. What changes is the physical
// placement: each VMM granule (`granule_bytes`, e.g. 2 MiB) holds
// `pages_per_granule` consecutive physical pages, and page byte offsets inside
// a granule are `within * page_tokens * per_pos_bytes`. The tail of a granule
// is unused when a page does not divide it.
//
//   block_table[logical_page] -> phys_page
//   granule = phys_page / pages_per_granule
//   within  = phys_page % pages_per_granule
//   offset  = granule * granule_bytes
//           + within  * page_tokens * per_pos_bytes
//           + page_off * per_pos_bytes
//
// `granule_bytes == 0` or `pages_per_granule == 0` degenerates to the
// contiguous paged formula, so one code path serves both. Descriptor-less
// launches synthesize a legacy descriptor and never reach this header's
// translation; `_vmm_paged` modules are only launched with a real descriptor.
//
// Layout must stay byte-identical to `VmmPagedKvSlotDesc` in
// crates/rdna-compute/src/kv_slots.rs. 48 bytes, 8-byte aligned (the final
// 4-byte pad after pages_per_granule is explicit).

#pragma once

#include <hip/hip_runtime.h>

#define HIPFIRE_KV_SLOT_PAGED 1
#define HIPFIRE_KV_SLOT_VMM_PAGED 1

struct KvSlotDesc {
    // GPU pointer to the physical page-index array, or nullptr for legacy
    // contiguous mode. In paged mode entries [0, ceil(seq_len/page_tokens))
    // are live; kernels read only positions below seq_len.
    const int* __restrict__ block_table;

    // Legacy contiguous slab offsets. Unused by VMM-paged descriptors.
    unsigned long long legacy_k_base;
    unsigned long long legacy_v_base;

    int seq_len;        // logical KV length; kernel reads [0, seq_len)
    int page_tokens;    // tokens per logical page; 0 = legacy contiguous

    // VMM granule packing. Both zero selects the contiguous paged formula.
    unsigned long long granule_bytes;
    unsigned int pages_per_granule;
    unsigned int _pad;
};

// Byte offset of position `pos` in the K arena. `per_pos_bytes` is the
// per-position stride, uniform across slots. The same physical page index is
// used for K and V; phase 1 admits equal K/V strides, so one packing applies.
__device__ __forceinline__ unsigned long long kv_vmm_paged_offset(
    const KvSlotDesc& s, int pos, int per_pos_bytes)
{
    const int page_idx = pos / s.page_tokens;
    const int page_off = pos - page_idx * s.page_tokens;
    const int phys_page = s.block_table[page_idx];
    const unsigned long long page_in_granule =
        (unsigned long long)(page_off) * (unsigned long long)per_pos_bytes;
    if (s.granule_bytes != 0ULL && s.pages_per_granule != 0u) {
        const unsigned int granule = (unsigned int)phys_page / s.pages_per_granule;
        const unsigned int within = (unsigned int)phys_page % s.pages_per_granule;
        return (unsigned long long)granule * s.granule_bytes
             + (unsigned long long)within
               * (unsigned long long)s.page_tokens
               * (unsigned long long)per_pos_bytes
             + page_in_granule;
    }
    return (unsigned long long)(unsigned int)phys_page
         * (unsigned long long)s.page_tokens
         * (unsigned long long)per_pos_bytes
         + page_in_granule;
}

__device__ __forceinline__ unsigned long long kv_offset_for_k(
    const KvSlotDesc& s, int pos, int per_pos_bytes)
{
    if (s.block_table != nullptr) {
        return kv_vmm_paged_offset(s, pos, per_pos_bytes);
    }
    return s.legacy_k_base + (unsigned long long)pos * (unsigned long long)per_pos_bytes;
}

__device__ __forceinline__ unsigned long long kv_offset_for_v(
    const KvSlotDesc& s, int pos, int per_pos_bytes)
{
    if (s.block_table != nullptr) {
        return kv_vmm_paged_offset(s, pos, per_pos_bytes);
    }
    return s.legacy_v_base + (unsigned long long)pos * (unsigned long long)per_pos_bytes;
}

// Descriptor-less fallbacks exist only so shared bodies compile. A
// `_vmm_paged` launch always passes descriptors; if one ever did not, the
// synthesized descriptor has no block table and every paged access would
// read the legacy base (zero), which would be caught by the launch gate.
__device__ __forceinline__ KvSlotDesc kv_slot_legacy(int seq_len, int max_seq)
{
    KvSlotDesc s;
    s.block_table = nullptr;
    s.legacy_k_base = 0ULL;
    s.legacy_v_base = 0ULL;
    s.seq_len = seq_len;
    s.page_tokens = 0;
    s.granule_bytes = 0ULL;
    s.pages_per_granule = 0u;
    s._pad = 0u;
    return s;
}

__device__ __forceinline__ KvSlotDesc kv_slot_legacy_lane(
    int seq_len, int max_seq, int row, int per_pos_bytes)
{
    const bool independent = max_seq < 0;
    const int lane_capacity = independent ? -max_seq : max_seq;
    const unsigned long long lane_bytes =
        (unsigned long long)lane_capacity * (unsigned long long)per_pos_bytes;
    const unsigned long long base =
        independent ? (unsigned long long)row * lane_bytes : 0ULL;

    KvSlotDesc s = kv_slot_legacy(seq_len, max_seq);
    s.legacy_k_base = base;
    s.legacy_v_base = base;
    return s;
}
