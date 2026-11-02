// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Per-request VMM KV descriptor — the address translation the `*_vmm`
// kernel module variants compile against (continuous batching over
// independently reserved per-request VMM KV owners).
//
// Each request owns its own stable VMM reservation per KV layer, so a row's
// KV byte address is the ABSOLUTE virtual address
//
//   k_base + pos * per_pos_bytes        (V: v_base + pos * per_pos_bytes)
//
// with no block table and no shared physical arena. `_vmm` modules are
// launched with NULL k/v arena pointers, so the bodies' unchanged
// `arena + kv_offset_for_*(desc, pos, stride)` expression evaluates to that
// absolute address. Never reinterpret legacy/paged offsets as pointers: this
// header is only prepended to modules renamed `*_vmm`; the legacy 24/32-byte
// headers and their code objects are untouched.
//
// Fail-closed bounds: every translation traps unless
//   0 <= pos < mapped_positions <= logical_bound.
// A masked/idle slot carries mapped_positions == 0, so any access traps
// instead of touching memory. The host validates the same predicate (plus the
// owner generation / request epoch) before upload (kv_slots.rs
// `validate_vmm_rows`); the device trap is the last line of defence, never
// the expected path. The causal bound still comes from per-row positions[]
// (see kv_slot_desc_paged.h's stale-read invariant).
//
// Layout must stay byte-identical to `VmmKvSlotDesc` in
// crates/rdna-compute/src/kv_slots.rs. 32 bytes, 8-byte aligned.

#pragma once

#include <hip/hip_runtime.h>

// `_vmm` modules take the paged-only kernel parameters (e.g. `use_v_base`).
#define HIPFIRE_KV_SLOT_PAGED 1
#define HIPFIRE_KV_SLOT_VMM 1

struct KvSlotDesc {
    unsigned long long k_base;           // absolute device VA, K owner
    unsigned long long v_base;           // absolute device VA, V owner
    unsigned int mapped_positions;       // physically mapped prefix (tokens)
    unsigned int logical_bound;          // admission-resolved context bound
    unsigned long long owner_generation; // VMM owner generation (host-checked)
};

__device__ __forceinline__ void kv_vmm_check(const KvSlotDesc& s, int pos)
{
    if (pos < 0 || (unsigned int)pos >= s.mapped_positions
        || s.mapped_positions > s.logical_bound) {
        __builtin_trap();
    }
}

__device__ __forceinline__ unsigned long long kv_offset_for_k(
    const KvSlotDesc& s, int pos, int per_pos_bytes)
{
    kv_vmm_check(s, pos);
    return s.k_base + (unsigned long long)pos * (unsigned long long)per_pos_bytes;
}

__device__ __forceinline__ unsigned long long kv_offset_for_v(
    const KvSlotDesc& s, int pos, int per_pos_bytes)
{
    kv_vmm_check(s, pos);
    return s.v_base + (unsigned long long)pos * (unsigned long long)per_pos_bytes;
}

// Descriptor-less fallbacks exist only so shared bodies compile. A `_vmm`
// launch always passes descriptors; if one ever did not, the synthesised
// descriptor has an empty mapped prefix and every access traps.
__device__ __forceinline__ KvSlotDesc kv_slot_legacy(int seq_len, int max_seq)
{
    KvSlotDesc s;
    s.k_base = 0ULL;
    s.v_base = 0ULL;
    s.mapped_positions = 0u;
    s.logical_bound = 0u;
    s.owner_generation = 0ULL;
    return s;
}

__device__ __forceinline__ KvSlotDesc kv_slot_legacy_lane(
    int seq_len, int max_seq, int row, int per_pos_bytes)
{
    return kv_slot_legacy(seq_len, max_seq);
}
