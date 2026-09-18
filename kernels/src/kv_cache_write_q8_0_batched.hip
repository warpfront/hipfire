// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include "kv_slot_desc.h"

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

#ifdef HIPFIRE_KV_FP8_E4M3
// Batched native E4M3FN KV write. Counterpart of kv_cache_write_fp8_e4m3
// for prefill batching — one wave per (head, batch row). Contiguous
// `positions[]` destinations; slot/independent addressing is not enabled
// for fp8. Same scale/code rule as the single writer.
//
// Grid: [n_kv_heads, batch_size, 1]. Block: [32, 1, 1].
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "kv_cache_write_fp8_e4m3_batched requires --offload-arch=gfx1201 (OCP E4M3FN cvt)"
#endif
extern "C" __global__ void kv_cache_write_fp8_e4m3_batched(
    unsigned char* __restrict__ dst,
    const float* __restrict__ src,        // [batch_size × n_kv_heads × head_dim]
    const int* __restrict__ positions,    // [batch_size]
    int n_kv_heads,
    int head_dim,
    int batch_size
) {
    const int head = blockIdx.x;
    const int bid = blockIdx.y;
    if (bid >= batch_size) return;
    if (head >= n_kv_heads) return;
    const int tid = threadIdx.x;

    const int kv_dim = n_kv_heads * head_dim;
    float v[8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        v[i] = src[(size_t)bid * kv_dim + (size_t)head * head_dim + tid * 8 + i];

    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) amax = fmaxf(amax, fabsf(v[i]));
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset));

    _Float16 s16 = (amax == 0.0f) ? (_Float16)1.0f : (_Float16)(amax / 448.0f);
    if (amax > 0.0f && s16 == (_Float16)0.0f)
        s16 = __builtin_bit_cast(_Float16, (unsigned short)0x0001);
    float sf = (float)s16;
    if (amax > 0.0f && 448.0f * sf < amax) {
        unsigned short bits = __builtin_bit_cast(unsigned short, s16);
        bits = (bits >= 0x7bff) ? (unsigned short)0x7bff : (unsigned short)(bits + 1u);
        s16 = __builtin_bit_cast(_Float16, bits);
        sf = (float)s16;
    }

    unsigned w0 = 0u;
    w0 = __builtin_amdgcn_cvt_pk_fp8_f32(v[0] / sf, v[1] / sf, w0, false);
    w0 = __builtin_amdgcn_cvt_pk_fp8_f32(v[2] / sf, v[3] / sf, w0, true);
    unsigned w1 = 0u;
    w1 = __builtin_amdgcn_cvt_pk_fp8_f32(v[4] / sf, v[5] / sf, w1, false);
    w1 = __builtin_amdgcn_cvt_pk_fp8_f32(v[6] / sf, v[7] / sf, w1, true);

    const int pos = positions[bid];
    const size_t stride = (size_t)n_kv_heads * (head_dim + 2);
    unsigned char* codes = dst + (size_t)pos * stride + (size_t)head * head_dim + tid * 8;
    __builtin_memcpy(codes + 0, &w0, 4);
    __builtin_memcpy(codes + 4, &w1, 4);
    if (tid == 0)
        *((_Float16*)(dst + (size_t)pos * stride + (size_t)n_kv_heads * head_dim + (size_t)head * 2)) = s16;
}
#ifdef HIPFIRE_KV_FP8_E4M3
// Batched V-side grouped-fragment write. Counterpart of
// kv_cache_write_fp8_e4m3_batched with the Vfill code placement (same
// amax/scale/cvt values, dim-major codes within each 8-key group, scales
// home). Grid/block/ABI identical to the K batched writer.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "kv_cache_write_fp8_e4m3_v_batched requires --offload-arch=gfx1201 (OCP E4M3FN cvt)"
#endif
extern "C" __global__ void kv_cache_write_fp8_e4m3_v_batched(
    unsigned char* __restrict__ dst,
    const float* __restrict__ src,        // [batch_size × n_kv_heads × head_dim]
    const int* __restrict__ positions,    // [batch_size]
    int n_kv_heads,
    int head_dim,
    int batch_size
) {
    const int head = blockIdx.x;
    const int bid = blockIdx.y;
    if (bid >= batch_size) return;
    if (head >= n_kv_heads) return;
    const int tid = threadIdx.x;

    const int kv_dim = n_kv_heads * head_dim;
    float v[8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        v[i] = src[(size_t)bid * kv_dim + (size_t)head * head_dim + tid * 8 + i];

    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) amax = fmaxf(amax, fabsf(v[i]));
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset));

    _Float16 s16 = (amax == 0.0f) ? (_Float16)1.0f : (_Float16)(amax / 448.0f);
    if (amax > 0.0f && s16 == (_Float16)0.0f)
        s16 = __builtin_bit_cast(_Float16, (unsigned short)0x0001);
    float sf = (float)s16;
    if (amax > 0.0f && 448.0f * sf < amax) {
        unsigned short bits = __builtin_bit_cast(unsigned short, s16);
        bits = (bits >= 0x7bff) ? (unsigned short)0x7bff : (unsigned short)(bits + 1u);
        s16 = __builtin_bit_cast(_Float16, bits);
        sf = (float)s16;
    }

    unsigned w0 = 0u;
    w0 = __builtin_amdgcn_cvt_pk_fp8_f32(v[0] / sf, v[1] / sf, w0, false);
    w0 = __builtin_amdgcn_cvt_pk_fp8_f32(v[2] / sf, v[3] / sf, w0, true);
    unsigned w1 = 0u;
    w1 = __builtin_amdgcn_cvt_pk_fp8_f32(v[4] / sf, v[5] / sf, w1, false);
    w1 = __builtin_amdgcn_cvt_pk_fp8_f32(v[6] / sf, v[7] / sf, w1, true);

    const int pos = positions[bid];
    const size_t stride = (size_t)n_kv_heads * (head_dim + 2);
    const unsigned char* w0b = (const unsigned char*)&w0;
    const unsigned char* w1b = (const unsigned char*)&w1;
    unsigned char* base = dst + (size_t)(pos >> 3) * stride * 8u
        + (size_t)head * (size_t)head_dim * 8u + (size_t)tid * 64u + (size_t)(pos & 7);
    base[0] = w0b[0]; base[8] = w0b[1]; base[16] = w0b[2]; base[24] = w0b[3];
    base[32] = w1b[0]; base[40] = w1b[1]; base[48] = w1b[2]; base[56] = w1b[3];
    if (tid == 0)
        *((_Float16*)(dst + (size_t)pos * stride + (size_t)n_kv_heads * head_dim + (size_t)head * 2)) = s16;
}
#endif
#endif

#ifdef HIPFIRE_KV_BF16
// Batched flat-BF16 KV write. Elementwise f32→bf16 RNE convert-and-store
// into `[pos][kv_head][d]` rows (2 bytes/element, no scales); the same
// kernel serves K and V with different dst. Mirrors the contiguous
// `kv_cache_write_bf16` mapping and RNE rule (duplicated, not shared: the
// JIT header mechanism is hard-blocked to kv_slot_desc.h).
//
// Grid: [n_kv_heads, batch_size, 1]. Block: [256, 1, 1] (one thread per
// element at D256; larger head_dim loops, smaller takes a prefix).
static __device__ __forceinline__ unsigned short hipfire_f32_to_bf16_rne_batched(float val) {
    const unsigned int bits = __builtin_bit_cast(unsigned int, val);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return (unsigned short)(((bits >> 16) & 0x8000u) | 0x7fc0u);
    }
    const unsigned int lsb = (bits >> 16) & 1u;
    const unsigned int lower = bits & 0xffffu;
    const unsigned int round = (lower > 0x8000u) || ((lower == 0x8000u) && (lsb != 0u));
    return (unsigned short)((bits >> 16) + round);
}
extern "C" __global__ void kv_cache_write_bf16_batched(
    unsigned short* __restrict__ dst,     // flat bf16 cache
    const float* __restrict__ src,        // [batch_size × n_kv_heads × head_dim]
    const int* __restrict__ positions,    // [batch_size]
    int n_kv_heads,
    int head_dim,
    int batch_size
) {
    const int head = blockIdx.x;
    const int bid = blockIdx.y;
    if (bid >= batch_size) return;
    if (head >= n_kv_heads) return;
    const int kv_dim = n_kv_heads * head_dim;
    const int pos = positions[bid];
    unsigned short* row = dst + (size_t)pos * kv_dim + (size_t)head * head_dim;
    const float* srow = src + (size_t)bid * kv_dim + (size_t)head * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        row[d] = hipfire_f32_to_bf16_rne_batched(srow[d]);
}
#endif