// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Flat F16 KV-cache write (IEEE fp16; slot-engine tier).
//
// Layout: element `(pos, kv_head, d)` lives at
//   `pos * kv_dim + kv_head * head_dim + d`, one f16 (2 bytes) each,
// where `kv_dim = n_kv_heads * head_dim`. No blocks, no per-block scale, no
// padding. Compare `kv_cache_write_q8_0.hip`, which packs 34-byte blocks of
// 32 elements and needs a wave reduction to find each block's amax — none of
// that exists here, so this kernel is a pure elementwise convert-and-store
// and the block/thread mapping is free to be a plain flat grid.
//
// The same kernel serves K and V: the caller launches it twice with different
// `dst`/`src`, exactly as the Q8_0 path does.
#include <hip/hip_runtime.h>
#include "kv_slot_desc.h"

// F32 -> F16 round-to-nearest-even via the compiler's `_Float16` narrowing
// cast — the same idiom `deepseek4_convert_f32_to_f16.hip` uses. Unlike the
// BF16 writer (which CAN hand-roll RNE because bf16 is the f32's top 16
// bits), fp16 needs the real exponent re-bias + mantissa round the hardware
// conversion instruction performs; do not "optimize" this into a shift.
static __device__ __forceinline__ unsigned short hipfire_f32_to_f16_rne(float v) {
    return __builtin_bit_cast(unsigned short, (_Float16)v);
}

// Decode / single-position write.
// Grid: [ceil(kv_dim / 64), 1, 1]. Block: [64, 1, 1].
extern "C" __global__ void kv_cache_write_f16(
    unsigned char* __restrict__ dst,
    const float* __restrict__ src,    // [kv_dim] FP32 KV vector
    const int* __restrict__ pos_buf,
    int n_kv_heads,
    int head_dim
) {
    const int kv_dim = n_kv_heads * head_dim;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_dim) return;

    const int pos = pos_buf[0];
    unsigned short* out =
        (unsigned short*)(dst + (size_t)pos * (size_t)kv_dim * 2);
    out[i] = hipfire_f32_to_f16_rne(src[i]);
}

// Batched prefill write.
// Grid: [ceil(kv_dim / 64), batch_size, 1]. Block: [64, 1, 1].
//
// `kv_offset_for_k` is used for both the K and the V launch, mirroring
// `kv_cache_write_q8_0_batched`. That is correct because `dst` already
// selects the K or V arena; the descriptor only adds a slot base within it,
// and maple passes `slot_descs == nullptr` (legacy, base 0) today.
extern "C" __global__ void kv_cache_write_f16_batched(
    unsigned char* __restrict__ dst,
    const float* __restrict__ src,              // [batch_size × kv_dim]
    const int* __restrict__ positions,          // [batch_size]
    int n_kv_heads,
    int head_dim,
    int batch_size,
    const KvSlotDesc* __restrict__ slot_descs,  // [n_slots] or nullptr = legacy
    const int* __restrict__ row_slot            // [batch_size] or nullptr = legacy
) {
    const int kv_dim = n_kv_heads * head_dim;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int bid = blockIdx.y;
    if (bid >= batch_size) return;
    if (i >= kv_dim) return;

    const int slot = (row_slot != nullptr) ? row_slot[bid] : 0;
    const KvSlotDesc desc =
        (slot_descs != nullptr) ? slot_descs[slot] : kv_slot_legacy(0, 0);
    const int per_pos_bytes = kv_dim * 2;

    const int pos = positions[bid];
    unsigned short* out =
        (unsigned short*)(dst + kv_offset_for_k(desc, pos, per_pos_bytes));
    out[i] = hipfire_f32_to_f16_rne(src[(size_t)bid * (size_t)kv_dim + i]);
}
