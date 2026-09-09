// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx1010-owned strided QKV-row sibling for Spark chunk prefill. Plain
// single-arena addressing only (no slot descriptors / tree args). Reads K or
// V from a concatenated QKV row: src = b*stride + col + head*hd + block*32 +
// lane. Quantization is byte-exact with kv_cache_write_q8_0 (single-query).
// Grid: [n_kv_heads * (head_dim/32), batch_size, 1]. Block: [32, 1, 1]. LDS: 0.
// Bodies compile only when __gfx1010__ is defined (headers above stay live).

#if defined(__gfx1010__)

extern "C" __global__ void kv_cache_write_q8_0_batched_strided_gfx1010(
    unsigned char* __restrict__ dst,
    const float* __restrict__ qkv,        // [batch_size × stride] F32
    const int* __restrict__ positions,    // [batch_size]
    int n_kv_heads,
    int head_dim,
    int batch_size,
    int stride,
    int col
) {
    const int gid = blockIdx.x;           // Q8_0 block index within one position
    const int bid = blockIdx.y;           // batch index
    if (bid >= batch_size) return;
    const int tid = threadIdx.x;          // 0..31

    const int blocks_per_head = head_dim / 32;
    const int total_blocks = n_kv_heads * blocks_per_head;
    if (gid >= total_blocks) return;

    const int pos = positions[bid];
    const int head_idx = gid / blocks_per_head;
    const int block_idx = gid % blocks_per_head;
    // 4D addressing: row → column offset → head → Q8 block → lane
    const int elem_offset = bid * stride + col + head_idx * head_dim + block_idx * 32 + tid;

    float val = qkv[elem_offset];

    // Warp max absolute value — identical order to single-query writer
    float amax = fabsf(val);
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset));

    float scale = amax / 127.0f;
    float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int q = __float2int_rn(val * inv_scale);
    q = max(-127, min(127, q));

    // Plain single-arena: pos * total_blocks * 34 + gid * 34
    unsigned char* out = dst + (size_t)pos * total_blocks * 34 + gid * 34;
    if (tid == 0) *((_Float16*)(out)) = (_Float16)scale;
    out[2 + tid] = (unsigned char)(signed char)q;
}

#endif // defined(__gfx1010__)
