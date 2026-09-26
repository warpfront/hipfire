// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// Broadcast embedded tokens to all HC residual streams (Phase B2, 2026-05-18).
//
// DeepSeek V4's `init_residual_streams` per-token sequence is:
//   1. embedding_lookup_q8(token) → embed_scratch [hidden]
//   2. memcpy embed_scratch into every slot of residual_streams [hc_mult, hidden]
// matching antirez ds4 `hc_from_plain_embedding` (ds4.c:4358) which inits
// ALL hc_mult streams with a copy of the embedding (not [embed, 0, 0, 0]).
//
// For batched prefill, step 2 becomes:
//   streams[b, h, d] = embed[b, d]   ∀ b ∈ [0, B), h ∈ [0, hc_mult), d ∈ [0, hidden)
//
// Single kernel launch instead of B * hc_mult d2d memcpys.
//
// Grid:  [(hidden+255)/256, batch_size, 1]
// Block: [256, 1, 1]
//
// At batch_size == 1 the broadcast pattern is identical to the per-token
// memcpy loop (each element written once per stream).

#include <hip/hip_runtime.h>

#define BLOCK_HIDDEN 256

__launch_bounds__(BLOCK_HIDDEN)
extern "C" __global__ void hc_streams_init_from_embed_batched(
    const float* __restrict__ embed,    // [batch, hidden]
    float*       __restrict__ streams,  // [batch, hc_mult, hidden]
    int hidden,
    int hc_mult,
    int batch_size
) {
    const int tid = threadIdx.x;
    const int b   = blockIdx.y;
    const int d   = blockIdx.x * BLOCK_HIDDEN + tid;
    if (d >= hidden || b >= batch_size) return;

    const float v = embed[(size_t)b * hidden + d];
    // Unroll over hc_mult — typically 4 for DeepSeek V4; bounded for safety.
    for (int h = 0; h < hc_mult; ++h) {
        streams[((size_t)b * hc_mult + h) * hidden + d] = v;
    }
}

// Qwen4 F16 prefill route: the residual streams hold BF16 bits (RNE).  Every
// HC reader rounds the stream to BF16 on load, so this is the value it uses.
__launch_bounds__(BLOCK_HIDDEN)
extern "C" __global__ void hc_streams_init_from_embed_batched_bf16(
    const float* __restrict__ embed,     // [batch, hidden]
    unsigned short* __restrict__ streams,  // [batch, hc_mult, hidden] BF16
    int hidden,
    int hc_mult,
    int batch_size
) {
    const int tid = threadIdx.x;
    const int b   = blockIdx.y;
    const int d   = blockIdx.x * BLOCK_HIDDEN + tid;
    if (d >= hidden || b >= batch_size) return;
    const unsigned int u = __float_as_uint(embed[(size_t)b * hidden + d]);
    // hip_bfloat16's float -> BF16 (RNE; NaN payloads keep a set bit).
    const unsigned short v = (u & 0x7F800000u) == 0x7F800000u
        ? (unsigned short)(((u & 0xFFFFu) ? (u | 0x10000u) : u) >> 16)
        : (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
    for (int h = 0; h < hc_mult; ++h) {
        streams[((size_t)b * hc_mult + h) * hidden + d] = v;
    }
}
