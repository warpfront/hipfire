// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Qwen4 sibling of hc_streams_init_from_embed_batched
// (hc_streams_init_from_embed_batched.hip): the same broadcast into BF16 streams.

#define BLOCK_HIDDEN 256

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
