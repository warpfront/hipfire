// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// LFM3 attention output gate: per-head sigmoid gate broadcast over the
// attention output. LFM3's `Lfm3Attention` applies
//
//     gate = sigmoid(gate_proj(hidden))   // [n_heads]
//     attn_out = attn_out * gate          // broadcast: each head's [head_dim]
//                                        //   slice scaled by its scalar gate
//
// `gate` holds the PRE-sigmoid logits (length n_heads); `out` is
// [n_heads * head_dim] in/out. One thread per output element.
//
// Bit-exact with sigmoid.hip's formula: s = 1/(1+exp(-g)).

extern "C" __global__ void sigmoid_head_gate_mul_f32(
    float* __restrict__ out,          // [n_heads * head_dim] in/out
    const float* __restrict__ gate,   // [n_heads] pre-sigmoid logits
    int head_dim,
    int total                         // n_heads * head_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int h = i / head_dim;
    float s = 1.0f / (1.0f + expf(-gate[h]));
    out[i] = out[i] * s;
}
