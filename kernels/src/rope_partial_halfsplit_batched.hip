// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched partial RoPE, half-split convention (HF rotate_half) — see
// kernels/src/rope_partial_halfsplit.hip for the per-position twin and
// docs/plans/qwen35-mq4-quality-gap.md §"RoPE convention probe / halfsplit fix"
// for the rationale. Pairs are (i, i + n_rot/2). Dispatched from
// rope_partial_interleaved_f32_batched as the default since 2026-05-12;
// the interleaved batched kernel remains available for legacy reproducibility
// behind HIPFIRE_ROPE_INTERLEAVED_LEGACY=1.
//
// Grid: [ceil(n_rot/2 / block), batch_size, 1]
// Q/K are [batch_size × n_heads × head_dim] row-major.

extern "C" __global__ void rope_partial_halfsplit_batched_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ positions,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int n_rot,
    float freq_base,
    int batch_size,
    int pos_offset   // added to positions[b] for the RoPE angle (NOT the KV slot);
                     // carries kv_cache.compact_offset so post-eviction batched
                     // Q/K rotate at absolute phase matching the cached keys. 0 = no-op.
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    int half = n_rot / 2;
    if (i >= half || b >= batch_size) return;

    int pos = positions[b] + pos_offset;

    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)n_rot);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    int q_stride = n_heads_q * head_dim;
    int k_stride = n_heads_k * head_dim;

    for (int h = 0; h < n_heads_q; h++) {
        int base = b * q_stride + h * head_dim;
        float v0 = q[base + i];
        float v1 = q[base + i + half];
        q[base + i]        = v0 * cos_a - v1 * sin_a;
        q[base + i + half] = v0 * sin_a + v1 * cos_a;
    }

    for (int h = 0; h < n_heads_k; h++) {
        int base = b * k_stride + h * head_dim;
        float v0 = k[base + i];
        float v1 = k[base + i + half];
        k[base + i]        = v0 * cos_a - v1 * sin_a;
        k[base + i + half] = v0 * sin_a + v1 * cos_a;
    }
}

// Strided single-buffer partial RoPE over packed QKV rows.
// Supports both half-split (interleaved==0) and legacy interleaved (interleaved!=0)
// pair layouts in one kernel. Pair p rotates:
//   halfsplit:   (p, p + nrot/2)
//   interleaved: (2p, 2p+1)
// Both use freq = 1/powf(fb, (2p)/nrot) and angle = (positions[b]+pos_offset)*freq.
// Dims [nrot, hd) and the V region are pass-through.
// Grid: [ceil((nrot/2)/block), bs, 1], block = min(32, nrot/2).
extern "C" __global__ void rope_partial_batched_strided_f32(
    float* __restrict__ qkv,
    const int* __restrict__ positions,  // [bs]
    int nhq,
    int nhk,
    int hd,
    int nrot,
    float fb,
    int bs,
    int pos_offset,
    int qoff,
    int koff,
    int stride,
    int interleaved
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;  // pair index
    int b = blockIdx.y;                             // batch row
    int half = nrot / 2;
    if (p >= half || b >= bs) return;

    int pos = positions[b] + pos_offset;

    float freq = 1.0f / powf(fb, (float)(2 * p) / (float)nrot);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // Pair layout selected by host flag (rope_interleaved_legacy).
    int d0 = interleaved ? (p * 2) : p;
    int d1 = interleaved ? (p * 2 + 1) : (p + half);

    // Wide row arithmetic — avoid i32 overflow on large batch*stride.
    long long row = (long long)b * (long long)stride;

    for (int h = 0; h < nhq; h++) {
        long long base = row + (long long)qoff + (long long)h * (long long)hd;
        float v0 = qkv[base + d0];
        float v1 = qkv[base + d1];
        qkv[base + d0] = v0 * cos_a - v1 * sin_a;
        qkv[base + d1] = v0 * sin_a + v1 * cos_a;
    }

    for (int h = 0; h < nhk; h++) {
        long long base = row + (long long)koff + (long long)h * (long long)hd;
        float v0 = qkv[base + d0];
        float v1 = qkv[base + d1];
        qkv[base + d0] = v0 * cos_a - v1 * sin_a;
        qkv[base + d1] = v0 * sin_a + v1 * cos_a;
    }
}
