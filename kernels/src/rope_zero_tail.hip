// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Partial RoPE, zero-tail convention (LFM3 full-attention layers).
//
// LFM3 keys RoPE per layer type. The `full_attention` layers use
// partial_rotary_factor = 0.5 with HF's ZERO-TAIL layout
// (modeling_lfm3.py compute_default_rope_parameters):
//
//   dim         = head_dim                       (128)
//   rope_angles = int(partial * dim) // 2        (32 rotated pairs)
//   inv_freq[i] = base^(-2i / dim)  for i in [0, rope_angles)
//   inv_freq    = cat(inv_freq, zeros(dim/2 - rope_angles))   (zero tail)
//   freqs       = pos * inv_freq                 ([seq, dim/2])
//   emb         = cat(freqs, freqs)              ([seq, dim])
//   q_embed     = q * cos + rotate_half(q) * sin (full-length rotate_half)
//
// which rotates pairs (d, d + head_dim/2) for d in [0, rope_angles) and
// leaves the zero-tail pairs (d, d + head_dim/2) for d in
// [rope_angles, head_dim/2) UNROTATED (cos=1, sin=0).
//
// This is NOT expressible with `rope_partial_halfsplit_f32` (that kernel
// pairs (i, i + n_rot/2) within the rotated prefix, i.e. it rotates the
// first n_rot dims against each other — wrong pairing AND 2x the
// frequency: base^(-2i/n_rot) = base^(-4i/dim) vs HF's base^(-2i/dim)).
// The `sliding_attention` layers use partial_rotary_factor = 1.0, where
// the zero tail is empty and the halfsplit kernel with n_rot = head_dim
// already matches HF — they keep using it.
extern "C" __global__ void rope_zero_tail_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ pos_buf,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int rope_angles,
    float freq_base
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    if (i >= rope_angles) return;

    int pos = *pos_buf;
    // HF: inv_freq[i] = 1 / base^(2i / head_dim) — the FULL head_dim in the
    // denominator, not the rotated-pair count.
    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    int shift = half; // rotate_half pairing: (d, d + head_dim/2)
    for (int h = 0; h < n_heads_q; h++) {
        int base = h * head_dim;
        float q0 = q[base + i];
        float q1 = q[base + i + shift];
        q[base + i]        = q0 * cos_a - q1 * sin_a;
        q[base + i + shift] = q0 * sin_a + q1 * cos_a;
    }

    for (int h = 0; h < n_heads_k; h++) {
        int base = h * head_dim;
        float k0 = k[base + i];
        float k1 = k[base + i + shift];
        k[base + i]        = k0 * cos_a - k1 * sin_a;
        k[base + i + shift] = k0 * sin_a + k1 * cos_a;
    }
}
