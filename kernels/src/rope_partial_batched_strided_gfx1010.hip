// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#if defined(__gfx1010__)

// Strided single-buffer partial RoPE over packed QKV rows.
// Supports both half-split (interleaved==0) and legacy interleaved (interleaved!=0)
// pair layouts in one kernel. Pair p rotates:
//   halfsplit:   (p, p + nrot/2)
//   interleaved: (2p, 2p+1)
// Both use freq = 1/powf(fb, (2p)/nrot) and angle = (positions[b]+pos_offset)*freq.
// Dims [nrot, hd) and the V region are pass-through.
// Grid: [ceil((nrot/2)/block), bs, 1], block = min(32, nrot/2).
extern "C" __global__ void rope_partial_batched_strided_f32_gfx1010(
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

#endif // defined(__gfx1010__)
