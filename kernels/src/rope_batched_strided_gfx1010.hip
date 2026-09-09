// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#if defined(__gfx1010__)

// Strided single-buffer RoPE over packed QKV rows.
// Each row is `stride` floats; Q lives at qoff and K at koff.
// Grid: [ceil((hd/2)/block), bs, 1], block = min(256, hd/2).
// V and any padding lanes outside Q/K regions are untouched.
extern "C" __global__ void rope_batched_strided_f32_gfx1010(
    float* __restrict__ qkv,
    const int* __restrict__ positions,  // [bs]
    int nhq,
    int nhk,
    int hd,
    float fb,
    int bs,
    int qoff,
    int koff,
    int stride
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // pair index
    int b = blockIdx.y;                             // batch row
    int half = hd / 2;
    if (i >= half || b >= bs) return;

    int pos = positions[b];
    float freq = 1.0f / powf(fb, (float)(2 * i) / (float)hd);
    float val = (float)pos * freq;
    float cos_val = cosf(val);
    float sin_val = sinf(val);

    // Wide row arithmetic — avoid i32 overflow on large batch*stride.
    long long row = (long long)b * (long long)stride;

    for (int h = 0; h < nhq; h++) {
        long long base = row + (long long)qoff + (long long)h * (long long)hd;
        float q0 = qkv[base + i];
        float q1 = qkv[base + i + half];
        qkv[base + i]        = q0 * cos_val - q1 * sin_val;
        qkv[base + i + half] = q0 * sin_val + q1 * cos_val;
    }

    for (int h = 0; h < nhk; h++) {
        long long base = row + (long long)koff + (long long)h * (long long)hd;
        float k0 = qkv[base + i];
        float k1 = qkv[base + i + half];
        qkv[base + i]        = k0 * cos_val - k1 * sin_val;
        qkv[base + i + half] = k0 * sin_val + k1 * cos_val;
    }
}

#endif // defined(__gfx1010__)
