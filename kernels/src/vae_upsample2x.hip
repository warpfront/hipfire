// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// FLUX VAE decoder nearest-neighbour 2x upsample (the first half of taming
// `Upsample`, before the 3x3 conv): `[c][h][w]` -> `[c][2h][2w]`, each input
// pixel replicated into its 2x2 output cell. Matches `nn::upsample_nearest2x`
// element-for-element (no interpolation arithmetic, so exact). One thread per
// input element writing four outputs.

#include <hip/hip_runtime.h>

extern "C" __global__ void vae_upsample2x_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int c,
    int h,
    int w
) {
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)c * h * w;
    if (t >= total) return;
    int hw = h * w;
    int ch = (int)(t / hw);
    int p = (int)(t % hw);
    int iy = p / w;
    int ix = p % w;
    int ow = 2 * w;
    float v = x[t];
    long base = (long)ch * 4 * hw + (long)(2 * iy) * ow + 2 * ix;
    y[base] = v;
    y[base + 1] = v;
    y[base + ow] = v;
    y[base + ow + 1] = v;
}
