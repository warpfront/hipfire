// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// FLUX VAE decoder 3x3 stride-1 pad-1 convolution, f32, channel-major
// `[c_in][h][w]` in, `[c_out][h][w]` out, weight `[c_out][c_in][3][3]`
// (row-major: the CPU `vae.rs` loader flattens it to `[c_out][c_in*9]`).
// One thread per output element; correctness-first, no WMMA yet. The
// summation order (ci outer, ky, kx inner) matches the CPU `nn::conv2d`.

#include <hip/hip_runtime.h>

extern "C" __global__ void vae_conv3x3_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int c_in,
    int c_out,
    int h,
    int wdt
) {
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)c_out * h * wdt;
    if (t >= total) return;
    int hw = h * wdt;
    int co = (int)(t / hw);
    int p = (int)(t % hw);
    int oy = p / wdt;
    int ox = p % wdt;
    float acc = bias[co];
    const float* wco = w + (long)co * c_in * 9;
    for (int ci = 0; ci < c_in; ++ci) {
        const float* xci = x + (long)ci * hw;
        const float* wci = wco + (long)ci * 9;
        for (int ky = 0; ky < 3; ++ky) {
            int iy = oy + ky - 1;
            if (iy < 0 || iy >= h) continue;
            for (int kx = 0; kx < 3; ++kx) {
                int ix = ox + kx - 1;
                if (ix < 0 || ix >= wdt) continue;
                acc += wci[ky * 3 + kx] * xci[iy * wdt + ix];
            }
        }
    }
    y[t] = acc;
}
