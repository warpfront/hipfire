// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// VAE encoder 3x3 STRIDE-2 convolution, f32, channel-major. This is the
// diffusers `Downsample2D` used by every `down_blocks.{i}` downsampler: an
// asymmetric zero pad of (left 0, right 1, top 0, bottom 1) followed by a
// 3x3 stride-2 conv with no further padding, which halves H and W.
//
// `x` is `[c_in][h][wdt]`, `w` is `[c_out][c_in*9]` (the CPU `vae.rs` loader
// flattens `[c_out][c_in][3][3]`), `bias` is `[c_out]`, `y` is
// `[c_out][h/2][wdt/2]`. Because the pad is only on the bottom/right, the
// input tap is `2*oy + ky` (NOT `2*oy + ky - 1`), and the only taps that can
// fall outside the image are the ones past the last row/column — hence the
// single-sided `>= h` / `>= wdt` guards.
//
// One thread per output element, correctness-first, no WMMA — the encoder
// runs once per reference image, so it never became the bottleneck the
// decoder's GEMM route was written for. The summation order (ci outer, then
// ky, kx) matches the CPU `nn::conv2d_strided`.

#include <hip/hip_runtime.h>

extern "C" __global__ void vae_conv3x3_s2_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int c_in,
    int c_out,
    int h,
    int wdt
) {
    const int oh = h / 2, ow = wdt / 2;
    const long total = (long)c_out * oh * ow;
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const int co = (int)(t / ((long)oh * ow));
    const int rem = (int)(t % ((long)oh * ow));
    const int oy = rem / ow, ox = rem % ow;
    float acc = bias[co];
    const float* wco = w + (long)co * c_in * 9;
    for (int ci = 0; ci < c_in; ++ci) {
        const float* xci = x + (long)ci * h * wdt;
        const float* wci = wco + (long)ci * 9;
        for (int ky = 0; ky < 3; ++ky) {
            int iy = 2 * oy + ky;            // pad top 0, bottom 1
            if (iy >= h) continue;
            for (int kx = 0; kx < 3; ++kx) {
                int ix = 2 * ox + kx;        // pad left 0, right 1
                if (ix >= wdt) continue;
                acc += wci[ky * 3 + kx] * xci[iy * wdt + ix];
            }
        }
    }
    y[t] = acc;
}
