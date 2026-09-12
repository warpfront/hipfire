// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// FLUX VAE decoder 1x1 convolution (per-pixel linear), f32. Weight
// `[c_out][c_in]`, bias `[c_out]`. Both the input and the output layouts are
// selectable so the same kernel serves every 1x1 projection in the decoder:
// `in_pos_major` 0 reads channel-major `[c_in][n]` (groupnorm outputs,
// residual inputs), 1 reads position-major `[n][c_in]` (the mid-attn ctx);
// `out_pos_major` 0 writes channel-major `[c_out][n]` (the residual
// `nin_shortcut`/`conv_shortcut`), 1 writes position-major `[n][c_out]`
// (the mid-attn q/k/v/proj_out, which feed position-major dot products
// exactly like the CPU `mid_attn_forward` transpose does). n = h*w.

#include <hip/hip_runtime.h>

extern "C" __global__ void vae_conv1x1_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int c_in,
    int c_out,
    int n,
    int in_pos_major,
    int out_pos_major
) {
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)c_out * n;
    if (t >= total) return;
    int co = (int)(t / n);
    int p = (int)(t % n);
    float acc = bias[co];
    const float* wco = w + (long)co * c_in;
    if (in_pos_major) {
        const float* xr = x + (long)p * c_in;
        for (int ci = 0; ci < c_in; ++ci) {
            acc += wco[ci] * xr[ci];
        }
    } else {
        for (int ci = 0; ci < c_in; ++ci) {
            acc += wco[ci] * x[(long)ci * n + p];
        }
    }
    y[out_pos_major ? ((long)p * c_out + co) : t] = acc;
}
