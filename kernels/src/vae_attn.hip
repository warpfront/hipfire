// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// FLUX VAE mid-block attention, f32, mirroring the CPU `mid_attn_forward`
// after its channel-major -> position-major transpose: q/k/v/ctx are
// position-major `[n][c]` (n = h*w, single head, head_dim = c).
//
// - vae_attn_scores_f32: s[qp][kp] = scale * dot(q[qp], k[kp])
// - vae_attn_ctx_f32:    ctx[qp][ch] = sum_kp probs[qp][kp] * v[kp][ch]
// - vae_attn_residual_f32: y[ch][p] = x[ch][p] + out[p][ch] — fuses the
//   CPU's (HW, C) -> (C, HW) transpose with the residual add back into
//   channel-major layout.

#include <hip/hip_runtime.h>

extern "C" __global__ void vae_attn_scores_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    float* __restrict__ s,
    int n,
    int c,
    float scale
) {
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)n * n;
    if (t >= total) return;
    int qp = (int)(t / n);
    int kp = (int)(t % n);
    const float* qr = q + (long)qp * c;
    const float* kr = k + (long)kp * c;
    float acc = 0.0f;
    for (int i = 0; i < c; ++i) {
        acc += qr[i] * kr[i];
    }
    s[t] = acc * scale;
}

extern "C" __global__ void vae_attn_ctx_f32(
    const float* __restrict__ probs,
    const float* __restrict__ v,
    float* __restrict__ ctx,
    int n,
    int c
) {
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)n * c;
    if (t >= total) return;
    int qp = (int)(t / c);
    int ch = (int)(t % c);
    const float* pr = probs + (long)qp * n;
    float acc = 0.0f;
    for (int kp = 0; kp < n; ++kp) {
        acc += pr[kp] * v[(long)kp * c + ch];
    }
    ctx[t] = acc;
}

extern "C" __global__ void vae_attn_residual_f32(
    const float* __restrict__ x,
    const float* __restrict__ out,
    float* __restrict__ y,
    int c,
    int n
) {
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)c * n;
    if (t >= total) return;
    int ch = (int)(t / n);
    int p = (int)(t % n);
    y[t] = x[t] + out[(long)p * c + ch];
}
