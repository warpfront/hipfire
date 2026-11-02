// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// FLUX VAE decoder GroupNorm, f32, channel-major `[c][h][w]`. Matches the
// CPU `nn::groupnorm`: mean/var are taken over the WHOLE group
// (ch_per_group channels x h*w positions), eps is added to the variance
// before the inverse sqrt, and gamma/beta are per-channel. One block per
// group with a shared-memory reduction (softmax.hip pattern); the three
// passes (mean, variance, normalize) walk the group strided by blockDim.x
// exactly like the softmax row loop.
//
// Two entries over one body. Every VAE GroupNorm except the mid-attention
// one is immediately followed by SiLU, and at 1024x1024 that separate
// elementwise pass costs another full read+write of a 512 MB tensor. The
// `_silu` entry folds `v / (1 + exp(-v))` into the normalize pass — the
// same expression `silu.hip` computes, applied to the same f32 value, so
// the fused result is bit-identical to the two-kernel pair.

#include <hip/hip_runtime.h>

__device__ __forceinline__ void vae_groupnorm_body(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int c,
    int hw,
    int groups,
    float eps,
    bool fuse_silu
) {
    extern __shared__ float sdata[];
    int g = blockIdx.x;
    int cpg = c / groups;
    long group_elems = (long)cpg * hw;
    const float* xg = x + (long)g * group_elems;
    float* yg = y + (long)g * group_elems;

    float sum = 0.0f;
    for (long i = threadIdx.x; i < group_elems; i += blockDim.x) {
        sum += xg[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)group_elems;
    __syncthreads();

    float vsum = 0.0f;
    for (long i = threadIdx.x; i < group_elems; i += blockDim.x) {
        float d = xg[i] - mean;
        vsum += d * d;
    }
    sdata[threadIdx.x] = vsum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float var = sdata[0] / (float)group_elems;
    __syncthreads();

    float inv = rsqrtf(var + eps);
    for (long i = threadIdx.x; i < group_elems; i += blockDim.x) {
        int ch = g * cpg + (int)(i / hw);
        float v = (xg[i] - mean) * inv * gamma[ch] + beta[ch];
        yg[i] = fuse_silu ? (v / (1.0f + expf(-v))) : v;
    }
}

extern "C" __global__ void vae_groupnorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int c,
    int hw,
    int groups,
    float eps
) {
    vae_groupnorm_body(x, gamma, beta, y, c, hw, groups, eps, false);
}

// GroupNorm with the SiLU that always follows it folded in.
extern "C" __global__ void vae_groupnorm_silu_f32(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int c,
    int hw,
    int groups,
    float eps
) {
    vae_groupnorm_body(x, gamma, beta, y, c, hw, groups, eps, true);
}
