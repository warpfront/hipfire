// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

#include <hip/hip_runtime.h>

__launch_bounds__(256, 1)
extern "C" __global__ void fused_rmsnorm_mq_rotate(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
    int K,
    float eps
) {
    extern __shared__ float smem[];
    float* reduce = smem;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = block_size >> 5;

    const long long batch_off = (long long)blockIdx.x * K;
    const float* x_b = x + batch_off;
    float* x_rot_b = x_rot + batch_off;

    float local_sum = 0.0f;
    int i = tid;
    for (; i + block_size * 3 < K; i += block_size * 4) {
        float v0 = x_b[i];
        float v1 = x_b[i + block_size];
        float v2 = x_b[i + block_size * 2];
        float v3 = x_b[i + block_size * 3];
        local_sum += v0 * v0;
        local_sum += v1 * v1;
        local_sum += v2 * v2;
        local_sum += v3 * v3;
    }
    for (; i < K; i += block_size) {
        float v = x_b[i];
        local_sum += v * v;
    }

    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down(local_sum, off);
    if (lane_id == 0) reduce[warp_id] = local_sum;
    __syncthreads();

    float block_sum = (tid < num_warps) ? reduce[tid] : 0.0f;
    if (warp_id == 0) {
        for (int off = 16; off > 0; off >>= 1)
            block_sum += __shfl_down(block_sum, off);
        if (tid == 0) reduce[0] = block_sum;
    }
    __syncthreads();
    const float rms = rsqrtf(reduce[0] / (float)K + eps);

    const int groups_total = K / 256;
    const int groups_per_warp = (groups_total + num_warps - 1) / num_warps;
    const int warp_group_start = warp_id * groups_per_warp;

    for (int gi = 0; gi < groups_per_warp; gi++) {
        const int group = warp_group_start + gi;
        if (group >= groups_total) break;

        const int d0 = lane_id * 8;
        const int base = group * 256 + d0;

        float v0 = (x_b[base]     * weight[base]     * rms) * signs1[d0];
        float v1 = (x_b[base + 1] * weight[base + 1] * rms) * signs1[d0 + 1];
        float v2 = (x_b[base + 2] * weight[base + 2] * rms) * signs1[d0 + 2];
        float v3 = (x_b[base + 3] * weight[base + 3] * rms) * signs1[d0 + 3];
        float v4 = (x_b[base + 4] * weight[base + 4] * rms) * signs1[d0 + 4];
        float v5 = (x_b[base + 5] * weight[base + 5] * rms) * signs1[d0 + 5];
        float v6 = (x_b[base + 6] * weight[base + 6] * rms) * signs1[d0 + 6];
        float v7 = (x_b[base + 7] * weight[base + 7] * rms) * signs1[d0 + 7];

        float t;
        t = v0; v0 = v0 + v1; v1 = t - v1;
        t = v2; v2 = v2 + v3; v3 = t - v3;
        t = v4; v4 = v4 + v5; v5 = t - v5;
        t = v6; v6 = v6 + v7; v7 = t - v7;

        t = v0; v0 = v0 + v2; v2 = t - v2;
        t = v1; v1 = v1 + v3; v3 = t - v3;
        t = v4; v4 = v4 + v6; v6 = t - v6;
        t = v5; v5 = v5 + v7; v7 = t - v7;

        t = v0; v0 = v0 + v4; v4 = t - v4;
        t = v1; v1 = v1 + v5; v5 = t - v5;
        t = v2; v2 = v2 + v6; v6 = t - v6;
        t = v3; v3 = v3 + v7; v7 = t - v7;

        #define HBFLY(v, pat, str) do { \
            float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
            if (lane_id & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
        } while (0)
        #define HBFLY8(pat, str) \
            HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
            HBFLY(v4,pat,str); HBFLY(v5,pat,str); HBFLY(v6,pat,str); HBFLY(v7,pat,str)

        HBFLY8(0x041F, 1);
        HBFLY8(0x081F, 2);
        HBFLY8(0x101F, 4);
        HBFLY8(0x201F, 8);
        HBFLY8(0x401F, 16);
        #undef HBFLY8
        #undef HBFLY

        const float s = 0.0625f;
        x_rot_b[base]     = v0 * s * signs2[d0];
        x_rot_b[base + 1] = v1 * s * signs2[d0 + 1];
        x_rot_b[base + 2] = v2 * s * signs2[d0 + 2];
        x_rot_b[base + 3] = v3 * s * signs2[d0 + 3];
        x_rot_b[base + 4] = v4 * s * signs2[d0 + 4];
        x_rot_b[base + 5] = v5 * s * signs2[d0 + 5];
        x_rot_b[base + 6] = v6 * s * signs2[d0 + 6];
        x_rot_b[base + 7] = v7 * s * signs2[d0 + 7];
    }
}
