// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1201 gated RMSNorm + AWQ/FWHT + MQ4v2 FP8 stream producer.
// One row workgroup first reproduces gated_norm_f32 head-by-head, stages the
// exact rotate input in LDS, then runs the incumbent wave-local FWHT and the
// standalone packer's scale_mode=1 recipe.

#ifndef HIPFIRE_GATED_NORM_FP8_KERNEL
#define HIPFIRE_GATED_NORM_FP8_KERNEL gated_norm_mq_rotate_mq4v2_fp8_gfx12
#endif

extern "C" __launch_bounds__(256, 1) __global__
void HIPFIRE_GATED_NORM_FP8_KERNEL(
    const float* __restrict__ x,
    const float* __restrict__ z,
    const float* __restrict__ weight,
#ifdef HIPFIRE_GATED_NORM_FP8_AWQ
    const float* __restrict__ awq_scale,
#endif
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
    unsigned char* __restrict__ x_fp8,
    float* __restrict__ fp8_half_sums,
    float* __restrict__ fp8_row_scales,
    int n_heads,
    int head_dim,
    float eps,
    int K,
    int N)
{
    const int token = (int)blockIdx.x;
    if (token >= N || head_dim != 128 || K <= 0) return;

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const long long row_off = (long long)token * K;
    const float* x_b = x + row_off;
    const float* z_b = z + row_off;
#ifndef HIPFIRE_FP8_PROD_INREG
    float* x_rot_b = x_rot + row_off;
#else
    if (K > 18432 || (K & 255)) return;
    float held[9][8];
#endif
    extern __shared__ float smem[];
    float* normalized = smem;
    float* pack_smem = smem + K;

    // Exact gated_norm_f32 lane ownership, accumulation, shuffle tree and
    // expression order.  Eight waves cover the row's heads in strides of 8.
    for (int head = wave; head < n_heads; head += 8) {
        const long long head_off = (long long)head * head_dim;
        const float* xh = x_b + head_off;
        const float* zh = z_b + head_off;
        float* oh = normalized + head_off;
        float sq_sum = 0.0f;
        for (int i = lane; i < head_dim; i += 32)
            sq_sum += xh[i] * xh[i];
        for (int o = 16; o > 0; o >>= 1)
            sq_sum += __shfl_xor(sq_sum, o);
        const float inv_rms = rsqrtf(sq_sum / (float)head_dim + eps);
        for (int i = lane; i < head_dim; i += 32) {
            const float normed = xh[i] * inv_rms * weight[i];
            const float z_val = zh[i];
            const float silu_z = z_val / (1.0f + expf(-z_val));
#ifdef HIPFIRE_GATED_NORM_FP8_AWQ
            oh[i] = normed * silu_z / awq_scale[head_off + i];
#else
            oh[i] = normed * silu_z;
#endif
        }
    }
    __syncthreads();

    float row_amax = 0.0f;
    const int groups_total = K >> 8;
#ifdef HIPFIRE_FP8_PROD_INREG
    #pragma unroll
    for (int gi = 0; gi < 9; ++gi) {
        const int group = wave + gi * 8;
        if (group >= groups_total) continue;
#else
    for (int group = wave; group < groups_total; group += 8) {
#endif
        const int d0 = lane * 8;
        const int base = group * 256 + d0;
        const float4 x0 = *reinterpret_cast<const float4*>(normalized + base);
        const float4 x1 = *reinterpret_cast<const float4*>(normalized + base + 4);
        const float4 s10 = *reinterpret_cast<const float4*>(signs1 + d0);
        const float4 s11 = *reinterpret_cast<const float4*>(signs1 + d0 + 4);
        const float4 s20 = *reinterpret_cast<const float4*>(signs2 + d0);
        const float4 s21 = *reinterpret_cast<const float4*>(signs2 + d0 + 4);

        float v0 = x0.x * s10.x;
        float v1 = x0.y * s10.y;
        float v2 = x0.z * s10.z;
        float v3 = x0.w * s10.w;
        float v4 = x1.x * s11.x;
        float v5 = x1.y * s11.y;
        float v6 = x1.z * s11.z;
        float v7 = x1.w * s11.w;

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
            if (lane & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
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

        const float scale = 0.0625f;
        float f0 = v0 * scale * s20.x;
        float f1 = v1 * scale * s20.y;
        float f2 = v2 * scale * s20.z;
        float f3 = v3 * scale * s20.w;
        float f4 = v4 * scale * s21.x;
        float f5 = v5 * scale * s21.y;
        float f6 = v6 * scale * s21.z;
        float f7 = v7 * scale * s21.w;
#ifdef HIPFIRE_FP8_PROD_INREG
        held[gi][0] = f0; held[gi][1] = f1; held[gi][2] = f2; held[gi][3] = f3;
        held[gi][4] = f4; held[gi][5] = f5; held[gi][6] = f6; held[gi][7] = f7;
#else
        x_rot_b[base    ] = f0;
        x_rot_b[base + 1] = f1;
        x_rot_b[base + 2] = f2;
        x_rot_b[base + 3] = f3;
        x_rot_b[base + 4] = f4;
        x_rot_b[base + 5] = f5;
        x_rot_b[base + 6] = f6;
        x_rot_b[base + 7] = f7;
#endif
        row_amax = fmaxf(row_amax, fabsf(f0));
        row_amax = fmaxf(row_amax, fabsf(f1));
        row_amax = fmaxf(row_amax, fabsf(f2));
        row_amax = fmaxf(row_amax, fabsf(f3));
        row_amax = fmaxf(row_amax, fabsf(f4));
        row_amax = fmaxf(row_amax, fabsf(f5));
        row_amax = fmaxf(row_amax, fabsf(f6));
        row_amax = fmaxf(row_amax, fabsf(f7));
    }

    const int gpr = K >> 8;
#ifdef HIPFIRE_FP8_PROD_INREG
    fp8stream_pack_row_inreg(
        held, x_fp8 + row_off,
        fp8_half_sums + (long long)token * (K >> 7),
        fp8_row_scales + token, row_amax, pack_smem, K, lane, wave, wave, 8);
#else
    fp8stream_pack_row_from_f32(
        x_rot_b, x_fp8 + row_off,
        fp8_half_sums + (long long)token * (K >> 7),
        fp8_row_scales + token, row_amax, pack_smem, K, tid, lane, wave);
#endif
}
