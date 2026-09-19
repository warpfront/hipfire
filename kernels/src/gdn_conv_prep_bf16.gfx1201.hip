// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Fixed Qwen3.5-27B GDN preparation geometry.
// q/k: [T,16,128], v: [T,48,128], x: [T,10240].
// Grid (ceil(T/64),10), block 256.  Each wave owns four channels from one
// 128-wide Q/K/V head and keeps its width-four convolution window in registers.
#define GDN_R4D_C 64
#define GDN_R4D_HD 128
#define GDN_R4D_HK 16
#define GDN_R4D_HV 48
#define GDN_R4D_UNITS 80
#define GDN_R4D_XDIM 10240

__device__ __forceinline__ float gdn_r4d_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return logf(1.0f + expf(x));
}

__device__ __forceinline__ unsigned short gdn_r4d_bf16(float x) {
    const __bf16 b = (__bf16)x;
    return __builtin_bit_cast(unsigned short, b);
}

extern "C" __launch_bounds__(256)
__global__ void gdn_conv_prep_bf16_gfx1201(
    const float* __restrict__ x,
    const float* __restrict__ conv_w,
    float* __restrict__ conv_state,
    float* __restrict__ a,
    float* __restrict__ b,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    unsigned short* __restrict__ q,
    unsigned short* __restrict__ k,
    unsigned short* __restrict__ v,
    int T,
    float q_scale,
    float eps) {
    if (T <= 0) return;

    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int chunk0 = (int)blockIdx.x * GDN_R4D_C;
    int rows = T - chunk0;
    if (rows <= 0) return;
    if (rows > GDN_R4D_C) rows = GDN_R4D_C;

    // One block owns the gates.  Each wave scans one head at a time; two
    // adjacent rows per lane make the inclusive C64 prefix wave-local.
    if (blockIdx.y == 0) {
        for (int h = wave; h < GDN_R4D_HV; h += 8) {
            const int r0 = lane * 2;
            const int r1 = r0 + 1;
            float g0 = 0.0f;
            float g1 = 0.0f;
            if (r0 < rows) {
                const long long off = (long long)(chunk0 + r0) * GDN_R4D_HV + h;
                g0 = -expf(a_log[h]) * gdn_r4d_softplus(a[off] + dt_bias[h]);
                b[off] = 1.0f / (1.0f + expf(-b[off]));
            }
            if (r1 < rows) {
                const long long off = (long long)(chunk0 + r1) * GDN_R4D_HV + h;
                g1 = -expf(a_log[h]) * gdn_r4d_softplus(a[off] + dt_bias[h]);
                b[off] = 1.0f / (1.0f + expf(-b[off]));
            }
            float pair = g0 + g1;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const float other = __shfl_up(pair, d, 32);
                if (lane >= d) pair += other;
            }
            const float before = pair - g0 - g1;
            if (r0 < rows)
                a[(long long)(chunk0 + r0) * GDN_R4D_HV + h] = before + g0;
            if (r1 < rows)
                a[(long long)(chunk0 + r1) * GDN_R4D_HV + h] = before + g0 + g1;
        }
    }

    const int unit = (int)blockIdx.y * 8 + wave;
    if (unit >= GDN_R4D_UNITS) return;
    const int dbase = unit * GDN_R4D_HD + lane * 4;

    float w[4][4];
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            w[j][c] = conv_w[(dbase + c) * 4 + j];

    float win[3][4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        if (chunk0 == 0) {
            win[0][c] = conv_state[(dbase + c) * 3 + 2];
            win[1][c] = conv_state[(dbase + c) * 3 + 1];
            win[2][c] = conv_state[(dbase + c) * 3];
        } else {
            win[0][c] = x[(long long)(chunk0 - 3) * GDN_R4D_XDIM + dbase + c];
            win[1][c] = x[(long long)(chunk0 - 2) * GDN_R4D_XDIM + dbase + c];
            win[2][c] = x[(long long)(chunk0 - 1) * GDN_R4D_XDIM + dbase + c];
        }
    }

    const int region = unit < GDN_R4D_HK ? 0 : (unit < 2 * GDN_R4D_HK ? 1 : 2);
    const int head = region == 0 ? unit : (region == 1 ? unit - GDN_R4D_HK
                                                       : unit - 2 * GDN_R4D_HK);

    for (int r = 0; r < rows; ++r) {
        float out4[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float cur = x[(long long)(chunk0 + r) * GDN_R4D_XDIM + dbase + c];
            float y = fmaf(w[3][c], cur, w[2][c] * win[2][c]);
            y = fmaf(w[1][c], win[1][c], y);
            y = fmaf(w[0][c], win[0][c], y);
            out4[c] = y / (1.0f + expf(-y));
            win[0][c] = win[1][c];
            win[1][c] = win[2][c];
            win[2][c] = cur;
        }

        if (region < 2) {
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < 4; ++c) sum += out4[c] * out4[c];
#pragma unroll
            for (int d = 16; d > 0; d >>= 1) sum += __shfl_xor(sum, d, 32);
            const float inv = rsqrtf(sum + eps);
            unsigned short* dst = (region == 0 ? q : k)
                + ((long long)(chunk0 + r) * GDN_R4D_HK + head) * GDN_R4D_HD
                + lane * 4;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                float z = out4[c] * inv;
                if (region == 0) z *= q_scale;
                dst[c] = gdn_r4d_bf16(z);
            }
        } else {
            unsigned short* dst = v
                + ((long long)(chunk0 + r) * GDN_R4D_HV + head) * GDN_R4D_HD
                + lane * 4;
#pragma unroll
            for (int c = 0; c < 4; ++c) dst[c] = gdn_r4d_bf16(out4[c]);
        }
    }

    // The first C64 owner alone advances this channel's ring, after consuming
    // the old values.  Later chunks use parent x as halo and never touch it.
    if (chunk0 == 0) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float old0 = conv_state[(dbase + c) * 3];
            const float old1 = conv_state[(dbase + c) * 3 + 1];
            conv_state[(dbase + c) * 3] = x[(long long)(T - 1) * GDN_R4D_XDIM + dbase + c];
            conv_state[(dbase + c) * 3 + 1] = T >= 2
                ? x[(long long)(T - 2) * GDN_R4D_XDIM + dbase + c] : old0;
            conv_state[(dbase + c) * 3 + 2] = T >= 3
                ? x[(long long)(T - 3) * GDN_R4D_XDIM + dbase + c]
                : (T == 2 ? old0 : old1);
        }
    }
}
