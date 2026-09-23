// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1201 row-wide FWHT producer with optional sigmoid gate and AWQ divide.
// It preserves the incumbent 32-lane group arithmetic and emits the exact
// scale_mode=1 MQ4v2 FP8 planes after writing the rotated F32 row.

#ifndef HIPFIRE_ROTATE_FP8_KERNEL
#define HIPFIRE_ROTATE_FP8_KERNEL mq_rotate_x_mq4v2_fp8_gfx12
#endif

extern "C" __launch_bounds__(256, 1) __global__
void HIPFIRE_ROTATE_FP8_KERNEL(
    const float* __restrict__ x_in,
#ifdef HIPFIRE_ROTATE_FP8_SIGMOID_GATE
    const float* __restrict__ gate,
#endif
#ifdef HIPFIRE_ROTATE_FP8_AWQ
    const float* __restrict__ awq_scale,
#endif
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_out,
    unsigned char* __restrict__ x_fp8,
    float* __restrict__ fp8_half_sums,
    float* __restrict__ fp8_row_scales,
    int K,
    int N)
{
    const int token = (int)blockIdx.x;
    if (token >= N || K <= 0) return;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int groups_total = K >> 8;
    const long long row_off = (long long)token * K;
    const float* x_in_b = x_in + row_off;
#ifdef HIPFIRE_ROTATE_FP8_SIGMOID_GATE
    const float* gate_b = gate + row_off;
#endif
    float* x_out_b = x_out + row_off;
    float row_amax = 0.0f;

    for (int group = wave; group < groups_total; group += 8) {
        const int d0 = lane * 8;
        const int base = group * 256 + d0;
        float x0 = x_in_b[base    ];
        float x1 = x_in_b[base + 1];
        float x2 = x_in_b[base + 2];
        float x3 = x_in_b[base + 3];
        float x4 = x_in_b[base + 4];
        float x5 = x_in_b[base + 5];
        float x6 = x_in_b[base + 6];
        float x7 = x_in_b[base + 7];
#ifdef HIPFIRE_ROTATE_FP8_SIGMOID_GATE
        const float sg0 = 1.0f / (1.0f + expf(-gate_b[base    ]));
        const float sg1 = 1.0f / (1.0f + expf(-gate_b[base + 1]));
        const float sg2 = 1.0f / (1.0f + expf(-gate_b[base + 2]));
        const float sg3 = 1.0f / (1.0f + expf(-gate_b[base + 3]));
        const float sg4 = 1.0f / (1.0f + expf(-gate_b[base + 4]));
        const float sg5 = 1.0f / (1.0f + expf(-gate_b[base + 5]));
        const float sg6 = 1.0f / (1.0f + expf(-gate_b[base + 6]));
        const float sg7 = 1.0f / (1.0f + expf(-gate_b[base + 7]));
        x0 = x0 * sg0;
        x1 = x1 * sg1;
        x2 = x2 * sg2;
        x3 = x3 * sg3;
        x4 = x4 * sg4;
        x5 = x5 * sg5;
        x6 = x6 * sg6;
        x7 = x7 * sg7;
#endif
#ifdef HIPFIRE_ROTATE_FP8_AWQ
        float v0 = (x0 / awq_scale[base    ]) * signs1[d0    ];
        float v1 = (x1 / awq_scale[base + 1]) * signs1[d0 + 1];
        float v2 = (x2 / awq_scale[base + 2]) * signs1[d0 + 2];
        float v3 = (x3 / awq_scale[base + 3]) * signs1[d0 + 3];
        float v4 = (x4 / awq_scale[base + 4]) * signs1[d0 + 4];
        float v5 = (x5 / awq_scale[base + 5]) * signs1[d0 + 5];
        float v6 = (x6 / awq_scale[base + 6]) * signs1[d0 + 6];
        float v7 = (x7 / awq_scale[base + 7]) * signs1[d0 + 7];
#else
        float v0 = x0 * signs1[d0    ];
        float v1 = x1 * signs1[d0 + 1];
        float v2 = x2 * signs1[d0 + 2];
        float v3 = x3 * signs1[d0 + 3];
        float v4 = x4 * signs1[d0 + 4];
        float v5 = x5 * signs1[d0 + 5];
        float v6 = x6 * signs1[d0 + 6];
        float v7 = x7 * signs1[d0 + 7];
#endif

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

        const float s = 0.0625f;
        float f0 = v0 * s * signs2[d0    ];
        float f1 = v1 * s * signs2[d0 + 1];
        float f2 = v2 * s * signs2[d0 + 2];
        float f3 = v3 * s * signs2[d0 + 3];
        float f4 = v4 * s * signs2[d0 + 4];
        float f5 = v5 * s * signs2[d0 + 5];
        float f6 = v6 * s * signs2[d0 + 6];
        float f7 = v7 * s * signs2[d0 + 7];
        x_out_b[base    ] = f0;
        x_out_b[base + 1] = f1;
        x_out_b[base + 2] = f2;
        x_out_b[base + 3] = f3;
        x_out_b[base + 4] = f4;
        x_out_b[base + 5] = f5;
        x_out_b[base + 6] = f6;
        x_out_b[base + 7] = f7;
        row_amax = fmaxf(row_amax, fabsf(f0));
        row_amax = fmaxf(row_amax, fabsf(f1));
        row_amax = fmaxf(row_amax, fabsf(f2));
        row_amax = fmaxf(row_amax, fabsf(f3));
        row_amax = fmaxf(row_amax, fabsf(f4));
        row_amax = fmaxf(row_amax, fabsf(f5));
        row_amax = fmaxf(row_amax, fabsf(f6));
        row_amax = fmaxf(row_amax, fabsf(f7));
    }

    __shared__ float smem8[8];
    const int gpr = K >> 8;
    fp8stream_pack_row_from_f32(
        x_out_b,
        x_fp8 + row_off,
        fp8_half_sums + (long long)token * gpr * 2,
        fp8_row_scales + token,
        row_amax,
        smem8,
        K,
        tid,
        lane,
        wave);
}
