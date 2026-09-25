// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1201 SwiGLU + AWQ/FWHT + MQ4v2 FP8 stream producer.
//
// One 256-thread workgroup owns a token row.  Each wave evaluates the same
// 256-value groups as fused_silu_mul_mq_rotate[_awq], stores the exact F32
// results, then the whole workgroup emits the scale_mode=1 E4M3 bytes,
// decoded-code half sums, and row scale consumed by the prepared FP8 GEMM.
// The producer arithmetic below is statement-identical to the incumbent
// 32-thread producer; changing the row geometry does not change any wave's
// lane ownership or FWHT reduction tree.

#ifndef HIPFIRE_SILU_FP8_KERNEL
#define HIPFIRE_SILU_FP8_KERNEL fused_silu_mul_mq_rotate_mq4v2_fp8_gfx12
#endif

extern "C" __launch_bounds__(256, 1) __global__
void HIPFIRE_SILU_FP8_KERNEL(
    const float* __restrict__ gate,
    const float* __restrict__ up,
#ifdef HIPFIRE_SILU_FP8_AWQ
    const float* __restrict__ awq_scale,
#endif
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
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
    const float* gate_b = gate + row_off;
#ifndef HIPFIRE_SILU_FP8_HIN
    const float* up_b = up + row_off;
#endif
#ifndef HIPFIRE_FP8_PROD_INREG
    float* x_rot_b = x_rot + row_off;
#else
    if (K > 18432 || (K & 255)) return;
    float held[9][8];
#endif
    float row_amax = 0.0f;

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

        #ifdef HIPFIRE_SILU_FP8_HIN
        #define SILU_MUL(g, u) (g)
        #else
        #define SILU_MUL(g, u) ((g) / (1.0f + expf(-(g))) * (u))
        #endif
#ifdef HIPFIRE_SILU_FP8_HIN
        float g0 = gate_b[base];     float u0 = 0.0f;
        float g1 = gate_b[base + 1]; float u1 = 0.0f;
        float g2 = gate_b[base + 2]; float u2 = 0.0f;
        float g3 = gate_b[base + 3]; float u3 = 0.0f;
        float g4 = gate_b[base + 4]; float u4 = 0.0f;
        float g5 = gate_b[base + 5]; float u5 = 0.0f;
        float g6 = gate_b[base + 6]; float u6 = 0.0f;
        float g7 = gate_b[base + 7]; float u7 = 0.0f;
#else
        float g0 = gate_b[base];     float u0 = up_b[base];
        float g1 = gate_b[base + 1]; float u1 = up_b[base + 1];
        float g2 = gate_b[base + 2]; float u2 = up_b[base + 2];
        float g3 = gate_b[base + 3]; float u3 = up_b[base + 3];
        float g4 = gate_b[base + 4]; float u4 = up_b[base + 4];
        float g5 = gate_b[base + 5]; float u5 = up_b[base + 5];
        float g6 = gate_b[base + 6]; float u6 = up_b[base + 6];
        float g7 = gate_b[base + 7]; float u7 = up_b[base + 7];
#endif
#ifdef HIPFIRE_SILU_FP8_AWQ
        float v0 = (SILU_MUL(g0, u0) / awq_scale[base    ]) * signs1[d0    ];
        float v1 = (SILU_MUL(g1, u1) / awq_scale[base + 1]) * signs1[d0 + 1];
        float v2 = (SILU_MUL(g2, u2) / awq_scale[base + 2]) * signs1[d0 + 2];
        float v3 = (SILU_MUL(g3, u3) / awq_scale[base + 3]) * signs1[d0 + 3];
        float v4 = (SILU_MUL(g4, u4) / awq_scale[base + 4]) * signs1[d0 + 4];
        float v5 = (SILU_MUL(g5, u5) / awq_scale[base + 5]) * signs1[d0 + 5];
        float v6 = (SILU_MUL(g6, u6) / awq_scale[base + 6]) * signs1[d0 + 6];
        float v7 = (SILU_MUL(g7, u7) / awq_scale[base + 7]) * signs1[d0 + 7];
#else
        float v0 = SILU_MUL(g0, u0) * signs1[d0    ];
        float v1 = SILU_MUL(g1, u1) * signs1[d0 + 1];
        float v2 = SILU_MUL(g2, u2) * signs1[d0 + 2];
        float v3 = SILU_MUL(g3, u3) * signs1[d0 + 3];
        float v4 = SILU_MUL(g4, u4) * signs1[d0 + 4];
        float v5 = SILU_MUL(g5, u5) * signs1[d0 + 5];
        float v6 = SILU_MUL(g6, u6) * signs1[d0 + 6];
        float v7 = SILU_MUL(g7, u7) * signs1[d0 + 7];
#endif
        #undef SILU_MUL

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

    __shared__ float smem8[8];
    const int gpr = K >> 8;
#ifdef HIPFIRE_FP8_PROD_INREG
    fp8stream_pack_row_inreg(
        held, x_fp8 + row_off,
        fp8_half_sums + (long long)token * (K >> 7),
        fp8_row_scales + token, row_amax, smem8, K, lane, wave, wave, 8);
#else
    fp8stream_pack_row_from_f32(
        x_rot_b,
        x_fp8 + row_off,
        fp8_half_sums + (long long)token * (K >> 7),
        fp8_row_scales + token,
        row_amax, smem8, K, tid, lane, wave);
#endif
}
