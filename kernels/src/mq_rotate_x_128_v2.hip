// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — Qwen4 MQ4G128V2 activation transform.

#include <hip/hip_runtime.h>

// qt=53 uses row-local ceil(K/128) groups.  Unlike the legacy G128
// transform, this entry point explicitly zero-fills the logical tail before
// the FWHT, so a K that is not a multiple of 128 can never read the next row.
// The butterfly body intentionally matches mq_rotate_x_128 and the CPU codec:
// sign1 -> radix-2 FWHT -> 1/sqrt(128) -> sign2.
__launch_bounds__(32, 16)
extern "C" __global__ void mq_rotate_x_128_v2(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int K
) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_total = (K + 127) / 128;
    if (group >= groups_total || K <= 0) return;

    const long long batch_off = (long long)blockIdx.y * K;
    const float* x_in_b = x_in + batch_off;
    float* x_out_b = x_out + batch_off;
    const int d0 = tid * 4;
    const int base = group * 128 + d0;

    const int i0 = base;
    const int i1 = base + 1;
    const int i2 = base + 2;
    const int i3 = base + 3;
    float v0 = (i0 < K ? x_in_b[i0] : 0.0f) * signs1[d0];
    float v1 = (i1 < K ? x_in_b[i1] : 0.0f) * signs1[d0 + 1];
    float v2 = (i2 < K ? x_in_b[i2] : 0.0f) * signs1[d0 + 2];
    float v3 = (i3 < K ? x_in_b[i3] : 0.0f) * signs1[d0 + 3];

    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;

#define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) (v) = _p - (v); else (v) = (v) + _p; \
    } while (0)
#define HBFLY4(pat, str) \
    HBFLY(v0, pat, str); HBFLY(v1, pat, str); HBFLY(v2, pat, str); \
    HBFLY(v3, pat, str)
    HBFLY4(0x041F, 1);
    HBFLY4(0x081F, 2);
    HBFLY4(0x101F, 4);
    HBFLY4(0x201F, 8);
    HBFLY4(0x401F, 16);
#undef HBFLY4
#undef HBFLY

    const float scale = 0.0883883476f;
    // x_out has the exact logical row width.  Tail lanes are transformed for
    // the padded group but are intentionally not written beyond K.
    if (i0 < K) x_out_b[i0] = v0 * scale * signs2[d0];
    if (i1 < K) x_out_b[i1] = v1 * scale * signs2[d0 + 1];
    if (i2 < K) x_out_b[i2] = v2 * scale * signs2[d0 + 2];
    if (i3 < K) x_out_b[i3] = v3 * scale * signs2[d0 + 3];
}

// mq_rotate_x_128_v2 writing F16 for the MoE down WMMA GEMM.
__launch_bounds__(32, 16)
extern "C" __global__ void mq_rotate_x_128_v2_f16(
    const float* __restrict__ x_in,
    _Float16* __restrict__ x_out,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int K
) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_total = (K + 127) / 128;
    if (group >= groups_total || K <= 0) return;

    const long long batch_off = (long long)blockIdx.y * K;
    const float* x_in_b = x_in + batch_off;
    _Float16* x_out_b = x_out + batch_off;
    const int d0 = tid * 4;
    const int base = group * 128 + d0;

    const int i0 = base;
    const int i1 = base + 1;
    const int i2 = base + 2;
    const int i3 = base + 3;
    float v0 = (i0 < K ? x_in_b[i0] : 0.0f) * signs1[d0];
    float v1 = (i1 < K ? x_in_b[i1] : 0.0f) * signs1[d0 + 1];
    float v2 = (i2 < K ? x_in_b[i2] : 0.0f) * signs1[d0 + 2];
    float v3 = (i3 < K ? x_in_b[i3] : 0.0f) * signs1[d0 + 3];

    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;

#define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) (v) = _p - (v); else (v) = (v) + _p; \
    } while (0)
#define HBFLY4(pat, str) \
    HBFLY(v0, pat, str); HBFLY(v1, pat, str); HBFLY(v2, pat, str); \
    HBFLY(v3, pat, str)
    HBFLY4(0x041F, 1);
    HBFLY4(0x081F, 2);
    HBFLY4(0x101F, 4);
    HBFLY4(0x201F, 8);
    HBFLY4(0x401F, 16);
#undef HBFLY4
#undef HBFLY

    const float scale = 0.0883883476f;
    // x_out has the exact logical row width.  Tail lanes are transformed for
    // the padded group but are intentionally not written beyond K.
    // The F32 product is pinned before the conversion (fast-math could fold
    // it into F16 math): bytes equal mq_rotate_x_128_v2 + convert_f32_to_f16.
#define ROT128_F16(i, v, d) do { \
        float r_ = (v) * scale * signs2[d]; \
        asm volatile("" : "+v"(r_)); \
        if ((i) < K) x_out_b[i] = (_Float16)r_; \
    } while (0)
    ROT128_F16(i0, v0, d0);
    ROT128_F16(i1, v1, d0 + 1);
    ROT128_F16(i2, v2, d0 + 2);
    ROT128_F16(i3, v3, d0 + 3);
#undef ROT128_F16
}
