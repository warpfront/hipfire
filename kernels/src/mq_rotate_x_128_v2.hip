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

// silu_mul_bf16_rt_f32(gate, up) followed by mq_rotate_x_128_v2, one launch:
// each lane forms its four activations with that kernel's expression (BF16
// round trips of gate, up and the product) and rotates them exactly as
// mq_rotate_x_128_v2 does, so the output is bitwise the two-launch sequence.
static __device__ __forceinline__ float rot128_bf16_rt(float v) {
    if (!isfinite(v)) return v;
    const unsigned int u = __float_as_uint(v);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

static __device__ __forceinline__ float rot128_silu_mul(float g, float u) {
    const float v = rot128_bf16_rt(g);
    return rot128_bf16_rt((v / (1.0f + expf(-v))) * rot128_bf16_rt(u));
}

//
// Optional shared expert (`shared_out` non-null): the grid's extra last
// y-row applies silu_mul.hip's shared_expert_activation_bf16_f32 to
// `shared_n` elements (four per lane at the same group * 128 + lane * 4
// positions) and replaces `selector[0]` with sigmoid(its BF16 round trip),
// the same expressions, so it is bitwise that launch as well.
extern "C" __global__ void mq_rotate_x_128_v2_silu_bf16(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ x_out,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int K,
    const float* __restrict__ shared_gate,
    const float* __restrict__ shared_up,
    float* __restrict__ shared_out,
    float* __restrict__ selector,
    int shared_n
) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    if (shared_out && blockIdx.y == gridDim.y - 1) {
        if (group == 0 && tid == 0)
            selector[0] = 1.0f / (1.0f + expf(-rot128_bf16_rt(selector[0])));
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int i = group * 128 + tid * 4 + e;
            if (i < shared_n) shared_out[i] = rot128_silu_mul(shared_gate[i], shared_up[i]);
        }
        return;
    }
    const int groups_total = (K + 127) / 128;
    if (group >= groups_total || K <= 0) return;

    const long long batch_off = (long long)blockIdx.y * K;
    const float* gate_b = gate + batch_off;
    const float* up_b = up + batch_off;
    float* x_out_b = x_out + batch_off;
    const int d0 = tid * 4;
    const int base = group * 128 + d0;

    const int i0 = base;
    const int i1 = base + 1;
    const int i2 = base + 2;
    const int i3 = base + 3;
    float v0 = (i0 < K ? rot128_silu_mul(gate_b[i0], up_b[i0]) : 0.0f) * signs1[d0];
    float v1 = (i1 < K ? rot128_silu_mul(gate_b[i1], up_b[i1]) : 0.0f) * signs1[d0 + 1];
    float v2 = (i2 < K ? rot128_silu_mul(gate_b[i2], up_b[i2]) : 0.0f) * signs1[d0 + 2];
    float v3 = (i3 < K ? rot128_silu_mul(gate_b[i3], up_b[i3]) : 0.0f) * signs1[d0 + 3];

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
