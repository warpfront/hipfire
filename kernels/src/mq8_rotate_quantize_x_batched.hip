// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Batched activation producer of the MQ8G256 prefill GEMM (`gemm_mq8g256_mmq_i8.gfx12.hip`,
// `gemm_mq8g256_mmq_i8.gfx11.hip`): F32 X [N, K] -> int8 codes [N, K] + one F32 scale per
// (token, 256-group), scales laid out [K/256, N] (`Xs[group * N + token]`) so a GEMM slab's scales
// are contiguous in the token index.
//
// One portable source for gfx12 and gfx11 (wave32 only). The per-group arithmetic is
// `mq8_rotate_quantize_x` (gemv_mq8g256.hip) operation for operation, for every token row:
//   v[i]   = x[i] * signs1[i]                          (signs1 = seed 42 table, signs2 = seed 1042)
//   FWHT-256: local butterfly over strides 1, 2, 4 inside a lane's 8 elements, then ds_swizzle
//             butterflies over lane strides 1, 2, 4, 8, 16 (v_lo' = v + p, v_hi' = p - v)
//   v[i]  *= 0.0625f * signs2[i]
//   amax   = max |v| over the 256 elements (fmaxf)
//   scale  = amax > 0 ? amax / 127.0f : 1.0f           (IEEE divide)
//   inv    = amax > 0 ? 127.0f / amax : 0.0f           (IEEE divide)
//   q[i]   = (signed char) clamp(rintf(v[i] * inv), -128, 127)
// The decode GEMV and the prefill GEMM therefore see bit-identical codes and scales.
//
// One wave32 = one (token, group): lane = 8 contiguous elements. Block 256 (8 waves),
// grid [ceil(N * (K/256) / 8), 1, 1]. K % 256 == 0, N * (K/256) < 2^31, 16-byte aligned X / signs / Xq.

#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __launch_bounds__(256)
__global__ void mq8_rotate_quantize_x_batched(
    const float* __restrict__ X,
    signed char* __restrict__ Xq,
    float* __restrict__ Xs,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int K, int N
) {
    // 32-bit wave/group indexing (N * K/256 < 2^31 groups, enforced by the launcher): the division below
    // is then a 32-bit udiv, not the 64-bit expansion.
    const unsigned wave = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31u);
    const unsigned groups_per_row = (unsigned)K >> 8;
    if (wave >= groups_per_row * (unsigned)N) return;   // wave-uniform
    const unsigned token = wave / groups_per_row;
    const unsigned grp = wave - token * groups_per_row;
    const int d0 = lane * 8;
    const long long off = (long long)token * (long long)K + (long long)grp * 256 + d0;

    const float4 a = *(const float4*)(X + off);
    const float4 b = *(const float4*)(X + off + 4);
    const float4 s1a = *(const float4*)(signs1 + d0);
    const float4 s1b = *(const float4*)(signs1 + d0 + 4);
    float v0 = a.x * s1a.x, v1 = a.y * s1a.y, v2 = a.z * s1a.z, v3 = a.w * s1a.w;
    float v4 = b.x * s1b.x, v5 = b.y * s1b.y, v6 = b.z * s1b.z, v7 = b.w * s1b.w;

    // Local butterfly
    float t;
    t = v0; v0 += v1; v1 = t - v1;
    t = v2; v2 += v3; v3 = t - v3;
    t = v4; v4 += v5; v5 = t - v5;
    t = v6; v6 += v7; v7 = t - v7;
    t = v0; v0 += v2; v2 = t - v2;
    t = v1; v1 += v3; v3 = t - v3;
    t = v4; v4 += v6; v6 = t - v6;
    t = v5; v5 += v7; v7 = t - v7;
    t = v0; v0 += v4; v4 = t - v4;
    t = v1; v1 += v5; v5 = t - v5;
    t = v2; v2 += v6; v6 = t - v6;
    t = v3; v3 += v7; v7 = t - v7;

    // Wave butterfly via ds_swizzle (xor within 32 lanes)
    #define MQ8B_HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (lane & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while (0)
    #define MQ8B_HBFLY8(pat, str) \
        MQ8B_HBFLY(v0, pat, str); MQ8B_HBFLY(v1, pat, str); MQ8B_HBFLY(v2, pat, str); MQ8B_HBFLY(v3, pat, str); \
        MQ8B_HBFLY(v4, pat, str); MQ8B_HBFLY(v5, pat, str); MQ8B_HBFLY(v6, pat, str); MQ8B_HBFLY(v7, pat, str)
    MQ8B_HBFLY8(0x041F, 1); MQ8B_HBFLY8(0x081F, 2); MQ8B_HBFLY8(0x101F, 4);
    MQ8B_HBFLY8(0x201F, 8); MQ8B_HBFLY8(0x401F, 16);
    #undef MQ8B_HBFLY8
    #undef MQ8B_HBFLY

    const float s = 0.0625f;
    const float4 s2a = *(const float4*)(signs2 + d0);
    const float4 s2b = *(const float4*)(signs2 + d0 + 4);
    v0 *= s * s2a.x; v1 *= s * s2a.y; v2 *= s * s2a.z; v3 *= s * s2a.w;
    v4 *= s * s2b.x; v5 *= s * s2b.y; v6 *= s * s2b.z; v7 *= s * s2b.w;

    float my_max = fmaxf(fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))),
                         fmaxf(fmaxf(fabsf(v4), fabsf(v5)), fmaxf(fabsf(v6), fabsf(v7))));
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) my_max = fmaxf(my_max, __shfl_xor(my_max, m, 32));

    const float group_scale = (my_max > 0.0f) ? (my_max / 127.0f) : 1.0f;
    const float inv_scale = (my_max > 0.0f) ? (127.0f / my_max) : 0.0f;
    if (lane == 0) Xs[(long long)grp * (long long)N + token] = group_scale;

    const float xv[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
    unsigned w0 = 0u, w1 = 0u;
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int q0 = (int)fminf(fmaxf(rintf(xv[e] * inv_scale), -128.0f), 127.0f);
        const int q1 = (int)fminf(fmaxf(rintf(xv[e + 4] * inv_scale), -128.0f), 127.0f);
        w0 |= ((unsigned)q0 & 255u) << (8 * e);
        w1 |= ((unsigned)q1 & 255u) << (8 * e);
    }
    *(uint2*)(Xq + off) = make_uint2(w0, w1);
}
