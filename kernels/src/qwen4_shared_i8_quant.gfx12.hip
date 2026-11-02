// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Producers of the Qwen4 shared-expert INT8 route (`qwen4_shared_i8_gemm.gfx12.hip`),
// gfx12. The host JIT concatenates `block_i8_128_quant.hip`
// (HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE), `block_i8_128_emit.hip` and this file.
// Both entries read BF16 bits and write `block_i8_128` planes ([K/128][rows]) whose
// quantizer is the standalone contract of block_i8_128_quant.hip.
//
//   qwen4_shared_rot256_i8_w   BF16 weight rows [rows][K] (K % 256 == 0) -> the plain
//       `mq_rotate_x` rotation (sign1 -> FWHT256 -> 0.0625 -> sign2, the body of
//       `mq_rotate_x_i4.hip` that `mq_rotate_x_i8_gfx12` runs on activations) -> A8 blocks
//       with the plane row `dst(row)`: mode 0 identity, mode 1 the GATE rows of the
//       interleaved gate/up plane (row r -> (r>>2)*8 + (r&3)), mode 2 its UP rows
//       (r -> (r>>2)*8 + 4 + (r&3)). The orthogonal rotation of the weight row and the
//       identically rotated activation give the same dot product up to rounding.
//   qwen4_shared_rot128_i8     BF16 rows [rows][K] (K % 128 == 0) -> the
//       `mq_rotate_x_128_v2` transform (sign1 -> FWHT128 -> 0.0883883476 -> sign2,
//       lane l holds elements 4l..4l+3) -> A8 blocks. Weights (selector null) and the
//       shared hidden activation both ride it; with a non-null `selector` the launch also
//       replaces selector[row] with sigmoid(its BF16 round trip) (the
//       `shared_expert_activation_bf16_f32` expression).
// Grid x = rows * (K / group), block 32.

#include <hip/hip_runtime.h>
#include <stdint.h>

#if defined(__gfx1200__) || defined(__gfx1201__)

static __device__ __forceinline__ float s8q_bf16_rt(float v) {
    if (!isfinite(v)) return v;
    const unsigned int u = __float_as_uint(v);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

__launch_bounds__(32, 16)
extern "C" __global__ void qwen4_shared_rot256_i8_w(
    const uint16_t* __restrict__ W,
    block_i8_128* __restrict__ Y,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int K, int rows, int dst_rows, int mode
) {
    const int tid = threadIdx.x;
    const int groups = K / 256;
    const int row = blockIdx.x / groups;
    const int group = blockIdx.x - row * groups;
    if (row >= rows) return;
    const int d0 = tid * 8;
    const int base = group * 256 + d0;
    const uint4 raw = *(const uint4*)(W + (long long)row * K + base);
    float v0 = __uint_as_float(raw.x << 16) * signs1[d0];
    float v1 = __uint_as_float(raw.x & 0xFFFF0000u) * signs1[d0 + 1];
    float v2 = __uint_as_float(raw.y << 16) * signs1[d0 + 2];
    float v3 = __uint_as_float(raw.y & 0xFFFF0000u) * signs1[d0 + 3];
    float v4 = __uint_as_float(raw.z << 16) * signs1[d0 + 4];
    float v5 = __uint_as_float(raw.z & 0xFFFF0000u) * signs1[d0 + 5];
    float v6 = __uint_as_float(raw.w << 16) * signs1[d0 + 6];
    float v7 = __uint_as_float(raw.w & 0xFFFF0000u) * signs1[d0 + 7];

    // Local butterfly: strides 1, 2, 4.
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

    // Wave butterfly: strides 1-16 in thread space via ds_swizzle.
    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while(0)
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
    const float f0 = v0 * s * signs2[d0];
    const float f1 = v1 * s * signs2[d0 + 1];
    const float f2 = v2 * s * signs2[d0 + 2];
    const float f3 = v3 * s * signs2[d0 + 3];
    const float f4 = v4 * s * signs2[d0 + 4];
    const float f5 = v5 * s * signs2[d0 + 5];
    const float f6 = v6 * s * signs2[d0 + 6];
    const float f7 = v7 * s * signs2[d0 + 7];

    int dst = row;
    if (mode == 1) dst = ((row >> 2) << 3) | (row & 3);
    if (mode == 2) dst = ((row >> 2) << 3) | 4 | (row & 3);
    emit_i8_sidecar_from_producer8(f0, f1, f2, f3, f4, f5, f6, f7, Y, group, dst, dst_rows, tid);
}

__launch_bounds__(32, 16)
extern "C" __global__ void qwen4_shared_rot128_i8(
    const uint16_t* __restrict__ H,
    block_i8_128* __restrict__ Y,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ selector,
    int K, int rows
) {
    const int tid = threadIdx.x;
    const int groups = K / 128;
    const int row = blockIdx.x / groups;
    const int group = blockIdx.x - row * groups;
    if (row >= rows) return;
    const int d0 = tid * 4;
    const uint2 raw = *(const uint2*)(H + (long long)row * K + group * 128 + d0);
    float v0 = __uint_as_float(raw.x << 16) * signs1[d0];
    float v1 = __uint_as_float(raw.x & 0xFFFF0000u) * signs1[d0 + 1];
    float v2 = __uint_as_float(raw.y << 16) * signs1[d0 + 2];
    float v3 = __uint_as_float(raw.y & 0xFFFF0000u) * signs1[d0 + 3];

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
        HBFLY(v0, pat, str); HBFLY(v1, pat, str); HBFLY(v2, pat, str); HBFLY(v3, pat, str)
    HBFLY4(0x041F, 1);
    HBFLY4(0x081F, 2);
    HBFLY4(0x101F, 4);
    HBFLY4(0x201F, 8);
    HBFLY4(0x401F, 16);
    #undef HBFLY4
    #undef HBFLY

    const float scale = 0.0883883476f;
    float xv[4];
    xv[0] = v0 * scale * signs2[d0];
    xv[1] = v1 * scale * signs2[d0 + 1];
    xv[2] = v2 * scale * signs2[d0 + 2];
    xv[3] = v3 * scale * signs2[d0 + 3];
    quantize_block_i8_128_wave(xv, Y + (long long)group * rows + row, tid);
    if (selector != nullptr && group == 0 && tid == 0) {
        selector[row] = 1.0f / (1.0f + expf(-s8q_bf16_rt(selector[row])));
    }
}

#else

extern "C" __global__ void qwen4_shared_rot256_i8_w(
    const void*, void*, const float*, const float*, int, int, int, int
) {}

extern "C" __global__ void qwen4_shared_rot128_i8(
    const void*, void*, const float*, const float*, float*, int, int
) {}

#endif
