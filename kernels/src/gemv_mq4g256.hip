// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#if HIPFIRE_MQ_ROTATE_EMIT_Q81
#include <hip/hip_fp16.h>
#include <stdint.h>
#define HIPFIRE_MQ_ROTATE_KERNEL mq_rotate_x_q8_1
struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[128];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");
#else
#define HIPFIRE_MQ_ROTATE_KERNEL mq_rotate_x
#endif

// MagnumQuant GEMV: FWHT-rotated HFQ4-G256.
// x_rot is PRE-ROTATED by the caller (one FWHT pass on x, not per-row).
// The inner loop is identical to standard HFQ4-G256 — zero rotation overhead.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq4g256(
    const char* __restrict__ A,
    const float* __restrict__ x_rot,   // PRE-ROTATED x (FWHT already applied)
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 136;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 136;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 4;

        unsigned char b0 = nibbles[byte_off];
        unsigned char b1 = nibbles[byte_off + 1];
        unsigned char b2 = nibbles[byte_off + 2];
        unsigned char b3 = nibbles[byte_off + 3];

        acc += (scale * (float)(b0 & 0xF) + zero) * x_rot[base_idx]
             + (scale * (float)(b0 >> 4)  + zero) * x_rot[base_idx + 1]
             + (scale * (float)(b1 & 0xF) + zero) * x_rot[base_idx + 2]
             + (scale * (float)(b1 >> 4)  + zero) * x_rot[base_idx + 3]
             + (scale * (float)(b2 & 0xF) + zero) * x_rot[base_idx + 4]
             + (scale * (float)(b2 >> 4)  + zero) * x_rot[base_idx + 5]
             + (scale * (float)(b3 & 0xF) + zero) * x_rot[base_idx + 6]
             + (scale * (float)(b3 >> 4)  + zero) * x_rot[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}

// Rotate x in-place: applies FWHT to each 256-element group.
// Called ONCE before the GEMV, not per-row.
// 32 threads per group, 8 elements per thread, ds_swizzle butterfly.
//
// Grid: [groups_total, batch, 1]. For decode the caller passes batch=1 and
// blockIdx.y is always 0 (byte-identical to the pre-batching version).
// For prefill, batch=N; each block processes one group × one batch row.
// x_in / x_out are laid out as [batch × K] row-major.
extern "C" __launch_bounds__(32, 16)
__global__ void HIPFIRE_MQ_ROTATE_KERNEL(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
#if HIPFIRE_MQ_ROTATE_EMIT_Q81
    block_q8_1_mmq* __restrict__ x_q8,
#endif
    int K
) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    if (group >= groups_total) return;

    const long long batch_off = (long long)blockIdx.y * K;
    const float* x_in_b  = x_in  + batch_off;
          float* x_out_b = x_out + batch_off;

    int d0 = tid * 8;
    int base = group * 256 + d0;

    float v0 = x_in_b[base]     * signs1[d0];
    float v1 = x_in_b[base + 1] * signs1[d0 + 1];
    float v2 = x_in_b[base + 2] * signs1[d0 + 2];
    float v3 = x_in_b[base + 3] * signs1[d0 + 3];
    float v4 = x_in_b[base + 4] * signs1[d0 + 4];
    float v5 = x_in_b[base + 5] * signs1[d0 + 5];
    float v6 = x_in_b[base + 6] * signs1[d0 + 6];
    float v7 = x_in_b[base + 7] * signs1[d0 + 7];

    // Local butterfly: strides 1, 2, 4
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

    // Wave butterfly: strides 1-16 in thread space via ds_swizzle
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
#if HIPFIRE_MQ_ROTATE_EMIT_Q81
    float4 out0 = make_float4(v0 * s * signs2[d0],
                             v1 * s * signs2[d0 + 1],
                             v2 * s * signs2[d0 + 2],
                             v3 * s * signs2[d0 + 3]);
    float4 out1 = make_float4(v4 * s * signs2[d0 + 4],
                             v5 * s * signs2[d0 + 5],
                             v6 * s * signs2[d0 + 6],
                             v7 * s * signs2[d0 + 7]);
    *reinterpret_cast<float4*>(x_out_b + base) = out0;
    *reinterpret_cast<float4*>(x_out_b + base + 4) = out1;

    float amax_a = fmaxf(fmaxf(fabsf(out0.x), fabsf(out0.y)), fmaxf(fabsf(out0.z), fabsf(out0.w)));
    float amax_b = fmaxf(fmaxf(fabsf(out1.x), fabsf(out1.y)), fmaxf(fabsf(out1.z), fabsf(out1.w)));
    float sum_a = out0.x + out0.y + out0.z + out0.w;
    float sum_b = out1.x + out1.y + out1.z + out1.w;
    amax_a = fmaxf(amax_a, __shfl_xor(amax_a, 2, 32));
    amax_b = fmaxf(amax_b, __shfl_xor(amax_b, 2, 32));
    sum_a += __shfl_xor(sum_a, 2, 32);
    sum_b += __shfl_xor(sum_b, 2, 32);
    amax_a = fmaxf(amax_a, __shfl_xor(amax_a, 1, 32));
    amax_b = fmaxf(amax_b, __shfl_xor(amax_b, 1, 32));
    sum_a += __shfl_xor(sum_a, 1, 32);
    sum_b += __shfl_xor(sum_b, 1, 32);
    const float amax = fmaxf(amax_a, amax_b);
    const float sum = sum_a + sum_b;
    const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
#define Q81(v) ((char)lrintf(fmaxf(-127.0f, fminf(127.0f, (v) * id))))
    const char4 q0 = make_char4(Q81(out0.x), Q81(out0.y), Q81(out0.z), Q81(out0.w));
    const char4 q1 = make_char4(Q81(out1.x), Q81(out1.y), Q81(out1.z), Q81(out1.w));
#undef Q81
    const int block_in_group = tid >> 4;
    const int qblock = group * 2 + block_in_group;
    const int qbyte = (tid & 15) * 8;
    block_q8_1_mmq* qb = x_q8 + (long long)blockIdx.y * (K / 128) + qblock;
    *reinterpret_cast<char4*>(qb->qs + qbyte) = q0;
    *reinterpret_cast<char4*>(qb->qs + qbyte + 4) = q1;
    if ((tid & 3) == 0) qb->ds4[(tid >> 2) & 3] = make_half2((_Float16)d, (_Float16)sum);
#else
    x_out_b[base]     = v0 * s * signs2[d0];
    x_out_b[base + 1] = v1 * s * signs2[d0 + 1];
    x_out_b[base + 2] = v2 * s * signs2[d0 + 2];
    x_out_b[base + 3] = v3 * s * signs2[d0 + 3];
    x_out_b[base + 4] = v4 * s * signs2[d0 + 4];
    x_out_b[base + 5] = v5 * s * signs2[d0 + 5];
    x_out_b[base + 6] = v6 * s * signs2[d0 + 6];
    x_out_b[base + 7] = v7 * s * signs2[d0 + 7];
#endif
}

#undef HIPFIRE_MQ_ROTATE_KERNEL
