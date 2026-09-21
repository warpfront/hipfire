// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// MQ4v2 FP8 producer-pack helpers for gfx1201 producer→fp8-stream fusion
// (`HIPFIRE_GFX12_FP8_STREAM`, exact-gfx1201, fp8-v2 default route only).
//
// The normal variants emit row-major E4M3 bytes.  `_frag` variants define
// `HIPFIRE_FP8_FRAGMENT_ORDER` and emit the same bytes in the register-direct
// WMMA layout below; decoded-code half sums and row scales stay row-major.
//
// For logical byte `(row,k)`, with `K` divisible by 16:
//   lane = (row & 15) + 16 * ((k >> 3) & 1)
//   byte_offset =
//     (((row >> 4) * (K / 16) + (k >> 4)) * 32 + lane) * 8 + (k & 7)
// Thus each 16-row x 16-K fragment is 256 contiguous bytes, and lane `l`
// owns its eight WMMA input bytes at fragment_base + 8*l.  Rows are allocated
// through `round_up(N,256)`, matching the GEMM workgroup tile; producer
// workgroups for the padded tail explicitly write FP8 zero. The mapping is a
// bijection over `[0,Npad) x [0,K)`.
//
// The scale/encode recipe is statement-identical to
// `pack_f16_to_fp8_mq4v2.gfx12.hip`.  For finite inputs, changing the address
// above is the only byte-plane difference from `prepare_mq4v2_fp8_x_f32`.

#ifndef HIPFIRE_MQ4V2_FP8_PRODUCER_PACK_HIP
#define HIPFIRE_MQ4V2_FP8_PRODUCER_PACK_HIP
#include <hip/hip_runtime.h>

__device__ inline int fp8stream_fast_log2_ceil(float x) {
    unsigned b = __builtin_bit_cast(unsigned, x);
    int e = (int)((b >> 23) & 0xFFu);
    unsigned m = b & 0x7FFFFFu;
    return e - 127 + (m != 0u ? 1 : 0);
}

__device__ inline float fp8stream_fast_pow2(int n) {
    return __builtin_bit_cast(float, (unsigned)(n + 127) << 23);
}

// OCP float8_e4m3fn decode (same finite form as act_quant_fp8_ue8m0).
__device__ inline float fp8stream_e4m3_to_f32(unsigned char b) {
    const int s = (b >> 7) & 1;
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    float v;
    if (e == 0) {
        v = (float)m * 0x1p-9f;
    } else if (e == 0xF && m == 0x7) {
        v = __int_as_float(0x7FC00000u);
    } else {
        const float pow2 = __builtin_bit_cast(float, (unsigned)(e + 120) << 23);
        v = pow2 * (1.0f + (float)m * 0.125f);
    }
    return s ? -v : v;
}

__device__ inline float fp8stream_wave_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor(v, off));
    return v;
}

__device__ inline float fp8stream_wave_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor(v, off);
    return v;
}

// a_n = 2^{ceil(log2(amax/224))} clamped to exp [-31, 9]; zero → 1.
__device__ inline float fp8stream_row_scale_from_absmax(float amax) {
    if (!(amax > 0.0f))
        return 1.0f;
    const float t = amax * (1.0f / 224.0f);
    int exp = fp8stream_fast_log2_ceil(t);
    if (exp < -31) exp = -31;
    if (exp > 9) exp = 9;
    return fp8stream_fast_pow2(exp);
}

__device__ __forceinline__ size_t fp8stream_frag_byte_offset(
    int row, int k, int K) {
    const int lane = (row & 15) + 16 * ((k >> 3) & 1);
    return ((((size_t)(row >> 4) * (size_t)(K >> 4) + (unsigned)(k >> 4))
             * 32u + (unsigned)lane) * 8u + (unsigned)(k & 7));
}

// One workgroup owns one padded row.  This is only reached by `_frag`
// producer launches whose grid is round_up(N,256); scale/sum planes retain
// their true-N extents and are intentionally not touched for padded rows.
__device__ inline void fp8stream_zero_frag_row(
    unsigned char* __restrict__ out, int row, int K, int tid) {
    for (int k = tid; k < K; k += 256)
        out[fp8stream_frag_byte_offset(row, k, K)] = 0;
}

// Pack one token row from its just-written F32 store (scale_mode=1).
//
// Uniform call over a 256-thread (8-wave) row workgroup; contains the two
// __syncthreads that separate the producer stores from the pack read and
// the scale publication from the encode (same barriers as the standalone
// pack). `smem8` is 8 shared floats (the producer's reduction smem reused
// after its last use). `x_row` is the row just stored by this workgroup
// (L1-resident re-read, no extra DRAM traffic); `out` is the byte-plane base,
// while `sums_row` (`[K/256,2]`) and `scale_slot` follow the standalone pack
// layout for row n. Reduces `row_amax` (the caller's per-thread finite absmax
// over the exact values it stored) with the standalone fmaxf order.
__device__ inline void fp8stream_pack_row_from_f32(
    const float* __restrict__ x_row,
    unsigned char* __restrict__ out,
    float* __restrict__ sums_row,
    float* __restrict__ scale_slot,
    float row_amax,
    float* __restrict__ smem8,
    int K,
    int token,
    int tid,
    int lane,
    int wave) {
    row_amax = fp8stream_wave_max(row_amax);
    if (lane == 0)
        smem8[wave] = row_amax;
    __syncthreads();
    float a_n;
    if (wave == 0 && lane == 0) {
        float amax = smem8[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w)
            amax = fmaxf(amax, smem8[w]);
        a_n = fp8stream_row_scale_from_absmax(amax);
        smem8[0] = a_n;
        *scale_slot = a_n;
    }
    __syncthreads();
    a_n = smem8[0];
    const float inv_a = 1.0f / a_n;

    const int halves = K >> 7; // K / 128
    for (int h = wave; h < halves; h += 8) {
        const int g = h >> 1;
        const int half = h & 1;
        const int k0 = g * 256 + half * 128 + lane * 4;

        const float v0 = x_row[k0 + 0] * inv_a;
        const float v1 = x_row[k0 + 1] * inv_a;
        const float v2 = x_row[k0 + 2] * inv_a;
        const float v3 = x_row[k0 + 3] * inv_a;

        unsigned int pack = 0u;
        pack = __builtin_amdgcn_cvt_pk_fp8_f32(v0, v1, pack, false);
        pack = __builtin_amdgcn_cvt_pk_fp8_f32(v2, v3, pack, true);
#ifdef HIPFIRE_FP8_FRAGMENT_ORDER
        *reinterpret_cast<unsigned int*>(
            out + fp8stream_frag_byte_offset(token, k0, K)) = pack;
#else
        *reinterpret_cast<unsigned int*>(out + (size_t)token * K + k0) = pack;
#endif

        const float s0 = fp8stream_e4m3_to_f32((unsigned char)(pack & 0xFFu));
        const float s1 = fp8stream_e4m3_to_f32((unsigned char)((pack >> 8) & 0xFFu));
        const float s2 = fp8stream_e4m3_to_f32((unsigned char)((pack >> 16) & 0xFFu));
        const float s3 = fp8stream_e4m3_to_f32((unsigned char)((pack >> 24) & 0xFFu));
        float half_sum = fp8stream_wave_sum(s0 + s1 + s2 + s3);

        if (lane == 0) {
            sums_row[g * 2 + half] = half_sum;
        }
    }
    (void)tid;
}

#endif // HIPFIRE_MQ4V2_FP8_PRODUCER_PACK_HIP
