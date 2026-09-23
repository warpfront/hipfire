// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// MQ4v2 gate/up FP8-WMMA activation pre-pass for gfx12 (RDNA4 / gfx1201).
//
// Converts one activation row of F16 to E4M3 FP8 bytes, writes per-half
// decoded row sums for the zero-point correction identity, and an optional
// per-row power-of-two scale a_n (unit scale when scale_mode==0).
//
// Geometry (rev 2: 8 waves/row for occupancy — same outputs bit-for-bit):
//   grid  [N, 1, 1]
//   block [256, 1, 1] — eight wave32s own one activation row
//   LDS   32 B        — eight wave maxima (reused to publish a_n)
// Preconditions (host-enforced): N>0, K>0, K % 256 == 0.
//
// scale_mode:
//   0 — a_n = 1 for every row (measurement / unit-scale arm)
//   1 — a_n = 2^{clamp(ceil(log2(absmax/224)), -31, 9)}; zero row → 1
//
// Half layout: half h covers group g = h/2, side h%2 —
//   K[g*256 + side*128 .. g*256 + side*128 + 128).
// Each lane owns four consecutive F16 values and emits one FP8 dword via
// __builtin_amdgcn_cvt_pk_fp8_f32. Half sums are reduced from the DECODED
// emitted bytes (not the pre-round F16 values).
//
// Why this is output-identical to the block-64 form: phase 1 is an fmax
// reduction (order-independent over finite inputs); phase 2 keeps the same
// (half, lane) -> four-consecutive-K ownership (half h = group h/2, side
// h%2, lane L at half_base + 4*L), so the packed bytes, the per-lane
// decoded-value sums and the wave-shuffle reduction tree are unchanged.

__device__ __forceinline__ int mq4v2_fast_log2_ceil(float x) {
    unsigned b = __builtin_bit_cast(unsigned, x);
    int e = (int)((b >> 23) & 0xFFu);
    unsigned m = b & 0x7FFFFFu;
    return e - 127 + (m != 0u ? 1 : 0);
}

__device__ __forceinline__ float mq4v2_fast_pow2(int n) {
    return __builtin_bit_cast(float, (unsigned)(n + 127) << 23);
}

// OCP float8_e4m3fn decode (same finite form as act_quant_fp8_ue8m0).
__device__ __forceinline__ float mq4v2_e4m3_to_f32(unsigned char b) {
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

__device__ __forceinline__ float mq4v2_wave_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor(v, off));
    return v;
}

__device__ __forceinline__ float mq4v2_wave_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor(v, off);
    return v;
}

// a_n = 2^{ceil(log2(amax/224))} clamped to exp [-31, 9]; zero → 1.
__device__ __forceinline__ float mq4v2_row_scale_from_absmax(float amax) {
    if (!(amax > 0.0f))
        return 1.0f;
    const float t = amax * (1.0f / 224.0f);
    int exp = mq4v2_fast_log2_ceil(t);
    if (exp < -31) exp = -31;
    if (exp > 9) exp = 9;
    return mq4v2_fast_pow2(exp);
}

extern "C" __launch_bounds__(256, 1) __global__
void pack_f16_to_fp8_mq4v2_gfx12(
    const _Float16* x_f16,       // [N,K]
    unsigned char* x_fp8,        // [N,K], E4M3 bytes
    float* half_sums,            // [N,K/256,2]
    float* row_scales,           // [N]
    int K,
    int N,
    int scale_mode)             // 0=unit, 1=pow2-to-224
{
    // 32 B static LDS: eight wave maxima, then republished a_n in slot 0.
    __shared__ float smem[8];

    const int n = (int)blockIdx.x;
    if (n >= N || K <= 0) return;

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5; // 0..7
    const int gpr = K >> 8;    // K / 256

    const _Float16* x_row = x_f16 + (long long)n * K;
    unsigned char* out_row = x_fp8 + (long long)n * K;

    float a_n = 1.0f;

    if (scale_mode == 1) {
        // 1) Strided F16 absmax over the full row (256-thread stride).
        float local_max = 0.0f;
        for (int k = tid; k < K; k += 256) {
            const float v = (float)x_row[k];
            local_max = fmaxf(local_max, fabsf(v));
        }
        // 2) Per-wave reduce; lane 0 of each wave writes one LDS float.
        local_max = mq4v2_wave_max(local_max);
        if (lane == 0)
            smem[wave] = local_max;
        __syncthreads();
        // Wave 0 resolves a_n and publishes it; all threads re-read.
        if (wave == 0 && lane == 0) {
            float amax = smem[0];
            #pragma unroll
            for (int w = 1; w < 8; ++w)
                amax = fmaxf(amax, smem[w]);
            a_n = mq4v2_row_scale_from_absmax(amax);
            smem[0] = a_n;
            row_scales[n] = a_n;
        }
        __syncthreads();
        a_n = smem[0];
    } else {
        // scale_mode == 0: unit scale, no absmax scan / barrier.
        if (tid == 0)
            row_scales[n] = 1.0f;
        a_n = 1.0f;
    }

    const float inv_a = 1.0f / a_n;

    // 3) Pack each 128-K half: wave-strided over the K/128 halves
    //    (h = wave, wave+8, ...; g = h/2, side = h%2). Lane L owns four
    //    consecutive values at half_base + 4*L — the same ownership as the
    //    block-64 form, so bytes and half sums match bit-for-bit.
    const int halves = K >> 7; // K / 128
    for (int h = wave; h < halves; h += 8) {
        const int g = h >> 1;
        const int half = h & 1;
        const int k0 = g * 256 + half * 128 + lane * 4;

        const float v0 = (float)x_row[k0 + 0] * inv_a;
        const float v1 = (float)x_row[k0 + 1] * inv_a;
        const float v2 = (float)x_row[k0 + 2] * inv_a;
        const float v3 = (float)x_row[k0 + 3] * inv_a;

        unsigned int pack = 0u;
        pack = __builtin_amdgcn_cvt_pk_fp8_f32(v0, v1, pack, false);
        pack = __builtin_amdgcn_cvt_pk_fp8_f32(v2, v3, pack, true);
        *reinterpret_cast<unsigned int*>(out_row + k0) = pack;

        // Half sum from DECODED emitted bytes (operand-exact for WMMA path).
        const float s0 = mq4v2_e4m3_to_f32((unsigned char)(pack & 0xFFu));
        const float s1 = mq4v2_e4m3_to_f32((unsigned char)((pack >> 8) & 0xFFu));
        const float s2 = mq4v2_e4m3_to_f32((unsigned char)((pack >> 16) & 0xFFu));
        const float s3 = mq4v2_e4m3_to_f32((unsigned char)((pack >> 24) & 0xFFu));
        float half_sum = mq4v2_wave_sum(s0 + s1 + s2 + s3);

        if (lane == 0) {
            // half_sums[(n * gpr + g) * 2 + half]
            half_sums[((long long)n * gpr + g) * 2 + half] = half_sum;
        }
    }
}

// F32-input variant of the pre-pass above: same ABI except `const float* X`,
// same grid/block/LDS/geometry, same scale_mode contract, same rounding
// instruction (`__builtin_amdgcn_cvt_pk_fp8_f32`, round-to-nearest-even).
//
// Rounding note: the F16 path casts F32->F16 on the host side
// (`convert_f32_to_f16`, one rn rounding) and this TU casts F16->F32 on load
// (exact) before the e4m3 cvt (second rn rounding) — two roundings. This
// variant rounds F32->E4M3 once, directly. Outputs are therefore identical
// except possibly in the last E4M3 ulp for values near a rounding boundary.
// That is expected and accepted; quality is covered by the WT2 KLD gate.
extern "C" __launch_bounds__(256, 1) __global__
void pack_f32_to_fp8_mq4v2_gfx12(
    const float* x_f32,          // [N,K]
    unsigned char* x_fp8,        // [N,K], E4M3 bytes
    float* half_sums,            // [N,K/256,2]
    float* row_scales,           // [N]
    int K,
    int N,
    int scale_mode)             // 0=unit, 1=pow2-to-224
{
    // 32 B static LDS: eight wave maxima, then republished a_n in slot 0.
    __shared__ float smem[8];

    const int n = (int)blockIdx.x;
    if (n >= N || K <= 0) return;

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5; // 0..7
    const int gpr = K >> 8;    // K / 256

    const float* x_row = x_f32 + (long long)n * K;
    unsigned char* out_row = x_fp8 + (long long)n * K;

    float a_n = 1.0f;

    if (scale_mode == 1) {
        // 1) Strided F32 absmax over the full row (256-thread stride).
        float local_max = 0.0f;
        for (int k = tid; k < K; k += 256) {
            const float v = x_row[k];
            local_max = fmaxf(local_max, fabsf(v));
        }
        // 2) Per-wave reduce; lane 0 of each wave writes one LDS float.
        local_max = mq4v2_wave_max(local_max);
        if (lane == 0)
            smem[wave] = local_max;
        __syncthreads();
        // Wave 0 resolves a_n and publishes it; all threads re-read.
        if (wave == 0 && lane == 0) {
            float amax = smem[0];
            #pragma unroll
            for (int w = 1; w < 8; ++w)
                amax = fmaxf(amax, smem[w]);
            a_n = mq4v2_row_scale_from_absmax(amax);
            smem[0] = a_n;
            row_scales[n] = a_n;
        }
        __syncthreads();
        a_n = smem[0];
    } else {
        // scale_mode == 0: unit scale, no absmax scan / barrier.
        if (tid == 0)
            row_scales[n] = 1.0f;
        a_n = 1.0f;
    }

    const float inv_a = 1.0f / a_n;

    // 3) Pack each 128-K half: wave-strided over the K/128 halves
    //    (h = wave, wave+8, ...; g = h/2, side = h%2). Lane L owns four
    //    consecutive values at half_base + 4*L — the same ownership as the
    //    block-64 form, so bytes and half sums match bit-for-bit.
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
        *reinterpret_cast<unsigned int*>(out_row + k0) = pack;

        // Half sum from DECODED emitted bytes (operand-exact for WMMA path).
        const float s0 = mq4v2_e4m3_to_f32((unsigned char)(pack & 0xFFu));
        const float s1 = mq4v2_e4m3_to_f32((unsigned char)((pack >> 8) & 0xFFu));
        const float s2 = mq4v2_e4m3_to_f32((unsigned char)((pack >> 16) & 0xFFu));
        const float s3 = mq4v2_e4m3_to_f32((unsigned char)((pack >> 24) & 0xFFu));
        float half_sum = mq4v2_wave_sum(s0 + s1 + s2 + s3);

        if (lane == 0) {
            // half_sums[(n * gpr + g) * 2 + half]
            half_sums[((long long)n * gpr + g) * 2 + half] = half_sum;
        }
    }
}
