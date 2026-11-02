// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx942 (CDNA3, wave64) plain FP8 activation producer — HIP twin of the
// PeaceMaker kernel symbol `pm_cdna_fp8_act_producer`.
//
// Port of `pack_f32_to_fp8_mq4v2_gfx12` (kernels/src/pack_f16_to_fp8_mq4v2.gfx12.hip,
// :194-287, helpers :25-87) with the SAME 8-argument ABI and the SAME output
// layout:
//   x_f32       const float*  [N,K]        input activations (16-byte aligned)
//   x_fp8       uchar*        [N,K]        E4M3 codes
//   half_sums   float*        [N,K/256,2]  per-128-half sums of the DECODED
//                                          emitted codes (OCP E4M3 domain)
//   row_scales  float*        [N]          a_n (power of two; 1.0 in mode 0)
//   K, N, scale_mode, row_scale_shift      (int)
//
// Geometry: grid [N,1,1], block [64,1,1] — ONE wave64 owns one row; no LDS,
// no barriers.
//   * amax: per-lane strided fmax over the row, then a native wave64
//     XOR32,16,8,4,2,1 max reduction. (fmax over finite values is
//     order-independent, so this equals the source's block-wide reduction.)
//   * Logical 32-lane groups: tid 0..31 = side 0, tid 32..63 = side 1.
//     For 256-wide group g, lane L (= tid) owns K = g*256 + 4*tid .. +3, i.e.
//     side 0 packs K[g*256, g*256+128) and side 1 packs K[g*256+128, g*256+256)
//     with four consecutive values per lane — identical (half, lane32) ->
//     four-consecutive-K ownership to the source (half h = 2g+side).
//   * Half sums: per-lane s0+s1+s2+s3 (left-associated, decoded from the
//     emitted bytes), then the source's XOR16,8,4,2,1 tree inside each
//     logical 32-lane group (__shfl_xor width 32). Lane32==0 of each group
//     stores half_sums[(n*K/256 + g)*2 + side].
//
// scale_mode (source semantics: only the value 1 selects the scaled path):
//   != 1 — a_n = 1.0 for every row; row_scale_shift is IGNORED (as in the source).
//   == 1 — t = RN_f32(amax * RN_f32(1/224)); e = ceil(log2(t)) from the f32 bit
//          pattern (exponent field, +1 if mantissa != 0), clamped to [-31, 9];
//          amax == 0 -> a_n = 1.0 (e = 0). a_n = 2^e, then (shift != 0)
//          a_n *= 2^row_scale_shift exactly as the source (float multiply).
//          Source route shift range: [-1, 3].
//
// Scaling: with a_n == 2^ae (ae = unbiased exponent of the final a_n, read from
// its bit pattern), the codes are formed from ldexpf(x, -ae-1): exact scaling
// by 2^(-ae-1), NO reciprocal and no division. The extra x0.5 is the FNUZ
// fold: the gfx942 hardware conversion is FNUZ (bias 8), so halving the input
// makes the emitted byte, decoded as OCP E4M3 (bias 7), represent x / a_n.
// It is applied unconditionally (also when ae == 0 / scale_mode != 1).
//
// Rounding/encoding: __builtin_amdgcn_cvt_pk_fp8_f32 (round-to-nearest-even),
// two dwords-halves per lane (word_sel false then true), as in the source.
// Half sums use a software OCP float8_e4m3fn decode of the SAME emitted bytes
// (same finite form as the source helper `mq4v2_e4m3_to_f32`), which keeps the
// half-sum domain OCP. Byte identity with the source holds over the OCP-finite
// range; behavior at the upper clamp / overflow is defined by the PM kernel and
// its CPU reference and is excluded from that identity.
//
// Unsupported / host-enforced preconditions (no device-side checks):
//   * Non-finite inputs (inf/NaN) are UNSUPPORTED: the amax and the scale are
//     undefined for them and the result is unspecified.
//   * K > 0 and K % 256 == 0 (the kernel indexes K/256 whole groups; any
//     remainder would be silently dropped and half_sums would be missing).
//   * N > 0; row_scale_shift in [-1, 3] (source route) so a_n stays a normal
//     f32 power of two (as for the source's fast_pow2 bit trick).
//   * x_f32 must be 16-byte aligned (rows are multiples of 1024 B).
//
// [UNVERIFIED: confirm on MI300X] Negative zero / negative underflow is
// expected to emit canonical FNUZ +0 (0x00); 0x80 is FNUZ NaN, NOT -0.
// FP16_OVFL=0 overflow is expected to emit 0x80; its OCP sidecar decode is
// -0. Byte identity excludes overflow and normalizes source OCP -0 to +0.

__device__ __forceinline__ int pm_cdna_act_log2_ceil(float x) {
    unsigned b = __builtin_bit_cast(unsigned, x);
    int e = (int)((b >> 23) & 0xFFu);
    unsigned m = b & 0x7FFFFFu;
    return e - 127 + (m != 0u ? 1 : 0);
}

__device__ __forceinline__ float pm_cdna_act_pow2(int n) {
    return __builtin_bit_cast(float, (unsigned)(n + 127) << 23);
}

// OCP float8_e4m3fn decode (software; finite form identical to the source).
__device__ __forceinline__ float pm_cdna_act_e4m3_to_f32(unsigned b) {
    const int s = (int)((b >> 7) & 1u);
    const int e = (int)((b >> 3) & 0xFu);
    const int m = (int)(b & 0x7u);
    float v;
    if (e == 0) {
        v = (float)m * 0x1p-9f;
    } else if (e == 0xF && m == 0x7) {
        v = __int_as_float(0x7FC00000);
    } else {
        const float p2 = __builtin_bit_cast(float, (unsigned)(e + 120) << 23);
        v = p2 * (1.0f + (float)m * 0.125f);
    }
    return s ? -v : v;
}

// Native wave64 max: XOR 32,16,8,4,2,1.
__device__ __forceinline__ float pm_cdna_act_wave_max(float v) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor(v, off, 64));
    return v;
}

// Logical-32 sum: source tree XOR 16,8,4,2,1, confined to each 32-lane group.
__device__ __forceinline__ float pm_cdna_act_group32_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor(v, off, 32);
    return v;
}

// a_n = 2^{ceil(log2(amax/224))} clamped to exp [-31, 9]; zero -> 1.
__device__ __forceinline__ float pm_cdna_act_scale_from_absmax(float amax) {
    if (!(amax > 0.0f))
        return 1.0f;
    const float t = amax * (1.0f / 224.0f);
    int e = pm_cdna_act_log2_ceil(t);
    if (e < -31) e = -31;
    if (e > 9) e = 9;
    return pm_cdna_act_pow2(e);
}

extern "C" __launch_bounds__(64, 1) __global__
void pm_cdna_fp8_act_producer(
    const float* x_f32,          // [N,K]
    unsigned char* x_fp8,        // [N,K], E4M3 codes
    float* half_sums,            // [N,K/256,2]
    float* row_scales,           // [N]
    int K,
    int N,
    int scale_mode,              // !=1 unit (shift ignored), 1=pow2-to-224
    int row_scale_shift)
{
    const int n = (int)blockIdx.x;
    if (n >= N || K <= 0) return;

    const int tid = (int)threadIdx.x;      // 0..63
    const int lane32 = tid & 31;
    const int side = tid >> 5;             // logical 32-lane group: 0 or 1
    const int gpr = K >> 8;                // K / 256

    const float* x_row = x_f32 + (long long)n * K;
    unsigned char* out_row = x_fp8 + (long long)n * K;

    // Lane tid owns K = g*256 + 4*tid .. +3 (== g*256 + side*128 + lane32*4).
    int ae = 0;                            // a_n == 2^ae
    if (scale_mode == 1) {
        float local_max = 0.0f;
        for (int g = 0; g < gpr; ++g) {
            const float4 v = *reinterpret_cast<const float4*>(x_row + g * 256 + tid * 4);
            local_max = fmaxf(local_max, fabsf(v.x));
            local_max = fmaxf(local_max, fabsf(v.y));
            local_max = fmaxf(local_max, fabsf(v.z));
            local_max = fmaxf(local_max, fabsf(v.w));
        }
        const float amax = pm_cdna_act_wave_max(local_max);
        float a_n = pm_cdna_act_scale_from_absmax(amax);
        if (row_scale_shift != 0) a_n *= pm_cdna_act_pow2(row_scale_shift);
        ae = (int)((__builtin_bit_cast(unsigned, a_n) >> 23) & 0xFFu) - 127;
        if (tid == 0)
            row_scales[n] = a_n;
    } else {
        if (tid == 0)
            row_scales[n] = 1.0f;
    }

    for (int g = 0; g < gpr; ++g) {
        const int k0 = g * 256 + tid * 4;
        const float4 x = *reinterpret_cast<const float4*>(x_row + k0);

        // gfx942 hardware conversion is FNUZ: fold the x0.5 into the exact
        // scaling so the same byte decodes (OCP) to x / a_n. Always applied,
        // including the unit-scale path (ae == 0 -> ldexpf(x, -1)).
        const float v0 = ldexpf(x.x, -ae - 1);
        const float v1 = ldexpf(x.y, -ae - 1);
        const float v2 = ldexpf(x.z, -ae - 1);
        const float v3 = ldexpf(x.w, -ae - 1);

        unsigned int pack = 0u;
        pack = __builtin_amdgcn_cvt_pk_fp8_f32(v0, v1, pack, false);
        pack = __builtin_amdgcn_cvt_pk_fp8_f32(v2, v3, pack, true);
        *reinterpret_cast<unsigned int*>(out_row + k0) = pack;

        // Half sum from the DECODED emitted bytes.
        const float s0 = pm_cdna_act_e4m3_to_f32(pack & 0xFFu);
        const float s1 = pm_cdna_act_e4m3_to_f32((pack >> 8) & 0xFFu);
        const float s2 = pm_cdna_act_e4m3_to_f32((pack >> 16) & 0xFFu);
        const float s3 = pm_cdna_act_e4m3_to_f32((pack >> 24) & 0xFFu);
        const float half_sum = pm_cdna_act_group32_sum(s0 + s1 + s2 + s3);

        if (lane32 == 0) {
            // half_sums[(n * gpr + g) * 2 + side]
            half_sums[((long long)n * gpr + g) * 2 + side] = half_sum;
        }
    }
}
