// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Model-lifetime decode of qt=55 FP8E4M3B128 expert blocks to BF16 (gfx942).
// The DS4 parent decoder (dequant_fp8_e4m3_ue8m0_blk128_to_bf16.gfx942.hip)
// generalised from UE8M0 to BF16 block scales and to the expert-blocked
// qt=55 payload; used for Tier C twins and test oracles, not on the hot path.
//
//   blocks: E consecutive qt=55 expert blocks, stride
//           S = round_up(R*K + 2*ceil(R/128)*ceil(K/128), 256);
//           block e = codes u8[R*K] row-major || s_inv bf16 [R/128, K/128] || pad
//   out:    BF16 bits [E, R, K] row-major
//
//   out[e][r][k] = bf16_rne(f32(e4m3fn(q)) * f32(s_inv[r/128][k/128]))
//
// Decode is the exact FNUZ fold used by gemm_fp8b128_moe_grouped_bf16mfma
// (same device function body): |q| read as FNUZ by v_cvt_f32_fp8 is
// |e4m3fn(q)|/2; times f32(2*s) is the exact F32 product; RNE on the bits;
// the code's sign XORed into the BF16 sign (0x80 -> teacher's -0 = 0x8000).
// NaN codes (0x7F/0xFF) are refused upstream by the converter.
//
// Geometry: block 256, grid [ceil(R*K/16 / 256), E]; each thread decodes 16
// consecutive codes (one dwordx4) into 32 output bytes. Contract: K % 16 == 0.

#include <hip/hip_runtime.h>

typedef float fp8dq_f32x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ unsigned fp8dq_rne_bias(unsigned u) {
    return u + 0x7FFFu + ((u >> 16) & 1u);
}

__device__ __forceinline__ unsigned fp8dq_pack_hi(unsigned hi, unsigned lo) {
    return __builtin_amdgcn_perm(hi, lo, 0x07060302u);
}

__device__ __forceinline__ void fp8dq_decode4(unsigned w, float s2, unsigned& o01, unsigned& o23) {
    const unsigned mag = w & 0x7F7F7F7Fu;
    fp8dq_f32x2 a = __builtin_amdgcn_cvt_pk_f32_fp8((int)mag, false);
    fp8dq_f32x2 b = __builtin_amdgcn_cvt_pk_f32_fp8((int)mag, true);
    a *= s2;
    b *= s2;
    const unsigned sg = w & 0x80808080u;
    o01 = fp8dq_pack_hi(fp8dq_rne_bias(__float_as_uint(a.y)), fp8dq_rne_bias(__float_as_uint(a.x)))
        ^ __builtin_amdgcn_perm(sg, sg, 0x010C000Cu);
    o23 = fp8dq_pack_hi(fp8dq_rne_bias(__float_as_uint(b.y)), fp8dq_rne_bias(__float_as_uint(b.x)))
        ^ __builtin_amdgcn_perm(sg, sg, 0x030C020Cu);
}

extern "C" __global__ __launch_bounds__(256) void dequant_fp8b128_to_bf16_gfx942(
    const unsigned char* __restrict__ blocks,
    unsigned short* __restrict__ out,
    int R,
    int K,
    long long block_stride) {
    const long long rk = (long long)R * K;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rk / 16) return;
    const int e = blockIdx.y;
    const unsigned char* __restrict__ blk = blocks + (long long)e * block_stride;
    const unsigned short* __restrict__ sc = reinterpret_cast<const unsigned short*>(blk + rk);
    const long long e0 = i * 16;
    const int r = (int)(e0 / K);
    const int k = (int)(e0 - (long long)r * K);
    const int nkb = (K + 127) >> 7;
    // 16 consecutive codes never straddle a 128-column scale block (K % 16 == 0).
    const float s2 = 2.0f * __uint_as_float((unsigned)sc[(r >> 7) * nkb + (k >> 7)] << 16);
    const uint4 w = *reinterpret_cast<const uint4*>(blk + e0);
    uint4 o0, o1;
    fp8dq_decode4(w.x, s2, o0.x, o0.y);
    fp8dq_decode4(w.y, s2, o0.z, o0.w);
    fp8dq_decode4(w.z, s2, o1.x, o1.y);
    fp8dq_decode4(w.w, s2, o1.z, o1.w);
    uint4* o = reinterpret_cast<uint4*>(out + (long long)e * rk + e0);
    o[0] = o0;
    o[1] = o1;
}
