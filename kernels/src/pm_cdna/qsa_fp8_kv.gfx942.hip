// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx942 (CDNA3, wave64) HIP twin of the QSA FP8 K/V row codec:
//   encode  qsa_kv_store<QSA_KV_FP8>        kernels/src/tensor_ops.hip:3091-3154
//   decode  the scalar read in qsa_kv_dot   kernels/src/tensor_ops.hip:3062-3087
// and of the PeaceMaker gfx942 kernel `crates/peacemaker-isa/src/kernels/cdna/
// qsa_fp8_kv.rs`. Route: pm-cdna3-ref/route/R4-fp8-kv.md (E1, "Proposed gfx942
// E1/E2 encoder and decoder contract").
//
// Stored format (identical to the gfx1201 QSA FP8 state, OCP E4M3FN bytes):
//   row = Hkv*D code bytes (head-major), then Hkv little-endian F16 scales,
//   row_bytes = Hkv*D + 2*Hkv.  One F16 scale per head per row.
//   scale s16 = amax==0 ? 1 : F16(amax/448); a positive rounded-to-zero scale
//   becomes bits 0x0001; if 448*f32(s16) < amax the F16 bits are incremented
//   by ONE ulp (capped at 0x7bff): an `if`, not a bump loop.
//   code = OCP E4M3FN RNE(value / f32(s16)), |code value| <= 448.
//
// gfx942 FP8 conversion instructions are FNUZ (bias 8, max 240, 0x80 = NaN,
// CDNA3 ISA §7.2 p.50-51), so the OCP byte is produced by the FNUZ-half fold:
// for finite z with |z| <= 448 the FNUZ grid of z/2 is exactly the OCP grid of
// z (exponent bias shift, subnormals included), RNE gives identical bytes, and
// z*0.5 is exact. The only divergence is signed zero: FNUZ has no -0, so the
// OCP byte 0x80 (input -0, negative values rounding to zero) is stored as 0x00.
// Decode reverses the fold: finite OCP byte b -> cvt_f32_fp8(b) * (2*f32(s16)),
// with 2*s16 formed in F32 (2*65504 is legal in F32, not in F16). Decode is the
// bare FNUZ read, exactly as the PM kernel does it: the encoder never emits the
// OCP codes 0x80 (-0), 0x7f/0xff (NaN); an imported 0x80 reads as FNUZ NaN and
// 0x7f/0xff as the finite FNUZ extremum (route: such bytes must be recognised
// and canonicalised before use, they are outside this codec's domain).
// [UNVERIFIED: confirm on MI300X: FNUZ conversion of -0 / negative underflow
// (the canonicalisation below does not depend on it), FP16_OVFL setup, and the
// compiled division sequence].
//
// Numerics contract: the quotient value / sf and the scale amax / 448 are plain
// IEEE binary32 round-to-nearest-even divisions. They MUST NOT be turned into
// reciprocal multiplies or approximate divisions: that would change stored
// bytes. The guards below refuse fast-math builds, contraction is disabled, and
// crates/peacemaker-isa/tests/cdna_qsa_fp8_kv.rs checks the compiled ISA for
// the v_div_scale/v_div_fmas/v_div_fixup sequence.
//
// Launch (all kernels): grid = (kv_heads, rows), block = 256, one block per
// (kv head, row), thread = channel (head_dim fixed at 256, kv_heads 2 or 4).
// Eight logical 32-thread groups keep the original max-reduction indexing
// (`tid >> 5` is a logical group, not a physical wave; width-32 shuffles are
// explicit so wave64 does not change the slot mapping).
//
// ABI is exactly the PM kernels' (`qsa_kv_{encode,decode}_fp8_h{2,4}_pm_gfx942`),
// pointer-only: specialised exported symbols `qsa_kv_{encode,decode}_fp8_h{2,4}_gfx942`
// (the PM names without `_pm`), kernarg 0 at offset 0 and kernarg 1 at offset 8
// (encode: dst u8*, src f32*; decode: dst f32*, src u8*), no by-value args.
// One templated device body per direction serves both head counts.

#include <hip/hip_runtime.h>

#if defined(__FAST_MATH__) || defined(__HIP_FAST_MATH__)
#error "qsa_fp8_kv.gfx942.hip requires IEEE division: do not build with -ffast-math"
#endif
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx942__)
#error "qsa_fp8_kv.gfx942.hip is gfx942-only (FNUZ FP8 conversion instructions)"
#endif

#pragma clang fp contract(off)

#define QSA_FP8_THREADS 256
#define QSA_FP8_HEAD_DIM 256

// FNUZ-half code of OCP code(z): canonical zero (never 0x80 for finite input).
static __device__ __forceinline__ unsigned int qsa_fp8_ocp_code(float z) {
    const float half = z * 0.5f;  // exact for |z| <= 448 and for every z that rounds to nonzero
    unsigned int code = (unsigned int)__builtin_amdgcn_cvt_pk_fp8_f32(half, 0.0f, 0u, false) & 0xffu;
    // FNUZ 0x80 is NaN; for finite in-domain input it can only be a rounded -0.
    if (code == 0x80u && half == half && fabsf(half) <= 224.0f) code = 0u;
    return code;
}

// The head's F16 scale bits from its absolute maximum (route E1).
static __device__ __forceinline__ unsigned short qsa_fp8_scale_bits(float amax) {
    _Float16 s16 = amax == 0.0f ? (_Float16)1.0f : (_Float16)(amax / 448.0f);
    if (amax > 0.0f && s16 == (_Float16)0.0f) {
        s16 = __builtin_bit_cast(_Float16, (unsigned short)0x0001);
    }
    float sf = (float)s16;
    if (amax > 0.0f && 448.0f * sf < amax) {
        unsigned short bits = __builtin_bit_cast(unsigned short, s16);
        bits = bits >= 0x7bff ? (unsigned short)0x7bff : (unsigned short)(bits + 1u);
        s16 = __builtin_bit_cast(_Float16, bits);
    }
    return __builtin_bit_cast(unsigned short, s16);
}

// One channel of one K or V row of KV_HEADS heads x 256 channels. Every thread of
// the block calls it (block 256, thread = channel); it synchronizes.
template <int KV_HEADS>
static __device__ __forceinline__ void qsa_kv_store_fp8_gfx942(
    unsigned char* row,
    float value,
    int kv_head,
    int channel,
    float* amax_lds) {
    float amax = fabsf(value);
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }
    if ((threadIdx.x & 31) == 0) amax_lds[threadIdx.x >> 5] = amax;
    __syncthreads();
    amax = amax_lds[0];
    for (int group = 1; group < QSA_FP8_THREADS / 32; ++group) {
        amax = fmaxf(amax, amax_lds[group]);
    }
    __syncthreads();
    const unsigned short s16_bits = qsa_fp8_scale_bits(amax);
    const float sf = (float)__builtin_bit_cast(_Float16, s16_bits);
    row[kv_head * QSA_FP8_HEAD_DIM + channel] = (unsigned char)qsa_fp8_ocp_code(value / sf);
    if (channel == 0) {
        *reinterpret_cast<unsigned short*>(row + KV_HEADS * QSA_FP8_HEAD_DIM + 2 * kv_head) = s16_bits;
    }
}

// OCP E4M3FN byte -> F32, times the F16 row scale (route "Decode").
static __device__ __forceinline__ float qsa_kv_load_fp8_gfx942(unsigned int code, unsigned short s16_bits) {
    const float scale2 = 2.0f * (float)__builtin_bit_cast(_Float16, s16_bits);  // F32, never F16
    return __builtin_amdgcn_cvt_f32_fp8(code, 0) * scale2;
}

// dst rows: Hkv*256 + 2*Hkv bytes; src rows: Hkv*256 floats. Grid (KV_HEADS, rows).
template <int KV_HEADS>
static __device__ __forceinline__ void qsa_kv_encode_fp8_body(unsigned char* dst, const float* src) {
    __shared__ float amax_lds[QSA_FP8_THREADS / 32];
    const int kv_head = blockIdx.x;
    const long long row = blockIdx.y;
    const int channel = threadIdx.x;
    const long long row_bytes = (long long)KV_HEADS * (QSA_FP8_HEAD_DIM + 2);
    const float value = src[row * KV_HEADS * QSA_FP8_HEAD_DIM + (long long)kv_head * QSA_FP8_HEAD_DIM + channel];
    qsa_kv_store_fp8_gfx942<KV_HEADS>(dst + row * row_bytes, value, kv_head, channel, amax_lds);
}

// Dequantize rows written by the encoder (or by the gfx1201 writer) to F32.
// dst rows: Hkv*256 floats; src rows: Hkv*256 + 2*Hkv bytes. Grid (KV_HEADS, rows).
template <int KV_HEADS>
static __device__ __forceinline__ void qsa_kv_decode_fp8_body(float* dst, const unsigned char* src) {
    const int kv_head = blockIdx.x;
    const long long row = blockIdx.y;
    const int channel = threadIdx.x;
    const long long row_bytes = (long long)KV_HEADS * (QSA_FP8_HEAD_DIM + 2);
    const unsigned char* base = src + row * row_bytes;
    const unsigned short s16_bits =
        *reinterpret_cast<const unsigned short*>(base + KV_HEADS * QSA_FP8_HEAD_DIM + 2 * kv_head);
    const unsigned int code = base[kv_head * QSA_FP8_HEAD_DIM + channel];
    dst[row * KV_HEADS * QSA_FP8_HEAD_DIM + (long long)kv_head * QSA_FP8_HEAD_DIM + channel] =
        qsa_kv_load_fp8_gfx942(code, s16_bits);
}

extern "C" __global__ __launch_bounds__(QSA_FP8_THREADS) void qsa_kv_encode_fp8_h2_gfx942(
    unsigned char* __restrict__ dst, const float* __restrict__ src) {
    qsa_kv_encode_fp8_body<2>(dst, src);
}
extern "C" __global__ __launch_bounds__(QSA_FP8_THREADS) void qsa_kv_encode_fp8_h4_gfx942(
    unsigned char* __restrict__ dst, const float* __restrict__ src) {
    qsa_kv_encode_fp8_body<4>(dst, src);
}
extern "C" __global__ __launch_bounds__(QSA_FP8_THREADS) void qsa_kv_decode_fp8_h2_gfx942(
    float* __restrict__ dst, const unsigned char* __restrict__ src) {
    qsa_kv_decode_fp8_body<2>(dst, src);
}
extern "C" __global__ __launch_bounds__(QSA_FP8_THREADS) void qsa_kv_decode_fp8_h4_gfx942(
    float* __restrict__ dst, const unsigned char* __restrict__ src) {
    qsa_kv_decode_fp8_body<4>(dst, src);
}
