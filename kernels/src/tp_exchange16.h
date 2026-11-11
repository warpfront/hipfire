// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// 16-bit partial-sum exchange codec for the dense tensor-parallel residual
// reduction (`HIPFIRE_TP_EXCHANGE`).
//
// A rank converts its f32 row-parallel partial to 16 bits (bf16 or f16) once,
// peers read the 16-bit copy, and every rank sums the rank-ordered 16-bit
// partials in f32 into its f32 residual. EVERY rank, the producer included,
// sums the rounded value of every partial (`up(pack(p))`), so all ranks add
// the same bytes and the residual stream stays replicated bit for bit.
//
// FMT 0 = bf16 (round-to-nearest-even, NaN canonicalised to a quiet bf16 NaN,
//             +-Inf kept, range of f32).
// FMT 1 = f16  (round-to-nearest-even via __float2half; |v| > 65504 becomes
//             +-Inf, which poisons the residual: the caller owns that
//             overflow check, see docs/multi-gpu.md).
//
// Include-guarded: the kernels are JIT-compiled from concatenated sources.

#ifndef HF_TP_EXCHANGE16_H
#define HF_TP_EXCHANGE16_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define HF_TP16_BF16 0
#define HF_TP16_F16 1

template <int FMT> __device__ __forceinline__ uint16_t tp16_pack(float v);

template <> __device__ __forceinline__ uint16_t tp16_pack<HF_TP16_BF16>(float v) {
    const uint32_t bits = __builtin_bit_cast(uint32_t, v);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return (uint16_t)(((bits >> 16) & 0x8000u) | 0x7fc0u);
    }
    // RNE on the 16 discarded bits; Inf (mantissa 0) stays Inf.
    const uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(rounded >> 16);
}

template <> __device__ __forceinline__ uint16_t tp16_pack<HF_TP16_F16>(float v) {
    return (uint16_t)__half_as_ushort(__float2half(v));
}

template <int FMT> __device__ __forceinline__ float tp16_unpack(uint16_t h);

template <> __device__ __forceinline__ float tp16_unpack<HF_TP16_BF16>(uint16_t h) {
    return __builtin_bit_cast(float, ((uint32_t)h) << 16);
}

template <> __device__ __forceinline__ float tp16_unpack<HF_TP16_F16>(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

// The value every rank adds for a partial `v`: its 16-bit round trip.
template <int FMT> __device__ __forceinline__ float tp16_round(float v) {
    return tp16_unpack<FMT>(tp16_pack<FMT>(v));
}

// Four packed 16-bit lanes in one 64-bit word (lane 0 in the low half).
template <int FMT> __device__ __forceinline__ unsigned long long tp16_pack4(const float4 &v) {
    const unsigned long long a = tp16_pack<FMT>(v.x);
    const unsigned long long b = tp16_pack<FMT>(v.y);
    const unsigned long long c = tp16_pack<FMT>(v.z);
    const unsigned long long d = tp16_pack<FMT>(v.w);
    return a | (b << 16) | (c << 32) | (d << 48);
}

template <int FMT> __device__ __forceinline__ float4 tp16_unpack4(unsigned long long w) {
    float4 r;
    r.x = tp16_unpack<FMT>((uint16_t)(w & 0xffffu));
    r.y = tp16_unpack<FMT>((uint16_t)((w >> 16) & 0xffffu));
    r.z = tp16_unpack<FMT>((uint16_t)((w >> 32) & 0xffffu));
    r.w = tp16_unpack<FMT>((uint16_t)((w >> 48) & 0xffffu));
    return r;
}

#endif  // HF_TP_EXCHANGE16_H
