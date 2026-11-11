// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Dense TP residual reduction with a 16-bit partial exchange
// (`HIPFIRE_TP_EXCHANGE=bf16|f16`, see tp_exchange16.h for the codec).
//
// `tp_pack16_*`: the producing rank converts its f32 partial to 16 bits into a
// buffer its peers read. `tp_fold16_*`: the consuming rank sums rank-ordered
// 16-bit partials in f32 and adds the sum to its f32 residual:
//
//     sum = r(p_0) + r(p_1) + ... + r(p_{n-1})      (left-associated, rank order)
//     x   = x + sum
//
// where r(p) is the 16-bit round trip. The producer's own term is either the
// rank's packed copy (`s_q`) or, for a rank that never publishes its own
// partial (the root of a rooted reduction), its f32 partial rounded in
// register (`first_f32`): both are the same bytes. `first_f32` is term 0, the
// `nsrc` 16-bit sources follow in order. `sum_out` (optional) receives the
// sum; `x` (optional) the residual add. All adds are plain f32 adds with no
// contraction, so every rank that folds the same operands gets the same bytes.
//
// With nsrc == 2 and no first_f32 this is exactly `x + (r(p0) + r(p1))`, the
// TP2 direct-peer form; the 16-bit sources may be P2P-mapped peer buffers.

#include <hip/hip_runtime.h>
#ifndef HF_TP_EXCHANGE16_H
#include "tp_exchange16.h"
#endif

template <int FMT>
__device__ __forceinline__ void tp_pack16_body(const float *__restrict__ src,
                                               uint16_t *__restrict__ dst, int count) {
    const int vec = count >> 2;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const float4 *__restrict__ sv = reinterpret_cast<const float4 *>(src);
    unsigned long long *__restrict__ dv = reinterpret_cast<unsigned long long *>(dst);
    for (int i = tid; i < vec; i += stride) {
        dv[i] = tp16_pack4<FMT>(sv[i]);
    }
    for (int j = (vec << 2) + tid; j < count; j += stride) {
        dst[j] = tp16_pack<FMT>(src[j]);
    }
}

template <int FMT>
__device__ __forceinline__ void tp_fold16_body(float *sum_out, float *x, const float *first_f32,
                                               const uint16_t *s0, const uint16_t *s1,
                                               const uint16_t *s2, const uint16_t *s3, int nsrc,
                                               int count) {
    const uint16_t *srcs[4] = {s0, s1, s2, s3};
    const int vec = count >> 2;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int q0 = first_f32 ? 0 : 1;
    for (int i = tid; i < vec; i += stride) {
        float4 acc;
        if (first_f32) {
            const float4 f = reinterpret_cast<const float4 *>(first_f32)[i];
            acc.x = tp16_round<FMT>(f.x);
            acc.y = tp16_round<FMT>(f.y);
            acc.z = tp16_round<FMT>(f.z);
            acc.w = tp16_round<FMT>(f.w);
        } else {
            acc = tp16_unpack4<FMT>(reinterpret_cast<const unsigned long long *>(srcs[0])[i]);
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (q >= q0 && q < nsrc) {
                const float4 b =
                    tp16_unpack4<FMT>(reinterpret_cast<const unsigned long long *>(srcs[q])[i]);
                acc.x = acc.x + b.x;
                acc.y = acc.y + b.y;
                acc.z = acc.z + b.z;
                acc.w = acc.w + b.w;
            }
        }
        if (sum_out) {
            reinterpret_cast<float4 *>(sum_out)[i] = acc;
        }
        if (x) {
            float4 xi = reinterpret_cast<float4 *>(x)[i];
            xi.x = xi.x + acc.x;
            xi.y = xi.y + acc.y;
            xi.z = xi.z + acc.z;
            xi.w = xi.w + acc.w;
            reinterpret_cast<float4 *>(x)[i] = xi;
        }
    }
    // Scalar tail from the original thread id (`i` has advanced for threads
    // that ran the vector loop).
    for (int j = (vec << 2) + tid; j < count; j += stride) {
        float acc = first_f32 ? tp16_round<FMT>(first_f32[j]) : tp16_unpack<FMT>(srcs[0][j]);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (q >= q0 && q < nsrc) {
                acc = acc + tp16_unpack<FMT>(srcs[q][j]);
            }
        }
        if (sum_out) {
            sum_out[j] = acc;
        }
        if (x) {
            x[j] = x[j] + acc;
        }
    }
}

extern "C" __global__ void tp_pack16_bf16(const float *__restrict__ src,
                                          uint16_t *__restrict__ dst, int count) {
    tp_pack16_body<HF_TP16_BF16>(src, dst, count);
}

extern "C" __global__ void tp_pack16_f16(const float *__restrict__ src,
                                         uint16_t *__restrict__ dst, int count) {
    tp_pack16_body<HF_TP16_F16>(src, dst, count);
}

extern "C" __global__ void tp_fold16_bf16(float *sum_out, float *x, const float *first_f32,
                                          const uint16_t *s0, const uint16_t *s1,
                                          const uint16_t *s2, const uint16_t *s3, int nsrc,
                                          int count) {
    tp_fold16_body<HF_TP16_BF16>(sum_out, x, first_f32, s0, s1, s2, s3, nsrc, count);
}

extern "C" __global__ void tp_fold16_f16(float *sum_out, float *x, const float *first_f32,
                                         const uint16_t *s0, const uint16_t *s1,
                                         const uint16_t *s2, const uint16_t *s3, int nsrc,
                                         int count) {
    tp_fold16_body<HF_TP16_F16>(sum_out, x, first_f32, s0, s1, s2, s3, nsrc, count);
}
