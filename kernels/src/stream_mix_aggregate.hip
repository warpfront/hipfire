// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Stream aggregation: weighted sum of the n residual streams into one branch
// input row, optionally followed by a plain-gamma RMSNorm.
//
//   agg[c] = round(sum_s pre[s] * x[s, c])                  (F32 accumulate)
//   out[c] = round(agg[c] * rsqrt(mean(agg^2) + eps) * gamma[c])   (has_norm)
//   out[c] = agg[c]                                                (no norm)
//
// `round` is bfloat16 round-to-nearest-even when `bf16 != 0`, identity
// otherwise. Gamma is applied plainly (no 1 + gamma); `gamma_kind` 0 = F32,
// 1 = BF16 bits. One workgroup per row; the rounded aggregate lives in LDS
// (dynamic, width * 4 bytes) between the sum and the norm.
//
// Grid:  [rows, 1, 1]
// Block: [256, 1, 1]
// Dynamic LDS: width * 4 bytes

#include <hip/hip_runtime.h>

#define SM_THREADS 256

__device__ __forceinline__ float sm_round_bf16(float x) {
    unsigned int u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

extern "C" __launch_bounds__(SM_THREADS)
__global__ void stream_mix_aggregate(
    const float* __restrict__ x,      // [rows, n, width] old streams
    const float* __restrict__ pre,    // [rows, n]
    const void*  __restrict__ gamma,  // [width] or null
    float*       __restrict__ out,    // [rows, width]
    int n,
    int width,
    float eps,
    int bf16,
    int has_norm,
    int gamma_kind
) {
    extern __shared__ float agg[];
    __shared__ float red[SM_THREADS];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* xr = x + (size_t)row * n * width;
    const float* pr = pre + (size_t)row * n;

    float ss = 0.0f;
    for (int c = tid; c < width; c += SM_THREADS) {
        float a = 0.0f;
        for (int s = 0; s < n; ++s) a += pr[s] * xr[(size_t)s * width + c];
        if (bf16) a = sm_round_bf16(a);
        agg[c] = a;
        ss += a * a;
    }

    float r = 1.0f;
    if (has_norm) {
        red[tid] = ss;
        __syncthreads();
        for (int off = SM_THREADS / 2; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        r = rsqrtf(red[0] / (float)width + eps);
    }

    float* o = out + (size_t)row * width;
    for (int c = tid; c < width; c += SM_THREADS) {
        float v = agg[c];
        if (has_norm) {
            float g;
            if (gamma_kind == 1) {
                g = __uint_as_float(((unsigned int)((const unsigned short*)gamma)[c]) << 16);
            } else {
                g = ((const float*)gamma)[c];
            }
            v = v * r * g;
            if (bf16) v = sm_round_bf16(v);
        }
        o[c] = v;
    }
}
