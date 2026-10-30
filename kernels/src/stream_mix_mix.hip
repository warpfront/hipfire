// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Stream postmix: residual mixing plus branch write-back, into a SEPARATE
// output buffer (ping-pong; the old streams are read by every output stream).
//
//   out[i, c] = round(sum_j res[i, j] * x[j, c] + post[i] * branch[c])
//
// res is [n, n] destination/source. Every control comes from the OLD streams
// (computed before the branch ran). `round` is bfloat16 round-to-nearest-even
// when `bf16 != 0`. Supported n <= 8.
//
// Grid:  [ceil(width / 256), rows, 1]
// Block: [256, 1, 1]

#include <hip/hip_runtime.h>

#define SM_THREADS 256
#define SM_MAX_N 8

__device__ __forceinline__ float sm_round_bf16(float x) {
    unsigned int u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

extern "C" __launch_bounds__(SM_THREADS)
__global__ void stream_mix_postmix(
    const float* __restrict__ x,       // [rows, n, width] old streams
    const float* __restrict__ branch,  // [rows, width]
    const float* __restrict__ post,    // [rows, n]
    const float* __restrict__ res,     // [rows, n, n]
    float*       __restrict__ out,     // [rows, n, width]
    int n,
    int width,
    int bf16
) {
    const int c = blockIdx.x * SM_THREADS + threadIdx.x;
    const int row = blockIdx.y;
    if (c >= width) return;

    const float* xr = x + (size_t)row * n * width + c;
    float xv[SM_MAX_N];
#pragma unroll
    for (int j = 0; j < SM_MAX_N; ++j) xv[j] = (j < n) ? xr[(size_t)j * width] : 0.0f;
    const float bv = branch[(size_t)row * width + c];
    const float* rr = res + (size_t)row * n * n;
    const float* pr = post + (size_t)row * n;
    float* o = out + (size_t)row * n * width + c;

#pragma unroll
    for (int i = 0; i < SM_MAX_N; ++i) {
        if (i < n) {
            float a = 0.0f;
#pragma unroll
            for (int j = 0; j < SM_MAX_N; ++j)
                if (j < n) a += rr[i * n + j] * xv[j];
            a += pr[i] * bv;
            if (bf16) a = sm_round_bf16(a);
            o[(size_t)i * width] = a;
        }
    }
}
