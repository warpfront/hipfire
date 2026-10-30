// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Stream broadcast: one row of width `width` is replicated into every one of
// `n_streams` residual streams.
//
//   dst[r, s, c] = round(src[r * src_stride + c])      for all s < n_streams
//
// `bf16 != 0` rounds the value to bfloat16 (round-to-nearest-even) first; the
// result stays an F32 value. Streams are an F32 plane that carries model-dtype
// values.
//
// Grid:  [ceil(width / 256), rows, 1]
// Block: [256, 1, 1]

#include <hip/hip_runtime.h>

__device__ __forceinline__ float sm_round_bf16(float x) {
    unsigned int u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;  // NaN keeps payload
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

extern "C" __launch_bounds__(256)
__global__ void stream_mix_broadcast(
    const float* __restrict__ src,   // [rows, src_stride]
    float*       __restrict__ dst,   // [rows, n_streams, width]
    int n_streams,
    int width,
    int src_stride,
    int bf16
) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int r = blockIdx.y;
    if (c >= width) return;
    float v = src[(size_t)r * src_stride + c];
    if (bf16) v = sm_round_bf16(v);
    float* d = dst + (size_t)r * n_streams * width + c;
    for (int s = 0; s < n_streams; ++s) d[(size_t)s * width] = v;
}
