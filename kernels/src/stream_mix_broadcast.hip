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

// BF16-stream broadcast: dst holds BF16 bits. The value is rounded to bfloat16
// (round-to-nearest-even) exactly as above and stored as its high half, so the
// stored stream is the bit-exact BF16 image of the F32-plane result. Requires
// bf16 != 0 (the launcher refuses otherwise).
extern "C" __launch_bounds__(256)
__global__ void stream_mix_broadcast_bf16(
    const float* __restrict__ src,        // [rows, src_stride]
    unsigned short* __restrict__ dst,     // [rows, n_streams, width] BF16 bits
    int n_streams,
    int width,
    int src_stride,
    int bf16
) {
    (void)bf16;
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int r = blockIdx.y;
    if (c >= width) return;
    const float v = sm_round_bf16(src[(size_t)r * src_stride + c]);
    unsigned int u = __float_as_uint(v);
    unsigned int h = u >> 16;
    if ((u & 0x7fffffffu) > 0x7f800000u) h |= 0x0040u;  // keep a NaN a NaN
    unsigned short* d = dst + (size_t)r * n_streams * width + c;
    for (int s = 0; s < n_streams; ++s) d[(size_t)s * width] = (unsigned short)h;
}

// Exact BF16 -> F32 widening of `rows` rows of `width` columns into a strided
// F32 destination (taps, DFlash capture): dst[r * dst_stride + dst_col + c].
extern "C" __launch_bounds__(256)
__global__ void stream_mix_widen_rows(
    const unsigned short* __restrict__ src,  // [rows, width] BF16 bits
    float* __restrict__ dst,
    int width,
    int dst_stride,
    int dst_col
) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int r = blockIdx.y;
    if (c >= width) return;
    dst[(size_t)r * dst_stride + dst_col + c] =
        __uint_as_float(((unsigned int)src[(size_t)r * width + c]) << 16);
}
