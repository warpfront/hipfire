// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Per-head RMSNorm (plain gamma, FP32 statistics) times sigmoid(gate), one
// BF16 rounding at the end:
//   out[r,h,:] = bf16( x * (1/sqrt(mean(x^2)+eps)) * gamma * sigmoid(gate) )
// One workgroup per (row, head); blockDim.x == head_dim (<= 1024, multiple of 32).
// Fixed-order LDS tree reduction => deterministic.

#include <hip/hip_runtime.h>

static __device__ __forceinline__ float grs_bf16_rn(float x) {
    unsigned u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

extern "C" __global__ void gated_rmsnorm_sigmoid_rows_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ gate,
    const float* __restrict__ gamma,
    int head_dim,
    int n_heads,
    int x_row_stride,
    int gate_row_stride,
    int out_row_stride,
    float eps,
    int round_bf16
) {
    __shared__ float red[1024];
    const int h = blockIdx.x;
    const int row = blockIdx.y;
    const int d = threadIdx.x;
    const long long off = (long long)h * head_dim + d;
    const float xv = x[(long long)row * x_row_stride + off];
    red[d] = xv * xv;
    __syncthreads();
    for (int s = head_dim >> 1; s > 0; s >>= 1) {
        if (d < s) red[d] += red[d + s];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(red[0] / (float)head_dim + eps);
    const float z = gate[(long long)row * gate_row_stride + off];
    const float sg = 1.0f / (1.0f + expf(-z));
    const float y = xv * inv * gamma[d] * sg;
    out[(long long)row * out_row_stride + off] = round_bf16 ? grs_bf16_rn(y) : y;
}
