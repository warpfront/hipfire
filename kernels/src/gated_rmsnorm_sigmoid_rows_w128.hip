// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Per-head RMSNorm (plain gamma, FP32 statistics) times sigmoid(gate), one BF16 rounding at the end,
// for head_dim 128 (gfx1201):
//   out[r,h,:] = bf16( x * (1/sqrt(mean(x^2)+eps)) * gamma * sigmoid(gate) )
// Bit-identical to gated_rmsnorm_sigmoid_rows_f32 (the same expressions), but one wave32 owns one
// (row, head): lane l holds channels 4l..4l+3 (float4 loads/stores) and the sum of squares is the
// incumbent's LDS tree (s = 64, 32, ..., 1) evaluated as a wave-shuffle tree (stage s = 64/32/16/8/4 are lane
// offsets 16/8/4/2/1, the last two stages are in-lane), so there is no LDS and no barrier.
//
// Launch: block 256 (8 waves, one (row, head) each), grid ceil(n_rows * n_heads / 8); head index fastest.

#include <hip/hip_runtime.h>

static __device__ __forceinline__ float grw_bf16_rn(float x) {
    unsigned u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

static __device__ __forceinline__ float grw_tree128(float s0, float s1, float s2, float s3) {
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        s0 = __fadd_rn(s0, __shfl_down(s0, off, 32));
        s1 = __fadd_rn(s1, __shfl_down(s1, off, 32));
        s2 = __fadd_rn(s2, __shfl_down(s2, off, 32));
        s3 = __fadd_rn(s3, __shfl_down(s3, off, 32));
    }
    const float r = __fadd_rn(__fadd_rn(s0, s2), __fadd_rn(s1, s3));
    return __shfl(r, 0, 32);
}

extern "C" __global__ __launch_bounds__(256) void gated_rmsnorm_sigmoid_rows_w128_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ gate,
    const float* __restrict__ gamma,
    int head_dim,   // 128
    int n_heads,
    int n_rows,
    int x_row_stride,
    int gate_row_stride,
    int out_row_stride,
    float eps,
    int round_bf16
) {
    const int lane = threadIdx.x & 31;
    const int wid = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (wid >= n_rows * n_heads) return;
    const int row = wid / n_heads;
    const int h = wid - row * n_heads;
    const long long off = (long long)h * head_dim + lane * 4;
    const float4 xv = *reinterpret_cast<const float4*>(x + (long long)row * x_row_stride + off);
    const float4 z = *reinterpret_cast<const float4*>(gate + (long long)row * gate_row_stride + off);
    const float4 g = *reinterpret_cast<const float4*>(gamma + lane * 4);
    const float red = grw_tree128(__fmul_rn(xv.x, xv.x), __fmul_rn(xv.y, xv.y), __fmul_rn(xv.z, xv.z),
                                  __fmul_rn(xv.w, xv.w));
    const float inv = 1.0f / sqrtf(red / (float)head_dim + eps);
    const float sx = 1.0f / (1.0f + expf(-z.x));
    const float sy = 1.0f / (1.0f + expf(-z.y));
    const float sz = 1.0f / (1.0f + expf(-z.z));
    const float sw = 1.0f / (1.0f + expf(-z.w));
    float4 y;
    y.x = xv.x * inv * g.x * sx;
    y.y = xv.y * inv * g.y * sy;
    y.z = xv.z * inv * g.z * sz;
    y.w = xv.w * inv * g.w * sw;
    if (round_bf16) {
        y.x = grw_bf16_rn(y.x);
        y.y = grw_bf16_rn(y.y);
        y.z = grw_bf16_rn(y.z);
        y.w = grw_bf16_rn(y.w);
    }
    *reinterpret_cast<float4*>(out + (long long)row * out_row_stride + off) = y;
}
