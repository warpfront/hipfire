// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — gfx1151 MQ4G128V2 (qt=53) dense four-row projection.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Wire row layout: ceil(K/128) groups, exactly 68 bytes/group:
// [0..2) fp16 scale, [2..4) fp16 zero, [4..68) 64 packed nibbles.
// A is row-major [M,K] and X/Y are row-major [N,K]/[N,M].  Each block
// owns one output row and four prompt rows.  The weight decode is shared by
// those four rows, while each row keeps the scalar gemv accumulator and
// reduction association unchanged.
// gfx11 wave32 ISA; tuned on gfx1151 (ArchCaps::qwen4_tuned_routes admits it).
#if defined(__GFX11__)
__launch_bounds__(32, 16)
extern "C" __global__ void gemm_mq4g128v2_multirow_gfx1151(
    const char* __restrict__ A,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int row = blockIdx.x;
    const int tile = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= M || tile * 4 >= N || K <= 0) return;

    const int token0 = tile * 4;
    const int token1 = token0 + 1;
    const int token2 = token0 + 2;
    const int token3 = token0 + 3;
    const float* x0 = X + (long long)token0 * K;
    const float* x1 = token1 < N ? X + (long long)token1 * K : x0;
    const float* x2 = token2 < N ? X + (long long)token2 * K : x0;
    const float* x3 = token3 < N ? X + (long long)token3 * K : x0;

    const int groups = (K + 127) / 128;
    const long long row_stride = (long long)groups * 68;
    const char* row_ptr = A + (long long)row * row_stride;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (int group = 0; group < groups; ++group) {
        const char* gp = row_ptr + (long long)group * 68;
        const unsigned short scale_bits = *(const unsigned short*)(gp + 0);
        const unsigned short zero_bits = *(const unsigned short*)(gp + 2);
        const float scale = __half2float(__ushort_as_half(scale_bits));
        const float zero = __half2float(__ushort_as_half(zero_bits));
        const unsigned char* payload = (const unsigned char*)(gp + 4);
        const int base = group * 128 + tid * 4;
        const unsigned char b0 = payload[tid * 2];
        const unsigned char b1 = payload[tid * 2 + 1];
        const int i0 = base;
        const int i1 = base + 1;
        const int i2 = base + 2;
        const int i3 = base + 3;
        const float x0_0 = i0 < K ? x0[i0] : 0.0f;
        const float x0_1 = i1 < K ? x0[i1] : 0.0f;
        const float x0_2 = i2 < K ? x0[i2] : 0.0f;
        const float x0_3 = i3 < K ? x0[i3] : 0.0f;
        const float x1_0 = i0 < K ? x1[i0] : 0.0f;
        const float x1_1 = i1 < K ? x1[i1] : 0.0f;
        const float x1_2 = i2 < K ? x1[i2] : 0.0f;
        const float x1_3 = i3 < K ? x1[i3] : 0.0f;
        const float x2_0 = i0 < K ? x2[i0] : 0.0f;
        const float x2_1 = i1 < K ? x2[i1] : 0.0f;
        const float x2_2 = i2 < K ? x2[i2] : 0.0f;
        const float x2_3 = i3 < K ? x2[i3] : 0.0f;
        const float x3_0 = i0 < K ? x3[i0] : 0.0f;
        const float x3_1 = i1 < K ? x3[i1] : 0.0f;
        const float x3_2 = i2 < K ? x3[i2] : 0.0f;
        const float x3_3 = i3 < K ? x3[i3] : 0.0f;
        acc0 += (scale * (float)(b0 & 0xF) + zero) * x0_0
            + (scale * (float)(b0 >> 4) + zero) * x0_1
            + (scale * (float)(b1 & 0xF) + zero) * x0_2
            + (scale * (float)(b1 >> 4) + zero) * x0_3;
        acc1 += (scale * (float)(b0 & 0xF) + zero) * x1_0
            + (scale * (float)(b0 >> 4) + zero) * x1_1
            + (scale * (float)(b1 & 0xF) + zero) * x1_2
            + (scale * (float)(b1 >> 4) + zero) * x1_3;
        acc2 += (scale * (float)(b0 & 0xF) + zero) * x2_0
            + (scale * (float)(b0 >> 4) + zero) * x2_1
            + (scale * (float)(b1 & 0xF) + zero) * x2_2
            + (scale * (float)(b1 >> 4) + zero) * x2_3;
        acc3 += (scale * (float)(b0 & 0xF) + zero) * x3_0
            + (scale * (float)(b0 >> 4) + zero) * x3_1
            + (scale * (float)(b1 & 0xF) + zero) * x3_2
            + (scale * (float)(b1 >> 4) + zero) * x3_3;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down(acc0, offset);
        acc1 += __shfl_down(acc1, offset);
        acc2 += __shfl_down(acc2, offset);
        acc3 += __shfl_down(acc3, offset);
    }
    if (tid == 0) {
        Y[(long long)token0 * M + row] = acc0;
        if (token1 < N) Y[(long long)token1 * M + row] = acc1;
        if (token2 < N) Y[(long long)token2 * M + row] = acc2;
        if (token3 < N) Y[(long long)token3 * M + row] = acc3;
    }
}
#endif
