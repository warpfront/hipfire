// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — MQ4G128V2 (qt=53) dense batched projection.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Wire row layout: ceil(K/128) groups, exactly 68 bytes/group:
// [0..2) fp16 scale, [2..4) fp16 zero, [4..68) 64 packed nibbles.
// A is row-major [M,K] and X/Y are row-major [N,K]/[N,M].  Each block
// owns one output element (batch row, matrix row), matching the scalar qt53
// decoder while allowing all prompt rows to be resident in one launch.
__launch_bounds__(32, 16)
extern "C" __global__ void gemm_mq4g128v2_batched(
    const char* __restrict__ A,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= M || token >= N || K <= 0) return;

    const int groups = (K + 127) / 128;
    const long long row_stride = (long long)groups * 68;
    const char* row_ptr = A + (long long)row * row_stride;
    const float* x = X + (long long)token * K;
    float acc = 0.0f;

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
        const float x0 = i0 < K ? x[i0] : 0.0f;
        const float x1 = i1 < K ? x[i1] : 0.0f;
        const float x2 = i2 < K ? x[i2] : 0.0f;
        const float x3 = i3 < K ? x[i3] : 0.0f;
        acc += (scale * (float)(b0 & 0xF) + zero) * x0
            + (scale * (float)(b0 >> 4) + zero) * x1
            + (scale * (float)(b1 & 0xF) + zero) * x2
            + (scale * (float)(b1 >> 4) + zero) * x3;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if (tid == 0)
        Y[(long long)token * M + row] = acc;
}
