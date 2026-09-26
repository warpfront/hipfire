// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// One warp per weight row, R activation rows per warp. x is pre-rotated F32
// [R,K], y is F32 [R,M]. Each group's header and packed bytes are read and
// decoded once; each activation row keeps the scalar GEMV's own four sums.
template<int R>
__device__ __forceinline__ void mq6g256v2_f32_rows_group(
    const char* gp, int boff, int tid, const float* x, int K, int base,
    float (&acc)[R]
) {
    const unsigned int hA = *(const unsigned int*)(gp);
    const unsigned int hB = *(const unsigned int*)(gp + 4);
    const unsigned int hs = (tid < 16) ? hA : hB;
    float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
    float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
    const unsigned char* d = (const unsigned char*)(gp + 8 + boff);
    unsigned char b0 = d[0]; unsigned char b1 = d[1]; unsigned char b2 = d[2];
    unsigned char b3 = d[3]; unsigned char b4 = d[4]; unsigned char b5 = d[5];
    int q0 = b0 & 63;
    int q1 = (b0 >> 6) | ((b1 & 0xF) << 2);
    int q2 = (b1 >> 4) | ((b2 & 3) << 4);
    int q3 = b2 >> 2;
    int q4 = b3 & 63;
    int q5 = (b3 >> 6) | ((b4 & 0xF) << 2);
    int q6 = (b4 >> 4) | ((b5 & 3) << 4);
    int q7 = b5 >> 2;
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const float* xr = x + (long long)r * K;
        acc[r] += (sc * (float)q0 + zp) * xr[base]
                + (sc * (float)q1 + zp) * xr[base + 1]
                + (sc * (float)q2 + zp) * xr[base + 2]
                + (sc * (float)q3 + zp) * xr[base + 3]
                + (sc * (float)q4 + zp) * xr[base + 4]
                + (sc * (float)q5 + zp) * xr[base + 5]
                + (sc * (float)q6 + zp) * xr[base + 6]
                + (sc * (float)q7 + zp) * xr[base + 7];
    }
}

template<int R>
__device__ __forceinline__ void mq6g256v2_f32_rows_body(
    const char* __restrict__ A, const float* __restrict__ x,
    float* __restrict__ y, int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 200;
    const int boff = tid * 6;
    float acc0[R], acc1[R], acc2[R], acc3[R];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        acc0[r] = acc1[r] = acc2[r] = acc3[r] = 0.0f;

    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;
    for (int q = 0; q < quads; ++q) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 200;
        const int base = g * 256 + tid * 8;
        mq6g256v2_f32_rows_group<R>(gp0,       boff, tid, x, K, base,       acc0);
        mq6g256v2_f32_rows_group<R>(gp0 + 200, boff, tid, x, K, base + 256, acc1);
        mq6g256v2_f32_rows_group<R>(gp0 + 400, boff, tid, x, K, base + 512, acc2);
        mq6g256v2_f32_rows_group<R>(gp0 + 600, boff, tid, x, K, base + 768, acc3);
    }
    const int g = quads << 2;
    const char* gp = row_ptr + g * 200;
    const int base = g * 256 + tid * 8;
    if (tail >= 1)
        mq6g256v2_f32_rows_group<R>(gp,       boff, tid, x, K, base,       acc0);
    if (tail >= 2)
        mq6g256v2_f32_rows_group<R>(gp + 200, boff, tid, x, K, base + 256, acc1);
    if (tail >= 3)
        mq6g256v2_f32_rows_group<R>(gp + 400, boff, tid, x, K, base + 512, acc2);

    #pragma unroll
    for (int r = 0; r < R; ++r) {
        float acc = (acc0[r] + acc1[r]) + (acc2[r] + acc3[r]);
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (tid == 0) y[(long long)r * M + row] = acc;
    }
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_mq6g256v2_f32_rows_r2(
    const char* A, const float* x, float* y, int M, int K
) { mq6g256v2_f32_rows_body<2>(A, x, y, M, K); }

__launch_bounds__(32, 12)
extern "C" __global__ void gemm_mq6g256v2_f32_rows_r3(
    const char* A, const float* x, float* y, int M, int K
) { mq6g256v2_f32_rows_body<3>(A, x, y, M, K); }

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_mq6g256v2_f32_rows_r4(
    const char* A, const float* x, float* y, int M, int K
) { mq6g256v2_f32_rows_body<4>(A, x, y, M, K); }
