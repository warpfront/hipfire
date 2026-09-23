// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef HIPFIRE_HFQ6_DP4A_RESIDUAL
#define HIPFIRE_HFQ6_DP4A_RESIDUAL 0
#endif

#define QK8_1 32

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

#if HIPFIRE_HFQ6_DP4A_RESIDUAL
#define HIPFIRE_KERNEL_NAME gemv_hfq6g256_residual_wave64_dp4a_gfx12
#else
#define HIPFIRE_KERNEL_NAME gemv_hfq6g256_wave64_dp4a_gfx12
#endif

extern "C" __global__ __launch_bounds__(64, 10)
void HIPFIRE_KERNEL_NAME(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ y,
    int M,
    int K
) {
    const int lane64 = threadIdx.x;
    const int lane = lane64 & 31;
    const int row = blockIdx.x * 2 + (lane64 >> 5);
    if (row >= M) return;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 200;
    const int byte_off = lane * 6;
    const int sub_in_block = (lane / 4) & 3;
    const int block_within_group = lane >> 4;
    const int qs_byte_off = 8 * (lane & 15);

    float acc = 0.0f;
    for (int g = 0; g < groups_per_row; ++g) {
        const char* gp = row_ptr + (long long)g * 200;
        const float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned char* d = (const unsigned char*)(gp + 8);
        const unsigned int b0 = d[byte_off];
        const unsigned int b1 = d[byte_off + 1];
        const unsigned int b2 = d[byte_off + 2];
        const unsigned int b3 = d[byte_off + 3];
        const unsigned int b4 = d[byte_off + 4];
        const unsigned int b5 = d[byte_off + 5];
        const unsigned int q0 = b0 & 63;
        const unsigned int q1 = (b0 >> 6) | ((b1 & 15) << 2);
        const unsigned int q2 = (b1 >> 4) | ((b2 & 3) << 4);
        const unsigned int q3 = b2 >> 2;
        const unsigned int q4 = b3 & 63;
        const unsigned int q5 = (b3 >> 6) | ((b4 & 15) << 2);
        const unsigned int q6 = (b4 >> 4) | ((b5 & 3) << 4);
        const unsigned int q7 = b5 >> 2;
        const int qa = (int)(q0 | (q1 << 8) | (q2 << 16) | (q3 << 24));
        const int qb = (int)(q4 | (q5 << 8) | (q6 << 16) | (q7 << 24));

        const block_q8_1_mmq* xb = Xq + 2 * g + block_within_group;
        const int* xqs = (const int*)(xb->qs + qs_byte_off);
        int dot = 0;
        dot = __builtin_amdgcn_sudot4(true, qa, true, xqs[0], dot, false);
        dot = __builtin_amdgcn_sudot4(true, qb, true, xqs[1], dot, false);
        const float2 ds = __half22float2(xb->ds4[sub_in_block]);
        acc += sc * ds.x * (float)dot + zp * ds.y * 0.25f;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (lane == 0) {
#if HIPFIRE_HFQ6_DP4A_RESIDUAL
        y[row] += acc;
#else
        y[row] = acc;
#endif
    }
}

#undef HIPFIRE_KERNEL_NAME
