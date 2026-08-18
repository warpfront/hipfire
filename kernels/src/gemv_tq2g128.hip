// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// TQ2G128 decode GEMV: 34-byte blocks of 128 ternary weights.
// Layout per block: [0..2) fp16 scale d, [2..34) 32 code bytes, 4 codes each.
// Lane t owns code byte t, i.e. weights 4t..4t+3, so a wave covers one block.
//
// This is x-load and ALU bound, not weight-bandwidth bound: 128 weights ride in
// 32 bytes but pull 128 fp32 of x (512 bytes). So the wins here are (a) one
// float4 load instead of 4 scalar loads, (b) folding d in once per block rather
// than per weight, and (c) two independent accumulators so the two blocks in an
// iteration do not serialise on a single FMA chain.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_tq2g128(
    const char* __restrict__ A, const float* __restrict__ x,
    float* __restrict__ y, int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 34;
    const char* row_ptr = A + (long long)row * row_bytes;

    // x is 16B-aligned here: the offset g*128 + tid*4 is always a multiple of
    // 4 floats, so the float4 load is legal for every lane and every block.
    #define TQ2_BLOCK_DOT(gp, xoff, acc)                                    \
        do {                                                                \
            const float d = (float)(*(const _Float16*)(gp));                \
            const unsigned int b = *((const unsigned char*)((gp) + 2 + tid));\
            const float4 xv = *(const float4*)((x) + (xoff));               \
            (acc) += d * ( (float)((int)( b       & 0x3) - 1) * xv.x        \
                         + (float)((int)((b >> 2) & 0x3) - 1) * xv.y        \
                         + (float)((int)((b >> 4) & 0x3) - 1) * xv.z        \
                         + (float)((int)((b >> 6) & 0x3) - 1) * xv.w );     \
        } while (0)

    float acc0 = 0.0f, acc1 = 0.0f;
    int g = 0;
    const int xbase = tid * 4;
    for (; g + 2 <= groups_per_row; g += 2) {
        const char* gp0 = row_ptr + g * 34;
        TQ2_BLOCK_DOT(gp0,      g * 128 + xbase,       acc0);
        TQ2_BLOCK_DOT(gp0 + 34, (g + 1) * 128 + xbase, acc1);
    }
    for (; g < groups_per_row; g++) {
        TQ2_BLOCK_DOT(row_ptr + g * 34, g * 128 + xbase, acc0);
    }
    #undef TQ2_BLOCK_DOT

    float acc = acc0 + acc1;
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if (tid == 0) y[row] = acc;
}
