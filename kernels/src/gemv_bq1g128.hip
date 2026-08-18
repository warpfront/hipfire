// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// BQ1G128 decode GEMV: 18-byte blocks of 128 binary weights.
// Layout per block: [0..2) fp16 scale d, [2..18) 16 code bytes, 8 codes each.
// Lane t owns nibble t of the code array (byte t>>1, low/high half by t&1), so
// two adjacent lanes share a byte and a wave covers one block.
//
// Do not read the GiB/s-of-weight-bytes figure as headroom here: a block ships
// 128 weights in 16 bytes but still pulls 128 fp32 of x (512 bytes), so this
// kernel is x-load and ALU bound. The sign is applied by XOR-ing straight into
// the float's sign bit, which avoids the per-weight select the branchy form
// generated -- that select, not the weight traffic, is why binary was slower
// than ternary in wall-clock at equal M,K despite reading half the bytes.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_bq1g128(
    const char* __restrict__ A, const float* __restrict__ x,
    float* __restrict__ y, int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 18;
    const char* row_ptr = A + (long long)row * row_bytes;

    const int byte_off = 2 + (tid >> 1);
    const int sh = (tid & 1) * 4;

    // bit==1 -> +v, bit==0 -> -v, with no branch: flip the sign bit when the
    // code bit is clear. ~b puts "is clear" in the low position for free.
    #define BQ1_SGN(nb, i, v) \
        __uint_as_float(__float_as_uint(v) ^ ((((nb) >> (i)) & 1u) << 31))

    #define BQ1_BLOCK_DOT(gp, xoff, acc)                                     \
        do {                                                                 \
            const float d = (float)(*(const _Float16*)(gp));                 \
            const unsigned int b = *((const unsigned char*)((gp) + byte_off));\
            const unsigned int nb = (~b) >> sh;                              \
            const float4 xv = *(const float4*)((x) + (xoff));                \
            (acc) += d * ( BQ1_SGN(nb, 0, xv.x) + BQ1_SGN(nb, 1, xv.y)       \
                         + BQ1_SGN(nb, 2, xv.z) + BQ1_SGN(nb, 3, xv.w) );    \
        } while (0)

    float acc0 = 0.0f, acc1 = 0.0f;
    int g = 0;
    const int xbase = tid * 4;
    for (; g + 2 <= groups_per_row; g += 2) {
        const char* gp0 = row_ptr + g * 18;
        BQ1_BLOCK_DOT(gp0,      g * 128 + xbase,       acc0);
        BQ1_BLOCK_DOT(gp0 + 18, (g + 1) * 128 + xbase, acc1);
    }
    for (; g < groups_per_row; g++) {
        BQ1_BLOCK_DOT(row_ptr + g * 18, g * 128 + xbase, acc0);
    }
    #undef BQ1_BLOCK_DOT
    #undef BQ1_SGN

    float acc = acc0 + acc1;
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if (tid == 0) y[row] = acc;
}
