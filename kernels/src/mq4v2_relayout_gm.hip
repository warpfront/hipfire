// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniil Markevich
// hipfire — see LICENSE and NOTICE in the project root.
//
// MQ4G256V2 (qt=44) row-major -> group-major relayout for the IU4 prefill
// `_gm_col_gfx1151` entries.
//
// Offset contract (mirrors `load_iu4_tile_gm` in
// gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip; 136 B/group, gpr = K/256):
//   row-major   byte offset of group (row, kb): ((row)*gpr + kb)*136
//   group-major byte offset of group (row, kb): (kb*M + row)*136
// so each 128-row x 1-group tile is one contiguous 128*136 B block.
//
// One thread per group; the 136 B group copies as 34 u32 words. Bit-exact by
// construction: the same bytes land in LDS, only the address changes.
// Decode GEMV, M/N tails, and other archs keep reading the row-major tensor.

#include <hip/hip_runtime.h>

extern "C" __global__ void mq4v2_relayout_gm(
    const char* __restrict__ src,
    char* __restrict__ dst,
    int M,
    int GPR)
{
    size_t gid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)M * (size_t)GPR;
    if (gid >= total) {
        return;
    }
    size_t kb = gid / (size_t)M;
    size_t row = gid % (size_t)M;
    const unsigned int* s =
        (const unsigned int*)(src + (row * (size_t)GPR + kb) * 136u);
    unsigned int* d =
        (unsigned int*)(dst + (kb * (size_t)M + row) * 136u);
#pragma unroll
    for (int i = 0; i < 34; ++i) {
        d[i] = s[i];
    }
}
