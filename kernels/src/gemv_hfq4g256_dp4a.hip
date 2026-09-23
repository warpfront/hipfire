// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef HIPFIRE_LM_DP4A_KERNEL
#define HIPFIRE_LM_DP4A_KERNEL gemv_hfq4g256_dp4a
#endif
#ifndef HIPFIRE_LM_DP4A_ROWS_PER_BLOCK
#define HIPFIRE_LM_DP4A_ROWS_PER_BLOCK 1
#endif

struct block_q8_1_mmq { half2 ds4[4]; int8_t qs[128]; };
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

extern "C" __global__ void HIPFIRE_LM_DP4A_KERNEL(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ y,
    int M,
    int K
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * HIPFIRE_LM_DP4A_ROWS_PER_BLOCK + warp_id;
    if (row >= M) return;
    const int groups = K / 256;
    const char* row_ptr = A + (long long)row * groups * 136;
    const int boff = lane * 4;
    const int sub = (lane >> 2) & 3;
    const int half = lane >> 4;
    const int qoff = 8 * (lane & 15);
    float acc = 0.0f;
#ifdef HIPFIRE_LM_DP4A_UNROLL2
#pragma unroll 2
#endif
    for (int g = 0; g < groups; ++g) {
        const char* gp = row_ptr + (long long)g * 136;
        const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        const unsigned int even = pk & 0x0F0F0F0Fu;
        const unsigned int odd = (pk >> 4) & 0x0F0F0F0Fu;
        const int wa = (int)__builtin_amdgcn_perm(odd, even, 0x05010400u);
        const int wb = (int)__builtin_amdgcn_perm(odd, even, 0x07030602u);
        const block_q8_1_mmq* xb = Xq + 2 * g + half;
        const int* xqs = reinterpret_cast<const int*>(xb->qs + qoff);
        int sum = __builtin_amdgcn_sudot4(false, wa, true, xqs[0], 0, false);
        sum = __builtin_amdgcn_sudot4(false, wb, true, xqs[1], sum, false);
        const float2 ds = __half22float2(xb->ds4[sub]);
        acc += sc * ds.x * (float)sum + zp * ds.y * 0.25f;
    }
    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) y[row] = acc;
}
