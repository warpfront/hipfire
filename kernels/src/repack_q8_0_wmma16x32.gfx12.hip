// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) block_q8_0 {
    _Float16 d;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "bad block_q8_0 size");

// Convert row-major Q8_0 to the exact lane-major fragment consumed by gfx12
// wave32 integer WMMA.  A 16-row x 32-K tile remains 544 bytes:
//   16 half scales + two (32 lanes x 8-byte) WMMA A fragments.
// This is a one-time, model-scoped transform; it is never part of replay.
extern "C" __global__ __launch_bounds__(32, 4)
void repack_q8_0_wmma16x32_gfx12(
    const block_q8_0* __restrict__ src,
    unsigned char* __restrict__ dst,
    int M,
    int K
) {
    const int lane = threadIdx.x;
    const int tile_m = blockIdx.x;
    const int bi = blockIdx.y;
    const int nblocks = K / 32;
    const int m_lane = lane & 15;
    const int k_grp = lane >> 4;
    const int row = tile_m * 16 + m_lane;
    const int safe_row = row < M ? row : M - 1;
    const block_q8_0* wb = src + (long long)safe_row * nblocks + bi;
    unsigned char* tile = dst + ((long long)tile_m * nblocks + bi) * 544;

    if (lane < 16)
        reinterpret_cast<_Float16*>(tile)[lane] = wb->d;

    typedef unsigned int __attribute__((ext_vector_type(2))) uint2_t;
    #pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
        const uint2_t v = *reinterpret_cast<const uint2_t*>(
            wb->qs + kt * 16 + k_grp * 8);
        *reinterpret_cast<uint2_t*>(tile + 32 + kt * 256 + lane * 8) = v;
    }
}
