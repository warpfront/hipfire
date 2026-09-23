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

// One-time Q8_0 -> affine HFQ6-G256 requantizer. Each workgroup converts one
// 256-value row group from 272 bytes to 200 bytes. This is intentionally a
// drift-gated performance experiment: the eight Q8_0 block scales collapse
// to one min/max affine scale.
extern "C" __global__ __launch_bounds__(256, 2)
void repack_q8_0_to_hfq6g256_gfx12(
    const block_q8_0* __restrict__ src,
    unsigned char* __restrict__ dst,
    int M,
    int K
) {
    const int groups_per_row = K / 256;
    const int linear = blockIdx.x;
    const int row = linear / groups_per_row;
    const int group = linear - row * groups_per_row;
    if (row >= M) return;
    const int lane = threadIdx.x;
    const int q8_blocks_per_row = K / 32;
    const block_q8_0* b = src + (long long)row * q8_blocks_per_row + group * 8;
    const int sb = lane >> 5;
    const int si = lane & 31;
    const float v = (float)b[sb].d * (float)b[sb].qs[si];

    __shared__ float vals[256];
    __shared__ float mins[256];
    __shared__ float maxs[256];
    vals[lane] = v;
    mins[lane] = v;
    maxs[lane] = v;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (lane < off) {
            mins[lane] = fminf(mins[lane], mins[lane + off]);
            maxs[lane] = fmaxf(maxs[lane], maxs[lane + off]);
        }
        __syncthreads();
    }

    const float lo = mins[0];
    const float hi = maxs[0];
    const float sc = hi > lo ? (hi - lo) * (1.0f / 63.0f) : 1.0f;
    const float zp = lo;
    unsigned char* out = dst + ((long long)row * groups_per_row + group) * 200;
    if (lane == 0) {
        *reinterpret_cast<float*>(out) = sc;
        *reinterpret_cast<float*>(out + 4) = zp;
    }
    __syncthreads();

    const int qi = max(0, min(63, __float2int_rn((vals[lane] - zp) / sc)));
    __shared__ unsigned char q[256];
    q[lane] = (unsigned char)qi;
    __syncthreads();
    if (lane < 64) {
        const int i = lane * 4;
        const unsigned int p = (unsigned int)q[i]
            | ((unsigned int)q[i + 1] << 6)
            | ((unsigned int)q[i + 2] << 12)
            | ((unsigned int)q[i + 3] << 18);
        unsigned char* d = out + 8 + lane * 3;
        d[0] = (unsigned char)p;
        d[1] = (unsigned char)(p >> 8);
        d[2] = (unsigned char)(p >> 16);
    }
}
