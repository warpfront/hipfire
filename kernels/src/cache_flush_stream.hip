// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Cache-evictor for A/B timing: streams over a large device buffer so the
// kernel under test runs with a cold L2/MALL. Grid = n_vec/256 blocks of
// 256 threads; each thread reads exactly one uint4 and XOR-folds it.
//
// NOTE on the write: the brief asked for only thread 0 of block 0 to write
// sink[0], but with a multi-block grid every other block's reads would be
// dead code (unused global loads are eliminable), silently deleting the
// eviction traffic on all but one block. Instead each block XOR-reduces its
// 256 threads' folds (shuffles + 8 words of LDS) and thread 0 of each block
// writes sink[blockIdx.x] — the harness sizes sink at n_blocks floats, so
// every thread's read provably feeds a written output and no load can be
// optimized away. Kernarg signature is unchanged.

__launch_bounds__(256, 1)
extern "C" __global__ void cache_flush_stream(
    const uint4* __restrict__ buf,
    unsigned n_vec,
    float* __restrict__ sink
) {
    extern __shared__ unsigned sh[];
    // sh[8]: one word per warp.

    const unsigned tid = threadIdx.x;
    const unsigned lane_id = tid & 31;
    const unsigned warp_id = tid >> 5;

    const unsigned i = tid + (unsigned)blockIdx.x * (unsigned)blockDim.x;
    uint4 v = buf[i];
    unsigned acc = v.x ^ v.y ^ v.z ^ v.w;

    acc ^= __shfl_down(acc, 16);
    acc ^= __shfl_down(acc, 8);
    acc ^= __shfl_down(acc, 4);
    acc ^= __shfl_down(acc, 2);
    acc ^= __shfl_down(acc, 1);
    if (lane_id == 0) sh[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        acc = (lane_id < 8) ? sh[lane_id] : 0;
        acc ^= __shfl_down(acc, 4);
        acc ^= __shfl_down(acc, 2);
        acc ^= __shfl_down(acc, 1);
        if (lane_id == 0) sink[blockIdx.x] = __uint_as_float(acc);
    }
    (void)n_vec;
}
