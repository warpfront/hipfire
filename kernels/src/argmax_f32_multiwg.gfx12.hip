// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx12 (RDNA4) multi-WG argmax_f32 — two-stage reduction.
//
// Original argmax.hip uses Grid=[1, 1, 1] (single WG over the entire input),
// which uses 1/56th of the GPU's CUs and clocks at ~232 µs/call on Qwen3.6's
// 248320-token vocab. This is 100× off the BW ceiling (1 MB at 700 GB/s
// = 1.4 µs).
//
// Multi-WG design:
//   Pass 1 (`argmax_f32_partial_gfx12`): Grid=[ceil(N/CHUNK), 1, 1], each
//     WG computes local argmax over CHUNK=2048 elements, writes (max_val,
//     max_idx) to a 2-element-per-WG scratch buffer.
//   Pass 2 (`argmax_f32_final_gfx12`): Grid=[1, 1, 1], single WG reduces
//     the partial results down to the global argmax.
//
// For N=248320, CHUNK=2048: 121 WGs in pass 1, each handling ~17 iters
// per thread × 256 threads. Then 1 small WG in pass 2 (121 values).
// Expected: pass1 ~5 µs + pass2 ~2 µs = ~7 µs total vs baseline 232 µs.

#define CHUNK 2048
#define BLOCK 256

extern "C" __global__ void argmax_f32_partial_gfx12(
    const float* __restrict__ data,
    float*       __restrict__ partial_vals,   // [num_wgs]
    int*         __restrict__ partial_idxs,   // [num_wgs]
    int N
) {
    const int wg = blockIdx.x;
    const int tid = threadIdx.x;
    const int start = wg * CHUNK;
    const int end = min(start + CHUNK, N);

    float lmax = -1e30f;
    int lidx = -1;

    for (int i = start + tid; i < end; i += BLOCK) {
        const float v = data[i];
        if (v > lmax) {
            lmax = v;
            lidx = i;
        }
    }

    __shared__ float smax[BLOCK];
    __shared__ int   sidx[BLOCK];
    smax[tid] = lmax;
    sidx[tid] = lidx;
    __syncthreads();

    #pragma unroll
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid]) {
            smax[tid] = smax[tid + s];
            sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_vals[wg] = smax[0];
        partial_idxs[wg] = sidx[0];
    }
}

extern "C" __global__ void argmax_f32_final_gfx12(
    const float* __restrict__ partial_vals,
    const int*   __restrict__ partial_idxs,
    int*         __restrict__ result,
    int num_wgs
) {
    const int tid = threadIdx.x;
    float lmax = -1e30f;
    int   lidx = -1;
    for (int i = tid; i < num_wgs; i += BLOCK) {
        const float v = partial_vals[i];
        if (v > lmax) {
            lmax = v;
            lidx = partial_idxs[i];
        }
    }
    __shared__ float smax[BLOCK];
    __shared__ int   sidx[BLOCK];
    smax[tid] = lmax;
    sidx[tid] = lidx;
    __syncthreads();
    #pragma unroll
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid]) {
            smax[tid] = smax[tid + s];
            sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) result[0] = sidx[0];
}
