// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx12 (RDNA4) wave32-optimized multirow GEMV: 8 rows per WG, 4 threads
// per row (subwave). Each subwave reads ITS row's K-dim with stride-4
// coalesced access. X cached in LDS, reused across all 8 rows.
//
// === Why subwave-per-row ===
//
// First-pass design (each thread iters ALL 8 rows at fixed K) had
// TERRIBLE memory access: every thread read 8 different rows of A at
// stride K, hitting 8 different cache lines per iter. Measured -54%
// on decode wall.
//
// Subwave design: lane-row mapping is `row = row_base + (tid >> 2)`,
// `k_lane = tid & 3`. Each row gets 4 lanes that stride K with offset
// k_lane, k_lane+4, k_lane+8, ... — coalesced within the 4-lane subwave.
//
// Per-row reduction: 4-lane shfl_down (offset 2, then 1).
//
// Grid:    [ceil(M/8), 1, 1]
// Block:   [32]
// LDS:     K × 4 bytes (dynamic)
//
// Shape requirements:
//   M % 8 == 0
//   K % 4 == 0

#if defined(__gfx1200__) || defined(__gfx1201__)

#define ROWS_PER_WG 8
#define LANES_PER_ROW 4

__launch_bounds__(32, 8)
extern "C" __global__ void gemv_f32_multirow_gfx12(
    const float* __restrict__ A,
    const float* __restrict__ X,
    float*       __restrict__ Y,
    int M, int K
) {
    const int tid = threadIdx.x;
    const int row_base = blockIdx.x * ROWS_PER_WG;
    if (row_base >= M) return;

    const int row_in_wg = tid >> 2;          // 0..7
    const int k_lane    = tid & 3;           // 0..3
    const int row = row_base + row_in_wg;

    // Cooperative LDS load (32 lanes stride-load K floats).
    extern __shared__ float x_lds[];
    for (int k = tid; k < K; k += 32) {
        x_lds[k] = X[k];
    }
    __syncthreads();

    if (row >= M) return;

    // Each subwave covers K with stride 4 starting at k_lane.
    float acc = 0.0f;
    const float* a_row = A + (long long)row * K;
    for (int k = k_lane; k < K; k += LANES_PER_ROW) {
        acc += a_row[k] * x_lds[k];
    }

    // 4-lane reduction via shfl_xor (works within wave32). Each subwave
    // has 4 contiguous lanes — shfl_xor with mask 2, then 1 sums them.
    acc += __shfl_xor(acc, 2);
    acc += __shfl_xor(acc, 1);

    if (k_lane == 0) {
        Y[row] = acc;
    }
}

#else

extern "C" __global__ void gemv_f32_multirow_gfx12(
    const float*, const float*, float*, int, int
) {}

#endif
