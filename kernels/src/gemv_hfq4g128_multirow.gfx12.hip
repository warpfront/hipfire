// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx12 (RDNA4) HFQ4-G128 multirow GEMV: 4 rows per WG, shared LDS-X.
//
// gfx1100 documented multirow as NOT a win because single-row was already
// BW-saturated on RDNA3 (577 GB/s on 9B GEMV = 60 percent peak). On gfx12
// decode the gemv_hfq4g128 kernel is at 16.9 µs/call (rocprof) where
// per-call compute is ~200 ns — meaning ~16 µs is dispatch/scheduling
// overhead. 4-rows-per-WG cuts WG count by 4×, amortizing dispatch.
//
// Per-call savings expected: ~75 percent of dispatch overhead × 12 µs ≈ 9 µs.
// Per PARO call: 16.9 → ~8 µs. Per token (131 calls): 1.0 ms savings.
//
// Per-WG layout:
//   32 threads, 4 rows. Each thread accumulates 4 separate per-row sums.
//   LDS holds X (K×4 bytes), loaded cooperatively once per WG.
//   Inner loop: scalar nibble unpack + FMA into all 4 row accumulators.
//
// Constraint: M must be divisible by 4 (caller falls back to single-row
// for M%4 != 0).

#if defined(__gfx1200__) || defined(__gfx1201__)

#define ROWS_PER_WG 4

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g128_multirow_gfx12(
    const char*  __restrict__ A,
    const float* __restrict__ x,
    float*       __restrict__ y,
    int M, int K
) {
    const int row_base = blockIdx.x * ROWS_PER_WG;
    if (row_base >= M) return;
    const int tid = threadIdx.x;
    const int byte_off = tid * 2;

    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 72;

    // 4 row pointers. Clamp OOB rows to row_base (output masked).
    const char* row_ptrs[ROWS_PER_WG];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WG; r++) {
        int row = row_base + r;
        int safe = (row < M) ? row : row_base;
        row_ptrs[r] = A + (long long)safe * row_bytes;
    }

    // LDS-cached X (K floats = K*4 bytes).
    extern __shared__ float x_lds[];
    for (int k = tid; k < K; k += 32) {
        x_lds[k] = x[k];
    }
    __syncthreads();

    // 4 independent accumulators per row.
    float acc[ROWS_PER_WG] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const int base = g * 128 + tid * 4;

        #pragma unroll
        for (int r = 0; r < ROWS_PER_WG; r++) {
            const char* gp = row_ptrs[r] + g * 72;
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            unsigned char b0 = *(const unsigned char*)(gp + 8 + byte_off);
            unsigned char b1 = *(const unsigned char*)(gp + 8 + byte_off + 1);

            float v0 = sc * (float)((b0     ) & 0xFu) + zp;
            float v1 = sc * (float)((b0 >> 4) & 0xFu) + zp;
            float v2 = sc * (float)((b1     ) & 0xFu) + zp;
            float v3 = sc * (float)((b1 >> 4) & 0xFu) + zp;

            acc[r] += v0 * x_lds[base]     + v1 * x_lds[base + 1]
                    + v2 * x_lds[base + 2] + v3 * x_lds[base + 3];
        }
    }

    // Wave-shfl reduce each row independently.
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WG; r++) {
        float v = acc[r];
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down(v, offset);
        }
        if (tid == 0) {
            int row = row_base + r;
            if (row < M) y[row] = v;
        }
    }
}

#else

extern "C" __global__ void gemv_hfq4g128_multirow_gfx12(
    const char*, const float*, float*, int, int
) {}

#endif
