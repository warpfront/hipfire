// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched Q8_0 GEMM: Y[batch, M] = X[batch, K] @ W[M, K]^T where W is Q8_0
// quantized (34 bytes per group of 32 elements: 2 byte FP16 scale + 32 INT8).
//
// For DFlash speculative decode the same Q8 lm_head runs against B = 15
// (draft positions) or B = 16 (target verify positions) hidden vectors.
// Serial-GEMV reads the weight matrix `batch` times; this kernel reads it
// once and broadcasts each weight load across MAX_BATCH per-row registers.
//
// On the 7900 XTX (gfx1100) this kernel benchmarks roughly equal to the
// serial-GEMV-with-staged-output path on a 4B Q8 lm_head (B=15-16,
// K=2560, M=248K) — the GPU pipelines async kernel launches efficiently
// enough that the per-launch overhead isn't the dominant cost. Kept here
// as a substrate for further tuning (smarter X layout, shared-memory
// weight broadcast) without disrupting the dispatch surface.
//
// Grid:  [M, 1, 1]    — one block per output row.
// Block: [32, 1, 1]   — one wave per row.
//
// FP-reduction order matches gemv_q8_0 (single-accumulator FMA loop) so
// the per-batch slice of the output equals what the per-row GEMV would
// produce — preserves greedy parity for spec decode.

#define MAX_BATCH 64

extern "C" __global__ void gemm_q8_0_batched(
    const unsigned char* __restrict__ A_q8,  // [M, K/32 * 34] packed Q8_0
    const float* __restrict__ X,              // [batch_size, K] row-major
    float* __restrict__ Y,                    // [batch_size, M] row-major
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q8 + (size_t)row * blocks_per_row * 34;

    float sums[MAX_BATCH];
    #pragma unroll
    for (int b = 0; b < MAX_BATCH; b++) sums[b] = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + tid];
        float dq = d * (float)qval;
        const int x_off = bi * 32 + tid;

        #pragma unroll
        for (int b = 0; b < MAX_BATCH; b++) {
            if (b < batch_size) {
                sums[b] += dq * X[(size_t)b * K + x_off];
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < MAX_BATCH; b++) {
        if (b < batch_size) {
            float s = sums[b];
            for (int offset = 16; offset > 0; offset >>= 1)
                s += __shfl_down(s, offset);
            if (tid == 0) Y[(size_t)b * M + row] = s;
        }
    }
}

// gemm_q8_0_batched with eight rows per workgroup sharing X through LDS
// (the per-row kernel re-reads all of X from memory for every output row).
// Each 32-lane group owns one row with the per-row kernel's exact lane
// order: every lane accumulates its K lane over the groups in ascending
// order, then the same shfl_down tree, so every output is bitwise equal.
// Grid: [ceil(M / 8), ceil(batch_size / MAX_BATCH), 1]: grid.y takes the
// batch in MAX_BATCH-row tiles, each exactly a per-tile launch.  Block 256.
#define Q8_LDS_GROUPS 4
// RPW weight rows per wave: every X value read from LDS feeds RPW FMAs.
template <int RPW>
__device__ __forceinline__ void gemm_q8_0_batched_lds_body(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    X += (size_t)blockIdx.y * MAX_BATCH * K;
    Y += (size_t)blockIdx.y * MAX_BATCH * M;
    batch_size = min(batch_size - (int)blockIdx.y * MAX_BATCH, MAX_BATCH);
    __shared__ __align__(16) float xs[MAX_BATCH * Q8_LDS_GROUPS * 32];
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * RPW;
    const int blocks_per_row = K / 32;
    const unsigned char* row_data[RPW];
#pragma unroll
    for (int r = 0; r < RPW; r++)
        row_data[r] = A_q8 + (size_t)min(row0 + r, M - 1) * blocks_per_row * 34;

    float sums[RPW][MAX_BATCH];
#pragma unroll
    for (int r = 0; r < RPW; r++)
#pragma unroll
        for (int b = 0; b < MAX_BATCH; b++) sums[r][b] = 0.0f;

    for (int b0 = 0; b0 < blocks_per_row; b0 += Q8_LDS_GROUPS) {
        const int groups = min(Q8_LDS_GROUPS, blocks_per_row - b0);
        const int width = groups * 32;
        __syncthreads();
        // 32 float4 per row chunk: thread t copies vector t % 32 of rows
        // t / 32, t / 32 + 8, ... (no per-element division).
        {
            const int v = threadIdx.x & 31;
            if (v * 4 < width) {
                for (int b = threadIdx.x >> 5; b < batch_size; b += 8)
                    *(float4*)(xs + b * Q8_LDS_GROUPS * 32 + v * 4) =
                        *(const float4*)(X + (size_t)b * K + b0 * 32 + v * 4);
            }
        }
        __syncthreads();
        for (int g = 0; g < groups; g++) {
            float dq[RPW];
#pragma unroll
            for (int r = 0; r < RPW; r++) {
                const unsigned char* block = row_data[r] + (b0 + g) * 34;
                dq[r] = (float)*((const _Float16*)block) * (float)(signed char)block[2 + lane];
            }
#pragma unroll
            for (int b = 0; b < MAX_BATCH; b++) {
                if (b < batch_size) {
                    const float x = xs[b * Q8_LDS_GROUPS * 32 + g * 32 + lane];
#pragma unroll
                    for (int r = 0; r < RPW; r++) sums[r][b] += dq[r] * x;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; r++) {
        if (row0 + r >= M) break;
#pragma unroll
        for (int b = 0; b < MAX_BATCH; b++) {
            if (b < batch_size) {
                float s = sums[r][b];
                for (int offset = 16; offset > 0; offset >>= 1)
                    s += __shfl_down(s, offset, 32);
                if (lane == 0) Y[(size_t)b * M + row0 + r] = s;
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256) void gemm_q8_0_batched_lds8(
    const unsigned char* __restrict__ A_q8, const float* __restrict__ X,
    float* __restrict__ Y, int M, int K, int batch_size) {
    gemm_q8_0_batched_lds_body<1>(A_q8, X, Y, M, K, batch_size);
}

// Two rows per wave (sixteen per workgroup); grid [ceil(M / 16), batch tiles].
extern "C" __global__ __launch_bounds__(256) void gemm_q8_0_batched_lds16(
    const unsigned char* __restrict__ A_q8, const float* __restrict__ X,
    float* __restrict__ Y, int M, int K, int batch_size) {
    gemm_q8_0_batched_lds_body<2>(A_q8, X, Y, M, K, batch_size);
}
#undef Q8_LDS_GROUPS
