// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx12 (RDNA4) GEMV: F16 weight × F32 X → F32 Y, 256-thread block.
// Drop-in replacement for gemv_f32 when the weight can be stored as F16
// (halves BW). Designed specifically for the lm_head decode path on
// PARO models where weight is F16-on-disk:
//
//   M = vocab_size (248320 on Qwen3.6)
//   K = dim        (2048)
//   batch = 1      (single-token decode)
//
// Why this kernel exists vs `gemm_f16_tiled`:
//   gemm_f16_tiled uses 32-thread blocks (Grid=[M, N]) optimized for
//   batched GEMM (N>1) — BW-starved at N=1 with the huge M=248k.
//   This kernel uses 256-thread blocks (matching gemv_f32 baseline) for
//   proper BW saturation per WG. Reads F16 weight inline, accumulates
//   in F32, applies alpha/beta epilogue identical to gemv_f32.
//
// rocprof on z-lab A3B-PARO post-fusion-stack:
//   gemv_f32 (lm_head) = 28.4% of decode GPU time
//   511 calls × 3544.7 µs/call = 1811 ms (decode-only, gen=500 bench)
//   2 GB read per call at ~570 GB/s effective. F16 storage halves the
//   weight read → expected ~1.75 ms/call = saves ~1.7 ms/token wall.
//
// Layout: same call contract as gemv_f32 (A row-major [M × K], x [K], y [M]).

extern "C" __global__ void gemv_f16_x32_lmhead_gfx12(
    const _Float16* __restrict__ A,   // [M × K] F16 weights
    const float*    __restrict__ x,   // [K] F32 input
    float*          __restrict__ y,   // [M] F32 output
    int M, int K,
    float alpha, float beta
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const _Float16* a_row = A + (long long)row * K;
    float sum = 0.0f;
    for (int k = tid; k < K; k += 256) {
        sum += (float)a_row[k] * x[k];
    }

    __shared__ float sdata[256];
    sdata[tid] = sum;
    __syncthreads();
    #pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        y[row] = alpha * sdata[0] + beta * y[row];
    }
}
