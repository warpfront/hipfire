// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// Masked row softmax with an f16 output, for the encoder's eager attention
// (arch 17).
//
// Two jobs at once, both of which `softmax_f32` cannot do:
//   1. MASK. Only columns in `[row_start[r], row_end[r])` are attended; the
//      rest get exactly 0.0. The encoder pads a sequence up to a multiple of
//      64 so the tiled GEMM's `k % 64` contract is met, and the padding rows
//      of K are zeros — without a mask their score of 0 would win a real
//      softmax share and dilute every real query. The bounds are scalars
//      because one encode is one sequence, so every row shares the same valid
//      span; a batched caller (several sequences padded into one row block)
//      would extend this to per-row bounds, which is a two-line change to the
//      two reads below.
//   2. WRITE f16 directly. The eager attention's second GEMM takes P as f16,
//      so `softmax_f32` + a separate `cast_f32_to_f16` over the n x n matrix
//      was two passes over 100 MB per head per layer. This is one.
//
// Grid: [rows], block 256, dynamic LDS = block * 4 (two tree reductions,
// max then sum, same shape as `softmax_f32`). An empty valid range
// (valid_end <= valid_start) writes zeros rather than NaN.
extern "C" __global__ void attention_rows_softmax_f16(
    const float* __restrict__ scores,   // [rows, n] f32
    __half* __restrict__ out,           // [rows, n] f16
    int valid_start,
    int valid_end,
    int n
) {
    extern __shared__ float sdata[];
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int nthr = (int)blockDim.x;
    const float* rowp = scores + (size_t)row * (size_t)n;
    int ks = valid_start;
    int ke = valid_end;
    if (ks < 0) ks = 0;
    if (ke > n) ke = n;
    __half* outp = out + (size_t)row * (size_t)n;

    if (ke <= ks) {
        for (int i = tid; i < n; i += nthr) outp[i] = __float2half(0.0f);
        return;
    }

    float mx = -1e30f;
    for (int i = ks + tid; i < ke; i += nthr) mx = fmaxf(mx, rowp[i]);
    sdata[tid] = mx;
    __syncthreads();
    for (int s = nthr / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    mx = sdata[0];
    __syncthreads();

    float sum = 0.0f;
    for (int i = ks + tid; i < ke; i += nthr) sum += expf(rowp[i] - mx);
    sdata[tid] = sum;
    __syncthreads();
    for (int s = nthr / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();

    const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int i = tid; i < n; i += nthr) {
        const float p = (i >= ks && i < ke) ? expf(rowp[i] - mx) * inv : 0.0f;
        outp[i] = __float2half(p);
    }
}
