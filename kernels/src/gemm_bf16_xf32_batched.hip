// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — Qwen4 native BF16 shared-expert projections.

#include <hip/hip_runtime.h>

// BF16-weight × F32-input batched GEMM with an F32 accumulator.
//
// Layouts:
//   W: [M, K] BF16, row-major
//   X: [N, K] F32, row-major
//   Y: [N, M] F32, row-major
//
// Each wave computes one (batch row, output row).  The BF16 values are widened
// by shifting their raw high 16 bits into an F32 word; no F16 shadow or host
// conversion is introduced.  Qwen4 uses this for all four shared-expert
// projections, so the downloaded BF16 values remain the exact source values.
__launch_bounds__(32, 16)
extern "C" __global__ void gemm_bf16_xf32_batched(
    const unsigned short* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int row = blockIdx.x;
    const int batch = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= M || batch >= N || lane >= 32) return;

    const unsigned short* w_row = W + (size_t)row * K;
    const float* x_row = X + (size_t)batch * K;
    float sum = 0.0f;
    for (int k = lane; k < K; k += 32) {
        const unsigned int bits = ((unsigned int)w_row[k]) << 16;
        const float weight = __uint_as_float(bits);
        sum += weight * x_row[k];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset, 32);
    if (lane == 0)
        Y[(size_t)batch * M + row] = sum;
}
