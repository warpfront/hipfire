// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Y[token, col] += sigmoid(scalar[token]) * temp[token, col]
//
// Completes the F32 shared_expert down step for the batched MoE FFN path
// when shared_expert.down is FP16 dense (z-lab/Qwen3.6-35B-A3B-PARO).
// The HFQ4-quantized sister (`gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched`)
// fuses GEMV + sigmoid-scaled residual into one launch; for the F32
// path we do `gemm_f32_batched` first into a scratch, then this kernel
// applies the sigmoid scaling and accumulates into x_batch.
//
// Layout: Y, temp are [N × M] row-major (token-major). scalar is [N].
// Grid: [N, ceil(M/256), 1]. Block: [256].

extern "C" __global__ void sigmoid_scaled_add_inplace_f32(
    float*       __restrict__ y,        // [N × M] accumulator (in/out)
    const float* __restrict__ temp,     // [N × M] GEMM output
    const float* __restrict__ scalar,   // [N] pre-sigmoid logits
    int N, int M
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= M) return;

    // sigmoid(x) = 1 / (1 + e^-x) — same convention as the HFQ4 sister.
    const float s_raw = scalar[row];
    const float s = 1.0f / (1.0f + __expf(-s_raw));

    const long long idx = (long long)row * M + col;
    y[idx] += s * temp[idx];
}
