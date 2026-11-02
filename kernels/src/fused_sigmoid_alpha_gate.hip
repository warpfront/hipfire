// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Fused sigmoid(dn_beta) + alpha_gate(dn_alpha). Both ops run in the
// DeltaNet attention preamble on independent buffers of size n_v_heads.
// Originally two back-to-back scalar launches; fused into one here.
//
// Grid: [ceil(n/256) * batch, 1, 1] (rows folded into grid.x: row =
// blockIdx.x / ceil(n/blockDim.x), block-in-row = blockIdx.x % that), so the
// batch is not bound by the 65535 grid.y limit. Each thread handles one head
// index for a single batch row. For decode the grid is [ceil(n/256),1,1]
// (row 0, byte-identical to the pre-batching version); for prefill beta/alpha
// are laid out as [N × n] row-major. dt_bias and a_log are shared per-head
// constants and do NOT get a batch offset.
//
// Bit-exact with the separate kernels since all operations are
// elementwise — no cross-thread dependencies or associativity changes.

extern "C" __global__ void fused_sigmoid_alpha_gate_f32(
    float* __restrict__ beta,         // [batch × n] in-place sigmoid
    float* __restrict__ alpha,        // [batch × n] in-place alpha_gate
    const float* __restrict__ dt_bias,// [n]
    const float* __restrict__ a_log,  // [n]
    int n
) {
    const unsigned blocks_per_row = ((unsigned)n + blockDim.x - 1) / blockDim.x;
    const unsigned row = blockIdx.x / blocks_per_row;
    int i = (blockIdx.x - row * blocks_per_row) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const long long batch_off = (long long)row * n;
    float* beta_b  = beta  + batch_off;
    float* alpha_b = alpha + batch_off;

    // Sigmoid on beta — matches sigmoid_f32.
    beta_b[i] = 1.0f / (1.0f + expf(-beta_b[i]));

    // Alpha gate on alpha — matches alpha_gate_f32 (softplus(x+bias) then
    // multiply by -exp(a_log), with the same numerical guards).
    float biased = alpha_b[i] + dt_bias[i];
    float sp = (biased > 20.0f) ? biased
             : ((biased < -20.0f) ? expf(biased) : logf(1.0f + expf(biased)));
    alpha_b[i] = sp * (-expf(a_log[i]));
}
