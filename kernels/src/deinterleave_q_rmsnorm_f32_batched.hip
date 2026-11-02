// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Prefill fusion of `deinterleave_f32_batched` with the FullAttention Q
// RMSNorm (`rmsnorm_batched` on fa_q, in place).
//
// Replaces, per FullAttention layer on the unfused prefill path,
//   deinterleave_f32_batched(interleaved -> q, gate)   (1 launch)
//   rmsnorm_batched(q, q_norm, q)                      (1 launch, in place)
// with ONE launch. The Q values never round-trip through global memory:
// each (head, row) block gathers its Q row from the interleaved input into
// registers, reduces, and stores the normed row directly, while the gate
// row passes through verbatim.
//
// Bit-exactness contract (eval md5 gate):
// - Gather is a pure float copy (identical values to what deinterleave
//   wrote); the gate passthrough stores the same floats deinterleave stored.
// - Reduction is the verbatim `rmsnorm_f32` (rmsnorm.hip) algorithm with the
//   SAME blockDim the unfused `rmsnorm_batched` launcher picks
//   (block = min(256, head_dim), host-enforced): same per-thread strided
//   accumulation order, same LDS halving tree over the same entry count,
//   same `rsqrt(sum/n + eps)`, same `(x * w) * rms` left-associated write
//   expression evaluated on the same input floats (register-held instead of
//   reloaded — a float->float move, exact).
//
// Constraints (enforced host-side; anything else stays on the old path):
// - head_dim <= 2048 (register file holds ceil(head_dim/block) <= 8 values
//   per thread with block = min(256, head_dim)).
// - q_out must not alias `interleaved` (it never does: fa_q vs fa_q_full
//   are distinct scratch planes); gate_out must alias neither input.
// - The K norm, RoPE, and triattn tap stay exactly as before (K comes from
//   the K projection, not the deinterleaver, so it cannot fold here).
//
// Grid: [n_heads, N, 1]. Block: [min(256, head_dim)]. Shared: block floats.

#define DEINT_QNORM_MAX_PER_THREAD 8

extern "C" __global__ void deinterleave_q_rmsnorm_f32_batched(
    const float* __restrict__ interleaved,  // [N x n_heads x head_dim x 2]
    float* __restrict__ q_out,              // [N x n_heads x head_dim], normed
    float* __restrict__ gate_out,           // [N x n_heads x head_dim], passthrough
    const float* __restrict__ weight,       // [head_dim]
    int n_heads,
    int head_dim,
    int N,
    float eps
) {
    extern __shared__ float sdata[];

    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int b = blockIdx.y;
    if (b >= N) return;
    const int tid = threadIdx.x;
    const int block = blockDim.x;

    const long long total = (long long)n_heads * head_dim;
    const float* inp_b = interleaved + (long long)b * total * 2;
    float* q_b = q_out + (long long)b * total;
    float* g_b = gate_out + (long long)b * total;
    const long long q_row = (long long)h * head_dim;        // row in Q/gate out
    const long long src_row = (long long)h * head_dim * 2;  // row in interleaved

    // Phase 1: gather Q into registers (+ verbatim gate passthrough) and
    // accumulate sum-of-squares in the exact rmsnorm_f32 strided order.
    float qv[DEINT_QNORM_MAX_PER_THREAD];
    float sum_sq = 0.0f;
    int m = 0;
    for (int d = tid; d < head_dim; d += block) {
        float q = inp_b[src_row + d];             // Q slot: h*2*HD + d
        float g = inp_b[src_row + head_dim + d];  // gate slot
        g_b[q_row + d] = g;
        qv[m++] = q;
        sum_sq += q * q;
    }

    // Phase 2: verbatim rmsnorm_f32 LDS halving reduction.
    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = block >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / (float)head_dim + eps);

    // Phase 3: verbatim rmsnorm_f32 write order and expression.
    m = 0;
    for (int d = tid; d < head_dim; d += block) {
        q_b[q_row + d] = qv[m++] * weight[d] * rms;
    }
}
