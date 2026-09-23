// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx12 (RDNA4) F32 fused shared_expert.down step for the PARO decode
// path. Mirrors MQ4's `gemv_hfq4g256_residual_sigmoid_scaled_gpu` —
// combines 4 sequential kernel launches into 1:
//
//   Old (non-MQ4 fallback at qwen35.rs:3002-3009):
//     1. sigmoid_f32(scalar_buf)
//     2. silu_mul_f32(shared_gate, shared_up, shared_hid)
//     3. gemv_f32(W_down, shared_hid, ffn_out)
//     4. scaled_add_inplace_gpu_scalar_f32(x_residual, ffn_out, scalar_buf)
//
//   New (this kernel):
//     gemv_f32_silu_mul_residual_sigmoid_scaled(W_down, gate, up, x_residual, scalar, dim, smi)
//
// Per MoE layer: 4 launches → 1 = 3 saved.
// Per token z-lab A3B-PARO decode: 40 MoE layers × 3 saved = 120 launches saved.
//
// === Math ===
//
// hidden[k] = silu(gate[k]) * up[k]      where silu(x) = x * sigmoid(x)
// y[m]      = x_residual[m] + sigmoid(scalar[0]) * sum_k W[m, k] * hidden[k]
//
// === Layout ===
//
// Grid:  [M, 1, 1] workgroups
// Block: [256]
// LDS:   256 floats (per-WG partial-sum reduction)
//
// Shapes for z-lab A3B-PARO: M = dim = 2048, K = smi = 512.

#if defined(__gfx1200__) || defined(__gfx1201__)

extern "C" __global__ void gemv_f32_silu_mul_residual_sigmoid_scaled_gfx12(
    const float* __restrict__ W,         // [M × K] F32 down weights
    const float* __restrict__ gate,      // [K] shared_expert.gate output
    const float* __restrict__ up,        // [K] shared_expert.up output
    float*       __restrict__ y,         // [M] x_residual — in/out
    const float* __restrict__ scalar,    // [1] shared_expert_gate raw logit
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    // Pre-compute sigmoid of the scalar gate (all threads redundantly).
    // The expf is in 1 lane; later we just use the shared register value.
    const float s_raw = scalar[0];
    const float sigmoid_s = 1.0f / (1.0f + __expf(-s_raw));

    // Per-thread accumulation across K with silu*up fused inline.
    const float* w_row = W + (long long)row * K;
    float sum = 0.0f;
    for (int k = tid; k < K; k += 256) {
        const float g = gate[k];
        const float silu_g = g / (1.0f + __expf(-g));
        const float hidden = silu_g * up[k];
        sum += w_row[k] * hidden;
    }

    // LDS reduction (256-way).
    __shared__ float sdata[256];
    sdata[tid] = sum;
    __syncthreads();
    #pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Residual + sigmoid-scaled accumulation.
    if (tid == 0) {
        y[row] += sigmoid_s * sdata[0];
    }
}

#else

extern "C" __global__ void gemv_f32_silu_mul_residual_sigmoid_scaled_gfx12(
    const float*, const float*, const float*, float*, const float*, int, int
) {}

#endif
