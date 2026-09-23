// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx12 (RDNA4) Q8_0 fused shared_expert.down step for PARO decode.
// Mirror of gemv_f32_silu_mul_residual_sigmoid_scaled but with Q8_0
// quantized weight (1 byte/elem + scale vs 4 bytes/elem F32).
//
//   hidden[k] = silu(gate[k]) * up[k]
//   y[m]      = x_residual[m] + sigmoid(scalar[0]) * sum_k W_q8[m,k] * hidden[k]
//
// Per MoE layer: 1 launch instead of 3 (silu_mul + gemv_q8_0 + sigmoid_scaled).
// Combined with Q8 storage, recovers the F32 down fusion win at Q8 BW.
//
// Layout: 256-thread block, one output row per WG. Stride-256 access.

#if defined(__gfx1200__) || defined(__gfx1201__)

extern "C" __global__ void gemv_q8_0_silu_mul_residual_sigmoid_scaled_gfx12(
    const char*  __restrict__ W,         // [M × (K/32) × 34] Q8_0 weights
    const float* __restrict__ gate,      // [K] shared_expert.gate output
    const float* __restrict__ up,        // [K] shared_expert.up output
    float*       __restrict__ y,         // [M] x_residual — in/out
    const float* __restrict__ scalar,    // [1] shared_expert_gate raw logit
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const float s_raw = scalar[0];
    const float sigmoid_s = 1.0f / (1.0f + __expf(-s_raw));

    const int num_blocks = K / 32;
    const char* row_base = W + (long long)row * num_blocks * 34;

    // Stride-256 access. Each thread covers K/256 K-elements.
    float sum = 0.0f;
    for (int k = tid; k < K; k += 256) {
        const int b = k >> 5;       // block index = k / 32
        const int e = k & 31;       // intra-block element = k % 32
        const char* block_base = row_base + b * 34;
        const _Float16 d = *reinterpret_cast<const _Float16*>(block_base);
        const float scale = (float)d;
        const int8_t v = *reinterpret_cast<const int8_t*>(block_base + 2 + e);
        const float w = scale * (float)v;
        const float g = gate[k];
        const float silu_g = g / (1.0f + __expf(-g));
        const float hidden = silu_g * up[k];
        sum += w * hidden;
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
        y[row] += sigmoid_s * sdata[0];
    }
}

#else

extern "C" __global__ void gemv_q8_0_silu_mul_residual_sigmoid_scaled_gfx12(
    const char*, const float*, const float*, float*, const float*, int, int
) {}

#endif
