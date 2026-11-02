// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched partial RoPE, half-split convention (HF rotate_half) — see
// kernels/src/rope_partial_halfsplit.hip for the per-position twin and
// docs/plans/qwen35-mq4-quality-gap.md §"RoPE convention probe / halfsplit fix"
// for the rationale. Pairs are (i, i + n_rot/2). Dispatched from
// rope_partial_interleaved_f32_batched as the default since 2026-05-12;
// the interleaved batched kernel remains available for legacy reproducibility
// behind HIPFIRE_ROPE_INTERLEAVED_LEGACY=1.
//
// Grid: [1, batch_size, 1], 256 threads per position. Each row computes
// its angles once in shared memory, then consecutive threads rotate adjacent
// dimensions across Q and K heads. Q/K are
// [batch_size × n_heads × head_dim] row-major.
//
// Optional trailing `angle_out` (explicit nullptr for every launch except the A3B F5
// K-only rope): stores each row's computed cos/sin floats ([n_rot] per row, cos then sin)
// without altering any rotation arithmetic. Launchers MUST pass the argument explicitly.

extern "C" __global__ void rope_partial_halfsplit_batched_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ positions,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int n_rot,
    float freq_base,
    int batch_size,
    int pos_offset,  // added to positions[b] for the RoPE angle (NOT the KV slot);
                     // carries kv_cache.compact_offset so post-eviction batched
                     // Q/K rotate at absolute phase matching the cached keys. 0 = no-op.
    float* __restrict__ angle_out  // optional [batch_size * n_rot] angle table; nullptr = off
) {
    int b = blockIdx.y;
    int half = n_rot / 2;
    if (b >= batch_size) return;

    // The launch supplies 2 * half floats of dynamic LDS. Retain precisely the
    // incumbent angle expressions and operation order for bit-identical output.
    extern __shared__ float angles[];
    float* cos_angles = angles;
    float* sin_angles = angles + half;
    int pos = positions[b] + pos_offset;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)n_rot);
        float angle = (float)pos * freq;
        cos_angles[i] = cosf(angle);
        sin_angles[i] = sinf(angle);
        // Optional capture of the exact angle floats the rotations below consume
        // (A3B F5 PM Q prep rotates Q by this same table). Row layout [n_rot]:
        // cos[0..half], sin[half..n_rot]. Null for every other caller.
        if (angle_out != nullptr) {
            angle_out[(long long)b * n_rot + i] = cos_angles[i];
            angle_out[(long long)b * n_rot + half + i] = sin_angles[i];
        }
    }
    __syncthreads();

    int q_stride = n_heads_q * head_dim;
    int k_stride = n_heads_k * head_dim;
    for (int j = threadIdx.x; j < n_heads_q * half; j += blockDim.x) {
        int h = j / half;
        int i = j - h * half;
        int base = b * q_stride + h * head_dim;
        float cos_a = cos_angles[i];
        float sin_a = sin_angles[i];
        float v0 = q[base + i];
        float v1 = q[base + i + half];
        q[base + i]        = v0 * cos_a - v1 * sin_a;
        q[base + i + half] = v0 * sin_a + v1 * cos_a;
    }

    for (int j = threadIdx.x; j < n_heads_k * half; j += blockDim.x) {
        int h = j / half;
        int i = j - h * half;
        int base = b * k_stride + h * head_dim;
        float cos_a = cos_angles[i];
        float sin_a = sin_angles[i];
        float v0 = k[base + i];
        float v1 = k[base + i + half];
        k[base + i]        = v0 * cos_a - v1 * sin_a;
        k[base + i + half] = v0 * sin_a + v1 * cos_a;
    }
}
