// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Bidirectional (non-causal) grouped-query attention for the LFM3-VL vision
// tower. Mirrors `vit_attention_f32` (full bidirectional softmax over all N
// patches of one frame) but takes SEPARATE Q / K / V buffers with distinct
// head counts (GQA), instead of a fused MHA QKV buffer.
//
// LFM3-VL vision attention: 16 query heads / 8 key-value heads (groups = 2),
// head_dim 64, is_causal=False. Every query patch attends to every key patch
// in the same frame (no mask). This is the vision encoder's one-shot prefill
// path — NOT the causal Q8 flash dispatch used by the text decoder.
//
// Layouts (row-major, one frame of N patches):
//   q  : [N, n_heads * head_dim]
//   k  : [N, n_kv   * head_dim]
//   v  : [N, n_kv   * head_dim]
//   out: [N, n_heads * head_dim]
//
// Grid = [num_heads, N]; one block computes one (head, query patch). Block
// size must be a power of two >= max(N reduced, head_dim). Shared memory:
// scores[N] + workspace[block_size].

extern "C" __global__ void vit_attention_gqa_f32(
    const float* __restrict__ q,    // [N, n_heads*head_dim]
    const float* __restrict__ k,    // [N, n_kv*head_dim]
    const float* __restrict__ v,    // [N, n_kv*head_dim]
    float* __restrict__ out,        // [N, n_heads*head_dim]
    int N, int n_heads, int n_kv, int head_dim, float scale
) {
    extern __shared__ float sdata[];
    int head = blockIdx.x;
    int qi = blockIdx.y;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    if (head >= n_heads || qi >= N) return;

    int q_stride = n_heads * head_dim;
    int kv_stride = n_kv * head_dim;
    // GQA: query head `head` reads kv head `kv_head = head * n_kv / n_heads`.
    int groups = n_heads / n_kv;          // query heads per kv head
    int kv_head = head / groups;
    int q_off = head * head_dim;
    int kv_off = kv_head * head_dim;

    float* scores = sdata;

    // Phase 1: Q[qi] @ K[j]^T  (bidirectional — all N keys)
    const float* q_ptr = q + qi * q_stride + q_off;
    float local_max = -1e30f;
    for (int j = tid; j < N; j += nthreads) {
        const float* k_ptr = k + j * kv_stride + kv_off;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q_ptr[d] * k_ptr[d];
        float s = dot * scale;
        scores[j] = s;
        local_max = fmaxf(local_max, s);
    }

    float* ws = sdata + N;
    ws[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) ws[tid] = fmaxf(ws[tid], ws[tid + s]);
        __syncthreads();
    }
    float max_val = ws[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < N; j += nthreads) {
        float e = expf(scores[j] - max_val);
        scores[j] = e;
        local_sum += e;
    }
    ws[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) ws[tid] += ws[tid + s];
        __syncthreads();
    }
    float sum_val = ws[0];
    __syncthreads();

    for (int j = tid; j < N; j += nthreads)
        scores[j] /= sum_val;
    __syncthreads();

    // Phase 2: weighted V sum
    float* out_row = out + qi * q_stride + head * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        for (int j = 0; j < N; j++)
            val += scores[j] * v[j * kv_stride + kv_off + d];
        out_row[d] = val;
    }
}
