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
// GRID-1: the grid is FLATTENED to [n_heads * N] (head = blockIdx.x / N,
// qi = blockIdx.x % N) — the old [n_heads, N] 2-D grid put N on gridDim.y,
// which caps at 65535 and tripped exactly when the LFM3-VL pixel budget
// (longest_edge 16.7M px → N ≈ 65536) was honored.
//
// The key loop is TILED with an online (running max/sum) softmax so the
// score scratch is a fixed TILE-sized LDS buffer instead of N floats — the
// old (N + block) * 4 bytes exceeded the ~64 KB workgroup LDS ceiling at
// N ≳ 16k. Keys are revisited in pass 2 (scores recomputed from Q·K), which
// doubles the QK work but keeps the launcher API scratch-free; this is a
// one-shot prefill path, not the decode hot loop.
//
// Numerics: online softmax is mathematically identical to the single-pass
// max/sum/normalize (floating-point association order differs slightly).

#define VIT_ATTN_TILE 2048

extern "C" __global__ void vit_attention_gqa_f32(
    const float* __restrict__ q,    // [N, n_heads*head_dim]
    const float* __restrict__ k,    // [N, n_kv*head_dim]
    const float* __restrict__ v,    // [N, n_kv*head_dim]
    float* __restrict__ out,        // [N, n_heads*head_dim]
    int N, int n_heads, int n_kv, int head_dim, float scale
) {
    extern __shared__ float sdata[];
    if (N <= 0) return; // guard the blockIdx.x / N decomposition below
    int head = blockIdx.x / N;
    int qi = blockIdx.x % N;
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

    float* tile_scores = sdata;                    // [VIT_ATTN_TILE]
    float* ws = sdata + VIT_ATTN_TILE;             // [nthreads] reduction scratch

    const float* q_ptr = q + qi * q_stride + q_off;

    // ── Pass 1: online softmax over key tiles → (max_val, sum_val) ──
    float run_max = -1e30f;
    float run_sum = 0.0f;
    for (int tile_start = 0; tile_start < N; tile_start += VIT_ATTN_TILE) {
        int tile_n = min(VIT_ATTN_TILE, N - tile_start);
        float tile_max = -1e30f;
        for (int t = tid; t < tile_n; t += nthreads) {
            const float* k_ptr = k + (tile_start + t) * kv_stride + kv_off;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dot += q_ptr[d] * k_ptr[d];
            float s = dot * scale;
            tile_scores[t] = s;
            tile_max = fmaxf(tile_max, s);
        }
        // Tile max reduce.
        ws[tid] = tile_max;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) ws[tid] = fmaxf(ws[tid], ws[tid + s]);
            __syncthreads();
        }
        tile_max = ws[0];
        __syncthreads();

        // Online rescale: fold this tile into the running (max, sum).
        float m_new = fmaxf(run_max, tile_max);
        float correction = (run_max == -1e30f) ? 0.0f : expf(run_max - m_new);
        float tile_sum = 0.0f;
        for (int t = tid; t < tile_n; t += nthreads)
            tile_sum += expf(tile_scores[t] - m_new);
        ws[tid] = tile_sum;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) ws[tid] += ws[tid + s];
            __syncthreads();
        }
        tile_sum = ws[0];
        __syncthreads();

        run_sum = run_sum * correction + tile_sum;
        run_max = m_new;
    }
    float max_val = run_max;
    float sum_val = run_sum;

    // ── Pass 2: recompute scores per tile, accumulate the weighted V sum ──
    float* out_row = out + qi * q_stride + head * head_dim;
    float inv_sum = 1.0f / sum_val;
    for (int d = tid; d < head_dim; d += nthreads)
        out_row[d] = 0.0f;
    __syncthreads();

    for (int tile_start = 0; tile_start < N; tile_start += VIT_ATTN_TILE) {
        int tile_n = min(VIT_ATTN_TILE, N - tile_start);
        for (int t = tid; t < tile_n; t += nthreads) {
            const float* k_ptr = k + (tile_start + t) * kv_stride + kv_off;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dot += q_ptr[d] * k_ptr[d];
            tile_scores[t] = expf(dot * scale - max_val) * inv_sum;
        }
        __syncthreads();
        for (int d = tid; d < head_dim; d += nthreads) {
            float val = 0.0f;
            for (int t = 0; t < tile_n; t++)
                val += tile_scores[t] * v[(tile_start + t) * kv_stride + kv_off + d];
            out_row[d] += val;
        }
        __syncthreads();
    }
}
