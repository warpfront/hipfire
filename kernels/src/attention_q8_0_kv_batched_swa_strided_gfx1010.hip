// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx1010-owned strided QKV-row + sliding-window sibling for Spark chunk
// prefill. Plain single-arena K/V byte offsets (exact match of
// attention_q8_0_kv_swa). No tree_bias / slot descriptors.
//
// Grid: [n_heads, batch_size, 1]
// Q comes from qkv[row * q_stride + head * head_dim] (strided concatenated
// QKV rows). Out is packed [batch × n_heads × head_dim].
// Per-query causal bound: seq = positions[row] + 1; windowed t_lo as SWA.
// All score / exp / norm / V loops iterate [t_lo, seq) with the same
// reduction tree and ascending V order as the single-query SWA kernel.
// window == 0 → full causal; any positive window is supported.
// Bodies compile only when __gfx1010__ is defined (headers above stay live).

#if defined(__gfx1010__)

extern "C" __global__ void attention_q8_0_kv_batched_swa_strided_gfx1010(
    const float* __restrict__ qkv,              // [batch_size × q_stride] F32
    const unsigned char* __restrict__ k_cache,  // plain arena Q8_0
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,                    // [batch_size × n_heads × head_dim]
    const int* __restrict__ positions,          // [batch_size]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int window,
    int q_stride,
    int batch_size
) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    if (b >= batch_size) return;
    if (h >= n_heads) return;

    const int seq_len = positions[b] + 1;
    // Sliding-window lookback. window==0 is full causal (matches SWA).
    const int t_lo = (window > 0 && seq_len > window) ? (seq_len - window) : 0;
    // max_seq is the plain-arena capacity; keep the symbol live for ABI parity
    // with the host launch (host validates max_ctx_len <= max_seq).
    (void)max_seq;

    extern __shared__ float sdata[];

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const float* q_head = qkv + (long long)b * q_stride + h * head_dim;
    const int out_stride = n_heads * head_dim;

    const int blocks_per_head = head_dim / 32;
    const int total_blocks_per_pos = n_kv_heads * blocks_per_head;
    const int kv_head_block_start = kv_h * blocks_per_head;

    float* scores = sdata;
    float* workspace = sdata + seq_len;
    float* q_shared = workspace + nthreads;

    // Preload this batch row's Q head into shared memory.
    for (int d = tid; d < head_dim; d += nthreads)
        q_shared[d] = q_head[d];
    __syncthreads();

    // Phase 1: Q @ K^T with Q8_0 dequant over [t_lo, seq_len).
    // Plain KV offsets match attention_q8_0_kv_swa exactly.
    float local_max = -1e30f;
    for (int t = t_lo + tid; t < seq_len; t += nthreads) {
        float dot = 0.0f;
        for (int bi = 0; bi < blocks_per_head; bi++) {
            const unsigned char* blk = k_cache
                                      + (size_t)t * total_blocks_per_pos * 34
                                      + (kv_head_block_start + bi) * 34;
            float d = (float)*((const _Float16*)blk);
            const float* qb = q_shared + bi * 32;
            for (int j = 0; j < 32; j++) {
                dot += qb[j] * (d * (float)((signed char)blk[2 + j]));
            }
        }
        float s = dot * scale_attn;
        scores[t] = s;
        local_max = fmaxf(local_max, s);
    }

    workspace[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] = fmaxf(workspace[tid], workspace[tid + s]);
        __syncthreads();
    }
    float max_val = workspace[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int t = t_lo + tid; t < seq_len; t += nthreads) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    workspace[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] += workspace[tid + s];
        __syncthreads();
    }
    float sum_val = workspace[0];
    __syncthreads();

    for (int t = t_lo + tid; t < seq_len; t += nthreads) {
        scores[t] /= sum_val;
    }
    __syncthreads();

    // Phase 2: weighted V sum with Q8_0 dequant over [t_lo, seq_len),
    // ascending t order identical to single-query SWA.
    float* out_head = out + (long long)b * out_stride + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        int bi = d / 32;
        int bj = d % 32;
        for (int t = t_lo; t < seq_len; t++) {
            const unsigned char* blk = v_cache
                                      + (size_t)t * total_blocks_per_pos * 34
                                      + (kv_head_block_start + bi) * 34;
            float vd = (float)*((const _Float16*)blk) * (float)(signed char)blk[2 + bj];
            val += scores[t] * vd;
        }
        out_head[d] = val;
    }
}

#endif // defined(__gfx1010__)
