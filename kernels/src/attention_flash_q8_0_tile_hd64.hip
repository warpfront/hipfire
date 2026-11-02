// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// head_dim == 64 twin of attention_flash_q8_0_tile.hip: same 13-argument ABI,
// same `[n_heads, max_tiles, 2 + head_dim]` partials, so the shared
// attention_flash_q8_0_reduce is reused unchanged. The reference tile walks
// head_dim in 128-wide halves (32 lanes x 4) and would read the next head's
// Q/K at head_dim 64 (SmolLM2, Llama-3.2-1B, TinyLlama).
//
// Grid: [n_heads, n_tiles, 1]. Block: [32, 1, 1] (one WAVE32).
// The wave splits into two 16-lane halves; each half scores (and later
// accumulates V for) every other position of the tile, 4 dims per lane, so
// every lane keeps the reference tile's 4-byte K/V footprint. Scores reduce
// within a half (xor 8..1); V partials combine across halves (xor 16) once.
// LDS: tile_size floats for scores (the launcher's extra head_dim floats are
// unused: Q stays in registers).
#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_q8_0_tile_hd64(
    const float* __restrict__ q,         // [n_heads, 64]
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,        // [n_heads, max_tiles, 2 + 64]
    const int* __restrict__ pos_buf,     // pos_buf[0] = last position
    int n_heads,
    int n_kv_heads,
    int head_dim,                        // must be 64 (launcher-gated)
    int max_seq,
    float scale_attn,
    int tile_size,
    int window,                          // sliding-window span; <= 0 = full causal
    int effective_seq_len                // retained ABI; dynamic length comes from pos_buf
) {
    (void)effective_seq_len;
    (void)head_dim;
    constexpr int HD = 64;
    const int logical_seq_len = pos_buf[0] + 1;
    const int seq_len = (window > 0) ? min(logical_seq_len, max_seq) : logical_seq_len;
    const int win_lo = (window > 0) ? (seq_len - window) : 0;
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tile_id = blockIdx.y;
    const int tile_start = tile_id * tile_size;
    const int tile_end = min(tile_start + tile_size, seq_len);
    if (tile_start >= seq_len) return;
    const int tile_len = tile_end - tile_start;

    const int kv_h = h / (n_heads / n_kv_heads);
    const int tid = threadIdx.x;
    const int sub = tid >> 4;            // position parity this half handles
    const int l = tid & 15;
    const int d = l * 4;                 // 4 dims per lane
    const int row_stride = n_kv_heads * 2 * 34;
    // Block (d / 32) of this kv head, then 4 bytes at (d % 32) after the f16 scale.
    const int blk_off = (kv_h * 2 + (d >> 5)) * 34;
    const int code_off = 2 + (d & 31);

    extern __shared__ float scores[];

    const float* q_head = q + h * HD;
    const float q0 = q_head[d + 0];
    const float q1 = q_head[d + 1];
    const float q2 = q_head[d + 2];
    const float q3 = q_head[d + 3];

    int valid_start = win_lo - tile_start;
    if (valid_start < 0) valid_start = 0;
    if (valid_start > tile_len) valid_start = tile_len;
    for (int i = tid; i < valid_start; i += 32) scores[i] = -INFINITY;

    // ═══ Phase A: scores, 8 positions per wave iteration ═══
    for (int t_base = valid_start; t_base < tile_len; t_base += 8) {
        float part[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int tl = t_base + 2 * j + sub;
            float acc = 0.0f;
            if (tl < tile_len) {
                const int t = tile_start + tl;
                const int slot = (window > 0) ? (t % max_seq) : t;
                const unsigned char* kb = k_cache + (size_t)slot * row_stride + blk_off;
                const float ks = (float)*((const _Float16*)kb);
                acc = q0 * (ks * (float)(signed char)kb[code_off + 0])
                    + q1 * (ks * (float)(signed char)kb[code_off + 1])
                    + q2 * (ks * (float)(signed char)kb[code_off + 2])
                    + q3 * (ks * (float)(signed char)kb[code_off + 3]);
            }
            part[j] = acc;
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            for (int off = 8; off > 0; off >>= 1)
                part[j] += __shfl_xor(part[j], off);
        }
        if (l == 0) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int tl = t_base + 2 * j + sub;
                if (tl < tile_len) scores[tl] = part[j] * scale_attn;
            }
        }
    }
    __syncthreads();

    // ═══ Phase B: tile max ═══
    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 32)
        local_max = fmaxf(local_max, scores[i]);
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    const float tile_max = local_max;

    // ═══ Phase C: exp + sum ═══
    float local_sum = 0.0f;
    for (int i = tid; i < tile_len; i += 32) {
        const float e = expf(scores[i] - tile_max);
        scores[i] = e;
        local_sum += e;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor(local_sum, off);
    __syncthreads();

    const int max_tiles = (max_seq + tile_size - 1) / tile_size;
    float* p = partials + (h * max_tiles + tile_id) * (2 + HD);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = local_sum;
    }

    // ═══ Phase D: V accumulation, each half over its parity ═══
    float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;
    for (int t_base = 0; t_base < tile_len; t_base += 8) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int tl = t_base + 2 * j + sub;
            if (tl < tile_len) {
                const float w = scores[tl];
                const int t = tile_start + tl;
                const int slot = (window > 0) ? (t % max_seq) : t;
                const unsigned char* vb = v_cache + (size_t)slot * row_stride + blk_off;
                const float vs = (float)*((const _Float16*)vb);
                out0 += w * (vs * (float)(signed char)vb[code_off + 0]);
                out1 += w * (vs * (float)(signed char)vb[code_off + 1]);
                out2 += w * (vs * (float)(signed char)vb[code_off + 2]);
                out3 += w * (vs * (float)(signed char)vb[code_off + 3]);
            }
        }
    }
    out0 += __shfl_xor(out0, 16);
    out1 += __shfl_xor(out1, 16);
    out2 += __shfl_xor(out2, 16);
    out3 += __shfl_xor(out3, 16);
    if (sub == 0) {
        p[2 + d + 0] = out0;
        p[2 + d + 1] = out1;
        p[2 + d + 2] = out2;
        p[2 + d + 3] = out3;
    }
}
