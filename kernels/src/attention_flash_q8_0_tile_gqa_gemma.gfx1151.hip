// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

// GQA-shared Q8_0 decode flash tile for the Gemma 4 26B-A4B shapes (gfx1151):
// head_dim 256 / GQA group 2 (sliding layers, window > 0) and head_dim 512 /
// GQA group 8 (full layers). One 256-thread workgroup per (kv head, tile of
// 128 keys) serves every q head of its kv head, so each K/V row is read once
// instead of once per q head, and a tile is spread over 8 waves instead of one.
//
// Same 13-argument ABI, window semantics and `[n_heads, max_tiles, 2 + hd]`
// f32 partials as `attention_flash_q8_0_tile`, consumed by the unchanged
// `attention_flash_q8_0_reduce`. Not bit-identical to it: the dot products
// and V sums associate differently. Tiles wholly below the window publish an
// empty partial (sum 0), which the reduce skips. Workgroup y serves tiles
// y, y + gridDim.y, ... so the host can cap the grid.
#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
#error "attention_flash_q8_0_tile_gqa_gemma requires --offload-arch=gfx1151"
#endif

#define GG_TILE 128
#define GG_THREADS 256
#define GG_BLK 34

// Unaligned 32-bit load (RDNA3 unaligned access mode), as the reference
// tile reads its codes.
__device__ __forceinline__ unsigned int gg_u32(const unsigned char* p) {
    return *(const unsigned int*)p;
}

__device__ __forceinline__ float gg_f16(const unsigned char* p) {
    return (float)*(const _Float16*)p;
}

__device__ __forceinline__ float gg_code(unsigned int w, int i) {
    return (float)(int)(signed char)(w >> (8 * i));
}

template <int HD, int G>
__device__ __forceinline__ void gg_tile(
    const float* __restrict__ q_lds, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials, float* __restrict__ sc,
    int n_kv_heads, int max_seq, int max_tiles, float scale_attn, int window, int seq_len,
    int win_lo, int kv_h, int tile_id) {
    constexpr int NB = HD / 32;        // Q8 blocks per kv head
    constexpr int NBH = NB / 2;        // blocks per scoring half
    constexpr int DPT = HD / GG_THREADS;  // V dims per thread
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int tile_start = tile_id * GG_TILE;
    const int tile_len = min(GG_TILE, seq_len - tile_start);
    const int h0 = kv_h * G;
    const size_t row_stride = (size_t)n_kv_heads * NB * GG_BLK;
    const int head_blk = kv_h * NB;

    // ── Scores: thread pair (2t, 2t+1) owns key t; each covers half of HD.
    {
        const int t = tid >> 1;
        const int half = tid & 1;
        const int tc = min(t, tile_len - 1);
        const int key = tile_start + tc;
        const int slot = window > 0 ? key % max_seq : key;
        const unsigned char* kr = k_cache + (size_t)slot * row_stride + (size_t)(head_blk + half * NBH) * GG_BLK;
        float acc[G];
#pragma unroll
        for (int g = 0; g < G; ++g) acc[g] = 0.0f;
#pragma unroll 2
        for (int b = 0; b < NBH; ++b) {
            const unsigned char* blk = kr + b * GG_BLK;
            const float ks = gg_f16(blk);
            unsigned int w[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) w[i] = gg_u32(blk + 2 + 4 * i);
            float a[32];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) a[4 * i + j] = ks * gg_code(w[i], j);
            }
            const int d0 = (half * NBH + b) * 32;
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float4* qv = (const float4*)(q_lds + g * HD + d0);
                float s = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float4 x = qv[i];
                    s = __builtin_fmaf(x.x, a[4 * i + 0], s);
                    s = __builtin_fmaf(x.y, a[4 * i + 1], s);
                    s = __builtin_fmaf(x.z, a[4 * i + 2], s);
                    s = __builtin_fmaf(x.w, a[4 * i + 3], s);
                }
                acc[g] += s;
            }
        }
#pragma unroll
        for (int g = 0; g < G; ++g) acc[g] += __shfl_xor(acc[g], 1);
        if (half == 0) {
            const bool valid = t < tile_len && tile_start + t >= win_lo;
#pragma unroll
            for (int g = 0; g < G; ++g) sc[g * GG_TILE + t] = valid ? acc[g] * scale_attn : -INFINITY;
        }
    }
    __syncthreads();

    // ── Softmax: one wave per head (G <= 8 waves); lane holds 4 keys.
    if (wave < G) {
        float* s_h = sc + wave * GG_TILE;
        float m = -1e30f;
#pragma unroll
        for (int i = 0; i < GG_TILE / 32; ++i) m = fmaxf(m, s_h[lane + 32 * i]);
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor(m, off));
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < GG_TILE / 32; ++i) {
            const float e = expf(s_h[lane + 32 * i] - m);
            s_h[lane + 32 * i] = e;
            sum += e;
        }
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor(sum, off);
        if (lane == 0) {
            float* p = partials + ((size_t)(h0 + wave) * max_tiles + tile_id) * (2 + HD);
            p[0] = m;
            p[1] = sum;
        }
    }
    __syncthreads();

    // ── V: thread owns DPT consecutive dims for all G heads.
    const int d = tid * DPT;
    const int vblk = head_blk + d / 32;
    const int coff = 2 + (d & 31);
    float out[G][DPT];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int j = 0; j < DPT; ++j) out[g][j] = 0.0f;
    const int t_lo = max(0, win_lo - tile_start);
#pragma unroll 4
    for (int t = t_lo; t < tile_len; ++t) {
        const int key = tile_start + t;
        const int slot = window > 0 ? key % max_seq : key;
        const unsigned char* vb = v_cache + (size_t)slot * row_stride + (size_t)vblk * GG_BLK;
        const float vs = gg_f16(vb);
        float a[DPT];
        if constexpr (DPT == 2) {
            const unsigned int c = *(const unsigned short*)(vb + coff);
            a[0] = vs * gg_code(c, 0);
            a[1] = vs * gg_code(c, 1);
        } else {
            a[0] = vs * (float)(int)(signed char)vb[coff];
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float w = sc[g * GG_TILE + t];
#pragma unroll
            for (int j = 0; j < DPT; ++j) out[g][j] = __builtin_fmaf(w, a[j], out[g][j]);
        }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        float* p = partials + ((size_t)(h0 + g) * max_tiles + tile_id) * (2 + HD) + 2 + d;
#pragma unroll
        for (int j = 0; j < DPT; ++j) p[j] = out[g][j];
    }
}

template <int HD, int G>
__device__ __forceinline__ void gg_kernel(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ pos_buf, int n_kv_heads, int max_seq, float scale_attn, int window) {
    __shared__ __attribute__((aligned(16))) float q_lds[G * HD];
    __shared__ __attribute__((aligned(16))) float sc[G * GG_TILE];
    const int kv_h = blockIdx.x;
    if (kv_h >= n_kv_heads) return;
    const int logical_seq_len = pos_buf[0] + 1;
    const int seq_len = window > 0 ? min(logical_seq_len, max_seq) : logical_seq_len;
    const int win_lo = window > 0 ? seq_len - window : 0;
    const int max_tiles = (max_seq + GG_TILE - 1) / GG_TILE;
    const int n_tiles = min((seq_len + GG_TILE - 1) / GG_TILE, max_tiles);
    const int tid = threadIdx.x;
    const int h0 = kv_h * G;

    bool q_loaded = false;
    for (int tile_id = blockIdx.y; tile_id < n_tiles; tile_id += gridDim.y) {
        if ((tile_id + 1) * GG_TILE <= win_lo) {
            // Wholly below the window: empty partial, skipped by the reduce.
            if (tid < G) {
                float* p = partials + ((size_t)(h0 + tid) * max_tiles + tile_id) * (2 + HD);
                p[0] = -1e30f;
                p[1] = 0.0f;
            }
            continue;
        }
        if (!q_loaded) {
            for (int i = tid; i < G * HD / 4; i += GG_THREADS)
                ((float4*)q_lds)[i] = ((const float4*)(q + (size_t)h0 * HD))[i];
            q_loaded = true;
        } else {
            __syncthreads();  // previous tile's sc readers are done
        }
        __syncthreads();
        gg_tile<HD, G>(q_lds, k_cache, v_cache, partials, sc, n_kv_heads, max_seq, max_tiles,
                       scale_attn, window, seq_len, win_lo, kv_h, tile_id);
    }
}

// Host admits head_dim 256 with n_heads == 2 * n_kv_heads, tile 128.
extern "C" __launch_bounds__(GG_THREADS, 1)
__global__ void attention_flash_q8_0_tile_gqa_d256g2_gfx1151(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ pos_buf, int n_heads, int n_kv_heads, int head_dim, int max_seq,
    float scale_attn, int tile_size, int window, int effective_seq_len) {
    (void)n_heads; (void)head_dim; (void)tile_size; (void)effective_seq_len;
    gg_kernel<256, 2>(q, k_cache, v_cache, partials, pos_buf, n_kv_heads, max_seq, scale_attn, window);
}

// Host admits head_dim 512 with n_heads == 8 * n_kv_heads, tile 128.
extern "C" __launch_bounds__(GG_THREADS, 1)
__global__ void attention_flash_q8_0_tile_gqa_d512g8_gfx1151(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ pos_buf, int n_heads, int n_kv_heads, int head_dim, int max_seq,
    float scale_attn, int tile_size, int window, int effective_seq_len) {
    (void)n_heads; (void)head_dim; (void)tile_size; (void)effective_seq_len;
    gg_kernel<512, 8>(q, k_cache, v_cache, partials, pos_buf, n_kv_heads, max_seq, scale_attn, window);
}
