// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//
// Batched (prefill) causal attention over a Q8_0 KV cache on gfx11 wave32
// WMMA, GQA-shared, optional sliding window. Flash-attention style: one pass
// over the keys with an online softmax, no partials.
//
// A workgroup (4 waves) owns one KV head and 16 "query rows": the G = n_heads
// / n_kv_heads q heads of 16 / G consecutive batch rows, so each K/V tile is
// read once for all of them. Row r of the tile is batch row 16/G * tile +
// r / G, q head G * kv_h + r % G.
//
// Per 32-key tile:
//   S = Q K^T   each wave covers a quarter of head_dim (Q in registers,
//               K dequantized from the cache straight into the B operand);
//               the four partial S tiles are summed in LDS.
//   softmax     online per row (scale, causal/window mask), P in f16 in LDS.
//   O += P V    each wave owns a quarter of head_dim of O; V is dequantized
//               into LDS transposed ([dim][key]) so the B operand is
//               contiguous.
// Q, K, P and V enter WMMA as f16; S, the softmax state and O are f32.
//
// Grid: [ceil(batch * G / 16), n_kv_heads]. Block: [128]. Requires
// head_dim in {256, 512}, G in {1, 2, 4, 8, 16}, K/V rows position-indexed
// (no ring), positions[] increasing within a launch.
#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define PG_KT 32
#define PG_WAVES 4
#define PG_THREADS (PG_WAVES * 32)

__device__ __forceinline__ unsigned short pg_u16(const unsigned char* p) {
    unsigned short v;
    __builtin_memcpy(&v, p, 2);
    return v;
}

__device__ __forceinline__ float pg_f16(const unsigned char* p) {
    return (float)__builtin_bit_cast(_Float16, pg_u16(p));
}

template <int D>
__device__ __forceinline__ void pg_body(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ out,
    const int* __restrict__ positions, int n_heads, int n_kv_heads, int batch, float scale,
    int window) {
    constexpr int NB = D / 32;            // Q8 blocks per kv head
    constexpr int DW = D / PG_WAVES;      // dims per wave
    constexpr int KS = DW / 16;           // 16-dim k-steps per wave
    __shared__ float s_part[PG_WAVES][16][PG_KT];
    __shared__ __attribute__((aligned(32))) _Float16 p_lds[16][PG_KT];
    __shared__ __attribute__((aligned(32))) _Float16 vt[D][PG_KT];
    __shared__ float alpha_lds[16];
    __shared__ float l_lds[16];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int kv_h = blockIdx.y;
    const int g = n_heads / n_kv_heads;
    const int rows_per_tile = 16 / g;
    const int row0 = blockIdx.x * rows_per_tile;
    const size_t row_stride = (size_t)n_kv_heads * NB * 34;
    const size_t head_off = (size_t)kv_h * NB * 34;

    // Tile key range from the first and last valid batch rows.
    const int last = min(row0 + rows_per_tile, batch) - 1;
    const int pos_min = positions[row0];
    const int pos_max = positions[last];
    const int k_lo = window > 0 ? max(0, pos_min - window + 1) : 0;
    const int k_hi = pos_max;

    // Q A-operand: lane row r = lane & 15, this wave's dims, as f16.
    const int r_a = lane & 15;
    const int brow_a = min(row0 + r_a / g, batch - 1);
    const float* q_row = q + ((size_t)brow_a * n_heads + kv_h * g + r_a % g) * D + wave * DW;
    half16_t qa[KS];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
        for (int i = 0; i < 16; ++i) qa[ks][i] = (_Float16)q_row[ks * 16 + i];
    }

    // Softmax ownership: thread t owns row t / 8, keys 4 * (t % 8) .. +3.
    const int sm_row = tid >> 3;
    const int sm_k0 = (tid & 7) * 4;
    const int sm_brow = row0 + sm_row / g;
    const bool sm_valid = sm_brow < batch;
    const int sm_pos = sm_valid ? positions[sm_brow] : -1;
    float m_run = -1e30f;
    float l_run = 0.0f;

    float8_t acc[KS];
#pragma unroll
    for (int c = 0; c < KS; ++c) acc[c] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};

    for (int kt0 = k_lo; kt0 <= k_hi; kt0 += PG_KT) {
        // ── V tile, transposed into LDS (keys clamped into the range).
        for (int p = tid; p < PG_KT * NB; p += PG_THREADS) {
            const int j = p % PG_KT;
            const int b = p / PG_KT;
            const int t = min(kt0 + j, k_hi);
            const unsigned char* blk = v_cache + (size_t)t * row_stride + head_off + b * 34;
            const float vs = pg_f16(blk);
#pragma unroll
            for (int i = 0; i < 32; i += 2) {
                const unsigned short c2 = pg_u16(blk + 2 + i);
                vt[b * 32 + i][j] = (_Float16)(vs * (float)(signed char)(c2 & 0xff));
                vt[b * 32 + i + 1][j] = (_Float16)(vs * (float)(signed char)(c2 >> 8));
            }
        }

        // ── Partial S over this wave's dims, two 16-key blocks.
#pragma unroll
        for (int kb = 0; kb < PG_KT / 16; ++kb) {
            const int t = min(kt0 + kb * 16 + (lane & 15), k_hi);
            const unsigned char* krow = k_cache + (size_t)t * row_stride + head_off;
            float8_t s = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int ks = 0; ks < KS; ++ks) {
                const int d0 = wave * DW + ks * 16;
                const unsigned char* blk = krow + (d0 / 32) * 34;
                const float sc = pg_f16(blk);
                const unsigned char* codes = blk + 2 + (d0 & 31);
                half16_t kb16;
#pragma unroll
                for (int i = 0; i < 16; i += 2) {
                    const unsigned short c2 = pg_u16(codes + i);
                    kb16[i] = (_Float16)(sc * (float)(signed char)(c2 & 0xff));
                    kb16[i + 1] = (_Float16)(sc * (float)(signed char)(c2 >> 8));
                }
                s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[ks], kb16, s);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) s_part[wave][2 * j + (lane >> 4)][kb * 16 + (lane & 15)] = s[j];
        }
        __syncthreads();

        // ── Online softmax.
        {
            float sv[4];
            float tmax = -1e30f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int key = sm_k0 + i;
                const int t = kt0 + key;
                const bool ok = sm_valid && t <= sm_pos && (window <= 0 || t > sm_pos - window);
                const float v = (s_part[0][sm_row][key] + s_part[1][sm_row][key] +
                                 s_part[2][sm_row][key] + s_part[3][sm_row][key]) * scale;
                sv[i] = ok ? v : -1e30f;
                tmax = fmaxf(tmax, sv[i]);
            }
            tmax = fmaxf(tmax, __shfl_xor(tmax, 1));
            tmax = fmaxf(tmax, __shfl_xor(tmax, 2));
            tmax = fmaxf(tmax, __shfl_xor(tmax, 4));
            const float m_new = fmaxf(m_run, tmax);
            const float alpha = __expf(m_run - m_new);
            float psum = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float p = sv[i] > -1e29f ? __expf(sv[i] - m_new) : 0.0f;
                p_lds[sm_row][sm_k0 + i] = (_Float16)p;
                psum += p;
            }
            psum += __shfl_xor(psum, 1);
            psum += __shfl_xor(psum, 2);
            psum += __shfl_xor(psum, 4);
            l_run = l_run * alpha + psum;
            m_run = m_new;
            if ((tid & 7) == 0) alpha_lds[sm_row] = alpha;
        }
        __syncthreads();

        // ── Rescale O, then O += P V over this wave's dims.
        {
            float al[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) al[j] = alpha_lds[2 * j + (lane >> 4)];
#pragma unroll
            for (int kb = 0; kb < PG_KT / 16; ++kb) {
                const half16_t pa = *(const half16_t*)&p_lds[lane & 15][kb * 16];
#pragma unroll
                for (int c = 0; c < KS; ++c) {
                    if (kb == 0) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) acc[c][j] *= al[j];
                    }
                    const int d = wave * DW + c * 16 + (lane & 15);
                    const half16_t vb = *(const half16_t*)&vt[d][kb * 16];
                    acc[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, vb, acc[c]);
                }
            }
        }
        __syncthreads();
    }

    if ((tid & 7) == 0) l_lds[sm_row] = l_run;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int r = 2 * j + (lane >> 4);
        const int brow = row0 + r / g;
        if (brow >= batch) continue;
        const float l = l_lds[r];
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        float* o = out + ((size_t)brow * n_heads + kv_h * g + r % g) * D + wave * DW;
#pragma unroll
        for (int c = 0; c < KS; ++c) o[c * 16 + (lane & 15)] = acc[c][j] * inv;
    }
}

#define PG_ENTRY(NAME, D)                                                                       \
    extern "C" __launch_bounds__(PG_THREADS, 1) __global__ void NAME(                           \
        const float* __restrict__ q, const unsigned char* __restrict__ k_cache,                 \
        const unsigned char* __restrict__ v_cache, float* __restrict__ out,                     \
        const int* __restrict__ positions, int n_heads, int n_kv_heads, int batch, float scale, \
        int window) {                                                                           \
        pg_body<D>(q, k_cache, v_cache, out, positions, n_heads, n_kv_heads, batch, scale,      \
                   window);                                                                     \
    }

PG_ENTRY(attention_q8_0_prefill_gqa_wmma_d256, 256)
PG_ENTRY(attention_q8_0_prefill_gqa_wmma_d512, 512)
