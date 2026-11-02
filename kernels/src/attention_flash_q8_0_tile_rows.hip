// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Multi-row Q8_0 flash attention tile — ONE KV scan serves ROWS query rows.
//
// attention_flash_q8_0_tile_batched grids [n_heads, tiles, ROW], so a verify
// batch re-reads the whole KV once per row. Here a block owns (head, tile,
// row-group) and keeps ROWS × (Q, running max, running sum, accumulator) in
// registers, so each K/V block is fetched once and reused across the group.
// That is the only reason this kernel exists; the arithmetic is the same
// two-pass softmax expressed in the online (FlashAttention) form, because a
// per-row score array would need LDS and cost the occupancy the register
// form keeps.
//
// Grid: [n_heads, max_tiles, ceil(rows_valid / ROWS)]. Block: [32, 1, 1].
// LDS: none.
// Partials: [row][head][max_tiles][2 + head_dim] — identical to the batched
// tile kernel, so attention_flash_asym_reduce_batched consumes it unchanged.
//
// Scope: full causal, single KV arena (no slot descriptors), no tree bias,
// no sliding window. The launcher refuses anything else.
#include <hip/hip_runtime.h>

static __device__ __forceinline__ unsigned int rows_load_u32(const unsigned char* p) {
    unsigned int packed;
    __builtin_memcpy(&packed, p, sizeof(packed));
    return packed;
}

static __device__ __forceinline__ unsigned long long rows_load_u64(const unsigned char* p) {
    unsigned long long packed;
    __builtin_memcpy(&packed, p, sizeof(packed));
    return packed;
}

template <int DPT>
static __device__ __forceinline__ void rows_dequant(
    const unsigned char* __restrict__ blk, int bj, float scale, float* __restrict__ out
) {
    if (DPT == 4) {
        const unsigned int packed = rows_load_u32(blk + 2 + bj);
        #pragma unroll
        for (int i = 0; i < 4; i++)
            out[i] = scale * (float)(signed char)(packed >> (8 * i));
    } else {
        const unsigned long long packed = rows_load_u64(blk + 2 + bj);
        #pragma unroll
        for (int i = 0; i < 8; i++)
            out[i] = scale * (float)(signed char)(packed >> (8 * i));
    }
}

// Five-stage XOR butterfly over the wave: each lane holds DPT partial
// products of one score, so every score costs one full cross-lane reduction.
// Spreading KV rows across lanes instead would remove it, but on gfx1100 the
// LDS staging that layout needs costs more occupancy than the shuffle costs
// time (measured: 1058-1989 us against 706 us for this form).
__device__ __forceinline__ float rows_reduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
    return v;
}

template <int ROWS, int DPT>
static __device__ __forceinline__ void flash_rows_body(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float scale_attn,
    int tile_size,
    int max_tiles,
    int batch_offset,
    int rows_valid
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tile_id = blockIdx.y;
    const int tid = threadIdx.x;
    const int row0 = (int)blockIdx.z * ROWS;

    int seq[ROWS];
    int seq_max = 0;
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const int row = row0 + r;
        seq[r] = (row < rows_valid) ? (positions[batch_offset + row] + 1) : 0;
        seq_max = max(seq_max, seq[r]);
    }
    const int tile_start = tile_id * tile_size;
    if (tile_start >= seq_max) return;
    const int tile_end = min(tile_start + tile_size, seq_max);
    const int tile_len = tile_end - tile_start;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int blocks_per_head = head_dim / 32;
    const int per_pos_bytes = n_kv_heads * blocks_per_head * 34;
    const int q_dim = n_heads * head_dim;
    const int d0 = tid * DPT;
    const int bj = d0 % 32;
    const int blk_off = (kv_h * blocks_per_head + d0 / 32) * 34;

    float mq[ROWS][DPT];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const int row = (row0 + r < rows_valid) ? (row0 + r) : 0;
        const float* qh = q + (size_t)row * q_dim + h * head_dim;
        #pragma unroll
        for (int i = 0; i < DPT; i++) mq[r][i] = qh[d0 + i];
    }

    float acc[ROWS][DPT];
    float run_max[ROWS];
    float run_sum[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        run_max[r] = -1e30f;
        run_sum[r] = 0.0f;
        #pragma unroll
        for (int i = 0; i < DPT; i++) acc[r][i] = 0.0f;
    }

    for (int t = tile_start; t < tile_end; t++) {
        const unsigned char* kb = k_cache + (size_t)t * per_pos_bytes + blk_off;
        float kd[DPT];
        rows_dequant<DPT>(kb, bj, (float)*((const _Float16*)kb), kd);
        float s[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            float p = 0.0f;
            #pragma unroll
            for (int i = 0; i < DPT; i++) p += mq[r][i] * kd[i];
            s[r] = rows_reduce_sum(p) * scale_attn;
        }
        const unsigned char* vb = v_cache + (size_t)t * per_pos_bytes + blk_off;
        float vd[DPT];
        rows_dequant<DPT>(vb, bj, (float)*((const _Float16*)vb), vd);
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            // Causal mask per row: rows in one group end at different positions.
            if (t >= seq[r]) continue;
            const float sr = s[r];
            if (sr > run_max[r]) {
                const float corr = __builtin_amdgcn_exp2f(run_max[r] - sr);
                run_sum[r] *= corr;
                #pragma unroll
                for (int i = 0; i < DPT; i++) acc[r][i] *= corr;
                run_max[r] = sr;
            }
            const float e = __builtin_amdgcn_exp2f(sr - run_max[r]);
            run_sum[r] += e;
            #pragma unroll
            for (int i = 0; i < DPT; i++) acc[r][i] += e * vd[i];
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const int row = row0 + r;
        if (row >= rows_valid) continue;
        float* p = partials
                 + ((long long)row * n_heads + h) * max_tiles * (2 + head_dim)
                 + (long long)tile_id * (2 + head_dim);
        if (tid == 0) {
            p[0] = run_max[r] * 0.69314718f;
            p[1] = run_sum[r];
        }
        #pragma unroll
        for (int i = 0; i < DPT; i++) p[2 + d0 + i] = acc[r][i];
    }
}

#define HIPFIRE_FLASH_ROWS_ENTRY(ROWS, DPT)                                     \
extern "C" __launch_bounds__(32)                                                \
__global__ void attention_flash_q8_0_rows##ROWS##_d##DPT(                        \
    const float* __restrict__ q,                                                \
    const unsigned char* __restrict__ k_cache,                                  \
    const unsigned char* __restrict__ v_cache,                                  \
    float* __restrict__ partials,                                               \
    const int* __restrict__ positions,                                          \
    int n_heads, int n_kv_heads, int head_dim, float scale_attn,                 \
    int tile_size, int max_tiles, int batch_offset, int rows_valid              \
) {                                                                             \
    flash_rows_body<ROWS, DPT>(q, k_cache, v_cache, partials, positions,        \
                               n_heads, n_kv_heads, head_dim, scale_attn,       \
                               tile_size, max_tiles, batch_offset, rows_valid); \
}

HIPFIRE_FLASH_ROWS_ENTRY(4, 4)
HIPFIRE_FLASH_ROWS_ENTRY(8, 4)
HIPFIRE_FLASH_ROWS_ENTRY(4, 8)
HIPFIRE_FLASH_ROWS_ENTRY(8, 8)
