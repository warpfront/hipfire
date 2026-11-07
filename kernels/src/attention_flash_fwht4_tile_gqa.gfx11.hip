// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and LICENSE-MIT in the project root.

// GQA-shared fwht4-K decode flash tile (gfx1100, head_dim 256, GQA group 6,
// tile 128, full causal). One 256-thread workgroup per (kv head, tile)
// serves all six q heads of its kv head, so each K/V tile is read from
// memory once instead of once per q head.
//
// Occupancy design (v2): NO K tile in LDS. Keys are partitioned across all
// 8 waves (16 keys each); each wave reads its keys straight from global
// with scalar-identical addressing (coalesced per-lane ushorts, broadcast
// cnorm) and shares them across the six heads through the LDS score array.
// V streams in 32-key slabs through LDS. Q is FWHT-rotated once at stage-in
// (waves 0-5 own heads 0-5). LDS is ~17.5 KB (Q 6 KB + scores 3 KB + V slab
// 8.5 KB worst-case Q8) -> 3 WGs/CU instead of 2, with ~11 barriers per
// tile instead of ~20.
//
// Output partials use the SAME [n_heads, max_tiles, 2 + 256] layout as
// attention_flash_fwht4_tile (rotated space for lloyd V, Q8_0 normal space
// for v_mode 8) and are consumed unchanged by the existing reduce kernels.
// Per-lane arithmetic mirrors the scalar tile exactly (same DOT4 shape per
// (head, key), same fma accumulation order in V); the bar is KLD parity +
// greedy identity (P1 precedent), not claimed bit-identity — though the
// oracle currently holds bit-identical.
//
// v_mode: 8 = Q8_0 V, 4 = lloyd4 V (256-wide FWHT-rotated). Other modes
// keep the scalar path (host-gated).
#include <hip/hip_runtime.h>
// turbo_common.h (TURBO_C4_256, fwht_shfl_forward) is prepended by the
// launcher, NOT #included here: the runtime compile has no -I to
// kernels/src (same pattern as the FA2 combo modules).

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1100__)
#error "attention_flash_fwht4_tile_gqa_gfx1100 requires --offload-arch=gfx1100"
#endif

#define FGQA_TILE 128
#define FGQA_HD 256
#define FGQA_GROUP 6
#define FGQA_WAVES 8
#define FGQA_THREADS 256
// K record: [f32 cnorm][128 B nibbles] per (pos, kv head).
#define FGQA_K_SEG 132
// V slab: 32 keys; lloyd4 132 B/head, Q8_0 272 B/head (worst case sizes it).
#define FGQA_V_TILE 32
// LDS carve: V slab (worst case) + Q 6*256 floats + scores 6*128 floats.
#define FGQA_V_DWORDS (FGQA_V_TILE * 272 / 4)
// Each of the 8 waves owns 16 keys of the tile for scoring.
#define FGQA_KEYS_PER_WAVE (FGQA_TILE / FGQA_WAVES)

extern "C" __launch_bounds__(FGQA_THREADS, 1)
__global__ void attention_flash_fwht4_tile_gqa_gfx11(
    const float* __restrict__ q,                // [n_heads, 256], f32
    const unsigned char* __restrict__ k_cache,  // fwht4 K rows
    const unsigned char* __restrict__ v_cache,  // Q8_0 or lloyd4 V rows
    float* __restrict__ partials,               // [n_heads, max_tiles, 2 + 256]
    const int* __restrict__ pos_buf,            // pos_buf[0] = last position
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles,
    int v_mode)
{
    const int seq_len = pos_buf[0] + 1;
    const int kv_h = blockIdx.x;
    if (kv_h >= n_kv_heads) return;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    __shared__ __attribute__((aligned(16))) unsigned int v_lds[FGQA_V_DWORDS];
    __shared__ __attribute__((aligned(16))) float q_lds[FGQA_GROUP * FGQA_HD];
    __shared__ __attribute__((aligned(16))) float sc[FGQA_GROUP * FGQA_TILE];

    const int h0 = kv_h * FGQA_GROUP;

    // ---- Q stage + FWHT rotation (waves 0-5 own heads 0-5) ----
    // Each wave rotates its head in two half-rounds (32 cooperating lanes
    // per 128-dim half, ds_swizzle wave-confined, lane id passed).
    if (wave < FGQA_GROUP) {
        const float* q_head = q + (size_t)(h0 + wave) * FGQA_HD;
        float* q_out = q_lds + (size_t)wave * FGQA_HD;
        for (int half = 0; half < 2; ++half) {
            const int d0 = half * 128 + lane * 4;
            float a = q_head[d0 + 0];
            float b = q_head[d0 + 1];
            float c = q_head[d0 + 2];
            float d = q_head[d0 + 3];
            fwht_shfl_forward(a, b, c, d, signs1, signs2, lane);
            q_out[d0 + 0] = a;
            q_out[d0 + 1] = b;
            q_out[d0 + 2] = c;
            q_out[d0 + 3] = d;
        }
    }
    __syncthreads();

    const int n_tiles = min((seq_len + FGQA_TILE - 1) / FGQA_TILE, max_tiles);
    const size_t k_bytes_per_pos = (size_t)n_kv_heads * FGQA_K_SEG;
    const size_t k_head_off = (size_t)kv_h * FGQA_K_SEG;

    for (int tile_id = blockIdx.y; tile_id < n_tiles; tile_id += gridDim.y) {
        if (tile_id != (int)blockIdx.y) __syncthreads();
        const int tile_start = tile_id * FGQA_TILE;
        const int tile_len = min(FGQA_TILE, seq_len - tile_start);

        // ---- Scores: all 8 waves participate, each wave owns 16 keys
        // (wave*16 .. wave*16+15). K is read straight from global with
        // scalar-identical addressing — coalesced per-lane ushorts plus a
        // broadcast cnorm — and shared across the six heads through sc.
        // No K LDS, no K-stage barrier. Per (head, key) the arithmetic is
        // the scalar tile's exactly (same DOT4, same wave tree, same
        // scale); heads run in two groups of three to bound registers.
        {
            for (int hg = 0; hg < 2; ++hg) {
                float mq[3][8];
#pragma unroll
                for (int h = 0; h < 3; ++h) {
                    const float* qh = q_lds + (size_t)(3 * hg + h) * FGQA_HD;
                    for (int half = 0; half < 2; ++half) {
                        const int d0 = half * 128 + lane * 4;
                        mq[h][half * 4 + 0] = qh[d0 + 0];
                        mq[h][half * 4 + 1] = qh[d0 + 1];
                        mq[h][half * 4 + 2] = qh[d0 + 2];
                        mq[h][half * 4 + 3] = qh[d0 + 3];
                    }
                }
                for (int k = 0; k < FGQA_KEYS_PER_WAVE; ++k) {
                    const int tl = wave * FGQA_KEYS_PER_WAVE + k;
                    if (tl < tile_len) {
                        const unsigned char* kb = k_cache
                            + (size_t)(tile_start + tl) * k_bytes_per_pos + k_head_off;
                        float cnorm;
                        __builtin_memcpy(&cnorm, kb, 4);
                        float partial[3] = { 0.0f, 0.0f, 0.0f };
                        for (int half = 0; half < 2; ++half) {
                            unsigned short pk = *(const unsigned short*)(kb + 4 + half * 64 + lane * 2);
                            float c0 = cnorm * TURBO_C4[(pk)       & 0xF];
                            float c1 = cnorm * TURBO_C4[((pk) >> 4)  & 0xF];
                            float c2 = cnorm * TURBO_C4[((pk) >> 8)  & 0xF];
                            float c3 = cnorm * TURBO_C4[((pk) >> 12) & 0xF];
#pragma unroll
                            for (int h = 0; h < 3; ++h) {
                                partial[h] += mq[h][half * 4 + 0] * c0
                                            + mq[h][half * 4 + 1] * c1
                                            + mq[h][half * 4 + 2] * c2
                                            + mq[h][half * 4 + 3] * c3;
                            }
                        }
                        for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
                            for (int h = 0; h < 3; ++h)
                                partial[h] += __shfl_xor(partial[h], off);
                        }
                        if (lane == 0) {
#pragma unroll
                            for (int h = 0; h < 3; ++h)
                                sc[(3 * hg + h) * FGQA_TILE + tl] = partial[h] * scale_attn;
                        }
                    }
                }
                // Keep group 1's Q loads from hoisting above group 0 (Q8 pattern).
                asm volatile("" ::: "memory");
            }
        }
        __syncthreads();

        // ---- Softmax: one wave per head, reference lane-strided order ----
        if (wave < FGQA_GROUP) {
            float* s_h = sc + wave * FGQA_TILE;
            float local_max = -1e30f;
            for (int i = lane; i < tile_len; i += 32)
                local_max = fmaxf(local_max, s_h[i]);
            for (int off = 16; off > 0; off >>= 1)
                local_max = fmaxf(local_max, __shfl_xor(local_max, off));
            const float tile_max = local_max;
            float local_sum = 0.0f;
            for (int i = lane; i < tile_len; i += 32) {
                const float e = expf(s_h[i] - tile_max);
                s_h[i] = e;
                local_sum += e;
            }
            for (int off = 16; off > 0; off >>= 1)
                local_sum += __shfl_xor(local_sum, off);
            if (lane == 0) {
                float* p = partials + ((size_t)(h0 + wave) * max_tiles + tile_id) * (2 + FGQA_HD);
                p[0] = tile_max;
                p[1] = local_sum;
            }
        }
        __syncthreads();

        // ---- V accumulation: waves 0-5 own heads 0-5. V streams in
        // 32-key slabs through LDS (one global read per K/V tile per
        // kv head); accumulators live across slabs, partials are written
        // once per tile (scalar-identical rotated space for lloyd V).
        const int v_rec_b = (v_mode == 8) ? 272 : 4 + FGQA_HD * v_mode / 8;
        const int v_rec_d = v_rec_b / 4;
        const size_t v_bytes_per_pos = (size_t)n_kv_heads * (size_t)v_rec_b;
        const size_t v_head_off = (size_t)kv_h * (size_t)v_rec_b;
        float out_vec[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        float ov[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        for (int slab = 0; slab < tile_len; slab += FGQA_V_TILE) {
            const int slab_len = min(FGQA_V_TILE, tile_len - slab);
            // Stage the slab cooperatively (tails clamp to the last row).
            {
                const size_t v_base =
                    (size_t)(tile_start + slab) * v_bytes_per_pos + v_head_off;
                const int v_slab_dwords = FGQA_V_TILE * v_rec_d;
                for (int i = tid; i < v_slab_dwords; i += FGQA_THREADS) {
                    const int k0 = i / v_rec_d;
                    const int w = i - k0 * v_rec_d;
                    const int k = k0 < slab_len ? k0 : slab_len - 1;
                    v_lds[i] = ((const unsigned int*)(v_cache + v_base
                        + (size_t)k * v_bytes_per_pos))[w];
                }
            }
            __syncthreads();
            if (wave < FGQA_GROUP) {
                const float* ws = sc + wave * FGQA_TILE + slab;
                if (v_mode == 8) {
                    for (int tl = 0; tl < slab_len; ++tl) {
                        const float w = ws[tl];
                        const unsigned char* vb =
                            (const unsigned char*)(v_lds + (size_t)tl * (size_t)v_rec_d);
                        for (int half = 0; half < 2; ++half) {
                            const int d_base = half * 128 + lane * 4;
                            const int v_bi = d_base / 32;
                            const int v_bj = d_base % 32;
                            const unsigned char* vbh = vb + (size_t)v_bi * 34u;
                            const float qq = (float)(*((const _Float16*)vbh));
                            out_vec[half * 4 + 0] += w * (qq * (float)((signed char)vbh[2 + v_bj + 0]));
                            out_vec[half * 4 + 1] += w * (qq * (float)((signed char)vbh[2 + v_bj + 1]));
                            out_vec[half * 4 + 2] += w * (qq * (float)((signed char)vbh[2 + v_bj + 2]));
                            out_vec[half * 4 + 3] += w * (qq * (float)((signed char)vbh[2 + v_bj + 3]));
                        }
                    }
                } else {
                    const int d0 = lane * 8;
                    for (int tl = 0; tl < slab_len; ++tl) {
                        const float w = ws[tl];
                        const unsigned char* vb =
                            (const unsigned char*)(v_lds + (size_t)tl * (size_t)v_rec_d);
                        float cnorm;
                        __builtin_memcpy(&cnorm, vb, 4);
                        const unsigned char* base = vb + 4 + lane * (unsigned)v_mode;
                        unsigned int packed = (unsigned int)base[0];
                        if (v_mode >= 2) packed |= ((unsigned int)base[1] << 8);
                        if (v_mode >= 3) packed |= ((unsigned int)base[2] << 16);
                        if (v_mode >= 4) packed |= ((unsigned int)base[3] << 24);
#pragma unroll
                        for (int i = 0; i < 8; i++) {
                            float c;
                            if (v_mode == 2)      c = TURBO_C2_256[(packed >> (i * 2)) & 3];
                            else if (v_mode == 3) c = TURBO_C3_256[(packed >> (i * 3)) & 7];
                            else                  c = TURBO_C4_256[(packed >> (i * 4)) & 15];
                            ov[i] += w * (cnorm * c);
                        }
                    }
                }
            }
            __syncthreads();
        }
        if (wave < FGQA_GROUP) {
            float* p = partials + ((size_t)(h0 + wave) * max_tiles + tile_id) * (2 + FGQA_HD);
            if (v_mode == 8) {
                for (int half = 0; half < 2; ++half) {
                    const int d_base = half * 128 + lane * 4;
                    p[2 + d_base + 0] = out_vec[half * 4 + 0];
                    p[2 + d_base + 1] = out_vec[half * 4 + 1];
                    p[2 + d_base + 2] = out_vec[half * 4 + 2];
                    p[2 + d_base + 3] = out_vec[half * 4 + 3];
                }
            } else {
                const int d0 = lane * 8;
#pragma unroll
                for (int i = 0; i < 8; i++)
                    p[2 + d0 + i] = ov[i];
            }
        }
    }
}
