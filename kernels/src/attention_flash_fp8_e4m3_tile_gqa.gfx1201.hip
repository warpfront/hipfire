// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// GQA-shared native-fp8 decode flash tile (gfx1201, head_dim 256, tile 128,
// full causal), compile-time GQA group GQA_GROUP (6: H24/KV4 dense 27B) or
// GQA_GROUP8 (8: H16/KV2 A3B). One 256-thread workgroup per (kv head, tile)
// serves all the q heads of its kv head, so each K/V tile is read from
// memory once instead of once per q head. At entry every thread issues its
// whole share of the tile's K rows, K/V scales and V bytes, so the tile is
// in flight before the first dot product.
//
// Output is byte-identical to `attention_flash_fp8_e4m3_tile` (same
// `[n_heads, max_tiles, 2 + head_dim]` f32 partials, consumed unchanged by
// the reduce). Every per-head, per-key operation reproduces the reference
// kernel's compiled arithmetic exactly (hipcc -O3 gfx1201):
//   * lane leaf, per 128-dim half: fma(q3,c3, fma(q2,c2, fma(q0,c0, q1*c1)));
//     partial = (0 + leaf_half0) + leaf_half1;
//   * xor-16 step (contracted by the reference): fma(p_i, ks, ks * p_{i+16})
//     — only lanes 0..15 feed lane 0's score, so the tree is those 16
//     values combined xor 8, 4, 2, 1 with plain adds; score = sum * scale;
//   * softmax: the reference's lane-strided max/exp/sum source, one wave per
//     head; the V weight is (e * v_scale), exactly the reference's fold;
//   * V: out_d = fma(w_t * vs_t, v_t,d, out_d) over keys in increasing order.
// Here a 16-lane row computes the 32 reference lanes' leaves for one key:
// row lane r holds reference lanes r (dims 4r, 128+4r) and r+16 (dims
// 64+4r, 192+4r); the xor-16 step is in-thread and the rest is DPP
// row_xmask within the row.
//
// Scores process group 6 in two halves (3+3) and group 8 in four quarters
// (2+2+2+2); P.V processes both groups in two halves of GROUP/2 heads.
// Per-head arithmetic, softmax wave mapping (wave w = head w) and key order
// are unchanged. Group 6 and group 8 are two instantiations of one template
// body (two entry symbols).
#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "attention_flash_fp8_e4m3_tile_gqa_gfx1201 requires --offload-arch=gfx1201"
#endif

#define GQA_TILE 128
#define GQA_HD 256
#define GQA_GROUP 6
#define GQA_GROUP8 8
#define GQA_WAVES 8
#define GQA_THREADS (GQA_WAVES * 32)
// Each wave scores 16 keys, two per pass (one per 16-lane row).
#define GQA_PASSES (GQA_TILE / GQA_WAVES / 2)
// V bytes staged per thread: 128 rows x 256 B / 256 threads = 16 x 8 B.
#define GQA_V_CHUNKS (GQA_TILE * GQA_HD / 8 / GQA_THREADS)
// Q floats staged per thread: G heads x 256 / 256 threads = G/2 x float2.
#define GQA_Q_CHUNKS(G) ((G) * GQA_HD / 2 / GQA_THREADS)

static_assert(GQA_GROUP % 2 == 0 && GQA_GROUP8 % 2 == 0 && GQA_GROUP8 <= GQA_WAVES,
              "a group is processed in two equal head halves, one softmax wave per head");
static_assert(GQA_Q_CHUNKS(GQA_GROUP) * 2 * GQA_THREADS == GQA_GROUP * GQA_HD &&
                  GQA_Q_CHUNKS(GQA_GROUP8) * 2 * GQA_THREADS == GQA_GROUP8 * GQA_HD,
              "Q staging must tile exactly");

typedef float gqa_f32x2 __attribute__((ext_vector_type(2)));

template <int Ctrl>
__device__ __forceinline__ float gqa_row_xmask_add(float v) {
    const int other = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), Ctrl, 0xf, 0xf, false);
    return v + __builtin_bit_cast(float, other);
}

// One reference-lane leaf over one 128-dim half: 4 codes of one dword.
__device__ __forceinline__ float gqa_leaf(const float4 qv, const gqa_f32x2 lo, const gqa_f32x2 hi) {
    return __builtin_fmaf(qv.w, hi.y, __builtin_fmaf(qv.z, hi.x, __builtin_fmaf(qv.x, lo.x, qv.y * lo.y)));
}

// HPG heads (GROUP/2) of one half; ws points at the half's first score row.
template <bool Hi, int HPG>
__device__ __forceinline__ void gqa_accum_v(const unsigned int* __restrict__ v_col,
                                            const float* __restrict__ ws,
                                            int tile_len,
                                            float (&acc)[HPG][2]) {
    int t = 0;
    for (; t + 3 < tile_len; t += 4) {
        float4 w[HPG];
#pragma unroll
        for (int j = 0; j < HPG; ++j) w[j] = *(const float4*)(ws + j * GQA_TILE + t);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const gqa_f32x2 c = __builtin_amdgcn_cvt_pk_f32_fp8(v_col[(t + k) * (GQA_HD / 4)], Hi);
#pragma unroll
            for (int j = 0; j < HPG; ++j) {
                const float wk = w[j][k];
                acc[j][0] = __builtin_fmaf(wk, c.x, acc[j][0]);
                acc[j][1] = __builtin_fmaf(wk, c.y, acc[j][1]);
            }
        }
    }
    for (; t < tile_len; ++t) {
        const gqa_f32x2 c = __builtin_amdgcn_cvt_pk_f32_fp8(v_col[t * (GQA_HD / 4)], Hi);
#pragma unroll
        for (int j = 0; j < HPG; ++j) {
            const float wk = ws[j * GQA_TILE + t];
            acc[j][0] = __builtin_fmaf(wk, c.x, acc[j][0]);
            acc[j][1] = __builtin_fmaf(wk, c.y, acc[j][1]);
        }
    }
}

// Scores of heads HPG*hg .. HPG*hg+HPG-1 for this wave's 16 keys (8 passes x
// 2 rows).
template <int HPG>
__device__ __forceinline__ void gqa_scores_group(
    int hg, const float* __restrict__ q_lds, const unsigned int (&kw)[GQA_PASSES][4],
    const float (&ks)[GQA_PASSES], float scale_attn, int tile_len, int wave, int row, int rl,
    float* __restrict__ sc) {
    float4 qr[HPG][4];
#pragma unroll
    for (int h = 0; h < HPG; ++h) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            qr[h][j] = *(const float4*)(q_lds + (HPG * hg + h) * GQA_HD + 64 * j + 4 * rl);
    }
#pragma unroll
    for (int p = 0; p < GQA_PASSES; ++p) {
        const int t = wave * (2 * GQA_PASSES) + 2 * p + row;
        gqa_f32x2 lo[4], hi[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            lo[j] = __builtin_amdgcn_cvt_pk_f32_fp8(kw[p][j], false);
            hi[j] = __builtin_amdgcn_cvt_pk_f32_fp8(kw[p][j], true);
        }
        float s[HPG];
#pragma unroll
        for (int h = 0; h < HPG; ++h) {
            float p_lo = 0.0f;
            p_lo += gqa_leaf(qr[h][0], lo[0], hi[0]);
            p_lo += gqa_leaf(qr[h][2], lo[2], hi[2]);
            float p_hi = 0.0f;
            p_hi += gqa_leaf(qr[h][1], lo[1], hi[1]);
            p_hi += gqa_leaf(qr[h][3], lo[3], hi[3]);
            float v = __builtin_fmaf(p_lo, ks[p], ks[p] * p_hi);
            v = gqa_row_xmask_add<0x168>(v);
            v = gqa_row_xmask_add<0x164>(v);
            v = gqa_row_xmask_add<0x162>(v);
            v = gqa_row_xmask_add<0x161>(v);
            s[h] = v * scale_attn;
        }
        if (rl == 0 && t < tile_len) {
#pragma unroll
            for (int h = 0; h < HPG; ++h) sc[(HPG * hg + h) * GQA_TILE + t] = s[h];
        }
    }
}

// One (kv head, tile). All loads are issued at entry (Q, V, K in consumption
// order) through buffer descriptors, so per-lane addressing is one 32-bit
// offset and the per-load strides fold into instruction offsets.
template <int GROUP>
__device__ __forceinline__ void gqa_tile(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    int n_kv_heads, int max_tiles, float scale_attn, int seq_len, int kv_h, int tile_id,
    unsigned int* __restrict__ v_lds, float* __restrict__ sc, float* __restrict__ vs_lds,
    float* __restrict__ q_lds) {
    constexpr int HPG = GROUP / 2;
    constexpr int Q_CHUNKS = GQA_Q_CHUNKS(GROUP);
    const int tile_start = tile_id * GQA_TILE;
    const int tile_len = min(GQA_TILE, seq_len - tile_start);

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = lane >> 4;
    const int rl = lane & 15;

    const unsigned row_stride = (unsigned)n_kv_heads * (GQA_HD + 2);
    const unsigned head_off = (unsigned)kv_h * GQA_HD;
    const unsigned scale_off = (unsigned)n_kv_heads * GQA_HD + (unsigned)kv_h * 2;
    const unsigned tile_bytes = (unsigned)tile_len * row_stride;
    const auto k_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)(k_cache + (size_t)tile_start * row_stride), 0, tile_bytes, 0x31004000u);
    const auto v_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)(v_cache + (size_t)tile_start * row_stride), 0, tile_bytes, 0x31004000u);

    const int h0 = kv_h * GROUP;
    float2 qv[Q_CHUNKS];
#pragma unroll
    for (int i = 0; i < Q_CHUNKS; ++i)
        qv[i] = ((const float2*)(q + (size_t)h0 * GQA_HD))[tid + GQA_THREADS * i];

    // V: 8-byte chunk tid + 256*i is row (tid >> 5) + 8*i, bytes 8*lane.
    uint2 vr[GQA_V_CHUNKS];
    const unsigned v_off = (unsigned)min(wave, tile_len - 1) * row_stride + head_off + 8u * lane;
#pragma unroll
    for (int i = 0; i < GQA_V_CHUNKS; ++i) {
        const unsigned voff = wave + 8 * i < tile_len
            ? v_off + (unsigned)(8 * i) * row_stride : v_off;
        const auto w = __builtin_amdgcn_raw_buffer_load_b64(v_rsrc, voff, 0, 0);
        vr[i] = make_uint2(w[0], w[1]);
    }
    float vs = 0.0f;
    if (tid < GQA_TILE) {
        const unsigned short b = __builtin_amdgcn_raw_buffer_load_b16(
            v_rsrc, (unsigned)min(tid, tile_len - 1) * row_stride + scale_off, 0, 0);
        vs = (float)__builtin_bit_cast(_Float16, b);
    }

    // K: pass p, row r of wave w is key 16w + 2p + r; lane r*16+rl reads the
    // dwords at 4rl + {0, 64, 128, 192} of its key's head segment.
    unsigned int kw[GQA_PASSES][4];
    float ks[GQA_PASSES];
#pragma unroll
    for (int p = 0; p < GQA_PASSES; ++p) {
        const unsigned t = (unsigned)min(wave * (2 * GQA_PASSES) + 2 * p + row, tile_len - 1);
        const unsigned koff = t * row_stride + head_off + 4u * rl;
#pragma unroll
        for (int j = 0; j < 4; ++j) kw[p][j] = __builtin_amdgcn_raw_buffer_load_b32(k_rsrc, koff + 64u * j, 0, 0);
        const unsigned short b = __builtin_amdgcn_raw_buffer_load_b16(k_rsrc, t * row_stride + scale_off, 0, 0);
        ks[p] = (float)__builtin_bit_cast(_Float16, b);
    }

#pragma unroll
    for (int i = 0; i < Q_CHUNKS; ++i) ((float2*)q_lds)[tid + GQA_THREADS * i] = qv[i];
#pragma unroll
    for (int i = 0; i < GQA_V_CHUNKS; ++i) *(uint2*)(v_lds + 2 * (tid + GQA_THREADS * i)) = vr[i];
    if (tid < GQA_TILE) vs_lds[tid] = vs;
    __syncthreads();

    // Scores: group 8 uses two-head batches to reduce live Q registers;
    // group 6 retains its existing two halves. Keep Q loads separated.
    if constexpr (GROUP == GQA_GROUP8) {
        gqa_scores_group<2>(0, q_lds, kw, ks, scale_attn, tile_len, wave, row, rl, sc);
        asm volatile("" ::: "memory");
        gqa_scores_group<2>(1, q_lds, kw, ks, scale_attn, tile_len, wave, row, rl, sc);
        asm volatile("" ::: "memory");
        gqa_scores_group<2>(2, q_lds, kw, ks, scale_attn, tile_len, wave, row, rl, sc);
        asm volatile("" ::: "memory");
        gqa_scores_group<2>(3, q_lds, kw, ks, scale_attn, tile_len, wave, row, rl, sc);
    } else {
        gqa_scores_group<HPG>(0, q_lds, kw, ks, scale_attn, tile_len, wave, row, rl, sc);
        asm volatile("" ::: "memory");
        gqa_scores_group<HPG>(1, q_lds, kw, ks, scale_attn, tile_len, wave, row, rl, sc);
    }
    __syncthreads();

    // ── Softmax: one wave per head, reference lane-strided order.
    if (wave < GROUP) {
        float* s_h = sc + wave * GQA_TILE;
        float local_max = -1e30f;
        for (int i = lane; i < tile_len; i += 32)
            local_max = fmaxf(local_max, s_h[i]);
        for (int off = 16; off > 0; off >>= 1)
            local_max = fmaxf(local_max, __shfl_xor(local_max, off));
        const float tile_max = local_max;
        float local_sum = 0.0f;
        for (int i = lane; i < tile_len; i += 32) {
            const float e = expf(s_h[i] - tile_max);
            s_h[i] = e * vs_lds[i];
            local_sum += e;
        }
        for (int off = 16; off > 0; off >>= 1)
            local_sum += __shfl_xor(local_sum, off);
        if (lane == 0) {
            float* p = partials + ((size_t)(h0 + wave) * max_tiles + tile_id) * (2 + GQA_HD);
            p[0] = tile_max;
            p[1] = local_sum;
        }
    }
    __syncthreads();

    // ── V: wave w owns dword column c = lane + 32*(w&1), code pair (w>>1)&1
    //    (dims 4c + 2*pair + {0,1}) and heads HPG*(w>>2) + {0..HPG-1}.
    const int col = lane + 32 * (wave & 1);
    const int g = wave >> 2;
    float acc[HPG][2];
#pragma unroll
    for (int j = 0; j < HPG; ++j) { acc[j][0] = 0.0f; acc[j][1] = 0.0f; }
    const float* ws = sc + HPG * g * GQA_TILE;
    if ((wave >> 1) & 1)
        gqa_accum_v<true, HPG>(v_lds + col, ws, tile_len, acc);
    else
        gqa_accum_v<false, HPG>(v_lds + col, ws, tile_len, acc);
    const int d = 4 * col + 2 * ((wave >> 1) & 1);
#pragma unroll
    for (int j = 0; j < HPG; ++j) {
        float* p = partials + ((size_t)(h0 + HPG * g + j) * max_tiles + tile_id) * (2 + GQA_HD);
        *(float2*)(p + 2 + d) = make_float2(acc[j][0], acc[j][1]);
    }
}

// Kernel body of one group instantiation: the grid-y tile loop over the
// shared LDS of one workgroup.
template <int GROUP>
__device__ __forceinline__ void gqa_decode(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ pos_buf, int n_kv_heads, int max_seq, float scale_attn) {
    const int seq_len = pos_buf[0] + 1;
    const int kv_h = blockIdx.x;
    if (kv_h >= n_kv_heads) return;
    const int max_tiles = (max_seq + GQA_TILE - 1) / GQA_TILE;
    const int n_tiles = min((seq_len + GQA_TILE - 1) / GQA_TILE, max_tiles);

    __shared__ __attribute__((aligned(16))) unsigned int v_lds[GQA_TILE * GQA_HD / 4];
    __shared__ __attribute__((aligned(16))) float sc[GROUP * GQA_TILE];
    __shared__ __attribute__((aligned(16))) float vs_lds[GQA_TILE];
    __shared__ __attribute__((aligned(16))) float q_lds[GROUP * GQA_HD];

    for (int tile_id = blockIdx.y; tile_id < n_tiles; tile_id += gridDim.y) {
        if (tile_id != (int)blockIdx.y) __syncthreads();
        gqa_tile<GROUP>(q, k_cache, v_cache, partials, n_kv_heads, max_tiles, scale_attn, seq_len,
                        kv_h, tile_id, v_lds, sc, vs_lds, q_lds);
    }
}

// Same 13-argument ABI as attention_flash_fp8_e4m3_tile. The host admits
// only head_dim 256, n_heads / n_kv_heads == 6, tile_size 128, window 0.
// Grid: [n_kv_heads, Y], Y <= max_tiles; workgroup y serves tiles y, y + Y,
// ... below ceil(seq_len / 128). Block: [256]. The host caps Y (512) so a
// graph-captured max_tiles grid does not dispatch thousands of idle
// 256-thread workgroups.
//
// The VMM row module (`vmm_rows_source`) rewrites exactly one of the two
// entries below into a row body; its define removes the other entry so that
// module exports a single kernel.
#ifndef VMM_ROWS_FP8_GQA8_TILE
extern "C" __launch_bounds__(GQA_THREADS, 1)
__global__ void attention_flash_fp8_e4m3_tile_gqa_gfx1201(
    const float* __restrict__ q,                // [n_heads, 256]
    const unsigned char* __restrict__ k_cache,  // token-local fp8 rows
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,               // [n_heads, max_tiles, 2 + 256]
    const int* __restrict__ pos_buf,            // pos_buf[0] = last position
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int window,
    int effective_seq_len
) {
    (void)n_heads; (void)head_dim; (void)tile_size; (void)window; (void)effective_seq_len;
    gqa_decode<GQA_GROUP>(q, k_cache, v_cache, partials, pos_buf, n_kv_heads, max_seq, scale_attn);
}
#endif

// Group-8 twin (n_heads / n_kv_heads == 8: H16/KV2 A3B). Same ABI, grid and
// partials layout; q rows are [n_heads = 8 * n_kv_heads, 256].
#ifndef VMM_ROWS_FP8_GQA_TILE
extern "C" __launch_bounds__(GQA_THREADS, 1)
__global__ void attention_flash_fp8_e4m3_tile_gqa8_gfx1201(
    const float* __restrict__ q,                // [n_heads, 256]
    const unsigned char* __restrict__ k_cache,  // token-local fp8 rows
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,               // [n_heads, max_tiles, 2 + 256]
    const int* __restrict__ pos_buf,            // pos_buf[0] = last position
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int window,
    int effective_seq_len
) {
    (void)n_heads; (void)head_dim; (void)tile_size; (void)window; (void)effective_seq_len;
    gqa_decode<GQA_GROUP8>(q, k_cache, v_cache, partials, pos_buf, n_kv_heads, max_seq, scale_attn);
}
#endif
