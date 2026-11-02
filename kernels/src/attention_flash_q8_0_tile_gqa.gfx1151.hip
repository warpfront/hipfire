// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// GQA-shared Q8_0 decode flash tile (gfx1151, head_dim 256, GQA group 6,
// tile 128, full causal). One 256-thread workgroup per (kv head, tile)
// serves all six q heads of its kv head, so each K/V tile is read from
// memory once instead of once per q head. At entry every thread issues its
// whole share of the tile's Q, V rows and K codes/scales, so the tile is in
// flight before the first dot product.
//
// Output is byte-identical to `attention_flash_q8_0_tile` (same
// `[n_heads, max_tiles, 2 + head_dim]` f32 partials, consumed unchanged by
// the reduce). Every per-head, per-key operation reproduces the reference
// kernel's compiled arithmetic exactly (hipcc -O3 gfx1151):
//   * dequant: a = (float)scale_f16 * (float)code (one rounding; the
//     reference's v_fma_mix with a -0 addend is the same product);
//   * lane leaf, per 128-dim half: fma(q3,a3, fma(q2,a2, fma(q0,a0, q1*a1)));
//     partial = (0 + leaf_half0) + leaf_half1;
//   * wave tree: plain adds over xor 16, 8, 4, 2, 1; score = sum * scale;
//   * softmax: the reference's lane-strided max/exp/sum source, one wave per
//     head;
//   * V: out_d = fma(w_t, vs_t * v_t,d, out_d) over keys in increasing order.
// Here a 16-lane row computes the 32 reference lanes' leaves for one key:
// row lane r holds reference lanes r (dims 4r, 128+4r) and r+16 (dims
// 64+4r, 192+4r); the xor-16 step is in-thread and the rest is DPP
// row_xmask within the row.
//
// Q8_0 row layout (per token): n_kv_heads x 8 blocks x 34 B, a block being
// an f16 scale then 32 int8 codes. A lane's four codes of one block start at
// 34*b + 2 + 4*(r%8): an unaligned dword for even b, read with one dword
// load as the reference does (RDNA3 unaligned access mode).
#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
#error "attention_flash_q8_0_tile_gqa_gfx1151 requires --offload-arch=gfx1151"
#endif

#define GQA_TILE 128
#define GQA_HD 256
#define GQA_GROUP 6
#define GQA_WAVES 8
#define GQA_THREADS (GQA_WAVES * 32)
#define GQA_BLK 34
// Bytes of one (key, kv head) segment: 8 blocks of 34 B.
#define GQA_SEG (GQA_HD / 32 * GQA_BLK)
// Each wave scores 16 keys, two per pass (one per 16-lane row).
#define GQA_PASSES (GQA_TILE / GQA_WAVES / 2)
// V bytes staged per thread: 128 segments x 272 B / 256 threads = 17 x 8 B.
#define GQA_V_CHUNKS (GQA_TILE * GQA_SEG / 8 / GQA_THREADS)
// Q floats staged per thread: 6 heads x 256 / 256 threads = 3 x float2.
#define GQA_Q_CHUNKS (GQA_GROUP * GQA_HD / 2 / GQA_THREADS)

template <int Ctrl>
__device__ __forceinline__ float gqa_row_xmask_add(float v) {
    const int other = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), Ctrl, 0xf, 0xf, false);
    return v + __builtin_bit_cast(float, other);
}

__device__ __forceinline__ float gqa_f16(unsigned short b) {
    return (float)__builtin_bit_cast(_Float16, b);
}

__device__ __forceinline__ float gqa_code(unsigned int w, int i) {
    return (float)(int)(signed char)(w >> (8 * i));
}

// One reference-lane leaf over one 128-dim half.
__device__ __forceinline__ float gqa_leaf(const float4 qv, const float (&a)[4]) {
    return __builtin_fmaf(qv.w, a[3], __builtin_fmaf(qv.z, a[2], __builtin_fmaf(qv.x, a[0], qv.y * a[1])));
}

// Scores of heads 3*hg .. 3*hg+2 for this wave's 16 keys (8 passes x 2 rows).
__device__ __forceinline__ void gqa_scores_group(
    int hg, const float* __restrict__ q_lds, const unsigned int (&kw)[GQA_PASSES][4],
    const unsigned short (&kh)[GQA_PASSES][4], float scale_attn, int tile_len, int wave, int row,
    int rl, float* __restrict__ sc) {
    float4 qr[3][4];
#pragma unroll
    for (int h = 0; h < 3; ++h) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            qr[h][j] = *(const float4*)(q_lds + (3 * hg + h) * GQA_HD + 64 * j + 4 * rl);
    }
#pragma unroll
    for (int p = 0; p < GQA_PASSES; ++p) {
        const int t = wave * (2 * GQA_PASSES) + 2 * p + row;
        float a[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float ks = gqa_f16(kh[p][j]);
#pragma unroll
            for (int i = 0; i < 4; ++i) a[j][i] = ks * gqa_code(kw[p][j], i);
        }
        float s[3];
#pragma unroll
        for (int h = 0; h < 3; ++h) {
            float p_lo = 0.0f;
            p_lo += gqa_leaf(qr[h][0], a[0]);
            p_lo += gqa_leaf(qr[h][2], a[2]);
            float p_hi = 0.0f;
            p_hi += gqa_leaf(qr[h][1], a[1]);
            p_hi += gqa_leaf(qr[h][3], a[3]);
            float v = p_lo + p_hi;
            v = gqa_row_xmask_add<0x168>(v);
            v = gqa_row_xmask_add<0x164>(v);
            v = gqa_row_xmask_add<0x162>(v);
            v = gqa_row_xmask_add<0x161>(v);
            s[h] = v * scale_attn;
        }
        if (rl == 0 && t < tile_len) {
#pragma unroll
            for (int h = 0; h < 3; ++h) sc[(3 * hg + h) * GQA_TILE + t] = s[h];
        }
    }
}

// Code pair of the V segment for one key: acc[j] += w_j * (vs * code).
__device__ __forceinline__ void gqa_accum_key(const unsigned char* __restrict__ seg, int soff, int coff,
                                              const float (&w)[3], float (&acc)[3][2]) {
    const float vs = gqa_f16(*(const unsigned short*)(seg + soff));
    const unsigned int c = *(const unsigned short*)(seg + coff);
    const float a0 = vs * gqa_code(c, 0);
    const float a1 = vs * gqa_code(c, 1);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        acc[j][0] = __builtin_fmaf(w[j], a0, acc[j][0]);
        acc[j][1] = __builtin_fmaf(w[j], a1, acc[j][1]);
    }
}

// One (kv head, tile). All loads are issued at entry (Q, V, K in consumption
// order) through buffer descriptors, so per-lane addressing is one 32-bit
// offset and the per-load strides fold into instruction offsets.
__device__ __forceinline__ void gqa_tile(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    int n_kv_heads, int max_tiles, float scale_attn, int seq_len, int kv_h, int tile_id,
    unsigned int* __restrict__ v_lds, float* __restrict__ sc, float* __restrict__ q_lds) {
    const int tile_start = tile_id * GQA_TILE;
    const int tile_len = min(GQA_TILE, seq_len - tile_start);

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = lane >> 4;
    const int rl = lane & 15;

    const unsigned row_stride = (unsigned)n_kv_heads * GQA_SEG;
    const unsigned head_off = (unsigned)kv_h * GQA_SEG;
    const unsigned tile_bytes = (unsigned)tile_len * row_stride;
    const auto k_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)(k_cache + (size_t)tile_start * row_stride), 0, tile_bytes, 0x31004000u);
    const auto v_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)(v_cache + (size_t)tile_start * row_stride), 0, tile_bytes, 0x31004000u);

    const int h0 = kv_h * GQA_GROUP;
    float2 qv[GQA_Q_CHUNKS];
#pragma unroll
    for (int i = 0; i < GQA_Q_CHUNKS; ++i)
        qv[i] = ((const float2*)(q + (size_t)h0 * GQA_HD))[tid + GQA_THREADS * i];

    // V: 8-byte chunk c = tid + 256*i is segment c / 34, bytes 8*(c % 34);
    // it lands at LDS byte 8c (segments packed at 272 B).
    uint2 vr[GQA_V_CHUNKS];
#pragma unroll
    for (int i = 0; i < GQA_V_CHUNKS; ++i) {
        const int c = tid + GQA_THREADS * i;
        const int seg = c / (GQA_SEG / 8);
        const int col = c - seg * (GQA_SEG / 8);
        const unsigned voff = (unsigned)min(seg, tile_len - 1) * row_stride + head_off + 8u * (unsigned)col;
        const auto w = __builtin_amdgcn_raw_buffer_load_b64(v_rsrc, voff, 0, 0);
        vr[i] = make_uint2(w[0], w[1]);
    }

    // K: pass p, row r of wave w is key 16w + 2p + r; lane r*16+rl reads, for
    // j = 0..3, the codes of dims 4rl + 64j: block 2j + rl/8, bytes
    // 2 + 4*(rl%8) of it, and that block's f16 scale.
    unsigned int kw[GQA_PASSES][4];
    unsigned short kh[GQA_PASSES][4];
    const unsigned blk_off = head_off + (unsigned)(GQA_BLK * (rl >> 3));
#pragma unroll
    for (int p = 0; p < GQA_PASSES; ++p) {
        const unsigned t = (unsigned)min(wave * (2 * GQA_PASSES) + 2 * p + row, tile_len - 1);
        const unsigned soff = t * row_stride + blk_off;
        const unsigned koff = soff + 2u + 4u * (unsigned)(rl & 7);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            kw[p][j] = __builtin_amdgcn_raw_buffer_load_b32(k_rsrc, koff + 2u * GQA_BLK * j, 0, 0);
            kh[p][j] = __builtin_amdgcn_raw_buffer_load_b16(k_rsrc, soff + 2u * GQA_BLK * j, 0, 0);
        }
    }

#pragma unroll
    for (int i = 0; i < GQA_Q_CHUNKS; ++i) ((float2*)q_lds)[tid + GQA_THREADS * i] = qv[i];
#pragma unroll
    for (int i = 0; i < GQA_V_CHUNKS; ++i) *(uint2*)(v_lds + 2 * (tid + GQA_THREADS * i)) = vr[i];
    __syncthreads();

    // ── Scores: two keys per pass (one per 16-lane row); heads in two
    //    groups of three so only one group's Q is register-resident. The
    //    memory clobber keeps the scheduler from hoisting group 1's Q loads
    //    into group 0.
    gqa_scores_group(0, q_lds, kw, kh, scale_attn, tile_len, wave, row, rl, sc);
    asm volatile("" ::: "memory");
    gqa_scores_group(1, q_lds, kw, kh, scale_attn, tile_len, wave, row, rl, sc);
    __syncthreads();

    // ── Softmax: one wave per head, reference lane-strided order.
    if (wave < GQA_GROUP) {
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
            s_h[i] = e;
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

    // ── V: waves 4g .. 4g+3 own heads 3g .. 3g+2; thread column pc (0..127)
    //    owns dims 2pc, 2pc+1 (block pc/16).
    const int g = wave >> 2;
    const int pc = ((wave & 3) << 5) | lane;
    const int soff = GQA_BLK * (pc >> 4);
    const int coff = soff + 2 + 2 * (pc & 15);
    const unsigned char* vb = (const unsigned char*)v_lds;
    const float* ws = sc + 3 * g * GQA_TILE;
    float acc[3][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
    int t = 0;
    for (; t + 3 < tile_len; t += 4) {
        float4 w4[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) w4[j] = *(const float4*)(ws + j * GQA_TILE + t);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float w[3] = {w4[0][k], w4[1][k], w4[2][k]};
            gqa_accum_key(vb + (t + k) * GQA_SEG, soff, coff, w, acc);
        }
    }
    for (; t < tile_len; ++t) {
        const float w[3] = {ws[t], ws[GQA_TILE + t], ws[2 * GQA_TILE + t]};
        gqa_accum_key(vb + t * GQA_SEG, soff, coff, w, acc);
    }
    const int d = 2 * pc;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        float* p = partials + ((size_t)(h0 + 3 * g + j) * max_tiles + tile_id) * (2 + GQA_HD);
        *(float2*)(p + 2 + d) = make_float2(acc[j][0], acc[j][1]);
    }
}

// Same 13-argument ABI as attention_flash_q8_0_tile. The host admits only
// head_dim 256, n_heads / n_kv_heads == 6, tile_size 128, window 0.
// Grid: [n_kv_heads, Y], Y <= max_tiles; workgroup y serves tiles y, y + Y,
// ... below ceil(seq_len / 128). Block: [256]. The host caps Y (512) so a
// graph-captured max_tiles grid does not dispatch thousands of idle
// 256-thread workgroups.
extern "C" __launch_bounds__(GQA_THREADS, 1)
__global__ void attention_flash_q8_0_tile_gqa_gfx1151(
    const float* __restrict__ q,                // [n_heads, 256]
    const unsigned char* __restrict__ k_cache,  // Q8_0 rows
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
    const int seq_len = pos_buf[0] + 1;
    const int kv_h = blockIdx.x;
    if (kv_h >= n_kv_heads) return;
    const int max_tiles = (max_seq + GQA_TILE - 1) / GQA_TILE;
    const int n_tiles = min((seq_len + GQA_TILE - 1) / GQA_TILE, max_tiles);

    __shared__ __attribute__((aligned(16))) unsigned int v_lds[GQA_TILE * GQA_SEG / 4];
    __shared__ __attribute__((aligned(16))) float sc[GQA_GROUP * GQA_TILE];
    __shared__ __attribute__((aligned(16))) float q_lds[GQA_GROUP * GQA_HD];

    for (int tile_id = blockIdx.y; tile_id < n_tiles; tile_id += gridDim.y) {
        if (tile_id != (int)blockIdx.y) __syncthreads();
        gqa_tile(q, k_cache, v_cache, partials, n_kv_heads, max_tiles, scale_attn, seq_len, kv_h,
                 tile_id, v_lds, sc, q_lds);
    }
}
