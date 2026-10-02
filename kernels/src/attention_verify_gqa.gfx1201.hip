// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// VerifyAttn Stage 0 (gfx1201): the batched-verify flash tile with one KV
// scan shared by all 6 q heads x 8 query rows of a kv head, split over a
// fixed, context-independent grid.
//
// `attention_verify_gqa_{q8,fp8}_gfx1201` writes the same
// `[row][head][max_tiles][2 + 256]` f32 partials as
// `attention_flash_{q8_0,fp8_e4m3}_tile_batched`, byte for byte, and
// `attention_verify_reduce_gfx1201` reduces them byte-identically to
// `attention_flash_asym_reduce_batched`. Shape is compile-time: head_dim 256,
// GQA group 6, tile 128, full causal, legacy single-slot KV (the host admits
// nothing else).
//
// Reference arithmetic reproduced (hipcc -O3 gfx1201 ISA of the tile_batched
// pair; reference lane l covers dims 8l..8l+7):
//   * kq = ks * dec(code), exact in f32 (Q8: f16 block scale of block l/4;
//     fp8: the head's f16 scale; the hardware fp8 decode equals the reference
//     software decode for every non-NaN code);
//   * leaf_l = fma(q7,kq7, ... fma(q1,kq1, fma(q0,kq0, +0))) in dim order;
//   * score = (xor-16, 8, 4, 2, 1 butterfly sum of the 32 leaves) * scale;
//   * softmax per tile: the reference's lane-strided max / expf / sum source;
//   * PV: out_d = fma(e_t, vq_{t,d}, out_d) over keys in increasing order,
//     keys at or past the row's causal bound skipped (never multiplied by 0).
// Phase A layout: a 16-lane row scores one key per pass; row lane r holds
// reference lanes r (dims 8r..8r+7) and r+16 (dims 128+8r..), so the xor-16
// step is in-thread and 8/4/2/1 are DPP row_xmask adds. Row 0 of a pass
// scores key 32m+r', row 1 key 32m+16+r'.
//
// Every wave owns 6 (row, head) pairs for the whole tile (scores, softmax and
// PV), so the tile loop has no workgroup barrier; the 8 waves of a workgroup
// read the same K/V bytes through L0.
#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "attention_verify_gqa_gfx1201 requires --offload-arch=gfx1201"
#endif

#define VA_TILE 128
#define VA_HD 256
#define VA_STRIDE (2 + VA_HD)
#define VA_GROUP 6
#define VA_WAVES 8
#define VA_THREADS (VA_WAVES * 32)
#define VA_ROWS 8
// (row, head) pairs per wave: 8 rows x 6 heads over 8 waves.
#define VA_PPW (VA_ROWS * VA_GROUP / VA_WAVES)
// Buffer descriptor dword3 for gfx12 raw buffer loads (as the decode tile).
#define VA_RSRC_FLAGS 0x31004000u

enum { VA_KV_Q8 = 0, VA_KV_FP8 = 1 };

template <int Ctrl>
__device__ __forceinline__ float va_row_xmask_add(float v) {
    const int other = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), Ctrl, 0xf, 0xf, false);
    return v + __builtin_bit_cast(float, other);
}

__device__ __forceinline__ float va_f16(unsigned short b) {
    return (float)__builtin_bit_cast(_Float16, b);
}

// Eight signed Q8 codes times their block scale (exact f32 products).
__device__ __forceinline__ void va_dequant_q8(unsigned int lo, unsigned int hi, float s, float (&out)[8]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = s * (float)(signed char)(lo >> (8 * i));
#pragma unroll
    for (int i = 0; i < 4; ++i) out[4 + i] = s * (float)(signed char)(hi >> (8 * i));
}

// Eight E4M3 codes times the head scale (exact f32 products).
__device__ __forceinline__ void va_dequant_fp8(unsigned int lo, unsigned int hi, float s, float (&out)[8]) {
    typedef float f32x2 __attribute__((ext_vector_type(2)));
    const f32x2 a = __builtin_amdgcn_cvt_pk_f32_fp8(lo, false);
    const f32x2 b = __builtin_amdgcn_cvt_pk_f32_fp8(lo, true);
    const f32x2 c = __builtin_amdgcn_cvt_pk_f32_fp8(hi, false);
    const f32x2 d = __builtin_amdgcn_cvt_pk_f32_fp8(hi, true);
    out[0] = a.x * s; out[1] = a.y * s; out[2] = b.x * s; out[3] = b.y * s;
    out[4] = c.x * s; out[5] = c.y * s; out[6] = d.x * s; out[7] = d.y * s;
}

// One reference-lane leaf: sequential fma over its 8 dims from +0.
__device__ __forceinline__ float va_leaf(const float (&q)[8], const float (&k)[8]) {
    float p = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) p = __builtin_fmaf(q[i], k[i], p);
    return p;
}

// K/V bytes of one 8-dim slice of one key: the codes (two dwords) and scale.
struct VaSlice {
    unsigned int lo, hi;
    unsigned short sc;
};

// Byte offsets (from the tile's first row) of the 8-dim slice starting at
// `dim` of kv head `kv_h`, key `t`.
template <int KV>
__device__ __forceinline__ VaSlice va_load_slice(__amdgpu_buffer_rsrc_t rsrc, unsigned row_off, int kv_h,
                                                 int n_kv_heads, int dim) {
    VaSlice s;
    if (KV == VA_KV_Q8) {
        const unsigned blk = row_off + (unsigned)(kv_h * (VA_HD / 32) + (dim >> 5)) * 34u;
        const auto w = __builtin_amdgcn_raw_buffer_load_b64(rsrc, blk + 2u + (unsigned)(dim & 31), 0, 0);
        s.lo = w[0];
        s.hi = w[1];
        s.sc = __builtin_amdgcn_raw_buffer_load_b16(rsrc, blk, 0, 0);
    } else {
        const auto w = __builtin_amdgcn_raw_buffer_load_b64(rsrc, row_off + (unsigned)(kv_h * VA_HD + dim), 0, 0);
        s.lo = w[0];
        s.hi = w[1];
        s.sc = __builtin_amdgcn_raw_buffer_load_b16(
            rsrc, row_off + (unsigned)(n_kv_heads * VA_HD + kv_h * 2), 0, 0);
    }
    return s;
}

template <int KV>
__device__ __forceinline__ void va_dequant(const VaSlice& s, float (&out)[8]) {
    if (KV == VA_KV_Q8)
        va_dequant_q8(s.lo, s.hi, va_f16(s.sc), out);
    else
        va_dequant_fp8(s.lo, s.hi, va_f16(s.sc), out);
}

// Every tile of one wave's workgroup range, for the wave's first NP pairs
// (the valid pairs of a wave are always a prefix j < NP, so the pair loops
// are straight-line and interleave across pairs).
template <int KV, int NP>
__device__ __forceinline__ void va_tiles(
    const unsigned char* __restrict__ k_cache, const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials, float (&sc)[VA_PPW][VA_TILE], const float (&qa)[VA_PPW][8],
    const float (&qb)[VA_PPW][8], const int (&pair_len)[VA_PPW], const size_t (&pair_base)[VA_PPW],
    int group_len, int n_tiles, int kv_h, int n_kv_heads, unsigned per_pos, float scale_attn) {
    const int lane = (int)threadIdx.x & 31;
    const int half = lane >> 4;
    const int rl = lane & 15;

    for (int tile = blockIdx.y; tile < n_tiles; tile += gridDim.y) {
        const int tile_start = tile * VA_TILE;
        const int tl_max = min(VA_TILE, group_len - tile_start);
        int tl[NP];
        bool full = true;
#pragma unroll
        for (int j = 0; j < NP; ++j) {
            tl[j] = min(VA_TILE, max(0, pair_len[j] - tile_start));
            full = full && (tl[j] == 0 || tl[j] == tl_max);
        }

        const size_t tile_base = (size_t)tile_start * per_pos;
        const auto k_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)(k_cache + tile_base), 0, (int)((unsigned)tl_max * per_pos), VA_RSRC_FLAGS);
        const auto v_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)(v_cache + tile_base), 0, (int)((unsigned)tl_max * per_pos), VA_RSRC_FLAGS);

        // ── Phase A: scores. Pass k: row `half` scores key 32(k>>4) + (k&15) + 16*half.
        {
            auto key_of = [&](int k) { return 32 * (k >> 4) + (k & 15) + 16 * half; };
            unsigned off = (unsigned)min(key_of(0), tl_max - 1) * per_pos;
            VaSlice ca = va_load_slice<KV>(k_rsrc, off, kv_h, n_kv_heads, 8 * rl);
            VaSlice cb = va_load_slice<KV>(k_rsrc, off, kv_h, n_kv_heads, 128 + 8 * rl);
            for (int k = 0; 32 * (k >> 4) + (k & 15) < tl_max; ++k) {
                const int t = key_of(k);
                const unsigned noff = (unsigned)min(key_of(k + 1), tl_max - 1) * per_pos;
                const VaSlice na = va_load_slice<KV>(k_rsrc, noff, kv_h, n_kv_heads, 8 * rl);
                const VaSlice nb = va_load_slice<KV>(k_rsrc, noff, kv_h, n_kv_heads, 128 + 8 * rl);
                float ka[8], kb[8];
                va_dequant<KV>(ca, ka);
                va_dequant<KV>(cb, kb);
                float v[NP];
#pragma unroll
                for (int j = 0; j < NP; ++j) v[j] = va_leaf(qa[j], ka) + va_leaf(qb[j], kb);
#pragma unroll
                for (int j = 0; j < NP; ++j) v[j] = va_row_xmask_add<0x168>(v[j]);
#pragma unroll
                for (int j = 0; j < NP; ++j) v[j] = va_row_xmask_add<0x164>(v[j]);
#pragma unroll
                for (int j = 0; j < NP; ++j) v[j] = va_row_xmask_add<0x162>(v[j]);
#pragma unroll
                for (int j = 0; j < NP; ++j) v[j] = va_row_xmask_add<0x161>(v[j]);
                if (rl == 0 && t < tl_max) {
#pragma unroll
                    for (int j = 0; j < NP; ++j) sc[j][t] = v[j] * scale_attn;
                }
                ca = na;
                cb = nb;
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();

        // ── Phases B/C: the reference's lane-strided softmax, one pair at a time.
#pragma unroll
        for (int j = 0; j < NP; ++j) {
            const int tile_len = tl[j];
            if (tile_len == 0) continue;
            float* scores = sc[j];
            float local_max = -1e30f;
            for (int i = lane; i < tile_len; i += 32)
                local_max = fmaxf(local_max, scores[i]);
            for (int off = 16; off > 0; off >>= 1)
                local_max = fmaxf(local_max, __shfl_xor(local_max, off));
            const float tile_max = local_max;
            float local_sum = 0.0f;
            for (int i = lane; i < tile_len; i += 32) {
                float e = expf(scores[i] - tile_max);
                scores[i] = e;
                local_sum += e;
            }
            for (int off = 16; off > 0; off >>= 1)
                local_sum += __shfl_xor(local_sum, off);
            if (lane == 0) {
                float* p = partials + (pair_base[j] + tile) * VA_STRIDE;
                p[0] = tile_max;
                p[1] = local_sum;
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();

        // ── Phase D: PV. Lane l owns dims 8l..8l+7 of every pair. The full
        //    path runs every pair (a pair with no key in this tile computes
        //    unused sums); the ragged path skips keys past each pair's bound.
        float acc[NP][8];
#pragma unroll
        for (int j = 0; j < NP; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[j][i] = 0.0f;
        {
            VaSlice cv = va_load_slice<KV>(v_rsrc, 0u, kv_h, n_kv_heads, 8 * lane);
            if (full) {
                // Four keys per step: one 16-byte LDS broadcast of each
                // pair's weights, the next step's four V slices in flight.
                const int t4 = tl_max & ~3;
                VaSlice c4[4];
#pragma unroll
                for (int u = 0; u < 4; ++u)
                    c4[u] = va_load_slice<KV>(v_rsrc, (unsigned)min(u, tl_max - 1) * per_pos, kv_h, n_kv_heads, 8 * lane);
                int t = 0;
                for (; t < t4; t += 4) {
                    VaSlice n4[4];
#pragma unroll
                    for (int u = 0; u < 4; ++u)
                        n4[u] = va_load_slice<KV>(v_rsrc, (unsigned)min(t + 4 + u, tl_max - 1) * per_pos, kv_h,
                                                  n_kv_heads, 8 * lane);
                    float4 w4[NP];
#pragma unroll
                    for (int j = 0; j < NP; ++j) w4[j] = *(const float4*)&sc[j][t];
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        float vq[8];
                        va_dequant<KV>(c4[u], vq);
#pragma unroll
                        for (int j = 0; j < NP; ++j) {
                            const float w = w4[j][u];
#pragma unroll
                            for (int i = 0; i < 8; ++i) acc[j][i] = __builtin_fmaf(w, vq[i], acc[j][i]);
                        }
                    }
#pragma unroll
                    for (int u = 0; u < 4; ++u) c4[u] = n4[u];
                }
                for (; t < tl_max; ++t) {
                    float vq[8];
                    va_dequant<KV>(c4[0], vq);
#pragma unroll
                    for (int j = 0; j < NP; ++j) {
                        const float w = sc[j][t];
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc[j][i] = __builtin_fmaf(w, vq[i], acc[j][i]);
                    }
                    c4[0] = c4[1];
                    c4[1] = c4[2];
                    c4[2] = c4[3];
                }
            } else {
                for (int t = 0; t < tl_max; ++t) {
                    const VaSlice nv = va_load_slice<KV>(
                        v_rsrc, (unsigned)min(t + 1, tl_max - 1) * per_pos, kv_h, n_kv_heads, 8 * lane);
                    float vq[8];
                    va_dequant<KV>(cv, vq);
#pragma unroll
                    for (int j = 0; j < NP; ++j) {
                        if (t >= tl[j]) continue;
                        const float w = sc[j][t];
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc[j][i] = __builtin_fmaf(w, vq[i], acc[j][i]);
                    }
                    cv = nv;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NP; ++j) {
            if (tl[j] == 0) continue;
            float* p = partials + (pair_base[j] + tile) * VA_STRIDE + 2 + 8 * lane;
#pragma unroll
            for (int i = 0; i < 8; i += 2) *(float2*)(p + i) = make_float2(acc[j][i], acc[j][i + 1]);
        }
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();
    }
}

template <int KV>
__device__ __forceinline__ void va_verify_gqa(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ positions, int n_rows, int n_kv_heads, int max_tiles,
    int batch_offset, float scale_attn) {
    __shared__ __attribute__((aligned(16))) float sc_lds[VA_WAVES][VA_PPW][VA_TILE];

    const int n_heads = n_kv_heads * VA_GROUP;
    const int n_rg = (n_rows + VA_ROWS - 1) / VA_ROWS;
    const int kv_h = (int)blockIdx.x / n_rg;
    const int rg = (int)blockIdx.x - kv_h * n_rg;
    if (kv_h >= n_kv_heads) return;
    const int row0 = rg * VA_ROWS;
    const int rows_here = min(VA_ROWS, n_rows - row0);

    const int wave = __builtin_amdgcn_readfirstlane((int)threadIdx.x >> 5);
    const int rl = (int)threadIdx.x & 15;

    const unsigned per_pos = (KV == VA_KV_Q8) ? (unsigned)n_kv_heads * (VA_HD / 32) * 34u
                                              : (unsigned)n_kv_heads * (VA_HD + 2);

    // Live key range of this row group, from the device positions.
    int group_len = 0;
    for (int r = 0; r < rows_here; ++r) group_len = max(group_len, positions[batch_offset + row0 + r] + 1);
    // The reference grid covers tiles < max_tiles only; never write past it.
    const int n_tiles = min((group_len + VA_TILE - 1) / VA_TILE, max_tiles);

    // This wave's pairs: p = wave + 8j -> (row p/6, head p%6); valid pairs
    // (row < rows_here) are the prefix j < np.
    int np = 0;
    int pair_len[VA_PPW];
    size_t pair_base[VA_PPW];
    float qa[VA_PPW][8], qb[VA_PPW][8];
#pragma unroll
    for (int j = 0; j < VA_PPW; ++j) {
        const int p = wave + VA_WAVES * j;
        const int r = p / VA_GROUP;
        const int head = kv_h * VA_GROUP + (p - r * VA_GROUP);
        pair_base[j] = ((size_t)(row0 + r) * n_heads + head) * max_tiles;
        pair_len[j] = (r < rows_here) ? positions[batch_offset + row0 + r] + 1 : 0;
        if (r < rows_here) {
            np = j + 1;
            const float* qr = q + ((size_t)(row0 + r) * n_heads + head) * VA_HD;
            const float4 a0 = *(const float4*)(qr + 8 * rl);
            const float4 a1 = *(const float4*)(qr + 8 * rl + 4);
            const float4 b0 = *(const float4*)(qr + 128 + 8 * rl);
            const float4 b1 = *(const float4*)(qr + 128 + 8 * rl + 4);
            qa[j][0] = a0.x; qa[j][1] = a0.y; qa[j][2] = a0.z; qa[j][3] = a0.w;
            qa[j][4] = a1.x; qa[j][5] = a1.y; qa[j][6] = a1.z; qa[j][7] = a1.w;
            qb[j][0] = b0.x; qb[j][1] = b0.y; qb[j][2] = b0.z; qb[j][3] = b0.w;
            qb[j][4] = b1.x; qb[j][5] = b1.y; qb[j][6] = b1.z; qb[j][7] = b1.w;
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) { qa[j][i] = 0.0f; qb[j][i] = 0.0f; }
        }
    }

#define VA_TILES(N)                                                                                     \
    case N:                                                                                             \
        va_tiles<KV, N>(k_cache, v_cache, partials, sc_lds[wave], qa, qb, pair_len, pair_base, group_len, \
                        n_tiles, kv_h, n_kv_heads, per_pos, scale_attn);                                \
        break;
    switch (np) {
        VA_TILES(1)
        VA_TILES(2)
        VA_TILES(3)
        VA_TILES(4)
        VA_TILES(5)
        VA_TILES(6)
        default:
            break;
    }
#undef VA_TILES
}

// Grid: [ceil(n_rows/8) * n_kv_heads, S], block 256. Workgroup x = kv_h *
// ceil(n_rows/8) + row group; workgroup y serves tiles y, y + S, ... below
// ceil((max row position + 1) / 128) of its row group. Partials rows are
// chunk-local; positions are read at batch_offset + row.
extern "C" __launch_bounds__(VA_THREADS, 1)
__global__ void attention_verify_gqa_q8_gfx1201(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ positions, int n_rows, int n_kv_heads, int max_tiles,
    int batch_offset, float scale_attn) {
    va_verify_gqa<VA_KV_Q8>(q, k_cache, v_cache, partials, positions, n_rows, n_kv_heads, max_tiles,
                            batch_offset, scale_attn);
}

extern "C" __launch_bounds__(VA_THREADS, 1)
__global__ void attention_verify_gqa_fp8_gfx1201(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ positions, int n_rows, int n_kv_heads, int max_tiles,
    int batch_offset, float scale_attn) {
    va_verify_gqa<VA_KV_FP8>(q, k_cache, v_cache, partials, positions, n_rows, n_kv_heads, max_tiles,
                             batch_offset, scale_attn);
}

// Byte-identical parallel twin of attention_flash_asym_reduce_batched at
// head_dim 256: one thread per dim, each running the reference's serial tile
// order (max pass, then expf-corrected fma accumulation), then g * (1/sum).
// Grid: [n_heads, n_rows], block 256.
extern "C" __launch_bounds__(VA_HD, 1)
__global__ void attention_verify_reduce_gfx1201(
    const float* __restrict__ partials, float* __restrict__ out,
    const int* __restrict__ positions, int n_heads, int max_tiles, int batch_offset) {
    const int h = blockIdx.x;
    const int local_bid = blockIdx.y;
    const int d = threadIdx.x;
    const int seq_len = positions[batch_offset + local_bid] + 1;
    const int n_tiles = (seq_len + VA_TILE - 1) / VA_TILE;
    const float* my_partials = partials + ((long long)local_bid * n_heads + h) * max_tiles * VA_STRIDE;

    float global_max = -1e30f;
    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = my_partials + tile * VA_STRIDE;
        if (p[1] > 0.0f)
            global_max = fmaxf(global_max, p[0]);
    }

    // Same serial per-dim order as the reference; the loads of 8 tiles are
    // issued ahead of their (in-order) accumulation.
    float global_sum = 0.0f;
    float g = 0.0f;
    int tile = 0;
    for (; tile + 8 <= n_tiles; tile += 8) {
        float pm[8], ps[8], pv[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const float* p = my_partials + (tile + u) * VA_STRIDE;
            pm[u] = p[0];
            ps[u] = p[1];
            pv[u] = p[2 + d];
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            if (ps[u] <= 0.0f) continue;
            const float corr = expf(pm[u] - global_max);
            g += pv[u] * corr;
            global_sum += ps[u] * corr;
        }
    }
    for (; tile < n_tiles; tile++) {
        const float* p = my_partials + tile * VA_STRIDE;
        const float tile_max = p[0];
        const float tile_sum = p[1];
        if (tile_sum <= 0.0f) continue;
        const float corr = expf(tile_max - global_max);
        g += p[2 + d] * corr;
        global_sum += tile_sum * corr;
    }

    const float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    out[(size_t)local_bid * n_heads * VA_HD + (size_t)h * VA_HD + d] = g * inv_sum;
}
