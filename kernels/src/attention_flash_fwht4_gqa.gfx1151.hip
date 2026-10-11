// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// GQA-shared fwht4 flash tile (gfx1151, head_dim 256, GQA group 6, tile 128,
// full causal): K at FWHT-rotated 4-bit (fwht4), V at lloyd4 (256-wide FWHT
// rotated centroids) or Q8_0.
//
// One 256-thread workgroup per (kv head, 128-key tile, group of R query
// rows) serves the R rows x 6 q heads of its kv head, so each K/V tile is
// read once per R rows instead of once per (row, q head) as in
// `attention_flash_fwht4_tile{,_batched}`. Decode is R = 1 with the
// positions pointer at pos_buf; the multi-row verify variants use R = 4
// (lloyd4 V) and R = 2 (Q8_0 V, whose 272-B V rows need the LDS).
//
// Partials are the reference layout `[row, n_heads, max_tiles, 2 + 256]`
// (row 0 only for decode): tile max, tile sum, then the unnormalized
// V accumulator — FWHT-rotated for lloyd4 V (the reduce below applies the
// inverse once), normal space for Q8_0 V. A row with no key in a tile writes
// nothing for it, like the reference (its reduce stops at its own tile count).
//
// Numerics: same quantities as the reference tile (rotated Q, centroid K,
// two-pass softmax per tile, rotated lloyd V), different precision and
// order. Q.K runs in v_dot2_f32_f16: rotated Q and the K centroids are
// rounded to f16 (relative 2^-11), products accumulate in f32, cnorm and
// scale multiply the f32 sum; the dot product is a 16-lane DPP tree.
// Lloyd V's cnorm is folded into the softmax weights. V sums run per wave
// over 16 keys, then across waves (decode) — not byte-identical to the
// reference; the oracle example compares both on the same cache bytes.
//
// Layout of the fwht4 K / lloyd4 V record (per key, per kv head, 132 B):
// f32 cnorm, then 128 B of nibbles, byte b holding dims 2b (low nibble) and
// 2b + 1 (high nibble) of the rotated vector. K is rotated per 128-dim half
// (fwht_shfl_forward), lloyd V over all 256 dims.
#include <hip/hip_runtime.h>
#include "turbo_common.h"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
#error "attention_flash_fwht4_gqa requires --offload-arch=gfx1151"
#endif

#define F4_TILE 128
#define F4_HD 256
#define F4_GROUP 6
#define F4_THREADS 256
#define F4_SEG 132
// Each wave scores 16 keys, two per pass (one per 16-lane row).
#define F4_PASSES (F4_TILE / (F4_THREADS / 32) / 2)

template <int Ctrl>
__device__ __forceinline__ float f4_row_xmask_add(float v) {
    const int other = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), Ctrl, 0xf, 0xf, false);
    return v + __builtin_bit_cast(float, other);
}

__device__ __forceinline__ int f4_row_len(const int* __restrict__ positions, int row, int n_rows) {
    return row < n_rows ? positions[row] + 1 : 0;
}

template <int R, int VM>
__device__ __forceinline__ void f4_gqa(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ positions, const float* __restrict__ signs1,
    const float* __restrict__ signs2, int n_heads, int n_kv_heads, int max_tiles,
    float scale_attn, int batch_offset, int n_rows) {
    constexpr int VSEG = VM == 4 ? F4_SEG : 272;
    constexpr int V_DW = F4_TILE * VSEG / 4;
    constexpr int V_IT = (V_DW + F4_THREADS - 1) / F4_THREADS;
    constexpr int NQ = R * F4_GROUP;

    __shared__ __attribute__((aligned(16))) float q_lds[NQ * F4_HD];
    __shared__ __attribute__((aligned(16))) float sc[NQ * F4_TILE];
    __shared__ __attribute__((aligned(16))) unsigned int v_lds[V_DW];
    __shared__ __attribute__((aligned(16))) unsigned int kh2[256];
    __shared__ __attribute__((aligned(16))) float2 vt2[VM == 4 ? 256 : 1];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = lane >> 4;
    const int rl = lane & 15;

    const int kv_h = blockIdx.x;
    if (kv_h >= n_kv_heads) return;
    const int r0 = blockIdx.z * R;
    if (r0 >= n_rows) return;
    const int* __restrict__ pos = positions + batch_offset;
    int max_len = 0;
#pragma unroll
    for (int r = 0; r < R; ++r) max_len = max(max_len, f4_row_len(pos, r0 + r, n_rows));
    const int n_tiles = min((max_len + F4_TILE - 1) / F4_TILE, max_tiles);
    if ((int)blockIdx.y >= n_tiles) return;

    // Byte -> (low, high) nibble centroid pairs: K as packed f16, V as f32.
    {
        typedef _Float16 h2x __attribute__((ext_vector_type(2)));
        kh2[tid] = __builtin_bit_cast(unsigned int, h2x{(_Float16)TURBO_C4[tid & 15], (_Float16)TURBO_C4[tid >> 4]});
    }
    if (VM == 4) vt2[tid] = make_float2(TURBO_C4_256[tid & 15], TURBO_C4_256[tid >> 4]);

    // Rotated Q of the group's NQ (row, head) vectors, natural dim order.
    const int h0 = kv_h * F4_GROUP;
    for (int j = wave; j < NQ; j += F4_THREADS / 32) {
        const int lr = min(r0 + j / F4_GROUP, n_rows - 1);
        const float* qh = q + ((size_t)lr * n_heads + h0 + j % F4_GROUP) * F4_HD;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            float4 v = *(const float4*)(qh + half * 128 + lane * 4);
            fwht_shfl_forward(v.x, v.y, v.z, v.w, signs1, signs2, lane);
            *(float4*)(q_lds + j * F4_HD + half * 128 + lane * 4) = v;
        }
    }

    const unsigned k_stride = (unsigned)n_kv_heads * F4_SEG;
    const unsigned v_stride = (unsigned)n_kv_heads * VSEG;
    for (int tile_id = blockIdx.y; tile_id < n_tiles; tile_id += gridDim.y) {
        __syncthreads();  // tables/Q on entry; sc/v_lds reuse afterwards
        const int tile_start = tile_id * F4_TILE;
        const int tile_len = min(F4_TILE, max_len - tile_start);
        const unsigned char* kb = k_cache + (size_t)tile_start * k_stride + (unsigned)kv_h * F4_SEG;
        const unsigned char* vb = v_cache + (size_t)tile_start * v_stride + (unsigned)kv_h * VSEG;

        // Issue V (into registers, staged to LDS below) and K before any math.
        unsigned int vr[V_IT];
#pragma unroll
        for (int i = 0; i < V_IT; ++i) {
            const int c = tid + F4_THREADS * i;
            if (c < V_DW) {
                const int seg = c / (VSEG / 4);
                const int col = c - seg * (VSEG / 4);
                vr[i] = *(const unsigned int*)(vb + (size_t)min(seg, tile_len - 1) * v_stride + 4 * col);
            }
        }
        // K: pass p, row r of wave w is key 16w + 2p + r; lane rl holds the
        // 8 code bytes of dims 16rl .. 16rl + 15.
        float kn[F4_PASSES];
        unsigned int kc[F4_PASSES][2];
#pragma unroll
        for (int p = 0; p < F4_PASSES; ++p) {
            const int t = min(wave * (2 * F4_PASSES) + 2 * p + row, tile_len - 1);
            const unsigned char* kp = kb + (size_t)t * k_stride;
            kn[p] = *(const float*)kp;
            kc[p][0] = *(const unsigned int*)(kp + 4 + 8 * rl);
            kc[p][1] = *(const unsigned int*)(kp + 8 + 8 * rl);
        }
#pragma unroll
        for (int i = 0; i < V_IT; ++i) {
            const int c = tid + F4_THREADS * i;
            if (c < V_DW) v_lds[c] = vr[i];
        }

        // ── Scores: dequantize this lane's 16 dims of its 8 keys once into
        //    packed f16 (centroid pair table), then three q vectors at a time
        //    with v_dot2_f32_f16.
        typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
#pragma unroll
        for (int p = 0; p < F4_PASSES; ++p) kn[p] *= scale_attn;
        unsigned int kf[F4_PASSES][8];
#pragma unroll
        for (int p = 0; p < F4_PASSES; ++p)
#pragma unroll
            for (int b = 0; b < 8; ++b) kf[p][b] = kh2[(kc[p][b >> 2] >> (8 * (b & 3))) & 255];
#pragma unroll 1
        for (int g = 0; g < NQ / 3; ++g) {
            h2_t qh[3][8];
#pragma unroll
            for (int h = 0; h < 3; ++h)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 x = *(const float4*)(q_lds + (3 * g + h) * F4_HD + 16 * rl + 4 * i);
                    qh[h][2 * i] = h2_t{(_Float16)x.x, (_Float16)x.y};
                    qh[h][2 * i + 1] = h2_t{(_Float16)x.z, (_Float16)x.w};
                }
#pragma unroll
            for (int p = 0; p < F4_PASSES; ++p) {
                const int t = wave * (2 * F4_PASSES) + 2 * p + row;
                float s[3];
#pragma unroll
                for (int h = 0; h < 3; ++h) {
                    float v = 0.0f;
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        v = __builtin_amdgcn_fdot2(qh[h][i], __builtin_bit_cast(h2_t, kf[p][i]), v, false);
                    v = f4_row_xmask_add<0x168>(v);
                    v = f4_row_xmask_add<0x164>(v);
                    v = f4_row_xmask_add<0x162>(v);
                    v = f4_row_xmask_add<0x161>(v);
                    s[h] = v * kn[p];
                }
                if (rl == 0 && t < tile_len) {
#pragma unroll
                    for (int h = 0; h < 3; ++h) sc[(3 * g + h) * F4_TILE + t] = s[h];
                }
            }
        }
        __syncthreads();

        // ── Softmax per (row, head): keys past the row's causal end get
        //    weight 0; a row with no key in this tile writes no partial. The
        //    stored lloyd4 weight carries the key's V cnorm (the sum does not).
        for (int j = wave; j < NQ; j += F4_THREADS / 32) {
            const int lr = r0 + j / F4_GROUP;
            const int len = min(max(f4_row_len(pos, lr, n_rows) - tile_start, 0), tile_len);
            float* s_h = sc + j * F4_TILE;
            float local_max = -1e30f;
            for (int i = lane; i < len; i += 32) local_max = fmaxf(local_max, s_h[i]);
            for (int off = 16; off > 0; off >>= 1) local_max = fmaxf(local_max, __shfl_xor(local_max, off));
            float local_sum = 0.0f;
            for (int i = lane; i < F4_TILE; i += 32) {
                const float e = i < len ? expf(s_h[i] - local_max) : 0.0f;
                s_h[i] = VM == 4 ? e * *(const float*)((const unsigned char*)v_lds + i * VSEG) : e;
                local_sum += e;
            }
            for (int off = 16; off > 0; off >>= 1) local_sum += __shfl_xor(local_sum, off);
            if (lane == 0 && len > 0) {
                float* p = partials + (((size_t)lr * n_heads + h0 + j % F4_GROUP) * max_tiles + tile_id) * (2 + F4_HD);
                p[0] = local_max;
                p[1] = local_sum;
            }
        }
        __syncthreads();

        // ── V. All 128 keys run: the softmax zeroed the weights past
        //    tile_len, and the staged V rows there repeat the last key, so
        //    they are finite.
        if (R == 1) {
            // Decode: wave w folds keys 16w .. 16w+15 for all six heads; lane
            // owns dims 8*lane .. +7 (lloyd4: code dword `lane`; Q8_0: 8
            // codes of block lane/4). Then the eight wave accumulators meet
            // in LDS (aliasing v_lds) two heads at a time.
            const unsigned char* vlb = (const unsigned char*)v_lds;
            float vacc[F4_GROUP][8];
#pragma unroll
            for (int h = 0; h < F4_GROUP; ++h)
#pragma unroll
                for (int i = 0; i < 8; ++i) vacc[h][i] = 0.0f;
#pragma unroll 1
            for (int kk = 0; kk < 16; kk += 4) {
                const int tb = wave * 16 + kk;
                float vv[4][8];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (VM == 4) {
                        const unsigned int cw = *(const unsigned int*)(vlb + (tb + k) * VSEG + 4 + 4 * lane);
#pragma unroll
                        for (int b = 0; b < 4; ++b) {
                            const float2 c = vt2[(cw >> (8 * b)) & 255];
                            vv[k][2 * b] = c.x;
                            vv[k][2 * b + 1] = c.y;
                        }
                    } else {
                        const unsigned char* blk = vlb + (tb + k) * VSEG + 34 * (lane >> 2);
                        const float vs = (float)__builtin_bit_cast(_Float16, *(const unsigned short*)blk);
                        const unsigned short* cp = (const unsigned short*)(blk + 2 + 8 * (lane & 3));
#pragma unroll
                        for (int b = 0; b < 4; ++b) {
                            const unsigned int cw = cp[b];
                            vv[k][2 * b] = vs * (float)(int)(signed char)cw;
                            vv[k][2 * b + 1] = vs * (float)(int)(signed char)(cw >> 8);
                        }
                    }
                }
#pragma unroll
                for (int h = 0; h < F4_GROUP; ++h) {
                    const float4 w4 = *(const float4*)(sc + h * F4_TILE + tb);
                    const float w[4] = {w4.x, w4.y, w4.z, w4.w};
#pragma unroll
                    for (int k = 0; k < 4; ++k)
#pragma unroll
                        for (int i = 0; i < 8; ++i) vacc[h][i] = __builtin_fmaf(w[k], vv[k][i], vacc[h][i]);
                }
            }
            float* red = (float*)v_lds;
#pragma unroll
            for (int rd = 0; rd < 3; ++rd) {
                __syncthreads();
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    float* dst = red + (wave * 2 + hh) * F4_HD + 8 * lane;
                    *(float4*)dst = make_float4(vacc[2 * rd + hh][0], vacc[2 * rd + hh][1], vacc[2 * rd + hh][2], vacc[2 * rd + hh][3]);
                    *(float4*)(dst + 4) = make_float4(vacc[2 * rd + hh][4], vacc[2 * rd + hh][5], vacc[2 * rd + hh][6], vacc[2 * rd + hh][7]);
                }
                __syncthreads();
                const int hh = tid >> 7;
                const int d = (tid & 127) * 2;
                float2 s = make_float2(0.0f, 0.0f);
#pragma unroll
                for (int w = 0; w < 8; ++w) {
                    const float2 x = *(const float2*)(red + (w * 2 + hh) * F4_HD + d);
                    s.x += x.x;
                    s.y += x.y;
                }
                float* p = partials + (((size_t)r0 * n_heads + h0 + 2 * rd + hh) * max_tiles + tile_id) * (2 + F4_HD);
                *(float2*)(p + 2 + d) = s;
            }
            continue;
        }
        // Verify (R > 1; R x 6 x 8 accumulators would not fit the decode
        // mapping): waves 4g .. 4g+3 own heads 3g .. 3g+2 of every row; thread
        // column pc (0..127) owns dims 2pc, 2pc + 1, eight keys per step so
        // the byte -> centroid LDS chains of different keys overlap.
        const int g = wave >> 2;
        const int pc = ((wave & 3) << 5) | lane;
        const unsigned char* vl = (const unsigned char*)v_lds;
        float acc[R][3][2];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int j = 0; j < 3; ++j) acc[r][j][0] = acc[r][j][1] = 0.0f;
        const float* ws = sc + 3 * g * F4_TILE;
        const int soff = 34 * (pc >> 4);
        const int coff = soff + 2 + 2 * (pc & 15);
#pragma unroll 1
        for (int t0 = 0; t0 < F4_TILE; t0 += 8) {
            float a0[8], a1[8];
            if (VM == 4) {
                unsigned int bt[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const unsigned char* seg = vl + (t0 + k) * VSEG;
                    bt[k] = seg[4 + pc];
                }
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const float2 c = vt2[bt[k]];
                    a0[k] = c.x;
                    a1[k] = c.y;
                }
            } else {
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const unsigned char* seg = vl + (t0 + k) * VSEG;
                    const float vs = (float)__builtin_bit_cast(_Float16, *(const unsigned short*)(seg + soff));
                    const unsigned int cw = *(const unsigned short*)(seg + coff);
                    a0[k] = vs * (float)(int)(signed char)cw;
                    a1[k] = vs * (float)(int)(signed char)(cw >> 8);
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    const float* wp = ws + (r * F4_GROUP + j) * F4_TILE + t0;
                    const float4 w0 = *(const float4*)wp;
                    const float4 w1 = *(const float4*)(wp + 4);
                    const float w[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        acc[r][j][0] = __builtin_fmaf(w[k], a0[k], acc[r][j][0]);
                        acc[r][j][1] = __builtin_fmaf(w[k], a1[k], acc[r][j][1]);
                    }
                }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int lr = r0 + r;
            if (f4_row_len(pos, lr, n_rows) <= tile_start) continue;
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                float* p = partials + (((size_t)lr * n_heads + h0 + 3 * g + j) * max_tiles + tile_id) * (2 + F4_HD);
                *(float2*)(p + 2 + 2 * pc) = make_float2(acc[r][j][0], acc[r][j][1]);
            }
        }
    }
}

// Shared 13-argument ABI. q points at the launch's first row; positions[
// batch_offset + row] is that row's last attended position (decode: pos_buf,
// batch_offset 0, n_rows 1). The host admits only head_dim 256,
// n_heads / n_kv_heads == 6, tile 128, no tree bias, no window.
// Grid: [n_kv_heads, Y, ceil(n_rows / R)], Y <= max_tiles; workgroup y serves
// tiles y, y + Y, ... below the row group's tile count. Block: [256].
#define F4_ENTRY(NAME, R, VM)                                                                   \
    extern "C" __launch_bounds__(F4_THREADS, 1) __global__ void NAME(                           \
        const float* __restrict__ q, const unsigned char* __restrict__ k_cache,                 \
        const unsigned char* __restrict__ v_cache, float* __restrict__ partials,                \
        const int* __restrict__ positions, const float* __restrict__ signs1,                    \
        const float* __restrict__ signs2, int n_heads, int n_kv_heads, int max_tiles,           \
        float scale_attn, int batch_offset, int n_rows) {                                       \
        f4_gqa<R, VM>(q, k_cache, v_cache, partials, positions, signs1, signs2, n_heads,        \
                      n_kv_heads, max_tiles, scale_attn, batch_offset, n_rows);                 \
    }

F4_ENTRY(attention_flash_fwht4_gqa_l4_gfx1151, 1, 4)
F4_ENTRY(attention_flash_fwht4_gqa_q8v_gfx1151, 1, 8)
F4_ENTRY(attention_flash_fwht4_gqa_rows4_l4_gfx1151, 4, 4)
F4_ENTRY(attention_flash_fwht4_gqa_rows2_q8v_gfx1151, 2, 8)

// Cross-tile reduce of the partials above. One 256-thread workgroup per
// (head, row): pass 1 finds the row's max tile max over all threads; in pass
// 2 wave w folds tiles w, w + 8, ... (lane owns dims 8*lane .. 8*lane + 7),
// the eight wave accumulators meet in LDS, and wave 0 normalizes (after the
// inverse FWHT-256 for lloyd4 V) and stores. Same math as
// attention_flash_{lloyd,q8_0}_reduce{,_batched} (single corr per tile,
// tile_sum <= 0 skipped), tile-parallel instead of one wave walking every
// tile in order, so the sums associate differently.
// Grid: [n_heads, n_rows]. Block: [256]. out points at the launch's first row.
template <bool ROT>
__device__ __forceinline__ void f4_reduce(
    const float* __restrict__ partials, float* __restrict__ out, const int* __restrict__ positions,
    int n_heads, int max_tiles, int batch_offset, const float* __restrict__ signs1,
    const float* __restrict__ signs2) {
    constexpr int STRIDE = 2 + F4_HD;
    constexpr int WAVES = F4_THREADS / 32;
    __shared__ float red_m[WAVES];
    __shared__ float red_s[WAVES];
    __shared__ __attribute__((aligned(16))) float red_g[WAVES * F4_HD];
    const int h = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int seq_len = positions[batch_offset + row] + 1;
    const int n_tiles = min((seq_len + F4_TILE - 1) / F4_TILE, max_tiles);
    const float* P = partials + ((size_t)row * n_heads + h) * max_tiles * STRIDE;

    float m = -1e30f;
    for (int t = tid; t < n_tiles; t += F4_THREADS)
        if (P[(size_t)t * STRIDE + 1] > 0.0f) m = fmaxf(m, P[(size_t)t * STRIDE]);
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor(m, off));
    if (lane == 0) red_m[wave] = m;
    __syncthreads();
    float gmax = red_m[0];
#pragma unroll
    for (int w = 1; w < WAVES; ++w) gmax = fmaxf(gmax, red_m[w]);

    float g[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float sum = 0.0f;
#pragma unroll 4
    for (int t = wave; t < n_tiles; t += WAVES) {
        const float* p = P + (size_t)t * STRIDE;
        const float tile_sum = p[1];
        if (tile_sum <= 0.0f) continue;
        const float corr = expf(p[0] - gmax);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 x = *(const float2*)(p + 2 + 8 * lane + 2 * i);
            g[2 * i] += x.x * corr;
            g[2 * i + 1] += x.y * corr;
        }
        sum += tile_sum * corr;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) red_g[wave * F4_HD + 8 * lane + i] = g[i];
    if (lane == 0) red_s[wave] = sum;
    __syncthreads();
    if (wave != 0) return;
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) g[i] = 0.0f;
#pragma unroll
    for (int w = 0; w < WAVES; ++w) {
        s += red_s[w];
#pragma unroll
        for (int i = 0; i < 8; ++i) g[i] += red_g[w * F4_HD + 8 * lane + i];
    }
    if (ROT)
        fwht_shfl_inverse_256(g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], signs1, signs2, lane);
    const float inv_sum = s > 0.0f ? 1.0f / s : 0.0f;
    float* o = out + ((size_t)row * n_heads + h) * F4_HD + 8 * lane;
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = g[i] * inv_sum;
}

#define F4_REDUCE_ENTRY(NAME, ROT)                                                              \
    extern "C" __launch_bounds__(F4_THREADS, 1) __global__ void NAME(                           \
        const float* __restrict__ partials, float* __restrict__ out,                            \
        const int* __restrict__ positions, int n_heads, int max_tiles, int batch_offset,        \
        const float* __restrict__ signs1, const float* __restrict__ signs2) {                   \
        f4_reduce<ROT>(partials, out, positions, n_heads, max_tiles, batch_offset, signs1,      \
                       signs2);                                                                 \
    }

F4_REDUCE_ENTRY(attention_flash_fwht4_gqa_reduce_l4_gfx1151, true)
F4_REDUCE_ENTRY(attention_flash_fwht4_gqa_reduce_q8v_gfx1151, false)
