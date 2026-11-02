// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// QSA attention over each row's selection (indexed_attention_select output),
// gathered through F16 WMMA on gfx11.  Replaces
// indexed_attention_attention_f32_batched_hg4 when HIPFIRE_QWEN4_QSA_WMMA_GATHER
// is set: same scale (1/16), softmax and sigmoid output gate, but Q, K, V and
// the probabilities are rounded to F16 and summed in WMMA / online-softmax
// order, so it is not bit-exact (opt-in until a KLD reference exists).
//
// indexed_attention_kv_f16vb first writes the cache rows [0, tokens) as F16:
//   k16  [token][kv_heads * 256]
//   vb16 [token / 4][kv_heads * 256][token % 4]
// vb16 is V block-transposed: the four tokens of one selected QSA block
// (compress 4) at 16 consecutive dims are one contiguous 128-byte run, which
// is the WMMA A fragment of V^T for those keys.
//
// indexed_attention_gathered_wmma_f16: one workgroup per (query row, KV head);
// the KV group's query heads (<= 16, zero-padded) are the WMMA N dimension,
// so each selected K and V row is read once per KV head.  The row's selection
// is walked in tiles of 128 entries.  Wave w scores entries 16w..16w+15
// (S^T = K Q^T; each lane gathers its entry's K row straight from k16), the
// tile's online-softmax max and sum go through LDS, and each wave then
// accumulates O^T += V^T P^T for its 32 dims over all 128 entries.
// Grid [rows, n_kv_heads], block 256, dynamic LDS ceil(selected, 128) * 4
// bytes.  head_dim 256, n_heads / n_kv_heads <= 16.

typedef _Float16 __attribute__((ext_vector_type(16))) qg_h16_t;
typedef _Float16 __attribute__((ext_vector_type(4))) qg_h4_t;
typedef float __attribute__((ext_vector_type(8))) qg_f8_t;

// gfx11 wave32 WMMA: a lane holds all 16 K elements of its row (lanes 16..31
// repeat lanes 0..15); accumulator element i of lane (l16, half) is matrix
// row 2i + half, column l16.
#define QG_CROW(i, half) (2 * (i) + (half))

static __device__ __forceinline__ float qg_other_half(float v) {
    return __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
}

// Cache rows [0, tokens) as F16 K and block-transposed V.  Grid [tokens,
// kv_heads], block 256.
extern "C" __global__ void indexed_attention_kv_f16vb(
    const float* __restrict__ k, const float* __restrict__ v, _Float16* __restrict__ k16,
    _Float16* __restrict__ vb16, int tokens, int kv_heads) {
    const int tok = blockIdx.x;
    const int c = threadIdx.x + blockIdx.y * 256;
    const int width = kv_heads * 256;
    if (tok >= tokens || c >= width) return;
    k16[(size_t)tok * width + c] = (_Float16)k[(size_t)tok * width + c];
    vb16[((size_t)(tok >> 2) * width + c) * 4 + (tok & 3)] = (_Float16)v[(size_t)tok * width + c];
}

// Q8 twin of indexed_attention_kv_f16vb: the cache rows are Q8_0 blocks (f16
// scale + 32 int8 codes, 34 bytes; kv_heads * 8 blocks per row, head-major),
// each element dequantized as code * scale before the F16 rounding.  Grid
// [tokens, kv_heads], block 256 (head_dim 256).
extern "C" __global__ void indexed_attention_kv_f16vb_q8(
    const unsigned char* __restrict__ k, const unsigned char* __restrict__ v,
    _Float16* __restrict__ k16, _Float16* __restrict__ vb16, int tokens, int kv_heads) {
    const int tok = blockIdx.x;
    const int head = blockIdx.y;
    const int d = threadIdx.x;
    const int width = kv_heads * 256;
    if (tok >= tokens || head >= kv_heads) return;
    const size_t block = (size_t)tok * kv_heads * 8 * 34 + (size_t)(head * 8 + (d >> 5)) * 34;
    const float ks = (float)*reinterpret_cast<const _Float16*>(k + block);
    const float vs = (float)*reinterpret_cast<const _Float16*>(v + block);
    const int c = head * 256 + d;
    k16[(size_t)tok * width + c] = (_Float16)((float)(signed char)k[block + 2 + (d & 31)] * ks);
    vb16[((size_t)(tok >> 2) * width + c) * 4 + (tok & 3)] =
        (_Float16)((float)(signed char)v[block + 2 + (d & 31)] * vs);
}

#define QG_TILE 128
extern "C" __global__ __launch_bounds__(256) void indexed_attention_gathered_wmma_f16(
    const float* __restrict__ q_with_gate, const _Float16* __restrict__ k16,
    const _Float16* __restrict__ vb16, const int* __restrict__ selected_indices,
    float* __restrict__ output, int rows, int position_start, int n_heads, int n_kv_heads,
    int budget_blocks, int compress, int capacity, int full_capacity) {
    constexpr int HD = 256;
    extern __shared__ int tokens[];                          // [ceil(selected_len, 128)]
    __shared__ __align__(16) _Float16 q_lds[16 * 16 * 16];   // [dim chunk][head][16 dims]
    __shared__ __align__(16) _Float16 p_lds[2][8][16 * 16];  // [parity][wave][head][16 entries]
    __shared__ float red_max[2][8][16];
    __shared__ float red_sum[2][8][16];
    const int row = blockIdx.x, kvh = blockIdx.y;
    if (row >= rows) return;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, l16 = lane & 15, half = lane >> 4;
    const int group = n_heads / n_kv_heads;
    // The selection length, as indexed_attention_attention_*_batched derives it.
    const int visible = position_start + row + 1;
    const int block_count = visible / compress;
    const int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    int selected_len = chosen * compress + visible - block_count * compress;
    if (selected_len > capacity) selected_len = capacity;
    const int n_tiles = (selected_len + QG_TILE - 1) / QG_TILE;
    const int* row_selected = selected_indices + (size_t)row * capacity;
    for (int i = tid; i < n_tiles * QG_TILE; i += 256) {
        const int token = i < selected_len ? row_selected[i] : -1;
        tokens[i] = (token >= 0 && token < full_capacity && token < visible) ? token : -1;
    }
    const int q_stride = n_heads * 2 * HD;
    const float* q_row = q_with_gate + (size_t)row * q_stride + (size_t)kvh * group * 2 * HD;
    for (int e = tid; e < 16 * HD / 4; e += 256) {
        const int h = e / (HD / 4), d = (e - h * (HD / 4)) * 4;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (h < group) v = *(const float4*)(q_row + h * 2 * HD + d);
        _Float16* dst = q_lds + ((d >> 4) * 16 + h) * 16 + (d & 15);
        dst[0] = (_Float16)v.x; dst[1] = (_Float16)v.y; dst[2] = (_Float16)v.z; dst[3] = (_Float16)v.w;
    }
    __syncthreads();
    const int width = n_kv_heads * HD;
    const _Float16* kbase = k16 + kvh * HD;
    float m_run = -INFINITY, l_run = 0.f;
    qg_f8_t o[2];
    o[0] = qg_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
    o[1] = o[0];
    for (int t = 0; t < n_tiles; ++t) {
        // LDS reduction slots alternate per tile: a wave that runs ahead into
        // the next tile never overwrites what a slower wave still reads.
        const int par = t & 1;
        const int base = t * QG_TILE;
        const int kb = base + 16 * wave;  // this wave's 16 score entries
        float sv[8];
        float lmax = -INFINITY;
        if (kb < selected_len) {
            const int my_token = tokens[kb + l16];
            const _Float16* kp = kbase + (size_t)(my_token < 0 ? 0 : my_token) * width;
            qg_f8_t s = qg_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int dc = 0; dc < HD / 16; ++dc) {
                const qg_h16_t a = *(const qg_h16_t*)(kp + dc * 16);
                const qg_h16_t b = *(const qg_h16_t*)(q_lds + (dc * 16 + l16) * 16);
                s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, s);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const bool ok = tokens[kb + QG_CROW(i, half)] >= 0;
                sv[i] = ok ? s[i] * 0.0625f : -INFINITY;
                lmax = fmaxf(lmax, sv[i]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) sv[i] = -INFINITY;
        }
        lmax = fmaxf(lmax, qg_other_half(lmax));
        if (half == 0) red_max[par][wave][l16] = lmax;
        __syncthreads();
        float m_tile = red_max[par][0][l16];
#pragma unroll
        for (int w = 1; w < 8; ++w) m_tile = fmaxf(m_tile, red_max[par][w][l16]);
        const float m_new = fmaxf(m_run, m_tile);
        float alpha = 1.f, lsum = 0.f;
        _Float16* pw = p_lds[par][wave] + l16 * 16;
        if (m_new != -INFINITY) {
            alpha = expf(m_run - m_new);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p = expf(sv[i] - m_new);
                lsum += p;
                pw[QG_CROW(i, half)] = (_Float16)p;
            }
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) pw[QG_CROW(i, half)] = (_Float16)0.0f;
        }
        lsum += qg_other_half(lsum);
        if (half == 0) red_sum[par][wave][l16] = lsum;
        __syncthreads();
        float tsum = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) tsum += red_sum[par][w][l16];
        l_run = l_run * alpha + tsum;
        m_run = m_new;
        o[0] *= alpha;
        o[1] *= alpha;
        // O^T[dims][heads] += V^T[dims][entries] P^T[entries][heads].
        for (int kc = 0; kc < 8; ++kc) {
            const int k0 = base + 16 * kc;
            if (k0 >= selected_len) break;
            const qg_h16_t b = *(const qg_h16_t*)(p_lds[par][kc] + l16 * 16);
            qg_h16_t a[2];
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const int r0 = k0 + 4 * g;
                const int t0 = tokens[r0], t1 = tokens[r0 + 1], t2 = tokens[r0 + 2], t3 = tokens[r0 + 3];
                // One whole selected block: a single 8-byte load per dim.
                const bool block = t0 >= 0 && (t0 & 3) == 0 && t1 == t0 + 1 && t2 == t0 + 2 && t3 == t0 + 3;
#pragma unroll
                for (int dt = 0; dt < 2; ++dt) {
                    const int c = kvh * HD + wave * 32 + dt * 16 + l16;
                    qg_h4_t v4;
                    if (block) {
                        v4 = *(const qg_h4_t*)(vb16 + ((size_t)(t0 >> 2) * width + c) * 4);
                    } else {
                        const int tt[4] = {t0, t1, t2, t3};
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            v4[e] = tt[e] >= 0
                                ? vb16[((size_t)(tt[e] >> 2) * width + c) * 4 + (tt[e] & 3)]
                                : (_Float16)0.0f;
                        }
                    }
#pragma unroll
                    for (int e = 0; e < 4; ++e) a[dt][4 * g + e] = v4[e];
                }
            }
            o[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[0], b, o[0]);
            o[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[1], b, o[1]);
        }
    }
    if (l16 >= group) return;
    const int head = kvh * group + l16;
    const float* gate = q_with_gate + (size_t)row * q_stride + (size_t)head * 2 * HD + HD;
    float* out = output + (size_t)row * n_heads * HD + (size_t)head * HD;
#pragma unroll
    for (int dt = 0; dt < 2; ++dt) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = wave * 32 + dt * 16 + QG_CROW(i, half);
            out[d] = l_run > 0.f ? o[dt][i] / l_run / (1.0f + expf(-gate[d])) : 0.f;
        }
    }
}
#undef QG_TILE
#undef QG_CROW
