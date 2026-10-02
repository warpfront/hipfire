// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx12 twin of indexed_attention_gathered_wmma.gfx1151.hip for the fp8 QSA
// state (the gfx1201 default): same algorithm, tiles and LDS layout; see that
// file for the design.  Differences:
//   * indexed_attention_kv_f16vb_fp8_gfx1201 decodes the fp8 rows (E4M3 code
//     times the row's per-head f16 scale, as qsa_kv_value in tensor_ops.hip)
//     into the same F16 k16 / vb16 layout;
//   * the gfx12 WMMA operand layout: a lane holds 8 K elements (lanes 0..15
//     elements 0..7, lanes 16..31 elements 8..15) and accumulator element i
//     of lane (l16, half) is matrix row i + 8 * half.
// Not bit-exact against indexed_attention_attention_fp8_batched_hg4 (opt-in
// until a KLD reference exists).

typedef _Float16 __attribute__((ext_vector_type(8))) qg_h8_t;
typedef _Float16 __attribute__((ext_vector_type(4))) qg_h4_t;
typedef float __attribute__((ext_vector_type(8))) qg_f8_t;
typedef float __attribute__((ext_vector_type(2))) qg_f2_t;

#define QG_CROW(i, half) ((i) + 8 * (half))

static __device__ __forceinline__ float qg_other_half(float v) {
    return __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
}

#if defined(__GFX12__)
// fp8 QSA rows [0, tokens) (kv_heads * 256 E4M3 codes, then kv_heads f16
// scales) as F16 K [token][kv_heads * 256] and block-transposed V
// [token / 4][kv_heads * 256][token % 4].  Grid [tokens, kv_heads], block 256.
extern "C" __global__ void indexed_attention_kv_f16vb_fp8_gfx1201(
    const unsigned char* __restrict__ k, const unsigned char* __restrict__ v,
    _Float16* __restrict__ k16, _Float16* __restrict__ vb16, int tokens, int kv_heads) {
    const int tok = blockIdx.x;
    const int kvh = blockIdx.y;
    const int width = kv_heads * 256;
    const int c = kvh * 256 + threadIdx.x;
    if (tok >= tokens) return;
    const size_t row_bytes = (size_t)kv_heads * (256 + 2);
    const unsigned char* kr = k + (size_t)tok * row_bytes;
    const unsigned char* vr = v + (size_t)tok * row_bytes;
    const float ks = (float)__builtin_bit_cast(
        _Float16, *reinterpret_cast<const unsigned short*>(kr + width + 2 * kvh));
    const float vs = (float)__builtin_bit_cast(
        _Float16, *reinterpret_cast<const unsigned short*>(vr + width + 2 * kvh));
    const qg_f2_t kp = __builtin_amdgcn_cvt_pk_f32_fp8((unsigned int)kr[c], false);
    const qg_f2_t vp = __builtin_amdgcn_cvt_pk_f32_fp8((unsigned int)vr[c], false);
    k16[(size_t)tok * width + c] = (_Float16)(kp.x * ks);
    vb16[((size_t)(tok >> 2) * width + c) * 4 + (tok & 3)] = (_Float16)(vp.x * vs);
}

#define QG_TILE 128
extern "C" __global__ __launch_bounds__(256) void indexed_attention_gathered_wmma_f16_gfx1201(
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
        const int par = t & 1;
        const int base = t * QG_TILE;
        const int kb = base + 16 * wave;
        float sv[8];
        float lmax = -INFINITY;
        if (kb < selected_len) {
            const int my_token = tokens[kb + l16];
            const _Float16* kp = kbase + (size_t)(my_token < 0 ? 0 : my_token) * width + 8 * half;
            qg_f8_t s = qg_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int dc = 0; dc < HD / 16; ++dc) {
                const qg_h8_t a = *(const qg_h8_t*)(kp + dc * 16);
                const qg_h8_t b = *(const qg_h8_t*)(q_lds + (dc * 16 + l16) * 16 + 8 * half);
                s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, s);
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
        for (int kc = 0; kc < 8; ++kc) {
            const int k0 = base + 16 * kc;
            if (k0 >= selected_len) break;
            const qg_h8_t b = *(const qg_h8_t*)(p_lds[par][kc] + l16 * 16 + 8 * half);
            qg_h8_t a[2];
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                const int r0 = k0 + 8 * half + 4 * g;
                const int t0 = tokens[r0], t1 = tokens[r0 + 1], t2 = tokens[r0 + 2], t3 = tokens[r0 + 3];
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
            o[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[0], b, o[0]);
            o[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[1], b, o[1]);
        }
    }
    if (l16 >= group) return;
    const int head = kvh * group + l16;
    const float* gate = q_with_gate + (size_t)row * q_stride + (size_t)head * 2 * HD + HD;
    float* out = output + (size_t)row * n_heads * HD + (size_t)head * HD;
    // Lane (head l16, half) holds dims 8 * half .. 8 * half + 7 of each tile.
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
#endif  // __GFX12__
#undef QG_CROW
