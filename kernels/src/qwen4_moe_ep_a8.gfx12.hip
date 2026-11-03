// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Qwen4 Flash-Next EP owner-routed symmetric-weight MoE prefill chain, int8-activation (A8) twin of the
// symmetric-IU4 PM chain (`qwen4_moe_iu4_sym_pm_gfx1201`), gfx1201 (`v_wmma_i32_16x16x16_iu8`).
//
// Same routing / grouping / output contracts as the A4 chain's two GEMMs:
//   * gate/up  (qt44 MQ4G256V2 136 B per 256, or the F1 drop-zero planar MQ4G256V2P): rows `r` (gate) and
//     `r + M/2` (up) of one expert; Y `[m_total, M/2]` BF16 bits of rt(silu(rt(g)) * rt(u)) with the shipped
//     hipcc SwiGLU expression `(v / (1 + expf(-v))) * w`; padding slots store +0.
//   * down     (qt53 MQ4G128V2 68 B per 128, or the planar MQ4G128V2P): Y `[m_total, M]` BF16 bits of rt(sum);
//     padding slots store +0.
//   Sentinel tiles (expert id outside [0, n_experts)) return before any load or store. Tile `t` of 16 grouped
//   slots is `blockIdx.y`; lane `lr` of a tile reads slot `sorted[16 t + lr]` (< 0 = padding: zero X, +0 out).
//   64-byte kernarg ABI of the A4 entries:
//     (expert_weight_ptrs, expert_tile_ids, sorted_slot_index, Xq, Y, M, K, x_row_div, m_total, x_src_rows,
//      n_experts).
//
// Activations: the signed `block_i8_128` sidecar (136 B per K128: `[f32 d][i32 s][128 i8]`, epoch-major
// `[K/128, x_src_rows]`, X row of slot `s` = `s / x_row_div`). Weights are symmetric (`zp == -8 * sc`): nibble `u`
// is the exact int8 `u - 8`. Per output and ascending K128 epoch h:
//   C_h = exact int32 sum_k128(A_signed * X_signed)            (8 K16 WMMA steps)
//   sum = fma(RN(float(sc_h) * d_h), float(C_h), sum), sum starts +0
// (|C_h| <= 128 * 8 * 127 < 2^24: float(C_h) is exact). The K order inside an epoch is permuted identically on A
// and X (lane `hi` takes elements `64*hi + 8*s .. +8` at step `s`); the int32 chain is order independent.
//
// Tile runs without an expert-offset table (the chain's launchers carry none): tile `t` is a LEADER iff
// `t % NT == 0` or `tile_ids[t - 1] != tile_ids[t]`; a leader covers itself and the following tiles of the same
// expert up to the next NT-aligned tile. Every non-sentinel tile is covered exactly once.
//
// Grid `(rows / 64, m_total / 16)`, block 128 (4 waves, 16 weight rows each; gate/up: 64 gate rows plus the
// matching 64 up rows per CTA; `rows` = M/2 for gate/up, M for down).

#include <hip/hip_runtime.h>
#include <stdint.h>

#if !defined(__gfx1201__) && defined(__HIP_DEVICE_COMPILE__)
#error "qwen4_moe_ep_a8.gfx12.hip is gfx1201-only"
#endif

struct ep8_block {
    float d;
    int s;
    signed char qs[128];
};

typedef int ep8_i32x2 __attribute__((ext_vector_type(2)));
typedef int ep8_i32x8 __attribute__((ext_vector_type(8)));
typedef unsigned ep8_u32x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float ep8_f16(unsigned short b) {
    return (float)__builtin_bit_cast(_Float16, b);
}
__device__ __forceinline__ unsigned short ep8_bf16_bits(float v) {
    const unsigned int u = __float_as_uint(v);
    if (!isfinite(v)) return (unsigned short)(u >> 16);
    return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float ep8_bf16_rt(float v) {
    if (!isfinite(v)) return v;
    const unsigned int u = __float_as_uint(v);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

struct ep8_aop {
    unsigned w[8];
    float sc;
};
struct ep8_xop {
    ep8_u32x2 w[8];
    float d;
};
// One weight row: nibble base `c`, header/scale base `s` (shipped: the group header base; planar: the f16 plane).
struct ep8_wrow {
    const unsigned char* c;
    const unsigned char* s;
};

template <bool DOWN, bool PLANAR>
__device__ __forceinline__ ep8_wrow ep8_wrow_of(const unsigned char* W, int r, int M, int K) {
    ep8_wrow o;
    if (PLANAR) {
        o.c = W + (size_t)r * (size_t)(K >> 1);
        o.s = W + (size_t)M * (size_t)(K >> 1) + 2 * (size_t)r * (size_t)(K >> 7);
    } else if (DOWN) {
        o.c = W + (size_t)r * (size_t)(K >> 7) * 68;
        o.s = o.c;
    } else {
        o.c = W + (size_t)r * (size_t)(K >> 8) * 136;
        o.s = o.c;
    }
    return o;
}

// K128 epoch `h`, K-half `hi` of one weight row: the f16 scale of this row/epoch and 32 contiguous nibble bytes.
template <bool DOWN, bool PLANAR>
__device__ __forceinline__ void ep8_load_a(ep8_aop& o, const ep8_wrow& r, int h, int hi) {
    const unsigned char* p;
    if (PLANAR) {
        o.sc = ep8_f16(*(const unsigned short*)(r.s + 2 * h));
        p = r.c + 64 * h + 32 * hi;
    } else if (DOWN) {
        const unsigned char* g = r.c + (size_t)h * 68;
        o.sc = ep8_f16(*(const unsigned short*)g);
        p = g + 4 + 32 * hi;
    } else {
        const unsigned char* g = r.c + (size_t)(h >> 1) * 136;
        o.sc = ep8_f16(*(const unsigned short*)(g + 4 * (h & 1)));
        p = g + 8 + 64 * (h & 1) + 32 * hi;
    }
    if (DOWN && !PLANAR) {
        // 68 B groups: 4-byte aligned only.
#pragma unroll
        for (int i = 0; i < 8; ++i) o.w[i] = *(const unsigned*)(p + 4 * i);
    } else {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const ep8_u32x2 v = *(const ep8_u32x2*)(p + 8 * s);
            o.w[2 * s] = v.x;
            o.w[2 * s + 1] = v.y;
        }
    }
}

__device__ __forceinline__ void ep8_load_x(ep8_xop& o, const ep8_block* xrow, bool live, int h, size_t x_src_rows,
                                           int hi) {
    if (live) {
        const ep8_block* b = xrow + (size_t)h * x_src_rows;
        o.d = b->d;
        const signed char* p = b->qs + 64 * hi;
#pragma unroll
        for (int s = 0; s < 8; ++s) o.w[s] = *(const ep8_u32x2*)(p + 8 * s);
    } else {
        o.d = 0.0f;
#pragma unroll
        for (int s = 0; s < 8; ++s) o.w[s] = (ep8_u32x2){0u, 0u};
    }
}

// Eight nibbles (one dword) -> eight signed int8 `u - 8` in the activation block's "nibble-pair order"
// (`block_i8_128_quant.hip`): bytes 0..3 = elements 0,2,4,6, bytes 4..7 = elements 1,3,5,7.
__device__ __forceinline__ ep8_i32x2 ep8_expand(unsigned w) {
    const unsigned lo = ((w & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u;
    const unsigned hi = (((w >> 4) & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u;
    ep8_i32x2 r;
    r.x = (int)lo;
    r.y = (int)hi;
    return r;
}

template <int NF, int NT, bool FULL>
__device__ __forceinline__ void ep8_epoch(float (&sum)[NF][NT][8], const ep8_aop (&a)[NF], const ep8_xop (&x)[NT],
                                          int ntv, int hi) {
    float scr[NF][8];
    ep8_i32x2 ax[NF][8];
#pragma unroll
    for (int f = 0; f < NF; ++f) {
#pragma unroll
        for (int j = 0; j < 8; ++j) scr[f][j] = __shfl(a[f].sc, 8 * hi + j, 32);
#pragma unroll
        for (int s = 0; s < 8; ++s) ax[f][s] = ep8_expand(a[f].w[s]);
    }
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        if (!FULL && i >= ntv) break;
#pragma unroll
        for (int f = 0; f < NF; ++f) {
            ep8_i32x8 c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const ep8_i32x2 xv = {(int)x[i].w[s].x, (int)x[i].w[s].y};
                c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, ax[f][s], true, xv, c, false);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
                sum[f][i][j] = __fmaf_rn(__fmul_rn(scr[f][j], x[i].d), (float)c[j], sum[f][i][j]);
        }
    }
}

template <bool DOWN, bool PLANAR, int NT, bool FULL>
__device__ __forceinline__ void ep8_body(const ep8_wrow (&wrow)[DOWN ? 1 : 2], const ep8_block* const (&xrow)[NT],
                                         const bool (&live)[NT], unsigned short* __restrict__ Y, int tile, int ntv,
                                         int rbase, int M, int K, size_t x_src_rows, int lr, int hi) {
    constexpr int NF = DOWN ? 1 : 2;
    float sum[NF][NT][8];
#pragma unroll
    for (int f = 0; f < NF; ++f)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) sum[f][i][j] = 0.0f;

    const int epochs = K >> 7;
    ep8_aop a0[NF], a1[NF];
    ep8_xop x0[NT], x1[NT];
#define EP8_LOAD(h, A, X)                                                                            \
    do {                                                                                             \
        _Pragma("unroll") for (int f = 0; f < NF; ++f) ep8_load_a<DOWN, PLANAR>(A[f], wrow[f], (h), hi); \
        _Pragma("unroll") for (int i = 0; i < NT; ++i) {                                             \
            if (FULL || i < ntv) ep8_load_x(X[i], xrow[i], live[i], (h), x_src_rows, hi);             \
        }                                                                                            \
    } while (0)
    EP8_LOAD(0, a0, x0);
    for (int h = 0; h < epochs; h += 2) {
        const bool two = (h + 1) < epochs;
        if (two) EP8_LOAD(h + 1, a1, x1);
        ep8_epoch<NF, NT, FULL>(sum, a0, x0, ntv, hi);
        if (two) {
            if (h + 2 < epochs) EP8_LOAD(h + 2, a0, x0);
            ep8_epoch<NF, NT, FULL>(sum, a1, x1, ntv, hi);
        }
    }
#undef EP8_LOAD

    const int cols = DOWN ? M : (M >> 1);
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        if (!FULL && i >= ntv) break;
        const int pos = (tile + i) * 16 + lr;
        unsigned short o[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float r;
            if (DOWN) {
                r = live[i] ? sum[0][i][j] : 0.0f;
            } else {
                const float v = ep8_bf16_rt(sum[0][i][j]);
                const float w = ep8_bf16_rt(sum[NF - 1][i][j]);
                r = live[i] ? ep8_bf16_rt((v / (1.0f + expf(-v))) * w) : 0.0f;
            }
            o[j] = ep8_bf16_bits(r);
        }
        uint4 pk;
        pk.x = (unsigned)o[0] | ((unsigned)o[1] << 16);
        pk.y = (unsigned)o[2] | ((unsigned)o[3] << 16);
        pk.z = (unsigned)o[4] | ((unsigned)o[5] << 16);
        pk.w = (unsigned)o[6] | ((unsigned)o[7] << 16);
        *(uint4*)(Y + (size_t)pos * cols + rbase + 8 * hi) = pk;
    }
}

template <bool DOWN, bool PLANAR, int NT>
__device__ __forceinline__ void ep8_moe(const unsigned long long* __restrict__ expert_weight_ptrs,
                                        const int* __restrict__ expert_tile_ids,
                                        const int* __restrict__ sorted_slot_index, const ep8_block* __restrict__ Xq,
                                        unsigned short* __restrict__ Y, int M, int K, int x_row_div, int m_total,
                                        int x_src_rows, int n_experts) {
    const int tile = blockIdx.y;
    if (tile * 16 >= m_total) return;
    const int e = expert_tile_ids[tile];
    if (e < 0 || e >= n_experts) return;
    if ((tile % NT) != 0 && expert_tile_ids[tile - 1] == e) return;  // not a leader
    const int tiles_total = m_total >> 4;
    int ntv = 1;
    while (ntv < NT && ((tile + ntv) % NT) != 0 && tile + ntv < tiles_total && expert_tile_ids[tile + ntv] == e) ++ntv;

    constexpr int NF = DOWN ? 1 : 2;
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31, lr = lane & 15, hi = lane >> 4;
    const int rbase = blockIdx.x * 64 + wave * 16;
    const unsigned char* W = (const unsigned char*)expert_weight_ptrs[e];
    ep8_wrow wrow[NF];
    wrow[0] = ep8_wrow_of<DOWN, PLANAR>(W, rbase + lr, M, K);
    if (!DOWN) wrow[NF - 1] = ep8_wrow_of<DOWN, PLANAR>(W, rbase + lr + (M >> 1), M, K);

    const ep8_block* xrow[NT];
    bool live[NT];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        int slot = -1;
        if (i < ntv) slot = sorted_slot_index[(tile + i) * 16 + lr];
        live[i] = slot >= 0;
        xrow[i] = Xq + (live[i] ? slot / x_row_div : 0);
    }
    if (ntv == NT) {
        ep8_body<DOWN, PLANAR, NT, true>(wrow, xrow, live, Y, tile, ntv, rbase, M, K, (size_t)x_src_rows, lr, hi);
    } else {
        ep8_body<DOWN, PLANAR, NT, false>(wrow, xrow, live, Y, tile, ntv, rbase, M, K, (size_t)x_src_rows, lr, hi);
    }
}

#define EP8_ENTRY(NAME, DOWN, PLANAR, NT)                                                                         \
    extern "C" __global__ __launch_bounds__(128) void NAME(                                                       \
        const unsigned long long* __restrict__ expert_weight_ptrs, const int* __restrict__ expert_tile_ids,      \
        const int* __restrict__ sorted_slot_index, const ep8_block* __restrict__ Xq,                              \
        unsigned short* __restrict__ Y, int M, int K, int x_row_div, int m_total, int x_src_rows,                  \
        int n_experts) {                                                                                          \
        ep8_moe<DOWN, PLANAR, NT>(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, Xq, Y, M, K, x_row_div, \
                                  m_total, x_src_rows, n_experts);                                                \
    }

EP8_ENTRY(qwen4_moe_ep_a8_gate_up_nt1, false, false, 1)
EP8_ENTRY(qwen4_moe_ep_a8_gate_up_nt2, false, false, 2)
EP8_ENTRY(qwen4_moe_ep_a8_gate_up_planar_nt1, false, true, 1)
EP8_ENTRY(qwen4_moe_ep_a8_gate_up_planar_nt2, false, true, 2)
EP8_ENTRY(qwen4_moe_ep_a8_down_nt1, true, false, 1)
EP8_ENTRY(qwen4_moe_ep_a8_down_nt2, true, false, 2)
EP8_ENTRY(qwen4_moe_ep_a8_down_planar_nt1, true, true, 1)
EP8_ENTRY(qwen4_moe_ep_a8_down_planar_nt2, true, true, 2)
