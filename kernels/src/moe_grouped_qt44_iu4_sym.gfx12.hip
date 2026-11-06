// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Raon grouped symmetric MQ4G256V2 (qt44) experts x packed A4 activations, gfx1201
// (`v_wmma_i32_16x16x32_iu4`). Opt-in W4A4 prefill route for the Raon-1-mini-preview
// routed experts: gate/up `[2176, 2560]` and down `[2560, 1280]` are BOTH qt44
// (136 B per 256 K: `[f16 sc0, zp0, sc1, zp1][128 B nibbles]`, every header
// `zp == -8 * sc`, so `w = sc * (u - 8)`), so ONE kernel serves both projections;
// the K padding of the down projection (1088 -> 1280) and the AWQ divides live in the
// activation record producer, not here.
//
// X is the signed `block_i4_128` sidecar (`quantize_int4_mmq_ds128`), epoch-major
// `[K/128, x_src_rows]`, row `slot / x_row_div`. Per output and ascending K128 epoch h:
//   C_h = exact int32 sum_k128(A_signed * X_signed)            (A = nibble ^ 0x8 as int4)
//   sum = fma(RN(float(sc_h) * d_h), float(C_h), sum), sum starts +0
// which is the arithmetic of `qwen4_moe_*_iu4_sym_pm_gfx1201`. Output `Y[position][M]`
// is F32 (not bit-identical to the F16 route: A4 activations).
//
// K order inside an epoch is permuted identically on A and X (lane `hi` takes bytes
// `32*hi + 8*s` of the 64-byte epoch at step `s`, so each lane streams 32 contiguous
// bytes per operand): the int32 chain is order independent.
//
// Tiling: 4 waves per CTA; wave `w` owns `RF` 16-row fragments `(w * RF + rf)`; the CTA
// covers `64 * RF` rows. A CTA is a *leader* when its 16-slot tile starts an
// `NT`-aligned group inside its expert's tile run (run position `% NT == 0`, from
// `expert_offsets`); the leader applies each weight fragment to up to `NT` consecutive
// tiles of the same expert and every non-leader returns. Grid `(M / (64 * RF),
// m_total / 16)`, block 128. Padding slots (`sorted_slot_index < 0`) read a zero X and
// store +0; sentinel tiles (`expert_tile_ids < 0`) return before any store.

#include <hip/hip_runtime.h>
#include <stdint.h>

#if !defined(__gfx1201__) && defined(__HIP_DEVICE_COMPILE__)
#error "moe_grouped_qt44_iu4_sym.gfx12.hip is gfx1201-only"
#endif

struct mg_block_i4_128 {
    float d;
    int s;
    unsigned char qs[64];
};

typedef int mg_i32x2 __attribute__((ext_vector_type(2)));
typedef int mg_i32x8 __attribute__((ext_vector_type(8)));
typedef unsigned mg_u32x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float mg_f16_lo(unsigned int w) {
    return (float)__builtin_bit_cast(_Float16, (unsigned short)(w & 0xFFFFu));
}

struct mg_aop {
    mg_u32x2 w[4];
    float sc;
};
struct mg_xop {
    mg_u32x2 w[4];
    float d;
};

__device__ __forceinline__ void mg_load_a(mg_aop& o, const unsigned char* row, int h, int hi) {
    const unsigned char* g = row + (size_t)(h >> 1) * 136;
    o.sc = mg_f16_lo(*(const unsigned int*)(g + 4 * (h & 1)));
    const unsigned char* p = g + 8 + 64 * (h & 1) + 32 * hi;
#pragma unroll
    for (int s = 0; s < 4; ++s) o.w[s] = *(const mg_u32x2*)(p + 8 * s);
}

__device__ __forceinline__ void mg_load_x(mg_xop& o, const mg_block_i4_128* xrow, bool live, int h,
                                            size_t x_src_rows, int hi) {
    if (live) {
        const mg_block_i4_128* b = xrow + (size_t)h * x_src_rows;
        o.d = b->d;
        const unsigned char* p = b->qs + 32 * hi;
#pragma unroll
        for (int s = 0; s < 4; ++s) o.w[s] = *(const mg_u32x2*)(p + 8 * s);
    } else {
        o.d = 0.0f;
#pragma unroll
        for (int s = 0; s < 4; ++s) o.w[s] = (mg_u32x2){0u, 0u};
    }
}

template <int NT, int RF, bool FULL>
__device__ __forceinline__ void mg_epoch(float (&sum)[NT][RF][8], const mg_aop (&a)[RF],
                                           const mg_xop (&x)[NT], int ntv, int hi) {
    float scr[RF][8];
#pragma unroll
    for (int rf = 0; rf < RF; ++rf) {
#pragma unroll
        for (int j = 0; j < 8; ++j) scr[rf][j] = __shfl(a[rf].sc, 8 * hi + j, 32);
    }
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        if (!FULL && i >= ntv) break;
#pragma unroll
        for (int rf = 0; rf < RF; ++rf) {
            mg_i32x8 c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const mg_i32x2 av = {(int)(a[rf].w[s].x ^ 0x88888888u), (int)(a[rf].w[s].y ^ 0x88888888u)};
                const mg_i32x2 xv = {(int)x[i].w[s].x, (int)x[i].w[s].y};
                c = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, av, true, xv, c, false);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
                sum[i][rf][j] = __fmaf_rn(__fmul_rn(scr[rf][j], x[i].d), (float)c[j], sum[i][rf][j]);
        }
    }
}

template <int NT, int RF, bool FULL>
__device__ __forceinline__ void mg_body(const unsigned char* const (&wrow)[RF], const mg_block_i4_128* const (&xrow)[NT],
                                          const bool (&live)[NT], float* __restrict__ Y, int tile, int ntv,
                                          int rbase, int M, int K, int m_total, size_t x_src_rows, int lr, int hi) {
    float sum[NT][RF][8];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int rf = 0; rf < RF; ++rf)
#pragma unroll
            for (int j = 0; j < 8; ++j) sum[i][rf][j] = 0.0f;

    const int epochs = K >> 7;  // even (K % 256 == 0)
    mg_aop a0[RF], a1[RF];
    mg_xop x0[NT], x1[NT];
#define MG_LOAD(h, A, X)                                                              \
    do {                                                                                \
        _Pragma("unroll") for (int rf = 0; rf < RF; ++rf) mg_load_a(A[rf], wrow[rf], (h), hi); \
        _Pragma("unroll") for (int i = 0; i < NT; ++i) {                                \
            if (FULL || i < ntv) mg_load_x(X[i], xrow[i], live[i], (h), x_src_rows, hi); \
        }                                                                               \
    } while (0)
    MG_LOAD(0, a0, x0);
    for (int h = 0; h < epochs; h += 2) {
        MG_LOAD(h + 1, a1, x1);
        mg_epoch<NT, RF, FULL>(sum, a0, x0, ntv, hi);
        if (h + 2 < epochs) MG_LOAD(h + 2, a0, x0);
        mg_epoch<NT, RF, FULL>(sum, a1, x1, ntv, hi);
    }
#undef MG_LOAD
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        if (!FULL && i >= ntv) break;
        const int pos = (tile + i) * 16 + lr;
        if (pos < m_total) {
#pragma unroll
            for (int rf = 0; rf < RF; ++rf) {
                float* y = Y + (size_t)pos * M + rbase + rf * 16 + 8 * hi;
                *(float4*)(y) = make_float4(sum[i][rf][0], sum[i][rf][1], sum[i][rf][2], sum[i][rf][3]);
                *(float4*)(y + 4) = make_float4(sum[i][rf][4], sum[i][rf][5], sum[i][rf][6], sum[i][rf][7]);
            }
        }
    }
}

template <int NT, int RF>
__device__ __forceinline__ void mg_moe_iu4(
    const unsigned long long* __restrict__ expert_weight_ptrs, const int* __restrict__ expert_tile_ids,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_slot_index,
    const mg_block_i4_128* __restrict__ Xq, float* __restrict__ Y, int M, int K, int x_row_div, int m_total,
    int x_src_rows, int n_experts) {
    const int tile = blockIdx.y;
    if (tile * 16 >= m_total) return;
    const int e = expert_tile_ids[tile];
    if (e < 0 || e >= n_experts) return;
    const int first = expert_offsets[e] >> 4;
    if (((tile - first) % NT) != 0) return;
    const int tiles_total = m_total >> 4;
    int ntv = 1;
    while (ntv < NT && tile + ntv < tiles_total && expert_tile_ids[tile + ntv] == e) ++ntv;

    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31, lr = lane & 15, hi = lane >> 4;
    const int rbase = blockIdx.x * (64 * RF) + wave * (16 * RF);
    const size_t row_bytes = (size_t)(K >> 8) * 136;
    const unsigned char* W = (const unsigned char*)expert_weight_ptrs[e];
    const unsigned char* wrow[RF];
#pragma unroll
    for (int rf = 0; rf < RF; ++rf) wrow[rf] = W + (size_t)(rbase + rf * 16 + lr) * row_bytes;

    const mg_block_i4_128* xrow[NT];
    bool live[NT];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        int slot = -1;
        if (i < ntv) slot = sorted_slot_index[(tile + i) * 16 + lr];
        live[i] = slot >= 0;
        xrow[i] = Xq + (live[i] ? slot / x_row_div : 0);
    }
    if (ntv == NT) {
        mg_body<NT, RF, true>(wrow, xrow, live, Y, tile, ntv, rbase, M, K, m_total, (size_t)x_src_rows, lr, hi);
    } else {
        mg_body<NT, RF, false>(wrow, xrow, live, Y, tile, ntv, rbase, M, K, m_total, (size_t)x_src_rows, lr, hi);
    }
}

#define MG_IU4_ENTRY(NT, RF)                                                                              \
    extern "C" __global__ __launch_bounds__(128) void moe_grouped_qt44_iu4_sym_gfx1201_nt##NT##r##RF(               \
        const unsigned long long* __restrict__ expert_weight_ptrs, const int* __restrict__ expert_tile_ids, \
        const int* __restrict__ expert_offsets, const int* __restrict__ sorted_slot_index,                  \
        const mg_block_i4_128* __restrict__ Xq, float* __restrict__ Y, int M, int K, int x_row_div,       \
        int m_total, int x_src_rows, int n_experts) {                                                       \
        mg_moe_iu4<NT, RF>(expert_weight_ptrs, expert_tile_ids, expert_offsets, sorted_slot_index, Xq, Y, \
                             M, K, x_row_div, m_total, x_src_rows, n_experts);                              \
    }

MG_IU4_ENTRY(1, 1)
MG_IU4_ENTRY(2, 1)
MG_IU4_ENTRY(4, 1)
MG_IU4_ENTRY(1, 2)
MG_IU4_ENTRY(2, 2)
