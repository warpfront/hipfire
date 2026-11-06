// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Grouped symmetric MQ4G256V2 (qt44) experts x e4m3 activations (W4FP8), gfx1201
// (`v_wmma_f32_16x16x16_fp8_fp8`). The FP8 twin of `moe_grouped_qt44_a8_sym.gfx12.hip`: same
// grouped ABI, tile-run scheme, F32 output and fold arithmetic, with X the `block_fp8_128` sidecar of
// `quantize_fp8_mmq_ds128` (136 B per K128: `[f32 d][4 B aux, unused][128 e4m3]`, epoch-major
// `[K/128, x_src_rows]`, codes in the same "nibble-pair order" as `block_i8_128`). Weights are
// symmetric (`zp == -8 * sc`, `w = sc * (u - 8)`): a nibble `u` becomes the e4m3 code of the integer
// `u - 8` (exact, -8..7) with a v_perm table and is fed as A. Per output and ascending K128 epoch h:
//   C_h = F32 WMMA sum_k128(e4m3(u - 8) * e4m3(x / d_h))     (8 K16 WMMA steps, products exact in F32)
//   sum = fma(RN(float(sc_h) * d_h), C_h, sum), sum starts +0
// The activation block scale is d = amax / 448 (full e4m3 range); the 3-bit mantissa makes the
// activation noise about 2^-4 relative per element, below the A4 route's and above the A8 route's.
//
// K order inside an epoch is permuted identically on A and X: lane `hi` takes elements
// `64*hi + 8*s .. +8` at step `s`, i.e. nibble dword `s` of its own 32 contiguous weight bytes
// and bytes `64*hi + 8*s` of the epoch's X.
//
// Tiling and leader/tile-run scheme: see the IU4 twin (4 waves, `RF` 16-row fragments per wave,
// `NT`-aligned expert runs from `expert_offsets`).

#include <hip/hip_runtime.h>
#include <stdint.h>

#if !defined(__gfx1201__) && defined(__HIP_DEVICE_COMPILE__)
#error "moe_grouped_qt44_fp8_sym.gfx12.hip is gfx1201-only"
#endif

struct fp8_block_128 {
    float d;
    int aux;
    unsigned char qs[128];
};

typedef int fp8_i32x2 __attribute__((ext_vector_type(2)));
typedef float fp8_f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned fp8_u32x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float fp8_f16_lo(unsigned int w) {
    return (float)__builtin_bit_cast(_Float16, (unsigned short)(w & 0xFFFFu));
}

struct fp8_aop {
    unsigned w[8];
    float sc;
};
struct fp8_xop {
    fp8_u32x2 w[8];
    float d;
};

__device__ __forceinline__ void fp8_load_a(fp8_aop& o, const unsigned char* row, int h, int hi) {
    const unsigned char* g = row + (size_t)(h >> 1) * 136;
    o.sc = fp8_f16_lo(*(const unsigned int*)(g + 4 * (h & 1)));
    const unsigned char* p = g + 8 + 64 * (h & 1) + 32 * hi;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const fp8_u32x2 v = *(const fp8_u32x2*)(p + 8 * s);
        o.w[2 * s] = v.x;
        o.w[2 * s + 1] = v.y;
    }
}

__device__ __forceinline__ void fp8_load_x(fp8_xop& o, const fp8_block_128* xrow, bool live, int h,
                                           size_t x_src_rows, int hi) {
    if (live) {
        const fp8_block_128* b = xrow + (size_t)h * x_src_rows;
        o.d = b->d;
        const unsigned char* p = b->qs + 64 * hi;
#pragma unroll
        for (int s = 0; s < 8; ++s) o.w[s] = *(const fp8_u32x2*)(p + 8 * s);
    } else {
        o.d = 0.0f;
#pragma unroll
        for (int s = 0; s < 8; ++s) o.w[s] = (fp8_u32x2){0u, 0u};
    }
}

// Four nibble values (one per byte, 0..15) -> four e4m3 bytes of (q - 8). The 16 table bytes live as
// four immediate dwords: [-8..-5] [-4..-1] [0..3] [4..7] (all exact in e4m3).
__device__ __forceinline__ unsigned fp8_lut8(unsigned sel) {
    const unsigned C0 = 0xCACCCED0u;
    const unsigned C1 = 0xB8C0C4C8u;
    const unsigned C2 = 0x44403800u;
    const unsigned C3 = 0x4E4C4A48u;
    const unsigned idx = sel & 0x07070707u;
    const unsigned lo = __builtin_amdgcn_perm(C1, C0, idx);
    const unsigned hi = __builtin_amdgcn_perm(C3, C2, idx);
    const unsigned selp = 0x03020100u | ((sel & 0x08080808u) >> 1);
    return __builtin_amdgcn_perm(hi, lo, selp);
}

// Eight nibbles (one dword) -> eight e4m3 codes of (u - 8) in the activation block's "nibble-pair
// order": bytes 0..3 = elements 0,2,4,6, bytes 4..7 = elements 1,3,5,7, so no byte shuffle is
// needed between the weight dword and the X bytes.
__device__ __forceinline__ fp8_i32x2 fp8_expand(unsigned w) {
    fp8_i32x2 r;
    r.x = (int)fp8_lut8(w & 0x0F0F0F0Fu);          // e0 e2 e4 e6
    r.y = (int)fp8_lut8((w >> 4) & 0x0F0F0F0Fu);   // e1 e3 e5 e7
    return r;
}

template <int NT, int RF, bool FULL>
__device__ __forceinline__ void fp8_epoch(float (&sum)[NT][RF][8], const fp8_aop (&a)[RF], const fp8_xop (&x)[NT],
                                          int ntv, int hi) {
    float scr[RF][8];
    fp8_i32x2 ax[RF][8];
#pragma unroll
    for (int rf = 0; rf < RF; ++rf) {
#pragma unroll
        for (int j = 0; j < 8; ++j) scr[rf][j] = __shfl(a[rf].sc, 8 * hi + j, 32);
#pragma unroll
        for (int s = 0; s < 8; ++s) ax[rf][s] = fp8_expand(a[rf].w[s]);
    }
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        if (!FULL && i >= ntv) break;
#pragma unroll
        for (int rf = 0; rf < RF; ++rf) {
            fp8_f32x8 c = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const fp8_i32x2 xv = {(int)x[i].w[s].x, (int)x[i].w[s].y};
                c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(ax[rf][s], xv, c);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
                sum[i][rf][j] = __fmaf_rn(__fmul_rn(scr[rf][j], x[i].d), c[j], sum[i][rf][j]);
        }
    }
}

template <int NT, int RF, bool FULL>
__device__ __forceinline__ void fp8_body(const unsigned char* const (&wrow)[RF], const fp8_block_128* const (&xrow)[NT],
                                         const bool (&live)[NT], float* __restrict__ Y, int tile, int ntv, int rbase,
                                         int M, int K, int m_total, size_t x_src_rows, int lr, int hi) {
    float sum[NT][RF][8];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int rf = 0; rf < RF; ++rf)
#pragma unroll
            for (int j = 0; j < 8; ++j) sum[i][rf][j] = 0.0f;

    const int epochs = K >> 7;  // even (K % 256 == 0)
    fp8_aop a0[RF], a1[RF];
    fp8_xop x0[NT], x1[NT];
#define FP8_LOAD(h, A, X)                                                                       \
    do {                                                                                        \
        _Pragma("unroll") for (int rf = 0; rf < RF; ++rf) fp8_load_a(A[rf], wrow[rf], (h), hi); \
        _Pragma("unroll") for (int i = 0; i < NT; ++i) {                                        \
            if (FULL || i < ntv) fp8_load_x(X[i], xrow[i], live[i], (h), x_src_rows, hi);        \
        }                                                                                       \
    } while (0)
    FP8_LOAD(0, a0, x0);
    for (int h = 0; h < epochs; h += 2) {
        FP8_LOAD(h + 1, a1, x1);
        fp8_epoch<NT, RF, FULL>(sum, a0, x0, ntv, hi);
        if (h + 2 < epochs) FP8_LOAD(h + 2, a0, x0);
        fp8_epoch<NT, RF, FULL>(sum, a1, x1, ntv, hi);
    }
#undef FP8_LOAD
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
__device__ __forceinline__ void fp8_moe(
    const unsigned long long* __restrict__ expert_weight_ptrs, const int* __restrict__ expert_tile_ids,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_slot_index,
    const fp8_block_128* __restrict__ Xq, float* __restrict__ Y, int M, int K, int x_row_div, int m_total,
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

    const fp8_block_128* xrow[NT];
    bool live[NT];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        int slot = -1;
        if (i < ntv) slot = sorted_slot_index[(tile + i) * 16 + lr];
        live[i] = slot >= 0;
        xrow[i] = Xq + (live[i] ? slot / x_row_div : 0);
    }
    if (ntv == NT) {
        fp8_body<NT, RF, true>(wrow, xrow, live, Y, tile, ntv, rbase, M, K, m_total, (size_t)x_src_rows, lr, hi);
    } else {
        fp8_body<NT, RF, false>(wrow, xrow, live, Y, tile, ntv, rbase, M, K, m_total, (size_t)x_src_rows, lr, hi);
    }
}

#define FP8_ENTRY(NT, RF)                                                                                   \
    extern "C" __global__ __launch_bounds__(128) void moe_grouped_qt44_fp8_sym_gfx1201_nt##NT##r##RF(       \
        const unsigned long long* __restrict__ expert_weight_ptrs, const int* __restrict__ expert_tile_ids, \
        const int* __restrict__ expert_offsets, const int* __restrict__ sorted_slot_index,                  \
        const fp8_block_128* __restrict__ Xq, float* __restrict__ Y, int M, int K, int x_row_div,           \
        int m_total, int x_src_rows, int n_experts) {                                                       \
        fp8_moe<NT, RF>(expert_weight_ptrs, expert_tile_ids, expert_offsets, sorted_slot_index, Xq, Y, M, K, \
                        x_row_div, m_total, x_src_rows, n_experts);                                         \
    }

FP8_ENTRY(1, 1)
FP8_ENTRY(2, 1)
FP8_ENTRY(1, 2)
FP8_ENTRY(2, 2)
