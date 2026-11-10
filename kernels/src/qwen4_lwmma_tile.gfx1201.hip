// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — lane-WMMA prefill tile GEMM: a staged-tile twin of lwmma_gemm_bf16_t{1,2}_m*_n1
// (qwen4_lane_wmma.gfx1201.hip) for the sub-512-row prefill (BF16 weights, f32 activations).
//
// Contract: every output word is BITWISE what lwmma_gemm_body writes for the same (split_k):
// per output, for each split-K chunk kz, four quarter chains (WMMA accumulation from +0.0 over the
// chunk's quarter w, 32-wide K blocks ascending, two K16 substeps each, one WMMA per activation
// term hi then lo), folded (c0 + c1) + (c2 + c3); then split_k == 1: V = p(0), split_k > 1:
// v = 0.0f, v += p(kz) ascending; Y = hc_act ? lwmma_hc_act(V) : V.  The fragment layout and K
// order are the incumbent's, so the result does not depend on which tile owns an output.
//
// Performance: one block owns a BM x BN output tile and walks K once, ascending, in 32-wide steps:
// the X tile [BM x 32] f32 is read from global ONCE per block and split ONCE into BF16 term planes
// in LDS, the W tile [BN x 32] BF16 is staged in LDS, both double-buffered with a one-step
// register prefetch; each wave loads its A/B fragments from LDS and runs its sub-tile's WMMAs.
// A wave owns its sub-tile for the whole K range, so the four quarter chains of an output run
// sequentially in one wave with a progressive fold (c0 kept; s01 = c0 + c1; c2 kept; then
// p = s01 + (c2 + c3), exactly the incumbent's grouping), and the split-K chunks fold in
// registers; no partial scratch and no counters.
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// The epilogues are the reference kernels' expressions: no multiply-add contraction.
#pragma clang fp contract(off)

typedef short __attribute__((ext_vector_type(8))) lwmma_bf16x8_t;
typedef float __attribute__((ext_vector_type(8))) lwmma_f8_t;

// tensor_ops.hip's bf16_to_f32(bf16_round(v)) == hip_bfloat16's RNE.
static __device__ __forceinline__ float lwmma_bf16r(float v) {
    return static_cast<float>(static_cast<hip_bfloat16>(v));
}

// Finite-input BF16 RNE bits: u += 0x7fff + ((u >> 16) & 1); bits = u >> 16.
static __device__ __forceinline__ unsigned int lwmma_rne_bits(float x) {
    unsigned int u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u);
    return u >> 16;
}

static __device__ __forceinline__ float lwmma_bits_to_f32(unsigned int bits) {
    return __uint_as_float(bits << 16);
}

// 8 f32 activations -> NT BF16 fragments (hi, lo, lo2).
template <int NT>
static __device__ __forceinline__ void lwmma_split8(const float4 xa, const float4 xb, lwmma_bf16x8_t* out) {
    const float x[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    unsigned int hi[8], lo[8], lo2[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
        hi[e] = lwmma_rne_bits(x[e]);
        if constexpr (NT >= 2) {
            const float r = x[e] - lwmma_bits_to_f32(hi[e]);
            lo[e] = lwmma_rne_bits(r);
            if constexpr (NT >= 3) {
                const float r2 = (x[e] - lwmma_bits_to_f32(hi[e])) - lwmma_bits_to_f32(lo[e]);
                lo2[e] = lwmma_rne_bits(r2);
            }
        }
    }
    uint4 p;
    p.x = hi[0] | (hi[1] << 16); p.y = hi[2] | (hi[3] << 16);
    p.z = hi[4] | (hi[5] << 16); p.w = hi[6] | (hi[7] << 16);
    out[0] = __builtin_bit_cast(lwmma_bf16x8_t, p);
    if constexpr (NT >= 2) {
        p.x = lo[0] | (lo[1] << 16); p.y = lo[2] | (lo[3] << 16);
        p.z = lo[4] | (lo[5] << 16); p.w = lo[6] | (lo[7] << 16);
        out[1] = __builtin_bit_cast(lwmma_bf16x8_t, p);
    }
    if constexpr (NT >= 3) {
        p.x = lo2[0] | (lo2[1] << 16); p.y = lo2[2] | (lo2[3] << 16);
        p.z = lo2[4] | (lo2[5] << 16); p.w = lo2[6] | (lo2[7] << 16);
        out[2] = __builtin_bit_cast(lwmma_bf16x8_t, p);
    }
}

// hc_act from gemv_bf16_xf32_k4.
static __device__ __forceinline__ float lwmma_hc_act(float value, float scale) {
    value = static_cast<float>(static_cast<hip_bfloat16>(value));
    value *= scale;
    value = static_cast<float>(static_cast<hip_bfloat16>(value));
    value = value / (1.0f + expf(-value));
    value = static_cast<float>(static_cast<hip_bfloat16>(value));
    return value;
}

// ---------------------------------------------------------------------------
// Y[rows, M] = epi(X . W^T).  Grid [ceil(M / BN), ceil(rows / BM)], block 32 * WM * WN.
// K % (128 * split_k) == 0.  Wave (wm, wn) owns sub-tile rows [wm * MT * 16, +MT * 16), columns
// [wn * NTN * 16, +NTN * 16) of the block tile.
template <int NT, int BM, int BN, int WM, int WN>
static __device__ __forceinline__ void lwmma_tile_body(
    const unsigned short* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int rows, int split_k, float scale, int hc_act) {
    constexpr int NTHR = 32 * WM * WN;
    constexpr int MT = BM / WM / 16;
    constexpr int NTN = BN / WN / 16;
    static_assert(MT * WM * 16 == BM && NTN * WN * 16 == BN, "tile / wave split");
    constexpr int LD = 40;                 // LDS row stride in BF16 elements (32 + 8 pad: 80 B)
    constexpr int XPL = BM * LD;           // one activation term plane
    constexpr int BUF = NT * XPL + BN * LD;  // X planes then the W tile
    constexpr int XCH = BM * 4;            // 8-float X chunks per K step
    constexpr int WCH = BN * 4;            // 8-element W chunks per K step
    constexpr int XPT = (XCH + NTHR - 1) / NTHR;
    constexpr int WPT = (WCH + NTHR - 1) / NTHR;
    __shared__ __attribute__((aligned(16))) unsigned short lds[2 * BUF];

    const int tid = threadIdx.x;
    const int wave = tid >> 5, lane = tid & 31, l16 = lane & 15, kh = lane >> 4;
    const int wm = wave / WN, wn = wave % WN;
    const int n0 = blockIdx.x * BN;
    const int t0 = blockIdx.y * BM;
    const int wmo = wm * MT * 16, wno = wn * NTN * 16;

    const float* xg[XPT];
#pragma unroll
    for (int u = 0; u < XPT; u++) {
        const int idx = tid + u * NTHR;
        const int row = min(t0 + (idx >> 2), rows - 1);
        xg[u] = X + (size_t)row * K + 8 * (idx & 3);
    }
    const unsigned short* wg[WPT];
#pragma unroll
    for (int u = 0; u < WPT; u++) {
        const int idx = tid + u * NTHR;
        const int row = min(n0 + (idx >> 2), M - 1);
        wg[u] = W + (size_t)row * K + 8 * (idx & 3);
    }

    float4 xv[XPT][2];
    uint4 wv[WPT];
    auto gload = [&](int s) {
        const int k0 = s * 32;
#pragma unroll
        for (int u = 0; u < XPT; u++) {
            if (XCH % NTHR == 0 || tid + u * NTHR < XCH) {
                const float4* p = (const float4*)(xg[u] + k0);
                xv[u][0] = p[0];
                xv[u][1] = p[1];
            }
        }
#pragma unroll
        for (int u = 0; u < WPT; u++) {
            if (WCH % NTHR == 0 || tid + u * NTHR < WCH) wv[u] = *(const uint4*)(wg[u] + k0);
        }
    };
    auto sstore = [&](int buf) {
        unsigned short* base = lds + buf * BUF;
#pragma unroll
        for (int u = 0; u < XPT; u++) {
            const int idx = tid + u * NTHR;
            if (XCH % NTHR == 0 || idx < XCH) {
                lwmma_bf16x8_t xa[NT];
                lwmma_split8<NT>(xv[u][0], xv[u][1], xa);
                const int off = (idx >> 2) * LD + 8 * (idx & 3);
#pragma unroll
                for (int t = 0; t < NT; t++) *(lwmma_bf16x8_t*)(base + t * XPL + off) = xa[t];
            }
        }
#pragma unroll
        for (int u = 0; u < WPT; u++) {
            const int idx = tid + u * NTHR;
            if (WCH % NTHR == 0 || idx < WCH)
                *(uint4*)(base + NT * XPL + (idx >> 2) * LD + 8 * (idx & 3)) = wv[u];
        }
    };

    lwmma_f8_t acc[MT][NTN], s01[MT][NTN], c2[MT][NTN], vv[MT][NTN];
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nt = 0; nt < NTN; nt++) {
            acc[mt][nt] = lwmma_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
            s01[mt][nt] = acc[mt][nt];
            c2[mt][nt] = acc[mt][nt];
            vv[mt][nt] = acc[mt][nt];
        }

    const int S = K / 32;                   // K steps
    const int nq = (K / split_k / 4) / 32;  // K steps per quarter chain

    gload(0);
    sstore(0);
    __syncthreads();

    // One K step: the redundant tail prefetch (clamped to the last step) keeps the loop branch-free.
    auto step = [&](int s) {
        const int cur = s & 1;
        gload(min(s + 1, S - 1));
        const unsigned short* base = lds + cur * BUF;
#pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            lwmma_bf16x8_t bf[NTN];
#pragma unroll
            for (int nt = 0; nt < NTN; nt++)
                bf[nt] = *(const lwmma_bf16x8_t*)(base + NT * XPL + (wno + nt * 16 + l16) * LD + sub * 16 + 8 * kh);
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                lwmma_bf16x8_t af[NT];
#pragma unroll
                for (int t = 0; t < NT; t++)
                    af[t] = *(const lwmma_bf16x8_t*)(base + t * XPL + (wmo + mt * 16 + l16) * LD + sub * 16 + 8 * kh);
#pragma unroll
                for (int t = 0; t < NT; t++) {
#pragma unroll
                    for (int nt = 0; nt < NTN; nt++)
                        acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(af[t], bf[nt], acc[mt][nt]);
                }
            }
        }
        sstore(cur ^ 1);
        __syncthreads();
    };

    // Chain g = kz * 4 + quarter; the K walk is ascending, so chain g owns steps [g * nq, (g + 1) * nq).
    int s = 0;
    for (int g = 0; g < 4 * split_k; g++) {
        for (int b = 0; b < nq; b++, s++) step(s);
        const int q = g & 3;
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int nt = 0; nt < NTN; nt++) {
                if (q == 0) {
                    s01[mt][nt] = acc[mt][nt];
                } else if (q == 1) {
                    s01[mt][nt] = s01[mt][nt] + acc[mt][nt];
                } else if (q == 2) {
                    c2[mt][nt] = acc[mt][nt];
                } else {
                    const lwmma_f8_t p = s01[mt][nt] + (c2[mt][nt] + acc[mt][nt]);
                    if (split_k == 1) vv[mt][nt] = p;
                    else vv[mt][nt] = vv[mt][nt] + p;
                }
                acc[mt][nt] = lwmma_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
            }
    }

#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nt = 0; nt < NTN; nt++) {
            const int col = n0 + wno + nt * 16 + l16;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int token = t0 + wmo + mt * 16 + 8 * kh + i;
                if (token < rows && col < M) {
                    const float v = vv[mt][nt][i];
                    Y[(size_t)token * M + col] = hc_act ? lwmma_hc_act(v, scale) : v;
                }
            }
        }
}

#define LWMMA_TILE_ENTRY(NT, BM, BN, WM, WN)                                                          \
    extern "C" __global__ __launch_bounds__(32 * WM * WN) void                                        \
        lwmma_tile_bf16_t##NT##_b##BM##x##BN##w##WM##x##WN(                                           \
            const unsigned short* __restrict__ W, const float* __restrict__ X, float* __restrict__ Y, \
            int M, int K, int rows, int split_k, float scale, int hc_act) {                          \
        lwmma_tile_body<NT, BM, BN, WM, WN>(W, X, Y, M, K, rows, split_k, scale, hc_act);             \
    }

#define LWMMA_TILE_ENTRIES(NT)           \
    LWMMA_TILE_ENTRY(NT, 64, 64, 4, 2)   \
    LWMMA_TILE_ENTRY(NT, 64, 128, 2, 4)  \
    LWMMA_TILE_ENTRY(NT, 128, 64, 4, 2)  \
    LWMMA_TILE_ENTRY(NT, 128, 128, 4, 4) \
    LWMMA_TILE_ENTRY(NT, 32, 64, 2, 2)   \
    LWMMA_TILE_ENTRY(NT, 32, 128, 2, 4)  \
    LWMMA_TILE_ENTRY(NT, 64, 64, 2, 2)   \
    LWMMA_TILE_ENTRY(NT, 64, 128, 4, 2)  \
    LWMMA_TILE_ENTRY(NT, 32, 128, 2, 2)  \
    LWMMA_TILE_ENTRY(NT, 64, 64, 4, 1)   \
    LWMMA_TILE_ENTRY(NT, 32, 64, 2, 1)   \
    LWMMA_TILE_ENTRY(NT, 128, 64, 8, 1)  \
    LWMMA_TILE_ENTRY(NT, 64, 256, 4, 4)

LWMMA_TILE_ENTRIES(1)
LWMMA_TILE_ENTRIES(2)
