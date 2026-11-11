// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// MQ8G256 (qt14) prefill GEMM for gfx11 (RDNA3 — gfx1100 and gfx1151): int8 FWHT-rotated weights x int8
// per-(token, 256-group) rotated activations on `v_wmma_i32_16x16x16_iu8` (wave32), F32 output.
// Sibling of `gemm_mq8g256_mmq_i8.gfx12.hip` (same numerics, same W / X / Y contracts, same layout A / B
// switch) built on the structure of `gemm_q8_0_mmq_i8.gfx11.hip`:
//
//   Y[n, m] = sum_e fma-fold over K128 epochs e of  RN(sw[m, e/2] * dx[n, e/2]) * dot128_e(m, n)
//
// per output element the K128 epochs are folded in ascending order into an F32 accumulator that starts at
// +0 as `acc = fmaf(RN(sw256 * dx256), (float)dot128, acc)`; dot128 is an exact int32 (|dot| <= 2^21, the
// 0x4B400000-seeded accumulator recovers it exactly), one W scale (the f16 header of the 256-group) and one
// X scale feed the two epochs of a group. The tile shape cannot change a bit; a CPU reference with this
// association reproduces the output (crates/rdna-compute/examples/mq8_prefill_lab.rs).
//
//   W : MQ8G256 (qt=14). Layout A (WB = false): the artifact's 258 B groups {f16 scale, int8 q[256]} read in
//       place (a group's codes are 2 B off a dword boundary in even groups of a 4-aligned row: aligned
//       dword loads + one funnel shift per dword, one exactly-in-bounds 16-bit load for the 17th dword;
//       2-byte aligned base). Layout B (WB = true): the f16 scale plane [M, K/256] followed, at the next
//       256 B boundary, by the row-major int8 code plane [M, K] (16 B loads; 16-byte aligned base);
//       `mq8g256_b_code_offset` is the plane offset.
//   X : the output of `mq8_rotate_quantize_x_batched`: int8 [N, K] row-major + F32 scales [K/256, N].
//   Y : F32 [N, M] row-major; RBF16 rounds each output to BF16 (RN-even, widened back to F32).
//
// RDNA3 operand/C layout: A and B are 16 int8 per lane (one K16 step), lanes r and r+16 carry identical
// operands, C element j of lane L is C[2j + (L >> 4)][L & 15]. The WEIGHTS ride on A and the tokens on B.
// The W rows are staged to LDS in slot order slot(r) = 2 * (r & 7) + ((r >> 3) & 1) inside each 16-row
// group so that a lane owns 8 CONSECUTIVE weight rows of one token (see gemm_q8_0_mmq_i8.gfx11.hip).
//
// Tile BM x BN walks K in 128-wide slabs through ONE LDS buffer (two barriers per slab), LDS rows padded to
// 144 B. Grid [ceil(M/BM), ceil(N/BN)]; block 2*BM threads (two threads stage one W row). K % 256 == 0.
// Tails: weight rows past M re-read row M-1, tokens past N re-read token N-1 with scale 0; stores are
// masked, every in-range output is written once.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

#if defined(__gfx1100__) || defined(__gfx1151__)

#define MQ8G11_PITCH 144
#define MQ8G11_MAGIC 0x4B400000

typedef int32_t __attribute__((ext_vector_type(4))) mq8g11_i32x4_t;
typedef int32_t __attribute__((ext_vector_type(8))) mq8g11_i32x8_t;
typedef float   __attribute__((ext_vector_type(8))) mq8g11_f32x8_t;
typedef unsigned __attribute__((ext_vector_type(4))) mq8g11_u32x4_t;

// LDS slot of weight row r (r = row inside the tile).
static __device__ __forceinline__ int mq8g11_slot(int r) {
    return (r & ~15) | (2 * (r & 7)) | ((r >> 3) & 1);
}

// Layout B: byte offset of the int8 code plane (the scale plane is M * (K/256) f16, padded to 256 B).
static __device__ __forceinline__ long long mq8g256_b_code_offset(int M, int K) {
    return (((long long)M * (long long)(K >> 8) * 2ll) + 255ll) & ~255ll;
}

// Bit-level RN-even F32 -> BF16 -> F32 round trip (`bf16_round_trip_f32_strided`): non-finite values pass through.
static __device__ __forceinline__ float mq8g11_bf16_rn(float v) {
    if (!isfinite(v)) return v;
    union { float f; unsigned int u; } box;
    box.f = v;
    const unsigned int lsb = (box.u >> 16) & 1u;
    box.u = (box.u + 0x7FFFu + lsb) & 0xFFFF0000u;
    return box.f;
}

template <int BM, int BN, int WM, int WN, bool RBF16, bool WB>
static __device__ __forceinline__ void gemm_mq8g256_mmq_i8_gfx11_body(
    const char* __restrict__ Wq, const signed char* __restrict__ Xq, const float* __restrict__ Xs,
    float* __restrict__ Y, int M, int K, int N) {
    constexpr int NW = WM * WN;
    constexpr int NT = NW * 32;
    static_assert(NT == 2 * BM, "W staging maps two threads per row");
    constexpr int RG = BM / WM / 16;
    constexpr int NB = BN / WN / 16;
    static_assert(RG * WM * 16 == BM && NB * WN * 16 == BN, "tile split");
    static_assert(BN <= NT, "one X scale per thread");
    constexpr int W_BYTES = BM * MQ8G11_PITCH;
    constexpr int X_BYTES = BN * MQ8G11_PITCH;
    constexpr int WS_BYTES = BM * 4;
    constexpr int XS_BYTES = BN * 4;
    constexpr int XCH = 8 * BN / NT;   // 16 B X chunks per thread (8 per 128-K row)
    static_assert(XCH * NT == 8 * BN, "X chunks tile evenly");
    __shared__ __attribute__((aligned(16))) char lds[W_BYTES + X_BYTES + WS_BYTES + XS_BYTES];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int m_lane = lane & 15;
    const int hi = lane >> 4;
    const int wm = wave % WM;
    const int wn = wave / WM;
    const int wave_row0 = wm * (RG * 16);
    const int wave_tok0 = wn * (NB * 16);
    const int tile_m = blockIdx.x * BM;
    const int tile_n = blockIdx.y * BN;
    if (tile_m >= M || tile_n >= N) return;

    const int nslab = K >> 7;
    const int ngrp = K >> 8;
    const int w_row_t = tid % BM;
    const int w_half = __builtin_amdgcn_readfirstlane(tid / BM);   // codes [64*half, 64*half + 64) of the slab
    const int w_row_g = min(tile_m + w_row_t, M - 1);
    const int w_slot = mq8g11_slot(w_row_t);
    const char* w_row;     // WB: this row's codes; !WB: this row's first 258 B group
    const char* w_sc;      // WB: this row's f16 scales
    if constexpr (WB) {
        w_row = Wq + mq8g256_b_code_offset(M, K) + (long long)w_row_g * K;
        w_sc = Wq + (long long)w_row_g * ngrp * 2;
    } else {
        w_row = Wq + (long long)w_row_g * ngrp * 258;
        w_sc = w_row;
    }

    const int xs_tok = min(tile_n + tid, N - 1);
    const bool xs_ok = tid < BN && tile_n + tid < N;

    mq8g11_u32x4_t xr[XCH];
    float xsr = 0.0f;
    unsigned wr[17];
    unsigned wsh = 0u;     // f16 bits of the slab's 256-group scale (half 0 only)
    unsigned sh = 0u;      // layout A: funnel shift of this thread's codes (0 aligned, 16 off by 2 B)
    const int bpar = (int)((uintptr_t)Wq & 3u);   // layout A: 0, or 2 for a 2-byte aligned base

    auto load_slab = [&](int s) {
        #pragma unroll
        for (int i = 0; i < XCH; i++) {
            const int r = min(tile_n + (tid >> 3) + (NT >> 3) * i, N - 1);
            xr[i] = *(const mq8g11_u32x4_t*)(Xq + (long long)r * K + s * 128 + (tid & 7) * 16);
        }
        xsr = xs_ok ? Xs[(long long)(s >> 1) * N + xs_tok] : 0.0f;
        if constexpr (WB) {
            const mq8g11_u32x4_t* p = (const mq8g11_u32x4_t*)(w_row + (long long)s * 128 + w_half * 64);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const mq8g11_u32x4_t v = p[j];
                wr[4 * j] = v.x; wr[4 * j + 1] = v.y; wr[4 * j + 2] = v.z; wr[4 * j + 3] = v.w;
            }
            wr[16] = 0u;
            if (w_half == 0) wsh = *(const unsigned short*)(w_sc + (long long)(s >> 1) * 2);
        } else {
            const char* g = w_row + (long long)(s >> 1) * 258;
            // Byte offset of this thread's 64 codes from Wq; the aligned-down pointer is derived from Wq by
            // integer addition (the address space stays global: an integer round trip would turn every
            // weight load into a FLAT load). bpar = Wq mod 4 (0 or 2) shifts the offset into address space.
            const long long cv = (long long)(g - Wq) + 2 + (s & 1) * 128 + w_half * 64 + bpar;
            sh = (unsigned)(cv & 2ll) * 8u;
            const char* p = Wq + ((cv & ~3ll) - bpar);
            const unsigned* pd = (const unsigned*)p;
            #pragma unroll
            for (int j = 0; j < 16; j++) wr[j] = pd[j];
            const unsigned tl = *(const unsigned short*)(p + (sh ? 64 : 62));
            wr[16] = sh ? tl : 0u;
            if (w_half == 0) wsh = *(const unsigned short*)(g);
        }
    };
    auto store_slab = [&]() {
        #pragma unroll
        for (int i = 0; i < XCH; i++) {
            const int q = tid + i * NT;
            *(mq8g11_u32x4_t*)(lds + W_BYTES + (q >> 3) * MQ8G11_PITCH + (q & 7) * 16) = xr[i];
        }
        float* xs_l = (float*)(lds + W_BYTES + X_BYTES + WS_BYTES);
        if (tid < BN) xs_l[tid] = xsr;
        unsigned qw[16];
        if constexpr (WB) {
            #pragma unroll
            for (int j = 0; j < 16; j++) qw[j] = wr[j];
        } else {
            #pragma unroll
            for (int j = 0; j < 16; j++) qw[j] = __builtin_amdgcn_alignbit(wr[j + 1], wr[j], sh);
        }
        char* wrow = lds + w_slot * MQ8G11_PITCH + w_half * 64;
        *(uint4*)(wrow) = make_uint4(qw[0], qw[1], qw[2], qw[3]);
        *(uint4*)(wrow + 16) = make_uint4(qw[4], qw[5], qw[6], qw[7]);
        *(uint4*)(wrow + 32) = make_uint4(qw[8], qw[9], qw[10], qw[11]);
        *(uint4*)(wrow + 48) = make_uint4(qw[12], qw[13], qw[14], qw[15]);
        float* ws_l = (float*)(lds + W_BYTES + X_BYTES);
        if (w_half == 0) ws_l[w_row_t] = __half2float(__ushort_as_half((unsigned short)wsh));
    };

    mq8g11_f32x8_t acc[RG][NB];
    #pragma unroll
    for (int rg = 0; rg < RG; rg++) {
        #pragma unroll
        for (int nb = 0; nb < NB; nb++) {
            acc[rg][nb] = (mq8g11_f32x8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }
    }

    load_slab(0);
    store_slab();
    __syncthreads();

    const char* xb = lds + W_BYTES;
    const float* ws_l = (const float*)(lds + W_BYTES + X_BYTES);
    const float* xs_l = (const float*)(lds + W_BYTES + X_BYTES + WS_BYTES);
    for (int s = 0; s < nslab; s++) {
        const bool more = (s + 1) < nslab;
        if (more) load_slab(s + 1);

        mq8g11_i32x4_t af0[4][RG], af1[4][RG];
        #pragma unroll
        for (int blk = 0; blk < 4; blk++) {
            #pragma unroll
            for (int rg = 0; rg < RG; rg++) {
                const char* p = lds + (wave_row0 + rg * 16 + m_lane) * MQ8G11_PITCH + blk * 32;
                af0[blk][rg] = *(const mq8g11_i32x4_t*)(p);
                af1[blk][rg] = *(const mq8g11_i32x4_t*)(p + 16);
            }
        }
        float wsv[RG][8];
        #pragma unroll
        for (int rg = 0; rg < RG; rg++) {
            const float4 w0 = *(const float4*)(ws_l + wave_row0 + rg * 16 + 8 * hi);
            const float4 w1 = *(const float4*)(ws_l + wave_row0 + rg * 16 + 8 * hi + 4);
            wsv[rg][0] = w0.x; wsv[rg][1] = w0.y; wsv[rg][2] = w0.z; wsv[rg][3] = w0.w;
            wsv[rg][4] = w1.x; wsv[rg][5] = w1.y; wsv[rg][6] = w1.z; wsv[rg][7] = w1.w;
        }
        float xsv[NB];
        #pragma unroll
        for (int nb = 0; nb < NB; nb++) xsv[nb] = xs_l[wave_tok0 + nb * 16 + m_lane];

        auto fold = [&](int rg, int nb, mq8g11_i32x8_t c) {
            // Whole-vector bitcast + subtract (the per-element form is miscompiled by the hipcc the
            // production gfx11 kernel was written against).
            const mq8g11_f32x8_t cf = __builtin_bit_cast(mq8g11_f32x8_t, c) - 12582912.0f;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                acc[rg][nb][j] = __builtin_fmaf(wsv[rg][j] * xsv[nb], cf[j], acc[rg][nb][j]);
            }
        };
        mq8g11_i32x8_t cprev = {0, 0, 0, 0, 0, 0, 0, 0};
        #pragma unroll
        for (int nb = 0; nb < NB; nb++) {
            mq8g11_i32x4_t bf0[4], bf1[4];
            #pragma unroll
            for (int blk = 0; blk < 4; blk++) {
                const char* p = xb + (wave_tok0 + nb * 16 + m_lane) * MQ8G11_PITCH + blk * 32;
                bf0[blk] = *(const mq8g11_i32x4_t*)(p);
                bf1[blk] = *(const mq8g11_i32x4_t*)(p + 16);
            }
            #pragma unroll
            for (int rg = 0; rg < RG; rg++) {
                mq8g11_i32x8_t c = {MQ8G11_MAGIC, MQ8G11_MAGIC, MQ8G11_MAGIC, MQ8G11_MAGIC,
                                    MQ8G11_MAGIC, MQ8G11_MAGIC, MQ8G11_MAGIC, MQ8G11_MAGIC};
                #pragma unroll
                for (int blk = 0; blk < 4; blk++) {
                    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af0[blk][rg], true, bf0[blk], c, false);
                    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af1[blk][rg], true, bf1[blk], c, false);
                }
                if (nb + rg > 0) {
                    // fold the previous tile after this tile's chain is issued
                    const int pt = nb * RG + rg - 1;
                    fold(pt % RG, pt / RG, cprev);
                }
                cprev = c;
            }
        }
        fold(RG - 1, NB - 1, cprev);

        __syncthreads();   // every wave is done reading the single buffer
        if (more) store_slab();
        __syncthreads();
    }

    // Epilogue: acc[rg][nb][j] = Y[token = tile_n + wave_tok0 + nb*16 + m_lane]
    //                             [row   = tile_m + wave_row0 + rg*16 + 8*hi + j].
    if constexpr (RBF16) {
        #pragma unroll
        for (int rg = 0; rg < RG; rg++) {
            #pragma unroll
            for (int nb = 0; nb < NB; nb++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) acc[rg][nb][j] = mq8g11_bf16_rn(acc[rg][nb][j]);
            }
        }
    }
    const bool vec_ok = ((M & 7) == 0) && (((uintptr_t)Y & 15u) == 0);
    #pragma unroll
    for (int nb = 0; nb < NB; nb++) {
        const int tok = tile_n + wave_tok0 + nb * 16 + m_lane;
        if (tok >= N) continue;
        #pragma unroll
        for (int rg = 0; rg < RG; rg++) {
            const int row0 = tile_m + wave_row0 + rg * 16 + 8 * hi;
            float* yp = Y + (long long)tok * M + row0;
            if (vec_ok) {
                if (row0 < M) {
                    *(float4*)(yp) = make_float4(acc[rg][nb][0], acc[rg][nb][1], acc[rg][nb][2], acc[rg][nb][3]);
                    *(float4*)(yp + 4) = make_float4(acc[rg][nb][4], acc[rg][nb][5], acc[rg][nb][6], acc[rg][nb][7]);
                }
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (row0 + j < M) yp[j] = acc[rg][nb][j];
                }
            }
        }
    }
}

#define MQ8G11_KERNEL(NAME, BMV, BNV, WMV, WNV, RBF16, WB)                                      \
    extern "C" __global__ __launch_bounds__(2 * (BMV), 2) void NAME(                            \
        const char* __restrict__ Wq, const signed char* __restrict__ Xq,                        \
        const float* __restrict__ Xs, float* __restrict__ Y, int M, int K, int N) {             \
        gemm_mq8g256_mmq_i8_gfx11_body<BMV, BNV, WMV, WNV, RBF16, WB>(Wq, Xq, Xs, Y, M, K, N);  \
    }
#define MQ8G11_TILES(WL, WB)                                                                    \
    MQ8G11_KERNEL(gemm_mq8g256_mmq_i8_gfx11_t128x128_##WL, 128, 128, 4, 2, false, WB)           \
    MQ8G11_KERNEL(gemm_mq8g256_mmq_i8_gfx11_t128x128_##WL##_bf16, 128, 128, 4, 2, true, WB)     \
    MQ8G11_KERNEL(gemm_mq8g256_mmq_i8_gfx11_t64x128_##WL, 64, 128, 2, 2, false, WB)             \
    MQ8G11_KERNEL(gemm_mq8g256_mmq_i8_gfx11_t64x128_##WL##_bf16, 64, 128, 2, 2, true, WB)       \
    MQ8G11_KERNEL(gemm_mq8g256_mmq_i8_gfx11_t64x64_##WL, 64, 64, 2, 2, false, WB)               \
    MQ8G11_KERNEL(gemm_mq8g256_mmq_i8_gfx11_t64x64_##WL##_bf16, 64, 64, 2, 2, true, WB)

MQ8G11_TILES(wa, false)
MQ8G11_TILES(wb, true)

#undef MQ8G11_TILES
#undef MQ8G11_KERNEL

#else

#define MQ8G11_STUB(NAME) \
    extern "C" __global__ void NAME(const char*, const signed char*, const float*, float*, int, int, int) {}
#define MQ8G11_STUBS(WL)                                                                                  \
    MQ8G11_STUB(gemm_mq8g256_mmq_i8_gfx11_t128x128_##WL) MQ8G11_STUB(gemm_mq8g256_mmq_i8_gfx11_t128x128_##WL##_bf16) \
    MQ8G11_STUB(gemm_mq8g256_mmq_i8_gfx11_t64x128_##WL) MQ8G11_STUB(gemm_mq8g256_mmq_i8_gfx11_t64x128_##WL##_bf16)   \
    MQ8G11_STUB(gemm_mq8g256_mmq_i8_gfx11_t64x64_##WL) MQ8G11_STUB(gemm_mq8g256_mmq_i8_gfx11_t64x64_##WL##_bf16)
MQ8G11_STUBS(wa)
MQ8G11_STUBS(wb)
#undef MQ8G11_STUBS
#undef MQ8G11_STUB

#endif
