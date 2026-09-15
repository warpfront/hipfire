// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniil Markevich
// hipfire — see LICENSE and NOTICE in the project root.
//
// EXPERIMENT (exp/iu4-direct-gfx11): MQ4G256V2 MMQ with weight nibbles fed
// DIRECTLY to wmma_i32_16x16x16_iu4 — no nibble->int8
// expansion in the loader, halved A-side LDS bytes — with int4 activations
// from the W4A4 recipe (per-128 MSE-clip, see block_i4_128 below).
//
// This is a copy of the X128 structure of
// gemm_mq4g256v2_residual_mmq.hip with the following changes:
//  - load_iu4_tile copies the 32 raw nibble words per 256-K row straight
//    into LDS (no expand VALU). x_qs stride 44 ints (32 nibble words +
//    8 half2 weight-header slots + 4 pad) vs 76 before.
//  - Y tile holds int4 activations: per 128-K block per column 64 B of
//    nibbles + (f32 d, i32 s) header = 72 B = 18 ints (vs 144 B Q8_1).
//  - Consumer: per 32-K block, TWO iu4 WMMAs (one per 16 K), same WMMA
//    count as today's iu8 pairs; 8 wmmas accumulate into ONE i32 C per
//    128-K half, then ONE fold:
//      f += sc_w*d_x*C + zp_w*d_x*s_x   (s_x = exact int sum of the 128 x_q)
//  - Nibble K-order contract: packed weight word bit [4m..4m+3] is K element
//    m; X nibbles packed identically (even K low, odd K high, two's
//    complement). Probe-verified on gfx1100: nibble position == K index on
//    BOTH sides (one-hot/misaligned-pattern tests), signA=false unsigned,
//    signB=true signed, C map acc[j]=C[2j+(lane>>4)][lane&15] unchanged.
//    NO permutation needed on either side.
//  - Lanes r and r+16 load byte-identical fragments (address depends on
//    lane%16 only), so duplicate-lane ownership is moot.
//
// LDS: (128*18 + 128*44)*4 = 31744 B (vs 57344 B baseline) -> 2 blocks/CU.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define HIPFIRE_RDNA3 1
#else
#define HIPFIRE_RDNA3 0
#endif

#define IU4_MMQ_X 128
#define IU4_MMQ_Y 128
#define IU4_MMQ_NWARPS 8
#define IU4_WARP_SIZE 32
#define IU4_TILE_Y_K 18
#define IU4_TILE_X_K 44
#define IU4_X_QS_WORDS 32
#define IU4_X_DM_OFF 32

// int4 activation block: one 128-K group for one token (column).
// qs: 64 B, even K low nibble / odd K high nibble, two's complement.
// d: per-128 scale (MSE-clip grid argmin). s: exact int sum of the 128 x_q.
struct block_i4_128 {
    float d;
    int s;
    unsigned char qs[64];
};
static_assert(sizeof(block_i4_128) == 72, "bad block size");

// int4 activation quantize for the iu4 MMQ path: same (X, Y, K, N) grid
// contract as quantize_q8_1_mmq_ds4_x128 (block (256,1,1), each thread 4
// floats, grid [(K+1023)/1024, N]), but emits block_i4_128 (72 B: f32 d,
// i32 s, 64 B nibbles) with the W4A4 recipe — per-128 amax, 8-candidate
// MSE-clip grid d_j=(amax/7)*0.5*(1+j/7), rintf, clamp[-8,7], strict-<
// argmin; zero group -> d=1, q=0, s=0. One wave32 covers exactly one
// 128-K group (lane l owns elements 4l..4l+3), so amax/MSE/sum reduce with
// width-32 shuffles (offsets 16..1), wave64-safe like the Q8_1 preludes.
// MSE accumulates in f32 with explicit single-rounding FMA (__fmaf_rn;
// f64 runs at 1/16 rate on RDNA3) to match the lane-mirrored CPU-pack
// oracle bit-exactly; s is an exact integer reduction. Arch-portable
// (no WMMA): shared by every arch whose launcher admits the iu4 consumer.
extern "C" __global__ void quantize_int4_mmq_ds128(
    const float* __restrict__ X,
    block_i4_128* __restrict__ Y,
    int K,
    int N
) {
    const int token = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int i0 = t * 4;
    if (token >= N || i0 >= K) return;
    const int lane = threadIdx.x & 31;

    const float4 xi = (i0 + 3 < K)
        ? *(const float4*)(X + (long long)token * K + i0)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float xv[4] = {xi.x, xi.y, xi.z, xi.w};

    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }

    float best_d = 1.0f;
    if (amax != 0.0f) {
        float best_mse = 1.0e30f;
        for (int j = 0; j < 8; ++j) {
            const float dj = (amax / 7.0f) * 0.5f * (1.0f + (float)j / 7.0f);
            float mse = 0.0f;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                float q = rintf(xv[e] / dj);
                q = fminf(fmaxf(q, -8.0f), 7.0f);
                const float err = __fmaf_rn(-q, dj, xv[e]);
                mse = __fmaf_rn(err, err, mse);
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                mse += __shfl_xor(mse, offset, 32);
            }
            if (mse < best_mse) {
                best_mse = mse;
                best_d = dj;
            }
        }
    }

    int q4[4];
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        float q = (amax == 0.0f) ? 0.0f : rintf(xv[e] / best_d);
        q = fminf(fmaxf(q, -8.0f), 7.0f);
        q4[e] = (int)q;
    }
    int s = q4[0] + q4[1] + q4[2] + q4[3];
    for (int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_xor(s, offset, 32);
    }

    const int kblock = i0 / 128;
    block_i4_128* out = Y + (long long)kblock * N + token;
    unsigned char* qb = out->qs;
    qb[lane * 2 + 0] = (unsigned char)((q4[0] & 15) | ((q4[1] & 15) << 4));
    qb[lane * 2 + 1] = (unsigned char)((q4[2] & 15) | ((q4[3] & 15) << 4));
    if ((i0 & 127) == 0) {
        out->d = best_d;
        out->s = s;
    }
}

#if !HIPFIRE_RDNA3

#ifdef HIPFIRE_MMQ_IU4_LUT
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_lloyd(
    const char*, const block_i4_128*, float*, int, int, int, int,
    unsigned, unsigned, unsigned, unsigned) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_add_lloyd(
    const char*, const block_i4_128*, float*, int, int, int, int,
    unsigned, unsigned, unsigned, unsigned) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_set_lloyd(
    const char*, const block_i4_128*, float*, int, int, int, int,
    unsigned, unsigned, unsigned, unsigned) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_lloyd(
    const char*, const block_i4_128*, float*, int, int, int, int,
    unsigned, unsigned, unsigned, unsigned) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_lloyd(
    const char*, const block_i4_128*, float*, int, int, int, int,
    unsigned, unsigned, unsigned, unsigned) {}
#else
extern C __global__ void gemm_mq4g256v2_residual_mmq_iu4(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
extern C __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_add(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
extern C __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_set(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
extern C __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
extern C __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
#endif

#else

enum data_layout {
    DATA_LAYOUT_I_MAJOR = 0,
    DATA_LAYOUT_J_MAJOR = 10,
};

template <int I_, int J_, typename T, data_layout dl = DATA_LAYOUT_I_MAJOR>
struct tile {};

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I * J / 32;
    T x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int) {
        return threadIdx.x % 16;
    }

    static __device__ __forceinline__ int get_j(const int l) {
        if constexpr (I == 16 && J == 16) {
            return 2 * l + (threadIdx.x / 16);
        } else {
            return l;
        }
    }
};

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_J_MAJOR> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::ne;
    T x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int l) {
        return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_j(l);
    }

    static __device__ __forceinline__ int get_j(const int l) {
        return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_i(l);
    }
};

using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
#ifdef HIPFIRE_MMQ_IU4_LUT
// Combined HL byte-table lookup: T[q]=(H[q]<<4)|L[q] packed as 16 LE bytes
// across T0..T3. Selector bytes are nibble indices 0..15; bit3 picks the
// upper half (T2/T3), low 3 bits the byte within the pair (same shape as
// mq4_lut8 / fp8.gfx12).
__device__ static inline unsigned iu4_lut8_hl(
    unsigned sel,
    unsigned T0, unsigned T1, unsigned T2, unsigned T3
) {
    const unsigned lo = __builtin_amdgcn_perm(T1, T0, sel & 0x07070707u);
    const unsigned hi = __builtin_amdgcn_perm(T3, T2, sel & 0x07070707u);
    const unsigned m = ((sel >> 3) & 0x01010101u) * 0xFFu;
    return (hi & m) | (lo & ~m);
}

// Per-dword nibble substitution: W holds 8 codes → Hword/Lword with the
// same even-low/odd-high nibble packing the iu4 WMMA expects.
__device__ static inline void iu4_split_hl_dword(
    unsigned W,
    unsigned T0, unsigned T1, unsigned T2, unsigned T3,
    unsigned* Hword, unsigned* Lword
) {
    const unsigned lo = W & 0x0F0F0F0Fu;
    const unsigned hi = (W >> 4) & 0x0F0F0F0Fu;
    const unsigned plo = iu4_lut8_hl(lo, T0, T1, T2, T3);
    const unsigned phi = iu4_lut8_hl(hi, T0, T1, T2, T3);
    const unsigned Hlo = (plo >> 4) & 0x0F0F0F0Fu;
    const unsigned Llo = plo & 0x0F0F0F0Fu;
    const unsigned Hhi = (phi >> 4) & 0x0F0F0F0Fu;
    const unsigned Lhi = phi & 0x0F0F0F0Fu;
    *Hword = Hlo | (Hhi << 4);
    *Lword = Llo | (Lhi << 4);
}
#endif

// Raw nibble-word copy: 32 words per 256-K row, NO expand VALU.
// Header loop identical to the iu8 path (dual-fp16 headers, ksc select).
template <bool FULL>
static __device__ __forceinline__ void load_iu4_tile(
    const char* __restrict__ A,
    int* __restrict__ tile_x,
    int row0,
    int kb,
    int M,
    int groups_per_row
) {
    int* x_qs = tile_x;
    half2* x_dm = (half2*)(x_qs + IU4_X_DM_OFF);

    const int txi = threadIdx.x;

    for (int i0 = 0; i0 < IU4_MMQ_Y; i0 += IU4_MMQ_NWARPS) {
        const int i = i0 + threadIdx.y;
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const char* gp = A + ((long long)row * groups_per_row + kb) * 136;
        x_qs[i * IU4_TILE_X_K + txi] = *(const int*)(gp + 8 + txi * 4);
    }

    constexpr int rows_per_warp_meta = 16;
    for (int i0 = 0; i0 < IU4_MMQ_Y; i0 += IU4_MMQ_NWARPS * rows_per_warp_meta) {
        const int i = i0 + threadIdx.y * rows_per_warp_meta + threadIdx.x / 2;
        if (i < IU4_MMQ_Y) {
            const int row = FULL ? row0 + i : min(row0 + i, M - 1);
            const char* gp = A + ((long long)row * groups_per_row + kb) * 136;
            const int ksc = threadIdx.x & 1;
            const unsigned int hs = *(const unsigned int*)(gp + 4 * ksc);
            const float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
            const float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
#ifdef HIPFIRE_MMQ_IU4_LUT
            // sc' = sc·(1/16); dmA.y' = zp − 120·sc' (Outcome B bias fold).
            // Headers on disk/GPU stay centered zp' from apply_lloyd_centering.
            const float sc_p = sc * 0.0625f;
            const float zp_p = zp - 120.0f * sc_p;
            const half2 dm = make_half2((_Float16)sc_p, (_Float16)zp_p);
#else
            const half2 dm = make_half2((_Float16)sc, (_Float16)zp);
#endif
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                x_dm[i * IU4_TILE_X_K + 4 * ksc + l] = dm;
        }
    }
}

// X128-structured dot over ONE 128-K half (h = 0/1): 8 iu4 WMMAs into one
// i32 C per (j0, n), then one fold with (d_x, s_x=int sum).
static __device__ __forceinline__ void vec_dot_i4_x128(
    const int* __restrict__ x,
    const int* __restrict__ y,
    float* __restrict__ sum,
    int h
#ifdef HIPFIRE_MMQ_IU4_LUT
    , unsigned T0, unsigned T1, unsigned T2, unsigned T3
#endif
) {
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;

    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;

    y += (threadIdx.y % ntx) * (tile_C::J * IU4_TILE_Y_K);

    const int* x_qs = x;
    const int* y_qs = y + 2;
    const int* y_ds = y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
    const int j_lane = tile_C::get_j(0);
    const int lane_row = threadIdx.x % 16;

    for (int j0 = 0; j0 < IU4_MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
#ifdef HIPFIRE_MMQ_IU4_LUT
            int32x8_t accH = {0, 0, 0, 0, 0, 0, 0, 0};
            int32x8_t accL = {0, 0, 0, 0, 0, 0, 0, 0};
            #pragma nounroll
            for (int t = 0; t < 4; ++t) {
                const int* ap = x_qs + (i0 + n * tile_C::I + lane_row) * IU4_TILE_X_K + h * 16 + t * 4;
                const int* bp = y_qs + (j0 + lane_row) * IU4_TILE_Y_K + t * 4;
                const int32x2_t av0 = *(const int32x2_t*)(ap + 0);
                const int32x2_t av1 = *(const int32x2_t*)(ap + 2);
                const int32x2_t bv0 = *(const int32x2_t*)(bp + 0);
                const int32x2_t bv1 = *(const int32x2_t*)(bp + 2);
                unsigned H0, L0, H1, L1, H2, L2, H3, L3;
                iu4_split_hl_dword((unsigned)av0[0], T0, T1, T2, T3, &H0, &L0);
                iu4_split_hl_dword((unsigned)av0[1], T0, T1, T2, T3, &H1, &L1);
                iu4_split_hl_dword((unsigned)av1[0], T0, T1, T2, T3, &H2, &L2);
                iu4_split_hl_dword((unsigned)av1[1], T0, T1, T2, T3, &H3, &L3);
                int32x2_t hav0; hav0[0] = (int)H0; hav0[1] = (int)H1;
                int32x2_t hav1; hav1[0] = (int)H2; hav1[1] = (int)H3;
                int32x2_t lav0; lav0[0] = (int)L0; lav0[1] = (int)L1;
                int32x2_t lav1; lav1[0] = (int)L2; lav1[1] = (int)L3;
                // signA=false unsigned, signB=true signed, no-sat — same as uniform.
                accH = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, hav0, true, bv0, accH, false);
                accH = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, hav1, true, bv1, accH, false);
                accL = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, lav0, true, bv0, accL, false);
                accL = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, lav1, true, bv1, accL, false);
            }
            const float d_x = *(const float*)(y_ds + (j0 + j_lane) * IU4_TILE_Y_K);
            const int s_x = *(const int*)(y_ds + (j0 + j_lane) * IU4_TILE_Y_K + 1);
            const int* accH_i = (const int*)&accH;
            const int* accL_i = (const int*)&accL;
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                const half2 dm = ((const half2*)(x_qs + i * IU4_TILE_X_K + IU4_X_DM_OFF))[4 * h];
                const float2 dmA = __half22float2(dm);
                // C = 16·ΣH·x + ΣL·x; sc'/zp' already in dmA from load_iu4_tile.
                const int C = 16 * accH_i[l] + accL_i[l];
                sum[(j0 / tile_C::J + n) * tile_C::ne + l] +=
                    dmA.x * d_x * (float)C + dmA.y * d_x * (float)s_x;
            }
#else
            int32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
            #pragma nounroll
            for (int t = 0; t < 4; ++t) {
                const int* ap = x_qs + (i0 + n * tile_C::I + lane_row) * IU4_TILE_X_K + h * 16 + t * 4;
                const int* bp = y_qs + (j0 + lane_row) * IU4_TILE_Y_K + t * 4;
                const int32x2_t av0 = *(const int32x2_t*)(ap + 0);
                const int32x2_t av1 = *(const int32x2_t*)(ap + 2);
                const int32x2_t bv0 = *(const int32x2_t*)(bp + 0);
                const int32x2_t bv1 = *(const int32x2_t*)(bp + 2);
                acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av0, true, bv0, acc, false);
                acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av1, true, bv1, acc, false);
            }
            const float d_x = *(const float*)(y_ds + (j0 + j_lane) * IU4_TILE_Y_K);
            const int s_x = *(const int*)(y_ds + (j0 + j_lane) * IU4_TILE_Y_K + 1);
            const int* acc_i = (const int*)&acc;
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                const half2 dm = ((const half2*)(x_qs + i * IU4_TILE_X_K + IU4_X_DM_OFF))[4 * h];
                const float2 dmA = __half22float2(dm);
                sum[(j0 / tile_C::J + n) * tile_C::ne + l] +=
                    dmA.x * d_x * (float)acc_i[l] + dmA.y * d_x * (float)s_x;
            }
#endif
        }
    }
}

template <bool FULL>
static __device__ __forceinline__ void gemm_iu4_body(
    const char* __restrict__ A,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
#ifdef HIPFIRE_MMQ_IU4_LUT
    , unsigned c16_0, unsigned c16_1, unsigned c16_2, unsigned c16_3
#endif
) {
    const int row0 = blockIdx.x * IU4_MMQ_Y;
    const int col0 = blockIdx.y * IU4_MMQ_X;
    if (!FULL && (row0 >= M || col0 >= N)) return;

    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;
#ifdef HIPFIRE_MMQ_IU4_LUT
    // Prologue: C16 signed bytes S[q] → U=S+120 → T[q]=(H<<4)|L = U (byte).
    // 16 bytes = 4 dwords in SGPRs; one lookup yields both nibbles.
    unsigned T0 = 0u, T1 = 0u, T2 = 0u, T3 = 0u;
    {
        const unsigned c16[4] = {c16_0, c16_1, c16_2, c16_3};
        unsigned T[4] = {0u, 0u, 0u, 0u};
        #pragma unroll
        for (int q = 0; q < 16; ++q) {
            const int S = (int)(signed char)((c16[q / 4] >> ((q % 4) * 8)) & 0xFFu);
            const unsigned U = (unsigned)(S + 120);
            T[q / 4] |= (U & 0xFFu) << ((q % 4) * 8);
        }
        T0 = T[0]; T1 = T[1]; T2 = T[2]; T3 = T[3];
    }
#endif

    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + IU4_MMQ_X * IU4_TILE_Y_K;

    float sum[IU4_MMQ_X * IU4_MMQ_Y / (IU4_MMQ_NWARPS * IU4_WARP_SIZE)] = {0.0f};

    for (int kb = 0; kb < groups_per_row; ++kb) {
        load_iu4_tile<FULL>(A, tile_x, row0, kb, M, groups_per_row);

        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l0 = 0; l0 < IU4_MMQ_X * IU4_TILE_Y_K; l0 += IU4_MMQ_NWARPS * IU4_WARP_SIZE) {
            const int l = l0 + threadIdx.y * IU4_WARP_SIZE + threadIdx.x;
            if (FULL || col0 + l / IU4_TILE_Y_K < N) tile_y[l] = by0[l];
            else tile_y[l] = 0;
        }
        __syncthreads();
#ifdef HIPFIRE_MMQ_IU4_LUT
        vec_dot_i4_x128(tile_x, tile_y, sum, 0, T0, T1, T2, T3);
#else
        vec_dot_i4_x128(tile_x, tile_y, sum, 0);
#endif
        __syncthreads();

        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < IU4_MMQ_X * IU4_TILE_Y_K; l0 += IU4_MMQ_NWARPS * IU4_WARP_SIZE) {
                const int l = l0 + threadIdx.y * IU4_WARP_SIZE + threadIdx.x;
                if (FULL || col0 + l / IU4_TILE_Y_K < N) tile_y[l] = by1[l];
                else tile_y[l] = 0;
            }
            __syncthreads();
#ifdef HIPFIRE_MMQ_IU4_LUT
            vec_dot_i4_x128(tile_x, tile_y, sum, 1, T0, T1, T2, T3);
#else
            vec_dot_i4_x128(tile_x, tile_y, sum, 1);
#endif
            __syncthreads();
        }
    }

    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    #pragma unroll
    for (int j0 = 0; j0 < IU4_MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if (FULL || (row0 + i < M && col0 + j < N)) {
                    float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                    const float v = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
                    if (add) {
                        *yp += v;
                    } else {
                        *yp = v;
                    }
                }
            }
        }
    }
}

#ifdef HIPFIRE_MMQ_IU4_LUT
#define IU4_ENTRY(name, full, bounds_min) extern "C" __global__ __launch_bounds__(IU4_MMQ_NWARPS * IU4_WARP_SIZE, bounds_min) void name(     const char* __restrict__ A,     const block_i4_128* __restrict__ Xq,     float* __restrict__ Y,     int M,     int K,     int N,     int add,     unsigned c16_0, unsigned c16_1, unsigned c16_2, unsigned c16_3 ) {     gemm_iu4_body<full>(A, Xq, Y, M, K, N, add, c16_0, c16_1, c16_2, c16_3); }

IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_lloyd, false, 2)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_add_lloyd, true, 2)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_set_lloyd, true, 2)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_lloyd, true, 3)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_lloyd, true, 3)
#else
#define IU4_ENTRY(name, full, bounds_min) extern "C" __global__ __launch_bounds__(IU4_MMQ_NWARPS * IU4_WARP_SIZE, bounds_min) void name(     const char* __restrict__ A,     const block_i4_128* __restrict__ Xq,     float* __restrict__ Y,     int M,     int K,     int N,     int add ) {     gemm_iu4_body<full>(A, Xq, Y, M, K, N, add); }

IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4, false, 2)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_add, true, 2)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_set, true, 2)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3, true, 3)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3, true, 3)
#endif

#endif
