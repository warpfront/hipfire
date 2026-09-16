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
//    into LDS (no expand VALU). x_qs stride 42 ints (32 nibble words +
//    8 half2 weight-header slots + 2 pad) vs 76 before.
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
// LDS: (128*18 + 128*42)*4 = 30720 B (vs 57344 B baseline) -> 2 blocks/CU.

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
// A2: stride 42. Rows r/r+8 collided at 44 (44r mod 32 = 12r); 42r mod 32 = 10r
// visits all 16 even start banks once; b64 alignment kept; LDS 31744 -> 30720 B.
// Halo: set 1722->1688 us, add 1684->1643, prefill 876->861 ms; 190 VGPR 0 spills.
#define IU4_TILE_X_K 42
#define IU4_X_QS_WORDS 32
#define IU4_X_DM_OFF 32

// block_i4_128 + quantize_int4_mmq_ds128 live in block_i4_128_quant.hip and
// are prepended by the host JIT (GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_SRC). Do not
// reintroduce a second MSE recipe copy here.

#if !HIPFIRE_RDNA3

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
            const half2 dm = make_half2((_Float16)sc, (_Float16)zp);
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
        }
    }
}

// W3 multi-chain (MC) dot: same wave map, LDS layout, fill, barriers, folds
// and epilogue as shipping; ONLY the WMMA issue structure changes. Instead
// of finishing one (j0,n) subtile's 8 K16 WMMAs before starting the next,
// the C chains of a subtile group stay live while u=0..7 issues across them,
// so independent WMMAs sit back-to-back for the scheduler to overlap.
//
// Bit-exactness: chain c's 8 WMMAs issue in the same (t,sub)=(u>>1,u&1)
// sequence into the same acc as shipping, and each chain folds with the
// shipping fold text (same loads, expression and order per output) once its
// 8 steps complete. Interleaving across chains never reorders one output's
// accumulation. No asm pins; fold blocks below are verbatim shipping.
//
// MC=2: chains (j0,n0),(j0,n1) — B fragment shared across n, A rows differ.
// MC=4: chains (c0,n0),(c0,n1),(c1,n0),(c1,n1) over a 64-wide jbase group —
//   A fragments shared across c, B fragments shared across n.
// ---- MC=2 copy serialization: four noinline block calls (J0 = 0/32/64/96)

// One MC=2 subtile pair's worth of sums (chains (J0,n0),(J0,n1)): 2x8 floats,
// carried by value so no array escapes across the block boundary.
struct iu4_mc2_sums {
    float v[16];
};

// Scalar arg plumbing for the block call below: aggregates cross byval
// (scratch) on this target, so the 16 sums cross as 16 scalar VGPR args.
// The struct return stays (returns ride in VGPRs, no scratch observed).
#define IU4_MC2_FARGS \
    float a00, float a01, float a02, float a03, float a04, float a05, float a06, float a07, \
    float a08, float a09, float a10, float a11, float a12, float a13, float a14, float a15
#define IU4_MC2_ACTUALS(s) \
    (s).v[0], (s).v[1], (s).v[2], (s).v[3], (s).v[4], (s).v[5], (s).v[6], (s).v[7], \
    (s).v[8], (s).v[9], (s).v[10], (s).v[11], (s).v[12], (s).v[13], (s).v[14], (s).v[15]

// Pack/unpack one copy's 16 sums (constant indices — forceinline so sum[]
// never escapes; it stays a VGPR array in the caller).
template <int J0>
static __device__ __forceinline__ iu4_mc2_sums iu4_mc2_pack(const float* __restrict__ sum) {
    iu4_mc2_sums s;
    #pragma unroll
    for (int l = 0; l < 8; ++l) {
        s.v[l] = sum[(J0 / 16 + 0) * 8 + l];
        s.v[8 + l] = sum[(J0 / 16 + 1) * 8 + l];
    }
    return s;
}

template <int J0>
static __device__ __forceinline__ void iu4_mc2_unpack(float* __restrict__ sum, iu4_mc2_sums s) {
    #pragma unroll
    for (int l = 0; l < 8; ++l) {
        sum[(J0 / 16 + 0) * 8 + l] = s.v[l];
        sum[(J0 / 16 + 1) * 8 + l] = s.v[8 + l];
    }
}

// One MC=2 copy: subtile pair at columns J0..J0+31, both row halves. Two C
// chains stay live over u=0..7 (same WMMA sequence as the inline MC=2 dot),
// then the two shipping folds run on the block's 16 sums.
//
// noinline is load-bearing: the four block calls in vec_dot_i4_x128_mc2 are
// opaque to scheduling/allocation, so copies serialize and at most one
// copy's accs+fragments+fold temps are live (plus the caller's sum[]). J0 is
// a template so every sum index stays constant (VGPRs, as shipping). The
// u-loop, both accs and both fold blocks are verbatim the MC=2 dot body;
// only sum[] accesses route through the by-value block struct.
template <int J0>
static __device__ __attribute__((noinline)) iu4_mc2_sums vec_dot_i4_x128_mc2_blk(
    const int* __restrict__ x_qs,
    const int* __restrict__ y_qs,
    const int* __restrict__ y_ds,
    int h,
    int i0,
    int j_lane,
    int lane_row,
    IU4_MC2_FARGS
) {
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    iu4_mc2_sums sums;
    sums.v[0] = a00; sums.v[1] = a01; sums.v[2] = a02; sums.v[3] = a03;
    sums.v[4] = a04; sums.v[5] = a05; sums.v[6] = a06; sums.v[7] = a07;
    sums.v[8] = a08; sums.v[9] = a09; sums.v[10] = a10; sums.v[11] = a11;
    sums.v[12] = a12; sums.v[13] = a13; sums.v[14] = a14; sums.v[15] = a15;
    int32x8_t acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    int32x8_t acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma nounroll
    for (int u = 0; u < 8; ++u) {
        const int t = u >> 1;
        const int s = u & 1;
        const int* ap0 = x_qs + (i0 + 0 * tile_C::I + lane_row) * IU4_TILE_X_K + h * 16 + t * 4 + s * 2;
        const int* ap1 = x_qs + (i0 + 1 * tile_C::I + lane_row) * IU4_TILE_X_K + h * 16 + t * 4 + s * 2;
        const int* bp = y_qs + (J0 + lane_row) * IU4_TILE_Y_K + t * 4 + s * 2;
        const int32x2_t av0 = *(const int32x2_t*)ap0;
        const int32x2_t av1 = *(const int32x2_t*)ap1;
        const int32x2_t bv = *(const int32x2_t*)bp;
        acc0 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av0, true, bv, acc0, false);
        acc1 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av1, true, bv, acc1, false);
    }
    {
        const float d_x = *(const float*)(y_ds + (J0 + j_lane) * IU4_TILE_Y_K);
        const int s_x = *(const int*)(y_ds + (J0 + j_lane) * IU4_TILE_Y_K + 1);
        const int* acc_i = (const int*)&acc0;
        #pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int i = i0 + 0 * tile_C::I + tile_C::get_i(l);
            const half2 dm = ((const half2*)(x_qs + i * IU4_TILE_X_K + IU4_X_DM_OFF))[4 * h];
            const float2 dmA = __half22float2(dm);
            sums.v[l] +=
                dmA.x * d_x * (float)acc_i[l] + dmA.y * d_x * (float)s_x;
        }
    }
    {
        const float d_x = *(const float*)(y_ds + (J0 + j_lane) * IU4_TILE_Y_K);
        const int s_x = *(const int*)(y_ds + (J0 + j_lane) * IU4_TILE_Y_K + 1);
        const int* acc_i = (const int*)&acc1;
        #pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int i = i0 + 1 * tile_C::I + tile_C::get_i(l);
            const half2 dm = ((const half2*)(x_qs + i * IU4_TILE_X_K + IU4_X_DM_OFF))[4 * h];
            const float2 dmA = __half22float2(dm);
            sums.v[8 + l] +=
                dmA.x * d_x * (float)acc_i[l] + dmA.y * d_x * (float)s_x;
        }
    }
    return sums;
}

static __device__ __forceinline__ void vec_dot_i4_x128_mc2(
    const int* __restrict__ x,
    const int* __restrict__ y,
    float* __restrict__ sum,
    int h
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

    // Four serialized copies (J0 = 0, 32, 64, 96): each block call takes its
    // 16 sums by value and returns them updated, so copies cannot overlap in
    // scheduling/allocation (at most one copy's chains+temps live + sum[]).
    iu4_mc2_sums sums = iu4_mc2_pack<0>(sum);
    sums = vec_dot_i4_x128_mc2_blk<0>(x_qs, y_qs, y_ds, h, i0, j_lane, lane_row, IU4_MC2_ACTUALS(sums));
    iu4_mc2_unpack<0>(sum, sums);
    sums = iu4_mc2_pack<32>(sum);
    sums = vec_dot_i4_x128_mc2_blk<32>(x_qs, y_qs, y_ds, h, i0, j_lane, lane_row, IU4_MC2_ACTUALS(sums));
    iu4_mc2_unpack<32>(sum, sums);
    sums = iu4_mc2_pack<64>(sum);
    sums = vec_dot_i4_x128_mc2_blk<64>(x_qs, y_qs, y_ds, h, i0, j_lane, lane_row, IU4_MC2_ACTUALS(sums));
    iu4_mc2_unpack<64>(sum, sums);
    sums = iu4_mc2_pack<96>(sum);
    sums = vec_dot_i4_x128_mc2_blk<96>(x_qs, y_qs, y_ds, h, i0, j_lane, lane_row, IU4_MC2_ACTUALS(sums));
    iu4_mc2_unpack<96>(sum, sums);
}

static __device__ __forceinline__ void vec_dot_i4_x128_mc4(
    const int* __restrict__ x,
    const int* __restrict__ y,
    float* __restrict__ sum,
    int h
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

    // unroll: same as MC=2 (constant jbase keeps sum[] in VGPRs; the sched
    // fence at the end of each unrolled iteration contains the four chains).
    #pragma unroll
    for (int jbase = 0; jbase < IU4_MMQ_X; jbase += 2 * ntx * tile_C::J) {
        int32x8_t c00 = {0, 0, 0, 0, 0, 0, 0, 0};
        int32x8_t c01 = {0, 0, 0, 0, 0, 0, 0, 0};
        int32x8_t c10 = {0, 0, 0, 0, 0, 0, 0, 0};
        int32x8_t c11 = {0, 0, 0, 0, 0, 0, 0, 0};
        #pragma nounroll
        for (int u = 0; u < 8; ++u) {
            const int t = u >> 1;
            const int s = u & 1;
            const int* ap0 = x_qs + (i0 + 0 * tile_C::I + lane_row) * IU4_TILE_X_K + h * 16 + t * 4 + s * 2;
            const int* ap1 = x_qs + (i0 + 1 * tile_C::I + lane_row) * IU4_TILE_X_K + h * 16 + t * 4 + s * 2;
            const int* bp0 = y_qs + (jbase + 0 * 32 + lane_row) * IU4_TILE_Y_K + t * 4 + s * 2;
            const int* bp1 = y_qs + (jbase + 1 * 32 + lane_row) * IU4_TILE_Y_K + t * 4 + s * 2;
            const int32x2_t av0 = *(const int32x2_t*)ap0;
            const int32x2_t av1 = *(const int32x2_t*)ap1;
            const int32x2_t bv0 = *(const int32x2_t*)bp0;
            const int32x2_t bv1 = *(const int32x2_t*)bp1;
            c00 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av0, true, bv0, c00, false);
            c01 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av1, true, bv0, c01, false);
            c10 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av0, true, bv1, c10, false);
            c11 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av1, true, bv1, c11, false);
        }
        {
            const int j0 = jbase + 0 * 32;
            const float d_x = *(const float*)(y_ds + (j0 + j_lane) * IU4_TILE_Y_K);
            const int s_x = *(const int*)(y_ds + (j0 + j_lane) * IU4_TILE_Y_K + 1);
            const int* acc_i = (const int*)&c00;
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + 0 * tile_C::I + tile_C::get_i(l);
                const half2 dm = ((const half2*)(x_qs + i * IU4_TILE_X_K + IU4_X_DM_OFF))[4 * h];
                const float2 dmA = __half22float2(dm);
                sum[(j0 / tile_C::J + 0) * tile_C::ne + l] +=
                    dmA.x * d_x * (float)acc_i[l] + dmA.y * d_x * (float)s_x;
            }
        }
        {
            const int j0 = jbase + 0 * 32;
            const float d_x = *(const float*)(y_ds + (j0 + j_lane) * IU4_TILE_Y_K);
            const int s_x = *(const int*)(y_ds + (j0 + j_lane) * IU4_TILE_Y_K + 1);
            const int* acc_i = (const int*)&c01;
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + 1 * tile_C::I + tile_C::get_i(l);
                const half2 dm = ((const half2*)(x_qs + i * IU4_TILE_X_K + IU4_X_DM_OFF))[4 * h];
                const float2 dmA = __half22float2(dm);
                sum[(j0 / tile_C::J + 1) * tile_C::ne + l] +=
                    dmA.x * d_x * (float)acc_i[l] + dmA.y * d_x * (float)s_x;
            }
        }
        {
            const int j0 = jbase + 1 * 32;
            const float d_x = *(const float*)(y_ds + (j0 + j_lane) * IU4_TILE_Y_K);
            const int s_x = *(const int*)(y_ds + (j0 + j_lane) * IU4_TILE_Y_K + 1);
            const int* acc_i = (const int*)&c10;
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + 0 * tile_C::I + tile_C::get_i(l);
                const half2 dm = ((const half2*)(x_qs + i * IU4_TILE_X_K + IU4_X_DM_OFF))[4 * h];
                const float2 dmA = __half22float2(dm);
                sum[(j0 / tile_C::J + 0) * tile_C::ne + l] +=
                    dmA.x * d_x * (float)acc_i[l] + dmA.y * d_x * (float)s_x;
            }
        }
        {
            const int j0 = jbase + 1 * 32;
            const float d_x = *(const float*)(y_ds + (j0 + j_lane) * IU4_TILE_Y_K);
            const int s_x = *(const int*)(y_ds + (j0 + j_lane) * IU4_TILE_Y_K + 1);
            const int* acc_i = (const int*)&c11;
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + 1 * tile_C::I + tile_C::get_i(l);
                const half2 dm = ((const half2*)(x_qs + i * IU4_TILE_X_K + IU4_X_DM_OFF))[4 * h];
                const float2 dmA = __half22float2(dm);
                sum[(j0 / tile_C::J + 1) * tile_C::ne + l] +=
                    dmA.x * d_x * (float)acc_i[l] + dmA.y * d_x * (float)s_x;
            }
        }
        // Scheduling fence only (no values, no order change): see MC=2.
        __builtin_amdgcn_sched_barrier(0);
    }
}

template <bool FULL, int MC = 1>
static __device__ __forceinline__ void gemm_iu4_body(
    const char* __restrict__ A,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    const int row0 = blockIdx.x * IU4_MMQ_Y;
    const int col0 = blockIdx.y * IU4_MMQ_X;
    if (!FULL && (row0 >= M || col0 >= N)) return;

    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;

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
        if constexpr (MC == 2) {
            vec_dot_i4_x128_mc2(tile_x, tile_y, sum, 0);
        } else if constexpr (MC == 4) {
            vec_dot_i4_x128_mc4(tile_x, tile_y, sum, 0);
        } else {
            vec_dot_i4_x128(tile_x, tile_y, sum, 0);
        }
        __syncthreads();

        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < IU4_MMQ_X * IU4_TILE_Y_K; l0 += IU4_MMQ_NWARPS * IU4_WARP_SIZE) {
                const int l = l0 + threadIdx.y * IU4_WARP_SIZE + threadIdx.x;
                if (FULL || col0 + l / IU4_TILE_Y_K < N) tile_y[l] = by1[l];
                else tile_y[l] = 0;
            }
            __syncthreads();
            if constexpr (MC == 2) {
                vec_dot_i4_x128_mc2(tile_x, tile_y, sum, 1);
            } else if constexpr (MC == 4) {
                vec_dot_i4_x128_mc4(tile_x, tile_y, sum, 1);
            } else {
                vec_dot_i4_x128(tile_x, tile_y, sum, 1);
            }
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

#define IU4_ENTRY(name, full, bounds_min) extern "C" __global__ __launch_bounds__(IU4_MMQ_NWARPS * IU4_WARP_SIZE, bounds_min) void name(     const char* __restrict__ A,     const block_i4_128* __restrict__ Xq,     float* __restrict__ Y,     int M,     int K,     int N,     int add ) {     gemm_iu4_body<full>(A, Xq, Y, M, K, N, add); }

IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4, false, 2)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_add, true, 2)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_set, true, 2)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3, true, 3)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3, true, 3)

#if defined(__gfx1151__)

// W3 candidates: shipping body with MC=2 / MC=4 WMMA issue (block [32,8,1],
// LDS 30720, same ABI/grid/wave map/fill/barriers/epilogue as the occ3
// entries above). Separate _mc2_/_mc4_ symbols so each can be gated.
#define IU4_MC_ENTRY(name, full, bounds_min, mc) extern "C" __global__ __launch_bounds__(IU4_MMQ_NWARPS * IU4_WARP_SIZE, bounds_min) void name(     const char* __restrict__ A,     const block_i4_128* __restrict__ Xq,     float* __restrict__ Y,     int M,     int K,     int N,     int add ) {     gemm_iu4_body<full, mc>(A, Xq, Y, M, K, N, add); }

IU4_MC_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_set_mc2_gfx1151, true, 3, 2)
IU4_MC_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_add_mc2_gfx1151, true, 3, 2)
IU4_MC_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_set_mc4_gfx1151, true, 3, 4)
IU4_MC_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_add_mc4_gfx1151, true, 3, 4)

#endif // __gfx1151__



#endif
