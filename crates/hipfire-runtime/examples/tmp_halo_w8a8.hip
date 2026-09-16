// SPDX-License-Identifier: MIT
// Throwaway Halo experiment: resident-int8 (pre-unpacked) twin of the IU4
// W4A4 prefill GEMM on gfx1151.
//
// Question: can a resident int8 weight copy (u8 [M][K] row-major + f16 header
// side array, ~8.25 bpw) beat the shipping IU4 GEMM on the gate prefill set
// (17408x5120x512), which rocprof shows 33-47% wave-cycles in s_waitcnt
// (fill-latency-bound)?
//
// Design (mirrors gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip occ3 structure):
//  - 128x128 tile, block [32,8,1] (8 wave32), full-tile only.
//  - K128 half-tiles per LDS stage: A plane 128 rows x 128 B (16 KB) +
//    X plane 128 cols x 128 B (16 KB) = 32768 B LDS.
//  - Fill = straight coalesced int copy of A u8 rows / X s8 columns into LDS
//    (no unpack, no header conversion in the fill; headers stay in global
//    side arrays and are loaded per row/column at fold time).
//  - __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32, unsigned A (0..15 codes),
//    signed B (-8..7 codes) — the F5.1 probe's verified map:
//    A[lane][byte] -> (row = lane&15, k = byte),
//    B[lane][byte] -> (k = byte, col = lane&15),
//    acc[lane][j] -> C[2j+(lane>>4)][lane&15] (same C map as iu4).
//  - Same per-output integer accumulation per 128-half (8 chained iu8 WMMAs,
//    same t order as the iu4 pairs) and the pinned IU4_FOLD_RN fold, so
//    outputs are bit-exact with the shipping kernel: same integer sums
//    (integer add is exact regardless of chain order), same fold DAG.
//  - `_pf` variant pipelines the A-side fragments one t-iteration ahead
//    (next-32K A bytes register-prefetched before the WMMA pair that
//    consumes the current 32K). Measured gfx1151: base 184 VGPR / 8 waves,
//    pf 198 VGPR / 7 waves, both 0 spills.
//
// Host layout contract (built on the host from MQ4G256V2 + block_i4_128):
//  - A:   u8[M][K] row-major codes 0..15.
//  - Ah:  half2[M][K/128]; entry (row, b) = (sc, zp) f16 pair of 128-half b
//         (raw copy of MQ4 group header bytes 4*h..4*h+3).
//  - X:   s8[N][K] row-major codes -8..7 (bit patterns in bytes).
//  - Xh:  int2[K/128][N] per-128-block header (x = d bits, y = s, both
//         verbatim from block_i4_128; one 8 B load per column per half).
//  - Y:   float, Y[col*M + row] (same column-major tile layout as shipping).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

#if defined(__gfx1151__)
#define W8A8_RESIDENT 1
#else
#define W8A8_RESIDENT 0
#endif

#if !W8A8_RESIDENT

extern "C" __global__ void gemm_w8a8_resident_gfx1151(
    const unsigned char*, const half2*, const signed char*, const int2*,
    float*, int, int, int, int) {}
extern "C" __global__ void gemm_w8a8_resident_add_gfx1151(
    const unsigned char*, const half2*, const signed char*, const int2*,
    float*, int, int, int, int) {}
extern "C" __global__ void gemm_w8a8_resident_pf_gfx1151(
    const unsigned char*, const half2*, const signed char*, const int2*,
    float*, int, int, int, int) {}
extern "C" __global__ void gemm_w8a8_resident_pf_add_gfx1151(
    const unsigned char*, const half2*, const signed char*, const int2*,
    float*, int, int, int, int) {}

#else

enum w8_data_layout {
    W8_DATA_LAYOUT_I_MAJOR = 0,
    W8_DATA_LAYOUT_J_MAJOR = 10,
};

template <int I_, int J_, typename T, w8_data_layout dl = W8_DATA_LAYOUT_I_MAJOR>
struct w8_tile {};

template <int I_, int J_, typename T>
struct w8_tile<I_, J_, T, W8_DATA_LAYOUT_I_MAJOR> {
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
struct w8_tile<I_, J_, T, W8_DATA_LAYOUT_J_MAJOR> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = w8_tile<I_, J_, T, W8_DATA_LAYOUT_I_MAJOR>::ne;
    T x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int l) {
        return w8_tile<I_, J_, T, W8_DATA_LAYOUT_I_MAJOR>::get_j(l);
    }

    static __device__ __forceinline__ int get_j(const int l) {
        return w8_tile<I_, J_, T, W8_DATA_LAYOUT_I_MAJOR>::get_i(l);
    }
};

using w8_int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using w8_int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

// IU4_FOLD_RN — W4.0/LF.0 bit-exact fold pin, copied verbatim from
// kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip so w8a8 outputs are
// bit-exact with the shipping kernel (same integer sums, same fold DAG):
//   t1   = RN(sc * d)            v_fma_mix_f32 (header.lo x d + (-0.0))
//   p    = RN(t1 * float(C))     v_mul_f32 (dual-issued on gfx1151)
//   t2   = RN(zp * d)            v_fma_mix_f32 (header.hi x d + (-0.0))
//   term = RN(t2 * float(s) + p) v_fmac_f32 (the zp term is the FMA product,
//                                          NOT the running sum)
//   sum  = RN(sum + term)        v_add_f32 (separate add, never fused)
#define IU4_FOLD_RN(sum_lvalue, sc, zp, d, acc_i32, s_i32) \
    do { \
        const float iu4_fold_d_ = (d); \
        const float iu4_fold_t1_ = __fmul_rn((sc), iu4_fold_d_); \
        const float iu4_fold_p_ = __fmul_rn(iu4_fold_t1_, (float)(acc_i32)); \
        const float iu4_fold_t2_ = __fmul_rn((zp), iu4_fold_d_); \
        const float iu4_fold_term_ = __fmaf_rn(iu4_fold_t2_, (float)(s_i32), iu4_fold_p_); \
        (sum_lvalue) = __fadd_rn((sum_lvalue), iu4_fold_term_); \
    } while (0)

#define W8A8_MMQ_X 128
#define W8A8_MMQ_Y 128
#define W8A8_MMQ_NWARPS 8
#define W8A8_WARP_SIZE 32
#define W8A8_ROW_INTS 32
#define W8A8_COL_INTS 32
#define W8A8_PLANE_INTS (128 * 32)

// Straight coalesced copy of one K128 half (128 lines x 128 B) into LDS.
// src base already includes the tile origin and (kb*256 + h*128) K offset;
// rows advance with stride K. No unpack, no header work in the fill.
static __device__ __forceinline__ void w8a8_fill_half(
    const unsigned char* __restrict__ src,
    int* __restrict__ dst,
    int K
) {
    const int tid = threadIdx.y * W8A8_WARP_SIZE + threadIdx.x;
    #pragma unroll 1
    for (int l0 = 0; l0 < W8A8_PLANE_INTS; l0 += W8A8_MMQ_NWARPS * W8A8_WARP_SIZE) {
        const int l = l0 + tid;
        const int r = l >> 5;
        const int cw = l & 31;
        dst[l] = *(const int*)(src + (long long)r * K + (long long)cw * 4);
    }
}

// i8 dot over ONE 128-K half: 8 chained iu8 WMMAs (t = 0..3, low/high 16K in
// shipping pair order) into one i32 C per (j0, n), then the pinned fold with
// (d_x, s_x) from the X side array and (sc, zp) from the A side array.
// Xh packs the per-128-block X header as int2 (x = d bits, y = s) so each
// column needs one 8 B header load instead of two address chains.
static __device__ __forceinline__ void w8a8_dot_half(
    const int* __restrict__ a,
    const int* __restrict__ x,
    const half2* __restrict__ Ah,
    const int2* __restrict__ Xh,
    float* __restrict__ sum,
    int row0,
    int col0,
    int b,
    int N,
    int blocks128
) {
    using tile_C = w8_tile<16, 16, int, W8_DATA_LAYOUT_J_MAJOR>;

    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;

    const int cb = (threadIdx.y % ntx) * tile_C::J;
    x += cb * W8A8_COL_INTS;

    const int* aq = a;
    const int* xq = x;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
    const int j_lane = tile_C::get_j(0);
    const int lane_row = threadIdx.x % 16;

    // Hoisted header bases: keeps row0/col0/b/N/blocks128 out of the fold
    // loop; one 64-bit base each stays live across (j0, n).
    const long long xh_base = (long long)b * N;
    const int lane_bcol = col0 + cb + j_lane;
    const long long ah_lane = (long long)(row0 + i0 + (threadIdx.x / 16)) * blocks128 + b;

    for (int j0 = 0; j0 < W8A8_MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            w8_int32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
            #pragma nounroll
            for (int t = 0; t < 4; ++t) {
                const int* ap = aq + (i0 + n * tile_C::I + lane_row) * W8A8_ROW_INTS + t * 8;
                const int* bp = xq + (j0 + lane_row) * W8A8_COL_INTS + t * 8;
                const w8_int32x4_t av0 = *(const w8_int32x4_t*)(ap + 0);
                const w8_int32x4_t av1 = *(const w8_int32x4_t*)(ap + 4);
                const w8_int32x4_t bv0 = *(const w8_int32x4_t*)(bp + 0);
                const w8_int32x4_t bv1 = *(const w8_int32x4_t*)(bp + 4);
                acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, av0, true, bv0, acc, false);
                acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, av1, true, bv1, acc, false);
            }
            const int2 ds = Xh[xh_base + lane_bcol + j0];
            const float d_x = __int_as_float(ds.x);
            const int s_x = ds.y;
            const long long ah_n = ah_lane + (long long)(n * 16) * blocks128;
            const int* acc_i = (const int*)&acc;
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const half2 dm = Ah[ah_n + (long long)(2 * l) * blocks128];
                const float2 dmA = __half22float2(dm);
                IU4_FOLD_RN(sum[(j0 / tile_C::J + n) * tile_C::ne + l],
                            dmA.x, dmA.y, d_x, acc_i[l], s_x);
            }
        }
    }
}

// PF twin: identical schedule, but the next 32K of A bytes are
// register-prefetched (second live A fragment pair) ahead of the WMMA pair
// that consumes the current 32K. X side unchanged.
static __device__ __forceinline__ void w8a8_dot_half_pf(
    const int* __restrict__ a,
    const int* __restrict__ x,
    const half2* __restrict__ Ah,
    const int2* __restrict__ Xh,
    float* __restrict__ sum,
    int row0,
    int col0,
    int b,
    int N,
    int blocks128
) {
    using tile_C = w8_tile<16, 16, int, W8_DATA_LAYOUT_J_MAJOR>;

    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;

    const int cb = (threadIdx.y % ntx) * tile_C::J;
    x += cb * W8A8_COL_INTS;

    const int* aq = a;
    const int* xq = x;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
    const int j_lane = tile_C::get_j(0);
    const int lane_row = threadIdx.x % 16;

    // Same hoisted header bases as the base dot.
    const long long xh_base = (long long)b * N;
    const int lane_bcol = col0 + cb + j_lane;
    const long long ah_lane = (long long)(row0 + i0 + (threadIdx.x / 16)) * blocks128 + b;


    for (int j0 = 0; j0 < W8A8_MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            w8_int32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
            const int* ap0 = aq + (i0 + n * tile_C::I + lane_row) * W8A8_ROW_INTS;
            const int* bp0 = xq + (j0 + lane_row) * W8A8_COL_INTS;
            w8_int32x4_t av0 = *(const w8_int32x4_t*)(ap0 + 0);
            w8_int32x4_t av1 = *(const w8_int32x4_t*)(ap0 + 4);
            // Rolled t loop with a predicated next-32K prefetch: the next A
            // pair is loaded before the WMMA pair that consumes the current
            // 32K, so its LDS latency overlaps WMMA execution. (An unrolled
            // immediate-offset form measured worse: 203 VGPR vs 198 here.)
            #pragma nounroll
            for (int t = 0; t < 4; ++t) {
                w8_int32x4_t nv0, nv1;
                if (t + 1 < 4) {
                    nv0 = *(const w8_int32x4_t*)(ap0 + (t + 1) * 8 + 0);
                    nv1 = *(const w8_int32x4_t*)(ap0 + (t + 1) * 8 + 4);
                } else {
                    nv0 = (w8_int32x4_t){0, 0, 0, 0};
                    nv1 = (w8_int32x4_t){0, 0, 0, 0};
                }
                const w8_int32x4_t bv0 = *(const w8_int32x4_t*)(bp0 + t * 8 + 0);
                const w8_int32x4_t bv1 = *(const w8_int32x4_t*)(bp0 + t * 8 + 4);
                acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, av0, true, bv0, acc, false);
                acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, av1, true, bv1, acc, false);
                av0 = nv0;
                av1 = nv1;
            }
            const int2 ds = Xh[xh_base + lane_bcol + j0];
            const float d_x = __int_as_float(ds.x);
            const int s_x = ds.y;
            const long long ah_n = ah_lane + (long long)(n * 16) * blocks128;
            const int* acc_i = (const int*)&acc;
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const half2 dm = Ah[ah_n + (long long)(2 * l) * blocks128];
                const float2 dmA = __half22float2(dm);
                IU4_FOLD_RN(sum[(j0 / tile_C::J + n) * tile_C::ne + l],
                            dmA.x, dmA.y, d_x, acc_i[l], s_x);
            }
        }
    }
}

template <bool IS_ADD, bool PF>
static __device__ __forceinline__ void w8a8_body(
    const unsigned char* __restrict__ A,
    const half2* __restrict__ Ah,
    const signed char* __restrict__ X,
    const int2* __restrict__ Xh,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int row_tile,
    int col_tile
) {
    const int row0 = row_tile * W8A8_MMQ_Y;
    const int col0 = col_tile * W8A8_MMQ_X;
    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;

    extern __shared__ int smem[];
    int* tile_x = smem;
    int* tile_a = tile_x + W8A8_PLANE_INTS;

    float sum[W8A8_MMQ_X * W8A8_MMQ_Y / (W8A8_MMQ_NWARPS * W8A8_WARP_SIZE)] = {0.0f};

    for (int kb = 0; kb < groups_per_row; ++kb) {
        #pragma unroll 1
        for (int h = 0; h < 2; ++h) {
            const int b = 2 * kb + h;
            w8a8_fill_half(A + (long long)row0 * K + (long long)kb * 256 + h * 128,
                           tile_a, K);
            w8a8_fill_half((const unsigned char*)(X + (long long)col0 * K + (long long)kb * 256 + h * 128),
                           tile_x, K);
            __syncthreads();
            if constexpr (PF) {
                w8a8_dot_half_pf(tile_a, tile_x, Ah, Xh, sum, row0, col0, b, N, blocks128);
            } else {
                w8a8_dot_half(tile_a, tile_x, Ah, Xh, sum, row0, col0, b, N, blocks128);
            }
            __syncthreads();
        }
    }

    using tile_C = w8_tile<16, 16, int, W8_DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    #pragma unroll
    for (int j0 = 0; j0 < W8A8_MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                const float v = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
                if constexpr (IS_ADD) {
                    *yp += v;
                } else {
                    *yp = v;
                }
            }
        }
    }
}

#define W8A8_ENTRY(name, is_add, pf) \
    extern "C" __global__ __launch_bounds__(W8A8_MMQ_NWARPS * W8A8_WARP_SIZE, 2) void name( \
        const unsigned char* __restrict__ A, \
        const half2* __restrict__ Ah, \
        const signed char* __restrict__ X, \
        const int2* __restrict__ Xh, \
        float* __restrict__ Y, \
        int M, int K, int N, int add) { \
        (void)add; \
        w8a8_body<is_add, pf>(A, Ah, X, Xh, Y, M, K, N, blockIdx.y, blockIdx.x); \
    }

W8A8_ENTRY(gemm_w8a8_resident_gfx1151, false, false)
W8A8_ENTRY(gemm_w8a8_resident_add_gfx1151, true, false)
W8A8_ENTRY(gemm_w8a8_resident_pf_gfx1151, false, true)
W8A8_ENTRY(gemm_w8a8_resident_pf_add_gfx1151, true, true)

#endif
