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

extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_add(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_set(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_add_pf_gfx11(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_set_pf_gfx11(
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

template <bool FULL>
static __device__ __forceinline__ void gemm_iu4_body(
    const char* __restrict__ A,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add,
    int row_tile,
    int col_tile
) {
    const int row0 = row_tile * IU4_MMQ_Y;
    const int col0 = col_tile * IU4_MMQ_X;
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
        vec_dot_i4_x128(tile_x, tile_y, sum, 0);
        __syncthreads();

        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < IU4_MMQ_X * IU4_TILE_Y_K; l0 += IU4_MMQ_NWARPS * IU4_WARP_SIZE) {
                const int l = l0 + threadIdx.y * IU4_WARP_SIZE + threadIdx.x;
                if (FULL || col0 + l / IU4_TILE_Y_K < N) tile_y[l] = by1[l];
                else tile_y[l] = 0;
            }
            __syncthreads();
            vec_dot_i4_x128(tile_x, tile_y, sum, 1);
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
// W5: in-register software-prefetch sister of gemm_iu4_body (FULL path only
// in practice; template kept for ABI symmetry). Same grid/block/LDS/wave
// map, same 4 barriers/group, same LDS planes, same dot/fold/epilogue text.
// Only the GLOBAL load issue point moves earlier: right after the barrier
// that publishes half-0, PF_W dwords/lane of the next Xq half are loaded
// into registers; the dot/fold then runs on current LDS data; those
// registers are stored to LDS in the existing fill slot (remaining 9-PF_W
// dwords still load from global there). Compiler loop on gfx1151/gfx1100:
// PF_W=9 → 198 VGPR, PF_W=2 → 193, PF_W=1 → 192/0/0 (occupancy 8). A-side
// stays direct globals->LDS (no budget for A prefetch). 
// __builtin_amdgcn_sched_barrier(0) keeps the prefetch loads above the
// WMMA loop; no asm memory clobbers.
template <bool FULL>
static __device__ __forceinline__ void gemm_iu4_body_pf(
    const char* __restrict__ A,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add,
    int row_tile,
    int col_tile
) {
    const int row0 = row_tile * IU4_MMQ_Y;
    const int col0 = col_tile * IU4_MMQ_X;
    if (!FULL && (row0 >= M || col0 >= N)) return;

    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;

    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + IU4_MMQ_X * IU4_TILE_Y_K;

    float sum[IU4_MMQ_X * IU4_MMQ_Y / (IU4_MMQ_NWARPS * IU4_WARP_SIZE)] = {0.0f};

    constexpr int PW = IU4_MMQ_NWARPS * IU4_WARP_SIZE;
    constexpr int PF_W = 1;
    int pf_x[PF_W];
    for (int kb = 0; kb < groups_per_row; ++kb) {
        load_iu4_tile<FULL>(A, tile_x, row0, kb, M, groups_per_row);
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l0 = 0; l0 < IU4_MMQ_X * IU4_TILE_Y_K; l0 += IU4_MMQ_NWARPS * IU4_WARP_SIZE) {
            const int l = l0 + threadIdx.y * IU4_WARP_SIZE + threadIdx.x;
            if (FULL || col0 + l / IU4_TILE_Y_K < N) tile_y[l] = by0[l];
            else tile_y[l] = 0;
        }
        __syncthreads();
        const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
        #pragma unroll
        for (int u = 0; u < PF_W; ++u) {
            const int l = u * PW + (threadIdx.y * IU4_WARP_SIZE + threadIdx.x);
            if (FULL || col0 + l / IU4_TILE_Y_K < N) pf_x[u] = by1[l];
            else pf_x[u] = 0;
        }
        __builtin_amdgcn_sched_barrier(0);
        vec_dot_i4_x128(tile_x, tile_y, sum, 0);
        __syncthreads();

        if (2 * kb + 1 < blocks128) {
            #pragma unroll
            for (int u = 0; u < PF_W; ++u) {
                const int l = u * PW + (threadIdx.y * IU4_WARP_SIZE + threadIdx.x);
                tile_y[l] = pf_x[u];
            }
            for (int u = PF_W; u < 9; ++u) {
                const int l = u * PW + (threadIdx.y * IU4_WARP_SIZE + threadIdx.x);
                if (FULL || col0 + l / IU4_TILE_Y_K < N) tile_y[l] = by1[l];
                else tile_y[l] = 0;
            }
            __syncthreads();
            vec_dot_i4_x128(tile_x, tile_y, sum, 1);
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

#define IU4_ENTRY(name, full, bounds_min) extern "C" __global__ __launch_bounds__(IU4_MMQ_NWARPS * IU4_WARP_SIZE, bounds_min) void name(     const char* __restrict__ A,     const block_i4_128* __restrict__ Xq,     float* __restrict__ Y,     int M,     int K,     int N,     int add ) {     gemm_iu4_body<full>(A, Xq, Y, M, K, N, add, blockIdx.x, blockIdx.y); }
#define IU4_ENTRY_COL(name, full, bounds_min) extern "C" __global__ __launch_bounds__(IU4_MMQ_NWARPS * IU4_WARP_SIZE, bounds_min) void name(     const char* __restrict__ A,     const block_i4_128* __restrict__ Xq,     float* __restrict__ Y,     int M,     int K,     int N,     int add ) {     gemm_iu4_body<full>(A, Xq, Y, M, K, N, add, blockIdx.y, blockIdx.x); }
#define IU4_ENTRY_PF_COL(name, full, bounds_min) extern "C" __global__ __launch_bounds__(IU4_MMQ_NWARPS * IU4_WARP_SIZE, bounds_min) void name(     const char* __restrict__ A,     const block_i4_128* __restrict__ Xq,     float* __restrict__ Y,     int M,     int K,     int N,     int add ) {     gemm_iu4_body_pf<full>(A, Xq, Y, M, K, N, add, blockIdx.y, blockIdx.x); }

IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4, false, 2)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_add, true, 2)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_set, true, 2)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3, true, 3)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3, true, 3)
#if defined(__gfx1151__) || defined(__gfx1100__)
IU4_ENTRY_COL(gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151, true, 3)
IU4_ENTRY_COL(gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151, true, 3)
IU4_ENTRY_PF_COL(gemm_mq4g256v2_residual_mmq_iu4_full_add_pf_gfx11, true, 3)
IU4_ENTRY_PF_COL(gemm_mq4g256v2_residual_mmq_iu4_full_set_pf_gfx11, true, 3)
#endif

#endif
