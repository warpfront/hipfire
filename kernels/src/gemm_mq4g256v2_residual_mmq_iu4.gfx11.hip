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

template <bool FULL>
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

#define IU4_ENTRY(name, full, bounds_min) extern "C" __global__ __launch_bounds__(IU4_MMQ_NWARPS * IU4_WARP_SIZE, bounds_min) void name(     const char* __restrict__ A,     const block_i4_128* __restrict__ Xq,     float* __restrict__ Y,     int M,     int K,     int N,     int add ) {     gemm_iu4_body<full>(A, Xq, Y, M, K, N, add); }

IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4, false, 2)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_add, true, 2)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_set, true, 2)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3, true, 3)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3, true, 3)

#if defined(__gfx1151__)

// WS4 W2: 8 compute + 2 producer waves; four independent output chains/wave;
// two A planes + two Xq-half planes; 61440 B LDS; block [32,10,1].
// Waves 0..7 compute (shipping i0/col_off, sum[64]); waves 8..9 producers.
// All ten waves rejoin every workgroup barrier. No static __shared__.

static __device__ __forceinline__ void load_iu4_stage_ws4_w2_gfx1151(
    const char* __restrict__ A,
    const int* __restrict__ Xq_base,
    int* __restrict__ Aplane,
    int* __restrict__ Yplane,
    int row0,
    int g,
    int b,
    int groups_per_row,
    bool do_A,
    bool do_Xq
) {
    const int e = threadIdx.y - 8;
    const int lane = threadIdx.x;

    // Packetize each phase: load ≤16 dwords/lane → store to LDS, then fence so
    // the compiler cannot hoist the next packet's loads over this store.
    // Phases (exact LDS addresses unchanged):
    //   do_A+do_Xq: A16 + A16 + Xq16 + Xq16 + (Xq4+hdr2)   = 16,16,16,16,6
    //   do_A only:  A16 + A16 + hdr2                        = 16,16,2
    //   do_Xq only: Xq16 + Xq16 + Xq4                       = 16,16,4
    // Headers (2 dwords) always ride in the last packet of a do_A phase.
    // Fence: asm volatile("" ::: "memory") — same idiom as other gfx11 kernels;
    // __builtin_amdgcn_sched_barrier has no memory clobber in this tree.

#define IU4_WS4_A_CODE_PKT(q) do { \
        const int r_ = 64 * (b) + 2 * (q) + e; \
        const char* gp_ = A + ((long long)(row0 + r_) * groups_per_row + g) * 136; \
        Aplane[r_ * IU4_TILE_X_K + lane] = *(const int*)(gp_ + 8 + 4 * lane); \
    } while (0)
#define IU4_WS4_XQ_PKT(q) do { \
        const int l_ = 64 * (q) + 32 * e + lane; \
        Yplane[l_] = Xq_base[l_]; \
    } while (0)
#define IU4_WS4_HDR_PKT() do { \
        const int r_ = 64 * (b) + 32 * e + lane; \
        const char* gp_ = A + ((long long)(row0 + r_) * groups_per_row + g) * 136; \
        const int h0_ = *(const int*)(gp_ + 0); \
        const int h1_ = *(const int*)(gp_ + 4); \
        half2* x_dm_ = (half2*)(Aplane + IU4_X_DM_OFF); \
        { \
            const unsigned int hs = (unsigned int)h0_; \
            const float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu))); \
            const float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16))); \
            const half2 dm = make_half2((_Float16)sc, (_Float16)zp); \
            x_dm_[r_ * IU4_TILE_X_K + 4 * 0 + 0] = dm; \
            x_dm_[r_ * IU4_TILE_X_K + 4 * 0 + 1] = dm; \
            x_dm_[r_ * IU4_TILE_X_K + 4 * 0 + 2] = dm; \
            x_dm_[r_ * IU4_TILE_X_K + 4 * 0 + 3] = dm; \
        } \
        { \
            const unsigned int hs = (unsigned int)h1_; \
            const float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu))); \
            const float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16))); \
            const half2 dm = make_half2((_Float16)sc, (_Float16)zp); \
            x_dm_[r_ * IU4_TILE_X_K + 4 * 1 + 0] = dm; \
            x_dm_[r_ * IU4_TILE_X_K + 4 * 1 + 1] = dm; \
            x_dm_[r_ * IU4_TILE_X_K + 4 * 1 + 2] = dm; \
            x_dm_[r_ * IU4_TILE_X_K + 4 * 1 + 3] = dm; \
        } \
    } while (0)
#define IU4_WS4_PKT_FENCE() asm volatile("" ::: "memory")

    if (do_A) {
        // Packet A0: 16 A-code dwords (q 0..15).
        IU4_WS4_A_CODE_PKT(0);  IU4_WS4_A_CODE_PKT(1);
        IU4_WS4_A_CODE_PKT(2);  IU4_WS4_A_CODE_PKT(3);
        IU4_WS4_A_CODE_PKT(4);  IU4_WS4_A_CODE_PKT(5);
        IU4_WS4_A_CODE_PKT(6);  IU4_WS4_A_CODE_PKT(7);
        IU4_WS4_A_CODE_PKT(8);  IU4_WS4_A_CODE_PKT(9);
        IU4_WS4_A_CODE_PKT(10); IU4_WS4_A_CODE_PKT(11);
        IU4_WS4_A_CODE_PKT(12); IU4_WS4_A_CODE_PKT(13);
        IU4_WS4_A_CODE_PKT(14); IU4_WS4_A_CODE_PKT(15);
        IU4_WS4_PKT_FENCE();

        // Packet A1: 16 A-code dwords (q 16..31).
        IU4_WS4_A_CODE_PKT(16); IU4_WS4_A_CODE_PKT(17);
        IU4_WS4_A_CODE_PKT(18); IU4_WS4_A_CODE_PKT(19);
        IU4_WS4_A_CODE_PKT(20); IU4_WS4_A_CODE_PKT(21);
        IU4_WS4_A_CODE_PKT(22); IU4_WS4_A_CODE_PKT(23);
        IU4_WS4_A_CODE_PKT(24); IU4_WS4_A_CODE_PKT(25);
        IU4_WS4_A_CODE_PKT(26); IU4_WS4_A_CODE_PKT(27);
        IU4_WS4_A_CODE_PKT(28); IU4_WS4_A_CODE_PKT(29);
        IU4_WS4_A_CODE_PKT(30); IU4_WS4_A_CODE_PKT(31);

        if (do_Xq) {
            IU4_WS4_PKT_FENCE();
            // Packet X0: 16 Xq dwords (q 0..15).
            IU4_WS4_XQ_PKT(0);  IU4_WS4_XQ_PKT(1);
            IU4_WS4_XQ_PKT(2);  IU4_WS4_XQ_PKT(3);
            IU4_WS4_XQ_PKT(4);  IU4_WS4_XQ_PKT(5);
            IU4_WS4_XQ_PKT(6);  IU4_WS4_XQ_PKT(7);
            IU4_WS4_XQ_PKT(8);  IU4_WS4_XQ_PKT(9);
            IU4_WS4_XQ_PKT(10); IU4_WS4_XQ_PKT(11);
            IU4_WS4_XQ_PKT(12); IU4_WS4_XQ_PKT(13);
            IU4_WS4_XQ_PKT(14); IU4_WS4_XQ_PKT(15);
            IU4_WS4_PKT_FENCE();

            // Packet X1: 16 Xq dwords (q 16..31).
            IU4_WS4_XQ_PKT(16); IU4_WS4_XQ_PKT(17);
            IU4_WS4_XQ_PKT(18); IU4_WS4_XQ_PKT(19);
            IU4_WS4_XQ_PKT(20); IU4_WS4_XQ_PKT(21);
            IU4_WS4_XQ_PKT(22); IU4_WS4_XQ_PKT(23);
            IU4_WS4_XQ_PKT(24); IU4_WS4_XQ_PKT(25);
            IU4_WS4_XQ_PKT(26); IU4_WS4_XQ_PKT(27);
            IU4_WS4_XQ_PKT(28); IU4_WS4_XQ_PKT(29);
            IU4_WS4_XQ_PKT(30); IU4_WS4_XQ_PKT(31);
            IU4_WS4_PKT_FENCE();

            // Last packet: Xq q32..35 (4) + headers (2) = 6 dwords.
            IU4_WS4_XQ_PKT(32); IU4_WS4_XQ_PKT(33);
            IU4_WS4_XQ_PKT(34); IU4_WS4_XQ_PKT(35);
            IU4_WS4_HDR_PKT();
        } else {
            IU4_WS4_PKT_FENCE();
            // Last packet: headers only (2 dwords).
            IU4_WS4_HDR_PKT();
        }
    } else if (do_Xq) {
        // Xq-only phase (no A): 16 + 16 + 4.
        IU4_WS4_XQ_PKT(0);  IU4_WS4_XQ_PKT(1);
        IU4_WS4_XQ_PKT(2);  IU4_WS4_XQ_PKT(3);
        IU4_WS4_XQ_PKT(4);  IU4_WS4_XQ_PKT(5);
        IU4_WS4_XQ_PKT(6);  IU4_WS4_XQ_PKT(7);
        IU4_WS4_XQ_PKT(8);  IU4_WS4_XQ_PKT(9);
        IU4_WS4_XQ_PKT(10); IU4_WS4_XQ_PKT(11);
        IU4_WS4_XQ_PKT(12); IU4_WS4_XQ_PKT(13);
        IU4_WS4_XQ_PKT(14); IU4_WS4_XQ_PKT(15);
        IU4_WS4_PKT_FENCE();

        IU4_WS4_XQ_PKT(16); IU4_WS4_XQ_PKT(17);
        IU4_WS4_XQ_PKT(18); IU4_WS4_XQ_PKT(19);
        IU4_WS4_XQ_PKT(20); IU4_WS4_XQ_PKT(21);
        IU4_WS4_XQ_PKT(22); IU4_WS4_XQ_PKT(23);
        IU4_WS4_XQ_PKT(24); IU4_WS4_XQ_PKT(25);
        IU4_WS4_XQ_PKT(26); IU4_WS4_XQ_PKT(27);
        IU4_WS4_XQ_PKT(28); IU4_WS4_XQ_PKT(29);
        IU4_WS4_XQ_PKT(30); IU4_WS4_XQ_PKT(31);
        IU4_WS4_PKT_FENCE();

        IU4_WS4_XQ_PKT(32); IU4_WS4_XQ_PKT(33);
        IU4_WS4_XQ_PKT(34); IU4_WS4_XQ_PKT(35);
    }

#undef IU4_WS4_PKT_FENCE
#undef IU4_WS4_HDR_PKT
#undef IU4_WS4_XQ_PKT
#undef IU4_WS4_A_CODE_PKT
}

// Four independent int32x8 chains per jbase; shared A/B b64 fragments;
// u=0..7 issue (c0,n0),(c0,n1),(c1,n0),(c1,n1). Shipping wave map + sum[64].
static __device__ __forceinline__ void vec_dot_i4_x128_ws4_w2_gfx1151(
    const int* __restrict__ x,
    const int* __restrict__ y,
    float* __restrict__ sum,
    int h
) {
    const int w = threadIdx.y;
    const int lane = threadIdx.x;
    const int lane_row = lane % 16;
    const int i0 = 32 * (w / 2);
    const int col_off = 16 * (w % 2);
    const int j_lane = lane % 16;

    const int* x_qs = x;
    const int* y_qs = y + 2;
    const int* y_ds = y;

    #pragma unroll
    for (int jbase = 0; jbase < IU4_MMQ_X; jbase += 64) {
        int32x8_t c00 = {0, 0, 0, 0, 0, 0, 0, 0};
        int32x8_t c01 = {0, 0, 0, 0, 0, 0, 0, 0};
        int32x8_t c10 = {0, 0, 0, 0, 0, 0, 0, 0};
        int32x8_t c11 = {0, 0, 0, 0, 0, 0, 0, 0};

        #pragma nounroll
        for (int u = 0; u < 8; ++u) {
            const int* ap0 = x_qs + (i0 + 0 * 16 + lane_row) * IU4_TILE_X_K + h * 16 + 2 * u;
            const int* ap1 = x_qs + (i0 + 1 * 16 + lane_row) * IU4_TILE_X_K + h * 16 + 2 * u;
            const int* bp0 = y_qs + (jbase + 0 * 32 + col_off + lane_row) * IU4_TILE_Y_K + 2 * u;
            const int* bp1 = y_qs + (jbase + 1 * 32 + col_off + lane_row) * IU4_TILE_Y_K + 2 * u;
            const int32x2_t av0 = *(const int32x2_t*)ap0;
            const int32x2_t av1 = *(const int32x2_t*)ap1;
            const int32x2_t bv0 = *(const int32x2_t*)bp0;
            const int32x2_t bv1 = *(const int32x2_t*)bp1;
            // (c0,n0), (c0,n1), (c1,n0), (c1,n1)
            c00 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av0, true, bv0, c00, false);
            c01 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av1, true, bv0, c01, false);
            c10 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av0, true, bv1, c10, false);
            c11 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av1, true, bv1, c11, false);
        }

#define IU4_WS4_FOLD(c_var, c_idx, n_idx) do { \
            const int j = jbase + 32 * (c_idx); \
            const float d_x = *(const float*)(y_ds + (j + col_off + j_lane) * IU4_TILE_Y_K); \
            const int s_x = *(const int*)(y_ds + (j + col_off + j_lane) * IU4_TILE_Y_K + 1); \
            const int* acc_i = (const int*)&(c_var); \
            _Pragma("unroll") \
            for (int l = 0; l < 8; ++l) { \
                const int i = i0 + (n_idx) * 16 + 2 * l + (lane / 16); \
                const half2 dm = ((const half2*)(x_qs + i * IU4_TILE_X_K + IU4_X_DM_OFF))[4 * h]; \
                const float2 dmA = __half22float2(dm); \
                sum[((j / 16) + (n_idx)) * 8 + l] += \
                    dmA.x * d_x * (float)acc_i[l] + dmA.y * d_x * (float)s_x; \
            } \
        } while (0)

        // fold order (c0,n0),(c0,n1),(c1,n0),(c1,n1)
        IU4_WS4_FOLD(c00, 0, 0);
        IU4_WS4_FOLD(c01, 0, 1);
        IU4_WS4_FOLD(c10, 1, 0);
        IU4_WS4_FOLD(c11, 1, 1);
#undef IU4_WS4_FOLD
    }
}

// dbg_mode: 0=prod, 1=noprod (waves 8..9 skip global loads/LDS stores; barriers
// only), 2=nocomp (compute skips vec_dot/fold; barriers + epilogue store of sum).
// W2.P3: wave-uniform role split (threadIdx.y; block [32,10,1]). sum[64] and
// C/fragment state live only on the compute branch — producers never see them.
// Barrier pairing (same COUNT, same order on all ten waves):
//   barrier | compute (wy < 8)                         | producer (wy >= 8)
//   --------+------------------------------------------+---------------------------
//   Binit   | wait (no work before first fill)         | prologue A0 both + Y0
//   B0(g)   | h0 vec_dot(A_cur, Y0, h=0)               | A_nxt top64 if next + Y1
//   B1(g)   | h1 vec_dot(A_cur, Y1, h=1)               | A_nxt bot64 + Y0 nxt if next
// Template so dbg_mode=0 codegen matches production entries bit-for-bit aside
// from the constexpr folds eliding nothing.
template <int dbg_mode = 0>
static __device__ __forceinline__ void gemm_iu4_body_ws4_w2_gfx1151(
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
    const int groups_per_row = K / 256;
    const int G = groups_per_row;
    const int wy = threadIdx.y;
    const int lane = threadIdx.x;

    // Dynamic LDS 61440 B: Y0, Y1, A0, A1 (§5.3).
    extern __shared__ int smem[];
    int* Y0 = smem;                                    // [0, 9216)
    int* Y1 = Y0 + IU4_MMQ_X * IU4_TILE_Y_K;           // [9216, 18432)
    int* A0 = Y1 + IU4_MMQ_X * IU4_TILE_Y_K;           // [18432, 39936)
    int* A1 = A0 + IU4_MMQ_Y * IU4_TILE_X_K;           // [39936, 61440)

    // Wave-uniform role split: compute waves 0..7 vs producer waves 8..9.
    // sum[64] is declared only inside the compute arm so it is not live on
    // producer waves (parent metadata: producer path drove the 256 VGPR alloc).
    if (wy < 8) {
        // ── compute path ────────────────────────────────────────────────
        float sum[64];
        #pragma unroll
        for (int i = 0; i < 64; ++i) sum[i] = 0.0f;

        // Binit: producers filled A0 + Y0; compute waits.
        __syncthreads(); // Binit

        for (int g = 0; g < G; ++g) {
            const int p = g & 1;
            int* A_cur = (p == 0) ? A0 : A1;

            // h0(g): read A[p]+Y0.
            if constexpr (dbg_mode != 2) {
                vec_dot_i4_x128_ws4_w2_gfx1151(A_cur, Y0, sum, 0);
            }
            __syncthreads(); // B0(g)

            // h1(g): read A[p]+Y1.
            if constexpr (dbg_mode != 2) {
                vec_dot_i4_x128_ws4_w2_gfx1151(A_cur, Y1, sum, 1);
            }
            __syncthreads(); // B1(g)
        }

        // Epilogue: compute waves only; shipping set/add and ownership map.
        // nocomp still stores sum (zeros) so the store path stays attributed.
        {
            const int w = wy;
            const int i0 = 32 * (w / 2);
            const int col_off = 16 * (w % 2);

            #pragma unroll
            for (int j0 = 0; j0 < IU4_MMQ_X; j0 += 32) {
                #pragma unroll
                for (int n = 0; n < 2; ++n) {
                    #pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        const int j = j0 + col_off + (lane % 16);
                        const int i = i0 + n * 16 + 2 * l + (lane / 16);
                        float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                        const float v = sum[(j0 / 16 + n) * 8 + l];
                        if (add) {
                            *yp += v;
                        } else {
                            *yp = v;
                        }
                    }
                }
            }
        }
    } else {
        // ── producer path ───────────────────────────────────────────────
        // No sum / C / fragment state. Same barrier COUNT and order as compute.

        // Prologue: fill A0 group0 both row-halves + Y0=(0,0).
        if constexpr (dbg_mode != 1) {
            const int* xq00 = (const int*)(Xq + (long long)0 * N + col0);
            load_iu4_stage_ws4_w2_gfx1151(
                A, xq00, A0, Y0, row0, 0, 0, groups_per_row, true, true);
            load_iu4_stage_ws4_w2_gfx1151(
                A, xq00, A0, Y0, row0, 0, 1, groups_per_row, true, false);
        }
        __syncthreads(); // Binit

        for (int g = 0; g < G; ++g) {
            const int p = g & 1;
            const bool next = (g + 1 < G);
            int* A_nxt = (p == 0) ? A1 : A0;

            // h0(g): write A_nxt top64 (if next) + Y1=(g,1).
            if constexpr (dbg_mode != 1) {
                const int* xq_g1 = (const int*)(Xq + (long long)(2 * g + 1) * N + col0);
                load_iu4_stage_ws4_w2_gfx1151(
                    A, xq_g1, A_nxt, Y1, row0, g + 1, 0, groups_per_row, next, true);
            }
            __syncthreads(); // B0(g)

            // h1(g): write A_nxt bot64 + Y0=(g+1,0) if next.
            if constexpr (dbg_mode != 1) {
                if (next) {
                    const int* xq_n0 = (const int*)(Xq + (long long)(2 * (g + 1)) * N + col0);
                    load_iu4_stage_ws4_w2_gfx1151(
                        A, xq_n0, A_nxt, Y0, row0, g + 1, 1, groups_per_row, true, true);
                }
            }
            __syncthreads(); // B1(g)
        }
    }
}


extern "C" __global__ __launch_bounds__(320, 1)
void gemm_mq4g256v2_residual_mmq_iu4_full_set_ws4_w2_gfx1151(
    const char* __restrict__ A,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    gemm_iu4_body_ws4_w2_gfx1151<0>(A, Xq, Y, M, K, N, add);
}

extern "C" __global__ __launch_bounds__(320, 1)
void gemm_mq4g256v2_residual_mmq_iu4_full_add_ws4_w2_gfx1151(
    const char* __restrict__ A,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    gemm_iu4_body_ws4_w2_gfx1151<0>(A, Xq, Y, M, K, N, add);
}

#if defined(HIPFIRE_WS4_DEBUG)
// Metadata-only attribution: noprod strips producer global/LDS traffic;
// nocomp strips vec_dot/fold while keeping barriers + epilogue stores.
extern "C" __global__ __launch_bounds__(320, 1)
void gemm_mq4g256v2_residual_mmq_iu4_full_set_ws4_w2_dbg_noprod_gfx1151(
    const char* __restrict__ A,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    gemm_iu4_body_ws4_w2_gfx1151<1>(A, Xq, Y, M, K, N, add);
}

extern "C" __global__ __launch_bounds__(320, 1)
void gemm_mq4g256v2_residual_mmq_iu4_full_add_ws4_w2_dbg_noprod_gfx1151(
    const char* __restrict__ A,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    gemm_iu4_body_ws4_w2_gfx1151<1>(A, Xq, Y, M, K, N, add);
}

extern "C" __global__ __launch_bounds__(320, 1)
void gemm_mq4g256v2_residual_mmq_iu4_full_set_ws4_w2_dbg_nocomp_gfx1151(
    const char* __restrict__ A,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    gemm_iu4_body_ws4_w2_gfx1151<2>(A, Xq, Y, M, K, N, add);
}

extern "C" __global__ __launch_bounds__(320, 1)
void gemm_mq4g256v2_residual_mmq_iu4_full_add_ws4_w2_dbg_nocomp_gfx1151(
    const char* __restrict__ A,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    gemm_iu4_body_ws4_w2_gfx1151<2>(A, Xq, Y, M, K, N, add);
}
#endif // HIPFIRE_WS4_DEBUG

#endif // __gfx1151__


#endif
