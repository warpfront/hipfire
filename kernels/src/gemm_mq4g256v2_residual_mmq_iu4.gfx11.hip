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
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_add_gm_col_gfx1151(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_set_gm_col_gfx1151(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_gm_col_gfx1151(
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

// IU4_FOLD_RN — W4.0/LF.0 bit-exact fold pin (all gfx11 entries share it).
// Every output element folds one int32 WMMA result C through EXACTLY this DAG
// (traced in emitted gfx1151 ISA, full_set_occ3_col h0 cluster: t1 L16340,
// p L16346, t2 L16347, term L16365, sum-add L16380; 128 instances per entry =
// 64 sums x h0/h1, so per-entry/per-arch differences are only scheduling):
//   t1   = RN(sc * d)            v_fma_mix_f32 (header.lo x d + (-0.0))
//   p    = RN(t1 * float(C))     v_mul_f32 (dual-issued on gfx1151)
//   t2   = RN(zp * d)            v_fma_mix_f32 (header.hi x d + (-0.0))
//   term = RN(t2 * float(s) + p) v_fmac_f32 (the zp term is the FMA product,
//                                          NOT the running sum)
//   sum  = RN(sum + term)        v_add_f32 (separate add, never fused)
// Spelling the products/FMA/final-add as __fmul_rn/__fmaf_rn/__fadd_rn pins the
// contraction choice explicitly instead of relying on fp-contract fusion order.
// The fpext+fmul idiom on the half2 lanes still folds to v_fma_mix_f32 in the
// backend exactly as before. d is evaluated once into a temporary; the sum
// lvalue keeps += read/modify/write semantics.
#define IU4_FOLD_RN(sum_lvalue, sc, zp, d, acc_i32, s_i32) \
    do { \
        const float iu4_fold_d_ = (d); \
        const float iu4_fold_t1_ = __fmul_rn((sc), iu4_fold_d_); \
        const float iu4_fold_p_ = __fmul_rn(iu4_fold_t1_, (float)(acc_i32)); \
        const float iu4_fold_t2_ = __fmul_rn((zp), iu4_fold_d_); \
        const float iu4_fold_term_ = __fmaf_rn(iu4_fold_t2_, (float)(s_i32), iu4_fold_p_); \
        (sum_lvalue) = __fadd_rn((sum_lvalue), iu4_fold_term_); \
    } while (0)

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
                IU4_FOLD_RN(sum[(j0 / tile_C::J + n) * tile_C::ne + l],
                            dmA.x, dmA.y, d_x, acc_i[l], s_x);
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

#define IU4_ENTRY(name, full, bounds_min) extern "C" __global__ __launch_bounds__(IU4_MMQ_NWARPS * IU4_WARP_SIZE, bounds_min) void name(     const char* __restrict__ A,     const block_i4_128* __restrict__ Xq,     float* __restrict__ Y,     int M,     int K,     int N,     int add ) {     gemm_iu4_body<full>(A, Xq, Y, M, K, N, add, blockIdx.x, blockIdx.y); }
#define IU4_ENTRY_COL(name, full, bounds_min) extern "C" __global__ __launch_bounds__(IU4_MMQ_NWARPS * IU4_WARP_SIZE, bounds_min) void name(     const char* __restrict__ A,     const block_i4_128* __restrict__ Xq,     float* __restrict__ Y,     int M,     int K,     int N,     int add ) {     gemm_iu4_body<full>(A, Xq, Y, M, K, N, add, blockIdx.y, blockIdx.x); }

IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4, false, 2)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_add, true, 2)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_set, true, 2)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3, true, 3)
IU4_ENTRY(gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3, true, 3)
#if defined(__gfx1151__)
// Halo IU4 low-footprint (LF.1): M128 x N128 tiled over SIXTEEN wave32s.
// Same 128x128 output tile, same LDS planes (Xq [0,9216), A [9216,30720)),
// same grid walk (column-adjacent) and same arithmetic DAG (IU4_FOLD_RN) as
// shipping. Differences vs shipping vec_dot_i4_x128 / gemm_iu4_body:
//  - 32 running sums/lane (4 column subtiles c=0..3 x 8 rows l=0..7);
//  - exactly ONE live int32x8 WMMA chain: each c is folded through
//    IU4_FOLD_RN into its own sums before the next c starts;
//  - A payload fill issues 4 row-dword loads before their stores/first wait
//    (two packets cover the 8 rows/wave); headers by waves 0..7 only;
//  - Xq half fill uses all 512 threads (5 words for tid<256, else 4).
// Per-lane ownership (w = threadIdx.y in [0,15], lane = threadIdx.x):
//   wave_row = 16*(w/2); wave_col = 64*(w%2);
//   i = wave_row + 2*l + lane/16; j = wave_col + 16*c + lane%16;
//   sum_index = 8*c + l. This is the frozen §4.1 mapping (CPU-enumerated
//   there to cover all 16384 outputs exactly once).
#define IU4_LF16_NWARPS 16
#define IU4_LF16_SUMS 32
#define IU4_LF16_XQ_WORDS 2304

// A-plane fill for one K group: payload packet-4 + replicated headers.
// Full-tile only (callers guarantee row0+i < M); mirrors load_iu4_tile.
static __device__ __forceinline__ void load_iu4_tile_lf16_gfx1151(
    const char* __restrict__ A,
    int* __restrict__ tile_x,
    int row0,
    int kb,
    int groups_per_row
) {
    const int lane = threadIdx.x;
    const int w = threadIdx.y;
    // Payload: 8 rows/thread (i = i0+w, i0 = 0,16,...,112) in two packets.
    // Each packet issues its 4 row-dword loads before the 4 LDS stores so
    // the wave has 4 A loads in flight ahead of the first VM wait.
    // Register note: four independent row pointers (not one base + stride):
    // the stride is block-uniform and the backend keeps these CSE'd without
    // extending a shared base's live range across the packets.
    #pragma unroll 1
    for (int p = 0; p < 2; ++p) {
        const int base = p * 64 + w;
        const char* gp0 = A + ((long long)(row0 + base) * groups_per_row + kb) * 136;
        const char* gp1 = A + ((long long)(row0 + base + 16) * groups_per_row + kb) * 136;
        const char* gp2 = A + ((long long)(row0 + base + 32) * groups_per_row + kb) * 136;
        const char* gp3 = A + ((long long)(row0 + base + 48) * groups_per_row + kb) * 136;
        const int v0 = *(const int*)(gp0 + 8 + lane * 4);
        const int v1 = *(const int*)(gp1 + 8 + lane * 4);
        const int v2 = *(const int*)(gp2 + 8 + lane * 4);
        const int v3 = *(const int*)(gp3 + 8 + lane * 4);
        tile_x[base * IU4_TILE_X_K + lane] = v0;
        tile_x[(base + 16) * IU4_TILE_X_K + lane] = v1;
        tile_x[(base + 32) * IU4_TILE_X_K + lane] = v2;
        tile_x[(base + 48) * IU4_TILE_X_K + lane] = v3;
    }
    // Headers: waves 0..7 only (uniform skip for waves 8..15); each of the
    // 256 global header words has exactly one writer, replicated to the 4
    // slots 32+4*ksc+[0,3] exactly as shipping.
    // Address: payload-form base (A+4*lane) + 136*((row0+i)*gpr+kb) + adj,
    // adj = 4*ksc-4*lane. Header = packet+4*ksc. The adj is tied to kb via
    // opaque asm so the backend cannot fold (A+4*lane)+adj into a loop-
    // invariant A+4*ksc 64-bit (that pair was a 2-VGPR spill across kb).
    if (w < 8) {
        const int i = 16 * w + lane / 2;
        const int ksc = lane & 1;
        int adj = 4 * ksc - 4 * lane;
        asm volatile("" : "+v"(adj) : "r"(kb));
        const char* pld = A + 4 * lane
            + ((long long)(row0 + i) * groups_per_row + kb) * 136;
        const unsigned int hs =
            *(const unsigned int*)(pld + adj);
        const float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        const float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        const half2 dm = make_half2((_Float16)sc, (_Float16)zp);
        half2* x_dm = (half2*)(tile_x + i * IU4_TILE_X_K + IU4_X_DM_OFF);
        #pragma unroll
        for (int l = 0; l < 4; ++l)
            x_dm[4 * ksc + l] = dm;
    }
}

// X128-structured dot over ONE 128-K half (h = 0/1) for the 16-wave tile:
// per column subtile c, 8 iu4 WMMAs (t = 0..3, low/high in shipping order)
// into one int32x8 C, then the pinned fold into sums[8*c+l] before the next
// c lifetime begins. d/s come from the single output column lane%16.
static __device__ __forceinline__ void vec_dot_i4_x128_lf16_gfx1151(
    const int* __restrict__ x,
    const int* __restrict__ y,
    float* __restrict__ sum,
    int h
) {
    const int lane = threadIdx.x;
    const int w = threadIdx.y;
    const int wave_row = 16 * (w / 2);
    const int wave_col = 64 * (w % 2);
    const int lane_row = lane % 16;
    const int lane_hi = lane / 16;
    const int a_row = wave_row + lane_row;
    // One live integer chain per half: t is unrolled in source (t0..t3 order
    // kept) while c unrolls to static sum indices. With the t loop rolled the
    // backend sinks all four folds past every WMMA loop and retains four
    // completed chains (113 VGPR); unrolled, each fold stays with its chain.
    // The sched fence retires each fold before the next chain starts, so the
    // allocator reuses a single int32x8 bank (98 VGPR). It emits no ISA.
    int32x8_t acc;
    float d_x;
    int s_x;
    int b_col;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        acc = (int32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
        b_col = wave_col + 16 * c + lane_row;
        {
            const int* ap = x + a_row * IU4_TILE_X_K + h * 16 + 0;
            const int* bp = y + b_col * IU4_TILE_Y_K + 2 + 0;
            const int32x2_t av0 = *(const int32x2_t*)(ap + 0);
            const int32x2_t av1 = *(const int32x2_t*)(ap + 2);
            const int32x2_t bv0 = *(const int32x2_t*)(bp + 0);
            const int32x2_t bv1 = *(const int32x2_t*)(bp + 2);
            acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av0, true, bv0, acc, false);
            acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av1, true, bv1, acc, false);
        }
        {
            const int* ap = x + a_row * IU4_TILE_X_K + h * 16 + 4;
            const int* bp = y + b_col * IU4_TILE_Y_K + 2 + 4;
            const int32x2_t av0 = *(const int32x2_t*)(ap + 0);
            const int32x2_t av1 = *(const int32x2_t*)(ap + 2);
            const int32x2_t bv0 = *(const int32x2_t*)(bp + 0);
            const int32x2_t bv1 = *(const int32x2_t*)(bp + 2);
            acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av0, true, bv0, acc, false);
            acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av1, true, bv1, acc, false);
        }
        {
            const int* ap = x + a_row * IU4_TILE_X_K + h * 16 + 8;
            const int* bp = y + b_col * IU4_TILE_Y_K + 2 + 8;
            const int32x2_t av0 = *(const int32x2_t*)(ap + 0);
            const int32x2_t av1 = *(const int32x2_t*)(ap + 2);
            const int32x2_t bv0 = *(const int32x2_t*)(bp + 0);
            const int32x2_t bv1 = *(const int32x2_t*)(bp + 2);
            acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av0, true, bv0, acc, false);
            acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av1, true, bv1, acc, false);
        }
        {
            const int* ap = x + a_row * IU4_TILE_X_K + h * 16 + 12;
            const int* bp = y + b_col * IU4_TILE_Y_K + 2 + 12;
            const int32x2_t av0 = *(const int32x2_t*)(ap + 0);
            const int32x2_t av1 = *(const int32x2_t*)(ap + 2);
            const int32x2_t bv0 = *(const int32x2_t*)(bp + 0);
            const int32x2_t bv1 = *(const int32x2_t*)(bp + 2);
            acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av0, true, bv0, acc, false);
            acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av1, true, bv1, acc, false);
        }
        d_x = *(const float*)(y + b_col * IU4_TILE_Y_K);
        s_x = *(const int*)(y + b_col * IU4_TILE_Y_K + 1);
        const int* acc_i = (const int*)&acc;
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int i = wave_row + 2 * l + lane_hi;
            const half2 dm = ((const half2*)(x + i * IU4_TILE_X_K + IU4_X_DM_OFF))[4 * h];
            const float2 dmA = __half22float2(dm);
            IU4_FOLD_RN(sum[8 * c + l], dmA.x, dmA.y, d_x, acc_i[l], s_x);
        }
        __builtin_amdgcn_sched_barrier(0);
    }
}
// Full-tile 16-wave body: four LDS barriers per K group, kb-increasing
// accumulation with h0 before h1, single residual add after all groups.
// Specialized on the store mode (host symbol fully determines it: set entry
// always stores, add entry always adds). The `add` kernarg is retained in the
// seven-arg ABI for layout parity but deliberately ignored here; do NOT
// reintroduce a runtime branch on it (its VGPR spans the whole kernel).
// The h1 guard is always true (2*kb+1 < 2*groups_per_row for every kb in
// [0, groups_per_row)) but is load-bearing for the allocator: the branch
// keeps the h1 live ranges out of the h0 scheduling region (measured 4
// spills with it, ~47 without). Do NOT straight-line it.
template <bool IS_ADD>
static __device__ __forceinline__ void gemm_iu4_body_lf16_gfx1151(
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
    (void)add;
    // NOTE: SGPR-pinning row0/col0 via asm "+s" was tried: zero VGPR change.
    // threadIdx.y cannot be pinned (backend rejects VGPR->SGPR as illegal).
    const int row0 = row_tile * IU4_MMQ_Y;
    const int col0 = col_tile * IU4_MMQ_X;
    const int groups_per_row = K / 256;
    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + IU4_MMQ_X * IU4_TILE_Y_K;
    // w/lane/wave_row/wave_col intentionally NOT materialised here: they are
    // pure functions of threadIdx and only feed the Y epilogue. Precomputing
    // the two address pieces (col-part = wave_col|(lane%16), row-part =
    // wave_row+lane/16) before kb spilled both across the whole loop.
    float sum[IU4_LF16_SUMS] = {0.0f};
    for (int kb = 0; kb < groups_per_row; ++kb) {
        load_iu4_tile_lf16_gfx1151(A, tile_x, row0, kb, groups_per_row);
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        #pragma unroll 1
        for (int l = threadIdx.y * IU4_WARP_SIZE + threadIdx.x; l < IU4_LF16_XQ_WORDS; l += 512)
            tile_y[l] = by0[l];
        __syncthreads();
        vec_dot_i4_x128_lf16_gfx1151(tile_x, tile_y, sum, 0);
        __syncthreads();
        if (2 * kb + 1 < 2 * groups_per_row) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            #pragma unroll 1
            for (int l = threadIdx.y * IU4_WARP_SIZE + threadIdx.x; l < IU4_LF16_XQ_WORDS; l += 512)
                tile_y[l] = by1[l];
            __syncthreads();
            vec_dot_i4_x128_lf16_gfx1151(tile_x, tile_y, sum, 1);
            __syncthreads();
        }
    }
    // Y epilogue: materialise row_part/col_part AFTER kb. Lane via v_mbcnt_lo
    // inside the asm (wave32 == threadIdx.x) so raw tid_x is not an asm input
    // and is not carried across kb. tid_y stays live as the wave index.
    // sum[0] anchors after accumulation. Y_index = (col0+j)*M+(row0+i).
    {
        const int tid_y = (int)__builtin_amdgcn_workitem_id_y();
        int row_part, col_part, tmp;
        asm volatile(
            "v_mbcnt_lo_u32_b32 %2, -1, 0\n\t"      // lane = 0..31
            "v_lshrrev_b32_e32 %0, 1, %3\n\t"       // w/2
            "v_bfe_u32 %1, %2, 4, 1\n\t"            // lane/16 (scratch in %1)
            "v_lshl_or_b32 %0, %0, 4, %1\n\t"       // row_part = 16*(w/2)+lane/16
            "v_and_b32_e32 %1, 1, %3\n\t"           // w&1
            "v_and_b32_e32 %2, 15, %2\n\t"          // lane&15
            "v_lshl_or_b32 %1, %1, 6, %2"           // col_part = 64*(w%2)+lane%16
            : "=&v"(row_part), "=&v"(col_part), "=&v"(tmp)
            : "v"(tid_y), "v"(sum[0]));
        const int row0e = row_tile * IU4_MMQ_Y;
        const int col0e = col_tile * IU4_MMQ_X;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            #pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int i = row_part + 2 * l;
                const int j = col_part + 16 * c;
                float* yp = Y + (long long)(col0e + j) * M + (row0e + i);
                const float v = sum[8 * c + l];
                if constexpr (IS_ADD) {
                    *yp += v;
                } else {
                    *yp = v;
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(512, 2) __attribute__((amdgpu_num_vgpr(96))) void gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151(
    const char* __restrict__ A,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    gemm_iu4_body_lf16_gfx1151<false>(A, Xq, Y, M, K, N, add, blockIdx.y, blockIdx.x);
}
extern "C" __global__ __launch_bounds__(512, 2) __attribute__((amdgpu_num_vgpr(96))) void gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151(
    const char* __restrict__ A,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    gemm_iu4_body_lf16_gfx1151<true>(A, Xq, Y, M, K, N, add, blockIdx.y, blockIdx.x);
}
#endif
#if defined(__gfx1151__) || defined(__gfx1100__)
IU4_ENTRY_COL(gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151, true, 3)
IU4_ENTRY_COL(gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151, true, 3)
#endif
#if defined(__gfx1151__)
// IU4 group-major (GM) experiment: byte-identical to the shipping 8-wave
// _occ3_col_gfx1151 entries except the A group address. Row-major shipping:
//   gp = A + ((row)*groups_per_row + kb)*136  (128-row x 1-group tile spans
//   128*K/256*136 B strides). Group-major:
//   gp = A + 136*((size_t)kb*M + row), so each 128-row x 1-group tile is one
//   contiguous 128*136 B block. Same ABI/grid/block/LDS; bit-exact by
//   construction (same bytes land in LDS).
template <bool FULL>
static __device__ __forceinline__ void load_iu4_tile_gm(
    const char* __restrict__ A,
    int* __restrict__ tile_x,
    int row0,
    int kb,
    int M,
    int groups_per_row
) {
    (void)groups_per_row;
    int* x_qs = tile_x;
    half2* x_dm = (half2*)(x_qs + IU4_X_DM_OFF);

    const int txi = threadIdx.x;

    for (int i0 = 0; i0 < IU4_MMQ_Y; i0 += IU4_MMQ_NWARPS) {
        const int i = i0 + threadIdx.y;
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const char* gp = A + 136 * ((size_t)kb * M + row);
        x_qs[i * IU4_TILE_X_K + txi] = *(const int*)(gp + 8 + txi * 4);
    }

    constexpr int rows_per_warp_meta = 16;
    for (int i0 = 0; i0 < IU4_MMQ_Y; i0 += IU4_MMQ_NWARPS * rows_per_warp_meta) {
        const int i = i0 + threadIdx.y * rows_per_warp_meta + threadIdx.x / 2;
        if (i < IU4_MMQ_Y) {
            const int row = FULL ? row0 + i : min(row0 + i, M - 1);
            const char* gp = A + 136 * ((size_t)kb * M + row);
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

template <bool FULL>
static __device__ __forceinline__ void gemm_iu4_body_gm(
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
        load_iu4_tile_gm<FULL>(A, tile_x, row0, kb, M, groups_per_row);

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

extern "C" __global__ __launch_bounds__(IU4_MMQ_NWARPS * IU4_WARP_SIZE, 3) void gemm_mq4g256v2_residual_mmq_iu4_full_add_gm_col_gfx1151(     const char* __restrict__ A,     const block_i4_128* __restrict__ Xq,     float* __restrict__ Y,     int M,     int K,     int N,     int add ) {     gemm_iu4_body_gm<true>(A, Xq, Y, M, K, N, add, blockIdx.y, blockIdx.x); }
extern "C" __global__ __launch_bounds__(IU4_MMQ_NWARPS * IU4_WARP_SIZE, 3) void gemm_mq4g256v2_residual_mmq_iu4_full_set_gm_col_gfx1151(     const char* __restrict__ A,     const block_i4_128* __restrict__ Xq,     float* __restrict__ Y,     int M,     int K,     int N,     int add ) {     gemm_iu4_body_gm<true>(A, Xq, Y, M, K, N, add, blockIdx.y, blockIdx.x); }

// LF16 group-major twin of load_iu4_tile_lf16_gfx1151: same packets, same
// header replication, same opaque-kb adj; only the group base changes from
// (row0+i)*gpr+kb to kb*M+(row0+i). Takes M explicitly (shipping infers
// full-tile bounds from the caller).
static __device__ __forceinline__ void load_iu4_tile_lf16_gm_gfx1151(
    const char* __restrict__ A,
    int* __restrict__ tile_x,
    int row0,
    int kb,
    int M,
    int groups_per_row
) {
    (void)groups_per_row;
    const int lane = threadIdx.x;
    const int w = threadIdx.y;
    #pragma unroll 1
    for (int p = 0; p < 2; ++p) {
        const int base = p * 64 + w;
        const char* gp0 = A + 136 * ((size_t)kb * M + (row0 + base));
        const char* gp1 = A + 136 * ((size_t)kb * M + (row0 + base + 16));
        const char* gp2 = A + 136 * ((size_t)kb * M + (row0 + base + 32));
        const char* gp3 = A + 136 * ((size_t)kb * M + (row0 + base + 48));
        const int v0 = *(const int*)(gp0 + 8 + lane * 4);
        const int v1 = *(const int*)(gp1 + 8 + lane * 4);
        const int v2 = *(const int*)(gp2 + 8 + lane * 4);
        const int v3 = *(const int*)(gp3 + 8 + lane * 4);
        tile_x[base * IU4_TILE_X_K + lane] = v0;
        tile_x[(base + 16) * IU4_TILE_X_K + lane] = v1;
        tile_x[(base + 32) * IU4_TILE_X_K + lane] = v2;
        tile_x[(base + 48) * IU4_TILE_X_K + lane] = v3;
    }
    if (w < 8) {
        const int i = 16 * w + lane / 2;
        const int ksc = lane & 1;
        int adj = 4 * ksc - 4 * lane;
        asm volatile("" : "+v"(adj) : "r"(kb));
        const char* pld = A + 4 * lane
            + 136 * ((size_t)kb * M + (row0 + i));
        const unsigned int hs =
            *(const unsigned int*)(pld + adj);
        const float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        const float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        const half2 dm = make_half2((_Float16)sc, (_Float16)zp);
        half2* x_dm = (half2*)(tile_x + i * IU4_TILE_X_K + IU4_X_DM_OFF);
        #pragma unroll
        for (int l = 0; l < 4; ++l)
            x_dm[4 * ksc + l] = dm;
    }
}

template <bool IS_ADD>
static __device__ __forceinline__ void gemm_iu4_body_lf16_gm_gfx1151(
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
    (void)add;
    const int row0 = row_tile * IU4_MMQ_Y;
    const int col0 = col_tile * IU4_MMQ_X;
    const int groups_per_row = K / 256;
    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + IU4_MMQ_X * IU4_TILE_Y_K;
    float sum[IU4_LF16_SUMS] = {0.0f};
    for (int kb = 0; kb < groups_per_row; ++kb) {
        load_iu4_tile_lf16_gm_gfx1151(A, tile_x, row0, kb, M, groups_per_row);
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        #pragma unroll 1
        for (int l = threadIdx.y * IU4_WARP_SIZE + threadIdx.x; l < IU4_LF16_XQ_WORDS; l += 512)
            tile_y[l] = by0[l];
        __syncthreads();
        vec_dot_i4_x128_lf16_gfx1151(tile_x, tile_y, sum, 0);
        __syncthreads();
        if (2 * kb + 1 < 2 * groups_per_row) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            #pragma unroll 1
            for (int l = threadIdx.y * IU4_WARP_SIZE + threadIdx.x; l < IU4_LF16_XQ_WORDS; l += 512)
                tile_y[l] = by1[l];
            __syncthreads();
            vec_dot_i4_x128_lf16_gfx1151(tile_x, tile_y, sum, 1);
            __syncthreads();
        }
    }
    {
        const int tid_y = (int)__builtin_amdgcn_workitem_id_y();
        int row_part, col_part, tmp;
        asm volatile(
            "v_mbcnt_lo_u32_b32 %2, -1, 0\n\t"
            "v_lshrrev_b32_e32 %0, 1, %3\n\t"
            "v_bfe_u32 %1, %2, 4, 1\n\t"
            "v_lshl_or_b32 %0, %0, 4, %1\n\t"
            "v_and_b32_e32 %1, 1, %3\n\t"
            "v_and_b32_e32 %2, 15, %2\n\t"
            "v_lshl_or_b32 %1, %1, 6, %2"
            : "=&v"(row_part), "=&v"(col_part), "=&v"(tmp)
            : "v"(tid_y), "v"(sum[0]));
        const int row0e = row_tile * IU4_MMQ_Y;
        const int col0e = col_tile * IU4_MMQ_X;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            #pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int i = row_part + 2 * l;
                const int j = col_part + 16 * c;
                float* yp = Y + (long long)(col0e + j) * M + (row0e + i);
                const float v = sum[8 * c + l];
                if constexpr (IS_ADD) {
                    *yp += v;
                } else {
                    *yp = v;
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(512, 2) __attribute__((amdgpu_num_vgpr(96))) void gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_gm_col_gfx1151(
    const char* __restrict__ A,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    gemm_iu4_body_lf16_gm_gfx1151<false>(A, Xq, Y, M, K, N, add, blockIdx.y, blockIdx.x);
}
#endif

#endif
