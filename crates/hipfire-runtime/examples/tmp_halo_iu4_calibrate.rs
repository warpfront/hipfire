//! A0 — IU4 WMMA-rate calibration + attribution twins (plan §6).
//!
//! Diagnostic only. No production patch.
//!
//! Halo (hipx device 1):
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-a0 \
//!     cargo run --release -p hipfire-runtime --example tmp_halo_iu4_calibrate
//!
//! Metadata symbols the parent must read after first JIT (radiowave /
//! llvm-readobj notes on the cached .hsaco under HIPFIRE_KERNEL_CACHE):
//!   iu4_probe_dependent
//!   iu4_probe_independent
//!   iu4_probe_independent_lds
//!   gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_a0_prod
//!   gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_a0_noload
//!   gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_a0_nofold
//!
//! Resource ceilings (plan §4): probes ≤192 VGPR, mode a/b LDS 0, mode c
//! exactly 2048 B; twins ≤192 VGPR, dynamic LDS 30720 B; spill/scratch 0.

use hip_bridge::KernargBlob;
use rdna_compute::Gpu;
use std::time::Instant;

const CU_TARGET: f64 = 40.0;
const BLOCK: [u32; 3] = [32, 8, 1];
const NWARPS: u32 = 8;
const PROBE_LDS_BYTES: u32 = 2048; // mode c: 8 waves × 256 B
const IU4_LDS_BYTES: u32 = 30720;
const RTC_HZ: f64 = 100_000_000.0; // AMD s_memrealtime wall-clock timebase (100 MHz)
const ATTR_ITERS: usize = 100;
const ATTR_WARMUP: usize = 5;

const PROBE_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <stdint.h>

using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

// Nonzero nibble-valid operands: host packs identical int32x2 A/B for lanes
// r and r+16 (address depends on lane%16 only). Four WMMAs/iter.
// Checksum buffer keeps the loop live. Periodic acc mask bounds i32 overflow.

static __device__ __forceinline__ int32x2_t load_op(const int* __restrict__ base, int pair) {
    return *(const int32x2_t*)(base + pair * 2);
}

static __device__ __forceinline__ void bound_acc(int32x8_t& acc) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = (acc[j] << 8) >> 8; // sign-extend low 24b
}

static __device__ __forceinline__ int acc_checksum(const int32x8_t& acc) {
    int cs = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) cs += acc[j] * (j + 1);
    return cs;
}

extern "C" __global__ __launch_bounds__(256, 2)
void iu4_probe_dependent(
    const int* __restrict__ op_a,
    const int* __restrict__ op_b,
    int* __restrict__ checksum,
    unsigned long long* __restrict__ ticks,
    int loops
) {
    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int pair = lane & 15;
    const int32x2_t av = load_op(op_a, pair);
    const int32x2_t bv = load_op(op_b, pair);

    unsigned long long t0 = 0, t1 = 0;
    if (lane == 0 && wave == 0) t0 = __builtin_amdgcn_s_memrealtime();

    int32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < loops; ++i) {
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av, true, bv, acc, false);
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av, true, bv, acc, false);
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av, true, bv, acc, false);
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av, true, bv, acc, false);
        if ((i & 31) == 31) bound_acc(acc);
    }

    if (lane == 0 && wave == 0) t1 = __builtin_amdgcn_s_memrealtime();

    const int tid = (int)blockIdx.x * 256 + wave * 32 + lane;
    checksum[tid] = acc_checksum(acc);
    if (lane == 0 && wave == 0 && blockIdx.x == 0) {
        ticks[0] = t0;
        ticks[1] = t1;
    }
}

extern "C" __global__ __launch_bounds__(256, 2)
void iu4_probe_independent(
    const int* __restrict__ op_a,
    const int* __restrict__ op_b,
    int* __restrict__ checksum,
    unsigned long long* __restrict__ ticks,
    int loops
) {
    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int pair = lane & 15;
    const int32x2_t av = load_op(op_a, pair);
    const int32x2_t bv = load_op(op_b, pair);

    unsigned long long t0 = 0, t1 = 0;
    if (lane == 0 && wave == 0) t0 = __builtin_amdgcn_s_memrealtime();

    int32x8_t acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    int32x8_t acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
    int32x8_t acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
    int32x8_t acc3 = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < loops; ++i) {
        acc0 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av, true, bv, acc0, false);
        acc1 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av, true, bv, acc1, false);
        acc2 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av, true, bv, acc2, false);
        acc3 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av, true, bv, acc3, false);
        if ((i & 31) == 31) {
            bound_acc(acc0); bound_acc(acc1); bound_acc(acc2); bound_acc(acc3);
        }
    }

    if (lane == 0 && wave == 0) t1 = __builtin_amdgcn_s_memrealtime();

    const int tid = (int)blockIdx.x * 256 + wave * 32 + lane;
    checksum[tid] = acc_checksum(acc0) + acc_checksum(acc1)
                  + acc_checksum(acc2) + acc_checksum(acc3);
    if (lane == 0 && wave == 0 && blockIdx.x == 0) {
        ticks[0] = t0;
        ticks[1] = t1;
    }
}

// independent+LDS: four chains; two ds_read_b64 operand loads per WMMA from a
// per-wave 256 B region. Bytes [wave*256,(wave+1)*256) hold two 128 B fragment
// arrays, indexed by (lane&15)*8; lanes r and r+16 broadcast intentionally.
// Init once, barrier once, no LDS writes during timed work.
extern "C" __global__ __launch_bounds__(256, 2)
void iu4_probe_independent_lds(
    const int* __restrict__ op_a,
    const int* __restrict__ op_b,
    int* __restrict__ checksum,
    unsigned long long* __restrict__ ticks,
    int loops
) {
    extern __shared__ unsigned char smem[];
    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int pair = lane & 15;

    unsigned char* region = smem + wave * 256;
    // A plane [0,128), B plane [128,256). Each plane: 16 slots × 8 B.
    if (lane < 16) {
        const int32x2_t av = load_op(op_a, pair);
        const int32x2_t bv = load_op(op_b, pair);
        *(int32x2_t*)(region + pair * 8) = av;
        *(int32x2_t*)(region + 128 + pair * 8) = bv;
    }
    __syncthreads();

    unsigned long long t0 = 0, t1 = 0;
    if (lane == 0 && wave == 0) t0 = __builtin_amdgcn_s_memrealtime();

    int32x8_t acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    int32x8_t acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
    int32x8_t acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
    int32x8_t acc3 = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < loops; ++i) {
        // Two 8 B LDS loads (ds_read_b64 shape) per WMMA: A then B fragment.
        unsigned long long a_bits = *(const unsigned long long*)(region + pair * 8);
        unsigned long long b_bits = *(const unsigned long long*)(region + 128 + pair * 8);
        int32x2_t av = *(const int32x2_t*)(&a_bits);
        int32x2_t bv = *(const int32x2_t*)(&b_bits);
        acc0 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av, true, bv, acc0, false);

        a_bits = *(const unsigned long long*)(region + pair * 8);
        b_bits = *(const unsigned long long*)(region + 128 + pair * 8);
        av = *(const int32x2_t*)(&a_bits);
        bv = *(const int32x2_t*)(&b_bits);
        acc1 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av, true, bv, acc1, false);

        a_bits = *(const unsigned long long*)(region + pair * 8);
        b_bits = *(const unsigned long long*)(region + 128 + pair * 8);
        av = *(const int32x2_t*)(&a_bits);
        bv = *(const int32x2_t*)(&b_bits);
        acc2 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av, true, bv, acc2, false);

        a_bits = *(const unsigned long long*)(region + pair * 8);
        b_bits = *(const unsigned long long*)(region + 128 + pair * 8);
        av = *(const int32x2_t*)(&a_bits);
        bv = *(const int32x2_t*)(&b_bits);
        acc3 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, av, true, bv, acc3, false);

        if ((i & 31) == 31) {
            bound_acc(acc0); bound_acc(acc1); bound_acc(acc2); bound_acc(acc3);
        }
    }

    if (lane == 0 && wave == 0) t1 = __builtin_amdgcn_s_memrealtime();

    const int tid = (int)blockIdx.x * 256 + wave * 32 + lane;
    checksum[tid] = acc_checksum(acc0) + acc_checksum(acc1)
                  + acc_checksum(acc2) + acc_checksum(acc3);
    if (lane == 0 && wave == 0 && blockIdx.x == 0) {
        ticks[0] = t0;
        ticks[1] = t1;
    }
}
"#;

/// Twin source: production IU4 layout with MODE=prod|noload|nofold full_set_occ3 entries.
/// Built from production helpers; only the three a0 diagnostic symbols are exported.
const TWIN_SRC: &str = r#"
// block_i4_128 (quant recipe stripped — host packs activations).
struct block_i4_128 {
    float d;
    int s;
    unsigned char qs[64];
};
static_assert(sizeof(block_i4_128) == 72, "bad block size");

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
#define IU4_TILE_X_K 42
#define IU4_X_QS_WORDS 32
#define IU4_X_DM_OFF 32

#if !HIPFIRE_RDNA3
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_a0_prod(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_a0_noload(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_a0_nofold(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
#else

enum data_layout { DATA_LAYOUT_I_MAJOR = 0, DATA_LAYOUT_J_MAJOR = 10 };
template <int I_, int J_, typename T, data_layout dl = DATA_LAYOUT_I_MAJOR> struct tile {};
template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I * J / 32;
    T x[ne] = {0};
    static __device__ __forceinline__ int get_i(const int) { return threadIdx.x % 16; }
    static __device__ __forceinline__ int get_j(const int l) {
        if constexpr (I == 16 && J == 16) return 2 * l + (threadIdx.x / 16);
        else return l;
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

template <bool FULL>
static __device__ __forceinline__ void load_iu4_tile(
    const char* __restrict__ A, int* __restrict__ tile_x,
    int row0, int kb, int M, int groups_per_row
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

// fold_mode: 0 = production f32 fold, 1 = integer checksum fold (no-fold twin)
template <int FOLD_MODE>
static __device__ __forceinline__ void vec_dot_i4_x128_mode(
    const int* __restrict__ x, const int* __restrict__ y,
    float* __restrict__ sum, int h
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
            const int* acc_i = (const int*)&acc;
            if constexpr (FOLD_MODE == 0) {
                const float d_x = *(const float*)(y_ds + (j0 + j_lane) * IU4_TILE_Y_K);
                const int s_x = *(const int*)(y_ds + (j0 + j_lane) * IU4_TILE_Y_K + 1);
                #pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                    const half2 dm = ((const half2*)(x_qs + i * IU4_TILE_X_K + IU4_X_DM_OFF))[4 * h];
                    const float2 dmA = __half22float2(dm);
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] +=
                        dmA.x * d_x * (float)acc_i[l] + dmA.y * d_x * (float)s_x;
                }
            } else {
                // Cheap observable checksum of each integer accumulator (no f32 scale fold).
                #pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int v = acc_i[l];
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] +=
                        (float)((v & 0xffff) ^ ((v >> 16) & 0xffff));
                }
            }
        }
    }
}

// NOLOAD: 0 = normal loads every group; 1 = group0 fills, later groups reuse X/Y
template <bool FULL, int NOLOAD, int FOLD_MODE>
static __device__ __forceinline__ void gemm_iu4_body_a0(
    const char* __restrict__ A,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int add
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
        const bool do_load = (NOLOAD == 0) || (kb == 0);
        if (do_load) {
            load_iu4_tile<FULL>(A, tile_x, row0, kb, M, groups_per_row);
            const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
            for (int l0 = 0; l0 < IU4_MMQ_X * IU4_TILE_Y_K; l0 += IU4_MMQ_NWARPS * IU4_WARP_SIZE) {
                const int l = l0 + threadIdx.y * IU4_WARP_SIZE + threadIdx.x;
                if (FULL || col0 + l / IU4_TILE_Y_K < N) tile_y[l] = by0[l];
                else tile_y[l] = 0;
            }
        }
        __syncthreads();
        vec_dot_i4_x128_mode<FOLD_MODE>(tile_x, tile_y, sum, 0);
        __syncthreads();

        if (2 * kb + 1 < blocks128) {
            if (do_load) {
                const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
                for (int l0 = 0; l0 < IU4_MMQ_X * IU4_TILE_Y_K; l0 += IU4_MMQ_NWARPS * IU4_WARP_SIZE) {
                    const int l = l0 + threadIdx.y * IU4_WARP_SIZE + threadIdx.x;
                    if (FULL || col0 + l / IU4_TILE_Y_K < N) tile_y[l] = by1[l];
                    else tile_y[l] = 0;
                }
            }
            // noload h1: reuse existing (stale) Y — do not introduce uninitialized half
            __syncthreads();
            vec_dot_i4_x128_mode<FOLD_MODE>(tile_x, tile_y, sum, 1);
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
                    if (add) *yp += v;
                    else *yp = v;
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(IU4_MMQ_NWARPS * IU4_WARP_SIZE, 3)
void gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_a0_prod(
    const char* __restrict__ A, const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y, int M, int K, int N, int add
) {
    gemm_iu4_body_a0<true, 0, 0>(A, Xq, Y, M, K, N, add);
}

extern "C" __global__ __launch_bounds__(IU4_MMQ_NWARPS * IU4_WARP_SIZE, 3)
void gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_a0_noload(
    const char* __restrict__ A, const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y, int M, int K, int N, int add
) {
    gemm_iu4_body_a0<true, 1, 0>(A, Xq, Y, M, K, N, add);
}

extern "C" __global__ __launch_bounds__(IU4_MMQ_NWARPS * IU4_WARP_SIZE, 3)
void gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_a0_nofold(
    const char* __restrict__ A, const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y, int M, int K, int N, int add
) {
    gemm_iu4_body_a0<true, 0, 1>(A, Xq, Y, M, K, N, add);
}

#endif
"#;

fn f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0 {
        return sign << 15;
    }
    if exp == 0xff {
        let m = if mant != 0 { 0x200 } else { 0 };
        return (sign << 15) | 0x7c00 | m;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1f {
        return (sign << 15) | 0x7c00;
    }
    if new_exp <= 0 {
        return sign << 15;
    }
    (sign << 15) | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

fn prng_u32(i: u64, salt: u32) -> u32 {
    let mut x = (i as u32)
        .wrapping_mul(0x9e3779b1)
        .wrapping_add(salt)
        .wrapping_mul(0x85ebca6b);
    x ^= x >> 16;
    x = x.wrapping_mul(0xc2b2ae35);
    x ^ (x >> 16)
}

/// Pack MQ4G256V2 weights: 136 B/group, dual fp16 headers, nibble payload @+8.
fn pack_mq4g256v2(m: usize, k: usize, salt: u32) -> Vec<u8> {
    const GROUP: usize = 256;
    const HALF: usize = 128;
    const GB: usize = 136;
    assert!(k % GROUP == 0);
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * GB];
    for r in 0..m {
        for g in 0..gpr {
            let dst = (r * gpr + g) * GB;
            // Nonzero dual headers: sc=0.05, zp=-0.4-ish per half (distinct).
            let sc0 = f16_bits(0.05 + (prng_u32((r * gpr + g) as u64, salt) % 20) as f32 * 0.001);
            let zp0 = f16_bits(-0.35);
            let sc1 = f16_bits(0.07 + (prng_u32((r * gpr + g) as u64, salt ^ 1) % 20) as f32 * 0.001);
            let zp1 = f16_bits(-0.45);
            blob[dst..dst + 2].copy_from_slice(&sc0.to_le_bytes());
            blob[dst + 2..dst + 4].copy_from_slice(&zp0.to_le_bytes());
            blob[dst + 4..dst + 6].copy_from_slice(&sc1.to_le_bytes());
            blob[dst + 6..dst + 8].copy_from_slice(&zp1.to_le_bytes());
            for i in 0..HALF {
                // Nibble-valid nonzero codes 1..7 (avoid 0-fill dead paths).
                let lo = 1 + (prng_u32((r * 1024 + g * HALF + i) as u64, salt) % 7) as u8;
                let hi = 1 + (prng_u32((r * 1024 + g * HALF + i) as u64, salt ^ 0x55) % 7) as u8;
                blob[dst + 8 + i] = (lo & 0xf) | ((hi & 0xf) << 4);
            }
        }
    }
    blob
}

/// Host-pack block_i4_128 activations: (K/128)*N blocks, 72 B each.
fn pack_i4_xq(k: usize, n: usize, salt: u32) -> Vec<u8> {
    assert!(k % 128 == 0);
    let blocks = k / 128;
    let mut blob = vec![0u8; blocks * n * 72];
    for b in 0..blocks {
        for t in 0..n {
            let off = (b * n + t) * 72;
            let d = 0.02f32 + (prng_u32((b * n + t) as u64, salt) % 50) as f32 * 0.0004;
            // Exact int sum of 128 codes in -8..7; use small nonzero codes.
            let mut s: i32 = 0;
            let mut qs = [0u8; 64];
            for i in 0..64 {
                let e0 = (1 + (prng_u32((b * 256 + t * 64 + i) as u64, salt) % 5) as i8) - 2; // -1..3
                let e1 = (1 + (prng_u32((b * 256 + t * 64 + i) as u64, salt ^ 3) % 5) as i8) - 2;
                qs[i] = ((e0 as u8) & 0xf) | (((e1 as u8) & 0xf) << 4);
                s += e0 as i32 + e1 as i32;
            }
            blob[off..off + 4].copy_from_slice(&d.to_le_bytes());
            blob[off + 4..off + 8].copy_from_slice(&s.to_le_bytes());
            blob[off + 8..off + 72].copy_from_slice(&qs);
        }
    }
    blob
}

/// Nibble-valid int32x2 operands: 16 pairs × 2 i32 = 32 i32 for A and B.
fn pack_probe_ops() -> (Vec<i32>, Vec<i32>) {
    // Unsigned A nibbles 1; signed B nibbles +1 (0x1).
    let a_word = 0x1111_1111u32 as i32;
    let b_word = 0x1111_1111u32 as i32;
    let mut a = vec![0i32; 32];
    let mut b = vec![0i32; 32];
    for pair in 0..16 {
        a[pair * 2] = a_word;
        a[pair * 2 + 1] = a_word.wrapping_add(pair as i32 + 1);
        b[pair * 2] = b_word;
        b[pair * 2 + 1] = b_word.wrapping_add((pair as i32) * 2 + 1);
    }
    (a, b)
}

fn median_f64(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

fn pct_diff(a: f64, b: f64) -> f64 {
    let den = a.max(b).max(1e-30);
    ((a - b).abs() / den) * 100.0
}

struct ProbeResult {
    name: &'static str,
    grid: u32,
    loops: i32,
    event_us: f64,
    rtc_us: f64,
    wmma_per_us_cu: f64,
    checksum_or: i32,
}

fn launch_probe(
    gpu: &mut Gpu,
    name: &str,
    grid: u32,
    loops: i32,
    d_a: &rdna_compute::GpuTensor,
    d_b: &rdna_compute::GpuTensor,
    d_cs: &rdna_compute::GpuTensor,
    d_ticks: &rdna_compute::GpuTensor,
    lds: u32,
) -> (f64, u64, u64) {
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    let mut args = KernargBlob::new();
    args.push_ptr(d_a.buf.as_ptr());
    args.push_ptr(d_b.buf.as_ptr());
    args.push_ptr(d_cs.buf.as_ptr());
    args.push_ptr(d_ticks.buf.as_ptr());
    args.push_i32(loops);
    gpu.hip
        .event_record(&start, gpu.active_stream.as_ref())
        .unwrap();
    gpu.launch_kernel_blob(name, [grid, 1, 1], BLOCK, lds, args.as_mut_slice())
        .unwrap_or_else(|e| panic!("launch {name} grid={grid} loops={loops}: {e:?}"));
    gpu.hip
        .event_record(&stop, gpu.active_stream.as_ref())
        .unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64;
    let _ = gpu.hip.event_destroy(start);
    let _ = gpu.hip.event_destroy(stop);
    let mut tick_bytes = vec![0u8; 16];
    if gpu.hip.memcpy_dtoh(&mut tick_bytes, &d_ticks.buf).is_err() {
        tick_bytes.fill(0);
    }
    let t0 = u64::from_le_bytes(tick_bytes[0..8].try_into().unwrap());
    let t1 = u64::from_le_bytes(tick_bytes[8..16].try_into().unwrap());
    (ms * 1000.0, t0, t1)
}

fn total_wmmas(grid: u32, loops: i32) -> f64 {
    // wave-level WMMA instructions: grid × 8 waves × loops × 4 WMMAs
    // Do NOT divide by lane count.
    (grid as f64) * (NWARPS as f64) * (loops as f64) * 4.0
}

fn ensure_probes(gpu: &mut Gpu) {
    const MOD: &str = "halo_iu4_a0_probes";
    for sym in [
        "iu4_probe_dependent",
        "iu4_probe_independent",
        "iu4_probe_independent_lds",
    ] {
        gpu.ensure_kernel_public(MOD, PROBE_SRC, sym)
            .unwrap_or_else(|e| panic!("compile {sym}: {e:?}"));
        eprintln!("A0 meta: compiled probe symbol {sym} (module {MOD})");
    }
}

fn ensure_twins(gpu: &mut Gpu) {
    const MOD: &str = "halo_iu4_a0_twins";
    for sym in [
        "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_a0_prod",
        "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_a0_noload",
        "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_a0_nofold",
    ] {
        gpu.ensure_kernel_public(MOD, TWIN_SRC, sym)
            .unwrap_or_else(|e| panic!("compile twin {sym}: {e:?}"));
        eprintln!("A0 meta: compiled twin symbol {sym} (module {MOD})");
    }
}

fn launch_twin(
    gpu: &mut Gpu,
    name: &str,
    d_a: &rdna_compute::GpuTensor,
    d_xq: &rdna_compute::GpuTensor,
    d_y: &rdna_compute::GpuTensor,
    m: i32,
    k: i32,
    n: i32,
) -> f64 {
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    let mut args = KernargBlob::new();
    args.push_ptr(d_a.buf.as_ptr());
    args.push_ptr(d_xq.buf.as_ptr());
    args.push_ptr(d_y.buf.as_ptr());
    args.push_i32(m);
    args.push_i32(k);
    args.push_i32(n);
    args.push_i32(0); // set
    let row_tiles = (m as u32).div_ceil(128);
    let col_tiles = (n as u32).div_ceil(128);
    gpu.hip
        .event_record(&start, gpu.active_stream.as_ref())
        .unwrap();
    gpu.launch_kernel_blob(
        name,
        [row_tiles, col_tiles, 1],
        BLOCK,
        IU4_LDS_BYTES,
        args.as_mut_slice(),
    )
    .unwrap_or_else(|e| panic!("launch twin {name}: {e:?}"));
    gpu.hip
        .event_record(&stop, gpu.active_stream.as_ref())
        .unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64;
    let _ = gpu.hip.event_destroy(start);
    let _ = gpu.hip.event_destroy(stop);
    ms * 1000.0 // µs
}

fn run_probes(gpu: &mut Gpu) {
    let (a_h, b_h) = pack_probe_ops();
    let a_bytes: Vec<u8> = a_h.iter().flat_map(|x| x.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b_h.iter().flat_map(|x| x.to_le_bytes()).collect();
    let d_a = gpu.upload_raw(&a_bytes, &[a_bytes.len()]).unwrap();
    let d_b = gpu.upload_raw(&b_bytes, &[b_bytes.len()]).unwrap();

    let grids = [80u32, 160, 320];
    let loops = [1024i32, 4096, 8192];
    let max_grid = *grids.iter().max().unwrap();
    let cs_elems = (max_grid as usize) * 256;
    let z = vec![0i32; cs_elems];
    let cs_bytes: Vec<u8> = z.iter().flat_map(|x| x.to_le_bytes()).collect();
    let d_cs = gpu.upload_raw(&cs_bytes, &[cs_bytes.len()]).unwrap();
    let tick_z = vec![0u8; 16];
    let d_ticks = gpu.upload_raw(&tick_z, &[16]).unwrap();

    let modes: [(&str, u32); 3] = [
        ("iu4_probe_dependent", 0),
        ("iu4_probe_independent", 0),
        ("iu4_probe_independent_lds", PROBE_LDS_BYTES),
    ];

    // Warmup each symbol once at mid config.
    for (name, lds) in &modes {
        let _ = launch_probe(gpu, name, 160, 1024, &d_a, &d_b, &d_cs, &d_ticks, *lds);
    }

    let mut results: Vec<ProbeResult> = Vec::new();
    eprintln!("\n=== A0 probe sweep (event + s_memrealtime) ===");
    eprintln!("RTC timebase assumed {RTC_HZ} Hz (AMD wall-clock). Cycle claims use this; if unsupported, event rates remain authoritative.");
    eprintln!(
        "{:<28} {:>5} {:>5} {:>10} {:>10} {:>14} {:>10}",
        "mode", "grid", "loop", "event_us", "rtc_us", "WMMAs/us/CU", "cs_or"
    );

    for (name, lds) in &modes {
        for &g in &grids {
            for &lp in &loops {
                // 3 paired samples; take median event time
                let mut us_samples = Vec::with_capacity(3);
                let mut rtc_us = 0.0f64;
                let mut last_t0 = 0u64;
                let mut last_t1 = 0u64;
                for _ in 0..3 {
                    let (us, t0, t1) =
                        launch_probe(gpu, name, g, lp, &d_a, &d_b, &d_cs, &d_ticks, *lds);
                    us_samples.push(us);
                    last_t0 = t0;
                    last_t1 = t1;
                    if t1 >= t0 && t0 != 0 {
                        rtc_us = (t1 - t0) as f64 / RTC_HZ * 1e6;
                    }
                }
                let event_us = median_f64(&mut us_samples);
                let tw = total_wmmas(g, lp);
                let rate = tw / event_us.max(1e-9) / CU_TARGET;
                // Checksum OR over a few words so the host observes liveness
                let mut cs_sample = vec![0u8; 64];
                if gpu.hip.memcpy_dtoh(&mut cs_sample, &d_cs.buf).is_err() {
                    cs_sample.fill(0);
                }
                let mut cs_or = 0i32;
                for chunk in cs_sample.chunks_exact(4) {
                    cs_or |= i32::from_le_bytes(chunk.try_into().unwrap());
                }
                eprintln!(
                    "{:<28} {:>5} {:>5} {:>10.2} {:>10.2} {:>14.3} {:>10}  ticks=[{last_t0},{last_t1}]",
                    name, g, lp, event_us, rtc_us, rate, cs_or
                );
                results.push(ProbeResult {
                    name,
                    grid: g,
                    loops: lp,
                    event_us,
                    rtc_us,
                    wmma_per_us_cu: rate,
                    checksum_or: cs_or,
                });
            }
        }
    }

    // Ratios at plateau candidate grid160 / loop4096
    let pick = |mode: &str, g: u32, lp: i32| -> Option<&ProbeResult> {
        results
            .iter()
            .find(|r| r.name == mode && r.grid == g && r.loops == lp)
    };
    let dep = pick("iu4_probe_dependent", 160, 4096);
    let ind = pick("iu4_probe_independent", 160, 4096);
    let lds = pick("iu4_probe_independent_lds", 160, 4096);
    if let (Some(d), Some(i), Some(l)) = (dep, ind, lds) {
        let dep_ind = d.wmma_per_us_cu / i.wmma_per_us_cu.max(1e-30);
        let lds_co = l.wmma_per_us_cu / i.wmma_per_us_cu.max(1e-30);
        eprintln!("\n=== A0 ratios @ grid160 loop4096 (event-based WMMAs/us/CU) ===");
        eprintln!(
            "dependent rate = {:.3}  independent = {:.3}  independent+LDS = {:.3}",
            d.wmma_per_us_cu, i.wmma_per_us_cu, l.wmma_per_us_cu
        );
        eprintln!("dependent/independent ratio = {dep_ind:.4}");
        eprintln!("LDS co-issue ratio (ind+LDS)/independent = {lds_co:.4}");
        eprintln!(
            "checksum_or dep={:#x} ind={:#x} lds={:#x} (nonzero => loop live)",
            d.checksum_or as u32, i.checksum_or as u32, l.checksum_or as u32
        );
        // Wave-level WMMA instruction count (analytical)
        eprintln!(
            "issued wave-level WMMAs @this config = {:.0}  (grid*8*loops*4; not /32)",
            total_wmmas(160, 4096)
        );
        if d.rtc_us > 0.0 {
            eprintln!(
                "RTC-derived dep us={:.2} (timebase {} Hz); cycle claims use RTC if ticks nonzero",
                d.rtc_us, RTC_HZ as u64
            );
        } else {
            eprintln!("RTC ticks unavailable/zero — cycle claims UNKNOWN; event rates retained");
        }
    }

    // Plateau: grid160 vs 320 and loop4096 vs 8192 within 5% on independent rate
    eprintln!("\n=== A0 plateau check (independent mode, ≤5%) ===");
    let mut plateau_ok = true;
    for mode in [
        "iu4_probe_dependent",
        "iu4_probe_independent",
        "iu4_probe_independent_lds",
    ] {
        let a = pick(mode, 160, 4096).map(|r| r.wmma_per_us_cu);
        let b = pick(mode, 320, 4096).map(|r| r.wmma_per_us_cu);
        let c = pick(mode, 160, 8192).map(|r| r.wmma_per_us_cu);
        if let (Some(a), Some(b), Some(c)) = (a, b, c) {
            let gdiff = pct_diff(a, b);
            let ldiff = pct_diff(a, c);
            let ok = gdiff <= 5.0 && ldiff <= 5.0;
            plateau_ok &= ok;
            eprintln!(
                "{mode}: grid160/320 diff={gdiff:.2}%  loop4096/8192 diff={ldiff:.2}%  => {}",
                if ok { "PLATEAU" } else { "NO-PLATEAU" }
            );
        }
    }
    if plateau_ok {
        eprintln!("A0 plateau: ACCEPT (all modes ≤5%)");
    } else {
        eprintln!("A0 plateau: INCONCLUSIVE (do not select redesign from this run)");
    }
}

fn run_attribution(gpu: &mut Gpu) {
    let shapes: [(usize, usize, usize); 2] = [(17408, 5120, 512), (5120, 17408, 512)];
    let syms = [
        "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_a0_prod",
        "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_a0_noload",
        "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_a0_nofold",
    ];

    eprintln!("\n=== A0 attribution twins (100 interleaved event samples) ===");
    for &(m, k, n) in &shapes {
        eprintln!("\n-- shape (M,K,N)=({m},{k},{n}) --");
        let t_pack = Instant::now();
        let a_h = pack_mq4g256v2(m, k, 0xA0A0);
        let xq_h = pack_i4_xq(k, n, 0xB1B1);
        eprintln!(
            "packed A={} B Xq={} B in {:.2}s",
            a_h.len(),
            xq_h.len(),
            t_pack.elapsed().as_secs_f64()
        );
        let d_a = gpu.upload_raw(&a_h, &[a_h.len()]).unwrap();
        let d_xq = gpu.upload_raw(&xq_h, &[xq_h.len()]).unwrap();
        // Three Y buffers so arms do not contend on the same write set mid-interleave.
        let y_bytes = n * m * 4;
        let z = vec![0u8; y_bytes];
        let d_y_prod = gpu.upload_raw(&z, &[y_bytes]).unwrap();
        let d_y_noload = gpu.upload_raw(&z, &[y_bytes]).unwrap();
        let d_y_nofold = gpu.upload_raw(&z, &[y_bytes]).unwrap();
        let ys = [&d_y_prod, &d_y_noload, &d_y_nofold];

        // Warmup
        for _ in 0..ATTR_WARMUP {
            for (si, sym) in syms.iter().enumerate() {
                let _ = launch_twin(gpu, sym, &d_a, &d_xq, ys[si], m as i32, k as i32, n as i32);
            }
        }

        let mut samples: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for _ in 0..ATTR_ITERS {
            for (si, sym) in syms.iter().enumerate() {
                let us = launch_twin(gpu, sym, &d_a, &d_xq, ys[si], m as i32, k as i32, n as i32);
                samples[si].push(us);
            }
        }

        let mut med = [0.0f64; 3];
        for i in 0..3 {
            med[i] = median_f64(&mut samples[i]);
            let mn = samples[i].iter().cloned().fold(f64::INFINITY, f64::min);
            let mx = samples[i]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "  {:<64} median_us={:.2}  min={:.2} max={:.2}",
                syms[i], med[i], mn, mx
            );
        }
        let e = (med[0] - med[1]) / med[0].max(1e-30);
        let fold_frac = (med[0] - med[2]) / med[0].max(1e-30);
        eprintln!(
            "  E = (production - no_load)/production = {e:.4}  ({:.2}%)",
            e * 100.0
        );
        eprintln!(
            "  fold bound (production - no_fold)/production = {fold_frac:.4}  ({:.2}%)  [register-confounded bound]",
            fold_frac * 100.0
        );
        if e >= 0.25 {
            eprintln!("  W2 eligibility screen on this shape: E≥25% PASS (still needs object budgets)");
        } else {
            eprintln!("  W2 eligibility screen on this shape: E≥25% FAIL");
        }
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!(
        "tmp_halo_iu4_calibrate arch={}  (A0 IU4 WMMA rate + attribution)",
        gpu.arch
    );
    if !matches!(
        gpu.arch.as_str(),
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" | "gfx1150" | "gfx1151"
    ) {
        eprintln!("skip: need gfx11 family for iu4 WMMA, got {}", gpu.arch);
        return;
    }

    // Ensure a non-null stream so event_record attaches cleanly.
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create().expect("stream"));
    }

    ensure_probes(&mut gpu);
    ensure_twins(&mut gpu);

    eprintln!(
        "\nA0: first JIT done. Parent must read code-object metadata for symbols listed in the file header before trusting any timing below."
    );
    eprintln!("HIPFIRE_KERNEL_CACHE entries live under the process cache dir (see env).");

    run_probes(&mut gpu);
    run_attribution(&mut gpu);

    eprintln!("\nA0 complete.");
}
