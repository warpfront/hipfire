// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniil Markevich
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

// Production MQ4G256V2 (qt=44) MMQ residual GEMM for gfx1100/gfx1151.
//
// Selected as the fast path when `batch >= 128 && batch % 128 == 0`
// (see `crates/rdna-compute/src/gemm.rs:17860-17909` and the qkv/qkvza/
// gate_up/residual call sites); otherwise the WMMA residual path runs.
// Applies the scale per 128-K half (`kt < 8` select), not a per-256 V1-ism.
//
// This mirrors the important parts of llama.cpp's AMD MMQ path:
// - pre-quantize the activation matrix into block_q8_1_mmq DS4 layout
// - process 128 output rows x 128 batch columns per workgroup
// - stage both HFQ4 weights and Q8_1 activations in LDS
// - use i8 WMMA for the dot product and apply HFQ4 scale/zero via Q8_1
//   scale/sum correction.

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define HIPFIRE_RDNA3 1
#else
#define HIPFIRE_RDNA3 0
#endif

#define MMQ_X 128
#define MMQ_Y 128
#define MMQ_NWARPS 8
#define WARP_SIZE 32
#define QK8_1 32
#define QI8_1 8
#define MMQ_TILE_NE_K 32
#define MMQ_TILE_Y_K 36
#define MMQ_TILE_X_K 76

struct block_q8_1_mmq {
    // ds4 layout: per-32-lane (d, sum) in the default (per-32) path. The
    // V2 (per-128, x128) path replicates ONE half-wide (d_half, s_half) to
    // ALL FOUR slots, so any legacy y_dm slot read stays valid while the
    // x128 consumer reads slot 0 only.
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};

static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

// Q8_1 activation quantize: arch-portable, used by RDNA3 (this file's WMMA
// MMQ kernel) and gfx906 (the dp4a MMQ kernel in
// kernels/src/gemm_mq4g256v2_residual_mmq_gfx906_*.hip). The body uses only
// width=32 shuffles, which are wave64-safe: each 32-lane half-wave reduces
// independently within the wave, and the kernel's amax+sum are per-32-lane
// (one Q8_1 sub-block per 32 lanes). Block dim is (256, 1, 1); each thread
// processes 4 float inputs. Per-32-lane group computes amax + sum over its
// 128-element Q8_1 sub-block.
extern "C" __global__ void quantize_q8_1_mmq_ds4(
    const float* __restrict__ X,
    block_q8_1_mmq* __restrict__ Y,
    int K,
    int N
) {
    const int token = blockIdx.y;
    const int i0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (token >= N || i0 >= K) return;

    const float4 xi = (i0 + 3 < K)
        ? *(const float4*)(X + (long long)token * K + i0)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, WARP_SIZE));
    }

    float sum = xi.x + xi.y + xi.z + xi.w;
    for (int offset = 4; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, WARP_SIZE);
    }

    const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    char4 q;
    q.x = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.x * id)));
    q.y = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.y * id)));
    q.z = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.z * id)));
    q.w = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.w * id)));

    const int kblock = i0 / 128;
    const int iqs = i0 & 127;
    block_q8_1_mmq* out = Y + (long long)kblock * N + token;
    ((char4*)out->qs)[iqs / 4] = q;

    if ((iqs & 31) == 0) {
        out->ds4[iqs / 32] = make_half2((_Float16)d, (_Float16)sum);
    }
}

// Q8_1 activation quantize at per-128 granularity for the V2 (X128)
// MMQ path: same (X, Y, K, N) signature and block_q8_1_mmq layout as
// quantize_q8_1_mmq_ds4, but amax+sum span the full 128-element K block
// (all 32 lanes of the wave segment, offsets 16..1) instead of one
// 32-element sub-block per 8 lanes (offsets 4..1). qs bytes use the
// half-wide id_half; ALL FOUR ds4 slots replicate (d_half, s_half).
// Coarser X quant than the default path by construction: expect relL2
// ~1e-2 vs the per-32 path, <=1e-5 vs a per-128 CPU oracle on the same X.
extern "C" __global__ void quantize_q8_1_mmq_ds4_x128(
    const float* __restrict__ X,
    block_q8_1_mmq* __restrict__ Y,
    int K,
    int N
) {
    const int token = blockIdx.y;
    const int i0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (token >= N || i0 >= K) return;

    const float4 xi = (i0 + 3 < K)
        ? *(const float4*)(X + (long long)token * K + i0)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, WARP_SIZE));
    }

    float sum = xi.x + xi.y + xi.z + xi.w;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, WARP_SIZE);
    }

    const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    char4 q;
    q.x = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.x * id)));
    q.y = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.y * id)));
    q.z = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.z * id)));
    q.w = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.w * id)));

    const int kblock = i0 / 128;
    const int iqs = i0 & 127;
    block_q8_1_mmq* out = Y + (long long)kblock * N + token;
    ((char4*)out->qs)[iqs / 4] = q;

    if ((iqs & 127) == 0) {
        const half2 ds = make_half2((_Float16)d, (_Float16)sum);
        #pragma unroll
        for (int l = 0; l < 4; ++l)
            out->ds4[l] = ds;
    }
}

#if !HIPFIRE_RDNA3

// MMQ compute kernels — RDNA3 only (use WMMA i8 builtin). gfx906 dp4a
// implementation lives in gemm_mq4g256v2_residual_mmq.gfx906.hip.
extern "C" __global__ void gemm_mq4g256v2_residual_mmq(
    const char*,
    const block_q8_1_mmq*,
    float*,
    int,
    int,
    int,
    int
) {}

extern "C" __global__ void gemm_mq4g256v2_residual_mmq_full_add(
    const char*,
    const block_q8_1_mmq*,
    float*,
    int,
    int,
    int,
    int
) {}

extern "C" __global__ void gemm_mq4g256v2_residual_mmq_full_set(
    const char*,
    const block_q8_1_mmq*,
    float*,
    int,
    int,
    int,
    int
) {}
#ifdef HIPFIRE_MMQ_LUT
// LUT-arg twin stubs (same file, distinct symbols). Real bodies are RDNA3-only.
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_lloyd(
    const char*, const block_q8_1_mmq*, float*, int, int, int, int,
    unsigned int, unsigned int, unsigned int, unsigned int
) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_x128_lloyd(
    const char*, const block_q8_1_mmq*, float*, int, int, int, int,
    unsigned int, unsigned int, unsigned int, unsigned int
) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_full_add_lloyd(
    const char*, const block_q8_1_mmq*, float*, int, int, int, int,
    unsigned int, unsigned int, unsigned int, unsigned int
) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_full_set_lloyd(
    const char*, const block_q8_1_mmq*, float*, int, int, int, int,
    unsigned int, unsigned int, unsigned int, unsigned int
) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_full_add_x128_lloyd(
    const char*, const block_q8_1_mmq*, float*, int, int, int, int,
    unsigned int, unsigned int, unsigned int, unsigned int
) {}
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_full_set_x128_lloyd(
    const char*, const block_q8_1_mmq*, float*, int, int, int, int,
    unsigned int, unsigned int, unsigned int, unsigned int
) {}
#endif

#else

template <int nbytes>
static __device__ __forceinline__ void memcpy_aligned(void* __restrict__ dst, const void* __restrict__ src) {
    if constexpr (nbytes == 16) {
        ((int4*)dst)[0] = ((const int4*)src)[0];
    } else if constexpr (nbytes == 8) {
        ((int2*)dst)[0] = ((const int2*)src)[0];
    } else if constexpr (nbytes == 4) {
        ((int*)dst)[0] = ((const int*)src)[0];
    }
}

enum data_layout {
    DATA_LAYOUT_I_MAJOR = 0,
    DATA_LAYOUT_J_MAJOR = 10,
    DATA_LAYOUT_I_MAJOR_MIRRORED = 20,
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
#if HIPFIRE_RDNA3
        if constexpr (I == 16 && J == 16) {
            return 2 * l + (threadIdx.x / 16);
        } else {
            return l;
        }
#else
        return ne * (threadIdx.x / 16) + l;
#endif
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

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR_MIRRORED> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I * J / 32 * 2;
    T x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int) {
        return threadIdx.x % 16;
    }

    static __device__ __forceinline__ int get_j(const int l) {
        return l;
    }
};

template <typename T, data_layout dl>
static __device__ __forceinline__ void load_ldmatrix_16x8(tile<16, 8, T, dl>& t, const T* __restrict__ xs0, const int stride) {
#if HIPFIRE_RDNA3
    static_assert(dl == DATA_LAYOUT_I_MAJOR_MIRRORED, "bad layout");
    static_assert(sizeof(t.x) == 32, "bad mirrored tile size");
    memcpy_aligned<16>(t.x + 0, xs0 + t.get_i(0) * stride + 0);
    memcpy_aligned<16>(t.x + 4, xs0 + t.get_i(0) * stride + 4);
#else
    static_assert(dl == DATA_LAYOUT_I_MAJOR, "bad layout");
    static_assert(sizeof(t.x) == 16, "bad tile size");
    memcpy_aligned<16>(t.x, xs0 + t.get_i(0) * stride + t.get_j(0));
#endif
}

template <data_layout dl_d, data_layout dl_ab>
static __device__ __forceinline__ void mma_i8(
    tile<16, 16, int, dl_d>& D,
    const tile<16, 8, int, dl_ab>& A,
    const tile<16, 8, int, dl_ab>& B
) {
    using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
    using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
    int32x8_t* acc = (int32x8_t*)D.x;
    const int32x4_t* a_vec = (const int32x4_t*)A.x;
    const int32x4_t* b_vec = (const int32x4_t*)B.x;
    acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[0], true, b_vec[0], acc[0], true);
    acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[1], true, b_vec[1], acc[0], true);
}

// quantize_q8_1_mmq_ds4 hoisted above the HIPFIRE_RDNA3 split (line ~38)
// because the body is wave-portable. Phase 0 §3.5 verified.

#ifdef HIPFIRE_MMQ_LUT
// MQ4G256V2-Lloyd (qt=52) MMQ-LUT: 16-entry signed i8 codebook as 4 kernarg
// dwords. Pattern from gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip:76-95.
// Outcome A (signed weight operand): mma_i8 calls
// __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, A, true, B, C, true) —
// both A_sign and B_sign are true, so C16 codes in [-120,120] feed the A
// (weight) operand directly. sc' = sc/16; zp' already folded in headers via
// apply_lloyd_centering. i32 C_frag headroom: worst |C| = 128·240·127 =
// 3,901,440 ≪ 2³¹ (X128 half; per-32 is 4× smaller). No overflow handling.
__device__ static inline unsigned int mq4_spread4(unsigned int x16) {
    const unsigned int a = (x16 & 0x00FFu) | ((x16 & 0xFF00u) << 8);
    return (a & 0x000F000Fu) | ((a & 0x00F000F0u) << 4);
}
__device__ static inline unsigned int mq4_lut8(unsigned int sel,
    unsigned int C0, unsigned int C1, unsigned int C2, unsigned int C3) {
    // v_perm_b32(hi, lo, s): result byte i = {lo,hi} byte s_i for s_i in 0..7.
    const unsigned int lo = __builtin_amdgcn_perm(C1, C0, sel & 0x07070707u);
    const unsigned int hi = __builtin_amdgcn_perm(C3, C2, sel & 0x07070707u);
    const unsigned int m = ((sel >> 3) & 0x01010101u) * 0xFFu;
    return (hi & m) | (lo & ~m);
}
#endif
template <bool FULL>
static __device__ __forceinline__ void load_mq4v2_tile(
    const char* __restrict__ A,
    int* __restrict__ tile_x,
    int row0,
    int kb,
    int M,
    int groups_per_row
#ifdef HIPFIRE_MMQ_LUT
    , unsigned int C0, unsigned int C1, unsigned int C2, unsigned int C3
#endif
) {
    int* x_qs = tile_x;
    half2* x_dm = (half2*)(x_qs + 2 * MMQ_TILE_NE_K);

    constexpr int threads_per_row = 32;
    constexpr int rows_per_warp = 1;
    const int txi = threadIdx.x;

    for (int i0 = 0; i0 < MMQ_Y; i0 += rows_per_warp * MMQ_NWARPS) {
        const int i = i0 + threadIdx.y;
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const char* gp = A + ((long long)row * groups_per_row + kb) * 136;
        const unsigned int qs0 = *(const unsigned int*)(gp + 8 + txi * 4);

#ifdef HIPFIRE_MMQ_LUT
        // One v_perm byte lookup per 8 codes (spread4 + lut8 × 2).
        x_qs[i * MMQ_TILE_X_K + 2 * txi + 0] =
            (int)mq4_lut8(mq4_spread4(qs0 & 0xFFFFu), C0, C1, C2, C3);
        x_qs[i * MMQ_TILE_X_K + 2 * txi + 1] =
            (int)mq4_lut8(mq4_spread4(qs0 >> 16), C0, C1, C2, C3);
#else
        const unsigned int q0 = (qs0 >>  0) & 0xFu;
        const unsigned int q1 = (qs0 >>  4) & 0xFu;
        const unsigned int q2 = (qs0 >>  8) & 0xFu;
        const unsigned int q3 = (qs0 >> 12) & 0xFu;
        const unsigned int q4 = (qs0 >> 16) & 0xFu;
        const unsigned int q5 = (qs0 >> 20) & 0xFu;
        const unsigned int q6 = (qs0 >> 24) & 0xFu;
        const unsigned int q7 = (qs0 >> 28) & 0xFu;
        x_qs[i * MMQ_TILE_X_K + 2 * txi + 0] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
        x_qs[i * MMQ_TILE_X_K + 2 * txi + 1] = q4 | (q5 << 8) | (q6 << 16) | (q7 << 24);
#endif
    }

    // MQ4V2 dual-fp16 headers: ksc 0 owns K 0..127 (dm slots 0..3),
    // ksc 1 owns K 128..255 (dm slots 4..7).
    constexpr int rows_per_warp_meta = 16;
    for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS * rows_per_warp_meta) {
        const int i = i0 + threadIdx.y * rows_per_warp_meta + threadIdx.x / 2;
        if (i < MMQ_Y) {
            const int row = FULL ? row0 + i : min(row0 + i, M - 1);
            const char* gp = A + ((long long)row * groups_per_row + kb) * 136;
            const int ksc = threadIdx.x & 1;
            const unsigned int hs = *(const unsigned int*)(gp + 4 * ksc);
#ifdef HIPFIRE_MMQ_LUT
            // sc' = sc/16: codes are 16·(L−7.5); zp' already in headers.
            // i32 headroom |C| ≤ 128·240·127 = 3,901,440 ≪ 2³¹ (see helpers).
            const float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu))) * 0.0625f;
#else
            const float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
#endif
            const float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
            const half2 dm = make_half2((_Float16)sc, (_Float16)zp);
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                x_dm[i * MMQ_TILE_X_K + 4 * ksc + l] = dm;
        }
    }
}

static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(
    const int* __restrict__ x,
    const int* __restrict__ y,
    float* __restrict__ sum,
    int k00
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;

    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int* x_qs = x;
    const half2* x_dm = (const half2*)(x_qs + 2 * MMQ_TILE_NE_K);
    const int* y_qs = y + 4;
    const half2* y_dm = (const half2*)y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
        const int k0 = k00 + k01;
        tile_A A_frag[ntx];
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix_16x8(A_frag[n], x_qs + (i0 + n * tile_A::I) * MMQ_TILE_X_K + k0, MMQ_TILE_X_K);
        }

        #pragma unroll
        for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
            tile_B B_frag;
            load_ldmatrix_16x8(B_frag, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
            const int j = j0 + tile_C::get_j(0);
            const float2 dsB = __half22float2(y_dm[j * MMQ_TILE_Y_K + k01 / QI8_1]);

            #pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C_frag;
                mma_i8(C_frag, A_frag[n], B_frag);

                #pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                    const float2 dmA = __half22float2(x_dm[i * MMQ_TILE_X_K + k0 / QI8_1]);
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += dmA.x * dsB.x * (float)C_frag.x[l];
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += dmA.y * dsB.y;
                }
            }
        }
    }
}

// V2 (X128) dot: the x128 prelude stores ONE (d, s) per 128-K half
// (replicated to all ds4 slots), so the 8 i8-WMMA dots of the half
// accumulate in ONE i32 C_frag and the float correction applies once:
//
//   C_l   = sum_{b<4} dot_b(l)            (i32 over the 128-K half)
//   sum_l += dmA.x * d * C_l + dmA.y * s  (one correction per half)
//
// Numerics differ from the default per-32 path by construction (coarser X
// quant); the oracle is per-128 quantization (expect <=1e-5 vs that
// oracle, ~1e-2 vs the per-32 path). Opt-in via HIPFIRE_GFX11_MMQ_X128;
// promotion to default only if WT2 KLD <= current+0.0005.
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma_x128(
    const int* __restrict__ x,
    const int* __restrict__ y,
    float* __restrict__ sum,
    int k00
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;

    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int* x_qs = x;
    const half2* x_dm = (const half2*)(x_qs + 2 * MMQ_TILE_NE_K);
    const int* y_qs = y + 4;
    const half2* y_dm = (const half2*)y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
    const int j_lane = tile_C::get_j(0);

    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            tile_C C_frag;
            #pragma nounroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
                const int k0 = k00 + k01;
                tile_A A_frag;
                load_ldmatrix_16x8(A_frag, x_qs + (i0 + n * tile_A::I) * MMQ_TILE_X_K + k0, MMQ_TILE_X_K);
                tile_B B_frag;
                load_ldmatrix_16x8(B_frag, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
                mma_i8(C_frag, A_frag, B_frag);
            }
            const float2 ds = __half22float2(y_dm[(j0 + j_lane) * MMQ_TILE_Y_K + 0]);
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                const float2 dmA = __half22float2(x_dm[i * MMQ_TILE_X_K + k00 / QI8_1]);
                sum[(j0 / tile_C::J + n) * tile_C::ne + l] +=
                    dmA.x * ds.x * (float)C_frag.x[l] + dmA.y * ds.y;
            }
        }
    }
}

template <bool X128>
static __device__ __forceinline__ void gemm_mq4g256v2_residual_mmq_base_body(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
#ifdef HIPFIRE_MMQ_LUT
    , unsigned int C0, unsigned int C1, unsigned int C2, unsigned int C3
#endif
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;
    if (row0 >= M || col0 >= N) return;

    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;

    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;

    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};

    for (int kb = 0; kb < groups_per_row; ++kb) {
#ifdef HIPFIRE_MMQ_LUT
        load_mq4v2_tile<false>(A, tile_x, row0, kb, M, groups_per_row, C0, C1, C2, C3);
#else
        load_mq4v2_tile<false>(A, tile_x, row0, kb, M, groups_per_row);
#endif

        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            const int j = l / MMQ_TILE_Y_K;
            tile_y[l] = (col0 + j < N) ? by0[l] : 0;
        }
        __syncthreads();
        if constexpr (X128) vec_dot_q8_1_q8_1_mma_x128(tile_x, tile_y, sum, 0);
        else vec_dot_q8_1_q8_1_mma(tile_x, tile_y, sum, 0);
        __syncthreads();

        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                const int j = l / MMQ_TILE_Y_K;
                tile_y[l] = (col0 + j < N) ? by1[l] : 0;
            }
            __syncthreads();
            if constexpr (X128) vec_dot_q8_1_q8_1_mma_x128(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            else vec_dot_q8_1_q8_1_mma(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();
        }
    }

    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if (row0 + i < M && col0 + j < N) {
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

#ifdef HIPFIRE_MMQ_LUT
extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_lloyd(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add,
    unsigned int C0, unsigned int C1, unsigned int C2, unsigned int C3
) {
    gemm_mq4g256v2_residual_mmq_base_body<false>(A, Xq, Y, M, K, N, add, C0, C1, C2, C3);
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_x128_lloyd(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add,
    unsigned int C0, unsigned int C1, unsigned int C2, unsigned int C3
) {
    gemm_mq4g256v2_residual_mmq_base_body<true>(A, Xq, Y, M, K, N, add, C0, C1, C2, C3);
}
#else
extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    gemm_mq4g256v2_residual_mmq_base_body<false>(A, Xq, Y, M, K, N, add);
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_x128(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    gemm_mq4g256v2_residual_mmq_base_body<true>(A, Xq, Y, M, K, N, add);
}
#endif

template <bool ADD, bool X128>
static __device__ __forceinline__ void gemm_mq4g256v2_residual_mmq_full_body(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N
#ifdef HIPFIRE_MMQ_LUT
    , unsigned int C0, unsigned int C1, unsigned int C2, unsigned int C3
#endif
) {
    const int row0 = blockIdx.x * MMQ_Y;
    const int col0 = blockIdx.y * MMQ_X;

    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;

    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + MMQ_X * MMQ_TILE_Y_K;

    float sum[MMQ_X * MMQ_Y / (MMQ_NWARPS * WARP_SIZE)] = {0.0f};

    for (int kb = 0; kb < groups_per_row; ++kb) {
#ifdef HIPFIRE_MMQ_LUT
        load_mq4v2_tile<true>(A, tile_x, row0, kb, M, groups_per_row, C0, C1, C2, C3);
#else
        load_mq4v2_tile<true>(A, tile_x, row0, kb, M, groups_per_row);
#endif

        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            tile_y[l] = by0[l];
        }
        __syncthreads();
        if constexpr (X128) vec_dot_q8_1_q8_1_mma_x128(tile_x, tile_y, sum, 0);
        else vec_dot_q8_1_q8_1_mma(tile_x, tile_y, sum, 0);
        __syncthreads();

        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                tile_y[l] = by1[l];
            }
            __syncthreads();
            if constexpr (X128) vec_dot_q8_1_q8_1_mma_x128(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            else vec_dot_q8_1_q8_1_mma(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();
        }
    }

    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                const float v = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
                if constexpr (ADD) {
                    *yp += v;
                } else {
                    *yp = v;
                }
            }
        }
    }
}

#ifdef HIPFIRE_MMQ_LUT
extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_full_add_lloyd(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int,
    unsigned int C0, unsigned int C1, unsigned int C2, unsigned int C3
) {
    gemm_mq4g256v2_residual_mmq_full_body<true, false>(A, Xq, Y, M, K, N, C0, C1, C2, C3);
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_full_set_lloyd(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int,
    unsigned int C0, unsigned int C1, unsigned int C2, unsigned int C3
) {
    gemm_mq4g256v2_residual_mmq_full_body<false, false>(A, Xq, Y, M, K, N, C0, C1, C2, C3);
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_full_add_x128_lloyd(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int,
    unsigned int C0, unsigned int C1, unsigned int C2, unsigned int C3
) {
    gemm_mq4g256v2_residual_mmq_full_body<true, true>(A, Xq, Y, M, K, N, C0, C1, C2, C3);
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_full_set_x128_lloyd(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int,
    unsigned int C0, unsigned int C1, unsigned int C2, unsigned int C3
) {
    gemm_mq4g256v2_residual_mmq_full_body<false, true>(A, Xq, Y, M, K, N, C0, C1, C2, C3);
}
#else
extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_full_add(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int
) {
    gemm_mq4g256v2_residual_mmq_full_body<true, false>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_full_set(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int
) {
    gemm_mq4g256v2_residual_mmq_full_body<false, false>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_full_add_x128(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int
) {
    gemm_mq4g256v2_residual_mmq_full_body<true, true>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(MMQ_NWARPS * WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_full_set_x128(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int
) {
    gemm_mq4g256v2_residual_mmq_full_body<false, true>(A, Xq, Y, M, K, N);
}
#endif

#endif
