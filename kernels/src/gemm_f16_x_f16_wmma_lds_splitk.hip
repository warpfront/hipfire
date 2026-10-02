// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// LDS-staged F16 WMMA GEMM with an optional deterministic K split, for the
// Qwen4 BF16 projections that run through a model-lifetime F16 weight shadow
// (Gpu::gemm_bf16_xf16_f16_wmma): the hyper-connection down projection
// (M = 320, K = 4 * hidden), the hyper-write gate (M = 4) and the PLE / dense
// BF16 projections.  Loaded on gfx1201 and gfx1151 only (exact arch, see
// Gpu::gemm_f16_x_f16_wmma_lds_splitk); every other arch errors at compile.
//
//   A : [M, K] F16 row-major  (weight shadow)
//   X : [B, K] F16 row-major  (activations)
//   Y : [B, M] F32 row-major, Y[b, m] = sum_k A[m, k] * X[b, k]
//
// Same staging and tile structure as gemm_f16_x_f16_wmma_lds256.hip (LDS
// tile-major [kt][row][16], one wave per WM x WN output patch, K staged KS
// elements per barrier pair).  The WMMA fragment map is per arch:
//   gfx1201: half8 operands and the `_w32_gfx12` builtin, lane l holds row
//     (l & 15), K elements [8 * (l >> 4), +8) of each 16-element substep; the
//     accumulator is contiguous per lane-half, acc[e] = C[8 * (l >> 4) + e]
//     [l & 15] (row = M, column = B), so a lane's eight outputs are eight
//     consecutive floats of one Y row and store as two float4.  (The gfx11
//     `_w32` builtin needs wmma-256b-insts and does not compile there.)
//   gfx1151: half16 operands (both lane halves carry the full substep) and
//     the gfx11 `_w32` builtin; acc[e] = C[2 * e + (l >> 4)][l & 15].  Its
//     LDS barrier is lgkmcnt + s_barrier, as in the lds256 file on gfx1151.
//
// K ORDER: ascending 16-element substeps inside each K split.  With SPLIT > 1
// blockIdx.z owns K range [z * K / SPLIT, (z + 1) * K / SPLIT) and writes its
// partial sums to Y + z * B * M; qwen4_wmma_lds_reduce then adds the SPLIT
// partials in ascending z order, so the result is deterministic.
//
// PRECONDITION: K % (KS * SPLIT) == 0 (the host asserts it).  M and B tails
// clamp the staged row to the last valid one and discard at the store; no
// thread returns after the block-uniform origin check.
//
// Grid:  [ceil(M / BM), ceil(B / BN), SPLIT]
// Block: (BM / (16 * WP)) * (BN / (16 * WQ)) * 32

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

static __device__ __forceinline__ float sk_bf16_scaled_add(float r, float v, float s) {
    const hip_bfloat16 lhs = static_cast<hip_bfloat16>(r);
    const hip_bfloat16 rhs = static_cast<hip_bfloat16>(v);
    const hip_bfloat16 gate = static_cast<hip_bfloat16>(s);
    const hip_bfloat16 product = static_cast<hip_bfloat16>(
        static_cast<float>(gate) * static_cast<float>(rhs));
    return static_cast<float>(static_cast<hip_bfloat16>(
        static_cast<float>(lhs) + static_cast<float>(product)));
}

typedef _Float16 __attribute__((ext_vector_type(16))) sk_half16_t;
typedef _Float16 __attribute__((ext_vector_type(8)))  sk_half8_t;
typedef float    __attribute__((ext_vector_type(8)))  sk_float8_t;

#if defined(__gfx1201__)
typedef sk_half8_t sk_frag_t;
#define SK_FRAG_OFFSET(kh) ((kh) * 8)
#define SK_WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c)
#define SK_BARRIER() __syncthreads()
#elif defined(__gfx1151__)
typedef sk_half16_t sk_frag_t;
#define SK_FRAG_OFFSET(kh) 0
#define SK_WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)
#define SK_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
#else
#error "gemm_f16_x_f16_wmma_lds_splitk.hip: gfx1201 and gfx1151 only"
#endif

template <int KT>
__device__ __forceinline__ const sk_half16_t* sk_src(
    const _Float16* __restrict__ S, int rows, int ld, int row_block, int k0, int v
) {
    const int row = v / KT;
    const int kt  = v % KT;
    const int r    = row_block + row;
    const int safe = (r < rows) ? r : (rows - 1);
    return (const sk_half16_t*)(S + (long long)safe * ld + k0 + kt * 16);
}

template <int ROWS, int KT>
__device__ __forceinline__ sk_half16_t* sk_dst(_Float16* lds, int v) {
    const int row = v / KT;
    const int kt  = v % KT;
    return (sk_half16_t*)(&lds[(kt * ROWS + row) * 16]);
}

// One K stage of WMMA out of LDS: the only place the accumulation order is
// written down.  `kh` is the lane half (on gfx1201, which 8 of the substep's
// 16 K elements this lane contributes).
template <int BM, int BN, int WP, int WQ, int KT>
__device__ __forceinline__ void sk_wmma_stage(
    const _Float16* a_lds, const _Float16* x_lds,
    int wave_m, int wave_n, int ml, int kh, sk_float8_t* acc
) {
    constexpr int WROWS = 16 * WP;
    constexpr int WCOLS = 16 * WQ;
    #pragma unroll
    for (int kt = 0; kt < KT; kt++) {
        sk_frag_t a[WP];
        #pragma unroll
        for (int i = 0; i < WP; i++) {
            a[i] = *(const sk_frag_t*)(
                &a_lds[(kt * BM + wave_m * WROWS + i * 16 + ml) * 16 + SK_FRAG_OFFSET(kh)]);
        }
        #pragma unroll
        for (int j = 0; j < WQ; j++) {
            const sk_frag_t b = *(const sk_frag_t*)(
                &x_lds[(kt * BN + wave_n * WCOLS + j * 16 + ml) * 16 + SK_FRAG_OFFSET(kh)]);
            #pragma unroll
            for (int i = 0; i < WP; i++) {
                acc[i * WQ + j] = SK_WMMA(a[i], b, acc[i * WQ + j]);
            }
        }
    }
}

template <int BM, int BN, int WP, int WQ, int KS, bool PIPE, int SPLIT>
__device__ __forceinline__ void sk_lds_tile(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ X,
    float*          __restrict__ Y,
    int M, int K, int B,
    _Float16* a_lds, _Float16* x_lds
) {
    constexpr int KT     = KS / 16;
    constexpr int WROWS  = 16 * WP;
    constexpr int WCOLS  = 16 * WQ;
    constexpr int WAVE_M = BM / WROWS;
    constexpr int WAVE_N = BN / WCOLS;
    constexpr int NT     = WAVE_M * WAVE_N * 32;
    constexpr int A_VEC  = BM * KT;
    constexpr int X_VEC  = BN * KT;
    constexpr int A_PASS = (A_VEC + NT - 1) / NT;
    constexpr int X_PASS = (X_VEC + NT - 1) / NT;

    static_assert(KS == 32 || KS == 64, "KS must be 32 or 64");
    static_assert(BM % WROWS == 0 && BN % WCOLS == 0, "tile must divide into waves");
    static_assert(NT <= 1024, "block exceeds the 1024-thread limit");
    static_assert((BM + BN) * KS * 2 <= 65536, "LDS exceeds 64 KB per workgroup");

    const int m_block = blockIdx.x * BM;
    const int n_block = blockIdx.y * BN;
    if (m_block >= M || n_block >= B) return;

    const int k_len   = K / SPLIT;
    const int k_begin = (int)blockIdx.z * k_len;
    const int k_end   = k_begin + k_len;

    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int wave_id = tid >> 5;
    const int ml      = lane & 15;
    const int kh      = lane >> 4;

    const int wave_m = wave_id % WAVE_M;
    const int wave_n = wave_id / WAVE_M;
    const int m_wave = m_block + wave_m * WROWS;
    const int n_wave = n_block + wave_n * WCOLS;

    sk_float8_t acc[WP * WQ];
    #pragma unroll
    for (int i = 0; i < WP * WQ; i++) {
        acc[i] = sk_float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    sk_half16_t a_reg[PIPE ? A_PASS : 1];
    sk_half16_t x_reg[PIPE ? X_PASS : 1];

    if constexpr (PIPE) {
        #pragma unroll
        for (int p = 0; p < A_PASS; p++) {
            const int v = tid + p * NT;
            if (A_VEC % NT != 0 && v >= A_VEC) continue;
            a_reg[p] = *sk_src<KT>(A, M, K, m_block, k_begin, v);
        }
        #pragma unroll
        for (int p = 0; p < X_PASS; p++) {
            const int v = tid + p * NT;
            if (X_VEC % NT != 0 && v >= X_VEC) continue;
            x_reg[p] = *sk_src<KT>(X, B, K, n_block, k_begin, v);
        }
        for (int k0 = k_begin; k0 < k_end; k0 += KS) {
            #pragma unroll
            for (int p = 0; p < A_PASS; p++) {
                const int v = tid + p * NT;
                if (A_VEC % NT != 0 && v >= A_VEC) continue;
                *sk_dst<BM, KT>(a_lds, v) = a_reg[p];
            }
            #pragma unroll
            for (int p = 0; p < X_PASS; p++) {
                const int v = tid + p * NT;
                if (X_VEC % NT != 0 && v >= X_VEC) continue;
                *sk_dst<BN, KT>(x_lds, v) = x_reg[p];
            }
            SK_BARRIER();
            const int k1 = k0 + KS;
            if (k1 < k_end) {
                #pragma unroll
                for (int p = 0; p < A_PASS; p++) {
                    const int v = tid + p * NT;
                    if (A_VEC % NT != 0 && v >= A_VEC) continue;
                    a_reg[p] = *sk_src<KT>(A, M, K, m_block, k1, v);
                }
                #pragma unroll
                for (int p = 0; p < X_PASS; p++) {
                    const int v = tid + p * NT;
                    if (X_VEC % NT != 0 && v >= X_VEC) continue;
                    x_reg[p] = *sk_src<KT>(X, B, K, n_block, k1, v);
                }
            }
            sk_wmma_stage<BM, BN, WP, WQ, KT>(a_lds, x_lds, wave_m, wave_n, ml, kh, acc);
            SK_BARRIER();
        }
    } else {
        for (int k0 = k_begin; k0 < k_end; k0 += KS) {
            #pragma unroll
            for (int p = 0; p < A_PASS; p++) {
                const int v = tid + p * NT;
                if (A_VEC % NT != 0 && v >= A_VEC) continue;
                *sk_dst<BM, KT>(a_lds, v) = *sk_src<KT>(A, M, K, m_block, k0, v);
            }
            #pragma unroll
            for (int p = 0; p < X_PASS; p++) {
                const int v = tid + p * NT;
                if (X_VEC % NT != 0 && v >= X_VEC) continue;
                *sk_dst<BN, KT>(x_lds, v) = *sk_src<KT>(X, B, K, n_block, k0, v);
            }
            SK_BARRIER();
            sk_wmma_stage<BM, BN, WP, WQ, KT>(a_lds, x_lds, wave_m, wave_n, ml, kh, acc);
            SK_BARRIER();
        }
    }

    float* Yz = Y + (long long)blockIdx.z * B * M;
#if defined(__gfx1201__)
    // Epilogue: lane (ml, kh) holds Y[n_wave + j*16 + ml][m_wave + i*16 + 8*kh + e].
    #pragma unroll
    for (int j = 0; j < WQ; j++) {
        const int out_b = n_wave + j * 16 + ml;
        if (out_b >= B) continue;
        float* yrow = Yz + (long long)out_b * M;
        #pragma unroll
        for (int i = 0; i < WP; i++) {
            const int m0 = m_wave + i * 16 + 8 * kh;
            const sk_float8_t v = acc[i * WQ + j];
            if (m0 + 8 <= M && (M & 3) == 0) {
                *(float4*)(yrow + m0)     = float4{v[0], v[1], v[2], v[3]};
                *(float4*)(yrow + m0 + 4) = float4{v[4], v[5], v[6], v[7]};
            } else {
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    if (m0 + e < M) yrow[m0 + e] = v[e];
                }
            }
        }
    }
#else
    // Epilogue: lane (ml, kh) holds Y[n_wave + j*16 + ml][m_wave + i*16 + 2*e + kh].
    #pragma unroll
    for (int j = 0; j < WQ; j++) {
        const int out_b = n_wave + j * 16 + ml;
        if (out_b >= B) continue;
        float* yrow = Yz + (long long)out_b * M;
        #pragma unroll
        for (int i = 0; i < WP; i++) {
            const sk_float8_t v = acc[i * WQ + j];
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const int out_m = m_wave + i * 16 + 2 * e + kh;
                if (out_m < M) yrow[out_m] = v[e];
            }
        }
    }
#endif
}

// Keep fused staging in its own instantiation: sharing it changes incumbent codegen.
// One K stage of WMMA out of LDS: the only place the accumulation order is
// written down.  `kh` is the lane half (on gfx1201, which 8 of the substep's
// 16 K elements this lane contributes).
template <int BM, int BN, int WP, int WQ, int KT>
__device__ __forceinline__ void sk_bsr_wmma_stage(
    const _Float16* a_lds, const _Float16* x_lds,
    int wave_m, int wave_n, int ml, int kh, sk_float8_t* acc
) {
    constexpr int WROWS = 16 * WP;
    constexpr int WCOLS = 16 * WQ;
    #pragma unroll
    for (int kt = 0; kt < KT; kt++) {
        sk_frag_t a[WP];
        #pragma unroll
        for (int i = 0; i < WP; i++) {
            a[i] = *(const sk_frag_t*)(
                &a_lds[(kt * BM + wave_m * WROWS + i * 16 + ml) * 16 + SK_FRAG_OFFSET(kh)]);
        }
        #pragma unroll
        for (int j = 0; j < WQ; j++) {
            const sk_frag_t b = *(const sk_frag_t*)(
                &x_lds[(kt * BN + wave_n * WCOLS + j * 16 + ml) * 16 + SK_FRAG_OFFSET(kh)]);
            #pragma unroll
            for (int i = 0; i < WP; i++) {
                acc[i * WQ + j] = SK_WMMA(a[i], b, acc[i * WQ + j]);
            }
        }
    }
}

// Separate fused tile: keep the incumbent tile source and aliasing contract unchanged.
template <int BM, int BN, int WP, int WQ, int KS, bool PIPE, int SPLIT>
__device__ __forceinline__ void sk_lds_tile_bsr(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ X,
    float* Y,
    int M, int K, int B,
    _Float16* a_lds, _Float16* x_lds,
    const float* R = nullptr, const float* Gate = nullptr
) {
    constexpr int KT     = KS / 16;
    constexpr int WROWS  = 16 * WP;
    constexpr int WCOLS  = 16 * WQ;
    constexpr int WAVE_M = BM / WROWS;
    constexpr int WAVE_N = BN / WCOLS;
    constexpr int NT     = WAVE_M * WAVE_N * 32;
    constexpr int A_VEC  = BM * KT;
    constexpr int X_VEC  = BN * KT;
    constexpr int A_PASS = (A_VEC + NT - 1) / NT;
    constexpr int X_PASS = (X_VEC + NT - 1) / NT;

    static_assert(KS == 32 || KS == 64, "KS must be 32 or 64");
    static_assert(SPLIT == 1, "epilogue requires completed accumulator");
    static_assert(BM % WROWS == 0 && BN % WCOLS == 0, "tile must divide into waves");
    static_assert(NT <= 1024, "block exceeds the 1024-thread limit");
    static_assert((BM + BN) * KS * 2 <= 65536, "LDS exceeds 64 KB per workgroup");

    const int m_block = blockIdx.x * BM;
    const int n_block = blockIdx.y * BN;
    if (m_block >= M || n_block >= B) return;

    const int k_len   = K / SPLIT;
    const int k_begin = (int)blockIdx.z * k_len;
    const int k_end   = k_begin + k_len;

    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int wave_id = tid >> 5;
    const int ml      = lane & 15;
    const int kh      = lane >> 4;

    const int wave_m = wave_id % WAVE_M;
    const int wave_n = wave_id / WAVE_M;
    const int m_wave = m_block + wave_m * WROWS;
    const int n_wave = n_block + wave_n * WCOLS;

    sk_float8_t acc[WP * WQ];
    #pragma unroll
    for (int i = 0; i < WP * WQ; i++) {
        acc[i] = sk_float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    sk_half16_t a_reg[PIPE ? A_PASS : 1];
    sk_half16_t x_reg[PIPE ? X_PASS : 1];

    if constexpr (PIPE) {
        #pragma unroll
        for (int p = 0; p < A_PASS; p++) {
            const int v = tid + p * NT;
            if (A_VEC % NT != 0 && v >= A_VEC) continue;
            a_reg[p] = *sk_src<KT>(A, M, K, m_block, k_begin, v);
        }
        #pragma unroll
        for (int p = 0; p < X_PASS; p++) {
            const int v = tid + p * NT;
            if (X_VEC % NT != 0 && v >= X_VEC) continue;
            x_reg[p] = *sk_src<KT>(X, B, K, n_block, k_begin, v);
        }
        for (int k0 = k_begin; k0 < k_end; k0 += KS) {
            #pragma unroll
            for (int p = 0; p < A_PASS; p++) {
                const int v = tid + p * NT;
                if (A_VEC % NT != 0 && v >= A_VEC) continue;
                *sk_dst<BM, KT>(a_lds, v) = a_reg[p];
            }
            #pragma unroll
            for (int p = 0; p < X_PASS; p++) {
                const int v = tid + p * NT;
                if (X_VEC % NT != 0 && v >= X_VEC) continue;
                *sk_dst<BN, KT>(x_lds, v) = x_reg[p];
            }
            SK_BARRIER();
            const int k1 = k0 + KS;
            if (k1 < k_end) {
                #pragma unroll
                for (int p = 0; p < A_PASS; p++) {
                    const int v = tid + p * NT;
                    if (A_VEC % NT != 0 && v >= A_VEC) continue;
                    a_reg[p] = *sk_src<KT>(A, M, K, m_block, k1, v);
                }
                #pragma unroll
                for (int p = 0; p < X_PASS; p++) {
                    const int v = tid + p * NT;
                    if (X_VEC % NT != 0 && v >= X_VEC) continue;
                    x_reg[p] = *sk_src<KT>(X, B, K, n_block, k1, v);
                }
            }
            sk_bsr_wmma_stage<BM, BN, WP, WQ, KT>(a_lds, x_lds, wave_m, wave_n, ml, kh, acc);
            SK_BARRIER();
        }
    } else {
        for (int k0 = k_begin; k0 < k_end; k0 += KS) {
            #pragma unroll
            for (int p = 0; p < A_PASS; p++) {
                const int v = tid + p * NT;
                if (A_VEC % NT != 0 && v >= A_VEC) continue;
                *sk_dst<BM, KT>(a_lds, v) = *sk_src<KT>(A, M, K, m_block, k0, v);
            }
            #pragma unroll
            for (int p = 0; p < X_PASS; p++) {
                const int v = tid + p * NT;
                if (X_VEC % NT != 0 && v >= X_VEC) continue;
                *sk_dst<BN, KT>(x_lds, v) = *sk_src<KT>(X, B, K, n_block, k0, v);
            }
            SK_BARRIER();
            sk_bsr_wmma_stage<BM, BN, WP, WQ, KT>(a_lds, x_lds, wave_m, wave_n, ml, kh, acc);
            SK_BARRIER();
        }
    }

    float* Yz = Y + (long long)blockIdx.z * B * M;
#if defined(__gfx1201__)
    // Epilogue: lane (ml, kh) holds Y[n_wave + j*16 + ml][m_wave + i*16 + 8*kh + e].
    #pragma unroll
    for (int j = 0; j < WQ; j++) {
        const int out_b = n_wave + j * 16 + ml;
        if (out_b >= B) continue;
        float* yrow = Yz + (long long)out_b * M;
        #pragma unroll
        for (int i = 0; i < WP; i++) {
            const int m0 = m_wave + i * 16 + 8 * kh;
            sk_float8_t v = acc[i * WQ + j];
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (m0 + e < M)
                    v[e] = sk_bf16_scaled_add(R[(long long)out_b * M + m0 + e], v[e], Gate[out_b]);
            }
            if (m0 + 8 <= M && (M & 3) == 0) {
                *(float4*)(yrow + m0)     = float4{v[0], v[1], v[2], v[3]};
                *(float4*)(yrow + m0 + 4) = float4{v[4], v[5], v[6], v[7]};
            } else {
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    if (m0 + e < M) yrow[m0 + e] = v[e];
                }
            }
        }
    }
#else
    // Epilogue: lane (ml, kh) holds Y[n_wave + j*16 + ml][m_wave + i*16 + 2*e + kh].
    #pragma unroll
    for (int j = 0; j < WQ; j++) {
        const int out_b = n_wave + j * 16 + ml;
        if (out_b >= B) continue;
        float* yrow = Yz + (long long)out_b * M;
        #pragma unroll
        for (int i = 0; i < WP; i++) {
            const sk_float8_t v = acc[i * WQ + j];
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const int out_m = m_wave + i * 16 + 2 * e + kh;
                if (out_m < M) {
                    yrow[out_m] = sk_bf16_scaled_add(R[(long long)out_b * M + out_m], v[e], Gate[out_b]);
                }
            }
        }
    }
#endif
}

#define SK_KERNEL(NAME, BM, BN, WP, WQ, KS, PIPE, SPLIT)                       \
    __launch_bounds__((BM / (16 * WP)) * (BN / (16 * WQ)) * 32, 1)             \
    extern "C" __global__ void NAME(                                           \
        const _Float16* __restrict__ A,                                        \
        const _Float16* __restrict__ X,                                        \
        float*          __restrict__ Y,                                        \
        int M, int K, int B                                                    \
    ) {                                                                        \
        __shared__ __align__(32) _Float16 a_lds[(KS / 16) * (BM) * 16];        \
        __shared__ __align__(32) _Float16 x_lds[(KS / 16) * (BN) * 16];        \
        sk_lds_tile<BM, BN, WP, WQ, KS, PIPE, SPLIT>(A, X, Y, M, K, B,        \
                                                      a_lds, x_lds);           \
    }

// gfx1201: the tiles Gpu::qwen4_lds_tile_gfx1201 selects.  Chosen from a 25-tile
// sweep on the R9700 (qcal/release-0.4.0/qwen4-hc-wmma/logs/sweep{1,2}.txt,
// event-timed, 10-20 reps per tile; ms, gemm_bf16_xf32_multirow in brackets):
//
//   M x K x B              64x128/32x64 p s4   128x128/32x64   128x256/32x64
//   320 x 20480 x 1536     0.340 (4.78)
//   320 x 10240 x 1900     0.217 (1.15)
//   512 x 2560 x 1900                          0.082 (0.52)
//   5120 x 2560 x 1536                                         0.493 (7.88)
//   20480 x 2560 x 1536                                        2.085 (64.1)
//
// Short M wants many workgroups: 64 x 128 with the 4-way K split puts 4x the
// waves of the gfx11 64 x 64 tile on the 64 CUs (64 x 64 pipelined: 0.503 ms
// at 320 x 20480 x 1536); K splits of 2 and 8 are within 3 %.  The unsplit
// pipelined 64 x 128 serves K % 256 != 0.  VGPRs (hipcc 7.15): gfx1201
// 64x128 p s4 234, 128x128 182, 128x256 175; gfx1151 64x128 p s4 161; no
// spills, no scratch.
SK_KERNEL(qwen4_wmma_lds_64_128_32_64_k64_p_s4, 64, 128, 2, 4, 64, true,  4)
SK_KERNEL(qwen4_wmma_lds_64_128_32_64_k64_p,    64, 128, 2, 4, 64, true,  1)
SK_KERNEL(qwen4_wmma_lds_128_128_32_64_k64,    128, 128, 2, 4, 64, false, 1)
SK_KERNEL(qwen4_wmma_lds_128_256_32_64_k64,    128, 256, 2, 4, 64, false, 1)
// gfx1151 sweep candidates.
SK_KERNEL(qwen4_wmma_lds_64_64_32_64_k64_p,      64,  64, 2, 4, 64, true,  1)
SK_KERNEL(qwen4_wmma_lds_64_64_32_64_k64_p_s4,   64,  64, 2, 4, 64, true,  4)
SK_KERNEL(qwen4_wmma_lds_64_64_32_64_k64_p_s2,   64,  64, 2, 4, 64, true,  2)
SK_KERNEL(qwen4_wmma_lds_64_128_32_64_k64_p_s2,  64, 128, 2, 4, 64, true,  2)
SK_KERNEL(qwen4_wmma_lds_64_128_32_64_k64_s4,    64, 128, 2, 4, 64, false, 4)
SK_KERNEL(qwen4_wmma_lds_64_128_32_64_k64_p_s8,  64, 128, 2, 4, 64, true,  8)
SK_KERNEL(qwen4_wmma_lds_64_64_32_32_k64_p_s4,   64,  64, 2, 2, 64, true,  4)
SK_KERNEL(qwen4_wmma_lds_64_128_32_32_k64_p_s4,  64, 128, 2, 2, 64, true,  4)


// Shared-down incumbent gfx1201 tile, single completed accumulator.
extern "C" __global__ __launch_bounds__(512, 1)
void qwen4_wmma_lds_128_256_32_64_k64_bsr(
    const _Float16* A, const _Float16* X, float* Y,
    int M, int K, int B, const float* R, const float* Gate) {
    __shared__ __align__(32) _Float16 a_lds[128 * 64];
    __shared__ __align__(32) _Float16 x_lds[256 * 64];
    sk_lds_tile_bsr<128, 256, 2, 4, 64, false, 1>(
        A, X, Y, M, K, B, a_lds, x_lds, R, Gate);
}
// Y[b, m] = sum over z ascending of P[z, b, m].  Grid [ceil(B*M / 256)],
// block 256.
extern "C" __global__ void qwen4_wmma_lds_reduce(
    const float* __restrict__ P, float* __restrict__ Y, int n, int split
) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float s = P[i];
    for (int z = 1; z < split; z++) s += P[(long long)z * n + i];
    Y[i] = s;
}
