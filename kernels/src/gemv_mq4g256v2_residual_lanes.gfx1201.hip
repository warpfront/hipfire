// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_WEIGHT_LOAD_PTR_U32
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#define HIPFIRE_WEIGHT_RSRC(base) 0
#define HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32(rsrc, ptr, vector_offset, scalar_offset) \
    (*(const unsigned int*)(ptr))
#endif

#ifndef HIPFIRE_RESIDUAL_LANES_KERNEL
#define HIPFIRE_RESIDUAL_LANES_KERNEL gemv_mq4g256v2_residual_lanes_gfx1201
#endif

// Lane-batched MQ4G256V2 GEMV with fused residual add for exact gfx1201:
//
//   y[b][row] += A[row] . x[b]        b < B <= 8 lanes, K = 4096 (the A3B `wo`)
//
// Per lane b the bytes are those of the singleton `gemv_mq4g256v2_residual`
// (gemv_mq4g256v2_residual.hip) run on (x[b], y[b]): the same per-lane
// accumulation order (four group-interleaved accumulators, accumulator j
// holding groups j, j+4, j+8, j+12 in order, each group an 8-term product sum
// added whole), the same `(acc0 + acc1) + (acc2 + acc3)` combine, the same
// 32-lane lower-lane-left reduction tree and the same `y + acc` residual add.
// The weight row is streamed ONCE into registers (the singleton's buffer-load
// path and cache policy, KS_PFW groups ahead) and decoded once per group; only
// the per-column FMA chains repeat, on the lane's x elements read straight from
// global memory.
//
// K-split: the four accumulator chains are independent until the combine, so a
// workgroup of KS_NSPL = 4 waves owns KS_R = 4 rows and wave w runs chain w
// (its four groups) of all four rows: a wave holds one chain of R x NB
// accumulators, never more. The finished chains are exchanged as raw f32 through
// LDS (one barrier, exact); wave w then owns the (row, b) pairs p = w (mod 4),
// forms (c0 + c1) + (c2 + c3), runs the 32-lane tree and adds the residual.
//
// ABI: A, x, y, M, B. x is [B][4096] f32 row-major, y is [B][M] f32
// row-major, 1 <= B <= 8, M arbitrary. Grid = ceil(M / KS_R) workgroups of
// KS_NSPL waves (block KS_NSPL * 32, static LDS KS_NSPL * KS_R * 8 * 32 f32);
// workgroup g owns rows g * KS_R ... + KS_R - 1 (KS_R even: absolute row parity
// selects the contraction form below). B <= KS_SMALL_NB runs a NB = KS_SMALL_NB
// body, otherwise a NB = 8 body (two bodies keep .text small).
//
// The KS_* geometry/tuning macros are overridable (lab sweeps: see
// `residual_lanes_source_with`); the defaults are the shipped configuration.

#ifndef KS_NSPL
#define KS_NSPL 4
#endif
#ifndef KS_R
#define KS_R 4
#endif
#ifndef KS_XD
#define KS_XD 4
#endif
#ifndef KS_PFW
#define KS_PFW 2
#endif
#ifndef KS_SMALL_NB
#define KS_SMALL_NB 5
#endif

#define KS_NCH (4 / KS_NSPL)          // chains per wave
#define KS_NG (KS_NCH * 4)            // groups per wave
#define KS_NT (KS_NSPL * 32)
#define KS_K 4096
#ifndef KS_W64
#define KS_W64 0     // 1: wave64, the two 32-lane halves of a wave run different chains (KS_NSPL = 4 only)
#endif
#ifndef KS_WPE
#define KS_WPE 8
#endif
static_assert((KS_R & 1) == 0, "rows per workgroup must be even: absolute row parity selects the contraction form");
static_assert(KS_NSPL == 2 || KS_NSPL == 4, "KS_NSPL");

// program-order pins: the FMAs of a column step complete before later x/weight loads issue
#define KS_PIN(a0, a1) asm volatile("" : "+v"(a0), "+v"(a1) :: "memory")
#define KS_PIN_X(xa, xb) asm volatile("" : "+v"(xa.x), "+v"(xa.y), "+v"(xa.z), "+v"(xa.w), "+v"(xb.x), "+v"(xb.y), "+v"(xb.z), "+v"(xb.w) :: "memory")

#define DOG_F0(a, wv, xlo, xhi) do { \
    float t_ = (wv[0]) * (xlo.x); \
    t_ = __builtin_fmaf((wv[1]), (xlo.y), t_); \
    t_ = __builtin_fmaf((wv[2]), (xlo.z), t_); \
    t_ = __builtin_fmaf((wv[3]), (xlo.w), t_); \
    t_ = __builtin_fmaf((wv[4]), (xhi.x), t_); \
    t_ = __builtin_fmaf((wv[5]), (xhi.y), t_); \
    t_ = __builtin_fmaf((wv[6]), (xhi.z), t_); \
    t_ = __builtin_fmaf((wv[7]), (xhi.w), t_); \
    (a) += t_; \
} while (0)
#define DOG_F1(a, wv, xlo, xhi) do { \
    float t_ = (wv[1]) * (xlo.y); \
    t_ = __builtin_fmaf((wv[0]), (xlo.x), t_); \
    t_ = __builtin_fmaf((wv[2]), (xlo.z), t_); \
    t_ = __builtin_fmaf((wv[3]), (xlo.w), t_); \
    t_ = __builtin_fmaf((wv[4]), (xhi.x), t_); \
    t_ = __builtin_fmaf((wv[5]), (xhi.y), t_); \
    t_ = __builtin_fmaf((wv[6]), (xhi.z), t_); \
    t_ = __builtin_fmaf((wv[7]), (xhi.w), t_); \
    (a) += t_; \
} while (0)

// lane l < 16: s_l + s_{l+16}; then inside the 16-lane row s_l + s_{l+8}, +4, +2, +1 (own operand first = lower lane left).
template <int N>
__device__ __forceinline__ void ks_tree32(float (&s)[N]) {
    #pragma unroll
    for (int i = 0; i < N; i++) {
#if KS_W64
        float t = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(s[i]), 0x401f));   // lane ^ 16 inside each 32-lane half
#else
        float t = __int_as_float((int)__builtin_amdgcn_permlanex16(
            0u, (unsigned int)__float_as_int(s[i]), 0x76543210u, 0xFEDCBA98u, false, false));
#endif
        asm volatile("" : "+v"(t));
        s[i] = s[i] + t;
    }
#define KS_DPP_STEP(CTRL) \
    _Pragma("unroll") \
    for (int i = 0; i < N; i++) { \
        float t = __int_as_float(__builtin_amdgcn_update_dpp(__float_as_int(s[i]), __float_as_int(s[i]), CTRL, 0xF, 0xF, false)); \
        asm volatile("" : "+v"(t)); \
        s[i] = s[i] + t; }
    KS_DPP_STEP(0x108)
    KS_DPP_STEP(0x104)
    KS_DPP_STEP(0x102)
    KS_DPP_STEP(0x101)
#undef KS_DPP_STEP
}

template <int NB, typename RS>
__device__ __forceinline__ void ks_run(const char* __restrict__ A, const float* __restrict__ x,
                                       float* __restrict__ y, float* __restrict__ lds,
                                       int M, int Bn, int tid, int wave, int row_base) {
    constexpr int R = KS_R, NG = KS_NG, D = KS_XD, PFW = KS_PFW, NS = NG * NB, NP = R * NB;
    constexpr int NSPL = KS_NSPL, NCH = KS_NCH;
    const long long row_stride = (long long)(KS_K / 256) * 136;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const RS rsrc = HIPFIRE_WEIGHT_RSRC(A);
#else
    const RS rsrc = 0;
#endif
    const int wch = wave * NCH;       // first chain of this wave (wave-uniform)
    unsigned int woff[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        const int row = (row_base + r < M) ? (row_base + r) : (M - 1);
#if KS_W64
        woff[r] = (unsigned int)((long long)row * row_stride);      // scalar part; the lane's chain goes into the vector offsets
#else
        woff[r] = (unsigned int)((long long)row * row_stride) + (unsigned int)(wch * 136);
#endif
    }
#if KS_W64
    const unsigned int hoff = ((tid < 16) ? 0u : 4u) + (unsigned int)(wch * 136);
    const unsigned int poff = 8u + (unsigned int)tid * 4u + (unsigned int)(wch * 136);
#else
    const unsigned int hoff = (tid < 16) ? 0u : 4u;
    const unsigned int poff = 8u + (unsigned int)tid * 4u;
#endif
    const float* xl = x + tid * 8 + wch * 256;

    unsigned int hs[NG][R], pk[NG][R];
    float4 xr[D][2];

#define KS_GS(ig) (((ig) / 4) + 4 * ((ig) % 4))     // group offset (relative to the wave's first chain)
#define KS_LOADW(ig) do { \
        _Pragma("unroll") \
        for (int r = 0; r < R; r++) { \
            const unsigned int g_off = woff[r] + (unsigned int)(KS_GS(ig) * 136); \
            hs[ig][r] = HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32(rsrc, A + (long long)g_off + hoff, hoff, g_off); \
            pk[ig][r] = HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32(rsrc, A + (long long)g_off + poff, poff, g_off); \
        } \
    } while (0)
#define KS_LOADX(s) do { \
        const int bc_ = ((s) % NB < Bn) ? ((s) % NB) : (Bn - 1); \
        const float* xp_ = xl + (long long)bc_ * KS_K + KS_GS((s) / NB) * 256; \
        xr[(s) % D][0] = *reinterpret_cast<const float4*>(xp_); \
        xr[(s) % D][1] = *reinterpret_cast<const float4*>(xp_ + 4); \
    } while (0)

    // prologue
    #pragma unroll
    for (int ig = 0; ig < PFW; ig++) KS_LOADW(ig);
    #pragma unroll
    for (int s = 0; s < D - 1; s++) { if (s < NS) KS_LOADX(s); }

    float acc[NCH][R][NB];
    #pragma unroll
    for (int ig = 0; ig < NG; ig++) {
        const int c = ig / 4;
        if ((ig % 4) == 0) {
            #pragma unroll
            for (int r = 0; r < R; r++)
                #pragma unroll
                for (int b = 0; b < NB; b++) acc[c][r][b] = 0.0f;
        }
        if (ig + PFW < NG) KS_LOADW(ig + PFW);
        float w[R][8];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const unsigned int h = hs[ig][r];
            const float sc = __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu)));
            const float zp = __half2float(__ushort_as_half((unsigned short)(h >> 16)));
            const unsigned int p = pk[ig][r];
            const unsigned int lo = p & 0x0F0F0F0Fu;
            const unsigned int hi = (p >> 4) & 0x0F0F0F0Fu;
            w[r][0] = __builtin_fmaf(sc, (float)(lo & 0xFFu), zp);
            w[r][1] = __builtin_fmaf(sc, (float)(hi & 0xFFu), zp);
            w[r][2] = __builtin_fmaf(sc, (float)((lo >> 8) & 0xFFu), zp);
            w[r][3] = __builtin_fmaf(sc, (float)((hi >> 8) & 0xFFu), zp);
            w[r][4] = __builtin_fmaf(sc, (float)((lo >> 16) & 0xFFu), zp);
            w[r][5] = __builtin_fmaf(sc, (float)((hi >> 16) & 0xFFu), zp);
            w[r][6] = __builtin_fmaf(sc, (float)((lo >> 24) & 0xFFu), zp);
            w[r][7] = __builtin_fmaf(sc, (float)((hi >> 24) & 0xFFu), zp);
        }
        #pragma unroll
        for (int b = 0; b < NB; b++) {
            const int s = ig * NB + b;
            if (s + D - 1 < NS) KS_LOADX(s + D - 1);
            float4 xa = xr[s % D][0];
            float4 xb = xr[s % D][1];
            KS_PIN_X(xa, xb);
            #pragma unroll
            for (int r = 0; r < R; r += 2) {
                DOG_F0(acc[c][r][b], w[r], xa, xb);
                DOG_F1(acc[c][r + 1][b], w[r + 1], xa, xb);
                KS_PIN(acc[c][r][b], acc[c][r + 1][b]);
            }
        }
    }

    // ---- per-chain (pair) values -> LDS, exact combine by the owner wave of each (row, b) pair ----
    constexpr int CPW = (NP + NSPL - 1) / NSPL;
    float yv[CPW];
    #pragma unroll
    for (int i = 0; i < CPW; i++) {
        const int p = (i * NSPL + wave < NP) ? (i * NSPL + wave) : (NP - 1);
        const int r = p / NB, b0 = p % NB;
        const int b = (b0 < Bn) ? b0 : (Bn - 1);
        const int row = (row_base + r < M) ? (row_base + r) : (M - 1);
        yv[i] = y[(long long)b * M + row];
    }
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int b = 0; b < NB; b++) {
            float e = acc[0][r][b];
            if (NCH == 2) e = e + acc[1][r][b];           // acc(2w) + acc(2w+1)
            lds[(wave * NP + r * NB + b) * 32 + tid] = e;
        }
    __syncthreads();
    float a[CPW];
    #pragma unroll
    for (int i = 0; i < CPW; i++) {
        const int p = (i * NSPL + wave < NP) ? (i * NSPL + wave) : (NP - 1);
        float c[NSPL];
        #pragma unroll
        for (int j = 0; j < NSPL; j++) c[j] = lds[(j * NP + p) * 32 + tid];
        if (NSPL == 4) a[i] = (c[0] + c[1]) + (c[2] + c[3]);
        else a[i] = c[0] + c[1];
    }
    ks_tree32<CPW>(a);
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < CPW; i++) {
            const int pi = i * NSPL + wave;
            const int p = (pi < NP) ? pi : (NP - 1);
            const int r = p / NB, b = p % NB;
            if (pi < NP && b < Bn && row_base + r < M) y[(long long)b * M + row_base + r] = yv[i] + a[i];
        }
    }
}

__launch_bounds__(KS_NT) __attribute__((amdgpu_waves_per_eu(KS_WPE)))
extern "C" __global__ void HIPFIRE_RESIDUAL_LANES_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,   // [B][4096]
    float* __restrict__ y,         // [B][M]
    int M, int B
) {
    __shared__ __attribute__((aligned(16))) float ks_lds[KS_NSPL * KS_R * 8 * 32];
    const int tid = threadIdx.x & 31;
#if KS_W64
    const int wave = (int)(threadIdx.x >> 5);                  // owner/chain index, lane-uniform per 32-lane half
#else
    const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 5));
#endif
    const int row_base = (int)blockIdx.x * KS_R;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    typedef __amdgpu_buffer_rsrc_t RsT;
#else
    typedef int RsT;
#endif
    if (B <= 0) return;
    // B <= KS_SMALL_NB: the NB = KS_SMALL_NB body (columns >= B re-read column B-1, results never stored);
    // otherwise the NB = 8 body (same clamp for B = 6, 7). Two bodies keep .text near the incumbent's.
    if (B <= KS_SMALL_NB) ks_run<KS_SMALL_NB, RsT>(A, x, y, ks_lds, M, B, tid, wave, row_base);
    else ks_run<8, RsT>(A, x, y, ks_lds, M, (B < 8 ? B : 8), tid, wave, row_base);
}
