// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// W8A8 Q8_0 GEMM for gfx12 (RDNA4 — gfx1200/gfx1201): int8 weights x int8
// per-row per-32-K activations on `v_wmma_i32_16x16x16_iu8`, F32 output.
//
//   Y[n, m] = sum_b (wscale[m, b] * xscale[n, b])
//                 * sum_{i<32} int8 W[m, 32b + i] * int8 X[n, 32b + i]
//
//   W  : Q8_0 as stored in the artifact: [M, K/32] blocks of 34 B
//        {f16 scale, int8 q[32]} (row stride 34*K/32, NOT 4-byte aligned in
//        general: the stager realigns through dword loads + a funnel shift).
//   X  : the output of `quantize_x_q8_blk32` below (int8 [N, K] row-major plus
//        F32 scales [K/32, N], block-major so a slab's scales are contiguous).
//   Y  : F32 [N, M] row-major (same ABI as `gemm_q8_0_wmma_gfx12`).
//
// Producer contract (`quantize_x_q8_blk32`; one definition for the kernel, the
// CPU oracle and the accuracy test):
//   amax = max |x| over the 32 contiguous K elements (fmaxf, f32)
//   d    = RN(amax / 127.0f)                       (IEEE divide)
//   q    = clamp(rintf(RN(x / d)), -127, 127)      (IEEE divide, ties-to-even)
//   zero block (amax == 0): d = 0, q = 0
// (the same rule as `quantize_int8_mmq_ds128`, at 32-K granularity).
//
// GEMM structure. A CTA owns a BM-row x BN-token tile and walks K in 64-wide
// slabs (two Q8_0 blocks) through a two-buffer LDS ring: global -> registers
// for slab s+1 issue before slab s is computed, the registers are published to
// the other LDS buffer after the compute, one barrier per slab. Each 16x16
// output tile of a wave takes two chained iu8 WMMAs per 32-K block into an
// accumulator seeded with 0x4B400000, so as_float(c) - 12582912 is the exact
// integer dot, which is folded into the F32 accumulator with
// fma(RN(ws * xs), dot, acc). The integer dot is exact (|dot| <= 32*128*127),
// so the whole GEMM is exact up to the F32 fold order (blocks ascending) and a
// CPU emulation reproduces it bit for bit (tests/gemm_q8_0_a8_gfx12.rs).
//
// iu8 K16 conventions (gemm_hfq4g256_residual_mmq.gfx12.hip): per lane int32x2
// A/B (8 int8), lanes 0-15 carry one K8 half of the step and lanes 16-31 the
// other, C layout acc[j] = C[8*(lane>>4) + j][lane&15]. Lane (m, g) reads the
// 16 contiguous int8 [g*16, g*16+16) of the 32-K block as one b128: dwords
// 0-1 feed WMMA 0 and dwords 2-3 feed WMMA 1. The K permutation is shared by
// both operands, so the integer dot is unchanged.
//
// LDS rows are padded to 80 B (64 B of data) so the b128 fragment reads of 8
// consecutive rows hit all 32 banks once.
//
// Grid [ceil(M/BM), ceil(N/BN)]; block NT = 2*BM threads. FAST variants need
// K % 64 == 0 and a 4-byte aligned W base; SLOW variants (any K % 32 == 0,
// 2-byte aligned W) read the weights through 16-bit loads.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

#if defined(__gfx1200__) || defined(__gfx1201__)

typedef int32_t __attribute__((ext_vector_type(2))) q8a8_i32x2_t;
typedef int32_t __attribute__((ext_vector_type(8))) q8a8_i32x8_t;
typedef float   __attribute__((ext_vector_type(8))) q8a8_f32x8_t;

#define Q8A8_PITCH 80
#define Q8A8_SLAB 64
#define Q8A8_MAGIC 0x4B400000

// ---------------------------------------------------------------------------
// Producer: F32 [N, K] -> int8 [N, K] + F32 scales [K/32, N]. K % 32 == 0.
// Thread = 4 elements; 8 consecutive lanes = one 32-K block of one token.
// Folded 1-D launch (block 256): grid = ceil(N * K / 4 / 256).
// ---------------------------------------------------------------------------
extern "C" __global__ void quantize_x_q8_blk32(
    const float* __restrict__ X,
    signed char* __restrict__ Xq,
    float*       __restrict__ Xs,
    int K, int N
) {
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long quads_per_row = K >> 2;
    const long long total = quads_per_row * (long long)N;
    // Out-of-range lanes still take part in the shuffles (their 8-lane group
    // is entirely out of range because quads_per_row % 8 == 0).
    float4 xi = make_float4(0.f, 0.f, 0.f, 0.f);
    const bool live = g < total;
    long long token = 0;
    long long kq = 0;
    if (live) {
        token = g / quads_per_row;
        kq = g - token * quads_per_row;
        xi = *(const float4*)(X + token * K + kq * 4);
    }
    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
    amax = fmaxf(amax, __shfl_xor(amax, 1, 32));
    amax = fmaxf(amax, __shfl_xor(amax, 2, 32));
    amax = fmaxf(amax, __shfl_xor(amax, 4, 32));
    if (!live) return;
    const bool zero = amax == 0.0f;
    const float d = zero ? 0.0f : __fdiv_rn(amax, 127.0f);
    const float xv[4] = {xi.x, xi.y, xi.z, xi.w};
    unsigned word = 0u;
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float qf = zero ? 0.0f
                              : fminf(fmaxf(rintf(__fdiv_rn(xv[e], d)), -127.0f), 127.0f);
        word |= ((unsigned)(int)qf & 255u) << (8 * e);
    }
    *(unsigned*)(Xq + token * K + kq * 4) = word;
    if ((kq & 7) == 0) {
        Xs[(kq >> 3) * (long long)N + token] = d;
    }
}

// ---------------------------------------------------------------------------
// GEMM body.
// ---------------------------------------------------------------------------
template <int BM, int BN, int WM, int WN, bool FAST>
static __device__ __forceinline__ void gemm_q8_0_mmq_i8_body(
    const char*        __restrict__ Wq,
    const signed char* __restrict__ Xq,
    const float*       __restrict__ Xs,
    float*             __restrict__ Y,
    int M, int K, int N
) {
    constexpr int NW = WM * WN;
    constexpr int NT = NW * 32;
    static_assert(NT == 2 * BM, "W staging maps two threads per row");
    constexpr int RG = BM / WM / 16;
    constexpr int NB = BN / WN / 16;
    static_assert(RG * WM * 16 == BM && NB * WN * 16 == BN, "tile split");
    constexpr int W_BYTES = BM * Q8A8_PITCH;
    constexpr int X_BYTES = BN * Q8A8_PITCH;
    constexpr int WS_BYTES = 2 * BM * 4;
    constexpr int XS_BYTES = 2 * BN * 4;
    constexpr int BUF_BYTES = W_BYTES + X_BYTES + WS_BYTES + XS_BYTES;
    constexpr int XCH = (4 * BN + NT - 1) / NT;      // 16 B X chunks per thread
    constexpr int XSCH = (2 * BN + NT - 1) / NT;     // X scale words per thread
    __shared__ __attribute__((aligned(16))) char lds[2 * BUF_BYTES];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int m_lane = lane & 15;
    const int k_grp = lane >> 4;
    const int wm = wave % WM;
    const int wn = wave / WM;
    const int wave_row0 = wm * (RG * 16);
    const int wave_tok0 = wn * (NB * 16);
    const int tile_m = blockIdx.x * BM;
    const int tile_n = blockIdx.y * BN;
    if (tile_m >= M || tile_n >= N) return;

    const int nb32 = K >> 5;
    const int nslab = (K + Q8A8_SLAB - 1) / Q8A8_SLAB;
    const long long row_stride = (long long)nb32 * 34;

    // Staging identities.
    // W staging: BM rows x 2 K32 blocks over 2*BM threads; the block half is
    // wave-uniform (BM % 32 == 0) so the realigning path is a scalar branch.
    const int w_row_t = tid % BM;           // W tile row staged by this thread
    const int w_half = __builtin_amdgcn_readfirstlane(tid / BM);  // which 32-K block
    const int w_row_g = min(tile_m + w_row_t, M - 1);
    const char* w_src = Wq + (long long)w_row_g * row_stride;

    long long x_row[XCH];
    #pragma unroll
    for (int i = 0; i < XCH; i++) {
        const int q = tid + i * NT;
        x_row[i] = (long long)min(tile_n + (q >> 2), N - 1) * K;
    }

    uint4 xr[XCH];
    float xsr[XSCH];
    unsigned wr[9];
    unsigned short wsr = 0;   // SLOW: this thread's block scale bits

    auto load_slab = [&](int s) {
        #pragma unroll
        for (int i = 0; i < XCH; i++) {
            // Past-K chunks (K % 64 == 32 tail) re-read the row's last 16 B; their
            // scales are zero, so they contribute nothing.
            const int kofs = min(s * Q8A8_SLAB + (((tid + i * NT) & 3) * 16), K - 16);
            xr[i] = *(const uint4*)(Xq + x_row[i] + kofs);
        }
        #pragma unroll
        for (int i = 0; i < XSCH; i++) {
            const int q = tid + i * NT;
            const int b = q / BN;
            const int t = q - b * BN;
            const int blk = 2 * s + b;
            const bool ok = q < 2 * BN && blk < nb32 && tile_n + t < N;
            const float v = Xs[(long long)min(blk, nb32 - 1) * N + min(tile_n + t, N - 1)];
            xsr[i] = ok ? v : 0.0f;
        }
        if constexpr (FAST) {
            // Slab = 68 B = 17 dwords at 68*s: half 0 reads dwords 0..8
            // (sc0 | q0 shifted by 2 B | low half of sc1), half 1 dwords 8..16
            // (sc1 in the high half of the first, q1 aligned after it).
            const unsigned* p = (const unsigned*)(w_src + (long long)s * 68 + w_half * 32);
            #pragma unroll
            for (int j = 0; j < 9; j++) wr[j] = p[j];
        } else {
            const int blk = 2 * s + w_half;
            if (blk < nb32) {
                const unsigned short* p = (const unsigned short*)(w_src + (long long)blk * 34);
                wsr = p[0];
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wr[j] = (unsigned)p[1 + 2 * j] | ((unsigned)p[2 + 2 * j] << 16);
                }
            } else {
                wsr = 0;
                #pragma unroll
                for (int j = 0; j < 8; j++) wr[j] = 0u;
            }
        }
    };

    auto store_slab = [&](int buf) {
        char* b = lds + buf * BUF_BYTES;
        #pragma unroll
        for (int i = 0; i < XCH; i++) {
            const int q = tid + i * NT;
            if ((q >> 2) < BN) {
                *(uint4*)(b + W_BYTES + (q >> 2) * Q8A8_PITCH + (q & 3) * 16) = xr[i];
            }
        }
        float* xs_l = (float*)(b + W_BYTES + X_BYTES + WS_BYTES);
        #pragma unroll
        for (int i = 0; i < XSCH; i++) {
            const int q = tid + i * NT;
            if (q < 2 * BN) xs_l[q] = xsr[i];   // [b][t] = q
        }
        unsigned qd[8];
        unsigned short sc;
        if constexpr (FAST) {
            if (w_half) {
                #pragma unroll
                for (int j = 0; j < 8; j++) qd[j] = wr[j + 1];
                sc = (unsigned short)(wr[0] >> 16);
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) qd[j] = (wr[j] >> 16) | (wr[j + 1] << 16);
                sc = (unsigned short)(wr[0] & 0xFFFFu);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) qd[j] = wr[j];
            sc = wsr;
        }
        char* wrow = b + w_row_t * Q8A8_PITCH + w_half * 32;
        *(uint4*)(wrow) = make_uint4(qd[0], qd[1], qd[2], qd[3]);
        *(uint4*)(wrow + 16) = make_uint4(qd[4], qd[5], qd[6], qd[7]);
        float* ws_l = (float*)(b + W_BYTES + X_BYTES);
        ws_l[w_half * BM + w_row_t] = __half2float(__ushort_as_half(sc));
    };

    q8a8_f32x8_t acc[RG][NB];
    #pragma unroll
    for (int rg = 0; rg < RG; rg++) {
        #pragma unroll
        for (int nb = 0; nb < NB; nb++) {
            acc[rg][nb] = (q8a8_f32x8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }
    }

    load_slab(0);
    store_slab(0);
    __syncthreads();

    for (int s = 0; s < nslab; s++) {
        const int cur = s & 1;
        const bool more = (s + 1) < nslab;
        if (more) load_slab(s + 1);

        const char* b = lds + cur * BUF_BYTES;
        const char* xb = b + W_BYTES;
        const float* ws_l = (const float*)(b + W_BYTES + X_BYTES);
        const float* xs_l = (const float*)(b + W_BYTES + X_BYTES + WS_BYTES);
        // Not unrolled: unrolling both K32 blocks doubles the live fragment and
        // WMMA-result registers and spills the 128x128 tile.
        #pragma unroll 1
        for (int blk = 0; blk < 2; blk++) {
            uint4 af[RG];
            uint4 bf[NB];
            #pragma unroll
            for (int rg = 0; rg < RG; rg++) {
                af[rg] = *(const uint4*)(b + (wave_row0 + rg * 16 + m_lane) * Q8A8_PITCH
                                           + blk * 32 + k_grp * 16);
            }
            #pragma unroll
            for (int nb = 0; nb < NB; nb++) {
                bf[nb] = *(const uint4*)(xb + (wave_tok0 + nb * 16 + m_lane) * Q8A8_PITCH
                                            + blk * 32 + k_grp * 16);
            }
            float wsv[RG][8];
            #pragma unroll
            for (int rg = 0; rg < RG; rg++) {
                const float4 w0 = *(const float4*)(ws_l + blk * BM + wave_row0 + rg * 16 + 8 * k_grp);
                const float4 w1 = *(const float4*)(ws_l + blk * BM + wave_row0 + rg * 16 + 8 * k_grp + 4);
                wsv[rg][0] = w0.x; wsv[rg][1] = w0.y; wsv[rg][2] = w0.z; wsv[rg][3] = w0.w;
                wsv[rg][4] = w1.x; wsv[rg][5] = w1.y; wsv[rg][6] = w1.z; wsv[rg][7] = w1.w;
            }
            float xsv[NB];
            #pragma unroll
            for (int nb = 0; nb < NB; nb++) {
                xsv[nb] = xs_l[blk * BN + wave_tok0 + nb * 16 + m_lane];
            }
            #pragma unroll
            for (int nb = 0; nb < NB; nb++) {
                #pragma unroll
                for (int rg = 0; rg < RG; rg++) {
                    // C seeded with 0x4B400000: as_float(c) - 12582912 is the exact integer
                    // dot (|dot| <= 32*128*127 < 2^22, so 0x4B400000 + dot stays in the
                    // binade where float == 12582912 + dot). A subtract replaces the
                    // int->float convert and pairs into VOPD with the scale multiply / fma.
                    q8a8_i32x8_t c = {Q8A8_MAGIC, Q8A8_MAGIC, Q8A8_MAGIC, Q8A8_MAGIC,
                                      Q8A8_MAGIC, Q8A8_MAGIC, Q8A8_MAGIC, Q8A8_MAGIC};
                    const q8a8_i32x2_t a0 = {(int)af[rg].x, (int)af[rg].y};
                    const q8a8_i32x2_t a1 = {(int)af[rg].z, (int)af[rg].w};
                    const q8a8_i32x2_t b0 = {(int)bf[nb].x, (int)bf[nb].y};
                    const q8a8_i32x2_t b1 = {(int)bf[nb].z, (int)bf[nb].w};
                    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a0, true, b0, c, false);
                    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a1, true, b1, c, false);
                    // Whole-vector bitcast + subtract: the per-element form
                    // `bit_cast<float>(c[j]) - M` is miscompiled by the hipcc this
                    // was written against (AMD clang 24.0.0git, d6f6cb69): every j
                    // reuses element 0's result.
                    const q8a8_f32x8_t cf = __builtin_bit_cast(q8a8_f32x8_t, c) - 12582912.0f;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        acc[rg][nb][j] = __builtin_fmaf(wsv[rg][j] * xsv[nb], cf[j], acc[rg][nb][j]);
                    }
                }
            }
        }
        if (more) store_slab(cur ^ 1);
        __syncthreads();
    }

    // Epilogue: acc[rg][nb][j] = Y[token = tile_n + wave_tok0 + nb*16 + m_lane]
    //                             [row   = tile_m + wave_row0 + rg*16 + 8*k_grp + j].
    const bool vec_ok = ((M & 7) == 0) && (((uintptr_t)Y & 15u) == 0);
    #pragma unroll
    for (int nb = 0; nb < NB; nb++) {
        const int tok = tile_n + wave_tok0 + nb * 16 + m_lane;
        if (tok >= N) continue;
        #pragma unroll
        for (int rg = 0; rg < RG; rg++) {
            const int row0 = tile_m + wave_row0 + rg * 16 + 8 * k_grp;
            float* yp = Y + (long long)tok * M + row0;
            if (vec_ok) {
                if (row0 < M) {
                    *(float4*)(yp) = make_float4(acc[rg][nb][0], acc[rg][nb][1], acc[rg][nb][2], acc[rg][nb][3]);
                    *(float4*)(yp + 4) = make_float4(acc[rg][nb][4], acc[rg][nb][5], acc[rg][nb][6], acc[rg][nb][7]);
                }
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (row0 + j < M) yp[j] = acc[rg][nb][j];
                }
            }
        }
    }
}

#define Q8A8_KERNEL(NAME, BM, BN, WM, WN, FAST)                                  \
    extern "C" __global__ __launch_bounds__(2 * (BM), 2) void NAME(              \
        const char* __restrict__ Wq, const signed char* __restrict__ Xq,         \
        const float* __restrict__ Xs, float* __restrict__ Y, int M, int K, int N) { \
        gemm_q8_0_mmq_i8_body<BM, BN, WM, WN, FAST>(Wq, Xq, Xs, Y, M, K, N);     \
    }

Q8A8_KERNEL(gemm_q8_0_mmq_i8_t128x128, 128, 128, 4, 2, true)
Q8A8_KERNEL(gemm_q8_0_mmq_i8_t128x64, 128, 64, 4, 2, true)
Q8A8_KERNEL(gemm_q8_0_mmq_i8_t64x64, 64, 64, 2, 2, true)
Q8A8_KERNEL(gemm_q8_0_mmq_i8_t128x128_slow, 128, 128, 4, 2, false)
Q8A8_KERNEL(gemm_q8_0_mmq_i8_t128x64_slow, 128, 64, 4, 2, false)
Q8A8_KERNEL(gemm_q8_0_mmq_i8_t64x64_slow, 64, 64, 2, 2, false)

#undef Q8A8_KERNEL

#else

extern "C" __global__ void quantize_x_q8_blk32(const float*, signed char*, float*, int, int) {}
#define Q8A8_STUB(NAME) \
    extern "C" __global__ void NAME(const char*, const signed char*, const float*, float*, int, int, int) {}
Q8A8_STUB(gemm_q8_0_mmq_i8_t128x128)
Q8A8_STUB(gemm_q8_0_mmq_i8_t128x64)
Q8A8_STUB(gemm_q8_0_mmq_i8_t64x64)
Q8A8_STUB(gemm_q8_0_mmq_i8_t128x128_slow)
Q8A8_STUB(gemm_q8_0_mmq_i8_t128x64_slow)
Q8A8_STUB(gemm_q8_0_mmq_i8_t64x64_slow)
#undef Q8A8_STUB

#endif
