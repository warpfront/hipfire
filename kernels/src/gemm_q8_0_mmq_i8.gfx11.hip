// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// W8A8 Q8_0 GEMM for gfx11 (RDNA3 — gfx1100 and gfx1151): int8 weights x int8 per-32-K Q8_1
// activations on `v_wmma_i32_16x16x16_iu8` (wave32), F32 output. Tiled sibling of
// `gemm_q8_0_mmq_4w.gfx1151.hip` (64x64, one K32 block per barrier pair), built on the structure of
// `gemm_q8_0_mmq_i8.gfx12.hip`: BM x BN tiles walking K in 64-wide slabs through ONE LDS buffer
// (global -> registers for slab s+1 before slab s computes, published after the compute, two
// barriers per slab), integer-exact dots seeded with 0x4B400000, and the software-pipelined
// WMMA/fold schedule (the WMMAs of tile t+1 are issued before the fold of tile t and a
// sched_barrier pins that order, so the matrix pipe works while the VALU folds).
//
//   Y[n, m] = sum_b fma-fold over 32-K blocks b of  RN(wscale[m, b] * dx[n, b]) * dot_b(m, n)
//
// BIT-IDENTICAL to `gemm_q8_0_mmq_4w_gfx1151` (tests/gemm_q8_0_mmq_gfx11_tiles.rs asserts it): per
// output element the K32 blocks are folded in ascending order into an F32 accumulator that starts at
// +0, each as fma(RN(ws * dx), (float)dot32, acc); dot32 is an exact int32 (|dot| <= 32*128*127), so
// the tile shape, the schedule and the seeded-accumulator trick cannot change a bit.
//
//   W : Q8_0 [M, K/32 * 34] bytes, per 32-K block {f16 scale, int8 q[32]} (read in place).
//   X : block_q8_1_mmq [K/128 x N] (144 B: half2 ds4[4] = (d, sum) per 32-K, int8 qs[128]), the output
//       of `quantize_q8_1_mmq_ds4` (ensure_q8_1_mmq_x); only d is used (as an exact f16 -> f32).
//   Y : F32 [N, M] row-major.
//
// RDNA3 operand/C layout: A and B are 16 int8 per lane (one K16 step), lanes r and r+16 carry
// identical operands, C element j of lane L is C[2j + (L >> 4)][L & 15]. The WEIGHTS ride on A and
// the activation tokens on B (C row = weight row, C col = token). The W rows are staged to LDS in the
// slot order slot(r) = 2 * (r & 7) + ((r >> 3) & 1) inside each 16-row group, so that lane m reads
// slot m and holds C rows 2j + hi = weight rows 8 * hi + j: a lane owns 8 CONSECUTIVE weight rows
// of one token, so the weight scales are two float4 LDS reads and the epilogue writes two float4
// per tile (as the gfx12 kernel), not 8 scattered dwords.
//
// LDS rows are padded to 80 B (64 B of data). Grid [ceil(M/BM), ceil(N/BN)]; block NT = 2*BM
// threads (two threads stage one W row). Needs K % 64 == 0 and a 4-byte aligned W base (row stride
// 34 * K/32 is then a multiple of 4).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#if defined(__gfx1100__) || defined(__gfx1151__)

#define Q8G11_PITCH 80
#define Q8G11_SLAB 64
#define Q8G11_MAGIC 0x4B400000

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[128];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

typedef int32_t __attribute__((ext_vector_type(4))) q8g11_i32x4_t;
typedef int32_t __attribute__((ext_vector_type(8))) q8g11_i32x8_t;
typedef float   __attribute__((ext_vector_type(8))) q8g11_f32x8_t;
typedef unsigned __attribute__((ext_vector_type(4))) q8g11_u32x4_t;

// LDS slot of weight row r (r = row inside the tile): see the header comment.
static __device__ __forceinline__ int q8g11_slot(int r) {
    return (r & ~15) | (2 * (r & 7)) | ((r >> 3) & 1);
}

// Bit-level RN-even F32 -> BF16 -> F32 round trip, the exact operation of
// `bf16_round_trip_f32_strided` (qwen4_bf16_round_trip.hip) and of `q8a8_bf16_rn` in
// `gemm_q8_0_mmq_i8.gfx12.hip`: non-finite values pass through unchanged. The RBF16 kernels apply
// it to each finished accumulator before the store, so the result equals the plain kernel followed
// by the in-place round kernel.
static __device__ __forceinline__ float q8g11_bf16_rn(float v) {
    if (!isfinite(v)) return v;
    union { float f; unsigned int u; } box;
    box.f = v;
    const unsigned int lsb = (box.u >> 16) & 1u;
    box.u = (box.u + 0x7FFFu + lsb) & 0xFFFF0000u;
    return box.f;
}

template <int BM, int BN, int WM, int WN, bool RBF16>
static __device__ __forceinline__ void gemm_q8_0_mmq_i8_gfx11_body(
    const char*           __restrict__ Wq,
    const block_q8_1_mmq* __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N
) {
    constexpr int NW = WM * WN;
    constexpr int NT = NW * 32;
    static_assert(NT == 2 * BM, "W staging maps two threads per row");
    constexpr int RG = BM / WM / 16;
    constexpr int NB = BN / WN / 16;
    static_assert(RG * WM * 16 == BM && NB * WN * 16 == BN, "tile split");
    constexpr int W_BYTES = BM * Q8G11_PITCH;
    constexpr int X_BYTES = BN * Q8G11_PITCH;
    constexpr int WS_BYTES = 2 * BM * 4;
    constexpr int XS_BYTES = 2 * BN * 4;
    constexpr int BUF_BYTES = W_BYTES + X_BYTES + WS_BYTES + XS_BYTES;
    constexpr int XCH = (4 * BN + NT - 1) / NT;      // 16 B X chunks per thread
    constexpr int XSCH = (2 * BN + NT - 1) / NT;     // X scale words per thread
    __shared__ __attribute__((aligned(16))) char lds[BUF_BYTES];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int m_lane = lane & 15;
    const int hi = lane >> 4;
    const int wm = wave % WM;
    const int wn = wave / WM;
    const int wave_row0 = wm * (RG * 16);
    const int wave_tok0 = wn * (NB * 16);
    const int tile_m = blockIdx.x * BM;
    const int tile_n = blockIdx.y * BN;
    if (tile_m >= M || tile_n >= N) return;

    const int nb32 = K >> 5;
    const int nslab = K / Q8G11_SLAB;
    const long long row_stride = (long long)nb32 * 34;

    // W staging: BM rows x 2 K32 blocks over 2*BM threads; the block half is wave-uniform
    // (BM % 32 == 0), so the realigning path is a scalar branch.
    const int w_row_t = tid % BM;           // W tile row staged by this thread
    const int w_half = __builtin_amdgcn_readfirstlane(tid / BM);  // which 32-K block
    const int w_row_g = min(tile_m + w_row_t, M - 1);
    const char* w_src = Wq + (long long)w_row_g * row_stride + w_half * 32;
    const int w_slot = q8g11_slot(w_row_t);

    // X staging: chunk q = tid + i*NT is (token q >> 2, 16 B chunk q & 3) of the slab's 64 B.
    unsigned x_off[XCH];
    #pragma unroll
    for (int i = 0; i < XCH; i++) {
        const int q = tid + i * NT;
        x_off[i] = (unsigned)min(tile_n + (q >> 2), N - 1) * 144u + 16u + (unsigned)(q & 3) * 16u;
    }
    // X scale words: q = tid + i*NT is (block b = q / BN of the slab, token q % BN).
    unsigned xs_off[XSCH];
    bool xs_ok[XSCH];
    #pragma unroll
    for (int i = 0; i < XSCH; i++) {
        const int q = tid + i * NT;
        const int b = q / BN;
        const int t = q - b * BN;
        xs_off[i] = (unsigned)min(tile_n + t, N - 1) * 144u + (unsigned)b * 4u;
        xs_ok[i] = q < 2 * BN && tile_n + t < N;
    }

    q8g11_u32x4_t xr[XCH];
    unsigned xsr[XSCH];
    unsigned wr[9];

    auto load_slab = [&](int s) {
        const char* xb = (const char*)(Xq + (long long)(s >> 1) * N);
        #pragma unroll
        for (int i = 0; i < XCH; i++) {
            xr[i] = *(const q8g11_u32x4_t*)(xb + x_off[i] + (s & 1) * 64);
        }
        #pragma unroll
        for (int i = 0; i < XSCH; i++) {
            xsr[i] = *(const unsigned*)(xb + xs_off[i] + (s & 1) * 8);
        }
        // Slab = 68 B = 17 dwords at 68*s: half 0 reads dwords 0..8 (sc0 | q0 shifted by 2 B |
        // low half of sc1), half 1 dwords 8..16 (sc1 in the high half of the first, q1 aligned).
        const unsigned* p = (const unsigned*)(w_src + (long long)s * 68);
        #pragma unroll
        for (int j = 0; j < 9; j++) wr[j] = p[j];
    };

    auto store_slab = [&]() {
        char* b = lds;
        #pragma unroll
        for (int i = 0; i < XCH; i++) {
            const int q = tid + i * NT;
            // XCH * NT == 4 * BN: every chunk of every thread is in range (no guard to diverge on).
            if (XCH * NT == 4 * BN || (q >> 2) < BN) {
                *(q8g11_u32x4_t*)(b + W_BYTES + (q >> 2) * Q8G11_PITCH + (q & 3) * 16) = xr[i];
            }
        }
        float* xs_l = (float*)(b + W_BYTES + X_BYTES + WS_BYTES);
        #pragma unroll
        for (int i = 0; i < XSCH; i++) {
            const int q = tid + i * NT;
            // Tokens >= N and padding words carry scale 0.
            const float v = xs_ok[i] ? __half2float(__ushort_as_half((unsigned short)(xsr[i] & 0xFFFFu))) : 0.0f;
            if (XSCH * NT == 2 * BN || q < 2 * BN) xs_l[q] = v;   // [b][t] = q
        }
        unsigned qd[8];
        unsigned short sc;
        if (w_half) {
            #pragma unroll
            for (int j = 0; j < 8; j++) qd[j] = wr[j + 1];
            sc = (unsigned short)(wr[0] >> 16);
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) qd[j] = (wr[j] >> 16) | (wr[j + 1] << 16);
            sc = (unsigned short)(wr[0] & 0xFFFFu);
        }
        char* wrow = b + w_slot * Q8G11_PITCH + w_half * 32;
        *(uint4*)(wrow) = make_uint4(qd[0], qd[1], qd[2], qd[3]);
        *(uint4*)(wrow + 16) = make_uint4(qd[4], qd[5], qd[6], qd[7]);
        float* ws_l = (float*)(b + W_BYTES + X_BYTES);
        ws_l[w_half * BM + w_row_t] = __half2float(__ushort_as_half(sc));
    };

    q8g11_f32x8_t acc[RG][NB];
    #pragma unroll
    for (int rg = 0; rg < RG; rg++) {
        #pragma unroll
        for (int nb = 0; nb < NB; nb++) {
            acc[rg][nb] = (q8g11_f32x8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }
    }

    load_slab(0);
    store_slab();
    __syncthreads();

    for (int s = 0; s < nslab; s++) {
        const bool more = (s + 1) < nslab;
        if (more) load_slab(s + 1);

        const char* b = lds;
        const char* xb = b + W_BYTES;
        const float* ws_l = (const float*)(b + W_BYTES + X_BYTES);
        const float* xs_l = (const float*)(b + W_BYTES + X_BYTES + WS_BYTES);
        // Not unrolled: unrolling both K32 blocks doubles the live fragment and WMMA-result
        // registers.
        #pragma unroll 1
        for (int blk = 0; blk < 2; blk++) {
            q8g11_i32x4_t af0[RG], af1[RG];
            q8g11_i32x4_t bf0[NB], bf1[NB];
            #pragma unroll
            for (int rg = 0; rg < RG; rg++) {
                const char* p = b + (wave_row0 + rg * 16 + m_lane) * Q8G11_PITCH + blk * 32;
                af0[rg] = *(const q8g11_i32x4_t*)(p);
                af1[rg] = *(const q8g11_i32x4_t*)(p + 16);
            }
            #pragma unroll
            for (int nb = 0; nb < NB; nb++) {
                const char* p = xb + (wave_tok0 + nb * 16 + m_lane) * Q8G11_PITCH + blk * 32;
                bf0[nb] = *(const q8g11_i32x4_t*)(p);
                bf1[nb] = *(const q8g11_i32x4_t*)(p + 16);
            }
            float wsv[RG][8];
            #pragma unroll
            for (int rg = 0; rg < RG; rg++) {
                const float4 w0 = *(const float4*)(ws_l + blk * BM + wave_row0 + rg * 16 + 8 * hi);
                const float4 w1 = *(const float4*)(ws_l + blk * BM + wave_row0 + rg * 16 + 8 * hi + 4);
                wsv[rg][0] = w0.x; wsv[rg][1] = w0.y; wsv[rg][2] = w0.z; wsv[rg][3] = w0.w;
                wsv[rg][4] = w1.x; wsv[rg][5] = w1.y; wsv[rg][6] = w1.z; wsv[rg][7] = w1.w;
            }
            float xsv[NB];
            #pragma unroll
            for (int nb = 0; nb < NB; nb++) {
                xsv[nb] = xs_l[blk * BN + wave_tok0 + nb * 16 + m_lane];
            }
            auto issue = [&](int t) {
                const int nb = t / RG;
                const int rg = t % RG;
                // C seeded with 0x4B400000: as_float(c) - 12582912 is the exact integer dot
                // (|dot| <= 32*128*127 < 2^22, so 0x4B400000 + dot stays in the binade where
                // float == 12582912 + dot).
                q8g11_i32x8_t c = {Q8G11_MAGIC, Q8G11_MAGIC, Q8G11_MAGIC, Q8G11_MAGIC,
                                   Q8G11_MAGIC, Q8G11_MAGIC, Q8G11_MAGIC, Q8G11_MAGIC};
                c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af0[rg], true, bf0[nb], c, false);
                c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af1[rg], true, bf1[nb], c, false);
                return c;
            };
            auto fold = [&](int t, q8g11_i32x8_t c) {
                const int nb = t / RG;
                const int rg = t % RG;
                // Whole-vector bitcast + subtract: the per-element form
                // `bit_cast<float>(c[j]) - M` is miscompiled by the hipcc this was written
                // against (AMD clang 24.0.0git, d6f6cb69).
                const q8g11_f32x8_t cf = __builtin_bit_cast(q8g11_f32x8_t, c) - 12582912.0f;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc[rg][nb][j] = __builtin_fmaf(wsv[rg][j] * xsv[nb], cf[j], acc[rg][nb][j]);
                }
            };
            // Software pipeline: the WMMAs of tile t+1 are issued before the fold of tile t and
            // sched_barrier pins that order. Every accumulator still sees its K32 blocks in
            // ascending order with the same per-element operations.
            constexpr int NTILE = RG * NB;
            q8g11_i32x8_t ccur = issue(0);
            #pragma unroll
            for (int t = 0; t < NTILE; t++) {
                q8g11_i32x8_t cnext = ccur;
                if (t + 1 < NTILE) cnext = issue(t + 1);
                __builtin_amdgcn_sched_barrier(0);
                fold(t, ccur);
                __builtin_amdgcn_sched_barrier(0);
                ccur = cnext;
            }
        }
        __syncthreads();   // every wave is done reading the single buffer
        if (more) store_slab();
        __syncthreads();
    }

    // Epilogue: acc[rg][nb][j] = Y[token = tile_n + wave_tok0 + nb*16 + m_lane]
    //                             [row   = tile_m + wave_row0 + rg*16 + 8*hi + j].
    if constexpr (RBF16) {
        #pragma unroll
        for (int rg = 0; rg < RG; rg++) {
            #pragma unroll
            for (int nb = 0; nb < NB; nb++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) acc[rg][nb][j] = q8g11_bf16_rn(acc[rg][nb][j]);
            }
        }
    }
    const bool vec_ok = ((M & 7) == 0) && (((uintptr_t)Y & 15u) == 0);
    #pragma unroll
    for (int nb = 0; nb < NB; nb++) {
        const int tok = tile_n + wave_tok0 + nb * 16 + m_lane;
        if (tok >= N) continue;
        #pragma unroll
        for (int rg = 0; rg < RG; rg++) {
            const int row0 = tile_m + wave_row0 + rg * 16 + 8 * hi;
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

#define Q8G11_KERNEL(NAME, BM, BN, WM, WN, RBF16)                                     \
    extern "C" __global__ __launch_bounds__(2 * (BM), 2) void NAME(              \
        const char* __restrict__ Wq, const block_q8_1_mmq* __restrict__ Xq,      \
        float* __restrict__ Y, int M, int K, int N) {                            \
        gemm_q8_0_mmq_i8_gfx11_body<BM, BN, WM, WN, RBF16>(Wq, Xq, Y, M, K, N);         \
    }

Q8G11_KERNEL(gemm_q8_0_mmq_i8_gfx11_t64x64, 64, 64, 2, 2, false)
Q8G11_KERNEL(gemm_q8_0_mmq_i8_gfx11_t64x128, 64, 128, 2, 2, false)
Q8G11_KERNEL(gemm_q8_0_mmq_i8_gfx11_t128x128, 128, 128, 4, 2, false)
Q8G11_KERNEL(gemm_q8_0_mmq_i8_gfx11_t64x64_bf16, 64, 64, 2, 2, true)
Q8G11_KERNEL(gemm_q8_0_mmq_i8_gfx11_t64x128_bf16, 64, 128, 2, 2, true)
Q8G11_KERNEL(gemm_q8_0_mmq_i8_gfx11_t128x128_bf16, 128, 128, 4, 2, true)

#undef Q8G11_KERNEL

#else

#define Q8G11_STUB(NAME) \
    extern "C" __global__ void NAME(const char*, const void*, float*, int, int, int) {}
Q8G11_STUB(gemm_q8_0_mmq_i8_gfx11_t64x64)
Q8G11_STUB(gemm_q8_0_mmq_i8_gfx11_t64x128)
Q8G11_STUB(gemm_q8_0_mmq_i8_gfx11_t128x128)
Q8G11_STUB(gemm_q8_0_mmq_i8_gfx11_t64x64_bf16)
Q8G11_STUB(gemm_q8_0_mmq_i8_gfx11_t64x128_bf16)
Q8G11_STUB(gemm_q8_0_mmq_i8_gfx11_t128x128_bf16)
#undef Q8G11_STUB

#endif
