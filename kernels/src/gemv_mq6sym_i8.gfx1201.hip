// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// MQ6SymG256 x A8 decode GEMV for gfx1201 (RDNA4), N = 1..8 tokens, signed dot4.
//
// Format MQ6SymG256 (KERNEL-PLAN §1.4, symtrunk/RESULT.md §0; NOT qt47):
// row-major [M, K], K % 256 == 0, group (256 weights) = 196 B:
//   +0  fp16 s0 (weights 0..127)      +2  fp16 s1 (weights 128..255)
//   +4  192 B little-endian 6-bit bitstream, code c = bits [6c, 6c+6) (4 codes
//       per 3 bytes: q0 | q1 << 6 | q2 << 12 | q3 << 18), stored q in 0..63,
//       decoded signed int8 value q - 32 in -32..31 (NOT two's complement).
// Row stride (K / 256) * 196 B (4-byte aligned only: every load is dword-aligned).
// w = f32(s_h) * (q - 32): no zero point, no correction term.
//
// Activation: `block_i8_128` (136 B {f32 d; i32 s; i8 qs[128]}), layout
// [K/128][N], produced by the fused FWHT + A8 producer (mq_rotate_x_i8_gfx12);
// the struct comes from the host prelude (block_i8_128_quant.hip). Byte order
// of qs ("nibble-pair order"): per 8-element K group g, qs[8g+0..3] = q[8g+0,
// 2, 4, 6] and qs[8g+4..7] = q[8g+1, 3, 5, 7].
//
// Arithmetic (one definition shared with the CPU oracle in
// crates/rdna-compute/tests/mq6sym_i8_decode_hw.rs):
//   for every 128-K block b in increasing order:
//     C  = sum_{k in b} (q_w[k] - 32) * qx[k]        exact int32, |C| <= 520192
//     sd = RN(f32(s_h) * d_b,n)                       h = b & 1
//     acc = fmaf(sd, (float)C, acc)                   acc starts at +0.0f
//   Y[n * M + row] = acc   (F32; the BF16 entries round with the consumer
//   formula isfinite(v) ? (u + 0x7FFF + ((u >> 16) & 1)) >> 16 : u >> 16,
//   u = bits(0.0f + v)).
// No FP atomics, no reassociated f32 sums: the integer C is order-free, the f32
// fold is one sequential chain per (row, token).
//
// Signed codes without a per-byte subtract: the kernel feeds the int8 dot4 the
// byte 4 * (q - 32) = ((q ^ 32) << 2) as int8 (flip bit 5 of every 6-bit field
// in the raw bitstream, place the field at bits 2..7 of its byte), so the dot4
// sum is exactly 4 * C and `C = sum >> 2` (arithmetic, exact; |4C| < 2^22).
//
// Entries (6 explicit args, 36 B padded to 16):
//   gemv_mq6sym_i8_rows_r{1..8}_gfx1201      (const u8* W, const block_i8_128* Xq,
//   gemv_mq6sym_i8_rows_r{1..8}_bf16_gfx1201  f32|u16* Y, i32 M, i32 K, i32 N)
// N must equal the entry's row count (the entry returns without writing
// otherwise; the host picks the entry from N). K % 256 == 0, M >= 1.
// Grid [ceil(M / 32), 1, 1], block 128 (four waves, eight output rows per wave),
// dynamic LDS = mq6sym_x_lds_bytes(K, N) = min(K/256, S6_CHUNK_GROUPS) * 2 * N * 132 B.
// Y is [N][M] row-major, overwritten. Rows >= M are never stored (weight loads
// clamp to row M - 1).
//
// Lane map (wave32): row = lane >> 2 (8 rows per wave), quarter q = lane & 3.
// Per 256-K group lane q owns codes [64 q, 64 q + 64): half h = q >> 1 (block
// 2g + h), codes 64 (q & 1) .. +63 inside that block = 12 payload dwords = three
// dword-aligned b128 loads. All N tokens' activations come from LDS (the CTA
// stages the A8 chunk once; the eight rows of a wave broadcast). Per block the
// four partial integer sums of a quad are reduce-scattered with DPP: pair
// exchange (xor 1) adds the two halves of a block and splits the tokens, the
// cross-pair exchange (xor 2) hands every lane both blocks' C for its own
// tokens (2 tokens/lane at N = 8, 1 at N = 4, replicated below), and each lane
// runs the ordered f32 fold (block 2g, then 2g + 1) only for its tokens.
//
// Test-only: compiling with -DS6_DEBUG_C also stores the per-block integer C
// ([K/128][N][M] i32 after Y's N*M words); never defined in production.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#if defined(__gfx1201__)

#ifndef S6_CHUNK_GROUPS
#define S6_CHUNK_GROUPS 12
#endif
// Weight-load lookahead in 256-K groups (1 or 2).
#ifndef S6_PF
#define S6_PF 2
#endif
#define S6_GROUP_BYTES 196u

typedef unsigned __attribute__((ext_vector_type(4), aligned(4))) s6_u32x4_t;

static __device__ __forceinline__ int s6_dot4(unsigned w, unsigned x, int acc) {
    return __builtin_amdgcn_sudot4(true, (int)w, true, (int)x, acc, false);
}

static __device__ __forceinline__ int s6_dpp_x1(int v) {
    return __builtin_amdgcn_update_dpp(0, v, 0xB1, 0xF, 0xF, false);  // quad_perm [1,0,3,2]
}
static __device__ __forceinline__ int s6_dpp_x2(int v) {
    return __builtin_amdgcn_update_dpp(0, v, 0x4E, 0xF, 0xF, false);  // quad_perm [2,3,0,1]
}

static __device__ __forceinline__ float s6_f16_lo(unsigned w) {
    return __half2float(__ushort_as_half((unsigned short)(w & 0xFFFFu)));
}
static __device__ __forceinline__ float s6_f16_hi(unsigned w) {
    return __half2float(__ushort_as_half((unsigned short)(w >> 16)));
}

static __device__ __forceinline__ unsigned short s6_bf16_bits(float v) {
    const unsigned u = __float_as_uint(0.0f + v);
    const unsigned r = isfinite(v) ? ((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16) : (u >> 16);
    return (unsigned short)r;
}

struct s6_wgroup {
    s6_u32x4_t a, b, c;
    unsigned hdr;
};

static __device__ __forceinline__ s6_wgroup s6_load(const unsigned char* __restrict__ row, int g, int q) {
    const unsigned char* gp = row + (size_t)g * S6_GROUP_BYTES;
    s6_wgroup w;
    w.hdr = *reinterpret_cast<const unsigned*>(gp);
    const s6_u32x4_t* cp = reinterpret_cast<const s6_u32x4_t*>(gp + 4 + 48 * q);
    w.a = cp[0];
    w.b = cp[1];
    w.c = cp[2];
    return w;
}

// Eight codes = six stream bytes B0..B5 (bit offset of code j is 6 j). With the
// bit-5-flipped fields placed at bits 2..7 of a byte (value 4 (q - 32)):
//   e0 = {f0, f2, f4, f6}   e1 = {f1, f3, f5, f7}   (the qs pair order)
// U01 = [B0, B1, B3, B4], U12 = [B1, B2, B4, B5] (byte 0 first):
//   f0,f4 = (U01 << 2) & 0xFC at bytes 0,2      f2,f6 = (U12 << 6) & 0xFC00 at bytes 1,3
//   f1,f5 = (U01 >> 4) & 0xFC at bytes 0,2      f3,f7 =  U12       & 0xFC00 at bytes 1,3
static __device__ __forceinline__ void s6_pair8(unsigned u01, unsigned u12, unsigned& e0, unsigned& e1) {
    e0 = (((u01 << 2) & 0x00FC00FCu)) | ((u12 << 6) & 0xFC00FC00u);
    e1 = ((u01 >> 4) & 0x00FC00FCu) | (u12 & 0xFC00FC00u);
}

// 12 bytes (w0, w1, w2 of the stream) -> four dwords of qs-ordered 4 (q - 32) bytes.
static __device__ __forceinline__ void s6_decode16(unsigned w0, unsigned w1, unsigned w2, unsigned* e) {
    // flip bit 5 of every 6-bit field of the raw 96-bit window
    w0 ^= 0x20820820u;
    w1 ^= 0x08208208u;
    w2 ^= 0x82082082u;
    // first eight codes: bytes 0..5 = w0.b0..b3, w1.b0, w1.b1
    s6_pair8(__builtin_amdgcn_perm(w1, w0, 0x04030100u), __builtin_amdgcn_perm(w1, w0, 0x05040201u), e[0], e[1]);
    // second eight codes: bytes 6..11 = w1.b2, w1.b3, w2.b0..b3
    s6_pair8(__builtin_amdgcn_perm(w2, w1, 0x06050302u), __builtin_amdgcn_perm(w2, w1, 0x07060403u), e[2], e[3]);
}

template <int R> struct S6Np { static constexpr int v = R <= 1 ? 1 : (R <= 2 ? 2 : (R <= 4 ? 4 : 8)); };

template <int R, bool BF16>
static __device__ __forceinline__ void s6_body(
    const unsigned char* __restrict__ W,
    const block_i8_128* __restrict__ Xq,
    void* __restrict__ Yv,
    int M, int K, int N
) {
    extern __shared__ __align__(16) unsigned char s6_lds[];
    if (N != R) return;
    constexpr int NP = S6Np<R>::v;          // tokens padded to a power of two
    constexpr int KT = NP >= 8 ? 2 : 1;     // tokens folded per lane
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int q = lane & 3;
    const int b0 = q & 1;
    const int h = q >> 1;                   // == b1: block half of this lane
    const int row = (int)blockIdx.x * 32 + wave * 8 + (lane >> 2);
    const int rowc = row < M ? row : M - 1;
    const int G = K >> 8;
    const size_t row_stride = (size_t)G * S6_GROUP_BYTES;
    const unsigned char* wrow = W + (size_t)rowc * row_stride;

    // First token of this lane's KT tokens (see the stage comments below).
    int tokb;
    if (NP == 1) tokb = 0;
    else if (NP == 2) tokb = b0;
    else if (NP == 4) tokb = 2 * b0 + h;
    else tokb = 4 * b0 + 2 * h;
    // Unique writer of the token(s) this lane folds.
    const bool writer = NP == 1 ? (q == 0) : (NP == 2 ? (h == 0) : true);

    float acc[KT];
    #pragma unroll
    for (int j = 0; j < KT; ++j) acc[j] = 0.0f;

    s6_wgroup cur = s6_load(wrow, 0, q);
    s6_wgroup nx1 = cur;
    if (S6_PF >= 2) nx1 = (G > 1) ? s6_load(wrow, 1, q) : cur;

    for (int c0 = 0; c0 < G; c0 += S6_CHUNK_GROUPS) {
        const int cg = (G - c0) < S6_CHUNK_GROUPS ? (G - c0) : S6_CHUNK_GROUPS;
        const int nbk = 2 * cg * R;                 // (block, token) cells of the chunk
        unsigned char* qs_lds = s6_lds;              // [nbk][128 B]
        float* d_lds = reinterpret_cast<float*>(s6_lds + (size_t)nbk * 128);  // [nbk]
        // Stage the A8 chunk: blocks 2*c0 .. 2*(c0+cg), all N tokens (contiguous in Xq).
        {
            const block_i8_128* src = Xq + (size_t)(2 * c0) * N;
            const int units = nbk * 16;
            #pragma unroll 4
            for (int u = tid; u < units; u += 128) {
                const int j = u >> 4, c = u & 15;
                const uint2 v = *reinterpret_cast<const uint2*>(
                    reinterpret_cast<const unsigned char*>(src + j) + 8 + 8 * c);
                *reinterpret_cast<uint2*>(qs_lds + (size_t)j * 128 + 8 * c) = v;
            }
            for (int j = tid; j < nbk; j += 128) d_lds[j] = src[j].d;
        }
        __syncthreads();

        for (int gg = 0; gg < cg; ++gg) {
            const int g = c0 + gg;
            s6_wgroup nx2 = nx1;
            if (S6_PF >= 2) {
                if (g + 2 < G) nx2 = s6_load(wrow, g + 2, q);
            } else {
                if (g + 1 < G) nx2 = s6_load(wrow, g + 1, q);
            }

            // Decode the lane's 64 codes -> 16 dwords of 4 (q - 32) bytes (4 triples x 4).
            unsigned e[16];
            s6_decode16(cur.a.x, cur.a.y, cur.a.z, e + 0);
            s6_decode16(cur.a.w, cur.b.x, cur.b.y, e + 4);
            s6_decode16(cur.b.z, cur.b.w, cur.c.x, e + 8);
            s6_decode16(cur.c.y, cur.c.z, cur.c.w, e + 12);

            const unsigned char* xb = qs_lds + (size_t)((2 * gg + h) * R) * 128 + 64 * (q & 1);
            int c[NP];
            #pragma unroll
            for (int n = 0; n < NP; ++n) {
                c[n] = 0;
                if (n < R) {
                    const uint4* xp = reinterpret_cast<const uint4*>(xb + n * 128);
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        const uint4 xv = xp[t];
                        c[n] = s6_dot4(e[4 * t + 0], xv.x, c[n]);
                        c[n] = s6_dot4(e[4 * t + 1], xv.y, c[n]);
                        c[n] = s6_dot4(e[4 * t + 2], xv.z, c[n]);
                        c[n] = s6_dot4(e[4 * t + 3], xv.w, c[n]);
                    }
                }
            }

            // Reduce-scatter. own[j] = C(4x) of this lane's block half h for token tokb + j,
            // oth[j] = the same token's C(4x) of the other half.
            int own[KT], oth[KT];
            if constexpr (NP == 1) {
                const int p = c[0] + s6_dpp_x1(c[0]);
                own[0] = p;
                oth[0] = s6_dpp_x2(p);
            } else if constexpr (NP == 2) {
                const int keep = b0 ? c[1] : c[0];
                const int send = b0 ? c[0] : c[1];
                const int p = keep + s6_dpp_x1(send);
                own[0] = p;
                oth[0] = s6_dpp_x2(p);
            } else if constexpr (NP == 4) {
                int p[2];
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int keep = b0 ? c[2 + i] : c[i];
                    const int send = b0 ? c[i] : c[2 + i];
                    p[i] = keep + s6_dpp_x1(send);
                }
                own[0] = h ? p[1] : p[0];
                oth[0] = s6_dpp_x2(h ? p[0] : p[1]);
            } else {
                int p[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int keep = b0 ? c[4 + i] : c[i];
                    const int send = b0 ? c[i] : c[4 + i];
                    p[i] = keep + s6_dpp_x1(send);
                }
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    own[j] = h ? p[2 + j] : p[j];
                    oth[j] = s6_dpp_x2(h ? p[j] : p[2 + j]);
                }
            }

            const float s0f = s6_f16_lo(cur.hdr);
            const float s1f = s6_f16_hi(cur.hdr);
            #pragma unroll
            for (int j = 0; j < KT; ++j) {
                const int tok = tokb + j;
                const int tr = tok < R ? tok : R - 1;
                const int C0 = (h ? oth[j] : own[j]) >> 2;
                const int C1 = (h ? own[j] : oth[j]) >> 2;
#ifdef S6_DEBUG_C
                // Test-only: per-block integer C after Y's N*M words ([K/128][N][M] i32).
                if (writer && tok < R && row < M) {
                    int* dbg = reinterpret_cast<int*>(Yv) + (size_t)N * M;
                    dbg[((size_t)(2 * g) * R + tok) * M + row] = C0;
                    dbg[((size_t)(2 * g + 1) * R + tok) * M + row] = C1;
                }
#endif
                const float d0 = d_lds[(2 * gg) * R + tr];
                const float d1 = d_lds[(2 * gg + 1) * R + tr];
                acc[j] = __builtin_fmaf(__fmul_rn(s0f, d0), (float)C0, acc[j]);
                acc[j] = __builtin_fmaf(__fmul_rn(s1f, d1), (float)C1, acc[j]);
            }
            if (S6_PF >= 2) {
                cur = nx1;
                nx1 = nx2;
            } else {
                cur = nx2;
            }
        }
        if (c0 + S6_CHUNK_GROUPS < G) __syncthreads();
    }

    if (row < M && writer) {
        #pragma unroll
        for (int j = 0; j < KT; ++j) {
            const int tok = tokb + j;
            if (tok < R) {
                if (BF16) {
                    reinterpret_cast<unsigned short*>(Yv)[(size_t)tok * M + row] = s6_bf16_bits(acc[j]);
                } else {
                    reinterpret_cast<float*>(Yv)[(size_t)tok * M + row] = acc[j];
                }
            }
        }
    }
}

#ifdef S6_DEBUG_C
// Test-only build: F32 entries only, distinct symbol names (`_dbgc_`).
#define S6_ENTRY(R)                                                                              \
extern "C" __global__ __launch_bounds__(128) void gemv_mq6sym_i8_rows_r##R##_dbgc_gfx1201(     \
    const unsigned char* __restrict__ W, const block_i8_128* __restrict__ Xq,                   \
    float* __restrict__ Y, int M, int K, int N) {                                               \
    s6_body<R, false>(W, Xq, Y, M, K, N);                                                       \
}
#else
#define S6_ENTRY(R)                                                                              \
extern "C" __global__ __launch_bounds__(128) void gemv_mq6sym_i8_rows_r##R##_gfx1201(          \
    const unsigned char* __restrict__ W, const block_i8_128* __restrict__ Xq,                   \
    float* __restrict__ Y, int M, int K, int N) {                                               \
    s6_body<R, false>(W, Xq, Y, M, K, N);                                                       \
}                                                                                               \
extern "C" __global__ __launch_bounds__(128) void gemv_mq6sym_i8_rows_r##R##_bf16_gfx1201(     \
    const unsigned char* __restrict__ W, const block_i8_128* __restrict__ Xq,                   \
    float* __restrict__ Y, int M, int K, int N) {                                               \
    s6_body<R, true>(W, Xq, Y, M, K, N);                                                        \
}
#endif

S6_ENTRY(1)
S6_ENTRY(2)
S6_ENTRY(3)
S6_ENTRY(4)
S6_ENTRY(5)
S6_ENTRY(6)
S6_ENTRY(7)
S6_ENTRY(8)

#endif  // __gfx1201__
