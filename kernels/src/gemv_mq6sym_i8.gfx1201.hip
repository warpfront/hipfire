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
// dynamic LDS = mq6sym_x_lds_bytes(N) = 2 * (2 * (N + B) * 144 + 8 * N) bytes, B = 2 if N == 7
// else 1 (the larger of the two body layouts, >= 528 * N, so one formula serves every entry;
// 1168 B at N = 1, 5312 B at N = 8).
// Y is [N][M] row-major, overwritten. Rows >= M are never stored (weight loads
// clamp to row M - 1).
//
// Two bodies, picked per entry by S6_TS_MIN (default 3):
//   N = 1, 2  K-split ("ks"): row = lane >> 2, quarter q = lane & 3. Per 256-K group lane q
//     owns codes [64 q, 64 q + 64): block half h = q >> 1 (block 2g + h), codes 64 (q & 1) ..
//     +63 of that block = 12 payload dwords = three dword-aligned b128 loads, all N tokens.
//     The four partial integer sums of a quad are reduce-scattered with DPP (xor 1 adds the two
//     halves of a block, xor 2 hands every lane both blocks' C for its tokens).
//   N >= 3    token-split ("ts"): lane (row, q = 2 tp + kh) owns the 128 codes of block kh of every
//     group (six b128 loads) for the tokens of token half tp ([tp * T0, tp * T0 + MT),
//     T0 = MT = ceil(N / 2)). A block's C never crosses lanes; only the two lanes of a (row,
//     tp) pair exchange C (DPP xor 1), each folding half of the pair's tokens.
// Both read every token's activations from LDS: a two-slot ring refilled one K256 group
// ahead (one b128 per thread, a barrier that drains only LDS so the weight loads in flight
// stay in flight); the eight rows of a wave broadcast. The ts layout pads every 128-B
// (block, token) cell to 144 B (and R = 7 by one more cell per block) so the four lanes of a
// row read distinct LDS bank groups. Each lane then runs the ordered f32 fold (block 2g, then
// 2g + 1) for its own tokens only.
//
// Test-only: compiling with -DS6_DEBUG_C also stores the per-block integer C
// ([K/128][N][M] i32 after Y's N*M words); never defined in production.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#if defined(__gfx1201__)


// Weight-load lookahead in 256-K groups (1 or 2); 1 measured faster at every cell.
#ifndef S6_PF
#define S6_PF 1
#endif
#define S6_GROUP_BYTES 196u

typedef unsigned __attribute__((ext_vector_type(4), aligned(4))) s6_u32x4_t;

// Timing-ablation bits (test only; 0 in production): 1 dot4 -> xor3, 2 no LDS x reads, 4 no DPP
// exchange, 8 no A8 staging/barriers, 16 no decode. Results are meaningless numerically.
#ifndef S6_ABL
#define S6_ABL 0
#endif
static __device__ __forceinline__ int s6_dot4(unsigned w, unsigned x, int acc) {
#if S6_ABL & 1
    return (int)(w ^ x ^ (unsigned)acc);
#else
    return __builtin_amdgcn_sudot4(true, (int)w, true, (int)x, acc, false);
#endif
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

// Streaming (non-temporal) weight loads: off by default (S6_NT=1 selects them).
#ifndef S6_NT
#define S6_NT 0
#endif

static __device__ __forceinline__ s6_wgroup s6_load(const unsigned char* __restrict__ row, int g, int q) {
    const unsigned char* gp = row + (size_t)g * S6_GROUP_BYTES;
    s6_wgroup w;
#if S6_NT
    w.hdr = __builtin_nontemporal_load(reinterpret_cast<const unsigned*>(gp));
    const s6_u32x4_t* cp = reinterpret_cast<const s6_u32x4_t*>(gp + 4 + 48 * q);
    w.a = __builtin_nontemporal_load(cp + 0);
    w.b = __builtin_nontemporal_load(cp + 1);
    w.c = __builtin_nontemporal_load(cp + 2);
#else
    w.hdr = *reinterpret_cast<const unsigned*>(gp);
    const s6_u32x4_t* cp = reinterpret_cast<const s6_u32x4_t*>(gp + 4 + 48 * q);
    w.a = cp[0];
    w.b = cp[1];
    w.c = cp[2];
#endif
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

// One thread's share of a group's A8 sidecar: group g = blocks 2g, 2g + 1 x R tokens = 2R (block,
// token) cells, contiguous in Xq ([K/128][N]). Threads 0 .. 16 R - 1 each move one 16-byte slice
// of a cell's qs (8 per cell; the qs array sits at +8, so loads are 8-byte aligned only); threads
// 0 .. 2 R - 1 also move one cell's f32 d.
struct s6_xreg {
    s6_u32x4_t v;
    float d;
};

template <int R>
static __device__ __forceinline__ s6_xreg s6_loadx(const block_i8_128* Xq, int N, int g, int tid) {
    s6_xreg r;
    r.v = (s6_u32x4_t){0u, 0u, 0u, 0u};
    r.d = 0.0f;
    const block_i8_128* cell0 = Xq + (size_t)(2 * g) * N;
    if (tid < 16 * R) {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(cell0 + (tid >> 3)) + 8 + 16 * (tid & 7);
        r.v = *reinterpret_cast<const s6_u32x4_t*>(p);
    }
    if (tid < 2 * R) r.d = cell0[tid].d;
    return r;
}

// LDS slot: qs [2][R][128 B] then d [2][R] f32.
template <int R>
static __device__ __forceinline__ void s6_storex(unsigned char* slot, const s6_xreg& r, int tid) {
    if (tid < 16 * R) {
        *reinterpret_cast<uint4*>(slot + (tid >> 3) * 128 + 16 * (tid & 7)) = make_uint4(r.v.x, r.v.y, r.v.z, r.v.w);
    }
    if (tid < 2 * R) reinterpret_cast<float*>(slot + 256 * R)[tid] = r.d;
}

// Workgroup barrier that only drains LDS (the weight loads in flight stay in flight).
static __device__ __forceinline__ void s6_sync_lds() {
    asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
}

template <int R, bool BF16>
static __device__ __forceinline__ void s6_body_ks(
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

    constexpr int XS = 264 * R;   // bytes per LDS slot: qs [2][R][128] + d [2][R] f32
    s6_wgroup cur = s6_load(wrow, 0, q);
    s6_wgroup nx1 = cur;
    if (S6_PF >= 2) nx1 = (G > 1) ? s6_load(wrow, 1, q) : cur;
    // A8 ring: group g's two blocks x R tokens live in LDS slot g & 1; the b128 for group g + 1
    // is already in flight (xr1), the one for g + 2 is issued at the top of iteration g.
#if !(S6_ABL & 8)
    s6_xreg xr1 = s6_loadx<R>(Xq, N, 0, tid);
    s6_storex<R>(s6_lds, xr1, tid);
    if (G > 1) xr1 = s6_loadx<R>(Xq, N, 1, tid);
    s6_sync_lds();
#else
    s6_xreg xr1 = s6_loadx<R>(Xq, N, 0, tid);
#endif

    for (int g = 0; g < G; ++g) {
        {
            s6_xreg xn = xr1;
#if !(S6_ABL & 8)
            if (g + 2 < G) xn = s6_loadx<R>(Xq, N, g + 2, tid);
#endif
            s6_wgroup nx2 = nx1;
            if (S6_PF >= 2) {
                if (g + 2 < G) nx2 = s6_load(wrow, g + 2, q);
            } else {
                if (g + 1 < G) nx2 = s6_load(wrow, g + 1, q);
            }
            unsigned char* slot = s6_lds + (g & 1) * XS;
            const float* d_lds = reinterpret_cast<const float*>(slot + 256 * R);

            // Decode the lane's 64 codes -> 16 dwords of 4 (q - 32) bytes (4 triples x 4).
            unsigned e[16];
#if S6_ABL & 16
            // ablation: no decode (wrong values, timing only)
            e[0] = cur.a.x; e[1] = cur.a.y; e[2] = cur.a.z; e[3] = cur.a.w;
            e[4] = cur.b.x; e[5] = cur.b.y; e[6] = cur.b.z; e[7] = cur.b.w;
            e[8] = cur.c.x; e[9] = cur.c.y; e[10] = cur.c.z; e[11] = cur.c.w;
            e[12] = cur.a.x ^ cur.c.w; e[13] = cur.a.y ^ cur.c.z; e[14] = cur.a.z ^ cur.c.y; e[15] = cur.a.w ^ cur.c.x;
#else
            s6_decode16(cur.a.x, cur.a.y, cur.a.z, e + 0);
            s6_decode16(cur.a.w, cur.b.x, cur.b.y, e + 4);
            s6_decode16(cur.b.z, cur.b.w, cur.c.x, e + 8);
            s6_decode16(cur.c.y, cur.c.z, cur.c.w, e + 12);
#endif

            const unsigned char* xb = slot + (size_t)(h * R) * 128 + 64 * (q & 1);
            int c[NP];
            #pragma unroll
            for (int n = 0; n < NP; ++n) {
                c[n] = 0;
                if (n < R) {
                    const uint4* xp = reinterpret_cast<const uint4*>(xb + n * 128);
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
#if S6_ABL & 2
                        // ablation: no LDS x reads (wrong values, timing only)
                        const uint4 xv = make_uint4(e[4 * t] + n, e[4 * t + 1] + n, e[4 * t + 2] + n, e[4 * t + 3] + n);
#else
                        const uint4 xv = xp[t];
#endif
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
#if S6_ABL & 4
            // ablation: no DPP exchange (wrong values, timing only)
            #pragma unroll
            for (int j = 0; j < KT; ++j) {
                own[j] = c[j];
                oth[j] = c[NP - 1 - j];
            }
#endif

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
                const float d0 = d_lds[tr];
                const float d1 = d_lds[R + tr];
                acc[j] = __builtin_fmaf(__fmul_rn(s0f, d0), (float)C0, acc[j]);
                acc[j] = __builtin_fmaf(__fmul_rn(s1f, d1), (float)C1, acc[j]);
            }
            if (S6_PF >= 2) {
                cur = nx1;
                nx1 = nx2;
            } else {
                cur = nx2;
            }
            // Publish group g + 1's A8 into the other slot (its last readers passed the previous
            // barrier), then release it to every wave.
#if !(S6_ABL & 8)
            if (g + 1 < G) s6_storex<R>(s6_lds + ((g + 1) & 1) * XS, xr1, tid);
#endif
            xr1 = xn;
        }
#if !(S6_ABL & 8)
        if (g + 1 < G) s6_sync_lds();
#endif
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

// ---------------------------------------------------------------------------------------------
// Token-split body (R >= S6_TS_MIN): lane (row, q = 2 tp + kh) owns the 128 codes of block kh of
// every group (so a block's integer C never crosses lanes) and the tokens of token-half tp
// (tokens [tp * T0, tp * T0 + MT)). Only the two lanes of a (row, tp) pair (xor 1) exchange C:
// each lane folds half of the pair's tokens and receives the other block's C for them.
// ---------------------------------------------------------------------------------------------
template <int R> struct S6Ts {
    static constexpr int T0 = (R + 1) / 2;       // tokens of half 0 (max per half)
    static constexpr int MT = T0;
    static constexpr int HJ = (MT + 1) / 2;      // owned tokens per lane (max)
    static constexpr int B = (R == 7) ? 2 : 1;   // padding cells per block: bank rotation of the
                                                 // four lanes' cells (stride 144 B = 9 chunks)
    static constexpr int CELLS = 2 * (R + B);
    static constexpr int SLOT = CELLS * 144 + 8 * R;  // qs cells, then d [2][R] f32
};

struct s6_wgroup2 {
    s6_u32x4_t v[6];
    unsigned hdr;
};

static __device__ __forceinline__ s6_wgroup2 s6_load2(const unsigned char* __restrict__ row, int g, int kh) {
    const unsigned char* gp = row + (size_t)g * S6_GROUP_BYTES;
    s6_wgroup2 w;
    w.hdr = *reinterpret_cast<const unsigned*>(gp);
    const s6_u32x4_t* cp = reinterpret_cast<const s6_u32x4_t*>(gp + 4 + 96 * kh);
    #pragma unroll
    for (int i = 0; i < 6; ++i) w.v[i] = cp[i];
    return w;
}

template <int R>
static __device__ __forceinline__ void s6_storex_ts(unsigned char* slot, const s6_xreg& r, int tid) {
    constexpr int B = S6Ts<R>::B;
    if (tid < 16 * R) {
        const int cc = tid >> 3;
        const int hh = cc / R;
        const int n = cc - hh * R;
        *reinterpret_cast<uint4*>(slot + (hh * (R + B) + n) * 144 + 16 * (tid & 7)) = make_uint4(r.v.x, r.v.y, r.v.z, r.v.w);
    }
    if (tid < 2 * R) reinterpret_cast<float*>(slot + S6Ts<R>::CELLS * 144)[tid] = r.d;
}

template <int R, bool BF16>
static __device__ __forceinline__ void s6_body_ts(
    const unsigned char* __restrict__ W,
    const block_i8_128* __restrict__ Xq,
    void* __restrict__ Yv,
    int M, int K, int N
) {
    extern __shared__ __align__(16) unsigned char s6_lds[];
    if (N != R) return;
    using TS = S6Ts<R>;
    constexpr int T0 = TS::T0, MT = TS::MT, HJ = TS::HJ, B = TS::B, SLOT = TS::SLOT;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int q = lane & 3;
    const int kh = q & 1;                   // block half owned by this lane
    const int tp = q >> 1;                  // token half
    const int row = (int)blockIdx.x * 32 + wave * 8 + (lane >> 2);
    const int rowc = row < M ? row : M - 1;
    const int G = K >> 8;
    const size_t row_stride = (size_t)G * S6_GROUP_BYTES;
    const unsigned char* wrow = W + (size_t)rowc * row_stride;
    const int tbase = tp * T0;

    float acc[HJ];
    #pragma unroll
    for (int i = 0; i < HJ; ++i) acc[i] = 0.0f;

    s6_wgroup2 cur = s6_load2(wrow, 0, kh);
    s6_wgroup2 nx1 = cur;
    if (S6_PF >= 2) nx1 = (G > 1) ? s6_load2(wrow, 1, kh) : cur;
    s6_xreg xr1 = s6_loadx<R>(Xq, N, 0, tid);
    s6_storex_ts<R>(s6_lds, xr1, tid);
    if (G > 1) xr1 = s6_loadx<R>(Xq, N, 1, tid);
    s6_sync_lds();

    for (int g = 0; g < G; ++g) {
        {
            s6_xreg xn = xr1;
#if !(S6_ABL & 8)
            if (g + 2 < G) xn = s6_loadx<R>(Xq, N, g + 2, tid);
#endif
            s6_wgroup2 nx2 = nx1;
            if (S6_PF >= 2) {
                if (g + 2 < G) nx2 = s6_load2(wrow, g + 2, kh);
            } else {
                if (g + 1 < G) nx2 = s6_load2(wrow, g + 1, kh);
            }
            unsigned char* slot = s6_lds + (g & 1) * SLOT;
            const float* d_lds = reinterpret_cast<const float*>(slot + TS::CELLS * 144);
            const unsigned char* xb = slot + (size_t)(kh * (R + B) + tbase) * 144;

            int c[MT];
            #pragma unroll
            for (int j = 0; j < MT; ++j) c[j] = 0;
            #pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                unsigned e[16];
                const s6_u32x4_t va = cur.v[3 * h2], vb = cur.v[3 * h2 + 1], vc = cur.v[3 * h2 + 2];
#if S6_ABL & 16
                for (int u = 0; u < 12; ++u) e[u] = cur.v[3 * h2 + u / 4][u & 3];
                e[12] = va.x ^ vc.w; e[13] = va.y ^ vc.z; e[14] = va.z ^ vc.y; e[15] = va.w ^ vc.x;
#else
                s6_decode16(va.x, va.y, va.z, e + 0);
                s6_decode16(va.w, vb.x, vb.y, e + 4);
                s6_decode16(vb.z, vb.w, vc.x, e + 8);
                s6_decode16(vc.y, vc.z, vc.w, e + 12);
#endif
                #pragma unroll
                for (int j = 0; j < MT; ++j) {
                    const uint4* xp = reinterpret_cast<const uint4*>(xb + j * 144 + 64 * h2);
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
#if S6_ABL & 2
                        const uint4 xv = make_uint4(e[4 * t] + j, e[4 * t + 1] + j, e[4 * t + 2] + j, e[4 * t + 3] + j);
#else
                        const uint4 xv = xp[t];
#endif
                        c[j] = s6_dot4(e[4 * t + 0], xv.x, c[j]);
                        c[j] = s6_dot4(e[4 * t + 1], xv.y, c[j]);
                        c[j] = s6_dot4(e[4 * t + 2], xv.z, c[j]);
                        c[j] = s6_dot4(e[4 * t + 3], xv.w, c[j]);
                    }
                }
            }

            // Pair exchange: lane kh owns slots j = kh * HJ + i; it sends the partner's slots.
            const float s0f = s6_f16_lo(cur.hdr);
            const float s1f = s6_f16_hi(cur.hdr);
            #pragma unroll
            for (int i = 0; i < HJ; ++i) {
                constexpr int ZERO = 0;
                const int hi = (HJ + i < MT) ? c[(HJ + i < MT) ? HJ + i : 0] : ZERO;
                const int own = kh ? hi : c[i];
                const int snd = kh ? c[i] : hi;
#if S6_ABL & 4
                const int rcv = snd;   // ablation: no DPP exchange
#else
                const int rcv = s6_dpp_x1(snd);
#endif
                const int C0 = (kh ? rcv : own) >> 2;
                const int C1 = (kh ? own : rcv) >> 2;
                const int tok = tbase + kh * HJ + i;
                const int tr = tok < R ? tok : R - 1;
#ifdef S6_DEBUG_C
                if (kh * HJ + i < MT && tok < R && row < M) {
                    int* dbg = reinterpret_cast<int*>(Yv) + (size_t)N * M;
                    dbg[((size_t)(2 * g) * R + tok) * M + row] = C0;
                    dbg[((size_t)(2 * g + 1) * R + tok) * M + row] = C1;
                }
#endif
                const float d0 = d_lds[tr];
                const float d1 = d_lds[R + tr];
                acc[i] = __builtin_fmaf(__fmul_rn(s0f, d0), (float)C0, acc[i]);
                acc[i] = __builtin_fmaf(__fmul_rn(s1f, d1), (float)C1, acc[i]);
            }
            if (S6_PF >= 2) {
                cur = nx1;
                nx1 = nx2;
            } else {
                cur = nx2;
            }
#if !(S6_ABL & 8)
            if (g + 1 < G) s6_storex_ts<R>(s6_lds + ((g + 1) & 1) * SLOT, xr1, tid);
#endif
            xr1 = xn;
        }
#if !(S6_ABL & 8)
        if (g + 1 < G) s6_sync_lds();
#endif
    }

    if (row < M) {
        #pragma unroll
        for (int i = 0; i < HJ; ++i) {
            const int j = kh * HJ + i;
            const int tok = tbase + j;
            if (j < MT && tok < R) {
                if (BF16) {
                    reinterpret_cast<unsigned short*>(Yv)[(size_t)tok * M + row] = s6_bf16_bits(acc[i]);
                } else {
                    reinterpret_cast<float*>(Yv)[(size_t)tok * M + row] = acc[i];
                }
            }
        }
    }
}

// Token-split from this row count up (the K-split body below 3 rows).
#ifndef S6_TS_MIN
#define S6_TS_MIN 3
#endif
template <int R, bool BF16>
static __device__ __forceinline__ void s6_body(
    const unsigned char* __restrict__ W, const block_i8_128* __restrict__ Xq,
    void* __restrict__ Yv, int M, int K, int N) {
    if constexpr (R >= S6_TS_MIN) s6_body_ts<R, BF16>(W, Xq, Yv, M, K, N);
    else s6_body_ks<R, BF16>(W, Xq, Yv, M, K, N);
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
// Test-only hook: -DS6_SUF=_name appends a suffix to every entry symbol (A/B variants in one
// process); empty in production.
#ifndef S6_SUF
#define S6_SUF
#endif
#define S6_CAT4_(a, b, c, d) a##b##c##d
#define S6_CAT4(a, b, c, d) S6_CAT4_(a, b, c, d)
#define S6_ENTRY(R)                                                                              \
extern "C" __global__ __launch_bounds__(128) void S6_CAT4(gemv_mq6sym_i8_rows_r, R, S6_SUF, _gfx1201)( \
    const unsigned char* __restrict__ W, const block_i8_128* __restrict__ Xq,                   \
    float* __restrict__ Y, int M, int K, int N) {                                               \
    s6_body<R, false>(W, Xq, Y, M, K, N);                                                       \
}                                                                                               \
extern "C" __global__ __launch_bounds__(128) void S6_CAT4(gemv_mq6sym_i8_rows_r, R, S6_SUF, _bf16_gfx1201)( \
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
