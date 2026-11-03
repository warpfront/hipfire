// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// MQ6SymG256 exact-INT8 GEMM for gfx12 (RDNA4 gfx1200/gfx1201): `block_i8_128`
// activations x SYMMETRIC 6-bit weight codes on the SIGNED iu8 WMMA
// (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`, signed A and signed B).
// KERNEL-PLAN §1.4 / symtrunk RESULT §0 format:
//   weights [M, K] row-major, K % 256 == 0, 196-B groups per 256 K:
//     +0 fp16 s0 (weights 0..127), +2 fp16 s1 (weights 128..255), +4 192 B of
//     little-endian 6-bit fields (4 codes / 3 bytes: q0 | q1<<6 | q2<<12 |
//     q3<<18), stored q in 0..63, decoded signed int8 value q - 32 (-32..31).
//   activations: block_i8_128 (136 B {f32 d; i32 s; i8 qs[128]}), layout
//     [K/128][N], qs in the "nibble-pair order" of block_i8_128_quant.hip.
// Arithmetic (frozen, equal to the dot4 kernel bytewise): per 128-K block h of a
// row/token pair the exact i32 C = sum (q-32) * qx (|C| <= 520192, exact in
// f32), sd = RN(f32(s_h) * d), acc = fmaf(sd, (float)C, acc) in increasing
// block order from +0.0; the final F32 store is acc, the BF16 store the
// incumbent RNE formula of the affine i8 consumer. No zero-point correction, no
// seed trick: the codes are decoded to SIGNED bytes once per CTA/wave and every
// 128-K block's WMMA accumulator is folded with its own scale before the next
// block starts.
//
// Entries (all share the ABI (const u8* W, const block_i8_128* Xq, void* Y,
// i32 M, i32 K, i32 N), 36 explicit bytes, Y = [N][M] row-major overwrite, F32 or
// (`_bf16`) BF16 bits; EXPLICIT entries, the launcher picks one by (M, N):
//   gemm_mq6sym_i8_set_gfx1201 / gemm_mq6sym_i8_set_bf16_gfx1201
//       prefill 128x128 tile (any N, tails masked), grid [ceil(M/128),
//       ceil(N/128)], block 256, dynamic LDS MQ6SYM_P128_LDS_BYTES (36864).
//   gemm_mq6sym_i8_set_m{32|64}n{16|32|48}_gfx1201 (kept: m32n16, m32n32, m64n16, m64n32,
//   m64n48) and gemm_mq6sym_i8_set_m64n16y_gfx1201, each with a `_bf16` twin:
//       narrow verify tiles for N <= 16 / 32 / 48 (tokens >= N are masked): a
//       CTA owns ROWS = 32/64 weight rows and EVERY token of the N tile,
//       one wave per 16 rows streaming its own weights straight from HBM into
//       registers (decode once, no weight LDS), the activation block of the
//       current K128 staged in LDS and shared by the CTA's waves; grid
//       [ceil(M/ROWS), 1, 1], block ROWS/16*32, static LDS only. The `y` twin is
//       token-split: grid [ceil(M/64), ceil(N/16), 1], each CTA a 16-token slice
//       (small-M shapes, M ~ 2560, N 17..48).
//   Measured launcher rule (card C, receipts parity/kernel-runs/t1/time-6): N<=16 m64n16
//   (m32n16 for M<=2560), N 17..32 m64n32 (m32n32 for M 6144..12288), N 33..48 m64n48,
//   M<=2560 and N>=24 m64n16y, lm_head-sized M m64n48 at every N.
//
// Narrow-tile operand roles: A = activations (WMMA row = token), B = weights
// (WMMA column = weight row), so each lane's accumulators belong to ONE weight
// row (D[8*(lane>>4)+j][lane&15]) = the row whose header the lane itself loaded.
// Within a K128 block lane half kg (= lane>>4) holds codes 64kg..64kg+63 of the
// row as 48 contiguous bytes (3 x b128), WMMA step j covers codes 64kg+8j..+7:
// weight operand dwords {c0,c2,c4,c6}{c1,c3,c5,c7} = qs byte order, so no shuffle.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <type_traits>

#if defined(__gfx1200__) || defined(__gfx1201__)

// block_i8_128 comes from the host JIT prelude (block_i8_128_quant.hip).

#define SYM_GROUP_BYTES 196u

typedef int32_t __attribute__((ext_vector_type(2)))  int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8)))  int32x8_t;
typedef float   __attribute__((ext_vector_type(8)))  float8_t;
typedef unsigned __attribute__((ext_vector_type(2))) u32x2_t;
typedef unsigned __attribute__((ext_vector_type(4))) u32x4_t;

static __device__ __forceinline__ int32x8_t sym_zero8() {
    return (int32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
}

// Four 6-bit fields f0..f3 (values 0..63) -> the four signed bytes q - 32 of one dword: the bytes
// are assembled as q + 0x60 (<= 0x9F, no carry between bytes) and bit 7 flipped, which is exactly
// the two's-complement byte of q - 32 (q = 0 -> 0xE0 = -32, q = 63 -> 0x1F = 31).
static __device__ __forceinline__ unsigned sym_pack(unsigned f0, unsigned f1, unsigned f2, unsigned f3) {
    return (((f3 << 24) | (f2 << 16) | (f1 << 8) | f0) + 0x60606060u) ^ 0x80808080u;
}

// Two 24-bit fields (codes 0..3 in `ta`, codes 4..7 in `tb`, code i of a field at bits 6i) of an
// 8-code group -> {c0,c2,c4,c6} and {c1,c3,c5,c7} as signed bytes: the byte order of
// block_i8_128.qs, so the iu8 WMMA pairs every code with its own activation without a shuffle.
static __device__ __forceinline__ void sym_group8(unsigned ta, unsigned tb, unsigned& ev, unsigned& od) {
    ev = sym_pack(ta & 63u, (ta >> 12) & 63u, tb & 63u, (tb >> 12) & 63u);
    od = sym_pack((ta >> 6) & 63u, (ta >> 18) & 63u, (tb >> 6) & 63u, (tb >> 18) & 63u);
}

// 16 codes in w0 | w1<<32 | w2<<64 (code i = bits [6i, 6i+6)) -> the two iu8
// operands of two K16 steps: e0 = codes 0..7 as {c0,c2,c4,c6 | c1,c3,c5,c7},
// e1 = codes 8..15 likewise, signed bytes q - 32.
static __device__ __forceinline__ void sym_expand16(
    unsigned w0, unsigned w1, unsigned w2, u32x2_t& e0, u32x2_t& e1) {
    unsigned a, b;
    sym_group8(w0, __builtin_amdgcn_alignbit(w1, w0, 24), a, b);
    e0[0] = a;
    e0[1] = b;
    sym_group8(__builtin_amdgcn_alignbit(w2, w1, 16), w2 >> 8, a, b);
    e1[0] = a;
    e1[1] = b;
}

// f32 scale of half h (0/1) from the group's header dword {s0 | s1 << 16}.
static __device__ __forceinline__ float sym_scale(unsigned hdr, int h) {
    return __half2float(__ushort_as_half((unsigned short)((hdr >> (16 * h)) & 0xFFFFu)));
}

static __device__ __forceinline__ unsigned short sym_bf16_bits(float y) {
    const float v = 0.0f + y;
    const unsigned u = __float_as_uint(v);
    return isfinite(v) ? (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16)
                       : (unsigned short)(u >> 16);
}

// ===================================================================
// Narrow verify tiles: NW waves, RG 16-row groups per wave, TG 16-token groups.
// ===================================================================

template <int RG>
struct SymWRegs {
    unsigned hdr[RG];
    u32x4_t v[RG][6];   // [rg][h*3 + t]: 48 bytes of half h, lane half kg
};

// LDS-only CTA barrier: wait for this wave's LDS traffic (not for in-flight global loads, which
// __syncthreads would drain through its release fence), then rendezvous. Weight prefetch loads stay
// in flight across it.
static __device__ __forceinline__ void sym_bar() {
    asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
}

// Weights: the next group's registers are fetched while the current group computes (two register
// sets; deeper prefetch measured slower on card C, see the T1 receipts).
// YS: token-split grid: blockIdx.y selects a TG*16-token slice of the N tokens (each CTA streams the
// weights of its rows again; the slices share them through L2), for small-M shapes with few waves.
template <int NW, int RG, int TG, bool YS, bool BF16>
static __device__ __forceinline__ void sym_narrow_body(
    const unsigned char*  __restrict__ W,
    const block_i8_128*   __restrict__ Xq,
    void*                 __restrict__ Yv,
    int M, int K, int N
) {
    constexpr int NT = NW * 32;
    constexpr int TOK = TG * 16;
    // One staged K128 activation block: TOK tokens x 136 B, in a buffer rounded up to whole
    // per-thread b64 words so every thread stores every word (words past the descriptor read as 0).
    constexpr int XQ = TOK * 17;             // b64 words of one staged block
    constexpr int NPF = (XQ + NT - 1) / NT;  // staged b64 words per thread
    constexpr int XB = NPF * NT * 8;         // bytes of one LDS buffer
    __shared__ __attribute__((aligned(16))) char LDS[2 * XB];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int lo = lane & 15;
    const int kg = lane >> 4;
    const int rs = blockIdx.x * (NW * RG * 16);
    const int gpr = K / 256;
    const int nkb = K / 128;
    const unsigned row_stride = (unsigned)gpr * SYM_GROUP_BYTES;
    constexpr int TOKN = TG * 16;
    const int tok0 = YS ? (int)blockIdx.y * TOKN : 0;     // first token of this CTA
    const int nloc = YS ? min(TOKN, N - tok0) : N;        // tokens this CTA stages and stores
    const unsigned x_bytes = (unsigned)nloc * (unsigned)sizeof(block_i8_128);

    unsigned hoff[RG];
    unsigned woff[RG];
    int wrow[RG];
    #pragma unroll
    for (int rg = 0; rg < RG; rg++) {
        const int row = rs + (wave * RG + rg) * 16 + lo;
        wrow[rg] = row;
        const int rowc = (row < M) ? row : (M - 1);
        hoff[rg] = (unsigned)rowc * row_stride;
        woff[rg] = hoff[rg] + 48u * (unsigned)kg;
    }
    const auto w_rsrc = __builtin_amdgcn_make_buffer_rsrc((void*)W, 0, 0xffffffffu, 0x31004000u);

    auto load_group = [&](SymWRegs<RG>& r, int g) {
        const unsigned goff = (unsigned)g * SYM_GROUP_BYTES;
        #pragma unroll
        for (int rg = 0; rg < RG; rg++) {
            r.hdr[rg] = __builtin_amdgcn_raw_buffer_load_b32(w_rsrc, hoff[rg], goff, 0);
            #pragma unroll
            for (int hv = 0; hv < 2; hv++) {
                #pragma unroll
                for (int t = 0; t < 3; t++) {
                    r.v[rg][hv * 3 + t] = __builtin_amdgcn_raw_buffer_load_b128(
                        w_rsrc, woff[rg], goff + 4u + (unsigned)(hv * 96 + t * 16), 0);
                }
            }
        }
    };

    // Staged activation blocks in flight: slot 0 = block of the current group's second half,
    // slot 1 = first block of the next group (issued together, ahead of the weight prefetch).
    u32x2_t xr[2][NPF];
    // Activation block kb of N tokens is N*136 contiguous bytes; reads past the
    // descriptor's N*136 bytes return zero (tokens >= N stage as zero).
    auto load_x = [&](int kb, int slot) {
        const auto x_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)(Xq + (long long)kb * N + tok0), 0, x_bytes, 0x31004000u);
        #pragma unroll
        for (int i = 0; i < NPF; i++) {
            xr[slot][i] = __builtin_amdgcn_raw_buffer_load_b64(
                x_rsrc, (unsigned)(tid + i * NT) * 8u, 0u, 0);
        }
    };
    auto store_x = [&](int buf, int slot) {
        #pragma unroll
        for (int i = 0; i < NPF; i++) {
            const int idx = tid + i * NT;
            *(u32x2_t*)(LDS + buf * XB + idx * 8) = xr[slot][i];
        }
    };

    float8_t facc[RG][TG];
    #pragma unroll
    for (int rg = 0; rg < RG; rg++) {
        #pragma unroll
        for (int tg = 0; tg < TG; tg++) {
            facc[rg][tg] = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    SymWRegs<RG> wa, wb;
    load_group(wa, 0);
    load_x(0, 0);
    store_x(0, 0);
    sym_bar();

    // One K128 block (half HH of the group in `cur`), activations staged in LDS
    // buffer HH (kb parity == half index).
    auto compute_half = [&](const SymWRegs<RG>& cur, auto hh_c) {
        constexpr int HH = decltype(hh_c)::value;
        const char* xb = LDS + HH * XB;
        int32x8_t cacc[RG][TG];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int32x2_t a[TG];
            #pragma unroll
            for (int tg = 0; tg < TG; tg++) {
                a[tg] = *(const int32x2_t*)(xb + (tg * 16 + lo) * 136 + 8 + 64 * kg + 8 * j);
            }
            #pragma unroll
            for (int rg = 0; rg < RG; rg++) {
                const int i = j >> 1;
                const unsigned* dw = (const unsigned*)&cur.v[rg][HH * 3];
                // chunk i = dwords 3i..3i+2 of this lane's 12 code dwords
                u32x2_t e0, e1;
                sym_expand16(dw[3 * i], dw[3 * i + 1], dw[3 * i + 2], e0, e1);
                const u32x2_t bw = (j & 1) ? e1 : e0;
                const int32x2_t b = (int32x2_t){(int)bw[0], (int)bw[1]};
                #pragma unroll
                for (int tg = 0; tg < TG; tg++) {
                    cacc[rg][tg] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, a[tg], true, b, (j == 0) ? sym_zero8() : cacc[rg][tg], false);
                }
            }
        }
        // Fold this block with its own scale: acc = fmaf(RN(s*d), (float)C, acc).
        float dv[TG][8];
        #pragma unroll
        for (int tg = 0; tg < TG; tg++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                dv[tg][j] = *(const float*)(xb + (tg * 16 + 8 * kg + j) * 136);
            }
        }
        #pragma unroll
        for (int rg = 0; rg < RG; rg++) {
            const float sf = sym_scale(cur.hdr[rg], HH);
            #pragma unroll
            for (int tg = 0; tg < TG; tg++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float sd = __fmul_rn(sf, dv[tg][j]);
                    facc[rg][tg][j] = __builtin_fmaf(sd, (float)cacc[rg][tg][j], facc[rg][tg][j]);
                }
            }
        }
    };

    // Issue order (the vmcnt is in-order): both activation blocks of the group first, then the
    // next group's weights; publishing an activation block therefore never waits for the
    // prefetch, which is only waited for at its own group's first use.
    auto process_group = [&](int g, SymWRegs<RG>& cur, SymWRegs<RG>& nxt) {
        const bool more_g = (g + 1 < gpr);
        const int kb0 = 2 * g;
        load_x(kb0 + 1, 0);
        if (more_g) {
            load_x(kb0 + 2, 1);
            load_group(nxt, g + 1);
        }
        compute_half(cur, std::integral_constant<int, 0>());
        store_x(1, 0);
        sym_bar();
        compute_half(cur, std::integral_constant<int, 1>());
        if (more_g) {
            store_x(0, 1);
            sym_bar();
        }
    };

    for (int g = 0; g < gpr; g += 2) {
        process_group(g, wa, wb);
        if (g + 1 < gpr) {
            process_group(g + 1, wb, wa);
        }
    }

    // Epilogue: lane holds D[token 8*kg + j][weight row lo] of each (rg, tg).
    #pragma unroll
    for (int rg = 0; rg < RG; rg++) {
        const int row = wrow[rg];
        if (row >= M) continue;
        #pragma unroll
        for (int tg = 0; tg < TG; tg++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int tok = tg * 16 + 8 * kg + j;
                if (tok < nloc) {
                    const float v = facc[rg][tg][j];
                    if constexpr (BF16) {
                        ((unsigned short*)Yv)[(long long)(tok0 + tok) * M + row] = sym_bf16_bits(v);
                    } else {
                        ((float*)Yv)[(long long)(tok0 + tok) * M + row] = v;
                    }
                }
            }
        }
    }
}

#define SYM_NARROW_ENTRY(NAME, NW, RG, TG, YS, BF)                                       \
    extern "C" __global__ __launch_bounds__((NW) * 32) void NAME(                        \
        const unsigned char* __restrict__ W, const block_i8_128* __restrict__ Xq,        \
        void* __restrict__ Y, int M, int K, int N) {                                     \
        sym_narrow_body<NW, RG, TG, YS, BF>(W, Xq, Y, M, K, N);                          \
    }
#define SYM_ROW(R, NW, TGN, TG, SUF, YS)                                                 \
    SYM_NARROW_ENTRY(gemm_mq6sym_i8_set_m##R##n##TGN##SUF##_gfx1201, NW, 1, TG, YS, false)   \
    SYM_NARROW_ENTRY(gemm_mq6sym_i8_set_bf16_m##R##n##TGN##SUF##_gfx1201, NW, 1, TG, YS, true)

SYM_ROW(32, 2, 16, 1, , false)
SYM_ROW(32, 2, 32, 2, , false)
SYM_ROW(64, 4, 16, 1, , false)
SYM_ROW(64, 4, 32, 2, , false)
SYM_ROW(64, 4, 48, 3, , false)
// token-split twins of the 16-token tiles (grid [ceil(M/rows), ceil(N/16)])
SYM_ROW(64, 4, 16, 1, y, true)
// KILLED on card C (receipts parity/kernel-runs/t1/time-6): m32n48, m32n16y, m32n32y, m64n32y, the
// m{32,64,128}n*p{2,3} deeper-prefetch tiles, m128 tiles, an LDS-only-barrier prefill, a
// 192-VGPR-capped prefill: each lost to the kept tiles in every cell.

// ===================================================================
// Prefill 128 x 128 tile (A = weights, B = activations), a fork of the affine
// `gemm_mq6g256v2_mmq_i8.gfx12.hip` K1 lean-issue design: 8 waves, 32-row x
// 64-token wave tile, K64 slabs in a two-slot LDS ring with 2 barriers per K128,
// banded CTA raster, wide b128 epilogue. Differences: 196-B groups (payload at
// +4), the codes are decoded to SIGNED bytes at staging (q - 32) and enter the
// WMMA signed, the WMMA chains start from 0 (no zero-point seed, no z' fold):
//   acc = fmaf(RN(s_h * d), (float)C, acc)
// Operand pairing / fragment swizzle / LDS map: see the affine kernel header.
// ===================================================================

#define P_BM 128
#define P_BN 128
#define P_BK 64
#define P_WAVES 8
#define P_NTHREADS (P_WAVES * 32)
#ifndef P_RASTER_BAND
#define P_RASTER_BAND 8
#endif
#define P_A_BYTES (P_BM * P_BK)
#define P_W_BYTES (P_BN * P_BK)
#define P_SLOT_BYTES (P_A_BYTES + P_W_BYTES)
#define P_SLOT0_OFF 0
#define P_SLOT1_OFF P_SLOT_BYTES
#define P_DS0_OFF (2 * P_SLOT_BYTES)
#define P_DS_BYTES (P_BM * 8)
#define P_DS1_OFF (P_DS0_OFF + P_DS_BYTES)
#define P_SZ0_OFF (P_DS1_OFF + P_DS_BYTES)
#define P_SZ_BYTES (P_BN * 8)
#define P_SZ1_OFF (P_SZ0_OFF + P_SZ_BYTES)
#define P_LDS_BYTES (P_SZ1_OFF + P_SZ_BYTES)
#define MQ6SYM_P128_LDS_BYTES 36864
_Static_assert(P_LDS_BYTES == MQ6SYM_P128_LDS_BYTES, "p128 planes must match the host LDS contract");
_Static_assert(32 * 132 * 4 <= MQ6SYM_P128_LDS_BYTES, "wide store tile exceeds LDS");

// One K32 sub-block (two K16 steps): per step 2 W and 4 A b64 fragment reads,
// 8 WMMAs (signed A = weights, signed B = activations).
template <int SB, bool FIRST>
static __device__ __forceinline__ void p_bundle(
    const char* __restrict__ slot,
    const unsigned (&a_lane)[2],
    const unsigned (&w_lane)[2],
    int32x8_t cacc[4][2]
) {
    #pragma unroll
    for (int e = 0; e < 2; e++) {
        const char* a_base = slot + a_lane[e] + SB * 512 + e * 256;
        const char* w_base = slot + P_A_BYTES + w_lane[e] + SB * 512 + e * 256;
        const int32x2_t b0 = *(const int32x2_t*)(a_base + 0 * 1024);
        const int32x2_t b1 = *(const int32x2_t*)(a_base + 1 * 1024);
        const int32x2_t b2 = *(const int32x2_t*)(a_base + 2 * 1024);
        const int32x2_t b3 = *(const int32x2_t*)(a_base + 3 * 1024);
        const int32x2_t w0 = *(const int32x2_t*)(w_base);
        const int32x2_t w1 = *(const int32x2_t*)(w_base + 1024);
        #pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            const int32x2_t b = (nb == 0) ? b0 : (nb == 1) ? b1 : (nb == 2) ? b2 : b3;
            #pragma unroll
            for (int rg = 0; rg < 2; rg++) {
                const int32x2_t a = (rg == 0) ? w0 : w1;
                if (FIRST && e == 0) {
                    cacc[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, a, true, b, sym_zero8(), false);
                } else {
                    cacc[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, a, true, b, cacc[nb][rg], false);
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

// Raw loads of one K16 step (3 dwords) of a lane's weight row.
static __device__ __forceinline__ void p_load_w3(
    __amdgpu_buffer_rsrc_t rsrc, unsigned voff, unsigned soff, unsigned (&d)[3]) {
    d[0] = __builtin_amdgcn_raw_buffer_load_b32(rsrc, voff, soff, 0);
    d[1] = __builtin_amdgcn_raw_buffer_load_b32(rsrc, voff, soff + 4u, 0);
    d[2] = __builtin_amdgcn_raw_buffer_load_b32(rsrc, voff, soff + 8u, 0);
}

// Decode a raw K16 step (signed) and publish its two operand fragments into W plane `wp`.
static __device__ __forceinline__ void p_publish_w(
    char* __restrict__ wp, const unsigned (&lds)[2], const unsigned (&d)[3]) {
    u32x2_t e0, e1;
    sym_expand16(d[0], d[1], d[2], e0, e1);
    *(u32x2_t*)(wp + lds[0]) = e0;
    *(u32x2_t*)(wp + lds[1]) = e1;
}

// Header dword {s0 | s1 << 16} -> the SZ plane word: lane word 0 = f32 s_h, word 1 unused (0).
static __device__ __forceinline__ float p_sz_value(unsigned hw, int h, int w) {
    return (w == 0) ? sym_scale(hw, h) : 0.0f;
}

template <bool BF16>
static __device__ __forceinline__ void gemm_mq6sym_p128_body(
    const char*           __restrict__ A,   // [M x groups_per_row x 196]
    const block_i8_128*   __restrict__ Xq,  // [K/128 x N]
    void*                 __restrict__ Yv,  // [N x M] F32 or BF16 bits
    int M, int K, int N
) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    int rs;
    int bs;
    {
        // Banded raster: bijective for any grid.
        const int n_row_tiles = gridDim.x;
        const int lin = blockIdx.y * n_row_tiles + blockIdx.x;
        const int per_band = P_RASTER_BAND * gridDim.y;
        const int band = lin / per_band;
        const int in_band = lin - band * per_band;
        const int band_row0 = band * P_RASTER_BAND;
        const int band_rows = min(n_row_tiles - band_row0, P_RASTER_BAND);
        rs = (band_row0 + in_band % band_rows) * P_BN;
        bs = (in_band / band_rows) * P_BM;
    }
    if (rs >= M || bs >= N) return;

    const int m_lane = lane & 15;
    const int k_grp  = lane >> 4;
    const int wt_pair = wave >> 1;
    const int tok_half = wave & 1;
    const int wave_row0 = wt_pair * 32;
    const int wave_tok0 = tok_half * 64;

    const int groups_per_row = K / 256;
    extern __shared__ char LDS[];

    const unsigned lane_lo = (unsigned)lane & 15u;
    const unsigned lane_half = (unsigned)lane & 16u;
    const unsigned lane_even = ((lane_lo ^ (lane_half >> 2)) | lane_half) * 8u;
    const unsigned lane_odd = ((lane_lo ^ (8u ^ (lane_half >> 2))) | lane_half) * 8u;
    const unsigned a_lane[2] = {
        (unsigned)tok_half * 4096u + lane_even,
        (unsigned)tok_half * 4096u + lane_odd,
    };
    const unsigned w_lane[2] = {
        (unsigned)wt_pair * 2048u + lane_even,
        (unsigned)wt_pair * 2048u + lane_odd,
    };

    float8_t acc[4][2];
    unsigned st_avoff[2];
    unsigned st_wvoff[2];
    unsigned st_alds[2];
    unsigned st_wlds[2][2];
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        const unsigned T = (unsigned)(r * 64 + (tid >> 2));
        const unsigned c = (unsigned)(tid & 3);
        st_avoff[r] = (unsigned)(bs + (int)T) * (unsigned)sizeof(block_i8_128) + c * 8u;
        {
            const unsigned e = c & 1u;
            const unsigned L = (T & 15u) + (c >> 1) * 16u;
            const unsigned sw = (e << 3) ^ ((L >> 4) << 2);
            st_alds[r] = (T >> 4) * 1024u + e * 256u + (L ^ sw) * 8u;
        }
        {
            const int grow_r = rs + (int)T;
            const int src_c = (grow_r < M) ? grow_r : (M - 1);
            st_wvoff[r] = (unsigned)src_c * (unsigned)groups_per_row * SYM_GROUP_BYTES + c * 12u;
            const unsigned sb = c >> 1;
            const unsigned kg = c & 1u;
            const unsigned L = (T & 15u) + kg * 16u;
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                const unsigned sw = ((unsigned)e << 3) ^ (kg << 2);
                st_wlds[r][e] = (T >> 4) * 1024u + sb * 512u + (unsigned)e * 256u + (L ^ sw) * 8u;
            }
        }
    }
    for (int nb = 0; nb < 4; nb++) {
        #pragma unroll
        for (int rg = 0; rg < 2; rg++) {
            acc[nb][rg] = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
    int32x8_t cacc[4][2];
    // The A descriptor spans exactly one K128 block of N tokens: tokens >= N
    // read zero. W rows are clamped (0xffffffff range).
    const unsigned a_bytes = (unsigned)N * (unsigned)sizeof(block_i8_128);
    const auto w_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)A, 0, 0xffffffffu, 0x31004000u);
    u32x2_t A_pf[2];
    unsigned W_pf[2][3];
    unsigned ds_nx = 0u;
    unsigned sz_nx = 0u;

    // --- Prologue: DS[0] + SZ[0] + slab 0 of block 0 into slot 0. ---
    for (int i = tid; i < P_BM * 2; i += P_NTHREADS) {
        const int tok_idx = i >> 1;
        const int w = i & 1;
        const int tok = bs + tok_idx;
        const unsigned dw = (tok < N) ? ((const unsigned*)(Xq + tok))[w] : 0u;
        ((unsigned*)(LDS + P_DS0_OFF + tok_idx * 8))[w] = dw;
    }
    for (int i = tid; i < P_BN * 2; i += P_NTHREADS) {
        const int row_idx = i >> 1;
        const int w = i & 1;
        const int grow = rs + row_idx;
        const int src_row = (grow < M) ? grow : (M - 1);
        const unsigned hw = *(const unsigned*)(A + (long long)src_row * groups_per_row * SYM_GROUP_BYTES);
        ((float*)(LDS + P_SZ0_OFF + row_idx * 8))[w] = (grow < M) ? p_sz_value(hw, 0, w) : 0.0f;
    }
    {
        const auto a0_rsrc = __builtin_amdgcn_make_buffer_rsrc((void*)Xq, 0, a_bytes, 0x31004000u);
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            #pragma unroll
            for (int sb = 0; sb < 2; sb++) {
                const u32x2_t av = __builtin_amdgcn_raw_buffer_load_b64(
                    a0_rsrc, st_avoff[r], 8u + (unsigned)sb * 32u, 0);
                *(u32x2_t*)(LDS + P_SLOT0_OFF + st_alds[r] + sb * 512) = av;
            }
            unsigned wd[3];
            p_load_w3(w_rsrc, st_wvoff[r], 4u, wd);
            p_publish_w(LDS + P_SLOT0_OFF + P_A_BYTES, st_wlds[r], wd);
        }
    }
    __syncthreads();

    for (int g = 0; g < groups_per_row; g++) {
        // Rolled: only cacc/acc carry across the two halves.
        #pragma clang loop unroll_count(1)
        for (int h = 0; h < 2; h++) {
            const int kb = 2 * g + h;
            const bool last = (h == 1) && (g + 1 == groups_per_row);
            const char* __restrict__ ds_lds = LDS + ((h == 0) ? P_DS0_OFF : P_DS1_OFF);
            const char* __restrict__ sz_lds = LDS + ((h == 0) ? P_SZ0_OFF : P_SZ1_OFF);
            const auto a_rsrc = __builtin_amdgcn_make_buffer_rsrc(
                (void*)(Xq + (long long)kb * N), 0, a_bytes, 0x31004000u);
            // --- Slot 0 compute; slab 1 of this block -> slot 1. ---
            {
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    A_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(a_rsrc, st_avoff[r], 72u, 0);
                    p_load_w3(w_rsrc, st_wvoff[r],
                        (unsigned)g * SYM_GROUP_BYTES + 4u + (unsigned)h * 96u + 48u, W_pf[r]);
                }
                p_bundle<0, true>(LDS + P_SLOT0_OFF, a_lane, w_lane, cacc);
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    asm volatile("" : "+v"(A_pf[r][0]), "+v"(A_pf[r][1]));
                    *(u32x2_t*)(LDS + P_SLOT1_OFF + st_alds[r]) = A_pf[r];
                }
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    A_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(a_rsrc, st_avoff[r], 104u, 0);
                }
                p_bundle<1, false>(LDS + P_SLOT0_OFF, a_lane, w_lane, cacc);
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    asm volatile("" : "+v"(A_pf[r][0]), "+v"(A_pf[r][1]));
                    asm volatile("" : "+v"(W_pf[r][0]), "+v"(W_pf[r][1]), "+v"(W_pf[r][2]));
                    *(u32x2_t*)(LDS + P_SLOT1_OFF + st_alds[r] + 512) = A_pf[r];
                    p_publish_w(LDS + P_SLOT1_OFF + P_A_BYTES, st_wlds[r], W_pf[r]);
                }
                // Slot-0 retirement plus slot-1 readiness.
                __syncthreads();
                if (!last) {
                    // Next-block fetch epoch: raw loads into private registers,
                    // first use at the slot-0 publishes below.
                    const int kb_nx = kb + 1;
                    const auto a_next_rsrc = __builtin_amdgcn_make_buffer_rsrc(
                        (void*)(Xq + (long long)kb_nx * N), 0, a_bytes, 0x31004000u);
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        A_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(a_next_rsrc, st_avoff[r], 8u, 0);
                    }
                    const int g_nx = g + h;
                    const int h_nx = h ^ 1;
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        p_load_w3(w_rsrc, st_wvoff[r],
                            (unsigned)g_nx * SYM_GROUP_BYTES + 4u + (unsigned)h_nx * 96u, W_pf[r]);
                    }
                    const int tok_ds = bs + (tid >> 1);
                    const int tok_dsc = (tok_ds < N) ? tok_ds : (N - 1);
                    ds_nx = ((const unsigned*)(Xq + (long long)kb_nx * N + tok_dsc))[tid & 1];
                    const int row_sz = rs + (tid >> 1);
                    const int src_sz = (row_sz < M) ? row_sz : (M - 1);
                    sz_nx = *(const unsigned*)(A + (long long)src_sz * groups_per_row * SYM_GROUP_BYTES
                        + (unsigned)g_nx * SYM_GROUP_BYTES);
                }
            }
            // --- Slot 1 compute, fold, next slab 0 -> slot 0. ---
            {
                p_bundle<0, false>(LDS + P_SLOT1_OFF, a_lane, w_lane, cacc);
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    asm volatile("" : "+v"(A_pf[r][0]), "+v"(A_pf[r][1]));
                    *(u32x2_t*)(LDS + P_SLOT0_OFF + st_alds[r]) = A_pf[r];
                }
                {
                    const auto a_next_rsrc = __builtin_amdgcn_make_buffer_rsrc(
                        (void*)(Xq + (long long)(kb + 1) * N), 0, last ? 0u : a_bytes, 0x31004000u);
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        A_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(a_next_rsrc, st_avoff[r], 40u, 0);
                    }
                }
                p_bundle<1, false>(LDS + P_SLOT1_OFF, a_lane, w_lane, cacc);
                // Fold (per 128-K block, reads only cacc and the current DS[h]/SZ[h]
                // planes): acc = fmaf(RN(s * d), (float)C, acc).
                #pragma unroll
                for (int rg = 0; rg < 2; rg++) {
                    float sc_row[8];
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const unsigned jbase = (unsigned)(wave_row0 + rg * 16 + 8 * k_grp + j) * 8u;
                        sc_row[j] = *(const float*)(sz_lds + jbase);
                    }
                    #pragma unroll
                    for (int nb = 0; nb < 4; nb++) {
                        const unsigned tbase = (unsigned)(wave_tok0 + nb * 16 + m_lane) * 8u;
                        const float d_x = *(const float*)(ds_lds + tbase);
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            acc[nb][rg][j] = __builtin_fmaf(
                                __fmul_rn(sc_row[j], d_x), (float)cacc[nb][rg][j], acc[nb][rg][j]);
                        }
                    }
                }
                if (!last) {
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        *(u32x2_t*)(LDS + P_SLOT0_OFF + st_alds[r] + 512) = A_pf[r];
                        p_publish_w(LDS + P_SLOT0_OFF + P_A_BYTES, st_wlds[r], W_pf[r]);
                    }
                    const int h_np = h ^ 1;
                    char* __restrict__ ds_n = LDS + ((h_np == 0) ? P_DS0_OFF : P_DS1_OFF);
                    {
                        const int tok_dsp = bs + (tid >> 1);
                        ((unsigned*)(ds_n + (tid >> 1) * 8))[tid & 1] = (tok_dsp < N) ? ds_nx : 0u;
                    }
                    char* __restrict__ sz_n = LDS + ((h_np == 0) ? P_SZ0_OFF : P_SZ1_OFF);
                    {
                        const int row_szp = rs + (tid >> 1);
                        const int w_szp = tid & 1;
                        ((float*)(sz_n + (tid >> 1) * 8))[w_szp] =
                            (row_szp < M) ? p_sz_value(sz_nx, h_np, w_szp) : 0.0f;
                    }
                }
                // Slot-1 retirement plus next slot-0 readiness; on the last
                // block this guards store-tile reuse of LDS.
                __syncthreads();
            }
        } // for h
    } // for g

    // Epilogue: wide tile stores for full tiles, LDS transpose + checked stores
    // otherwise. `slots` reuses the dead compute planes.
    float* slots = (float*)LDS;
    const int store_i = tid & 15;
    const int store_x = (tid >> 4) & 15;
    const bool full_tile = rs + P_BM <= M && bs + P_BM <= N;
    const long long tile_elem0 = (long long)bs * M + rs;
    if (full_tile && (M & 3) == 0
        && ((reinterpret_cast<uintptr_t>(Yv) & (BF16 ? 7u : 15u)) == 0)) {
        // Wide path: each lane stores 4 consecutive rows (b128 F32 / b64 BF16).
        const long long wide_lane_el = (long long)wave * M + lane * 4;
        const unsigned dump_tok = (unsigned)(tok_half * 16 + m_lane);
        #pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            #pragma unroll
            for (int rg = 0; rg < 2; rg++) {
                #pragma unroll
                for (int c = 0; c < 2; c++) {
                    float4 v4;
                    v4.x = acc[nb][rg][4 * c + 0];
                    v4.y = acc[nb][rg][4 * c + 1];
                    v4.z = acc[nb][rg][4 * c + 2];
                    v4.w = acc[nb][rg][4 * c + 3];
                    *(float4*)(slots + dump_tok * 132u
                        + (unsigned)(wave_row0 + rg * 16 + 8 * k_grp + 4 * c)) = v4;
                }
            }
            __syncthreads();
            #pragma unroll
            for (int it = 0; it < 4; it++) {
                const float4 v4 = *(const float4*)(slots + (unsigned)(it * 8 + wave) * 132u
                    + (unsigned)lane * 4u);
                const long long el = tile_elem0 + wide_lane_el
                    + (long long)((it >> 1) * 64 + nb * 16 + (it & 1) * 8) * M;
                if constexpr (BF16) {
                    uint2 pk;
                    pk.x = (unsigned)sym_bf16_bits(v4.x) | ((unsigned)sym_bf16_bits(v4.y) << 16);
                    pk.y = (unsigned)sym_bf16_bits(v4.z) | ((unsigned)sym_bf16_bits(v4.w) << 16);
                    *(uint2*)((unsigned short*)Yv + el) = pk;
                } else {
                    *(float4*)((float*)Yv + el) = v4;
                }
            }
            __syncthreads();
        }
        return;
    }
    if constexpr (!BF16) {
        if (full_tile) {
            const unsigned store_lane_off =
                ((unsigned)store_x * (unsigned)M + (unsigned)store_i) * sizeof(float);
            float* const y_tile = (float*)Yv + tile_elem0;
            const auto y_tile_rsrc = __builtin_amdgcn_make_buffer_rsrc(
                y_tile, 0, 0xffffffffu, 0x31004000u);
            #pragma unroll
            for (int rg = 0; rg < 2; rg++) {
                #pragma unroll
                for (int nb = 0; nb < 4; nb++) {
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        slots[(unsigned)(wave * 320 + (8 * k_grp + j) * 20 + m_lane)] = acc[nb][rg][j];
                    }
                    __syncthreads();
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        const int p = k >> 1;
                        const int hh = k & 1;
                        const int owner = (p << 1) | hh;
                        const float v = slots[(unsigned)(owner * 320 + store_i * 20 + store_x)];
                        const unsigned phase_off =
                            ((unsigned)(hh * 64 + nb * 16) * (unsigned)M
                                + (unsigned)(p * 32 + rg * 16)) * sizeof(float);
                        __builtin_amdgcn_raw_buffer_store_b32(
                            __builtin_bit_cast(unsigned, v),
                            y_tile_rsrc, store_lane_off, phase_off, 0);
                    }
                    __syncthreads();
                }
            }
            return;
        }
    }
    // Guarded path (M / N tails, unaligned Y, BF16 without the wide path).
    #pragma unroll
    for (int rg = 0; rg < 2; rg++) {
        #pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                slots[(unsigned)(wave * 320 + (8 * k_grp + j) * 20 + m_lane)] = acc[nb][rg][j];
            }
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                const int p = k >> 1;
                const int hh = k & 1;
                const int row_rel = p * 32 + rg * 16 + store_i;
                const int col_rel = hh * 64 + nb * 16 + store_x;
                const int owner = (p << 1) | hh;
                const float v = slots[(unsigned)(owner * 320 + store_i * 20 + store_x)];
                const int row = rs + row_rel;
                const int col = bs + col_rel;
                if (row < M && col < N) {
                    if constexpr (BF16) {
                        ((unsigned short*)Yv)[(long long)col * M + row] = sym_bf16_bits(v);
                    } else {
                        ((float*)Yv)[(long long)col * M + row] = v;
                    }
                }
            }
            __syncthreads();
        }
    }
}

extern "C" __global__ __launch_bounds__(P_NTHREADS, 2)
void gemm_mq6sym_i8_set_gfx1201(
    const unsigned char*  __restrict__ W,
    const block_i8_128*   __restrict__ Xq,
    void*                 __restrict__ Y,
    int M, int K, int N
) {
    gemm_mq6sym_p128_body<false>((const char*)W, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(P_NTHREADS, 2)
void gemm_mq6sym_i8_set_bf16_gfx1201(
    const unsigned char*  __restrict__ W,
    const block_i8_128*   __restrict__ Xq,
    void*                 __restrict__ Y,
    int M, int K, int N
) {
    gemm_mq6sym_p128_body<true>((const char*)W, Xq, Y, M, K, N);
}
#endif  // gfx1200 / gfx1201
