// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// MQ4G256V2 A8 MMQ for gfx12 (RDNA4 — gfx1200/gfx1201): int8 per-128-K
// activations x MQ4V2 weight nibbles on `v_wmma_i32_16x16x16_iu8`.
//
// A fork of the K1 lean-issue iu4 kernel
// (`gemm_mq4g256v2_residual_mmq_iu4_v3.gfx12.hip`): same ABI (7-arg SET/ADD,
// 7-arg gate/up SiLU), same 128x128 tile, 8 waves, 32-row x 64-token wave
// tile, K64 slabs in a two-slot LDS ring with 2 barriers per K128, same
// banded CTA raster and wide b128 epilogue, same SRD addressing, same fetch
// epoch. What changes (design: /home/kaden/qcal/perf/fp8-4k5/a8-route.md
// §2.1):
//   - Xq: `block_i8_128` {f32 d, i32 s, int8 qs[128]} (136 B, layout
//     [K/128 x N]) from `quantize_int8_mmq_ds128` / the fused producers,
//     qs in the nibble-pair byte order documented in block_i8_128_quant.hip.
//   - Weights: the MQ4V2 file layout, staged to LDS as K1's XOR-swizzled
//     K32 nibble fragments; each K16 step expands one packed dword per row
//     group AFTER the LDS read: lo = pk & 0x0F0F0F0F, hi = (pk >> 4) &
//     0x0F0F0F0F (3 VALU per 8 weights). The codes go to the WMMA raw and
//     UNSIGNED (signed_a = false) against signed int8 activations, as in
//     `gemm_hfq4g256_residual_mmq.gfx12.hip`; lo/hi is exactly the qs byte
//     order, so no activation shuffle exists in the K loop.
//   - Symmetric MQ4V2 only (w = sc * (q - 8)): the -8 * sum(x8) zero-point
//     term is folded into the magic seed once per token per K128 (see
//     I8_ZP_SEED). The per-element fold is K1's:
//       fc  = as_float(cacc) - 12582912.0f      (exact integer sum(q-8)*x8)
//       t1  = RN(sc * d)
//       acc = fma(t1, fc, acc)
//     |sum q*x8| <= 15*127*128 = 243,840 and |8*s| <= 130,048, so every
//     intermediate stays inside 12,582,912 +- 2^22 where the float
//     reinterpretation of 0x4B400000 + n is exactly 12582912 + n. The GEMM is
//     integer-exact up to the pinned fold: a CPU reference reproduces it bit
//     for bit (crates/hipfire-runtime/examples/tmp_i8_gfx12_oracle.rs).
//
// gfx12 iu8 K16 conventions (gemm_hfq4g256_residual_mmq.gfx12.hip): int32x2
// A/B per lane (8 int8), lanes 0-15 carry one K8 half of the step and lanes
// 16-31 the other, contiguous-row C output acc[j] = C[8*k_grp + j][m_lane].
// The K16 step (sb, e) of a K64 slab uses, for lane L, the 8 K elements at
// slab offset sb*32 + (L>>4)*16 + e*8 (dword e of K1's 8-byte W lane): any
// K permutation shared by both operands leaves the integer dot unchanged.
//
// LDS (28672 B; the store tiles overlap the dead compute planes):
//   slot s (s = 0, 1) @ s*12288:
//     A_s = 32 K16 fragments x 256 B (fragment order below)   (8192 B)
//     W_s = 16 K32 nibble fragments x 256 B (K1's order)      (4096 B)
//   DS0/DS1 = [128][8] token (d f32, s i32) per kb parity   @ 24576 (2 x 1024)
//   SZ0/SZ1 = [128][8] row (sc f32, zp f32) per kb parity   @ 26624 (2 x 1024)
//   Four 28672 B workgroups fit one WGP; occupancy is set by <= 192 VGPR.
//
// A fragment order: fragment f = g8*4 + sb*2 + e (g8 = 16-token group, sb =
// K32 sub-block, e = K16 half) holds the 8-byte B-operand lanes of one K16
// step; lane L's bytes sit at f*256 + (L ^ sw(e, L>>4))*8 with K1's swizzle
// sw = (e << 3) ^ ((L >> 4) << 2). Staging writes (8 tokens x 4 chunks per
// wave) then hit every bank exactly twice (the b64 floor) and the consumer's
// per-step b64 read is a contiguous 256 B fragment. W fragments are K1's
// verbatim (fragment parity = sb).
//
// I8_APF_K32 (default 1, a8-route.md §2.2 VGPR mitigation): the A slab
// prefetch is split into its two K32 halves (4 private VGPRs instead of 8);
// each half is published to the free slot after one K32 compute bundle.
// I8_ZP_SEED (default 1): the WMMA C seed of each (token, K128) chain is
// 0x4B400000 - 8*s, broadcast from one VGPR per token; 0 keeps K1's
// constant seed and subtracts as_float(0x4B400000 + 8*s) in the fold
// instead (the same exact fc, hence bit-identical output).
//
// Grid: [ceil(M/128), ceil(N/128)] (gate/up: [2M/128, ...]). Block
// [256,1,1]. LDS 28672 bytes. Compile gate: gfx1200 / gfx1201 only.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#if defined(__gfx1200__) || defined(__gfx1201__)

// block_i8_128 comes from the host JIT prelude (block_i8_128_quant.hip).

#define I8_BM 128
#define I8_BN 128
#define I8_BK 64
#define I8_WAVES 8
#define I8_NTHREADS (I8_WAVES * 32)
#ifndef I8_RASTER_BAND
#define I8_RASTER_BAND 8
#endif
#ifndef I8_APF_K32
#define I8_APF_K32 1
#endif
#ifndef I8_ZP_SEED
#define I8_ZP_SEED 1
#endif

#define I8_FRAG_BYTES 256
#define I8_A_BYTES (I8_BM * I8_BK)
#define I8_W_BYTES (I8_BN * I8_BK / 2)
#define I8_SLOT_BYTES (I8_A_BYTES + I8_W_BYTES)
#define I8_SLOT0_OFF 0
#define I8_SLOT1_OFF I8_SLOT_BYTES
#define I8_DS0_OFF (2 * I8_SLOT_BYTES)
#define I8_DS_BYTES (I8_BM * 8)
#define I8_DS1_OFF (I8_DS0_OFF + I8_DS_BYTES)
#define I8_SZ0_OFF (I8_DS1_OFF + I8_DS_BYTES)
#define I8_SZ_BYTES (I8_BN * 8)
#define I8_SZ1_OFF (I8_SZ0_OFF + I8_SZ_BYTES)
#define I8_LDS_BYTES (I8_SZ1_OFF + I8_SZ_BYTES)
_Static_assert(I8_LDS_BYTES == 28672, "i8 planes must match the host LDS contract");
_Static_assert(32 * 132 * 4 <= I8_LDS_BYTES, "wide store tile exceeds LDS");
#define I8_MAGIC 0x4B400000

typedef int32_t __attribute__((ext_vector_type(2)))  int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8)))  int32x8_t;
typedef float   __attribute__((ext_vector_type(8)))  float8_t;
typedef unsigned __attribute__((ext_vector_type(2))) u32x2_t;

// One packed MQ4V2 dword (8 nibbles, bit 4i = K i) -> the iu8 operand
// {K0,K2,K4,K6 | K1,K3,K5,K7} as raw unsigned codes.
static __device__ __forceinline__ int32x2_t i8_unpack_w(unsigned pk) {
    int32x2_t w;
    w[0] = (int)(pk & 0x0F0F0F0Fu);
    w[1] = (int)((pk >> 4) & 0x0F0F0F0Fu);
    return w;
}

static __device__ __forceinline__ int32x8_t i8_splat(int v) {
    return (int32x8_t){v, v, v, v, v, v, v, v};
}

// Paired fold (gfx1201 VOPD form): per component
//   acc = fma(RN(sc * d), as_float(c) - magic, acc)
// with the stages interleaved pairwise for V_DUAL_MUL/SUB/FMAC packets.
static __device__ __forceinline__ float2 i8_fold_pair(
    float sum0, float sum1, float sc0, float sc1,
    float d, int c0, int c1, float magic) {
    const float t1_0 = __fmul_rn(sc0, d);
    const float t1_1 = __fmul_rn(sc1, d);
    const float fc0 = __builtin_bit_cast(float, c0) - magic;
    const float fc1 = __builtin_bit_cast(float, c1) - magic;
    float2 out;
    out.x = __builtin_fmaf(t1_0, fc0, sum0);
    out.y = __builtin_fmaf(t1_1, fc1, sum1);
    return out;
}

// One K32 sub-block (two K16 steps): per step 2 W dwords expanded after one
// b64 W read per row group, 4 A b64 fragment reads, 8 WMMAs. SB is a
// template constant (LDS immediates); FIRST seeds each chain's first step.
// a_lane[e] / w_lane[sb] are the swizzled lane byte bases (plus the wave's
// token/row group), so every read is VGPR + immediate.
template <int SB, bool FIRST>
static __device__ __forceinline__ void i8_bundle(
    const char* __restrict__ slot,
    const unsigned (&a_lane)[2],
    const unsigned (&w_lane)[2],
    int32x8_t cacc[4][2],
    const int (&seed)[4]
) {
    const u32x2_t wr0 = *(const u32x2_t*)(slot + I8_A_BYTES + w_lane[SB] + SB * 256);
    const u32x2_t wr1 = *(const u32x2_t*)(slot + I8_A_BYTES + w_lane[SB] + SB * 256 + 512);
    #pragma unroll
    for (int e = 0; e < 2; e++) {
        const char* a_base = slot + a_lane[e] + SB * 512 + e * 256;
        const int32x2_t b0 = *(const int32x2_t*)(a_base + 0 * 1024);
        const int32x2_t b1 = *(const int32x2_t*)(a_base + 1 * 1024);
        const int32x2_t b2 = *(const int32x2_t*)(a_base + 2 * 1024);
        const int32x2_t b3 = *(const int32x2_t*)(a_base + 3 * 1024);
        const int32x2_t w0 = i8_unpack_w(wr0[e]);
        const int32x2_t w1 = i8_unpack_w(wr1[e]);
        #pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            const int32x2_t b = (nb == 0) ? b0 : (nb == 1) ? b1 : (nb == 2) ? b2 : b3;
            #pragma unroll
            for (int rg = 0; rg < 2; rg++) {
                const int32x2_t a = (rg == 0) ? w0 : w1;
                if (FIRST && e == 0) {
                    cacc[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        false, a, true, b, i8_splat(seed[nb]), false);
                } else {
                    cacc[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        false, a, true, b, cacc[nb][rg], false);
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

// EPI: I8_EPI_SET / I8_EPI_ADD: Y = / += result. I8_EPI_SILU (gate/up): A =
// W_gate, U = W_up, M rows each; virtual 16-row group g8 of a tile holds gate
// (g8 even) or up (g8 odd) rows blockIdx-band*64 + 16*(g8>>1) + 0..15, so a
// wave's rg 0 / rg 1 accumulators are gate / up of the same h rows, and
// H = I8_SILU_MUL(g, u) (the K1 / fused_silu_mul_mq_rotate_awq.hip form).
enum { I8_EPI_SET = 0, I8_EPI_ADD = 1, I8_EPI_SILU = 2 };
#define I8_SILU_MUL(g, u) ((g) / (1.0f + expf(-(g))) * (u))

template <int EPI>
static __device__ __forceinline__ void gemm_mq4g256v2_residual_mmq_i8_gfx12_body(
    const char*           __restrict__ A,   // [M x groups_per_row x 136]
    const char*           __restrict__ U,   // SILU: W_up, same layout as A
    const block_i8_128*   __restrict__ Xq,  // [K/128 x N]
    float*                __restrict__ Y,   // [N x M] column-major
    int M, int K, int N
) {
    constexpr bool ADD = EPI == I8_EPI_ADD;
    constexpr bool SILU = EPI == I8_EPI_SILU;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    int rs;
    int bs;
    {
        // Banded raster (K1 IU4_G12_RASTER): bijective for any grid.
        const int n_row_tiles = gridDim.x;
        const int lin = blockIdx.y * n_row_tiles + blockIdx.x;
        const int per_band = I8_RASTER_BAND * gridDim.y;
        const int band = lin / per_band;
        const int in_band = lin - band * per_band;
        const int band_row0 = band * I8_RASTER_BAND;
        const int band_rows = min(n_row_tiles - band_row0, I8_RASTER_BAND);
        rs = (band_row0 + in_band % band_rows) * I8_BN;
        bs = (in_band / band_rows) * I8_BM;
    }
    if (rs >= (SILU ? 2 * M : M) || bs >= N) return;

    // SILU: first h row of the tile. Staging W row R (= r*64 + (tid>>2)) is
    // gate/up by bit 6 of tid and SZ row tid>>1 by bit 5: wave-uniform.
    const int hs = rs >> 1;
    const char* __restrict__ Aw = A;
    const char* __restrict__ As = A;
    int sz_row = 0;
    if constexpr (SILU) {
        static_assert(I8_BN * 2 == I8_NTHREADS, "one SZ word per thread");
        Aw = __builtin_amdgcn_readfirstlane((tid >> 6) & 1) ? U : A;
        As = __builtin_amdgcn_readfirstlane((tid >> 5) & 1) ? U : A;
        sz_row = hs + (tid >> 6) * 16 + ((tid >> 1) & 15);
    }

    const int m_lane = lane & 15;
    const int k_grp  = lane >> 4;
    const int wt_pair = wave >> 1;
    const int tok_half = wave & 1;
    const int wave_row0 = wt_pair * 32;
    const int wave_tok0 = tok_half * 64;

    const int groups_per_row = K / 256;
    extern __shared__ char LDS[];

    // Swizzled lane byte offsets (K1): parity-0 / parity-1 fragments.
    const unsigned lane_lo = (unsigned)lane & 15u;
    const unsigned lane_half = (unsigned)lane & 16u;
    const unsigned lane_even = ((lane_lo ^ (lane_half >> 2)) | lane_half) * 8u;
    const unsigned lane_odd = ((lane_lo ^ (8u ^ (lane_half >> 2))) | lane_half) * 8u;
    // A fragment (g8 = tok_half*4 + nb, sb, e) at g8*1024 + sb*512 + e*256;
    // W fragment (g8w = wt_pair*2 + rg, sb) at g8w*512 + sb*256 (+ A plane).
    const unsigned a_lane[2] = {
        (unsigned)tok_half * 4096u + lane_even,
        (unsigned)tok_half * 4096u + lane_odd,
    };
    const unsigned w_lane[2] = {
        (unsigned)wt_pair * 1024u + lane_even,
        (unsigned)wt_pair * 1024u + lane_odd,
    };

    float8_t acc[4][2];
    // Staging maps (all loop-invariant):
    //   A (per K32 half sb): round r stages token T = r*64 + (tid>>2), 8-byte
    //     chunk c = tid&3 (lane half c>>1, K16 half e = c&1) of that half:
    //     global T*136 + c*8 (+ 8 + slab*64 + sb*32 in the scalar offset),
    //     LDS frag (T>>4)*4 + sb*2 + e, lane (T&15) + 16*(c>>1), swizzled.
    //   W (K1 verbatim): row R = r*64 + (tid>>2), quad q = tid&3 (16 nibbles)
    //     -> frag (R>>4)*2 + (q>>1), lane (R&15) + 16*(q&1), swizzled.
    unsigned st_avoff[2];
    unsigned st_wvoff[2];
    unsigned st_alds[2];
    unsigned st_wlds[2];
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
        if constexpr (SILU) {
            const int hrow = hs + (r * 2 + (tid >> 7)) * 16 + ((tid >> 2) & 15);
            const int src_c = (hrow < M) ? hrow : (M - 1);
            st_wvoff[r] = (unsigned)src_c * (unsigned)groups_per_row * 136u + c * 8u;
        } else {
            const int grow_r = rs + (int)T;
            const int src_c = (grow_r < M) ? grow_r : (M - 1);
            st_wvoff[r] = (unsigned)src_c * (unsigned)groups_per_row * 136u + c * 8u;
        }
        {
            const unsigned frag = (T >> 4) * 2u + (c >> 1);
            const unsigned L = (T & 15u) + (c & 1u) * 16u;
            const unsigned sw = ((frag & 1u) << 3) ^ ((L >> 4) << 2);
            st_wlds[r] = frag * (unsigned)I8_FRAG_BYTES + (L ^ sw) * 8u;
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
        (void*)Aw, 0, 0xffffffffu, 0x31004000u);
#if I8_APF_K32
    u32x2_t A_pf[2];
#else
    u32x2_t A_pf[4];
#endif
    u32x2_t W_pf[2];
    unsigned ds_nx = 0u;
    unsigned sz_nx = 0u;

    // --- Prologue: DS[0] + SZ[0] + slab 0 of block 0 into slot 0. ---
    for (int i = tid; i < I8_BM * 2; i += I8_NTHREADS) {
        const int tok_idx = i >> 1;
        const int w = i & 1;
        const int tok = bs + tok_idx;
        const unsigned dw = (tok < N) ? ((const unsigned*)(Xq + tok))[w] : 0u;
        ((unsigned*)(LDS + I8_DS0_OFF + tok_idx * 8))[w] = dw;
    }
    if constexpr (SILU) {
        const int w = tid & 1;
        const int src_row = (sz_row < M) ? sz_row : (M - 1);
        const unsigned hw = *(const unsigned*)(As + (long long)src_row * groups_per_row * 136);
        const float f = (w == 0)
            ? __half2float(__ushort_as_half((unsigned short)(hw & 0xFFFFu)))
            : __half2float(__ushort_as_half((unsigned short)(hw >> 16)));
        ((float*)(LDS + I8_SZ0_OFF + (tid >> 1) * 8))[w] = (sz_row < M) ? f : 0.0f;
    } else {
        for (int i = tid; i < I8_BN * 2; i += I8_NTHREADS) {
            const int row_idx = i >> 1;
            const int w = i & 1;
            const int grow = rs + row_idx;
            const int src_row = (grow < M) ? grow : (M - 1);
            const unsigned hw = *(const unsigned*)(A + (long long)src_row * groups_per_row * 136);
            const float f = (w == 0)
                ? __half2float(__ushort_as_half((unsigned short)(hw & 0xFFFFu)))
                : __half2float(__ushort_as_half((unsigned short)(hw >> 16)));
            ((float*)(LDS + I8_SZ0_OFF + row_idx * 8))[w] = (grow < M) ? f : 0.0f;
        }
    }
    {
        const auto a0_rsrc = __builtin_amdgcn_make_buffer_rsrc((void*)Xq, 0, a_bytes, 0x31004000u);
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            #pragma unroll
            for (int sb = 0; sb < 2; sb++) {
                const u32x2_t av = __builtin_amdgcn_raw_buffer_load_b64(
                    a0_rsrc, st_avoff[r], 8u + (unsigned)sb * 32u, 0);
                *(u32x2_t*)(LDS + I8_SLOT0_OFF + st_alds[r] + sb * 512) = av;
            }
            const u32x2_t wv = __builtin_amdgcn_raw_buffer_load_b64(w_rsrc, st_wvoff[r], 8u, 0);
            *(u32x2_t*)(LDS + I8_SLOT0_OFF + I8_A_BYTES + st_wlds[r]) = wv;
        }
    }
    __syncthreads();

    for (int g = 0; g < groups_per_row; g++) {
        // Rolled (K1): only cacc/acc carry across the two halves.
        #pragma clang loop unroll_count(1)
        for (int h = 0; h < 2; h++) {
            const int kb = 2 * g + h;
            const bool last = (h == 1) && (g + 1 == groups_per_row);
            const char* __restrict__ ds_lds = LDS + ((h == 0) ? I8_DS0_OFF : I8_DS1_OFF);
            const char* __restrict__ sz_lds = LDS + ((h == 0) ? I8_SZ0_OFF : I8_SZ1_OFF);
            // Per-(token, K128) WMMA seeds. I8_ZP_SEED: 0x4B400000 - 8*s from
            // this block's staged token sum (published before the previous
            // rendezvous, or by the prologue).
            int seed[4];
            #pragma unroll
            for (int nb = 0; nb < 4; nb++) {
#if I8_ZP_SEED
                const int s_x = *(const int*)(ds_lds
                    + (unsigned)(wave_tok0 + nb * 16 + m_lane) * 8u + 4u);
                seed[nb] = I8_MAGIC - 8 * s_x;
#else
                seed[nb] = I8_MAGIC;
#endif
            }
            const auto a_rsrc = __builtin_amdgcn_make_buffer_rsrc(
                (void*)(Xq + (long long)kb * N), 0, a_bytes, 0x31004000u);
            // --- Slot 0 compute; slab 1 of this block -> slot 1. ---
            {
                #pragma unroll
                for (int r = 0; r < 2; r++) {
#if I8_APF_K32
                    A_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(a_rsrc, st_avoff[r], 72u, 0);
#else
                    A_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(a_rsrc, st_avoff[r], 72u, 0);
                    A_pf[2 + r] = __builtin_amdgcn_raw_buffer_load_b64(a_rsrc, st_avoff[r], 104u, 0);
#endif
                    W_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(
                        w_rsrc, st_wvoff[r], (unsigned)(g * 136 + h * 64 + 40), 0);
                }
                i8_bundle<0, true>(LDS + I8_SLOT0_OFF, a_lane, w_lane, cacc, seed);
#if I8_APF_K32
                // Slab 1 K32 half 0 -> slot 1 (free since the last rendezvous),
                // then fetch half 1 into the same private registers.
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    asm volatile("" : "+v"(A_pf[r][0]), "+v"(A_pf[r][1]));
                    *(u32x2_t*)(LDS + I8_SLOT1_OFF + st_alds[r]) = A_pf[r];
                }
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    A_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(a_rsrc, st_avoff[r], 104u, 0);
                }
#endif
                i8_bundle<1, false>(LDS + I8_SLOT0_OFF, a_lane, w_lane, cacc, seed);
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    asm volatile("" : "+v"(A_pf[r][0]), "+v"(A_pf[r][1]));
#if I8_APF_K32
                    *(u32x2_t*)(LDS + I8_SLOT1_OFF + st_alds[r] + 512) = A_pf[r];
#else
                    *(u32x2_t*)(LDS + I8_SLOT1_OFF + st_alds[r]) = A_pf[r];
                    *(u32x2_t*)(LDS + I8_SLOT1_OFF + st_alds[r] + 512) = A_pf[2 + r];
#endif
                    *(u32x2_t*)(LDS + I8_SLOT1_OFF + I8_A_BYTES + st_wlds[r]) = W_pf[r];
                }
                // Slot-0 retirement plus slot-1 readiness.
                __syncthreads();
                if (!last) {
                    // Next-block fetch epoch (K1): raw loads into private
                    // registers, first use at the slot-0 publishes below.
                    const int kb_nx = kb + 1;
                    const auto a_next_rsrc = __builtin_amdgcn_make_buffer_rsrc(
                        (void*)(Xq + (long long)kb_nx * N), 0, a_bytes, 0x31004000u);
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        A_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(a_next_rsrc, st_avoff[r], 8u, 0);
#if !I8_APF_K32
                        A_pf[2 + r] = __builtin_amdgcn_raw_buffer_load_b64(a_next_rsrc, st_avoff[r], 40u, 0);
#endif
                    }
                    const int g_nx = g + h;
                    const int h_nx = h ^ 1;
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        W_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(
                            w_rsrc, st_wvoff[r], (unsigned)(g_nx * 136 + h_nx * 64 + 8), 0);
                    }
                    const int tok_ds = bs + (tid >> 1);
                    const int tok_dsc = (tok_ds < N) ? tok_ds : (N - 1);
                    ds_nx = ((const unsigned*)(Xq + (long long)kb_nx * N + tok_dsc))[tid & 1];
                    if constexpr (SILU) {
                        const int src_sz = (sz_row < M) ? sz_row : (M - 1);
                        sz_nx = *(const unsigned*)(As + (long long)src_sz * groups_per_row * 136
                            + g_nx * 136 + h_nx * 4);
                    } else {
                        const int row_sz = rs + (tid >> 1);
                        const int src_sz = (row_sz < M) ? row_sz : (M - 1);
                        sz_nx = *(const unsigned*)(A + (long long)src_sz * groups_per_row * 136
                            + g_nx * 136 + h_nx * 4);
                    }
                }
            }
            // --- Slot 1 compute, fold, next slab 0 -> slot 0. ---
            {
                i8_bundle<0, false>(LDS + I8_SLOT1_OFF, a_lane, w_lane, cacc, seed);
#if I8_APF_K32
                // Next slab 0 K32 half 0 -> slot 0 (retired at the rendezvous
                // above; a no-op rewrite on the last block), then half 1. The
                // last block's descriptor has zero records: no access.
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    asm volatile("" : "+v"(A_pf[r][0]), "+v"(A_pf[r][1]));
                    *(u32x2_t*)(LDS + I8_SLOT0_OFF + st_alds[r]) = A_pf[r];
                }
                {
                    const auto a_next_rsrc = __builtin_amdgcn_make_buffer_rsrc(
                        (void*)(Xq + (long long)(kb + 1) * N), 0, last ? 0u : a_bytes, 0x31004000u);
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        A_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(a_next_rsrc, st_avoff[r], 40u, 0);
                    }
                }
#endif
                i8_bundle<1, false>(LDS + I8_SLOT1_OFF, a_lane, w_lane, cacc, seed);
                // Fold (per 128-K block, reads only cacc and the current
                // DS[h]/SZ[h] planes): fc exact, t1 = RN(sc*d), fma.
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
                        int* acc_i = (int*)&cacc[nb][rg];
                        const unsigned tbase = (unsigned)(wave_tok0 + nb * 16 + m_lane) * 8u;
                        const float d_x = *(const float*)(ds_lds + tbase);
#if I8_ZP_SEED
                        const float magic = 12582912.0f;
#else
                        const int s_x = *(const int*)(ds_lds + tbase + 4u);
                        const float magic = __builtin_bit_cast(float, I8_MAGIC + 8 * s_x);
#endif
                        #pragma unroll
                        for (int j = 0; j < 8; j += 2) {
                            const float2 pair = i8_fold_pair(
                                acc[nb][rg][j], acc[nb][rg][j + 1],
                                sc_row[j], sc_row[j + 1],
                                d_x, acc_i[j], acc_i[j + 1], magic);
                            acc[nb][rg][j] = pair.x;
                            acc[nb][rg][j + 1] = pair.y;
                        }
                    }
                }
                if (!last) {
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
#if I8_APF_K32
                        *(u32x2_t*)(LDS + I8_SLOT0_OFF + st_alds[r] + 512) = A_pf[r];
#else
                        *(u32x2_t*)(LDS + I8_SLOT0_OFF + st_alds[r]) = A_pf[r];
                        *(u32x2_t*)(LDS + I8_SLOT0_OFF + st_alds[r] + 512) = A_pf[2 + r];
#endif
                        *(u32x2_t*)(LDS + I8_SLOT0_OFF + I8_A_BYTES + st_wlds[r]) = W_pf[r];
                    }
                    const int h_np = h ^ 1;
                    char* __restrict__ ds_n = LDS + ((h_np == 0) ? I8_DS0_OFF : I8_DS1_OFF);
                    {
                        const int tok_dsp = bs + (tid >> 1);
                        ((unsigned*)(ds_n + (tid >> 1) * 8))[tid & 1] = (tok_dsp < N) ? ds_nx : 0u;
                    }
                    char* __restrict__ sz_n = LDS + ((h_np == 0) ? I8_SZ0_OFF : I8_SZ1_OFF);
                    {
                        const int row_szp = SILU ? sz_row : rs + (tid >> 1);
                        const int w_szp = tid & 1;
                        const float f = (w_szp == 0)
                            ? __half2float(__ushort_as_half((unsigned short)(sz_nx & 0xFFFFu)))
                            : __half2float(__ushort_as_half((unsigned short)(sz_nx >> 16)));
                        ((float*)(sz_n + (tid >> 1) * 8))[w_szp] = (row_szp < M) ? f : 0.0f;
                    }
                }
                // Slot-1 retirement plus next slot-0 readiness; on the last
                // block this guards store-tile reuse of LDS.
                __syncthreads();
            }
        } // for h
    } // for g

    // Epilogues (K1 verbatim: wide b128 tile stores for full tiles, LDS
    // transpose + checked b32 stores otherwise).
    float* slots = (float*)LDS;
    const int store_i = tid & 15;
    const int store_x = (tid >> 4) & 15;
    if constexpr (SILU) {
        const unsigned h_lane_off =
            ((unsigned)store_x * (unsigned)M + (unsigned)store_i) * sizeof(float);
        float* const h_tile = Y + (long long)bs * M + hs;
        const auto h_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            h_tile, 0, 0xffffffffu, 0x31004000u);
        const bool full_cols = bs + I8_BM <= N;
        if (full_cols && (M & 3) == 0 && (reinterpret_cast<uintptr_t>(Y) & 15u) == 0) {
            const unsigned dump_tok = (unsigned)(tok_half * 16 + m_lane);
            const unsigned wide_h_off =
                ((unsigned)(tid >> 4) * (unsigned)M + (unsigned)(tid & 15) * 4u) * sizeof(float);
            #pragma unroll
            for (int nb = 0; nb < 4; nb++) {
                #pragma unroll
                for (int c = 0; c < 2; c++) {
                    float4 v4;
                    v4.x = I8_SILU_MUL(acc[nb][0][4 * c + 0], acc[nb][1][4 * c + 0]);
                    v4.y = I8_SILU_MUL(acc[nb][0][4 * c + 1], acc[nb][1][4 * c + 1]);
                    v4.z = I8_SILU_MUL(acc[nb][0][4 * c + 2], acc[nb][1][4 * c + 2]);
                    v4.w = I8_SILU_MUL(acc[nb][0][4 * c + 3], acc[nb][1][4 * c + 3]);
                    *(float4*)(slots + dump_tok * 68u
                        + (unsigned)(wt_pair * 16 + 8 * k_grp + 4 * c)) = v4;
                }
                __syncthreads();
                #pragma unroll
                for (int it = 0; it < 2; it++) {
                    const float4 v4 = *(const float4*)(slots
                        + (unsigned)(it * 16 + (tid >> 4)) * 68u + (unsigned)(tid & 15) * 4u);
                    const unsigned phase_off =
                        (unsigned)(it * 64 + nb * 16) * (unsigned)M * sizeof(float);
                    *(float4*)((char*)h_tile + wide_h_off + phase_off) = v4;
                }
                __syncthreads();
            }
            return;
        }
        #pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                slots[(unsigned)(wave * 320 + (8 * k_grp + j) * 20 + m_lane)] =
                    I8_SILU_MUL(acc[nb][0][j], acc[nb][1][j]);
            }
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                const int p = k >> 1;
                const int hh = k & 1;
                const int owner = (p << 1) | hh;
                const float v = slots[(unsigned)(owner * 320 + store_i * 20 + store_x)];
                if (full_cols) {
                    const unsigned phase_off =
                        ((unsigned)(hh * 64 + nb * 16) * (unsigned)M + (unsigned)(p * 16))
                        * sizeof(float);
                    __builtin_amdgcn_raw_buffer_store_b32(
                        __builtin_bit_cast(unsigned, v), h_rsrc, h_lane_off, phase_off, 0);
                } else {
                    const int col = bs + hh * 64 + nb * 16 + store_x;
                    if (col < N) Y[(long long)col * M + hs + p * 16 + store_i] = v;
                }
            }
            __syncthreads();
        }
        return;
    }
    const unsigned store_lane_off =
        ((unsigned)store_x * (unsigned)M + (unsigned)store_i) * sizeof(float);
    float* const y_tile = Y + (long long)bs * M + rs;
    const auto y_tile_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        y_tile, 0, 0xffffffffu, 0x31004000u);
    const bool full_tile = rs + I8_BN <= M && bs + I8_BM <= N;
    if (full_tile && (M & 3) == 0 && (reinterpret_cast<uintptr_t>(Y) & 15u) == 0) {
        const unsigned wide_lane_off =
            ((unsigned)wave * (unsigned)M + (unsigned)lane * 4u) * sizeof(float);
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
                const unsigned phase_off =
                    (unsigned)((it >> 1) * 64 + nb * 16 + (it & 1) * 8) * (unsigned)M * sizeof(float);
                float4* const yp = (float4*)((char*)y_tile + wide_lane_off + phase_off);
                if constexpr (ADD) {
                    const float4 old = *yp;
                    float4 out;
                    out.x = old.x + v4.x;
                    out.y = old.y + v4.y;
                    out.z = old.z + v4.z;
                    out.w = old.w + v4.w;
                    *yp = out;
                } else {
                    *yp = v4;
                }
            }
            __syncthreads();
        }
        return;
    }
    if (full_tile) {
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
                    if constexpr (ADD) {
                        const unsigned old_bits = __builtin_amdgcn_raw_buffer_load_b32(
                            y_tile_rsrc, store_lane_off, phase_off, 0);
                        const float out = __builtin_bit_cast(float, old_bits) + v;
                        __builtin_amdgcn_raw_buffer_store_b32(
                            __builtin_bit_cast(unsigned, out),
                            y_tile_rsrc, store_lane_off, phase_off, 0);
                    } else {
                        __builtin_amdgcn_raw_buffer_store_b32(
                            __builtin_bit_cast(unsigned, v),
                            y_tile_rsrc, store_lane_off, phase_off, 0);
                    }
                }
                __syncthreads();
            }
        }
    } else {
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
                        float* yp = Y + (long long)col * M + row;
                        if constexpr (ADD) {
                            *yp += v;
                        } else {
                            *yp = v;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(I8_NTHREADS, 2)
void gemm_mq4g256v2_residual_mmq_i8_full_add(
    const char*           __restrict__ A,
    const block_i8_128*   __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int
) {
    gemm_mq4g256v2_residual_mmq_i8_gfx12_body<I8_EPI_ADD>(A, nullptr, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(I8_NTHREADS, 2)
void gemm_mq4g256v2_residual_mmq_i8_full_set(
    const char*           __restrict__ A,
    const block_i8_128*   __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int
) {
    gemm_mq4g256v2_residual_mmq_i8_gfx12_body<I8_EPI_SET>(A, nullptr, Xq, Y, M, K, N);
}

// Gate/up SwiGLU: G/U = gate/up MQ4V2 rows (M each, M % 128 == 0), H = h
// [N][M]. Grid [2M/128, ceil(N/128)], block [256], LDS 28672.
extern "C" __global__ __launch_bounds__(I8_NTHREADS, 2)
void gemm_mq4g256v2_gate_up_silu_mmq_i8(
    const char*           __restrict__ G,
    const char*           __restrict__ U,
    const block_i8_128*   __restrict__ Xq,
    float*                __restrict__ H,
    int M, int K, int N
) {
    gemm_mq4g256v2_residual_mmq_i8_gfx12_body<I8_EPI_SILU>(G, U, Xq, H, M, K, N);
}

#undef I8_SILU_MUL

#else

extern "C" __global__ void gemm_mq4g256v2_residual_mmq_i8_full_add(
    const char*, const void*, float*, int, int, int, int
) {}

extern "C" __global__ void gemm_mq4g256v2_residual_mmq_i8_full_set(
    const char*, const void*, float*, int, int, int, int
) {}

extern "C" __global__ void gemm_mq4g256v2_gate_up_silu_mmq_i8(
    const char*, const char*, const void*, float*, int, int, int
) {}

#endif
