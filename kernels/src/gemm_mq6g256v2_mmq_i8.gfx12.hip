// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// MQ6G256V2 exact-INT8 MMQ for gfx12 (RDNA4 — gfx1200/gfx1201): int8 per-128-K
// activations (`block_i8_128`) x MQ6G256V2 6-bit weight codes on
// `v_wmma_i32_16x16x16_iu8`. Prefill GEMM, overwrite (SET) epilogues, F32 or
// BF16 output.
//
// A fork of `gemm_mq4g256v2_residual_mmq_i8.gfx12.hip` (K1 lean-issue design):
// same 128x128 tile, 8 waves, 32-row x 64-token wave tile, K64 slabs in a
// two-slot LDS ring with 2 barriers per K128, banded CTA raster, wide b128
// epilogue, SRD addressing and next-block fetch epoch (I8_APF_K32 behaviour,
// I8_ZP_SEED behaviour: both fixed on). What changes:
//   - Weights: MQ6G256V2 group (200 B) = {u32 {f16 s0 | f16 z0 << 16}, u32
//     {s1 | z1 << 16}, 192 B little-endian 6-bit bitstream (code c = bits
//     [6c, 6c+6)); half h (codes 128h..128h+127) = payload bytes [96h, 96h+96),
//     K64 slab sl of the half = [48 sl, +48), K16 step ks = 12 bytes}. w = s_h *
//     q + z_h, q in [0, 63]. Staging loads one K16 step (3 dwords, 16 codes)
//     per lane per round and DECODES AT STAGING into an unpacked byte plane
//     (once per CTA, not per wave); the K loop reads the plane directly.
//   - Operand order: the codes go to the WMMA raw and UNSIGNED (signed_a =
//     false) against signed int8 activations. For each 8-code group the weight
//     operand dwords are {c0,c2,c4,c6} then {c1,c3,c5,c7} (the nibble-pair
//     order of block_i8_128.qs), so the pairing with the activation bytes needs
//     no shuffle: a lane (row, kg) at K16 step (sb, e) of a K64 slab holds the
//     8 codes of K = sb*32 + kg*16 + e*8 + 0..7. Decode: 6-bit fields are
//     spread into bytes (sp4), then two v_perm gather {c0,c2,c4,c6} /
//     {c1,c3,c5,c7}.
//   - Integer core (centered through the seed, like MQ4's I8_ZP_SEED): the WMMA
//     C seed of each (token, K128 block) chain is 0x4B400000 - 32 * s_t,kb
//     (broadcast from one VGPR per token), so the int32 accumulator ends at
//     0x4B400000 + C with C = sum_k (q_k - 32) * qx_k EXACT. |C| <= 32*127*128 =
//     520,192 < 2^22, so the float reinterpretation of 0x4B400000 + n is exactly
//     12582912 + n and fc = as_float(acc) - 12582912.0f is exact (also the seed
//     stays inside 2^22: |32 s| <= 520,192).
//   - Fold, in K128-block order (z' = z + 32 s from the f16 header, f32):
//       acc = fma(RN(s_h * d), fc, acc)
//       acc = fma(RN(z'_h * d), (float)s, acc)
//     so sum_k w*x = d * (s_h * C + z'_h * s) with d, s the block's activation
//     scale and sum. y = acc (no extra scale). BF16 output:
//     isfinite(v) ? (u + 0x7FFF + ((u >> 16) & 1)) >> 16 : u >> 16 with
//     u = as_uint(0.0f + y) (the incumbent formula).
//   - Tails: M and N need not be tile multiples. Weight rows >= M clamp to row
//     M-1 (SZ planes zeroed, outputs never stored); the activation descriptor
//     spans exactly one K128 block of N tokens, so tokens >= N read zero; the
//     epilogue guards r < M, t < N.
//
// gfx12 iu8 K16 conventions (gemm_hfq4g256_residual_mmq.gfx12.hip): int32x2
// A/B per lane (8 int8), lanes 0-15 carry one K8 half of the step and lanes
// 16-31 the other, contiguous-row C output acc[j] = C[8*k_grp + j][m_lane].
//
// Entries (ABI shared by both): (const char* A, const block_i8_128* Xq,
// void* Y, int M, int K, int N), K % 256 == 0, Y = [N][M] row-major, overwrite:
//   gemm_mq6g256v2_mmq_i8_set_gfx12        F32 out
//   gemm_mq6g256v2_mmq_i8_set_bf16_gfx12   BF16 bits out
// Grid [ceil(M/128), ceil(N/128)], block [256,1,1], dynamic LDS =
// MQ6_I8_GFX12_LDS_BYTES (36864 B). Compile gate: gfx1200 / gfx1201 only
// (empty stubs otherwise).
//
// LDS map (36864 B; the epilogue store tiles overlap the dead compute planes):
//   slot s (s = 0, 1) @ s*16384:
//     A_s = 32 K16 fragments x 256 B (fragment order below)          (8192 B)
//     W_s = 32 K16 fragments x 256 B (unpacked codes, A's order)     (8192 B)
//   DS0/DS1 = [128][8] token (d f32, s i32) per kb parity  @ 32768 (2 x 1024)
//   SZ0/SZ1 = [128][8] row (s f32, z' f32) per kb parity   @ 34816 (2 x 1024)
// Fragment order (A and W): fragment f = g8*4 + sb*2 + e (g8 = 16-token / 16-row
// group, sb = K32 sub-block, e = K16 half) holds the 8-byte operand lanes of
// one K16 step; lane L's bytes sit at f*256 + (L ^ sw(e, L>>4))*8 with K1's
// swizzle sw = (e << 3) ^ ((L >> 4) << 2).
//
// VGPRs: see the build report (<= 192 target, no scratch); __launch_bounds__
// (256, 2).

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

#define I8_FRAG_BYTES 256
#define I8_A_BYTES (I8_BM * I8_BK)
#define I8_W_BYTES (I8_BN * I8_BK)
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
// The dynamic LDS the host passes.
#define MQ6_I8_GFX12_LDS_BYTES 36864
_Static_assert(I8_LDS_BYTES == MQ6_I8_GFX12_LDS_BYTES, "i8 planes must match the host LDS contract");
_Static_assert(MQ6_I8_GFX12_LDS_BYTES <= 65536, "LDS above 64 KiB");
_Static_assert(32 * 132 * 4 <= MQ6_I8_GFX12_LDS_BYTES, "wide store tile exceeds LDS");
#define I8_MAGIC 0x4B400000
#define I8_GROUP_BYTES 200u

typedef int32_t __attribute__((ext_vector_type(2)))  int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8)))  int32x8_t;
typedef float   __attribute__((ext_vector_type(8)))  float8_t;
typedef unsigned __attribute__((ext_vector_type(2))) u32x2_t;

static __device__ __forceinline__ int32x8_t i8_splat(int v) {
    return (int32x8_t){v, v, v, v, v, v, v, v};
}

// Spread the four 6-bit codes of a 24-bit field into the four bytes of a dword.
static __device__ __forceinline__ unsigned i8_sp4(unsigned t) {
    return ((t & 0x3Fu) | ((t << 2) & 0x3F00u) | ((t << 4) & 0x3F0000u)
        | ((t << 6) & 0x3F000000u)) & 0x3F3F3F3Fu;
}

// 16 codes in w0 | w1<<32 | w2<<64 (code i = bits [6i, 6i+6)) -> the two iu8
// operands of the K16 step: e0 = codes 0..7 as {c0,c2,c4,c6 | c1,c3,c5,c7},
// e1 = codes 8..15 likewise (raw unsigned codes 0..63).
static __device__ __forceinline__ void i8_expand16(
    unsigned w0, unsigned w1, unsigned w2, u32x2_t& e0, u32x2_t& e1) {
    const unsigned b0 = i8_sp4(w0);                      // c0 c1 c2 c3
    const unsigned b1 = i8_sp4((w0 >> 24) | (w1 << 8));  // c4 c5 c6 c7
    const unsigned b2 = i8_sp4((w1 >> 16) | (w2 << 16)); // c8 c9 c10 c11
    const unsigned b3 = i8_sp4(w2 >> 8);                 // c12 .. c15
    // v_perm_b32(S0, S1, sel): result byte i = {S1 bytes 0-3, S0 bytes 4-7}[sel_i].
    e0[0] = __builtin_amdgcn_perm(b1, b0, 0x06040200u);
    e0[1] = __builtin_amdgcn_perm(b1, b0, 0x07050301u);
    e1[0] = __builtin_amdgcn_perm(b3, b2, 0x06040200u);
    e1[1] = __builtin_amdgcn_perm(b3, b2, 0x07050301u);
}

// f16 header word {lo: s, hi: z} -> {s, z + 32 s} (f32); `w` picks the lane's half.
static __device__ __forceinline__ float i8_sz_value(unsigned hw, int w) {
    const float s_f = __half2float(__ushort_as_half((unsigned short)(hw & 0xFFFFu)));
    const float z_f = __half2float(__ushort_as_half((unsigned short)(hw >> 16)));
    return (w == 0) ? s_f : (z_f + 32.0f * s_f);
}

// One K32 sub-block (two K16 steps): per step 2 W and 4 A b64 fragment reads,
// 8 WMMAs. SB is a template constant (LDS immediates); FIRST seeds each chain's
// first step. a_lane[e] / w_lane[e] are the swizzled lane byte bases (plus the
// wave's token / row group), so every read is VGPR + immediate.
template <int SB, bool FIRST>
static __device__ __forceinline__ void i8_bundle(
    const char* __restrict__ slot,
    const unsigned (&a_lane)[2],
    const unsigned (&w_lane)[2],
    int32x8_t cacc[4][2],
    const int (&seed)[4]
) {
    #pragma unroll
    for (int e = 0; e < 2; e++) {
        const char* a_base = slot + a_lane[e] + SB * 512 + e * 256;
        const char* w_base = slot + I8_A_BYTES + w_lane[e] + SB * 512 + e * 256;
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

// Raw loads of one K16 step (3 dwords) of a lane's weight row.
static __device__ __forceinline__ void i8_load_w3(
    __amdgpu_buffer_rsrc_t rsrc, unsigned voff, unsigned soff, unsigned (&d)[3]) {
    d[0] = __builtin_amdgcn_raw_buffer_load_b32(rsrc, voff, soff, 0);
    d[1] = __builtin_amdgcn_raw_buffer_load_b32(rsrc, voff, soff + 4u, 0);
    d[2] = __builtin_amdgcn_raw_buffer_load_b32(rsrc, voff, soff + 8u, 0);
}

// Decode a raw K16 step and publish its two operand fragments into W plane `wp`.
static __device__ __forceinline__ void i8_publish_w(
    char* __restrict__ wp, const unsigned (&lds)[2], const unsigned (&d)[3]) {
    u32x2_t e0, e1;
    i8_expand16(d[0], d[1], d[2], e0, e1);
    *(u32x2_t*)(wp + lds[0]) = e0;
    *(u32x2_t*)(wp + lds[1]) = e1;
}

static __device__ __forceinline__ unsigned short i8_bf16_bits(float y) {
    const float v = 0.0f + y;
    const unsigned u = __float_as_uint(v);
    return isfinite(v) ? (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16)
                       : (unsigned short)(u >> 16);
}

template <bool BF16>
static __device__ __forceinline__ void gemm_mq6g256v2_mmq_i8_gfx12_body(
    const char*           __restrict__ A,   // [M x groups_per_row x 200]
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
    if (rs >= M || bs >= N) return;

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
    // W fragment (g8w = wt_pair*2 + rg, sb, e) at g8w*1024 + sb*512 + e*256
    // (+ A plane).
    const unsigned a_lane[2] = {
        (unsigned)tok_half * 4096u + lane_even,
        (unsigned)tok_half * 4096u + lane_odd,
    };
    const unsigned w_lane[2] = {
        (unsigned)wt_pair * 2048u + lane_even,
        (unsigned)wt_pair * 2048u + lane_odd,
    };

    float8_t acc[4][2];
    // Staging maps (all loop-invariant):
    //   A (per K32 half sb): round r stages token T = r*64 + (tid>>2), 8-byte
    //     chunk c = tid&3 (lane half c>>1, K16 half e = c&1) of that half:
    //     global T*136 + c*8 (+ 8 + slab*64 + sb*32 in the scalar offset),
    //     LDS frag (T>>4)*4 + sb*2 + e, lane (T&15) + 16*(c>>1), swizzled.
    //   W: round r stages row R = r*64 + (tid>>2), K16 step ks = tid&3 of the
    //     slab (12 bytes = 16 codes): sb = ks>>1, kg = ks&1; operand e (codes
    //     8e..8e+7) -> frag (R>>4)*4 + sb*2 + e, lane (R&15) + 16*kg, swizzled.
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
            st_wvoff[r] = (unsigned)src_c * (unsigned)groups_per_row * I8_GROUP_BYTES + c * 12u;
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
    for (int i = tid; i < I8_BM * 2; i += I8_NTHREADS) {
        const int tok_idx = i >> 1;
        const int w = i & 1;
        const int tok = bs + tok_idx;
        const unsigned dw = (tok < N) ? ((const unsigned*)(Xq + tok))[w] : 0u;
        ((unsigned*)(LDS + I8_DS0_OFF + tok_idx * 8))[w] = dw;
    }
    for (int i = tid; i < I8_BN * 2; i += I8_NTHREADS) {
        const int row_idx = i >> 1;
        const int w = i & 1;
        const int grow = rs + row_idx;
        const int src_row = (grow < M) ? grow : (M - 1);
        const unsigned hw = *(const unsigned*)(A + (long long)src_row * groups_per_row * I8_GROUP_BYTES);
        ((float*)(LDS + I8_SZ0_OFF + row_idx * 8))[w] = (grow < M) ? i8_sz_value(hw, w) : 0.0f;
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
            unsigned wd[3];
            i8_load_w3(w_rsrc, st_wvoff[r], 8u, wd);
            i8_publish_w(LDS + I8_SLOT0_OFF + I8_A_BYTES, st_wlds[r], wd);
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
            // Per-(token, K128) WMMA seeds: 0x4B400000 - 32*s from this
            // block's staged token sum (published before the previous
            // rendezvous, or by the prologue).
            int seed[4];
            #pragma unroll
            for (int nb = 0; nb < 4; nb++) {
                const int s_x = *(const int*)(ds_lds
                    + (unsigned)(wave_tok0 + nb * 16 + m_lane) * 8u + 4u);
                seed[nb] = I8_MAGIC - 32 * s_x;
            }
            const auto a_rsrc = __builtin_amdgcn_make_buffer_rsrc(
                (void*)(Xq + (long long)kb * N), 0, a_bytes, 0x31004000u);
            // --- Slot 0 compute; slab 1 of this block -> slot 1. ---
            {
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    A_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(a_rsrc, st_avoff[r], 72u, 0);
                    i8_load_w3(w_rsrc, st_wvoff[r],
                        (unsigned)g * I8_GROUP_BYTES + 8u + (unsigned)h * 96u + 48u, W_pf[r]);
                }
                i8_bundle<0, true>(LDS + I8_SLOT0_OFF, a_lane, w_lane, cacc, seed);
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
                i8_bundle<1, false>(LDS + I8_SLOT0_OFF, a_lane, w_lane, cacc, seed);
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    asm volatile("" : "+v"(A_pf[r][0]), "+v"(A_pf[r][1]));
                    asm volatile("" : "+v"(W_pf[r][0]), "+v"(W_pf[r][1]), "+v"(W_pf[r][2]));
                    *(u32x2_t*)(LDS + I8_SLOT1_OFF + st_alds[r] + 512) = A_pf[r];
                    i8_publish_w(LDS + I8_SLOT1_OFF + I8_A_BYTES, st_wlds[r], W_pf[r]);
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
                    }
                    const int g_nx = g + h;
                    const int h_nx = h ^ 1;
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        i8_load_w3(w_rsrc, st_wvoff[r],
                            (unsigned)g_nx * I8_GROUP_BYTES + 8u + (unsigned)h_nx * 96u, W_pf[r]);
                    }
                    const int tok_ds = bs + (tid >> 1);
                    const int tok_dsc = (tok_ds < N) ? tok_ds : (N - 1);
                    ds_nx = ((const unsigned*)(Xq + (long long)kb_nx * N + tok_dsc))[tid & 1];
                    const int row_sz = rs + (tid >> 1);
                    const int src_sz = (row_sz < M) ? row_sz : (M - 1);
                    sz_nx = *(const unsigned*)(A + (long long)src_sz * groups_per_row * I8_GROUP_BYTES
                        + (unsigned)g_nx * I8_GROUP_BYTES + (unsigned)h_nx * 4u);
                }
            }
            // --- Slot 1 compute, fold, next slab 0 -> slot 0. ---
            {
                i8_bundle<0, false>(LDS + I8_SLOT1_OFF, a_lane, w_lane, cacc, seed);
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
                i8_bundle<1, false>(LDS + I8_SLOT1_OFF, a_lane, w_lane, cacc, seed);
                // Fold (per 128-K block, reads only cacc and the current
                // DS[h]/SZ[h] planes): fc exact,
                //   acc = fma(RN(s*d), fc, acc); acc = fma(RN(z'*d), (float)s, acc).
                #pragma unroll
                for (int rg = 0; rg < 2; rg++) {
                    float sc_row[8];
                    float zp_row[8];
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const unsigned jbase = (unsigned)(wave_row0 + rg * 16 + 8 * k_grp + j) * 8u;
                        sc_row[j] = *(const float*)(sz_lds + jbase);
                        zp_row[j] = *(const float*)(sz_lds + jbase + 4u);
                    }
                    #pragma unroll
                    for (int nb = 0; nb < 4; nb++) {
                        int* acc_i = (int*)&cacc[nb][rg];
                        const unsigned tbase = (unsigned)(wave_tok0 + nb * 16 + m_lane) * 8u;
                        const float d_x = *(const float*)(ds_lds + tbase);
                        const float s_f = (float)*(const int*)(ds_lds + tbase + 4u);
                        const float magic = 12582912.0f;
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            const float fc = __builtin_bit_cast(float, acc_i[j]) - magic;
                            float v = __builtin_fmaf(__fmul_rn(sc_row[j], d_x), fc, acc[nb][rg][j]);
                            v = __builtin_fmaf(__fmul_rn(zp_row[j], d_x), s_f, v);
                            acc[nb][rg][j] = v;
                        }
                    }
                }
                if (!last) {
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        *(u32x2_t*)(LDS + I8_SLOT0_OFF + st_alds[r] + 512) = A_pf[r];
                        i8_publish_w(LDS + I8_SLOT0_OFF + I8_A_BYTES, st_wlds[r], W_pf[r]);
                    }
                    const int h_np = h ^ 1;
                    char* __restrict__ ds_n = LDS + ((h_np == 0) ? I8_DS0_OFF : I8_DS1_OFF);
                    {
                        const int tok_dsp = bs + (tid >> 1);
                        ((unsigned*)(ds_n + (tid >> 1) * 8))[tid & 1] = (tok_dsp < N) ? ds_nx : 0u;
                    }
                    char* __restrict__ sz_n = LDS + ((h_np == 0) ? I8_SZ0_OFF : I8_SZ1_OFF);
                    {
                        const int row_szp = rs + (tid >> 1);
                        const int w_szp = tid & 1;
                        ((float*)(sz_n + (tid >> 1) * 8))[w_szp] =
                            (row_szp < M) ? i8_sz_value(sz_nx, w_szp) : 0.0f;
                    }
                }
                // Slot-1 retirement plus next slot-0 readiness; on the last
                // block this guards store-tile reuse of LDS.
                __syncthreads();
            }
        } // for h
    } // for g

    // Epilogue (K1: wide tile stores for full tiles, LDS transpose + checked
    // stores otherwise). `slots` reuses the dead compute planes.
    float* slots = (float*)LDS;
    const int store_i = tid & 15;
    const int store_x = (tid >> 4) & 15;
    const bool full_tile = rs + I8_BM <= M && bs + I8_BM <= N;
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
                    pk.x = (unsigned)i8_bf16_bits(v4.x) | ((unsigned)i8_bf16_bits(v4.y) << 16);
                    pk.y = (unsigned)i8_bf16_bits(v4.z) | ((unsigned)i8_bf16_bits(v4.w) << 16);
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
                        ((unsigned short*)Yv)[(long long)col * M + row] = i8_bf16_bits(v);
                    } else {
                        ((float*)Yv)[(long long)col * M + row] = v;
                    }
                }
            }
            __syncthreads();
        }
    }
}

extern "C" __global__ __launch_bounds__(I8_NTHREADS, 2)
void gemm_mq6g256v2_mmq_i8_set_gfx12(
    const char*           __restrict__ A,
    const block_i8_128*   __restrict__ Xq,
    void*                 __restrict__ Y,
    int M, int K, int N
) {
    gemm_mq6g256v2_mmq_i8_gfx12_body<false>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(I8_NTHREADS, 2)
void gemm_mq6g256v2_mmq_i8_set_bf16_gfx12(
    const char*           __restrict__ A,
    const block_i8_128*   __restrict__ Xq,
    void*                 __restrict__ Y,
    int M, int K, int N
) {
    gemm_mq6g256v2_mmq_i8_gfx12_body<true>(A, Xq, Y, M, K, N);
}

#else

extern "C" __global__ void gemm_mq6g256v2_mmq_i8_set_gfx12(
    const char*, const void*, void*, int, int, int
) {}

extern "C" __global__ void gemm_mq6g256v2_mmq_i8_set_bf16_gfx12(
    const char*, const void*, void*, int, int, int
) {}

#endif
