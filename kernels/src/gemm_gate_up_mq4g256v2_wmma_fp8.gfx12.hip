// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// FP8-WMMA MQ4-G256v2 gate/up prefill candidate for gfx1201 (RDNA4), B=12.
//
// Research kernel, default-off (host gate: HIPFIRE_GFX12_MQ4V2_FP8_GATEUP).
// Exact gfx1201, eager only. Companion to the F16 incumbent
// gemm_gate_up_mq4g256v2_wmma_gfx12_bt12 in
// kernels/src/gemm_gate_up_mq4g256v2_wmma_gfx12_bt.hip.
//
// Geometry (identical to the incumbent): one wave32 workgroup computes a
// 16-row x 192-batch tile (B=12 independent chains of 16 batch rows).
// Grid: [ceil((gate_m+up_m)/16), ceil(N/192), 1]. Block: [32].
// Lane decomposition: m_lane = tid & 15, kg = tid >> 4. gfx12 C-mapping:
// acc[b][j] holds C[8*kg+j][m_lane] for batch chain b. Overwrite
// semantics (y = row_scale * s_final * acc).
//
// Twelve-chain invariant: all 12 v8f WMMA chains stay live through every
// K16 tile. No schedule with fewer chains is admissible here.
//
// Per 256-K group the two 128-K halves are explicit. Group layout is the
// serialized MQ4v2 contract: 8 header bytes (two packed FP16 (scale,zero)
// pairs for K[0,128) and K[128,256)) followed by 128 payload bytes
// (256 nibbles). Half h covers payload bytes [h*64, h*64+64). Within a
// half, K is re-sliced into four 32-K pairs u: lane group kg owns
// K[u*32+kg*16, +16) as two consecutive 8-element fragments (8 payload
// bytes at +u*16+kg*8), so one 16-byte X load and one 8-byte weight load
// feed two WMMAs. Dot products are K-order invariant, so this differs from
// the F16 incumbent's K assignment only in f32 summation order.
//
// Accumulation: a single acc[12] kept in units of the current half's
// per-row scale; at each half boundary acc *= s_prev/s_new and the
// zero-point term (z_h/s_h)*sum_half(x) is added, then the WMMAs
// accumulate raw q.x directly. No LDS, no partial chains.
//
// Nibble q -> E4M3: every integer 0..15 is exact in E4M3
// (sign0|exp4|mant3, bias 7). Derivation: value = 2^(exp-7)*(1+mant/8)
// for normals, 0 for exp=mant=0:
//
//   q=0  -> 0x00 (zero)
//   q=1  -> 0x38: exp=7,mant=0  -> 2^0*1     = 1
//   q=2  -> 0x40: exp=8,mant=0  -> 2^1*1     = 2
//   q=3  -> 0x44: exp=8,mant=4  -> 2*1.5     = 3
//   q=4  -> 0x48: exp=9,mant=0  -> 4*1       = 4
//   q=5  -> 0x4A: exp=9,mant=2  -> 4*1.25    = 5
//   q=6  -> 0x4C: exp=9,mant=4  -> 4*1.5     = 6
//   q=7  -> 0x4E: exp=9,mant=6  -> 4*1.75    = 7
//   q=8  -> 0x50: exp=10,mant=0 -> 8*1       = 8
//   q=9  -> 0x51: exp=10,mant=1 -> 8*1.125   = 9
//   q=10 -> 0x52: exp=10,mant=2 -> 8*1.25    = 10
//   q=11 -> 0x53: exp=10,mant=3 -> 8*1.375   = 11
//   q=12 -> 0x54: exp=10,mant=4 -> 8*1.5     = 12
//   q=13 -> 0x55: exp=10,mant=5 -> 8*1.625   = 13
//   q=14 -> 0x56: exp=10,mant=6 -> 8*1.75    = 14
//   q=15 -> 0x57: exp=10,mant=7 -> 8*1.875   = 15
//
// The 16 bytes live as four 32-bit immediates (q 0..3, 4..7, 8..11,
// 12..15) looked up with v_perm_b32. No LDS/global lookup table.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef int   __attribute__((ext_vector_type(2))) v2i_t;
typedef int   __attribute__((ext_vector_type(4))) v4i_t;
typedef float __attribute__((ext_vector_type(8))) v8f_t;

// Symmetric MQ4V2 twin: the artifact guarantees z = -8*d for every
// 128-value half-group. Center the nibble in E4M3 before WMMA so the fold is
// one scale FMA and neither activation half-sums nor zero metadata are read.
#ifndef HIPFIRE_FP8_SYMFOLD
#define HIPFIRE_FP8_SYMFOLD 0
#endif

#ifndef HIPFIRE_FP8_WPRESHUFFLE
#define HIPFIRE_FP8_WPRESHUFFLE 0
#endif

// Expand one 32-bit nibble pack (8 q values, q_i = (pk >> 4*i) & 0xF)
// into one per-lane FP8 WMMA operand (8 E4M3 bytes) bit-cast to v2i_t.
// Byte i of the result is the E4M3 encoding of q_i per the table above.
// The 16 table bytes live as four 32-bit immediates selected by the top
// two nibble bits; the low two bits pick the byte offset. No LDS/global
// lookup table.
// Spread the 4 nibbles of a 16-bit value into 4 byte lanes (nibble i ->
// byte i), so v_perm_b32 can use them as byte selectors. LUT-arg path only;
// the uniform path below uses the cheaper even/odd split + gather instead.
#ifdef HIPFIRE_FP8_LUT_ARG
__device__ static inline unsigned int mq4_spread4(unsigned int x16) {
    const unsigned int a = (x16 & 0x00FFu) | ((x16 & 0xFF00u) << 8);
    return (a & 0x000F000Fu) | ((a & 0x00F000F0u) << 4);
}
#endif // HIPFIRE_FP8_LUT_ARG

// Table lookup of 8 nibbles: byte i of out[0] = E4M3(nibble i), byte i of
// out[1] = E4M3(nibble i+4). Same operand order as the scalar reference.
#ifdef HIPFIRE_FP8_LUT_ARG
// LUT-arg form (MQ4G256V2-Lloyd, qt=52): the 16 table bytes arrive as four
// kernel-arg dwords (centered codebook E4M3(L-7.5), 4 bytes packed LE per
// dword) instead of the uniform immediates below. Uniform builds are
// textually unchanged.
__device__ static inline unsigned int mq4_lut8(unsigned int sel,
    unsigned int C0, unsigned int C1, unsigned int C2, unsigned int C3) {
    // v_perm_b32(hi, lo, s): result byte i = {lo,hi} byte s_i for s_i in 0..7.
    const unsigned int lo = __builtin_amdgcn_perm(C1, C0, sel & 0x07070707u);
    const unsigned int hi = __builtin_amdgcn_perm(C3, C2, sel & 0x07070707u);
    const unsigned int m = ((sel >> 3) & 0x01010101u) * 0xFFu;
    return (hi & m) | (lo & ~m);
}
#else
// gfx120x device only. The !__HIP_DEVICE_COMPILE__ arm lets the host pass
// (which also preprocesses this TU) through; a non-gfx12 DEVICE compile
// still hits the #error below instead of silently building the wrong idiom.
#if defined(__gfx1200__) || defined(__gfx1201__) || !defined(__HIP_DEVICE_COMPILE__)
#if HIPFIRE_FP8_SYMFOLD
// q -> E4M3(q-8), held in four immediate dwords:
// [-8,-7,-6,-5], [-4,-3,-2,-1], [0,1,2,3], [4,5,6,7].
// Two table perms plus the top-bit select replace scalar int->f32->fp8
// conversion. The selector construction lowers to the same v_perm/lshl-or
// idiom as the unsigned table without any per-element arithmetic.
__device__ static inline unsigned int mq4_lut8(unsigned int sel) {
    const unsigned int C0 = 0xCACCCED0u;
    const unsigned int C1 = 0xB8C0C4C8u;
    const unsigned int C2 = 0x44403800u;
    const unsigned int C3 = 0x4E4C4A48u;
    const unsigned int idx = sel & 0x07070707u;
    const unsigned int lo = __builtin_amdgcn_perm(C1, C0, idx);
    const unsigned int hi = __builtin_amdgcn_perm(C3, C2, idx);
    const unsigned int selp = 0x03020100u | ((sel & 0x08080808u) >> 1);
    return __builtin_amdgcn_perm(hi, lo, selp);
}
#else
// Uniform-table gfx120x fast path (bit-exact vs the two-perm form above:
// exhaustive on-device check of all 65536 x16 x 2 halves, see
// /tmp/wmma_probe/expand_probe2.hip candidate B). The upper half of the
// table is affine (q 8..15 -> 0x50+(q-8)), so one 8-byte perm serves q&7
// and the final select is itself a v_perm keyed on the top nibble bit.
// Per 4 nibbles this swaps one table v_perm plus the v_mul mask expansion
// and the and/andn blend for one blend v_perm plus and/or/add, removing
// the multi-cycle v_mul_lo_u32 pair from the chain.
// Probe (gfx1201, load-fed): expand-only 0.74x, expand+wmma 0.77x.
__device__ static inline unsigned int mq4_lut8(unsigned int sel) {
    // q 0..7: 00 38 40 44 48 4A 4C 4E (single 8-byte table).
    // q 8..15: 50 51 52 53 54 55 56 57 = 0x50+(q-8).
    const unsigned int C0 = 0x44403800u;
    const unsigned int C1 = 0x4E4C4A48u;
    const unsigned int idx = sel & 0x07070707u;
    const unsigned int lo = __builtin_amdgcn_perm(C1, C0, idx);
    // idx bytes are q&7 (<8): +0x50 never carries across byte lanes.
    const unsigned int hi = idx + 0x50505050u;
    // v_perm_b32(hi, lo, s): result byte i = {lo,hi} byte s_i for s_i in 0..7.
    // Selector byte i = i | ((q_i>>3)<<2): 0..3 -> lo lane, 4..7 -> hi lane.
    // (sel & 0x08080808u) keeps only top bits, so >>1 lands cleanly on bit 2.
    const unsigned int selp = 0x03020100u | ((sel & 0x08080808u) >> 1);
    return __builtin_amdgcn_perm(hi, lo, selp);
}
#endif
#else
#error "uniform mq4_lut8 perm-blend fast path requires gfx1200/gfx1201"
#endif
#endif

#ifdef HIPFIRE_FP8_LUT_ARG
__device__ static inline v2i_t mq4_qpk_to_e4m3(unsigned int pk,
    unsigned int C0, unsigned int C1, unsigned int C2, unsigned int C3) {
    v2i_t out;
    out[0] = (int)mq4_lut8(mq4_spread4(pk & 0xFFFFu), C0, C1, C2, C3);
    out[1] = (int)mq4_lut8(mq4_spread4(pk >> 16), C0, C1, C2, C3);
    return out;
}
#else
__device__ static inline v2i_t mq4_qpk_to_e4m3(unsigned int pk) {
    // Even/odd nibble split + gather. pk byte i holds nibbles (2i+1, 2i),
    // so ev/od isolate them into per-byte slots (values 0..15, high nibble
    // clear) and one v_perm each gathers consecutive nibbles [n0..n3] /
    // [n4..n7] — bit-identical inputs to what mq4_spread4 produced above,
    // hence bit-identical E4M3 bytes out of the shared mq4_lut8 (pure
    // integer byte routing, no arithmetic anywhere). Per 8 nibbles this is
    // 3 (split) + 2 (gather) VALU vs 2x spread4 (~16 as compiled).
    // v_perm_b32(hi, lo, s): result byte i = {lo,hi} byte s_i.
    const unsigned int ev = pk & 0x0F0F0F0Fu;
    const unsigned int od = (pk >> 4) & 0x0F0F0F0Fu;
    const unsigned int g0 = __builtin_amdgcn_perm(od, ev, 0x05010400u);
    const unsigned int g1 = __builtin_amdgcn_perm(od, ev, 0x07030602u);
    v2i_t out;
    out[0] = (int)mq4_lut8(g0);
    out[1] = (int)mq4_lut8(g1);
    return out;
}
#endif
// Batch-tile width (chains per wave). Host selects the symbol by N:
// N%192==0 -> BV=12, N%128==0 -> BV=8, N%64==0 -> BV=4.
#ifndef HIPFIRE_FP8_BV
#define HIPFIRE_FP8_BV 12
#endif
#ifndef HIPFIRE_FP8_GATEUP_KERNEL
#define HIPFIRE_FP8_GATEUP_KERNEL gemm_gate_up_mq4g256v2_wmma_fp8_gfx12_bt12
#endif

// HIPFIRE_FP8_RESIDUAL=1 compiles the single-matrix residual form
// (Y[N x M] += W[M x K] . X) with the incumbent residual ABI
// (A, X_fp8, X_half_sums, X_row_scales, Y, M, K, N); the default is the
// two-matrix gate/up overwrite form. The K loop is shared.
#ifndef HIPFIRE_FP8_RESIDUAL
#define HIPFIRE_FP8_RESIDUAL 0
#endif

// HIPFIRE_FP8_QKVZA=1 compiles the 4-way fused DeltaNet projection form
// (A_qkv/A_z/A_beta/A_alpha -> Y_qkv/Y_z/Y_beta/Y_alpha, overwrite) with the
// incumbent qkvza ABI.
#ifndef HIPFIRE_FP8_QKVZA
#define HIPFIRE_FP8_QKVZA 0
#endif

// HIPFIRE_FP8_QKV=1 compiles the 3-way fused full-attention projection form
// (A_q/A_k/A_v -> Y_q/Y_k/Y_v, overwrite) with the incumbent QKV ABI
// (3 weights, 3 outputs, 3 row counts q_m/k_m/v_m); routing mirrors the
// F16 gfx1201 BT kernel (kernels/src/gemm_qkv_mq4g256v2_wmma_gfx1201_bt.hip).
#ifndef HIPFIRE_FP8_QKV
#define HIPFIRE_FP8_QKV 0
#endif

// Two-slab form: HIPFIRE_FP8_SLABS=2 gives each wave32 two 16-row slabs
// (rows [rs, rs+32)) so one 16 B X fragment feeds four WMMAs (two slabs x
// two K fragments) instead of two. acc[s][b], per-slab headers/scales and
// per-slab boundary correction; batch columns (sb, X, half-sums) shared.
// Grid becomes [ceil(M/32), ceil(N/(16*BV)), 1]. SLABS=1 is the current
// single-slab kernel, kept textually identical in the #else branches.
#ifndef HIPFIRE_FP8_SLABS
#define HIPFIRE_FP8_SLABS 1
#endif

#ifndef HIPFIRE_FP8_V2_TILE
#define HIPFIRE_FP8_V2_TILE 0
#endif

#if HIPFIRE_FP8_V2_TILE

// gfx1201 staged-tile v2: BM tokens x BN weight rows x BK, WAVES wave32s.
// Cooperative LDS A/W/S/s/z, expand-once at W staging, half-local P then
// T = fma(s_h,P,T); T = fma(z_h,S_h,T) at each 128-K half; a[t] at store.
// Balanced aspects (128x128x64/8w, 64x256x64/8w, 128x64x64/4w control) cut
// A-slab re-reads at the same 64-accumulator wave footprint; 256x128 stays
// pruned (no register headroom). U0 skipped: no ds256 path.
#ifndef HIPFIRE_FP8_V2_BM
#define HIPFIRE_FP8_V2_BM 256
#endif
#ifndef HIPFIRE_FP8_V2_BN
#define HIPFIRE_FP8_V2_BN 64
#endif
#ifndef HIPFIRE_FP8_V2_BK
#define HIPFIRE_FP8_V2_BK 64
#endif
#ifndef HIPFIRE_FP8_V2_WAVES
#define HIPFIRE_FP8_V2_WAVES 8
#endif

#ifdef HIPFIRE_FP8_LUT_ARG
#error "HIPFIRE_FP8_V2_TILE is uniform MQ4V2 only; LUT_ARG is not a v2 candidate"
#endif
#if HIPFIRE_FP8_V2_BK != 32 && HIPFIRE_FP8_V2_BK != 64
#error "HIPFIRE_FP8_V2_BK must be 32 or 64"
#endif
#if HIPFIRE_FP8_V2_BN != 32 && HIPFIRE_FP8_V2_BN != 64 && HIPFIRE_FP8_V2_BN != 128 && HIPFIRE_FP8_V2_BN != 256
#error "HIPFIRE_FP8_V2_BN must be 32, 64, 128 or 256"
#endif
#if HIPFIRE_FP8_V2_BM != 64 && HIPFIRE_FP8_V2_BM != 128 && HIPFIRE_FP8_V2_BM != 256
#error "HIPFIRE_FP8_V2_BM must be 64, 128 or 256"
#endif
#if (HIPFIRE_FP8_V2_WAVES * 32) % HIPFIRE_FP8_V2_BN != 0
#error "HIPFIRE_FP8_V2_WAVES*32 must be a multiple of HIPFIRE_FP8_V2_BN"
#endif
#if ((HIPFIRE_FP8_V2_BK / 8) % ((HIPFIRE_FP8_V2_WAVES * 32) / HIPFIRE_FP8_V2_BN)) != 0
#error "W staging producers must divide BK/8 chunks"
#endif
// Fragment-order LDS is family-gated: residual and qkvza (K-heavy shapes
// that outrun the VGPR 189->198 allocation cliff) stage A/W in WMMA
// fragment order; gate_up and qkv keep the row layout so their HSACO .text
// is byte-identical to the pre-fragment build. The switch keys on the
// existing per-family entry macros, so no host change is needed.
#if HIPFIRE_FP8_RESIDUAL || HIPFIRE_FP8_QKVZA
#define HIPFIRE_FP8_V2_FRAG_ORDER 1
#else
#define HIPFIRE_FP8_V2_FRAG_ORDER 0
#endif
#if HIPFIRE_FP8_V2_FRAG_ORDER
// Fragment-order LDS (fold-probe result): A/W planes are stored in WMMA
// fragment order — each (16-row block, K16 step) fragment is 256 contiguous
// bytes with lane L's 8 operand bytes at frag + L*8 (lane (ml,kg): row ml,
// cols kg*8, exactly what v_wmma_f32_16x16x16_fp8_fp8 consumes). The K-loop
// read is one ds_read_b64 at (wave-uniform frag base) + lane*8 with the
// (nb/wf, kstep) frag offset in the immediate — no address VALU in the loop.
// K-step bank shift: frag F lives at F*288, not F*256. Consecutive ksteps of
// a block sit 288 B apart (8 banks apart), so the four staging quads (one
// kstep each) never collide on writes; reads stay at the width floor (each
// bank twice per wave = the 256 B minimum). Enumerated per-issue worst = 2
// (floor) for every A/W store issue on all four admitted aspects; the old
// stride-(BK+8) row layout measured a ~19x bank-conflict penalty on the K-loop
// reads (loads alone cost the same as loads+WMMA). Plane sizes are
// (rows/16)*(BK/16)*288 — identical totals to the old rows*B*(BK+8), so the
// host vlds formula is untouched.
#define HIPFIRE_FP8_V2_NTHREADS (HIPFIRE_FP8_V2_WAVES * 32)
#define HIPFIRE_FP8_V2_FRAG     288
#define HIPFIRE_FP8_V2_A_OFF    0
#define HIPFIRE_FP8_V2_A_BYTES  ((HIPFIRE_FP8_V2_BM / 16) * (HIPFIRE_FP8_V2_BK / 16) * HIPFIRE_FP8_V2_FRAG)
#define HIPFIRE_FP8_V2_W_OFF    HIPFIRE_FP8_V2_A_BYTES
#define HIPFIRE_FP8_V2_W_BYTES  ((HIPFIRE_FP8_V2_BN / 16) * (HIPFIRE_FP8_V2_BK / 16) * HIPFIRE_FP8_V2_FRAG)
// Frag byte offset of (16-row block nb/wf, kstep ks) from the wave's frag-0
// base pointer (a_fptr/w_fptr). Unrolled indices only: always an immediate.
#define HIPFIRE_FP8_V2_AFRAG(nb, ks) (((nb) * HIPFIRE_FP8_V2_K16 + (ks)) * HIPFIRE_FP8_V2_FRAG)
#define HIPFIRE_FP8_V2_WFRAG(wf, ks) (((wf) * HIPFIRE_FP8_V2_K16 + (ks)) * HIPFIRE_FP8_V2_FRAG)
#else
// Balanced-aspect geometries (128x128x64/8w, 64x256x64/8w, 128x64x64/4w):
// stride stays BK+8 = 72, so the §3.2 bank argument is unchanged — a lane
// reading 8 contiguous bytes at (r*72 + kstep*16 + kg*8), r = lane&15,
// touches banks (18*r + 4*kstep + 2*kg + {0,1}) mod 32: all 16 even starts
// per 16-lane half, every bank once per half, twice per full wave (the
// width floor, not a conflict). Only the row extents change: A rows
// 0..BM-1, W rows 0..BN-1. Staging writers: every lane is active on every
// round (row = r*NTHREADS/4 + (tid>>2), quad tid&3); consecutive lanes
// touch consecutive 8 B chunks (coalesced globals, sequential LDS banks).
#define HIPFIRE_FP8_V2_NTHREADS (HIPFIRE_FP8_V2_WAVES * 32)
#define HIPFIRE_FP8_V2_STRIDE   (HIPFIRE_FP8_V2_BK + 8)
#define HIPFIRE_FP8_V2_A_OFF    0
#define HIPFIRE_FP8_V2_A_BYTES  (HIPFIRE_FP8_V2_BM * HIPFIRE_FP8_V2_STRIDE)
#define HIPFIRE_FP8_V2_W_OFF    HIPFIRE_FP8_V2_A_BYTES
#define HIPFIRE_FP8_V2_W_BYTES  (HIPFIRE_FP8_V2_BN * HIPFIRE_FP8_V2_STRIDE)
#endif
#if HIPFIRE_FP8_SYMFOLD
#define HIPFIRE_FP8_V2_S_OFF    (HIPFIRE_FP8_V2_W_OFF + HIPFIRE_FP8_V2_W_BYTES)
#define HIPFIRE_FP8_V2_S_BYTES  0
#define HIPFIRE_FP8_V2_SZ_OFF   HIPFIRE_FP8_V2_S_OFF
#define HIPFIRE_FP8_V2_SZ_BYTES (HIPFIRE_FP8_V2_BN * 4)
#else
#define HIPFIRE_FP8_V2_S_OFF    (HIPFIRE_FP8_V2_W_OFF + HIPFIRE_FP8_V2_W_BYTES)
#define HIPFIRE_FP8_V2_S_BYTES  (HIPFIRE_FP8_V2_BM * 4)
#define HIPFIRE_FP8_V2_SZ_OFF   (HIPFIRE_FP8_V2_S_OFF + HIPFIRE_FP8_V2_S_BYTES)
#define HIPFIRE_FP8_V2_SZ_BYTES (HIPFIRE_FP8_V2_BN * 8)
#endif
#define HIPFIRE_FP8_V2_LDS_BYTES (HIPFIRE_FP8_V2_SZ_OFF + HIPFIRE_FP8_V2_SZ_BYTES)
#define HIPFIRE_FP8_V2_NSLABS   (128 / HIPFIRE_FP8_V2_BK)
#define HIPFIRE_FP8_V2_K16      (HIPFIRE_FP8_V2_BK / 16)
#define HIPFIRE_FP8_V2_WCHUNKS  (HIPFIRE_FP8_V2_BK / 8)
#define HIPFIRE_FP8_V2_NPROD    (HIPFIRE_FP8_V2_NTHREADS / HIPFIRE_FP8_V2_BN)
#define HIPFIRE_FP8_V2_WITER    (HIPFIRE_FP8_V2_WCHUNKS / HIPFIRE_FP8_V2_NPROD)
// The T/P[4][2] accumulator footprint is a fixed 64-token x 32-row wave
// tile (4 token fragments x 2 weight fragments of 8 v8f), so every admitted
// aspect satisfies BM*BN == NTHREADS*64: 256x64x64/8w, 128x128x64/8w,
// 64x256x64/8w, 128x64x64/4w. wave_tok = tok_part*64 below relies on this.
#if (HIPFIRE_FP8_V2_BM * HIPFIRE_FP8_V2_BN) != (HIPFIRE_FP8_V2_NTHREADS * 64)
#error "HIPFIRE_FP8_V2 BM*BN must equal NTHREADS*64 (fixed 64x32 wave tile)"
#endif
// The coalesced store reuses LDS base for a 512-float/wave transpose tile
// (8 waves = 16 KiB); A/W/S/SZ are dead by then but the tile must fit.
#if HIPFIRE_FP8_V2_LDS_BYTES < (HIPFIRE_FP8_V2_WAVES * 2048)
#error "HIPFIRE_FP8_V2 LDS must hold the 512-float/wave store transpose tile"
#endif

extern "C" __launch_bounds__(HIPFIRE_FP8_V2_NTHREADS, 2) __global__
#if HIPFIRE_FP8_QKVZA
void HIPFIRE_FP8_GATEUP_KERNEL(
    const char* A_qkv,
    const char* A_z,
    const char* A_beta,
    const char* A_alpha,
    const unsigned char* X_fp8,
    const float* X_half_sums,
    const float* X_row_scales,
    float* Y_qkv,
    float* Y_z,
    float* Y_beta,
    float* Y_alpha,
    int qkv_m,
    int z_m,
    int beta_m,
    int alpha_m,
    int K,
    int N
) {
    const int tm = qkv_m + z_m + beta_m + alpha_m;
#elif HIPFIRE_FP8_RESIDUAL
void HIPFIRE_FP8_GATEUP_KERNEL(
    const char* A_w,
    const unsigned char* X_fp8,
    const float* X_half_sums,
    const float* X_row_scales,
    float* Y_res,
    int res_m,
    int K,
    int N
) {
    const int tm = res_m;
#elif HIPFIRE_FP8_QKV
void HIPFIRE_FP8_GATEUP_KERNEL(
    const char* A_q,
    const char* A_k,
    const char* A_v,
    const unsigned char* X_fp8,
    const float* X_half_sums,
    const float* X_row_scales,
    float* Y_q,
    float* Y_k,
    float* Y_v,
    int q_m,
    int k_m,
    int v_m,
    int K,
    int N
) {
    const int tm = q_m + k_m + v_m;
#else
void HIPFIRE_FP8_GATEUP_KERNEL(
    const char* A_gate,
    const char* A_up,
    const unsigned char* X_fp8,
    const float* X_half_sums,
    const float* X_row_scales,
    float* Y_gate,
    float* Y_up,
    int gate_m,
    int up_m,
    int K,
    int N
) {
    const int tm = gate_m + up_m;
#endif
    const int rs = (int)blockIdx.x * HIPFIRE_FP8_V2_BN;
    const int bs = (int)blockIdx.y * HIPFIRE_FP8_V2_BM;
    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int kg = lane >> 4;
    const int n_wt_parts = HIPFIRE_FP8_V2_BN / 32;
    const int tok_part = wave / n_wt_parts;
    const int wt_part = wave - tok_part * n_wt_parts;
    const int wave_tok = tok_part * 64;
    const int wave_wt = wt_part * 32;
    const int gpr = K / 256;
    const int ksteps = K >> 4;

    extern __shared__ char lds[];
    char* lds_A = lds + HIPFIRE_FP8_V2_A_OFF;
    char* lds_W = lds + HIPFIRE_FP8_V2_W_OFF;
    char* lds_S = lds + HIPFIRE_FP8_V2_S_OFF;
    char* lds_SZ = lds + HIPFIRE_FP8_V2_SZ_OFF;

#if HIPFIRE_FP8_V2_FRAG_ORDER
    // Fragment-order K-loop read pointers: one VGPR each, fully
    // loop-invariant (wave-uniform frag base + lane*8). Every K-loop LDS read
    // is *(v2i_t*)(ptr + IMM) with the (nb/wf, kstep) frag offset — at most
    // 15*288 = 4320 — folded into the ds immediate: no address VALU in the
    // loop. wave_tok/wave_wt are multiples of 16, so >>4 is the frag block.
    const unsigned lane8 = (unsigned)lane * 8u;
    const char* a_fptr = lds_A + (unsigned)(wave_tok >> 4)
        * (unsigned)HIPFIRE_FP8_V2_K16 * (unsigned)HIPFIRE_FP8_V2_FRAG + lane8;
    const char* w_fptr = lds_W + (unsigned)(wave_wt >> 4)
        * (unsigned)HIPFIRE_FP8_V2_K16 * (unsigned)HIPFIRE_FP8_V2_FRAG + lane8;
#endif
    // Staged-tile staging map (b64-ideal form; cf. iu4 b64 rewrite): 4 lanes
    // share a slab row (quad = tid&3 stages 2 packs/chunks per lane per
    // round); rows/round = NTHREADS/4, so A_ROUNDS/W_ROUNDS cover the BM/BN
    // extents exactly with every lane active on every round.
#if HIPFIRE_FP8_V2_FRAG_ORDER
    // Fragment order: each producer thread owns row a_row/w_row's 16 bytes
    // at quad*16 (kstep quad): block = row>>4, frag F = block*K16+quad,
    // slot = (row&15)*8. The 16 B scatter as two b64 frag slots
    // (F*288+slot, +128); W packs as one b64 each at (F*288+(ml+16*pp)*8).
    // Width ceilings (headers rule out anything wider): A global row base is
    // tok*K+g*256+h*128+slab*BK, 16 B-aligned (256|K), so even-chunk b128
    // pairs are exact; W packs are 4 B-stepped (b32 ceiling).
#else
    // Width ceilings (headers rule out anything wider): A global row base is
    // tok*K+g*256+h*128+slab*BK, 16 B-aligned (256|K), so even-chunk b128
    // pairs are exact; W packs are 4 B-stepped (b32 ceiling); LDS rows are
    // stride-72 (8 B-aligned), so all LDS traffic is b64 (dual-issue pairs).
#endif
    // All offsets below are loop-invariant (bs/rs/tid only), computed once
    // at kernel top; per-trip uniform bases (g/h/slab) stay in SGPRs so the
    // K-loop issues base+offset wide loads/stores with no per-load address
    // VALU. OOB clamps to the last valid index with a branch-free
    // zero-select (exact old ternary semantics); W keeps expand(pk) with a
    // zero pack (expand(0) is all-zero E4M3, bit-identical to the old zero).
#define HIPFIRE_FP8_V2_A_ROUNDS (HIPFIRE_FP8_V2_BM * 4 / HIPFIRE_FP8_V2_NTHREADS)
#define HIPFIRE_FP8_V2_W_ROUNDS (HIPFIRE_FP8_V2_BN * 4 / HIPFIRE_FP8_V2_NTHREADS)
#define HIPFIRE_FP8_V2_RPR (HIPFIRE_FP8_V2_NTHREADS / 4)
#if HIPFIRE_FP8_V2_NTHREADS == HIPFIRE_FP8_V2_BM
#define HIPFIRE_FP8_V2_S_SH 0
#elif HIPFIRE_FP8_V2_NTHREADS == HIPFIRE_FP8_V2_BM * 2
#define HIPFIRE_FP8_V2_S_SH 1
#elif HIPFIRE_FP8_V2_NTHREADS == HIPFIRE_FP8_V2_BM * 4
#define HIPFIRE_FP8_V2_S_SH 2
#else
#error "HIPFIRE_FP8_V2 BM must divide NTHREADS by 1, 2 or 4"
#endif
#if HIPFIRE_FP8_SYMFOLD
#define HIPFIRE_FP8_V2_SZITER ((HIPFIRE_FP8_V2_BN + HIPFIRE_FP8_V2_NTHREADS - 1) / HIPFIRE_FP8_V2_NTHREADS)
#else
#define HIPFIRE_FP8_V2_SZITER ((HIPFIRE_FP8_V2_BN * 2 + HIPFIRE_FP8_V2_NTHREADS - 1) / HIPFIRE_FP8_V2_NTHREADS)
#endif
    // Route a valid global row GR (< tm) to (ASRC, LR). All matrices cover
    // [0, tm) exactly as the old single-row router did; callers clamp first.
#if HIPFIRE_FP8_QKVZA
#define HIPFIRE_FP8_V2_PICK(GR, ASRC, LR) do { int _v2g = (GR); const char* _v2a = A_qkv; int _v2l = _v2g; if (_v2g >= qkv_m + z_m + beta_m) { _v2a = A_alpha; _v2l = _v2g - (qkv_m + z_m + beta_m); } else if (_v2g >= qkv_m + z_m) { _v2a = A_beta; _v2l = _v2g - (qkv_m + z_m); } else if (_v2g >= qkv_m) { _v2a = A_z; _v2l = _v2g - qkv_m; } (ASRC) = _v2a; (LR) = _v2l; } while (0)
#elif HIPFIRE_FP8_RESIDUAL
#define HIPFIRE_FP8_V2_PICK(GR, ASRC, LR) do { (ASRC) = A_w; (LR) = (GR); } while (0)
#elif HIPFIRE_FP8_QKV
#define HIPFIRE_FP8_V2_PICK(GR, ASRC, LR) do { int _v2g = (GR); const char* _v2a = A_q; int _v2l = _v2g; if (_v2g >= q_m + k_m) { _v2a = A_v; _v2l = _v2g - (q_m + k_m); } else if (_v2g >= q_m) { _v2a = A_k; _v2l = _v2g - q_m; } (ASRC) = _v2a; (LR) = _v2l; } while (0)
#else
#define HIPFIRE_FP8_V2_PICK(GR, ASRC, LR) do { int _v2g = (GR); const char* _v2a = A_gate; int _v2l = _v2g; if (_v2g >= gate_m) { _v2a = A_up; _v2l = _v2g - gate_m; } (ASRC) = _v2a; (LR) = _v2l; } while (0)
#endif
#if HIPFIRE_FP8_QKVZA
#define HIPFIRE_FP8_V2_ROWS(GR, ROWS) do { int _v2g = (GR); int _v2r = qkv_m; if (_v2g >= qkv_m + z_m + beta_m) _v2r = alpha_m; else if (_v2g >= qkv_m + z_m) _v2r = beta_m; else if (_v2g >= qkv_m) _v2r = z_m; (ROWS) = _v2r; } while (0)
#elif HIPFIRE_FP8_RESIDUAL
#define HIPFIRE_FP8_V2_ROWS(GR, ROWS) do { (void)(GR); (ROWS) = res_m; } while (0)
#elif HIPFIRE_FP8_QKV
#define HIPFIRE_FP8_V2_ROWS(GR, ROWS) do { int _v2g = (GR); int _v2r = q_m; if (_v2g >= q_m + k_m) _v2r = v_m; else if (_v2g >= q_m) _v2r = k_m; (ROWS) = _v2r; } while (0)
#else
#define HIPFIRE_FP8_V2_ROWS(GR, ROWS) do { (ROWS) = (GR) >= gate_m ? up_m : gate_m; } while (0)
#endif
    const unsigned st_quad = (unsigned)(tid & 3);
    const unsigned st_grp = (unsigned)(tid >> 2);
    const v2i_t st_z64 = {0, 0};
    // A entries: one b128 pair (2 chunks) per round.
    unsigned st_agoff[HIPFIRE_FP8_V2_A_ROUNDS];
#if HIPFIRE_FP8_V2_FRAG_ORDER
    // Combined slot-0 address (F*288 + ml*8); slot 1 is +128 (immediate).
    unsigned st_ab0[HIPFIRE_FP8_V2_A_ROUNDS];
#else
    unsigned st_aldsoff[HIPFIRE_FP8_V2_A_ROUNDS];
#endif
    unsigned st_aok[HIPFIRE_FP8_V2_A_ROUNDS];
    #pragma unroll
    for (int r = 0; r < HIPFIRE_FP8_V2_A_ROUNDS; r++) {
        const int a_row = r * HIPFIRE_FP8_V2_RPR + (int)st_grp;
        const int a_tok = bs + a_row;
        const int a_tok_c = (a_tok < N) ? a_tok : (N - 1);
        st_aok[r] = (a_tok < N) ? 1u : 0u;
        st_agoff[r] = (unsigned)a_tok_c * (unsigned)K + st_quad * 16u;
#if HIPFIRE_FP8_V2_FRAG_ORDER
        st_ab0[r] = ((unsigned)(a_row >> 4) * (unsigned)HIPFIRE_FP8_V2_K16 + st_quad)
            * (unsigned)HIPFIRE_FP8_V2_FRAG + (unsigned)(a_row & 15) * 8u;
#else
        st_aldsoff[r] = (unsigned)a_row * (unsigned)HIPFIRE_FP8_V2_STRIDE + st_quad * 16u;
#endif
    }
    // W entries: two packs per round, routed per round on the clamped row.
    const char* st_wbase[HIPFIRE_FP8_V2_W_ROUNDS];
    unsigned st_wok[HIPFIRE_FP8_V2_W_ROUNDS];
    unsigned st_wpoff[HIPFIRE_FP8_V2_W_ROUNDS][2];
#if HIPFIRE_FP8_V2_FRAG_ORDER
    // Combined frag-slot address (F*288 + (ml+16*pp)*8); pack pk = quad*2+pp
    // covers kstep quad, kg-half pp — the slot lane (ml,pp) reads.
    unsigned st_wslot[HIPFIRE_FP8_V2_W_ROUNDS][2];
#else
    unsigned st_wldsoff[HIPFIRE_FP8_V2_W_ROUNDS][2];
#endif
    #pragma unroll
    for (int r = 0; r < HIPFIRE_FP8_V2_W_ROUNDS; r++) {
        const int w_row = r * HIPFIRE_FP8_V2_RPR + (int)st_grp;
        const int w_grow = rs + w_row;
        const int w_grow_c = (w_grow < tm) ? w_grow : (tm - 1);
        const char* w_asrc;
        int w_lr;
        HIPFIRE_FP8_V2_PICK(w_grow_c, w_asrc, w_lr);
#if HIPFIRE_FP8_WPRESHUFFLE
        st_wbase[r] = w_asrc
            + ((((unsigned)(w_lr >> 4) * (unsigned)ksteps + st_quad) * 32u
                + (unsigned)(w_lr & 15)) * 4u);
#else
        st_wbase[r] = w_asrc + (unsigned int)w_lr * (unsigned int)gpr * 136u;
#endif
        st_wok[r] = (w_grow < tm) ? 1u : 0u;
        #pragma unroll
        for (int pp = 0; pp < 2; pp++) {
            const unsigned pk = st_quad * 2u + (unsigned)pp;
#if HIPFIRE_FP8_WPRESHUFFLE
            st_wpoff[r][pp] = (unsigned)pp * 64u;
#else
            st_wpoff[r][pp] = pk * 4u;
#endif
#if HIPFIRE_FP8_V2_FRAG_ORDER
            st_wslot[r][pp] = ((unsigned)(w_row >> 4) * (unsigned)HIPFIRE_FP8_V2_K16 + st_quad)
                * (unsigned)HIPFIRE_FP8_V2_FRAG + (unsigned)((w_row & 15) + pp * 16) * 8u;
#else
            st_wldsoff[r][pp] = (unsigned)w_row * (unsigned)HIPFIRE_FP8_V2_STRIDE + pk * 8u;
#endif
        }
    }
#if !HIPFIRE_FP8_SYMFOLD
    // S entry: one f32 per lane (2^S_SH lanes share a row, dup benign).
    const int st_srow = tid >> HIPFIRE_FP8_V2_S_SH;
    const int st_stok = bs + st_srow;
    const int st_stok_c = (st_stok < N) ? st_stok : (N - 1);
    const unsigned st_sok = (st_stok < N) ? 1u : 0u;
    const unsigned st_soff = (unsigned)st_stok_c * (unsigned)gpr * 2u;
    const unsigned st_slds = (unsigned)st_srow * 4u;
#endif
    // SZ entries: symmetric stores scale only; asymmetric stores scale+zero.
    const char* st_szbase[HIPFIRE_FP8_V2_SZITER];
    unsigned st_szok[HIPFIRE_FP8_V2_SZITER];
#if HIPFIRE_FP8_WPRESHUFFLE
    size_t st_szhstride[HIPFIRE_FP8_V2_SZITER];
#endif
#if !HIPFIRE_FP8_SYMFOLD
    unsigned st_szword[HIPFIRE_FP8_V2_SZITER];
#endif
    unsigned st_szlds[HIPFIRE_FP8_V2_SZITER];
    #pragma unroll
    for (int e = 0; e < HIPFIRE_FP8_V2_SZITER; e++) {
#if HIPFIRE_FP8_SYMFOLD
        const int sz_row = tid * HIPFIRE_FP8_V2_SZITER + e;
#else
        const int sz_idx = tid * HIPFIRE_FP8_V2_SZITER + e;
        const int sz_row = sz_idx & (HIPFIRE_FP8_V2_BN - 1);
#endif
        const int sz_grow = rs + sz_row;
        const int sz_grow_c = (sz_grow < tm) ? sz_grow : (tm - 1);
        const char* sz_asrc;
        int sz_lr;
        HIPFIRE_FP8_V2_PICK(sz_grow_c, sz_asrc, sz_lr);
#if HIPFIRE_FP8_WPRESHUFFLE
        int sz_rows;
        HIPFIRE_FP8_V2_ROWS(sz_grow_c, sz_rows);
        const size_t sz_row_tiles = (size_t)sz_rows >> 4;
        st_szbase[e] = sz_asrc + (size_t)sz_rows * (size_t)K / 2u
            + ((size_t)(sz_lr >> 4) * 4u) * 16u + (unsigned)(sz_lr & 15);
        st_szhstride[e] = sz_row_tiles * 64u;
#else
        st_szbase[e] = sz_asrc + (unsigned int)sz_lr * (unsigned int)gpr * 136u;
#endif
        st_szok[e] = (sz_grow < tm) ? 1u : 0u;
#if HIPFIRE_FP8_SYMFOLD
        st_szlds[e] = (unsigned)sz_row * 4u;
#else
        st_szword[e] = ((unsigned)sz_idx / (unsigned)HIPFIRE_FP8_V2_BN) & 1u;
        st_szlds[e] = (unsigned)sz_row * 8u + st_szword[e] * 4u;
#endif
    }

    const v8f_t zc = v8f_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    v8f_t T[4][2];
    v8f_t P[4][2];
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        T[nb][0] = zc;
        T[nb][1] = zc;
    }

    #pragma nounroll
    for (int g = 0; g < gpr; g++) {
        #pragma nounroll
        for (int h = 0; h < 2; h++) {
            // Raw prefetch (R1): prologue stages slab 0, then each iteration
            // issues slab s+1's global loads into registers BEFORE slab s's
            // compute, and they stay RAW there: the OOB zero-selects consume
            // A_pf/Wpk_pf only at publication after compute, behind an opaque
            // register pin, so no loadcnt wait sits between the future loads
            // and the current slab's first WMMA. Expand + LDS publish of s+1
            // still follow compute (single buffer); only latency, not
            // execution, is hidden. A_pf holds the lane's raw b64 chunk pairs
            // (A_ROUNDS*2 v2i_t); Wpk_pf holds raw packs (mask + expand at
            // publish, identical values).
            v2i_t A_pf[HIPFIRE_FP8_V2_A_ROUNDS * 2];
            unsigned int Wpk_pf[HIPFIRE_FP8_V2_W_ROUNDS * 2];
            // Prologue: stage slab 0 fully (S/SZ + A + W). Every lane is
            // active on every round; OOB clamps to the last valid index with
            // a branch-free zero-select (exact old ternary semantics).
            {
#if !HIPFIRE_FP8_SYMFOLD
                // S: one f32 per lane.
                {
                    const unsigned int s_idx = st_soff + (unsigned int)g * 2u + (unsigned int)h;
                    // Unconditional load from the clamped address; the select
                    // on the value is a cndmask, never an EXEC predicate.
                    const float s_ld = X_half_sums[s_idx];
                    const float s_v = st_sok ? s_ld : 0.0f;
                    *(float*)(lds_S + st_slds) = s_v;
                }
#endif
                // SZ: scale only for symmetric, scale+zero otherwise.
                {
#if HIPFIRE_FP8_WPRESHUFFLE
                    const unsigned int half_block = (unsigned int)g * 2u + (unsigned int)h;
#else
                    const unsigned int hwoff = (unsigned int)g * 136u + (unsigned int)h * 4u;
#endif
                    #pragma unroll
                    for (int e = 0; e < HIPFIRE_FP8_V2_SZITER; e++) {
#if HIPFIRE_FP8_WPRESHUFFLE
                        const unsigned char* hp = (const unsigned char*)st_szbase[e]
                            + (size_t)half_block * st_szhstride[e];
                        const unsigned short h16 =
                            (unsigned short)hp[16] | ((unsigned short)hp[0] << 8);
#elif HIPFIRE_FP8_SYMFOLD
                        const unsigned short h16 = *(const unsigned short*)(st_szbase[e] + hwoff);
#else
                        const unsigned int hw = *(const unsigned int*)(st_szbase[e] + hwoff);
                        const unsigned short h16 = (unsigned short)(st_szword[e] ? (hw >> 16) : (hw & 0xFFFFu));
#endif
                        const float f = __half2float(__ushort_as_half(h16));
                        *(float*)(lds_SZ + st_szlds[e]) = st_szok[e] ? f : 0.0f;
                    }
                }
                // A: one b64 chunk pair per round (even-chunk 16 B alignment
                // lets the compiler fuse each pair's loads to one b128).
                {
                    const unsigned char* xbase = X_fp8 + (unsigned int)g * 256u + (unsigned int)h * 128u;
                    #pragma unroll
                    for (int r = 0; r < HIPFIRE_FP8_V2_A_ROUNDS; r++) {
                        const unsigned char* asrc = xbase + st_agoff[r];
                        const v2i_t v0l = *(const v2i_t*)asrc;
                        const v2i_t v1l = *(const v2i_t*)(asrc + 8u);
                        const v2i_t v0 = st_aok[r] ? v0l : st_z64;
                        const v2i_t v1 = st_aok[r] ? v1l : st_z64;
                        char* adst = lds_A +
#if HIPFIRE_FP8_V2_FRAG_ORDER
                            st_ab0[r];
                        *(v2i_t*)adst = v0;
                        *(v2i_t*)(adst + 128u) = v1;
#else
                            st_aldsoff[r];
                        *(v2i_t*)adst = v0;
                        *(v2i_t*)(adst + 8u) = v1;
#endif
                    }
                }
                // W: two raw packs per round; expand immediately (bit-exact).
                {
#if HIPFIRE_FP8_WPRESHUFFLE
                    const unsigned int pack_base =
                        ((unsigned int)g * 16u + (unsigned int)h * 8u) * 128u;
#else
                    const unsigned int pack_base = (unsigned int)g * 136u + 8u
                        + (unsigned int)h * 64u;
#endif
                    #pragma unroll
                    for (int r = 0; r < HIPFIRE_FP8_V2_W_ROUNDS; r++) {
                        #pragma unroll
                        for (int pp = 0; pp < 2; pp++) {
                            const unsigned int pk = *(const unsigned int*)(st_wbase[r] + pack_base + st_wpoff[r][pp]);
                            const v2i_t e8 = mq4_qpk_to_e4m3(st_wok[r] ? pk : 0u);
                            *(v2i_t*)(lds_W +
#if HIPFIRE_FP8_V2_FRAG_ORDER
                                st_wslot[r][pp]) = e8;
#else
                                st_wldsoff[r][pp]) = e8;
#endif
                        }
                    }
                }
            }
            __syncthreads();
            #pragma nounroll
            for (int slab = 0; slab < HIPFIRE_FP8_V2_NSLABS; slab++) {
                // Prefetch slab s+1 globals before compute s.
                if (slab + 1 < HIPFIRE_FP8_V2_NSLABS) {
                    {
                        const unsigned char* xbase = X_fp8 + (unsigned int)g * 256u + (unsigned int)h * 128u
                            + (unsigned int)(slab + 1) * (unsigned int)HIPFIRE_FP8_V2_BK;
                        #pragma unroll
                        for (int r = 0; r < HIPFIRE_FP8_V2_A_ROUNDS; r++) {
                            const unsigned char* asrc = xbase + st_agoff[r];
                            const v2i_t v0l = *(const v2i_t*)asrc;
                            const v2i_t v1l = *(const v2i_t*)(asrc + 8u);
                            A_pf[r * 2] = v0l;
                            A_pf[r * 2 + 1] = v1l;
                        }
                    }
                    {
#if HIPFIRE_FP8_WPRESHUFFLE
                        const unsigned int pack_base =
                            ((unsigned int)g * 16u + (unsigned int)h * 8u
                                + (unsigned int)(slab + 1) * 4u) * 128u;
#else
                        const unsigned int pack_base = (unsigned int)g * 136u + 8u
                            + (unsigned int)h * 64u
                            + (unsigned int)(slab + 1) * (unsigned int)(HIPFIRE_FP8_V2_BK / 2);
#endif
                        #pragma unroll
                        for (int r = 0; r < HIPFIRE_FP8_V2_W_ROUNDS; r++) {
                            #pragma unroll
                            for (int pp = 0; pp < 2; pp++) {
                                const unsigned int pk = *(const unsigned int*)(st_wbase[r] + pack_base + st_wpoff[r][pp]);
                                // Raw bits only (R1): zero-select moves to publication.
                                Wpk_pf[r * 2 + pp] = pk;
                            }
                        }
                    }
                    // No sched pin here (iu4fold v0): the previous trip's fold
                    // VALU must be free to sink among this trip's WMMAs, and
                    // the prefetch loads hoist for latency on their own. True
                    // deps (A_pf/Wpk_pf publish after compute) still order
                    // what must be ordered.
                }

                // Inner loop: ds_read_b64 A/W -> v_wmma fp8 16x16x16. Nothing else.
                if (slab == 0) {
                    {
#if HIPFIRE_FP8_V2_FRAG_ORDER
                        // Frag immediates off a_fptr/w_fptr; no address VALU.
                        const v2i_t w0 = *(const v2i_t*)(w_fptr + HIPFIRE_FP8_V2_WFRAG(0, 0));
                        const v2i_t w1 = *(const v2i_t*)(w_fptr + HIPFIRE_FP8_V2_WFRAG(1, 0));
                        #pragma unroll
                        for (int nb = 0; nb < 4; nb++) {
                            const v2i_t a = *(const v2i_t*)(a_fptr + HIPFIRE_FP8_V2_AFRAG(nb, 0));
#else
                        const int off = kg * 8;
                        const v2i_t w0 = *(const v2i_t*)(lds_W + (wave_wt + ml) * HIPFIRE_FP8_V2_STRIDE + off);
                        const v2i_t w1 = *(const v2i_t*)(lds_W + (wave_wt + 16 + ml) * HIPFIRE_FP8_V2_STRIDE + off);
                        #pragma unroll
                        for (int nb = 0; nb < 4; nb++) {
                            const v2i_t a = *(const v2i_t*)(lds_A + (wave_tok + nb * 16 + ml) * HIPFIRE_FP8_V2_STRIDE + off);
#endif
                            P[nb][0] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(w0, a, zc);
                            P[nb][1] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(w1, a, zc);
                        }
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    #pragma unroll
                    for (int kstep = 1; kstep < HIPFIRE_FP8_V2_K16; kstep++) {
#if HIPFIRE_FP8_V2_FRAG_ORDER
                        const v2i_t w0 = *(const v2i_t*)(w_fptr + HIPFIRE_FP8_V2_WFRAG(0, kstep));
                        const v2i_t w1 = *(const v2i_t*)(w_fptr + HIPFIRE_FP8_V2_WFRAG(1, kstep));
                        #pragma unroll
                        for (int nb = 0; nb < 4; nb++) {
                            const v2i_t a = *(const v2i_t*)(a_fptr + HIPFIRE_FP8_V2_AFRAG(nb, kstep));
#else
                        const int off = kstep * 16 + kg * 8;
                        const v2i_t w0 = *(const v2i_t*)(lds_W + (wave_wt + ml) * HIPFIRE_FP8_V2_STRIDE + off);
                        const v2i_t w1 = *(const v2i_t*)(lds_W + (wave_wt + 16 + ml) * HIPFIRE_FP8_V2_STRIDE + off);
                        #pragma unroll
                        for (int nb = 0; nb < 4; nb++) {
                            const v2i_t a = *(const v2i_t*)(lds_A + (wave_tok + nb * 16 + ml) * HIPFIRE_FP8_V2_STRIDE + off);
#endif
                            P[nb][0] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(w0, a, P[nb][0]);
                            P[nb][1] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(w1, a, P[nb][1]);
                        }
                        __builtin_amdgcn_sched_barrier(0);
                    }
                } else {
                    #pragma unroll
                    for (int kstep = 0; kstep < HIPFIRE_FP8_V2_K16; kstep++) {
#if HIPFIRE_FP8_V2_FRAG_ORDER
                        const v2i_t w0 = *(const v2i_t*)(w_fptr + HIPFIRE_FP8_V2_WFRAG(0, kstep));
                        const v2i_t w1 = *(const v2i_t*)(w_fptr + HIPFIRE_FP8_V2_WFRAG(1, kstep));
                        #pragma unroll
                        for (int nb = 0; nb < 4; nb++) {
                            const v2i_t a = *(const v2i_t*)(a_fptr + HIPFIRE_FP8_V2_AFRAG(nb, kstep));
#else
                        const int off = kstep * 16 + kg * 8;
                        const v2i_t w0 = *(const v2i_t*)(lds_W + (wave_wt + ml) * HIPFIRE_FP8_V2_STRIDE + off);
                        const v2i_t w1 = *(const v2i_t*)(lds_W + (wave_wt + 16 + ml) * HIPFIRE_FP8_V2_STRIDE + off);
                        #pragma unroll
                        for (int nb = 0; nb < 4; nb++) {
                            const v2i_t a = *(const v2i_t*)(lds_A + (wave_tok + nb * 16 + ml) * HIPFIRE_FP8_V2_STRIDE + off);
#endif
                            P[nb][0] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(w0, a, P[nb][0]);
                            P[nb][1] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(w1, a, P[nb][1]);
                        }
                        __builtin_amdgcn_sched_barrier(0);
                    }
                }
                // R1 pin: opaque register-only identity on the raw future
                // slab. The zero-selects below must consume A_pf/Wpk_pf only
                // here, after current-slab compute; without this the compiler
                // hoists the cndmasks (and their loadcnt waits) above the
                // current slab's first WMMA. No memory effect, no fence.
                if (slab + 1 < HIPFIRE_FP8_V2_NSLABS) {
                    #pragma unroll
                    for (int r = 0; r < HIPFIRE_FP8_V2_A_ROUNDS * 2; r++) {
                        asm volatile("" : "+v"(A_pf[r][0]), "+v"(A_pf[r][1]));
                    }
                    #pragma unroll
                    for (int r = 0; r < HIPFIRE_FP8_V2_W_ROUNDS * 2; r++) {
                        asm volatile("" : "+v"(Wpk_pf[r]));
                    }
                }
                // Publish prefetched slab s+1 (buffer free: compute s done).
                // The barrier is REQUIRED: without it fast waves overwrite
                // slab s while stragglers still read it (K2 race found).
                // Single buffer: slab selects the GLOBAL source only; the LDS
#if HIPFIRE_FP8_V2_FRAG_ORDER
                // address is the same frag slot the prologue used.
#else
                // address is the same slab slot the prologue used.
#endif
                if (slab + 1 < HIPFIRE_FP8_V2_NSLABS) {
                    __syncthreads();
                    #pragma unroll
                    for (int r = 0; r < HIPFIRE_FP8_V2_A_ROUNDS; r++) {
                        char* adst = lds_A +
#if HIPFIRE_FP8_V2_FRAG_ORDER
                            st_ab0[r];
                        *(v2i_t*)adst = st_aok[r] ? A_pf[r * 2] : st_z64;
                        *(v2i_t*)(adst + 128u) = st_aok[r] ? A_pf[r * 2 + 1] : st_z64;
#else
                            st_aldsoff[r];
                        *(v2i_t*)adst = st_aok[r] ? A_pf[r * 2] : st_z64;
                        *(v2i_t*)(adst + 8u) = st_aok[r] ? A_pf[r * 2 + 1] : st_z64;
#endif
                    }
                    {
                        const unsigned int pack_base = (unsigned int)g * 136u + 8u
                            + (unsigned int)h * 64u
                            + (unsigned int)(slab + 1) * (unsigned int)(HIPFIRE_FP8_V2_BK / 2);
                        #pragma unroll
                        for (int r = 0; r < HIPFIRE_FP8_V2_W_ROUNDS; r++) {
                            #pragma unroll
                            for (int pp = 0; pp < 2; pp++) {
                                // Raw Wpk zero-selected here at publication;
                                // expand(0) is all-zero E4M3 (bit-identical).
                                const v2i_t e8 = mq4_qpk_to_e4m3(st_wok[r] ? Wpk_pf[r * 2 + pp] : 0u);
                                *(v2i_t*)(lds_W +
#if HIPFIRE_FP8_V2_FRAG_ORDER
                                    st_wslot[r][pp]) = e8;
#else
                                    st_wldsoff[r][pp]) = e8;
#endif
                            }
                        }
                    }
                }

                if (slab == HIPFIRE_FP8_V2_NSLABS - 1) {
#if HIPFIRE_FP8_SYMFOLD
                    // q was centered to q-8 before WMMA, so z=-8*d is already
                    // represented in P. Fold one scale FMA per half-group and
                    // output; no activation sum or zero metadata participates.
                    #pragma unroll
                    for (int wf = 0; wf < 2; wf++) {
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            const int grow = wave_wt + wf * 16 + 8 * kg + j;
                            const float s_h = *(const float*)(lds_SZ + (unsigned int)grow * 4u);
                            #pragma unroll
                            for (int nb = 0; nb < 4; nb++)
                                T[nb][wf][j] = fmaf(s_h, P[nb][wf][j], T[nb][wf][j]);
                        }
                    }
#else
                    // Fold at the 128-K half boundary. Stream s/z; S per token frag.
                    float Sh[4];
                    #pragma unroll
                    for (int nb = 0; nb < 4; nb++) {
                        Sh[nb] = *(const float*)(lds_S + (unsigned int)(wave_tok + nb * 16 + ml) * 4u);
                    }
                    #pragma unroll
                    for (int wf = 0; wf < 2; wf++) {
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            const int grow = wave_wt + wf * 16 + 8 * kg + j;
                            const float* szp = (const float*)(lds_SZ + (unsigned int)grow * 8u);
                            const float s_h = szp[0];
                            const float z_h = szp[1];
                            #pragma unroll
                            for (int nb = 0; nb < 4; nb++) {
                                T[nb][wf][j] = fmaf(s_h, P[nb][wf][j], T[nb][wf][j]);
                                T[nb][wf][j] = fmaf(z_h, Sh[nb], T[nb][wf][j]);
                            }
                        }
                    }
#endif
                }

                __syncthreads();
            }
        }
    }

    // Coalesced store: the WMMA C layout leaves lane (ml,kg) holding
    // T[nb][wf][j] for token wave_tok+nb*16+ml and rows
    // wave_wt+wf*16+8*kg+j, so one store instruction touches 16 output rows
    // at 4 B each (~90 GB/s effective). Transpose each nb's 16x32 wave tile
    // through LDS instead: lane (ml,kg) dumps its 16 values at tile row ml,
    // cols wf*16+8*kg+j (the kg pair covers all 32 columns), then all 32
    // lanes cooperatively stream the 16 token rows with lane tid owning
    // column tid, so every global store is one contiguous 128 B segment.
    // Per-element arithmetic is unchanged (y = a[t]*T, one residual add),
    // hence bit-identical; only the visit order changes. A/W/S/SZ are dead
    // (all folds complete), so the tile reuses LDS base: 8 waves x 16 x 32
    // floats = 16 KiB < 24 KiB, no host change.
    float* lds_tile = (float*)lds;
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        #pragma unroll
        for (int wf = 0; wf < 2; wf++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                // Row-skew (col+ml)&31: without it all 16 ml lanes hit bank
                // col%32 on the dump (16-way conflict). Drain undoes it with
                // (lane+r16)&31, a permutation of the same 128 B line.
                const int col = (wf * 16 + 8 * kg + j + ml) & 31;
                lds_tile[(unsigned int)(wave * 512 + ml * 32 + col)] = T[nb][wf][j];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int r16 = 0; r16 < 16; r16++) {
            const int t = bs + wave_tok + nb * 16 + r16;
            if (t < N) {
                const float at = X_row_scales[t];
                const int r = rs + wave_wt + lane;
                const float v = lds_tile[(unsigned int)(wave * 512 + r16 * 32 + ((lane + r16) & 31))];
                const float y = at * v;
                if (r < tm) {
#if HIPFIRE_FP8_QKVZA
                    float* Yp;
                    int pr, os;
                    if (r < qkv_m) { Yp = Y_qkv; pr = r; os = qkv_m; }
                    else if (r < qkv_m + z_m) { Yp = Y_z; pr = r - qkv_m; os = z_m; }
                    else if (r < qkv_m + z_m + beta_m) { Yp = Y_beta; pr = r - (qkv_m + z_m); os = beta_m; }
                    else { Yp = Y_alpha; pr = r - (qkv_m + z_m + beta_m); os = alpha_m; }
                    Yp[(unsigned int)t * (unsigned int)os + (unsigned int)pr] = y;
#elif HIPFIRE_FP8_RESIDUAL
                    Y_res[(unsigned int)t * (unsigned int)res_m + (unsigned int)r] += y;
#elif HIPFIRE_FP8_QKV
                    float* Yp;
                    int pr, os;
                    if (r < q_m) { Yp = Y_q; pr = r; os = q_m; }
                    else if (r < q_m + k_m) { Yp = Y_k; pr = r - q_m; os = k_m; }
                    else { Yp = Y_v; pr = r - (q_m + k_m); os = v_m; }
                    Yp[(unsigned int)t * (unsigned int)os + (unsigned int)pr] = y;
#else
                    float* Yp;
                    int pr, os;
                    if (r < gate_m) { Yp = Y_gate; pr = r; os = gate_m; }
                    else { Yp = Y_up; pr = r - gate_m; os = up_m; }
                    Yp[(unsigned int)t * (unsigned int)os + (unsigned int)pr] = y;
#endif
                }
            }
        }
        __syncthreads();
    }
}

#else
extern "C" __launch_bounds__(32, 2) __global__
#if HIPFIRE_FP8_QKVZA
void HIPFIRE_FP8_GATEUP_KERNEL(
    const char* A_qkv,
    const char* A_z,
    const char* A_beta,
    const char* A_alpha,
    const unsigned char* X_fp8,
    const float* X_half_sums,
    const float* X_row_scales,
    float* Y_qkv,
    float* Y_z,
    float* Y_beta,
    float* Y_alpha,
    int qkv_m,
    int z_m,
    int beta_m,
    int alpha_m,
    int K,
    int N
#ifdef HIPFIRE_FP8_LUT_ARG
    , unsigned int LQ0, unsigned int LQ1, unsigned int LQ2, unsigned int LQ3,
    unsigned int LZ0, unsigned int LZ1, unsigned int LZ2, unsigned int LZ3,
    unsigned int LB0, unsigned int LB1, unsigned int LB2, unsigned int LB3,
    unsigned int LA0, unsigned int LA1, unsigned int LA2, unsigned int LA3
#endif
) {
    const int tm = qkv_m + z_m + beta_m + alpha_m;
#elif HIPFIRE_FP8_RESIDUAL
void HIPFIRE_FP8_GATEUP_KERNEL(
    const char* A_w,
    const unsigned char* X_fp8,
    const float* X_half_sums,
    const float* X_row_scales,
    float* Y_res,
    int res_m,
    int K,
    int N
#ifdef HIPFIRE_FP8_LUT_ARG
    , unsigned int LC0, unsigned int LC1, unsigned int LC2, unsigned int LC3
#endif
) {
    const int tm = res_m;
#elif HIPFIRE_FP8_QKV
void HIPFIRE_FP8_GATEUP_KERNEL(
    const char* A_q,
    const char* A_k,
    const char* A_v,
    const unsigned char* X_fp8,
    const float* X_half_sums,
    const float* X_row_scales,
    float* Y_q,
    float* Y_k,
    float* Y_v,
    int q_m,
    int k_m,
    int v_m,
    int K,
    int N
#ifdef HIPFIRE_FP8_LUT_ARG
    , unsigned int LQ0, unsigned int LQ1, unsigned int LQ2, unsigned int LQ3,
    unsigned int LK0, unsigned int LK1, unsigned int LK2, unsigned int LK3,
    unsigned int LV0, unsigned int LV1, unsigned int LV2, unsigned int LV3
#endif
) {
    const int tm = q_m + k_m + v_m;
#else
void HIPFIRE_FP8_GATEUP_KERNEL(
    const char* A_gate,
    const char* A_up,
    const unsigned char* X_fp8,
    const float* X_half_sums,
    const float* X_row_scales,
    float* Y_gate,
    float* Y_up,
    int gate_m,
    int up_m,
    int K,
    int N
#ifdef HIPFIRE_FP8_LUT_ARG
    , unsigned int LG0, unsigned int LG1, unsigned int LG2, unsigned int LG3,
    unsigned int LU0, unsigned int LU1, unsigned int LU2, unsigned int LU3
#endif
) {
    const int tm = gate_m + up_m;
#endif
    // SLABS=2 waves own two 16-row slabs; SLABS=1 keeps the current grid.
#if HIPFIRE_FP8_SLABS == 2
    const int rs = blockIdx.x * 32;
#else
    const int rs = blockIdx.x * 16;
#endif
    const int bs = blockIdx.y * (16 * HIPFIRE_FP8_BV);
    const int tid = threadIdx.x;
    if (rs >= tm || bs >= N) return;

    const int ml = tid & 15;
    const int kg = tid >> 4;

#if HIPFIRE_FP8_SLABS == 2
    // Two-slab routing: one A-row per slab (own header source each).
    const int sr0 = ((rs + ml) < tm) ? (rs + ml) : (tm - 1);
    const int sr1 = ((rs + 16 + ml) < tm) ? (rs + 16 + ml) : (tm - 1);
    const int srx[2] = { sr0, sr1 };
    const char* Ax[2];
    int lrx[2];
    #pragma unroll
    for (int s = 0; s < 2; s++) {
        const int srs = srx[s];
#if HIPFIRE_FP8_QKVZA
        if (srs < qkv_m) { Ax[s] = A_qkv; lrx[s] = srs; }
        else if (srs < qkv_m + z_m) { Ax[s] = A_z; lrx[s] = srs - qkv_m; }
        else if (srs < qkv_m + z_m + beta_m) { Ax[s] = A_beta; lrx[s] = srs - (qkv_m + z_m); }
        else { Ax[s] = A_alpha; lrx[s] = srs - (qkv_m + z_m + beta_m); }
#elif HIPFIRE_FP8_RESIDUAL
        Ax[s] = A_w;
        lrx[s] = srs;
#elif HIPFIRE_FP8_QKV
        if (srs < q_m) { Ax[s] = A_q; lrx[s] = srs; }
        else if (srs < q_m + k_m) { Ax[s] = A_k; lrx[s] = srs - q_m; }
        else { Ax[s] = A_v; lrx[s] = srs - (q_m + k_m); }
#else
        if (srs < gate_m) { Ax[s] = A_gate; lrx[s] = srs; } else { Ax[s] = A_up; lrx[s] = srs - gate_m; }
#endif
    }
#ifdef HIPFIRE_FP8_LUT_ARG
    // Per-slab source-tensor LUT select, loop-invariant. All four arms select
    // per 16-row slab (residual trivially copies the single LC* into SLC*).
    // The select keys on the SLAB START ROW (wave-uniform, from blockIdx), not
    // the lane row, so the four dwords resolve to SGPRs and never occupy VGPRs
    // across the K loop. Host contract: every source m is a multiple of 16, so
    // no 16-row slab straddles two sources (the launcher refuses otherwise).
#if HIPFIRE_FP8_RESIDUAL
    unsigned int SLC0[2], SLC1[2], SLC2[2], SLC3[2];
    #pragma unroll
    for (int s = 0; s < 2; s++) {
        SLC0[s]=LC0; SLC1[s]=LC1; SLC2[s]=LC2; SLC3[s]=LC3;
    }
#elif HIPFIRE_FP8_QKVZA
    // Four-tensor fan-out: candidate evaluation costs more VGPRs than a
    // copy, so QKVZA selects one LUT per slab. The select keys on the SLAB
    // START ROW (wave-uniform, from blockIdx), not the lane row, so the four
    // dwords resolve to SGPRs and never occupy VGPRs across the K loop. Host
    // contract: qkv_m, z_m, beta_m, alpha_m are all multiples of 16, so no
    // 16-row slab straddles two sources (the launcher refuses otherwise).
    unsigned int SLC0[2], SLC1[2], SLC2[2], SLC3[2];
    #pragma unroll
    for (int s = 0; s < 2; s++) {
        const int srs = rs + 16 * s;
        if (srs < qkv_m) { SLC0[s]=LQ0; SLC1[s]=LQ1; SLC2[s]=LQ2; SLC3[s]=LQ3; }
        else if (srs < qkv_m + z_m) { SLC0[s]=LZ0; SLC1[s]=LZ1; SLC2[s]=LZ2; SLC3[s]=LZ3; }
        else if (srs < qkv_m + z_m + beta_m) { SLC0[s]=LB0; SLC1[s]=LB1; SLC2[s]=LB2; SLC3[s]=LB3; }
        else { SLC0[s]=LA0; SLC1[s]=LA1; SLC2[s]=LA2; SLC3[s]=LA3; }
    }
#elif HIPFIRE_FP8_QKV
    unsigned int SLC0[2], SLC1[2], SLC2[2], SLC3[2];
    #pragma unroll
    for (int s = 0; s < 2; s++) {
        const int srs = rs + 16 * s;
        if (srs < q_m) { SLC0[s]=LQ0; SLC1[s]=LQ1; SLC2[s]=LQ2; SLC3[s]=LQ3; }
        else if (srs < q_m + k_m) { SLC0[s]=LK0; SLC1[s]=LK1; SLC2[s]=LK2; SLC3[s]=LK3; }
        else { SLC0[s]=LV0; SLC1[s]=LV1; SLC2[s]=LV2; SLC3[s]=LV3; }
    }
#else
    unsigned int SLC0[2], SLC1[2], SLC2[2], SLC3[2];
    #pragma unroll
    for (int s = 0; s < 2; s++) {
        const int srs = rs + 16 * s;
        if (srs < gate_m) { SLC0[s]=LG0; SLC1[s]=LG1; SLC2[s]=LG2; SLC3[s]=LG3; }
        else { SLC0[s]=LU0; SLC1[s]=LU1; SLC2[s]=LU2; SLC3[s]=LU3; }
    }
#endif
#endif
#else
    // Route this lane's A-row (own header source).
    const int mr = rs + ml;
    const int sr = (mr < tm) ? mr : (tm - 1);
#if HIPFIRE_FP8_QKVZA
    const char* A;
    int lr;
    if (sr < qkv_m) { A = A_qkv; lr = sr; }
    else if (sr < qkv_m + z_m) { A = A_z; lr = sr - qkv_m; }
    else if (sr < qkv_m + z_m + beta_m) { A = A_beta; lr = sr - (qkv_m + z_m); }
    else { A = A_alpha; lr = sr - (qkv_m + z_m + beta_m); }
#elif HIPFIRE_FP8_RESIDUAL
    const char* A = A_w;
    const int lr = sr;
#elif HIPFIRE_FP8_QKV
    const char* A;
    int lr;
    if (sr < q_m) { A = A_q; lr = sr; }
    else if (sr < q_m + k_m) { A = A_k; lr = sr - q_m; }
    else { A = A_v; lr = sr - (q_m + k_m); }
#else
    const char* A;
    int lr;
    if (sr < gate_m) { A = A_gate; lr = sr; } else { A = A_up; lr = sr - gate_m; }
#endif
#ifdef HIPFIRE_FP8_LUT_ARG
    // Per-tile source-tensor LUT select, loop-invariant. All four arms select
    // one LUT for the 16-row tile (residual uses LC* args directly). The
    // select keys on the TILE START ROW (wave-uniform) so the dwords stay
    // SGPR. Host contract: every source m is a multiple of 16 (no tile
    // straddles two sources); see SLABS=2 twin.
#if HIPFIRE_FP8_RESIDUAL
    // Single LC* args used directly at the decode sites (no copy).
#elif HIPFIRE_FP8_QKVZA
    // Four-tensor fan-out: one LUT per 16-row tile, keyed on the TILE START
    // ROW (wave-uniform) so the dwords stay SGPR. Host contract: all four m
    // are multiples of 16 (no tile straddles two sources); see SLABS=2 twin.
    unsigned int LC0, LC1, LC2, LC3;
    if (rs < qkv_m) { LC0=LQ0; LC1=LQ1; LC2=LQ2; LC3=LQ3; }
    else if (rs < qkv_m + z_m) { LC0=LZ0; LC1=LZ1; LC2=LZ2; LC3=LZ3; }
    else if (rs < qkv_m + z_m + beta_m) { LC0=LB0; LC1=LB1; LC2=LB2; LC3=LB3; }
    else { LC0=LA0; LC1=LA1; LC2=LA2; LC3=LA3; }
#elif HIPFIRE_FP8_QKV
    unsigned int LC0, LC1, LC2, LC3;
    if (rs < q_m) { LC0=LQ0; LC1=LQ1; LC2=LQ2; LC3=LQ3; }
    else if (rs < q_m + k_m) { LC0=LK0; LC1=LK1; LC2=LK2; LC3=LK3; }
    else { LC0=LV0; LC1=LV1; LC2=LV2; LC3=LV3; }
#else
    unsigned int LC0, LC1, LC2, LC3;
    if (rs < gate_m) { LC0=LG0; LC1=LG1; LC2=LG2; LC3=LG3; }
    else { LC0=LU0; LC1=LU1; LC2=LU2; LC3=LU3; }
#endif
#endif
#endif

    const int gpr = K / 256;
#if HIPFIRE_FP8_SLABS == 2
    // Per-slab weight-row bases; 32-bit offsets per the note below.
    const char* rbx[2] = { Ax[0] + (unsigned int)lrx[0] * (unsigned int)gpr * 136u,
                           Ax[1] + (unsigned int)lrx[1] * (unsigned int)gpr * 136u };
#else
    // All byte offsets below fit in 32 bits: X holds N*K bytes with
    // N*K < 2^32, each weight row holds gpr*136 bytes, and Y indices stay
    // far below 2^32 at admitted shapes. This keeps hoisted per-chain
    // products to one VGPR each instead of 64-bit pairs.
    const char* rb = A + (unsigned int)lr * (unsigned int)gpr * 136u;
#endif

    // Batch rows for the 12 chains; out-of-range rows alias row 0 for
    // safe loads, stores stay guarded.
    int sb[HIPFIRE_FP8_BV];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_FP8_BV; b++) {
        const int t = bs + b * 16 + ml;
        sb[b] = (t < N) ? t : 0;
    }

#if HIPFIRE_FP8_SLABS == 2
    // Two long-lived accumulators per chain (one per slab), each in units
    // of its slab's current half scale; per-slab current scale below.
    v8f_t acc[2][HIPFIRE_FP8_BV];
    #pragma unroll
    for (int s = 0; s < 2; s++)
        #pragma unroll
        for (int b = 0; b < HIPFIRE_FP8_BV; b++)
            acc[s][b] = v8f_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // Lane-owned (row ml) current scale per slab; 0 until first nonzero half.
    float s_cur_x[2] = { 0.0f, 0.0f };
#else
    // Single long-lived accumulator per chain, kept in units of the
    // CURRENT half's per-row scale s_cur (row = 8*kg+j for element j).
    // At each half boundary: acc *= s_prev/s_new, then the zero-point term
    // is added as (z_h/s_h)*S_b, and the WMMAs accumulate raw q.x directly.
    // This removes the 12 partial chains and the LDS accumulator; the cost
    // is one f32 multiply per element per half (~6e-8 relative), far below
    // the E4M3 activation rounding this path already accepts.
    v8f_t acc[HIPFIRE_FP8_BV];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_FP8_BV; b++)
        acc[b] = v8f_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // Lane-owned (row ml) current scale; 0 until the first nonzero half.
    float s_cur_self = 0.0f;
#endif
    const unsigned char* xb_base[HIPFIRE_FP8_BV];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_FP8_BV; b++) xb_base[b] = X_fp8 + (unsigned int)sb[b] * (unsigned int)K;

#if HIPFIRE_FP8_SLABS == 2
    // Two-slab K loop: per-slab headers from rbx[s], shared half-sums S0/S1
    // and X base. Per (group, half): per-slab boundary correction into
    // acc[s][b], then per 32-K pair decode one weight fragment per slab
    // (same kt) and issue 4 WMMAs per chain off ONE X fragment pair.
    #pragma nounroll
    for (int g = 0; g < gpr; g++) {
        float sc0x[2], zp0x[2], sc1x[2], zp1x[2];
        const char* payx[2];
        #pragma unroll
        for (int s = 0; s < 2; s++) {
            const char* gpx = rbx[s] + (unsigned int)g * 136u;
            const unsigned int hAx = *(const unsigned int*)(gpx);
            const unsigned int hBx = *(const unsigned int*)(gpx + 4);
            sc0x[s] = __half2float(__ushort_as_half((unsigned short)(hAx & 0xFFFFu)));
            zp0x[s] = __half2float(__ushort_as_half((unsigned short)(hAx >> 16)));
            sc1x[s] = __half2float(__ushort_as_half((unsigned short)(hBx & 0xFFFFu)));
            zp1x[s] = __half2float(__ushort_as_half((unsigned short)(hBx >> 16)));
            payx[s] = gpx + 8;
        }

        // Half-sums for both halves of this group (shared by both slabs).
        float S0[HIPFIRE_FP8_BV], S1[HIPFIRE_FP8_BV];
        #pragma unroll
        for (int b = 0; b < HIPFIRE_FP8_BV; b++) {
            const float2 s2 = *(const float2*)(X_half_sums + ((unsigned int)sb[b] * (unsigned int)gpr + (unsigned int)g) * 2u);
            S0[b] = s2.x; S1[b] = s2.y;
        }

        #pragma nounroll
        for (int h = 0; h < 2; h++) {
            // Per-slab boundary state; a zero scale zeroes that slab's
            // weight operand below while keeping its previous units.
            float rx[2], zqx[2];
            bool livex[2];
            #pragma unroll
            for (int s = 0; s < 2; s++) {
                const float s_new = (h == 0) ? sc0x[s] : sc1x[s];
                const float zx = (h == 0) ? zp0x[s] : zp1x[s];
                const bool live = (s_new != 0.0f);
                const float s_next = live ? s_new : s_cur_x[s];
                rx[s] = (live && s_cur_x[s] != 0.0f) ? (s_cur_x[s] / s_next) : 1.0f;
                zqx[s] = (s_next != 0.0f) ? (zx / s_next) : 0.0f;
                s_cur_x[s] = s_next;
                livex[s] = live;
            }

            // Rescale into the new half's units and add the zero-point term
            // in those units, per slab.
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const float r0 = __shfl(rx[0], 8 * kg + j);
                const float r1 = __shfl(rx[1], 8 * kg + j);
                const float z0 = __shfl(zqx[0], 8 * kg + j);
                const float z1 = __shfl(zqx[1], 8 * kg + j);
                #pragma unroll
                for (int b = 0; b < HIPFIRE_FP8_BV; b++) {
                    const float Sb = (h == 0) ? S0[b] : S1[b];
                    acc[0][b][j] = fmaf(z0, Sb, acc[0][b][j] * r0);
                    acc[1][b][j] = fmaf(z1, Sb, acc[1][b][j] * r1);
                }
            }

            // Same K re-slicing as SLABS=1, decoded once per slab; one 16 B
            // X load per chain feeds four WMMAs (two slabs x two fragments).
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                const uint2 pk0 = *(const uint2*)(payx[0] + h * 64 + u * 16 + kg * 8);
                const uint2 pk1 = *(const uint2*)(payx[1] + h * 64 + u * 16 + kg * 8);
#ifdef HIPFIRE_FP8_LUT_ARG
                v2i_t q0a = mq4_qpk_to_e4m3(pk0.x, SLC0[0], SLC1[0], SLC2[0], SLC3[0]);
                v2i_t q0b = mq4_qpk_to_e4m3(pk0.y, SLC0[0], SLC1[0], SLC2[0], SLC3[0]);
                v2i_t q1a = mq4_qpk_to_e4m3(pk1.x, SLC0[1], SLC1[1], SLC2[1], SLC3[1]);
                v2i_t q1b = mq4_qpk_to_e4m3(pk1.y, SLC0[1], SLC1[1], SLC2[1], SLC3[1]);
#else
                v2i_t q0a = mq4_qpk_to_e4m3(pk0.x);
                v2i_t q0b = mq4_qpk_to_e4m3(pk0.y);
                v2i_t q1a = mq4_qpk_to_e4m3(pk1.x);
                v2i_t q1b = mq4_qpk_to_e4m3(pk1.y);
#endif
                if (!livex[0]) { q0a[0] = 0; q0a[1] = 0; q0b[0] = 0; q0b[1] = 0; }
                if (!livex[1]) { q1a[0] = 0; q1a[1] = 0; q1b[0] = 0; q1b[1] = 0; }
                const unsigned int x_off = (unsigned int)g * 256u + (unsigned int)h * 128u + (unsigned int)u * 32u + (unsigned int)kg * 16u;
                #pragma unroll
                for (int b = 0; b < HIPFIRE_FP8_BV; b++) {
                    const v2i_t xa = *(const v2i_t*)(xb_base[b] + x_off);
                    const v2i_t xb = *(const v2i_t*)(xb_base[b] + x_off + 8u);
                    acc[0][b] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(q0a, xa, acc[0][b]);
                    acc[0][b] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(q0b, xb, acc[0][b]);
                    acc[1][b] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(q1a, xa, acc[1][b]);
                    acc[1][b] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(q1b, xb, acc[1][b]);
                }
            }
        }
    }
#else
    #pragma nounroll
    for (int g = 0; g < gpr; g++) {
        const char* gp = rb + (unsigned int)g * 136u;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const float sc0 = __half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu)));
        const float zp0 = __half2float(__ushort_as_half((unsigned short)(hA >> 16)));
        const float sc1 = __half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu)));
        const float zp1 = __half2float(__ushort_as_half((unsigned short)(hB >> 16)));
        const char* payload = gp + 8;

        // Half-sums for both halves of this group: 12 x 8-byte loads.
        float S0[HIPFIRE_FP8_BV], S1[HIPFIRE_FP8_BV];
        #pragma unroll
        for (int b = 0; b < HIPFIRE_FP8_BV; b++) {
            const float2 s2 = *(const float2*)(X_half_sums + ((unsigned int)sb[b] * (unsigned int)gpr + (unsigned int)g) * 2u);
            S0[b] = s2.x; S1[b] = s2.y;
        }

        #pragma nounroll
        for (int h = 0; h < 2; h++) {
            const float s_new_self = (h == 0) ? sc0 : sc1;
            const float z_self = (h == 0) ? zp0 : zp1;
            // A zero scale means this half contributes exactly 0: keep the
            // previous units and zero the weight operand below.
            const bool live_self = (s_new_self != 0.0f);
            const float s_next_self = live_self ? s_new_self : s_cur_self;
            const float r_self = (live_self && s_cur_self != 0.0f) ? (s_cur_self / s_next_self) : 1.0f;
            const float zq_self = (s_next_self != 0.0f) ? (z_self / s_next_self) : 0.0f;
            s_cur_self = s_next_self;

            // Rescale into the new half's units and add the zero-point term
            // in those units: 16 shuffles + 96 fma.
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const float r_j = __shfl(r_self, 8 * kg + j);
                const float zq_j = __shfl(zq_self, 8 * kg + j);
                #pragma unroll
                for (int b = 0; b < HIPFIRE_FP8_BV; b++)
                    acc[b][j] = fmaf(zq_j, (h == 0) ? S0[b] : S1[b], acc[b][j] * r_j);
            }

            // K re-slicing (dot products are K-order invariant): within each
            // 32-K pair u, lane kg owns K elements [u*32+kg*16, +16) as two
            // consecutive 8-element fragments, so one 16-byte X load and one
            // 8-byte weight load feed two WMMAs.
            const char* half_base = payload + h * 64;
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                const uint2 pk2 = *(const uint2*)(half_base + u * 16 + kg * 8);
#ifdef HIPFIRE_FP8_LUT_ARG
                v2i_t q8a = mq4_qpk_to_e4m3(pk2.x, LC0, LC1, LC2, LC3);
                v2i_t q8b = mq4_qpk_to_e4m3(pk2.y, LC0, LC1, LC2, LC3);
#else
                v2i_t q8a = mq4_qpk_to_e4m3(pk2.x);
                v2i_t q8b = mq4_qpk_to_e4m3(pk2.y);
#endif
                if (!live_self) { q8a[0] = 0; q8a[1] = 0; q8b[0] = 0; q8b[1] = 0; }
                const unsigned int x_off = (unsigned int)g * 256u + (unsigned int)h * 128u + (unsigned int)u * 32u + (unsigned int)kg * 16u;
                #pragma unroll
                for (int b = 0; b < HIPFIRE_FP8_BV; b++) {
                    const v2i_t xa = *(const v2i_t*)(xb_base[b] + x_off);
                    const v2i_t xb = *(const v2i_t*)(xb_base[b] + x_off + 8u);
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(q8a, xa, acc[b]);
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(q8b, xb, acc[b]);
                }
            }
        }
    }
#endif

#if HIPFIRE_FP8_SLABS == 2
    // Per-slab final scales, gathered from the lane owning row 8*kg+j of
    // each slab.
    float s_finx[2][8];
    #pragma unroll
    for (int s = 0; s < 2; s++)
        #pragma unroll
        for (int j = 0; j < 8; j++) s_finx[s][j] = __shfl(s_cur_x[s], 8 * kg + j);

    // Final store per slab with the incumbent output mapping.
    #pragma unroll
    for (int b = 0; b < HIPFIRE_FP8_BV; b++) {
        const int oc = bs + b * 16 + ml;
        if (oc < N) {
            const float an = X_row_scales[oc];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int orow0 = rs + 8 * kg + j;
                if (orow0 < tm) {
#if HIPFIRE_FP8_QKVZA
                    float* Y0;
                    int pr0, os0;
                    if (orow0 < qkv_m) { Y0 = Y_qkv; pr0 = orow0; os0 = qkv_m; }
                    else if (orow0 < qkv_m + z_m) { Y0 = Y_z; pr0 = orow0 - qkv_m; os0 = z_m; }
                    else if (orow0 < qkv_m + z_m + beta_m) { Y0 = Y_beta; pr0 = orow0 - (qkv_m + z_m); os0 = beta_m; }
                    else { Y0 = Y_alpha; pr0 = orow0 - (qkv_m + z_m + beta_m); os0 = alpha_m; }
                    Y0[(unsigned int)oc * (unsigned int)os0 + (unsigned int)pr0] = an * s_finx[0][j] * acc[0][b][j];
#elif HIPFIRE_FP8_RESIDUAL
                    Y_res[(unsigned int)oc * (unsigned int)res_m + (unsigned int)orow0] += an * s_finx[0][j] * acc[0][b][j];
#elif HIPFIRE_FP8_QKV
                    float* Y0;
                    int pr0, os0;
                    if (orow0 < q_m) { Y0 = Y_q; pr0 = orow0; os0 = q_m; }
                    else if (orow0 < q_m + k_m) { Y0 = Y_k; pr0 = orow0 - q_m; os0 = k_m; }
                    else { Y0 = Y_v; pr0 = orow0 - (q_m + k_m); os0 = v_m; }
                    Y0[(unsigned int)oc * (unsigned int)os0 + (unsigned int)pr0] = an * s_finx[0][j] * acc[0][b][j];
#else
                    float* Y0;
                    int pr0, os0;
                    if (orow0 < gate_m) { Y0 = Y_gate; pr0 = orow0; os0 = gate_m; }
                    else { Y0 = Y_up; pr0 = orow0 - gate_m; os0 = up_m; }
                    Y0[(unsigned int)oc * (unsigned int)os0 + (unsigned int)pr0] = an * s_finx[0][j] * acc[0][b][j];
#endif
                }
                const int orow1 = rs + 16 + 8 * kg + j;
                if (orow1 < tm) {
#if HIPFIRE_FP8_QKVZA
                    float* Y1;
                    int pr1, os1;
                    if (orow1 < qkv_m) { Y1 = Y_qkv; pr1 = orow1; os1 = qkv_m; }
                    else if (orow1 < qkv_m + z_m) { Y1 = Y_z; pr1 = orow1 - qkv_m; os1 = z_m; }
                    else if (orow1 < qkv_m + z_m + beta_m) { Y1 = Y_beta; pr1 = orow1 - (qkv_m + z_m); os1 = beta_m; }
                    else { Y1 = Y_alpha; pr1 = orow1 - (qkv_m + z_m + beta_m); os1 = alpha_m; }
                    Y1[(unsigned int)oc * (unsigned int)os1 + (unsigned int)pr1] = an * s_finx[1][j] * acc[1][b][j];
#elif HIPFIRE_FP8_RESIDUAL
                    Y_res[(unsigned int)oc * (unsigned int)res_m + (unsigned int)orow1] += an * s_finx[1][j] * acc[1][b][j];
#elif HIPFIRE_FP8_QKV
                    float* Y1;
                    int pr1, os1;
                    if (orow1 < q_m) { Y1 = Y_q; pr1 = orow1; os1 = q_m; }
                    else if (orow1 < q_m + k_m) { Y1 = Y_k; pr1 = orow1 - q_m; os1 = k_m; }
                    else { Y1 = Y_v; pr1 = orow1 - (q_m + k_m); os1 = v_m; }
                    Y1[(unsigned int)oc * (unsigned int)os1 + (unsigned int)pr1] = an * s_finx[1][j] * acc[1][b][j];
#else
                    float* Y1;
                    int pr1, os1;
                    if (orow1 < gate_m) { Y1 = Y_gate; pr1 = orow1; os1 = gate_m; }
                    else { Y1 = Y_up; pr1 = orow1 - gate_m; os1 = up_m; }
                    Y1[(unsigned int)oc * (unsigned int)os1 + (unsigned int)pr1] = an * s_finx[1][j] * acc[1][b][j];
#endif
                }
            }
        }
    }
#else
    // Leave accumulator units (final per-row scale) and add the zero-point
    // term, both gathered from the lane owning row 8*kg+j.
    float s_fin[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) s_fin[j] = __shfl(s_cur_self, 8 * kg + j);

    // Final store with incumbent output mapping.
    #pragma unroll
    for (int b = 0; b < HIPFIRE_FP8_BV; b++) {
        const int oc = bs + b * 16 + ml;
        if (oc < N) {
            const float an = X_row_scales[oc];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int orow = rs + 8 * kg + j;
                if (orow < tm) {
#if HIPFIRE_FP8_QKVZA
                    float* Y;
                    int pr, os;
                    if (orow < qkv_m) { Y = Y_qkv; pr = orow; os = qkv_m; }
                    else if (orow < qkv_m + z_m) { Y = Y_z; pr = orow - qkv_m; os = z_m; }
                    else if (orow < qkv_m + z_m + beta_m) { Y = Y_beta; pr = orow - (qkv_m + z_m); os = beta_m; }
                    else { Y = Y_alpha; pr = orow - (qkv_m + z_m + beta_m); os = alpha_m; }
                    Y[(unsigned int)oc * (unsigned int)os + (unsigned int)pr] = an * s_fin[j] * acc[b][j];
#elif HIPFIRE_FP8_RESIDUAL
                    Y_res[(unsigned int)oc * (unsigned int)res_m + (unsigned int)orow] += an * s_fin[j] * acc[b][j];
#elif HIPFIRE_FP8_QKV
                    float* Y;
                    int pr, os;
                    if (orow < q_m) { Y = Y_q; pr = orow; os = q_m; }
                    else if (orow < q_m + k_m) { Y = Y_k; pr = orow - q_m; os = k_m; }
                    else { Y = Y_v; pr = orow - (q_m + k_m); os = v_m; }
                    Y[(unsigned int)oc * (unsigned int)os + (unsigned int)pr] = an * s_fin[j] * acc[b][j];
#else
                    float* Y;
                    int pr, os;
                    if (orow < gate_m) { Y = Y_gate; pr = orow; os = gate_m; }
                    else { Y = Y_up; pr = orow - gate_m; os = up_m; }
                    Y[(unsigned int)oc * (unsigned int)os + (unsigned int)pr] = an * s_fin[j] * acc[b][j];
#endif
                }
            }
        }
    }
#endif
}
#endif
