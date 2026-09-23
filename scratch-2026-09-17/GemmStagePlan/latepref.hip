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
#define HIPFIRE_FP8_V2_S_OFF    (HIPFIRE_FP8_V2_W_OFF + HIPFIRE_FP8_V2_W_BYTES)
#define HIPFIRE_FP8_V2_S_BYTES  (HIPFIRE_FP8_V2_BM * 4)
#define HIPFIRE_FP8_V2_SZ_OFF   (HIPFIRE_FP8_V2_S_OFF + HIPFIRE_FP8_V2_S_BYTES)
#define HIPFIRE_FP8_V2_SZ_BYTES (HIPFIRE_FP8_V2_BN * 8)
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
#define HIPFIRE_FP8_V2_SZITER ((HIPFIRE_FP8_V2_BN * 2 + HIPFIRE_FP8_V2_NTHREADS - 1) / HIPFIRE_FP8_V2_NTHREADS)
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
        st_wbase[r] = w_asrc + (unsigned int)w_lr * (unsigned int)gpr * 136u;
        st_wok[r] = (w_grow < tm) ? 1u : 0u;
        #pragma unroll
        for (int pp = 0; pp < 2; pp++) {
            const unsigned pk = st_quad * 2u + (unsigned)pp;
            st_wpoff[r][pp] = pk * 4u;
#if HIPFIRE_FP8_V2_FRAG_ORDER
            st_wslot[r][pp] = ((unsigned)(w_row >> 4) * (unsigned)HIPFIRE_FP8_V2_K16 + st_quad)
                * (unsigned)HIPFIRE_FP8_V2_FRAG + (unsigned)((w_row & 15) + pp * 16) * 8u;
#else
            st_wldsoff[r][pp] = (unsigned)w_row * (unsigned)HIPFIRE_FP8_V2_STRIDE + pk * 8u;
#endif
        }
    }
    // S entry: one f32 per lane (2^S_SH lanes share a row, dup benign).
    const int st_srow = tid >> HIPFIRE_FP8_V2_S_SH;
    const int st_stok = bs + st_srow;
    const int st_stok_c = (st_stok < N) ? st_stok : (N - 1);
    const unsigned st_sok = (st_stok < N) ? 1u : 0u;
    const unsigned st_soff = (unsigned)st_stok_c * (unsigned)gpr * 2u;
    const unsigned st_slds = (unsigned)st_srow * 4u;
    // SZ entries: SZITER f32 words per lane (dup benign when BN*2 < NTHREADS).
    const char* st_szbase[HIPFIRE_FP8_V2_SZITER];
    unsigned st_szok[HIPFIRE_FP8_V2_SZITER];
    unsigned st_szword[HIPFIRE_FP8_V2_SZITER];
    unsigned st_szlds[HIPFIRE_FP8_V2_SZITER];
    #pragma unroll
    for (int e = 0; e < HIPFIRE_FP8_V2_SZITER; e++) {
        const int sz_idx = tid * HIPFIRE_FP8_V2_SZITER + e;
        const int sz_row = sz_idx & (HIPFIRE_FP8_V2_BN - 1);
        const int sz_grow = rs + sz_row;
        const int sz_grow_c = (sz_grow < tm) ? sz_grow : (tm - 1);
        const char* sz_asrc;
        int sz_lr;
        HIPFIRE_FP8_V2_PICK(sz_grow_c, sz_asrc, sz_lr);
        st_szbase[e] = sz_asrc + (unsigned int)sz_lr * (unsigned int)gpr * 136u;
        st_szok[e] = (sz_grow < tm) ? 1u : 0u;
        st_szword[e] = ((unsigned)sz_idx / (unsigned)HIPFIRE_FP8_V2_BN) & 1u;
        st_szlds[e] = (unsigned)sz_row * 8u + st_szword[e] * 4u;
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
            // Overlap probe (K2 measurement): prologue stages slab 0, then
            // each iteration issues slab s+1's global loads into registers
            // BEFORE slab s's compute, so load latency hides under WMMAs.
            // Expand + LDS publish of s+1 still follow compute (single
            // buffer); only latency, not execution, is hidden. A_pf holds the
            // lane's b64 chunk pairs (A_ROUNDS*2 v2i_t); Wpk_pf holds raw
            // packs (expand at publish, identical values).
            v2i_t A_pf[HIPFIRE_FP8_V2_A_ROUNDS * 2];
            unsigned int Wpk_pf[HIPFIRE_FP8_V2_W_ROUNDS * 2];
            // Prologue: stage slab 0 fully (S/SZ + A + W). Every lane is
            // active on every round; OOB clamps to the last valid index with
            // a branch-free zero-select (exact old ternary semantics).
            {
                // S: one f32 per lane.
                {
                    const unsigned int s_idx = st_soff + (unsigned int)g * 2u + (unsigned int)h;
                    // Unconditional load from the clamped address; the select
                    // on the value is a cndmask, never an EXEC predicate.
                    const float s_ld = X_half_sums[s_idx];
                    const float s_v = st_sok ? s_ld : 0.0f;
                    *(float*)(lds_S + st_slds) = s_v;
                }
                // SZ: SZITER f32 words per lane.
                {
                    const unsigned int hwoff = (unsigned int)g * 136u + (unsigned int)h * 4u;
                    #pragma unroll
                    for (int e = 0; e < HIPFIRE_FP8_V2_SZITER; e++) {
                        const unsigned int hw = *(const unsigned int*)(st_szbase[e] + hwoff);
                        const unsigned short h16 = (unsigned short)(st_szword[e] ? (hw >> 16) : (hw & 0xFFFFu));
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
                    const unsigned int pack_base = (unsigned int)g * 136u + 8u
                        + (unsigned int)h * 64u;
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
                __builtin_amdgcn_sched_barrier(0);
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
                            A_pf[r * 2] = st_aok[r] ? v0l : st_z64;
                            A_pf[r * 2 + 1] = st_aok[r] ? v1l : st_z64;
                        }
                    }
                    {
                        const unsigned int pack_base = (unsigned int)g * 136u + 8u
                            + (unsigned int)h * 64u
                            + (unsigned int)(slab + 1) * (unsigned int)(HIPFIRE_FP8_V2_BK / 2);
                        #pragma unroll
                        for (int r = 0; r < HIPFIRE_FP8_V2_W_ROUNDS; r++) {
                            #pragma unroll
                            for (int pp = 0; pp < 2; pp++) {
                                const unsigned int pk = *(const unsigned int*)(st_wbase[r] + pack_base + st_wpoff[r][pp]);
                                Wpk_pf[r * 2 + pp] = st_wok[r] ? pk : 0u;
                            }
                        }
                    }
                    // No sched pin here (iu4fold v0): the previous trip's fold
                    // VALU must be free to sink among this trip's WMMAs, and
                    // the prefetch loads hoist for latency on their own. True
                    // deps (A_pf/Wpk_pf publish after compute) still order
                    // what must be ordered.
                }

                __builtin_amdgcn_sched_barrier(0);
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
                        *(v2i_t*)adst = A_pf[r * 2];
                        *(v2i_t*)(adst + 128u) = A_pf[r * 2 + 1];
#else
                            st_aldsoff[r];
                        *(v2i_t*)adst = A_pf[r * 2];
                        *(v2i_t*)(adst + 8u) = A_pf[r * 2 + 1];
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
                                // Wpk is already zero-selected at prefetch;
                                // expand(0) is all-zero E4M3 (bit-identical).
                                const v2i_t e8 = mq4_qpk_to_e4m3(Wpk_pf[r * 2 + pp]);
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
#endif
