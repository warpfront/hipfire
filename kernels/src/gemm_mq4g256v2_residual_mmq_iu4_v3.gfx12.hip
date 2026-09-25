// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// MQ4G256V2 iu4-direct MMQ for gfx12 (RDNA4 — gfx1200/gfx1201), v2 staged tile.
//
// Feeds MQ4V2 weight nibbles DIRECTLY to
// `__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12` (K=32 per call) with
// int4-quantized activations — no nibble expand, no per-fragment e4m3
// correction. Shares the dispatch contract / 7-arg signature with the gfx11
// iu4 kernel (`gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`) and the gfx12 iu8
// MMQ (`gemm_hfq4g256_residual_mmq.gfx12.hip`):
//   - A:  MQ4V2 packed weights [M rows × groups_per_row × 136 bytes]
//         (per 256-K group: 8-byte header = two fp16 (sc,zp) pairs, one per
//         128-K half, then 128 bytes nibble-packed with bit[4m..4m+3] = K
//         element m — same contract as the gfx11 kernel).
//   - Xq: activations pre-quantized to `block_i4_128` (72 B: f32 d, i32 s,
//         64 B nibbles, even K low / odd K high) via `quantize_int4_mmq_ds128`
//         (`ensure_int4_mmq_x`), layout [K/128 × N].
//   - Y:  output [N × M] column-major: Y[col * M + row].
//   - add: 1 => Y[..] += result (residual / _full_add); 0 => Y[..] = result.
//
// gfx12 K32 conventions (probe-verified on gfx1201, 2026-09-17 — one-hot /
// half-split / nibble-slot sweeps over a single-call WMMA):
//   - int32x2 A/B per lane (16 nibbles/lane = 8 per int32, low nibble first).
//   - K-tile lane split: k_grp = tid >> 4. Lanes 0-15 carry K[0..15] of row
//     m_lane (= tid & 15); lanes 16-31 carry K[16..31] of the same row.
//     A/B halves pair straight (cross-half dot reads 0).
//   - signed_a = false (weight nibbles are unsigned codes), signed_b = true
//     (int4 activations are two's complement, -8..7).
//   - Contiguous-row C output: acc[j] = C[8*k_grp + j][m_lane] (j=0..7),
//     int32x8 — same C map as the gfx12 iu8 MMQ.
//   - Per-half MQ4V2 (sc,zp) via __shfl from src_lane = 8*k_grp + j.
//
// v2 staged tile (port of the fp8 256x64x64 cooperative structure to the
// iu4 K32 contract; integer math is bit-identical to v1):
//   - Workgroup tile: IU4_BN rows x IU4_BM tokens (128x128), IU4_WAVES=8
//     waves (256 threads). Wave w covers 32 rows x 64 tokens = 2 row-groups
//     of 16 x 4 column blocks of 16: wt_pair = w>>1 selects the 32-row
//     group, tok_half = w&1 selects the 64-token half.
//   - Per 128-K block (kb): the activation slab (128 tokens x 64 K nibbles)
//     AND the weight slab (128 rows x 64 K nibbles) are staged ONCE in LDS
//     in WMMA fragment order (below) and consumed by all 8 waves. Staged A
//     bytes feed 4 row-pair waves x 4 (slab,sb) reads; staged W bytes feed
//     2 token-half waves x 4 nb x 4 reads.
//   - K-loop body per (slab, K32-sub-block) = ds_read_b64 (W frag) +
//     ds_read_b64 (A frag) + v_wmma_i32_16x16x32_iu4; the per-128-K fold
//     (IU4_FOLD_RN, once per block after the 2nd slab) is the ONLY other
//     K-loop VALU. All fragment addresses are hoisted to kernel top
//     (frA/frW tables); slab/sb/nb/rg are unroll constants, so the K-step
//     carries no dynamic address math.
//   - A and W are double-buffered per 64-K slab: slot 0 is (A0,W0), slot 1
//     is (A1,W1). DS and SZ metadata ping-pong per half-block (plane[h],
//     kb = 2g+h) while token (d,s) is re-read from the DS plane at fold time
//     (same values, same DAG).
//     Steady state per block: prefetch slab 1 into A_pf/W_pf while computing
//     slab 0 from slot 0, publish slab 1 to the free slot 1, then rendezvous
//     once for both slot-0 retirement and slot-1 readiness. The next-block
//     fetch epoch issues into the same private payload, overlaps slot-1
//     compute plus fold, publishes next slab 0 to the now-free slot 0, then
//     rendezvous once for both slot-1 retirement and slot-0 readiness.
//     Two barriers per block; each publish targets the alternate slot, so no
//     readers-done barrier is needed before its stores. Bit-identical: same K
//     order and pinned IU4_FOLD_RN DAG.
//     LDS 20480 B keeps 3 WG/CU on gfx1201 (3 x 20480 = 61440 <= 65536/CU).
//   - Coalesced store (fp8 K2 mechanism): each (rg, nb) phase dumps the
//     wave's 16x16 f32 sub-tile to an LDS slot ([16][20], stride-20 pad for
//     bank-clean streaming), then all 256 threads stream 64 rows x 32 cols
//     with consecutive threads on consecutive rows (128 B segments).
//     Per-element arithmetic is unchanged, hence bit-identical.
//   - Math is bit-identical to v1: same 4xK32 grouping per 128-K block per
//     column block (same K order), same pinned IU4_FOLD_RN DAG. Integer MMA:
//     any oracle difference is a bug.
//
// LDS layout (20480 B = 20 KiB on gfx1201; compute planes and store slots
// overlap — A/W/DS/SZ are all dead once folds complete):
//   A0 = 16 WMMA fragments x 256 B (fragment order, below)  @ 0     (4096 B)
//   W0 = 16 WMMA fragments x 256 B                         @ 4096  (4096 B)
//   W1 = 16 WMMA fragments x 256 B                         @ 8192  (4096 B)
//   DS0 = [128][8] token (d f32, s i32), even halves        @ 12288 (1024 B)
//   DS1 = [128][8] token (d f32, s i32), odd halves         @ 13312 (1024 B)
//   SZ0 = [128][8] row (sc f32, zp f32), even halves        @ 14336 (1024 B)
//   SZ1 = [128][8] row (sc f32, zp f32), odd halves         @ 15360 (1024 B)
//   A1 = 16 WMMA fragments x 256 B                         @ 16384 (4096 B)
//   end                                                             = 20480 B
//   store slots = 8 waves x [16][20] f32 @ 0                        (10240 B)
//
// Fragment order (probe-verified on gfx1201, 2026-09-18: the old
// [128][40] row layout cost a ~19x bank-conflict penalty on fragment
// reads — 1.8 TB/s aggregate vs 35.5 TB/s contiguous — while the loads
// alone (no WMMA) took the same time as the full fed loop). Each 64-K
// slab = 8 row/token groups of 16 x 2 K32 steps = 16 fragments; fragment
// (g8, sb) = 256 contiguous bytes at slab + (g8*2+sb)*256, holding the
// exact 8-byte lanes v_wmma_i32_16x16x32_iu4 expects: lane L = tid & 31
// reads slab + (g8*2+sb)*256 + L*8, i.e. row (g8*16 + (L&15))'s K-half
// (L>>4) at K offset sb*16 + (L>>4)*8 bytes (low nibble first, as before).
// The K-loop fragment read is one ds_read_b64 at a hoisted address.
//
// Bank enumeration (32 banks x 4 B; lane L reads 8 bytes at frag*256+L*8):
// dword addresses frag*64 + 2*L .. +2*L+1 run 0..63 over the wave: 64
// consecutive dwords = every bank exactly twice = the b64 width floor
// (unavoidable: 64 dwords over 32 banks), matching the probe's contiguous
// pattern. Staging writes permute (row R, quad q) to frag (R>>4)*2+(q>>1),
// lane (R&15)+16*(q&1): bank = 2*(R&15), so each wave's 32 writes cover 8
// banks x 4 hits (4-way write conflict). Accepted: each slab byte is
// written once and read 4x (A: 4 waves share a token half) / 2x (W: 2 waves
// share a row pair), so the conflict-free reads dominate. DS/SZ 4 B reads
// unchanged (banks 2*R: 16 distinct evens, broadcast within a k_grp half).
// Store stream reads: bank = 5*i + x/4, 5 coprime 32 (2-way floor across
// x-pairs); global stores are 2x64 B segments per wave32.
//
// Grid: [ceil(M/128), ceil(N/128)]. Block: [256,1,1]. LDS: 20480 bytes.
//
// IU4_G12_RASTER (the `_g12r` modules; `HIPFIRE_G12_RASTER=0` selects the
// incumbent build) changes only CTA order and the full-tile store paths; the
// K order, fold DAG and per-element SET/ADD/SILU arithmetic are untouched,
// so its outputs are bit-identical:
//   - Raster: the linear dispatch id walks bands of IU4_RASTER_BAND (8) row
//     tiles (SILU: virtual gate+up tiles, replacing IU4_SILU_RGROUPS), rows
//     fastest within a band, then tokens. The ~96 resident CTAs then share
//     ~8 W row tiles and ~12 token tiles (about 7 MB, L2-sized) instead of
//     streaming the whole weight matrix per token tile.
//   - Wide SET/ADD epilogue: per 16-token block nb, all waves dump
//     acc[nb][0..1] into one [32 tok][132] f32 LDS tile, then each wave
//     writes whole 128-row token columns (512 contiguous B) with b128 stores:
//     8 barriers instead of 16, 16 b128 instead of 64 b32 global stores
//     (plus as many old-Y loads for ADD) per thread. SILU does the same with
//     a [32 tok][68] tile of its 64 h rows (256 contiguous B per column).
//     Needs full tiles (SILU: full token columns), M % 4 == 0 and a 16 B
//     aligned Y; anything else takes the incumbent stores.
//   Measured card-E standalone and in pp8192 rocprof: see
//   scratch-g12raster2/REPORT.md.
//
// Sweep note: 64x256 / 256x64 aspects need a different wave map (wt_pair /
// tok_half split) plus matching host grid/LDS; the fold/prefetch/store
// mechanisms transfer unchanged. Host, oracle, and timing launchers hardcode
// the same 128/128/256-thread, 20480-byte contract and must change together.
//
// Compile gate: gfx1200 / gfx1201 only. Non-gfx12 archs get empty stubs (the

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#if defined(__gfx1200__) || defined(__gfx1201__)

// block_i4_128 is provided by the host JIT prelude
// (kernels/src/block_i4_128_quant.hip, concatenated by
// GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_SRC).

// Workgroup tile geometry (tokens x rows x staged-K, waves). Overridable
// for sweeps; host + oracle must match.
#ifndef IU4_BM
#define IU4_BM 128
#endif
#ifndef IU4_BN
#define IU4_BN 128
#endif
#ifndef IU4_BK
#define IU4_BK 64
#endif
#ifndef IU4_WAVES
#define IU4_WAVES 8
#endif
#define IU4_NTHREADS (IU4_WAVES * 32)
// Row-tile band of the IU4_G12_RASTER CTA order.
#ifndef IU4_RASTER_BAND
#define IU4_RASTER_BAND 8
#endif
// Individually selectable K1 levers (CACC2 is exploratory, off by default).
#ifndef IU4V3_MAGIC
#define IU4V3_MAGIC 1
#endif
#ifndef IU4V3_SWIZZLE
#define IU4V3_SWIZZLE 1
#endif
#ifndef IU4V3_BUFFER
#define IU4V3_BUFFER 1
#endif
#ifndef IU4V3_CACC2
#define IU4V3_CACC2 0
#endif
// A CACC2 implementation needs a second 64-VGPR accumulator and a revised
// fold/compute schedule; do not silently compile an identical kernel.
_Static_assert(!IU4V3_CACC2, "IU4V3_CACC2 is not implemented");

// Fragment-order slab: each 64-K slab (IU4_BK nibbles/row) holds 8 row/token
// groups of 16 x 2 K32 steps = 16 fragments of 256 contiguous bytes
// (32 lanes x 8 B, lane L at frag*256 + L*8). Same 4096 data bytes as the
// old [128][32] rows, no pad: contiguous-lane b64 reads sit at the b64
// width floor (see bank enumeration above).
#define IU4_FRAG_BYTES 256
#define IU4_A_FRAGS ((IU4_BM / 16) * (IU4_BK / 32))
#define IU4_W_FRAGS ((IU4_BN / 16) * (IU4_BK / 32))
#define IU4_A_BYTES (IU4_A_FRAGS * IU4_FRAG_BYTES)
#define IU4_W_BYTES (IU4_W_FRAGS * IU4_FRAG_BYTES)
#define IU4_A_OFF 0
#define IU4_W0_OFF (IU4_A_OFF + IU4_A_BYTES)
#define IU4_W1_OFF (IU4_W0_OFF + IU4_W_BYTES)
#define IU4_DS_OFF (IU4_W1_OFF + IU4_W_BYTES)
#define IU4_DS_BYTES (IU4_BM * 8)
#define IU4_DS1_OFF (IU4_DS_OFF + IU4_DS_BYTES)
#define IU4_SZ0_OFF (IU4_DS1_OFF + IU4_DS_BYTES)
#define IU4_SZ_BYTES (IU4_BN * 8)
#define IU4_SZ1_OFF (IU4_SZ0_OFF + IU4_SZ_BYTES)
#define IU4_A1_OFF (IU4_SZ1_OFF + IU4_SZ_BYTES)
#define IU4_LDS_BYTES (IU4_A1_OFF + IU4_A_BYTES)
_Static_assert(IU4_LDS_BYTES <= 20480, "iu4 fragment planes exceed host LDS");
// Store slots reuse LDS base: 8 waves x [16][20] f32.
#define IU4_SLOT_COLS 20
#define IU4_SLOT_FLOATS (16 * IU4_SLOT_COLS)

typedef int8_t  __attribute__((ext_vector_type(8)))  int8x8_t;
typedef int32_t __attribute__((ext_vector_type(2)))  int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8)))  int32x8_t;
typedef float   __attribute__((ext_vector_type(8)))  float8_t;
// 8-byte staging word: the width ceiling for A/W traffic (see the staging-map
// note at the K-loop top). Ext-vector so global/LDS accesses codegen to
// single b64 instructions with elementwise branch-free selects.
typedef unsigned __attribute__((ext_vector_type(2))) u32x2_t;

static __device__ __forceinline__ u32x2_t iu4_weight_staging_word(u32x2_t value) {
#if defined(IU4_SYMMETRIC_FOLD) && defined(__gfx1201__)
    // Flip each uint4 code's sign bit so the signed WMMA consumes q-8.
    return value ^ (u32x2_t){0x88888888u, 0x88888888u};
#else
    return value;
#endif
}
#if defined(IU4_SYMMETRIC_FOLD) && defined(__gfx1201__)
#define IU4_WEIGHT_SIGNED true
#else
#define IU4_WEIGHT_SIGNED false
#endif
// IU4_FOLD_RN — W4.0/LF.0 bit-exact fold pin, shared with the gfx11 iu4
// kernel. Every output element folds one int32 WMMA result C through EXACTLY
// this DAG:
//   t1   = RN(sc * d)
//   p    = RN(t1 * float(C))
//   t2   = RN(zp * d)
//   term = RN(t2 * float(s) + p)
//   sum  = RN(sum + term)
#define IU4_FOLD_RN(sum_lvalue, sc, zp, d, acc_i32, s_i32) \
    do { \
        const float iu4_fold_d_ = (d); \
        const float iu4_fold_t1_ = __fmul_rn((sc), iu4_fold_d_); \
        const float iu4_fold_p_ = __fmul_rn(iu4_fold_t1_, (float)(acc_i32)); \
        const float iu4_fold_t2_ = __fmul_rn((zp), iu4_fold_d_); \
        const float iu4_fold_term_ = __fmaf_rn(iu4_fold_t2_, (float)(s_i32), iu4_fold_p_); \
        (sum_lvalue) = __fadd_rn((sum_lvalue), iu4_fold_term_); \
    } while (0)
#if defined(__gfx1201__)
// IU4_FOLD_RN_PAIR — gfx1201 VOPD fold pairing (exact-arithmetic form 1).
// Two independent j-components through the pinned IU4_FOLD_RN DAG with
// stages interleaved pairwise (t1_0,t1_1 then p_0,p_1 ...) so the backend
// can form V_DUAL_MUL_F32 / V_DUAL_FMAC_F32 / V_DUAL_ADD_F32 packets.
// Bit-exact by construction: the same RN multiply/FMA/add nodes in the
// same per-component order, no contraction, no precision change; i32->f32
// conversions stay scalar (no dual conversion form exists on gfx1201).
// By-value sums + float2 return (ext_vector elements cannot bind to
// float&); forceinline keeps acc traffic direct. The gfx1200 path keeps
// the scalar IU4_FOLD_RN loop below untouched.
static __device__ __forceinline__ float2 iu4_fold_rn_pair(
    float sum0, float sum1,
    float sc0, float sc1, float zp0, float zp1,
    float d, int c0, int c1, float sf) {
    const float t1_0 = __fmul_rn(sc0, d);
    const float t1_1 = __fmul_rn(sc1, d);
    const float p_0 = __fmul_rn(t1_0, (float)c0);
    const float p_1 = __fmul_rn(t1_1, (float)c1);
    const float t2_0 = __fmul_rn(zp0, d);
    const float t2_1 = __fmul_rn(zp1, d);
    const float term_0 = __fmaf_rn(t2_0, sf, p_0);
    const float term_1 = __fmaf_rn(t2_1, sf, p_1);
    float2 out;
    out.x = __fadd_rn(sum0, term_0);
    out.y = __fadd_rn(sum1, term_1);
    return out;
}
#if defined(IU4_SYMMETRIC_FOLD) && defined(__gfx1201__)
static __device__ __forceinline__ float2 iu4_symmetric_fold_rn_pair(
    float sum0, float sum1,
    float sc0, float sc1,
    float d, int c0, int c1) {
    const float t1_0 = __fmul_rn(sc0, d);
    const float t1_1 = __fmul_rn(sc1, d);
#if IU4V3_MAGIC
    const float fc0 = __builtin_bit_cast(float, c0) - 12582912.0f;
    const float fc1 = __builtin_bit_cast(float, c1) - 12582912.0f;
#else
    const float fc0 = (float)c0;
    const float fc1 = (float)c1;
#endif
    const float p_0 = __fmul_rn(t1_0, fc0);
    const float p_1 = __fmul_rn(t1_1, fc1);
    float2 out;
    out.x = __fadd_rn(sum0, p_0);
    out.y = __fadd_rn(sum1, p_1);
    return out;
}
#endif
#endif

// One K32-sub-block compute bundle: 2 W-frags + 4 A-frags +
// 8 WMMAs (2 row-groups x 4 column blocks), then a scheduling fence.
// SB is a template constant (folded into LDS immediates); FIRST picks
// the shared zero C vector for each chain's first step (avoids 64 separate
// Compute reads slab sb of A from the single A plane and
// of W from Wn (n = sb); the caller re-points w_lds per phase. Both W planes
// share the fragment-order layout, so the hoisted frA/frW offsets apply to
// either. Each fragment read is one ds_read_b64 at frag*256 + lane*8.
// K-step VALU: ds_reads + WMMAs only.
template <int SB, bool FIRST>
static __device__ __forceinline__ void iu4_bundle(
    const char* __restrict__ w_lds,
    const char* __restrict__ a_lds,
    const unsigned (&frW)[2][2],
    const unsigned (&frA)[4][2],
    int32x8_t cacc[4][2],
    const int32x8_t& zc
) {
    const int32x2_t af0 = *(const int32x2_t*)(w_lds + frW[0][SB]);
    const int32x2_t af1 = *(const int32x2_t*)(w_lds + frW[1][SB]);
    const int32x2_t bf0 = *(const int32x2_t*)(a_lds + frA[0][SB]);
    const int32x2_t bf1 = *(const int32x2_t*)(a_lds + frA[1][SB]);
    const int32x2_t bf2 = *(const int32x2_t*)(a_lds + frA[2][SB]);
    const int32x2_t bf3 = *(const int32x2_t*)(a_lds + frA[3][SB]);
    #pragma unroll
    for (int rg = 0; rg < 2; rg++) {
        const int32x2_t a = (rg == 0) ? af0 : af1;
        #pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            const int32x2_t b = (nb == 0) ? bf0 : (nb == 1) ? bf1 : (nb == 2) ? bf2 : bf3;
            if constexpr (FIRST) {
                cacc[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                    IU4_WEIGHT_SIGNED, a, true, b, zc, false
                );
            } else {
                cacc[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                    IU4_WEIGHT_SIGNED, a, true, b, cacc[nb][rg], false
                );
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

// Shared compute body. `EPI` selects the writeback:
//   IU4_EPI_SET / IU4_EPI_ADD: Y = / += result (full_set / full_add / residual).
//   IU4_EPI_SILU (F1-lite gate/up): A = W_gate, U = W_up, M rows each. The grid
//     covers 2M virtual rows; virtual 16-row group g8 of a tile holds gate
//     (g8 even) or up (g8 odd) rows blockIdx.x*64 + 16*(g8>>1) + 0..15, so a
//     wave's rg=0 / rg=1 accumulators are gate / up of the same 16 h rows (an
//     address interleave; weights untouched). Every g and u folds exactly as in
//     full_set, and Y = h [N][M] = IU4_SILU_MUL(g, u), spelled as SILU_MUL in
//     fused_silu_mul_mq_rotate_awq.hip so the h-producer reproduces it.
enum { IU4_EPI_SET = 0, IU4_EPI_ADD = 1, IU4_EPI_SILU = 2 };
#define IU4_SILU_MUL(g, u) ((g) / (1.0f + expf(-(g))) * (u))
#ifndef IU4_SILU_RGROUPS
#define IU4_SILU_RGROUPS 2
#endif
template <int EPI>
static __device__ __forceinline__ void gemm_mq4g256v2_residual_mmq_iu4_gfx12_body(
    const char*           __restrict__ A,   // [M × groups_per_row × 136]
    const char*           __restrict__ U,   // SILU: W_up, same layout as A
    const block_i4_128*   __restrict__ Xq,  // [K/128 × N]
    float*                __restrict__ Y,   // [N × M] column-major
    int M, int K, int N
) {
    constexpr bool ADD = EPI == IU4_EPI_ADD;
    constexpr bool SILU = EPI == IU4_EPI_SILU;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    int rs = blockIdx.x * IU4_BN;
    int bs = blockIdx.y * IU4_BM;
#if defined(IU4_G12_RASTER)
    // Banded raster (see header), SET/ADD and SILU alike: bijective for any
    // grid, partial last band. SILU bands count virtual (gate+up) row tiles.
    {
        const int n_row_tiles = gridDim.x;
        const int lin = blockIdx.y * n_row_tiles + blockIdx.x;
        const int per_band = IU4_RASTER_BAND * gridDim.y;
        const int band = lin / per_band;
        const int in_band = lin - band * per_band;
        const int band_row0 = band * IU4_RASTER_BAND;
        const int band_rows = min(n_row_tiles - band_row0, IU4_RASTER_BAND);
        rs = (band_row0 + in_band % band_rows) * IU4_BN;
        bs = (in_band / band_rows) * IU4_BM;
    }
#else
    if constexpr (SILU) {
        // Raster: the fused grid streams both matrices, twice the weight
        // bytes of one SET. Visit IU4_SILU_RGROUPS contiguous row-tile groups
        // one after the other, each over every token tile, so a group's
        // weights (one SET's worth for 2 groups) stay cache-resident across
        // token tiles like the separate SETs'. Host: gridDim.x % RGROUPS == 0.
        const int per = (int)gridDim.x / IU4_SILU_RGROUPS;
        const int lin = (int)blockIdx.x + (int)gridDim.x * (int)blockIdx.y;
        const int span = per * (int)gridDim.y;
        const int grp = lin / span;
        const int in = lin - grp * span;
        rs = (grp * per + in % per) * IU4_BN;
        bs = (in / per) * IU4_BM;
    }
#endif

    if (rs >= (SILU ? 2 * M : M) || bs >= N) return;

    // SILU: first h row of the tile. Staging W row R (= r*64 + (tid>>2)) is
    // gate/up by bit 6 of tid and SZ row tid>>1 by bit 5: both wave-uniform,
    // so each wave reads one of the two matrices from an SGPR base.
    const int hs = rs >> 1;
    const char* __restrict__ Aw = A;
    const char* __restrict__ As = A;
    int sz_row = 0;
    if constexpr (SILU) {
        static_assert(IU4_BN * 2 == IU4_NTHREADS, "one SZ word per thread");
        Aw = __builtin_amdgcn_readfirstlane((tid >> 6) & 1) ? U : A;
        As = __builtin_amdgcn_readfirstlane((tid >> 5) & 1) ? U : A;
        sz_row = hs + (tid >> 6) * 16 + ((tid >> 1) & 15);
    }

    const int m_lane = lane & 15;  // 0..15: row within a 16-row group
    const int k_grp  = lane >> 4;  // 0/1: which K-half this lane carries
    const int wt_pair = wave >> 1;     // 0..3: which 32-row group
    const int tok_half = wave & 1;     // 0/1: which 64-token half
    const int wave_row0 = wt_pair * 32;
    const int wave_tok0 = tok_half * 64;

    const int groups_per_row = K / 256;
    extern __shared__ char LDS[];
    // The current A/W slab always shares slot parity.
    char* __restrict__ a_lds = LDS + IU4_A_OFF;
    char* __restrict__ w_lds = LDS + IU4_W0_OFF;
    // DS/SZ ping-pong bases (plane[h], kb = 2g+h) selected inside the loop.
    // Hoisted K-loop addresses: full LDS byte offsets of every fragment this
    // lane will ever read. Computed once at kernel top (outside ALL loops);
    // the K-step indexes them with unroll constants only.
    unsigned frA[4][2];
    unsigned frW[2][2];
    // Fragment-order offsets: token group g8 = tok_half*4+nb (A) or row
    // group g8 = wt_pair*2+rg (W); fragment (g8,sb) lives at (g8*2+sb)*256
    // and this lane's 8 bytes at +lane8. All loop-invariant (wave/nb/rg/sb
    // are kernel-top or unroll constants; lane = tid&31 is fixed).
    // π_F(L) is an XOR within each 16-lane half, preserving the bijection.
    // Four (fragment parity, lane half) pairs distribute each wave's staging
    // writes across all banks instead of four copies of the same eight banks.
#if IU4V3_SWIZZLE
    const unsigned lane_lo = (unsigned)lane & 15u;
    const unsigned lane_half = (unsigned)lane & 16u;
    const unsigned lane_even = ((lane_lo ^ (lane_half >> 2)) | lane_half) * 8u;
    const unsigned lane_odd = ((lane_lo ^ (8u ^ (lane_half >> 2))) | lane_half) * 8u;
#else
    const unsigned lane_even = (unsigned)lane * 8u;
    const unsigned lane_odd = lane_even;
#endif
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        const unsigned g8a = (unsigned)(tok_half * 4 + nb);
        #pragma unroll
        for (int sb = 0; sb < 2; sb++) {
            frA[nb][sb] = (g8a * 2u + (unsigned)sb) * (unsigned)IU4_FRAG_BYTES
                + (sb ? lane_odd : lane_even);
        }
    }
    #pragma unroll
    for (int rg = 0; rg < 2; rg++) {
        const unsigned g8w = (unsigned)(wt_pair * 2 + rg);
        #pragma unroll
        for (int sb = 0; sb < 2; sb++) {
            frW[rg][sb] = (g8w * 2u + (unsigned)sb) * (unsigned)IU4_FRAG_BYTES
                + (sb ? lane_odd : lane_even);
        }
    }

    // Output accumulators (gfx12 wave32 layout):
    // acc[nb][rg][j] = Y[wave_tok0 + nb*16 + m_lane][rs + wave_row0 + rg*16 + 8*k_grp + j].
    float8_t acc[4][2];
    // b64-coalesced staging map. Each lane stages 2 rounds; round r covers 64
    // slab rows (R = r*64 + (tid>>2)) with quad q = tid&3 (8 contiguous
    // bytes) per lane: 256 lanes x 2 rounds x 8 B = one 4 KiB slab exactly
    // once. Lanes in a quad-group share a row, so consecutive lanes read
    // consecutive quads (coalesced global segments). The LDS destination is
    // the fragment-order slot holding those bytes for the consumer lane:
    // frag (R>>4)*2+(q>>1), lane (R&15)+16*(q&1) — a bijection, so every
    // fragment slot is written exactly once per publish (same bytes as the
    // old row map, hence oracle-safe). b64 is the width ceiling, NOT b128:
    // every 16 B chunk of the Xq (72 B blocks) and W (136 B groups) nibble
    // arrays sits at 8-mod-16 behind an 8-byte header — b128 is
    // alignment-impossible on the global side, b64 is exact.
    // All offsets below are loop-invariant (bs/rs/tid only), computed once at
    // Hoisted vector offsets; buffer descriptor bases and scalar offsets
    // advance with kb/g/h, leaving no 64-bit vector address reconstruction.
    // W rows remain clamped. With IU4V3_BUFFER the A descriptor spans
    // precisely one K128 block, so out-of-range token offsets return zero.
    unsigned st_avoff[2];
    unsigned st_wvoff[2];
    unsigned st_ldsoff[2];
#if !IU4V3_BUFFER
    unsigned st_aok[2];
#endif
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        const int tok_r = bs + r * 64 + (tid >> 2);
#if IU4V3_BUFFER
        st_avoff[r] = (unsigned)tok_r * (unsigned)sizeof(block_i4_128)
            + (unsigned)(tid & 3) * 8u;
#else
        const int tok_c = (tok_r < N) ? tok_r : (N - 1);
        st_aok[r] = (tok_r < N) ? 1u : 0u;
        st_avoff[r] = (unsigned)tok_c * (unsigned)sizeof(block_i4_128)
            + (unsigned)(tid & 3) * 8u;
#endif
        if constexpr (SILU) {
            // Virtual row R: group R>>4 = 2r + (tid>>6) -> gate/up by bit 0,
            // h row hs + 16*(R>>5) + (R&15).
            const int hrow = hs + (r * 2 + (tid >> 7)) * 16 + ((tid >> 2) & 15);
            const int src_c = (hrow < M) ? hrow : (M - 1);
            st_wvoff[r] = (unsigned)src_c * (unsigned)groups_per_row * 136u
                + (unsigned)(tid & 3) * 8u;
        } else {
            const int grow_r = rs + r * 64 + (tid >> 2);
            const int src_c = (grow_r < M) ? grow_r : (M - 1);
            st_wvoff[r] = (unsigned)src_c * (unsigned)groups_per_row * 136u
                + (unsigned)(tid & 3) * 8u;
        }
        const unsigned R = (unsigned)(r * 64 + (tid >> 2));
        const unsigned q = (unsigned)(tid & 3);
        const unsigned frag = (R >> 4) * 2u + (q >> 1);
        const unsigned L = (R & 15u) + (q & 1u) * 16u;
#if IU4V3_SWIZZLE
        const unsigned sw = ((frag & 1u) << 3) ^ ((L >> 4) << 2);
        st_ldsoff[r] = frag * (unsigned)IU4_FRAG_BYTES + (L ^ sw) * 8u;
#else
        st_ldsoff[r] = frag * (unsigned)IU4_FRAG_BYTES + L * 8u;
#endif
    }
#if !IU4V3_BUFFER
    const u32x2_t z64 = { 0u, 0u };
#endif
    for (int nb = 0; nb < 4; nb++) {
        #pragma unroll
        for (int rg = 0; rg < 2; rg++) {
            acc[nb][rg] = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
    // The first WMMA of each K128 block injects a bit-pattern that is also
    // the exact float 12582912.0; integer WMMA adds the bounded signed dot.
#if IU4V3_MAGIC
    const int32x8_t zc = (int32x8_t){0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000,
                                      0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000};
#else
    const int32x8_t zc = (int32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
#endif
    int32x8_t cacc[4][2];
#if IU4V3_BUFFER
    const unsigned a_bytes = (unsigned)N * (unsigned)sizeof(block_i4_128);
    const auto w_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)Aw, 0, 0xffffffffu, 0x31004000u);
#endif
    // Slab prefetch registers (slab s+1 globals while computing slab s): b64
    // pairs, same 8 dwords/lane as before (2 rounds x 1 u32x2 per plane).
    u32x2_t A_pf[2];
    u32x2_t W_pf[2];
    // Next-block raw metadata words for the rescheduled fetch epoch (loaded
    // after B1, consumed at the publish below B2a): one token DS word and
    // one packed weight header word per lane.
    unsigned ds_nx = 0u;
    unsigned sz_nx = 0u;

    // Token (d, s) for this lane's column blocks is re-read from the staged
    // DS plane at fold time (same f32/i32 values, same DAG order) instead of
    // being held in td_x/ts_x across compute: frees 8 VGPRs in the hot
    // compute+staging window. DS-next staging still waits for the fold.

    // --- Prologue: DS[0] + SZ[0] + slab 0 of block 0 (A plane + W0). ---
    for (int i = tid; i < IU4_BM * 2; i += IU4_NTHREADS) {
        const int tok_idx = i >> 1;
        const int w = i & 1;
        const int tok = bs + tok_idx;
        const unsigned dw = (tok < N)
            ? ((const unsigned*)(Xq + tok))[w]
            : 0u;
        ((unsigned*)(LDS + IU4_DS_OFF + tok_idx * 8))[w] = dw;
    }
    if constexpr (SILU) {
        // SZ row tid>>1 (virtual) = gate/up (bit 4) of h row sz_row.
        const int w = tid & 1;
        const int src_row = (sz_row < M) ? sz_row : (M - 1);
        const unsigned hw = *(const unsigned*)(As + (long long)src_row * groups_per_row * 136);
        const float f = (w == 0)
            ? __half2float(__ushort_as_half((unsigned short)(hw & 0xFFFFu)))
            : __half2float(__ushort_as_half((unsigned short)(hw >> 16)));
        ((float*)(LDS + IU4_SZ0_OFF + (tid >> 1) * 8))[w] = (sz_row < M) ? f : 0.0f;
    } else {
        for (int i = tid; i < IU4_BN * 2; i += IU4_NTHREADS) {
            const int row_idx = i >> 1;
            const int w = i & 1;
            const int grow = rs + row_idx;
            const int src_row = (grow < M) ? grow : (M - 1);
            const unsigned hw = *(const unsigned*)(A + (long long)src_row * groups_per_row * 136);
            const float f = (w == 0)
                ? __half2float(__ushort_as_half((unsigned short)(hw & 0xFFFFu)))
                : __half2float(__ushort_as_half((unsigned short)(hw >> 16)));
            ((float*)(LDS + IU4_SZ0_OFF + row_idx * 8))[w] = (grow < M) ? f : 0.0f;
        }
    }
    // Prologue slab 0 of block 0, b64-coalesced (same map as the K-loop).
    {
#if IU4V3_BUFFER
        const auto a0_rsrc = __builtin_amdgcn_make_buffer_rsrc((void*)Xq, 0, a_bytes, 0x31004000u);
#else
        const char* __restrict__ xqB0 = (const char*)Xq + 8;
        const char* __restrict__ wB0 = Aw + 8;
#endif
        #pragma unroll
        for (int r = 0; r < 2; r++) {
#if IU4V3_BUFFER
            const u32x2_t av = __builtin_amdgcn_raw_buffer_load_b64(a0_rsrc, st_avoff[r], 8u, 0);
            const u32x2_t wv = __builtin_amdgcn_raw_buffer_load_b64(w_rsrc, st_wvoff[r], 8u, 0);
            *(u32x2_t*)(a_lds + st_ldsoff[r]) = av;
#else
            const u32x2_t av = *(const u32x2_t*)(xqB0 + st_avoff[r]);
            const u32x2_t wv = *(const u32x2_t*)(wB0 + st_wvoff[r]);
            *(u32x2_t*)(a_lds + st_ldsoff[r]) = st_aok[r] ? av : z64;
#endif
            *(u32x2_t*)(w_lds + st_ldsoff[r]) = iu4_weight_staging_word(wv);
        }
    }
    __syncthreads();

    for (int g = 0; g < groups_per_row; g++) {
        // Rolled (not unrolled: unroll_count(1)): the unrolled h x 2 body
        // keeps block k's staging+fold temporaries live across block k+1's
        // prefetch+compute and blows VGPRs past 256 with spills. Rolled, only
        // cacc/acc carry across iterations.
        #pragma clang loop unroll_count(1)
        for (int h = 0; h < 2; h++) {
            const int kb = 2 * g + h;
            const bool last = (h == 1) && (g + 1 == groups_per_row);
            // DS/SZ planes for this half-block (kb parity = h); the next
            // block stages the opposite planes, so its writes never race
            // this block's compute+fold reads.
            char* __restrict__ ds_lds = LDS + ((h == 0) ? IU4_DS_OFF : IU4_DS1_OFF);
            char* __restrict__ sz_lds = LDS + ((h == 0) ? IU4_SZ0_OFF : IU4_SZ1_OFF);
            // --- Slot 0 compute with slot 1 prefetch as producer. ---
            {
                // Slab 1 prefetch, b64-coalesced: uniform SGPR bases carry
                // kb/g/h, lane VGPR offsets were hoisted to kernel top.
#if IU4V3_BUFFER
                const auto a_rsrc = __builtin_amdgcn_make_buffer_rsrc(
                    (void*)(Xq + (long long)kb * N), 0, a_bytes, 0x31004000u);
#else
                const char* __restrict__ xqB1 = (const char*)(Xq + (long long)kb * N) + 8 + 32;
                const char* __restrict__ wB1 = Aw + (long long)g * 136 + 8 + (long long)h * 64 + 32;
#endif
                #pragma unroll
                for (int r = 0; r < 2; r++) {
#if IU4V3_BUFFER
                    A_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(a_rsrc, st_avoff[r], 40u, 0);
                    W_pf[r] = iu4_weight_staging_word(__builtin_amdgcn_raw_buffer_load_b64(
                        w_rsrc, st_wvoff[r], (unsigned)(g * 136 + h * 64 + 40), 0));
#else
                    A_pf[r] = *(const u32x2_t*)(xqB1 + st_avoff[r]);
                    W_pf[r] = iu4_weight_staging_word(*(const u32x2_t*)(wB1 + st_wvoff[r]));
#endif
                }
                // No scheduling pin here: the previous trip's fold VALU must
                // remain free to sink among this trip's WMMAs. Raw A_pf is
                // consumed only at the alternate-slot publication below.
                // Every 128-K block restarts its int32 accumulation from zero
                // (each block folds independently): FIRST=true on slab 0 of
                // every (g,h), exactly as in v2. No h-dependence here.
                iu4_bundle<0, true>(w_lds, a_lds, frW, frA, cacc, zc);
                iu4_bundle<1, false>(w_lds, a_lds, frW, frA, cacc, zc);
                // Publish slab 1 to the free alternate slot. The raw A pin
                // keeps its zero-select at this first use, after slot-0
                // compute, without imposing a memory or scheduling fence.
                a_lds = LDS + IU4_A1_OFF;
                w_lds = LDS + IU4_W1_OFF;
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    asm volatile("" : "+v"(A_pf[r][0]), "+v"(A_pf[r][1]));
#if IU4V3_BUFFER
                    *(u32x2_t*)(a_lds + st_ldsoff[r]) = A_pf[r];
#else
                    *(u32x2_t*)(a_lds + st_ldsoff[r]) = st_aok[r] ? A_pf[r] : z64;
#endif
                    *(u32x2_t*)(w_lds + st_ldsoff[r]) = W_pf[r];
                }
                // Slot-0 retirement plus slot-1 readiness.
                __syncthreads();
                if (!last) {
                    // Next-block fetch epoch: all six next-block raw loads
                    // (4 b64 nibble + 1 DS word + 1 packed header word) issue
                    // here, immediately after B1, into private registers. No
                    // masks, header conversion, DS stores or first use yet:
                    // first use is the alternate-slot publish after compute1
                    // plus fold, so the waits overlap that useful window.
                    // A_pf/W_pf are dead (slab 1 published above) and reused
                    // as the next-payload nibble words; ds_nx/sz_nx hold the
                    // raw next metadata words.
                    const int kb_nx = kb + 1;
#if IU4V3_BUFFER
                    const auto a_next_rsrc = __builtin_amdgcn_make_buffer_rsrc(
                        (void*)(Xq + (long long)kb_nx * N), 0, a_bytes, 0x31004000u);
#else
                    const char* __restrict__ xq_nx = (const char*)(Xq + (long long)kb_nx * N) + 8;
#endif
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
#if IU4V3_BUFFER
                        A_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(a_next_rsrc, st_avoff[r], 8u, 0);
#else
                        A_pf[r] = *(const u32x2_t*)(xq_nx + st_avoff[r]);
#endif
                    }
                    const int g_nx = g + h;
                    const int h_nx = h ^ 1;
#if !IU4V3_BUFFER
                    const char* __restrict__ w_nx = Aw + (long long)g_nx * 136 + 8 + (long long)h_nx * 64;
#endif
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
#if IU4V3_BUFFER
                        W_pf[r] = iu4_weight_staging_word(__builtin_amdgcn_raw_buffer_load_b64(
                            w_rsrc, st_wvoff[r], (unsigned)(g_nx * 136 + h_nx * 64 + 8), 0));
#else
                        W_pf[r] = iu4_weight_staging_word(*(const u32x2_t*)(w_nx + st_wvoff[r]));
#endif
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
            // --- Slot 1 compute, current fold, next slot-0 publish. ---
            {
                iu4_bundle<0, false>(w_lds, a_lds, frW, frA, cacc, zc);
                iu4_bundle<1, false>(w_lds, a_lds, frW, frA, cacc, zc);
                // Fold while the next fetch epoch remains in flight: 4 K32
                // int32 partials per (nb, rg)
                // through the pinned IU4_FOLD_RN DAG. Reads only cacc plus
                // the current DS[h]/SZ[h] planes, independent of the private
                // next payload. Same f32 values, same DAG order: per-half
                // (sc,zp) straight from the staged SZ plane, (d,s) re-read
                // from DS. Bank-clean on gfx1201: 8 consecutive rows x 2
                // words hit 16 distinct banks, identical across the 16 lanes
                // of a k_grp half (LDS broadcast); the halves cover all 32
                // banks once.
                #pragma unroll
                for (int rg = 0; rg < 2; rg++) {
                    float sc_row[8];
#if !defined(IU4_SYMMETRIC_FOLD) || !defined(__gfx1201__)
                    float zp_row[8];
#endif
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const unsigned jbase = (unsigned)(wave_row0 + rg * 16 + 8 * k_grp + j) * 8u;
                        sc_row[j] = *(const float*)(sz_lds + jbase);
#if !defined(IU4_SYMMETRIC_FOLD) || !defined(__gfx1201__)
                        zp_row[j] = *(const float*)(sz_lds + jbase + 4u);
#endif
                    }
                    #pragma unroll
                    for (int nb = 0; nb < 4; nb++) {
                        int* acc_i = (int*)&cacc[nb][rg];
                        const unsigned tbase = (unsigned)(wave_tok0 + nb * 16 + m_lane) * 8u;
                        const float d_x = *(const float*)(ds_lds + tbase);
#if defined(IU4_SYMMETRIC_FOLD) && defined(__gfx1201__)
                        #pragma unroll
                        for (int j = 0; j < 8; j += 2) {
                            const float2 pair = iu4_symmetric_fold_rn_pair(
                                acc[nb][rg][j], acc[nb][rg][j + 1],
                                sc_row[j], sc_row[j + 1],
                                d_x, acc_i[j], acc_i[j + 1]);
                            acc[nb][rg][j] = pair.x;
                            acc[nb][rg][j + 1] = pair.y;
                        }
#else
                        const int s_x = *(const int*)(ds_lds + tbase + 4u);
#if defined(__gfx1201__)
                        // Paired fold: same values, same DAG as IU4_FOLD_RN;
                        // sf hoisted (the int sum is loop-invariant, the
                        // compiler CSEs the baseline's per-j conversion to
                        // one anyway: 4 sf converts per half, unchanged).
                        const float sf_x = (float)s_x;
                        #pragma unroll
                        for (int j = 0; j < 8; j += 2) {
                            const float2 pair = iu4_fold_rn_pair(
                                acc[nb][rg][j], acc[nb][rg][j + 1],
                                sc_row[j], sc_row[j + 1],
                                zp_row[j], zp_row[j + 1],
                                d_x, acc_i[j], acc_i[j + 1], sf_x);
                            acc[nb][rg][j] = pair.x;
                            acc[nb][rg][j + 1] = pair.y;
                        }
#else
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            IU4_FOLD_RN(acc[nb][rg][j], sc_row[j], zp_row[j], d_x, acc_i[j], s_x);
                        }
#endif
#endif
                    }
                }
                // Slot 0 was retired at the preceding rendezvous and is free
                // while slot 1 computes, so publish needs no readers-done
                // barrier here.
                if (!last) {
                    // Publish next block into free slot 0 (first use of the
                    // fetch epoch): original A zero-select, clamped W value,
                    // masked DS word and converted header word.
                    a_lds = LDS + IU4_A_OFF;
                    w_lds = LDS + IU4_W0_OFF;
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
#if IU4V3_BUFFER
                        *(u32x2_t*)(a_lds + st_ldsoff[r]) = A_pf[r];
#else
                        *(u32x2_t*)(a_lds + st_ldsoff[r]) = st_aok[r] ? A_pf[r] : z64;
#endif
                        *(u32x2_t*)(w_lds + st_ldsoff[r]) = W_pf[r];
                    }
                    const int h_np = h ^ 1;
                    char* __restrict__ ds_n = LDS + ((h_np == 0) ? IU4_DS_OFF : IU4_DS1_OFF);
                    {
                        const int tok_dsp = bs + (tid >> 1);
                        ((unsigned*)(ds_n + (tid >> 1) * 8))[tid & 1] = (tok_dsp < N) ? ds_nx : 0u;
                    }
                    char* __restrict__ sz_n = LDS + ((h_np == 0) ? IU4_SZ0_OFF : IU4_SZ1_OFF);
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
                // block this guards store-slot reuse of LDS base.
                __syncthreads();
            }
        } // for h
    } // for g
    // Coalesced store: the WMMA C layout leaves lane (m_lane,k_grp) holding
    // acc[nb][rg][j] for token wave_tok0+nb*16+m_lane and rows at 4 B each.
    // Transpose each (rg,nb) 16x16 wave tile through LDS instead: all 8 waves
    // dump in parallel, then all 256 threads stream the 64x32 phase tile.
    //
    // Full tiles use one tile-relative buffer descriptor. Its base and each
    // phase stride stay uniform in SGPRs; the only vector byte offset is the
    // lane's (x*M+i), formed once here instead of rebuilding a 64-bit global
    // address for every store. Edge tiles retain the original checked path.
    float* slots = (float*)LDS;
    const int store_i = tid & 15;
    const int store_x = (tid >> 4) & 15;
    if constexpr (SILU) {
        // Wave (wt_pair p, tok_half t) holds gate (rg 0) and up (rg 1) of h
        // rows hs + 16p + 8*k_grp + j. Four phases (nb) of 64 h rows x 32
        // tokens, streamed like the SET phases with h's own row stride M.
        const unsigned h_lane_off =
            ((unsigned)store_x * (unsigned)M + (unsigned)store_i) * sizeof(float);
        float* const h_tile = Y + (long long)bs * M + hs;
        const auto h_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            h_tile, 0, 0xffffffffu, 0x31004000u);
        const bool full_cols = bs + IU4_BM <= N;
#if defined(IU4_G12_RASTER)
        // Wide h epilogue: per nb, all waves dump their 16 h rows x 16 tokens
        // into a [32 tok][68] f32 tile (8704 B), then 16 lanes per token
        // column write its 64 h rows (256 contiguous B) as b128 stores.
        // Same IU4_SILU_MUL values; only the store shape changes.
        if (full_cols && (M & 3) == 0 && (reinterpret_cast<uintptr_t>(Y) & 15u) == 0) {
            const unsigned dump_tok = (unsigned)(tok_half * 16 + m_lane);
            const unsigned wide_h_off =
                ((unsigned)(tid >> 4) * (unsigned)M + (unsigned)(tid & 15) * 4u) * sizeof(float);
            #pragma unroll
            for (int nb = 0; nb < 4; nb++) {
                #pragma unroll
                for (int c = 0; c < 2; c++) {
                    float4 v4;
                    v4.x = IU4_SILU_MUL(acc[nb][0][4 * c + 0], acc[nb][1][4 * c + 0]);
                    v4.y = IU4_SILU_MUL(acc[nb][0][4 * c + 1], acc[nb][1][4 * c + 1]);
                    v4.z = IU4_SILU_MUL(acc[nb][0][4 * c + 2], acc[nb][1][4 * c + 2]);
                    v4.w = IU4_SILU_MUL(acc[nb][0][4 * c + 3], acc[nb][1][4 * c + 3]);
                    *(float4*)(slots + dump_tok * 68u
                        + (unsigned)(wt_pair * 16 + 8 * k_grp + 4 * c)) = v4;
                }
                __syncthreads();
                #pragma unroll
                for (int it = 0; it < 2; it++) {
                    // Token column it*64 + nb*16 + (tid>>4), h rows (tid&15)*4..+3.
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
#endif
        #pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                slots[(unsigned)(wave * 320 + (8 * k_grp + j) * 20 + m_lane)] =
                    IU4_SILU_MUL(acc[nb][0][j], acc[nb][1][j]);
            }
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                const int p = k >> 1;
                const int h = k & 1;
                const int owner = (p << 1) | h;
                const float v = slots[(unsigned)(owner * 320 + store_i * 20 + store_x)];
                if (full_cols) {
                    const unsigned phase_off =
                        ((unsigned)(h * 64 + nb * 16) * (unsigned)M + (unsigned)(p * 16))
                        * sizeof(float);
                    __builtin_amdgcn_raw_buffer_store_b32(
                        __builtin_bit_cast(unsigned, v), h_rsrc, h_lane_off, phase_off, 0);
                } else {
                    const int col = bs + h * 64 + nb * 16 + store_x;
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
    const bool full_tile = rs + IU4_BN <= M && bs + IU4_BM <= N;
#if defined(IU4_G12_RASTER)
    // Wide epilogue (see header). Tile [32 tok][132] f32 = 16896 B over the
    // dead compute planes; the loop's final barrier already fenced them.
    // Plain float4 accesses (no ext-vector bit casts: clang 23 lowers
    // __builtin_bit_cast(T, v[i]) to element 0).
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
                // Token column (it>>1)*64 + nb*16 + (it&1)*8 + wave, rows lane*4..+3.
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
#endif
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
                    const int h = k & 1;
                    const int owner = (p << 1) | h;
                    const float v = slots[(unsigned)(owner * 320 + store_i * 20 + store_x)];
                    const unsigned phase_off =
                        ((unsigned)(h * 64 + nb * 16) * (unsigned)M
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
                    const int p = k >> 1;    // row-pair 0..3
                    const int h = k & 1;     // token-half 0..1
                    const int row_rel = p * 32 + rg * 16 + store_i;
                    const int col_rel = h * 64 + nb * 16 + store_x;
                    const int owner = (p << 1) | h;
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


extern "C" __global__ __launch_bounds__(IU4_NTHREADS, 2)
void gemm_mq4g256v2_residual_mmq_iu4_full_add_v3(
    const char*           __restrict__ A,
    const block_i4_128*   __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int
) {
    gemm_mq4g256v2_residual_mmq_iu4_gfx12_body<IU4_EPI_ADD>(
        A, nullptr, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(IU4_NTHREADS, 2)
void gemm_mq4g256v2_residual_mmq_iu4_full_set_v3(
    const char*           __restrict__ A,
    const block_i4_128*   __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int
) {
    gemm_mq4g256v2_residual_mmq_iu4_gfx12_body<IU4_EPI_SET>(
        A, nullptr, Xq, Y, M, K, N);
}

// F1-lite gate/up SwiGLU: G/U = gate/up MQ4V2 rows (M each, M % 64 == 0),
// H = h [N][M]. Grid [2M/128, ceil(N/128)], block [256], LDS 20480.
extern "C" __global__ __launch_bounds__(IU4_NTHREADS, 2)
void gemm_mq4g256v2_gate_up_silu_mmq_iu4_v3(
    const char*           __restrict__ G,
    const char*           __restrict__ U,
    const block_i4_128*   __restrict__ Xq,
    float*                __restrict__ H,
    int M, int K, int N
) {
    gemm_mq4g256v2_residual_mmq_iu4_gfx12_body<IU4_EPI_SILU>(
        G, U, Xq, H, M, K, N);
}

#undef IU4_SILU_MUL

#else


extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_add_v3(
    const char*, const void*, float*, int, int, int, int
) {}

extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_set_v3(
    const char*, const void*, float*, int, int, int, int
) {}

extern "C" __global__ void gemm_mq4g256v2_gate_up_silu_mmq_iu4_v3(
    const char*, const char*, const void*, float*, int, int, int
) {}

#endif