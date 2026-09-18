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
//   - Per 128-K block (kb): the activation slab (128 tokens x 64 K nibbles
//     = 32 B/row) AND the weight slab (128 rows x 32 B/row) are staged ONCE
//     in LDS and consumed by all 8 waves. Staged A bytes feed 4 row-pair
//     waves x 4 (slab,sb) reads; staged W bytes feed 2 token-half waves x
//     4 nb x 4 reads.
//   - K-loop body per (slab, K32-sub-block) = ds_read_b64 (W frag) +
//     ds_read_b64 (A frag) + v_wmma_i32_16x16x32_iu4; the per-128-K fold
//     (IU4_FOLD_RN, 4 chunks of the PREVIOUS half delayed one half and
//     pair-ordered among this half's slab-0 bundles) is the ONLY other
//     K-loop VALU. All fragment addresses are hoisted to kernel top
//     (frA/frW tables); slab/sb/nb/rg are unroll constants, so the K-step
//     carries no dynamic address math.
//   - Double-buffered slab staging (Change A): the W (weight) slab is
//     double-buffered per 64-K slab (W0/W1); the A (activation) slab stays
//     single-buffered (A) with v2's publish fence. DS and SZ metadata
//     ping-pong per half-block (plane[h], kb = 2g+h) while token (d,s) is
//     re-read from the DS plane at fold time (same values, same DAG).
//     Steady state per half: prefetch slab 1 into A_pf/W_pf, delayed fold
//     of the previous half's rg0 pairs, rg-split slab-0 bundles for rg0
//     (FIRST restart), delayed fold of the previous half's rg1 pairs,
//     rg-split slab-0 bundles for rg1, B0 (slab-0 A-reads done), publish
//     slab 1 to (A,W1), refill A_pf with next half's A slab 0, one barrier,
//     full slab-1 bundles, cooperatively stage next half's W0/DS/SZ
//     straight to LDS, B2a (slab-1 A-reads done), publish A_pf to A, one
//     barrier (B2b). The last half's fold drains after the loop (no next
//     half to overlap with). Four barriers per half; slab 1 loads issue
//     above slab 0's WMMAs (sched_barrier-pinned, as in v2). Bit-identical:
//     same K order, same pinned IU4_FOLD_RN DAG. Single cacc bank (a
//     runtime-selected dual bank cndmasks every access and spills); the
//     pair order is correctness by construction (same-bank WAR), needing
//     no scheduler pinning. LDS 19456 B keeps 3 WG/CU on gfx1201.
//   - Coalesced store (fp8 K2 mechanism): each (rg, nb) phase dumps the
//     wave's 16x16 f32 sub-tile to an LDS slot ([16][20], stride-20 pad for
//     bank-clean streaming), then all 256 threads stream 64 rows x 32 cols
//     with consecutive threads on consecutive rows (128 B segments).
//     Per-element arithmetic is unchanged, hence bit-identical.
//   - Math is bit-identical to v1: same 4xK32 grouping per 128-K block per
//     column block (same K order), same pinned IU4_FOLD_RN DAG. Integer MMA:
//     any oracle difference is a bug.
//
// LDS layout (19456 B = 19 KiB on gfx1201; compute planes and store slots
// overlap — A/W/DS/SZ are all dead once folds complete):
//   A  = [128][40] activation nibbles, 32 data + 8 pad  @ 0      (5120 B)
//   W0 = [128][40] weight nibbles, 32 data + 8 pad      @ 5120   (5120 B)
//   W1 = [128][40] second slab buffer                  @ 10240  (5120 B)
//   DS0 = [128][8] token (d f32, s i32), even halves    @ 15360  (1024 B)
//   DS1 = [128][8] token (d f32, s i32), odd halves     @ 16384  (1024 B)
//   SZ0 = [128][8] row (sc f32, zp f32), even halves    @ 17408  (1024 B)
//   SZ1 = [128][8] row (sc f32, zp f32), odd halves     @ 18432  (1024 B)
//   end                                                         = 19456 B
//   store slots = 8 waves x [16][20] f32 @ 0                    (10240 B)
//
// Bank enumeration (32 banks x 4 B; lane reads 8 contiguous bytes at
// R*40 + C, R = lane row, C = unroll const): bank start = (10*R + C/4)
// mod 32. 10*R mod 32 over 16 consecutive R is 16 distinct EVEN banks, and
// the 8 B read covers the adjacent odd bank: every bank exactly once per
// 16-lane half, twice per full wave (k_grp halves differ by an even
// constant) = the width floor, no conflict. Applies to A reads (token
// rows), W reads (weight rows), and 4 B DS/SZ reads (banks 2*R: 16
// distinct evens). Staging writes are consecutive dwords (sequential
// banks). Store stream reads: bank = 5*i + x/4, 5 coprime 32 (2-way floor
// across x-pairs); global stores are 2x64 B segments per wave32.
//
// Grid: [ceil(M/128), ceil(N/128)]. Block: [256,1,1]. LDS: 19456 bytes.
//
// Sweep note: 64x256 / 256x64 aspects need a different wave map (wt_pair /
// tok_half split) plus matching host grid/LDS; the fold/prefetch/store
// mechanisms transfer unchanged. Host (gemm.rs launch arm) and the oracle
// launcher hardcode the same 128/128/256-thread/19456 contract — a geometry
// change must update all three consistently.
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
// Packed-row stride: BK/2 data bytes (64 nibbles) + 8 pad. Stride 40 gives
// bank starts 10*r (see bank enumeration above).
#define IU4_STRIDE (IU4_BK / 2 + 8)
#define IU4_A_OFF 0
#define IU4_A_BYTES (IU4_BM * IU4_STRIDE)
#define IU4_W0_OFF (IU4_A_OFF + IU4_A_BYTES)
#define IU4_W_BYTES (IU4_BN * IU4_STRIDE)
#define IU4_W1_OFF (IU4_W0_OFF + IU4_W_BYTES)
#define IU4_DS_OFF (IU4_W1_OFF + IU4_W_BYTES)
#define IU4_DS_BYTES (IU4_BM * 8)
#define IU4_DS1_OFF (IU4_DS_OFF + IU4_DS_BYTES)
#define IU4_SZ0_OFF (IU4_DS1_OFF + IU4_DS_BYTES)
#define IU4_SZ_BYTES (IU4_BN * 8)
#define IU4_SZ1_OFF (IU4_SZ0_OFF + IU4_SZ_BYTES)
#define IU4_LDS_BYTES (IU4_SZ1_OFF + IU4_SZ_BYTES)
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

// One K32-sub-block compute bundle: 2 W-frags + 4 A-frags +
// 8 WMMAs (2 row-groups x 4 column blocks), then a scheduling fence.
// SB is a template constant (folded into LDS immediates); FIRST picks
// the shared zero C vector for each chain's first step (avoids 64 separate
// clears per block). Compute reads slab sb of A from the single A plane and
// of W from Wn (n = sb); the caller re-points w_lds per phase. Both W planes
// share the [128][40] layout, so the hoisted frA/frW offsets apply to either.
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
                    false, a, true, b, zc, false
                );
            } else {
                cacc[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                    false, a, true, b, cacc[nb][rg], false
                );
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

// One row-group slice of a compute bundle: the 4 column blocks of RG for one
// K32 sub-block (4 WMMAs). Lets the K-loop order each rg's fold strictly
// before the bundles that overwrite that rg (same-bank WAR), while the other
// rg's fold overlaps freely around them. Same fragments, same chain order
// per pair as iu4_bundle (FIRST restarts the pair from zc); only the rg loop
// is hoisted to the caller. Adds one W-frag reload per rg half (the shared
// af load is per-call now) — LDS broadcast traffic, no extra live values.
template <int SB, bool FIRST, int RG>
static __device__ __forceinline__ void iu4_bundle_rg(
    const char* __restrict__ w_lds,
    const char* __restrict__ a_lds,
    const unsigned (&frW)[2][2],
    const unsigned (&frA)[4][2],
    int32x8_t cacc[4][2],
    const int32x8_t& zc
) {
    const int32x2_t a = *(const int32x2_t*)(w_lds + frW[RG][SB]);
    const int32x2_t bf0 = *(const int32x2_t*)(a_lds + frA[0][SB]);
    const int32x2_t bf1 = *(const int32x2_t*)(a_lds + frA[1][SB]);
    const int32x2_t bf2 = *(const int32x2_t*)(a_lds + frA[2][SB]);
    const int32x2_t bf3 = *(const int32x2_t*)(a_lds + frA[3][SB]);
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        const int32x2_t b = (nb == 0) ? bf0 : (nb == 1) ? bf1 : (nb == 2) ? bf2 : bf3;
        if constexpr (FIRST) {
            cacc[nb][RG] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                false, a, true, b, zc, false
            );
        } else {
            cacc[nb][RG] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                false, a, true, b, cacc[nb][RG], false
            );
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

// One fold chunk: row-group RG, column blocks [NB0, NB1) of the single cacc
// bank through the pinned IU4_FOLD_RN DAG (bit-identical to the inline fold
// it replaces: same values, same per-element DAG order). The full per-half
// fold is 4 chunks (2 RG x 2 nb-pairs); the K-loop issues each rg's chunks
// delayed by one half, strictly before the bundles that overwrite that rg —
// so the rg0 fold overlaps the previous half's tail/prefetch and the rg1
// fold overlaps this half's rg0 WMMAs. Planes/acc come by pointer/reference
// (no copies); every index is an unroll constant.
template <int RG, int NB0, int NB1>
static __device__ __forceinline__ void iu4_fold_chunk(
    const int32x8_t (&bank)[4][2],
    const char* __restrict__ ds_lds,
    const char* __restrict__ sz_lds,
    float8_t (&acc)[4][2],
    int wave_tok0,
    int wave_row0,
    int m_lane,
    int k_grp
) {
    // Per-rg (sc,zp) row arrays, shared across the chunk's nb-pair (same
    // addresses/values/DAG order as the original inline fold).
    float sc_row[8];
    float zp_row[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const unsigned jbase = (unsigned)(wave_row0 + RG * 16 + 8 * k_grp + j) * 8u;
        sc_row[j] = *(const float*)(sz_lds + jbase);
        zp_row[j] = *(const float*)(sz_lds + jbase + 4u);
    }
    #pragma unroll
    for (int nb = NB0; nb < NB1; nb++) {
        const int* acc_i = (const int*)&bank[nb][RG];
        const unsigned tbase = (unsigned)(wave_tok0 + nb * 16 + m_lane) * 8u;
        const float d_x = *(const float*)(ds_lds + tbase);
        const int s_x = *(const int*)(ds_lds + tbase + 4u);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            IU4_FOLD_RN(acc[nb][RG][j], sc_row[j], zp_row[j], d_x, acc_i[j], s_x);
        }
    }
}

// Shared compute body for residual / full_add / full_set. `ADD` selects
// += (residual) vs = (set) writeback.
template <bool ADD>
static __device__ __forceinline__ void gemm_mq4g256v2_residual_mmq_iu4_gfx12_body(
    const char*           __restrict__ A,   // [M × groups_per_row × 136]
    const block_i4_128*   __restrict__ Xq,  // [K/128 × N]
    float*                __restrict__ Y,   // [N × M] column-major
    int M, int K, int N
) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int rs = blockIdx.x * IU4_BN;
    const int bs = blockIdx.y * IU4_BM;

    if (rs >= M || bs >= N) return;

    const int m_lane = lane & 15;  // 0..15: row within a 16-row group
    const int k_grp  = lane >> 4;  // 0/1: which K-half this lane carries
    const int wt_pair = wave >> 1;     // 0..3: which 32-row group
    const int tok_half = wave & 1;     // 0/1: which 64-token half
    const int wave_row0 = wt_pair * 32;
    const int wave_tok0 = tok_half * 64;

    const int groups_per_row = K / 256;
    extern __shared__ char LDS[];
    // A slab base is fixed (single buffer); W base re-points W0 <-> W1.
    char* __restrict__ a_lds = LDS + IU4_A_OFF;
    char* __restrict__ w_lds = LDS + IU4_W0_OFF;
    // DS/SZ ping-pong bases (plane[h], kb = 2g+h) selected inside the loop.
    // Hoisted K-loop addresses: full LDS byte offsets of every fragment this
    // lane will ever read. Computed once at kernel top (outside ALL loops);
    // the K-step indexes them with unroll constants only.
    unsigned frA[4][2];
    unsigned frW[2][2];
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        const unsigned tok = (unsigned)(wave_tok0 + nb * 16 + m_lane);
        #pragma unroll
        for (int sb = 0; sb < 2; sb++) {
            frA[nb][sb] = tok * (unsigned)IU4_STRIDE
                + (unsigned)sb * 16u + (unsigned)k_grp * 8u;
        }
    }
    #pragma unroll
    for (int rg = 0; rg < 2; rg++) {
        const unsigned row = (unsigned)(wave_row0 + rg * 16 + m_lane);
        #pragma unroll
        for (int sb = 0; sb < 2; sb++) {
            frW[rg][sb] = row * (unsigned)IU4_STRIDE
                + (unsigned)sb * 16u + (unsigned)k_grp * 8u;
        }
    }

    // Output accumulators (gfx12 wave32 layout):
    // acc[nb][rg][j] = Y[wave_tok0 + nb*16 + m_lane][rs + wave_row0 + rg*16 + 8*k_grp + j].
    float8_t acc[4][2];
    // b64-coalesced staging map. Each lane stages 2 rounds; round r covers 64
    // slab rows (r*64 + (tid>>2)) with quad q = tid&3 (8 contiguous bytes) per
    // lane: 256 lanes x 2 rounds x 8 B = one 4 KiB slab exactly once. Lanes in
    // a quad-group share a row, so consecutive lanes read consecutive quads
    // (coalesced global segments) and write sequential LDS banks (same 2-way
    // floor as the old map). b64 is the width ceiling, NOT b128: every 16 B
    // chunk of the Xq (72 B blocks) and W (136 B groups) nibble arrays sits at
    // 8-mod-16 behind an 8-byte header, and stride-40 LDS rows alternate
    // 0/8-mod-16 — b128 is alignment-impossible on both sides, b64 is exact.
    // All offsets below are loop-invariant (bs/rs/tid only), computed once at
    // kernel top; per-trip SGPR bases (Xq + kb*N, A + g*136 + ...) carry kb/g/h
    // so the K-loop issues base(SGPR) + offset(VGPR) + imm wide loads/stores
    // with zero per-load address VALU. OOB rows clamp to the last valid index
    // (same addresses the old ternaries produced); A values additionally zero-
    // select (exact old semantics) via branch-free ternary (cndmask, no EXEC
    // masking), while W keeps the clamped value exactly as before.
    unsigned st_avoff[2];
    unsigned st_wvoff[2];
    unsigned st_ldsoff[2];
    unsigned st_aok[2];
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        const int tok_r = bs + r * 64 + (tid >> 2);
        const int tok_c = (tok_r < N) ? tok_r : (N - 1);
        st_aok[r] = (tok_r < N) ? 1u : 0u;
        st_avoff[r] = (unsigned)tok_c * (unsigned)sizeof(block_i4_128)
            + (unsigned)(tid & 3) * 8u;
        const int grow_r = rs + r * 64 + (tid >> 2);
        const int src_c = (grow_r < M) ? grow_r : (M - 1);
        st_wvoff[r] = (unsigned)src_c * (unsigned)groups_per_row * 136u
            + (unsigned)(tid & 3) * 8u;
        st_ldsoff[r] = (unsigned)(r * 64 + (tid >> 2)) * (unsigned)IU4_STRIDE
            + (unsigned)(tid & 3) * 8u;
    }
    const u32x2_t z64 = { 0u, 0u };
    for (int nb = 0; nb < 4; nb++) {
        #pragma unroll
        for (int rg = 0; rg < 2; rg++) {
            acc[nb][rg] = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
    // One reusable zero C vector for each chain's first WMMA.
    const int32x8_t zc = (int32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
    // Single int32 accumulator bank: the fold of pair-group rg is ordered
    // strictly before the bundles that overwrite rg (same-bank WAR the
    // scheduler must respect), while the other rg's fold overlaps freely.
    // Pair-ordered delayed fold gives WMMA/VALU overlap with no ping-pong
    // bank (a runtime-selected VGPR bank cndmasks every access) and no
    // scheduler pinning.
    int32x8_t cacc[4][2];
    // Slab prefetch registers (slab s+1 globals while computing slab s): b64
    // pairs, same 8 dwords/lane as before (2 rounds x 1 u32x2 per plane).
    u32x2_t A_pf[2];
    u32x2_t W_pf[2];

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
    // Prologue slab 0 of block 0, b64-coalesced (same map as the K-loop).
    {
        const char* __restrict__ xqB0 = (const char*)Xq + 8;
        const char* __restrict__ wB0 = A + 8;
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            const u32x2_t av = *(const u32x2_t*)(xqB0 + st_avoff[r]);
            *(u32x2_t*)(a_lds + st_ldsoff[r]) = st_aok[r] ? av : z64;
            *(u32x2_t*)(w_lds + st_ldsoff[r]) = *(const u32x2_t*)(wB0 + st_wvoff[r]);
        }
    }
    __syncthreads();
    // g/h rolled (unroll_count(1)): unrolling keeps block k's staging+fold
    // temporaries live across blocks and blows VGPRs past 256 with spills
    // (the scheduler overlaps the whole unrolled window). Rolled, only
    // cacc/acc carry across iterations. Pair order inside is still manual
    // (chunk/bundle adjacency in program order); the small rolled window
    // keeps pressure low so the scheduler can actually overlap them.
    #pragma clang loop unroll_count(1)
    for (int g = 0; g < groups_per_row; g++) {
        #pragma clang loop unroll_count(1)
        for (int h = 0; h < 2; h++) {
            const int kb = 2 * g + h;
            const bool last = (h == 1) && (g + 1 == groups_per_row);
            const bool have_prev = !(g == 0 && h == 0);
            // Prev-half DS/SZ planes (half h^1). NOTE: this trip's own
            // staging targets these same planes, so the folds below must
            // complete their reads first — guaranteed: folds sit in slab 0,
            // staging sits after slab-1 compute, with B1 between.
            char* __restrict__ ds_prev = LDS + ((h == 0) ? IU4_DS1_OFF : IU4_DS_OFF);
            char* __restrict__ sz_prev = LDS + ((h == 0) ? IU4_SZ1_OFF : IU4_SZ0_OFF);
            // --- Slab 0 compute (A, W0) with slab 1 prefetch as producer. ---
            {
                // Slab 1 prefetch, b64-coalesced: uniform SGPR bases carry
                // kb/g/h, lane VGPR offsets were hoisted to kernel top.
                // Issues first so the rg0 fold below covers its latency.
                const char* __restrict__ xqB1 = (const char*)(Xq + (long long)kb * N) + 8 + 32;
                const char* __restrict__ wB1 = A + (long long)g * 136 + 8 + (long long)h * 64 + 32;
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    const u32x2_t av = *(const u32x2_t*)(xqB1 + st_avoff[r]);
                    A_pf[r] = st_aok[r] ? av : z64;
                    W_pf[r] = *(const u32x2_t*)(wB1 + st_wvoff[r]);
                }
                // Delayed fold of the previous half's rg0 pairs. Strictly
                // before the rg0 bundles below (same-bank WAR the scheduler
                // must respect); overlaps the prefetch above. Every 128-K
                // half restarts its int32 accumulation from zero (each half
                // folds independently): FIRST=true on slab 0 of every half.
                if (have_prev) {
                    iu4_fold_chunk<0, 0, 2>(cacc, ds_prev, sz_prev, acc, wave_tok0, wave_row0, m_lane, k_grp);
                    iu4_fold_chunk<0, 2, 4>(cacc, ds_prev, sz_prev, acc, wave_tok0, wave_row0, m_lane, k_grp);
                    __builtin_amdgcn_sched_group_barrier(2, 64, 0);
                }
                iu4_bundle_rg<0, true, 0>(w_lds, a_lds, frW, frA, cacc, zc);
                iu4_bundle_rg<1, false, 0>(w_lds, a_lds, frW, frA, cacc, zc);
                __builtin_amdgcn_sched_group_barrier(8, 8, 0);
                // Delayed fold of the previous half's rg1 pairs. Independent
                // of the rg0 bundles above (different pairs) so it overlaps
                // them; strictly before the rg1 bundles below (same-bank WAR).
                if (have_prev) {
                    iu4_fold_chunk<1, 0, 2>(cacc, ds_prev, sz_prev, acc, wave_tok0, wave_row0, m_lane, k_grp);
                    iu4_fold_chunk<1, 2, 4>(cacc, ds_prev, sz_prev, acc, wave_tok0, wave_row0, m_lane, k_grp);
                    __builtin_amdgcn_sched_group_barrier(2, 64, 0);
                }
                iu4_bundle_rg<0, true, 1>(w_lds, a_lds, frW, frA, cacc, zc);
                iu4_bundle_rg<1, false, 1>(w_lds, a_lds, frW, frA, cacc, zc);
                __builtin_amdgcn_sched_group_barrier(8, 8, 0);
                // B0: slab-0 A-reads done on every wave (A is single-buffered:
                // the publish below overwrites slab 0 with slab 1). W1 writes
                // need no fence but share it.
                __syncthreads();
                // Publish slab 1 (A_pf -> A plane, W_pf -> W1) and refill
                // A_pf with the next half's A slab 0 (guarded; the loads
                // issue above compute 1 + the next half's fold chunks and
                // overlap them; the publish to A waits for B2a). Re-point
                // W at W1 (A is fixed).
                w_lds = LDS + IU4_W1_OFF;
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    *(u32x2_t*)(a_lds + st_ldsoff[r]) = A_pf[r];
                    *(u32x2_t*)(w_lds + st_ldsoff[r]) = W_pf[r];
                }
                if (!last) {
                    // Next half's A slab 0 (from Xq; only kb_n0 needed).
                    const int kb_n0 = kb + 1;
                    const char* __restrict__ xqBn0 = (const char*)(Xq + (long long)kb_n0 * N) + 8;
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        const u32x2_t av = *(const u32x2_t*)(xqBn0 + st_avoff[r]);
                        A_pf[r] = st_aok[r] ? av : z64;
                    }
                }
                // B1: slab-1 publish visible to all waves before compute 1.
                // Also separates the folds above from the staging below (the
                // folds read planes h^1, staging writes planes h^1).
                __syncthreads();
            }
            // --- Slab 1 compute (A, W1), next-half stage. ---
            // Previous half fully folded in slab 0 above; these bundles are
            // unconstrained.
            {
                iu4_bundle<0, false>(w_lds, a_lds, frW, frA, cacc, zc);
                iu4_bundle<1, false>(w_lds, a_lds, frW, frA, cacc, zc);
                if (!last) {
                    const int g_n = g + h;
                    const int h_n = h ^ 1;
                    // Next half's W slab 0 staged straight to W0
                    // (cooperative global -> LDS): W0 reads completed at B1
                    // on every wave; values flow through transient registers
                    // only. Re-point W at W0.
                    w_lds = LDS + IU4_W0_OFF;
                    const char* __restrict__ wB0n = A + (long long)g_n * 136 + 8 + (long long)h_n * 64;
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        *(u32x2_t*)(w_lds + st_ldsoff[r]) = *(const u32x2_t*)(wB0n + st_wvoff[r]);
                    }
                }
                if (!last) {
                    const int g_n2 = g + h;
                    const int h_n2 = h ^ 1;
                    const int kb_n2 = kb + 1;
                    // DS/SZ planes for the next half (opposite parity: plane
                    // h^1 — folded at the next half's slab 0, separated from
                    // these writes by B1 and the fold-before-need order).
                    // DS plane for the next half, clamp-not-branch (same
                    // values as the old ternary: clamped address, zero value).
                    char* __restrict__ ds_n = LDS + ((h_n2 == 0) ? IU4_DS_OFF : IU4_DS1_OFF);
                    {
                        const int tok_ds = bs + (tid >> 1);
                        const int w_ds = tid & 1;
                        const int tok_dsc = (tok_ds < N) ? tok_ds : (N - 1);
                        const unsigned dw = ((const unsigned*)(Xq + (long long)kb_n2 * N + tok_dsc))[w_ds];
                        ((unsigned*)(ds_n + (tid >> 1) * 8))[w_ds] = (tok_ds < N) ? dw : 0u;
                    }
                    // SZ plane for the next half (opposite parity, same
                    // separation as DS above), clamp-not-branch.
                    char* __restrict__ sz_n = LDS + ((h_n2 == 0) ? IU4_SZ0_OFF : IU4_SZ1_OFF);
                    {
                        const int row_sz = rs + (tid >> 1);
                        const int w_sz = tid & 1;
                        const int src_sz = (row_sz < M) ? row_sz : (M - 1);
                        const unsigned hw = *(const unsigned*)(A + (long long)src_sz * groups_per_row * 136
                            + g_n2 * 136 + h_n2 * 4);
                        const float f = (w_sz == 0)
                            ? __half2float(__ushort_as_half((unsigned short)(hw & 0xFFFFu)))
                            : __half2float(__ushort_as_half((unsigned short)(hw >> 16)));
                        ((float*)(sz_n + (tid >> 1) * 8))[w_sz] = (row_sz < M) ? f : 0.0f;
                    }
                }
                // B2a: slab-1 A-reads done on every wave, so the single A
                // plane may be republished with next slab 0.
                __syncthreads();
                if (!last) {
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        *(u32x2_t*)(a_lds + st_ldsoff[r]) = A_pf[r];
                    }
                }
                // B2b: next-half A/W0/DS/SZ publish visible to all waves
                // (entry condition of the next half); on the last half it
                // guards the store-slot reuse of LDS base below.
                __syncthreads();
            }
        } // for h
    } // for g
    // Drain: fold of the last half-block (always h=1, planes 1). Every
    // earlier half folded at the next half's slab 0; this one has no next
    // half. Guarded for groups_per_row == 0 (loop skipped, cacc unwritten
    // — acc stays zero exactly as before). Before the store-slot LDS reuse
    // below (B2b-last already ordered the planes).
    if (groups_per_row > 0) {
        char* __restrict__ ds_d = LDS + IU4_DS1_OFF;
        char* __restrict__ sz_d = LDS + IU4_SZ1_OFF;
        iu4_fold_chunk<0, 0, 2>(cacc, ds_d, sz_d, acc, wave_tok0, wave_row0, m_lane, k_grp);
        iu4_fold_chunk<0, 2, 4>(cacc, ds_d, sz_d, acc, wave_tok0, wave_row0, m_lane, k_grp);
        iu4_fold_chunk<1, 0, 2>(cacc, ds_d, sz_d, acc, wave_tok0, wave_row0, m_lane, k_grp);
        iu4_fold_chunk<1, 2, 4>(cacc, ds_d, sz_d, acc, wave_tok0, wave_row0, m_lane, k_grp);
    }
    // Coalesced store: the WMMA C layout leaves lane (m_lane,k_grp) holding
    // acc[nb][rg][j] for token wave_tok0+nb*16+m_lane and rows
    // rows at 4 B each. Transpose each (rg,nb) 16x16 wave tile through LDS
    // instead: all 8 waves dump in parallel (one [16][20] slot each; the
    // stride-20 pad keeps streaming reads bank-clean), then all 256 threads
    // stream the 64x32 phase tile with thread (i=tid&15, x=(tid>>4)&15)
    // covering rows (consecutive threads = consecutive rows = 128 B global
    // segments) and k covering 4 row-pairs x 2 token-halves. Per-element
    // arithmetic is unchanged (one residual add), hence bit-identical; only
    // the slots reuse LDS base: 8 x 320 floats = 10240 B < 19456 B.
    float* slots = (float*)LDS;
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
                const int i = tid & 15;
                const int x = (tid >> 4) & 15;
                const int row_rel = p * 32 + rg * 16 + i;
                const int col_rel = h * 64 + nb * 16 + x;
                const int owner = (p << 1) | h;
                const float v = slots[(unsigned)(owner * 320 + i * 20 + x)];
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

// residual: += (the `add` kernarg picks += vs = at runtime).
extern "C" __global__ __launch_bounds__(IU4_NTHREADS, 2)
void gemm_mq4g256v2_residual_mmq_iu4(
    const char*           __restrict__ A,
    const block_i4_128*   __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int add
) {
    if (add) {
        gemm_mq4g256v2_residual_mmq_iu4_gfx12_body<true>(A, Xq, Y, M, K, N);
    } else {
        gemm_mq4g256v2_residual_mmq_iu4_gfx12_body<false>(A, Xq, Y, M, K, N);
    }
}

extern "C" __global__ __launch_bounds__(IU4_NTHREADS, 2)
void gemm_mq4g256v2_residual_mmq_iu4_full_add(
    const char*           __restrict__ A,
    const block_i4_128*   __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int
) {
    gemm_mq4g256v2_residual_mmq_iu4_gfx12_body<true>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(IU4_NTHREADS, 2)
void gemm_mq4g256v2_residual_mmq_iu4_full_set(
    const char*           __restrict__ A,
    const block_i4_128*   __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int
) {
    gemm_mq4g256v2_residual_mmq_iu4_gfx12_body<false>(A, Xq, Y, M, K, N);
}

#else

// Stubs for non-gfx12 archs (the gfx11 iu4 kernel and gfx906 dp4a MMQ kernel
// live in their own arch guards / files).
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4(
    const char*, const void*, float*, int, int, int, int
) {}

extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_add(
    const char*, const void*, float*, int, int, int, int
) {}

extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_set(
    const char*, const void*, float*, int, int, int, int
) {}

#endif