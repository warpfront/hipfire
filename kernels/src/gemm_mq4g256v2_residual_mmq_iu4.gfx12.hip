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
//     (IU4_FOLD_RN, once per block after the 2nd slab) is the ONLY other
//     K-loop VALU. All fragment addresses are hoisted to kernel top
//     (frA/frW tables); slab/sb/nb/rg are unroll constants, so the K-step
//     carries no dynamic address math.
//   - Double-buffered slab staging (Change A): the W (weight) slab is
//     double-buffered per 64-K slab (W0/W1); the A (activation) slab stays
//     single-buffered (A) with v2's publish fence. DS and SZ metadata
//     ping-pong per half-block (plane[h], kb = 2g+h) while token (d,s) is
//     re-read from the DS plane at fold time (same values, same DAG).
//     Steady state per block: prefetch slab 1 into A_pf/W_pf, compute slab
//     0 from (A,W0), B0 (slab-0 A-reads done), publish slab 1 to (A,W1),
//     refill A_pf with next block's A slab 0, one barrier, compute slab 1
//     from (A,W1), cooperatively stage next block's W0/DS/SZ straight to
//     LDS, fold, B2a (slab-1 A-reads done), publish A_pf to A, one barrier.
//     Four barriers per block (vs five before); slab 1 loads issue above
//     slab 0's WMMAs (sched_barrier-pinned, as in v2), next-block W staging
//     issues above the fold, and next-block A loads issue above compute 1
//     + fold. Bit-identical: same K order, same pinned IU4_FOLD_RN DAG.
//     LDS 19456 B keeps 3 WG/CU on gfx1201 (3 x 19456 = 58368 < 65536/CU).
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
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        #pragma unroll
        for (int rg = 0; rg < 2; rg++) {
            acc[nb][rg] = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
    // One reusable zero C vector for each chain's first WMMA.
    const int32x8_t zc = (int32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
    int32x8_t cacc[4][2];
    // Slab prefetch registers (slab s+1 globals while computing slab s).
    unsigned A_pf[IU4_BM * 8 / IU4_NTHREADS];
    unsigned W_pf[IU4_BN * 8 / IU4_NTHREADS];

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
    for (int i = tid; i < IU4_BM * 8; i += IU4_NTHREADS) {
        const int row = i >> 3;
        const int dw = i & 7;
        const int tok = bs + row;
        unsigned v = 0u;
        if (tok < N) {
            const block_i4_128* xb = Xq + tok;
            v = ((const unsigned*)(xb->qs))[dw];
        }
        ((unsigned*)(a_lds + row * IU4_STRIDE))[dw] = v;
    }
    for (int i = tid; i < IU4_BN * 8; i += IU4_NTHREADS) {
        const int row = i >> 3;
        const int dw = i & 7;
        const int grow = rs + row;
        const int src_row = (grow < M) ? grow : (M - 1);
        const unsigned v = *(const unsigned*)(A + (long long)src_row * groups_per_row * 136
            + 8 + dw * 4);
        ((unsigned*)(w_lds + row * IU4_STRIDE))[dw] = v;
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
            // --- Slab 0 compute (A, W0) with slab 1 prefetch as producer. ---
            {
                #pragma unroll
                for (int t = 0; t < IU4_BM * 8 / IU4_NTHREADS; t++) {
                    const int i = tid + IU4_NTHREADS * t;
                    const int row = i >> 3;
                    const int dw = i & 7;
                    const int tok = bs + row;
                    unsigned av = 0u;
                    if (tok < N) {
                        const block_i4_128* xb = Xq + (long long)kb * N + tok;
                        av = ((const unsigned*)(xb->qs + 32))[dw];
                    }
                    A_pf[t] = av;
                    const int grow = rs + row;
                    const int src_row = (grow < M) ? grow : (M - 1);
                    W_pf[t] = *(const unsigned*)(A + (long long)src_row * groups_per_row * 136
                        + g * 136 + 8 + h * 64 + 32 + dw * 4);
                }
                // Pin the prefetched loads above the WMMAs.

                __builtin_amdgcn_sched_barrier(0);
                // Every 128-K block restarts its int32 accumulation from zero
                // (each block folds independently): FIRST=true on slab 0 of
                // every (g,h), exactly as in v2. No h-dependence here.
                iu4_bundle<0, true>(w_lds, a_lds, frW, frA, cacc, zc);
                iu4_bundle<1, false>(w_lds, a_lds, frW, frA, cacc, zc);
                // B0: slab-0 A-reads done on every wave (A is single-buffered:
                // the publish below overwrites slab 0 with slab 1). W1 writes
                // need no fence but share it.
                __syncthreads();
                // Publish slab 1 (A_pf -> A plane, W_pf -> W1) and refill
                // A_pf with the next block's A slab 0 (guarded; the loads
                // issue above compute 1 + fold below and overlap them; the
                // publish to A waits for B2a). Re-point W at W1 (A is fixed).
                w_lds = LDS + IU4_W1_OFF;
                #pragma unroll
                for (int t = 0; t < IU4_BM * 8 / IU4_NTHREADS; t++) {
                    const int i = tid + IU4_NTHREADS * t;
                    const int row = i >> 3;
                    const int dw = i & 7;
                    ((unsigned*)(a_lds + row * IU4_STRIDE))[dw] = A_pf[t];
                    ((unsigned*)(w_lds + row * IU4_STRIDE))[dw] = W_pf[t];
                }
                if (!last) {
                    // Next block's A slab 0 (from Xq; only kb_n0 needed).
                    const int kb_n0 = kb + 1;
                    #pragma unroll
                    for (int t = 0; t < IU4_BM * 8 / IU4_NTHREADS; t++) {
                        const int i = tid + IU4_NTHREADS * t;
                        const int row = i >> 3;
                        const int dw = i & 7;
                        const int tok = bs + row;
                        unsigned av = 0u;
                        if (tok < N) {
                            const block_i4_128* xb = Xq + (long long)kb_n0 * N + tok;
                            av = ((const unsigned*)(xb->qs))[dw];
                        }
                        A_pf[t] = av;
                    }
                }
                // B1: slab-1 publish visible to all waves before compute 1.
                __syncthreads();
            }
            // --- Slab 1 compute (A, W1), next-block W/DS/SZ stage, fold. ---
            {
                iu4_bundle<0, false>(w_lds, a_lds, frW, frA, cacc, zc);
                iu4_bundle<1, false>(w_lds, a_lds, frW, frA, cacc, zc);
                if (!last) {
                    const int g_n = g + h;
                    const int h_n = h ^ 1;
                    // Next block's W slab 0 staged straight to W0
                    // (cooperative global -> LDS): W0 reads completed at B1
                    // on every wave; values flow through transient registers
                    // only. Re-point W at W0. These loads issue above the
                    // fold below and overlap its VALU. (Next A slab 0 sits
                    // in A_pf since trip 0; its publish to the single A
                    // plane waits for B2a below.)
                    w_lds = LDS + IU4_W0_OFF;
                    for (int i = tid; i < IU4_BN * 8; i += IU4_NTHREADS) {
                        const int row = i >> 3;
                        const int dw = i & 7;
                        const int grow = rs + row;
                        const int src_row = (grow < M) ? grow : (M - 1);
                        const unsigned wv = *(const unsigned*)(A + (long long)src_row * groups_per_row * 136
                            + g_n * 136 + 8 + h_n * 64 + dw * 4);
                        ((unsigned*)(w_lds + row * IU4_STRIDE))[dw] = wv;
                    }
                }
                // Fold: 4 K32 int32 partials per (nb, rg) through the pinned
                // IU4_FOLD_RN DAG. Per-half (sc,zp) via intra-wave shuffle
                // from the lane owning each output row; (d,s) re-read here
                // from the staged DS plane (identical values, identical DAG).
                #pragma unroll
                for (int rg = 0; rg < 2; rg++) {
                    const unsigned rbase = (unsigned)(wave_row0 + rg * 16 + m_lane) * 8u;
                    const float sc_rg = *(const float*)(sz_lds + rbase);
                    const float zp_rg = *(const float*)(sz_lds + rbase + 4u);
                    float sc_row[8];
                    float zp_row[8];
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const int src_lane = 8 * k_grp + j;
                        sc_row[j] = __shfl(sc_rg, src_lane);
                        zp_row[j] = __shfl(zp_rg, src_lane);
                    }
                    #pragma unroll
                    for (int nb = 0; nb < 4; nb++) {
                        int* acc_i = (int*)&cacc[nb][rg];
                        const unsigned tbase = (unsigned)(wave_tok0 + nb * 16 + m_lane) * 8u;
                        const float d_x = *(const float*)(ds_lds + tbase);
                        const int s_x = *(const int*)(ds_lds + tbase + 4u);
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            IU4_FOLD_RN(acc[nb][rg][j], sc_row[j], zp_row[j], d_x, acc_i[j], s_x);
                        }
                    }
                }
                if (!last) {
                    const int g_n2 = g + h;
                    const int h_n2 = h ^ 1;
                    const int kb_n2 = kb + 1;
                    // DS/SZ planes for the next block (opposite parity: the
                    // fold above read DS[h]/SZ[h], staged here is [h^1]).
                    char* __restrict__ ds_n = LDS + ((h_n2 == 0) ? IU4_DS_OFF : IU4_DS1_OFF);
                    for (int i = tid; i < IU4_BM * 2; i += IU4_NTHREADS) {
                        const int tok_idx = i >> 1;
                        const int w = i & 1;
                        const int tok = bs + tok_idx;
                        const unsigned dw = (tok < N)
                            ? ((const unsigned*)(Xq + (long long)kb_n2 * N + tok))[w]
                            : 0u;
                        ((unsigned*)(ds_n + tok_idx * 8))[w] = dw;
                    }
                    // SZ plane for the next block (opposite parity: the fold
                    // above read SZ[h], staged here is SZ[h^1]).
                    char* __restrict__ sz_n = LDS + ((h_n2 == 0) ? IU4_SZ0_OFF : IU4_SZ1_OFF);
                    for (int i = tid; i < IU4_BN * 2; i += IU4_NTHREADS) {
                        const int row_idx = i >> 1;
                        const int w = i & 1;
                        const int grow = rs + row_idx;
                        const int src_row = (grow < M) ? grow : (M - 1);
                        const unsigned hw = *(const unsigned*)(A + (long long)src_row * groups_per_row * 136
                            + g_n2 * 136 + h_n2 * 4);
                        const float f = (w == 0)
                            ? __half2float(__ushort_as_half((unsigned short)(hw & 0xFFFFu)))
                            : __half2float(__ushort_as_half((unsigned short)(hw >> 16)));
                        ((float*)(sz_n + row_idx * 8))[w] = (grow < M) ? f : 0.0f;
                    }
                }
                // B2a: slab-1 A-reads done on every wave (fold complete), so
                // the single A plane may be republished with next slab 0.
                __syncthreads();
                if (!last) {
                    #pragma unroll
                    for (int t = 0; t < IU4_BM * 8 / IU4_NTHREADS; t++) {
                        const int i = tid + IU4_NTHREADS * t;
                        const int row = i >> 3;
                        const int dw = i & 7;
                        ((unsigned*)(a_lds + row * IU4_STRIDE))[dw] = A_pf[t];
                    }
                }
                // B2b: next-block A/W0/DS/SZ publish visible to all waves
                // (entry condition of the next trip); on the last block it
                // guards the store-slot reuse of LDS base below.
                __syncthreads();
            }
        } // for h
    } // for g
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