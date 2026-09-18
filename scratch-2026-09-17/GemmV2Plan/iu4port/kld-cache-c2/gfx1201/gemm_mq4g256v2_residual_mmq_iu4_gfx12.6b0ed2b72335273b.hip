// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniil Markevich
// hipfire — see LICENSE and NOTICE in the project root.
//
// Shared int4 activation block + wave32 MSE-clip quant recipe for the
// iu4-direct MMQ path. Concatenated by the host JIT into:
//   - gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip (standalone quantize_int4_mmq_ds128)
//   - fused RMSNorm/FWHT and SwiGLU/FWHT IU4 producer sidecars (C2)
//
// Do not edit a second copy of the amax / 8-candidate MSE / rintf+clamp /
// integer-sum path. Every consumer of this fragment must remain bit-identical
// on the 72-byte block_i4_128 layout.

#ifndef HIPFIRE_BLOCK_I4_128_QUANT_HIP
#define HIPFIRE_BLOCK_I4_128_QUANT_HIP

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

// int4 activation block: one 128-K group for one token (column).
// qs: 64 B, even K low nibble / odd K high nibble, two's complement.
// d: per-128 scale (MSE-clip grid argmin). s: exact int sum of the 128 x_q.
struct block_i4_128 {
    float d;
    int s;
    unsigned char qs[64];
};
static_assert(sizeof(block_i4_128) == 72, "bad block size");

// Wave32-synchronous quant of one 128-element group.
// Lane l owns elements 4l..4l+3 in xv[4]. Reductions use offsets 16..1
// (wave64-safe width-32 shuffles). Writes qs for every lane; lane 0 of the
// group publishes d and s. Arithmetic matches the original standalone body
// (amax → 8 MSE candidates with strict-`<` → rintf/clamp → exact int sum).
__device__ __forceinline__ void quantize_block_i4_128_wave(
    const float xv[4],
    block_i4_128* __restrict__ out,
    int lane
) {
    float amax = fmaxf(fmaxf(fabsf(xv[0]), fabsf(xv[1])), fmaxf(fabsf(xv[2]), fabsf(xv[3])));
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }

    float best_d = 1.0f;
    if (amax != 0.0f) {
        float best_mse = 1.0e30f;
        for (int j = 0; j < 8; ++j) {
            const float dj = (amax / 7.0f) * 0.5f * (1.0f + (float)j / 7.0f);
            float mse = 0.0f;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                float q = rintf(xv[e] / dj);
                q = fminf(fmaxf(q, -8.0f), 7.0f);
                const float err = __fmaf_rn(-q, dj, xv[e]);
                mse = __fmaf_rn(err, err, mse);
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                mse += __shfl_xor(mse, offset, 32);
            }
            if (mse < best_mse) {
                best_mse = mse;
                best_d = dj;
            }
        }
    }

    int q4[4];
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        float q = (amax == 0.0f) ? 0.0f : rintf(xv[e] / best_d);
        q = fminf(fmaxf(q, -8.0f), 7.0f);
        q4[e] = (int)q;
    }
    int s = q4[0] + q4[1] + q4[2] + q4[3];
    for (int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_xor(s, offset, 32);
    }

    unsigned char* qb = out->qs;
    qb[lane * 2 + 0] = (unsigned char)((q4[0] & 15) | ((q4[1] & 15) << 4));
    qb[lane * 2 + 1] = (unsigned char)((q4[2] & 15) | ((q4[3] & 15) << 4));
    if (lane == 0) {
        out->d = best_d;
        out->s = s;
    }
}

// Remap producer-final 8 floats/lane into the standalone quant ownership
// (lane l owns 4l..4l+3 of each 128-half) and emit two block_i4_128 rows.
// Producer lane p holds values 8p..8p+7. For half h∈{0,1}:
//   source lane = 16*h + (l >> 1); even l → first float4, odd l → second.
// Sidecar index: (2*group + h)*N + token.
__device__ __forceinline__ void emit_iu4_sidecar_from_producer8(
    float f0, float f1, float f2, float f3,
    float f4, float f5, float f6, float f7,
    block_i4_128* __restrict__ Y,
    int group,
    int token,
    int N,
    int lane
) {
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int src = 16 * h + (lane >> 1);
        float s0 = __shfl(f0, src, 32);
        float s1 = __shfl(f1, src, 32);
        float s2 = __shfl(f2, src, 32);
        float s3 = __shfl(f3, src, 32);
        float s4 = __shfl(f4, src, 32);
        float s5 = __shfl(f5, src, 32);
        float s6 = __shfl(f6, src, 32);
        float s7 = __shfl(f7, src, 32);
        float xv[4];
        if ((lane & 1) == 0) {
            xv[0] = s0; xv[1] = s1; xv[2] = s2; xv[3] = s3;
        } else {
            xv[0] = s4; xv[1] = s5; xv[2] = s6; xv[3] = s7;
        }
        block_i4_128* out = Y + (long long)(2 * group + h) * N + token;
        quantize_block_i4_128_wave(xv, out, lane);
    }
}

// Standalone quantizer: same (X, Y, K, N) grid contract as
// quantize_q8_1_mmq_ds4_x128 (block (256,1,1), each thread 4 floats,
// grid [(K+1023)/1024, N]). One wave32 covers exactly one 128-K group.
#ifndef HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE
extern "C" __global__ void quantize_int4_mmq_ds128(
    const float* __restrict__ X,
    block_i4_128* __restrict__ Y,
    int K,
    int N
) {
    const int token = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int i0 = t * 4;
    if (token >= N || i0 >= K) return;
    const int lane = threadIdx.x & 31;

    const float4 xi = (i0 + 3 < K)
        ? *(const float4*)(X + (long long)token * K + i0)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float xv[4] = {xi.x, xi.y, xi.z, xi.w};

    const int kblock = i0 / 128;
    block_i4_128* out = Y + (long long)kblock * N + token;
    quantize_block_i4_128_wave(xv, out, lane);
}
#endif // HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE

#endif // HIPFIRE_BLOCK_I4_128_QUANT_HIP
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
//   - Register prefetch (fp8 K2 mechanism, re-implemented for packed
//     dwords): prologue stages slab 0; trip 0 issues slab-1 globals into
//     A_pf/W_pf BEFORE trip-0 compute (sched_barrier-pinned), publishes
//     after compute under a publish barrier (fast-wave clobber guard).
//     Single LDS buffer; compute byte-identical.
//   - Coalesced store (fp8 K2 mechanism): each (rg, nb) phase dumps the
//     wave's 16x16 f32 sub-tile to an LDS slot ([16][20], stride-20 pad for
//     bank-clean streaming), then all 256 threads stream 64 rows x 32 cols
//     with consecutive threads on consecutive rows (128 B segments).
//     Per-element arithmetic is unchanged, hence bit-identical.
//   - Math is bit-identical to v1: same 4xK32 grouping per 128-K block per
//     column block (same K order), same pinned IU4_FOLD_RN DAG. Integer MMA:
//     any oracle difference is a bug.
//
// LDS layout (12 KiB; compute_ws and store slots overlap — A/W/DS/SZ are
// all dead once folds complete):
//   A  = [128][40] activation nibbles, 32 data + 8 pad  @ 0      (5120 B)
//   W  = [128][40] weight nibbles, 32 data + 8 pad      @ 5120   (5120 B)
//   DS = [128][8]  token (d f32, s i32) per kb block    @ 10240  (1024 B)
//   SZ = [128][8]  row (sc f32, zp f32) per half        @ 11264  (1024 B)
//   end                                                         = 12288 B
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
// Grid: [ceil(M/128), ceil(N/128)]. Block: [256,1,1]. LDS: 12288 bytes.
//
// Sweep note: 64x256 / 256x64 aspects need a different wave map (wt_pair /
// tok_half split) plus matching host grid/LDS; the fold/prefetch/store
// mechanisms transfer unchanged. Host (gemm.rs launch arm) and the oracle
// launcher hardcode the same 128/128/256-thread/12288 contract — a geometry
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
#define IU4_W_OFF (IU4_A_BYTES)
#define IU4_W_BYTES (IU4_BN * IU4_STRIDE)
#define IU4_DS_OFF (IU4_W_OFF + IU4_W_BYTES)
#define IU4_DS_BYTES (IU4_BM * 8)
#define IU4_SZ_OFF (IU4_DS_OFF + IU4_DS_BYTES)
#define IU4_SZ_BYTES (IU4_BN * 8)
#define IU4_LDS_BYTES (IU4_SZ_OFF + IU4_SZ_BYTES)
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
// clears per block). Single LDS slot: slab 1 overwrites slab 0 on publish,
// so both trips share these addresses. K-step VALU: ds_reads + WMMAs only.
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
    char* __restrict__ a_lds = LDS + IU4_A_OFF;
    char* __restrict__ w_lds = LDS + IU4_W_OFF;
    char* __restrict__ ds_lds = LDS + IU4_DS_OFF;
    char* __restrict__ sz_lds = LDS + IU4_SZ_OFF;

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

    // Token (d, s) for this lane's 4 column blocks, per kb block (live
    // across both slabs; loaded from the staged DS plane, not global).
    float td_x[4];
    int ts_x[4];

    for (int g = 0; g < groups_per_row; g++) {
        for (int h = 0; h < 2; h++) {
            const int kb = 2 * g + h;

            // --- Stage metadata: DS (token d/s for kb) + SZ (row sc/zp). ---
            for (int i = tid; i < IU4_BM * 2; i += IU4_NTHREADS) {
                const int tok_idx = i >> 1;
                const int w = i & 1;
                const int tok = bs + tok_idx;
                const unsigned dw = (tok < N)
                    ? ((const unsigned*)(Xq + (long long)kb * N + tok))[w]
                    : 0u;
                ((unsigned*)(ds_lds + tok_idx * 8))[w] = dw;
            }
            for (int i = tid; i < IU4_BN * 2; i += IU4_NTHREADS) {
                const int row_idx = i >> 1;
                const int w = i & 1;
                const int grow = rs + row_idx;
                const int src_row = (grow < M) ? grow : (M - 1);
                const unsigned hw = *(const unsigned*)(A + (long long)src_row * groups_per_row * 136
                    + g * 136 + h * 4);
                const float f = (w == 0)
                    ? __half2float(__ushort_as_half((unsigned short)(hw & 0xFFFFu)))
                    : __half2float(__ushort_as_half((unsigned short)(hw >> 16)));
                ((float*)(sz_lds + row_idx * 8))[w] = (grow < M) ? f : 0.0f;
            }
            __syncthreads();
            // Lane token metadata for the fold (from LDS, live over slabs).
            #pragma unroll
            for (int nb = 0; nb < 4; nb++) {
                const unsigned tbase = (unsigned)(wave_tok0 + nb * 16 + m_lane) * 8u;
                td_x[nb] = *(const float*)(ds_lds + tbase);
                ts_x[nb] = *(const int*)(ds_lds + tbase + 4u);
            }

            // --- Prologue: stage slab 0 (A + W). ---
            for (int i = tid; i < IU4_BM * 8; i += IU4_NTHREADS) {
                const int row = i >> 3;
                const int dw = i & 7;
                const int tok = bs + row;
                unsigned v = 0u;
                if (tok < N) {
                    const block_i4_128* xb = Xq + (long long)kb * N + tok;
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
                    + g * 136 + 8 + h * 64 + dw * 4);
                ((unsigned*)(w_lds + row * IU4_STRIDE))[dw] = v;
            }
            __syncthreads();

            // --- Trip 0: prefetch slab 1, compute slab 0, publish. ---
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
                iu4_bundle<0, true>(w_lds, a_lds, frW, frA, cacc, zc);
                iu4_bundle<1, false>(w_lds, a_lds, frW, frA, cacc, zc);
                __syncthreads();
                // Publish slab 1 over slab 0's slot (buffer free: trip-0
                // compute done; single LDS buffer, same addresses).
                #pragma unroll
                for (int t = 0; t < IU4_BM * 8 / IU4_NTHREADS; t++) {
                    const int i = tid + IU4_NTHREADS * t;
                    const int row = i >> 3;
                    const int dw = i & 7;
                    ((unsigned*)(a_lds + row * IU4_STRIDE))[dw] = A_pf[t];
                    ((unsigned*)(w_lds + row * IU4_STRIDE))[dw] = W_pf[t];
                }
                __syncthreads();
            }
            // --- Trip 1: compute slab 1, fold at the 128-K boundary. ---
            {
                iu4_bundle<0, false>(w_lds, a_lds, frW, frA, cacc, zc);
                iu4_bundle<1, false>(w_lds, a_lds, frW, frA, cacc, zc);
                // Fold: 4 K32 int32 partials per (nb, rg) through the pinned
                // IU4_FOLD_RN DAG. Per-half (sc,zp) via intra-wave shuffle
                // from the lane owning each output row; (d,s) streamed from
                // the staged DS plane.
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
                        const float d_x = td_x[nb];
                        const int s_x = ts_x[nb];
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            IU4_FOLD_RN(acc[nb][rg][j], sc_row[j], zp_row[j], d_x, acc_i[j], s_x);
                        }
                    }
                }
                __syncthreads();
            }
        } // for h
    } // for g
    // Coalesced store: the WMMA C layout leaves lane (m_lane,k_grp) holding
    // acc[nb][rg][j] for token wave_tok0+nb*16+m_lane and rows
    // wave_row0+rg*16+8*k_grp+j, so one store instruction touches 16 output
    // rows at 4 B each. Transpose each (rg,nb) 16x16 wave tile through LDS
    // instead: all 8 waves dump in parallel (one [16][20] slot each; the
    // stride-20 pad keeps streaming reads bank-clean), then all 256 threads
    // stream the 64x32 phase tile with thread (i=tid&15, x=(tid>>4)&15)
    // covering rows (consecutive threads = consecutive rows = 128 B global
    // segments) and k covering 4 row-pairs x 2 token-halves. Per-element
    // arithmetic is unchanged (one residual add), hence bit-identical; only
    // the visit order changes. A/W/DS/SZ are dead (all folds complete), so
    // the slots reuse LDS base: 8 x 320 floats = 10240 B < 12288 B.
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