// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx1201 wave64 research variant of the MQ4G256V2 iu4-direct MMQ.
// It preserves the shipped seven-argument ABI, block_i4_128 metadata path,
// double-buffered 64-K LDS body, K32 order, and pinned five-node RN fold.
//
// WMMA lane map (probe-verified on gfx1201, 2026-09-19):
//   operand lane L: row = L&15, K = 8*(L>>4)..8*(L>>4)+7, one packed i4x8.
//   C lane quarters 0,1,2,3 own row bases 0,8,4,12 respectively; each lane
//   returns four contiguous rows at column L&15.
//
// Workgroup: 128 rows x 128 tokens x K64, 256 threads = four wave64 waves.
// IU4_W64_RGS=2 is the direct 2x4 wave-tile probe.  Each wave owns 32 rows
// and the kernel traverses the same K loop twice, once per 64-token half,
// retaining 32 f32 plus 32 i32 accumulator dwords/lane.
// IU4_W64_RGS=4 is the 4x4 retile.  Waves map as 2 row halves x 2 token
// halves, traverse K once, and retain 64 f32 plus 64 i32 dwords/lane.
//
// A 64-K A or W slab remains 16 fragments x 256 B.  Wave64 lanes read one
// dword each: fragment lane L is row L&15 and K-quarter L>>4.  Cooperative
// b64 global loads split into two b32 LDS stores to that fragment order.
//
// LDS (20480 B): A0 0, W0 4096, W1 8192, DS0 12288, DS1 13312,
// SZ0 14336, SZ1 15360, A1 16384.  Store slots reuse the base after the
// final compute barrier.  Grid [ceil(M/128),ceil(N/128)], block 256,
// dynamic LDS 20480 B.

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
#ifndef IU4_W64_RGS
#define IU4_W64_RGS 2
#endif
#define IU4_WAVES 4
#define IU4_NTHREADS (IU4_WAVES * 64)
// Fragment-order slab: each 64-K slab holds 8 row/token groups x 2 K32
// steps.  A fragment is 64 wave64 lanes x 4 B = 256 B.
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
// Four [16][20] f32 store slots reuse the LDS base.
#define IU4_SLOT_COLS 20
#define IU4_SLOT_FLOATS (16 * IU4_SLOT_COLS)

typedef int32_t __attribute__((ext_vector_type(4)))  int32x4_t;
typedef float   __attribute__((ext_vector_type(4)))  float4_t;
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
#endif

// One K32-sub-block compute bundle for wave64.  Each lane carries one
// packed i4x8 scalar and four C elements; lane = row + 16*K-quarter.
template <int SB, bool FIRST, int RGS>
static __device__ __forceinline__ void iu4_bundle_w64(
    const char* __restrict__ w_lds,
    const char* __restrict__ a_lds,
    const unsigned (&frW)[RGS][2],
    const unsigned (&frA)[4][2],
    int32x4_t cacc[4][RGS],
    const int32x4_t& zc
) {
    int af[RGS];
    #pragma unroll
    for (int rg = 0; rg < RGS; ++rg) af[rg] = *(const int*)(w_lds + frW[rg][SB]);
    int bf[4];
    #pragma unroll
    for (int nb = 0; nb < 4; ++nb) bf[nb] = *(const int*)(a_lds + frA[nb][SB]);
    #pragma unroll
    for (int rg = 0; rg < RGS; ++rg) {
        #pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            if constexpr (FIRST) {
                cacc[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w64_gfx12(
                    false, af[rg], true, bf[nb], zc, false);
            } else {
                cacc[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w64_gfx12(
                    false, af[rg], true, bf[nb], cacc[nb][rg], false);
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

// Shared compute body for residual / full_add / full_set. `ADD` selects
// += (residual) vs = (set) writeback.
template <bool ADD, int RGS>
static __device__ __forceinline__ void gemm_mq4g256v2_residual_mmq_iu4_w64_tile(
    const char*           __restrict__ A,
    const block_i4_128*   __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int wave_row0, int wave_tok0
) {
    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int rs = blockIdx.x * IU4_BN;
    const int bs = blockIdx.y * IU4_BM;

    if (rs >= M || bs >= N) return;

    const int m_lane = lane & 15;
    const int k_grp  = lane >> 4;
    // Wave64 C quarters are emitted in bit-reversed K-quarter order:
    // lane groups 0,1,2,3 own C rows 0,8,4,12 respectively.
    const int c_row = ((k_grp & 1) << 3) | ((k_grp & 2) << 1);

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
    unsigned frW[RGS][2];
    // Fragment (g8,sb) lives at (g8*2+sb)*256; this wave64 lane's packed
    // i4x8 scalar is the dword at +lane*4.
    const unsigned lane4 = (unsigned)lane * 4u;
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        const unsigned g8a = (unsigned)(wave_tok0 / 16 + nb);
        #pragma unroll
        for (int sb = 0; sb < 2; sb++) {
            frA[nb][sb] = (g8a * 2u + (unsigned)sb) * (unsigned)IU4_FRAG_BYTES + lane4;
        }
    }
    #pragma unroll
    for (int rg = 0; rg < RGS; rg++) {
        const unsigned g8w = (unsigned)(wave_row0 / 16 + rg);
        #pragma unroll
        for (int sb = 0; sb < 2; sb++) {
            frW[rg][sb] = (g8w * 2u + (unsigned)sb) * (unsigned)IU4_FRAG_BYTES + lane4;
        }
    }

    // acc[nb][rg][j] maps through c_row to a 16x16 output fragment.
    float4_t acc[4][RGS];
    // Each thread stages two slab rows and one 8-byte quad per row.  The b64
    // global word is split into consecutive K-quarter dwords qd=0,1 at
    // consumer lanes (R&15)+16*(2*(q&1)+qd) in fragment (R>>4,q>>1).
    // This is a bijection over each 4 KiB slab.  Global b128 is unavailable:
    // nibble payloads begin eight bytes into 72-B Xq blocks and 136-B groups.
    // All offsets below are loop-invariant (bs/rs/tid only), computed once at
    // kernel top; per-trip SGPR bases (Xq + kb*N, A + g*136 + ...) carry kb/g/h
    // so the K-loop issues base(SGPR) + offset(VGPR) + imm wide loads/stores
    // with zero per-load address VALU. OOB rows clamp to the last valid index
    // (same addresses the old ternaries produced); A values additionally zero-
    // select (exact old semantics) via branch-free ternary (cndmask, no EXEC
    // masking), while W keeps the clamped value exactly as before.
    unsigned st_avoff[2];
    unsigned st_wvoff[2];
    unsigned st_ldsoff[2][2];
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
        const unsigned R = (unsigned)(r * 64 + (tid >> 2));
        const unsigned q = (unsigned)(tid & 3);
        const unsigned frag_base = ((R >> 4) * 2u + (q >> 1)) * (unsigned)IU4_FRAG_BYTES;
        const unsigned kg0 = 2u * (q & 1u);
        st_ldsoff[r][0] = frag_base + ((R & 15u) + kg0 * 16u) * 4u;
        st_ldsoff[r][1] = frag_base + ((R & 15u) + (kg0 + 1u) * 16u) * 4u;
    }
    for (int nb = 0; nb < 4; nb++) {
        #pragma unroll
        for (int rg = 0; rg < RGS; rg++) {
            acc[nb][rg] = (float4_t){0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
    // One reusable zero C vector for each chain's first WMMA.
    const int32x4_t zc = (int32x4_t){0, 0, 0, 0};
    int32x4_t cacc[4][RGS];
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
            const u32x2_t wv = *(const u32x2_t*)(wB0 + st_wvoff[r]);
            #pragma unroll
            for (int qd = 0; qd < 2; ++qd) {
                *(unsigned*)(a_lds + st_ldsoff[r][qd]) = st_aok[r] ? av[qd] : 0u;
                *(unsigned*)(w_lds + st_ldsoff[r][qd]) = wv[qd];
            }
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
                const char* __restrict__ xqB1 = (const char*)(Xq + (long long)kb * N) + 8 + 32;
                const char* __restrict__ wB1 = A + (long long)g * 136 + 8 + (long long)h * 64 + 32;
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    A_pf[r] = *(const u32x2_t*)(xqB1 + st_avoff[r]);
                    W_pf[r] = *(const u32x2_t*)(wB1 + st_wvoff[r]);
                }
                // No scheduling pin here: the previous trip's fold VALU must
                // remain free to sink among this trip's WMMAs. Raw A_pf is
                // consumed only at the alternate-slot publication below.
                // Every 128-K block restarts its int32 accumulation from zero
                // (each block folds independently): FIRST=true on slab 0 of
                // every (g,h), exactly as in v2. No h-dependence here.
                iu4_bundle_w64<0, true, RGS>(w_lds, a_lds, frW, frA, cacc, zc);
                iu4_bundle_w64<1, false, RGS>(w_lds, a_lds, frW, frA, cacc, zc);
                // Publish slab 1 to the free alternate slot. The raw A pin
                // keeps its zero-select at this first use, after slot-0
                // compute, without imposing a memory or scheduling fence.
                a_lds = LDS + IU4_A1_OFF;
                w_lds = LDS + IU4_W1_OFF;
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    asm volatile("" : "+v"(A_pf[r][0]), "+v"(A_pf[r][1]));
                    #pragma unroll
                    for (int qd = 0; qd < 2; ++qd) {
                        *(unsigned*)(a_lds + st_ldsoff[r][qd]) = st_aok[r] ? A_pf[r][qd] : 0u;
                        *(unsigned*)(w_lds + st_ldsoff[r][qd]) = W_pf[r][qd];
                    }
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
                    const char* __restrict__ xq_nx = (const char*)(Xq + (long long)kb_nx * N) + 8;
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        A_pf[r] = *(const u32x2_t*)(xq_nx + st_avoff[r]);
                    }
                    const int g_nx = g + h;
                    const int h_nx = h ^ 1;
                    const char* __restrict__ w_nx = A + (long long)g_nx * 136 + 8 + (long long)h_nx * 64;
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        W_pf[r] = *(const u32x2_t*)(w_nx + st_wvoff[r]);
                    }
                    const int tok_ds = bs + (tid >> 1);
                    const int tok_dsc = (tok_ds < N) ? tok_ds : (N - 1);
                    ds_nx = ((const unsigned*)(Xq + (long long)kb_nx * N + tok_dsc))[tid & 1];
                    const int row_sz = rs + (tid >> 1);
                    const int src_sz = (row_sz < M) ? row_sz : (M - 1);
                    sz_nx = *(const unsigned*)(A + (long long)src_sz * groups_per_row * 136
                        + g_nx * 136 + h_nx * 4);
                }
            }
            // --- Slot 1 compute, current fold, next slot-0 publish. ---
            {
                iu4_bundle_w64<0, false, RGS>(w_lds, a_lds, frW, frA, cacc, zc);
                iu4_bundle_w64<1, false, RGS>(w_lds, a_lds, frW, frA, cacc, zc);
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
                for (int rg = 0; rg < RGS; rg++) {
                    float sc_row[4];
                    float zp_row[4];
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const unsigned jbase = (unsigned)(wave_row0 + rg * 16 + c_row + j) * 8u;
                        sc_row[j] = *(const float*)(sz_lds + jbase);
                        zp_row[j] = *(const float*)(sz_lds + jbase + 4u);
                    }
                    #pragma unroll
                    for (int nb = 0; nb < 4; nb++) {
                        int* acc_i = (int*)&cacc[nb][rg];
                        const unsigned tbase = (unsigned)(wave_tok0 + nb * 16 + m_lane) * 8u;
                        const float d_x = *(const float*)(ds_lds + tbase);
                        const int s_x = *(const int*)(ds_lds + tbase + 4u);
#if defined(__gfx1201__)
                        // Paired fold: same values, same DAG as IU4_FOLD_RN;
                        // sf hoisted (the int sum is loop-invariant, the
                        // compiler CSEs the baseline's per-j conversion to
                        // one anyway: 4 sf converts per half, unchanged).
                        const float sf_x = (float)s_x;
                        #pragma unroll
                        for (int j = 0; j < 4; j += 2) {
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
                        for (int j = 0; j < 4; j++) {
                            IU4_FOLD_RN(acc[nb][rg][j], sc_row[j], zp_row[j], d_x, acc_i[j], s_x);
                        }
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
                        #pragma unroll
                        for (int qd = 0; qd < 2; ++qd) {
                            *(unsigned*)(a_lds + st_ldsoff[r][qd]) = st_aok[r] ? A_pf[r][qd] : 0u;
                            *(unsigned*)(w_lds + st_ldsoff[r][qd]) = W_pf[r][qd];
                        }
                    }
                    const int h_np = h ^ 1;
                    char* __restrict__ ds_n = LDS + ((h_np == 0) ? IU4_DS_OFF : IU4_DS1_OFF);
                    {
                        const int tok_dsp = bs + (tid >> 1);
                        ((unsigned*)(ds_n + (tid >> 1) * 8))[tid & 1] = (tok_dsp < N) ? ds_nx : 0u;
                    }
                    char* __restrict__ sz_n = LDS + ((h_np == 0) ? IU4_SZ0_OFF : IU4_SZ1_OFF);
                    {
                        const int row_szp = rs + (tid >> 1);
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
    // Transpose each (rg,nb) 16x16 wave tile through four LDS slots, then
    // all 256 threads stream the phase tile.
    //
    // Full tiles use one tile-relative buffer descriptor. Its base and each
    // phase stride stay uniform in SGPRs; the only vector byte offset is the
    // lane's (x*M+i), formed once here instead of rebuilding a 64-bit global
    // address for every store. Edge tiles retain the original checked path.
    float* slots = (float*)LDS;
    const int store_i = tid & 15;
    const int store_x = (tid >> 4) & 15;
    const unsigned store_lane_off =
        ((unsigned)store_x * (unsigned)M + (unsigned)store_i) * sizeof(float);
    float* const y_tile = Y + (long long)bs * M + rs;
    const auto y_tile_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        y_tile, 0, 0xffffffffu, 0x31004000u);
    const bool full_tile = rs + IU4_BN <= M && bs + IU4_BM <= N;
    #pragma unroll
    for (int rg = 0; rg < RGS; ++rg) {
        #pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                slots[(unsigned)(wave * 320 + (c_row + j) * 20 + m_lane)] = acc[nb][rg][j];
            }
            __syncthreads();
            #pragma unroll
            for (int owner = 0; owner < 4; ++owner) {
                int row_rel;
                int col_rel;
                if constexpr (RGS == 2) {
                    row_rel = owner * 32 + rg * 16;
                    col_rel = wave_tok0 + nb * 16;
                } else {
                    row_rel = (owner >> 1) * 64 + rg * 16;
                    col_rel = (owner & 1) * 64 + nb * 16;
                }
                const float v = slots[(unsigned)(owner * 320 + store_i * 20 + store_x)];
                if (full_tile) {
                    const unsigned phase_off =
                        ((unsigned)col_rel * (unsigned)M + (unsigned)row_rel) * sizeof(float);
                    if constexpr (ADD) {
                        const unsigned old_bits = __builtin_amdgcn_raw_buffer_load_b32(
                            y_tile_rsrc, store_lane_off, phase_off, 0);
                        const float out = __builtin_bit_cast(float, old_bits) + v;
                        __builtin_amdgcn_raw_buffer_store_b32(
                            __builtin_bit_cast(unsigned, out), y_tile_rsrc, store_lane_off, phase_off, 0);
                    } else {
                        __builtin_amdgcn_raw_buffer_store_b32(
                            __builtin_bit_cast(unsigned, v), y_tile_rsrc, store_lane_off, phase_off, 0);
                    }
                } else {
                    const int row = rs + row_rel + store_i;
                    const int col = bs + col_rel + store_x;
                    if (row < M && col < N) {
                        float* yp = Y + (long long)col * M + row;
                        if constexpr (ADD) *yp += v;
                        else *yp = v;
                    }
                }
            }
            __syncthreads();
        }
    }
}

template <bool ADD>
static __device__ __forceinline__ void gemm_mq4g256v2_residual_mmq_iu4_w64_body(
    const char* __restrict__ A, const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y, int M, int K, int N) {
    const int wave = threadIdx.x >> 6;
#if IU4_W64_RGS == 2
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        gemm_mq4g256v2_residual_mmq_iu4_w64_tile<ADD, 2>(
            A, Xq, Y, M, K, N, wave * 32, pass * 64);
    }
#elif IU4_W64_RGS == 4
    gemm_mq4g256v2_residual_mmq_iu4_w64_tile<ADD, 4>(
        A, Xq, Y, M, K, N, (wave >> 1) * 64, (wave & 1) * 64);
#else
#error "IU4_W64_RGS must be 2 or 4"
#endif
}

#ifndef IU4_W64_KERNEL
#define IU4_W64_KERNEL gemm_mq4g256v2_residual_mmq_iu4_w64
#endif
#ifndef IU4_W64_ADD_KERNEL
#define IU4_W64_ADD_KERNEL gemm_mq4g256v2_residual_mmq_iu4_w64_full_add
#endif
#ifndef IU4_W64_SET_KERNEL
#define IU4_W64_SET_KERNEL gemm_mq4g256v2_residual_mmq_iu4_w64_full_set
#endif

extern "C" __global__ __launch_bounds__(IU4_NTHREADS, 2)
void IU4_W64_KERNEL(
    const char* __restrict__ A, const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y, int M, int K, int N, int add) {
    if (add) gemm_mq4g256v2_residual_mmq_iu4_w64_body<true>(A, Xq, Y, M, K, N);
    else gemm_mq4g256v2_residual_mmq_iu4_w64_body<false>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(IU4_NTHREADS, 2)
void IU4_W64_ADD_KERNEL(
    const char* __restrict__ A, const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y, int M, int K, int N, int) {
    gemm_mq4g256v2_residual_mmq_iu4_w64_body<true>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(IU4_NTHREADS, 2)
void IU4_W64_SET_KERNEL(
    const char* __restrict__ A, const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y, int M, int K, int N, int) {
    gemm_mq4g256v2_residual_mmq_iu4_w64_body<false>(A, Xq, Y, M, K, N);
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