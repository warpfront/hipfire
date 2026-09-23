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

#define HIPFIRE_IU4_WEIGHT_PREFETCH 1
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
//   - Double-buffered slab staging (Change A): the W (weight) slab is
//     double-buffered per 64-K slab (W0/W1); the A (activation) slab stays
//     single-buffered (A) with v2's publish fence. DS and SZ metadata
//     ping-pong per half-block (plane[h], kb = 2g+h) while token (d,s) is
//     re-read from the DS plane at fold time (same values, same DAG).
//     Steady state per block: prefetch slab 1 into A_pf/W_pf, compute slab
//     0 from (A,W0), B0 (slab-0 A-reads done), publish slab 1 to (A,W1),
//     B1, then the next-block fetch epoch: all six next-block raw loads
//     (next A slab 0, next W slab 0, next DS word, next header word) issue
//     immediately after B1 into private registers. Compute slab 1 from
//     (A,W1), fold the current block, B2a (slab-1 A-reads done; also
//     drains the fetch epoch after the useful compute1+fold window),
//     publish next (masked/converted) to (A,W0,DS[h^1],SZ[h^1]), B2b.
//     Four barriers per block (vs five before); slab 1 loads issue above
//     slab 0's WMMAs (sched_barrier-pinned, as in v2) and the fetch epoch
//     issues above compute 1 + fold. Bit-identical: same K order, same
//     pinned IU4_FOLD_RN DAG.
//     LDS 16384 B keeps 3 WG/CU on gfx1201 (3 x 16384 = 49152 < 65536/CU;
//     the host still passes 19456 B, a harmless superset, same occupancy).
//   - Coalesced store (fp8 K2 mechanism): each (rg, nb) phase dumps the
//     wave's 16x16 f32 sub-tile to an LDS slot ([16][20], stride-20 pad for
//     bank-clean streaming), then all 256 threads stream 64 rows x 32 cols
//     with consecutive threads on consecutive rows (128 B segments).
//     Per-element arithmetic is unchanged, hence bit-identical.
//   - Math is bit-identical to v1: same 4xK32 grouping per 128-K block per
//     column block (same K order), same pinned IU4_FOLD_RN DAG. Integer MMA:
//     any oracle difference is a bug.
//
// LDS layout (16384 B = 16 KiB on gfx1201; compute planes and store slots
// overlap — A/W/DS/SZ are all dead once folds complete):
//   A  = 16 WMMA fragments x 256 B (fragment order, below)  @ 0     (4096 B)
//   W0 = 16 WMMA fragments x 256 B                         @ 4096  (4096 B)
//   W1 = 16 WMMA fragments x 256 B (second slab buffer)    @ 8192  (4096 B)
//   DS0 = [128][8] token (d f32, s i32), even halves        @ 12288 (1024 B)
//   DS1 = [128][8] token (d f32, s i32), odd halves         @ 13312 (1024 B)
//   SZ0 = [128][8] row (sc f32, zp f32), even halves        @ 14336 (1024 B)
//   SZ1 = [128][8] row (sc f32, zp f32), odd halves         @ 15360 (1024 B)
//   end                                                             = 16384 B
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
// Grid: [ceil(M/128), ceil(N/128)]. Block: [256,1,1]. LDS: 16384 bytes
// (host passes 19456; kernel uses the first 16384).
//
// Sweep note: 64x256 / 256x64 aspects need a different wave map (wt_pair /
// tok_half split) plus matching host grid/LDS; the fold/prefetch/store
// mechanisms transfer unchanged. Host (gemm.rs launch arm) and the oracle
// launcher hardcode the same 128/128/256-thread contract (19456 B passed;
// kernel uses 16384) — a geometry change must update all three consistently.
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
#define IU4_LDS_BYTES (IU4_SZ1_OFF + IU4_SZ_BYTES)
// The host launch arm passes 19456 B (must cover the kernel's footprint);
// fail loudly here if a geometry change outgrows it.
_Static_assert(IU4_LDS_BYTES <= 19456, "iu4 fragment planes exceed host LDS");
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

#if defined(HIPFIRE_IU4_WEIGHT_CPOL)
#ifndef HIPFIRE_WEIGHT_CPOL_AUX
#define HIPFIRE_WEIGHT_CPOL_AUX 20
#endif
static __device__ __forceinline__ u32x2_t iu4_weight_load_b64(
    const char* __restrict__ base, unsigned byte_offset) {
    const auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<char*>(base), 0, 0xffffffffu, 0x31004000u);
    return __builtin_amdgcn_raw_buffer_load_b64(
        rsrc, byte_offset, 0, HIPFIRE_WEIGHT_CPOL_AUX);
}
static __device__ __forceinline__ unsigned iu4_weight_load_b32(
    const char* __restrict__ base, unsigned byte_offset) {
    const auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<char*>(base), 0, 0xffffffffu, 0x31004000u);
    return __builtin_amdgcn_raw_buffer_load_b32(
        rsrc, byte_offset, 0, HIPFIRE_WEIGHT_CPOL_AUX);
}
#define IU4_WEIGHT_LOAD_B64(base, byte_offset) \
    iu4_weight_load_b64((base), (byte_offset))
#define IU4_WEIGHT_LOAD_B32(base, byte_offset) \
    iu4_weight_load_b32((base), (byte_offset))
#else
#define IU4_WEIGHT_LOAD_B64(base, byte_offset) \
    (*(const u32x2_t*)((base) + (byte_offset)))
#define IU4_WEIGHT_LOAD_B32(base, byte_offset) \
    (*(const unsigned*)((base) + (byte_offset)))
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
#endif

// One K32-sub-block compute bundle: 2 W-frags + 4 A-frags +
// 8 WMMAs (2 row-groups x 4 column blocks).
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
                    false, a, true, b, zc, false
                );
            } else {
                cacc[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                    false, a, true, b, cacc[nb][rg], false
                );
            }
        }
    }
#if !defined(HIPFIRE_IU4_SCHED_GROUPS)
    // Default scheduling fence. The opt-in PLR1 route groups both K32
    // bundles at their call site so the second bundle's six DS reads may
    // issue above the first bundle's eight WMMAs.
    __builtin_amdgcn_sched_barrier(0);
#endif
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
    // Fragment-order offsets: token group g8 = tok_half*4+nb (A) or row
    // group g8 = wt_pair*2+rg (W); fragment (g8,sb) lives at (g8*2+sb)*256
    // and this lane's 8 bytes at +lane8. All loop-invariant (wave/nb/rg/sb
    // are kernel-top or unroll constants; lane = tid&31 is fixed).
    const unsigned lane8 = (unsigned)lane * 8u;
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        const unsigned g8a = (unsigned)(tok_half * 4 + nb);
        #pragma unroll
        for (int sb = 0; sb < 2; sb++) {
            frA[nb][sb] = (g8a * 2u + (unsigned)sb) * (unsigned)IU4_FRAG_BYTES + lane8;
        }
    }
    #pragma unroll
    for (int rg = 0; rg < 2; rg++) {
        const unsigned g8w = (unsigned)(wt_pair * 2 + rg);
        #pragma unroll
        for (int sb = 0; sb < 2; sb++) {
            frW[rg][sb] = (g8w * 2u + (unsigned)sb) * (unsigned)IU4_FRAG_BYTES + lane8;
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
        const unsigned R = (unsigned)(r * 64 + (tid >> 2));
        const unsigned q = (unsigned)(tid & 3);
        st_ldsoff[r] = ((R >> 4) * 2u + (q >> 1)) * (unsigned)IU4_FRAG_BYTES
            + ((R & 15u) + (q & 1u) * 16u) * 8u;
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
    int32x8_t cacc[4][2];
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
        const unsigned hw = IU4_WEIGHT_LOAD_B32(
            A + (long long)src_row * groups_per_row * 136, 0);
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
            *(u32x2_t*)(w_lds + st_ldsoff[r]) =
                IU4_WEIGHT_LOAD_B64(wB0, st_wvoff[r]);
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
            // --- Slab 0 compute (A, W0) with slab 1 prefetch as producer. ---
            {
                // Slab 1 prefetch, b64-coalesced: uniform SGPR bases carry
                // kb/g/h, lane VGPR offsets were hoisted to kernel top.
                const char* __restrict__ xqB1 = (const char*)(Xq + (long long)kb * N) + 8 + 32;
                const char* __restrict__ wB1 = A + (long long)g * 136 + 8 + (long long)h * 64 + 32;
#if defined(HIPFIRE_IU4_WEIGHT_PREFETCH)
                // One hint for the next 64-K weight slab. This is a scalar
                // cache prefetch only; the existing vector loads remain the
                // source of the WMMA payload.
                __builtin_amdgcn_s_prefetch_data((const void*)wB1, 1);
#endif
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    const u32x2_t av = *(const u32x2_t*)(xqB1 + st_avoff[r]);
                    A_pf[r] = st_aok[r] ? av : z64;
                    W_pf[r] = IU4_WEIGHT_LOAD_B64(wB1, st_wvoff[r]);
                }
                // Variant 0: NO pin barrier here. The previous trip's fold
                // VALU (above its B2a) must be free to sink among this
                // trip's WMMAs; a sched_barrier(0) would retire the whole
                // fold before the chain starts. True deps (cacc WAR per
                // element, A_pf/W_pf publish after B0) still order what
                // must be ordered; the scheduler hoists the prefetch loads
                // for latency on its own.
                // Every 128-K block restarts its int32 accumulation from zero
                // (each block folds independently): FIRST=true on slab 0 of
                // every (g,h), exactly as in v2. No h-dependence here.
                iu4_bundle<0, true>(w_lds, a_lds, frW, frA, cacc, zc);
                iu4_bundle<1, false>(w_lds, a_lds, frW, frA, cacc, zc);
#if defined(HIPFIRE_IU4_SCHED_GROUPS)
                // PLR1 for the two K32 steps in this slab. Groups are listed
                // bottom-up after both force-inlined bundles: DS_READ=0x100,
                // WMMA/MFMA=0x008. Each step is exactly 6 ds_read_b64
                // (2 W + 4 A) and 8 iu4 WMMAs; no VMEM or independent VALU
                // occurs inside a step.
                __builtin_amdgcn_sched_group_barrier(0x100, 6, 0);
                __builtin_amdgcn_sched_group_barrier(0x100, 6, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, 8, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, 8, 0);
#endif
                // B0: slab-0 A-reads done on every wave (A is single-buffered:
                // the publish below overwrites slab 0 with slab 1). W1 writes
                // need no fence but share it.
                __syncthreads();
                // Publish slab 1 (A_pf -> A plane, W_pf -> W1) and re-point
                // W at W1 (A is fixed). Nothing next-block is live yet: the
                // fetch epoch is BELOW B1, never above.
                w_lds = LDS + IU4_W1_OFF;
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    *(u32x2_t*)(a_lds + st_ldsoff[r]) = A_pf[r];
                    *(u32x2_t*)(w_lds + st_ldsoff[r]) = W_pf[r];
                }
                // B1: slab-1 publish visible to all waves before compute 1.
                __syncthreads();
                if (!last) {
                    // Next-block fetch epoch: all six next-block raw loads
                    // (4 b64 nibble + 1 DS word + 1 packed header word) issue
                    // here, immediately after B1, into private registers. No
                    // masks, header conversion, DS stores or first use yet:
                    // first use is the publish below B2a, so the waits drain
                    // after compute1 + fold and the epoch overlaps them.
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
                        W_pf[r] = IU4_WEIGHT_LOAD_B64(w_nx, st_wvoff[r]);
                    }
                    const int tok_ds = bs + (tid >> 1);
                    const int tok_dsc = (tok_ds < N) ? tok_ds : (N - 1);
                    ds_nx = ((const unsigned*)(Xq + (long long)kb_nx * N + tok_dsc))[tid & 1];
                    const int row_sz = rs + (tid >> 1);
                    const int src_sz = (row_sz < M) ? row_sz : (M - 1);
                    sz_nx = IU4_WEIGHT_LOAD_B32(
                        A + (long long)src_sz * groups_per_row * 136
                            + g_nx * 136 + h_nx * 4,
                        0);
                }
            }
            // --- Slab 1 compute (A, W1), current fold, next-block publish. ---
            {
                iu4_bundle<0, false>(w_lds, a_lds, frW, frA, cacc, zc);
                iu4_bundle<1, false>(w_lds, a_lds, frW, frA, cacc, zc);
#if defined(HIPFIRE_IU4_SCHED_GROUPS)
                // Same 2-step, 6-DS/8-WMMA PLR1 ladder for slab 1.
                __builtin_amdgcn_sched_group_barrier(0x100, 6, 0);
                __builtin_amdgcn_sched_group_barrier(0x100, 6, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, 8, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, 8, 0);
#endif
                // Fold (rescheduled above B2a so the fetch epoch overlaps
                // compute1 + this VALU): 4 K32 int32 partials per (nb, rg)
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
                    }
                }
                // B2a: slab-1 A-reads done on every wave, so the single A
                // plane may be republished with next slab 0. This drain also
                // completes the fetch epoch's in-flight loads, AFTER the
                // useful compute1 + fold window above.
                __syncthreads();
                if (!last) {
                    // Publish next block (first use of the fetch epoch):
                    // original A zero-select, clamped W value, masked DS
                    // word and converted header word, with the original
                    // producer map. Re-point W at W0.
                    w_lds = LDS + IU4_W0_OFF;
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        *(u32x2_t*)(a_lds + st_ldsoff[r]) = st_aok[r] ? A_pf[r] : z64;
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
                        const int row_szp = rs + (tid >> 1);
                        const int w_szp = tid & 1;
                        const float f = (w_szp == 0)
                            ? __half2float(__ushort_as_half((unsigned short)(sz_nx & 0xFFFFu)))
                            : __half2float(__ushort_as_half((unsigned short)(sz_nx >> 16)));
                        ((float*)(sz_n + (tid >> 1) * 8))[w_szp] = (row_szp < M) ? f : 0.0f;
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

#if defined(HIPFIRE_IU4_WAVES8)
#define IU4_WAVES8_ATTR __attribute__((amdgpu_waves_per_eu(8, 8)))
#else
#define IU4_WAVES8_ATTR
#endif

// residual: += (the `add` kernarg picks += vs = at runtime).
extern "C" __global__ __launch_bounds__(IU4_NTHREADS, 2) IU4_WAVES8_ATTR
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

extern "C" __global__ __launch_bounds__(IU4_NTHREADS, 2) IU4_WAVES8_ATTR
void gemm_mq4g256v2_residual_mmq_iu4_full_add(
    const char*           __restrict__ A,
    const block_i4_128*   __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int
) {
    gemm_mq4g256v2_residual_mmq_iu4_gfx12_body<true>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(IU4_NTHREADS, 2) IU4_WAVES8_ATTR
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