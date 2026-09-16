// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_WEIGHT_LOAD_PTR_U32
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#define HIPFIRE_WEIGHT_RSRC(base) 0
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, byte_offset) \
    (*(const unsigned int*)(ptr))
#endif

#ifndef HIPFIRE_RESIDUAL_KERNEL
#define HIPFIRE_RESIDUAL_KERNEL gemv_mq4g256v2_residual
#endif

// HFQ4-G256 GEMV with fused residual add: y[row] += A[row] · x.
//
// VARIANT R36c1 res_w_preload:
// BOD residual mixed (occ~23%, mem_busy~45%, L2~78%, vgpr=72): post
// R15 hoist_x32, free micro-schedule (DOG order / LDS-x) is exhausted —
// TLP already hides latency. Gate_up R35c2 won by fully separating the
// weight MEMORY phase from the DOG VALU phase after x-hoist: sequential
// sc/zp/pk streams for the whole group-quad, then a pure FMA burst.
//
// Hypothesis: same separation helps residual's dual-row weight stream
// (2 rows × 4 groups = 8 headers). Baseline interleaves load+FMA per
// DOG; here load all 8 sc/zp/pk first (row0 g0..g3 then row1 g0..g3 —
// sequential along each row_ptr), then 8 DOG FMAs with identical math
// and acc slots. Value-preserving. Expect VGPR up modestly (hold 8
// headers); if DROPPED-WAVES dominates, this path is closed for residual.

#if defined(__gfx1200__) || defined(__gfx1201__)
// RDNA4-only occupancy probe: force the compiler to target 32 resident
// one-wave blocks (64 VGPR/thread budget) on this low-occupancy kernel.
// The non-gfx12 device code remains byte-identical to the baseline.
__launch_bounds__(32, 32)
#else
__launch_bounds__(32, 16)
#endif
extern "C" __global__ void HIPFIRE_RESIDUAL_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row0 = blockIdx.x << 1;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const bool have_row1 = row1 < M;
    const int tid = threadIdx.x;

#if defined(HIPFIRE_GFX1151_RESIDUAL_K4096)
    // Exact Qwen3.6 A3B decode contract. Making both quads static lets LLVM
    // schedule the two independent row chains across the full K extent and
    // removes the dead generic tail machinery on gfx1151.
    const int groups_per_row = 16;
#else
    const int groups_per_row = K / 256;
#endif
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_ptr0 = A + (long long)row0 * row_stride;
    const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * row_stride;
    const int boff = tid * 4;
    const unsigned int row_weight_offset0 = (unsigned int)((long long)row0 * row_stride);
    const unsigned int row_weight_offset1 =
        (unsigned int)((long long)(have_row1 ? row1 : row0) * row_stride);
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    // One matrix-wide SRD keeps the dual-row kernel at 94 VGPR / zero scratch.
    const auto weight_rsrc = HIPFIRE_WEIGHT_RSRC(A);
#endif

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float bcc0 = 0.0f, bcc1 = 0.0f, bcc2 = 0.0f, bcc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define DOG_X8(sc, zp, pk, a, x0, x1, x2, x3, x4, x5, x6, x7) do { \
        (a) += ((sc) * (float)((pk) & 0xFu)         + (zp)) * (x0) \
             + ((sc) * (float)(((pk) >> 4) & 0xFu)  + (zp)) * (x1) \
             + ((sc) * (float)(((pk) >> 8) & 0xFu)  + (zp)) * (x2) \
             + ((sc) * (float)(((pk) >> 12) & 0xFu) + (zp)) * (x3) \
             + ((sc) * (float)(((pk) >> 16) & 0xFu) + (zp)) * (x4) \
             + ((sc) * (float)(((pk) >> 20) & 0xFu) + (zp)) * (x5) \
             + ((sc) * (float)(((pk) >> 24) & 0xFu) + (zp)) * (x6) \
             + ((sc) * (float)(((pk) >> 28) & 0xFu) + (zp)) * (x7); \
    } while (0)

    #if HIPFIRE_GFX1151_RESIDUAL_HYBRID_BUFFER
    #define HIPFIRE_RESIDUAL_LOAD_SC(weight_rsrc, gp, weight_offset) (*(const unsigned int*)(gp))
    #define HIPFIRE_RESIDUAL_LOAD_ZP(weight_rsrc, gp, weight_offset) (*(const unsigned int*)((gp) + 4))
    #else
    #define HIPFIRE_RESIDUAL_LOAD_SC(weight_rsrc, gp, weight_offset) HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, gp, weight_offset)
    #define HIPFIRE_RESIDUAL_LOAD_ZP(weight_rsrc, gp, weight_offset) HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, (gp) + 4, (weight_offset) + 4)
    #endif

    #define LOAD_W(gp, weight_offset, sc, zp, pk) do { \
        const unsigned int hA = HIPFIRE_RESIDUAL_LOAD_SC(weight_rsrc, gp, weight_offset); \
        const unsigned int hB = HIPFIRE_RESIDUAL_LOAD_ZP(weight_rsrc, gp, weight_offset); \
        const unsigned int hs = (tid < 16) ? hA : hB; \
        (sc) = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu))); \
        (zp) = __half2float(__ushort_as_half((unsigned short)(hs >> 16))); \
        (pk) = HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, (gp) + 8 + boff, (weight_offset) + 8 + boff); \
    } while (0)

#if defined(HIPFIRE_GFX1151_RESIDUAL_K4096)
    // Keep the fixed four-quad bound without expanding the body. Full unroll
    // doubles the instruction footprint; factor-two unroll spills private=16
    // and is not admissible for direct gfx1151 PM4 replay.
    #pragma unroll 1
#endif
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;

        // Hoist full 32-wide x for this quad (4 groups × 8) via float4 pairs.
        const float4 xv0a = *reinterpret_cast<const float4*>(x + base0);
        const float4 xv0b = *reinterpret_cast<const float4*>(x + base0 + 4);
        const float4 xv1a = *reinterpret_cast<const float4*>(x + base0 + 256);
        const float4 xv1b = *reinterpret_cast<const float4*>(x + base0 + 260);
        const float4 xv2a = *reinterpret_cast<const float4*>(x + base0 + 512);
        const float4 xv2b = *reinterpret_cast<const float4*>(x + base0 + 516);
        const float4 xv3a = *reinterpret_cast<const float4*>(x + base0 + 768);
        const float4 xv3b = *reinterpret_cast<const float4*>(x + base0 + 772);

        // Weight MEMORY phase: sequential along row0 then row1 (8 headers).
        float sc0a, zp0a, sc1a, zp1a, sc2a, zp2a, sc3a, zp3a;
        unsigned int pk0a, pk1a, pk2a, pk3a;
        float sc0b, zp0b, sc1b, zp1b, sc2b, zp2b, sc3b, zp3b;
        unsigned int pk0b, pk1b, pk2b, pk3b;

#if defined(__gfx1100__)
        LOAD_W(row_ptr0 + g * 136, row_weight_offset0 + g * 136, sc0a, zp0a, pk0a);
        LOAD_W(row_ptr0 + (g + 1) * 136, row_weight_offset0 + (g + 1) * 136, sc1a, zp1a, pk1a);
        LOAD_W(row_ptr0 + (g + 2) * 136, row_weight_offset0 + (g + 2) * 136, sc2a, zp2a, pk2a);
        LOAD_W(row_ptr0 + (g + 3) * 136, row_weight_offset0 + (g + 3) * 136, sc3a, zp3a, pk3a);

        // Finish row0 before making row1's headers live. The empty exact-gfx1100
        // dependency barrier prevents LLVM from hoisting the second row's loads
        // back across this point; it emits no ISA instruction.
        DOG_X8(sc0a, zp0a, pk0a, acc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(sc1a, zp1a, pk1a, acc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8(sc2a, zp2a, pk2a, acc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        DOG_X8(sc3a, zp3a, pk3a, acc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);

        asm volatile("" : "+v"(acc0), "+v"(acc1), "+v"(acc2), "+v"(acc3) :: "memory");

        LOAD_W(row_ptr1 + g * 136, row_weight_offset1 + g * 136, sc0b, zp0b, pk0b);
        LOAD_W(row_ptr1 + (g + 1) * 136, row_weight_offset1 + (g + 1) * 136, sc1b, zp1b, pk1b);
        LOAD_W(row_ptr1 + (g + 2) * 136, row_weight_offset1 + (g + 2) * 136, sc2b, zp2b, pk2b);
        LOAD_W(row_ptr1 + (g + 3) * 136, row_weight_offset1 + (g + 3) * 136, sc3b, zp3b, pk3b);
        DOG_X8(sc0b, zp0b, pk0b, bcc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(sc1b, zp1b, pk1b, bcc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8(sc2b, zp2b, pk2b, bcc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        DOG_X8(sc3b, zp3b, pk3b, bcc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);
#else
        LOAD_W(row_ptr0 + g * 136, row_weight_offset0 + g * 136, sc0a, zp0a, pk0a);
        LOAD_W(row_ptr0 + (g + 1) * 136, row_weight_offset0 + (g + 1) * 136, sc1a, zp1a, pk1a);
        LOAD_W(row_ptr0 + (g + 2) * 136, row_weight_offset0 + (g + 2) * 136, sc2a, zp2a, pk2a);
        LOAD_W(row_ptr0 + (g + 3) * 136, row_weight_offset0 + (g + 3) * 136, sc3a, zp3a, pk3a);
        LOAD_W(row_ptr1 + g * 136, row_weight_offset1 + g * 136, sc0b, zp0b, pk0b);
        LOAD_W(row_ptr1 + (g + 1) * 136, row_weight_offset1 + (g + 1) * 136, sc1b, zp1b, pk1b);
        LOAD_W(row_ptr1 + (g + 2) * 136, row_weight_offset1 + (g + 2) * 136, sc2b, zp2b, pk2b);
        LOAD_W(row_ptr1 + (g + 3) * 136, row_weight_offset1 + (g + 3) * 136, sc3b, zp3b, pk3b);

        DOG_X8(sc0a, zp0a, pk0a, acc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(sc0b, zp0b, pk0b, bcc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(sc1a, zp1a, pk1a, acc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8(sc1b, zp1b, pk1b, bcc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8(sc2a, zp2a, pk2a, acc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        DOG_X8(sc2b, zp2b, pk2b, bcc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        DOG_X8(sc3a, zp3a, pk3a, acc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);
        DOG_X8(sc3b, zp3b, pk3b, bcc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);
#endif
    }

    if (tail >= 1) {
        const int g = quads << 2;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float sc0, zp0, sc1, zp1;
        unsigned int pk0, pk1;
        LOAD_W(row_ptr0 + g * 136, row_weight_offset0 + g * 136, sc0, zp0, pk0);
        LOAD_W(row_ptr1 + g * 136, row_weight_offset1 + g * 136, sc1, zp1, pk1);
        DOG_X8(sc0, zp0, pk0, acc0,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8(sc1, zp1, pk1, bcc0,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float sc0, zp0, sc1, zp1;
        unsigned int pk0, pk1;
        LOAD_W(row_ptr0 + g * 136, row_weight_offset0 + g * 136, sc0, zp0, pk0);
        LOAD_W(row_ptr1 + g * 136, row_weight_offset1 + g * 136, sc1, zp1, pk1);
        DOG_X8(sc0, zp0, pk0, acc1,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8(sc1, zp1, pk1, bcc1,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float sc0, zp0, sc1, zp1;
        unsigned int pk0, pk1;
        LOAD_W(row_ptr0 + g * 136, row_weight_offset0 + g * 136, sc0, zp0, pk0);
        LOAD_W(row_ptr1 + g * 136, row_weight_offset1 + g * 136, sc1, zp1, pk1);
        DOG_X8(sc0, zp0, pk0, acc2,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8(sc1, zp1, pk1, bcc2,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    #undef LOAD_W
    #undef HIPFIRE_RESIDUAL_LOAD_SC
    #undef HIPFIRE_RESIDUAL_LOAD_ZP
    #undef DOG_X8

    float acc = (acc0 + acc1) + (acc2 + acc3);
    float bcc = (bcc0 + bcc1) + (bcc2 + bcc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        bcc += __shfl_down(bcc, offset);

    if (tid == 0) {
        if (have_row1) {
            float2 yy = *reinterpret_cast<float2*>(y + row0);
            yy.x += acc;
            yy.y += bcc;
            *reinterpret_cast<float2*>(y + row0) = yy;
        } else {
            y[row0] += acc;
        }
    }
}

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1200__) || defined(__gfx1201__)
// Group-major twin of gemv_mq4g256v2_residual (residual-decode oracle):
// byte-identical except the A group base address. Shipping:
//   row_ptr0 = A + row0 * groups_per_row * 136; gp = row_ptr0 + g * 136
// (same for row1). Group-major: gp = A + 136 * (g * M + row). Same ABI
// (A, x, y, M, K), grid, block 32, LDS 0; same MEMORY-then-DOG order,
// tails, and y[row] += acc epilogue; bit-exact by construction.
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq4g256v2_residual_gm(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row0 = blockIdx.x << 1;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const bool have_row1 = row1 < M;
    const int tid = threadIdx.x;

#if defined(HIPFIRE_GFX1151_RESIDUAL_K4096)
    const int groups_per_row_gm = 16;
#else
    const int groups_per_row_gm = K / 256;
#endif
    const int boff_gm = tid * 4;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto weight_rsrc_gm = HIPFIRE_WEIGHT_RSRC(A);
#endif

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float bcc0 = 0.0f, bcc1 = 0.0f, bcc2 = 0.0f, bcc3 = 0.0f;
    const int quads_gm = groups_per_row_gm >> 2;
    const int tail_gm = groups_per_row_gm & 3;

    #define DOG_X8_GM(sc, zp, pk, a, x0, x1, x2, x3, x4, x5, x6, x7) do { \
        (a) += ((sc) * (float)((pk) & 0xFu)         + (zp)) * (x0) \
             + ((sc) * (float)(((pk) >> 4) & 0xFu)  + (zp)) * (x1) \
             + ((sc) * (float)(((pk) >> 8) & 0xFu)  + (zp)) * (x2) \
             + ((sc) * (float)(((pk) >> 12) & 0xFu) + (zp)) * (x3) \
             + ((sc) * (float)(((pk) >> 16) & 0xFu) + (zp)) * (x4) \
             + ((sc) * (float)(((pk) >> 20) & 0xFu) + (zp)) * (x5) \
             + ((sc) * (float)(((pk) >> 24) & 0xFu) + (zp)) * (x6) \
             + ((sc) * (float)(((pk) >> 28) & 0xFu) + (zp)) * (x7); \
    } while (0)

    #if HIPFIRE_GFX1151_RESIDUAL_HYBRID_BUFFER
    #define HIPFIRE_RESIDUAL_LOAD_SC_GM(weight_rsrc, gp, weight_offset) (*(const unsigned int*)(gp))
    #define HIPFIRE_RESIDUAL_LOAD_ZP_GM(weight_rsrc, gp, weight_offset) (*(const unsigned int*)((gp) + 4))
    #else
    #define HIPFIRE_RESIDUAL_LOAD_SC_GM(weight_rsrc, gp, weight_offset) HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, gp, weight_offset)
    #define HIPFIRE_RESIDUAL_LOAD_ZP_GM(weight_rsrc, gp, weight_offset) HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, (gp) + 4, (weight_offset) + 4)
    #endif

    // Group-major address invariants hoisted out of the group loop (mirrors
    // shipping's hoisted row_ptr/row_weight_offset): per-group qbase walks
    // one uniform stride per group, plus loop-invariant row offsets. The two
    // rows are adjacent, so row r+1's base is exactly +136 B past row r's
    // (+0 when the trailing odd row clamps r1 to r0): one base + uniform
    // delta replaces the second full A+qbase+roff computation per group.
    const long long m136_gm = (long long)M * 136;
    const long long r0off_gm = (long long)row0 * 136;
    const int r1s_gm = have_row1 ? row1 : row0;
    const long long r1off_gm = (long long)r1s_gm * 136;
    const long long r1_delta_gm = have_row1 ? 136 : 0;
    const unsigned int r1_delta_u_gm = (unsigned int)r1_delta_gm;
    // Lane-offset selection: on exact-gfx1100 the per-iteration launder below
    // keeps (stride+boff) offsets iteration-local (else LICM hoists them to
    // the preheader, +15 VGPR vs shipping's 81). Other archs use boff_gm
    // directly (parity already holds there; identical codegen).
    int boff_q_gm = boff_gm;
#if defined(__gfx1100__)
    #define BOFF_GM boff_q_gm
#else
    #define BOFF_GM boff_gm
#endif
    // Single-row loader: the exact-gfx1100 branch below keeps shipping's
    // row0-then-row1 split (with its scheduling barrier), so each row's four
    // groups load exactly as before, only off the group-major base.
    #define LOAD_W_GM_1R(qbase, roff, sc, zp, pk) do { \
        const char* gp_gm = A + (qbase) + (roff); \
        const unsigned int woff_gm = (unsigned int)((qbase) + (roff)); \
        const unsigned int hA_gm = HIPFIRE_RESIDUAL_LOAD_SC_GM(weight_rsrc_gm, gp_gm, woff_gm); \
        const unsigned int hB_gm = HIPFIRE_RESIDUAL_LOAD_ZP_GM(weight_rsrc_gm, gp_gm, woff_gm); \
        const unsigned int hs_gm = (tid < 16) ? hA_gm : hB_gm; \
        (sc) = __half2float(__ushort_as_half((unsigned short)(hs_gm & 0xFFFFu))); \
        (zp) = __half2float(__ushort_as_half((unsigned short)(hs_gm >> 16))); \
        (pk) = HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc_gm, (gp_gm) + 8 + BOFF_GM, (woff_gm) + 8 + BOFF_GM); \
    } while (0)
    #define LOAD_W_GM_PAIR(qbase, sc0, zp0, pk0, sc1, zp1, pk1) do { \
        const char* gp0_gm = A + (qbase) + r0off_gm; \
        const char* gp1_gm = gp0_gm + r1_delta_gm; \
        const unsigned int w0_gm = (unsigned int)((qbase) + r0off_gm); \
        const unsigned int w1_gm = w0_gm + r1_delta_u_gm; \
        const unsigned int hA0_gm = HIPFIRE_RESIDUAL_LOAD_SC_GM(weight_rsrc_gm, gp0_gm, w0_gm); \
        const unsigned int hB0_gm = HIPFIRE_RESIDUAL_LOAD_ZP_GM(weight_rsrc_gm, gp0_gm, w0_gm); \
        const unsigned int hs0_gm = (tid < 16) ? hA0_gm : hB0_gm; \
        (sc0) = __half2float(__ushort_as_half((unsigned short)(hs0_gm & 0xFFFFu))); \
        (zp0) = __half2float(__ushort_as_half((unsigned short)(hs0_gm >> 16))); \
        (pk0) = HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc_gm, (gp0_gm) + 8 + BOFF_GM, (w0_gm) + 8 + BOFF_GM); \
        const unsigned int hA1_gm = HIPFIRE_RESIDUAL_LOAD_SC_GM(weight_rsrc_gm, gp1_gm, w1_gm); \
        const unsigned int hB1_gm = HIPFIRE_RESIDUAL_LOAD_ZP_GM(weight_rsrc_gm, gp1_gm, w1_gm); \
        const unsigned int hs1_gm = (tid < 16) ? hA1_gm : hB1_gm; \
        (sc1) = __half2float(__ushort_as_half((unsigned short)(hs1_gm & 0xFFFFu))); \
        (zp1) = __half2float(__ushort_as_half((unsigned short)(hs1_gm >> 16))); \
        (pk1) = HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc_gm, (gp1_gm) + 8 + BOFF_GM, (w1_gm) + 8 + BOFF_GM); \
    } while (0)

    long long qb_gm = 0;
    const long long m4_gm = 4 * m136_gm;
    for (int q = 0; q < quads_gm; q++) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;
#if defined(__gfx1100__)
        boff_q_gm = boff_gm;
        asm volatile("" : "+v"(boff_q_gm));
#endif

        const float4 xv0a = *reinterpret_cast<const float4*>(x + base0);
        const float4 xv0b = *reinterpret_cast<const float4*>(x + base0 + 4);
        const float4 xv1a = *reinterpret_cast<const float4*>(x + base0 + 256);
        const float4 xv1b = *reinterpret_cast<const float4*>(x + base0 + 260);
        const float4 xv2a = *reinterpret_cast<const float4*>(x + base0 + 512);
        const float4 xv2b = *reinterpret_cast<const float4*>(x + base0 + 516);
        const float4 xv3a = *reinterpret_cast<const float4*>(x + base0 + 768);
        const float4 xv3b = *reinterpret_cast<const float4*>(x + base0 + 772);

        float sc0a, zp0a, sc1a, zp1a, sc2a, zp2a, sc3a, zp3a;
        unsigned int pk0a, pk1a, pk2a, pk3a;
        float sc0b, zp0b, sc1b, zp1b, sc2b, zp2b, sc3b, zp3b;
        unsigned int pk0b, pk1b, pk2b, pk3b;
        const long long q0_gm = qb_gm;
        const long long q1_gm = q0_gm + m136_gm;
        const long long q2_gm = q1_gm + m136_gm;
        const long long q3_gm = q2_gm + m136_gm;
        qb_gm += m4_gm;

#if defined(__gfx1100__)
        // Sequential within each half: one address pair + one value set live
        // at a time (reused sc0a/zp0a/pk0a), matching shipping's peak. Same
        // load bytes and same row0-quartet/barrier/row1-quartet DOG order;
        // bit-exact by construction.
        LOAD_W_GM_1R(q0_gm, r0off_gm, sc0a, zp0a, pk0a);
        DOG_X8_GM(sc0a, zp0a, pk0a, acc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        LOAD_W_GM_1R(q1_gm, r0off_gm, sc0a, zp0a, pk0a);
        DOG_X8_GM(sc0a, zp0a, pk0a, acc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        LOAD_W_GM_1R(q2_gm, r0off_gm, sc0a, zp0a, pk0a);
        DOG_X8_GM(sc0a, zp0a, pk0a, acc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        LOAD_W_GM_1R(q3_gm, r0off_gm, sc0a, zp0a, pk0a);
        DOG_X8_GM(sc0a, zp0a, pk0a, acc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);

        asm volatile("" : "+v"(acc0), "+v"(acc1), "+v"(acc2), "+v"(acc3) :: "memory");
        LOAD_W_GM_1R(q0_gm, r1off_gm, sc0a, zp0a, pk0a);
        DOG_X8_GM(sc0a, zp0a, pk0a, bcc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        LOAD_W_GM_1R(q1_gm, r1off_gm, sc0a, zp0a, pk0a);
        DOG_X8_GM(sc0a, zp0a, pk0a, bcc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        LOAD_W_GM_1R(q2_gm, r1off_gm, sc0a, zp0a, pk0a);
        DOG_X8_GM(sc0a, zp0a, pk0a, bcc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        LOAD_W_GM_1R(q3_gm, r1off_gm, sc0a, zp0a, pk0a);
        DOG_X8_GM(sc0a, zp0a, pk0a, bcc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);
#else
        LOAD_W_GM_PAIR(q0_gm, sc0a, zp0a, pk0a, sc0b, zp0b, pk0b);
        LOAD_W_GM_PAIR(q1_gm, sc1a, zp1a, pk1a, sc1b, zp1b, pk1b);
        LOAD_W_GM_PAIR(q2_gm, sc2a, zp2a, pk2a, sc2b, zp2b, pk2b);
        LOAD_W_GM_PAIR(q3_gm, sc3a, zp3a, pk3a, sc3b, zp3b, pk3b);

        DOG_X8_GM(sc0a, zp0a, pk0a, acc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8_GM(sc0b, zp0b, pk0b, bcc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8_GM(sc1a, zp1a, pk1a, acc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8_GM(sc1b, zp1b, pk1b, bcc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8_GM(sc2a, zp2a, pk2a, acc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        DOG_X8_GM(sc2b, zp2b, pk2b, bcc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        DOG_X8_GM(sc3a, zp3a, pk3a, acc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);
        DOG_X8_GM(sc3b, zp3b, pk3b, bcc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);
#endif
    }

    if (tail_gm >= 1) {
        const int g = quads_gm << 2;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float sc0, zp0, sc1, zp1;
        unsigned int pk0, pk1;
        LOAD_W_GM_PAIR(qb_gm, sc0, zp0, pk0, sc1, zp1, pk1);
        qb_gm += m136_gm;
        DOG_X8_GM(sc0, zp0, pk0, acc0,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8_GM(sc1, zp1, pk1, bcc0,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    if (tail_gm >= 2) {
        const int g = (quads_gm << 2) + 1;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float sc0, zp0, sc1, zp1;
        unsigned int pk0, pk1;
        LOAD_W_GM_PAIR(qb_gm, sc0, zp0, pk0, sc1, zp1, pk1);
        qb_gm += m136_gm;
        DOG_X8_GM(sc0, zp0, pk0, acc1,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8_GM(sc1, zp1, pk1, bcc1,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    if (tail_gm >= 3) {
        const int g = (quads_gm << 2) + 2;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float sc0, zp0, sc1, zp1;
        unsigned int pk0, pk1;
        LOAD_W_GM_PAIR(qb_gm, sc0, zp0, pk0, sc1, zp1, pk1);
        DOG_X8_GM(sc0, zp0, pk0, acc2,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8_GM(sc1, zp1, pk1, bcc2,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    #undef LOAD_W_GM_PAIR
    #undef LOAD_W_GM_1R
    #undef HIPFIRE_RESIDUAL_LOAD_SC_GM
    #undef HIPFIRE_RESIDUAL_LOAD_ZP_GM
    #undef DOG_X8_GM

    float acc = (acc0 + acc1) + (acc2 + acc3);
    float bcc = (bcc0 + bcc1) + (bcc2 + bcc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        bcc += __shfl_down(bcc, offset);

    if (tid == 0) {
        if (have_row1) {
            float2 yy = *reinterpret_cast<float2*>(y + row0);
            yy.x += acc;
            yy.y += bcc;
            *reinterpret_cast<float2*>(y + row0) = yy;
        } else {
            y[row0] += acc;
        }
    }
}
#endif // GM twin (gfx11 + gfx12)
