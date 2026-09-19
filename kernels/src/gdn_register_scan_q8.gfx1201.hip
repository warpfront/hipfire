// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx1201 register-resident GDN scan (chunk-64, split-2, f16 state).
//
// Geometry: grid [n_heads,2,1], block [256] (eight wave32 waves). One WG per
// (value_head, value_row_half). wave=tid/32, ml=lane&15, kg=lane>>4,
// value_group=wave/2, key_half=wave%2. Value row = split*64+group*16+ml.
// State fragment S[b][j] <-> key key_half*64+b*16+kg*8+j, b=0..3, j=0..7:
// 16x64 f16 elements per wave in registers (16 VGPR), never in LDS.
// Delta fragments D[t][j] (f32, 32 VGPR) <-> token t*16+kg*8+j of the lane's
// own value row.
//
// Per-C64 chunk (plan 3.2): inclusive gate prefix G, E=exp(G),
// L[i,j]=beta_i*exp(G_i-G_j)*(k_i.k_j) j<i, rhs=beta*(v-E*S_in*k),
// (I+L)*delta=rhs once (f32 forward substitution, wave-local, coefficients
// broadcast from column-major expanded L in LDS), M[i,j]=exp(G_i-G_j)*(q_i.k_j)
// j<=i, out=E*S_in*q+M*delta, S_out=E_last*S_in+sum (E_last/E_j)*delta_j*k_j^T.
// WMMA operands f16, accumulators/solve f32. Delta rounded to half only as a
// WMMA operand. S promoted for the four carry K16 steps, rounded once per C64.
// Commit folds old EF in f32 and requants with the exact shipping epilogue
// arithmetic (rint/clamp, zero-row scale 1). No EF fold at entry.
//
// LDS (static 25,728 B, phase-overlaid, no S/delta spill):
//   CMP  [0,8320)      compact strict-lower L (2016 f32) / lower M (2080 f32)
//   HOLE [8320,24704)  F16K full-K f16 panel | EXP expanded L/M col-major f32
//                       | XCH pair-exchange 8192 + K16/Q16/Kt panels
//   VEC  [24704,25728) G[64] E[64] BETA[64] ELAST Ge + pad
// Live peaks: Gram 25,728; S.K 29,696; QK 29,824; fused S.Q/M.d 29,696.
//
// Admitted: exact gfx1201, H48/D128 dense Qwen3.8-27B shape (kernel takes
// n_heads generally), Q8+EF ordinary sequential prefill, 64<=N<=512 per
// launch. Ragged final C64 uses masks. N<64 and other shapes must use the
// unchanged serial route (host admission, not probed here).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define GDN_RS_HD 128
#define GDN_RS_C 64
// Static LDS in floats (total 7456 = 29,824 B, cap 30,720):
//   CMP  [0,2080)      compact strict-lower L (2016) / lower M (2080).
//                        Dead during S.K/fused/carry/commit; reused as XCH.
//   HOLE [2080,6176)   F16K full-K f16 panel | EXP expanded L/M col-major.
//   PANF [6176,7200)   K16/Q16 streaming f16 panels (2048 halves).
//   VEC  [7200,7456)   G[64] E[64] BETA[64] ELAST Ge + pad.
// Live peaks: Gram 25,728; expand 25,728; S.K 29,696; QK 29,824; fused 29,696.
#define GDN_RS_LDS_FLOATS 7456
#define GDN_RS_CMP 0
#define GDN_RS_HOLE 2080
// Pair-exchange (8192 B) reuses the CMP footprint; CMP is dead whenever XCH
// is live (S.K pair-sum, fused S.Q pair-sum, commit row-max).
#define GDN_RS_XCH 0
// K16/Q16 streaming f16 panel (16x128 = 2048 halves = 1024 floats).
#define GDN_RS_PANEL 6176
// Carry Kt panels (two key halves x 64 keys x 64 tokens f16 = 8192 halves =
// 4096 floats) exactly fill HOLE; everything else is dead during carry.
#define GDN_RS_KT 2080
#define GDN_RS_EXP 2080
#define GDN_RS_VEC 7200

typedef _Float16 __attribute__((ext_vector_type(8))) rs_h8;
typedef float __attribute__((ext_vector_type(8))) rs_f8;

__device__ __forceinline__ rs_h8 rs_halfv(rs_f8 x) {
    return __builtin_convertvector(x, rs_h8);
}
__device__ __forceinline__ rs_f8 rs_floatv(rs_h8 x) {
    return __builtin_convertvector(x, rs_f8);
}
__device__ __forceinline__ rs_f8 rs_mma(rs_h8 a, rs_h8 b, rs_f8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}


extern "C" __global__ void gated_delta_net_q8_register_scan_gfx1201(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    signed char* __restrict__ s_q8,
    float* __restrict__ s_scales,
    float* __restrict__ output,
    int n_tokens,
    int n_heads,
    int head_dim,
    unsigned int frame,
    __half* __restrict__ s_ef_residual) {
    const int h = blockIdx.x;
    const int split = blockIdx.y;
    if (h >= n_heads || head_dim != GDN_RS_HD || n_tokens <= 0) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int kg = lane >> 4;
    const int wave = tid >> 5;
    const int vg = wave >> 1;
    const int kh = wave & 1;
    const int vrow = split * 64 + vg * 16 + ml;
    const int stride = n_heads * GDN_RS_HD;
    // All global offsets fit 32 bits (max ~3.2M elements); keep index math
    // 32-bit to halve address registers. Pointers stay 64-bit.
    const unsigned ustride = (unsigned)(n_heads * GDN_RS_HD);
    const unsigned h_off = (unsigned)h * GDN_RS_HD;
    const unsigned lane_base =
        (unsigned)blockIdx.z * (unsigned)(n_heads * GDN_RS_HD * GDN_RS_HD);
    const unsigned h_state = h_off * GDN_RS_HD;

    __shared__ float LDSB[GDN_RS_LDS_FLOATS];

    // ---- entry: reconstruct this WG's value-half state, f16, no EF fold ----
    rs_h8 S[4];
    {
        const signed char* src = s_q8 + lane_base + h_state +
                                 (unsigned)vrow * GDN_RS_HD + (unsigned)(kh * 64);
        const float sc =
            s_scales[lane_base + h_off + (unsigned)vrow];
#pragma unroll
        for (int b = 0; b < 4; b++) {
            rs_h8 frag;
            #pragma unroll
            for (int j = 0; j < 8; j++)
                frag[j] = (__half)(sc * (float)src[b * 16 + kg * 8 + j]);
            S[b] = frag;
        }
    }
    rs_f8 D[4] = {{0, 0, 0, 0, 0, 0, 0, 0},
                  {0, 0, 0, 0, 0, 0, 0, 0},
                  {0, 0, 0, 0, 0, 0, 0, 0},
                  {0, 0, 0, 0, 0, 0, 0, 0}};

    for (int c0 = 0; c0 < n_tokens; c0 += GDN_RS_C) {
        const int cn = (c0 + GDN_RS_C < n_tokens) ? GDN_RS_C : (n_tokens - c0);

        // ---- gate vectors: G prefix (serial, lane 0), E, BETA ----
        // All 256 threads participate (slot tid&63 written 4x identically).
        {
            const int i = tid & 63;
            const int t = c0 + i;
            const int tc = (t < n_tokens) ? t : (n_tokens - 1);
            float gv = gate[(unsigned)tc * (unsigned)n_heads + (unsigned)h];
            float bv = beta[(unsigned)tc * (unsigned)n_heads + (unsigned)h];
            const bool valid = (t < n_tokens);
            LDSB[GDN_RS_VEC + i] = valid ? gv : 0.0f;
            LDSB[GDN_RS_VEC + 128 + i] = valid ? bv : 0.0f;
        }
        __syncthreads();
        if (tid == 0) {
            float run = 0.0f;
#pragma clang loop unroll(disable)
            for (int i = 0; i < 64; i++) {
                run += LDSB[GDN_RS_VEC + i];
                LDSB[GDN_RS_VEC + i] = run;
            }
        }
        __syncthreads();
        { const int i = tid & 63; LDSB[GDN_RS_VEC + 64 + i] = expf(LDSB[GDN_RS_VEC + i]); }
        if (tid == 0) {
            LDSB[GDN_RS_VEC + 192] = expf(LDSB[GDN_RS_VEC + (cn - 1)]);
            LDSB[GDN_RS_VEC + 193] = LDSB[GDN_RS_VEC + (cn - 1)];
        }
        __syncthreads();

        // ---- stage full K f16 panel (64x128) into HOLE ----
        {
            _Float16* f16k = reinterpret_cast<_Float16*>(LDSB + GDN_RS_HOLE);
#pragma clang loop unroll(disable)
            for (int u = 0; u < 4; u++) {
                const int x = tid * 8 + u * 2048;
                const int tc = x / 128;
                const int kc = x % 128;
                const int gt = c0 + tc;
                float4 w = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 w2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                // Clamped load + select (no divergent branch over w/pack).
                const int gtc = (gt < n_tokens) ? gt : (n_tokens - 1);
                {
                    const float* kp = k + (unsigned)gtc * ustride + h_off + (unsigned)kc;
                    float4 lw = *reinterpret_cast<const float4*>(kp);
                    float4 lw2 = *reinterpret_cast<const float4*>(kp + 4);
                    const bool valid = (gt < n_tokens);
                    w.x = valid ? lw.x : 0.0f;                     w.y = valid ? lw.y : 0.0f;
                    w.z = valid ? lw.z : 0.0f;                     w.w = valid ? lw.w : 0.0f;
                    w2.x = valid ? lw2.x : 0.0f;                     w2.y = valid ? lw2.y : 0.0f;
                    w2.z = valid ? lw2.z : 0.0f;                     w2.w = valid ? lw2.w : 0.0f;
                }
                rs_f8 pack;
                pack[0] = w.x; pack[1] = w.y; pack[2] = w.z; pack[3] = w.w;
                pack[4] = w2.x; pack[5] = w2.y; pack[6] = w2.z; pack[7] = w2.w;
                *(reinterpret_cast<rs_h8*>(f16k + x)) = rs_halfv(pack);
            }
        }
        __syncthreads();

        // ---- KK Gram -> compact strict-lower L (packed row i: i(i-1)/2+j) ----
        // WMMA rule (see FA2 header): lane (ml,kg) supplies A[ml][kg*8+j] and
        // B[ml][kg*8+j]; acc[j] = C[8*kg+j][ml]. So C rows vary with j and C
        // cols with ml. A_mat rows R0+ml (R0 = vg*16, pair-redundant across
        // the key-half wave pair); B_mat rows p*16+ml. acc[j] covers Gram row
        // R0+8*kg+j, col p*16+ml: all 32 lanes write disjoint packed slots.
        {
            const _Float16* f16k = reinterpret_cast<const _Float16*>(LDSB + GDN_RS_HOLE);
            const int R0 = vg * 16;
#pragma clang loop unroll(disable)
            for (int p = 0; p < 4; p++) {
                rs_f8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma clang loop unroll(disable)
                for (int s = 0; s < 8; s++) {
                    rs_h8 A, B;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        A[j] = f16k[(R0 + ml) * 128 + s * 16 + kg * 8 + j];
                        B[j] = f16k[(p * 16 + ml) * 128 + s * 16 + kg * 8 + j];
                    }
                    acc = rs_mma(A, B, acc);
                }
                // Packed writes need valid slots: L row 0 has none, and
                // upper-triangle indices overlap later rows' slots. Invalid
                // lanes store junk 0.0 to dead slot CMP[2079] (free during
                // the L phase: L uses 0..2015). Masked rows still write real
                // slots (0.0) so no stale data survives across chunks.
                {
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const int row = R0 + kg * 8 + j;
                        const int col = p * 16 + ml;
                        const float bv = (row < cn) ? LDSB[GDN_RS_VEC + 128 + row] : 0.0f;
                        const float Gr = LDSB[GDN_RS_VEC + row];
                        const int poff = row * (row - 1) / 2;
                        float lv = 0.0f;
                        bool valid = (col < row && row < cn && row != 0);
                        if (valid)
                            lv = bv * expf(Gr - LDSB[GDN_RS_VEC + col]) * acc[j];
                        LDSB[GDN_RS_CMP + (valid ? poff + col : 2079)] = lv;
                    }
                }
            }
        }
        __syncthreads();
        // ---- expand compact L -> EXP column-major (ZERO diagonal: the unit
        // is implicit in the rank-update solve; a 1.0 here would subtract
#pragma clang loop unroll(disable)
        for (int u = 0; u < 16; u++) {
            const int x = tid + u * 256;
            const int i = x / 64;
            const int j = x % 64;
            float e = 0.0f;
            if (j < i) e = LDSB[GDN_RS_CMP + i * (i - 1) / 2 + j];
            LDSB[GDN_RS_EXP + j * 64 + i] = e;
        }
        __syncthreads();

        // ---- S.K (K16 panels restreamed into CMP space) + pair-sum + rhs ----
        {
            _Float16* panel = reinterpret_cast<_Float16*>(LDSB + GDN_RS_PANEL);
#pragma clang loop unroll(disable)
            for (int tt = 0; tt < 4; tt++) {
                // stream 16 K rows for this token tile: 2048 halves, one h8
                // per thread (x = tid*8 covers [0,2048))
                {
                    const int x = tid * 8;
                    const int tr = x / 128;
                    const int kc = x % 128;
                    const int gt = c0 + tt * 16 + tr;
                    float4 w = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    float4 w2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    // Clamped load + select (no divergent branch over w/pack).
                    const int gtc = (gt < n_tokens) ? gt : (n_tokens - 1);
                    {
                        const float* kp =
                            k + (unsigned)gtc * ustride + h_off + (unsigned)kc;
                        float4 lw = *reinterpret_cast<const float4*>(kp);
                        float4 lw2 = *reinterpret_cast<const float4*>(kp + 4);
                        const bool valid = (gt < n_tokens);
                        w.x = valid ? lw.x : 0.0f;                         w.y = valid ? lw.y : 0.0f;
                        w.z = valid ? lw.z : 0.0f;                         w.w = valid ? lw.w : 0.0f;
                        w2.x = valid ? lw2.x : 0.0f;                         w2.y = valid ? lw2.y : 0.0f;
                        w2.z = valid ? lw2.z : 0.0f;                         w2.w = valid ? lw2.w : 0.0f;
                    }
                    rs_f8 pack;
                    pack[0] = w.x; pack[1] = w.y; pack[2] = w.z; pack[3] = w.w;
                    pack[4] = w2.x; pack[5] = w2.y; pack[6] = w2.z; pack[7] = w2.w;
                    *(reinterpret_cast<rs_h8*>(panel + x)) = rs_halfv(pack);
                }
                __syncthreads();
                rs_f8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    const int s = kh * 4 + b;
                    rs_h8 A;
                    #pragma unroll
                    for (int j = 0; j < 8; j++)
                        A[j] = panel[ml * 128 + s * 16 + kg * 8 + j];
                    acc = rs_mma(A, S[b], acc);
                }
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    LDSB[GDN_RS_XCH + (wave * 32 + lane) * 8 + j] = acc[j];
                __syncthreads();
                // rhs into D[tt]; pair-sum folded inline (no sk array)
                rs_f8 dd;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int tok = tt * 16 + kg * 8 + j;
                    const int gt = c0 + tok;
                    float vv = 0.0f;
                    {
                        const int gtc = (gt < n_tokens) ? gt : (n_tokens - 1);
                        const float lv =
                            v[(unsigned)gtc * ustride + h_off + (unsigned)vrow];
                        vv = (gt < n_tokens) ? lv : 0.0f;
                    }
                    const float ee = LDSB[GDN_RS_VEC + 64 + tok];
                    const float bb = (tok < cn) ? LDSB[GDN_RS_VEC + 128 + tok] : 0.0f;
                    const float skj = acc[j] +
                                      LDSB[GDN_RS_XCH + (((wave ^ 1) * 32 + lane)) * 8 + j];
                    dd[j] = (tok < cn) ? (bb * (vv - ee * skj)) : 0.0f;
                }
                // Uniform switch: constant index keeps D in registers while
                // the bounded tt loop serializes copies for the scheduler.
                switch (tt) {
                    case 0: D[0] = dd; break;
                    case 1: D[1] = dd; break;
                    case 2: D[2] = dd; break;
                    default: D[3] = dd; break;
                }
            }
        }
        __syncthreads();

        // ---- solve (I+L)*delta=rhs: 64 rank updates, wave-local ----
#pragma clang loop unroll(disable)
        for (int i = 0; i < 64; i++) {
            float vv = 0.0f;
#pragma unroll
            for (int t = 0; t < 4; t++) {
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    vv = (t * 16 + kg * 8 + j == i) ? D[t][j] : vv;
            }
            vv += __shfl_xor(vv, 16);
            const float l0 = LDSB[GDN_RS_EXP + i * 64 + lane];
            const float l1 = LDSB[GDN_RS_EXP + i * 64 + 32 + lane];
#pragma unroll
            for (int t = 0; t < 4; t++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int r = t * 16 + kg * 8 + j;
                    const float c = __shfl(r < 32 ? l0 : l1, r & 31);
                    D[t][j] -= c * vv;
                }
            }
            asm volatile(
                "" : "+v"(D[0]), "+v"(D[1]), "+v"(D[2]), "+v"(D[3]) : : "memory");
        }

        // ---- restage full K f16 (EXP dead) ----
        {
            _Float16* f16k = reinterpret_cast<_Float16*>(LDSB + GDN_RS_HOLE);
#pragma clang loop unroll(disable)
            for (int u = 0; u < 4; u++) {
                const int x = tid * 8 + u * 2048;
                const int tc = x / 128;
                const int kc = x % 128;
                const int gt = c0 + tc;
                float4 w = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 w2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                // Clamped load + select (no divergent branch over w/pack).
                const int gtc = (gt < n_tokens) ? gt : (n_tokens - 1);
                {
                    const float* kp = k + (unsigned)gtc * ustride + h_off + (unsigned)kc;
                    float4 lw = *reinterpret_cast<const float4*>(kp);
                    float4 lw2 = *reinterpret_cast<const float4*>(kp + 4);
                    const bool valid = (gt < n_tokens);
                    w.x = valid ? lw.x : 0.0f;                     w.y = valid ? lw.y : 0.0f;
                    w.z = valid ? lw.z : 0.0f;                     w.w = valid ? lw.w : 0.0f;
                    w2.x = valid ? lw2.x : 0.0f;                     w2.y = valid ? lw2.y : 0.0f;
                    w2.z = valid ? lw2.z : 0.0f;                     w2.w = valid ? lw2.w : 0.0f;
                }
                rs_f8 pack;
                pack[0] = w.x; pack[1] = w.y; pack[2] = w.z; pack[3] = w.w;
                pack[4] = w2.x; pack[5] = w2.y; pack[6] = w2.z; pack[7] = w2.w;
                *(reinterpret_cast<rs_h8*>(f16k + x)) = rs_halfv(pack);
            }
        }
        __syncthreads();

        // ---- QK Gram -> compact lower-incl M (packed i(i+1)/2+j) ----
        // Same WMMA rule: acc[j] = C[8*kg+j][ml]. A_mat rows R0+ml (Q rows,
        // R0 = vg*16 pair-redundant, streamed from global); B_mat rows
        // p*16+ml (K rows from staged f16k). acc[j] covers QK row R0+8*kg+j,
        // col p*16+ml: all lanes write disjoint packed slots.
        {
            const _Float16* f16k = reinterpret_cast<const _Float16*>(LDSB + GDN_RS_HOLE);
            const int R0 = vg * 16;
#pragma clang loop unroll(disable)
            for (int p = 0; p < 4; p++) {
                rs_f8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma clang loop unroll(disable)
                for (int s = 0; s < 8; s++) {
                    rs_h8 A, B;
                    // Branchless masked load: clamp index, zero value.
                    {
                        const int ar = R0 + ml;
                        const int agt = c0 + ar;
                        const int agtc = (agt < n_tokens) ? agt : (n_tokens - 1);
                        const float* ap =
                            q + (unsigned)agtc * ustride + h_off + s * 16 + kg * 8;
                        #pragma unroll
                        for (int j = 0; j < 8; j++)
                            A[j] = (__half)(((agt < n_tokens) ? ap[j] : 0.0f));
                    }
                    #pragma unroll
                    for (int j = 0; j < 8; j++)
                        B[j] = f16k[(p * 16 + ml) * 128 + s * 16 + kg * 8 + j];
                    acc = rs_mma(A, B, acc);
                }
                // Invalid (col>row) packed indices overlap later rows' slots;
                // route junk to dead EXP[0] (expanded L is dead after solve).
                // Masked rows still write real slots (0.0) against staleness.
                {
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const int row = R0 + kg * 8 + j;
                        const int col = p * 16 + ml;
                        const float Gr = LDSB[GDN_RS_VEC + row];
                        const int poff = row * (row + 1) / 2;
                        float mv = 0.0f;
                        bool valid = (col <= row && row < cn);
                        if (valid)
                            mv = expf(Gr - LDSB[GDN_RS_VEC + col]) * acc[j];
                        // Single LDS store with per-lane address (no branch).
                        LDSB[valid ? GDN_RS_CMP + poff + col : GDN_RS_EXP] = mv;
                    }
                }
                __syncthreads();
            }
        }

        // ---- expand compact M -> EXP (zero upper) ----
#pragma clang loop unroll(disable)
        for (int u = 0; u < 16; u++) {
            const int x = tid + u * 256;
            const int i = x / 64;
            const int j = x % 64;
            float e = 0.0f;
            if (j <= i) e = LDSB[GDN_RS_CMP + i * (i + 1) / 2 + j];
            LDSB[GDN_RS_EXP + j * 64 + i] = e;
        }
        __syncthreads();

        // ---- fused S.Q tile + M.delta tile + output per tt ----
        {
            _Float16* qpanel = reinterpret_cast<_Float16*>(LDSB + GDN_RS_PANEL);
#pragma clang loop unroll(disable)
            for (int tt = 0; tt < 4; tt++) {
                // stream 16 Q rows for this token tile: 2048 halves, one h8
                // per thread (x = tid*8 covers [0,2048))
                {
                    const int x = tid * 8;
                    const int tr = x / 128;
                    const int kc = x % 128;
                    const int gt = c0 + tt * 16 + tr;
                    float4 w = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    float4 w2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    // Clamped load + select (no divergent branch over w/pack).
                    const int gtc = (gt < n_tokens) ? gt : (n_tokens - 1);
                    {
                        const float* qp =
                            q + (unsigned)gtc * ustride + h_off + (unsigned)kc;
                        float4 lw = *reinterpret_cast<const float4*>(qp);
                        float4 lw2 = *reinterpret_cast<const float4*>(qp + 4);
                        const bool valid = (gt < n_tokens);
                        w.x = valid ? lw.x : 0.0f;                         w.y = valid ? lw.y : 0.0f;
                        w.z = valid ? lw.z : 0.0f;                         w.w = valid ? lw.w : 0.0f;
                        w2.x = valid ? lw2.x : 0.0f;                         w2.y = valid ? lw2.y : 0.0f;
                        w2.z = valid ? lw2.z : 0.0f;                         w2.w = valid ? lw2.w : 0.0f;
                    }
                    rs_f8 pack;
                    pack[0] = w.x; pack[1] = w.y; pack[2] = w.z; pack[3] = w.w;
                    pack[4] = w2.x; pack[5] = w2.y; pack[6] = w2.z; pack[7] = w2.w;
                    *(reinterpret_cast<rs_h8*>(qpanel + x)) = rs_halfv(pack);
                }
                __syncthreads();
                rs_f8 qacc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    const int s = kh * 4 + b;
                    rs_h8 A;
                    #pragma unroll
                    for (int j = 0; j < 8; j++)
                        A[j] = qpanel[ml * 128 + s * 16 + kg * 8 + j];
                    qacc = rs_mma(A, S[b], qacc);
                }
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    LDSB[GDN_RS_XCH + (wave * 32 + lane) * 8 + j] = qacc[j];
                __syncthreads();
                // S.Q pair-sum stays in XCH (mine + partner partials) until
                // the output below; M.d does not touch XCH. No sq array.
                // Uniform switch on s: constant D index keeps delta in regs;
                // each case is its own scheduler region.
                rs_f8 res = {0, 0, 0, 0, 0, 0, 0, 0};
                for (int s = 0; s < 4; s++) {
                    rs_h8 A2, B2;
                    const rs_f8 Ds = (s == 0) ? D[0]
                                     : (s == 1) ? D[1] : (s == 2) ? D[2] : D[3];
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        A2[j] = (__half)LDSB[GDN_RS_EXP +
                                             (s * 16 + kg * 8 + j) * 64 + tt * 16 + ml];
                        B2[j] = (__half)(Ds[j]);
                    }
                    res = rs_mma(A2, B2, res);
                }
                // Both key-half waves write identical outputs (benign race,
                // same value) so no divergent branch spans res/XCH.
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int tok = tt * 16 + kg * 8 + j;
                    const int gt = c0 + tok;
                    const int gtc = (gt < n_tokens) ? gt : (n_tokens - 1);
                    const float ee = LDSB[GDN_RS_VEC + 64 + tok];
                    const float sqj =
                        LDSB[GDN_RS_XCH + (wave * 32 + lane) * 8 + j] +
                        LDSB[GDN_RS_XCH + (((wave ^ 1) * 32 + lane)) * 8 + j];
                    const float ov = ee * sqj + res[j];
                    if (gt < n_tokens)
                        output[(unsigned)gtc * ustride + h_off + (unsigned)vrow] = ov;
                }
                __syncthreads();
            }
        }

        // ---- carry: two Kt panels (key half x 64 keys x 64 tokens f16) ----
        {
            _Float16* kt = reinterpret_cast<_Float16*>(LDSB + GDN_RS_KT);
            const int kt_lane = vg * 32 + lane;
#pragma clang loop unroll(disable)
            for (int u = 0; u < 32; u++) {
                const int x = kt_lane + u * 128;
                const int r = x / 64;
                const int tc = x % 64;
                const int gt = c0 + tc;
                float kk = 0.0f;
                {
                    const int gtc = (gt < n_tokens) ? gt : (n_tokens - 1);
                    const float lv = k[(unsigned)gtc * ustride + h_off +
                                       (unsigned)(kh * 64) + (unsigned)r];
                    kk = (gt < n_tokens) ? lv : 0.0f;
                }
                kt[kh * 4096 + x] = (__half)kk;
            }
            __syncthreads();
            const float elast = LDSB[GDN_RS_VEC + 192];
            const float Ge = LDSB[GDN_RS_VEC + 193];
#pragma clang loop unroll(disable)
            for (int kb = 0; kb < 4; kb++) {
                // Constant S index via uniform select (kb bounded).
                const rs_h8 Skb =
                    (kb == 0) ? S[0] : (kb == 1) ? S[1] : (kb == 2) ? S[2] : S[3];
                rs_f8 acc = rs_floatv(Skb);
                #pragma unroll
                for (int j = 0; j < 8; j++) acc[j] *= elast;
#pragma clang loop unroll(disable)
                for (int s = 0; s < 4; s++) {
                    rs_h8 A, B;
                    const rs_f8 Ds = (s == 0) ? D[0]
                                     : (s == 1) ? D[1] : (s == 2) ? D[2] : D[3];
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const int stok = s * 16 + kg * 8 + j;
                        A[j] =
                            kt[kh * 4096 + (kb * 16 + ml) * 64 + stok];
                        const float cf =
                            (stok < cn) ? expf(Ge - LDSB[GDN_RS_VEC + stok]) : 0.0f;
                        B[j] = (__half)(cf * Ds[j]);
                    }
                    acc = rs_mma(A, B, acc);
                }
                // Constant S index via uniform select.
                if (kb == 0) S[0] = rs_halfv(acc);
                else if (kb == 1) S[1] = rs_halfv(acc);
                else if (kb == 2) S[2] = rs_halfv(acc);
                else S[3] = rs_halfv(acc);
            }
        }
        __syncthreads();
    }

    // ---- commit: promote S, fold old EF once, requant (shipping arithmetic)
    {
        const bool use_ef = (s_ef_residual != nullptr);
        __half* efr_base =
            use_ef ? (s_ef_residual + lane_base + h_state +
                      (unsigned)vrow * GDN_RS_HD)
                   : nullptr;
        // Pass 1: row absmax over promoted S + old EF (no pre array).
        float my_max = 0.0f;
#pragma unroll
        for (int b = 0; b < 4; b++) {
            rs_f8 c = rs_floatv(S[b]);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float x = c[j];
                if (use_ef) x += __half2float(efr_base[kh * 64 + b * 16 + kg * 8 + j]);
                my_max = fmaxf(my_max, fabsf(x));
            }
        }
        LDSB[GDN_RS_XCH + tid] = my_max;
        __syncthreads();
        const int w0 = vg * 2;
        float m = LDSB[GDN_RS_XCH + w0 * 32 + ml];
        m = fmaxf(m, LDSB[GDN_RS_XCH + w0 * 32 + 16 + ml]);
        m = fmaxf(m, LDSB[GDN_RS_XCH + (w0 + 1) * 32 + ml]);
        m = fmaxf(m, LDSB[GDN_RS_XCH + (w0 + 1) * 32 + 16 + ml]);
        const float inv_s = (m > 0.0f) ? (127.0f / m) : 0.0f;
        signed char* dst =
            s_q8 + lane_base + h_state + (unsigned)vrow * GDN_RS_HD;
        if (use_ef) {
            const float scale = (m > 0.0f) ? (m / 127.0f) : 1.0f;
            // Pass 2: re-promote per b (8 live f32, not 32).
#pragma unroll
            for (int b = 0; b < 4; b++) {
                rs_f8 c = rs_floatv(S[b]);
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int key = kh * 64 + b * 16 + kg * 8 + j;
                    const float x = c[j] + __half2float(efr_base[key]);
                    const float qf =
                        fminf(fmaxf(rintf(x * inv_s), -128.0f), 127.0f);
                    efr_base[key] = __float2half(x - qf * scale);
                    dst[key] = (signed char)qf;
                }
            }
            if (kh == 0 && kg == 0)
                s_scales[lane_base + h_off +
                         vrow] = scale;
        } else {
            unsigned int rng = (unsigned int)(h * 1664525u + vrow * 1013904223u +
                                              lane * 214013u + frame * 2531011u);
#pragma unroll
            for (int b = 0; b < 4; b++) {
                rs_f8 c = rs_floatv(S[b]);
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int key = kh * 64 + b * 16 + kg * 8 + j;
                    rng = rng * 1664525u + 1013904223u;
                    const float noise = (float)(rng >> 8) * (1.0f / 16777216.0f);
                    dst[key] = (signed char)fminf(
                        fmaxf(floorf(c[j] * inv_s + noise), -128.0f), 127.0f);
                }
            }
            if (kh == 0 && kg == 0) {
                const float scale = (m > 0.0f) ? (m / 127.0f) : 1.0f;
                s_scales[lane_base + h_off +
                         vrow] = scale;
            }
        }
    }
}
