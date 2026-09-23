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
// Carry Kt panel (64 keys x 64 tokens f16 = 4096 halves = 2048 floats) over
// HOLE; everything else is dead during carry except VEC.
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

    __shared__ float LDSB[GDN_RS_LDS_FLOATS];

    // ---- entry: reconstruct this WG's value-half state, f16, no EF fold ----
    rs_h8 S[4];
    {
        const size_t lane_base = (size_t)blockIdx.z * n_heads * GDN_RS_HD * GDN_RS_HD;
        const signed char* src = s_q8 + lane_base + (size_t)h * GDN_RS_HD * GDN_RS_HD +
                                 (size_t)vrow * GDN_RS_HD + kh * 64;
        const float sc =
            s_scales[(size_t)blockIdx.z * n_heads * GDN_RS_HD + h * GDN_RS_HD + vrow];
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
        if (tid < 64) {
            const int t = c0 + tid;
            float gv = 0.0f, bv = 0.0f;
            if (t < n_tokens) {
                gv = gate[(size_t)t * n_heads + h];
                bv = beta[(size_t)t * n_heads + h];
            }
            LDSB[GDN_RS_VEC + tid] = gv;
            LDSB[GDN_RS_VEC + 128 + tid] = bv;
        }
        __syncthreads();
        if (tid == 0) {
            float run = 0.0f;
#pragma unroll
            for (int i = 0; i < 64; i++) {
                run += LDSB[GDN_RS_VEC + i];
                LDSB[GDN_RS_VEC + i] = run;
            }
        }
        __syncthreads();
        if (tid < 64) LDSB[GDN_RS_VEC + 64 + tid] = expf(LDSB[GDN_RS_VEC + tid]);
        if (tid == 0) {
            LDSB[GDN_RS_VEC + 192] = expf(LDSB[GDN_RS_VEC + (cn - 1)]);
            LDSB[GDN_RS_VEC + 193] = LDSB[GDN_RS_VEC + (cn - 1)];
        }
        __syncthreads();

        // ---- stage full K f16 panel (64x128) into HOLE ----
        {
            _Float16* f16k = reinterpret_cast<_Float16*>(LDSB + GDN_RS_HOLE);
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int x = tid * 8 + u * 2048;
                const int tc = x / 128;
                const int kc = x % 128;
                const int gt = c0 + tc;
                float4 w = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 w2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (gt < n_tokens) {
                    const float* kp = k + (size_t)gt * stride + h * GDN_RS_HD + kc;
                    w = *reinterpret_cast<const float4*>(kp);
                    w2 = *reinterpret_cast<const float4*>(kp + 4);
                }
                rs_f8 pack;
                pack[0] = w.x; pack[1] = w.y; pack[2] = w.z; pack[3] = w.w;
                pack[4] = w2.x; pack[5] = w2.y; pack[6] = w2.z; pack[7] = w2.w;
                *(reinterpret_cast<rs_h8*>(f16k + x)) = rs_halfv(pack);
            }
        }
        __syncthreads();

        // ---- KK Gram -> compact strict-lower L (packed row i: i(i-1)/2+j) ----
        {
            const _Float16* f16k = reinterpret_cast<const _Float16*>(LDSB + GDN_RS_HOLE);
            const int R0 = wave * 8;
#pragma unroll
            for (int p = 0; p < 4; p++) {
                rs_f8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int s = 0; s < 8; s++) {
                    rs_h8 A, B;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        A[j] = f16k[(R0 + (ml & 7)) * 128 + s * 16 + kg * 8 + j];
                        B[j] = f16k[(p * 16 + ml) * 128 + s * 16 + kg * 8 + j];
                    }
                    acc = rs_mma(A, B, acc);
                }
                if (ml < 8) {
                    const int row = R0 + ml;
                    const float bv = (row < cn) ? LDSB[GDN_RS_VEC + 128 + row] : 0.0f;
                    const float Gr = LDSB[GDN_RS_VEC + row];
                    const int poff = row * (row - 1) / 2;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const int col = p * 16 + kg * 8 + j;
                        float lv = 0.0f;
                        if (col < row && row < cn)
                            lv = bv * expf(Gr - LDSB[GDN_RS_VEC + col]) * acc[j];
                        LDSB[GDN_RS_CMP + poff + col] = lv;
                    }
                }
            }
        }
        __syncthreads();

        // ---- expand compact L -> EXP column-major (unit diagonal) ----
#pragma unroll
        for (int u = 0; u < 16; u++) {
            const int x = tid + u * 256;
            const int i = x / 64;
            const int j = x % 64;
            float e = 0.0f;
            if (j < i)
                e = LDSB[GDN_RS_CMP + i * (i - 1) / 2 + j];
            else if (j == i)
                e = 1.0f;
            LDSB[GDN_RS_EXP + j * 64 + i] = e;
        }
        __syncthreads();

        // ---- S.K (K16 panels restreamed into CMP space) + pair-sum + rhs ----
        {
            _Float16* panel = reinterpret_cast<_Float16*>(LDSB + GDN_RS_PANEL);
#pragma unroll
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
                    if (gt < n_tokens) {
                        const float* kp =
                            k + (size_t)gt * stride + h * GDN_RS_HD + kc;
                        w = *reinterpret_cast<const float4*>(kp);
                        w2 = *reinterpret_cast<const float4*>(kp + 4);
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
                rs_f8 sk;
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    sk[j] = acc[j] +
                            LDSB[GDN_RS_XCH + (((wave ^ 1) * 32 + lane)) * 8 + j];
                // rhs into D[tt]
                rs_f8 dd;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int tok = tt * 16 + kg * 8 + j;
                    const int gt = c0 + tok;
                    float vv = 0.0f;
                    if (gt < n_tokens)
                        vv = v[(size_t)gt * stride + h * GDN_RS_HD + vrow];
                    const float ee = LDSB[GDN_RS_VEC + 64 + tok];
                    const float bb = (tok < cn) ? LDSB[GDN_RS_VEC + 128 + tok] : 0.0f;
                    dd[j] = (tok < cn) ? (bb * (vv - ee * sk[j])) : 0.0f;
                }
                D[tt] = dd;
            }
        }
        __syncthreads();

        // ---- solve (I+L)*delta=rhs: 64 rank updates, wave-local ----
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB

        // ---- restage full K f16 (EXP dead) ----
        {
            _Float16* f16k = reinterpret_cast<_Float16*>(LDSB + GDN_RS_HOLE);
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int x = tid * 8 + u * 2048;
                const int tc = x / 128;
                const int kc = x % 128;
                const int gt = c0 + tc;
                float4 w = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 w2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (gt < n_tokens) {
                    const float* kp = k + (size_t)gt * stride + h * GDN_RS_HD + kc;
                    w = *reinterpret_cast<const float4*>(kp);
                    w2 = *reinterpret_cast<const float4*>(kp + 4);
                }
                rs_f8 pack;
                pack[0] = w.x; pack[1] = w.y; pack[2] = w.z; pack[3] = w.w;
                pack[4] = w2.x; pack[5] = w2.y; pack[6] = w2.z; pack[7] = w2.w;
                *(reinterpret_cast<rs_h8*>(f16k + x)) = rs_halfv(pack);
            }
        }
        __syncthreads();

        // ---- QK Gram -> compact lower-incl M (packed i(i+1)/2+j) ----
        // A rows (Q output rows R0+(ml&7)) stream from global Q; B rows (K
        // columns p*16+ml) come from the staged full-K f16 panel.
        {
            const _Float16* f16k = reinterpret_cast<const _Float16*>(LDSB + GDN_RS_HOLE);
            const int R0 = wave * 8;
#pragma unroll
            for (int p = 0; p < 4; p++) {
                rs_f8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int s = 0; s < 8; s++) {
                    rs_h8 A, B;
                    {
                        const int ar = R0 + (ml & 7);
                        const int agt = c0 + ar;
                        rs_f8 apack = {0, 0, 0, 0, 0, 0, 0, 0};
                        if (agt < n_tokens) {
                            const float* ap =
                                q + (size_t)agt * stride + h * GDN_RS_HD +
                                s * 16 + kg * 8;
                            #pragma unroll
                            for (int j = 0; j < 8; j++) apack[j] = ap[j];
                        }
                        A = rs_halfv(apack);
                    }
                    #pragma unroll
                    for (int j = 0; j < 8; j++)
                        B[j] = f16k[(p * 16 + ml) * 128 + s * 16 + kg * 8 + j];
                    acc = rs_mma(A, B, acc);
                }
                if (ml < 8) {
                    const int row = R0 + ml;
                    const float Gr = LDSB[GDN_RS_VEC + row];
                    const int poff = row * (row + 1) / 2;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const int col = p * 16 + kg * 8 + j;
                        float mv = 0.0f;
                        if (col <= row && row < cn)
                            mv = expf(Gr - LDSB[GDN_RS_VEC + col]) * acc[j];
                        LDSB[GDN_RS_CMP + poff + col] = mv;
                    }
                }
                __syncthreads();
            }
        }

        // ---- expand compact M -> EXP (zero upper) ----
#pragma unroll
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
#pragma unroll
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
                    if (gt < n_tokens) {
                        const float* qp =
                            q + (size_t)gt * stride + h * GDN_RS_HD + kc;
                        w = *reinterpret_cast<const float4*>(qp);
                        w2 = *reinterpret_cast<const float4*>(qp + 4);
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
                rs_f8 sq;
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    sq[j] = qacc[j] +
                            LDSB[GDN_RS_XCH + (((wave ^ 1) * 32 + lane)) * 8 + j];
                rs_f8 res = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int s = 0; s < 4; s++) {
                    rs_h8 A2, B2;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        A2[j] = (__half)LDSB[GDN_RS_EXP +
                                             (s * 16 + kg * 8 + j) * 64 + tt * 16 + ml];
                        B2[j] = (__half)(D[s][j]);
                    }
                    res = rs_mma(A2, B2, res);
                }
                if (kh == 0) {
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const int tok = tt * 16 + kg * 8 + j;
                        const int gt = c0 + tok;
                        if (gt < n_tokens) {
                            const float ee = LDSB[GDN_RS_VEC + 64 + tok];
                            output[(size_t)gt * stride + h * GDN_RS_HD + vrow] =
                                ee * sq[j] + res[j];
                        }
                    }
                }
                __syncthreads();
            }
        }

        // ---- carry: Kt panel (64 keys x 64 tokens f16, transposed), WMMA ----
        {
            _Float16* kt = reinterpret_cast<_Float16*>(LDSB + GDN_RS_KT);
#pragma unroll
            for (int u = 0; u < 16; u++) {
                const int x = tid + u * 256;
                const int r = x / 64;
                const int tc = x % 64;
                const int gt = c0 + tc;
                float kk = 0.0f;
                if (gt < n_tokens)
                    kk = k[(size_t)gt * stride + h * GDN_RS_HD + kh * 64 + r];
                kt[x] = (__half)kk;
            }
            __syncthreads();
            const float elast = LDSB[GDN_RS_VEC + 192];
            const float Ge = LDSB[GDN_RS_VEC + 193];
#pragma unroll
            for (int kb = 0; kb < 4; kb++) {
                rs_f8 acc = rs_floatv(S[kb]);
                #pragma unroll
                for (int j = 0; j < 8; j++) acc[j] *= elast;
#pragma unroll
                for (int s = 0; s < 4; s++) {
                    rs_h8 A, B;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const int stok = s * 16 + kg * 8 + j;
                        A[j] = kt[(kb * 16 + ml) * 64 + stok];
                        const float cf =
                            (stok < cn) ? expf(Ge - LDSB[GDN_RS_VEC + stok]) : 0.0f;
                        B[j] = (__half)(cf * D[s][j]);
                    }
                    acc = rs_mma(A, B, acc);
                }
                S[kb] = rs_halfv(acc);
            }
        }
        __syncthreads();
    }

    // ---- commit: promote S, fold old EF once, requant (shipping arithmetic)
    {
        const size_t lane_base = (size_t)blockIdx.z * n_heads * GDN_RS_HD * GDN_RS_HD;
        const bool use_ef = (s_ef_residual != nullptr);
        rs_f8 pre[4];
#pragma unroll
        for (int b = 0; b < 4; b++) pre[b] = rs_floatv(S[b]);
        __half* efr_base =
            use_ef ? (s_ef_residual + lane_base + (size_t)h * GDN_RS_HD * GDN_RS_HD +
                      (size_t)vrow * GDN_RS_HD)
                   : nullptr;
        if (use_ef) {
#pragma unroll
            for (int b = 0; b < 4; b++) {
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    pre[b][j] += __half2float(efr_base[kh * 64 + b * 16 + kg * 8 + j]);
            }
        }
        float my_max = 0.0f;
#pragma unroll
        for (int b = 0; b < 4; b++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const float a = fabsf(pre[b][j]);
                my_max = fmaxf(my_max, a);
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
            s_q8 + lane_base + (size_t)h * GDN_RS_HD * GDN_RS_HD + (size_t)vrow * GDN_RS_HD;
        if (use_ef) {
            const float scale = (m > 0.0f) ? (m / 127.0f) : 1.0f;
#pragma unroll
            for (int b = 0; b < 4; b++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int key = kh * 64 + b * 16 + kg * 8 + j;
                    const float qf =
                        fminf(fmaxf(rintf(pre[b][j] * inv_s), -128.0f), 127.0f);
                    efr_base[key] = __float2half(pre[b][j] - qf * scale);
                    dst[key] = (signed char)qf;
                }
            }
            if (kh == 0 && kg == 0)
                s_scales[(size_t)blockIdx.z * n_heads * GDN_RS_HD + h * GDN_RS_HD +
                         vrow] = scale;
        } else {
            unsigned int rng = (unsigned int)(h * 1664525u + vrow * 1013904223u +
                                              lane * 214013u + frame * 2531011u);
#pragma unroll
            for (int b = 0; b < 4; b++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int key = kh * 64 + b * 16 + kg * 8 + j;
                    rng = rng * 1664525u + 1013904223u;
                    const float noise = (float)(rng >> 8) * (1.0f / 16777216.0f);
                    dst[key] = (signed char)fminf(
                        fmaxf(floorf(pre[b][j] * inv_s + noise), -128.0f), 127.0f);
                }
            }
            if (kh == 0 && kg == 0) {
                const float scale = (m > 0.0f) ? (m / 127.0f) : 1.0f;
                s_scales[(size_t)blockIdx.z * n_heads * GDN_RS_HD + h * GDN_RS_HD +
                         vrow] = scale;
            }
        }
    }
}
