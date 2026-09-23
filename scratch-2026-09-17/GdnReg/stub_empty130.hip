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
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
//STUB
}
