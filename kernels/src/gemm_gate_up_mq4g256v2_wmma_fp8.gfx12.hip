// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// FP8-WMMA MQ4-G256v2 gate/up prefill candidate for gfx1201 (RDNA4), B=12.
//
// Research kernel, default-off (host gate: HIPFIRE_GFX12_MQ4V2_FP8_GATEUP).
// Exact gfx1201, eager only. Companion to the F16 incumbent
// gemm_gate_up_mq4g256v2_wmma_gfx12_bt12 in
// kernels/src/gemm_gate_up_mq4g256v2_wmma_gfx12_bt.hip.
//
// Geometry (identical to the incumbent): one wave32 workgroup computes a
// 16-row x 192-batch tile (B=12 independent chains of 16 batch rows).
// Grid: [ceil((gate_m+up_m)/16), ceil(N/192), 1]. Block: [32].
// Lane decomposition: m_lane = tid & 15, kg = tid >> 4. gfx12 C-mapping:
// acc[b][j] holds C[8*kg+j][m_lane] for batch chain b. Overwrite
// semantics (y = row_scale * s_final * acc).
//
// Twelve-chain invariant: all 12 v8f WMMA chains stay live through every
// K16 tile. No schedule with fewer chains is admissible here.
//
// Per 256-K group the two 128-K halves are explicit. Group layout is the
// serialized MQ4v2 contract: 8 header bytes (two packed FP16 (scale,zero)
// pairs for K[0,128) and K[128,256)) followed by 128 payload bytes
// (256 nibbles). Half h covers payload bytes [h*64, h*64+64). Within a
// half, K is re-sliced into four 32-K pairs u: lane group kg owns
// K[u*32+kg*16, +16) as two consecutive 8-element fragments (8 payload
// bytes at +u*16+kg*8), so one 16-byte X load and one 8-byte weight load
// feed two WMMAs. Dot products are K-order invariant, so this differs from
// the F16 incumbent's K assignment only in f32 summation order.
//
// Accumulation: a single acc[12] kept in units of the current half's
// per-row scale; at each half boundary acc *= s_prev/s_new and the
// zero-point term (z_h/s_h)*sum_half(x) is added, then the WMMAs
// accumulate raw q.x directly. No LDS, no partial chains.
//
// Nibble q -> E4M3: every integer 0..15 is exact in E4M3
// (sign0|exp4|mant3, bias 7). Derivation: value = 2^(exp-7)*(1+mant/8)
// for normals, 0 for exp=mant=0:
//
//   q=0  -> 0x00 (zero)
//   q=1  -> 0x38: exp=7,mant=0  -> 2^0*1     = 1
//   q=2  -> 0x40: exp=8,mant=0  -> 2^1*1     = 2
//   q=3  -> 0x44: exp=8,mant=4  -> 2*1.5     = 3
//   q=4  -> 0x48: exp=9,mant=0  -> 4*1       = 4
//   q=5  -> 0x4A: exp=9,mant=2  -> 4*1.25    = 5
//   q=6  -> 0x4C: exp=9,mant=4  -> 4*1.5     = 6
//   q=7  -> 0x4E: exp=9,mant=6  -> 4*1.75    = 7
//   q=8  -> 0x50: exp=10,mant=0 -> 8*1       = 8
//   q=9  -> 0x51: exp=10,mant=1 -> 8*1.125   = 9
//   q=10 -> 0x52: exp=10,mant=2 -> 8*1.25    = 10
//   q=11 -> 0x53: exp=10,mant=3 -> 8*1.375   = 11
//   q=12 -> 0x54: exp=10,mant=4 -> 8*1.5     = 12
//   q=13 -> 0x55: exp=10,mant=5 -> 8*1.625   = 13
//   q=14 -> 0x56: exp=10,mant=6 -> 8*1.75    = 14
//   q=15 -> 0x57: exp=10,mant=7 -> 8*1.875   = 15
//
// The 16 bytes live as four 32-bit immediates (q 0..3, 4..7, 8..11,
// 12..15) looked up with v_perm_b32. No LDS/global lookup table.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef int   __attribute__((ext_vector_type(2))) v2i_t;
typedef int   __attribute__((ext_vector_type(4))) v4i_t;
typedef float __attribute__((ext_vector_type(8))) v8f_t;

// Expand one 32-bit nibble pack (8 q values, q_i = (pk >> 4*i) & 0xF)
// into one per-lane FP8 WMMA operand (8 E4M3 bytes) bit-cast to v2i_t.
// Byte i of the result is the E4M3 encoding of q_i per the table above.
// The 16 table bytes live as four 32-bit immediates selected by the top
// two nibble bits; the low two bits pick the byte offset. No LDS/global
// lookup table.
// Spread the 4 nibbles of a 16-bit value into 4 byte lanes (nibble i ->
// byte i), so v_perm_b32 can use them as byte selectors.
__device__ static inline unsigned int mq4_spread4(unsigned int x16) {
    const unsigned int a = (x16 & 0x00FFu) | ((x16 & 0xFF00u) << 8);
    return (a & 0x000F000Fu) | ((a & 0x00F000F0u) << 4);
}

// Table lookup of 8 nibbles: byte i of out[0] = E4M3(nibble i), byte i of
// out[1] = E4M3(nibble i+4). Same operand order as the scalar reference.
#ifdef HIPFIRE_FP8_LUT_ARG
// LUT-arg form (MQ4G256V2-Lloyd, qt=52): the 16 table bytes arrive as four
// kernel-arg dwords (centered codebook E4M3(L-7.5), 4 bytes packed LE per
// dword) instead of the uniform immediates below. Uniform builds are
// textually unchanged.
__device__ static inline unsigned int mq4_lut8(unsigned int sel,
    unsigned int C0, unsigned int C1, unsigned int C2, unsigned int C3) {
    // v_perm_b32(hi, lo, s): result byte i = {lo,hi} byte s_i for s_i in 0..7.
    const unsigned int lo = __builtin_amdgcn_perm(C1, C0, sel & 0x07070707u);
    const unsigned int hi = __builtin_amdgcn_perm(C3, C2, sel & 0x07070707u);
    const unsigned int m = ((sel >> 3) & 0x01010101u) * 0xFFu;
    return (hi & m) | (lo & ~m);
}
#else
__device__ static inline unsigned int mq4_lut8(unsigned int sel) {
    // q 0..3:   00 38 40 44. q 4..7:   48 4A 4C 4E.
    // q 8..11:  50 51 52 53. q 12..15: 54 55 56 57.
    const unsigned int C0 = 0x44403800u;
    const unsigned int C1 = 0x4E4C4A48u;
    const unsigned int C2 = 0x53525150u;
    const unsigned int C3 = 0x57565554u;
    // v_perm_b32(hi, lo, s): result byte i = {lo,hi} byte s_i for s_i in 0..7.
    const unsigned int lo = __builtin_amdgcn_perm(C1, C0, sel & 0x07070707u);
    const unsigned int hi = __builtin_amdgcn_perm(C3, C2, sel & 0x07070707u);
    const unsigned int m = ((sel >> 3) & 0x01010101u) * 0xFFu;
    return (hi & m) | (lo & ~m);
}
#endif

#ifdef HIPFIRE_FP8_LUT_ARG
__device__ static inline v2i_t mq4_qpk_to_e4m3(unsigned int pk,
    unsigned int C0, unsigned int C1, unsigned int C2, unsigned int C3) {
    v2i_t out;
    out[0] = (int)mq4_lut8(mq4_spread4(pk & 0xFFFFu), C0, C1, C2, C3);
    out[1] = (int)mq4_lut8(mq4_spread4(pk >> 16), C0, C1, C2, C3);
    return out;
}
#else
__device__ static inline v2i_t mq4_qpk_to_e4m3(unsigned int pk) {
    v2i_t out;
    out[0] = (int)mq4_lut8(mq4_spread4(pk & 0xFFFFu));
    out[1] = (int)mq4_lut8(mq4_spread4(pk >> 16));
    return out;
}
#endif
#ifdef HIPFIRE_FP8_LUT_ARG
// Candidate select among per-tensor decoded dwords by loop-invariant side.
// Every kernel-arg LUT dword is consumed directly at these call sites (SGPR
// sources into v_perm_b32); the per-lane decision collapses to VALU selects,
// so no divergent LUT copy ever occupies VGPRs across the K loop.
__device__ static inline unsigned int mq4_pick2(unsigned int a, unsigned int b, int s) {
    return s ? b : a;
}
__device__ static inline unsigned int mq4_pick3(unsigned int a, unsigned int b, unsigned int c, int s) {
    return (s == 0) ? a : ((s == 1) ? b : c);
}
#endif

// Batch-tile width (chains per wave). Host selects the symbol by N:
// N%192==0 -> BV=12, N%128==0 -> BV=8, N%64==0 -> BV=4.
#ifndef HIPFIRE_FP8_BV
#define HIPFIRE_FP8_BV 12
#endif
#ifndef HIPFIRE_FP8_GATEUP_KERNEL
#define HIPFIRE_FP8_GATEUP_KERNEL gemm_gate_up_mq4g256v2_wmma_fp8_gfx12_bt12
#endif

// HIPFIRE_FP8_RESIDUAL=1 compiles the single-matrix residual form
// (Y[N x M] += W[M x K] . X) with the incumbent residual ABI
// (A, X_fp8, X_half_sums, X_row_scales, Y, M, K, N); the default is the
// two-matrix gate/up overwrite form. The K loop is shared.
#ifndef HIPFIRE_FP8_RESIDUAL
#define HIPFIRE_FP8_RESIDUAL 0
#endif

// HIPFIRE_FP8_QKVZA=1 compiles the 4-way fused DeltaNet projection form
// (A_qkv/A_z/A_beta/A_alpha -> Y_qkv/Y_z/Y_beta/Y_alpha, overwrite) with the
// incumbent qkvza ABI.
#ifndef HIPFIRE_FP8_QKVZA
#define HIPFIRE_FP8_QKVZA 0
#endif

// HIPFIRE_FP8_QKV=1 compiles the 3-way fused full-attention projection form
// (A_q/A_k/A_v -> Y_q/Y_k/Y_v, overwrite) with the incumbent QKV ABI
// (3 weights, 3 outputs, 3 row counts q_m/k_m/v_m); routing mirrors the
// F16 gfx1201 BT kernel (kernels/src/gemm_qkv_mq4g256v2_wmma_gfx1201_bt.hip).
#ifndef HIPFIRE_FP8_QKV
#define HIPFIRE_FP8_QKV 0
#endif

// Two-slab form: HIPFIRE_FP8_SLABS=2 gives each wave32 two 16-row slabs
// (rows [rs, rs+32)) so one 16 B X fragment feeds four WMMAs (two slabs x
// two K fragments) instead of two. acc[s][b], per-slab headers/scales and
// per-slab boundary correction; batch columns (sb, X, half-sums) shared.
// Grid becomes [ceil(M/32), ceil(N/(16*BV)), 1]. SLABS=1 is the current
// single-slab kernel, kept textually identical in the #else branches.
#ifndef HIPFIRE_FP8_SLABS
#define HIPFIRE_FP8_SLABS 1
#endif

extern "C" __launch_bounds__(32, 2) __global__
#if HIPFIRE_FP8_QKVZA
void HIPFIRE_FP8_GATEUP_KERNEL(
    const char* A_qkv,
    const char* A_z,
    const char* A_beta,
    const char* A_alpha,
    const unsigned char* X_fp8,
    const float* X_half_sums,
    const float* X_row_scales,
    float* Y_qkv,
    float* Y_z,
    float* Y_beta,
    float* Y_alpha,
    int qkv_m,
    int z_m,
    int beta_m,
    int alpha_m,
    int K,
    int N
#ifdef HIPFIRE_FP8_LUT_ARG
    , unsigned int LQ0, unsigned int LQ1, unsigned int LQ2, unsigned int LQ3,
    unsigned int LZ0, unsigned int LZ1, unsigned int LZ2, unsigned int LZ3,
    unsigned int LB0, unsigned int LB1, unsigned int LB2, unsigned int LB3,
    unsigned int LA0, unsigned int LA1, unsigned int LA2, unsigned int LA3
#endif
) {
    const int tm = qkv_m + z_m + beta_m + alpha_m;
#elif HIPFIRE_FP8_RESIDUAL
void HIPFIRE_FP8_GATEUP_KERNEL(
    const char* A_w,
    const unsigned char* X_fp8,
    const float* X_half_sums,
    const float* X_row_scales,
    float* Y_res,
    int res_m,
    int K,
    int N
#ifdef HIPFIRE_FP8_LUT_ARG
    , unsigned int LC0, unsigned int LC1, unsigned int LC2, unsigned int LC3
#endif
) {
    const int tm = res_m;
#elif HIPFIRE_FP8_QKV
void HIPFIRE_FP8_GATEUP_KERNEL(
    const char* A_q,
    const char* A_k,
    const char* A_v,
    const unsigned char* X_fp8,
    const float* X_half_sums,
    const float* X_row_scales,
    float* Y_q,
    float* Y_k,
    float* Y_v,
    int q_m,
    int k_m,
    int v_m,
    int K,
    int N
#ifdef HIPFIRE_FP8_LUT_ARG
    , unsigned int LQ0, unsigned int LQ1, unsigned int LQ2, unsigned int LQ3,
    unsigned int LK0, unsigned int LK1, unsigned int LK2, unsigned int LK3,
    unsigned int LV0, unsigned int LV1, unsigned int LV2, unsigned int LV3
#endif
) {
    const int tm = q_m + k_m + v_m;
#else
void HIPFIRE_FP8_GATEUP_KERNEL(
    const char* A_gate,
    const char* A_up,
    const unsigned char* X_fp8,
    const float* X_half_sums,
    const float* X_row_scales,
    float* Y_gate,
    float* Y_up,
    int gate_m,
    int up_m,
    int K,
    int N
#ifdef HIPFIRE_FP8_LUT_ARG
    , unsigned int LG0, unsigned int LG1, unsigned int LG2, unsigned int LG3,
    unsigned int LU0, unsigned int LU1, unsigned int LU2, unsigned int LU3
#endif
) {
    const int tm = gate_m + up_m;
#endif
    // SLABS=2 waves own two 16-row slabs; SLABS=1 keeps the current grid.
#if HIPFIRE_FP8_SLABS == 2
    const int rs = blockIdx.x * 32;
#else
    const int rs = blockIdx.x * 16;
#endif
    const int bs = blockIdx.y * (16 * HIPFIRE_FP8_BV);
    const int tid = threadIdx.x;
    if (rs >= tm || bs >= N) return;

    const int ml = tid & 15;
    const int kg = tid >> 4;

#if HIPFIRE_FP8_SLABS == 2
    // Two-slab routing: one A-row per slab (own header source each).
    const int sr0 = ((rs + ml) < tm) ? (rs + ml) : (tm - 1);
    const int sr1 = ((rs + 16 + ml) < tm) ? (rs + 16 + ml) : (tm - 1);
    const int srx[2] = { sr0, sr1 };
    const char* Ax[2];
    int lrx[2];
    #pragma unroll
    for (int s = 0; s < 2; s++) {
        const int srs = srx[s];
#if HIPFIRE_FP8_QKVZA
        if (srs < qkv_m) { Ax[s] = A_qkv; lrx[s] = srs; }
        else if (srs < qkv_m + z_m) { Ax[s] = A_z; lrx[s] = srs - qkv_m; }
        else if (srs < qkv_m + z_m + beta_m) { Ax[s] = A_beta; lrx[s] = srs - (qkv_m + z_m); }
        else { Ax[s] = A_alpha; lrx[s] = srs - (qkv_m + z_m + beta_m); }
#elif HIPFIRE_FP8_RESIDUAL
        Ax[s] = A_w;
        lrx[s] = srs;
#elif HIPFIRE_FP8_QKV
        if (srs < q_m) { Ax[s] = A_q; lrx[s] = srs; }
        else if (srs < q_m + k_m) { Ax[s] = A_k; lrx[s] = srs - q_m; }
        else { Ax[s] = A_v; lrx[s] = srs - (q_m + k_m); }
#else
        if (srs < gate_m) { Ax[s] = A_gate; lrx[s] = srs; } else { Ax[s] = A_up; lrx[s] = srs - gate_m; }
#endif
    }
#ifdef HIPFIRE_FP8_LUT_ARG
    // Per-slab source-tensor index, loop-invariant (same row routing as
    // Ax/lrx above). The LUT dwords stay SGPR (kernel args); the K-loop
    // decodes every candidate and selects per slab, so no divergent LUT
    // copy ever occupies VGPRs. Residual keeps the single-LUT SLC arrays
    // (uniform copies, already SGPR-clean).
#if HIPFIRE_FP8_RESIDUAL
    unsigned int SLC0[2], SLC1[2], SLC2[2], SLC3[2];
    #pragma unroll
    for (int s = 0; s < 2; s++) {
        SLC0[s]=LC0; SLC1[s]=LC1; SLC2[s]=LC2; SLC3[s]=LC3;
    }
#elif HIPFIRE_FP8_QKVZA
    // Four-tensor fan-out: candidate evaluation costs more VGPRs than a
    // copy, so QKVZA selects one LUT per slab. The select keys on the SLAB
    // START ROW (wave-uniform, from blockIdx), not the lane row, so the four
    // dwords resolve to SGPRs and never occupy VGPRs across the K loop. Host
    // contract: qkv_m, z_m, beta_m, alpha_m are all multiples of 16, so no
    // 16-row slab straddles two sources (the launcher refuses otherwise).
    unsigned int SLC0[2], SLC1[2], SLC2[2], SLC3[2];
    #pragma unroll
    for (int s = 0; s < 2; s++) {
        const int srs = rs + 16 * s;
        if (srs < qkv_m) { SLC0[s]=LQ0; SLC1[s]=LQ1; SLC2[s]=LQ2; SLC3[s]=LQ3; }
        else if (srs < qkv_m + z_m) { SLC0[s]=LZ0; SLC1[s]=LZ1; SLC2[s]=LZ2; SLC3[s]=LZ3; }
        else if (srs < qkv_m + z_m + beta_m) { SLC0[s]=LB0; SLC1[s]=LB1; SLC2[s]=LB2; SLC3[s]=LB3; }
        else { SLC0[s]=LA0; SLC1[s]=LA1; SLC2[s]=LA2; SLC3[s]=LA3; }
    }
#else
    int lut_side[2];
    #pragma unroll
    for (int s = 0; s < 2; s++) {
        const int srs = srx[s];
#if HIPFIRE_FP8_QKV
        lut_side[s] = (srs < q_m) ? 0 : (srs < q_m + k_m) ? 1 : 2;
#else
        lut_side[s] = (srs < gate_m) ? 0 : 1;
#endif
    }
#endif
#endif
#else
    // Route this lane's A-row (own header source).
    const int mr = rs + ml;
    const int sr = (mr < tm) ? mr : (tm - 1);
#if HIPFIRE_FP8_QKVZA
    const char* A;
    int lr;
    if (sr < qkv_m) { A = A_qkv; lr = sr; }
    else if (sr < qkv_m + z_m) { A = A_z; lr = sr - qkv_m; }
    else if (sr < qkv_m + z_m + beta_m) { A = A_beta; lr = sr - (qkv_m + z_m); }
    else { A = A_alpha; lr = sr - (qkv_m + z_m + beta_m); }
#elif HIPFIRE_FP8_RESIDUAL
    const char* A = A_w;
    const int lr = sr;
#elif HIPFIRE_FP8_QKV
    const char* A;
    int lr;
    if (sr < q_m) { A = A_q; lr = sr; }
    else if (sr < q_m + k_m) { A = A_k; lr = sr - q_m; }
    else { A = A_v; lr = sr - (q_m + k_m); }
#else
    const char* A;
    int lr;
    if (sr < gate_m) { A = A_gate; lr = sr; } else { A = A_up; lr = sr - gate_m; }
#endif
#ifdef HIPFIRE_FP8_LUT_ARG
    // Lane-row source-tensor index, loop-invariant (row routing above). The
    // LUT dwords stay SGPR (kernel args); decode evaluates every candidate
    // and selects, so no divergent LUT copy ever occupies VGPRs.
    // Residual uses the single LC* args directly at the decode sites.
#if HIPFIRE_FP8_RESIDUAL
    // Single LC* args used directly at the decode sites (no copy).
#elif HIPFIRE_FP8_QKVZA
    // Four-tensor fan-out: one LUT per 16-row tile, keyed on the TILE START
    // ROW (wave-uniform) so the dwords stay SGPR. Host contract: all four m
    // are multiples of 16 (no tile straddles two sources); see SLABS=2 twin.
    unsigned int LC0, LC1, LC2, LC3;
    if (rs < qkv_m) { LC0=LQ0; LC1=LQ1; LC2=LQ2; LC3=LQ3; }
    else if (rs < qkv_m + z_m) { LC0=LZ0; LC1=LZ1; LC2=LZ2; LC3=LZ3; }
    else if (rs < qkv_m + z_m + beta_m) { LC0=LB0; LC1=LB1; LC2=LB2; LC3=LB3; }
    else { LC0=LA0; LC1=LA1; LC2=LA2; LC3=LA3; }
#else
#if HIPFIRE_FP8_QKV
    const int lut_side = (sr < q_m) ? 0 : (sr < q_m + k_m) ? 1 : 2;
#else
    const int lut_side = (sr < gate_m) ? 0 : 1;
#endif
#endif
#endif
#endif

    const int gpr = K / 256;
#if HIPFIRE_FP8_SLABS == 2
    // Per-slab weight-row bases; 32-bit offsets per the note below.
    const char* rbx[2] = { Ax[0] + (unsigned int)lrx[0] * (unsigned int)gpr * 136u,
                           Ax[1] + (unsigned int)lrx[1] * (unsigned int)gpr * 136u };
#else
    // All byte offsets below fit in 32 bits: X holds N*K bytes with
    // N*K < 2^32, each weight row holds gpr*136 bytes, and Y indices stay
    // far below 2^32 at admitted shapes. This keeps hoisted per-chain
    // products to one VGPR each instead of 64-bit pairs.
    const char* rb = A + (unsigned int)lr * (unsigned int)gpr * 136u;
#endif

    // Batch rows for the 12 chains; out-of-range rows alias row 0 for
    // safe loads, stores stay guarded.
    int sb[HIPFIRE_FP8_BV];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_FP8_BV; b++) {
        const int t = bs + b * 16 + ml;
        sb[b] = (t < N) ? t : 0;
    }

#if HIPFIRE_FP8_SLABS == 2
    // Two long-lived accumulators per chain (one per slab), each in units
    // of its slab's current half scale; per-slab current scale below.
    v8f_t acc[2][HIPFIRE_FP8_BV];
    #pragma unroll
    for (int s = 0; s < 2; s++)
        #pragma unroll
        for (int b = 0; b < HIPFIRE_FP8_BV; b++)
            acc[s][b] = v8f_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // Lane-owned (row ml) current scale per slab; 0 until first nonzero half.
    float s_cur_x[2] = { 0.0f, 0.0f };
#else
    // Single long-lived accumulator per chain, kept in units of the
    // CURRENT half's per-row scale s_cur (row = 8*kg+j for element j).
    // At each half boundary: acc *= s_prev/s_new, then the zero-point term
    // is added as (z_h/s_h)*S_b, and the WMMAs accumulate raw q.x directly.
    // This removes the 12 partial chains and the LDS accumulator; the cost
    // is one f32 multiply per element per half (~6e-8 relative), far below
    // the E4M3 activation rounding this path already accepts.
    v8f_t acc[HIPFIRE_FP8_BV];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_FP8_BV; b++)
        acc[b] = v8f_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // Lane-owned (row ml) current scale; 0 until the first nonzero half.
    float s_cur_self = 0.0f;
#endif
    const unsigned char* xb_base[HIPFIRE_FP8_BV];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_FP8_BV; b++) xb_base[b] = X_fp8 + (unsigned int)sb[b] * (unsigned int)K;

#if HIPFIRE_FP8_SLABS == 2
    // Two-slab K loop: per-slab headers from rbx[s], shared half-sums S0/S1
    // and X base. Per (group, half): per-slab boundary correction into
    // acc[s][b], then per 32-K pair decode one weight fragment per slab
    // (same kt) and issue 4 WMMAs per chain off ONE X fragment pair.
    #pragma nounroll
    for (int g = 0; g < gpr; g++) {
        float sc0x[2], zp0x[2], sc1x[2], zp1x[2];
        const char* payx[2];
        #pragma unroll
        for (int s = 0; s < 2; s++) {
            const char* gpx = rbx[s] + (unsigned int)g * 136u;
            const unsigned int hAx = *(const unsigned int*)(gpx);
            const unsigned int hBx = *(const unsigned int*)(gpx + 4);
            sc0x[s] = __half2float(__ushort_as_half((unsigned short)(hAx & 0xFFFFu)));
            zp0x[s] = __half2float(__ushort_as_half((unsigned short)(hAx >> 16)));
            sc1x[s] = __half2float(__ushort_as_half((unsigned short)(hBx & 0xFFFFu)));
            zp1x[s] = __half2float(__ushort_as_half((unsigned short)(hBx >> 16)));
            payx[s] = gpx + 8;
        }

        // Half-sums for both halves of this group (shared by both slabs).
        float S0[HIPFIRE_FP8_BV], S1[HIPFIRE_FP8_BV];
        #pragma unroll
        for (int b = 0; b < HIPFIRE_FP8_BV; b++) {
            const float2 s2 = *(const float2*)(X_half_sums + ((unsigned int)sb[b] * (unsigned int)gpr + (unsigned int)g) * 2u);
            S0[b] = s2.x; S1[b] = s2.y;
        }

        #pragma nounroll
        for (int h = 0; h < 2; h++) {
            // Per-slab boundary state; a zero scale zeroes that slab's
            // weight operand below while keeping its previous units.
            float rx[2], zqx[2];
            bool livex[2];
            #pragma unroll
            for (int s = 0; s < 2; s++) {
                const float s_new = (h == 0) ? sc0x[s] : sc1x[s];
                const float zx = (h == 0) ? zp0x[s] : zp1x[s];
                const bool live = (s_new != 0.0f);
                const float s_next = live ? s_new : s_cur_x[s];
                rx[s] = (live && s_cur_x[s] != 0.0f) ? (s_cur_x[s] / s_next) : 1.0f;
                zqx[s] = (s_next != 0.0f) ? (zx / s_next) : 0.0f;
                s_cur_x[s] = s_next;
                livex[s] = live;
            }

            // Rescale into the new half's units and add the zero-point term
            // in those units, per slab.
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const float r0 = __shfl(rx[0], 8 * kg + j);
                const float r1 = __shfl(rx[1], 8 * kg + j);
                const float z0 = __shfl(zqx[0], 8 * kg + j);
                const float z1 = __shfl(zqx[1], 8 * kg + j);
                #pragma unroll
                for (int b = 0; b < HIPFIRE_FP8_BV; b++) {
                    const float Sb = (h == 0) ? S0[b] : S1[b];
                    acc[0][b][j] = fmaf(z0, Sb, acc[0][b][j] * r0);
                    acc[1][b][j] = fmaf(z1, Sb, acc[1][b][j] * r1);
                }
            }

            // Same K re-slicing as SLABS=1, decoded once per slab; one 16 B
            // X load per chain feeds four WMMAs (two slabs x two fragments).
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                const uint2 pk0 = *(const uint2*)(payx[0] + h * 64 + u * 16 + kg * 8);
                const uint2 pk1 = *(const uint2*)(payx[1] + h * 64 + u * 16 + kg * 8);
#ifdef HIPFIRE_FP8_LUT_ARG
#if HIPFIRE_FP8_RESIDUAL || HIPFIRE_FP8_QKVZA
                v2i_t q0a = mq4_qpk_to_e4m3(pk0.x, SLC0[0], SLC1[0], SLC2[0], SLC3[0]);
                v2i_t q0b = mq4_qpk_to_e4m3(pk0.y, SLC0[0], SLC1[0], SLC2[0], SLC3[0]);
                v2i_t q1a = mq4_qpk_to_e4m3(pk1.x, SLC0[1], SLC1[1], SLC2[1], SLC3[1]);
                v2i_t q1b = mq4_qpk_to_e4m3(pk1.y, SLC0[1], SLC1[1], SLC2[1], SLC3[1]);
#else
                // Selectors depend only on the nibble packs, so both slabs
                // share them; each slab picks its side's candidate below.
                const unsigned int s0a = mq4_spread4(pk0.x & 0xFFFFu);
                const unsigned int s0aH = mq4_spread4(pk0.x >> 16);
                const unsigned int s0b = mq4_spread4(pk0.y & 0xFFFFu);
                const unsigned int s0bH = mq4_spread4(pk0.y >> 16);
                const unsigned int s1a = mq4_spread4(pk1.x & 0xFFFFu);
                const unsigned int s1aH = mq4_spread4(pk1.x >> 16);
                const unsigned int s1b = mq4_spread4(pk1.y & 0xFFFFu);
                const unsigned int s1bH = mq4_spread4(pk1.y >> 16);
                v2i_t q0a, q0b, q1a, q1b;
#if HIPFIRE_FP8_QKV
                q0a[0] = mq4_pick3(mq4_lut8(s0a, LQ0, LQ1, LQ2, LQ3), mq4_lut8(s0a, LK0, LK1, LK2, LK3), mq4_lut8(s0a, LV0, LV1, LV2, LV3), lut_side[0]);
                q0a[1] = mq4_pick3(mq4_lut8(s0aH, LQ0, LQ1, LQ2, LQ3), mq4_lut8(s0aH, LK0, LK1, LK2, LK3), mq4_lut8(s0aH, LV0, LV1, LV2, LV3), lut_side[0]);
                q0b[0] = mq4_pick3(mq4_lut8(s0b, LQ0, LQ1, LQ2, LQ3), mq4_lut8(s0b, LK0, LK1, LK2, LK3), mq4_lut8(s0b, LV0, LV1, LV2, LV3), lut_side[0]);
                q0b[1] = mq4_pick3(mq4_lut8(s0bH, LQ0, LQ1, LQ2, LQ3), mq4_lut8(s0bH, LK0, LK1, LK2, LK3), mq4_lut8(s0bH, LV0, LV1, LV2, LV3), lut_side[0]);
                q1a[0] = mq4_pick3(mq4_lut8(s1a, LQ0, LQ1, LQ2, LQ3), mq4_lut8(s1a, LK0, LK1, LK2, LK3), mq4_lut8(s1a, LV0, LV1, LV2, LV3), lut_side[1]);
                q1a[1] = mq4_pick3(mq4_lut8(s1aH, LQ0, LQ1, LQ2, LQ3), mq4_lut8(s1aH, LK0, LK1, LK2, LK3), mq4_lut8(s1aH, LV0, LV1, LV2, LV3), lut_side[1]);
                q1b[0] = mq4_pick3(mq4_lut8(s1b, LQ0, LQ1, LQ2, LQ3), mq4_lut8(s1b, LK0, LK1, LK2, LK3), mq4_lut8(s1b, LV0, LV1, LV2, LV3), lut_side[1]);
                q1b[1] = mq4_pick3(mq4_lut8(s1bH, LQ0, LQ1, LQ2, LQ3), mq4_lut8(s1bH, LK0, LK1, LK2, LK3), mq4_lut8(s1bH, LV0, LV1, LV2, LV3), lut_side[1]);
#else
                q0a[0] = mq4_pick2(mq4_lut8(s0a, LG0, LG1, LG2, LG3), mq4_lut8(s0a, LU0, LU1, LU2, LU3), lut_side[0]);
                q0a[1] = mq4_pick2(mq4_lut8(s0aH, LG0, LG1, LG2, LG3), mq4_lut8(s0aH, LU0, LU1, LU2, LU3), lut_side[0]);
                q0b[0] = mq4_pick2(mq4_lut8(s0b, LG0, LG1, LG2, LG3), mq4_lut8(s0b, LU0, LU1, LU2, LU3), lut_side[0]);
                q0b[1] = mq4_pick2(mq4_lut8(s0bH, LG0, LG1, LG2, LG3), mq4_lut8(s0bH, LU0, LU1, LU2, LU3), lut_side[0]);
                q1a[0] = mq4_pick2(mq4_lut8(s1a, LG0, LG1, LG2, LG3), mq4_lut8(s1a, LU0, LU1, LU2, LU3), lut_side[1]);
                q1a[1] = mq4_pick2(mq4_lut8(s1aH, LG0, LG1, LG2, LG3), mq4_lut8(s1aH, LU0, LU1, LU2, LU3), lut_side[1]);
                q1b[0] = mq4_pick2(mq4_lut8(s1b, LG0, LG1, LG2, LG3), mq4_lut8(s1b, LU0, LU1, LU2, LU3), lut_side[1]);
                q1b[1] = mq4_pick2(mq4_lut8(s1bH, LG0, LG1, LG2, LG3), mq4_lut8(s1bH, LU0, LU1, LU2, LU3), lut_side[1]);
#endif
#endif
#else
                v2i_t q0a = mq4_qpk_to_e4m3(pk0.x);
                v2i_t q0b = mq4_qpk_to_e4m3(pk0.y);
                v2i_t q1a = mq4_qpk_to_e4m3(pk1.x);
                v2i_t q1b = mq4_qpk_to_e4m3(pk1.y);
#endif
                if (!livex[0]) { q0a[0] = 0; q0a[1] = 0; q0b[0] = 0; q0b[1] = 0; }
                if (!livex[1]) { q1a[0] = 0; q1a[1] = 0; q1b[0] = 0; q1b[1] = 0; }
                const unsigned int x_off = (unsigned int)g * 256u + (unsigned int)h * 128u + (unsigned int)u * 32u + (unsigned int)kg * 16u;
                #pragma unroll
                for (int b = 0; b < HIPFIRE_FP8_BV; b++) {
                    const v2i_t xa = *(const v2i_t*)(xb_base[b] + x_off);
                    const v2i_t xb = *(const v2i_t*)(xb_base[b] + x_off + 8u);
                    acc[0][b] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(q0a, xa, acc[0][b]);
                    acc[0][b] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(q0b, xb, acc[0][b]);
                    acc[1][b] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(q1a, xa, acc[1][b]);
                    acc[1][b] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(q1b, xb, acc[1][b]);
                }
            }
        }
    }
#else
    #pragma nounroll
    for (int g = 0; g < gpr; g++) {
        const char* gp = rb + (unsigned int)g * 136u;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const float sc0 = __half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu)));
        const float zp0 = __half2float(__ushort_as_half((unsigned short)(hA >> 16)));
        const float sc1 = __half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu)));
        const float zp1 = __half2float(__ushort_as_half((unsigned short)(hB >> 16)));
        const char* payload = gp + 8;

        // Half-sums for both halves of this group: 12 x 8-byte loads.
        float S0[HIPFIRE_FP8_BV], S1[HIPFIRE_FP8_BV];
        #pragma unroll
        for (int b = 0; b < HIPFIRE_FP8_BV; b++) {
            const float2 s2 = *(const float2*)(X_half_sums + ((unsigned int)sb[b] * (unsigned int)gpr + (unsigned int)g) * 2u);
            S0[b] = s2.x; S1[b] = s2.y;
        }

        #pragma nounroll
        for (int h = 0; h < 2; h++) {
            const float s_new_self = (h == 0) ? sc0 : sc1;
            const float z_self = (h == 0) ? zp0 : zp1;
            // A zero scale means this half contributes exactly 0: keep the
            // previous units and zero the weight operand below.
            const bool live_self = (s_new_self != 0.0f);
            const float s_next_self = live_self ? s_new_self : s_cur_self;
            const float r_self = (live_self && s_cur_self != 0.0f) ? (s_cur_self / s_next_self) : 1.0f;
            const float zq_self = (s_next_self != 0.0f) ? (z_self / s_next_self) : 0.0f;
            s_cur_self = s_next_self;

            // Rescale into the new half's units and add the zero-point term
            // in those units: 16 shuffles + 96 fma.
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const float r_j = __shfl(r_self, 8 * kg + j);
                const float zq_j = __shfl(zq_self, 8 * kg + j);
                #pragma unroll
                for (int b = 0; b < HIPFIRE_FP8_BV; b++)
                    acc[b][j] = fmaf(zq_j, (h == 0) ? S0[b] : S1[b], acc[b][j] * r_j);
            }

            // K re-slicing (dot products are K-order invariant): within each
            // 32-K pair u, lane kg owns K elements [u*32+kg*16, +16) as two
            // consecutive 8-element fragments, so one 16-byte X load and one
            // 8-byte weight load feed two WMMAs.
            const char* half_base = payload + h * 64;
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                const uint2 pk2 = *(const uint2*)(half_base + u * 16 + kg * 8);
#ifdef HIPFIRE_FP8_LUT_ARG
#if HIPFIRE_FP8_RESIDUAL || HIPFIRE_FP8_QKVZA
                v2i_t q8a = mq4_qpk_to_e4m3(pk2.x, LC0, LC1, LC2, LC3);
                v2i_t q8b = mq4_qpk_to_e4m3(pk2.y, LC0, LC1, LC2, LC3);
#else
                // Selectors depend only on the nibble packs (shared); the lane
                // picks its side's candidate (LUT dwords stay SGPR).
                const unsigned int sxa = mq4_spread4(pk2.x & 0xFFFFu);
                const unsigned int sxaH = mq4_spread4(pk2.x >> 16);
                const unsigned int sxb = mq4_spread4(pk2.y & 0xFFFFu);
                const unsigned int sxbH = mq4_spread4(pk2.y >> 16);
                v2i_t q8a, q8b;
#if HIPFIRE_FP8_QKV
                q8a[0] = mq4_pick3(mq4_lut8(sxa, LQ0, LQ1, LQ2, LQ3), mq4_lut8(sxa, LK0, LK1, LK2, LK3), mq4_lut8(sxa, LV0, LV1, LV2, LV3), lut_side);
                q8a[1] = mq4_pick3(mq4_lut8(sxaH, LQ0, LQ1, LQ2, LQ3), mq4_lut8(sxaH, LK0, LK1, LK2, LK3), mq4_lut8(sxaH, LV0, LV1, LV2, LV3), lut_side);
                q8b[0] = mq4_pick3(mq4_lut8(sxb, LQ0, LQ1, LQ2, LQ3), mq4_lut8(sxb, LK0, LK1, LK2, LK3), mq4_lut8(sxb, LV0, LV1, LV2, LV3), lut_side);
                q8b[1] = mq4_pick3(mq4_lut8(sxbH, LQ0, LQ1, LQ2, LQ3), mq4_lut8(sxbH, LK0, LK1, LK2, LK3), mq4_lut8(sxbH, LV0, LV1, LV2, LV3), lut_side);
#else
                q8a[0] = mq4_pick2(mq4_lut8(sxa, LG0, LG1, LG2, LG3), mq4_lut8(sxa, LU0, LU1, LU2, LU3), lut_side);
                q8a[1] = mq4_pick2(mq4_lut8(sxaH, LG0, LG1, LG2, LG3), mq4_lut8(sxaH, LU0, LU1, LU2, LU3), lut_side);
                q8b[0] = mq4_pick2(mq4_lut8(sxb, LG0, LG1, LG2, LG3), mq4_lut8(sxb, LU0, LU1, LU2, LU3), lut_side);
                q8b[1] = mq4_pick2(mq4_lut8(sxbH, LG0, LG1, LG2, LG3), mq4_lut8(sxbH, LU0, LU1, LU2, LU3), lut_side);
#endif
#endif
#else
                v2i_t q8a = mq4_qpk_to_e4m3(pk2.x);
                v2i_t q8b = mq4_qpk_to_e4m3(pk2.y);
#endif
                if (!live_self) { q8a[0] = 0; q8a[1] = 0; q8b[0] = 0; q8b[1] = 0; }
                const unsigned int x_off = (unsigned int)g * 256u + (unsigned int)h * 128u + (unsigned int)u * 32u + (unsigned int)kg * 16u;
                #pragma unroll
                for (int b = 0; b < HIPFIRE_FP8_BV; b++) {
                    const v2i_t xa = *(const v2i_t*)(xb_base[b] + x_off);
                    const v2i_t xb = *(const v2i_t*)(xb_base[b] + x_off + 8u);
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(q8a, xa, acc[b]);
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(q8b, xb, acc[b]);
                }
            }
        }
    }
#endif

#if HIPFIRE_FP8_SLABS == 2
    // Per-slab final scales, gathered from the lane owning row 8*kg+j of
    // each slab.
    float s_finx[2][8];
    #pragma unroll
    for (int s = 0; s < 2; s++)
        #pragma unroll
        for (int j = 0; j < 8; j++) s_finx[s][j] = __shfl(s_cur_x[s], 8 * kg + j);

    // Final store per slab with the incumbent output mapping.
    #pragma unroll
    for (int b = 0; b < HIPFIRE_FP8_BV; b++) {
        const int oc = bs + b * 16 + ml;
        if (oc < N) {
            const float an = X_row_scales[oc];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int orow0 = rs + 8 * kg + j;
                if (orow0 < tm) {
#if HIPFIRE_FP8_QKVZA
                    float* Y0;
                    int pr0, os0;
                    if (orow0 < qkv_m) { Y0 = Y_qkv; pr0 = orow0; os0 = qkv_m; }
                    else if (orow0 < qkv_m + z_m) { Y0 = Y_z; pr0 = orow0 - qkv_m; os0 = z_m; }
                    else if (orow0 < qkv_m + z_m + beta_m) { Y0 = Y_beta; pr0 = orow0 - (qkv_m + z_m); os0 = beta_m; }
                    else { Y0 = Y_alpha; pr0 = orow0 - (qkv_m + z_m + beta_m); os0 = alpha_m; }
                    Y0[(unsigned int)oc * (unsigned int)os0 + (unsigned int)pr0] = an * s_finx[0][j] * acc[0][b][j];
#elif HIPFIRE_FP8_RESIDUAL
                    Y_res[(unsigned int)oc * (unsigned int)res_m + (unsigned int)orow0] += an * s_finx[0][j] * acc[0][b][j];
#elif HIPFIRE_FP8_QKV
                    float* Y0;
                    int pr0, os0;
                    if (orow0 < q_m) { Y0 = Y_q; pr0 = orow0; os0 = q_m; }
                    else if (orow0 < q_m + k_m) { Y0 = Y_k; pr0 = orow0 - q_m; os0 = k_m; }
                    else { Y0 = Y_v; pr0 = orow0 - (q_m + k_m); os0 = v_m; }
                    Y0[(unsigned int)oc * (unsigned int)os0 + (unsigned int)pr0] = an * s_finx[0][j] * acc[0][b][j];
#else
                    float* Y0;
                    int pr0, os0;
                    if (orow0 < gate_m) { Y0 = Y_gate; pr0 = orow0; os0 = gate_m; }
                    else { Y0 = Y_up; pr0 = orow0 - gate_m; os0 = up_m; }
                    Y0[(unsigned int)oc * (unsigned int)os0 + (unsigned int)pr0] = an * s_finx[0][j] * acc[0][b][j];
#endif
                }
                const int orow1 = rs + 16 + 8 * kg + j;
                if (orow1 < tm) {
#if HIPFIRE_FP8_QKVZA
                    float* Y1;
                    int pr1, os1;
                    if (orow1 < qkv_m) { Y1 = Y_qkv; pr1 = orow1; os1 = qkv_m; }
                    else if (orow1 < qkv_m + z_m) { Y1 = Y_z; pr1 = orow1 - qkv_m; os1 = z_m; }
                    else if (orow1 < qkv_m + z_m + beta_m) { Y1 = Y_beta; pr1 = orow1 - (qkv_m + z_m); os1 = beta_m; }
                    else { Y1 = Y_alpha; pr1 = orow1 - (qkv_m + z_m + beta_m); os1 = alpha_m; }
                    Y1[(unsigned int)oc * (unsigned int)os1 + (unsigned int)pr1] = an * s_finx[1][j] * acc[1][b][j];
#elif HIPFIRE_FP8_RESIDUAL
                    Y_res[(unsigned int)oc * (unsigned int)res_m + (unsigned int)orow1] += an * s_finx[1][j] * acc[1][b][j];
#elif HIPFIRE_FP8_QKV
                    float* Y1;
                    int pr1, os1;
                    if (orow1 < q_m) { Y1 = Y_q; pr1 = orow1; os1 = q_m; }
                    else if (orow1 < q_m + k_m) { Y1 = Y_k; pr1 = orow1 - q_m; os1 = k_m; }
                    else { Y1 = Y_v; pr1 = orow1 - (q_m + k_m); os1 = v_m; }
                    Y1[(unsigned int)oc * (unsigned int)os1 + (unsigned int)pr1] = an * s_finx[1][j] * acc[1][b][j];
#else
                    float* Y1;
                    int pr1, os1;
                    if (orow1 < gate_m) { Y1 = Y_gate; pr1 = orow1; os1 = gate_m; }
                    else { Y1 = Y_up; pr1 = orow1 - gate_m; os1 = up_m; }
                    Y1[(unsigned int)oc * (unsigned int)os1 + (unsigned int)pr1] = an * s_finx[1][j] * acc[1][b][j];
#endif
                }
            }
        }
    }
#else
    // Leave accumulator units (final per-row scale) and add the zero-point
    // term, both gathered from the lane owning row 8*kg+j.
    float s_fin[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) s_fin[j] = __shfl(s_cur_self, 8 * kg + j);

    // Final store with incumbent output mapping.
    #pragma unroll
    for (int b = 0; b < HIPFIRE_FP8_BV; b++) {
        const int oc = bs + b * 16 + ml;
        if (oc < N) {
            const float an = X_row_scales[oc];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int orow = rs + 8 * kg + j;
                if (orow < tm) {
#if HIPFIRE_FP8_QKVZA
                    float* Y;
                    int pr, os;
                    if (orow < qkv_m) { Y = Y_qkv; pr = orow; os = qkv_m; }
                    else if (orow < qkv_m + z_m) { Y = Y_z; pr = orow - qkv_m; os = z_m; }
                    else if (orow < qkv_m + z_m + beta_m) { Y = Y_beta; pr = orow - (qkv_m + z_m); os = beta_m; }
                    else { Y = Y_alpha; pr = orow - (qkv_m + z_m + beta_m); os = alpha_m; }
                    Y[(unsigned int)oc * (unsigned int)os + (unsigned int)pr] = an * s_fin[j] * acc[b][j];
#elif HIPFIRE_FP8_RESIDUAL
                    Y_res[(unsigned int)oc * (unsigned int)res_m + (unsigned int)orow] += an * s_fin[j] * acc[b][j];
#elif HIPFIRE_FP8_QKV
                    float* Y;
                    int pr, os;
                    if (orow < q_m) { Y = Y_q; pr = orow; os = q_m; }
                    else if (orow < q_m + k_m) { Y = Y_k; pr = orow - q_m; os = k_m; }
                    else { Y = Y_v; pr = orow - (q_m + k_m); os = v_m; }
                    Y[(unsigned int)oc * (unsigned int)os + (unsigned int)pr] = an * s_fin[j] * acc[b][j];
#else
                    float* Y;
                    int pr, os;
                    if (orow < gate_m) { Y = Y_gate; pr = orow; os = gate_m; }
                    else { Y = Y_up; pr = orow - gate_m; os = up_m; }
                    Y[(unsigned int)oc * (unsigned int)os + (unsigned int)pr] = an * s_fin[j] * acc[b][j];
#endif
                }
            }
        }
    }
#endif
}
