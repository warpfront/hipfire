// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// int8-WMMA MQ4-G256v2 gate/up prefill candidate for gfx11 (RDNA3/3.5:
// gfx1100/gfx1151), port of the gfx1201 FP8 design
// (kernels/src/gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip, PR #757).
//
// Research kernel, default-off (host gate: HIPFIRE_GFX11_MQ4V2_IU8).
// Exact gfx1100/gfx1151, eager only. Companion to the MMQ incumbent
// gemm_mq4g256v2_residual_mmq_full_{set,add} in
// kernels/src/gemm_mq4g256v2_residual_mmq.hip.
//
// Geometry (mirrors the gfx12 champion, single slab): one wave32 workgroup
// computes a 16-row x (16*BV)-batch tile (BV independent chains).
// Grid: [ceil(tm/16), ceil(N/(16*BV)), 1]. Block: [32]. LDS: 0.
// gfx11 interleaved C mapping: acc[b][j] holds C[2*j+kg][ml] where
// ml = tid&15, kg = tid>>4. Overwrite semantics for gate_up/qkvza/qkv
// += for the residual form.
//
// Design deltas from the gfx12 FP8 file:
//  1. int8 activations: X_q is signed int8 (per-row scale ax from the
//     pre-pass in kernels/src/pack_f32_to_i8_mq4v2.gfx11.hip); S_half are
//     int32 sums of the *rounded int8* values (exact).
//  2. Weights feed the A operand as UNSIGNED int8 (signA=false): nibble q
//     verbatim, zero-extended to bytes. Activations feed B as SIGNED
//     (signB=true). Zero-point identity stays exact with no offset trick:
//       Y = ax * sum_h s_h * (D_h - z_h * S_h),
//     D_h = dot(q_u8, x_i8) over the 128-K half (i32 WMMA).
//  3. The i32 accumulator cannot be rescaled in place — and need not be:
//     facc sums ABSOLUTE per-half contributions (each half in its own
//     output units), so no s_prev/s_new ratio ever appears:
//       facc += s_h * float(iacc) + z_h * float(S_h); iacc = 0.
//     This is the v2 identity Y = ax * sum_h (s_h*D_h + z_h*S_h) with
//     W = s*q + z (zero ADDED, matching the F16 incumbent
//     `sc*q+zp` and the MMQ `sc*d*C + zp*(d*sum)` correction).
//     float(iacc) is exact (|iacc| <= 128*15*127 = 243,840 < 2^24).
//     Zero-scale halves work naturally (s_h = 0 zeroes the dot term
//     while the z_h*S_h term survives, exactly like the F16 path).
//     Cost: 8 cvt + 16 fma per chain per half.
//  4. No nibble LUT: q values 0..15 already fit int8, so the weight operand
//     is built with scalar shift/or (16 ops per 16-K fragment) instead of
//     v_perm. One 16 B X load + one 8 B weight load feed one WMMA (K=16);
//     eight WMMAs cover each 128-K half (ascending v = 0..7).
//  5. Fold/final-scale shuffles gather from lane (2*j+kg), the lane owning
//     row 2*j+kg of the 16-row tile under the interleaved mapping
//     (gfx12 uses 8*kg+j under its mapping).
//
// Dot products are K-order invariant, so ascending-v traversal differs from
// the MMQ incumbent only in f32 summation order: relL2 parity gate, not
// bit-exact. Deterministic fixed order, no atomics.
//
// Verified kills from gfx12 (do NOT retry here): dual alternating-half
// chains, packed/fma-reduced correction, folding zq.S into an extra WMMA,
// 4-slab, X-in-LDS.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef int   __attribute__((ext_vector_type(4))) v4i_t;
typedef int   __attribute__((ext_vector_type(8))) v8i_t;
typedef float __attribute__((ext_vector_type(8))) v8f_t;

// Batch-tile width (chains per wave). Host selects the symbol by N:
// N%192==0 -> BV=12, N%128==0 -> BV=8, N%64==0 -> BV=4.
#ifndef HIPFIRE_IU8_BV
#define HIPFIRE_IU8_BV 12
#endif
#ifndef HIPFIRE_IU8_GATEUP_KERNEL
#define HIPFIRE_IU8_GATEUP_KERNEL gemm_gate_up_mq4g256v2_wmma_iu8_gfx11_bt12
#endif

// HIPFIRE_IU8_RESIDUAL=1 compiles the single-matrix residual form
// (Y[N x M] += W[M x K] . X) with the incumbent residual ABI
// (A, X_i8, X_half_sums, X_row_scales, Y, M, K, N); the default is the
// two-matrix gate/up overwrite form.
#ifndef HIPFIRE_IU8_RESIDUAL
#define HIPFIRE_IU8_RESIDUAL 0
#endif

// HIPFIRE_IU8_QKVZA=1 compiles the 4-way fused DeltaNet projection form
// (A_qkv/A_z/A_beta/A_alpha -> Y_qkv/Y_z/Y_beta/Y_alpha, overwrite) with the
// incumbent qkvza ABI.
#ifndef HIPFIRE_IU8_QKVZA
#define HIPFIRE_IU8_QKVZA 0
#endif

// HIPFIRE_IU8_QKV=1 compiles the 3-way fused full-attention projection form
// (A_q/A_k/A_v -> Y_q/Y_k/Y_v, overwrite) with the incumbent QKV ABI.
#ifndef HIPFIRE_IU8_QKV
#define HIPFIRE_IU8_QKV 0
#endif

// Unpack 16 nibbles (8 payload bytes = 2 u32, q_i = (pk >> 4*i) & 0xF)
// into one per-lane unsigned-int8 WMMA A operand (16 int8 values in v4i_t).
// Nibble values 0..15 are exact in int8; zero-extension is the encoding.
__device__ static inline v4i_t mq4_qpk_to_u8(unsigned int pk0, unsigned int pk1) {
    v4i_t out;
    out[0] = (int)(((pk0 >>  0) & 0xFu)
                 | (((pk0 >>  4) & 0xFu) <<  8)
                 | (((pk0 >>  8) & 0xFu) << 16)
                 | (((pk0 >> 12) & 0xFu) << 24));
    out[1] = (int)(((pk0 >> 16) & 0xFu)
                 | (((pk0 >> 20) & 0xFu) <<  8)
                 | (((pk0 >> 24) & 0xFu) << 16)
                 | (((pk0 >> 28) & 0xFu) << 24));
    out[2] = (int)(((pk1 >>  0) & 0xFu)
                 | (((pk1 >>  4) & 0xFu) <<  8)
                 | (((pk1 >>  8) & 0xFu) << 16)
                 | (((pk1 >> 12) & 0xFu) << 24));
    out[3] = (int)(((pk1 >> 16) & 0xFu)
                 | (((pk1 >> 20) & 0xFu) <<  8)
                 | (((pk1 >> 24) & 0xFu) << 16)
                 | (((pk1 >> 28) & 0xFu) << 24));
    return out;
}

extern "C" __launch_bounds__(32, 2) __global__
#if HIPFIRE_IU8_QKVZA
void HIPFIRE_IU8_GATEUP_KERNEL(
    const char* A_qkv,
    const char* A_z,
    const char* A_beta,
    const char* A_alpha,
    const signed char* X_i8,
    const int* X_half_sums,
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
    int N) {
    const int tm = qkv_m + z_m + beta_m + alpha_m;
#elif HIPFIRE_IU8_RESIDUAL
void HIPFIRE_IU8_GATEUP_KERNEL(
    const char* A_w,
    const signed char* X_i8,
    const int* X_half_sums,
    const float* X_row_scales,
    float* Y_res,
    int res_m,
    int K,
    int N) {
    const int tm = res_m;
#elif HIPFIRE_IU8_QKV
void HIPFIRE_IU8_GATEUP_KERNEL(
    const char* A_q,
    const char* A_k,
    const char* A_v,
    const signed char* X_i8,
    const int* X_half_sums,
    const float* X_row_scales,
    float* Y_q,
    float* Y_k,
    float* Y_v,
    int q_m,
    int k_m,
    int v_m,
    int K,
    int N) {
    const int tm = q_m + k_m + v_m;
#else
void HIPFIRE_IU8_GATEUP_KERNEL(
    const char* A_gate,
    const char* A_up,
    const signed char* X_i8,
    const int* X_half_sums,
    const float* X_row_scales,
    float* Y_gate,
    float* Y_up,
    int gate_m,
    int up_m,
    int K,
    int N) {
    const int tm = gate_m + up_m;
#endif
    const int rs = blockIdx.x * 16;
    const int bs = blockIdx.y * (16 * HIPFIRE_IU8_BV);
    const int tid = threadIdx.x;
    if (rs >= tm || bs >= N) return;

    const int ml = tid & 15;
    const int kg = tid >> 4;

    // Route this lane's A-row (own header source).
    const int mr = rs + ml;
    const int sr = (mr < tm) ? mr : (tm - 1);
#if HIPFIRE_IU8_QKVZA
    const char* A;
    int lr;
    if (sr < qkv_m) { A = A_qkv; lr = sr; }
    else if (sr < qkv_m + z_m) { A = A_z; lr = sr - qkv_m; }
    else if (sr < qkv_m + z_m + beta_m) { A = A_beta; lr = sr - (qkv_m + z_m); }
    else { A = A_alpha; lr = sr - (qkv_m + z_m + beta_m); }
#elif HIPFIRE_IU8_RESIDUAL
    const char* A = A_w;
    const int lr = sr;
#elif HIPFIRE_IU8_QKV
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

    const int gpr = K / 256;
    // All byte offsets below fit in 32 bits: X holds N*K bytes with
    // N*K < 2^32, each weight row holds gpr*136 bytes, and Y indices stay
    // far below 2^32 at admitted shapes.
    const char* rb = A + (unsigned int)lr * (unsigned int)gpr * 136u;

    // Batch rows for the chains; out-of-range rows alias row 0 for
    // safe loads, stores stay guarded.
    int sb[HIPFIRE_IU8_BV];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_IU8_BV; b++) {
        const int t = bs + b * 16 + ml;
        sb[b] = (t < N) ? t : 0;
    }

    // Long-lived accumulators per chain: iacc holds the current 128-K half
    // in raw i32 WMMA units; facc holds prior halves in f32 units of their
    // own half scales (folded at each boundary, 8 cvt + 16 fma per chain).
    v8i_t iacc[HIPFIRE_IU8_BV];
    v8f_t facc[HIPFIRE_IU8_BV];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_IU8_BV; b++) {
        iacc[b] = v8i_t{0, 0, 0, 0, 0, 0, 0, 0};
        facc[b] = v8f_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    // 32-bit X row offsets (one VGPR each, not two): full X byte offsets
    // fit in 32 bits (N*K < 2^32 at admitted shapes).
    unsigned int xb_off[HIPFIRE_IU8_BV];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_IU8_BV; b++) xb_off[b] = (unsigned int)sb[b] * (unsigned int)K;

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

        // Half-sum base per chain; the live half is loaded inside the
        // h-loop (one live Si array, not two).
        unsigned int hs_base[HIPFIRE_IU8_BV];
        #pragma unroll
        for (int b = 0; b < HIPFIRE_IU8_BV; b++)
            hs_base[b] = (((unsigned int)sb[b] * (unsigned int)gpr + (unsigned int)g) * 2u);

        #pragma nounroll
        for (int h = 0; h < 2; h++) {
            // Absolute per-half scale/zero of this lane's row. A zero
            // scale zeroes the dot term below while the z*S term survives
            // (v2 W = s*q + z); no live flags, no carried scale state.
            const float s_h_self = (h == 0) ? sc0 : sc1;
            const float z_h_self = (h == 0) ? zp0 : zp1;


            // Eight K16 slices cover the 128-K half (ascending v; dot
            // products are K-order invariant). One 16 B X load and one 8 B
            // weight load (16 nibbles) feed one WMMA per chain.
            int Si[HIPFIRE_IU8_BV];
            #pragma unroll
            for (int b = 0; b < HIPFIRE_IU8_BV; b++)
                Si[b] = *(X_half_sums + hs_base[b] + (unsigned int)h);
            const char* half_base = payload + h * 64;
            // BV>=8 (BT8): rolled v-slices — the fully unrolled v x b nest
            // exceeds the VGPR file (256 + spills); rolling v keeps one
            // slice live (BT8: 208 VGPR, 0 spills) while the unrolled b-loop
            // preserves chain ILP. Smaller BVs keep the unrolled form.
#if HIPFIRE_IU8_BV >= 8
            #pragma unroll 1
#else
            #pragma unroll
#endif
            for (int v = 0; v < 8; v++) {
                const unsigned int pk0 = *(const unsigned int*)(half_base + v * 8);
                const unsigned int pk1 = *(const unsigned int*)(half_base + v * 8 + 4);
                v4i_t w = mq4_qpk_to_u8(pk0, pk1);
                const unsigned int x_off = (unsigned int)g * 256u + (unsigned int)h * 128u + (unsigned int)v * 16u;
                #pragma unroll
                for (int b = 0; b < HIPFIRE_IU8_BV; b++) {
                    const v4i_t x = *(const v4i_t*)(X_i8 + xb_off[b] + x_off);
                    // A = weights unsigned, B = activations signed.
                    iacc[b] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, w, true, x, iacc[b], false);
                }
            }
            // Fold the just-finished half into the absolute f32 acc:
            // facc += s_h*iacc + z_h*S_h, gathered per output row from its
            // owning lane (2*j+kg under the gfx11 interleaved C mapping:
            // acc[b][j] = C[2*j+kg][ml]).
            for (int j = 0; j < 8; j++) {
                const float s_j = __shfl(s_h_self, 2 * j + kg);
                const float z_j = __shfl(z_h_self, 2 * j + kg);
                #pragma unroll
                for (int b = 0; b < HIPFIRE_IU8_BV; b++) {
                    const float Sb = (float)Si[b];
                    facc[b][j] = fmaf(s_j, (float)iacc[b][j], fmaf(z_j, Sb, facc[b][j]));
                }
            }
            #pragma unroll
            for (int b = 0; b < HIPFIRE_IU8_BV; b++)
                iacc[b] = v8i_t{0, 0, 0, 0, 0, 0, 0, 0};
        }
    }

    // Final store with the gfx11 interleaved output mapping. facc is
    // already in absolute output units: Y = ax * facc.
    #pragma unroll
    for (int b = 0; b < HIPFIRE_IU8_BV; b++) {
        const int oc = bs + b * 16 + ml;
        if (oc < N) {
            const float an = X_row_scales[oc];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int orow = rs + 2 * j + kg;
                if (orow < tm) {
#if HIPFIRE_IU8_QKVZA
                    float* Y;
                    int pr, os;
                    if (orow < qkv_m) { Y = Y_qkv; pr = orow; os = qkv_m; }
                    else if (orow < qkv_m + z_m) { Y = Y_z; pr = orow - qkv_m; os = z_m; }
                    else if (orow < qkv_m + z_m + beta_m) { Y = Y_beta; pr = orow - (qkv_m + z_m); os = beta_m; }
                    else { Y = Y_alpha; pr = orow - (qkv_m + z_m + beta_m); os = alpha_m; }
                    Y[(unsigned int)oc * (unsigned int)os + (unsigned int)pr] = an * facc[b][j];
#elif HIPFIRE_IU8_RESIDUAL
                    Y_res[(unsigned int)oc * (unsigned int)res_m + (unsigned int)orow] += an * facc[b][j];
#elif HIPFIRE_IU8_QKV
                    float* Y;
                    int pr, os;
                    if (orow < q_m) { Y = Y_q; pr = orow; os = q_m; }
                    else if (orow < q_m + k_m) { Y = Y_k; pr = orow - q_m; os = k_m; }
                    else { Y = Y_v; pr = orow - (q_m + k_m); os = v_m; }
                    Y[(unsigned int)oc * (unsigned int)os + (unsigned int)pr] = an * facc[b][j];
#else
                    float* Y;
                    int pr, os;
                    if (orow < gate_m) { Y = Y_gate; pr = orow; os = gate_m; }
                    else { Y = Y_up; pr = orow - gate_m; os = up_m; }
                    Y[(unsigned int)oc * (unsigned int)os + (unsigned int)pr] = an * facc[b][j];
#endif
                }
            }
        }
    }
}
