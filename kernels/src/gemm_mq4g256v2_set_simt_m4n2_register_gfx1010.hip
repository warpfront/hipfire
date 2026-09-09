// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Standalone FP32 MQ4G256V2 set experiment for exact gfx1010 only.
// M4xN2 register-only lane partition: four contiguous eight-lane row
// subgroups; each physical lane emulates baseline virtual lanes
// l, l+8, l+16, l+24. Exactly 32 persistent FP32 accumulators/thread.
// Production remains Q4 (gemm_mq4g256v2_set_simt_r1t8_q4_gfx1010) until
// Main admits this candidate. No residual path, LDS, barriers, WMMA,
// FP16 intermediate math, or cross-architecture body emission.
//
// Grounded on:
//   gemm_mq4g256v2_simt_experiments_gfx1010.hip:23-238 (Q4 set control)
//   gemv_mq4g256v2.hip:46-215 (portable four-chain oracle)
//   gemm_mq4g256v2_residual_simt.hip:31-243 (R1/N8 residual math twin)
//
// ABI: A*, X*, Y*, i32 M/K/N (raw 36, padded 48). Block 32, LDS 0,
// __launch_bounds__(32,4). Grid [ceil(M/4), ceil(N/2), 1]. Group 136B
// (hA+0, hB+4, payload+8). Y overwrite (set), not residual.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__gfx1010__)

// ---------------------------------------------------------------------------
// SetM4N2Register — four rows x two tokens; register-only virtual-lane fold.
// Each physical tid: r=tid>>3 row subgroup, l=tid&7 subgroup lane.
// Virtual baseline lanes: v0=l, v1=l+8, v2=l+16, v3=l+24.
// Half0 (+0 headers) owns v0/v1; half1 (+4 headers) owns v2/v3.
// Only one virtual lane's 4x8 decoded weights are live at a time.
// Local fold of baseline stages 16/8: C=(P0+P2)+(P1+P3); then width-8
// shuffle-down 4/2/1. Inactive row A reads clamp to row_base; inactive
// second-token X reads clamp to t0; all lanes execute; stores masked.
// ---------------------------------------------------------------------------

__launch_bounds__(32, 4)
extern "C" __global__ void gemm_mq4g256v2_set_simt_m4n2_register_gfx1010(
    const char* __restrict__ A,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int tid = (int)threadIdx.x;
    const int r = tid >> 3;                 // row subgroup [0,3]
    const int l = tid & 7;                  // subgroup lane [0,7]

    const int row_base = (int)blockIdx.x * 4;
    const int row = row_base + r;
    const int row_active = row < M;
    // Clamp inactive-row A traffic to the always-valid tile base row.
    const int safe_row = row_active ? row : row_base;

    const int t0 = (int)blockIdx.y * 2;
    const int t1 = t0 + 1;
    const int have_t1 = t1 < batch_size;
    // Clamp inactive second-token X traffic to t0 (always valid on launch).
    const int safe_t1 = have_t1 ? t1 : t0;

    // Baseline virtual lane ids emulated by this physical lane.
    const int v0 = l;
    const int v1 = l + 8;
    const int v2 = l + 16;
    const int v3 = l + 24;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)safe_row * groups_per_row * 136;

    // acc[s][c][t]: s=virtual lane 0..3, c=group-modulo chain 0..3, t=token 0..1.
    // Named + literal token indices only — no runtime-indexed accumulator select.
    float s0c0t0 = 0.0f, s0c0t1 = 0.0f;
    float s0c1t0 = 0.0f, s0c1t1 = 0.0f;
    float s0c2t0 = 0.0f, s0c2t1 = 0.0f;
    float s0c3t0 = 0.0f, s0c3t1 = 0.0f;

    float s1c0t0 = 0.0f, s1c0t1 = 0.0f;
    float s1c1t0 = 0.0f, s1c1t1 = 0.0f;
    float s1c2t0 = 0.0f, s1c2t1 = 0.0f;
    float s1c3t0 = 0.0f, s1c3t1 = 0.0f;

    float s2c0t0 = 0.0f, s2c0t1 = 0.0f;
    float s2c1t0 = 0.0f, s2c1t1 = 0.0f;
    float s2c2t0 = 0.0f, s2c2t1 = 0.0f;
    float s2c3t0 = 0.0f, s2c3t1 = 0.0f;

    float s3c0t0 = 0.0f, s3c0t1 = 0.0f;
    float s3c1t0 = 0.0f, s3c1t1 = 0.0f;
    float s3c2t0 = 0.0f, s3c2t1 = 0.0f;
    float s3c3t0 = 0.0f, s3c3t1 = 0.0f;

    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;

    // Dual-half header decode identical to Q4 control (lines 62-68):
    // u32 at +0 / +4 hold packed fp16 scale (lo) + zero (hi). Half select is
    // fixed per virtual-lane class (v0/v1 → +0, v2/v3 → +4), matching
    // baseline tid<16 ? hA : hB for emulated lanes l / l+8 / l+16 / l+24.
    #define LOAD_HALF0_SC_ZP(gp, sc, zp) do {                                   \
        const unsigned int hA = *(const unsigned int*)(gp);                     \
        (sc) = __half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu)));  \
        (zp) = __half2float(__ushort_as_half((unsigned short)(hA >> 16)));      \
    } while (0)

    #define LOAD_HALF1_SC_ZP(gp, sc, zp) do {                                   \
        const unsigned int hB = *(const unsigned int*)((gp) + 4);               \
        (sc) = __half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu)));  \
        (zp) = __half2float(__ushort_as_half((unsigned short)(hB >> 16)));      \
    } while (0)

    // Exact Q4 eight-nibble dequant: w_i = sc * (float)q_i + zp (FP32).
    #define DEQUANT8(pk, sc, zp, wpfx)                                          \
        const float wpfx##0 = (sc) * (float)(((pk)       ) & 0xFu) + (zp);      \
        const float wpfx##1 = (sc) * (float)(((pk) >>  4) & 0xFu) + (zp);      \
        const float wpfx##2 = (sc) * (float)(((pk) >>  8) & 0xFu) + (zp);      \
        const float wpfx##3 = (sc) * (float)(((pk) >> 12) & 0xFu) + (zp);      \
        const float wpfx##4 = (sc) * (float)(((pk) >> 16) & 0xFu) + (zp);      \
        const float wpfx##5 = (sc) * (float)(((pk) >> 20) & 0xFu) + (zp);      \
        const float wpfx##6 = (sc) * (float)(((pk) >> 24) & 0xFu) + (zp);      \
        const float wpfx##7 = (sc) * (float)(((pk) >> 28) & 0xFu) + (zp)

    // Exact baseline eight-product FP32 source expression into one chain.
    #define DOT8(acc, wpfx, x0, x1, x2, x3, x4, x5, x6, x7)                     \
        (acc) += wpfx##0 * (x0) + wpfx##1 * (x1) + wpfx##2 * (x2) + wpfx##3 * (x3) \
               + wpfx##4 * (x4) + wpfx##5 * (x5) + wpfx##6 * (x6) + wpfx##7 * (x7)

    // One virtual lane, four groups already header-decoded: load 4 payloads,
    // pre-dequant all 4x8 weights, consume token t0 then t1 into named chains.
    // Weights for this virtual lane drop before the next virtual lane runs.
    #define PROCESS_VLANE_FOUR(                                                         \
        v, sc0, zp0, sc1, zp1, sc2, zp2, sc3, zp3,                                      \
        a0t0, a0t1, a1t0, a1t1, a2t0, a2t1, a3t0, a3t1) do {                            \
        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + (v) * 4);                   \
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + (v) * 4);                   \
        unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + (v) * 4);                   \
        unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + (v) * 4);                   \
        DEQUANT8(pk0, sc0, zp0, w0_);                                                   \
        DEQUANT8(pk1, sc1, zp1, w1_);                                                   \
        DEQUANT8(pk2, sc2, zp2, w2_);                                                   \
        DEQUANT8(pk3, sc3, zp3, w3_);                                                   \
        const int base = g * 256 + (v) * 8;                                             \
        /* token t0 */                                                                  \
        {                                                                               \
            const float* xb = X + (long long)t0 * K;                                    \
            const float x0 = xb[base    ];                                              \
            const float x1 = xb[base + 1];                                              \
            const float x2 = xb[base + 2];                                              \
            const float x3 = xb[base + 3];                                              \
            const float x4 = xb[base + 4];                                              \
            const float x5 = xb[base + 5];                                              \
            const float x6 = xb[base + 6];                                              \
            const float x7 = xb[base + 7];                                              \
            DOT8(a0t0, w0_, x0, x1, x2, x3, x4, x5, x6, x7);                            \
            const float x1_0 = xb[base + 256    ];                                      \
            const float x1_1 = xb[base + 256 + 1];                                      \
            const float x1_2 = xb[base + 256 + 2];                                      \
            const float x1_3 = xb[base + 256 + 3];                                      \
            const float x1_4 = xb[base + 256 + 4];                                      \
            const float x1_5 = xb[base + 256 + 5];                                      \
            const float x1_6 = xb[base + 256 + 6];                                      \
            const float x1_7 = xb[base + 256 + 7];                                      \
            DOT8(a1t0, w1_, x1_0, x1_1, x1_2, x1_3, x1_4, x1_5, x1_6, x1_7);            \
            const float x2_0 = xb[base + 512    ];                                      \
            const float x2_1 = xb[base + 512 + 1];                                      \
            const float x2_2 = xb[base + 512 + 2];                                      \
            const float x2_3 = xb[base + 512 + 3];                                      \
            const float x2_4 = xb[base + 512 + 4];                                      \
            const float x2_5 = xb[base + 512 + 5];                                      \
            const float x2_6 = xb[base + 512 + 6];                                      \
            const float x2_7 = xb[base + 512 + 7];                                      \
            DOT8(a2t0, w2_, x2_0, x2_1, x2_2, x2_3, x2_4, x2_5, x2_6, x2_7);            \
            const float x3_0 = xb[base + 768    ];                                      \
            const float x3_1 = xb[base + 768 + 1];                                      \
            const float x3_2 = xb[base + 768 + 2];                                      \
            const float x3_3 = xb[base + 768 + 3];                                      \
            const float x3_4 = xb[base + 768 + 4];                                      \
            const float x3_5 = xb[base + 768 + 5];                                      \
            const float x3_6 = xb[base + 768 + 6];                                      \
            const float x3_7 = xb[base + 768 + 7];                                      \
            DOT8(a3t0, w3_, x3_0, x3_1, x3_2, x3_3, x3_4, x3_5, x3_6, x3_7);            \
        }                                                                               \
        /* token t1 (safe_t1 when t1 out of range) */                                   \
        {                                                                               \
            const float* xb = X + (long long)safe_t1 * K;                               \
            const float x0 = xb[base    ];                                              \
            const float x1 = xb[base + 1];                                              \
            const float x2 = xb[base + 2];                                              \
            const float x3 = xb[base + 3];                                              \
            const float x4 = xb[base + 4];                                              \
            const float x5 = xb[base + 5];                                              \
            const float x6 = xb[base + 6];                                              \
            const float x7 = xb[base + 7];                                              \
            DOT8(a0t1, w0_, x0, x1, x2, x3, x4, x5, x6, x7);                            \
            const float x1_0 = xb[base + 256    ];                                      \
            const float x1_1 = xb[base + 256 + 1];                                      \
            const float x1_2 = xb[base + 256 + 2];                                      \
            const float x1_3 = xb[base + 256 + 3];                                      \
            const float x1_4 = xb[base + 256 + 4];                                      \
            const float x1_5 = xb[base + 256 + 5];                                      \
            const float x1_6 = xb[base + 256 + 6];                                      \
            const float x1_7 = xb[base + 256 + 7];                                      \
            DOT8(a1t1, w1_, x1_0, x1_1, x1_2, x1_3, x1_4, x1_5, x1_6, x1_7);            \
            const float x2_0 = xb[base + 512    ];                                      \
            const float x2_1 = xb[base + 512 + 1];                                      \
            const float x2_2 = xb[base + 512 + 2];                                      \
            const float x2_3 = xb[base + 512 + 3];                                      \
            const float x2_4 = xb[base + 512 + 4];                                      \
            const float x2_5 = xb[base + 512 + 5];                                      \
            const float x2_6 = xb[base + 512 + 6];                                      \
            const float x2_7 = xb[base + 512 + 7];                                      \
            DOT8(a2t1, w2_, x2_0, x2_1, x2_2, x2_3, x2_4, x2_5, x2_6, x2_7);            \
            const float x3_0 = xb[base + 768    ];                                      \
            const float x3_1 = xb[base + 768 + 1];                                      \
            const float x3_2 = xb[base + 768 + 2];                                      \
            const float x3_3 = xb[base + 768 + 3];                                      \
            const float x3_4 = xb[base + 768 + 4];                                      \
            const float x3_5 = xb[base + 768 + 5];                                      \
            const float x3_6 = xb[base + 768 + 6];                                      \
            const float x3_7 = xb[base + 768 + 7];                                      \
            DOT8(a3t1, w3_, x3_0, x3_1, x3_2, x3_3, x3_4, x3_5, x3_6, x3_7);            \
        }                                                                               \
    } while (0)

    // Single-group tail: one chain, half0 then half1 virtual lanes, t0 then t1.
    #define PROCESS_VLANE_ONE(v, sc, zp, at0, at1) do {                                 \
        unsigned int pk = *(const unsigned int*)(gp + 8 + (v) * 4);                     \
        DEQUANT8(pk, sc, zp, w_);                                                       \
        const int base = g * 256 + (v) * 8;                                             \
        {                                                                               \
            const float* xb = X + (long long)t0 * K;                                    \
            const float x0 = xb[base    ];                                              \
            const float x1 = xb[base + 1];                                              \
            const float x2 = xb[base + 2];                                              \
            const float x3 = xb[base + 3];                                              \
            const float x4 = xb[base + 4];                                              \
            const float x5 = xb[base + 5];                                              \
            const float x6 = xb[base + 6];                                              \
            const float x7 = xb[base + 7];                                              \
            DOT8(at0, w_, x0, x1, x2, x3, x4, x5, x6, x7);                              \
        }                                                                               \
        {                                                                               \
            const float* xb = X + (long long)safe_t1 * K;                               \
            const float x0 = xb[base    ];                                              \
            const float x1 = xb[base + 1];                                              \
            const float x2 = xb[base + 2];                                              \
            const float x3 = xb[base + 3];                                              \
            const float x4 = xb[base + 4];                                              \
            const float x5 = xb[base + 5];                                              \
            const float x6 = xb[base + 6];                                              \
            const float x7 = xb[base + 7];                                              \
            DOT8(at1, w_, x0, x1, x2, x3, x4, x5, x6, x7);                              \
        }                                                                               \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + (long long)g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        // ---- Half 0: +0 headers, virtual lanes v0 then v1 ----
        float sc0, zp0, sc1, zp1, sc2, zp2, sc3, zp3;
        LOAD_HALF0_SC_ZP(gp0, sc0, zp0);
        LOAD_HALF0_SC_ZP(gp1, sc1, zp1);
        LOAD_HALF0_SC_ZP(gp2, sc2, zp2);
        LOAD_HALF0_SC_ZP(gp3, sc3, zp3);

        PROCESS_VLANE_FOUR(
            v0, sc0, zp0, sc1, zp1, sc2, zp2, sc3, zp3,
            s0c0t0, s0c0t1, s0c1t0, s0c1t1, s0c2t0, s0c2t1, s0c3t0, s0c3t1);
        PROCESS_VLANE_FOUR(
            v1, sc0, zp0, sc1, zp1, sc2, zp2, sc3, zp3,
            s1c0t0, s1c0t1, s1c1t0, s1c1t1, s1c2t0, s1c2t1, s1c3t0, s1c3t1);

        // ---- Half 1: +4 headers, virtual lanes v2 then v3 ----
        LOAD_HALF1_SC_ZP(gp0, sc0, zp0);
        LOAD_HALF1_SC_ZP(gp1, sc1, zp1);
        LOAD_HALF1_SC_ZP(gp2, sc2, zp2);
        LOAD_HALF1_SC_ZP(gp3, sc3, zp3);

        PROCESS_VLANE_FOUR(
            v2, sc0, zp0, sc1, zp1, sc2, zp2, sc3, zp3,
            s2c0t0, s2c0t1, s2c1t0, s2c1t1, s2c2t0, s2c2t1, s2c3t0, s2c3t1);
        PROCESS_VLANE_FOUR(
            v3, sc0, zp0, sc1, zp1, sc2, zp2, sc3, zp3,
            s3c0t0, s3c0t1, s3c1t0, s3c1t1, s3c2t0, s3c2t1, s3c3t0, s3c3t1);
    }

    // Tail groups → chain c = g&3, same mapping as Q4/GEMV (no fourth tail).
    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + (long long)g * 136;
        float sc, zp;
        LOAD_HALF0_SC_ZP(gp, sc, zp);
        PROCESS_VLANE_ONE(v0, sc, zp, s0c0t0, s0c0t1);
        PROCESS_VLANE_ONE(v1, sc, zp, s1c0t0, s1c0t1);
        LOAD_HALF1_SC_ZP(gp, sc, zp);
        PROCESS_VLANE_ONE(v2, sc, zp, s2c0t0, s2c0t1);
        PROCESS_VLANE_ONE(v3, sc, zp, s3c0t0, s3c0t1);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + (long long)g * 136;
        float sc, zp;
        LOAD_HALF0_SC_ZP(gp, sc, zp);
        PROCESS_VLANE_ONE(v0, sc, zp, s0c1t0, s0c1t1);
        PROCESS_VLANE_ONE(v1, sc, zp, s1c1t0, s1c1t1);
        LOAD_HALF1_SC_ZP(gp, sc, zp);
        PROCESS_VLANE_ONE(v2, sc, zp, s2c1t0, s2c1t1);
        PROCESS_VLANE_ONE(v3, sc, zp, s3c1t0, s3c1t1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + (long long)g * 136;
        float sc, zp;
        LOAD_HALF0_SC_ZP(gp, sc, zp);
        PROCESS_VLANE_ONE(v0, sc, zp, s0c2t0, s0c2t1);
        PROCESS_VLANE_ONE(v1, sc, zp, s1c2t0, s1c2t1);
        LOAD_HALF1_SC_ZP(gp, sc, zp);
        PROCESS_VLANE_ONE(v2, sc, zp, s2c2t0, s2c2t1);
        PROCESS_VLANE_ONE(v3, sc, zp, s3c2t0, s3c2t1);
    }

    #undef PROCESS_VLANE_ONE
    #undef PROCESS_VLANE_FOUR
    #undef DOT8
    #undef DEQUANT8
    #undef LOAD_HALF1_SC_ZP
    #undef LOAD_HALF0_SC_ZP

    // Pairwise chain combine P[v,t] = (A0+A1)+(A2+A3) — exact Q4/GEMV parens.
    // Local fold of baseline shuffle stages 16/8:
    //   R16(l)=P0+P2, R16(l+8)=P1+P3, R8(l)=(P0+P2)+(P1+P3).
    // Then width-8 shuffle-down 4/2/1 (= baseline stages 4/2/1 on residue mod 8).
    // width=8 keeps partitions [0,7],[8,15],[16,23],[24,31] as independent rows.

    // ---- token t0 ----
    {
        const float P0 = (s0c0t0 + s0c1t0) + (s0c2t0 + s0c3t0);
        const float P1 = (s1c0t0 + s1c1t0) + (s1c2t0 + s1c3t0);
        const float P2 = (s2c0t0 + s2c1t0) + (s2c2t0 + s2c3t0);
        const float P3 = (s3c0t0 + s3c1t0) + (s3c2t0 + s3c3t0);
        float val = (P0 + P2) + (P1 + P3);
        val += __shfl_down(val, 4, 8);
        val += __shfl_down(val, 2, 8);
        val += __shfl_down(val, 1, 8);
        if (l == 0 && row_active)
            Y[(long long)t0 * (long long)M + row] = val;
    }

    // ---- token t1 (store masked when t1 >= N) ----
    {
        const float P0 = (s0c0t1 + s0c1t1) + (s0c2t1 + s0c3t1);
        const float P1 = (s1c0t1 + s1c1t1) + (s1c2t1 + s1c3t1);
        const float P2 = (s2c0t1 + s2c1t1) + (s2c2t1 + s2c3t1);
        const float P3 = (s3c0t1 + s3c1t1) + (s3c2t1 + s3c3t1);
        float val = (P0 + P2) + (P1 + P3);
        val += __shfl_down(val, 4, 8);
        val += __shfl_down(val, 2, 8);
        val += __shfl_down(val, 1, 8);
        if (l == 0 && row_active && have_t1)
            Y[(long long)t1 * (long long)M + row] = val;
    }
}

#endif // defined(__gfx1010__)
