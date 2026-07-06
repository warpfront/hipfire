// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// 4-way fused HFQ4-G256 GEMV for the Qwen3.5 DeltaNet linear-attention
// layer preamble. One kernel launch performs all four projections from a
// single shared x_rot input:
//
//   y_qkv   = W_qkv   · x_rot      (M=qkv_m,   e.g. 6144 on 0.8B)
//   y_z     = W_z     · x_rot      (M=z_m,     e.g. 2048)
//   y_beta  = W_beta  · x_rot      (M=beta_m,  e.g. 16)
//   y_alpha = W_alpha · x_rot      (M=alpha_m, e.g. 16)
//
// Grid = qkv_m + z_m + beta_m + alpha_m workgroups × 32 threads. Each
// workgroup picks its target matrix/row via blockIdx.x range check, then
// runs the SAME 4x-unroll inner loop as gemv_hfq4g256.hip — including
// the 5302926 tail-group accumulator fix (acc[g%4], not all-into-acc0).
//
// Arch coverage: works on every RDNA generation (gfx1010 / gfx1013 /
// gfx1030 / gfx1100+) because gemv_hfq4g256.hip and its arch variants
// were unified onto the same 4-accumulator inner loop after the 5302926
// fix. The launch_bounds(32, 16) target matches the gfx1100 register
// budget; on gfx1010 this is the same single-wave-per-SIMD-half
// occupancy used by the standalone gemv_hfq4g256.hip — both kernels
// share the same VGPR pressure profile.
//
// Why fuse:
//   1. Four launches → one → saves ~3 × (host submit + dispatch floor)
//      per LA layer. On gfx1100 that's ~36 µs/layer ≈ ~640 µs per
//      forward. On gfx1010 the per-launch dispatch floor is even higher,
//      so the relative win is at least as large.
//   2. The 16-wave w_beta and w_alpha launches are catastrophically
//      under-subscribed on their own (16 / N_max_waves slots ≪ 1%).
//      Co-scheduling them with the 6144-wave wqkv body lets the wave
//      scheduler keep real work in flight during the tails.
//
// Bit-exact with four sequential gemv_hfq4g256 calls — the inner loop
// is a verbatim copy.

__launch_bounds__(32, 16)
extern "C" __global__ void fused_qkvza_hfq4g256(
    const char*  __restrict__ A_qkv,
    const char*  __restrict__ A_z,
    const char*  __restrict__ A_beta,
    const char*  __restrict__ A_alpha,
    const float* __restrict__ x,
    float*       __restrict__ y_qkv,
    float*       __restrict__ y_z,
    float*       __restrict__ y_beta,
    float*       __restrict__ y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K
) {
    const int gid = blockIdx.x;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (gid >= total_m) return;

    // Route this workgroup to its matrix + output row.
    const char* A;
    float* y;
    int row;
    if (gid < qkv_m) {
        A = A_qkv;   y = y_qkv;   row = gid;
    } else if (gid < qkv_m + z_m) {
        A = A_z;     y = y_z;     row = gid - qkv_m;
    } else if (gid < qkv_m + z_m + beta_m) {
        A = A_beta;  y = y_beta;  row = gid - (qkv_m + z_m);
    } else {
        A = A_alpha; y = y_alpha; row = gid - (qkv_m + z_m + beta_m);
    }

    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    // Four accumulators for the 4x group unroll. Tail groups go back into
    // acc[gi] to preserve the 4-way interleaving the combine step assumes
    // (this is the 5302926 bug fix — DO NOT collapse into acc0).
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        int base = g * 256 + tid * 8;

        #define DOG_PAIR(gp_a, gp_b, b_a, b_b, acc_a, acc_b) do { \
            const float xa0 = x[(b_a) + 0]; \
            const float xa1 = x[(b_a) + 1]; \
            const float xa2 = x[(b_a) + 2]; \
            const float xa3 = x[(b_a) + 3]; \
            const float xa4 = x[(b_a) + 4]; \
            const float xa5 = x[(b_a) + 5]; \
            const float xa6 = x[(b_a) + 6]; \
            const float xa7 = x[(b_a) + 7]; \
            const float xb0 = x[(b_b) + 0]; \
            const float xb1 = x[(b_b) + 1]; \
            const float xb2 = x[(b_b) + 2]; \
            const float xb3 = x[(b_b) + 3]; \
            const float xb4 = x[(b_b) + 4]; \
            const float xb5 = x[(b_b) + 5]; \
            const float xb6 = x[(b_b) + 6]; \
            const float xb7 = x[(b_b) + 7]; \
            const float xsa = ((xa0 + xa1) + (xa2 + xa3)) + ((xa4 + xa5) + (xa6 + xa7)); \
            const float xsb = ((xb0 + xb1) + (xb2 + xb3)) + ((xb4 + xb5) + (xb6 + xb7)); \
            const float sca = __builtin_bit_cast(float, *(const unsigned int*)(gp_a)); \
            const float zpa = __builtin_bit_cast(float, *(const unsigned int*)((gp_a) + 4)); \
            const unsigned int pka = *(const unsigned int*)((gp_a) + 8 + boff); \
            const float scb = __builtin_bit_cast(float, *(const unsigned int*)(gp_b)); \
            const float zpb = __builtin_bit_cast(float, *(const unsigned int*)((gp_b) + 4)); \
            const unsigned int pkb = *(const unsigned int*)((gp_b) + 8 + boff); \
            const float dota = (float)(pka & 0xFu) * xa0 \
                 + (float)((pka >> 4) & 0xFu) * xa1 \
                 + (float)((pka >> 8) & 0xFu) * xa2 \
                 + (float)((pka >> 12) & 0xFu) * xa3 \
                 + (float)((pka >> 16) & 0xFu) * xa4 \
                 + (float)((pka >> 20) & 0xFu) * xa5 \
                 + (float)((pka >> 24) & 0xFu) * xa6 \
                 + (float)((pka >> 28) & 0xFu) * xa7; \
            const float dotb = (float)(pkb & 0xFu) * xb0 \
                 + (float)((pkb >> 4) & 0xFu) * xb1 \
                 + (float)((pkb >> 8) & 0xFu) * xb2 \
                 + (float)((pkb >> 12) & 0xFu) * xb3 \
                 + (float)((pkb >> 16) & 0xFu) * xb4 \
                 + (float)((pkb >> 20) & 0xFu) * xb5 \
                 + (float)((pkb >> 24) & 0xFu) * xb6 \
                 + (float)((pkb >> 28) & 0xFu) * xb7; \
            (acc_a) += sca * dota + zpa * xsa; \
            (acc_b) += scb * dotb + zpb * xsb; \
        } while (0)

        DOG_PAIR(gp0, gp1, base, base + 256, acc0, acc1);
        DOG_PAIR(gp2, gp3, base + 512, base + 768, acc2, acc3);
        #undef DOG_PAIR
    }

    // Tail handling: preserve acc[gi] interleaving. See 5302926 bug report.
    #define TAIL_DOG(pk, sc, zp, b, a) do { \
        const float x0 = x[(b) + 0]; \
        const float x1 = x[(b) + 1]; \
        const float x2 = x[(b) + 2]; \
        const float x3 = x[(b) + 3]; \
        const float x4 = x[(b) + 4]; \
        const float x5 = x[(b) + 5]; \
        const float x6 = x[(b) + 6]; \
        const float x7 = x[(b) + 7]; \
        const float xs = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7)); \
        const float dot = (float)((pk) & 0xFu) * x0 \
             + (float)(((pk) >> 4) & 0xFu) * x1 \
             + (float)(((pk) >> 8) & 0xFu) * x2 \
             + (float)(((pk) >> 12) & 0xFu) * x3 \
             + (float)(((pk) >> 16) & 0xFu) * x4 \
             + (float)(((pk) >> 20) & 0xFu) * x5 \
             + (float)(((pk) >> 24) & 0xFu) * x6 \
             + (float)(((pk) >> 28) & 0xFu) * x7; \
        (a) += (sc) * dot + (zp) * xs; \
    } while (0)

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    acc += __shfl_down(acc, 16);
    acc += __shfl_down(acc, 8);
    acc += __shfl_down(acc, 4);
    acc += __shfl_down(acc, 2);
    acc += __shfl_down(acc, 1);

    if (tid == 0) y[row] = acc;
}
