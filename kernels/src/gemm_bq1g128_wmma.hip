// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// WMMA-accelerated prefill GEMM for BQ1G128 (PrismML Q1_0 binary) weights
// + F32 activations -> F32 output. gfx1100+ only.
//
// Binary sibling of gemm_tq2g128_wmma.hip; see that file for the design.
// Layout difference: 8 codes per byte, so ONE K=16 sub-tile is 2 code bytes,
// and the sign decode is bit i of a 16-bit word -> +-d.
//
//   A : BQ1G128-packed [M, K/128 x 18] bytes ([f16 d][16 B codes] per group)
//   X : [B, K] F32 row-major (converted to F16 fragments in-register)
//   Y : [B, M] F32 row-major
// Grid: [ceil(M/16), ceil(B/16)]. Block: [32].

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

// B-tiles per wave. A 16-wide batch tile makes each weight row get re-read
// ceil(B/16) times, which dominates at the FFN shape (a 23.7 MB ternary
// weight re-read 32x at B=512). Decoding the A fragment ONCE and firing it
// at BT consecutive batch tiles cuts that traffic by BT.
#ifndef HIPFIRE_LOWBIT_WMMA_BT
#define HIPFIRE_LOWBIT_WMMA_BT 4
#endif

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_bq1g128_wmma(
    const char*     __restrict__ A,
    const float*    __restrict__ X,
    float*          __restrict__ Y,
    int M, int K, int B
) {
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * (16 * HIPFIRE_LOWBIT_WMMA_BT);
    const int tid  = threadIdx.x;
    const int lane = tid & 15;

    if (row_start >= M || batch_start >= B) return;

    const int my_m   = row_start + lane;
    const int safe_m = (my_m < M) ? my_m : (M - 1);

    const int groups_per_row = K / 128;
    const long long row_stride = (long long)groups_per_row * 18;
    const char* a_row = A + (long long)safe_m * row_stride;

    // One X row pointer per B-tile this wave owns.
    const float* x_rows[HIPFIRE_LOWBIT_WMMA_BT];
    float8_t acc[HIPFIRE_LOWBIT_WMMA_BT];
    #pragma unroll
    for (int bt = 0; bt < HIPFIRE_LOWBIT_WMMA_BT; bt++) {
        const int my_b = batch_start + bt * 16 + lane;
        const int safe_b = (my_b < B) ? my_b : (B - 1);
        x_rows[bt] = X + (long long)safe_b * K;
        acc[bt] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = a_row + g * 18;
        const _Float16 d = *(const _Float16*)gp;
        const _Float16 nd = -d;
        const unsigned char* qs = (const unsigned char*)(gp + 2);

        #pragma unroll 1
        for (int kt = 0; kt < 8; kt++) {
            const unsigned char* q = qs + kt * 2;
            const unsigned int pk = (unsigned int)q[0] | ((unsigned int)q[1] << 8);

            half16_t a_frag;
            #define DQ(i) a_frag[i] = ((pk >> (i)) & 1u) ? d : nd
            DQ(0);  DQ(1);  DQ(2);  DQ(3);
            DQ(4);  DQ(5);  DQ(6);  DQ(7);
            DQ(8);  DQ(9);  DQ(10); DQ(11);
            DQ(12); DQ(13); DQ(14); DQ(15);
            #undef DQ

            const int k = g * 128 + kt * 16;
            #pragma unroll
            for (int bt = 0; bt < HIPFIRE_LOWBIT_WMMA_BT; bt++) {
                if (batch_start + bt * 16 >= B) break;
                // X is F32 in this pipeline (the fused MQ4 prefill kernels
                // take F32 too, and there is no F16 staging buffer). Convert
                // in-register: 16 cvt ops against a 16x16x16 matrix op is
                // noise, and it avoids a whole staging pass + scratch buffer.
                const float* xp = x_rows[bt] + k;
                half16_t x_frag;
                #pragma unroll
                for (int e = 0; e < 16; e++) x_frag[e] = (_Float16)xp[e];
                acc[bt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    a_frag, x_frag, acc[bt]);
            }
        }
    }

    #pragma unroll
    for (int bt = 0; bt < HIPFIRE_LOWBIT_WMMA_BT; bt++) {
        const int out_col = batch_start + bt * 16 + lane;
        if (out_col >= B) continue;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[bt][j];
            }
        }
    }
}
