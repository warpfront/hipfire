// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// WMMA-accelerated prefill GEMM for TQ2G128 (PrismML Q2_0 ternary) weights
// + F32 activations -> F32 output. gfx1100+ only (RDNA3/3.5 wave32 WMMA).
//
// Mirrors gemm_hfq4g256_wmma.hip: no LDS, no staging. Each lane dequantises
// its own 16-element A fragment straight into registers as F16 and feeds the
// matrix core. The earlier register-blocked scalar-FMA version of this GEMM
// tops out around 4 TFLOP/s because it is a VECTOR kernel; this one uses the
// matrix cores the MQ4 prefill path has always used.
//
// Layout:
//   A : TQ2G128-packed [M, K/128 x 34] bytes  ([f16 d][32 B codes] per group)
//   X : [B, K] F32 row-major (converted to F16 fragments in-register)
//   Y : [B, M] F32 row-major
//
// Grid:  [ceil(M / 16), ceil(B / 16)]
// Block: [32]
//
// The ternary layout falls out unusually cleanly here: codes pack LSB-first,
// 4 per byte, so 4 consecutive code bytes hold exactly 16 codes -- one whole
// K=16 WMMA sub-tile per u32, with code i at bit 2*i. (The bytes are read
// individually and recombined because a group is 34 B, so the code array sits
// at offset 2 mod 4 and a direct u32 load would be misaligned.)

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
extern "C" __global__ void gemm_tq2g128_wmma(
    const char*     __restrict__ A,    // TQ2G128 packed
    const float*    __restrict__ X,    // [B, K] F32
    float*          __restrict__ Y,    // [B, M] F32
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
    const long long row_stride = (long long)groups_per_row * 34;
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
        const char* gp = a_row + g * 34;
        const _Float16 d = *(const _Float16*)gp;
        const unsigned char* qs = (const unsigned char*)(gp + 2);

        // 128 weights per group / 16 per WMMA step = 8 sub-tiles.
        #pragma unroll 1
        for (int kt = 0; kt < 8; kt++) {
            const unsigned char* q = qs + kt * 4;
            const unsigned int pk = (unsigned int)q[0]
                                  | ((unsigned int)q[1] << 8)
                                  | ((unsigned int)q[2] << 16)
                                  | ((unsigned int)q[3] << 24);

            half16_t a_frag;
            #define DQ(i) a_frag[i] = d * (_Float16)((int)((pk >> (2 * (i))) & 0x3u) - 1)
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

    // row = 2*j + (tid>>4), col (= batch) = tid & 15.
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
