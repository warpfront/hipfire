// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#ifndef HIPFIRE_QKVZA_Q8_DUALROW_CHUNK
#define HIPFIRE_QKVZA_Q8_DUALROW_CHUNK 8
#endif

#ifndef HIPFIRE_QKVZA_Q8_DUALROW_NAME
#define HIPFIRE_QKVZA_Q8_DUALROW_NAME fused_qkvza_q8_0_dualrow_w32_u8_gfx12
#endif

// One wave32 owns two adjacent output rows. Both rows share every x load and
// all lanes take the same projection-routing branch. Staging CHUNK blocks
// before consuming either accumulator makes the pressure/window tradeoff
// explicit while preserving each row's original accumulation order.
extern "C" __launch_bounds__(32, 12)
__global__ void HIPFIRE_QKVZA_Q8_DUALROW_NAME(
    const unsigned char* __restrict__ A_qkv,
    const unsigned char* __restrict__ A_z,
    const unsigned char* __restrict__ A_beta,
    const unsigned char* __restrict__ A_alpha,
    const float* __restrict__ x,
    float* __restrict__ y_qkv,
    float* __restrict__ y_z,
    float* __restrict__ y_beta,
    float* __restrict__ y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K
) {
    const int lane = threadIdx.x;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    const int gid0 = blockIdx.x * 2;
    const int gid1 = gid0 + 1;
    if (gid0 >= total_m) return;

    const unsigned char* A0;
    float* y0;
    int row0;
    if (gid0 < qkv_m) {
        A0 = A_qkv; y0 = y_qkv; row0 = gid0;
    } else if (gid0 < qkv_m + z_m) {
        A0 = A_z; y0 = y_z; row0 = gid0 - qkv_m;
    } else if (gid0 < qkv_m + z_m + beta_m) {
        A0 = A_beta; y0 = y_beta; row0 = gid0 - qkv_m - z_m;
    } else {
        A0 = A_alpha; y0 = y_alpha; row0 = gid0 - qkv_m - z_m - beta_m;
    }

    const bool have1 = gid1 < total_m;
    const unsigned char* A1 = A0;
    float* y1 = y0;
    int row1 = row0;
    if (have1) {
        if (gid1 < qkv_m) {
            A1 = A_qkv; y1 = y_qkv; row1 = gid1;
        } else if (gid1 < qkv_m + z_m) {
            A1 = A_z; y1 = y_z; row1 = gid1 - qkv_m;
        } else if (gid1 < qkv_m + z_m + beta_m) {
            A1 = A_beta; y1 = y_beta; row1 = gid1 - qkv_m - z_m;
        } else {
            A1 = A_alpha; y1 = y_alpha; row1 = gid1 - qkv_m - z_m - beta_m;
        }
    }

    const int nblocks = K / 32;
    const unsigned char* row_data0 = A0 + (size_t)row0 * nblocks * 34;
    const unsigned char* row_data1 = A1 + (size_t)row1 * nblocks * 34;
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (int base = 0; base < nblocks; base += HIPFIRE_QKVZA_Q8_DUALROW_CHUNK) {
        float xv[HIPFIRE_QKVZA_Q8_DUALROW_CHUNK];
        float d0[HIPFIRE_QKVZA_Q8_DUALROW_CHUNK];
        float q0[HIPFIRE_QKVZA_Q8_DUALROW_CHUNK];
        float d1[HIPFIRE_QKVZA_Q8_DUALROW_CHUNK];
        float q1[HIPFIRE_QKVZA_Q8_DUALROW_CHUNK];

        #pragma unroll
        for (int sub = 0; sub < HIPFIRE_QKVZA_Q8_DUALROW_CHUNK; sub++) {
            const int bi = base + sub;
            const unsigned char* block0 = row_data0 + bi * 34;
            const unsigned char* block1 = row_data1 + bi * 34;
            xv[sub] = x[bi * 32 + lane];
            d0[sub] = (float)*reinterpret_cast<const _Float16*>(block0);
            q0[sub] = (float)(signed char)block0[2 + lane];
            d1[sub] = (float)*reinterpret_cast<const _Float16*>(block1);
            q1[sub] = (float)(signed char)block1[2 + lane];
        }

        #pragma unroll
        for (int sub = 0; sub < HIPFIRE_QKVZA_Q8_DUALROW_CHUNK; sub++) {
            sum0 += d0[sub] * q0[sub] * xv[sub];
            if (have1) sum1 += d1[sub] * q1[sub] * xv[sub];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset, 32);
        sum1 += __shfl_down(sum1, offset, 32);
    }
    if (lane == 0) {
        y0[row0] = sum0;
        if (have1) y1[row1] = sum1;
    }
}
