// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64

#include <hip/hip_runtime.h>

#ifndef HIPFIRE_QKVZA_Q8_ROWS_PER_WAVE
#define HIPFIRE_QKVZA_Q8_ROWS_PER_WAVE 2
#endif

#if HIPFIRE_QKVZA_Q8_ROWS_PER_WAVE == 2
#define HIPFIRE_QKVZA_Q8_ROWS_NAME fused_qkvza_q8_0_wave64_rows2_multiacc_gfx12
#elif HIPFIRE_QKVZA_Q8_ROWS_PER_WAVE == 4
#define HIPFIRE_QKVZA_Q8_ROWS_NAME fused_qkvza_q8_0_wave64_rows4_multiacc_gfx12
#elif HIPFIRE_QKVZA_Q8_ROWS_PER_WAVE == 8
#define HIPFIRE_QKVZA_Q8_ROWS_NAME fused_qkvza_q8_0_wave64_rows8_multiacc_gfx12
#else
#error "HIPFIRE_QKVZA_Q8_ROWS_PER_WAVE must be 2, 4, or 8"
#endif

#if defined(__gfx1200__) || defined(__gfx1201__)

// Pack 2/4/8 Q8 projection rows into one wave64.  Each subgroup owns one
// row and covers a 32-value Q8 block with 32/16/8 lanes respectively.  Four
// independent accumulators keep the weight-load window open while avoiding
// the single FMA dependency chain in the current rows=2 kernel.
extern "C" __launch_bounds__(64, 8)
__global__ void HIPFIRE_QKVZA_Q8_ROWS_NAME(
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
    constexpr int ROWS = HIPFIRE_QKVZA_Q8_ROWS_PER_WAVE;
    constexpr int LPG = 64 / ROWS;
    constexpr int ELEMS = 32 / LPG;
    const int lane = threadIdx.x;
    const int subgroup = lane / LPG;
    const int lane_in = lane & (LPG - 1);
    const int gid = blockIdx.x * ROWS + subgroup;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (gid >= total_m) return;

    const unsigned char* A;
    float* y;
    int row;
    if (gid < qkv_m) {
        A = A_qkv; y = y_qkv; row = gid;
    } else if (gid < qkv_m + z_m) {
        A = A_z; y = y_z; row = gid - qkv_m;
    } else if (gid < qkv_m + z_m + beta_m) {
        A = A_beta; y = y_beta; row = gid - (qkv_m + z_m);
    } else {
        A = A_alpha; y = y_alpha; row = gid - (qkv_m + z_m + beta_m);
    }

    const int nblocks = K / 32;
    const unsigned char* row_data = A + (size_t)row * nblocks * 34;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int bi = 0; bi < nblocks; ++bi) {
        const unsigned char* block = row_data + bi * 34;
        const float d = (float)*reinterpret_cast<const _Float16*>(block);
        const int k0 = bi * 32 + lane_in * ELEMS;
        float v = 0.0f;
#pragma unroll
        for (int j = 0; j < ELEMS; ++j) {
            const signed char q = (signed char)block[2 + lane_in * ELEMS + j];
            v += d * (float)q * x[k0 + j];
        }
        if ((bi & 3) == 0) a0 += v;
        else if ((bi & 3) == 1) a1 += v;
        else if ((bi & 3) == 2) a2 += v;
        else a3 += v;
    }

    float sum = (a0 + a1) + (a2 + a3);
    for (int offset = LPG / 2; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset, LPG);
    if (lane_in == 0) y[row] = sum;
}

#endif

#undef HIPFIRE_QKVZA_Q8_ROWS_NAME
