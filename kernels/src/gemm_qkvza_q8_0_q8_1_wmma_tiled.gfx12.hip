// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[128];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

typedef int32_t __attribute__((ext_vector_type(2))) int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;
typedef float   __attribute__((ext_vector_type(8))) float8_t;

// Q8_0 x Q8_1 decode projection over prepacked lane-major weights.  One
// wave32 computes 16 rows.  Unlike the row-major arm, every A-fragment load
// is a contiguous 256-byte wave transaction.
extern "C" __global__ __launch_bounds__(32, 4)
void gemm_qkvza_q8_0_q8_1_wmma_tiled_gfx12(
    const unsigned char* __restrict__ A_qkv,
    const unsigned char* __restrict__ A_z,
    const unsigned char* __restrict__ A_beta,
    const unsigned char* __restrict__ A_alpha,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ y_qkv,
    float* __restrict__ y_z,
    float* __restrict__ y_beta,
    float* __restrict__ y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K
) {
    const int lane = threadIdx.x;
    const int m_lane = lane & 15;
    const int k_grp = lane >> 4;
    const int row_start = blockIdx.x * 16;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (row_start >= total_m) return;

    const unsigned char* A;
    int local_tile;
    if (row_start < qkv_m) {
        A = A_qkv; local_tile = row_start / 16;
    } else if (row_start < qkv_m + z_m) {
        A = A_z; local_tile = (row_start - qkv_m) / 16;
    } else if (row_start < qkv_m + z_m + beta_m) {
        A = A_beta; local_tile = (row_start - qkv_m - z_m) / 16;
    } else {
        A = A_alpha; local_tile = (row_start - qkv_m - z_m - beta_m) / 16;
    }

    const int nblocks = K / 32;
    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int bi = 0; bi < nblocks; ++bi) {
        const unsigned char* tile = A + ((long long)local_tile * nblocks + bi) * 544;
        const _Float16* scales = reinterpret_cast<const _Float16*>(tile);
        const float dw_self = (float)scales[m_lane];
        const block_q8_1_mmq* xb = Xq + (bi >> 2);
        const int sb = bi & 3;
        const float dx = m_lane == 0 ? __half2float(xb->ds4[sb].x) : 0.0f;
        int32x8_t cacc = {0, 0, 0, 0, 0, 0, 0, 0};

        #pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            const int32x2_t wi = *reinterpret_cast<const int32x2_t*>(
                tile + 32 + kt * 256 + lane * 8);
            int32x2_t xi = {0, 0};
            if (m_lane == 0) {
                xi = *reinterpret_cast<const int32x2_t*>(
                    xb->qs + sb * 32 + kt * 16 + k_grp * 8);
            }
            cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                true, wi, true, xi, cacc, true
            );
        }

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int src_lane = 8 * k_grp + j;
            const float dw = __shfl(dw_self, src_lane);
            acc[j] += dw * dx * (float)cacc[j];
        }
    }

    if (m_lane == 0) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int out_row = row_start + 8 * k_grp + j;
            if (out_row >= total_m) continue;
            if (out_row < qkv_m) {
                y_qkv[out_row] = acc[j];
            } else if (out_row < qkv_m + z_m) {
                y_z[out_row - qkv_m] = acc[j];
            } else if (out_row < qkv_m + z_m + beta_m) {
                y_beta[out_row - qkv_m - z_m] = acc[j];
            } else {
                y_alpha[out_row - qkv_m - z_m - beta_m] = acc[j];
            }
        }
    }
}
