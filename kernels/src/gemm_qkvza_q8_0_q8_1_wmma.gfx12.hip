// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) block_q8_0 {
    _Float16 d;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "bad block_q8_0 size");

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[128];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

typedef int8_t  __attribute__((ext_vector_type(8))) int8x8_t;
typedef int32_t __attribute__((ext_vector_type(2))) int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;
typedef float   __attribute__((ext_vector_type(8))) float8_t;

#ifndef HIPFIRE_QKVZA_I8_WMMA_UNROLL
#define HIPFIRE_QKVZA_I8_WMMA_UNROLL 1
#endif

// Single-token Q8_0 x Q8_1 QKVZA projection. One wave32 computes 16 output
// rows. Only the m_lane==0 B column is live; the remaining WMMA columns are
// zero padding. Each Q8_0/Q8_1 32-value block is two gfx12 i8 WMMA calls.
extern "C" __global__ __launch_bounds__(32, 4)
void gemm_qkvza_q8_0_q8_1_wmma_gfx12(
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
    const int tid = threadIdx.x;
    const int m_lane = tid & 15;
    const int k_grp = tid >> 4;
    const int row_start = blockIdx.x * 16;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (row_start >= total_m) return;

    const int my_row = row_start + m_lane;
    const int safe_row = my_row < total_m ? my_row : total_m - 1;
    const unsigned char* A;
    int local_row;
    if (safe_row < qkv_m) {
        A = A_qkv; local_row = safe_row;
    } else if (safe_row < qkv_m + z_m) {
        A = A_z; local_row = safe_row - qkv_m;
    } else if (safe_row < qkv_m + z_m + beta_m) {
        A = A_beta; local_row = safe_row - qkv_m - z_m;
    } else {
        A = A_alpha; local_row = safe_row - qkv_m - z_m - beta_m;
    }

    const int nblocks = K / 32;
    const block_q8_0* W = reinterpret_cast<const block_q8_0*>(A)
        + (long long)local_row * nblocks;
    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int base = 0; base < nblocks; base += HIPFIRE_QKVZA_I8_WMMA_UNROLL) {
        int32x8_t cacc[HIPFIRE_QKVZA_I8_WMMA_UNROLL];
        int8x8_t wv[HIPFIRE_QKVZA_I8_WMMA_UNROLL][2];
        int8x8_t xv[HIPFIRE_QKVZA_I8_WMMA_UNROLL][2];
        float dx[HIPFIRE_QKVZA_I8_WMMA_UNROLL];
        float dw_self[HIPFIRE_QKVZA_I8_WMMA_UNROLL];

        // Stage a window of independent weight/input blocks before issuing
        // their WMMA chains. This exposes global-load latency without LDS.
        #pragma unroll
        for (int u = 0; u < HIPFIRE_QKVZA_I8_WMMA_UNROLL; u++) {
            const int bi = base + u;
            const block_q8_1_mmq* xb = Xq + (bi >> 2);
            const int sb = bi & 3;
            dx[u] = m_lane == 0 ? __half2float(xb->ds4[sb].x) : 0.0f;
            dw_self[u] = (float)W[bi].d;
            cacc[u] = (int32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
            #pragma unroll
            for (int kt = 0; kt < 2; kt++) {
                wv[u][kt] = *(const int8x8_t*)(W[bi].qs + kt * 16 + k_grp * 8);
                xv[u][kt] = (int8x8_t){0, 0, 0, 0, 0, 0, 0, 0};
                if (m_lane == 0)
                    xv[u][kt] = *(const int8x8_t*)(
                        xb->qs + sb * 32 + kt * 16 + k_grp * 8);
            }
        }

        #pragma unroll
        for (int u = 0; u < HIPFIRE_QKVZA_I8_WMMA_UNROLL; u++) {
            #pragma unroll
            for (int kt = 0; kt < 2; kt++) {
                int32x2_t wi = *(const int32x2_t*)&wv[u][kt];
                int32x2_t xi = *(const int32x2_t*)&xv[u][kt];
                cacc[u] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                    true, wi, true, xi, cacc[u], true
                );
            }
        }

        #pragma unroll
        for (int u = 0; u < HIPFIRE_QKVZA_I8_WMMA_UNROLL; u++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int src_lane = 8 * k_grp + j;
                const float dw = __shfl(dw_self[u], src_lane);
                acc[j] += dw * dx[u] * (float)cacc[u][j];
            }
        }
    }

    if (m_lane == 0) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
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
