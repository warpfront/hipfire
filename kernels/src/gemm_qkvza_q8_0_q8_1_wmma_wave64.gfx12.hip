// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64

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

typedef int32_t __attribute__((ext_vector_type(4))) int32x4_t;
typedef float   __attribute__((ext_vector_type(4))) float4_t;

// Wave64 counterpart of the gfx12 Q8_0 x Q8_1 WMMA projection.  Four
// lanes provide each matrix row (rather than two in wave32), matching the
// native wave64 one-dword operand layout.  One wave still computes 16 rows.
extern "C" __global__ __launch_bounds__(64, 4)
void gemm_qkvza_q8_0_q8_1_wmma_wave64_gfx12(
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
    float4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int bi = 0; bi < nblocks; ++bi) {
        const block_q8_1_mmq* xb = Xq + (bi >> 2);
        const int sb = bi & 3;
        const float dx = m_lane == 0 ? __half2float(xb->ds4[sb].x) : 0.0f;
        const float dw_self = (float)W[bi].d;
        int32x4_t cacc = {0, 0, 0, 0};

        #pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            const int32_t wi = *(const int32_t*)(
                W[bi].qs + kt * 16 + k_grp * 4);
            int32_t xi = 0;
            if (m_lane == 0) {
                xi = *(const int32_t*)(
                    xb->qs + sb * 32 + kt * 16 + k_grp * 4);
            }
            cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64_gfx12(
                true, wi, true, xi, cacc, true
            );
        }

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int src_lane = 4 * k_grp + j;
            const float dw = __shfl(dw_self, src_lane);
            acc[j] += dw * dx * (float)cacc[j];
        }
    }

    if (m_lane == 0) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int out_row = row_start + 4 * k_grp + j;
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
