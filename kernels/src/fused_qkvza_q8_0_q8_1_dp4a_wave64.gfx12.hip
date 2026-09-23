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

// True-wave64 spatial-K Q8_0 x Q8_1 GEMV.  Eight lanes consume one 32-value
// block with one signed dot4 each; eight blocks are in flight per iteration.
// The same activation dword feeds two adjacent output rows.
extern "C" __launch_bounds__(64, 8)
__global__ void fused_qkvza_q8_0_q8_1_dp4a_wave64_gfx12(
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
    const int sub = lane & 7;
    const int blk_in_iter = lane >> 3;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    const int row0 = blockIdx.x * 2;
    const int row1 = row0 + 1;
    if (row0 >= total_m) return;

    const unsigned char* A0;
    const unsigned char* A1;
    float* y0;
    float* y1;
    int lr0;
    int lr1;

    if (row0 < qkv_m) {
        A0 = A_qkv; y0 = y_qkv; lr0 = row0;
    } else if (row0 < qkv_m + z_m) {
        A0 = A_z; y0 = y_z; lr0 = row0 - qkv_m;
    } else if (row0 < qkv_m + z_m + beta_m) {
        A0 = A_beta; y0 = y_beta; lr0 = row0 - qkv_m - z_m;
    } else {
        A0 = A_alpha; y0 = y_alpha; lr0 = row0 - qkv_m - z_m - beta_m;
    }

    const bool have1 = row1 < total_m;
    if (row1 < qkv_m) {
        A1 = A_qkv; y1 = y_qkv; lr1 = row1;
    } else if (row1 < qkv_m + z_m) {
        A1 = A_z; y1 = y_z; lr1 = row1 - qkv_m;
    } else if (row1 < qkv_m + z_m + beta_m) {
        A1 = A_beta; y1 = y_beta; lr1 = row1 - qkv_m - z_m;
    } else {
        A1 = A_alpha; y1 = y_alpha; lr1 = row1 - qkv_m - z_m - beta_m;
    }

    const int nblocks = K / 32;
    const block_q8_0* W0 = reinterpret_cast<const block_q8_0*>(A0)
        + (long long)lr0 * nblocks;
    const block_q8_0* W1 = reinterpret_cast<const block_q8_0*>(A1)
        + (long long)lr1 * nblocks;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int base = 0; base < nblocks; base += 8) {
        const int b = base + blk_in_iter;
        if (b >= nblocks) continue;
        const block_q8_1_mmq* xb = Xq + (b >> 2);
        const int sb = b & 3;
        int xq;
        __builtin_memcpy(&xq, xb->qs + sb * 32 + sub * 4, 4);
        const float dx = __half2float(xb->ds4[sb].x);

        int wq0;
        __builtin_memcpy(&wq0, W0[b].qs + sub * 4, 4);
        const int dot0 = __builtin_amdgcn_sudot4(true, wq0, true, xq, 0, false);
        acc0 += (float)W0[b].d * dx * (float)dot0;

        if (have1) {
            int wq1;
            __builtin_memcpy(&wq1, W1[b].qs + sub * 4, 4);
            const int dot1 = __builtin_amdgcn_sudot4(true, wq1, true, xq, 0, false);
            acc1 += (float)W1[b].d * dx * (float)dot1;
        }
    }

    #pragma unroll
    for (int mask = 32; mask > 0; mask >>= 1) {
        acc0 += __shfl_xor(acc0, mask, 64);
        acc1 += __shfl_xor(acc1, mask, 64);
    }
    if (lane == 0) {
        y0[lr0] = acc0;
        if (have1) y1[lr1] = acc1;
    }
}
