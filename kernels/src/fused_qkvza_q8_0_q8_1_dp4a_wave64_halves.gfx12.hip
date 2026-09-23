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
struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[128];
};
static_assert(sizeof(block_q8_0) == 34, "bad block_q8_0 size");
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

// Occupancy-friendly DP4A shape: preserve the champion's two independent
// wave32 halves and one accumulator per half.  Each half covers one row; its
// four eight-lane teams sweep four Q8 blocks at a time.
extern "C" __launch_bounds__(64, 10)
__global__ void fused_qkvza_q8_0_q8_1_dp4a_wave64_halves_gfx12(
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
    const int half_lane = lane & 31;
    const int sub = half_lane & 7;
    const int blk_in_iter = half_lane >> 3;
    const int gid = blockIdx.x * 2 + (lane >> 5);
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
        A = A_beta; y = y_beta; row = gid - qkv_m - z_m;
    } else {
        A = A_alpha; y = y_alpha; row = gid - qkv_m - z_m - beta_m;
    }

    const int nblocks = K / 32;
    const block_q8_0* W = reinterpret_cast<const block_q8_0*>(A)
        + (long long)row * nblocks;
    float acc = 0.0f;
    for (int base = 0; base < nblocks; base += 4) {
        const int b = base + blk_in_iter;
        if (b >= nblocks) continue;
        const block_q8_1_mmq* xb = Xq + (b >> 2);
        const int sb = b & 3;
        int xq;
        int wq;
        __builtin_memcpy(&xq, xb->qs + sb * 32 + sub * 4, 4);
        __builtin_memcpy(&wq, W[b].qs + sub * 4, 4);
        const int dot = __builtin_amdgcn_sudot4(true, wq, true, xq, 0, false);
        const float dx = __half2float(xb->ds4[sb].x);
        acc += (float)W[b].d * dx * (float)dot;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down(acc, off, 32);
    if (half_lane == 0) y[row] = acc;
}
