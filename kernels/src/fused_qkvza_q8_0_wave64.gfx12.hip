// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64

#include <hip/hip_runtime.h>

#ifndef HIPFIRE_QKVZA_Q8_W64_SHARE_X
#define HIPFIRE_QKVZA_Q8_W64_SHARE_X 0
#endif

// gfx12 wave64 Q8 QKVZA: two adjacent projection rows per wave. Each wave32
// half retains the reference Q8 dot/reduction order; the halves share x in
// cache. Adjacent gids normally target the same matrix, with only the three
// matrix boundaries potentially splitting EXEC between halves.
extern "C" __launch_bounds__(64, 10)
__global__ void fused_qkvza_q8_0_wave64_gfx12(
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
    const int half_lane = lane & 31;
    const int gid = blockIdx.x * 2 + (lane >> 5);
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (gid >= total_m) return;

    const unsigned char* A;
    float* y;
    int row;
    if (gid < qkv_m) {
        A=A_qkv; y=y_qkv; row=gid;
    } else if (gid < qkv_m + z_m) {
        A=A_z; y=y_z; row=gid-qkv_m;
    } else if (gid < qkv_m + z_m + beta_m) {
        A=A_beta; y=y_beta; row=gid-(qkv_m+z_m);
    } else {
        A=A_alpha; y=y_alpha; row=gid-(qkv_m+z_m+beta_m);
    }

    const int nblocks = K / 32;
    const unsigned char* row_data = A + (size_t)row * nblocks * 34;
#if HIPFIRE_QKVZA_Q8_W64_SHARE_X
#define QKVZA_Q8_W64_LOAD_X(ptr) ({ \
    float _x = lane < 32 ? *(ptr) : 0.0f; \
    const unsigned int _peer = __builtin_amdgcn_permlane64( \
        __builtin_bit_cast(unsigned int, _x)); \
    lane < 32 ? _x : __builtin_bit_cast(float, _peer); \
})
#else
#define QKVZA_Q8_W64_LOAD_X(ptr) (*(ptr))
#endif
    float sum = 0.0f;
    const int outer_iters = nblocks / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_data + oi * 8 * 34;
        const float* xb = x + oi * 256;
        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const unsigned char* block = base + sub * 34;
            const float d = (float)*((const _Float16*)block);
            const signed char qval = (signed char)block[2 + half_lane];
            const float xv = QKVZA_Q8_W64_LOAD_X(xb + sub * 32 + half_lane);
            sum += d * (float)qval * xv;
        }
    }
    for (int bi = outer_iters * 8; bi < nblocks; bi++) {
        const unsigned char* block = row_data + bi * 34;
        const float d = (float)*((const _Float16*)block);
        const signed char qval = (signed char)block[2 + half_lane];
        const float xv = QKVZA_Q8_W64_LOAD_X(x + bi * 32 + half_lane);
        sum += d * (float)qval * xv;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset, 32);
    if (half_lane == 0) y[row] = sum;
#undef QKVZA_Q8_W64_LOAD_X
}
