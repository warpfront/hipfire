// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64

#include <hip/hip_runtime.h>

#ifndef HIPFIRE_Q8_W64_UNROLL
#define HIPFIRE_Q8_W64_UNROLL 8
#endif

#ifndef HIPFIRE_Q8_W64_SHARE_X
#define HIPFIRE_Q8_W64_SHARE_X 0
#endif

#ifndef HIPFIRE_Q8_W64_BUFFER
#define HIPFIRE_Q8_W64_BUFFER 0
#endif

// gfx12 wave64 Q8_0 GEMV. Each wave owns two adjacent rows, preserving the
// established wave32 dot and reduction order within each half while sharing
// the hot x-vector through cache.
extern "C" __launch_bounds__(64, 10)
__global__ void gemv_q8_0_wave64_gfx12(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int lane = threadIdx.x;
    const int half_lane = lane & 31;
    const int row = blockIdx.x * 2 + (lane >> 5);
    if (row >= M) return;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q8 + (size_t)row * blocks_per_row * 34;
#if HIPFIRE_Q8_W64_BUFFER
    const auto weight_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<unsigned char*>(A_q8), 0, 0xffffffffu, 0x31004000u);
#define Q8_W64_LOAD_SCALE(ptr) __builtin_bit_cast(_Float16, \
    __builtin_amdgcn_raw_buffer_load_b16( \
        weight_rsrc, (unsigned int)((ptr) - A_q8), 0, 16))
#define Q8_W64_LOAD_Q(ptr) ((signed char)__builtin_amdgcn_raw_buffer_load_b8( \
    weight_rsrc, (unsigned int)((ptr) - A_q8), 0, 16))
#else
#define Q8_W64_LOAD_SCALE(ptr) (*((const _Float16*)(ptr)))
#define Q8_W64_LOAD_Q(ptr) ((signed char)*(ptr))
#endif

#if HIPFIRE_Q8_W64_SHARE_X
#define Q8_W64_LOAD_X(ptr) ({ \
    float _x = lane < 32 ? *(ptr) : 0.0f; \
    const unsigned int _peer = __builtin_amdgcn_permlane64( \
        __builtin_bit_cast(unsigned int, _x)); \
    lane < 32 ? _x : __builtin_bit_cast(float, _peer); \
})
#else
#define Q8_W64_LOAD_X(ptr) (*(ptr))
#endif

    float sum = 0.0f;
    const int outer_iters = blocks_per_row / HIPFIRE_Q8_W64_UNROLL;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_data + oi * HIPFIRE_Q8_W64_UNROLL * 34;
        const float* xb = x + oi * HIPFIRE_Q8_W64_UNROLL * 32;
#pragma unroll
        for (int sub = 0; sub < HIPFIRE_Q8_W64_UNROLL; sub++) {
            const unsigned char* block = base + sub * 34;
            const float d = (float)Q8_W64_LOAD_SCALE(block);
            const signed char qval = Q8_W64_LOAD_Q(block + 2 + half_lane);
            const float xv = Q8_W64_LOAD_X(xb + sub * 32 + half_lane);
            sum += d * (float)qval * xv;
        }
    }
    for (int bi = outer_iters * HIPFIRE_Q8_W64_UNROLL; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 34;
        const float d = (float)Q8_W64_LOAD_SCALE(block);
        const signed char qval = Q8_W64_LOAD_Q(block + 2 + half_lane);
        const float xv = Q8_W64_LOAD_X(x + bi * 32 + half_lane);
        sum += d * (float)qval * xv;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset, 32);
    if (half_lane == 0) y[row] = sum;

#undef Q8_W64_LOAD_X
#undef Q8_W64_LOAD_Q
#undef Q8_W64_LOAD_SCALE
}
