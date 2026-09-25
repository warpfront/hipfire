// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// int8 per-128-K activation block for the gfx12 A8 MMQ route
// (`gemm_mq4g256v2_residual_mmq_i8.gfx12.hip`). Concatenated by the host JIT
// ahead of the GEMM source; the standalone quantizer below is the reference
// producer (fused producers must emit bit-identical blocks).
//
// Quantizer contract (shared with the A8 quality emulation and the CPU
// oracle, one definition for all three):
//   amax = max |x| over the 128 contiguous K elements (fmaxf, f32)
//   d    = RN(amax / 127.0f)                      (IEEE divide)
//   q    = clamp(rintf(RN(x / d)), -127, 127)     (IEEE divide, ties-to-even)
//   zero block (amax == 0): d = 0, q = 0
//   s    = sum of the 128 q (exact int32)
//
// Byte order of qs (the "nibble-pair order"): within every 8-element K group
// g (16 per block), qs[8g + 0..3] = q[8g + 0], q[8g + 2], q[8g + 4], q[8g + 6]
// and qs[8g + 4..7] = q[8g + 1], q[8g + 3], q[8g + 5], q[8g + 7]. This is
// exactly the byte order the GEMM gets from one packed MQ4V2 weight dword
// (8 nibbles) with lo = pk & 0x0F0F0F0F, hi = (pk >> 4) & 0x0F0F0F0F, so the
// iu8 WMMA pairs each weight code with its own activation without any byte
// shuffle in the K loop. s and d are order-free.

#ifndef HIPFIRE_BLOCK_I8_128_QUANT_HIP
#define HIPFIRE_BLOCK_I8_128_QUANT_HIP

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

// int8 activation block: one 128-K group for one token (column), 136 B.
struct block_i8_128 {
    float d;
    int s;
    signed char qs[128];
};
static_assert(sizeof(block_i8_128) == 136, "bad block_i8_128 size");

// Wave32-synchronous quant of one 128-element group. Lane l owns elements
// 4l..4l+3 in xv[4]; lanes 2g and 2g+1 therefore own the 8-element group g.
// Every lane writes one dword of qs; lane 0 publishes d and s.
__device__ __forceinline__ void quantize_block_i8_128_wave(
    const float xv[4],
    block_i8_128* __restrict__ out,
    int lane
) {
    float amax = fmaxf(fmaxf(fabsf(xv[0]), fabsf(xv[1])), fmaxf(fabsf(xv[2]), fabsf(xv[3])));
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }
    const bool zero = amax == 0.0f;
    const float d = zero ? 0.0f : __fdiv_rn(amax, 127.0f);

    int q[4];
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float qf = zero ? 0.0f : fminf(fmaxf(rintf(__fdiv_rn(xv[e], d)), -127.0f), 127.0f);
        q[e] = (int)qf;
    }
    int s = q[0] + q[1] + q[2] + q[3];
    for (int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_xor(s, offset, 32);
    }

    // Nibble-pair order: the even lane of a pair holds q[8g+0..3] and writes
    // dword 2g = {q0, q2, q4, q6}; the odd lane holds q[8g+4..7] and writes
    // dword 2g+1 = {q1, q3, q5, q7}. Each lane sends the half its partner
    // needs (even: its odd codes, odd: its even codes).
    const unsigned ev = ((unsigned)q[0] & 255u) | (((unsigned)q[2] & 255u) << 8);
    const unsigned od = ((unsigned)q[1] & 255u) | (((unsigned)q[3] & 255u) << 8);
    const bool odd = (lane & 1) != 0;
    const unsigned recv = (unsigned)__shfl_xor((int)(odd ? ev : od), 1, 32);
    const unsigned word = odd ? (recv | (od << 16)) : (ev | (recv << 16));
    reinterpret_cast<unsigned*>(out->qs)[lane] = word;
    if (lane == 0) {
        out->d = d;
        out->s = s;
    }
}

// Standalone quantizer: same (X, Y, K, N) grid contract as
// quantize_int4_mmq_ds128 (block (256,1,1), each thread 4 floats, grid
// [(K+1023)/1024, N]); one wave32 covers exactly one 128-K group.
// Layout [K/128 x N]: Y[kb * N + token]. K % 128 == 0.
#ifndef HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE
extern "C" __global__ void quantize_int8_mmq_ds128(
    const float* __restrict__ X,
    block_i8_128* __restrict__ Y,
    int K,
    int N
) {
    const int token = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int i0 = t * 4;
    if (token >= N || i0 >= K) return;
    const int lane = threadIdx.x & 31;

    const float4 xi = *(const float4*)(X + (long long)token * K + i0);
    const float xv[4] = {xi.x, xi.y, xi.z, xi.w};

    const int kblock = i0 / 128;
    block_i8_128* out = Y + (long long)kblock * N + token;
    quantize_block_i8_128_wave(xv, out, lane);
}
#endif // HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE

#endif // HIPFIRE_BLOCK_I8_128_QUANT_HIP
