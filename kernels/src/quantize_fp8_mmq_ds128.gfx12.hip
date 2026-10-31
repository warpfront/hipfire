// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// e4m3 per-128-K activation block for the gfx1201 FP8 WMMA routes
// (`moe_grouped_qt44_fp8_sym.gfx12.hip`). Same layout and launch contract as
// `quantize_int8_mmq_ds128` (`block_i8_128_quant.hip`): block (256,1,1), each thread 4 floats, folded
// 1-D grid [((K+1023)/1024) * N, 1, 1], output `[K/128, N]` of 136 B blocks, K % 128 == 0.
//
// Quantizer contract (one definition shared with the CPU oracle in `bench_moe_grouped_qt44_iu4`):
//   amax = max |x| over the 128 contiguous K elements (fmaxf, f32)
//   d    = RN(amax / 448.0f)                         (IEEE divide; 448 = e4m3 max)
//   code = e4m3_RNE(clamp(RN(x / d), -448, 448))     (IEEE divide; v_cvt_pk_fp8_f32, ties-to-even)
//   d == 0 (zero block or underflow): d = 0, every code 0
// Block: [f32 d][4 B aux, written 0][128 e4m3]. Byte order of the codes is the int8 sidecar's
// "nibble-pair order": within every 8-element K group g (16 per block), bytes 8g+0..3 =
// code[8g+0,2,4,6] and bytes 8g+4..7 = code[8g+1,3,5,7], which is the byte order one packed MQ4V2
// weight dword (8 nibbles, even/odd split) yields, so weight and activation codes pair without a
// byte shuffle in the K loop.

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

#if !defined(__gfx1201__) && defined(__HIP_DEVICE_COMPILE__)
#error "quantize_fp8_mmq_ds128.gfx12.hip is gfx1201-only"
#endif

struct block_fp8_128 {
    float d;
    int aux;
    unsigned char qs[128];
};
static_assert(sizeof(block_fp8_128) == 136, "bad block_fp8_128 size");

extern "C" __global__ void quantize_fp8_mmq_ds128(
    const float* __restrict__ X,
    block_fp8_128* __restrict__ Y,
    int K,
    int N
) {
    const int per_row = (K + (int)blockDim.x * 4 - 1) / ((int)blockDim.x * 4);
    if (per_row <= 0) return;
    const int token = (int)(blockIdx.x / (unsigned)per_row);
    const int t = (int)(blockIdx.x % (unsigned)per_row) * blockDim.x + threadIdx.x;
    const int i0 = t * 4;
    if (token >= N || i0 >= K) return;
    const int lane = threadIdx.x & 31;

    const float4 xi = *(const float4*)(X + (long long)token * K + i0);
    const float xv[4] = {xi.x, xi.y, xi.z, xi.w};

    float amax = fmaxf(fmaxf(fabsf(xv[0]), fabsf(xv[1])), fmaxf(fabsf(xv[2]), fabsf(xv[3])));
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }
    const float d = __fdiv_rn(amax, 448.0f);
    const bool zero = d == 0.0f;

    float q[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        q[e] = zero ? 0.0f : fminf(fmaxf(__fdiv_rn(xv[e], d), -448.0f), 448.0f);
    }
    unsigned p = 0u;
    p = (unsigned)__builtin_amdgcn_cvt_pk_fp8_f32(q[0], q[1], (int)p, false);
    p = (unsigned)__builtin_amdgcn_cvt_pk_fp8_f32(q[2], q[3], (int)p, true);

    // Nibble-pair order: the even lane of a pair holds codes 8g+0..3 and writes dword 2g =
    // {c0, c2, c4, c6}; the odd lane holds codes 8g+4..7 and writes dword 2g+1 = {c1, c3, c5, c7}.
    // Each lane sends the half its partner needs (even: its odd codes, odd: its even codes).
    const unsigned ev = (p & 0xFFu) | ((p >> 8) & 0xFF00u);           // q0, q2
    const unsigned od = ((p >> 8) & 0xFFu) | ((p >> 16) & 0xFF00u);   // q1, q3
    const bool odd = (lane & 1) != 0;
    const unsigned recv = (unsigned)__shfl_xor((int)(odd ? ev : od), 1, 32);
    const unsigned word = odd ? (recv | (od << 16)) : (ev | (recv << 16));

    block_fp8_128* out = Y + (long long)(i0 / 128) * N + token;
    reinterpret_cast<unsigned*>(out->qs)[lane] = word;
    if (lane == 0) {
        out->d = d;
        out->aux = 0;
    }
}
