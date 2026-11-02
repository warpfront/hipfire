// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// A3B shared-expert down (M = 2048, K = 512, MQ4G256V2 / qt=44): sigmoid-gated
// scaled residual GEMV,  y[row] += sigmoid(c_buf[0]) * (A[row] . x).
//
// Geometry ported from gemv_hfq4g256_residual_sigmoid_scaled_gpu (the
// two-row-per-wave mq4r twin, 142 us/step vs 492 us/step for the one-row
// gemv_mq4g256v2_residual_sigmoid_scaled_k512):
//   * one wave32 owns HIPFIRE_MQ4G256V2_K512_ROWS (=2) consecutive rows and
//     loads each x slice once for all of them (grid M / ROWS);
//   * the sigmoid gate and the residual rows are fetched up front, so the
//     only work after the reduction is the single fma and the store.
//
// BYTE-EXACT with gemv_mq4g256v2_residual_sigmoid_scaled_k512 (same dual-half
// f16 header decode, same acc0/acc1 group slots, same shfl_down tree, same
// precise expf/divide gate). The compiler-contracted FMA orientation of the
// original is pinned explicitly (verified against its gfx1201 ISA):
//   d_i = fma(scale, q_i, zero); s = d1*x1 (rounded); s = fma(d0,x0,s);
//   s = fma(d2,x2,s) ... fma(d7,x7,s); acc_g = 0 + s;
//   acc = acc0 + acc1; y = fma(gate, acc, y).
//
// Launch: grid M / ROWS, block 32, no LDS. Requires K == 512, M % ROWS == 0.

#ifndef HIPFIRE_MQ4G256V2_K512_ROWS
#define HIPFIRE_MQ4G256V2_K512_ROWS 2
#endif
#ifndef HIPFIRE_MQ4G256V2_K512_R2_KERNEL
#define HIPFIRE_MQ4G256V2_K512_R2_KERNEL gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2
#endif

__launch_bounds__(32, 16)
extern "C" __global__ void HIPFIRE_MQ4G256V2_K512_R2_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ c_buf,
    int M, int K
) {
    constexpr int ROWS = HIPFIRE_MQ4G256V2_K512_ROWS;
    const int row0 = blockIdx.x * ROWS;
    if (row0 >= M) return;
    const int tid = threadIdx.x;
    const int boff = tid * 4;

    // Both 256-wide groups of x for this lane's 8-element slices.
    const float4 xa0 = *reinterpret_cast<const float4*>(x + tid * 8);
    const float4 xb0 = *reinterpret_cast<const float4*>(x + tid * 8 + 4);
    const float4 xa1 = *reinterpret_cast<const float4*>(x + 256 + tid * 8);
    const float4 xb1 = *reinterpret_cast<const float4*>(x + 256 + tid * 8 + 4);
    const float xs0[8] = {xa0.x, xa0.y, xa0.z, xa0.w, xb0.x, xb0.y, xb0.z, xb0.w};
    const float xs1[8] = {xa1.x, xa1.y, xa1.z, xa1.w, xb1.x, xb1.y, xb1.z, xb1.w};

    float yv[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; r++) yv[r] = y[row0 + r];
    const float gate = 1.0f / (1.0f + expf(-c_buf[0]));

    float res[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const char* rp = A + (long long)(row0 + r) * 272;
        const unsigned int hA0 = *reinterpret_cast<const unsigned int*>(rp);
        const unsigned int hB0 = *reinterpret_cast<const unsigned int*>(rp + 4);
        const unsigned int hA1 = *reinterpret_cast<const unsigned int*>(rp + 136);
        const unsigned int hB1 = *reinterpret_cast<const unsigned int*>(rp + 140);
        const unsigned int pk0 = *reinterpret_cast<const unsigned int*>(rp + 8 + boff);
        const unsigned int pk1 = *reinterpret_cast<const unsigned int*>(rp + 144 + boff);
        const unsigned int hs0 = (tid < 16) ? hA0 : hB0;
        const unsigned int hs1 = (tid < 16) ? hA1 : hB1;

        float acc0 = 0.0f, acc1 = 0.0f;
#define MQ4V2_K512_GROUP(hs, pk, xs, a) do { \
            const float sc = __half2float(__ushort_as_half((unsigned short)((hs) & 0xFFFFu))); \
            const float zp = __half2float(__ushort_as_half((unsigned short)((hs) >> 16))); \
            float d[8]; \
            _Pragma("unroll") \
            for (int i = 0; i < 8; i++) \
                d[i] = __fmaf_rn(sc, (float)(((pk) >> (4 * i)) & 0xFu), zp); \
            float s = __fmul_rn(d[1], (xs)[1]); \
            s = __fmaf_rn(d[0], (xs)[0], s); \
            _Pragma("unroll") \
            for (int i = 2; i < 8; i++) s = __fmaf_rn(d[i], (xs)[i], s); \
            (a) += s; \
        } while (0)
        MQ4V2_K512_GROUP(hs0, pk0, xs0, acc0);
        MQ4V2_K512_GROUP(hs1, pk1, xs1, acc1);
#undef MQ4V2_K512_GROUP

        float acc = acc0 + acc1;
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        res[r] = __fmaf_rn(gate, acc, yv[r]);
    }
    if (tid == 0) {
#pragma unroll
        for (int r = 0; r < ROWS; r++) y[row0 + r] = res[r];
    }
}
