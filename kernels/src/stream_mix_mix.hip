// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Stream postmix: residual mixing plus branch write-back, into a SEPARATE
// output buffer (ping-pong; the old streams are read by every output stream).
//
//   out[i, c] = round(sum_j res[i, j] * x[j, c] + post[i] * branch[c])
//
// res is [n, n] destination/source. Every control comes from the OLD streams
// (computed before the branch ran). `round` is bfloat16 round-to-nearest-even
// when `bf16 != 0`. Supported n <= 8.
//
// Grid:  [ceil(width / 256), rows, 1]
// Block: [256, 1, 1]

#include <hip/hip_runtime.h>

#define SM_THREADS 256
#define SM_MAX_N 8

__device__ __forceinline__ float sm_round_bf16(float x) {
    unsigned int u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

extern "C" __launch_bounds__(SM_THREADS)
__global__ void stream_mix_postmix(
    const float* __restrict__ x,       // [rows, n, width] old streams
    const float* __restrict__ branch,  // [rows, width]
    const float* __restrict__ post,    // [rows, n]
    const float* __restrict__ res,     // [rows, n, n]
    float*       __restrict__ out,     // [rows, n, width]
    int n,
    int width,
    int bf16
) {
    const int c = blockIdx.x * SM_THREADS + threadIdx.x;
    const int row = blockIdx.y;
    if (c >= width) return;

    const float* xr = x + (size_t)row * n * width + c;
    float xv[SM_MAX_N];
#pragma unroll
    for (int j = 0; j < SM_MAX_N; ++j) xv[j] = (j < n) ? xr[(size_t)j * width] : 0.0f;
    const float bv = branch[(size_t)row * width + c];
    const float* rr = res + (size_t)row * n * n;
    const float* pr = post + (size_t)row * n;
    float* o = out + (size_t)row * n * width + c;

#pragma unroll
    for (int i = 0; i < SM_MAX_N; ++i) {
        if (i < n) {
            float a = 0.0f;
#pragma unroll
            for (int j = 0; j < SM_MAX_N; ++j)
                if (j < n) a += rr[i * n + j] * xv[j];
            a += pr[i] * bv;
            if (bf16) a = sm_round_bf16(a);
            o[(size_t)i * width] = a;
        }
    }
}

// BF16-stream postmix: the streams are BF16 bits (u16), read and written at
// half the F32-plane width. Every value a stage writes is already rounded to
// bfloat16 (the F32 plane only carries those values), so the narrow store is
// exact and the widening on load is exact: out is bit-for-bit the BF16 image of
// stream_mix_postmix's F32 output on the widened inputs. Requires bf16 != 0 and
// even width; each thread owns two adjacent columns (one 32-bit access per
// stream).
//
// Grid:  [ceil(width / 512), rows, 1]
// Block: [256, 1, 1]
__device__ __forceinline__ unsigned short sm_bf16_bits(float rounded) {
    unsigned int u = __float_as_uint(rounded);
    unsigned int h = u >> 16;
    // A NaN keeps its payload through sm_round_bf16; keep it a NaN (quiet bit).
    if ((u & 0x7fffffffu) > 0x7f800000u) h |= 0x0040u;
    return (unsigned short)h;
}

extern "C" __launch_bounds__(SM_THREADS)
__global__ void stream_mix_postmix_bf16(
    const unsigned short* __restrict__ x,  // [rows, n, width] old streams (BF16 bits)
    const float* __restrict__ branch,      // [rows, width]
    const float* __restrict__ post,        // [rows, n]
    const float* __restrict__ res,         // [rows, n, n]
    unsigned short* __restrict__ out,      // [rows, n, width] (BF16 bits)
    int n,
    int width,
    int bf16_unused
) {
    (void)bf16_unused;
    const int c = (blockIdx.x * SM_THREADS + threadIdx.x) * 2;
    const int row = blockIdx.y;
    if (c >= width) return;

    const unsigned short* xr = x + (size_t)row * n * width + c;
    float xv0[SM_MAX_N], xv1[SM_MAX_N];
#pragma unroll
    for (int j = 0; j < SM_MAX_N; ++j) {
        if (j < n) {
            const unsigned int w = *(const unsigned int*)(xr + (size_t)j * width);
            xv0[j] = __uint_as_float(w << 16);
            xv1[j] = __uint_as_float(w & 0xffff0000u);
        } else {
            xv0[j] = 0.0f;
            xv1[j] = 0.0f;
        }
    }
    const float2 bv = *(const float2*)(branch + (size_t)row * width + c);
    const float* rr = res + (size_t)row * n * n;
    const float* pr = post + (size_t)row * n;
    unsigned short* o = out + (size_t)row * n * width + c;

#pragma unroll
    for (int i = 0; i < SM_MAX_N; ++i) {
        if (i < n) {
            float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
            for (int j = 0; j < SM_MAX_N; ++j) {
                if (j < n) {
                    a0 += rr[i * n + j] * xv0[j];
                    a1 += rr[i * n + j] * xv1[j];
                }
            }
            a0 += pr[i] * bv.x;
            a1 += pr[i] * bv.y;
            a0 = sm_round_bf16(a0);
            a1 = sm_round_bf16(a1);
            *(unsigned int*)(o + (size_t)i * width) =
                (unsigned int)sm_bf16_bits(a0) | ((unsigned int)sm_bf16_bits(a1) << 16);
        }
    }
}
