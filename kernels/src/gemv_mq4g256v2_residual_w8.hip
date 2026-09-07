// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_WEIGHT_LOAD_PTR_U32
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#define HIPFIRE_WEIGHT_RSRC(base) 0
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, byte_offset) \
    (*(const unsigned int*)(ptr))
#endif

#ifndef HIPFIRE_RESIDUAL_W8_KERNEL
#define HIPFIRE_RESIDUAL_W8_KERNEL gemv_mq4g256v2_residual_w8
#endif

// HFQ4-G256 GEMV with fused residual add: y[row] += A[row] · x.
//
// VARIANT res_w8: same arithmetic as gemv_mq4g256v2_residual (same DOG_X8
// expression, same group -> acc[g % 4] mapping, same combine and shfl tree)
// but the work is split by ACCUMULATOR, not by K range: 256 threads = 8
// waves; wave (r, k) (r = row 0/1, k = acc slot 0..3) iterates g = k, k+4,
// ... over the groups of its row, so each wave-lane holds a single float
// accumulator. The 8 per-lane partials meet in LDS and are combined by one
// wave per row exactly like the baseline. Value-preserving by construction.
__launch_bounds__(256, 2)
extern "C" __global__ void HIPFIRE_RESIDUAL_W8_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row0 = blockIdx.x << 1;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const bool have_row1 = row1 < M;
    const int tid = threadIdx.x;
    const int w = tid >> 5;
    const int r = w >> 2;
    const int k = w & 3;
    const int lane = tid & 31;

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_ptr0 = A + (long long)row0 * row_stride;
    const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * row_stride;
    const char* row_ptr = (r == 0) ? row_ptr0 : row_ptr1;
    const int boff = lane * 4;

    __shared__ float part[2][4][32];
    __shared__ float row1red;

    float acc = 0.0f;

    #define DOG_X8(sc, zp, pk, a, x0, x1, x2, x3, x4, x5, x6, x7) do { \
        (a) += ((sc) * (float)((pk) & 0xFu)         + (zp)) * (x0) \
             + ((sc) * (float)(((pk) >> 4) & 0xFu)  + (zp)) * (x1) \
             + ((sc) * (float)(((pk) >> 8) & 0xFu)  + (zp)) * (x2) \
             + ((sc) * (float)(((pk) >> 12) & 0xFu) + (zp)) * (x3) \
             + ((sc) * (float)(((pk) >> 16) & 0xFu) + (zp)) * (x4) \
             + ((sc) * (float)(((pk) >> 20) & 0xFu) + (zp)) * (x5) \
             + ((sc) * (float)(((pk) >> 24) & 0xFu) + (zp)) * (x6) \
             + ((sc) * (float)(((pk) >> 28) & 0xFu) + (zp)) * (x7); \
    } while (0)

    #define LOAD_W(gp, sc, zp, pk) do { \
        const unsigned int hA = HIPFIRE_WEIGHT_LOAD_PTR_U32(0, gp, 0); \
        const unsigned int hB = HIPFIRE_WEIGHT_LOAD_PTR_U32(0, (gp) + 4, 4); \
        const unsigned int hs = (lane < 16) ? hA : hB; \
        (sc) = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu))); \
        (zp) = __half2float(__ushort_as_half((unsigned short)(hs >> 16))); \
        (pk) = HIPFIRE_WEIGHT_LOAD_PTR_U32(0, (gp) + 8 + boff, 8 + boff); \
    } while (0)

    for (int g = k; g < groups_per_row; g += 4) {
        const int base = g * 256 + lane * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float sc, zp;
        unsigned int pk;
        LOAD_W(row_ptr + g * 136, sc, zp, pk);
        DOG_X8(sc, zp, pk, acc,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    #undef LOAD_W
    #undef DOG_X8

    part[r][k][lane] = acc;
    __syncthreads();

    float red = 0.0f;
    if (w == 0 || w == 4) {
        const int rr = w >> 2;
        red = (part[rr][0][lane] + part[rr][1][lane])
            + (part[rr][2][lane] + part[rr][3][lane]);
        for (int offset = 16; offset > 0; offset >>= 1)
            red += __shfl_down(red, offset);
        if (lane == 0 && rr == 1)
            row1red = red;
    }
    __syncthreads();

    if (tid == 0) {
        if (have_row1) {
            float2 yy = *reinterpret_cast<float2*>(y + row0);
            yy.x += red;
            yy.y += row1red;
            *reinterpret_cast<float2*>(y + row0) = yy;
        } else {
            y[row0] += red;
        }
    }
}
