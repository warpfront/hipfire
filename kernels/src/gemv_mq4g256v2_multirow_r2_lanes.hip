// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Lane-batched twin of `gemv_mq4g256v2_multirow_r2` (the specialized R=2,
// branchless dual-row MQ4-G256-v2 GEMV of `gemv_mq4g256v2_multirow.hip`),
// used by the DFlash continuous-batching draft's lane-batched ctx fill / fc.
//
// `gemv_mq4g256v2_multirow_r2_lanes` computes `batch` independent GEMVs
// `y[z] = W . x[z]` over ONE shared weight matrix `A [M, K]`: grid.z selects the
// segment, `x` is `[batch, K]` (row z at `x + z*K`), `y` is `[batch, M]` (row z
// at `y + z*M`). Grid `[ceil(M/2), 1, batch]`, block 32. Every wave runs the
// SAME two weight rows with the SAME instruction sequence (4-way interleaved
// group accumulators, pairwise combine, shuffle-down 16..1) as the singleton
// R2 kernel — the body below is a verbatim copy of its active (non-gfx1151
// experiment) path, so a segment's output is byte-identical to a singleton
// launch on that segment's `x` row. This file is deliberately separate from
// `gemv_mq4g256v2_multirow.hip`: that source's sha256 is pinned by the
// accepted PM decode twin (`pm_decode_twins.rs`).
//
// Unlike a 2-row-batch GEMM this never rounds `x` to F16 and never reduces
// through a WMMA tile: it is NOT interchangeable with the overwrite WMMA
// class. Only 1-row original segments belong here.
//
// Source prefix (kernels.rs): `HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE` define +
// `gfx12_weight_cache_policy.inc`, the same prefix as the singleton module.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifndef HIPFIRE_WEIGHT_LOAD_PTR_U32
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#define HIPFIRE_WEIGHT_RSRC(base) 0
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, byte_offset) \
    (*(const unsigned int*)(ptr))
#endif

__device__ __forceinline__ void gemv_mq4g256v2_multirow_r2_lanes_body(
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

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    // Clamp OOB row1 to row0 so weight reads stay in-bounds; store is masked.
    const char* row_ptr0 = A + (long long)row0 * row_stride;
    const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * row_stride;
    const int boff = tid * 4;
    const unsigned int row_weight_offset0 = (unsigned int)((long long)row0 * row_stride);
    const unsigned int row_weight_offset1 =
        (unsigned int)((long long)(have_row1 ? row1 : row0) * row_stride);
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    // The full lm-head matrix fits the raw-buffer 32-bit offset range; one SRD
    // avoids extending the live descriptor set in this 94-VGPR kernel.
    const auto weight_rsrc = HIPFIRE_WEIGHT_RSRC(A);
#endif

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define LOAD_X8_INTO(b, pfx) \
        const float4 pfx##v0 = *reinterpret_cast<const float4*>(x + (b)); \
        const float4 pfx##v1 = *reinterpret_cast<const float4*>(x + (b) + 4); \
        const float pfx##0 = pfx##v0.x, pfx##1 = pfx##v0.y, pfx##2 = pfx##v0.z, pfx##3 = pfx##v0.w; \
        const float pfx##4 = pfx##v1.x, pfx##5 = pfx##v1.y, pfx##6 = pfx##v1.z, pfx##7 = pfx##v1.w

    #define LOAD_LM_HEADER(gp, weight_offset) \
        HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, gp, weight_offset)
    #define LOAD_LM_PACKED(gp, weight_offset) \
        HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, gp, weight_offset)

    #define DOG_X8(gp, weight_offset, a, pfx) do { \
        const unsigned int hA = LOAD_LM_HEADER(gp, weight_offset); \
        const unsigned int hB = LOAD_LM_HEADER((gp) + 4, (weight_offset) + 4); \
        const unsigned int hs = (tid < 16) ? hA : hB; \
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu))); \
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16))); \
        unsigned int pk = LOAD_LM_PACKED((gp) + 8 + boff, (weight_offset) + 8 + boff); \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * pfx##0 \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * pfx##1 \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * pfx##2 \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * pfx##3 \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * pfx##4 \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * pfx##5 \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * pfx##6 \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * pfx##7; \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;

        // Hoist all 32 x values for this group-quad before any DOG.
        LOAD_X8_INTO(base0, xa);
        LOAD_X8_INTO(base0 + 256, xb);
        LOAD_X8_INTO(base0 + 512, xc);
        LOAD_X8_INTO(base0 + 768, xd);

        // Branchless dual-row DOG with all x resident (no mid-quad x reload).
        DOG_X8(row_ptr0 + g * 136, row_weight_offset0 + g * 136, a0, xa);
        DOG_X8(row_ptr1 + g * 136, row_weight_offset1 + g * 136, b0, xa);
        DOG_X8(row_ptr0 + (g + 1) * 136, row_weight_offset0 + (g + 1) * 136, a1, xb);
        DOG_X8(row_ptr1 + (g + 1) * 136, row_weight_offset1 + (g + 1) * 136, b1, xb);
        DOG_X8(row_ptr0 + (g + 2) * 136, row_weight_offset0 + (g + 2) * 136, a2, xc);
        DOG_X8(row_ptr1 + (g + 2) * 136, row_weight_offset1 + (g + 2) * 136, b2, xc);
        DOG_X8(row_ptr0 + (g + 3) * 136, row_weight_offset0 + (g + 3) * 136, a3, xd);
        DOG_X8(row_ptr1 + (g + 3) * 136, row_weight_offset1 + (g + 3) * 136, b3, xd);
    }

    // Tail uses sequential x (at most 3 groups; hoist buys little).
    #define LOAD_X8(b) \
        const float4 xv0 = *reinterpret_cast<const float4*>(x + (b)); \
        const float4 xv1 = *reinterpret_cast<const float4*>(x + (b) + 4); \
        const float x0 = xv0.x, x1 = xv0.y, x2 = xv0.z, x3 = xv0.w; \
        const float x4 = xv1.x, x5 = xv1.y, x6 = xv1.z, x7 = xv1.w

    #define DOG_X8_TAIL(gp, weight_offset, a) do { \
        const unsigned int hA = LOAD_LM_HEADER(gp, weight_offset); \
        const unsigned int hB = LOAD_LM_HEADER((gp) + 4, (weight_offset) + 4); \
        const unsigned int hs = (tid < 16) ? hA : hB; \
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu))); \
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16))); \
        unsigned int pk = LOAD_LM_PACKED((gp) + 8 + boff, (weight_offset) + 8 + boff); \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * x0 \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x1 \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x2 \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x3 \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x4 \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x5 \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x6 \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x7; \
    } while (0)

    if (tail >= 1) {
        const int g = quads << 2;
        const int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8_TAIL(row_ptr0 + g * 136, row_weight_offset0 + g * 136, a0);
        DOG_X8_TAIL(row_ptr1 + g * 136, row_weight_offset1 + g * 136, b0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8_TAIL(row_ptr0 + g * 136, row_weight_offset0 + g * 136, a1);
        DOG_X8_TAIL(row_ptr1 + g * 136, row_weight_offset1 + g * 136, b1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8_TAIL(row_ptr0 + g * 136, row_weight_offset0 + g * 136, a2);
        DOG_X8_TAIL(row_ptr1 + g * 136, row_weight_offset1 + g * 136, b2);
    }
    #undef DOG_X8_TAIL
    #undef LOAD_X8
    #undef DOG_X8
    #undef LOAD_X8_INTO
    #undef LOAD_LM_PACKED
    #undef LOAD_LM_HEADER

    float s0 = (a0 + a1) + (a2 + a3);
    float s1 = (b0 + b1) + (b2 + b3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        s0 += __shfl_down(s0, offset);
        s1 += __shfl_down(s1, offset);
    }

    if (tid == 0) {
        y[row0] = s0;
        if (have_row1) y[row1] = s1;
    }
}

// Same launch bounds as the singleton `gemv_mq4g256v2_multirow_r2`.
__launch_bounds__(32, 18)
extern "C" __global__ void gemv_mq4g256v2_multirow_r2_lanes(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    // blockIdx.z = segment: x row z at z*K floats, y row z at z*M floats
    // (64-bit offsets; `batch*K` / `batch*M` may exceed 32 bits in principle).
    const unsigned long long z = (unsigned long long)blockIdx.z;
    gemv_mq4g256v2_multirow_r2_lanes_body(
        A,
        x + z * (unsigned long long)(unsigned int)K,
        y + z * (unsigned long long)(unsigned int)M,
        M, K);
}
