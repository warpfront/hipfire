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

#if defined(HIPFIRE_GFX1151_LM_HEAD_X_BUFFER)
using hipfire_lm_i32x4 = int __attribute__((ext_vector_type(4)));
using hipfire_lm_f32x4 = float __attribute__((ext_vector_type(4)));
#endif

// Fused three-matrix R=2 GEMV (gfx1151 decode a/b/z projections) for the MQ4-G256v2 format.
//
// MIRROR SOURCE: kernels/src/gemv_mq4g256v2_multirow.hip, kernel
// gemv_mq4g256v2_multirow_r2. The device body below is a verbatim copy of that
// kernel (same x hoist, DOG_X8 expression text, 4-way accumulators, tail, and
// shuffle reduction) with only two changes: the row pair is selected from an
// explicit matrix-local `block` parameter instead of blockIdx.x, and it is a
// __device__ function. Keep the two bodies in sync; the incumbent file is
// intentionally untouched (its source sha256 is pinned by pm_decode_twins.rs).
// Runtime source = "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1" +
// gfx12_weight_cache_policy.inc + this file, like GEMV_MQ4G256V2_MULTIROW_SRC.

// R13c1 lever (gfx1201): re-hoist full 32-wide x per group-quad on the
// specialized R=2 branchless body (R12 kept sequential float4×2 x loads).
// BOD: L2-resident/mem-busy occ~89% mem_busy~93% vgpr=96. R12 sequential
// x was meant to free VGPRs but profile kept vgpr=96 / SATURATED — so the
// live-x cut bought no occupancy and serializes each dual-row DOG pair
// behind a fresh x load. Hypothesis: hoist all four groups' x once
// (float4 packs), then fire eight branchless DOGs with all x resident —
// same math (4-way acc[g%4] + pairwise combine, 5302926) and same
// branchless dual-row clamp, but removes x-load → DOG serialization so
// dual-issue can cover more of the weight-load latency under L2-hit.
__device__ __forceinline__ void gemv_mq4g256v2_multirow_r2_body(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int block
) {
    const int row0 = block << 1;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const bool have_row1 = row1 < M;
    const int tid = threadIdx.x;

#if defined(HIPFIRE_GFX1151_LM_HEAD_K2048)
    const int groups_per_row = 8;
#else
    const int groups_per_row = K / 256;
#endif
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
#if defined(HIPFIRE_GFX1151_LM_HEAD_X_BUFFER)
    const auto x_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<float*>(x), 0, 0xffffffffu, 0x31004000u);
#endif

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

#if defined(HIPFIRE_GFX1151_LM_HEAD_X_BUFFER)
    #define LOAD_X8_INTO(b, pfx) \
        const hipfire_lm_f32x4 pfx##v0 = __builtin_bit_cast( \
            hipfire_lm_f32x4, __builtin_amdgcn_raw_buffer_load_b128( \
                x_rsrc, (unsigned int)(b) * 4u, 0, 0)); \
        const hipfire_lm_f32x4 pfx##v1 = __builtin_bit_cast( \
            hipfire_lm_f32x4, __builtin_amdgcn_raw_buffer_load_b128( \
                x_rsrc, ((unsigned int)(b) + 4u) * 4u, 0, 0)); \
        const float pfx##0 = pfx##v0.x, pfx##1 = pfx##v0.y, pfx##2 = pfx##v0.z, pfx##3 = pfx##v0.w; \
        const float pfx##4 = pfx##v1.x, pfx##5 = pfx##v1.y, pfx##6 = pfx##v1.z, pfx##7 = pfx##v1.w
#else
    #define LOAD_X8_INTO(b, pfx) \
        const float4 pfx##v0 = *reinterpret_cast<const float4*>(x + (b)); \
        const float4 pfx##v1 = *reinterpret_cast<const float4*>(x + (b) + 4); \
        const float pfx##0 = pfx##v0.x, pfx##1 = pfx##v0.y, pfx##2 = pfx##v0.z, pfx##3 = pfx##v0.w; \
        const float pfx##4 = pfx##v1.x, pfx##5 = pfx##v1.y, pfx##6 = pfx##v1.z, pfx##7 = pfx##v1.w
#endif

#if defined(HIPFIRE_GFX1151_LM_HEAD_HYBRID_BUFFER)
    // Uniform per-group headers are cheaper on the scalar/global path. Only
    // the lane-divergent packed dword uses temporal buffer VMEM.
    #define LOAD_LM_HEADER(gp, weight_offset) (*(const unsigned int*)(gp))
#else
    #define LOAD_LM_HEADER(gp, weight_offset) \
        HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, gp, weight_offset)
#endif
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
#if defined(HIPFIRE_GFX1151_LM_HEAD_X_BUFFER)
        asm volatile("" : :
            "v"(xav0), "v"(xav1), "v"(xbv0), "v"(xbv1),
            "v"(xcv0), "v"(xcv1), "v"(xdv0), "v"(xdv1) : "memory");
#endif

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

// Grid = ceil(M0/2) + ceil(M1/2) + ceil(M2/2) wave32 blocks. Each block belongs to
// one matrix and runs the r2 body with a matrix-local block index; x and K shared.
__launch_bounds__(32, 18)
extern "C" __global__ void gemv_mq4g256v2_multirow_r2_x3(
    const char* __restrict__ A0,
    float* __restrict__ y0,
    int M0,
    const char* __restrict__ A1,
    float* __restrict__ y1,
    int M1,
    const char* __restrict__ A2,
    float* __restrict__ y2,
    int M2,
    const float* __restrict__ x,
    int K
) {
    int block = blockIdx.x;
    const int blocks0 = (M0 + 1) >> 1;
    if (block < blocks0) {
        gemv_mq4g256v2_multirow_r2_body(A0, x, y0, M0, K, block);
        return;
    }
    block -= blocks0;
    const int blocks1 = (M1 + 1) >> 1;
    if (block < blocks1) {
        gemv_mq4g256v2_multirow_r2_body(A1, x, y1, M1, K, block);
        return;
    }
    block -= blocks1;
    gemv_mq4g256v2_multirow_r2_body(A2, x, y2, M2, K, block);
}
