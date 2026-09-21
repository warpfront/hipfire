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

#ifndef HIPFIRE_FUSED_GATE_UP_KERNEL
#define HIPFIRE_FUSED_GATE_UP_KERNEL fused_gate_up_mq4g256v2
#endif

#ifndef HIPFIRE_MQ4V2_WPRESHUFFLE
#define HIPFIRE_MQ4V2_WPRESHUFFLE 0
#endif

#if HIPFIRE_MQ4V2_WPRESHUFFLE
__device__ __forceinline__ unsigned int mq4v2_wp_header(
    const char* a, int rows, int k, int row, int half_block) {
    const unsigned char* p = (const unsigned char*)a
        + (size_t)rows * (size_t)k / 2u
        + (((size_t)half_block * (size_t)(rows >> 4)
            + (unsigned)(row >> 4)) * 4u) * 16u
        + (unsigned)(row & 15);
    return (unsigned int)p[16] | ((unsigned int)p[0] << 8)
         | ((unsigned int)p[32] << 16) | ((unsigned int)p[48] << 24);
}

__device__ __forceinline__ unsigned int mq4v2_wp_packed(
    const char* a, int k, int row, int group, int tid) {
    const int kstep = group * 16 + (tid >> 1);
    const int lane = (tid & 1) * 16 + (row & 15);
    return *(const unsigned int*)(a
        + (((size_t)(row >> 4) * (size_t)(k >> 4)
            + (unsigned)kstep) * 32u + (unsigned)lane) * 4u);
}
#endif

// Fused Gate+Up HFQ4-G256: two GEMVs in one launch (saves 1 launch per
// FFN layer vs separate gate/up calls). Grid = [gate_m + up_m] — each
// block processes one row from either A_gate or A_up based on its index.
//
// Per-row math is identical to gemv_hfq4g256.hip: 4-accumulator
// group-interleaved summation with pairwise combine for cross-arch
// byte-exact output against the MQ4 quality-gate baselines. See
// gemv_hfq4g256.hip and gemv_hfq4g256.gfx1100.hip for the full writeup
// on the FP-associativity drift that motivates this structure.
//
// Launch bounds are (32, 16) to match the RDNA3 variant's register
// budget and keep codegen similar across archs.

__launch_bounds__(32, 16)
extern "C" __global__ void HIPFIRE_FUSED_GATE_UP_KERNEL(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int gate_m, int up_m, int K
#ifdef HIPFIRE_MQ4G256V2_LUT
    // Lloyd (qt=52): per-tensor centered f16 codebooks, 8 dwords each
    // (gate then up); the block picks one by row source below.
    , const unsigned int G0, const unsigned int G1, const unsigned int G2, const unsigned int G3,
    const unsigned int G4, const unsigned int G5, const unsigned int G6, const unsigned int G7,
    const unsigned int U0, const unsigned int U1, const unsigned int U2, const unsigned int U3,
    const unsigned int U4, const unsigned int U5, const unsigned int U6, const unsigned int U7
#endif
) {
    const int gid = blockIdx.x;
    const int tid = threadIdx.x;

    const char* A;
    float* y;
    int local_row;
    if (gid < gate_m) {
        A = A_gate; y = y_gate; local_row = gid;
    } else {
        A = A_up; y = y_up; local_row = gid - gate_m;
    }
#ifdef HIPFIRE_MQ4G256V2_LUT
    // Stage this row-source's 16 f16 levels to LDS (lanes 0..7, one dword
    // each), then decode nibbles through them. See gemv_mq4g256v2.hip.
    __shared__ unsigned short LUT_SMEM[16];
    {
        const bool is_gate = gid < gate_m;
        unsigned int d;
        if (tid == 0) d = is_gate ? G0 : U0; else if (tid == 1) d = is_gate ? G1 : U1;
        else if (tid == 2) d = is_gate ? G2 : U2; else if (tid == 3) d = is_gate ? G3 : U3;
        else if (tid == 4) d = is_gate ? G4 : U4; else if (tid == 5) d = is_gate ? G5 : U5;
        else if (tid == 6) d = is_gate ? G6 : U6; else d = is_gate ? G7 : U7;
        if (tid < 8) ((unsigned int*)LUT_SMEM)[tid] = d;
    }
    __syncthreads();
    #define MQ4V2_DEQ(q) (__half2float(__ushort_as_half(LUT_SMEM[(q)])))
#else
    #define MQ4V2_DEQ(q) ((float)(q))
#endif

#if defined(HIPFIRE_FUSED_GATE_UP_K1024)
    const int groups_per_row = 4;
#elif defined(HIPFIRE_GLIMMER_GATE_UP_K6656)
    // Muse Glimmer AR decode fused gate+up path: exact host-enforced shape is
    // gate_m=up_m=19968, K=6656 (26 groups, quads=6, tail=2). Arithmetic
    // (accumulator ownership, FMA order, tail placement, reduction tree)
    // stays identical to the generic template.
    const int groups_per_row = 26;
#else
    const int groups_per_row = K / 256;
#endif
    const char* row_ptr = A + (long long)local_row * groups_per_row * 136;
    const int boff = tid * 4;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto row_rsrc = HIPFIRE_WEIGHT_RSRC(row_ptr);
#endif

#if HIPFIRE_MQ4V2_WPRESHUFFLE
    const int matrix_rows = gid < gate_m ? gate_m : up_m;
    #define LOAD_W(G, HS, PK) do { \
        (HS) = mq4v2_wp_header(A, matrix_rows, K, local_row, \
                               2 * (G) + (tid >> 4)); \
        (PK) = mq4v2_wp_packed(A, K, local_row, (G), tid); \
    } while (0)
#else
    #define LOAD_W(G, HS, PK) do { \
        const char* _gp = row_ptr + (G) * 136; \
        const unsigned int _ha = HIPFIRE_WEIGHT_LOAD_PTR_U32( \
            row_rsrc, _gp, (G) * 136); \
        const unsigned int _hb = HIPFIRE_WEIGHT_LOAD_PTR_U32( \
            row_rsrc, _gp + 4, (G) * 136 + 4); \
        (HS) = (tid < 16) ? _ha : _hb; \
        (PK) = HIPFIRE_WEIGHT_LOAD_PTR_U32( \
            row_rsrc, _gp + 8 + boff, (G) * 136 + 8 + boff); \
    } while (0)
#endif

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        unsigned int hs_sc0, hs_sc1, hs_sc2, hs_sc3;
        unsigned int pk0, pk1, pk2, pk3;
        LOAD_W(g + 0, hs_sc0, pk0);
        LOAD_W(g + 1, hs_sc1, pk1);
        LOAD_W(g + 2, hs_sc2, pk2);
        LOAD_W(g + 3, hs_sc3, pk3);
        float sc0 = __half2float(__ushort_as_half((unsigned short)(hs_sc0 & 0xFFFFu)));
        float zp0 = __half2float(__ushort_as_half((unsigned short)(hs_sc0 >> 16)));
        float sc1 = __half2float(__ushort_as_half((unsigned short)(hs_sc1 & 0xFFFFu)));
        float zp1 = __half2float(__ushort_as_half((unsigned short)(hs_sc1 >> 16)));
        float sc2 = __half2float(__ushort_as_half((unsigned short)(hs_sc2 & 0xFFFFu)));
        float zp2 = __half2float(__ushort_as_half((unsigned short)(hs_sc2 >> 16)));
        float sc3 = __half2float(__ushort_as_half((unsigned short)(hs_sc3 & 0xFFFFu)));
        float zp3 = __half2float(__ushort_as_half((unsigned short)(hs_sc3 >> 16)));

        int base = g * 256 + tid * 8;

        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * MQ4V2_DEQ((pk) & 0xFu)        + zp) * x[(b)]     \
                 + (sc * MQ4V2_DEQ(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                 + (sc * MQ4V2_DEQ(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                 + (sc * MQ4V2_DEQ(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                 + (sc * MQ4V2_DEQ(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                 + (sc * MQ4V2_DEQ(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                 + (sc * MQ4V2_DEQ(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                 + (sc * MQ4V2_DEQ(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

        DOG(pk0, sc0, zp0, base, acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
    }

    // Tail groups must accumulate into acc[g % 4] — see
    // gemv_hfq4g256.gfx1100.hip for the full bug explanation.
    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * MQ4V2_DEQ((pk) & 0xFu)        + zp) * x[(b)]     \
             + (sc * MQ4V2_DEQ(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
             + (sc * MQ4V2_DEQ(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
             + (sc * MQ4V2_DEQ(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
             + (sc * MQ4V2_DEQ(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
             + (sc * MQ4V2_DEQ(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
             + (sc * MQ4V2_DEQ(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
             + (sc * MQ4V2_DEQ(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        unsigned int hs_sc, pk;
        LOAD_W(g, hs_sc, pk);
        float sc = __half2float(__ushort_as_half((unsigned short)(hs_sc & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs_sc >> 16)));
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        unsigned int hs_sc, pk;
        LOAD_W(g, hs_sc, pk);
        float sc = __half2float(__ushort_as_half((unsigned short)(hs_sc & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs_sc >> 16)));
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        unsigned int hs_sc, pk;
        LOAD_W(g, hs_sc, pk);
        float sc = __half2float(__ushort_as_half((unsigned short)(hs_sc & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs_sc >> 16)));
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG
    #undef LOAD_W
    #undef MQ4V2_DEQ

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[local_row] = acc;
}
