// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_MQ4G256V2_XBATCH_KERNEL
#define HIPFIRE_MQ4G256V2_XBATCH_KERNEL gemv_mq4g256v2_xbatch
#endif

// ─────────────────────────────────────────────────────────────────────────
// x-batched variant: one weight row loaded ONCE, dotted against B separate
// activation vectors.
//
// Why: the multi-slot decode path used to run one GEMV launch per slot, so
// the 270 MB lm_head weight matrix (M=248320, K=2048, HFQ4-G256) was read B
// times per decode step. Every slot wants the same weights against a
// different x, so the reads collapse to one pass. This is the last
// per-slot-serialised op in the multi-slot forward; everything else is
// already a single launch across all rows.
//
// Bit-exactness is the binding constraint, not speed: the multi-slot path is
// gated by a golden test demanding 0.000x divergence from the reference's
// per-token GEMV. So each acc[b] here accumulates the SAME terms in the SAME
// order as gemv_mq4g256v2.hip's scalar kernel -- the only change is which x
// vector is indexed. The weight decode (sc/zp/pk) is hoisted out of the b loop, which
// is a pure CSE of identical values, not a reassociation.
//
// x is [B x K] row-major, y is [B x M] row-major, so y[b*M + row] lands
// exactly where a caller holding a [n_slots x vocab] logits block wants it.
//
// Its own source (gemv_mq4g256v2.hip's modules are unaffected): the kernel
// resource gate fingerprints those, and this must not perturb their
// VGPR/occupancy numbers. HIPFIRE_MQ4G256V2_RESIDUAL_EPILOGUE selects
// `y[b] += acc` (symbol named by HIPFIRE_MQ4G256V2_XBATCH_KERNEL).

#ifndef HIPFIRE_MQ4G256V2_XBATCH_MAX
#define HIPFIRE_MQ4G256V2_XBATCH_MAX 4
#endif

// Occupancy 8, not the scalar kernel's 16: B accumulator sets multiply the
// live float state by B. Asking for 16 waves/SIMD at B=4 makes the compiler
// spill, and a spilled accumulator is exactly what the scalar kernel's
// comment warns changes the FMA sequence.
__launch_bounds__(32, 8)
extern "C" __global__ void HIPFIRE_MQ4G256V2_XBATCH_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,   // [B x K]
    float* __restrict__ y,         // [B x M]
    int M, int K, int B
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

#if defined(HIPFIRE_MQ4G256V2_K2048)
    const int groups_per_row = 8;
#else
    const int groups_per_row = K / 256;
#endif
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    float acc0[HIPFIRE_MQ4G256V2_XBATCH_MAX];
    float acc1[HIPFIRE_MQ4G256V2_XBATCH_MAX];
    float acc2[HIPFIRE_MQ4G256V2_XBATCH_MAX];
    float acc3[HIPFIRE_MQ4G256V2_XBATCH_MAX];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_MQ4G256V2_XBATCH_MAX; b++) {
        acc0[b] = 0.0f; acc1[b] = 0.0f; acc2[b] = 0.0f; acc3[b] = 0.0f;
    }

    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        // Plain pointer loads: the buffer-load / cache-policy paths the
        // scalar kernel selects between are tuned for one x vector per pass,
        // and this kernel's whole point is that the weight pass is amortised
        // over B. Kept simple and identical across arches on purpose.
        const unsigned int hA0 = *(const unsigned int*)(gp0);
        const unsigned int hB0 = *(const unsigned int*)(gp0 + 4);
        const unsigned int hs0 = (tid < 16) ? hA0 : hB0;
        float sc0 = __half2float(__ushort_as_half((unsigned short)(hs0 & 0xFFFFu)));
        float zp0 = __half2float(__ushort_as_half((unsigned short)(hs0 >> 16)));
        const unsigned int hA1 = *(const unsigned int*)(gp1);
        const unsigned int hB1 = *(const unsigned int*)(gp1 + 4);
        const unsigned int hs1 = (tid < 16) ? hA1 : hB1;
        float sc1 = __half2float(__ushort_as_half((unsigned short)(hs1 & 0xFFFFu)));
        float zp1 = __half2float(__ushort_as_half((unsigned short)(hs1 >> 16)));
        const unsigned int hA2 = *(const unsigned int*)(gp2);
        const unsigned int hB2 = *(const unsigned int*)(gp2 + 4);
        const unsigned int hs2 = (tid < 16) ? hA2 : hB2;
        float sc2 = __half2float(__ushort_as_half((unsigned short)(hs2 & 0xFFFFu)));
        float zp2 = __half2float(__ushort_as_half((unsigned short)(hs2 >> 16)));
        const unsigned int hA3 = *(const unsigned int*)(gp3);
        const unsigned int hB3 = *(const unsigned int*)(gp3 + 4);
        const unsigned int hs3 = (tid < 16) ? hA3 : hB3;
        float sc3 = __half2float(__ushort_as_half((unsigned short)(hs3 & 0xFFFFu)));
        float zp3 = __half2float(__ushort_as_half((unsigned short)(hs3 >> 16)));

        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
        unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

        const int base = g * 256 + tid * 8;

        #define XDOG(pk, sc, zp, bs, a) \
            (a) += (sc * (float)((pk) & 0xFu)         + zp) * xb[(bs)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * xb[(bs) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * xb[(bs) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * xb[(bs) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * xb[(bs) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * xb[(bs) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * xb[(bs) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * xb[(bs) + 7]

        #pragma unroll
        for (int b = 0; b < HIPFIRE_MQ4G256V2_XBATCH_MAX; b++) {
            if (b >= B) break;
            const float* xb = x + (long long)b * K;
            XDOG(pk0, sc0, zp0, base,       acc0[b]);
            XDOG(pk1, sc1, zp1, base + 256, acc1[b]);
            XDOG(pk2, sc2, zp2, base + 512, acc2[b]);
            XDOG(pk3, sc3, zp3, base + 768, acc3[b]);
        }
        #undef XDOG
    }

    // Tail groups accumulate into acc[g % 4], matching the scalar kernel's
    // 4-way interleaving that the combine below assumes.
    #define XTAIL(pk, sc, zp, bs, a) \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * xb[(bs)]     \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * xb[(bs) + 1] \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * xb[(bs) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * xb[(bs) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * xb[(bs) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * xb[(bs) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * xb[(bs) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * xb[(bs) + 7]

    for (int t = 0; t < tail; t++) {
        const int g = (quads << 2) + t;
        const char* gp = row_ptr + g * 136;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        const int base = g * 256 + tid * 8;
        #pragma unroll
        for (int b = 0; b < HIPFIRE_MQ4G256V2_XBATCH_MAX; b++) {
            if (b >= B) break;
            const float* xb = x + (long long)b * K;
            if (t == 0)      { XTAIL(pk, sc, zp, base, acc0[b]); }
            else if (t == 1) { XTAIL(pk, sc, zp, base, acc1[b]); }
            else             { XTAIL(pk, sc, zp, base, acc2[b]); }
        }
    }
    #undef XTAIL

    #pragma unroll
    for (int b = 0; b < HIPFIRE_MQ4G256V2_XBATCH_MAX; b++) {
        if (b >= B) break;
        float acc = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
#if defined(HIPFIRE_MQ4G256V2_RESIDUAL_EPILOGUE)
        if (tid == 0) y[(long long)b * M + row] += acc;
#else
        if (tid == 0) y[(long long)b * M + row] = acc;
#endif
    }
}
