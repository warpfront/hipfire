// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_MQ6G256V2_KERNEL
#define HIPFIRE_MQ6G256V2_KERNEL gemv_mq6g256v2
#endif

#ifndef HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#endif

// MQ6-G256 v2 GEMV: neutral Magnum V2 (qt=47).
// Per-group 200 B: [0..2) fp16 s0, [2..4) fp16 z0, [4..6) fp16 s1, [6..8) fp16 z1,
// [8..200) 192 B legacy 6-bit payload (4 weights per 3 bytes, 4/3 B).
// Half 0 governs q[0..128), half 1 q[128..256); reconstruction q*f32(s[h])+f32(z[h]).
// K%256 required. Same 4-accumulator interleaving as gemv_mq4g256v2 for byte-exact.

__launch_bounds__(32, 16)
extern "C" __global__ void HIPFIRE_MQ6G256V2_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

#if defined(HIPFIRE_MQ6G256V2_K2048)
    const int groups_per_row = 8;
#else
    const int groups_per_row = K / 256;
#endif
    const char* row_ptr = A + (long long)row * groups_per_row * 200;
    const unsigned int row_weight_offset =
        (unsigned int)((long long)row * groups_per_row * 200);
    const int boff = tid * 6;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto weight_rsrc = HIPFIRE_WEIGHT_RSRC(A);
    #if defined(HIPFIRE_GFX1151_LM_HEAD_HYBRID_BUFFER)
        #define LOAD_WEIGHT_HEADER(ptr, offset) (*(const unsigned int*)(ptr))
    #else
        #define LOAD_WEIGHT_HEADER(ptr, offset) \
            HIPFIRE_WEIGHT_LOAD_U32(weight_rsrc, (offset))
    #endif
    #define LOAD_WEIGHT_PACKED_BYTE(ptr, offset) HIPFIRE_WEIGHT_LOAD_U8(weight_rsrc, (offset))
#else
    #define LOAD_WEIGHT_HEADER(ptr, offset) (*(const unsigned int*)(ptr))
    #define LOAD_WEIGHT_PACKED_BYTE(ptr, offset) (*(const unsigned char*)(ptr))
#endif

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 200;
        const char* gp1 = gp0 + 200;
        const char* gp2 = gp1 + 200;
        const char* gp3 = gp2 + 200;

        const unsigned int goff = row_weight_offset + (unsigned int)g * 200u;
        const unsigned int hA0 = LOAD_WEIGHT_HEADER(gp0, goff);
        const unsigned int hB0 = LOAD_WEIGHT_HEADER(gp0 + 4, goff + 4u);
        const unsigned int hs0 = (tid < 16) ? hA0 : hB0;
        float sc0 = __half2float(__ushort_as_half((unsigned short)(hs0 & 0xFFFFu)));
        float zp0 = __half2float(__ushort_as_half((unsigned short)(hs0 >> 16)));
        const unsigned int hA1 = LOAD_WEIGHT_HEADER(gp1, goff + 200u);
        const unsigned int hB1 = LOAD_WEIGHT_HEADER(gp1 + 4, goff + 204u);
        const unsigned int hs1 = (tid < 16) ? hA1 : hB1;
        float sc1 = __half2float(__ushort_as_half((unsigned short)(hs1 & 0xFFFFu)));
        float zp1 = __half2float(__ushort_as_half((unsigned short)(hs1 >> 16)));
        const unsigned int hA2 = LOAD_WEIGHT_HEADER(gp2, goff + 400u);
        const unsigned int hB2 = LOAD_WEIGHT_HEADER(gp2 + 4, goff + 404u);
        const unsigned int hs2 = (tid < 16) ? hA2 : hB2;
        float sc2 = __half2float(__ushort_as_half((unsigned short)(hs2 & 0xFFFFu)));
        float zp2 = __half2float(__ushort_as_half((unsigned short)(hs2 >> 16)));
        const unsigned int hA3 = LOAD_WEIGHT_HEADER(gp3, goff + 600u);
        const unsigned int hB3 = LOAD_WEIGHT_HEADER(gp3 + 4, goff + 604u);
        const unsigned int hs3 = (tid < 16) ? hA3 : hB3;
        float sc3 = __half2float(__ushort_as_half((unsigned short)(hs3 & 0xFFFFu)));
        float zp3 = __half2float(__ushort_as_half((unsigned short)(hs3 >> 16)));

        int base = g * 256 + tid * 8;

#define MQ6_DOG(gp, goff_base, sc, zp, b, a) do { \
        const unsigned char* d = (const unsigned char*)((gp) + 8 + boff); \
        unsigned char b0 = d[0]; unsigned char b1 = d[1]; unsigned char b2 = d[2]; \
        unsigned char b3 = d[3]; unsigned char b4 = d[4]; unsigned char b5 = d[5]; \
        int q0 = b0 & 63; \
        int q1 = (b0 >> 6) | ((b1 & 0xF) << 2); \
        int q2 = (b1 >> 4) | ((b2 & 3) << 4); \
        int q3 = b2 >> 2; \
        int q4 = b3 & 63; \
        int q5 = (b3 >> 6) | ((b4 & 0xF) << 2); \
        int q6 = (b4 >> 4) | ((b5 & 3) << 4); \
        int q7 = b5 >> 2; \
        (a) += (sc * (float)q0 + zp) * x[(b)] \
             + (sc * (float)q1 + zp) * x[(b) + 1] \
             + (sc * (float)q2 + zp) * x[(b) + 2] \
             + (sc * (float)q3 + zp) * x[(b) + 3] \
             + (sc * (float)q4 + zp) * x[(b) + 4] \
             + (sc * (float)q5 + zp) * x[(b) + 5] \
             + (sc * (float)q6 + zp) * x[(b) + 6] \
             + (sc * (float)q7 + zp) * x[(b) + 7]; \
    } while(0)

        MQ6_DOG(gp0, goff, sc0, zp0, base, acc0);
        MQ6_DOG(gp1, goff + 200u, sc1, zp1, base + 256, acc1);
        MQ6_DOG(gp2, goff + 400u, sc2, zp2, base + 512, acc2);
        MQ6_DOG(gp3, goff + 600u, sc3, zp3, base + 768, acc3);
#undef MQ6_DOG
    }

#define MQ6_TAIL_DOG(gp, goff_, sc, zp, b, a) do { \
        const unsigned char* d = (const unsigned char*)((gp) + 8 + boff); \
        unsigned char b0 = d[0]; unsigned char b1 = d[1]; unsigned char b2 = d[2]; \
        unsigned char b3 = d[3]; unsigned char b4 = d[4]; unsigned char b5 = d[5]; \
        int q0 = b0 & 63; \
        int q1 = (b0 >> 6) | ((b1 & 0xF) << 2); \
        int q2 = (b1 >> 4) | ((b2 & 3) << 4); \
        int q3 = b2 >> 2; \
        int q4 = b3 & 63; \
        int q5 = (b3 >> 6) | ((b4 & 0xF) << 2); \
        int q6 = (b4 >> 4) | ((b5 & 3) << 4); \
        int q7 = b5 >> 2; \
        (a) += (sc * (float)q0 + zp) * x[(b)] \
             + (sc * (float)q1 + zp) * x[(b) + 1] \
             + (sc * (float)q2 + zp) * x[(b) + 2] \
             + (sc * (float)q3 + zp) * x[(b) + 3] \
             + (sc * (float)q4 + zp) * x[(b) + 4] \
             + (sc * (float)q5 + zp) * x[(b) + 5] \
             + (sc * (float)q6 + zp) * x[(b) + 6] \
             + (sc * (float)q7 + zp) * x[(b) + 7]; \
    } while(0)

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 200;
        const unsigned int goff = row_weight_offset + (unsigned int)g * 200u;
        const unsigned int hA = LOAD_WEIGHT_HEADER(gp, goff);
        const unsigned int hB = LOAD_WEIGHT_HEADER(gp + 4, goff + 4u);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        int base = g * 256 + tid * 8;
        MQ6_TAIL_DOG(gp, goff, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 200;
        const unsigned int goff = row_weight_offset + (unsigned int)g * 200u;
        const unsigned int hA = LOAD_WEIGHT_HEADER(gp, goff);
        const unsigned int hB = LOAD_WEIGHT_HEADER(gp + 4, goff + 4u);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        int base = g * 256 + tid * 8;
        MQ6_TAIL_DOG(gp, goff, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 200;
        const unsigned int goff = row_weight_offset + (unsigned int)g * 200u;
        const unsigned int hA = LOAD_WEIGHT_HEADER(gp, goff);
        const unsigned int hB = LOAD_WEIGHT_HEADER(gp + 4, goff + 4u);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        int base = g * 256 + tid * 8;
        MQ6_TAIL_DOG(gp, goff, sc, zp, base, acc2);
    }
#undef MQ6_TAIL_DOG
#undef LOAD_WEIGHT_PACKED_BYTE
#undef LOAD_WEIGHT_HEADER

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}

// ─────────────────────────────────────────────────────────────────────────
// x-batched variant: one weight row decoded ONCE per launch and dotted
// against B activation rows (x is [B x K], y is [B x M], row-major). Each
// acc[b] accumulates the same terms in the same order as the scalar kernel
// above; only the x row changes. The batched path on GPUs without WMMA.
// Compiled only into its own module, so the scalar kernel's module is
// unaffected. HIPFIRE_MQ6G256V2_RESIDUAL_EPILOGUE selects y += acc.
#if defined(HIPFIRE_MQ6G256V2_XBATCH)

#ifndef HIPFIRE_MQ6G256V2_XBATCH_MAX
#define HIPFIRE_MQ6G256V2_XBATCH_MAX 4
#endif

// Occupancy 8, not 16: B accumulator sets multiply the live float state.
__launch_bounds__(32, 8)
extern "C" __global__ void HIPFIRE_MQ6G256V2_XBATCH_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,   // [B x K]
    float* __restrict__ y,         // [B x M]
    int M, int K, int B
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 200;
    const int boff = tid * 6;

    float acc0[HIPFIRE_MQ6G256V2_XBATCH_MAX];
    float acc1[HIPFIRE_MQ6G256V2_XBATCH_MAX];
    float acc2[HIPFIRE_MQ6G256V2_XBATCH_MAX];
    float acc3[HIPFIRE_MQ6G256V2_XBATCH_MAX];
    #pragma unroll
    for (int b = 0; b < HIPFIRE_MQ6G256V2_XBATCH_MAX; b++) {
        acc0[b] = 0.0f; acc1[b] = 0.0f; acc2[b] = 0.0f; acc3[b] = 0.0f;
    }

    // Decode one group's header and this lane's eight 6-bit values.
#define MQ6X_DECODE(gp, sc, zp, q) do { \
        const unsigned int hA = *(const unsigned int*)(gp); \
        const unsigned int hB = *(const unsigned int*)((gp) + 4); \
        const unsigned int hs = (tid < 16) ? hA : hB; \
        sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu))); \
        zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16))); \
        const unsigned char* d = (const unsigned char*)((gp) + 8 + boff); \
        unsigned char b0 = d[0]; unsigned char b1 = d[1]; unsigned char b2 = d[2]; \
        unsigned char b3 = d[3]; unsigned char b4 = d[4]; unsigned char b5 = d[5]; \
        q[0] = b0 & 63; \
        q[1] = (b0 >> 6) | ((b1 & 0xF) << 2); \
        q[2] = (b1 >> 4) | ((b2 & 3) << 4); \
        q[3] = b2 >> 2; \
        q[4] = b3 & 63; \
        q[5] = (b3 >> 6) | ((b4 & 0xF) << 2); \
        q[6] = (b4 >> 4) | ((b5 & 3) << 4); \
        q[7] = b5 >> 2; \
    } while (0)
    // Same expression and association as MQ6_DOG in the scalar kernel.
#define MQ6X_DOT(sc, zp, q, xb, bs, a) \
        (a) += (sc * (float)q[0] + zp) * xb[(bs)] \
             + (sc * (float)q[1] + zp) * xb[(bs) + 1] \
             + (sc * (float)q[2] + zp) * xb[(bs) + 2] \
             + (sc * (float)q[3] + zp) * xb[(bs) + 3] \
             + (sc * (float)q[4] + zp) * xb[(bs) + 4] \
             + (sc * (float)q[5] + zp) * xb[(bs) + 5] \
             + (sc * (float)q[6] + zp) * xb[(bs) + 6] \
             + (sc * (float)q[7] + zp) * xb[(bs) + 7]

    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int qd = 0; qd < quads; qd++) {
        const int g = qd << 2;
        const char* gp0 = row_ptr + g * 200;
        float sc0, zp0, sc1, zp1, sc2, zp2, sc3, zp3;
        int q0[8], q1[8], q2[8], q3[8];
        MQ6X_DECODE(gp0, sc0, zp0, q0);
        MQ6X_DECODE(gp0 + 200, sc1, zp1, q1);
        MQ6X_DECODE(gp0 + 400, sc2, zp2, q2);
        MQ6X_DECODE(gp0 + 600, sc3, zp3, q3);
        const int base = g * 256 + tid * 8;
        #pragma unroll
        for (int b = 0; b < HIPFIRE_MQ6G256V2_XBATCH_MAX; b++) {
            if (b >= B) break;
            const float* xb = x + (long long)b * K;
            MQ6X_DOT(sc0, zp0, q0, xb, base, acc0[b]);
            MQ6X_DOT(sc1, zp1, q1, xb, base + 256, acc1[b]);
            MQ6X_DOT(sc2, zp2, q2, xb, base + 512, acc2[b]);
            MQ6X_DOT(sc3, zp3, q3, xb, base + 768, acc3[b]);
        }
    }

    // Tail groups accumulate into acc[t], matching the scalar kernel.
    for (int t = 0; t < tail; t++) {
        const int g = (quads << 2) + t;
        float sc, zp;
        int q[8];
        MQ6X_DECODE(row_ptr + g * 200, sc, zp, q);
        const int base = g * 256 + tid * 8;
        #pragma unroll
        for (int b = 0; b < HIPFIRE_MQ6G256V2_XBATCH_MAX; b++) {
            if (b >= B) break;
            const float* xb = x + (long long)b * K;
            if (t == 0)      { MQ6X_DOT(sc, zp, q, xb, base, acc0[b]); }
            else if (t == 1) { MQ6X_DOT(sc, zp, q, xb, base, acc1[b]); }
            else             { MQ6X_DOT(sc, zp, q, xb, base, acc2[b]); }
        }
    }
#undef MQ6X_DOT
#undef MQ6X_DECODE

    #pragma unroll
    for (int b = 0; b < HIPFIRE_MQ6G256V2_XBATCH_MAX; b++) {
        if (b >= B) break;
        float acc = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
#if defined(HIPFIRE_MQ6G256V2_RESIDUAL_EPILOGUE)
        if (tid == 0) y[(long long)b * M + row] += acc;
#else
        if (tid == 0) y[(long long)b * M + row] = acc;
#endif
    }
}

#endif  // HIPFIRE_MQ6G256V2_XBATCH
