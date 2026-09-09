// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Developer experiment: FP32 MQ4G256V2 set kernels for gfx1010 only; the
// residual oracle remains in gemm_mq4g256v2_residual_simt.hip unchanged.
//
// Four plain-overwrite arms (Y = dot, no residual read):
//   gemm_mq4g256v2_set_simt_r1t8_q4_gfx1010   — R1/T8 four-groups-live control
//   gemm_mq4g256v2_set_simt_r1t8_g1_gfx1010   — R1/T8 one-group-live schedule
//   gemm_mq4g256v2_set_simt_r1t16_g1_gfx1010  — R1/T16 one-group-live schedule
//   gemm_mq4g256v2_set_simt_r2t8_g1_gfx1010   — R2/T8 one-group-live, shared X
//
// Shared ABI: A*, X*, Y*, i32 M/K/N (raw 36, padded 48). Block 32, LDS 0,
// __launch_bounds__(32,4). Group is 136B (hA+0, hB+4, payload+8). Expression
// order, four-chain routing (g%4), pairwise combine, and shuffle tree match
// the residual oracle; only the epilogue store is set (Y=) not residual (Y+=).
// Bodies compile only when __gfx1010__ is defined (headers above stay live).

#if defined(__gfx1010__)

// ---------------------------------------------------------------------------
// SetR1T8Q4 — control: clone residual four-groups-live body; Y= only.
// Grid: [M, ceil(N/8), 1]
// ---------------------------------------------------------------------------

__launch_bounds__(32, 4)
extern "C" __global__ void gemm_mq4g256v2_set_simt_r1t8_q4_gfx1010(
    const char* __restrict__ A,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int BATCH_TILE = 8;
    const int row = blockIdx.x;
    if (row >= M) return;
    const int batch_start = blockIdx.y * BATCH_TILE;
    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    float acc0[8], acc1[8], acc2[8], acc3[8];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        acc0[b] = 0.0f;
        acc1[b] = 0.0f;
        acc2[b] = 0.0f;
        acc3[b] = 0.0f;
    }

    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;

    #define LOAD_V2_SC_ZP(gp, sc, zp) do {                                      \
        const unsigned int hA = *(const unsigned int*)(gp);                     \
        const unsigned int hB = *(const unsigned int*)((gp) + 4);               \
        const unsigned int hs = (tid < 16) ? hA : hB;                           \
        (sc) = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));  \
        (zp) = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));      \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        float sc0, zp0, sc1, zp1, sc2, zp2, sc3, zp3;
        LOAD_V2_SC_ZP(gp0, sc0, zp0);
        LOAD_V2_SC_ZP(gp1, sc1, zp1);
        LOAD_V2_SC_ZP(gp2, sc2, zp2);
        LOAD_V2_SC_ZP(gp3, sc3, zp3);

        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
        unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

        float w0_0 = sc0 * (float)((pk0       ) & 0xFu) + zp0;
        float w0_1 = sc0 * (float)((pk0 >>  4) & 0xFu) + zp0;
        float w0_2 = sc0 * (float)((pk0 >>  8) & 0xFu) + zp0;
        float w0_3 = sc0 * (float)((pk0 >> 12) & 0xFu) + zp0;
        float w0_4 = sc0 * (float)((pk0 >> 16) & 0xFu) + zp0;
        float w0_5 = sc0 * (float)((pk0 >> 20) & 0xFu) + zp0;
        float w0_6 = sc0 * (float)((pk0 >> 24) & 0xFu) + zp0;
        float w0_7 = sc0 * (float)((pk0 >> 28) & 0xFu) + zp0;

        float w1_0 = sc1 * (float)((pk1       ) & 0xFu) + zp1;
        float w1_1 = sc1 * (float)((pk1 >>  4) & 0xFu) + zp1;
        float w1_2 = sc1 * (float)((pk1 >>  8) & 0xFu) + zp1;
        float w1_3 = sc1 * (float)((pk1 >> 12) & 0xFu) + zp1;
        float w1_4 = sc1 * (float)((pk1 >> 16) & 0xFu) + zp1;
        float w1_5 = sc1 * (float)((pk1 >> 20) & 0xFu) + zp1;
        float w1_6 = sc1 * (float)((pk1 >> 24) & 0xFu) + zp1;
        float w1_7 = sc1 * (float)((pk1 >> 28) & 0xFu) + zp1;

        float w2_0 = sc2 * (float)((pk2       ) & 0xFu) + zp2;
        float w2_1 = sc2 * (float)((pk2 >>  4) & 0xFu) + zp2;
        float w2_2 = sc2 * (float)((pk2 >>  8) & 0xFu) + zp2;
        float w2_3 = sc2 * (float)((pk2 >> 12) & 0xFu) + zp2;
        float w2_4 = sc2 * (float)((pk2 >> 16) & 0xFu) + zp2;
        float w2_5 = sc2 * (float)((pk2 >> 20) & 0xFu) + zp2;
        float w2_6 = sc2 * (float)((pk2 >> 24) & 0xFu) + zp2;
        float w2_7 = sc2 * (float)((pk2 >> 28) & 0xFu) + zp2;

        float w3_0 = sc3 * (float)((pk3       ) & 0xFu) + zp3;
        float w3_1 = sc3 * (float)((pk3 >>  4) & 0xFu) + zp3;
        float w3_2 = sc3 * (float)((pk3 >>  8) & 0xFu) + zp3;
        float w3_3 = sc3 * (float)((pk3 >> 12) & 0xFu) + zp3;
        float w3_4 = sc3 * (float)((pk3 >> 16) & 0xFu) + zp3;
        float w3_5 = sc3 * (float)((pk3 >> 20) & 0xFu) + zp3;
        float w3_6 = sc3 * (float)((pk3 >> 24) & 0xFu) + zp3;
        float w3_7 = sc3 * (float)((pk3 >> 28) & 0xFu) + zp3;

        const int base = g * 256 + tid * 8;

        #pragma unroll
        for (int b = 0; b < BATCH_TILE; b++) {
            if (b >= local_bs) break;
            const float* xb = X + (long long)(batch_start + b) * K;
            const float x0 = xb[base    ];
            const float x1 = xb[base + 1];
            const float x2 = xb[base + 2];
            const float x3 = xb[base + 3];
            const float x4 = xb[base + 4];
            const float x5 = xb[base + 5];
            const float x6 = xb[base + 6];
            const float x7 = xb[base + 7];

            acc0[b] += w0_0 * x0 + w0_1 * x1 + w0_2 * x2 + w0_3 * x3
                     + w0_4 * x4 + w0_5 * x5 + w0_6 * x6 + w0_7 * x7;

            const float x1_0 = xb[base + 256    ];
            const float x1_1 = xb[base + 256 + 1];
            const float x1_2 = xb[base + 256 + 2];
            const float x1_3 = xb[base + 256 + 3];
            const float x1_4 = xb[base + 256 + 4];
            const float x1_5 = xb[base + 256 + 5];
            const float x1_6 = xb[base + 256 + 6];
            const float x1_7 = xb[base + 256 + 7];

            acc1[b] += w1_0 * x1_0 + w1_1 * x1_1 + w1_2 * x1_2 + w1_3 * x1_3
                     + w1_4 * x1_4 + w1_5 * x1_5 + w1_6 * x1_6 + w1_7 * x1_7;

            const float x2_0 = xb[base + 512    ];
            const float x2_1 = xb[base + 512 + 1];
            const float x2_2 = xb[base + 512 + 2];
            const float x2_3 = xb[base + 512 + 3];
            const float x2_4 = xb[base + 512 + 4];
            const float x2_5 = xb[base + 512 + 5];
            const float x2_6 = xb[base + 512 + 6];
            const float x2_7 = xb[base + 512 + 7];

            acc2[b] += w2_0 * x2_0 + w2_1 * x2_1 + w2_2 * x2_2 + w2_3 * x2_3
                     + w2_4 * x2_4 + w2_5 * x2_5 + w2_6 * x2_6 + w2_7 * x2_7;

            const float x3_0 = xb[base + 768    ];
            const float x3_1 = xb[base + 768 + 1];
            const float x3_2 = xb[base + 768 + 2];
            const float x3_3 = xb[base + 768 + 3];
            const float x3_4 = xb[base + 768 + 4];
            const float x3_5 = xb[base + 768 + 5];
            const float x3_6 = xb[base + 768 + 6];
            const float x3_7 = xb[base + 768 + 7];

            acc3[b] += w3_0 * x3_0 + w3_1 * x3_1 + w3_2 * x3_2 + w3_3 * x3_3
                     + w3_4 * x3_4 + w3_5 * x3_5 + w3_6 * x3_6 + w3_7 * x3_7;
        }
    }

    #define TAIL_BODY(acc_arr)                                                  \
        {                                                                        \
            const char* gp = row_ptr + g * 136;                                  \
            float sc, zp;                                                        \
            LOAD_V2_SC_ZP(gp, sc, zp);                                           \
            unsigned int pk = *(const unsigned int*)(gp + 8 + boff);             \
            float w0 = sc * (float)((pk       ) & 0xFu) + zp;                    \
            float w1 = sc * (float)((pk >>  4) & 0xFu) + zp;                    \
            float w2 = sc * (float)((pk >>  8) & 0xFu) + zp;                    \
            float w3 = sc * (float)((pk >> 12) & 0xFu) + zp;                    \
            float w4 = sc * (float)((pk >> 16) & 0xFu) + zp;                    \
            float w5 = sc * (float)((pk >> 20) & 0xFu) + zp;                    \
            float w6 = sc * (float)((pk >> 24) & 0xFu) + zp;                    \
            float w7 = sc * (float)((pk >> 28) & 0xFu) + zp;                    \
            const int base = g * 256 + tid * 8;                                  \
            _Pragma("unroll")                                                    \
            for (int b = 0; b < BATCH_TILE; b++) {                               \
                if (b >= local_bs) break;                                        \
                const float* xb = X + (long long)(batch_start + b) * K;          \
                const float x0 = xb[base    ];                                   \
                const float x1 = xb[base + 1];                                   \
                const float x2 = xb[base + 2];                                   \
                const float x3 = xb[base + 3];                                   \
                const float x4 = xb[base + 4];                                   \
                const float x5 = xb[base + 5];                                   \
                const float x6 = xb[base + 6];                                   \
                const float x7 = xb[base + 7];                                   \
                (acc_arr)[b] += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3            \
                              + w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7;           \
            }                                                                    \
        }

    if (tail >= 1) {
        const int g = quads << 2;
        TAIL_BODY(acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        TAIL_BODY(acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        TAIL_BODY(acc2);
    }
    #undef TAIL_BODY
    #undef LOAD_V2_SC_ZP

    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        if (b >= local_bs) break;
        float val = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
        val += __shfl_down(val, 16, 32);
        val += __shfl_down(val,  8, 32);
        val += __shfl_down(val,  4, 32);
        val += __shfl_down(val,  2, 32);
        val += __shfl_down(val,  1, 32);
        if (tid == 0) Y[(long long)(batch_start + b) * M + row] = val;
    }
}

// ---------------------------------------------------------------------------
// Shared G1 one-group decode+consume helper shape (R1 paths).
// Four explicit calls per quad — no runtime acc[s] indexing.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// SetR1T8G1 — one group live; four chains x 8 tokens.
// Grid: [M, ceil(N/8), 1]
// ---------------------------------------------------------------------------

__launch_bounds__(32, 4)
extern "C" __global__ void gemm_mq4g256v2_set_simt_r1t8_g1_gfx1010(
    const char* __restrict__ A,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int BATCH_TILE = 8;
    const int row = blockIdx.x;
    if (row >= M) return;
    const int batch_start = blockIdx.y * BATCH_TILE;
    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    float acc0[8], acc1[8], acc2[8], acc3[8];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        acc0[b] = 0.0f;
        acc1[b] = 0.0f;
        acc2[b] = 0.0f;
        acc3[b] = 0.0f;
    }

    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;

    #define LOAD_V2_SC_ZP(gp, sc, zp) do {                                      \
        const unsigned int hA = *(const unsigned int*)(gp);                     \
        const unsigned int hB = *(const unsigned int*)((gp) + 4);               \
        const unsigned int hs = (tid < 16) ? hA : hB;                           \
        (sc) = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));  \
        (zp) = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));      \
    } while (0)

    // Decode one group, consume into the named chain, drop weights before next.
    #define PROCESS_GROUP_R1T8(g_idx, acc_arr) do {                             \
        const char* gp = row_ptr + (long long)(g_idx) * 136;                    \
        float sc, zp;                                                            \
        LOAD_V2_SC_ZP(gp, sc, zp);                                               \
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);                 \
        const float w0 = sc * (float)((pk       ) & 0xFu) + zp;                  \
        const float w1 = sc * (float)((pk >>  4) & 0xFu) + zp;                  \
        const float w2 = sc * (float)((pk >>  8) & 0xFu) + zp;                  \
        const float w3 = sc * (float)((pk >> 12) & 0xFu) + zp;                  \
        const float w4 = sc * (float)((pk >> 16) & 0xFu) + zp;                  \
        const float w5 = sc * (float)((pk >> 20) & 0xFu) + zp;                  \
        const float w6 = sc * (float)((pk >> 24) & 0xFu) + zp;                  \
        const float w7 = sc * (float)((pk >> 28) & 0xFu) + zp;                  \
        const int base = (g_idx) * 256 + tid * 8;                                \
        _Pragma("unroll")                                                        \
        for (int b = 0; b < BATCH_TILE; b++) {                                   \
            if (b >= local_bs) break;                                            \
            const float* xb = X + (long long)(batch_start + b) * K;              \
            const float x0 = xb[base    ];                                       \
            const float x1 = xb[base + 1];                                       \
            const float x2 = xb[base + 2];                                       \
            const float x3 = xb[base + 3];                                       \
            const float x4 = xb[base + 4];                                       \
            const float x5 = xb[base + 5];                                       \
            const float x6 = xb[base + 6];                                       \
            const float x7 = xb[base + 7];                                       \
            (acc_arr)[b] += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3                \
                          + w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7;               \
        }                                                                        \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        PROCESS_GROUP_R1T8(g + 0, acc0);
        PROCESS_GROUP_R1T8(g + 1, acc1);
        PROCESS_GROUP_R1T8(g + 2, acc2);
        PROCESS_GROUP_R1T8(g + 3, acc3);
    }

    if (tail >= 1) {
        const int g = quads << 2;
        PROCESS_GROUP_R1T8(g, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        PROCESS_GROUP_R1T8(g, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        PROCESS_GROUP_R1T8(g, acc2);
    }
    #undef PROCESS_GROUP_R1T8
    #undef LOAD_V2_SC_ZP

    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        if (b >= local_bs) break;
        float val = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
        val += __shfl_down(val, 16, 32);
        val += __shfl_down(val,  8, 32);
        val += __shfl_down(val,  4, 32);
        val += __shfl_down(val,  2, 32);
        val += __shfl_down(val,  1, 32);
        if (tid == 0) Y[(long long)(batch_start + b) * M + row] = val;
    }
}

// ---------------------------------------------------------------------------
// SetR1T16G1 — one group live; four chains x 16 tokens.
// Grid: [M, ceil(N/16), 1]
// ---------------------------------------------------------------------------

__launch_bounds__(32, 4)
extern "C" __global__ void gemm_mq4g256v2_set_simt_r1t16_g1_gfx1010(
    const char* __restrict__ A,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int BATCH_TILE = 16;
    const int row = blockIdx.x;
    if (row >= M) return;
    const int batch_start = blockIdx.y * BATCH_TILE;
    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    float acc0[16], acc1[16], acc2[16], acc3[16];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        acc0[b] = 0.0f;
        acc1[b] = 0.0f;
        acc2[b] = 0.0f;
        acc3[b] = 0.0f;
    }

    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;

    #define LOAD_V2_SC_ZP(gp, sc, zp) do {                                      \
        const unsigned int hA = *(const unsigned int*)(gp);                     \
        const unsigned int hB = *(const unsigned int*)((gp) + 4);               \
        const unsigned int hs = (tid < 16) ? hA : hB;                           \
        (sc) = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));  \
        (zp) = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));      \
    } while (0)

    #define PROCESS_GROUP_R1T16(g_idx, acc_arr) do {                            \
        const char* gp = row_ptr + (long long)(g_idx) * 136;                    \
        float sc, zp;                                                            \
        LOAD_V2_SC_ZP(gp, sc, zp);                                               \
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);                 \
        const float w0 = sc * (float)((pk       ) & 0xFu) + zp;                  \
        const float w1 = sc * (float)((pk >>  4) & 0xFu) + zp;                  \
        const float w2 = sc * (float)((pk >>  8) & 0xFu) + zp;                  \
        const float w3 = sc * (float)((pk >> 12) & 0xFu) + zp;                  \
        const float w4 = sc * (float)((pk >> 16) & 0xFu) + zp;                  \
        const float w5 = sc * (float)((pk >> 20) & 0xFu) + zp;                  \
        const float w6 = sc * (float)((pk >> 24) & 0xFu) + zp;                  \
        const float w7 = sc * (float)((pk >> 28) & 0xFu) + zp;                  \
        const int base = (g_idx) * 256 + tid * 8;                                \
        _Pragma("unroll")                                                        \
        for (int b = 0; b < BATCH_TILE; b++) {                                   \
            if (b >= local_bs) break;                                            \
            const float* xb = X + (long long)(batch_start + b) * K;              \
            const float x0 = xb[base    ];                                       \
            const float x1 = xb[base + 1];                                       \
            const float x2 = xb[base + 2];                                       \
            const float x3 = xb[base + 3];                                       \
            const float x4 = xb[base + 4];                                       \
            const float x5 = xb[base + 5];                                       \
            const float x6 = xb[base + 6];                                       \
            const float x7 = xb[base + 7];                                       \
            (acc_arr)[b] += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3                \
                          + w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7;               \
        }                                                                        \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        PROCESS_GROUP_R1T16(g + 0, acc0);
        PROCESS_GROUP_R1T16(g + 1, acc1);
        PROCESS_GROUP_R1T16(g + 2, acc2);
        PROCESS_GROUP_R1T16(g + 3, acc3);
    }

    if (tail >= 1) {
        const int g = quads << 2;
        PROCESS_GROUP_R1T16(g, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        PROCESS_GROUP_R1T16(g, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        PROCESS_GROUP_R1T16(g, acc2);
    }
    #undef PROCESS_GROUP_R1T16
    #undef LOAD_V2_SC_ZP

    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        if (b >= local_bs) break;
        float val = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
        val += __shfl_down(val, 16, 32);
        val += __shfl_down(val,  8, 32);
        val += __shfl_down(val,  4, 32);
        val += __shfl_down(val,  2, 32);
        val += __shfl_down(val,  1, 32);
        if (tid == 0) Y[(long long)(batch_start + b) * M + row] = val;
    }
}

// ---------------------------------------------------------------------------
// SetR2T8G1 — two rows x eight tokens; shared X per token; G1 schedule.
// Grid: [ceil(M/2), ceil(N/8), 1]
// Odd final row clamps safe A pointer to row0 and masks the second store.
// Block-uniform: no early return that desyncs rows; only stores are masked.
// ---------------------------------------------------------------------------

__launch_bounds__(32, 4)
extern "C" __global__ void gemm_mq4g256v2_set_simt_r2t8_g1_gfx1010(
    const char* __restrict__ A,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int BATCH_TILE = 8;
    const int row0 = (int)blockIdx.x * 2;
    const int row1 = row0 + 1;
    const int have_row1 = row1 < M;
    // Clamp A for the second row so the odd-M final block stays uniform.
    const int safe_row1 = have_row1 ? row1 : row0;

    const int batch_start = blockIdx.y * BATCH_TILE;
    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row0_ptr = A + (long long)row0 * groups_per_row * 136;
    const char* row1_ptr = A + (long long)safe_row1 * groups_per_row * 136;
    const int boff = tid * 4;

    // a* = row0 chains; b* = row1 chains. Independent four-chain order.
    float a0[8], a1[8], a2[8], a3[8];
    float b0[8], b1[8], b2[8], b3[8];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        a0[b] = 0.0f; a1[b] = 0.0f; a2[b] = 0.0f; a3[b] = 0.0f;
        b0[b] = 0.0f; b1[b] = 0.0f; b2[b] = 0.0f; b3[b] = 0.0f;
    }

    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;

    #define LOAD_V2_SC_ZP_AT(gp, sc, zp) do {                                   \
        const unsigned int hA = *(const unsigned int*)(gp);                     \
        const unsigned int hB = *(const unsigned int*)((gp) + 4);               \
        const unsigned int hs = (tid < 16) ? hA : hB;                           \
        (sc) = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));  \
        (zp) = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));      \
    } while (0)

    // One group slot: decode eight weights for safe row0 and safe row1, then
    // for each active token load x0..x7 once and apply both row expressions.
    #define PROCESS_GROUP_R2T8(g_idx, a_arr, b_arr) do {                        \
        const char* gp_a = row0_ptr + (long long)(g_idx) * 136;                 \
        const char* gp_b = row1_ptr + (long long)(g_idx) * 136;                 \
        float sc_a, zp_a, sc_b, zp_b;                                            \
        LOAD_V2_SC_ZP_AT(gp_a, sc_a, zp_a);                                      \
        LOAD_V2_SC_ZP_AT(gp_b, sc_b, zp_b);                                      \
        unsigned int pk_a = *(const unsigned int*)(gp_a + 8 + boff);             \
        unsigned int pk_b = *(const unsigned int*)(gp_b + 8 + boff);             \
        const float wa0 = sc_a * (float)((pk_a       ) & 0xFu) + zp_a;           \
        const float wa1 = sc_a * (float)((pk_a >>  4) & 0xFu) + zp_a;           \
        const float wa2 = sc_a * (float)((pk_a >>  8) & 0xFu) + zp_a;           \
        const float wa3 = sc_a * (float)((pk_a >> 12) & 0xFu) + zp_a;           \
        const float wa4 = sc_a * (float)((pk_a >> 16) & 0xFu) + zp_a;           \
        const float wa5 = sc_a * (float)((pk_a >> 20) & 0xFu) + zp_a;           \
        const float wa6 = sc_a * (float)((pk_a >> 24) & 0xFu) + zp_a;           \
        const float wa7 = sc_a * (float)((pk_a >> 28) & 0xFu) + zp_a;           \
        const float wb0 = sc_b * (float)((pk_b       ) & 0xFu) + zp_b;           \
        const float wb1 = sc_b * (float)((pk_b >>  4) & 0xFu) + zp_b;           \
        const float wb2 = sc_b * (float)((pk_b >>  8) & 0xFu) + zp_b;           \
        const float wb3 = sc_b * (float)((pk_b >> 12) & 0xFu) + zp_b;           \
        const float wb4 = sc_b * (float)((pk_b >> 16) & 0xFu) + zp_b;           \
        const float wb5 = sc_b * (float)((pk_b >> 20) & 0xFu) + zp_b;           \
        const float wb6 = sc_b * (float)((pk_b >> 24) & 0xFu) + zp_b;           \
        const float wb7 = sc_b * (float)((pk_b >> 28) & 0xFu) + zp_b;           \
        const int base = (g_idx) * 256 + tid * 8;                                \
        _Pragma("unroll")                                                        \
        for (int b = 0; b < BATCH_TILE; b++) {                                   \
            if (b >= local_bs) break;                                            \
            const float* xb = X + (long long)(batch_start + b) * K;              \
            const float x0 = xb[base    ];                                       \
            const float x1 = xb[base + 1];                                       \
            const float x2 = xb[base + 2];                                       \
            const float x3 = xb[base + 3];                                       \
            const float x4 = xb[base + 4];                                       \
            const float x5 = xb[base + 5];                                       \
            const float x6 = xb[base + 6];                                       \
            const float x7 = xb[base + 7];                                       \
            (a_arr)[b] += wa0 * x0 + wa1 * x1 + wa2 * x2 + wa3 * x3              \
                        + wa4 * x4 + wa5 * x5 + wa6 * x6 + wa7 * x7;             \
            (b_arr)[b] += wb0 * x0 + wb1 * x1 + wb2 * x2 + wb3 * x3              \
                        + wb4 * x4 + wb5 * x5 + wb6 * x6 + wb7 * x7;             \
        }                                                                        \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        PROCESS_GROUP_R2T8(g + 0, a0, b0);
        PROCESS_GROUP_R2T8(g + 1, a1, b1);
        PROCESS_GROUP_R2T8(g + 2, a2, b2);
        PROCESS_GROUP_R2T8(g + 3, a3, b3);
    }

    if (tail >= 1) {
        const int g = quads << 2;
        PROCESS_GROUP_R2T8(g, a0, b0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        PROCESS_GROUP_R2T8(g, a1, b1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        PROCESS_GROUP_R2T8(g, a2, b2);
    }
    #undef PROCESS_GROUP_R2T8
    #undef LOAD_V2_SC_ZP_AT

    // Per-token pairwise combine + width-32 shuffle; row0 always stores when
    // launched (grid.x = ceil(M/2)); row1 store masked by have_row1.
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        if (b >= local_bs) break;

        float val_a = (a0[b] + a1[b]) + (a2[b] + a3[b]);
        val_a += __shfl_down(val_a, 16, 32);
        val_a += __shfl_down(val_a,  8, 32);
        val_a += __shfl_down(val_a,  4, 32);
        val_a += __shfl_down(val_a,  2, 32);
        val_a += __shfl_down(val_a,  1, 32);
        if (tid == 0) Y[(long long)(batch_start + b) * M + row0] = val_a;

        float val_b = (b0[b] + b1[b]) + (b2[b] + b3[b]);
        val_b += __shfl_down(val_b, 16, 32);
        val_b += __shfl_down(val_b,  8, 32);
        val_b += __shfl_down(val_b,  4, 32);
        val_b += __shfl_down(val_b,  2, 32);
        val_b += __shfl_down(val_b,  1, 32);
        if (tid == 0 && have_row1)
            Y[(long long)(batch_start + b) * M + row1] = val_b;
    }
}

#endif // defined(__gfx1010__)
