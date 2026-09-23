// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx12 (RDNA4) HFQ4-G128 GEMV: 4-way group unroll, 4 separate accumulators
// for ILP. Mirrors the proven gfx1100 pattern from
// `gemv_hfq4g256.gfx1100.hip` (acc0..acc3 reduces FMA-chain serialization).
//
// Baseline `gemv_hfq4g128.hip` uses ONE accumulator, serializing all FMAs.
// For K=2048 → 16 groups, each iter has 4 FMAs serially. 4-way ILP cuts
// the dependency chain length by 4×.
//
// rocprof: gemv_hfq4g128 = 16.0 % of z-lab A3B-PARO decode GPU time
// (222.7 ms / 1395.9 ms, 13000 calls, 17.1 µs/call). Target: ~12 µs/call
// with the 4-acc pattern.
//
// __launch_bounds__: gfx12 wave32 has 1536 VGPRs per SIMD. With ~32 VGPRs
// per WG estimate, 16 occupancy is comfortable.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g128_gfx12(
    const char*  __restrict__ A,
    const float* __restrict__ x,
    float*       __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int byte_off = tid * 2;

    // 4 independent accumulators — 4-way group unroll.
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 72;
        const char* gp1 = gp0 + 72;
        const char* gp2 = gp1 + 72;
        const char* gp3 = gp2 + 72;

        // Per-group scale + zero, 4 groups in parallel.
        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
        float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4));
        float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));
        float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4));

        // Nibble loads — 2 bytes per thread per group = 4 weights.
        unsigned char b00 = *(const unsigned char*)(gp0 + 8 + byte_off);
        unsigned char b01 = *(const unsigned char*)(gp0 + 8 + byte_off + 1);
        unsigned char b10 = *(const unsigned char*)(gp1 + 8 + byte_off);
        unsigned char b11 = *(const unsigned char*)(gp1 + 8 + byte_off + 1);
        unsigned char b20 = *(const unsigned char*)(gp2 + 8 + byte_off);
        unsigned char b21 = *(const unsigned char*)(gp2 + 8 + byte_off + 1);
        unsigned char b30 = *(const unsigned char*)(gp3 + 8 + byte_off);
        unsigned char b31 = *(const unsigned char*)(gp3 + 8 + byte_off + 1);

        int base = g * 128 + tid * 4;

        #define DOG(b0, b1, sc, zp, base_idx, a) \
            (a) += (sc * (float)((b0) & 0xFu)        + zp) * x[(base_idx)]     \
                 + (sc * (float)(((b0) >> 4) & 0xFu) + zp) * x[(base_idx) + 1] \
                 + (sc * (float)((b1) & 0xFu)        + zp) * x[(base_idx) + 2] \
                 + (sc * (float)(((b1) >> 4) & 0xFu) + zp) * x[(base_idx) + 3]

        DOG(b00, b01, sc0, zp0, base,       acc0);
        DOG(b10, b11, sc1, zp1, base + 128, acc1);
        DOG(b20, b21, sc2, zp2, base + 256, acc2);
        DOG(b30, b31, sc3, zp3, base + 384, acc3);
        #undef DOG
    }

    // Tail groups (groups_per_row % 4). Each tail group accumulates into
    // its own slot (acc[g % 4]) to preserve the 4-way interleaving — same
    // bug rule as the gfx1100 sister documented at commit 5302926.
    #define TAIL_DOG(b0, b1, sc, zp, base_idx, a) \
        (a) += (sc * (float)((b0) & 0xFu)        + zp) * x[(base_idx)]     \
             + (sc * (float)(((b0) >> 4) & 0xFu) + zp) * x[(base_idx) + 1] \
             + (sc * (float)((b1) & 0xFu)        + zp) * x[(base_idx) + 2] \
             + (sc * (float)(((b1) >> 4) & 0xFu) + zp) * x[(base_idx) + 3]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 72;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned char b0 = *(const unsigned char*)(gp + 8 + byte_off);
        unsigned char b1 = *(const unsigned char*)(gp + 8 + byte_off + 1);
        int base = g * 128 + tid * 4;
        TAIL_DOG(b0, b1, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 72;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned char b0 = *(const unsigned char*)(gp + 8 + byte_off);
        unsigned char b1 = *(const unsigned char*)(gp + 8 + byte_off + 1);
        int base = g * 128 + tid * 4;
        TAIL_DOG(b0, b1, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 72;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned char b0 = *(const unsigned char*)(gp + 8 + byte_off);
        unsigned char b1 = *(const unsigned char*)(gp + 8 + byte_off + 1);
        int base = g * 128 + tid * 4;
        TAIL_DOG(b0, b1, sc, zp, base, acc2);
    }
    #undef TAIL_DOG

    // Combine 4 partial sums, then wave-shfl reduce across lanes.
    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }
    if (tid == 0) y[row] = acc;
}
