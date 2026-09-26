// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — QT44 grouped top-10 gate/up with F32 scalar parity.
//
// This is the F32 parity path on every GPU other than the gfx1151 exact-shape
// optimization.  It keeps the indexed decode kernel's F32 input, MQ4G256V2
// dequantization, four-accumulator lane traversal, and 32-lane shuffle
// reduction while reusing one decoded weight row across 16 routed slots of
// the same expert tile. Other grouped qt44 MoE callers may still use WMMA,
// whose F16 staging/accumulation is not the Qwen4 source arithmetic.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Layout:
//   expert_weight_ptrs: [E] device pointers to [M, K] MQ4G256V2 rows
//   expert_tile_ids:   [ceil(grouped_rows / 16)] expert ID per slot tile
//   sorted_slot_index: [grouped_rows] flat token/top-k slot, -1 for padding
//   X_src:             [N, K] F32 rotated token rows
//   Y_grouped:         [grouped_rows, M] F32 grouped output
//
// Grid: [M, ceil(grouped_rows / 16), 1], block 32.
// Each wave computes one output row and 16 routed slots.  A decoded weight
// value is reused across all valid slots, while each slot keeps the same four
// lane accumulators and reduction order as the indexed decode kernel.
__launch_bounds__(32, 8)
extern "C" __global__ void gemm_mq4g256v2_moe_grouped_top10_simt(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index,
    const float* __restrict__ X_src,
    float* __restrict__ Y_grouped,
    int M,
    int K,
    int x_row_div,
    int grouped_rows
) {
    const int row = blockIdx.x;
    const int slot_start = blockIdx.y * 16;
    const int tid = threadIdx.x;
    if (row >= M || slot_start >= grouped_rows || tid >= 32) return;

    const int expert_id = expert_tile_ids[blockIdx.y];
    if (expert_id < 0) return;
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    int flat[16];
    const float* x_rows[16];
    #pragma unroll
    for (int slot = 0; slot < 16; ++slot) {
        const int grouped_slot = slot_start + slot;
        flat[slot] = (grouped_slot < grouped_rows) ? sorted_slot_index[grouped_slot] : -1;
        if (flat[slot] >= 0) {
            const int x_row = (x_row_div > 1) ? (flat[slot] / x_row_div) : flat[slot];
            x_rows[slot] = X_src + (long long)x_row * K;
        } else {
            x_rows[slot] = nullptr;
        }
    }

    const int groups_per_row = K / 256;
    const char* __restrict__ row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;
    const int hoff = (tid < 16) ? 0 : 4;

    float acc0[16] = {};
    float acc1[16] = {};
    float acc2[16] = {};
    float acc3[16] = {};

    // Keep the indexed kernel's four accumulators: group g contributes to
    // accumulator g&3, then the four partials are combined in the same order.
    // Process complete four-group quads with statically named accumulator
    // arrays. The remaining 0..3 groups use the same static calls, retaining
    // the generic K/groups_per_row contract without a dynamic array pointer.
    #define ACCUM_GROUP(G, ACC) do { \
        const char* gp = row_ptr + (G) * 136; \
        const unsigned int h = *(const unsigned int*)(gp + hoff); \
        const float sc = __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu))); \
        const float zp = __half2float(__ushort_as_half((unsigned short)(h >> 16))); \
        const unsigned int pk = *(const unsigned int*)(gp + 8 + boff); \
        const int base = (G) * 256 + tid * 8; \
        const float w0 = sc * (float)((pk >> 0) & 0xFu) + zp; \
        const float w1 = sc * (float)((pk >> 4) & 0xFu) + zp; \
        const float w2 = sc * (float)((pk >> 8) & 0xFu) + zp; \
        const float w3 = sc * (float)((pk >> 12) & 0xFu) + zp; \
        const float w4 = sc * (float)((pk >> 16) & 0xFu) + zp; \
        const float w5 = sc * (float)((pk >> 20) & 0xFu) + zp; \
        const float w6 = sc * (float)((pk >> 24) & 0xFu) + zp; \
        const float w7 = sc * (float)((pk >> 28) & 0xFu) + zp; \
        _Pragma("unroll") \
        for (int slot = 0; slot < 16; ++slot) { \
            const float* x = x_rows[slot]; \
            if (x == nullptr) continue; \
            (ACC)[slot] += w0 * x[base + 0] + w1 * x[base + 1] \
                + w2 * x[base + 2] + w3 * x[base + 3] \
                + w4 * x[base + 4] + w5 * x[base + 5] \
                + w6 * x[base + 6] + w7 * x[base + 7]; \
        } \
    } while (0)

    int g = 0;
    for (; g + 3 < groups_per_row; g += 4) {
        ACCUM_GROUP(g, acc0);
        ACCUM_GROUP(g + 1, acc1);
        ACCUM_GROUP(g + 2, acc2);
        ACCUM_GROUP(g + 3, acc3);
    }
    for (; g < groups_per_row; ++g) {
        switch (g & 3) {
        case 0:
            ACCUM_GROUP(g, acc0);
            break;
        case 1:
            ACCUM_GROUP(g, acc1);
            break;
        case 2:
            ACCUM_GROUP(g, acc2);
            break;
        default:
            ACCUM_GROUP(g, acc3);
            break;
        }
    }
    #undef ACCUM_GROUP

    float acc[16];
    #pragma unroll
    for (int slot = 0; slot < 16; ++slot) {
        acc[slot] = (acc0[slot] + acc1[slot]) + (acc2[slot] + acc3[slot]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int slot = 0; slot < 16; ++slot) {
            acc[slot] += __shfl_down(acc[slot], offset, 32);
        }
    }
    if (tid == 0) {
        #pragma unroll
        for (int slot = 0; slot < 16; ++slot) {
            const int grouped_slot = slot_start + slot;
            if (grouped_slot < grouped_rows) {
                Y_grouped[(long long)grouped_slot * M + row] = acc[slot];
            }
        }
    }
}
