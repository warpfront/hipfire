// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1151 QT44 grouped top-10 gate/up O2×R8.
//
// This exact-shape gfx1151 path keeps the static production kernel's F32 input,
// MQ4G256V2 decoding, four static accumulator chains, and full wave32
// reduction, while pairing two adjacent output rows with eight route slots.
//
// The producer groups routed slots by expert into sixteen-slot tiles with the
// expert's slots sorted first, so an expert with at most eight routed slots
// leaves its tile's second eight-slot subtile entirely sentinel — and the
// mapping's second subtile block used to fetch that expert's whole weight row
// for slots that cannot contribute to any output.  Such a subtile now skips
// the K walk and stores the same zeros the full walk produced.  On the retained
// 291-token prefill shape, where the padded tile count is dominated by experts
// holding five or six slots, that duplicate fetch is the kernel's largest
// single traffic item.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define HF_MQ4_O2_R8_SLOTS 8

// Layouts are the live grouped gate/up contract:
//   expert_weight_ptrs: [E] device pointers to [M, K] MQ4G256V2 rows
//   expert_tile_ids:   [ceil(grouped_rows / 16)] expert ID per old 16-slot tile
//   sorted_slot_index: [grouped_rows] flat token/top-k slot, -1 for padding
//   X_src:             [N, K] F32 rotated token rows
//   Y_grouped:         [grouped_rows, M] F32 grouped output
//
// Grid: [ceil(M / 2), ceil(grouped_rows / 8), 1], block 32.  blockIdx.y is
// an eight-slot subtile, but the metadata remains at the old 16-slot tile
// granularity: subtile s reads expert_tile_ids[s / 2].  Thus the two subtile
// blocks belonging to one old tile use exactly the same expert ID.
//
// Arithmetic invariants:
//   * Each block owns two adjacent output rows and eight slots.  It has
//     4 chains × 8 slots × 2 output rows = 64 static accumulators, the same
//     accumulator count as the live O1×R16 source.
//   * For each group and slot, the eight X scalars are loaded once and consumed
//     by both output rows.  The two rows decode their own packed weight word and
//     header; no LDS, barrier, new math, or cross-wave cooperation is used.
//   * Each output row processes groups in the same complete-quad/tail order as
//     the live source.  Its final chain association remains
//     (acc0 + acc1) + (acc2 + acc3), followed by the same full-wave shuffle
//     offsets 16, 8, 4, 2, 1.  The second row follows the identical sequence.
//   * A subtile with no live slot accumulates nothing, so eliding its walk
//     leaves every accumulator at zero and the epilogue writes exactly the
//     zeros the full walk wrote.  The group walk bound is the only dynamic
//     part of the loop structure.
//   * Odd M leaves row1 invalid.  The row1 pointer aliases row0 only to keep
//     pointer formation harmless; all row1 accumulation and stores are gated.
//     A negative expert ID leaves the output sentinel untouched, matching the
//     live source's early return.  Negative sorted slots write zero, and
//     partial final eight-slot subtiles write only grouped_rows-valid slots.
//
// Two rows retain 64 accumulators and need the second row's weight state.  The
// lower residency request avoids forcing spills while retaining the production
// arithmetic and output contract.
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g256v2_moe_grouped_top10_o2_r8(
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
    const int row0 = blockIdx.x * 2;
    const int row1 = row0 + 1;
    const int subtile = blockIdx.y;
    const int slot_start = subtile * HF_MQ4_O2_R8_SLOTS;
    const int tid = threadIdx.x;
    if (row0 >= M || slot_start >= grouped_rows || tid >= 32) return;

    // Production expert metadata is still one entry per old 16-slot tile.
    // Every pair of R8 subtiles therefore shares the same expert pointer.
    const int old_tile = subtile >> 1;
    const int expert_id = expert_tile_ids[old_tile];
    if (expert_id < 0) return;
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const bool row1_valid = row1 < M;
    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* __restrict__ row_ptr0 =
        A + (long long)row0 * row_stride;
    // Do not form an out-of-range pointer for odd M.  This alias is never
    // accumulated or stored when row1_valid is false.
    const char* __restrict__ row_ptr1 = row1_valid
        ? A + (long long)row1 * row_stride
        : row_ptr0;

    const int boff = tid * 4;
    const int hoff = (tid < 16) ? 0 : 4;

    int flat[HF_MQ4_O2_R8_SLOTS];
    const float* x_rows[HF_MQ4_O2_R8_SLOTS];
    bool any_live = false;
#pragma unroll
    for (int slot = 0; slot < HF_MQ4_O2_R8_SLOTS; ++slot) {
        const int grouped_slot = slot_start + slot;
        flat[slot] = (grouped_slot < grouped_rows)
            ? sorted_slot_index[grouped_slot] : -1;
        if (flat[slot] >= 0) {
            const int x_row = (x_row_div > 1)
                ? (flat[slot] / x_row_div) : flat[slot];
            x_rows[slot] = X_src + (long long)x_row * K;
            any_live = true;
        } else {
            x_rows[slot] = nullptr;
        }
    }

    // With no live slot every weight word of this subtile is multiplied by
    // nothing, so the walk and its DRAM fetches are elided.  The fold, shuffle
    // and store still run and still write the zero every sentinel slot owes.
    const int group_walk = any_live ? groups_per_row : 0;

    float acc00[HF_MQ4_O2_R8_SLOTS] = {};
    float acc01[HF_MQ4_O2_R8_SLOTS] = {};
    float acc02[HF_MQ4_O2_R8_SLOTS] = {};
    float acc03[HF_MQ4_O2_R8_SLOTS] = {};
    float acc10[HF_MQ4_O2_R8_SLOTS] = {};
    float acc11[HF_MQ4_O2_R8_SLOTS] = {};
    float acc12[HF_MQ4_O2_R8_SLOTS] = {};
    float acc13[HF_MQ4_O2_R8_SLOTS] = {};

    // Decode one weight row for each adjacent output, then consume each X
    // scalar through both rows before advancing to the next slot.  Keeping the
    // x0..x7 locals in this scope expresses the exact one-load/two-consumer
    // invariant without introducing an X tile or LDS staging.
#define ACCUM_GROUP(G, ACC0, ACC1) do { \
        const char* gp0 = row_ptr0 + (long long)(G) * 136; \
        const unsigned int h0 = *(const unsigned int*)(gp0 + hoff); \
        const float sc0 = __half2float(__ushort_as_half( \
            (unsigned short)(h0 & 0xFFFFu))); \
        const float zp0 = __half2float(__ushort_as_half( \
            (unsigned short)(h0 >> 16))); \
        const unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff); \
        const float w00 = sc0 * (float)((pk0 >> 0) & 0xFu) + zp0; \
        const float w01 = sc0 * (float)((pk0 >> 4) & 0xFu) + zp0; \
        const float w02 = sc0 * (float)((pk0 >> 8) & 0xFu) + zp0; \
        const float w03 = sc0 * (float)((pk0 >> 12) & 0xFu) + zp0; \
        const float w04 = sc0 * (float)((pk0 >> 16) & 0xFu) + zp0; \
        const float w05 = sc0 * (float)((pk0 >> 20) & 0xFu) + zp0; \
        const float w06 = sc0 * (float)((pk0 >> 24) & 0xFu) + zp0; \
        const float w07 = sc0 * (float)((pk0 >> 28) & 0xFu) + zp0; \
        const char* gp1 = row_ptr1 + (long long)(G) * 136; \
        const unsigned int h1 = row1_valid \
            ? *(const unsigned int*)(gp1 + hoff) : 0u; \
        const float sc1 = row1_valid \
            ? __half2float(__ushort_as_half( \
                (unsigned short)(h1 & 0xFFFFu))) : 0.0f; \
        const float zp1 = row1_valid \
            ? __half2float(__ushort_as_half( \
                (unsigned short)(h1 >> 16))) : 0.0f; \
        const unsigned int pk1 = row1_valid \
            ? *(const unsigned int*)(gp1 + 8 + boff) : 0u; \
        const float w10 = sc1 * (float)((pk1 >> 0) & 0xFu) + zp1; \
        const float w11 = sc1 * (float)((pk1 >> 4) & 0xFu) + zp1; \
        const float w12 = sc1 * (float)((pk1 >> 8) & 0xFu) + zp1; \
        const float w13 = sc1 * (float)((pk1 >> 12) & 0xFu) + zp1; \
        const float w14 = sc1 * (float)((pk1 >> 16) & 0xFu) + zp1; \
        const float w15 = sc1 * (float)((pk1 >> 20) & 0xFu) + zp1; \
        const float w16 = sc1 * (float)((pk1 >> 24) & 0xFu) + zp1; \
        const float w17 = sc1 * (float)((pk1 >> 28) & 0xFu) + zp1; \
        const int base = (G) * 256 + tid * 8; \
        _Pragma("unroll") \
        for (int slot = 0; slot < HF_MQ4_O2_R8_SLOTS; ++slot) { \
            const float* x = x_rows[slot]; \
            if (x == nullptr) continue; \
            const float x0 = x[base + 0]; \
            const float x1 = x[base + 1]; \
            const float x2 = x[base + 2]; \
            const float x3 = x[base + 3]; \
            const float x4 = x[base + 4]; \
            const float x5 = x[base + 5]; \
            const float x6 = x[base + 6]; \
            const float x7 = x[base + 7]; \
            (ACC0)[slot] += w00 * x0 + w01 * x1 + w02 * x2 + w03 * x3 \
                + w04 * x4 + w05 * x5 + w06 * x6 + w07 * x7; \
            if (row1_valid) { \
                (ACC1)[slot] += w10 * x0 + w11 * x1 + w12 * x2 + w13 * x3 \
                    + w14 * x4 + w15 * x5 + w16 * x6 + w17 * x7; \
            } \
        } \
    } while (0)

    int g = 0;
    for (; g + 3 < group_walk; g += 4) {
        ACCUM_GROUP(g, acc00, acc10);
        ACCUM_GROUP(g + 1, acc01, acc11);
        ACCUM_GROUP(g + 2, acc02, acc12);
        ACCUM_GROUP(g + 3, acc03, acc13);
    }
    for (; g < group_walk; ++g) {
        switch (g & 3) {
        case 0:
            ACCUM_GROUP(g, acc00, acc10);
            break;
        case 1:
            ACCUM_GROUP(g, acc01, acc11);
            break;
        case 2:
            ACCUM_GROUP(g, acc02, acc12);
            break;
        default:
            ACCUM_GROUP(g, acc03, acc13);
            break;
        }
    }
#undef ACCUM_GROUP

    float sum0[HF_MQ4_O2_R8_SLOTS];
    float sum1[HF_MQ4_O2_R8_SLOTS];
#pragma unroll
    for (int slot = 0; slot < HF_MQ4_O2_R8_SLOTS; ++slot) {
        sum0[slot] = (acc00[slot] + acc01[slot])
            + (acc02[slot] + acc03[slot]);
        sum1[slot] = (acc10[slot] + acc11[slot])
            + (acc12[slot] + acc13[slot]);
    }
    // Full-wave reduction is deliberate: each output row has the exact same
    // lane association as the baseline O1×R16 kernel.
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int slot = 0; slot < HF_MQ4_O2_R8_SLOTS; ++slot) {
            sum0[slot] += __shfl_down(sum0[slot], offset, 32);
            sum1[slot] += __shfl_down(sum1[slot], offset, 32);
        }
    }

    if (tid == 0) {
#pragma unroll
        for (int slot = 0; slot < HF_MQ4_O2_R8_SLOTS; ++slot) {
            const int grouped_slot = slot_start + slot;
            if (grouped_slot < grouped_rows) {
                Y_grouped[(long long)grouped_slot * M + row0] = sum0[slot];
                if (row1_valid)
                    Y_grouped[(long long)grouped_slot * M + row1] = sum1[slot];
            }
        }
    }
}

#undef HF_MQ4_O2_R8_SLOTS
