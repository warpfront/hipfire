// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — QT53 grouped top-10 down with same-expert slot weight reuse.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Portable F32 parity consumer for grouped qt53 down.
//
// The arithmetic deliberately follows
// gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded: all 32 lanes own
// their four-element K slice, accumulate in F32, and reduce by the same
// wave32 shuffle sequence.  The only structural change is that each lane
// reuses its decoded weight values across the sixteen same-expert route slots
// in the tile.  The old grouped kernel used sixteen lanes per slot plus a
// shared-memory scalar sum, which has a different reduction association.
//
// The logical K tail, group padding, slot sentinel handling, and output layout
// are unchanged from both consumers.  Y_grouped remains unweighted; sealed
// combine applies route weights exactly once.

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_mq4g128v2_moe_grouped_top10_multirow(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index,
    const float* __restrict__ X_src,
    float* __restrict__ Y_grouped,
    int M,
    int K,
    int x_row_div,
    int grouped_rows,
    int x_src_rows,
    int n_exp
) {
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int tile = blockIdx.y;
    const int slot_start = tile * 16;
    if (row >= M || slot_start >= grouped_rows || tid >= 32) return;

    const int expert = expert_tile_ids[tile];
    float acc[16];
    const float* x_rows[16];
    #pragma unroll
    for (int slot = 0; slot < 16; ++slot) {
        acc[slot] = 0.0f;
        x_rows[slot] = nullptr;
    }

    if (expert >= 0 && expert < n_exp && K > 0 && x_row_div > 0) {
        #pragma unroll
        for (int slot = 0; slot < 16; ++slot) {
            const int grouped_slot = slot_start + slot;
            if (grouped_slot >= grouped_rows) continue;
            const int flat = sorted_slot_index[grouped_slot];
            if (flat < 0) continue;
            const int x_row = flat / x_row_div;
            if (x_row >= 0 && x_row < x_src_rows)
                x_rows[slot] = X_src + (long long)x_row * K;
        }

        const char* A = reinterpret_cast<const char*>(expert_ptrs[expert]);
        const int groups = (K + 127) / 128;
        const long long row_stride = (long long)groups * 68;
        const char* row_ptr = A + (long long)row * row_stride;

        // This is the indexed kernel's lane decomposition exactly.  A lane
        // decodes its two payload bytes once per group, then reuses those
        // four decoded weights for every valid slot in this expert tile.
        for (int group = 0; group < groups; ++group) {
            const char* gp = row_ptr + (long long)group * 68;
            const unsigned short scale_bits =
                *(const unsigned short*)(gp + 0);
            const unsigned short zero_bits =
                *(const unsigned short*)(gp + 2);
            const float scale = __half2float(__ushort_as_half(scale_bits));
            const float zero = __half2float(__ushort_as_half(zero_bits));
            const unsigned char* payload = (const unsigned char*)(gp + 4);
            const int base = group * 128 + tid * 4;
            const unsigned char b0 = payload[tid * 2];
            const unsigned char b1 = payload[tid * 2 + 1];
            const int i0 = base;
            const int i1 = base + 1;
            const int i2 = base + 2;
            const int i3 = base + 3;
            #pragma unroll
            for (int slot = 0; slot < 16; ++slot) {
                const float* x = x_rows[slot];
                if (x == nullptr) continue;
                const float x0 = i0 < K ? x[i0] : 0.0f;
                const float x1 = i1 < K ? x[i1] : 0.0f;
                const float x2 = i2 < K ? x[i2] : 0.0f;
                const float x3 = i3 < K ? x[i3] : 0.0f;
                acc[slot] += (scale * (float)(b0 & 0xF) + zero) * x0
                    + (scale * (float)(b0 >> 4) + zero) * x1
                    + (scale * (float)(b1 & 0xF) + zero) * x2
                    + (scale * (float)(b1 >> 4) + zero) * x3;
            }
        }
    }

    // Match the indexed decode kernel's wave32 association rather than the
    // legacy grouped kernel's sixteen-lane shared-memory reduction.
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int slot = 0; slot < 16; ++slot)
            acc[slot] += __shfl_down(acc[slot], offset, 32);
    }

    if (tid == 0) {
        #pragma unroll
        for (int slot = 0; slot < 16; ++slot) {
            const int grouped_slot = slot_start + slot;
            if (grouped_slot < grouped_rows)
                Y_grouped[(long long)grouped_slot * M + row] = acc[slot];
        }
    }
}

