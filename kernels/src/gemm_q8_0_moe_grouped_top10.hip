// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — Qwen4 Q8F16/qt3 grouped down projection.

#include <hip/hip_runtime.h>

// Correctness-first grouped Q8 GEMM.  Each block computes one output row and
// sixteen sorted route slots.  Sixteen lanes per slot reduce the K dimension;
// this keeps the grouped permutation contract explicit and works for Qwen4's
// K=640 down projection without padding it to an unrelated quant geometry.
//
// Y_grouped remains unweighted.  The top-10 grouped combine applies route
// weights exactly once after this launch.
__launch_bounds__(256, 2)
extern "C" __global__ void gemm_q8_0_moe_grouped_top10(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ expert_tile_ids,             // [grouped_rows/16]
    const int* __restrict__ sorted_slot_index,           // [grouped_rows]
    const float* __restrict__ X_src,                     // [x_src_rows, K]
    float* __restrict__ Y_grouped,                       // [grouped_rows, M]
    int M,
    int K,
    int x_row_div,
    int grouped_rows,
    int x_src_rows,
    int n_exp
) {
    const int row = blockIdx.x;
    const int tile = blockIdx.y;
    const int lane = threadIdx.x;
    const int slot = lane >> 4;
    const int sublane = lane & 15;
    if (row >= M || tile * 16 >= grouped_rows || lane >= 256) return;

    const int expert = expert_tile_ids[tile];
    const int grouped_slot = tile * 16 + slot;
    int x_row = -1;
    if (grouped_slot < grouped_rows) {
        const int flat = sorted_slot_index[grouped_slot];
        if (flat >= 0 && x_row_div > 0) {
            x_row = flat / x_row_div;
            if (x_row < 0 || x_row >= x_src_rows) x_row = -1;
        }
    }

    float partial = 0.0f;
    if (expert >= 0 && expert < n_exp && x_row >= 0 && K > 0 && (K & 31) == 0) {
        const unsigned char* __restrict__ weights =
            reinterpret_cast<const unsigned char*>(expert_ptrs[expert]);
        const int blocks = K >> 5;
        const size_t row_base = (size_t)row * blocks * 34;
        const float* x = X_src + (size_t)x_row * K;
        for (int k = sublane; k < K; k += 16) {
            const int block = k >> 5;
            const unsigned char* qblock = weights + row_base + (size_t)block * 34;
            const float scale = (float)(*reinterpret_cast<const _Float16*>(qblock));
            const signed char q = reinterpret_cast<const signed char*>(qblock + 2)[k & 31];
            partial += scale * (float)q * x[k];
        }
    }

    __shared__ float partials[16][16];
    partials[slot][sublane] = partial;
    __syncthreads();
    if (sublane == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 16; ++i) sum += partials[slot][i];
        if (grouped_slot < grouped_rows)
            Y_grouped[(size_t)grouped_slot * M + row] = sum;
    }
}
