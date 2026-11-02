// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Additive band mask for bidirectional sliding-window attention scores
// (embeddinggemma2, arch 18): a sliding layer attends to every position with
// |row - col| <= window, so every score outside that band is set to a large
// finite negative (not -inf: the downstream row softmax computes
// exp(x - rowmax), and a finite near-min value keeps that exactly 0 without
// an inf-inf NaN on any degenerate row). The diagonal is always in band, so
// no row is fully masked.
//
// Applied to the materialized [rows, cols] f32 score matrix of the eager
// attention decomposition, after the QK^T GEMM and before
// `attention_rows_softmax_f16` (whose own [valid_start, valid_end) column
// bound still handles sequence padding — the two masks compose).
extern "C" __global__ void attention_scores_band_mask_f32(
    float* __restrict__ scores,
    int rows,
    int cols,
    int window
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) return;
    int d = row - col;
    if (d > window || -d > window) {
        scores[(long)row * cols + col] = -3.0e38f;
    }
}
