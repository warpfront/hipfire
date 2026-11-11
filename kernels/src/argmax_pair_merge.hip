// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Merge of two vocabulary-shard argmaxes (rank 0's packed [rows][w0] shard and the
// peer's packed [rows][w1] shard, each already reduced by argmax_f32 to a local id)
// into the argmax_f32 answer over the concatenated row: the first index of the
// largest finite value, 0 when no value is finite. Shard 0 holds the lower global
// ids, so `>=` keeps it on equal values (+0 == -0 included). One thread per row;
// `out_top1` may alias `local_top1` (i0 is read before the row's write).
extern "C" __global__ void argmax_f32_pair_merge(
    const float* local_logits,
    const int* local_top1,
    const float* peer_logits,
    const int* peer_top1,
    int* out_top1,
    int rows,
    int w0,
    int w1) {
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    const int i0 = local_top1[row];
    const int i1 = peer_top1[row];
    const float v0 = local_logits[(size_t)row * (size_t)w0 + (size_t)i0];
    const float v1 = peer_logits[(size_t)row * (size_t)w1 + (size_t)i1];
    const bool f0 = isfinite(v0);
    const bool f1 = isfinite(v1);
    out_top1[row] = (f0 && (!f1 || v0 >= v1)) ? i0 : (f1 ? w0 + i1 : 0);
}
