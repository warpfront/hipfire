// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Flash-Next expert-parallel row all-gather with explicit slot owners (copy
// only), the balanced-ownership twin of `ep_gather_rows_by_owner`
// (kernels/src/ep_gather_rows_by_owner.hip; its module is unchanged).
//
// Same grid, same body, same bit-exact uint4 copy. The only difference is where
// the owner comes from: `slot_owner[s]` (i32, device memory, written by the
// `qwen4_ep_owner_assign` kernel of the same call) names the rank that
// computed slot `s`, instead of `slot_experts[s] % n_ranks`. Rows owned by
// `self_rank`, and slots with a negative owner, are never written.
//
// grid  = (ceil(n16 / blockDim.x), n_slots), block = (256, 1, 1)
// src0..src7: row-buffer base of rank 0..7 (entries >= n_ranks may be null)
// dst:        this rank's row-buffer base

extern "C" __global__ void ep_gather_rows_by_slot_owner(
    const int *__restrict__ slot_owner, const uint4 *src0, const uint4 *src1,
    const uint4 *src2, const uint4 *src3, const uint4 *src4, const uint4 *src5,
    const uint4 *src6, const uint4 *src7, uint4 *dst, int n_slots, int n_ranks,
    int self_rank, int n16) {
    const int slot = (int)blockIdx.y;
    if (slot >= n_slots)
        return;
    const int owner = slot_owner[slot];
    if (owner < 0 || owner >= n_ranks || owner >= 8 || owner == self_rank)
        return;
    const uint4 *src;
    switch (owner) {
    case 0: src = src0; break;
    case 1: src = src1; break;
    case 2: src = src2; break;
    case 3: src = src3; break;
    case 4: src = src4; break;
    case 5: src = src5; break;
    case 6: src = src6; break;
    default: src = src7; break;
    }
    const size_t row = (size_t)slot * (size_t)n16;
    const size_t stride = (size_t)gridDim.x * (size_t)blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)n16; i += stride)
        dst[row + i] = src[row + i];
}
