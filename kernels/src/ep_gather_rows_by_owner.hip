// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Flash-Next expert-parallel row all-gather (copy only).
//
// Every rank runs the unchanged single-device program, so its expanded-row
// buffer holds `n_slots` rows of `n16` uint4 each. Slot `s` is routed to expert
// `slot_experts[s]`, which lives on rank `e % n_ranks`. This kernel, launched on
// each rank, copies every row owned by ANOTHER rank from that owner's buffer
// (peer-mapped VRAM) into the local buffer at the same row offset. Rows owned by
// `self_rank`, and slots whose expert id is outside `[0, n_experts)` (the router's
// -1 marker for a non-finite row, or any corrupt id), are never written.
//
// Pure uint4 load/store: bit-exact for -0.0, NaN payloads and denormals (no
// arithmetic, no float typing).
//
// grid  = (ceil(n16 / blockDim.x), n_slots), block = (256, 1, 1)
// src0..src7: row-buffer base of rank 0..7 (entries >= n_ranks may be null)
// dst:        this rank's row-buffer base
// n_experts:  expert ids are valid in `[0, n_experts)`; any other id owns nothing

extern "C" __global__ void ep_gather_rows_by_owner(
    const int *__restrict__ slot_experts, const uint4 *src0, const uint4 *src1,
    const uint4 *src2, const uint4 *src3, const uint4 *src4, const uint4 *src5,
    const uint4 *src6, const uint4 *src7, uint4 *dst, int n_slots, int n_ranks,
    int self_rank, int n16, int n_experts) {
    const int slot = (int)blockIdx.y;
    if (slot >= n_slots)
        return;
    const int e = slot_experts[slot];
    if (e < 0 || e >= n_experts)
        return;
    const int owner = e % n_ranks;
    if (owner >= 8 || owner == self_rank)
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
