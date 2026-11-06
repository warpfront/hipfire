// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Sequence-parallel QSA selection exchange. Each rank selects only its own rows of a
// prefill tile, packs the chosen block ids into a compact [rows][budget_blocks] i32
// buffer, all-gathers the compact rows, and expands the peer rows back into the full
// [rows][capacity] selection layout.
//
// The expand kernel reproduces qsa_select_batched's row layout
// (kernels/src/tensor_ops.hip) byte for byte. For tile row R (absolute, tile start
// position P = position_start):
//   visible = P + R + 1
//   rbc     = min(visible / compress, block_count)
//   chosen  = min(budget_blocks, rbc)
//   slot k*compress + s (k < chosen, s < compress) = block_k * compress + s
//   slots chosen*compress + t (t >= 0) = rbc*compress + t while that token is < visible
//   every other slot (up to capacity) = -1
//
// Every compact value is a block id >= 0 (the pack stores 0 for an unfilled slot), so the
// SP all-gather wire's 0xFFFFFFFF EMPTY marker can never occur in the payload.
//
// Both kernels run blockDim.x == 256 over a flat grid of rows * width work items.

#include <hip/hip_runtime.h>

// compact[R][j] = selected[R][j * compress] < 0 ? 0 : selected[R][j * compress] / compress
// for R in [row0, row0 + rows), j < budget_blocks. Width is budget_blocks.
extern "C" __global__ __launch_bounds__(256) void qsa_sp_pack_blocks(
    const int* __restrict__ selected, int* __restrict__ compact, int row0, int rows,
    int capacity, int budget_blocks, int compress) {
    const size_t idx = (size_t)blockIdx.x * 256 + threadIdx.x;
    const size_t total = (size_t)rows * (size_t)budget_blocks;
    if (idx >= total) return;
    const size_t r = (size_t)row0 + idx / (size_t)budget_blocks;
    const int j = (int)(idx % (size_t)budget_blocks);
    const int v = selected[r * (size_t)capacity + (size_t)j * (size_t)compress];
    compact[r * (size_t)budget_blocks + (size_t)j] = v < 0 ? 0 : v / compress;
}

// selected rows [row0, row0 + rows) rebuilt from compact. Width is capacity; every slot is
// written exactly once.
extern "C" __global__ __launch_bounds__(256) void qsa_sp_expand_blocks(
    const int* __restrict__ compact, int* __restrict__ selected, int row0, int rows,
    int position_start, int block_count, int budget_blocks, int compress, int capacity) {
    const size_t idx = (size_t)blockIdx.x * 256 + threadIdx.x;
    const size_t total = (size_t)rows * (size_t)capacity;
    if (idx >= total) return;
    const size_t r = (size_t)row0 + idx / (size_t)capacity;
    const int i = (int)(idx % (size_t)capacity);
    const int visible = position_start + (int)r + 1;
    const int rbc = min(visible / compress, block_count);
    const int chosen = min(budget_blocks, rbc);
    int out;
    if (i < chosen * compress) {
        out = compact[r * (size_t)budget_blocks + (size_t)(i / compress)] * compress + i % compress;
    } else {
        const int tail_pos = rbc * compress + (i - chosen * compress);
        out = tail_pos < visible ? tail_pos : -1;
    }
    selected[r * (size_t)capacity + (size_t)i] = out;
}
