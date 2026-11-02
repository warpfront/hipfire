// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Sparse row compaction of raw f32 probabilities for the sampled-head DFlash
// draft (FAST_SAMPLE, CAP256).
//
// One block (256 threads) per row. For row `r` of `probs[rows, vocab]` it
// keeps every column `c` with `p >= tau[r]` (per-row device threshold; NaN
// compares false, so NaN is excluded; `tau[r] <= 0` keeps every non-NaN
// column) in ASCENDING column order:
//
//   counts[r]               = TRUE number of kept columns (even when it
//                             exceeds `capacity`)
//   idx[r * capacity + k]   = column id of the k-th kept column (k < capacity)
//   val[r * capacity + k]   = the raw f32 bits of probs[r * vocab + idx]
//
// Ranks >= capacity are counted but not stored. Slots [min(count, capacity),
// capacity) are left untouched.
//
// Order is deterministic: thread `t` owns the contiguous column segment
// [t * seg, min((t + 1) * seg, vocab)) with seg = ceil(vocab / 256); a block
// exclusive scan of the per-thread counts gives each segment its first rank,
// so rank order == column order. No atomics, no float arithmetic.
//
// Kernarg order: probs, tau (f32[rows]), counts, idx, val, vocab, capacity.
// Grid [rows, 1, 1], block 256.

#include <hip/hip_runtime.h>

#define SRC_THREADS 256

extern "C" __global__ __launch_bounds__(SRC_THREADS) void sampled_rows_compact_f32(
    const float* __restrict__ probs,
    const float* __restrict__ tau_rows,
    int* __restrict__ counts,
    int* __restrict__ idx,
    float* __restrict__ val,
    int vocab,
    int capacity)
{
    __shared__ int scan[SRC_THREADS];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* __restrict__ p = probs + (size_t)row * (size_t)vocab;
    const float tau = tau_rows[row];

    const int seg = (vocab + SRC_THREADS - 1) / SRC_THREADS;
    const int start = tid * seg;               // may be >= vocab: empty segment
    const int end = min(start + seg, vocab);

    int mine = 0;
    for (int c = start; c < end; ++c) {
        mine += (p[c] >= tau) ? 1 : 0;
    }

    // Inclusive Hillis-Steele scan in shared memory (wave-size agnostic).
    scan[tid] = mine;
    __syncthreads();
    for (int off = 1; off < SRC_THREADS; off <<= 1) {
        int add = (tid >= off) ? scan[tid - off] : 0;
        __syncthreads();
        scan[tid] += add;
        __syncthreads();
    }
    const int excl = scan[tid] - mine;

    if (tid == SRC_THREADS - 1) {
        counts[row] = scan[tid];
    }

    if (excl >= capacity) return;              // whole segment past the cap
    int rank = excl;
    int* __restrict__ idx_row = idx + (size_t)row * (size_t)capacity;
    float* __restrict__ val_row = val + (size_t)row * (size_t)capacity;
    for (int c = start; c < end; ++c) {
        const float x = p[c];
        if (x >= tau) {
            if (rank >= capacity) break;
            idx_row[rank] = c;
            val_row[rank] = x;
            ++rank;
        }
    }
}
