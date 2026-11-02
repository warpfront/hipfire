// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Penalty stage of the host logit policy (`sampler::apply_logit_policy_cpu`:
// repeat, then presence/frequency) applied in place to `n_rows` logit rows.
//
// The host builds the table (`sampler::PenaltyTable`): for every distinct
// in-vocabulary token of a row's history window it computes the repeat factor
// `penalty^(count * recency)` capped at 1.5 and the subtrahend
// `frequency * count + presence` with the CPU's own f32 arithmetic (glibc
// `powf` included), so the device does no transcendental work. What is left
// here is one f32 division or multiplication and one subtraction per entry.
// HIP compiles f32 division correctly rounded by default, and the pragma
// below keeps the multiply-then-subtract from contracting into an FMA, so
// each step rounds exactly like the CPU's: the rows end bit-identical to the
// host policy (`crates/hipfire-runtime/tests/penalty_table_gpu.rs`).
//
// Table layout (u32 words): `row_end[0..n_rows]` (cumulative entry count;
// row r owns entries [row_end[r-1], row_end[r])), then one entry per token:
// `{token, factor bits, subtrahend bits}`. Tokens are distinct within a row,
// so the threads of a block never write the same logit.
//
// flags bit 0: apply the repeat factor (positive logit / factor, else
// * factor). flags bit 1: subtract the subtrahend.
//
// Launch: grid = n_rows, any block size (128).

#pragma clang fp contract(off)

extern "C" __global__ void logit_penalty_table_rows(
    float* __restrict__ logits,
    const unsigned int* __restrict__ table,
    int vocab_size,
    int n_rows,
    unsigned int flags
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const unsigned int begin = row == 0 ? 0u : table[row - 1];
    const unsigned int end = table[row];
    const unsigned int* entries = table + n_rows;
    float* out = logits + (size_t)row * (size_t)vocab_size;
    for (unsigned int e = begin + threadIdx.x; e < end; e += blockDim.x) {
        const unsigned int tok = entries[3u * e];
        if (tok >= (unsigned int)vocab_size) continue;
        float v = out[tok];
        if (flags & 1u) {
            const float factor = __uint_as_float(entries[3u * e + 1u]);
            v = (v > 0.0f) ? (v / factor) : (v * factor);
        }
        if (flags & 2u) {
            v = v - __uint_as_float(entries[3u * e + 2u]);
        }
        out[tok] = v;
    }
}
