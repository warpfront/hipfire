// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Lane-table sibling of `conv1d_silu_split_f32` (conv1d_silu_split.hip) for
// the continuous-batching DFlash verify trunk: ONE launch performs the
// DeltaNet conv1d + SiLU + Q/K/V split of every lane (request) of the chunk.
//
// The verify trunk keeps all lanes' rows in one shared scratch: lane i owns
// the contiguous rows [row0[i], row0[i] + n_tokens[i]) of the row-major
// `input` ([row, n_channels]) and of `q_out`/`k_out` ([row, k_dim]) /
// `v_out` ([row, v_dim]), and has its OWN conv ring `state[i]` ([channel, 3]).
// grid.y selects the lane; the lane table rides in the kernarg segment BY
// VALUE (no device table, no H2D upload), like tape_copy_batched.
//
// Byte-exactness vs one `conv1d_silu_split_f32` launch per lane over that
// lane's row view (grid.y = 1, state_base = 0, row = t):
//   * the per-token loop body is the original's VERBATIM: the same
//     expression `weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
//     + weight[c * 4 + 1] * s1 + weight[c * 4] * s2`, the same SiLU, the
//     same state-ring update order, the same Q/K/V split. The only changes
//     are where the lane's row base / state / token count come from (the
//     table instead of kernel arguments / grid.y arithmetic), none of which
//     enters the arithmetic. Keeping the weight[] indexing in place (not
//     hoisted) is LOAD-BEARING for byte-exact output: hoisting the loads
//     changes the compiler's FMA fusion decisions and nudges results by ~1 ULP.
//   * a (channel, lane) thread is sequential over that lane's tokens exactly
//     as the per-lane launch's thread for that channel; channels are
//     independent and lanes touch disjoint rows and disjoint state rings, so
//     running lanes concurrently instead of one launch after another changes
//     no byte.
//
// The host passes grid.y = number of lanes (at most CONV_LANES_MAX).

#define CONV_LANES_MAX 16

struct ConvLaneTable {
    unsigned long long state[CONV_LANES_MAX];
    int row0[CONV_LANES_MAX];
    int n_tokens[CONV_LANES_MAX];
};

extern "C" __global__ void conv1d_silu_split_f32_lanes(
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const ConvLaneTable t,
    int k_dim,
    int v_dim
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = blockIdx.y;
    int n_channels = 2 * k_dim + v_dim;
    if (c >= n_channels) return;

    float* __restrict__ state = (float*)t.state[lane];
    const int n_tokens = t.n_tokens[lane];
    const int row0 = t.row0[lane];

    // Same loop as `conv1d_silu_split_f32` (see its comment): weight[]
    // indexing stays in place.
    for (int t_ = 0; t_ < n_tokens; t_++) {
        int row = row0 + t_;
        int state_base = 0;
        float x = input[row * n_channels + c];
        float s0 = state[state_base + c * 3];
        float s1 = state[state_base + c * 3 + 1];
        float s2 = state[state_base + c * 3 + 2];

        float y = weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
                + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;

        // SiLU
        float result = y / (1.0f + expf(-y));

        // Update ring buffer state
        state[state_base + c * 3 + 2] = s1;
        state[state_base + c * 3 + 1] = s0;
        state[state_base + c * 3]     = x;

        // Write to Q/K/V based on channel index.
        if (c < k_dim) {
            q_out[row * k_dim + c] = result;
        } else if (c < 2 * k_dim) {
            k_out[row * k_dim + (c - k_dim)] = result;
        } else {
            v_out[row * v_dim + (c - 2 * k_dim)] = result;
        }
    }
}
