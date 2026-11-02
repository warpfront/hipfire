// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Lane-table sibling of `dynamic_causal_conv_f32` (dynamic_conv_f32.hip) for
// the continuous-batching DFlash draft: ONE launch performs the DFlash2
// causal grouped dynamic convolution of every lane (request block) of the
// batched draft forward.
//
// The draft keeps all lanes' rows in one shared row scratch: lane i owns the
// contiguous rows [row0[i], row0[i] + rows[i]) of the row-major `input`
// ([row, hidden]), `output` ([row, hidden]) and `dynamic` ([row, stride]).
// The convolution is causal across the rows of ONE lane only (left-zero
// padded at the lane's first row), so a single launch over all rows would mix
// lane-boundary predecessors; every causal guard below therefore uses the
// LANE-LOCAL row `r`, never the global row. `base` is lane independent
// ([kernel_size, hidden], shared). blockIdx.z selects the lane; the lane
// table rides in the kernarg segment BY VALUE (no device table, no H2D
// upload).
//
// Byte-exactness vs one `dynamic_causal_conv_f32` launch per lane over that
// lane's row views (input/output/dynamic sub-views, rows = rows[i]): the
// body from `idx` on is the original's VERBATIM, with the same local
// idx/r/c/g and the same expressions. The only change is that the input,
// output and dynamic pointers are rebased by the lane's row0 (a pointer
// offset, not arithmetic). Each (lane, idx) thread is independent, so running
// lanes concurrently instead of one launch after another changes no byte.
//
// Grid [ceil(max_lane_rows * hidden / 256), 1, lanes], block [256, 1, 1],
// LDS 0. Blocks past a lane's rows * hidden (and lanes with rows <= 0)
// return before any memory access.

#define DRAFT_CONV_LANES_MAX 16

struct DraftConvTable {
    int row0[DRAFT_CONV_LANES_MAX];
    int rows[DRAFT_CONV_LANES_MAX];
};
static_assert(sizeof(DraftConvTable) == 128, "DraftConvTable ABI size");
static_assert(alignof(DraftConvTable) == 4, "DraftConvTable ABI alignment");

extern "C" __global__ void dynamic_causal_conv_f32_lanes(
    const float* __restrict__ input,
    const float* __restrict__ base,
    const float* __restrict__ dynamic,
    float* __restrict__ output,
    const DraftConvTable table,
    int hidden,
    int kernel_size,
    int groups,
    int group_size,
    int dynamic_row_stride,
    int dynamic_offset) {
    int lane = blockIdx.z;
    int rows = table.rows[lane];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * hidden;
    if (idx >= total) {
        return;
    }
    // Lane rebase: pointer offsets only.
    const size_t row0 = (size_t)table.row0[lane];
    input += row0 * (size_t)hidden;
    output += row0 * (size_t)hidden;
    dynamic += row0 * (size_t)dynamic_row_stride;

    int r = idx / hidden;
    int c = idx % hidden;
    int g = c / group_size;

    float acc = 0.0f;

    // Optimized DFlash2 path: kernel_size == 2.
    // Branch is uniform across the grid (kernel_size is scalar), so no divergence.
    if (kernel_size == 2) {
        // off = 0
        float b0 = base[c];
        float d0 = dynamic[r * dynamic_row_stride + dynamic_offset + g];
        float w0 = b0 + d0;
        float x0 = input[r * hidden + c];
        acc += w0 * x0;
        // off = 1, causal guard r >= 1 (lane-local row)
        if (r >= 1) {
            float b1 = base[hidden + c];
            float d1 = dynamic[r * dynamic_row_stride + dynamic_offset + groups + g];
            float w1 = b1 + d1;
            float x1 = input[(r - 1) * hidden + c];
            acc += w1 * x1;
        }
    } else {
        // Generic causal loop: left-zero-padded.
        for (int off = 0; off < kernel_size; ++off) {
            if (r < off) {
                continue;
            }
            float b = base[off * hidden + c];
            float d = dynamic[r * dynamic_row_stride + dynamic_offset + off * groups + g];
            float w = b + d;
            float x = input[(r - off) * hidden + c];
            acc += w * x;
        }
    }

    output[idx] = acc;
}
