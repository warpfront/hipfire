// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Row-major F16-weight x F32-input GEMV with a fused F32 bias:
//
//   y[m] = bias[m] + sum_k W_f16[m, k] * x_f32[k]
//
// Same weight layout as `gemv_f16_xf32` ([M, K] F16 row-major, one row per
// output) and the same F32 input/accumulate contract; this entry adds the
// bias to the store and reads the weight 8 halves at a time.
//
// Use case: the FLUX MMDiT modulation linears. `double_blocks.*.{img,txt}_mod
// .lin` is [6d, d] and `single_blocks.*.modulation.lin` is [3d, d], and both
// are applied to ONE d-wide vector (`silu(vec)`) per denoise step. Routing
// those through the 128-row WMMA macro-tile computes 127 padding rows for
// every real one: the tile streams the same weight bytes but does 128x the
// FLOPs, which on a bandwidth-limited part turns a weight-stream-bound op
// into a compute-bound one. A GEMV reads the same bytes and does 1x the work.
//
// `bias` may be null, in which case only the dot product is stored.
//
// Launch: grid = [ceil(M / GEMV_MB_WAVES), 1, 1],
//         block = [GEMV_MB_WAVES * 32, 1, 1]   (wave32; one wave per row)
//
// Per row: K F16 weight loads + K F32 input loads + K FMA. The weight stream
// is unique per row and the input stream is shared by every row, so this is
// bound by the M*K*2 weight bytes.

// Waves (= output rows) per workgroup.
#define GEMV_MB_WAVES 4

// 8 halves = 16 B, the widest naturally-aligned load a lane can issue against
// a row whose K is a multiple of 8 (every FLUX modulation linear has K = d =
// 3072). A ragged K falls back to the scalar stride-32 loop.
typedef _Float16 gemv_half8 __attribute__((ext_vector_type(8)));

__launch_bounds__(GEMV_MB_WAVES * 32)
extern "C" __global__ void gemv_f16_bias_xf32(
    const _Float16* __restrict__ W,     // [M, K] F16, row-major
    const float*    __restrict__ x,     // [K] F32
    const float*    __restrict__ bias,  // [M] F32, may be null
    float*          __restrict__ y,     // [M] F32
    int M, int K
) {
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int row = blockIdx.x * GEMV_MB_WAVES + wave;
    if (row >= M) return;

    const _Float16* w_row = W + (size_t)row * (size_t)K;

    float sum = 0.0f;
    if ((K & 7) == 0) {
        const int chunks = K >> 3;
        // ONE accumulator on purpose. An 8-wide split of the FMA dependency
        // chain (one partial per element of the 16 B load) measured 98.8 ms
        // against this loop's 93.1 ms for the 76 launches on gfx1150 — the
        // extra live registers cost more occupancy than the shorter chain
        // buys, because the loop is weight-bandwidth bound, not FMA bound.
        for (int c = lane; c < chunks; c += 32) {
            const gemv_half8 w = *(const gemv_half8*)(w_row + ((size_t)c << 3));
            const float* xp = x + ((size_t)c << 3);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                sum += (float)w[j] * xp[j];
            }
        }
    } else {
        for (int i = lane; i < K; i += 32) {
            sum += (float)w_row[i] * x[i];
        }
    }

    // Wave-shuffle reduction (wave32), same tree as `gemv_f16_xf32`.
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (lane == 0) {
        y[row] = bias ? (sum + bias[row]) : sum;
    }
}
