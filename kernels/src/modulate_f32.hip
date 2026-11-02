// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// FLUX adaLN-Zero modulation affine, broadcast over rows, in place-capable if
// `out == x`. Seed for the MMDiT forward. The CPU reference
// applies `h[r,i] = (1.0 + scale[i]) * normed[r,i] + shift[i]` where
// `scale`/`shift` are `d`-wide row vectors shared by every row — the
// double-block (twice per stream), single-block, and final-head modulations
// all reduce to this one op. `out` may alias `x` (in-place) but NOT `shift`
///
// Layouts: x [n_rows, d], shift [d], scale [d], out [n_rows, d], row-major.
extern "C" __global__ void modulate_f32(
    const float* __restrict__ x,
    const float* __restrict__ shift,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int n_rows, int d) {

    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    long total = (long)n_rows * d;
    if (idx < 0 || (long)idx >= total) return;

    int i = idx % d;
    out[idx] = x[idx] * (1.0f + scale[i]) + shift[i];
}
