// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// FLUX.1 2D axial RoPE, in-place — the tuned form of `rope_2d_flux_f32.hip`.
// Same math, same ABI, same in-place semantics; three changes, each of which
// removes work rather than trading accuracy for speed.
//
// The seed kernel measured 4.93 ms per call at real FLUX geometry on gfx1151
// (4096 image rows x 24 heads x head_dim 128), which is ~20 GB/s against a
// 256 GB/s roof — about 12x slower than a plain strided copy of the same
// bytes, while every other elementwise kernel in this forward reaches
// 220-228 GB/s. It was never memory-bound. Three causes:
//
//   1. A DOUBLE-PRECISION `pow(theta, 2p/d_axis)` per pair per thread — 64 per
//      thread — on a consumer RDNA part with no fast FP64 path. The exponent
//      depends only on (axis, pair index): it is the same value for every row
//      and every head, and never depends on the data. Here it becomes
//      `expf(-(2p/d_axis) * logf(theta))`, one hardware `v_exp_f32` per pair.
//      This is the whole ballgame; the other two are cheap extras.
//
//   2. Axis 0 carries position 0.0 (the channel axis is the identity in the
//      BFL construction), so its pairs rotate by angle 0 — `c=1, s=0`, which
//      reads and rewrites each value unchanged. The seed kernel paid full
//      cost, `pow` included, for 8 of its 64 pairs. Skipping a zero-position
//      axis is BIT-EXACT (the rotation is exactly the identity) and also drops
//      that axis's traffic.
//
//   3. One thread per (row, head) made adjacent lanes `head_dim` floats apart,
//      so a wave touched 32 separate 512-byte regions. One thread per
//      (row, head, pair) with a `float2` load/store puts adjacent lanes on
//      adjacent 8-byte pairs, so a wave32 covers 256 contiguous bytes.
//
// Accuracy: `expf`/`logf` in f32 carry ~1e-7 relative error against the f64
// `pow`, and the largest angle at FLUX geometry is ~63 rad, so the angle error
// is ~6e-6 rad and the cos/sin error is of the same order — three orders below
// the f16 GEMM rounding (~3e-4) already present in this forward. The identity
// skip and the thread remap are both exactly bit-preserving.
//
// `sincosf` (accurate, with range reduction) is used rather than `__sincosf`:
// angles reach ~63 rad, beyond where the raw hardware sin/cos stay accurate,
// and the kernel has bandwidth headroom to spare after change 1.
extern "C" __global__ __launch_bounds__(256) void rope_2d_flux_f32_fast(
    float* __restrict__ x,
    const float* __restrict__ ids,   // nullable: [n_img][4] positions per image row
    int row_offset, int n_img, int heads, int hd,
    int grid_w,
    int ax0, int ax1, int ax2, int ax3,
    double theta) {

    const int pairs0 = ax0 >> 1;
    const int pairs1 = ax1 >> 1;
    const int pairs2 = ax2 >> 1;
    const int pairs3 = ax3 >> 1;
    const int pairs_total = pairs0 + pairs1 + pairs2 + pairs3;
    if (pairs_total <= 0) return;

    const long total = (long)n_img * (long)heads * (long)pairs_total;
    const long idx = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
    if (idx >= total) return;

    // Adjacent lanes get adjacent pair indices, hence adjacent memory.
    const int p_global = (int)(idx % (long)pairs_total);
    const long rest = idx / (long)pairs_total;
    const int h = (int)(rest % (long)heads);
    const long t = rest / (long)heads; // image row within the n_img slice

    // Positions for this image row: an id table (if provided) or the
    // derived model pos = (0, row, col, 0) — axis 0 (and axis 3) are the
    // identity channel axes in that derived case.
    float pos0, pos1, pos2, pos3;
    if (ids != nullptr) {
        pos0 = ids[t * 4 + 0]; pos1 = ids[t * 4 + 1]; pos2 = ids[t * 4 + 2]; pos3 = ids[t * 4 + 3];
    } else {
        pos0 = 0.0f; pos1 = (float)(t / (long)grid_w); pos2 = (float)(t % (long)grid_w); pos3 = 0.0f;
    }

    // Which axial region this pair falls in, and that axis's position.
    float p_axis; int p, d_axis;
    if (p_global < pairs0) { p_axis = pos0; p = p_global; d_axis = ax0; }
    else if (p_global < pairs0 + pairs1) { p_axis = pos1; p = p_global - pairs0; d_axis = ax1; }
    else if (p_global < pairs0 + pairs1 + pairs2) { p_axis = pos2; p = p_global - pairs0 - pairs1; d_axis = ax2; }
    else { p_axis = pos3; p = p_global - pairs0 - pairs1 - pairs2; d_axis = ax3; }

    // Angle 0 means c=1, s=0, i.e. exactly the identity — skip the pair
    // entirely rather than read and rewrite it unchanged. Always true for
    // axis 0 and axis 3 in the derived-position case, and true for row 0 /
    // column 0 of the image grid.
    if (p_axis == 0.0f) return;

    // theta^(-2p/d_axis) without the f64 pow. The exponent depends only on
    // (axis, pair), never on the data.
    const float inv_freq =
        expf(-((2.0f * (float)p) / (float)d_axis) * logf((float)theta));
    const float angle = p_axis * inv_freq;

    float c, s;
    sincosf(angle, &s, &c);

    const long base = ((row_offset + t) * (long)heads + (long)h) * (long)hd;
    float2* pair = (float2*)(x + base + 2 * (long)p_global);
    const float2 ab = *pair;
    pair->x = ab.x * c - ab.y * s;
    pair->y = ab.x * s + ab.y * c;
}
