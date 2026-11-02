// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// FLUX.1 2D axial RoPE, in-place — seed kernel for the MMDiT forward's image
// rotation. Mirrors the CPU reference `flux::rope_2d` exactly:
// image token rows get positions (0, row, col) (channel axis 0 = identity),
// and `head_dim` splits into the `axes_dim` [ax0, ax1, ax2] axial regions
// ([16, 56, 56] for real FLUX dev/schnell, summing to head_dim 128). Within
// each region, interleaved pairs `(x[i], x[i+1])` are rotated by the standard
// `(a·c − b·s, a·s + b·c)` around angle `pos[axis] / theta^(2p/d_axis)`.
//
// Text rows must NOT be rotated: the caller passes `row_offset` = index of
// the first image row within the `[n_all, heads*hd]` buffer, and only rows
// `[row_offset, row_offset + n_img)` are touched. In the BFL text-first
// concat, image rows sit at the END, so `row_offset` = n_text.
//
// One thread per (row, head); each loops the 3 axial regions and their
// interleaved pairs. Correctness-first — a table-lookup / fused variant is
// Phase-5 work.
extern "C" __global__ void rope_2d_flux_f32(
    float* __restrict__ x,
    const float* __restrict__ ids,   // nullable: [n_img][4] positions per image row
    int row_offset, int n_img, int heads, int hd,
    int grid_w,
    int ax0, int ax1, int ax2, int ax3,
    double theta) {
    int stride = heads * hd;
    long total = (long)n_img * heads;
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < 0 || idx >= total) return;
    int h = idx % heads;
    int t = idx / heads;
    float pos[4];
    if (ids != nullptr) {
        pos[0] = ids[t * 4 + 0]; pos[1] = ids[t * 4 + 1];
        pos[2] = ids[t * 4 + 2]; pos[3] = ids[t * 4 + 3];
    } else {
        pos[0] = 0.0f; pos[1] = (float)(t / grid_w); pos[2] = (float)(t % grid_w); pos[3] = 0.0f;
    }
    int axes[4] = { ax0, ax1, ax2, ax3 };
    int base = ((size_t)(row_offset + t) * heads + h) * hd;
    int region = 0;
    for (int aa = 0; aa < 4; ++aa) {
        int d_axis = axes[aa];
        int n_pairs = d_axis / 2;
        float p_axis = pos[aa];
        for (int p = 0; p < n_pairs; ++p) {
            double angle = (double)p_axis / pow(theta, (2.0 * (double)p) / (double)d_axis);
            float c = cosf((float)angle);
            float s = sinf((float)angle);
            int i = base + 2 * (region + p);
            float a = x[i];
            float b = x[i + 1];
            x[i]     = a * c - b * s;
            x[i + 1] = a * s + b * c;
        }
        region += n_pairs;
    }
}
