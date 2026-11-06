// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// LFM3-VL vision conv layer: double-gated depthwise 2D convolution.
//
// Mirrors `Lfm3VlVisionConv2d` (transformers `modeling_lfm3_vl.py`):
//   B, C, x = in_proj(hidden).chunk(3)          // bcx = [n, 3*hidden] = [B | C | x]
//   gated   = B * x                              // per-token gate
//   conv    = depthwise_conv2d(gated, k=3, pad=1, groups=hidden)  // over the patch grid
//   out     = C * conv                           // post-gate, pre out_proj
//
// The conv runs over the FULL raster patch grid (grid_h, grid_w) with zero
// padding ('same', output grid == input grid). Patches arrive in
// (gy, gx, sy, sx) spatial-merge-grouped order (the order `extract_patches`
// emits and the patchifier consumes); the kernel decodes each token's raster
// (py, px) on the fly so no host-side reshape/permute is needed.
//
// For a single image num_frames = 1 and N = grid_h * grid_w. Multi-frame
// (video) packing is Phase 3+; this kernel assumes one frame per call.
//
// Layouts:
//   bcx    : [N, 3*hidden]  (B at [:, 0:hidden], C at [:, hidden:2*hidden], x at [:, 2*hidden:3*hidden])
//   weight : [hidden, 3, 3] flattened as [hidden*9]  (depthwise; the in_channels/groups=1 axis is collapsed)
//   out    : [N, hidden]    = C[t] * depthwise_conv(B*x)[t]
//
// Grid = [ceil(N/bx), ceil(hidden/by)]; one thread per (token, channel).

extern "C" __global__ void conv2d_gated_f32(
    const float* __restrict__ bcx,    // [N, 3*hidden]
    const float* __restrict__ weight, // [hidden, 3, 3] (flattened [hidden*9])
    float* __restrict__ out,          // [N, hidden]
    int N, int hidden, int grid_h, int grid_w, int sms
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;   // token (patch)
    int c = blockIdx.y * blockDim.y + threadIdx.y;   // channel
    if (t >= N || c >= hidden) return;

    int gw = grid_w / sms;                 // merged-grid width
    int mh = grid_h / sms;                 // merged-grid height
    int blk = sms * sms;                   // patches per merge block (4)

    // Decode merge-grouped index t -> raster (py, px).
    //   t = ((gy*gw + gx)*sms + sy)*sms + sx ; py = gy*sms + sy ; px = gx*sms + sx
    int block = t / blk;
    int within = t % blk;
    int gx = block % gw;
    int gy = block / gw;
    int sx = within % sms;
    int sy = within / sms;
    int py = gy * sms + sy;
    int px = gx * sms + sx;

    int stride3 = 3 * hidden;
    // Center token gates.
    float b = bcx[t * stride3 + c];            // B
    float cc = bcx[t * stride3 + hidden + c];  // C
    float xv = bcx[t * stride3 + 2 * hidden + c]; // x

    const float* w = weight + c * 9;           // [3,3] for this channel

    // Depthwise 3x3 conv over gated = B*x, zero-padded.
    float conv = 0.0f;
    for (int dy = -1; dy <= 1; dy++) {
        int ny = py + dy;
        if (ny < 0 || ny >= grid_h) continue;
        for (int dx = -1; dx <= 1; dx++) {
            int nx = px + dx;
            if (nx < 0 || nx >= grid_w) continue;
            // raster (ny, nx) -> merge-grouped index nt
            int ngy = ny / sms, nsy = ny % sms;
            int ngx = nx / sms, nsx = nx % sms;
            int nt = ((ngy * gw + ngx) * sms + nsy) * sms + nsx;
            float b_n = bcx[nt * stride3 + c];
            float x_n = bcx[nt * stride3 + 2 * hidden + c];
            conv += w[(dy + 1) * 3 + (dx + 1)] * (b_n * x_n);
        }
    }

    out[t * hidden + c] = cc * conv;
}
