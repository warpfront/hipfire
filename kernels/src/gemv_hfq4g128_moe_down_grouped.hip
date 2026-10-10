// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//
// Grouped MoE down projection for HFQ4-G128 expert weights (batched rows).
// Routed slots arrive sorted by expert in 16-slot tiles (`moe_scatter_fused_k8`
// layout): tile t serves expert `tile_ids[t]` (-1 = unused) and grouped rows
// 16t .. 16t+15, whose activation row is `sorted_slots[row]` (-1 = padding).
//
// One 256-thread workgroup computes eight output rows (one wave each) for
// the sixteen slots of one tile, so each expert row is read once per tile
// instead of once per routed slot. A group (f32 scale, f32 zero, 64 nibble
// bytes = 128 values) is split over four lanes of 32 values. Output:
// y[grouped_row, row] = per_expert_scale[e] * (W_e · act[slot]); the mixture
// weights are applied by `moe_down_combine_grouped_k8`.
#include <hip/hip_runtime.h>

#define GD_ROWS 8
#define GD_SLOTS 16

__launch_bounds__(256, 2)
extern "C" __global__ void gemv_hfq4g128_moe_down_grouped(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ tile_ids,
    const int* __restrict__ sorted_slots,
    const float* __restrict__ per_expert_scale,
    const float* __restrict__ act,        // [slots, K]
    float* __restrict__ y,                // [m_total, M]
    int M, int K
) {
    const int tile = blockIdx.y;
    const int expert = tile_ids[tile];
    if (expert < 0) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int row = blockIdx.x * GD_ROWS + (tid >> 5);
    if (row >= M) return;

    int slot[GD_SLOTS];
#pragma unroll
    for (int j = 0; j < GD_SLOTS; ++j) slot[j] = sorted_slots[tile * GD_SLOTS + j];

    const int groups_per_row = (K + 127) / 128;
    const char* wrow = reinterpret_cast<const char*>(expert_ptrs[expert])
        + (long long)row * groups_per_row * 72;
    float acc[GD_SLOTS];
#pragma unroll
    for (int j = 0; j < GD_SLOTS; ++j) acc[j] = 0.0f;

    for (int c = lane; c < groups_per_row * 4; c += 32) {
        const int g = c >> 2;
        const int base = g * 128 + (c & 3) * 32;
        if (base >= K) continue;
        const char* gptr = wrow + g * 72;
        const float scale = __builtin_bit_cast(float, *(const unsigned int*)gptr);
        const float zero = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const uint2* nib = (const uint2*)(gptr + 8 + (c & 3) * 16);
        const uint2 w0 = nib[0];
        const uint2 w1 = nib[1];
        const unsigned int w[4] = {w0.x, w0.y, w1.x, w1.y};
        float q[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) q[i] = (float)((w[i >> 3] >> (4 * (i & 7))) & 0xf);
        const int len = min(32, K - base);
#pragma unroll
        for (int j = 0; j < GD_SLOTS; ++j) {
            if (slot[j] < 0) continue;
            const float* x = act + (size_t)slot[j] * K + base;
            float dot = 0.0f, xs = 0.0f;
            if (len == 32 && (K & 3) == 0) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float4 xv = *(const float4*)(x + 4 * i);
                    dot = __builtin_fmaf(q[4 * i + 0], xv.x, dot);
                    dot = __builtin_fmaf(q[4 * i + 1], xv.y, dot);
                    dot = __builtin_fmaf(q[4 * i + 2], xv.z, dot);
                    dot = __builtin_fmaf(q[4 * i + 3], xv.w, dot);
                    xs += (xv.x + xv.y) + (xv.z + xv.w);
                }
            } else {
                for (int i = 0; i < len; ++i) {
                    dot = __builtin_fmaf(q[i], x[i], dot);
                    xs += x[i];
                }
            }
            acc[j] += __builtin_fmaf(scale, dot, zero * xs);
        }
    }
    const float es = per_expert_scale[expert];
#pragma unroll
    for (int j = 0; j < GD_SLOTS; ++j) {
        float v = acc[j];
        for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down(v, offset);
        if (lane == 0 && slot[j] >= 0) y[(size_t)(tile * GD_SLOTS + j) * M + row] = es * v;
    }
}
