// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// Device-routed MoE down projection for HFQ4-G128 expert weights.  Expert
// selection and mixture weights remain device-resident so the launch sequence
// and kernargs are stable across generated tokens.
//
// One 256-thread workgroup serves eight output rows, one wave per row. A
// group (f32 scale, f32 zero, 64 nibble bytes = 128 values) is split over
// four lanes of 32 values, so a row's groups are read with wide loads and all
// eight experts' loads are in flight together; the activations (8 * K floats,
// cache-resident) are read as float4. Each lane folds the mixture weight in
// before one wave reduction.
#include <hip/hip_runtime.h>

#define MD_ROWS 8

__launch_bounds__(256, 2)
extern "C" __global__ void gemv_hfq4g128_moe_down_residual_scaled_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ per_expert_scale,
    const float* __restrict__ hidden_batch,
    float* __restrict__ x_residual,
    int M, int K
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int row = blockIdx.x * MD_ROWS + (tid >> 5);
    if (row >= M) return;

    const int groups_per_row = (K + 127) / 128;
    const int chunks = groups_per_row * 4;
    float mixed = 0.0f;
    for (int c = lane; c < chunks; c += 32) {
        const int g = c >> 2;
        const int base = g * 128 + (c & 3) * 32;
        if (base >= K) continue;
#pragma unroll
        for (int krank = 0; krank < 8; ++krank) {
            const int expert_id = topk_indices[krank];
            const char* gptr = reinterpret_cast<const char*>(expert_ptrs[expert_id])
                + ((long long)row * groups_per_row + g) * 72;
            const float scale = __builtin_bit_cast(float, *(const unsigned int*)gptr);
            const float zero = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
            const uint2* nib = (const uint2*)(gptr + 8 + (c & 3) * 16);
            const uint2 w0 = nib[0];
            const uint2 w1 = nib[1];
            const unsigned int w[4] = {w0.x, w0.y, w1.x, w1.y};
            const float* x = hidden_batch + (size_t)krank * K + base;
            float dot = 0.0f, xs = 0.0f;
            if (base + 32 <= K && (K & 3) == 0) {
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const float4 xv = *(const float4*)(x + 4 * q);
                    const unsigned int wq = w[q >> 1] >> (16 * (q & 1));
                    dot = __builtin_fmaf((float)(wq & 0xf), xv.x, dot);
                    dot = __builtin_fmaf((float)((wq >> 4) & 0xf), xv.y, dot);
                    dot = __builtin_fmaf((float)((wq >> 8) & 0xf), xv.z, dot);
                    dot = __builtin_fmaf((float)((wq >> 12) & 0xf), xv.w, dot);
                    xs += (xv.x + xv.y) + (xv.z + xv.w);
                }
            } else {
                for (int j = 0; j < 32 && base + j < K; ++j) {
                    const float xv = x[j];
                    dot = __builtin_fmaf((float)((w[j >> 3] >> (4 * (j & 7))) & 0xf), xv, dot);
                    xs += xv;
                }
            }
            const float coef = topk_weights[krank] * per_expert_scale[expert_id];
            mixed = __builtin_fmaf(coef, __builtin_fmaf(scale, dot, zero * xs), mixed);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        mixed += __shfl_down(mixed, offset);
    if (lane == 0) x_residual[row] = mixed;
}
