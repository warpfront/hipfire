// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Rank-ordered fold of the atomic-free MoE down's per-rank outputs into a residual:
//
//   x_residual[col] = ((x_residual[col] + w[0]*e[0][col]) + w[1]*e[1][col]) + ...
//
// The residual first, then route ranks 0, 1, ..., K_TOP-1, each product rounded on its own
// before its add. That is the value `gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed`
// leaves when its K_TOP `atomicAdd(&x_residual[row], w * acc)` land in rank order, but here
// it does not depend on the order the workgroups complete in. Expert weights read over PCIe
// (a pinned host-mapped owner) make that order vary from run to run; resident weights almost
// never do. `moe_down_combine_k8_batched` instead sums the K_TOP products first and adds the
// sum once, which rounds differently.
//
// Shapes:
//   expert_outputs : [K_TOP × M] f32, unscaled per-rank down outputs
//                    (`gemv_mq2g256_lloyd_moe_down_expanded_k4` at N = 1)
//   topk_weights   : [K_TOP]     f32
//   x_residual     : [M]         f32, accumulated in place
//
// Grid: ceil(M / 256). Block: 256. One thread owns one column.

extern "C" __global__ void moe_down_add_scaled_rank_ordered(
    const float* __restrict__ expert_outputs,
    const float* __restrict__ topk_weights,
    float*       __restrict__ x_residual,
    int M, int K_TOP
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= M) return;
    float x = x_residual[col];
    for (int k = 0; k < K_TOP; k++) {
        float product = topk_weights[k] * expert_outputs[(long long)k * M + col];
        // Pin the rounded product in a VGPR: fusing it into fma(w, e, x) would round once and
        // no longer match the atomic route's separately rounded `w * acc`.
        asm("" : "+v"(product));
        x += product;
    }
    x_residual[col] = x;
}
