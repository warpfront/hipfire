// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1201 decode (`HIPFIRE_G12_DEC_NORM`): `rmsnorm_f32` as a grid of
// (n/256, rows) workgroups instead of one workgroup per row. Every workgroup
// recomputes its row's sum of squares with rmsnorm_f32's per-thread
// fma(v, v, acc) chain (same elements, same order; loads batched so up to 20
// are in flight instead of one) and its LDS halving tree, so rms is
// bit-identical; it then writes only its own 256 outputs. Launch with
// blockDim.x == 256 (the halving tree is unrolled for it, so every step has
// its own barrier) and `out` not aliasing `x`: other workgroups of the row
// are still reading x.
extern "C" __global__ __launch_bounds__(256, 1)
void rmsnorm_f32_rowsplit(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int n, float eps
) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    constexpr int bs = 256;
    const int row = blockIdx.y * n;
    float sum_sq = 0.0f;
    int i = tid;
#define RMSNORM_ROWSPLIT_STEP(B) \
    for (; i + ((B) - 1) * bs < n; i += (B) * bs) { \
        float vb[B]; \
        _Pragma("unroll") for (int u = 0; u < (B); ++u) vb[u] = x[row + i + u * bs]; \
        _Pragma("unroll") for (int u = 0; u < (B); ++u) sum_sq = __fmaf_rn(vb[u], vb[u], sum_sq); \
    }
    RMSNORM_ROWSPLIT_STEP(20)
    RMSNORM_ROWSPLIT_STEP(8)
    RMSNORM_ROWSPLIT_STEP(4)
#undef RMSNORM_ROWSPLIT_STEP
    for (; i < n; i += bs) {
        const float v = x[row + i];
        sum_sq = __fmaf_rn(v, v, sum_sq);
    }
    sdata[tid] = sum_sq;
    __syncthreads();
#pragma unroll
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / (float)n + eps);
    const int d = blockIdx.x * bs + tid;
    if (d < n) {
        out[row + d] = x[row + d] * weight[d] * rms;
    }
}
