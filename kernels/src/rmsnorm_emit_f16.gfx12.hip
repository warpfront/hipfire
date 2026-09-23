// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Decode-only gfx12 producer for Q8_0 x FP16 WMMA. Preserve rmsnorm_f32's
// reduction and F32 output expression exactly, and emit the same normalized
// activation as an FP16 sidecar while the value is still in a register.
extern "C" __global__ void rmsnorm_f32_emit_f16(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    _Float16* __restrict__ out_f16,
    int n,
    float eps
) {
    extern __shared__ float sdata[];
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[blockIdx.x * n + i];
        sum_sq += v * v;
    }
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    const float rms = rsqrtf(sdata[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int idx = blockIdx.x * n + i;
        const float v = x[idx] * weight[i] * rms;
        out[idx] = v;
        out_f16[idx] = (_Float16)v;
    }
}
