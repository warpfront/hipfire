// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// AWQ pre-divide WITHOUT rotation, for rotation-free quantized dtypes
// (HFP4G32 + AWQ sidecar): y[r*K+i] = x[r*K+i] / s[i].
//
// The FWHT-fused `*_mq_rotate_awq` producers MUST NOT be used on this path —
// they FWHT-rotate, and HFP4 weights are encoded unrotated (RotationPlan::None).
// The divide happens in the unrotated x basis, mirroring the offline weight
// side `W' = W * diag(s)`, so `(W·s)·(x/s) = W·x`.
//
// Alias-safe (pure elementwise): y may alias x, so prefill callers can divide
// a freshly-normalized buffer in place. s is the loader-upconverted FP32
// sidecar, length K.

extern "C" __global__ void awq_divide_x(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ s,
    int rows,
    int k
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long n = (long long)rows * k;
    if (i < n) y[i] = x[i] / s[i % k];
}
