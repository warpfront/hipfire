// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// The ordinary Z SET writes contiguous [N, folded_m]. Keep its kernel and
// arithmetic unchanged; only scatter the three live row intervals here.
extern "C" __global__ void split_mq4v2_z_betaalpha(
    const float* __restrict__ folded,
    float* __restrict__ z,
    float* __restrict__ beta,
    float* __restrict__ alpha,
    int z_m, int folded_m, int n)
{
    int token = (int)blockIdx.x;
    if (token >= n) return;
    const float* src = folded + (size_t)token * folded_m;
    float* dst = z + (size_t)token * z_m;
    for (int row = (int)threadIdx.x; row < z_m; row += (int)blockDim.x)
        dst[row] = src[row];
    for (int row = (int)threadIdx.x; row < 48; row += (int)blockDim.x) {
        beta[(size_t)token * 48 + row] = src[z_m + row];
        alpha[(size_t)token * 48 + row] = src[z_m + 48 + row];
    }
}
