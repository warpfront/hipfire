// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// DFlash2 target features on Qwen4: the completed output of one trunk layer,
// averaged over its hyper-connection streams, written as F32 into one layer
// slot of the capture rows. `streams` is `[row, stream, hidden]`, F32 or
// BF16-stored (the first rows*hc*hidden halves of an F32-typed buffer).

#include <hip/hip_runtime.h>

extern "C" __global__ void hc_mean_capture_f32(
    const float* streams,
    float* out,
    int rows,
    int hc,
    int hidden,
    int out_stride,
    int out_offset,
    int bf16) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= rows * hidden) return;
    const int r = i / hidden;
    const int h = i - r * hidden;
    const size_t base = (size_t)r * hc * hidden + h;
    float sum = 0.0f;
    if (bf16) {
        const unsigned short* s = (const unsigned short*)streams;
        for (int b = 0; b < hc; ++b)
            sum += __uint_as_float((unsigned int)s[base + (size_t)b * hidden] << 16);
    } else {
        for (int b = 0; b < hc; ++b) sum += streams[base + (size_t)b * hidden];
    }
    out[(size_t)r * out_stride + out_offset + h] = sum / (float)hc;
}
