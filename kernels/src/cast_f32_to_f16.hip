// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

// Standalone f32 → f16 elementwise cast.
//
// The same body lives inline in `gemm_hfq4g256_residual_fp16.hip` and
// `gemm_hfq6g256_residual_fp16.hip` where it is invoked via the
// per-call `ensure_fp16_x` machinery. This standalone copy exists so
// callers outside the FP16-GEMM hot path (e.g. the vision-encoder
// attention's K/V pre-cast, see `attention_dflash_wmma_n64_f16kv.hip`)
// can launch the cast directly without pulling in the GEMM kernel
// source.

#include <hip/hip_runtime.h>

extern "C" __global__ void cast_f32_to_f16(
    const float* __restrict__ in,
    _Float16* __restrict__ out,
    int n
) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) out[i] = (_Float16)in[i];
}

// Row-pitched cast: `in` is a packed `[rows, cols]` f32 matrix, `out` a
// `[rows, pitch]` f16 matrix (`pitch >= cols`). The `pitch - cols` pad columns
// of every row are written as +0.0, so a consumer that reads K = `pitch`
// elements per row (a GEMM whose K was rounded up to its tile) sees exact
// zeros there and the pad contributes nothing against a zero-padded weight.
// The pad is rewritten on every launch, so `out` need not be pre-zeroed.
extern "C" __global__ void cast_f32_to_f16_pitched(
    const float* __restrict__ in,
    _Float16* __restrict__ out,
    int rows,
    int cols,
    int pitch
) {
    size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    size_t total = (size_t)rows * (size_t)pitch;
    if (i >= total) return;
    size_t row = i / (size_t)pitch;
    int col = (int)(i - row * (size_t)pitch);
    out[i] = (col < cols) ? (_Float16)in[row * (size_t)cols + col] : (_Float16)0.0f;
}
