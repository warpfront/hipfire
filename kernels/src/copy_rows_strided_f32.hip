// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// 2D strided F32 row copy — one launch for what used to be one `copy_d2d`
// per row.
//
//   dst[row * dst_row_stride + dst_col_offset + c] =
//   src[row * src_row_stride + c]          for row in [0, n_rows), c in [0, len)
//
// Motivation: `Gpuf::assemble_rows` (FLUX.1 MMDiT single block) concatenates
// `[att(d), mlp_gelu(4d)]` into the `linear2` input, which the fused weight
// interleaves along K. Done as one D2D memcpy per (row, chunk) that is
// 2 * 4608 = 9216 tiny copies per block; at FLUX shapes each is only 12 or
// 48 KB, so the whole assemble is launch-latency bound. This kernel does the
// same work in one dispatch per chunk.
//
// `vec4` selects a float4 fast path (4 elements per thread). The caller sets
// it only when `len`, both row strides and `dst_col_offset` are multiples of
// 4 AND both base pointers are 16-byte aligned; the scalar path keeps any
// other shape correct. At FLUX shapes (len 3072 / 12288, dst_row_stride
// 15360, dst_col_offset 0 / 3072) the fast path is always taken.
//
// Grid: x over the columns of one row, y over rows (grid-stride on y so a
// row count above the launch cap still works). Block 256, matching the other
// bulk-copy kernels in this tree.
extern "C" __global__ __launch_bounds__(256) void copy_rows_strided_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n_rows, int len,
    int src_row_stride, int dst_row_stride,
    int dst_col_offset,
    int vec4) {

    const int col = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    if (vec4) {
        const int len4 = len >> 2;
        if (col >= len4) return;
        for (int row = (int)blockIdx.y; row < n_rows; row += (int)gridDim.y) {
            const float4* s =
                (const float4*)(src + (size_t)row * (size_t)src_row_stride);
            float4* d = (float4*)(dst + (size_t)row * (size_t)dst_row_stride
                                      + (size_t)dst_col_offset);
            d[col] = s[col];
        }
    } else {
        if (col >= len) return;
        for (int row = (int)blockIdx.y; row < n_rows; row += (int)gridDim.y) {
            dst[(size_t)row * (size_t)dst_row_stride + (size_t)dst_col_offset
                + (size_t)col] =
                src[(size_t)row * (size_t)src_row_stride + (size_t)col];
        }
    }
}
