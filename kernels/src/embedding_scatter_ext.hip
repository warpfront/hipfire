// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// VL batched prefill: overwrite rows of the token-embedding batch with
// external (vision-tower) embeddings. Runs immediately after
// `embedding_lookup_q8_batched`, which has already written every row from
// the token table; this kernel replaces the rows backing `<|image_pad|>`
// tokens with the request's spliced visual embeddings.
//
// Per row `r`:
//   ext_index[r] < 0            -> row untouched (token-embedding lookup wins)
//   ext_index[r] >= 0           -> x[r][:] = row_ptr[r][ext_index[r] * dim ..+ dim]
//   row_ptr[r] == 0 (null)      -> row untouched (defensive: slot lost its
//                                  matrix; must never happen — the scheduler
//                                  refuses VL rows before vision_forward ran)
//
// Grid: [n_rows]. Block: [256] striding dim. x is [n_rows × dim] f32.

extern "C" __global__ void embedding_scatter_ext_f32(
    float* __restrict__ x,                          // [n_rows × dim]
    const int* __restrict__ ext_index,              // [n_rows]
    const unsigned long long* __restrict__ row_ptr, // [n_rows] matrix base pointers
    int dim)
{
    const int row = blockIdx.x;
    const int src = ext_index[row];
    if (src < 0) return;
    const unsigned long long p = row_ptr[row];
    if (p == 0ull) return;
    const float* __restrict__ src_buf =
        reinterpret_cast<const float*>(p) + (size_t)src * (size_t)dim;
    float* __restrict__ dst = x + (size_t)row * (size_t)dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        dst[d] = src_buf[d];
    }
}
