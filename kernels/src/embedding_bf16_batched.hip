// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

// Batched BF16 embedding lookup. The table is resident BF16 and the output is
// F32; token ids remain device-resident so token/chunk forward has no host
// round-trip or per-token allocation.
extern "C" __global__ void embedding_bf16_batched(
    const __hip_bfloat16* __restrict__ table,
    float* __restrict__ output,
    const int* __restrict__ token_ids,
    int dim
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int token_id = token_ids[batch_idx];
    const __hip_bfloat16* row = table + (size_t)token_id * dim;
    float* out_row = output + (size_t)batch_idx * dim;
    for (int elem = tid; elem < dim; elem += blockDim.x) {
        out_row[elem] = (float)row[elem];
    }
}
