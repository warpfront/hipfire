// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched F32 embedding lookup (device-id gather). Copies N rows of an F32
// table into `output[n, dim]` (F32), reading token ids from a device buffer
// so the call is graph-replayable — a host-side D2D memcpy per row would
// bake token offsets into the captured graph.
extern "C" __global__ void embedding_f32_batched(
    const float* __restrict__ table,      // [vocab × dim]
    float* __restrict__ output,           // [n × dim] row-major
    const int* __restrict__ token_ids,    // [n]
    int dim
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int token_id = token_ids[batch_idx];
    const float* row = table + (size_t)token_id * dim;
    float* out_row = output + (size_t)batch_idx * dim;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        out_row[elem] = row[elem];
    }
}
