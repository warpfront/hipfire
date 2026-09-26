// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — QT53 top-10 indexed down projection with expanded output.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// The kernel writes unweighted expanded rows.  moe_down_combine_top10_batched
// is the sole owner of route weighting.  This keeps the ten route products
// deterministic and prevents a second weighting in the grouped path.
//
// Wire rows use exactly 68*ceil(K/128) bytes.  This is deliberately not the
// legacy 72-byte HFQ4G128/Paro row stride.
__launch_bounds__(32, 8)
extern "C" __global__ void gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ rot_batch,
    float* __restrict__ expert_outputs,
    int M,
    int K,
    int N,
    int n_exp
) {
    const int row = blockIdx.x;
    const int rank = blockIdx.y;
    const int token = blockIdx.z;
    const int tid = threadIdx.x;
    if (row >= M || rank >= 10 || token >= N || K <= 0) return;

    const int flat = token * 10 + rank;
    const int expert = topk_indices[flat];
    float acc = 0.0f;
    if (expert >= 0 && expert < n_exp) {
        const char* A = reinterpret_cast<const char*>(expert_ptrs[expert]);
        const int groups = (K + 127) / 128;
        const long long row_stride = (long long)groups * 68;
        const char* row_ptr = A + (long long)row * row_stride;
        const float* x = rot_batch + (long long)flat * K;
        for (int group = 0; group < groups; ++group) {
            const char* gp = row_ptr + (long long)group * 68;
            const unsigned short scale_bits = *(const unsigned short*)(gp + 0);
            const unsigned short zero_bits = *(const unsigned short*)(gp + 2);
            const float scale = __half2float(__ushort_as_half(scale_bits));
            const float zero = __half2float(__ushort_as_half(zero_bits));
            const unsigned char* payload = (const unsigned char*)(gp + 4);
            const int base = group * 128 + tid * 4;
            const unsigned char b0 = payload[tid * 2];
            const unsigned char b1 = payload[tid * 2 + 1];
            const int i0 = base;
            const int i1 = base + 1;
            const int i2 = base + 2;
            const int i3 = base + 3;
            const float x0 = i0 < K ? x[i0] : 0.0f;
            const float x1 = i1 < K ? x[i1] : 0.0f;
            const float x2 = i2 < K ? x[i2] : 0.0f;
            const float x3 = i3 < K ? x[i3] : 0.0f;
            acc += (scale * (float)(b0 & 0xF) + zero) * x0
                + (scale * (float)(b0 >> 4) + zero) * x1
                + (scale * (float)(b1 & 0xF) + zero) * x2
                + (scale * (float)(b1 >> 4) + zero) * x3;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if (tid == 0)
        expert_outputs[(long long)flat * M + row] = acc;
}
