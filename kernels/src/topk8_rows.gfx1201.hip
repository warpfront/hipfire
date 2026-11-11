// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Row-batched draft-head re-score (HIPFIRE_QWEN4_DRAFT_TOPK8_ROWS, exact gfx1201). Compiled after
// topk8_select.hip and qwen4_gemv_mq6g256v2.hip (module `topk8_rows_mq6g256v2`), whose
// `topk8_*` helpers and `mq6g256v2_row_dot` the bodies below use. Each row of a launch runs
// the body of the one-row kernel it mirrors on its own inputs and its own output slot, so a
// row's bytes do not depend on which launch computed it.

// `topk8_partial_f32` (qwen4_gemv_q8_0.hip) for blockIdx.y rows. Grid (128, rows), block 256.
// Row r reads v + r * v_stride and writes its own TOPK8_PARTIAL_BYTES (8192) slot at
// part + r * 8192: pv (128 * 8 f32) then pi (128 * 8 i32), the one-row layout.
__launch_bounds__(256)
extern "C" __global__ void topk8_partial_f32_rows(
    const float* __restrict__ v_base, int n, int chunk, float* __restrict__ part_base,
    int front, int special, int tail, int v_stride
) {
    __shared__ float sv[8];
    __shared__ int si[8];
    const float* __restrict__ v = v_base + (long long)blockIdx.y * v_stride;
    float* __restrict__ pv = part_base + (long long)blockIdx.y * 2048;
    int* __restrict__ pi = (int*)(pv + 1024);
    const int begin = blockIdx.x * chunk;
    const int end = min(n, begin + chunk);
    float val[8];
    int idx[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const int j = begin + threadIdx.x + u * 256;
        const bool ok = j < end;
        val[u] = ok ? v[j] : -INFINITY;
        idx[u] = ok ? j : 0x7fffffff;
    }
    for (int round = 0; round < 8; ++round) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int u = 0; u < 8; ++u)
            if (topk8_better(val[u], idx[u], bv, bi)) { bv = val[u]; bi = idx[u]; }
        topk8_block_best(bv, bi, sv, si);
#pragma unroll
        for (int u = 0; u < 8; ++u)
            if (idx[u] == bi) { val[u] = -INFINITY; idx[u] = 0x7fffffff; }
        if (threadIdx.x == 0) {
            const int token = bi < front ? bi : (bi < front + tail ? bi - front + special : bi - tail);
            pv[blockIdx.x * 8 + round] = bv;
            pi[blockIdx.x * 8 + round] = token;
        }
    }
}

// `topk8_rescore_mq6g256v2_k2560` for blockIdx.x rows (m = 128 * 8). Block r reads partial slot r
// (8192 B at part + r * 8192), x at x_base + r * x_stride (floats), and writes the 18-word
// TOPK8_OUT_BYTES slot at out_base + r * out_stride (ints).
__launch_bounds__(256)
extern "C" __global__ void topk8_rescore_mq6g256v2_k2560_rows(
    const float* __restrict__ part_base, const char* __restrict__ A,
    const float* __restrict__ x_base, int* __restrict__ out_base, int x_stride, int out_stride
) {
    __shared__ int cand[8];
    __shared__ float exact[8];
    const float* __restrict__ pv = part_base + (long long)blockIdx.x * 2048;
    const int* __restrict__ pi = (const int*)(pv + 1024);
    const float* __restrict__ x = x_base + (long long)blockIdx.x * x_stride;
    int* __restrict__ out = out_base + (long long)blockIdx.x * out_stride;
    topk8_select(pv, pi, 1024, cand);
    const int wave = threadIdx.x >> 5;
    const float s = mq6g256v2_row_dot(A, x, cand[wave], 2560, threadIdx.x & 31);
    if ((threadIdx.x & 31) == 0) exact[wave] = s;
    __syncthreads();
    if (threadIdx.x == 0) topk8_finish(exact, cand, out);
}
