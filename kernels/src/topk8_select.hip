// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// ── Draft-head argmax through an approximate head ─────────────────────────
// A low-bit copy of the LM head ranks the vocabulary; its top 8 are then
// scored exactly against the head's own rows, so the result is the head's
// argmax whenever that token ranks in the approximate top 8. Prepended to the
// sources of the per-format re-score kernels (qwen4_gemv_q8_0.hip, and
// gemv_mq6g256v2.hip under HIPFIRE_TOPK8_RESCORE).

#include <hip/hip_runtime.h>

// (value, index) order of argmax_f32: larger value, then smaller index.
static __device__ __forceinline__ bool topk8_better(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

// Block-wide best (value, index) of each thread's candidate; every thread gets it.
static __device__ __forceinline__ void topk8_block_best(float& v, int& i, float* sv, int* si) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float ov = __shfl_down(v, offset);
        const int oi = __shfl_down(i, offset);
        if (topk8_better(ov, oi, v, i)) { v = ov; i = oi; }
    }
    const int wave = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) { sv[wave] = v; si[wave] = i; }
    __syncthreads();
    v = sv[0]; i = si[0];
    for (int w = 1; w < (int)(blockDim.x >> 5); ++w)
        if (topk8_better(sv[w], si[w], v, i)) { v = sv[w]; i = si[w]; }
    __syncthreads();
}

// Block of 256: the token ids of the top 8 of m <= 2048 partial candidates
// (topk8_partial_f32's pv/pi) -> cand[0..8) (shared), visible to every thread.
static __device__ __forceinline__ void topk8_select(
    const float* __restrict__ pv, const int* __restrict__ pi, int m, int* cand
) {
    __shared__ float sv[8];
    __shared__ int si[8];
    float val[8];
    int idx[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const int j = threadIdx.x + u * 256;
        const bool ok = j < m;
        val[u] = ok ? pv[j] : -INFINITY;
        idx[u] = ok ? pi[j] : 0x7fffffff;
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
        if (threadIdx.x == 0) cand[round] = bi;
    }
    __syncthreads();
}

// Thread 0: the best exact score -> out[0] its token, out[1] its margin over
// the best other candidate (f32 bits); out[2..10) the 8 candidate tokens and
// out[10..18) their exact scores (f32 bits), the distribution a sampled draft
// is drawn from.
static __device__ __forceinline__ void topk8_finish(const float* exact, const int* cand, int* out) {
    float bv = exact[0];
    int bi = cand[0];
    int bw = 0;
    for (int w = 1; w < 8; ++w)
        if (topk8_better(exact[w], cand[w], bv, bi)) { bv = exact[w]; bi = cand[w]; bw = w; }
    float second = -INFINITY;
    for (int w = 0; w < 8; ++w)
        if (w != bw) second = fmaxf(second, exact[w]);
    out[0] = bi;
    out[1] = __float_as_int(bv - second);
    for (int w = 0; w < 8; ++w) {
        out[2 + w] = cand[w];
        out[10 + w] = __float_as_int(exact[w]);
    }
}
