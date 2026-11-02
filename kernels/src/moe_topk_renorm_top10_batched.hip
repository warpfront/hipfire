// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched top-10 + renorm given pre-softmaxed probabilities. This is kept as
// a dedicated symbol: Qwen4's fixed top-10 route must not silently enter the
// incumbent k=8 producer.
#define BLOCK_SIZE 256
#define MAX_NEXP   1024
#define K_TOP      10

// Router fault word (host-mapped, written with plain system-scope stores like the
// bounded-wait block, so PCIe atomics are not required). 0 = clear; otherwise
// ((row + 1) << 24) | (layer + 1) with the global row. The smaller code (earliest
// row, then earliest layer) is kept; with simultaneous writers the last store wins.
__device__ __forceinline__ void router_record_fault(
    unsigned long long* fault, int layer, int row) {
    if (fault == nullptr) return;
    const unsigned long long code =
        ((unsigned long long)(unsigned int)(row + 1) << 24) | (unsigned long long)(unsigned int)(layer + 1);
    const unsigned long long current =
        __hip_atomic_load(fault, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    if (current == 0ull || code < current)
        __hip_atomic_store(fault, code, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

extern "C" __launch_bounds__(BLOCK_SIZE, 2)
__global__ void moe_topk_renorm_top10_batched(
    const float* __restrict__ probs_in,
    int*         __restrict__ topk_idx,
    float*       __restrict__ topk_w,
    int n_exp,
    int norm_topk,
    unsigned long long* __restrict__ fault, // host-mapped router fault word, or null
    int layer,                              // sealed call layer index
    int row_base                            // global row of this launch's row 0
) {
    __shared__ float smem_v[MAX_NEXP];
    __shared__ int   smem_i[MAX_NEXP];
    __shared__ float reduce_v[BLOCK_SIZE];
    __shared__ int   reduce_i[BLOCK_SIZE];
    __shared__ int   picked_idx[K_TOP];
    __shared__ float picked_w[K_TOP];

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const float* __restrict__ row_probs = probs_in + (long long)bid * n_exp;
    int* __restrict__ row_idx = topk_idx + bid * K_TOP;
    float* __restrict__ row_w = topk_w + bid * K_TOP;

    int nonfinite = 0;
    for (int i = tid; i < MAX_NEXP; i += BLOCK_SIZE) {
        const float value = (i < n_exp) ? row_probs[i] : -INFINITY;
        nonfinite |= (i < n_exp) && !isfinite(value);
        smem_v[i] = value;
        smem_i[i] = i;
    }
    // Workgroup-uniform vote; also the barrier that publishes smem_v/smem_i.
    const int row_nonfinite = __syncthreads_or(nonfinite);

    for (int k = 0; k < K_TOP; ++k) {
        float best_v = -INFINITY;
        int best_i = -1;
        for (int i = tid; i < n_exp; i += BLOCK_SIZE) {
            float value = smem_v[i];
            // The strict comparison intentionally retains the lower-index
            // winner on ties, matching the CPU route reference.
            if (value > best_v) {
                best_v = value;
                best_i = smem_i[i];
            }
        }
        reduce_v[tid] = best_v;
        reduce_i[tid] = best_i;
        __syncthreads();
        for (int stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float lhs = reduce_v[tid];
                float rhs = reduce_v[tid + stride];
                if (rhs > lhs) {
                    reduce_v[tid] = rhs;
                    reduce_i[tid] = reduce_i[tid + stride];
                }
            }
            __syncthreads();
        }
        const int winner = reduce_i[0];
        if (tid == 0) {
            picked_idx[k] = winner;
            picked_w[k] = reduce_v[0];
        }
        if (winner >= 0 && winner < n_exp && ((winner & (BLOCK_SIZE - 1)) == tid)) {
            smem_v[winner] = -INFINITY;
        }
        __syncthreads();
    }

    // A row fails closed as a whole (marker -1, weight 0 for every slot, plus the
    // fault record) when ANY of its n_exp probabilities is non-finite (the
    // selection above skips NaN, so the picks alone cannot show it), or a pick is
    // invalid or non-finite. Finite rows are unchanged.
    __shared__ int row_bad;
    if (tid == 0) {
        int bad = row_nonfinite;
        for (int k = 0; k < K_TOP; ++k) {
            if (picked_idx[k] < 0 || picked_idx[k] >= n_exp || !isfinite(picked_w[k])) bad = 1;
        }
        row_bad = bad;
        if (!bad) {
            float sum = 0.0f;
            for (int k = 0; k < K_TOP; ++k) sum += picked_w[k];
            if (norm_topk && sum > 0.0f) {
                for (int k = 0; k < K_TOP; ++k) picked_w[k] /= sum;
            }
        } else {
            router_record_fault(fault, layer, row_base + bid);
        }
    }
    __syncthreads();

    if (tid < K_TOP) {
        row_idx[tid] = row_bad ? -1 : picked_idx[tid];
        row_w[tid] = row_bad ? 0.0f : picked_w[tid];
    }
}

// Pre-softmax guard for the unfused route. softmax_f32 turns a NaN or +Inf logit
// into an all-NaN row, but a -Inf logit into a finite probability, so any
// non-finite logit is recorded here and the row is poisoned with a NaN (the
// scores are softmaxed in place afterwards and are scratch). The renorm above
// then fails the row closed. One workgroup per row; finite rows are untouched.
extern "C" __launch_bounds__(BLOCK_SIZE)
__global__ void moe_router_logits_guard_top10(
    float*       __restrict__ logits,   // [rows x n_exp]
    int n_exp,
    unsigned long long* __restrict__ fault,
    int layer,
    int row_base
) {
    const int tid = threadIdx.x;
    float* __restrict__ row = logits + (long long)blockIdx.x * n_exp;
    int bad = 0;
    for (int i = tid; i < n_exp; i += BLOCK_SIZE) bad |= !isfinite(row[i]);
    if (__syncthreads_or(bad)) {
        if (tid == 0) {
            row[0] = __int_as_float(0x7fc00000);
            router_record_fault(fault, layer, row_base + (int)blockIdx.x);
        }
    }
}
