// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — fixed E512/top-10 softmax router with original-id membership.

#include <hip/hip_runtime.h>
#include <math.h>

// This separately registered variant preserves the incumbent router's
// selection, float32 reduction order, normalization and BF16 boundaries.
// Only the logit loads differ: excluded experts read as negative infinity
// and are exempt from the non-finite fail-closed check; kept experts fault
// like the dense router.
__device__ __forceinline__ float router_bf16_rt(float v) {
    if (!isfinite(v)) return v;
    const unsigned int u = __float_as_uint(v);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

// Router fault word, identical to the dense router's (see
// moe_router_softmax_top10_f32.hip): 0 = clear, otherwise
// ((row + 1) << 24) | (layer + 1); the smaller code is kept.
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

__device__ __forceinline__ void router_fail_row(
    int* topk_idx, float* topk_w, int token, unsigned long long* fault, int layer, int row) {
    const size_t output_base = (size_t)token * 10;
    for (int rank = 0; rank < 10; ++rank) {
        topk_idx[output_base + rank] = -1;
        topk_w[output_base + rank] = 0.0f;
    }
    router_record_fault(fault, layer, row);
}

__device__ __forceinline__ float router_wave_max(float v) {
#if defined(__GFX11__) || defined(__GFX12__)
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x161, 0xf, 0xf, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x162, 0xf, 0xf, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x164, 0xf, 0xf, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x168, 0xf, 0xf, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false)));
#else
    for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_xor(v, offset, 32));
#endif
    return v;
}

__device__ __forceinline__ int router_wave_min(int v) {
#if defined(__GFX11__) || defined(__GFX12__)
    v = min(v, __builtin_amdgcn_update_dpp(0, v, 0x161, 0xf, 0xf, false));
    v = min(v, __builtin_amdgcn_update_dpp(0, v, 0x162, 0xf, 0xf, false));
    v = min(v, __builtin_amdgcn_update_dpp(0, v, 0x164, 0xf, 0xf, false));
    v = min(v, __builtin_amdgcn_update_dpp(0, v, 0x168, 0xf, 0xf, false));
    v = min(v, __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false));
#else
    for (int offset = 16; offset > 0; offset >>= 1) v = min(v, __shfl_xor(v, offset, 32));
#endif
    return v;
}

__launch_bounds__(256, 1)
extern "C" __global__ void moe_router_softmax_top10_masked_f32(
    const float* __restrict__ logits,
    int* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int n_exp,
    int top_k,
    int normalize_topk_prob,
    int round_logits,
    const unsigned char* __restrict__ keep,
    unsigned long long* __restrict__ fault, // host-mapped router fault word, or null
    int layer,                              // sealed call layer index
    int row_base                            // global row of this launch's row 0
) {
    __shared__ float wave_values[80];
    __shared__ int wave_indices[80];
    __shared__ float selected_values[10];
    __shared__ float probabilities[10];
    const int thread = threadIdx.x;
    const int lane = thread & 31;
    const int wave = thread >> 5;
    const int token = blockIdx.x;

    if (n_exp != 512 || top_k != 10) return;

    // Excluded experts are never read: their logits (even NaN/Inf) cannot
    // fault the row. Only the kept logits are checked, after the optional BF16
    // round trip, exactly like the dense router: any NaN/Inf kept logit fails
    // the whole row closed (ids -1, weights 0) and records (layer, row) in
    // the device fault word. The vote is workgroup-uniform.
    const size_t token_base = (size_t)token * 512;
    const bool first_kept = keep[thread] != 0;
    const bool second_kept = keep[thread + 256] != 0;
    float first_logit = first_kept ? logits[token_base + thread] : -INFINITY;
    float second_logit = second_kept ? logits[token_base + thread + 256] : -INFINITY;
    if (round_logits != 0) {
        first_logit = router_bf16_rt(first_logit);
        second_logit = router_bf16_rt(second_logit);
    }
    if (__syncthreads_or((first_kept && !isfinite(first_logit)) || (second_kept && !isfinite(second_logit)))) {
        if (thread == 0) router_fail_row(topk_idx, topk_w, token, fault, layer, row_base + token);
        return;
    }
    float first_value = first_logit;
    float second_value = second_logit;
    const int first_index = thread;
    const int second_index = thread + 256;

    for (int rank = 0; rank < 10; ++rank) {
        const float best_value = router_wave_max(fmaxf(first_value, second_value));
        const int candidate = first_value == best_value ? first_index
            : second_value == best_value ? second_index : 0x7fffffff;
        const int best_index = router_wave_min(candidate);
        if (lane == 0) {
            wave_values[wave * 10 + rank] = best_value;
            wave_indices[wave * 10 + rank] = best_index;
        }
        if (first_index == best_index) first_value = -INFINITY;
        if (second_index == best_index) second_value = -INFINITY;
    }
    __syncthreads();

    if (wave == 0) {
        const int slot0 = lane;
        const int slot1 = lane + 32;
        const int slot2 = lane + 64;
        float value0 = wave_values[slot0];
        float value1 = wave_values[slot1];
        float value2 = slot2 < 80 ? wave_values[slot2] : -INFINITY;
        const int index0 = wave_indices[slot0];
        const int index1 = wave_indices[slot1];
        const int index2 = slot2 < 80 ? wave_indices[slot2] : 0x7fffffff;

        for (int rank = 0; rank < top_k; ++rank) {
            const float best_value = router_wave_max(fmaxf(fmaxf(value0, value1), value2));
            int candidate = 0x7fffffff;
            if (value0 == best_value) candidate = min(candidate, index0);
            if (value1 == best_value) candidate = min(candidate, index1);
            if (value2 == best_value) candidate = min(candidate, index2);
            const int best_index = router_wave_min(candidate);
            if (lane == 0) {
                topk_idx[(size_t)token * 10 + rank] = best_index;
                selected_values[rank] = best_value;
            }
            if (index0 == best_index) value0 = -INFINITY;
            if (index1 == best_index) value1 = -INFINITY;
            if (index2 == best_index) value2 = -INFINITY;
        }
    }

    // Identical full index-order softmax and selected renormalization.
    // Excluded entries contribute expf(-INFINITY) == 0 to the denominator.
    __shared__ float exps[512];
    __syncthreads();
    {
        const float max_logit = selected_values[0];
        exps[thread] = expf(first_logit - max_logit);
        exps[thread + 256] = expf(second_logit - max_logit);
    }
    __syncthreads();
    if (thread == 0) {
        const size_t output_base = (size_t)token * 10;
        const float max_logit = selected_values[0];
        float denom = 0.0f;
        for (int expert = 0; expert < 512; ++expert)
            denom += exps[expert];
        if (!(denom > 0.0f) || !isfinite(denom)) {
            router_fail_row(topk_idx, topk_w, token, fault, layer, row_base + token);
            return;
        }

        const float inv_denom = 1.0f / denom;
        for (int rank = 0; rank < top_k; ++rank)
            probabilities[rank] = expf(selected_values[rank] - max_logit) * inv_denom;

        if (normalize_topk_prob != 0) {
            float selected_sum = 0.0f;
            for (int rank = 0; rank < top_k; ++rank)
                selected_sum += probabilities[rank];
            if (!(selected_sum > 0.0f) || !isfinite(selected_sum)) {
                router_fail_row(topk_idx, topk_w, token, fault, layer, row_base + token);
                return;
            }
            const float inv_selected = 1.0f / selected_sum;
            for (int rank = 0; rank < top_k; ++rank)
                probabilities[rank] *= inv_selected;
        }

        for (int rank = 0; rank < top_k; ++rank)
            topk_w[output_base + rank] =
                round_logits != 0 ? router_bf16_rt(probabilities[rank]) : probabilities[rank];
    }
}
