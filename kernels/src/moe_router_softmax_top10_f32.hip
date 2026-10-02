// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — fixed E512/top-10 softmax router.

#include <hip/hip_runtime.h>
#include <math.h>

// One 256-thread workgroup handles one token.  Each wave selects its best ten
// from 64 logits, then wave 0 selects the global best ten from those 80
// candidates.  This keeps stable ordering while avoiding ten serial 512-entry
// scans and the barrier-heavy full sort.
//
// topk_idx is an i32 bit-pattern stored in the project's F32-typed scratch
// convention.  `normalize_topk_prob != 0` applies the Qwen4 selected-probability
// renormalization after the float32 softmax.

// bf16_round_trip_f32's rounding (RNE; non-finite values unchanged).
__device__ __forceinline__ float router_bf16_rt(float v) {
    if (!isfinite(v)) return v;
    const unsigned int u = __float_as_uint(v);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

// Wave32 max / min, the result in every lane. Max and min are exact and
// order-independent, so the reduction shape does not matter; gfx11/gfx12 use
// row_xmask DPP and permlanex16 instead of LDS-crossbar shuffles.
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
    // The cast keeps min() in int: with permlanex16's unsigned result it
    // resolves through double (four F64 ops per call).
    v = min(v, (int)__builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false));
#else
    for (int offset = 16; offset > 0; offset >>= 1) v = min(v, __shfl_xor(v, offset, 32));
#endif
    return v;
}

// `round_logits != 0` (the BF16 recipe) reads every logit through that BF16
// round trip, the source storage boundary a separate bf16_round_trip_f32
// launch would apply, and stores the weights through it as well.
__launch_bounds__(256, 1)
extern "C" __global__ void moe_router_softmax_top10_f32(
    const float* __restrict__ logits, // [tokens, n_exp]
    int* __restrict__ topk_idx,       // [tokens, 10]
    float* __restrict__ topk_w,       // [tokens, 10]
    int n_exp,
    int top_k,
    int normalize_topk_prob,
    int round_logits
) {
    __shared__ float wave_values[80];
    __shared__ int wave_indices[80];
    __shared__ float selected_values[10];
    __shared__ float probabilities[10];
    const int thread = threadIdx.x;
    const int lane = thread & 31;
    const int wave = thread >> 5;
    const int token = blockIdx.x;

    // The sealed E512/top-10 route checks this geometry before launch. Keep the guard uniform
    // across the workgroup so no lane can return across the block barrier.
    if (n_exp != 512 || top_k != 10) return;

    const size_t token_base = (size_t)token * 512;
    float first_logit = logits[token_base + thread];
    float second_logit = logits[token_base + thread + 256];
    if (round_logits != 0) {
        first_logit = router_bf16_rt(first_logit);
        second_logit = router_bf16_rt(second_logit);
    }
    float first_value = first_logit;
    float second_value = second_logit;
    const int first_index = thread;
    const int second_index = thread + 256;

    // The global top ten must be among each wave's local top ten. Each rank
    // takes the wave max, then the lowest index holding it: the
    // stable (value desc, index asc) order.
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

    // Wave 0 merges the 80 wave-local candidates the same way; each lane owns
    // at most three.
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

    // Preserve the reference's float32 operation order: full softmax in
    // expert-index order, followed by selected-probability renormalization.
    // The 512 exponentials are computed in parallel into LDS; one serial pass
    // then sums them in expert order, keeping routed-output numerics.
    __shared__ __align__(16) float exps[512];
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
        // Still one left-to-right sum; 16-byte reads batched ahead of the
        // adds so the chain waits on add latency, not 512 LDS round trips.
#pragma unroll 16
        for (int expert = 0; expert < 512; expert += 4) {
            const float4 e = *(const float4*)(exps + expert);
            denom += e.x;
            denom += e.y;
            denom += e.z;
            denom += e.w;
        }
        if (!(denom > 0.0f) || !isfinite(denom)) return;

        const float inv_denom = 1.0f / denom;
        for (int rank = 0; rank < top_k; ++rank)
            probabilities[rank] = expf(selected_values[rank] - max_logit) * inv_denom;

        if (normalize_topk_prob != 0) {
            float selected_sum = 0.0f;
            for (int rank = 0; rank < top_k; ++rank)
                selected_sum += probabilities[rank];
            if (!(selected_sum > 0.0f) || !isfinite(selected_sum)) return;
            const float inv_selected = 1.0f / selected_sum;
            for (int rank = 0; rank < top_k; ++rank)
                probabilities[rank] *= inv_selected;
        }

        for (int rank = 0; rank < top_k; ++rank)
            topk_w[output_base + rank] =
                round_logits != 0 ? router_bf16_rt(probabilities[rank]) : probabilities[rank];
    }
}
