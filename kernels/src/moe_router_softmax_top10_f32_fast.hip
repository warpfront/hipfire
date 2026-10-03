// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — fixed E512/top-10 softmax router, wave-per-token (opt-in
// `HIPFIRE_QWEN4_ROUTER_FAST`).  Same outputs, byte for byte, as
// moe_router_softmax_top10_f32 (the incumbent stays the decode / recorded
// symbol and is the exactness reference).
//
// The incumbent runs one 256-thread workgroup per token: ten wave-local
// selection rounds, a barrier, a merge, a barrier, 512 exponentials into LDS,
// a barrier, and then ONE thread sums the 512 exponentials serially while the
// other 255 wait.  Here one wave32 owns one token (8 tokens per workgroup,
// no workgroup barriers) and keeps the 512 logits in 16 registers per lane.
//
// Exactness:
//  * Selection.  For finite logits the incumbent's two-stage (per-wave top ten,
//    then merge) selection is the global stable top ten in (value desc, index
//    asc) order, which ten rounds of "wave max, lowest index holding it,
//    remove" over all 512 entries produce directly.
//  * Non-finite logits (NaN, +/-inf; checked after the optional BF16 round
//    trip) make that equivalence fail (removed entries re-tie with real -inf
//    entries, NaN never equals the max), so such a token runs the incumbent's
//    two-stage selection verbatim, one virtual wave at a time.
//  * The softmax denominator is the same left-to-right float32 sum over
//    expert index 0..511 (one lane, LDS-staged), the same expf calls, the same
//    selected-probability renormalization and early-return rules.

#include <hip/hip_runtime.h>
#include <math.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__) && !defined(__gfx1201__)
#error "moe_router_softmax_top10_f32_fast.hip is gfx1151/gfx1201-only"
#endif

// bf16_round_trip_f32's rounding (RNE; non-finite values unchanged).
__device__ __forceinline__ float router_bf16_rt(float v) {
    if (!isfinite(v)) return v;
    const unsigned int u = __float_as_uint(v);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

__device__ __forceinline__ float router_wave_max(float v) {
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x161, 0xf, 0xf, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x162, 0xf, 0xf, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x164, 0xf, 0xf, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x168, 0xf, 0xf, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false)));
    return v;
}

__device__ __forceinline__ int router_wave_min(int v) {
    v = min(v, __builtin_amdgcn_update_dpp(0, v, 0x161, 0xf, 0xf, false));
    v = min(v, __builtin_amdgcn_update_dpp(0, v, 0x162, 0xf, 0xf, false));
    v = min(v, __builtin_amdgcn_update_dpp(0, v, 0x164, 0xf, 0xf, false));
    v = min(v, __builtin_amdgcn_update_dpp(0, v, 0x168, 0xf, 0xf, false));
    v = min(v, __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false));
    return v;
}

__device__ __forceinline__ void router_wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
    __builtin_amdgcn_wave_barrier();
}

#define ROUTER_TOKENS_PER_BLOCK 8

__launch_bounds__(256)
extern "C" __global__ void moe_router_softmax_top10_f32_fast(
    const float* __restrict__ logits, // [tokens, 512]
    int* __restrict__ topk_idx,       // [tokens, 10]
    float* __restrict__ topk_w,       // [tokens, 10]
    int n_exp,
    int top_k,
    int normalize_topk_prob,
    int round_logits,
    int tokens
) {
    __shared__ float s_exps[ROUTER_TOKENS_PER_BLOCK][512];
    __shared__ float s_cv[ROUTER_TOKENS_PER_BLOCK][80];
    __shared__ int s_ci[ROUTER_TOKENS_PER_BLOCK][80];

    // Uniform across the workgroup; no workgroup barrier is used below.
    if (n_exp != 512 || top_k != 10) return;
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int token = blockIdx.x * ROUTER_TOKENS_PER_BLOCK + wave;
    if (token >= tokens) return;

    const float* row = logits + (size_t)token * 512;
    float x[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) x[j] = row[lane + 32 * j];
    if (round_logits != 0) {
#pragma unroll
        for (int j = 0; j < 16; ++j) x[j] = router_bf16_rt(x[j]);
    }

    int nonfinite = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) nonfinite |= !isfinite(x[j]);
    const bool any_nonfinite = __any(nonfinite);

    float bv[10];
    int bi[10];
    if (!any_nonfinite) {
        float v[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) v[j] = x[j];
#pragma unroll
        for (int rank = 0; rank < 10; ++rank) {
            float m = v[0];
#pragma unroll
            for (int j = 1; j < 16; ++j) m = fmaxf(m, v[j]);
            const float best = router_wave_max(m);
            int cand = 0x7fffffff;
#pragma unroll
            for (int j = 15; j >= 0; --j) cand = v[j] == best ? lane + 32 * j : cand;
            const int best_index = router_wave_min(cand);
            bv[rank] = best;
            bi[rank] = best_index;
#pragma unroll
            for (int j = 0; j < 16; ++j) v[j] = (lane + 32 * j) == best_index ? -INFINITY : v[j];
        }
    } else {
        // Incumbent selection verbatim: eight virtual waves of 64 entries each
        // pick a local top ten, then a 32-lane merge of the 80 candidates.
#pragma unroll 1
        for (int vw = 0; vw < 8; ++vw) {
            float first_value = row[vw * 32 + lane];
            float second_value = row[256 + vw * 32 + lane];
            if (round_logits != 0) {
                first_value = router_bf16_rt(first_value);
                second_value = router_bf16_rt(second_value);
            }
            const int first_index = vw * 32 + lane;
            const int second_index = 256 + vw * 32 + lane;
            for (int rank = 0; rank < 10; ++rank) {
                const float best_value = router_wave_max(fmaxf(first_value, second_value));
                const int candidate = first_value == best_value ? first_index
                    : second_value == best_value ? second_index : 0x7fffffff;
                const int best_index = router_wave_min(candidate);
                if (lane == 0) {
                    s_cv[wave][vw * 10 + rank] = best_value;
                    s_ci[wave][vw * 10 + rank] = best_index;
                }
                if (first_index == best_index) first_value = -INFINITY;
                if (second_index == best_index) second_value = -INFINITY;
            }
        }
        router_wave_sync();
        float value0 = s_cv[wave][lane];
        float value1 = s_cv[wave][lane + 32];
        float value2 = lane + 64 < 80 ? s_cv[wave][lane + 64] : -INFINITY;
        const int index0 = s_ci[wave][lane];
        const int index1 = s_ci[wave][lane + 32];
        const int index2 = lane + 64 < 80 ? s_ci[wave][lane + 64] : 0x7fffffff;
#pragma unroll
        for (int rank = 0; rank < 10; ++rank) {
            const float best_value = router_wave_max(fmaxf(fmaxf(value0, value1), value2));
            int candidate = 0x7fffffff;
            if (value0 == best_value) candidate = min(candidate, index0);
            if (value1 == best_value) candidate = min(candidate, index1);
            if (value2 == best_value) candidate = min(candidate, index2);
            const int best_index = router_wave_min(candidate);
            bv[rank] = best_value;
            bi[rank] = best_index;
            if (index0 == best_index) value0 = -INFINITY;
            if (index1 == best_index) value1 = -INFINITY;
            if (index2 == best_index) value2 = -INFINITY;
        }
    }

    const size_t output_base = (size_t)token * 10;
    if (lane == 0) {
#pragma unroll
        for (int rank = 0; rank < 10; ++rank) topk_idx[output_base + rank] = bi[rank];
    }

    // 512 exponentials in parallel, then the incumbent's left-to-right sum.
    const float max_logit = bv[0];
#pragma unroll
    for (int j = 0; j < 16; ++j) s_exps[wave][lane + 32 * j] = expf(x[j] - max_logit);
    router_wave_sync();
    if (lane == 0) {
        const float4* e4 = reinterpret_cast<const float4*>(s_exps[wave]);
        float denom = 0.0f;
#pragma unroll 8
        for (int q = 0; q < 128; ++q) {
            const float4 e = e4[q];
            denom += e.x;
            denom += e.y;
            denom += e.z;
            denom += e.w;
        }
        if (!(denom > 0.0f) || !isfinite(denom)) return;

        float probabilities[10];
        const float inv_denom = 1.0f / denom;
#pragma unroll
        for (int rank = 0; rank < 10; ++rank)
            probabilities[rank] = expf(bv[rank] - max_logit) * inv_denom;

        if (normalize_topk_prob != 0) {
            float selected_sum = 0.0f;
#pragma unroll
            for (int rank = 0; rank < 10; ++rank) selected_sum += probabilities[rank];
            if (!(selected_sum > 0.0f) || !isfinite(selected_sum)) return;
            const float inv_selected = 1.0f / selected_sum;
#pragma unroll
            for (int rank = 0; rank < 10; ++rank) probabilities[rank] *= inv_selected;
        }

#pragma unroll
        for (int rank = 0; rank < 10; ++rank)
            topk_w[output_base + rank] =
                round_logits != 0 ? router_bf16_rt(probabilities[rank]) : probabilities[rank];
    }
}
