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
// Exactness (finite rows):
//  * Selection.  For finite logits the incumbent's two-stage (per-wave top ten,
//    then merge) selection is the global stable top ten in (value desc, index
//    asc) order, which ten rounds of "wave max, lowest index holding it,
//    remove" over all 512 entries produce directly.
//  * The softmax denominator is the same left-to-right float32 sum over
//    expert index 0..511 (one lane, LDS-staged), the same expf calls, the same
//    selected-probability renormalization.
//
// Non-finite logits (NaN, +/-inf; checked after the optional BF16 round trip)
// fail the whole row closed exactly as the incumbent does: ids -1, weights 0 and
// a (layer, row) record in the device fault word.

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

// Invalid-row result: expert marker -1 and weight 0 for all ten slots of local
// row `token`, plus the fault record for global row `row`.
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
    int tokens,
    unsigned long long* __restrict__ fault, // host-mapped router fault word, or null
    int layer,                              // sealed call layer index
    int row_base                            // global row of this launch's row 0
) {
    __shared__ float s_exps[ROUTER_TOKENS_PER_BLOCK][512];

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

    // Any NaN/Inf logit (after the optional BF16 round trip) fails the whole row
    // closed: expert marker -1, weight 0 for every slot, and the (row, layer)
    // recorded in the device fault word. The vote is wave-uniform (one wave owns
    // the token) and no later barrier exists, so the wave returns together.
    int nonfinite = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) nonfinite |= !isfinite(x[j]);
    if (__any(nonfinite)) {
        if (lane == 0) router_fail_row(topk_idx, topk_w, token, fault, layer, row_base + token);
        return;
    }

    float bv[10];
    int bi[10];
    {
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
        if (!(denom > 0.0f) || !isfinite(denom)) {
            router_fail_row(topk_idx, topk_w, token, fault, layer, row_base + token);
            return;
        }

        float probabilities[10];
        const float inv_denom = 1.0f / denom;
#pragma unroll
        for (int rank = 0; rank < 10; ++rank)
            probabilities[rank] = expf(bv[rank] - max_logit) * inv_denom;

        if (normalize_topk_prob != 0) {
            float selected_sum = 0.0f;
#pragma unroll
            for (int rank = 0; rank < 10; ++rank) selected_sum += probabilities[rank];
            if (!(selected_sum > 0.0f) || !isfinite(selected_sum)) {
                router_fail_row(topk_idx, topk_w, token, fault, layer, row_base + token);
                return;
            }
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
