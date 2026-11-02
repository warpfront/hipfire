// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — S1 slow-link layer-major prefill kernels (panel combine, ownership masks, cohort
// counts, sorted-row remap, row merge).  A module of their own so the legacy
// moe_down_combine_grouped_top10 source (and its recorded identity) stays byte-for-byte unchanged.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

// Verbatim copies of the two rounding helpers of moe_down_combine_grouped_top10.hip: the BF16
// contract is static_cast<__hip_bfloat16>(float) (RNE) and its widening.
static __device__ __forceinline__ __hip_bfloat16 moe_top10_bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

static __device__ __forceinline__ float moe_top10_bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

// ---------------------------------------------------------------------------
// S1 slow-link layer-major prefill: the ordered combine of bounded expert
// panels.  The accumulator `acc` is the F32 widening of the running BF16 sum
// per token/column (+0.0 at the start of the layer, no residual).  Every valid
// route slot of a token is applied exactly once, in exactly the canonical
// order of moe_combine_order_top10 (ascending expert id, ties by rank), with
// exactly the arithmetic of moe_down_combine_grouped_top10_bf16in (product and
// each sum rounded to BF16), but across several launches: one launch per
// panel of grouped rows, panels visited in ascending grouped-row order.
//
// A slot belongs to a panel iff its id is valid and
//   * its grouped row is valid (0 <= row < grouped_rows) and lies inside
//     [row_base, row_base + panel_rows)  -- stable grouping puts a slot's row
//     inside its own expert's rows, so ascending panels preserve the canonical
//     order even when one expert's rows are split across panels; or
//   * its grouped row is invalid (the canonical kernel then multiplies +0 by
//     the weight) and its id lies in the panel's ownership range
//     [own_lo, own_hi); the ownership ranges of the panels of a layer
//     partition [0, n_experts) in panel order.
// Slots of other panels are SKIPPED, not added as zeros: a skipped term leaves
// the accumulator untouched, while an extra +0 term would turn a -0 into +0.
// Unrouted ranks (ids outside [0, n_experts)) are never applied here; the
// canonical kernel's trailing (10 - valid) +0 products run once per layer in
// slowlink_combine_pad.  No atomics, no completion-order sums.
// Grid [ceil(hidden / 512), tokens], block 64, hidden % 8 == 0.
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(64) void slowlink_combine_panel(
    const unsigned short* __restrict__ y,         // [panel_rows, hidden] BF16 bits, panel-local rows
    const int* __restrict__ inverse_perm,         // [tokens * 10] flat slot -> window grouped row
    const int* __restrict__ topk,                 // [tokens * 10] expert ids
    const float* __restrict__ weights,            // [tokens * 10]
    float* __restrict__ acc,                      // [tokens, hidden] widened BF16 accumulator
    int hidden, int tokens, int n_experts, int grouped_rows,
    int row_base, int panel_rows, int own_lo, int own_hi) {
    const int token = blockIdx.y;
    if (token >= tokens) return;
    const size_t rb = (size_t)token * 10;
    int ids[10];
    int rows[10];
    int used = 0, mine = 0;
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        const int id = topk[rb + r];
        const int row = inverse_perm[rb + r];
        ids[r] = id;
        rows[r] = row;
        if (id < 0 || id >= n_experts) {
            used |= 1 << r;
            continue;
        }
        const bool row_ok = row >= 0 && row < grouped_rows;
        const bool owned = row_ok ? (row >= row_base && row - row_base < panel_rows)
                                  : (id >= own_lo && id < own_hi);
        if (owned) mine |= 1 << r;
    }
    if (mine == 0) return;  // uniform per block
    const int col0 = (blockIdx.x * 64 + threadIdx.x) * 8;
    if (col0 >= hidden) return;
    used |= ~mine;
    float* a = acc + (size_t)token * hidden + col0;
    float v[8];
    {
        const float4 a0 = *(const float4*)a, a1 = *(const float4*)(a + 4);
        v[0] = a0.x; v[1] = a0.y; v[2] = a0.z; v[3] = a0.w;
        v[4] = a1.x; v[5] = a1.y; v[6] = a1.z; v[7] = a1.w;
    }
    for (int count = 0; count < 10; ++count) {
        int sel = -1, sel_id = 0x7fffffff;
#pragma unroll
        for (int r = 0; r < 10; ++r) {
            if (!(used & (1 << r)) && ids[r] < sel_id) {
                sel_id = ids[r];
                sel = r;
            }
        }
        if (sel < 0) break;
        used |= 1 << sel;
        const float wt = moe_top10_bf16_to_f32(moe_top10_bf16_round(weights[rb + sel]));
        const int row = rows[sel];
        uint4 packed = make_uint4(0, 0, 0, 0);
        if (row >= 0 && row < grouped_rows)
            packed = *(const uint4*)(y + (size_t)(row - row_base) * hidden + col0);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const unsigned int w = (&packed.x)[e >> 1];
            const float raw_output = __uint_as_float((e & 1) ? (w & 0xFFFF0000u) : (w << 16));
            const __hip_bfloat16 output = moe_top10_bf16_round(raw_output);
            const __hip_bfloat16 weighted = moe_top10_bf16_round(
                moe_top10_bf16_to_f32(output) * wt);
            v[e] = moe_top10_bf16_to_f32(moe_top10_bf16_round(
                v[e] + moe_top10_bf16_to_f32(weighted)));
        }
    }
    *(float4*)a = make_float4(v[0], v[1], v[2], v[3]);
    *(float4*)(a + 4) = make_float4(v[4], v[5], v[6], v[7]);
}

// The canonical kernel's terminal products, once per layer: every token adds
// (10 - valid) +0 products (bf16(bf16(+0) * bf16(+0)) = +0) after its last
// valid rank.  Only a -0 accumulator changes.  Same grid as the panel kernel.
// A row whose first route id is SLOWLINK_ROW_DONE is a cohort row whose incumbent combine
// (which adds no trailing products) already ran in place, per expert binding: it is skipped.
#define SLOWLINK_ROW_DONE ((int)0x80808080)
extern "C" __global__ __launch_bounds__(64) void slowlink_combine_pad(
    const int* __restrict__ topk, float* __restrict__ acc, int hidden, int tokens,
    int n_experts) {
    const int token = blockIdx.y;
    if (token >= tokens) return;
    const size_t rb = (size_t)token * 10;
    if (topk[rb] == SLOWLINK_ROW_DONE) return;  // uniform per block
    int valid = 0;
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        const int id = topk[rb + r];
        valid += (id >= 0 && id < n_experts) ? 1 : 0;
    }
    if (valid >= 10) return;  // uniform per block
    const int col0 = (blockIdx.x * 64 + threadIdx.x) * 8;
    if (col0 >= hidden) return;
    float* a = acc + (size_t)token * hidden + col0;
    float v[8];
    {
        const float4 a0 = *(const float4*)a, a1 = *(const float4*)(a + 4);
        v[0] = a0.x; v[1] = a0.y; v[2] = a0.z; v[3] = a0.w;
        v[4] = a1.x; v[5] = a1.y; v[6] = a1.z; v[7] = a1.w;
    }
    const float zero = moe_top10_bf16_to_f32(moe_top10_bf16_round(0.0f * 0.0f));
    for (int k = valid; k < 10; ++k) {
#pragma unroll
        for (int e = 0; e < 8; ++e)
            v[e] = moe_top10_bf16_to_f32(moe_top10_bf16_round(v[e] + zero));
    }
    *(float4*)a = make_float4(v[0], v[1], v[2], v[3]);
    *(float4*)(a + 4) = make_float4(v[4], v[5], v[6], v[7]);
}

// S1 K1-cohort support: the ids of a short-tail tile restricted to one expert binding
// [lo, hi); every other id becomes -1 (the K1 kernels' no-load exit), so a K1 launch against
// one staged binding computes exactly the slots whose expert that binding holds.
extern "C" __global__ __launch_bounds__(256) void slowlink_mask_ids(
    const int* __restrict__ ids, int* __restrict__ out, int n, int lo, int hi) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int v = ids[i];
    out[i] = (v >= lo && v < hi) ? v : -1;
}

// S1 single count barrier: count the cohort (K1 short-tail / F32-tail / decode tiles) slots
// per expert into the layout's cohort count words, read back beside the stable grouping's
// padded counts at the one barrier.  Integer atomic adds are order-independent; ids outside
// [0, n_exp) are unrouted and add nothing.
extern "C" __global__ __launch_bounds__(256) void slowlink_fold_k1_counts(
    const int* __restrict__ ids, int* __restrict__ counts, int n, int n_exp) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int v = ids[i];
    if (v >= 0 && v < n_exp) atomicAdd(&counts[v], 1);
}

// S1 aggregate-route panel support: re-index one panel's grouped rows to a compact local slot
// space, `local[i] = i` for a real row and -1 for a pad row, so the panel's unscatter/rotation
// and down GEMM address a panel-sized F16 activation instead of one row per window slot.
extern "C" __global__ __launch_bounds__(256) void slowlink_remap_sorted(
    const int* __restrict__ sorted, int* __restrict__ local, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    local[i] = sorted[i] >= 0 ? i : -1;
}

// S1 aggregate-route K1 cohort: the aggregate lane-union GEMVs write EVERY slot's row (zero
// for an id outside the launch's binding), so one binding's launch cannot own the shared
// output.  Copy the rows of the slots whose id lies in [lo, hi) from the binding's scratch
// `src` to the persistent `dst`; every other slot's row of `dst` is left untouched.
// Grid (slots, ceil(width / 256)), block 256.
extern "C" __global__ __launch_bounds__(256) void slowlink_merge_rows(
    const float* __restrict__ src, float* __restrict__ dst, const int* __restrict__ ids,
    int slots, int width, int lo, int hi) {
    const int s = blockIdx.x;
    if (s >= slots) return;
    const int id = ids[s];
    if (id < lo || id >= hi) return;
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= width) return;
    dst[(long long)s * width + j] = src[(long long)s * width + j];
}
