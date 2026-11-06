// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// `ffn_moe_combine_f32_w2`: the gfx1201 export of the MoE combine with an explicit rounding policy
// (`ffn_moe_combine_f32`, kernels/src/ffn_moe_combine_policy.hip). Same 92-byte kernarg order, same grid
// (rows, ceil(hidden / 256)), same output bits; two output columns per thread (float2), block 128.
//
//   acc = +0; for each rank j in `order` (0: rank order, 1: stable ascending expert id; ranks with an out-of-range id,
//   an unrouted slot (row < 0) or a row >= n_down_rows are skipped):  v = down_rows[r(j), col]; v = bf16(v) if
//   round_expert_outputs; acc = __fadd_rn(acc, __fmul_rn(v, w[j]))   (this hipcc contracts it to ONE fma per rank, in the
//   incumbent object and here; the product is exact in neither, so the numeric contract is the fused chain)
//   acc = bf16(acc) if round_routed_sum; routed_out = acc; acc = __fadd_rn(acc, shared) if shared; bf16 if round_final.
//
// Per output element the operation sequence is the incumbent's. Only data movement and instruction overhead change:
//  * the rank order, validity and gather rows are computed ONCE per wave on the scalar unit (stable sort = sorting network on
//    packed (key << 3 | rank) words; keys are < 2^28 so the packed word is 31 bits) instead of per lane through dynamically
//    indexed VGPR arrays (the incumbent's v_movrel / readfirstlane waterfall loops dominate its instruction count);
//  * all K row loads of a thread are issued back to back (clamped to row 0 when a rank is skipped; the skip is a select on
//    the accumulate), not one dependent load per iteration;
//  * two columns per thread (one 8-byte load per rank), block 128, one CTA = one row x one 256-column tile.
// Everything outside the fast path (top_k not in 1..8, n_experts >= 2^28, n_down_rows <= 0, odd hidden or a pointer
// that is not 8-byte aligned) runs the incumbent's per-column code verbatim.
//
// Grid  (rows, ceil(hidden / 256))   Block 128   LDS 0
// Kernarg order: as `ffn_moe_combine_f32`. top_k <= 8.

#include <hip/hip_runtime.h>

#define CB_W 2                 // columns per thread (one float2 per rank)
#define CB_BLK (256 / CB_W)    // threads per CTA

typedef float cb_t __attribute__((ext_vector_type(2)));
#define CB_C(v, c) ((v)[c])
#define CB_SET(v, c, x) ((v)[c] = (x))

static __device__ __forceinline__ float ffn_bf16_rne(float x) {
    unsigned int u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;  // NaN passes through
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    return __uint_as_float(u);
}

struct CbArgs {
    const float* down_rows;
    const int* row_of_slot;
    const int* topk_idx;
    const float* topk_w;
    const float* shared;
    float* routed_out;
    float* out;
    int rows, hidden, top_k, n_experts, order, rex, rrs, rfin, n_down_rows;
};

// The incumbent's per-column body, verbatim (fallback).
static __device__ __forceinline__ void cb_slow_col(const CbArgs& a, long long row, int col) {
    if (row >= a.rows || col >= a.hidden) return;
    const int top_k = a.top_k;
    const int* ids = a.topk_idx + row * (long long)top_k;
    const float* ws = a.topk_w + row * (long long)top_k;

    int ord[8];
    int key[8];
    for (int j = 0; j < 8; ++j) {
        ord[j] = j;
        key[j] = 0x7fffffff;
    }
    for (int j = 0; j < top_k; ++j) {
        const int e = ids[j];
        key[j] = (e < 0 || e >= a.n_experts) ? 0x7fffffff : e;
    }
    if (a.order == 1) {
        // stable insertion sort by ascending expert id
        for (int i = 1; i < top_k; ++i) {
            const int oi = ord[i];
            const int ki = key[oi];
            int p = i - 1;
            while (p >= 0 && key[ord[p]] > ki) {
                ord[p + 1] = ord[p];
                --p;
            }
            ord[p + 1] = oi;
        }
    }

    float acc = 0.0f;
    for (int i = 0; i < top_k; ++i) {
        const int j = ord[i];
        if (key[j] == 0x7fffffff) continue;
        const long long slot = row * (long long)top_k + j;
        const long long r = (a.row_of_slot != nullptr) ? (long long)a.row_of_slot[slot] : slot;
        if (r < 0 || r >= a.n_down_rows) continue;
        float v = a.down_rows[r * (long long)a.hidden + col];
        if (a.rex) v = ffn_bf16_rne(v);
        const float prod = __fmul_rn(v, ws[j]);
        acc = __fadd_rn(acc, prod);
    }
    if (a.rrs) acc = ffn_bf16_rne(acc);
    const long long o = row * (long long)a.hidden + col;
    if (a.routed_out != nullptr) a.routed_out[o] = acc;
    if (a.shared != nullptr) acc = __fadd_rn(acc, a.shared[o]);
    if (a.rfin) acc = ffn_bf16_rne(acc);
    a.out[o] = acc;
}

#define CB_CE(i, j)                                  \
    {                                                \
        const unsigned lo_ = pk[i] < pk[j] ? pk[i] : pk[j]; \
        const unsigned hi_ = pk[i] < pk[j] ? pk[j] : pk[i]; \
        pk[i] = lo_;                                 \
        pk[j] = hi_;                                 \
    }

// stable sort of the packed (key << 3 | rank) words: the rank sits in the low bits, so equal keys keep rank order
static __device__ __forceinline__ void cb_sort(unsigned (&pk)[1]) { (void)pk; }
static __device__ __forceinline__ void cb_sort(unsigned (&pk)[2]) { CB_CE(0, 1); }
static __device__ __forceinline__ void cb_sort(unsigned (&pk)[3]) { CB_CE(0, 2); CB_CE(0, 1); CB_CE(1, 2); }
static __device__ __forceinline__ void cb_sort(unsigned (&pk)[4]) { CB_CE(0, 2); CB_CE(1, 3); CB_CE(0, 1); CB_CE(2, 3); CB_CE(1, 2); }
static __device__ __forceinline__ void cb_sort(unsigned (&pk)[5]) { CB_CE(0, 3); CB_CE(1, 4); CB_CE(0, 2); CB_CE(1, 3); CB_CE(0, 1); CB_CE(2, 4); CB_CE(1, 2); CB_CE(3, 4); CB_CE(2, 3); }
static __device__ __forceinline__ void cb_sort(unsigned (&pk)[6]) { CB_CE(0, 5); CB_CE(1, 3); CB_CE(2, 4); CB_CE(1, 2); CB_CE(3, 4); CB_CE(0, 3); CB_CE(2, 5); CB_CE(0, 1); CB_CE(2, 3); CB_CE(4, 5); CB_CE(1, 2); CB_CE(3, 4); }
static __device__ __forceinline__ void cb_sort(unsigned (&pk)[7]) { CB_CE(0, 1); CB_CE(2, 3); CB_CE(0, 2); CB_CE(1, 3); CB_CE(1, 2); CB_CE(4, 5); CB_CE(4, 6); CB_CE(5, 6); CB_CE(0, 4); CB_CE(2, 6); CB_CE(2, 4); CB_CE(1, 5); CB_CE(3, 5); CB_CE(1, 2); CB_CE(3, 4); CB_CE(5, 6); }
static __device__ __forceinline__ void cb_sort(unsigned (&pk)[8]) { CB_CE(0, 2); CB_CE(1, 3); CB_CE(4, 6); CB_CE(5, 7); CB_CE(0, 4); CB_CE(1, 5); CB_CE(2, 6); CB_CE(3, 7); CB_CE(0, 1); CB_CE(2, 3); CB_CE(4, 5); CB_CE(6, 7); CB_CE(2, 4); CB_CE(3, 5); CB_CE(1, 4); CB_CE(3, 6); CB_CE(1, 2); CB_CE(3, 4); CB_CE(5, 6); }

// Fast path: top_k == K. Every thread handles columns col0 .. col0 + CB_W - 1 of (row, tile).
template <int K>
static __device__ __forceinline__ void cb_fast(const CbArgs& a, long long row, int tile) {
    const unsigned inv = (unsigned)a.n_experts;  // key of a skipped rank: sorts after every valid id
    const int* ids = a.topk_idx + row * K;
    unsigned pk[K];
#pragma unroll
    for (int j = 0; j < K; ++j) {
        const int e = ids[j];
        const unsigned key = (e < 0 || e >= a.n_experts) ? inv : (unsigned)e;
        pk[j] = (key << 3) | (unsigned)j;
    }
    if (a.order == 1) cb_sort(pk);

    const bool has_ros = a.row_of_slot != nullptr;
    const int* rp = has_ros ? a.row_of_slot : a.topk_idx;  // a valid address when row_of_slot is null
    // phase 1: every scalar load (slot -> row, weight) issued back to back, addressed by the sorted rank
    int raw[K];
    float w[K];
#pragma unroll
    for (int q = 0; q < K; ++q) {
        const long long slot = row * K + (long long)(pk[q] & 7u);
        raw[q] = rp[slot];
        w[q] = a.topk_w[slot];
    }
    // phase 2: gather rows, validity
    long long rr[K];
    bool ok[K];
    bool all_ok = true;
#pragma unroll
    for (int q = 0; q < K; ++q) {
        const unsigned p = pk[q];
        const long long slot = row * K + (long long)(p & 7u);
        const long long r = has_ros ? (long long)raw[q] : slot;
        const bool v = ((p >> 3) != inv) & (r >= 0) & (r < (long long)a.n_down_rows);
        ok[q] = v;
        rr[q] = v ? r : 0;  // n_down_rows > 0 here, so row 0 exists
        all_ok = all_ok & v;
    }

    const int col0 = tile * 256 + (int)threadIdx.x * CB_W;
    if (col0 >= a.hidden) return;
    const unsigned off = (unsigned)col0 * 4u;
    cb_t x[K];
#pragma unroll
    for (int q = 0; q < K; ++q)
        x[q] = *reinterpret_cast<const cb_t*>(reinterpret_cast<const char*>(a.down_rows + rr[q] * (long long)a.hidden) + off);
    const long long o = row * (long long)a.hidden + col0;
    cb_t sh = {};
    if (a.shared != nullptr) sh = *reinterpret_cast<const cb_t*>(a.shared + o);

    float acc[CB_W];
#pragma unroll
    for (int c = 0; c < CB_W; ++c) acc[c] = 0.0f;
    if (all_ok && a.rex) {
#pragma unroll
        for (int q = 0; q < K; ++q)
#pragma unroll
            for (int c = 0; c < CB_W; ++c)
                acc[c] = __fadd_rn(acc[c], __fmul_rn(ffn_bf16_rne(CB_C(x[q], c)), w[q]));
    } else if (all_ok) {
#pragma unroll
        for (int q = 0; q < K; ++q)
#pragma unroll
            for (int c = 0; c < CB_W; ++c)
                acc[c] = __fadd_rn(acc[c], __fmul_rn(CB_C(x[q], c), w[q]));
    } else {
#pragma unroll
        for (int q = 0; q < K; ++q)
#pragma unroll
            for (int c = 0; c < CB_W; ++c) {
                float v = CB_C(x[q], c);
                if (a.rex) v = ffn_bf16_rne(v);
                const float na = __fadd_rn(acc[c], __fmul_rn(v, w[q]));
                acc[c] = ok[q] ? na : acc[c];
            }
    }
    if (a.rrs) {
#pragma unroll
        for (int c = 0; c < CB_W; ++c) acc[c] = ffn_bf16_rne(acc[c]);
    }
    cb_t res;
#pragma unroll
    for (int c = 0; c < CB_W; ++c) CB_SET(res, c, acc[c]);
    if (a.routed_out != nullptr) *reinterpret_cast<cb_t*>(a.routed_out + o) = res;
    if (a.shared != nullptr) {
#pragma unroll
        for (int c = 0; c < CB_W; ++c) acc[c] = __fadd_rn(acc[c], CB_C(sh, c));
    }
    if (a.rfin) {
#pragma unroll
        for (int c = 0; c < CB_W; ++c) acc[c] = ffn_bf16_rne(acc[c]);
    }
#pragma unroll
    for (int c = 0; c < CB_W; ++c) CB_SET(res, c, acc[c]);
    *reinterpret_cast<cb_t*>(a.out + o) = res;
}

extern "C" __global__ __launch_bounds__(CB_BLK) void ffn_moe_combine_f32_w2(
    const float* __restrict__ down_rows,
    const int* __restrict__ row_of_slot,
    const int* __restrict__ topk_idx,
    const float* __restrict__ topk_w,
    const float* shared,
    float* routed_out,
    float* out,
    int rows,
    int hidden,
    int top_k,
    int n_experts,
    int order,
    int round_expert_outputs,
    int round_routed_sum,
    int round_final,
    int n_down_rows
) {
    if (rows <= 0 || hidden <= 0) return;
    const long long row = blockIdx.x;
    const int tile = (int)blockIdx.y;
    if (row >= rows) return;
    const CbArgs a = {down_rows, row_of_slot, topk_idx, topk_w, shared, routed_out, out,
                      rows, hidden, top_k, n_experts, order, round_expert_outputs, round_routed_sum, round_final, n_down_rows};

    bool fast = top_k >= 1 && top_k <= 8 && n_experts >= 1 && n_experts < (1 << 28) && n_down_rows > 0 &&
                (hidden % CB_W) == 0;
    {
        const unsigned long long m = 4ull * CB_W - 1ull;
        const unsigned long long al = (unsigned long long)down_rows | (unsigned long long)out |
                                      (unsigned long long)routed_out | (unsigned long long)shared;
        fast = fast && (al & m) == 0;
    }
    if (fast) {
        switch (top_k) {
            case 1: cb_fast<1>(a, row, tile); return;
            case 2: cb_fast<2>(a, row, tile); return;
            case 3: cb_fast<3>(a, row, tile); return;
            case 4: cb_fast<4>(a, row, tile); return;
            case 5: cb_fast<5>(a, row, tile); return;
            case 6: cb_fast<6>(a, row, tile); return;
            case 7: cb_fast<7>(a, row, tile); return;
            default: cb_fast<8>(a, row, tile); return;
        }
    }
    for (int c = (int)threadIdx.x; c < 256; c += CB_BLK) cb_slow_col(a, row, tile * 256 + c);
}
