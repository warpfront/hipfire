// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// MoE combine with an explicit rounding policy, one thread per output column.
//
//   acc = 0
//   for each routed rank j (rank order, or ascending expert id):
//       v = down_rows[r(j), col]            r = row_of_slot[row*top_k + j] or the slot itself
//       v = bf16(v)                         if round_expert_outputs
//       acc = acc + v * w[j]                (separate rounded multiply and add: no fma)
//   acc = bf16(acc)                         if round_routed_sum
//   routed_out[row, col] = acc              if routed_out
//   acc = acc + shared[row, col]            if shared
//   acc = bf16(acc)                         if round_final
//   out[row, col] = acc
//
// Ranks with an expert id outside [0, n_experts), an unrouted slot (row < 0) or
// a row outside [0, n_down_rows) are skipped.  In ascending-expert-id order,
// out-of-range ids sort last (and are skipped); equal ids keep rank order.
//
// Grid  (rows, ceil(hidden / 256))   Block 256   LDS 0
// Kernarg order:
//   down_rows*, row_of_slot* (nullable), topk_idx*, topk_w*, shared* (nullable),
//   routed_out* (nullable), out*, rows, hidden, top_k, n_experts, order,
//   round_expert_outputs, round_routed_sum, round_final, n_down_rows
// top_k <= 8.
//
// ffn_round_bf16_f32(x, n): in place BF16 round-to-nearest-even of x[0..n).
// Grid ceil(n / 256), block 256.

#include <hip/hip_runtime.h>

static __device__ __forceinline__ float ffn_bf16_rne(float x) {
    unsigned int u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;  // NaN passes through
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    return __uint_as_float(u);
}

extern "C" __global__ __launch_bounds__(256) void ffn_moe_combine_f32(
    const float* down_rows,
    const int* row_of_slot,
    const int* topk_idx,
    const float* topk_w,
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
    const long long row = blockIdx.x;
    const int col = blockIdx.y * 256 + threadIdx.x;
    if (row >= rows || col >= hidden) return;

    const int* ids = topk_idx + row * (long long)top_k;
    const float* ws = topk_w + row * (long long)top_k;

    int ord[8];
    int key[8];
    for (int j = 0; j < 8; ++j) {
        ord[j] = j;
        key[j] = 0x7fffffff;
    }
    for (int j = 0; j < top_k; ++j) {
        const int e = ids[j];
        key[j] = (e < 0 || e >= n_experts) ? 0x7fffffff : e;
    }
    if (order == 1) {
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
        const long long r = (row_of_slot != nullptr) ? (long long)row_of_slot[slot] : slot;
        if (r < 0 || r >= n_down_rows) continue;
        float v = down_rows[r * (long long)hidden + col];
        if (round_expert_outputs) v = ffn_bf16_rne(v);
        const float prod = __fmul_rn(v, ws[j]);
        acc = __fadd_rn(acc, prod);
    }
    if (round_routed_sum) acc = ffn_bf16_rne(acc);
    const long long o = row * (long long)hidden + col;
    if (routed_out != nullptr) routed_out[o] = acc;
    if (shared != nullptr) acc = __fadd_rn(acc, shared[o]);
    if (round_final) acc = ffn_bf16_rne(acc);
    out[o] = acc;
}

extern "C" __global__ __launch_bounds__(256) void ffn_round_bf16_f32(float* x, int n) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i < n) x[i] = ffn_bf16_rne(x[i]);
}
