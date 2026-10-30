// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Grouped (expert-sorted) FFN gate/up activation with the un-scatter fused in:
//   y_grouped row g = [gate (logical_n) | up (logical_n)]
//   out row (flat slot = sorted_slot_index[g]) =
//       awq_divide( pad( cast_out( act( cast_in(g), cast_in(u) ) ) ) )
// Padding rows (sorted_slot_index[g] < 0) are skipped and write nothing.  A
// flat slot outside [0, total_slots) is skipped as well.  Columns
// [logical_n, phys_n) are +0.0 before the AWQ divide.  When awq_table is
// non-null the expert of grouped row g is expert_tile_ids[g / block_m]; an
// expert outside [0, n_experts) (or a null table entry) writes a zero row.
//
// Entries: ffn_swiglu_grouped_f32, ffn_swiglu_clamp_grouped_f32 (p0 = limit),
//          ffn_situ_grouped_f32 (p0 = gate_beta, p1 = value_beta).
//
// Grid  (m_total, ceil(phys_n / 256))   grouped row on x (limit 2^31)
// Block 256
// LDS   0
//
// Kernarg order:
//   y_grouped*, sorted_slot_index*, expert_tile_ids*, awq_table* (nullable),
//   out*, m_total, block_m, logical_n, phys_n, n_experts, total_slots,
//   round_inputs, round_output, p0, p1

#include <hip/hip_runtime.h>

static __device__ __forceinline__ float ffn_bf16_rne(float x) {
    unsigned int u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;  // NaN passes through
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    return __uint_as_float(u);
}

// ACT: 0 = SwiGLU, 1 = SwiGLU with asymmetric clamp, 2 = SiTU-GLU.
template <int ACT>
static __device__ __forceinline__ float ffn_act(float g, float u, float p0, float p1) {
    if constexpr (ACT == 0) {
        return g / (1.0f + expf(-g)) * u;
    } else if constexpr (ACT == 1) {
        if (g > p0) g = p0;
        if (u > p0) u = p0;
        if (u < -p0) u = -p0;
        return g / (1.0f + expf(-g)) * u;
    } else {
        const float sig = 1.0f / (1.0f + expf(-g));
        const float gate = (p0 * tanhf(g / p0)) * sig;
        const float value = p1 * tanhf(u / p1);
        return gate * value;
    }
}

template <int ACT>
static __device__ __forceinline__ void ffn_grouped_body(
    const float* y_grouped, const int* sorted_slot_index, const int* expert_tile_ids,
    const unsigned long long* awq_table, float* out,
    int m_total, int block_m, int logical_n, int phys_n, int n_experts, int total_slots,
    int round_inputs, int round_output, float p0, float p1
) {
    const int g_row = blockIdx.x;
    if (g_row >= m_total) return;
    const int flat = sorted_slot_index[g_row];
    if (flat < 0 || flat >= total_slots) return;

    const int col = blockIdx.y * 256 + threadIdx.x;
    if (col >= phys_n) return;

    const float* scale = nullptr;
    if (awq_table != nullptr) {
        const int e = expert_tile_ids[g_row / block_m];
        if (e >= 0 && e < n_experts)
            scale = reinterpret_cast<const float*>(awq_table[e]);
        if (scale == nullptr) {
            out[(long long)flat * phys_n + col] = 0.0f;
            return;
        }
    }

    float r = 0.0f;
    if (col < logical_n) {
        const long long y_base = (long long)g_row * 2 * logical_n;
        float g = y_grouped[y_base + col];
        float u = y_grouped[y_base + logical_n + col];
        if (round_inputs) {
            g = ffn_bf16_rne(g);
            u = ffn_bf16_rne(u);
        }
        r = ffn_act<ACT>(g, u, p0, p1);
        if (round_output) r = ffn_bf16_rne(r);
    }
    if (scale != nullptr) r = r / scale[col];
    out[(long long)flat * phys_n + col] = r;
}

#define FFN_GROUPED_ENTRY(NAME, ACT)                                                     \
extern "C" __global__ __launch_bounds__(256) void NAME(                                  \
    const float* y_grouped, const int* sorted_slot_index, const int* expert_tile_ids,    \
    const unsigned long long* awq_table, float* out,                                     \
    int m_total, int block_m, int logical_n, int phys_n, int n_experts, int total_slots, \
    int round_inputs, int round_output, float p0, float p1                               \
) {                                                                                      \
    ffn_grouped_body<ACT>(y_grouped, sorted_slot_index, expert_tile_ids, awq_table,      \
                          out, m_total, block_m, logical_n, phys_n, n_experts,           \
                          total_slots, round_inputs, round_output, p0, p1);              \
}

FFN_GROUPED_ENTRY(ffn_swiglu_grouped_f32, 0)
FFN_GROUPED_ENTRY(ffn_swiglu_clamp_grouped_f32, 1)
FFN_GROUPED_ENTRY(ffn_situ_grouped_f32, 2)

// Single-expert grouping tables: one dense weight driven through the grouped
// WMMA core / indexed GEMV as "expert 0" of a one-entry pointer table.
//   words [0, 6)                      three u64 weight pointers (w0, w1, w2)
//   words [6, 6 + rows_pad)           expert index per row, all 0
//   words [6 + rows_pad, 6 + 2 rows_pad)   sorted_slot_index: i for i < rows, else -1
//   words [6 + 2 rows_pad, +rows_pad/16)   expert_tile_ids, all 0
// rows_pad = rows rounded up to a multiple of 16.
// Grid ceil(rows_pad / 256) (at least 1), block 256.
// Kernarg order: tables*, rows, rows_pad, w0, w1, w2
extern "C" __global__ __launch_bounds__(256) void ffn_single_expert_tables(
    int* tables, int rows, int rows_pad,
    unsigned long long w0, unsigned long long w1, unsigned long long w2
) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i == 0) {
        unsigned long long* ptrs = reinterpret_cast<unsigned long long*>(tables);
        ptrs[0] = w0;
        ptrs[1] = w1;
        ptrs[2] = w2;
    }
    int* index = tables + 6;
    int* sorted = tables + 6 + rows_pad;
    int* tiles = tables + 6 + 2 * rows_pad;
    if (i < rows_pad) {
        index[i] = 0;
        sorted[i] = (i < rows) ? i : -1;
    }
    if (i < rows_pad / 16) tiles[i] = 0;
}
