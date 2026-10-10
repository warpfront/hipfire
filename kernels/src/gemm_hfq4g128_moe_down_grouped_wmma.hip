// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//
// Grouped HFQ4-G128 MoE down projection on wave32 WMMA (gfx11). Same
// contract as `gemv_hfq4g128_moe_down_grouped`: 16-slot tiles sorted by
// expert (`moe_scatter_fused_k8` layout), `y[grouped_row, :] =
// per_expert_scale[e] * W_e · act[sorted_slots[grouped_row]]`, padding rows
// unwritten. Adapted from `gemm_paro_q4g128_moe_grouped_wmma_k2` for rows
// whose K is not a multiple of 128: a row holds ceil(K/128) 72-byte groups
// and the last group's tail beyond K is skipped (K % 16 == 0). Weights are
// dequantized in f32 and rounded once to f16; X is F16.
//
// Grid: [ceil(M/16), m_total/16]. Block: [32].
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>


typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq4g128_moe_down_grouped_wmma(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ tile_ids,
    const int* __restrict__ sorted_slots,
    const float* __restrict__ per_expert_scale,
    const _Float16* __restrict__ act,    // [slots, K]
    float* __restrict__ y,               // [m_total, M]
    int M, int K
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile = blockIdx.y;
    if (row_start >= M) return;
    const int expert = tile_ids[tile];
    if (expert < 0) return;
    const char* __restrict__ A = reinterpret_cast<const char*>(expert_ptrs[expert]);

    const int lane = tid & 15;
    const int slot = sorted_slots[tile * 16 + lane];
    const _Float16* x_base = slot >= 0 ? act + (long long)slot * K : nullptr;
    const int safe_row = row_start + lane < M ? row_start + lane : M - 1;
    const int groups_per_row = (K + 127) / 128;
    const char* row_base = A + (long long)safe_row * groups_per_row * 72;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 72;
        const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const int kt_end = min(8, (K - g * 128) / 16);
        for (int kt = 0; kt < kt_end; ++kt) {
            const int k_off = kt * 16;
            const unsigned int pk0 = *(const unsigned int*)(gp + 8 + k_off / 2);
            const unsigned int pk1 = *(const unsigned int*)(gp + 8 + k_off / 2 + 4);
            const half16_t b = x_base != nullptr
                ? *(const half16_t*)(x_base + g * 128 + k_off) : zero_b;
            half16_t a;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                a[i] = (_Float16)__builtin_fmaf(sc, (float)((pk0 >> (4 * i)) & 0xFu), zp);
                a[8 + i] = (_Float16)__builtin_fmaf(sc, (float)((pk1 >> (4 * i)) & 0xFu), zp);
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
        }
    }

    if (slot >= 0) {
        const float es = per_expert_scale[expert];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) y[(long long)(tile * 16 + lane) * M + out_row] = es * acc[j];
        }
    }
}
