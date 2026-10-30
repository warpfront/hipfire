// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Generic FFN gate/up activation over dense rows:
//   out[row, col] = awq_divide( pad( cast_out( act( cast_in(g), cast_in(u) ) ) ) )
// with the activation selected at compile time per entry point:
//   ffn_swiglu_rows_f32        silu(g) * u
//   ffn_swiglu_clamp_rows_f32  g = min(g, p0); u = clamp(u, -p0, p0); silu(g) * u
//   ffn_situ_rows_f32          ((p0*tanh(g/p0)) * sigmoid(g)) * (p1*tanh(u/p1))
// Plain IEEE-style float math (expf / tanhf); no fast-math, no fma contraction
// across the activation's own multiplies.
//
// Columns [logical_n, phys_n) are written as +0.0 after the activation and the
// optional BF16 output rounding, and BEFORE the AWQ divide.
//
// Grid  (rows, ceil(phys_n / 256))   rows on x (limit 2^31), column tile on y
// Block 256
// LDS   0
//
// Kernarg order:
//   gate*, up*, out*, awq_a*, row_expert*, rows, logical_n, phys_n, gate_ld,
//   up_ld, awq_mode, n_experts, round_inputs, round_output, p0, p1
// gate / up already include the caller's element offset.  awq_a is the F32
// vector (awq_mode 1) or the u64 pointer table (awq_mode 2); row_expert is the
// i32 expert of each row (awq_mode 2 only).

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
static __device__ __forceinline__ void ffn_rows_body(
    const float* gate, const float* up, float* out,
    const void* awq_a, const int* row_expert,
    int rows, int logical_n, int phys_n, int gate_ld, int up_ld,
    int awq_mode, int n_experts, int round_inputs, int round_output,
    float p0, float p1
) {
    const long long row = blockIdx.x;
    const int col = blockIdx.y * 256 + threadIdx.x;
    if (row >= rows || col >= phys_n) return;

    float r = 0.0f;
    if (col < logical_n) {
        float g = gate[row * (long long)gate_ld + col];
        float u = up[row * (long long)up_ld + col];
        if (round_inputs) {
            g = ffn_bf16_rne(g);
            u = ffn_bf16_rne(u);
        }
        r = ffn_act<ACT>(g, u, p0, p1);
        if (round_output) r = ffn_bf16_rne(r);
    }

    if (awq_mode == 1) {
        r = r / static_cast<const float*>(awq_a)[col];
    } else if (awq_mode == 2) {
        const int e = row_expert[row];
        const unsigned long long* table = static_cast<const unsigned long long*>(awq_a);
        const float* scale = (e >= 0 && e < n_experts)
            ? reinterpret_cast<const float*>(table[e]) : nullptr;
        r = (scale != nullptr) ? (r / scale[col]) : 0.0f;
    }
    out[row * (long long)phys_n + col] = r;
}

#define FFN_ROWS_ENTRY(NAME, ACT)                                                        \
extern "C" __global__ __launch_bounds__(256) void NAME(                                  \
    const float* gate, const float* up, float* out,                                      \
    const void* awq_a, const int* row_expert,                                            \
    int rows, int logical_n, int phys_n, int gate_ld, int up_ld,                         \
    int awq_mode, int n_experts, int round_inputs, int round_output,                     \
    float p0, float p1                                                                   \
) {                                                                                      \
    ffn_rows_body<ACT>(gate, up, out, awq_a, row_expert, rows, logical_n, phys_n,        \
                       gate_ld, up_ld, awq_mode, n_experts, round_inputs,                \
                       round_output, p0, p1);                                            \
}

FFN_ROWS_ENTRY(ffn_swiglu_rows_f32, 0)
FFN_ROWS_ENTRY(ffn_swiglu_clamp_rows_f32, 1)
FFN_ROWS_ENTRY(ffn_situ_rows_f32, 2)
