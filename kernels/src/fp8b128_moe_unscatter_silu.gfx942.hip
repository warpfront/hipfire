// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// FP8B128 grouped MoE gate/up unscatter fused with the teacher's BF16 SwiGLU
// (gfx942).
//
// Numerical contract: the det teacher, fastfwd.py:476-479, runs the grouped
// gate/up GEMM in BF16 (`gu`), then `F.silu(gate)` in BF16, then `* up` in
// BF16, each op rounding its result to BF16:
//
//   g = bf16_rne(gate_acc)            // F32 accumulator -> BF16 gate
//   u = bf16_rne(up_acc)              // F32 accumulator -> BF16 up
//   s = bf16_rne(g / (1 + expf(-g)))  // BF16 silu(gate)
//   h = bf16_rne(s * u)               // BF16 product
//
// `h` is stored as F32 holding the BF16 value (low 16 mantissa bits zero).
// expf is the precise libm expf: no __expf, no fast-math, no reassociation.
// (The pre-existing moe_gate_up_unscatter_silu_top10.hip rounds only the
// final product, not silu(gate); this kernel is a new, separate FP8 path and
// leaves that kernel untouched.)
//
//   Y_grouped:         F32 [n_sorted, 2*mi]; row = grouped slot, columns
//                      [0, mi) gate, [mi, 2*mi) up
//   sorted_slot_index: i32 [n_sorted]; flat (token*top_k + k_rank) row of each
//                      grouped slot, -1 = padding (skipped)
//   activation:        F32 [slots, mi]; row `flat` written once
//
// Every live flat slot appears at most once in sorted_slot_index, so each
// activation element has exactly one writer: plain stores, no atomics. Rows
// with flat < 0 or flat >= slots are skipped (flat slot bound).
//
// Geometry: grid [n_sorted, ceil(mi / 256)], block [256]; one thread per
// output element.

#include <hip/hip_runtime.h>

// bf16 round-to-nearest-even of an F32 (the bf16_round_trip.hip rounding:
// non-finite values pass through unchanged).
static __device__ __forceinline__ float fp8act_bf16_rne(float v) {
    if (!isfinite(v)) return v;
    const unsigned int u = __float_as_uint(v);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

extern "C" __global__ __launch_bounds__(256) void fp8b128_moe_unscatter_silu_gfx942(
    const float* __restrict__ Y_grouped,
    const int* __restrict__ sorted_slot_index,
    float* __restrict__ activation,
    int mi,
    int n_sorted,
    int slots
) {
    const int row = blockIdx.x;
    if (row >= n_sorted) return;
    const int flat = sorted_slot_index[row];
    if (flat < 0 || flat >= slots) return;
    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= mi) return;

    const long long y_base = (long long)row * 2 * mi;
    const float g = fp8act_bf16_rne(Y_grouped[y_base + col]);
    const float u = fp8act_bf16_rne(Y_grouped[y_base + mi + col]);
    const float s = fp8act_bf16_rne(g / (1.0f + expf(-g)));
    activation[(long long)flat * mi + col] = fp8act_bf16_rne(s * u);
}
