// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// gfx12/RDNA4 sister of gemm_f16_x_f16_wmma.hip.
//
// Same math, same (A, X, Y, M, K, B) signature, same [B, M] F32 output layout,
// same grid ([ceil(M/16), ceil(B/16)]) and block ([32]).
// Only the WMMA fragment mapping changes, mirroring the validated
// gemm_f16_wmma_mb8.gfx12.hip port:
// - half8_t operands and the `_w32_gfx12` builtin (gfx11 uses half16_t / `_w32`)
// - the K=16 contraction is split across the two lane-halves
//   (k_grp = tid >> 4 owns K elements [k_grp*8 .. +8))
// - the accumulator reads out contiguous-per-half,
//   acc[j] = C[8*k_grp + j][tid & 15],
//   instead of gfx11's interleaved C[2*j + (tid>>4)][tid & 15].
//
// The gfx11 `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` intrinsic needs
// `wmma-256b-insts,wavefrontsize32` and does not compile for gfx1201;
// this file uses `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` instead.
//
// Tail handling matches the gfx11 contract (clamped loads, masked stores)
// and additionally zero-fills a ragged K tail so any K is safe; on
// K % 16 == 0 the extra guard is dead code and results are unchanged.
//
// Layout (output is [B, M] like the gfx11 original):
//   A : [M, K] F16 row-major  (weight)
//   X : [B, K] F16 row-major  (caller F32→F16-converts upstream)
//   Y : [B, M] F32 row-major
//
// Compute: Y[b, m] = sum_k A[m, k] * X[b, k]
//
// Grid:  [ceil(M / 16), ceil(B / 16)]
// Block: [32]
// LDS:   0

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_f16_x_f16_wmma_gfx12(
    const _Float16* __restrict__ A,    // [M, K] F16
    const _Float16* __restrict__ X,    // [B, K] F16
    float*          __restrict__ Y,    // [B, M] F32
    int M, int K, int B
) {
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid    = threadIdx.x;
    const int lane_m = tid & 15;
    const int k_grp  = tid >> 4;

    if (row_start >= M || batch_start >= B) return;

    const int my_m = row_start + lane_m;
    const int my_b = batch_start + lane_m;
    const bool m_in_bounds = (my_m < M);
    const bool b_in_bounds = (my_b < B);

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // K loop: 16 elements per WMMA step, split 8/8 across the lane-halves.
    for (int k = 0; k < K; k += 16) {
        half8_t a_frag;
        if (m_in_bounds) {
            const _Float16* src = A + (long long)my_m * K + k;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int kk = k + k_grp * 8 + j;
                a_frag[j] = (kk < K) ? src[k_grp * 8 + j] : (_Float16)0.0f;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) a_frag[j] = (_Float16)0.0f;
        }
        half8_t x_frag;
        if (b_in_bounds) {
            const _Float16* src = X + (long long)my_b * K + k;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int kk = k + k_grp * 8 + j;
                x_frag[j] = (kk < K) ? src[k_grp * 8 + j] : (_Float16)0.0f;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) x_frag[j] = (_Float16)0.0f;
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, x_frag, acc);
    }

    // Write 8 outputs per thread following the gfx12 WMMA layout:
    // acc[j] at lane t is C[8*k_grp + j][lane_m].
    const int out_col = batch_start + lane_m;
    if (out_col < B) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 8 * k_grp + j;
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
