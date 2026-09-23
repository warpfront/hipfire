// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

// gfx12 (RDNA4) BF16 WMMA replacement for `gemm_f32_batched.hip`. Same call
// contract: A/B/Y all F32. Inside the kernel we downcast A and B to BF16
// per K-tile (bit-truncation: BF16 is the top 16 bits of F32) for the WMMA
// call (`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12`).
//
// === Why BF16 instead of FP16 ===
//
// FP16 WMMA broad-opt-in (gemm_f32_wmma.gfx12.hip via HIPFIRE_GEMM_F32_WMMA_GFX12)
// measured +138% perf but COHERENCE REGRESSED on humaneval_2/3. Root cause:
// the router (m=num_experts, k=dim) is precision-sensitive in its DYNAMIC
// RANGE — logits can be outside FP16's ±65k window, clamping to inf and
// breaking top-K routing.
//
// BF16 has the SAME 8-bit exponent as F32 (only mantissa is reduced, 7 bits
// vs 23). Router logits stay representable; precision of relative magnitudes
// is preserved enough for top-K decisions.
//
// Implementation: the WMMA intrinsic takes its bf16 operands as `short8_t`
// (the raw bit pattern). We do F32 → BF16 truncation inline via uint shift:
//   bf16_bits = (uint16_t)(__builtin_bit_cast(uint32_t, f32) >> 16)
// This is the cheapest BF16 conversion — equivalent to round-toward-zero.
// Round-to-nearest-even would add a small bias but our top-K targets are
// not bit-exact sensitive (the bench gate is fluency, not byte-equality).
//
// === Layout (unchanged from FP16 sister) ===
//
// Grid:    [ceil(M/16), ceil(N/16), 1]
// Block:   [32]
// Output:  Y[m_tile * 16 + j][n_tile * 16 + m_lane]
// C-map:   acc[j] = C[8*k_grp + j][m_lane]   (gfx12 wave32 convention)

#if defined(__gfx1200__) || defined(__gfx1201__)

typedef int16_t __attribute__((ext_vector_type(8))) short8_t;
typedef float   __attribute__((ext_vector_type(8))) float8_t;

static __device__ __forceinline__ int16_t f32_to_bf16_truncate(float f) {
    uint32_t bits = __builtin_bit_cast(uint32_t, f);
    return (int16_t)((bits >> 16) & 0xFFFFu);
}

__launch_bounds__(32, 4)
extern "C" __global__ void gemm_f32_wmma_bf16_gfx12(
    const float* __restrict__ A,   // [M × K] F32
    const float* __restrict__ B,   // [N × K] F32
    float*       __restrict__ Y,   // [M × N] F32 output
    int M, int K, int N
) {
    const int tid = threadIdx.x;
    const int m_tile = blockIdx.x;
    const int n_tile = blockIdx.y;
    if (m_tile * 16 >= M || n_tile * 16 >= N) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int m_row = m_tile * 16 + m_lane;
    const int n_row = n_tile * 16 + m_lane;
    const int safe_m = (m_row < M) ? m_row : (M - 1);
    const int safe_n = (n_row < N) ? n_row : (N - 1);
    const float* a_row = A + (long long)safe_m * K;
    const float* b_row = B + (long long)safe_n * K;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int kt = 0; kt < K; kt += 16) {
        const int k_off = kt + k_grp * 8;

        short8_t a_b, b_b;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            a_b[i] = f32_to_bf16_truncate(a_row[k_off + i]);
            b_b[i] = f32_to_bf16_truncate(b_row[k_off + i]);
        }

        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a_b, b_b, acc);
    }

    const int out_n = n_tile * 16 + m_lane;
    if (out_n < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_m = m_tile * 16 + 8 * k_grp + j;
            if (out_m < M) {
                Y[(long long)out_m * N + out_n] = acc[j];
            }
        }
    }
}

#else

extern "C" __global__ void gemm_f32_wmma_bf16_gfx12(
    const float*, const float*, float*, int, int, int
) {}

#endif
