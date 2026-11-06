// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// `gemm_q8_0_wmma` (gemm_q8_0_wmma.hip) with a BF16 epilogue: every output is rounded RN-even to
// BF16 before its store, so the result equals `gemm_q8_0_wmma` followed by the in-place
// `bf16_round_trip_f32_strided` pass (what Raon's BF16 GEMM-output boundary runs after each F16
// Q8_0 projection).
//
// The kernel body is the incumbent's, line for line: same geometry (grid [ceil(M/16), ceil(N/16)],
// block 32, LDS 0), same weight-row clamp, same FP16 X, same two WMMA tiles per Q8 block in the
// same K order from acc = 0, same RDNA3 wave32 output mapping and the same `out_row < M` /
// `out_col < N` store guards. Only the stored value differs: `q8g11_bf16_rn(acc[j])`. The helper
// is the one of `gemm_q8_0_mmq_i8.gfx11.hip` (`q8g11_bf16_rn`) and of `q8a8_bf16_rn` in
// `gemm_q8_0_mmq_i8.gfx12.hip`: bit-level RN-even, non-finite values pass through unchanged. It
// is NOT `__float2bfloat16` (a hardware cvt canonicalises NaN payloads).
//
// Y[N, M] = X[N, K] @ A_q8[M, K]^T, Y FP32 holding BF16-rounded values.
//   A: Q8_0 packed weights, 34 bytes per K=32 block, K a multiple of 32.
//   X: FP16 [N, K] row-major (caller stages F32->F16 via convert_f32_to_f16).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

static __device__ __forceinline__ float q8g11_bf16_rn(float v) {
    if (!isfinite(v)) return v;
    union { float f; unsigned int u; } box;
    box.f = v;
    const unsigned int lsb = (box.u >> 16) & 1u;
    box.u = (box.u + 0x7FFFu + lsb) & 0xFFFF0000u;
    return box.f;
}

// Same arch-split occupancy as the incumbent (see gemm_q8_0_wmma.hip).
#if defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
__launch_bounds__(32, 8)
#else
__launch_bounds__(32, 2)
#endif
extern "C" __global__ void gemm_q8_0_wmma_bf16(
    const char*     __restrict__ A_q8,    // [M, K/32 * 34] packed Q8_0
    const _Float16* __restrict__ X,       // [N, K] FP16
    float*          __restrict__ Y,       // [N, M] FP32 (BF16-rounded values)
    int M, int K, int N
) {
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid         = threadIdx.x;

    if (row_start >= M)   return;
    if (batch_start >= N) return;

    // Each thread owns one row in the 16-row tile (tid & 15).
    const int my_row   = row_start + (tid & 15);
    const int safe_row = (my_row < M) ? my_row : (M - 1);

    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)blocks_per_row * 34;
    const char* row_base = A_q8 + (long long)safe_row * row_stride;

    // Each thread also owns one batch col (tid & 15) for the X load.
    const int safe_batch = ((batch_start + (tid & 15)) < N)
                           ? (batch_start + (tid & 15)) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const char* bp = row_base + bi * 34;

        // fp16 scale (offset 0, 2 bytes — may be unaligned across blocks)
        _Float16 sc;
        {
            unsigned short s_bits;
            __builtin_memcpy(&s_bits, bp, 2);
            __builtin_memcpy(&sc, &s_bits, 2);
        }

        const _Float16* x_base = X + (long long)safe_batch * K + bi * 32;

        // === WMMA tile 0 (K-elements 0..16) ===
        const signed char* w0 = (const signed char*)(bp + 2);
        half16_t a0;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            a0[i] = sc * (_Float16)(float)(int)w0[i];
        }
        half16_t b0 = *(const half16_t*)(x_base);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc);

        // === WMMA tile 1 (K-elements 16..32) ===
        const signed char* w1 = (const signed char*)(bp + 2 + 16);
        half16_t a1;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            a1[i] = sc * (_Float16)(float)(int)w1[i];
        }
        half16_t b1 = *(const half16_t*)(x_base + 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc);
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // A operand = weight rows (m-dim), B operand = batch X cols (n-dim).
    const int out_col = batch_start + (tid & 15);
    if (out_col >= N) return;

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int out_row = row_start + 2 * j + (tid >> 4);
        if (out_row < M) {
            Y[(long long)out_col * M + out_row] = q8g11_bf16_rn(acc[j]);
        }
    }
}
