// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) FP16 WMMA replacement for `gemm_f32_batched.hip`.
// Same call contract: Y[M × N] = A[M × K] @ B[N × K]^T (but see swap-trick
// usage at qwen35.rs:5380 — callers can swap A↔B to flip output to [N × M]).
//
// All operands are F32 in/out. Inside the kernel we downcast A and B to F16
// per K-tile for the WMMA call (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`).
// For z-lab/Qwen3.6-35B-A3B-PARO's shared_expert weights, this is LOSSLESS to
// source storage because the upstream loader (`load_fp16_weight_from_source`)
// expands raw F16 safetensors data into F32 GPU buffers — the F16 precision is
// what was on disk; we're just recovering it for WMMA throughput.
//
// === Why this kernel matters ===
//
// rocprof attribution on z-lab post-admit-fix shows `gemm_f32_batched` at
// **58.7% of GPU time** (176 ms / 300 ms, 520 calls). The kernel is
// scalar F32 FMA: 1 (m, n) tile per WG, 32 threads doing 1 reduction each.
// Replacing with FP16 WMMA (32 MACs/cycle on gfx12 vs ~1 for scalar FMA)
// is structurally analogous to the HFQ4G128 scalar → WMMA port that
// landed +250% prefill at 6c097ad5.
//
// === Layout ===
//
// Grid:    [ceil(M/16), ceil(N/16), 1]
// Block:   [32]
// Output:  Y[m_tile * 16 + j][n_tile * 16 + m_lane] = sum_k A_m_row[k] * B_n_row[k]
//
// gfx12 wave32 WMMA C-mapping: acc[j] = C[8*k_grp + j][m_lane], so j ∈ [0,8)
// maps to output rows {8*k_grp .. 8*k_grp+7} within the 16-row M tile, and
// m_lane ∈ [0,16) maps to output cols within the 16-col N tile.
//
// Shape requirements (caller-enforced):
//   M % 16 == 0
//   N % 16 == 0
//   K % 16 == 0   (need at least one full 16-K WMMA call; pad if needed)

#if defined(__gfx1200__) || defined(__gfx1201__)

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 4)
extern "C" __global__ void gemm_f32_wmma_gfx12(
    const float* __restrict__ A,   // [M × K] F32
    const float* __restrict__ B,   // [N × K] F32
    float*       __restrict__ Y,   // [M × N] F32 output, written direct
    int M, int K, int N
) {
    const int tid = threadIdx.x;
    const int m_tile = blockIdx.x;
    const int n_tile = blockIdx.y;
    if (m_tile * 16 >= M || n_tile * 16 >= N) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    // Per-lane row pointers. Each lane maps to:
    //   - A row m_tile*16 + m_lane  (lane carries this row's K data)
    //   - B row n_tile*16 + m_lane  (lane carries this col's K data)
    const int m_row = m_tile * 16 + m_lane;
    const int n_row = n_tile * 16 + m_lane;
    const int safe_m = (m_row < M) ? m_row : (M - 1);
    const int safe_n = (n_row < N) ? n_row : (N - 1);
    const float* a_row = A + (long long)safe_m * K;
    const float* b_row = B + (long long)safe_n * K;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // WMMA over K in 16-element tiles. k_grp splits each 16-tile across two
    // lane groups; each lane carries 8 K elements (half of the 16-tile).
    for (int kt = 0; kt < K; kt += 16) {
        const int k_off = kt + k_grp * 8;

        // Downcast 8 F32 → 8 F16 inline. Compiler vectorizes the convert.
        half8_t a_h, b_h;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            a_h[i] = (_Float16)a_row[k_off + i];
            b_h[i] = (_Float16)b_row[k_off + i];
        }

        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_h, b_h, acc);
    }

    // gfx12 WMMA C-layout: acc[j] = C[8*k_grp + j][m_lane].
    // Y is [M × N] row-major (matches gemm_f32_batched output convention),
    // so Y[m_global * N + n_global] = acc[j] for m_global = m_tile*16 + 8*k_grp + j,
    // n_global = n_tile*16 + m_lane.
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

extern "C" __global__ void gemm_f32_wmma_gfx12(
    const float*, const float*, float*, int, int, int
) {}

#endif
