// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// gfx12 (RDNA4) FP16 WMMA non-grouped sister of `gemm_hfq4g128_mmq.gfx1151.hip`.
// Replaces the scalar-FMA per-row baseline (`gemm_hfq4g128.hip`) for the
// **non-MoE PARO Q4G128 dispatch sites** on gfx12 (gfx1200/gfx1201).
//
// === Why this kernel exists (rocprof attribution 2026-05-22 on hiptrx/gfx1201) ===
//
// `gemm_hfq4g128` accounts for **69.9% of GPU time (470 ms / 673 ms)** on the
// shisa A3B-PARO prefill path at --prefill 256. This is the catch-all batched
// GEMM for ParoQ4G128 weights NOT covered by the routed-MoE grouped path:
//   - LA wqkv, LA wz (DeltaNetMoe layers)
//   - FullAttn wq, wk, wv, wo (FullAttnMoe layers)
//   - Shared-expert gate, up, down (every MoE layer)
//
// The gfx1151 i8 MMQ sister (`gemm_hfq4g128_mmq.gfx1151.hip`) closed this
// hole on Strix Halo with +129.5% A/B (427 → 980 tok/s pp256). gfx12 was
// never ported; the dispatcher's gfx12 path fell back to the scalar
// baseline and stalled at ~730 tok/s. This kernel closes the gap.
//
// === FP16 WMMA vs i8 MMQ on gfx12 ===
//
// The MoE grouped GEMM gfx12 ports today proved FP16 WMMA wins on gfx12
// (+30% kernel rate vs i8 MMQ on the G128 MoE path). MQ4's own dispatcher
// documents the same finding for G256 (i8 MMQ -11.6% vs FP16 WMMA on gfx12,
// dispatch.rs:12538-12550). This non-grouped kernel therefore uses FP16
// WMMA, mirroring `gemm_paro_q4g128_moe_grouped_wmma.gfx12.hip` minus the
// expert-routing scatter.
//
// === gfx12 conventions (vs gfx1151 sister) ===
//
//   1. WMMA builtin: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`
//      (with _gfx12 suffix; gfx11/gfx1151 uses the base _w32 builtin).
//   2. A/B operands: half8_t (8 fp16/lane) instead of int8x16_t (16 i8/lane).
//   3. K-dim split via k_grp = tid >> 4 (gfx12 wave32 layout).
//   4. C-output mapping: acc[j] = C[8*k_grp + j][m_lane] (vs gfx11's
//      acc[j] = C[2*j + (tid>>4)][lane]).
//   5. Activation contract: X is FP16 (ensure_fp16_x converts F32 once);
//      NOT Q8_1 pre-quantized (the FP16 WMMA path doesn't need that).
//
// === Per-block contract ===
//
//   - Grid: [M/16, N/16, 1]. One WMMA tile (16-row × 16-batch) per WG.
//   - Block: [32] (wave32).
//   - All 16 lanes in each k_grp share the m_lane → output row mapping.
//   - Output contract: y[batch_idx * M + row] = dot(A_row, x_batch[batch_idx])
//     — direct store. Matches the baseline `gemm_hfq4g128` contract.
//
// Shape requirements (caller-enforced; mirrors gfx1151 sister):
//   M % 16 == 0   — output rows are 16-tile aligned
//   N % 16 == 0   — batch rows are 16-tile aligned
//   K % 128 == 0  — HFQ4-G128 group size

#if defined(__gfx1200__) || defined(__gfx1201__)

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq4g128_wmma_gfx12(
    const char*       __restrict__ A,             // [M × (K/128) × 72] HFQ4-G128
    const _Float16*   __restrict__ X,             // [N × K] FP16 activations
    float*            __restrict__ Y,             // [N × M] direct store
    int M, int K, int N
) {
    const int tid = threadIdx.x;
    const int row_start  = blockIdx.x * 16;
    const int slot_start = blockIdx.y * 16;
    if (row_start >= M || slot_start >= N) return;

    const int m_lane = tid & 15;   // 0..15: output row within tile
    const int k_grp  = tid >> 4;   // 0/1: K-half this lane carries

    // X gather: each lane (m_lane) reads one batch row's worth of X.
    const int slot_idx = slot_start + m_lane;
    const _Float16* x_base = (slot_idx < N)
        ? (X + (long long)slot_idx * K) : (const _Float16*)nullptr;

    // A-row selection: pad OOB rows to last valid (masked-store at write).
    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int groups_per_row = K / 128;
    const char* row_base = A + (long long)safe_row * groups_per_row * 72;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    const half8_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 72;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 128) : nullptr;

        #define DQ8(pk) \
            a_reg[0] = sc_h * (_Float16)(float)(((pk) >>  0) & 0xFu) + zp_h; \
            a_reg[1] = sc_h * (_Float16)(float)(((pk) >>  4) & 0xFu) + zp_h; \
            a_reg[2] = sc_h * (_Float16)(float)(((pk) >>  8) & 0xFu) + zp_h; \
            a_reg[3] = sc_h * (_Float16)(float)(((pk) >> 12) & 0xFu) + zp_h; \
            a_reg[4] = sc_h * (_Float16)(float)(((pk) >> 16) & 0xFu) + zp_h; \
            a_reg[5] = sc_h * (_Float16)(float)(((pk) >> 20) & 0xFu) + zp_h; \
            a_reg[6] = sc_h * (_Float16)(float)(((pk) >> 24) & 0xFu) + zp_h; \
            a_reg[7] = sc_h * (_Float16)(float)(((pk) >> 28) & 0xFu) + zp_h

        // G128 has 8 K-tiles per group. Unroll 4 K-tiles per outer iter (matches
        // MoE grouped sister) → 2 iters/group. Each tile = 1 b load + 1 a dequant
        // + 1 WMMA. k_grp splits each 16-K tile across 2 lane groups.
        for (int kt = 0; kt < 8; kt += 4) {
            const int k_off_a = (kt + 0) * 16;
            const int k_off_b = (kt + 1) * 16;
            const int k_off_c = (kt + 2) * 16;
            const int k_off_d = (kt + 3) * 16;

            unsigned int pka = *(const unsigned int*)(gp + 8 + k_off_a / 2 + k_grp * 4);
            unsigned int pkb = *(const unsigned int*)(gp + 8 + k_off_b / 2 + k_grp * 4);
            unsigned int pkc = *(const unsigned int*)(gp + 8 + k_off_c / 2 + k_grp * 4);
            unsigned int pkd = *(const unsigned int*)(gp + 8 + k_off_d / 2 + k_grp * 4);

            half8_t b_a, b_b, b_c, b_d;
            if (xg != nullptr) {
                b_a = *(const half8_t*)(xg + k_off_a + k_grp * 8);
                b_b = *(const half8_t*)(xg + k_off_b + k_grp * 8);
                b_c = *(const half8_t*)(xg + k_off_c + k_grp * 8);
                b_d = *(const half8_t*)(xg + k_off_d + k_grp * 8);
            } else {
                b_a = zero_b; b_b = zero_b; b_c = zero_b; b_d = zero_b;
            }

            half8_t a_reg;
            DQ8(pka);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            DQ8(pkb);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            DQ8(pkc);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            DQ8(pkd);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
        }
        #undef DQ8
    }

    // gfx12 wave32 WMMA C-mapping: acc[j] = C[8*k_grp + j][m_lane].
    // Output Y is laid out [N × M] (mirrors baseline `gemm_hfq4g128` and
    // gfx1151 sister contract).
    const int out_col = slot_start + m_lane;
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 8 * k_grp + j;
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}

#else

// Stub for non-gfx12 archs (gfx1151 has its own i8 MMQ sister; gfx11 dGPU
// would need a separate port; CDNA archs unsupported for PARO).
extern "C" __global__ void gemm_hfq4g128_wmma_gfx12(
    const char*, const _Float16*, float*, int, int, int
) {}

#endif
