// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// G128-native m2 i8 MMQ MoE grouped-GEMM for HFQ4G128 (ParoQuant) on gfx12
// (R9700 / gfx1201, gfx1200).
//
// === Why G128-native instead of porting G256's k4 ===
//
// G128 has structurally less ILP runway than G256 within a single HFQ4 group:
//   - G128: 128 K-elements / group = 4 sub-blocks (Q8_1 blocks of 32 K) × 2
//     i8 WMMAs (16 K each) = 8 WMMAs per group per M-row
//   - G256: 256 K-elements / group = 8 sub-blocks × 2 = 16 WMMAs per group
//
// k4 (G256) amortizes the per-sub-block scale-FMA chain across 4 cacc
// accumulators interleaved over 4 sub-block pairs — needs 8 sub-blocks to
// fully extend the pipeline. On G128 it has only 4 sub-blocks, so the
// pipeline depth opportunity is half, and the k4 G128 sibling measures
// at -0.4 % (neutral) vs k2 baseline (702 tok/s prefill on shisa A3B-PARO
// @ --prefill 256 / gfx1201).
//
// The right G128 lever is M-direction reg-blocking (the m2 pattern). The
// B operand (X-gather row, the expensive side of grouped-GEMM) is loaded
// ONCE per K-substep and reused across two M-blocks; A doubles. Per
// K-substep: 2 packed4 dequants + 1 b load + 2 WMMA (was 1+1+1). The
// WMMA-per-X-byte ratio doubles → halves B-gather BW per output FLOP.
//
// Direct G256 analogue: `gemm_hfq4g256_moe_grouped_wmma_m2.gfx12.hip`
// (FP16 WMMA). Here we cross that pattern with the G128 i8 MMQ baseline.
//
// === Per-block contract ===
//
//   - Tile (blockIdx.x, blockIdx.y) owns a 32-row × 16-slot output tile.
//   - All 16 slots share the same expert (scatter pad alignment unchanged).
//   - Each lane m_lane = tid & 15 maps to ONE slot AND TWO M-rows
//     (row_start + m_lane for acc0, row_start + 16 + m_lane for acc1).
//   - X is gathered via sorted_slot_index[slot_start + m_lane]; -1
//     padding lanes substitute zero rows for the B operand.
//
// Grid: [ceil(M/32), m_total_max/16]. Block: [32]. LDS: 0.
//
// Compile gate: gfx1200 / gfx1201 only.

#if defined(__gfx1200__) || defined(__gfx1201__)

#define QK8_1 32

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};

static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

typedef int8_t  __attribute__((ext_vector_type(8)))  int8x8_t;
typedef int32_t __attribute__((ext_vector_type(2)))  int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8)))  int32x8_t;
typedef float   __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_paro_q4g128_moe_grouped_mmq_m2_gfx12(
    const unsigned long long*  __restrict__ expert_weight_ptrs, // [E]
    const int*                 __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                 __restrict__ sorted_slot_index,  // [m_total]
    const block_q8_1_mmq*      __restrict__ X_q8_src,           // [K/128 × x_src_rows]
    float*                     __restrict__ Y_grouped,          // [m_total × M]
    int M, int K, int x_row_div, int m_total, int x_src_rows
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 32;   // 2× wider M-stride than k2
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int m_lane = tid & 15;   // 0..15: slot index within tile
    const int k_grp  = tid >> 4;   // 0/1: which K-half this lane carries

    // B operand: single X-gather row per lane, shared across both M-blocks.
    const int slot_idx = slot_start + m_lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }

    // A operand: two M-block rows per lane.
    const int safe_row_0 = (row_start + m_lane < M)        ? (row_start + m_lane)        : (M - 1);
    const int safe_row_1 = (row_start + 16 + m_lane < M)   ? (row_start + 16 + m_lane)   : (M - 1);
    const int groups_per_row = K / 128;                              // G128
    const char* row_base_0 = A + (long long)safe_row_0 * groups_per_row * 72;
    const char* row_base_1 = A + (long long)safe_row_1 * groups_per_row * 72;

    float8_t acc0 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8_t acc1 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp0 = row_base_0 + g * 72;
        const char* gp1 = row_base_1 + g * 72;
        const float sc_self_0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        const float zp_self_0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        const float sc_self_1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        const float zp_self_1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));

        // Shuffle sc/zp to 8 destination output rows for BOTH M-blocks.
        // Same src_lane pattern as k2 since output mapping (k_grp lane split)
        // is unchanged; only the A-row binding doubles.
        float sc_row_0[8], zp_row_0[8], sc_row_1[8], zp_row_1[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int src_lane = 8 * k_grp + j;
            sc_row_0[j] = __shfl(sc_self_0, src_lane);
            zp_row_0[j] = __shfl(zp_self_0, src_lane);
            sc_row_1[j] = __shfl(sc_self_1, src_lane);
            zp_row_1[j] = __shfl(zp_self_1, src_lane);
        }

        // HFQ4G128: 1 Q8_1 mmq block per HFQ4 group (no kb_in_group loop).
        const int kb_q8 = g;
        const block_q8_1_mmq* xb = (x_row >= 0)
            ? (X_q8_src + (long long)kb_q8 * x_src_rows + x_row)
            : nullptr;

        // Preload all 8 (sb=0..3, kt=0..1) operands per group — mirror k2's
        // pattern to overlap A/B memory latency with WMMA execution.
        // m2 doubles A preload (pk0 + pk1) but shares the B preload (b_arr).
        unsigned int pk0_arr[8], pk1_arr[8];
        int8x8_t b_arr[8];
        float d_x_arr[4];
        float sum_x_arr[4];

        #pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            if (xb != nullptr) {
                half2 dm = xb->ds4[sb];
                float2 dm_f = __half22float2(dm);
                d_x_arr[sb]   = dm_f.x;
                sum_x_arr[sb] = dm_f.y;
            } else {
                d_x_arr[sb]   = 0.0f;
                sum_x_arr[sb] = 0.0f;
            }
            #pragma unroll
            for (int kt = 0; kt < 2; kt++) {
                const int slot = sb * 2 + kt;
                // K-offset within HFQ4G128 group:
                //   sb * 32  (sub-block within group)
                // + kt * 16  (WMMA call within sub-block)
                // + k_grp * 8 (lane-group's K-half).
                const int k_off = sb * 32 + kt * 16 + k_grp * 8;
                pk0_arr[slot] = *(const unsigned int*)(gp0 + 8 + k_off / 2);
                pk1_arr[slot] = *(const unsigned int*)(gp1 + 8 + k_off / 2);
                if (xb != nullptr) {
                    const int8_t* qs_p = xb->qs + sb * 32 + kt * 16 + k_grp * 8;
                    b_arr[slot] = *(const int8x8_t*)qs_p;
                } else {
                    b_arr[slot] = (int8x8_t){0, 0, 0, 0, 0, 0, 0, 0};
                }
            }
        }

        // 4 sub-blocks × 2 WMMA-kt. Compute body — B is loaded once per slot,
        // reused across both M-blocks (the m2 BW win).
        #pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            int32x8_t cacc0 = {0, 0, 0, 0, 0, 0, 0, 0};
            int32x8_t cacc1 = {0, 0, 0, 0, 0, 0, 0, 0};

            #pragma unroll
            for (int kt = 0; kt < 2; kt++) {
                const int slot = sb * 2 + kt;
                const unsigned int pk0 = pk0_arr[slot];
                const unsigned int pk1 = pk1_arr[slot];

                // Dequant A nibbles → int8 for both M-blocks.
                int8x8_t a_vec_0, a_vec_1;
                a_vec_0[0] = (int8_t)((pk0 >>  0) & 0xFu);
                a_vec_0[1] = (int8_t)((pk0 >>  4) & 0xFu);
                a_vec_0[2] = (int8_t)((pk0 >>  8) & 0xFu);
                a_vec_0[3] = (int8_t)((pk0 >> 12) & 0xFu);
                a_vec_0[4] = (int8_t)((pk0 >> 16) & 0xFu);
                a_vec_0[5] = (int8_t)((pk0 >> 20) & 0xFu);
                a_vec_0[6] = (int8_t)((pk0 >> 24) & 0xFu);
                a_vec_0[7] = (int8_t)((pk0 >> 28) & 0xFu);

                a_vec_1[0] = (int8_t)((pk1 >>  0) & 0xFu);
                a_vec_1[1] = (int8_t)((pk1 >>  4) & 0xFu);
                a_vec_1[2] = (int8_t)((pk1 >>  8) & 0xFu);
                a_vec_1[3] = (int8_t)((pk1 >> 12) & 0xFu);
                a_vec_1[4] = (int8_t)((pk1 >> 16) & 0xFu);
                a_vec_1[5] = (int8_t)((pk1 >> 20) & 0xFu);
                a_vec_1[6] = (int8_t)((pk1 >> 24) & 0xFu);
                a_vec_1[7] = (int8_t)((pk1 >> 28) & 0xFu);

                int32x2_t a_i32_0 = *(const int32x2_t*)&a_vec_0;
                int32x2_t a_i32_1 = *(const int32x2_t*)&a_vec_1;
                int32x2_t b_i32   = *(const int32x2_t*)&b_arr[slot];

                // signed_a = false (HFQ4 nibbles 0..15, unsigned-safe);
                // signed_b = true (Q8_1 signed -127..127). clamp = true.
                cacc0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                    false, a_i32_0, true, b_i32, cacc0, true
                );
                cacc1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                    false, a_i32_1, true, b_i32, cacc1, true
                );
            } // for kt

            const float d_x_self   = d_x_arr[sb];
            const float sum_x_self = sum_x_arr[sb];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float dot_f_0 = (float)cacc0[j];
                float dot_f_1 = (float)cacc1[j];
                acc0[j] += sc_row_0[j] * d_x_self * dot_f_0 + zp_row_0[j] * sum_x_self;
                acc1[j] += sc_row_1[j] * d_x_self * dot_f_1 + zp_row_1[j] * sum_x_self;
            }
        } // for sb
    } // for g

    // gfx12 wave32 WMMA C-mapping: acc[j] = C[8*k_grp + j][m_lane].
    const int out_col = slot_start + m_lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row_0 = row_start + 8 * k_grp + j;
            if (out_row_0 < M) {
                Y_grouped[(long long)out_col * M + out_row_0] = acc0[j];
            }
            const int out_row_1 = row_start + 16 + 8 * k_grp + j;
            if (out_row_1 < M) {
                Y_grouped[(long long)out_col * M + out_row_1] = acc1[j];
            }
        }
    }
}

#else

// Stub for non-gfx12 archs.
extern "C" __global__ void gemm_paro_q4g128_moe_grouped_mmq_m2_gfx12(
    const unsigned long long*, const int*, const int*,
    const void*, float*, int, int, int, int, int
) {}

#endif
