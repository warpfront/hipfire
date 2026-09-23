// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) sister of `gemm_paro_q4g128_moe_grouped_wmma_k2` —
// the cross-arch PARO G128 FP16 WMMA kernel. Crosses that kernel's G128
// layout (72 B/group, 128 K-elements/group, 8 WMMA K-tiles per group) with
// the gfx12 WMMA conventions from `gemm_hfq4g256_moe_grouped_wmma.gfx12.hip`:
//
//   1. WMMA builtin: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
//      (the cross-arch k2 sibling uses _w32 without the _gfx12 suffix, which
//      does NOT compile on gfx12 — the entire reason this port exists).
//   2. A/B operands: half8_t (8 fp16/lane) instead of half16_t.
//   3. K dim split across 2 lane-groups (k_grp = tid >> 4); each carries
//      a half of the 16-K tile.
//   4. C-output mapping: acc[j] = C[8*k_grp + j][m_lane].
//
// === Why this kernel exists (the structural lever) ===
//
// On gfx12, the analogous lever for MQ4 G256 grouped GEMM is FP16 WMMA
// (gemm_hfq4g256_moe_grouped_wmma_gfx12), not i8 MMQ. The MQ4 i8 MMQ
// variant on gfx12 was tested and measured **-11.6 % regression** vs the
// FP16 WMMA path (per the comment in dispatch.rs:12538-12550). PARO G128's
// initial gfx12 port (`gemm_paro_q4g128_moe_grouped_mmq.gfx12.hip`,
// commit f2e2c254) took the i8 MMQ route instead — the exact path MQ4
// explicitly abandoned. This ports the correct FP16 WMMA lever.
//
// Expected lift: closes (or shrinks) the prefill gap between PARO G128
// (currently 705 tok/s on shisa A3B-PARO @ --prefill 256) and MQ4 G256
// (~2966 tok/s on same hardware) — both via the same FP16 WMMA path
// modulo the G128/G256 layout difference (2× group iterations on G128
// since groups_per_row doubles).
//
// === Per-block contract (unchanged from G256 gfx12 sister) ===
//
//   - Tile (blockIdx.x, blockIdx.y) owns a 16-row × 16-slot output tile.
//   - All 16 slots share the same expert (scatter pad alignment unchanged).
//   - Each lane m_lane = tid & 15 corresponds to one slot AND one M-row.
//   - X is gathered via sorted_slot_index[slot_start + m_lane]; -1
//     padding lanes substitute zero rows for the B operand.
//   - X_src is FP16 (caller ensure_fp16_x converts from F32 once).
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_paro_q4g128_moe_grouped_wmma_gfx12(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E]
    const int*                __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const _Float16*           __restrict__ X_src,              // [n_rows × K]
    float*                    __restrict__ Y_grouped,          // [m_total × M]
    int M, int K, int x_row_div, int m_total
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    // X gather (per m_lane; both k_grp lanes for the same m_lane read the
    // same source row). -1 padding lanes contribute a zero B row.
    const int slot_idx = slot_start + m_lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base = (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    // A-row selection: pad OOB lanes to the last valid M-row.
    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int groups_per_row = K / 128;                                  // G128
    const char* row_base = A + (long long)safe_row * groups_per_row * 72; // 72 B/group

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

        // G128 has 8 K-tiles per group (vs G256's 16). Unroll 4 K-tiles per
        // inner iter (matches G256 gfx12 sister's pattern) → 2 iters/group.
        // Each K-tile: pk32 → 8 nibbles → 8 fp16 dequant → WMMA against b.
        // k_grp splits each K-tile across 2 lane groups: each lane carries
        // K[k_grp*8 .. k_grp*8 + 7] (the gfx12 wave32 16x16x16 layout).
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
    const int out_col = slot_start + m_lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 8 * k_grp + j;
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
