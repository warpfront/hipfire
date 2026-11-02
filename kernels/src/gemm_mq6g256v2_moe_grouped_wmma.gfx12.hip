// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) grouped-WMMA GEMM for MoE prefill over MQ6G256V2 (qt=47)
// routed experts. gfx12 sister of `gemm_mq6g256v2_moe_grouped_wmma_k2`.
//
// Structure clones `gemm_mq4g256v2_moe_grouped_wmma_gfx12` (dispatch,
// sentinel tile_ids, sorted_slot_index X-gather, x_row_div, wave32 WMMA
// operand shape, C-output mapping, K4 unroll). Only the weight layout
// axis changes: MQ6V2 dual-half header + 6-bit packed payload.
//
// V1/V2 incompatibility (do not collapse identities):
//   - MQ6/HFQ6 V1 (qt legacy 200 B/group): single f32 scale + f32 zero at
//     bytes [0..8), one affine grid for all 256 weights.
//   - MQ6G256V2 (qt=47, 200 B/group): dual f16 pairs
//       [0..2) f16 s0, [2..4) f16 z0, [4..6) f16 s1, [6..8) f16 z1
//     lane/weight half 0..127 uses pair 0; half 128..255 uses pair 1.
//     Payload is still 192 B of packed 6-bit at +8 — same packing as V1,
//     different header semantics. Wrong header decode is fluent corruption.
//
// Port contract vs MQ4V2 gfx12 sister:
//   - Group stride 200 B (not 136).
//   - Per-K-tile packed payload 12 B (16 × 6 bits), not 8 B nibbles;
//     per-k_grp half-tile is 6 B (8 × 6 bits).
//   - Dual-half header selection identical: upper = (kt >= 8); K4 bodies
//     {0-3},{4-7},{8-11},{12-15} each wholly inside one half.
//   - 6-bit unpack: two overlapping uint views (d0 at +0, d1 bits 0..23
//     from bytes +3..+5), shifts 0/6/12/18 × mask 63u — same as
//     `gemm_hfq6g256_moe_grouped_wmma_gfx12` / MQ6V2 dense WMMA.
//   - d1 is a 3-byte assemble (not *(uint*)(dp+3)) so the last tile of
//     the last group does not VM-fault past tightly packed expert ends.
//
// Affine-grid note under k_grp split: tile kt covers weights
// [kt*16, kt*16+16); grid boundary at weight 128 is the tile 7/8 seam —
// no K-tile straddles the two halves, and neither does either k_grp half.
//
// Degenerate groups: encoder may write a zero scale with constant payload;
// reconstruction remains scale*q + zero with no special case here.
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots × K], x_row_div = 1 → x_row = flat
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq6g256v2_moe_grouped_wmma_gfx12(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E]
    const int*                __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const _Float16*           __restrict__ X_src,              // [n_rows × K]
    float*                    __restrict__ Y_grouped,          // [m_total × M]
    int M, int K, int x_row_div, int m_total,
    int n_experts
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    // Sentinel early-return: the dispatcher may launch up to m_total_max/16
    // tiles to skip the m_total dtoh sync; scatter pre-fills the upper-bound
    // range of expert_tile_ids with -1.
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0 || expert_id >= n_experts) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    // X-row gather (per m_lane; both k_grp lanes for the same m_lane read the
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

    // A-row selection: pad OOB lanes to the last valid M-row (the masked
    // output write below discards the result anyway).
    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 200;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    const half8_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 200;

        // MQ6G256V2 dual affine grid: f16 s0,z0 / s1,z1 (not V1 f32 pair).
        const unsigned int hdr_lo = *(const unsigned int*)(gp);
        const unsigned int hdr_hi = *(const unsigned int*)(gp + 4);
        const _Float16 sc_0 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_lo & 0xFFFFu));
        const _Float16 zp_0 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_lo >> 16));
        const _Float16 sc_1 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_hi & 0xFFFFu));
        const _Float16 zp_1 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_hi >> 16));

        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        // 6-bit dequant: 8 values from 6 packed bytes via d0 (bytes 0..3)
        // and d1 (bytes 3..5, 24 useful bits). Shifts 0/6/12/18 × mask 63u.
        #define DQ6(i, dw, sh) a_reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
        #define DQ6_8(d0, d1) \
            DQ6(0, (d0), 0);  DQ6(1, (d0), 6);  DQ6(2, (d0), 12); DQ6(3, (d0), 18); \
            DQ6(4, (d1), 0);  DQ6(5, (d1), 6);  DQ6(6, (d1), 12); DQ6(7, (d1), 18)

        // K4 unroll: 4 K-tiles per inner body, 4 bodies = 16 K-tiles per group.
        for (int kt = 0; kt < 16; kt += 4) {
            // All four tiles of this body lie in the same affine grid — see
            // the grid-selection note above.
            const bool upper = (kt >= 8);
            const _Float16 sc_h = upper ? sc_1 : sc_0;
            const _Float16 zp_h = upper ? zp_1 : zp_0;

            // K-tile stride 12 B; per-k_grp half-tile 6 B.
            const int bb_a = (kt + 0) * 12 + k_grp * 6;
            const int bb_b = (kt + 1) * 12 + k_grp * 6;
            const int bb_c = (kt + 2) * 12 + k_grp * 6;
            const int bb_d = (kt + 3) * 12 + k_grp * 6;
            const unsigned char* dp_a = (const unsigned char*)(gp + 8 + bb_a);
            const unsigned char* dp_b = (const unsigned char*)(gp + 8 + bb_b);
            const unsigned char* dp_c = (const unsigned char*)(gp + 8 + bb_c);
            const unsigned char* dp_d = (const unsigned char*)(gp + 8 + bb_d);
            // d1 uses only bits 0..23 (3 bytes); a *(uint*)(dp+3) fetches a 4th
            // byte (dp+6) that is 1 B past the group on the last tile of the last
            // group → GPU VM read fault on tightly-packed expert buffers. Read 3 B.
            unsigned int d0a = *(const unsigned int*)(dp_a);
            unsigned int d1a = (unsigned int)dp_a[3] | ((unsigned int)dp_a[4] << 8) | ((unsigned int)dp_a[5] << 16);
            unsigned int d0b = *(const unsigned int*)(dp_b);
            unsigned int d1b = (unsigned int)dp_b[3] | ((unsigned int)dp_b[4] << 8) | ((unsigned int)dp_b[5] << 16);
            unsigned int d0c = *(const unsigned int*)(dp_c);
            unsigned int d1c = (unsigned int)dp_c[3] | ((unsigned int)dp_c[4] << 8) | ((unsigned int)dp_c[5] << 16);
            unsigned int d0d = *(const unsigned int*)(dp_d);
            unsigned int d1d = (unsigned int)dp_d[3] | ((unsigned int)dp_d[4] << 8) | ((unsigned int)dp_d[5] << 16);

            half8_t b_a, b_b, b_c, b_d;
            if (xg != nullptr) {
                b_a = *(const half8_t*)(xg + (kt + 0) * 16 + k_grp * 8);
                b_b = *(const half8_t*)(xg + (kt + 1) * 16 + k_grp * 8);
                b_c = *(const half8_t*)(xg + (kt + 2) * 16 + k_grp * 8);
                b_d = *(const half8_t*)(xg + (kt + 3) * 16 + k_grp * 8);
            } else {
                b_a = zero_b;
                b_b = zero_b;
                b_c = zero_b;
                b_d = zero_b;
            }

            half8_t a_reg;
            DQ6_8(d0a, d1a);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            DQ6_8(d0b, d1b);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            DQ6_8(d0c, d1c);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            DQ6_8(d0d, d1d);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
        }
        #undef DQ6_8
        #undef DQ6
    }

    // gfx12 wave32 WMMA C-mapping: acc[j] = C[8*k_grp + j][m_lane].
    // Direct write (=) — Y_grouped is fresh per call.
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
