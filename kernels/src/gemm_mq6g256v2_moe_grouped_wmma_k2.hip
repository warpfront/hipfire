// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Grouped-WMMA-GEMM for MoE prefill over MQ6G256V2 (qt=47) routed experts.
// gfx11 (RDNA3) wave32 sister — structure from gemm_mq4g256v2_moe_grouped_wmma_k2,
// dual-header + 6-bit payload decode from gemm_mq6g256v2_residual_wmma /
// gemm_hfq6g256_moe_grouped_wmma_k2.
//
// V1 vs V2 are wire-incompatible; do not collapse headers or dispatch paths:
//
//   qt=6 / HFQ6G256 / MQ6G256 V1 — 200 B/group:
//     [0..4)  f32 scale (all 256 weights)
//     [4..8)  f32 zero  (all 256 weights)
//     [8..200) 192 packed 6-bit bytes
//
//   qt=47 MQ6G256V2 — 200 B/group (same stride, different header):
//     [0..2)  f16 scale, weights   0..127
//     [2..4)  f16 zero,  weights   0..127
//     [4..6)  f16 scale, weights 128..255
//     [6..8)  f16 zero,  weights 128..255
//     [8..200) 192 packed 6-bit bytes   (identical packing to V1)
//
// 6-bit packing: 4 weights per 3 bytes. Per K-tile (16 values) = 12 B.
// Overlapping uint reads at +0/+3/+6/+9; extract ((dw >> sh) & 63u) at
// shifts 0/6/12/18. The last 3 B of each tile are assembled from bytes
// (not a 4-byte load) so the final tile of a tightly packed group never
// reads one byte past the buffer.
//
// K-tile → grid mapping: tile kt covers weights [kt*16, kt*16+16). Tiles 0-7
// are weights 0..127 (grid 0); tiles 8-15 are weights 128..255 (grid 1). The
// loop consumes tiles in pairs (kt, kt+1) with kt even, and every such pair
// lies wholly inside one grid — (6,7) → weights 96..127, (8,9) → 128..159.
// So the grid is selected once per pair, not per tile.
//
// Degenerate groups: the encoder writes scale bits 0 and all codes 0 when
// hi == lo, giving 0 * q + zero == zero. That is the intended constant-group
// value and needs no special case here.
//
// Per-block contract (clone of MQ4V2 grouped):
//   - Tile (blockIdx.x, blockIdx.y) owns a 16-row × 16-slot output tile;
//     all 16 slots share one expert (scatter pads expert_offsets to 16).
//   - expert_tile_ids[tile_y] == -1 → early-return (sync-free dispatch).
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.
// Activations must already be FWHT-G256 rotated (F32/FP16 tensor as supplied).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots × K], x_row_div = 1 → x_row = flat
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq6g256v2_moe_grouped_wmma_k2(
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

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0 || expert_id >= n_experts) return;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int lane = tid & 15;
    const int slot_idx = slot_start + lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base = (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 200;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 200;

        // Dual affine grid. Stored as raw f16 bits (V2), unlike V1's f32 pair.
        const unsigned int hdr_lo = *(const unsigned int*)(gp);
        const unsigned int hdr_hi = *(const unsigned int*)(gp + 4);
        const _Float16 sc_0 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_lo & 0xFFFFu));
        const _Float16 zp_0 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_lo >> 16));
        const _Float16 sc_1 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_hi & 0xFFFFu));
        const _Float16 zp_1 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_hi >> 16));

        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        // 8 iterations × 2 K-tiles each = 16 K-tiles per group.
        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;

            // Both tiles of this pair lie in the same affine grid — see the
            // K-tile mapping note above.
            const bool upper = (k_off_a >= 128);
            const _Float16 sc_h = upper ? sc_1 : sc_0;
            const _Float16 zp_h = upper ? zp_1 : zp_0;

            // MQ6V2 weight: 12 B per 16-value K-tile, read as four
            // overlapping uints (+0/+3/+6/+9) for clean 6-bit factoring.
            const unsigned char* wa = (const unsigned char*)(gp + 8 + kt * 12);
            const unsigned char* wb = (const unsigned char*)(gp + 8 + (kt + 1) * 12);
            // a3/c3 use only bits 0..23 (3 bytes); a *(uint*)(w+9) fetches a 4th
            // byte (w+12) that is 1 B past the group on the last tile of the last
            // group → GPU VM read fault on tightly-packed expert buffers. Read 3 B.
            unsigned int a0 = *(const unsigned int*)(wa);
            unsigned int a1 = *(const unsigned int*)(wa + 3);
            unsigned int a2 = *(const unsigned int*)(wa + 6);
            unsigned int a3 = (unsigned int)wa[9] | ((unsigned int)wa[10] << 8) | ((unsigned int)wa[11] << 16);
            unsigned int c0 = *(const unsigned int*)(wb);
            unsigned int c1 = *(const unsigned int*)(wb + 3);
            unsigned int c2 = *(const unsigned int*)(wb + 6);
            unsigned int c3 = (unsigned int)wb[9] | ((unsigned int)wb[10] << 8) | ((unsigned int)wb[11] << 16);

            half16_t b_a, b_b;
            if (xg != nullptr) {
                b_a = *(const half16_t*)(xg + k_off_a);
                b_b = *(const half16_t*)(xg + k_off_b);
            } else {
                b_a = zero_b;
                b_b = zero_b;
            }

            half16_t a_a;
            #define DQ6(reg, i, dw, sh) reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
            DQ6(a_a, 0,  a0, 0); DQ6(a_a, 1,  a0, 6); DQ6(a_a, 2,  a0, 12); DQ6(a_a, 3,  a0, 18);
            DQ6(a_a, 4,  a1, 0); DQ6(a_a, 5,  a1, 6); DQ6(a_a, 6,  a1, 12); DQ6(a_a, 7,  a1, 18);
            DQ6(a_a, 8,  a2, 0); DQ6(a_a, 9,  a2, 6); DQ6(a_a, 10, a2, 12); DQ6(a_a, 11, a2, 18);
            DQ6(a_a, 12, a3, 0); DQ6(a_a, 13, a3, 6); DQ6(a_a, 14, a3, 12); DQ6(a_a, 15, a3, 18);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

            DQ6(a_a, 0,  c0, 0); DQ6(a_a, 1,  c0, 6); DQ6(a_a, 2,  c0, 12); DQ6(a_a, 3,  c0, 18);
            DQ6(a_a, 4,  c1, 0); DQ6(a_a, 5,  c1, 6); DQ6(a_a, 6,  c1, 12); DQ6(a_a, 7,  c1, 18);
            DQ6(a_a, 8,  c2, 0); DQ6(a_a, 9,  c2, 6); DQ6(a_a, 10, c2, 12); DQ6(a_a, 11, c2, 18);
            DQ6(a_a, 12, c3, 0); DQ6(a_a, 13, c3, 6); DQ6(a_a, 14, c3, 12); DQ6(a_a, 15, c3, 18);
            #undef DQ6

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
        }
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    const int out_col = slot_start + lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
