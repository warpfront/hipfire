#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Grouped-WMMA-GEMM for MoE prefill over MQ4G256V2 (qt=44) routed experts.
//
// Transcribed from `gemm_hfq4g256_moe_grouped_wmma_k2` (qt=13). The gather /
// tiling / WMMA pipeline is identical, line for line. The ONLY difference is
// the group header and which affine grid each K-tile dequantizes against.
//
// Why this kernel has to exist: qt44 had GEMV (decode) and dense GEMM paths
// but no MoE grouped-expert path at all, so a qt44 A3B MoE model failed prefill
// with `unsupported moe.prefill-grouped-gemm-dtype`. Routed experts are ~99% of
// such a model's tensors, so without this the format cannot be used for MoE.
//
// Group layout, 136 B stride for BOTH formats:
//
//   qt13 MQ4G256      [0..4)  f32 scale (all 256 weights)
//                     [4..8)  f32 min   (all 256 weights)
//                     [8..136) 128 nibble bytes
//
//   qt44 MQ4G256V2    [0..2)  f16 scale, weights   0..127
//                     [2..4)  f16 zero,  weights   0..127
//                     [4..6)  f16 scale, weights 128..255
//                     [6..8)  f16 zero,  weights 128..255
//                     [8..136) 128 nibble bytes   (identical packing)
//
// Nibble packing is unchanged: byte i holds weight 2i in the low nibble and
// weight 2i+1 in the high nibble. The affine-grid boundary at weight 128
// therefore falls exactly on byte 64, so no byte straddles the two grids.
//
// K-tile → grid mapping: tile kt covers weights [kt*16, kt*16+16). Tiles 0-7
// are weights 0..127 (grid 0); tiles 8-15 are weights 128..255 (grid 1). The
// loop consumes tiles in pairs (kt, kt+1) with kt even, and every such pair
// lies wholly inside one grid — (6,7) → weights 96..127, (8,9) → 128..159.
// So the grid is selected once per pair, not per tile.
//
// Degenerate groups: the encoder writes scale bits 0 and all nibbles 0 when
// hi == lo, giving 0 * q + zero == zero. That is the intended constant-group
// value and needs no special case here.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots × K], x_row_div = 1 → x_row = flat
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g256v2_moe_grouped_wmma_k2(
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
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;

        // Dual affine grid. Stored as raw f16 bits, unlike qt13's f32 pair.
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

            unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
            unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);
            unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);
            unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);

            half16_t b_a, b_b;
            if (xg != nullptr) {
                b_a = *(const half16_t*)(xg + k_off_a);
                b_b = *(const half16_t*)(xg + k_off_b);
            } else {
                b_a = zero_b;
                b_b = zero_b;
            }

            half16_t a_a;
            #define DQ(reg, i, pk, sh) reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(a_a, 0,  pk0a,  0); DQ(a_a, 1,  pk0a,  4); DQ(a_a, 2,  pk0a,  8); DQ(a_a, 3,  pk0a, 12);
            DQ(a_a, 4,  pk0a, 16); DQ(a_a, 5,  pk0a, 20); DQ(a_a, 6,  pk0a, 24); DQ(a_a, 7,  pk0a, 28);
            DQ(a_a, 8,  pk1a,  0); DQ(a_a, 9,  pk1a,  4); DQ(a_a, 10, pk1a,  8); DQ(a_a, 11, pk1a, 12);
            DQ(a_a, 12, pk1a, 16); DQ(a_a, 13, pk1a, 20); DQ(a_a, 14, pk1a, 24); DQ(a_a, 15, pk1a, 28);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

            DQ(a_a, 0,  pk0b,  0); DQ(a_a, 1,  pk0b,  4); DQ(a_a, 2,  pk0b,  8); DQ(a_a, 3,  pk0b, 12);
            DQ(a_a, 4,  pk0b, 16); DQ(a_a, 5,  pk0b, 20); DQ(a_a, 6,  pk0b, 24); DQ(a_a, 7,  pk0b, 28);
            DQ(a_a, 8,  pk1b,  0); DQ(a_a, 9,  pk1b,  4); DQ(a_a, 10, pk1b,  8); DQ(a_a, 11, pk1b, 12);
            DQ(a_a, 12, pk1b, 16); DQ(a_a, 13, pk1b, 20); DQ(a_a, 14, pk1b, 24); DQ(a_a, 15, pk1b, 28);
            #undef DQ

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
        }
    }

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
