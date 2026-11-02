#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) grouped-WMMA GEMM for MoE prefill over MQ4G256V2 (qt=44)
// routed experts. gfx12 sister of `gemm_mq4g256v2_moe_grouped_wmma_k2`.
//
// Why this kernel has to exist: qt44 was the ONLY MoE quant type with a
// grouped kernel and no gfx12 variant, so the launcher refused RDNA4 outright
// ("no gfx12 kernel for qt44 MoE experts"). Routed experts are ~99% of an A3B
// MoE's tensors, so on RDNA4 a qt44 MoE model could not prefill at all.
//
// Port contract: the weight decode is byte-for-byte the gfx11 sister's. The
// ONLY changes are the wave32 WMMA operand shape and the C-output mapping,
// mirroring the `gemm_mq3g256_lloyd_moe_grouped_wmma_k2` → `...wmma.gfx12`
// port:
//   - half16_t (16 fp16/lane) → half8_t (8 fp16/lane)
//   - K split across two lane-groups: k_grp = tid >> 4, m_lane = tid & 15
//   - __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 → ..._w32_gfx12
//   - C-mapping acc[j] = C[2*j + (tid>>4)][tid&15] → C[8*k_grp + j][m_lane]
//   - K4 unroll (4 K-tiles per inner body) instead of K2
//
// Group layout, 136 B stride (identical to the gfx11 sister):
//
//   qt44 MQ4G256V2    [0..2)  f16 scale, weights   0..127
//                     [2..4)  f16 zero,  weights   0..127
//                     [4..6)  f16 scale, weights 128..255
//                     [6..8)  f16 zero,  weights 128..255
//                     [8..136) 128 nibble bytes
//
// Nibble packing: byte i holds weight 2i in the low nibble and weight 2i+1 in
// the high nibble. Eight consecutive weights are therefore exactly one u32.
//
// Affine-grid selection under the k_grp split. Tile kt covers weights
// [kt*16, kt*16+16), so the grid boundary at weight 128 falls on the tile 7/8
// seam — no K-tile straddles the two grids, and neither does either k_grp half
// of one (k_grp 0 takes [kt*16, kt*16+8), k_grp 1 takes [kt*16+8, kt*16+16)).
// The K4 body covers tiles {0-3},{4-7},{8-11},{12-15}, each wholly inside one
// grid, so `upper` is selected once per body rather than per tile.
//
// Degenerate groups: the encoder writes scale bits 0 and all nibbles 0 when
// hi == lo, giving 0 * q + zero == zero. That is the intended constant-group
// value and needs no special case here.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots × K], x_row_div = 1 → x_row = flat
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g256v2_moe_grouped_wmma_gfx12(
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
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    const half8_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0};

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

        // K4 unroll: 4 K-tiles per inner body, 4 bodies = 16 K-tiles per group.
        for (int kt = 0; kt < 16; kt += 4) {
            // All four tiles of this body lie in the same affine grid — see
            // the grid-selection note above.
            const bool upper = (kt >= 8);
            const _Float16 sc_h = upper ? sc_1 : sc_0;
            const _Float16 zp_h = upper ? zp_1 : zp_0;

            // Each lane reads ONE u32 (4 B = 8 nibbles = 8 weights) per K-tile,
            // at byte 8 + kt*8 + k_grp*4 within the group.
            const char* np = gp + 8 + kt * 8 + k_grp * 4;
            unsigned int pa = *(const unsigned int*)(np +  0);
            unsigned int pb = *(const unsigned int*)(np +  8);
            unsigned int pc = *(const unsigned int*)(np + 16);
            unsigned int pd = *(const unsigned int*)(np + 24);

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

            #define DQ(d, pk, sh) (d) = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            // One half8_t covers 8 weights (one k_grp's half of a K-tile).
            #define DQ_8(d, pk) \
                DQ((d)[0], (pk),  0); DQ((d)[1], (pk),  4); \
                DQ((d)[2], (pk),  8); DQ((d)[3], (pk), 12); \
                DQ((d)[4], (pk), 16); DQ((d)[5], (pk), 20); \
                DQ((d)[6], (pk), 24); DQ((d)[7], (pk), 28)

            half8_t a_reg;
            DQ_8(a_reg, pa); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            DQ_8(a_reg, pb); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            DQ_8(a_reg, pc); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            DQ_8(a_reg, pd); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
            #undef DQ_8
            #undef DQ
        }
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
