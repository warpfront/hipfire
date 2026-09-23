#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx11/RDNA3 (incl. gfx1151/RDNA3.5) HFQ6/MQ6 sister of
// `gemm_hfq6g256_moe_grouped_wmma_gfx12`. Pure composition of two
// already-validated patterns:
//   - The gfx11 grouped-WMMA structure from
//     `gemm_hfq4g256_moe_grouped_wmma_k2` (half16 operands, RDNA3
//     `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, C-mapping
//     acc[j] = C[2*j + (tid>>4)][tid&15], 2× K-tile pipeline, expert
//     scatter via expert_tile_ids/sorted_slot_index, x_row_div gather).
//   - The HFQ6-G256 6-bit dequant from `gemm_hfq6g256_moe_grouped_wmma.gfx12`
//     (200 B/group: 4 B scale + 4 B zero + 192 B packed 6-bit).
//
// HFQ6 vs HFQ4 deltas (everything else identical to the HFQ4 _k2 kernel):
//   - Group stride: 200 B instead of 136 B.
//   - Each 16-K tile is 12 B (16 × 6 bits = 96 bits) instead of 8 B. The
//     16 values are read as 4 overlapping unsigned-int loads at +0/+3/+6/+9
//     and extracted with shifts 0/6/12/18 × mask 63u (4 values per uint).
//     The +3-stride overlap and bit ordering match both the on-disk packing
//     and the gfx12 sister's extraction exactly.
//
// MQ6G256 uses the IDENTICAL on-disk layout (FWHT rotation is applied to X
// by the caller before dispatch), so the kernel is dtype-agnostic between
// HFQ6 and MQ6.
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots × K], x_row_div = 1 → x_row = flat
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq6g256_moe_grouped_wmma_k2(
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

    // Sentinel early-return: expert_tile_ids[tile_y] == -1 marks tiles
    // beyond the live m_total (the dispatcher may launch up to
    // m_total_max/16 tiles to skip the m_total dtoh sync).
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    // X gather: each lane (tid & 15) gets its own routed slot. Padding lanes
    // (sorted_slot_index == -1) get a NULL gather → zero B row.
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

    // A-row selection: pad OOB rows to the last valid one (the masked output
    // write below discards the result anyway).
    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 200;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 200;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        // HFQ6 16-value dequant from a 12-byte K-tile: 4 overlapping uint32
        // at +0/+3/+6/+9, each yielding 4 values via shifts 0/6/12/18 & 63.
        #define DQ6(reg, i, dw, sh) reg[i] = sc_h * (_Float16)(float)(((dw) >> (sh)) & 63u) + zp_h
        #define DQ6_16(reg, e0, e1, e2, e3) \
            DQ6(reg, 0,  (e0), 0);  DQ6(reg, 1,  (e0), 6);  DQ6(reg, 2,  (e0), 12); DQ6(reg, 3,  (e0), 18); \
            DQ6(reg, 4,  (e1), 0);  DQ6(reg, 5,  (e1), 6);  DQ6(reg, 6,  (e1), 12); DQ6(reg, 7,  (e1), 18); \
            DQ6(reg, 8,  (e2), 0);  DQ6(reg, 9,  (e2), 6);  DQ6(reg, 10, (e2), 12); DQ6(reg, 11, (e2), 18); \
            DQ6(reg, 12, (e3), 0);  DQ6(reg, 13, (e3), 6);  DQ6(reg, 14, (e3), 12); DQ6(reg, 15, (e3), 18)

        // 8 iterations × 2 K-tiles each = 16 K-tiles per group.
        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;

            // Weight: HFQ6 12-byte K-tiles at gp + 8 + kt*12.
            const unsigned char* dpa = (const unsigned char*)(gp + 8 + kt * 12);
            const unsigned char* dpb = (const unsigned char*)(gp + 8 + (kt + 1) * 12);
            unsigned int e0a = *(const unsigned int*)(dpa + 0);
            unsigned int e1a = *(const unsigned int*)(dpa + 3);
            unsigned int e2a = *(const unsigned int*)(dpa + 6);
            unsigned int e3a = *(const unsigned int*)(dpa + 9);
            unsigned int e0b = *(const unsigned int*)(dpb + 0);
            unsigned int e1b = *(const unsigned int*)(dpb + 3);
            unsigned int e2b = *(const unsigned int*)(dpb + 6);
            unsigned int e3b = *(const unsigned int*)(dpb + 9);

            // X loads — substitute zero vector when the lane has no source token.
            half16_t b_a, b_b;
            if (xg != nullptr) {
                b_a = *(const half16_t*)(xg + k_off_a);
                b_b = *(const half16_t*)(xg + k_off_b);
            } else {
                b_a = zero_b;
                b_b = zero_b;
            }

            half16_t a_a;
            DQ6_16(a_a, e0a, e1a, e2a, e3a);
            // WMMA A — b_b may still be in flight.
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

            DQ6_16(a_a, e0b, e1b, e2b, e3b);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
        }
        #undef DQ6_16
        #undef DQ6
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // A operand = weight rows (M-dim), B operand = slot lane (m_total-dim).
    // Direct store (=) since Y_grouped is fresh per call. Padding lanes
    // (slot_idx >= m_total) are silently dropped.
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
