#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) grouped-WMMA GEMM for MoE prefill over drop-zero planar MQ4G256V2P (qt=137)
// routed experts (DESIGN F1). Twin of `gemm_mq4g256v2_moe_grouped_wmma_gfx12`
// (gemm_mq4g256v2_moe_grouped_wmma.gfx12.hip): same grid, block, kernargs, K4 unroll, WMMA
// operand mapping, fp16 dequant `d = sc * q + zp` and fp16 accumulation order, so the output
// is bit-identical to the shipped kernel fed the shipped (qt44) bytes of the same weights.
//
// Expert tensor [M, K] is planar (hipfire_runtime::mq4_planar):
//   [codes plane: M*K/2 bytes][scales plane: M*(K/128) LE f16]
//   row r codes   : base + r*(K/2)
//   row r scales  : base + M*(K/2) + r*(K/256)*4;  scale word of 256-group g is the aligned u32
//                   at +4g: low f16 = weights 0..127 (sc_0), high f16 = weights 128..255 (sc_1)
// The shipped zero is f16(-8*sc); in f16 arithmetic -8 * sc is exact (x8 only moves the
// exponent; the loader proved the shipped zero finite), so zp = (_Float16)(-8) * sc_h is that
// same value.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g256v2p_moe_grouped_wmma_gfx12(
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

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int slot_idx = slot_start + m_lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base = (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* codes_row = A + (long long)safe_row * (K / 2);
    const char* scales_row = A + (long long)M * (K / 2) + (long long)safe_row * groups_per_row * 4;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    const half8_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0};
    const _Float16 neg8 = (_Float16)(-8.0f);

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = codes_row + g * 128;

        const unsigned int hdr = *(const unsigned int*)(scales_row + g * 4);
        const _Float16 sc_0 = __builtin_bit_cast(_Float16, (unsigned short)(hdr & 0xFFFFu));
        const _Float16 sc_1 = __builtin_bit_cast(_Float16, (unsigned short)(hdr >> 16));
        const _Float16 zp_0 = neg8 * sc_0;
        const _Float16 zp_1 = neg8 * sc_1;

        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        for (int kt = 0; kt < 16; kt += 4) {
            const bool upper = (kt >= 8);
            const _Float16 sc_h = upper ? sc_1 : sc_0;
            const _Float16 zp_h = upper ? zp_1 : zp_0;

            // Each lane reads ONE u32 (8 nibbles = 8 weights) per K-tile, at codes byte
            // kt*8 + k_grp*4 within the group's 128-byte code block.
            const char* np = gp + kt * 8 + k_grp * 4;
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
