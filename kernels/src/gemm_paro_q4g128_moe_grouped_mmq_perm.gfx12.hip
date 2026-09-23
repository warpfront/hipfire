// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// G128 i8 MMQ MoE grouped-GEMM for HFQ4G128 (ParoQuant) on gfx12, with
// vectorized nibble unpack via `__builtin_amdgcn_perm` (v_perm_b32).
//
// === Why the perm lever ===
//
// The k2 baseline (`gemm_paro_q4g128_moe_grouped_mmq.gfx12.hip`) unpacks
// each pk32 (8 nibbles) using 8 scalar `(pk >> X) & 0xFu` ops + 8 byte
// stores into an int8x8_t. Compiler does NOT auto-vectorize this — disasm
// confirms 54 v_lshrrev_b32 instructions and 0 v_perm_b32 instructions
// across the WMMA inner loop.
//
// This variant rewrites the unpack as:
//   lo = pk & 0x0F0F0F0F           (1 v_and_b32)
//   hi = (pk >> 4) & 0x0F0F0F0F    (1 v_lshrrev_b32 + 1 v_and_b32, often fused)
//   int32_0 = perm(lo, hi, 0x05010400)  // interleave {lo[0], hi[0], lo[1], hi[1]}
//                                          → bytes {n0, n1, n2, n3}
//   int32_1 = perm(lo, hi, 0x07030602)  // interleave {lo[2], hi[2], lo[3], hi[3]}
//                                          → bytes {n4, n5, n6, n7}
//
// Total: ~4 instructions per pk vs 16 scalar ops. With 8 WMMA k-substeps
// per group × 16 groups per row, savings are 1500+ VALU ops per WG.
//
// Selector encoding (per LLVM amdgcn.perm convention):
//   sel.byte[i] in [0,3]: result.byte[i] = src0.byte[sel.byte[i]]
//   sel.byte[i] in [4,7]: result.byte[i] = src1.byte[sel.byte[i] - 4]
// First builtin arg = src0; second = src1.
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

// Pack 8 nibbles from pk32 into the int32x2 layout expected by the WMMA
// i8 16x16x16 intrinsic. Result bytes {n0, n1, n2, n3, n4, n5, n6, n7}
// where pk[bits 4i+3:4i] = ni.
static __device__ __forceinline__ int32x2_t unpack_q4_nibbles(unsigned int pk) {
    const unsigned int lo = pk & 0x0F0F0F0Fu;
    const unsigned int hi = (pk >> 4) & 0x0F0F0F0Fu;
    int32x2_t out;
    // sel = 0x05010400 → {src0.b[0], src1.b[0], src0.b[1], src1.b[1]} = {lo[0],hi[0],lo[1],hi[1]}
    // sel = 0x07030602 → {src0.b[2], src1.b[2], src0.b[3], src1.b[3]} = {lo[2],hi[2],lo[3],hi[3]}
    out[0] = (int32_t)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
    out[1] = (int32_t)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
    return out;
}

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_paro_q4g128_moe_grouped_mmq_perm_gfx12(
    const unsigned long long*  __restrict__ expert_weight_ptrs, // [E]
    const int*                 __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                 __restrict__ sorted_slot_index,  // [m_total]
    const block_q8_1_mmq*      __restrict__ X_q8_src,           // [K/128 × x_src_rows]
    float*                     __restrict__ Y_grouped,          // [m_total × M]
    int M, int K, int x_row_div, int m_total, int x_src_rows
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int slot_idx = slot_start + m_lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }

    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int groups_per_row = K / 128;
    const char* row_base = A + (long long)safe_row * groups_per_row * 72;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 72;
        const float sc_self = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp_self = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        float sc_row[8];
        float zp_row[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int src_lane = 8 * k_grp + j;
            sc_row[j] = __shfl(sc_self, src_lane);
            zp_row[j] = __shfl(zp_self, src_lane);
        }

        const int kb_q8 = g;
        const block_q8_1_mmq* xb = (x_row >= 0)
            ? (X_q8_src + (long long)kb_q8 * x_src_rows + x_row)
            : nullptr;

        // Preload all 8 operand slots per group.
        unsigned int pk_arr[8];
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
                const int k_off = sb * 32 + kt * 16 + k_grp * 8;
                pk_arr[slot] = *(const unsigned int*)(gp + 8 + k_off / 2);
                if (xb != nullptr) {
                    const int8_t* qs_p = xb->qs + sb * 32 + kt * 16 + k_grp * 8;
                    b_arr[slot] = *(const int8x8_t*)qs_p;
                } else {
                    b_arr[slot] = (int8x8_t){0, 0, 0, 0, 0, 0, 0, 0};
                }
            }
        }

        // 4 sub-blocks × 2 WMMA-kt. Vectorized nibble unpack via perm_b32.
        #pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            int32x8_t cacc = {0, 0, 0, 0, 0, 0, 0, 0};

            #pragma unroll
            for (int kt = 0; kt < 2; kt++) {
                const int slot = sb * 2 + kt;
                const int32x2_t a_i32 = unpack_q4_nibbles(pk_arr[slot]);
                const int32x2_t b_i32 = *(const int32x2_t*)&b_arr[slot];

                cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                    false, a_i32, true, b_i32, cacc, true
                );
            }

            const float d_x_self   = d_x_arr[sb];
            const float sum_x_self = sum_x_arr[sb];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float dot_f = (float)cacc[j];
                acc[j] += sc_row[j] * d_x_self * dot_f
                        + zp_row[j] * sum_x_self;
            }
        }
    }

    const int out_col = slot_start + m_lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 8 * k_grp + j;
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}

#else

extern "C" __global__ void gemm_paro_q4g128_moe_grouped_mmq_perm_gfx12(
    const unsigned long long*, const int*, const int*,
    const void*, float*, int, int, int, int, int
) {}

#endif
