// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// gfx1151 i8-WMMA MMQ MoE grouped GEMM for HFQ6/MQ6 G256.
//
// This is the HFQ6/MQ6 sister of
// `gemm_hfq4g256_moe_grouped_mmq_gfx1151`. X is prequantized to Q8_1
// blocks; HFQ6 weights are unpacked as unsigned int8 values in [0, 63].
//
// For each 32-K sub-block:
//   dot((sc * q + zp), x)
//     = sc * d_x * dot_i8(q, qx) + zp * sum_x
//
// The 6-bit q range fits signed i8, so the gfx11 i8 WMMA intrinsic can use
// signed_a=true without changing values.

#if defined(__gfx1151__)

#define QK8_1 32

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};

static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

typedef int8_t  __attribute__((ext_vector_type(16))) int8x16_t;
typedef int32_t __attribute__((ext_vector_type(4)))  int32x4_t;
typedef int32_t __attribute__((ext_vector_type(8)))  int32x8_t;
typedef float   __attribute__((ext_vector_type(8)))  float8_t;

// `x_row_div` controls how sorted_slot_index maps to X_q8_src rows:
//   gate_up: X_src_f32 had [N x K],       x_row_div = K_TOP
//   down:    X_src_f32 had [N*K_TOP x K], x_row_div = 1
//
// X_q8_src is produced by `quantize_q8_1_mmq_ds4`.
// Layout: X_q8_src[kb_q8 * x_src_rows + x_row].
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq6g256_moe_grouped_mmq_gfx1151(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E]
    const int*                __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const block_q8_1_mmq*     __restrict__ X_q8_src,           // [K/128 x x_src_rows]
    float*                    __restrict__ Y_grouped,          // [m_total x M]
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

    const int lane = tid & 15;
    const int slot_idx = slot_start + lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }

    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 200;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int blocks128 = K / 128;

    #define DQ6_4(dst, base, out) do { \
        unsigned int p = ((unsigned int)(base)[0]) \
                       | ((unsigned int)(base)[1] << 8) \
                       | ((unsigned int)(base)[2] << 16); \
        (dst)[(out) + 0] = (int8_t)((p >>  0) & 63u); \
        (dst)[(out) + 1] = (int8_t)((p >>  6) & 63u); \
        (dst)[(out) + 2] = (int8_t)((p >> 12) & 63u); \
        (dst)[(out) + 3] = (int8_t)((p >> 18) & 63u); \
    } while (0)

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 200;
        const float sc_self = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp_self = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        float sc_row[8];
        float zp_row[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int src_lane = 2 * j + (tid >> 4);
            sc_row[j] = __shfl(sc_self, src_lane);
            zp_row[j] = __shfl(zp_self, src_lane);
        }

        for (int kb_in_group = 0; kb_in_group < 2; kb_in_group++) {
            const int kb_q8 = 2 * g + kb_in_group;
            if (kb_q8 >= blocks128) break;

            const block_q8_1_mmq* xb = (x_row >= 0)
                ? (X_q8_src + (long long)kb_q8 * x_src_rows + x_row)
                : nullptr;

            #pragma unroll
            for (int sb = 0; sb < 4; sb++) {
                float d_x_self = 0.0f;
                float sum_x_self = 0.0f;
                if (xb != nullptr) {
                    half2 dm = xb->ds4[sb];
                    float2 dm_f = __half22float2(dm);
                    d_x_self = dm_f.x;
                    sum_x_self = dm_f.y;
                }

                int32x8_t cacc = {0, 0, 0, 0, 0, 0, 0, 0};

                #pragma unroll
                for (int kt = 0; kt < 2; kt++) {
                    const int k_off = kb_in_group * 128 + sb * 32 + kt * 16;
                    const unsigned char* dp = (const unsigned char*)(gp + 8 + (k_off / 16) * 12);

                    int8x16_t a_vec;
                    DQ6_4(a_vec, dp + 0, 0);
                    DQ6_4(a_vec, dp + 3, 4);
                    DQ6_4(a_vec, dp + 6, 8);
                    DQ6_4(a_vec, dp + 9, 12);

                    int8x16_t b_vec;
                    if (xb != nullptr) {
                        const int8_t* qs_p = xb->qs + sb * 32 + kt * 16;
                        b_vec = *(const int8x16_t*)qs_p;
                    } else {
                        b_vec = (int8x16_t){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                    }

                    int32x4_t a_i32 = *(const int32x4_t*)&a_vec;
                    int32x4_t b_i32 = *(const int32x4_t*)&b_vec;
                    cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                        true, a_i32, true, b_i32, cacc, true
                    );
                }

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc[j] += sc_row[j] * d_x_self * (float)cacc[j]
                            + zp_row[j] * sum_x_self;
                }
            }
        }
    }

    #undef DQ6_4

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

#else

extern "C" __global__ void gemm_hfq6g256_moe_grouped_mmq_gfx1151(
    const unsigned long long*, const int*, const int*,
    const void*, float*, int, int, int, int, int
) {}

#endif
