// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.
//
// Batch-tiled gfx1100 WMMA QKVZA (MQ4G256V2-Lloyd (qt=52) prefill).
// T0+ twin of gemm_qkvza_mq4g256v2_wmma_gfx11_bt.hip: identical geometry, grid, launch bounds and
// interleaved-C contract, except the per-nibble dequant reads the centered
// f16 codebook (8 kernel-arg dwords per tensor staged to LDS) instead of
// the uniform affine grid: w = sc*C[q] + zp' with zp' pre-folded in the
// headers. Distinct `_lloyd` symbols so the module cannot alias uniform.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define GEN_QKVZA_BT_LUT(BV) GEN_QKVZA_BT_LUT_I(BV, _lloyd)
#define GEN_QKVZA_BT_LUT_I(BV, SUF) \
extern "C" __launch_bounds__(32, 2) __global__ void gemm_qkvza_mq4g256v2_wmma_gfx11_bt##BV##SUF( \
    const char* __restrict__ A_qkv, \
    const char* __restrict__ A_z, \
    const char* __restrict__ A_beta, \
    const char* __restrict__ A_alpha, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y_qkv, \
    float* __restrict__ Y_z, \
    float* __restrict__ Y_beta, \
    float* __restrict__ Y_alpha, \
    int qkv_m, int z_m, int beta_m, int alpha_m, \
    int K, int N, \
    unsigned int LUT_Q0, \
    unsigned int LUT_Q1, \
    unsigned int LUT_Q2, \
    unsigned int LUT_Q3, \
    unsigned int LUT_Q4, \
    unsigned int LUT_Q5, \
    unsigned int LUT_Q6, \
    unsigned int LUT_Q7, \
    unsigned int LUT_Z0, \
    unsigned int LUT_Z1, \
    unsigned int LUT_Z2, \
    unsigned int LUT_Z3, \
    unsigned int LUT_Z4, \
    unsigned int LUT_Z5, \
    unsigned int LUT_Z6, \
    unsigned int LUT_Z7, \
    unsigned int LUT_B0, \
    unsigned int LUT_B1, \
    unsigned int LUT_B2, \
    unsigned int LUT_B3, \
    unsigned int LUT_B4, \
    unsigned int LUT_B5, \
    unsigned int LUT_B6, \
    unsigned int LUT_B7, \
    unsigned int LUT_A0, \
    unsigned int LUT_A1, \
    unsigned int LUT_A2, \
    unsigned int LUT_A3, \
    unsigned int LUT_A4, \
    unsigned int LUT_A5, \
    unsigned int LUT_A6, \
    unsigned int LUT_A7 \
) { \
    const int total_m = qkv_m + z_m + beta_m + alpha_m; \
    const int row_start = blockIdx.x * 16; \
    const int batch_start = blockIdx.y * (16 * BV); \
    const int tid = threadIdx.x; \
    if (row_start >= total_m) return; \
    if (batch_start >= N) return; \
    __shared__ unsigned short LUT_SMEM[64]; \
{ \
unsigned int d; \
if (tid == 0) d = LUT_Q0; else if (tid == 1) d = LUT_Q1; else if (tid == 2) d = LUT_Q2; else if (tid == 3) d = LUT_Q3; else if (tid == 4) d = LUT_Q4; else if (tid == 5) d = LUT_Q5; else if (tid == 6) d = LUT_Q6; else if (tid == 7) d = LUT_Q7; else if (tid == 8) d = LUT_Z0; else if (tid == 9) d = LUT_Z1; else if (tid == 10) d = LUT_Z2; else if (tid == 11) d = LUT_Z3; else if (tid == 12) d = LUT_Z4; else if (tid == 13) d = LUT_Z5; else if (tid == 14) d = LUT_Z6; else if (tid == 15) d = LUT_Z7; else if (tid == 16) d = LUT_B0; else if (tid == 17) d = LUT_B1; else if (tid == 18) d = LUT_B2; else if (tid == 19) d = LUT_B3; else if (tid == 20) d = LUT_B4; else if (tid == 21) d = LUT_B5; else if (tid == 22) d = LUT_B6; else if (tid == 23) d = LUT_B7; else if (tid == 24) d = LUT_A0; else if (tid == 25) d = LUT_A1; else if (tid == 26) d = LUT_A2; else if (tid == 27) d = LUT_A3; else if (tid == 28) d = LUT_A4; else if (tid == 29) d = LUT_A5; else if (tid == 30) d = LUT_A6; else if (tid == 31) d = LUT_A7; else d = LUT_A7; \
if (tid < 32) ((unsigned int*)LUT_SMEM)[tid] = d; \
} \
__syncthreads(); \
    const int ml = tid & 15; \
    const int my_row = row_start + ml; \
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1); \
    const int lut_base = ((safe_row < qkv_m) ? 0 : ((safe_row < qkv_m + z_m) ? 16 : ((safe_row < qkv_m + z_m + beta_m) ? 32 : 48))); \
    const char* A; \
    int local_row; \
    if (safe_row < qkv_m) { \
        A = A_qkv; \
        local_row = safe_row; \
    } else if (safe_row < qkv_m + z_m) { \
        A = A_z; \
        local_row = safe_row - qkv_m; \
    } else if (safe_row < qkv_m + z_m + beta_m) { \
        A = A_beta; \
        local_row = safe_row - (qkv_m + z_m); \
    } else { \
        A = A_alpha; \
        local_row = safe_row - (qkv_m + z_m + beta_m); \
    } \
    const int groups_per_row = K / 256; \
    const long long row_stride = (long long)groups_per_row * 136; \
    const char* row_base = A + local_row * row_stride; \
    int sb[BV]; \
    for (int b = 0; b < BV; b++) { \
        int t = batch_start + b * 16 + ml; \
        sb[b] = (t < N) ? t : 0; \
    } \
    float8_t acc[BV]; \
    for (int b = 0; b < BV; b++) \
        acc[b] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; \
    for (int g = 0; g < groups_per_row; g++) { \
        const char* gp = row_base + g * 136; \
        const unsigned int hA = *(const unsigned int*)(gp); \
        const unsigned int hB = *(const unsigned int*)(gp + 4); \
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu))); \
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16))); \
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu))); \
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16))); \
        const _Float16* xb[BV]; \
        for (int b = 0; b < BV; b++) \
            xb[b] = X + (long long)sb[b] * K + g * 256; \
        for (int kt = 0; kt < 16; kt += 2) { \
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1; \
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1; \
            const int k_off_a = kt * 16; \
            unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2); \
            unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4); \
            const int k_off_b = (kt + 1) * 16; \
            unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2); \
            unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4); \
            half16_t a_reg; \
            a_reg[0]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk0a >> 0) & 0xFu)]) + zp_h; \
            a_reg[1]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk0a >> 4) & 0xFu)]) + zp_h; \
            a_reg[2]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk0a >> 8) & 0xFu)]) + zp_h; \
            a_reg[3]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk0a >> 12) & 0xFu)]) + zp_h; \
            a_reg[4]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk0a >> 16) & 0xFu)]) + zp_h; \
            a_reg[5]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk0a >> 20) & 0xFu)]) + zp_h; \
            a_reg[6]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk0a >> 24) & 0xFu)]) + zp_h; \
            a_reg[7]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk0a >> 28) & 0xFu)]) + zp_h; \
            a_reg[8]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk1a >> 0) & 0xFu)]) + zp_h; \
            a_reg[9]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk1a >> 4) & 0xFu)]) + zp_h; \
            a_reg[10] = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk1a >> 8) & 0xFu)]) + zp_h; \
            a_reg[11] = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk1a >> 12) & 0xFu)]) + zp_h; \
            a_reg[12] = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk1a >> 16) & 0xFu)]) + zp_h; \
            a_reg[13] = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk1a >> 20) & 0xFu)]) + zp_h; \
            a_reg[14] = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk1a >> 24) & 0xFu)]) + zp_h; \
            a_reg[15] = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk1a >> 28) & 0xFu)]) + zp_h; \
            _Pragma("unroll") \
            for (int b = 0; b < BV; b++) { \
                half16_t b_a = *(const half16_t*)(xb[b] + k_off_a); \
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc[b]); \
            } \
            a_reg[0]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk0b >> 0) & 0xFu)]) + zp_h; \
            a_reg[1]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk0b >> 4) & 0xFu)]) + zp_h; \
            a_reg[2]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk0b >> 8) & 0xFu)]) + zp_h; \
            a_reg[3]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk0b >> 12) & 0xFu)]) + zp_h; \
            a_reg[4]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk0b >> 16) & 0xFu)]) + zp_h; \
            a_reg[5]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk0b >> 20) & 0xFu)]) + zp_h; \
            a_reg[6]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk0b >> 24) & 0xFu)]) + zp_h; \
            a_reg[7]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk0b >> 28) & 0xFu)]) + zp_h; \
            a_reg[8]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk1b >> 0) & 0xFu)]) + zp_h; \
            a_reg[9]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk1b >> 4) & 0xFu)]) + zp_h; \
            a_reg[10] = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk1b >> 8) & 0xFu)]) + zp_h; \
            a_reg[11] = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk1b >> 12) & 0xFu)]) + zp_h; \
            a_reg[12] = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk1b >> 16) & 0xFu)]) + zp_h; \
            a_reg[13] = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk1b >> 20) & 0xFu)]) + zp_h; \
            a_reg[14] = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk1b >> 24) & 0xFu)]) + zp_h; \
            a_reg[15] = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[lut_base + ((pk1b >> 28) & 0xFu)]) + zp_h; \
            _Pragma("unroll") \
            for (int b = 0; b < BV; b++) { \
                half16_t b_b = *(const half16_t*)(xb[b] + k_off_b); \
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc[b]); \
            } \
        } \
    } \
    _Pragma("unroll") \
    for (int b = 0; b < BV; b++) { \
        const int out_col = batch_start + b * 16 + ml; \
        if (out_col < N) { \
            _Pragma("unroll") \
            for (int j = 0; j < 8; j++) { \
                int out_row = row_start + 2 * j + (tid >> 4); \
                if (out_row < total_m) { \
                    float* Y; \
                    int proj_row; \
                    int out_stride; \
                    if (out_row < qkv_m) { \
                        Y = Y_qkv; \
                        proj_row = out_row; \
                        out_stride = qkv_m; \
                    } else if (out_row < qkv_m + z_m) { \
                        Y = Y_z; \
                        proj_row = out_row - qkv_m; \
                        out_stride = z_m; \
                    } else if (out_row < qkv_m + z_m + beta_m) { \
                        Y = Y_beta; \
                        proj_row = out_row - (qkv_m + z_m); \
                        out_stride = beta_m; \
                    } else { \
                        Y = Y_alpha; \
                        proj_row = out_row - (qkv_m + z_m + beta_m); \
                        out_stride = alpha_m; \
                    } \
                    Y[(long long)out_col * out_stride + proj_row] = acc[b][j]; \
                } \
            } \
        } \
    } \
}

GEN_QKVZA_BT_LUT(1)
GEN_QKVZA_BT_LUT(4)
GEN_QKVZA_BT_LUT(12)
