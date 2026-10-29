// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Batch-tiled gfx1100 WMMA residual for MQ4G256V2-Lloyd (qt=52) prefill.
// T0 experiment twin of gemm_mq4g256v2_residual_wmma_gfx11_bt.hip: identical
// geometry, grid, launch bounds and interleaved-C contract, except the
// per-nibble dequant reads the centered f16 codebook (16 levels as 8 kernarg
// dwords, staged once to 32 B of LDS) instead of the uniform affine grid:
// w = sc*C[q] + zp' with zp' pre-folded in the headers.
//
// Staging sits after the two block-uniform early returns, so the single
// __syncthreads is safe. LDS: 32 B static. No dynamic shared memory.
//
// Grid: [ceil(M/16), ceil(N/(16*B))]. Block: [32].
// Instantiated: bt4/6/8 (64/96/128 batch columns).
// Output: acc[j] = C[2*j + (tid>>4)][tid & 15] (interleaved).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define GEN_RESID_BT_LUT(BV) \
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq4g256v2_residual_wmma_gfx11_bt##BV##_lloyd( \
    const char* __restrict__ A, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y, \
    int M, int K, int N, \
    unsigned int LUT0, unsigned int LUT1, unsigned int LUT2, unsigned int LUT3, \
    unsigned int LUT4, unsigned int LUT5, unsigned int LUT6, unsigned int LUT7) { \
    const int row_start = blockIdx.x * 16; \
    const int batch_start = blockIdx.y * (16 * BV); \
    const int tid = threadIdx.x; \
    if (row_start >= M) return; \
    if (batch_start >= N) return; \
    __shared__ unsigned short LUT_SMEM[16]; \
    { \
        unsigned int d; \
        if (tid == 0) d = LUT0; else if (tid == 1) d = LUT1; \
        else if (tid == 2) d = LUT2; else if (tid == 3) d = LUT3; \
        else if (tid == 4) d = LUT4; else if (tid == 5) d = LUT5; \
        else if (tid == 6) d = LUT6; else d = LUT7; \
        if (tid < 8) ((unsigned int*)LUT_SMEM)[tid] = d; \
    } \
    __syncthreads(); \
    const int ml = tid & 15; \
    const int safe_row = (row_start + ml < M) ? (row_start + ml) : (M - 1); \
    const int groups_per_row = K / 256; \
    const char* row_base = A + (long long)safe_row * groups_per_row * 136; \
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
        const _Float16* xg[BV]; \
        for (int b = 0; b < BV; b++) \
            xg[b] = X + (long long)sb[b] * K + g * 256; \
        _Pragma("unroll") \
        for (int kt = 0; kt < 16; kt++) { \
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1; \
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1; \
            const int k_off = kt * 16; \
            unsigned int pk0 = *(const unsigned int*)(gp + 8 + k_off / 2); \
            unsigned int pk1 = *(const unsigned int*)(gp + 8 + k_off / 2 + 4); \
            half16_t a_reg; \
            a_reg[0]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[((pk0 >>  0) & 0xFu)]) + zp_h; \
            a_reg[1]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[((pk0 >>  4) & 0xFu)]) + zp_h; \
            a_reg[2]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[((pk0 >>  8) & 0xFu)]) + zp_h; \
            a_reg[3]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[((pk0 >> 12) & 0xFu)]) + zp_h; \
            a_reg[4]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[((pk0 >> 16) & 0xFu)]) + zp_h; \
            a_reg[5]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[((pk0 >> 20) & 0xFu)]) + zp_h; \
            a_reg[6]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[((pk0 >> 24) & 0xFu)]) + zp_h; \
            a_reg[7]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[((pk0 >> 28) & 0xFu)]) + zp_h; \
            a_reg[8]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[((pk1 >>  0) & 0xFu)]) + zp_h; \
            a_reg[9]  = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[((pk1 >>  4) & 0xFu)]) + zp_h; \
            a_reg[10] = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[((pk1 >>  8) & 0xFu)]) + zp_h; \
            a_reg[11] = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[((pk1 >> 12) & 0xFu)]) + zp_h; \
            a_reg[12] = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[((pk1 >> 16) & 0xFu)]) + zp_h; \
            a_reg[13] = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[((pk1 >> 20) & 0xFu)]) + zp_h; \
            a_reg[14] = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[((pk1 >> 24) & 0xFu)]) + zp_h; \
            a_reg[15] = sc_h * (_Float16)__ushort_as_half(LUT_SMEM[((pk1 >> 28) & 0xFu)]) + zp_h; \
            _Pragma("unroll") \
            for (int b = 0; b < BV; b++) { \
                half16_t b_reg = *(const half16_t*)(xg[b] + k_off); \
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc[b]); \
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
                if (out_row < M) { \
                    Y[(long long)out_col * M + out_row] += acc[b][j]; \
                } \
            } \
        } \
    } \
}

GEN_RESID_BT_LUT(4)
GEN_RESID_BT_LUT(6)
GEN_RESID_BT_LUT(8)
