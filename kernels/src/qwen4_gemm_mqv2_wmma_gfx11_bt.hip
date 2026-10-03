// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Qwen4 copy of gemm_mqv2_wmma_gfx11_bt.hip as #774 left it (split-decode residual BT and
// X-LDS kernels on gfx1151, the MQ6 BT8 x4 entries). Qwen4's entries load from here on every
// arch, the shipped modules on gfx1151 only.
//
// Generic batch-tiled gfx11 WMMA prefill for MQ{2,3,5,6}G256V2.
// Modeled on accepted MQ4 gfx11 BT kernels:
//   gemm_qkvza_mq4g256v2_wmma_gfx11_bt.hip  (overwrite BT4)
//   gemm_qkv_mq4g256v2_wmma_gfx11_bt.hip    (overwrite BT4)
//   gemm_gate_up_mq4g256v2_wmma_gfx11_bt.hip (overwrite BT12)
//   gemm_mq4g256v2_residual_wmma_gfx11_bt.hip (residual += BT4)
//
// V2 group: 8 B dual-FP16 header + contiguous bit-packed payload.
// GROUP_BYTES = 8 + 32*BITS. Half header selected by kt < 8.
// Weights decode once per K-tile and reuse across B independent N-tile accs.
// Output: acc[j] = C[2*j + (tid>>4)][tid & 15].
// Exact gfx1100 / gfx1151 only; launch_bounds(32, 2).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#if defined(__GFX11__)

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// Decode one 16-weight K-tile from contiguous bit-packed payload after the
// 8-byte header. Reads only the current payload byte and, when the code
// straddles a byte boundary ((bit_in_byte + BITS) > 8), the next byte.
// Never reads outside the tile's bit range within the group payload.
template <int BITS>
__device__ __forceinline__ half16_t decode_tile(const unsigned char* __restrict__ payload,
                                                int kt,
                                                _Float16 sc_h,
                                                _Float16 zp_h) {
    constexpr unsigned int qmask = (1u << BITS) - 1u;
    half16_t out;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        const int bit_pos = (kt * 16 + i) * BITS;
        const int byte_i = bit_pos >> 3;
        const int bit_in_byte = bit_pos & 7;
        unsigned int v = (unsigned int)payload[byte_i];
        if ((bit_in_byte + BITS) > 8)
            v |= ((unsigned int)payload[byte_i + 1]) << 8;
        const unsigned int q = (v >> bit_in_byte) & qmask;
        out[i] = sc_h * (_Float16)(float)q + zp_h;
    }
    return out;
}

// The WMMA A operand is replicated across the two 16-lane halves, so each
// half decodes only its own eight of the sixteen weights (elements 8h..8h+7,
// whose codes start exactly BITS bytes further for h = 1) and the halves swap
// their results through permlanex16.  Every weight is produced by the same
// expression as decode_tile, so the assembled operand is bitwise identical.
typedef _Float16 __attribute__((ext_vector_type(8))) half8_bt_t;
typedef unsigned int __attribute__((ext_vector_type(4))) uint4_bt_t;
typedef unsigned int __attribute__((ext_vector_type(8))) uint8_bt_t;
template <int BITS>
__device__ __forceinline__ half16_t decode_tile_split(const unsigned char* __restrict__ payload,
                                                      int kt,
                                                      _Float16 sc_h,
                                                      _Float16 zp_h,
                                                      int lane_half) {
    constexpr unsigned int qmask = (1u << BITS) - 1u;
    const unsigned char* __restrict__ p = payload + lane_half * BITS;
    half8_bt_t mine;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int bit_pos = (kt * 16 + j) * BITS;
        const int byte_i = bit_pos >> 3;
        const int bit_in_byte = bit_pos & 7;
        unsigned int v = (unsigned int)p[byte_i];
        if ((bit_in_byte + BITS) > 8)
            v |= ((unsigned int)p[byte_i + 1]) << 8;
        const unsigned int q = (v >> bit_in_byte) & qmask;
        mine[j] = sc_h * (_Float16)(float)q + zp_h;
    }
    const uint4_bt_t m = __builtin_bit_cast(uint4_bt_t, mine);
    uint4_bt_t o;
#pragma unroll
    for (int k = 0; k < 4; k++)
        o[k] = (unsigned int)__builtin_amdgcn_permlanex16(
            (int)m[k], (int)m[k], 0x76543210u, 0xfedcba98u, false, false);
    uint8_bt_t both;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        both[k] = lane_half ? o[k] : m[k];
        both[k + 4] = lane_half ? m[k] : o[k];
    }
    return __builtin_bit_cast(half16_t, both);
}

// decode_tile_split<6> from the tile's 48-bit little-endian code window
// `code` (bytes lane_half * 6 .. +6 of the tile's 12): q_j = bits 6j..6j+5,
// the same codes the byte-wise reader extracts, and the same expression.
__device__ __forceinline__ half16_t decode6_tile_split(unsigned long long code,
                                                       _Float16 sc_h,
                                                       _Float16 zp_h,
                                                       int lane_half) {
    half8_bt_t mine;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const unsigned int q = (unsigned int)(code >> (6 * j)) & 63u;
        mine[j] = sc_h * (_Float16)(float)q + zp_h;
    }
    const uint4_bt_t m = __builtin_bit_cast(uint4_bt_t, mine);
    uint4_bt_t o;
#pragma unroll
    for (int k = 0; k < 4; k++)
        o[k] = (unsigned int)__builtin_amdgcn_permlanex16(
            (int)m[k], (int)m[k], 0x76543210u, 0xfedcba98u, false, false);
    uint8_bt_t both;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        both[k] = lane_half ? o[k] : m[k];
        both[k + 4] = lane_half ? m[k] : o[k];
    }
    return __builtin_bit_cast(half16_t, both);
}

// Packed-pair variant of decode6_tile_split: two codes per VALU op.  The
// codes ride in the mantissas of 1024 + q (0x6400 | q, exact for q < 64);
// v_pk_add_f16 of -1024 (0xe400) recovers q exactly, and v_pk_fma_f16
// computes sc * q + zp in one rounding (RNE), the same fused expression as
// the scalar path's v_fma_f16 with the same operand order (sc, q, zp).
// Explicit asm pins the fusion and operand order against reordering or
// contraction.  `sc2`/`zp2` carry the raw half bits replicated in both halves.
__device__ __forceinline__ unsigned int pk_add_f16_bt(unsigned int a, unsigned int b) {
    unsigned int r;
    asm("v_pk_add_f16 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}
__device__ __forceinline__ unsigned int pk_fma_f16_bt(unsigned int a, unsigned int b,
                                                      unsigned int c) {
    unsigned int r;
    asm("v_pk_fma_f16 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
    return r;
}
__device__ __forceinline__ half16_t decode6_tile_split_pk(unsigned long long code,
                                                          unsigned int sc2,
                                                          unsigned int zp2,
                                                          int lane_half) {
    uint4_bt_t m;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const unsigned int q0 = (unsigned int)(code >> (12 * k)) & 63u;
        const unsigned int q1 = (unsigned int)(code >> (12 * k + 6)) & 63u;
        const unsigned int q2bits = 0x64006400u | q0 | (q1 << 16);
        const unsigned int q2 = pk_add_f16_bt(q2bits, 0xe400e400u);
        m[k] = pk_fma_f16_bt(sc2, q2, zp2);
    }
    uint4_bt_t o;
#pragma unroll
    for (int k = 0; k < 4; k++)
        o[k] = (unsigned int)__builtin_amdgcn_permlanex16(
            (int)m[k], (int)m[k], 0x76543210u, 0xfedcba98u, false, false);
    uint8_bt_t both;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        both[k] = lane_half ? o[k] : m[k];
        both[k + 4] = lane_half ? m[k] : o[k];
    }
    return __builtin_bit_cast(half16_t, both);
}

// ---------------------------------------------------------------------------
// QKVZA overwrite BT4 — four projection regions (QKV/Z/beta/alpha)
// Grid: [ceil((qkv_m+z_m+beta_m+alpha_m)/16), ceil(N/(16*BV))]. Block: [32].
// ---------------------------------------------------------------------------
#define GEN_QKVZA_BT(BITS, BV) \
extern "C" __launch_bounds__(32, 2) __global__ void \
gemm_qkvza_mq##BITS##g256v2_wmma_gfx11_bt##BV( \
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
    int K, int N \
) { \
    constexpr int GROUP_BYTES = 8 + 32 * (BITS); \
    const int total_m = qkv_m + z_m + beta_m + alpha_m; \
    const int row_start = blockIdx.x * 16; \
    const int batch_start = blockIdx.y * (16 * (BV)); \
    const int tid = threadIdx.x; \
    if (row_start >= total_m) return; \
    if (batch_start >= N) return; \
    const int ml = tid & 15; \
    const int my_row = row_start + ml; \
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1); \
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
    const long long row_stride = (long long)groups_per_row * GROUP_BYTES; \
    const char* row_base = A + local_row * row_stride; \
    int sb[BV]; \
    for (int b = 0; b < (BV); b++) { \
        int t = batch_start + b * 16 + ml; \
        sb[b] = (t < N) ? t : 0; \
    } \
    float8_t acc[BV]; \
    for (int b = 0; b < (BV); b++) \
        acc[b] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; \
    for (int g = 0; g < groups_per_row; g++) { \
        const char* gp = row_base + g * GROUP_BYTES; \
        const unsigned int hA = *(const unsigned int*)(gp); \
        const unsigned int hB = *(const unsigned int*)(gp + 4); \
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu))); \
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16))); \
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu))); \
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16))); \
        const unsigned char* payload = (const unsigned char*)(gp + 8); \
        const _Float16* xb[BV]; \
        for (int b = 0; b < (BV); b++) \
            xb[b] = X + (long long)sb[b] * K + g * 256; \
        for (int kt = 0; kt < 16; kt += 2) { \
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1; \
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1; \
            const int k_off_a = kt * 16; \
            const int k_off_b = (kt + 1) * 16; \
            half16_t a_reg = decode_tile<BITS>(payload, kt, sc_h, zp_h); \
            _Pragma("unroll") \
            for (int b = 0; b < (BV); b++) { \
                half16_t b_a = *(const half16_t*)(xb[b] + k_off_a); \
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc[b]); \
            } \
            a_reg = decode_tile<BITS>(payload, kt + 1, sc_h, zp_h); \
            _Pragma("unroll") \
            for (int b = 0; b < (BV); b++) { \
                half16_t b_b = *(const half16_t*)(xb[b] + k_off_b); \
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc[b]); \
            } \
        } \
    } \
    _Pragma("unroll") \
    for (int b = 0; b < (BV); b++) { \
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

// ---------------------------------------------------------------------------
// QKV overwrite BT4 — three projection regions (Q/K/V)
// Grid: [ceil((q_m+k_m+v_m)/16), ceil(N/(16*BV))]. Block: [32].
// ---------------------------------------------------------------------------
#define GEN_QKV_BT(BITS, BV) \
extern "C" __launch_bounds__(32, 2) __global__ void \
gemm_qkv_mq##BITS##g256v2_wmma_gfx11_bt##BV( \
    const char* __restrict__ A_q, \
    const char* __restrict__ A_k, \
    const char* __restrict__ A_v, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y_q, \
    float* __restrict__ Y_k, \
    float* __restrict__ Y_v, \
    int q_m, int k_m, int v_m, \
    int K, int N \
) { \
    constexpr int GROUP_BYTES = 8 + 32 * (BITS); \
    const int total_m = q_m + k_m + v_m; \
    const int row_start = blockIdx.x * 16; \
    const int batch_start = blockIdx.y * (16 * (BV)); \
    const int tid = threadIdx.x; \
    if (row_start >= total_m) return; \
    if (batch_start >= N) return; \
    const int ml = tid & 15; \
    const int my_row = row_start + ml; \
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1); \
    const char* A; \
    int local_row; \
    if (safe_row < q_m) { \
        A = A_q; \
        local_row = safe_row; \
    } else if (safe_row < q_m + k_m) { \
        A = A_k; \
        local_row = safe_row - q_m; \
    } else { \
        A = A_v; \
        local_row = safe_row - q_m - k_m; \
    } \
    const int groups_per_row = K / 256; \
    const long long row_stride = (long long)groups_per_row * GROUP_BYTES; \
    const char* row_base = A + local_row * row_stride; \
    int sb[BV]; \
    for (int b = 0; b < (BV); b++) { \
        int t = batch_start + b * 16 + ml; \
        sb[b] = (t < N) ? t : 0; \
    } \
    float8_t acc[BV]; \
    for (int b = 0; b < (BV); b++) \
        acc[b] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; \
    for (int g = 0; g < groups_per_row; g++) { \
        const char* gp = row_base + g * GROUP_BYTES; \
        const unsigned int hA = *(const unsigned int*)(gp); \
        const unsigned int hB = *(const unsigned int*)(gp + 4); \
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu))); \
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16))); \
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu))); \
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16))); \
        const unsigned char* payload = (const unsigned char*)(gp + 8); \
        const _Float16* xb[BV]; \
        for (int b = 0; b < (BV); b++) \
            xb[b] = X + (long long)sb[b] * K + g * 256; \
        for (int kt = 0; kt < 16; kt += 2) { \
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1; \
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1; \
            const int k_off_a = kt * 16; \
            const int k_off_b = (kt + 1) * 16; \
            half16_t a_reg = decode_tile<BITS>(payload, kt, sc_h, zp_h); \
            _Pragma("unroll") \
            for (int b = 0; b < (BV); b++) { \
                half16_t b_a = *(const half16_t*)(xb[b] + k_off_a); \
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc[b]); \
            } \
            a_reg = decode_tile<BITS>(payload, kt + 1, sc_h, zp_h); \
            _Pragma("unroll") \
            for (int b = 0; b < (BV); b++) { \
                half16_t b_b = *(const half16_t*)(xb[b] + k_off_b); \
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc[b]); \
            } \
        } \
    } \
    _Pragma("unroll") \
    for (int b = 0; b < (BV); b++) { \
        const int out_col = batch_start + b * 16 + ml; \
        if (out_col < N) { \
            _Pragma("unroll") \
            for (int j = 0; j < 8; j++) { \
                int out_row = row_start + 2 * j + (tid >> 4); \
                if (out_row < total_m) { \
                    float* Y; \
                    int proj_row; \
                    int out_stride; \
                    if (out_row < q_m) { \
                        Y = Y_q; \
                        proj_row = out_row; \
                        out_stride = q_m; \
                    } else if (out_row < q_m + k_m) { \
                        Y = Y_k; \
                        proj_row = out_row - q_m; \
                        out_stride = k_m; \
                    } else { \
                        Y = Y_v; \
                        proj_row = out_row - q_m - k_m; \
                        out_stride = v_m; \
                    } \
                    Y[(long long)out_col * out_stride + proj_row] = acc[b][j]; \
                } \
            } \
        } \
    } \
}

// ---------------------------------------------------------------------------
// Gate/Up overwrite BT12 — two projection regions (gate/up)
// Grid: [ceil((gate_m+up_m)/16), ceil(N/(16*BV))]. Block: [32].
// ---------------------------------------------------------------------------
#define GEN_GATE_UP_BT(BITS, BV) \
extern "C" __launch_bounds__(32, 2) __global__ void \
gemm_gate_up_mq##BITS##g256v2_wmma_gfx11_bt##BV( \
    const char* __restrict__ A_gate, \
    const char* __restrict__ A_up, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y_gate, \
    float* __restrict__ Y_up, \
    int gate_m, int up_m, \
    int K, int N \
) { \
    constexpr int GROUP_BYTES = 8 + 32 * (BITS); \
    const int total_m = gate_m + up_m; \
    const int row_start = blockIdx.x * 16; \
    const int batch_start = blockIdx.y * (16 * (BV)); \
    const int tid = threadIdx.x; \
    if (row_start >= total_m) return; \
    if (batch_start >= N) return; \
    const int ml = tid & 15; \
    const int my_row = row_start + ml; \
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1); \
    const char* A; \
    int local_row; \
    if (safe_row < gate_m) { \
        A = A_gate; \
        local_row = safe_row; \
    } else { \
        A = A_up; \
        local_row = safe_row - gate_m; \
    } \
    const int groups_per_row = K / 256; \
    const long long row_stride = (long long)groups_per_row * GROUP_BYTES; \
    const char* row_base = A + local_row * row_stride; \
    int sb[BV]; \
    for (int b = 0; b < (BV); b++) { \
        int t = batch_start + b * 16 + ml; \
        sb[b] = (t < N) ? t : 0; \
    } \
    float8_t acc[BV]; \
    for (int b = 0; b < (BV); b++) \
        acc[b] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; \
    for (int g = 0; g < groups_per_row; g++) { \
        const char* gp = row_base + g * GROUP_BYTES; \
        const unsigned int hA = *(const unsigned int*)(gp); \
        const unsigned int hB = *(const unsigned int*)(gp + 4); \
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu))); \
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16))); \
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu))); \
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16))); \
        const unsigned char* payload = (const unsigned char*)(gp + 8); \
        const _Float16* xb[BV]; \
        for (int b = 0; b < (BV); b++) \
            xb[b] = X + (long long)sb[b] * K + g * 256; \
        for (int kt = 0; kt < 16; kt += 2) { \
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1; \
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1; \
            const int k_off_a = kt * 16; \
            const int k_off_b = (kt + 1) * 16; \
            half16_t a_reg = decode_tile<BITS>(payload, kt, sc_h, zp_h); \
            _Pragma("unroll") \
            for (int b = 0; b < (BV); b++) { \
                half16_t b_a = *(const half16_t*)(xb[b] + k_off_a); \
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc[b]); \
            } \
            a_reg = decode_tile<BITS>(payload, kt + 1, sc_h, zp_h); \
            _Pragma("unroll") \
            for (int b = 0; b < (BV); b++) { \
                half16_t b_b = *(const half16_t*)(xb[b] + k_off_b); \
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc[b]); \
            } \
        } \
    } \
    _Pragma("unroll") \
    for (int b = 0; b < (BV); b++) { \
        const int out_col = batch_start + b * 16 + ml; \
        if (out_col < N) { \
            _Pragma("unroll") \
            for (int j = 0; j < 8; j++) { \
                int out_row = row_start + 2 * j + (tid >> 4); \
                if (out_row < total_m) { \
                    float* Y; \
                    int proj_row; \
                    int out_stride; \
                    if (out_row < gate_m) { \
                        Y = Y_gate; \
                        proj_row = out_row; \
                        out_stride = gate_m; \
                    } else { \
                        Y = Y_up; \
                        proj_row = out_row - gate_m; \
                        out_stride = up_m; \
                    } \
                    Y[(long long)out_col * out_stride + proj_row] = acc[b][j]; \
                } \
            } \
        } \
    } \
}

// ---------------------------------------------------------------------------
// Residual += BT4 — single projection, Y += W@X
// Grid: [ceil(M/16), ceil(N/(16*BV))]. Block: [32].
// Ascending WMMA order (kt = 0..15), matching MQ4 residual base.
// ---------------------------------------------------------------------------
// gfx1151 decodes each half's eight weights and swaps them through
// permlanex16 (decode_tile_split, bitwise equal); every other gfx11 arch keeps
// the full decode_tile so its residual BT kernels compile exactly as before.
#if defined(__gfx1151__)
#define RESID_BT_DECODE(BITS, payload, kt, sc_h, zp_h) \
    decode_tile_split<BITS>(payload, kt, sc_h, zp_h, tid >> 4)
#else
#define RESID_BT_DECODE(BITS, payload, kt, sc_h, zp_h) decode_tile<BITS>(payload, kt, sc_h, zp_h)
#endif
#define GEN_RESID_BT(BITS, BV) \
extern "C" __launch_bounds__(32, 2) __global__ void \
gemm_mq##BITS##g256v2_residual_wmma_gfx11_bt##BV( \
    const char* __restrict__ A, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y, \
    int M, int K, int N \
) { \
    constexpr int GROUP_BYTES = 8 + 32 * (BITS); \
    const int row_start = blockIdx.x * 16; \
    const int batch_start = blockIdx.y * (16 * (BV)); \
    const int tid = threadIdx.x; \
    if (row_start >= M) return; \
    if (batch_start >= N) return; \
    const int ml = tid & 15; \
    const int safe_row = (row_start + ml < M) ? (row_start + ml) : (M - 1); \
    const int groups_per_row = K / 256; \
    const char* row_base = A + (long long)safe_row * groups_per_row * GROUP_BYTES; \
    int sb[BV]; \
    for (int b = 0; b < (BV); b++) { \
        int t = batch_start + b * 16 + ml; \
        sb[b] = (t < N) ? t : 0; \
    } \
    float8_t acc[BV]; \
    for (int b = 0; b < (BV); b++) \
        acc[b] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; \
    for (int g = 0; g < groups_per_row; g++) { \
        const char* gp = row_base + g * GROUP_BYTES; \
        const unsigned int hA = *(const unsigned int*)(gp); \
        const unsigned int hB = *(const unsigned int*)(gp + 4); \
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu))); \
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16))); \
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu))); \
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16))); \
        const unsigned char* payload = (const unsigned char*)(gp + 8); \
        const _Float16* xg[BV]; \
        for (int b = 0; b < (BV); b++) \
            xg[b] = X + (long long)sb[b] * K + g * 256; \
        _Pragma("unroll") \
        for (int kt = 0; kt < 16; kt++) { \
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1; \
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1; \
            const int k_off = kt * 16; \
            half16_t a_reg = RESID_BT_DECODE(BITS, payload, kt, sc_h, zp_h); \
            _Pragma("unroll") \
            for (int b = 0; b < (BV); b++) { \
                half16_t b_reg = *(const half16_t*)(xg[b] + k_off); \
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc[b]); \
            } \
        } \
    } \
    _Pragma("unroll") \
    for (int b = 0; b < (BV); b++) { \
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

// Instantiations: BITS ∈ {2,3,5,6}; QKVZA/QKV/residual BT4; gate_up BT12.
// gfx1100 policy extras (bits 3/5/6 only — no MQ2 production variants):
//   QKVZA/QKV BT12, gate_up BT6, residual BT6/BT8.
GEN_QKVZA_BT(2, 4)
GEN_QKVZA_BT(3, 4)
GEN_QKVZA_BT(5, 4)
GEN_QKVZA_BT(6, 4)

GEN_QKV_BT(2, 4)
GEN_QKV_BT(3, 4)
GEN_QKV_BT(5, 4)
GEN_QKV_BT(6, 4)

GEN_GATE_UP_BT(2, 12)
GEN_GATE_UP_BT(3, 12)
GEN_GATE_UP_BT(5, 12)
GEN_GATE_UP_BT(6, 12)

GEN_RESID_BT(2, 4)
GEN_RESID_BT(3, 4)
GEN_RESID_BT(5, 4)
GEN_RESID_BT(6, 4)

// gfx1100 policy tiles for MQ{3,5,6}V2 (mirror accepted MQ4 BT set).
GEN_QKVZA_BT(3, 12)
GEN_QKVZA_BT(5, 12)
GEN_QKVZA_BT(6, 12)

GEN_QKV_BT(3, 12)
GEN_QKV_BT(5, 12)
GEN_QKV_BT(6, 12)

GEN_GATE_UP_BT(3, 6)
GEN_GATE_UP_BT(5, 6)
GEN_GATE_UP_BT(6, 6)

GEN_RESID_BT(3, 6)
GEN_RESID_BT(5, 6)
GEN_RESID_BT(6, 6)

GEN_RESID_BT(3, 8)
GEN_RESID_BT(5, 8)
GEN_RESID_BT(6, 8)
#ifndef HCW_EPILOGUE_BLOCK
#define HCW_EPILOGUE_BLOCK
// ---------------------------------------------------------------------------
// H4 HC-write epilogue: the producer's own output stage applies
// hyper_write_bf16x2 / hyper_write_f32's expression to the streams in place.
// Stage 1 (per accumulator lane, tokens x 8 channels): the GEMM output as the
// unfused kernel stores it (0.0f + acc) is rounded to BF16 bits and parked in
// LDS (the unfused HC write rounds it to BF16 before use, so nothing is lost);
// an optional F32 mirror gets the unrounded value.  Stage 2 (accumulators
// dead): each item = (token t, channels c0..c0+7) reads the four stream
// columns br*H + c0.. of token t (old values), applies the four gates (the
// paired read's hyper_norm_gate logits through the same expression as
// hyper_write_bf16x2) and overwrites them; one item owns its columns.
// Requires H % 8 == 0 (the caller's M).  Keep byte-identical across the files
// carrying this block (gfx11 bt, gfx12 bt, lds256).
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float hcw_bf16f(float v) {
    return static_cast<float>(static_cast<hip_bfloat16>(v));
}
static __device__ __forceinline__ unsigned int hcw_bf16bits(float v) {
    return (unsigned int)static_cast<hip_bfloat16>(v).data;
}
// NaN propagation is operand-order dependent (the first NaN source wins) and the
// compiler commutes fadd/fmul freely, so the two products/sums below are pinned to
// the operand order hyper_write_bf16x2 (stream + injection) and hyper_write_f32
// (injection + stream) compile to; gate * mixed is gate-first in both.
static __device__ __forceinline__ float hcw_mul(float a, float b) {
    float r;
    asm("v_mul_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}
static __device__ __forceinline__ float hcw_add(float a, float b) {
    float r;
    asm("v_add_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}
static __device__ __forceinline__ float hcw_gate(float logit) {
    const float gate_logit = hcw_bf16f(logit);
    const float gate_scaled = hcw_bf16f(gate_logit / 4.0f);
    return hcw_bf16f(2.0f * hcw_bf16f(1.0f / (1.0f + expf(-gate_scaled))));
}
// BF16 bits of (0.0f + r[i]), element i in half (i & 1) of word (i >> 1).
static __device__ __forceinline__ uint4 hcw_pack8(const float (&r)[8]) {
    unsigned int w[4];
#pragma unroll
    for (int p = 0; p < 4; p++)
        w[p] = hcw_bf16bits(0.0f + r[2 * p]) | (hcw_bf16bits(0.0f + r[2 * p + 1]) << 16);
    return make_uint4(w[0], w[1], w[2], w[3]);
}
static __device__ __forceinline__ void hcw_mirror8(const float (&r)[8], int t, int c0, int H,
                                                   float* mirror) {
    float* o = mirror + (long long)t * H + c0;
    *(float4*)o = make_float4(0.0f + r[0], 0.0f + r[1], 0.0f + r[2], 0.0f + r[3]);
    *(float4*)(o + 4) = make_float4(0.0f + r[4], 0.0f + r[5], 0.0f + r[6], 0.0f + r[7]);
}
static __device__ __forceinline__ void hcw_apply8(const uint4 mq, int t, int c0, int H,
                                                  void* streams, const float* gates,
                                                  int bf16_state) {
    const float4 gv = *(const float4*)(gates + (long long)t * 4);
    // All four stream runs are requested before any arithmetic consumes them.
    uint4 sw[4];
    float4 sf[4][2];
    if (bf16_state) {
        const unsigned short* base = (const unsigned short*)streams + (long long)t * 4 * H + c0;
#pragma unroll
        for (int br = 0; br < 4; br++) sw[br] = *(const uint4*)(base + (long long)br * H);
    } else {
        const float* base = (const float*)streams + (long long)t * 4 * H + c0;
#pragma unroll
        for (int br = 0; br < 4; br++) {
            sf[br][0] = *(const float4*)(base + (long long)br * H);
            sf[br][1] = *(const float4*)(base + (long long)br * H + 4);
        }
    }
    const unsigned int mw[4] = {mq.x, mq.y, mq.z, mq.w};
    float mr[8];
#pragma unroll
    for (int p = 0; p < 4; p++) {
        mr[2 * p] = __uint_as_float(mw[p] << 16);
        mr[2 * p + 1] = __uint_as_float(mw[p] & 0xFFFF0000u);
    }
    const float g[4] = {hcw_gate(gv.x), hcw_gate(gv.y), hcw_gate(gv.z), hcw_gate(gv.w)};
    if (bf16_state) {
        unsigned short* base = (unsigned short*)streams + (long long)t * 4 * H + c0;
#pragma unroll
        for (int br = 0; br < 4; br++) {
            const unsigned int in[4] = {sw[br].x, sw[br].y, sw[br].z, sw[br].w};
            unsigned int out[4];
#pragma unroll
            for (int p = 0; p < 4; p++) {
                const float lo = hcw_add(__uint_as_float(in[p] << 16), hcw_bf16f(hcw_mul(g[br], mr[2 * p])));
                const float hi = hcw_add(__uint_as_float(in[p] & 0xFFFF0000u), hcw_bf16f(hcw_mul(g[br], mr[2 * p + 1])));
                out[p] = hcw_bf16bits(lo) | (hcw_bf16bits(hi) << 16);
            }
            *(uint4*)(base + (long long)br * H) = make_uint4(out[0], out[1], out[2], out[3]);
        }
    } else {
        float* base = (float*)streams + (long long)t * 4 * H + c0;
#pragma unroll
        for (int br = 0; br < 4; br++) {
            const float v[8] = {sf[br][0].x, sf[br][0].y, sf[br][0].z, sf[br][0].w,
                                sf[br][1].x, sf[br][1].y, sf[br][1].z, sf[br][1].w};
            float o[8];
#pragma unroll
            for (int i = 0; i < 8; i++)
                o[i] = hcw_bf16f(hcw_add(hcw_bf16f(hcw_mul(g[br], mr[i])), hcw_bf16f(v[i])));
            *(float4*)(base + (long long)br * H) = make_float4(o[0], o[1], o[2], o[3]);
            *(float4*)(base + (long long)br * H + 4) = make_float4(o[4], o[5], o[6], o[7]);
        }
    }
}
#endif  // HCW_EPILOGUE_BLOCK


// ---------------------------------------------------------------------------
// Residual += BT with the X tile shared through LDS by RW row tiles.
// The BT kernel above re-reads the same 16*BV-token X tile from L2 in every
// 16-row wave; here RW waves (one 16-row tile each) stage each 64-wide K
// chunk of that tile into LDS once and all read it from there.  Every wave's
// A operand (decode_tile_split), B operand bytes, WMMA sequence and epilogue
// are the BT kernel's, so results are bitwise identical to it.
// Grid: [ceil(M/(16*RW)), ceil(N/(16*BV))]. Block: [32*RW].  X rows are `ldx`
// halves apart (>= K; see Gpu::f16_row_pitch).
// ---------------------------------------------------------------------------
template <int BITS, int BV, int RW, bool OVERWRITE, bool OUT_BF16 = false, int PREFETCH = 1,
          bool HC = false, bool DQ = false>
__device__ __forceinline__ void resid_bt_xlds(const char* __restrict__ A,
                                              const _Float16* __restrict__ X,
                                              float* __restrict__ Y,
                                              int M, int K, int N, int ldx,
                                              int row_block = -1,
                                              void* hc_streams = nullptr,
                                              const float* hc_gates = nullptr,
                                              int hc_bf16 = 0) {
    constexpr int GROUP_BYTES = 8 + 32 * BITS;
    constexpr int TOK = 16 * BV;
    constexpr int CK = 64;              // K per LDS chunk: four K-tiles
    constexpr int STRIDE = CK + 8;      // halfs per token row; 144 B spreads banks
    constexpr int NT = 32 * RW;
    constexpr int VPT = CK / 8;         // 16 B vectors per token per chunk
    constexpr int VECS = TOK * VPT / NT;
    // HC parking reuses xs for TOK x (16 * RW) BF16 channel halfs.  RW <= 4 parks at
    // the staging STRIDE (72 for RW4: the incumbent layout, unchanged).  RW8 needs 128
    // halfs per token, and the unpadded 16 * RW stride (no padding) is what keeps the
    // BV16/RW8 tile at exactly TOK * 128 * 2 B = 64 KB, the LDS limit.  Every HC
    // store and the stage-2 read use PARK_STRIDE, so the chain (WMMA -> permlanex16
    // swap -> hcw_pack8 -> LDS -> hcw_apply8) is the incumbent's, only re-addressed.
    constexpr int PARK_STRIDE = (RW <= 4) ? STRIDE : 16 * RW;
    constexpr int STAGE_HALFS = TOK * STRIDE;
    constexpr int PARK_HALFS = HC ? TOK * PARK_STRIDE : 0;
    constexpr int LDS_HALFS = STAGE_HALFS > PARK_HALFS ? STAGE_HALFS : PARK_HALFS;
    __shared__ __attribute__((aligned(16))) _Float16 xs[LDS_HALFS];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int row_start = ((row_block < 0 ? (int)blockIdx.x : row_block) * RW + (tid >> 5)) * 16;
    const int batch_start = blockIdx.y * TOK;
    const int safe_row = (row_start + ml < M) ? (row_start + ml) : (M - 1);
    const int groups_per_row = K / 256;
    static_assert(BITS == 6, "the chunk payload reader is written for 6-bit codes");
    // Row bytes through a buffer resource: the per-lane row offset rides in
    // voffset and the uniform group/chunk offset in soffset, so the payload
    // is six 8-byte loads per chunk instead of 24 byte loads with 64-bit
    // address arithmetic each.
    const __amdgpu_buffer_rsrc_t arsrc =
        __builtin_amdgcn_make_buffer_rsrc((void*)A, 0, 0x7FFFFFFF, 0x31004000);
    const int row_off = safe_row * groups_per_row * GROUP_BYTES;
    const int lane_half = lane >> 4;

    // Staging slots: consecutive threads copy consecutive 16 B of a token row.
    // Tokens past N read row 0, like the BT kernel's clamped sb[].
    // Slot i covers token i * (NT / VPT) + tid / VPT, so its LDS offset is
    // lds0 plus a compile-time constant.  Named scalars, not an array: the
    // array stays in scratch because SROA runs before the loops unroll.
    static_assert(VECS == 4 || VECS == 6 || VECS == 8 || VECS == 12 || VECS == 16);
    static_assert(PREFETCH == 1 || PREFETCH == 2);
    static_assert(LDS_HALFS * sizeof(_Float16) <= 65536);
#define XL_FOR8(F) F(0) F(1) F(2) F(3) F(4) F(5) F(6) F(7) F(8) F(9) F(10) F(11) F(12) F(13) F(14) F(15)
#define XL_OFF(I) \
    const int xo##I = ((batch_start + I * (NT / VPT) + tid / VPT < N) \
        ? (batch_start + I * (NT / VPT) + tid / VPT) : 0) * ldx + (tid % VPT) * 8;
    XL_FOR8(XL_OFF)
#undef XL_OFF
    _Float16* lds0 = xs + (tid / VPT) * STRIDE + (tid % VPT) * 8;
#define XL_LOAD(I) if constexpr (I < VECS) st##I = *(const uint4*)(X + xo##I + cn * CK);
#define XL_STORE(I) if constexpr (I < VECS) *(uint4*)(lds0 + I * (NT / VPT) * STRIDE) = st##I;
#define XL_NEXT(I) if constexpr (I < VECS && PREFETCH == 2) nx##I = *(const uint4*)(X + xo##I + cn * CK);
#define XL_ADVANCE(I) if constexpr (I < VECS && PREFETCH == 2) st##I = nx##I;
    // The barriers order LDS only: __syncthreads would also wait for the
    // in-flight next-chunk X and payload loads (WGP-mode workgroup fence).
#define XL_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
    uint4 st0, st1, st2, st3, st4, st5, st6, st7, st8, st9, st10, st11, st12, st13, st14, st15;
    uint4 nx0, nx1, nx2, nx3, nx4, nx5, nx6, nx7, nx8, nx9, nx10, nx11, nx12, nx13, nx14, nx15;
    {
        const int cn = 0;
        XL_FOR8(XL_LOAD)
    }
    if constexpr (PREFETCH == 2) {
        const int cn = (K > CK) ? 1 : 0;
        XL_FOR8(XL_NEXT)
    }

    float8_t acc[BV];
#pragma unroll
    for (int b = 0; b < BV; b++)
        acc[b] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int nchunks = K / CK;
    for (int g = 0; g < groups_per_row; g++) {
        const int goff = g * GROUP_BYTES;
        const unsigned int hA = __builtin_amdgcn_raw_buffer_load_b32(arsrc, row_off, goff, 0);
        const unsigned int hB = __builtin_amdgcn_raw_buffer_load_b32(arsrc, row_off + 4, goff, 0);
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu)));
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16)));
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu)));
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)));
        // DQ: raw half bits replicated into both 16-bit halves (v_perm_b32).
        unsigned int sc2_0 = 0, zp2_0 = 0, sc2_1 = 0, zp2_1 = 0;
        if constexpr (DQ) {
            sc2_0 = __builtin_amdgcn_perm(hA, hA, 0x01000100u);
            zp2_0 = __builtin_amdgcn_perm(hA, hA, 0x03020302u);
            sc2_1 = __builtin_amdgcn_perm(hB, hB, 0x01000100u);
            zp2_1 = __builtin_amdgcn_perm(hB, hB, 0x03020302u);
        }
#pragma unroll
        for (int cq = 0; cq < 4; cq++) {
            const int c = g * 4 + cq;
            // The chunk's four tiles: payload bytes cq * 48 .. +48 (8-aligned).
            unsigned int d[12];
#pragma unroll
            for (int u = 0; u < 6; u++) {
                const auto v = __builtin_amdgcn_raw_buffer_load_b64(
                    arsrc, row_off + 8 + u * 8, goff + cq * 48, 0);
                d[2 * u] = v[0];
                d[2 * u + 1] = v[1];
            }
            XL_FOR8(XL_STORE)
            XL_BARRIER();
            // Next chunk's global loads fly while this chunk computes (the
            // last chunk harmlessly reloads itself).
            if constexpr (PREFETCH == 2) {
                XL_FOR8(XL_ADVANCE)
                const int cn = (c + 2 < nchunks) ? (c + 2) : c;
                XL_FOR8(XL_NEXT)
            } else {
                const int cn = (c + 1 < nchunks) ? (c + 1) : c;
                XL_FOR8(XL_LOAD)
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int kt = cq * 4 + j;
                const _Float16 sc_h = (kt < 8) ? sc0 : sc1;
                const _Float16 zp_h = (kt < 8) ? zp0 : zp1;
                // Tile j is chunk bytes 12j .. 12j+11; this half's six start
                // at 12j + 6 * lane_half.
                const unsigned int lo = lane_half
                    ? __builtin_amdgcn_alignbit(d[3 * j + 2], d[3 * j + 1], 16) : d[3 * j];
                const unsigned int hi = lane_half ? (d[3 * j + 2] >> 16) : (d[3 * j + 1] & 0xFFFFu);
                half16_t a_reg;
                if constexpr (DQ) {
                    a_reg = decode6_tile_split_pk(((unsigned long long)hi << 32) | lo,
                                                  (kt < 8) ? sc2_0 : sc2_1,
                                                  (kt < 8) ? zp2_0 : zp2_1, lane_half);
                } else {
                    a_reg = decode6_tile_split(
                        ((unsigned long long)hi << 32) | lo, sc_h, zp_h, lane_half);
                }
#pragma unroll
                for (int b = 0; b < BV; b++) {
                    half16_t b_reg = *(const half16_t*)(xs + (b * 16 + ml) * STRIDE + j * 16);
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc[b]);
                }
            }
            // WMMA has no memory chain, so without this the DAG scheduler
            // sinks every chunk's WMMAs past the barriers and spills the
            // B operands that were read before them.
#pragma unroll
            for (int b = 0; b < BV; b++) asm volatile("" : "+v"(acc[b]));
            XL_BARRIER();
        }
    }
    // Lane l < 16 holds rows 2j of column l, lane l + 16 rows 2j + 1: one
    // cross-half swap gives the low lane rows 0..7 and the high lane rows
    // 8..15, stored as two 16-byte runs instead of eight scattered floats.
#pragma unroll
    for (int b = 0; b < BV; b++) {
        float r[8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float give = lane_half ? acc[b][i] : acc[b][4 + i];
            const float got = __int_as_float(__builtin_amdgcn_permlanex16(
                __float_as_int(give), __float_as_int(give), 0x76543210u, 0xfedcba98u, false, false));
            const float own = lane_half ? acc[b][4 + i] : acc[b][i];
            r[2 * i] = lane_half ? got : own;
            r[2 * i + 1] = lane_half ? own : got;
        }
        const int out_col = batch_start + b * 16 + ml;
        const int row0 = row_start + 8 * lane_half;
        if (out_col >= N || row0 >= M) continue;
        if (HC) {
            // M % 8 == 0 (caller): no partial row run.  Park the BF16 bits of
            // 0.0f + acc in LDS (xs: the staging buffer is dead after the last
            // chunk's closing barrier) at row stride PARK_STRIDE; `Y`, when
            // non-null, is the unfused kernel's own F32 output (debug/capture mirror).
            if (Y) hcw_mirror8(r, out_col, row0, M, Y);
            *(uint4*)(xs + (b * 16 + ml) * PARK_STRIDE + 16 * (tid >> 5) + 8 * lane_half) = hcw_pack8(r);
            continue;
        }
        if (OUT_BF16) {
            // Overwrite values rounded to BF16 (RNE; non-finite truncate), as
            // BF16 bits: what a consumer that rounds each value first reads.
            unsigned short* outb = (unsigned short*)Y + (long long)out_col * M + row0;
            unsigned short h[8];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const unsigned int u = __float_as_uint(0.0f + r[i]);
                h[i] = isfinite(0.0f + r[i]) ? (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16)
                                             : (unsigned short)(u >> 16);
            }
            if (row0 + 8 > M) {
                for (int i = 0; row0 + i < M; i++) outb[i] = h[i];
            } else {
                *(uint4*)outb = __builtin_bit_cast(uint4, h);
            }
            continue;
        }
        float* out = Y + (long long)out_col * M + row0;
        // Overwrite is the memset-then-accumulate result: +0.0 plus the sum
        // (not folded: IEEE keeps +0.0 + -0.0 = +0.0).
        if (row0 + 8 > M) {
            for (int i = 0; row0 + i < M; i++) out[i] = (OVERWRITE ? 0.0f : out[i]) + r[i];
        } else {
            float4 o0 = OVERWRITE ? make_float4(0.0f, 0.0f, 0.0f, 0.0f) : *(const float4*)out;
            float4 o1 = OVERWRITE ? make_float4(0.0f, 0.0f, 0.0f, 0.0f) : *(const float4*)(out + 4);
            o0.x += r[0]; o0.y += r[1]; o0.z += r[2]; o0.w += r[3];
            o1.x += r[4]; o1.y += r[5]; o1.z += r[6]; o1.w += r[7];
            *(float4*)out = o0;
            *(float4*)(out + 4) = o1;
        }
    }
    if (HC) {
        XL_BARRIER();
        // Stage 2: TOK tokens x (16 * RW) channels of this block, 8 channels per item.
        constexpr int CPT = 2 * RW;  // 8-channel chunks per token
#pragma unroll 2
        for (int it = 0; it < TOK * CPT / NT; it++) {
            const int item = tid + it * NT;
            const int tl = item / CPT;
            const int c8 = item % CPT;
            const int t = batch_start + tl;
            const int c0 = (row_block < 0 ? (int)blockIdx.x : row_block) * RW * 16 + c8 * 8;
            if (t >= N || c0 >= M) continue;
            hcw_apply8(*(const uint4*)(xs + tl * PARK_STRIDE + c8 * 8), t, c0, M, hc_streams, hc_gates,
                       hc_bf16);
        }
    }
#undef XL_STORE
#undef XL_NEXT
#undef XL_ADVANCE
#undef XL_BARRIER
#undef XL_LOAD
#undef XL_FOR8
}

extern "C" __launch_bounds__(128, 1) __global__ void
gemm_mq6g256v2_residual_wmma_gfx11_bt8_x4(const char* __restrict__ A,
                                          const _Float16* __restrict__ X,
                                          float* __restrict__ Y,
                                          int M, int K, int N, int ldx) {
    resid_bt_xlds<6, 8, 4, false>(A, X, Y, M, K, N, ldx);
}

// Y = W·X (the memset + residual pair in one launch).
extern "C" __launch_bounds__(128, 1) __global__ void
gemm_mq6g256v2_wmma_gfx11_bt8_x4(const char* __restrict__ A,
                                 const _Float16* __restrict__ X,
                                 float* __restrict__ Y,
                                 int M, int K, int N, int ldx) {
    resid_bt_xlds<6, 8, 4, true>(A, X, Y, M, K, N, ldx);
}

// Y = W·X stored as BF16 bits (Y is [N, M] BF16).
extern "C" __launch_bounds__(128, 1) __global__ void
gemm_mq6g256v2_wmma_gfx11_bt8_x4_bf16out(const char* __restrict__ A,
                                         const _Float16* __restrict__ X,
                                         float* __restrict__ Y,
                                         int M, int K, int N, int ldx) {
    resid_bt_xlds<6, 8, 4, true, true>(A, X, Y, M, K, N, ldx);
}

// The x4 kernels with eight row tiles (128 rows) per workgroup: each X chunk
// is fetched once for twice the rows (prefill-sized N).  Bitwise the x4 ones.
extern "C" __launch_bounds__(256, 1) __global__ void
gemm_mq6g256v2_residual_wmma_gfx11_bt8_x8(const char* __restrict__ A,
                                          const _Float16* __restrict__ X,
                                          float* __restrict__ Y,
                                          int M, int K, int N, int ldx) {
    resid_bt_xlds<6, 8, 8, false>(A, X, Y, M, K, N, ldx);
}

extern "C" __launch_bounds__(256, 1) __global__ void
gemm_mq6g256v2_wmma_gfx11_bt8_x8(const char* __restrict__ A,
                                 const _Float16* __restrict__ X,
                                 float* __restrict__ Y,
                                 int M, int K, int N, int ldx) {
    resid_bt_xlds<6, 8, 8, true>(A, X, Y, M, K, N, ldx);
}

extern "C" __launch_bounds__(256, 1) __global__ void
gemm_mq6g256v2_wmma_gfx11_bt8_x8_bf16out(const char* __restrict__ A,
                                         const _Float16* __restrict__ X,
                                         float* __restrict__ Y,
                                         int M, int K, int N, int ldx) {
    resid_bt_xlds<6, 8, 8, true, true>(A, X, Y, M, K, N, ldx);
}

// U3 tiles: the gfx1151 default table (HIPFIRE_QWEN4_MQ6_X4_TILE) picks among these.
#define U3_TILE(BV, RW, PF) \
extern "C" __launch_bounds__(32 * RW, 1) __global__ void \
gemm_mq6g256v2_wmma_gfx11_u3_b##BV##_r##RW##_p##PF( \
    const char* A, const _Float16* X, float* Y, int M, int K, int N, int ldx) { \
    if (M <= 0 || K <= 0 || N <= 0) return; \
    resid_bt_xlds<6, BV, RW, true, false, PF>(A, X, Y, M, K, N, ldx); \
} \
extern "C" __launch_bounds__(32 * RW, 1) __global__ void \
gemm_mq6g256v2_wmma_gfx11_u3_b##BV##_r##RW##_p##PF##_bf16out( \
    const char* A, const _Float16* X, float* Y, int M, int K, int N, int ldx) { \
    if (M <= 0 || K <= 0 || N <= 0) return; \
    resid_bt_xlds<6, BV, RW, true, true, PF>(A, X, Y, M, K, N, ldx); \
}
U3_TILE(8, 4, 1)
U3_TILE(8, 4, 2)
U3_TILE(8, 8, 1)
U3_TILE(8, 8, 2)
U3_TILE(12, 4, 1)
U3_TILE(12, 4, 2)
U3_TILE(12, 8, 1)
U3_TILE(12, 8, 2)
// BV16 U3 tiles (TOK = 256 tokens; VECS = 16 for RW4, 8 for RW8).
U3_TILE(16, 4, 1)
U3_TILE(16, 4, 2)
U3_TILE(16, 8, 1)
U3_TILE(16, 8, 2)
#undef U3_TILE

// Tile-twin multi-region and HC-fused entries, generated separately from U3_TILE
// (packed X: row pitch K)
// so the incumbent U3 wrappers and the *_bt8_x4_{regions,hcw} entries stay byte-
// identical.  `_regions`: the incumbent three-region ABI with r_i = ceil(M_i / (16*RW))
// row blocks per region (RW 16-row tiles per block) and the same dispatch /
// local_rb pattern.  `_hcw`: the incumbent HC ABI (M % 8 == 0), body
// resid_bt_xlds<6, BV, RW, true, false, PF, true>.
#define U3_TWIN_TILE(BV, RW, PF) \
extern "C" __launch_bounds__(32 * RW, 1) __global__ void \
gemm_mq6g256v2_wmma_gfx11_u3_b##BV##_r##RW##_p##PF##_regions( \
    const char* A0, const char* A1, const char* A2, const _Float16* X, \
    float* Y0, float* Y1, float* Y2, int M0, int M1, int M2, int K, int N) { \
    if (N <= 0 || K <= 0) return; \
    const int r0 = (M0 + 16 * RW - 1) / (16 * RW); \
    const int r1 = (M1 + 16 * RW - 1) / (16 * RW); \
    const int rb = blockIdx.x; \
    const char* A; \
    float* Y; \
    int M, local_rb; \
    if (rb < r0) { \
        A = A0; Y = Y0; M = M0; local_rb = rb; \
    } else if (rb < r0 + r1) { \
        A = A1; Y = Y1; M = M1; local_rb = rb - r0; \
    } else { \
        A = A2; Y = Y2; M = M2; local_rb = rb - r0 - r1; \
    } \
    if (M <= 0 || local_rb >= (M + 16 * RW - 1) / (16 * RW)) return; \
    resid_bt_xlds<6, BV, RW, true, false, PF>(A, X, Y, M, K, N, K, local_rb); \
} \
extern "C" __launch_bounds__(32 * RW, 1) __global__ void \
gemm_mq6g256v2_wmma_gfx11_u3_b##BV##_r##RW##_p##PF##_hcw( \
    const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, \
    int M, int K, int N, \
    void* hc_streams, const float* hc_gates, int hc_bf16) { \
    resid_bt_xlds<6, BV, RW, true, false, PF, true>(A, X, Y, M, K, N, K, -1, hc_streams, \
                                                     hc_gates, hc_bf16); \
}
U3_TWIN_TILE(8, 4, 1)
U3_TWIN_TILE(8, 4, 2)
U3_TWIN_TILE(8, 8, 1)
U3_TWIN_TILE(8, 8, 2)
U3_TWIN_TILE(16, 4, 1)
U3_TWIN_TILE(16, 4, 2)
U3_TWIN_TILE(16, 8, 1)
U3_TWIN_TILE(16, 8, 2)
#undef U3_TWIN_TILE

extern "C" __launch_bounds__(128, 1) __global__ void
gemm_mq6g256v2_wmma_gfx11_bt8_x4_regions(
    const char* A0, const char* A1, const char* A2, const _Float16* X,
    float* Y0, float* Y1, float* Y2, int M0, int M1, int M2, int K, int N, int ldx) {
    if (N <= 0 || K <= 0) return;
    const int r0 = (M0 + 63) / 64;
    const int r1 = (M1 + 63) / 64;
    const int rb = blockIdx.x;
    const char* A;
    float* Y;
    int M, local_rb;
    if (rb < r0) {
        A = A0; Y = Y0; M = M0; local_rb = rb;
    } else if (rb < r0 + r1) {
        A = A1; Y = Y1; M = M1; local_rb = rb - r0;
    } else {
        A = A2; Y = Y2; M = M2; local_rb = rb - r0 - r1;
    }
    if (M <= 0 || local_rb >= (M + 63) / 64) return;
    resid_bt_xlds<6, 8, 4, true>(A, X, Y, M, K, N, ldx, local_rb);
}

// DQ entries: the U3 b8 tiles / regions / hcw paths with the packed-pair dequant
// (resid_bt_xlds DQ = true).  Incumbent entries are unchanged; this block stays
// before the final incumbent kernel (gemm_..._bt8_x4_hcw) so that kernel's code is
// not perturbed by appended entries.
#define U3_DQ_TILE(BV, RW, PF) \
extern "C" __launch_bounds__(32 * RW, 1) __global__ void \
gemm_mq6g256v2_wmma_gfx11_u3_b##BV##_r##RW##_p##PF##_dq( \
    const char* A, const _Float16* X, float* Y, int M, int K, int N) { \
    if (M <= 0 || K <= 0 || N <= 0) return; \
    resid_bt_xlds<6, BV, RW, true, false, PF, false, true>(A, X, Y, M, K, N, K); \
} \
extern "C" __launch_bounds__(32 * RW, 1) __global__ void \
gemm_mq6g256v2_wmma_gfx11_u3_b##BV##_r##RW##_p##PF##_dq_bf16out( \
    const char* A, const _Float16* X, float* Y, int M, int K, int N) { \
    if (M <= 0 || K <= 0 || N <= 0) return; \
    resid_bt_xlds<6, BV, RW, true, true, PF, false, true>(A, X, Y, M, K, N, K); \
}
U3_DQ_TILE(8, 4, 1)
U3_DQ_TILE(8, 4, 2)
U3_DQ_TILE(8, 8, 1)
U3_DQ_TILE(8, 8, 2)
#undef U3_DQ_TILE

extern "C" __launch_bounds__(128, 1) __global__ void
gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p1_dq_regions(
    const char* A0, const char* A1, const char* A2, const _Float16* X,
    float* Y0, float* Y1, float* Y2, int M0, int M1, int M2, int K, int N) {
    if (N <= 0 || K <= 0) return;
    const int r0 = (M0 + 63) / 64;
    const int r1 = (M1 + 63) / 64;
    const int rb = blockIdx.x;
    const char* A;
    float* Y;
    int M, local_rb;
    if (rb < r0) {
        A = A0; Y = Y0; M = M0; local_rb = rb;
    } else if (rb < r0 + r1) {
        A = A1; Y = Y1; M = M1; local_rb = rb - r0;
    } else {
        A = A2; Y = Y2; M = M2; local_rb = rb - r0 - r1;
    }
    if (M <= 0 || local_rb >= (M + 63) / 64) return;
    resid_bt_xlds<6, 8, 4, true, false, 1, false, true>(A, X, Y, M, K, N, K, local_rb);
}

extern "C" __launch_bounds__(128, 1) __global__ void
gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p1_dq_hcw(const char* __restrict__ A,
                                             const _Float16* __restrict__ X,
                                             float* __restrict__ Y,
                                             int M, int K, int N,
                                             void* hc_streams, const float* hc_gates, int hc_bf16) {
    if (M <= 0 || K <= 0 || N <= 0) return;
    resid_bt_xlds<6, 8, 4, true, false, 1, true, true>(A, X, Y, M, K, N, K, -1, hc_streams, hc_gates,
                                                      hc_bf16);
}

// Y = W·X applied to the HC streams in place: the unfused kernel's `0.0f + acc`
// feeds hyper_write_{bf16x2,f32} with the paired read's gate logits.  M % 8 == 0.
// `Y` is an optional F32 mirror of the GEMM output (may be null).
extern "C" __launch_bounds__(128, 1) __global__ void
gemm_mq6g256v2_wmma_gfx11_bt8_x4_hcw(const char* __restrict__ A,
                                     const _Float16* __restrict__ X,
                                     float* __restrict__ Y,
                                     int M, int K, int N,
                                     void* hc_streams, const float* hc_gates, int hc_bf16) {
    resid_bt_xlds<6, 8, 4, true, false, 1, true>(A, X, Y, M, K, N, K, -1, hc_streams, hc_gates,
                                                hc_bf16);
}

#endif // __GFX11__
