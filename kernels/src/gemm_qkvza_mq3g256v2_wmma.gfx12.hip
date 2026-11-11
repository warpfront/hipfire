// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) variant of the MQ3 / MQ3G256V2-G256 4-way fused qkvza WMMA
// preamble. Port of:
//   - gemm_qkvza_hfq3g256_wmma.hip (gfx11 MQ3 — 3-bit cross-byte unpack)
//   - gemm_qkvza_hfq4g256_wmma.gfx12.hip (gfx12 MQ4 K4 + half8_t pattern)
//
// gfx12-specific changes vs the gfx11 MQ3 source:
//   1. _w32 → _w32_gfx12 builtin
//   2. half16_t A/B → half8_t (each lane carries 8 of 16 K values)
//   3. K-split via tid >> 4 (k_grp picks which 8 of 16 K values)
//   4. C-output mapping: contiguous rows per lane-group:
//        acc[j] = C[8 * (tid >> 4) + j][tid & 15]
//   5. K4 unroll: 4 K-tiles per body, all loads up-front
//
// MQ3G256V2 storage layout (per 256-element group, 104 B total):
//   [0..4)  fp16 s0/z0 (w 0..127), [4..8) fp16 s1/z1 (w128..255)
//   [8..104] 96 B of packed 3-bit weights, 32 chunks of 8 weights × 3 B
//
// For gfx12 each lane reads ONE chunk (8 weights = 3 bytes) per K-tile
// vs 2 chunks (16 weights = 6 bytes) on gfx11. K-tile kt + lane-group
// k_grp covers chunk (2*kt + k_grp), at byte offset 8 + kt*6 + k_grp*3.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

// ---- qt49 (MQ3G256V2) split-K group decode ---------------------------------
// One 104 B group = fp16 {sc0,zp0,sc1,zp1} + 96 B of 3-bit codes. Rows are
// 104 B apart, so every group is 8-byte aligned: a lane loads the whole group
// as 13 u64 instead of 48 single-byte loads. Chunk kt (16 K per wave-wide
// WMMA step) owns 6 bytes = two 24-bit halves, one per k_grp lane half.
#define MQ3_KS 8  // waves per block; each wave owns every KS-th group of K
typedef _Float16 __attribute__((ext_vector_type(2))) half2_t;

__device__ __forceinline__ half2_t mq3_codes2(unsigned int c24, int shift, half2_t sc2, half2_t zp2) {
    unsigned int p = ((c24 >> shift) & 7u) | (((c24 >> (shift + 3)) & 7u) << 16) | 0x64006400u;
    half2_t v = __builtin_bit_cast(half2_t, p) - (half2_t){(_Float16)1024.0f, (_Float16)1024.0f};
    return __builtin_elementwise_fma(sc2, v, zp2);
}

// Dequantize chunk kt (24 bits of this lane's k_grp) into a WMMA A fragment.
__device__ __forceinline__ half8_t mq3_chunk(unsigned int c24, half2_t sc2, half2_t zp2) {
    half2_t p0 = mq3_codes2(c24, 0, sc2, zp2);
    half2_t p1 = mq3_codes2(c24, 6, sc2, zp2);
    half2_t p2 = mq3_codes2(c24, 12, sc2, zp2);
    half2_t p3 = mq3_codes2(c24, 18, sc2, zp2);
    half8_t r;
    r[0] = p0[0]; r[1] = p0[1]; r[2] = p1[0]; r[3] = p1[1];
    r[4] = p2[0]; r[5] = p2[1]; r[6] = p3[0]; r[7] = p3[1];
    return r;
}

// 24-bit code word of chunk kt for k_grp, from the 12 data u64 (d[0..11]).
#define MQ3_C24(d, kt, k_grp) ({                                              \
    const int _s = (kt) * 48, _wi = _s >> 6, _sh = _s & 63;                   \
    unsigned long long _b = (d)[_wi] >> _sh;                                  \
    if (_sh > 16) _b |= (d)[_wi + 1] << (64 - _sh);                           \
    (unsigned int)(_b >> ((k_grp) * 24)) & 0xFFFFFFu; })

// Group header -> per-half scale/zero pairs (first 8 K-chunks use half 0).
#define MQ3_HDR(w0, sc0, zp0, sc1, zp1) do {                                  \
    unsigned int _hA = (unsigned int)(w0), _hB = (unsigned int)((w0) >> 32);  \
    sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(_hA & 0xFFFFu))); \
    zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(_hA >> 16)));     \
    sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(_hB & 0xFFFFu))); \
    zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(_hB >> 16)));     \
} while (0)

__launch_bounds__(32 * MQ3_KS)
extern "C" __global__ void gemm_qkvza_mq3g256v2_wmma_gfx12(
    const char*    __restrict__ A_qkv,
    const char*    __restrict__ A_z,
    const char*    __restrict__ A_beta,
    const char*    __restrict__ A_alpha,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y_qkv,
    float*         __restrict__ Y_z,
    float*         __restrict__ Y_beta,
    float*         __restrict__ Y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K, int N
) {
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x & 31;
    const int wv = threadIdx.x >> 5;  // split-K wave (block = 32 * MQ3_KS threads)

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int my_row = row_start + m_lane;
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    const char* A;
    int local_row;
    if (safe_row < qkv_m) {
        A = A_qkv;   local_row = safe_row;
    } else if (safe_row < qkv_m + z_m) {
        A = A_z;     local_row = safe_row - qkv_m;
    } else if (safe_row < qkv_m + z_m + beta_m) {
        A = A_beta;  local_row = safe_row - (qkv_m + z_m);
    } else {
        A = A_alpha; local_row = safe_row - (qkv_m + z_m + beta_m);
    }

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 104;
    const char* row_base = A + local_row * row_stride;

    const int safe_batch = (batch_start + m_lane < N) ? (batch_start + m_lane) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Split-K: wave `wv` of the block owns groups wv, wv+MQ3_KS, ...; wave 0
    // sums the partials through LDS before the epilogue.
    for (int g = wv; g < groups_per_row; g += MQ3_KS) {
        const unsigned long long* gp = (const unsigned long long*)(row_base + (long long)g * 104);
        unsigned long long w[13];
        #pragma unroll
        for (int i = 0; i < 13; i++) w[i] = gp[i];
        _Float16 sc0, zp0, sc1, zp1;
        MQ3_HDR(w[0], sc0, zp0, sc1, zp1);
        const half2_t scv0 = {sc0, sc0}, zpv0 = {zp0, zp0};
        const half2_t scv1 = {sc1, sc1}, zpv1 = {zp1, zp1};
        const unsigned long long* d = w + 1;
        const _Float16* xg = X + (long long)safe_batch * K + g * 256;

        #pragma unroll
        for (int kt = 0; kt < 16; kt++) {
            half8_t b = *(const half8_t*)(xg + kt * 16 + k_grp * 8);
            half8_t a_reg = (kt < 8) ? mq3_chunk(MQ3_C24(d, kt, k_grp), scv0, zpv0)
                                     : mq3_chunk(MQ3_C24(d, kt, k_grp), scv1, zpv1);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b, acc);
        }
    }

    __shared__ float red[MQ3_KS - 1][32][8];
    if (wv > 0) {
        #pragma unroll
        for (int j = 0; j < 8; j++) red[wv - 1][tid][j] = acc[j];
    }
    __syncthreads();
    if (wv != 0) return;
    #pragma unroll
    for (int w2 = 0; w2 < MQ3_KS - 1; w2++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[j] += red[w2][tid][j];
    }

    // gfx12 wave32 WMMA C-mapping: contiguous rows per lane-group.
    const int out_col = batch_start + m_lane;
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 8 * k_grp + j;
            if (out_row < total_m) {
                float* Y;
                int proj_row;
                int out_stride;
                if (out_row < qkv_m) {
                    Y = Y_qkv;   proj_row = out_row;                            out_stride = qkv_m;
                } else if (out_row < qkv_m + z_m) {
                    Y = Y_z;     proj_row = out_row - qkv_m;                    out_stride = z_m;
                } else if (out_row < qkv_m + z_m + beta_m) {
                    Y = Y_beta;  proj_row = out_row - (qkv_m + z_m);            out_stride = beta_m;
                } else {
                    Y = Y_alpha; proj_row = out_row - (qkv_m + z_m + beta_m);   out_stride = alpha_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
