// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Default for eligible exact-gfx1100 eager HIP launches; capture keeps base.
//
// gfx1100 (RDNA3) RAW-slab LDS-stage gate+up for MQ4G256V2 (qt=44), DFlash
// N<=16 tier. Composes:
//   - residual authority `gemm_mq4g256v2_residual_wmma_gfx1100_ldsstage`
//     (packed DEQUANT_A_FRAG_PK consume, half16 WMMA, dual barriers, rolled
//     4-fragment loop, partials[8*32*8] + wave-0 reduce, gfx11 interleaved C)
//   - gate_up split geometry from `gemm_gate_up_mq4g256v2_wmma_gfx12_ldsstage`
//     / base gate_up (A_gate|A_up row routing on load, overwrite Y_gate|Y_up)
//
// Cooperative slab fill (all 256 threads): one 16-row x 512-K RAW quantized
// slab into `staged_weights[16][272]` = 4352 B via coalesced dwordx4 loads
// (`load_row = tid/17`, `load_vec = tid%17`) plus a 16-thread tail for row 15.
// Each cooperative load row is routed individually to A_gate or A_up from the
// CLAMPED global row (including the row-15 tail — a mid-tile gate|up boundary
// may straddle the 16-row tile). 8-wave K partition (wave = 64-wide K slice of
// the 512 slab: `group_in_slab = wave>>2`, `quarter = wave&3`, header
// `quarter<2 ? hA : hB`). Requires K % 512 == 0 (launcher-enforced).
//
// Symbol: gemm_gate_up_mq4g256v2_wmma_gfx1100_ldsstage
// ABI (9): A_gate, A_up, X, Y_gate, Y_up, gate_m, up_m, K, N
// Grid:  [ceil((gate_m+up_m)/16), ceil(N/16), 1]
// Block: [256, 1, 1]. Static LDS 4352 + 8192 = 12544 B; dynamic shared_mem=0.
//
// K-reorder (disjoint 64-wide slices per wave, fixed wave-0..7 LDS reduction)
// => NOT bit-exact vs packed base; parity gate relL2 <= 5e-5. Same candidate
// repeated copies must be bit-exact (deterministic fixed reduction, no
// atomics). OVERWRITE Y (never +=). Compile with
// `hipcc --offload-arch=gfx1100`.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef unsigned int __attribute__((ext_vector_type(4))) uint4v_t;

#if defined(__gfx1100__)
// Exact nibble integers 0..15: RTZ == RN bit-for-bit (exact in f16 and f32).
// Packed idiom matches gemm_mq4g256v2_residual_wmma_gfx1100_ldsstage.hip
// (DEQUANT_A_FRAG_PK / DEQUANT_PK_H2 / pkrtz_nibble2).
__device__ __forceinline__ __half2 pkrtz_nibble2(float a, float b) {
    const auto v = __builtin_amdgcn_cvt_pkrtz(a, b);
    static_assert(sizeof(v) == sizeof(__half2), "pkrtz size");
    __half2 h2;
    __builtin_memcpy(&h2, &v, sizeof(__half2));
    return h2;
}

#define DEQUANT_PK_H2(pk, sc_h, zp_h, h01, h23, h45, h67)                                         \
    do {                                                                                          \
        const __half2 sc2 = __half2half2(sc_h);                                                   \
        const __half2 zp2 = __half2half2(zp_h);                                                   \
        const __half2 q01 = pkrtz_nibble2((float)(((pk) >> 0) & 0xFu),                            \
                                          (float)(((pk) >> 4) & 0xFu));                           \
        const __half2 q23 = pkrtz_nibble2((float)(((pk) >> 8) & 0xFu),                            \
                                          (float)(((pk) >> 12) & 0xFu));                          \
        const __half2 q45 = pkrtz_nibble2((float)(((pk) >> 16) & 0xFu),                           \
                                          (float)(((pk) >> 20) & 0xFu));                          \
        const __half2 q67 = pkrtz_nibble2((float)(((pk) >> 24) & 0xFu),                           \
                                          (float)(((pk) >> 28) & 0xFu));                          \
        (h01) = __hfma2(q01, sc2, zp2);                                                           \
        (h23) = __hfma2(q23, sc2, zp2);                                                           \
        (h45) = __hfma2(q45, sc2, zp2);                                                           \
        (h67) = __hfma2(q67, sc2, zp2);                                                           \
    } while (0)

#define DEQUANT_A_FRAG_PK(pk0, pk1, sc_h, zp_h, frag)                                             \
    do {                                                                                          \
        __half2 _h[8];                                                                            \
        DEQUANT_PK_H2((pk0), (sc_h), (zp_h), _h[0], _h[1], _h[2], _h[3]);                        \
        DEQUANT_PK_H2((pk1), (sc_h), (zp_h), _h[4], _h[5], _h[6], _h[7]);                        \
        static_assert(sizeof(_h) == sizeof(frag), "DEQUANT_A_FRAG_PK size mismatch");             \
        __builtin_memcpy(&(frag), _h, sizeof(frag));                                              \
    } while (0)
#endif

__launch_bounds__(256, 4)
extern "C" __global__ void gemm_gate_up_mq4g256v2_wmma_gfx1100_ldsstage(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const _Float16* __restrict__ X,
    float* __restrict__ Y_gate,
    float* __restrict__ Y_up,
    int gate_m, int up_m,
    int K, int N
) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int total_m = gate_m + up_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;

    // Uniform block-only early return (blockIdx is wave-uniform). No per-lane
    // returns around barriers.
    if (row_start >= total_m || batch_start >= N) return;

    const int m_lane = lane & 15;
    // Invalid N lanes: clamp/pad X from batch 0 (existing residual/base pattern).
    const int safe_batch = (batch_start + m_lane < N)
        ? (batch_start + m_lane) : 0;
    const _Float16* x_base = X + (long long)safe_batch * K;
    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const int slabs = K / 512;

    // Two packed G256 groups per row, 16 rows: 16 * 272 = 4352 B.
    // The logical 272-vector flat fill uses dwordx4 loads. Vectors 0..254
    // cover rows 0..14, vector 255 starts row 15, and lanes 0..15 load its
    // remaining 16 vectors. Within each row, consecutive threads access
    // consecutive 16-byte addresses.
    __shared__ __align__(16) unsigned char staged_weights[16][272];
    __shared__ float partials[8 * 32 * 8];

    const int load_row = tid / 17;
    const int load_vec = tid - load_row * 17;
    // Route each cooperative load row individually from the CLAMPED global
    // row — do not assume the whole 16-row tile stays on one of A_gate/A_up.
    const int safe_load_row = (row_start + load_row < total_m)
        ? (row_start + load_row) : (total_m - 1);
    const char* A_load;
    int local_load_row;
    if (safe_load_row < gate_m) {
        A_load = A_gate;  local_load_row = safe_load_row;
    } else {
        A_load = A_up;    local_load_row = safe_load_row - gate_m;
    }

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int slab = 0; slab < slabs; ++slab) {
        const char* src0 = A_load + (long long)local_load_row * row_stride
            + slab * 272 + load_vec * 16;
        unsigned char* dst0 = staged_weights[load_row] + load_vec * 16;
        *(uint4v_t*)dst0 = *(const uint4v_t*)src0;

        // Row-15 tail: separate clamp + gate/up route (may differ from the
        // primary load_row path when the tile straddles gate_m).
        if (tid < 16) {
            const int vec = tid + 1;
            const int safe_row15 = (row_start + 15 < total_m)
                ? (row_start + 15) : (total_m - 1);
            const char* A15;
            int local_row15;
            if (safe_row15 < gate_m) {
                A15 = A_gate;  local_row15 = safe_row15;
            } else {
                A15 = A_up;    local_row15 = safe_row15 - gate_m;
            }
            const char* src1 = A15 + (long long)local_row15 * row_stride
                + slab * 272 + vec * 16;
            unsigned char* dst1 = staged_weights[15] + vec * 16;
            *(uint4v_t*)dst1 = *(const uint4v_t*)src1;
        }
        __syncthreads();

        const int group_in_slab = wave >> 2;
        const int quarter_in_group = wave & 3;
        const unsigned char* gp = staged_weights[m_lane] + group_in_slab * 136;
        // quarter 0..1 -> weights 0..127 (half0); quarter 2..3 -> 128..255 (half1).
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const unsigned int hs = (quarter_in_group < 2) ? hA : hB;
        const _Float16 sc_h =
            (_Float16)__half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        const _Float16 zp_h =
            (_Float16)__half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        const _Float16* xg = x_base + slab * 512 + wave * 64;
        const int frag_base = quarter_in_group * 4;

        // Deliberately NOT unrolled (`unroll 1`): a rolled 4-iteration loop
        // keeps exactly one fragment's pk0/pk1/a_reg/b_reg live at a time
        // (plus the loop-carried float8 acc and sc_h/zp_h scalars). A fully
        // unrolled body holds 4 fragments' temporaries simultaneously and
        // costs 126 VGPRs -> 1 WG/CU; the rolled form targets <= 64 VGPRs
        // (2 WGs/CU). A 4-trip loop is negligible at this shape.
        #pragma unroll 1
        for (int f = 0; f < 4; ++f) {
            // Global 16-wide fragment within the group; 8 bytes = 2 u32.
            const int frag = frag_base + f;
            const unsigned int pk0 = *(const unsigned int*)(gp + 8 + frag * 8);
            const unsigned int pk1 = *(const unsigned int*)(gp + 8 + frag * 8 + 4);

            // Packed RTZ + __hfma2 fragment (residual ldsstage idiom); scalar
            // fallback preserves the prior base-kernel DQ path off gfx1100.
            half16_t a_reg;
#if defined(__gfx1100__)
            DEQUANT_A_FRAG_PK(pk0, pk1, (__half)sc_h, (__half)zp_h, a_reg);
#else
            #define DQ(i, pk, sh) a_reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(0,  pk0,  0); DQ(1,  pk0,  4); DQ(2,  pk0,  8); DQ(3,  pk0, 12);
            DQ(4,  pk0, 16); DQ(5,  pk0, 20); DQ(6,  pk0, 24); DQ(7,  pk0, 28);
            DQ(8,  pk1,  0); DQ(9,  pk1,  4); DQ(10, pk1,  8); DQ(11, pk1, 12);
            DQ(12, pk1, 16); DQ(13, pk1, 20); DQ(14, pk1, 24); DQ(15, pk1, 28);
            #undef DQ
#endif

            half16_t b_reg = *(const half16_t*)(xg + f * 16);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }

        // All waves must finish consuming the slab before the cooperative fill
        // overwrites it on the next iteration.
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        partials[(wave * 32 + lane) * 8 + j] = acc[j];
    }
    __syncthreads();

    // Wave 0 deterministic fixed-order reduction with gate_up OVERWRITE
    // semantics (y = sum, never +=). gfx11 interleaved C mapping:
    // acc[j] = C[2*j + (lane>>4)][lane & 15].
    if (wave == 0) {
        const int out_col = batch_start + m_lane;
        if (out_col < N) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float sum = partials[(0 * 32 + lane) * 8 + j];
                #pragma unroll
                for (int w = 1; w < 8; ++w) {
                    sum += partials[(w * 32 + lane) * 8 + j];
                }
                const int out_row = row_start + 2 * j + (lane >> 4);
                if (out_row < total_m) {
                    float* Y;
                    int proj_row;
                    int out_stride;
                    if (out_row < gate_m) {
                        Y = Y_gate;  proj_row = out_row;           out_stride = gate_m;
                    } else {
                        Y = Y_up;    proj_row = out_row - gate_m;  out_stride = up_m;
                    }
                    Y[(long long)out_col * out_stride + proj_row] = sum;
                }
            }
        }
    }
}

#if defined(__gfx1100__)
#undef DEQUANT_PK_H2
#undef DEQUANT_A_FRAG_PK
#endif
