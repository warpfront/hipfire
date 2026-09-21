// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Integer-A8 (iu8) MMQ residual GEMM port for gfx12 (RDNA4 — R9700/gfx1201).
//
// This is the gfx12 sister of `gemm_mq4g256v2_residual_mmq.hip` (RDNA3 LDS-tile
// MMQ, W int4 expanded to int8 x Q8_1 int8 activations on
// `wmma_i32_16x16x16_iu8`). It shares the same dispatch contract / 7-arg
// signature:
//   - A:  MQ4G256V2 packed weights [M rows x groups_per_row x 136 bytes]
//         (per 256-K group: TWO dual-fp16 headers — (sc0, zp0) for K 0..127
//         at gp+0, (sc1, zp1) for K 128..255 at gp+4 — then 128 bytes of
//         nibble-packed codes at gp+8, i.e. the same 136 B geometry as HFQ4).
//   - Xq: activations pre-quantized to Q8_1 (block_q8_1_mmq, per-32) via
//         `quantize_q8_1_mmq_ds4`, layout [K/128 x N] => Xq[kb_q8 * N + col].
//   - Y:  output [N x M] column-major: Y[col * M + row].
//   - add: 1 => Y[..] += result (residual / _full_add); 0 => Y[..] = result.
//
// The compute body is ported from `gemm_hfq4g256_residual_mmq.gfx12.hip`
// (same single-wave 16-row tile, same Q8_1 correction math); only the weight
// metadata differs: one (sc, zp) header per 128-K half instead of one f32
// (sc, zp) pair per 256-K group. Per output element:
//   out += sc_half * d_x * dot(q4, q8) + zp_half * sum_x
// which matches the RDNA3 `vec_dot_q8_1_q8_1_mma` convention
// (dmA.x * dsB.x * C + dmA.y * dsB.y per 32-K sub-block dot).
//
// Weight nibbles (0..15) feed the A operand unsigned (signed_a = false);
// Q8_1 activations (-127..127) feed B signed (signed_b = true).
//
// What this kernel lacks vs the iu4 twins
// (`gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip`):
//   - no symfold twin (`IU4_SYMMETRIC_FOLD` has no iu8 counterpart — symmetric
//     weights run the uniform path here, correct but without the folded
//     zero-point LDS saving);
//   - K32 WMMA (iu4 uses `wmma_i32_16x16x32_iu4`): this kernel issues K16
//     `wmma_i32_16x16x16_iu8` (halved operand width: int32x2 A/B per lane);
//   - no X128 variant (per-128 X prelude unsupported — per-32 only).
//
// Grid: [ceil(M/16), ceil(N/16)]. Block: [32,1,1]. LDS: 0.
// Partial M/N tiles are masked in-kernel (clamped A-row, `col_valid` /
// `out_row < M` guards), so odd prefill batches need no host padding.
//
// Compile gate: gfx1200 / gfx1201 only. Other archs get empty stubs (the
// RDNA3 LDS-tile MMQ kernel lives in `gemm_mq4g256v2_residual_mmq.hip`).

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

// Shared compute body for residual / full_add / full_set. `ADD` selects
// += (residual) vs = (set) writeback.
template <bool ADD>
static __device__ __forceinline__ void gemm_mq4g256v2_residual_mmq_gfx12_body(
    const char*           __restrict__ A,   // [M x groups_per_row x 136]
    const block_q8_1_mmq* __restrict__ Xq,  // [K/128 x N]
    float*                __restrict__ Y,   // [N x M] column-major
    int M, int K, int N
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int col_start = blockIdx.y * 16;

    if (row_start >= M || col_start >= N) return;

    const int m_lane = tid & 15;   // 0..15: row within the 16-row tile (A operand)
    const int k_grp  = tid >> 4;   // 0/1: which K-half this lane carries

    // Column (N-dim) handled by THIS lane's m_lane. Both k_grp halves of the
    // same m_lane access the same Q8_1 column (different K halves of qs[]).
    const int col = col_start + m_lane;
    const bool col_valid = (col < N);

    // A-row selection: pad OOB rows to the last valid row (M-1). Each lane
    // (m_lane) owns ONE row of the WMMA A operand; both lane halves (k_grp 0/1)
    // read from the SAME row (different K halves).
    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;

    // Output accumulator (gfx12 wave32 layout): acc[j] = Y[8*k_grp + j][m_lane].
    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const int blocks128 = K / 128;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;

        // 2 Q8_1 blocks per group (256 K-elements / 128 = 2). Each block holds
        // 4 sub-blocks of 32 K = 4 (d_x, sum_x) pairs. The V2 header for THIS
        // half (kb_in_group) owns both the nibble K-range and the scale that
        // corrects it: ksc 0 -> gp+0 for K 0..127, ksc 1 -> gp+4 for K 128..255.
        for (int kb_in_group = 0; kb_in_group < 2; kb_in_group++) {
            const int kb_q8 = 2 * g + kb_in_group;
            if (kb_q8 >= blocks128) break;

            // V2 dual-fp16 header for this lane's row and this K half.
            const unsigned int hs =
                *(const unsigned int*)(gp + 4 * kb_in_group);
            const float sc_self =
                __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
            const float zp_self =
                __half2float(__ushort_as_half((unsigned short)(hs >> 16)));

            // acc[j] = Y[8*k_grp + j][m_lane]: output row r = row_start + 8*k_grp + j.
            // The sc/zp for row r belongs to the lane whose own row is r — i.e. the
            // lane with m_lane = 8*k_grp + j. sc_self/zp_self at lane L holds row L's
            // value, so shuffle from src_lane = 8*k_grp + j (confined to [0,16)
            // within each k_grp half).
            float sc_row[8];
            float zp_row[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int src_lane = 8 * k_grp + j;
                sc_row[j] = __shfl(sc_self, src_lane);
                zp_row[j] = __shfl(zp_self, src_lane);
            }

            // Pull the Q8_1 block for THIS lane's column (or null for OOB cols).
            const block_q8_1_mmq* xb = col_valid
                ? (Xq + (long long)kb_q8 * N + col)
                : nullptr;

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
                    // V2 packed nibbles, 1 uint32 = 8 nibbles per lane.
                    // K-offset within group:
                    //   kb_in_group * 128  (Q8_1 block within group)
                    // + sb * 32             (sub-block)
                    // + kt * 16             (WMMA call within sub-block)
                    // + k_grp * 8           (lane-group's K half).
                    const int k_off = kb_in_group * 128 + sb * 32 + kt * 16 + k_grp * 8;
                    pk_arr[slot] = *(const unsigned int*)(gp + 8 + k_off / 2);
                    // Q8_1 quants: 8 int8 per lane.
                    if (xb != nullptr) {
                        const int8_t* qs_p = xb->qs + sb * 32 + kt * 16 + k_grp * 8;
                        b_arr[slot] = *(const int8x8_t*)qs_p;
                    } else {
                        b_arr[slot] = (int8x8_t){0, 0, 0, 0, 0, 0, 0, 0};
                    }
                }
            }

            // 4 sub-blocks of 32 K each. Per sub-block: TWO i8 WMMA calls
            // (K=16 each) on an int32 accumulator that resets at the start of
            // the sub-block, then the scale FMA across (sc_row, zp_row, d_x, sum_x).
            #pragma unroll
            for (int sb = 0; sb < 4; sb++) {
                int32x8_t cacc = {0, 0, 0, 0, 0, 0, 0, 0};

                #pragma unroll
                for (int kt = 0; kt < 2; kt++) {
                    const int slot = sb * 2 + kt;
                    const unsigned int pk = pk_arr[slot];

                    int8x8_t a_vec;
                    a_vec[0] = (int8_t)((pk >>  0) & 0xFu);
                    a_vec[1] = (int8_t)((pk >>  4) & 0xFu);
                    a_vec[2] = (int8_t)((pk >>  8) & 0xFu);
                    a_vec[3] = (int8_t)((pk >> 12) & 0xFu);
                    a_vec[4] = (int8_t)((pk >> 16) & 0xFu);
                    a_vec[5] = (int8_t)((pk >> 20) & 0xFu);
                    a_vec[6] = (int8_t)((pk >> 24) & 0xFu);
                    a_vec[7] = (int8_t)((pk >> 28) & 0xFu);

                    int32x2_t a_i32 = *(const int32x2_t*)&a_vec;
                    int32x2_t b_i32 = *(const int32x2_t*)&b_arr[slot];

                    // signed_a = false (V2 nibbles are 0..15, unsigned-safe);
                    // signed_b = true (Q8_1 is signed -127..127). clamp = true
                    // saturates the int32 accumulator (safe for K=16 chunks:
                    // max abs dot = 16 * 15 * 127 ≈ 30k).
                    cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        false, a_i32, true, b_i32, cacc, true
                    );
                } // for kt

                // Scale FMA: per output element (j in [0,8)).
                //   dot_real = sc * d_x * sum_k(q4*q8) + zp * (d_x * sum_k(q8))
                //            = sc * d_x * cacc[j] + zp * sum_real_x
                // Q8_1 stores the sum of REAL x values directly in ds4[].y, so
                // we multiply by sum_x_self only (NOT d_x_self * sum_x_self).
                // Matches the RDNA3 `vec_dot_q8_1_q8_1_mma` convention.
                const float d_x_self   = d_x_arr[sb];
                const float sum_x_self = sum_x_arr[sb];
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float dot_f = (float)cacc[j];
                    acc[j] += sc_row[j] * d_x_self * dot_f
                            + zp_row[j] * sum_x_self;
                }
            } // for sb
        } // for kb_in_group
    } // for g

    // gfx12 wave32 WMMA C-mapping: acc[j] = C[8*k_grp + j][m_lane].
    // Column = col (= col_start + m_lane); row = row_start + 8*k_grp + j.
    if (col_valid) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 8 * k_grp + j;
            if (out_row < M) {
                float* yp = Y + (long long)col * M + out_row;
                if constexpr (ADD) {
                    *yp += acc[j];
                } else {
                    *yp = acc[j];
                }
            }
        }
    }
}

// residual: += (the `add` kernarg is honored at runtime to pick += vs =).
extern "C" __global__ __launch_bounds__(32, 2)
void gemm_mq4g256v2_residual_mmq(
    const char*           __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int add
) {
    if (add) {
        gemm_mq4g256v2_residual_mmq_gfx12_body<true>(A, Xq, Y, M, K, N);
    } else {
        gemm_mq4g256v2_residual_mmq_gfx12_body<false>(A, Xq, Y, M, K, N);
    }
}

extern "C" __global__ __launch_bounds__(32, 2)
void gemm_mq4g256v2_residual_mmq_full_add(
    const char*           __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int
) {
    gemm_mq4g256v2_residual_mmq_gfx12_body<true>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(32, 2)
void gemm_mq4g256v2_residual_mmq_full_set(
    const char*           __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int
) {
    gemm_mq4g256v2_residual_mmq_gfx12_body<false>(A, Xq, Y, M, K, N);
}

#else

// Stubs for non-gfx12 archs (the RDNA3 LDS-tile MMQ kernel lives in
// `gemm_mq4g256v2_residual_mmq.hip` under its own arch guard).
extern "C" __global__ void gemm_mq4g256v2_residual_mmq(
    const char*, const void*, float*, int, int, int, int
) {}

extern "C" __global__ void gemm_mq4g256v2_residual_mmq_full_add(
    const char*, const void*, float*, int, int, int, int
) {}

extern "C" __global__ void gemm_mq4g256v2_residual_mmq_full_set(
    const char*, const void*, float*, int, int, int, int
) {}

#endif
