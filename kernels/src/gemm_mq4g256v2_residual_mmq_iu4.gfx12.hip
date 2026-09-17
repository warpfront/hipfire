// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// MQ4G256V2 iu4-direct MMQ for gfx12 (RDNA4 — gfx1200/gfx1201).
//
// Feeds MQ4V2 weight nibbles DIRECTLY to
// `__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12` (K=32 per call) with
// int4-quantized activations — no nibble expand, no per-fragment e4m3
// correction. Shares the dispatch contract / 7-arg signature with the gfx11
// iu4 kernel (`gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`) and the gfx12 iu8
// MMQ (`gemm_hfq4g256_residual_mmq.gfx12.hip`):
//   - A:  MQ4V2 packed weights [M rows × groups_per_row × 136 bytes]
//         (per 256-K group: 8-byte header = two fp16 (sc,zp) pairs, one per
//         128-K half, then 128 bytes nibble-packed with bit[4m..4m+3] = K
//         element m — same contract as the gfx11 kernel).
//   - Xq: activations pre-quantized to `block_i4_128` (72 B: f32 d, i32 s,
//         64 B nibbles, even K low / odd K high) via `quantize_int4_mmq_ds128`
//         (`ensure_int4_mmq_x`), layout [K/128 × N].
//   - Y:  output [N × M] column-major: Y[col * M + row].
//   - add: 1 => Y[..] += result (residual / _full_add); 0 => Y[..] = result.
//
// gfx12 K32 conventions (probe-verified on gfx1201, 2026-09-17 — one-hot /
// half-split / nibble-slot sweeps over a single-call WMMA):
//   - int32x2 A/B per lane (16 nibbles/lane = 8 per int32, low nibble first).
//   - K-tile lane split: k_grp = tid >> 4. Lanes 0-15 carry K[0..15] of row
//     m_lane (= tid & 15); lanes 16-31 carry K[16..31] of the same row.
//     A/B halves pair straight (cross-half dot reads 0).
//   - signed_a = false (weight nibbles are unsigned codes), signed_b = true
//     (int4 activations are two's complement, -8..7).
//   - Contiguous-row C output: acc[j] = C[8*k_grp + j][m_lane] (j=0..7),
//     int32x8 — same C map as the gfx12 iu8 MMQ.
//   - Per-half MQ4V2 (sc,zp) via __shfl from src_lane = 8*k_grp + j.
//
// Tiling: one wave covers a 16-row × 64-column tile (IU4_NB=4 column blocks
// of 16). Each weight fragment is loaded ONCE per (group, sub-block) into
// registers and reused across the 4 column blocks — weight/Xq traffic is
// 4× lower than a 16-column wave tile (the GEMM is weight-traffic bound on
// gfx1201: a 16-wide tile re-reads weights N/16 times).
//
// Compute: per 128-K block and per column block, FOUR K32 iu4 WMMAs
// accumulate into ONE int32 C, then ONE fold with (d_x, s_x = exact int sum
// of the 128 x_q). Integer accumulation is exact, so the 4×K32 grouping is
// bit-identical to gfx11's 8×K16; the fold spells the pinned IU4_FOLD_RN DAG
// verbatim (same file comment as gfx11: t1=sc*d, p=t1*C, t2=zp*d,
// term=t2*s+p, sum+=term).
//
// Grid: [ceil(M/16), ceil(N/64)]. Block: [32,1,1]. LDS: 0.
//
// Compile gate: gfx1200 / gfx1201 only. Non-gfx12 archs get empty stubs (the
// gfx11 iu4 kernel lives in its own file / arch guard).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#if defined(__gfx1200__) || defined(__gfx1201__)

// block_i4_128 is provided by the host JIT prelude
// (kernels/src/block_i4_128_quant.hip, concatenated by
// GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_SRC).

// Column blocks of 16 per wave (64-column wave tile). Each weight fragment
// is reused across IU4_NB WMMAs (one per column block).
#define IU4_NB 4

typedef int8_t  __attribute__((ext_vector_type(8)))  int8x8_t;
typedef int32_t __attribute__((ext_vector_type(2)))  int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8)))  int32x8_t;
typedef float   __attribute__((ext_vector_type(8)))  float8_t;

// IU4_FOLD_RN — W4.0/LF.0 bit-exact fold pin, shared with the gfx11 iu4
// kernel. Every output element folds one int32 WMMA result C through EXACTLY
// this DAG:
//   t1   = RN(sc * d)
//   p    = RN(t1 * float(C))
//   t2   = RN(zp * d)
//   term = RN(t2 * float(s) + p)
//   sum  = RN(sum + term)
#define IU4_FOLD_RN(sum_lvalue, sc, zp, d, acc_i32, s_i32) \
    do { \
        const float iu4_fold_d_ = (d); \
        const float iu4_fold_t1_ = __fmul_rn((sc), iu4_fold_d_); \
        const float iu4_fold_p_ = __fmul_rn(iu4_fold_t1_, (float)(acc_i32)); \
        const float iu4_fold_t2_ = __fmul_rn((zp), iu4_fold_d_); \
        const float iu4_fold_term_ = __fmaf_rn(iu4_fold_t2_, (float)(s_i32), iu4_fold_p_); \
        (sum_lvalue) = __fadd_rn((sum_lvalue), iu4_fold_term_); \
    } while (0)

// Shared compute body for residual / full_add / full_set. `ADD` selects
// += (residual) vs = (set) writeback.
template <bool ADD>
static __device__ __forceinline__ void gemm_mq4g256v2_residual_mmq_iu4_gfx12_body(
    const char*           __restrict__ A,   // [M × groups_per_row × 136]
    const block_i4_128*   __restrict__ Xq,  // [K/128 × N]
    float*                __restrict__ Y,   // [N × M] column-major
    int M, int K, int N
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int col_start = blockIdx.y * (16 * IU4_NB);

    if (row_start >= M || col_start >= N) return;

    const int m_lane = tid & 15;   // 0..15: row within the 16-row tile (A operand)
    const int k_grp  = tid >> 4;   // 0/1: which K-half this lane carries

    // Columns (N-dim) handled by THIS lane: one per column block. Both k_grp
    // halves of the same m_lane access the same int4 columns (different K
    // halves of qs[]).
    int col_nb[IU4_NB];
    bool col_valid[IU4_NB];
    #pragma unroll
    for (int nb = 0; nb < IU4_NB; nb++) {
        col_nb[nb] = col_start + nb * 16 + m_lane;
        col_valid[nb] = (col_nb[nb] < N);
    }

    // A-row selection: pad OOB rows to the last valid row (M-1). Each lane
    // (m_lane) owns ONE row of the WMMA A operand; both lane halves (k_grp 0/1)
    // read from the SAME row (different K halves).
    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;

    // Output accumulators (gfx12 wave32 layout):
    // acc[nb][j] = Y[8*k_grp + j][col_nb[nb]].
    float8_t acc[IU4_NB];
    #pragma unroll
    for (int nb = 0; nb < IU4_NB; nb++) {
        acc[nb] = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        // MQ4V2 per-group header: word h holds the fp16 (sc, zp) pair for
        // 128-K half h (same contract as the gfx11 iu4 kernel's ksc select).
        const unsigned int hs0 = *(const unsigned int*)(gp + 0);
        const unsigned int hs1 = *(const unsigned int*)(gp + 4);
        const float sc0_self = __half2float(__ushort_as_half((unsigned short)(hs0 & 0xFFFFu)));
        const float zp0_self = __half2float(__ushort_as_half((unsigned short)(hs0 >> 16)));
        const float sc1_self = __half2float(__ushort_as_half((unsigned short)(hs1 & 0xFFFFu)));
        const float zp1_self = __half2float(__ushort_as_half((unsigned short)(hs1 >> 16)));

        // 2 int4 blocks per group (256 K-elements / 128 = 2). Each block holds
        // 4 sub-blocks of 32 K covered by ONE K32 WMMA each, per column block.
        for (int kb_in_group = 0; kb_in_group < 2; kb_in_group++) {
            const int kb = 2 * g + kb_in_group;

            // Per-half scale rows for the fold below: acc[nb][j] holds output
            // row row_start + 8*k_grp + j, whose (sc,zp) live on the lane with
            // m_lane = 8*k_grp + j. Shared across column blocks (same rows).
            const float sc_self = (kb_in_group == 0) ? sc0_self : sc1_self;
            const float zp_self = (kb_in_group == 0) ? zp0_self : zp1_self;
            float sc_row[8];
            float zp_row[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int src_lane = 8 * k_grp + j;
                sc_row[j] = __shfl(sc_self, src_lane);
                zp_row[j] = __shfl(zp_self, src_lane);
            }

            // Pull the int4 block for each column block (null for OOB cols).
            const unsigned char* qs_nb[IU4_NB];
            float d_x_nb[IU4_NB];
            int s_x_nb[IU4_NB];
            #pragma unroll
            for (int nb = 0; nb < IU4_NB; nb++) {
                const block_i4_128* xb = col_valid[nb]
                    ? (Xq + (long long)kb * N + col_nb[nb])
                    : nullptr;
                d_x_nb[nb] = (xb != nullptr) ? xb->d : 0.0f;
                s_x_nb[nb] = (xb != nullptr) ? xb->s : 0;
                qs_nb[nb]  = (xb != nullptr) ? xb->qs : nullptr;
            }

            // 4 sub-blocks of 32 K each accumulate into ONE int32 C per column
            // block; then ONE fold with (d_x, s_x). Exact integer arithmetic:
            // grouping into 4×K32 matches gfx11's 8×K16 bit-for-bit. The A
            // fragment loads ONCE and is reused across the IU4_NB columns.
            int32x8_t cacc[IU4_NB];
            #pragma unroll
            for (int nb = 0; nb < IU4_NB; nb++) {
                cacc[nb] = (int32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
            }
            #pragma unroll
            for (int sb = 0; sb < 4; sb++) {
                // K-offset within group:
                //   kb_in_group * 128 (int4 block within group)
                // + sb * 32            (sub-block)
                // + k_grp * 16         (lane-group's K half).
                const int k_off = kb_in_group * 128 + sb * 32 + k_grp * 16;
                // Weight nibbles: 2 packed words (16 nibbles) per lane.
                const int* ap = (const int*)(gp + 8 + k_off / 2);
                const int32x2_t a_i32 = *(const int32x2_t*)ap;
                // Activations: 8 nibble bytes (16 nibbles) per lane, per column.
                #pragma unroll
                for (int nb = 0; nb < IU4_NB; nb++) {
                    int32x2_t b_i32;
                    if (qs_nb[nb] != nullptr) {
                        b_i32 = *(const int32x2_t*)(qs_nb[nb] + sb * 16 + k_grp * 8);
                    } else {
                        b_i32 = (int32x2_t){0, 0};
                    }
                    cacc[nb] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                        false, a_i32, true, b_i32, cacc[nb], false
                    );
                }
            }

            #pragma unroll
            for (int nb = 0; nb < IU4_NB; nb++) {
                const int* acc_i = (const int*)&cacc[nb];
                const float d_x = d_x_nb[nb];
                const int s_x = s_x_nb[nb];
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    IU4_FOLD_RN(acc[nb][j], sc_row[j], zp_row[j], d_x, acc_i[j], s_x);
                }
            }
        } // for kb_in_group
    } // for g

    // gfx12 wave32 WMMA C-mapping: acc[nb][j] = C[8*k_grp + j][col_nb[nb]].
    #pragma unroll
    for (int nb = 0; nb < IU4_NB; nb++) {
        if (col_valid[nb]) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int out_row = row_start + 8 * k_grp + j;
                if (out_row < M) {
                    float* yp = Y + (long long)col_nb[nb] * M + out_row;
                    if constexpr (ADD) {
                        *yp += acc[nb][j];
                    } else {
                        *yp = acc[nb][j];
                    }
                }
            }
        }
    }
}

// residual: += (the `add` kernarg picks += vs = at runtime).
extern "C" __global__ __launch_bounds__(32, 2)
void gemm_mq4g256v2_residual_mmq_iu4(
    const char*           __restrict__ A,
    const block_i4_128*   __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int add
) {
    if (add) {
        gemm_mq4g256v2_residual_mmq_iu4_gfx12_body<true>(A, Xq, Y, M, K, N);
    } else {
        gemm_mq4g256v2_residual_mmq_iu4_gfx12_body<false>(A, Xq, Y, M, K, N);
    }
}

extern "C" __global__ __launch_bounds__(32, 2)
void gemm_mq4g256v2_residual_mmq_iu4_full_add(
    const char*           __restrict__ A,
    const block_i4_128*   __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int
) {
    gemm_mq4g256v2_residual_mmq_iu4_gfx12_body<true>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(32, 2)
void gemm_mq4g256v2_residual_mmq_iu4_full_set(
    const char*           __restrict__ A,
    const block_i4_128*   __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int
) {
    gemm_mq4g256v2_residual_mmq_iu4_gfx12_body<false>(A, Xq, Y, M, K, N);
}

#else

// Stubs for non-gfx12 archs (the gfx11 iu4 kernel and gfx906 dp4a MMQ kernel
// live in their own arch guards / files).
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4(
    const char*, const void*, float*, int, int, int, int
) {}

extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_add(
    const char*, const void*, float*, int, int, int, int
) {}

extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_set(
    const char*, const void*, float*, int, int, int, int
) {}

#endif
