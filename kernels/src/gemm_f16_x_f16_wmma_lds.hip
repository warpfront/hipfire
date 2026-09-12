// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

// LDS-staged WMMA GEMM: F16 weight × F16 input → F32 output, fused bias.
// gfx1100+ only (RDNA3 / RDNA3.5 wave32 WMMA, 16×16×16 tiles).
//
// Same contract as gemm_f16_x_f16_wmma, plus an optional bias:
//   A    : [M, K] F16 row-major  (weight)
//   X    : [B, K] F16 row-major  (activations)
//   Y    : [B, M] F32 row-major
//   Bias : [M]    F32, added to every row of Y when has_bias != 0
//
// Compute: Y[b, m] = bias[m] + sum_k A[m, k] * X[b, k]
//
// WHY THIS KERNEL EXISTS
// ----------------------
// gemm_f16_x_f16_wmma gives one wave32 a 16×16 output tile with LDS 0. Per
// K-step of 16 that wave reads 16 A rows × 16 halves (512 B) plus the same of
// X (512 B) and issues 2·16·16·16 = 8192 FLOP — a fixed arithmetic intensity
// of 8 FLOP/byte, independent of M, K and B. At the FLUX.1-dev MMDiT shapes
// that means the kernel moves 43–186× the compulsory traffic (measured on
// gfx1150: single/linear2 [M=3072,K=15360,B=4608] moves 54.4 GB to read a
// 292 MB working set).
//
// This kernel raises the macro-tile to 128×128, so each staged byte feeds 8×
// more MACs: intensity 2·128·128 / ((128+128)·2) = 64 FLOP/byte. It also fuses
// the bias that the caller otherwise pays as a separate full-tensor
// read-modify-write pass over Y.
//
// Prior art that shaped the design (docs/lessons_learned/
// gfx12_prefill_wmma_2026_05_19.md): register blocking + LDS staging measured
// +56% on pure FP16 WMMA GEMM at 4K³, and regressed only on *quantized*
// on-the-fly-dequant kernels, where dequant serializes the K-substep loop.
// This kernel is the pure-FP16 case, so it is on the side of that split.
//
// TILING
//   Macro-tile : 128 (M, out-features) × 128 (B, tokens)
//   Block      : 256 threads = 8 waves of 32
//   Wave grid  : 4 along M × 2 along B
//                wave_m = wave_id & 3 → M rows [wave_m*32, +32)  (2 subtiles)
//                wave_n = wave_id >> 2 → B cols [wave_n*64, +64) (4 subtiles)
//   Per wave   : 2 × 4 = 8 float8_t accumulators (64 VGPRs)
//   K stage    : 64 elements (4 WMMA k-substeps) per barrier pair
//
// LDS LAYOUT — tile-major [kt][row][ki], NOT row-major.
// 16 lanes each read a 32-byte half16 from a different row at the same kt.
// Tile-major puts those 16 reads 16 halves (8 dwords) apart, giving four-way
// bank pressure; a row-major [row][64] layout would put them 64 halves
// (32 dwords) apart, i.e. all on one bank — a 16-way conflict. This is the
// same reasoning as gemm_hfq4g256_residual_wmma_gfx1100_muse_lds_g256.hip.
//   a_lds : [4][128][16] F16 = 16384 B
//   x_lds : [4][128][16] F16 = 16384 B
//   total 32768 B, so 2 workgroups/CU by LDS on a 64 KB-per-CU part.
//
// PRECONDITION: K % 64 == 0. Every FLUX.1-dev K is a multiple of 64
// (3072, 4096, 12288, 15360, 768, 256, 64). The Rust wrapper enforces it and
// falls back to gemm_f16_x_f16_wmma otherwise.
//
// M and B tails are handled by clamping OOB rows to the last valid row and
// discarding at the store, per the house pattern. No thread early-returns
// after the block-uniform origin check — every thread must reach both
// __syncthreads() of every stage.
//
// Grid:  [ceil(M / 128), ceil(B / 128)]
// Block: [256]
// LDS:   32768 (static)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define LDS_BM 128   // macro-tile rows (M, out features)
#define LDS_BN 128   // macro-tile cols (B, tokens)
#define LDS_KS 64    // K elements staged per barrier pair
#define LDS_KT 4     // = LDS_KS / 16, WMMA k-substeps per stage

__launch_bounds__(256, 1)
extern "C" __global__ void gemm_f16_x_f16_wmma_lds(
    const _Float16* __restrict__ A,     // [M, K] F16
    const _Float16* __restrict__ X,     // [B, K] F16
    float*          __restrict__ Y,     // [B, M] F32
    const float*    __restrict__ Bias,  // [M] F32, may be null
    int M, int K, int B, int has_bias
) {
    const int m_block = blockIdx.x * LDS_BM;
    const int n_block = blockIdx.y * LDS_BN;

    // Block-uniform origin check — legal before any barrier because every
    // thread in the block agrees on it.
    if (m_block >= M || n_block >= B) return;

    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int wave_id = tid >> 5;
    const int ml      = lane & 15;   // fragment row; lanes 16..31 mirror 0..15

    const int wave_m = wave_id & 3;         // 0..3 → 32 M rows each
    const int wave_n = wave_id >> 2;        // 0..1 → 64 B cols each
    const int m_wave = m_block + wave_m * 32;
    const int n_wave = n_block + wave_n * 64;

    __shared__ __align__(32) _Float16 a_lds[LDS_KT][LDS_BM][16];
    __shared__ __align__(32) _Float16 x_lds[LDS_KT][LDS_BN][16];

    float8_t acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i] = float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    // Staging map: 128 rows × 4 k-subtiles = 512 half16 vectors per operand,
    // 256 threads → 2 vectors each. Consecutive tid walk the 4 subtiles of one
    // row before moving to the next row, so each group of 4 lanes covers 128
    // contiguous bytes of a single source row.
    const int stage_row = tid >> 2;          // 0..63 on the first pass
    const int stage_kt  = tid & 3;           // 0..3

    for (int k0 = 0; k0 < K; k0 += LDS_KS) {
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const int row = stage_row + half * 64;

            // Clamp OOB rows to the last valid one. They are computed but
            // discarded at the store; they must never read out of bounds.
            const int a_row = m_block + row;
            const int x_row = n_block + row;
            const int a_safe = (a_row < M) ? a_row : (M - 1);
            const int x_safe = (x_row < B) ? x_row : (B - 1);

            const long long a_off = (long long)a_safe * K + k0 + stage_kt * 16;
            const long long x_off = (long long)x_safe * K + k0 + stage_kt * 16;

            *(half16_t*)(&a_lds[stage_kt][row][0]) = *(const half16_t*)(A + a_off);
            *(half16_t*)(&x_lds[stage_kt][row][0]) = *(const half16_t*)(X + x_off);
        }

        __syncthreads();

        #pragma unroll
        for (int kt = 0; kt < LDS_KT; kt++) {
            const half16_t a0 = *(const half16_t*)(&a_lds[kt][wave_m * 32 + ml][0]);
            const half16_t a1 = *(const half16_t*)(&a_lds[kt][wave_m * 32 + 16 + ml][0]);

            // One B fragment live at a time keeps peak VGPR pressure down;
            // both A fragments are reused across all four N subtiles.
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const half16_t b = *(const half16_t*)(&x_lds[kt][wave_n * 64 + j * 16 + ml][0]);
                acc[j]     = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b, acc[j]);
                acc[4 + j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b, acc[4 + j]);
            }
        }

        __syncthreads();
    }

    // Epilogue. RDNA3 wave32 WMMA: acc[e] = C[2*e + (lane>>4)][lane & 15],
    // where the C row is the M index (from the A fragment) and the C column is
    // the B index (from the B fragment).
    const int row_half = lane >> 4;

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int out_b = n_wave + j * 16 + ml;
            if (out_b >= B) {
                continue;
            }
            const float8_t v = acc[i * 4 + j];
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const int out_m = m_wave + i * 16 + 2 * e + row_half;
                if (out_m < M) {
                    const float bv = has_bias ? Bias[out_m] : 0.0f;
                    Y[(long long)out_b * M + out_m] = v[e] + bv;
                }
            }
        }
    }
}
