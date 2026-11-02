// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// IMMUTABLE ORACLE. Do not edit, reformat, or "keep in sync" with
// kernels/src/gemm_fp8b128_moe_grouped_bf16mfma.gfx942.hip. This file is the
// retained P1 (cdc37659a) transport/order of the gfx942 qt=55 FP8E4M3B128
// grouped MoE GEMM, extracted as a standalone source from the `REFERENCE=true`
// instantiation of the production kernel at 28d70996c so that later edits to
// the production kernel's shared helpers or loop cannot move it. Its SHA-256 is
// pinned by the unit test `fp8b128_moe_reference_snapshot_is_pinned` in
// crates/rdna-compute/src/cdna/gfx942.rs; changing a byte here means
// re-pinning the oracle, which is forbidden (the candidate is always compared
// against THESE bytes; never regenerate the oracle from a candidate).
//
// Exports (the test-only reference module, same names and ABI as before):
//   gemm_fp8b128_moe_grouped_bf16mfma_reference        (X: F32 holding BF16 values)
//   gemm_fp8b128_moe_grouped_bf16mfma_reference_xbf16  (X: BF16 bits)
// Geometry: 256 threads (4 wave64), grid [ceil(R/64), n_tiles]; wave w owns
// output rows blockIdx.x*64 + w*16 + (0..15) of tile blockIdx.y.
//
// Tiles: tile_desc[t] = {expert, row0, nrows, 0} (i32x4). expert < 0 skips the
// tile. sorted_rows[i] is a flat slot id or -1 (padding: Y row written +0);
// X row = sorted_rows[i] / x_row_div. Y [n_sorted, R] F32 is overwritten for
// every row of a live tile and nothing else is written.
//
// Weights (qt=55 block of expert e at W + e * expert_stride): u8 codes [R, K]
// row-major (OCP e4m3fn) then bf16 s_inv [R/128, K/128]. Decode:
//   W = bf16_rne(f32(e4m3fn(q)) * f32(s_inv))
// via the hardware FNUZ convert on |q| (half the e4m3fn magnitude) times
// f32(2*s_inv), RNE to BF16 on the bits, then the code's sign XORed in.
//
// MFMA: __builtin_amdgcn_mfma_f32_16x16x16bf16_1k, controls (0,0,0):
//   A: lane l holds a[r] = A[m = l%16, k = (l/16)*4 + r]   (weights)
//   B: lane l holds b[r] = B[k = (l/16)*4 + r, n = l%16]   (activations)
//   D: lane l holds c[r] = D[m = (l/16)*4 + r, n = l%16]
// Reduction order for every output: 128-wide K blocks kb ascending; two 64-wide
// halves h = 0, 1; four MFMA steps t = 0..3 contracting physical
// k = 128*kb + 64*h + 16*t + 4*s + r into the F32 accumulator (initial +0).
// In this P1 transport lane group s loads its four codes of every K16 step as
// four 4-byte dwords at byte offset 4*s + 16*t of the half (no cross-lane
// shuffle). X is staged through LDS in groups of D = 2 K-blocks; the weights of
// the next group are fetched into registers while the current group computes.
//
// Contract: K % 128 == 0, R % 16 == 0, expert_stride % 16 == 0.

#include <hip/hip_runtime.h>

// gfx942-only at build time.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx942__)
#error "fp8_moe_grouped_reference_p1_cdc37659a.gfx942.hip is gfx942-only"
#endif

// 16-row subtiles per wave (rows per workgroup = 64 * FP8REF_NSUB; the
// launcher's FP8B128_MOE_REFERENCE_ROWS_PER_WG is pinned to this by a unit
// test) and K-blocks per prefetch group.
#define FP8REF_NSUB 1
#define FP8REF_D 2

typedef short fp8ref_s16x4 __attribute__((ext_vector_type(4)));
typedef float fp8ref_f32x4 __attribute__((ext_vector_type(4)));
typedef float fp8ref_f32x2 __attribute__((ext_vector_type(2)));
typedef unsigned fp8ref_u32x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ unsigned fp8ref_rne_bias(unsigned u) {
    return u + 0x7FFFu + ((u >> 16) & 1u);
}

// (hi.bits[31:16] << 16) | lo.bits[31:16]
__device__ __forceinline__ unsigned fp8ref_pack_hi(unsigned hi, unsigned lo) {
    return __builtin_amdgcn_perm(hi, lo, 0x07060302u);
}

__device__ __forceinline__ unsigned fp8ref_f32x2_to_bf16x2(float a, float b) {
    return fp8ref_pack_hi(fp8ref_rne_bias(__float_as_uint(b)), fp8ref_rne_bias(__float_as_uint(a)));
}

// Four codes (byte i = element k+i) -> four BF16: o01 = bf16(k)|bf16(k+1)<<16.
// s2 = f32(2 * s_inv).
__device__ __forceinline__ void fp8ref_decode4(unsigned w, float s2, unsigned& o01, unsigned& o23) {
    const unsigned mag = w & 0x7F7F7F7Fu;
    fp8ref_f32x2 a = __builtin_amdgcn_cvt_pk_f32_fp8((int)mag, false);
    fp8ref_f32x2 b = __builtin_amdgcn_cvt_pk_f32_fp8((int)mag, true);
    a *= s2;
    b *= s2;
    const unsigned sg = w & 0x80808080u;
    o01 = fp8ref_pack_hi(fp8ref_rne_bias(__float_as_uint(a.y)), fp8ref_rne_bias(__float_as_uint(a.x)))
        ^ __builtin_amdgcn_perm(sg, sg, 0x010C000Cu);
    o23 = fp8ref_pack_hi(fp8ref_rne_bias(__float_as_uint(b.y)), fp8ref_rne_bias(__float_as_uint(b.x)))
        ^ __builtin_amdgcn_perm(sg, sg, 0x030C020Cu);
}

__device__ __forceinline__ float fp8ref_scale2(unsigned short bits) {
    return 2.0f * __uint_as_float((unsigned)bits << 16);
}

__device__ __forceinline__ fp8ref_f32x4 fp8ref_mfma(unsigned a0, unsigned a1, unsigned b0, unsigned b1,
                                                    fp8ref_f32x4 c) {
    fp8ref_u32x2 a = {a0, a1};
    fp8ref_u32x2 b = {b0, b1};
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
        __builtin_bit_cast(fp8ref_s16x4, a), __builtin_bit_cast(fp8ref_s16x4, b), c, 0, 0, 0);
}

template <bool XBF16>
__device__ __forceinline__ void fp8ref_grouped_body(
    const unsigned char* __restrict__ W,
    unsigned long long expert_stride,
    const int4* __restrict__ tile_desc,
    const int* __restrict__ sorted_rows,
    const void* __restrict__ X,
    float* __restrict__ Y,
    int R, int K, int x_row_div) {
    constexpr int NSUB = FP8REF_NSUB;
    constexpr int D = FP8REF_D;
    // [D][16 rows][128 bf16 + 8 pad] as dwords; pad breaks the 256-B row stride.
    __shared__ unsigned xl[D][16][68];

    const int4 td = tile_desc[blockIdx.y];
    const int expert = td.x;
    const int row0 = td.y;
    const int nrows = td.z;
    if (expert < 0) return;
    const unsigned char* __restrict__ blk = W + (unsigned long long)expert * expert_stride;
    const unsigned short* __restrict__ sc = reinterpret_cast<const unsigned short*>(blk + (size_t)R * K);
    const int nkb = K >> 7;

    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int n = lane & 15;
    const int s = lane >> 4;
    const int row_base = blockIdx.x * 64 * NSUB + wave * 16 * NSUB;

    // X staging: thread -> tile row tid/16, 8 consecutive k at (tid%16)*8 per block.
    const int st_slot = tid >> 4;
    const int st_k = (tid & 15) * 8;
    const int flat = st_slot < nrows ? sorted_rows[row0 + st_slot] : -1;
    const long long xrow = flat >= 0 ? (long long)(x_row_div > 1 ? flat / x_row_div : flat) : -1;

    const unsigned char* wrow[NSUB];
    const unsigned short* srow[NSUB];
#pragma unroll
    for (int j = 0; j < NSUB; ++j) {
        const int r = min(row_base + j * 16 + n, R - 1);
        wrow[j] = blk + (size_t)r * K + 4 * s;
        srow[j] = sc + (min(row_base + j * 16, R - 1) >> 7) * nkb;
    }

    fp8ref_f32x4 acc[NSUB];
#pragma unroll
    for (int j = 0; j < NSUB; ++j) acc[j] = fp8ref_f32x4{0.0f, 0.0f, 0.0f, 0.0f};

    uint4 wc[D][NSUB][2], wn[D][NSUB][2];
    unsigned short scc[D][NSUB], scn[D][NSUB];
    uint4 xc[D], xn[D];

    auto load_group = [&](int kb0, uint4 (&w)[D][NSUB][2], unsigned short (&sv)[D][NSUB], uint4 (&xv)[D]) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
            const int kb = kb0 + d;
            if (kb < nkb) {
#pragma unroll
                for (int j = 0; j < NSUB; ++j) {
                    const unsigned char* wp = wrow[j] + kb * 128;
                    w[d][j][0] = uint4{*reinterpret_cast<const unsigned*>(wp), *reinterpret_cast<const unsigned*>(wp + 16),
                                       *reinterpret_cast<const unsigned*>(wp + 32), *reinterpret_cast<const unsigned*>(wp + 48)};
                    w[d][j][1] = uint4{*reinterpret_cast<const unsigned*>(wp + 64), *reinterpret_cast<const unsigned*>(wp + 80),
                                       *reinterpret_cast<const unsigned*>(wp + 96), *reinterpret_cast<const unsigned*>(wp + 112)};
                    sv[d][j] = srow[j][kb];
                }
                if (xrow >= 0) {
                    const long long off = xrow * K + kb * 128 + st_k;
                    if constexpr (XBF16) {
                        xv[d] = *reinterpret_cast<const uint4*>(reinterpret_cast<const unsigned short*>(X) + off);
                    } else {
                        const float4 a = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(X) + off);
                        const float4 b = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(X) + off + 4);
                        xv[d] = uint4{fp8ref_f32x2_to_bf16x2(a.x, a.y), fp8ref_f32x2_to_bf16x2(a.z, a.w),
                                      fp8ref_f32x2_to_bf16x2(b.x, b.y), fp8ref_f32x2_to_bf16x2(b.z, b.w)};
                    }
                } else {
                    xv[d] = uint4{0u, 0u, 0u, 0u};
                }
            }
        }
    };

    load_group(0, wc, scc, xc);
    for (int kb0 = 0; kb0 < nkb; kb0 += D) {
        __syncthreads();
#pragma unroll
        for (int d = 0; d < D; ++d) {
            if (kb0 + d < nkb) *reinterpret_cast<uint4*>(&xl[d][st_slot][st_k / 2]) = xc[d];
        }
        __syncthreads();
        if (kb0 + D < nkb) load_group(kb0 + D, wn, scn, xn);
#pragma unroll
        for (int d = 0; d < D; ++d) {
            if (kb0 + d < nkb) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const unsigned* xr = &xl[d][n][32 * h + 2 * s];
                    const unsigned b0x = xr[0], b0y = xr[1];
                    const unsigned b1x = xr[8], b1y = xr[9];
                    const unsigned b2x = xr[16], b2y = xr[17];
                    const unsigned b3x = xr[24], b3y = xr[25];
#pragma unroll
                    for (int j = 0; j < NSUB; ++j) {
                        const float s2 = fp8ref_scale2(scc[d][j]);
                        const uint4 w = wc[d][j][h];
                        unsigned a0, a1;
                        fp8ref_decode4(w.x, s2, a0, a1);
                        acc[j] = fp8ref_mfma(a0, a1, b0x, b0y, acc[j]);
                        fp8ref_decode4(w.y, s2, a0, a1);
                        acc[j] = fp8ref_mfma(a0, a1, b1x, b1y, acc[j]);
                        fp8ref_decode4(w.z, s2, a0, a1);
                        acc[j] = fp8ref_mfma(a0, a1, b2x, b2y, acc[j]);
                        fp8ref_decode4(w.w, s2, a0, a1);
                        acc[j] = fp8ref_mfma(a0, a1, b3x, b3y, acc[j]);
                    }
                }
            }
        }
#pragma unroll
        for (int d = 0; d < D; ++d) {
            if (kb0 + D + d < nkb) {
                xc[d] = xn[d];
#pragma unroll
                for (int j = 0; j < NSUB; ++j) {
                    wc[d][j][0] = wn[d][j][0];
                    wc[d][j][1] = wn[d][j][1];
                    scc[d][j] = scn[d][j];
                }
            }
        }
    }

    if (n < nrows) {
        float* __restrict__ yrow = Y + (size_t)(row0 + n) * R;
#pragma unroll
        for (int j = 0; j < NSUB; ++j) {
            const int r0 = row_base + j * 16 + s * 4;
            if (r0 < R) {
                *reinterpret_cast<float4*>(yrow + r0) = float4{acc[j][0], acc[j][1], acc[j][2], acc[j][3]};
            }
        }
    }
}

// X: [x_rows, K] F32 (BF16-exact values; staged RNE to BF16).
extern "C" __global__ __launch_bounds__(256) void gemm_fp8b128_moe_grouped_bf16mfma_reference(
    const unsigned char* __restrict__ W, unsigned long long expert_stride,
    const int4* __restrict__ tile_desc, const int* __restrict__ sorted_rows,
    const float* __restrict__ X, float* __restrict__ Y, int R, int K, int x_row_div) {
    fp8ref_grouped_body<false>(W, expert_stride, tile_desc, sorted_rows, X, Y, R, K, x_row_div);
}

// X: [x_rows, K] BF16 bits.
extern "C" __global__ __launch_bounds__(256) void gemm_fp8b128_moe_grouped_bf16mfma_reference_xbf16(
    const unsigned char* __restrict__ W, unsigned long long expert_stride,
    const int4* __restrict__ tile_desc, const int* __restrict__ sorted_rows,
    const unsigned short* __restrict__ X, float* __restrict__ Y, int R, int K, int x_row_div) {
    fp8ref_grouped_body<true>(W, expert_stride, tile_desc, sorted_rows, X, Y, R, K, x_row_div);
}
