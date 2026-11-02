// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Grouped MoE GEMM over qt=55 FP8E4M3B128 expert blocks, BF16 MFMA, gfx942
// (MI300X, wave64). One kernel for prefill and decode: every launch is a list
// of tiles of up to 16 expert-sorted rows (decode = a tile per selected expert
// with one row), so a row's output bytes never depend on which other rows
// share the launch, its position in a tile, or the tile index.
//
// Tiles: tile_desc[t] = {expert, row0, nrows, 0} (i32x4). expert < 0 skips the
// tile. The tile covers sorted rows row0 .. row0+nrows-1 (nrows 1..16):
// sorted_rows[i] is a flat slot id (token*top_k + j) or -1 (padding: Y row
// written +0); X row = sorted_rows[i] / x_row_div. Y row i (of [n_sorted, R])
// is overwritten for every row of a live tile and nothing else is written.
//
// Weights (qt=55 block of expert e at W + e * expert_stride):
//   codes  u8   [R, K] row-major, OCP e4m3fn            at +0
//   s_inv  bf16 [R/128, K/128] row-major (multiplier)  at +R*K
// Decode (frozen numerical contract, teacher fp8_teacher.dequant_e4m3_block):
//   W = bf16_rne(f32(e4m3fn(q)) * f32(s_inv))
// computed with the hardware FNUZ convert: |q| (sign bit cleared, so 0x80
// never reaches the convert where it would be FNUZ NaN) read as FNUZ is
// exactly |e4m3fn(q)|/2, times f32(2*s_inv) is the exact F32 product, RNE to
// BF16 on the bits, then the code's sign bit is XORed into the BF16 sign
// (RNE is sign-symmetric, so this equals rounding the signed product and
// reproduces the teacher's -0 for 0x80). NaN codes 0x7F/0xFF are refused by
// the converter and never appear in an artifact.
//
// Activations: X rows are BF16 (the `_xbf16` entry) or F32 holding BF16 values
// (the F32 entry rounds RNE to BF16 while staging; identity on BF16-exact
// input). Output Y [n_sorted, R] F32, overwritten (no atomics).
//
// MFMA: __builtin_amdgcn_mfma_f32_16x16x16bf16_1k, lane layout (see
// gemm_bf16_mfma.gfx942.hip):
//   A: lane l holds a[r] = A[m = l%16, k = (l/16)*4 + r]   (weights, m = output row)
//   B: lane l holds b[r] = B[k = (l/16)*4 + r, n = l%16]   (activations, n = slot)
//   D: lane l holds c[r] = D[m = (l/16)*4 + r, n = l%16]
// Reduction order, identical for every output and every launch shape (no
// split-K): 128-wide K blocks kb ascending; inside a block two 64-wide halves
// h = 0, 1; inside a half four MFMA steps t = 0..3, step t contracting the 16
// contiguous physical k = 128*kb + 64*h + 16*t + (0..15) into the F32
// accumulator (initial value +0); within a step lane group s = lane/16 supplies
// k offsets 4*s+r (r = 0..3). Contiguous 16-byte code loads are transposed
// across the four strip lanes before decode; this changes only transport,
// never the BF16 operands or the ascending MFMA contraction sequence.
//
// Geometry: block 256 = 4 wave64; grid [ceil(R / (64*NSUB)), n_tiles].
// Wave w of workgroup x owns output rows x*64*NSUB + w*16*NSUB + j*16 + (0..15)
// (j < NSUB) for the rows of tile y (MFMA column n = row0 + n). X for the tile
// is staged through LDS in groups of D K-blocks (columns n >= nrows stay 0);
// weights of the next group are fetched into registers while the current
// group computes.
//
// Contract: K % 128 == 0, R % 16 == 0, expert_stride % 16 == 0 (the qt=55
// stride is a multiple of 256), X rows 32-byte aligned (K % 128 == 0).

#include <hip/hip_runtime.h>

// gfx942-only at build time: a device compile for any other target (RDNA
// included) is refused, so no non-gfx942 code object of this module (or of
// its FP8MOE_TEST_REFERENCE variant) can be produced or packed.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx942__)
#error "gemm_fp8b128_moe_grouped_bf16mfma.gfx942.hip is gfx942-only"
#endif

// 16-row subtiles per wave (rows per workgroup = 64 * NSUB; the launcher's
// FP8B128_MOE_ROWS_PER_WG is pinned to this by a unit test) and K-blocks per
// prefetch group. NSUB=1, D=2 fixes the launch geometry for every row count;
// neither value depends on the number of tokens or the expert occupancy.
// Each output keeps the same ascending K16 contraction order.
#define FP8MOE_NSUB 1
#define FP8MOE_D 2

typedef short fp8moe_s16x4 __attribute__((ext_vector_type(4)));
typedef float fp8moe_f32x4 __attribute__((ext_vector_type(4)));
typedef float fp8moe_f32x2 __attribute__((ext_vector_type(2)));
typedef unsigned fp8moe_u32x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ unsigned fp8moe_rne_bias(unsigned u) {
    return u + 0x7FFFu + ((u >> 16) & 1u);
}

// (hi.bits[31:16] << 16) | lo.bits[31:16]
__device__ __forceinline__ unsigned fp8moe_pack_hi(unsigned hi, unsigned lo) {
    return __builtin_amdgcn_perm(hi, lo, 0x07060302u);
}

__device__ __forceinline__ unsigned fp8moe_f32x2_to_bf16x2(float a, float b) {
    return fp8moe_pack_hi(fp8moe_rne_bias(__float_as_uint(b)), fp8moe_rne_bias(__float_as_uint(a)));
}

// Four codes (byte i = element k+i) -> four BF16: o01 = bf16(k)|bf16(k+1)<<16.
// s2 = f32(2 * s_inv).
__device__ __forceinline__ void fp8moe_decode4(unsigned w, float s2, unsigned& o01, unsigned& o23) {
    const unsigned mag = w & 0x7F7F7F7Fu;
    fp8moe_f32x2 a = __builtin_amdgcn_cvt_pk_f32_fp8((int)mag, false);
    fp8moe_f32x2 b = __builtin_amdgcn_cvt_pk_f32_fp8((int)mag, true);
    a *= s2;
    b *= s2;
    const unsigned sg = w & 0x80808080u;
    o01 = fp8moe_pack_hi(fp8moe_rne_bias(__float_as_uint(a.y)), fp8moe_rne_bias(__float_as_uint(a.x)))
        ^ __builtin_amdgcn_perm(sg, sg, 0x010C000Cu);
    o23 = fp8moe_pack_hi(fp8moe_rne_bias(__float_as_uint(b.y)), fp8moe_rne_bias(__float_as_uint(b.x)))
        ^ __builtin_amdgcn_perm(sg, sg, 0x030C020Cu);
}

__device__ __forceinline__ float fp8moe_scale2(unsigned short bits) {
    return 2.0f * __uint_as_float((unsigned)bits << 16);
}

__device__ __forceinline__ fp8moe_f32x4 fp8moe_mfma(unsigned a0, unsigned a1, unsigned b0, unsigned b1,
                                                    fp8moe_f32x4 c) {
    fp8moe_u32x2 a = {a0, a1};
    fp8moe_u32x2 b = {b0, b1};
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
        __builtin_bit_cast(fp8moe_s16x4, a), __builtin_bit_cast(fp8moe_s16x4, b), c, 0, 0, 0);
}

// Raw-code 4x4 transpose: source strip s holds L_s[j], destination strip s
// needs L_t[s] in register t. Exchange matrix-index bit 0, then bit 1.
// Every shuffle executes with the full wave64 (including xor32); selecting
// before shuffling would incorrectly use the source lane's register index.
__device__ __forceinline__ uint4 fp8moe_transpose(uint4 v, int s) {
    const unsigned x1 = __shfl_xor(v.y, 16, 64);
    const unsigned y1 = __shfl_xor(v.x, 16, 64);
    const unsigned z1 = __shfl_xor(v.w, 16, 64);
    const unsigned w1 = __shfl_xor(v.z, 16, 64);
    const uint4 a = (s & 1) ? uint4{x1, v.y, z1, v.w}
                            : uint4{v.x, y1, v.z, w1};
    const unsigned x2 = __shfl_xor(a.z, 32, 64);
    const unsigned y2 = __shfl_xor(a.w, 32, 64);
    const unsigned z2 = __shfl_xor(a.x, 32, 64);
    const unsigned w2 = __shfl_xor(a.y, 32, 64);
    return (s & 2) ? uint4{x2, y2, a.z, a.w}
                   : uint4{a.x, a.y, z2, w2};
}

template <bool XBF16, bool REFERENCE = false>
__device__ __forceinline__ void fp8moe_grouped_body(
    const unsigned char* __restrict__ W,
    unsigned long long expert_stride,
    const int4* __restrict__ tile_desc,
    const int* __restrict__ sorted_rows,
    const void* __restrict__ X,
    float* __restrict__ Y,
    int R, int K, int x_row_div, int n_exp) {
    constexpr int NSUB = FP8MOE_NSUB;
    constexpr int D = FP8MOE_D;
    // [D][16 rows][128 bf16 + 8 pad] as dwords; pad breaks the 256-B row stride.
    __shared__ unsigned xl[D][16][68];

    const int4 td = tile_desc[blockIdx.y];
    const int expert = td.x;
    const int row0 = td.y;
    const int nrows = td.z;
    // Fail closed: an expert id outside [0, n_exp) (-1 sentinel, or a non-finite-router / corrupt id)
    // skips the tile before `W + expert * expert_stride` is formed.
    if (expert < 0 || expert >= n_exp) return;
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
        wrow[j] = blk + (size_t)r * K + (REFERENCE ? 4 : 16) * s;
        srow[j] = sc + (min(row_base + j * 16, R - 1) >> 7) * nkb;
    }

    fp8moe_f32x4 acc[NSUB];
#pragma unroll
    for (int j = 0; j < NSUB; ++j) acc[j] = fp8moe_f32x4{0.0f, 0.0f, 0.0f, 0.0f};

    // Raw next-group operands: W stays contiguous (transposed at consumption),
    // each scale keeps its own register (no packing wait at load time), and F32 X
    // stays raw (x[d][0..1] = 8 floats) until LDS publication rounds it to BF16.
    // BF16 X uses x[d][0] only.
    uint4 wc[D][NSUB][2], wn[D][NSUB][2];
    unsigned scc[D][NSUB], scn[D][NSUB];
    uint4 xc[D][2], xn[D][2];

    auto load_x_group = [&](int kb0, uint4 (&xv)[D][2]) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
            const int kb = kb0 + d;
            if (kb < nkb) {
                if (xrow >= 0) {
                    const long long off = xrow * K + kb * 128 + st_k;
                    if constexpr (XBF16) {
                        xv[d][0] = *reinterpret_cast<const uint4*>(reinterpret_cast<const unsigned short*>(X) + off);
                    } else {
                        xv[d][0] = *reinterpret_cast<const uint4*>(reinterpret_cast<const float*>(X) + off);
                        xv[d][1] = *reinterpret_cast<const uint4*>(reinterpret_cast<const float*>(X) + off + 4);
                    }
                } else {
                    xv[d][0] = uint4{0u, 0u, 0u, 0u};
                    xv[d][1] = uint4{0u, 0u, 0u, 0u};
                }
            }
        }
    };

    // Identical BF16 words to the load-time conversion; +0.0f raw padding
    // rounds to the same zero words.
    auto publish_x = [&](const uint4 (&xv)[2]) -> uint4 {
        if constexpr (XBF16) {
            return xv[0];
        } else {
            const uint4 a = xv[0];
            const uint4 b = xv[1];
            return uint4{fp8moe_f32x2_to_bf16x2(__uint_as_float(a.x), __uint_as_float(a.y)),
                         fp8moe_f32x2_to_bf16x2(__uint_as_float(a.z), __uint_as_float(a.w)),
                         fp8moe_f32x2_to_bf16x2(__uint_as_float(b.x), __uint_as_float(b.y)),
                         fp8moe_f32x2_to_bf16x2(__uint_as_float(b.z), __uint_as_float(b.w))};
        }
    };

    // Production keeps W raw (contiguous uint4); transpose happens at consumption.
    auto load_w_group = [&](int kb0, uint4 (&w)[D][NSUB][2], unsigned (&sv)[D][NSUB]) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
            const int kb = kb0 + d;
            if (kb < nkb) {
#pragma unroll
                for (int j = 0; j < NSUB; ++j) {
                    const unsigned char* wp = wrow[j] + kb * 128;
                    if constexpr (REFERENCE) {
                        w[d][j][0] = uint4{*reinterpret_cast<const unsigned*>(wp), *reinterpret_cast<const unsigned*>(wp + 16),
                                           *reinterpret_cast<const unsigned*>(wp + 32), *reinterpret_cast<const unsigned*>(wp + 48)};
                        w[d][j][1] = uint4{*reinterpret_cast<const unsigned*>(wp + 64), *reinterpret_cast<const unsigned*>(wp + 80),
                                           *reinterpret_cast<const unsigned*>(wp + 96), *reinterpret_cast<const unsigned*>(wp + 112)};
                    } else {
                        w[d][j][0] = *reinterpret_cast<const uint4*>(wp);
                        w[d][j][1] = *reinterpret_cast<const uint4*>(wp + 64);
                    }
                    sv[d][j] = srow[j][kb];
                }
            }
        }
    };

    load_x_group(0, xc);
    load_w_group(0, wc, scc);
    for (int kb0 = 0; kb0 < nkb; kb0 += D) {
        __syncthreads();
#pragma unroll
        for (int d = 0; d < D; ++d) {
            if (kb0 + d < nkb) *reinterpret_cast<uint4*>(&xl[d][st_slot][st_k / 2]) = publish_x(xc[d]);
        }
        __syncthreads();
        if (kb0 + D < nkb) {
            load_x_group(kb0 + D, xn);
            load_w_group(kb0 + D, wn, scn);
        }
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
                        const float s2 = fp8moe_scale2((unsigned short)scc[d][j]);
                        uint4 w;
                        if constexpr (REFERENCE) {
                            w = wc[d][j][h];
                        } else {
                            w = fp8moe_transpose(wc[d][j][h], s);
                        }
                        unsigned a0, a1;
                        fp8moe_decode4(w.x, s2, a0, a1);
                        acc[j] = fp8moe_mfma(a0, a1, b0x, b0y, acc[j]);
                        fp8moe_decode4(w.y, s2, a0, a1);
                        acc[j] = fp8moe_mfma(a0, a1, b1x, b1y, acc[j]);
                        fp8moe_decode4(w.z, s2, a0, a1);
                        acc[j] = fp8moe_mfma(a0, a1, b2x, b2y, acc[j]);
                        fp8moe_decode4(w.w, s2, a0, a1);
                        acc[j] = fp8moe_mfma(a0, a1, b3x, b3y, acc[j]);
                    }
                }
            }
        }
#pragma unroll
        for (int d = 0; d < D; ++d) {
            if (kb0 + D + d < nkb) {
                xc[d][0] = xn[d][0];
                xc[d][1] = xn[d][1];
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
extern "C" __global__ __launch_bounds__(256) void gemm_fp8b128_moe_grouped_bf16mfma(
    const unsigned char* __restrict__ W,
    unsigned long long expert_stride,
    const int4* __restrict__ tile_desc,
    const int* __restrict__ sorted_rows,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int R, int K, int x_row_div, int n_exp) {
    fp8moe_grouped_body<false>(W, expert_stride, tile_desc, sorted_rows, X, Y, R, K, x_row_div, n_exp);
}

// X: [x_rows, K] BF16 bits.
extern "C" __global__ __launch_bounds__(256) void gemm_fp8b128_moe_grouped_bf16mfma_xbf16(
    const unsigned char* __restrict__ W,
    unsigned long long expert_stride,
    const int4* __restrict__ tile_desc,
    const int* __restrict__ sorted_rows,
    const unsigned short* __restrict__ X,
    float* __restrict__ Y,
    int R, int K, int x_row_div, int n_exp) {
    fp8moe_grouped_body<true>(W, expert_stride, tile_desc, sorted_rows, X, Y, R, K, x_row_div, n_exp);
}

// Retained cdc37659a transport/order for exact F32 A/B, absent from production
// code objects. The diagnostic harness alone defines FP8MOE_TEST_REFERENCE.
#if defined(FP8MOE_TEST_REFERENCE)
extern "C" __global__ __launch_bounds__(256) void gemm_fp8b128_moe_grouped_bf16mfma_reference(
    const unsigned char* __restrict__ W, unsigned long long expert_stride,
    const int4* __restrict__ tile_desc, const int* __restrict__ sorted_rows,
    const float* __restrict__ X, float* __restrict__ Y, int R, int K, int x_row_div, int n_exp) {
    fp8moe_grouped_body<false, true>(W, expert_stride, tile_desc, sorted_rows, X, Y, R, K, x_row_div, n_exp);
}
extern "C" __global__ __launch_bounds__(256) void gemm_fp8b128_moe_grouped_bf16mfma_reference_xbf16(
    const unsigned char* __restrict__ W, unsigned long long expert_stride,
    const int4* __restrict__ tile_desc, const int* __restrict__ sorted_rows,
    const unsigned short* __restrict__ X, float* __restrict__ Y, int R, int K, int x_row_div, int n_exp) {
    fp8moe_grouped_body<true, true>(W, expert_stride, tile_desc, sorted_rows, X, Y, R, K, x_row_div, n_exp);
}
#endif

// Tile descriptors for the padded-16 grouped layout (the stable scatter's
// `tiles` output: expert per 16-row tile, -1 tail): tile_desc[t] =
// {tile_ids[t], 16*t, 16, 0}. Grid [ceil(n_tiles/256)], block 256; one
// store per tile, no atomics.
extern "C" __global__ __launch_bounds__(256) void gemm_fp8b128_moe_tile_desc_pad16(
    const int* __restrict__ tile_ids,
    int4* __restrict__ tile_desc,
    int n_tiles) {
    const int t = blockIdx.x * 256 + threadIdx.x;
    if (t < n_tiles) tile_desc[t] = int4{tile_ids[t], 16 * t, 16, 0};
}
