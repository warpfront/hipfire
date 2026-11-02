// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// HIP twin of the PeaceMaker kernel `gemm_fp8b128_moe_grouped_fp8mfma_pm`
// (crates/peacemaker-isa/src/kernels/cdna/fp8_moe_gate_up.rs): FP8 x FP8
// grouped MoE GEMM tile on v_mfma_f32_16x16x32_fp8_fp8, gfx942 (MI300X,
// wave64). Same ABI, LDS layout, lane mapping and numerical order; the module
// doc of the PM kernel is the specification. Compiled offline, not run.
//
// ABI (72 kernarg bytes): W, expert_stride (u64), tile_desc (i32x4 {expert,
// row0, nrows, 0}), sorted_rows (i32, -1 = padding), X (FNUZ e4m3 codes
// [x_rows, K] from the activation producer: cvt_pk_fp8_f32((x/a)*0.5)),
// x_scale (f32 [x_rows], power-of-two a), Y (f32 [n_sorted, R]), R, K,
// x_row_div, n_experts (u32; a tile whose expert is outside [0, n_experts) is skipped).
// Grid [R/64, n_tiles], block 256. Contract: K % 128 == 0,
// R % 64 == 0, nrows 1..16, weight codes never 0x7F/0xFF, activation codes
// never 0x80, finite scales.
//
// Weights: qt=55 expert block at W + e*expert_stride: OCP e4m3fn codes
// [R, K], then BF16 s_inv [R/128, K/128] (a multiplier). The stored bytes are
// fed to the FNUZ MFMA after an in-register remap 0x80 (OCP -0, FNUZ NaN) ->
// 0x00; every other finite OCP byte b reads as FNUZ fn(b)/2.
//
// Order per output: total = +0; for each 128-K block ascending: partial =
// four chained MFMAs (K = 32 each, the first with C = 0), then
// total = fma(partial, (4*a_row)*s_w, total). No split-K, no atomics.
//
// MFMA layout (CDNA3 ISA 7.1.4, K_L = 8): lane l, n = l%16, s = l/16:
//   A (weights, transposed product) bytes i: W[ob + n, 128kb + 32t + 8s + i]
//   B (activations) bytes i: X[slot n, 128kb + 32t + 8s + i]
//   D item r: output column ob + 4s + r of slot n.

#include <hip/hip_runtime.h>

#pragma clang fp contract(off)

typedef float fp8mfma_f32x4 __attribute__((ext_vector_type(4)));

#define FP8MFMA_LDS_ROW 144

// Clear the sign bit of each byte whose low seven bits are zero (0x80 -> 0x00).
__device__ __forceinline__ unsigned fp8mfma_remap(unsigned w) {
    const unsigned c = 0x7F7F7F7Fu;
    return w & (((w & c) + c) | c);
}

__device__ __forceinline__ long fp8mfma_pair(unsigned lo, unsigned hi) {
    return (long)(((unsigned long long)hi << 32) | lo);
}

extern "C" __global__ __launch_bounds__(256) void gemm_fp8b128_moe_grouped_fp8mfma(
    const unsigned char* __restrict__ W,
    unsigned long long expert_stride,
    const int4* __restrict__ tile_desc,
    const int* __restrict__ sorted_rows,
    const unsigned char* __restrict__ X,
    const float* __restrict__ x_scale,
    float* __restrict__ Y,
    int R, int K, int x_row_div, int n_experts) {
    // [2 buffers][16 rows][128 codes + 16 pad]
    __shared__ __attribute__((aligned(16))) unsigned char xl[2][16][FP8MFMA_LDS_ROW];

    const int4 td = tile_desc[blockIdx.y];
    const int expert = td.x;
    const int row0 = td.y;
    const int nrows = td.z;
    // Fail closed before W + expert * expert_stride is formed (-1 sentinel or an out-of-range id).
    if (expert < 0 || expert >= n_experts) return;
    const unsigned char* __restrict__ blk = W + (unsigned long long)expert * expert_stride;
    const unsigned short* __restrict__ sc = reinterpret_cast<const unsigned short*>(blk + (size_t)R * K);
    const int nkb = K >> 7;

    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int n = lane & 15;
    const int s = lane >> 4;
    const int slot = tid >> 4;
    const int ob = blockIdx.x * 64 + wave * 16;

    const int flat_n = n < nrows ? sorted_rows[row0 + n] : -1;
    const int flat_s = slot < nrows ? sorted_rows[row0 + slot] : -1;
    const float a = flat_n >= 0 ? x_scale[(unsigned)flat_n / (unsigned)x_row_div] : 1.0f;
    const float fa = 4.0f * a;
    const unsigned char* __restrict__ xsrc =
        flat_s >= 0 ? X + (size_t)((unsigned)flat_s / (unsigned)x_row_div) * K + 8 * n : nullptr;
    const unsigned char* __restrict__ wrow = blk + (size_t)(ob + n) * K + 8 * s;
    const unsigned short* __restrict__ srow = sc + (ob >> 7) * nkb;

    fp8mfma_f32x4 total = {0.0f, 0.0f, 0.0f, 0.0f};
    uint2 xv = make_uint2(0u, 0u);
    uint2 wv[4];
    unsigned short sv;

    auto load = [&](int kb) {
        if (xsrc) xv = *reinterpret_cast<const uint2*>(xsrc + kb * 128);
#pragma unroll
        for (int t = 0; t < 4; ++t) wv[t] = *reinterpret_cast<const uint2*>(wrow + kb * 128 + 32 * t);
        sv = srow[kb];
    };

    load(0);
    for (int kb = 0; kb < nkb; ++kb) {
        unsigned char(*buf)[FP8MFMA_LDS_ROW] = xl[kb & 1];
        *reinterpret_cast<uint2*>(&buf[slot][8 * n]) = xv;
        long wa[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) wa[t] = fp8mfma_pair(fp8mfma_remap(wv[t].x), fp8mfma_remap(wv[t].y));
        const float scale = fa * __uint_as_float((unsigned)sv << 16);
        load(kb + 1 < nkb ? kb + 1 : kb);
        __syncthreads();
        long xb[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) xb[t] = *reinterpret_cast<const long*>(&buf[n][32 * t + 8 * s]);
        fp8mfma_f32x4 p = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int t = 0; t < 4; ++t) p = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(wa[t], xb[t], p, 0, 0, 0);
#pragma unroll
        for (int r = 0; r < 4; ++r) total[r] = __builtin_fmaf(p[r], scale, total[r]);
    }

    if (n < nrows) {
        float* __restrict__ y = Y + (size_t)(row0 + n) * R + ob + 4 * s;
        *reinterpret_cast<float4*>(y) = flat_n < 0 ? make_float4(0.0f, 0.0f, 0.0f, 0.0f)
                                                : make_float4(total[0], total[1], total[2], total[3]);
    }
}
