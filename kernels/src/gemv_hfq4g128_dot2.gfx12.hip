// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) HFQ4-G128 GEMV using __builtin_amdgcn_fdot2 (FP16x2 paired
// multiply, 2 FMAs per VALU). Mirror of the gfx11 dot2 family already
// shipped for HFQ3G256 / HFQ4G256 / HFQ6G256 / HFP4G32, ported to the
// HFQ4G128 (group=128) format that PARO LA decode uses.
//
// MOTIVATION (user observation 2026-05-22): the production-clean decode
// stack at 98.8 tok/s on z-lab A3B-PARO has gemv_hfq4g128 at 25 percent
// of GPU time — the largest remaining single slot. The kernel was using
// scalar F32×F32 FMA in the inner loop. The dot2 instruction does 2 FP16
// muls + FP32 add per VALU, cutting the inner-loop multiply count ~2×
// per group.
//
// Layout (unchanged from gemv_hfq4g128.hip):
//   Per row: groups_per_row × 72 bytes per group.
//   Per group: 4 bytes scale F32 + 4 bytes zero F32 + 64 bytes nibbles.
//   Each WG handles one output row. 32 threads per WG.
//
// Per group per thread (4 weights → 2 fdot2 calls):
//   Read 2 nibble bytes b0, b1 → 4 weights v0..v3
//   Dequant to F16 in pairs: (v0, v1), (v2, v3)
//   X already F32 → convert pairs to half2 inline
//   Accumulate: acc += fdot2((v0, v1), (x0, x1)) + fdot2((v2, v3), (x2, x3))
//
// rocprof expected: per-call 16.9 µs → ~10 µs (2× compute density on the
// inner-loop FMA). Per token: 131 calls × ~7 µs saved = ~900 µs/token
// wall savings = ~+8 percent decode.

#if defined(__gfx1200__) || defined(__gfx1201__)

typedef _Float16 __attribute__((ext_vector_type(2))) half2_t;

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq4g128_dot2_gfx12(
    const char*  __restrict__ A,
    const float* __restrict__ x,
    float*       __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int byte_off = tid * 2;

    float acc = 0.0f;
    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_ptr + g * 72;
        const float sc_f = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp_f = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16 sc = (_Float16)sc_f;
        const _Float16 zp = (_Float16)zp_f;

        const unsigned char b0 = *(const unsigned char*)(gp + 8 + byte_off);
        const unsigned char b1 = *(const unsigned char*)(gp + 8 + byte_off + 1);

        // Dequantize 4 nibbles to F16 in pairs.
        const _Float16 v0 = sc * (_Float16)(float)(b0 & 0xF) + zp;
        const _Float16 v1 = sc * (_Float16)(float)(b0 >> 4)  + zp;
        const _Float16 v2 = sc * (_Float16)(float)(b1 & 0xF) + zp;
        const _Float16 v3 = sc * (_Float16)(float)(b1 >> 4)  + zp;

        const int base_idx = g * 128 + tid * 4;
        const _Float16 x0 = (_Float16)x[base_idx];
        const _Float16 x1 = (_Float16)x[base_idx + 1];
        const _Float16 x2 = (_Float16)x[base_idx + 2];
        const _Float16 x3 = (_Float16)x[base_idx + 3];

        half2_t w01 = {v0, v1};
        half2_t w23 = {v2, v3};
        half2_t x01 = {x0, x1};
        half2_t x23 = {x2, x3};

        // 2 fdot2 instructions = 4 FMAs in F32 accumulator.
        acc = __builtin_amdgcn_fdot2(w01, x01, acc, false);
        acc = __builtin_amdgcn_fdot2(w23, x23, acc, false);
    }

    // Wave32 shfl_down reduce.
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }
    if (tid == 0) y[row] = acc;
}

#else

extern "C" __global__ void gemv_hfq4g128_dot2_gfx12(
    const char*, const float*, float*, int, int
) {}

#endif
