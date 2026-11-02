// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Depthwise causal short convolution over N rows with an explicit raw-input
// history, optional SiLU and optional single BF16 rounding of the output.
//
//   padded = [hist[0..K-2] (oldest -> newest), in[0..N-1]]
//   out[r,c] = act( sum_j w[c,j] * padded[r+j, c] )      j = 0 (oldest) .. K-1 (newest)
//   hist_out[j,c] = padded[N+j, c]                        (raw, pre-conv values)
//
// FP32 accumulation in tap order oldest -> newest, no bias. `hist_in` and
// `hist_out` may alias only when the launch uses a single row chunk
// (rows_per_chunk >= n_rows); the launcher enforces that.
// Weights are [channel, tap] with explicit strides. K <= 8.

#include <hip/hip_runtime.h>

#define CCD_KMAX 8

static __device__ __forceinline__ float ccd_bf16_rn(float x) {
    unsigned u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

extern "C" __global__ void causal_conv_depthwise_silu_rows_f32(
    float* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ weight,
    const float* hist_in,
    float* hist_out,
    int n_rows,
    int n_channels,
    int ksize,
    int rows_per_chunk,
    int in_row_stride,
    int out_row_stride,
    int w_ch_stride,
    int w_tap_stride,
    int hist_tap_stride,
    int apply_silu,
    int round_bf16
) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;
    const int km1 = ksize - 1;
    const int r0 = blockIdx.y * rows_per_chunk;
    int r1 = r0 + rows_per_chunk;
    if (r1 > n_rows) r1 = n_rows;

    float w[CCD_KMAX];
    float win[CCD_KMAX];
#pragma unroll
    for (int j = 0; j < CCD_KMAX; j++) {
        w[j] = (j < ksize) ? weight[(long long)c * w_ch_stride + (long long)j * w_tap_stride] : 0.0f;
        float v = 0.0f;
        if (j < km1) {
            const int p = r0 + j;  // padded index of window slot j
            v = (p < km1) ? hist_in[(long long)p * hist_tap_stride + c]
                          : in[(long long)(p - km1) * in_row_stride + c];
        }
        win[j] = v;
    }

    for (int r = r0; r < r1; r++) {
        const float x = in[(long long)r * in_row_stride + c];
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < CCD_KMAX - 1; j++) {
            if (j < km1) acc = __fadd_rn(acc, __fmul_rn(w[j], win[j]));
        }
        acc = __fadd_rn(acc, __fmul_rn(w[km1], x));
        float y = apply_silu ? acc / (1.0f + expf(-acc)) : acc;
        out[(long long)r * out_row_stride + c] = round_bf16 ? ccd_bf16_rn(y) : y;
#pragma unroll
        for (int j = 0; j < CCD_KMAX - 2; j++) {
            if (j + 1 < km1) win[j] = win[j + 1];
        }
        win[km1 - 1] = x;
    }

    if (r1 == n_rows) {
        for (int j = 0; j < km1; j++) {
            const int p = n_rows + j;
            float v = (p < km1) ? hist_in[(long long)p * hist_tap_stride + c]
                                : in[(long long)(p - km1) * in_row_stride + c];
            hist_out[(long long)j * hist_tap_stride + c] = v;
        }
    }
}
