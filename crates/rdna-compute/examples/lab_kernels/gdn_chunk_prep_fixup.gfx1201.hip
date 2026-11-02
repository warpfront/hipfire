// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// LAB-ONLY shape-parameterised copy of the production gdn_chunk_prep_fixup.gfx1201.hip.
// Shape macros (override with -D at runtime compile; defaults reproduce the
// production 16/48/ratio3/XDIM10240/UNITS80 build exactly):
//   GDN_LAB_HK    key heads                    (default 16)
//   GDN_LAB_HV    value heads                  (default 48)
//   GDN_LAB_RATIO value heads per key head     (default HV/HK)
//   GDN_LAB_UNITS 128-wide conv/QKV units      (default 2*HK+HV)
//   GDN_LAB_XDIM  conv channel row width       (default UNITS*128)
// A3B: -DGDN_LAB_HK=16 -DGDN_LAB_HV=32 -DGDN_LAB_RATIO=2 -DGDN_LAB_XDIM=8192 -DGDN_LAB_UNITS=64
#ifndef GDN_LAB_HK
#define GDN_LAB_HK 16
#endif
#ifndef GDN_LAB_HV
#define GDN_LAB_HV 48
#endif
#ifndef GDN_LAB_RATIO
#define GDN_LAB_RATIO ((GDN_LAB_HV) / (GDN_LAB_HK))
#endif
#ifndef GDN_LAB_UNITS
#define GDN_LAB_UNITS (2 * (GDN_LAB_HK) + (GDN_LAB_HV))
#endif
#ifndef GDN_LAB_XDIM
#define GDN_LAB_XDIM ((GDN_LAB_UNITS) * 128)
#endif
#if ((GDN_LAB_HK) * (GDN_LAB_RATIO) != (GDN_LAB_HV))
#error "GDN_LAB_HV must equal GDN_LAB_HK * GDN_LAB_RATIO"
#endif
#if ((GDN_LAB_UNITS) != 2 * (GDN_LAB_HK) + (GDN_LAB_HV))
#error "GDN_LAB_UNITS must equal 2*HK+HV"
#endif
#if ((GDN_LAB_XDIM) != (GDN_LAB_UNITS) * 128)
#error "GDN_LAB_XDIM must equal UNITS*128"
#endif

// Completion pass for the F2 QKVZA+GDN epilogue (`gemm_mq4g256v2_fp8_qkvzagdn_row_b1`).
// The fused GEMM writes q/k/v for every token except the first three of each
// 128-token tile after the first, and leaves raw projected rows in `x` only for
// tile positions 0..2 and 125..127 and the final three tokens. This pass runs
// the remaining pieces of `gdn_chunk_prep` with its exact expressions (same
// operand order and helpers; compiled with the same -ffp-contract=off):
//   blockIdx.y == 0: the C64 gate owner (softplus/exp prefix and sigmoid), and
//   blockIdx.y >= 1: the 3-token tile heads 128j..128j+2 (j >= 1) of one
//   Q/K/V unit per wave, and (blockIdx.x == 0) the persistent conv ring.
// Grid (max(ceil(T/64), ceil(T/128)), ceil(UNITS/8)+1), block 256 (default 11, A3B 9).
#define GDN_CHUNK_C 64
#define GDN_FUSED_TILE 128
#define GDN_CHUNK_HD 128
#define GDN_CHUNK_HK (GDN_LAB_HK)
#define GDN_CHUNK_HV (GDN_LAB_HV)
#define GDN_CHUNK_UNITS (GDN_LAB_UNITS)
#define GDN_CHUNK_XDIM (GDN_LAB_XDIM)

__device__ __forceinline__ float gdn_chunk_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return logf(1.0f + expf(x));
}

__device__ __forceinline__ unsigned short gdn_chunk_bf16(float x) {
    return __half_as_ushort(__float2half_rn(x));
}

extern "C" __launch_bounds__(256)
__global__ void gdn_chunk_prep_fixup(
    const float* __restrict__ x,
    const float* __restrict__ conv_w,
    float* __restrict__ conv_state,
    float* __restrict__ a,
    float* __restrict__ b,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    unsigned short* __restrict__ q,
    unsigned short* __restrict__ k,
    unsigned short* __restrict__ v,
    int T,
    float q_scale,
    float eps) {
    if (T <= 0) return;

    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    if (blockIdx.y == 0) {
        const int chunk0 = (int)blockIdx.x * GDN_CHUNK_C;
        int rows = T - chunk0;
        if (rows <= 0) return;
        if (rows > GDN_CHUNK_C) rows = GDN_CHUNK_C;
        for (int h = wave; h < GDN_CHUNK_HV; h += 8) {
            const int r0 = lane * 2;
            const int r1 = r0 + 1;
            float g0 = 0.0f;
            float g1 = 0.0f;
            if (r0 < rows) {
                const long long off = (long long)(chunk0 + r0) * GDN_CHUNK_HV + h;
                g0 = -expf(a_log[h]) * gdn_chunk_softplus(a[off] + dt_bias[h]);
                b[off] = 1.0f / (1.0f + expf(-b[off]));
            }
            if (r1 < rows) {
                const long long off = (long long)(chunk0 + r1) * GDN_CHUNK_HV + h;
                g1 = -expf(a_log[h]) * gdn_chunk_softplus(a[off] + dt_bias[h]);
                b[off] = 1.0f / (1.0f + expf(-b[off]));
            }
            float pair = g0 + g1;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const float other = __shfl_up(pair, d, 32);
                if (lane >= d) pair += other;
            }
            const float before = pair - g0 - g1;
            if (r0 < rows)
                a[(long long)(chunk0 + r0) * GDN_CHUNK_HV + h] = before + g0;
            if (r1 < rows)
                a[(long long)(chunk0 + r1) * GDN_CHUNK_HV + h] = before + g0 + g1;
        }
        return;
    }

    const int unit = ((int)blockIdx.y - 1) * 8 + wave;
    if (unit >= GDN_CHUNK_UNITS) return;
    const int dbase = unit * GDN_CHUNK_HD + lane * 4;
    const int region = unit < GDN_CHUNK_HK ? 0 : (unit < 2 * GDN_CHUNK_HK ? 1 : 2);
    const int head = region == 0 ? unit : (region == 1 ? unit - GDN_CHUNK_HK
                                                       : unit - 2 * GDN_CHUNK_HK);

    const int chunk0 = ((int)blockIdx.x + 1) * GDN_FUSED_TILE;
    if (chunk0 < T) {
        int rows = T - chunk0;
        if (rows > 3) rows = 3;

        float w[4][4];
#pragma unroll
        for (int c = 0; c < 4; ++c)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                w[j][c] = conv_w[(dbase + c) * 4 + j];

        float win[3][4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            win[0][c] = x[(long long)(chunk0 - 3) * GDN_CHUNK_XDIM + dbase + c];
            win[1][c] = x[(long long)(chunk0 - 2) * GDN_CHUNK_XDIM + dbase + c];
            win[2][c] = x[(long long)(chunk0 - 1) * GDN_CHUNK_XDIM + dbase + c];
        }

        for (int r = 0; r < rows; ++r) {
            float out4[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float cur = x[(long long)(chunk0 + r) * GDN_CHUNK_XDIM + dbase + c];
                float y = fmaf(w[3][c], cur, w[2][c] * win[2][c]);
                y = fmaf(w[1][c], win[1][c], y);
                y = fmaf(w[0][c], win[0][c], y);
                out4[c] = y / (1.0f + expf(-y));
                win[0][c] = win[1][c];
                win[1][c] = win[2][c];
                win[2][c] = cur;
            }

            if (region < 2) {
                float sum = 0.0f;
#pragma unroll
                for (int c = 0; c < 4; ++c) sum += out4[c] * out4[c];
#pragma unroll
                for (int d = 16; d > 0; d >>= 1) sum += __shfl_xor(sum, d, 32);
                const float inv = rsqrtf(sum + eps);
                unsigned short* dst = (region == 0 ? q : k)
                    + ((long long)(chunk0 + r) * GDN_CHUNK_HK + head) * GDN_CHUNK_HD
                    + lane * 4;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    float z = out4[c] * inv;
                    if (region == 0) z *= q_scale;
                    dst[c] = gdn_chunk_bf16(z);
                }
            } else {
                unsigned short* dst = v
                    + ((long long)(chunk0 + r) * GDN_CHUNK_HV + head) * GDN_CHUNK_HD
                    + lane * 4;
#pragma unroll
                for (int c = 0; c < 4; ++c) dst[c] = gdn_chunk_bf16(out4[c]);
            }
        }
    }

    // Same ring update as the first C64 owner of gdn_chunk_prep. The fused
    // GEMM consumed the old ring (tile 0's halo) before this pass launched.
    if (blockIdx.x == 0) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float old0 = conv_state[(dbase + c) * 3];
            const float old1 = conv_state[(dbase + c) * 3 + 1];
            conv_state[(dbase + c) * 3] = x[(long long)(T - 1) * GDN_CHUNK_XDIM + dbase + c];
            conv_state[(dbase + c) * 3 + 1] = T >= 2
                ? x[(long long)(T - 2) * GDN_CHUNK_XDIM + dbase + c] : old0;
            conv_state[(dbase + c) * 3 + 2] = T >= 3
                ? x[(long long)(T - 3) * GDN_CHUNK_XDIM + dbase + c]
                : (T == 2 ? old0 : old1);
        }
    }
}
