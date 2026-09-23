// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_MIXED_GATEUP_LPG
#define HIPFIRE_MIXED_GATEUP_LPG 8
#endif

#if HIPFIRE_MIXED_GATEUP_LPG == 8
#define HIPFIRE_MIXED_GATEUP_SPATIAL_NAME \
    gemv_mixed_moe_gate_up_swiglu_k8_indexed_batched_spatial_lpg8_gfx12
#elif HIPFIRE_MIXED_GATEUP_LPG == 16
#define HIPFIRE_MIXED_GATEUP_SPATIAL_NAME \
    gemv_mixed_moe_gate_up_swiglu_k8_indexed_batched_spatial_lpg16_gfx12
#elif HIPFIRE_MIXED_GATEUP_LPG == 32
#define HIPFIRE_MIXED_GATEUP_SPATIAL_NAME \
    gemv_mixed_moe_gate_up_swiglu_k8_indexed_batched_spatial_lpg32_gfx12
#else
#error "HIPFIRE_MIXED_GATEUP_LPG must be 8, 16, or 32"
#endif

// Spatial-K wave64 mixed-expert gate+up/SwiGLU.  A wave is divided into
// gate/up subgroup pairs. LPG=8 owns four output rows per wave, LPG=16 owns
// two, and LPG=32 is the one-row control. Each subgroup covers all 256 values
// of a quant group and reduces only within its LPG partition.
extern "C" __launch_bounds__(64, 8)
__global__ void HIPFIRE_MIXED_GATEUP_SPATIAL_NAME(
    const unsigned long long* __restrict__ expert_ptrs,
    const unsigned char* __restrict__ dtype_tags,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x,
    float* __restrict__ y_hidden,
    int MI, int K, int K_TOP
) {
    constexpr int LPG = HIPFIRE_MIXED_GATEUP_LPG;
    constexpr int ELEMS = 256 / LPG;
    constexpr int ROWS_PER_WAVE = 64 / (2 * LPG);

    const int lane = threadIdx.x;
    const int subgroup = lane / LPG;
    const int lane_in = lane & (LPG - 1);
    const int row_slot = subgroup >> 1;
    const bool is_up = (subgroup & 1) != 0;
    const int row = blockIdx.x * ROWS_PER_WAVE + row_slot;
    const bool active = row < MI;
    const int safe_row = active ? row : MI - 1;
    const int matrix_row = safe_row + (is_up ? MI : 0);
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const unsigned char tag = dtype_tags[expert_id];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x_row = x + (long long)bid * K;
    const int groups = K / 256;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    __shared__ float codebooks[8 * 8];

    if (tag == 0) {
        // MQ6 affine: 32, 16, or 8 values per lane; four values per 3 bytes.
        for (int g = 0; g < groups; ++g) {
            const char* gp = A + ((long long)matrix_row * groups + g) * 200;
            const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
            const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            const unsigned char* d =
                (const unsigned char*)(gp + 8) + lane_in * (ELEMS * 6 / 8);
            const int xb = g * 256 + lane_in * ELEMS;
            float dot = 0.0f;
#pragma unroll
            for (int j = 0; j < ELEMS / 4; ++j) {
                const unsigned char b0 = d[j * 3];
                const unsigned char b1 = d[j * 3 + 1];
                const unsigned char b2 = d[j * 3 + 2];
                const int q0 = b0 & 63;
                const int q1 = (b0 >> 6) | ((b1 & 15) << 2);
                const int q2 = (b1 >> 4) | ((b2 & 3) << 4);
                const int q3 = b2 >> 2;
                const int xi = xb + j * 4;
                dot += (sc * (float)q0 + zp) * x_row[xi]
                     + (sc * (float)q1 + zp) * x_row[xi + 1]
                     + (sc * (float)q2 + zp) * x_row[xi + 2]
                     + (sc * (float)q3 + zp) * x_row[xi + 3];
            }
            if ((g & 3) == 0) a0 += dot;
            else if ((g & 3) == 1) a1 += dot;
            else if ((g & 3) == 2) a2 += dot;
            else a3 += dot;
        }
    } else if (tag == 2) {
        // MQ4 affine: two values per packed byte.
        for (int g = 0; g < groups; ++g) {
            const char* gp = A + ((long long)matrix_row * groups + g) * 136;
            const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
            const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            const unsigned char* d =
                (const unsigned char*)(gp + 8) + lane_in * (ELEMS / 2);
            const int xb = g * 256 + lane_in * ELEMS;
            float dot = 0.0f;
#pragma unroll
            for (int j = 0; j < ELEMS / 2; ++j) {
                const unsigned int p = d[j];
                const int xi = xb + j * 2;
                dot += (sc * (float)(p & 15u) + zp) * x_row[xi]
                     + (sc * (float)(p >> 4) + zp) * x_row[xi + 1];
            }
            if ((g & 3) == 0) a0 += dot;
            else if ((g & 3) == 1) a1 += dot;
            else if ((g & 3) == 2) a2 += dot;
            else a3 += dot;
        }
    } else if (tag == 3) {
        // MQ3 Lloyd: one subgroup-local eight-entry fp16 codebook per row/group.
        for (int g = 0; g < groups; ++g) {
            const char* gp = A + ((long long)matrix_row * groups + g) * 112;
            if (lane_in < 8) {
                const unsigned short hc = *(const unsigned short*)(gp + lane_in * 2);
                codebooks[subgroup * 8 + lane_in] =
                    __half2float(__builtin_bit_cast(__half, hc));
            }
            __syncthreads();
            const unsigned char* d =
                (const unsigned char*)(gp + 16) + lane_in * (ELEMS * 3 / 8);
            const int xb = g * 256 + lane_in * ELEMS;
            const float* cb = codebooks + subgroup * 8;
            float dot = 0.0f;
#pragma unroll
            for (int j = 0; j < ELEMS / 8; ++j) {
                const unsigned int p = (unsigned int)d[j * 3]
                    | ((unsigned int)d[j * 3 + 1] << 8)
                    | ((unsigned int)d[j * 3 + 2] << 16);
                const int xi = xb + j * 8;
                dot += cb[p & 7u] * x_row[xi]
                     + cb[(p >> 3) & 7u] * x_row[xi + 1]
                     + cb[(p >> 6) & 7u] * x_row[xi + 2]
                     + cb[(p >> 9) & 7u] * x_row[xi + 3]
                     + cb[(p >> 12) & 7u] * x_row[xi + 4]
                     + cb[(p >> 15) & 7u] * x_row[xi + 5]
                     + cb[(p >> 18) & 7u] * x_row[xi + 6]
                     + cb[(p >> 21) & 7u] * x_row[xi + 7];
            }
            __syncthreads();
            if ((g & 3) == 0) a0 += dot;
            else if ((g & 3) == 1) a1 += dot;
            else if ((g & 3) == 2) a2 += dot;
            else a3 += dot;
        }
    }

    float acc = (a0 + a1) + (a2 + a3);
    for (int offset = LPG / 2; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset, LPG);

    const int pair_leader = (subgroup & ~1) * LPG;
    const float up = __shfl(acc, pair_leader + LPG, 64);
    if (active && !is_up && lane_in == 0) {
        const long long out = ((long long)bid * K_TOP + krank) * MI + row;
        y_hidden[out] = (acc / (1.0f + expf(-acc))) * up;
    }
}

#undef HIPFIRE_MIXED_GATEUP_SPATIAL_NAME
