// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1151 small-M E8-SoA U4 GEMV with four-way K splitting.

// This candidate is intentionally narrow: K=4096 and M<=512. Four independent
// wave32s process one U4 quad apiece, increasing resident work for projection
// shapes that launch too few one-wave blocks to hide shared-memory latency.
// The partial layout lets wave 0 reconstruct the incumbent's exact accumulation
// order, so the output remains bit-identical rather than merely close.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float cvt_e4m3_scale_to_f32_soa_ks4(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) return 0.015625f * ((float)mant) * 0.125f;
    if (exp == 0xF && mant == 7u) return 448.0f;
    const float pow2 = __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + ((float)mant) * 0.125f);
}

__device__ __forceinline__ void e8_decode_index_soa_ks4(unsigned int idx, float p[8]) {
    const int coset = (int)((idx >> 31) & 1u);
    unsigned int e[8];
    unsigned int sl = 0u;
    for (int i = 0; i < 7; i++) {
        e[i] = (idx >> (unsigned)(4 * i)) & 0xFu;
        sl += e[i];
    }
    const unsigned int e7_high = (idx >> 28) & 0x7u;
    const unsigned int p7 = e7_high << 1;
    const unsigned int lsb = (sl + p7) & 1u;
    e[7] = p7 | lsb;
    for (int i = 0; i < 8; i++) {
        const float c = (float)((int)e[i] - 7);
        p[i] = coset ? c + 0.5f : c;
    }
}

__launch_bounds__(128)
extern "C" __global__ void gemv_mfp4g32_e8_soa_u4_ksplit4_gfx1151(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K
) {
    const int row = blockIdx.x;
    if (row >= M || K != 4096) return;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int n_blocks = K / 32;
    const int scale_padded = ((n_blocks + 15) >> 4) << 4;
    const long long row_total = 16LL + scale_padded + (long long)n_blocks * 16;
    const char* hdr = A + (long long)row * row_total;
    const float row_scale_f =
        (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);
    const char* scale_base = hdr + 16;
    const char* cw_base = hdr + 16 + scale_padded;
    const int block_in_group = lane >> 2;
    const int cw_slot = lane & 3;

    // K=4096 has exactly 16 groups = four U4 quads. One wave owns one
    // quad, then exposes each lane's four independent incumbent accumulators.
    const int g = wave << 2;
    const int b0 = g * 8 + block_in_group;
    const int b1 = b0 + 8;
    const int b2 = b0 + 16;
    const int b3 = b0 + 24;

    const float ssc0 = row_scale_f
        * cvt_e4m3_scale_to_f32_soa_ks4(*(const unsigned char*)(scale_base + b0))
        * 0.88f;
    const float ssc1 = row_scale_f
        * cvt_e4m3_scale_to_f32_soa_ks4(*(const unsigned char*)(scale_base + b1))
        * 0.88f;
    const float ssc2 = row_scale_f
        * cvt_e4m3_scale_to_f32_soa_ks4(*(const unsigned char*)(scale_base + b2))
        * 0.88f;
    const float ssc3 = row_scale_f
        * cvt_e4m3_scale_to_f32_soa_ks4(*(const unsigned char*)(scale_base + b3))
        * 0.88f;

    const unsigned int ew0 =
        *(const unsigned int*)(cw_base + (long long)b0 * 16 + cw_slot * 4);
    const unsigned int ew1 =
        *(const unsigned int*)(cw_base + (long long)b1 * 16 + cw_slot * 4);
    const unsigned int ew2 =
        *(const unsigned int*)(cw_base + (long long)b2 * 16 + cw_slot * 4);
    const unsigned int ew3 =
        *(const unsigned int*)(cw_base + (long long)b3 * 16 + cw_slot * 4);

    const int base0 = g * 256 + lane * 8;
    const int base1 = base0 + 256;
    const int base2 = base0 + 512;
    const int base3 = base0 + 768;
    float pe[8];

    e8_decode_index_soa_ks4(ew0, pe);
    const float acc0 =
        ssc0 * pe[0] * x[base0] + ssc0 * pe[1] * x[base0 + 1]
        + ssc0 * pe[2] * x[base0 + 2] + ssc0 * pe[3] * x[base0 + 3]
        + ssc0 * pe[4] * x[base0 + 4] + ssc0 * pe[5] * x[base0 + 5]
        + ssc0 * pe[6] * x[base0 + 6] + ssc0 * pe[7] * x[base0 + 7];
    e8_decode_index_soa_ks4(ew1, pe);
    const float acc1 =
        ssc1 * pe[0] * x[base1] + ssc1 * pe[1] * x[base1 + 1]
        + ssc1 * pe[2] * x[base1 + 2] + ssc1 * pe[3] * x[base1 + 3]
        + ssc1 * pe[4] * x[base1 + 4] + ssc1 * pe[5] * x[base1 + 5]
        + ssc1 * pe[6] * x[base1 + 6] + ssc1 * pe[7] * x[base1 + 7];
    e8_decode_index_soa_ks4(ew2, pe);
    const float acc2 =
        ssc2 * pe[0] * x[base2] + ssc2 * pe[1] * x[base2 + 1]
        + ssc2 * pe[2] * x[base2 + 2] + ssc2 * pe[3] * x[base2 + 3]
        + ssc2 * pe[4] * x[base2 + 4] + ssc2 * pe[5] * x[base2 + 5]
        + ssc2 * pe[6] * x[base2 + 6] + ssc2 * pe[7] * x[base2 + 7];
    e8_decode_index_soa_ks4(ew3, pe);
    const float acc3 =
        ssc3 * pe[0] * x[base3] + ssc3 * pe[1] * x[base3 + 1]
        + ssc3 * pe[2] * x[base3 + 2] + ssc3 * pe[3] * x[base3 + 3]
        + ssc3 * pe[4] * x[base3 + 4] + ssc3 * pe[5] * x[base3 + 5]
        + ssc3 * pe[6] * x[base3 + 6] + ssc3 * pe[7] * x[base3 + 7];

    __shared__ float partial[4][4][32];
    partial[wave][0][lane] = acc0;
    partial[wave][1][lane] = acc1;
    partial[wave][2][lane] = acc2;
    partial[wave][3][lane] = acc3;
    __syncthreads();

    if (wave == 0) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            sum0 += partial[q][0][lane];
            sum1 += partial[q][1][lane];
            sum2 += partial[q][2][lane];
            sum3 += partial[q][3][lane];
        }
        float acc = sum0 + sum1 + sum2 + sum3;
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (lane == 0) y[row] = acc;
    }
}
