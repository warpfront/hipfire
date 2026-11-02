// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — mfp4-E8 (qt=42, 128-wide rotation) decode GEMV, generic wave32/64.
//
// Why this exists: the shipped E8 GEMV family walks the row as `K / 256` groups
// of 8 blocks (32 threads x 8 weights = 256 K-elements per group), which covers
// a reduction axis only when K % 256 == 0. qwen4's routed-expert `down_proj`
// reduces over K = 640 = 5 x 128, which 256 cannot tile at all.
//
// The fix here is deliberately NOT a 128-wide thread mapping. The group is a
// work-mapping convenience, not a wire-format property: the row layout is
// `[16 B header][(K/32) x 17 B blocks]` regardless of how the encoder chose to
// segment its FWHT. So this kernel keeps the shipped 8-block group and adds a
// remainder pass for the `(K/32) % 8` trailing blocks. That covers ANY K that is
// a multiple of 32, and for K % 256 == 0 the remainder is empty and the executed
// instruction stream is the shipped group loop.
//
// Byte-identical decode to gemv_mfp4g32_e8.hip for the same weight data and the
// same rotated activation: same E4M3 scale LUT, same E8 lattice decode, same
// QUANT_STEP=0.88 fold, same shfl reduction order over the full groups.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float cvt_e4m3_scale_to_f32(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) return 0.015625f * ((float)mant) * 0.125f;
    if (exp == 0xF && mant == 7u) return 448.0f;
    const float pow2 = __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + ((float)mant) * 0.125f);
}

__device__ __forceinline__ void e8_decode_index(unsigned int idx, float p[8]) {
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
        float c = (float)((int)e[i] - 7);
        p[i] = coset ? c + 0.5f : c;
    }
}

__launch_bounds__(32, 7)
extern "C" __global__ void gemv_mfp4g32_e8g128(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    // Coverage: groups_per_row whole 8-block groups + rem_blocks trailing
    // blocks. groups_per_row * 256 + rem_blocks * 32 == K exactly.
    const int n_blocks       = K / 32;
    const int groups_per_row = n_blocks >> 3;
    const int rem_blocks     = n_blocks & 7;
    const int row_block_bytes = n_blocks * 17;
    const long long row_off = (long long)row * (16 + (long long)row_block_bytes);
    const char* hdr = A + row_off;
    const char* row_ptr = hdr + 16;

    const float row_scale_f = (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);

    const int block_in_iter = tid >> 2;
    const int cw_off = (tid & 3) * 4;
    const int thread_block_off = block_in_iter * 17;

    float acc0 = 0.0f, acc1 = 0.0f;
    const int pairs = groups_per_row >> 1;
    const int tail  = groups_per_row & 1;

    for (int p = 0; p < pairs; p++) {
        const int g = p << 1;
        const char* gp0 = row_ptr + (long long)g * 136 + thread_block_off;
        const char* gp1 = gp0 + 136;

        const float ssc0 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp0) * 0.88f;
        const float ssc1 = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp1) * 0.88f;

        const unsigned int ew0 = *(const unsigned int*)(gp0 + 1 + cw_off);
        const unsigned int ew1 = *(const unsigned int*)(gp1 + 1 + cw_off);

        const int base0 = g * 256 + tid * 8;
        const int base1 = base0 + 256;

        float pe[8];
        e8_decode_index(ew0, pe);
        acc0 += ssc0*pe[0]*x[base0  ] + ssc0*pe[1]*x[base0+1]
              + ssc0*pe[2]*x[base0+2] + ssc0*pe[3]*x[base0+3]
              + ssc0*pe[4]*x[base0+4] + ssc0*pe[5]*x[base0+5]
              + ssc0*pe[6]*x[base0+6] + ssc0*pe[7]*x[base0+7];

        e8_decode_index(ew1, pe);
        acc1 += ssc1*pe[0]*x[base1  ] + ssc1*pe[1]*x[base1+1]
              + ssc1*pe[2]*x[base1+2] + ssc1*pe[3]*x[base1+3]
              + ssc1*pe[4]*x[base1+4] + ssc1*pe[5]*x[base1+5]
              + ssc1*pe[6]*x[base1+6] + ssc1*pe[7]*x[base1+7];
    }

    if (tail) {
        const int g = pairs << 1;
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        const int base = g * 256 + tid * 8;
        float pe[8];
        e8_decode_index(ew, pe);
        acc0 += ssc*pe[0]*x[base  ] + ssc*pe[1]*x[base+1]
              + ssc*pe[2]*x[base+2] + ssc*pe[3]*x[base+3]
              + ssc*pe[4]*x[base+4] + ssc*pe[5]*x[base+5]
              + ssc*pe[6]*x[base+6] + ssc*pe[7]*x[base+7];
    }

    // Remainder: the trailing rem_blocks blocks, addressed as a partial group.
    // Thread `tid` covers block (tid>>2) slot (tid&3), i.e. x elements
    // [groups_per_row*256 + tid*8, +8), so a thread whose block lies past the
    // remainder contributes nothing and is masked out. The x index stays inside
    // [0, K) for every unmasked thread, which is what makes the coverage exact
    // rather than approximate.
    if (rem_blocks != 0 && block_in_iter < rem_blocks) {
        const char* gp = row_ptr + (long long)groups_per_row * 136 + thread_block_off;
        const float ssc = row_scale_f * cvt_e4m3_scale_to_f32(*(const unsigned char*)gp) * 0.88f;
        const unsigned int ew = *(const unsigned int*)(gp + 1 + cw_off);
        const int base = groups_per_row * 256 + tid * 8;
        float pe[8];
        e8_decode_index(ew, pe);
        acc0 += ssc*pe[0]*x[base  ] + ssc*pe[1]*x[base+1]
              + ssc*pe[2]*x[base+2] + ssc*pe[3]*x[base+3]
              + ssc*pe[4]*x[base+4] + ssc*pe[5]*x[base+5]
              + ssc*pe[6]*x[base+6] + ssc*pe[7]*x[base+7];
    }

    float acc = acc0 + acc1;
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
