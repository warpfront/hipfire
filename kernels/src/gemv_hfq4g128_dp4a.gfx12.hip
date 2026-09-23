// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) HFQ4-G128 GEMV using __builtin_amdgcn_sdot4 (signed i8x4 → i32
// dot, 4 muls + adds per VALU). Mirror of the gfx906 / gfx942 wave64_dp4a
// family adapted to wave32 and HFQ4G128 layout.
//
// Pre-pass: caller must call ensure_q8_1_mmq_x(x, 1, K) to pre-quantize X
// to the Q8_1 mmq block format (144 bytes per 128-K block: 4 × half2 ds4
// scales + 128 int8 values). The returned Xq pointer is what we read here.
//
// Math:
//   For each HFQ4G128 group (128 K-elements):
//     nibbles are uint4 0..15. We treat them as signed (nib - 8) for sdot4.
//     w[k] = scale * (nib - 8) + (zero + 8 * scale)
//     i.e. w[k] = scale * nib_signed + zp_eff   where zp_eff = zero + 8*scale
//
//   Q8_1 mmq layout: 4 sub-blocks × 32 K each, with per-sub-block scale
//   d and sum_x: x[k] = d_sub * x_int8_signed[k] (with sum_x for the
//   per-sub-block sum needed for the zp_eff term).
//
//   Per K-element: w[k] * x[k]
//   = (scale * nib_signed + zp_eff) * (d_sub * x_int8)
//   = scale * d_sub * (nib_signed · x_int8)  -- via sdot4
//   + zp_eff * d_sub * sum(x_int8)            -- via per-sub-block sum_x
//
// Wave32 mapping (32 lanes):
//   Per group (128 K-elements):
//     4 K per lane → 1 sdot4 call per lane per group.
//   Sub-block index s = (lane / 4) % 4  (since 4 lanes × 4 K = 16 K per sub-block,
//   8 sub-blocks per group ... wait — actually for G128 a "sub-block" is 32 K).
//   With 128 K / 32 (sub-block) = 4 sub-blocks per HFQ4G128 group.
//   Lane covers K = lane * 4 .. lane * 4 + 3.
//   Sub-block index = lane / 8 (lanes 0..7 → sub-block 0, etc.).
//
// rocprof attribution motivating this kernel:
//   gemv_hfq4g128 (post-dot2 ship) at ~25 percent of decode GPU time.
//   dot2 = 2 FMA/VALU. sdot4 = 4 FMA/VALU. Expected ~2× speedup on
//   compute-bound HFQ4G128 GEMV decode.

#if defined(__gfx1200__) || defined(__gfx1201__)

#define QK8_1 32

struct block_q8_1_mmq {
    __half2 ds4[4];          // (d, sum_x) per 32-K sub-block
    int8_t qs[4 * QK8_1];    // 128 int8
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g128_dp4a_gfx12(
    const char*  __restrict__ A,                   // [M] HFQ4G128 rows
    const block_q8_1_mmq* __restrict__ Xq,         // [K/128] Q8_1 mmq blocks
    float*       __restrict__ y,                   // [M] F32 output
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int lane = threadIdx.x;

    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;

    // Per-lane mapping inside one HFQ4G128 group:
    //   K range:    [4*lane, 4*lane + 3]
    //   sub-block:  lane / 8  (0..3; 8 lanes × 4 K = 32 K per sub-block)
    //   qs byte:    4*lane within the 128-byte qs array
    const int sub_in_block = lane >> 3;            // 0..3
    const int qs_byte_off = lane * 4;              // 0,4,8,...,124
    const int nib_byte_off = lane * 2;             // 0,2,4,...,62 within the 64-byte nibble blob

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_ptr + (long long)g * 72;
        const float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const float zp_eff = zp + 8.0f * sc;

        // Read 2 nibble bytes → 4 nibbles, shift to signed int8 (nib - 8).
        const unsigned char b0 = *(const unsigned char*)(gp + 8 + nib_byte_off);
        const unsigned char b1 = *(const unsigned char*)(gp + 8 + nib_byte_off + 1);
        const int n0 = (int)(b0 & 0xFu) - 8;
        const int n1 = (int)((b0 >> 4) & 0xFu) - 8;
        const int n2 = (int)(b1 & 0xFu) - 8;
        const int n3 = (int)((b1 >> 4) & 0xFu) - 8;
        // Pack 4 signed nibbles (now in -8..+7 range, fits in int8) into int4x4.
        const unsigned int w_packed =
              ((unsigned int)(n0 & 0xFFu))
            | ((unsigned int)(n1 & 0xFFu) << 8)
            | ((unsigned int)(n2 & 0xFFu) << 16)
            | ((unsigned int)(n3 & 0xFFu) << 24);

        // Load Q8_1 mmq block. Block index = g (since G128 = 1 block).
        const block_q8_1_mmq* xb = &Xq[g];
        // Read this lane's 4 int8 X values (4 bytes = 1 uint32).
        const unsigned int x_packed = *(const unsigned int*)(&xb->qs[qs_byte_off]);

        // Per-sub-block scale d and sum_x.
        const float d_sub = (float)(xb->ds4[sub_in_block].x);
        const float sumx_sub = (float)(xb->ds4[sub_in_block].y);

        // gfx12: signed × unsigned dot via sudot4 (sdot4 needs dot1-insts
        // feature flag not enabled by default for gfx1200/1201). For our
        // signed-shifted nibbles (n - 8) and signed Q8_1 X, treat the X as
        // unsigned via byte cast — wait, X is signed so this doesn't work.
        // Use the iu8 mixed-sign dot via WMMA-style cast: pack both as
        // unsigned bytes (X+128) and adjust for the offset post-hoc.
        const unsigned int x_packed_u =
              ((unsigned int)((unsigned char)(((int)(int8_t)(x_packed & 0xFFu)) + 128))
            | ((unsigned int)((unsigned char)(((int)(int8_t)((x_packed >> 8) & 0xFFu)) + 128))) << 8
            | ((unsigned int)((unsigned char)(((int)(int8_t)((x_packed >> 16) & 0xFFu)) + 128))) << 16
            | ((unsigned int)((unsigned char)(((int)(int8_t)((x_packed >> 24) & 0xFFu)) + 128))) << 24);
        // sudot4: signed × unsigned i8x4 → i32. Signature is
        // (bool a_sign, int a, bool b_sign, int b, int c, bool clamp).
        // w is signed (-8..+7 nibbles), x is unsigned (after +128 offset).
        const int dot = __builtin_amdgcn_sudot4(
            true, (int)w_packed, false, (int)x_packed_u, 0, false
        );
        // sudot4 computed sum(w_i * (x_i + 128)) = sum(w_i * x_i) + 128 * sum(w_i)
        // So actual dot = dot - 128 * sum(w_i_signed)
        const int sum_w_signed = n0 + n1 + n2 + n3;
        const int dot_corrected = dot - 128 * sum_w_signed;

        // Accumulate: scale * d_sub * dot + zp_eff * sum_x_per_lane
        // sum_x is per-sub-block (32 K). Each lane covers only 4 of those
        // 32 K-elements. The 8 lanes in a sub-block together cover all 32.
        // We need sum_x_lane * zp_eff per lane, but `sum_x` in the Q8_1
        // block is the per-sub-block total — divided across 8 lanes.
        // For correctness we'd need PER-LANE sum_x. Since sum_x_per_lane =
        // x_int8[0] + x_int8[1] + x_int8[2] + x_int8[3], compute inline:
        const int sx_lane = (int)(int8_t)(x_packed & 0xFFu)
                          + (int)(int8_t)((x_packed >> 8) & 0xFFu)
                          + (int)(int8_t)((x_packed >> 16) & 0xFFu)
                          + (int)(int8_t)((x_packed >> 24) & 0xFFu);

        acc += sc * d_sub * (float)dot_corrected + zp_eff * d_sub * (float)sx_lane;
        (void)sumx_sub;  // unused; per-lane sx_lane is what we need
    }

    // Wave32 shfl_down reduce.
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }
    if (lane == 0) y[row] = acc;
}

#else

extern "C" __global__ void gemv_hfq4g128_dp4a_gfx12(
    const char*, const void*, float*, int, int
) {}

#endif
