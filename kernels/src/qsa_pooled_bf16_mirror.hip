// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Derived BF16 mirror of the QSA pooled index keys (HIPFIRE_QWEN4_QSA_POOLED_BF16,
// exact gfx1151, F32 QSA state, index_dim 128; module qsa_pooled_bf16_mirror).
//
// `qsa_pool_rope` (tensor_ops.hip) rounds every pooled key to BF16 before it
// stores the value as F32, so the low 16 bits of each canonical F32 element are
// zero and the high 16 bits ARE the BF16 value: the conversion is a lossless
// truncation, no rounding. The canonical F32 pooled arena stays authoritative;
// the mirror only feeds the BF16 PM score kernel, which reads half the bytes.
//
// Layout (both tensors): [block][index_dim], blocks first_block..end_block.
// Both pointers are the bases of the whole tensors, so element e of block b
// sits at b * index_dim + e in each.

#include <hip/hip_runtime.h>

// One thread converts four consecutive elements (index_dim % 4 == 0, so a
// 16-byte F32 load at a multiple of four elements is aligned). grid.x =
// ceil((end_block - first_block) * index_dim / 4 / 256), block = 256.
extern "C" __global__ __launch_bounds__(256) void qsa_pooled_f32_to_bf16_mirror(
    const float* __restrict__ pooled,
    unsigned short* __restrict__ mirror,
    int first_block,
    int end_block,
    int index_dim
) {
    const size_t quads =
        (size_t)(end_block - first_block) * (size_t)index_dim / 4;
    const size_t q = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (q >= quads) {
        return;
    }
    const size_t base = (size_t)first_block * (size_t)index_dim + q * 4;
    const uint4 v = *(const uint4*)(pooled + base);
    uint2 out;
    out.x = (v.x >> 16) | (v.y & 0xFFFF0000u);
    out.y = (v.z >> 16) | (v.w & 0xFFFF0000u);
    *(uint2*)(mirror + base) = out;
}

// Lab-only exactness probe: counts canonical F32 elements of blocks
// [first_block, end_block) whose low 16 bits are nonzero (the mirror would
// truncate them). Zero for every BF16-rounded pooled arena. Writes only
// `count`, which the caller zeroes. Same grid as the converter.
extern "C" __global__ __launch_bounds__(256) void qsa_pooled_f32_low_nonzero_count(
    const float* __restrict__ pooled,
    unsigned int* __restrict__ count,
    int first_block,
    int end_block,
    int index_dim
) {
    const size_t quads =
        (size_t)(end_block - first_block) * (size_t)index_dim / 4;
    const size_t q = (size_t)blockIdx.x * 256 + threadIdx.x;
    unsigned int bad = 0;
    if (q < quads) {
        const size_t base = (size_t)first_block * (size_t)index_dim + q * 4;
        const uint4 v = *(const uint4*)(pooled + base);
        bad = ((v.x & 0xFFFFu) != 0u) + ((v.y & 0xFFFFu) != 0u)
            + ((v.z & 0xFFFFu) != 0u) + ((v.w & 0xFFFFu) != 0u);
    }
    if (bad != 0u) {
        atomicAdd(count, bad);
    }
}
