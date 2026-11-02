// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Decode packed Qwen4 MQv2 rows into the rotated F32 basis consumed by the
// ordinary MQ GEMV kernels.  The caller applies the same orthogonal FWHT a
// second time to obtain the natural embedding basis (H^-1 == H for both
// qt44/G256 and qt53/G128).  Token ids are device-resident int32 values.
//
// qt44 (format=44): row stride ceil(K/256)*136 bytes, two fp16 scale/zero
// pairs per 256-element group (one pair for each 128-value half).
// qt53 (format=53): row stride ceil(K/128)*68 bytes, one fp16 scale/zero pair
// per 128-element group.
extern "C" __global__ void embedding_mq4v2_batched(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,
    const int* __restrict__ token_ids,
    int n,
    int dim,
    int format) {
    const int token = (int)blockIdx.x;
    const int group = (int)blockIdx.y;
    const int lane = (int)threadIdx.x;
    if (token >= n || dim <= 0) return;

    const int group_width = format == 44 ? 256 : 128;
    const int groups = (dim + group_width - 1) / group_width;
    if (group >= groups || lane >= 32) return;
    const int row_bytes = format == 44 ? groups * 136 : groups * 68;
    const int token_id = token_ids[token];
    const unsigned char* row = table + (size_t)token_id * row_bytes;
    const unsigned char* gp = row + (size_t)group * (format == 44 ? 136 : 68);
    float* out = output + (size_t)token * dim;

    if (format == 44) {
        const int base = group * 256 + lane * 8;
        const unsigned int h0 = *(const unsigned int*)(gp + (lane < 16 ? 0 : 4));
        const float scale = __half2float(__ushort_as_half((unsigned short)(h0 & 0xffffu)));
        const float zero = __half2float(__ushort_as_half((unsigned short)(h0 >> 16)));
        const unsigned int packed = *(const unsigned int*)(gp + 8 + lane * 4);
        if (base + 0 < dim) out[base + 0] = scale * (float)(packed & 0xfu) + zero;
        if (base + 1 < dim) out[base + 1] = scale * (float)((packed >> 4) & 0xfu) + zero;
        if (base + 2 < dim) out[base + 2] = scale * (float)((packed >> 8) & 0xfu) + zero;
        if (base + 3 < dim) out[base + 3] = scale * (float)((packed >> 12) & 0xfu) + zero;
        if (base + 4 < dim) out[base + 4] = scale * (float)((packed >> 16) & 0xfu) + zero;
        if (base + 5 < dim) out[base + 5] = scale * (float)((packed >> 20) & 0xfu) + zero;
        if (base + 6 < dim) out[base + 6] = scale * (float)((packed >> 24) & 0xfu) + zero;
        if (base + 7 < dim) out[base + 7] = scale * (float)((packed >> 28) & 0xfu) + zero;
    } else {
        const int base = group * 128 + lane * 4;
        const unsigned short scale_bits = *(const unsigned short*)(gp + 0);
        const unsigned short zero_bits = *(const unsigned short*)(gp + 2);
        const float scale = __half2float(__ushort_as_half(scale_bits));
        const float zero = __half2float(__ushort_as_half(zero_bits));
        const unsigned char b0 = gp[4 + lane * 2];
        const unsigned char b1 = gp[5 + lane * 2];
        if (base + 0 < dim) out[base + 0] = scale * (float)(b0 & 0xfu) + zero;
        if (base + 1 < dim) out[base + 1] = scale * (float)(b0 >> 4) + zero;
        if (base + 2 < dim) out[base + 2] = scale * (float)(b1 & 0xfu) + zero;
        if (base + 3 < dim) out[base + 3] = scale * (float)(b1 >> 4) + zero;
    }
}
