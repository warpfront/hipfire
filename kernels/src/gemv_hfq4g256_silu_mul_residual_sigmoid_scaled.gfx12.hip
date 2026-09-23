// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx12 (RDNA4) HFQ4G256 fused shared_expert.down step. Mirror of the
// Q8 sister with HFQ4G256 weight reads (~0.53 byte/elem vs Q8's 1.06 byte/elem).

#if defined(__gfx1200__) || defined(__gfx1201__)

extern "C" __global__ void gemv_hfq4g256_silu_mul_residual_sigmoid_scaled_gfx12(
    const char*  __restrict__ W,         // [M × (K/256) × 136]
    const float* __restrict__ gate,      // [K]
    const float* __restrict__ up,        // [K]
    float*       __restrict__ y,         // [M] x_residual — in/out
    const float* __restrict__ scalar,    // [1]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const float s_raw = scalar[0];
    const float sigmoid_s = 1.0f / (1.0f + __expf(-s_raw));

    const int num_blocks = K / 256;  // 2 for K=512
    const char* row_base = W + (long long)row * num_blocks * 136;

    // 32 threads × 8 elements per block per thread = 256 elements covered.
    // 256-thread block: each thread covers 1 element per block.
    // For K=512, num_blocks=2. With 256 threads and 256 elem per block,
    // tid=i covers k = b*256 + i in block b.
    float sum = 0.0f;
    for (int b = 0; b < num_blocks; b++) {
        const char* block_base = row_base + b * 136;
        const float scale = __builtin_bit_cast(float, *(const unsigned int*)(block_base));
        const float min_val = __builtin_bit_cast(float, *(const unsigned int*)(block_base + 4));

        // tid 0..255: read nibble byte at offset 8 + tid/2, extract low/high based on tid%2.
        const int nibble_byte_off = tid >> 1;
        const unsigned char nibble_byte = *(const unsigned char*)(block_base + 8 + nibble_byte_off);
        const unsigned int nib = ((tid & 1) == 0) ? (nibble_byte & 0xFu) : (nibble_byte >> 4);
        const float w = scale * (float)nib + min_val;

        const int k = b * 256 + tid;
        const float g = gate[k];
        const float silu_g = g / (1.0f + __expf(-g));
        const float hidden = silu_g * up[k];
        sum += w * hidden;
    }

    __shared__ float sdata[256];
    sdata[tid] = sum;
    __syncthreads();
    #pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        y[row] += sigmoid_s * sdata[0];
    }
}

#else

extern "C" __global__ void gemv_hfq4g256_silu_mul_residual_sigmoid_scaled_gfx12(
    const char*, const float*, const float*, float*, const float*, int, int
) {}

#endif
