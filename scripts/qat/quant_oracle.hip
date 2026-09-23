// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Training-only launch wrappers around the shipping activation quantizers.
// This file intentionally imports the runtime fragments instead of carrying a
// second quantization recipe.

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "../../kernels/src/block_i4_128_quant.hip"
#include "../../kernels/src/pack_f16_to_fp8_mq4v2.gfx12.hip"

namespace {

__device__ __forceinline__ int decode_i4(unsigned char nibble) {
    int q = (int)(nibble & 15u);
    return q >= 8 ? q - 16 : q;
}

template <bool ProducerFourCandidates>
__global__ void quant_oracle_a4_kernel(
    const float* __restrict__ z,
    float* __restrict__ dq,
    block_i4_128* __restrict__ blocks,
    int K,
    int T
) {
    const int group = (int)blockIdx.x;
    const int token = (int)blockIdx.y;
    const int lane = (int)threadIdx.x;
    if (lane >= 32 || token >= T || group * 128 >= K) return;

    const int k0 = group * 128 + lane * 4;
    float xv[4];
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int k = k0 + e;
        xv[e] = k < K ? z[(long long)k * T + token] : 0.0f;
    }

    __shared__ block_i4_128 shared_block;
    quantize_block_i4_128_wave<ProducerFourCandidates>(xv, &shared_block, lane);
    __syncwarp();

    const unsigned char p0 = shared_block.qs[lane * 2 + 0];
    const unsigned char p1 = shared_block.qs[lane * 2 + 1];
    const int q0 = decode_i4(p0);
    const int q1 = decode_i4(p0 >> 4);
    const int q2 = decode_i4(p1);
    const int q3 = decode_i4(p1 >> 4);
    const int qv[4] = {q0, q1, q2, q3};
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int k = k0 + e;
        if (k < K) dq[(long long)k * T + token] = shared_block.d * (float)qv[e];
    }

    __syncwarp();
    if (lane == 0) blocks[(long long)group * T + token] = shared_block;
}

} // namespace

extern "C" int quant_oracle_a4(
    const float* z,
    float* dq,
    unsigned char* block_bytes,
    int K,
    int T,
    int site,
    hipStream_t stream
) {
    if (!z || !dq || !block_bytes || K <= 0 || T <= 0 || (K % 128) != 0)
        return (int)hipErrorInvalidValue;
    const dim3 grid((unsigned)(K / 128), (unsigned)T, 1);
    const dim3 block(32, 1, 1);
    block_i4_128* blocks = reinterpret_cast<block_i4_128*>(block_bytes);
    if (site == 0) {
        hipLaunchKernelGGL(
            (quant_oracle_a4_kernel<true>), grid, block, 0, stream,
            z, dq, blocks, K, T);
    } else if (site == 1) {
        hipLaunchKernelGGL(
            (quant_oracle_a4_kernel<false>), grid, block, 0, stream,
            z, dq, blocks, K, T);
    } else {
        return (int)hipErrorInvalidValue;
    }
    return (int)hipGetLastError();
}

// Launches the unmodified standalone shipping entry point. x_token_major is
// [T,K], while blocks retain the runtime's [K/128,T] block ordering.
extern "C" int quant_oracle_shipping8(
    const float* x_token_major,
    unsigned char* block_bytes,
    int K,
    int T,
    hipStream_t stream
) {
    if (!x_token_major || !block_bytes || K <= 0 || T <= 0 || (K % 128) != 0)
        return (int)hipErrorInvalidValue;
    const dim3 grid((unsigned)((K + 1023) / 1024), (unsigned)T, 1);
    const dim3 block(256, 1, 1);
    hipLaunchKernelGGL(
        quantize_int4_mmq_ds128, grid, block, 0, stream,
        x_token_major, reinterpret_cast<block_i4_128*>(block_bytes), K, T);
    return (int)hipGetLastError();
}

// Launches the shipping MQ4V2 E4M3 pre-pass. source_f16 selects the exact
// two-rounding F16 producer variant; otherwise input is the F32 variant.
extern "C" int quant_oracle_a8_f32(
    const float* x_token_major,
    unsigned char* qs,
    float* half_sums,
    float* row_scales,
    int K,
    int T,
    hipStream_t stream
) {
    if (!x_token_major || !qs || !half_sums || !row_scales || K <= 0 || T <= 0 || (K % 256) != 0)
        return (int)hipErrorInvalidValue;
    hipLaunchKernelGGL(
        pack_f32_to_fp8_mq4v2_gfx12, dim3((unsigned)T), dim3(256), 0, stream,
        x_token_major, qs, half_sums, row_scales, K, T, 1);
    return (int)hipGetLastError();
}

extern "C" int quant_oracle_a8_f16(
    const _Float16* x_token_major,
    unsigned char* qs,
    float* half_sums,
    float* row_scales,
    int K,
    int T,
    hipStream_t stream
) {
    if (!x_token_major || !qs || !half_sums || !row_scales || K <= 0 || T <= 0 || (K % 256) != 0)
        return (int)hipErrorInvalidValue;
    hipLaunchKernelGGL(
        pack_f16_to_fp8_mq4v2_gfx12, dim3((unsigned)T), dim3(256), 0, stream,
        x_token_major, qs, half_sums, row_scales, K, T, 1);
    return (int)hipGetLastError();
}
