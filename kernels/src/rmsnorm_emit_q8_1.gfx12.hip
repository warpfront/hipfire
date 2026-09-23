// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// llama.cpp-compatible DS4 activation block.  Four independent 32-element
// Q8_1 sub-blocks share one 144-byte allocation unit.
struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[128];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

// Preserve rmsnorm_f32's reduction and output expression exactly while also
// emitting the normalized activation in Q8_1 form.  The ordinary standalone
// Q8_1 pre-pass was as expensive as the DP4A matvec win on decode-sized input;
// folding it into the already-resident RMSNorm output removes that extra read
// and dispatch.  One 256-thread block handles one token.
extern "C" __global__ void rmsnorm_f32_emit_q8_1(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    block_q8_1_mmq* __restrict__ out_q8,
    int n,
    float eps
) {
    extern __shared__ float sdata[];
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[blockIdx.x * n + i];
        sum_sq += v * v;
    }
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    const float rms = rsqrtf(sdata[0] / (float)n + eps);
    const int lane = threadIdx.x & 31;
    const int batch = gridDim.x;

    // Each 32-lane half-wave owns one aligned Q8_1 sub-block per pass.
    // n=2048 uses eight passes; no extra LDS or block barrier is required.
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int idx = blockIdx.x * n + i;
        const float v = x[idx] * weight[i] * rms;
        out[idx] = v;

        float amax = fabsf(v);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor(amax, off, 32));

        const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        int q = (int)lrintf(fmaxf(-127.0f, fminf(127.0f, v * id)));

        float sum = v;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_xor(sum, off, 32);

        const int kblock = i >> 7;
        const int sub = (i >> 5) & 3;
        block_q8_1_mmq* qb = out_q8 + (long long)kblock * batch + blockIdx.x;
        qb->qs[sub * 32 + lane] = (int8_t)q;
        if (lane == 0)
            qb->ds4[sub] = make_half2((_Float16)d, (_Float16)sum);
    }
}
