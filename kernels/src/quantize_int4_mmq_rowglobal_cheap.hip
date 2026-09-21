// SPDX-License-Identifier: MIT
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

struct block_i4_128 {
    float d;
    int s;
    unsigned char qs[64];
};

// The producer already reduced each K256 tile. This stage touches only the
// tiny amax plane before reading every F32 activation exactly once for RTN.
extern "C" __global__ void quantize_int4_mmq_rowglobal_cheap(
    const float* __restrict__ X,
    block_i4_128* __restrict__ Y,
    const float* __restrict__ tile_amax,
    int K,
    int N
) {
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    if (token >= N) return;

    const int tiles = K / 256;
    float amax = 0.0f;
    for (int tile = tid; tile < tiles; tile += blockDim.x)
        amax = fmaxf(amax, tile_amax[(long long)token * tiles + tile]);

    __shared__ float wave_max[8];
    const int lane = tid & 31;
    const int wave = tid >> 5;
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    if (lane == 0) wave_max[wave] = amax;
    __syncthreads();
    if (tid == 0) {
        float row_max = wave_max[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w)
            row_max = fmaxf(row_max, wave_max[w]);
        wave_max[0] = (row_max == 0.0f) ? 1.0f : row_max / 7.0f;
    }
    __syncthreads();
    const float d = wave_max[0];

    const int blocks = K / 128;
    for (int b = tid; b < blocks; b += blockDim.x) {
        block_i4_128* out = Y + (long long)b * N + token;
        out->d = d;
        out->s = 0;
    }
    const float* row = X + (long long)token * K;
    for (int byte = tid; byte < K / 2; byte += blockDim.x) {
        const int i = byte * 2;
        float q0 = rintf(row[i] / d);
        float q1 = rintf(row[i + 1] / d);
        q0 = fminf(fmaxf(q0, -8.0f), 7.0f);
        q1 = fminf(fmaxf(q1, -8.0f), 7.0f);
        block_i4_128* out = Y + (long long)(i / 128) * N + token;
        out->qs[(i & 127) >> 1] =
            (unsigned char)(((int)q0 & 15) | (((int)q1 & 15) << 4));
    }
}
