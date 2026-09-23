#include <hip/hip_runtime.h>

// NOTE: the width argument MUST be 32. The default shuffle width is
// warpSize (64 on gfx11), which mixes in nonexistent lanes 32..63 of this
// 32-thread block and poisons the reduction (observed as NaN row scales).
__device__ __forceinline__ float iu8_wave_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor(v, off, 32));
    return v;
}

__device__ __forceinline__ int iu8_wave_sum_i(int v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor(v, off, 32);
    return v;
}

extern "C" __launch_bounds__(256, 1) __global__
void pack_f32_to_i8_mq4v2_gfx11(
    const float* x_f32,          // [N,K]
    signed char* x_i8,           // [N,K], int8 bytes
    int* half_sums,              // [N,K/256,2] int32 sums of rounded int8
    float* row_scales,           // [N] ax
    int K,
    int N)
{
    // 32 B static LDS: eight wave maxima, then republished ax in slot 0.
    __shared__ float smem[8];

    const int n = (int)blockIdx.x;
    if (n >= N || K <= 0) return;

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5; // 0..7
    const int gpr = K >> 8;    // K / 256

    const float* x_row = x_f32 + (long long)n * K;
    signed char* out_row = x_i8 + (long long)n * K;

    // 1) Strided F32 absmax over the full row (256-thread stride).
    float local_max = 0.0f;
    for (int k = tid; k < K; k += 256) {
        const float v = x_row[k];
        local_max = fmaxf(local_max, fabsf(v));
    }
    // 2) Per-wave reduce; lane 0 of each wave writes one LDS float.
    local_max = iu8_wave_max(local_max);
    if (lane == 0)
        smem[wave] = local_max;
    __syncthreads();
    // Wave 0 resolves ax and publishes it; all threads re-read.
    if (wave == 0 && lane == 0) {
        float amax = smem[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w)
            amax = fmaxf(amax, smem[w]);
        const float ax = (amax > 0.0f) ? (amax * (1.0f / 127.0f)) : 1.0f;
        smem[0] = ax;
        row_scales[n] = ax;
    }
    __syncthreads();
    const float ax = smem[0];
    const float inv_a = 1.0f / ax;

    // 3) Pack each 128-K half: wave-strided over the K/128 halves
    //    (h = wave, wave+8, ...; g = h/2, side = h%2). Lane L owns four
    //    consecutive values at half_base + 4*L. Half sums are reduced from
    //    the ROUNDED int8 values (operand-exact for the WMMA path).
    const int halves = K >> 7; // K / 128
    for (int h = wave; h < halves; h += 8) {
        const int g = h >> 1;
        const int half = h & 1;
        const int k0 = g * 256 + half * 128 + lane * 4;

        const float v0 = x_row[k0 + 0] * inv_a;
        const float v1 = x_row[k0 + 1] * inv_a;
        const float v2 = x_row[k0 + 2] * inv_a;
        const float v3 = x_row[k0 + 3] * inv_a;
        const int q0 = (int)lrintf(fmaxf(-127.0f, fminf(127.0f, v0)));
        const int q1 = (int)lrintf(fmaxf(-127.0f, fminf(127.0f, v1)));
        const int q2 = (int)lrintf(fmaxf(-127.0f, fminf(127.0f, v2)));
        const int q3 = (int)lrintf(fmaxf(-127.0f, fminf(127.0f, v3)));

        out_row[k0 + 0] = (signed char)q0;
        out_row[k0 + 1] = (signed char)q1;
        out_row[k0 + 2] = (signed char)q2;
        out_row[k0 + 3] = (signed char)q3;

        const int half_sum = iu8_wave_sum_i(q0 + q1 + q2 + q3);
        if (lane == 0) {
            // half_sums[(n * gpr + g) * 2 + half]
            half_sums[((long long)n * gpr + g) * 2 + half] = half_sum;
        }
    }
}
