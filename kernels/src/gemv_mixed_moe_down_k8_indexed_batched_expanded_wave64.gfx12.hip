// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifdef HIPFIRE_MIXED_DOWN_BLOCK_COMBINE
#define HIPFIRE_MIXED_DOWN_NAME \
    gemv_mixed_moe_down_k8_indexed_batched_block_combine_wave64_gfx12
#define HIPFIRE_MIXED_DOWN_OUTPUT_ARGS \
    const float* __restrict__ topk_weights, \
    float* __restrict__ x_residual
#elif defined(HIPFIRE_MIXED_DOWN_ATOMIC_COMBINE)
#define HIPFIRE_MIXED_DOWN_NAME \
    gemv_mixed_moe_down_k8_indexed_batched_atomic_combine_wave64_gfx12
#define HIPFIRE_MIXED_DOWN_OUTPUT_ARGS \
    const float* __restrict__ topk_weights, \
    float* __restrict__ x_residual
#else
#define HIPFIRE_MIXED_DOWN_NAME \
    gemv_mixed_moe_down_k8_indexed_batched_expanded_wave64_gfx12
#define HIPFIRE_MIXED_DOWN_OUTPUT_ARGS \
    float* __restrict__ expert_outputs
#endif

// gfx12 mixed-expert down GEMV. One wave64 packs two adjacent output rows;
// each half keeps the established four-chain per-row accumulation while both
// halves address the same routed activation. Expert rank remains in grid.y,
// preserving the control path's eight-way expert parallelism.
#ifdef HIPFIRE_MIXED_DOWN_BLOCK_COMBINE
extern "C" __launch_bounds__(512, 1)
#else
extern "C" __launch_bounds__(64, 8)
#endif
__global__ void HIPFIRE_MIXED_DOWN_NAME(
    const unsigned long long* __restrict__ expert_ptrs,
    const unsigned char* __restrict__ dtype_tags,
    const int* __restrict__ topk_indices,
    const float* __restrict__ rot_batch,
    HIPFIRE_MIXED_DOWN_OUTPUT_ARGS,
    int M, int K, int K_TOP
) {
    const int lane = threadIdx.x;
    const int half_lane = lane & 31;
    const int row = blockIdx.x * 2 + (lane >> 5);
    if (row >= M) return;
#ifdef HIPFIRE_MIXED_DOWN_BLOCK_COMBINE
    const int krank = threadIdx.y;
#else
    const int krank = blockIdx.y;
#endif
    const int bid = blockIdx.z;
    const int expert_id = topk_indices[bid * K_TOP + krank];
    const unsigned char tag = dtype_tags[expert_id];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;
    const int groups = K / 256;
    float a0=0.f, a1=0.f, a2=0.f, a3=0.f;
#ifdef HIPFIRE_MIXED_DOWN_BLOCK_COMBINE
    __shared__ float cb[8][16];
    __shared__ float partial[8][2];
#else
    __shared__ float cb[16];
#endif

#define MIXED_DOWN_X8(gg) \
    const int xb = (gg) * 256 + half_lane * 8; \
    const float x0=x[xb],x1=x[xb+1],x2=x[xb+2],x3=x[xb+3], \
                x4=x[xb+4],x5=x[xb+5],x6=x[xb+6],x7=x[xb+7]

#define MIXED_DOWN_MQ6(gg, acc) do { \
        const char* gp = A + ((long long)row * groups + (gg)) * 200; \
        const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp); \
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4)); \
        const unsigned char* d = (const unsigned char*)(gp + 8) + half_lane * 6; \
        const unsigned char b0=d[0],b1=d[1],b2=d[2],b3=d[3],b4=d[4],b5=d[5]; \
        const int q0=b0&63,q1=(b0>>6)|((b1&15)<<2),q2=(b1>>4)|((b2&3)<<4),q3=b2>>2, \
                  q4=b3&63,q5=(b3>>6)|((b4&15)<<2),q6=(b4>>4)|((b5&3)<<4),q7=b5>>2; \
        MIXED_DOWN_X8(gg); \
        (acc) += (sc*(float)q0+zp)*x0 + (sc*(float)q1+zp)*x1 \
               +(sc*(float)q2+zp)*x2 + (sc*(float)q3+zp)*x3 \
               +(sc*(float)q4+zp)*x4 + (sc*(float)q5+zp)*x5 \
               +(sc*(float)q6+zp)*x6 + (sc*(float)q7+zp)*x7; \
    } while (0)

#define MIXED_DOWN_MQ4(gg, acc) do { \
        const char* gp = A + ((long long)row * groups + (gg)) * 136; \
        const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp); \
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4)); \
        const unsigned int pk = *(const unsigned int*)(gp + 8 + half_lane * 4); \
        MIXED_DOWN_X8(gg); \
        (acc) += (sc*(float)(pk&15u)+zp)*x0 + (sc*(float)((pk>>4)&15u)+zp)*x1 \
               +(sc*(float)((pk>>8)&15u)+zp)*x2 + (sc*(float)((pk>>12)&15u)+zp)*x3 \
               +(sc*(float)((pk>>16)&15u)+zp)*x4 + (sc*(float)((pk>>20)&15u)+zp)*x5 \
               +(sc*(float)((pk>>24)&15u)+zp)*x6 + (sc*(float)((pk>>28)&15u)+zp)*x7; \
    } while (0)

    if (tag == 0 || tag == 2) {
        int g = 0;
        if (tag == 0) {
            for (; g + 3 < groups; g += 4) {
                MIXED_DOWN_MQ6(g,a0); MIXED_DOWN_MQ6(g+1,a1);
                MIXED_DOWN_MQ6(g+2,a2); MIXED_DOWN_MQ6(g+3,a3);
            }
            for (; g < groups; g++) MIXED_DOWN_MQ6(g,a0);
        } else {
            for (; g + 3 < groups; g += 4) {
                MIXED_DOWN_MQ4(g,a0); MIXED_DOWN_MQ4(g+1,a1);
                MIXED_DOWN_MQ4(g+2,a2); MIXED_DOWN_MQ4(g+3,a3);
            }
            for (; g < groups; g++) MIXED_DOWN_MQ4(g,a0);
        }
    } else if (tag == 3) {
        const int row_bytes = groups * 112;
        const char* gp = A + (long long)row * row_bytes;
        const int boff = half_lane * 3;
        for (int g = 0; g < groups; g++, gp += 112) {
            if (half_lane < 8) {
                const unsigned short hc = *(const unsigned short*)(gp + half_lane * 2);
#ifdef HIPFIRE_MIXED_DOWN_BLOCK_COMBINE
                cb[krank][(lane >> 5) * 8 + half_lane] =
                    __half2float(__builtin_bit_cast(__half, hc));
#else
                cb[(lane >> 5) * 8 + half_lane] =
                    __half2float(__builtin_bit_cast(__half, hc));
#endif
            }
#ifdef HIPFIRE_MIXED_DOWN_BLOCK_COMBINE
            __builtin_amdgcn_wave_barrier();
#else
            __syncthreads();
#endif
            const unsigned char* d = (const unsigned char*)(gp + 16) + boff;
            const unsigned int pk = (unsigned int)d[0]
                | ((unsigned int)d[1] << 8) | ((unsigned int)d[2] << 16);
            MIXED_DOWN_X8(g);
#ifdef HIPFIRE_MIXED_DOWN_BLOCK_COMBINE
            const float* cp = cb[krank] + (lane >> 5) * 8;
#else
            const float* cp = cb + (lane >> 5) * 8;
#endif
            a0 += cp[pk&7u]*x0 + cp[(pk>>3)&7u]*x1 + cp[(pk>>6)&7u]*x2 + cp[(pk>>9)&7u]*x3
                + cp[(pk>>12)&7u]*x4 + cp[(pk>>15)&7u]*x5 + cp[(pk>>18)&7u]*x6 + cp[(pk>>21)&7u]*x7;
#ifdef HIPFIRE_MIXED_DOWN_BLOCK_COMBINE
            __builtin_amdgcn_wave_barrier();
#else
            __syncthreads();
#endif
        }
    }

#undef MIXED_DOWN_MQ4
#undef MIXED_DOWN_MQ6
#undef MIXED_DOWN_X8
    float acc = (a0 + a1) + (a2 + a3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset, 32);
#ifdef HIPFIRE_MIXED_DOWN_BLOCK_COMBINE
    if (half_lane == 0) partial[krank][lane >> 5] = acc;
    __syncthreads();
    if (krank == 0 && half_lane == 0) {
        float combined = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            if (k < K_TOP) {
                combined += topk_weights[bid * K_TOP + k] * partial[k][lane >> 5];
            }
        }
        x_residual[(long long)bid * M + row] += combined;
    }
#else
    if (half_lane == 0) {
#ifdef HIPFIRE_MIXED_DOWN_ATOMIC_COMBINE
        const float scale = topk_weights[bid * K_TOP + krank];
        atomicAdd(&x_residual[(long long)bid * M + row], scale * acc);
#else
        const long long out = ((long long)bid * K_TOP + krank) * M + row;
        expert_outputs[out] = acc;
#endif
    }
#endif
}

#undef HIPFIRE_MIXED_DOWN_NAME
#undef HIPFIRE_MIXED_DOWN_OUTPUT_ARGS
