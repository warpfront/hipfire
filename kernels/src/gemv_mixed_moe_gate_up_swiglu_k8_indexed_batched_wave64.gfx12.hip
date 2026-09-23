// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64

// gfx12 MQ4P gate+up/SwiGLU fusion. One wave64 owns one output row:
// lanes 0..31 compute gate, lanes 32..63 compute up. The two halves address
// the same x tile, allowing the scalar/vector caches to satisfy the second
// read without paying eight explicit v_permlane64 instructions per group.
// Supported tags are exactly MQ4P's MQ6 affine, MQ4 affine, and MQ3-Lloyd
// expert population.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__gfx1200__) || defined(__gfx1201__)

#ifdef HIPFIRE_MIXED_GATEUP_WAVE_BARRIER
#define HIPFIRE_MIXED_GATEUP_NAME \
    gemv_mixed_moe_gate_up_swiglu_k8_indexed_batched_wave64_wavebarrier_gfx12
#elif defined(HIPFIRE_MIXED_GATEUP_SHARE_X)
#define HIPFIRE_MIXED_GATEUP_NAME \
    gemv_mixed_moe_gate_up_swiglu_k8_indexed_batched_wave64_sharex_gfx12
#else
#define HIPFIRE_MIXED_GATEUP_NAME \
    gemv_mixed_moe_gate_up_swiglu_k8_indexed_batched_wave64_gfx12
#endif

extern "C" __launch_bounds__(64, 8)
__global__ void HIPFIRE_MIXED_GATEUP_NAME(
    const unsigned long long* __restrict__ expert_ptrs,
    const unsigned char* __restrict__ dtype_tags,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x,
    float* __restrict__ y_hidden,
    int MI, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= MI) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int lane = threadIdx.x;
    const int half_lane = lane & 31;
    const int matrix_row = row + ((lane >> 5) * MI);

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const unsigned char tag = dtype_tags[expert_id];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x_row = x + (long long)bid * K;
    const int groups = K / 256;
    float a0=0.f, a1=0.f, a2=0.f, a3=0.f;
    __shared__ float cb[16];

#ifdef HIPFIRE_MIXED_GATEUP_SHARE_X
#define MIXED_SHARED_X(dst, idx) do { \
        /* Lower wave32 loads once; upper wave32 receives the matching lane. */ \
        /* All 64 lanes still execute permlane64 with full-wave EXEC. */ \
        const float _own = (lane < 32) ? x_row[(idx)] : 0.0f; \
        const unsigned int _peer = __builtin_amdgcn_permlane64( \
            __builtin_bit_cast(unsigned int, _own)); \
        (dst) = (lane < 32) ? _own : __builtin_bit_cast(float, _peer); \
    } while (0)
#else
#define MIXED_SHARED_X(dst, idx) do { \
        (dst) = x_row[(idx)]; \
    } while (0)
#endif

#define MIXED_LOAD_X8(gg) \
    const int xb = (gg) * 256 + half_lane * 8; \
    float x0,x1,x2,x3,x4,x5,x6,x7; \
    MIXED_SHARED_X(x0,xb); MIXED_SHARED_X(x1,xb+1); \
    MIXED_SHARED_X(x2,xb+2); MIXED_SHARED_X(x3,xb+3); \
    MIXED_SHARED_X(x4,xb+4); MIXED_SHARED_X(x5,xb+5); \
    MIXED_SHARED_X(x6,xb+6); MIXED_SHARED_X(x7,xb+7)

#ifdef HIPFIRE_MIXED_GATEUP_WAVE_BARRIER
#define MIXED_CODEBOOK_SYNC() __builtin_amdgcn_wave_barrier()
#else
#define MIXED_CODEBOOK_SYNC() __syncthreads()
#endif

#define MIXED_MQ6_STEP(gg, acc) do { \
        const char* gp = A + ((long long)matrix_row * groups + (gg)) * 200; \
        const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp); \
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4)); \
        const unsigned char* d = (const unsigned char*)(gp + 8) + half_lane * 6; \
        const unsigned char b0=d[0],b1=d[1],b2=d[2],b3=d[3],b4=d[4],b5=d[5]; \
        const int q0=b0&63,q1=(b0>>6)|((b1&15)<<2),q2=(b1>>4)|((b2&3)<<4),q3=b2>>2, \
                  q4=b3&63,q5=(b3>>6)|((b4&15)<<2),q6=(b4>>4)|((b5&3)<<4),q7=b5>>2; \
        MIXED_LOAD_X8(gg); \
        (acc) += (sc*(float)q0+zp)*x0 + (sc*(float)q1+zp)*x1 \
               +(sc*(float)q2+zp)*x2 + (sc*(float)q3+zp)*x3 \
               +(sc*(float)q4+zp)*x4 + (sc*(float)q5+zp)*x5 \
               +(sc*(float)q6+zp)*x6 + (sc*(float)q7+zp)*x7; \
    } while (0)

#define MIXED_MQ4_STEP(gg, acc) do { \
        const char* gp = A + ((long long)matrix_row * groups + (gg)) * 136; \
        const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp); \
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4)); \
        const unsigned int pk = *(const unsigned int*)(gp + 8 + half_lane * 4); \
        MIXED_LOAD_X8(gg); \
        (acc) += (sc*(float)(pk&15u)+zp)*x0 + (sc*(float)((pk>>4)&15u)+zp)*x1 \
               +(sc*(float)((pk>>8)&15u)+zp)*x2 + (sc*(float)((pk>>12)&15u)+zp)*x3 \
               +(sc*(float)((pk>>16)&15u)+zp)*x4 + (sc*(float)((pk>>20)&15u)+zp)*x5 \
               +(sc*(float)((pk>>24)&15u)+zp)*x6 + (sc*(float)((pk>>28)&15u)+zp)*x7; \
    } while (0)

    if (tag == 0 || tag == 2) {
        int g = 0;
        if (tag == 0) {
            for (; g + 3 < groups; g += 4) {
                MIXED_MQ6_STEP(g,a0); MIXED_MQ6_STEP(g+1,a1);
                MIXED_MQ6_STEP(g+2,a2); MIXED_MQ6_STEP(g+3,a3);
            }
            for (; g < groups; g++) MIXED_MQ6_STEP(g,a0);
        } else {
            for (; g + 3 < groups; g += 4) {
                MIXED_MQ4_STEP(g,a0); MIXED_MQ4_STEP(g+1,a1);
                MIXED_MQ4_STEP(g+2,a2); MIXED_MQ4_STEP(g+3,a3);
            }
            for (; g < groups; g++) MIXED_MQ4_STEP(g,a0);
        }
    } else if (tag == 3) {
        const int row_bytes = groups * 112;
        const char* gp = A + (long long)matrix_row * row_bytes;
        const int boff = half_lane * 3;
        for (int g = 0; g < groups; g++, gp += 112) {
            if (half_lane < 8) {
                const unsigned short hc = *(const unsigned short*)(gp + half_lane * 2);
                cb[(lane >> 5) * 8 + half_lane] =
                    __half2float(__builtin_bit_cast(__half, hc));
            }
            MIXED_CODEBOOK_SYNC();
            const unsigned char* d = (const unsigned char*)(gp + 16) + boff;
            const unsigned int pk = (unsigned int)d[0]
                | ((unsigned int)d[1] << 8) | ((unsigned int)d[2] << 16);
            MIXED_LOAD_X8(g);
            const float* cp = cb + (lane >> 5) * 8;
            a0 += cp[pk&7u]*x0 + cp[(pk>>3)&7u]*x1 + cp[(pk>>6)&7u]*x2 + cp[(pk>>9)&7u]*x3
                + cp[(pk>>12)&7u]*x4 + cp[(pk>>15)&7u]*x5 + cp[(pk>>18)&7u]*x6 + cp[(pk>>21)&7u]*x7;
            MIXED_CODEBOOK_SYNC();
        }
    }

#undef MIXED_MQ4_STEP
#undef MIXED_MQ6_STEP
#undef MIXED_LOAD_X8
#undef MIXED_SHARED_X
#undef MIXED_CODEBOOK_SYNC
    float acc = (a0 + a1) + (a2 + a3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset, 32);
    const float up = __shfl(acc, 32, 64);
    if (lane == 0) {
        const long long out = ((long long)bid * K_TOP + krank) * MI + row;
        y_hidden[out] = (acc / (1.0f + expf(-acc))) * up;
    }
}

#undef HIPFIRE_MIXED_GATEUP_NAME
#endif
