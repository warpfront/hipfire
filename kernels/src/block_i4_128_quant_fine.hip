// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniil Markevich
// hipfire — see LICENSE and NOTICE in the project root.
//
// Shared int4 activation block + wave32 MSE-clip quant recipe for the
// iu4-direct MMQ path. Concatenated by the host JIT into:
//   - gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip (standalone quantize_int4_mmq_ds128)
//   - fused RMSNorm/FWHT and SwiGLU/FWHT IU4 producer sidecars (C2)
//
// Do not edit a second copy of the amax / 8-candidate MSE / rintf+clamp /
// integer-sum path. Every consumer of this fragment must remain bit-identical
// on the 72-byte block_i4_128 layout.

#ifndef HIPFIRE_BLOCK_I4_128_QUANT_HIP
#define HIPFIRE_BLOCK_I4_128_QUANT_HIP

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>
// The shipped activation format uses one scale per K128 block.  Experimental
// gfx1201 screens may split that block into K64 or K32 scale/sum groups while
// retaining the same 64-byte nibble payload and K128 outer indexing.
#ifndef IU4_A4_GROUP_K
#define IU4_A4_GROUP_K 128
#endif

#if IU4_A4_GROUP_K != 32 && IU4_A4_GROUP_K != 64 && IU4_A4_GROUP_K != 128
#error "IU4_A4_GROUP_K must be 32, 64, or 128"
#endif

#define IU4_A4_SUBGROUPS (128 / IU4_A4_GROUP_K)
#define IU4_A4_SUBGROUP_LANES (IU4_A4_GROUP_K / 4)
#define IU4_A4_HEADER_BYTES (IU4_A4_SUBGROUPS * 8)
#if IU4_A4_GROUP_K == 128
#define IU4_A4_PAYLOAD_OFFSET 8
#else
#define IU4_A4_PAYLOAD_OFFSET IU4_A4_HEADER_BYTES
#endif

// Default 1 (RTN, d=amax/7) on gfx1201.  The gfx1201 producer search can be
// restored independently; the portable/standalone path remains its original
// eight-candidate search.
#ifndef IU4_A4_CANDIDATES
#define IU4_A4_CANDIDATES 1
#endif

#if IU4_A4_CANDIDATES != 1 && IU4_A4_CANDIDATES != 2 && IU4_A4_CANDIDATES != 4
#error "IU4_A4_CANDIDATES must be 1, 2, or 4"
#endif

// One outer K128 block per token.  The default branch is deliberately the
// shipped scalar-header spelling; the default code object must remain exact.
#if IU4_A4_GROUP_K == 128
struct block_i4_128 {
    float d;
    int s;
    unsigned char qs[64];
};
static_assert(sizeof(block_i4_128) == 72, "bad block size");
#else
struct block_i4_128 {
    float d[IU4_A4_SUBGROUPS];
    int s[IU4_A4_SUBGROUPS];
    unsigned char qs[64];
};
static_assert(sizeof(block_i4_128) == IU4_A4_HEADER_BYTES + 64, "bad block size");
#endif

template <bool ProducerCandidateSet>
__device__ __forceinline__ void quantize_block_i4_128_wave(
    const float xv[4],
    block_i4_128* __restrict__ out,
    int lane
) {
#if IU4_A4_GROUP_K == 128
    float amax = fmaxf(fmaxf(fabsf(xv[0]), fabsf(xv[1])), fmaxf(fabsf(xv[2]), fabsf(xv[3])));
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }

    float best_d = 1.0f;
    if (amax != 0.0f) {
        float best_mse = 1.0e30f;
        if constexpr (ProducerCandidateSet) {
#if IU4_A4_CANDIDATES == 1
            best_d = amax / 7.0f;
#else
            const float candidate_base = (amax / 7.0f) * 0.5f;
            #pragma unroll 1
            for (int ci = 0; ci < IU4_A4_CANDIDATES; ++ci) {
                float multiplier;
#if IU4_A4_CANDIDATES == 2
                switch (ci) {
                    case 0: multiplier = 0x1.b6db6ep+0f; break;
                    default: multiplier = 2.0f; break;
                }
#else
                switch (ci) {
                    case 0: multiplier = 1.0f; break;
                    case 1: multiplier = 0x1.492492p+0f; break;
                    case 2: multiplier = 0x1.b6db6ep+0f; break;
                    default: multiplier = 2.0f; break;
                }
#endif
                const float dj = candidate_base * multiplier;
                float mse = 0.0f;
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float q = rintf(xv[e] / dj);
                    q = fminf(fmaxf(q, -8.0f), 7.0f);
                    const float err = __fmaf_rn(-q, dj, xv[e]);
                    mse = __fmaf_rn(err, err, mse);
                }
                for (int offset = 16; offset > 0; offset >>= 1) {
                    mse += __shfl_xor(mse, offset, 32);
                }
                if (mse < best_mse) {
                    best_mse = mse;
                    best_d = dj;
                }
            }
#endif
        } else {
            for (int j = 0; j < 8; ++j) {
                const float dj = (amax / 7.0f) * 0.5f * (1.0f + (float)j / 7.0f);
                float mse = 0.0f;
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float q = rintf(xv[e] / dj);
                    q = fminf(fmaxf(q, -8.0f), 7.0f);
                    const float err = __fmaf_rn(-q, dj, xv[e]);
                    mse = __fmaf_rn(err, err, mse);
                }
                for (int offset = 16; offset > 0; offset >>= 1) {
                    mse += __shfl_xor(mse, offset, 32);
                }
                if (mse < best_mse) {
                    best_mse = mse;
                    best_d = dj;
                }
            }
        }
    }

    int q4[4];
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        float q = (amax == 0.0f) ? 0.0f : rintf(xv[e] / best_d);
        q = fminf(fmaxf(q, -8.0f), 7.0f);
        q4[e] = (int)q;
    }
    int s = q4[0] + q4[1] + q4[2] + q4[3];
    for (int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_xor(s, offset, 32);
    }

    unsigned char* qb = out->qs;
    qb[lane * 2 + 0] = (unsigned char)((q4[0] & 15) | ((q4[1] & 15) << 4));
    qb[lane * 2 + 1] = (unsigned char)((q4[2] & 15) | ((q4[3] & 15) << 4));
    if (lane == 0) {
        out->d = best_d;
        out->s = s;
    }
#else
    const int subgroup = lane / IU4_A4_SUBGROUP_LANES;
    float amax = fmaxf(fmaxf(fabsf(xv[0]), fabsf(xv[1])), fmaxf(fabsf(xv[2]), fabsf(xv[3])));
    for (int offset = IU4_A4_SUBGROUP_LANES / 2; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, IU4_A4_SUBGROUP_LANES));
    }

    float best_d = 1.0f;
    if (amax != 0.0f) {
        float best_mse = 1.0e30f;
        if constexpr (ProducerCandidateSet) {
#if IU4_A4_CANDIDATES == 1
            best_d = amax / 7.0f;
#else
            const float candidate_base = (amax / 7.0f) * 0.5f;
            #pragma unroll 1
            for (int ci = 0; ci < IU4_A4_CANDIDATES; ++ci) {
                float multiplier;
#if IU4_A4_CANDIDATES == 2
                switch (ci) {
                    case 0: multiplier = 0x1.b6db6ep+0f; break;
                    default: multiplier = 2.0f; break;
                }
#else
                switch (ci) {
                    case 0: multiplier = 1.0f; break;
                    case 1: multiplier = 0x1.492492p+0f; break;
                    case 2: multiplier = 0x1.b6db6ep+0f; break;
                    default: multiplier = 2.0f; break;
                }
#endif
                const float dj = candidate_base * multiplier;
                float mse = 0.0f;
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float q = rintf(xv[e] / dj);
                    q = fminf(fmaxf(q, -8.0f), 7.0f);
                    const float err = __fmaf_rn(-q, dj, xv[e]);
                    mse = __fmaf_rn(err, err, mse);
                }
                for (int offset = IU4_A4_SUBGROUP_LANES / 2; offset > 0; offset >>= 1) {
                    mse += __shfl_xor(mse, offset, IU4_A4_SUBGROUP_LANES);
                }
                if (mse < best_mse) {
                    best_mse = mse;
                    best_d = dj;
                }
            }
#endif
        } else {
            for (int j = 0; j < 8; ++j) {
                const float dj = (amax / 7.0f) * 0.5f * (1.0f + (float)j / 7.0f);
                float mse = 0.0f;
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float q = rintf(xv[e] / dj);
                    q = fminf(fmaxf(q, -8.0f), 7.0f);
                    const float err = __fmaf_rn(-q, dj, xv[e]);
                    mse = __fmaf_rn(err, err, mse);
                }
                for (int offset = IU4_A4_SUBGROUP_LANES / 2; offset > 0; offset >>= 1) {
                    mse += __shfl_xor(mse, offset, IU4_A4_SUBGROUP_LANES);
                }
                if (mse < best_mse) {
                    best_mse = mse;
                    best_d = dj;
                }
            }
        }
    }

    int q4[4];
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        float q = (amax == 0.0f) ? 0.0f : rintf(xv[e] / best_d);
        q = fminf(fmaxf(q, -8.0f), 7.0f);
        q4[e] = (int)q;
    }
    int sum = q4[0] + q4[1] + q4[2] + q4[3];
    for (int offset = IU4_A4_SUBGROUP_LANES / 2; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, IU4_A4_SUBGROUP_LANES);
    }

    unsigned char* qb = out->qs;
    qb[lane * 2 + 0] = (unsigned char)((q4[0] & 15) | ((q4[1] & 15) << 4));
    qb[lane * 2 + 1] = (unsigned char)((q4[2] & 15) | ((q4[3] & 15) << 4));
    if ((lane & (IU4_A4_SUBGROUP_LANES - 1)) == 0) {
        out->d[subgroup] = best_d;
        out->s[subgroup] = sum;
    }
#endif
}

// Remap producer-final 8 floats/lane into the standalone quant ownership
// (lane l owns 4l..4l+3 of each 128-half) and emit two block_i4_128 rows.
// Producer lane p holds values 8p..8p+7. For half h∈{0,1}:
//   source lane = 16*h + (l >> 1); even l → first float4, odd l → second.
// Sidecar index: (2*group + h)*N + token.
__device__ __forceinline__ void emit_iu4_sidecar_from_producer8(
    float f0, float f1, float f2, float f3,
    float f4, float f5, float f6, float f7,
    block_i4_128* __restrict__ Y,
    int group,
    int token,
    int N,
    int lane
) {
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int src = 16 * h + (lane >> 1);
        float s0 = __shfl(f0, src, 32);
        float s1 = __shfl(f1, src, 32);
        float s2 = __shfl(f2, src, 32);
        float s3 = __shfl(f3, src, 32);
        float s4 = __shfl(f4, src, 32);
        float s5 = __shfl(f5, src, 32);
        float s6 = __shfl(f6, src, 32);
        float s7 = __shfl(f7, src, 32);
        float xv[4];
        if ((lane & 1) == 0) {
            xv[0] = s0; xv[1] = s1; xv[2] = s2; xv[3] = s3;
        } else {
            xv[0] = s4; xv[1] = s5; xv[2] = s6; xv[3] = s7;
        }
        block_i4_128* out = Y + (long long)(2 * group + h) * N + token;
#if defined(__gfx1201__)
        quantize_block_i4_128_wave<true>(xv, out, lane);
#else
        quantize_block_i4_128_wave<false>(xv, out, lane);
#endif
    }
}

// Standalone quantizer: same (X, Y, K, N) grid contract as
// quantize_q8_1_mmq_ds4_x128 (block (256,1,1), each thread 4 floats,
// grid [(K+1023)/1024, N]). One wave32 covers exactly one 128-K group.
#ifndef HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE
extern "C" __global__ void quantize_int4_mmq_ds128(
    const float* __restrict__ X,
    block_i4_128* __restrict__ Y,
    int K,
    int N
) {
    const int token = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int i0 = t * 4;
    if (token >= N || i0 >= K) return;
    const int lane = threadIdx.x & 31;

    const float4 xi = (i0 + 3 < K)
        ? *(const float4*)(X + (long long)token * K + i0)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float xv[4] = {xi.x, xi.y, xi.z, xi.w};

    const int kblock = i0 / 128;
    block_i4_128* out = Y + (long long)kblock * N + token;
    quantize_block_i4_128_wave<false>(xv, out, lane);
}
#endif // HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE

#endif // HIPFIRE_BLOCK_I4_128_QUANT_HIP
