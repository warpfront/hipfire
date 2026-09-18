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

// int4 activation block: one 128-K group for one token (column).
// qs: 64 B, even K low nibble / odd K high nibble, two's complement.
// d: per-128 scale (MSE-clip grid argmin). s: exact int sum of the 128 x_q.
struct block_i4_128 {
    float d;
    int s;
    unsigned char qs[64];
};
static_assert(sizeof(block_i4_128) == 72, "bad block size");

// Wave32-synchronous quant of one 128-element group.
// Lane l owns elements 4l..4l+3 in xv[4]. Reductions use offsets 16..1
// (wave64-safe width-32 shuffles). Writes qs for every lane; lane 0 of the
// group publishes d and s. Arithmetic matches the original standalone body
// (amax → 8 MSE candidates with strict-`<` → rintf/clamp → exact int sum).
__device__ __forceinline__ void quantize_block_i4_128_wave(
    const float xv[4],
    block_i4_128* __restrict__ out,
    int lane
) {
    float amax = fmaxf(fmaxf(fabsf(xv[0]), fabsf(xv[1])), fmaxf(fabsf(xv[2]), fabsf(xv[3])));
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }

    float best_d = 1.0f;
    if (amax != 0.0f) {
        float best_mse = 1.0e30f;
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
        quantize_block_i4_128_wave(xv, out, lane);
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
    quantize_block_i4_128_wave(xv, out, lane);
}

// MQ4E8 slice-B paired-C4 activation arm (runtime-gated by
// HIPFIRE_IU4_XMASTER in the host `ensure_int4_mmq_x` path; the pass-1
// kernel above is never modified, so unset-flag bytes are bit-identical).
//
// Pass 2a (one wave32 per token): master[token] = max over the token's full
// K row of pass-1 block scales / 4 (f32 multiply by 0.25 is exact), and
// rowzero[token] = 1 iff every X element of the row is exactly zero. A zero
// row keeps master 1.0 so pass-2b can preserve the pass-1 d=1/q=0/s=0 bytes.
// Grid [N,1,1], block [32,1,1]: lanes cooperatively scan the row with fully
// coalesced reads (lane l takes elements l, l+32, ...), then wave-reduce.
// (A one-thread-per-token scan is 16x overfetch: concurrent rows sit K*4 B
// apart, so every 4 B read pulls an otherwise-unused 64 B line.)
extern "C" __global__ void int4_mmq_xmaster_reduce(
    const float* __restrict__ X,
    const block_i4_128* __restrict__ Y,
    float* __restrict__ master,
    int* __restrict__ rowzero,
    int K,
    int N
) {
    const int token = blockIdx.x;
    const int lane = threadIdx.x;
    if (token >= N) return;
    const int nb = (K + 127) / 128;
    float maxd = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const float d = (Y + (long long)b * N + token)->d;
        maxd = fmaxf(maxd, d);
    }
    int nonzero = 0;
    const float* row = X + (long long)token * K;
    for (int i = lane; i < K; i += 32) {
        nonzero |= (row[i] != 0.0f);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        maxd = fmaxf(maxd, __shfl_xor(maxd, offset, 32));
        nonzero |= __shfl_xor(nonzero, offset, 32);
    }
    if (lane == 0) {
        if (nonzero == 0) {
            master[token] = 1.0f;
            rowzero[token] = 1;
        } else {
            master[token] = maxd * 0.25f;
            rowzero[token] = 0;
        }
    }
}
// is stored in the existing f32 d field, q recomputed against it, and
// s = exact int sum of the 128 x_q; the iu4 consumer is untouched. Zero rows
// return early so pass-1 bytes survive. e<=2 keeps the paired weight+act
// shift (e_w<=8, e_x<=2, combined<=10) inside the int32 bound at K=17408.
extern "C" __global__ void int4_mmq_xmaster_fixup(
    const float* __restrict__ X,
    block_i4_128* __restrict__ Y,
    const float* __restrict__ master,
    const int* __restrict__ rowzero,
    int K,
    int N
) {
    const int token = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int i0 = t * 4;
    if (token >= N || i0 >= K) return;
    if (rowzero[token] != 0) return;
    const int lane = threadIdx.x & 31;

    const float4 xi = (i0 + 3 < K)
        ? *(const float4*)(X + (long long)token * K + i0)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float xv[4] = {xi.x, xi.y, xi.z, xi.w};

    const float m = master[token];
    float best_dc = m;
    float best_sse = 1.0e30f;
    #pragma unroll
    for (int e = 0; e < 3; ++e) {
        const float dc = (e == 0) ? m : ((e == 1) ? (m * 2.0f) : (m * 4.0f));
        float sse = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float q = rintf(xv[k] / dc);
            q = fminf(fmaxf(q, -8.0f), 7.0f);
            const float err = __fmaf_rn(-q, dc, xv[k]);
            sse = __fmaf_rn(err, err, sse);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            sse += __shfl_xor(sse, offset, 32);
        }
        if (sse < best_sse) {
            best_sse = sse;
            best_dc = dc;
        }
    }

    int q4[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        float q = rintf(xv[k] / best_dc);
        q = fminf(fmaxf(q, -8.0f), 7.0f);
        q4[k] = (int)q;
    }
    int s = q4[0] + q4[1] + q4[2] + q4[3];
    for (int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_xor(s, offset, 32);
    }

    const int kblock = i0 / 128;
    block_i4_128* out = Y + (long long)kblock * N + token;
    unsigned char* qb = out->qs;
    qb[lane * 2 + 0] = (unsigned char)((q4[0] & 15) | ((q4[1] & 15) << 4));
    qb[lane * 2 + 1] = (unsigned char)((q4[2] & 15) | ((q4[3] & 15) << 4));
    if (lane == 0) {
        out->d = best_dc;
        out->s = s;
    }
}
#endif // HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE

#endif // HIPFIRE_BLOCK_I4_128_QUANT_HIP
