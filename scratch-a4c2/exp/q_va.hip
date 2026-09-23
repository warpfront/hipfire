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
// The gfx11 fused producers default to the measured 2-point {5,7} search
// through the host compile flag. The standalone quantizer retains its original
// 8-point body. Without a flag gfx1201 is RTN (d=amax/7), measured 2026-09-21
// on the symmetric XT artifact vs N=2: TTFT 1694->1666 ms (3,546 tok/s),
// pp8192 +1.7%, iu4 c24 0.0787->0.0860 (budget 0.10), fp8v2 unchanged 0.0495.
// The host now passes -DIU4_A4_CANDIDATES=2 on gfx1201 too (HIPFIRE_G12_A4C2,
// default on), served by the one-pass emit_iu4_sidecar_c2_gfx12 below.
#ifndef IU4_A4_CANDIDATES
#if defined(__gfx1100__) || defined(__gfx1151__)
#define IU4_A4_CANDIDATES 8
#else
#define IU4_A4_CANDIDATES 1
#endif
#endif

#if IU4_A4_CANDIDATES != 1 && IU4_A4_CANDIDATES != 2 && IU4_A4_CANDIDATES != 4 && IU4_A4_CANDIDATES != 8
#error "IU4_A4_CANDIDATES must be 1, 2, 4, or 8"
#endif

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
// group publishes d and s. The portable specialization retains the original
// 8-candidate body. IU4_A4_CANDIDATES selects the fused producer search: 8 is
// the full grid, 4 keeps {0,2,5,7}, 2 keeps the measured-most-selected {5,7},
// and 1 is plain RTN with d=max(abs(x))/7.
//
// -DHIPFIRE_IU4_RTN_RCP=1 (gfx1201 `_v2` RMSNorm producers, RTN only) forms
// the producer codes with one IEEE reciprocal per lane instead of four
// divides; the codes are bit-identical. With r = RN(1/d) and y = RN(x*r),
// |y - x/d| <= |x/d| * (2^-24 + 2^-24 + 2^-48) < 2^-20.2 because
// |x/d| <= 7(1 + 2^-24). If the computed distance from y to the nearest
// half-integer exceeds 2^-19 (the subtraction may be contracted with the
// product: error <= 2^-22 + 2^-25), x/d is more than
// 2^-19 - 2^-22 - 2^-25 - 2^-20.2 > 2^-21 from every half-integer, so
// RN(x/d) (within half an ulp, <= 2^-22 below 8) is not a tie and lies in
// y's rounding cell:
// rint(y) == rint(RN(x/d)). Any lane with a value inside that window
// recomputes its four codes with the divide. amax outside [2^-100, FLT_MAX]
// (zero, tiny, Inf, NaN) keeps the divide; NaN elements clamp to -8 either
// way (fmaxf(NaN, -8) = -8).
template <bool ProducerCandidateSet>
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
        if constexpr (ProducerCandidateSet) {
#if IU4_A4_CANDIDATES == 1
            best_d = amax / 7.0f;
#else
#if IU4_A4_CANDIDATES == 8
            #pragma unroll 1
            for (int ci = 0; ci < 8; ++ci) {
                const float dj = (amax / 7.0f) * 0.5f * (1.0f + (float)ci / 7.0f);
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
#endif
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
#if defined(HIPFIRE_IU4_RTN_RCP) && IU4_A4_CANDIDATES == 1
    if (ProducerCandidateSet && amax >= 0x1p-100f && amax <= 0x1.fffffep+127f) {
        const float r = 1.0f / best_d;
        float qf[4];
        bool near_half = false;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float y = xv[e] * r;
            qf[e] = rintf(y);
            near_half |= (0.5f - fabsf(y - qf[e])) <= 0x1p-19f;
        }
        if (near_half) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) qf[e] = rintf(xv[e] / best_d);
        }
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float q = fminf(fmaxf(qf[e], -8.0f), 7.0f);
            q4[e] = (int)q;
        }
    } else {
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            float q = (amax == 0.0f) ? 0.0f : rintf(xv[e] / best_d);
            q = fminf(fmaxf(q, -8.0f), 7.0f);
            q4[e] = (int)q;
        }
    }
#else
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        float q = (amax == 0.0f) ? 0.0f : rintf(xv[e] / best_d);
        q = fminf(fmaxf(q, -8.0f), 7.0f);
        q4[e] = (int)q;
    }
#endif
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

#if defined(__gfx1201__) && IU4_A4_CANDIDATES == 2
// gfx1201 one-pass 2-candidate emit (`HIPFIRE_G12_A4C2`, which compiles the
// gfx1201 JIT with -DIU4_A4_CANDIDATES=2). Bit-identical to the remap below
// plus `quantize_block_i4_128_wave<true>`'s {5,7} search, without the
// remap, the LDS permutes, or (on the common path) any per-element divide.
//
// It quantizes in producer ownership: lane p holds elements 8j..8j+7 of half
// h = p >> 4 (j = p & 15), so each DPP row of 16 lanes is one 128-block.
// - amax: max of |x| is order-free (no -0 after fabsf; NaN is ignored by
//   max_num exactly as in the butterfly), reduced by row_xmask 8, 4, 2, 1.
// - mse: remap lane l = 2j + b takes xv = x[4b..4b+3] of producer lane j, so
//   its fma chain is this lane's chain over 4b..4b+3 (acc[b]), and the
//   remap's xor-16, 8, 4, 2 butterfly steps are xor-8, 4, 2, 1 on j. Its last
//   step (xor-1 on l) adds acc[1] of the same lane; + is commutative, so all
//   16 lanes hold the remap's sum.
// - s: an exact integer sum, order-free. qs bytes 4j..4j+3 are this lane's
//   eight codes in element order: one u32 store.
// - codes: r = rcp(d) refined by one Newton step, so |r*d - 1| <= 2^-23 even
//   for a 2^-12-accurate rcp. For amax in [2^-100, FLT_MAX] both d are
//   normal and |x/d| < 8.17 (d0 = (6/7) amax/7), hence
//   |y - x/d| <= 8.17 * 2^-23 + 2^-21 < 2^-19.3 for y = RN(x*r), and the
//   computed y - rint(y) (exact, or contracted with the product: error
//   <= 2^-25) is within 2^-19.3 of x/d - rint(y). If every |y - rint(y)| is
//   <= 0.5 - 2^-18, then |RN(x/d) - rint(y)| <= 0.5 - 2^-18 + 2^-19.3 + 2^-21
//   < 0.5 (RN(x/d) is within 2^-21 of x/d below 16), so rint(RN(x/d)) ==
//   rint(y), and d1 = amax/7 codes lie in [-7, 7] without the clamp. Any
//   other lane (a value within 2^-18 of a half-integer, or amax tiny, Inf or
//   NaN) recomputes its codes with the reference divides.
// - best_d = 1 (amax == 0, or neither mse below the 1e30 sentinel: NaN
//   elements, huge rows) reproduces the reference's final rint(x / 1),
//   forced to 0 when amax == 0.
template <int Mask>
__device__ __forceinline__ float iu4_row_xmask_f32(float v) {
    return __int_as_float(
        __builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 | Mask, 0xF, 0xF, true));
}
template <int Mask>
__device__ __forceinline__ int iu4_row_xmask_i32(int v) {
    return __builtin_amdgcn_update_dpp(0, v, 0x160 | Mask, 0xF, 0xF, true);
}

__device__ __forceinline__ bool emit_iu4_sidecar_c2_gfx12(
    const float x[8],
    block_i4_128* __restrict__ Y,
    int group,
    int token,
    int N,
    int lane
) {
    float amax = fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3]))),
                       fmaxf(fmaxf(fabsf(x[4]), fabsf(x[5])), fmaxf(fabsf(x[6]), fabsf(x[7]))));
    amax = fmaxf(amax, iu4_row_xmask_f32<8>(amax));
    amax = fmaxf(amax, iu4_row_xmask_f32<4>(amax));
    amax = fmaxf(amax, iu4_row_xmask_f32<2>(amax));
    amax = fmaxf(amax, iu4_row_xmask_f32<1>(amax));
    const float candidate_base = (amax / 7.0f) * 0.5f;
    const float d0 = candidate_base * 0x1.b6db6ep+0f;
    const float d1 = candidate_base * 2.0f;
    float r0 = __builtin_amdgcn_rcpf(d0);
    float r1 = __builtin_amdgcn_rcpf(d1);
    r0 = __fmaf_rn(r0, __fmaf_rn(-d0, r0, 1.0f), r0);
    r1 = __fmaf_rn(r1, __fmaf_rn(-d1, r1, 1.0f), r1);
    float q0[8], q1[8];
    float frac = 0.0f;
    float acc0[2] = {0.0f, 0.0f};
    float acc1[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float y0 = x[k] * r0;
        const float y1 = x[k] * r1;
        q0[k] = rintf(y0);
        q1[k] = rintf(y1);
        frac = fmaxf(frac, fmaxf(fabsf(y0 - q0[k]), fabsf(y1 - q1[k])));
        q0[k] = fminf(fmaxf(q0[k], -8.0f), 7.0f);
        const float e0 = __fmaf_rn(-q0[k], d0, x[k]);
        const float e1 = __fmaf_rn(-q1[k], d1, x[k]);
        acc0[k >> 2] = __fmaf_rn(e0, e0, acc0[k >> 2]);
        acc1[k >> 2] = __fmaf_rn(e1, e1, acc1[k >> 2]);
    }
    #pragma unroll
    for (int b = 0; b < 2; ++b) {
        acc0[b] += iu4_row_xmask_f32<8>(acc0[b]);
        acc1[b] += iu4_row_xmask_f32<8>(acc1[b]);
        acc0[b] += iu4_row_xmask_f32<4>(acc0[b]);
        acc1[b] += iu4_row_xmask_f32<4>(acc1[b]);
        acc0[b] += iu4_row_xmask_f32<2>(acc0[b]);
        acc1[b] += iu4_row_xmask_f32<2>(acc1[b]);
        acc0[b] += iu4_row_xmask_f32<1>(acc0[b]);
        acc1[b] += iu4_row_xmask_f32<1>(acc1[b]);
    }
    const float mse0 = acc0[0] + acc0[1];
    const float mse1 = acc1[0] + acc1[1];
    const bool take0 = mse0 < 1.0e30f;
    const bool use1 = mse1 < (take0 ? mse0 : 1.0e30f);
    const bool fast = amax >= 0x1p-100f && amax <= 0x1.fffffep+127f
        && frac <= 0.5f - 0x1p-18f && (take0 || use1);
    if (__builtin_amdgcn_ballot_w32(!fast) != 0u) return false;
    int s = 0;
    unsigned int word = 0u;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int q = (int)(use1 ? q1[k] : q0[k]);
        s += q;
        word |= (unsigned int)(q & 15) << (4 * k);
    }
    s += iu4_row_xmask_i32<8>(s);
    s += iu4_row_xmask_i32<4>(s);
    s += iu4_row_xmask_i32<2>(s);
    s += iu4_row_xmask_i32<1>(s);
    block_i4_128* out = Y + (long long)(2 * group + (lane >> 4)) * N + token;
    reinterpret_cast<unsigned int*>(out->qs)[lane & 15] = word;
    if ((lane & 15) == 0) {
        out->d = use1 ? d1 : d0;
        out->s = s;
    }
    return true;
}
#endif

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
#if defined(__gfx1201__) && IU4_A4_CANDIDATES == 2
    {
        const float x[8] = {f0, f1, f2, f3, f4, f5, f6, f7};
        if (emit_iu4_sidecar_c2_gfx12(x, Y, group, token, N, lane)) return;
    }
#endif
#if 1
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
#if defined(__gfx1201__) || defined(__gfx1100__) || defined(__gfx1151__)
        quantize_block_i4_128_wave<true>(xv, out, lane);
#else
        quantize_block_i4_128_wave<false>(xv, out, lane);
#endif
    }
#endif
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
