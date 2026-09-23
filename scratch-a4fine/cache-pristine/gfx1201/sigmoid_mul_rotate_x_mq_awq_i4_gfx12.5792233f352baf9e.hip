#define HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE 1
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
// Default 1 (RTN, d=amax/7) on gfx1201 (measured 2026-09-21 on the symmetric XT
// artifact vs N=2: TTFT 1694->1666 ms (3,546 tok/s), pp8192 +1.7%, iu4 c24
// 0.0787->0.0860 (budget 0.10), fp8v2 route unchanged 0.0495). -DIU4_A4_CANDIDATES=2/4 to revert.
#ifndef IU4_A4_CANDIDATES
#define IU4_A4_CANDIDATES 1
#endif

#if IU4_A4_CANDIDATES != 1 && IU4_A4_CANDIDATES != 2 && IU4_A4_CANDIDATES != 4
#error "IU4_A4_CANDIDATES must be 1, 2, or 4"
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
// 8-candidate body. On gfx1201, IU4_A4_CANDIDATES selects the producer search:
// 4 keeps the incumbent {0,2,5,7} subset, 2 keeps the measured-most-selected
// {5,7} subset, and 1 is plain RTN with d=max(abs(x))/7.
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
#define HIPFIRE_ROTATE_SIGMOID_GATE 1
#define HIPFIRE_ROTATE_AWQ 1
#define HIPFIRE_ROTATE_KERNEL sigmoid_mul_rotate_x_mq_awq_i4_gfx12
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Fused standalone FWHT rotation + `block_i4_128` IU4 sidecar for the wo
// (residual) prefill path.
//
// Back-to-back pair this removes (pp2048/gfx1151: 256 × `mq_rotate_x_batched`
// at 40 µs + 256 × `quantize_int4_mmq_ds128` at 113 µs, one pair per
// layer-chunk on the wo input):
//   1. `mq_rotate_x_batched` (or the AWQ twin): dn_normed / fa_attn_out
//      (gated_norm / sigmoid outputs — NOT rmsnorm outputs, so the C2
//      RMSNorm/SwiGLU producers can never cover them) → x_rot f32.
//   2. `quantize_int4_mmq_ds128` (inside the IU4 residual GEMM via
//      `ensure_int4_mmq_x`): x_rot f32 → `block_i4_128` Xq, single consumer.
//
// `x_out` may be null when the prepared IU4 consumer is the only downstream
// reader. The sidecar still receives the same register values and bytes.
//
// -DHIPFIRE_ROTATE_AWQ=1 builds the AWQ twin `rotate_x_mq_awq_i4` (extra
// `awq_scale[K]` argument, divided BEFORE the FWHT — same formation as
// `rotate_x_mq_awq`, completing `(W·s)·(x/s)`).
// -DHIPFIRE_ROTATE_SIGMOID_GATE=1 adds a read-only gate input and forms
// `x_in[i] * sigmoid(gate[i])` with the exact `sigmoid_mul_f32` expression
// before the AWQ divide/FWHT. The original input remains unmodified.
//
// Bit-exactness: the FWHT body is a verbatim copy of `mq_rotate_x` /
// `rotate_x_mq_awq` (same grid, same multiply order, same butterfly, same
// scale). The sidecar reuses `emit_iu4_sidecar_from_producer8` from the
// shared recipe (prepended by the host JIT with
// `HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE=1`, same as the C2 producers):
// f32 store→load is lossless, so the registers hold exactly what the
// standalone quantizer would reload, the lane remap delivers the same
// `xv[4]` per lane, and the shared `quantize_block_i4_128_wave` (amax →
// 8-candidate strict-`<` MSE → rintf/clamp → exact int sum) yields
// identical Xq bytes. Lane ownership matches because both this kernel and
// the C2 producers hold 8 consecutive values per lane
// (`d0 = tid * 8`, `base = group * 256 + d0`).

#ifdef HIPFIRE_ROTATE_KERNEL
/* caller (host JIT concat) already named the symbol */
#elif defined(HIPFIRE_ROTATE_AWQ)
#define HIPFIRE_ROTATE_KERNEL rotate_x_mq_awq_i4
#else
#define HIPFIRE_ROTATE_KERNEL mq_rotate_x_i4
#endif

__launch_bounds__(32, 16)
extern "C" __global__ void HIPFIRE_ROTATE_KERNEL(
    const float* __restrict__ x_in,
#ifdef HIPFIRE_ROTATE_SIGMOID_GATE
    const float* __restrict__ gate,        // [N, K], FP32
#endif
#ifdef HIPFIRE_ROTATE_AWQ
    const float* __restrict__ awq_scale,   // [K], FP32
#endif
    float* __restrict__ x_out,             // nullable when emit_f32=false
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    block_i4_128* __restrict__ x_i4,       // [K/128, N] block_i4_128
    int K,
    int N
) {
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    // Flattened 1-D geometry, same as `mq_rotate_x`: one block per
    // (256-group, token); batch/group derive from the same blockIdx.x.
    const int linear_group = blockIdx.x;
    const int batch = linear_group / groups_total;
    const int group = linear_group - batch * groups_total;

    const long long batch_off = (long long)batch * K;
    const float* x_in_b  = x_in  + batch_off;
#ifdef HIPFIRE_ROTATE_SIGMOID_GATE
    const float* gate_b = gate + batch_off;
#endif
          float* x_out_b = (x_out != nullptr) ? (x_out + batch_off) : nullptr;

    int d0 = tid * 8;
    int base = group * 256 + d0;

    float x0 = x_in_b[base    ];
    float x1 = x_in_b[base + 1];
    float x2 = x_in_b[base + 2];
    float x3 = x_in_b[base + 3];
    float x4 = x_in_b[base + 4];
    float x5 = x_in_b[base + 5];
    float x6 = x_in_b[base + 6];
    float x7 = x_in_b[base + 7];
#ifdef HIPFIRE_ROTATE_SIGMOID_GATE
    // Exact `sigmoid_mul_f32` expression and operation order. The old global
    // store/reload is lossless for f32, so these registers are identical.
    float s0 = 1.0f / (1.0f + expf(-gate_b[base    ]));
    float s1 = 1.0f / (1.0f + expf(-gate_b[base + 1]));
    float s2 = 1.0f / (1.0f + expf(-gate_b[base + 2]));
    float s3 = 1.0f / (1.0f + expf(-gate_b[base + 3]));
    float s4 = 1.0f / (1.0f + expf(-gate_b[base + 4]));
    float s5 = 1.0f / (1.0f + expf(-gate_b[base + 5]));
    float s6 = 1.0f / (1.0f + expf(-gate_b[base + 6]));
    float s7 = 1.0f / (1.0f + expf(-gate_b[base + 7]));
    x0 = x0 * s0;
    x1 = x1 * s1;
    x2 = x2 * s2;
    x3 = x3 * s3;
    x4 = x4 * s4;
    x5 = x5 * s5;
    x6 = x6 * s6;
    x7 = x7 * s7;
#endif

#ifdef HIPFIRE_ROTATE_AWQ
    // Exact order of `rotate_x_mq_awq`: (x / awq_scale) * sign.
    float v0 = (x0 / awq_scale[base    ]) * signs1[d0    ];
    float v1 = (x1 / awq_scale[base + 1]) * signs1[d0 + 1];
    float v2 = (x2 / awq_scale[base + 2]) * signs1[d0 + 2];
    float v3 = (x3 / awq_scale[base + 3]) * signs1[d0 + 3];
    float v4 = (x4 / awq_scale[base + 4]) * signs1[d0 + 4];
    float v5 = (x5 / awq_scale[base + 5]) * signs1[d0 + 5];
    float v6 = (x6 / awq_scale[base + 6]) * signs1[d0 + 6];
    float v7 = (x7 / awq_scale[base + 7]) * signs1[d0 + 7];
#else
    float v0 = x0 * signs1[d0    ];
    float v1 = x1 * signs1[d0 + 1];
    float v2 = x2 * signs1[d0 + 2];
    float v3 = x3 * signs1[d0 + 3];
    float v4 = x4 * signs1[d0 + 4];
    float v5 = x5 * signs1[d0 + 5];
    float v6 = x6 * signs1[d0 + 6];
    float v7 = x7 * signs1[d0 + 7];
#endif

    // Local butterfly: strides 1, 2, 4.
    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;

    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;

    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

    // Wave butterfly: strides 1-16 in thread space via ds_swizzle.
    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while(0)

    #define HBFLY8(pat, str) \
        HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
        HBFLY(v4,pat,str); HBFLY(v5,pat,str); HBFLY(v6,pat,str); HBFLY(v7,pat,str)

    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8
    #undef HBFLY

    const float s = 0.0625f;
    float f0 = v0 * s * signs2[d0    ];
    float f1 = v1 * s * signs2[d0 + 1];
    float f2 = v2 * s * signs2[d0 + 2];
    float f3 = v3 * s * signs2[d0 + 3];
    float f4 = v4 * s * signs2[d0 + 4];
    float f5 = v5 * s * signs2[d0 + 5];
    float f6 = v6 * s * signs2[d0 + 6];
    float f7 = v7 * s * signs2[d0 + 7];

    if (x_out_b != nullptr) {
        x_out_b[base    ] = f0;
        x_out_b[base + 1] = f1;
        x_out_b[base + 2] = f2;
        x_out_b[base + 3] = f3;
        x_out_b[base + 4] = f4;
        x_out_b[base + 5] = f5;
        x_out_b[base + 6] = f6;
        x_out_b[base + 7] = f7;
    }

    // IU4 sidecar: remap this wave's 8-floats/lane into the standalone
    // quant ownership and emit two block_i4_128 rows for this 256-group.
    // token = batch; N is the batch extent for the [K/128, N] layout.
    emit_iu4_sidecar_from_producer8(
        f0, f1, f2, f3, f4, f5, f6, f7,
        x_i4, group, batch, N, tid);
}
