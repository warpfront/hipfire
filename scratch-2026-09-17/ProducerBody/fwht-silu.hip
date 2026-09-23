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
#endif // HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE

#endif // HIPFIRE_BLOCK_I4_128_QUANT_HIP

#define HIPFIRE_IU4_SIDECAR 1
#define HIPFIRE_SILU_MQ_ROTATE_KERNEL fused_silu_mul_mq_rotate_awq_i4_gfx12
#include <hip/hip_runtime.h>

// AWQ-aware variant of `fused_silu_mul_mq_rotate`.
//
// Phase A Stage A — F2 follow-up. Original kernel is used as the
// down_proj / w_down input stage (silu(gate)*up → FWHT). Stage A's AWQ
// whitelist excluded down_proj because no AWQ-aware silu_mul_rotate
// existed for that path. This kernel completes the AWQ math
// `(W·s) · (silu(g)*u / s) = W · silu(g)*u` by dividing each element by
// `awq_scale[i]` AFTER the silu*up reduction but BEFORE the signs1 gather
// and FWHT. Exact mirror of `fused_rmsnorm_mq_rotate_awq` (derived from fused_rmsnorm_mq_rotate.hip via -DHIPFIRE_RMSNORM_AWQ) Phase 1c.
//
// Backward compatibility: dispatched ONLY when the upcoming linear
// (down_proj / w_down) carries an `awq_scale`. Otherwise the standard
// kernel runs. Both produce identical output on .hfq files without AWQ.
//
// Structure matches `fused_silu_mul_mq_rotate` exactly — only phase 1
// differs (extra AWQ divide between silu*up and signs1 gather).
//
// awq_scale: 1D FP32 buffer length K (host-side F16 → F32 conversion
// happens in the loader; see hfq.rs::load_awq_scale).
//
// HIPFIRE_IU4_SIDECAR: C2 producer path — same as the plain silu twin.

#ifndef HIPFIRE_SILU_MQ_ROTATE_KERNEL
#define HIPFIRE_SILU_MQ_ROTATE_KERNEL fused_silu_mul_mq_rotate_awq
#endif

#if defined(__gfx1201__) && defined(HIPFIRE_IU4_SIDECAR)
static __device__ __forceinline__ int silu_fwht_xor_b32_gfx1201(
    int bits,
    int stride
) {
    switch (stride) {
        case 1:
            return __builtin_amdgcn_mov_dpp(bits, 0xb1, 0xf, 0xf, false);
        case 2:
            return __builtin_amdgcn_mov_dpp(bits, 0x4e, 0xf, 0xf, false);
        case 4:
            return __builtin_amdgcn_permlane16(
                bits, bits, 0x32107654, 0xba98fedc, false, false);
        case 8:
            return __builtin_amdgcn_permlane16(
                bits, bits, 0xfedcba98, 0x76543210, false, false);
        case 16:
            return __builtin_amdgcn_permlanex16(
                bits, bits, 0x76543210, 0xfedcba98, false, false);
        default:
            return bits;
    }
}
#endif

extern "C" __launch_bounds__(32, 16)
__global__ void HIPFIRE_SILU_MQ_ROTATE_KERNEL(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ awq_scale,   // [K], FP32
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
#ifdef HIPFIRE_IU4_SIDECAR
    float* __restrict__ x_rot,          // nullable when emit_f32=false
    block_i4_128* __restrict__ x_i4,    // [K/128, N]
    int K,
    int N
#else
    float* __restrict__ x_rot,
    int K
#endif
) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    if (group >= groups_total) return;

    // Batch row: grid.y is the batch dim. For decode grid=[groups_total, 1, 1]
    // and blockIdx.y = 0 (byte-identical to pre-batching). For prefill,
    // grid=[groups_total, N, 1]; gate, up, and x_rot are [N × K] row-major.
    const long long batch_off = (long long)blockIdx.y * K;
    const float* gate_b  = gate  + batch_off;
    const float* up_b    = up    + batch_off;
#ifdef HIPFIRE_IU4_SIDECAR
          float* x_rot_b = (x_rot != nullptr) ? (x_rot + batch_off) : nullptr;
#else
          float* x_rot_b = x_rot + batch_off;
#endif

    int d0 = tid * 8;
    int base = group * 256 + d0;

    // Phase 1: read gate + up, compute silu(gate)*up, AWQ divide, signs1.
    // silu(z) = z / (1 + exp(-z)). AWQ divide happens BEFORE signs1+FWHT
    // to mirror the offline weight-side pre-scaling W' = W · diag(s).
    // awq_scale is indexed in the unrotated feature basis (1D, length K).
    #define SILU_MUL(g, u) ((g) / (1.0f + expf(-(g))) * (u))
    float g0 = gate_b[base];     float u0 = up_b[base];
    float g1 = gate_b[base + 1]; float u1 = up_b[base + 1];
    float g2 = gate_b[base + 2]; float u2 = up_b[base + 2];
    float g3 = gate_b[base + 3]; float u3 = up_b[base + 3];
    float g4 = gate_b[base + 4]; float u4 = up_b[base + 4];
    float g5 = gate_b[base + 5]; float u5 = up_b[base + 5];
    float g6 = gate_b[base + 6]; float u6 = up_b[base + 6];
    float g7 = gate_b[base + 7]; float u7 = up_b[base + 7];

    float v0 = (SILU_MUL(g0, u0) / awq_scale[base    ]) * signs1[d0    ];
    float v1 = (SILU_MUL(g1, u1) / awq_scale[base + 1]) * signs1[d0 + 1];
    float v2 = (SILU_MUL(g2, u2) / awq_scale[base + 2]) * signs1[d0 + 2];
    float v3 = (SILU_MUL(g3, u3) / awq_scale[base + 3]) * signs1[d0 + 3];
    float v4 = (SILU_MUL(g4, u4) / awq_scale[base + 4]) * signs1[d0 + 4];
    float v5 = (SILU_MUL(g5, u5) / awq_scale[base + 5]) * signs1[d0 + 5];
    float v6 = (SILU_MUL(g6, u6) / awq_scale[base + 6]) * signs1[d0 + 6];
    float v7 = (SILU_MUL(g7, u7) / awq_scale[base + 7]) * signs1[d0 + 7];
    #undef SILU_MUL

    // Local butterfly: strides 1, 2, 4 — identical to mq_rotate_x.
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

    // Exchange the same XOR partner with gfx1201 register-lane permutes.
    #if defined(__gfx1201__) && defined(HIPFIRE_IU4_SIDECAR)
    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(silu_fwht_xor_b32_gfx1201(__float_as_int(v), (str))); \
        if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while(0)
    #else
    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while(0)
    #endif

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
#ifdef HIPFIRE_IU4_SIDECAR
    if (x_rot_b != nullptr) {
        x_rot_b[base    ] = f0;
        x_rot_b[base + 1] = f1;
        x_rot_b[base + 2] = f2;
        x_rot_b[base + 3] = f3;
        x_rot_b[base + 4] = f4;
        x_rot_b[base + 5] = f5;
        x_rot_b[base + 6] = f6;
        x_rot_b[base + 7] = f7;
    }
    emit_iu4_sidecar_from_producer8(
        f0, f1, f2, f3, f4, f5, f6, f7,
        x_i4, group, (int)blockIdx.y, N, tid);
#else
    x_rot_b[base    ] = f0;
    x_rot_b[base + 1] = f1;
    x_rot_b[base + 2] = f2;
    x_rot_b[base + 3] = f3;
    x_rot_b[base + 4] = f4;
    x_rot_b[base + 5] = f5;
    x_rot_b[base + 6] = f6;
    x_rot_b[base + 7] = f7;
#endif
}
