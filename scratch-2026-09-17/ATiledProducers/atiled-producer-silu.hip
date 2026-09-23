#define HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE 1
#define HIPFIRE_IU4_ATILED 1
#define HIPFIRE_IU4_SIDECAR 1
#define HIPFIRE_SILU_MQ_ROTATE_KERNEL compile_atiled_silu
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
// (wave64-safe width-32 shuffles). The register result lets row-major and
// A-fragment-tiled emitters share one arithmetic recipe.
struct iu4_quantized_lane {
    float d;
    int s;
    unsigned short qs;
};

__device__ __forceinline__ iu4_quantized_lane quantize_block_i4_128_wave_regs(
    const float xv[4],
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

    iu4_quantized_lane out;
    out.d = best_d;
    out.s = s;
    out.qs = (unsigned short)(
        (q4[0] & 15) | ((q4[1] & 15) << 4)
        | ((q4[2] & 15) << 8) | ((q4[3] & 15) << 12));
    return out;
}

__device__ __forceinline__ void quantize_block_i4_128_wave(
    const float xv[4],
    block_i4_128* __restrict__ out,
    int lane
) {
    const iu4_quantized_lane q = quantize_block_i4_128_wave_regs(xv, lane);
    ((unsigned short*)out->qs)[lane] = q.qs;
    if (lane == 0) {
        out->d = q.d;
        out->s = q.s;
    }
}

// Remap producer-final 8 floats/lane into the standalone quant ownership
// (lane l owns 4l..4l+3 of each 128-half). Producer lane p holds values
// 8p..8p+7. For half h∈{0,1}, source lane = 16*h + (l>>1);
// even l selects the first float4 and odd l the second.
__device__ __forceinline__ iu4_quantized_lane quantize_iu4_producer8_half(
    float f0, float f1, float f2, float f3,
    float f4, float f5, float f6, float f7,
    int h,
    int lane
) {
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
    return quantize_block_i4_128_wave_regs(xv, lane);
}

// Row-major sidecar index: (2*group + h)*N + token.
__device__ __forceinline__ void emit_iu4_sidecar_from_producer8(
    float f0, float f1, float f2, float f3,
    float f4, float f5, float f6, float f7,
    block_i4_128* __restrict__ Y,
    int group,
    int token,
    int K,
    int N,
    int lane
) {
    (void)K;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const iu4_quantized_lane q = quantize_iu4_producer8_half(
            f0, f1, f2, f3, f4, f5, f6, f7, h, lane);
        block_i4_128* out = Y + (long long)(2 * group + h) * N + token;
        ((unsigned short*)out->qs)[lane] = q.qs;
        if (lane == 0) {
            out->d = q.d;
            out->s = q.s;
        }
    }
}

typedef unsigned __attribute__((ext_vector_type(2))) iu4_u32x2_t;

// A-fragment-tiled layout consumed directly by the gfx12 IU4 WMMA:
//   At[K/64 slabs][ceil(N/128) token tiles][4096 bytes], then
//   Ds[K/128 blocks][N tokens][{d:f32,s:i32}].
// Each slab contains 128 tokens x 32 packed bytes in the exact fragment/lane
// order formerly produced by the GEMM's AoS→LDS staging bijection.
__device__ __forceinline__ void store_iu4_atiled_quant(
    const iu4_quantized_lane q,
    unsigned char* __restrict__ Y,
    int kb,
    int token,
    int K,
    int N,
    int lane
) {
    const int writer = lane & 7;
    const int slab = writer >> 2;
    const int quad = writer & 3;
    const int src0 = slab * 16 + quad * 4;
    const unsigned q0 = __shfl((unsigned)q.qs, src0 + 0, 32);
    const unsigned q1 = __shfl((unsigned)q.qs, src0 + 1, 32);
    const unsigned q2 = __shfl((unsigned)q.qs, src0 + 2, 32);
    const unsigned q3 = __shfl((unsigned)q.qs, src0 + 3, 32);

    const long long token_tiles = (N + 127) / 128;
    const long long at_bytes = (long long)(K / 64) * token_tiles * 4096;
    if (lane < 8) {
        const unsigned R = (unsigned)(token & 127);
        const unsigned frag = (R >> 4) * 2u + (unsigned)(quad >> 1);
        const unsigned frag_lane = (R & 15u) + (unsigned)(quad & 1) * 16u;
        const long long tile = token >> 7;
        const long long slab_index = (long long)(2 * kb + slab);
        const long long dst = (slab_index * token_tiles + tile) * 4096
            + (long long)frag * 256 + (long long)frag_lane * 8;
        const iu4_u32x2_t packed = {
            (q0 & 0xffffu) | (q1 << 16),
            (q2 & 0xffffu) | (q3 << 16)
        };
        *(iu4_u32x2_t*)(Y + dst) = packed;
    }
    if (lane == 0) {
        iu4_u32x2_t ds;
        ds[0] = __float_as_uint(q.d);
        ds[1] = (unsigned)q.s;
        const long long ds_index = (long long)kb * N + token;
        *(iu4_u32x2_t*)(Y + at_bytes + ds_index * 8) = ds;
    }
}

__device__ __forceinline__ void emit_iu4_atiled_from_producer8(
    float f0, float f1, float f2, float f3,
    float f4, float f5, float f6, float f7,
    block_i4_128* __restrict__ Y,
    int group,
    int token,
    int K,
    int N,
    int lane
) {
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const iu4_quantized_lane q = quantize_iu4_producer8_half(
            f0, f1, f2, f3, f4, f5, f6, f7, h, lane);
        store_iu4_atiled_quant(q, (unsigned char*)Y, 2 * group + h, token, K, N, lane);
    }
}

#ifdef HIPFIRE_IU4_ATILED
#define emit_iu4_from_producer8 emit_iu4_atiled_from_producer8
#else
#define emit_iu4_from_producer8 emit_iu4_sidecar_from_producer8
#endif

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

// Oracle/helper producer for the tiled contract. Production prefill uses the
// fused producer entry points above; this entry makes the layout bijection
// independently byte-checkable against `quantize_int4_mmq_ds128`.
extern "C" __global__ void quantize_int4_mmq_ds128_atiled(
    const float* __restrict__ X,
    unsigned char* __restrict__ Y,
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
    const iu4_quantized_lane q = quantize_block_i4_128_wave_regs(xv, lane);
    store_iu4_atiled_quant(q, Y, i0 / 128, token, K, N, lane);
}
#endif // HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE

#endif // HIPFIRE_BLOCK_I4_128_QUANT_HIP

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Fused SwiGLU + FWHT rotation: rotate(silu(gate) * up).
// Used as the w_down input stage for MQ4 weights — replaces the sequence:
//   silu_mul_f32(gate, up, ffn_hidden)   // ~9 µs launch
//   mq_rotate_x(ffn_hidden, x_rot)       // ~9 µs launch
// with a single kernel.
//
// Parallelism is per 256-element group, same as mq_rotate_x — each workgroup
// owns its own group and applies the per-group butterfly. Unlike the siluxin
// GEMV experiment this has NO redundant compute: each element's silu is
// computed exactly once across the whole forward, matching the standalone
// silu_mul_f32 kernel.
//
// Structure matches mq_rotate_x exactly — only phase 1 differs (read
// gate[i]+up[i] and compute silu*up, vs read x[i]).
//
// HIPFIRE_IU4_SIDECAR: C2 producer path — after final *s*signs2, optionally
// store f32 and always emit block_i4_128 via the shared wave quant recipe.

#ifndef HIPFIRE_SILU_MQ_ROTATE_KERNEL
#define HIPFIRE_SILU_MQ_ROTATE_KERNEL fused_silu_mul_mq_rotate
#endif

#if defined(__gfx1030__) && !defined(HIPFIRE_IU4_SIDECAR)
extern "C" __launch_bounds__(32)
__global__ void HIPFIRE_SILU_MQ_ROTATE_KERNEL(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
    int K
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
          float* x_rot_b = x_rot + batch_off;

    int d0 = tid * 8;
    int base = group * 256 + d0;

    // Phase 1: register-only silu(gate)*up*signs1 — one element at a time.
    // silu(z) = z / (1 + exp(-z)). No arrays or loops (private-mem fail-closed).
    float v0, v1, v2, v3, v4, v5, v6, v7;
    {
        float g = gate_b[base];
        float u = up_b[base];
        v0 = g / (1.0f + expf(-(g))) * u * signs1[d0];
    }
    {
        float g = gate_b[base + 1];
        float u = up_b[base + 1];
        v1 = g / (1.0f + expf(-(g))) * u * signs1[d0 + 1];
    }
    {
        float g = gate_b[base + 2];
        float u = up_b[base + 2];
        v2 = g / (1.0f + expf(-(g))) * u * signs1[d0 + 2];
    }
    {
        float g = gate_b[base + 3];
        float u = up_b[base + 3];
        v3 = g / (1.0f + expf(-(g))) * u * signs1[d0 + 3];
    }
    {
        float g = gate_b[base + 4];
        float u = up_b[base + 4];
        v4 = g / (1.0f + expf(-(g))) * u * signs1[d0 + 4];
    }
    {
        float g = gate_b[base + 5];
        float u = up_b[base + 5];
        v5 = g / (1.0f + expf(-(g))) * u * signs1[d0 + 5];
    }
    {
        float g = gate_b[base + 6];
        float u = up_b[base + 6];
        v6 = g / (1.0f + expf(-(g))) * u * signs1[d0 + 6];
    }
    {
        float g = gate_b[base + 7];
        float u = up_b[base + 7];
        v7 = g / (1.0f + expf(-(g))) * u * signs1[d0 + 7];
    }
#else
extern "C" __launch_bounds__(32, 16)
__global__ void HIPFIRE_SILU_MQ_ROTATE_KERNEL(
    const float* __restrict__ gate,
    const float* __restrict__ up,
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

    // Phase 1: read gate + up, compute silu(gate)*up, apply signs1.
    // silu(z) = z / (1 + exp(-z)).
    #define SILU_MUL(g, u) ((g) / (1.0f + expf(-(g))) * (u))
    float g0 = gate_b[base];     float u0 = up_b[base];
    float g1 = gate_b[base + 1]; float u1 = up_b[base + 1];
    float g2 = gate_b[base + 2]; float u2 = up_b[base + 2];
    float g3 = gate_b[base + 3]; float u3 = up_b[base + 3];
    float g4 = gate_b[base + 4]; float u4 = up_b[base + 4];
    float g5 = gate_b[base + 5]; float u5 = up_b[base + 5];
    float g6 = gate_b[base + 6]; float u6 = up_b[base + 6];
    float g7 = gate_b[base + 7]; float u7 = up_b[base + 7];

    float v0 = SILU_MUL(g0, u0) * signs1[d0];
    float v1 = SILU_MUL(g1, u1) * signs1[d0 + 1];
    float v2 = SILU_MUL(g2, u2) * signs1[d0 + 2];
    float v3 = SILU_MUL(g3, u3) * signs1[d0 + 3];
    float v4 = SILU_MUL(g4, u4) * signs1[d0 + 4];
    float v5 = SILU_MUL(g5, u5) * signs1[d0 + 5];
    float v6 = SILU_MUL(g6, u6) * signs1[d0 + 6];
    float v7 = SILU_MUL(g7, u7) * signs1[d0 + 7];
    #undef SILU_MUL
#endif

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

    // Wave butterfly via ds_swizzle — same as mq_rotate_x.
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
    float f0 = v0 * s * signs2[d0];
    float f1 = v1 * s * signs2[d0 + 1];
    float f2 = v2 * s * signs2[d0 + 2];
    float f3 = v3 * s * signs2[d0 + 3];
    float f4 = v4 * s * signs2[d0 + 4];
    float f5 = v5 * s * signs2[d0 + 5];
    float f6 = v6 * s * signs2[d0 + 6];
    float f7 = v7 * s * signs2[d0 + 7];
#ifdef HIPFIRE_IU4_SIDECAR
    if (x_rot_b != nullptr) {
        x_rot_b[base]     = f0;
        x_rot_b[base + 1] = f1;
        x_rot_b[base + 2] = f2;
        x_rot_b[base + 3] = f3;
        x_rot_b[base + 4] = f4;
        x_rot_b[base + 5] = f5;
        x_rot_b[base + 6] = f6;
        x_rot_b[base + 7] = f7;
    }
    emit_iu4_from_producer8(
        f0, f1, f2, f3, f4, f5, f6, f7,
        x_i4, group, (int)blockIdx.y, K, N, tid);
#else
    x_rot_b[base]     = f0;
    x_rot_b[base + 1] = f1;
    x_rot_b[base + 2] = f2;
    x_rot_b[base + 3] = f3;
    x_rot_b[base + 4] = f4;
    x_rot_b[base + 5] = f5;
    x_rot_b[base + 6] = f6;
    x_rot_b[base + 7] = f7;
#endif
}
