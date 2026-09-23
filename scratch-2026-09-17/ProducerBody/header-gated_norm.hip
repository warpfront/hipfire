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
template <bool FixedCandidateHeaders>
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
        if constexpr (FixedCandidateHeaders) {
            const float candidate_base = (amax / 7.0f) * 0.5f;
            #pragma unroll 1
            for (int j = 0; j < 8; ++j) {
                float header;
                switch (j) {
                    case 0: header = 0x1.000000p+0f; break;
                    case 1: header = 0x1.24924ap+0f; break;
                    case 2: header = 0x1.492492p+0f; break;
                    case 3: header = 0x1.6db6dcp+0f; break;
                    case 4: header = 0x1.924924p+0f; break;
                    case 5: header = 0x1.b6db6ep+0f; break;
                    case 6: header = 0x1.db6db8p+0f; break;
                    default: header = 0x1.000000p+1f; break;
                }
                const float dj = candidate_base * header;
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

#define HIPFIRE_GATED_NORM_MQ_ROTATE_AWQ 1
#define HIPFIRE_GATED_NORM_MQ_ROTATE_KERNEL gated_norm_mq_rotate_awq_i4_gfx12
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Batched gated RMSNorm + FWHT rotation + `block_i4_128` IU4 sidecar
// (exact gfx1201).
//
// gfx1201-fusion slice 4: LA post-GDN producer for the iu4 residual `wo`
// path. Replaces, per LA layer:
//   gated_norm_f32_batched  (grid [n_heads, N], block 32)
//   rotate_x_mq[_awq]_batched (grid [(K/256)*N], block 32)
//   quantize_int4_mmq_ds128 (inside the IU4 residual GEMM via ensure_int4_mmq_x)
// with ONE launch. Bit-exact contract: every f32 element must equal the old
// pipeline's gated_norm store reloaded by the rotate (the F32 store/load
// round trip is exact), and every 72-byte block must equal the standalone
// quantizer on that row (shared `quantize_block_i4_128_wave` recipe).
//
// Phase A replicates gated_norm_f32's per-lane accumulation and XOR
// reduction exactly (one wave32 per head, head_dim == 128 required, two
// waves per 256-group — the batched form of the decode
// gated_norm_mq_rotate_gfx1100 kernel). Phase B is mq_rotate_x's original
// lane ownership and butterfly verbatim, with the final F32 values held in
// registers: optionally stored to x_rot (byte-identical for surviving F32
// readers) and always remapped through `emit_iu4_sidecar_from_producer8`
// (prepended by the host JIT with
// `HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE=1`, same as the C2 producers).
//
// -DHIPFIRE_GATED_NORM_MQ_ROTATE_AWQ=1 builds the AWQ twin
// `gated_norm_mq_rotate_awq_i4_gfx12` (extra `awq_scale[K]` argument,
// divided BEFORE the FWHT — same formation as `rotate_x_mq_awq`,
// completing `(W·s)·(x/s)`). The divide is folded into the LDS staging
// exactly like the admitted f16 twin: `(normed*silu)/scale` in LDS, then
// `*signs1` in Phase B — identical expression order to the standalone
// chain, so bit-identical.
//
// Grid: [groups_per_row, N, 1] (groups_per_row = K/256 = n_heads/2).
// Block: [64, 1, 1]. LDS: 256 floats (one group, incumbent handoff).

#ifndef HIPFIRE_GATED_NORM_MQ_ROTATE_KERNEL
#define HIPFIRE_GATED_NORM_MQ_ROTATE_KERNEL gated_norm_mq_rotate_i4_gfx12
#endif

extern "C" __launch_bounds__(64, 8)
__global__ void HIPFIRE_GATED_NORM_MQ_ROTATE_KERNEL(
    const float* __restrict__ x,
    const float* __restrict__ z,
    const float* __restrict__ weight,
#ifdef HIPFIRE_GATED_NORM_MQ_ROTATE_AWQ
    const float* __restrict__ awq_scale,   // [K], FP32, unrotated basis
#endif
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,             // nullable when emit_f32=false
    block_i4_128* __restrict__ x_i4,       // [K/128, N] block_i4_128
    int n_heads,
    int head_dim,
    float eps,
    int K,
    int N)
{
    if (head_dim != 128) return;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    // Two heads per 256-group at head_dim == 128.
    if (wave >= 2) return;
    const int group = blockIdx.x;
    const int batch = blockIdx.y;
    const int groups_per_row = K / 256;
    if (group >= groups_per_row) return;
    const int head = group * 2 + wave;
    if (head >= n_heads) return;

    const long long batch_off = (long long)batch * K;
    const float* xh = x + batch_off + (long long)head * head_dim;
    const float* zh = z + batch_off + (long long)head * head_dim;
#ifdef HIPFIRE_GATED_NORM_MQ_ROTATE_AWQ
    const long long head_off = (long long)head * head_dim;
#endif

    __shared__ float normalized[256];
    float* oh = normalized + wave * head_dim;

    // Phase A: match gated_norm_f32's per-lane accumulation and XOR
    // reduction exactly.
    float sq_sum = 0.0f;
    for (int i = lane; i < head_dim; i += 32)
        sq_sum += xh[i] * xh[i];
    for (int o = 16; o > 0; o >>= 1)
        sq_sum += __shfl_xor(sq_sum, o);
    const float inv_rms = rsqrtf(sq_sum / (float)head_dim + eps);

    for (int i = lane; i < head_dim; i += 32) {
        const float normed = xh[i] * inv_rms * weight[i];
        const float z_val = zh[i];
        const float silu_z = z_val / (1.0f + expf(-z_val));
#ifdef HIPFIRE_GATED_NORM_MQ_ROTATE_AWQ
        // (normed*silu)/scale: matches rotate_x_mq_awq reading the stored
        // gated-norm output and computing (x/scale)*signs1.
        oh[i] = normed * silu_z / awq_scale[head_off + i];
#else
        oh[i] = normed * silu_z;
#endif
    }
    __syncthreads();

    // Phase B: mq_rotate_x's original lane ownership and butterfly.
    // Each 256-value MQ group spans exactly two normalized heads.
    if (wave != 0) return;
    const int d0 = lane * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(normalized + d0);
    const float4 x1 = *reinterpret_cast<const float4*>(normalized + d0 + 4);
    const float4 s10 = *reinterpret_cast<const float4*>(signs1 + d0);
    const float4 s11 = *reinterpret_cast<const float4*>(signs1 + d0 + 4);
    const float4 s20 = *reinterpret_cast<const float4*>(signs2 + d0);
    const float4 s21 = *reinterpret_cast<const float4*>(signs2 + d0 + 4);

    float v0 = x0.x * s10.x;
    float v1 = x0.y * s10.y;
    float v2 = x0.z * s10.z;
    float v3 = x0.w * s10.w;
    float v4 = x1.x * s11.x;
    float v5 = x1.y * s11.y;
    float v6 = x1.z * s11.z;
    float v7 = x1.w * s11.w;

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

    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (lane & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while (0)
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

    const float scale = 0.0625f;
    float f0 = v0 * scale * s20.x;
    float f1 = v1 * scale * s20.y;
    float f2 = v2 * scale * s20.z;
    float f3 = v3 * scale * s20.w;
    float f4 = v4 * scale * s21.x;
    float f5 = v5 * scale * s21.y;
    float f6 = v6 * scale * s21.z;
    float f7 = v7 * scale * s21.w;

    if (x_rot != nullptr) {
        float* out_b = x_rot + batch_off + (long long)group * 256;
        out_b[d0]     = f0;
        out_b[d0 + 1] = f1;
        out_b[d0 + 2] = f2;
        out_b[d0 + 3] = f3;
        out_b[d0 + 4] = f4;
        out_b[d0 + 5] = f5;
        out_b[d0 + 6] = f6;
        out_b[d0 + 7] = f7;
    }

    // IU4 sidecar: remap this wave's 8-floats/lane into the standalone
    // quant ownership and emit two block_i4_128 rows for this 256-group.
    // token = batch; N is the batch extent for the [K/128, N] layout.
    emit_iu4_sidecar_from_producer8(
        f0, f1, f2, f3, f4, f5, f6, f7,
        x_i4, group, batch, N, lane);
}
