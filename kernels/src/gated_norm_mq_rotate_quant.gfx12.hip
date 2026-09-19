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
