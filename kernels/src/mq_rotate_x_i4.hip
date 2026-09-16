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
// This kernel keeps the f32 store (every downstream reader is preserved
// byte-for-byte) and additionally emits the Xq sidecar in-register, so the
// IU4 residual consumer takes an `Int4MmqPrepared` and the quantize launch
// disappears.
//
// -DHIPFIRE_ROTATE_AWQ=1 builds the AWQ twin `rotate_x_mq_awq_i4` (extra
// `awq_scale[K]` argument, divided BEFORE the FWHT — same formation as
// `rotate_x_mq_awq`, completing `(W·s)·(x/s)`).
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
#ifdef HIPFIRE_ROTATE_AWQ
    const float* __restrict__ awq_scale,   // [K], FP32
#endif
    float* __restrict__ x_out,             // [N, K] f32, always written
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
          float* x_out_b = x_out + batch_off;

    int d0 = tid * 8;
    int base = group * 256 + d0;

#ifdef HIPFIRE_ROTATE_AWQ
    // Exact order of `rotate_x_mq_awq`: (x / awq_scale) * sign.
    float v0 = (x_in_b[base    ] / awq_scale[base    ]) * signs1[d0    ];
    float v1 = (x_in_b[base + 1] / awq_scale[base + 1]) * signs1[d0 + 1];
    float v2 = (x_in_b[base + 2] / awq_scale[base + 2]) * signs1[d0 + 2];
    float v3 = (x_in_b[base + 3] / awq_scale[base + 3]) * signs1[d0 + 3];
    float v4 = (x_in_b[base + 4] / awq_scale[base + 4]) * signs1[d0 + 4];
    float v5 = (x_in_b[base + 5] / awq_scale[base + 5]) * signs1[d0 + 5];
    float v6 = (x_in_b[base + 6] / awq_scale[base + 6]) * signs1[d0 + 6];
    float v7 = (x_in_b[base + 7] / awq_scale[base + 7]) * signs1[d0 + 7];
#else
    float v0 = x_in_b[base    ] * signs1[d0    ];
    float v1 = x_in_b[base + 1] * signs1[d0 + 1];
    float v2 = x_in_b[base + 2] * signs1[d0 + 2];
    float v3 = x_in_b[base + 3] * signs1[d0 + 3];
    float v4 = x_in_b[base + 4] * signs1[d0 + 4];
    float v5 = x_in_b[base + 5] * signs1[d0 + 5];
    float v6 = x_in_b[base + 6] * signs1[d0 + 6];
    float v7 = x_in_b[base + 7] * signs1[d0 + 7];
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

    x_out_b[base    ] = f0;
    x_out_b[base + 1] = f1;
    x_out_b[base + 2] = f2;
    x_out_b[base + 3] = f3;
    x_out_b[base + 4] = f4;
    x_out_b[base + 5] = f5;
    x_out_b[base + 6] = f6;
    x_out_b[base + 7] = f7;

    // IU4 sidecar: remap this wave's 8-floats/lane into the standalone
    // quant ownership and emit two block_i4_128 rows for this 256-group.
    // token = batch; N is the batch extent for the [K/128, N] layout.
    emit_iu4_sidecar_from_producer8(
        f0, f1, f2, f3, f4, f5, f6, f7,
        x_i4, group, batch, N, tid);
}
