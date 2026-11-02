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
// -DHIPFIRE_ROTATE_GATE_IL=1 (with the gate) reads it in place from the FA
// Q/gate projection rows `[N x K/256 x (256 q, 256 gate)]` that
// `qwen35_fa_prep_fp8q_nogate_batched_gfx1201` leaves unconsumed, instead of
// a compact `[N x K]` gate copy: the same f32 values, so identical bytes.
//
// Bit-exactness of the f32 path: the FWHT body is a verbatim copy of
// `mq_rotate_x` / `rotate_x_mq_awq` (same grid, same multiply order, same
// butterfly, same scale), and f32 store→load is lossless, so the registers
// hold exactly what the standalone quantizer would reload. Lane ownership
// matches the C2 producers because both hold 8 consecutive values per lane
// (`d0 = tid * 8`, `base = group * 256 + d0`).
//
// The sidecar goes through `emit_iu4_sidecar_from_producer8` from the shared
// recipe (prepended by the host JIT with
// `HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE=1`). Its Xq bytes depend on the
// candidate policy: the compiled emit (legacy; `cand_mode` 0) uses the build's
// `IU4_A4_CANDIDATES` (gfx12 C2 {5,7}), so it matches the C2 producers but is
// NOT in general byte-identical to the standalone quantizer. Only the
// `cand_mode == 8` exact path (`emit_iu4_sidecar_from_producer8<true>`: the
// same lane remap, then `quantize_block_i4_128_wave<false>`, the 8-candidate
// strict-`<` MSE → rintf/clamp → exact int sum) matches the standalone
// quantizer's bytes.
//
// -DHIPFIRE_ROTATE_F16_OUT=1 (plain rotate only) adds an `x_f16` output: the
// RNE F16 cast of the raw, UNROTATED input row (the bytes `convert_f32_to_f16`
// leaves for the BF16 router / shared-expert F16 WMMA GEMMs), stored from the
// registers the lane already loaded. One launch then emits both the F16 X and
// the IU4 sidecar of a row read once (`mq_rotate_x_i4_f16out`).
#if defined(HIPFIRE_ROTATE_F16_OUT) && \
    (defined(HIPFIRE_ROTATE_AWQ) || defined(HIPFIRE_ROTATE_SIGMOID_GATE))
#error "HIPFIRE_ROTATE_F16_OUT is the plain-rotate twin only"
#endif

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
    const float* __restrict__ gate,        // [N, K] FP32; GATE_IL: [N, 2K] Q/gate rows
#endif
#ifdef HIPFIRE_ROTATE_AWQ
    const float* __restrict__ awq_scale,   // [K], FP32
#endif
    float* __restrict__ x_out,             // nullable when emit_f32=false
#ifdef HIPFIRE_ROTATE_F16_OUT
    _Float16* __restrict__ x_f16,          // [N, K] F16 = (_Float16) of the UNROTATED x_in
#endif
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    block_i4_128* __restrict__ x_i4,       // [K/128, N] block_i4_128
    int K,
    int N
#ifdef HIPFIRE_IU4_RUNTIME_CAND
    , int cand_mode                        // 0: compiled emit (legacy); 8: exact reference8
#endif
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
#ifdef HIPFIRE_ROTATE_GATE_IL
    // Group g's gate is floats [512 g + 256, 512 g + 512) of the 2K-float
    // row; `base` below adds the other `g * 256 + d0`.
    const float* gate_b = gate + (long long)batch * (2LL * K) + (group + 1) * 256;
#else
    const float* gate_b = gate + batch_off;
#endif
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
#ifdef HIPFIRE_ROTATE_F16_OUT
    // `convert_f32_to_f16` of the raw (unrotated) row, from the registers this lane
    // already holds: one 16-byte store per lane, no extra DRAM read. The cast is
    // the same RNE `(_Float16)` the standalone conversion applies per element.
    {
        typedef _Float16 hipfire_h8 __attribute__((ext_vector_type(8)));
        hipfire_h8 h = {(_Float16)x0, (_Float16)x1, (_Float16)x2, (_Float16)x3,
                        (_Float16)x4, (_Float16)x5, (_Float16)x6, (_Float16)x7};
        *reinterpret_cast<hipfire_h8*>(x_f16 + batch_off + base) = h;
    }
#endif
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

#ifdef HIPFIRE_IU4_RUNTIME_CAND
    // -DHIPFIRE_IU4_RUNTIME_CAND=1 (only the two gfx12 `mq_rotate_x_i4_gfx12`
    // / `rotate_x_mq_awq_i4_gfx12` host concats) appends a trailing
    // `int cand_mode` kernel argument (wave-uniform). 8 selects the
    // standalone-quantizer reference through the shared
    // `emit_iu4_sidecar_from_producer8<true>` (same f0..f7 remap, then
    // `quantize_block_i4_128_wave<false>`); any other value runs the compiled
    // emit below, unchanged. The FWHT/f32 body above is identical in both modes.
    if (cand_mode == 8) {
        emit_iu4_sidecar_from_producer8<true>(
            f0, f1, f2, f3, f4, f5, f6, f7,
            x_i4, group, batch, N, tid);
        return;
    }
#endif
    // IU4 sidecar: remap this wave's 8-floats/lane into the standalone
    // quant ownership and emit two block_i4_128 rows for this 256-group.
    // token = batch; N is the batch extent for the [K/128, N] layout.
    emit_iu4_sidecar_from_producer8(
        f0, f1, f2, f3, f4, f5, f6, f7,
        x_i4, group, batch, N, tid);
}
