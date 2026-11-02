// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// `mq_rotate_x_i4_f16out` (the Qwen4 MoE front producer: plain FWHT256 rotate +
// `block_i4_128` IU4 sidecar + the RNE F16 cast of the raw row) with a THIRD output: the
// A8 `block_i8_128` blocks of the same rotated row, the bytes `mq_rotate_x_i8_gfx12` emits
// (`Gpu::mq6_i8_prepare_x`), for the shared-expert INT8 route
// (`HIPFIRE_QWEN4_SHARED_INT8`, `shared_i8.rs`): the one read of the F32 activation row feeds
// the router / selector F16 X, the routed gate/up IU4 sidecar and the shared gate/up INT8 GEMM.
//
// The host concatenates `block_i4_128_quant.hip` (HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE),
// `block_i8_128_quant.hip` (HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE) and this file; the
// body up to the emits is a verbatim copy of `mq_rotate_x_i4.hip`'s plain, F16-out path, so
// the F16 X and the IU4 sidecar are byte-identical to `mq_rotate_x_i4_f16out`. The A8 emit is
// `block_i8_128_emit.hip`'s `emit_i8_sidecar_from_producer8` under another name (that header
// rebinds the `block_i4_128` names, so it cannot be concatenated beside the IU4 quantizer).
//
// Arguments (after `x_i4`: `x_i8`): x_in, x_out (null: unused), x_f16, signs1, signs2, x_i4,
// x_i8, K, N. Grid [(K/256)*N, 1, 1], block 32.

#if defined(__gfx1201__) || defined(__gfx1200__)

template <int Mask>
__device__ __forceinline__ float i8d_row_xmask_f32(float v) {
    return __int_as_float(
        __builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 | Mask, 0xF, 0xF, true));
}
template <int Mask>
__device__ __forceinline__ int i8d_row_xmask_i32(int v) {
    return __builtin_amdgcn_update_dpp(0, v, 0x160 | Mask, 0xF, 0xF, true);
}

// `emit_i8_sidecar_from_producer8` (block_i8_128_emit.hip): lane p holds the 8 consecutive
// producer-final values 8p..8p+7 of one 256-group; Y is [K/128 x N] block_i8_128, block index
// (2*group + half)*N + token.
__device__ __forceinline__ void i8d_emit_from_producer8(
    float f0, float f1, float f2, float f3,
    float f4, float f5, float f6, float f7,
    block_i8_128* __restrict__ Y,
    int group,
    int token,
    int N,
    int lane
) {
    const float x[8] = {f0, f1, f2, f3, f4, f5, f6, f7};
    float amax = fmaxf(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3]))),
                       fmaxf(fmaxf(fabsf(x[4]), fabsf(x[5])), fmaxf(fabsf(x[6]), fabsf(x[7]))));
    amax = fmaxf(amax, i8d_row_xmask_f32<8>(amax));
    amax = fmaxf(amax, i8d_row_xmask_f32<4>(amax));
    amax = fmaxf(amax, i8d_row_xmask_f32<2>(amax));
    amax = fmaxf(amax, i8d_row_xmask_f32<1>(amax));
    const bool zero = amax == 0.0f;
    const float d = zero ? 0.0f : __fdiv_rn(amax, 127.0f);

    int q[8];
    int s = 0;
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float qf = zero ? 0.0f : fminf(fmaxf(rintf(__fdiv_rn(x[e], d)), -127.0f), 127.0f);
        q[e] = (int)qf;
        s += q[e];
    }
    s += i8d_row_xmask_i32<8>(s);
    s += i8d_row_xmask_i32<4>(s);
    s += i8d_row_xmask_i32<2>(s);
    s += i8d_row_xmask_i32<1>(s);

    const unsigned lo = ((unsigned)q[0] & 255u) | (((unsigned)q[2] & 255u) << 8)
                      | (((unsigned)q[4] & 255u) << 16) | ((unsigned)q[6] << 24);
    const unsigned hi = ((unsigned)q[1] & 255u) | (((unsigned)q[3] & 255u) << 8)
                      | (((unsigned)q[5] & 255u) << 16) | ((unsigned)q[7] << 24);
    block_i8_128* out = Y + (long long)(2 * group + (lane >> 4)) * N + token;
    const int j = lane & 15;
    reinterpret_cast<uint2*>(out->qs)[j] = make_uint2(lo, hi);
    if (j == 0) {
        *reinterpret_cast<uint2*>(out) = make_uint2(__float_as_uint(d), (unsigned)s);
    }
}

__launch_bounds__(32, 16)
extern "C" __global__ void mq_rotate_x_i4_f16out_i8(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,             // nullable
    _Float16* __restrict__ x_f16,          // [N, K] F16 = (_Float16) of the UNROTATED x_in
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    block_i4_128* __restrict__ x_i4,       // [K/128, N] block_i4_128
    block_i8_128* __restrict__ x_i8,       // [K/128, N] block_i8_128
    int K,
    int N
) {
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    const int linear_group = blockIdx.x;
    const int batch = linear_group / groups_total;
    const int group = linear_group - batch * groups_total;

    const long long batch_off = (long long)batch * K;
    const float* x_in_b  = x_in  + batch_off;
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
    {
        typedef _Float16 hipfire_h8 __attribute__((ext_vector_type(8)));
        hipfire_h8 h = {(_Float16)x0, (_Float16)x1, (_Float16)x2, (_Float16)x3,
                        (_Float16)x4, (_Float16)x5, (_Float16)x6, (_Float16)x7};
        *reinterpret_cast<hipfire_h8*>(x_f16 + batch_off + base) = h;
    }

    float v0 = x0 * signs1[d0    ];
    float v1 = x1 * signs1[d0 + 1];
    float v2 = x2 * signs1[d0 + 2];
    float v3 = x3 * signs1[d0 + 3];
    float v4 = x4 * signs1[d0 + 4];
    float v5 = x5 * signs1[d0 + 5];
    float v6 = x6 * signs1[d0 + 6];
    float v7 = x7 * signs1[d0 + 7];

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

    emit_iu4_sidecar_from_producer8(
        f0, f1, f2, f3, f4, f5, f6, f7,
        x_i4, group, batch, N, tid);
    i8d_emit_from_producer8(
        f0, f1, f2, f3, f4, f5, f6, f7,
        x_i8, group, batch, N, tid);
}

#else

extern "C" __global__ void mq_rotate_x_i4_f16out_i8(
    const float*, float*, void*, const float*, const float*, void*, void*, int, int
) {}

#endif
