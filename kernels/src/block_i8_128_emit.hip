// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Fused-producer emit of `block_i8_128` for the gfx12 A8 route. The host JIT
// concatenates, in order: `block_i8_128_quant.hip` (with
// HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE), this file, the producer's
// defines, and one unmodified A4 producer body (`fused_rmsnorm_mq_rotate.hip`,
// `fused_silu_mul_mq_rotate_awq.hip`, `gated_norm_mq_rotate_quant.gfx12.hip`,
// `mq_rotate_x_i4.hip`). Those bodies name their sidecar `block_i4_128` and
// emit it through `emit_iu4_sidecar_from_producer8`; the two aliases at the
// end of this file rebind both names, so everything up to the emit (norm,
// sigmoid, SiLU, AWQ divide, FWHT) is the A4 producer's source verbatim.
//
// Ownership: every A4 producer calls the emit with lane p of a wave holding
// the 8 consecutive producer-final values 8p..8p+7 of one 256-group, so lanes
// 0-15 form 128-block 2*group and lanes 16-31 form 2*group+1, and lane
// j = p & 15 owns exactly the 8-element K group j of its block. The block
// reductions run over the 16-lane DPP row. qs bytes 8j..8j+7 of that group
// are {q0, q2, q4, q6 | q1, q3, q5, q7} (the nibble-pair order): one b64
// store per lane, no cross-lane byte shuffle.
//
// Bit identity with `quantize_int8_mmq_ds128` on the same f32 values:
// amax is a max (NaN-ignoring fmaxf, fabsf has no -0) and s an exact int
// sum, so the reduction order is free; d and every q use the standalone's
// per-element expressions (IEEE `__fdiv_rn`, rintf, clamp), zero block
// d = 0, q = 0.

#ifndef HIPFIRE_BLOCK_I8_128_EMIT_HIP
#define HIPFIRE_BLOCK_I8_128_EMIT_HIP

#ifndef HIPFIRE_BLOCK_I8_128_QUANT_HIP
#error "block_i8_128_emit.hip requires block_i8_128_quant.hip first"
#endif
#ifdef HIPFIRE_BLOCK_I4_128_QUANT_HIP
#error "block_i8_128_emit.hip rebinds block_i4_128; do not concatenate block_i4_128_quant.hip"
#endif

template <int Mask>
__device__ __forceinline__ float i8_row_xmask_f32(float v) {
    return __int_as_float(
        __builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 | Mask, 0xF, 0xF, true));
}
template <int Mask>
__device__ __forceinline__ int i8_row_xmask_i32(int v) {
    return __builtin_amdgcn_update_dpp(0, v, 0x160 | Mask, 0xF, 0xF, true);
}

// Same arguments as `emit_iu4_sidecar_from_producer8`; Y is [K/128 x N]
// block_i8_128, block index (2*group + half)*N + token.
__device__ __forceinline__ void emit_i8_sidecar_from_producer8(
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
    amax = fmaxf(amax, i8_row_xmask_f32<8>(amax));
    amax = fmaxf(amax, i8_row_xmask_f32<4>(amax));
    amax = fmaxf(amax, i8_row_xmask_f32<2>(amax));
    amax = fmaxf(amax, i8_row_xmask_f32<1>(amax));
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
    s += i8_row_xmask_i32<8>(s);
    s += i8_row_xmask_i32<4>(s);
    s += i8_row_xmask_i32<2>(s);
    s += i8_row_xmask_i32<1>(s);

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

#define block_i4_128 block_i8_128
#define emit_iu4_sidecar_from_producer8 emit_i8_sidecar_from_producer8

#endif // HIPFIRE_BLOCK_I8_128_EMIT_HIP
