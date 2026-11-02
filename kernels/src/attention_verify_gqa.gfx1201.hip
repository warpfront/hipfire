// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// VerifyAttn Stage 0 (gfx1201): the batched-verify flash tile with one KV
// scan shared by all 6 q heads x 8 query rows of a kv head, split over a
// fixed, context-independent grid.
//
// `attention_verify_gqa_{q8,fp8}_gfx1201` writes the same
// `[row][head][max_tiles][2 + 256]` f32 partials as
// `attention_flash_{q8_0,fp8_e4m3}_tile_batched`, byte for byte, and
// `attention_verify_reduce_gfx1201` reduces them byte-identically to
// `attention_flash_asym_reduce_batched`. Shape is compile-time: head_dim 256,
// GQA group 6 (8 rows per workgroup; fp8 and Q8) or, fp8 only, GQA group 8
// (`attention_verify_gqa_fp8_gqa8_gfx1201`, 6 rows per workgroup), tile 128,
// full causal, legacy single-slot KV (the host admits nothing else).
//
// Reference arithmetic reproduced (hipcc -O3 gfx1201 ISA of the tile_batched
// pair; reference lane l covers dims 8l..8l+7):
//   * kq = ks * dec(code), exact in f32 (Q8: f16 block scale of block l/4;
//     fp8: the head's f16 scale; the hardware fp8 decode equals the reference
//     software decode for every non-NaN code);
//   * leaf_l = fma(q7,kq7, ... fma(q1,kq1, fma(q0,kq0, +0))) in dim order;
//   * score = (xor-16, 8, 4, 2, 1 butterfly sum of the 32 leaves) * scale;
//   * softmax per tile: the reference's lane-strided max / expf / sum source;
//   * PV: out_d = fma(e_t, vq_{t,d}, out_d) over keys in increasing order,
//     keys at or past the row's causal bound skipped (never multiplied by 0).
// Phase A layout: a 16-lane row scores one key per pass; row lane r holds
// reference lanes r (dims 8r..8r+7) and r+16 (dims 128+8r..), so the xor-16
// step is in-thread and 8/4/2/1 are DPP row_xmask adds. Row 0 of a pass
// scores key 32m+r', row 1 key 32m+16+r'.
//
// Every wave owns 6 (row, head) pairs for the whole tile (scores, softmax and
// PV), so the tile loop has no workgroup barrier; the 8 waves of a workgroup
// read the same K/V bytes through L0.
//
// fwht4 / fwht3 K twins (`attention_verify_gqa_{fwht4,fwht3}_gfx1201`; V is
// Q8_0, the host admits nothing else) write the same partials as
// `attention_flash_{fwht4,fwht3}_tile_batched` (v_mode 8) byte for byte, via
// the reference's hipcc -O3 gfx1201 ISA arithmetic:
//   * Q is f32 un-rotated; per (row, head) pair the 32 lanes run the
//     reference's own signed-FWHT (`fwht_shfl_forward` 128-wide per half for
//     fwht4, `fwht_shfl_forward_256` for fwht3: same source, same ds_swizzle
//     butterfly, same lane layout: reference lane l owns dims half*128+4l..
//     (fwht4) / 8l.. (fwht3)). The rotated register sets of reference lanes
//     rl and rl+16 are then gathered into row lane rl (a xor-16 shuffle).
//   * kq_i = cnorm * C[idx_i] (f32 product, C = TURBO_C4 / TURBO_C3_256, same
//     literals; looked up through an LDS table of idx pairs), and the
//     reference lane's score partial is the reference's contracted chain:
//       fwht4: half leaf = fma(m3,k3, fma(m2,k2, fma(m0,k0, m1*k1)));
//              partial   = (0 + leaf(half 0)) + leaf(half 1)
//       fwht3: partial   = fma(m7,k7, fma(m6,k6, fma(m5,k5, fma(m4,k4,
//                            fma(m3,k3, fma(m2,k2, fma(m0,k0, m1*k1)))))))
//     (m1*k1 is the one plain multiply: fp-contract fuses the first add of
//     the source expression onto the m0 product). The xor-16/8/4/2/1 sum,
//     scale, softmax and PV are the Q8/fp8 twin's.
//   * V (Q8_0): fwht3 reference lane l owns dims 8l..8l+7 (the Q8 layout);
//     fwht4 reference lane l owns dims 4l..4l+3 and 128+4l..128+4l+3, loaded
//     as two 4-code groups of their own blocks; partials are written in that
//     dim order.
#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "attention_verify_gqa_gfx1201 requires --offload-arch=gfx1201"
#endif

#define VA_TILE 128
#define VA_HD 256
#define VA_STRIDE (2 + VA_HD)
#define VA_GROUP 6
#define VA_WAVES 8
#define VA_THREADS (VA_WAVES * 32)
#define VA_ROWS 8
// Group-8 shape (A3B, H16/KV2): 6 rows x 8 heads = 48 pairs = 6 per wave.
#define VA_GROUP8 8
#define VA_ROWS8 6
// (row, head) pairs per wave: 8 rows x 6 heads over 8 waves.
#define VA_PPW (VA_ROWS * VA_GROUP / VA_WAVES)
// The pair loops are specialised for 1..6 pairs per wave (VA_TILES below);
// every row/group shape must land on exactly VA_PPW pairs per wave.
static_assert(VA_ROWS * VA_GROUP == VA_WAVES * VA_PPW && VA_ROWS8 * VA_GROUP8 == VA_WAVES * VA_PPW &&
                  VA_PPW == 6,
              "VerifyAttn row groups must hold 6 (row, head) pairs per wave");
// Buffer descriptor dword3 for gfx12 raw buffer loads (as the decode tile).
#define VA_RSRC_FLAGS 0x31004000u

enum { VA_KV_Q8 = 0, VA_KV_FP8 = 1, VA_KV_FWHT4 = 2, VA_KV_FWHT3 = 3 };

template <int Ctrl>
__device__ __forceinline__ float va_row_xmask_add(float v) {
    const int other = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), Ctrl, 0xf, 0xf, false);
    return v + __builtin_bit_cast(float, other);
}

__device__ __forceinline__ float va_f16(unsigned short b) {
    return (float)__builtin_bit_cast(_Float16, b);
}

// Eight signed Q8 codes times their block scale (exact f32 products).
__device__ __forceinline__ void va_dequant_q8(unsigned int lo, unsigned int hi, float s, float (&out)[8]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = s * (float)(signed char)(lo >> (8 * i));
#pragma unroll
    for (int i = 0; i < 4; ++i) out[4 + i] = s * (float)(signed char)(hi >> (8 * i));
}

// Eight E4M3 codes times the head scale (exact f32 products).
__device__ __forceinline__ void va_dequant_fp8(unsigned int lo, unsigned int hi, float s, float (&out)[8]) {
    typedef float f32x2 __attribute__((ext_vector_type(2)));
    const f32x2 a = __builtin_amdgcn_cvt_pk_f32_fp8(lo, false);
    const f32x2 b = __builtin_amdgcn_cvt_pk_f32_fp8(lo, true);
    const f32x2 c = __builtin_amdgcn_cvt_pk_f32_fp8(hi, false);
    const f32x2 d = __builtin_amdgcn_cvt_pk_f32_fp8(hi, true);
    out[0] = a.x * s; out[1] = a.y * s; out[2] = b.x * s; out[3] = b.y * s;
    out[4] = c.x * s; out[5] = c.y * s; out[6] = d.x * s; out[7] = d.y * s;
}

// One reference-lane leaf: sequential fma over its 8 dims from +0.
__device__ __forceinline__ float va_leaf(const float (&q)[8], const float (&k)[8]) {
    float p = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) p = __builtin_fmaf(q[i], k[i], p);
    return p;
}

// K/V bytes of one slice of one key. Q8 / fp8 (and the Q8 V of fwht3): the
// 8 codes of an 8-dim slice (lo, hi) and the scale (sc). fwht K: lo = the
// reference lane's packed indices, cn = the head's f32 cnorm (bits). fwht4 V:
// lo / hi = the 4 codes of the two dim groups, sc / sc2 their block scales.
struct VaSlice {
    unsigned int lo, hi;
    unsigned short sc, sc2;
    unsigned int cn;
};

// Byte offsets (from the tile's first row) of the 8-dim slice starting at
// `dim` of kv head `kv_h`, key `t`.
template <int KV>
__device__ __forceinline__ VaSlice va_load_slice(__amdgpu_buffer_rsrc_t rsrc, unsigned row_off, int kv_h,
                                                 int n_kv_heads, int dim) {
    VaSlice s;
    if (KV == VA_KV_Q8) {
        const unsigned blk = row_off + (unsigned)(kv_h * (VA_HD / 32) + (dim >> 5)) * 34u;
        const auto w = __builtin_amdgcn_raw_buffer_load_b64(rsrc, blk + 2u + (unsigned)(dim & 31), 0, 0);
        s.lo = w[0];
        s.hi = w[1];
        s.sc = __builtin_amdgcn_raw_buffer_load_b16(rsrc, blk, 0, 0);
    } else {
        const auto w = __builtin_amdgcn_raw_buffer_load_b64(rsrc, row_off + (unsigned)(kv_h * VA_HD + dim), 0, 0);
        s.lo = w[0];
        s.hi = w[1];
        s.sc = __builtin_amdgcn_raw_buffer_load_b16(
            rsrc, row_off + (unsigned)(n_kv_heads * VA_HD + kv_h * 2), 0, 0);
    }
    return s;
}

template <int KV>
__device__ __forceinline__ void va_dequant(const VaSlice& s, float (&out)[8]) {
    if (KV == VA_KV_Q8)
        va_dequant_q8(s.lo, s.hi, va_f16(s.sc), out);
    else
        va_dequant_fp8(s.lo, s.hi, va_f16(s.sc), out);
}

// ── fwht K: Lloyd-Max centroids (turbo_common.h TURBO_C4 / TURBO_C3_256, the
// same literals) and the reference's signed-FWHT Q rotation (verbatim
// arithmetic of turbo_common.h fwht_shfl_forward / fwht_shfl_forward_256).
__constant__ float VA_C4[16] = {
    -0.241565f, -0.182875f, -0.143012f, -0.111016f, -0.083262f, -0.057983f, -0.034295f, -0.011225f,
     0.011225f,  0.034295f,  0.057983f,  0.083262f,  0.111016f,  0.143012f,  0.182875f,  0.241565f
};
__constant__ float VA_C3_256[8] = {-0.134860f, -0.083320f, -0.046469f, -0.015176f,
                                   0.015176f,  0.046469f,  0.083320f,  0.134860f};

#define VA_HADAMARD_BFLY(v, pattern, stride, tid) do {                  \
    float _p = __int_as_float(                                          \
        __builtin_amdgcn_ds_swizzle(__float_as_int(v), (pattern)));     \
    if ((tid) & (stride)) { (v) = _p - (v); }                           \
    else                  { (v) = (v) + _p; }                           \
} while(0)

__device__ __forceinline__ void va_fwht_forward_128(float& a, float& b, float& c, float& d,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int tid)
{
    int d0 = tid * 4;
    a *= signs1[d0]; b *= signs1[d0+1]; c *= signs1[d0+2]; d *= signs1[d0+3];

    float t;
    t = a; a = a + b; b = t - b;
    t = c; c = c + d; d = t - d;

    t = a; a = a + c; c = t - c;
    t = b; b = b + d; d = t - d;

    VA_HADAMARD_BFLY(a, 0x041F, 1, tid); VA_HADAMARD_BFLY(b, 0x041F, 1, tid);
    VA_HADAMARD_BFLY(c, 0x041F, 1, tid); VA_HADAMARD_BFLY(d, 0x041F, 1, tid);
    VA_HADAMARD_BFLY(a, 0x081F, 2, tid); VA_HADAMARD_BFLY(b, 0x081F, 2, tid);
    VA_HADAMARD_BFLY(c, 0x081F, 2, tid); VA_HADAMARD_BFLY(d, 0x081F, 2, tid);
    VA_HADAMARD_BFLY(a, 0x101F, 4, tid); VA_HADAMARD_BFLY(b, 0x101F, 4, tid);
    VA_HADAMARD_BFLY(c, 0x101F, 4, tid); VA_HADAMARD_BFLY(d, 0x101F, 4, tid);
    VA_HADAMARD_BFLY(a, 0x201F, 8, tid); VA_HADAMARD_BFLY(b, 0x201F, 8, tid);
    VA_HADAMARD_BFLY(c, 0x201F, 8, tid); VA_HADAMARD_BFLY(d, 0x201F, 8, tid);
    VA_HADAMARD_BFLY(a, 0x401F, 16, tid); VA_HADAMARD_BFLY(b, 0x401F, 16, tid);
    VA_HADAMARD_BFLY(c, 0x401F, 16, tid); VA_HADAMARD_BFLY(d, 0x401F, 16, tid);

    const float s = 0.08838834764831845f;
    a *= s * signs2[d0]; b *= s * signs2[d0+1]; c *= s * signs2[d0+2]; d *= s * signs2[d0+3];
}

__device__ __forceinline__ void va_fwht_forward_256(
    float& v0, float& v1, float& v2, float& v3,
    float& v4, float& v5, float& v6, float& v7,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int tid)
{
    const int d0 = tid * 8;

    v0 *= signs1[d0];   v1 *= signs1[d0+1]; v2 *= signs1[d0+2]; v3 *= signs1[d0+3];
    v4 *= signs1[d0+4]; v5 *= signs1[d0+5]; v6 *= signs1[d0+6]; v7 *= signs1[d0+7];

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

    #define VA_HBFLY8(pat, str) \
        VA_HADAMARD_BFLY(v0,(pat),(str),tid); VA_HADAMARD_BFLY(v1,(pat),(str),tid); \
        VA_HADAMARD_BFLY(v2,(pat),(str),tid); VA_HADAMARD_BFLY(v3,(pat),(str),tid); \
        VA_HADAMARD_BFLY(v4,(pat),(str),tid); VA_HADAMARD_BFLY(v5,(pat),(str),tid); \
        VA_HADAMARD_BFLY(v6,(pat),(str),tid); VA_HADAMARD_BFLY(v7,(pat),(str),tid)
    VA_HBFLY8(0x041F, 1);
    VA_HBFLY8(0x081F, 2);
    VA_HBFLY8(0x101F, 4);
    VA_HBFLY8(0x201F, 8);
    VA_HBFLY8(0x401F, 16);
    #undef VA_HBFLY8

    const float s = 0.0625f;
    v0 *= s * signs2[d0];   v1 *= s * signs2[d0+1]; v2 *= s * signs2[d0+2]; v3 *= s * signs2[d0+3];
    v4 *= s * signs2[d0+4]; v5 *= s * signs2[d0+5]; v6 *= s * signs2[d0+6]; v7 *= s * signs2[d0+7];
}

// One (row, head) pair's Q for this lane's 16-lane row: qa = reference lane
// rl, qb = reference lane rl + 16 (Q8 / fp8: dims 8rl.. and 128+8rl..; fwht:
// the rotated mq registers of those reference lanes). Executed by all 32
// lanes of the wave (the rotation is a wave-wide butterfly).
template <int KV>
__device__ __forceinline__ void va_load_q(const float* __restrict__ qr, const float* __restrict__ signs1,
                                          const float* __restrict__ signs2, float (&qa)[8], float (&qb)[8]) {
    const int lane = (int)threadIdx.x & 31;
    if (KV == VA_KV_FWHT4 || KV == VA_KV_FWHT3) {
        float mine[8];
        if (KV == VA_KV_FWHT4) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int d0 = half * 128 + lane * 4;
                float a = qr[d0 + 0];
                float b = qr[d0 + 1];
                float c = qr[d0 + 2];
                float d = qr[d0 + 3];
                va_fwht_forward_128(a, b, c, d, signs1, signs2, lane);
                mine[half * 4 + 0] = a;
                mine[half * 4 + 1] = b;
                mine[half * 4 + 2] = c;
                mine[half * 4 + 3] = d;
            }
        } else {
            const int d0 = lane * 8;
            mine[0] = qr[d0 + 0]; mine[1] = qr[d0 + 1];
            mine[2] = qr[d0 + 2]; mine[3] = qr[d0 + 3];
            mine[4] = qr[d0 + 4]; mine[5] = qr[d0 + 5];
            mine[6] = qr[d0 + 6]; mine[7] = qr[d0 + 7];
            va_fwht_forward_256(mine[0], mine[1], mine[2], mine[3], mine[4], mine[5], mine[6], mine[7],
                                signs1, signs2, lane);
        }
        const bool hi = lane >= 16;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float other = __shfl_xor(mine[i], 16);
            qa[i] = hi ? other : mine[i];
            qb[i] = hi ? mine[i] : other;
        }
    } else {
        const int rl = lane & 15;
        const float4 a0 = *(const float4*)(qr + 8 * rl);
        const float4 a1 = *(const float4*)(qr + 8 * rl + 4);
        const float4 b0 = *(const float4*)(qr + 128 + 8 * rl);
        const float4 b1 = *(const float4*)(qr + 128 + 8 * rl + 4);
        qa[0] = a0.x; qa[1] = a0.y; qa[2] = a0.z; qa[3] = a0.w;
        qa[4] = a1.x; qa[5] = a1.y; qa[6] = a1.z; qa[7] = a1.w;
        qb[0] = b0.x; qb[1] = b0.y; qb[2] = b0.z; qb[3] = b0.w;
        qb[4] = b1.x; qb[5] = b1.y; qb[6] = b1.z; qb[7] = b1.w;
    }
}

// K slice of reference lane `rl + 16 * which` of one key. Q8 / fp8: its 8
// dims (8l..8l+7). fwht4 (132 B/head: f32 cnorm, 2 x 64 B nibble planes):
// the lane's two 16-bit nibble groups (plane h at 4 + 64h + 2l). fwht3
// (100 B/head: f32 cnorm, 96 B of 3-bit indices): the 24 bits at byte
// 4 + 3l, from two dword loads (never past the head's record).
template <int KV>
__device__ __forceinline__ VaSlice va_load_k(__amdgpu_buffer_rsrc_t rsrc, unsigned row_off, int kv_h,
                                             int n_kv_heads, int rl, int which) {
    if (KV == VA_KV_FWHT4) {
        VaSlice s = {};
        const unsigned hb = row_off + (unsigned)kv_h * 132u;
        const unsigned l = (unsigned)(rl + 16 * which);
        s.cn = __builtin_amdgcn_raw_buffer_load_b32(rsrc, hb, 0, 0);
        const unsigned p0 = __builtin_amdgcn_raw_buffer_load_b16(rsrc, hb + 4u + 2u * l, 0, 0);
        const unsigned p1 = __builtin_amdgcn_raw_buffer_load_b16(rsrc, hb + 68u + 2u * l, 0, 0);
        s.lo = p0 | (p1 << 16);
        return s;
    } else if (KV == VA_KV_FWHT3) {
        VaSlice s = {};
        const unsigned hb = row_off + (unsigned)kv_h * 100u;
        const unsigned l = (unsigned)(rl + 16 * which);
        const unsigned off = 4u + 3u * l;
        s.cn = __builtin_amdgcn_raw_buffer_load_b32(rsrc, hb, 0, 0);
        const unsigned a = __builtin_amdgcn_raw_buffer_load_b32(rsrc, hb + (off & ~3u), 0, 0);
        const unsigned b = __builtin_amdgcn_raw_buffer_load_b32(rsrc, hb + ((off + 2u) & ~3u), 0, 0);
        s.lo = __builtin_amdgcn_alignbyte(b, a, off & 3u) & 0xFFFFFFu;
        return s;
    } else {
        return va_load_slice<KV>(rsrc, row_off, kv_h, n_kv_heads, 128 * which + 8 * rl);
    }
}

// kq_i = cnorm * C[idx_i] for the fwht K slice (8 dims in mq order), through
// the LDS pair table (fwht4: byte -> {C4[b & 15], C4[b >> 4]}; fwht3: 6 bits
// -> {C3[e & 7], C3[e >> 3]}); Q8 / fp8: va_dequant.
template <int KV>
__device__ __forceinline__ void va_dequant_k(const VaSlice& s, const float2* lut, float (&out)[8]) {
    if (KV == VA_KV_FWHT4) {
        const float cn = __builtin_bit_cast(float, s.cn);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 e = lut[(s.lo >> (8 * i)) & 0xFFu];
            out[2 * i] = cn * e.x;
            out[2 * i + 1] = cn * e.y;
        }
    } else if (KV == VA_KV_FWHT3) {
        const float cn = __builtin_bit_cast(float, s.cn);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 e = lut[(s.lo >> (6 * i)) & 63u];
            out[2 * i] = cn * e.x;
            out[2 * i + 1] = cn * e.y;
        }
    } else {
        va_dequant<KV>(s, out);
    }
}

// The reference lane's score partial (see the header for the chains).
template <int KV>
__device__ __forceinline__ float va_leaf_k(const float (&q)[8], const float (&k)[8]) {
    if (KV == VA_KV_FWHT4) {
        const float h0 = __builtin_fmaf(
            q[3], k[3], __builtin_fmaf(q[2], k[2], __builtin_fmaf(q[0], k[0], q[1] * k[1])));
        const float h1 = __builtin_fmaf(
            q[7], k[7], __builtin_fmaf(q[6], k[6], __builtin_fmaf(q[4], k[4], q[5] * k[5])));
        float p = 0.0f;
        p = p + h0;
        p = p + h1;
        return p;
    } else if (KV == VA_KV_FWHT3) {
        float p = __builtin_fmaf(q[0], k[0], q[1] * k[1]);
#pragma unroll
        for (int i = 2; i < 8; ++i) p = __builtin_fmaf(q[i], k[i], p);
        return p;
    } else {
        return va_leaf(q, k);
    }
}

// V slice of lane `lane` of one key. Q8 / fp8 / fwht3: dims 8l..8l+7. fwht4
// (Q8_0 V, the reference lane's two dim groups 4l..4l+3 and 128+4l..): the 4
// codes and block scale of each group (blocks l/8 and 4 + l/8 of the head).
template <int KV>
__device__ __forceinline__ VaSlice va_load_v(__amdgpu_buffer_rsrc_t rsrc, unsigned row_off, int kv_h,
                                             int n_kv_heads, int lane) {
    if (KV == VA_KV_FWHT4) {
        VaSlice s = {};
        const unsigned b0 = row_off + (unsigned)(kv_h * (VA_HD / 32) + (lane >> 3)) * 34u;
        const unsigned b1 = b0 + 4u * 34u;
        const unsigned cj = 2u + 4u * (unsigned)(lane & 7);
        s.lo = __builtin_amdgcn_raw_buffer_load_b32(rsrc, b0 + cj, 0, 0);
        s.hi = __builtin_amdgcn_raw_buffer_load_b32(rsrc, b1 + cj, 0, 0);
        s.sc = __builtin_amdgcn_raw_buffer_load_b16(rsrc, b0, 0, 0);
        s.sc2 = __builtin_amdgcn_raw_buffer_load_b16(rsrc, b1, 0, 0);
        return s;
    } else {
        return va_load_slice<(KV == VA_KV_FWHT3) ? VA_KV_Q8 : KV>(rsrc, row_off, kv_h, n_kv_heads, 8 * lane);
    }
}

template <int KV>
__device__ __forceinline__ void va_dequant_v(const VaSlice& s, float (&out)[8]) {
    if (KV == VA_KV_FWHT4) {
        const float s0 = va_f16(s.sc);
        const float s1 = va_f16(s.sc2);
#pragma unroll
        for (int i = 0; i < 4; ++i) out[i] = s0 * (float)(signed char)(s.lo >> (8 * i));
#pragma unroll
        for (int i = 0; i < 4; ++i) out[4 + i] = s1 * (float)(signed char)(s.hi >> (8 * i));
    } else {
        va_dequant<(KV == VA_KV_FWHT3) ? VA_KV_Q8 : KV>(s, out);
    }
}

// Every tile of one wave's workgroup range, for the wave's first NP pairs
// (the valid pairs of a wave are always a prefix j < NP, so the pair loops
// are straight-line and interleave across pairs).
template <int KV, int NP>
__device__ __forceinline__ void va_tiles(
    const unsigned char* __restrict__ k_cache, const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials, float (&sc)[VA_PPW][VA_TILE], const float (&qa)[VA_PPW][8],
    const float (&qb)[VA_PPW][8], const int (&pair_len)[VA_PPW], const size_t (&pair_base)[VA_PPW],
    int group_len, int n_tiles, int kv_h, int n_kv_heads, unsigned k_pp, unsigned v_pp, float scale_attn,
    const float2* lut) {
    const int lane = (int)threadIdx.x & 31;
    const int half = lane >> 4;
    const int rl = lane & 15;

    for (int tile = blockIdx.y; tile < n_tiles; tile += gridDim.y) {
        const int tile_start = tile * VA_TILE;
        const int tl_max = min(VA_TILE, group_len - tile_start);
        int tl[NP];
        bool full = true;
#pragma unroll
        for (int j = 0; j < NP; ++j) {
            tl[j] = min(VA_TILE, max(0, pair_len[j] - tile_start));
            full = full && (tl[j] == 0 || tl[j] == tl_max);
        }

        const auto k_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)(k_cache + (size_t)tile_start * k_pp), 0, (int)((unsigned)tl_max * k_pp), VA_RSRC_FLAGS);
        const auto v_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)(v_cache + (size_t)tile_start * v_pp), 0, (int)((unsigned)tl_max * v_pp), VA_RSRC_FLAGS);

        // ── Phase A: scores. Pass k: row `half` scores key 32(k>>4) + (k&15) + 16*half.
        {
            auto key_of = [&](int k) { return 32 * (k >> 4) + (k & 15) + 16 * half; };
            unsigned off = (unsigned)min(key_of(0), tl_max - 1) * k_pp;
            VaSlice ca = va_load_k<KV>(k_rsrc, off, kv_h, n_kv_heads, rl, 0);
            VaSlice cb = va_load_k<KV>(k_rsrc, off, kv_h, n_kv_heads, rl, 1);
            for (int k = 0; 32 * (k >> 4) + (k & 15) < tl_max; ++k) {
                const int t = key_of(k);
                const unsigned noff = (unsigned)min(key_of(k + 1), tl_max - 1) * k_pp;
                const VaSlice na = va_load_k<KV>(k_rsrc, noff, kv_h, n_kv_heads, rl, 0);
                const VaSlice nb = va_load_k<KV>(k_rsrc, noff, kv_h, n_kv_heads, rl, 1);
                float ka[8], kb[8];
                va_dequant_k<KV>(ca, lut, ka);
                va_dequant_k<KV>(cb, lut, kb);
                float v[NP];
#pragma unroll
                for (int j = 0; j < NP; ++j) v[j] = va_leaf_k<KV>(qa[j], ka) + va_leaf_k<KV>(qb[j], kb);
#pragma unroll
                for (int j = 0; j < NP; ++j) v[j] = va_row_xmask_add<0x168>(v[j]);
#pragma unroll
                for (int j = 0; j < NP; ++j) v[j] = va_row_xmask_add<0x164>(v[j]);
#pragma unroll
                for (int j = 0; j < NP; ++j) v[j] = va_row_xmask_add<0x162>(v[j]);
#pragma unroll
                for (int j = 0; j < NP; ++j) v[j] = va_row_xmask_add<0x161>(v[j]);
                if (rl == 0 && t < tl_max) {
#pragma unroll
                    for (int j = 0; j < NP; ++j) sc[j][t] = v[j] * scale_attn;
                }
                ca = na;
                cb = nb;
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();

        // ── Phases B/C: the reference's lane-strided softmax, one pair at a time.
#pragma unroll
        for (int j = 0; j < NP; ++j) {
            const int tile_len = tl[j];
            if (tile_len == 0) continue;
            float* scores = sc[j];
            float local_max = -1e30f;
            for (int i = lane; i < tile_len; i += 32)
                local_max = fmaxf(local_max, scores[i]);
            for (int off = 16; off > 0; off >>= 1)
                local_max = fmaxf(local_max, __shfl_xor(local_max, off));
            const float tile_max = local_max;
            float local_sum = 0.0f;
            for (int i = lane; i < tile_len; i += 32) {
                float e = expf(scores[i] - tile_max);
                scores[i] = e;
                local_sum += e;
            }
            for (int off = 16; off > 0; off >>= 1)
                local_sum += __shfl_xor(local_sum, off);
            if (lane == 0) {
                float* p = partials + (pair_base[j] + tile) * VA_STRIDE;
                p[0] = tile_max;
                p[1] = local_sum;
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();

        // ── Phase D: PV. Lane l owns dims 8l..8l+7 of every pair. The full
        //    path runs every pair (a pair with no key in this tile computes
        //    unused sums); the ragged path skips keys past each pair's bound.
        float acc[NP][8];
#pragma unroll
        for (int j = 0; j < NP; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[j][i] = 0.0f;
        {
            VaSlice cv = va_load_v<KV>(v_rsrc, 0u, kv_h, n_kv_heads, lane);
            if (full) {
                // Four keys per step: one 16-byte LDS broadcast of each
                // pair's weights, the next step's four V slices in flight.
                const int t4 = tl_max & ~3;
                VaSlice c4[4];
#pragma unroll
                for (int u = 0; u < 4; ++u)
                    c4[u] = va_load_v<KV>(v_rsrc, (unsigned)min(u, tl_max - 1) * v_pp, kv_h, n_kv_heads, lane);
                int t = 0;
                for (; t < t4; t += 4) {
                    VaSlice n4[4];
#pragma unroll
                    for (int u = 0; u < 4; ++u)
                        n4[u] = va_load_v<KV>(v_rsrc, (unsigned)min(t + 4 + u, tl_max - 1) * v_pp, kv_h,
                                              n_kv_heads, lane);
                    float4 w4[NP];
#pragma unroll
                    for (int j = 0; j < NP; ++j) w4[j] = *(const float4*)&sc[j][t];
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        float vq[8];
                        va_dequant_v<KV>(c4[u], vq);
#pragma unroll
                        for (int j = 0; j < NP; ++j) {
                            const float w = w4[j][u];
#pragma unroll
                            for (int i = 0; i < 8; ++i) acc[j][i] = __builtin_fmaf(w, vq[i], acc[j][i]);
                        }
                    }
#pragma unroll
                    for (int u = 0; u < 4; ++u) c4[u] = n4[u];
                }
                for (; t < tl_max; ++t) {
                    float vq[8];
                    va_dequant_v<KV>(c4[0], vq);
#pragma unroll
                    for (int j = 0; j < NP; ++j) {
                        const float w = sc[j][t];
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc[j][i] = __builtin_fmaf(w, vq[i], acc[j][i]);
                    }
                    c4[0] = c4[1];
                    c4[1] = c4[2];
                    c4[2] = c4[3];
                }
            } else {
                for (int t = 0; t < tl_max; ++t) {
                    const VaSlice nv = va_load_v<KV>(
                        v_rsrc, (unsigned)min(t + 1, tl_max - 1) * v_pp, kv_h, n_kv_heads, lane);
                    float vq[8];
                    va_dequant_v<KV>(cv, vq);
#pragma unroll
                    for (int j = 0; j < NP; ++j) {
                        if (t >= tl[j]) continue;
                        const float w = sc[j][t];
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc[j][i] = __builtin_fmaf(w, vq[i], acc[j][i]);
                    }
                    cv = nv;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NP; ++j) {
            if (tl[j] == 0) continue;
            if (KV == VA_KV_FWHT4) {
                // Reference lane l: dims 4l..4l+3 then 128+4l..128+4l+3.
                float* p = partials + (pair_base[j] + tile) * VA_STRIDE + 2 + 4 * lane;
                *(float2*)(p) = make_float2(acc[j][0], acc[j][1]);
                *(float2*)(p + 2) = make_float2(acc[j][2], acc[j][3]);
                *(float2*)(p + 128) = make_float2(acc[j][4], acc[j][5]);
                *(float2*)(p + 130) = make_float2(acc[j][6], acc[j][7]);
            } else {
                float* p = partials + (pair_base[j] + tile) * VA_STRIDE + 2 + 8 * lane;
#pragma unroll
                for (int i = 0; i < 8; i += 2) *(float2*)(p + i) = make_float2(acc[j][i], acc[j][i + 1]);
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();
    }
}

// GROUP q heads x ROWS query rows per workgroup (6 x 8 or 8 x 6): both give
// VA_PPW = 6 (row, head) pairs per wave. The defaults are the group-6 shape,
// which the lane kernel (attention_verify_lanes.gfx1201.hip) instantiates.
template <int KV, int GROUP = VA_GROUP, int ROWS = VA_ROWS>
__device__ __forceinline__ void va_verify_gqa(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ positions, int n_rows, int n_kv_heads, int max_tiles,
    int batch_offset, float scale_attn, const float* __restrict__ signs1 = nullptr,
    const float* __restrict__ signs2 = nullptr) {
    __shared__ __attribute__((aligned(16))) float sc_lds[VA_WAVES][VA_PPW][VA_TILE];
    // fwht K index-pair table (fwht4: 256 byte entries; fwht3: 64 six-bit entries).
    constexpr int lut_n = (KV == VA_KV_FWHT4) ? 256 : (KV == VA_KV_FWHT3) ? 64 : 1;
    __shared__ float2 lut_lds[lut_n];

    const int n_heads = n_kv_heads * GROUP;
    const int n_rg = (n_rows + ROWS - 1) / ROWS;
    const int kv_h = (int)blockIdx.x / n_rg;
    const int rg = (int)blockIdx.x - kv_h * n_rg;
    if (kv_h >= n_kv_heads) return;
    if (KV == VA_KV_FWHT4 || KV == VA_KV_FWHT3) {
        const int t = (int)threadIdx.x;
        if (KV == VA_KV_FWHT4)
            lut_lds[t & 255] = make_float2(VA_C4[t & 15], VA_C4[(t >> 4) & 15]);
        else if (t < 64)
            lut_lds[t & 63] = make_float2(VA_C3_256[t & 7], VA_C3_256[(t >> 3) & 7]);
        __syncthreads();
    }
    const int row0 = rg * ROWS;
    const int rows_here = min(ROWS, n_rows - row0);

    const int wave = __builtin_amdgcn_readfirstlane((int)threadIdx.x >> 5);

    // Per-key record strides: V is Q8_0 for Q8 / fwht, K is the head-packed fwht record.
    const unsigned v_pp = (KV == VA_KV_FP8) ? (unsigned)n_kv_heads * (VA_HD + 2)
                                            : (unsigned)n_kv_heads * (VA_HD / 32) * 34u;
    const unsigned k_pp = (KV == VA_KV_FWHT4) ? (unsigned)n_kv_heads * 132u
                          : (KV == VA_KV_FWHT3) ? (unsigned)n_kv_heads * 100u
                                                : v_pp;

    // Live key range of this row group, from the device positions.
    int group_len = 0;
    for (int r = 0; r < rows_here; ++r) group_len = max(group_len, positions[batch_offset + row0 + r] + 1);
    // The reference grid covers tiles < max_tiles only; never write past it.
    const int n_tiles = min((group_len + VA_TILE - 1) / VA_TILE, max_tiles);

    // This wave's pairs: p = wave + 8j -> (row p/GROUP, head p%GROUP); valid
    // pairs (row < rows_here) are the prefix j < np.
    int np = 0;
    int pair_len[VA_PPW];
    size_t pair_base[VA_PPW];
    float qa[VA_PPW][8], qb[VA_PPW][8];
#pragma unroll
    for (int j = 0; j < VA_PPW; ++j) {
        const int p = wave + VA_WAVES * j;
        const int r = p / GROUP;
        const int head = kv_h * GROUP + (p - r * GROUP);
        pair_base[j] = ((size_t)(row0 + r) * n_heads + head) * max_tiles;
        pair_len[j] = (r < rows_here) ? positions[batch_offset + row0 + r] + 1 : 0;
        if (r < rows_here) {
            np = j + 1;
            const float* qr = q + ((size_t)(row0 + r) * n_heads + head) * VA_HD;
            va_load_q<KV>(qr, signs1, signs2, qa[j], qb[j]);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) { qa[j][i] = 0.0f; qb[j][i] = 0.0f; }
        }
    }

#define VA_TILES(N)                                                                                     \
    case N:                                                                                             \
        va_tiles<KV, N>(k_cache, v_cache, partials, sc_lds[wave], qa, qb, pair_len, pair_base, group_len, \
                        n_tiles, kv_h, n_kv_heads, k_pp, v_pp, scale_attn, lut_lds);                    \
        break;
    switch (np) {
        VA_TILES(1)
        VA_TILES(2)
        VA_TILES(3)
        VA_TILES(4)
        VA_TILES(5)
        VA_TILES(6)
        default:
            break;
    }
#undef VA_TILES
}

// Grid: [ceil(n_rows/8) * n_kv_heads, S], block 256. Workgroup x = kv_h *
// ceil(n_rows/8) + row group; workgroup y serves tiles y, y + S, ... below
// ceil((max row position + 1) / 128) of its row group. Partials rows are
// chunk-local; positions are read at batch_offset + row.
extern "C" __launch_bounds__(VA_THREADS, 1)
__global__ void attention_verify_gqa_q8_gfx1201(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ positions, int n_rows, int n_kv_heads, int max_tiles,
    int batch_offset, float scale_attn) {
    va_verify_gqa<VA_KV_Q8>(q, k_cache, v_cache, partials, positions, n_rows, n_kv_heads, max_tiles,
                            batch_offset, scale_attn);
}

extern "C" __launch_bounds__(VA_THREADS, 1)
__global__ void attention_verify_gqa_fp8_gfx1201(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ positions, int n_rows, int n_kv_heads, int max_tiles,
    int batch_offset, float scale_attn) {
    va_verify_gqa<VA_KV_FP8>(q, k_cache, v_cache, partials, positions, n_rows, n_kv_heads, max_tiles,
                             batch_offset, scale_attn);
}

// Group-8 fp8 twin (n_heads / n_kv_heads == 8: H16/KV2 A3B). Same 10-argument
// ABI and `[row][head][max_tiles][2 + 256]` partials as the entries above;
// one workgroup serves VA_ROWS8 = 6 rows x 8 heads = 48 (row, head) pairs, so
// every wave still owns exactly VA_PPW = 6 pairs (wave w = head w of rows
// 0..5) and the register / LDS footprint equals the group-6 kernel's. Grid:
// [ceil(n_rows/6) * n_kv_heads, S], block 256; the reduce below is group
// agnostic (n_heads is an argument). The reduce is shared, not copied.
// The lane module (lane_fa.rs concatenates this file with the lane entries)
// defines VA_LANE_MODULE so it keeps exporting exactly the entries it exported
// before the group-8 twin existed.
#ifndef VA_LANE_MODULE
extern "C" __launch_bounds__(VA_THREADS, 1)
__global__ void attention_verify_gqa_fp8_gqa8_gfx1201(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ positions, int n_rows, int n_kv_heads, int max_tiles,
    int batch_offset, float scale_attn) {
    va_verify_gqa<VA_KV_FP8, VA_GROUP8, VA_ROWS8>(q, k_cache, v_cache, partials, positions, n_rows,
                                                  n_kv_heads, max_tiles, batch_offset, scale_attn);
}

// fwht K twins (Q8_0 V): the 10-argument ABI above + the K rotation's sign
// tables (128 f32 for fwht4, 256 for fwht3). Same grid / block / split.
extern "C" __launch_bounds__(VA_THREADS, 1)
__global__ void attention_verify_gqa_fwht4_gfx1201(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ positions, int n_rows, int n_kv_heads, int max_tiles,
    int batch_offset, float scale_attn, const float* __restrict__ signs1,
    const float* __restrict__ signs2) {
    va_verify_gqa<VA_KV_FWHT4>(q, k_cache, v_cache, partials, positions, n_rows, n_kv_heads, max_tiles,
                               batch_offset, scale_attn, signs1, signs2);
}

extern "C" __launch_bounds__(VA_THREADS, 1)
__global__ void attention_verify_gqa_fwht3_gfx1201(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ positions, int n_rows, int n_kv_heads, int max_tiles,
    int batch_offset, float scale_attn, const float* __restrict__ signs1,
    const float* __restrict__ signs2) {
    va_verify_gqa<VA_KV_FWHT3>(q, k_cache, v_cache, partials, positions, n_rows, n_kv_heads, max_tiles,
                               batch_offset, scale_attn, signs1, signs2);
}
#endif

// Byte-identical parallel twin of attention_flash_asym_reduce_batched at
// head_dim 256: one thread per dim, each running the reference's serial tile
// order (max pass, then expf-corrected fma accumulation), then g * (1/sum).
// Grid: [n_heads, n_rows], block 256.
extern "C" __launch_bounds__(VA_HD, 1)
__global__ void attention_verify_reduce_gfx1201(
    const float* __restrict__ partials, float* __restrict__ out,
    const int* __restrict__ positions, int n_heads, int max_tiles, int batch_offset) {
    const int h = blockIdx.x;
    const int local_bid = blockIdx.y;
    const int d = threadIdx.x;
    const int seq_len = positions[batch_offset + local_bid] + 1;
    const int n_tiles = (seq_len + VA_TILE - 1) / VA_TILE;
    const float* my_partials = partials + ((long long)local_bid * n_heads + h) * max_tiles * VA_STRIDE;

    float global_max = -1e30f;
    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = my_partials + tile * VA_STRIDE;
        if (p[1] > 0.0f)
            global_max = fmaxf(global_max, p[0]);
    }

    // Same serial per-dim order as the reference; the loads of 8 tiles are
    // issued ahead of their (in-order) accumulation.
    float global_sum = 0.0f;
    float g = 0.0f;
    int tile = 0;
    for (; tile + 8 <= n_tiles; tile += 8) {
        float pm[8], ps[8], pv[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const float* p = my_partials + (tile + u) * VA_STRIDE;
            pm[u] = p[0];
            ps[u] = p[1];
            pv[u] = p[2 + d];
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            if (ps[u] <= 0.0f) continue;
            const float corr = expf(pm[u] - global_max);
            g += pv[u] * corr;
            global_sum += ps[u] * corr;
        }
    }
    for (; tile < n_tiles; tile++) {
        const float* p = my_partials + tile * VA_STRIDE;
        const float tile_max = p[0];
        const float tile_sum = p[1];
        if (tile_sum <= 0.0f) continue;
        const float corr = expf(tile_max - global_max);
        g += p[2 + d] * corr;
        global_sum += tile_sum * corr;
    }

    const float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    out[(size_t)local_bid * n_heads * VA_HD + (size_t)h * VA_HD + d] = g * inv_sum;
}
