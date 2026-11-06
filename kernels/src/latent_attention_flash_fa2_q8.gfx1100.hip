// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// GEN-FA2 F2a-Q8 flash prefill, q128w8-kt64 body for gfx1100 (symbol `latent_attention_flash_prefill_fa2_q8_q128w8`).
// The gfx11 twin of kernels/src/latent_attention_flash_fa2_q8.hip (gfx12): same kernarg ABI (21 kernargs, 108 bytes),
// block 256, grid = ceil(n_rows / 128) x n_heads, same schedule (CTA = one head x 128 query rows, 8 waves of 16 rows;
// one 64-key K stage (dequantized once per CTA at the LDS store) and one V^T stage per tile; the global -> VGPR prefetch
// of the next tile carries the raw codes and F16 scale bits and is widened at the store one tile later).
//
// What differs from the gfx12 body is only the gfx11 wave32 WMMA fragment map, taken statement for statement from the
// incumbent's `LAF_LAYOUT_G11` path (kernels/src/latent_attention_flash.hip):
//   * A/B operands hold all 16 k per lane (lanes 16..31 replicate 0..15); C[j] = C[2 * j + (lane >> 4)][lane & 15];
//   * so the score mask key of accumulator j is `ks * 16 + 2 * j + hh`, the P^T operand is built with one xor-16
//     exchange per word pair, and the epilogue swaps accumulator halves so each lane stores 8 consecutive columns.
// The Q^T operands of the rope tail (dims 128..191, 4 fragments) are staged in LDS instead of VGPRs: a gfx11 operand is
// twice the gfx12 one, and 12 resident Q fragments (96 VGPRs) do not fit next to the O^T accumulator, the score chain
// and the prefetch. The nope fragments (dims 0..127, 8 fragments) stay resident.
//
// Numeric recipe = the incumbent's gfx11 `latent_attention_flash_prefill_q8`, statement for statement: Q rounded once to
// BF16, K/V dequantized as BF16 RNE of the F32 product `code * scale`, WMMA chains over ascending dims (QK) and keys
// (PV) in 64-key tiles, F32 scores * scale, tile max, p = exp2((s - m') * log2e) rounded to BF16 before both the PV
// product and the denominator, denominator summed from the rounded values in the fixed (sub-tile, j) order, per-lane
// partial l combined with one xor-16 exchange, `round_y`, the `*pos + pos_offset + i` / `max_rows` mask, empty rows ->
// zeros, shared tail head stride 0. The output is meant to be bit-identical to the incumbent on gfx1100.
//
// Requirements (the contract domain enforces them; a violating launch leaves y untouched): wa 128, wb 64, dv 128,
// blockDim.x == 256, dynamic LDS == 62464, every offset/stride a multiple of 8 elements, ka_tok / v_tok multiples of
// 16 bytes, 16-byte aligned base pointers.

#define F2A_NW 8
#define F2A_KT 64

#include <hip/hip_runtime.h>
#include <stdint.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1100__)
#error "this F2a-Q8 body is gfx1100 only"
#endif

#define F2A_WA 128
#define F2A_WB 64
#define F2A_DV 128
#define F2A_DQ (F2A_WA + F2A_WB)
#define F2A_NT (F2A_NW * 32)           // threads per CTA
#define F2A_QT (16 * F2A_NW)           // query rows per CTA
#define F2A_NKS (F2A_KT / 16)          // 16-key sub-tiles per key tile
#define F2A_KP 200                     // 192 + 8
#define F2A_VPW 36
#define F2A_KRW (F2A_NT / 8)           // K rows filled per pass (8 threads x 16 B = one 128-byte half row)
#define F2A_KR (F2A_KT / F2A_KRW)      // fill passes per thread
#define F2A_KBYTES (F2A_KT * F2A_KP * 2)
#define F2A_VBYTES (F2A_DV * F2A_VPW * 4)
#define F2A_VOFF F2A_KBYTES
#define F2A_QSTEPS (F2A_WA / 16)       // resident Q fragments (nope dims)
#define F2A_TSTEPS (F2A_WB / 16)       // LDS-staged Q fragments (rope tail dims)
#define F2A_TP (F2A_WB + 8)            // LDS pitch of a staged Q tail row (elements)
#define F2A_TOFF (F2A_KBYTES + F2A_VBYTES)
#define F2A_TBYTES (F2A_QT * F2A_TP * 2)
#define F2A_LDS (F2A_KBYTES + F2A_VBYTES + F2A_TBYTES)
#define F2A_VGS F2A_NW                 // V column groups (8 columns) covered per pass
#define F2A_VG (16 / F2A_VGS)          // passes per lane over the 16 column groups
static_assert(F2A_KR >= 1 && F2A_KT % F2A_KRW == 0, "K fill geometry");
static_assert(F2A_VG >= 1 && 16 % F2A_VGS == 0, "V fill geometry");
static_assert(F2A_LDS == 62464, "dynamic LDS of the gfx1100 body");

static __device__ __forceinline__ float f2a_exp(float x) {
    return __builtin_amdgcn_exp2f(x * 1.4426950408889634f);
}

typedef short __attribute__((ext_vector_type(16))) f2a_op_t;   // 16 k per lane
typedef unsigned int __attribute__((ext_vector_type(8))) f2a_u8_t;
typedef float __attribute__((ext_vector_type(8))) f2a_f8_t;

static __device__ __forceinline__ uint32_t f2a_f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (u >> 16) | 0x40u;
    u += 0x7fffu + ((u >> 16) & 1u);
    return u >> 16;
}

// Round-to-nearest-even of a finite value, result left in the high 16 bits.
static __device__ __forceinline__ uint32_t f2a_rne_hi(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
}

// Signed byte `b` (0..3) of word `w` as float.
static __device__ __forceinline__ float f2a_i8(uint32_t w, int b) {
    return (float)((int)(w << (24 - 8 * b)) >> 24);
}

// F16 scale (raw bits, zero-extended) widened to F32.
static __device__ __forceinline__ float f2a_h2f(uint32_t bits) {
    return (float)__builtin_bit_cast(_Float16, (uint16_t)bits);
}

// Eight int8 codes (two words) times `s`, rounded to BF16 (RNE), packed as 8 BF16 (element e in half e & 1 of word e >> 1).
static __device__ __forceinline__ uint4 f2a_q8_to_bf16x8(uint2 w, float s) {
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t ww = (i < 2) ? w.x : w.y;
        const int b = (i & 1) * 2;
        o[i] = (f2a_rne_hi(f2a_i8(ww, b) * s) >> 16) | f2a_rne_hi(f2a_i8(ww, b + 1) * s);
    }
    return make_uint4(o[0], o[1], o[2], o[3]);
}

static __device__ __forceinline__ f2a_f8_t f2a_wmma(f2a_op_t a, f2a_op_t b, f2a_f8_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

// 16 BF16 (two 16-byte LDS reads) as one gfx11 operand. p: 16-byte aligned.
static __device__ __forceinline__ f2a_op_t f2a_ldop(const uint16_t* p) {
    const uint4 u0 = *(const uint4*)p;
    const uint4 u1 = *(const uint4*)(p + 8);
    const f2a_u8_t w = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
    return __builtin_bit_cast(f2a_op_t, w);
}

// LDS-only workgroup barrier: the "local" fences leave the global prefetches in flight. Each wave waits for its own
// LDS operations (lgkmcnt(0), vmcnt/expcnt left open: gfx11 s_waitcnt 0xFC07) before it signals; in WGP mode a
// workgroup's waves span both CUs, whose LDS queues are not ordered against each other (see
// attention_verify_wmma.gfx1151.hip).
#define F2A_WAIT_LGKM0 0xFC07
#define F2A_BAR()                                                         \
    do {                                                                  \
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");   \
        __builtin_amdgcn_s_waitcnt(F2A_WAIT_LGKM0);                       \
        __builtin_amdgcn_s_barrier();                                     \
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");   \
    } while (0)

// ---- global -> VGPR prefetch of one key tile (as the gfx12 body) -------------------------------------------------
static __device__ __forceinline__ void f2a_load_k(
    uint4 (&kr)[F2A_KR][3], const uint8_t* __restrict__ ka, const uint16_t* __restrict__ kb,
    int kb0, int kend, int h, int n_heads, int ka_tok, int ka_head, int kb_tok, int kb_head, int krow0, int part
) {
#pragma unroll
    for (int r = 0; r < F2A_KR; ++r) {
        const int row = kb0 + krow0 + r * F2A_KRW;
        const int rr = row < kend ? row : 0;
        const uint8_t* rowp = ka + (size_t)rr * ka_tok;
        const uint8_t* pa = rowp + (size_t)h * ka_head + part * 8;
        const uint16_t* pb = kb + (size_t)rr * kb_tok + (size_t)h * kb_head + part * 8;
        const uint2 lo = *(const uint2*)pa;
        const uint2 hi = *(const uint2*)(pa + 64);
        const uint32_t sc = *(const unsigned short*)(rowp + (size_t)n_heads * F2A_WA + 2 * (size_t)h);
        kr[r][0] = make_uint4(lo.x, lo.y, hi.x, hi.y);
        kr[r][1] = make_uint4(sc, 0u, 0u, 0u);
        kr[r][2] = *(const uint4*)pb;
    }
}

static __device__ __forceinline__ void f2a_fix_k(uint4 (&kr)[F2A_KR][3], int kbase, int kend, int krow0) {
    if (kbase + F2A_KT > kend) {
#pragma unroll
        for (int r = 0; r < F2A_KR; ++r) {
            if (!(kbase + krow0 + r * F2A_KRW < kend)) {
                kr[r][0] = make_uint4(0u, 0u, 0u, 0u);
                kr[r][1] = make_uint4(0u, 0u, 0u, 0u);
                kr[r][2] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    }
}

static __device__ __forceinline__ void f2a_store_k(
    uint16_t* ksb, const uint4 (&kr)[F2A_KR][3], int krow0, int part
) {
#pragma unroll
    for (int r = 0; r < F2A_KR; ++r) {
        const float sc = f2a_h2f(kr[r][1].x);
        const int row = krow0 + r * F2A_KRW;
        uint16_t* d = ksb + (size_t)row * F2A_KP + part * 8;
        *(uint4*)d = f2a_q8_to_bf16x8(make_uint2(kr[r][0].x, kr[r][0].y), sc);
        *(uint4*)(d + 64) = f2a_q8_to_bf16x8(make_uint2(kr[r][0].z, kr[r][0].w), sc);
        *(uint4*)(d + 128) = kr[r][2];
    }
}

static __device__ __forceinline__ void f2a_load_v(
    uint4 (&va)[F2A_VG], uint4 (&vb)[F2A_VG], const uint8_t* __restrict__ vv, int kb0, int kend, int h, int n_heads,
    int v_tok, int v_head, int vklo, int vg0
) {
    const uint8_t* ra = vv + (size_t)(kb0 + vklo < kend ? kb0 + vklo : 0) * v_tok;
    const uint8_t* rb = vv + (size_t)(kb0 + vklo + 2 < kend ? kb0 + vklo + 2 : 0) * v_tok;
    const uint8_t* vpa = ra + (size_t)h * v_head;
    const uint8_t* vpb = rb + (size_t)h * v_head;
    const uint32_t sa = *(const unsigned short*)(ra + (size_t)n_heads * F2A_DV + 2 * (size_t)h);
    const uint32_t sb = *(const unsigned short*)(rb + (size_t)n_heads * F2A_DV + 2 * (size_t)h);
#pragma unroll
    for (int it = 0; it < F2A_VG; ++it) {
        const int g = vg0 + it * F2A_VGS;
        const uint2 ca = *(const uint2*)(vpa + g * 8);
        const uint2 cb = *(const uint2*)(vpb + g * 8);
        va[it] = make_uint4(ca.x, ca.y, sa, 0u);
        vb[it] = make_uint4(cb.x, cb.y, sb, 0u);
    }
}

static __device__ __forceinline__ void f2a_fix_v(uint4 (&va)[F2A_VG], uint4 (&vb)[F2A_VG], int kbase, int kend, int vklo) {
    if (kbase + F2A_KT > kend) {
        const bool oka = kbase + vklo < kend, okb = kbase + vklo + 2 < kend;
#pragma unroll
        for (int it = 0; it < F2A_VG; ++it) {
            if (!oka) va[it] = make_uint4(0u, 0u, 0u, 0u);
            if (!okb) vb[it] = make_uint4(0u, 0u, 0u, 0u);
        }
    }
}

// One V^T task (Q8): word p of rows g*8 .. g*8+7 of the [128][F2A_VPW] table packs keys (low | high << 16).
static __device__ __forceinline__ void f2a_v_store(uint32_t* vs, int g, int p, uint4 ar, uint4 br) {
    const uint4 a = f2a_q8_to_bf16x8(make_uint2(ar.x, ar.y), f2a_h2f(ar.z));
    const uint4 b = f2a_q8_to_bf16x8(make_uint2(br.x, br.y), f2a_h2f(br.z));
    const uint32_t aw[4] = {a.x, a.y, a.z, a.w};
    const uint32_t bw[4] = {b.x, b.y, b.z, b.w};
    uint32_t* col = vs + (g * 8) * F2A_VPW;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        col[(2 * i) * F2A_VPW + p] = (aw[i] & 0xffffu) | (bw[i] << 16);
        col[(2 * i + 1) * F2A_VPW + p] = (aw[i] >> 16) | (bw[i] & 0xffff0000u);
    }
}

static __device__ __forceinline__ void f2a_store_v(
    uint32_t* vs, const uint4 (&va)[F2A_VG], const uint4 (&vb)[F2A_VG], int vg0, int vp
) {
#pragma unroll
    for (int it = 0; it < F2A_VG; ++it) f2a_v_store(vs, vg0 + it * F2A_VGS, vp, va[it], vb[it]);
}

// ---- S^T = K . Q^T for one key tile from the LDS stage ----------------------------------------------------------
// Contraction step s covers dims [16 s, 16 s + 16): steps 0..7 against the resident nope fragments, 8..11 against the
// tail fragments read from LDS (`qt` = this lane's staged tail row). Same chain order as the incumbent (ascending dims).
static __device__ __forceinline__ void f2a_ld_k(f2a_op_t (&fr)[F2A_NKS], const uint16_t* ksb, int s, int l16) {
#pragma unroll
    for (int ks = 0; ks < F2A_NKS; ++ks) fr[ks] = f2a_ldop(ksb + (size_t)(ks * 16 + l16) * F2A_KP + 16 * s);
}

static __device__ __forceinline__ void f2a_qk(
    f2a_f8_t (&acc)[F2A_NKS], const uint16_t* ksb, const f2a_op_t (&qf)[F2A_QSTEPS], const uint16_t* qt, int l16
) {
#pragma unroll
    for (int s = 0; s < F2A_QSTEPS + F2A_TSTEPS; ++s) {
        f2a_op_t fr[F2A_NKS];
        f2a_ld_k(fr, ksb, s, l16);
        const f2a_op_t qs = (s < F2A_QSTEPS) ? qf[s < F2A_QSTEPS ? s : 0] : f2a_ldop(qt + 16 * (s - F2A_QSTEPS));
#pragma unroll
        for (int ks = 0; ks < F2A_NKS; ++ks) acc[ks] = f2a_wmma(fr[ks], qs, acc[ks]);
    }
}

struct F2aState {
    f2a_f8_t o[8];
    float m;
    float l;
};

static __device__ __forceinline__ void f2a_softmax(
    F2aState& st, f2a_op_t (&pf)[F2A_NKS], const uint16_t* ksb, const f2a_op_t (&qf)[F2A_QSTEPS], const uint16_t* qt,
    int kb0, int l16, int hh, int lim_first, int lim_i, float scale
) {
    f2a_f8_t acc_s[F2A_NKS];
#pragma unroll
    for (int ks = 0; ks < F2A_NKS; ++ks) acc_s[ks] = f2a_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
    f2a_qk(acc_s, ksb, qf, qt, l16);

    const bool need_mask = kb0 + (F2A_KT - 1) > lim_first;
    float sv[F2A_NKS][8];
    float tmax = -INFINITY;
#pragma unroll
    for (int ks = 0; ks < F2A_NKS; ++ks) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float s = acc_s[ks][j] * scale;
            if (need_mask) {
                const int kj = kb0 + ks * 16 + 2 * j + hh;
                s = (kj <= lim_i) ? s : -INFINITY;
            }
            sv[ks][j] = s;
            tmax = fmaxf(tmax, s);
        }
    }
    tmax = fmaxf(tmax, __shfl_xor(tmax, 16, 32));
    const float m_new = fmaxf(st.m, tmax);
    const float mshift = (m_new == -INFINITY) ? 0.0f : m_new;
    const float corr = f2a_exp(st.m - mshift);  // m == -inf -> 0 (nothing accumulated yet)
    st.m = m_new;

    float lsum = 0.0f;
#pragma unroll
    for (int ks = 0; ks < F2A_NKS; ++ks) {
        uint32_t pw[8];  // rounded probabilities, bf16 in the high half
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            pw[j] = f2a_rne_hi(f2a_exp(sv[ks][j] - mshift));
            lsum += __uint_as_float(pw[j]);
        }
        // own word q packs acc[2q], acc[2q+1] = keys (4q + hh, 4q + 2 + hh).
        uint32_t own[4], oth[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) own[w] = (pw[2 * w] >> 16) | pw[2 * w + 1];
#pragma unroll
        for (int w = 0; w < 4; ++w) oth[w] = __shfl_xor((int)own[w], 16, 32);
        // W[2q + r]: r == hh -> own word q, else the partner's.
        f2a_u8_t W;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            W[2 * w] = hh ? oth[w] : own[w];
            W[2 * w + 1] = hh ? own[w] : oth[w];
        }
        pf[ks] = __builtin_bit_cast(f2a_op_t, W);
    }
    st.l = st.l * corr + lsum;
#pragma unroll
    for (int d = 0; d < 8; ++d) st.o[d] *= corr;
}

// ---- O^T += V^T . P^T -----------------------------------------------------------------------------------------------
static __device__ __forceinline__ void f2a_pv(F2aState& st, const f2a_op_t (&pf)[F2A_NKS], const uint32_t* vsb, int l16) {
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        const uint32_t* arow = vsb + (size_t)(d * 16 + l16) * F2A_VPW;
#pragma unroll
        for (int ks = 0; ks < F2A_NKS; ++ks) {
            const uint4 u0 = *(const uint4*)(arow + ks * 8);
            const uint4 u1 = *(const uint4*)(arow + ks * 8 + 4);
            const f2a_u8_t w = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
            st.o[d] = f2a_wmma(__builtin_bit_cast(f2a_op_t, w), pf[ks], st.o[d]);
        }
    }
}

extern "C" __global__ __launch_bounds__(F2A_NT) void latent_attention_flash_prefill_fa2_q8_q128w8(
    const float* __restrict__ q,         // [n_rows, H, dq]
    float* __restrict__ y,               // [n_rows, H, dv]
    const uint8_t* __restrict__ ka,      // INT8 K_nope plane view base (byte strides)
    const uint16_t* __restrict__ kb,
    const uint8_t* __restrict__ vv,      // INT8 V plane view base (byte strides)
    const int* __restrict__ pos_ptr,     // device scalar: rows cached before this call
    int n_rows, int n_heads, int wa, int wb, int dv, int pos_offset, int max_rows,
    int ka_tok, int ka_head, int kb_tok, int kb_head, int v_tok, int v_head,
    float scale, int round_y
) {
    extern __shared__ __attribute__((aligned(16))) unsigned char f2a_smem[];
    // Contract guard (workgroup-uniform, before any barrier).
    if (wa != F2A_WA || wb != F2A_WB || dv != F2A_DV || (int)blockDim.x != F2A_NT) return;

    const int ntiles = (n_rows + F2A_QT - 1) / F2A_QT;
    if ((int)blockIdx.x >= ntiles) return;  // grid.x is the contract's ceil(n_rows / 128)

    const int tid = threadIdx.x;
    const int wave = tid >> 5, lane = tid & 31, l16 = lane & 15, hh = lane >> 4;
    const int h = blockIdx.y;
    const int tile = ntiles - 1 - (int)blockIdx.x;  // heaviest tile first
    const int i0 = tile * F2A_QT;

    const int base = *pos_ptr + pos_offset;
    const int wrow0 = i0 + wave * 16;
    const int row_l = wrow0 + l16;
    const int lim_row = min(row_l, n_rows - 1);
    const int lim_i = min(base + lim_row, max_rows - 1);
    const int lim_first = min(base + min(wrow0, n_rows - 1), max_rows - 1);
    const int lim_last = min(base + min(wrow0 + 15, n_rows - 1), max_rows - 1);
    const int kend = max(0, min(base + min(i0 + F2A_QT - 1, n_rows - 1), max_rows - 1) + 1);

    uint16_t* ksb = (uint16_t*)f2a_smem;
    uint32_t* vsb = (uint32_t*)(f2a_smem + F2A_VOFF);
    uint16_t* qtb = (uint16_t*)(f2a_smem + F2A_TOFF);

    // ---- Q^T, BF16-rounded once. Lane (hh, l16) holds query row wrow0 + l16 (both halves the same row, gfx11
    // replication): nope dims 0..127 as 8 resident fragments; the tail dims 128..191 of the wave's 16 rows are staged in
    // LDS by the half hh (32 dims each) and read back as 4 operands per key tile. Rows past n_rows are zeros.
    f2a_op_t qf[F2A_QSTEPS];
    const uint16_t* qt = qtb + (size_t)(wave * 16 + l16) * F2A_TP;
    {
        const bool ok = row_l < n_rows;
        const float* src = q + ((size_t)(ok ? row_l : 0) * n_heads + h) * F2A_DQ;
#pragma unroll
        for (int s = 0; s < F2A_QSTEPS; ++s) {
            f2a_u8_t w = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
            if (ok) {
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const float4 a = *(const float4*)(src + 16 * s + 4 * c);
                    w[2 * c] = f2a_f32_to_bf16(a.x) | (f2a_f32_to_bf16(a.y) << 16);
                    w[2 * c + 1] = f2a_f32_to_bf16(a.z) | (f2a_f32_to_bf16(a.w) << 16);
                }
            }
            qf[s] = __builtin_bit_cast(f2a_op_t, w);
        }
        uint32_t* dst = (uint32_t*)(qtb + (size_t)(wave * 16 + l16) * F2A_TP + 32 * hh);
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            uint32_t w0 = 0u, w1 = 0u;
            if (ok) {
                const float4 a = *(const float4*)(src + F2A_WA + 32 * hh + 4 * c);
                w0 = f2a_f32_to_bf16(a.x) | (f2a_f32_to_bf16(a.y) << 16);
                w1 = f2a_f32_to_bf16(a.z) | (f2a_f32_to_bf16(a.w) << 16);
            }
            dst[2 * c] = w0;
            dst[2 * c + 1] = w1;
        }
    }

    F2aState st;
#pragma unroll
    for (int d = 0; d < 8; ++d) st.o[d] = f2a_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
    st.m = -INFINITY;
    st.l = 0.0f;

    // Fill geometry. K: thread = (row group krow0, half-row part). V: lane = key pair vp of the tile.
    const int krow0 = tid >> 3, part = tid & 7;
    const int vg0 = wave, vp = lane;
    const int vks = vp >> 3, vq = (vp >> 1) & 3, vr = vp & 1;
    const int vklo = vks * 16 + 4 * vq + vr;

    uint4 kr[F2A_KR][3];
    uint4 va[F2A_VG], vb[F2A_VG];

    if (kend > 0) {
        f2a_load_k(kr, ka, kb, 0, kend, h, n_heads, ka_tok, ka_head, kb_tok, kb_head, krow0, part);
        f2a_load_v(va, vb, vv, 0, kend, h, n_heads, v_tok, v_head, vklo, vg0);
    }
    for (int kb0 = 0; kb0 < kend; kb0 += F2A_KT) {
        // K stage is free: every wave finished its QK of the previous tile before the previous V barrier, and its PV
        // before this barrier. The first barrier also publishes the staged Q tail.
        f2a_fix_k(kr, kb0, kend, krow0);
        f2a_store_k(ksb, kr, krow0, part);
        f2a_load_k(kr, ka, kb, kb0 + F2A_KT, kend, h, n_heads, ka_tok, ka_head, kb_tok, kb_head, krow0, part);
        F2A_BAR();  // K ready; every wave also finished its PV (V^T reads) of the previous tile

        const bool wave_live = kb0 <= lim_last;
        f2a_op_t pf[F2A_NKS];
        if (wave_live) f2a_softmax(st, pf, ksb, qf, qt, kb0, l16, hh, lim_first, lim_i, scale);

        f2a_fix_v(va, vb, kb0, kend, vklo);
        f2a_store_v(vsb, va, vb, vg0, vp);
        f2a_load_v(va, vb, vv, kb0 + F2A_KT, kend, h, n_heads, v_tok, v_head, vklo, vg0);
        F2A_BAR();  // V^T ready (and every wave finished its QK)

        if (wave_live) f2a_pv(st, pf, vsb, l16);
    }

    // ---- epilogue: y = O / l (the incumbent's gfx11 path) -----------------------------------------------------------
    const float l_tot = st.l + __shfl_xor(st.l, 16, 32);
    const bool row_ok = row_l < n_rows;
    float* yrow = y + ((size_t)(row_ok ? row_l : 0) * n_heads + h) * F2A_DV;
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        // acc[j] = column 2j + hh: swap halves so each lane owns 8 consecutive columns.
        float o[8], own[4], rcv[4];
#pragma unroll
        for (int mm = 0; mm < 4; ++mm) {
            const float send = hh ? st.o[d][mm] : st.o[d][4 + mm];
            own[mm] = hh ? st.o[d][4 + mm] : st.o[d][mm];
            rcv[mm] = __shfl_xor(send, 16, 32);
        }
#pragma unroll
        for (int mm = 0; mm < 4; ++mm) {
            o[2 * mm] = hh ? rcv[mm] : own[mm];
            o[2 * mm + 1] = hh ? own[mm] : rcv[mm];
        }
        float r[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float v = (l_tot > 0.0f) ? o[j] / l_tot : 0.0f;
            if (round_y) v = __uint_as_float(f2a_f32_to_bf16(v) << 16);
            r[j] = v;
        }
        if (row_ok) {
            float4* dst = (float4*)(yrow + d * 16 + 8 * hh);
            dst[0] = make_float4(r[0], r[1], r[2], r[3]);
            dst[1] = make_float4(r[4], r[5], r[6], r[7]);
        }
    }
}
