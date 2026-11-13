// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// GEN-FA2 F2a-Q8 flash prefill, q128w8-kt64 body (symbol `latent_attention_flash_prefill_fa2_q8_q128w8`).
// INT8 expanded-mirror variant of `latent_attention_flash_prefill_q8` (kernels/src/latent_attention_flash.hip):
// K dequantized once per CTA at the LDS store, V dequantized in registers. Same 108-byte kernarg ABI (21 kernargs),
// block 256, dynamic LDS 44032 B, grid = the contract's ceil(n_rows / 64) x n_heads.
//
//   * CTA = one head x one query tile of 128 rows (8 compute waves, one wave per 16 query rows); CTAs past
//     ceil(n_rows / 128) return at once (before any barrier).
//   * Q (BF16-rounded once) is resident in VGPRs as 12 B-operand fragments (dq 192 = 128 + 64 shared tail); the F32
//     O^T accumulator (dv 128) is resident.
//   * K (ka cols 0..127 ++ kb cols 128..191 of one padded row) and V^T are streamed through LDS, one 64-key stage,
//     with LDS-only workgroup barriers (the "local" fences keep the global prefetches in flight).
//   * Load schedule (deferred widening): the global -> VGPR prefetch of the NEXT tile is issued right after the current
//     tile's stage is written and carries the RAW F16 scale bits; the F16 -> F32 widening (the same conversion, exact)
//     happens at the LDS store one tile later, so no load has a consumer before the store and the prefetch overlaps the
//     whole tile. Loads are unconditional, from a row clamped to [0, kend) (row 0 when row >= kend; kend >= 1 whenever
//     a load is issued); the zero fill (codes 0, scale bits 0, tail 0) of rows / keys at or past kend is applied to the
//     registers at the store, only in the CTA's last (partial) tile.
//   * S^T = K . Q^T (lane-local softmax state), O^T = V^T . P^T with the incumbent's fragment maps and key permutation
//     inside a 16-key sub-tile.
//
// Numeric recipe = the incumbent's, statement for statement: F32 scores * scale, tile max,
// p = exp2((s - m') * log2e) rounded to BF16 BEFORE both the PV product and the denominator, denominator summed from the
// rounded values in the fixed (sub-tile, j) order, per-lane partial l combined with one xor-16 exchange, `round_y`, the
// `*pos + pos_offset + i` / `max_rows` mask, empty rows -> zeros, shared tail head stride 0. Dequantization is
// BF16 RNE of the F32 product `code * scale` (F16 scale of (key j, head h) at byte `j * tok + n_heads * 128 + 2 * h` of
// an INT8 plane row `[n_heads * 128 codes | n_heads F16 scales]`). The output is bit-identical to the pre-defer body.
//
// Requirements (the contract domain enforces them; a violating launch leaves y untouched): wa 128, wb 64, dv 128,
// blockDim.x == 256, dynamic LDS == 44032, every offset/stride a multiple of 8 elements, ka_tok / v_tok multiples of
// 16 bytes, 16-byte aligned base pointers.

#define F2A_NW 8
#define F2A_KT 64

#include <hip/hip_runtime.h>
#include <stdint.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__GFX12__)
#error "F2a kernels are gfx12 (gfx1201) only"
#endif

#define F2A_WA 128
#define F2A_WB 64
#define F2A_DV 128
#define F2A_DQ (F2A_WA + F2A_WB)
#define F2A_NT (F2A_NW * 32)           // threads per CTA
#define F2A_QT (16 * F2A_NW)           // query rows per CTA
#define F2A_NKS (F2A_KT / 16)          // 16-key sub-tiles per key tile
#define F2A_KP 200                     // 192 + 8 (conflict-free b128 reads)
#define F2A_VPW 36
#define F2A_KRW (F2A_NT / 8)           // K rows filled per pass (8 threads x 16 B = one 128-byte half row)
#define F2A_KR (F2A_KT / F2A_KRW)      // fill passes per thread
#define F2A_KBYTES (F2A_KT * F2A_KP * 2)
#define F2A_VBYTES (F2A_DV * F2A_VPW * 4)
#define F2A_VOFF F2A_KBYTES
#define F2A_LDS (F2A_KBYTES + F2A_VBYTES)
#define F2A_VGS F2A_NW                 // V column groups (8 columns) covered per pass
#define F2A_VG (16 / F2A_VGS)          // passes per lane over the 16 column groups
static_assert(F2A_KR >= 1 && F2A_KT % F2A_KRW == 0, "K fill geometry");
static_assert(F2A_VG >= 1 && 16 % F2A_VGS == 0, "V fill geometry");

static __device__ __forceinline__ float f2a_exp(float x) {
    return __builtin_amdgcn_exp2f(x * 1.4426950408889634f);
}

typedef short __attribute__((ext_vector_type(8))) f2a_s8_t;
typedef unsigned int __attribute__((ext_vector_type(4))) f2a_u4_t;
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

// F16 scale (raw bits, zero-extended) widened to F32: the shipped body's `(float)(*(const _Float16*)p)`, at the store.
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

static __device__ __forceinline__ f2a_f8_t f2a_wmma(f2a_s8_t a, f2a_s8_t b, f2a_f8_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

// LDS-only workgroup barrier (see the shipped body): the "local" fences leave the global prefetches in flight.
extern "C" __device__ void f2a_wait_dscnt(unsigned short) __asm("llvm.amdgcn.s.wait.dscnt");
#define F2A_BAR()                                                         \
    do {                                                                  \
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");   \
        f2a_wait_dscnt(0);                                                \
        __builtin_amdgcn_s_barrier();                                     \
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");   \
    } while (0)

// ---- global -> VGPR prefetch of one key tile ----------------------------------------------------------------
// K: thread `part` of 8 owns codes [part*8, part*8+8) and [64 + part*8, ...) of the row's 128 ka codes (kr[r][0] =
// {lo.x, lo.y, hi.x, hi.y}), the row's RAW F16 scale bits (kr[r][1].x) and the BF16 tail words kr[r][2]. Rows at or past
// `kend` read the (valid) row 0; the shipped zero fill (codes 0, scale 0, tail 0) is applied to the registers at the store
// (`f2a_fix_k`, only for the CTA's last, partial tile), so the consumer of a load is a whole tile away. The caller
// guarantees kend >= 1.
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

// The shipped zero fill of the rows of the tile at `kbase` that lie at or past `kend` (uniform: only a partial last tile).
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

// The K tile is dequantized here, once per CTA: BF16 RNE of code * scale, written in the LDS layout.
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

// V: lane = key pair `vp` of the tile (keys (4q + r) | (4q + 2 + r) of 16-key sub-tile vks, word 2q + r), one 8-column
// group g per pass. Raw codes of 8 columns in `.x .y`, the key's RAW F16 scale bits in `.z` (same in every pass). Keys at
// or past `kend` load the (valid) row 0; `f2a_fix_v` zeroes them (codes 0, scale 0, the shipped fill) at the store.
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

// One V^T task (Q8): the two keys' raw codes are dequantized here (BF16 RNE of code * scale), then word packing.
// SWZ: row j of a column group lives at word (row * 32) + (p ^ (4 j)) (16-byte chunk XOR (row & 7); g * 8 is a multiple of 8).
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

// Per-lane LDS read offsets of the swizzled tiles (unused without SWZ): K chunk c = 2 s + hh of row (ks * 16 + l16) sits
// at chunk c ^ (l16 & 7); V^T chunk 2 ks + hh likewise. kofs[t]: element offset (t = s & 3, + 64 * (s >> 2) + ks * 16 * KP);
// vofs[ks]: word offset (+ d * 16 * 32).
struct F2aOfs {
    int kofs[4];
    int vofs[4];
};

static __device__ __forceinline__ F2aOfs f2a_lane_ofs(int l16, int hh) {
    F2aOfs o;
    const int x = l16 & 7;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int low = ((2 * t) ^ (x & 6)) + (hh ^ (x & 1));
        o.kofs[t] = l16 * F2A_KP + 8 * low;
        o.vofs[t] = l16 * F2A_VPW + 4 * low;
    }
    return o;
}

// ---- S^T = K . Q^T for one key tile from the LDS stage ----------------------------------------------------------
static __device__ __forceinline__ void f2a_ld_k(
    f2a_s8_t (&fr)[F2A_NKS], const uint16_t* ksb, int s, int l16, int hh, const F2aOfs& ofs
) {
#pragma unroll
    for (int ks = 0; ks < F2A_NKS; ++ks) {
        const uint4 u = *(const uint4*)(ksb + (size_t)(ks * 16 + l16) * F2A_KP + 16 * s + 8 * hh);
        fr[ks] = __builtin_bit_cast(f2a_s8_t, u);
    }
}

// The LDS fragments are software-pipelined one contraction step ahead and the scheduler is fenced per step.
static __device__ __forceinline__ void f2a_qk(
    f2a_f8_t (&acc)[F2A_NKS], const uint16_t* ksb, const f2a_s8_t (&qf)[12], int l16, int hh, const F2aOfs& ofs
) {
    f2a_s8_t cur[F2A_NKS], nxt[F2A_NKS];
    f2a_ld_k(cur, ksb, 0, l16, hh, ofs);
#pragma unroll
    for (int s = 0; s < 12; ++s) {
        if (s + 1 < 12) f2a_ld_k(nxt, ksb, s + 1, l16, hh, ofs);
#pragma unroll
        for (int ks = 0; ks < F2A_NKS; ++ks) acc[ks] = f2a_wmma(cur[ks], qf[s], acc[ks]);
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int ks = 0; ks < F2A_NKS; ++ks) cur[ks] = nxt[ks];
    }
}

struct F2aState {
    f2a_f8_t o[8];
    float m;
    float l;
};

static __device__ __forceinline__ void f2a_softmax(
    F2aState& st, f2a_s8_t (&pf)[F2A_NKS], const uint16_t* ksb, const f2a_s8_t (&qf)[12], int kb0, int l16, int hh,
    int lim_first, int lim_i, float scale, const F2aOfs& ofs
) {
    f2a_f8_t acc_s[F2A_NKS];
#pragma unroll
    for (int ks = 0; ks < F2A_NKS; ++ks) acc_s[ks] = f2a_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
    f2a_qk(acc_s, ksb, qf, l16, hh, ofs);

    const bool need_mask = kb0 + (F2A_KT - 1) > lim_first;
    float sv[F2A_NKS][8];
    float tmax = -INFINITY;
#pragma unroll
    for (int ks = 0; ks < F2A_NKS; ++ks) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float s = acc_s[ks][j] * scale;
            if (need_mask) {
                const int kj = kb0 + ks * 16 + 8 * hh + j;
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
        // word m' packs acc[4(m'>>1) + (m'&1)] | acc[... + 2] << 16.
        uint32_t pk[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int lo = 4 * (w >> 1) + (w & 1);
            pk[w] = (pw[lo] >> 16) | pw[lo + 2];
        }
        const f2a_u4_t W = {pk[0], pk[1], pk[2], pk[3]};
        pf[ks] = __builtin_bit_cast(f2a_s8_t, W);
    }
    st.l = st.l * corr + lsum;
#pragma unroll
    for (int d = 0; d < 8; ++d) st.o[d] *= corr;
}

static __device__ __forceinline__ void f2a_ld_v(
    f2a_s8_t (&fr)[F2A_NKS], const uint32_t* vsb, int d, int l16, int hh, const F2aOfs& ofs
) {
    const uint32_t* arow = vsb + (size_t)(d * 16 + l16) * F2A_VPW;
#pragma unroll
    for (int ks = 0; ks < F2A_NKS; ++ks) {
        const uint4 u = *(const uint4*)(arow + ks * 8 + 4 * hh);
        fr[ks] = __builtin_bit_cast(f2a_s8_t, u);
    }
}

static __device__ __forceinline__ void f2a_pv(
    F2aState& st, const f2a_s8_t (&pf)[F2A_NKS], const uint32_t* vsb, int l16, int hh, const F2aOfs& ofs
) {
    f2a_s8_t cur[F2A_NKS], nxt[F2A_NKS];
    f2a_ld_v(cur, vsb, 0, l16, hh, ofs);
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        if (d + 1 < 8) f2a_ld_v(nxt, vsb, d + 1, l16, hh, ofs);
#pragma unroll
        for (int ks = 0; ks < F2A_NKS; ++ks) st.o[d] = f2a_wmma(cur[ks], pf[ks], st.o[d]);
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int ks = 0; ks < F2A_NKS; ++ks) cur[ks] = nxt[ks];
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
    // Contract guard (workgroup-uniform, before any barrier), as the shipped body.
    if (wa != F2A_WA || wb != F2A_WB || dv != F2A_DV || (int)blockDim.x != F2A_NT) return;

    const int ntiles = (n_rows + F2A_QT - 1) / F2A_QT;
    if ((int)blockIdx.x >= ntiles) return;  // grid.x is the contract's ceil(n_rows / 128)

    const int tid = threadIdx.x;
    const int wave = tid >> 5, lane = tid & 31, l16 = lane & 15, hh = lane >> 4;
    const int h = blockIdx.y;
    const int tile = ntiles - 1 - (int)blockIdx.x;  // heaviest tile first
    const int i0 = tile * F2A_QT;

    // Visible limit per query row: keys j <= lim_i. Rows past n_rows replicate the last row (their output is
    // never stored).
    const int base = *pos_ptr + pos_offset;
    const int wrow0 = i0 + wave * 16;
    const int row_l = wrow0 + l16;
    const int lim_row = min(row_l, n_rows - 1);
    const int lim_i = min(base + lim_row, max_rows - 1);
    const int lim_first = min(base + min(wrow0, n_rows - 1), max_rows - 1);
    const int lim_last = min(base + min(wrow0 + 15, n_rows - 1), max_rows - 1);
    const int kend = max(0, min(base + min(i0 + F2A_QT - 1, n_rows - 1), max_rows - 1) + 1);

    // ---- Q^T B-operand fragments, BF16-rounded once, resident for the loop.
    f2a_s8_t qf[12];
    {
        const bool ok = row_l < n_rows;
        const float* src = q + ((size_t)(ok ? row_l : 0) * n_heads + h) * F2A_DQ + 8 * hh;
#pragma unroll
        for (int s = 0; s < 12; ++s) {
            f2a_u4_t w = {0u, 0u, 0u, 0u};
            if (ok) {
                const float4 a = *(const float4*)(src + 16 * s);
                const float4 b = *(const float4*)(src + 16 * s + 4);
                w = f2a_u4_t{f2a_f32_to_bf16(a.x) | (f2a_f32_to_bf16(a.y) << 16),
                             f2a_f32_to_bf16(a.z) | (f2a_f32_to_bf16(a.w) << 16),
                             f2a_f32_to_bf16(b.x) | (f2a_f32_to_bf16(b.y) << 16),
                             f2a_f32_to_bf16(b.z) | (f2a_f32_to_bf16(b.w) << 16)};
            }
            qf[s] = __builtin_bit_cast(f2a_s8_t, w);
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
    const F2aOfs ofs = f2a_lane_ofs(l16, hh);

    uint4 kr[F2A_KR][3];
    uint4 va[F2A_VG], vb[F2A_VG];

    uint16_t* ksb = (uint16_t*)f2a_smem;
    uint32_t* vsb = (uint32_t*)(f2a_smem + F2A_VOFF);
    if (kend > 0) {
        f2a_load_k(kr, ka, kb, 0, kend, h, n_heads, ka_tok, ka_head, kb_tok, kb_head, krow0, part);
        f2a_load_v(va, vb, vv, 0, kend, h, n_heads, v_tok, v_head, vklo, vg0);
    }
    for (int kb0 = 0; kb0 < kend; kb0 += F2A_KT) {
        const bool more = kb0 + F2A_KT < kend;  // CTA-uniform; the next-tile loads are issued regardless (rows >= kend clamp to row 0)
        // K stage is free: every wave finished its QK of the previous tile before the previous V barrier (OVL: and its PV
        // before the previous iteration's closing barrier).
        f2a_fix_k(kr, kb0, kend, krow0);
        f2a_store_k(ksb, kr, krow0, part);
        f2a_load_k(kr, ka, kb, kb0 + F2A_KT, kend, h, n_heads, ka_tok, ka_head, kb_tok, kb_head, krow0, part);
        F2A_BAR();  // K ready; every wave also finished its PV (V^T reads) of the previous tile

        const bool wave_live = kb0 <= lim_last;
        f2a_s8_t pf[F2A_NKS];
        if (wave_live) f2a_softmax(st, pf, ksb, qf, kb0, l16, hh, lim_first, lim_i, scale, ofs);

        f2a_fix_v(va, vb, kb0, kend, vklo);
        f2a_store_v(vsb, va, vb, vg0, vp);
        f2a_load_v(va, vb, vv, kb0 + F2A_KT, kend, h, n_heads, v_tok, v_head, vklo, vg0);
        F2A_BAR();  // V^T ready (and every wave finished its QK)

        if (wave_live) f2a_pv(st, pf, vsb, l16, hh, ofs);
    }

    // ---- epilogue: y = O / l (the shipped gfx12 path) -------------------------------------------------------
    const float l_tot = st.l + __shfl_xor(st.l, 16, 32);
    const bool row_ok = row_l < n_rows;
    float* yrow = y + ((size_t)(row_ok ? row_l : 0) * n_heads + h) * F2A_DV;
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        float r[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float v = (l_tot > 0.0f) ? st.o[d][j] / l_tot : 0.0f;
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
