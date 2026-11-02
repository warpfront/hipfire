// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// VerifyAttn Stage 0 (gfx1100): Q8_0 verify attention with one KV scan
// shared by all 6 q heads x 8 query rows of a kv head, split over a fixed,
// context-independent grid. gfx1100 verify runs two references, so this file
// carries a byte-identical twin of each:
//
//   * `attention_verify_gqa_q8_gfx1100` (TILE mode) writes the same
//     `[row][head][max_tiles][2 + 256]` f32 partials as
//     `attention_flash_q8_0_tile_batched` (the "batched" verify variant: every
//     verify at logical context <= HIPFIRE_FA_PERTOKEN_MIN_CTX, B < 4, and
//     every graph-captured verify).
//   * `attention_verify_gqa_q8_rows_gfx1100` (ROWS mode) writes the same
//     partials as the multi-row tile `attention_flash_q8_0_rows{8,4}_d8`
//     (eager Q8 verify, 4 <= B <= 32, logical context above the threshold),
//     including the empty records R4/R8 write for a row whose own causal
//     bound ends before a tile its row group still covers.
//   * `attention_verify_reduce_gfx1100` reduces either byte-identically to
//     `attention_flash_asym_reduce_batched`.
//
// Shape is compile-time: head_dim 256, GQA group 6, tile 128, full causal,
// legacy single-slot KV (the host admits nothing else).
//
// Reference arithmetic reproduced (hipcc -O3 gfx1100 ISA of the references;
// reference lane l covers dims 8l..8l+7):
//   * kq = ks * (float)code, exact in f32 (ks = f16 block scale of block l/4);
//   * leaf_l = fma(q7,kq7, ... fma(q1,kq1, fma(q0,kq0, +0))) in dim order;
//   * score = (xor-16, 8, 4, 2, 1 butterfly sum of the 32 leaves) * scale
//     (TILE: 1/16; ROWS: log2(e)/16, the host's f32 values);
//   * TILE softmax per tile: the reference's lane-strided max / expf / sum;
//     PV: out_d = fma(e_t, vq_{t,d}, out_d) over keys in increasing order;
//   * ROWS online softmax per key in increasing order (R4/R8 source):
//       if (s > m) { c = exp2(m - s); sum *= c; out *= c; m = s; }
//       e = exp2(s - m); sum += e; out_d = fma(e, vq_d, out_d)
//     with v_exp_f32 (__builtin_amdgcn_exp2f) and m starting at -1e30; the
//     record is [m * 0.69314718f, sum, out[256]]. The running max m is the
//     prefix "leftmost max" of the scores, which is associative, so it is
//     scanned across lanes; each key's e (or, at a key that raises the max,
//     its raw score) is precomputed per pair and the sum / out recurrences
//     stay serial in key order, exactly as the reference runs them.
//   * keys at or past a row's causal bound are skipped (never multiplied by 0).
// Phase A layout: a 16-lane row scores one key per pass; row lane r holds
// reference lanes r (dims 8r..8r+7) and r+16 (dims 128+8r..), so the xor-16
// step is in-thread and 8/4/2/1 are DPP row_xmask adds.
//
// A row group of R <= 8 rows runs on R waves, each owning 6 (row, head)
// pairs for the whole tile (scores, softmax and PV), so every dequantized K/V
// slice serves 6 pairs and the tile loop has no workgroup barrier; the waves
// of a workgroup read the same K/V bytes through L0.
//
// fwht4 / fwht3 K (Q8 V only): `attention_verify_gqa_{fwht4,fwht3}_gfx1100`
// (TILE mode, the 10-argument q8 entry ABI followed by `signs1`, `signs2`)
// write the same partials as `attention_flash_{fwht4,fwht3}_tile_batched` at
// v_mode 8. Q arrives un-rotated f32; every wave rotates its 6 pairs with the
// references' own `fwht_shfl_forward` / `fwht_shfl_forward_256` (turbo_common.h)
// on the reference lane layout (lane = reference lane), then redistributes
// the rotated values to the 16-lane-row layout with ds_bpermute (exact moves).
// Reference arithmetic (hipcc -O3 gfx1100 ISA of the two references):
//   * kq = cnorm * C[nibble] (one f32 product, table = TURBO_C4 / TURBO_C3_256);
//   * fwht4 reference lane l: dims 4l..4l+3 (half 0) and 128+4l..+3 (half 1);
//     X_h = fma(m3,k3, fma(m2,k2, fma(m0,k0, m1*k1))), leaf = (0 + X_0) + X_1;
//   * fwht3 reference lane l: dims 8l..8l+7, leaf =
//     fma(m7,k7, ... fma(m2,k2, fma(m0,k0, m1*k1))...);
//   * score = (xor-16, 8, 4, 2, 1 butterfly sum of the 32 leaves) * scale,
//     then the TILE softmax and PV exactly as for Q8 (V is Q8_0 in both).
// The table lookup is a ds_bpermute from a per-lane register table (lane L
// holds C[L & 15] / C[L & 7]); it returns the exact table float.
#include <hip/hip_runtime.h>
#ifndef VA_TURBO_COMMON_INLINED
#include "turbo_common.h"
#endif

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1100__)
#error "attention_verify_gqa_gfx1100 requires --offload-arch=gfx1100"
#endif

#define VA_TILE 128
#define VA_HD 256
#define VA_STRIDE (2 + VA_HD)
#define VA_GROUP 6
#define VA_WAVES 8
#define VA_THREADS (VA_WAVES * 32)
#define VA_ROWS 8
// (row, head) pairs per wave: 8 rows x 6 heads over 8 waves.
#define VA_PPW (VA_ROWS * VA_GROUP / VA_WAVES)
// Buffer descriptor dword3 for raw buffer loads (as the gfx1100 decode tile).
#define VA_RSRC_FLAGS 0x31004000u

enum { VA_MODE_TILE = 0, VA_MODE_ROWS = 1 };

template <int Ctrl>
__device__ __forceinline__ float va_row_xmask_add(float v) {
    const int other = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), Ctrl, 0xf, 0xf, false);
    return v + __builtin_bit_cast(float, other);
}

typedef _Float16 va_h2 __attribute__((ext_vector_type(2)));

// Four signed Q8 codes as exact f16 values: byte ^ 0x80 = code + 128 is
// placed under exponent byte 0x64 (f16 1024 + u, v_perm_b32), then 1152 is
// subtracted (v_pk_add_f16). Every step is exact, and code 0 gives +0, as
// (float)(signed char)0 does.
__device__ __forceinline__ void va_codes4(unsigned int w, va_h2& c01, va_h2& c23) {
    const unsigned int u = w ^ 0x80808080u;
    const va_h2 bias = {(_Float16)1152.0f, (_Float16)1152.0f};
    c01 = __builtin_bit_cast(va_h2, __builtin_amdgcn_perm(0x64646464u, u, 0x04010400u)) - bias;
    c23 = __builtin_bit_cast(va_h2, __builtin_amdgcn_perm(0x64646464u, u, 0x04030402u)) - bias;
}

// Eight signed Q8 codes times their f16 block scale: f16 x f16 products are
// exact in f32 (v_fma_mix_f32), equal to the references' ks * (float)code,
// signed zeros included.
__device__ __forceinline__ void va_dequant_q8(unsigned int lo, unsigned int hi, _Float16 s, float (&out)[8]) {
    va_h2 c[4];
    va_codes4(lo, c[0], c[1]);
    va_codes4(hi, c[2], c[3]);
    const float sf = (float)s;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[2 * i] = sf * (float)c[i].x;
        out[2 * i + 1] = sf * (float)c[i].y;
    }
}

// One reference-lane leaf: sequential fma over its 8 dims from +0.
__device__ __forceinline__ float va_leaf(const float (&q)[8], const float (&k)[8]) {
    float p = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) p = __builtin_fmaf(q[i], k[i], p);
    return p;
}

// K/V bytes of one 8-dim slice of one key: the codes (two dwords) and scale.
struct VaSlice {
    unsigned int lo, hi;
    unsigned short sc;
};

// The 8-dim Q8 slice starting at `dim` of kv head `kv_h`, key row `row_off`
// bytes from the tile's first row.
__device__ __forceinline__ VaSlice va_load_slice(__amdgpu_buffer_rsrc_t rsrc, unsigned row_off, int kv_h, int dim) {
    VaSlice s;
    const unsigned blk = row_off + (unsigned)(kv_h * (VA_HD / 32) + (dim >> 5)) * 34u;
    const auto w = __builtin_amdgcn_raw_buffer_load_b64(rsrc, blk + 2u + (unsigned)(dim & 31), 0, 0);
    s.lo = w[0];
    s.hi = w[1];
    s.sc = __builtin_amdgcn_raw_buffer_load_b16(rsrc, blk, 0, 0);
    return s;
}

__device__ __forceinline__ void va_dequant(const VaSlice& s, float (&out)[8]) {
    va_dequant_q8(s.lo, s.hi, __builtin_bit_cast(_Float16, s.sc), out);
}

// ── fwht K kinds ─────────────────────────────────────────────────────────
enum { VA_K_Q8 = 0, VA_K_F4 = 1, VA_K_F3 = 2 };
// K bytes per kv head: f32 cnorm + 4-bit nibbles (fwht4) / 3-bit (fwht3).
#define VA_K4_HEAD 132
#define VA_K3_HEAD 100

template <int KIND>
struct VaKRegs;
template <>
struct VaKRegs<VA_K_Q8> {
    VaSlice a, b;  // reference lanes rl and rl + 16 (dims 8rl.., 128 + 8rl..)
};
template <>
struct VaKRegs<VA_K_F4> {
    float cn;
    // 16 packed nibbles per reference lane: p[0] = lane rl half 0, p[1] = lane
    // rl + 16 half 0, p[2] = lane rl half 1, p[3] = lane rl + 16 half 1.
    unsigned short p[4];
};
template <>
struct VaKRegs<VA_K_F3> {
    float cn;
    unsigned int a, b;  // 24-bit packed indices of reference lanes rl, rl + 16
};

// Exact table float: lane `idx` of `tab` (the per-lane register table).
__device__ __forceinline__ float va_lut(float tab, unsigned int idx) {
    return __builtin_bit_cast(
        float, __builtin_amdgcn_ds_bpermute((int)(idx << 2), __builtin_bit_cast(int, tab)));
}

// The K bytes one 16-lane row lane needs of key row `row_off` (bytes from the
// tile's first row) of kv head `kv_h`, for reference lanes rl and rl + 16.
template <int KIND>
__device__ __forceinline__ VaKRegs<KIND> va_k_load(__amdgpu_buffer_rsrc_t rsrc, unsigned row_off, int kv_h, int rl) {
    VaKRegs<KIND> r;
    if constexpr (KIND == VA_K_Q8) {
        r.a = va_load_slice(rsrc, row_off, kv_h, 8 * rl);
        r.b = va_load_slice(rsrc, row_off, kv_h, 128 + 8 * rl);
    } else if constexpr (KIND == VA_K_F4) {
        const unsigned kb = row_off + (unsigned)(kv_h * VA_K4_HEAD);
        r.cn = __builtin_bit_cast(float, __builtin_amdgcn_raw_buffer_load_b32(rsrc, kb, 0, 0));
#pragma unroll
        for (int j = 0; j < 4; ++j)
            r.p[j] = __builtin_amdgcn_raw_buffer_load_b16(rsrc, kb + 4u + 32u * (unsigned)j + 2u * (unsigned)rl, 0, 0);
    } else {
        const unsigned kb = row_off + (unsigned)(kv_h * VA_K3_HEAD);
        r.cn = __builtin_bit_cast(float, __builtin_amdgcn_raw_buffer_load_b32(rsrc, kb, 0, 0));
        const unsigned oa = kb + 4u + 3u * (unsigned)rl;
        const unsigned ob = oa + 48u;
        r.a = (unsigned)__builtin_amdgcn_raw_buffer_load_b16(rsrc, oa, 0, 0) |
              ((unsigned)__builtin_amdgcn_raw_buffer_load_b8(rsrc, oa + 2u, 0, 0) << 16);
        r.b = (unsigned)__builtin_amdgcn_raw_buffer_load_b16(rsrc, ob, 0, 0) |
              ((unsigned)__builtin_amdgcn_raw_buffer_load_b8(rsrc, ob + 2u, 0, 0) << 16);
    }
    return r;
}

// Four fwht4 centroids of one packed u16: cnorm * C[nibble i], i = 0..3.
__device__ __forceinline__ void va_dec4(float cn, float tab, unsigned int pk, float* out) {
#pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = cn * va_lut(tab, (pk >> (4 * i)) & 0xFu);
}

// The reference's 4-term dot: ((m0*k0 + m1*k1) + m2*k2) + m3*k3 contracted as
// fma(m3,k3, fma(m2,k2, fma(m0,k0, m1*k1))).
__device__ __forceinline__ float va_dot4(const float* m, const float* k) {
    float x = m[1] * k[1];
    x = __builtin_fmaf(m[0], k[0], x);
    x = __builtin_fmaf(m[2], k[2], x);
    return __builtin_fmaf(m[3], k[3], x);
}

// fwht4 reference-lane leaf: partial = 0; partial += X_half0; partial += X_half1.
__device__ __forceinline__ float va_leaf_f4(const float (&q)[8], const float* k) {
    float p = 0.0f;
    p += va_dot4(q, k);
    p += va_dot4(q + 4, k + 4);
    return p;
}

// fwht3 reference-lane leaf (8-term dot, same contraction shape as va_dot4).
__device__ __forceinline__ float va_leaf_f3(const float (&q)[8], const float (&k)[8]) {
    float x = q[1] * k[1];
    x = __builtin_fmaf(q[0], k[0], x);
#pragma unroll
    for (int i = 2; i < 8; ++i) x = __builtin_fmaf(q[i], k[i], x);
    return x;
}

// Pre-butterfly sum (leaf(rl) + leaf(rl + 16)) of one key for the first NP pairs.
template <int KIND, int NP>
__device__ __forceinline__ void va_k_score(const VaKRegs<KIND>& c, float tab, const float (&qa)[VA_PPW][8],
                                           const float (&qb)[VA_PPW][8], float (&v)[NP]) {
    if constexpr (KIND == VA_K_Q8) {
        float ka[8], kb[8];
        va_dequant(c.a, ka);
        va_dequant(c.b, kb);
#pragma unroll
        for (int j = 0; j < NP; ++j) v[j] = va_leaf(qa[j], ka) + va_leaf(qb[j], kb);
    } else if constexpr (KIND == VA_K_F4) {
        float ka[8], kb[8];
        va_dec4(c.cn, tab, c.p[0], ka);
        va_dec4(c.cn, tab, c.p[2], ka + 4);
        va_dec4(c.cn, tab, c.p[1], kb);
        va_dec4(c.cn, tab, c.p[3], kb + 4);
#pragma unroll
        for (int j = 0; j < NP; ++j) v[j] = va_leaf_f4(qa[j], ka) + va_leaf_f4(qb[j], kb);
    } else {
        float ka[8], kb[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            ka[i] = c.cn * va_lut(tab, (c.a >> (3 * i)) & 7u);
            kb[i] = c.cn * va_lut(tab, (c.b >> (3 * i)) & 7u);
        }
#pragma unroll
        for (int j = 0; j < NP; ++j) v[j] = va_leaf_f3(qa[j], ka) + va_leaf_f3(qb[j], kb);
    }
}

// One pair's q8-layout Q registers: qa = reference lane rl, qb = rl + 16.
// fwht kinds rotate on the reference lane layout (lane = reference lane) and
// redistribute with exact lane moves.
template <int KIND>
__device__ __forceinline__ void va_load_q(const float* qr, const float* signs1, const float* signs2, int lane,
                                          int rl, float (&qa)[8], float (&qb)[8]) {
    if constexpr (KIND == VA_K_Q8) {
        const float4 a0 = *(const float4*)(qr + 8 * rl);
        const float4 a1 = *(const float4*)(qr + 8 * rl + 4);
        const float4 b0 = *(const float4*)(qr + 128 + 8 * rl);
        const float4 b1 = *(const float4*)(qr + 128 + 8 * rl + 4);
        qa[0] = a0.x; qa[1] = a0.y; qa[2] = a0.z; qa[3] = a0.w;
        qa[4] = a1.x; qa[5] = a1.y; qa[6] = a1.z; qa[7] = a1.w;
        qb[0] = b0.x; qb[1] = b0.y; qb[2] = b0.z; qb[3] = b0.w;
        qb[4] = b1.x; qb[5] = b1.y; qb[6] = b1.z; qb[7] = b1.w;
    } else {
        float mq[8];
        if constexpr (KIND == VA_K_F4) {
            for (int half = 0; half < 2; half++) {
                const int d0 = half * 128 + lane * 4;
                float a = qr[d0 + 0];
                float b = qr[d0 + 1];
                float c = qr[d0 + 2];
                float d = qr[d0 + 3];
                fwht_shfl_forward(a, b, c, d, signs1, signs2, lane);
                mq[half * 4 + 0] = a;
                mq[half * 4 + 1] = b;
                mq[half * 4 + 2] = c;
                mq[half * 4 + 3] = d;
            }
        } else {
            const int d0 = lane * 8;
            float mq0 = qr[d0 + 0], mq1 = qr[d0 + 1];
            float mq2 = qr[d0 + 2], mq3 = qr[d0 + 3];
            float mq4 = qr[d0 + 4], mq5 = qr[d0 + 5];
            float mq6 = qr[d0 + 6], mq7 = qr[d0 + 7];
            fwht_shfl_forward_256(mq0, mq1, mq2, mq3, mq4, mq5, mq6, mq7, signs1, signs2, lane);
            mq[0] = mq0; mq[1] = mq1; mq[2] = mq2; mq[3] = mq3;
            mq[4] = mq4; mq[5] = mq5; mq[6] = mq6; mq[7] = mq7;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qa[i] = __shfl(mq[i], rl);
            qb[i] = __shfl(mq[i], rl + 16);
        }
    }
}

// ROWS mode: out = fma(w, vq, out) and sum += w for one key of one pair,
// after the reference's rescale when this key raises the running max
// (`raise`; `w` then holds the raw score, else the precomputed e).
__device__ __forceinline__ void va_rows_step(bool raise, float w, const float (&vq)[8], float& m, float& sum,
                                             float (&acc)[8]) {
    if (raise) {
        const float corr = __builtin_amdgcn_exp2f(m - w);
        sum *= corr;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] *= corr;
        m = w;
        w = __builtin_amdgcn_exp2f(w - m);
    }
    sum += w;
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = __builtin_fmaf(w, vq[i], acc[i]);
}

// ROWS: word k of a pair's raise mask (bit l = key 32k + l raises the
// running max), selected with uniform compares instead of an indexed array.
__device__ __forceinline__ unsigned int va_word(const unsigned int (&r)[4], int k) {
    return k == 0 ? r[0] : k == 1 ? r[1] : k == 2 ? r[2] : r[3];
}

__device__ __forceinline__ bool va_raised(const unsigned int (&r)[4], int t) {
    return (va_word(r, t >> 5) >> (t & 31)) & 1u;
}

// Every tile of one wave's workgroup range, for the wave's first NP pairs
// (the valid pairs of a wave are always a prefix j < NP, so the pair loops
// are straight-line and interleave across pairs).
template <int MODE, int NP, int KIND>
__device__ __forceinline__ void va_tiles(
    const unsigned char* __restrict__ k_cache, const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials, float (&sc)[VA_PPW][VA_TILE], const float (&qa)[VA_PPW][8],
    const float (&qb)[VA_PPW][8], const int (&pair_len)[VA_PPW], const int (&pair_tiles)[VA_PPW],
    const size_t (&pair_base)[VA_PPW], int group_len, int n_tiles, int kv_h, unsigned per_pos_k,
    unsigned per_pos, float tab, float scale_attn) {
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

        const size_t k_tile_base = (size_t)tile_start * per_pos_k;
        const size_t tile_base = (size_t)tile_start * per_pos;
        const auto k_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)(k_cache + k_tile_base), 0, (int)((unsigned)tl_max * per_pos_k), VA_RSRC_FLAGS);
        const auto v_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)(v_cache + tile_base), 0, (int)((unsigned)tl_max * per_pos), VA_RSRC_FLAGS);

        // ── Phase A: scores. Pass k: row `half` scores key 32(k>>4) + (k&15) + 16*half.
        {
            auto key_of = [&](int k) { return 32 * (k >> 4) + (k & 15) + 16 * half; };
            unsigned off = (unsigned)min(key_of(0), tl_max - 1) * per_pos_k;
            VaKRegs<KIND> ck = va_k_load<KIND>(k_rsrc, off, kv_h, rl);
            for (int k = 0; 32 * (k >> 4) + (k & 15) < tl_max; ++k) {
                const int t = key_of(k);
                const unsigned noff = (unsigned)min(key_of(k + 1), tl_max - 1) * per_pos_k;
                const VaKRegs<KIND> nk = va_k_load<KIND>(k_rsrc, noff, kv_h, rl);
                float v[NP];
                va_k_score<KIND, NP>(ck, tab, qa, qb, v);
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
                ck = nk;
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();

        // ROWS: raise[j][u] bit l = key 4l+u of pair j raises the running max.
        unsigned raise[NP][4];
        if (MODE == VA_MODE_TILE) {
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
        } else {
            // ── Phase B (ROWS): the running max as a lane scan, 32 keys at a
            //    time (lane l holds key 32k + l of block k). m_t = leftmost
            //    max of (-1e30, s_0..s_t) under `s > m`, which is associative.
            //    Each key's slot becomes e_t = exp2(s_t - m_t), or s_t itself
            //    where s_t raises the max.
#pragma unroll
            for (int j = 0; j < NP; ++j) {
                const int tile_len = tl[j];
#pragma unroll
                for (int k = 0; k < 4; ++k) raise[j][k] = 0u;
                float carry = -1e30f;
                for (int k = 0; 32 * k < tile_len; ++k) {
                    const int t = 32 * k + lane;
                    const bool valid = t < tile_len;
                    const float s = valid ? sc[j][t] : -1e30f;
                    float m = s;
#pragma unroll
                    for (int d = 1; d < 32; d <<= 1) {
                        const float left = __shfl_up(m, d);
                        if (lane >= d && !(m > left)) m = left;
                    }
                    float prev = __shfl_up(m, 1);
                    if (lane == 0) prev = -1e30f;
                    if (prev > carry) carry = prev;
                    prev = carry;
                    const bool r = valid && s > prev;
                    const float mt = r ? s : prev;
                    if (valid) sc[j][t] = r ? s : __builtin_amdgcn_exp2f(s - mt);
                    const unsigned int bits = (unsigned int)__builtin_amdgcn_ballot_w32(r);
                    raise[j][0] = k == 0 ? bits : raise[j][0];
                    raise[j][1] = k == 1 ? bits : raise[j][1];
                    raise[j][2] = k == 2 ? bits : raise[j][2];
                    raise[j][3] = k == 3 ? bits : raise[j][3];
                    carry = __shfl(mt, 31);
                }
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();

        // ── Phase D: PV. Lane l owns dims 8l..8l+7 of every pair. The full
        //    path runs every pair (a pair with no key in this tile computes
        //    unused sums); the ragged path skips keys past each pair's bound.
        float acc[NP][8];
        float rm[NP], rs[NP];
#pragma unroll
        for (int j = 0; j < NP; ++j) {
            rm[j] = -1e30f;
            rs[j] = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[j][i] = 0.0f;
        }
        {
            VaSlice cv = va_load_slice(v_rsrc, 0u, kv_h, 8 * lane);
            if (full) {
                // Four keys per step: one 16-byte LDS broadcast of each
                // pair's weights, the next step's four V slices in flight.
                const int t4 = tl_max & ~3;
                VaSlice c4[4];
#pragma unroll
                for (int u = 0; u < 4; ++u)
                    c4[u] = va_load_slice(v_rsrc, (unsigned)min(u, tl_max - 1) * per_pos, kv_h, 8 * lane);
                int t = 0;
                for (; t < t4; t += 4) {
                    VaSlice n4[4];
#pragma unroll
                    for (int u = 0; u < 4; ++u)
                        n4[u] = va_load_slice(v_rsrc, (unsigned)min(t + 4 + u, tl_max - 1) * per_pos, kv_h,
                                              8 * lane);
                    float4 w4[NP];
#pragma unroll
                    for (int j = 0; j < NP; ++j) w4[j] = *(const float4*)&sc[j][t];
                    // ROWS: raise bits of keys t..t+3 (t % 4 == 0, same mask word).
                    unsigned int nib[NP];
#pragma unroll
                    for (int j = 0; j < NP; ++j)
                        nib[j] = MODE == VA_MODE_ROWS ? (va_word(raise[j], t >> 5) >> (t & 31)) & 0xFu : 0u;
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        float vq[8];
                        va_dequant(c4[u], vq);
#pragma unroll
                        for (int j = 0; j < NP; ++j) {
                            const float w = w4[j][u];
                            if (MODE == VA_MODE_TILE) {
#pragma unroll
                                for (int i = 0; i < 8; ++i) acc[j][i] = __builtin_fmaf(w, vq[i], acc[j][i]);
                            } else {
                                va_rows_step((nib[j] >> u) & 1u, w, vq, rm[j], rs[j], acc[j]);
                            }
                        }
                    }
#pragma unroll
                    for (int u = 0; u < 4; ++u) c4[u] = n4[u];
                }
                for (; t < tl_max; ++t) {
                    float vq[8];
                    va_dequant(c4[0], vq);
#pragma unroll
                    for (int j = 0; j < NP; ++j) {
                        const float w = sc[j][t];
                        if (MODE == VA_MODE_TILE) {
#pragma unroll
                            for (int i = 0; i < 8; ++i) acc[j][i] = __builtin_fmaf(w, vq[i], acc[j][i]);
                        } else {
                            va_rows_step(va_raised(raise[j], t), w, vq, rm[j], rs[j], acc[j]);
                        }
                    }
                    c4[0] = c4[1];
                    c4[1] = c4[2];
                    c4[2] = c4[3];
                }
            } else {
                for (int t = 0; t < tl_max; ++t) {
                    const VaSlice nv = va_load_slice(
                        v_rsrc, (unsigned)min(t + 1, tl_max - 1) * per_pos, kv_h, 8 * lane);
                    float vq[8];
                    va_dequant(cv, vq);
#pragma unroll
                    for (int j = 0; j < NP; ++j) {
                        if (t >= tl[j]) continue;
                        const float w = sc[j][t];
                        if (MODE == VA_MODE_TILE) {
#pragma unroll
                            for (int i = 0; i < 8; ++i) acc[j][i] = __builtin_fmaf(w, vq[i], acc[j][i]);
                        } else {
                            va_rows_step(va_raised(raise[j], t), w, vq, rm[j], rs[j], acc[j]);
                        }
                    }
                    cv = nv;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NP; ++j) {
            if (MODE == VA_MODE_TILE) {
                if (tl[j] == 0) continue;
            } else {
                // R4/R8 write every valid row of a row group whose bound
                // reaches this tile; a row with no key here keeps the
                // initial state.
                if (tile >= pair_tiles[j]) continue;
                if (tl[j] == 0) {
                    rm[j] = -1e30f;
                    rs[j] = 0.0f;
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[j][i] = 0.0f;
                }
            }
            float* p = partials + (pair_base[j] + tile) * VA_STRIDE;
            if (MODE == VA_MODE_ROWS && lane == 0) {
                p[0] = rm[j] * 0.69314718f;
                p[1] = rs[j];
            }
            p += 2 + 8 * lane;
#pragma unroll
            for (int i = 0; i < 8; i += 2) *(float2*)(p + i) = make_float2(acc[j][i], acc[j][i + 1]);
        }
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();
    }
}

// group_rows (ROWS only): the reference's row-group size (8 or 4); a pair
// writes the tiles its row group's bound reaches.
template <int MODE, int KIND>
__device__ __forceinline__ void va_verify_gqa(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ positions, int n_rows, int n_kv_heads, int max_tiles,
    int batch_offset, float scale_attn, int group_rows, const float* __restrict__ signs1,
    const float* __restrict__ signs2) {
    __shared__ __attribute__((aligned(16))) float sc_lds[VA_WAVES][VA_PPW][VA_TILE];

    const int n_heads = n_kv_heads * VA_GROUP;
    const int n_rg = (n_rows + VA_ROWS - 1) / VA_ROWS;
    const int kv_h = (int)blockIdx.x / n_rg;
    const int rg = (int)blockIdx.x - kv_h * n_rg;
    if (kv_h >= n_kv_heads) return;
    const int row0 = rg * VA_ROWS;
    const int rows_here = min(VA_ROWS, n_rows - row0);

    const int wave = __builtin_amdgcn_readfirstlane((int)threadIdx.x >> 5);
    const int rl = (int)threadIdx.x & 15;
    // One wave per row of the group: a group of R rows (6R pairs) runs on R
    // waves of exactly 6 pairs, so each dequantized K/V slice always serves 6
    // pairs; waves R..7 exit (the host sizes the split grid by active waves).
    const int n_waves = rows_here;
    if (wave >= n_waves) return;

    const unsigned per_pos = (unsigned)n_kv_heads * (VA_HD / 32) * 34u;
    const unsigned per_pos_k = KIND == VA_K_Q8   ? per_pos
                               : KIND == VA_K_F4 ? (unsigned)n_kv_heads * VA_K4_HEAD
                                                 : (unsigned)n_kv_heads * VA_K3_HEAD;
    const int lane = (int)threadIdx.x & 31;
    // Per-lane register table for va_lut: lane L holds C[L & 15] / C[L & 7].
    float tab = 0.0f;
    if (KIND == VA_K_F4) tab = TURBO_C4[lane & 15];
    if (KIND == VA_K_F3) tab = TURBO_C3_256[lane & 7];

    // Live key range of this row group, from the device positions.
    int group_len = 0;
    for (int r = 0; r < rows_here; ++r) group_len = max(group_len, positions[batch_offset + row0 + r] + 1);
    // The reference grid covers tiles < max_tiles only; never write past it.
    const int n_tiles = min((group_len + VA_TILE - 1) / VA_TILE, max_tiles);

    // This wave's pairs: p = wave + n_waves * j -> (row p/6, head p%6), all
    // six valid.
    int pair_len[VA_PPW];
    int pair_tiles[VA_PPW];
    size_t pair_base[VA_PPW];
    float qa[VA_PPW][8], qb[VA_PPW][8];
#pragma unroll
    for (int j = 0; j < VA_PPW; ++j) {
        const int p = wave + n_waves * j;
        const int r = p / VA_GROUP;
        const int head = kv_h * VA_GROUP + (p - r * VA_GROUP);
        pair_base[j] = ((size_t)(row0 + r) * n_heads + head) * max_tiles;
        pair_len[j] = positions[batch_offset + row0 + r] + 1;
        pair_tiles[j] = 0;
        if (MODE == VA_MODE_ROWS) {
            const int g0 = r - r % group_rows;
            const int g1 = min(g0 + group_rows, rows_here);
            int len = 0;
            for (int rr = g0; rr < g1; ++rr) len = max(len, positions[batch_offset + row0 + rr] + 1);
            pair_tiles[j] = min((len + VA_TILE - 1) / VA_TILE, max_tiles);
        }
        const float* qr = q + ((size_t)(row0 + r) * n_heads + head) * VA_HD;
        va_load_q<KIND>(qr, signs1, signs2, lane, rl, qa[j], qb[j]);
    }
    va_tiles<MODE, VA_PPW, KIND>(k_cache, v_cache, partials, sc_lds[wave], qa, qb, pair_len, pair_tiles,
                                 pair_base, group_len, n_tiles, kv_h, per_pos_k, per_pos, tab, scale_attn);
}

// Grid: [ceil(n_rows/8) * n_kv_heads, S], block 256. Workgroup x = kv_h *
// ceil(n_rows/8) + row group; workgroup y serves tiles y, y + S, ... below
// ceil((max row position + 1) / 128) of its row group. Partials rows are
// chunk-local; positions are read at batch_offset + row.
extern "C" __launch_bounds__(VA_THREADS, 1)
__global__ void attention_verify_gqa_q8_gfx1100(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ positions, int n_rows, int n_kv_heads, int max_tiles,
    int batch_offset, float scale_attn) {
    va_verify_gqa<VA_MODE_TILE, VA_K_Q8>(q, k_cache, v_cache, partials, positions, n_rows, n_kv_heads, max_tiles,
                                         batch_offset, scale_attn, VA_ROWS, nullptr, nullptr);
}

// fwht4 K (132 B/head: f32 cnorm + 4-bit FWHT-rotated nibbles), Q8_0 V: TILE
// twin of attention_flash_fwht4_tile_batched at v_mode 8. Same grid as q8;
// `signs1`/`signs2` are the reference's 128-entry sign tables, Q is un-rotated.
extern "C" __launch_bounds__(VA_THREADS, 1)
__global__ void attention_verify_gqa_fwht4_gfx1100(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ positions, int n_rows, int n_kv_heads, int max_tiles,
    int batch_offset, float scale_attn, const float* __restrict__ signs1,
    const float* __restrict__ signs2) {
    va_verify_gqa<VA_MODE_TILE, VA_K_F4>(q, k_cache, v_cache, partials, positions, n_rows, n_kv_heads, max_tiles,
                                         batch_offset, scale_attn, VA_ROWS, signs1, signs2);
}

// fwht3 K (100 B/head: f32 cnorm + 3-bit, 256-wide rotation), Q8_0 V: TILE
// twin of attention_flash_fwht3_tile_batched at v_mode 8. `signs1`/`signs2`
// are the reference's 256-entry sign tables.
extern "C" __launch_bounds__(VA_THREADS, 1)
__global__ void attention_verify_gqa_fwht3_gfx1100(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ positions, int n_rows, int n_kv_heads, int max_tiles,
    int batch_offset, float scale_attn, const float* __restrict__ signs1,
    const float* __restrict__ signs2) {
    va_verify_gqa<VA_MODE_TILE, VA_K_F3>(q, k_cache, v_cache, partials, positions, n_rows, n_kv_heads, max_tiles,
                                         batch_offset, scale_attn, VA_ROWS, signs1, signs2);
}

// ROWS twin of attention_flash_q8_0_rows{8,4}_d8: same grid and arguments
// plus `group_rows` (the reference's rows per block, 8 or 4); `scale_attn`
// is the reference's log2(e) / sqrt(256).
extern "C" __launch_bounds__(VA_THREADS, 1)
__global__ void attention_verify_gqa_q8_rows_gfx1100(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,
    const int* __restrict__ positions, int n_rows, int n_kv_heads, int max_tiles,
    int batch_offset, float scale_attn, int group_rows) {
    va_verify_gqa<VA_MODE_ROWS, VA_K_Q8>(q, k_cache, v_cache, partials, positions, n_rows, n_kv_heads, max_tiles,
                                         batch_offset, scale_attn, group_rows, nullptr, nullptr);
}

// Byte-identical parallel twin of attention_flash_asym_reduce_batched at
// head_dim 256: one thread per dim, each running the reference's serial tile
// order (max pass, then expf-corrected fma accumulation), then g * (1/sum).
// Grid: [n_heads, n_rows], block 256.
extern "C" __launch_bounds__(VA_HD, 1)
__global__ void attention_verify_reduce_gfx1100(
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
