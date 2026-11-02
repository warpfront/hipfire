// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// HIPFIRE_COMPILER_FLAGS: -mcumode
//
// gfx1151 (Strix Halo) twin of the gfx11 Q8 FA2 prefill body
// (attention_q8_0_fa2_gqa.gfx11.hip, production configuration: Q16 tile,
// KT32, warp-specialized fill). Selected on exact gfx1151 only;
// HIPFIRE_GFX1151_FA2_TWIN=0 restores the gfx11 module. Same launch contract
// and kernargs as `attention_q8_0_fa2_gqa_gfx11` (grid [ceil(batch/16), 4, 1],
// block 256, 32,768 B dynamic LDS; f16 Q from the preconvert below).
//
// Geometry, WMMA lane mapping, LDS plane layouts and every arithmetic op
// (dequant, QK/PV WMMA order, mask, online softmax, P row, rescale,
// normalization) are the gfx11 FILL path's verbatim, so outputs are
// bit-identical. What differs is only where and in which order work runs:
//   * CU mode (-mcumode): a workgroup's eight waves share one CU, so each
//     SIMD holds three compute waves and one helper (WGP mode puts waves w
//     and w+4 on one SIMD: compute 2/1/2/1 per SIMD), and __syncthreads()
//     needs no buffer_gl0_inv.
//   * Heaviest q-tile first: q_base comes from the reversed blockIdx.x, so
//     a whole-chunk launch (kernel.gfx11_q8_fa2_wide) dispatches the longest
//     causal prefixes first and ends on the shortest.
//   * The QK accumulators are zeroed by opaque v_movs each tile instead of
//     copies of a zero vector held across the loop (-8 VGPRs), and Q rows
//     are loaded unconditionally from a clamped row: a row past the batch
//     only feeds its own query's scores, which the mask replaces by -inf,
//     and that query is never stored.
//   * Helper V stores: lane hl stores key pair (hl & 15) of block s for
//     code dwords 2(hl>>4), 2(hl>>4)+1, and odd half-waves write their dims
//     in (c ^ 1) order, so each ds_store_b32 of a helper wave covers 32
//     distinct LDS banks (the gfx11 slot order hits 4). The V plane holds
//     the same bytes; only the lane that writes each dword changes.
//   * The O rescale is skipped when alpha == 1.0f on every lane of a wave
//     (multiplying by 1.0f is the identity under the kernel's IEEE mode).
// Measured on a captured H2 layer (16 x 512-row commits, pp8192) against
// the gfx11 module: about -28% kernel time whole-chunk, byte-identical.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef int32_t __attribute__((ext_vector_type(2))) int32x2_t;
typedef _Float16 __attribute__((ext_vector_type(2))) half2_t;

typedef union {
    uint32_t u;
    _Float16 h[2];
} fa2_dw_t;

typedef union {
    uint32_t u;
    half2_t h2;
} fa2_pk_t;

// 2*bit_reverse4(c): the gfx11 plane swizzle.
__device__ __forceinline__ int fa2_swiz(int c)
{
    const int r = ((c & 1) << 3) | ((c & 2) << 1)
                | ((c & 4) >> 1) | ((c & 8) >> 3);
    return 2 * r;
}

constexpr int FA2_FILL_HT = 64;               // helper threads (2 waves)
constexpr int FA2_FILL_HK = 32 * 8 / 64;      // K slots per helper lane
constexpr int FA2_FILL_HV = 32 * 16 / 64;     // V slots per helper lane
constexpr int FA2_ROW_B = 4 * (256 / 32) * 34;

// Two biased codes (bytes picked by `sel` from tb, interleaved with 0x64)
// times the f16 scale pair s2: RNE(s*c) per half.
__device__ __forceinline__ uint32_t fa2_pk_deq2(uint32_t tb, uint32_t sel, uint32_t s2)
{
    fa2_pk_t a, s;
    a.u = __builtin_amdgcn_perm(0x64646464u, tb, sel);
    s.u = s2;
    const half2_t bias = {(_Float16)1152.0f, (_Float16)1152.0f};
    a.h2 = (a.h2 - bias) * s.h2;
    return a.u;
}

// K slot idx = k*8+b: scale (low 16 bits) + the block's 8 code dwords.
// Out-of-range keys read the (valid) last row and stage zeros.
__device__ __forceinline__ void fa2_fill_k_load(
    const unsigned char* k_cache, int ktile, int seq_len, int kv_blk, int idx,
    uint32_t& su, uint32_t* uc)
{
    const int gk = ktile + (idx >> 3);
    const bool ok = gk < seq_len;
    const unsigned char* blk = k_cache
        + (unsigned long long)min(gk, seq_len - 1) * (unsigned long long)FA2_ROW_B
        + (unsigned long long)(kv_blk + (idx & 7)) * 34ULL;
    const uint32_t s = (uint32_t)blk[0] | ((uint32_t)blk[1] << 8);
    su = ok ? s : 0u;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        uint32_t u;
        __builtin_memcpy(&u, blk + 2 + w * 4, 4);
        uc[w] = ok ? u : 0u;
    }
}

// K pair p of key k lands at K[k][(p + S(k&15)) & 127].
__device__ __forceinline__ void fa2_fill_k_store(uint32_t* Kdw, int idx, uint32_t su,
                                                 const uint32_t* uc)
{
    const int k = idx >> 3;
    const int rot = fa2_swiz(k & 15);
    const int base = k * 128;
    const int pb = (idx & 7) * 16;
    const uint32_t s2 = su | (su << 16);
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const uint32_t tb = uc[w] ^ 0x80808080u;
        int32x2_t v;
        v[0] = (int32_t)fa2_pk_deq2(tb, 0x04010400u, s2);
        v[1] = (int32_t)fa2_pk_deq2(tb, 0x04030402u, s2);
        *(int32x2_t*)(Kdw + base + ((pb + w * 2 + rot) & 127)) = v;
    }
}

// V slot = ((m*8+b)<<2) | wq: key pair m, block b, code dwords 2wq, 2wq+1
// of both keys. Raw = packed scales (s0 | s1<<16), u[0..1] (key 2m),
// u[2..3] (key 2m+1).
__device__ __forceinline__ void fa2_fill_v_load(
    const unsigned char* v_cache, int ktile, int seq_len, int kv_blk, int slot,
    uint32_t& s01, uint32_t* u)
{
    const int pair = slot >> 2;
    const int w_base = (slot & 3) * 2;
    const int k0 = ktile + 2 * (pair >> 3);
    const unsigned long long boff = (unsigned long long)(kv_blk + (pair & 7)) * 34ULL;
    const bool ok0 = k0 < seq_len;
    const bool ok1 = k0 + 1 < seq_len;
    const unsigned char* blk0 = v_cache
        + (unsigned long long)min(k0, seq_len - 1) * (unsigned long long)FA2_ROW_B + boff;
    const unsigned char* blk1 = v_cache
        + (unsigned long long)min(k0 + 1, seq_len - 1) * (unsigned long long)FA2_ROW_B + boff;
    const uint32_t s0 = ok0 ? ((uint32_t)blk0[0] | ((uint32_t)blk0[1] << 8)) : 0u;
    const uint32_t s1 = ok1 ? ((uint32_t)blk1[0] | ((uint32_t)blk1[1] << 8)) : 0u;
    s01 = s0 | (s1 << 16);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        uint32_t u0, u1;
        __builtin_memcpy(&u0, blk0 + 2 + (w_base + i) * 4, 4);
        __builtin_memcpy(&u1, blk1 + 2 + (w_base + i) * 4, 4);
        u[i] = ok0 ? u0 : 0u;
        u[2 + i] = ok1 ? u1 : 0u;
    }
}

// V slot s of helper lane hl (0..63): key pair m = hl & 15, block b = s,
// code dwords 2wq, 2wq+1 with wq = hl >> 4 (a permutation of the gfx11
// slot set hl + 64s).
__device__ __forceinline__ int fa2_vslot(int hl, int s)
{
    return ((((hl & 15) * 8 + s) << 2) | (hl >> 4));
}

// All eight V slots of helper lane hl. Key pair m for dim d lands at
// V_T[d][(m + S(d&15)) & 15]; each dword holds (s0*code of key 2m, s1*code
// of key 2m+1) for one dim. Row d = 32s + rowc with rowc = 4w + (c ^ odd),
// so d & 15 = rowc & 15: each store is one of eight per-lane bases plus the
// immediate 2048*s bytes. Odd half-waves produce each code pair's dims
// swapped (selectors exchanged) and store them at c ^ 1.
__device__ __forceinline__ void fa2_fill_v_store_all(uint32_t* Vdw, int hl, const uint32_t* rs,
                                                     const uint32_t* ru)
{
    const int odd = (hl >> 4) & 1;
    const int m = hl & 15;
    const int wq = hl >> 4;
    const uint32_t sel0 = odd ? 0x04030402u : 0x04010400u;
    const uint32_t sel1 = odd ? 0x04010400u : 0x04030402u;
    uint32_t* base[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int rowc = (2 * wq + i) * 4 + (c ^ odd);
            base[i][c] = Vdw + rowc * 16 + ((m + fa2_swiz(rowc & 15)) & 15);
        }
    }
#pragma unroll
    for (int s = 0; s < FA2_FILL_HV; ++s) {
        const uint32_t* u = ru + s * 4;
        const uint32_t s01 = rs[s];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t xl =
                __builtin_amdgcn_perm(u[2 + i], u[i], 0x05010400u) ^ 0x80808080u;
            const uint32_t xh =
                __builtin_amdgcn_perm(u[2 + i], u[i], 0x07030602u) ^ 0x80808080u;
            uint32_t dw[4];
            dw[0] = fa2_pk_deq2(xl, sel0, s01);
            dw[1] = fa2_pk_deq2(xl, sel1, s01);
            dw[2] = fa2_pk_deq2(xh, sel0, s01);
            dw[3] = fa2_pk_deq2(xh, sel1, s01);
#pragma unroll
            for (int c = 0; c < 4; ++c)
                base[i][c][s * 512] = dw[c];
        }
    }
}

// K fragment for WMMA A: key row `key`, dim chunk dc (four aligned b64s).
__device__ __forceinline__ half16_t fa2_fill_kfrag(const uint32_t* Kdw, int key, int dc)
{
    const int rot = fa2_swiz(key & 15);
    const int base = key * 128;
    half16_t kf;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const int off = base + ((dc * 8 + i + rot) & 127);
        const int32x2_t pair = *(const int32x2_t*)(Kdw + off);
        fa2_dw_t d0, d1;
        d0.u = (uint32_t)pair[0];
        d1.u = (uint32_t)pair[1];
        kf[2 * i] = d0.h[0];
        kf[2 * i + 1] = d0.h[1];
        kf[2 * i + 2] = d1.h[0];
        kf[2 * i + 3] = d1.h[1];
    }
    return kf;
}

// Q fragment (B operand) of dim chunk dc: two aligned half8 loads of the
// f16 row at qq_base (clamped to the last valid row for rows past the batch).
__device__ __forceinline__ half16_t fa2_qfrag(const _Float16* __restrict__ q16, unsigned qq_base,
                                              int dc)
{
    half16_t qf;
    const half8_t* qd = (const half8_t*)(q16 + qq_base + (unsigned)(dc * 16));
    const half8_t lo = qd[0];
    const half8_t hi = qd[1];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        qf[e] = lo[e];
        qf[8 + e] = hi[e];
    }
    return qf;
}

// V^T fragment (A operand) of dim chunk dc, key half ps (0 or 8 pairs).
__device__ __forceinline__ half16_t fa2_vfrag(const uint32_t* Vdw, int ml, int dc, int ps)
{
    const int d = dc * 16 + ml;
    const int rot = fa2_swiz(d & 15);
    const int vbase = d * 16;
    half16_t vf;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const int off = vbase + ((ps + i + rot) & 15);
        const int32x2_t pair = *(const int32x2_t*)(Vdw + off);
        fa2_dw_t d0, d1;
        d0.u = (uint32_t)pair[0];
        d1.u = (uint32_t)pair[1];
        vf[2 * i] = d0.h[0];
        vf[2 * i + 1] = d0.h[1];
        vf[2 * i + 2] = d1.h[0];
        vf[2 * i + 3] = d1.h[1];
    }
    return vf;
}

// Eight +0.0f accumulators produced by opaque v_movs, so the compiler
// cannot keep a zero vector live across the tile loop.
__device__ __forceinline__ float8_t fa2_zero8()
{
    float a0, a1, a2, a3, a4, a5, a6, a7;
    asm volatile("v_mov_b32 %0, 0\n\tv_mov_b32 %1, 0\n\tv_mov_b32 %2, 0\n\tv_mov_b32 %3, 0\n\t"
                 "v_mov_b32 %4, 0\n\tv_mov_b32 %5, 0\n\tv_mov_b32 %6, 0\n\tv_mov_b32 %7, 0"
                 : "=v"(a0), "=v"(a1), "=v"(a2), "=v"(a3), "=v"(a4), "=v"(a5), "=v"(a6), "=v"(a7));
    return (float8_t){a0, a1, a2, a3, a4, a5, a6, a7};
}

// QK(t) for both KT32 subtiles: S^T = K * Q^T, one Q fragment per dim
// chunk shared by the two subtiles' WMMAs, dc order 0..15.
__device__ __forceinline__ void fa2_fill_qk(
    const _Float16* __restrict__ q16, const uint32_t* Kdw, unsigned qq_base,
    int ml, bool do0, bool do1, float8_t& sacc0, float8_t& sacc1)
{
    sacc0 = fa2_zero8();
    sacc1 = fa2_zero8();
    for (int dc = 0; dc < 16; ++dc) {
        const half16_t qf = fa2_qfrag(q16, qq_base, dc);
        if (do0)
            sacc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                fa2_fill_kfrag(Kdw, ml, dc), qf, sacc0);
        if (do1)
            sacc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                fa2_fill_kfrag(Kdw, 16 + ml, dc), qf, sacc1);
        // Bound K/Q-load hoisting to this chunk.
        asm volatile("" ::: "memory");
    }
}

// One 16-key subtile: mask/scale, online softmax (xor-16 pair reductions),
// replicated P row, O rescale, PV (O^T = V^T * P^T).
__device__ __forceinline__ void fa2_sub(
    const uint32_t* Vdw, const int* __restrict__ positions, float scale_attn,
    int kt_sub, int gmin, int ps, float8_t sacc,
    bool qok_ml, int qr_ml, int ml, int half, float& m_old, float& l_val,
    float8_t* Ofr)
{
    const bool full = (kt_sub + 15) <= gmin;
    const int pos_ml = (!full && qok_ml) ? positions[qr_ml] : 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int key_g = kt_sub + 2 * j + half;
        const bool ok = qok_ml && (full || key_g <= pos_ml);
        sacc[j] = ok ? sacc[j] * scale_attn : -INFINITY;
    }
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        mx = fmaxf(mx, sacc[j]);
    mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
    const float m_new = fmaxf(m_old, mx);
    const float alpha =
        (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
    m_old = m_new;
    float earr[8];
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float e = (m_new == -INFINITY) ? 0.0f : __expf(sacc[j] - m_new);
        earr[j] = e;
        sum += e;
    }
    sum += __shfl_xor(sum, 16, 32);
    l_val = l_val * alpha + sum;
    half16_t prow;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float peer = __shfl_xor(earr[j], 16, 32);
        const float e0 = (half == 0) ? earr[j] : peer;
        const float e1 = (half == 0) ? peer : earr[j];
        prow[2 * j] = (_Float16)e0;
        prow[2 * j + 1] = (_Float16)e1;
    }
    // Skip the O rescale when alpha == 1.0f on every lane of the wave (the
    // running max did not move): with f32 denormals preserved and RNE,
    // x * 1.0f == x for every non-NaN x, so skipping it changes no bit.
    const bool resc = __builtin_amdgcn_ballot_w32(alpha != 1.0f) != 0u;
    if (resc) {
#pragma unroll
        for (int dc = 0; dc < 16; ++dc)
            Ofr[dc] *= alpha;
    }
#pragma unroll
    for (int dc = 0; dc < 16; ++dc) {
        const half16_t vf = fa2_vfrag(Vdw, ml, dc, ps);
        Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vf, prow, Ofr[dc]);
        // Bound V-load hoisting to this chunk.
        asm volatile("" ::: "memory");
    }
}

__device__ __forceinline__ void fa2_fill_softmax_pv(
    const uint32_t* Vdw, const int* __restrict__ positions, float scale_attn,
    int ktile, int gmin, bool do0, bool do1, float8_t sacc0, float8_t sacc1,
    bool qok_ml, int qr_ml, int ml, int half, float& m_old, float& l_val,
    float8_t* Ofr)
{
#pragma unroll 1
    for (int sub = 0; sub < 2; ++sub) {
        const bool dos = (sub == 0) ? do0 : do1;
        if (!dos)
            continue;
        float8_t sacc = (sub == 0) ? sacc0 : sacc1;
        fa2_sub(Vdw, positions, scale_attn, ktile + sub * 16, gmin, sub * 8, sacc, qok_ml, qr_ml, ml, half,
                m_old, l_val, Ofr);
    }
}

// Workgroup body: KV head kv_h, sixteen positions from q_base. Waves 0..5
// compute (head 6*kv_h + wave), waves 6..7 fill: per tile, phase A (QK(t)
// while helpers dequantize V(t)), barrier, phase B (softmax/PV(t) while
// helpers dequantize K(t+1)), barrier.
__device__ __forceinline__ void fa2_gqa_body(
    const _Float16* __restrict__ q16,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int batch_size,
    float scale_attn,
    int kv_h,
    int q_base)
{
    constexpr int KT = 32;
    constexpr int LDS_DWORDS = KT * 256;

    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int half = lane >> 4;
    const bool compute = wave < 6;

    const int kv_blk = kv_h * (256 / 32);

    extern __shared__ uint32_t LDS[];
    uint32_t* Kdw = LDS;
    uint32_t* Vdw = LDS + KT * 128;

    float8_t Ofr[16];
    float m_old = -INFINITY;
    float l_val = 0.0f;
    int h_ml = 0;
    int qr_ml = 0;
    bool qok_ml = false;
    unsigned qq_base = 0;
    if (compute) {
        h_ml = 6 * kv_h + wave;
        qr_ml = q_base + ml;
        qok_ml = qr_ml < batch_size;
        qq_base = (unsigned)min(qr_ml, batch_size - 1) * 6144u + (unsigned)h_ml * 256u;
#pragma unroll
        for (int dc = 0; dc < 16; ++dc)
            Ofr[dc] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    // Per-WG causal bounds (positions[] is authoritative), published through
    // two transient LDS mailbox dwords (later overwritten by V).
    int wmx = -1;
    int wmn = 0x7fffffff;
    if (tid < 16) {
        const int qr = q_base + tid;
        if (qr < batch_size) {
            const int p = positions[qr];
            wmx = p;
            wmn = p;
        }
    }
    wmx = max(wmx, __shfl_xor(wmx, 16, 32));
    wmx = max(wmx, __shfl_xor(wmx, 8, 32));
    wmx = max(wmx, __shfl_xor(wmx, 4, 32));
    wmx = max(wmx, __shfl_xor(wmx, 2, 32));
    wmx = max(wmx, __shfl_xor(wmx, 1, 32));
    wmn = min(wmn, __shfl_xor(wmn, 16, 32));
    wmn = min(wmn, __shfl_xor(wmn, 8, 32));
    wmn = min(wmn, __shfl_xor(wmn, 4, 32));
    wmn = min(wmn, __shfl_xor(wmn, 2, 32));
    wmn = min(wmn, __shfl_xor(wmn, 1, 32));
    if (tid == 0) {
        LDS[LDS_DWORDS - 2] = (uint32_t)wmx;
        LDS[LDS_DWORDS - 1] = (uint32_t)wmn;
    }
    __syncthreads();
    const int gmax = __builtin_amdgcn_readfirstlane((int)LDS[LDS_DWORDS - 2]);
    const int gmin = __builtin_amdgcn_readfirstlane((int)LDS[LDS_DWORDS - 1]);
    const int seq_len = gmax + 1;

    const bool fill_compute = __builtin_amdgcn_readfirstlane(wave) < 6;
    const int hl = tid - 6 * 32;
    uint32_t rv_s[FA2_FILL_HV];
    uint32_t rv_u[FA2_FILL_HV * 4];
    if (!fill_compute) {
#pragma unroll
        for (int s = 0; s < FA2_FILL_HV; ++s)
            fa2_fill_v_load(v_cache, 0, seq_len, kv_blk, fa2_vslot(hl, s),
                            rv_s[s], rv_u + s * 4);
    }
    // Tile 0 K: one slot per thread (all eight waves).
    {
        uint32_t su;
        uint32_t uc[8];
        fa2_fill_k_load(k_cache, 0, seq_len, kv_blk, tid, su, uc);
        fa2_fill_k_store(Kdw, tid, su, uc);
    }
    __syncthreads();
    if (fill_compute) {
        for (int tile = 0; ; ++tile) {
            const int ktile = tile * KT;
            if (ktile >= seq_len)
                break;
            const bool do0 = ktile <= gmax;
            const bool do1 = ktile + 16 <= gmax;
            float8_t sacc0, sacc1;
            if (do0 || do1)
                fa2_fill_qk(q16, Kdw, qq_base, ml, do0, do1, sacc0, sacc1);
            __syncthreads();  // B1: V(t) in plane; K(t) consumed
            if (do0 || do1)
                fa2_fill_softmax_pv(Vdw, positions, scale_attn, ktile, gmin, do0, do1,
                                    sacc0, sacc1, qok_ml, qr_ml, ml, half, m_old,
                                    l_val, Ofr);
            __syncthreads();  // B2: K(t+1) in plane; V(t) consumed
        }
    } else {
        uint32_t rk_s[FA2_FILL_HK];
        uint32_t rk_u[FA2_FILL_HK * 8];
        for (int tile = 0; ; ++tile) {
            const int ktile = tile * KT;
            if (ktile >= seq_len)
                break;
            const bool next = ktile + KT < seq_len;
            if (next) {
#pragma unroll
                for (int t = 0; t < FA2_FILL_HK; ++t)
                    fa2_fill_k_load(k_cache, ktile + KT, seq_len, kv_blk,
                                    hl + t * FA2_FILL_HT, rk_s[t], rk_u + t * 8);
            }
            fa2_fill_v_store_all(Vdw, hl, rv_s, rv_u);
            __syncthreads();  // B1
            if (next) {
#pragma unroll
                for (int s = 0; s < FA2_FILL_HV; ++s)
                    fa2_fill_v_load(v_cache, ktile + KT, seq_len, kv_blk,
                                    fa2_vslot(hl, s), rv_s[s], rv_u + s * 4);
#pragma unroll
                for (int t = 0; t < FA2_FILL_HK; ++t)
                    fa2_fill_k_store(Kdw, hl + t * FA2_FILL_HT, rk_s[t], rk_u + t * 8);
            }
            __syncthreads();  // B2
        }
    }

    // Direct epilogue: Ofr[dc][j] at lane (ml, half) = O[query=ml][dim=
    // dc*16+2*j+half]; the pair-local l_val normalizes; one writer each.
    if (compute) {
        if (qok_ml) {
            const float inv = l_val > 0.0f ? 1.0f / l_val : 0.0f;
            const unsigned bo =
                (unsigned)qr_ml * 6144u + (unsigned)h_ml * 256u;
#pragma unroll
            for (int dc = 0; dc < 16; ++dc) {
                const unsigned bd = bo + (unsigned)(dc * 16 + half);
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    out[bd + (unsigned)(j * 2)] = Ofr[dc][j] * inv;
            }
        }
    }
}

// f32 Q [batch, 24, 256] -> f16 scratch, the gfx11 preconvert under this
// module's own symbol (the function cache is keyed by symbol).
extern "C" __global__ __launch_bounds__(128, 1) void attention_fa2_q_preconvert_gfx1151(
    const float* __restrict__ q,
    _Float16* __restrict__ q16,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int batch_size,
    int do_rotate)
{
    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int r = (int)blockIdx.x * 4 + wave;
    if (r >= batch_size * 24)
        return;
    const int qr = r / 24;
    const int h = r % 24;
    const float* rp = q + (unsigned)qr * 6144u + (unsigned)h * 256u;
    _Float16* wp = q16 + (unsigned)qr * 6144u + (unsigned)h * 256u;
    float v0 = rp[lane * 8 + 0], v1 = rp[lane * 8 + 1];
    float v2 = rp[lane * 8 + 2], v3 = rp[lane * 8 + 3];
    float v4 = rp[lane * 8 + 4], v5 = rp[lane * 8 + 5];
    float v6 = rp[lane * 8 + 6], v7 = rp[lane * 8 + 7];
    (void)signs1;
    (void)signs2;
    (void)do_rotate;
    half8_t o;
    o[0] = (_Float16)v0; o[1] = (_Float16)v1;
    o[2] = (_Float16)v2; o[3] = (_Float16)v3;
    o[4] = (_Float16)v4; o[5] = (_Float16)v5;
    o[6] = (_Float16)v6; o[7] = (_Float16)v7;
    *(half8_t*)(wp + lane * 8) = o;
}

extern "C" __global__ __launch_bounds__(256, 1) void attention_q8_0_fa2_gqa_gfx1151(
    const _Float16* __restrict__ q16,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int batch_size,
    float scale_attn)
{
    if (n_heads != 24 || n_kv_heads != 4 || head_dim != 256)
        return;
    const int kv_h = blockIdx.y;
    if (kv_h >= 4)
        return;
    // Heaviest q-tile (latest positions, longest causal prefix) first.
    const int q_base = ((int)gridDim.x - 1 - (int)blockIdx.x) * 16;
    if (q_base >= batch_size)
        return;
    fa2_gqa_body(q16, k_cache, v_cache, out, positions, batch_size,
                 scale_attn, kv_h, q_base);
}
