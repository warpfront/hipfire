// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// LAB ONLY: parametrized copy of the production KMODE=0 path of
// kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip (gfx1201 GQA-fused FA2
// prefill attention, Q8_0 K/V, f16 pre-converted Q). Production is untouched;
// this file exists to screen H=16/Hkv=2/D=256 (GQA8) next to the incumbent
// H=24/Hkv=4/D=256 (GQA6) geometry.
//
// Compile-time parameters (all optional, `#ifndef`-guarded; define before
// this file, e.g. via KernelCompiler prepended defines):
//   LAB_FA_HEADS     query heads H              (default 24)
//   LAB_FA_KV_HEADS  KV heads Hkv               (default 4)
//   LAB_FA_BLOCK     threads per block, derived default 32*(GQA/2 + 1)
//                    (128 @ GQA6, 160 @ GQA8). May be overridden upward.
//   GQA = H / Hkv. D is fixed at 256. Requirements (static_assert below):
//   H % Hkv == 0, GQA even, GQA >= 2, LAB_FA_BLOCK multiple of 32, >= 128 and
//   >= 32*(GQA/2 + 1).
//
// Geometry. One workgroup owns one KV head and eight token positions:
// 8*GQA live rows packed into GQA/2 sixteen-row WMMA tiles, one compute wave
// per tile (waves 0..GQA/2-1), plus one trailing helper wave. Dynamic LDS is
// exactly 65,536 B at every GQA (K plane 32 KiB + V plane 32 KiB, KT64 tile,
// fragment-major f16, frozen layout from production).
//   Grid:  direct  [ceil(batch/8), Hkv, 1]
//          partial [ceil(batch/8), Hkv, n_splits]   (n_splits 1..8)
//          merge   [ceil(batch*H/8), 1, 1], block 256 (8 recs per block)
//          qpreconvert [ceil(batch*H/4), 1, 1], block 128 (4 rows per block)
//   Row mapping (compute wave w, packed index t=0..15):
//     head_local = 2*w + (t >> 3); q_local = t & 7;
//     h = GQA*kv_h + head_local;   q_row = 8*blockIdx.x + q_local.
//
// Loader / barrier caveat. The cooperative K/V fills are written for exactly
// 128 filler threads (`idx = tid + t*128`, 4 K slots + 2 V slots per thread).
// At block > 128 (GQA8: 160) only tid < 128 executes the original fill loads;
// the remaining wave(s) skip the fill but still reach every __syncthreads(),
// and also skip compute (`wave < GQA/2` only). With GQA8 that means waves 0..3
// are both fillers and compute waves; wave 4 is a barrier-only helper. At the
// default 128 the guard is a compile-time constant true, so the 24/4 object
// follows the production KMODE=0 source path.
//
// Memory layouts (all strides derive from H/Hkv, D=256):
//   q16  [batch, H, 256] f16 row-major (written by the preconvert kernel);
//   out  [batch, H, 256] f32;
//   K/V  cache row per position: Hkv * 8 blocks * 34 B (f16 scale + 32 int8);
//        row_stride = Hkv*8*34 = 1088 B @ Hkv4, 544 B @ Hkv2.
//   Partial scratch (SPLIT_LSE record layout, f32 elements):
//        lse[records] then oacc[records][256], records = batch*H*n_splits,
//        record = (q_row*H + h)*n_splits + split.
//        Total floats = records*257. (Product host sizing of H=24 is
//        batch*24*n_splits*257; this lab must size with its own H.)
//
// Excluded from this copy (production KMODE=0 path never instantiates them):
// KMODE 3 (fwht3 K) / KMODE 8 (native fp8) fills, HIPFIRE_FA2_FP8 renames,
// stage-b/packet/qresident bodies, and the legacy 258-float [m,l,O] partial
// record epilogue/merge (unreachable when HIPFIRE_FA2_SPLIT_LSE_ENTRY is on,
// which it always is for KMODE=0 non-FP8). Arithmetic (Q f16 rounding, WMMA
// order, __expf/__logf online softmax, epilogue) is unchanged from production.
//
// No slots, no windows, no FP8, gfx1201 only.

#include <hip/hip_runtime.h>

#ifndef LAB_FA_HEADS
#define LAB_FA_HEADS 24
#endif
#ifndef LAB_FA_KV_HEADS
#define LAB_FA_KV_HEADS 4
#endif
#define LAB_FA_GQA (LAB_FA_HEADS / LAB_FA_KV_HEADS)
#define LAB_FA_CWAVES (LAB_FA_GQA / 2)
#ifndef LAB_FA_BLOCK
#define LAB_FA_BLOCK (32 * (LAB_FA_CWAVES + 1))
#endif
// Elements per (position) Q/O row across all heads: H * D.
#define LAB_FA_QSTRIDE (LAB_FA_HEADS * 256)

static_assert(LAB_FA_HEADS % LAB_FA_KV_HEADS == 0, "H must be a multiple of Hkv");
static_assert(LAB_FA_GQA >= 2 && (LAB_FA_GQA % 2) == 0,
              "GQA must be even (two heads per 16-row tile)");
static_assert(LAB_FA_BLOCK % 32 == 0, "block must be whole wave32s");
static_assert(LAB_FA_BLOCK >= 128, "fill loops need the 128 original filler threads");
static_assert(LAB_FA_BLOCK >= 32 * (LAB_FA_CWAVES + 1),
              "block must hold GQA/2 compute waves plus one helper wave");

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

typedef union {
    uint32_t u;
    _Float16 h[2];
} fa2_dw_t;

// Frozen fragment-offset helpers (half-element units; one fragment = 8).
// lane = 16*kg+ml. K plane starts at half 0, V plane at half 16384.
__device__ __forceinline__ int fa2_koff(int sub, int dc, int lane, int j)
{
    return ((dc * 4 + sub) * 32 + lane) * 8 + j;
}
__device__ __forceinline__ int fa2_voff(int sub, int dc, int lane, int j)
{
    return ((sub * 16 + dc) * 32 + lane) * 8 + j;
}
__device__ __forceinline__ int fa2_kfrag8(int sub, int dc, int lane)
{
    return (dc * 4 + sub) * 32 + lane;
}
__device__ __forceinline__ int fa2_vfrag8(int sub, int dc, int lane)
{
    return (sub * 16 + dc) * 32 + lane;
}

// Shared workgroup body. PARTIAL=false writes normalized O to `out`.
// PARTIAL=true requires SPLIT_LSE: CK split-KV accumulator contract in one
// caller-owned allocation, F32 LSE records first, then normalized F32 O
// accumulators, record index ((query * H + head) * n_splits + split);
// tile0/tile1 are ignored and each split takes its range of the live tiles
// derived from positions[]. Q arrives f16 pre-converted row-major
// [batch, H, 256] (see attention_fa2_q_preconvert_gfx1201).
template <bool PARTIAL, bool SPLIT_LSE = false>
__device__ __forceinline__ void fa2_gqa_body(
    const _Float16* __restrict__ q16,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int batch_size,
    float scale_attn,
    int kv_h,
    int q_base,
    int tile0,
    int tile1,
    int split,
    int n_splits)
{
    static_assert(!PARTIAL || SPLIT_LSE, "lab kernel only carries the SPLIT_LSE partial layout");

    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int kg = lane >> 4;
    const bool compute = wave < LAB_FA_CWAVES;
    // The cooperative fills are written for 128 threads; extra waves (block
    // > 128) skip the fill but still hit the barriers. Compile-time true at
    // block == 128.
    const bool filler = (LAB_FA_BLOCK == 128) || (tid < 128);

    const int row_stride = LAB_FA_KV_HEADS * (256 / 32) * 34;
    const int kv_blk = kv_h * (256 / 32);

    extern __shared__ uint32_t LDS[];
    // Fragment-major f16 planes: K at byte 0, V at byte 32768.
    _Float16* K16 = (_Float16*)LDS;
    _Float16* V16 = (_Float16*)(LDS + 8192);

    // ---- Transition 1: online state init (compute waves) ----
    // Q arrives f16 pre-converted; within a KT64 tile, each dim-chunk Q
    // fragment is loaded once (one aligned 16-byte half8 load per 8 dims, no
    // cvt) and reused for all four subtiles' QK WMMAs (F4a); softmax/PV
    // still run sub0..sub3 in order so online order and PV accumulation stay
    // unchanged.
    float8_t Ofr[16];
    float m_old = -INFINITY;
    float l_val = 0.0f;
    int h_ml = 0;
    int qr_ml = 0;
    bool qok_ml = false;
    unsigned qq_base = 0;
    if (compute) {
        h_ml = LAB_FA_GQA * kv_h + (wave << 1) + (ml >> 3);
        qr_ml = q_base + (ml & 7);
        qok_ml = qr_ml < batch_size;
        qq_base = (unsigned)qr_ml * (unsigned)LAB_FA_QSTRIDE
            + (unsigned)h_ml * 256u;
#pragma unroll
        for (int dc = 0; dc < 16; ++dc)
            Ofr[dc] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    // ---- Per-WG causal bounds (positions[] is authoritative) ----
    // Wave-local: each wave's lanes 0..7 read the same eight positions and
    // reduce them with the identical integer min/max graph, so all waves
    // derive the same gmin/gmax/seq_len with no mailbox and no barrier.
    // Other lanes keep the existing sentinel values.
    int wmx = -1;
    int wmn = 0x7fffffff;
    if (lane < 8) {
        const int qr = q_base + lane;
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
    // Uniformize through lane 0 of this wave; every wave reduced the same
    // eight positions, so every wave's lane 0 holds the same values.
    const int gmax = __builtin_amdgcn_readfirstlane(wmx);
    const int gmin = __builtin_amdgcn_readfirstlane(wmn);
    const int seq_len = gmax + 1;

    // ---- Transitions 2-4: KT64 tiles ----
    // SPLIT_LSE: this split owns a disjoint range of the live KT64 tiles
    // derived from positions[].
    int t_begin = tile0;
    int t_end = tile1;
    if constexpr (SPLIT_LSE) {
        const int tiles_total = (seq_len + 63) / 64;
        const int tiles_per_split = (tiles_total + n_splits - 1) / n_splits;
        t_begin = split * tiles_per_split;
        t_end = min(t_begin + tiles_per_split, tiles_total);
    }
    for (int tile = t_begin; tile < t_end; ++tile) {
        const int ktile = tile * 64;
        if (ktile >= seq_len)
            break;

        // ---- Transition 2: cooperative KT64 fill (128 filler threads) ----
        if (filler) {
        // K: 512 blocks (k*8+b), one block per slot. Codes are loaded on
        // demand per pair (no staging array); dequant arithmetic is
        // unchanged and the four halves land in their Kfrag(sub=k>>4, dc,
        // lane) fragment via fa2_koff. The 4-wide dim run at 4-aligned d0
        // never crosses an 8-dim fragment boundary, so all four share one
        // (dc, lane) with j = d0&7 .. +3.
        for (int t = 0; t < 4; ++t) {
            const int idx = tid + t * 128;
            const int k = idx >> 3;
            const int b = idx & 7;
            const int gk = ktile + k;
            const unsigned char* blk = (gk < seq_len)
                ? k_cache + (unsigned long long)gk * (unsigned long long)row_stride
                    + (unsigned long long)(kv_blk + b) * 34ULL
                : nullptr;
            uint16_t su = 0;
            if (blk != nullptr)
                su = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
            _Float16 s = ((fa2_dw_t){.u = (uint32_t)su}).h[0];
            const float sf = (float)s;
            const int sub = k >> 4;
            const int mlk = k & 15;
#pragma unroll 1
            // 8 quads cover the 32 codes (blk+2 may be 2-mod-4;
            // memcpy emits a correct unaligned load either way).
            for (int w = 0; w < 8; ++w) {
                uint32_t u = 0;
                if (blk != nullptr)
                    __builtin_memcpy(&u, blk + 2 + w * 4, 4);
                const float c0 = (float)(int8_t)u;
                const float c1 = (float)(int8_t)(u >> 8);
                const float c2 = (float)(int8_t)(u >> 16);
                const float c3 = (float)(int8_t)(u >> 24);
                const int d0 = b * 32 + w * 4;
                const int dc = d0 >> 4;
                const int ln = 16 * ((d0 >> 3) & 1) + mlk;
                const int o = fa2_koff(sub, dc, ln, d0 & 7);
                K16[o] = (_Float16)(sf * c0);
                K16[o + 1] = (_Float16)(sf * c1);
                K16[o + 2] = (_Float16)(sf * c2);
                K16[o + 3] = (_Float16)(sf * c3);
            }
        }
        // V: 256 key-pairs (m*8+b); each slot reads the even and odd block.
        // Dequant arithmetic is unchanged; the pair (V[2m][d], V[2m+1][d])
        // lands in its Vfrag(sub=(2m)>>4, dc, lane) fragment via fa2_voff.
        // 2m/2m+1 never straddle a sub/kg boundary, so both halves share
        // one fragment at consecutive j. Codes are loaded on demand per dim
        // (no staging array).
        for (int t = 0; t < 2; ++t) {
            const int pair = tid + t * 128;
            const int m = pair >> 3;
            const int b = pair & 7;
            const int k0 = ktile + 2 * m;
            const unsigned char* blk0 = (k0 < seq_len)
                ? v_cache + (unsigned long long)k0 * (unsigned long long)row_stride
                    + (unsigned long long)(kv_blk + b) * 34ULL
                : nullptr;
            const unsigned char* blk1 = (k0 + 1 < seq_len)
                ? v_cache + (unsigned long long)(k0 + 1) * (unsigned long long)row_stride
                    + (unsigned long long)(kv_blk + b) * 34ULL
                : nullptr;
            uint16_t s0u = 0;
            uint16_t s1u = 0;
            if (blk0 != nullptr)
                s0u = (uint16_t)blk0[0] | ((uint16_t)blk0[1] << 8);
            if (blk1 != nullptr)
                s1u = (uint16_t)blk1[0] | ((uint16_t)blk1[1] << 8);
            _Float16 s0 = ((fa2_dw_t){.u = (uint32_t)s0u}).h[0];
            _Float16 s1 = ((fa2_dw_t){.u = (uint32_t)s1u}).h[0];
            const float sf0 = (float)s0;
            const float sf1 = (float)s1;
            const int sub = (2 * m) >> 4;
            const int kgm = ((2 * m) >> 3) & 1;
            const int j0 = (2 * m) & 7;
#pragma unroll 1
            for (int w = 0; w < 8; ++w) {
                uint32_t u0 = 0, u1 = 0;
                if (blk0 != nullptr)
                    __builtin_memcpy(&u0, blk0 + 2 + w * 4, 4);
                if (blk1 != nullptr)
                    __builtin_memcpy(&u1, blk1 + 2 + w * 4, 4);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int d = b * 32 + w * 4 + c;
                    const float c0 = (float)(int8_t)(u0 >> (c * 8));
                    const float c1 = (float)(int8_t)(u1 >> (c * 8));
                    const int dc = d >> 4;
                    const int ln = 16 * kgm + (d & 15);
                    const int o = fa2_voff(sub, dc, ln, j0);
                    V16[o] = (_Float16)(sf0 * c0);
                    V16[o + 1] = (_Float16)(sf1 * c1);
                }
        }
        }
        }
        __syncthreads();

        // ---- Transition 3: four KT16 QK/softmax/PV subtiles ----
        // F4a: each Q dim-chunk fragment is loaded once per tile and shared
        // by all four subtiles' QK WMMAs; softmax/PV still run sub0..sub3 in
        // order so online-softmax order and PV accumulation are unchanged
        // (bit-exact: every subtile accumulates dc 0..15 in the same order
        // as the old per-subtile reload).
        if (compute) {
            const bool do0 = ktile <= gmax;
            const bool do1 = ktile + 16 <= gmax;
            const bool do2 = ktile + 32 <= gmax;
            const bool do3 = ktile + 48 <= gmax;
            float8_t sacc0 =
                (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            float8_t sacc1 =
                (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            float8_t sacc2 =
                (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            float8_t sacc3 =
                (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            if (do0 || do1 || do2 || do3) {
                // QK as S^T = K*Q^T: A = K fragment (row ml = key),
                // B = Q fragment loaded once per chunk (row ml = query).
                // acc[j] = S[query=ml][key=8*kg+j].
                const unsigned qb = qq_base;
                // Unrolled by 4: each group's Q loads issue once, then each
                // live subtile's K loads + WMMAs consume the shared fragments
                // (software pipelining for LDS/L1 latency). One scheduling
                // barrier per group bounds the in-flight fragments.
#pragma unroll
                for (int dg = 0; dg < 4; ++dg) {
                    half8_t qfa[4];
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const int dc = dg * 4 + u;
                        // One aligned 16-byte half8 load per 8 dims, no cvt:
                        // the pre-convert wrote f16 with the identical cast
                        // the old f32 path applied per fragment, so the WMMA
                        // B operand is bit-identical. qd[kg] selects this
                        // lane's 8 dims (dc*16+kg*8..+8).
                        const half8_t* qd =
                            (const half8_t*)(q16 + qb + (unsigned)(dc * 16));
                        half8_t qf = {0, 0, 0, 0, 0, 0, 0, 0};
                        if (qok_ml)
                            qf = qd[kg];
                        qfa[u] = qf;
                    }
                    if (do0) {
                        half8_t kfa[4];
                        const half8_t* Kf = (const half8_t*)K16;
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int dc = dg * 4 + u;
                            // One aligned half8 load = Kfrag(sub=0, dc,
                            // lane): K[key=ml][dim=dc*16+kg*8..+8].
                            kfa[u] = Kf[fa2_kfrag8(0, dc, lane)];
                        }
#pragma unroll
                        for (int u = 0; u < 4; ++u)
                            sacc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                kfa[u], qfa[u], sacc0);
                    }
                    if (do1) {
                        half8_t kfa[4];
                        const half8_t* Kf = (const half8_t*)K16;
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int dc = dg * 4 + u;
                            // One aligned half8 load = Kfrag(sub=1, dc,
                            // lane): K[key=16+ml][dim=dc*16+kg*8..+8].
                            kfa[u] = Kf[fa2_kfrag8(1, dc, lane)];
                        }
#pragma unroll
                        for (int u = 0; u < 4; ++u)
                            sacc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                kfa[u], qfa[u], sacc1);
                    }
                    if (do2) {
                        half8_t kfa[4];
                        const half8_t* Kf = (const half8_t*)K16;
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int dc = dg * 4 + u;
                            // One aligned half8 load = Kfrag(sub=2, dc,
                            // lane): K[key=32+ml][dim=dc*16+kg*8..+8].
                            kfa[u] = Kf[fa2_kfrag8(2, dc, lane)];
                        }
#pragma unroll
                        for (int u = 0; u < 4; ++u)
                            sacc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                kfa[u], qfa[u], sacc2);
                    }
                    if (do3) {
                        half8_t kfa[4];
                        const half8_t* Kf = (const half8_t*)K16;
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int dc = dg * 4 + u;
                            // One aligned half8 load = Kfrag(sub=3, dc,
                            // lane): K[key=48+ml][dim=dc*16+kg*8..+8].
                            kfa[u] = Kf[fa2_kfrag8(3, dc, lane)];
                        }
#pragma unroll
                        for (int u = 0; u < 4; ++u)
                            sacc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                kfa[u], qfa[u], sacc3);
                    }
                    asm volatile("" ::: "memory");
                }
            }
#pragma unroll 1
            for (int sub = 0; sub < 4; ++sub) {
                const bool dos = (sub == 0) ? do0
                    : (sub == 1) ? do1 : (sub == 2) ? do2 : do3;
                if (!dos)
                    continue;
                float8_t sacc = (sub == 0) ? sacc0
                    : (sub == 1) ? sacc1 : (sub == 2) ? sacc2 : sacc3;
                const int kt_sub = ktile + sub * 16;
                const bool full = (kt_sub + 15) <= gmin;
                // This lane's softmax query is ml; its position is loaded
                // once per subtile (masked path only).
                const int pos_ml =
                    (!full && qok_ml) ? positions[qr_ml] : 0;

                // Mask + row max. The mx chain serializes the unrolled
                // rows; the other key half is combined with one shuffle.
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int key_g = kt_sub + 8 * kg + j;
                    const bool ok = qok_ml && (full || key_g <= pos_ml);
                    const float x = ok ? sacc[j] * scale_attn : -INFINITY;
                    // NOTE: max-accumulation happens in the second pass
                    // below; this pass only masks into sc.
                    sacc[j] = x;
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

                // Probabilities pack straight into the PV B operand.
                // The sum chain likewise serializes rows.
                half8_t prow;
                float sum = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float e = (m_new == -INFINITY)
                        ? 0.0f
                        : __expf(sacc[j] - m_new);
                    prow[j] = (_Float16)e;
                    sum += e;
                }
                sum += __shfl_xor(sum, 16, 32);
                l_val = l_val * alpha + sum;

                // O rescale with the lane's OWN alpha: Ofr rows are the
                // lane's own softmax query (O[query=ml][dim]).
#pragma unroll
                for (int dc = 0; dc < 16; ++dc)
                    Ofr[dc] *= alpha;

                // PV as O^T = V^T*P^T: A = V^T fragment (row ml = dim of
                // the dc tile), B = prow. Ofr[dc][j] = O[query][dim].
                // One aligned half8 load = Vfrag(sub, dc, lane):
                // V[key=sub*16+kg*8..+8][dim=dc*16+ml].
                const half8_t* Vf = (const half8_t*)V16;
#pragma unroll
                for (int dc = 0; dc < 16; ++dc) {
                    const half8_t vf = Vf[fa2_vfrag8(sub, dc, lane)];
                    Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        vf, prow, Ofr[dc]);
                    // Bound V-load hoisting to this chunk.
                    asm volatile("" ::: "memory");
                }
            }
        }
        // Tile turnover: no wave may overwrite LDS before every PV read
        // has finished.
        __syncthreads();
    }

    // ---- Transition 4: completion ----
    // Ofr[dc][j] at lane (ml,kg) = O[query=ml][dim=dc*16+8*kg+j]: own
    // query rows, 8 consecutive dims per dc. Own l_val normalizes.
    if (compute) {
        if (!PARTIAL) {
            if (qok_ml) {
                const float inv = l_val > 0.0f ? 1.0f / l_val : 0.0f;
                // 32-bit row base (whole O tensor < 10 MiB).
                const unsigned bo =
                    (unsigned)qr_ml * (unsigned)LAB_FA_QSTRIDE
                    + (unsigned)h_ml * 256u;
#pragma unroll
                for (int dc = 0; dc < 16; ++dc) {
                    const unsigned bd = bo + (unsigned)(dc * 16 + kg * 8);
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        out[bd + (unsigned)j] = Ofr[dc][j] * inv;
                }
            }
        } else {
            // CK-style split-KV contract. Keep LSE and Oacc in separate
            // logical planes so the combine kernel has unambiguous strides:
            //   lse[record]
            //   oacc[record][256] (normalized partial output)
            // record = ((query * H + head) * n_splits + split).
            const unsigned records =
                (unsigned)batch_size * (unsigned)LAB_FA_HEADS
                    * (unsigned)n_splits;
            float* lse = out;
            float* oacc = out + records;
            const unsigned record =
                ((unsigned)qr_ml * (unsigned)LAB_FA_HEADS + (unsigned)h_ml)
                    * (unsigned)n_splits + (unsigned)split;
            const float inv = l_val > 0.0f ? 1.0f / l_val : 0.0f;
            if (kg == 0) {
                if (qok_ml) {
                    lse[record] = l_val > 0.0f
                        ? m_old + __logf(l_val)
                        : -INFINITY;
                }
            }
            if (qok_ml) {
                float* rec = oacc + record * 256u;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const unsigned rj = (unsigned)(kg * 8 + j);
#pragma unroll
                    for (int dc = 0; dc < 16; ++dc)
                        rec[rj + (unsigned)(dc * 16)] = Ofr[dc][j] * inv;
                }
            }
        }
    }
}

// ---- Direct entry ----
// Grid [ceil(batch/8), LAB_FA_KV_HEADS, 1], block LAB_FA_BLOCK, dynamic LDS
// 65536. Gates on the compiled geometry (n_heads/n_kv_heads/head_dim must
// equal LAB_FA_HEADS/LAB_FA_KV_HEADS/256).
extern "C" __global__ __launch_bounds__(LAB_FA_BLOCK, 1) void attention_q8_0_fa2_gqa_gfx1201(
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
    if (n_heads != LAB_FA_HEADS || n_kv_heads != LAB_FA_KV_HEADS || head_dim != 256)
        return;
    const int kv_h = blockIdx.y;
    if (kv_h >= LAB_FA_KV_HEADS)
        return;
    const int q_base = blockIdx.x * 8;
    if (q_base >= batch_size)
        return;
    fa2_gqa_body<false>(q16, k_cache, v_cache, out, positions, batch_size,
                        scale_attn, kv_h, q_base, 0, 0x3fffffff, 0, 1);
}

// ---- Q pre-convert (f32 -> f16) ----
// Reads f32 Q row-major [batch, H, 256] and writes f16 scratch with the same
// shape using the identical `(_Float16)` cast (RNE). One wave32 per Q row,
// 4 rows per 128-thread block (independent of GQA/LAB_FA_BLOCK):
// grid [ceil(batch*H/4), 1, 1], block 128, no LDS. signs1/signs2/do_rotate
// are ignored (KMODE=0 has no FWHT); kept for ABI parity with production.
extern "C" __global__ __launch_bounds__(128, 1) void attention_fa2_q_preconvert_gfx1201(
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
    if (r >= batch_size * LAB_FA_HEADS)
        return;
    const int qr = r / LAB_FA_HEADS;
    const int h = r % LAB_FA_HEADS;
    const float* rp = q + (unsigned)qr * (unsigned)LAB_FA_QSTRIDE + (unsigned)h * 256u;
    _Float16* wp = q16 + (unsigned)qr * (unsigned)LAB_FA_QSTRIDE + (unsigned)h * 256u;
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

// ---- Partial entry (SPLIT_LSE records) ----
// Grid [ceil(batch/8), LAB_FA_KV_HEADS, n_splits], block LAB_FA_BLOCK,
// dynamic LDS 65536. partials: lse[batch*H*n_splits] then
// oacc[batch*H*n_splits][256] f32.
extern "C" __global__ __launch_bounds__(LAB_FA_BLOCK, 1) void attention_q8_0_fa2_gqa_partial_gfx1201(
    const _Float16* __restrict__ q16,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int batch_size,
    float scale_attn,
    int n_splits)
{
    if (n_heads != LAB_FA_HEADS || n_kv_heads != LAB_FA_KV_HEADS || head_dim != 256)
        return;
    if (n_splits < 1 || n_splits > 8)
        return;
    const int kv_h = blockIdx.y;
    if (kv_h >= LAB_FA_KV_HEADS)
        return;
    const int q_base = blockIdx.x * 8;
    if (q_base >= batch_size)
        return;
    const int split = blockIdx.z;
    if (split >= n_splits)
        return;
    // Each split derives its tile range from the live seq_len inside the
    // body (SPLIT_LSE) and writes LSE + normalized O.
    fa2_gqa_body<true, true>(q16, k_cache, v_cache, partials, positions,
                             batch_size, scale_attn, kv_h, q_base, 0, 0,
                             split, n_splits);
}

// ---- Merge entry (SPLIT_LSE records) ----
// Grid [ceil(batch*H/8), 1, 1], block 256 (one wave32 per (query, head) rec),
// no LDS. Stable max-LSE combine; a split with no live tiles carries
// LSE=-inf and contributes nothing.
extern "C" __global__ __launch_bounds__(256) void attention_q8_0_fa2_gqa_merge_gfx1201(
    const float* __restrict__ partials,
    float* __restrict__ out,
    int batch_size,
    int n_heads,
    int head_dim,
    int n_splits)
{
    if (n_heads != LAB_FA_HEADS || head_dim != 256)
        return;
    if (n_splits < 1 || n_splits > 8)
        return;
    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int rec = blockIdx.x * 8 + wave;
    if (rec >= batch_size * n_heads)
        return;
    const long long records =
        (long long)batch_size * n_heads * n_splits;
    const float* lse = partials;
    const float* oacc = partials + records;
    float m = -INFINITY;
    for (int s = 0; s < n_splits; ++s)
        m = fmaxf(m, lse[(long long)rec * n_splits + s]);
    for (int d = lane; d < head_dim; d += 32) {
        float num = 0.0f;
        float den = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            const long long record = (long long)rec * n_splits + s;
            const float split_lse = lse[record];
            const float w = isfinite(split_lse) ? __expf(split_lse - m) : 0.0f;
            den += w;
            num += w * oacc[record * head_dim + d];
        }
        out[(long long)rec * head_dim + d] =
            (den > 0.0f) ? num / den : 0.0f;
    }
}
