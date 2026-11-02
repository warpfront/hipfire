// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Shared-K/V split-KV sparse-attention (QSA) decode, Q8_0 K/V cache,
// head_dim 256, GQA group 12 (n_heads == 12 * n_kv_heads), 1..=32 query rows (grid z = rows).
//
// Provenance: the q8 subset of kernels/src/qsa_decode_splitkv.hip at commit
// f1771197a (branch perf/qsa-split-attend-043). `qsa_decode_splitkv_partial_q8`
// and `qsa_decode_splitkv_merge_f32` and every helper, macro and constant they
// use are copied verbatim (same arithmetic, same names); the F32-cache partial
// kernel `qsa_decode_splitkv_partial_f32` and the helpers only it uses are not
// carried: F32 K/V keeps `qsa_decode_splitk.hip`. q8 only. KLD-gated, not
// bit-exact against the per-head exact route (dense-style KT64 tile schedule,
// per-tile online softmax, half-dot reduction order).
//
// Design (port of the INTERNAL DESIGN of the dense
// attention_q8_0_fa2_gqa.gfx1201.hip, whose source is untouched):
//   * one workgroup owns one KV head and loads every K and V element ONCE
//     per KT64 key tile, then reuses it for all 12 query heads of the group
//     (one head per wave32);
//   * online softmax (m, l) and the O accumulator stay in VGPRs; K/V tiles
//     stream through LDS;
//   * split-KV partial + merge with the dense product LSE ABI: the caller
//     owns one allocation with F32 lse[records] FIRST and then normalized
//     F32 oacc[records][256], record = ((row * n_heads + head) * n_splits +
//     split), records = rows * n_heads * n_splits. An empty split carries
//     LSE = -inf and O = 0. The merge is the dense combine: ascending
//     max-LSE, __expf weights, sum(w * O) / sum(w).
//   * split boundaries are the dense ones: live tiles = ceil(len / 64),
//     tiles_per_split = ceil(live / n_splits), split s owns tiles
//     [s * tps, min((s + 1) * tps, live)) (len replaces dense seq_len).
//   * The key set is the SELECTED list: a "key" is selected rank r of this
//     row; its cache token is selected[row * capacity + r]. A token that is
//     < 0, >= visible or >= full_capacity is invalid: score -inf, weight 0,
//     K and V rows are zero-filled in LDS. visible = position_start + row + 1;
//     len = min(capacity, min(budget_blocks, visible / compress) * compress
//                          + visible % compress).
//   * Pure FP32 FMA math (no WMMA): Q8_0 K/V is dequantized in registers into
//     one 32 KiB F32 LDS bank that the 64-key tile streams through in four
//     passes (K channels 0..127, K 128..255, V 0..127, V 128..255). Lane
//     mapping: wave = query head, lane owns channels 4*lane..+3 and
//     128+4*lane..+3 for its Q and O registers.
//   * QK per key: each of the 32 lanes computes a 4-product FP32 chain over
//     its channels of the current 128-channel half, the 32 lanes reduce it
//     with an xor-16/8/4/2/1 add tree and lane 0 stores it to the wave's LDS
//     score plane; the second half adds its half-sum there. This is NOT the
//     serial d-order dot of the scalar QSA reference, so scores differ in the
//     last bits; agreement is only established at the gated-output oracle.
//   * Online softmax is per KT64 tile; P stays FP32 and is exchanged through
//     LDS, PV runs over selected ranks in ascending order with plain FP32
//     accumulation. Score scale is 1/sqrt(256) = 1/16, exact in FP32.
//   * The merge is the dense merge loop (max-LSE, per-channel ascending-split
//     den/num with isfinite/__expf weights) with the split guard (<= 32) and
//     the QSA gate epilogue num / den / (1 + expf(-gate)); a record with
//     den <= 0 outputs 0.
//
// Partial kernel  qsa_decode_splitkv_partial_q8
//   grid [n_kv_heads, n_splits, rows], block [384, 1, 1] (12 wave32s),
//   dynamic LDS 36096 B = 64*128*4 (KV bank) + 12*64*4 (weights)
//   + 64*4 (tokens); no static LDS. n_splits 1..32 (production 32).
//   Every __syncthreads() is block-uniform: all control flow that encloses a
//   barrier depends only on kernel arguments and blockIdx. 8 barriers per
//   KT64 tile: (1) K0 bank stored, (2) K0 read, (3) K1 bank stored, (4) K1
//   read + scores visible to all lanes of each wave, (5) V0 bank stored +
//   weights published, (6) V0 read, (7) V1 bank stored, (8) V1 read (tile
//   turnover). Q8_0 K/V (34 B blocks, see QsaSkvQ8) is fetched once per half
//   tile (each cache byte and block scale loaded by exactly one lane, aligned
//   b32 loads) and dequantized in registers into the F32 LDS bank. Staging
//   registers: 10 per thread (8 dwords + dz + token).
// Merge kernel  qsa_decode_splitkv_merge_f32
//   grid [ceil(rows * n_heads / 8), 1, 1], block 256 (one wave32 per
//   record), no LDS.

#include <hip/hip_runtime.h>

#define QSA_SKV_HD 256
#define QSA_SKV_GROUP 12
#define QSA_SKV_KT 64
#define QSA_SKV_THREADS 384
#define QSA_SKV_MAX_SPLITS 32

// LDS layout, float units: bank [0, 8192), weights [8192, 8960),
// tokens [8960, 9024) -> 9024 * 4 = 36096 bytes.
#define QSA_SKV_W_OFF 8192
#define QSA_SKV_T_OFF (8192 + QSA_SKV_GROUP * QSA_SKV_KT)

// Partial q . k over one 128-channel half for all 64 keys of the tile. Per
// key every lane forms a 4-product FP32 chain over its channels, the 32 lanes
// reduce it with an xor-16/8/4/2/1 add tree, and lane 0 stores it into the
// wave's LDS score plane: FIRST stores the half-0 sum, the second half stores
// (half-0 sum) + (half-1 sum). Keys are independent, so the loop is unrolled
// by 4 for shuffle ILP only; there is no private score array.
template <bool FIRST>
static __device__ __forceinline__ void qsa_skv_qk_half(
    float* __restrict__ head_scores, const float4* __restrict__ bank4, float4 q, int lane) {
#pragma unroll 4
    for (int k = 0; k < QSA_SKV_KT; ++k) {
        const float4 kv = bank4[k * 32 + lane];
        float a = q.x * kv.x;
        a += q.y * kv.y;
        a += q.z * kv.z;
        a += q.w * kv.w;
        a += __shfl_xor(a, 16, 32);
        a += __shfl_xor(a, 8, 32);
        a += __shfl_xor(a, 4, 32);
        a += __shfl_xor(a, 2, 32);
        a += __shfl_xor(a, 1, 32);
        if (lane == 0) {
            if constexpr (FIRST) {
                head_scores[k] = a;
            } else {
                head_scores[k] = head_scores[k] + a;
            }
        }
    }
}

// P*V over one 128-channel half for all 64 keys, ascending selected rank.
// Four keys per iteration; the loop is not unrolled and the compiler barrier
// bounds the V/weight load lifetime to one iteration (o is a scalar-indexed
// float4, no private array).
static __device__ __forceinline__ void qsa_skv_pv_half(
    float4& o, const float4* __restrict__ bank4, const float4* __restrict__ w4, int lane) {
#pragma unroll 1
    for (int kk = 0; kk < QSA_SKV_KT / 4; ++kk) {
        const float4 w = w4[kk];
        const float4 v0 = bank4[(4 * kk + 0) * 32 + lane];
        const float4 v1 = bank4[(4 * kk + 1) * 32 + lane];
        const float4 v2 = bank4[(4 * kk + 2) * 32 + lane];
        const float4 v3 = bank4[(4 * kk + 3) * 32 + lane];
        o.x += w.x * v0.x; o.y += w.x * v0.y; o.z += w.x * v0.z; o.w += w.x * v0.w;
        o.x += w.y * v1.x; o.y += w.y * v1.y; o.z += w.y * v1.z; o.w += w.y * v1.w;
        o.x += w.z * v2.x; o.y += w.z * v2.y; o.z += w.z * v2.z; o.w += w.z * v2.w;
        o.x += w.w * v3.x; o.y += w.w * v3.y; o.z += w.w * v3.z; o.w += w.w * v3.w;
        asm volatile("" ::: "memory");
    }
}

// ---------------------------------------------------------------------------
// Q8_0 K/V cache format. The partial-kernel body below is format-generic; the
// format supplies the staging registers, the global fetch of one 64-key x
// 128-channel half tile (K or V) and the commit of that half tile to the
// shared F32 LDS bank (layout: key * 128 + channel, float4 columns, exactly
// what qsa_skv_qk_half / qsa_skv_pv_half read). `half` is 0 (channels 0..127)
// or 1 (channels 128..255). Everything after the commit is format-blind.
// ---------------------------------------------------------------------------

// Q8_0 cache [token][n_kv_heads][8 blocks][34 B]: block = f16 scale (2 B)
// then 32 int8 codes; head row 272 B, token row n_kv_heads * 272 B (544 B for
// the 2-KV-head model). value = f32(scale) * code (exact in FP32: 11-bit
// scale mantissa x 8-bit code).
//
// Fetch geometry. Channels 0..127 are blocks 0..3 = head-row bytes [0, 136),
// channels 128..255 blocks 4..7 = bytes [136, 272). A block PAIR is 68 B =
// 17 dwords and starts at 68 * k, i.e. 4 B aligned, so a pair can be read
// with plain aligned b32 loads and no byte is shared with a neighbouring
// pair. One half tile = 64 keys x 2 pairs = 128 pairs; each pair is read by
// two lanes, one per block ("A" = even block, "B" = odd block of the pair):
//   lane tid < 256: key = tid >> 2, blk = tid & 3 (block within the half;
//   pair = blk >> 1, role = blk & 1; channels 32 * blk .. +31).
//   A (role 0) reads pair dwords 0..8  (9 loads): dword 0 = scale0 | codes
//     0,1; dwords 1..7 = codes 2..29; dword 8 = codes 30,31 | scale1.
//   B (role 1) reads pair dwords 9..16 (8 loads) = the 32 codes of its block,
//     dword-aligned.
// Every byte of the half tile (64 * 136 B = 8704 B = 128 * 9 * 4 + 128 * 8 *
// 4) is loaded by exactly one lane, once; each scale is part of exactly one
// lane's dwords (A's dword 0 / dword 8) and reaches its other consumer (the
// B lane, scale1) by one xor-1 lane shuffle, never a reload. Lanes 256..383
// load nothing (the fetch is already 8-9 independent b32 loads per lane).
// Dequantization is in registers (a code quad = one v_alignbyte of two
// adjacent dwords for A, the dword itself for B), then the lane commits its
// 8 float4 (32 channels) to the F32 bank. A lane processes its 8 quads in the
// rotated order j = (t + tid) & 7 (loads and stores alike), so the 8 lanes
// of a ds_write_b128 group hit 8 distinct 16 B bank groups instead of one.
//
// Requirement: the cache base is 4 B aligned (the per-head offset 272 and the
// token row 272 * n_kv_heads are).
#define QSA_SKV_Q8_HEAD_BYTES 272
#define QSA_SKV_Q8_HALF_BYTES 136
#define QSA_SKV_Q8_PAIR_BYTES 68
#define QSA_SKV_Q8_LOADERS (QSA_SKV_KT * 4) /* 256 lanes (waves 0..7) */

static __device__ __forceinline__ float qsa_skv_h2f(uint32_t u) {
    return (float)__builtin_bit_cast(_Float16, (unsigned short)u);
}

struct QsaSkvQ8 {
    struct Stage {
        // A: e[t] = pair dword 1 + j(t); B: e[t] = pair dword 9 + j(t).
        uint32_t e[8];
        uint32_t dz;  // A only: pair dword 0 (scale0 | codes 0,1)
        int tk;       // validated token of this lane's key (-1 invalid)
    };
    typedef const unsigned char* Ptr;
    static __device__ __forceinline__ Ptr head(const void* base, int kv_head) {
        return reinterpret_cast<const unsigned char*>(base) +
               (size_t)kv_head * QSA_SKV_Q8_HEAD_BYTES;
    }
    static __device__ __forceinline__ size_t row_stride(int n_kv_heads) {
        return (size_t)n_kv_heads * QSA_SKV_Q8_HEAD_BYTES;
    }

    // An invalid / out-of-range key (and the idle lanes) stage zeros: scale 0,
    // codes 0 -> a zero bank row.
    template <bool FROM_LDS>
    static __device__ __forceinline__ void issue(
        Stage& st, Ptr cache, const int* __restrict__ sel_row, const int* lds_tokens, int rank0,
        int len, int visible, int full_capacity, size_t stride, int half) {
        const int tid = (int)threadIdx.x;
        int t = -1;
        st.dz = 0u;
#pragma unroll
        for (int i = 0; i < 8; ++i) st.e[i] = 0u;
        if (tid < QSA_SKV_Q8_LOADERS) {
            const int key = tid >> 2;
            if constexpr (FROM_LDS) {
                t = lds_tokens[key];
            } else {
                const int rank = rank0 + key;
                if (rank < len) {
                    const int s = sel_row[rank];
                    if (s >= 0 && s < full_capacity && s < visible) t = s;
                }
            }
            if (t >= 0) {
                const int blk = tid & 3;
                const int role = blk & 1;
                const unsigned char* pbase = cache + (size_t)t * stride +
                                             half * QSA_SKV_Q8_HALF_BYTES +
                                             (blk >> 1) * QSA_SKV_Q8_PAIR_BYTES;
                // A: dwords 1..8 (byte 4 + 4j), B: dwords 9..16 (byte 36 + 4j).
                const unsigned char* ebase = pbase + (role ? 36 : 4);
                const int rot = tid & 7;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int j = (i + rot) & 7;
                    st.e[i] = *reinterpret_cast<const uint32_t*>(ebase + 4 * j);
                }
                if (role == 0) st.dz = *reinterpret_cast<const uint32_t*>(pbase);
            }
        }
        st.tk = t;
    }

    // Dequantize the staged dwords into the F32 bank. WRITE_TOKENS (K pass 0
    // only) publishes the validated token of each key (-1 invalid); one
    // writer per key (blk 0). All shuffles are inside wave-uniform control
    // flow (tid < 256 = whole waves 0..7).
    template <bool WRITE_TOKENS>
    static __device__ __forceinline__ void store(float4* __restrict__ bank4,
                                                 int* __restrict__ lds_tokens, const Stage& st) {
        const int tid = (int)threadIdx.x;
        if (tid < QSA_SKV_Q8_LOADERS) {
            const int key = tid >> 2;
            const int blk = tid & 3;
            const int role = blk & 1;
            const int rot = tid & 7;
            if constexpr (WRITE_TOKENS) {
                if (blk == 0) lds_tokens[key] = st.tk;
            }
            // scale1 = high half of pair dword 8 = e[t] at j(t) == 7 (A lanes);
            // the B lane (xor 1) takes it from its A partner.
            uint32_t s1x = 0u;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                s1x = (((i + rot) & 7) == 7) ? (st.e[i] >> 16) : s1x;
            }
            const uint32_t s1 = (uint32_t)__shfl_xor((int)s1x, 1, 32);
            const float s = qsa_skv_h2f(role ? s1 : (st.dz & 0xffffu));
            float4* out = bank4 + key * 32 + blk * 8;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int j = (i + rot) & 7;
                const uint32_t cur = st.e[i];
                // A: quad j = bytes 4j+2..4j+5 of the A stream = funnel of
                // dword j (lo) and dword j + 1 (= cur). B: the dword itself.
                const uint32_t prev = (j == 0) ? st.dz : st.e[(i + 7) & 7];
                const uint32_t lo = role ? cur : prev;
                const uint32_t w = __builtin_amdgcn_alignbyte(cur, lo, role ? 0u : 2u);
                out[j] = make_float4((float)(int8_t)w * s, (float)(int8_t)(w >> 8) * s,
                                     (float)(int8_t)(w >> 16) * s, (float)(int8_t)(w >> 24) * s);
            }
        }
    }
};

// Dynamic LDS: one extern array for every entry point (no static LDS).
extern __shared__ float4 qsa_skv_lds4[];

// Shared partial body. Fmt = QsaSkvQ8 (Q8_0 blocks); the cache pointers are
// untyped (the kernarg layout of the f1771197a partial kernels).
template <class Fmt>
static __device__ __forceinline__ void qsa_skv_partial_body(
    const float* __restrict__ q_with_gate,
    const void* __restrict__ full_keys,
    const void* __restrict__ full_values,
    const int* __restrict__ selected,
    float* __restrict__ partials,
    int rows,
    int position_start,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int budget_blocks,
    int compress,
    int capacity,
    int full_capacity,
    int n_splits) {
    float* lds = reinterpret_cast<float*>(qsa_skv_lds4);
    float4* bank4 = qsa_skv_lds4;
    const float4* w4_all = reinterpret_cast<const float4*>(lds + QSA_SKV_W_OFF);
    int* tokens = reinterpret_cast<int*>(lds + QSA_SKV_T_OFF);

    // Block-uniform argument/grid guards: no barrier has been issued yet.
    if (head_dim != QSA_SKV_HD || n_kv_heads < 1 || n_heads != n_kv_heads * QSA_SKV_GROUP) return;
    if (n_splits < 1 || n_splits > QSA_SKV_MAX_SPLITS || compress < 1) return;
    if ((int)blockDim.x != QSA_SKV_THREADS) return;
    const int kv_head = (int)blockIdx.x;
    const int split = (int)blockIdx.y;
    const int row = (int)blockIdx.z;
    if (kv_head >= n_kv_heads || split >= n_splits || row >= rows) return;

    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;  // query head within the group
    const int lane = tid & 31;
    const int head = kv_head * QSA_SKV_GROUP + wave;

    // Selected-list length, validity window and dense-style tile split.
    const int visible = position_start + row + 1;
    const int block_count = visible / compress;
    const int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    int len = chosen * compress + (visible - block_count * compress);
    if (len > capacity) len = capacity;
    if (len < 0) len = 0;
    const int tiles_total = (len + QSA_SKV_KT - 1) / QSA_SKV_KT;
    const int tiles_per_split = (tiles_total + n_splits - 1) / n_splits;
    const int t_begin = split * tiles_per_split;
    const int t_end = min(t_begin + tiles_per_split, tiles_total);

    const size_t kv_stride = Fmt::row_stride(n_kv_heads);
    const typename Fmt::Ptr keys = Fmt::head(full_keys, kv_head);
    const typename Fmt::Ptr values = Fmt::head(full_values, kv_head);
    const int* sel_row = selected + (size_t)row * capacity;

    // Q: lane owns channels 4*lane..+3 (qa) and 128+4*lane..+3 (qb).
    const float4* q4 = reinterpret_cast<const float4*>(
        q_with_gate + ((size_t)row * 2 * n_heads + (size_t)head * 2) * QSA_SKV_HD);
    const float4 qa = q4[lane];
    const float4 qb = q4[lane + 32];

    float4 oa = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 ob = oa;
    float m_old = -INFINITY;
    float l_val = 0.0f;

    typename Fmt::Stage st;
    const float4* my_w4 = w4_all + wave * (QSA_SKV_KT / 4);
    // This wave's 64-entry LDS plane: reduced raw scores, then weights.
    float* my_scores = lds + QSA_SKV_W_OFF + wave * QSA_SKV_KT;

    if (t_begin < t_end) {
        Fmt::template issue<false>(st, keys, sel_row, tokens, t_begin * QSA_SKV_KT, len, visible,
                                   full_capacity, kv_stride, 0);
    }
    for (int tile = t_begin; tile < t_end; ++tile) {
        const int rank0 = tile * QSA_SKV_KT;

        // ---- K channels 0..127 ----
        Fmt::template store<true>(bank4, tokens, st);
        __syncthreads();
        Fmt::template issue<true>(st, keys, sel_row, tokens, rank0, len, visible, full_capacity,
                                  kv_stride, 1);
        qsa_skv_qk_half<true>(my_scores, bank4, qa, lane);
        __syncthreads();

        // ---- K channels 128..255 ----
        Fmt::template store<false>(bank4, tokens, st);
        __syncthreads();
        Fmt::template issue<true>(st, values, sel_row, tokens, rank0, len, visible, full_capacity,
                                  kv_stride, 0);
        qsa_skv_qk_half<false>(my_scores, bank4, qb, lane);
        // All K bank reads are done and every wave's reduced scores are
        // visible to all of its lanes once this barrier passes.
        __syncthreads();

        // ---- V channels 0..127 (bank commit first, then online softmax) ----
        Fmt::template store<false>(bank4, tokens, st);

        // Online softmax (this wave only): lane L owns keys 2L and 2L+1; the
        // raw scores are replaced in place by the weights.
        const int2 tok2 = *reinterpret_cast<const int2*>(tokens + 2 * lane);
        const float2 sc = *reinterpret_cast<const float2*>(my_scores + 2 * lane);
        // 1/sqrt(256) = 0.0625 exactly; invalid keys are masked to -inf.
        const float s0 = tok2.x >= 0 ? sc.x * 0.0625f : -INFINITY;
        const float s1 = tok2.y >= 0 ? sc.y * 0.0625f : -INFINITY;
        float mx = fmaxf(s0, s1);
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        mx = fmaxf(mx, __shfl_xor(mx, 8, 32));
        mx = fmaxf(mx, __shfl_xor(mx, 4, 32));
        mx = fmaxf(mx, __shfl_xor(mx, 2, 32));
        mx = fmaxf(mx, __shfl_xor(mx, 1, 32));
        const float m_new = fmaxf(m_old, mx);
        const float alpha = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
        m_old = m_new;
        const float e0 = (m_new == -INFINITY) ? 0.0f : __expf(s0 - m_new);
        const float e1 = (m_new == -INFINITY) ? 0.0f : __expf(s1 - m_new);
        *reinterpret_cast<float2*>(my_scores + 2 * lane) = make_float2(e0, e1);
        float sum = e0 + e1;
        sum += __shfl_xor(sum, 16, 32);
        sum += __shfl_xor(sum, 8, 32);
        sum += __shfl_xor(sum, 4, 32);
        sum += __shfl_xor(sum, 2, 32);
        sum += __shfl_xor(sum, 1, 32);
        l_val = l_val * alpha + sum;
        oa.x *= alpha; oa.y *= alpha; oa.z *= alpha; oa.w *= alpha;
        ob.x *= alpha; ob.y *= alpha; ob.z *= alpha; ob.w *= alpha;
        // V0 bank is committed and every wave's weights are published.
        __syncthreads();

        // ---- V channels 0..127 ----
        Fmt::template issue<true>(st, values, sel_row, tokens, rank0, len, visible, full_capacity,
                                  kv_stride, 1);
        qsa_skv_pv_half(oa, bank4, my_w4, lane);
        __syncthreads();

        // ---- V channels 128..255 ----
        Fmt::template store<false>(bank4, tokens, st);
        __syncthreads();
        if (tile + 1 < t_end) {
            Fmt::template issue<false>(st, keys, sel_row, tokens, rank0 + QSA_SKV_KT, len, visible,
                                       full_capacity, kv_stride, 0);
        }
        qsa_skv_pv_half(ob, bank4, my_w4, lane);
        // Tile turnover: no wave may overwrite the bank before every PV read
        // has finished.
        __syncthreads();
    }

    // ---- Completion: dense CK-style record (normalized O + LSE) ----
    const size_t records = (size_t)rows * n_heads * n_splits;
    float* lse = partials;
    float* oacc = partials + records;
    const size_t record = ((size_t)row * n_heads + head) * n_splits + split;
    const float inv = l_val > 0.0f ? 1.0f / l_val : 0.0f;
    if (lane == 0) lse[record] = l_val > 0.0f ? m_old + __logf(l_val) : -INFINITY;
    float4* rec4 = reinterpret_cast<float4*>(oacc + record * QSA_SKV_HD);
    rec4[lane] = make_float4(oa.x * inv, oa.y * inv, oa.z * inv, oa.w * inv);
    rec4[lane + 32] = make_float4(ob.x * inv, ob.y * inv, ob.z * inv, ob.w * inv);
}

// Q8_0 cache twin: identical grid/block/LDS contract and partial-record ABI;
// full_keys / full_values point at [token][n_kv_heads][8][34 B] Q8_0 blocks.
extern "C" __global__ __launch_bounds__(QSA_SKV_THREADS) void qsa_decode_splitkv_partial_q8(
    const float* __restrict__ q_with_gate,
    const unsigned char* __restrict__ full_keys,
    const unsigned char* __restrict__ full_values,
    const int* __restrict__ selected,
    float* __restrict__ partials,
    int rows,
    int position_start,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int budget_blocks,
    int compress,
    int capacity,
    int full_capacity,
    int n_splits) {
    qsa_skv_partial_body<QsaSkvQ8>(q_with_gate, full_keys, full_values, selected, partials, rows,
                                   position_start, n_heads, n_kv_heads, head_dim, budget_blocks,
                                   compress, capacity, full_capacity, n_splits);
}

// Merge: the dense product LSE merge (attention_q8_0_fa2_gqa_merge_gfx1201,
// SPLIT_LSE records) verbatim — same per-record max-LSE loop, same
// per-channel ascending-split den/num loop with isfinite/__expf weights and
// `den > 0 ? num / den : 0` — differing only in the split guard (<= 32) and
// the QSA sigmoid gate epilogue, applied to the normalized result with the
// QSA association num / den / (1 + expf(-gate)). One wave32 per
// (row, head) record.
extern "C" __global__ __launch_bounds__(256) void qsa_decode_splitkv_merge_f32(
    const float* __restrict__ partials,
    const float* __restrict__ q_with_gate,
    float* __restrict__ out,
    int rows,
    int n_heads,
    int head_dim,
    int n_splits) {
    if (head_dim != QSA_SKV_HD || n_heads < 1) return;
    if (n_splits < 1 || n_splits > QSA_SKV_MAX_SPLITS) return;
    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int rec = (int)blockIdx.x * 8 + wave;
    if (rec >= rows * n_heads) return;
    const long long records =
        (long long)rows * n_heads * n_splits;
    const float* lse = partials;
    const float* oacc = partials + records;
    float m = -INFINITY;
    for (int s = 0; s < n_splits; ++s)
        m = fmaxf(m, lse[(long long)rec * n_splits + s]);
    const int row = rec / n_heads;
    const int head = rec - row * n_heads;
    const float* gate =
        q_with_gate + ((size_t)row * 2 * n_heads + (size_t)head * 2) * QSA_SKV_HD + QSA_SKV_HD;
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
            (den > 0.0f) ? num / den / (1.0f + expf(-gate[d])) : 0.0f;
    }
}
