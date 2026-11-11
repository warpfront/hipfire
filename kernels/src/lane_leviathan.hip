// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// ---------------------------------------------------------------------------
// Lane-wide sampled verify (HIPFIRE_QWEN4_LANE_DEVICE_SAMPLE): ONE launch draws
// every lane's speculative-verify window with Leviathan acceptance.
//
// Grid = one 512-thread workgroup per verify row (all lanes x all rows). A row
// workgroup builds its target distribution p under the shared device sampler's
// law (`sample_top_p_parallel.hip`: blocked tokens -> -inf, then the recency
// weighted repeat / presence / frequency penalty of the row's scope, a TOP_K
// gather of G = 20 (top_k_req <= 20) or 64 candidates, softmax at the
// temperature against the maximum, the request cap and min-p cut, the nucleus
// at top_p) and publishes the nucleus `(token, unnormalized prob)` plus its
// mass. The LAST row workgroup of a lane to arrive (device-scope ticket) runs
// the lane's sequential verdict:
//
//   row j < n_drafts: u = U(rng); accept draft d iff q(d) > 0 && u*q(d) < p(d);
//     an accepted EOS draft ends the lane (no bonus); a rejection draws the
//     replacement from the normalized residual (p - q)+ with a second uniform
//     (p itself when the residual is empty). Rejection implies p(d) < q(d), so
//     the residual is zero at d; d is skipped outright, so a replacement never
//     repeats its draft whatever the rounding (callers rely on that: the
//     window ends at the first token that differs from its draft).
//   row n_drafts (every draft accepted): the bonus is drawn from p.
//
// A row with a single-token p (greedy temperature, or no finite candidate: the
// shared sampler's token-0 fallback) is decided without a uniform: accept iff
// the draft is that token, else emit it. `n_drafts == 0` is one plain draw
// from p (an AR step). q is the draft's sparse distribution as drawn (at most
// LLV_Q_MAX `(token, prob)` pairs; a deterministic draft is the point mass).
//
// Uniforms come from the shared sampler's xorshift32 (13, 17, 5) on the lane's
// u32 state. A p draw (bonus, AR) is the sampler's own inverse CDF over the
// unnormalized nucleus (u = state / 4294967295, `cum >= u * mass`, fall through
// to the last entry), so an AR step draws exactly as `sample_topk_finalize`
// does on distinct logits. The accept test and the residual use the 24-bit
// uniform (state >> 8) * 2^-24 in [0, 1), so p(d) >= q(d) always accepts.
//
// Top-G of a 248K row in one workgroup: pass 1 takes every thread's maximum
// over the row's unpenalized tokens; the G-th largest of those maxima, tau, has
// at least G tokens at or above it, so the top-G of the penalized row lies in
// { unpenalized v >= tau } U { penalized scope tokens with v >= tau }. Pass 2
// collects that set into LDS and one bitonic sort by (value desc, index asc)
// yields the top-G. A set beyond LDS capacity (massive exact ties) is reduced
// tile by tile with the same keys, so the result never depends on order.
// Logits are read only: no row is mutated.
//
// Vocab-sharded draws (X1, EP2 mesh): `mode` (kernarg) selects one of
//   mode 0  today's kernel, whole row, byte for byte (cand ptrs null, LL_BASE 0).
//   mode 1  SELECT: the row workgroup scans its row L[0..LL_VOCAB) as the shard
//           [LL_BASE, LL_BASE + LL_VOCAB) of the full vocabulary (sort keys,
//           scope and blocked lookups use global ids = index + LL_BASE; tokens
//           outside the shard are skipped), then publishes g and the g sorted
//           keys to its candidate row and returns (no p table, no ticket).
//   mode 2  MERGED: the row workgroup skips the scan, merges the local and the
//           peer candidate rows into s.cand[0..g), g = min(G, g0 + g1), and runs
//           today's p table, ticket and lane verdict (llv_tail) unchanged.
// Candidate row (LLV_CAND_WORDS u32 words, row = blockIdx.x of the launch):
//   word 0 = g, word 1 = 0, words 2.. = 64 u64 sort keys (ascending = value
//   descending, global id ascending; entries >= g are not written). The top-G of
//   a union is the top-G of the union of each part's top-G under one total
//   order, so the merged list equals the unsharded kernel's.
// ---------------------------------------------------------------------------

#define LLV_BLOCK 512
#define LLV_G_MAX 64
#define LLV_CAND_CAP 2048
#define LLV_H_CAP 1024
#define LLV_FILTER_BITS 8192
#define LLV_TILE 1024
#define LLV_MAX_ROWS 16
#define LLV_Q_MAX 64
#define LLV_NEG_FLOOR -1e29f

// Lane record (u32 words).
#define LL_ROW0 0
#define LL_NROWS 1
#define LL_RNG 2
#define LL_EOS 3
#define LL_TEMP 4
#define LL_TOP_P 5
#define LL_TOP_K 6
#define LL_MIN_P 7
#define LL_REP 8
#define LL_PRES 9
#define LL_FREQ 10
#define LL_VOCAB 11
#define LL_DRAFTS 12
#define LL_Q 13
#define LL_BLOCKED 14
#define LL_NBLOCKED 15
#define LL_OUT 16
#define LL_BASE 17
#define LLV_LANE_WORDS 20

// Row record (u32 words).
#define LR_PTR_LO 0
#define LR_PTR_HI 1
#define LR_LANE 2
#define LR_J 3
#define LR_PEN_OFF 4
#define LR_PEN_LEN 5
#define LLV_ROW_WORDS 6

// q record per draft: count, tok[LLV_Q_MAX], prob[LLV_Q_MAX].
#define LLV_Q_WORDS (1 + 2 * LLV_Q_MAX)

// p table per row: n, mass, point, point_token, tok[64], prob[64].
#define LP_N 0
#define LP_MASS 1
#define LP_POINT 2
#define LP_POINT_TOK 3
#define LP_TOK 4
#define LP_PROB (4 + LLV_G_MAX)
#define LLV_P_WORDS (4 + 2 * LLV_G_MAX)

// Out record per lane: ticket, done, accepted, n_tokens, rng, pad, pad, pad, tokens[16].
#define LO_TICKET 0
#define LO_DONE 1
#define LO_ACCEPTED 2
#define LO_NTOK 3
#define LO_RNG 4
#define LO_TOKENS 8
#define LLV_OUT_WORDS (8 + LLV_MAX_ROWS)

// Candidate row (modes 1 and 2): g, 0, then LLV_G_MAX u64 sort keys.
#define LLV_CAND_WORDS (2 + 2 * LLV_G_MAX)
#define LLV_MODE_FULL 0
#define LLV_MODE_SELECT 1
#define LLV_MODE_MERGED 2

static __device__ __forceinline__ unsigned llv_key(float v) {
    v = (v == 0.0f) ? 0.0f : v;
    unsigned b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
static __device__ __forceinline__ float llv_unkey(unsigned k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}
// Ascending order of the sort key = value descending, then index ascending.
static __device__ __forceinline__ unsigned long long llv_sortkey(float v, unsigned idx) {
    return ((unsigned long long)(~llv_key(v)) << 32) | (unsigned long long)idx;
}
static __device__ __forceinline__ float llv_sk_val(unsigned long long sk) {
    return llv_unkey(~(unsigned)(sk >> 32));
}

static __device__ void llv_bitonic(unsigned long long* a, int n, int tid) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int t = tid; t < (n >> 1); t += LLV_BLOCK) {
                int lo = ((t & ~(j - 1)) << 1) | (t & (j - 1));
                int hi = lo + j;
                unsigned long long x = a[lo];
                unsigned long long y = a[hi];
                bool asc = ((lo & k) == 0);
                if ((x > y) == asc) {
                    a[lo] = y;
                    a[hi] = x;
                }
            }
            __syncthreads();
        }
    }
}

static __device__ __forceinline__ int llv_pow2(int n) {
    int p = 2;
    while (p < n) p <<= 1;
    return p;
}

static __device__ __forceinline__ unsigned llv_xorshift(unsigned s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

struct LlvShared {
    unsigned long long cand[LLV_CAND_CAP];
    unsigned long long hkey[LLV_H_CAP];
    float hval[LLV_H_CAP];
    unsigned filter[LLV_FILTER_BITS / 32];
    int ncand;
    int nh;
    int last;
};

// Position of the run-last key of `tok` in the sorted scope keys, or -1.
static __device__ __forceinline__ int llv_h_find(const LlvShared& s, unsigned tok) {
    if (!((s.filter[(tok >> 5) & (LLV_FILTER_BITS / 32 - 1)] >> (tok & 31)) & 1u)) return -1;
    unsigned long long probe = ((unsigned long long)tok << 32) | 0xffffffffull;
    int lo = 0, hi = s.nh;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (s.hkey[mid] <= probe) lo = mid + 1; else hi = mid;
    }
    int p = lo - 1;
    return (p >= 0 && (unsigned)(s.hkey[p] >> 32) == tok) ? p : -1;
}

// p of one row into `P` (global, LLV_P_WORDS). Thread 0 only, sorted top-G in s.cand[0..g).
static __device__ void llv_build_p(const LlvShared& s, int g, float temp, float top_p, int top_k_req,
                                   float min_p, int G, float* P) {
    unsigned* Pu = (unsigned*)P;
    if (g == 0) {
        Pu[LP_N] = 0u;
        Pu[LP_POINT] = 1u;
        Pu[LP_POINT_TOK] = 0u;
        P[LP_MASS] = 0.0f;
        return;
    }
    if (temp <= 1e-6f) {
        Pu[LP_N] = 0u;
        Pu[LP_POINT] = 1u;
        Pu[LP_POINT_TOK] = (unsigned)s.cand[0];
        P[LP_MASS] = 0.0f;
        return;
    }
    float inv_temp = 1.0f / temp;
    float max_val = llv_sk_val(s.cand[0]);
    float probs[LLV_G_MAX];
#pragma unroll
    for (int k = 0; k < LLV_G_MAX; k++) {
        probs[k] = (k < g) ? expf((llv_sk_val(s.cand[k]) - max_val) * inv_temp) : 0.0f;
    }
    int cap = G;
    if (top_k_req > 0 && top_k_req < G) cap = top_k_req;
    if (min_p > 0.0f) {
        float floor_p = min_p * probs[0];
        int mp_cap = cap;
        for (int i = 1; i < cap; i++) {
            if (probs[i] < floor_p) { mp_cap = i; break; }
        }
        cap = mp_cap;
    }
    if (cap < 1) cap = 1;
    float sum = 0.0f;
    for (int i = 0; i < cap; i++) sum += probs[i];
    if (sum <= 0.0f) {
        Pu[LP_N] = 0u;
        Pu[LP_POINT] = 1u;
        Pu[LP_POINT_TOK] = (unsigned)s.cand[0];
        P[LP_MASS] = 0.0f;
        return;
    }
    float p_thresh = top_p * sum;
    float trunc_sum = 0.0f;
    int trunc_k = cap;
    for (int i = 0; i < cap; i++) {
        trunc_sum += probs[i];
        if (trunc_sum >= p_thresh) { trunc_k = i + 1; break; }
    }
    if (trunc_k > g) trunc_k = g;
    for (int i = 0; i < trunc_k; i++) {
        Pu[LP_TOK + i] = (unsigned)s.cand[i];
        P[LP_PROB + i] = probs[i];
    }
    Pu[LP_N] = (unsigned)trunc_k;
    Pu[LP_POINT] = 0u;
    Pu[LP_POINT_TOK] = 0u;
    P[LP_MASS] = trunc_sum;
}

// Inverse-CDF draw over the unnormalized nucleus (the shared sampler's tail).
static __device__ __forceinline__ unsigned llv_draw_p(const unsigned* Pu, const float* P, float u) {
    int n = (int)Pu[LP_N];
    float r = u * P[LP_MASS];
    float cum = 0.0f;
    for (int i = 0; i < n; i++) {
        cum += P[LP_PROB + i];
        if (cum >= r) return Pu[LP_TOK + i];
    }
    return Pu[LP_TOK + n - 1];
}

static __device__ __forceinline__ float llv_q_prob(const unsigned* Q, unsigned tok) {
    int n = (int)Q[0];
    for (int i = 0; i < n && i < LLV_Q_MAX; i++) {
        if (Q[1 + i] == tok) return __uint_as_float(Q[1 + LLV_Q_MAX + i]);
    }
    return 0.0f;
}

static __device__ __forceinline__ float llv_p_prob(const unsigned* Pu, const float* P, unsigned tok) {
    int n = (int)Pu[LP_N];
    for (int i = 0; i < n; i++) {
        if (Pu[LP_TOK + i] == tok) return P[LP_PROB + i] / P[LP_MASS];
    }
    return 0.0f;
}

// Residual (p - q)+ draw over p's support minus the rejected draft `d`; an
// empty residual (rounding) falls back to p without `d`.
static __device__ unsigned llv_draw_residual(const unsigned* Pu, const float* P, const unsigned* Q,
                                             unsigned d, float u) {
    int n = (int)Pu[LP_N];
    float mass_p = P[LP_MASS];
    float mass = 0.0f;
    for (int i = 0; i < n; i++) {
        if (Pu[LP_TOK + i] == d) continue;
        float r = P[LP_PROB + i] / mass_p - llv_q_prob(Q, Pu[LP_TOK + i]);
        if (r > 0.0f) mass += r;
    }
    unsigned last = d;
    if (mass > 0.0f) {
        float target = u * mass;
        float acc = 0.0f;
        for (int i = 0; i < n; i++) {
            if (Pu[LP_TOK + i] == d) continue;
            float r = P[LP_PROB + i] / mass_p - llv_q_prob(Q, Pu[LP_TOK + i]);
            if (!(r > 0.0f)) continue;
            acc += r;
            if (target < acc) return Pu[LP_TOK + i];
            last = Pu[LP_TOK + i];
        }
        return last;
    }
    float rest = 0.0f;
    for (int i = 0; i < n; i++) {
        if (Pu[LP_TOK + i] != d) rest += P[LP_PROB + i];
    }
    float target = u * rest;
    float cum = 0.0f;
    for (int i = 0; i < n; i++) {
        if (Pu[LP_TOK + i] == d) continue;
        cum += P[LP_PROB + i];
        last = Pu[LP_TOK + i];
        if (target < cum) return last;
    }
    return last;
}

// Mode 1 tail: publish the sorted top-g of the shard (s.cand[0..g), synced).
static __device__ void llv_store_cands(const LlvShared& s, unsigned long long* cand_local, int row, int g,
                                       int tid) {
    unsigned* w = (unsigned*)cand_local + (size_t)row * LLV_CAND_WORDS;
    unsigned long long* keys = (unsigned long long*)(w + 2);
    if (tid == 0) {
        w[0] = (unsigned)g;
        w[1] = 0u;
    }
    for (int t = tid; t < g; t += LLV_BLOCK) keys[t] = s.cand[t];
}

// Mode 2 head: merge the two ascending candidate rows into s.cand[0..g) and
// return g = min(G, g0 + g1). Keys are unique (distinct global ids), so a
// key's merged position is its own index plus the count of the other list's
// smaller keys. Ends synced.
static __device__ int llv_merge_cands(LlvShared& s, const unsigned long long* cand_local,
                                      const unsigned long long* cand_peer, int row, int G, int tid) {
    const unsigned* w0 = (const unsigned*)cand_local + (size_t)row * LLV_CAND_WORDS;
    const unsigned* w1 = (const unsigned*)cand_peer + (size_t)row * LLV_CAND_WORDS;
    int g0 = (int)w0[0];
    int g1 = (int)w1[0];
    if (g0 > LLV_G_MAX) g0 = LLV_G_MAX;
    if (g1 > LLV_G_MAX) g1 = LLV_G_MAX;
    const unsigned long long* k0 = (const unsigned long long*)(w0 + 2);
    const unsigned long long* k1 = (const unsigned long long*)(w1 + 2);
    // Stage both lists in LDS (hkey is free in this mode): one parallel load round.
    if (tid < g0) s.hkey[tid] = k0[tid];
    if (tid < g1) s.hkey[LLV_G_MAX + tid] = k1[tid];
    __syncthreads();
    if (tid < g0) {
        unsigned long long key = s.hkey[tid];
        int pos = tid;
        for (int j = 0; j < g1; j++) pos += (s.hkey[LLV_G_MAX + j] < key) ? 1 : 0;
        if (pos < G) s.cand[pos] = key;
    }
    if (tid < g1) {
        unsigned long long key = s.hkey[LLV_G_MAX + tid];
        int pos = tid;
        for (int j = 0; j < g0; j++) pos += (s.hkey[j] < key) ? 1 : 0;
        if (pos < G) s.cand[pos] = key;
    }
    __syncthreads();
    const int tot = g0 + g1;
    return tot < G ? tot : G;
}

// p table of the row, the lane ticket and (last row of the lane) the lane verdict.
// `g` and s.cand[0..g) are final and visible to the whole workgroup.
static __device__ void llv_tail(LlvShared& s, unsigned* region, float* ptab, const unsigned* Ln, int row,
                                int tid, int g, float temp, float top_p, int top_k_req, float min_p, int G) {
    float* P = ptab + (size_t)row * LLV_P_WORDS;
    if (tid == 0) {
        llv_build_p(s, g, temp, top_p, top_k_req, min_p, G, P);
        __threadfence();
        unsigned* out = region + Ln[LL_OUT];
        unsigned prev = atomicAdd(&out[LO_TICKET], 1u);
        s.last = (prev + 1u == Ln[LL_NROWS]) ? 1 : 0;
    }
    __syncthreads();
    if (!s.last) return;
    __threadfence();

    // ---- Lane verdict (last row of the lane to arrive). ----
    const int nrows = (int)Ln[LL_NROWS];
    const int row0 = (int)Ln[LL_ROW0];
    float* lp = (float*)s.cand;  // nrows * LLV_P_WORDS floats staged in LDS
    for (int w = tid; w < nrows * LLV_P_WORDS; w += LLV_BLOCK) {
        unsigned bits = __hip_atomic_load((const unsigned*)ptab + (size_t)row0 * LLV_P_WORDS + w,
                                          __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        lp[w] = __uint_as_float(bits);
    }
    __syncthreads();
    if (tid != 0) return;
    unsigned* out = region + Ln[LL_OUT];
    const unsigned* drafts = region + Ln[LL_DRAFTS];
    const unsigned* qrec = region + Ln[LL_Q];
    const unsigned eos = Ln[LL_EOS];
    const int n_drafts = nrows - 1;
    unsigned rng = Ln[LL_RNG];
    int accepted = 0;
    int ntok = 0;
    for (int j = 0; j < nrows; j++) {
        const float* Pj = lp + j * LLV_P_WORDS;
        const unsigned* Pu = (const unsigned*)Pj;
        const bool point = Pu[LP_POINT] != 0u;
        if (j < n_drafts) {
            const unsigned d = drafts[j];
            if (point) {
                unsigned t = Pu[LP_POINT_TOK];
                out[LO_TOKENS + ntok++] = t;
                if (t == d) {
                    accepted++;
                    if (d == eos) break;
                    continue;
                }
                break;
            }
            const unsigned* Q = qrec + j * LLV_Q_WORDS;
            float qx = llv_q_prob(Q, d);
            float px = llv_p_prob(Pu, Pj, d);
            rng = llv_xorshift(rng);
            float u = (float)(rng >> 8) * (1.0f / 16777216.0f);
            if (qx > 0.0f && u * qx < px) {
                out[LO_TOKENS + ntok++] = d;
                accepted++;
                if (d == eos) break;
                continue;
            }
            rng = llv_xorshift(rng);
            float u2 = (float)(rng >> 8) * (1.0f / 16777216.0f);
            out[LO_TOKENS + ntok++] = llv_draw_residual(Pu, Pj, Q, d, u2);
            break;
        }
        if (point) {
            out[LO_TOKENS + ntok++] = Pu[LP_POINT_TOK];
        } else {
            rng = llv_xorshift(rng);
            float u = (float)rng / 4294967295.0f;
            out[LO_TOKENS + ntok++] = llv_draw_p(Pu, Pj, u);
        }
        break;
    }
    out[LO_ACCEPTED] = (unsigned)accepted;
    out[LO_NTOK] = (unsigned)ntok;
    out[LO_RNG] = rng;
    out[LO_DONE] = 1u;
}

extern "C" __global__ __launch_bounds__(LLV_BLOCK) void lane_leviathan_verify(
    unsigned* __restrict__ region,  // [out | lanes | rows | inputs], u32 words
    int lanes_off,                  // word offset of the lane records
    int rows_off,                   // word offset of the row records
    float* __restrict__ ptab,       // [n_rows * LLV_P_WORDS]
    int mode,                       // LLV_MODE_*: 0 full row (today), 1 select, 2 merged
    unsigned long long* cand_local, // modes 1/2: [rows][LLV_CAND_WORDS] u32 (null in mode 0)
    const unsigned long long* cand_peer // mode 2: peer rank's table, same layout (null otherwise)
) {
    __shared__ LlvShared s;
    const int tid = (int)threadIdx.x;
    const int row = (int)blockIdx.x;
    const unsigned* R = region + rows_off + row * LLV_ROW_WORDS;
    const int lane = (int)R[LR_LANE];
    const unsigned* Ln = region + lanes_off + lane * LLV_LANE_WORDS;
    const float* L = (const float*)(((unsigned long long)R[LR_PTR_HI] << 32) | (unsigned long long)R[LR_PTR_LO]);
    const int vocab = (int)Ln[LL_VOCAB];
    const float temp = __uint_as_float(Ln[LL_TEMP]);
    const float top_p = __uint_as_float(Ln[LL_TOP_P]);
    const int top_k_req = (int)Ln[LL_TOP_K];
    const float min_p = __uint_as_float(Ln[LL_MIN_P]);
    const float rep = __uint_as_float(Ln[LL_REP]);
    const float pres = __uint_as_float(Ln[LL_PRES]);
    const float freq = __uint_as_float(Ln[LL_FREQ]);
    const int n_blocked = (int)Ln[LL_NBLOCKED];
    const int pen_len = (int)R[LR_PEN_LEN];
    const bool penalty = (rep > 1.0f || pres > 0.0f || freq > 0.0f) && pen_len > 0;
    const int n_scope = penalty ? pen_len : 0;
    const int G = (top_k_req <= 20) ? 20 : LLV_G_MAX;
    // Global id of L[0] (mode 1 only; modes 0 and 2 index the row from 0).
    const unsigned base = (mode == LLV_MODE_SELECT) ? Ln[LL_BASE] : 0u;
    if (mode == LLV_MODE_MERGED) {
        const int gm = llv_merge_cands(s, cand_local, cand_peer, row, G, tid);
        llv_tail(s, region, ptab, Ln, row, tid, gm, temp, top_p, top_k_req, min_p, G);
        return;
    }

    // ---- Scope / blocked token table (sorted keys, penalized values). ----
    // Scope key (tok << 32 | i), blocked key (tok << 32 | 0x80000000 | b): a token's
    // scope occurrences sort before its blocked marks.
    const int nk = n_scope + n_blocked;
    if (tid == 0) {
        s.ncand = 0;
        s.nh = nk > 0 ? llv_pow2(nk) : 0;
        s.last = 0;
    }
    for (int t = tid; t < LLV_FILTER_BITS / 32; t += LLV_BLOCK) s.filter[t] = 0u;
    __syncthreads();
    if (nk > 0) {
        const unsigned* scope = region + R[LR_PEN_OFF];
        const unsigned* blocked = region + Ln[LL_BLOCKED];
        const int n = s.nh;
        for (int t = tid; t < n; t += LLV_BLOCK) {
            unsigned long long k = 0xffffffffffffffffull;
            if (t < n_scope) {
                k = ((unsigned long long)scope[t] << 32) | (unsigned)t;
            } else if (t < nk) {
                k = ((unsigned long long)blocked[t - n_scope] << 32) | (0x80000000u | (unsigned)(t - n_scope));
            }
            s.hkey[t] = k;
        }
        __syncthreads();
        llv_bitonic(s.hkey, n, tid);
        const float window_len = (float)pen_len;
        for (int p = tid; p < nk; p += LLV_BLOCK) {
            unsigned long long key = s.hkey[p];
            unsigned tok = (unsigned)(key >> 32);
            s.hval[p] = 0.0f;
            if (tok - base >= (unsigned)vocab) continue;
            if (p + 1 < n && (unsigned)(s.hkey[p + 1] >> 32) == tok) continue;
            unsigned long long first = (unsigned long long)tok << 32;
            int lo = 0, hi = p;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (s.hkey[mid] < first) lo = mid + 1; else hi = mid;
            }
            int wcount = 0;
            while (lo + wcount <= p && !((unsigned)s.hkey[lo + wcount] & 0x80000000u)) wcount++;
            bool is_blocked = ((unsigned)key & 0x80000000u) != 0u;
            float v = L[tok - base];
            if (is_blocked) {
                v = -INFINITY;
            } else if (wcount > 0) {
                int i = (int)(unsigned)s.hkey[lo + wcount - 1];
                if (rep > 1.0f) {
                    float recency = ((float)i + 1.0f) / window_len;
                    float effective = powf(rep, (float)wcount * recency);
                    if (effective > 1.5f) effective = 1.5f;
                    v = (v > 0.0f) ? (v / effective) : (v * effective);
                }
                if (pres > 0.0f || freq > 0.0f) v -= freq * (float)wcount + pres;
            }
            s.hval[p] = v;
            atomicOr(&s.filter[(tok >> 5) & (LLV_FILTER_BITS / 32 - 1)], 1u << (tok & 31));
        }
        __syncthreads();
    }
    const bool has_h = nk > 0;

    // ---- Pass 1: per-thread maximum over unpenalized tokens. ----
    const bool vec = ((((unsigned long long)L) & 15ull) == 0ull);
    const int n4 = vec ? (vocab >> 2) : 0;
    const float4* L4 = (const float4*)L;
    float m = -INFINITY;
#define LLV_P1(v, ix)                                                              \
    if ((v) > m) {                                                                 \
        if (!has_h || llv_h_find(s, (unsigned)(ix) + base) < 0) m = (v);           \
    }
    {
        int i = tid;
        for (; i + 3 * LLV_BLOCK < n4; i += 4 * LLV_BLOCK) {
            float4 a = L4[i];
            float4 b = L4[i + LLV_BLOCK];
            float4 c = L4[i + 2 * LLV_BLOCK];
            float4 d = L4[i + 3 * LLV_BLOCK];
            int ia = i * 4, ib = (i + LLV_BLOCK) * 4, ic = (i + 2 * LLV_BLOCK) * 4, id = (i + 3 * LLV_BLOCK) * 4;
            LLV_P1(a.x, ia) LLV_P1(a.y, ia + 1) LLV_P1(a.z, ia + 2) LLV_P1(a.w, ia + 3)
            LLV_P1(b.x, ib) LLV_P1(b.y, ib + 1) LLV_P1(b.z, ib + 2) LLV_P1(b.w, ib + 3)
            LLV_P1(c.x, ic) LLV_P1(c.y, ic + 1) LLV_P1(c.z, ic + 2) LLV_P1(c.w, ic + 3)
            LLV_P1(d.x, id) LLV_P1(d.y, id + 1) LLV_P1(d.z, id + 2) LLV_P1(d.w, id + 3)
        }
        for (; i < n4; i += LLV_BLOCK) {
            float4 a = L4[i];
            int ia = i * 4;
            LLV_P1(a.x, ia) LLV_P1(a.y, ia + 1) LLV_P1(a.z, ia + 2) LLV_P1(a.w, ia + 3)
        }
        for (int ix = n4 * 4 + tid; ix < vocab; ix += LLV_BLOCK) {
            float v = L[ix];
            LLV_P1(v, ix)
        }
    }
#undef LLV_P1
    // tau = G-th largest thread maximum.
    s.cand[tid] = llv_sortkey(m, (unsigned)tid);
    __syncthreads();
    llv_bitonic(s.cand, LLV_BLOCK, tid);
    const float tau = llv_sk_val(s.cand[G - 1]);
    __syncthreads();

    // ---- Pass 2: collect { v >= tau } over unpenalized tokens. ----
#define LLV_P2(v, ix)                                                              \
    if ((v) >= tau && (v) > LLV_NEG_FLOOR) {                                       \
        if (!has_h || llv_h_find(s, (unsigned)(ix) + base) < 0) {                  \
            int slot = atomicAdd(&s.ncand, 1);                                     \
            if (slot < LLV_CAND_CAP) s.cand[slot] = llv_sortkey((v), (unsigned)(ix) + base); \
        }                                                                          \
    }
    {
        int i = tid;
        for (; i + 3 * LLV_BLOCK < n4; i += 4 * LLV_BLOCK) {
            float4 a = L4[i];
            float4 b = L4[i + LLV_BLOCK];
            float4 c = L4[i + 2 * LLV_BLOCK];
            float4 d = L4[i + 3 * LLV_BLOCK];
            int ia = i * 4, ib = (i + LLV_BLOCK) * 4, ic = (i + 2 * LLV_BLOCK) * 4, id = (i + 3 * LLV_BLOCK) * 4;
            LLV_P2(a.x, ia) LLV_P2(a.y, ia + 1) LLV_P2(a.z, ia + 2) LLV_P2(a.w, ia + 3)
            LLV_P2(b.x, ib) LLV_P2(b.y, ib + 1) LLV_P2(b.z, ib + 2) LLV_P2(b.w, ib + 3)
            LLV_P2(c.x, ic) LLV_P2(c.y, ic + 1) LLV_P2(c.z, ic + 2) LLV_P2(c.w, ic + 3)
            LLV_P2(d.x, id) LLV_P2(d.y, id + 1) LLV_P2(d.z, id + 2) LLV_P2(d.w, id + 3)
        }
        for (; i < n4; i += LLV_BLOCK) {
            float4 a = L4[i];
            int ia = i * 4;
            LLV_P2(a.x, ia) LLV_P2(a.y, ia + 1) LLV_P2(a.z, ia + 2) LLV_P2(a.w, ia + 3)
        }
        for (int ix = n4 * 4 + tid; ix < vocab; ix += LLV_BLOCK) {
            float v = L[ix];
            LLV_P2(v, ix)
        }
    }
#undef LLV_P2
    __syncthreads();
    // Penalized / blocked scope tokens with v >= tau.
    if (has_h) {
        for (int p = tid; p < nk; p += LLV_BLOCK) {
            unsigned tok = (unsigned)(s.hkey[p] >> 32);
            if (tok - base >= (unsigned)vocab) continue;
            if (p + 1 < s.nh && (unsigned)(s.hkey[p + 1] >> 32) == tok) continue;
            float v = s.hval[p];
            if (v >= tau && v > LLV_NEG_FLOOR) {
                int slot = atomicAdd(&s.ncand, 1);
                if (slot < LLV_CAND_CAP) s.cand[slot] = llv_sortkey(v, tok);
            }
        }
        __syncthreads();
    }
    int nc = s.ncand;
    int g;
    if (nc <= LLV_CAND_CAP) {
        int n = llv_pow2(nc);
        for (int t = nc + tid; t < n; t += LLV_BLOCK) s.cand[t] = 0xffffffffffffffffull;
        __syncthreads();
        llv_bitonic(s.cand, n, tid);
        g = nc < G ? nc : G;
    } else {
        // Tile reduction (exact ties beyond LDS): keep the running top-G in
        // cand[0..G), merge each tile's {sortkey <= kth} and re-sort.
        __syncthreads();
        for (int t = tid; t < LLV_G_MAX; t += LLV_BLOCK) s.cand[t] = 0xffffffffffffffffull;
        unsigned long long kth = llv_sortkey(tau, 0xffffffffu);
        if (tid == 0) s.ncand = G;
        __syncthreads();
        for (int tbase = 0; tbase < vocab; tbase += LLV_TILE) {
            for (int ix = tbase + tid; ix < tbase + LLV_TILE && ix < vocab; ix += LLV_BLOCK) {
                float v = L[ix];
                if (v > LLV_NEG_FLOOR) {
                    unsigned long long sk = llv_sortkey(v, (unsigned)ix + base);
                    if (sk <= kth && (!has_h || llv_h_find(s, (unsigned)ix + base) < 0)) {
                        int slot = atomicAdd(&s.ncand, 1);
                        s.cand[slot] = sk;
                    }
                }
            }
            __syncthreads();
            int cnt = s.ncand;
            if (cnt > G) {
                int n = llv_pow2(cnt);
                for (int t = cnt + tid; t < n; t += LLV_BLOCK) s.cand[t] = 0xffffffffffffffffull;
                __syncthreads();
                llv_bitonic(s.cand, n, tid);
                if (s.cand[G - 1] < kth) kth = s.cand[G - 1];
                __syncthreads();
                if (tid == 0) s.ncand = G;
                __syncthreads();
            }
        }
        if (has_h) {
            for (int p = tid; p < nk; p += LLV_BLOCK) {
                unsigned tok = (unsigned)(s.hkey[p] >> 32);
                if (tok - base >= (unsigned)vocab) continue;
                if (p + 1 < s.nh && (unsigned)(s.hkey[p + 1] >> 32) == tok) continue;
                float v = s.hval[p];
                if (v > LLV_NEG_FLOOR) {
                    unsigned long long sk = llv_sortkey(v, tok);
                    if (sk <= kth) {
                        int slot = atomicAdd(&s.ncand, 1);
                        s.cand[slot] = sk;
                    }
                }
            }
            __syncthreads();
            int cnt = s.ncand;
            int n = llv_pow2(cnt);
            for (int t = cnt + tid; t < n; t += LLV_BLOCK) s.cand[t] = 0xffffffffffffffffull;
            __syncthreads();
            llv_bitonic(s.cand, n, tid);
        }
        g = 0;
        while (g < G && s.cand[g] != 0xffffffffffffffffull) g++;
    }

    if (mode == LLV_MODE_SELECT) {
        llv_store_cands(s, cand_local, row, g, tid);
        return;
    }

    // ---- p table, then the lane ticket and verdict. ----
    llv_tail(s, region, ptab, Ln, row, tid, g, temp, top_p, top_k_req, min_p, G);
}
