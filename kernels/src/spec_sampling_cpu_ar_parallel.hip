// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Parallel target-row builder for the CPU-AR speculative sampling law:
//
//   spec_cpu_ar_rows_parallel_f32   one 256-thread workgroup per target row
//
// This is a NEW target law, not a bit-exact twin of the serial reference
// (spec_sampling_cpu_ar.hip, left untouched). Differences, all intentional:
//
//   * Pool selection is canonical top-`pool` under the strict total order
//       a precedes b  <=>  a.logit > b.logit || (a.logit == b.logit && a.id < b.id)
//     over the finite logits of the row. Numerically equal logits (+0 and -0
//     included) therefore tie on the LOWER token id. The serial reference
//     keeps a history-dependent slot set on ties.
//   * Aggregate softmax mass (and the capped/min_p-cut mass) is a fixed
//     64-leaf binary tree (strides 32,16,8,4,2,1, left = lower index), not a
//     sequential sum. Row maximum is a fixed 256-lane tree.
//   * Weights with equal rounded exp keep canonical (logit, id) order.
//
// Everything else follows the reference literally: finite-only inputs,
// w = expf(RN(RN(l - max) * RN(1/temperature))), non-finite exp -> 0,
// stable descending sort by weight, min_p floor RN(min_p*w0) with the first
// rank i>=1 below it cutting, top_p threshold RN(top_p*mass), serial
// cumulative over the <=64 pool, the crossing token is included, residual
// correction on no-crossing, p>0 retained in order, zero tails, and the
// argmax fallback (point mass) when the full-pool mass is <=0 or non-finite.
//
// Nothing fuses: fp-contract is off and every adjacent add/mul/div is an
// explicit round-to-nearest intrinsic. `expf` is the plain HIP expf.
//
// Algorithm. The vocabulary is processed in tiles of 1024 tokens in ascending
// tile order. Per tile, lane t loads tokens base+t+256*j (j=0..3), builds four
// records (finite -> (logit,id); invalid -> padding (-inf, UINT32_MAX), which
// follows every real record), orders them with a fixed 5-comparator network
// and writes them to LDS. Eight workgroup merge levels join sorted list pairs,
// every output rank computed independently by a binary co-rank (merge-path)
// search, retaining at most `pool` records per list. The tile's top-`pool`
// is then merged with the running top-`pool` the same way (rank r by lane r).
// The canonical global top-`pool` is contained in the union of per-tile
// canonical top-`pool` lists, so no concatenation or serial vocabulary pass
// exists. Each lane also keeps its best record (fixed 4-item tree per tile,
// then tile order) for the max tree; the tree yields the row maximum and the
// fallback token.
//
// Kernarg order: logits (f32 [rows*vocab]), counts (u32 [rows]),
// ids (u32 [rows*64]), probs (f32 [rows*64]),
// vocab i32, pool i32, cap i32, temperature f32, top_p f32, min_p f32.
// Grid [rows,1,1], block [256,1,1], dynamic shared 0. No global scratch, no
// atomics. Output layout: counts[row] u32, ids[row*64+j] u32, probs[row*64+j]
// f32 in descending-probability order; slots past counts[row] are zero.
//
// Static LDS: 2*1024*8 (tile ping-pong) + 2*64*8 (running pool) + 256*8
// (max tree) + 2*64*4 (weights, sum tree) + 8 (control) = 19976 bytes.

#include <hip/hip_runtime.h>
#include <stdint.h>

#pragma clang fp contract(off)

#define SCP_THREADS 256
#define SCP_TILE 1024
#define SCP_PER_LANE 4
#define SCP_CAP 64
#define SCP_PAD_ID 0xFFFFFFFFu

struct ScpEntry {
    float logit;
    uint32_t id;
};

static __device__ __forceinline__ float scp_neg_inf() {
    return __uint_as_float(0xff800000u);
}

static __device__ __forceinline__ ScpEntry scp_pad() {
    ScpEntry e;
    e.logit = scp_neg_inf();
    e.id = SCP_PAD_ID;
    return e;
}

// Strict canonical precedence: higher logit first, equal logits -> lower id.
// Padding (-inf, UINT32_MAX) follows every real (finite) record and does not
// precede another identical padding record.
static __device__ __forceinline__ bool scp_before(ScpEntry a, ScpEntry b) {
    return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
}

// Winner of two records; ties (identical padding only) keep the left operand.
static __device__ __forceinline__ ScpEntry scp_best(ScpEntry left, ScpEntry right) {
    return scp_before(right, left) ? right : left;
}

static __device__ __forceinline__ void scp_cswap(ScpEntry& a, ScpEntry& b) {
    if (scp_before(b, a)) { const ScpEntry t = a; a = b; b = t; }
}

// Fixed 5-comparator sorting network for four records.
static __device__ __forceinline__ void scp_sort4(ScpEntry (&x)[4]) {
    scp_cswap(x[0], x[1]);
    scp_cswap(x[2], x[3]);
    scp_cswap(x[0], x[2]);
    scp_cswap(x[1], x[3]);
    scp_cswap(x[1], x[2]);
}

// Record at output position `rank` of the merge of sorted lists a[0,na) and
// b[0,nb); the left list wins comparator ties (identical padding). Requires
// rank < na + nb. Binary co-rank: find i = the number of `a` records among the
// first `rank` outputs, then the output is the better of a[i], b[rank-i].
static __device__ __forceinline__ ScpEntry scp_merge_rank(
    const ScpEntry* a, int na, const ScpEntry* b, int nb, int rank)
{
    int lo = rank - nb; if (lo < 0) lo = 0;
    int hi = rank < na ? rank : na;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        // a[mid] stays within the first `rank` outputs iff it is not after
        // b[rank-mid-1] (ties go left).
        if (!scp_before(b[rank - mid - 1], a[mid])) lo = mid + 1; else hi = mid;
    }
    const int i = lo;
    const int j = rank - lo;
    if (i < na && (j >= nb || !scp_before(b[j], a[i]))) return a[i];
    return b[j];
}

static __device__ __forceinline__ float scp_max0(float v) {
    // Rust f32::max(v, 0.0): NaN loses.
    return fmaxf(v, 0.0f);
}

extern "C" __global__ __launch_bounds__(256) void spec_cpu_ar_rows_parallel_f32(
    const float* __restrict__ logits,
    uint32_t* __restrict__ counts,
    uint32_t* __restrict__ ids,
    float* __restrict__ probs,
    int vocab,
    int pool,
    int cap_in,
    float temperature,
    float top_p,
    float min_p)
{
    __shared__ ScpEntry s_buf[2][SCP_TILE];     // tile merge ping-pong
    __shared__ ScpEntry s_run[2][SCP_CAP];      // running top-pool ping-pong
    __shared__ ScpEntry s_max[SCP_THREADS];     // max tree leaves
    __shared__ float s_w[SCP_CAP];              // weights, then sorted probabilities
    __shared__ float s_sum[SCP_CAP];            // 64-leaf mass tree
    __shared__ int s_cap;
    __shared__ int s_n;

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const float* L = logits + (size_t)row * (size_t)vocab;
    const float NEG_INF = scp_neg_inf();

    if (pool < 1) pool = 1;
    if (pool > SCP_CAP) pool = SCP_CAP;

    // Running pool starts as all padding; the tile-level merge fills it.
    if (tid < SCP_CAP) s_run[0][tid] = scp_pad();
    int cur = 0;
    ScpEntry lane_max = scp_pad();
    const int len0 = pool < SCP_PER_LANE ? pool : SCP_PER_LANE;
    const int tiles = vocab > 0 ? (int)(((long long)vocab + SCP_TILE - 1) / SCP_TILE) : 0;

    for (int tile = 0; tile < tiles; ++tile) {
        const int base = tile * SCP_TILE;

        // Strided coalesced loads: lane t owns base + t + 256*j.
        ScpEntry x[4];
        #pragma unroll
        for (int j = 0; j < SCP_PER_LANE; ++j) {
            const int idx = base + tid + SCP_THREADS * j;
            ScpEntry e = scp_pad();
            if (idx < vocab) {
                const float l = L[idx];
                if (isfinite(l)) { e.logit = l; e.id = (uint32_t)idx; }
            }
            x[j] = e;
        }
        // Lane maximum: fixed four-item comparator tree, then tile order.
        const ScpEntry m01 = scp_best(x[0], x[1]);
        const ScpEntry m23 = scp_best(x[2], x[3]);
        lane_max = scp_best(lane_max, scp_best(m01, m23));

        scp_sort4(x);
        #pragma unroll
        for (int k = 0; k < SCP_PER_LANE; ++k) {
            if (k < len0) s_buf[0][tid * len0 + k] = x[k];
        }
        __syncthreads();

        // Eight merge levels: 256 sorted lists -> 1, each list <= pool long.
        int len = len0;
        int lists = SCP_THREADS;
        int src = 0;
        for (int level = 0; level < 8; ++level) {
            const int nlen = (2 * len < pool) ? 2 * len : pool;
            const int pairs = lists >> 1;
            const int total = pairs * nlen;
            const ScpEntry* sb = s_buf[src];
            ScpEntry* db = s_buf[src ^ 1];
            for (int o = tid; o < total; o += SCP_THREADS) {
                const int p = o / nlen;
                const int r = o - p * nlen;
                const ScpEntry* a = sb + (2 * p) * len;
                db[o] = scp_merge_rank(a, len, a + len, len, r);
            }
            __syncthreads();
            src ^= 1;
            len = nlen;
            lists = pairs;
        }
        // 8 levels (even) leave the tile's top-pool (len == pool) in s_buf[0].

        // Merge the tile pool with the running pool: lane r < pool owns rank r.
        if (tid < pool) {
            s_run[cur ^ 1][tid] = scp_merge_rank(s_run[cur], pool, s_buf[src], pool, tid);
        }
        __syncthreads();
        cur ^= 1;
    }

    // Max tree over the 256 lane maxima: strides 128..1, lower index left.
    s_max[tid] = lane_max;
    __syncthreads();
    #pragma unroll
    for (int s = SCP_THREADS >> 1; s >= 1; s >>= 1) {
        if (tid < s) s_max[tid] = scp_best(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }
    const ScpEntry top = s_max[0];
    const bool have_top = top.id != SCP_PAD_ID;
    const float max_logit = have_top ? top.logit : NEG_INF;

    // Weights. Records beyond pool / padding contribute +0.
    const float inv_temp = __fdiv_rn(1.0f, temperature);
    uint32_t* s_et = (uint32_t*)&s_buf[1][0]; // 64 u32; s_buf is idle here
    if (tid < SCP_CAP) {
        float p = 0.0f;
        uint32_t t = 0u;
        if (tid < pool) {
            const ScpEntry e = s_run[cur][tid];
            t = e.id;
            if (e.id != SCP_PAD_ID && isfinite(e.logit)) {
                p = expf(__fmul_rn(__fadd_rn(e.logit, -max_logit), inv_temp));
                if (!isfinite(p)) p = 0.0f;
            }
        }
        s_w[tid] = p;
        s_et[tid] = t;
        s_sum[tid] = p;
    }
    __syncthreads();
    #pragma unroll
    for (int s = SCP_CAP >> 1; s >= 1; s >>= 1) {
        if (tid < s) s_sum[tid] = __fadd_rn(s_sum[tid], s_sum[tid + s]);
        __syncthreads();
    }
    float sum = s_sum[0];
    const bool bad = (sum <= 0.0f || !isfinite(sum));

    // Lane 0: bounded (<=64) sort, min_p / cap cut. Others wait at the barrier.
    if (tid == 0) {
        int cap = cap_in;
        if (!bad) {
            // stable insertion sort, descending weight (strict >)
            for (int i = 1; i < pool; ++i) {
                int j = i;
                while (j > 0 && s_w[j] > s_w[j - 1]) {
                    const float tp = s_w[j]; s_w[j] = s_w[j - 1]; s_w[j - 1] = tp;
                    const uint32_t tt = s_et[j]; s_et[j] = s_et[j - 1]; s_et[j - 1] = tt;
                    --j;
                }
            }
            if (cap < 1) cap = 1;
            if (cap > pool) cap = pool;
            if (min_p > 0.0f) {
                const float floor_p = __fmul_rn(min_p, s_w[0]);
                for (int i = 1; i < cap; ++i) {
                    if (s_w[i] < floor_p) { cap = i; break; }
                }
            }
        }
        s_cap = cap;
    }
    __syncthreads();

    // A cut pool needs the masked prefix mass through the same fixed tree.
    const int cap_cut = s_cap;
    if (!bad && cap_cut < pool) {
        if (tid < SCP_CAP) s_sum[tid] = tid < cap_cut ? s_w[tid] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int s = SCP_CAP >> 1; s >= 1; s >>= 1) {
            if (tid < s) s_sum[tid] = __fadd_rn(s_sum[tid], s_sum[tid + s]);
            __syncthreads();
        }
        sum = s_sum[0];
    }

    // Lane 0: serial nucleus over the <=64 pool, normalization, compaction.
    if (tid == 0) {
        int n;
        if (bad) {
            // Point mass at the canonical maximum (lowest id among equal logits),
            // token 0 when no finite logit exists.
            s_et[0] = have_top ? top.id : 0u;
            s_w[0] = 1.0f;
            n = 1;
        } else {
            const int cap = cap_cut;
            const float threshold = __fmul_rn(top_p, sum);
            float cumulative = 0.0f;
            int boundary = -1;
            for (int i = 0; i < cap; ++i) {
                cumulative = __fadd_rn(cumulative, s_w[i]);
                if (cumulative >= threshold) { boundary = i + 1; break; }
            }
            if (boundary >= 0) {
                n = boundary;
                for (int i = 0; i < n; ++i) s_w[i] = __fdiv_rn(s_w[i], cumulative);
            } else {
                n = cap;
                for (int i = 0; i < n; ++i) s_w[i] = __fdiv_rn(s_w[i], sum);
                s_w[0] = __fadd_rn(s_w[0], scp_max0(__fadd_rn(1.0f, -__fdiv_rn(cumulative, sum))));
            }
            int m = 0; // retain p > 0, order preserved
            for (int i = 0; i < n; ++i) {
                if (s_w[i] > 0.0f) { s_w[m] = s_w[i]; s_et[m] = s_et[i]; ++m; }
            }
            n = m;
        }
        s_n = n;
        counts[row] = (uint32_t)n;
    }
    __syncthreads();

    if (tid < SCP_CAP) {
        const int n = s_n;
        ids[(size_t)row * SCP_CAP + tid] = tid < n ? s_et[tid] : 0u;
        probs[(size_t)row * SCP_CAP + tid] = tid < n ? s_w[tid] : 0.0f;
    }
}
