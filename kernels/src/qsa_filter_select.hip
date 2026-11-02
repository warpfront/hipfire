// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exact sampled-threshold filter selector (B1). Three kernels, all launched
// with blockDim.x == 256:
//
//   qsa_filter_threshold   grid (rows)                    one row per WG
//   qsa_filter_compact     grid (ceil(block_count/4096), rows)
//   qsa_filter_finish      grid (rows)                    one row per WG
//
// Shared conventions
//   scores       const float*, row r at scores + r * score_stride. Values are
//                nonnegative non-NaN f32; only the first rbc(r) entries are read.
//   rbc(r)       = min((position_start + r + 1) / compress, block_count)
//   key(r, b)    = (u64(raw f32 bits of scores[r][b]) << 32) | (0xffffffff - b)
//                  Unique per (row, b); u64 DESC order == score DESC, block ASC.
//   chosen(r)    = min(budget, rbc(r)). The finish kernel requires
//                  budget <= 512 for every launch (LDS winner capacity); larger
//                  budgets are clamped to 512 for memory safety only.
//   ctl          u64[rows * 2]. ctl[2r] = threshold key. ctl[2r+1]: low u32 =
//                candidate count (keys >= threshold, counted past cap), high
//                u32 = repaired flag (0/1). The threshold kernel initialises
//                word 1 to 0; compact atomically adds to its low half; finish
//                stores (1 << 32) | count on repair.
//   cand         u64[rows * cap]; slot r*cap+s holds a key >= threshold. Only
//                the first min(count, cap) slots are valid; order is
//                nondeterministic (finish re-ranks exactly).
//
// Exactness: certified rows (chosen <= count <= cap) are ranked from the
// complete candidate set, which contains the chosen largest keys because the
// chosen-th largest key >= threshold. Every other row (count < chosen or
// count > cap) is recomputed by an exact 64-bit radix select over all rbc
// keys. No row is ever approximated. Each output slot gets exactly one write.
#include <hip/hip_runtime.h>

#define QSA_FILTER_THREADS 256
#define QSA_FILTER_MAX_WIN 512
#define QSA_FILTER_MAX_SAMPLES 4096
#define QSA_FILTER_MAX_STAGE 4096

static __device__ __forceinline__ unsigned long long qsa_filter_key(float score, int block) {
    return (static_cast<unsigned long long>(__float_as_uint(score)) << 32)
        | static_cast<unsigned long long>(0xffffffffu - static_cast<unsigned int>(block));
}

static __device__ __forceinline__ int qsa_filter_rbc(
    int row, int block_count, int compress, int position_start) {
    return min((position_start + row + 1) / compress, block_count);
}

// Exact k-th largest (1-based) of n u64 keys produced by key_of(i), 1 <= k <= n.
// Eight 8-bit MSD radix passes; matching keys are re-derived each pass, so no
// key storage is needed. Block-uniform; every thread of the WG must call it.
// LDS: hist[256], scan0[256], scan1[256], bc[2] (all caller-owned).
template <typename KeyFn>
static __device__ __forceinline__ unsigned long long qsa_filter_kth(
    KeyFn key_of, int n, unsigned int k,
    unsigned int* hist, unsigned int* scan0, unsigned int* scan1, unsigned int* bc) {
    const int tid = (int)threadIdx.x;
    unsigned long long prefix = 0;
    unsigned int krem = k;
    for (int pass = 0; pass < 8; ++pass) {
        const int shift = 56 - 8 * pass;
        const unsigned long long hi_mask = pass == 0 ? 0ull : (~0ull) << (shift + 8);
        hist[tid] = 0;
        __syncthreads();
        for (int i = tid; i < n; i += QSA_FILTER_THREADS) {
            const unsigned long long key = key_of(i);
            if (((key ^ prefix) & hi_mask) == 0)
                atomicAdd(&hist[(unsigned int)(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        // Inclusive scan over digits in DESC order: thread t owns digit 255-t.
        const unsigned int h = hist[255 - tid];
        unsigned int* src = scan0;
        unsigned int* dst = scan1;
        src[tid] = h;
        __syncthreads();
#pragma unroll
        for (int off = 1; off < QSA_FILTER_THREADS; off <<= 1) {
            unsigned int v = src[tid];
            if (tid >= off) v += src[tid - off];
            dst[tid] = v;
            unsigned int* t = src; src = dst; dst = t;
            __syncthreads();
        }
        const unsigned int incl = src[tid];
        if (incl >= krem && incl - h < krem) {
            bc[0] = (unsigned int)(255 - tid);
            bc[1] = krem - (incl - h);
        }
        __syncthreads();
        prefix |= static_cast<unsigned long long>(bc[0]) << shift;
        krem = bc[1];
        // bc is rewritten only after the next pass's syncs; hist/scan likewise.
    }
    return prefix;
}

// ---------------------------------------------------------------------------
// Kernel 1. Args (in order):
//   const float* scores, int score_stride, unsigned long long* ctl,
//   int rows, int block_count, int compress, int position_start,
//   int samples, int target, int cap
// rbc <= cap (or rbc == 0): threshold 0. Otherwise S = clamp(samples, 1, 4096)
// (host should pass <= 4096; production 2048) evenly strided samples
// b_i = i * rbc / S are gathered ONCE into an LDS key array (8 * S bytes, 16 KiB
// at S = 2048, 32 KiB at 4096); the 8 radix passes then run on LDS only.
// j = clamp(ceil(S * target / rbc), 1, S); threshold = exact j-th largest
// sample key. Estimated static LDS: 8 * 4096 + 2 KiB + 8 B ~= 34 KiB (unverified).
extern "C" __global__ __launch_bounds__(256) void qsa_filter_threshold(
    const float* scores, int score_stride, unsigned long long* ctl,
    int rows, int block_count, int compress, int position_start,
    int samples, int target, int cap) {
    const int row = (int)blockIdx.x;
    if (row >= rows) return;
    __shared__ unsigned long long samp[QSA_FILTER_MAX_SAMPLES];
    __shared__ unsigned int hist[256];
    __shared__ unsigned int scan0[256];
    __shared__ unsigned int scan1[256];
    __shared__ unsigned int bc[2];
    const int tid = (int)threadIdx.x;
    const int rbc = qsa_filter_rbc(row, block_count, compress, position_start);
    if (rbc <= cap || rbc <= 0) {
        if (tid == 0) { ctl[(size_t)row * 2] = 0; ctl[(size_t)row * 2 + 1] = 0; }
        return;
    }
    const int s = min(max(samples, 1), QSA_FILTER_MAX_SAMPLES);
    long long jj = ((long long)s * (long long)max(target, 0) + rbc - 1) / rbc;
    jj = jj < 1 ? 1 : (jj > s ? s : jj);
    const float* row_scores = scores + (size_t)row * (size_t)score_stride;
    for (int i = tid; i < s; i += QSA_FILTER_THREADS) {
        const int b = (int)(((long long)i * (long long)rbc) / s);
        samp[i] = qsa_filter_key(row_scores[b], b);
    }
    __syncthreads();
    const unsigned long long threshold = qsa_filter_kth(
        [&](int i) -> unsigned long long { return samp[i]; },
        s, (unsigned int)jj, hist, scan0, scan1, bc);
    if (tid == 0) { ctl[(size_t)row * 2] = threshold; ctl[(size_t)row * 2 + 1] = 0; }
}

// ---------------------------------------------------------------------------
// Kernel 2. Grid (ceil(block_count / 4096), rows). Args (in order):
//   const float* scores, int score_stride, unsigned long long* ctl,
//   unsigned long long* cand, int rows, int block_count, int compress,
//   int position_start, int cap
// Each WG covers blocks [x*4096, x*4096+4096) of row y in 16 coalesced
// 256-wide steps; one 32-bit atomicAdd per wave per step on the low half of
// ctl[2y+1] reserves slots. Counting continues beyond cap; only slots < cap
// are written.
extern "C" __global__ __launch_bounds__(256) void qsa_filter_compact(
    const float* scores, int score_stride, unsigned long long* ctl,
    unsigned long long* cand, int rows, int block_count, int compress,
    int position_start, int cap) {
    const int row = (int)blockIdx.y;
    if (row >= rows) return;
    const int rbc = qsa_filter_rbc(row, block_count, compress, position_start);
    const int base = (int)blockIdx.x * 4096;
    if (base >= rbc) return;
    const int tid = (int)threadIdx.x;
    const unsigned int lane = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
    const unsigned long long threshold = ctl[(size_t)row * 2];
    unsigned int* count = reinterpret_cast<unsigned int*>(&ctl[(size_t)row * 2 + 1]);  // low half (little endian)
    const float* row_scores = scores + (size_t)row * (size_t)score_stride;
    unsigned long long* row_cand = cand + (size_t)row * (size_t)cap;
#pragma unroll 4
    for (int step = 0; step < 16; ++step) {
        const int b = base + step * QSA_FILTER_THREADS + tid;
        unsigned long long key = 0;
        bool hit = false;
        if (b < rbc) {
            key = qsa_filter_key(row_scores[b], b);
            hit = key >= threshold;
        }
        const unsigned long long mask = __ballot(hit);
        if (mask == 0) continue;
        const int leader = __ffsll((long long)mask) - 1;
        unsigned int first = 0;
        if ((int)lane == leader) first = atomicAdd(count, (unsigned int)__popcll(mask));
        first = (unsigned int)__shfl((int)first, leader);
        if (hit) {
            const unsigned int slot = first + (unsigned int)__popcll(mask & ((1ull << lane) - 1ull));
            if (slot < (unsigned int)cap) row_cand[slot] = key;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3. Grid (rows), block 256. Args (in order):
//   const float* scores, int score_stride, unsigned long long* ctl,
//   const unsigned long long* cand, int* selected, int* mirror (may be null),
//   int rows, int block_count, int budget, int compress, int position_start,
//   int capacity, int cap
// Output row r at selected + r * capacity: top chosen blocks by key DESC, each
// expanded to `compress` consecutive tokens, then the causal tail tokens
// [rbc*compress, visible), then -1 up to capacity. `mirror`, if non-null,
// receives a copy of the LAST row only (row == rows - 1), same indexing as the
// row output (mirror[0..capacity)).
// Estimated static LDS (unverified): cand_lds 32 KiB + win 4 KiB + radix
// scratch 3 KiB ~= 39 KiB. Certified rows with count <= 4096 stage candidates
// in LDS once; certified rows with count > 4096 (only if cap > 4096) and repair
// rows read global memory per radix pass.
extern "C" __global__ __launch_bounds__(256) void qsa_filter_finish(
    const float* scores, int score_stride, unsigned long long* ctl,
    const unsigned long long* cand, int* selected, int* mirror,
    int rows, int block_count, int budget, int compress, int position_start,
    int capacity, int cap) {
    const int row = (int)blockIdx.x;
    if (row >= rows) return;
    __shared__ unsigned long long cand_lds[QSA_FILTER_MAX_STAGE];
    __shared__ unsigned long long win[QSA_FILTER_MAX_WIN];
    __shared__ unsigned int hist[256];
    __shared__ unsigned int scan0[256];
    __shared__ unsigned int scan1[256];
    __shared__ unsigned int bc[2];
    __shared__ unsigned int nwin;
    const int tid = (int)threadIdx.x;
    if (row != rows - 1) mirror = nullptr;
    const int visible = position_start + row + 1;
    const int rbc = qsa_filter_rbc(row, block_count, compress, position_start);
    const int chosen = min(min(budget, rbc), QSA_FILTER_MAX_WIN);
    int* output = selected + (size_t)row * (size_t)capacity;
    const float* row_scores = scores + (size_t)row * (size_t)score_stride;
    const unsigned long long ctl1 = ctl[(size_t)row * 2 + 1];
    const unsigned int count = (unsigned int)ctl1;

    // Contract: certified iff chosen <= count <= cap; otherwise the repaired
    // flag is set (also for chosen == 0 overflow, which needs no radix work).
    // Host contract is cap <= 4096; the min() only keeps LDS staging in bounds
    // if it is violated (such rows are repaired exactly instead).
    const bool certified = count >= (unsigned int)chosen
        && count <= (unsigned int)min(cap, QSA_FILTER_MAX_STAGE);
    __syncthreads();  // every thread has read ctl[2r+1] before thread 0 may overwrite it
    if (!certified && tid == 0)
        ctl[(size_t)row * 2 + 1] = (1ull << 32) | static_cast<unsigned long long>(count);
    if (chosen > 0) {
        if (tid == 0) nwin = 0;
        const unsigned long long* row_cand = cand + (size_t)row * (size_t)cap;
        // Exact k-th key (k = chosen) over n keys from key_of, then every key
        // >= cut goes to win[] (keys are unique, so exactly `chosen` of them).
        auto select_top = [&](auto key_of, int n) {
            const unsigned long long cut = qsa_filter_kth(
                key_of, n, (unsigned int)chosen, hist, scan0, scan1, bc);
            for (int i = tid; i < n; i += QSA_FILTER_THREADS) {
                const unsigned long long key = key_of(i);
                if (key >= cut) {
                    const unsigned int slot = atomicAdd(&nwin, 1u);
                    if (slot < QSA_FILTER_MAX_WIN) win[slot] = key;
                }
            }
        };
        if (certified) {
            // One global read per candidate; the 8 radix passes and the winner
            // scan run on LDS. certified implies count <= 4096 (see above).
            for (int i = tid; i < (int)count; i += QSA_FILTER_THREADS) cand_lds[i] = row_cand[i];
            __syncthreads();
            select_top([&](int i) -> unsigned long long { return cand_lds[i]; }, (int)count);
        } else {
            select_top([&](int i) -> unsigned long long { return qsa_filter_key(row_scores[i], i); }, rbc);
        }
        __syncthreads();
        // Keys are unique, so nwin == chosen. Pad to a power of two with 0
        // (below every real key, whose low word is nonzero) and bitonic-sort DESC.
        int pow2 = 2;
        while (pow2 < chosen) pow2 <<= 1;
        for (int i = chosen + tid; i < pow2; i += QSA_FILTER_THREADS) win[i] = 0;
        __syncthreads();
        for (int size = 2; size <= pow2; size <<= 1) {
            for (int step = size >> 1; step > 0; step >>= 1) {
                for (int t = tid; t < (pow2 >> 1); t += QSA_FILTER_THREADS) {
                    const int i = ((t & ~(step - 1)) << 1) | (t & (step - 1));
                    const int j = i + step;
                    const unsigned long long a = win[i], b = win[j];
                    const bool desc = (i & size) == 0;
                    if (desc ? (a < b) : (a > b)) { win[i] = b; win[j] = a; }
                }
                __syncthreads();
            }
        }
    }

    const int tail_start = rbc * compress;
    const int offset = chosen * compress;
    // Disjoint single-writer regions: winners [0, min(offset, capacity)),
    // tail [offset, written), padding [written, capacity).
    const int written = min(capacity, offset + (visible - tail_start));
    for (int i = written + tid; i < capacity; i += QSA_FILTER_THREADS) {
        output[i] = -1;
        if (mirror) mirror[i] = -1;
    }
    for (int rank = tid; rank < chosen; rank += QSA_FILTER_THREADS) {
        const int block = (int)(0xffffffffu - static_cast<unsigned int>(win[rank]));
        const int first = rank * compress;
        if (compress == 4 && first + 4 <= capacity) {
            const int base_token = block * 4;
            const int4 tokens = make_int4(base_token, base_token + 1, base_token + 2, base_token + 3);
            __builtin_memcpy(output + first, &tokens, sizeof(tokens));
            if (mirror) __builtin_memcpy(mirror + first, &tokens, sizeof(tokens));
        } else for (int slot = 0; slot < compress; ++slot) {
            const int index = first + slot;
            const int token = block * compress + slot;
            if (index < capacity && token < visible) {
                output[index] = token;
                if (mirror) mirror[index] = token;
            }
        }
    }
    for (int token = tail_start + tid;
         token < visible && offset + token - tail_start < capacity; token += QSA_FILTER_THREADS) {
        output[offset + token - tail_start] = token;
        if (mirror) mirror[offset + token - tail_start] = token;
    }
}
