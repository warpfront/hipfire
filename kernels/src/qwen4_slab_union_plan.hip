// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Device-side lane-union planner of the sealed expert-slab decode (K1,
// `mq_i4_moe_gemv`). One workgroup turns the call's routed ids
// (`topk[S]`, `S = 10 * lanes`, slot = lane * 10 + rank) into the fixed-capacity
// per-codec-bucket descriptors the four K1 symbols read, with no host readback:
// the bucket counts are data dependent, so every bucket is laid out at its
// worst-case capacity `S` and its unused tail is empty (`n = 0` experts, which
// the K1 kernels skip).
//
// Buckets (K1's fixed codec order): 0 = W3 (codec byte 1), 1 = MQ4 (codec byte
// 0). `uniform_bucket >= 0` puts every slot in that bucket and ignores the
// codec table (a layer whose experts all share one codec carries no table).
//
// Output words (u32), for bucket b in {0, 1}:
//   expert_ids [b * S + e]            distinct experts of the bucket in
//                                      first-occurrence order, tail 0
//   expert_offsets [2 S + b (S+1) + e] run of S + 1 offsets from 0 to the
//                                      bucket's slot count n_b; tail = n_b
//   slot_ids [2 S + 2 (S+1) + b S + p] original slot ids, grouped by expert in
//                                      ascending slot order, tail 0
// Total 6 S + 2 words. Every word is written exactly once per invocation, by
// exactly one thread (single-owner map, with U_b = distinct experts and n_b =
// slots of bucket b, t = thread = original slot, f = first-occurrence slot of
// t's expert, e = s_eidx[f] the expert's index in its bucket):
//   ids[b S + e], e < U_b          the first-occurrence thread of expert e
//   ids[b S + t], t >= U_b         thread t (tail)
//   offs[b (S+1) + e], e < U_b     the first-occurrence thread of expert e
//                                  (the start of its run; the previous
//                                  expert's end is NOT written separately)
//   offs[b (S+1) + t], t >= U_b,
//                      t < S       thread t (tail = n_b; offs[b (S+1) + U_b]
//                                  = n_b is owned by tail thread U_b, or by
//                                  thread 0 when U_b = S)
//   offs[b (S+1) + S]              thread 0 (= n_b), for both buckets
//   slots[b S + p], p < n_b        the thread whose (start + rank) is p
//   slots[b S + t], t >= n_b       thread t (tail)
//
// Capacity: S = 10 * lanes <= SLAB_PLAN_MAX_SLOTS = 320 (32 lanes). One
// thread per slot, so the launch is SLAB_PLAN_THREADS = 512 threads (one
// workgroup, 5 * 320 * 4 = 6400 B LDS); threads t >= S are inactive but still
// reach every barrier. S outside 1..=320 returns uniformly before any
// barrier and writes nothing (the host refuses such a call before launch).
//
// Unrouted slots: an id outside [0, n_experts) (the router's -1 marker for a
// non-finite row, or any corrupt id) joins no bucket, no expert, no offset run
// and no slot list, and `codec` is never read for it. The planner never
// substitutes another expert for it; the K1 kernels therefore load no weights
// for it. The tails are sized from the live slots only (n_b, U_b above count
// live slots only; an unrouted thread owns no head word and its tail words are
// written by the tail rules like any other thread's).

#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__) && !defined(__gfx1151__)
#error "qwen4_slab_union_plan.hip is gfx1201/gfx1151-only"
#endif

#define SLAB_PLAN_MAX_SLOTS 320
#define SLAB_PLAN_THREADS 512
static_assert(SLAB_PLAN_MAX_SLOTS <= SLAB_PLAN_THREADS, "one thread per slot");

extern "C" __global__ __launch_bounds__(SLAB_PLAN_THREADS) void qwen4_slab_union_plan(
    const int* __restrict__ topk, const unsigned char* __restrict__ codec,
    unsigned* __restrict__ out, int S, int uniform_bucket, int n_experts) {
    __shared__ int s_id[SLAB_PLAN_MAX_SLOTS];
    __shared__ int s_b[SLAB_PLAN_MAX_SLOTS];
    __shared__ int s_first[SLAB_PLAN_MAX_SLOTS];
    __shared__ int s_total[SLAB_PLAN_MAX_SLOTS];
    __shared__ int s_eidx[SLAB_PLAN_MAX_SLOTS];
    if (S < 1 || S > SLAB_PLAN_MAX_SLOTS) return;  // kernarg-uniform: no barrier divergence
    const int t = threadIdx.x;
    bool live = false;
    if (t < S) {
        const int id = topk[t];
        live = id >= 0 && id < n_experts;
        s_id[t] = live ? id : -1;
        s_b[t] = live ? (uniform_bucket >= 0 ? uniform_bucket : (codec[id] == 1 ? 0 : 1)) : -1;
    }
    __syncthreads();
    int rank = 0, total = 0, f = t;
    if (t < S) {
        const int id = s_id[t];
        f = -1;
        if (live) {
            for (int j = 0; j < S; ++j) {
                if (s_id[j] == id) {
                    if (f < 0) f = j;
                    ++total;
                    if (j < t) ++rank;
                }
            }
        }
        s_first[t] = live && rank == 0;
        s_total[t] = total;
    }
    __syncthreads();
    if (t < S && live && rank == 0) {
        const int b = s_b[t];
        int e = 0;
        for (int j = 0; j < t; ++j) e += s_first[j] && s_b[j] == b;
        s_eidx[t] = e;
    }
    __syncthreads();
    unsigned* ids = out;
    unsigned* offs = out + 2 * S;
    unsigned* slots = out + 2 * S + 2 * (S + 1);
    int ub0 = 0, ub1 = 0, nb0 = 0, nb1 = 0;
    for (int j = 0; j < S; ++j) {
        if (s_first[j]) {
            if (s_b[j] == 0) { ++ub0; nb0 += s_total[j]; } else { ++ub1; nb1 += s_total[j]; }
        }
    }
    if (t < S) {
        if (live) {
            const int b = s_b[t];
            const int e = s_eidx[f];
            int start = 0;
            for (int j = 0; j < S; ++j) {
                if (s_first[j] && s_b[j] == b && s_eidx[j] < e) start += s_total[j];
            }
            slots[b * S + start + rank] = (unsigned)t;
            if (rank == 0) {
                ids[b * S + e] = (unsigned)s_id[t];
                offs[b * (S + 1) + e] = (unsigned)start;
            }
        }
        if (t >= ub0) { ids[t] = 0u; offs[t] = (unsigned)nb0; }
        if (t >= ub1) { ids[S + t] = 0u; offs[(S + 1) + t] = (unsigned)nb1; }
        if (t >= nb0) slots[t] = 0u;
        if (t >= nb1) slots[S + t] = 0u;
    }
    // Final offset of each bucket run (offs[S] = n_b); no other thread writes it.
    if (t == 0) { offs[S] = (unsigned)nb0; offs[(S + 1) + S] = (unsigned)nb1; }
}
