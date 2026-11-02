// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Diagnostic-only device references of the host CPU-AR operation order
// (crates/hipfire-runtime/src/spec_sampling.rs), not exact-host probability twins:
//
//   spec_cpu_ar_rows_f32        SparseDist::build_from_logits   (target p rows)
//   spec_cpu_ar_candidates_f32  SparseDist::build_from_candidates + sample(u)
//   spec_accept_ordered_u32     verify_sampled_draft / accept_sampled_prefix
//
// Every probability is f32 and every operation is performed in the host
// program's order, serially (one thread per row / per launch). The literal
// serial code and API are unchanged. Nothing fuses: fp-contract is off and
// each mul/add/sub/div adjacent to another operation is an explicit
// round-to-nearest intrinsic; `exp` is plain HIP `expf`. The CPU probability
// bit-equality gate was retired after the first gfx1201 synthetic row differed
// at token 42939: GPU 0x3d66ae67 versus CPU 0x3d66ae68 (1 ULP). This reference
// is diagnostic-only; no exact-host probability claim is made. The canonical
// PM candidate still requires its twinHW gate against the HIP twin on gfx1151
// and gfx1201; retiring the CPU comparison establishes no PM/HIP parity or
// performance claim.
//
// Dist rows are fixed-capacity 64: counts[row] u32, ids[row*64+j] u32,
// probs[row*64+j] f32, in SparseDist::entries() order (descending
// probability, NOT ascending token). Slots past counts[row] are written as 0.

#include <hip/hip_runtime.h>
#include <stdint.h>

#pragma clang fp contract(off)

#define SCA_CAP 64

static __device__ __forceinline__ float sca_max0(float v) {
    // Rust f32::max(v, 0.0): NaN loses.
    return fmaxf(v, 0.0f);
}

// ===========================================================================
// spec_cpu_ar_rows_f32
//
// Serial SparseDist::build_from_logits, one thread per row. The pool slots
// (token id then logit, 128 f32 words per row) live in `scratch`.
//
// Kernarg order: logits (f32 [rows*vocab]), counts (u32 [rows]),
// ids (u32 [rows*64]), probs (f32 [rows*64]), scratch (f32 [rows*128]),
// vocab i32, pool i32, cap i32, temperature f32, top_p f32, min_p f32.
// Grid [rows,1,1], block 1.
// ===========================================================================
extern "C" __global__ __launch_bounds__(1) void spec_cpu_ar_rows_f32(
    const float* __restrict__ logits,
    uint32_t* __restrict__ counts,
    uint32_t* __restrict__ ids,
    float* __restrict__ probs,
    float* __restrict__ scratch,
    int vocab,
    int pool,
    int cap_in,
    float temperature,
    float top_p,
    float min_p)
{
    if (threadIdx.x != 0) return;
    const int row = blockIdx.x;
    const float* L = logits + (size_t)row * (size_t)vocab;
    uint32_t* slot_tok = (uint32_t*)(scratch + (size_t)row * 128);
    float* slot_val = scratch + (size_t)row * 128 + SCA_CAP;
    uint32_t* out_ids = ids + (size_t)row * SCA_CAP;
    float* out_p = probs + (size_t)row * SCA_CAP;
    const float NEG_INF = __uint_as_float(0xff800000u);
    const float POS_INF = __uint_as_float(0x7f800000u);
    if (pool < 1) pool = 1;
    if (pool > SCA_CAP) pool = SCA_CAP;

    // sample_pool's fixed slots: each finite logit above the current minimum
    // replaces it, then the minimum is rescanned.
    for (int j = 0; j < pool; ++j) { slot_tok[j] = 0u; slot_val[j] = NEG_INF; }
    int min_pos = 0;
    float min_val = NEG_INF;
    float max_logit = NEG_INF;
    for (int i = 0; i < vocab; ++i) {
        const float l = L[i];
        if (!isfinite(l)) continue;
        if (l > max_logit) max_logit = l;
        if (l > min_val) {
            slot_tok[min_pos] = (uint32_t)i;
            slot_val[min_pos] = l;
            min_val = POS_INF;
            for (int j = 0; j < pool; ++j) {
                const float v = slot_val[j];
                if (v < min_val) { min_val = v; min_pos = j; }
            }
        }
    }

    float ep[SCA_CAP];
    uint32_t et[SCA_CAP];
    const float inv_temp = __fdiv_rn(1.0f, temperature);
    float sum = 0.0f;
    for (int j = 0; j < pool; ++j) {
        const float l = slot_val[j];
        float p = 0.0f;
        if (isfinite(l)) {
            p = expf(__fmul_rn(__fadd_rn(l, -max_logit), inv_temp));
            if (!isfinite(p)) p = 0.0f;
        }
        ep[j] = p;
        et[j] = slot_tok[j];
        sum = __fadd_rn(sum, p);
    }
    int n;
    if (sum <= 0.0f || !isfinite(sum)) {
        // argmax fallback (llama::argmax): first strictly-greater finite.
        int best = 0;
        float bv = NEG_INF;
        for (int i = 0; i < vocab; ++i) {
            const float v = L[i];
            if (isfinite(v) && v > bv) { bv = v; best = i; }
        }
        et[0] = (uint32_t)best;
        ep[0] = 1.0f;
        n = 1;
    } else {
        // stable insertion sort, descending
        for (int i = 1; i < pool; ++i) {
            int j = i;
            while (j > 0 && ep[j] > ep[j - 1]) {
                const float tp = ep[j]; ep[j] = ep[j - 1]; ep[j - 1] = tp;
                const uint32_t tt = et[j]; et[j] = et[j - 1]; et[j - 1] = tt;
                --j;
            }
        }
        int cap = cap_in;
        if (cap < 1) cap = 1;
        if (cap > pool) cap = pool;
        if (min_p > 0.0f) {
            const float floor_p = __fmul_rn(min_p, ep[0]);
            for (int i = 1; i < cap; ++i) {
                if (ep[i] < floor_p) { cap = i; break; }
            }
        }
        // The uncut pool keeps the slot-order sum; a cut sums the kept prefix.
        if (cap < pool) {
            float s2 = 0.0f;
            for (int i = 0; i < cap; ++i) s2 = __fadd_rn(s2, ep[i]);
            sum = s2;
        }
        const float threshold = __fmul_rn(top_p, sum);
        float cumulative = 0.0f;
        int boundary = -1;
        for (int i = 0; i < cap; ++i) {
            cumulative = __fadd_rn(cumulative, ep[i]);
            if (cumulative >= threshold) { boundary = i + 1; break; }
        }
        if (boundary >= 0) {
            n = boundary;
            for (int i = 0; i < n; ++i) ep[i] = __fdiv_rn(ep[i], cumulative);
        } else {
            n = cap;
            for (int i = 0; i < n; ++i) ep[i] = __fdiv_rn(ep[i], sum);
            ep[0] = __fadd_rn(ep[0], sca_max0(__fadd_rn(1.0f, -__fdiv_rn(cumulative, sum))));
        }
        int m = 0; // retain p > 0, order preserved
        for (int i = 0; i < n; ++i) {
            if (ep[i] > 0.0f) { ep[m] = ep[i]; et[m] = et[i]; ++m; }
        }
        n = m;
    }
    counts[row] = (uint32_t)n;
    for (int j = 0; j < SCA_CAP; ++j) {
        out_ids[j] = j < n ? et[j] : 0u;
        out_p[j] = j < n ? ep[j] : 0.0f;
    }
}

// ===========================================================================
// spec_cpu_ar_candidates_f32
//
// Draft q row from <=8 (id, logit) candidates plus the inverse-CDF draw.
// candidates: 16 u32 words, [0,8) token ids, [8,16) f32 logit bits; slots with
// a non-finite logit are dropped. On "no finite candidate" / "no finite
// mass" (the host's Err) counts[row] = 0 and picked[0] = 0xFFFFFFFF.
//
// Kernarg order: candidates, counts, ids, probs, picked, row i32, cap i32,
// temperature f32, top_p f32, min_p f32, uniform f32.
// Grid [1,1,1], block 1.
// ===========================================================================
static __device__ __forceinline__ int sca_total_key(float f) {
    int i = __float_as_int(f);
    return i ^ (int)(((uint32_t)(i >> 31)) >> 1);
}

extern "C" __global__ __launch_bounds__(1) void spec_cpu_ar_candidates_f32(
    const uint32_t* __restrict__ candidates,
    uint32_t* __restrict__ counts,
    uint32_t* __restrict__ ids,
    float* __restrict__ probs,
    uint32_t* __restrict__ picked,
    int row,
    int cap_in,
    float temperature,
    float top_p,
    float min_p,
    float uniform)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint32_t tok[8];
    float lg[8];
    float p[8];
    int n = 0;
    for (int i = 0; i < 8; ++i) {
        float l = __uint_as_float(candidates[8 + i]);
        if (isfinite(l)) { tok[n] = candidates[i]; lg[n] = l; ++n; }
    }
    // sort: logit descending by total order, then token ascending
    for (int i = 1; i < n; ++i) {
        float l = lg[i];
        uint32_t t = tok[i];
        int j = i;
        while (j > 0) {
            int kj = sca_total_key(lg[j - 1]);
            int ki = sca_total_key(l);
            bool before = ki > kj || (ki == kj && t < tok[j - 1]);
            if (!before) break;
            lg[j] = lg[j - 1];
            tok[j] = tok[j - 1];
            --j;
        }
        lg[j] = l;
        tok[j] = t;
    }
    int cap = cap_in < 1 ? 1 : cap_in;
    if (n > cap) n = cap;
    bool fail = (n == 0);
    uint32_t* out_ids = ids + (size_t)row * SCA_CAP;
    float* out_p = probs + (size_t)row * SCA_CAP;
    uint32_t pick = 0xffffffffu;
    int len = 0;
    if (!fail) {
        const float mx = lg[0];
        const float inv_temp = __fdiv_rn(1.0f, temperature);
        for (int i = 0; i < n; ++i) {
            p[i] = expf(__fmul_rn(__fadd_rn(lg[i], -mx), inv_temp));
        }
        if (min_p > 0.0f) {
            float floor_p = __fmul_rn(min_p, p[0]);
            int m = 0;
            for (int i = 0; i < n; ++i) {
                if (p[i] >= floor_p) { p[m] = p[i]; tok[m] = tok[i]; ++m; }
            }
            n = m;
        }
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) sum = __fadd_rn(sum, p[i]);
        if (!(sum > 0.0f) || !isfinite(sum)) {
            fail = true;
        } else {
            const float threshold = __fmul_rn(top_p, sum);
            float kept = 0.0f;
            len = n;
            for (int i = 0; i < n; ++i) {
                kept = __fadd_rn(kept, p[i]);
                if (kept >= threshold) { len = i + 1; break; }
            }
            float mass = 0.0f;
            for (int i = 0; i < len; ++i) mass = __fadd_rn(mass, p[i]);
            for (int i = 0; i < len; ++i) p[i] = __fdiv_rn(p[i], mass);
            // SparseDist::sample(u)
            float acc = 0.0f;
            pick = len > 0 ? tok[len - 1] : 0u;
            for (int i = 0; i < len; ++i) {
                acc = __fadd_rn(acc, p[i]);
                if (uniform < acc) { pick = tok[i]; break; }
            }
        }
    }
    if (fail) len = 0;
    counts[row] = (uint32_t)len;
    for (int j = 0; j < SCA_CAP; ++j) {
        out_ids[j] = j < len ? tok[j] : 0u;
        out_p[j] = j < len ? p[j] : 0.0f;
    }
    picked[0] = fail ? 0xffffffffu : pick;
}

// ===========================================================================
// spec_accept_ordered_u32
//
// accept_sampled_prefix over resident rows. Target rows 0..=k, draft rows
// 0..k. uniforms[j] is row j's verdict draw, uniforms[j+1] the residual draw
// when row j rejects, uniforms[k] the bonus draw. eos = 0xFFFFFFFF: none.
// result[0] = accepted drafts, result[1] = terminal token (the residual
// replacement or bonus), or 0xFFFFFFFF when the last accepted draft was EOS.
//
// Kernarg order: tcounts, tids, tprobs, dcounts, dids, dprobs, proposals,
// uniforms, result, k i32, eos u32. Grid [1,1,1], block 1.
// ===========================================================================
static __device__ __forceinline__ float sca_prob(
    const uint32_t* counts, const uint32_t* ids, const float* probs, int row, uint32_t token)
{
    const int n = (int)counts[row];
    const uint32_t* rid = ids + (size_t)row * SCA_CAP;
    const float* rp = probs + (size_t)row * SCA_CAP;
    for (int i = 0; i < n; ++i) {
        if (rid[i] == token) return rp[i];
    }
    return 0.0f;
}

static __device__ __forceinline__ uint32_t sca_sample(
    const uint32_t* counts, const uint32_t* ids, const float* probs, int row, float u)
{
    const int n = (int)counts[row];
    const uint32_t* rid = ids + (size_t)row * SCA_CAP;
    const float* rp = probs + (size_t)row * SCA_CAP;
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc = __fadd_rn(acc, rp[i]);
        if (u < acc) return rid[i];
    }
    return n > 0 ? rid[n - 1] : 0u;
}

extern "C" __global__ __launch_bounds__(1) void spec_accept_ordered_u32(
    const uint32_t* __restrict__ tcounts,
    const uint32_t* __restrict__ tids,
    const float* __restrict__ tprobs,
    const uint32_t* __restrict__ dcounts,
    const uint32_t* __restrict__ dids,
    const float* __restrict__ dprobs,
    const uint32_t* __restrict__ proposals,
    const float* __restrict__ uniforms,
    uint32_t* __restrict__ result,
    int k,
    uint32_t eos)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int row = 0; row < k; ++row) {
        const uint32_t draft = proposals[row];
        const float qx = sca_prob(dcounts, dids, dprobs, row, draft);
        const float px = sca_prob(tcounts, tids, tprobs, row, draft);
        const float u = uniforms[row];
        if (qx > 0.0f && __fmul_rn(u, qx) < px) {
            if (draft == eos) {
                result[0] = (uint32_t)(row + 1);
                result[1] = 0xffffffffu;
                return;
            }
            continue;
        }
        // sample_residual(p, q, u2)
        const float u2 = uniforms[row + 1];
        const int n = (int)tcounts[row];
        const uint32_t* pid = tids + (size_t)row * SCA_CAP;
        const float* pp = tprobs + (size_t)row * SCA_CAP;
        float mass = 0.0f;
        for (int i = 0; i < n; ++i) {
            float r = sca_max0(__fadd_rn(pp[i], -sca_prob(dcounts, dids, dprobs, row, pid[i])));
            mass = __fadd_rn(mass, r);
        }
        uint32_t token;
        if (!(mass > 0.0f)) {
            token = sca_sample(tcounts, tids, tprobs, row, u2);
        } else {
            const float target = __fmul_rn(u2, mass);
            float acc = 0.0f;
            bool have_last = false;
            uint32_t last = 0u;
            bool done = false;
            token = 0u;
            for (int i = 0; i < n; ++i) {
                float r = sca_max0(__fadd_rn(pp[i], -sca_prob(dcounts, dids, dprobs, row, pid[i])));
                if (r <= 0.0f) continue;
                acc = __fadd_rn(acc, r);
                if (target < acc) { token = pid[i]; done = true; break; }
                last = pid[i];
                have_last = true;
            }
            if (!done) token = have_last ? last : sca_sample(tcounts, tids, tprobs, row, u2);
        }
        result[0] = (uint32_t)row;
        result[1] = token;
        return;
    }
    // all accepted: bonus from p at row k
    result[0] = (uint32_t)k;
    result[1] = sca_sample(tcounts, tids, tprobs, k, uniforms[k]);
}
