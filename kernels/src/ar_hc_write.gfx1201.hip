// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#ifndef HF_DEVICE_WAIT_DEADLINE_H
#include "device_wait_deadline.h"
#endif

// gfx1201 TP2 all-reduce fused into the BF16 Hyper-Connection write (the Radiance
// `ar_hc_write` shape; Gate 0b lab kernel `fhc_w1`, `examples/fn_tp2_coll/fused_hc.rs`,
// measured -3.07 ms at 8 rows / -1.87 ms at 1 row vs the EP2-shaped incumbent).
//
// One launch per block output (mixer out/o-proj K-partial, or the FFN partial = shared-down +
// routed combine):
//   (a) round this rank's f32 partial [rows, 2560] to bf16 and push it into the PEER's
//       fine-grained VRAM mailbox bank `tag & 1` as 16-byte tagged granules {d0d1, tag, d2d3,
//       tag} (system-scope write-through b128 stores, one shot);
//   (b) load this lane's stream words and the row gates while the push is in flight;
//   (c) bounded poll of this rank's own mailbox for the peer's granules (same granule range);
//   (d) fold in fixed rank order, `m = 0.0f + (first + second)` of the two bf16-rounded partials
//       (identical bits on both ranks);
//   (e) apply the production `hyper_write_bf16x2` arithmetic to the four BF16 streams in place:
//       `stream' = bf16(bf16(gate * bf16(m)) + stream)`, gate from the row's four gate logits.
//
// Tags. `tag` is the device-resident generation word (`gen_ptr[0]`, single writer: this rank)
// plus one; it skips 0 (the initial mailbox content) by jumping from 0xFFFFFFFF to 2 so the
// bank parity keeps alternating across the wrap. The word advances once per launch: every block
// reads it at the start, takes an arrival ticket (`gen_ptr[16]`) at the end, and the last block
// to arrive stores the new generation and rewinds the ticket. No host write happens per launch,
// so a captured graph replays unchanged kernargs. Both ranks issue the same launches, so their
// generations stay equal; a stale granule of the same bank is two launches old and can never
// carry the current tag (the exchange order proves the peer has consumed bank (g-2)&1).
//
// Bounded wait (device_wait_deadline.h, 500 ms, site AR_HC_WRITE_GFX1201): a stalled peer
// records its site and the generation in the wait block and the block exits before it folds,
// writes the streams or takes its arrival ticket; the host fails the step.
//
// Geometry: `count = rows * 2560` f32 partials, 4 values per granule, `grid <= 64` blocks, up
// to 512 lanes, at most AR_HC_KMAX granules per lane (host: `ar_hc_geometry`). A granule never
// crosses a row (2560 % 4 == 0).
// Operand layout: streams `[rows][4][2560]` bf16 bits, gates `[rows][4]` f32 logits,
// partial `[rows][2560]` f32. `mixed_dbg` (non-null) also receives the folded `m` as f32
// (oracle only; null in every product launch).

#define AR_HC_KMAX 6
#define AR_HC_HIDDEN 2560
#define AR_HC_WIDE 10240
#define AR_HC_TICKET_WORD 16

typedef unsigned int arhc_u4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ unsigned arhc_bf16bits(float f) {
    unsigned u = __float_as_uint(f);
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ float arhc_bf16r(float f) { return __uint_as_float(arhc_bf16bits(f) << 16); }

__device__ __forceinline__ float arhc_rnd(float v) {
    return static_cast<float>(static_cast<hip_bfloat16>(v));
}
__device__ __forceinline__ unsigned arhc_rnd_bits(float v) {
    return (unsigned)static_cast<hip_bfloat16>(v).data;
}
__device__ __forceinline__ float arhc_gate(float logit) {
    const float gate_logit = arhc_rnd(logit);
    const float gate_scaled = arhc_rnd(gate_logit / 4.0f);
    return arhc_rnd(2.0f * arhc_rnd(1.0f / (1.0f + expf(-gate_scaled))));
}
__device__ __forceinline__ float arhc_inj(float g, float m) { return arhc_rnd(g * arhc_rnd(m)); }

__device__ __forceinline__ void arhc_st128_sys(arhc_u4 *p, arhc_u4 v) {
    asm volatile("global_store_b128 %0, %1, off scope:SCOPE_SYS" ::"v"(p), "v"(v) : "memory");
}

// K system-scope b128 loads and their wait in ONE asm statement: the compiler can never reuse a
// destination register while its load is in flight.
template <int K>
__device__ __forceinline__ void arhc_load_batch(arhc_u4 (&r)[K], const arhc_u4 *const (&p)[K]);

template <>
__device__ __forceinline__ void arhc_load_batch<1>(arhc_u4 (&r)[1], const arhc_u4 *const (&p)[1]) {
    asm volatile(
        "global_load_b128 %0, %1, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0])
        : "v"(p[0])
        : "memory");
}
template <>
__device__ __forceinline__ void arhc_load_batch<2>(arhc_u4 (&r)[2], const arhc_u4 *const (&p)[2]) {
    asm volatile(
        "global_load_b128 %0, %2, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %1, %3, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0]), "=&v"(r[1])
        : "v"(p[0]), "v"(p[1])
        : "memory");
}
template <>
__device__ __forceinline__ void arhc_load_batch<3>(arhc_u4 (&r)[3], const arhc_u4 *const (&p)[3]) {
    asm volatile(
        "global_load_b128 %0, %3, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %1, %4, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %2, %5, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0]), "=&v"(r[1]), "=&v"(r[2])
        : "v"(p[0]), "v"(p[1]), "v"(p[2])
        : "memory");
}
template <>
__device__ __forceinline__ void arhc_load_batch<4>(arhc_u4 (&r)[4], const arhc_u4 *const (&p)[4]) {
    asm volatile(
        "global_load_b128 %0, %4, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %1, %5, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %2, %6, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %3, %7, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0]), "=&v"(r[1]), "=&v"(r[2]), "=&v"(r[3])
        : "v"(p[0]), "v"(p[1]), "v"(p[2]), "v"(p[3])
        : "memory");
}
template <>
__device__ __forceinline__ void arhc_load_batch<5>(arhc_u4 (&r)[5], const arhc_u4 *const (&p)[5]) {
    asm volatile(
        "global_load_b128 %0, %5, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %1, %6, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %2, %7, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %3, %8, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %4, %9, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0]), "=&v"(r[1]), "=&v"(r[2]), "=&v"(r[3]), "=&v"(r[4])
        : "v"(p[0]), "v"(p[1]), "v"(p[2]), "v"(p[3]), "v"(p[4])
        : "memory");
}
template <>
__device__ __forceinline__ void arhc_load_batch<6>(arhc_u4 (&r)[6], const arhc_u4 *const (&p)[6]) {
    asm volatile(
        "global_load_b128 %0, %6, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %1, %7, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %2, %8, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %3, %9, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %4, %10, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %5, %11, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0]), "=&v"(r[1]), "=&v"(r[2]), "=&v"(r[3]), "=&v"(r[4]), "=&v"(r[5])
        : "v"(p[0]), "v"(p[1]), "v"(p[2]), "v"(p[3]), "v"(p[4]), "v"(p[5])
        : "memory");
}

// Prefetched operands of one granule's HC write: the four branches' stream words at
// [row, col, col + 4) and the row's four gates. They depend only on local data written by
// earlier launches, so they load while the push is in flight; the apply step after the fold
// performs the production arithmetic unchanged.
//
// Two stream formats, the two production HC writes: BF16 bits (`hyper_write_bf16x2`, the F16
// prefill route, rows >= QWEN4_F16_WMMA_MIN_TOKENS) and F32 words holding BF16-exact values
// (`hyper_write_f32`, decode / few-row verify / short chunks). Same arithmetic per element;
// only the stream load/store width differs (`F32S`).
template <bool F32S>
struct ArhcPre;
template <>
struct ArhcPre<false> {
    uint2 w[4];
    float g[4];
};
template <>
struct ArhcPre<true> {
    uint4 w[4];
    float g[4];
};

template <bool F32S>
__device__ __forceinline__ void arhc_load(const void *__restrict__ streams, const float *__restrict__ gates,
                                          int row, int col, ArhcPre<F32S> &p) {
    const float4 gv = *reinterpret_cast<const float4 *>(gates + (long long)row * 4);
    p.g[0] = arhc_gate(gv.x); p.g[1] = arhc_gate(gv.y); p.g[2] = arhc_gate(gv.z); p.g[3] = arhc_gate(gv.w);
#pragma unroll
    for (int br = 0; br < 4; ++br) {
        const long long e = (long long)row * AR_HC_WIDE + br * AR_HC_HIDDEN + col;
        if constexpr (F32S) {
            p.w[br] = *reinterpret_cast<const uint4 *>(reinterpret_cast<const float *>(streams) + e);
        } else {
            p.w[br] = *reinterpret_cast<const uint2 *>(reinterpret_cast<const unsigned short *>(streams) + e);
        }
    }
}
template <bool F32S>
__device__ __forceinline__ void arhc_apply(void *__restrict__ streams, int row, int col, const ArhcPre<F32S> &p,
                                           const float (&m)[4]) {
#pragma unroll
    for (int br = 0; br < 4; ++br) {
        const long long e = (long long)row * AR_HC_WIDE + br * AR_HC_HIDDEN + col;
        if constexpr (F32S) {
            // hyper_write_f32: bf16(bf16(input) + bf16(gate * bf16(mixed))), stored as a float.
            const unsigned w[4] = {p.w[br].x, p.w[br].y, p.w[br].z, p.w[br].w};
            unsigned ow[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                ow[q] = __float_as_uint(arhc_rnd(arhc_inj(p.g[br], m[q]) + arhc_rnd(__uint_as_float(w[q]))));
            }
            *reinterpret_cast<uint4 *>(reinterpret_cast<float *>(streams) + e) =
                make_uint4(ow[0], ow[1], ow[2], ow[3]);
        } else {
            const unsigned w[2] = {p.w[br].x, p.w[br].y};
            unsigned o[2];
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const float lo = arhc_inj(p.g[br], m[2 * q]) + __uint_as_float(w[q] << 16);
                const float hi = arhc_inj(p.g[br], m[2 * q + 1]) + __uint_as_float(w[q] & 0xFFFF0000u);
                o[q] = arhc_rnd_bits(lo) | (arhc_rnd_bits(hi) << 16);
            }
            *reinterpret_cast<uint2 *>(reinterpret_cast<unsigned short *>(streams) + e) = make_uint2(o[0], o[1]);
        }
    }
}

template <int K, bool F32S>
__device__ __forceinline__ bool arhc_body(
    const float *__restrict__ part, void *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride, unsigned tag,
    hf_device_wait_block *__restrict__ wb, int local_is_first, int g0, int g1, float *__restrict__ mixed_dbg) {
    const int tid = threadIdx.x, T = blockDim.x;
    float pv[K][4];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = g0 + tid + k * T;
        if (j < g1) {
            const float4 a = *reinterpret_cast<const float4 *>(part + (long long)j * 4);
            pv[k][0] = a.x; pv[k][1] = a.y; pv[k][2] = a.z; pv[k][3] = a.w;
        }
    }
    const unsigned bank = tag & 1u;
    arhc_u4 *pb = reinterpret_cast<arhc_u4 *>(peer_mbox + bank * bank_stride);
    const arhc_u4 *lb = reinterpret_cast<const arhc_u4 *>(local_mbox + bank * bank_stride);
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = g0 + tid + k * T;
        if (j < g1) {
            arhc_st128_sys(pb + j, arhc_u4{arhc_bf16bits(pv[k][0]) | (arhc_bf16bits(pv[k][1]) << 16), tag,
                                           arhc_bf16bits(pv[k][2]) | (arhc_bf16bits(pv[k][3]) << 16), tag});
        }
    }
    arhc_u4 r[K];
    const arhc_u4 *lp[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = g0 + tid + k * T;
        lp[k] = lb + (j < g1 ? j : g1 - 1);
    }
    ArhcPre<F32S> pre[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = g0 + tid + k * T;
        if (j < g1) {
            const long long e0 = (long long)j * 4;
            arhc_load<F32S>(streams, gates, (int)(e0 / AR_HC_HIDDEN), (int)(e0 % AR_HC_HIDDEN), pre[k]);
#pragma unroll
            for (int br = 0; br < 4; ++br) {
                asm volatile("" ::"v"(pre[k].w[br].x), "v"(pre[k].w[br].y));
                if constexpr (F32S) asm volatile("" ::"v"(pre[k].w[br].z), "v"(pre[k].w[br].w));
            }
        }
    }
    if (!hf_bounded_wait(
            [&] {
                arhc_load_batch<K>(r, lp);
                bool ok = true;
#pragma unroll
                for (int k = 0; k < K; ++k) {
                    const int j = g0 + tid + k * T;
                    if (j < g1) ok &= (r[k].y == tag) & (r[k].w == tag);
                }
                return ok;
            },
            wb, HF_WAIT_SITE_AR_HC_WRITE_GFX1201, tag)) {
        return false;
    }
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = g0 + tid + k * T;
        if (j < g1) {
            const float in[4] = {__uint_as_float(r[k].x << 16), __uint_as_float(r[k].x & 0xFFFF0000u),
                                 __uint_as_float(r[k].z << 16), __uint_as_float(r[k].z & 0xFFFF0000u)};
            float m[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float loc = arhc_bf16r(pv[k][e]);
                const float fi = local_is_first ? loc : in[e];
                const float si = local_is_first ? in[e] : loc;
                float xi = 0.0f;
                xi = xi + (fi + si);
                m[e] = xi;
            }
            const long long e0 = (long long)j * 4;
            if (mixed_dbg) {
#pragma unroll
                for (int e = 0; e < 4; ++e) mixed_dbg[e0 + e] = m[e];
            }
            arhc_apply<F32S>(streams, (int)(e0 / AR_HC_HIDDEN), (int)(e0 % AR_HC_HIDDEN), pre[k], m);
        }
    }
    return true;
}

template <bool F32S>
__device__ __forceinline__ void arhc_entry(
    const float *__restrict__ part, void *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int count, float *__restrict__ mixed_dbg) {
    const int G = count / 4;
    const int per = (G + (int)gridDim.x - 1) / (int)gridDim.x;
    const int g0 = (int)blockIdx.x * per;
    const int g1 = min(G, g0 + per);
    unsigned tag = __hip_atomic_load(gen_ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (tag == 0u) tag = 2u;
    bool ok = true;
    if (g0 < g1) {
        const int K = (g1 - g0 + (int)blockDim.x - 1) / (int)blockDim.x;
        switch (K) {
        case 1: ok = arhc_body<1, F32S>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1, mixed_dbg); break;
        case 2: ok = arhc_body<2, F32S>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1, mixed_dbg); break;
        case 3: ok = arhc_body<3, F32S>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1, mixed_dbg); break;
        case 4: ok = arhc_body<4, F32S>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1, mixed_dbg); break;
        case 5: ok = arhc_body<5, F32S>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1, mixed_dbg); break;
        case 6: ok = arhc_body<6, F32S>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1, mixed_dbg); break;
        default: ok = false; break;
        }
    }
    // Every lane of the block has read the generation word before this barrier (it is the
    // first statement), so the single advance below cannot race a reader of this launch.
    // A timed-out wait never reaches it: the generation stays put and the host fails the step.
    __syncthreads();
    __shared__ int sh_ok;
    if (threadIdx.x == 0) sh_ok = 0;
    __syncthreads();
    if (!ok) sh_ok = 1;
    __syncthreads();
    if (threadIdx.x == 0 && sh_ok == 0) {
        const unsigned old = __hip_atomic_fetch_add(gen_ptr + AR_HC_TICKET_WORD, 1u, __ATOMIC_RELAXED,
                                                    __HIP_MEMORY_SCOPE_AGENT);
        if (old == gridDim.x - 1u) {
            __hip_atomic_store(gen_ptr + AR_HC_TICKET_WORD, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(gen_ptr, tag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

// BF16-bit streams (`[rows][4][2560]` u16).
extern "C" __global__ __launch_bounds__(512) void ar_hc_write_gfx1201(
    const float *__restrict__ part, unsigned short *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int count, float *__restrict__ mixed_dbg) {
    arhc_entry<false>(part, streams, gates, peer_mbox, local_mbox, bank_stride, gen_ptr, wb, local_is_first, count,
                      mixed_dbg);
}

// F32 streams (`[rows][4][2560]` f32 holding BF16-exact values), same ABI.
extern "C" __global__ __launch_bounds__(512) void ar_hc_write_f32s_gfx1201(
    const float *__restrict__ part, float *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int count, float *__restrict__ mixed_dbg) {
    arhc_entry<true>(part, streams, gates, peer_mbox, local_mbox, bank_stride, gen_ptr, wb, local_is_first, count,
                     mixed_dbg);
}
