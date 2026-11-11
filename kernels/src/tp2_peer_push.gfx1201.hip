// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#ifndef HF_DEVICE_WAIT_DEADLINE_H
#include "device_wait_deadline.h"
#endif

// Exact gfx1201 TP2 graph-resident residual reduction through tagged peer-VRAM
// granules. A rank pushes its f32 partial straight into the PEER's fine-grained
// VRAM mailbox (`hipDeviceMallocFinegrained`, allocated before peer access is
// enabled) and polls only its own mailbox: no separate flag, no fence, no
// cross-device read, no cross-device read-modify-write.
//
// Wire format: one 16-byte granule carries two partial floats and the
// generation tag twice, {d0, g, d1, g}. Granules are written with system-scope
// write-through `global_store_b128` and polled with system-scope
// `global_load_b128` (inline asm: there is no 128-bit system-scope atomic in
// HIP). A granule is accepted only when BOTH tags equal the current generation,
// so each 8-byte half validates itself and a torn 16-byte write is rejected.
// An aligned 8-byte word is single-copy atomic on this path; a stale granule
// from an earlier generation of the same bank can never carry the current tag.
//
// Two banks alternate with the generation parity. Both ranks exchange once per
// collective, so a rank that saw its peer's granules for generation g knows the
// peer finished the collective g-1 and therefore every read of bank (g-2)&1;
// no consumed flag is needed. The generation is a device-resident counter
// (single writer: the owning rank), so a captured graph replays unchanged
// kernargs and the counter advances on every replay. Tag 0 is never used (it is
// the initial mailbox content); the counter skips from 0xFFFFFFFF to 2 so bank
// parity stays alternating across the wrap.
//
// The arithmetic is the incumbent's: x[i] = x[i] + (p0[i] + p1[i]) in rank
// order, fp32 adds only, so the result is bit-identical to
// `tp_graph_signal_reduce_gfx1100`. The wait is bounded (device_wait_deadline.h):
// a stalled peer leaves HF_WAIT_SITE_TP_PEER_PUSH_GFX1201 and the generation in
// the wait block, and the wave exits before folding or advancing the counter.
//
// One workgroup of exactly 512 lanes; count must be even and at most 8192
// (granules <= 8 per lane); the host checks both. The partial and the residual
// are loaded before the first dependent step so their latency overlaps the
// generation load and the peer wait.

typedef unsigned int hf_peer_u4 __attribute__((ext_vector_type(4)));

#define HF_PEER_PUSH_LANES 512
#define HF_PEER_PUSH_KMAX 8

__device__ __forceinline__ void hf_peer_st128_sys(hf_peer_u4 *p, hf_peer_u4 v) {
    asm volatile("global_store_b128 %0, %1, off scope:SCOPE_SYS" ::"v"(p), "v"(v) : "memory");
}

// Issue K system-scope b128 loads and wait for them in ONE asm statement, so
// the compiler can never copy or reuse a destination register while its load
// is still in flight (a separate wait asm would let it).
template <int K>
__device__ __forceinline__ void hf_peer_load_batch(hf_peer_u4 (&r)[K], const hf_peer_u4 *const (&p)[K]);

template <>
__device__ __forceinline__ void hf_peer_load_batch<1>(hf_peer_u4 (&r)[1], const hf_peer_u4 *const (&p)[1]) {
    asm volatile(
        "global_load_b128 %0, %1, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0])
        : "v"(p[0])
        : "memory");
}

template <>
__device__ __forceinline__ void hf_peer_load_batch<2>(hf_peer_u4 (&r)[2], const hf_peer_u4 *const (&p)[2]) {
    asm volatile(
        "global_load_b128 %0, %2, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %1, %3, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0]), "=&v"(r[1])
        : "v"(p[0]), "v"(p[1])
        : "memory");
}

template <>
__device__ __forceinline__ void hf_peer_load_batch<3>(hf_peer_u4 (&r)[3], const hf_peer_u4 *const (&p)[3]) {
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
__device__ __forceinline__ void hf_peer_load_batch<4>(hf_peer_u4 (&r)[4], const hf_peer_u4 *const (&p)[4]) {
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
__device__ __forceinline__ void hf_peer_load_batch<5>(hf_peer_u4 (&r)[5], const hf_peer_u4 *const (&p)[5]) {
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
__device__ __forceinline__ void hf_peer_load_batch<6>(hf_peer_u4 (&r)[6], const hf_peer_u4 *const (&p)[6]) {
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

template <>
__device__ __forceinline__ void hf_peer_load_batch<7>(hf_peer_u4 (&r)[7], const hf_peer_u4 *const (&p)[7]) {
    asm volatile(
        "global_load_b128 %0, %7, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %1, %8, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %2, %9, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %3, %10, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %4, %11, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %5, %12, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %6, %13, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0]), "=&v"(r[1]), "=&v"(r[2]), "=&v"(r[3]), "=&v"(r[4]), "=&v"(r[5]), "=&v"(r[6])
        : "v"(p[0]), "v"(p[1]), "v"(p[2]), "v"(p[3]), "v"(p[4]), "v"(p[5]), "v"(p[6])
        : "memory");
}

template <>
__device__ __forceinline__ void hf_peer_load_batch<8>(hf_peer_u4 (&r)[8], const hf_peer_u4 *const (&p)[8]) {
    asm volatile(
        "global_load_b128 %0, %8, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %1, %9, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %2, %10, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %3, %11, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %4, %12, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %5, %13, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %6, %14, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %7, %15, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0]), "=&v"(r[1]), "=&v"(r[2]), "=&v"(r[3]), "=&v"(r[4]), "=&v"(r[5]), "=&v"(r[6]), "=&v"(r[7])
        : "v"(p[0]), "v"(p[1]), "v"(p[2]), "v"(p[3]), "v"(p[4]), "v"(p[5]), "v"(p[6]), "v"(p[7])
        : "memory");
}

template <int K>
__device__ __forceinline__ void tp2_peer_push_body(
    float *__restrict__ x, const float *__restrict__ part,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wait_block,
    int local_is_first, int granules) {
    const int tid = threadIdx.x;
    const float2 *__restrict__ part2 = reinterpret_cast<const float2 *>(part);
    float2 *__restrict__ x2 = reinterpret_cast<float2 *>(x);
    float2 pv[K], xr[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = tid + k * HF_PEER_PUSH_LANES;
        if (j < granules) {
            pv[k] = part2[j];
            xr[k] = x2[j];
        }
    }
    unsigned g = __hip_atomic_load(gen_ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (g == 0u) g = 2u;
    const unsigned bank = g & 1u;
    hf_peer_u4 *pb = reinterpret_cast<hf_peer_u4 *>(peer_mbox + bank * bank_stride);
    const hf_peer_u4 *lb = reinterpret_cast<const hf_peer_u4 *>(local_mbox + bank * bank_stride);
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = tid + k * HF_PEER_PUSH_LANES;
        if (j < granules) {
            hf_peer_st128_sys(
                pb + j, hf_peer_u4{__float_as_uint(pv[k].x), g, __float_as_uint(pv[k].y), g});
        }
    }
    hf_peer_u4 r[K];
    // Out-of-range granules read the last valid one (result ignored): the
    // batch asm needs K valid addresses.
    const hf_peer_u4 *lp[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = tid + k * HF_PEER_PUSH_LANES;
        lp[k] = lb + (j < granules ? j : granules - 1);
    }
    if (!hf_bounded_wait(
            [&] {
                hf_peer_load_batch<K>(r, lp);
                bool ok = true;
#pragma unroll
                for (int k = 0; k < K; ++k) {
                    const int j = tid + k * HF_PEER_PUSH_LANES;
                    if (j < granules) ok &= (r[k].y == g) & (r[k].w == g);
                }
                return ok;
            },
            wait_block, HF_WAIT_SITE_TP_PEER_PUSH_GFX1201, g)) {
        return;
    }
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = tid + k * HF_PEER_PUSH_LANES;
        if (j < granules) {
            float2 xi = xr[k];
            const float2 loc = pv[k];
            const float2 in = make_float2(__uint_as_float(r[k].x), __uint_as_float(r[k].z));
            const float2 fi = local_is_first ? loc : in;
            const float2 si = local_is_first ? in : loc;
            xi.x = xi.x + (fi.x + si.x);
            xi.y = xi.y + (fi.y + si.y);
            x2[j] = xi;
        }
    }
    // Every lane loaded the generation before this barrier, so the single
    // write below cannot race a reader.
    __syncthreads();
    if (tid == 0) __hip_atomic_store(gen_ptr, g, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

extern "C" __global__ __launch_bounds__(512) void tp2_peer_push_reduce_gfx1201(
    float *__restrict__ x, const float *__restrict__ local_partial,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wait_block,
    int local_is_first, int count) {
    if (blockIdx.x != 0) return;
    const int granules = count >> 1;
    const int k = (granules + HF_PEER_PUSH_LANES - 1) / HF_PEER_PUSH_LANES;
    switch (k) {
    case 1: tp2_peer_push_body<1>(x, local_partial, peer_mbox, local_mbox, bank_stride, gen_ptr, wait_block, local_is_first, granules); break;
    case 2: tp2_peer_push_body<2>(x, local_partial, peer_mbox, local_mbox, bank_stride, gen_ptr, wait_block, local_is_first, granules); break;
    case 3: tp2_peer_push_body<3>(x, local_partial, peer_mbox, local_mbox, bank_stride, gen_ptr, wait_block, local_is_first, granules); break;
    case 4: tp2_peer_push_body<4>(x, local_partial, peer_mbox, local_mbox, bank_stride, gen_ptr, wait_block, local_is_first, granules); break;
    case 5: tp2_peer_push_body<5>(x, local_partial, peer_mbox, local_mbox, bank_stride, gen_ptr, wait_block, local_is_first, granules); break;
    case 6: tp2_peer_push_body<6>(x, local_partial, peer_mbox, local_mbox, bank_stride, gen_ptr, wait_block, local_is_first, granules); break;
    case 7: tp2_peer_push_body<7>(x, local_partial, peer_mbox, local_mbox, bank_stride, gen_ptr, wait_block, local_is_first, granules); break;
    default: tp2_peer_push_body<HF_PEER_PUSH_KMAX>(x, local_partial, peer_mbox, local_mbox, bank_stride, gen_ptr, wait_block, local_is_first, granules); break;
    }
}
