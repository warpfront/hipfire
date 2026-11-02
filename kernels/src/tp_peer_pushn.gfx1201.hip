// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#ifndef HF_DEVICE_WAIT_DEADLINE_H
#include "device_wait_deadline.h"
#endif
#ifndef HF_TP_EXCHANGE16_H
#include "tp_exchange16.h"
#endif

// gfx1201 TP3/TP4 graph-resident residual reduction with a bf16 exchange
// through tagged peer-VRAM granules. Same protocol as
// `tp2_peer_push_reduce_gfx1201` (see that file for the memory model and the
// bank/generation argument), generalised to every rank pushing to every peer:
//
//   rank r writes its bf16-packed partial into slot r of each peer's mailbox
//   bank (b = g & 1) as 16-byte granules {w0, g, w1, g} (four bf16 values and
//   the generation tag twice), polls its own mailbox for the other ranks' slots,
//   and folds
//
//       sum = r(p_0) + r(p_1) + ... + r(p_{n-1})   (left-associated, rank order)
//       x   = x + sum
//
//   with r() the bf16 round trip applied to every term including the rank's own
//   (computed in register from the same bits it publishes). This is the
//   arithmetic of `tp_pack16_bf16` + `tp_fold16_bf16` (tp_exchange16.hip), so
//   the residual is bit-identical to the event/copy route.
//
// Every rank exchanges with every peer once per collective, so a rank that saw
// all of its peers' granules for generation g knows each peer finished the
// collective g-1 (and with it every read of bank (g-2)&1): two banks, no
// consumed flag. Tag 0 is never used; the counter skips from 0xFFFFFFFF to 2.
//
// One workgroup of exactly 512 lanes. count must be a multiple of 4 and at most
// 8192 (4 granules per lane); rank < nranks <= 4, nranks >= 3 (the TP2 f32 route
// has its own kernel). Mailbox peers are passed as four rank-indexed pointers
// (the own entry is the local mailbox; entries >= nranks are unused).

typedef unsigned int hf_peern_u4 __attribute__((ext_vector_type(4)));

#define HF_PEERN_LANES 512

__device__ __forceinline__ void hf_peern_st128_sys(hf_peern_u4 *p, hf_peern_u4 v) {
    asm volatile("global_store_b128 %0, %1, off scope:SCOPE_SYS" ::"v"(p), "v"(v) : "memory");
}

// M system-scope b128 loads and their wait in ONE asm statement, so the
// compiler cannot copy or reuse a destination register while a load is in
// flight.
template <int M>
__device__ __forceinline__ void hf_peern_load_batch(hf_peern_u4 (&r)[M], const hf_peern_u4 *const (&p)[M]);

template <>
__device__ __forceinline__ void hf_peern_load_batch<1>(hf_peern_u4 (&r)[1], const hf_peern_u4 *const (&p)[1]) {
    asm volatile(
        "global_load_b128 %0, %1, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0])
        : "v"(p[0])
        : "memory");
}

template <>
__device__ __forceinline__ void hf_peern_load_batch<2>(hf_peern_u4 (&r)[2], const hf_peern_u4 *const (&p)[2]) {
    asm volatile(
        "global_load_b128 %0, %2, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %1, %3, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0]), "=&v"(r[1])
        : "v"(p[0]), "v"(p[1])
        : "memory");
}

template <>
__device__ __forceinline__ void hf_peern_load_batch<3>(hf_peern_u4 (&r)[3], const hf_peern_u4 *const (&p)[3]) {
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
__device__ __forceinline__ void hf_peern_load_batch<4>(hf_peern_u4 (&r)[4], const hf_peern_u4 *const (&p)[4]) {
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
__device__ __forceinline__ void hf_peern_load_batch<5>(hf_peern_u4 (&r)[5], const hf_peern_u4 *const (&p)[5]) {
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
__device__ __forceinline__ void hf_peern_load_batch<6>(hf_peern_u4 (&r)[6], const hf_peern_u4 *const (&p)[6]) {
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
__device__ __forceinline__ void hf_peern_load_batch<7>(hf_peern_u4 (&r)[7], const hf_peern_u4 *const (&p)[7]) {
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
__device__ __forceinline__ void hf_peern_load_batch<8>(hf_peern_u4 (&r)[8], const hf_peern_u4 *const (&p)[8]) {
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

template <>
__device__ __forceinline__ void hf_peern_load_batch<9>(hf_peern_u4 (&r)[9], const hf_peern_u4 *const (&p)[9]) {
    asm volatile(
        "global_load_b128 %0, %9, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %1, %10, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %2, %11, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %3, %12, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %4, %13, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %5, %14, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %6, %15, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %7, %16, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %8, %17, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0]), "=&v"(r[1]), "=&v"(r[2]), "=&v"(r[3]), "=&v"(r[4]), "=&v"(r[5]), "=&v"(r[6]), "=&v"(r[7]), "=&v"(r[8])
        : "v"(p[0]), "v"(p[1]), "v"(p[2]), "v"(p[3]), "v"(p[4]), "v"(p[5]), "v"(p[6]), "v"(p[7]), "v"(p[8])
        : "memory");
}

template <>
__device__ __forceinline__ void hf_peern_load_batch<10>(hf_peern_u4 (&r)[10], const hf_peern_u4 *const (&p)[10]) {
    asm volatile(
        "global_load_b128 %0, %10, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %1, %11, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %2, %12, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %3, %13, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %4, %14, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %5, %15, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %6, %16, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %7, %17, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %8, %18, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %9, %19, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0]), "=&v"(r[1]), "=&v"(r[2]), "=&v"(r[3]), "=&v"(r[4]), "=&v"(r[5]), "=&v"(r[6]), "=&v"(r[7]), "=&v"(r[8]), "=&v"(r[9])
        : "v"(p[0]), "v"(p[1]), "v"(p[2]), "v"(p[3]), "v"(p[4]), "v"(p[5]), "v"(p[6]), "v"(p[7]), "v"(p[8]), "v"(p[9])
        : "memory");
}

template <>
__device__ __forceinline__ void hf_peern_load_batch<11>(hf_peern_u4 (&r)[11], const hf_peern_u4 *const (&p)[11]) {
    asm volatile(
        "global_load_b128 %0, %11, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %1, %12, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %2, %13, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %3, %14, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %4, %15, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %5, %16, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %6, %17, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %7, %18, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %8, %19, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %9, %20, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %10, %21, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0]), "=&v"(r[1]), "=&v"(r[2]), "=&v"(r[3]), "=&v"(r[4]), "=&v"(r[5]), "=&v"(r[6]), "=&v"(r[7]), "=&v"(r[8]), "=&v"(r[9]), "=&v"(r[10])
        : "v"(p[0]), "v"(p[1]), "v"(p[2]), "v"(p[3]), "v"(p[4]), "v"(p[5]), "v"(p[6]), "v"(p[7]), "v"(p[8]), "v"(p[9]), "v"(p[10])
        : "memory");
}

template <>
__device__ __forceinline__ void hf_peern_load_batch<12>(hf_peern_u4 (&r)[12], const hf_peern_u4 *const (&p)[12]) {
    asm volatile(
        "global_load_b128 %0, %12, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %1, %13, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %2, %14, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %3, %15, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %4, %16, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %5, %17, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %6, %18, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %7, %19, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %8, %20, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %9, %21, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %10, %22, off scope:SCOPE_SYS\n\t"
        "global_load_b128 %11, %23, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0]), "=&v"(r[1]), "=&v"(r[2]), "=&v"(r[3]), "=&v"(r[4]), "=&v"(r[5]), "=&v"(r[6]), "=&v"(r[7]), "=&v"(r[8]), "=&v"(r[9]), "=&v"(r[10]), "=&v"(r[11])
        : "v"(p[0]), "v"(p[1]), "v"(p[2]), "v"(p[3]), "v"(p[4]), "v"(p[5]), "v"(p[6]), "v"(p[7]), "v"(p[8]), "v"(p[9]), "v"(p[10]), "v"(p[11])
        : "memory");
}

__device__ __forceinline__ float hf_peern_lo(unsigned w) { return tp16_unpack<HF_TP16_BF16>((uint16_t)(w & 0xffffu)); }
__device__ __forceinline__ float hf_peern_hi(unsigned w) { return tp16_unpack<HF_TP16_BF16>((uint16_t)(w >> 16)); }

// K granules per lane, P = nranks - 1 peers.
template <int K, int P>
__device__ __forceinline__ void tp_peer_pushn_body(
    float *__restrict__ x, const float *__restrict__ part,
    char *mbox0, char *mbox1, char *mbox2, char *mbox3,
    unsigned slot_stride, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wait_block,
    int rank, int granules) {
    const int tid = threadIdx.x;
    const float4 *__restrict__ part4 = reinterpret_cast<const float4 *>(part);
    float4 *__restrict__ x4 = reinterpret_cast<float4 *>(x);
    float4 xr[K];
    unsigned wlo[K], whi[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = tid + k * HF_PEERN_LANES;
        if (j < granules) {
            const float4 f = part4[j];
            xr[k] = x4[j];
            wlo[k] = (unsigned)tp16_pack<HF_TP16_BF16>(f.x) | ((unsigned)tp16_pack<HF_TP16_BF16>(f.y) << 16);
            whi[k] = (unsigned)tp16_pack<HF_TP16_BF16>(f.z) | ((unsigned)tp16_pack<HF_TP16_BF16>(f.w) << 16);
        }
    }
    unsigned g = __hip_atomic_load(gen_ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (g == 0u) g = 2u;
    const unsigned bank = g & 1u;
    // Push to the peers ONE AT A TIME (rotation rank+1, rank+2, ...), with a
    // release fence and a workgroup barrier between destinations: a GPU writing
    // to several peers concurrently through write-through stores collapses to
    // ~1.5 GB/s (45 us for 3 x 20 KB at TP4), while the same stores to one
    // destination at a time take ~2 us each. Slot `rank` of the peer's bank.
#pragma unroll
    for (int s = 1; s <= P; ++s) {
        const int p = (rank + s) % (P + 1);
        char *dst = p == 0 ? mbox0 : p == 1 ? mbox1 : p == 2 ? mbox2 : mbox3;
        hf_peern_u4 *pb = reinterpret_cast<hf_peern_u4 *>(
            dst + bank * bank_stride + (unsigned)rank * slot_stride);
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const int j = tid + k * HF_PEERN_LANES;
            if (j < granules) hf_peern_st128_sys(pb + j, hf_peern_u4{wlo[k], g, whi[k], g});
        }
        if (s < P) {
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
            __syncthreads();
        }
    }
    // poll slots q != rank of the local mailbox
    const char *local = (rank == 0 ? mbox0 : rank == 1 ? mbox1 : rank == 2 ? mbox2 : mbox3) + bank * bank_stride;
    hf_peern_u4 r[K * P];
    const hf_peern_u4 *lp[K * P];
#pragma unroll
    for (int qi = 0; qi < P; ++qi) {
        const int q = qi < rank ? qi : qi + 1;
        const hf_peern_u4 *base = reinterpret_cast<const hf_peern_u4 *>(local + (unsigned)q * slot_stride);
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const int j = tid + k * HF_PEERN_LANES;
            lp[qi * K + k] = base + (j < granules ? j : granules - 1);
        }
    }
    if (!hf_bounded_wait(
            [&] {
                hf_peern_load_batch<K * P>(r, lp);
                bool ok = true;
#pragma unroll
                for (int qi = 0; qi < P; ++qi) {
#pragma unroll
                    for (int k = 0; k < K; ++k) {
                        const int j = tid + k * HF_PEERN_LANES;
                        if (j < granules) ok &= (r[qi * K + k].y == g) & (r[qi * K + k].w == g);
                    }
                }
                return ok;
            },
            wait_block, HF_WAIT_SITE_TP_PEER_PUSHN_GFX1201, g)) {
        return;
    }
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = tid + k * HF_PEERN_LANES;
        if (j < granules) {
            float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int q = 0; q < P + 1; ++q) {
                unsigned lo, hi;
                if (q == rank) {
                    lo = wlo[k];
                    hi = whi[k];
                } else {
                    const int qi = q < rank ? q : q - 1;
                    lo = r[qi * K + k].x;
                    hi = r[qi * K + k].z;
                }
                const float4 t = make_float4(hf_peern_lo(lo), hf_peern_hi(lo), hf_peern_lo(hi), hf_peern_hi(hi));
                if (q == 0) {
                    acc = t;
                } else {
                    acc.x = acc.x + t.x;
                    acc.y = acc.y + t.y;
                    acc.z = acc.z + t.z;
                    acc.w = acc.w + t.w;
                }
            }
            float4 xi = xr[k];
            xi.x = xi.x + acc.x;
            xi.y = xi.y + acc.y;
            xi.z = xi.z + acc.z;
            xi.w = xi.w + acc.w;
            x4[j] = xi;
        }
    }
    // Every lane loaded the generation before this barrier.
    __syncthreads();
    if (tid == 0) __hip_atomic_store(gen_ptr, g, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

#define PEERN_CASE(KK, PP) \
    case (KK) * 4 + (PP): \
        tp_peer_pushn_body<KK, PP>(x, local_partial, mbox0, mbox1, mbox2, mbox3, slot_stride, bank_stride, gen_ptr, wait_block, rank, granules); \
        break;

extern "C" __global__ __launch_bounds__(512) void tp_peer_pushn_reduce_bf16_gfx1201(
    float *__restrict__ x, const float *__restrict__ local_partial,
    char *mbox0, char *mbox1, char *mbox2, char *mbox3,
    unsigned slot_stride, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wait_block,
    int rank, int nranks, int count) {
    if (blockIdx.x != 0) return;
    const int granules = count >> 2;
    const int k = (granules + HF_PEERN_LANES - 1) / HF_PEERN_LANES;
    switch (k * 4 + (nranks - 1)) {
        PEERN_CASE(1, 2) PEERN_CASE(1, 3)
        PEERN_CASE(2, 2) PEERN_CASE(2, 3)
        PEERN_CASE(3, 2) PEERN_CASE(3, 3)
        PEERN_CASE(4, 2) PEERN_CASE(4, 3)
    default: break;
    }
}

// Developer-only fault injection (`HIPFIRE_TP_PEER_PUSH_FAULT`): stall one
// rank's stream for `ticks` of the 100 MHz realtime counter before a replay so
// its peers' bounded waits run out. Launched eagerly, never captured.
extern "C" __global__ void tp_peer_push_fault_spin_gfx1201(unsigned ticks) {
    if (threadIdx.x != 0) return;
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < (unsigned long long)ticks) {
        __builtin_amdgcn_s_sleep(1);
    }
}
