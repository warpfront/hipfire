// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// clang's HIP `float4` is a wrapper type rather than a native vector, so the
// nontemporal builtin cannot consume `float4*` directly. Match RCCL's gfx11
// fallback and issue two global non-temporal dword-pair loads.
__device__ __forceinline__ float4 tp_load_nontemporal_f32x4(const float4 *ptr) {
    typedef __attribute__((address_space(1))) const unsigned long long *global_u64_ptr;
    const unsigned long long *p64 = reinterpret_cast<const unsigned long long *>(ptr);
    union {
        unsigned long long u64[2];
        float4 f4;
    } value;
    value.u64[0] = __builtin_nontemporal_load((global_u64_ptr)p64);
    value.u64[1] = __builtin_nontemporal_load((global_u64_ptr)(p64 + 1));
    return value.f4;
}

__device__ __forceinline__ void tp_store_nontemporal_f32x4(float4 *ptr, float4 value) {
    typedef __attribute__((address_space(1))) unsigned long long *global_u64_ptr;
    unsigned long long *p64 = reinterpret_cast<unsigned long long *>(ptr);
    union {
        unsigned long long u64[2];
        float4 f4;
    } bits;
    bits.f4 = value;
    __builtin_nontemporal_store(bits.u64[0], (global_u64_ptr)p64);
    __builtin_nontemporal_store(bits.u64[1], (global_u64_ptr)(p64 + 1));
}

// Reusable graph-resident two-rank producer barrier for gfx1100 TP. Each rank
// is the sole writer of its local signal and advances it once per barrier.
// The peer waits for the same generation. The generation lives on device, so
// graph replay needs neither host-side signal reset nor a changing kernarg.
extern "C" __global__ void tp_graph_signal_barrier_gfx1100(
    unsigned int *__restrict__ local_signal,
    const unsigned int *__restrict__ peer_signal) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // This stream is the only writer of local_signal. FIFO graph execution
    // makes load+exchange a safe increment while avoiding an unnecessary
    // cross-device atomic RMW. Release publishes every producer before this
    // barrier; the peer's acquire makes those writes visible before it reads.
    const unsigned int epoch = __hip_atomic_load(
        local_signal, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) + 1u;
    __hip_atomic_exchange(
        local_signal, epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);

    while (__hip_atomic_load(
               peer_signal, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
        __builtin_amdgcn_s_sleep(1);
    }
}

// Fused TP2 producer barrier + rank-ordered residual reduction. Keeping the
// synchronization and the 20 KiB Qwen-27B residual add in one workgroup
// removes one graph node at every attention/FFN boundary. Every lane performs
// the peer acquire itself before reading p_first/p_second, avoiding reliance
// on cross-lane propagation of a single lane's system-scope acquire.
extern "C" __global__ void tp_graph_signal_reduce_gfx1100(
    unsigned int *__restrict__ local_signal,
    const unsigned int *__restrict__ peer_signal,
    float *__restrict__ x,
    const float *__restrict__ local_partial,
    float *__restrict__ peer_shadow,
    const float *__restrict__ local_shadow,
    int local_is_first,
    int count) {
    if (blockIdx.x != 0) return;

    const int vec = count >> 2;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // gfx1100 graph replay can retain stale peer reads even after a system-
    // scope signal acquire. Push this rank's 20 KiB partial into memory owned
    // by the peer instead: after the barrier every consumer reads only local
    // VRAM. Each writer lane publishes its remote stores before thread 0
    // advances the peer-visible epoch.
    const float4 *__restrict__ localv = reinterpret_cast<const float4 *>(local_partial);
    float4 *__restrict__ peerv = reinterpret_cast<float4 *>(peer_shadow);
    for (int i = tid; i < vec; i += stride) {
        tp_store_nontemporal_f32x4(peerv + i, localv[i]);
    }
    for (int i = (vec << 2) + tid; i < count; i += stride) {
        __builtin_nontemporal_store(local_partial[i], peer_shadow + i);
    }
    __threadfence_system();
    __syncthreads();

    __shared__ unsigned int epoch;
    if (threadIdx.x == 0) {
        epoch = __hip_atomic_load(
            local_signal, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) + 1u;
        __hip_atomic_exchange(
            local_signal, epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();

    while (__hip_atomic_load(
               peer_signal, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
        __builtin_amdgcn_s_sleep(1);
    }

    float4 *__restrict__ xv = reinterpret_cast<float4 *>(x);
    const float4 *__restrict__ shadowv = reinterpret_cast<const float4 *>(local_shadow);
    for (int i = tid; i < vec; i += stride) {
        float4 xi = xv[i];
        const float4 local = localv[i];
        // The shadow is physically local but peer-produced. Bypass any stale
        // private line retained from the preceding graph node.
        const float4 shadow = tp_load_nontemporal_f32x4(shadowv + i);
        const float4 fi = local_is_first ? local : shadow;
        const float4 si = local_is_first ? shadow : local;
        xi.x = xi.x + (fi.x + si.x);
        xi.y = xi.y + (fi.y + si.y);
        xi.z = xi.z + (fi.z + si.z);
        xi.w = xi.w + (fi.w + si.w);
        xv[i] = xi;
    }
    for (int i = (vec << 2) + tid; i < count; i += stride) {
        const float local = local_partial[i];
        const float shadow = __builtin_nontemporal_load(local_shadow + i);
        const float fi = local_is_first ? local : shadow;
        const float si = local_is_first ? shadow : local;
        x[i] = x[i] + (fi + si);
    }
}
