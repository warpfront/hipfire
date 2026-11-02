// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// SKU-emulation validator kernels (EMU-DESIGN.md §6, E8). Model-free,
// integer-only and strictly finite:
//   - no atomics, no stop flags, no spin or wait loops, no device sleeps;
//   - every write slot is unique to the writing thread (or wave);
//   - no CU masking of any kind (the probe observes the real placement).
//
// Explicit arch guards: the supported device targets are gfx1100, gfx1151 and
// gfx1201. On any other target each kernel compiles to a poison-writing body
// so a mistaken launch can never produce a passing receipt; the host refuses
// to launch on an unsupported arch before loading this module.

#include <hip/hip_runtime.h>
#include <stdint.h>

#if defined(__gfx1100__) || defined(__gfx1151__) || defined(__gfx1201__)
#define EMU_VALIDATE_ARCH_SUPPORTED 1
#else
#define EMU_VALIDATE_ARCH_SUPPORTED 0
#endif

#define EMU_FNV_OFFSET 0x811C9DC5u
#define EMU_FNV_PRIME 0x01000193u
#define EMU_POISON 0xFFFFFFFFu

// Per-wave HW_ID1 probe (HW_REG_HW_ID1, hwreg id 23 on gfx10/11/12, full
// 32-bit field). Lane 0 of every wave stores the raw register value into its
// own slot hwid[blockIdx.x * waves_per_block + wave]; the host pre-fills the
// buffer with 0xFFFFFFFF and decodes on the CPU. A fixed-length dependent
// integer chain keeps each block resident for a short, bounded time so the
// dispatcher spreads blocks across compute units; its per-slot result goes to
// aux[slot] so the host can verify the chain completed and was not elided.
extern "C" __global__ void emu_hwid_probe(uint32_t *__restrict__ hwid,
                                          uint32_t *__restrict__ aux,
                                          uint32_t iters) {
    const uint32_t wave_size = (uint32_t)warpSize;
    const uint32_t waves_per_block = (blockDim.x + wave_size - 1u) / wave_size;
    const uint32_t wave = threadIdx.x / wave_size;
    const uint32_t lane = threadIdx.x % wave_size;
    const uint32_t slot = blockIdx.x * waves_per_block + wave;
    uint32_t x = slot;
    for (uint32_t i = 0; i < iters; ++i) {
        x = x * 1664525u + 1013904223u;
    }
    if (lane == 0) {
#if EMU_VALIDATE_ARCH_SUPPORTED
        // size = 32 bits (31 << 11), offset 0, id 23.
        hwid[slot] = (uint32_t)__builtin_amdgcn_s_getreg(23 | (0 << 6) | (31 << 11));
#else
        hwid[slot] = EMU_POISON;
#endif
        aux[slot] = x;
    }
}

// Integer vector read. Thread t folds vectors t, t+stride, t+2*stride, ... with
// FNV-1a over their four 32-bit words (stride = gridDim.x * blockDim.x) and
// writes the result to its own slot sink[t]. The sink has exactly
// gridDim.x * blockDim.x slots; no shared or conditional slot exists.
extern "C" __global__ void emu_read_checksum(const uint4 *__restrict__ src,
                                             uint32_t *__restrict__ sink,
                                             uint64_t n_vec) {
    const uint64_t tid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
#if EMU_VALIDATE_ARCH_SUPPORTED
    uint32_t acc = EMU_FNV_OFFSET ^ (uint32_t)tid;
    for (uint64_t v = tid; v < n_vec; v += stride) {
        const uint4 q = src[v];
        acc = (acc ^ q.x) * EMU_FNV_PRIME;
        acc = (acc ^ q.y) * EMU_FNV_PRIME;
        acc = (acc ^ q.z) * EMU_FNV_PRIME;
        acc = (acc ^ q.w) * EMU_FNV_PRIME;
    }
    sink[tid] = acc;
#else
    sink[tid] = EMU_POISON;
#endif
}

// Integer copy of n_vec 16-byte vectors; each destination vector is written by
// exactly one thread (grid-stride). Reads n_vec*16 bytes and writes n_vec*16.
extern "C" __global__ void emu_copy_u32x4(uint4 *__restrict__ dst,
                                          const uint4 *__restrict__ src,
                                          uint64_t n_vec) {
#if EMU_VALIDATE_ARCH_SUPPORTED
    const uint64_t tid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
    for (uint64_t v = tid; v < n_vec; v += stride) {
        dst[v] = src[v];
    }
#else
    (void)dst;
    (void)src;
    (void)n_vec;
#endif
}
