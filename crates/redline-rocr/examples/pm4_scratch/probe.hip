// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
//
// Kernels for `pm4_scratch_probe`: real private segments (dynamic indexing of
// a per-lane array), so a PM4 dispatch must program scratch to run them.
#include <hip/hip_runtime.h>

__device__ __forceinline__ unsigned mix32(unsigned x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Data-dependent private-array indexing: 48 dwords/lane stay in scratch.
extern "C" __global__ void scratch_dyn(const unsigned* __restrict__ in,
                                       unsigned* __restrict__ out, int n,
                                       int rounds) {
    unsigned priv[48];
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < 48; j++) {
        priv[j] = in[(i * 5u + (unsigned)j) % (unsigned)n] + (unsigned)j * 0x9e3779b9u;
    }
    unsigned idx = in[i % (unsigned)n] % 48u;
    unsigned acc = i;
    for (int r = 0; r < rounds; r++) {
        acc = acc * 1664525u + priv[idx] + 1013904223u;
        idx = (acc >> 9) % 48u;
        priv[idx] ^= acc;
    }
    out[i] = acc ^ priv[(i + 1u) % 48u];
}

// No private segment: the non-scratch kernel a mixed stream interleaves.
extern "C" __global__ void plain_mix(const unsigned* __restrict__ in,
                                     unsigned* __restrict__ out) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = in[i] * 3u + 1u;
}

// Per-wave scratch aliasing detector. Every lane writes a unique tag into its
// private array, then re-reads it for `spins` rounds while the rest of the
// grid runs: any two live waves sharing a scratch slice corrupt each other and
// `fail` counts the bad reads. `out` is a pure function of the lane id.
extern "C" __global__ void scratch_alias(unsigned* __restrict__ out,
                                         unsigned* __restrict__ fail,
                                         int spins, unsigned seed) {
    volatile unsigned priv[64];
    const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned tag = mix32(gid ^ seed);
    for (int j = 0; j < 64; j++) {
        priv[j] = tag + (unsigned)j;
    }
    unsigned bad = 0;
    unsigned sum = 0;
    for (int s = 0; s < spins; s++) {
        const unsigned k = ((unsigned)s * 7u + (gid & 63u)) & 63u;
        const unsigned v = priv[k];
        bad += (v != tag + k) ? 1u : 0u;
        sum += v;
    }
    out[gid] = sum ^ tag;
    if (bad != 0) {
        atomicAdd(fail, bad);
    }
}
