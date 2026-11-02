// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched descriptor copy for the DFlash verify GDN tape capture.
//!
//! The continuous-batching verify trunk snaps, per DeltaNet layer and per
//! lane, the pre-conv1d qkv rows and the post-sigmoid alpha/beta rows into the
//! lane's rollback tape: three `hipMemcpyDtoD` (`__amd_rocclr_copyBuffer`)
//! per lane per layer, 1,152 per C8 step on a 48-layer target. This kernel
//! performs one layer's copies for every lane of the chunk in ONE launch.
//!
//! The descriptor table rides in the kernarg segment (no device table, no
//! H2D upload): copy `d` moves `cnt[d]` bytes from `src[d]` to `dst[d]`.
//! Grid: x = ceil(max cnt / TAPE_COPY_CHUNK) chunks, y = descriptor index;
//! each block copies one TAPE_COPY_CHUNK-byte slice of one descriptor. A pure
//! byte copy over disjoint destination ranges — no arithmetic, no atomics —
//! so the destination bytes equal the per-copy memcpy path's by construction.
//! 16 B vector loop when source, destination and slice bounds are 16 B
//! aligned, byte loop otherwise.

#include <hip/hip_runtime.h>

#define TAPE_COPY_MAX 48
#define TAPE_COPY_CHUNK 16384ull

struct TapeCopyBatch {
    unsigned long long src[TAPE_COPY_MAX];
    unsigned long long dst[TAPE_COPY_MAX];
    unsigned long long cnt[TAPE_COPY_MAX];
    unsigned int n;
};

extern "C" __launch_bounds__(256) __global__ void tape_copy_batched(const TapeCopyBatch t) {
    const unsigned int d = blockIdx.y;
    if (d >= t.n) return;
    const unsigned long long cnt = t.cnt[d];
    const unsigned long long beg = (unsigned long long)blockIdx.x * TAPE_COPY_CHUNK;
    if (beg >= cnt) return;
    const unsigned long long end = (beg + TAPE_COPY_CHUNK < cnt) ? beg + TAPE_COPY_CHUNK : cnt;
    const unsigned long long sa = t.src[d];
    const unsigned long long da = t.dst[d];
    const unsigned char* __restrict__ s = (const unsigned char*)sa;
    unsigned char* __restrict__ o = (unsigned char*)da;
    if (((sa | da | beg | end) & 15ull) == 0) {
        const float4* __restrict__ s4 = (const float4*)(s + beg);
        float4* __restrict__ o4 = (float4*)(o + beg);
        const unsigned long long n4 = (end - beg) >> 4;
        for (unsigned long long i = threadIdx.x; i < n4; i += blockDim.x) {
            o4[i] = s4[i];
        }
    } else {
        for (unsigned long long i = beg + threadIdx.x; i < end; i += blockDim.x) {
            o[i] = s[i];
        }
    }
}
