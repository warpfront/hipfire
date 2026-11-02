// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! railgun copy kernels (railgun design §1.1 `Node::Copy` / `Node::CopyBatch`,
//! milestone MC). Owned by `crates/railgun` (`railgun::copy`), which holds the
//! Rust side of the ABI: `CopyEntry` layout, slot counts, symbols and launch
//! geometry.
//!
//! Pure byte copies with `hipMemcpyDtoD` semantics for any alignment and any
//! size up to `0xFFFFFFFF` bytes per copy: every destination byte in
//! `[dst, dst+bytes)` is written exactly once with the source byte at the same
//! offset, no byte outside `[src, src+bytes)` is read and none outside
//! `[dst, dst+bytes)` written. Overlapping source and destination ranges are
//! undefined, as for `hipMemcpyDtoD`; `railgun::Program::prepare` refuses them.
//!
//! Layout of one copy (thread `g` of `stride` threads assigned to it):
//!   head  bytes [0, h)          h = bytes until dst is 16 B aligned (≤ 15)
//!   body  16 B vectors          dst-aligned stores; the source load is
//!                               aligned when src and dst agree mod 16, and an
//!                               unaligned 16 B load otherwise
//!   tail  bytes after the body  (≤ 15)
//! Work is split by index, so the result is independent of grid size and of
//! execution order.
//!
//! Index arithmetic is 32-bit on purpose: the certifier's gfx12/gfx11 ISA
//! tables have no 64-bit vector compares, and a 32-bit byte count keeps every
//! loop bound a 32-bit compare. With `stride ≤ 2^31` (grid.x ≤ 2^23 blocks of
//! 256) no index sum overflows: `i < 2^28` vectors, `done + i < 2^32` bytes.
//!
//! Every access is `kernel pointer argument + 32-bit byte offset` (SADDR +
//! VGPR offset), so the certifier's A1 attributes each access to exactly the
//! pointer arguments it may use, with their read/write modes.

#include <hip/hip_runtime.h>

// One `railgun_copy_batch` table entry: `railgun::copy::CopyEntry` is the
// Rust mirror (24 B little-endian; `reserved` is written as 0). The table
// holds no pointers: slots index the launch's pointer arguments, and the
// offsets are bytes from the slot's resource base. `prepare` checks every
// entry against its resource, and `offset + bytes ≤ 2^32` so the 32-bit
// offset arithmetic below never wraps.
struct RailgunCopyEntry {
    unsigned int src_slot;    // source argument s<src_slot>
    unsigned int src_offset;
    unsigned int dst_slot;    // destination argument d<dst_slot>
    unsigned int dst_offset;
    unsigned int bytes;
    unsigned int reserved;
};

// Slots per side (`railgun::copy::SRC_SLOTS` / `DST_SLOTS`). An entry naming
// a slot outside them copies nothing.
#define RAILGUN_COPY_SLOTS 8u

// Global address space (1): generic pointers would lower to FLAT, which may
// also address LDS/scratch apertures.
typedef __attribute__((address_space(1))) unsigned char railgun_gu8;
typedef unsigned int railgun_u32x4 __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) railgun_u32x4 railgun_gu32x4;
// Same vector, alignment 1: the backend's unaligned-access mode (on for
// amdhsa) turns this into one 16 B load that touches exactly its 16 bytes.
typedef railgun_u32x4 railgun_u32x4_u __attribute__((aligned(1)));
typedef __attribute__((address_space(1))) railgun_u32x4_u railgun_gu32x4_u;

// Copy `n` bytes from `s + so` to `d + dof`. Every access addresses
// `s`/`d` plus a 32-bit offset (`so + x`, `dof + x`), never a pointer
// advanced by an offset: the backend addresses it as SADDR + VGPR offset, so
// A1 sees each access's base as exactly `s` or `d`. (`head` comes from the
// low bits of `d`; folded into a pointer it would make the source loads look
// as if they might read `d`.) The caller guarantees `so + n ≤ 2^32` and
// `dof + n ≤ 2^32`.
__device__ __forceinline__ void railgun_copy_range(
    const railgun_gu8* __restrict__ s,
    unsigned int so,
    railgun_gu8* __restrict__ d,
    unsigned int dof,
    unsigned int n,
    unsigned int g,
    unsigned int stride)
{
    unsigned int head = (16u - (((unsigned int)(unsigned long long)d + dof) & 15u)) & 15u;
    if (head > n) head = n;
    for (unsigned int i = g; i < head; i += stride) d[dof + i] = s[so + i];

    const unsigned int rest = n - head;
    const unsigned int nvec = rest >> 4;
    if ((((unsigned int)(unsigned long long)s + so + head) & 15u) == 0) {
        for (unsigned int i = g; i < nvec; i += stride) {
            const unsigned int o = head + (i << 4);
            *(railgun_gu32x4*)(d + (dof + o)) = *(const railgun_gu32x4*)(s + (so + o));
        }
    } else {
        for (unsigned int i = g; i < nvec; i += stride) {
            const unsigned int o = head + (i << 4);
            *(railgun_gu32x4*)(d + (dof + o)) = *(const railgun_gu32x4_u*)(s + (so + o));
        }
    }

    const unsigned int done = head + (nvec << 4);
    const unsigned int tail = n - done;
    for (unsigned int i = g; i < tail; i += stride) d[dof + done + i] = s[so + done + i];
}

// `Node::Copy`: both pointers are kernargs, so A1 attributes the read to
// `src` and the write to `dst`. Grid (gx, 1, 1), block (256, 1, 1).
extern "C" __launch_bounds__(256)
__global__ void railgun_copy(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int bytes)
{
    const unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;
    railgun_copy_range((const railgun_gu8*)src, 0u, (railgun_gu8*)dst, 0u, bytes, g, stride);
}

// `Node::CopyBatch`: entry `blockIdx.y` of the read-only slot table. The
// resources are the pointer arguments `s0..s7` (only read) and `d0..d7`
// (only written); an entry selects one of each by index, and the select is
// a union of kernarg pointers, so A1 attributes every access to the slot
// arguments with exact modes. Grid (gx, n, 1), block (256, 1, 1); entries
// must be pairwise disjoint in their destinations and must not write any
// entry's source (checked at prepare), so the result equals the entries'
// memcpys in any order.
extern "C" __launch_bounds__(256)
__global__ void railgun_copy_batch(
    const struct RailgunCopyEntry* __restrict__ table,
    unsigned int n,
    const unsigned char* __restrict__ s0,
    const unsigned char* __restrict__ s1,
    const unsigned char* __restrict__ s2,
    const unsigned char* __restrict__ s3,
    const unsigned char* __restrict__ s4,
    const unsigned char* __restrict__ s5,
    const unsigned char* __restrict__ s6,
    const unsigned char* __restrict__ s7,
    unsigned char* __restrict__ d0,
    unsigned char* __restrict__ d1,
    unsigned char* __restrict__ d2,
    unsigned char* __restrict__ d3,
    unsigned char* __restrict__ d4,
    unsigned char* __restrict__ d5,
    unsigned char* __restrict__ d6,
    unsigned char* __restrict__ d7)
{
    const unsigned int e = blockIdx.y;
    if (e >= n) return;
    const struct RailgunCopyEntry t = table[e];
    const unsigned int a = t.src_slot, b = t.dst_slot;
    // Both below 8 (a power of two) iff their OR is: one scalar compare.
    if ((a | b) >= RAILGUN_COPY_SLOTS) return;
    const unsigned char* s = a == 0 ? s0 : a == 1 ? s1 : a == 2 ? s2 : a == 3 ? s3
                           : a == 4 ? s4 : a == 5 ? s5 : a == 6 ? s6 : s7;
    unsigned char* d = b == 0 ? d0 : b == 1 ? d1 : b == 2 ? d2 : b == 3 ? d3
                     : b == 4 ? d4 : b == 5 ? d5 : b == 6 ? d6 : d7;
    const unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;
    railgun_copy_range((const railgun_gu8*)s, t.src_offset, (railgun_gu8*)d, t.dst_offset, t.bytes, g, stride);
}
