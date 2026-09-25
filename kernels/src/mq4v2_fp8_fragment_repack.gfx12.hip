// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Per-call MQ4G256v2 -> F2 fp8 fragment repack for gfx1201
// (plan fp8-4k5 §11.4 / §11.6; consumer: builder module
// `gemm_mq4g256v2_wmma_fp8_gfx12_b1`).
//
// Source: symmetric uniform qt44, per matrix `[M][K/256][136 B]`; each
// 136-byte group is 2 x (f16 scale, f16 zero) for K halves [0,128) and
// [128,256), then 128 B of nibbles (element 2i = low nibble of byte i).
// Only the scale is read (symmetric fold); the zero word is ignored.
//
// One entry, one 80-byte kernarg ABI:
//   0 W0, 8 W1, 16 W2, 24 W3, 32 Wf, 40 Rw, 48 Ew,
//   56 M0, 60 M1, 64 M2, 68 M3, 72 K, 76 family.
// family 0: virtual rows concatenate W0..W3 (M0..M3 rows) without padding.
// family 1: paired gate/up (W0 gate, W1 up, M0 = M1 = gate_m); within each
//           256-row tile rb, virtual row t = rg*64 + i*16 + row16 maps to
//           hrow = rb*128 + rg*32 + (i/2)*16 + row16 of W[i%2], valid iff
//           hrow < M0.
// Any other family value writes nothing (the host rejects it first).
// Mt = sum(Mi), Mp = ceil(Mt/256)*256, B = K/128; the host guarantees
// K % 256 == 0, active Mi % 4 == 0, aligned and in-bounds allocations.
//
// Outputs (all padding initialized):
//   Wf[(rb*B+kb)*32768 + rg*8192 + (u*4+i)*512 + lane*16 + st*8 + j]
//     = E4M3(q-8) of virtual row rb*256+rg*64+i*16+(lane%16) at
//       k = kb*128 + u*32 + st*16 + (lane/16)*8 + j; zero for a zero-scale
//       half and for padding rows.
//   Rw[(rb*B+kb)*256 + r]: ratio fold; e[-1] = 1, e[b] = sw[b] if
//     sw[b] != 0 else e[b-1]; Rw[0] = 1, Rw[b] = RN(e[b-1] / e[b]).
//   Ew[virtual_row] = e[B-1].  Padding rows: Rw = 1, Ew = 1.
// Non-finite scales are outside the contract (rejected before launch).
//
// Launch: [Mp/64, K/256, 1], block 256, no dynamic LDS; one 64-row x K256
// tile per CTA. CTA (x, g) writes Wf for kb = 2g, 2g+1 of its 64 rows, Rw for
// the same (row, kb) pairs, and, when g = K/256 - 1, Ew for its rows. The
// separate metadata entry of the first draft was removed by measured cutover
// (hub-approved): its per-row header scan re-fetched every qt44 cache line.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace {

constexpr unsigned kGroupBytes = 136u;

struct Fp8fSource {
    const unsigned char* row;  // start of the source row (group 0)
    bool valid;
};

__device__ __forceinline__ Fp8fSource fp8f_source(
    const unsigned char* W0, const unsigned char* W1,
    const unsigned char* W2, const unsigned char* W3,
    unsigned M0, unsigned M1, unsigned M2, unsigned M3,
    unsigned row_bytes, int family, unsigned v) {
    Fp8fSource s{nullptr, false};
    if (family == 1) {
        const unsigned t = v & 255u;
        const unsigned hrow = (v >> 8) * 128u + (t >> 6) * 32u
            + ((t >> 5) & 1u) * 16u + (t & 15u);
        if (hrow < M0) {
            s.row = (((t >> 4) & 1u) ? W1 : W0) + (size_t)hrow * row_bytes;
            s.valid = true;
        }
        return s;
    }
    unsigned r = v;
    if (r < M0) { s.row = W0 + (size_t)r * row_bytes; s.valid = true; return s; }
    r -= M0;
    if (r < M1) { s.row = W1 + (size_t)r * row_bytes; s.valid = true; return s; }
    r -= M1;
    if (r < M2) { s.row = W2 + (size_t)r * row_bytes; s.valid = true; return s; }
    r -= M2;
    if (r < M3) { s.row = W3 + (size_t)r * row_bytes; s.valid = true; return s; }
    return s;
}

__device__ __forceinline__ float fp8f_half_to_f32(unsigned bits16) {
    return __half2float(__ushort_as_half((unsigned short)bits16));
}

// 8 nibbles (nibble n = bits 4n..4n+3) -> 8 bytes E4M3(q-8), byte n of the
// pair = code of nibble n. Same table and byte routing as the shipped
// symfold `mq4_qpk_to_e4m3`: q 0..15 -> D0 CE CC CA C8 C4 C0 B8 00 38 40 44
// 48 4A 4C 4E.
__device__ __forceinline__ unsigned fp8f_lut4(unsigned sel) {
    const unsigned C0 = 0xCACCCED0u;
    const unsigned C1 = 0xB8C0C4C8u;
    const unsigned C2 = 0x44403800u;
    const unsigned C3 = 0x4E4C4A48u;
    const unsigned idx = sel & 0x07070707u;
    const unsigned lo = __builtin_amdgcn_perm(C1, C0, idx);
    const unsigned hi = __builtin_amdgcn_perm(C3, C2, idx);
    const unsigned selp = 0x03020100u | ((sel & 0x08080808u) >> 1);
    return __builtin_amdgcn_perm(hi, lo, selp);
}

__device__ __forceinline__ uint2 fp8f_expand8(unsigned pk) {
    const unsigned ev = pk & 0x0F0F0F0Fu;
    const unsigned od = (pk >> 4) & 0x0F0F0F0Fu;
    return make_uint2(fp8f_lut4(__builtin_amdgcn_perm(od, ev, 0x05010400u)),
                      fp8f_lut4(__builtin_amdgcn_perm(od, ev, 0x07030602u)));
}

}  // namespace

extern "C" __global__ __launch_bounds__(256) void mq4v2_fp8_fragment_repack_gfx1201(
    const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
    const unsigned char* __restrict__ W2, const unsigned char* __restrict__ W3,
    unsigned char* __restrict__ Wf, float* __restrict__ Rw, float* __restrict__ Ew,
    int M0, int M1, int M2, int M3, int K, int family) {
    if (family != 0 && family != 1) return;
    // Thread -> (u, i, row16): warp w covers u = w/2 and row16-blocks
    // i = 2(w%2), 2(w%2)+1, so each store instruction writes two
    // contiguous 256-byte runs and the pair of stores a contiguous 1 KiB.
    const unsigned t = threadIdx.x;
    const unsigned x = blockIdx.x;  // 64-row tile: rb = x/4, rg = x%4
    const unsigned g = blockIdx.y;  // K256 group -> kb = 2g, 2g+1
    const unsigned B = (unsigned)K >> 7;
    const unsigned gpr = (unsigned)K >> 8;
    const unsigned r16 = t & 15u;
    const unsigned i = (t >> 4) & 3u;
    const unsigned u = t >> 6;
    const unsigned v = x * 64u + i * 16u + r16;
    const Fp8fSource src = fp8f_source(W0, W1, W2, W3, (unsigned)M0, (unsigned)M1,
                                       (unsigned)M2, (unsigned)M3, gpr * kGroupBytes, family, v);

    uint2 hdr = make_uint2(0u, 0u);
    uint2 p[2][2] = {{make_uint2(0u, 0u), make_uint2(0u, 0u)}, {make_uint2(0u, 0u), make_uint2(0u, 0u)}};
    if (src.valid) {
        const unsigned char* gp = src.row + (size_t)g * kGroupBytes;
        hdr = *(const uint2*)gp;
#pragma unroll
        for (unsigned h = 0; h < 2u; ++h) {
            const unsigned char* q = gp + 8u + h * 64u + u * 16u;
            p[h][0] = *(const uint2*)q;
            p[h][1] = *(const uint2*)(q + 8u);
        }
    }
    unsigned char* tile = Wf + (size_t)((x >> 2) * B + 2u * g) * 32768u
        + (x & 3u) * 8192u + (u * 4u + i) * 512u + r16 * 16u;
#pragma unroll
    for (unsigned h = 0; h < 2u; ++h) {
        // A zero scale (+0 or -0) contributes zero operands.
        const unsigned sbits = (h ? hdr.y : hdr.x) & 0x7FFFu;
        uint4 lo = make_uint4(0u, 0u, 0u, 0u);  // lane r16:    k16 0..7
        uint4 hi = make_uint4(0u, 0u, 0u, 0u);  // lane r16+16: k16 8..15
        if (sbits != 0u) {
            const uint2 a0 = fp8f_expand8(p[h][0].x);  // st0, k16 0..7
            const uint2 a1 = fp8f_expand8(p[h][0].y);  // st0, k16 8..15
            const uint2 b0 = fp8f_expand8(p[h][1].x);  // st1, k16 0..7
            const uint2 b1 = fp8f_expand8(p[h][1].y);  // st1, k16 8..15
            lo = make_uint4(a0.x, a0.y, b0.x, b0.y);
            hi = make_uint4(a1.x, a1.y, b1.x, b1.y);
        }
        unsigned char* dst = tile + (size_t)h * 32768u;
        *(uint4*)dst = lo;
        *(uint4*)(dst + 256u) = hi;
    }

    // Metadata: threads u == 0 own row t = i*16 + r16 of this 64-row tile and
    // write Rw for kb = 2g, 2g+1 (two contiguous 256-byte runs per CTA) and,
    // in the last group, Ew. e[2g-1] is the last nonzero scale at or before
    // half 2g-1 (1 if none): one 2-byte read of group g-1's header, scanning
    // further back only across zero scales. Ratios are IEEE RN f32 divisions
    // (HIP's default correctly rounded division; v_div_fixup_f32 in the ISA).
    if (u == 0u) {
        float* rw = Rw + (size_t)((x >> 2) * B + 2u * g) * 256u + (x & 3u) * 64u + i * 16u + r16;
        if (!src.valid) {
            rw[0] = 1.0f;
            rw[256] = 1.0f;
            if (g + 1u == gpr) Ew[v] = 1.0f;
        } else {
            float e = 1.0f;
            for (int b = (int)(2u * g) - 1; b >= 0; --b) {
                const unsigned sb = *(const unsigned short*)(src.row
                    + (size_t)(b >> 1) * kGroupBytes + (unsigned)(b & 1) * 4u);
                if (sb & 0x7FFFu) {
                    e = fp8f_half_to_f32(sb);
                    break;
                }
            }
            const float s0 = fp8f_half_to_f32(hdr.x & 0xFFFFu);
            const float s1 = fp8f_half_to_f32(hdr.y & 0xFFFFu);
            const float e0 = (s0 != 0.0f) ? s0 : e;
            const float e1 = (s1 != 0.0f) ? s1 : e0;
            rw[0] = (g == 0u) ? 1.0f : e / e0;
            rw[256] = e0 / e1;
            if (g + 1u == gpr) Ew[v] = e1;
        }
    }
}
