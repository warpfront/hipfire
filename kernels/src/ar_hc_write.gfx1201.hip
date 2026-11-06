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
//
// ---------------------------------------------------------------------------------------
// Wire formats. Both are ABI-identical entries (`ar_hc_write_*` tagged, `ar_hc_write_d8_*`
// dense). The host picks one PER LAUNCH from the row count (HIPFIRE_EP_AR_WIRE_DENSE, default
// on: dense from 16 rows, tagged below; 0 = tagged always) and gives each wire its own
// mailbox region, so a slot is only ever written and read under one format. The partial values,
// their bf16 rounding, the fold order and the HC write are the same code in both; only what
// travels through the mailbox differs.
//
// Dense wire. A granule is 8 bytes {d0d1, d2d3} (the four bf16 payload values, NO tag): half the
// wire bytes of the tagged 16-byte granule. The measured bottleneck of a 256-row launch is the
// posted-store bandwidth of the PCIe peer link (~40 GB/s per direction, independent of the
// store width), so halving the bytes halves that term.
//   EMPTY = 0xFFFFFFFF. A mailbox dword is "valid" iff it is not EMPTY. The mailbox is EMPTY
//   everywhere at load/reset (memset 0xFF), a producer stores payload dwords, the CONSUMER
//   stores EMPTY back over every slot it consumed (one b64 store per granule, after its poll
//   succeeded and its payload sits in registers).
// Why a reader never folds a torn or stale granule:
//   1. Torn. A granule is accepted only when BOTH of its dwords are non-EMPTY, and each dword
//      is validated on its own, so the argument needs only that a naturally aligned dword
//      store (and the dword load of it) is single-copy atomic. That holds for the gfx12
//      memory pipeline and for PCIe (a Memory Write TLP is dword-granular: every dword of a
//      write lands whole or not at all; the ORDER in which the dwords of one TLP, or of two
//      TLPs, become visible is unspecified and is not used). A b64/b128 store may be split
//      or merged on the wire in any way; a half-visible granule has an EMPTY dword and is
//      not accepted. (The tagged wire instead relies on the 8-byte {payload, tag} pair
//      arriving as a unit.) Tested with a hostile peer that stores dwords one at a time in
//      both orders with a gap (ar_hc_write_tear_oracle).
//   2. Stale. A slot of bank b holds a non-EMPTY dword only between the producer's store and
//      the consumer's re-arm of the SAME launch: bank b = tag & 1 is next written by the
//      peer's launch g+2, which starts only after the peer completed launch g+1, which needed
//      this rank's launch-(g+1) payload, which this rank pushes only after its launch g
//      kernel completed (stream order), and the end of a kernel waits for its outstanding
//      system-scope stores, including the re-arm (a write-through store to this rank's own
//      fine-grained VRAM). So the re-arm of launch g is globally performed before the peer's
//      first store of launch g+2 exists. The re-arm of a slot never races the peer's write of
//      it in the same launch: the consumer re-arms only slots whose payload it has read. Two
//      banks are still required: without them the peer's launch g+1 could store into a slot
//      this rank's launch g has read but not yet re-armed, and the re-arm would erase it.
//   3. Payload == EMPTY. A finite partial never rounds to bf16 0xFFFF (that is a NaN), so a
//      payload dword is 0xFFFFFFFF only for a pair of NaNs; the producer remaps that dword
//      to 0xFFFFFFFE (still a NaN pair): finite results are byte-identical, a NaN pair stays
//      NaN (only its payload bits can differ from the tagged wire).
//   4. Abort. A timed-out wait leaves granules behind; the host resets both ranks (memset
//      0xFF of the mailbox, zero generation) before the next launch, as for the tagged wire.
// Contract kept from the tagged wire, one weaker consequence: both ranks must issue the same
// launches with the same row counts. A peer that pushed MORE granules than this rank polled
// leaves unread payload that a later launch of the same bank would accept (the tagged wire's
// tag would reject it); a peer that pushed fewer makes this rank time out on both wires.

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

// float -> bf16 bits, bit-identical to `hip_bfloat16(v).data` for every one of the 2^32 inputs
// (round to nearest even; Inf passes; a NaN keeps its upper bits and gets the low mantissa bit
// when any of its lower 16 bits is set). Written as selects: the hip_bfloat16 if / else-if
// compiled to divergent branches, ~2000 instructions per granule of the HC apply (measured: the
// apply was 6.2 of the 9.3 us of a 1-row launch). Proved against hip_bfloat16 over all 2^32
// inputs on device by `ar_hc_write_oracle` (rounding check). The host prepends
// `#define ARHC_RND_HIP_BF16` (kill switch `HIPFIRE_EP_AR_FAST_APPLY=0`) to compile the
// hip_bfloat16 casts instead.
#ifdef ARHC_RND_HIP_BF16
__device__ __forceinline__ unsigned arhc_rnd_bits(float v) { return (unsigned)static_cast<hip_bfloat16>(v).data; }
__device__ __forceinline__ float arhc_rnd(float v) { return static_cast<float>(static_cast<hip_bfloat16>(v)); }
#else
__device__ __forceinline__ unsigned arhc_rnd_bits(float v) {
    const unsigned u = __float_as_uint(v);
    const unsigned rne = u + 0x7FFFu + ((u >> 16) & 1u);
    const unsigned nan = (u & 0xFFFFu) != 0u ? (u | 0x10000u) : u;
    return ((~u & 0x7F800000u) != 0u ? rne : nan) >> 16;
}
__device__ __forceinline__ float arhc_rnd(float v) { return __uint_as_float(arhc_rnd_bits(v) << 16); }
#endif
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

// Dense wire: 8-byte granules {d0d1, d2d3} (4 bf16 values), NO tag. A slot is valid when neither
// of its two dwords equals ARHC_EMPTY (all ones); the consumer re-arms what it read.
#define ARHC_EMPTY 0xFFFFFFFFu
typedef unsigned int arhc_u2 __attribute__((ext_vector_type(2)));
template <bool DENSE>
struct ArhcRec {
    using type = arhc_u4;
};
template <>
struct ArhcRec<true> {
    using type = arhc_u2;
};

// A payload dword is never the EMPTY marker. Two bf16 payload halves of 0xFFFF are both NaNs
// (a finite partial never rounds to bf16 0xFFFF), so the remap below changes only the payload
// bits of a NaN pair and no finite value.
__device__ __forceinline__ unsigned arhc_nonempty(unsigned w) { return w == ARHC_EMPTY ? ARHC_EMPTY - 1u : w; }

__device__ __forceinline__ void arhc_st64_sys(arhc_u2 *p, arhc_u2 v) {
    asm volatile("global_store_b64 %0, %1, off scope:SCOPE_SYS" ::"v"(p), "v"(v) : "memory");
}

// b64 form of arhc_load_batch.
template <int K>
__device__ __forceinline__ void arhc_load_batch64(arhc_u2 (&r)[K], const arhc_u2 *const (&p)[K]);

template <>
__device__ __forceinline__ void arhc_load_batch64<1>(arhc_u2 (&r)[1], const arhc_u2 *const (&p)[1]) {
    asm volatile(
        "global_load_b64 %0, %1, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0])
        : "v"(p[0])
        : "memory");
}
template <>
__device__ __forceinline__ void arhc_load_batch64<2>(arhc_u2 (&r)[2], const arhc_u2 *const (&p)[2]) {
    asm volatile(
        "global_load_b64 %0, %2, off scope:SCOPE_SYS\n\t"
        "global_load_b64 %1, %3, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0]), "=&v"(r[1])
        : "v"(p[0]), "v"(p[1])
        : "memory");
}
template <>
__device__ __forceinline__ void arhc_load_batch64<3>(arhc_u2 (&r)[3], const arhc_u2 *const (&p)[3]) {
    asm volatile(
        "global_load_b64 %0, %3, off scope:SCOPE_SYS\n\t"
        "global_load_b64 %1, %4, off scope:SCOPE_SYS\n\t"
        "global_load_b64 %2, %5, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0]), "=&v"(r[1]), "=&v"(r[2])
        : "v"(p[0]), "v"(p[1]), "v"(p[2])
        : "memory");
}
template <>
__device__ __forceinline__ void arhc_load_batch64<4>(arhc_u2 (&r)[4], const arhc_u2 *const (&p)[4]) {
    asm volatile(
        "global_load_b64 %0, %4, off scope:SCOPE_SYS\n\t"
        "global_load_b64 %1, %5, off scope:SCOPE_SYS\n\t"
        "global_load_b64 %2, %6, off scope:SCOPE_SYS\n\t"
        "global_load_b64 %3, %7, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0]), "=&v"(r[1]), "=&v"(r[2]), "=&v"(r[3])
        : "v"(p[0]), "v"(p[1]), "v"(p[2]), "v"(p[3])
        : "memory");
}
template <>
__device__ __forceinline__ void arhc_load_batch64<5>(arhc_u2 (&r)[5], const arhc_u2 *const (&p)[5]) {
    asm volatile(
        "global_load_b64 %0, %5, off scope:SCOPE_SYS\n\t"
        "global_load_b64 %1, %6, off scope:SCOPE_SYS\n\t"
        "global_load_b64 %2, %7, off scope:SCOPE_SYS\n\t"
        "global_load_b64 %3, %8, off scope:SCOPE_SYS\n\t"
        "global_load_b64 %4, %9, off scope:SCOPE_SYS\n\t"
        "s_wait_loadcnt 0"
        : "=&v"(r[0]), "=&v"(r[1]), "=&v"(r[2]), "=&v"(r[3]), "=&v"(r[4])
        : "v"(p[0]), "v"(p[1]), "v"(p[2]), "v"(p[3]), "v"(p[4])
        : "memory");
}
template <>
__device__ __forceinline__ void arhc_load_batch64<6>(arhc_u2 (&r)[6], const arhc_u2 *const (&p)[6]) {
    asm volatile(
        "global_load_b64 %0, %6, off scope:SCOPE_SYS\n\t"
        "global_load_b64 %1, %7, off scope:SCOPE_SYS\n\t"
        "global_load_b64 %2, %8, off scope:SCOPE_SYS\n\t"
        "global_load_b64 %3, %9, off scope:SCOPE_SYS\n\t"
        "global_load_b64 %4, %10, off scope:SCOPE_SYS\n\t"
        "global_load_b64 %5, %11, off scope:SCOPE_SYS\n\t"
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
struct ArhcWord {
    using type = uint2;  // four BF16 bits
};
template <>
struct ArhcWord<true> {
    using type = uint4;  // four F32 words
};
template <bool F32S>
struct ArhcPre {
    typename ArhcWord<F32S>::type w[4];
    float g[4];
};

// Stream words [row, br, col .. col + 4) of one branch.
template <bool F32S>
__device__ __forceinline__ typename ArhcWord<F32S>::type arhc_load_br(const void *__restrict__ streams, int row,
                                                                      int col, int br) {
    const long long e = (long long)row * AR_HC_WIDE + br * AR_HC_HIDDEN + col;
    if constexpr (F32S) {
        return *reinterpret_cast<const uint4 *>(reinterpret_cast<const float *>(streams) + e);
    } else {
        return *reinterpret_cast<const uint2 *>(reinterpret_cast<const unsigned short *>(streams) + e);
    }
}

// The production HC write of one branch's four elements, in place.
template <bool F32S>
__device__ __forceinline__ void arhc_apply_br(void *__restrict__ streams, int row, int col, int br,
                                              const typename ArhcWord<F32S>::type &wv, float g, const float (&m)[4]) {
    const long long e = (long long)row * AR_HC_WIDE + br * AR_HC_HIDDEN + col;
    if constexpr (F32S) {
        // hyper_write_f32: bf16(bf16(input) + bf16(gate * bf16(mixed))), stored as a float.
        const unsigned w[4] = {wv.x, wv.y, wv.z, wv.w};
        unsigned ow[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            ow[q] = __float_as_uint(arhc_rnd(arhc_inj(g, m[q]) + arhc_rnd(__uint_as_float(w[q]))));
        }
        *reinterpret_cast<uint4 *>(reinterpret_cast<float *>(streams) + e) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
    } else {
        const unsigned w[2] = {wv.x, wv.y};
        unsigned o[2];
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const float lo = arhc_inj(g, m[2 * q]) + __uint_as_float(w[q] << 16);
            const float hi = arhc_inj(g, m[2 * q + 1]) + __uint_as_float(w[q] & 0xFFFF0000u);
            o[q] = arhc_rnd_bits(lo) | (arhc_rnd_bits(hi) << 16);
        }
        *reinterpret_cast<uint2 *>(reinterpret_cast<unsigned short *>(streams) + e) = make_uint2(o[0], o[1]);
    }
}

template <bool F32S>
__device__ __forceinline__ void arhc_load(const void *__restrict__ streams, const float *__restrict__ gates,
                                          int row, int col, ArhcPre<F32S> &p) {
    const float4 gv = *reinterpret_cast<const float4 *>(gates + (long long)row * 4);
    p.g[0] = arhc_gate(gv.x); p.g[1] = arhc_gate(gv.y); p.g[2] = arhc_gate(gv.z); p.g[3] = arhc_gate(gv.w);
#pragma unroll
    for (int br = 0; br < 4; ++br) p.w[br] = arhc_load_br<F32S>(streams, row, col, br);
}
template <bool F32S>
__device__ __forceinline__ void arhc_apply(void *__restrict__ streams, int row, int col, const ArhcPre<F32S> &p,
                                           const float (&m)[4]) {
#pragma unroll
    for (int br = 0; br < 4; ++br) arhc_apply_br<F32S>(streams, row, col, br, p.w[br], p.g[br], m);
}

template <int K, bool F32S, bool DENSE>
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
    typename ArhcRec<DENSE>::type *pb = reinterpret_cast<typename ArhcRec<DENSE>::type *>(peer_mbox + bank * bank_stride);
    const typename ArhcRec<DENSE>::type *lb =
        reinterpret_cast<const typename ArhcRec<DENSE>::type *>(local_mbox + bank * bank_stride);
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = g0 + tid + k * T;
        if (j < g1) {
            const unsigned w0 = arhc_bf16bits(pv[k][0]) | (arhc_bf16bits(pv[k][1]) << 16);
            const unsigned w1 = arhc_bf16bits(pv[k][2]) | (arhc_bf16bits(pv[k][3]) << 16);
            if constexpr (DENSE) {
                arhc_st64_sys(pb + j, arhc_u2{arhc_nonempty(w0), arhc_nonempty(w1)});
            } else {
                arhc_st128_sys(pb + j, arhc_u4{w0, tag, w1, tag});
            }
        }
    }
    typename ArhcRec<DENSE>::type r[K];
    const typename ArhcRec<DENSE>::type *lp[K];
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
                if constexpr (DENSE) arhc_load_batch64<K>(r, lp); else arhc_load_batch<K>(r, lp);
                bool ok = true;
#pragma unroll
                for (int k = 0; k < K; ++k) {
                    const int j = g0 + tid + k * T;
                    if (j < g1) {
                        if constexpr (DENSE) {
                            ok &= (r[k].x != ARHC_EMPTY) & (r[k].y != ARHC_EMPTY);
                        } else {
                            ok &= (r[k].y == tag) & (r[k].w == tag);
                        }
                    }
                }
                return ok;
            },
            wb, HF_WAIT_SITE_AR_HC_WRITE_GFX1201, tag)) {
        return false;
    }
    if constexpr (DENSE) {
        // Re-arm: every slot this lane consumed goes back to EMPTY (its payload is already in
        // `r`). The peer cannot write this bank again before this kernel has completed (header).
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const int j = g0 + tid + k * T;
            if (j < g1) arhc_st64_sys(const_cast<arhc_u2 *>(lb) + j, arhc_u2{ARHC_EMPTY, ARHC_EMPTY});
        }
    }
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = g0 + tid + k * T;
        if (j < g1) {
            unsigned rw0, rw1;
            if constexpr (DENSE) {
                rw0 = r[k].x;
                rw1 = r[k].y;
            } else {
                rw0 = r[k].x;
                rw1 = r[k].z;
            }
            const float in[4] = {__uint_as_float(rw0 << 16), __uint_as_float(rw0 & 0xFFFF0000u),
                                 __uint_as_float(rw1 << 16), __uint_as_float(rw1 & 0xFFFF0000u)};
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

template <bool F32S, bool DENSE>
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
        case 1: ok = arhc_body<1, F32S, DENSE>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1, mixed_dbg); break;
        case 2: ok = arhc_body<2, F32S, DENSE>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1, mixed_dbg); break;
        case 3: ok = arhc_body<3, F32S, DENSE>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1, mixed_dbg); break;
        case 4: ok = arhc_body<4, F32S, DENSE>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1, mixed_dbg); break;
        case 5: ok = arhc_body<5, F32S, DENSE>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1, mixed_dbg); break;
        case 6: ok = arhc_body<6, F32S, DENSE>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1, mixed_dbg); break;
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

// Branch-split launch (tagged wire, small row counts; host: ar_hc_split_geometry). One lane
// per (granule, branch) ITEM instead of one lane per granule: item i covers granule j = i >> 2,
// branch br = i & 3. The br == 0 lane pushes the granule (same slot, same tagged format, same
// bank as arhc_body), all four lanes of the item group poll it (the same 16 bytes), fold it
// with the same expression and apply ONLY their branch with ONLY their branch's gate. The
// per-element arithmetic and the wire are those of arhc_body<K, F32S, false>; the apply's ALU
// chain per lane is a quarter as long (the decode apply is ALU-latency-bound). Block b owns
// items [b * per, min(4G, (b + 1) * per)), per % 4 == 0, per <= blockDim (one item per lane).
template <bool F32S>
__device__ __forceinline__ void arhc_split_entry(
    const float *__restrict__ part, void *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int count, float *__restrict__ mixed_dbg) {
    const int I = count;  // 4 items per granule, 4 values per granule
    const int per = (((I + (int)gridDim.x - 1) / (int)gridDim.x) + 3) & ~3;
    const int i0 = (int)blockIdx.x * per;
    const int i1 = min(I, i0 + per);
    unsigned tag = __hip_atomic_load(gen_ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (tag == 0u) tag = 2u;
    bool ok = true;
    const int i = i0 + (int)threadIdx.x;
    if ((int)threadIdx.x < per && i < i1) {
        const int j = i >> 2, br = i & 3;
        const float4 a = *reinterpret_cast<const float4 *>(part + (long long)j * 4);
        const float pv[4] = {a.x, a.y, a.z, a.w};
        const unsigned bank = tag & 1u;
        if (br == 0) {
            const unsigned w0 = arhc_bf16bits(pv[0]) | (arhc_bf16bits(pv[1]) << 16);
            const unsigned w1 = arhc_bf16bits(pv[2]) | (arhc_bf16bits(pv[3]) << 16);
            arhc_st128_sys(reinterpret_cast<arhc_u4 *>(peer_mbox + bank * bank_stride) + j, arhc_u4{w0, tag, w1, tag});
        }
        const long long e0 = (long long)j * 4;
        const int row = (int)(e0 / AR_HC_HIDDEN), col = (int)(e0 % AR_HC_HIDDEN);
        const float g = arhc_gate(gates[(long long)row * 4 + br]);
        const typename ArhcWord<F32S>::type wv = arhc_load_br<F32S>(streams, row, col, br);
        asm volatile("" ::"v"(wv.x), "v"(wv.y));
        if constexpr (F32S) asm volatile("" ::"v"(wv.z), "v"(wv.w));
        arhc_u4 r[1];
        const arhc_u4 *const lp[1] = {reinterpret_cast<const arhc_u4 *>(local_mbox + bank * bank_stride) + j};
        ok = hf_bounded_wait(
            [&] {
                arhc_load_batch<1>(r, lp);
                return (r[0].y == tag) & (r[0].w == tag);
            },
            wb, HF_WAIT_SITE_AR_HC_WRITE_GFX1201, tag);
        if (ok) {
            const unsigned rw0 = r[0].x, rw1 = r[0].z;
            const float in[4] = {__uint_as_float(rw0 << 16), __uint_as_float(rw0 & 0xFFFF0000u),
                                 __uint_as_float(rw1 << 16), __uint_as_float(rw1 & 0xFFFF0000u)};
            float m[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float loc = arhc_bf16r(pv[e]);
                const float fi = local_is_first ? loc : in[e];
                const float si = local_is_first ? in[e] : loc;
                float xi = 0.0f;
                xi = xi + (fi + si);
                m[e] = xi;
            }
            if (mixed_dbg && br == 0) {
#pragma unroll
                for (int e = 0; e < 4; ++e) mixed_dbg[e0 + e] = m[e];
            }
            arhc_apply_br<F32S>(streams, row, col, br, wv, g, m);
        }
    }
    // Generation advance: as in arhc_entry.
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

// Entries. Four ABI-identical kernels: {BF16, F32} streams x {tagged, dense} wire.
//   tagged (`ar_hc_write_*`): the 16-byte tagged granule (kill switch, HIPFIRE_EP_AR_WIRE_DENSE=0).
//   dense  (`ar_hc_write_d8_*`): the 8-byte untagged granule with receiver re-arm (see header).

// BF16-bit streams (`[rows][4][2560]` u16).
extern "C" __global__ __launch_bounds__(512) void ar_hc_write_gfx1201(
    const float *__restrict__ part, unsigned short *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int count, float *__restrict__ mixed_dbg) {
    arhc_entry<false, false>(part, streams, gates, peer_mbox, local_mbox, bank_stride, gen_ptr, wb, local_is_first,
                             count, mixed_dbg);
}

// F32 streams (`[rows][4][2560]` f32 holding BF16-exact values), same ABI.
extern "C" __global__ __launch_bounds__(512) void ar_hc_write_f32s_gfx1201(
    const float *__restrict__ part, float *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int count, float *__restrict__ mixed_dbg) {
    arhc_entry<true, false>(part, streams, gates, peer_mbox, local_mbox, bank_stride, gen_ptr, wb, local_is_first,
                            count, mixed_dbg);
}

// Dense wire, BF16-bit streams.
extern "C" __global__ __launch_bounds__(512) void ar_hc_write_d8_gfx1201(
    const float *__restrict__ part, unsigned short *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int count, float *__restrict__ mixed_dbg) {
    arhc_entry<false, true>(part, streams, gates, peer_mbox, local_mbox, bank_stride, gen_ptr, wb, local_is_first,
                            count, mixed_dbg);
}

// Dense wire, F32 streams.
extern "C" __global__ __launch_bounds__(512) void ar_hc_write_d8_f32s_gfx1201(
    const float *__restrict__ part, float *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int count, float *__restrict__ mixed_dbg) {
    arhc_entry<true, true>(part, streams, gates, peer_mbox, local_mbox, bank_stride, gen_ptr, wb, local_is_first,
                           count, mixed_dbg);
}

// Branch-split tagged entries (arhc_split_entry), same ABI: decode-size launches.
extern "C" __global__ __launch_bounds__(512) void ar_hc_write_s_gfx1201(
    const float *__restrict__ part, unsigned short *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int count, float *__restrict__ mixed_dbg) {
    arhc_split_entry<false>(part, streams, gates, peer_mbox, local_mbox, bank_stride, gen_ptr, wb, local_is_first,
                            count, mixed_dbg);
}

extern "C" __global__ __launch_bounds__(512) void ar_hc_write_s_f32s_gfx1201(
    const float *__restrict__ part, float *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int count, float *__restrict__ mixed_dbg) {
    arhc_split_entry<true>(part, streams, gates, peer_mbox, local_mbox, bank_stride, gen_ptr, wb, local_is_first,
                           count, mixed_dbg);
}
