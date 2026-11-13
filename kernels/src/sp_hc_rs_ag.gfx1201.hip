// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Sequence-parallel (SP) prefill exchange for the gfx1201 EP2 Flash-Next trunk: a reduce-scatter
// fused into the BF16 Hyper-Connection write (`sp_hc_rs_d8_*`) and the all-gather of per-row
// activations the next sharded op needs (`sp_ag_rows_*`).
//
// This file is APPENDED to the helpers of `ar_hc_write.gfx1201.hip` (module source =
// device_wait_deadline.h + ar_hc_write.gfx1201.hip + this file): it reuses ArhcRec,
// arhc_push / arhc_valid / arhc_unpack / arhc_rearm / arhc_fold of the dense wire (both partial
// formats), arhc_nonempty, arhc_st64_sys, arhc_load_recs<K>, ArhcPre/arhc_load/arhc_apply,
// hf_bounded_wait, the dense (EMPTY = 0xFFFFFFFF, consumer re-arm) wire and the generation /
// ticket protocol unchanged. The wire argument (torn / stale / payload-EMPTY / abort) in the
// header of ar_hc_write.gfx1201.hip applies verbatim: only WHICH granules travel and WHO
// consumes them differ.
//
// Reduce-scatter (RS). Today `ar_hc_write_d8_*` pushes the whole partial tile to the peer and
// both ranks fold and write every row. Here a launch names two row ranges of the same tile
// (`part`, `streams`, `gates` are the FULL tile base pointers, row r at r*2560 floats /
// r*4*2560 stream elements / r*4 gates):
//   own  = [own_row0,  own_row0  + own_rows)   rows THIS rank folds and writes,
//   peer = [peer_row0, peer_row0 + peer_rows)  rows the PEER folds and writes.
//   (a) push the partial of the PEER range as dense granules (bf16 partials `sp_hc_rs_d8_*`:
//       8 bytes; f32 partials `sp_hc_rs_p32_d16_*`: 16 bytes) into the peer mailbox bank
//       `tag & 1`; slot index = granule index relative to peer_row0 (slot 0 = first granule of
//       peer_row0);
//   (b) bounded poll (site SP_HC_RS_GFX1201) of this rank's own mailbox slots
//       0 .. own_rows * 640: the peer pushed its PEER range = this rank's own range, in the
//       same relative order. The poll side uses the K-templated batch (K <= AR_HC_KMAX);
//   (c) re-arm every consumed slot to EMPTY;
//   (d) fold the own rows exactly like arhc_body: arhc_fold<P32>(local partial, peer partial)
//       in rank order;
//   (e) arhc_apply on the own rows only. Rows outside the own range of `streams` are never
//       written. `mixed_dbg` (nullable) receives m of the own rows at the full-tile offset.
// The folded value of an element is bit-identical to the all-reduce's of the same partial
// format, so the own rows of a reduce-scatter launch equal the same rows of an
// `ar_hc_write_d8_*` / `ar_hc_write_p32_d16_*` launch.
// A rank with own_rows == 0 still pushes and still advances the generation; the push side
// and the poll side may have different granule counts (uneven splits), so every block takes a
// slice of BOTH.
//
// All-gather (AG). `sp_ag_rows_bf16_d8_gfx1201` exchanges rows of a `[rows][row_elems]` buffer
// (row_elems % 4 == 0): this rank pushes its own rows (dense 8-byte granules of four bf16
// values, slot index relative to own_row0), polls the peer's rows (peer_rows * row_elems / 4
// granules), writes them into the buffer rows of the peer range and re-arms. Payload dwords go
// through arhc_nonempty (a NaN-pair dword 0xFFFFFFFF arrives as 0xFFFFFFFE; every finite value
// is byte-exact).
// Element formats of the AG buffer:
//   bf16    (`sp_ag_rows_bf16_d8_gfx1201`):    buffer u16 bits, two per dword;
//   f32bf16 (`sp_ag_rows_f32bf16_d8_gfx1201`): buffer f32 words that hold BF16-EXACT values
//           (hc_mixed: hyper_read stores bf16r(value / 4)); the push side sends the upper 16
//           bits of each value, the receive side stores `bits << 16`. A value that is not
//           bf16-exact is truncated.
//
// Generation / tag / ticket: read the generation word at the start (first statement), tag =
// gen + 1 (0xFFFFFFFF wraps to 2), bank = tag & 1, the last block to arrive stores the new
// generation. A timed-out wait never reaches the ticket (the host resets both ranks).
//
// Mailbox banks are `bank_stride` bytes apart; a launch pushes and polls at most
// bank_stride / (granule bytes) granules: 8 (bf16 RS, AG) or 16 (f32 RS) bytes each (host checks it).
//
// Geometry (host: sp_rs_geometry / sp_ag_geometry): grid <= 64 blocks, <= 512 lanes, the poll
// side at most AR_HC_KMAX granules per lane.

#define HF_WAIT_SITE_SP_HC_RS_GFX1201 12u
#define HF_WAIT_SITE_SP_AG_ROWS_GFX1201 13u

#define SP_GRANULES_PER_ROW (AR_HC_HIDDEN / 4)

// Plain-load push of `p0..p1` (granule indices relative to the start of `src`): four granules
// in flight per lane.
template <bool P32>
__device__ __forceinline__ void sp_push(const float *__restrict__ src, typename ArhcRec<true, P32>::type *pb, int p0,
                                        int p1) {
    const int tid = threadIdx.x, T = blockDim.x;
    for (int j0 = p0 + tid; j0 < p1; j0 += 4 * T) {
        float4 a[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int j = j0 + u * T;
            a[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (j < p1) a[u] = *reinterpret_cast<const float4 *>(src + (long long)j * 4);
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int j = j0 + u * T;
            if (j < p1) {
                const float v[4] = {a[u].x, a[u].y, a[u].z, a[u].w};
                arhc_push<true, P32>(pb, j, 0, v, 0u);  // dense: one record per granule, no plane
            }
        }
    }
}

// Arrival ticket + generation advance (the tail of arhc_entry).
__device__ __forceinline__ void sp_arrive(bool ok, unsigned *__restrict__ gen_ptr, unsigned tag) {
    // Every lane of the block has read the generation word before this barrier (it is the first
    // statement of the entry), so the single advance below cannot race a reader of this launch.
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

// ---------------------------------------------------------------------------------------
// Reduce-scatter
// ---------------------------------------------------------------------------------------

// Poll / fold / apply of the own granules g0..g1 (relative to own_row0) of this block.
template <int K, bool F32S, bool P32>
__device__ __forceinline__ bool sp_rs_body(
    const float *__restrict__ part, void *__restrict__ streams, const float *__restrict__ gates,
    const char *local_mbox, unsigned bank_stride, unsigned tag, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int own_row0, int g0, int g1, float *__restrict__ mixed_dbg) {
    using Rec = typename ArhcRec<true, P32>::type;
    const int tid = threadIdx.x, T = blockDim.x;
    const long long base = (long long)own_row0 * AR_HC_HIDDEN;
    float pv[K][4];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = g0 + tid + k * T;
        if (j < g1) {
            const float4 a = *reinterpret_cast<const float4 *>(part + base + (long long)j * 4);
            pv[k][0] = a.x; pv[k][1] = a.y; pv[k][2] = a.z; pv[k][3] = a.w;
        }
    }
    const Rec *lb = reinterpret_cast<const Rec *>(local_mbox + (tag & 1u) * bank_stride);
    Rec r[K];
    const Rec *lp[K];
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
            const long long e0 = base + (long long)j * 4;
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
                arhc_load_recs<K>(r, lp);
                bool ok = true;
#pragma unroll
                for (int k = 0; k < K; ++k) {
                    const int j = g0 + tid + k * T;
                    if (j < g1) ok &= arhc_valid<true, P32>(r + k, tag);
                }
                return ok;
            },
            wb, HF_WAIT_SITE_SP_HC_RS_GFX1201, tag)) {
        return false;
    }
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = g0 + tid + k * T;
        if (j < g1) arhc_rearm<P32>(const_cast<Rec *>(lb), j);
    }
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = g0 + tid + k * T;
        if (j < g1) {
            float in[4];
            arhc_unpack<true, P32>(r + k, in);
            float m[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) m[e] = arhc_fold<P32>(pv[k][e], in[e], local_is_first);
            const long long e0 = base + (long long)j * 4;
            if (mixed_dbg) {
#pragma unroll
                for (int e = 0; e < 4; ++e) mixed_dbg[e0 + e] = m[e];
            }
            arhc_apply<F32S>(streams, (int)(e0 / AR_HC_HIDDEN), (int)(e0 % AR_HC_HIDDEN), pre[k], m);
        }
    }
    return true;
}

template <bool F32S, bool P32>
__device__ __forceinline__ void sp_rs_entry(
    const float *__restrict__ part, void *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int own_row0, int own_rows, int peer_row0, int peer_rows,
    float *__restrict__ mixed_dbg) {
    unsigned tag = __hip_atomic_load(gen_ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (tag == 0u) tag = 2u;
    const int nb = (int)gridDim.x;
    const int Go = own_rows * SP_GRANULES_PER_ROW;
    const int Gp = peer_rows * SP_GRANULES_PER_ROW;
    const int perP = (Gp + nb - 1) / nb;
    const int p0 = (int)blockIdx.x * perP;
    const int p1 = min(Gp, p0 + perP);
    const int perO = (Go + nb - 1) / nb;
    const int o0 = (int)blockIdx.x * perO;
    const int o1 = min(Go, o0 + perO);
    // (a) push the peer range
    if (p0 < p1) {
        auto *pb = reinterpret_cast<typename ArhcRec<true, P32>::type *>(peer_mbox + (tag & 1u) * bank_stride);
        sp_push<P32>(part + (long long)peer_row0 * AR_HC_HIDDEN, pb, p0, p1);
    }
    // (b)-(e) own range
    bool ok = true;
    if (o0 < o1) {
        const int K = (o1 - o0 + (int)blockDim.x - 1) / (int)blockDim.x;
        switch (K) {
        case 1: ok = sp_rs_body<1, F32S, P32>(part, streams, gates, local_mbox, bank_stride, tag, wb, local_is_first, own_row0, o0, o1, mixed_dbg); break;
        case 2: ok = sp_rs_body<2, F32S, P32>(part, streams, gates, local_mbox, bank_stride, tag, wb, local_is_first, own_row0, o0, o1, mixed_dbg); break;
        case 3: ok = sp_rs_body<3, F32S, P32>(part, streams, gates, local_mbox, bank_stride, tag, wb, local_is_first, own_row0, o0, o1, mixed_dbg); break;
        case 4: ok = sp_rs_body<4, F32S, P32>(part, streams, gates, local_mbox, bank_stride, tag, wb, local_is_first, own_row0, o0, o1, mixed_dbg); break;
        case 5: ok = sp_rs_body<5, F32S, P32>(part, streams, gates, local_mbox, bank_stride, tag, wb, local_is_first, own_row0, o0, o1, mixed_dbg); break;
        case 6: ok = sp_rs_body<6, F32S, P32>(part, streams, gates, local_mbox, bank_stride, tag, wb, local_is_first, own_row0, o0, o1, mixed_dbg); break;
        default: ok = false; break;
        }
    }
    sp_arrive(ok, gen_ptr, tag);
}

// bf16 partials (FFN), BF16-bit streams (`[rows][4][2560]` u16). Arguments: those of
// ar_hc_write_d8_gfx1201 with `count` replaced by the four row-range ints.
extern "C" __global__ __launch_bounds__(512) void sp_hc_rs_d8_gfx1201(
    const float *__restrict__ part, unsigned short *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int own_row0, int own_rows, int peer_row0, int peer_rows,
    float *__restrict__ mixed_dbg) {
    sp_rs_entry<false, false>(part, streams, gates, peer_mbox, local_mbox, bank_stride, gen_ptr, wb, local_is_first,
        own_row0, own_rows, peer_row0, peer_rows, mixed_dbg);
}

// F32 streams (`[rows][4][2560]` f32 holding BF16-exact values), same ABI.
extern "C" __global__ __launch_bounds__(512) void sp_hc_rs_d8_f32s_gfx1201(
    const float *__restrict__ part, float *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int own_row0, int own_rows, int peer_row0, int peer_rows,
    float *__restrict__ mixed_dbg) {
    sp_rs_entry<true, false>(part, streams, gates, peer_mbox, local_mbox, bank_stride, gen_ptr, wb, local_is_first,
        own_row0, own_rows, peer_row0, peer_rows, mixed_dbg);
}

// f32 partials (mixer out-projection), same ABI.
extern "C" __global__ __launch_bounds__(512) void sp_hc_rs_p32_d16_gfx1201(
    const float *__restrict__ part, unsigned short *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int own_row0, int own_rows, int peer_row0, int peer_rows,
    float *__restrict__ mixed_dbg) {
    sp_rs_entry<false, true>(part, streams, gates, peer_mbox, local_mbox, bank_stride, gen_ptr, wb, local_is_first,
        own_row0, own_rows, peer_row0, peer_rows, mixed_dbg);
}

extern "C" __global__ __launch_bounds__(512) void sp_hc_rs_p32_d16_f32s_gfx1201(
    const float *__restrict__ part, float *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int own_row0, int own_rows, int peer_row0, int peer_rows,
    float *__restrict__ mixed_dbg) {
    sp_rs_entry<true, true>(part, streams, gates, peer_mbox, local_mbox, bank_stride, gen_ptr, wb, local_is_first,
        own_row0, own_rows, peer_row0, peer_rows, mixed_dbg);
}

// ---------------------------------------------------------------------------------------
// All-gather
// ---------------------------------------------------------------------------------------

// Wire words of granule `j` (four elements) of the buffer rows starting at `src`.
template <bool F32B>
__device__ __forceinline__ arhc_u2 sp_ag_load(const unsigned char *src, long long j) {
    if constexpr (F32B) {
        const uint4 a = *reinterpret_cast<const uint4 *>(src + j * 16);
        return arhc_u2{arhc_nonempty((a.x >> 16) | (a.y & 0xFFFF0000u)),
                       arhc_nonempty((a.z >> 16) | (a.w & 0xFFFF0000u))};
    } else {
        const uint2 a = *reinterpret_cast<const uint2 *>(src + j * 8);
        return arhc_u2{arhc_nonempty(a.x), arhc_nonempty(a.y)};
    }
}

template <bool F32B>
__device__ __forceinline__ void sp_ag_store(unsigned char *dst, long long j, arhc_u2 w) {
    if constexpr (F32B) {
        *reinterpret_cast<uint4 *>(dst + j * 16) =
            make_uint4(w.x << 16, w.x & 0xFFFF0000u, w.y << 16, w.y & 0xFFFF0000u);
    } else {
        *reinterpret_cast<uint2 *>(dst + j * 8) = make_uint2(w.x, w.y);
    }
}

// Poll granules q0..q1 (relative to the start of `dst`, the first peer row) of this block.
template <int K, bool F32B>
__device__ __forceinline__ bool sp_ag_poll_body(
    unsigned char *dst, const char *local_mbox, unsigned bank_stride, unsigned tag,
    hf_device_wait_block *__restrict__ wb, int q0, int q1) {
    const int tid = threadIdx.x, T = blockDim.x;
    const arhc_u2 *lb = reinterpret_cast<const arhc_u2 *>(local_mbox + (tag & 1u) * bank_stride);
    arhc_u2 r[K];
    const arhc_u2 *lp[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = q0 + tid + k * T;
        lp[k] = lb + (j < q1 ? j : q1 - 1);
    }
    if (!hf_bounded_wait(
            [&] {
                arhc_load_batch64<K>(r, lp);
                bool ok = true;
#pragma unroll
                for (int k = 0; k < K; ++k) {
                    const int j = q0 + tid + k * T;
                    if (j < q1) ok &= (r[k].x != ARHC_EMPTY) & (r[k].y != ARHC_EMPTY);
                }
                return ok;
            },
            wb, HF_WAIT_SITE_SP_AG_ROWS_GFX1201, tag)) {
        return false;
    }
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = q0 + tid + k * T;
        if (j < q1) arhc_st64_sys(const_cast<arhc_u2 *>(lb) + j, arhc_u2{ARHC_EMPTY, ARHC_EMPTY});
    }
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = q0 + tid + k * T;
        if (j < q1) sp_ag_store<F32B>(dst, j, r[k]);
    }
    return true;
}

template <bool F32B>
__device__ __forceinline__ void sp_ag_entry(
    unsigned char *buf, int row_elems, char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int own_row0, int own_rows, int peer_row0, int peer_rows) {
    unsigned tag = __hip_atomic_load(gen_ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (tag == 0u) tag = 2u;
    const int nb = (int)gridDim.x;
    const int gpr = row_elems >> 2;
    const int Go = own_rows * gpr;
    const int Gp = peer_rows * gpr;
    const long long row_bytes = (long long)row_elems * (F32B ? 4 : 2);
    // push own rows
    const int perP = (Go + nb - 1) / nb;
    const int p0 = (int)blockIdx.x * perP;
    const int p1 = min(Go, p0 + perP);
    if (p0 < p1) {
        const unsigned char *src = buf + (long long)own_row0 * row_bytes;
        arhc_u2 *pb = reinterpret_cast<arhc_u2 *>(peer_mbox + (tag & 1u) * bank_stride);
        const int tid = threadIdx.x, T = blockDim.x;
        for (int j0 = p0 + tid; j0 < p1; j0 += 4 * T) {
            arhc_u2 v[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int j = j0 + u * T;
                v[u] = arhc_u2{0u, 0u};
                if (j < p1) v[u] = sp_ag_load<F32B>(src, j);
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int j = j0 + u * T;
                if (j < p1) arhc_st64_sys(pb + j, v[u]);
            }
        }
    }
    // poll the peer rows
    const int perQ = (Gp + nb - 1) / nb;
    const int q0 = (int)blockIdx.x * perQ;
    const int q1 = min(Gp, q0 + perQ);
    bool ok = true;
    if (q0 < q1) {
        unsigned char *dst = buf + (long long)peer_row0 * row_bytes;
        const int K = (q1 - q0 + (int)blockDim.x - 1) / (int)blockDim.x;
        switch (K) {
        case 1: ok = sp_ag_poll_body<1, F32B>(dst, local_mbox, bank_stride, tag, wb, q0, q1); break;
        case 2: ok = sp_ag_poll_body<2, F32B>(dst, local_mbox, bank_stride, tag, wb, q0, q1); break;
        case 3: ok = sp_ag_poll_body<3, F32B>(dst, local_mbox, bank_stride, tag, wb, q0, q1); break;
        case 4: ok = sp_ag_poll_body<4, F32B>(dst, local_mbox, bank_stride, tag, wb, q0, q1); break;
        case 5: ok = sp_ag_poll_body<5, F32B>(dst, local_mbox, bank_stride, tag, wb, q0, q1); break;
        case 6: ok = sp_ag_poll_body<6, F32B>(dst, local_mbox, bank_stride, tag, wb, q0, q1); break;
        default: ok = false; break;
        }
    }
    sp_arrive(ok, gen_ptr, tag);
}

// bf16 bits buffer (`[rows][row_elems]` u16, 8-byte aligned).
extern "C" __global__ __launch_bounds__(512) void sp_ag_rows_bf16_d8_gfx1201(
    unsigned short *buf, int row_elems, char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int own_row0, int own_rows, int peer_row0, int peer_rows) {
    sp_ag_entry<false>(reinterpret_cast<unsigned char *>(buf), row_elems, peer_mbox, local_mbox, bank_stride,
                       gen_ptr, wb, own_row0, own_rows, peer_row0, peer_rows);
}

// f32 buffer of BF16-exact values (`[rows][row_elems]` f32, 16-byte aligned), same ABI.
extern "C" __global__ __launch_bounds__(512) void sp_ag_rows_f32bf16_d8_gfx1201(
    float *buf, int row_elems, char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int own_row0, int own_rows, int peer_row0, int peer_rows) {
    sp_ag_entry<true>(reinterpret_cast<unsigned char *>(buf), row_elems, peer_mbox, local_mbox, bank_stride,
                      gen_ptr, wb, own_row0, own_rows, peer_row0, peer_rows);
}
