// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `ar_hc_write_bench --breakdown`: phase breakdown of the fused AR + HC write launch.
//!
//! Lab copies of the product kernel body (`kernels/src/ar_hc_write.gfx1201.hip`, its helpers are
//! reused verbatim: the lab source is appended to the product source) with compile-time phase
//! switches, run on a lab mailbox / generation / wait block of their own (never the product
//! exchange's), so phases can be removed one at a time:
//!   m0  empty launch (same geometry)                          -> launch floor
//!   m1  load partial + push                                   -> + push
//!   m3  push + bounded poll (+ dense re-arm)                  -> + rendezvous poll
//!   m7  push + poll + fold + HC apply (stream/gate prefetch)  -> + fold/apply
//!   m15 m7 + device generation + arrival ticket               -> + ticket tail (= product body)
//!   m4  fold + HC apply only (local data, no wire)            -> local apply cost
//!   m8  generation read + ticket only
//!   m11 push + poll + ticket (no apply)
//!   m31 m15 + realtime stamps per block (t0 entry, t1 push issued, t2 poll done, t3 apply
//!       issued, t4 after ticket) -> in-kernel phase times and the kernel-to-kernel gap.
//! Modes without the ticket take the tag from a kernarg the host advances per launch (the same
//! sequence on both ranks), so the poll still rendezvous on fresh data.
//! Every pipelined series is gated: a spin kernel (~3 ms) runs first and the start event is
//! recorded after it, so all `iters` launches are already queued when the GPU reaches them and
//! the series measures GPU-side back-to-back time, not the host enqueue rate. The product launch
//! is timed both gated and ungated (`prod_pipe_ungated` = the method of the original table).

use hip_bridge::{DeviceBuffer, KernargBlob};
use hipfire_runtime::ar_hc_exchange::ArHcExchange;
use rdna_compute::ar_hc_write::{ar_hc_geometry, StreamKind, AR_HC_HIDDEN};
use rdna_compute::Gpu;
use std::ffi::c_void;
use std::sync::{Arc, Barrier};

pub const LAB_SRC: &str = r#"
// ---- ar_hc_write_bench --breakdown lab kernels (NOT product kernels) ----
#define ARLAB_PUSH 1
#define ARLAB_POLL 2
#define ARLAB_APPLY 4
#define ARLAB_TICKET 8
#define ARLAB_STAMP 16
#define ARLAB_SLOTS 256

template <int K, bool F32S, bool DENSE, int MODE>
__device__ __forceinline__ bool arlab_body(
    const float *__restrict__ part, void *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride, unsigned tag,
    hf_device_wait_block *__restrict__ wb, int local_is_first, int g0, int g1, unsigned long long *st) {
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
    if constexpr ((MODE & ARLAB_PUSH) != 0) {
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
    }
    if constexpr ((MODE & ARLAB_STAMP) != 0) { if (tid == 0) st[1] = wall_clock64(); }
    ArhcPre<F32S> pre[K];
    if constexpr ((MODE & ARLAB_APPLY) != 0) {
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
    }
    unsigned rw[K][2];
    if constexpr ((MODE & ARLAB_POLL) != 0) {
        typename ArhcRec<DENSE>::type r[K];
        const typename ArhcRec<DENSE>::type *lp[K];
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const int j = g0 + tid + k * T;
            lp[k] = lb + (j < g1 ? j : g1 - 1);
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
#pragma unroll
            for (int k = 0; k < K; ++k) {
                const int j = g0 + tid + k * T;
                if (j < g1) arhc_st64_sys(const_cast<arhc_u2 *>(lb) + j, arhc_u2{ARHC_EMPTY, ARHC_EMPTY});
            }
        }
#pragma unroll
        for (int k = 0; k < K; ++k) {
            if constexpr (DENSE) { rw[k][0] = r[k].x; rw[k][1] = r[k].y; }
            else { rw[k][0] = r[k].x; rw[k][1] = r[k].z; }
        }
    } else {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            rw[k][0] = arhc_bf16bits(pv[k][0]) | (arhc_bf16bits(pv[k][1]) << 16);
            rw[k][1] = arhc_bf16bits(pv[k][2]) | (arhc_bf16bits(pv[k][3]) << 16);
        }
    }
    if constexpr ((MODE & ARLAB_STAMP) != 0) { if (tid == 0) st[2] = wall_clock64(); }
    if constexpr ((MODE & (ARLAB_APPLY | ARLAB_POLL)) != 0) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const int j = g0 + tid + k * T;
            if (j < g1) {
                const float in[4] = {__uint_as_float(rw[k][0] << 16), __uint_as_float(rw[k][0] & 0xFFFF0000u),
                                     __uint_as_float(rw[k][1] << 16), __uint_as_float(rw[k][1] & 0xFFFF0000u)};
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
                if constexpr ((MODE & ARLAB_APPLY) != 0) {
                    const long long e0 = (long long)j * 4;
                    arhc_apply<F32S>(streams, (int)(e0 / AR_HC_HIDDEN), (int)(e0 % AR_HC_HIDDEN), pre[k], m);
                } else {
                    asm volatile("" ::"v"(m[0]), "v"(m[1]), "v"(m[2]), "v"(m[3]));
                }
            }
        }
    }
    if constexpr ((MODE & ARLAB_STAMP) != 0) { if (tid == 0) st[3] = wall_clock64(); }
    return true;
}

// ROLES (dense wire, full AR semantics): lanes [0, T/2) only push this block's granules
// [g0, g1) (four in flight per lane, stride T/2), lanes [T/2, T) only poll / re-arm / fold /
// apply them in rounds of up to AR_HC_KMAX granules per lane (stride T/2), so the apply of
// early-arriving granules overlaps the push of later ones.
#define ARLAB_ROLES 64
template <int K, bool F32S>
__device__ __forceinline__ bool arlab_roles_round(
    const float *__restrict__ part, void *__restrict__ streams, const float *__restrict__ gates,
    const arhc_u2 *lb, unsigned tag, hf_device_wait_block *__restrict__ wb, int local_is_first, int jb, int S, int g1) {
    float pv[K][4];
    arhc_u2 r[K];
    const arhc_u2 *lp[K];
    ArhcPre<F32S> pre[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = jb + k * S;
        lp[k] = lb + (j < g1 ? j : g1 - 1);
        if (j < g1) {
            const float4 a = *reinterpret_cast<const float4 *>(part + (long long)j * 4);
            pv[k][0] = a.x; pv[k][1] = a.y; pv[k][2] = a.z; pv[k][3] = a.w;
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
                arhc_load_batch64<K>(r, lp);
                bool ok = true;
#pragma unroll
                for (int k = 0; k < K; ++k) {
                    const int j = jb + k * S;
                    if (j < g1) ok &= (r[k].x != ARHC_EMPTY) & (r[k].y != ARHC_EMPTY);
                }
                return ok;
            },
            wb, HF_WAIT_SITE_AR_HC_WRITE_GFX1201, tag)) {
        return false;
    }
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = jb + k * S;
        if (j < g1) arhc_st64_sys(const_cast<arhc_u2 *>(lb) + j, arhc_u2{ARHC_EMPTY, ARHC_EMPTY});
    }
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int j = jb + k * S;
        if (j < g1) {
            const unsigned rw0 = r[k].x, rw1 = r[k].y;
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
            arhc_apply<F32S>(streams, (int)(e0 / AR_HC_HIDDEN), (int)(e0 % AR_HC_HIDDEN), pre[k], m);
        }
    }
    return true;
}

template <bool F32S>
__device__ __forceinline__ bool arlab_roles_body(
    const float *__restrict__ part, void *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride, unsigned tag,
    hf_device_wait_block *__restrict__ wb, int local_is_first, int g0, int g1) {
    const int T = (int)blockDim.x, H = T / 2, tid = (int)threadIdx.x;
    const unsigned bank = tag & 1u;
    if (tid < H) {
        arhc_u2 *pb = reinterpret_cast<arhc_u2 *>(peer_mbox + bank * bank_stride);
        for (int j0 = g0 + tid; j0 < g1; j0 += 4 * H) {
            float4 a[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int j = j0 + u * H;
                a[u] = make_float4(0.f, 0.f, 0.f, 0.f);
                if (j < g1) a[u] = *reinterpret_cast<const float4 *>(part + (long long)j * 4);
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int j = j0 + u * H;
                if (j < g1) {
                    const unsigned w0 = arhc_bf16bits(a[u].x) | (arhc_bf16bits(a[u].y) << 16);
                    const unsigned w1 = arhc_bf16bits(a[u].z) | (arhc_bf16bits(a[u].w) << 16);
                    arhc_st64_sys(pb + j, arhc_u2{arhc_nonempty(w0), arhc_nonempty(w1)});
                }
            }
        }
        return true;
    }
    const arhc_u2 *lb = reinterpret_cast<const arhc_u2 *>(local_mbox + bank * bank_stride);
    bool ok = true;
    for (int jb = g0 + (tid - H); jb < g1 && ok; jb += AR_HC_KMAX * H) {
        switch (min(AR_HC_KMAX, (g1 - jb + H - 1) / H)) {
        case 1: ok = arlab_roles_round<1, F32S>(part, streams, gates, lb, tag, wb, local_is_first, jb, H, g1); break;
        case 2: ok = arlab_roles_round<2, F32S>(part, streams, gates, lb, tag, wb, local_is_first, jb, H, g1); break;
        case 3: ok = arlab_roles_round<3, F32S>(part, streams, gates, lb, tag, wb, local_is_first, jb, H, g1); break;
        case 4: ok = arlab_roles_round<4, F32S>(part, streams, gates, lb, tag, wb, local_is_first, jb, H, g1); break;
        case 5: ok = arlab_roles_round<5, F32S>(part, streams, gates, lb, tag, wb, local_is_first, jb, H, g1); break;
        default: ok = arlab_roles_round<6, F32S>(part, streams, gates, lb, tag, wb, local_is_first, jb, H, g1); break;
        }
    }
    return ok;
}

template <bool F32S, bool DENSE, int MODE>
__device__ __forceinline__ void arlab_entry(
    const float *__restrict__ part, void *__restrict__ streams, const float *__restrict__ gates,
    char *peer_mbox, const char *local_mbox, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int count, unsigned host_tag, unsigned long long *stamps) {
    unsigned long long t0 = 0;
    if constexpr ((MODE & ARLAB_STAMP) != 0) t0 = wall_clock64();
    const int G = count / 4;
    const int per = (G + (int)gridDim.x - 1) / (int)gridDim.x;
    const int g0 = (int)blockIdx.x * per;
    const int g1 = min(G, g0 + per);
    unsigned tag = host_tag;
    if constexpr ((MODE & ARLAB_TICKET) != 0) {
        tag = __hip_atomic_load(gen_ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
        if (tag == 0u) tag = 2u;
    }
    unsigned long long *st = nullptr;
    if constexpr ((MODE & ARLAB_STAMP) != 0) {
        st = stamps + ((unsigned long long)(tag % ARLAB_SLOTS) * gridDim.x + blockIdx.x) * 8ull;
        if (threadIdx.x == 0) st[0] = t0;
    }
    bool ok = true;
    if constexpr ((MODE & ARLAB_ROLES) != 0) {
        if (g0 < g1) ok = arlab_roles_body<F32S>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1);
    } else if constexpr ((MODE & (ARLAB_PUSH | ARLAB_POLL | ARLAB_APPLY)) != 0) {
        if (g0 < g1) {
            const int K = (g1 - g0 + (int)blockDim.x - 1) / (int)blockDim.x;
            switch (K) {
            case 1: ok = arlab_body<1, F32S, DENSE, MODE>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1, st); break;
            case 2: ok = arlab_body<2, F32S, DENSE, MODE>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1, st); break;
            case 3: ok = arlab_body<3, F32S, DENSE, MODE>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1, st); break;
            case 4: ok = arlab_body<4, F32S, DENSE, MODE>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1, st); break;
            case 5: ok = arlab_body<5, F32S, DENSE, MODE>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1, st); break;
            case 6: ok = arlab_body<6, F32S, DENSE, MODE>(part, streams, gates, peer_mbox, local_mbox, bank_stride, tag, wb, local_is_first, g0, g1, st); break;
            default: ok = false; break;
            }
        }
    }
    if constexpr ((MODE & ARLAB_TICKET) != 0) {
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
    if constexpr ((MODE & ARLAB_STAMP) != 0) { if (threadIdx.x == 0) st[4] = wall_clock64(); }
}

#define ARLAB_ENTRY(NAME, F32S, DENSE, MODE)                                                              \
    extern "C" __global__ __launch_bounds__(512) void NAME(                                               \
        const float *__restrict__ part, void *__restrict__ streams, const float *__restrict__ gates,      \
        char *peer_mbox, const char *local_mbox, unsigned bank_stride, unsigned *__restrict__ gen_ptr,    \
        hf_device_wait_block *__restrict__ wb, int local_is_first, int count, unsigned host_tag,          \
        unsigned long long *stamps) {                                                                     \
        arlab_entry<F32S, DENSE, MODE>(part, streams, gates, peer_mbox, local_mbox, bank_stride, gen_ptr, \
                                       wb, local_is_first, count, host_tag, stamps);                     \
    }
#define ARLAB_MODES(PFX, F32S, DENSE)        \
    ARLAB_ENTRY(PFX##_m0, F32S, DENSE, 0)    \
    ARLAB_ENTRY(PFX##_m1, F32S, DENSE, 1)    \
    ARLAB_ENTRY(PFX##_m3, F32S, DENSE, 3)    \
    ARLAB_ENTRY(PFX##_m4, F32S, DENSE, 4)    \
    ARLAB_ENTRY(PFX##_m7, F32S, DENSE, 7)    \
    ARLAB_ENTRY(PFX##_m8, F32S, DENSE, 8)    \
    ARLAB_ENTRY(PFX##_m11, F32S, DENSE, 11)  \
    ARLAB_ENTRY(PFX##_m15, F32S, DENSE, 15)  \
    ARLAB_ENTRY(PFX##_m31, F32S, DENSE, 31)
ARLAB_MODES(arlab_bf16_t, false, false)
ARLAB_MODES(arlab_bf16_d, false, true)
ARLAB_MODES(arlab_f32s_t, true, false)
ARLAB_MODES(arlab_f32s_d, true, true)
ARLAB_ENTRY(arlab_bf16_d_m79, false, true, 79)
ARLAB_ENTRY(arlab_f32s_d_m79, true, true, 79)

// Gate kernel: one lane spins `ticks` of the realtime counter.
extern "C" __global__ void arlab_spin(unsigned long long ticks) {
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) __builtin_amdgcn_s_sleep(8);
}
"#;

/// (mode, label) in the order they are run and printed.
pub const MODES: [(u32, &str); 10] = [
    (0, "m0_empty"),
    (1, "m1_push"),
    (3, "m3_push_poll"),
    (7, "m7_push_poll_apply"),
    (15, "m15_full"),
    (4, "m4_apply_only"),
    (8, "m8_ticket_only"),
    (11, "m11_push_poll_ticket"),
    (31, "m31_full_stamped"),
    (79, "m79_roles_full"),
];

const SLOTS: usize = 256;
const SPIN_TICKS: u64 = 300_000; // 3 ms at the 100 MHz realtime counter

#[derive(Clone, Debug, Default)]
pub struct BdRow {
    pub kind: &'static str,
    pub dense: bool,
    pub rows: usize,
    pub geometry: (u32, u32),
    /// (label, gated pipe us, iso median us, iso min us)
    pub series: Vec<(String, f64, f64, f64)>,
    /// m31 stamp means in us: [push issued, poll done, apply issued, ticket done] since block
    /// entry (mean over blocks and launches), body span (first entry .. last ticket of one
    /// launch, median), gap (last ticket of launch g .. first entry of g+1, median), and the
    /// counter cross-check (stamp span / event span of the same series).
    pub stamps: Option<[f64; 7]>,
}

pub struct LabBufs {
    pub mbox: DeviceBuffer,
    pub bank_stride: usize,
    gen: DeviceBuffer,
    wb: DeviceBuffer,
    stamps: DeviceBuffer,
}

/// Lab buffers of one rank. Call BEFORE peer access is enabled (like the product mailbox).
pub fn alloc(gpu: &mut Gpu, max_rows: usize) -> Result<LabBufs, String> {
    let e = |e: hip_bridge::HipError| e.to_string();
    let bank_stride = (max_rows * AR_HC_HIDDEN / 4 * 16).div_ceil(256) * 256;
    let mbox = gpu
        .hip
        .malloc_peer_finegrained(2 * bank_stride)
        .map_err(e)?;
    gpu.hip.memset(&mbox, 0xFF, 2 * bank_stride).map_err(e)?;
    let gen = gpu.hip.malloc(256).map_err(e)?;
    let wb = gpu.hip.malloc(16).map_err(e)?;
    let stamps = gpu.hip.malloc(SLOTS * 64 * 8 * 8).map_err(e)?;
    Ok(LabBufs {
        mbox,
        bank_stride,
        gen,
        wb,
        stamps,
    })
}

fn reset(gpu: &mut Gpu, b: &LabBufs) -> Result<(), String> {
    let e = |e: hip_bridge::HipError| e.to_string();
    gpu.hip
        .memset(&b.mbox, 0xFF, 2 * b.bank_stride)
        .map_err(e)?;
    gpu.hip.memset(&b.gen, 0, 256).map_err(e)?;
    let mut w = [0u8; 16];
    w[8..].copy_from_slice(&50_000_000u64.to_le_bytes()); // 500 ms
    gpu.hip.memcpy_htod(&b.wb, &w).map_err(e)?;
    gpu.hip
        .memset(&b.stamps, 0, SLOTS * 64 * 8 * 8)
        .map_err(e)?;
    gpu.hip.device_synchronize().map_err(e)
}

fn check_wb(gpu: &mut Gpu, b: &LabBufs, what: &str) -> Result<(), String> {
    let mut w = [0u8; 16];
    gpu.hip
        .memcpy_dtoh(&mut w, &b.wb)
        .map_err(|e| e.to_string())?;
    let code = u32::from_le_bytes(w[0..4].try_into().unwrap());
    if code != 0 {
        return Err(format!("{what}: lab wait timed out (site {code})"));
    }
    Ok(())
}

fn sym(kind: StreamKind, dense: bool, mode: u32) -> String {
    let k = match kind {
        StreamKind::F32 => "f32s",
        _ => "bf16",
    };
    format!("arlab_{k}_{}_m{mode}", if dense { "d" } else { "t" })
}

/// Run every mode for every (kind, wire, rows) case on one rank; both ranks call this with
/// the same arguments (identical launch sequences).
#[allow(clippy::too_many_arguments)]
pub fn run(
    rank: usize,
    gpu: &mut Gpu,
    b: &LabBufs,
    peer_mbox: usize,
    x: &Arc<ArHcExchange>,
    barrier: &Arc<Barrier>,
    rows_list: &[usize],
    kinds: &[StreamKind],
    iters: usize,
    iso: usize,
) -> Result<Vec<BdRow>, String> {
    let e = |e: hip_bridge::HipError| e.to_string();
    let src = format!(
        "{}{}",
        rdna_compute::kernels::AR_HC_WRITE_GFX1201_SRC,
        LAB_SRC
    );
    for kind in kinds {
        for dense in [false, true] {
            for (m, _) in MODES.iter().copied().filter(|&(m, _)| dense || m != 79) {
                gpu.ensure_kernel_public("arhc_lab", &src, &sym(*kind, dense, m))
                    .map_err(e)?;
            }
        }
    }
    gpu.ensure_kernel_public("arhc_lab", &src, "arlab_spin")
        .map_err(e)?;
    let start = gpu.hip.event_create().map_err(e)?;
    let stop = gpu.hip.event_create().map_err(e)?;
    let spin = |gpu: &Gpu| -> Result<(), String> {
        let mut k = KernargBlob::new();
        k.push_u64(SPIN_TICKS);
        gpu.launch_kernel_blob("arlab_spin", [1, 1, 1], [32, 1, 1], 0, k.as_mut_slice())
            .map_err(|e| e.to_string())
    };
    let drain = |gpu: &mut Gpu| -> Result<(), String> {
        gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
        gpu.take_device_wait_error().map_err(|e| e.to_string())
    };
    let ev_us = |gpu: &Gpu| -> Result<f64, String> {
        Ok(f64::from(
            gpu.hip
                .event_elapsed_ms(&start, &stop)
                .map_err(|e| e.to_string())?,
        ) * 1000.0)
    };
    let mut out = Vec::new();
    for &kind in kinds {
        for &rows in rows_list {
            let n = rows * AR_HC_HIDDEN;
            let wide = rows * 4 * AR_HC_HIDDEN;
            let partial = gpu.hip.malloc(n * 4).map_err(e)?;
            let streams = gpu.hip.malloc(wide * kind.element_bytes()).map_err(e)?;
            let gates = gpu.hip.malloc(rows * 16).map_err(e)?;
            let hp: Vec<u8> = (0..n)
                .flat_map(|i| (((i % 97) as f32 - 48.0) / 64.0).to_le_bytes())
                .collect();
            gpu.hip.memcpy_htod(&partial, &hp).map_err(e)?;
            gpu.hip
                .memset(&streams, 0, wide * kind.element_bytes())
                .map_err(e)?;
            gpu.hip.memset(&gates, 0, rows * 16).map_err(e)?;
            let geom = ar_hc_geometry(rows).ok_or("no geometry")?;
            // product reference: ungated (the original bench method) and gated
            let prod = |gpu: &mut Gpu| -> Result<(), String> {
                // SAFETY: live buffers of the extents `kind` / `rows` imply on this device.
                unsafe {
                    x.write(
                        rank,
                        gpu,
                        partial.as_ptr(),
                        streams.as_ptr(),
                        kind,
                        gates.as_ptr(),
                        rows,
                        std::ptr::null_mut::<c_void>(),
                    )
                }
                .map_err(|e| e.to_string())
            };
            let mut prod_series = Vec::new();
            for gated in [false, true] {
                for _ in 0..30 {
                    prod(gpu)?;
                }
                drain(gpu)?;
                barrier.wait();
                if gated {
                    spin(gpu)?;
                }
                gpu.hip.event_record(&start, None).map_err(e)?;
                for _ in 0..iters {
                    prod(gpu)?;
                }
                gpu.hip.event_record(&stop, None).map_err(e)?;
                drain(gpu)?;
                prod_series.push(ev_us(gpu)? / iters as f64);
            }
            let mut isos = Vec::with_capacity(iso);
            for _ in 0..iso {
                barrier.wait();
                gpu.hip.event_record(&start, None).map_err(e)?;
                prod(gpu)?;
                gpu.hip.event_record(&stop, None).map_err(e)?;
                drain(gpu)?;
                isos.push(ev_us(gpu)?);
            }
            isos.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut row_dense = [BdRow::default(), BdRow::default()];
            for dense in [false, true] {
                let mut r = BdRow {
                    kind: kind.name(),
                    dense,
                    rows,
                    geometry: geom,
                    ..Default::default()
                };
                r.series.push((
                    "prod_pipe_ungated".into(),
                    prod_series[0],
                    isos[isos.len() / 2],
                    isos[0],
                ));
                r.series
                    .push(("prod_pipe_gated".into(), prod_series[1], 0.0, 0.0));
                for (mode, label) in MODES.iter().copied().filter(|&(m, _)| dense || m != 79) {
                    let name = sym(kind, dense, mode);
                    drain(gpu)?;
                    barrier.wait();
                    reset(gpu, b)?;
                    barrier.wait();
                    let mut tag = 1u32 << 20;
                    let launch = |gpu: &Gpu, tag: u32| -> Result<(), String> {
                        let mut k = KernargBlob::new();
                        k.push_ptr(partial.as_ptr());
                        k.push_ptr(streams.as_ptr());
                        k.push_ptr(gates.as_ptr());
                        k.push_ptr(peer_mbox as *const c_void);
                        k.push_ptr(b.mbox.as_ptr());
                        k.push_u32(b.bank_stride as u32);
                        k.push_ptr(b.gen.as_ptr());
                        k.push_ptr(b.wb.as_ptr());
                        k.push_i32(i32::from(rank == 0));
                        k.push_i32(n as i32);
                        k.push_u32(tag);
                        k.push_ptr(b.stamps.as_ptr());
                        gpu.launch_kernel_blob(
                            &name,
                            [geom.0, 1, 1],
                            [geom.1, 1, 1],
                            0,
                            k.as_mut_slice(),
                        )
                        .map_err(|e| e.to_string())
                    };
                    for _ in 0..30 {
                        tag += 1;
                        launch(gpu, tag)?;
                    }
                    drain(gpu)?;
                    check_wb(gpu, b, &name)?;
                    barrier.wait();
                    if mode == 31 {
                        gpu.hip
                            .memset(&b.stamps, 0, SLOTS * 64 * 8 * 8)
                            .map_err(e)?;
                        gpu.hip.device_synchronize().map_err(e)?;
                        barrier.wait();
                    }
                    spin(gpu)?;
                    gpu.hip.event_record(&start, None).map_err(e)?;
                    let n_pipe = if mode == 31 { iters.min(SLOTS) } else { iters };
                    for _ in 0..n_pipe {
                        tag += 1;
                        launch(gpu, tag)?;
                    }
                    gpu.hip.event_record(&stop, None).map_err(e)?;
                    drain(gpu)?;
                    check_wb(gpu, b, &name)?;
                    let span_us = ev_us(gpu)?;
                    let pipe = span_us / n_pipe as f64;
                    if mode == 31 {
                        r.stamps = Some(stamp_summary(gpu, b, geom.0 as usize, span_us)?);
                    }
                    let mut isos = Vec::with_capacity(iso);
                    for _ in 0..iso {
                        barrier.wait();
                        tag += 1;
                        gpu.hip.event_record(&start, None).map_err(e)?;
                        launch(gpu, tag)?;
                        gpu.hip.event_record(&stop, None).map_err(e)?;
                        drain(gpu)?;
                        isos.push(ev_us(gpu)?);
                    }
                    check_wb(gpu, b, &name)?;
                    isos.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    r.series.push((
                        label.to_string(),
                        pipe,
                        isos.get(isos.len() / 2).copied().unwrap_or(0.0),
                        isos.first().copied().unwrap_or(0.0),
                    ));
                }
                row_dense[usize::from(dense)] = r;
            }
            barrier.wait();
            for buf in [partial, streams, gates] {
                gpu.hip.free(buf).map_err(e)?;
            }
            out.extend(row_dense);
        }
    }
    Ok(out)
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn stamp_summary(
    gpu: &mut Gpu,
    b: &LabBufs,
    grid: usize,
    span_us: f64,
) -> Result<[f64; 7], String> {
    // slot s, block blk holds 8 words at (s * grid + blk) * 8
    let mut raw = vec![0u8; SLOTS * 64 * 8 * 8];
    gpu.hip
        .memcpy_dtoh(&mut raw, &b.stamps)
        .map_err(|e| e.to_string())?;
    let w = |i: usize| u64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().unwrap());
    let tick_us = 0.01; // 100 MHz realtime counter
    let mut launches = Vec::new();
    let mut phase = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for s in 0..SLOTS {
        let mut lo = u64::MAX;
        let mut hi = 0u64;
        let mut complete = true;
        for blk in 0..grid {
            let base = (s * grid + blk) * 8;
            let t: Vec<u64> = (0..5).map(|q| w(base + q)).collect();
            if t.iter().any(|&v| v == 0) {
                complete = false;
                break;
            }
            lo = lo.min(t[0]);
            hi = hi.max(t[4]);
            for q in 0..4 {
                phase[q].push((t[q + 1] as f64 - t[0] as f64) * tick_us);
            }
        }
        if complete {
            launches.push((lo, hi));
        }
    }
    launches.sort();
    let mut spans: Vec<f64> = launches
        .iter()
        .map(|(a, b)| (*b as f64 - *a as f64) * tick_us)
        .collect();
    let mut gaps: Vec<f64> = launches
        .windows(2)
        .map(|p| (p[1].0 as f64 - p[0].1 as f64) * tick_us)
        .collect();
    let stamp_span = match (launches.first(), launches.last()) {
        (Some(f), Some(l)) if launches.len() > 1 => (l.1 as f64 - f.0 as f64) * tick_us,
        _ => 0.0,
    };
    let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len().max(1) as f64;
    Ok([
        mean(&phase[0]),
        mean(&phase[1]),
        mean(&phase[2]),
        mean(&phase[3]),
        median(&mut spans),
        median(&mut gaps),
        if span_us > 0.0 {
            stamp_span / span_us
        } else {
            0.0
        },
    ])
}
