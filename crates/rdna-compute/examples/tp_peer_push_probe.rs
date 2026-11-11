// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab microbench: in-kernel peer-VRAM push + system-scope flag collective
//! against the incumbent TP2 `tp_graph_signal_reduce_gfx1100` collective, on
//! two exact gfx1201 cards.
//!
//! Model under test: the partial is pushed by the producing rank straight
//! into the PEER's fine-grained VRAM mailbox (`hipDeviceMallocFinegrained`,
//! allocated before peer access is enabled); the consumer polls only its own
//! VRAM, folds `x + (p0 + p1)` in fixed rank order and writes the residual.
//! Only system-scope loads/stores/fences touch the peer: no cross-device RMW.
//! Two banks alternate with a device-resident generation; a symmetric exchange
//! means "peer's ready(g-1) observed" proves the peer finished reading
//! bank(g-2), so no consumed flag is needed.
//!
//! Arms (all in one graph per rank: 128 x [producer -> collective]):
//!   base    producers only (no collective): the boundary floor
//!   inc     incumbent tp_graph_signal_reduce_gfx1100 (host-mapped coherent
//!           shadow payload + signal memory), via the library launcher
//!   push[:threads[:mode]]  nontemporal payload push, fence, release flag
//!           (mode bits 8/16/32/64 trim fences and polls; bits 1/2/4 are
//!           timing-only diagnostics that skip the wait, fence or payload)
//!   push3   the same protocol with all independent loads in flight first
//!   ll      tagged 8-byte words with system-scope atomic stores and polls
//!           (correct, slow: 33 us per collective, 5,120 tiny PCIe writes)
//!   ll128   tagged 16-byte granules {d0, g, d1, g}, write-through b128
//!           stores and b128 polls in inline asm, no fence, no flag
//!   prod    the shipped kernel `tp2_peer_push_reduce_gfx1201`
//! Findings that shaped it: plain/nontemporal stores to the peer stay in the
//! writer's L2 until kernel end (the peer's poll never sees them), and
//! nontemporal polling loads read a stale L2 line; only system-scope
//! write-through stores and system-scope loads publish.
//!
//! The producer reads the previous residual (`p = h(salt, rank, k, i) -
//! x * 2^-9`, explicit round-to-nearest ops) so the 128 collectives are a
//! true dependent chain and any stale mailbox read perturbs every later
//! value; the host replays the exact arithmetic and compares bit for bit.
//!
//! Env: PROBE_ARMS (default base,inc,push3:256:0,ll128; comma-separated), PROBE_CORRECT_REPLAYS
//! (10000), PROBE_PIPELINE (replays queued per verified batch, 1),
//! PROBE_TIMING_N (200), PROBE_TIMING_TRIALS (15), PROBE_SEED,
//! PROBE_DELAYS (1: random per-replay rank delay + launch order in the
//! correctness run), PROBE_TIMEOUT_TEST (1).
//!
//! Needs two visible gfx1201 devices. Run through the fabric wrapper.

use hip_bridge::{
    DeviceBuffer, Event, Function, Graph, GraphExec, HipRuntime, KernargBlob, Module,
};
use rdna_compute::device_wait::{timeout_ticks, DeviceWaitBlock};
use rdna_compute::{Gpu, KernelCompiler};
use std::ffi::c_void;
use std::time::Instant;

const COUNT: usize = 5_120;
const PAYLOAD: usize = COUNT * 4; // 20,480 B f32 partial
const BARRIERS: usize = 128;
const CAPTURE_MODE_RELAXED: u32 = 2;

// push mailbox layout (bytes), one fine-grained allocation per rank
const CANARY: usize = 256;
const P_BANK0: usize = CANARY;
const P_BANK_STRIDE: usize = PAYLOAD + CANARY;
const P_FLAGS: usize = P_BANK0 + 2 * P_BANK_STRIDE; // bank b flag at +256*b
const P_TOTAL: usize = P_FLAGS + 512 + CANARY;
// ll mailbox: one u64 (data | gen<<32) per float
const LL_PAYLOAD: usize = COUNT * 8;
const LL_BANK0: usize = CANARY;
const LL_BANK_STRIDE: usize = LL_PAYLOAD + CANARY;
const LL_TOTAL: usize = LL_BANK0 + 2 * LL_BANK_STRIDE;

const SOURCE: &str = concat!(
    include_str!("../../../kernels/src/device_wait_deadline.h"),
    r#"
#define PROBE_SITE 4u

__device__ __forceinline__ unsigned probe_mix(unsigned h) {
    h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
    return h;
}
__device__ __forceinline__ float probe_hashf(unsigned salt, unsigned rank, unsigned k, unsigned i) {
    unsigned h = probe_mix(salt * 0x9E3779B1u ^ probe_mix(rank * 0x632BE5ABu + k * 0x1B873593u + i * 0x27D4EB2Fu + 12345u));
    return __fadd_rn(__fmul_rn(__uint2float_rn(h >> 8), 0x1p-23f), -1.0f);
}

extern "C" __global__ void probe_producer(float *__restrict__ part, const float *__restrict__ x,
                                          const unsigned *__restrict__ salt, int rank, int k, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    part[i] = __fsub_rn(probe_hashf(salt[0], (unsigned)rank, (unsigned)k, (unsigned)i),
                        __fmul_rn(x[i], 0x1p-9f));
}

extern "C" __global__ void probe_spin(unsigned ticks) {
    if (threadIdx.x != 0) return;
    unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < (unsigned long long)ticks) { __builtin_amdgcn_s_sleep(1); }
}

__device__ __forceinline__ float4 probe_ld_nt4(const float4 *ptr) {
    typedef __attribute__((address_space(1))) const unsigned long long *g64;
    const unsigned long long *p = reinterpret_cast<const unsigned long long *>(ptr);
    union { unsigned long long u[2]; float4 f; } v;
    v.u[0] = __builtin_nontemporal_load((g64)p);
    v.u[1] = __builtin_nontemporal_load((g64)(p + 1));
    return v.f;
}
__device__ __forceinline__ void probe_st_nt4(float4 *ptr, float4 value) {
    typedef __attribute__((address_space(1))) unsigned long long *g64;
    unsigned long long *p = reinterpret_cast<unsigned long long *>(ptr);
    union { unsigned long long u[2]; float4 f; } v;
    v.f = value;
    __builtin_nontemporal_store(v.u[0], (g64)p);
    __builtin_nontemporal_store(v.u[1], (g64)(p + 1));
}

// push + system-scope release flag. One CTA of `blockDim.x` lanes. `mode`
// bits are lab-only diagnostics: 1 skip the peer wait, 2 skip the per-lane
// system fence, 4 skip the payload push (flag only). Mode 0 is the protocol.
extern "C" __global__ __launch_bounds__(1024) void probe_push_reduce(
    float *__restrict__ x, const float *__restrict__ part,
    char *peer_mbox, const char *local_mbox,
    unsigned bank0_off, unsigned bank_stride, unsigned flag_off,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int count, int mode) {
    if (blockIdx.x != 0) return;
    __shared__ unsigned sh_gen;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    if (tid == 0) {
        unsigned g = __hip_atomic_load(gen_ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
        __hip_atomic_store(gen_ptr, g, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        sh_gen = g;
    }
    __syncthreads();
    const unsigned g = sh_gen;
    const unsigned bank = g & 1u;
    const int vec = count >> 2;
    const float4 *__restrict__ localv = reinterpret_cast<const float4 *>(part);
    float4 *__restrict__ peerv = reinterpret_cast<float4 *>(peer_mbox + bank0_off + bank * bank_stride);
    if (!(mode & 4)) {
        for (int i = tid; i < vec; i += nt) probe_st_nt4(peerv + i, localv[i]);
        for (int i = (vec << 2) + tid; i < count; i += nt)
            __builtin_nontemporal_store(part[i], reinterpret_cast<float *>(peerv) + i);
    }
    if (!(mode & 2)) {
        if (mode & 8) __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
        else __threadfence_system();
    }
    __syncthreads();
    if (tid == 0) {
        unsigned *pflag = reinterpret_cast<unsigned *>(peer_mbox + flag_off + bank * 256u);
        if (mode & 16) __hip_atomic_store(pflag, g, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        else __hip_atomic_store(pflag, g, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    const unsigned *lflag = reinterpret_cast<const unsigned *>(local_mbox + flag_off + bank * 256u);
    if (!(mode & 1)) {
        auto ready = [&] {
            const unsigned f = (mode & 32)
                ? __hip_atomic_load(lflag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
                : __hip_atomic_load(lflag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
            return static_cast<int>(f - g) >= 0;
        };
        if (mode & 64) {
            __shared__ int sh_fail;
            if (tid == 0) sh_fail = hf_bounded_wait(ready, wb, PROBE_SITE, g) ? 0 : 1;
            __syncthreads();
            if (sh_fail) return;
        } else if (!hf_bounded_wait(ready, wb, PROBE_SITE, g)) {
            return;
        }
        if (mode & 32) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    }
    const float4 *__restrict__ inv = reinterpret_cast<const float4 *>(local_mbox + bank0_off + bank * bank_stride);
    float4 *__restrict__ xv = reinterpret_cast<float4 *>(x);
    for (int i = tid; i < vec; i += nt) {
        float4 xi = xv[i];
        const float4 loc = localv[i];
        const float4 in = probe_ld_nt4(inv + i);
        const float4 fi = local_is_first ? loc : in;
        const float4 si = local_is_first ? in : loc;
        xi.x = xi.x + (fi.x + si.x); xi.y = xi.y + (fi.y + si.y);
        xi.z = xi.z + (fi.z + si.z); xi.w = xi.w + (fi.w + si.w);
        xv[i] = xi;
    }
    const float *__restrict__ inf = reinterpret_cast<const float *>(inv);
    for (int i = (vec << 2) + tid; i < count; i += nt) {
        const float loc = part[i];
        const float in = __builtin_nontemporal_load(inf + i);
        const float fi = local_is_first ? loc : in;
        const float si = local_is_first ? in : loc;
        x[i] = x[i] + (fi + si);
    }
}

// Tagged-word push: each 8-byte word is {float bits, generation}; the
// consumer polls the data itself, so no fence and no separate flag.
#define LL_KMAX 32
extern "C" __global__ __launch_bounds__(256) void probe_push_reduce_ll(
    float *__restrict__ x, const float *__restrict__ part,
    char *peer_mbox, const char *local_mbox,
    unsigned bank0_off, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int count) {
    if (blockIdx.x != 0) return;
    __shared__ unsigned sh_gen;
    const int tid = threadIdx.x;
    if (tid == 0) {
        unsigned g = __hip_atomic_load(gen_ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
        __hip_atomic_store(gen_ptr, g, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        sh_gen = g;
    }
    __syncthreads();
    const unsigned g = sh_gen;
    const unsigned bank = g & 1u;
    unsigned long long *pb = reinterpret_cast<unsigned long long *>(peer_mbox + bank0_off + bank * bank_stride);
    const unsigned long long *lb = reinterpret_cast<const unsigned long long *>(local_mbox + bank0_off + bank * bank_stride);
    #pragma unroll
    for (int k = 0; k < LL_KMAX; ++k) {
        const int i = tid + k * 256;
        if (i < count) {
            const unsigned long long v = ((unsigned long long)g << 32) | __float_as_uint(part[i]);
            __hip_atomic_store(pb + i, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
    unsigned long long v[LL_KMAX];
    if (!hf_bounded_wait(
            [&] {
                bool ok = true;
                #pragma unroll
                for (int k = 0; k < LL_KMAX; ++k) {
                    const int i = tid + k * 256;
                    if (i < count) {
                        v[k] = __hip_atomic_load(lb + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                        ok &= (static_cast<unsigned>(v[k] >> 32) == g);
                    }
                }
                return ok;
            },
            wb, PROBE_SITE, g)) {
        return;
    }
    #pragma unroll
    for (int k = 0; k < LL_KMAX; ++k) {
        const int i = tid + k * 256;
        if (i < count) {
            const float in = __uint_as_float(static_cast<unsigned>(v[k]));
            const float loc = part[i];
            const float fi = local_is_first ? loc : in;
            const float si = local_is_first ? in : loc;
            x[i] = x[i] + (fi + si);
        }
    }
}

// push3: same protocol as `push` with release-only fence, relaxed flag store
// and relaxed system polling + one acquire fence, restructured so the
// independent loads (partial, residual, generation) are all in flight before
// the first dependent step, and the generation is stored once after the fold
// instead of through a barrier at entry. count must be a multiple of 4 and
// <= 8 * 4 * blockDim.x. `mode` bits are lab diagnostics: 1 skip the wait,
// 2 skip the fence, 4 skip the payload.
#define P3_KV 8
extern "C" __global__ __launch_bounds__(1024) void probe_push3_reduce(
    float *__restrict__ x, const float *__restrict__ part,
    char *peer_mbox, const char *local_mbox,
    unsigned bank0_off, unsigned bank_stride, unsigned flag_off,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int count, int mode) {
    if (blockIdx.x != 0) return;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int vec = count >> 2;
    const float4 *__restrict__ localv = reinterpret_cast<const float4 *>(part);
    float4 *__restrict__ xv = reinterpret_cast<float4 *>(x);
    float4 pv[P3_KV], xr[P3_KV];
    #pragma unroll
    for (int k = 0; k < P3_KV; ++k) {
        const int i = tid + k * nt;
        if (i < vec) { pv[k] = localv[i]; xr[k] = xv[i]; }
    }
    const unsigned g = __hip_atomic_load(gen_ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    const unsigned bank = g & 1u;
    float4 *__restrict__ peerv = reinterpret_cast<float4 *>(peer_mbox + bank0_off + bank * bank_stride);
    if (!(mode & 4)) {
        #pragma unroll
        for (int k = 0; k < P3_KV; ++k) {
            const int i = tid + k * nt;
            if (i < vec) probe_st_nt4(peerv + i, pv[k]);
        }
    }
    if (!(mode & 2)) __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    __syncthreads();
    if (tid == 0) {
        __hip_atomic_store(reinterpret_cast<unsigned *>(peer_mbox + flag_off + bank * 256u), g,
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    const unsigned *lflag = reinterpret_cast<const unsigned *>(local_mbox + flag_off + bank * 256u);
    if (!(mode & 1)) {
        if (!hf_bounded_wait(
                [&] { return static_cast<int>(__hip_atomic_load(lflag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) - g) >= 0; },
                wb, PROBE_SITE, g)) {
            return;
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    }
    const float4 *__restrict__ inv = reinterpret_cast<const float4 *>(local_mbox + bank0_off + bank * bank_stride);
    float4 in[P3_KV];
    #pragma unroll
    for (int k = 0; k < P3_KV; ++k) {
        const int i = tid + k * nt;
        if (i < vec) in[k] = probe_ld_nt4(inv + i);
    }
    #pragma unroll
    for (int k = 0; k < P3_KV; ++k) {
        const int i = tid + k * nt;
        if (i < vec) {
            float4 xi = xr[k];
            const float4 loc = pv[k];
            const float4 fi = local_is_first ? loc : in[k];
            const float4 si = local_is_first ? in[k] : loc;
            xi.x = xi.x + (fi.x + si.x); xi.y = xi.y + (fi.y + si.y);
            xi.z = xi.z + (fi.z + si.z); xi.w = xi.w + (fi.w + si.w);
            xv[i] = xi;
        }
    }
    // Every lane has loaded the generation (its load precedes its own
    // fence/barrier), so this single write cannot race a reader.
    if (tid == 0) __hip_atomic_store(gen_ptr, g, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// ll128: tagged 16-byte granules {d0, g, d1, g}, written with system-scope
// write-through b128 stores and polled with system-scope b128 loads (inline
// asm: the compiler has no 128-bit system-scope atomic). Each 8-byte half
// validates itself, so a half-torn 16-byte write is still detected; no fence
// and no flag. Fixed 512 lanes x 5 granules = 2560 granules = 5120 floats.
typedef unsigned int probe_u4 __attribute__((ext_vector_type(4)));
#define LL128_K 5
extern "C" __global__ __launch_bounds__(512) void probe_ll128_reduce(
    float *__restrict__ x, const float *__restrict__ part,
    char *peer_mbox, const char *local_mbox,
    unsigned bank0_off, unsigned bank_stride,
    unsigned *__restrict__ gen_ptr, hf_device_wait_block *__restrict__ wb,
    int local_is_first, int count) {
    if (blockIdx.x != 0) return;
    const int tid = threadIdx.x;
    float2 pv[LL128_K], xr[LL128_K];
    const float2 *__restrict__ part2 = reinterpret_cast<const float2 *>(part);
    float2 *__restrict__ x2 = reinterpret_cast<float2 *>(x);
    #pragma unroll
    for (int k = 0; k < LL128_K; ++k) { const int j = tid + k * 512; pv[k] = part2[j]; xr[k] = x2[j]; }
    const unsigned g = __hip_atomic_load(gen_ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    const unsigned bank = g & 1u;
    probe_u4 *pb = reinterpret_cast<probe_u4 *>(peer_mbox + bank0_off + bank * bank_stride);
    const probe_u4 *lb = reinterpret_cast<const probe_u4 *>(local_mbox + bank0_off + bank * bank_stride);
    probe_u4 w[LL128_K];
    #pragma unroll
    for (int k = 0; k < LL128_K; ++k) {
        w[k] = probe_u4{__float_as_uint(pv[k].x), g, __float_as_uint(pv[k].y), g};
    }
    asm volatile(
        "global_store_b128 %0, %5, off scope:SCOPE_SYS\n\t"
        "global_store_b128 %1, %6, off scope:SCOPE_SYS\n\t"
        "global_store_b128 %2, %7, off scope:SCOPE_SYS\n\t"
        "global_store_b128 %3, %8, off scope:SCOPE_SYS\n\t"
        "global_store_b128 %4, %9, off scope:SCOPE_SYS"
        :: "v"(pb + tid), "v"(pb + tid + 512), "v"(pb + tid + 1024), "v"(pb + tid + 1536), "v"(pb + tid + 2048),
           "v"(w[0]), "v"(w[1]), "v"(w[2]), "v"(w[3]), "v"(w[4]) : "memory");
    probe_u4 r[LL128_K];
    if (!hf_bounded_wait(
            [&] {
                asm volatile(
                    "global_load_b128 %0, %5, off scope:SCOPE_SYS\n\t"
                    "global_load_b128 %1, %6, off scope:SCOPE_SYS\n\t"
                    "global_load_b128 %2, %7, off scope:SCOPE_SYS\n\t"
                    "global_load_b128 %3, %8, off scope:SCOPE_SYS\n\t"
                    "global_load_b128 %4, %9, off scope:SCOPE_SYS\n\t"
                    "s_wait_loadcnt 0"
                    : "=&v"(r[0]), "=&v"(r[1]), "=&v"(r[2]), "=&v"(r[3]), "=&v"(r[4])
                    : "v"(lb + tid), "v"(lb + tid + 512), "v"(lb + tid + 1024), "v"(lb + tid + 1536), "v"(lb + tid + 2048)
                    : "memory");
                bool ok = true;
                #pragma unroll
                for (int k = 0; k < LL128_K; ++k) ok &= (r[k].y == g) & (r[k].w == g);
                return ok;
            },
            wb, PROBE_SITE, g)) {
        return;
    }
    #pragma unroll
    for (int k = 0; k < LL128_K; ++k) {
        const int j = tid + k * 512;
        float2 xi = xr[k];
        const float2 loc = pv[k];
        const float2 in = make_float2(__uint_as_float(r[k].x), __uint_as_float(r[k].z));
        const float2 fi = local_is_first ? loc : in;
        const float2 si = local_is_first ? in : loc;
        xi.x = xi.x + (fi.x + si.x);
        xi.y = xi.y + (fi.y + si.y);
        x2[j] = xi;
    }
    __syncthreads();
    if (tid == 0) __hip_atomic_store(gen_ptr, g, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
"#
);

// ---- host mirror of the device arithmetic -------------------------------

fn mix(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

fn hashf(salt: u32, rank: u32, k: u32, i: u32) -> f32 {
    let inner = mix(rank
        .wrapping_mul(0x632B_E5AB)
        .wrapping_add(k.wrapping_mul(0x1B87_3593))
        .wrapping_add(i.wrapping_mul(0x27D4_EB2F))
        .wrapping_add(12345));
    let h = mix(salt.wrapping_mul(0x9E37_79B1) ^ inner);
    ((h >> 8) as f32) * f32::from_bits(0x3400_0000) - 1.0 // 2^-23
}

fn x0(i: usize) -> f32 {
    (mix(i as u32 ^ 0xABCD_1234) >> 8) as f32 * f32::from_bits(0x3400_0000) * 8.0 - 8.0
}

/// One graph replay (128 barriers) on the shared residual; both ranks hold
/// the same residual, so one host chain covers both.
fn emulate_replay(x: &mut [f32], salt: u32) {
    let c = f32::from_bits(0x3b00_0000); // 2^-9
    let threads = 8;
    let chunk = x.len().div_ceil(threads);
    std::thread::scope(|s| {
        for (ci, part) in x.chunks_mut(chunk).enumerate() {
            s.spawn(move || {
                for (j, xi) in part.iter_mut().enumerate() {
                    let i = (ci * chunk + j) as u32;
                    let mut v = *xi;
                    for k in 0..BARRIERS as u32 {
                        let p0 = hashf(salt, 0, k, i) - v * c;
                        let p1 = hashf(salt, 1, k, i) - v * c;
                        v += p0 + p1;
                    }
                    *xi = v;
                }
            });
        }
    });
}

// ---- rig ------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Base,
    Inc,
    /// (threads, diagnostic mode bits; 0 = the protocol)
    Push(u32, i32),
    /// restructured push (threads, diagnostic mode bits)
    Push3(u32, i32),
    Ll,
    /// tagged b128 granules, 512 lanes
    Ll128,
    /// the shipped kernel `tp2_peer_push_reduce_gfx1201` (kernels.rs source)
    Prod,
}

impl Arm {
    fn name(self) -> String {
        match self {
            Arm::Base => "base".into(),
            Arm::Inc => "inc".into(),
            Arm::Push(t, m) => format!("push:{t}:{m}"),
            Arm::Push3(t, m) => format!("push3:{t}:{m}"),
            Arm::Ll => "ll".into(),
            Arm::Ll128 => "ll128".into(),
            Arm::Prod => "prod".into(),
        }
    }
}

struct Rank {
    gpu: Gpu,
    _module: Module,
    _module_prod: Module,
    f_shipped: Function,
    f_prod: Function,
    f_spin: Function,
    f_push: Function,
    f_push3: Function,
    f_ll: Function,
    f_ll128: Function,
    x: DeviceBuffer,
    part: DeviceBuffer,
    salt: DeviceBuffer,
    gen_push: DeviceBuffer,
    gen_ll: DeviceBuffer,
    mbox: DeviceBuffer,
    mbox_ll: DeviceBuffer,
    signal: DeviceBuffer,
    shadow_dev: *mut c_void,
    wb_host: *mut DeviceWaitBlock,
    wb_dev: *mut c_void,
    capturing: std::cell::Cell<bool>,
    blobs: std::cell::RefCell<Vec<Vec<u8>>>,
}

struct Captured {
    graph: Graph,
    exec: GraphExec,
    _blobs: Vec<Vec<u8>>,
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 4) }
}

fn sub(buf: &DeviceBuffer, off: usize, len: usize) -> DeviceBuffer {
    // SAFETY: borrowed view into a live allocation, only used for D2H/memset.
    unsafe { DeviceBuffer::from_raw(buf.as_ptr().cast::<u8>().add(off).cast(), len) }
}

impl Rank {
    fn bind(&self) {
        self.gpu.bind_thread().expect("bind");
    }
    fn stream(&self) -> &hip_bridge::Stream {
        self.gpu.active_stream.as_ref().expect("stream")
    }
    fn sync(&self) {
        self.bind();
        self.gpu
            .hip
            .stream_synchronize(self.stream())
            .expect("sync");
    }
    fn memset(&self, buf: &DeviceBuffer, v: i32, n: usize) {
        self.bind();
        self.gpu.hip.memset(buf, v, n).expect("memset");
    }
    fn reset_state(&self, x_init: &[f32]) {
        self.bind();
        self.gpu.hip.device_synchronize().expect("dev sync");
        self.gpu
            .hip
            .memcpy_htod(&self.x, bytes_of(x_init))
            .expect("x0");
        self.memset(&self.gen_push, 0, 4);
        self.memset(&self.gen_ll, 0, 4);
        self.memset(&self.signal, 0, 8);
        self.memset(&self.salt, 0, 4);
        // push mailbox: canaries 0xCA, payload poison 0xFF (NaN), flags 0
        self.memset(&self.mbox, 0xCA, P_TOTAL);
        for b in 0..2 {
            self.memset(
                &sub(&self.mbox, P_BANK0 + b * P_BANK_STRIDE, PAYLOAD),
                0xFF,
                PAYLOAD,
            );
        }
        self.memset(&sub(&self.mbox, P_FLAGS, 512), 0, 512);
        self.memset(&self.mbox_ll, 0xCA, LL_TOTAL);
        for b in 0..2 {
            self.memset(
                &sub(&self.mbox_ll, LL_BANK0 + b * LL_BANK_STRIDE, LL_PAYLOAD),
                0,
                LL_PAYLOAD,
            );
        }
        unsafe { std::ptr::write_volatile(&mut (*self.wb_host).code, 0) };
        self.bind();
        self.gpu.hip.device_synchronize().expect("dev sync");
    }
    fn download_x(&self) -> Vec<f32> {
        self.bind();
        let mut raw = vec![0u8; PAYLOAD];
        self.gpu.hip.memcpy_dtoh(&mut raw, &self.x).expect("dtoh x");
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    fn wait_code(&self) -> u32 {
        unsafe { std::ptr::read_volatile(&(*self.wb_host).code) }
    }
}

/// Launch through a kernarg blob: graph capture on this HIP build refuses
/// `kernelParams` launches at instantiate, so captured launches keep their
/// blob alive in `rank.blobs` (the library launchers do the same through
/// `GraphState::capture_blobs`).
fn launch(rank: &Rank, f: &Function, grid: u32, block: u32, mut blob: KernargBlob) {
    rank.bind();
    blob.pad_to(16);
    let mut v = blob.into_vec();
    let r = if rank.capturing.get() {
        let mut blobs = rank.blobs.borrow_mut();
        blobs.push(v);
        let buf = blobs.last_mut().unwrap();
        // SAFETY: blob bytes match the kernel signature; the heap buffer
        // stays live until the graph is destroyed.
        unsafe {
            rank.gpu.hip.launch_kernel_blob(
                f,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                Some(rank.stream()),
                buf.as_mut_slice(),
            )
        }
    } else {
        // SAFETY: one-shot launch, bytes live across the call.
        unsafe {
            rank.gpu.hip.launch_kernel_blob(
                f,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                Some(rank.stream()),
                v.as_mut_slice(),
            )
        }
    };
    r.expect("launch");
}

fn enqueue_producer(rank: &Rank, rk: i32, k: i32) {
    let mut b = KernargBlob::new();
    b.push_ptr(rank.part.as_ptr());
    b.push_ptr(rank.x.as_ptr());
    b.push_ptr(rank.salt.as_ptr());
    b.push_i32(rk);
    b.push_i32(k);
    b.push_i32(COUNT as i32);
    launch(rank, &rank.f_prod, (COUNT as u32).div_ceil(256), 256, b);
}

fn enqueue_push(ranks: &[Rank; 2], rk: usize, threads: u32, mode: i32, v3: bool) {
    let r = &ranks[rk];
    let peer = &ranks[1 - rk];
    let mut b = KernargBlob::new();
    b.push_ptr(r.x.as_ptr());
    b.push_ptr(r.part.as_ptr());
    b.push_ptr(peer.mbox.as_ptr());
    b.push_ptr(r.mbox.as_ptr());
    b.push_u32(P_BANK0 as u32);
    b.push_u32(P_BANK_STRIDE as u32);
    b.push_u32(P_FLAGS as u32);
    b.push_ptr(r.gen_push.as_ptr());
    b.push_ptr(r.wb_dev);
    b.push_i32(i32::from(rk == 0));
    b.push_i32(COUNT as i32);
    b.push_i32(mode);
    launch(r, if v3 { &r.f_push3 } else { &r.f_push }, 1, threads, b);
}

fn enqueue_prod(ranks: &[Rank; 2], rk: usize) {
    let r = &ranks[rk];
    let peer = &ranks[1 - rk];
    let mut b = KernargBlob::new();
    b.push_ptr(r.x.as_ptr());
    b.push_ptr(r.part.as_ptr());
    // bank 0 at the base, bank 1 one stride later (the runtime's layout)
    b.push_ptr(unsafe { peer.mbox_ll.as_ptr().cast::<u8>().add(LL_BANK0).cast() });
    b.push_ptr(unsafe { r.mbox_ll.as_ptr().cast::<u8>().add(LL_BANK0).cast() });
    b.push_u32(LL_BANK_STRIDE as u32);
    b.push_ptr(r.gen_ll.as_ptr());
    b.push_ptr(r.wb_dev);
    b.push_i32(i32::from(rk == 0));
    b.push_i32(COUNT as i32);
    launch(r, &r.f_shipped, 1, 512, b);
}

fn enqueue_ll(ranks: &[Rank; 2], rk: usize, threads: u32, ll128: bool) {
    let r = &ranks[rk];
    let peer = &ranks[1 - rk];
    let mut b = KernargBlob::new();
    b.push_ptr(r.x.as_ptr());
    b.push_ptr(r.part.as_ptr());
    b.push_ptr(peer.mbox_ll.as_ptr());
    b.push_ptr(r.mbox_ll.as_ptr());
    b.push_u32(LL_BANK0 as u32);
    b.push_u32(LL_BANK_STRIDE as u32);
    b.push_ptr(r.gen_ll.as_ptr());
    b.push_ptr(r.wb_dev);
    b.push_i32(i32::from(rk == 0));
    b.push_i32(COUNT as i32);
    launch(r, if ll128 { &r.f_ll128 } else { &r.f_ll }, 1, threads, b);
}

fn capture(ranks: &mut [Rank; 2], arm: Arm) -> [Captured; 2] {
    for r in ranks.iter_mut() {
        r.bind();
        r.gpu.graphs.capture_blobs.clear();
        r.blobs.borrow_mut().clear();
        r.capturing.set(true);
        r.gpu.graphs.capture_mode = true;
        let s = r.gpu.active_stream.as_ref().expect("stream");
        r.gpu
            .hip
            .stream_begin_capture(s, CAPTURE_MODE_RELAXED)
            .expect("begin capture");
    }
    for k in 0..BARRIERS {
        for rk in 0..2 {
            enqueue_producer(&ranks[rk], rk as i32, k as i32);
        }
        for rk in 0..2 {
            match arm {
                Arm::Base => {}
                Arm::Push(t, m) => enqueue_push(ranks, rk, t, m, false),
                Arm::Push3(t, m) => enqueue_push(ranks, rk, t, m, true),
                Arm::Ll => enqueue_ll(ranks, rk, 256, false),
                Arm::Ll128 => enqueue_ll(ranks, rk, 512, true),
                Arm::Prod => enqueue_prod(ranks, rk),
                Arm::Inc => {
                    let slot = k & 1;
                    // Segment (consumer*2+slot); each rank writes its peer's
                    // and reads its own, via its own context alias.
                    let seg = PAYLOAD.next_multiple_of(256);
                    // Each rank writes the peer's segment and reads its own
                    // through its OWN context's alias of the host allocation.
                    let own = ranks[rk].shadow_dev.cast::<u8>();
                    let peer_seg = unsafe { own.add(((1 - rk) * 2 + slot) * seg) }.cast::<c_void>();
                    let local_seg =
                        unsafe { own.add((rk * 2 + slot) * seg) }.cast::<c_void>() as *const c_void;
                    let local_sig = unsafe { DeviceBuffer::from_raw(ranks[rk].signal.as_ptr(), 8) };
                    let peer_sig =
                        unsafe { DeviceBuffer::from_raw(ranks[1 - rk].signal.as_ptr(), 8) };
                    let x = unsafe { DeviceBuffer::from_raw(ranks[rk].x.as_ptr(), PAYLOAD) };
                    let part = unsafe { DeviceBuffer::from_raw(ranks[rk].part.as_ptr(), PAYLOAD) };
                    ranks[rk]
                        .gpu
                        .tp_graph_signal_reduce_gfx1100(
                            &local_sig,
                            &peer_sig,
                            &x,
                            &part,
                            peer_seg,
                            local_seg,
                            rk == 0,
                            COUNT,
                        )
                        .expect("incumbent reduce");
                }
            }
        }
    }
    let mut out = Vec::new();
    for r in ranks.iter_mut() {
        r.bind();
        let s = r.gpu.active_stream.as_ref().expect("stream");
        let graph = r.gpu.hip.stream_end_capture(s).expect("end capture");
        r.gpu.graphs.capture_mode = false;
        r.capturing.set(false);
        let exec = r.gpu.hip.graph_instantiate(&graph).expect("instantiate");
        out.push(Captured {
            graph,
            exec,
            _blobs: std::mem::take(&mut r.gpu.graphs.capture_blobs)
                .into_iter()
                .chain(r.blobs.take())
                .collect(),
        });
    }
    let b = out.pop().unwrap();
    let a = out.pop().unwrap();
    [a, b]
}

fn launch_graph(ranks: &[Rank; 2], caps: &[Captured; 2], rk: usize) {
    let r = &ranks[rk];
    r.bind();
    r.gpu
        .hip
        .graph_launch(&caps[rk].exec, r.stream())
        .expect("graph launch");
}

fn upload_salt(r: &Rank, salt: &[u8; 4]) {
    r.bind();
    r.gpu
        .hip
        .memcpy_htod_async(&r.salt, salt, r.stream())
        .expect("salt");
}

fn spin(r: &Rank, us: u32) {
    let t = us * 100; // 100 MHz realtime counter
    let mut b = KernargBlob::new();
    b.push_u32(t);
    r.bind();
    launch(r, &r.f_spin, 1, 64, b);
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn correctness(
    ranks: &mut [Rank; 2],
    arm: Arm,
    replays: usize,
    delays: bool,
    seed: u64,
    pipeline: usize,
) -> bool {
    let x_init: Vec<f32> = (0..COUNT).map(x0).collect();
    for r in ranks.iter() {
        r.reset_state(&x_init);
    }
    let caps = capture(ranks, arm);
    let mut host = x_init.clone();
    let mut rng = Rng(seed | 1);
    let mut bad = 0usize;
    let t0 = Instant::now();
    let batch = pipeline.max(1);
    let mut replay = 0usize;
    while replay < replays && bad <= 5 {
        let nb = batch.min(replays - replay);
        let salts: Vec<[u8; 4]> = (0..nb)
            .map(|j| (replay as u32 + j as u32 + 1).to_le_bytes())
            .collect();
        for salt in &salts {
            for r in ranks.iter() {
                upload_salt(r, salt);
            }
            let order = if delays && rng.next() & 1 == 1 {
                [1, 0]
            } else {
                [0, 1]
            };
            if delays {
                let victim = (rng.next() & 1) as usize;
                let us = (rng.next() % 60) as u32;
                spin(&ranks[victim], us);
            }
            for &rk in &order {
                launch_graph(ranks, &caps, rk);
            }
        }
        for r in ranks.iter() {
            r.sync();
        }
        for salt in &salts {
            emulate_replay(&mut host, u32::from_le_bytes(*salt));
        }
        let xs = [ranks[0].download_x(), ranks[1].download_x()];
        for (rk, xr) in xs.iter().enumerate() {
            let mismatches = xr
                .iter()
                .zip(&host)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            if mismatches != 0 {
                bad += 1;
                if bad <= 5 {
                    let i = xr
                        .iter()
                        .zip(&host)
                        .position(|(a, b)| a.to_bits() != b.to_bits())
                        .unwrap();
                    println!(
                        "FAIL {} replays {replay}..{} rank {rk}: {mismatches}/{COUNT} words differ, first i={i} got {:e} want {:e}",
                        arm.name(), replay + nb, xr[i], host[i]
                    );
                }
            }
        }
        replay += nb;
    }
    // gens, wait blocks, canaries
    let codes = [ranks[0].wait_code(), ranks[1].wait_code()];
    let mut canary_ok = true;
    if matches!(
        arm,
        Arm::Push(..) | Arm::Push3(..) | Arm::Ll | Arm::Ll128 | Arm::Prod
    ) {
        for (rk, r) in ranks.iter().enumerate() {
            r.bind();
            let (buf, total, spots): (&DeviceBuffer, usize, Vec<usize>) = match arm {
                Arm::Push(..) | Arm::Push3(..) => (
                    &r.mbox,
                    P_TOTAL,
                    vec![
                        0,
                        P_BANK0 + PAYLOAD,
                        P_BANK0 + P_BANK_STRIDE + PAYLOAD,
                        P_FLAGS + 512,
                    ],
                ),
                _ => (
                    &r.mbox_ll,
                    LL_TOTAL,
                    vec![
                        0,
                        LL_BANK0 + LL_PAYLOAD,
                        LL_BANK0 + LL_BANK_STRIDE + LL_PAYLOAD,
                    ],
                ),
            };
            let mut raw = vec![0u8; total];
            r.gpu.hip.memcpy_dtoh(&mut raw, buf).expect("dtoh mbox");
            for s in spots {
                if raw[s..s + CANARY].iter().any(|&b| b != 0xCA) {
                    canary_ok = false;
                    println!("FAIL {} rank {rk}: canary at +{s} overwritten", arm.name());
                }
            }
        }
    }
    let ok = bad == 0 && codes == [0, 0] && canary_ok;
    let (ra, rb) = (
        ranks[0].gpu.take_device_wait_error().is_ok(),
        ranks[1].gpu.take_device_wait_error().is_ok(),
    );
    println!(
        "{} {}: {replays} replays (pipeline {batch}) x {BARRIERS} collectives ({} generations), {bad} bad batches/replays, wait codes {codes:?}, incumbent-wait ok {ra}/{rb}, canaries {canary_ok}, {:.1}s",
        if ok && ra && rb { "PASS" } else { "FAIL" },
        arm.name(),
        replays * BARRIERS,
        t0.elapsed().as_secs_f64()
    );
    destroy(ranks, caps);
    ok && ra && rb
}

fn destroy(ranks: &[Rank; 2], caps: [Captured; 2]) {
    for (r, c) in ranks.iter().zip(caps) {
        r.bind();
        r.gpu.hip.device_synchronize().expect("sync");
        r.gpu.hip.graph_exec_destroy(c.exec).expect("destroy exec");
        r.gpu.hip.graph_destroy(c.graph).expect("destroy graph");
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn timing(ranks: &mut [Rank; 2], arms: &[Arm], n: usize, trials: usize) {
    let x_init: Vec<f32> = (0..COUNT).map(x0).collect();
    for r in ranks.iter() {
        r.reset_state(&x_init);
    }
    let caps: Vec<[Captured; 2]> = arms.iter().map(|&a| capture(ranks, a)).collect();
    let mut events: Vec<Vec<Event>> = Vec::new();
    for r in ranks.iter() {
        r.bind();
        events.push(
            (0..2 * n)
                .map(|_| r.gpu.hip.event_create().expect("event"))
                .collect(),
        );
    }
    // warm up each arm, then interleave arms per trial
    let mut wall = vec![vec![]; arms.len()];
    let mut gpu_med: Vec<Vec<f64>> = vec![vec![]; arms.len()];
    let mut gpu_min: Vec<Vec<f64>> = vec![vec![]; arms.len()];
    for trial in 0..trials + 1 {
        for (ai, _) in arms.iter().enumerate() {
            let t = Instant::now();
            for rep in 0..n {
                for rk in 0..2 {
                    let r = &ranks[rk];
                    r.bind();
                    r.gpu
                        .hip
                        .event_record(&events[rk][2 * rep], Some(r.stream()))
                        .expect("ev");
                    launch_graph(ranks, &caps[ai], rk);
                    r.gpu
                        .hip
                        .event_record(&events[rk][2 * rep + 1], Some(r.stream()))
                        .expect("ev");
                }
            }
            for r in ranks.iter() {
                r.sync();
            }
            let w = t.elapsed().as_secs_f64() * 1e6 / n as f64;
            if trial == 0 {
                continue; // warmup trial
            }
            // GPU time per replay on rank 0 and rank 1, take the larger
            let mut per: Vec<f64> = Vec::with_capacity(n);
            for rep in 0..n {
                let mut worst = 0f64;
                for rk in 0..2 {
                    let r = &ranks[rk];
                    r.bind();
                    let ms = r
                        .gpu
                        .hip
                        .event_elapsed_ms(&events[rk][2 * rep], &events[rk][2 * rep + 1])
                        .expect("elapsed") as f64;
                    worst = worst.max(ms * 1e3);
                }
                per.push(worst);
            }
            let mut sorted = per.clone();
            gpu_min[ai].push(sorted.iter().cloned().fold(f64::INFINITY, f64::min));
            gpu_med[ai].push(median(&mut sorted));
            wall[ai].push(w);
        }
    }
    let codes = [ranks[0].wait_code(), ranks[1].wait_code()];
    let xs = [ranks[0].download_x(), ranks[1].download_x()];
    let same = xs[0]
        .iter()
        .zip(&xs[1])
        .all(|(a, b)| a.to_bits() == b.to_bits());
    println!("timing: n={n} replays/trial, {trials} trials/arm, wait codes {codes:?}, ranks' final x identical {same}");
    let mut base_med = 0.0;
    for (ai, a) in arms.iter().enumerate() {
        let mut g = gpu_med[ai].clone();
        let med = median(&mut g);
        if *a == Arm::Base {
            base_med = med;
        }
        println!(
            "RAW arm={} gpu_us_per_replay_median_per_trial={:?} gpu_us_min_per_trial={:?} wall_us_per_replay={:?}",
            a.name(),
            gpu_med[ai].iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>(),
            gpu_min[ai].iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>(),
            wall[ai].iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>(),
        );
        println!(
            "SUMMARY arm={:<5} chain {:7.2} us/replay  = {:6.3} us per [producer+collective]  marginal collective vs base {:6.3} us",
            a.name(),
            med,
            med / BARRIERS as f64,
            (med - base_med) / BARRIERS as f64
        );
    }
    for (arm_caps, _) in caps.into_iter().zip(arms) {
        destroy(ranks, arm_caps);
    }
}

fn main() {
    let arms_s =
        std::env::var("PROBE_ARMS").unwrap_or_else(|_| "base,inc,push3:256:0,ll128,prod".into());
    let arms: Vec<Arm> = arms_s
        .split(',')
        .map(|s| match s.trim() {
            "base" => Arm::Base,
            "inc" => Arm::Inc,
            "ll" => Arm::Ll,
            "ll128" => Arm::Ll128,
            "prod" => Arm::Prod,
            other => {
                let p: Vec<&str> = other.split(':').collect();
                match p[0] {
                    "push3" => Arm::Push3(
                        p.get(1).map_or(256, |v| v.parse().unwrap()),
                        p.get(2).map_or(0, |v| v.parse().unwrap()),
                    ),
                    "push" => Arm::Push(
                        p.get(1).map_or(256, |v| v.parse().unwrap()),
                        p.get(2).map_or(0, |v| v.parse().unwrap()),
                    ),
                    _ => panic!("unknown arm {other}"),
                }
            }
        })
        .collect();
    let replays = env_usize("PROBE_CORRECT_REPLAYS", 10_000);
    let n = env_usize("PROBE_TIMING_N", 200);
    let trials = env_usize("PROBE_TIMING_TRIALS", 15);
    let delays = env_usize("PROBE_DELAYS", 1) != 0;
    let pipeline = env_usize("PROBE_PIPELINE", 1);
    let seed = env_usize("PROBE_SEED", 0x5EED_1234) as u64;

    let mut gpus = vec![
        Gpu::init_with_device(0).expect("init gpu0"),
        Gpu::init_with_device(1).expect("init gpu1"),
    ];
    for g in &gpus {
        assert_eq!(g.arch, "gfx1201", "probe needs gfx1201, got {}", g.arch);
        g.bind_thread().unwrap();
    }
    let arch = gpus[0].arch.clone();
    let mut compiler = KernelCompiler::new(&arch, String::new()).expect("compiler");
    let artifact = compiler
        .compile("tp_peer_push_probe", SOURCE)
        .expect("compile probe")
        .to_owned();
    let image = std::fs::read(&artifact).expect("read hsaco");
    let prod_artifact = compiler
        .compile(
            "tp2_peer_push_shipped",
            rdna_compute::kernels::TP_PEER_PUSH_GFX1201_SRC,
        )
        .expect("compile shipped kernel")
        .to_owned();
    let prod_image = std::fs::read(&prod_artifact).expect("read shipped hsaco");

    // Allocation order: everything the peer will touch is allocated BEFORE
    // peer access is enabled.
    let host_seg = PAYLOAD.next_multiple_of(256);
    gpus[0].bind_thread().unwrap();
    let shadow_host = gpus[0]
        .hip
        .host_malloc(host_seg * 4, hip_bridge::HIP_HOST_MALLOC_MAPPED)
        .expect("host shadow");
    let mut ranks: Vec<Rank> = Vec::new();
    for (rk, mut gpu) in gpus.drain(..).enumerate() {
        gpu.bind_thread().unwrap();
        gpu.active_stream = Some(gpu.hip.stream_create().expect("stream"));
        gpu.ensure_tp_graph_signal_gfx1100_kernels()
            .expect("incumbent kernels");
        gpu.ensure_device_wait_block().expect("wait block");
        let hip: &HipRuntime = &gpu.hip;
        let module = hip.module_load_data(&image).expect("module");
        let module_prod = hip.module_load_data(&prod_image).expect("shipped module");
        let f_shipped = hip
            .module_get_function(&module_prod, "tp2_peer_push_reduce_gfx1201")
            .unwrap();
        let f_prod = hip.module_get_function(&module, "probe_producer").unwrap();
        let f_spin = hip.module_get_function(&module, "probe_spin").unwrap();
        let f_push = hip
            .module_get_function(&module, "probe_push_reduce")
            .unwrap();
        let f_push3 = hip
            .module_get_function(&module, "probe_push3_reduce")
            .unwrap();
        let f_ll128 = hip
            .module_get_function(&module, "probe_ll128_reduce")
            .unwrap();
        let f_ll = hip
            .module_get_function(&module, "probe_push_reduce_ll")
            .unwrap();
        let x = hip.malloc(PAYLOAD).unwrap();
        let part = hip.malloc(PAYLOAD).unwrap();
        let salt = hip.malloc(256).unwrap();
        let gen_push = hip.malloc(256).unwrap();
        let gen_ll = hip.malloc(256).unwrap();
        let mbox = hip
            .malloc_peer_finegrained(P_TOTAL)
            .expect("fine-grained mailbox");
        let mbox_ll = hip
            .malloc_peer_finegrained(LL_TOTAL)
            .expect("fine-grained ll mailbox");
        let signal = hip.malloc_signal(8).expect("signal");
        let shadow_dev = hip
            .host_get_device_pointer(shadow_host, 0)
            .expect("shadow alias");
        let wb_host = hip
            .host_malloc(
                16,
                hip_bridge::HIP_HOST_MALLOC_MAPPED | hip_bridge::HIP_HOST_MALLOC_COHERENT,
            )
            .expect("wb") as *mut DeviceWaitBlock;
        let khz = hip
            .get_device_attribute(10_017, gpu.device_id)
            .expect("wall clock") as u64;
        unsafe {
            std::ptr::write_volatile(
                wb_host,
                DeviceWaitBlock {
                    code: 0,
                    detail: 0,
                    timeout_ticks: timeout_ticks(khz),
                },
            )
        };
        let wb_dev = hip
            .host_get_device_pointer(wb_host.cast(), 0)
            .expect("wb alias");
        println!(
            "rank {rk}: device {} arch {} mailbox {:p} ({} B fine-grained) ll {:p} ({} B) wall_clock {} kHz",
            gpu.device_id, gpu.arch, mbox.as_ptr(), P_TOTAL, mbox_ll.as_ptr(), LL_TOTAL, khz
        );
        ranks.push(Rank {
            gpu,
            _module: module,
            _module_prod: module_prod,
            f_shipped,
            f_prod,
            f_spin,
            f_push,
            f_push3,
            f_ll,
            f_ll128,
            x,
            part,
            salt,
            gen_push,
            gen_ll,
            mbox,
            mbox_ll,
            signal,
            shadow_dev,
            wb_host,
            wb_dev,
            capturing: std::cell::Cell::new(false),
            blobs: std::cell::RefCell::new(Vec::new()),
        });
    }
    // peer access
    for i in 0..2 {
        let (a, b) = (ranks[i].gpu.device_id, ranks[1 - i].gpu.device_id);
        ranks[i].bind();
        assert!(
            ranks[i].gpu.hip.can_access_peer(a, b).unwrap(),
            "no peer access {a}->{b}"
        );
        ranks[i].gpu.hip.enable_peer_access(b).expect("enable peer");
    }
    let [r0, r1]: [Rank; 2] = ranks.try_into().ok().expect("two ranks");
    let mut ranks = [r0, r1];

    // The incumbent signal memory was zeroed in reset_state; run correctness
    // arms first (each from a clean reset), then the interleaved timing.
    let mut all_ok = true;
    for &arm in &arms {
        // base has no collective; push with diagnostic mode bits is timing-only
        if arm == Arm::Base || matches!(arm, Arm::Push(_, m) | Arm::Push3(_, m) if m & 7 != 0) {
            continue;
        }
        all_ok &= correctness(&mut ranks, arm, replays, delays, seed, pipeline);
    }
    if all_ok {
        timing(&mut ranks, &arms, n, trials);
    } else {
        println!("correctness failed; timing skipped");
    }

    if env_usize("PROBE_TIMEOUT_TEST", 1) != 0 {
        // Missing-peer injection: only rank 0 enters the collective; its
        // bounded wait must record PROBE_SITE and exit within ~0.5 s. Each
        // protocol arm gets its own reset (the previous one leaves its
        // generation advanced on rank 0 only).
        for &arm in &arms {
            let launch_one: Box<dyn Fn(&[Rank; 2])> = match arm {
                Arm::Push3(t, m) if m & 7 == 0 => Box::new(move |r| enqueue_push(r, 0, t, m, true)),
                Arm::Ll128 => Box::new(|r| enqueue_ll(r, 0, 512, true)),
                Arm::Prod => Box::new(|r| enqueue_prod(r, 0)),
                _ => continue,
            };
            let x_init: Vec<f32> = (0..COUNT).map(x0).collect();
            for r in ranks.iter() {
                r.reset_state(&x_init);
            }
            let t = Instant::now();
            launch_one(&ranks);
            ranks[0].sync();
            let code = ranks[0].wait_code();
            // PROBE_SITE (4) for the lab kernels, the shipped site (9) for `prod`
            let want = if arm == Arm::Prod { 9 } else { 4 };
            println!(
                "{} timeout injection {}: bounded wait exited after {:.0} ms, wait code {code} (expect {want})",
                if code == want { "PASS" } else { "FAIL" },
                arm.name(),
                t.elapsed().as_secs_f64() * 1e3
            );
            all_ok &= code == want;
        }
    }
    println!("{}", if all_ok { "ALL PASS" } else { "SOME FAIL" });
    std::process::exit(if all_ok { 0 } else { 1 });
}
