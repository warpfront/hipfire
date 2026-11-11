// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Hardware-vs-emulator probes of the PeaceMaker emulator's helper-call capabilities (item 21):
//   (a) probe_call          s_getpc_b64 / s_swappc_b64 / s_setpc_b64 call + return of two different leaf callees from
//                           the kernel (sequential, uniform; the second consumes the first's result), a divergent
//                           call site (only some lanes active, EXEC at entry and after), values live across calls
//   (b) probe_scratch       per-lane scratch frame of a leaf callee (s32-based, dynamic per-lane index, the Leviathan
//                           helper's shape)
//   (c) probe_atomic        flat_atomic_add_u32: returning form (distinct dwords) and non-returning form (one dword)
//   (d) probe_atomic_collide / probe_atomic_lds
//                           negative controls: hardware runs them, the emulator MUST refuse
//   (e) probe_salu_f32      s_sub_f32, s_cmp_{gt,neq,ngt}_f32, s_cvt_f32_i32, v_cmpx_{gt,nge}_f32_e32 sweeps
//   (f) probe_v_*           the VALU float instructions of the lane_leviathan objects the emulator had no measured
//                           model for: v_subrev_f32, v_dual_{sub,fmamk,fmaak}_f32, v_frexp_{exp,mant}, v_cvt_*, v_rcp/exp,
//                           v_ldexp, v_rndne, v_cndmask_b32 abs/neg, v_fma_f32 neg, v_div_*, v_cmp*/v_cmpx*/class (see (f) below)
//
// Every output word is independent of the code object's load base: PCs are reported as differences to a reference
// `s_getpc_b64` taken in the kernel, and no function pointer is formed (absolute addresses are loader relocations).
// All kernels share one signature `(const u32* in, u32* out, u32 n)` and run with block 64 (two wave32 waves).
// Built by `build.sh` (device-only code objects, gfx1201 and gfx1200).
#include <hip/hip_runtime.h>
#include <stdint.h>

#define PROBE(name)                                                                                              \
    extern "C" __global__ __launch_bounds__(64) void name(const uint32_t* __restrict__ in,                        \
                                                           uint32_t* __restrict__ out, uint32_t n)

// ---- scalar-state readers (each ends with the SALU-producer wait so a VALU consumer never races the write) ----
static __device__ __forceinline__ uint64_t pc_now() {
    uint64_t p;
    asm volatile("s_getpc_b64 %0\n\ts_wait_alu depctr_sa_sdst(0)" : "=s"(p));
    return p;
}
// The return address of the current call: valid only at function entry, before any call clobbers s[30:31].
static __device__ __forceinline__ uint64_t ra_now() {
    uint64_t p;
    asm volatile("s_mov_b64 %0, s[30:31]\n\ts_wait_alu depctr_sa_sdst(0)" : "=s"(p));
    return p;
}
static __device__ __forceinline__ uint32_t exec_lo_now() {
    uint32_t v;
    asm volatile("s_mov_b32 %0, exec_lo\n\ts_wait_alu depctr_sa_sdst(0)" : "=s"(v));
    return v;
}
static __device__ __forceinline__ uint64_t uni64(uint64_t v) {
    return ((uint64_t)__builtin_amdgcn_readfirstlane((uint32_t)(v >> 32)) << 32) |
           (uint64_t)__builtin_amdgcn_readfirstlane((uint32_t)v);
}
static __device__ __forceinline__ uint32_t diff(uint64_t pc, uint64_t ref) {
    return (uint32_t)(pc - uni64(ref));
}

// ======================================== (a) probe_call ========================================
struct RB {
    uint32_t pc, ra, exec, val;
};
// Leaf callee: its own getpc, its return address, EXEC at entry, a lane-dependent result.
__device__ __attribute__((noinline)) RB call_b(uint64_t ref, uint32_t x, uint32_t y) {
    uint64_t ra = ra_now();
    uint64_t pc = pc_now();
    uint32_t ex = exec_lo_now();
    RB r;
    r.pc = diff(pc, ref);
    r.ra = diff(ra, ref);
    r.exec = ex;
    r.val = (x * 0x01000193u) ^ (y + (x >> 3));
    return r;
}
struct RA {
    uint32_t pc, ra, exec, val, live;
};
// Second leaf callee (a different extent from call_b): its own getpc, return address, EXEC at entry, a result and a
// value computed in the callee.
__device__ __attribute__((noinline)) RA call_a(uint64_t ref, uint32_t x, uint32_t y) {
    uint64_t ra = ra_now();
    uint64_t pc = pc_now();
    uint32_t ex = exec_lo_now();
    RA r;
    r.pc = diff(pc, ref);
    r.ra = diff(ra, ref);
    r.exec = ex;
    r.live = (x ^ 0x9e3779b9u) + y * 3u;
    r.val = r.live ^ (x * 0x85ebca6bu);
    return r;
}
// 24 words per lane.
PROBE(probe_call) {
    uint32_t t = threadIdx.x;
    uint32_t* o = out + t * 24u;
    uint64_t ref = pc_now();
    uint32_t x = in[t];
    uint32_t live1 = x * 3u + 7u;  // live across every call below
    uint32_t live2 = t ^ 0x5a5a5a5au;
    RA a = call_a(ref, x, t);  // uniform call site, full EXEC
    uint64_t pc1 = pc_now();
    uint32_t ex1 = exec_lo_now();
    RB b = call_b(ref, a.val ^ x, live1);  // second leaf, consumes the first's result
    uint64_t pc2 = pc_now();
    uint32_t ex2 = exec_lo_now();
    RB d;
    d.pc = d.ra = d.exec = d.val = 0xffffffffu;
    uint32_t exd = 0xfeedfeedu;
    if (x & 1u) {  // divergent call site (call_b again, a different return address): only some lanes active
        d = call_b(ref, x ^ t, live1 + b.val);
        exd = exec_lo_now();
    }
    uint64_t pc3 = pc_now();
    uint32_t ex3 = exec_lo_now();
    o[0] = a.pc;
    o[1] = a.ra;
    o[2] = a.exec;
    o[3] = a.val;
    o[4] = a.live;
    o[5] = diff(pc1, ref);
    o[6] = ex1;
    o[7] = b.pc;
    o[8] = b.ra;
    o[9] = b.exec;
    o[10] = b.val;
    o[11] = diff(pc2, ref);
    o[12] = ex2;
    o[13] = live1;
    o[14] = live2;
    o[15] = d.pc;
    o[16] = d.ra;
    o[17] = d.exec;
    o[18] = d.val;
    o[19] = exd;
    o[20] = diff(pc3, ref);
    o[21] = ex3;
    o[22] = x;
    o[23] = 0xc0de0000u | t;
}

// ======================================== (b) probe_scratch ========================================
struct RSI {
    uint32_t v, addr;
};
// Leaf frame (the Leviathan helper's shape): its own scratch array addressed from s32, no call inside.
__device__ __attribute__((noinline)) RSI scratch_inner(uint32_t lane, uint32_t seed) {
    uint32_t b[48];
    for (uint32_t i = 0; i < 48u; ++i) b[i] = (seed ^ (i * 0x45d9f3bu)) + lane * 0x27d4eb2fu + i;
    for (uint32_t k = 0; k < 6u; ++k) b[(lane * 7u + k * 11u + seed) % 48u] += (k + 1u) * 0x1001u;
    uint32_t i0 = (seed * 3u + lane) % 48u;
    RSI r;
    r.v = b[i0] ^ b[(i0 + 17u) % 48u] ^ b[(lane + seed) % 48u];
    r.addr = (uint32_t)(uintptr_t)&b[i0];
    return r;
}
// 2 words per lane: leaf frame only.
PROBE(probe_scratch) {
    uint32_t t = threadIdx.x;
    RSI r = scratch_inner(t, in[t]);
    uint32_t* o = out + t * 2u;
    o[0] = r.v;
    o[1] = r.addr;
}

// ======================================== (c) probe_atomic ========================================
// `p` is a generic pointer: the compiler must emit flat atomics.
__device__ __attribute__((noinline)) uint32_t atomic_fn(unsigned* p, unsigned* counter, uint32_t lane, uint32_t v) {
    uint32_t old = __hip_atomic_fetch_add(p + lane, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);  // returning
    (void)__hip_atomic_fetch_add(counter, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);            // non-returning
    return old;
}
// out[0..64) old values, out[64..128) the data dwords (initialised by the host), out[128] the counter (host-initialised).
PROBE(probe_atomic) {
    uint32_t t = threadIdx.x;
    out[t] = atomic_fn(out + 64, out + 128, t, in[t]);
}

// ======================================== (d) negative controls ========================================
__device__ __attribute__((noinline)) uint32_t add_ret(unsigned* p, uint32_t v) {
    return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
// Returning flat add, every lane on one dword (out[64], host-initialised). Hardware order of the returned values is
// not asserted; the emulator must refuse ("two active lanes meet in one dword").
PROBE(probe_atomic_collide) {
    uint32_t t = threadIdx.x;
    out[t] = add_ret(out + 64, in[t]);
}
// Returning flat add through a generic pointer to a __shared__ word. Hardware runs it; the emulator must refuse
// (LDS aperture).
PROBE(probe_atomic_lds) {
    __shared__ uint32_t w[64];
    uint32_t t = threadIdx.x;
    w[t] = in[t];
    __syncthreads();
    uint32_t old = add_ret(&w[t], 5u + t);
    __syncthreads();
    out[t] = old;
    out[64 + t] = w[t];
}

// ======================================== (e) probe_salu_f32 ========================================
// `in` holds n records (a, b, i) of bit patterns (n a multiple of 64); wave `chunk` owns records [32 chunk, 32 chunk + 32).
// out[5 r .. 5 r + 5) for record r: s_sub_f32(a,b), SCC of s_cmp_gt_f32(a,b), s_cmp_neq_f32, s_cmp_ngt_f32,
// s_cvt_f32_i32(i as i32); out[5 n + 2 chunk] / out[5 n + 2 chunk + 1]: the 32-bit EXEC mask left by
// v_cmpx_gt_f32_e32 / v_cmpx_nge_f32_e32 with lane l comparing record 32 chunk + l (a_l, b_l).
PROBE(probe_salu_f32) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t chunk = __builtin_amdgcn_readfirstlane(blockIdx.x * 2u + (threadIdx.x >> 5));
    uint32_t base = chunk * 32u;
    for (uint32_t j = 0; j < 32u; ++j) {
        uint32_t idx = base + j;
        uint32_t a = __builtin_amdgcn_readfirstlane(in[3u * idx]);
        uint32_t b = __builtin_amdgcn_readfirstlane(in[3u * idx + 1u]);
        uint32_t iv = __builtin_amdgcn_readfirstlane(in[3u * idx + 2u]);
        uint32_t sub, gt, neq, ngt, cvt;
        asm volatile("s_sub_f32 %0, %1, %2\n\ts_wait_alu depctr_sa_sdst(0)" : "=s"(sub) : "s"(a), "s"(b));
        asm volatile("s_cmp_gt_f32 %1, %2\n\ts_cselect_b32 %0, 1, 0\n\ts_wait_alu depctr_sa_sdst(0)"
                     : "=s"(gt) : "s"(a), "s"(b) : "scc");
        asm volatile("s_cmp_neq_f32 %1, %2\n\ts_cselect_b32 %0, 1, 0\n\ts_wait_alu depctr_sa_sdst(0)"
                     : "=s"(neq) : "s"(a), "s"(b) : "scc");
        asm volatile("s_cmp_ngt_f32 %1, %2\n\ts_cselect_b32 %0, 1, 0\n\ts_wait_alu depctr_sa_sdst(0)"
                     : "=s"(ngt) : "s"(a), "s"(b) : "scc");
        asm volatile("s_cvt_f32_i32 %0, %1\n\ts_wait_alu depctr_sa_sdst(0)" : "=s"(cvt) : "s"(iv));
        if (lane == 0) {
            out[5u * idx + 0] = sub;
            out[5u * idx + 1] = gt;
            out[5u * idx + 2] = neq;
            out[5u * idx + 3] = ngt;
            out[5u * idx + 4] = cvt;
        }
    }
    uint32_t al = in[3u * (base + lane)];
    uint32_t bl = in[3u * (base + lane) + 1u];
    uint32_t m_gt, m_nge, save;
    asm volatile("s_mov_b32 %1, exec_lo\n\tv_cmpx_gt_f32_e32 %2, %3\n\ts_mov_b32 %0, exec_lo\n\ts_mov_b32 exec_lo, %1\n\t"
                 "s_wait_alu depctr_sa_sdst(0)"
                 : "=&s"(m_gt), "=&s"(save) : "v"(al), "v"(bl));
    asm volatile("s_mov_b32 %1, exec_lo\n\tv_cmpx_nge_f32_e32 %2, %3\n\ts_mov_b32 %0, exec_lo\n\ts_mov_b32 exec_lo, %1\n\t"
                 "s_wait_alu depctr_sa_sdst(0)"
                 : "=&s"(m_nge), "=&s"(save) : "v"(al), "v"(bl));
    if (lane == 0) {
        out[5u * n + 2u * chunk] = m_gt;
        out[5u * n + 2u * chunk + 1u] = m_nge;
    }
}

// ======================================== (f) VALU float-corner probes (probe_v_*) ========================================
// Every instruction the emulator has no measured model for (or only a hypothesis) in the committed lane_leviathan objects
// (hipcc gfx1200/gfx1201 and the PM twin, whose device ELF is byte-identical to the gfx1201 hipcc object): the
// llv_tail helper and the kernel. Each target is ONE raw inline-asm instruction of exactly the form the objects contain
// (no compiled FP expression), so the hardware answer is that instruction's, not the compiler's.
//
// All probe_v_* kernels read `n` records (a, b, c) of raw bit patterns from `in` (record `idx = 64 blockIdx.x + threadIdx.x`,
// `n` a multiple of 64) and write `W` words per record to out[W idx ..] (W and the field order per kernel below; the
// host fixes the record sets: NaN of every class and sign and payload, +-inf, +-0, denormals, finite extremes, random).
// Compare kernels pack one result bit per field. Nothing here is compiled FP arithmetic.
#define RECORD                                                                                                         \
    const uint32_t idx = blockIdx.x * 64u + threadIdx.x;                                                               \
    if (idx >= n) return;                                                                                              \
    const uint32_t a = in[3u * idx], b = in[3u * idx + 1u], c = in[3u * idx + 2u];                                     \
    (void)a; (void)b; (void)c;

// v_sub_f32_e32 (measured, regression) and v_subrev_f32_e32 (the llv_tail helper's v_subrev_f32_e32, D = S1 - S0).
// W = 2: [sub(a,b), subrev(a,b)]
PROBE(probe_v_sub) {
    RECORD
    uint32_t s, r;
    asm volatile("v_sub_f32_e32 %0, %1, %2" : "=v"(s) : "v"(a), "v"(b));
    asm volatile("v_subrev_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    out[2u * idx] = s;
    out[2u * idx + 1u] = r;
}

// v_dual_sub_f32 as X and as Y half (both halves subtract; X: a - b, Y: b - a).
// W = 2: [X(a,b), Y(b,a)]
PROBE(probe_v_dual_sub) {
    RECORD
    uint32_t x, y;
    asm volatile("v_mov_b32 v0, %2\n\tv_mov_b32 v1, %3\n\tv_mov_b32 v2, %3\n\tv_mov_b32 v3, %2\n\t"
                 "v_dual_sub_f32 v4, v0, v2 :: v_dual_sub_f32 v5, v1, v3\n\tv_mov_b32 %0, v4\n\tv_mov_b32 %1, v5"
                 : "=&v"(x), "=&v"(y) : "v"(a), "v"(b) : "v0", "v1", "v2", "v3", "v4", "v5");
    out[2u * idx] = x;
    out[2u * idx + 1u] = y;
}

// Literal K list of the fmamk/fmaak probes: the real helper constant, 1.0, +0, -0, +inf, qNaN, sNaN, min denormal.
#define KLIST(X) X(0x32a5705f) X(0x3f800000) X(0x00000000) X(0x80000000) X(0x7f800000) X(0x7fc00000) X(0x7f800001) X(0x00000001)

// v_dual_fmamk_f32 (D = S0 * K + S1) as the X half and as the Y half, for each K of KLIST.
// W = 16: for K in KLIST: [X(a, K, b), Y(a, K, b)]
PROBE(probe_v_dual_fmamk) {
    RECORD
    uint32_t r[16];
    uint32_t j = 0;
#define FMAMK(K)                                                                                                       \
    asm volatile("v_mov_b32 v0, %2\n\tv_mov_b32 v1, %2\n\tv_mov_b32 v2, %3\n\tv_mov_b32 v3, %3\n\t"                   \
                 "v_dual_fmamk_f32 v4, v0, " #K ", v2 :: v_dual_mov_b32 v5, v1\n\tv_mov_b32 %0, v4\n\t"               \
                 "v_dual_mov_b32 v4, v0 :: v_dual_fmamk_f32 v5, v1, " #K ", v3\n\tv_mov_b32 %1, v5"                    \
                 : "=&v"(r[j]), "=&v"(r[j + 1]) : "v"(a), "v"(b) : "v0", "v1", "v2", "v3", "v4", "v5");               \
    j += 2;
    KLIST(FMAMK)
#undef FMAMK
    for (uint32_t k = 0; k < 16u; ++k) out[16u * idx + k] = r[k];
}

// v_dual_fmaak_f32 (D = S0 * S1 + K) as the X half and as the Y half, for each K of KLIST.
// W = 16: for K in KLIST: [X(a, b, K), Y(a, b, K)]
PROBE(probe_v_dual_fmaak) {
    RECORD
    uint32_t r[16];
    uint32_t j = 0;
#define FMAAK(K)                                                                                                       \
    asm volatile("v_mov_b32 v0, %2\n\tv_mov_b32 v1, %2\n\tv_mov_b32 v2, %3\n\tv_mov_b32 v3, %3\n\t"                   \
                 "v_dual_fmaak_f32 v4, v0, v2, " #K " :: v_dual_mov_b32 v5, v1\n\tv_mov_b32 %0, v4\n\t"               \
                 "v_dual_mov_b32 v4, v0 :: v_dual_fmaak_f32 v5, v1, v3, " #K "\n\tv_mov_b32 %1, v5"                    \
                 : "=&v"(r[j]), "=&v"(r[j + 1]) : "v"(a), "v"(b) : "v0", "v1", "v2", "v3", "v4", "v5");               \
    j += 2;
    KLIST(FMAAK)
#undef FMAAK
    for (uint32_t k = 0; k < 16u; ++k) out[16u * idx + k] = r[k];
}

// v_fmamk_f32_e32 (VOP2, the helper's single use) for each K of KLIST (D = S0 * K + S1).
// W = 8: for K in KLIST: [fmamk(a, K, b)]
PROBE(probe_v_fmamk) {
    RECORD
    uint32_t r[8];
    uint32_t j = 0;
#define FMAMK1(K)                                                                                                      \
    asm volatile("v_fmamk_f32 %0, %1, " #K ", %2" : "=v"(r[j]) : "v"(a), "v"(b));                                      \
    ++j;
    KLIST(FMAMK1)
#undef FMAMK1
    for (uint32_t k = 0; k < 8u; ++k) out[8u * idx + k] = r[k];
}

// v_frexp_exp_i32_f32_e32 and v_frexp_mant_f32_e32 (kernel). W = 2: [frexp_exp(a), frexp_mant(a)]
PROBE(probe_v_frexp) {
    RECORD
    uint32_t e, m;
    asm volatile("v_frexp_exp_i32_f32_e32 %0, %1" : "=v"(e) : "v"(a));
    asm volatile("v_frexp_mant_f32_e32 %0, %1" : "=v"(m) : "v"(a));
    out[2u * idx] = e;
    out[2u * idx + 1u] = m;
}

// v_cvt_i32_f32_e32 (a as f32), v_cvt_f32_i32_e32 / v_cvt_f32_u32_e32 (a as an integer), v_rndne_f32_e32. W = 4
PROBE(probe_v_cvt) {
    RECORD
    uint32_t i32, fi, fu, rn;
    asm volatile("v_cvt_i32_f32_e32 %0, %1" : "=v"(i32) : "v"(a));
    asm volatile("v_cvt_f32_i32_e32 %0, %1" : "=v"(fi) : "v"(a));
    asm volatile("v_cvt_f32_u32_e32 %0, %1" : "=v"(fu) : "v"(a));
    asm volatile("v_rndne_f32_e32 %0, %1" : "=v"(rn) : "v"(a));
    out[4u * idx] = i32;
    out[4u * idx + 1u] = fi;
    out[4u * idx + 2u] = fu;
    out[4u * idx + 3u] = rn;
}

// v_rcp_f32_e32 (a). W = 1
PROBE(probe_v_rcp) {
    RECORD
    uint32_t r;
    asm volatile("v_rcp_f32_e32 %0, %1" : "=v"(r) : "v"(a));
    out[idx] = r;
}

// v_exp_f32_e32 (a). W = 1
PROBE(probe_v_exp) {
    RECORD
    uint32_t r;
    asm volatile("v_exp_f32_e32 %0, %1" : "=v"(r) : "v"(a));
    out[idx] = r;
}

// v_ldexp_f32 (a, b as an integer; VOP3 as in the helper). W = 1
PROBE(probe_v_ldexp) {
    RECORD
    uint32_t r;
    asm volatile("v_ldexp_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    out[idx] = r;
}

// v_cndmask_b32_e64 with an abs/neg source modifier on one of the data operands; the mask bit of lane l is c[0] (sel)
// and ~c[0] (inv). W = 8: for (sel, inv): [|a|, b], [-a, b], [a, |b|], [a, -b]
PROBE(probe_v_cndmask) {
    RECORD
    const uint32_t sel = c & 1u, inv = (c & 1u) ^ 1u;
    uint32_t m, r[8];
    asm volatile("v_cmp_ne_u32_e64 %0, 0, %4\n\tv_cndmask_b32_e64 %1, |%2|, %3, %0" : "=&s"(m), "=v"(r[0]) : "v"(a), "v"(b), "v"(sel));
    asm volatile("v_cmp_ne_u32_e64 %0, 0, %4\n\tv_cndmask_b32_e64 %1, -%2, %3, %0" : "=&s"(m), "=v"(r[1]) : "v"(a), "v"(b), "v"(sel));
    asm volatile("v_cmp_ne_u32_e64 %0, 0, %4\n\tv_cndmask_b32_e64 %1, %2, |%3|, %0" : "=&s"(m), "=v"(r[2]) : "v"(a), "v"(b), "v"(sel));
    asm volatile("v_cmp_ne_u32_e64 %0, 0, %4\n\tv_cndmask_b32_e64 %1, %2, -%3, %0" : "=&s"(m), "=v"(r[3]) : "v"(a), "v"(b), "v"(sel));
    asm volatile("v_cmp_ne_u32_e64 %0, 0, %4\n\tv_cndmask_b32_e64 %1, |%2|, %3, %0" : "=&s"(m), "=v"(r[4]) : "v"(a), "v"(b), "v"(inv));
    asm volatile("v_cmp_ne_u32_e64 %0, 0, %4\n\tv_cndmask_b32_e64 %1, -%2, %3, %0" : "=&s"(m), "=v"(r[5]) : "v"(a), "v"(b), "v"(inv));
    asm volatile("v_cmp_ne_u32_e64 %0, 0, %4\n\tv_cndmask_b32_e64 %1, %2, |%3|, %0" : "=&s"(m), "=v"(r[6]) : "v"(a), "v"(b), "v"(inv));
    asm volatile("v_cmp_ne_u32_e64 %0, 0, %4\n\tv_cndmask_b32_e64 %1, %2, -%3, %0" : "=&s"(m), "=v"(r[7]) : "v"(a), "v"(b), "v"(inv));
    for (uint32_t k = 0; k < 8u; ++k) out[8u * idx + k] = r[k];
}

// v_fma_f32 with no modifier, -S0, -S2 and -S0 -S2 (the helper's forms; S1 negation is not used by any object). W = 4
PROBE(probe_v_fma) {
    RECORD
    uint32_t r0, r1, r2, r3;
    asm volatile("v_fma_f32 %0, %1, %2, %3" : "=v"(r0) : "v"(a), "v"(b), "v"(c));
    asm volatile("v_fma_f32 %0, -%1, %2, %3" : "=v"(r1) : "v"(a), "v"(b), "v"(c));
    asm volatile("v_fma_f32 %0, %1, %2, -%3" : "=v"(r2) : "v"(a), "v"(b), "v"(c));
    asm volatile("v_fma_f32 %0, -%1, %2, -%3" : "=v"(r3) : "v"(a), "v"(b), "v"(c));
    out[4u * idx] = r0;
    out[4u * idx + 1u] = r1;
    out[4u * idx + 2u] = r2;
    out[4u * idx + 3u] = r3;
}

// The v_div_* family of the helper and kernel (regression: measured by the gfx1201 div receipts). W = 5:
// [div_fmas(a,b,c) with VCC = 0, div_fmas with VCC = all ones, div_fixup(a,b,c), div_scale(a,b,c) result, div_scale VCC bit]
PROBE(probe_v_div) {
    RECORD
    uint32_t f0, f1, fx, sc, sv;
    asm volatile("s_mov_b32 vcc_lo, 0\n\ts_wait_alu depctr_sa_sdst(0)\n\tv_div_fmas_f32 %0, %1, %2, %3"
                 : "=v"(f0) : "v"(a), "v"(b), "v"(c) : "vcc");
    asm volatile("s_mov_b32 vcc_lo, -1\n\ts_wait_alu depctr_sa_sdst(0)\n\tv_div_fmas_f32 %0, %1, %2, %3"
                 : "=v"(f1) : "v"(a), "v"(b), "v"(c) : "vcc");
    asm volatile("v_div_fixup_f32 %0, %1, %2, %3" : "=v"(fx) : "v"(a), "v"(b), "v"(c));
    asm volatile("v_div_scale_f32 %0, vcc_lo, %2, %3, %4\n\tv_cndmask_b32_e64 %1, 0, 1, vcc_lo"
                 : "=&v"(sc), "=v"(sv) : "v"(a), "v"(b), "v"(c) : "vcc");
    out[5u * idx] = f0;
    out[5u * idx + 1u] = f1;
    out[5u * idx + 2u] = fx;
    out[5u * idx + 3u] = sc;
    out[5u * idx + 4u] = sv;
}

// Only forms the committed objects contain (and the PeaceMaker ISA decode table has rows for) are probed. The predicate
// set is that of the objects: v_cmp_{eq,ge,gt,le,lt,o,ngt,neq,nlt}_f32 and v_cmpx_{ge,gt,lt,nge,nlt}_f32_e32; v_cmp_ge_f32
// has only the e64 form (no VOPC e32 row, none in any object: its e32 bit stays 0 in the kernel and is not compared).
//
// v_cmp_<P>_f32 for the 9 predicates P of PREDS (flag = has a VOPC e32 form): VOPC e32 (VCC), and VOP3 e64 plain, |a|, |b|,
// -a, -b, |a| |b|. One result bit per (P, form), bit index 7 pidx + form, packed 32 per word. W = 2
#define PREDS(X) X(eq, 1) X(ge, 0) X(gt, 1) X(le, 1) X(lt, 1) X(o, 1) X(ngt, 1) X(neq, 1) X(nlt, 1)
#define CMP_E32_1(P)                                                                                                   \
    asm volatile("v_cmp_" #P "_f32_e32 vcc_lo, %1, %2\n\tv_cndmask_b32_e64 %0, 0, 1, vcc_lo" : "=v"(r) : "v"(a), "v"(b) : "vcc");
#define CMP_E32_0(P) r = 0;
#define CAT_(A, B) A##B
#define CAT(A, B) CAT_(A, B)
PROBE(probe_v_cmp) {
    RECORD
    uint32_t w[2] = {0, 0};
    uint32_t bit = 0, m, r;
#define CMPF(P, E32)                                                                                                   \
    CAT(CMP_E32_, E32)(P)                                                                                              \
    w[bit >> 5] |= r << (bit & 31u); ++bit;                                                                           \
    asm volatile("v_cmp_" #P "_f32_e64 %0, %2, %3\n\tv_cndmask_b32_e64 %1, 0, 1, %0" : "=&s"(m), "=v"(r) : "v"(a), "v"(b)); \
    w[bit >> 5] |= r << (bit & 31u); ++bit;                                                                           \
    asm volatile("v_cmp_" #P "_f32_e64 %0, |%2|, %3\n\tv_cndmask_b32_e64 %1, 0, 1, %0" : "=&s"(m), "=v"(r) : "v"(a), "v"(b)); \
    w[bit >> 5] |= r << (bit & 31u); ++bit;                                                                           \
    asm volatile("v_cmp_" #P "_f32_e64 %0, %2, |%3|\n\tv_cndmask_b32_e64 %1, 0, 1, %0" : "=&s"(m), "=v"(r) : "v"(a), "v"(b)); \
    w[bit >> 5] |= r << (bit & 31u); ++bit;                                                                           \
    asm volatile("v_cmp_" #P "_f32_e64 %0, -%2, %3\n\tv_cndmask_b32_e64 %1, 0, 1, %0" : "=&s"(m), "=v"(r) : "v"(a), "v"(b)); \
    w[bit >> 5] |= r << (bit & 31u); ++bit;                                                                           \
    asm volatile("v_cmp_" #P "_f32_e64 %0, %2, -%3\n\tv_cndmask_b32_e64 %1, 0, 1, %0" : "=&s"(m), "=v"(r) : "v"(a), "v"(b)); \
    w[bit >> 5] |= r << (bit & 31u); ++bit;                                                                           \
    asm volatile("v_cmp_" #P "_f32_e64 %0, |%2|, |%3|\n\tv_cndmask_b32_e64 %1, 0, 1, %0" : "=&s"(m), "=v"(r) : "v"(a), "v"(b)); \
    w[bit >> 5] |= r << (bit & 31u); ++bit;
    PREDS(CMPF)
#undef CMPF
    for (uint32_t k = 0; k < 2u; ++k) out[2u * idx + k] = w[k];
}

// v_cmpx_<P>_f32_e32: 1 where the lane survives in EXEC (the destination is zeroed first, only a
// surviving lane executes the following v_mov_b32 1; EXEC is restored afterwards). One bit per P, bit index pidx. W = 1
#define PREDSX(X) X(ge, 1) X(gt, 1) X(lt, 1) X(nge, 1) X(nlt, 1)
#define CMPX_1(P)                                                                                                      \
    asm volatile("v_mov_b32 %0, 0\n\ts_mov_b32 %1, exec_lo\n\tv_cmpx_" #P "_f32_e32 %2, %3\n\tv_mov_b32 %0, 1\n\t"      \
                 "s_mov_b32 exec_lo, %1\n\ts_wait_alu depctr_sa_sdst(0)" : "=&v"(r), "=&s"(save) : "v"(a), "v"(b));
#define CMPX_0(P) r = 0;
PROBE(probe_v_cmpx) {
    RECORD
    uint32_t w = 0, bit = 0, r, save;
#define CMPX(P, E32)                                                                                                   \
    CAT(CMPX_, E32)(P)                                                                                                 \
    w |= r << bit; ++bit;
    PREDSX(CMPX)
#undef CMPX
    out[idx] = w;
}

// v_cmp_class_f32_e64 (a = value, mask b / (c & 0x3ff) / (1 << (c % 10)); VOP3 e64 plain, |a|, -a). Bit index 3 mask + form,
// masks: 0 = b, 1 = c & 0x3ff, 2 = 1 << (c % 10); forms: e64, |a|, -a. W = 1
PROBE(probe_v_class) {
    RECORD
    const uint32_t masks[3] = {b, c & 0x3ffu, 1u << (c % 10u)};
    uint32_t w = 0, bit = 0, m, r;
    for (uint32_t k = 0; k < 3u; ++k) {
        const uint32_t mk = masks[k];
        asm volatile("v_cmp_class_f32_e64 %0, %2, %3\n\tv_cndmask_b32_e64 %1, 0, 1, %0" : "=&s"(m), "=v"(r) : "v"(a), "v"(mk));
        w |= r << bit; ++bit;
        asm volatile("v_cmp_class_f32_e64 %0, |%2|, %3\n\tv_cndmask_b32_e64 %1, 0, 1, %0" : "=&s"(m), "=v"(r) : "v"(a), "v"(mk));
        w |= r << bit; ++bit;
        asm volatile("v_cmp_class_f32_e64 %0, -%2, %3\n\tv_cndmask_b32_e64 %1, 0, 1, %0" : "=&s"(m), "=v"(r) : "v"(a), "v"(mk));
        w |= r << bit; ++bit;
    }
    out[idx] = w;
}
