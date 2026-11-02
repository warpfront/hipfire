// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exact-gfx1100 MQ6G256V2 residual prefill GEMM, hipcc "pk twin" ports (PM-KRSI-1100 round r2,
// candidates R2Mq6-01-pk-w8n256 and R2Mq6-04-pk-w4n128): Y[n*M+m] += sum_k W[m,k]*X[n,k].
// Same arithmetic and bytes as `gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds`: every output runs ONE f32
// WMMA chain over ascending 16-K steps from +0.0 with A = dequantised W (sc_h * q + zp_h, a single fused
// f16 rounding, q exact) and B = X f16, then one `Y += acc`. Bit-identical to the incumbent on the card.
//
// Kernargs (A, X, Y, M, K, N) and the Y/X/W layouts are the incumbent's. K must be a multiple of 256.
//   gemm_mq6g256v2_residual_gfx1100_port_w8n256  tile 128x256, block 256, grid [ceil(M/128), ceil(N/256)]
//   gemm_mq6g256v2_residual_gfx1100_port_w4n128  tile  64x128, block 128, grid [ceil(M/64),  ceil(N/128)]
// Selected by `mq6_resid_port_gfx1100_geom` (gemm.rs): M >= 6144 -> w8n256, 2560 <= M < 6144 -> w4n128.
//
// hipcc rendering of the PM twin design (qwen4_mq6_x4_gfx11 header):
//  * a wave owns 16 rows x NT tokens (NT/16 independent chains); one decoded A fragment feeds
//    NT/16 WMMAs; WAVES waves stacked along M share one X tile;
//  * X is staged once per workgroup in LDS, 32-K chunks, double buffered, ONE barrier per chunk,
//    fed from raw buffer loads (per-lane voffset kept in registers, per-chunk soffset in an SGPR,
//    token index clamped to N-1 in the prologue, no reliance on hardware out-of-range behaviour);
//  * A is decoded straight into WMMA fragments per wave, two codes per VALU op: the 6-bit codes
//    are gathered with v_perm_b32, dropped into the mantissa of 1024+q / 64+q, v_pk_add_f16 of the
//    bias recovers q exactly and one v_pk_fma_f16 is sc*q+zp (the same single rounding as the
//    incumbent's contracted v_fma_f16); the other half-wave's 8 values come from v_permlanex16;
//  * W codes are read per lane as one b64 per 16-K tile at tile byte 4*lane_half (buffer load);
//  * the epilogue swaps accumulator halves across the half-waves and does two b128 Y read-modify-
//    writes per tile.
// Each kernel body is the card-proven candidate source verbatim (only the symbol name differs); the
// per-geometry constants live in one namespace per kernel. Static LDS 2*NT*80 B.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__gfx1100__)

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef _Float16 __attribute__((ext_vector_type(2))) half2_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef unsigned int u32;
typedef u32 __attribute__((ext_vector_type(2))) u32x2_t;
typedef u32 __attribute__((ext_vector_type(4), aligned(16))) u32x4a_t;
typedef u32 __attribute__((ext_vector_type(8), aligned(16))) u32x8a_t;

// Per-lane byte selectors: lane half lh reads codes from window bytes 2*lh .. 2*lh+5 of its 8 loaded bytes.
struct Sel {
    u32 s0, s1, s2, s3;
};

// Staged decode of one 16-K tile A fragment (7 stages, each stage is a few independent VALU ops across the four
// code pairs) so the caller can spread the work over the WMMAs of the previous K-step.
// lo/hi = this lane's tile bytes [4*lh, 4*lh+8) of the 12-byte tile payload; hdr = group header dword (sc | zp << 16).
struct Dec {
    u32 v[4];      // pair words, then packed q/f16 results
    u32 o[4];      // other half-wave's values
    u32 sc2, zp2;
    u32x8a_t f;
};
__device__ __forceinline__ void dec_s0(Dec& d, u32 lo, u32 hi, u32 hdr, const Sel& s) {
    d.v[0] = __builtin_amdgcn_perm(hi, lo, s.s0);
    d.v[1] = __builtin_amdgcn_perm(hi, lo, s.s1);
    d.v[2] = __builtin_amdgcn_perm(hi, lo, s.s2);
    d.v[3] = __builtin_amdgcn_perm(hi, lo, s.s3);
    d.sc2 = __builtin_amdgcn_perm(hdr, hdr, 0x01000100u);
    d.zp2 = __builtin_amdgcn_perm(hdr, hdr, 0x03020302u);
}
__device__ __forceinline__ void dec_s1(Dec& d) {
#pragma unroll
    for (int k = 0; k < 4; k++) d.v[k] = d.v[k] | (d.v[k] << 10);
}
__device__ __forceinline__ void dec_s2(Dec& d) {
    d.v[0] = (d.v[0] & 0x003F003Fu) | 0x64006400u;
    d.v[1] = (d.v[1] & 0x03F003F0u) | 0x54005400u;
    d.v[2] = (d.v[2] & 0x003F003Fu) | 0x64006400u;
    d.v[3] = (d.v[3] & 0x03F003F0u) | 0x54005400u;
}
__device__ __forceinline__ void dec_s3(Dec& d) {
    const half2_t b1024 = {(_Float16)1024.0f, (_Float16)1024.0f};
    const half2_t b64 = {(_Float16)64.0f, (_Float16)64.0f};
    d.v[0] = __builtin_bit_cast(u32, __builtin_bit_cast(half2_t, d.v[0]) - b1024);
    d.v[1] = __builtin_bit_cast(u32, __builtin_bit_cast(half2_t, d.v[1]) - b64);
    d.v[2] = __builtin_bit_cast(u32, __builtin_bit_cast(half2_t, d.v[2]) - b1024);
    d.v[3] = __builtin_bit_cast(u32, __builtin_bit_cast(half2_t, d.v[3]) - b64);
}
__device__ __forceinline__ void dec_s4(Dec& d) {
    const half2_t sc = __builtin_bit_cast(half2_t, d.sc2);
    const half2_t zp = __builtin_bit_cast(half2_t, d.zp2);
#pragma unroll
    for (int k = 0; k < 4; k++)
        d.v[k] = __builtin_bit_cast(u32, __builtin_elementwise_fma(sc, __builtin_bit_cast(half2_t, d.v[k]), zp));
}
__device__ __forceinline__ void dec_s5(Dec& d) {
#pragma unroll
    for (int k = 0; k < 4; k++)
        d.o[k] = (u32)__builtin_amdgcn_permlanex16((int)d.v[k], (int)d.v[k], 0x76543210u, 0xFEDCBA98u, false, false);
}
__device__ __forceinline__ void dec_s6(Dec& d, bool lhb) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
        d.f[k] = lhb ? d.o[k] : d.v[k];
        d.f[4 + k] = lhb ? d.v[k] : d.o[k];
    }
}

__device__ __forceinline__ u32x8a_t ldb(const char* p) {
    const u32x4a_t b0 = *(const u32x4a_t*)(p);
    const u32x4a_t b1 = *(const u32x4a_t*)(p + 16);
    return u32x8a_t{b0[0], b0[1], b0[2], b0[3], b1[0], b1[1], b1[2], b1[3]};
}
__device__ __forceinline__ float8_t mma(const u32x8a_t& a, const u32x8a_t& b, const float8_t& c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(half16_t, a), __builtin_bit_cast(half16_t, b), c);
}

// Order LDS only: no vmcnt wait, no cache invalidate (the in-flight next-chunk loads keep flying).
#define MQ_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
#define SB() __builtin_amdgcn_sched_barrier(0)

namespace w8n256 {
constexpr int WAVES = 8;
constexpr int NT = 256;                       // tokens per workgroup
constexpr int TILES = NT / 16;                 // independent accumulator chains per wave
constexpr int NTHREADS = 32 * WAVES;
constexpr int XPITCH = 80;                     // LDS bytes per token row (64 data + 16 pad)
constexpr int BUFB = NT * XPITCH;              // one X chunk buffer
constexpr int XPER = NT * 4 / NTHREADS;        // 16-byte X pieces per thread per 32-K chunk
constexpr int TOK_STEP = NTHREADS / 4;
constexpr int XH = (XPER >= 4) ? XPER / 2 : XPER;   // X pieces staged in the first half of a chunk (rest in the second)
constexpr int GROUP_BYTES = 8 + 32 * 6;        // 200
static_assert((NT * 4) % NTHREADS == 0, "stage split");

extern "C" __launch_bounds__(NTHREADS, 1) __global__ void
gemm_mq6g256v2_residual_gfx1100_port_w8n256(const char* __restrict__ A,
                                           const _Float16* __restrict__ X,
                                           float* __restrict__ Y, int M, int K, int N) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave_id = tid >> 5;
    const int ml = lane & 15;
    const int lh = lane >> 4;
    const bool lhb = lh != 0;
    const int row_start = blockIdx.x * (16 * WAVES) + wave_id * 16;
    const int n_start = blockIdx.y * NT;
    const int G = K / 256;
    const int nch = G * 8;

    __shared__ __align__(16) char x_lds[2 * BUFB];

    // --- W addressing: row bytes through a buffer resource, lane offset in voffset, group in soffset.
    const int safe_row = (row_start + ml < M) ? (row_start + ml) : (M - 1);
    const __amdgpu_buffer_rsrc_t wrsrc = __builtin_amdgcn_make_buffer_rsrc((void*)A, 0, 0x7FFFFFFF, 0x31004000);
    const int row_off = safe_row * G * GROUP_BYTES;
    const int wv = row_off + 8 + 4 * lh;   // tile payload b64 at group + 8 + 12*kt + 4*lh
    const int hv = row_off;                // header (hA, hB) b64

    // --- X addressing: raw buffer, tokens clamped into [0, N-1] (outputs of clamped tokens are never stored).
    const __amdgpu_buffer_rsrc_t xrsrc = __builtin_amdgcn_make_buffer_rsrc((void*)X, 0, 0x7FFFFFFF, 0x31004000);
    const int xpart = tid & 3;
    const int xtok0 = tid >> 2;
    int xv[XPER];
#pragma unroll
    for (int i = 0; i < XPER; i++) {
        const int gt = n_start + xtok0 + i * TOK_STEP;
        const int t = (gt < N) ? gt : (N - 1);
        xv[i] = (t * K) * 2 + xpart * 16;
    }
    const int xdst = xtok0 * XPITCH + xpart * 16;
    const int bbase = ml * XPITCH;

    // --- per-lane selectors (kept opaque so they stay in VGPRs).
    Sel sel;
    {
        const u32 so = (u32)(lh * 2) * 0x0101u;
        sel.s0 = 0x0C0C0100u + so;
        sel.s1 = 0x0C0C0201u + so;
        sel.s2 = 0x0C0C0403u + so;
        sel.s3 = 0x0C0C0504u + so;
        asm volatile("" : "+v"(sel.s0), "+v"(sel.s1), "+v"(sel.s2), "+v"(sel.s3));
    }

    float8_t acc[TILES];
#pragma unroll
    for (int j = 0; j < TILES; j++)
        acc[j] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    if (G > 0) {
        u32x4a_t xr[XH];
        u32x2_t w_0_0, w_0_1, w_1_0, w_1_1;   // this lane's 8 payload bytes of tile (chunk parity, tile in chunk)
        u32 hdA, hdB;                          // current group header halves
        u32x8a_t a_cur;

        // ---- prologue: X chunk 0, W chunks 0 and 1 and header of group 0, A fragment of tile 0.
        {
            u32x4a_t xp[XPER];
#pragma unroll
            for (int i = 0; i < XPER; i++) xp[i] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[i], 0, 0);
#pragma unroll
            for (int i = 0; i < XPER; i++) *(u32x4a_t*)(&x_lds[xdst + i * TOK_STEP * XPITCH]) = xp[i];
        }
        {
            const u32x2_t h = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, hv, 0, 0);
            hdA = h[0];
            hdB = h[1];
        }
        w_0_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv, 0, 0);
        w_0_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 12, 0, 0);
        w_1_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 24, 0, 0);
        w_1_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 36, 0, 0);
        {
            Dec d;
            dec_s0(d, w_0_0[0], w_0_0[1], hdA, sel);
            dec_s1(d); dec_s2(d); dec_s3(d); dec_s4(d); dec_s5(d); dec_s6(d, lhb);
            a_cur = d.f;
        }
        MQ_BARRIER();

        for (int g = 0; g < G; g++) {
            const int gn = (g + 1 < G) ? (g + 1) : g;
            u32 hnA = hdA, hnB = hdB;
            { // ---- chunk 0
            const char* xb = &x_lds[0 * BUFB + bbase];
            char* xd = &x_lds[1 * BUFB + xdst];
            int cx = g * 8 + 1; cx = (cx < nch) ? cx : (nch - 1); const int xsoff = cx * 64;
            u32x8a_t B0, B1, B2;
            Dec d1, dn;
            B0 = ldb(xb + 0);
            B1 = ldb(xb + 1280);
            B2 = ldb(xb + 2560);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[0], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[1], xsoff, 0);
            SB();
            acc[0] = mma(a_cur, B0, acc[0]);
            B0 = ldb(xb + 3840);
            SB();
            acc[1] = mma(a_cur, B1, acc[1]);
            B1 = ldb(xb + 5120);
            dec_s0(d1, w_0_1[0], w_0_1[1], hdA, sel);
            SB();
            acc[2] = mma(a_cur, B2, acc[2]);
            B2 = ldb(xb + 6400);
            SB();
            acc[3] = mma(a_cur, B0, acc[3]);
            B0 = ldb(xb + 7680);
            dec_s1(d1);
            SB();
            acc[4] = mma(a_cur, B1, acc[4]);
            B1 = ldb(xb + 8960);
            w_0_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 48, g * GROUP_BYTES, 0);
            w_0_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 60, g * GROUP_BYTES, 0);
            SB();
            acc[5] = mma(a_cur, B2, acc[5]);
            B2 = ldb(xb + 10240);
            dec_s2(d1);
            SB();
            acc[6] = mma(a_cur, B0, acc[6]);
            B0 = ldb(xb + 11520);
            SB();
            acc[7] = mma(a_cur, B1, acc[7]);
            B1 = ldb(xb + 12800);
            dec_s3(d1);
            SB();
            acc[8] = mma(a_cur, B2, acc[8]);
            B2 = ldb(xb + 14080);
            SB();
            acc[9] = mma(a_cur, B0, acc[9]);
            B0 = ldb(xb + 15360);
            dec_s4(d1);
            SB();
            acc[10] = mma(a_cur, B1, acc[10]);
            B1 = ldb(xb + 16640);
            *(u32x4a_t*)(xd + 0 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[11] = mma(a_cur, B2, acc[11]);
            B2 = ldb(xb + 17920);
            dec_s5(d1);
            *(u32x4a_t*)(xd + 1 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[12] = mma(a_cur, B0, acc[12]);
            B0 = ldb(xb + 19200);
            SB();
            acc[13] = mma(a_cur, B1, acc[13]);
            B1 = ldb(xb + 32);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[2], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[3], xsoff, 0);
            SB();
            acc[14] = mma(a_cur, B2, acc[14]);
            B2 = ldb(xb + 1312);
            dec_s6(d1, lhb);
            SB();
            acc[15] = mma(a_cur, B0, acc[15]);
            B0 = ldb(xb + 2592);
            SB();
            acc[0] = mma(d1.f, B1, acc[0]);
            B1 = ldb(xb + 3872);
            SB();
            acc[1] = mma(d1.f, B2, acc[1]);
            B2 = ldb(xb + 5152);
            dec_s0(dn, w_1_0[0], w_1_0[1], hdA, sel);
            SB();
            acc[2] = mma(d1.f, B0, acc[2]);
            B0 = ldb(xb + 6432);
            SB();
            acc[3] = mma(d1.f, B1, acc[3]);
            B1 = ldb(xb + 7712);
            dec_s1(dn);
            SB();
            acc[4] = mma(d1.f, B2, acc[4]);
            B2 = ldb(xb + 8992);
            SB();
            acc[5] = mma(d1.f, B0, acc[5]);
            B0 = ldb(xb + 10272);
            dec_s2(dn);
            SB();
            acc[6] = mma(d1.f, B1, acc[6]);
            B1 = ldb(xb + 11552);
            SB();
            acc[7] = mma(d1.f, B2, acc[7]);
            B2 = ldb(xb + 12832);
            dec_s3(dn);
            SB();
            acc[8] = mma(d1.f, B0, acc[8]);
            B0 = ldb(xb + 14112);
            SB();
            acc[9] = mma(d1.f, B1, acc[9]);
            B1 = ldb(xb + 15392);
            dec_s4(dn);
            SB();
            acc[10] = mma(d1.f, B2, acc[10]);
            B2 = ldb(xb + 16672);
            SB();
            acc[11] = mma(d1.f, B0, acc[11]);
            B0 = ldb(xb + 17952);
            dec_s5(dn);
            *(u32x4a_t*)(xd + 2 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[12] = mma(d1.f, B1, acc[12]);
            B1 = ldb(xb + 19232);
            SB();
            acc[13] = mma(d1.f, B2, acc[13]);
            *(u32x4a_t*)(xd + 3 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[14] = mma(d1.f, B0, acc[14]);
            dec_s6(dn, lhb);
            SB();
            acc[15] = mma(d1.f, B1, acc[15]);
            SB();
            MQ_BARRIER();
            a_cur = dn.f;
            }
            { // ---- chunk 1
            const char* xb = &x_lds[1 * BUFB + bbase];
            char* xd = &x_lds[0 * BUFB + xdst];
            int cx = g * 8 + 2; cx = (cx < nch) ? cx : (nch - 1); const int xsoff = cx * 64;
            u32x8a_t B0, B1, B2;
            Dec d1, dn;
            B0 = ldb(xb + 0);
            B1 = ldb(xb + 1280);
            B2 = ldb(xb + 2560);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[0], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[1], xsoff, 0);
            SB();
            acc[0] = mma(a_cur, B0, acc[0]);
            B0 = ldb(xb + 3840);
            SB();
            acc[1] = mma(a_cur, B1, acc[1]);
            B1 = ldb(xb + 5120);
            dec_s0(d1, w_1_1[0], w_1_1[1], hdA, sel);
            SB();
            acc[2] = mma(a_cur, B2, acc[2]);
            B2 = ldb(xb + 6400);
            SB();
            acc[3] = mma(a_cur, B0, acc[3]);
            B0 = ldb(xb + 7680);
            dec_s1(d1);
            SB();
            acc[4] = mma(a_cur, B1, acc[4]);
            B1 = ldb(xb + 8960);
            w_1_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 72, g * GROUP_BYTES, 0);
            w_1_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 84, g * GROUP_BYTES, 0);
            SB();
            acc[5] = mma(a_cur, B2, acc[5]);
            B2 = ldb(xb + 10240);
            dec_s2(d1);
            SB();
            acc[6] = mma(a_cur, B0, acc[6]);
            B0 = ldb(xb + 11520);
            SB();
            acc[7] = mma(a_cur, B1, acc[7]);
            B1 = ldb(xb + 12800);
            dec_s3(d1);
            SB();
            acc[8] = mma(a_cur, B2, acc[8]);
            B2 = ldb(xb + 14080);
            SB();
            acc[9] = mma(a_cur, B0, acc[9]);
            B0 = ldb(xb + 15360);
            dec_s4(d1);
            SB();
            acc[10] = mma(a_cur, B1, acc[10]);
            B1 = ldb(xb + 16640);
            *(u32x4a_t*)(xd + 0 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[11] = mma(a_cur, B2, acc[11]);
            B2 = ldb(xb + 17920);
            dec_s5(d1);
            *(u32x4a_t*)(xd + 1 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[12] = mma(a_cur, B0, acc[12]);
            B0 = ldb(xb + 19200);
            SB();
            acc[13] = mma(a_cur, B1, acc[13]);
            B1 = ldb(xb + 32);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[2], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[3], xsoff, 0);
            SB();
            acc[14] = mma(a_cur, B2, acc[14]);
            B2 = ldb(xb + 1312);
            dec_s6(d1, lhb);
            SB();
            acc[15] = mma(a_cur, B0, acc[15]);
            B0 = ldb(xb + 2592);
            SB();
            acc[0] = mma(d1.f, B1, acc[0]);
            B1 = ldb(xb + 3872);
            SB();
            acc[1] = mma(d1.f, B2, acc[1]);
            B2 = ldb(xb + 5152);
            dec_s0(dn, w_0_0[0], w_0_0[1], hdA, sel);
            SB();
            acc[2] = mma(d1.f, B0, acc[2]);
            B0 = ldb(xb + 6432);
            SB();
            acc[3] = mma(d1.f, B1, acc[3]);
            B1 = ldb(xb + 7712);
            dec_s1(dn);
            SB();
            acc[4] = mma(d1.f, B2, acc[4]);
            B2 = ldb(xb + 8992);
            SB();
            acc[5] = mma(d1.f, B0, acc[5]);
            B0 = ldb(xb + 10272);
            dec_s2(dn);
            SB();
            acc[6] = mma(d1.f, B1, acc[6]);
            B1 = ldb(xb + 11552);
            SB();
            acc[7] = mma(d1.f, B2, acc[7]);
            B2 = ldb(xb + 12832);
            dec_s3(dn);
            SB();
            acc[8] = mma(d1.f, B0, acc[8]);
            B0 = ldb(xb + 14112);
            SB();
            acc[9] = mma(d1.f, B1, acc[9]);
            B1 = ldb(xb + 15392);
            dec_s4(dn);
            SB();
            acc[10] = mma(d1.f, B2, acc[10]);
            B2 = ldb(xb + 16672);
            SB();
            acc[11] = mma(d1.f, B0, acc[11]);
            B0 = ldb(xb + 17952);
            dec_s5(dn);
            *(u32x4a_t*)(xd + 2 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[12] = mma(d1.f, B1, acc[12]);
            B1 = ldb(xb + 19232);
            SB();
            acc[13] = mma(d1.f, B2, acc[13]);
            *(u32x4a_t*)(xd + 3 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[14] = mma(d1.f, B0, acc[14]);
            dec_s6(dn, lhb);
            SB();
            acc[15] = mma(d1.f, B1, acc[15]);
            SB();
            MQ_BARRIER();
            a_cur = dn.f;
            }
            { // ---- chunk 2
            const char* xb = &x_lds[0 * BUFB + bbase];
            char* xd = &x_lds[1 * BUFB + xdst];
            int cx = g * 8 + 3; cx = (cx < nch) ? cx : (nch - 1); const int xsoff = cx * 64;
            u32x8a_t B0, B1, B2;
            Dec d1, dn;
            B0 = ldb(xb + 0);
            B1 = ldb(xb + 1280);
            B2 = ldb(xb + 2560);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[0], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[1], xsoff, 0);
            SB();
            acc[0] = mma(a_cur, B0, acc[0]);
            B0 = ldb(xb + 3840);
            SB();
            acc[1] = mma(a_cur, B1, acc[1]);
            B1 = ldb(xb + 5120);
            dec_s0(d1, w_0_1[0], w_0_1[1], hdA, sel);
            SB();
            acc[2] = mma(a_cur, B2, acc[2]);
            B2 = ldb(xb + 6400);
            SB();
            acc[3] = mma(a_cur, B0, acc[3]);
            B0 = ldb(xb + 7680);
            dec_s1(d1);
            SB();
            acc[4] = mma(a_cur, B1, acc[4]);
            B1 = ldb(xb + 8960);
            w_0_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 96, g * GROUP_BYTES, 0);
            w_0_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 108, g * GROUP_BYTES, 0);
            SB();
            acc[5] = mma(a_cur, B2, acc[5]);
            B2 = ldb(xb + 10240);
            dec_s2(d1);
            SB();
            acc[6] = mma(a_cur, B0, acc[6]);
            B0 = ldb(xb + 11520);
            SB();
            acc[7] = mma(a_cur, B1, acc[7]);
            B1 = ldb(xb + 12800);
            dec_s3(d1);
            SB();
            acc[8] = mma(a_cur, B2, acc[8]);
            B2 = ldb(xb + 14080);
            SB();
            acc[9] = mma(a_cur, B0, acc[9]);
            B0 = ldb(xb + 15360);
            dec_s4(d1);
            SB();
            acc[10] = mma(a_cur, B1, acc[10]);
            B1 = ldb(xb + 16640);
            *(u32x4a_t*)(xd + 0 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[11] = mma(a_cur, B2, acc[11]);
            B2 = ldb(xb + 17920);
            dec_s5(d1);
            *(u32x4a_t*)(xd + 1 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[12] = mma(a_cur, B0, acc[12]);
            B0 = ldb(xb + 19200);
            SB();
            acc[13] = mma(a_cur, B1, acc[13]);
            B1 = ldb(xb + 32);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[2], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[3], xsoff, 0);
            SB();
            acc[14] = mma(a_cur, B2, acc[14]);
            B2 = ldb(xb + 1312);
            dec_s6(d1, lhb);
            SB();
            acc[15] = mma(a_cur, B0, acc[15]);
            B0 = ldb(xb + 2592);
            SB();
            acc[0] = mma(d1.f, B1, acc[0]);
            B1 = ldb(xb + 3872);
            SB();
            acc[1] = mma(d1.f, B2, acc[1]);
            B2 = ldb(xb + 5152);
            dec_s0(dn, w_1_0[0], w_1_0[1], hdA, sel);
            SB();
            acc[2] = mma(d1.f, B0, acc[2]);
            B0 = ldb(xb + 6432);
            SB();
            acc[3] = mma(d1.f, B1, acc[3]);
            B1 = ldb(xb + 7712);
            dec_s1(dn);
            SB();
            acc[4] = mma(d1.f, B2, acc[4]);
            B2 = ldb(xb + 8992);
            SB();
            acc[5] = mma(d1.f, B0, acc[5]);
            B0 = ldb(xb + 10272);
            dec_s2(dn);
            SB();
            acc[6] = mma(d1.f, B1, acc[6]);
            B1 = ldb(xb + 11552);
            SB();
            acc[7] = mma(d1.f, B2, acc[7]);
            B2 = ldb(xb + 12832);
            dec_s3(dn);
            SB();
            acc[8] = mma(d1.f, B0, acc[8]);
            B0 = ldb(xb + 14112);
            SB();
            acc[9] = mma(d1.f, B1, acc[9]);
            B1 = ldb(xb + 15392);
            dec_s4(dn);
            SB();
            acc[10] = mma(d1.f, B2, acc[10]);
            B2 = ldb(xb + 16672);
            SB();
            acc[11] = mma(d1.f, B0, acc[11]);
            B0 = ldb(xb + 17952);
            dec_s5(dn);
            *(u32x4a_t*)(xd + 2 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[12] = mma(d1.f, B1, acc[12]);
            B1 = ldb(xb + 19232);
            SB();
            acc[13] = mma(d1.f, B2, acc[13]);
            *(u32x4a_t*)(xd + 3 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[14] = mma(d1.f, B0, acc[14]);
            dec_s6(dn, lhb);
            SB();
            acc[15] = mma(d1.f, B1, acc[15]);
            SB();
            MQ_BARRIER();
            a_cur = dn.f;
            }
            { // ---- chunk 3
            const char* xb = &x_lds[1 * BUFB + bbase];
            char* xd = &x_lds[0 * BUFB + xdst];
            int cx = g * 8 + 4; cx = (cx < nch) ? cx : (nch - 1); const int xsoff = cx * 64;
            u32x8a_t B0, B1, B2;
            Dec d1, dn;
            B0 = ldb(xb + 0);
            B1 = ldb(xb + 1280);
            B2 = ldb(xb + 2560);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[0], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[1], xsoff, 0);
            SB();
            acc[0] = mma(a_cur, B0, acc[0]);
            B0 = ldb(xb + 3840);
            SB();
            acc[1] = mma(a_cur, B1, acc[1]);
            B1 = ldb(xb + 5120);
            dec_s0(d1, w_1_1[0], w_1_1[1], hdA, sel);
            SB();
            acc[2] = mma(a_cur, B2, acc[2]);
            B2 = ldb(xb + 6400);
            SB();
            acc[3] = mma(a_cur, B0, acc[3]);
            B0 = ldb(xb + 7680);
            dec_s1(d1);
            SB();
            acc[4] = mma(a_cur, B1, acc[4]);
            B1 = ldb(xb + 8960);
            w_1_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 120, g * GROUP_BYTES, 0);
            w_1_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 132, g * GROUP_BYTES, 0);
            SB();
            acc[5] = mma(a_cur, B2, acc[5]);
            B2 = ldb(xb + 10240);
            dec_s2(d1);
            SB();
            acc[6] = mma(a_cur, B0, acc[6]);
            B0 = ldb(xb + 11520);
            SB();
            acc[7] = mma(a_cur, B1, acc[7]);
            B1 = ldb(xb + 12800);
            dec_s3(d1);
            SB();
            acc[8] = mma(a_cur, B2, acc[8]);
            B2 = ldb(xb + 14080);
            SB();
            acc[9] = mma(a_cur, B0, acc[9]);
            B0 = ldb(xb + 15360);
            dec_s4(d1);
            SB();
            acc[10] = mma(a_cur, B1, acc[10]);
            B1 = ldb(xb + 16640);
            *(u32x4a_t*)(xd + 0 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[11] = mma(a_cur, B2, acc[11]);
            B2 = ldb(xb + 17920);
            dec_s5(d1);
            *(u32x4a_t*)(xd + 1 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[12] = mma(a_cur, B0, acc[12]);
            B0 = ldb(xb + 19200);
            SB();
            acc[13] = mma(a_cur, B1, acc[13]);
            B1 = ldb(xb + 32);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[2], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[3], xsoff, 0);
            SB();
            acc[14] = mma(a_cur, B2, acc[14]);
            B2 = ldb(xb + 1312);
            dec_s6(d1, lhb);
            SB();
            acc[15] = mma(a_cur, B0, acc[15]);
            B0 = ldb(xb + 2592);
            SB();
            acc[0] = mma(d1.f, B1, acc[0]);
            B1 = ldb(xb + 3872);
            SB();
            acc[1] = mma(d1.f, B2, acc[1]);
            B2 = ldb(xb + 5152);
            dec_s0(dn, w_0_0[0], w_0_0[1], hdB, sel);
            SB();
            acc[2] = mma(d1.f, B0, acc[2]);
            B0 = ldb(xb + 6432);
            SB();
            acc[3] = mma(d1.f, B1, acc[3]);
            B1 = ldb(xb + 7712);
            dec_s1(dn);
            SB();
            acc[4] = mma(d1.f, B2, acc[4]);
            B2 = ldb(xb + 8992);
            SB();
            acc[5] = mma(d1.f, B0, acc[5]);
            B0 = ldb(xb + 10272);
            dec_s2(dn);
            SB();
            acc[6] = mma(d1.f, B1, acc[6]);
            B1 = ldb(xb + 11552);
            SB();
            acc[7] = mma(d1.f, B2, acc[7]);
            B2 = ldb(xb + 12832);
            dec_s3(dn);
            SB();
            acc[8] = mma(d1.f, B0, acc[8]);
            B0 = ldb(xb + 14112);
            SB();
            acc[9] = mma(d1.f, B1, acc[9]);
            B1 = ldb(xb + 15392);
            dec_s4(dn);
            SB();
            acc[10] = mma(d1.f, B2, acc[10]);
            B2 = ldb(xb + 16672);
            SB();
            acc[11] = mma(d1.f, B0, acc[11]);
            B0 = ldb(xb + 17952);
            dec_s5(dn);
            *(u32x4a_t*)(xd + 2 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[12] = mma(d1.f, B1, acc[12]);
            B1 = ldb(xb + 19232);
            SB();
            acc[13] = mma(d1.f, B2, acc[13]);
            *(u32x4a_t*)(xd + 3 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[14] = mma(d1.f, B0, acc[14]);
            dec_s6(dn, lhb);
            SB();
            acc[15] = mma(d1.f, B1, acc[15]);
            SB();
            MQ_BARRIER();
            a_cur = dn.f;
            }
            { // ---- chunk 4
            const char* xb = &x_lds[0 * BUFB + bbase];
            char* xd = &x_lds[1 * BUFB + xdst];
            int cx = g * 8 + 5; cx = (cx < nch) ? cx : (nch - 1); const int xsoff = cx * 64;
            u32x8a_t B0, B1, B2;
            Dec d1, dn;
            B0 = ldb(xb + 0);
            B1 = ldb(xb + 1280);
            B2 = ldb(xb + 2560);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[0], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[1], xsoff, 0);
            SB();
            acc[0] = mma(a_cur, B0, acc[0]);
            B0 = ldb(xb + 3840);
            SB();
            acc[1] = mma(a_cur, B1, acc[1]);
            B1 = ldb(xb + 5120);
            dec_s0(d1, w_0_1[0], w_0_1[1], hdB, sel);
            SB();
            acc[2] = mma(a_cur, B2, acc[2]);
            B2 = ldb(xb + 6400);
            SB();
            acc[3] = mma(a_cur, B0, acc[3]);
            B0 = ldb(xb + 7680);
            dec_s1(d1);
            SB();
            acc[4] = mma(a_cur, B1, acc[4]);
            B1 = ldb(xb + 8960);
            w_0_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 144, g * GROUP_BYTES, 0);
            w_0_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 156, g * GROUP_BYTES, 0);
            SB();
            acc[5] = mma(a_cur, B2, acc[5]);
            B2 = ldb(xb + 10240);
            dec_s2(d1);
            SB();
            acc[6] = mma(a_cur, B0, acc[6]);
            B0 = ldb(xb + 11520);
            SB();
            acc[7] = mma(a_cur, B1, acc[7]);
            B1 = ldb(xb + 12800);
            dec_s3(d1);
            SB();
            acc[8] = mma(a_cur, B2, acc[8]);
            B2 = ldb(xb + 14080);
            SB();
            acc[9] = mma(a_cur, B0, acc[9]);
            B0 = ldb(xb + 15360);
            dec_s4(d1);
            SB();
            acc[10] = mma(a_cur, B1, acc[10]);
            B1 = ldb(xb + 16640);
            *(u32x4a_t*)(xd + 0 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[11] = mma(a_cur, B2, acc[11]);
            B2 = ldb(xb + 17920);
            dec_s5(d1);
            *(u32x4a_t*)(xd + 1 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[12] = mma(a_cur, B0, acc[12]);
            B0 = ldb(xb + 19200);
            SB();
            acc[13] = mma(a_cur, B1, acc[13]);
            B1 = ldb(xb + 32);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[2], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[3], xsoff, 0);
            SB();
            acc[14] = mma(a_cur, B2, acc[14]);
            B2 = ldb(xb + 1312);
            dec_s6(d1, lhb);
            SB();
            acc[15] = mma(a_cur, B0, acc[15]);
            B0 = ldb(xb + 2592);
            SB();
            acc[0] = mma(d1.f, B1, acc[0]);
            B1 = ldb(xb + 3872);
            SB();
            acc[1] = mma(d1.f, B2, acc[1]);
            B2 = ldb(xb + 5152);
            dec_s0(dn, w_1_0[0], w_1_0[1], hdB, sel);
            SB();
            acc[2] = mma(d1.f, B0, acc[2]);
            B0 = ldb(xb + 6432);
            SB();
            acc[3] = mma(d1.f, B1, acc[3]);
            B1 = ldb(xb + 7712);
            dec_s1(dn);
            SB();
            acc[4] = mma(d1.f, B2, acc[4]);
            B2 = ldb(xb + 8992);
            SB();
            acc[5] = mma(d1.f, B0, acc[5]);
            B0 = ldb(xb + 10272);
            dec_s2(dn);
            SB();
            acc[6] = mma(d1.f, B1, acc[6]);
            B1 = ldb(xb + 11552);
            SB();
            acc[7] = mma(d1.f, B2, acc[7]);
            B2 = ldb(xb + 12832);
            dec_s3(dn);
            SB();
            acc[8] = mma(d1.f, B0, acc[8]);
            B0 = ldb(xb + 14112);
            SB();
            acc[9] = mma(d1.f, B1, acc[9]);
            B1 = ldb(xb + 15392);
            dec_s4(dn);
            SB();
            acc[10] = mma(d1.f, B2, acc[10]);
            B2 = ldb(xb + 16672);
            SB();
            acc[11] = mma(d1.f, B0, acc[11]);
            B0 = ldb(xb + 17952);
            dec_s5(dn);
            *(u32x4a_t*)(xd + 2 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[12] = mma(d1.f, B1, acc[12]);
            B1 = ldb(xb + 19232);
            SB();
            acc[13] = mma(d1.f, B2, acc[13]);
            *(u32x4a_t*)(xd + 3 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[14] = mma(d1.f, B0, acc[14]);
            dec_s6(dn, lhb);
            SB();
            acc[15] = mma(d1.f, B1, acc[15]);
            SB();
            MQ_BARRIER();
            a_cur = dn.f;
            }
            { // ---- chunk 5
            const char* xb = &x_lds[1 * BUFB + bbase];
            char* xd = &x_lds[0 * BUFB + xdst];
            int cx = g * 8 + 6; cx = (cx < nch) ? cx : (nch - 1); const int xsoff = cx * 64;
            u32x8a_t B0, B1, B2;
            Dec d1, dn;
            B0 = ldb(xb + 0);
            B1 = ldb(xb + 1280);
            B2 = ldb(xb + 2560);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[0], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[1], xsoff, 0);
            SB();
            acc[0] = mma(a_cur, B0, acc[0]);
            B0 = ldb(xb + 3840);
            SB();
            acc[1] = mma(a_cur, B1, acc[1]);
            B1 = ldb(xb + 5120);
            dec_s0(d1, w_1_1[0], w_1_1[1], hdB, sel);
            SB();
            acc[2] = mma(a_cur, B2, acc[2]);
            B2 = ldb(xb + 6400);
            SB();
            acc[3] = mma(a_cur, B0, acc[3]);
            B0 = ldb(xb + 7680);
            dec_s1(d1);
            SB();
            acc[4] = mma(a_cur, B1, acc[4]);
            B1 = ldb(xb + 8960);
            w_1_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 168, g * GROUP_BYTES, 0);
            w_1_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 180, g * GROUP_BYTES, 0);
            SB();
            acc[5] = mma(a_cur, B2, acc[5]);
            B2 = ldb(xb + 10240);
            dec_s2(d1);
            SB();
            acc[6] = mma(a_cur, B0, acc[6]);
            B0 = ldb(xb + 11520);
            SB();
            acc[7] = mma(a_cur, B1, acc[7]);
            B1 = ldb(xb + 12800);
            dec_s3(d1);
            SB();
            acc[8] = mma(a_cur, B2, acc[8]);
            B2 = ldb(xb + 14080);
            SB();
            acc[9] = mma(a_cur, B0, acc[9]);
            B0 = ldb(xb + 15360);
            dec_s4(d1);
            SB();
            acc[10] = mma(a_cur, B1, acc[10]);
            B1 = ldb(xb + 16640);
            *(u32x4a_t*)(xd + 0 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[11] = mma(a_cur, B2, acc[11]);
            B2 = ldb(xb + 17920);
            dec_s5(d1);
            *(u32x4a_t*)(xd + 1 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[12] = mma(a_cur, B0, acc[12]);
            B0 = ldb(xb + 19200);
            SB();
            acc[13] = mma(a_cur, B1, acc[13]);
            B1 = ldb(xb + 32);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[2], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[3], xsoff, 0);
            SB();
            acc[14] = mma(a_cur, B2, acc[14]);
            B2 = ldb(xb + 1312);
            dec_s6(d1, lhb);
            SB();
            acc[15] = mma(a_cur, B0, acc[15]);
            B0 = ldb(xb + 2592);
            SB();
            acc[0] = mma(d1.f, B1, acc[0]);
            B1 = ldb(xb + 3872);
            SB();
            acc[1] = mma(d1.f, B2, acc[1]);
            B2 = ldb(xb + 5152);
            dec_s0(dn, w_0_0[0], w_0_0[1], hdB, sel);
            SB();
            acc[2] = mma(d1.f, B0, acc[2]);
            B0 = ldb(xb + 6432);
            SB();
            acc[3] = mma(d1.f, B1, acc[3]);
            B1 = ldb(xb + 7712);
            dec_s1(dn);
            SB();
            acc[4] = mma(d1.f, B2, acc[4]);
            B2 = ldb(xb + 8992);
            SB();
            acc[5] = mma(d1.f, B0, acc[5]);
            B0 = ldb(xb + 10272);
            dec_s2(dn);
            SB();
            acc[6] = mma(d1.f, B1, acc[6]);
            B1 = ldb(xb + 11552);
            SB();
            acc[7] = mma(d1.f, B2, acc[7]);
            B2 = ldb(xb + 12832);
            dec_s3(dn);
            SB();
            acc[8] = mma(d1.f, B0, acc[8]);
            B0 = ldb(xb + 14112);
            SB();
            acc[9] = mma(d1.f, B1, acc[9]);
            B1 = ldb(xb + 15392);
            dec_s4(dn);
            SB();
            acc[10] = mma(d1.f, B2, acc[10]);
            B2 = ldb(xb + 16672);
            SB();
            acc[11] = mma(d1.f, B0, acc[11]);
            B0 = ldb(xb + 17952);
            dec_s5(dn);
            *(u32x4a_t*)(xd + 2 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[12] = mma(d1.f, B1, acc[12]);
            B1 = ldb(xb + 19232);
            SB();
            acc[13] = mma(d1.f, B2, acc[13]);
            *(u32x4a_t*)(xd + 3 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[14] = mma(d1.f, B0, acc[14]);
            dec_s6(dn, lhb);
            SB();
            acc[15] = mma(d1.f, B1, acc[15]);
            SB();
            MQ_BARRIER();
            a_cur = dn.f;
            }
            { // ---- chunk 6
            const char* xb = &x_lds[0 * BUFB + bbase];
            char* xd = &x_lds[1 * BUFB + xdst];
            int cx = g * 8 + 7; cx = (cx < nch) ? cx : (nch - 1); const int xsoff = cx * 64;
            u32x8a_t B0, B1, B2;
            Dec d1, dn;
            B0 = ldb(xb + 0);
            B1 = ldb(xb + 1280);
            B2 = ldb(xb + 2560);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[0], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[1], xsoff, 0);
            SB();
            acc[0] = mma(a_cur, B0, acc[0]);
            B0 = ldb(xb + 3840);
            SB();
            acc[1] = mma(a_cur, B1, acc[1]);
            B1 = ldb(xb + 5120);
            dec_s0(d1, w_0_1[0], w_0_1[1], hdB, sel);
            SB();
            acc[2] = mma(a_cur, B2, acc[2]);
            B2 = ldb(xb + 6400);
            SB();
            acc[3] = mma(a_cur, B0, acc[3]);
            B0 = ldb(xb + 7680);
            dec_s1(d1);
            SB();
            acc[4] = mma(a_cur, B1, acc[4]);
            B1 = ldb(xb + 8960);
            w_0_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 0, gn * GROUP_BYTES, 0);
            w_0_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 12, gn * GROUP_BYTES, 0);
            { const u32x2_t h = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, hv, gn * GROUP_BYTES, 0); hnA = h[0]; hnB = h[1]; }
            SB();
            acc[5] = mma(a_cur, B2, acc[5]);
            B2 = ldb(xb + 10240);
            dec_s2(d1);
            SB();
            acc[6] = mma(a_cur, B0, acc[6]);
            B0 = ldb(xb + 11520);
            SB();
            acc[7] = mma(a_cur, B1, acc[7]);
            B1 = ldb(xb + 12800);
            dec_s3(d1);
            SB();
            acc[8] = mma(a_cur, B2, acc[8]);
            B2 = ldb(xb + 14080);
            SB();
            acc[9] = mma(a_cur, B0, acc[9]);
            B0 = ldb(xb + 15360);
            dec_s4(d1);
            SB();
            acc[10] = mma(a_cur, B1, acc[10]);
            B1 = ldb(xb + 16640);
            *(u32x4a_t*)(xd + 0 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[11] = mma(a_cur, B2, acc[11]);
            B2 = ldb(xb + 17920);
            dec_s5(d1);
            *(u32x4a_t*)(xd + 1 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[12] = mma(a_cur, B0, acc[12]);
            B0 = ldb(xb + 19200);
            SB();
            acc[13] = mma(a_cur, B1, acc[13]);
            B1 = ldb(xb + 32);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[2], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[3], xsoff, 0);
            SB();
            acc[14] = mma(a_cur, B2, acc[14]);
            B2 = ldb(xb + 1312);
            dec_s6(d1, lhb);
            SB();
            acc[15] = mma(a_cur, B0, acc[15]);
            B0 = ldb(xb + 2592);
            SB();
            acc[0] = mma(d1.f, B1, acc[0]);
            B1 = ldb(xb + 3872);
            SB();
            acc[1] = mma(d1.f, B2, acc[1]);
            B2 = ldb(xb + 5152);
            dec_s0(dn, w_1_0[0], w_1_0[1], hdB, sel);
            SB();
            acc[2] = mma(d1.f, B0, acc[2]);
            B0 = ldb(xb + 6432);
            SB();
            acc[3] = mma(d1.f, B1, acc[3]);
            B1 = ldb(xb + 7712);
            dec_s1(dn);
            SB();
            acc[4] = mma(d1.f, B2, acc[4]);
            B2 = ldb(xb + 8992);
            SB();
            acc[5] = mma(d1.f, B0, acc[5]);
            B0 = ldb(xb + 10272);
            dec_s2(dn);
            SB();
            acc[6] = mma(d1.f, B1, acc[6]);
            B1 = ldb(xb + 11552);
            SB();
            acc[7] = mma(d1.f, B2, acc[7]);
            B2 = ldb(xb + 12832);
            dec_s3(dn);
            SB();
            acc[8] = mma(d1.f, B0, acc[8]);
            B0 = ldb(xb + 14112);
            SB();
            acc[9] = mma(d1.f, B1, acc[9]);
            B1 = ldb(xb + 15392);
            dec_s4(dn);
            SB();
            acc[10] = mma(d1.f, B2, acc[10]);
            B2 = ldb(xb + 16672);
            SB();
            acc[11] = mma(d1.f, B0, acc[11]);
            B0 = ldb(xb + 17952);
            dec_s5(dn);
            *(u32x4a_t*)(xd + 2 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[12] = mma(d1.f, B1, acc[12]);
            B1 = ldb(xb + 19232);
            SB();
            acc[13] = mma(d1.f, B2, acc[13]);
            *(u32x4a_t*)(xd + 3 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[14] = mma(d1.f, B0, acc[14]);
            dec_s6(dn, lhb);
            SB();
            acc[15] = mma(d1.f, B1, acc[15]);
            SB();
            MQ_BARRIER();
            a_cur = dn.f;
            }
            { // ---- chunk 7
            const char* xb = &x_lds[1 * BUFB + bbase];
            char* xd = &x_lds[0 * BUFB + xdst];
            int cx = g * 8 + 8; cx = (cx < nch) ? cx : (nch - 1); const int xsoff = cx * 64;
            u32x8a_t B0, B1, B2;
            Dec d1, dn;
            B0 = ldb(xb + 0);
            B1 = ldb(xb + 1280);
            B2 = ldb(xb + 2560);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[0], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[1], xsoff, 0);
            SB();
            acc[0] = mma(a_cur, B0, acc[0]);
            B0 = ldb(xb + 3840);
            SB();
            acc[1] = mma(a_cur, B1, acc[1]);
            B1 = ldb(xb + 5120);
            dec_s0(d1, w_1_1[0], w_1_1[1], hdB, sel);
            SB();
            acc[2] = mma(a_cur, B2, acc[2]);
            B2 = ldb(xb + 6400);
            SB();
            acc[3] = mma(a_cur, B0, acc[3]);
            B0 = ldb(xb + 7680);
            dec_s1(d1);
            SB();
            acc[4] = mma(a_cur, B1, acc[4]);
            B1 = ldb(xb + 8960);
            w_1_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 24, gn * GROUP_BYTES, 0);
            w_1_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 36, gn * GROUP_BYTES, 0);
            SB();
            acc[5] = mma(a_cur, B2, acc[5]);
            B2 = ldb(xb + 10240);
            dec_s2(d1);
            SB();
            acc[6] = mma(a_cur, B0, acc[6]);
            B0 = ldb(xb + 11520);
            SB();
            acc[7] = mma(a_cur, B1, acc[7]);
            B1 = ldb(xb + 12800);
            dec_s3(d1);
            SB();
            acc[8] = mma(a_cur, B2, acc[8]);
            B2 = ldb(xb + 14080);
            SB();
            acc[9] = mma(a_cur, B0, acc[9]);
            B0 = ldb(xb + 15360);
            dec_s4(d1);
            SB();
            acc[10] = mma(a_cur, B1, acc[10]);
            B1 = ldb(xb + 16640);
            *(u32x4a_t*)(xd + 0 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[11] = mma(a_cur, B2, acc[11]);
            B2 = ldb(xb + 17920);
            dec_s5(d1);
            *(u32x4a_t*)(xd + 1 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[12] = mma(a_cur, B0, acc[12]);
            B0 = ldb(xb + 19200);
            SB();
            acc[13] = mma(a_cur, B1, acc[13]);
            B1 = ldb(xb + 32);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[2], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[3], xsoff, 0);
            SB();
            acc[14] = mma(a_cur, B2, acc[14]);
            B2 = ldb(xb + 1312);
            dec_s6(d1, lhb);
            SB();
            acc[15] = mma(a_cur, B0, acc[15]);
            B0 = ldb(xb + 2592);
            SB();
            acc[0] = mma(d1.f, B1, acc[0]);
            B1 = ldb(xb + 3872);
            SB();
            acc[1] = mma(d1.f, B2, acc[1]);
            B2 = ldb(xb + 5152);
            dec_s0(dn, w_0_0[0], w_0_0[1], hnA, sel);
            SB();
            acc[2] = mma(d1.f, B0, acc[2]);
            B0 = ldb(xb + 6432);
            SB();
            acc[3] = mma(d1.f, B1, acc[3]);
            B1 = ldb(xb + 7712);
            dec_s1(dn);
            SB();
            acc[4] = mma(d1.f, B2, acc[4]);
            B2 = ldb(xb + 8992);
            SB();
            acc[5] = mma(d1.f, B0, acc[5]);
            B0 = ldb(xb + 10272);
            dec_s2(dn);
            SB();
            acc[6] = mma(d1.f, B1, acc[6]);
            B1 = ldb(xb + 11552);
            SB();
            acc[7] = mma(d1.f, B2, acc[7]);
            B2 = ldb(xb + 12832);
            dec_s3(dn);
            SB();
            acc[8] = mma(d1.f, B0, acc[8]);
            B0 = ldb(xb + 14112);
            SB();
            acc[9] = mma(d1.f, B1, acc[9]);
            B1 = ldb(xb + 15392);
            dec_s4(dn);
            SB();
            acc[10] = mma(d1.f, B2, acc[10]);
            B2 = ldb(xb + 16672);
            SB();
            acc[11] = mma(d1.f, B0, acc[11]);
            B0 = ldb(xb + 17952);
            dec_s5(dn);
            *(u32x4a_t*)(xd + 2 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[12] = mma(d1.f, B1, acc[12]);
            B1 = ldb(xb + 19232);
            SB();
            acc[13] = mma(d1.f, B2, acc[13]);
            *(u32x4a_t*)(xd + 3 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[14] = mma(d1.f, B0, acc[14]);
            dec_s6(dn, lhb);
            SB();
            acc[15] = mma(d1.f, B1, acc[15]);
            SB();
            MQ_BARRIER();
            a_cur = dn.f;
            }
            hdA = hnA;
            hdB = hnB;
        }
    }

    // ---- epilogue. RDNA3 wave32 WMMA: acc[j][i] = C[2*i + lh][ml]. One cross-half swap gives lane half 0
    // rows 0..7 and half 1 rows 8..15 of column ml (pure bit moves); Y += acc per element.
#pragma unroll
    for (int j = 0; j < TILES; j++) {
        float r[8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float lo_v = acc[j][i], hi_v = acc[j][4 + i];
            asm("" : "+v"(lo_v), "+v"(hi_v));   // keep the select a v_cndmask (no dynamic extract)
            const float give = lhb ? lo_v : hi_v;
            const float got = __int_as_float(__builtin_amdgcn_permlanex16(
                __float_as_int(give), __float_as_int(give), 0x76543210u, 0xFEDCBA98u, false, false));
            const float own = lhb ? hi_v : lo_v;
            r[2 * i] = lhb ? got : own;
            r[2 * i + 1] = lhb ? own : got;
        }
        const int out_col = n_start + j * 16 + ml;
        const int row0 = row_start + 8 * lh;
        if (out_col < N && row0 < M) {
            float* out = Y + (long long)out_col * M + row0;
            if (row0 + 8 <= M && (M & 3) == 0) {
                float4 o0 = *(const float4*)out;
                float4 o1 = *(const float4*)(out + 4);
                o0.x += r[0]; o0.y += r[1]; o0.z += r[2]; o0.w += r[3];
                o1.x += r[4]; o1.y += r[5]; o1.z += r[6]; o1.w += r[7];
                *(float4*)out = o0;
                *(float4*)(out + 4) = o1;
            } else {
#pragma unroll
                for (int i = 0; i < 8; i++)
                    if (row0 + i < M) out[i] += r[i];
            }
        }
    }
}
} // namespace w8n256

namespace w4n128 {
constexpr int WAVES = 4;
constexpr int NT = 128;                       // tokens per workgroup
constexpr int TILES = NT / 16;                 // independent accumulator chains per wave
constexpr int NTHREADS = 32 * WAVES;
constexpr int XPITCH = 80;                     // LDS bytes per token row (64 data + 16 pad)
constexpr int BUFB = NT * XPITCH;              // one X chunk buffer
constexpr int XPER = NT * 4 / NTHREADS;        // 16-byte X pieces per thread per 32-K chunk
constexpr int TOK_STEP = NTHREADS / 4;
constexpr int XH = (XPER >= 4) ? XPER / 2 : XPER;   // X pieces staged in the first half of a chunk (rest in the second)
constexpr int GROUP_BYTES = 8 + 32 * 6;        // 200
static_assert((NT * 4) % NTHREADS == 0, "stage split");

extern "C" __launch_bounds__(NTHREADS, 1) __global__ void
gemm_mq6g256v2_residual_gfx1100_port_w4n128(const char* __restrict__ A,
                                           const _Float16* __restrict__ X,
                                           float* __restrict__ Y, int M, int K, int N) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave_id = tid >> 5;
    const int ml = lane & 15;
    const int lh = lane >> 4;
    const bool lhb = lh != 0;
    const int row_start = blockIdx.x * (16 * WAVES) + wave_id * 16;
    const int n_start = blockIdx.y * NT;
    const int G = K / 256;
    const int nch = G * 8;

    __shared__ __align__(16) char x_lds[2 * BUFB];

    // --- W addressing: row bytes through a buffer resource, lane offset in voffset, group in soffset.
    const int safe_row = (row_start + ml < M) ? (row_start + ml) : (M - 1);
    const __amdgpu_buffer_rsrc_t wrsrc = __builtin_amdgcn_make_buffer_rsrc((void*)A, 0, 0x7FFFFFFF, 0x31004000);
    const int row_off = safe_row * G * GROUP_BYTES;
    const int wv = row_off + 8 + 4 * lh;   // tile payload b64 at group + 8 + 12*kt + 4*lh
    const int hv = row_off;                // header (hA, hB) b64

    // --- X addressing: raw buffer, tokens clamped into [0, N-1] (outputs of clamped tokens are never stored).
    const __amdgpu_buffer_rsrc_t xrsrc = __builtin_amdgcn_make_buffer_rsrc((void*)X, 0, 0x7FFFFFFF, 0x31004000);
    const int xpart = tid & 3;
    const int xtok0 = tid >> 2;
    int xv[XPER];
#pragma unroll
    for (int i = 0; i < XPER; i++) {
        const int gt = n_start + xtok0 + i * TOK_STEP;
        const int t = (gt < N) ? gt : (N - 1);
        xv[i] = (t * K) * 2 + xpart * 16;
    }
    const int xdst = xtok0 * XPITCH + xpart * 16;
    const int bbase = ml * XPITCH;

    // --- per-lane selectors (kept opaque so they stay in VGPRs).
    Sel sel;
    {
        const u32 so = (u32)(lh * 2) * 0x0101u;
        sel.s0 = 0x0C0C0100u + so;
        sel.s1 = 0x0C0C0201u + so;
        sel.s2 = 0x0C0C0403u + so;
        sel.s3 = 0x0C0C0504u + so;
        asm volatile("" : "+v"(sel.s0), "+v"(sel.s1), "+v"(sel.s2), "+v"(sel.s3));
    }

    float8_t acc[TILES];
#pragma unroll
    for (int j = 0; j < TILES; j++)
        acc[j] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    if (G > 0) {
        u32x4a_t xr[XH];
        u32x2_t w_0_0, w_0_1, w_1_0, w_1_1;   // this lane's 8 payload bytes of tile (chunk parity, tile in chunk)
        u32 hdA, hdB;                          // current group header halves
        u32x8a_t a_cur;

        // ---- prologue: X chunk 0, W chunks 0 and 1 and header of group 0, A fragment of tile 0.
        {
            u32x4a_t xp[XPER];
#pragma unroll
            for (int i = 0; i < XPER; i++) xp[i] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[i], 0, 0);
#pragma unroll
            for (int i = 0; i < XPER; i++) *(u32x4a_t*)(&x_lds[xdst + i * TOK_STEP * XPITCH]) = xp[i];
        }
        {
            const u32x2_t h = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, hv, 0, 0);
            hdA = h[0];
            hdB = h[1];
        }
        w_0_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv, 0, 0);
        w_0_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 12, 0, 0);
        w_1_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 24, 0, 0);
        w_1_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 36, 0, 0);
        {
            Dec d;
            dec_s0(d, w_0_0[0], w_0_0[1], hdA, sel);
            dec_s1(d); dec_s2(d); dec_s3(d); dec_s4(d); dec_s5(d); dec_s6(d, lhb);
            a_cur = d.f;
        }
        MQ_BARRIER();

        for (int g = 0; g < G; g++) {
            const int gn = (g + 1 < G) ? (g + 1) : g;
            u32 hnA = hdA, hnB = hdB;
            { // ---- chunk 0
            const char* xb = &x_lds[0 * BUFB + bbase];
            char* xd = &x_lds[1 * BUFB + xdst];
            int cx = g * 8 + 1; cx = (cx < nch) ? cx : (nch - 1); const int xsoff = cx * 64;
            u32x8a_t B0, B1, B2;
            Dec d1, dn;
            B0 = ldb(xb + 0);
            B1 = ldb(xb + 1280);
            B2 = ldb(xb + 2560);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[0], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[1], xsoff, 0);
            SB();
            acc[0] = mma(a_cur, B0, acc[0]);
            B0 = ldb(xb + 3840);
            SB();
            acc[1] = mma(a_cur, B1, acc[1]);
            B1 = ldb(xb + 5120);
            dec_s0(d1, w_0_1[0], w_0_1[1], hdA, sel);
            dec_s1(d1);
            SB();
            acc[2] = mma(a_cur, B2, acc[2]);
            B2 = ldb(xb + 6400);
            dec_s2(d1);
            SB();
            acc[3] = mma(a_cur, B0, acc[3]);
            B0 = ldb(xb + 7680);
            dec_s3(d1);
            SB();
            acc[4] = mma(a_cur, B1, acc[4]);
            B1 = ldb(xb + 8960);
            dec_s4(d1);
            w_0_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 48, g * GROUP_BYTES, 0);
            w_0_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 60, g * GROUP_BYTES, 0);
            SB();
            acc[5] = mma(a_cur, B2, acc[5]);
            B2 = ldb(xb + 32);
            dec_s5(d1);
            *(u32x4a_t*)(xd + 0 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[6] = mma(a_cur, B0, acc[6]);
            B0 = ldb(xb + 1312);
            dec_s6(d1, lhb);
            *(u32x4a_t*)(xd + 1 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[7] = mma(a_cur, B1, acc[7]);
            B1 = ldb(xb + 2592);
            SB();
            acc[0] = mma(d1.f, B2, acc[0]);
            B2 = ldb(xb + 3872);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[2], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[3], xsoff, 0);
            SB();
            acc[1] = mma(d1.f, B0, acc[1]);
            B0 = ldb(xb + 5152);
            dec_s0(dn, w_1_0[0], w_1_0[1], hdA, sel);
            dec_s1(dn);
            SB();
            acc[2] = mma(d1.f, B1, acc[2]);
            B1 = ldb(xb + 6432);
            dec_s2(dn);
            SB();
            acc[3] = mma(d1.f, B2, acc[3]);
            B2 = ldb(xb + 7712);
            dec_s3(dn);
            *(u32x4a_t*)(xd + 2 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[4] = mma(d1.f, B0, acc[4]);
            B0 = ldb(xb + 8992);
            dec_s4(dn);
            SB();
            acc[5] = mma(d1.f, B1, acc[5]);
            dec_s5(dn);
            *(u32x4a_t*)(xd + 3 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[6] = mma(d1.f, B2, acc[6]);
            dec_s6(dn, lhb);
            SB();
            acc[7] = mma(d1.f, B0, acc[7]);
            SB();
            MQ_BARRIER();
            a_cur = dn.f;
            }
            { // ---- chunk 1
            const char* xb = &x_lds[1 * BUFB + bbase];
            char* xd = &x_lds[0 * BUFB + xdst];
            int cx = g * 8 + 2; cx = (cx < nch) ? cx : (nch - 1); const int xsoff = cx * 64;
            u32x8a_t B0, B1, B2;
            Dec d1, dn;
            B0 = ldb(xb + 0);
            B1 = ldb(xb + 1280);
            B2 = ldb(xb + 2560);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[0], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[1], xsoff, 0);
            SB();
            acc[0] = mma(a_cur, B0, acc[0]);
            B0 = ldb(xb + 3840);
            SB();
            acc[1] = mma(a_cur, B1, acc[1]);
            B1 = ldb(xb + 5120);
            dec_s0(d1, w_1_1[0], w_1_1[1], hdA, sel);
            dec_s1(d1);
            SB();
            acc[2] = mma(a_cur, B2, acc[2]);
            B2 = ldb(xb + 6400);
            dec_s2(d1);
            SB();
            acc[3] = mma(a_cur, B0, acc[3]);
            B0 = ldb(xb + 7680);
            dec_s3(d1);
            SB();
            acc[4] = mma(a_cur, B1, acc[4]);
            B1 = ldb(xb + 8960);
            dec_s4(d1);
            w_1_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 72, g * GROUP_BYTES, 0);
            w_1_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 84, g * GROUP_BYTES, 0);
            SB();
            acc[5] = mma(a_cur, B2, acc[5]);
            B2 = ldb(xb + 32);
            dec_s5(d1);
            *(u32x4a_t*)(xd + 0 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[6] = mma(a_cur, B0, acc[6]);
            B0 = ldb(xb + 1312);
            dec_s6(d1, lhb);
            *(u32x4a_t*)(xd + 1 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[7] = mma(a_cur, B1, acc[7]);
            B1 = ldb(xb + 2592);
            SB();
            acc[0] = mma(d1.f, B2, acc[0]);
            B2 = ldb(xb + 3872);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[2], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[3], xsoff, 0);
            SB();
            acc[1] = mma(d1.f, B0, acc[1]);
            B0 = ldb(xb + 5152);
            dec_s0(dn, w_0_0[0], w_0_0[1], hdA, sel);
            dec_s1(dn);
            SB();
            acc[2] = mma(d1.f, B1, acc[2]);
            B1 = ldb(xb + 6432);
            dec_s2(dn);
            SB();
            acc[3] = mma(d1.f, B2, acc[3]);
            B2 = ldb(xb + 7712);
            dec_s3(dn);
            *(u32x4a_t*)(xd + 2 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[4] = mma(d1.f, B0, acc[4]);
            B0 = ldb(xb + 8992);
            dec_s4(dn);
            SB();
            acc[5] = mma(d1.f, B1, acc[5]);
            dec_s5(dn);
            *(u32x4a_t*)(xd + 3 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[6] = mma(d1.f, B2, acc[6]);
            dec_s6(dn, lhb);
            SB();
            acc[7] = mma(d1.f, B0, acc[7]);
            SB();
            MQ_BARRIER();
            a_cur = dn.f;
            }
            { // ---- chunk 2
            const char* xb = &x_lds[0 * BUFB + bbase];
            char* xd = &x_lds[1 * BUFB + xdst];
            int cx = g * 8 + 3; cx = (cx < nch) ? cx : (nch - 1); const int xsoff = cx * 64;
            u32x8a_t B0, B1, B2;
            Dec d1, dn;
            B0 = ldb(xb + 0);
            B1 = ldb(xb + 1280);
            B2 = ldb(xb + 2560);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[0], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[1], xsoff, 0);
            SB();
            acc[0] = mma(a_cur, B0, acc[0]);
            B0 = ldb(xb + 3840);
            SB();
            acc[1] = mma(a_cur, B1, acc[1]);
            B1 = ldb(xb + 5120);
            dec_s0(d1, w_0_1[0], w_0_1[1], hdA, sel);
            dec_s1(d1);
            SB();
            acc[2] = mma(a_cur, B2, acc[2]);
            B2 = ldb(xb + 6400);
            dec_s2(d1);
            SB();
            acc[3] = mma(a_cur, B0, acc[3]);
            B0 = ldb(xb + 7680);
            dec_s3(d1);
            SB();
            acc[4] = mma(a_cur, B1, acc[4]);
            B1 = ldb(xb + 8960);
            dec_s4(d1);
            w_0_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 96, g * GROUP_BYTES, 0);
            w_0_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 108, g * GROUP_BYTES, 0);
            SB();
            acc[5] = mma(a_cur, B2, acc[5]);
            B2 = ldb(xb + 32);
            dec_s5(d1);
            *(u32x4a_t*)(xd + 0 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[6] = mma(a_cur, B0, acc[6]);
            B0 = ldb(xb + 1312);
            dec_s6(d1, lhb);
            *(u32x4a_t*)(xd + 1 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[7] = mma(a_cur, B1, acc[7]);
            B1 = ldb(xb + 2592);
            SB();
            acc[0] = mma(d1.f, B2, acc[0]);
            B2 = ldb(xb + 3872);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[2], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[3], xsoff, 0);
            SB();
            acc[1] = mma(d1.f, B0, acc[1]);
            B0 = ldb(xb + 5152);
            dec_s0(dn, w_1_0[0], w_1_0[1], hdA, sel);
            dec_s1(dn);
            SB();
            acc[2] = mma(d1.f, B1, acc[2]);
            B1 = ldb(xb + 6432);
            dec_s2(dn);
            SB();
            acc[3] = mma(d1.f, B2, acc[3]);
            B2 = ldb(xb + 7712);
            dec_s3(dn);
            *(u32x4a_t*)(xd + 2 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[4] = mma(d1.f, B0, acc[4]);
            B0 = ldb(xb + 8992);
            dec_s4(dn);
            SB();
            acc[5] = mma(d1.f, B1, acc[5]);
            dec_s5(dn);
            *(u32x4a_t*)(xd + 3 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[6] = mma(d1.f, B2, acc[6]);
            dec_s6(dn, lhb);
            SB();
            acc[7] = mma(d1.f, B0, acc[7]);
            SB();
            MQ_BARRIER();
            a_cur = dn.f;
            }
            { // ---- chunk 3
            const char* xb = &x_lds[1 * BUFB + bbase];
            char* xd = &x_lds[0 * BUFB + xdst];
            int cx = g * 8 + 4; cx = (cx < nch) ? cx : (nch - 1); const int xsoff = cx * 64;
            u32x8a_t B0, B1, B2;
            Dec d1, dn;
            B0 = ldb(xb + 0);
            B1 = ldb(xb + 1280);
            B2 = ldb(xb + 2560);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[0], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[1], xsoff, 0);
            SB();
            acc[0] = mma(a_cur, B0, acc[0]);
            B0 = ldb(xb + 3840);
            SB();
            acc[1] = mma(a_cur, B1, acc[1]);
            B1 = ldb(xb + 5120);
            dec_s0(d1, w_1_1[0], w_1_1[1], hdA, sel);
            dec_s1(d1);
            SB();
            acc[2] = mma(a_cur, B2, acc[2]);
            B2 = ldb(xb + 6400);
            dec_s2(d1);
            SB();
            acc[3] = mma(a_cur, B0, acc[3]);
            B0 = ldb(xb + 7680);
            dec_s3(d1);
            SB();
            acc[4] = mma(a_cur, B1, acc[4]);
            B1 = ldb(xb + 8960);
            dec_s4(d1);
            w_1_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 120, g * GROUP_BYTES, 0);
            w_1_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 132, g * GROUP_BYTES, 0);
            SB();
            acc[5] = mma(a_cur, B2, acc[5]);
            B2 = ldb(xb + 32);
            dec_s5(d1);
            *(u32x4a_t*)(xd + 0 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[6] = mma(a_cur, B0, acc[6]);
            B0 = ldb(xb + 1312);
            dec_s6(d1, lhb);
            *(u32x4a_t*)(xd + 1 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[7] = mma(a_cur, B1, acc[7]);
            B1 = ldb(xb + 2592);
            SB();
            acc[0] = mma(d1.f, B2, acc[0]);
            B2 = ldb(xb + 3872);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[2], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[3], xsoff, 0);
            SB();
            acc[1] = mma(d1.f, B0, acc[1]);
            B0 = ldb(xb + 5152);
            dec_s0(dn, w_0_0[0], w_0_0[1], hdB, sel);
            dec_s1(dn);
            SB();
            acc[2] = mma(d1.f, B1, acc[2]);
            B1 = ldb(xb + 6432);
            dec_s2(dn);
            SB();
            acc[3] = mma(d1.f, B2, acc[3]);
            B2 = ldb(xb + 7712);
            dec_s3(dn);
            *(u32x4a_t*)(xd + 2 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[4] = mma(d1.f, B0, acc[4]);
            B0 = ldb(xb + 8992);
            dec_s4(dn);
            SB();
            acc[5] = mma(d1.f, B1, acc[5]);
            dec_s5(dn);
            *(u32x4a_t*)(xd + 3 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[6] = mma(d1.f, B2, acc[6]);
            dec_s6(dn, lhb);
            SB();
            acc[7] = mma(d1.f, B0, acc[7]);
            SB();
            MQ_BARRIER();
            a_cur = dn.f;
            }
            { // ---- chunk 4
            const char* xb = &x_lds[0 * BUFB + bbase];
            char* xd = &x_lds[1 * BUFB + xdst];
            int cx = g * 8 + 5; cx = (cx < nch) ? cx : (nch - 1); const int xsoff = cx * 64;
            u32x8a_t B0, B1, B2;
            Dec d1, dn;
            B0 = ldb(xb + 0);
            B1 = ldb(xb + 1280);
            B2 = ldb(xb + 2560);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[0], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[1], xsoff, 0);
            SB();
            acc[0] = mma(a_cur, B0, acc[0]);
            B0 = ldb(xb + 3840);
            SB();
            acc[1] = mma(a_cur, B1, acc[1]);
            B1 = ldb(xb + 5120);
            dec_s0(d1, w_0_1[0], w_0_1[1], hdB, sel);
            dec_s1(d1);
            SB();
            acc[2] = mma(a_cur, B2, acc[2]);
            B2 = ldb(xb + 6400);
            dec_s2(d1);
            SB();
            acc[3] = mma(a_cur, B0, acc[3]);
            B0 = ldb(xb + 7680);
            dec_s3(d1);
            SB();
            acc[4] = mma(a_cur, B1, acc[4]);
            B1 = ldb(xb + 8960);
            dec_s4(d1);
            w_0_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 144, g * GROUP_BYTES, 0);
            w_0_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 156, g * GROUP_BYTES, 0);
            SB();
            acc[5] = mma(a_cur, B2, acc[5]);
            B2 = ldb(xb + 32);
            dec_s5(d1);
            *(u32x4a_t*)(xd + 0 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[6] = mma(a_cur, B0, acc[6]);
            B0 = ldb(xb + 1312);
            dec_s6(d1, lhb);
            *(u32x4a_t*)(xd + 1 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[7] = mma(a_cur, B1, acc[7]);
            B1 = ldb(xb + 2592);
            SB();
            acc[0] = mma(d1.f, B2, acc[0]);
            B2 = ldb(xb + 3872);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[2], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[3], xsoff, 0);
            SB();
            acc[1] = mma(d1.f, B0, acc[1]);
            B0 = ldb(xb + 5152);
            dec_s0(dn, w_1_0[0], w_1_0[1], hdB, sel);
            dec_s1(dn);
            SB();
            acc[2] = mma(d1.f, B1, acc[2]);
            B1 = ldb(xb + 6432);
            dec_s2(dn);
            SB();
            acc[3] = mma(d1.f, B2, acc[3]);
            B2 = ldb(xb + 7712);
            dec_s3(dn);
            *(u32x4a_t*)(xd + 2 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[4] = mma(d1.f, B0, acc[4]);
            B0 = ldb(xb + 8992);
            dec_s4(dn);
            SB();
            acc[5] = mma(d1.f, B1, acc[5]);
            dec_s5(dn);
            *(u32x4a_t*)(xd + 3 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[6] = mma(d1.f, B2, acc[6]);
            dec_s6(dn, lhb);
            SB();
            acc[7] = mma(d1.f, B0, acc[7]);
            SB();
            MQ_BARRIER();
            a_cur = dn.f;
            }
            { // ---- chunk 5
            const char* xb = &x_lds[1 * BUFB + bbase];
            char* xd = &x_lds[0 * BUFB + xdst];
            int cx = g * 8 + 6; cx = (cx < nch) ? cx : (nch - 1); const int xsoff = cx * 64;
            u32x8a_t B0, B1, B2;
            Dec d1, dn;
            B0 = ldb(xb + 0);
            B1 = ldb(xb + 1280);
            B2 = ldb(xb + 2560);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[0], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[1], xsoff, 0);
            SB();
            acc[0] = mma(a_cur, B0, acc[0]);
            B0 = ldb(xb + 3840);
            SB();
            acc[1] = mma(a_cur, B1, acc[1]);
            B1 = ldb(xb + 5120);
            dec_s0(d1, w_1_1[0], w_1_1[1], hdB, sel);
            dec_s1(d1);
            SB();
            acc[2] = mma(a_cur, B2, acc[2]);
            B2 = ldb(xb + 6400);
            dec_s2(d1);
            SB();
            acc[3] = mma(a_cur, B0, acc[3]);
            B0 = ldb(xb + 7680);
            dec_s3(d1);
            SB();
            acc[4] = mma(a_cur, B1, acc[4]);
            B1 = ldb(xb + 8960);
            dec_s4(d1);
            w_1_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 168, g * GROUP_BYTES, 0);
            w_1_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 180, g * GROUP_BYTES, 0);
            SB();
            acc[5] = mma(a_cur, B2, acc[5]);
            B2 = ldb(xb + 32);
            dec_s5(d1);
            *(u32x4a_t*)(xd + 0 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[6] = mma(a_cur, B0, acc[6]);
            B0 = ldb(xb + 1312);
            dec_s6(d1, lhb);
            *(u32x4a_t*)(xd + 1 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[7] = mma(a_cur, B1, acc[7]);
            B1 = ldb(xb + 2592);
            SB();
            acc[0] = mma(d1.f, B2, acc[0]);
            B2 = ldb(xb + 3872);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[2], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[3], xsoff, 0);
            SB();
            acc[1] = mma(d1.f, B0, acc[1]);
            B0 = ldb(xb + 5152);
            dec_s0(dn, w_0_0[0], w_0_0[1], hdB, sel);
            dec_s1(dn);
            SB();
            acc[2] = mma(d1.f, B1, acc[2]);
            B1 = ldb(xb + 6432);
            dec_s2(dn);
            SB();
            acc[3] = mma(d1.f, B2, acc[3]);
            B2 = ldb(xb + 7712);
            dec_s3(dn);
            *(u32x4a_t*)(xd + 2 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[4] = mma(d1.f, B0, acc[4]);
            B0 = ldb(xb + 8992);
            dec_s4(dn);
            SB();
            acc[5] = mma(d1.f, B1, acc[5]);
            dec_s5(dn);
            *(u32x4a_t*)(xd + 3 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[6] = mma(d1.f, B2, acc[6]);
            dec_s6(dn, lhb);
            SB();
            acc[7] = mma(d1.f, B0, acc[7]);
            SB();
            MQ_BARRIER();
            a_cur = dn.f;
            }
            { // ---- chunk 6
            const char* xb = &x_lds[0 * BUFB + bbase];
            char* xd = &x_lds[1 * BUFB + xdst];
            int cx = g * 8 + 7; cx = (cx < nch) ? cx : (nch - 1); const int xsoff = cx * 64;
            u32x8a_t B0, B1, B2;
            Dec d1, dn;
            B0 = ldb(xb + 0);
            B1 = ldb(xb + 1280);
            B2 = ldb(xb + 2560);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[0], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[1], xsoff, 0);
            SB();
            acc[0] = mma(a_cur, B0, acc[0]);
            B0 = ldb(xb + 3840);
            SB();
            acc[1] = mma(a_cur, B1, acc[1]);
            B1 = ldb(xb + 5120);
            dec_s0(d1, w_0_1[0], w_0_1[1], hdB, sel);
            dec_s1(d1);
            SB();
            acc[2] = mma(a_cur, B2, acc[2]);
            B2 = ldb(xb + 6400);
            dec_s2(d1);
            SB();
            acc[3] = mma(a_cur, B0, acc[3]);
            B0 = ldb(xb + 7680);
            dec_s3(d1);
            SB();
            acc[4] = mma(a_cur, B1, acc[4]);
            B1 = ldb(xb + 8960);
            dec_s4(d1);
            w_0_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 0, gn * GROUP_BYTES, 0);
            w_0_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 12, gn * GROUP_BYTES, 0);
            { const u32x2_t h = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, hv, gn * GROUP_BYTES, 0); hnA = h[0]; hnB = h[1]; }
            SB();
            acc[5] = mma(a_cur, B2, acc[5]);
            B2 = ldb(xb + 32);
            dec_s5(d1);
            *(u32x4a_t*)(xd + 0 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[6] = mma(a_cur, B0, acc[6]);
            B0 = ldb(xb + 1312);
            dec_s6(d1, lhb);
            *(u32x4a_t*)(xd + 1 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[7] = mma(a_cur, B1, acc[7]);
            B1 = ldb(xb + 2592);
            SB();
            acc[0] = mma(d1.f, B2, acc[0]);
            B2 = ldb(xb + 3872);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[2], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[3], xsoff, 0);
            SB();
            acc[1] = mma(d1.f, B0, acc[1]);
            B0 = ldb(xb + 5152);
            dec_s0(dn, w_1_0[0], w_1_0[1], hdB, sel);
            dec_s1(dn);
            SB();
            acc[2] = mma(d1.f, B1, acc[2]);
            B1 = ldb(xb + 6432);
            dec_s2(dn);
            SB();
            acc[3] = mma(d1.f, B2, acc[3]);
            B2 = ldb(xb + 7712);
            dec_s3(dn);
            *(u32x4a_t*)(xd + 2 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[4] = mma(d1.f, B0, acc[4]);
            B0 = ldb(xb + 8992);
            dec_s4(dn);
            SB();
            acc[5] = mma(d1.f, B1, acc[5]);
            dec_s5(dn);
            *(u32x4a_t*)(xd + 3 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[6] = mma(d1.f, B2, acc[6]);
            dec_s6(dn, lhb);
            SB();
            acc[7] = mma(d1.f, B0, acc[7]);
            SB();
            MQ_BARRIER();
            a_cur = dn.f;
            }
            { // ---- chunk 7
            const char* xb = &x_lds[1 * BUFB + bbase];
            char* xd = &x_lds[0 * BUFB + xdst];
            int cx = g * 8 + 8; cx = (cx < nch) ? cx : (nch - 1); const int xsoff = cx * 64;
            u32x8a_t B0, B1, B2;
            Dec d1, dn;
            B0 = ldb(xb + 0);
            B1 = ldb(xb + 1280);
            B2 = ldb(xb + 2560);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[0], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[1], xsoff, 0);
            SB();
            acc[0] = mma(a_cur, B0, acc[0]);
            B0 = ldb(xb + 3840);
            SB();
            acc[1] = mma(a_cur, B1, acc[1]);
            B1 = ldb(xb + 5120);
            dec_s0(d1, w_1_1[0], w_1_1[1], hdB, sel);
            dec_s1(d1);
            SB();
            acc[2] = mma(a_cur, B2, acc[2]);
            B2 = ldb(xb + 6400);
            dec_s2(d1);
            SB();
            acc[3] = mma(a_cur, B0, acc[3]);
            B0 = ldb(xb + 7680);
            dec_s3(d1);
            SB();
            acc[4] = mma(a_cur, B1, acc[4]);
            B1 = ldb(xb + 8960);
            dec_s4(d1);
            w_1_0 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 24, gn * GROUP_BYTES, 0);
            w_1_1 = __builtin_amdgcn_raw_buffer_load_b64(wrsrc, wv + 36, gn * GROUP_BYTES, 0);
            SB();
            acc[5] = mma(a_cur, B2, acc[5]);
            B2 = ldb(xb + 32);
            dec_s5(d1);
            *(u32x4a_t*)(xd + 0 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[6] = mma(a_cur, B0, acc[6]);
            B0 = ldb(xb + 1312);
            dec_s6(d1, lhb);
            *(u32x4a_t*)(xd + 1 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[7] = mma(a_cur, B1, acc[7]);
            B1 = ldb(xb + 2592);
            SB();
            acc[0] = mma(d1.f, B2, acc[0]);
            B2 = ldb(xb + 3872);
            xr[0] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[2], xsoff, 0);
            xr[1] = __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xv[3], xsoff, 0);
            SB();
            acc[1] = mma(d1.f, B0, acc[1]);
            B0 = ldb(xb + 5152);
            dec_s0(dn, w_0_0[0], w_0_0[1], hnA, sel);
            dec_s1(dn);
            SB();
            acc[2] = mma(d1.f, B1, acc[2]);
            B1 = ldb(xb + 6432);
            dec_s2(dn);
            SB();
            acc[3] = mma(d1.f, B2, acc[3]);
            B2 = ldb(xb + 7712);
            dec_s3(dn);
            *(u32x4a_t*)(xd + 2 * TOK_STEP * XPITCH) = xr[0];
            SB();
            acc[4] = mma(d1.f, B0, acc[4]);
            B0 = ldb(xb + 8992);
            dec_s4(dn);
            SB();
            acc[5] = mma(d1.f, B1, acc[5]);
            dec_s5(dn);
            *(u32x4a_t*)(xd + 3 * TOK_STEP * XPITCH) = xr[1];
            SB();
            acc[6] = mma(d1.f, B2, acc[6]);
            dec_s6(dn, lhb);
            SB();
            acc[7] = mma(d1.f, B0, acc[7]);
            SB();
            MQ_BARRIER();
            a_cur = dn.f;
            }
            hdA = hnA;
            hdB = hnB;
        }
    }

    // ---- epilogue. RDNA3 wave32 WMMA: acc[j][i] = C[2*i + lh][ml]. One cross-half swap gives lane half 0
    // rows 0..7 and half 1 rows 8..15 of column ml (pure bit moves); Y += acc per element.
#pragma unroll
    for (int j = 0; j < TILES; j++) {
        float r[8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float lo_v = acc[j][i], hi_v = acc[j][4 + i];
            asm("" : "+v"(lo_v), "+v"(hi_v));   // keep the select a v_cndmask (no dynamic extract)
            const float give = lhb ? lo_v : hi_v;
            const float got = __int_as_float(__builtin_amdgcn_permlanex16(
                __float_as_int(give), __float_as_int(give), 0x76543210u, 0xFEDCBA98u, false, false));
            const float own = lhb ? hi_v : lo_v;
            r[2 * i] = lhb ? got : own;
            r[2 * i + 1] = lhb ? own : got;
        }
        const int out_col = n_start + j * 16 + ml;
        const int row0 = row_start + 8 * lh;
        if (out_col < N && row0 < M) {
            float* out = Y + (long long)out_col * M + row0;
            if (row0 + 8 <= M && (M & 3) == 0) {
                float4 o0 = *(const float4*)out;
                float4 o1 = *(const float4*)(out + 4);
                o0.x += r[0]; o0.y += r[1]; o0.z += r[2]; o0.w += r[3];
                o1.x += r[4]; o1.y += r[5]; o1.z += r[6]; o1.w += r[7];
                *(float4*)out = o0;
                *(float4*)(out + 4) = o1;
            } else {
#pragma unroll
                for (int i = 0; i < 8; i++)
                    if (row0 + i < M) out[i] += r[i];
            }
        }
    }
}
} // namespace w4n128

#endif // __gfx1100__
