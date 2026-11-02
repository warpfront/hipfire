// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Ben1078
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exact-gfx1100 small-batch MQ4G256V2 GEMV on int8 activations for
// speculative verify (MTP K=3 -> 4 rows), default on for exact gfx1100;
// HIPFIRE_GFX1100_Q8X_VERIFY=0 restores the f16 WMMA tiers (see
// crates/rdna-compute/src/gemv_q8x.rs).
//
// The f16 WMMA residual tiers spend 12 of 16 tile columns on padding at 4
// rows; this route reads every weight row once and dots it against all B
// activation rows with v_dot4_i32_iu8 (unsigned 4-bit codes x signed int8).
//
//   mq4v2_q8x_quant (Q8X_QUANT): x f32 [B x K] (already FWHT-rotated) ->
//     int8, one f32 scale per 32 elements, packed even/odd to match the
//     nibble order of the MQ4 codes. grid (K/256, B), block 32.
//   GEMV per 32-element block l of a 256-group, batch row b:
//     acc += (d * sc) * sum(q * xq) + zp * (d * sum(xq))
//   The integer sums are exact; the activation quantization is the only
//   approximation (q8_1 class). Not byte-identical to the WMMA route.
//
// Variants (compile-time, one module per batch / residual flavour):
//   v3 (Q8X_GEMV3): Q8X3_R = 8 warps x Q8X3_W = 2 rows per workgroup, K in
//     chunks of <= Q8X3_LDSKB = 32 KiB of staged activations, Q8X3_P = 2-deep
//     weight ring. 256 threads, LDS <= 32 KiB, ~2 workgroups per WGP.
//   v4 (Q8X_GEMV4): persistent grid (one resident wave of workgroups from
//     the occupancy API), all of K staged once (LDS = ceil(K/1024) * 4 *
//     (PSX + PDD) bytes, <= 64 KiB: K <= 5120 at B = 4, K <= 17408 at B = 2),
//     Q8X3_R = 8 warps, 3-deep nontemporal dwordx4 weight ring that runs
//     across rows; rows split between two warps are finished in fixed order.
// Both are deterministic (no atomics; split partials reduced in fixed order
// through LDS) and have a _multi twin that serves up to four weights sharing
// one activation (q/k/v, GDN qkv/z/beta/alpha, gate/up) in one launch.
// Resources (hipcc 7.2, --offload-arch=gfx1100 -O3, no scratch spills):
//   v4: 91 VGPRs at B = 4, 77 at B = 2; v3 (W = 2): 96 / 107 / 118 VGPRs at
//   B = 3 / 4 / 5. wave32, 256-thread workgroups; the launcher floors v4 at 2
//   resident workgroups per WGP (the occupancy API counts 64 KiB LDS per CU).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef Q8X_BMAX
#define Q8X_BMAX 4
#endif
#ifndef Q8X_KERNEL
#define Q8X_KERNEL mq4v2_q8x_gemv
#endif
#ifndef Q8X_MULTI_KERNEL
#define Q8X_MULTI_KERNEL mq4v2_q8x_gemv_multi
#endif

#ifdef Q8X_QUANT
// grid (K/256, B), block 32. Lane t owns elements g*256 + t*8 .. +7.
extern "C" __global__ void mq4v2_q8x_quant(
    const float* __restrict__ x,
    unsigned int* __restrict__ xq,   // [B][K/256][32] uint2
    float* __restrict__ dx,          // [B][K/32]
    int K)
{
    const int g = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
    const float* xp = x + (long long)b * K + g * 256 + t * 8;
    const float4 va = *(const float4*)xp;
    const float4 vb = *(const float4*)(xp + 4);
    const float v[8] = {va.x, va.y, va.z, va.w, vb.x, vb.y, vb.z, vb.w};
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(v[j]));
    // 32-element block = 4 lanes (4k..4k+3).
    amax = fmaxf(amax, __shfl_xor(amax, 1));
    amax = fmaxf(amax, __shfl_xor(amax, 2));
    const float d = amax * (1.0f / 127.0f);
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    int q[8];
#pragma unroll
    for (int j = 0; j < 8; j++) q[j] = (int)rintf(v[j] * id);
    const unsigned int e = (unsigned)(q[0] & 0xFF) | ((unsigned)(q[2] & 0xFF) << 8)
                         | ((unsigned)(q[4] & 0xFF) << 16) | ((unsigned)(q[6] & 0xFF) << 24);
    const unsigned int o = (unsigned)(q[1] & 0xFF) | ((unsigned)(q[3] & 0xFF) << 8)
                         | ((unsigned)(q[5] & 0xFF) << 16) | ((unsigned)(q[7] & 0xFF) << 24);
    uint2* xq2 = (uint2*)xq;
    xq2[((long long)b * (K >> 8) + g) * 32 + t] = make_uint2(e, o);
    if ((t & 3) == 0) dx[(long long)b * (K >> 5) + g * 8 + (t >> 2)] = d;
}
#endif

#if defined(Q8X_GEMV3) || defined(Q8X_GEMV4)
// Shared warp helpers of the v3/v4 kernels.
// v from lane t ^ OFF (OFF in 1, 2, 4, 8: DPP row_xmask; 16: permlanex16).
template <int OFF>
__device__ __forceinline__ float q8x_xor(float v) {
    int x = __float_as_int(v);
    if constexpr (OFF == 16)
        x = __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xfedcba98u, false, false);
    else
        x = __builtin_amdgcn_update_dpp(0, x, 0x160 + OFF, 0xf, 0xf, false);
    return __int_as_float(x);
}

// Transposed warp reduction of N = 2^LVP values per lane: stage s halves the
// values each lane keeps (lanes with bit 4 - s set keep the upper half), so N
// values cost N - 1 exchanges instead of 5 * N; then plain xor stages.
// Every lane t returns the warp total of value t >> (5 - LVP). Written with
// template recursion so every array index is a compile-time constant in the
// source: with a loop index, InstCombine folds `upper ? v[i + h] : v[i]` into
// a dynamically indexed access before unrolling, which lowers to long
// compare/select chains (measured: +22% kernel time at batch 5).
template <int I, int HALF, int OFF, int N>
__device__ __forceinline__ void q8x_rpair(float (&v)[N], bool upper) {
    if constexpr (I < HALF) {
        const float a = v[I];
        const float b = v[I + HALF];
        v[I] = (upper ? b : a) + q8x_xor<OFF>(upper ? a : b);
        q8x_rpair<I + 1, HALF, OFF, N>(v, upper);
    }
}

template <int ST, int LVP, int N>
__device__ __forceinline__ void q8x_rstages(float (&v)[N], int t) {
    if constexpr (ST < LVP) {
        constexpr int OFF = 16 >> ST;
        q8x_rpair<0, (N >> (ST + 1)), OFF, N>(v, (t & OFF) != 0);
        q8x_rstages<ST + 1, LVP, N>(v, t);
    }
}

template <int OFF>
__device__ __forceinline__ float q8x_rtail(float r) {
    if constexpr (OFF >= 1) return q8x_rtail<OFF / 2>(r + q8x_xor<OFF>(r));
    else return r;
}

template <int LVP>
__device__ __forceinline__ float q8x_reduce(float (&v)[1 << LVP], int t) {
    q8x_rstages<0, LVP, (1 << LVP)>(v, t);
    return q8x_rtail<(16 >> LVP)>(v[0]);
}
#endif

#if defined(Q8X_GEMV3) || defined(Q8X_GEMV4)
// v3: one warp step = four 256-groups (lane t =
// group t >> 3, 32-element block t & 7, one dwordx4 of codes) with the int8
// activations staged in LDS once per workgroup, together with each block's
// (d, d * sum(xq)). Activations then come from ds_read (lgkmcnt), so the
// in-order vmcnt only tracks weight loads and the P-deep weight ring really
// stays in flight; the zero-point term costs one FMA per block instead of
// eight dot4 per row.
//   per lane, block, batch row: acc += (d * sc) * sum(q * xq) + zp * (d * sum(xq))
// K is processed in chunks of at most Q8X3_LDSKB KiB of staged activations
// (1280 bytes per 4-group step per batch row); one chunk covers K <= 6144 at
// batch 4 with the default 32 KiB.
//   Q8X3_R: warps per workgroup (each W rows, all of K), Q8X3_W: rows per
//   warp, Q8X3_P: weight ring depth (steps), Q8X3_NT: nontemporal weights,
//   Q8X3_X: 1 = read the next step's activations from LDS one step ahead.
#ifndef Q8X3_R
#define Q8X3_R 8
#endif
#ifndef Q8X3_W
#define Q8X3_W 1
#endif
#ifndef Q8X3_P
#define Q8X3_P 3
#endif
#ifndef Q8X3_NT
#define Q8X3_NT 1
#endif
#ifndef Q8X3_X
#define Q8X3_X 0
#endif
#ifndef Q8X3_LDSKB
#define Q8X3_LDSKB 32
#endif
#ifndef Q8X3_RQ
#define Q8X3_RQ 4
#endif
#define Q8X3_NB Q8X_BMAX
#define Q8X3_T (32 * Q8X3_R)
#define Q8X3_ROWS (Q8X3_R * Q8X3_W)
#define Q8X3_V (Q8X3_W * Q8X3_NB)
#define Q8X3_LVP (Q8X3_V <= 1 ? 0 : Q8X3_V <= 2 ? 1 : Q8X3_V <= 4 ? 2 : Q8X3_V <= 8 ? 3 : Q8X3_V <= 16 ? 4 : 5)
#define Q8X3_VP (1 << Q8X3_LVP)
static_assert(Q8X3_V <= 32, "mq4v2_q8x v3: W * batch must be <= 32");

typedef unsigned int q8x3_u4 __attribute__((ext_vector_type(4)));

extern __shared__ __attribute__((aligned(16))) char q8x3_smem[];

__device__ __forceinline__ q8x3_u4 q8x3_ldw(const char* p) {
#if Q8X3_NT
    return __builtin_nontemporal_load((const q8x3_u4*)p);
#else
    return *(const q8x3_u4*)p;
#endif
}

__device__ __forceinline__ unsigned int q8x3_ldh(const char* p) {
#if Q8X3_NT
    return __builtin_nontemporal_load((const unsigned int*)p);
#else
    return *(const unsigned int*)p;
#endif
}

__device__ __forceinline__ int q8x3_sum(q8x3_u4 a, q8x3_u4 b) {
    int s = __builtin_amdgcn_sudot4(true, (int)a.x, false, 0x01010101, 0, false);
    s = __builtin_amdgcn_sudot4(true, (int)a.y, false, 0x01010101, s, false);
    s = __builtin_amdgcn_sudot4(true, (int)a.z, false, 0x01010101, s, false);
    s = __builtin_amdgcn_sudot4(true, (int)a.w, false, 0x01010101, s, false);
    s = __builtin_amdgcn_sudot4(true, (int)b.x, false, 0x01010101, s, false);
    s = __builtin_amdgcn_sudot4(true, (int)b.y, false, 0x01010101, s, false);
    s = __builtin_amdgcn_sudot4(true, (int)b.z, false, 0x01010101, s, false);
    return __builtin_amdgcn_sudot4(true, (int)b.w, false, 0x01010101, s, false);
}

// LDS layout of a staged chunk, per 256-group g (relative to the chunk):
//   xq : Q8X3_PSX bytes per group; batch row b's 8 blocks at b*256 + l*32
//   dd : Q8X3_PDD bytes per group; (d, d*sum) of (b, l) at b*64 + l*8
// Batch offsets are compile-time immediates of the ds_read. The group pitches
// are skewed (PSX == 16, PDD == 64 mod 256) so the four groups of a warp step
// land on different banks: a ds_read_b128 of the step then takes the minimum
// two bank cycles and a ds_read_b64 one.
#define Q8X3_PSX (Q8X3_NB * 256 + 16)
#define Q8X3_PDD (((Q8X3_NB * 64 + 191) / 256) * 256 + 64)   // >= NB*64, == 64 mod 256

// Steps per staged chunk for this K and batch: all of K when it fits in
// Q8X3_LDSKB KiB, else the largest multiple of P that fits (at least P), so
// every chunk starts on ring slot 0. The launcher sizes the dynamic LDS with
// the same formula (steps * 4 * (PSX + PDD) bytes).
__device__ __forceinline__ int q8x3_chunk_steps(int nst) {
    int cs = (Q8X3_LDSKB * 1024) / (4 * (Q8X3_PSX + Q8X3_PDD));
    if (nst <= cs) return nst;
    cs -= cs % Q8X3_P;
    return cs < Q8X3_P ? Q8X3_P : cs;
}

// Stage the activation blocks of steps [c0, c1) (all batch rows) into LDS
// (layout above), with each block's (d, d * sum(xq)).
__device__ __forceinline__ void q8x3_stage(
    const unsigned int* __restrict__ xq, const float* __restrict__ dx,
    int gpr, int c0, int c1, int cs)
{
    const int tid = threadIdx.x;
    const int g0 = 4 * c0;
    const int ng = min(4 * c1, gpr) - g0;             // groups in this chunk
    const int nq = Q8X3_NB * ng * 8;                  // blocks to stage
    char* sxb = q8x3_smem;
    char* sdb = q8x3_smem + (long long)cs * 4 * Q8X3_PSX;
    const q8x3_u4* gx = (const q8x3_u4*)xq;
    for (int q0 = 0; q0 < nq; q0 += Q8X3_T * Q8X3_RQ) {
        q8x3_u4 va[Q8X3_RQ], vb[Q8X3_RQ];
        float vd[Q8X3_RQ];
#pragma unroll
        for (int i = 0; i < Q8X3_RQ; i++) {
            const int q = min(q0 + i * Q8X3_T + tid, nq - 1);
            const int b = q / (ng * 8);
            const int r = q - b * (ng * 8);              // (g - g0) * 8 + l
            const long long gq = (long long)b * gpr * 8 + g0 * 8 + r;   // global block index
            va[i] = gx[2 * gq];
            vb[i] = gx[2 * gq + 1];
            vd[i] = dx[gq];
        }
#pragma unroll
        for (int i = 0; i < Q8X3_RQ; i++) {
            const int q = q0 + i * Q8X3_T + tid;
            if (q < nq) {
                const int b = q / (ng * 8);
                const int r = q - b * (ng * 8);
                const int g = r >> 3, l = r & 7;
                q8x3_u4* px = (q8x3_u4*)(sxb + g * Q8X3_PSX + b * 256 + l * 32);
                px[0] = va[i];
                px[1] = vb[i];
                *(float2*)(sdb + g * Q8X3_PDD + b * 64 + l * 8) =
                    make_float2(vd[i], vd[i] * (float)q8x3_sum(va[i], vb[i]));
            }
        }
    }
}

// Activations of block l of chunk group lg for every batch row.
__device__ __forceinline__ void q8x3_ldsx(
    q8x3_u4 (&xa)[Q8X3_NB], q8x3_u4 (&xb)[Q8X3_NB], float2 (&dd)[Q8X3_NB], int cs, int lg, int l)
{
    const char* px = q8x3_smem + lg * Q8X3_PSX + l * 32;
    const char* pd = q8x3_smem + (long long)cs * 4 * Q8X3_PSX + lg * Q8X3_PDD + l * 8;
#pragma unroll
    for (int b = 0; b < Q8X3_NB; b++) {
        xa[b] = *(const q8x3_u4*)(px + b * 256);
        xb[b] = *(const q8x3_u4*)(px + b * 256 + 16);
        dd[b] = *(const float2*)(pd + b * 64);
    }
}
#endif

#ifdef Q8X_GEMV3
// Rows row0 .. row0 + R*W - 1 of one weight (one workgroup).
__device__ __forceinline__ void q8x3_rows(
    const char* __restrict__ A,
    const unsigned int* __restrict__ xq,
    const float* __restrict__ dx,
    float* __restrict__ y,
    int M, int K, int row0)
{
    const int tid = threadIdx.x;
    const int t = tid & 31;
    const int wv = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int rbase = row0 + wv * Q8X3_W;
    const int gpr = K >> 8;
    const int nst = (gpr + 3) >> 2;
    const int cs = q8x3_chunk_steps(nst);
    const int j = t >> 3;
    const int l = t & 7;
    const bool active = rbase < M;

    float acc[Q8X3_W][Q8X3_NB];
#pragma unroll
    for (int r = 0; r < Q8X3_W; r++)
#pragma unroll
        for (int b = 0; b < Q8X3_NB; b++) acc[r][b] = 0.0f;

    // Rows past M re-read row M - 1 (never stored).
    const int rfirst = min(rbase, M - 1);
    const char* rb0 = A + (long long)rfirst * gpr * 136;
    int roff[Q8X3_W];
#pragma unroll
    for (int r = 0; r < Q8X3_W; r++) roff[r] = (min(rbase + r, M - 1) - rfirst) * gpr * 136;
    const int lw = 8 + 16 * l;
    const int lh = 4 * (l >> 2);

    // Weight ring, prologue first so the DRAM latency overlaps the staging.
    q8x3_u4 wr[Q8X3_P][Q8X3_W];
    unsigned int hr[Q8X3_P][Q8X3_W];
#pragma unroll
    for (int p = 0; p < Q8X3_P; p++) {
        const int g = min(4 * min(p, nst - 1) + j, gpr - 1);
#pragma unroll
        for (int r = 0; r < Q8X3_W; r++) {
            wr[p][r] = q8x3_ldw(rb0 + roff[r] + g * 136 + lw);
            hr[p][r] = q8x3_ldh(rb0 + roff[r] + g * 136 + lh);
        }
    }
    asm volatile("" ::: "memory");

    for (int c0 = 0; c0 < nst; c0 += cs) {
        const int c1 = min(c0 + cs, nst);
        if (c0 > 0) __syncthreads();
        q8x3_stage(xq, dx, gpr, c0, c1, cs);
        __syncthreads();
        if (active) {
            auto step = [&](const int st, const int slot) {
                const int g = 4 * st + j;
                const bool gv = g < gpr;
                const int lg = min(g, gpr - 1) - 4 * c0;      // group within the chunk
                q8x3_u4 xa[Q8X3_NB], xb[Q8X3_NB];
                float2 dd[Q8X3_NB];
                q8x3_ldsx(xa, xb, dd, cs, lg, l);
#pragma unroll
                for (int r = 0; r < Q8X3_W; r++) {
                    const q8x3_u4 w = wr[slot][r];
                    const unsigned int h = hr[slot][r];
                    const float sc = gv ? __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu))) : 0.0f;
                    const float zp = gv ? __half2float(__ushort_as_half((unsigned short)(h >> 16))) : 0.0f;
                    const int q[8] = {
                        (int)(w.x & 0x0F0F0F0Fu), (int)((w.x >> 4) & 0x0F0F0F0Fu),
                        (int)(w.y & 0x0F0F0F0Fu), (int)((w.y >> 4) & 0x0F0F0F0Fu),
                        (int)(w.z & 0x0F0F0F0Fu), (int)((w.z >> 4) & 0x0F0F0F0Fu),
                        (int)(w.w & 0x0F0F0F0Fu), (int)((w.w >> 4) & 0x0F0F0F0Fu),
                    };
#pragma unroll
                    for (int b = 0; b < Q8X3_NB; b++) {
                        int s = __builtin_amdgcn_sudot4(false, q[0], true, (int)xa[b].x, 0, false);
                        s = __builtin_amdgcn_sudot4(false, q[1], true, (int)xa[b].y, s, false);
                        s = __builtin_amdgcn_sudot4(false, q[2], true, (int)xa[b].z, s, false);
                        s = __builtin_amdgcn_sudot4(false, q[3], true, (int)xa[b].w, s, false);
                        s = __builtin_amdgcn_sudot4(false, q[4], true, (int)xb[b].x, s, false);
                        s = __builtin_amdgcn_sudot4(false, q[5], true, (int)xb[b].y, s, false);
                        s = __builtin_amdgcn_sudot4(false, q[6], true, (int)xb[b].z, s, false);
                        s = __builtin_amdgcn_sudot4(false, q[7], true, (int)xb[b].w, s, false);
                        acc[r][b] += (dd[b].x * sc) * (float)s + zp * dd[b].y;
                    }
                }
                const int gn = min(4 * min(st + Q8X3_P, nst - 1) + j, gpr - 1);
#pragma unroll
                for (int r = 0; r < Q8X3_W; r++) {
                    wr[slot][r] = q8x3_ldw(rb0 + roff[r] + gn * 136 + lw);
                    hr[slot][r] = q8x3_ldh(rb0 + roff[r] + gn * 136 + lh);
                }
                asm volatile("" ::: "memory");
            };
            // Chunks start at multiples of cs (a multiple of P, or 0), so the
            // ring slot of step st + p is p.
            int st = c0;
            for (; st + Q8X3_P <= c1; st += Q8X3_P) {
#pragma unroll
                for (int p = 0; p < Q8X3_P; p++) step(st + p, p);
            }
#pragma unroll
            for (int p = 0; p + 1 < Q8X3_P; p++) {
                if (st + p < c1) step(st + p, p);
            }
        }
    }

    float v[Q8X3_VP];
#pragma unroll
    for (int i = 0; i < Q8X3_VP; i++) v[i] = i < Q8X3_V ? acc[i / Q8X3_NB][i % Q8X3_NB] : 0.0f;
    const float tot = q8x_reduce<Q8X3_LVP>(v, t);
    const int idx = t >> (5 - Q8X3_LVP);
    if ((t & ((1 << (5 - Q8X3_LVP)) - 1)) == 0 && idx < Q8X3_V) {
        const int row = rbase + idx / Q8X3_NB;
        if (row < M) {
            float* yp = y + (long long)(idx % Q8X3_NB) * M + row;
#ifdef Q8X_RESIDUAL
            *yp += tot;
#else
            *yp = tot;
#endif
        }
    }
}

__launch_bounds__(Q8X3_T)
extern "C" __global__ void Q8X_KERNEL(
    const char* __restrict__ A,
    const unsigned int* __restrict__ xq,
    const float* __restrict__ dx,
    float* __restrict__ y,
    int M, int K, int B)
{
    (void)B;
    q8x3_rows(A, xq, dx, y, M, K, blockIdx.x * Q8X3_ROWS);
}

__launch_bounds__(Q8X3_T)
extern "C" __global__ void Q8X_MULTI_KERNEL(
    const char* __restrict__ A0, float* __restrict__ y0, int M0,
    const char* __restrict__ A1, float* __restrict__ y1, int M1,
    const char* __restrict__ A2, float* __restrict__ y2, int M2,
    const char* __restrict__ A3, float* __restrict__ y3, int M3,
    const unsigned int* __restrict__ xq,
    const float* __restrict__ dx,
    int K, int B)
{
    (void)B;
    int blk = blockIdx.x;
    const char* A = A0;
    float* y = y0;
    int M = M0;
    const int n0 = (M0 + Q8X3_ROWS - 1) / Q8X3_ROWS;
    const int n1 = (M1 + Q8X3_ROWS - 1) / Q8X3_ROWS;
    const int n2 = (M2 + Q8X3_ROWS - 1) / Q8X3_ROWS;
    if (blk >= n0) {
        blk -= n0;
        A = A1; y = y1; M = M1;
        if (blk >= n1) {
            blk -= n1;
            A = A2; y = y2; M = M2;
            if (blk >= n2) {
                blk -= n2;
                A = A3; y = y3; M = M3;
            }
        }
    }
    q8x3_rows(A, xq, dx, y, M, K, blk * Q8X3_ROWS);
}
#endif

#ifdef Q8X_GEMV4
// v4: persistent v3. grid = G workgroups (the launcher sizes G to the
// resident capacity); workgroup w owns rows [total*w/G, total*(w+1)/G) of the
// concatenated weights (rows of A0, then A1, ...), stages the activations
// once (all of K; the launcher only picks v4 when they fit in LDS), and its R
// warps split the workgroup's (row, 4-group step) pairs into equal contiguous
// ranges. A warp streams its range one row at a time with the weight ring
// running across row boundaries (no per-row pipeline refill, no tail of idle
// warps). A row split between warp i-1 (tail) and warp i (head) is finished
// after a barrier as y (+)= tail + head in that fixed order, so results are
// deterministic. Rows per workgroup >= R keeps every range >= one row, so a
// row is split at most once.
#ifndef Q8X4_XP
#define Q8X4_XP 0
#endif
#define Q8X4_V Q8X3_NB
#define Q8X4_LVP (Q8X4_V <= 1 ? 0 : Q8X4_V <= 2 ? 1 : Q8X4_V <= 4 ? 2 : 3)
#define Q8X4_VP (1 << Q8X4_LVP)

// The (up to four) weights sharing the activation are passed as separate
// scalars (not arrays in a struct: that lands in scratch memory and turns the
// weight loads into flat loads). Row r of the concatenated weights maps to
// (matrix, row in it) by compare/select, branch-free.
#define Q8X4_MATS                                                           \
    const char* __restrict__ A0, float* __restrict__ y0, int M0,            \
    const char* __restrict__ A1, float* __restrict__ y1, int M1,            \
    const char* __restrict__ A2, float* __restrict__ y2, int M2,            \
    const char* __restrict__ A3, float* __restrict__ y3, int M3
#define Q8X4_MATS_ARGS A0, y0, M0, A1, y1, M1, A2, y2, M2, A3, y3, M3

__device__ __forceinline__ const char* q8x4_rowptr(Q8X4_MATS, int r, int rowbytes) {
    (void)y0; (void)y1; (void)y2; (void)y3; (void)M3;
    const int c1 = M0, c2 = c1 + M1, c3 = c2 + M2;
    const char* a = r >= c3 ? A3 : r >= c2 ? A2 : r >= c1 ? A1 : A0;
    const int lr = r - (r >= c3 ? c3 : r >= c2 ? c2 : r >= c1 ? c1 : 0);
    return a + (long long)lr * rowbytes;
}

__device__ __forceinline__ void q8x4_store(Q8X4_MATS, int r, int b, float v) {
    (void)A0; (void)A1; (void)A2; (void)A3;
    const int c1 = M0, c2 = c1 + M1, c3 = c2 + M2;
    float* y = r >= c3 ? y3 : r >= c2 ? y2 : r >= c1 ? y1 : y0;
    const int m = r >= c3 ? M3 : r >= c2 ? M2 : r >= c1 ? M1 : M0;
    const int lr = r - (r >= c3 ? c3 : r >= c2 ? c2 : r >= c1 ? c1 : 0);
    float* yp = y + (long long)b * m + lr;
#ifdef Q8X_RESIDUAL
    *yp += v;
#else
    *yp = v;
#endif
}

__device__ __forceinline__ void q8x4_run(
    Q8X4_MATS,
    const unsigned int* __restrict__ xq,
    const float* __restrict__ dx,
    int K)
{
    __shared__ float part[Q8X3_R][2][Q8X4_V];     // [warp][head, tail][b]
    const int tid = threadIdx.x;
    const int t = tid & 31;
    const int wv = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int gpr = K >> 8;
    const int nst = (gpr + 3) >> 2;
    const int rowbytes = gpr * 136;
    const int j = t >> 3;
    const int l = t & 7;
    const long long total = (long long)M0 + M1 + M2 + M3;
    const int G = gridDim.x;
    const int wg = blockIdx.x;
    const int r0 = (int)(total * wg / G);
    const int r1 = (int)(total * (wg + 1) / G);
    const int T = (r1 - r0) * nst;
    const int ua = r0 * nst + (int)((long long)T * wv / Q8X3_R);
    const int ub = r0 * nst + (int)((long long)T * (wv + 1) / Q8X3_R);
    const bool any = ua < ub;

    // Prefetch cursor: flat step pu = prow * nst + pst, row pointer pbase;
    // it stops at ub - 1 (the ring tail re-reads that step).
    int prow = ua / nst;
    int pst = ua - prow * nst;
    int pu = ua;
    const char* pbase = q8x4_rowptr(Q8X4_MATS_ARGS, any ? prow : 0, rowbytes);
    q8x3_u4 wr[Q8X3_P];
    unsigned int hr[Q8X3_P];
    auto issue = [&](const int slot) {
        const int gl = min(4 * pst + j, gpr - 1);
        const char* p = pbase + gl * 136;
        wr[slot] = q8x3_ldw(p + 8 + 16 * l);
        hr[slot] = q8x3_ldh(p + 4 * (l >> 2));
        const bool adv = pu + 1 < ub;
        const bool wrap = adv && pst + 1 == nst;
        pu += adv ? 1 : 0;
        pst = wrap ? 0 : pst + (adv ? 1 : 0);
        prow += wrap ? 1 : 0;
        pbase = q8x4_rowptr(Q8X4_MATS_ARGS, prow, rowbytes);
    };
    if (any) {
#pragma unroll
        for (int p = 0; p < Q8X3_P; p++) issue(p);
    }
    asm volatile("" ::: "memory");

    // Stage all of K once (single chunk).
    q8x3_stage(xq, dx, gpr, 0, nst, nst);
    __syncthreads();

    float acc[Q8X3_NB];
#pragma unroll
    for (int b = 0; b < Q8X3_NB; b++) acc[b] = 0.0f;
    float headv = 0.0f, tailv = 0.0f;
    bool has_head = false;
    int crow = ua / nst;
    int cst = ua - crow * nst;

#if Q8X4_XP
    // LDS activation double buffer: step i of the range reads lx*[i & 1]; the
    // next step's reads issue before this step's math (LDS completes in
    // order, so waiting on this step's reads never waits on the next's).
    q8x3_u4 lxa[2][Q8X3_NB], lxb[2][Q8X3_NB];
    float2 ldd[2][Q8X3_NB];
    if (any) q8x3_ldsx(lxa[0], lxb[0], ldd[0], nst, min(4 * cst + j, gpr - 1), l);
#endif
    auto step = [&](const int u, const int slot, const int buf) {
        const int g = 4 * cst + j;
        const bool gv = g < gpr;
#if Q8X4_XP
        {
            const int nx = cst + 1 == nst ? 0 : cst + 1;
            q8x3_ldsx(lxa[buf ^ 1], lxb[buf ^ 1], ldd[buf ^ 1], nst, min(4 * nx + j, gpr - 1), l);
        }
        const q8x3_u4 (&xa)[Q8X3_NB] = lxa[buf];
        const q8x3_u4 (&xb)[Q8X3_NB] = lxb[buf];
        const float2 (&dd)[Q8X3_NB] = ldd[buf];
#else
        (void)buf;
        const int lg = min(g, gpr - 1);
        q8x3_u4 xa[Q8X3_NB], xb[Q8X3_NB];
        float2 dd[Q8X3_NB];
        q8x3_ldsx(xa, xb, dd, nst, lg, l);
#endif
        const q8x3_u4 w = wr[slot];
        const unsigned int h = hr[slot];
        const float sc = gv ? __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu))) : 0.0f;
        const float zp = gv ? __half2float(__ushort_as_half((unsigned short)(h >> 16))) : 0.0f;
        const int q[8] = {
            (int)(w.x & 0x0F0F0F0Fu), (int)((w.x >> 4) & 0x0F0F0F0Fu),
            (int)(w.y & 0x0F0F0F0Fu), (int)((w.y >> 4) & 0x0F0F0F0Fu),
            (int)(w.z & 0x0F0F0F0Fu), (int)((w.z >> 4) & 0x0F0F0F0Fu),
            (int)(w.w & 0x0F0F0F0Fu), (int)((w.w >> 4) & 0x0F0F0F0Fu),
        };
#pragma unroll
        for (int b = 0; b < Q8X3_NB; b++) {
            int s = __builtin_amdgcn_sudot4(false, q[0], true, (int)xa[b].x, 0, false);
            s = __builtin_amdgcn_sudot4(false, q[1], true, (int)xa[b].y, s, false);
            s = __builtin_amdgcn_sudot4(false, q[2], true, (int)xa[b].z, s, false);
            s = __builtin_amdgcn_sudot4(false, q[3], true, (int)xa[b].w, s, false);
            s = __builtin_amdgcn_sudot4(false, q[4], true, (int)xb[b].x, s, false);
            s = __builtin_amdgcn_sudot4(false, q[5], true, (int)xb[b].y, s, false);
            s = __builtin_amdgcn_sudot4(false, q[6], true, (int)xb[b].z, s, false);
            s = __builtin_amdgcn_sudot4(false, q[7], true, (int)xb[b].w, s, false);
            acc[b] += (dd[b].x * sc) * (float)s + zp * dd[b].y;
        }
        issue(slot);
        asm volatile("" ::: "memory");
        // End of a row segment (the row or this warp's range ends here). The
        // branch issues no loads, so it does not disturb the vmcnt state.
        const bool row_end = cst == nst - 1;
        if (row_end || u == ub - 1) {
            float v[Q8X4_VP];
#pragma unroll
            for (int i = 0; i < Q8X4_VP; i++) v[i] = i < Q8X4_V ? acc[i] : 0.0f;
            const float tot = q8x_reduce<Q8X4_LVP>(v, t);
            const bool head_part = (u - cst) < ua;         // row began before this range
            if (head_part) {
                headv = tot;
                has_head = true;
            } else if (!row_end) {
                tailv = tot;                               // row continues past this range
            } else {
                const int idx = t >> (5 - Q8X4_LVP);
                if ((t & ((1 << (5 - Q8X4_LVP)) - 1)) == 0 && idx < Q8X4_V)
                    q8x4_store(Q8X4_MATS_ARGS, crow, idx, tot);
            }
#pragma unroll
            for (int b = 0; b < Q8X3_NB; b++) acc[b] = 0.0f;
        }
        const bool cwrap = cst + 1 == nst;
        cst = cwrap ? 0 : cst + 1;
        crow += cwrap ? 1 : 0;
    };

    if (any) {
        // Unrolled by UNR (a multiple of P, even with XP); ring slot p % P,
        // LDS buffer p & 1. The remainder is peeled so the loop back-edge
        // carries one exact wait state.
        constexpr int UNR = ((Q8X4_XP != 0) && ((Q8X3_P & 1) != 0)) ? 2 * Q8X3_P : Q8X3_P;
        int u = ua;
        for (; u + UNR <= ub; u += UNR) {
#pragma unroll
            for (int p = 0; p < UNR; p++) step(u + p, p % Q8X3_P, p & 1);
        }
#pragma unroll
        for (int p = 0; p + 1 < UNR; p++) {
            if (u + p < ub) step(u + p, p % Q8X3_P, p & 1);
        }
    }

    // Split rows: warp i's head continues warp i-1's tail (fixed order).
    const int idx = t >> (5 - Q8X4_LVP);
    const bool writer = (t & ((1 << (5 - Q8X4_LVP)) - 1)) == 0 && idx < Q8X4_V;
    if (writer) {
        part[wv][0][idx] = headv;
        part[wv][1][idx] = tailv;
    }
    __syncthreads();
    if (has_head && writer) q8x4_store(Q8X4_MATS_ARGS, ua / nst, idx, part[wv - 1][1][idx] + headv);
}

__launch_bounds__(Q8X3_T)
extern "C" __global__ void Q8X_KERNEL(
    const char* __restrict__ A,
    const unsigned int* __restrict__ xq,
    const float* __restrict__ dx,
    float* __restrict__ y,
    int M, int K, int B)
{
    (void)B;
    q8x4_run(A, y, M, A, y, 0, A, y, 0, A, y, 0, xq, dx, K);
}

__launch_bounds__(Q8X3_T)
extern "C" __global__ void Q8X_MULTI_KERNEL(
    Q8X4_MATS,
    const unsigned int* __restrict__ xq,
    const float* __restrict__ dx,
    int K, int B)
{
    (void)B;
    q8x4_run(Q8X4_MATS_ARGS, xq, dx, K);
}
#endif