// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Chunked (C64) generic gated delta rule with per-key-channel diagonal log decay
// (the channel-wise gated-DeltaNet / KDA family; Raon RDA). gfx1201, head_dim 128.
// Two kernels, no atomics, no per-launch allocation, deterministic:
//
//   gated_delta_vector_chunk_prep_gfx12   chunk-parallel, persistent grid striding over the (chunk, head)
//                                         pairs, block 256
//   gated_delta_vector_chunk_scan_gfx12   state carry, grid (head * 8 value tiles), block 32
//
// Serial reference (gated_delta_net_vector_serial.hip), per token t:
//   S <- S diag(exp(a_t));  E_t = v_t - S k_t;  S <- S + beta_t E_t k_t^T;  y_t = S q_t.
//
// Within a chunk of C=64 tokens let ell_i = sum_{r<=i} a_r (per key channel, reset at
// the chunk start), Et_i = beta_i E_i, G_ij = sum_c k_ic k_jc exp(ell_ic-ell_jc) (i>j),
// M_ij = sum_c q_ic k_jc exp(ell_ic-ell_jc) (j<=i). With S0 the chunk-entry state:
//   (I + diag(beta) G) Et = diag(beta) (V - Kdec S0^T),   Kdec = K * exp(ell)
//   Y   = Qdec S0^T + M Et,                               Qdec = Q * exp(ell)
//   S_T = S0 diag(exp(ell_T)) + Et^T Kend,                Kend_j = k_j * exp(ell_T - ell_j)
// With T' = (I + diag(beta) G)^-1:  U' = T' beta V,  W' = T' beta Kdec,
//   Et = U' - W' S0^T,   Y = M U' + (Qdec - M W') S0^T.
// prep computes {-W' (hi, lo), Q' = Qdec - M W', Kend^T (hi, lo), U'^T (f32), M, exp(ell_T)} per
// (chunk, head); scan walks the chunks of a head keeping its [V tile, K]
// state in registers (FP32) and applying the four WMMA stages per chunk.
//
// Numerics (see METHODOLOGY-AUDIT.md sec. 7): ell is a log-space cumulative sum reset per
// chunk; every WMMA/f16 operand and every f32 exponent is a CAUSAL RELATIVE exponent
// (<= 0). No reciprocal / exp(-global ell) is ever formed. Pair matrices G, M are built in
// f32 in 16-row sub-blocks: blocks I > J use the first row of I as reference (both
// factors <= 1), diagonal blocks use the middle row (factors bounded by exp(8*|a|max)).
// Per-token log decay is clamped to [-10, 0] so the diagonal factor products stay inside
// the f32 range (Raon's gate is -5*sigmoid, always inside). Pair matrices use bf16 hi+lo
// operands (hi*hi + hi*lo + lo*hi, ~2^-16); L = diag(beta) G is stored hi+lo f16 (packed
// 16x16 tiles, ~2^-22). Everything on the FP32-state path is split hi+lo f16 with f32 WMMA
// accumulation: T' (block solve + merges at f16, then one Newton-Schulz step against the
// exact L), beta K Kdec and beta V, U' (FP32), W' (-W' hi and lo planes), Kend (hi and lo
// planes), the entry state in E = U' - W' S0^T, and E in the state update. A single f16
// rounding of any of these costs 2e-4..5e-4 relative state error (measured by a CPU emulation
// of this dataflow on the captured layers); only the output-side operands (M, Q', Wf in
// Q' = Qdec - M W', U' and Q' in Y) stay plain f16.
//
// WMMA K-slot mapping: lane (lo = lane&15, hi = lane>>4) holds K slots 8*hi+e, e=0..7, of
// row (A) / column (B) lo; the f32 accumulator element e of lane (lo, hi) is row 8*hi+e of
// column lo. An accumulator tile is therefore directly a B operand over its row index.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define CK_C 64
#define CK_D 128
#define CK_EP 132  // f32 pitch of the chunk-relative cumulative log decay (bank-conflict free fragment reads)
#define CK_GP 68   // f16 pitch used by the (reused) UT solve helpers
#define CK_AMIN (-10.0f)
#define CK_TILES 10
__device__ __forceinline__ int ck_tidx(int m, int n) { return m * (m + 1) / 2 + n; }
__device__ __forceinline__ unsigned short& ck_el(unsigned short* a, int r, int c) {
    return a[ck_tidx(r >> 4, c >> 4) * 256 + (r & 15) * 16 + (c & 15)];
}
#define CK_WG 256   // prepare workgroup threads (8 waves; the 10 block pairs run as 8 + 2)

// scratch layout per (chunk, head) block (bytes)
#define CK_OFF_WN 0u        // [8 ntile][64 tok][16] f16  -W'
#define CK_OFF_QP 16384u    // [8 ntile][64 tok][16] f16  Q'
#define CK_OFF_KT 32768u    // [128 c][64 tok] f16       Kend^T, high half (f16)
#define CK_OFF_KL 49152u    // [128 c][64 tok] f16       Kend^T, low half: f16(Kend - hi)
#define CK_OFF_UT 65536u    // [128 v][64 tok] f32       U'^T (FP32: it seeds Et and so the state)
#define CK_OFF_M 98304u     // [64][64] f16              M
#define CK_OFF_DT 106496u   // [128] f32                 exp(ell_T)
#define CK_OFF_WL 107008u   // [8 ntile][64 tok][16] f16  -W' low half: f16(-W' - hi)
#define CK_BLK 123392ull

typedef _Float16 ck_h8 __attribute__((ext_vector_type(8)));
typedef float ck_f8 __attribute__((ext_vector_type(8)));
typedef short ck_s8 __attribute__((ext_vector_type(8)));  // 8 bf16 bit patterns

__device__ __forceinline__ ck_f8 ck_z8() { return (ck_f8){0, 0, 0, 0, 0, 0, 0, 0}; }
__device__ __forceinline__ ck_f8 ck_mma(ck_h8 a, ck_h8 b, ck_f8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}
__device__ __forceinline__ ck_f8 ck_mmab(ck_s8 a, ck_s8 b, ck_f8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
// bf16 round-to-nearest-even bit pattern of a finite float
__device__ __forceinline__ unsigned ck_bf16_bits(float x) {
    unsigned u = __float_as_uint(x);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
// Split 8 floats into bf16 hi (truncated) + lo (truncated residual): x = hi + lo + O(2^-16 |x|). One
// v_perm per pair of elements instead of a rounding sequence per element.
__device__ __forceinline__ void ck_split_bf16(const float (&x)[8], ck_s8& hi, ck_s8& lo) {
    unsigned h[4], l[4], hb[8], lb[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        hb[e] = __float_as_uint(x[e]);
        lb[e] = __float_as_uint(x[e] - __uint_as_float(hb[e] & 0xffff0000u));
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        h[p] = __builtin_amdgcn_perm(hb[2 * p + 1], hb[2 * p], 0x07060302u);
        l[p] = __builtin_amdgcn_perm(lb[2 * p + 1], lb[2 * p], 0x07060302u);
    }
    hi = __builtin_bit_cast(ck_s8, make_uint4(h[0], h[1], h[2], h[3]));
    lo = __builtin_bit_cast(ck_s8, make_uint4(l[0], l[1], l[2], l[3]));
}
__device__ __forceinline__ ck_h8 ck_ld16(const void* p) {
    return __builtin_bit_cast(ck_h8, *reinterpret_cast<const uint4*>(p));
}
__device__ __forceinline__ ck_h8 ck_pack(ck_f8 a) {
    ck_h8 r;
#pragma unroll
    for (int e = 0; e < 8; ++e) r[e] = (_Float16)a[e];
    return r;
}
__device__ __forceinline__ unsigned short ck_h16(float x) { return __half_as_ushort(__float2half_rn(x)); }
__device__ __forceinline__ float ck_h2f(unsigned short x) { return __half2float(__ushort_as_half(x)); }
__device__ __forceinline__ float ck_bf16_rn(float x) {
    unsigned u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}
__device__ __forceinline__ int ck_acc_row(int lane, int e) { return 8 * (lane >> 4) + e; }

// ---- incumbent UT-solve fragment helpers (gdn_chunk_kkt_solve.gfx11.hip, gfx12 mapping) ----
__device__ __forceinline__ ck_h8 ck_frag_shared(const unsigned short* p, int pitch, int row0, int k0, int lane) {
    union { unsigned short h[8]; ck_h8 v; } u;
    const int row = row0 + (lane & 15);
    const int kh = k0 + 4 * (lane >> 4);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        u.h[e] = p[row * pitch + kh + e];
        u.h[e + 4] = p[row * pitch + kh + 8 + e];
    }
    return u.v;
}
__device__ __forceinline__ ck_h8 ck_frag_transpose(const unsigned short* p, int pitch, int row0, int k0, int lane) {
    union { unsigned short h[8]; ck_h8 v; } u;
    const int row = row0 + (lane & 15);
    const int kh = k0 + 4 * (lane >> 4);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        u.h[e] = p[(kh + e) * pitch + row];
        u.h[e + 4] = p[(kh + 8 + e) * pitch + row];
    }
    return u.v;
}

struct __attribute__((aligned(16))) CkLds {
    float ell[CK_C][CK_EP];                // 33792  chunk-relative cumulative log decay
    // 64x64 lower-triangular matrices, packed as the 10 lower 16x16 tiles (tile (m, n), n <= m, at
    // m (m + 1) / 2 + n, row-major, pitch 16): every tile any stage reads is lower.
    unsigned short Lh[CK_TILES * 256];     //  5120  strict-lower diag(beta) G, high f16
    unsigned short Ll[CK_TILES * 256];     //  5120  low f16: L = Lh + Ll to ~2^-22
    unsigned short Th[CK_TILES * 256];     //  5120  T' = (I + L)^-1: block-solve copy, then the high half
    unsigned short Tl[CK_TILES * 256];     //  5120  zero, then the refinement residual R, then T' low half
    union {
        unsigned short tmp[32][36];                                                    //  2304
        unsigned short stage[8][256];                                                  //  4096
    } u;
    float beta[CK_C];                      //   256
};

extern "C" __launch_bounds__(CK_WG)
__global__ void gated_delta_vector_chunk_prep_gfx12(
    const float* __restrict__ query,    // [rows, Hqk, K] at q_rs
    const float* __restrict__ key,
    const float* __restrict__ value,    // [rows, H, V]
    const float* __restrict__ log_decay,// PerKey [rows,H,K]; PerHead [rows,H]
    const float* __restrict__ beta,     // [rows, H]
    unsigned char* __restrict__ scratch,
    int seg_rows, int n_heads, int qk_group, int decay_per_key,
    int q_rs, int k_rs, int v_rs, int a_rs, int b_rs,
    unsigned long long* __restrict__ prof)   // nullable debug timestamps (10 ns ticks)
{
    const int tid = (int)threadIdx.x, lane = tid & 31, lo = lane & 15, hi = lane >> 4;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    __shared__ CkLds sm;
    const int n_chunks = (seg_rows + CK_C - 1) / CK_C;
    const int total = n_chunks * n_heads;
    // Persistent grid: each workgroup walks chunk-heads b, b + G, ... so the per-workgroup launch
    // cost is paid once.
    for (int ch = (int)blockIdx.x; ch < total; ch += (int)gridDim.x) {
    const int chunk = ch / n_heads, h = ch - chunk * n_heads;
    const int t0 = chunk * CK_C;
    int rows = seg_rows - t0;
    if (rows > CK_C) rows = CK_C;
    const int hq = h / qk_group;

    const float* qb = query + (size_t)t0 * q_rs + (size_t)hq * CK_D;
    const float* kb = key + (size_t)t0 * k_rs + (size_t)hq * CK_D;
    const float* vb = value + (size_t)t0 * v_rs + (size_t)h * CK_D;
    unsigned char* blk = scratch + ((size_t)chunk * n_heads + h) * CK_BLK;

    unsigned long long* const pf = prof ? prof + ((size_t)chunk * n_heads + h) * 8 : nullptr;
#define CK_STAMP(i) do { if (pf && tid == 0) pf[i] = (unsigned long long)wall_clock64(); } while (0)
    CK_STAMP(0);

    // ---- P0: zero Lh/Ll/Tl and the strictly-upper block tiles of M (global), chunk-relative cumulative log decay, beta ----
    for (int x = tid; x < CK_TILES * 256 / 8; x += CK_WG) {
        reinterpret_cast<uint4*>(&sm.Lh[0])[x] = make_uint4(0, 0, 0, 0);
        reinterpret_cast<uint4*>(&sm.Ll[0])[x] = make_uint4(0, 0, 0, 0);
        reinterpret_cast<uint4*>(&sm.Tl[0])[x] = make_uint4(0, 0, 0, 0);
    }
    for (int x = tid; x < 512; x += CK_WG) {
        const int r = x >> 3, cb = (x & 7) >> 1;
        if (cb > (r >> 4)) *reinterpret_cast<uint4*>(blk + CK_OFF_M + r * 128 + (x & 7) * 16) = make_uint4(0, 0, 0, 0);
    }
    // Thread (c4 = tid & 31, g = tid >> 5) scans 8 tokens x 4 channels with float4 loads; the eight
    // group totals (LDS, reusing the stage union) are added to the later groups after a barrier.
    float* const gt = reinterpret_cast<float*>(&sm.u.stage[0][0]);   // [8][128]
    if (tid < 256) {
        const int c4 = tid & 31, g = tid >> 5;
        float4 acc4 = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 av[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int t = g * 8 + r;
            float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
            if (t < rows) {
                if (decay_per_key) {
                    x = *reinterpret_cast<const float4*>(&log_decay[(size_t)(t0 + t) * a_rs + (size_t)h * CK_D + 4 * c4]);
                } else {
                    const float s1 = log_decay[(size_t)(t0 + t) * a_rs + h];
                    x = make_float4(s1, s1, s1, s1);
                }
                x.x = fminf(fmaxf(x.x, CK_AMIN), 0.0f); x.y = fminf(fmaxf(x.y, CK_AMIN), 0.0f);
                x.z = fminf(fmaxf(x.z, CK_AMIN), 0.0f); x.w = fminf(fmaxf(x.w, CK_AMIN), 0.0f);
            }
            av[r] = x;
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            acc4.x += av[r].x; acc4.y += av[r].y; acc4.z += av[r].z; acc4.w += av[r].w;
            *reinterpret_cast<float4*>(&sm.ell[g * 8 + r][4 * c4]) = acc4;
        }
        *reinterpret_cast<float4*>(&gt[g * 128 + 4 * c4]) = acc4;
    }
    if (tid < CK_C) sm.beta[tid] = tid < rows ? beta[(size_t)(t0 + tid) * b_rs + h] : 0.0f;
    __syncthreads();
    if (tid < 256) {
        const int c4 = tid & 31, g = tid >> 5;
        float4 base = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int g2 = 0; g2 < g; ++g2) {
            const float4 t4 = *reinterpret_cast<const float4*>(&gt[g2 * 128 + 4 * c4]);
            base.x += t4.x; base.y += t4.y; base.z += t4.z; base.w += t4.w;
        }
        if (g > 0) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                float4* e4 = reinterpret_cast<float4*>(&sm.ell[g * 8 + r][4 * c4]);
                float4 v4 = *e4;
                v4.x += base.x; v4.y += base.y; v4.z += base.z; v4.w += base.w;
                *e4 = v4;
            }
        }
    }
    __syncthreads();
    CK_STAMP(1);

    // ---- P1: pair matrices ----
    // Wave w owns block pairs p = w and w + 8 (I, J <= I, p = I (I + 1) / 2 + J): G (k.k) and M (q.k) tiles by bf16
    // hi+lo WMMA (hi*hi + hi*lo + lo*hi, ~2^-16 relative) over the 128 channels. Rows i in I use
    // factor exp(ell_i - ell_ref), rows j in J exp(ell_ref - ell_j); ref is the first row of I
    // (I > J, both exponents <= 0) or the middle row of I (diagonal, bounded by exp(8 |a|max)).
    for (int pr = wave; pr < 10; pr += CK_WG / 32) {
        const int I = pr >= 6 ? 3 : pr >= 3 ? 2 : pr >= 1 ? 1 : 0;
        const int J = pr - I * (I + 1) / 2;
        const int refrow = (I == J) ? 16 * I + 8 : 16 * I;
        const int ti = 16 * I + lo, tj = 16 * J + lo;
        const bool vi = ti < rows, vj = tj < rows;
        const float* kri = kb + (size_t)ti * k_rs;
        const float* qri = qb + (size_t)ti * q_rs;
        const float* krj = kb + (size_t)tj * k_rs;
        ck_f8 accG = ck_z8(), accM = ck_z8();
        float4 nk0 = make_float4(0, 0, 0, 0), nk1 = nk0, nq0 = nk0, nq1 = nk0, nj0 = nk0, nj1 = nk0;
        {
            const int c0 = 8 * hi;
            if (vi) {
                nk0 = *reinterpret_cast<const float4*>(kri + c0); nk1 = *reinterpret_cast<const float4*>(kri + c0 + 4);
                nq0 = *reinterpret_cast<const float4*>(qri + c0); nq1 = *reinterpret_cast<const float4*>(qri + c0 + 4);
            }
            if (vj) {
                nj0 = *reinterpret_cast<const float4*>(krj + c0); nj1 = *reinterpret_cast<const float4*>(krj + c0 + 4);
            }
        }
#pragma unroll 1
        for (int kk = 0; kk < 8; ++kk) {
            const float4 ck0 = nk0, ck1 = nk1, cq0 = nq0, cq1 = nq1, cj0 = nj0, cj1 = nj1;
            if (kk < 7) {
                const int c0 = 16 * (kk + 1) + 8 * hi;
                nk0 = nk1 = nq0 = nq1 = nj0 = nj1 = make_float4(0, 0, 0, 0);
                if (vi) {
                    nk0 = *reinterpret_cast<const float4*>(kri + c0); nk1 = *reinterpret_cast<const float4*>(kri + c0 + 4);
                    nq0 = *reinterpret_cast<const float4*>(qri + c0); nq1 = *reinterpret_cast<const float4*>(qri + c0 + 4);
                }
                if (vj) {
                    nj0 = *reinterpret_cast<const float4*>(krj + c0); nj1 = *reinterpret_cast<const float4*>(krj + c0 + 4);
                }
            }
            const int c0 = 16 * kk + 8 * hi;
            const float kv[8] = {ck0.x, ck0.y, ck0.z, ck0.w, ck1.x, ck1.y, ck1.z, ck1.w};
            const float qv[8] = {cq0.x, cq0.y, cq0.z, cq0.w, cq1.x, cq1.y, cq1.z, cq1.w};
            const float jv[8] = {cj0.x, cj0.y, cj0.z, cj0.w, cj1.x, cj1.y, cj1.z, cj1.w};
            const float4 e0 = *reinterpret_cast<const float4*>(&sm.ell[refrow][c0]);
            const float4 e1 = *reinterpret_cast<const float4*>(&sm.ell[refrow][c0 + 4]);
            const float4 i0 = *reinterpret_cast<const float4*>(&sm.ell[ti][c0]);
            const float4 i1 = *reinterpret_cast<const float4*>(&sm.ell[ti][c0 + 4]);
            const float4 j0 = *reinterpret_cast<const float4*>(&sm.ell[tj][c0]);
            const float4 j1 = *reinterpret_cast<const float4*>(&sm.ell[tj][c0 + 4]);
            const float er[8] = {e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w};
            const float ei[8] = {i0.x, i0.y, i0.z, i0.w, i1.x, i1.y, i1.z, i1.w};
            const float ej[8] = {j0.x, j0.y, j0.z, j0.w, j1.x, j1.y, j1.z, j1.w};
            float xk[8], xq[8], yv[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float pf = __expf(ei[e] - er[e]);
                xk[e] = kv[e] * pf;
                xq[e] = qv[e] * pf;
                yv[e] = jv[e] * __expf(er[e] - ej[e]);
            }
            ck_s8 kh, kl, qh, ql, yh, yl;
            ck_split_bf16(xk, kh, kl);
            ck_split_bf16(xq, qh, ql);
            ck_split_bf16(yv, yh, yl);
            accG = ck_mmab(kh, yh, accG);
            accG = ck_mmab(kh, yl, accG);
            accG = ck_mmab(kl, yh, accG);
            accM = ck_mmab(qh, yh, accM);
            accM = ck_mmab(qh, yl, accM);
            accM = ck_mmab(ql, yh, accM);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int ri = 16 * I + 8 * hi + e, cj = 16 * J + lo;
            if (cj < ri) {
                const float lv = sm.beta[ri] * accG[e];
                const unsigned short hb = ck_h16(lv);
                ck_el(sm.Lh, ri, cj) = hb;
                ck_el(sm.Ll, ri, cj) = ck_h16(lv - ck_h2f(hb));
            }
            *reinterpret_cast<unsigned short*>(blk + CK_OFF_M + ri * 128 + cj * 2) = cj <= ri ? ck_h16(accM[e]) : (unsigned short)0;
        }
    }
    __syncthreads();
    CK_STAMP(2);

    // ---- P2: T' = (I + L)^-1: incumbent block solve + WMMA merges on a copy of the high half
    // (f16 accurate), then one Newton-Schulz step against the exact L = Lh + Ll to a hi+lo f16 result ----
    for (int x = tid; x < CK_TILES * 256 / 8; x += CK_WG)
        reinterpret_cast<uint4*>(&sm.Th[0])[x] = reinterpret_cast<const uint4*>(&sm.Lh[0])[x];
    __syncthreads();
    {
        unsigned short* T0 = &sm.Th[0];
        const ck_h8 zf = __builtin_bit_cast(ck_h8, make_uint4(0, 0, 0, 0));
        // A[r0.., k0..] and B[k0.., n0..] 16x16 fragments of a packed lower-triangular matrix
        auto fa = [&](const unsigned short* a, int r0, int k0) -> ck_h8 {
            return k0 > r0 ? zf : ck_frag_shared(a + ck_tidx(r0 >> 4, k0 >> 4) * 256, 16, 0, 0, lane);
        };
        auto fb = [&](const unsigned short* a, int n0, int k0) -> ck_h8 {
            return n0 > k0 ? zf : ck_frag_transpose(a + ck_tidx(k0 >> 4, n0 >> 4) * 256, 16, 0, 0, lane);
        };
        if (wave < 4 && lane < 16) {
            const int base = wave * 16;
            float col[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                float value = (i == lane) ? 1.0f : 0.0f;
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    if (j < i) value -= ck_h2f(ck_el(T0, base + i, base + j)) * col[j];
                col[i] = value;
            }
#pragma unroll
            for (int i = 0; i < 16; ++i) ck_el(T0, base + i, base + lane) = ck_h16(col[i]);
        }
        __syncthreads();
        if (wave < 2) {
            const int origin = wave * 32;
            ck_f8 x = ck_z8();
            x = ck_mma(fa(T0, origin + 16, origin), fb(T0, origin, origin), x);
            const int xn = lane & 15;
#pragma unroll
            for (int e = 0; e < 8; ++e) sm.u.tmp[wave * 16 + xn][ck_acc_row(lane, e)] = ck_h16(x[e]);
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            ck_f8 y = ck_z8();
            y = ck_mma(fa(T0, origin + 16, origin + 16),
                       ck_frag_shared(&sm.u.tmp[0][0], 36, wave * 16, 0, lane), y);
            const int yc = origin + (lane & 15);
#pragma unroll
            for (int e = 0; e < 8; ++e) ck_el(T0, origin + 16 + ck_acc_row(lane, e), yc) = ck_h16(-y[e]);
        }
        __syncthreads();
        if (wave < 4) {
            const int rb = wave >> 1, cb = wave & 1;
            ck_f8 y = ck_z8();
#pragma unroll
            for (int kk = 0; kk < 2; ++kk)
                y = ck_mma(fa(T0, 32 + rb * 16, kk * 16), fb(T0, cb * 16, kk * 16), y);
            const int yn = cb * 16 + (lane & 15);
#pragma unroll
            for (int e = 0; e < 8; ++e) sm.u.tmp[yn][rb * 16 + ck_acc_row(lane, e)] = ck_h16(y[e]);
        }
        __syncthreads();
        if (wave < 4) {
            const int rb = wave >> 1, cb = wave & 1;
            ck_f8 z = ck_z8();
#pragma unroll
            for (int kk = 0; kk < 2; ++kk)
                z = ck_mma(fa(T0, 32 + rb * 16, 32 + kk * 16),
                           ck_frag_shared(&sm.u.tmp[0][0], 36, cb * 16, kk * 16, lane), z);
            const int zc = cb * 16 + (lane & 15);
#pragma unroll
            for (int e = 0; e < 8; ++e) ck_el(T0, 32 + rb * 16 + ck_acc_row(lane, e), zc) = ck_h16(-z[e]);
        }
        __syncthreads();

        unsigned short* Tl = &sm.Tl[0];
        // R = I - (I + L) T0 on the lower block tiles (n <= m), stored f16 in Tl
#pragma unroll
        for (int s2 = 0; s2 < 2; ++s2) {
            const int t = wave + 8 * s2;
            if (t < 10) {
                const int m = t >= 6 ? 3 : t >= 3 ? 2 : t >= 1 ? 1 : 0, n = t - m * (m + 1) / 2;
                ck_f8 acc = ck_z8();
                for (int kk = n; kk <= m; ++kk) {
                    const ck_h8 bT = fb(T0, n * 16, kk * 16);
                    uint4 ah = __builtin_bit_cast(uint4, fa(sm.Lh, m * 16, kk * 16));
                    uint4 al = __builtin_bit_cast(uint4, fa(sm.Ll, m * 16, kk * 16));
                    ah.x ^= 0x80008000u; ah.y ^= 0x80008000u; ah.z ^= 0x80008000u; ah.w ^= 0x80008000u;
                    al.x ^= 0x80008000u; al.y ^= 0x80008000u; al.z ^= 0x80008000u; al.w ^= 0x80008000u;
                    acc = ck_mma(__builtin_bit_cast(ck_h8, ah), bT, acc);
                    acc = ck_mma(__builtin_bit_cast(ck_h8, al), bT, acc);
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int row = m * 16 + 8 * hi + e, col = n * 16 + lo;
                    ck_el(Tl, row, col) = ck_h16((row == col ? 1.0f : 0.0f) - ck_h2f(ck_el(T0, row, col)) + acc[e]);
                }
            }
        }
        __syncthreads();
        // T1 = T0 + T0 R (f32), held in registers until every tile has read T0 and R
        ck_f8 T1[2];
#pragma unroll
        for (int s2 = 0; s2 < 2; ++s2) {
            T1[s2] = ck_z8();
            const int t = wave + 8 * s2;
            if (t < 10) {
                const int m = t >= 6 ? 3 : t >= 3 ? 2 : t >= 1 ? 1 : 0, n = t - m * (m + 1) / 2;
                ck_f8 acc = ck_z8();
                for (int kk = n; kk <= m; ++kk)
                    acc = ck_mma(fa(T0, m * 16, kk * 16), fb(Tl, n * 16, kk * 16), acc);
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    T1[s2][e] = ck_h2f(ck_el(T0, m * 16 + 8 * hi + e, n * 16 + lo)) + acc[e];
            }
        }
        __syncthreads();
#pragma unroll
        for (int s2 = 0; s2 < 2; ++s2) {
            const int t = wave + 8 * s2;
            if (t < 10) {
                const int m = t >= 6 ? 3 : t >= 3 ? 2 : t >= 1 ? 1 : 0, n = t - m * (m + 1) / 2;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int row = m * 16 + 8 * hi + e, col = n * 16 + lo;
                    const unsigned short hb = ck_h16(T1[s2][e]);
                    ck_el(T0, row, col) = hb;
                    ck_el(Tl, row, col) = ck_h16(T1[s2][e] - ck_h2f(hb));
                }
            }
        }
        __syncthreads();
    }

    CK_STAMP(3);
    // ---- P3: wave w owns channel tile w (key tile for W'/Q'/Kend, value tile for U') ----
    if (wave < 8) {
        const int c = wave * 16 + lo;
        const float ellT = sm.ell[63][c];
        const float* const kl0 = kb + (size_t)(8 * hi) * k_rs + c;
        const float* const ql0 = qb + (size_t)(8 * hi) * q_rs + c;
        const float* const vl0 = vb + (size_t)(8 * hi) * v_rs + c;
        ck_h8 Bk[4], Bkl[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            ck_h8 kf, kl;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int tok = 16 * k + 8 * hi + e;
                const float kv = tok < rows ? kl0[(16 * k + e) * k_rs] : 0.0f;
                const float el = sm.ell[tok][c];
                const float bx = sm.beta[tok] * kv * __expf(el);
                Bk[k][e] = (_Float16)bx;
                Bkl[k][e] = (_Float16)(bx - (float)Bk[k][e]);
                const float kend = kv * __expf(ellT - el);
                const _Float16 hh = (_Float16)kend;
                kf[e] = hh;
                kl[e] = (_Float16)(kend - (float)hh);
            }
            // Kend^T [c][tok], hi and lo f16 planes
            *reinterpret_cast<uint4*>(blk + CK_OFF_KT + c * 128 + (16 * k + 8 * hi) * 2) = __builtin_bit_cast(uint4, kf);
            *reinterpret_cast<uint4*>(blk + CK_OFF_KL + c * 128 + (16 * k + 8 * hi) * 2) = __builtin_bit_cast(uint4, kl);
        }
        if (hi == 0) *reinterpret_cast<float*>(blk + CK_OFF_DT + c * 4) = __expf(ellT);
        __builtin_amdgcn_sched_barrier(0);
        const unsigned short* Th = &sm.Th[0];
        const unsigned short* Tl = &sm.Tl[0];
        ck_f8 accW[4];
        ck_h8 Wf[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            accW[m] = ck_z8();
#pragma unroll
            for (int kk = 0; kk <= m; ++kk) {
                const ck_h8 ah = ck_ld16(Th + ck_tidx(m, kk) * 256 + lo * 16 + hi * 8);
                const ck_h8 al = ck_ld16(Tl + ck_tidx(m, kk) * 256 + lo * 16 + hi * 8);
                accW[m] = ck_mma(ah, Bk[kk], accW[m]);
                accW[m] = ck_mma(ah, Bkl[kk], accW[m]);
                accW[m] = ck_mma(al, Bk[kk], accW[m]);
            }
            Wf[m] = ck_pack(accW[m]);
        }
        unsigned short* st = &sm.u.stage[wave][0];
        auto put = [&](unsigned off, int m, auto fn) {
#pragma unroll
            for (int e = 0; e < 8; ++e) st[(8 * hi + e) * 16 + lo] = fn(e);
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            *reinterpret_cast<uint4*>(blk + off + wave * 2048 + (16 * m + (lane >> 1)) * 32 + (lane & 1) * 16) =
                *reinterpret_cast<const uint4*>(&st[lane * 8]);
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        };
        // -W' (hi and lo planes) -> scratch tiles
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            put(CK_OFF_WN, m, [&](int e) { return ck_h16(-accW[m][e]); });
            put(CK_OFF_WL, m, [&](int e) {
                const float x = -accW[m][e];
                return ck_h16(x - ck_h2f(ck_h16(x)));
            });
        }
        __builtin_amdgcn_sched_barrier(0);
        // Q' = Qdec - M W'   (M from the scratch copy written by the pair phase)
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            ck_f8 acc2 = ck_z8();
#pragma unroll
            for (int kk = 0; kk <= m; ++kk)
                acc2 = ck_mma(ck_ld16(blk + CK_OFF_M + (16 * m + lo) * 128 + (16 * kk + 8 * hi) * 2), Wf[kk], acc2);
            put(CK_OFF_QP, m, [&](int e) {
                const int tok = 16 * m + 8 * hi + e;
                const float qv = tok < rows ? ql0[(16 * m + e) * q_rs] : 0.0f;
                return ck_h16(qv * __expf(sm.ell[tok][c]) - acc2[e]);
            });
        }
        __builtin_amdgcn_sched_barrier(0);
        // U' = T' beta V (beta V split hi+lo f16), stored transposed [v][tok] in FP32
        ck_h8 Bu[4], Bl[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int tok = 16 * k + 8 * hi + e;
                const float vv = tok < rows ? vl0[(16 * k + e) * v_rs] : 0.0f;
                const float x = sm.beta[tok] * vv;
                const _Float16 xh = (_Float16)x;
                Bu[k][e] = xh;
                Bl[k][e] = (_Float16)(x - (float)xh);
            }
        }
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            ck_f8 acc = ck_z8();
#pragma unroll
            for (int kk = 0; kk <= m; ++kk) {
                const ck_h8 ah = ck_ld16(Th + ck_tidx(m, kk) * 256 + lo * 16 + hi * 8);
                const ck_h8 al = ck_ld16(Tl + ck_tidx(m, kk) * 256 + lo * 16 + hi * 8);
                acc = ck_mma(ah, Bu[kk], acc);
                acc = ck_mma(ah, Bl[kk], acc);
                acc = ck_mma(al, Bu[kk], acc);
            }
            float* dst = reinterpret_cast<float*>(blk + CK_OFF_UT + c * 256 + (16 * m + 8 * hi) * 4);
            *reinterpret_cast<float4*>(dst) = make_float4(acc[0], acc[1], acc[2], acc[3]);
            *reinterpret_cast<float4*>(dst + 4) = make_float4(acc[4], acc[5], acc[6], acc[7]);
        }
    }
    CK_STAMP(4);
    __syncthreads();  // sm is reused by the next chunk-head
    }
}

// Scan: one wave owns (head, 16 value rows) and walks the chunks of the segment. The
// state tile is kept transposed ([key 8 tiles of 16][16 values]) in FP32 registers.
extern "C" __launch_bounds__(32)
__global__ void gated_delta_vector_chunk_scan_gfx12(
    const unsigned char* __restrict__ scratch,
    float* __restrict__ state,            // slot base; element (h, v, k) at h*shs + v*srs + k
    float* __restrict__ output,           // [rows, H, V] at ors, segment-relative
    int seg_rows, int n_heads, int round_out_bf16,
    int out_row_stride, int state_head_stride, int state_row_stride)
{
    const int h = (int)blockIdx.x >> 3, vt = (int)blockIdx.x & 7;
    if (h >= n_heads) return;
    const int lane = (int)threadIdx.x, lo = lane & 15, hi = lane >> 4;
    const int v = vt * 16 + lo;

    ck_f8 St[8];
    float* sp = state + (size_t)h * state_head_stride + (size_t)v * state_row_stride;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const float4 a = *reinterpret_cast<const float4*>(sp + nt * 16 + 8 * hi);
        const float4 b = *reinterpret_cast<const float4*>(sp + nt * 16 + 8 * hi + 4);
        St[nt] = (ck_f8){a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    }

    const int nch = (seg_rows + CK_C - 1) / CK_C;
    for (int chunk = 0; chunk < nch; ++chunk) {
        const unsigned char* blk = scratch + ((size_t)chunk * n_heads + h) * CK_BLK;
        const int rows = (seg_rows - chunk * CK_C) < CK_C ? (seg_rows - chunk * CK_C) : CK_C;

        ck_h8 Sf[8], Sl[8];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const _Float16 h16v = (_Float16)St[nt][e];
                Sf[nt][e] = h16v;
                Sl[nt][e] = (_Float16)(St[nt][e] - (float)h16v);
            }
        }

        ck_h8 Uf[4];
        ck_f8 E[4];
#pragma unroll
        for (int jt = 0; jt < 4; ++jt) {
            const float* up = reinterpret_cast<const float*>(blk + CK_OFF_UT + v * 256 + (16 * jt + 8 * hi) * 4);
            const float4 ua = *reinterpret_cast<const float4*>(up);
            const float4 ub = *reinterpret_cast<const float4*>(up + 4);
            E[jt] = (ck_f8){ua.x, ua.y, ua.z, ua.w, ub.x, ub.y, ub.z, ub.w};
            Uf[jt] = ck_pack(E[jt]);
        }
        // Et = U' - W' S0^T   (Wn holds -W')
#pragma unroll
        for (int m = 0; m < 4; ++m) {
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                const ck_h8 wh = ck_ld16(blk + CK_OFF_WN + kk * 2048 + (16 * m + lo) * 32 + hi * 16);
                const ck_h8 wl = ck_ld16(blk + CK_OFF_WL + kk * 2048 + (16 * m + lo) * 32 + hi * 16);
                E[m] = ck_mma(wh, Sf[kk], E[m]);
                E[m] = ck_mma(wh, Sl[kk], E[m]);
                E[m] = ck_mma(wl, Sf[kk], E[m]);
            }
            __builtin_amdgcn_sched_barrier(0);
        }
        // Y = M U' + Q' S0^T, one token tile at a time
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            ck_f8 Y = ck_z8();
#pragma unroll
            for (int jt = 0; jt <= m; ++jt)
                Y = ck_mma(ck_ld16(blk + CK_OFF_M + (16 * m + lo) * 128 + (16 * jt + 8 * hi) * 2), Uf[jt], Y);
#pragma unroll
            for (int kk = 0; kk < 8; ++kk)
                Y = ck_mma(ck_ld16(blk + CK_OFF_QP + kk * 2048 + (16 * m + lo) * 32 + hi * 16), Sf[kk], Y);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int tok = 16 * m + 8 * hi + e;
                if (tok < rows) {
                    const float y = round_out_bf16 ? ck_bf16_rn(Y[e]) : Y[e];
                    output[(size_t)(chunk * CK_C + tok) * out_row_stride + (size_t)h * CK_D + v] = y;
                }
            }
            __builtin_amdgcn_sched_barrier(0);
        }
        // S <- S diag(exp(ell_T)) + Et^T Kend
        ck_h8 Eh[4], El[4];
#pragma unroll
        for (int jt = 0; jt < 4; ++jt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const _Float16 h = (_Float16)E[jt][e];
                Eh[jt][e] = h;
                El[jt][e] = (_Float16)(E[jt][e] - (float)h);
            }
        }
        const float* dt = reinterpret_cast<const float*>(blk + CK_OFF_DT);
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const float4 da = *reinterpret_cast<const float4*>(dt + nt * 16 + 8 * hi);
            const float4 db = *reinterpret_cast<const float4*>(dt + nt * 16 + 8 * hi + 4);
            St[nt][0] *= da.x; St[nt][1] *= da.y; St[nt][2] *= da.z; St[nt][3] *= da.w;
            St[nt][4] *= db.x; St[nt][5] *= db.y; St[nt][6] *= db.z; St[nt][7] *= db.w;
#pragma unroll
            for (int jt = 0; jt < 4; ++jt) {
                const ck_h8 kh = ck_ld16(blk + CK_OFF_KT + (nt * 16 + lo) * 128 + (16 * jt + 8 * hi) * 2);
                const ck_h8 kl = ck_ld16(blk + CK_OFF_KL + (nt * 16 + lo) * 128 + (16 * jt + 8 * hi) * 2);
                St[nt] = ck_mma(kh, Eh[jt], St[nt]);
                St[nt] = ck_mma(kh, El[jt], St[nt]);
                St[nt] = ck_mma(kl, Eh[jt], St[nt]);
            }
            __builtin_amdgcn_sched_barrier(0);
        }
    }

#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        *reinterpret_cast<float4*>(sp + nt * 16 + 8 * hi) = make_float4(St[nt][0], St[nt][1], St[nt][2], St[nt][3]);
        *reinterpret_cast<float4*>(sp + nt * 16 + 8 * hi + 4) = make_float4(St[nt][4], St[nt][5], St[nt][6], St[nt][7]);
    }
}
