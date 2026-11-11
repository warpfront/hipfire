// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

// Absorbed-latent support kernels (arch independent: plain FP32 FMAs, LDS tiling;
// no matrix-unit builtins, so one source serves every RDNA generation).
//
// Packed expansion weight (private layout, BF16, written by the pack kernel and
// read only by the gemm kernel). For head h the block starts at element
// h * latent * (nope + v) and holds
//     AkT_h [nope][latent]   AkT_h[d][c] = W[h*(nope+v) + d     ][c]   (key rows, transposed view)
//     AvT_h [latent][v]      AvT_h[c][e] = W[h*(nope+v) + nope+e][c]   (value rows, transposed view)
// i.e. both halves are "reduction-index major, output-index contiguous", which
// is the one operand shape the gemm kernel reads.
//
// Gemm kernel: for every head h and row r
//     out[r][h][n] = sum_k x[r][h][k] * Wh[k][n]        (k ascending, FP32 FMA)
// with x a F32 plane [rows, H, x_hs] (first K channels used), Wh a BF16
// [K][N] matrix (N contiguous) and out a F32 plane [rows, H, o_hs]. The same
// kernel serves
//     q route:  x = q [rows,H,nope+tail], Wh = AkT_h (K=nope, N=latent),
//               tail channels copied to out[..][latent .. latent+tail)
//     v route:  x = z [rows,H,latent],    Wh = AvT_h (K=latent, N=v), optional BF16 round.
//
// Work decomposition (all constants come from the launcher, none from data):
//   * a thread owns 8 consecutive n (one 16-byte weight load per k) for
//     LAB_RPT = 8 consecutive rows => 64 accumulators, 64 FMAs per weight load;
//   * a block is [ks][rgn][nl] threads: nl lanes along n (tile = 8*nl outputs),
//     rgn row groups (tile = 8*rgn rows), ks k-segments;
//   * x for the block's rows is staged in LDS (sq[ks][dc][TR], k-chunks of dc),
//     read back as broadcast float4s; weights stream from global/L2 once per
//     block row tile.
// Large row counts use ks = 1: every output is one ascending-k FMA chain.
// Small row counts (decode) use ks > 1: the k range is split in ks contiguous
// segments, each summed ascending from zero by its own threads, and the
// segment partials are added in ascending segment order through LDS. Both are
// fixed-order, no atomics; the two configurations differ only in the F32
// summation order and the configuration is a pure function of the launch
// geometry.

#define LAB_RPT 8

static __device__ __forceinline__ uint16_t lab_f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// Pack: w_t F32 [latent][n_heads*(nope+v)] (row c = column c of W) -> BF16 layout above.
extern "C" __global__ __launch_bounds__(256) void latent_attention_absorb_pack_bf16(
    const float* __restrict__ w_t,
    uint16_t* __restrict__ packed,
    int n_heads, int nope, int v, int latent
) {
    const size_t total = (size_t)n_heads * (size_t)latent * (size_t)(nope + v);
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t head_elems = (size_t)latent * (size_t)(nope + v);
    const size_t h = i / head_elems;
    size_t o = i - h * head_elems;
    const size_t k_elems = (size_t)nope * (size_t)latent;
    size_t c, col;
    if (o < k_elems) {
        const size_t d = o / (size_t)latent;
        c = o - d * (size_t)latent;
        col = h * (size_t)(nope + v) + d;
    } else {
        o -= k_elems;
        c = o / (size_t)v;
        const size_t e = o - c * (size_t)v;
        col = h * (size_t)(nope + v) + (size_t)nope + e;
    }
    const size_t ncols = (size_t)n_heads * (size_t)(nope + v);
    packed[i] = lab_f32_to_bf16(w_t[c * ncols + col]);
}

// acc[i][j] = fma(q[i], bf16(w[j]), acc[i][j]) for one reduction index.
static __device__ __forceinline__ void lab_step(float (&acc)[LAB_RPT][8], const uint4 u, const float* qs) {
    float wf[8];
    wf[0] = __uint_as_float(u.x << 16);
    wf[1] = __uint_as_float(u.x & 0xffff0000u);
    wf[2] = __uint_as_float(u.y << 16);
    wf[3] = __uint_as_float(u.y & 0xffff0000u);
    wf[4] = __uint_as_float(u.z << 16);
    wf[5] = __uint_as_float(u.z & 0xffff0000u);
    wf[6] = __uint_as_float(u.w << 16);
    wf[7] = __uint_as_float(u.w & 0xffff0000u);
    const float4 q0 = *(const float4*)qs;
    const float4 q1 = *(const float4*)(qs + 4);
    const float qv[LAB_RPT] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
    for (int i = 0; i < LAB_RPT; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(qv[i], wf[j], acc[i][j]);
    }
}

extern __shared__ float lab_lds[];

// Launch: grid = (ceil(N / (8*nl)), ceil(rows / (8*rgn)), n_heads),
//         block = nl * rgn * ks threads, dynamic LDS = max(ks*dc*8*rgn, ks*8*rgn*8*nl) floats.
//   kl = ceil(K / ks) (k-segment length), dc = LDS k-chunk (<= kl).
//   ovec != 0: out rows are 16-byte aligned (vector stores on the ks == 1 path).
//   round_bf16 != 0: one RNE BF16 rounding of every output (stored as F32).
//   tail > 0: out[r][h][N + j] = x[r][h][K + j] for j < tail (row tail copy).
extern "C" __global__ __launch_bounds__(128) void latent_attention_absorb_gemm(
    const float* __restrict__ x,
    const uint16_t* __restrict__ w,
    float* __restrict__ out,
    int rows, int n_heads, int K, int N,
    int x_hs, int o_hs, uint64_t w_hs,
    int nl, int ks, int kl, int dc,
    int ovec, int round_bf16, int tail
) {
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int rgn = nthr / (nl * ks);
    const int TR = rgn * LAB_RPT;
    const int NT = nl * 8;
    const int lane_n = tid % nl;
    const int rg = (tid / nl) % rgn;
    const int kq = tid / (nl * rgn);
    const int n0 = blockIdx.x * NT + lane_n * 8;
    const int row0 = blockIdx.y * TR;
    const int h = blockIdx.z;
    const bool nvalid = n0 < N;
    const size_t x_rs = (size_t)n_heads * (size_t)x_hs;
    const size_t o_rs = (size_t)n_heads * (size_t)o_hs;

    if (tail > 0 && blockIdx.x == 0) {
        const int trow = min(TR, rows - row0);
        for (int i = tid; i < trow * tail; i += nthr) {
            const int r = i / tail;
            const int j = i - r * tail;
            const size_t grow = (size_t)(row0 + r);
            out[grow * o_rs + (size_t)h * o_hs + N + j] = x[grow * x_rs + (size_t)h * x_hs + K + j];
        }
    }

    float acc[LAB_RPT][8];
#pragma unroll
    for (int i = 0; i < LAB_RPT; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;
    }

    const int kbeg = min(K, kq * kl);
    const int kend = min(K, (kq + 1) * kl);
    const uint16_t* wh = w + (size_t)w_hs * (size_t)h;
    const int nchunks = (kl + dc - 1) / dc;
    float* sq = lab_lds;

    for (int ci = 0; ci < nchunks; ++ci) {
        __syncthreads();
        const int total = ks * dc * TR;
        for (int e = tid; e < total; e += nthr) {
            const int r = e % TR;
            const int dd = e / TR;
            const int seg = dd / dc;
            const int dl = dd - seg * dc;
            const int k = seg * kl + ci * dc + dl;
            const int kmax = min(K, (seg + 1) * kl);
            const int grow = row0 + r;
            float val = 0.0f;
            if (grow < rows && k < kmax) val = x[(size_t)grow * x_rs + (size_t)h * x_hs + k];
            sq[dd * TR + r] = val;
        }
        __syncthreads();

        const int kb = kbeg + ci * dc;
        const int ke = min(kend, kb + dc);
        const float* sqp = sq + (size_t)(kq * dc) * TR + rg * LAB_RPT;
        if (nvalid) {
            int k = kb;
            for (; k + 8 <= ke; k += 8) {
                uint4 wv[8];
#pragma unroll
                for (int u = 0; u < 8; ++u) wv[u] = *(const uint4*)(wh + (size_t)(k + u) * N + n0);
#pragma unroll
                for (int u = 0; u < 8; ++u) lab_step(acc, wv[u], sqp + (size_t)(k + u - kb) * TR);
            }
            for (; k < ke; ++k) {
                const uint4 wv = *(const uint4*)(wh + (size_t)k * N + n0);
                lab_step(acc, wv, sqp + (size_t)(k - kb) * TR);
            }
        }
    }

    if (ks == 1) {
        if (nvalid) {
#pragma unroll
            for (int i = 0; i < LAB_RPT; ++i) {
                const int grow = row0 + rg * LAB_RPT + i;
                if (grow < rows) {
                    if (round_bf16) {
#pragma unroll
                        for (int j = 0; j < 8; ++j)
                            acc[i][j] = __uint_as_float(((uint32_t)lab_f32_to_bf16(acc[i][j])) << 16);
                    }
                    float* dst = out + (size_t)grow * o_rs + (size_t)h * o_hs + n0;
                    if (ovec) {
                        *(float4*)dst = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
                        *(float4*)(dst + 4) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
                    } else {
#pragma unroll
                        for (int j = 0; j < 8; ++j) dst[j] = acc[i][j];
                    }
                }
            }
        }
    } else {
        __syncthreads();
        float* red = lab_lds;  // [ks][TR][NT]
        if (nvalid) {
#pragma unroll
            for (int i = 0; i < LAB_RPT; ++i) {
                float* d = red + ((size_t)kq * TR + rg * LAB_RPT + i) * NT + lane_n * 8;
                *(float4*)d = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
                *(float4*)(d + 4) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
            }
        }
        __syncthreads();
        for (int o = tid; o < TR * NT; o += nthr) {
            const int nn = o % NT;
            const int r = o / NT;
            const int gn = blockIdx.x * NT + nn;
            const int grow = row0 + r;
            if (gn < N && grow < rows) {
                float s = red[(size_t)r * NT + nn];
                for (int q = 1; q < ks; ++q) s += red[((size_t)q * TR + r) * NT + nn];
                if (round_bf16) s = __uint_as_float(((uint32_t)lab_f32_to_bf16(s)) << 16);
                out[(size_t)grow * o_rs + (size_t)h * o_hs + gn] = s;
            }
        }
    }
}
