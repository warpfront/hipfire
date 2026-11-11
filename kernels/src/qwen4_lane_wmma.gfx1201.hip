// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — lane-WMMA lab: purpose-built gfx1201 WMMA kernels for the Qwen3.8 Flash-Next HC mixers and
// the shared expert (decode family, rows 1..=64 and also correct for rows > 64).
//
// Weights are BF16 ([M, K] u16) or planar W8 (q i8 [M, K], d f16 [M, K/32], value q * d).  The
// int8 -> BF16 widening is exact, so the WMMA operand is the integer q and the per-32 scale d is
// applied to the f32 accumulator per 32-K block (acc = fmaf(d, block_acc, acc)), never folded into
// the operand.  Activations are f32 and are fed as NT in 1..=3 BF16 split terms
// (hi = bf16(x), lo = bf16(x - hi), lo2 = bf16(x - hi - lo)), each term one WMMA against the SAME
// weight fragment, issued hi, lo, lo2 into the same accumulator.
//
// gfx12 WMMA fragment (kernels/src/hyper_read_up_wmma.gfx1201.hip): lane l, l16 = l & 15,
// kh = l >> 4.  a = 8 BF16 of X row (t0 + l16), K elements [k + 8 * kh, +8) of the 16-wide
// substep at k; b = 8 BF16 of W row (n0 + l16), same K elements;
// acc[i] = C[token t0 + 8 * kh + i][column n0 + l16], C = X . W^T.
//
// A1: lwmma_gemm_{bf16,w8}_t{1,2,3}_m{1,2,3,4}_n{1,2}   Y[rows, M] = epi(X . W^T)
// A2: lwmma_read_up_{bf16,w8}_t{1,2,3}                   HC up projection + branch mix
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// The epilogues are the reference kernels' expressions: no multiply-add contraction.
#pragma clang fp contract(off)

typedef short __attribute__((ext_vector_type(8))) lwmma_bf16x8_t;
typedef float __attribute__((ext_vector_type(8))) lwmma_f8_t;

// tensor_ops.hip's bf16_to_f32(bf16_round(v)) == hip_bfloat16's RNE.
static __device__ __forceinline__ float lwmma_bf16r(float v) {
    return static_cast<float>(static_cast<hip_bfloat16>(v));
}

// Finite-input BF16 RNE bits: u += 0x7fff + ((u >> 16) & 1); bits = u >> 16.
static __device__ __forceinline__ unsigned int lwmma_rne_bits(float x) {
    unsigned int u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u);
    return u >> 16;
}

static __device__ __forceinline__ float lwmma_bits_to_f32(unsigned int bits) {
    return __uint_as_float(bits << 16);
}

// 8 f32 activations -> NT BF16 fragments (hi, lo, lo2).
template <int NT>
static __device__ __forceinline__ void lwmma_split8(const float4 xa, const float4 xb, lwmma_bf16x8_t* out) {
    const float x[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    unsigned int hi[8], lo[8], lo2[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
        hi[e] = lwmma_rne_bits(x[e]);
        if constexpr (NT >= 2) {
            const float r = x[e] - lwmma_bits_to_f32(hi[e]);
            lo[e] = lwmma_rne_bits(r);
            if constexpr (NT >= 3) {
                const float r2 = (x[e] - lwmma_bits_to_f32(hi[e])) - lwmma_bits_to_f32(lo[e]);
                lo2[e] = lwmma_rne_bits(r2);
            }
        }
    }
    uint4 p;
    p.x = hi[0] | (hi[1] << 16); p.y = hi[2] | (hi[3] << 16);
    p.z = hi[4] | (hi[5] << 16); p.w = hi[6] | (hi[7] << 16);
    out[0] = __builtin_bit_cast(lwmma_bf16x8_t, p);
    if constexpr (NT >= 2) {
        p.x = lo[0] | (lo[1] << 16); p.y = lo[2] | (lo[3] << 16);
        p.z = lo[4] | (lo[5] << 16); p.w = lo[6] | (lo[7] << 16);
        out[1] = __builtin_bit_cast(lwmma_bf16x8_t, p);
    }
    if constexpr (NT >= 3) {
        p.x = lo2[0] | (lo2[1] << 16); p.y = lo2[2] | (lo2[3] << 16);
        p.z = lo2[4] | (lo2[5] << 16); p.w = lo2[6] | (lo2[7] << 16);
        out[2] = __builtin_bit_cast(lwmma_bf16x8_t, p);
    }
}

// int8 -> f32 -> BF16 bits (exact: |q| <= 128 has at most 7 mantissa bits) for 8 packed bytes.
static __device__ __forceinline__ lwmma_bf16x8_t lwmma_q8_frag(const uint2 u) {
    const unsigned int w[2] = {u.x, u.y};
    uint4 p;
    unsigned int o[4];
#pragma unroll
    for (int h = 0; h < 2; h++) {
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int q0 = (int)(signed char)((w[h] >> (16 * e)) & 0xffu);
            const int q1 = (int)(signed char)((w[h] >> (16 * e + 8)) & 0xffu);
            o[h * 2 + e] = (__float_as_uint((float)q0) >> 16) | (__float_as_uint((float)q1) & 0xffff0000u);
        }
    }
    p.x = o[0]; p.y = o[1]; p.z = o[2]; p.w = o[3];
    return __builtin_bit_cast(lwmma_bf16x8_t, p);
}

// hc_act from gemv_bf16_xf32_k4.
static __device__ __forceinline__ float lwmma_hc_act(float value, float scale) {
    value = static_cast<float>(static_cast<hip_bfloat16>(value));
    value *= scale;
    value = static_cast<float>(static_cast<hip_bfloat16>(value));
    value = value / (1.0f + expf(-value));
    value = static_cast<float>(static_cast<hip_bfloat16>(value));
    return value;
}

// ---------------------------------------------------------------------------
// A1: Y[rows, M] = epi(X . W^T).  Grid [ceil(M / (16 * NN)), split_k, ceil(rows / 64)], block 128.
// K % (128 * split_k) == 0.  split_k > 1: partial[kz][token][M] + last-block deterministic fold
// (ascending kz from 0.0f); `counters` [gridDim.z * gridDim.x] is left zeroed.
template <int WF, int NT, int MT, int NN>
static __device__ __forceinline__ void lwmma_gemm_body(
    const void* __restrict__ Wv,
    const _Float16* __restrict__ D,
    const float* __restrict__ X,
    float* __restrict__ Y,
    float* __restrict__ partial,
    unsigned int* __restrict__ counters,
    int M, int K, int rows, int split_k, float scale, int hc_act, int bx, size_t counter_slot) {
    constexpr int NACC = MT * NN;
    constexpr int WAVE_STRIDE = NACC * 8 * 32;  // floats per wave in the LDS fold
    __shared__ float red[4 * WAVE_STRIDE];
    __shared__ int last_flag;
    const int tid = threadIdx.x;
    const int wave = tid >> 5, lane = tid & 31, l16 = lane & 15, kh = lane >> 4;
    const int n0 = bx * 16 * NN;
    const int t0 = blockIdx.z * 64;
    const int kz = blockIdx.y;
    const int kc = K / split_k;
    const int kq = kc / 4;
    const int kbeg = kz * kc + wave * kq;
    const int nblk = kq / 32;
    const int dstride = K / 32;

    const unsigned short* wb[NN];
    const signed char* wq[NN];
    const _Float16* dr[NN];
#pragma unroll
    for (int nn = 0; nn < NN; nn++) {
        const int row = min(n0 + nn * 16 + l16, M - 1);
        wb[nn] = (const unsigned short*)Wv + (size_t)row * K;
        wq[nn] = (const signed char*)Wv + (size_t)row * K;
        dr[nn] = WF ? D + (size_t)row * dstride : D;
    }
    const float* xr[MT];
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
        xr[mt] = X + (size_t)min(t0 + mt * 16 + l16, rows - 1) * K + 8 * kh;

    lwmma_f8_t acc[MT][NN];
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nn = 0; nn < NN; nn++) acc[mt][nn] = lwmma_f8_t{0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll 2
    for (int b = 0; b < nblk; b++) {
        const int k0 = kbeg + b * 32;
        lwmma_bf16x8_t wf[2][NN];
        float dv[NN];
#pragma unroll
        for (int nn = 0; nn < NN; nn++) {
#pragma unroll
            for (int s = 0; s < 2; s++) {
                const int ke = k0 + s * 16 + 8 * kh;
                if (WF == 0)
                    wf[s][nn] = __builtin_bit_cast(lwmma_bf16x8_t, *(const uint4*)(wb[nn] + ke));
                else
                    wf[s][nn] = lwmma_q8_frag(*(const uint2*)(wq[nn] + ke));
            }
            dv[nn] = WF ? (float)dr[nn][k0 >> 5] : 0.0f;
        }
        float4 xv[MT][2][2];
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int s = 0; s < 2; s++) {
                const float4* p = (const float4*)(xr[mt] + k0 + s * 16);
                xv[mt][s][0] = p[0];
                xv[mt][s][1] = p[1];
            }
        lwmma_f8_t blk[MT][NN];
        if (WF) {
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nn = 0; nn < NN; nn++) blk[mt][nn] = lwmma_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
        }
#pragma unroll
        for (int s = 0; s < 2; s++) {
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                lwmma_bf16x8_t xa[NT];
                lwmma_split8<NT>(xv[mt][s][0], xv[mt][s][1], xa);
#pragma unroll
                for (int t = 0; t < NT; t++) {
#pragma unroll
                    for (int nn = 0; nn < NN; nn++) {
                        if (WF)
                            blk[mt][nn] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                                xa[t], wf[s][nn], blk[mt][nn]);
                        else
                            acc[mt][nn] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                                xa[t], wf[s][nn], acc[mt][nn]);
                    }
                }
            }
        }
        if (WF) {
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nn = 0; nn < NN; nn++)
#pragma unroll
                    for (int i = 0; i < 8; i++)
                        acc[mt][nn][i] = __builtin_fmaf(dv[nn], blk[mt][nn][i], acc[mt][nn][i]);
        }
    }

    // Wave fold through LDS: (w0 + w1) + (w2 + w3).
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nn = 0; nn < NN; nn++)
#pragma unroll
            for (int i = 0; i < 8; i++)
                red[wave * WAVE_STRIDE + (((mt * NN + nn) * 8 + i) << 5) + lane] = acc[mt][nn][i];
    __syncthreads();

#pragma unroll 1
    for (int e = tid; e < WAVE_STRIDE; e += 128) {
        const int idx = e >> 5, ln = e & 31;
        const int i = idx & 7, q = idx >> 3;
        const int nn = q % NN, mt = q / NN;
        const int token = t0 + mt * 16 + 8 * (ln >> 4) + i;
        const int col = n0 + nn * 16 + (ln & 15);
        if (token >= rows || col >= M) continue;
        const float v = (red[e] + red[WAVE_STRIDE + e]) + (red[2 * WAVE_STRIDE + e] + red[3 * WAVE_STRIDE + e]);
        if (split_k == 1) {
            Y[(size_t)token * M + col] = hc_act ? lwmma_hc_act(v, scale) : v;
        } else {
            partial[((size_t)kz * rows + token) * M + col] = v;
        }
    }
    if (split_k == 1) return;

    __threadfence();
    __syncthreads();
    unsigned int* const counter = counters + counter_slot;
    if (tid == 0) {
        const unsigned int old = atomicAdd(counter, 1u);
        last_flag = (old == (unsigned int)(split_k - 1)) ? 1 : 0;
    }
    __syncthreads();
    if (!last_flag) return;
    __threadfence();
    // Every thread folds its EPT elements together, z outer, so each z step issues EPT independent
    // loads (and the unrolled z steps overlap) instead of one dependent round trip per (element, z).
    // Per element the sum still runs in ascending kz from 0.0f.
    constexpr int EPT = WAVE_STRIDE / 128;
    size_t off[EPT];
    bool ok[EPT];
    float v[EPT];
#pragma unroll
    for (int u = 0; u < EPT; u++) {
        const int e = tid + u * 128;
        const int idx = e >> 5, ln = e & 31;
        const int i = idx & 7, q = idx >> 3;
        const int nn = q % NN, mt = q / NN;
        const int token = t0 + mt * 16 + 8 * (ln >> 4) + i;
        const int col = n0 + nn * 16 + (ln & 15);
        ok[u] = token < rows && col < M;
        off[u] = ok[u] ? (size_t)token * M + col : 0;
        v[u] = 0.0f;
    }
    const size_t zstride = (size_t)rows * M;
#pragma unroll 4
    for (int z = 0; z < split_k; z++) {
        float p[EPT];
#pragma unroll
        for (int u = 0; u < EPT; u++)
            p[u] = __hip_atomic_load(&partial[z * zstride + off[u]], __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_AGENT);
#pragma unroll
        for (int u = 0; u < EPT; u++) v[u] += p[u];
    }
#pragma unroll
    for (int u = 0; u < EPT; u++)
        if (ok[u]) Y[off[u]] = hc_act ? lwmma_hc_act(v[u], scale) : v[u];
    if (tid == 0) *counter = 0u;
}

#define LWMMA_GEMM_ENTRY(WFN, WF, NT, MT, NN)                                                         \
    extern "C" __global__ __launch_bounds__(128) void lwmma_gemm_##WFN##_t##NT##_m##MT##_n##NN(        \
        const void* __restrict__ W, const _Float16* __restrict__ D, const float* __restrict__ X,    \
        float* __restrict__ Y, float* __restrict__ partial, unsigned int* __restrict__ counters,    \
        int M, int K, int rows, int split_k, float scale, int hc_act) {                             \
        lwmma_gemm_body<WF, NT, MT, NN>(W, D, X, Y, partial, counters, M, K, rows, split_k, scale,    \
                                      hc_act, blockIdx.x,                                          \
                                      (size_t)blockIdx.z * gridDim.x + blockIdx.x);                 \
    }

#define LWMMA_GEMM_MN(WFN, WF, NT)          \
    LWMMA_GEMM_ENTRY(WFN, WF, NT, 1, 1)     \
    LWMMA_GEMM_ENTRY(WFN, WF, NT, 1, 2)     \
    LWMMA_GEMM_ENTRY(WFN, WF, NT, 2, 1)     \
    LWMMA_GEMM_ENTRY(WFN, WF, NT, 2, 2)     \
    LWMMA_GEMM_ENTRY(WFN, WF, NT, 3, 1)     \
    LWMMA_GEMM_ENTRY(WFN, WF, NT, 3, 2)     \
    LWMMA_GEMM_ENTRY(WFN, WF, NT, 4, 1)     \
    LWMMA_GEMM_ENTRY(WFN, WF, NT, 4, 2)

LWMMA_GEMM_MN(bf16, 0, 1)
LWMMA_GEMM_MN(bf16, 0, 2)
LWMMA_GEMM_MN(bf16, 0, 3)
LWMMA_GEMM_MN(w8, 1, 1)
LWMMA_GEMM_MN(w8, 1, 2)
LWMMA_GEMM_MN(w8, 1, 3)

// Multi-part entries (bf16 weights, two activation terms, one n-tile: the wide MoE block's dense
// front, router | selector | shared gate | shared up).  Up to four (W, Y, M, split_k) parts that
// read the SAME X [rows, K] run in ONE launch.  Grid [sum_p ceil(M_p / 16), max_p split_k_p,
// ceil(rows / 64)]: the flat x index walks the parts' column tiles in order, a block whose kz is
// past its part's split_k exits at once, and every other block runs the plain entry's body with
// its part's own (W, Y, M, split_k), a part-private slice of `partial` (offset = the previous
// parts' split_k * rows * M floats; parts with split_k == 1 take none) and the counter slot of its
// flat tile, so each output word is computed by exactly the per-part launch's code and fold.
// W0..W3 Y0..Y3 X partial counters M0..M3 split_k0..3 K rows.  M_p == 0 parts are empty.
#define LWMMA_GEMM_MULTI_ENTRY(MT)                                                                   \
    extern "C" __global__ __launch_bounds__(128) void lwmma_gemm_bf16_t2_m##MT##_n1_multi(           \
        const void* __restrict__ W0, const void* __restrict__ W1, const void* __restrict__ W2,      \
        const void* __restrict__ W3, float* __restrict__ Y0, float* __restrict__ Y1,                \
        float* __restrict__ Y2, float* __restrict__ Y3, const float* __restrict__ X,                \
        float* __restrict__ partial, unsigned int* __restrict__ counters, int M0, int M1, int M2,   \
        int M3, int S0, int S1, int S2, int S3, int K, int rows) {                                  \
        const int t0 = (M0 + 15) >> 4, t1 = (M1 + 15) >> 4, t2 = (M2 + 15) >> 4;                    \
        const int bx = blockIdx.x;                                                                  \
        const int p = bx < t0 ? 0 : bx < t0 + t1 ? 1 : bx < t0 + t1 + t2 ? 2 : 3;                   \
        const int base = p == 0 ? 0 : p == 1 ? t0 : p == 2 ? t0 + t1 : t0 + t1 + t2;               \
        const int sk = p == 0 ? S0 : p == 1 ? S1 : p == 2 ? S2 : S3;                               \
        if ((int)blockIdx.y >= sk) return;                                                          \
        const int M = p == 0 ? M0 : p == 1 ? M1 : p == 2 ? M2 : M3;                                 \
        const void* const W = p == 0 ? W0 : p == 1 ? W1 : p == 2 ? W2 : W3;                        \
        float* const Y = p == 0 ? Y0 : p == 1 ? Y1 : p == 2 ? Y2 : Y3;                             \
        size_t poff = 0;                                                                            \
        if (p > 0 && S0 > 1) poff += (size_t)S0 * rows * M0;                                        \
        if (p > 1 && S1 > 1) poff += (size_t)S1 * rows * M1;                                        \
        if (p > 2 && S2 > 1) poff += (size_t)S2 * rows * M2;                                        \
        lwmma_gemm_body<0, 2, MT, 1>(W, nullptr, X, Y, partial + poff, counters, M, K, rows, sk,    \
                                     0.0f, 0, bx - base,                                            \
                                     (size_t)blockIdx.z * gridDim.x + bx);                          \
    }

LWMMA_GEMM_MULTI_ENTRY(1)
LWMMA_GEMM_MULTI_ENTRY(2)
LWMMA_GEMM_MULTI_ENTRY(3)
LWMMA_GEMM_MULTI_ENTRY(4)

// ---------------------------------------------------------------------------
// A2: HC up projection [4 * hidden, K] of `low` [rows, K] with hyper_read_projected_f32's branch
// mix fused.  Grid [hidden / 16, ceil(rows / (16 * TT))], block 256 = 8 waves = TT token tiles x
// KP = 8 / TT K parts.  Every wave parks its partial logits in LDS; the epilogue then runs on all
// 256 threads, one (token, column) output each pass.  K % 32 == 0, K <= 512, hidden % 16 == 0.
template <int WF, int NT, int TT>
static __device__ __forceinline__ void lwmma_read_up_body(
    const void* __restrict__ Wv,
    const _Float16* __restrict__ D,
    const float* __restrict__ low,
    const float* __restrict__ normalized,
    float* __restrict__ mixed,
    int hidden, int K, int rows) {
    // TT token tiles per block, KP = 8 / TT K parts: wave = part * TT + tile.  Part p takes the 32-K
    // blocks [p * nb / KP, (p + 1) * nb / KP).
    constexpr int KP = 8 / TT;
    __shared__ float red[KP * TT * 32 * 32];  // [part][tile][branch * 8 + i][lane]
    const int tid = threadIdx.x;
    const int wave = tid >> 5, lane = tid & 31, l16 = lane & 15, kh = lane >> 4;
    const int tt = wave % TT, part = wave / TT;
    const int j0 = blockIdx.x * 16;
    const int t0 = blockIdx.y * 16 * TT + tt * 16;
    const int nb = K / 32;
    const int b0 = part * nb / KP;
    const int b1 = (part + 1) * nb / KP;

    lwmma_f8_t acc[4];
#pragma unroll
    for (int br = 0; br < 4; br++) acc[br] = lwmma_f8_t{0, 0, 0, 0, 0, 0, 0, 0};

    if (t0 < rows) {
        const float* xr = low + (size_t)min(t0 + l16, rows - 1) * K + 8 * kh;
        const unsigned short* wb[4];
        const signed char* wq[4];
        const _Float16* dr[4];
#pragma unroll
        for (int br = 0; br < 4; br++) {
            const size_t row = (size_t)br * hidden + j0 + l16;
            wb[br] = (const unsigned short*)Wv + row * K;
            wq[br] = (const signed char*)Wv + row * K;
            dr[br] = WF ? D + row * nb : D;
        }
#pragma unroll 1
        for (int b = b0; b < b1; b++) {
            const int k0 = b * 32;
            lwmma_bf16x8_t wf[2][4];
            float dv[4];
#pragma unroll
            for (int br = 0; br < 4; br++) {
#pragma unroll
                for (int s = 0; s < 2; s++) {
                    const int ke = k0 + s * 16 + 8 * kh;
                    if (WF == 0)
                        wf[s][br] = __builtin_bit_cast(lwmma_bf16x8_t, *(const uint4*)(wb[br] + ke));
                    else
                        wf[s][br] = lwmma_q8_frag(*(const uint2*)(wq[br] + ke));
                }
                dv[br] = WF ? (float)dr[br][b] : 0.0f;
            }
            float4 xv[2][2];
#pragma unroll
            for (int s = 0; s < 2; s++) {
                const float4* p = (const float4*)(xr + k0 + s * 16);
                xv[s][0] = p[0];
                xv[s][1] = p[1];
            }
            lwmma_f8_t blk[4];
            if (WF) {
#pragma unroll
                for (int br = 0; br < 4; br++) blk[br] = lwmma_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
            }
#pragma unroll
            for (int s = 0; s < 2; s++) {
                lwmma_bf16x8_t xa[NT];
                lwmma_split8<NT>(xv[s][0], xv[s][1], xa);
#pragma unroll
                for (int t = 0; t < NT; t++) {
#pragma unroll
                    for (int br = 0; br < 4; br++) {
                        if (WF)
                            blk[br] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                                xa[t], wf[s][br], blk[br]);
                        else
                            acc[br] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                                xa[t], wf[s][br], acc[br]);
                    }
                }
            }
            if (WF) {
#pragma unroll
                for (int br = 0; br < 4; br++)
#pragma unroll
                    for (int i = 0; i < 8; i++)
                        acc[br][i] = __builtin_fmaf(dv[br], blk[br][i], acc[br][i]);
            }
        }
    }

    // Park every part's fragment; the sum below runs ((part0 + part1) + part2) + ... per logit.
#pragma unroll
    for (int br = 0; br < 4; br++)
#pragma unroll
        for (int i = 0; i < 8; i++) red[((part * TT + tt) * 32 + br * 8 + i) * 32 + lane] = acc[br][i];
    __syncthreads();
    const size_t wide = (size_t)4 * hidden;
    // Output o = (tile, token-in-tile tk, column c): the fragment lane holding it is
    // (tk >> 3) * 16 + c at element tk & 7 (acc[i] = C[8 * kh + i][l16]).
    for (int o = tid; o < TT * 256; o += 256) {
        const int t = o >> 8, tk = (o >> 4) & 15, c = o & 15;
        const int token = blockIdx.y * 16 * TT + t * 16 + tk;
        if (token >= rows) continue;
        const int fl = (tk >> 3) * 16 + c, fi = tk & 7;
        const int j = j0 + c;
        float nv[4];
#pragma unroll
        for (int br = 0; br < 4; br++) nv[br] = normalized[(size_t)token * wide + br * hidden + j];
        // hyper_read_projected_f32's mix, branch order from +0.0.
        float value = 0.0f;
#pragma unroll
        for (int br = 0; br < 4; br++) {
            float h = red[((0 * TT + t) * 32 + br * 8 + fi) * 32 + fl];
#pragma unroll
            for (int p = 1; p < KP; p++) h += red[((p * TT + t) * 32 + br * 8 + fi) * 32 + fl];
            const float gate_input = lwmma_bf16r(h);
            const float gate = lwmma_bf16r(1.0f / (1.0f + expf(-gate_input)));
            const float normed = lwmma_bf16r(nv[br]);
            const float term = lwmma_bf16r(gate * normed);
            value += term;
        }
        mixed[(size_t)token * hidden + j] = lwmma_bf16r(value / 4.0f);
    }
}

#define LWMMA_READ_UP_ENTRY(WFN, WF, NT, TT)                                                           \
    extern "C" __global__ __launch_bounds__(256) void lwmma_read_up_##WFN##_t##NT##_g##TT(              \
        const void* __restrict__ W, const _Float16* __restrict__ D, const float* __restrict__ low,   \
        const float* __restrict__ normalized, float* __restrict__ mixed, int hidden, int K,         \
        int rows) {                                                                                 \
        lwmma_read_up_body<WF, NT, TT>(W, D, low, normalized, mixed, hidden, K, rows);                 \
    }

#define LWMMA_READ_UP_G(WFN, WF, NT)       \
    LWMMA_READ_UP_ENTRY(WFN, WF, NT, 1)    \
    LWMMA_READ_UP_ENTRY(WFN, WF, NT, 2)    \
    LWMMA_READ_UP_ENTRY(WFN, WF, NT, 4)

LWMMA_READ_UP_G(bf16, 0, 1)
LWMMA_READ_UP_G(bf16, 0, 2)
LWMMA_READ_UP_G(bf16, 0, 3)
LWMMA_READ_UP_G(w8, 1, 1)
LWMMA_READ_UP_G(w8, 1, 2)
LWMMA_READ_UP_G(w8, 1, 3)
