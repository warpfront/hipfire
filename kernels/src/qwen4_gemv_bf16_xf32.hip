// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Qwen4 copy of gemv_bf16_xf32.hip as #774 left it (gemv_bf16_xf32 body unrolled on
// gfx1151; the K4 split, HC write-norm, scaled-add and multi-matrix/multi-row entries). Qwen4's
// entries load from here on every arch, the shipped module on gfx1151 only.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// BF16-weight × F32-input GEMV: y[m] = W_bf16[m, k] @ x_f32[k].
//
// Native-bf16 reference path (the KLD/PPL oracle). The model ships in bf16;
// this keeps the EXACT downloaded weight values rather than re-quantizing to
// f16 (a different exponent/mantissa split). bf16 is the high 16 bits of an
// f32, so the widen is a lossless 16-bit left shift — no conversion error.
// F32 input precision is preserved (input is read as f32, accumulate in f32),
// matching gemv_f16_xf32 / gemv_q8_0 / gemv_f32.
//   grid = [M, 1, 1], block = [32, 1, 1]  (1 warp per output row)
//
// Batch-1 GEMV is pure weight streaming, so the figure of merit is achieved
// GB/s. At small M there are too few waves to hide memory latency (M=1024 is
// 4 waves/SIMD on gfx1201), so per-wave memory-level parallelism has to come
// from wider loads: each thread pulls 8 bf16 (one 16-byte dwordx4) per step
// instead of a single 2-byte scalar, and 4 accumulators keep the FMAs off one
// dependency chain. Mirrors the unroll + occupancy hint that makes
// `gemv_q8_0` measure at the roof.

// bf16 -> f32 is a lossless 16-bit shift. A dwordx4 holds 8 bf16: within each
// u32 the LOW half is the even element, the HIGH half the odd one (LE).
#define HF_BF16_LO(u) __uint_as_float((u) << 16)
#define HF_BF16_HI(u) __uint_as_float((u) & 0xFFFF0000u)

// One output row: dot(w_row, x), valid in lane 0. Every entry point below
// runs this same body, so a row's value does not depend on which launch
// computed it.
static __device__ __forceinline__ float gemv_bf16_xf32_row(
    const unsigned short* __restrict__ w_row,
    const float*          __restrict__ x,
    int K,
    int tid
) {

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    // Wide path: 8 bf16 per thread per step => 256 elements per wave step.
    // Row base is 16B-aligned whenever K % 8 == 0, which every hipfire decode
    // shape satisfies; the scalar tail below covers the rest.
    if ((K & 7) == 0) {
        // Batch-1 GEMV is latency-bound on the weight stream, so the loop is
        // unrolled eight steps: the same per-chain add order (each sub-step
        // feeds a0..a3 in the original sequence, and `i` only ever increases),
        // but eight independent dwordx4 loads in flight per lane instead of
        // one. Measured on gfx1151 decode shapes: +3% achieved GB/s on the
        // DRAM-bound matrices, bitwise-identical rows.
        int i = tid * 8;
        for (; i + 8 * 32 * 8 <= K; i += 32 * 8 * 8) {
            uint4 wv[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                wv[s] = *(const uint4*)(w_row + i + s * (32 * 8));
            }
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const float4 xa = *(const float4*)(x + i + s * (32 * 8));
                const float4 xb = *(const float4*)(x + i + s * (32 * 8) + 4);
                a0 += HF_BF16_LO(wv[s].x) * xa.x;
                a1 += HF_BF16_HI(wv[s].x) * xa.y;
                a2 += HF_BF16_LO(wv[s].y) * xa.z;
                a3 += HF_BF16_HI(wv[s].y) * xa.w;
                a0 += HF_BF16_LO(wv[s].z) * xb.x;
                a1 += HF_BF16_HI(wv[s].z) * xb.y;
                a2 += HF_BF16_LO(wv[s].w) * xb.z;
                a3 += HF_BF16_HI(wv[s].w) * xb.w;
            }
        }
        for (; i + 8 <= K; i += 32 * 8) {
            const uint4 wv = *(const uint4*)(w_row + i);
            const float4 xa = *(const float4*)(x + i);
            const float4 xb = *(const float4*)(x + i + 4);
            a0 += HF_BF16_LO(wv.x) * xa.x;
            a1 += HF_BF16_HI(wv.x) * xa.y;
            a2 += HF_BF16_LO(wv.y) * xa.z;
            a3 += HF_BF16_HI(wv.y) * xa.w;
            a0 += HF_BF16_LO(wv.z) * xb.x;
            a1 += HF_BF16_HI(wv.z) * xb.y;
            a2 += HF_BF16_LO(wv.w) * xb.z;
            a3 += HF_BF16_HI(wv.w) * xb.w;
        }
    }

    // Scalar tail: the final K % 8 elements (and all of K when K % 8 != 0).
    for (int i = ((K & 7) == 0 ? (K & ~7) : 0) + tid; i < K; i += 32) {
        a0 += __uint_as_float(((unsigned int)w_row[i]) << 16) * x[i];
    }

    float sum = (a0 + a1) + (a2 + a3);

    // Wave-shuffle reduction (wave32).
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    return sum;
}

// gfx1151 runs the shared row body (eight dwordx4 loads in flight, measured
// there). Every other arch keeps the original single-step kernel verbatim, so
// its code is unchanged for the models that already ship on it.
#if defined(__gfx1151__)
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_bf16_xf32(
    const unsigned short* __restrict__ W,  // [M, K] bf16 (raw u16)
    const float*          __restrict__ x,  // [K] F32
    float*                __restrict__ y,  // [M] F32
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const float sum = gemv_bf16_xf32_row(W + (size_t)row * K, x, K, threadIdx.x);
    if (threadIdx.x == 0) y[row] = sum;
}
#else
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_bf16_xf32(
    const unsigned short* __restrict__ W,  // [M, K] bf16 (raw u16)
    const float*          __restrict__ x,  // [K] F32
    float*                __restrict__ y,  // [M] F32
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const unsigned short* w_row = W + (size_t)row * K;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    // Wide path: 8 bf16 per thread per step => 256 elements per wave step.
    // Row base is 16B-aligned whenever K % 8 == 0, which every hipfire decode
    // shape satisfies; the scalar tail below covers the rest.
    if ((K & 7) == 0) {
        for (int i = tid * 8; i + 8 <= K; i += 32 * 8) {
            const uint4 wv = *(const uint4*)(w_row + i);
            const float4 xa = *(const float4*)(x + i);
            const float4 xb = *(const float4*)(x + i + 4);
            a0 += HF_BF16_LO(wv.x) * xa.x;
            a1 += HF_BF16_HI(wv.x) * xa.y;
            a2 += HF_BF16_LO(wv.y) * xa.z;
            a3 += HF_BF16_HI(wv.y) * xa.w;
            a0 += HF_BF16_LO(wv.z) * xb.x;
            a1 += HF_BF16_HI(wv.z) * xb.y;
            a2 += HF_BF16_LO(wv.w) * xb.z;
            a3 += HF_BF16_HI(wv.w) * xb.w;
        }
    }

    // Scalar tail: the final K % 8 elements (and all of K when K % 8 != 0).
    for (int i = ((K & 7) == 0 ? (K & ~7) : 0) + tid; i < K; i += 32) {
        a0 += __uint_as_float(((unsigned int)w_row[i]) << 16) * x[i];
    }

    float sum = (a0 + a1) + (a2 + a3);

    // Wave-shuffle reduction (wave32).
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
#endif

// Long-K rows (decode HC projections, K = 4 * 2560): four waves split each
// row's K into quarters, each running gemv_bf16_xf32_row on its quarter, and
// the quarter sums fold as (q0 + q1) + (q2 + q3). A different summation order
// than gemv_bf16_xf32 (not bitwise equal to it); four times the waves for
// the few-row, long-K shapes that one wave per row leaves latency-bound.
// `hc_act` applies hc_activation_fused_f32 (BF16 round trip, `scale`, round
// trip, SiLU, round trip) to each output. Grid [M], block 128, K % 32 == 0.
__launch_bounds__(128)
extern "C" __global__ void gemv_bf16_xf32_k4(
    const unsigned short* __restrict__ W,  // [M, K] bf16 (raw u16)
    const float*          __restrict__ x,  // [K] F32
    float*                __restrict__ y,  // [M] F32
    int M, int K,
    float scale,
    int hc_act
) {
    __shared__ float quarter[4];
    const int row = blockIdx.x;
    if (row >= M) return;
    const int wave = threadIdx.x >> 5;
    const int part = K / 4;
    const float s = gemv_bf16_xf32_row(
        W + (size_t)row * K + wave * part, x + wave * part, part, threadIdx.x & 31);
    if ((threadIdx.x & 31) == 0) quarter[wave] = s;
    __syncthreads();
    if (threadIdx.x != 0) return;
    float value = (quarter[0] + quarter[1]) + (quarter[2] + quarter[3]);
    if (hc_act) {
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
        value *= scale;
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
        value = value / (1.0f + expf(-value));
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
    }
    y[row] = value;
}

// gemv_bf16_xf32_row over R activation rows (x row stride `x_stride`): each
// dwordx4 of weights is loaded once and folded into every row's four chains
// in gemv_bf16_xf32_row's per-chain order (the gfx1151 eight-step unroll
// keeps that order too), then each row takes the same fold and shuffle tree,
// so every row's sum is bitwise the single-row body's.
template <int R>
static __device__ __forceinline__ void gemv_bf16_xf32_row_multi(
    const unsigned short* __restrict__ w_row,
    const float*          __restrict__ x,
    int x_stride,
    int K,
    int tid,
    float (&sum)[R]
) {
    float a0[R], a1[R], a2[R], a3[R];
#pragma unroll
    for (int r = 0; r < R; ++r) a0[r] = a1[r] = a2[r] = a3[r] = 0.0f;
    if ((K & 7) == 0) {
        for (int i = tid * 8; i + 8 <= K; i += 32 * 8) {
            const uint4 wv = *(const uint4*)(w_row + i);
            const float w0 = HF_BF16_LO(wv.x), w1 = HF_BF16_HI(wv.x);
            const float w2 = HF_BF16_LO(wv.y), w3 = HF_BF16_HI(wv.y);
            const float w4 = HF_BF16_LO(wv.z), w5 = HF_BF16_HI(wv.z);
            const float w6 = HF_BF16_LO(wv.w), w7 = HF_BF16_HI(wv.w);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float4 xa = *(const float4*)(x + (size_t)r * x_stride + i);
                const float4 xb = *(const float4*)(x + (size_t)r * x_stride + i + 4);
                a0[r] += w0 * xa.x;
                a1[r] += w1 * xa.y;
                a2[r] += w2 * xa.z;
                a3[r] += w3 * xa.w;
                a0[r] += w4 * xb.x;
                a1[r] += w5 * xb.y;
                a2[r] += w6 * xb.z;
                a3[r] += w7 * xb.w;
            }
        }
    }
    for (int i = ((K & 7) == 0 ? (K & ~7) : 0) + tid; i < K; i += 32) {
        const float w = __uint_as_float(((unsigned int)w_row[i]) << 16);
#pragma unroll
        for (int r = 0; r < R; ++r) a0[r] += w * x[(size_t)r * x_stride + i];
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = (a0[r] + a1[r]) + (a2[r] + a3[r]);
        for (int offset = 16; offset > 0; offset >>= 1) {
            s += __shfl_down(s, offset);
        }
        sum[r] = s;
    }
}

// gemv_bf16_xf32_k4 of R activation rows (x row stride K, y row stride M) in
// one launch: each weight quarter is read once for all rows, and every row's
// value (quarter sums, fold, `hc_act` epilogue) is bitwise the single-row
// kernel's, so a few-row verify matches the decode it replaces. Grid [M],
// block 128, K % 32 == 0, 2 <= R <= 8.
template <int R>
static __device__ __forceinline__ void gemv_bf16_xf32_k4_rows_body(
    const unsigned short* __restrict__ W,
    const float*          __restrict__ x,
    float*                __restrict__ y,
    int M, int K,
    float scale,
    int hc_act
) {
    __shared__ float quarter[R][4];
    const int row = blockIdx.x;
    if (row >= M) return;
    const int wave = threadIdx.x >> 5;
    const int part = K / 4;
    float s[R];
    gemv_bf16_xf32_row_multi<R>(
        W + (size_t)row * K + wave * part, x + wave * part, K, part, threadIdx.x & 31, s);
    if ((threadIdx.x & 31) == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) quarter[r][wave] = s[r];
    }
    __syncthreads();
    const int r = threadIdx.x;
    if (r >= R) return;
    float value = (quarter[r][0] + quarter[r][1]) + (quarter[r][2] + quarter[r][3]);
    if (hc_act) {
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
        value *= scale;
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
        value = value / (1.0f + expf(-value));
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
    }
    y[(size_t)r * M + row] = value;
}

#define GEMV_BF16_XF32_K4_ROWS(R)                                                   \
    __launch_bounds__(128) extern "C" __global__ void gemv_bf16_xf32_k4_rows_r##R(  \
        const unsigned short* __restrict__ W, const float* __restrict__ x,          \
        float* __restrict__ y, int M, int K, float scale, int hc_act) {             \
        gemv_bf16_xf32_k4_rows_body<R>(W, x, y, M, K, scale, hc_act);               \
    }
GEMV_BF16_XF32_K4_ROWS(2)
GEMV_BF16_XF32_K4_ROWS(3)
GEMV_BF16_XF32_K4_ROWS(4)
GEMV_BF16_XF32_K4_ROWS(5)
GEMV_BF16_XF32_K4_ROWS(6)
GEMV_BF16_XF32_K4_ROWS(7)
GEMV_BF16_XF32_K4_ROWS(8)
#undef GEMV_BF16_XF32_K4_ROWS

// gemv_bf16_xf32_k4_rows_r<R> over many R-row tiles in one launch (prompt
// fill): grid [M, tiles], blockIdx.y selects the tile, whose R activation rows
// (x row stride K) and output rows (y row stride M) start R * blockIdx.y rows
// in. Same body as the non-tiled kernel, so every row is bitwise its value.
#define GEMV_BF16_XF32_K4_ROWS_TILED(R)                                             \
    __launch_bounds__(128) extern "C" __global__ void                               \
    gemv_bf16_xf32_k4_rows_tiled_r##R(                                              \
        const unsigned short* __restrict__ W, const float* __restrict__ x,          \
        float* __restrict__ y, int M, int K, float scale, int hc_act) {             \
        gemv_bf16_xf32_k4_rows_body<R>(                                             \
            W, x + (size_t)blockIdx.y * R * K, y + (size_t)blockIdx.y * R * M,      \
            M, K, scale, hc_act);                                                   \
    }
GEMV_BF16_XF32_K4_ROWS_TILED(2)
GEMV_BF16_XF32_K4_ROWS_TILED(3)
GEMV_BF16_XF32_K4_ROWS_TILED(4)
GEMV_BF16_XF32_K4_ROWS_TILED(5)
GEMV_BF16_XF32_K4_ROWS_TILED(6)
GEMV_BF16_XF32_K4_ROWS_TILED(7)
GEMV_BF16_XF32_K4_ROWS_TILED(8)
#undef GEMV_BF16_XF32_K4_ROWS_TILED

// Decode HC write + next HC read norm (tensor_ops.hip's hyper_write_f32 and
// hyper_norm_f32 on one F32 row, bitwise): one block per branch; each thread
// writes its ten columns of the new streams, keeps them, and normalizes them
// with hyper_norm_rows' partials, 256-wide LDS tree and expression.
//
// `quarters_in` (optional) replaces `gates`: gate[b] is the k4 fold
// (q[b][0] + q[b][1]) + (q[b][2] + q[b][3]) of four quarter dots.
// `quarters_out` (optional) produces those quarters for the NEXT hyper write
// of these same streams: its norm (`next_norm`) has hyper_norm's `inv` (same
// rows), so block b normalizes its branch into LDS and waves 0..3 run
// gemv_bf16_xf32_k4's quarter b of gate rows 0..3 (gemv_bf16_xf32_row over
// that branch) — the next write then needs neither its norm nor its gate
// GEMV. `clear` (optional) zeroes `clear_elements` floats, the
// zero_f32 of a Clear step this launch absorbs. Grid [branches = 4],
// block 256, hidden == 2560.
static __device__ __forceinline__ float hf_bf16(float v) {
    return static_cast<float>(static_cast<hip_bfloat16>(v));
}

__launch_bounds__(256)
extern "C" __global__ void hyper_write_norm_f32(
    const float* __restrict__ input,
    const float* __restrict__ mixed,
    const float* __restrict__ gates,
    const float* __restrict__ quarters_in,
    float* output,
    const unsigned short* __restrict__ norm_weight,
    float* __restrict__ normalized,
    const unsigned short* __restrict__ next_norm,
    const unsigned short* __restrict__ next_inject,
    float* __restrict__ quarters_out,
    float* __restrict__ clear,
    int clear_elements,
    int branches,
    int hidden
) {
    // Grid [branches, rows]: block (branch, row) owns one branch of one token
    // row; the weights are per branch.
    const int branch = (int)blockIdx.x;
    const int row = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const int woff = branch * hidden;
    const int offset = row * branches * hidden + woff;
    mixed += (size_t)row * hidden;
    gates += (size_t)row * branches;
    if (quarters_in) quarters_in += (size_t)row * 16;
    if (quarters_out) quarters_out += (size_t)row * 16;
    // Optional: a scratch prefix the following steps expect zeroed.
    if (clear) {
        const int blocks = (int)gridDim.y * branches;
        for (int i = (row * branches + branch) * 256 + tid; i < clear_elements; i += blocks * 256)
            clear[i] = 0.0f;
    }
    __shared__ float rms_sum[256];
    __shared__ __align__(16) float xn[2560];
    const int wave = tid >> 5;
    const int lane = tid & 31;
    // Every load issues up front, including the next write's norm weights and
    // (waves 0..3) its gate row quarter: gemv_bf16_xf32_row's eight bf16 per
    // lane at lane * 8 + s * 256, s = 0..9 for K = 2560.
    float raw[10], mix[10], weight[10], next_weight[10];
    uint4 wv[10];
#pragma unroll
    for (int u = 0; u < 10; ++u) {
        raw[u] = input[offset + tid + u * 256];
        mix[u] = mixed[tid + u * 256];
        weight[u] = HF_BF16_LO((unsigned int)norm_weight[woff + tid + u * 256]);
    }
    if (quarters_out) {
#pragma unroll
        for (int u = 0; u < 10; ++u)
            next_weight[u] = HF_BF16_LO((unsigned int)next_norm[woff + tid + u * 256]);
        if (wave < 4) {
            const unsigned short* w_row = next_inject + (size_t)wave * 4 * hidden + woff;
#pragma unroll
            for (int s = 0; s < 10; ++s) wv[s] = *(const uint4*)(w_row + lane * 8 + s * 256);
        }
    }
    const float* q = quarters_in + branch * 4;
    const float gate_logit = hf_bf16(
        quarters_in ? (q[0] + q[1]) + (q[2] + q[3]) : gates[branch]);
    const float gate_scaled = hf_bf16(gate_logit / (float)branches);
    const float gate = hf_bf16(2.0f * hf_bf16(1.0f / (1.0f + expf(-gate_scaled))));
    float partial = 0.0f;
#pragma unroll
    for (int u = 0; u < 10; ++u) {
        const float injection = hf_bf16(gate * hf_bf16(mix[u]));
        raw[u] = hf_bf16(hf_bf16(raw[u]) + injection);
        output[offset + tid + u * 256] = raw[u];
        const float normed_in = hf_bf16(raw[u]);
        partial += normed_in * normed_in;
    }
    rms_sum[tid] = partial;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) rms_sum[tid] += rms_sum[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(rms_sum[0] / (float)hidden + 1.0e-6f);
#pragma unroll
    for (int u = 0; u < 10; ++u) {
        normalized[offset + tid + u * 256] = hf_bf16(hf_bf16(raw[u]) * inv * (1.0f + weight[u]));
    }
    if (!quarters_out) return;
#pragma unroll
    for (int u = 0; u < 10; ++u) {
        xn[tid + u * 256] = hf_bf16(hf_bf16(raw[u]) * inv * (1.0f + next_weight[u]));
    }
    __syncthreads();
    if (wave >= 4) return;
    // gemv_bf16_xf32_row's accumulation for K = 2560, on the preloaded row.
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
    for (int s = 0; s < 10; ++s) {
        const float4 xa = *(const float4*)(xn + lane * 8 + s * 256);
        const float4 xb = *(const float4*)(xn + lane * 8 + s * 256 + 4);
        a0 += HF_BF16_LO(wv[s].x) * xa.x;
        a1 += HF_BF16_HI(wv[s].x) * xa.y;
        a2 += HF_BF16_LO(wv[s].y) * xa.z;
        a3 += HF_BF16_HI(wv[s].y) * xa.w;
        a0 += HF_BF16_LO(wv[s].z) * xb.x;
        a1 += HF_BF16_HI(wv[s].z) * xb.y;
        a2 += HF_BF16_LO(wv[s].w) * xb.z;
        a3 += HF_BF16_HI(wv[s].w) * xb.w;
    }
    float sum = (a0 + a1) + (a2 + a3);
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down(sum, off);
    if (lane == 0) quarters_out[wave * 4 + branch] = sum;
}

// gemv_bf16_xf32 followed by bf16_scaled_add_f32 on each output row: the
// source BF16 shared-expert fold residual[row] = bf16(bf16(residual[row]) +
// bf16(bf16(scalar[0]) * bf16(W[row] . x))), the same expression, so the
// residual is bitwise the GEMV + scaled-add launches (the GEMV output itself
// is not stored).
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_bf16_xf32_bf16_scaled_add(
    const unsigned short* __restrict__ W,  // [M, K] bf16 (raw u16)
    const float*          __restrict__ x,  // [K] F32
    float*                __restrict__ residual,  // [M] F32
    const float*          __restrict__ scalar,  // [1]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const float sum = gemv_bf16_xf32_row(W + (size_t)row * K, x, K, threadIdx.x);
    if (threadIdx.x != 0) return;
    const hip_bfloat16 lhs = static_cast<hip_bfloat16>(residual[row]);
    const hip_bfloat16 rhs = static_cast<hip_bfloat16>(sum);
    const hip_bfloat16 gate = static_cast<hip_bfloat16>(scalar[0]);
    const hip_bfloat16 product = static_cast<hip_bfloat16>(
        static_cast<float>(gate) * static_cast<float>(rhs));
    residual[row] = static_cast<float>(static_cast<hip_bfloat16>(
        static_cast<float>(lhs) + static_cast<float>(product)));
}

// Four BF16 matrices reading the same x (same K) in one launch: blocks
// [0, M0) are rows of W0, the next M1 rows of W1, and so on. Each row is the
// same computation gemv_bf16_xf32 does for it.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_bf16_xf32_x4(
    const unsigned short* __restrict__ W0,
    const unsigned short* __restrict__ W1,
    const unsigned short* __restrict__ W2,
    const unsigned short* __restrict__ W3,
    const float*          __restrict__ x,
    float* __restrict__ y0,
    float* __restrict__ y1,
    float* __restrict__ y2,
    float* __restrict__ y3,
    int M0, int M1, int M2, int M3, int K
) {
    int row = blockIdx.x;
    const unsigned short* W = W0;
    float* y = y0;
    if (row >= M0) {
        row -= M0; W = W1; y = y1;
        if (row >= M1) {
            row -= M1; W = W2; y = y2;
            if (row >= M2) {
                row -= M2; W = W3; y = y3;
                if (row >= M3) return;
            }
        }
    }
    const float sum = gemv_bf16_xf32_row(W + (size_t)row * K, x, K, threadIdx.x);
    if (threadIdx.x == 0) y[row] = sum;
}

// gemv_bf16_xf32_x4 over R activation rows (x row stride K, y row stride of
// each output's M): each weight row is loaded once and dotted with every x
// row in gemv_bf16_xf32_row's exact order, so each output is bitwise the
// one-row kernel's. Unused matrices have M = 0.
template <int R>
__device__ __forceinline__ void gemv_bf16_xf32_rows_body(
    const unsigned short* __restrict__ w_row,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int row
) {
    const int tid = threadIdx.x;
    float a0[R], a1[R], a2[R], a3[R];
#pragma unroll
    for (int r = 0; r < R; ++r) a0[r] = a1[r] = a2[r] = a3[r] = 0.0f;
    if ((K & 7) == 0) {
        int i = tid * 8;
        for (; i + 8 * 32 * 8 <= K; i += 32 * 8 * 8) {
            uint4 wv[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) wv[s] = *(const uint4*)(w_row + i + s * (32 * 8));
#pragma unroll
            for (int s = 0; s < 8; ++s) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float* xr = x + (size_t)r * K;
                    const float4 xa = *(const float4*)(xr + i + s * (32 * 8));
                    const float4 xb = *(const float4*)(xr + i + s * (32 * 8) + 4);
                    a0[r] += HF_BF16_LO(wv[s].x) * xa.x;
                    a1[r] += HF_BF16_HI(wv[s].x) * xa.y;
                    a2[r] += HF_BF16_LO(wv[s].y) * xa.z;
                    a3[r] += HF_BF16_HI(wv[s].y) * xa.w;
                    a0[r] += HF_BF16_LO(wv[s].z) * xb.x;
                    a1[r] += HF_BF16_HI(wv[s].z) * xb.y;
                    a2[r] += HF_BF16_LO(wv[s].w) * xb.z;
                    a3[r] += HF_BF16_HI(wv[s].w) * xb.w;
                }
            }
        }
        for (; i + 8 <= K; i += 32 * 8) {
            const uint4 wv = *(const uint4*)(w_row + i);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float* xr = x + (size_t)r * K;
                const float4 xa = *(const float4*)(xr + i);
                const float4 xb = *(const float4*)(xr + i + 4);
                a0[r] += HF_BF16_LO(wv.x) * xa.x;
                a1[r] += HF_BF16_HI(wv.x) * xa.y;
                a2[r] += HF_BF16_LO(wv.y) * xa.z;
                a3[r] += HF_BF16_HI(wv.y) * xa.w;
                a0[r] += HF_BF16_LO(wv.z) * xb.x;
                a1[r] += HF_BF16_HI(wv.z) * xb.y;
                a2[r] += HF_BF16_LO(wv.w) * xb.z;
                a3[r] += HF_BF16_HI(wv.w) * xb.w;
            }
        }
    }
    for (int i = ((K & 7) == 0 ? (K & ~7) : 0) + tid; i < K; i += 32) {
        const float w = __uint_as_float(((unsigned int)w_row[i]) << 16);
#pragma unroll
        for (int r = 0; r < R; ++r) a0[r] += w * x[(size_t)r * K + i];
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float sum = (a0[r] + a1[r]) + (a2[r] + a3[r]);
        for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset);
        if (tid == 0) y[(size_t)r * M + row] = sum;
    }
}

#define GEMV_BF16_XF32_X4_ROWS(R)                                                        \
    __launch_bounds__(32, 12) extern "C" __global__ void gemv_bf16_xf32_x4_rows_r##R(     \
        const unsigned short* __restrict__ W0, const unsigned short* __restrict__ W1,     \
        const unsigned short* __restrict__ W2, const unsigned short* __restrict__ W3,     \
        const float* __restrict__ x, float* __restrict__ y0, float* __restrict__ y1,      \
        float* __restrict__ y2, float* __restrict__ y3, int M0, int M1, int M2, int M3,   \
        int K) {                                                                         \
        int row = blockIdx.x;                                                            \
        const unsigned short* W = W0;                                                    \
        float* y = y0;                                                                   \
        int M = M0;                                                                      \
        if (row >= M0) {                                                                 \
            row -= M0; W = W1; y = y1; M = M1;                                           \
            if (row >= M1) {                                                             \
                row -= M1; W = W2; y = y2; M = M2;                                       \
                if (row >= M2) {                                                         \
                    row -= M2; W = W3; y = y3; M = M3;                                   \
                    if (row >= M3) return;                                               \
                }                                                                        \
            }                                                                            \
        }                                                                                \
        gemv_bf16_xf32_rows_body<R>(W + (size_t)row * K, x, y, M, K, row);               \
    }
GEMV_BF16_XF32_X4_ROWS(2)
GEMV_BF16_XF32_X4_ROWS(3)
GEMV_BF16_XF32_X4_ROWS(4)
GEMV_BF16_XF32_X4_ROWS(5)
GEMV_BF16_XF32_X4_ROWS(6)
GEMV_BF16_XF32_X4_ROWS(7)
GEMV_BF16_XF32_X4_ROWS(8)
#undef GEMV_BF16_XF32_X4_ROWS
