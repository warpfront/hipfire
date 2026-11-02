// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Qwen4 decode MoE head, exact gfx1151 (HIPFIRE_QWEN4_DECODE_HEAD_FUSE): the
// BF16 router + selector rows (`gemv_bf16_xf32_x4` with M2 = M3 = 0, module
// qwen4_gemv_bf16_xf32) and the Q8_0 shared gate/up pair
// (`gemv_q8_0_k2560_staged_pair`, module qwen4_gemv_q8_0) in one launch.
//
// Self-contained: the device bodies below are verbatim copies of the
// incumbents' (the incumbent .hip files are not edited; their source hashes
// are pinned). Each copy names its source file and function; a change to an
// incumbent body MUST be mirrored here (the hardware test
// `qwen4_decode_head_fuse_bitwise_incumbent`, crates/rdna-compute/tests/
// qwen4_decode_head_fuse.rs, compares all four outputs bitwise).

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// ── Mirror of kernels/src/qwen4_gemv_bf16_xf32.hip ───────────────────────
// bf16 -> f32 is a lossless 16-bit shift. A dwordx4 holds 8 bf16: within each
// u32 the LOW half is the even element, the HIGH half the odd one (LE).
#define HF_BF16_LO(u) __uint_as_float((u) << 16)
#define HF_BF16_HI(u) __uint_as_float((u) & 0xFFFF0000u)

// Mirror of `gemv_bf16_xf32_row` (qwen4_gemv_bf16_xf32.hip).
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

// ── Mirror of kernels/src/qwen4_gemv_q8_0.hip ────────────────────────────
// Mirror of `gemv_q8_0_row` (qwen4_gemv_q8_0.hip).
// One row's dot (valid in lane 0) from `row_data` (global or LDS).
static __device__ __forceinline__ float gemv_q8_0_row(
    const unsigned char* row_data,
    const float* __restrict__ x,
    int K,
    int tid
) {
    const int blocks_per_row = K / 32;
    float sum = 0.0f;

    // Process 8 Q8_0 blocks (256 elements) per outer iteration
    const int outer_iters = blocks_per_row / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_data + oi * 8 * 34;
        const float* xb = x + oi * 256;

        // Unrolled: 8 blocks, each: load scale (f16→f32), load byte, FMA
        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const unsigned char* block = base + sub * 34;
            float d = (float)*((const _Float16*)block);
            signed char qval = (signed char)block[2 + tid];
            sum += d * (float)qval * xb[sub * 32 + tid];
        }
    }

    // Handle remaining blocks (if K is not multiple of 256)
    for (int bi = outer_iters * 8; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + tid];
        sum += d * (float)qval * x[bi * 32 + tid];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    return sum;
}

// Mirror of `gemv_q8_0_k2560_staged_row` (qwen4_gemv_q8_0.hip).
// gemv_q8_0 for K = 2560 (2720-byte rows): the wave first copies its row into
// LDS with dword loads, every load before the first store (32 x 4 B per load
// instead of the row loop's 32 x 1 B quant loads and broadcast scale loads),
// then runs the same row loop on the copy.
static __device__ __forceinline__ void gemv_q8_0_k2560_staged_row(
    const unsigned char* __restrict__ A_q8, const float* __restrict__ x, float* __restrict__ y,
    int row
) {
    __shared__ __align__(16) unsigned int staged_row[680];
    const int tid = threadIdx.x;
    const unsigned int* src = reinterpret_cast<const unsigned int*>(A_q8 + (size_t)row * 2720);
    unsigned int v[22];
#pragma unroll
    for (int u = 0; u < 21; ++u) v[u] = src[tid + u * 32];
    v[21] = tid < 8 ? src[tid + 672] : 0u;
#pragma unroll
    for (int u = 0; u < 21; ++u) staged_row[tid + u * 32] = v[u];
    if (tid < 8) staged_row[tid + 672] = v[21];
    // One wave: its LDS accesses complete in order, so no workgroup fence.
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
    const float sum = gemv_q8_0_row(
        reinterpret_cast<const unsigned char*>(staged_row), x, 2560, tid);
    if (tid == 0) y[row] = sum;
}

// ── The fused entry ──────────────────────────────────────────────────────
// Grid M0 + M1 + 2 * MQ, block 32. Blocks [0, M0) are rows of W0, the next M1
// rows of W1 (the body and routing of `gemv_bf16_xf32_x4`, qwen4_gemv_bf16_xf32.hip,
// with M2 = M3 = 0), then rows [0, MQ) of A0 and rows [0, MQ) of A1
// (`gemv_q8_0_k2560_staged_pair`, qwen4_gemv_q8_0.hip; K = 2560 Q8_0 rows).
// Every output row is bitwise the one its incumbent launch writes.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_bf16_xf32_x2_q8_0_k2560_staged_pair(
    const unsigned short* __restrict__ W0,
    const unsigned short* __restrict__ W1,
    const unsigned char* __restrict__ A0,
    const unsigned char* __restrict__ A1,
    const float*          __restrict__ x,
    float* __restrict__ y0,
    float* __restrict__ y1,
    float* __restrict__ q0,
    float* __restrict__ q1,
    int M0, int M1, int MQ, int K
) {
    int row = blockIdx.x;
    if (row < M0 + M1) {
        // gemv_bf16_xf32_x4 body.
        const unsigned short* W = W0;
        float* y = y0;
        if (row >= M0) { row -= M0; W = W1; y = y1; }
        const float sum = gemv_bf16_xf32_row(W + (size_t)row * K, x, K, threadIdx.x);
        if (threadIdx.x == 0) y[row] = sum;
        return;
    }
    // gemv_q8_0_k2560_staged_pair body (grid 2 * MQ).
    row -= M0 + M1;
    if (row < MQ) {
        gemv_q8_0_k2560_staged_row(A0, x, q0, row);
    } else if (row < 2 * MQ) {
        gemv_q8_0_k2560_staged_row(A1, x, q1, row - MQ);
    }
}
