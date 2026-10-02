// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Qwen4 decode/verify siblings of gemv_q8_0 (gemv_q8_0.hip) and the Q8_0
// draft-head re-score (topk8_select.hip is prepended to this source).

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

template <int R>
static __device__ __forceinline__ void gemv_q8_0_row_multi(
    const unsigned char* row_data, const float* __restrict__ x, int x_stride, int K, int tid,
    float (&sum)[R]);

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

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q8_0_k2560_staged(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M
) {
    if ((int)blockIdx.x >= M) return;
    gemv_q8_0_k2560_staged_row(A_q8, x, y, blockIdx.x);
}

// Two [M x 2560] matrices of one activation in one launch (grid 2M): rows
// [0, M) of A0 then of A1, each bitwise gemv_q8_0_k2560_staged's.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q8_0_k2560_staged_pair(
    const unsigned char* __restrict__ A0,
    const unsigned char* __restrict__ A1,
    const float* __restrict__ x,
    float* __restrict__ y0,
    float* __restrict__ y1,
    int M
) {
    const int row = blockIdx.x;
    if (row < M) {
        gemv_q8_0_k2560_staged_row(A0, x, y0, row);
    } else if (row < 2 * M) {
        gemv_q8_0_k2560_staged_row(A1, x, y1, row - M);
    }
}

// gemv_q8_0 for K = 320 (340-byte rows, Qwen4 decode HC read up), staged like
// gemv_q8_0_k2560_staged; four rows (waves) per block, the row's loads issued
// before the x loads, x held in registers. 132 -> 218 GB/s on gfx1151 against
// gemv_q8_0_wide.
__launch_bounds__(128)
extern "C" __global__ void gemv_q8_0_k320_staged(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M
) {
    __shared__ __align__(16) unsigned int staged[4][85];
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + wave;
    if (row >= M) return;
    const unsigned int* src = reinterpret_cast<const unsigned int*>(A_q8 + (size_t)row * 340);
    const unsigned int v0 = src[lane];
    const unsigned int v1 = src[lane + 32];
    const unsigned int v2 = lane < 21 ? src[lane + 64] : 0u;
    float xr[10];
#pragma unroll
    for (int b = 0; b < 10; ++b) xr[b] = x[b * 32 + lane];
    staged[wave][lane] = v0;
    staged[wave][lane + 32] = v1;
    if (lane < 21) staged[wave][lane + 64] = v2;
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
    const unsigned char* data = reinterpret_cast<const unsigned char*>(staged[wave]);
    // gemv_q8_0_row's per-lane order over the ten blocks and its tree.
    float sum = 0.0f;
#pragma unroll
    for (int b = 0; b < 10; ++b) {
        const unsigned char* block = data + b * 34;
        const float d = (float)*((const _Float16*)block);
        const signed char qval = (signed char)block[2 + lane];
        sum += d * (float)qval * xr[b];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (lane == 0) y[row] = sum;
}

// Long-K rows (Qwen4 decode HC read down, K = 4 * 2560): eight waves take an
// eighth of each row each, the partial sums fold ((p0 + p1) + (p2 + p3)) +
// ((p4 + p5) + (p6 + p7)), and `hc_act` applies the HC activation (BF16 round
// trip, `scale`, round trip, SiLU, round trip) as gemv_bf16_xf32_k4 does.
// Grid [M], block 256, K % 256 == 0.
__launch_bounds__(256)
extern "C" __global__ void gemv_q8_0_k8(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K,
    float scale,
    int hc_act
) {
    __shared__ float part[8];
    const int row = blockIdx.x;
    if (row >= M) return;
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int pk = K / 8;
    const float s = gemv_q8_0_row(
        A_q8 + ((size_t)row * 8 + wave) * (pk / 32) * 34, x + wave * pk, pk, lane);
    if (lane == 0) part[wave] = s;
    __syncthreads();
    if (threadIdx.x != 0) return;
    float value = ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]));
    if (hc_act) {
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
        value *= scale;
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
        value = value / (1.0f + expf(-value));
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
    }
    y[row] = value;
}

// Multi-row variants of the staged / split decode kernels above: each weight
// row is loaded once and dotted with `rows` activation rows (x row stride K,
// y row stride M). Every row's value is bitwise the single-row kernel's.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q8_0_k2560_staged_rows(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int rows
) {
    __shared__ __align__(16) unsigned int staged_row[680];
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const unsigned int* src = reinterpret_cast<const unsigned int*>(A_q8 + (size_t)row * 2720);
    unsigned int v[22];
#pragma unroll
    for (int u = 0; u < 21; ++u) v[u] = src[tid + u * 32];
    v[21] = tid < 8 ? src[tid + 672] : 0u;
#pragma unroll
    for (int u = 0; u < 21; ++u) staged_row[tid + u * 32] = v[u];
    if (tid < 8) staged_row[tid + 672] = v[21];
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
    for (int r = 0; r < rows; ++r) {
        const float sum = gemv_q8_0_row(
            reinterpret_cast<const unsigned char*>(staged_row), x + (size_t)r * 2560, 2560, tid);
        if (tid == 0) y[(size_t)r * M + row] = sum;
    }
}

__launch_bounds__(64)
extern "C" __global__ void gemv_q8_0_k320_staged_rows(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int rows
) {
    __shared__ __align__(16) unsigned int staged[2][85];
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 2 + wave;
    if (row >= M) return;
    const unsigned int* src = reinterpret_cast<const unsigned int*>(A_q8 + (size_t)row * 340);
    const unsigned int v0 = src[lane];
    const unsigned int v1 = src[lane + 32];
    const unsigned int v2 = lane < 21 ? src[lane + 64] : 0u;
    staged[wave][lane] = v0;
    staged[wave][lane + 32] = v1;
    if (lane < 21) staged[wave][lane + 64] = v2;
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
    const unsigned char* data = reinterpret_cast<const unsigned char*>(staged[wave]);
    switch (rows) {
#define K320_ROWS(R)                                                                   \
    case R: {                                                                          \
        float s[R];                                                                    \
        gemv_q8_0_row_multi<R>(data, x, 320, 320, lane, s);                            \
        if (lane == 0)                                                                 \
            for (int r = 0; r < R; ++r) y[(size_t)r * M + row] = s[r];                 \
        return;                                                                        \
    }
    K320_ROWS(2) K320_ROWS(3) K320_ROWS(4) K320_ROWS(5) K320_ROWS(6) K320_ROWS(7) K320_ROWS(8)
#undef K320_ROWS
    }
    for (int r = 0; r < rows; ++r) {
        const float sum = gemv_q8_0_row(data, x + (size_t)r * 320, 320, lane);
        if (lane == 0) y[(size_t)r * M + row] = sum;
    }
}

// gemv_q8_0_row over R activation rows: each (scale, quant) pair is loaded
// once and folded into every row's sum in the single-row order.
template <int R>
static __device__ __forceinline__ void gemv_q8_0_row_multi(
    const unsigned char* row_data,
    const float* __restrict__ x,
    int x_stride,
    int K,
    int tid,
    float (&sum)[R]
) {
    const int blocks_per_row = K / 32;
#pragma unroll
    for (int r = 0; r < R; ++r) sum[r] = 0.0f;
    const int outer_iters = blocks_per_row / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_data + oi * 8 * 34;
        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const unsigned char* block = base + sub * 34;
            float d = (float)*((const _Float16*)block);
            signed char qval = (signed char)block[2 + tid];
            const float* xb = x + oi * 256 + sub * 32 + tid;
#pragma unroll
            for (int r = 0; r < R; ++r)
                sum[r] += d * (float)qval * xb[r * x_stride];
        }
    }
    for (int bi = outer_iters * 8; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + tid];
#pragma unroll
        for (int r = 0; r < R; ++r)
            sum[r] += d * (float)qval * x[r * x_stride + bi * 32 + tid];
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
        for (int offset = 16; offset > 0; offset >>= 1)
            sum[r] += __shfl_down(sum[r], offset);
}

template <int R>
static __device__ __forceinline__ void gemv_q8_0_k8_rows_body(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K,
    float scale,
    int hc_act
) {
    __shared__ float part[R][8];
    const int row = blockIdx.x;
    if (row >= M) return;
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int pk = K / 8;
    float s[R];
    gemv_q8_0_row_multi<R>(A_q8 + ((size_t)row * 8 + wave) * (pk / 32) * 34, x + wave * pk,
                           K, pk, lane, s);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) part[r][wave] = s[r];
    }
    __syncthreads();
    const int r = threadIdx.x;
    if (r >= R) return;
    float value = ((part[r][0] + part[r][1]) + (part[r][2] + part[r][3])) + ((part[r][4] + part[r][5]) + (part[r][6] + part[r][7]));
    if (hc_act) {
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
        value *= scale;
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
        value = value / (1.0f + expf(-value));
        value = static_cast<float>(static_cast<hip_bfloat16>(value));
    }
    y[(size_t)r * M + row] = value;
}

#define GEMV_Q8_0_K8_ROWS(R)                                                        \
    __launch_bounds__(256) extern "C" __global__ void gemv_q8_0_k8_rows_r##R(        \
        const unsigned char* __restrict__ A_q8, const float* __restrict__ x,        \
        float* __restrict__ y, int M, int K, float scale, int hc_act) {             \
        gemv_q8_0_k8_rows_body<R>(A_q8, x, y, M, K, scale, hc_act);                 \
    }
GEMV_Q8_0_K8_ROWS(2)
GEMV_Q8_0_K8_ROWS(3)
GEMV_Q8_0_K8_ROWS(4)
GEMV_Q8_0_K8_ROWS(5)
GEMV_Q8_0_K8_ROWS(6)
GEMV_Q8_0_K8_ROWS(7)
GEMV_Q8_0_K8_ROWS(8)
#undef GEMV_Q8_0_K8_ROWS

// Requantize BF16 to Q8_0, one thread per 32-block: scale = f16(amax / 127),
// q = round(w / scale) with the stored (f16) scale.
extern "C" __global__ void quantize_bf16_q8_0(
    const unsigned short* __restrict__ w,
    unsigned char* __restrict__ out,
    int blocks
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    float v[32];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        v[i] = __uint_as_float((unsigned int)w[(size_t)b * 32 + i] << 16);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    const _Float16 d16 = (_Float16)(amax / 127.0f);
    const float d = (float)d16;
    unsigned char* o = out + (size_t)b * 34;
    *reinterpret_cast<_Float16*>(o) = d16;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int q = d == 0.0f ? 0 : max(-127, min(127, (int)rintf(v[i] / d)));
        o[2 + i] = (unsigned char)(signed char)q;
    }
}

// Grid G, block 256: block b's top 8 of v[b*chunk, (b+1)*chunk) -> pv/pi[b*8 + j].
// chunk <= 256 * 8. Row j of v is token j, except for a head reordered as
// [0, front) ++ [special, special + tail) ++ [front, special): pi gets token ids.
__launch_bounds__(256)
extern "C" __global__ void topk8_partial_f32(
    const float* __restrict__ v, int n, int chunk, float* __restrict__ pv, int* __restrict__ pi,
    int front, int special, int tail
) {
    __shared__ float sv[8];
    __shared__ int si[8];
    const int begin = blockIdx.x * chunk;
    const int end = min(n, begin + chunk);
    float val[8];
    int idx[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const int j = begin + threadIdx.x + u * 256;
        const bool ok = j < end;
        val[u] = ok ? v[j] : -INFINITY;
        idx[u] = ok ? j : 0x7fffffff;
    }
    for (int round = 0; round < 8; ++round) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int u = 0; u < 8; ++u)
            if (topk8_better(val[u], idx[u], bv, bi)) { bv = val[u]; bi = idx[u]; }
        topk8_block_best(bv, bi, sv, si);
#pragma unroll
        for (int u = 0; u < 8; ++u)
            if (idx[u] == bi) { val[u] = -INFINITY; idx[u] = 0x7fffffff; }
        if (threadIdx.x == 0) {
            const int token = bi < front ? bi : (bi < front + tail ? bi - front + special : bi - tail);
            pv[blockIdx.x * 8 + round] = bv;
            pi[blockIdx.x * 8 + round] = token;
        }
    }
}

// One block of 256: the top 8 of m <= 2048 partial candidates, each scored
// exactly against its Q8_0 row (K = 2560, the staged decode kernel's dot),
// the best written to out[0].
__launch_bounds__(256)
extern "C" __global__ void topk8_rescore_q8_0_k2560(
    const float* __restrict__ pv, const int* __restrict__ pi, int m,
    const unsigned char* __restrict__ A_q8, const float* __restrict__ x, int* __restrict__ out
) {
    __shared__ int cand[8];
    __shared__ float exact[8];
    topk8_select(pv, pi, m, cand);
    const int wave = threadIdx.x >> 5;
    const float s = gemv_q8_0_row(A_q8 + (size_t)cand[wave] * 2720, x, 2560, threadIdx.x & 31);
    if ((threadIdx.x & 31) == 0) exact[wave] = s;
    __syncthreads();
    if (threadIdx.x == 0) topk8_finish(exact, cand, out);
}
