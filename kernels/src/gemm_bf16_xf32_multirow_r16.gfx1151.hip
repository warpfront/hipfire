// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact BF16-weight × F32-input gfx1151 R16 GEMM.
//
// This gfx1151-specific kernel owns sixteen output-channel rows and two token
// rows, with no output-row cooperation or LDS staging.

#include <hip/hip_runtime.h>

#define HF_BF16_LO(u) __uint_as_float((u) << 16)
#define HF_BF16_HI(u) __uint_as_float((u) & 0xFFFF0000u)
#define HF_R16_ROWS 16
#define HF_R16_STREAMS 2

// Layouts match kernels/src/gemm_bf16_xf32_multirow.hip:
//   W: [M, K] BF16, row-major
//   X: [N, K] F32, row-major
//   Y: [N, M] F32, row-major
//
// Mapping and arithmetic invariants:
//   * Grid is [ceil(N / 2), ceil(M / 16), 1], block is [32, 1, 1].  One wave
//     owns sixteen adjacent output-channel rows and two adjacent token rows,
//     so the token tile is the fast-varying grid dimension: the sixteen weight
//     rows a block group reads stay resident while the token tiles of that
//     group sweep past, which is what keeps the weight stream off DRAM.
//   * The stream loop is statically unrolled so each a0/a1/a2/a3[row][stream]
//     accumulator retains the original R4 association while each BF16 weight
//     vector is reused across both X rows and each X vector across all sixteen
//     output rows.  Sixteen output rows per wave make the activation traffic
//     per FMA four times smaller than the four-row tile's and sixteen times
//     smaller than the single-row gemv's.
//   * The wide K path is the reference lane*8 / +256 traversal.  For every
//     wide step, one lane loads and widens one uint4 of BF16 weights per
//     output row, then loads and consumes each X stream's eight floats before
//     advancing to the next output row; no array of live float4 X vectors is
//     required.  The scalar path is the reference lane / +32 traversal used
//     whenever K % 8 != 0.
//   * Every (row, stream) pair uses four accumulators, combines as
//     `(a0 + a1) + (a2 + a3)`, then performs the same shuffle offsets
//     16, 8, 4, 2, 1.  BF16 weights are widened by the same raw-bit shifts and
//     are never converted or staged through another precision, so each output
//     element is bitwise identical to the four-row
//     `gemm_bf16_xf32_multirow` result.
//   * Final partial N streams alias batch_base, the first valid input row.
//     Their results are not stored.  The `row_base + row >= M` guard keeps an
//     output row out of range at the top of M from being read or written
//     (every production M on this route is a multiple of sixteen); M/N guards
//     occur before any dereference; this kernel has no inter-thread barrier or
//     shared state.

// Lane 0 of the `v += __shfl_down(v, off, 32)` tree (off = 16..1) with the
// partner values moved by permlanex16 / DPP folded into the adds instead of
// LDS bpermute.  Same additions, operands and order; only lane 0's result is
// defined.
static __device__ __forceinline__ float hf_lane0_tree_sum(float v) {
    v += __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x108, 0xf, 0xf, true));  // row_shl:8
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x104, 0xf, 0xf, true));  // row_shl:4
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x102, 0xf, 0xf, true));  // row_shl:2
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x101, 0xf, 0xf, true));  // row_shl:1
    return v;
}

// Two lane-0 trees in one pass: lanes 0-15 carry a, lanes 16-31 carry b.
// After the cross-row step lane i holds a[i] + a[i+16] and lane 16+i holds
// b[16+i] + b[i] (the same sum, IEEE add commutes); the in-row DPP steps then
// reduce both rows with the reference offsets, so a ends in lane 0 and b in
// lane 16 bit-identical to two hf_lane0_tree_sum calls.  a is returned in
// lane 0; b is broadcast (readlane 16).
static __device__ __forceinline__ void hf_lane0_tree_sum2(float& a, float& b) {
    const bool lo = (threadIdx.x & 31) < 16;
    float v = lo ? a : b;
    const float o = lo ? b : a;
    v += __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(o), __float_as_int(o), 0x76543210u, 0xfedcba98u, false, false));
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x108, 0xf, 0xf, true));  // row_shl:8
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x104, 0xf, 0xf, true));  // row_shl:4
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x102, 0xf, 0xf, true));  // row_shl:2
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x101, 0xf, 0xf, true));  // row_shl:1
    a = v;
    b = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 16));
}

__launch_bounds__(32, 4)
extern "C" __global__ void gemm_bf16_xf32_multirow_r16_gfx1151(
    const unsigned short* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int row_base = blockIdx.y * HF_R16_ROWS;
    const int batch_base = blockIdx.x * HF_R16_STREAMS;
    const int lane = threadIdx.x;
    if (row_base >= M || batch_base >= N || lane >= 32) return;

    int eff_row[HF_R16_ROWS];
    const unsigned short* w_row[HF_R16_ROWS];
#pragma unroll
    for (int row = 0; row < HF_R16_ROWS; ++row) {
        eff_row[row] = (row_base + row < M) ? row_base + row : row_base;
        w_row[row] = W + (size_t)eff_row[row] * K;
    }

    float a0[HF_R16_ROWS][HF_R16_STREAMS];
    float a1[HF_R16_ROWS][HF_R16_STREAMS];
    float a2[HF_R16_ROWS][HF_R16_STREAMS];
    float a3[HF_R16_ROWS][HF_R16_STREAMS];
#pragma unroll
    for (int row = 0; row < HF_R16_ROWS; ++row) {
#pragma unroll
        for (int stream = 0; stream < HF_R16_STREAMS; ++stream) {
            a0[row][stream] = 0.0f;
            a1[row][stream] = 0.0f;
            a2[row][stream] = 0.0f;
            a3[row][stream] = 0.0f;
        }
    }

    if ((K & 7) == 0) {
        // Eight BF16 values per lane per step, exactly matching R4's global
        // K order.  X vectors are loaded and consumed one stream and one
        // output row at a time.
        for (int i = lane * 8; i + 8 <= K; i += 32 * 8) {
            // Widen every output row's weight vector before consuming any of
            // them: the sixteen loads are independent, so the whole group
            // issues before the first dependent accumulate stalls.
            float w[HF_R16_ROWS][8];
#pragma unroll
            for (int row = 0; row < HF_R16_ROWS; ++row) {
                const uint4 wv = *(const uint4*)(w_row[row] + i);
                w[row][0] = HF_BF16_LO(wv.x);
                w[row][1] = HF_BF16_HI(wv.x);
                w[row][2] = HF_BF16_LO(wv.y);
                w[row][3] = HF_BF16_HI(wv.y);
                w[row][4] = HF_BF16_LO(wv.z);
                w[row][5] = HF_BF16_HI(wv.z);
                w[row][6] = HF_BF16_LO(wv.w);
                w[row][7] = HF_BF16_HI(wv.w);
            }
#pragma unroll
            for (int stream = 0; stream < HF_R16_STREAMS; ++stream) {
                const int batch = (batch_base + stream < N)
                    ? batch_base + stream
                    : batch_base;
                const float* x_row = X + (size_t)batch * K;
                const float4 xa = *(const float4*)(x_row + i);
                const float4 xb = *(const float4*)(x_row + i + 4);
#pragma unroll
                for (int row = 0; row < HF_R16_ROWS; ++row) {
                    a0[row][stream] += w[row][0] * xa.x;
                    a1[row][stream] += w[row][1] * xa.y;
                    a2[row][stream] += w[row][2] * xa.z;
                    a3[row][stream] += w[row][3] * xa.w;
                    a0[row][stream] += w[row][4] * xb.x;
                    a1[row][stream] += w[row][5] * xb.y;
                    a2[row][stream] += w[row][6] * xb.z;
                    a3[row][stream] += w[row][7] * xb.w;
                }
            }
        }
    } else {
        // For K%8 != 0 the reference intentionally uses scalar traversal for
        // every element, not only the final partial eight-value group.
        for (int i = lane; i < K; i += 32) {
#pragma unroll
            for (int row = 0; row < HF_R16_ROWS; ++row) {
                const float weight =
                    __uint_as_float(((unsigned int)w_row[row][i]) << 16);
#pragma unroll
                for (int stream = 0; stream < HF_R16_STREAMS; ++stream) {
                    const int batch = (batch_base + stream < N)
                        ? batch_base + stream
                        : batch_base;
                    const float* x_row = X + (size_t)batch * K;
                    const float x = x_row[i];
                    a0[row][stream] += weight * x;
                }
            }
        }
    }

    float sum[HF_R16_ROWS][HF_R16_STREAMS];
#pragma unroll
    for (int row = 0; row < HF_R16_ROWS; ++row) {
#pragma unroll
        for (int stream = 0; stream < HF_R16_STREAMS; ++stream) {
            sum[row][stream] = (a0[row][stream] + a1[row][stream])
                + (a2[row][stream] + a3[row][stream]);
        }
    }
#pragma unroll
    for (int row = 0; row < HF_R16_ROWS; ++row) {
#pragma unroll
        for (int stream = 0; stream < HF_R16_STREAMS; ++stream) {
            sum[row][stream] = hf_lane0_tree_sum(sum[row][stream]);
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < HF_R16_ROWS; ++row) {
            if (row_base + row >= M) continue;
#pragma unroll
            for (int stream = 0; stream < HF_R16_STREAMS; ++stream) {
                if (batch_base + stream < N) {
                    Y[(size_t)(batch_base + stream) * M + row_base + row] =
                        sum[row][stream];
                }
            }
        }
    }
}

// W-LDS sibling for K % 256 == 0: four waves own adjacent token pairs of the
// same sixteen rows and share each 256-wide weight chunk through LDS instead
// of each wave re-reading it from L2 (the R16 kernel is L2-bound on that
// stream).  Every lane still consumes exactly the uint4 at lane * 8 of each
// chunk, with the R16 kernel's accumulators, order and reduction, so the
// result is bitwise identical to it.
// Block [128, 1, 1].  Whichever operand is smaller should vary slowest in
// the grid so it is the one re-read from cache:
//   r16w4:  grid [ceil(M / 16), ceil(N / 8)], row groups fastest (weights
//           re-read; 320 x 10240 at N = 1131: 3.5x faster than tokens fastest)
//   r16w4t: grid [ceil(N / 8), ceil(M / 16)], token tiles fastest (X re-read;
//           10240 x 2560 at N = 1131, where the 52 MB weight does not stay
//           cached across the row-group sweep).
template <bool TOKENS_FAST>
static __device__ __forceinline__ void r16w4_body(
    const unsigned short* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    __shared__ uint4 ws[HF_R16_ROWS * 32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int row_base = (TOKENS_FAST ? blockIdx.y : blockIdx.x) * HF_R16_ROWS;
    const int tile = TOKENS_FAST ? blockIdx.x : blockIdx.y;
    const int batch_base = (tile * 4 + (tid >> 5)) * HF_R16_STREAMS;
    // No early return: every wave takes part in the barriers.  Token pairs
    // past N alias batch 0 and are not stored.
    int batch[HF_R16_STREAMS];
#pragma unroll
    for (int stream = 0; stream < HF_R16_STREAMS; ++stream)
        batch[stream] = (batch_base + stream < N) ? batch_base + stream
                      : (batch_base < N ? batch_base : 0);

    // Staging: thread tid copies uint4 v = j * 128 + tid of the 16 x 32 chunk.
    const uint4* wsrc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int v = j * 128 + tid;
        const int r = row_base + (v >> 5);
        wsrc[j] = (const uint4*)(W + (size_t)(r < M ? r : row_base) * K) + (v & 31);
    }
    uint4 st0 = wsrc[0][0], st1 = wsrc[1][0], st2 = wsrc[2][0], st3 = wsrc[3][0];

    float a0[HF_R16_ROWS][HF_R16_STREAMS];
    float a1[HF_R16_ROWS][HF_R16_STREAMS];
    float a2[HF_R16_ROWS][HF_R16_STREAMS];
    float a3[HF_R16_ROWS][HF_R16_STREAMS];
#pragma unroll
    for (int row = 0; row < HF_R16_ROWS; ++row) {
#pragma unroll
        for (int stream = 0; stream < HF_R16_STREAMS; ++stream) {
            a0[row][stream] = 0.0f;
            a1[row][stream] = 0.0f;
            a2[row][stream] = 0.0f;
            a3[row][stream] = 0.0f;
        }
    }

    const int chunks = K / 256;
    for (int c = 0; c < chunks; ++c) {
        ws[tid] = st0;
        ws[128 + tid] = st1;
        ws[256 + tid] = st2;
        ws[384 + tid] = st3;
        __syncthreads();
        const int cn = (c + 1 < chunks) ? c + 1 : c;
        st0 = wsrc[0][cn * 32];
        st1 = wsrc[1][cn * 32];
        st2 = wsrc[2][cn * 32];
        st3 = wsrc[3][cn * 32];
        const int i = c * 256 + lane * 8;
        // Each (row, stream) chain sees the R16 kernel's exact sequence; the
        // chains are independent, so walking rows outermost keeps only one
        // widened weight vector live.
        float4 xa[HF_R16_STREAMS], xb[HF_R16_STREAMS];
#pragma unroll
        for (int stream = 0; stream < HF_R16_STREAMS; ++stream) {
            const float* x_row = X + (size_t)batch[stream] * K;
            xa[stream] = *(const float4*)(x_row + i);
            xb[stream] = *(const float4*)(x_row + i + 4);
        }
#pragma unroll
        for (int row = 0; row < HF_R16_ROWS; ++row) {
            const uint4 wv = ws[row * 32 + lane];
            const float w0 = HF_BF16_LO(wv.x), w1 = HF_BF16_HI(wv.x);
            const float w2 = HF_BF16_LO(wv.y), w3 = HF_BF16_HI(wv.y);
            const float w4 = HF_BF16_LO(wv.z), w5 = HF_BF16_HI(wv.z);
            const float w6 = HF_BF16_LO(wv.w), w7 = HF_BF16_HI(wv.w);
#pragma unroll
            for (int stream = 0; stream < HF_R16_STREAMS; ++stream) {
                a0[row][stream] += w0 * xa[stream].x;
                a1[row][stream] += w1 * xa[stream].y;
                a2[row][stream] += w2 * xa[stream].z;
                a3[row][stream] += w3 * xa[stream].w;
                a0[row][stream] += w4 * xb[stream].x;
                a1[row][stream] += w5 * xb[stream].y;
                a2[row][stream] += w6 * xb[stream].z;
                a3[row][stream] += w7 * xb[stream].w;
            }
        }
        __syncthreads();
    }

    float sum[HF_R16_ROWS][HF_R16_STREAMS];
#pragma unroll
    for (int row = 0; row < HF_R16_ROWS; ++row) {
#pragma unroll
        for (int stream = 0; stream < HF_R16_STREAMS; ++stream) {
            sum[row][stream] =
                (a0[row][stream] + a1[row][stream]) + (a2[row][stream] + a3[row][stream]);
        }
        static_assert(HF_R16_STREAMS == 2, "streams are reduced as one pair");
        hf_lane0_tree_sum2(sum[row][0], sum[row][1]);
    }
    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < HF_R16_ROWS; ++row) {
            if (row_base + row >= M) continue;
#pragma unroll
            for (int stream = 0; stream < HF_R16_STREAMS; ++stream) {
                if (batch_base + stream < N) {
                    Y[(size_t)(batch_base + stream) * M + row_base + row] =
                        sum[row][stream];
                }
            }
        }
    }
}

__launch_bounds__(128, 1)
extern "C" __global__ void gemm_bf16_xf32_multirow_r16w4_gfx1151(
    const unsigned short* __restrict__ W, const float* __restrict__ X,
    float* __restrict__ Y, int M, int K, int N) {
    r16w4_body<false>(W, X, Y, M, K, N);
}

__launch_bounds__(128, 1)
extern "C" __global__ void gemm_bf16_xf32_multirow_r16w4t_gfx1151(
    const unsigned short* __restrict__ W, const float* __restrict__ X,
    float* __restrict__ Y, int M, int K, int N) {
    r16w4_body<true>(W, X, Y, M, K, N);
}
