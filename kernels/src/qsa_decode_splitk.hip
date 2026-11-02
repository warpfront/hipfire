// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Split-K sparse-attention (QSA) decode for the F32 K/V cache, head_dim 256,
// 1..8 query rows. The selected list of one (row, head) is cut into `splits`
// contiguous slices; each slice is a partial softmax and a combine kernel
// merges them. Selected-list length, validity, key scaling and the gate
// follow `qsa_attention_batched` in tensor_ops.hip exactly:
//   visible = position_start + row + 1
//   block_count = visible / compress
//   chosen = min(budget_blocks, block_count)
//   len = min(capacity, chosen * compress + visible - block_count * compress)
// A selected token that is negative, >= full_capacity or >= visible is
// invalid: weight 0 and value 0. Every output row depends only on its own
// position, never on the total row count.
//
// Partial kernel  qsa_decode_splitk_partial_f32
//   grid [n_heads, splits, rows], block 256,
//   dynamic LDS 8 * ceil(max_selected / splits) bytes
//   (float scores[n] then int tokens[n], n = this slice's length).
//   Workspace record ((row * n_heads + head) * splits + s) * 258 floats:
//   [0] = m (slice max score; -FLT_MAX when the slice has no valid token),
//   [1] = l (sum of exp(score - m); exactly 0 when empty),
//   [2..257] = acc (sum of weight * value per channel in slice order;
//   exactly 0 when empty).
//   scoring_layout 0: one thread per token, serial d-order dot (the same
//     float4 batches of 32 loads as the incumbent), so each score is
//     bitwise equal to the incumbent's.
//   scoring_layout 1: eight wave32 token groups; each lane owns 8 channels
//     (two coalesced float4) and the per-token dot is a shuffle reduction.
//     Scores differ from layout 0 in the last bits (tolerance-gated).
//   scoring_layout 2: grid [n_kv_heads, splits, rows]; one block serves the 12
//     query heads of a kv head (n_heads == 12 * n_kv_heads, head_dim 256),
//     loading each K row and V element once. Score dot uses the same xor
//     add tree as layout 1 (numerical parity is tested); workspace records
//     are unchanged. See splitk_partial_kv_group.
// Combine kernel  qsa_decode_splitk_combine_f32
//   grid [n_heads, 1, rows], block 256, no LDS. Merges slices in ascending
//   order (slices with l <= 0 ignored), divides by the merged l, applies
//   1 / (1 + exp(-gate)); a row with no valid token outputs 0.

#include <hip/hip_runtime.h>

#define SPLITK_FLT_LOWEST (-3.402823466e+38f)
#define SPLITK_WS_STRIDE 258
#define SPLITK_HD 256

// gfx942 rounding pins (see QSA_DOT_ACC / QSA_SUM_ACC in tensor_ops.hip).
#if defined(__gfx942__)
static __device__ __forceinline__ float splitk_sum_acc_unfused(float acc, float w, float v) {
#pragma clang fp contract(off)
    return acc + w * v;
}
#define SPLITK_DOT_ACC(acc, a, b) ((acc) = __fmaf_rn((a), (b), (acc)))
#define SPLITK_SUM_ACC(acc, w, v) ((acc) = splitk_sum_acc_unfused((acc), (w), (v)))
#else
#define SPLITK_DOT_ACC(acc, a, b) ((acc) += (a) * (b))
#define SPLITK_SUM_ACC(acc, w, v) ((acc) += (w) * (v))
#endif

// scoring_layout 2: one block per (kv head, split, row) serves the 12 query
// heads that share that kv head. Each selected K row is loaded once for all
// 12 dots and each selected V element once for all 12 accumulators. Workspace
// records are the same per-(row, query head, split) 258-float records the
// per-head layouts write, so the combine kernel is unchanged.
//   grid [n_kv_heads, splits, rows], block 256, head_dim 256,
//   n_heads == 12 * n_kv_heads,
//   dynamic LDS 52 * ceil(max_selected / splits) bytes
//   (float weights[12][n] then int tokens[n]), static LDS 384 bytes here
//   (maxes[12][8]) plus the kernel's own 32 (wave_max) = 416.
// Scores use the same per-lane 8-product sum and the same xor-16,8,4,2,1 add
// tree as scoring_layout 1, run as a 16-value transposed reduction; numerical
// parity with layout 1 is tested, not asserted here.
#define SPLITK_GROUP 12

static __device__ __forceinline__ void splitk_partial_kv_group(
    float* __restrict__ lds,
    const float* __restrict__ q_with_gate,
    const float* __restrict__ full_keys,
    const float* __restrict__ full_values,
    const int* __restrict__ selected,
    float* __restrict__ workspace,
    int rows,
    int position_start,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int budget_blocks,
    int compress,
    int capacity,
    int full_capacity,
    int splits) {
    __shared__ float maxes[SPLITK_GROUP][8];
    const int tid = (int)threadIdx.x;
    const int kv_head = (int)blockIdx.x;
    const int split = (int)blockIdx.y;
    const int row = (int)blockIdx.z;
    if (row >= rows || kv_head >= n_kv_heads || split >= splits) return;
    if (head_dim != SPLITK_HD || n_heads != n_kv_heads * SPLITK_GROUP) return;

    const int visible = position_start + row + 1;
    const int block_count = visible / compress;
    const int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    int len = chosen * compress + visible - block_count * compress;
    if (len > capacity) len = capacity;
    const int per = (len + splits - 1) / splits;
    const int begin = split * per;
    const int end = min(len, begin + per);
    const int n = end > begin ? end - begin : 0;

    const int head0 = kv_head * SPLITK_GROUP;
    float* ws = workspace + (size_t)((row * n_heads + head0) * splits + split) * SPLITK_WS_STRIDE;
    const size_t ws_head_step = (size_t)splits * SPLITK_WS_STRIDE;
    const int channel = tid;
    if (n == 0) {
        if (tid < SPLITK_GROUP) {
            float* rec = ws + (size_t)tid * ws_head_step;
            rec[0] = SPLITK_FLT_LOWEST;
            rec[1] = 0.0f;
        }
#pragma unroll
        for (int j = 0; j < SPLITK_GROUP; ++j) ws[(size_t)j * ws_head_step + 2 + channel] = 0.0f;
        return;
    }

    float* weights = lds;
    int* row_tokens = reinterpret_cast<int*>(lds + SPLITK_GROUP * n);
    const int kv_stride = n_kv_heads * head_dim;
    const int* slice_selected = selected + row * capacity + begin;
    const float* keys = full_keys + kv_head * head_dim;
    const float inv_scale_div = sqrtf((float)head_dim);

    // Scoring: wave32 owns tokens wave, wave + 8, ...; lane owns channels
    // [4*lane..) and [128 + 4*lane..). After the 12 dots every lane holds a
    // partial for each head; a transposed butterfly (16 values padded with
    // zeros) leaves lane L with head idx = 8*b4 + 4*b3 + 2*b2 + b1 (bit k of
    // L), and the final xor-1 step makes lanes 2m and 2m+1 agree.
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const bool b4 = (lane & 16) != 0;
    const bool b3 = (lane & 8) != 0;
    const bool b2 = (lane & 4) != 0;
    const bool b1 = (lane & 2) != 0;
    const int idx = (b4 ? 8 : 0) + (b3 ? 4 : 0) + (b2 ? 2 : 0) + (b1 ? 1 : 0);
    const bool writer = (lane & 1) == 0 && idx < SPLITK_GROUP;
    float4 qa[SPLITK_GROUP];
    float4 qb[SPLITK_GROUP];
#pragma unroll
    for (int j = 0; j < SPLITK_GROUP; ++j) {
        const float4* q4 = reinterpret_cast<const float4*>(
            q_with_gate + row * 2 * n_heads * head_dim + (head0 + j) * head_dim * 2);
        qa[j] = q4[lane];
        qb[j] = q4[lane + 32];
    }
    float lane_max = SPLITK_FLT_LOWEST;
    constexpr int TB = 2;
    for (int base = wave; base < n; base += 8 * TB) {
        float4 ka[TB];
        float4 kb[TB];
        int tok[TB];
#pragma unroll
        for (int u = 0; u < TB; ++u) {
            const int i = base + 8 * u;
            int token = -1;
            ka[u] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            kb[u] = ka[u];
            if (i < n) {
                const int t = slice_selected[i];
                if (t >= 0 && t < full_capacity && t < visible) {
                    token = t;
                    const float4* key4 = reinterpret_cast<const float4*>(keys + (size_t)t * kv_stride);
                    ka[u] = key4[lane];
                    kb[u] = key4[lane + 32];
                }
            }
            tok[u] = token;
        }
#pragma unroll
        for (int u = 0; u < TB; ++u) {
            const int i = base + 8 * u;
            if (i < n) {
                float v[16];
#pragma unroll
                for (int j = 0; j < SPLITK_GROUP; ++j) {
                    float p = qa[j].x * ka[u].x;
                    p += qa[j].y * ka[u].y;
                    p += qa[j].z * ka[u].z;
                    p += qa[j].w * ka[u].w;
                    p += qb[j].x * kb[u].x;
                    p += qb[j].y * kb[u].y;
                    p += qb[j].z * kb[u].z;
                    p += qb[j].w * kb[u].w;
                    v[j] = p;
                }
#pragma unroll
                for (int j = SPLITK_GROUP; j < 16; ++j) v[j] = 0.0f;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const float send = b4 ? v[k] : v[k + 8];
                    const float keep = b4 ? v[k + 8] : v[k];
                    v[k] = keep + __shfl_xor(send, 16, 32);
                }
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float send = b3 ? v[k] : v[k + 4];
                    const float keep = b3 ? v[k + 4] : v[k];
                    v[k] = keep + __shfl_xor(send, 8, 32);
                }
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const float send = b2 ? v[k] : v[k + 2];
                    const float keep = b2 ? v[k + 2] : v[k];
                    v[k] = keep + __shfl_xor(send, 4, 32);
                }
                {
                    const float send = b1 ? v[0] : v[1];
                    const float keep = b1 ? v[1] : v[0];
                    v[0] = keep + __shfl_xor(send, 2, 32);
                }
                v[0] += __shfl_xor(v[0], 1, 32);
                const float score = tok[u] >= 0 ? v[0] / inv_scale_div : 0.0f;
                if (writer) {
                    weights[idx * n + i] = score;
                    if (tok[u] >= 0) lane_max = fmaxf(lane_max, score);
                }
                if (lane == 0) row_tokens[i] = tok[u] >= 0 ? tok[u] * kv_stride : -1;
            }
        }
    }
    if (writer) maxes[idx][wave] = lane_max;
    __syncthreads();
    if (tid < SPLITK_GROUP) {
        float m = maxes[tid][0];
#pragma unroll
        for (int w = 1; w < 8; ++w) m = fmaxf(m, maxes[tid][w]);
        maxes[tid][0] = m;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < SPLITK_GROUP; ++j) {
        const float m = maxes[j][0];
        float* wj = weights + j * n;
        for (int i = tid; i < n; i += (int)blockDim.x) {
            wj[i] = row_tokens[i] >= 0 ? expf(wj[i] - m) : 0.0f;
        }
    }
    __syncthreads();

    // Value phase: thread = channel; each selected V element is loaded once
    // and feeds all 12 (denominator, numerator) chains in ascending order.
    const float* values = full_values + kv_head * head_dim + channel;
    auto load = [&](int entry) -> float { return values[(unsigned)entry]; };
    float den[SPLITK_GROUP];
    float num[SPLITK_GROUP];
#pragma unroll
    for (int j = 0; j < SPLITK_GROUP; ++j) {
        den[j] = 0.0f;
        num[j] = 0.0f;
    }
    auto accumulate = [&](int i, float value) {
#pragma unroll
        for (int j = 0; j < SPLITK_GROUP; ++j) {
            const float weight = weights[j * n + i];
            den[j] += weight;
            SPLITK_SUM_ACC(num[j], weight, value);
        }
    };
    constexpr int UNROLL = 12;
    int i = 0;
    float value[UNROLL];
    if (n >= UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int offset = row_tokens[u];
            value[u] = load(offset < 0 ? 0 : offset);
        }
    }
    for (; i + UNROLL <= n; i += UNROLL) {
        float next[UNROLL];
        if (i + 2 * UNROLL <= n) {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const int offset = row_tokens[i + UNROLL + u];
                next[u] = load(offset < 0 ? 0 : offset);
            }
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            accumulate(i + u, row_tokens[i + u] < 0 ? 0.0f : value[u]);
        }
        if (i + 2 * UNROLL <= n) {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) value[u] = next[u];
        }
    }
    for (; i < n; ++i) {
        const int token = row_tokens[i];
        accumulate(i, token < 0 ? 0.0f : load(token));
    }
#pragma unroll
    for (int j = 0; j < SPLITK_GROUP; ++j) {
        float* rec = ws + (size_t)j * ws_head_step;
        if (tid == 0) {
            rec[0] = maxes[j][0];
            rec[1] = den[j];
        }
        rec[2 + channel] = num[j];
    }
}

extern "C" __global__ __launch_bounds__(256) void qsa_decode_splitk_partial_f32(
    const float* __restrict__ q_with_gate,
    const float* __restrict__ full_keys,
    const float* __restrict__ full_values,
    const int* __restrict__ selected,
    float* __restrict__ workspace,
    int rows,
    int position_start,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int budget_blocks,
    int compress,
    int capacity,
    int full_capacity,
    int splits,
    int scoring_layout) {
    extern __shared__ float splitk_lds[];
    __shared__ float wave_max[8];
    const int tid = (int)threadIdx.x;
    const int head = (int)blockIdx.x;
    const int split = (int)blockIdx.y;
    const int row = (int)blockIdx.z;
    if (scoring_layout == 2) {
        splitk_partial_kv_group(splitk_lds, q_with_gate, full_keys, full_values, selected, workspace,
                                rows, position_start, n_heads, n_kv_heads, head_dim, budget_blocks,
                                compress, capacity, full_capacity, splits);
        return;
    }
    if (row >= rows || head >= n_heads || split >= splits) return;

    const int visible = position_start + row + 1;
    const int block_count = visible / compress;
    const int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    int len = chosen * compress + visible - block_count * compress;
    if (len > capacity) len = capacity;
    const int per = (len + splits - 1) / splits;
    const int begin = split * per;
    const int end = min(len, begin + per);
    const int n = end > begin ? end - begin : 0;

    float* ws = workspace + (size_t)((row * n_heads + head) * splits + split) * SPLITK_WS_STRIDE;
    const int channel = tid;
    if (n == 0) {
        if (tid == 0) {
            ws[0] = SPLITK_FLT_LOWEST;
            ws[1] = 0.0f;
        }
        if (channel < head_dim) ws[2 + channel] = 0.0f;
        return;
    }

    float* row_scores = splitk_lds;
    int* row_tokens = reinterpret_cast<int*>(splitk_lds + n);
    const int kv_head = head / (n_heads / n_kv_heads);
    const int kv_stride = n_kv_heads * head_dim;
    const float* q = q_with_gate + row * 2 * n_heads * head_dim + head * head_dim * 2;
    const int* slice_selected = selected + row * capacity + begin;
    const float* keys = full_keys + kv_head * head_dim;
    const float inv_scale_div = sqrtf((float)head_dim);

    float partial_max = SPLITK_FLT_LOWEST;
    if (scoring_layout == 1 && head_dim == SPLITK_HD) {
        // Eight wave32 token groups; lane owns channels [4*lane..) and
        // [128 + 4*lane..) as two coalesced float4. Four tokens in flight.
        const int wave = tid >> 5;
        const int lane = tid & 31;
        const float4* q4 = reinterpret_cast<const float4*>(q);
        const float4 qa = q4[lane];
        const float4 qb = q4[lane + 32];
        constexpr int TB = 4;
        for (int base = wave; base < n; base += 8 * TB) {
            float4 ka[TB];
            float4 kb[TB];
            int tok[TB];
#pragma unroll
            for (int u = 0; u < TB; ++u) {
                const int i = base + 8 * u;
                int token = -1;
                ka[u] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                kb[u] = ka[u];
                if (i < n) {
                    const int t = slice_selected[i];
                    if (t >= 0 && t < full_capacity && t < visible) {
                        token = t;
                        const float4* key4 = reinterpret_cast<const float4*>(keys + (size_t)t * kv_stride);
                        ka[u] = key4[lane];
                        kb[u] = key4[lane + 32];
                    }
                }
                tok[u] = token;
            }
#pragma unroll
            for (int u = 0; u < TB; ++u) {
                const int i = base + 8 * u;
                if (i < n) {
                    float score = 0.0f;
                    if (tok[u] >= 0) {
                        float p = qa.x * ka[u].x;
                        p += qa.y * ka[u].y;
                        p += qa.z * ka[u].z;
                        p += qa.w * ka[u].w;
                        p += qb.x * kb[u].x;
                        p += qb.y * kb[u].y;
                        p += qb.z * kb[u].z;
                        p += qb.w * kb[u].w;
                        for (int offset = 16; offset > 0; offset >>= 1) {
                            p += __shfl_xor(p, offset, 32);
                        }
                        score = p / inv_scale_div;
                    }
                    if (lane == 0) {
                        row_tokens[i] = tok[u] >= 0 ? tok[u] * kv_stride : -1;
                        row_scores[i] = score;
                        if (tok[u] >= 0) partial_max = fmaxf(partial_max, score);
                    }
                }
            }
        }
    } else {
        // One thread per token; serial d-order dot.
        for (int i = tid; i < n; i += (int)blockDim.x) {
            const int token = slice_selected[i];
            const bool ok = token >= 0 && token < full_capacity && token < visible;
            row_tokens[i] = ok ? token * kv_stride : -1;
            float score = 0.0f;
            if (ok) {
                const float* key = keys + (size_t)token * kv_stride;
                if (head_dim == SPLITK_HD) {
                    // Thirty-two key float4 loads in flight per batch; the
                    // FMAs still run in serial d order.
                    const float4* key4 = reinterpret_cast<const float4*>(key);
                    const float4* q4 = reinterpret_cast<const float4*>(q);
                    for (int c = 0; c < 64; c += 32) {
                        float4 k4[32];
#pragma unroll
                        for (int u = 0; u < 32; ++u) k4[u] = key4[c + u];
#pragma unroll
                        for (int u = 0; u < 32; ++u) {
                            const float4 qv = q4[c + u];
                            SPLITK_DOT_ACC(score, qv.x, k4[u].x);
                            SPLITK_DOT_ACC(score, qv.y, k4[u].y);
                            SPLITK_DOT_ACC(score, qv.z, k4[u].z);
                            SPLITK_DOT_ACC(score, qv.w, k4[u].w);
                        }
                    }
                } else {
                    for (int d = 0; d < head_dim; ++d) SPLITK_DOT_ACC(score, q[d], key[d]);
                }
                score /= inv_scale_div;
                partial_max = fmaxf(partial_max, score);
            }
            row_scores[i] = score;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial_max = fmaxf(partial_max, __shfl_xor(partial_max, offset, 32));
    }
    if ((tid & 31) == 0) wave_max[tid >> 5] = partial_max;
    __syncthreads();
    float max_score = wave_max[0];
    for (int w = 1; w < ((int)blockDim.x + 31) / 32; ++w) max_score = fmaxf(max_score, wave_max[w]);
    for (int i = tid; i < n; i += (int)blockDim.x) {
        row_scores[i] = row_tokens[i] >= 0 ? expf(row_scores[i] - max_score) : 0.0f;
    }
    __syncthreads();
    if (channel >= head_dim) return;

    const float* values = full_values + kv_head * head_dim + channel;
    // `entry` is a row_tokens entry (>= 0), a cache element offset.
    auto load = [&](int entry) -> float { return values[(unsigned)entry]; };
    float denominator = 0.0f;
    float numerator = 0.0f;
    // Value loads run one batch ahead of the in-order accumulation.
    constexpr int UNROLL = 24;
    int i = 0;
    float value[UNROLL];
    if (n >= UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int offset = row_tokens[u];
            value[u] = load(offset < 0 ? 0 : offset);
        }
    }
    for (; i + UNROLL <= n; i += UNROLL) {
        float next[UNROLL];
        if (i + 2 * UNROLL <= n) {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const int offset = row_tokens[i + UNROLL + u];
                next[u] = load(offset < 0 ? 0 : offset);
            }
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const float weight = row_scores[i + u];
            denominator += weight;
            SPLITK_SUM_ACC(numerator, weight, (row_tokens[i + u] < 0 ? 0.0f : value[u]));
        }
        if (i + 2 * UNROLL <= n) {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) value[u] = next[u];
        }
    }
    for (; i < n; ++i) {
        const int token = row_tokens[i];
        const float weight = row_scores[i];
        const float v = token < 0 ? 0.0f : load(token);
        denominator += weight;
        SPLITK_SUM_ACC(numerator, weight, v);
    }
    if (tid == 0) {
        ws[0] = max_score;
        ws[1] = denominator;
    }
    ws[2 + channel] = numerator;
}

extern "C" __global__ __launch_bounds__(256) void qsa_decode_splitk_combine_f32(
    const float* __restrict__ q_with_gate,
    const float* __restrict__ workspace,
    float* __restrict__ output,
    int rows,
    int n_heads,
    int head_dim,
    int splits) {
    const int head = (int)blockIdx.x;
    const int row = (int)blockIdx.z;
    const int channel = (int)threadIdx.x;
    if (row >= rows || head >= n_heads || channel >= head_dim) return;
    const float* ws = workspace + (size_t)((row * n_heads + head) * splits) * SPLITK_WS_STRIDE;

    float global_max = SPLITK_FLT_LOWEST;
    bool any = false;
    for (int s = 0; s < splits; ++s) {
        const float* rec = ws + s * SPLITK_WS_STRIDE;
        if (rec[1] > 0.0f) {
            global_max = any ? fmaxf(global_max, rec[0]) : rec[0];
            any = true;
        }
    }
    float denominator = 0.0f;
    float numerator = 0.0f;
    if (any) {
        for (int s = 0; s < splits; ++s) {
            const float* rec = ws + s * SPLITK_WS_STRIDE;
            const float l = rec[1];
            if (l > 0.0f) {
                const float scale = expf(rec[0] - global_max);
                denominator += l * scale;
                numerator += rec[2 + channel] * scale;
            }
        }
    }
    const float gate =
        q_with_gate[row * 2 * n_heads * head_dim + head * head_dim * 2 + head_dim + channel];
    output[row * n_heads * head_dim + head * head_dim + channel] =
        denominator > 0.0f ? numerator / denominator / (1.0f + expf(-gate)) : 0.0f;
}
