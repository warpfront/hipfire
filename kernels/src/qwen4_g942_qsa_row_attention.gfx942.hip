// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// qwen4_g942_qsa_row_attention: the single-row parallel QSA attention entries that exact gfx942
// can select (the MTP draft head's row, `indexed_attention_attention` when the selected rows fit
// the dynamic LDS limit), as a module of their own (gfx942 only, its own cache key, its own
// frozen hipcc anchor / twin seat).
//
//   qwen4_g942_qsa_row_f32       = indexed_attention_attention_f32
//   qwen4_g942_qsa_row_fp8_fnuz  = indexed_attention_attention_fp8_fnuz
//
// gfx942 admits exactly two K/V formats for the QSA state: F32 and E4M3 FNUZ (`QSA_KV_FP8`
// is gfx12-only, `QSA_HAS_Q8` is gfx11/gfx12-only, so no other parallel row entry exists
// there). The serial fallback entries (`..._serial`, selections past the LDS limit) stay in
// tensor_ops.
//
// Every block between the `BEGIN` / `END verbatim tensor_ops.hip` markers is a byte-for-byte copy
// of tensor_ops.hip: the format helpers and key dots, the gfx942 rounding pins with
// `qsa_attention_row_parallel`, the `qsa_attention_single` wrapper and the
// `QSA_ATTENTION_SINGLE` entry macro. Only the two entry names differ (an entry cannot reuse the
// tensor_ops symbol: the per-context function table is keyed by entry name). A unit test in
// rdna-compute (`g942_qsa_row_source_is_tensor_ops_verbatim`) maps the names back and asserts
// each block occurs in TENSOR_OPS_SRC, so an edit of the production body that is not mirrored
// here fails the build. Grid [n_heads, ceil(head_dim / 256)], block 256, dynamic LDS
// `selected_len * 8` bytes (the host's `QSA_ATTENTION_LDS_BYTES_PER_ROW`).

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

// BEGIN verbatim tensor_ops.hip
#define QSA_KV_F32 0
#define QSA_KV_FP8 1
#define QSA_KV_Q8 2
#define QSA_KV_FP8_FNUZ 3
#if defined(__GFX12__) || defined(__gfx942__)
#define QSA_HAS_FP8 1
#endif
#if defined(__GFX11__) || defined(__GFX12__)
#define QSA_HAS_Q8 1
#endif

typedef float qsa_f32x2 __attribute__((ext_vector_type(2)));

template <int F>
static __device__ __forceinline__ int qsa_kv_row_bytes(int kv_heads, int head_dim) {
    if constexpr (F == QSA_KV_F32) {
        return kv_heads * head_dim * 4;
    } else if constexpr (F == QSA_KV_FP8 || F == QSA_KV_FP8_FNUZ) {
        return kv_heads * (head_dim + 2);
    } else {
        return kv_heads * (head_dim / 32) * 34;
    }
}

// Signed int8 code `byte` (0..3) of a little-endian word, as a float.
static __device__ __forceinline__ float qsa_q8_code(unsigned int word, int byte) {
    return (float)((int)(word << (24 - 8 * byte)) >> 24);
}

static __device__ __forceinline__ float qsa_half_bits(unsigned int bits) {
    return (float)__builtin_bit_cast(_Float16, (unsigned short)bits);
}

#if defined(QSA_HAS_FP8)
static __device__ __forceinline__ float qsa_fp8_code(unsigned int code) {
    const qsa_f32x2 pair = __builtin_amdgcn_cvt_pk_f32_fp8(code, false);
    return pair.x;
}
#endif

// Value channel `channel` of KV head `kv_head` in the quantized token row `row`.
template <int F>
static __device__ __forceinline__ float qsa_kv_value(
    const unsigned char* row,
    int kv_heads,
    int head_dim,
    int kv_head,
    int channel) {
    static_assert(F != QSA_KV_F32, "F32 values keep each kernel's own loads");
    if constexpr (F == QSA_KV_Q8) {
        const unsigned char* block = row + (kv_head * (head_dim / 32) + channel / 32) * 34;
        return (float)(signed char)block[2 + (channel & 31)]
            * qsa_half_bits(*reinterpret_cast<const unsigned short*>(block));
    } else {
#if defined(QSA_HAS_FP8)
    const float scale = qsa_half_bits(*reinterpret_cast<const unsigned short*>(
        row + kv_heads * head_dim + 2 * kv_head));
#if defined(__gfx942__)
    return qsa_fp8_code(row[kv_head * head_dim + channel]) * (2.0f * scale);
#else
    return qsa_fp8_code(row[kv_head * head_dim + channel]) * scale;
#endif
#else
    return 0.0f;
#endif
    }
}

// Unscaled QK dot of `heads` query rows (`q + h * q_stride`, head_dim floats
// each) with KV head `kv_head` of the quantized token row `row`, added to
// `score[h]`; every head sees the same key loads and the association above.
template <int F, int HEADS>
static __device__ __forceinline__ void qsa_kv_key_dots(
    const float* __restrict__ q,
    int q_stride,
    const unsigned char* row,
    int kv_heads,
    int head_dim,
    int kv_head,
    float (&score)[HEADS]) {
#if defined(__gfx942__)
#pragma clang fp contract(off)
    // Keep one-head and grouped-head instantiations on the same rounded
    // multiply/add chain; the compiler otherwise contracts only the latter.
#endif
    static_assert(F != QSA_KV_F32, "F32 keys keep each kernel's own loop");
    if constexpr (F == QSA_KV_Q8) {
        // A pair of blocks is 68 bytes, 4-byte aligned (the host admits q8 for
        // head_dim 256 only, so every head and token row is):
        // word 0 = scale0 | codes0[0..1], words 1..7 = codes0[2..29],
        // word 8 = codes0[30..31] | scale1, words 9..16 = codes1[0..31].
        const unsigned int* pairs =
            reinterpret_cast<const unsigned int*>(row + kv_head * (head_dim / 32) * 34);
        float total[HEADS];
#pragma unroll
        for (int h = 0; h < HEADS; ++h) total[h] = 0.0f;
        for (int pair = 0; pair < head_dim / 64; ++pair) {
            unsigned int w[17];
#pragma unroll
            for (int i = 0; i < 17; ++i) w[i] = pairs[pair * 17 + i];
            const float scale0 = qsa_half_bits(w[0] & 0xFFFFu);
            const float scale1 = qsa_half_bits(w[8] >> 16);
            float code[64];
            code[0] = qsa_q8_code(w[0], 2);
            code[1] = qsa_q8_code(w[0], 3);
#pragma unroll
            for (int i = 1; i < 8; ++i) {
#pragma unroll
                for (int b = 0; b < 4; ++b) code[4 * i - 2 + b] = qsa_q8_code(w[i], b);
            }
            code[30] = qsa_q8_code(w[8], 0);
            code[31] = qsa_q8_code(w[8], 1);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int b = 0; b < 4; ++b) code[32 + 4 * i + b] = qsa_q8_code(w[9 + i], b);
            }
#pragma unroll
            for (int h = 0; h < HEADS; ++h) {
                const float* qh = q + h * q_stride + pair * 64;
                float dot0 = 0.0f;
                float dot1 = 0.0f;
#pragma unroll
                for (int d = 0; d < 32; ++d) dot0 += qh[d] * code[d];
#pragma unroll
                for (int d = 0; d < 32; ++d) dot1 += qh[32 + d] * code[32 + d];
                total[h] += dot0 * scale0;
                total[h] += dot1 * scale1;
            }
        }
#pragma unroll
        for (int h = 0; h < HEADS; ++h) score[h] += total[h];
    } else {
#if defined(QSA_HAS_FP8)
    // Codes are 4-byte aligned: the host admits fp8 for head_dim 256 and an
    // even KV-head count only.
    const unsigned int* codes = reinterpret_cast<const unsigned int*>(row + kv_head * head_dim);
    const float scale = qsa_half_bits(*reinterpret_cast<const unsigned short*>(
        row + kv_heads * head_dim + 2 * kv_head));
    float dot[HEADS];
#pragma unroll
    for (int h = 0; h < HEADS; ++h) dot[h] = 0.0f;
#pragma unroll 4
    for (int d4 = 0; d4 < head_dim / 4; ++d4) {
        const unsigned int word = codes[d4];
        const qsa_f32x2 lo = __builtin_amdgcn_cvt_pk_f32_fp8(word, false);
        const qsa_f32x2 hi = __builtin_amdgcn_cvt_pk_f32_fp8(word, true);
#pragma unroll
        for (int h = 0; h < HEADS; ++h) {
            const float* qh = q + h * q_stride + 4 * d4;
            dot[h] += qh[0] * lo.x;
            dot[h] += qh[1] * lo.y;
            dot[h] += qh[2] * hi.x;
            dot[h] += qh[3] * hi.y;
        }
    }
#pragma unroll
#if defined(__gfx942__)
    for (int h = 0; h < HEADS; ++h) score[h] += dot[h] * (2.0f * scale);
#else
    for (int h = 0; h < HEADS; ++h) score[h] += dot[h] * scale;
#endif
#endif
    }
}
// END verbatim tensor_ops.hip

// BEGIN verbatim tensor_ops.hip
// gfx942 pins for the parallel attention kernels below (qsa_attention_row_parallel,
// _hg4, _hg12). gfx942 hipcc -O3 packs independent multiplies into
// v_pk_mul_f32 (SLP) and then adds them unfused, and fuses others into
// v_pk_fma_f32, so a bare `acc += a * b` does not compile to the same
// rounding as the serial kernel's. Observed in the gfx942 ISA of
// indexed_attention_attention_f32_batched_serial: the key-dot chain is a
// fused v_fmac chain in d order, while `numerator += weight * value` is a
// v_mul_f32 followed by a packed add (product rounded, then added).
// QSA_DOT_ACC / QSA_SUM_ACC pin exactly those two roundings in the parallel
// kernels. __fadd_rn/__fmul_rn are plain operators in this HIP toolchain
// (__clang_hip_math.h), so the unfused sum needs contract(off) to keep the
// compiler from fusing it back into an fma. Every other architecture keeps
// the bare expressions.
#if defined(__gfx942__)
static __device__ __forceinline__ float qsa_sum_acc_unfused(float acc, float w, float v) {
#pragma clang fp contract(off)
    return acc + w * v;
}
#define QSA_DOT_ACC(acc, a, b) ((acc) = __fmaf_rn((a), (b), (acc)))
#define QSA_SUM_ACC(acc, w, v) ((acc) = qsa_sum_acc_unfused((acc), (w), (v)))
#else
#define QSA_DOT_ACC(acc, a, b) ((acc) += (a) * (b))
#define QSA_SUM_ACC(acc, w, v) ((acc) += (w) * (v))
#endif

// One query row of the parallel attention below against `selected_len`
// entries of `row_selected`; rows at or past `visible` count as invalid.
template <int F>
static __device__ __forceinline__ void qsa_attention_row_parallel(
    const float* __restrict__ q,
    const unsigned char* full_keys,
    const unsigned char* full_values,
    const int* row_selected,
    float* output_row,
    int selected_len,
    int visible,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int full_capacity) {
    extern __shared__ float row_scores[];
    int* row_tokens = reinterpret_cast<int*>(row_scores + selected_len);
    __shared__ float wave_max[8];
    const int tid = (int)threadIdx.x;
    const int head = (int)blockIdx.x;
    const int kv_head = head / (n_heads / n_kv_heads);
    const int kv_stride = n_kv_heads * head_dim;
    // F32 rows are kept as cache element offsets (token * kv_stride), the
    // quantized formats' as token indices.
    const int row_unit = F == QSA_KV_F32 ? kv_stride : 1;
    const size_t row_bytes = (size_t)qsa_kv_row_bytes<F>(n_kv_heads, head_dim);
    const float* keys = reinterpret_cast<const float*>(full_keys) + kv_head * head_dim;
    float partial_max = -3.402823466e+38f;
    for (int selected_row = tid; selected_row < selected_len; selected_row += (int)blockDim.x) {
        int token = row_selected[selected_row];
        bool ok = token >= 0 && token < full_capacity && token < visible;
        row_tokens[selected_row] = ok ? token * row_unit : -1;
        float score = 0.0f;
        if (ok) {
            if constexpr (F == QSA_KV_F32) {
                const float* key = keys + token * kv_stride;
                if (head_dim == 256) {
                    // Thirty-two key float4 loads in flight per batch; the FMAs
                    // still run in serial d order.
                    const float4* key4 = reinterpret_cast<const float4*>(key);
                    const float4* q4 = reinterpret_cast<const float4*>(q);
                    for (int c = 0; c < 64; c += 32) {
                        float4 k4[32];
#pragma unroll
                        for (int u = 0; u < 32; ++u) k4[u] = key4[c + u];
#pragma unroll
                        for (int u = 0; u < 32; ++u) {
                            const float4 qv = q4[c + u];
                            QSA_DOT_ACC(score, qv.x, k4[u].x);
                            QSA_DOT_ACC(score, qv.y, k4[u].y);
                            QSA_DOT_ACC(score, qv.z, k4[u].z);
                            QSA_DOT_ACC(score, qv.w, k4[u].w);
                        }
                    }
                } else if ((head_dim & 3) == 0) {
                    for (int d = 0; d < head_dim; d += 4) {
                        const float4 k4 = *reinterpret_cast<const float4*>(key + d);
                        QSA_DOT_ACC(score, q[d], k4.x);
                        QSA_DOT_ACC(score, q[d + 1], k4.y);
                        QSA_DOT_ACC(score, q[d + 2], k4.z);
                        QSA_DOT_ACC(score, q[d + 3], k4.w);
                    }
                } else {
                    for (int d = 0; d < head_dim; ++d) QSA_DOT_ACC(score, q[d], key[d]);
                }
            } else {
                float dots[1] = {0.0f};
                qsa_kv_key_dots<F, 1>(q, 0, full_keys + (size_t)token * row_bytes, n_kv_heads,
                                      head_dim, kv_head, dots);
                score = dots[0];
            }
            score /= sqrtf((float)head_dim);
            partial_max = fmaxf(partial_max, score);
        }
        row_scores[selected_row] = score;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial_max = fmaxf(partial_max, __shfl_xor(partial_max, offset, 32));
    }
    if ((tid & 31) == 0) wave_max[tid >> 5] = partial_max;
    __syncthreads();
    float max_score = wave_max[0];
    for (int w = 1; w < ((int)blockDim.x + 31) / 32; ++w) max_score = fmaxf(max_score, wave_max[w]);
    for (int selected_row = tid; selected_row < selected_len; selected_row += (int)blockDim.x) {
        row_scores[selected_row] = row_tokens[selected_row] >= 0
            ? expf(row_scores[selected_row] - max_score)
            : 0.0f;
    }
    __syncthreads();
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (head >= n_heads || channel >= head_dim) return;
    const float* values = reinterpret_cast<const float*>(full_values) + kv_head * head_dim + channel;
    // `entry` is a row_tokens entry (>= 0).
    auto load = [&](int entry) -> float {
        if constexpr (F == QSA_KV_F32) {
            return values[(unsigned)entry];
        } else {
            return qsa_kv_value<F>(full_values + (size_t)entry * row_bytes, n_kv_heads, head_dim,
                                   kv_head, channel);
        }
    };
    float denominator = 0.0f;
    float numerator = 0.0f;
    // Value loads run one batch ahead of the in-order accumulation.
    constexpr int UNROLL = 24;
    int i = 0;
    float value[UNROLL];
    if (selected_len >= UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int offset = row_tokens[u];
            value[u] = load(offset < 0 ? 0 : offset);
        }
    }
    for (; i + UNROLL <= selected_len; i += UNROLL) {
        float next[UNROLL];
        if (i + 2 * UNROLL <= selected_len) {
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
            QSA_SUM_ACC(numerator, weight, (row_tokens[i + u] < 0 ? 0.0f : value[u]));
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) value[u] = next[u];
    }
    for (; i < selected_len; ++i) {
        const int token = row_tokens[i];
        const float weight = row_scores[i];
        const float value = token < 0 ? 0.0f : load(token);
        denominator += weight;
        QSA_SUM_ACC(numerator, weight, value);
    }
    float gate = q[head_dim + channel];
    output_row[head * head_dim + channel] =
        denominator > 0.0f ? numerator / denominator / (1.0f + expf(-gate)) : 0.0f;
}
// END verbatim tensor_ops.hip

// BEGIN verbatim tensor_ops.hip
// selections do not follow the batched length formula); same arithmetic as
// the batched kernel.
template <int F>
static __device__ __forceinline__ void qsa_attention_single(
    const float* q_with_gate,
    const unsigned char* full_keys,
    const unsigned char* full_values,
    const int* selected_indices,
    float* output,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int selected_len,
    int full_capacity) {
    qsa_attention_row_parallel<F>(
        q_with_gate + (int)blockIdx.x * head_dim * 2, full_keys, full_values,
        selected_indices, output, selected_len, full_capacity,
        n_heads, n_kv_heads, head_dim, full_capacity);
}
// END verbatim tensor_ops.hip

// BEGIN verbatim tensor_ops.hip
#define QSA_ATTENTION_SINGLE(NAME, BODY, BOUNDS, FORMAT, CACHE)                             \
    extern "C" __global__ BOUNDS void NAME(                                                 \
        const float* q_with_gate, const CACHE* full_keys, const CACHE* full_values,        \
        const int* selected_indices, float* output, int n_heads, int n_kv_heads,           \
        int head_dim, int selected_len, int full_capacity) {                                \
        BODY<FORMAT>(q_with_gate, reinterpret_cast<const unsigned char*>(full_keys),        \
                     reinterpret_cast<const unsigned char*>(full_values), selected_indices, \
                     output, n_heads, n_kv_heads, head_dim, selected_len, full_capacity);   \
    }
#define QSA_NO_BOUNDS
// END verbatim tensor_ops.hip

QSA_ATTENTION_SINGLE(qwen4_g942_qsa_row_f32, qsa_attention_single, __launch_bounds__(256),
                     QSA_KV_F32, float)
QSA_ATTENTION_SINGLE(qwen4_g942_qsa_row_fp8_fnuz, qsa_attention_single, __launch_bounds__(256),
                     QSA_KV_FP8_FNUZ, unsigned char)
