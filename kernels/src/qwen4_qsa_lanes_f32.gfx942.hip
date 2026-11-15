// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// qwen4_qsa_lanes_*_gfx942: the Flash-Next QSA decode core of many lanes, one
// launch per kernel instead of one per lane per kernel (exact gfx942, F32 and
// E4M3-FNUZ K/V state, F32 index keys).
//
// A lane's QSA Core is five launches on gfx942 (`execute_indexed_attention_hc_impl`):
//   prologue  indexed_attention_decode_prologue_{f32,fp8_fnuz}
//   pool      indexed_attention_pool_rope_f32
//   scores    indexed_attention_select_scores_rows8_f32   (eager live route, block_count > 0)
//   select    indexed_attention_select_from_scores       (+ the persistent-selection mirror)
//   attention indexed_attention_attention_{f32,fp8_fnuz}_batched  (per-head policy)
// Each kernel below is the production device template of that launch with the lane on
// `blockIdx.z` and every per-lane pointer / position / row count from a device lane table
// (`Qwen4QsaLaneDesc`, one per lane, cached by content) plus a by-value per-step integer
// table (`Qwen4QsaLaneInts`: position and rows of each lane). Lane `l` runs the template on
// ITS OWN state, scratch, position and rows, and writes only its own bytes, so every output
// and state byte equals the per-lane launches': ragged contexts are native, the grids only
// widen to the longest lane and a block outside its lane's extent returns before any
// barrier (a whole workgroup, never a thread subset).
//
// tensor_ops.hip is JIT-compiled by every arch and is not edited for this route, so the
// blocks between the BEGIN / END markers are verbatim copies of it (and of rope_yarn.h):
// the bf16 helpers, RopeYarn, the QSA K/V format helpers, the decode prologue, the pool,
// the rows-R score template, the parallel attention row and the BODY of
// indexed_attention_select_from_scores (a __global__ kernel cannot be called, so its body
// sits in a device function of the same parameters). A unit test in rdna-compute
// (`qsa_lanes_gfx942_source_is_tensor_ops_verbatim`) asserts each block occurs byte for
// byte in TENSOR_OPS_SRC, so an edit of a production body that is not mirrored here fails
// the build. The select_from_scores body and the attention row derivation copied from
// `qsa_attention_batched` are pinned the same way.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// BEGIN verbatim tensor_ops.hip
using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ float bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

static __device__ __forceinline__ __hip_bfloat16 bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}
// END verbatim tensor_ops.hip


// BEGIN verbatim tensor_ops.hip
struct RopeYarn {
    float freq_scale;  // 1 / factor
    float corr_low;
    float corr_high;
    float mscale;
};

// sin/cos of `angle` (= position * inv_freq[pair]) under `yarn`.
static __device__ __forceinline__ void rope_yarn_sincos(
    float angle, int pair, RopeYarn yarn, float* sine, float* cosine) {
    if (yarn.freq_scale != 1.0f) {
        float span = yarn.corr_high - yarn.corr_low;
        float ramp = fminf(fmaxf(((float)pair - yarn.corr_low)
            / (span == 0.0f ? 0.001f : span), 0.0f), 1.0f);
        angle *= ramp * yarn.freq_scale + (1.0f - ramp);
    }
    *sine = sinf(angle);
    *cosine = cosf(angle);
    if (yarn.mscale != 1.0f) {
        *sine *= yarn.mscale;
        *cosine *= yarn.mscale;
    }
}
// END verbatim tensor_ops.hip


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

// Store one channel of a K or V row in a quantized format. Every thread of
// the block calls it (block 256 over one KV head, `channel` = threadIdx.x;
// lanes at or past head_dim pass any value and store nothing). fp8: the
// head's absmax is reduced through width-32 shuffles and `amax_lds` (one float
// per 32-lane group, including both halves of a gfx942 wave64), which
// synchronizes. q8: each wave32 is one 32-channel Q8_0 block (no barrier).
template <int F>
static __device__ __forceinline__ void qsa_kv_store(
    unsigned char* row,
    float value,
    int kv_heads,
    int head_dim,
    int kv_head,
    int channel,
    float* amax_lds) {
    static_assert(F != QSA_KV_F32, "F32 rows are stored directly");
    if constexpr (F == QSA_KV_Q8) {
        const bool active = channel < head_dim;
        float amax = active ? fabsf(value) : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
        }
        const float scale = amax / 127.0f;
        const float inv_scale = amax > 0.0f ? 127.0f / amax : 0.0f;
        int q = __float2int_rn(value * inv_scale);
        q = max(-127, min(127, q));
        if (!active) return;
        unsigned char* block = row + (kv_head * (head_dim / 32) + channel / 32) * 34;
        if ((channel & 31) == 0) *reinterpret_cast<_Float16*>(block) = (_Float16)scale;
        block[2 + (channel & 31)] = (unsigned char)(signed char)q;
    } else {
#if defined(QSA_HAS_FP8)
    const bool active = channel < head_dim;
    float amax = active ? fabsf(value) : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }
    if ((threadIdx.x & 31) == 0) amax_lds[threadIdx.x >> 5] = amax;
    __syncthreads();
    amax = amax_lds[0];
    for (int wave = 1; wave < ((int)blockDim.x + 31) / 32; ++wave) {
        amax = fmaxf(amax, amax_lds[wave]);
    }
    __syncthreads();
    _Float16 s16 = amax == 0.0f ? (_Float16)1.0f : (_Float16)(amax / 448.0f);
    if (amax > 0.0f && s16 == (_Float16)0.0f) {
        s16 = __builtin_bit_cast(_Float16, (unsigned short)0x0001);
    }
    float sf = (float)s16;
    if (amax > 0.0f && 448.0f * sf < amax) {
        unsigned short bits = __builtin_bit_cast(unsigned short, s16);
        bits = bits >= 0x7bff ? (unsigned short)0x7bff : (unsigned short)(bits + 1u);
        s16 = __builtin_bit_cast(_Float16, bits);
        sf = (float)s16;
    }
    if (!active) return;
#if defined(__gfx942__)
    // FNUZ uses 0x80 for NaN, so every quantized negative zero (including
    // nonzero negative underflow) becomes 0x00. All other finite OCP codes
    // are identical after this exact half-range fold.
    row[kv_head * head_dim + channel] =
        (unsigned char)__builtin_amdgcn_cvt_pk_fp8_f32((value / sf) * 0.5f, 0.0f, 0u, false);
#else
    row[kv_head * head_dim + channel] =
        (unsigned char)__builtin_amdgcn_cvt_pk_fp8_f32(value / sf, 0.0f, 0u, false);
#endif
    if (channel == 0) {
        *reinterpret_cast<_Float16*>(row + kv_heads * head_dim + 2 * kv_head) = s16;
    }
#endif
    }
}

static __device__ __forceinline__ float qsa_index_f32(float value) {
    return value;
}

static __device__ __forceinline__ float qsa_index_f32(__hip_bfloat16 value) {
    return bf16_to_f32(value);
}

// Store a BF16-exact index-key value in an F32 or BF16 arena.
static __device__ __forceinline__ void qsa_index_store(float* slot, __hip_bfloat16 value) {
    *slot = bf16_to_f32(value);
}

static __device__ __forceinline__ void qsa_index_store(__hip_bfloat16* slot, __hip_bfloat16 value) {
    *slot = value;
}

// One head's RMSNorm + HalfSplit RoPE at `position` (block 256, one channel
// per thread; returns the channel's value, `head_dim` <= 256). Every block
// thread must call it (it synchronizes).
static __device__ __forceinline__ float qsa_norm_rope_head(
    const float* head_values,
    const __hip_bfloat16* norm,
    int head_dim,
    int position,
    int rotary_dim,
    RopeYarn yarn,
    float* input_row,
    float* row_sum) {
    const int channel = (int)threadIdx.x;
    if (channel < head_dim) {
        input_row[channel] = head_values[channel];
    }
    __syncthreads();
    // One wave runs the serial i-order sum (identical in every lane) instead
    // of every wave repeating it.
    if (threadIdx.x < 32) {
        float sum = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            float row_value = input_row[i];
            sum += row_value * row_value;
        }
        if (threadIdx.x == 0) *row_sum = sum;
    }
    __syncthreads();
    if (channel >= head_dim) return 0.0f;
    float inv = rsqrtf(*row_sum / (float)head_dim + 1.0e-6f);
    float value = input_row[channel] * inv
        * (1.0f + bf16_to_f32(norm[channel]));
    if (channel < rotary_dim) {
        int half = rotary_dim / 2;
        int partner = channel < half ? channel + half : channel - half;
        float other = input_row[partner] * inv
            * (1.0f + bf16_to_f32(norm[partner]));
        int base = channel < half ? channel : partner;
        float angle = (float)position
            * powf(10000000.0f, -((float)(2 * base)) / (float)rotary_dim);
        float sine, cosine;
        rope_yarn_sincos(angle, base, yarn, &sine, &cosine);
        value = channel < half ? value * cosine - other * sine
            : other * sine + value * cosine;
    }
    return value;
}
// END verbatim tensor_ops.hip


// BEGIN verbatim tensor_ops.hip
template <int F, typename RawKey>
static __device__ __forceinline__ void qsa_decode_prologue(
    float* index_row,
    float* qgate,
    float* keys,
    const float* values,
    unsigned char* full_keys,
    unsigned char* full_values,
    RawKey* raw_index_keys,
    const __hip_bfloat16* index_q_norm,
    const __hip_bfloat16* q_norm,
    const __hip_bfloat16* k_norm,
    int index_heads,
    int index_dim,
    int index_kv_width,
    int heads,
    int kv_heads,
    int head_dim,
    int position_start,
    RopeYarn yarn) {
    __shared__ float input_row[256];
    __shared__ float row_sum;
    __shared__ float kv_amax[8];
    const int c = (int)threadIdx.x;
    const int r = (int)blockIdx.y;
    const int position = position_start + r;
    const int index_q_width = index_heads * index_dim;
    const int kv_width = kv_heads * head_dim;
    index_row += r * (index_q_width + index_kv_width);
    qgate += r * 2 * heads * head_dim;
    keys += r * kv_width;
    values += r * kv_width;
    int block = (int)blockIdx.x;
    if (block < index_heads) {
        float* row = index_row + block * index_dim;
        const float value = qsa_norm_rope_head(row, index_q_norm, index_dim, position,
                                               min(index_dim, 64), yarn, input_row, &row_sum);
        if (c < index_dim) row[c] = value;
        return;
    }
    block -= index_heads;
    if (block < heads) {
        float* row = qgate + block * 2 * head_dim;
        const float value = qsa_norm_rope_head(row, q_norm, head_dim, position,
                                               min(head_dim, 64), yarn, input_row, &row_sum);
        if (c < head_dim) row[c] = value;
        return;
    }
    block -= heads;
    if (block < kv_heads) {
        float* row = keys + block * head_dim;
        const float value = qsa_norm_rope_head(row, k_norm, head_dim, position,
                                               min(head_dim, 64), yarn, input_row, &row_sum);
        if constexpr (F == QSA_KV_F32) {
            if (c < head_dim) {
                const int channel = block * head_dim + c;
                row[c] = value;
                reinterpret_cast<float*>(full_keys)[position * kv_width + channel] = value;
                reinterpret_cast<float*>(full_values)[position * kv_width + channel] =
                    values[channel];
            }
        } else {
            const float v = c < head_dim ? values[block * head_dim + c] : 0.0f;
            if (c < head_dim) row[c] = value;
            const size_t row_bytes = (size_t)qsa_kv_row_bytes<F>(kv_heads, head_dim);
            qsa_kv_store<F>(full_keys + (size_t)position * row_bytes, value, kv_heads, head_dim,
                            block, c, kv_amax);
            qsa_kv_store<F>(full_values + (size_t)position * row_bytes, v, kv_heads, head_dim,
                            block, c, kv_amax);
        }
        return;
    }
    for (int i = c; i < index_kv_width; i += 256) {
        float v = index_row[index_q_width + i];
        if (isfinite(v)) {
            v = __uint_as_float((__float_as_uint(v) + 0x7FFFu + ((__float_as_uint(v) >> 16) & 1u))
                                & 0xFFFF0000u);
            index_row[index_q_width + i] = v;
        }
        if constexpr (F == QSA_KV_F32 || F == QSA_KV_FP8_FNUZ) {
            raw_index_keys[position * index_kv_width + i] = v;
        } else {
            raw_index_keys[(size_t)position * index_kv_width + i] = bf16_round(v);
        }
    }
}
// END verbatim tensor_ops.hip


// BEGIN verbatim tensor_ops.hip
template <typename Key>
static __device__ __forceinline__ void qsa_pool_rope(
    const Key* raw_keys,
    Key* pooled,
    const __hip_bfloat16* norm,
    int block_count,
    int compress,
    int index_dim,
    int first_block,
    RopeYarn yarn) {
    // The grid starts at `first_block` (0 for the full pool): blocks below it
    // already hold this kernel's output for their unchanged raw keys (the
    // incremental decode caller), so an incremental launch covers only the
    // blocks its rows complete.
    int block = first_block + (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if ( block >= block_count || compress <= 0) return;
    bool active = channel < index_dim;
    int base = block * compress * index_dim;
    float mean = 0.0f;
    if (active) {
        for (int row = 0; row < compress; ++row) {
            mean += qsa_index_f32(raw_keys[base + row * index_dim + channel])
                / (float)compress;
        }
    }
    __hip_bfloat16 mean_bf16 = bf16_round(mean);
    __shared__ float partial[256];
    float squared = active
        ? bf16_to_f32(bf16_round(bf16_to_f32(mean_bf16)
            * bf16_to_f32(mean_bf16)))
        : 0.0f;
    partial[threadIdx.x] = squared;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (!active) return;
    float inv = rsqrtf(partial[0] / (float)index_dim + 1.0e-6f);
    float norm_value = norm == nullptr
        ? 0.0f
        : bf16_to_f32(norm[channel]);
    float value = bf16_to_f32(bf16_round(
        bf16_to_f32(mean_bf16) * inv * (1.0f + norm_value)));
    int rotary_dim = index_dim < 64 ? index_dim : 64;
    int half = rotary_dim / 2;
    if (channel < rotary_dim) {
        int partner = channel < half ? channel + half : channel - half;
        float other_mean = 0.0f;
        for (int row = 0; row < compress; ++row) {
            other_mean += qsa_index_f32(raw_keys[base + row * index_dim + partner])
                / (float)compress;
        }
        __hip_bfloat16 other_bf16 = bf16_round(other_mean);
        float other_norm = norm == nullptr
            ? 0.0f
            : bf16_to_f32(norm[partner]);
        float other = bf16_to_f32(bf16_round(
            bf16_to_f32(other_bf16) * inv * (1.0f + other_norm)));
        int pair = channel < half ? channel : partner;
        float angle = (float)(block * compress)
            * powf(10000000.0f, -((float)(2 * pair)) / (float)rotary_dim);
        float sine, cosine;
        rope_yarn_sincos(angle, pair, yarn, &sine, &cosine);
        value = channel < half ? value * cosine - other * sine
            : other * sine + value * cosine;
    }
    qsa_index_store(pooled + block * index_dim + channel, bf16_round(value));
}
// END verbatim tensor_ops.hip


// BEGIN verbatim tensor_ops.hip
template <int R>
__device__ __forceinline__ void indexed_attention_select_scores_rows(
    const float* query,
    const float* pooled,
    float* scores,
    int rows,
    int query_row_stride,
    int block_count,
    int index_dim,
    int compress,
    int position_start,
    int score_stride) {
    __shared__ __align__(16) float qs[R * 4 * 128];
    const int row0 = (int)blockIdx.y * R;
    const int nrows = min(R, rows - row0);
    const int width = 4 * index_dim;
    for (int i = (int)threadIdx.x; i < nrows * width; i += 256) {
        const int r = i / width;
        qs[r * 4 * 128 + (i - r * width) / index_dim * 128 + (i - r * width) % index_dim] =
            query[(row0 + r) * query_row_stride + (i - r * width)];
    }
    __syncthreads();
    const int block = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (block >= block_count) return;
    const float4* p4 = reinterpret_cast<const float4*>(pooled + block * index_dim);
    float dot[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int h = 0; h < 4; ++h) dot[r][h] = 0.0f;
    for (int d4 = 0; d4 < index_dim / 4; ++d4) {
        const float4 p = p4[d4];
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const float4 a = *reinterpret_cast<const float4*>(qs + r * 4 * 128 + h * 128 + d4 * 4);
                dot[r][h] += a.x * p.x;
                dot[r][h] += a.y * p.y;
                dot[r][h] += a.z * p.z;
                dot[r][h] += a.w * p.w;
            }
        }
    }
    // No break/continue: the loop must fully unroll so `dot` stays in
    // registers (with them, the 16-row kernel spills it to scratch).
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int visible_blocks = (position_start + row0 + r + 1) / compress;
        if (r < nrows && block < visible_blocks) {
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < 4; ++h) score += fmaxf(dot[r][h] / sqrtf((float)index_dim), 0.0f);
            scores[(row0 + r) * score_stride + block] = score;
        }
    }
}
// END verbatim tensor_ops.hip


// BEGIN verbatim tensor_ops.hip
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


// select_from_scores: the production kernel's parameters and body, as a device function.
static __device__ __forceinline__ void qsa_lanes_select_from_scores_body(
    const float* scores,
    int score_stride,
    int* selected,
    int rows,
    int block_count,
    int budget_blocks,
    int compress,
    int position_start,
    int capacity,
    int* mirror) {
// BEGIN verbatim tensor_ops.hip
    const int row = (int)blockIdx.x;
    if (row >= rows) return;
    if (row != rows - 1) mirror = nullptr;
    __shared__ unsigned hist[8][256];  // one copy per wave: fewer atomic collisions
    __shared__ int wave_totals[8];
    __shared__ int wave_equals[8];
    __shared__ unsigned radix_prefix;
    __shared__ int radix_need;
    __shared__ int2 entries[512];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int visible = position_start + row + 1;
    const int row_block_count = min(visible / compress, block_count);
    const int chosen = min(budget_blocks, row_block_count);
    const unsigned* keys = reinterpret_cast<const unsigned*>(scores + row * score_stride);
    int* row_selected = selected + row * capacity;
    for (int i = tid; i < capacity; i += 256) {
        row_selected[i] = -1;
        if (mirror) mirror[i] = -1;
    }
    if (chosen > 0) {
        unsigned prefix = 0u;
        int need = chosen;
        for (int shift = 24; shift >= 0; shift -= 8) {
            const unsigned high = shift == 24 ? 0u : ~0u << (shift + 8);
#pragma unroll
            for (int w = 0; w < 8; ++w) hist[w][tid] = 0u;
            __syncthreads();
            for (int block = tid; block < row_block_count; block += 256) {
                const unsigned key = keys[block];
                if ((key & high) == prefix) atomicAdd(&hist[wave][(key >> shift) & 255u], 1u);
            }
            __syncthreads();
            // Thread t owns digit 255 - t: an inclusive scan over t gives the
            // count at or above each digit; the digit where it first reaches
            // `need` holds the chosen-th largest key.
            int count = 0;
#pragma unroll
            for (int w = 0; w < 8; ++w) count += (int)hist[w][255 - tid];
            int at_or_above = count;
            for (int offset = 1; offset < 32; offset <<= 1) {
                const int up = __shfl_up(at_or_above, offset, 32);
                if (lane >= offset) at_or_above += up;
            }
            if (lane == 31) wave_totals[wave] = at_or_above;
            __syncthreads();
            for (int w = 0; w < wave; ++w) at_or_above += wave_totals[w];
            if (at_or_above >= need && at_or_above - count < need) {
                radix_prefix = prefix | ((unsigned)(255 - tid) << shift);
                radix_need = need - (at_or_above - count);
            }
            __syncthreads();
            prefix = radix_prefix;
            need = radix_need;
        }
        // prefix = T; `need` keys equal to T are taken, every key above T is.
        const int above = chosen - need;
        const int per = (row_block_count + 255) / 256;
        const int first = min(tid * per, row_block_count);
        const int last = min(first + per, row_block_count);
        // Above and equal counts scan separately: ties can number every block
        // (65,536 at 262K context), which overflows a 16-bit packed field.
        int above_count = 0;
        int equal_count = 0;
        for (int block = first; block < last; ++block) {
            const unsigned key = keys[block];
            above_count += key > prefix;
            equal_count += key == prefix;
        }
        int above_inclusive = above_count;
        int equal_inclusive = equal_count;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int above_up = __shfl_up(above_inclusive, offset, 32);
            const int equal_up = __shfl_up(equal_inclusive, offset, 32);
            if (lane >= offset) {
                above_inclusive += above_up;
                equal_inclusive += equal_up;
            }
        }
        if (lane == 31) {
            wave_totals[wave] = above_inclusive;
            wave_equals[wave] = equal_inclusive;
        }
        __syncthreads();
        int above_slot = above_inclusive - above_count;
        int equal_seen = equal_inclusive - equal_count;
        for (int w = 0; w < wave; ++w) {
            above_slot += wave_totals[w];
            equal_seen += wave_equals[w];
        }
        for (int block = first; block < last; ++block) {
            const unsigned key = keys[block];
            if (key > prefix) {
                entries[above_slot++] = make_int2((int)key, block);
            } else if (key == prefix) {
                if (equal_seen < need) entries[above + equal_seen] = make_int2((int)key, block);
                ++equal_seen;
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int i = tid + k * 256;
            if (i >= chosen) break;
            const int2 entry = entries[i];
            int rank = i;
            if (i < above) {
                rank = 0;
                const unsigned key = (unsigned)entry.x;
                for (int j = 0; j < above; ++j) {
                    const int2 other = entries[j];
                    rank += (unsigned)other.x > key
                        || ((unsigned)other.x == key && other.y < entry.y);
                }
            }
            for (int slot = 0; slot < compress; ++slot) {
                const int index = rank * compress + slot;
                const int token = entry.y * compress + slot;
                if (index < capacity && token < visible) {
                    row_selected[index] = token;
                    if (mirror) mirror[index] = token;
                }
            }
        }
    }
    __syncthreads();  // the -1 fill above precedes the tail writes
    const int tail_start = row_block_count * compress;
    const int offset = chosen * compress;
    for (int token = tail_start + tid;
         token < visible && offset + (token - tail_start) < capacity;
         token += 256) {
        row_selected[offset + (token - tail_start)] = token;
        if (mirror) mirror[offset + (token - tail_start)] = token;
    }
// END verbatim tensor_ops.hip
}

// ---------------------------------------------------------------------------------------------
// Lane wrappers. Everything below is the batching, not arithmetic.
// ---------------------------------------------------------------------------------------------

// Most lanes one launch covers: the by-value `Qwen4QsaLaneInts` is 8 bytes per slot.
#define QWEN4_QSA_LANES_MAX 128

// One lane's pointers (`struct Qwen4QsaLaneDesc` in rdna-compute cdna/gfx942.rs, eleven
// 8-byte pointers). State: the K/V caches, the F32 raw / pooled index keys, the persistent
// selection. Scratch: the lane's FIRST row of each row scratch (rows are contiguous inside a
// lane) and its selected-index rows.
struct Qwen4QsaLaneDesc {
    unsigned char* full_keys;
    unsigned char* full_values;
    float* raw_idx;
    float* pooled;
    int* sel_persist;
    float* index_row;
    float* qgate;
    float* keys;
    const float* values;
    int* selected;
    float* output;
};

// Per-step scalars of every lane: `state.position` (position_start) and the lane's rows
// (1..=8). Unused slots are zero. `block_count = (pos + rows) / compress` and the pool's
// `first_block = pos / compress` derive from them exactly as the per-lane launches' do.
struct Qwen4QsaLaneInts {
    int pos[QWEN4_QSA_LANES_MAX];
    int rows[QWEN4_QSA_LANES_MAX];
};

// Every pointer a lane kernel dereferences (the lane table's pointers, the shared norms and the
// score scratch) is device global memory. The table's pointers are loaded from memory, so the
// compiler cannot prove an address space for them and would emit FLAT loads and stores; the cast
// through `address_space(1)` and back states the fact, address-space inference then makes every
// access through them a GLOBAL instruction. Same address, same value: no arithmetic changes.
template <typename T>
static __device__ __forceinline__ T* qsa_lanes_global(T* p) {
    return (T*)(__attribute__((address_space(1))) T*)(unsigned long)p;
}

// Lane `l`'s descriptor, every pointer in global memory.
static __device__ __forceinline__ Qwen4QsaLaneDesc qwen4_qsa_lanes_desc(
    const Qwen4QsaLaneDesc* table, int l) {
    Qwen4QsaLaneDesc d = table[l];
    d.full_keys = qsa_lanes_global(d.full_keys);
    d.full_values = qsa_lanes_global(d.full_values);
    d.raw_idx = qsa_lanes_global(d.raw_idx);
    d.pooled = qsa_lanes_global(d.pooled);
    d.sel_persist = qsa_lanes_global(d.sel_persist);
    d.index_row = qsa_lanes_global(d.index_row);
    d.qgate = qsa_lanes_global(d.qgate);
    d.keys = qsa_lanes_global(d.keys);
    d.values = qsa_lanes_global(d.values);
    d.selected = qsa_lanes_global(d.selected);
    d.output = qsa_lanes_global(d.output);
    return d;
}

// prologue: grid [index_heads + heads + kv_heads + 1, max_rows, lanes], block 256.
template <int F>
static __device__ __forceinline__ void qwen4_qsa_lanes_prologue(
    const Qwen4QsaLaneDesc* table,
    const __hip_bfloat16* index_q_norm,
    const __hip_bfloat16* q_norm,
    const __hip_bfloat16* k_norm,
    int index_heads,
    int index_dim,
    int index_kv_width,
    int heads,
    int kv_heads,
    int head_dim,
    RopeYarn yarn,
    Qwen4QsaLaneInts ints) {
    const int l = (int)blockIdx.z;
    if ((int)blockIdx.y >= ints.rows[l]) return;
    const Qwen4QsaLaneDesc d = qwen4_qsa_lanes_desc(table, l);
    qsa_decode_prologue<F, float>(
        d.index_row, d.qgate, d.keys, d.values, d.full_keys, d.full_values, d.raw_idx,
        index_q_norm, q_norm, k_norm, index_heads, index_dim, index_kv_width, heads, kv_heads,
        head_dim, ints.pos[l], yarn);
}

#define QWEN4_QSA_LANES_PROLOGUE(NAME, FORMAT)                                                \
    extern "C" __global__ __launch_bounds__(256) void NAME(                                   \
        const Qwen4QsaLaneDesc* table, const __hip_bfloat16* index_q_norm,                    \
        const __hip_bfloat16* q_norm, const __hip_bfloat16* k_norm, int index_heads,          \
        int index_dim, int index_kv_width, int heads, int kv_heads, int head_dim,             \
        RopeYarn yarn, Qwen4QsaLaneInts ints) {                                               \
        qwen4_qsa_lanes_prologue<FORMAT>(table, index_q_norm, q_norm, k_norm, index_heads,    \
                                         index_dim, index_kv_width, heads, kv_heads,          \
                                         head_dim, yarn, ints);                               \
    }
QWEN4_QSA_LANES_PROLOGUE(qwen4_qsa_lanes_prologue_f32_gfx942, QSA_KV_F32)
QWEN4_QSA_LANES_PROLOGUE(qwen4_qsa_lanes_prologue_fp8_fnuz_gfx942, QSA_KV_FP8_FNUZ)
#undef QWEN4_QSA_LANES_PROLOGUE

// pool: grid [max_l ceil(rows / compress), ceil(index_dim / 256), lanes], block 256. The
// template returns, before its first read or barrier, for every block at or past the lane's
// block count (surplus blocks of the widest lane, and a lane whose rows complete no block).
extern "C" __global__ void qwen4_qsa_lanes_pool_f32_gfx942(
    const Qwen4QsaLaneDesc* table,
    const __hip_bfloat16* norm,
    int compress,
    int index_dim,
    RopeYarn yarn,
    Qwen4QsaLaneInts ints) {
    const int l = (int)blockIdx.z;
    const Qwen4QsaLaneDesc d = qwen4_qsa_lanes_desc(table, l);
    qsa_pool_rope<float>(d.raw_idx, d.pooled, norm, (ints.pos[l] + ints.rows[l]) / compress,
                         compress, index_dim, ints.pos[l] / compress, yarn);
}

// scores: grid [ceil(max_blocks / 256), 1, lanes], block 256 (rows <= 8 per lane: one
// rows8 tile, as the per-lane launch). Lane `l` writes its rows into its own region
// `scores + l * lane_stride` at row stride `score_stride`; no lane reads another's.
extern "C" __global__ __launch_bounds__(256) void qwen4_qsa_lanes_scores_rows8_f32_gfx942(
    const Qwen4QsaLaneDesc* table,
    float* scores,
    int lane_stride,
    int score_stride,
    int query_row_stride,
    int index_dim,
    int compress,
    Qwen4QsaLaneInts ints) {
    const int l = (int)blockIdx.z;
    const int pos = ints.pos[l];
    const int rows = ints.rows[l];
    const int block_count = (pos + rows) / compress;
    // A whole workgroup past the lane's blocks: the per-lane grid does not have it.
    if ((int)blockIdx.x * 256 >= block_count) return;
    const Qwen4QsaLaneDesc d = qwen4_qsa_lanes_desc(table, l);
    indexed_attention_select_scores_rows<8>(d.index_row, d.pooled,
                                            scores + (size_t)l * (size_t)lane_stride, rows,
                                            query_row_stride, block_count, index_dim, compress,
                                            pos, score_stride);
}

// select: grid [max_rows, 1, lanes], block 256.
extern "C" __global__ __launch_bounds__(256) void qwen4_qsa_lanes_select_from_scores_gfx942(
    const Qwen4QsaLaneDesc* table,
    const float* scores,
    int lane_stride,
    int score_stride,
    int budget_blocks,
    int compress,
    int capacity,
    Qwen4QsaLaneInts ints) {
    const int l = (int)blockIdx.z;
    const int pos = ints.pos[l];
    const int rows = ints.rows[l];
    if ((int)blockIdx.x >= rows) return;
    const Qwen4QsaLaneDesc d = qwen4_qsa_lanes_desc(table, l);
    qsa_lanes_select_from_scores_body(scores + (size_t)l * (size_t)lane_stride, score_stride,
                                      d.selected, rows, (pos + rows) / compress, budget_blocks,
                                      compress, pos, capacity, d.sel_persist);
}

// attention: grid [heads, 1, lanes * max_rows], block 256, dynamic LDS = the longest
// selected list * 8 bytes. Lane = blockIdx.z / max_rows, row = blockIdx.z % max_rows (the
// template owns blockIdx.y as the channel chunk). `visible` and the selected length are
// `qsa_attention_batched`'s per-row derivation with the lane's own position.
template <int F>
static __device__ __forceinline__ void qwen4_qsa_lanes_attention(
    const Qwen4QsaLaneDesc* table,
    int max_rows,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int budget_blocks,
    int compress,
    int capacity,
    int full_capacity,
    Qwen4QsaLaneInts ints) {
    const int l = (int)blockIdx.z / max_rows;
    const int row = (int)blockIdx.z - l * max_rows;
    if (row >= ints.rows[l]) return;
    const Qwen4QsaLaneDesc d = qwen4_qsa_lanes_desc(table, l);
    int visible = ints.pos[l] + row + 1;
    int block_count = visible / compress;
    int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    int selected_len = chosen * compress + visible - block_count * compress;
    if (selected_len > capacity) selected_len = capacity;
    qsa_attention_row_parallel<F>(
        d.qgate + row * 2 * n_heads * head_dim + (int)blockIdx.x * head_dim * 2, d.full_keys,
        d.full_values, d.selected + row * capacity, d.output + row * n_heads * head_dim,
        selected_len, visible, n_heads, n_kv_heads, head_dim, full_capacity);
}

#define QWEN4_QSA_LANES_ATTENTION(NAME, FORMAT)                                               \
    extern "C" __global__ __launch_bounds__(256) void NAME(                                   \
        const Qwen4QsaLaneDesc* table, int max_rows, int n_heads, int n_kv_heads,             \
        int head_dim, int budget_blocks, int compress, int capacity, int full_capacity,       \
        Qwen4QsaLaneInts ints) {                                                              \
        qwen4_qsa_lanes_attention<FORMAT>(table, max_rows, n_heads, n_kv_heads, head_dim,     \
                                          budget_blocks, compress, capacity, full_capacity,   \
                                          ints);                                              \
    }
QWEN4_QSA_LANES_ATTENTION(qwen4_qsa_lanes_attention_f32_gfx942, QSA_KV_F32)
QWEN4_QSA_LANES_ATTENTION(qwen4_qsa_lanes_attention_fp8_fnuz_gfx942, QSA_KV_FP8_FNUZ)
#undef QWEN4_QSA_LANES_ATTENTION
