// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx11-only structural FA2 experiment: wide query rows and register-resident
// Q, while retaining ordinary f16 WMMA arithmetic. Q8_0 K/V codes and their
// f16 scales are staged verbatim in LDS; code*scale is rounded to f16 only
// while each WMMA fragment is built. There is no integer WMMA arithmetic and
// no requantization of Q or probabilities.
//
// Geometry (H=24, Hkv=4, D=256 only):
//   block: 512 threads / 16 wave32 / 256 query-head rows
//   grid:  [ceil(6*batch/256), 4, 1]
//   KT:    128 keys
//   D:     two D128 passes (8 16-dimension output chunks per pass)
//   LDS:   K codes+scales (128*(256+16) = 34,816 B)
//          V D128 codes+scales (128*(128+8) = 17,408 B)
//          total 52,224 B, one block/CU and therefore 16 resident waves/CU
//
// Every wave owns sixteen query-head rows through the measured gfx11 wave32
// WMMA lane-pair basis: lane r and r+16 are the two output parities of one
// row and must supply identical full half16 operands. Each lane keeps its own
// half of Q (8 f16 values per D16 chunk); xor16 reconstructs the complete
// identical Q fragment immediately before WMMA. O is D128-split so resident Q
// plus f32 output accumulators remain below the 256-VGPR architectural cap.

#include <hip/hip_runtime.h>

#define HIPFIRE_FA2_WIDE_THREADS 512
#define HIPFIRE_FA2_WIDE_KT 128
#define HIPFIRE_FA2_WIDE_DCS 8

typedef _Float16 __attribute__((ext_vector_type(8))) fa2w_half8_t;
typedef _Float16 __attribute__((ext_vector_type(16))) fa2w_half16_t;
typedef float __attribute__((ext_vector_type(8))) fa2w_float8_t;
typedef uint32_t __attribute__((ext_vector_type(4))) fa2w_u32x4_t;

typedef union {
    uint32_t u;
    _Float16 h[2];
} fa2w_dw_t;

typedef union {
    fa2w_u32x4_t v;
    uint32_t w[4];
    signed char s[16];
    unsigned char u[16];
} fa2w_bytes16_t;

static __device__ __forceinline__ _Float16 fa2w_scale(uint16_t u)
{
    fa2w_dw_t x;
    x.u = (uint32_t)u;
    return x.h[0];
}

extern "C" __global__ __launch_bounds__(HIPFIRE_FA2_WIDE_THREADS, 1)
void attention_q8_0_fa2_gqa_wide_gfx11(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int batch_size,
    float scale_attn)
{
    constexpr int KT = HIPFIRE_FA2_WIDE_KT;
    constexpr int THREADS = HIPFIRE_FA2_WIDE_THREADS;
    constexpr int DCS = HIPFIRE_FA2_WIDE_DCS;
    constexpr int ROWS_PER_WG = THREADS / 2;
    constexpr int K_BYTES = KT * 256;
    constexpr int V_BYTES = KT * DCS * 16;
    constexpr int K_SCALE_COUNT = KT * 8;
    constexpr int V_SCALE_COUNT = KT * (DCS / 2);

    if (n_heads != 24 || n_kv_heads != 4 || head_dim != 256)
        return;
    const int kv_h = (int)blockIdx.y;
    if (kv_h >= 4)
        return;

    const int row_base = ROWS_PER_WG * (int)blockIdx.x;
    const int total_rows = 6 * batch_size;
    if (row_base >= total_rows)
        return;

    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int half = lane >> 4;
    const int row = row_base + 16 * wave + ml;
    const int query = row / 6;
    const int head = 6 * kv_h + row % 6;
    const bool qok = row < total_rows;

    // One global pass over Q. Each lane retains one 8-value half-fragment;
    // its xor16 partner retains the other half of the same query-head row.
    fa2w_half8_t Qhalf[16];
#pragma unroll
    for (int dc = 0; dc < 16; ++dc) {
        fa2w_float8_t x = (fa2w_float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (qok) {
            const float* qp = q + (unsigned long long)query * 6144ULL
                + (unsigned long long)head * 256ULL
                + (unsigned)(dc * 16 + half * 8);
            x = *(const fa2w_float8_t*)qp;
        }
        fa2w_half8_t qh;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            qh[i] = (_Float16)x[i];
        Qhalf[dc] = qh;
    }

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* K8 = smem;
    unsigned char* V8 = smem + K_BYTES;
    uint16_t* SK = (uint16_t*)(smem + K_BYTES + V_BYTES);
    uint16_t* SV = SK + K_SCALE_COUNT;

    // Bounds mailbox aliases SK and is consumed before the first cooperative
    // fill. The second barrier is required: without it a fast staging lane can
    // overwrite bounds[0] while another wave is still loading seq_len.
    int* bounds = (int*)SK;
    if (tid == 0) {
        bounds[0] = -1;
        bounds[1] = 0x7fffffff;
    }
    __syncthreads();
    const int q_first = row_base / 6;
    const int q_last = (min(row_base + ROWS_PER_WG, total_rows) - 1) / 6;
    if (tid <= q_last - q_first) {
        const int p = positions[q_first + tid];
        atomicMax(bounds + 0, p);
        atomicMin(bounds + 1, p);
    }
    __syncthreads();
    const int seq_len = bounds[0] + 1;
    const int pos = qok ? positions[query] : -1;
    __syncthreads();

    // D128 split: each pass repeats the K scan but keeps only 64 f32 output
    // registers per lane. Q remains resident across both passes.
    for (int ds = 0; ds < 16; ds += DCS) {
        fa2w_float8_t Ofr[DCS];
#pragma unroll
        for (int dc = 0; dc < DCS; ++dc)
            Ofr[dc] = (fa2w_float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float m_old = -INFINITY;
        float l_val = 0.0f;

        for (int ktile = 0; ktile < seq_len; ktile += KT) {
            // K codes/scales: all eight Q8_0 blocks are needed for QK.
            for (int idx = tid; idx < K_SCALE_COUNT; idx += THREADS) {
                const int k = idx >> 3;
                const int b = idx & 7;
                const int gk = ktile + k;
                uint16_t sk = 0;
                fa2w_bytes16_t lo, hi;
                lo.v = (fa2w_u32x4_t){0u, 0u, 0u, 0u};
                hi.v = (fa2w_u32x4_t){0u, 0u, 0u, 0u};
                if (gk < seq_len) {
                    const unsigned long long ro = (unsigned long long)gk * 1088ULL
                        + (unsigned long long)(kv_h * 8 + b) * 34ULL;
                    const unsigned char* kr = k_cache + ro;
                    sk = (uint16_t)kr[0] | ((uint16_t)kr[1] << 8);
                    __builtin_memcpy(&lo.v, kr + 2, 16);
                    __builtin_memcpy(&hi.v, kr + 18, 16);
                }
                *(fa2w_u32x4_t*)(K8 + k * 256 + b * 32) = lo.v;
                *(fa2w_u32x4_t*)(K8 + k * 256 + b * 32 + 16) = hi.v;
                SK[b * KT + k] = sk;
            }

            // V codes/scales: only this pass's contiguous D128 slice.
            for (int idx = tid; idx < V_SCALE_COUNT; idx += THREADS) {
                const int k = idx / (DCS / 2);
                const int lb = idx - k * (DCS / 2);
                const int b = (ds >> 1) + lb;
                const int gk = ktile + k;
                uint16_t sv = 0;
                fa2w_bytes16_t lo, hi;
                lo.v = (fa2w_u32x4_t){0u, 0u, 0u, 0u};
                hi.v = (fa2w_u32x4_t){0u, 0u, 0u, 0u};
                if (gk < seq_len) {
                    const unsigned long long ro = (unsigned long long)gk * 1088ULL
                        + (unsigned long long)(kv_h * 8 + b) * 34ULL;
                    const unsigned char* vr = v_cache + ro;
                    sv = (uint16_t)vr[0] | ((uint16_t)vr[1] << 8);
                    __builtin_memcpy(&lo.v, vr + 2, 16);
                    __builtin_memcpy(&hi.v, vr + 18, 16);
                }
#pragma unroll
                for (int d = 0; d < 16; ++d) {
                    V8[(lb * 32 + d) * KT + k] = lo.u[d];
                    V8[(lb * 32 + 16 + d) * KT + k] = hi.u[d];
                }
                SV[lb * KT + k] = sv;
            }
            __syncthreads();

#pragma unroll 1
            for (int sub = 0; sub < KT / 16; ++sub) {
                // QK: code*scale is converted directly into the f16 A
                // fragment. The resident Q halves reconstruct an identical
                // full B fragment in both members of each lane pair.
                fa2w_float8_t score =
                    (fa2w_float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                const int key = sub * 16 + ml;
#pragma unroll
                for (int dc = 0; dc < 16; ++dc) {
                    const float sk = (float)fa2w_scale(SK[(dc >> 1) * KT + key]);
                    fa2w_bytes16_t kc;
                    kc.v = *(const fa2w_u32x4_t*)(K8 + key * 256 + dc * 16);
                    fa2w_half16_t kf;
#pragma unroll
                    for (int i = 0; i < 16; ++i)
                        kf[i] = (_Float16)(sk * (float)kc.s[i]);

                    const fa2w_half8_t qh = Qhalf[dc];
                    fa2w_half16_t qf;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const _Float16 peer = (_Float16)__shfl_xor((float)qh[i], 16, 32);
                        qf[i] = half == 0 ? qh[i] : peer;
                        qf[i + 8] = half == 0 ? peer : qh[i];
                    }
                    score = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kf, qf, score);
                    asm volatile("" ::: "memory");
                }

                const int key0 = ktile + sub * 16;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int gk = key0 + 2 * j + half;
                    score[j] = qok && gk < seq_len && gk <= pos
                        ? score[j] * scale_attn
                        : -INFINITY;
                }
                float mx = -INFINITY;
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    mx = fmaxf(mx, score[j]);
                mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
                const float m_new = fmaxf(m_old, mx);
                const float alpha = m_old == -INFINITY ? 0.0f : __expf(m_old - m_new);
                m_old = m_new;

                float e[8];
                float sum = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float x = m_new == -INFINITY ? 0.0f : __expf(score[j] - m_new);
                    e[j] = x;
                    sum += x;
                }
                sum += __shfl_xor(sum, 16, 32);
                l_val = l_val * alpha + sum;

                fa2w_half16_t prow;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float peer = __shfl_xor(e[j], 16, 32);
                    prow[2 * j] = (_Float16)(half == 0 ? e[j] : peer);
                    prow[2 * j + 1] = (_Float16)(half == 0 ? peer : e[j]);
                }
#pragma unroll
                for (int dc = 0; dc < DCS; ++dc)
                    Ofr[dc] *= alpha;

                // PV: V remains bytes in LDS; each key's own Q8 scale is
                // applied while the f16 A fragment is assembled.
#pragma unroll
                for (int dc = 0; dc < DCS; ++dc) {
                    const int d = dc * 16 + ml;
                    fa2w_half16_t vf;
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const int lk = sub * 16 + i;
                        const float sv = (float)fa2w_scale(SV[(d >> 5) * KT + lk]);
                        const signed char code = (signed char)V8[d * KT + lk];
                        vf[i] = (_Float16)(sv * (float)code);
                    }
                    Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vf, prow, Ofr[dc]);
                    asm volatile("" ::: "memory");
                }
            }
            __syncthreads();
        }

        if (qok) {
            const float inv = l_val > 0.0f ? 1.0f / l_val : 0.0f;
            const unsigned ob = (unsigned)query * 6144u + (unsigned)head * 256u;
#pragma unroll
            for (int dc = 0; dc < DCS; ++dc) {
                const unsigned d0 = ob + (unsigned)((ds + dc) * 16 + half);
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    out[d0 + (unsigned)(2 * j)] = Ofr[dc][j] * inv;
            }
        }
    }
}
