// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Non-causal DFlash draft attention (gfx1151), WMMA wave32, head_dim 128,
// keys split across the waves of a workgroup.
//
// Why: the DFlash draft attends B = 16 query rows (one 16-row tile) over the
// whole context. `attention_dflash_wmma_n64_f32` runs one 2-wave workgroup
// per (head, 32-row tile): 32 workgroups for a 32-head draft, the second
// wave idle at B = 16, each walking every key serially (1.4 ms at L = 2K on
// gfx1151, 7.5 ms at 8K). Here each of the 8 waves of a (head, 16-row tile)
// workgroup walks every 8th 16-key tile with its own online softmax
// (m, l, O in registers); the waves merge once at the end.
//
// Per wave and key tile: K (16 b128 loads per lane, coalesced) -> f16 rows in
// the wave's LDS region; S = Q.K^T (8 WMMAs, Q f16 in shared LDS); S * scale
// masked to -inf past L through the wave's S tile so each lane owns one query
// row (the WMMA A layout of P); the row's online-softmax step; V (16 b128
// loads, issued before S) -> f16 V^T in the same region; O = alpha*O + P.V
// (8 WMMAs). Every global load of a tile is issued before the first one is
// consumed. Merge: M = max_w m_w, O = sum_w O_w * exp(m_w - M), l likewise,
// accumulated in wave order (deterministic), out = O / l.
//
// Layout (row-major, flat): q/out [B][n_heads][128], k/v [L][n_kv_heads][128].
// Grid [n_heads, ceil(B / 16)], block 256 (8 waves).

#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
#error "attention_dflash_wmma_kvsplit_gfx1151 requires --offload-arch=gfx1151"
#endif

typedef _Float16 __attribute__((ext_vector_type(16))) ks_half16;
typedef _Float16 __attribute__((ext_vector_type(8))) ks_half8;
typedef _Float16 __attribute__((ext_vector_type(4))) ks_half4;
typedef float __attribute__((ext_vector_type(8))) ks_float8;
typedef float __attribute__((ext_vector_type(4))) ks_float4;

#define KS_HD 128
#define KS_DC (KS_HD / 16)       // 16-dim chunks
#define KS_WAVES 8
#define KS_RS (KS_HD + 8)        // LDS Q/K row stride in halfs (bank spread)
#define KS_VS 24                 // LDS V^T row stride in halfs (16 keys + bank spread)
#define KS_SS 17                 // LDS S row stride in floats (bank spread)
#define KS_REGION (KS_HD * KS_VS) // per-wave K / V^T region in halfs (>= 16 * KS_RS)
#define KS_RSRC 0x31004000u      // gfx11 buffer descriptor word 3 (raw dwords)

// Order one wave's LDS stores before its own lanes' later LDS loads (and the
// reverse). A wave's LDS instructions execute in order; the fences keep the
// compiler from moving loads and stores across each other.
#define KS_WAVE_LDS_SYNC()                                                 \
    do {                                                                   \
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");   \
        __builtin_amdgcn_wave_barrier();                                   \
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");   \
    } while (0)

__device__ __forceinline__ ks_half16 ks_row16(const _Float16* p) {
    const ks_half8 lo = *(const ks_half8*)p;
    const ks_half8 hi = *(const ks_half8*)(p + 8);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}

extern "C" __launch_bounds__(KS_WAVES * 32)
__global__ void attention_dflash_wmma_kvsplit_gfx1151(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B, int L, int n_heads, int n_kv_heads,
    float scale
) {
    __shared__ __attribute__((aligned(16))) _Float16 q_lds[16 * KS_RS];
    // Per wave: K rows [key][dim], then V^T [dim][key]; after the walk, the merged O (f32).
    __shared__ __attribute__((aligned(16))) _Float16 kv_lds[KS_WAVES][KS_REGION];
    __shared__ float s_lds[KS_WAVES][16 * KS_SS];
    __shared__ float m_lds[KS_WAVES][16];
    __shared__ float l_lds[KS_WAVES][16];
    float* o_lds = (float*)&kv_lds[0][0];

    const int head = blockIdx.x;
    const int q0 = blockIdx.y * 16;
    const int tid = threadIdx.x;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int lane = tid & 31;
    const int half = lane & 15;
    const int half_id = lane >> 4;

    const int kv_head = head / (n_heads / n_kv_heads);
    const int q_stride = n_heads * KS_HD;
    const int kv_stride = n_kv_heads * KS_HD;
    // Rows at or past B / L read as +0 (buffer range): no load is conditional
    // (a conditional load is sunk into a branch and waits on its own).
    const int kv_bytes = (int)((unsigned)L * (unsigned)kv_stride * 4u);
    const auto k_rsrc = __builtin_amdgcn_make_buffer_rsrc((void*)(k + kv_head * KS_HD), 0, kv_bytes, KS_RSRC);
    const auto v_rsrc = __builtin_amdgcn_make_buffer_rsrc((void*)(v + kv_head * KS_HD), 0, kv_bytes, KS_RSRC);
    const auto q_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)(q + (size_t)q0 * q_stride + head * KS_HD), 0,
        (int)((unsigned)max(min(B - q0, 16) - 1, 0) * (unsigned)q_stride * 4u + KS_HD * 4u), KS_RSRC);

    // Q tile -> f16 rows in LDS: 16 rows x 32 float4, 2 per thread.
#pragma unroll
    for (int u = tid; u < 16 * KS_HD / 4; u += KS_WAVES * 32) {
        const int r = u >> 5;
        const int c = (u & 31) * 4;
        // Whole-vector bit_cast: this clang lowers `__builtin_bit_cast(float, w[i])`
        // as a load of element 0.
        const ks_float4 f = __builtin_bit_cast(
            ks_float4, __builtin_amdgcn_raw_buffer_load_b128(q_rsrc, ((unsigned)r * (unsigned)q_stride + (unsigned)c) * 4u, 0, 0));
        *(ks_half4*)&q_lds[r * KS_RS + c] = __builtin_convertvector(f, ks_half4);
    }
    __syncthreads();

    ks_float8 o[KS_DC];
#pragma unroll
    for (int dc = 0; dc < KS_DC; ++dc) o[dc] = (ks_float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float m = -INFINITY;  // running max of row `half` (both 16-lane rows hold it)
    float l = 0.0f;
    _Float16* reg = kv_lds[wave];
    float* st = s_lds[wave];
    const unsigned lane_off = 16u * (unsigned)lane;  // bytes: dims 4*lane .. 4*lane+3

    const int n_tiles = (L + 15) / 16;
    for (int t = wave; t < n_tiles; t += KS_WAVES) {
        const unsigned tile_off = (unsigned)(t * 16) * (unsigned)kv_stride * 4u + lane_off;
        const unsigned row_bytes = (unsigned)kv_stride * 4u;
        ks_float4 kr[16], vr[16];
#pragma unroll
        for (int i = 0; i < 16; ++i)
            kr[i] = __builtin_bit_cast(ks_float4, __builtin_amdgcn_raw_buffer_load_b128(k_rsrc, tile_off + i * row_bytes, 0, 0));
#pragma unroll
        for (int i = 0; i < 16; ++i)
            vr[i] = __builtin_bit_cast(ks_float4, __builtin_amdgcn_raw_buffer_load_b128(v_rsrc, tile_off + i * row_bytes, 0, 0));
        // Keep all 32 loads ahead of their first use (the scheduler otherwise
        // sinks each load next to its store and waits on it alone).
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int i = 0; i < 16; ++i) *(ks_half4*)&reg[i * KS_RS + 4 * lane] = __builtin_convertvector(kr[i], ks_half4);
        KS_WAVE_LDS_SYNC();

        // S[16 rows][16 keys]: A row `half` = query, B column `half` = key.
        ks_float8 s = (ks_float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int dc = 0; dc < KS_DC; ++dc)
            s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ks_row16(&q_lds[half * KS_RS + dc * 16]),
                                                           ks_row16(&reg[half * KS_RS + dc * 16]), s);
        // C element j of this lane = S[row 2j + half_id][key half].
        const bool col_ok = t * 16 + half < L;
#pragma unroll
        for (int j = 0; j < 8; ++j) st[(2 * j + half_id) * KS_SS + half] = col_ok ? s[j] * scale : -INFINITY;
        KS_WAVE_LDS_SYNC();

        // V^T over the K rows (every lane's K reads are behind the sync above).
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            ks_half8 lo, hi;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                lo[i] = (_Float16)vr[i][e];
                hi[i] = (_Float16)vr[8 + i][e];
            }
            *(ks_half8*)&reg[(4 * lane + e) * KS_VS] = lo;
            *(ks_half8*)&reg[(4 * lane + e) * KS_VS + 8] = hi;
        }

        // Online softmax of row `half` (key t*16 is always live, so m_new is finite).
        float sv[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) sv[c] = st[half * KS_SS + c];
        KS_WAVE_LDS_SYNC();
        float tm = sv[0];
#pragma unroll
        for (int c = 1; c < 16; ++c) tm = fmaxf(tm, sv[c]);
        const float m_new = fmaxf(m, tm);
        const float alpha = __expf(m - m_new);  // m = -inf on the first tile: 0
        ks_half16 p;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            const float e = __expf(sv[c] - m_new);
            p[c] = (_Float16)e;
            sum += e;
        }
        l = l * alpha + sum;
        m = m_new;

        // O rows 2j + half_id scale by their row's alpha (held by lane 2j + half_id);
        // B[key][dim] column `half` = V^T row dc*16 + half.
        float aj[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) aj[j] = __shfl(alpha, 2 * j + half_id);
#pragma unroll
        for (int dc = 0; dc < KS_DC; ++dc) {
#pragma unroll
            for (int j = 0; j < 8; ++j) o[dc][j] *= aj[j];
            o[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(p, ks_row16(&reg[(dc * 16 + half) * KS_VS]), o[dc]);
        }
        KS_WAVE_LDS_SYNC();  // the next tile's K stores come after these loads
    }

    // Merge the waves. A wave with no tile (n_tiles < KS_WAVES) has m = -inf, l = 0.
    if (lane < 16) {
        m_lds[wave][lane] = m;
        l_lds[wave][lane] = l;
    }
    __syncthreads();
    float wrow[8];  // this wave's weight exp(m_w - M) for C row 2j + half_id
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int r = 2 * j + half_id;
        float mm = m_lds[0][r];
#pragma unroll
        for (int w = 1; w < KS_WAVES; ++w) mm = fmaxf(mm, m_lds[w][r]);
        const float mw = m_lds[wave][r];
        wrow[j] = (mw == -INFINITY) ? 0.0f : __expf(mw - mm);
    }
    for (int w = 0; w < KS_WAVES; ++w) {
        if (wave == w) {
#pragma unroll
            for (int dc = 0; dc < KS_DC; ++dc) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float* dst = &o_lds[(2 * j + half_id) * KS_HD + dc * 16 + half];
                    const float x = o[dc][j] * wrow[j];
                    *dst = (w == 0) ? x : *dst + x;
                }
            }
        }
        __syncthreads();
    }
    // out = O / sum_w l_w * exp(m_w - M), 2048 values over 256 threads.
    for (int i = tid; i < 16 * KS_HD; i += KS_WAVES * 32) {
        const int r = i / KS_HD;
        if (q0 + r >= B) continue;
        float mm = m_lds[0][r];
#pragma unroll
        for (int w = 1; w < KS_WAVES; ++w) mm = fmaxf(mm, m_lds[w][r]);
        float den = 0.0f;
#pragma unroll
        for (int w = 0; w < KS_WAVES; ++w) {
            const float mw = m_lds[w][r];
            den += (mw == -INFINITY) ? 0.0f : l_lds[w][r] * __expf(mw - mm);
        }
        out[(size_t)(q0 + r) * q_stride + head * KS_HD + (i % KS_HD)] = o_lds[i] / den;
    }
}
