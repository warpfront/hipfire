// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// VerifyAttn (gfx1151): the byte-identical parallel twin of
// `attention_q8_0_flash_prefill_wmma` (legacy single-slot, module
// `attention_q8_0_flash_prefill_wmma_gfx11_hd256`), the kernel every gfx1151
// DFlash / MTP verify (and every prefill tail below 64 rows) runs today.
//
// The reference runs one wave per (q head, 16-row query tile) that walks the
// whole context in 16-key steps: S = Q.K^T (16 chained WMMAs over head_dim),
// an online softmax per row, O = alpha*O + P.V (one WMMA per 16-dim chunk).
// Only two things in that walk are serial: the row statistics (m, l) and the
// O accumulator chain. S depends on nothing but Q and one K tile, so it is
// computed context-parallel; the (m, l, O) walk then runs per 16-dim chunk.
//
//   attention_verify_wmma_qk_gfx1151   grid [n_kv * n_rg, splits], NF waves.
//     One workgroup per (kv head, row group, split); tiles y, y+splits, ...
//     The K tile is dequantised once into LDS for all fragments of the kv
//     head; each wave runs the reference's 16-WMMA chain for one fragment
//     and stores the reference's f16 S tile (scaled, causal-masked with f16
//     -inf) to scratch.
//   attention_verify_wmma_pv1s_d4_gfx1151 / _pv2_d2_gfx1151
//                                        grid [n_kv * n_rg, 16/DCW].
//     One workgroup per (kv head, row group, DCW 16-dim chunks); every tile
//     in order: the reference's per-row softmax step on the stored S (max,
//     alpha, P = f16(expf(s - m)), in-order sum, l = l*alpha + sum), then
//     O *= alpha and O = WMMA(P, V^T, O) with V^T dequantised once into LDS
//     for all fragments. The final O * (1/l) goes straight to `out`.
//
// Fragments: a fragment is 16 (head, row) pairs of one kv head's 6 heads x
// the row group's rows. `packed == 0` keeps the reference fragment exactly
// (fragment = one head, fragment row = query row, 6 fragments); `packed == 1`
// packs pair p -> (head p / rows, row p % rows), ceil(6*rows/16) fragments.
// Every WMMA output element depends only on its own A row, B column and C
// element, so both layouts reproduce the reference bits; the oracle proves it.
// Row group rg = reference query tile rg (rows 16rg .. 16rg+15), so every row
// walks exactly the reference's key range (the max position of its tile).
//
// Arithmetic reproduced from the reference source (same expressions, same
// hipcc -O3 flags): Q f16(q); K, V f16(f16scale * code); S f16(acc * scale)
// or f16 -inf where kt + c > position; softmax per row as the reference's
// Phase B; keys at or past the row group's live length read as +0 (buffer
// range) exactly like the reference's zero fill.
#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
#error "attention_verify_wmma_gfx1151 requires --offload-arch=gfx1151"
#endif

typedef _Float16 __attribute__((ext_vector_type(16))) vw_half16;
typedef _Float16 __attribute__((ext_vector_type(8))) vw_half8;
typedef float __attribute__((ext_vector_type(8))) vw_float8;
typedef float __attribute__((ext_vector_type(4))) vw_float4;

#define VW_HD 256
#define VW_GROUP 6
#define VW_NT 16                       // keys per tile (reference N_TILE)
#define VW_MT 16                       // rows per fragment (reference M_TILE)
#define VW_DC (VW_HD / 16)             // 16-dim chunks
#define VW_NFMAX 6                     // fragments per (kv head, row group)
#define VW_THREADS (VW_NFMAX * 32)
#define VW_BLK 34
#define VW_SEG (VW_HD / 32 * VW_BLK)   // bytes of one (key, kv head): 272
#define VW_KS (VW_HD + 8)              // LDS K row stride in halfs (bank spread)
#define VW_VS 24                       // LDS V^T row stride in halfs (bank spread)
#define VW_KUNITS (VW_NT * VW_HD / 8)  // 8-dim K chunks per tile: 512
#define VW_QK_MIN_WAVES 4              // S launch block is >= 4 waves
#define VW_KU_MAX (VW_KUNITS / (VW_QK_MIN_WAVES * 32))  // K chunks per thread
#define VW_RSRC 0x31004000u

// LDS-only workgroup barrier: keeps in-flight global prefetches in flight.
// Each wave waits for its own LDS stores (lgkmcnt(0), vmcnt/expcnt left
// open: gfx11 s_waitcnt 0xFC07) before it signals the barrier. The LDS-only
// release fence alone emits no wait (the memory legalizer assumes one total
// LDS order), but in WGP mode a workgroup's waves span both CUs and their
// LDS request queues are not ordered against each other, so another wave's
// read after the barrier could overtake a store still queued. That lost the
// K tile store of the S launch's loop (1 of 1,200 launches in a two-process
// stress; never serial or on one CU).
#define VW_WAIT_LGKM0 0xFC07
#define VW_LDS_BARRIER()                                                   \
    do {                                                                   \
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");   \
        __builtin_amdgcn_s_waitcnt(VW_WAIT_LGKM0);                         \
        __builtin_amdgcn_s_barrier();                                      \
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");   \
    } while (0)

struct VwGroup {
    int q_base;   // first global row of the row group
    int rows;     // rows in the row group (1..16)
    int pack;     // pair stride: 16 (reference layout) or rows (packed)
    int nf;       // fragments
    int seq_len;  // reference seq_len: max position of the row group + 1
};

__device__ __forceinline__ VwGroup vw_group(const int* __restrict__ positions, int n_rows, int rg,
                                            int packed) {
    VwGroup g;
    g.q_base = rg * VW_MT;
    g.rows = min(VW_MT, n_rows - g.q_base);
    g.pack = packed ? g.rows : VW_MT;
    g.nf = (VW_GROUP * g.pack + VW_MT - 1) / VW_MT;
    int s = 0;
    for (int r = 0; r < g.rows; ++r) s = max(s, positions[g.q_base + r] + 1);
    g.seq_len = s;
    return g;
}

// Pair p of the row group -> (head within the kv group, row within the group).
__device__ __forceinline__ bool vw_pair(const VwGroup& g, int p, int& hl, int& r) {
    hl = p / g.pack;
    r = p - hl * g.pack;
    return hl < VW_GROUP && r < g.rows;
}

__device__ __forceinline__ float vw_code(unsigned int w, int i) {
    return (float)((signed char)(w >> (8 * i)));
}

// Eight f16(scale * code) values, the reference's dequant expression.
__device__ __forceinline__ vw_half8 vw_dequant8(unsigned int lo, unsigned int hi, unsigned short sc) {
    const float s = (float)__builtin_bit_cast(_Float16, sc);
    vw_half8 h;
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = (_Float16)(s * vw_code(lo, i));
#pragma unroll
    for (int i = 0; i < 4; ++i) h[4 + i] = (_Float16)(s * vw_code(hi, i));
    return h;
}

// Scratch S tile of (row group, kv head, fragment): [t_stride][16 rows][16 keys] f16.
__device__ __forceinline__ size_t vw_s_base(int rg, int kv, int n_kv_heads, int f, int t_stride) {
    return (((size_t)(rg * n_kv_heads + kv) * VW_NFMAX + f) * (size_t)t_stride) * (VW_MT * VW_NT);
}

// Grid [n_kv * n_rg, splits], block 32 * max(NF(row group 0), 4). Workgroup
// x = kv * n_rg + rg; workgroup y scores tiles y, y + splits, ... below
// ceil(seq_len / 16) (clamped to t_stride, the scratch stride).
extern "C" __launch_bounds__(VW_THREADS)
__global__ void attention_verify_wmma_qk_gfx1151(
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,
    const int* __restrict__ positions, _Float16* __restrict__ s_scr, int n_rows, int n_kv_heads,
    int t_stride, int packed, float scale_attn) {
    __shared__ __attribute__((aligned(16))) _Float16 k_lds[2 * VW_NT * VW_KS];

    const int n_rg = (n_rows + VW_MT - 1) / VW_MT;
    const int kv = (int)blockIdx.x / n_rg;
    const int rg = (int)blockIdx.x - kv * n_rg;
    const VwGroup g = vw_group(positions, n_rows, rg, packed);
    const int n_tiles = min((g.seq_len + VW_NT - 1) / VW_NT, t_stride);
    const int split = (int)blockIdx.y;
    const int n_split = (int)gridDim.y;
    if (split >= n_tiles) return;

    const int tid = (int)threadIdx.x;
    const int nthreads = (int)blockDim.x;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int lane = tid & 31;
    const int half = lane & 15;
    const int half_id = lane >> 4;
    const bool active = wave < g.nf;
    const int n_heads = n_kv_heads * VW_GROUP;
    const int q_stride = n_heads * VW_HD;
    const unsigned row_stride = (unsigned)n_kv_heads * VW_SEG;
    const unsigned head_off = (unsigned)kv * VW_SEG;
    // Keys at or past seq_len read as zero, as the reference's k_ok fill.
    const auto k_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)k_cache, 0, (int)((unsigned)g.seq_len * row_stride), VW_RSRC);

    // A fragments: pair (wave*16 + half), all 16 dim chunks, f16(q) or +0.
    // Buffer loads with an out-of-range offset for invalid pairs return +0,
    // so no load is conditional (a conditional load would be sunk into a
    // branch and wait on its own) and all 64 issue back to back.
    vw_half16 qa[VW_DC];
    int cpos1[8];
    {
        int hl, r;
        const bool ok = active && vw_pair(g, wave * VW_MT + half, hl, r);
        const auto q_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)q, 0, (int)((unsigned)n_rows * (unsigned)q_stride * 4u), VW_RSRC);
        const unsigned q_off = ok ? ((unsigned)(g.q_base + r) * (unsigned)q_stride + (unsigned)(kv * VW_GROUP + hl) * VW_HD) * 4u
                                  : 0x7ff00000u;
#pragma unroll
        for (int dc = 0; dc < VW_DC; ++dc) {
#pragma unroll
            for (int j = 0; j < 16; j += 4) {
                // Whole-vector bit_cast: this clang lowers
                // `__builtin_bit_cast(float, w[i])` as a load of element 0.
                const vw_float4 f = __builtin_bit_cast(
                    vw_float4, __builtin_amdgcn_raw_buffer_load_b128(q_rsrc, q_off + (unsigned)(dc * 16 + j) * 4u, 0, 0));
                qa[dc][j + 0] = (_Float16)f[0];
                qa[dc][j + 1] = (_Float16)f[1];
                qa[dc][j + 2] = (_Float16)f[2];
                qa[dc][j + 3] = (_Float16)f[3];
            }
        }
        // C rows 2j + half_id of this fragment: causal bound + 1, or 0.
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int hj, rj;
            const bool okj = active && vw_pair(g, wave * VW_MT + 2 * j + half_id, hj, rj);
            const int pj = positions[g.q_base + (okj ? rj : 0)];
            cpos1[j] = (pj + 1) & -(int)okj;
        }
    }
    _Float16* s_out = s_scr + vw_s_base(rg, kv, n_kv_heads, active ? wave : 0, t_stride);

    // K staging: unit u = key (u >> 5), 8-dim chunk (u & 31). Loads are
    // unconditional (units past the tile repeat the last one, never staged).
    unsigned int k_lo[VW_KU_MAX], k_hi[VW_KU_MAX];
    unsigned short k_sc[VW_KU_MAX];
    auto k_load = [&](int t) {
#pragma unroll
        for (int i = 0; i < VW_KU_MAX; ++i) {
            const int u = min(tid + i * nthreads, VW_KUNITS - 1);
            const int kr = u >> 5;
            const int c = u & 31;
            const unsigned blk = (unsigned)(t * VW_NT + kr) * row_stride + head_off + (unsigned)(c >> 2) * VW_BLK;
            const auto w = __builtin_amdgcn_raw_buffer_load_b64(k_rsrc, blk + 2u + 8u * (unsigned)(c & 3), 0, 0);
            k_lo[i] = w[0];
            k_hi[i] = w[1];
            k_sc[i] = __builtin_amdgcn_raw_buffer_load_b16(k_rsrc, blk, 0, 0);
        }
    };
    auto k_stage = [&](int buf) {
#pragma unroll
        for (int i = 0; i < VW_KU_MAX; ++i) {
            const int u = tid + i * nthreads;
            if (u < VW_KUNITS) {
                const int kr = u >> 5;
                const int c = u & 31;
                *(vw_half8*)&k_lds[buf + kr * VW_KS + c * 8] = vw_dequant8(k_lo[i], k_hi[i], k_sc[i]);
            }
        }
    };

    // LDS buffer offsets stay 32-bit ints (a pointer selected by `buf` would
    // go through 64-bit flat arithmetic and read in-flight load registers).
    int t = split;
    k_load(t);
    k_stage(0);
    int boff = 0;
    for (; t < n_tiles; t += n_split) {
        const int tn = t + n_split;
        k_load(min(tn, n_tiles - 1));  // unconditional: keeps the wait partial
        VW_LDS_BARRIER();
        if (active) {
            vw_float8 s = (vw_float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const int kb = boff + half * VW_KS;
#pragma unroll
            for (int dc = 0; dc < VW_DC; ++dc) {
                const vw_half8 b0 = *(const vw_half8*)&k_lds[kb + dc * 16];
                const vw_half8 b1 = *(const vw_half8*)&k_lds[kb + dc * 16 + 8];
                const vw_half16 b = __builtin_shufflevector(b0, b1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                                            12, 13, 14, 15);
                s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[dc], b, s);
            }
            const int kt = t * VW_NT;
            _Float16* so = s_out + (size_t)t * (VW_MT * VW_NT);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float sv = s[j] * scale_attn;
                const bool valid = (kt + half) < cpos1[j];
                so[(2 * j + half_id) * VW_NT + half] = valid ? (_Float16)sv : (_Float16)(-INFINITY);
            }
        }
        boff ^= VW_NT * VW_KS;
        if (tn < n_tiles) k_stage(boff);
    }
}

// Value of `v` in lane (lane ^ 16), the other 16-lane row.
__device__ __forceinline__ unsigned vw_x16u(unsigned v) {
    return __builtin_amdgcn_permlanex16(0u, v, 0x76543210u, 0xfedcba98u, false, false);
}
__device__ __forceinline__ float vw_x16(float v) {
    return __builtin_bit_cast(float, vw_x16u(__builtin_bit_cast(unsigned, v)));
}

// S of one lane's pair for one tile: all 16 keys, or (SPLIT) its row's 8.
template <bool SPLIT>
using vw_svec = _Float16 __attribute__((ext_vector_type(SPLIT ? 8 : 16)));

// One reference step (Phase B + O rescale + P.V) for this lane's pair.
// SPLIT: the two 16-lane rows of a wave split each pair's 16 keys (row 0
// keys 0..7, row 1 keys 8..15): the max and the in-order sum cross rows with
// v_permlanex16 (row 1 continues row 0's partial sum from +0 in key order),
// and P is exchanged the same way. Otherwise both rows run the reference's
// whole per-row step.
template <int DCW, bool SPLIT>
__device__ __forceinline__ void vw_step(const vw_svec<SPLIT>& sv, bool my_ok, int half, int half_id, float& m,
                                        float& l, vw_float8 (&o)[DCW], const _Float16* __restrict__ v_lds, int vo) {
    vw_half16 p;
    float a;
    if constexpr (SPLIT) {
        typedef unsigned int u4 __attribute__((ext_vector_type(4)));
        vw_half8 p8;
        if (my_ok) {
            float tm = -INFINITY;
#pragma unroll
            for (int c = 0; c < 8; ++c) tm = fmaxf(tm, (float)sv[c]);
            tm = fmaxf(tm, vw_x16(tm));
            const float m_old = m;
            const float m_new = fmaxf(m_old, tm);
            const float alpha = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
            float e[8];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                e[c] = (m_new == -INFINITY) ? 0.0f : __expf((float)sv[c] - m_new);
                p8[c] = (_Float16)e[c];
            }
            float part = 0.0f;
#pragma unroll
            for (int c = 0; c < 8; ++c) part += e[c];
            float acc = vw_x16(part);
#pragma unroll
            for (int c = 0; c < 8; ++c) acc += e[c];
            const float tot = vw_x16(acc);
            const float sum = half_id ? acc : tot;
            m = m_new;
            l = l * alpha + sum;
            a = alpha;
        } else {
#pragma unroll
            for (int c = 0; c < 8; ++c) p8[c] = (_Float16)0.0f;
            a = 1.0f;
        }
        const u4 own = __builtin_bit_cast(u4, p8);
        u4 oth;
#pragma unroll
        for (int k = 0; k < 4; ++k) oth[k] = vw_x16u(own[k]);
        const u4 lo = half_id ? oth : own;
        const u4 hi = half_id ? own : oth;
        p = __builtin_bit_cast(vw_half16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
    } else {
        if (my_ok) {
            float tm = -INFINITY;
#pragma unroll
            for (int c = 0; c < VW_NT; ++c) tm = fmaxf(tm, (float)sv[c]);
            const float m_old = m;
            const float m_new = fmaxf(m_old, tm);
            const float alpha = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < VW_NT; ++c) {
                const float e = (m_new == -INFINITY) ? 0.0f : __expf((float)sv[c] - m_new);
                p[c] = (_Float16)e;
                sum += e;
            }
            m = m_new;
            l = l * alpha + sum;
            a = alpha;
        } else {
#pragma unroll
            for (int c = 0; c < VW_NT; ++c) p[c] = (_Float16)0.0f;
            a = 1.0f;
        }
    }
    // alpha of C rows 2j + half_id (held by lane 2j + half_id).
    float aj[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) aj[j] = __shfl(a, 2 * j + half_id);
#pragma unroll
    for (int dc = 0; dc < DCW; ++dc) {
#pragma unroll
        for (int j = 0; j < 8; ++j) o[dc][j] *= aj[j];
        const _Float16* vb = &v_lds[vo + (dc * 16 + half) * VW_VS];
        const vw_half8 b0 = *(const vw_half8*)vb;
        const vw_half8 b1 = *(const vw_half8*)(vb + 8);
        const vw_half16 b = __builtin_shufflevector(b0, b1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        o[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(p, b, o[dc]);
    }
}

// Grid [n_kv * n_rg, 16 / DCW], block 32 * max(NF(row group 0), DCW) (so
// each thread stages at most one 8-dim V chunk). Workgroup x = kv * n_rg +
// rg; workgroup y owns dim chunks [DCW*y, DCW*y + DCW). The walk is serial
// over tiles, so every thread keeps PD tiles of S and V loads in flight
// (a register ring, loop unrolled by PD); loads are unconditional with
// clamped addresses so the waits stay partial.
template <int DCW, bool SPLIT, int PD>
__device__ __forceinline__ void vw_pv(const unsigned char* __restrict__ v_cache,
                                      const int* __restrict__ positions,
                                      const _Float16* __restrict__ s_scr, float* __restrict__ out,
                                      int n_rows, int n_kv_heads, int t_stride, int packed) {
    constexpr int DW = DCW * 16;           // dims owned
    constexpr int VUNITS = VW_NT * DW / 8; // 8-dim V chunks per tile (32 * DCW)
    __shared__ __attribute__((aligned(16))) _Float16 v_lds[2 * DW * VW_VS];

    const int n_rg = (n_rows + VW_MT - 1) / VW_MT;
    const int kv = (int)blockIdx.x / n_rg;
    const int rg = (int)blockIdx.x - kv * n_rg;
    const int d0 = (int)blockIdx.y * DW;
    const VwGroup g = vw_group(positions, n_rows, rg, packed);
    const int n_tiles = min((g.seq_len + VW_NT - 1) / VW_NT, t_stride);

    const int tid = (int)threadIdx.x;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int lane = tid & 31;
    const int half = lane & 15;
    const int half_id = lane >> 4;
    const bool active = wave < g.nf;
    const int n_heads = n_kv_heads * VW_GROUP;
    const int q_stride = n_heads * VW_HD;
    const unsigned row_stride = (unsigned)n_kv_heads * VW_SEG;
    const auto v_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)v_cache, 0, (int)((unsigned)g.seq_len * row_stride), VW_RSRC);

    int my_hl, my_r;
    const bool my_ok = active && vw_pair(g, wave * VW_MT + half, my_hl, my_r);
    const _Float16* s_in = s_scr + vw_s_base(rg, kv, n_kv_heads, active ? wave : 0, t_stride) + half * VW_NT
                         + (SPLIT ? 8 * half_id : 0);

    // This thread's V chunk: key v_kr of the tile, dims v_d .. v_d+7 (threads
    // past VUNITS load a duplicate and never stage it).
    const int v_u = tid % VUNITS;
    const bool v_mine = tid < VUNITS;
    const int v_kr = v_u / (DW / 8);
    const int v_c = v_u - v_kr * (DW / 8);
    const int v_d = d0 + v_c * 8;
    const unsigned v_blk = (unsigned)v_kr * row_stride + (unsigned)kv * VW_SEG + (unsigned)(v_d >> 5) * VW_BLK;

    struct Ring {
        unsigned int lo, hi;
        unsigned short sc;
        vw_svec<SPLIT> s;
    };
    Ring rf[PD];
    auto issue = [&](Ring& r, int t) {
        // V past seq_len reads +0 (buffer range); S clamps to the last live tile.
        const unsigned blk = (unsigned)(t * VW_NT) * row_stride + v_blk;
        const auto w = __builtin_amdgcn_raw_buffer_load_b64(v_rsrc, blk + 2u + (unsigned)(v_d & 31), 0, 0);
        r.lo = w[0];
        r.hi = w[1];
        r.sc = __builtin_amdgcn_raw_buffer_load_b16(v_rsrc, blk, 0, 0);
        r.s = *(const vw_svec<SPLIT>*)(s_in + (size_t)min(t, n_tiles - 1) * (VW_MT * VW_NT));
    };

    vw_float8 o[DCW];
#pragma unroll
    for (int dc = 0; dc < DCW; ++dc) o[dc] = (vw_float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float m = -INFINITY;
    float l = 0.0f;

    // Stage tile t's V chunk from its ring slot into LDS.
    auto stage = [&](const Ring& r, int t) -> int {
        const int vo = (t & 1) * (DW * VW_VS);  // 32-bit LDS offset, never a selected pointer
        if (v_mine) {
            const vw_half8 h = vw_dequant8(r.lo, r.hi, r.sc);
#pragma unroll
            for (int e = 0; e < 8; ++e) v_lds[vo + (v_c * 8 + e) * VW_VS + v_kr] = h[e];
        }
        return vo;
    };

#pragma unroll
    for (int k = 0; k < PD; ++k) issue(rf[k], k);
    // Main loop: every ring slot live; refill it PD tiles ahead (clamped
    // loads past the end are harmless) once its old contents are dead, so
    // the loads land in the slot's own registers (no latch copies that would
    // wait on them). Uniform control flow keeps the vmcnt waits partial.
    int t0 = 0;
    for (; t0 + PD <= n_tiles; t0 += PD) {
#pragma unroll
        for (int k = 0; k < PD; ++k) {
            const int vo = stage(rf[k], t0 + k);
            VW_LDS_BARRIER();
            if (active) vw_step<DCW, SPLIT>(rf[k].s, my_ok, half, half_id, m, l, o, v_lds, vo);
            issue(rf[k], t0 + k + PD);
        }
    }
    // Tail: the last n_tiles - t0 < PD tiles are already in the ring.
#pragma unroll
    for (int k = 0; k < PD - 1; ++k) {
        if (t0 + k < n_tiles) {
            const int vo = stage(rf[k], t0 + k);
            VW_LDS_BARRIER();
            if (active) vw_step<DCW, SPLIT>(rf[k].s, my_ok, half, half_id, m, l, o, v_lds, vo);
        }
    }

    if (!active) return;
    // O * (1/l), the reference's epilogue, for the valid pairs of the C rows.
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float denom = __shfl(l, 2 * j + half_id);
        int hj, rj;
        if (!vw_pair(g, wave * VW_MT + 2 * j + half_id, hj, rj)) continue;
        const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
        float* orow = out + (size_t)(g.q_base + rj) * q_stride + (size_t)(kv * VW_GROUP + hj) * VW_HD + d0 + half;
#pragma unroll
        for (int dc = 0; dc < DCW; ++dc) orow[dc * 16] = o[dc][j] * inv;
    }
}

#define VW_PV_ENTRY(DCW, SPLIT, SFX, PD)                                                              \
    extern "C" __launch_bounds__(VW_THREADS) __global__ void attention_verify_wmma_pv##DCW##SFX##_d##PD##_gfx1151( \
        const unsigned char* __restrict__ v_cache, const int* __restrict__ positions,                 \
        const _Float16* __restrict__ s_scr, float* __restrict__ out, int n_rows, int n_kv_heads,      \
        int t_stride, int packed) {                                                                   \
        vw_pv<DCW, SPLIT, PD>(v_cache, positions, s_scr, out, n_rows, n_kv_heads, t_stride, packed);  \
    }
// The two P.V walkers kept from the Halo geometry sweep (24 variants of
// DCW 1/2/4 x SPLIT x PD 1/2/4/8): pv1s_d4 wins at B 1..4 (1-3 fragments),
// pv2_d2 at B 16 (6 fragments). The host picks by fragment count.
VW_PV_ENTRY(1, true, s, 4)
VW_PV_ENTRY(2, false, , 2)
