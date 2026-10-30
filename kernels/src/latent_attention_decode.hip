// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

// Split-KV decode attention (one query row), FP32 throughout, deterministic,
// no atomics, graph-capture safe. Operand contract: latent_attention_tiled.rs.
//
//   key(j, h)   = k_a(j, h)[0..wa] ++ k_b(j, h)[0..wb]        (BF16, exact)
//   score(j)    = scale * dot(q[h][0..wa+wb], key(j, h))       (q is F32)
//   y[h]        = sum_j softmax_j(score) * v(j, h)[0..dv]
//
// Visible keys: j < len, len = min(*pos + pos_offset + 1, max_rows), the
// scalar read on the device. Strided views: element (j, h, d) of a source is
// base[j * tok + h * head + d] (strides in BF16 elements, base already
// advanced by the view offset).
//
// Kernel 1 (partial), grid (splits, n_heads), 256 threads. The split layout
// depends on len only: per = round_up(ceil(len / splits), 16), split s owns
// keys [s*per, min((s+1)*per, len)). A block whose range is empty exits before
// touching any memory; the merge visits only the non-empty splits. Each block
// walks its range in tiles of 64 keys:
//   * scores: a group of 16 lanes owns one key row (16-byte vector loads of
//     8 BF16, vectors strided by 16 lanes, k_a then k_b treated as one row of
//     (wa + wb) / 8 vectors), per-lane ascending FMA chain, then a fixed
//     xor-butterfly over the 16 lanes. 16 groups x 4 keys cover the tile.
//   * softmax: every 16-lane group derives the tile max / sum redundantly
//     with the same fixed butterfly, so m and l are uniform in the block.
//   * values: thread (vg, vl) owns the 8 columns [8*vl, 8*vl+8) for the keys
//     kk = vg, vg + VG, ... of the tile in ascending order (VG = 256 / (dv/8)
//     key groups); online-softmax rescale by the tile correction factor.
//   After the last tile the VG per-group accumulators are summed in ascending
//   group order through LDS and the record (m, l, acc[0..dv]) is written to
//   ws[h][split][dv + 2].
//
// Kernel 2 (merge), one block per head: fixed ascending split order,
//   M = max m_s, w_s = exp(m_s - M), L = sum l_s * w_s,
//   y = (sum acc_s * w_s) / L.
// len == 0 (or L == 0) writes zeros.
//
// Limits (checked by the launcher): wa, wb, dv multiples of 8; wa + wb <= 2048;
// dv <= 2048; splits <= 64.

#define LA_DEC_THREADS 256
#define LA_DEC_TILE 64
#define LA_DEC_GROUP 16
#define LA_DEC_KEYS_PER_GROUP (LA_DEC_TILE / (LA_DEC_THREADS / LA_DEC_GROUP))
#define LA_DEC_MAX_D 2048
#define LA_DEC_MAX_SPLITS 64
#define LA_DEC_SPLIT_ALIGN 16

static __device__ __forceinline__ uint16_t la_f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static __device__ __forceinline__ float la_bf16_to_f32(uint16_t h) {
    return __uint_as_float(((uint32_t)h) << 16);
}

static __device__ __forceinline__ float la_dec_lo(uint32_t w) { return __uint_as_float(w << 16); }
static __device__ __forceinline__ float la_dec_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// Number of visible keys; depends on the device scalar and launch constants.
static __device__ __forceinline__ int la_dec_visible(const int* pos_ptr, int pos_offset, int max_rows) {
    long long len = (long long)(*pos_ptr) + (long long)pos_offset + 1ll;
    if (len > (long long)max_rows) len = (long long)max_rows;
    if (len < 0) len = 0;
    return (int)len;
}

// Keys per split; a function of len and the split count only.
static __device__ __forceinline__ int la_dec_per(int len, int splits) {
    const int c = (len + splits - 1) / splits;
    return (c + (LA_DEC_SPLIT_ALIGN - 1)) & ~(LA_DEC_SPLIT_ALIGN - 1);
}

// Sum over the 16 lanes of an aligned group, fixed butterfly (every lane of
// the group ends with the same bits: each step adds commutative pairs).
static __device__ __forceinline__ float la_dec_sum16(float v) {
    v += __shfl_xor(v, 8);
    v += __shfl_xor(v, 4);
    v += __shfl_xor(v, 2);
    v += __shfl_xor(v, 1);
    return v;
}
static __device__ __forceinline__ float la_dec_max16(float v) {
    v = fmaxf(v, __shfl_xor(v, 8));
    v = fmaxf(v, __shfl_xor(v, 4));
    v = fmaxf(v, __shfl_xor(v, 2));
    v = fmaxf(v, __shfl_xor(v, 1));
    return v;
}

static __device__ __forceinline__ float la_dec_dot8(float4 qa, float4 qb, uint4 u, float s) {
    s = fmaf(qa.x, la_dec_lo(u.x), s);
    s = fmaf(qa.y, la_dec_hi(u.x), s);
    s = fmaf(qa.z, la_dec_lo(u.y), s);
    s = fmaf(qa.w, la_dec_hi(u.y), s);
    s = fmaf(qb.x, la_dec_lo(u.z), s);
    s = fmaf(qb.y, la_dec_hi(u.z), s);
    s = fmaf(qb.z, la_dec_lo(u.w), s);
    s = fmaf(qb.w, la_dec_hi(u.w), s);
    return s;
}

static __device__ __forceinline__ void la_dec_axpy8(float* acc, float p, uint4 u) {
    acc[0] = fmaf(p, la_dec_lo(u.x), acc[0]);
    acc[1] = fmaf(p, la_dec_hi(u.x), acc[1]);
    acc[2] = fmaf(p, la_dec_lo(u.y), acc[2]);
    acc[3] = fmaf(p, la_dec_hi(u.y), acc[3]);
    acc[4] = fmaf(p, la_dec_lo(u.z), acc[4]);
    acc[5] = fmaf(p, la_dec_hi(u.z), acc[5]);
    acc[6] = fmaf(p, la_dec_lo(u.w), acc[6]);
    acc[7] = fmaf(p, la_dec_hi(u.w), acc[7]);
}

extern "C" __global__ __launch_bounds__(LA_DEC_THREADS) void latent_attention_decode_partial(
    const float* __restrict__ q,         // [n_heads, wa + wb] F32
    const uint16_t* __restrict__ ka,     // k_a view base (offset applied)
    const uint16_t* __restrict__ kb,     // k_b view base
    const uint16_t* __restrict__ vv,     // v view base
    float* __restrict__ ws,              // [n_heads][splits][dv + 2]
    const int* __restrict__ pos_ptr,     // device scalar
    long long ka_tok, long long ka_head,
    long long kb_tok, long long kb_head,
    long long v_tok, long long v_head,
    int wa, int wb, int dv, int pos_offset, int max_rows, int splits, float scale
) {
    const int split = blockIdx.x;
    const int h = blockIdx.y;

    const int len = la_dec_visible(pos_ptr, pos_offset, max_rows);
    const int per = la_dec_per(len, splits);
    const long long begin_ll = (long long)split * (long long)per;
    if (begin_ll >= (long long)len) return;  // empty split (also len == 0): nothing touched
    const int begin = (int)begin_ll;
    const int end = (begin_ll + per < (long long)len) ? (int)(begin_ll + per) : len;

    __shared__ __attribute__((aligned(16))) float sq[LA_DEC_MAX_D];  // q, then the group-sum scratch
    __shared__ float ss[LA_DEC_TILE];
    __shared__ float sp[LA_DEC_TILE];

    const int tid = threadIdx.x;
    const int dq = wa + wb;
    const int nva = wa >> 3;
    const int nv = dq >> 3;
    const int nvv = dv >> 3;                       // value vectors per row
    const int vgroups = LA_DEC_THREADS / nvv;      // >= 1 (dv <= 2048)
    const int grp = tid >> 4;
    const int lane16 = tid & 15;
    const int vg = tid / nvv;
    const int vl = tid - vg * nvv;
    const bool v_active = vg < vgroups;

    const float* qr = q + (size_t)h * (size_t)dq;
    for (int d = tid; d < dq; d += LA_DEC_THREADS) sq[d] = qr[d];

    const uint16_t* ka_h = ka + (size_t)h * (size_t)ka_head;
    const uint16_t* kb_h = kb + (size_t)h * (size_t)kb_head;
    const uint16_t* v_h = vv + (size_t)h * (size_t)v_head + (size_t)vl * 8;

    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    __syncthreads();  // q staged

    for (int t0 = begin; t0 < end; t0 += LA_DEC_TILE) {
        const int cnt = (end - t0 < LA_DEC_TILE) ? (end - t0) : LA_DEC_TILE;

        // ---- scores: 16 groups x 4 keys ----
        {
            const uint16_t* pa[LA_DEC_KEYS_PER_GROUP];
            const uint16_t* pb[LA_DEC_KEYS_PER_GROUP];
            bool ok[LA_DEC_KEYS_PER_GROUP];
            float s[LA_DEC_KEYS_PER_GROUP];
            #pragma unroll
            for (int r = 0; r < LA_DEC_KEYS_PER_GROUP; ++r) {
                const int kk = grp + LA_DEC_GROUP * r;
                ok[r] = kk < cnt;
                const size_t j = (size_t)(t0 + kk);
                pa[r] = ka_h + j * (size_t)ka_tok;
                pb[r] = kb_h + j * (size_t)kb_tok;
                s[r] = 0.0f;
            }
            for (int vi = lane16; vi < nv; vi += LA_DEC_GROUP) {
                const float4 q0 = *(const float4*)(sq + vi * 8);
                const float4 q1 = *(const float4*)(sq + vi * 8 + 4);
                const bool in_a = vi < nva;
                const int off = in_a ? vi * 8 : (vi - nva) * 8;
                uint4 u[LA_DEC_KEYS_PER_GROUP];
                #pragma unroll
                for (int r = 0; r < LA_DEC_KEYS_PER_GROUP; ++r) {
                    u[r] = make_uint4(0u, 0u, 0u, 0u);
                    if (ok[r]) u[r] = *(const uint4*)((in_a ? pa[r] : pb[r]) + off);
                }
                #pragma unroll
                for (int r = 0; r < LA_DEC_KEYS_PER_GROUP; ++r) s[r] = la_dec_dot8(q0, q1, u[r], s[r]);
            }
            #pragma unroll
            for (int r = 0; r < LA_DEC_KEYS_PER_GROUP; ++r) s[r] = la_dec_sum16(s[r]);
            if (lane16 == 0) {
                #pragma unroll
                for (int r = 0; r < LA_DEC_KEYS_PER_GROUP; ++r)
                    ss[grp + LA_DEC_GROUP * r] = ok[r] ? s[r] * scale : -INFINITY;
            }
        }
        __syncthreads();  // A: scores visible

        // ---- tile softmax statistics (identical in every 16-lane group) ----
        float sv[LA_DEC_KEYS_PER_GROUP];
        float tmax = -INFINITY;
        #pragma unroll
        for (int r = 0; r < LA_DEC_KEYS_PER_GROUP; ++r) {
            sv[r] = ss[lane16 + LA_DEC_GROUP * r];
            tmax = fmaxf(tmax, sv[r]);
        }
        tmax = la_dec_max16(tmax);
        const float m_new = fmaxf(m, tmax);
        const float corr = (m == -INFINITY) ? 0.0f : expf(m - m_new);
        float pv[LA_DEC_KEYS_PER_GROUP];
        float psum = 0.0f;
        #pragma unroll
        for (int r = 0; r < LA_DEC_KEYS_PER_GROUP; ++r) {
            pv[r] = (sv[r] == -INFINITY) ? 0.0f : expf(sv[r] - m_new);
            psum += pv[r];
        }
        psum = la_dec_sum16(psum);
        l = l * corr + psum;
        m = m_new;
        if (tid < LA_DEC_GROUP) {
            #pragma unroll
            for (int r = 0; r < LA_DEC_KEYS_PER_GROUP; ++r) sp[lane16 + LA_DEC_GROUP * r] = pv[r];
        }
        __syncthreads();  // B: probabilities visible, scores consumed

        // ---- values: ascending keys per group, online rescale ----
        if (v_active) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] *= corr;
            int kk = vg;
            for (; kk + 3 * vgroups < cnt; kk += 4 * vgroups) {
                const float p0 = sp[kk];
                const float p1 = sp[kk + vgroups];
                const float p2 = sp[kk + 2 * vgroups];
                const float p3 = sp[kk + 3 * vgroups];
                const uint4 u0 = *(const uint4*)(v_h + (size_t)(t0 + kk) * (size_t)v_tok);
                const uint4 u1 = *(const uint4*)(v_h + (size_t)(t0 + kk + vgroups) * (size_t)v_tok);
                const uint4 u2 = *(const uint4*)(v_h + (size_t)(t0 + kk + 2 * vgroups) * (size_t)v_tok);
                const uint4 u3 = *(const uint4*)(v_h + (size_t)(t0 + kk + 3 * vgroups) * (size_t)v_tok);
                la_dec_axpy8(acc, p0, u0);
                la_dec_axpy8(acc, p1, u1);
                la_dec_axpy8(acc, p2, u2);
                la_dec_axpy8(acc, p3, u3);
            }
            for (; kk < cnt; kk += vgroups) {
                const uint4 u0 = *(const uint4*)(v_h + (size_t)(t0 + kk) * (size_t)v_tok);
                la_dec_axpy8(acc, sp[kk], u0);
            }
        }
    }

    // Every thread has passed barrier A of the last tile, so no thread still
    // reads sq: reuse it for the per-group accumulators.
    if (v_active) {
        float4* dst = (float4*)(sq + (size_t)vg * (size_t)dv + (size_t)vl * 8);
        dst[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        dst[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
    }
    __syncthreads();

    float* rec = ws + ((size_t)h * (size_t)splits + (size_t)split) * (size_t)(dv + 2);
    if (tid == 0) {
        rec[0] = m;
        rec[1] = l;
    }
    for (int d = tid; d < dv; d += LA_DEC_THREADS) {
        float t = sq[d];
        for (int g = 1; g < vgroups; ++g) t += sq[(size_t)g * (size_t)dv + d];
        rec[2 + d] = t;
    }
}

extern "C" __global__ __launch_bounds__(LA_DEC_THREADS) void latent_attention_decode_merge(
    const float* __restrict__ ws,        // [n_heads][splits][dv + 2]
    float* __restrict__ y,               // [n_heads, dv]
    const int* __restrict__ pos_ptr,     // device scalar
    int dv, int pos_offset, int max_rows, int splits, int round_y_bf16
) {
    __shared__ float sw[LA_DEC_MAX_SPLITS];  // w_s
    __shared__ float sl[LA_DEC_MAX_SPLITS];  // l_s * w_s

    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    float* yr = y + (size_t)h * (size_t)dv;

    const int len = la_dec_visible(pos_ptr, pos_offset, max_rows);
    if (len == 0) {
        for (int d = tid; d < dv; d += LA_DEC_THREADS) yr[d] = 0.0f;
        return;
    }
    const int per = la_dec_per(len, splits);
    const int ns = (len + per - 1) / per;  // non-empty splits, ascending [0, ns)
    const size_t rec_stride = (size_t)(dv + 2);
    const float* wh = ws + (size_t)h * (size_t)splits * rec_stride;

    float M = -INFINITY;
    for (int s = 0; s < ns; ++s) M = fmaxf(M, wh[(size_t)s * rec_stride]);
    if (tid < ns) {
        const float ms = wh[(size_t)tid * rec_stride];
        const float w = (ms == -INFINITY) ? 0.0f : expf(ms - M);
        sw[tid] = w;
        sl[tid] = wh[(size_t)tid * rec_stride + 1] * w;
    }
    __syncthreads();
    float L = 0.0f;
    for (int s = 0; s < ns; ++s) L += sl[s];

    for (int d = tid; d < dv; d += LA_DEC_THREADS) {
        float a = 0.0f;
        for (int s = 0; s < ns; ++s) a = fmaf(wh[(size_t)s * rec_stride + 2 + d], sw[s], a);
        float o = (L == 0.0f) ? 0.0f : a / L;
        if (round_y_bf16) o = la_bf16_to_f32(la_f32_to_bf16(o));
        yr[d] = o;
    }
}

// ============================================================================
// Q8 expanded-mirror variant of the partials kernel (merge kernel unchanged).
//
// `ka` (K_nope) and `vv` (V) are INT8 mirror planes (see
// latent_attention_decompress.hip): a plane row is `n_heads * (w + 2)` bytes,
// `[n_heads * w codes | n_heads F16 scales]`; ka_tok / v_tok are BYTE strides of
// whole plane rows, ka_head / v_head == w; the scale of (key j, head h) is the
// F16 at byte `j * tok + n_heads * w + 2 * h`. The tail view `kb` is BF16.
//
//   score(j) = scale * ( ks_j * dot(q[0..wa], codes_j)  +  dot(q[wa..], tail_j) )
//   y       += p_j * vs_j * codes_v_j
//
// i.e. the per-row scales are applied once per row (after the 16-lane butterfly
// for K, folded into the probability for V); the codes widen exactly to F32.
// Split layout, softmax and the fixed reduction orders are those of the BF16
// kernel, so the same determinism contract holds.
// ============================================================================

static __device__ __forceinline__ float la_dec_i8(uint32_t w, int b) {
    return (float)((int)(w << (24 - 8 * b)) >> 24);
}

static __device__ __forceinline__ float la_dec_f16_at(const uint8_t* p) {
    return (float)(*(const _Float16*)p);
}

static __device__ __forceinline__ float la_dec_dot8_q8(float4 qa, float4 qb, uint2 u, float s) {
    s = fmaf(qa.x, la_dec_i8(u.x, 0), s);
    s = fmaf(qa.y, la_dec_i8(u.x, 1), s);
    s = fmaf(qa.z, la_dec_i8(u.x, 2), s);
    s = fmaf(qa.w, la_dec_i8(u.x, 3), s);
    s = fmaf(qb.x, la_dec_i8(u.y, 0), s);
    s = fmaf(qb.y, la_dec_i8(u.y, 1), s);
    s = fmaf(qb.z, la_dec_i8(u.y, 2), s);
    s = fmaf(qb.w, la_dec_i8(u.y, 3), s);
    return s;
}

static __device__ __forceinline__ void la_dec_axpy8_q8(float* acc, float p, uint2 u) {
    acc[0] = fmaf(p, la_dec_i8(u.x, 0), acc[0]);
    acc[1] = fmaf(p, la_dec_i8(u.x, 1), acc[1]);
    acc[2] = fmaf(p, la_dec_i8(u.x, 2), acc[2]);
    acc[3] = fmaf(p, la_dec_i8(u.x, 3), acc[3]);
    acc[4] = fmaf(p, la_dec_i8(u.y, 0), acc[4]);
    acc[5] = fmaf(p, la_dec_i8(u.y, 1), acc[5]);
    acc[6] = fmaf(p, la_dec_i8(u.y, 2), acc[6]);
    acc[7] = fmaf(p, la_dec_i8(u.y, 3), acc[7]);
}

extern "C" __global__ __launch_bounds__(LA_DEC_THREADS) void latent_attention_decode_partial_q8(
    const float* __restrict__ q,         // [n_heads, wa + wb] F32
    const uint8_t* __restrict__ ka,      // INT8 K_nope plane view base
    const uint16_t* __restrict__ kb,     // BF16 tail view base
    const uint8_t* __restrict__ vv,      // INT8 V plane view base
    float* __restrict__ ws,              // [n_heads][splits][dv + 2]
    const int* __restrict__ pos_ptr,     // device scalar
    long long ka_tok, long long ka_head,
    long long kb_tok, long long kb_head,
    long long v_tok, long long v_head,
    int wa, int wb, int dv, int pos_offset, int max_rows, int splits, float scale,
    int n_heads
) {
    const int split = blockIdx.x;
    const int h = blockIdx.y;

    const int len = la_dec_visible(pos_ptr, pos_offset, max_rows);
    const int per = la_dec_per(len, splits);
    const long long begin_ll = (long long)split * (long long)per;
    if (begin_ll >= (long long)len) return;  // empty split (also len == 0): nothing touched
    const int begin = (int)begin_ll;
    const int end = (begin_ll + per < (long long)len) ? (int)(begin_ll + per) : len;

    __shared__ __attribute__((aligned(16))) float sq[LA_DEC_MAX_D];  // q, then the group-sum scratch
    __shared__ float ss[LA_DEC_TILE];
    __shared__ float sp[LA_DEC_TILE];

    const int tid = threadIdx.x;
    const int dq = wa + wb;
    const int nva = wa >> 3;
    const int nv = dq >> 3;
    const int nvv = dv >> 3;                       // value vectors per row
    const int vgroups = LA_DEC_THREADS / nvv;      // >= 1 (dv <= 2048)
    const int grp = tid >> 4;
    const int lane16 = tid & 15;
    const int vg = tid / nvv;
    const int vl = tid - vg * nvv;
    const bool v_active = vg < vgroups;

    const float* qr = q + (size_t)h * (size_t)dq;
    for (int d = tid; d < dq; d += LA_DEC_THREADS) sq[d] = qr[d];

    const uint8_t* ka_h = ka + (size_t)h * (size_t)ka_head;
    const uint16_t* kb_h = kb + (size_t)h * (size_t)kb_head;
    const uint8_t* v_h = vv + (size_t)h * (size_t)v_head + (size_t)vl * 8;
    const size_t ka_sc = (size_t)n_heads * (size_t)wa + 2 * (size_t)h;  // scale byte offset in a K row
    const size_t v_sc = (size_t)n_heads * (size_t)dv + 2 * (size_t)h;   // scale byte offset in a V row

    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    __syncthreads();  // q staged

    for (int t0 = begin; t0 < end; t0 += LA_DEC_TILE) {
        const int cnt = (end - t0 < LA_DEC_TILE) ? (end - t0) : LA_DEC_TILE;

        // ---- scores: 16 groups x 4 keys ----
        {
            const uint8_t* pa[LA_DEC_KEYS_PER_GROUP];
            const uint16_t* pb[LA_DEC_KEYS_PER_GROUP];
            bool ok[LA_DEC_KEYS_PER_GROUP];
            float sn[LA_DEC_KEYS_PER_GROUP];   // nope part (codes), scaled after the butterfly
            float st[LA_DEC_KEYS_PER_GROUP];   // tail part
            float ksc[LA_DEC_KEYS_PER_GROUP];
            #pragma unroll
            for (int r = 0; r < LA_DEC_KEYS_PER_GROUP; ++r) {
                const int kk = grp + LA_DEC_GROUP * r;
                ok[r] = kk < cnt;
                const size_t j = (size_t)(t0 + kk);
                pa[r] = ka_h + j * (size_t)ka_tok;
                pb[r] = kb_h + j * (size_t)kb_tok;
                sn[r] = 0.0f;
                st[r] = 0.0f;
                ksc[r] = ok[r] ? la_dec_f16_at(ka + j * (size_t)ka_tok + ka_sc) : 0.0f;
            }
            for (int vi = lane16; vi < nv; vi += LA_DEC_GROUP) {
                const float4 q0 = *(const float4*)(sq + vi * 8);
                const float4 q1 = *(const float4*)(sq + vi * 8 + 4);
                if (vi < nva) {
                    uint2 u[LA_DEC_KEYS_PER_GROUP];
                    #pragma unroll
                    for (int r = 0; r < LA_DEC_KEYS_PER_GROUP; ++r) {
                        u[r] = make_uint2(0u, 0u);
                        if (ok[r]) u[r] = *(const uint2*)(pa[r] + vi * 8);
                    }
                    #pragma unroll
                    for (int r = 0; r < LA_DEC_KEYS_PER_GROUP; ++r) sn[r] = la_dec_dot8_q8(q0, q1, u[r], sn[r]);
                } else {
                    const int off = (vi - nva) * 8;
                    uint4 u[LA_DEC_KEYS_PER_GROUP];
                    #pragma unroll
                    for (int r = 0; r < LA_DEC_KEYS_PER_GROUP; ++r) {
                        u[r] = make_uint4(0u, 0u, 0u, 0u);
                        if (ok[r]) u[r] = *(const uint4*)(pb[r] + off);
                    }
                    #pragma unroll
                    for (int r = 0; r < LA_DEC_KEYS_PER_GROUP; ++r) st[r] = la_dec_dot8(q0, q1, u[r], st[r]);
                }
            }
            #pragma unroll
            for (int r = 0; r < LA_DEC_KEYS_PER_GROUP; ++r) {
                sn[r] = la_dec_sum16(sn[r]);
                st[r] = la_dec_sum16(st[r]);
            }
            if (lane16 == 0) {
                #pragma unroll
                for (int r = 0; r < LA_DEC_KEYS_PER_GROUP; ++r)
                    ss[grp + LA_DEC_GROUP * r] = ok[r] ? fmaf(sn[r], ksc[r], st[r]) * scale : -INFINITY;
            }
        }
        __syncthreads();  // A: scores visible

        // ---- tile softmax statistics (identical in every 16-lane group) ----
        float sv[LA_DEC_KEYS_PER_GROUP];
        float tmax = -INFINITY;
        #pragma unroll
        for (int r = 0; r < LA_DEC_KEYS_PER_GROUP; ++r) {
            sv[r] = ss[lane16 + LA_DEC_GROUP * r];
            tmax = fmaxf(tmax, sv[r]);
        }
        tmax = la_dec_max16(tmax);
        const float m_new = fmaxf(m, tmax);
        const float corr = (m == -INFINITY) ? 0.0f : expf(m - m_new);
        float pv[LA_DEC_KEYS_PER_GROUP];
        float psum = 0.0f;
        #pragma unroll
        for (int r = 0; r < LA_DEC_KEYS_PER_GROUP; ++r) {
            pv[r] = (sv[r] == -INFINITY) ? 0.0f : expf(sv[r] - m_new);
            psum += pv[r];
        }
        psum = la_dec_sum16(psum);
        l = l * corr + psum;
        m = m_new;
        if (tid < LA_DEC_GROUP) {
            #pragma unroll
            for (int r = 0; r < LA_DEC_KEYS_PER_GROUP; ++r) sp[lane16 + LA_DEC_GROUP * r] = pv[r];
        }
        __syncthreads();  // B: probabilities visible, scores consumed

        // ---- values: ascending keys per group, online rescale ----
        if (v_active) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] *= corr;
            int kk = vg;
            for (; kk + 3 * vgroups < cnt; kk += 4 * vgroups) {
                const size_t j0 = (size_t)(t0 + kk) * (size_t)v_tok;
                const size_t j1 = (size_t)(t0 + kk + vgroups) * (size_t)v_tok;
                const size_t j2 = (size_t)(t0 + kk + 2 * vgroups) * (size_t)v_tok;
                const size_t j3 = (size_t)(t0 + kk + 3 * vgroups) * (size_t)v_tok;
                const float p0 = sp[kk] * la_dec_f16_at(vv + j0 + v_sc);
                const float p1 = sp[kk + vgroups] * la_dec_f16_at(vv + j1 + v_sc);
                const float p2 = sp[kk + 2 * vgroups] * la_dec_f16_at(vv + j2 + v_sc);
                const float p3 = sp[kk + 3 * vgroups] * la_dec_f16_at(vv + j3 + v_sc);
                const uint2 u0 = *(const uint2*)(v_h + j0);
                const uint2 u1 = *(const uint2*)(v_h + j1);
                const uint2 u2 = *(const uint2*)(v_h + j2);
                const uint2 u3 = *(const uint2*)(v_h + j3);
                la_dec_axpy8_q8(acc, p0, u0);
                la_dec_axpy8_q8(acc, p1, u1);
                la_dec_axpy8_q8(acc, p2, u2);
                la_dec_axpy8_q8(acc, p3, u3);
            }
            for (; kk < cnt; kk += vgroups) {
                const size_t j0 = (size_t)(t0 + kk) * (size_t)v_tok;
                const float p0 = sp[kk] * la_dec_f16_at(vv + j0 + v_sc);
                const uint2 u0 = *(const uint2*)(v_h + j0);
                la_dec_axpy8_q8(acc, p0, u0);
            }
        }
    }

    // Every thread has passed barrier A of the last tile, so no thread still
    // reads sq: reuse it for the per-group accumulators.
    if (v_active) {
        float4* dst = (float4*)(sq + (size_t)vg * (size_t)dv + (size_t)vl * 8);
        dst[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        dst[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
    }
    __syncthreads();

    float* rec = ws + ((size_t)h * (size_t)splits + (size_t)split) * (size_t)(dv + 2);
    if (tid == 0) {
        rec[0] = m;
        rec[1] = l;
    }
    for (int d = tid; d < dv; d += LA_DEC_THREADS) {
        float t = sq[d];
        for (int g = 1; g < vgroups; ++g) t += sq[(size_t)g * (size_t)dv + d];
        rec[2 + d] = t;
    }
}
