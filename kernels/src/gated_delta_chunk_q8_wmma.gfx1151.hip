// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Q8-state variant of gated_delta_chunk_wmma.gfx1151.hip's gated_delta_chunk_gate_wmma (the
// chunked WY-form Gated DeltaNet prefill recurrence + gated RMSNorm on gfx11 WMMA; see that
// file for the recurrence).  The recurrence here is a verbatim copy; only the state load and
// final store differ, so a Q8 GDN slot is decoded to and requantized from the F32 state inside
// the kernel instead of by gdn_state_q8_to_f32 / gdn_state_f32_to_q8 launches around it.  It
// lives in its own module so that the F32 kernel's code object is not perturbed by it.
// Grid [value_heads], block 256.  key_dim == value_dim == 128.

typedef _Float16 __attribute__((ext_vector_type(16))) gc_h16;
typedef float __attribute__((ext_vector_type(8))) gc_f8;
typedef unsigned int __attribute__((ext_vector_type(8))) gc_u8;
typedef _Float16 __attribute__((ext_vector_type(2))) gc_h2;
// __syncthreads also waits for every outstanding global access (WGP-mode
// workgroup fence), which would drain the prefetch.
#define GC_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
// Wave-private LDS handoff: order this wave's stores before its loads.
#define GC_WAVE_SYNC() do { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront"); __builtin_amdgcn_wave_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront"); } while (0)
#define GC_WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)
static __device__ __forceinline__ float gc_bf16r(float v) { return (float)(hip_bfloat16)v; }
static __device__ __forceinline__ unsigned int gc_pack(float a, float b) {
    return __builtin_bit_cast(unsigned int, gc_h2{(_Float16)a, (_Float16)b});
}
// C-layout tile (lane: column l16, rows 2r + hi) -> the B operand of that
// matrix (lane: column l16, all 16 rows), by swapping half-waves.
static __device__ __forceinline__ gc_h16 gc_c_to_b(const gc_f8 c, int hi) {
    gc_u8 out;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const unsigned int own = gc_pack(c[2 * p], c[2 * p + 1]);
        const unsigned int oth = __builtin_amdgcn_permlanex16(own, own, 0x76543210u, 0xfedcba98u, false, false);
        const unsigned int a = hi ? oth : own, b = hi ? own : oth;
        out[2 * p] = __builtin_amdgcn_perm(b, a, 0x05040100u);
        out[2 * p + 1] = __builtin_amdgcn_perm(b, a, 0x07060302u);
    }
    return __builtin_bit_cast(gc_h16, out);
}
// Q8 GDN state codes, mirrored expression-for-expression from tensor_ops.hip (gdn_q8_round /
// gdn_q8_inv / gdn_q8_scale); `seed` is head * 1664525 + row * 1013904223 + frame * 2531011.
// value * inv is a separately rounded product (the conversion kernel's ISA multiplies, then
// folds only the exact noise product into the add), so contraction is pinned off here.
static __device__ __forceinline__ signed char gc_q8_round(float value, float inv, unsigned int seed, int column) {
    unsigned int rng = (seed + (unsigned int)column * 214013u) * 1664525u + 1013904223u;
    const float noise = (float)(rng >> 8) * (1.0f / 16777216.0f);
    return (signed char)fminf(fmaxf(floorf(__fadd_rn(__fmul_rn(value, inv), noise)), -128.0f), 127.0f);
}
static __device__ __forceinline__ float gc_q8_inv(float peak) { return peak > 0.0f ? 127.0f / peak : 0.0f; }
static __device__ __forceinline__ float gc_q8_scale(float peak) { return peak > 0.0f ? peak / 127.0f : 1.0f; }

// gated_delta_chunk_gate_wmma with the state either F32 (state_q8 == 0; `frame` unused) or a Q8
// GDN slot (state_q8 == 1): decoded to the F32 `scale * code` on load, requantized from the
// final registers with the last row's absolute `frame` (position + rows - 1).  state_q8 == 2 is
// the Q8 path plus a capture of the final F32 state ([head][key][value]) right after the slot
// (slot bytes = value_heads * 128 * 132), for tests.  Arguments are gated_delta_chunk_gate_wmma's
// with `float* state` -> `void* state`, then `int state_q8, unsigned int frame` appended.
__launch_bounds__(256)
extern "C" __global__ void gated_delta_chunk_gate_q8_wmma(
    const unsigned short* __restrict__ projection, const unsigned short* __restrict__ qn, const unsigned short* __restrict__ kn,
    const float* __restrict__ gate, const float* __restrict__ beta, void* __restrict__ state,
    const float* __restrict__ z, const hip_bfloat16* __restrict__ norm, void* __restrict__ output,
    int rows, int qkv_width, int key_heads, int value_heads, float query_scale, int output_bf16,
    int state_q8, unsigned int frame) {
    const int vh = blockIdx.x;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int kh = vh / (value_heads / key_heads);
    const int v_offset = 2 * key_heads * 128 + vh * 128;
    const int value_width = value_heads * 128;
    const size_t state_base = (size_t)vh * 128 * 128;
    __shared__ __align__(16) _Float16 k_s[16][136];    // K rows
    __shared__ __align__(16) _Float16 q_s[16][136];    // scaled Q rows
    __shared__ __align__(16) _Float16 kb_s[16][136];   // k beta e^G rows
    __shared__ __align__(16) _Float16 kdT[128][16];    // (k e^{GL-G})^T
    __shared__ float vb_s[16][132];                    // v beta rows (F32)
    __shared__ __align__(16) _Float16 t_s[16][16];
    __shared__ __align__(16) _Float16 p_s[16][16];
    __shared__ float g_s[16], bt_s[16], inv_s[16];
    __shared__ __align__(16) unsigned short out_s[16][136];
    const int dv0 = wave * 16;
    gc_f8 S[8];
    if (state_q8 == 0) {
        const float* state_f32 = (const float*)state;
#pragma unroll
        for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int r = 0; r < 8; ++r)
                S[t][r] = state_f32[state_base + (size_t)(16 * t + 2 * r + hi) * 128 + dv0 + l16];
    } else {
        // Q8 slot: this lane's value row dv0 + l16, keys 16t + 2r + hi, as the exact
        // `scale * code` F32 gdn_state_q8_to_f32 would store (byte 2r + hi of tile t's 16 codes).
        const signed char* codes = (const signed char*)state;
        const float scale = ((const float*)(codes + (size_t)value_heads * 128 * 128))[vh * 128 + dv0 + l16];
        const int4* row = (const int4*)(codes + ((size_t)vh * 128 + dv0 + l16) * 128);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int4 w = row[t];
            const int words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int r = 0; r < 8; ++r)
                S[t][r] = scale * (float)(signed char)(words[r >> 1] >> ((r & 1) * 16 + hi * 8));
        }
    }
    const int si = tid & 15, sd = (tid >> 4) * 8;
    // Raw prefetch registers, widened where used so the loads stay in flight.
    uint4 pk, pq, pv; float pg = 0.f, pb = 0.f;
    const float nw = (float)norm[tid & 127];
    auto fetch = [&](int c0) {
        const int nn = min(16, rows - c0);
        pk = make_uint4(0, 0, 0, 0); pq = pk; pv = pk;
        if (si < nn) {
            const size_t b = ((size_t)(c0 + si) * key_heads + kh) * 128 + sd;
            pk = *(const uint4*)(kn + b);
            pq = *(const uint4*)(qn + b);
            pv = *(const uint4*)(projection + (size_t)(c0 + si) * qkv_width + v_offset + sd);
        }
        if (tid < 16) {
            pg = tid < nn ? gate[(c0 + tid) * value_heads + vh] : 0.f;
            pb = tid < nn ? beta[(c0 + tid) * value_heads + vh] : 0.f;
        }
    };
    // Gated RMSNorm of a finished chunk's rows (out_s, all 128 columns).
    auto epilogue = [&](int c0, const float* zc) {
        const int nn = min(16, rows - c0);
        {
            const int r = 2 * wave + hi;
            const uint4 w = *(const uint4*)&out_s[r][l16 * 8];
            const unsigned int wv[4] = {w.x, w.y, w.z, w.w};
            float sum = 0.f;
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const float lo = __uint_as_float(wv[h] << 16), hv = __uint_as_float(wv[h] & 0xFFFF0000u);
                sum += lo * lo; sum += hv * hv;
            }
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 16);
            if (l16 == 0) inv_s[r] = rsqrtf(sum / 128.0f + 1.0e-6f);
        }
        GC_BARRIER();
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int e = tid + 256 * i, r = e >> 7, c = e & 127;
            if (r < nn) {
                const size_t at = (size_t)(c0 + r) * value_width + vh * 128 + c;
                const float v = __uint_as_float((unsigned int)out_s[r][c] << 16);
                const float normalized = gc_bf16r(v * inv_s[r] * nw);
                const float g = gc_bf16r(zc[i]);
                const float y = gc_bf16r(normalized / (1.0f + expf(-g)));
                if (output_bf16)
                    ((unsigned short*)output)[at] = (unsigned short)(__float_as_uint(y) >> 16);
                else
                    ((float*)output)[at] = y;
            }
        }
    };
    fetch(0);
    for (int t0 = 0; t0 < rows; t0 += 16) {
        float zc[8];
        {
            const int nn = min(16, rows - t0);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int e = tid + 256 * i, r = e >> 7;
                zc[i] = r < nn ? z[(size_t)(t0 + r) * value_width + vh * 128 + (e & 127)] : 0.0f;
            }
        }
        const unsigned int kw[4] = {pk.x, pk.y, pk.z, pk.w}, qw[4] = {pq.x, pq.y, pq.z, pq.w};
        float kf[8];
        const float vf[8] = {__uint_as_float(pv.x << 16), __uint_as_float(pv.x & 0xFFFF0000u),
                             __uint_as_float(pv.y << 16), __uint_as_float(pv.y & 0xFFFF0000u),
                             __uint_as_float(pv.z << 16), __uint_as_float(pv.z & 0xFFFF0000u),
                             __uint_as_float(pv.w << 16), __uint_as_float(pv.w & 0xFFFF0000u)};
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            kf[2 * h] = __uint_as_float(kw[h] << 16); kf[2 * h + 1] = __uint_as_float(kw[h] & 0xFFFF0000u);
            *(unsigned int*)&k_s[si][sd + 2 * h] = gc_pack(kf[2 * h], kf[2 * h + 1]);
            *(unsigned int*)&q_s[si][sd + 2 * h] = gc_pack(__uint_as_float(qw[h] << 16) * query_scale,
                                                          __uint_as_float(qw[h] & 0xFFFF0000u) * query_scale);
        }
        if (tid < 16) {
            float G = pg;
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) {
                const float up = __shfl_up(G, o, 16);
                if (tid >= o) G += up;
            }
            g_s[tid] = G;
            bt_s[tid] = pb;
        }
        GC_BARRIER();
        {
            const float b = bt_s[si], eg = expf(g_s[si]), ed = expf(g_s[15] - g_s[si]);
#pragma unroll
            for (int h = 0; h < 8; ++h) {
                vb_s[si][sd + h] = vf[h] * b;
                kdT[sd + h][si] = (_Float16)(kf[h] * ed);
            }
#pragma unroll
            for (int h = 0; h < 4; ++h)
                *(unsigned int*)&kb_s[si][sd + 2 * h] = gc_pack(kf[2 * h] * b * eg, kf[2 * h + 1] * b * eg);
        }
        if (t0 + 16 < rows) fetch(t0 + 16);
        GC_BARRIER();
        if (wave == 0) {
            gc_f8 c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < 8; ++s)
                c = GC_WMMA(*(const gc_h16*)&k_s[l16][16 * s], *(const gc_h16*)&k_s[l16][16 * s], c);
            // L strictly lower; T = (I - L)^-1 = (I + L)(I + L^2)(I + L^4)(I + L^8).
            gc_f8 T, P;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int i = 2 * r + hi, j = l16;
                P[r] = j < i ? -bt_s[i] * c[r] * expf(g_s[i] - g_s[j]) : 0.f;
                T[r] = P[r] + (i == j ? 1.f : 0.f);
            }
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                // P as A: rows via LDS (t_s scratch), as B: half-wave swap.
#pragma unroll
                for (int r = 0; r < 8; ++r) t_s[2 * r + hi][l16] = (_Float16)P[r];
                GC_WAVE_SYNC();
                const gc_h16 pa = *(const gc_h16*)&t_s[l16][0];
                const gc_h16 pbv = gc_c_to_b(P, hi);
                GC_WAVE_SYNC();
#pragma unroll
                for (int r = 0; r < 8; ++r) t_s[2 * r + hi][l16] = (_Float16)T[r];
                GC_WAVE_SYNC();
                const gc_h16 ta = *(const gc_h16*)&t_s[l16][0];
                gc_f8 P2 = {0, 0, 0, 0, 0, 0, 0, 0};
                P2 = GC_WMMA(pa, pbv, P2);
                T = GC_WMMA(ta, gc_c_to_b(P2, hi), T);
                P = P2;
                GC_WAVE_SYNC();
            }
#pragma unroll
            for (int r = 0; r < 8; ++r) t_s[2 * r + hi][l16] = (_Float16)T[r];
        } else if (wave == 1) {
            gc_f8 c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < 8; ++s)
                c = GC_WMMA(*(const gc_h16*)&q_s[l16][16 * s], *(const gc_h16*)&k_s[l16][16 * s], c);
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int i = 2 * r + hi, j = l16;
                p_s[i][j] = (_Float16)(j <= i ? c[r] * expf(g_s[i] - g_s[j]) : 0.f);
            }
        }
        // K' = (k beta e^G) S and O_inter = Q S on this wave's slice (S as B from registers).
        gc_f8 Ks = {0, 0, 0, 0, 0, 0, 0, 0}, Oi = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const gc_h16 sb = gc_c_to_b(S[t], hi);
            Ks = GC_WMMA(*(const gc_h16*)&kb_s[l16][16 * t], sb, Ks);
            Oi = GC_WMMA(*(const gc_h16*)&q_s[l16][16 * t], sb, Oi);
        }
        gc_f8 Dm;
#pragma unroll
        for (int r = 0; r < 8; ++r) Dm[r] = vb_s[2 * r + hi][dv0 + l16] - Ks[r];
        GC_BARRIER();  // T, P ready
        gc_f8 Vn = {0, 0, 0, 0, 0, 0, 0, 0};
        Vn = GC_WMMA(*(const gc_h16*)&t_s[l16][0], gc_c_to_b(Dm, hi), Vn);
        const gc_h16 vb = gc_c_to_b(Vn, hi);
        gc_f8 O = {0, 0, 0, 0, 0, 0, 0, 0};
        O = GC_WMMA(*(const gc_h16*)&p_s[l16][0], vb, O);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int i = 2 * r + hi;
            out_s[i][dv0 + l16] = __builtin_bit_cast(unsigned short, (hip_bfloat16)(O[r] + expf(g_s[i]) * Oi[r]));
        }
        const float eGL = expf(g_s[15]);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
#pragma unroll
            for (int r = 0; r < 8; ++r) S[t][r] *= eGL;
            S[t] = GC_WMMA(*(const gc_h16*)&kdT[16 * t + l16][0], vb, S[t]);
        }
        GC_BARRIER();  // chunk inputs consumed; out_s complete
        epilogue(t0, zc);
    }
    if (state_q8 == 0) {
        float* state_f32 = (float*)state;
#pragma unroll
        for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int r = 0; r < 8; ++r)
                state_f32[state_base + (size_t)(16 * t + 2 * r + hi) * 128 + dv0 + l16] = S[t][r];
        return;
    }
    // Q8 slot (tensor_ops.hip gdn_state_f32_to_q8's format and arithmetic).  This wave owns
    // value row dv0 + l16 over all 128 key channels (64 per lane, the other 64 in lane l16 + 16),
    // so the row's absmax is a lane max plus one half-wave swap; no other wave is involved.
    signed char* codes = (signed char*)state;
    float* scales = (float*)(codes + (size_t)value_heads * 128 * 128);
    if (state_q8 == 2) {
        // Test capture: the final F32 state ([head][key][value]) after the slot.
        float* capture = scales + (size_t)value_heads * 128;
#pragma unroll
        for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int r = 0; r < 8; ++r)
                capture[state_base + (size_t)(16 * t + 2 * r + hi) * 128 + dv0 + l16] = S[t][r];
    }
    const int row = dv0 + l16;
    float peak = fabsf(S[0][0]);
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int r = 0; r < 8; ++r) peak = fmaxf(peak, fabsf(S[t][r]));
    peak = fmaxf(peak, __shfl_xor(peak, 16, 32));
    const float inv = gc_q8_inv(peak);
    if (hi == 0) scales[vh * 128 + row] = gc_q8_scale(peak);
    const unsigned int seed = (unsigned int)vh * 1664525u + (unsigned int)row * 1013904223u + frame * 2531011u;
    // The recurrence's keys are rows of S; the slot's are bytes of a value row.  Codes go
    // through a wave-private LDS staging of 16 rows x 64 bytes (k_s / q_s are dead after the
    // last chunk's barrier) so each row is stored as contiguous 16-byte words.
    unsigned char* stage = (wave < 4 ? (unsigned char*)k_s : (unsigned char*)q_s) + (wave & 3) * 1024;
    signed char* dst = codes + ((size_t)vh * 128 + row) * 128;
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
#pragma unroll
        for (int tt = 0; tt < 4; ++tt) {
            const int t = 4 * pass + tt;
            unsigned int a0 = 0, a1 = 0;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const unsigned int c = (unsigned int)gc_q8_round(S[t][r], inv, seed, 16 * t + 2 * r + hi) & 0xFFu;
                if (r < 4) a0 |= c << (8 * r); else a1 |= c << (8 * (r - 4));
            }
            // Lane hi=0 holds even keys, hi=1 odd keys of the same row: swap halves so lane hi
            // owns keys 16t + 8hi .. +7 interleaved (even, odd).
            const unsigned int keep = hi ? a1 : a0, send = hi ? a0 : a1;
            const unsigned int recv = __builtin_amdgcn_permlanex16(send, send, 0x76543210u, 0xfedcba98u, false, false);
            const unsigned int ev = hi ? recv : keep, od = hi ? keep : recv;
            *(uint2*)(stage + l16 * 64 + tt * 16 + hi * 8) =
                make_uint2(__builtin_amdgcn_perm(od, ev, 0x05010400u), __builtin_amdgcn_perm(od, ev, 0x07030602u));
        }
        GC_WAVE_SYNC();
        const uint4 lo = *(const uint4*)(stage + l16 * 64 + hi * 32);
        const uint4 up = *(const uint4*)(stage + l16 * 64 + hi * 32 + 16);
        GC_WAVE_SYNC();
        *(uint4*)(dst + 64 * pass + hi * 32) = lo;
        *(uint4*)(dst + 64 * pass + hi * 32 + 16) = up;
    }
}
