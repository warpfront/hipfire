// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx12/RDNA4 sibling of attention_flux_vt_wmma_f16kv_f32.hip. Identical
// algorithm (V transposed while staged into LDS so the PV B-fragment is a
// contiguous ds_read_b128; online-softmax max/sum kept in registers). The only
// differences are the WMMA fragment mappings:
//
//   gfx11 (w32):   a[j] = A[l&15][j]        b[j] = B[j][l&15]
//                  c[j] = C[2*j + (l>>4)][l&15]           j = 0..15 / 0..7
//   gfx12 (w32):   a[j] = A[l&15][8*(l>>4)+j]  b[j] = B[8*(l>>4)+j][l&15]
//                  c[j] = C[8*(l>>4)+j][l&15]            j = 0..7
//
// so operands are half8 and each lane's 8 accumulator rows are *consecutive*
// (8*hi + j) rather than strided (2*j + hi). Every LDS stride and the banking
// argument are unchanged; see the base kernel for those notes.
//
// Q and out dtypes are template parameters, exactly as in the gfx11 sibling:
// `QT` x `OT` over {float, _Float16}, four entries, one kernarg layout.
//
// NOTE: compile-verified for gfx1201 only. No RDNA4 part was available when
// this was written, so it has not been run. That applies to the f16 entries
// too — they are compile-verified, never executed.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define AV_HD 128
#define AV_DC 8   // AV_HD / 16
#define AV_MT 64  // query rows per block
#define AV_NT 64  // keys per online-softmax tile
#define AV_NC 4   // AV_NT / 16
#define AV_VCHUNK 32
#define AV_VNC 2         // AV_VCHUNK / 16
#define AV_VT_STRIDE 40  // halves; 80 B, 16 B aligned
#define AV_S_STRIDE 72   // halves; 144 B, 16 B aligned
#define AV_VT_HALVES (AV_HD * AV_VT_STRIDE)
#define AV_LDS_BYTES ((AV_VT_HALVES + AV_MT * AV_S_STRIDE) * 2)  // 19456

// Epilogue store, specialised on the out dtype. KEEP IN SYNC with the copy in
// attention_flux_vt_wmma_f16kv_f32.hip (and the vtk kernel) — the three are
// deliberately identical, and the empty asm in the `_Float16` specialisation
// is LOAD-BEARING, not a leftover. It pins the f32 product in a VGPR so the
// narrowing stays a separate RNE rounding; without it the backend folds
// `fptrunc (fmul a, b)` into one `v_fma_mixlo_f16`, which rounds once and puts
// the f16 entry up to 1 ulp away from "f32 entry + RNE cast" — the equality
// that lets a caller drop its cast kernel. `#pragma clang fp contract(off)`
// does not suppress that fold. See the gfx11 sibling for the full note.
template <typename OT>
struct av_store;

template <>
struct av_store<float> {
    __device__ __forceinline__ static void put(float *p, float v) { *p = v; }
};

template <>
struct av_store<_Float16> {
    __device__ __forceinline__ static void put(_Float16 *p, float v) {
        asm("" : "+v"(v));
        *p = (_Float16)v;
    }
};

template <typename QT, typename OT>
__device__ __forceinline__ void
av_vt_gfx12_body(const QT *__restrict__ q, const _Float16 *__restrict__ k,
                 const _Float16 *__restrict__ v, OT *__restrict__ out, int B,
                 int L, int n_heads, int n_kv_heads, int head_dim,
                 float scale) {
    const int head = blockIdx.x;
    const int q_start = blockIdx.y * AV_MT;
    const int tid = threadIdx.x;

    if (head >= n_heads || q_start >= B) return;
    if (head_dim != AV_HD) return;

    const int rep = n_heads / n_kv_heads;
    const int kv_head = head / rep;
    const int q_stride = n_heads * AV_HD;
    const int kv_stride = n_kv_heads * AV_HD;

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int h = lane & 15;
    const int hi = lane >> 4;
    const int rb = wave * 16;
    const int ro = rb + hi * 8;  // this lane's first accumulator row

    extern __shared__ half8_t av_smem[];
    _Float16 *Vt = (_Float16 *)av_smem;
    _Float16 *Slds = Vt + AV_VT_HALVES;

    const _Float16 *kbase = k + kv_head * AV_HD;
    const _Float16 *vbase = v + kv_head * AV_HD;

    half8_t Qf[AV_DC];
    {
        const int gq = min(q_start + rb + h, B - 1);
        const QT *qr = q + gq * q_stride + head * AV_HD;
#pragma unroll
        for (int dc = 0; dc < AV_DC; ++dc) {
            const int d0 = dc * 16 + hi * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) Qf[dc][j] = (_Float16)qr[d0 + j];
        }
    }

    float8_t Of[AV_DC];
#pragma unroll
    for (int dc = 0; dc < AV_DC; ++dc) {
        Of[dc] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    float m_run[8], l_run[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        m_run[j] = -INFINITY;
        l_run[j] = 0.0f;
    }

    for (int kt = 0; kt < L; kt += AV_NT) {
        // ---------------- Phase A: S = scale * Q K^T ----------------
        float8_t sa[AV_NC];
#pragma unroll
        for (int c = 0; c < AV_NC; ++c) {
            sa[c] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }
        const _Float16 *krow[AV_NC];
#pragma unroll
        for (int c = 0; c < AV_NC; ++c) {
            const int gk = min(kt + c * 16 + h, L - 1);
            krow[c] = kbase + gk * kv_stride;
        }
#pragma unroll
        for (int dc = 0; dc < AV_DC; ++dc) {
            const int d0 = dc * 16 + hi * 8;
#pragma unroll
            for (int c = 0; c < AV_NC; ++c) {
                const half8_t breg = *(const half8_t *)(krow[c] + d0);
                sa[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    Qf[dc], breg, sa[c]);
            }
        }

        // ---------------- Phase B: online softmax, in registers -------------
        float cmask[AV_NC];
#pragma unroll
        for (int c = 0; c < AV_NC; ++c) {
            cmask[c] = ((kt + c * 16 + h) < L) ? 0.0f : -INFINITY;
        }
#pragma unroll
        for (int c = 0; c < AV_NC; ++c) {
#pragma unroll
            for (int j = 0; j < 8; ++j) sa[c][j] = sa[c][j] * scale + cmask[c];
        }

        float tm[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float mx = sa[0][j];
#pragma unroll
            for (int c = 1; c < AV_NC; ++c) mx = fmaxf(mx, sa[c][j]);
            tm[j] = mx;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            tm[j] = fmaxf(tm[j], __shfl_xor(tm[j], 1));
            tm[j] = fmaxf(tm[j], __shfl_xor(tm[j], 2));
            tm[j] = fmaxf(tm[j], __shfl_xor(tm[j], 4));
            tm[j] = fmaxf(tm[j], __shfl_xor(tm[j], 8));
        }

        float alpha[8], mnew[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            mnew[j] = fmaxf(m_run[j], tm[j]);
            alpha[j] =
                (m_run[j] == -INFINITY) ? 0.0f : __expf(m_run[j] - mnew[j]);
            m_run[j] = mnew[j];
        }

        float rs[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) rs[j] = 0.0f;
#pragma unroll
        for (int c = 0; c < AV_NC; ++c) {
            _Float16 *srow = Slds + ro * AV_S_STRIDE + c * 16 + h;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float e = __expf(sa[c][j] - mnew[j]);
                rs[j] += e;
                srow[j * AV_S_STRIDE] = (_Float16)e;
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            rs[j] += __shfl_xor(rs[j], 1);
            rs[j] += __shfl_xor(rs[j], 2);
            rs[j] += __shfl_xor(rs[j], 4);
            rs[j] += __shfl_xor(rs[j], 8);
            l_run[j] = alpha[j] * l_run[j] + rs[j];
        }
#pragma unroll
        for (int dc = 0; dc < AV_DC; ++dc) {
#pragma unroll
            for (int j = 0; j < 8; ++j) Of[dc][j] *= alpha[j];
        }

        // ---------------- Phase C: O += P V, V transposed in LDS ------------
#pragma unroll
        for (int vc = 0; vc < AV_NT / AV_VCHUNK; ++vc) {
            __syncthreads();
            {
                const int kk0 = wave * 8;
                half8_t col[4];
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int gk = min(kt + vc * AV_VCHUNK + kk0 + r, L - 1);
                    const _Float16 *vr = vbase + gk * kv_stride;
#pragma unroll
                    for (int i = 0; i < 4; ++i) col[i][r] = vr[i * 32 + lane];
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    half8_t *dst =
                        (half8_t *)(Vt + (i * 32 + lane) * AV_VT_STRIDE + kk0);
                    *dst = col[i];
                }
            }
            __syncthreads();

#pragma unroll
            for (int nl = 0; nl < AV_VNC; ++nl) {
                const int c = vc * AV_VNC + nl;
                const half8_t areg = *(const half8_t *)(
                    Slds + (rb + h) * AV_S_STRIDE + c * 16 + hi * 8);
#pragma unroll
                for (int dc = 0; dc < AV_DC; ++dc) {
                    const half8_t breg =
                        *(const half8_t *)(Vt + (dc * 16 + h) * AV_VT_STRIDE +
                                           nl * 16 + hi * 8);
                    Of[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        areg, breg, Of[dc]);
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float lv = l_run[j];
        l_run[j] = (lv > 0.0f) ? (1.0f / lv) : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int gq = q_start + ro + j;
        if (gq < B) {
            OT *orow = out + gq * q_stride + head * AV_HD + h;
#pragma unroll
            for (int dc = 0; dc < AV_DC; ++dc) {
                av_store<OT>::put(&orow[dc * 16], Of[dc][j] * l_run[j]);
            }
        }
    }
}

// One entry per {Q dtype} x {out dtype}; see the gfx11 sibling. Note this
// kernel takes no `qmajor` argument, so its kernarg layout is one int shorter.
#define AV_VT_GFX12_ENTRY(SUFFIX, QT, OT)                                    \
    extern "C" __global__ void __launch_bounds__(128, 2)                     \
        attention_flux_vt_wmma_f16kv_f32_gfx12##SUFFIX(                      \
            const QT *__restrict__ q, const _Float16 *__restrict__ k,        \
            const _Float16 *__restrict__ v, OT *__restrict__ out, int B,     \
            int L, int n_heads, int n_kv_heads, int head_dim, float scale) { \
        av_vt_gfx12_body<QT, OT>(q, k, v, out, B, L, n_heads, n_kv_heads,    \
                                 head_dim, scale);                           \
    }

AV_VT_GFX12_ENTRY(, float, float)
AV_VT_GFX12_ENTRY(_qf16, _Float16, float)
AV_VT_GFX12_ENTRY(_of16, float, _Float16)
AV_VT_GFX12_ENTRY(_qf16_of16, _Float16, _Float16)
