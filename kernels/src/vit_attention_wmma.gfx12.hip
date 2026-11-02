// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.
//
// Non-causal flash attention for the Qwen3.5/3.8 vision tower (head_dim 72,
// 16 heads, bidirectional, no mask), gfx12/RDNA4 wave32 WMMA.
//
// Derived from attention_flux_vt_wmma_f16kv_f32.gfx12.hip (the FLUX `vt` kernel,
// head_dim 128 hard-coded) because that body is already a working non-causal
// WMMA flash kernel with V transposed into LDS and register-resident softmax.
// Why a fork instead of a template: head_dim 72 is not a multiple of the
// 16-wide WMMA K/N tile, so the kernel works on head_dim zero-padded to
// AV_HD = 80 (5 WMMA chunks). Q/K/V arrive as F16 [N, n_heads, 80] produced by
// `vit_qkv_pack_f16` (zero in d in [head_dim, 80)), so padding contributes
// exactly 0 to QK^T and to the dropped output columns. Differences vs flux:
// AV_HD 80 / AV_DC 5, no GQA (n_kv_heads == n_heads), V staging guards the
// 80-column tail, F32 output only, written to out[gq*hidden + head*head_dim+d]
// for d < head_dim.
//
// Grid  = [n_heads, ceil(N/64)], block = [128] (4 waves),
// dynamic LDS (80*40 + 64*72)*2 = 15616 B.

// gfx12 fragment mapping (a[j]=A[l&15][8*(l>>4)+j], c[j]=C[8*(l>>4)+j][l&15])
// as in the flux gfx12 sibling. Compile-verified only; never run on RDNA4.
//
#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define AV_HD 80  // head_dim padded to a multiple of 16 (Qwen ViT: 72 -> 80)
#define AV_DC 5   // AV_HD / 16
#define AV_MT 64  // query rows per block
#define AV_NT 64  // keys per online-softmax tile
#define AV_NC 4   // AV_NT / 16
#define AV_VCHUNK 32
#define AV_VNC 2         // AV_VCHUNK / 16
#define AV_VT_STRIDE 40  // halves; 80 B, 16 B aligned
#define AV_S_STRIDE 72   // halves; 144 B, 16 B aligned
#define AV_VT_HALVES (AV_HD * AV_VT_STRIDE)
#define AV_LDS_BYTES ((AV_VT_HALVES + AV_MT * AV_S_STRIDE) * 2)  // 15616

__device__ __forceinline__ void
av_vit_gfx12_body(const _Float16 *__restrict__ q, const _Float16 *__restrict__ k,
       const _Float16 *__restrict__ v, float *__restrict__ out, int N,
       int n_heads, int head_dim, int hidden, float scale) {
    const int head = blockIdx.x;
    const int q_start = blockIdx.y * AV_MT;
    const int tid = threadIdx.x;

    if (head >= n_heads || q_start >= N) return;
    if (head_dim > AV_HD) return;

    const int kv_stride = n_heads * AV_HD;  // Qh/Kh/Vh: [N, n_heads, AV_HD]
    const int q_stride = kv_stride;

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int h = lane & 15;
    const int hi = lane >> 4;
    const int rb = wave * 16;
    const int ro = rb + hi * 8;  // this lane's first accumulator row

    extern __shared__ half8_t av_smem[];
    _Float16 *Vt = (_Float16 *)av_smem;
    _Float16 *Slds = Vt + AV_VT_HALVES;

    const _Float16 *kbase = k + head * AV_HD;
    const _Float16 *vbase = v + head * AV_HD;

    half8_t Qf[AV_DC];
    {
        const int gq = min(q_start + rb + h, N - 1);
        const _Float16 *qr = q + gq * q_stride + head * AV_HD;
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

    for (int kt = 0; kt < N; kt += AV_NT) {
        // ---------------- Phase A: S = scale * Q K^T ----------------
        float8_t sa[AV_NC];
#pragma unroll
        for (int c = 0; c < AV_NC; ++c) {
            sa[c] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }
        const _Float16 *krow[AV_NC];
#pragma unroll
        for (int c = 0; c < AV_NC; ++c) {
            const int gk = min(kt + c * 16 + h, N - 1);
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
            cmask[c] = ((kt + c * 16 + h) < N) ? 0.0f : -INFINITY;
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
                half8_t col[3];
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int gk = min(kt + vc * AV_VCHUNK + kk0 + r, N - 1);
                    const _Float16 *vr = vbase + gk * kv_stride;
#pragma unroll
                    for (int i = 0; i < 3; ++i)
                        col[i][r] = (i * 32 + lane < AV_HD) ? vr[i * 32 + lane] : (_Float16)0;
                }
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    if (i * 32 + lane >= AV_HD) continue;
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
        if (gq < N) {
            float *orow = out + gq * hidden + head * head_dim + h;
#pragma unroll
            for (int dc = 0; dc < AV_DC; ++dc) {
                if (dc * 16 + h < head_dim) orow[dc * 16] = Of[dc][j] * l_run[j];
            }
        }
    }
}

// Single F32-out entry. Args: (Qh, Kh, Vh, out, N, n_heads, head_dim, hidden,
// scale).
extern "C" __global__ void __launch_bounds__(128, 2)
    vit_attention_wmma_f16_gfx12(
        const _Float16 *__restrict__ q, const _Float16 *__restrict__ k,
        const _Float16 *__restrict__ v, float *__restrict__ out, int N,
        int n_heads, int head_dim, int hidden, float scale) {
    av_vit_gfx12_body(q, k, v, out, N, n_heads, head_dim, hidden, scale);
}
