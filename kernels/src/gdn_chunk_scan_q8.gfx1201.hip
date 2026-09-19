// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Chunked-scan Q8 gated DeltaNet for gfx1201 prefill (default OFF,
// `HIPFIRE_GFX12_GDN_CHUNKED=1`).
//
// Shape: one workgroup per (head, value-row half). The 64x128 f32 state
// half-tile stays resident in LDS for the whole segment; the token axis is
// looped on-device in chunks of CS<=32 (serial across chunks, S carried in
// LDS). Within a chunk the FLA decomposition replaces the 512-deep
// per-token dependent chain:
//
//   G_i      = sum_{m<=i} g_m                       (INCLUSIVE cumsum)
//   D[i,j]   = exp(G_i - G_j)                       (pairwise decay, on the fly)
//   w_j      = exp(G_j) * (S_in @ k_j)              (inter-chunk pull, f32 VALU)
//   A[j,l]   = exp(G_j-G_l)*(k_j.k_l), l<j; L=diag(beta)*A   (KK Gram)
//   rhs      = beta o (v - w)
//   (I + L) delta = rhs                             (unit-lower fwd-subst)
//   out_i    = exp(G_i)*(S_in @ q_i)
//              + sum_{j<=i} exp(G_i-G_j)*(q_i.k_j) * delta_j (QK Gram)
//   S_out    = exp(G_C)*S_in + sum_j decay[j]*outer(delta_j,k_j)  (carry)
//
// The two C×C Grams (KK, QK) read pure activations and run on WMMA (f16 in,
// f32 accum, gfx12 wave32) when `HIPFIRE_GDN_SCAN_WMMA != 0` (default 1).
// Everything that touches S (S@k, S@q, T/M applies, carry) stays f32 VALU:
// S is the object the epilogue quantises, so f16 on it would be a
// state-precision change, not an optimisation.
// `HIPFIRE_GDN_SCAN_WMMA=0` recompiles both Grams in f32 VALU from the same
// f16 tiles (A/B vehicle for the WMMA leg).
//
// The 512 commit boundary is unchanged: exactly one EF-fold + absmax +
// scale=max/127 + rint requant per row at segment end, verbatim from
// `gated_delta_net_q8_fast.hip`. Non-EF callers must stay on the serial
// kernel (stochastic seed interacts with any f32 trajectory change).
//
// LDS budget (CS_MAX=32, S_ROWS=64, HD=128):
//   S        64*128 f32          = 32768 B
//   Kf + Qf  2 * 32*128 f16      = 16384 B
//   T/M      32*32 f32 (shared)  =  4096 B
//   Dlt      32*64 f32           =  8192 B
//   G/expG/decay/beta 4*32 f32   =   512 B
//   total = 61952 B < 64 KB/workgroup.
//
// Threading: block 256 = 8 waves of 32. Waves 0-3 own one 16x16 Gram tile
// each (KK tiles (0,0),(0,1),(1,0),(1,1), then QK likewise); waves 4-7 do
// the f32 S@k / S@q dots meanwhile. The solve/apply/carry/out phases use
// all waves strided. Commit: each wave owns 8 rows with full-wave
// cooperation per row (same __shfl_xor tree as the serial kernel).

#ifndef HIPFIRE_GDN_SCAN_WMMA
#define HIPFIRE_GDN_SCAN_WMMA 1
#endif
typedef _Float16 __attribute__((ext_vector_type(8))) scan_half8_t;
typedef float __attribute__((ext_vector_type(8))) scan_float8_t;

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define SCAN_HD 128
#define SCAN_CS_MAX 32
#define SCAN_S_ROWS 64
#define SCAN_R_SPLITS 2

#ifndef HIPFIRE_GDN_QK_HEAD_DIV
#define HIPFIRE_GDN_QK_HEAD_DIV 1
#endif

#ifndef HIPFIRE_GDN_SCAN_KERNEL
#define HIPFIRE_GDN_SCAN_KERNEL gated_delta_net_q8_scan_gfx1201
#endif

typedef _Float16 __attribute__((ext_vector_type(8))) scan_half8_t;
typedef float __attribute__((ext_vector_type(8))) scan_float8_t;

__shared__ float S_tile[SCAN_S_ROWS * SCAN_HD];
__shared__ _Float16 Kf_tile[SCAN_CS_MAX * SCAN_HD];
__shared__ _Float16 Qf_tile[SCAN_CS_MAX * SCAN_HD];
__shared__ float TM_buf[SCAN_CS_MAX * SCAN_CS_MAX];
__shared__ float Dlt_buf[SCAN_CS_MAX * SCAN_S_ROWS];
__shared__ float Gv_buf[SCAN_CS_MAX];
__shared__ float ExpG_buf[SCAN_CS_MAX];
__shared__ float Decay_buf[SCAN_CS_MAX];
__shared__ float Beta_buf[SCAN_CS_MAX];

extern "C" __launch_bounds__(256, 1)
__global__ void HIPFIRE_GDN_SCAN_KERNEL(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    signed char* __restrict__ s_q8,
    float* __restrict__ s_scales,
    float* __restrict__ output,
    int n_tokens,
    int n_heads,
    int head_dim,
    int frame_unused,
    __half* __restrict__ s_ef_residual,
    int chunk_size
) {
    (void)head_dim;
    (void)frame_unused;
    const int h = blockIdx.x;
    const int split = blockIdx.y;
    if (h >= n_heads || split >= SCAN_R_SPLITS) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int hid = lane >> 4;
    const int half = lane & 15;
    const int stride = n_heads * SCAN_HD;
    const int qk_heads = n_heads / HIPFIRE_GDN_QK_HEAD_DIV;
    const int qk_stride = qk_heads * SCAN_HD;
    const int qk_head = h / HIPFIRE_GDN_QK_HEAD_DIV;
    const int r_base = split * SCAN_S_ROWS;

    int CS = chunk_size;
    if (CS < 1) CS = 1;
    if (CS > SCAN_CS_MAX) CS = SCAN_CS_MAX;

    // ── S load: dequant scale*code, EF folded only at commit (verbatim). ──
    {
        const size_t state_lane_base = 0;
        (void)state_lane_base;
        signed char* sq_base = s_q8 + h * SCAN_HD * SCAN_HD + r_base * SCAN_HD;
        for (int idx = tid; idx < SCAN_S_ROWS * SCAN_HD; idx += 256) {
            int r = idx / SCAN_HD;
            int c = idx - r * SCAN_HD;
            float sc = s_scales[h * SCAN_HD + r_base + r];
            S_tile[idx] = sc * (float)sq_base[r * SCAN_HD + c];
        }
    }

    // Chunk loop — serial across chunks, S carried in LDS.
    for (int c0 = 0; c0 < n_tokens; c0 += CS) {
        int C = n_tokens - c0;
        if (C > CS) C = CS;

        // ── 0. gate/beta scalars + Kf/Qf f16 tiles. ──
        if (lane < C) {
            int bt = c0 + lane;
            Gv_buf[lane] = gate[bt * n_heads + h];
            Beta_buf[lane] = beta[bt * n_heads + h];
        }
        {
            const float* k_base = k + c0 * qk_stride + qk_head * SCAN_HD;
            const float* q_base = q + c0 * qk_stride + qk_head * SCAN_HD;
            for (int idx = tid; idx < SCAN_CS_MAX * SCAN_HD; idx += 256) {
                int jr = idx / SCAN_HD;
                int c = idx - jr * SCAN_HD;
                float kv = (jr < C) ? k_base[jr * qk_stride + c] : 0.0f;
                float qv = (jr < C) ? q_base[jr * qk_stride + c] : 0.0f;
                Kf_tile[idx] = (_Float16)kv;
                Qf_tile[idx] = (_Float16)qv;
            }
        }
        __syncthreads();

        // ── 1. inclusive cumsum G (wave 0, Kogge-Stone __shfl_up) + exps. ──
        if (wave == 0) {
            float val = (lane < C) ? Gv_buf[lane] : 0.0f;
            for (int d = 1; d < 32; d <<= 1) {
                float up = __shfl_up(val, d);
                if (lane >= d) val += up;
            }
            if (lane < C) Gv_buf[lane] = val;
        }
        __syncthreads();
        if (tid < C) {
            float g = Gv_buf[tid];
            ExpG_buf[tid] = expf(g);
        }
        __syncthreads();
        {
            float gc = Gv_buf[C - 1];
            if (tid < C) Decay_buf[tid] = expf(gc - Gv_buf[tid]);
        }
        __syncthreads();

        // ── 2. KK Gram -> L in TM_buf (waves 0-3) AND S@k -> Dlt (waves 4-7).
        if (wave < 4) {
            int tr = (wave >> 1) & 1;
            int tc = wave & 1;
#if HIPFIRE_GDN_SCAN_WMMA
                // One cooperative 16x16x16 product per (tr,tc) tile: every
                // lane feeds A row `half` / B col `half`, K-frag hid*8; the
                // hardware contracts K across halves. Lane (hid,half) then
                // owns D rows hid*8..+8 of column `half`.
                scan_float8_t acc = {0,0,0,0,0,0,0,0};
                for (int d0 = 0; d0 < SCAN_HD; d0 += 16) {
                    int dk = d0 + hid * 8;
                    scan_half8_t a_reg, b_reg;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        a_reg[j] = Kf_tile[(tr * 16 + half) * SCAN_HD + dk + j];
                        b_reg[j] = Kf_tile[(tc * 16 + half) * SCAN_HD + dk + j];
                    }
                    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_reg, acc);
                }
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int rr = tr * 16 + hid * 8 + j;
                    int cc = tc * 16 + half;
                    if (rr < SCAN_CS_MAX && cc < SCAN_CS_MAX)
                        TM_buf[rr * SCAN_CS_MAX + cc] =
                            (rr < C && cc < C && cc < rr)
                            ? Beta_buf[rr] * expf(Gv_buf[rr] - Gv_buf[cc]) * acc[j]
                            : 0.0f;
                }
#else
                // f32 VALU Gram from the same f16 tiles (A/B vehicle).
                for (int rr = lane; rr < 16; rr += 32) {
                    int gr = tr * 16 + rr;
                    for (int cc = 0; cc < 16; cc++) {
                        int gc2 = tc * 16 + cc;
                        if (gr < SCAN_CS_MAX && gc2 < SCAN_CS_MAX) {
                            float e = 0.0f;
                            if (gr < C && gc2 < C && gc2 < gr) {
                                float kk = 0.0f;
                                #pragma unroll 4
                                for (int d = 0; d < SCAN_HD; d++)
                                    kk += (float)Kf_tile[gr * SCAN_HD + d] * (float)Kf_tile[gc2 * SCAN_HD + d];
                                e = Beta_buf[gr] * expf(Gv_buf[gr] - Gv_buf[gc2]) * kk;
                            }
                            TM_buf[gr * SCAN_CS_MAX + gc2] = e;
                        }
                    }
                }
#endif
        } else {
            // S@k dots into Dlt (w-phase): Dlt[j,r] = expG_j * (S_r . k_j).
            for (int k = tid - 128; k < C * SCAN_S_ROWS; k += 128) {
                int j = k / SCAN_S_ROWS;
                int r = k - j * SCAN_S_ROWS;
                float acc = 0.0f;
                #pragma unroll 4
                for (int c = 0; c < SCAN_HD; c++)
                    acc += S_tile[r * SCAN_HD + c] * (float)Kf_tile[j * SCAN_HD + c];
                Dlt_buf[j * SCAN_S_ROWS + r] = ExpG_buf[j] * acc;
            }
        }
        __syncthreads();

        // ── 3. invert (I+L) in place in TM_buf (wave 0; C<=32, tiny). ──
        if (wave == 0 && lane < 32) {
            for (int i = 0; i < C; i++) {
                // T[i,j] = -sum_{l=j}^{i-1} L[i,l]*T[l,j] (j<i); T[i,i]=1.
                // Row i of L is intact (only rows <i rewritten so far).
                if (lane < i) {
                    float acc = 0.0f;
                    for (int l = lane; l < i; l++)
                        acc += TM_buf[i * SCAN_CS_MAX + l] * TM_buf[l * SCAN_CS_MAX + lane];
                    TM_buf[i * SCAN_CS_MAX + lane] = -acc;
                } else if (lane == i) {
                    TM_buf[i * SCAN_CS_MAX + lane] = 1.0f;
                }
                // lanes > i idle this row; sync each row (wave-level).
                __syncthreads();
            }
        }
        __syncthreads();

        // ── 4. rhs = beta o (v - w), then fwd-subst in place in Dlt. ──
        {
            const float* v_base = v + c0 * stride + h * SCAN_HD + r_base;
            for (int k = tid; k < C * SCAN_S_ROWS; k += 256) {
                int j = k / SCAN_S_ROWS;
                int r = k - j * SCAN_S_ROWS;
                float vjr = v_base[j * stride + r];
                Dlt_buf[k] = Beta_buf[j] * (vjr - Dlt_buf[k]);
            }
        }
        __syncthreads();
        for (int i = 0; i < C; i++) {
            for (int r = tid; r < SCAN_S_ROWS; r += 256) {
                float acc = Dlt_buf[i * SCAN_S_ROWS + r];
                for (int l = 0; l < i; l++)
                    acc -= TM_buf[i * SCAN_CS_MAX + l] * Dlt_buf[l * SCAN_S_ROWS + r];
                Dlt_buf[i * SCAN_S_ROWS + r] = acc;
            }
            __syncthreads();
        }

        // ── 5. QK Gram -> M in TM_buf (waves 0-3) AND S@q folded into out
        // phase directly (waves 4-7 precompute sq into... reuse Dlt? No:
        // Dlt holds delta. Compute S@q on the fly in step 6 per (i,r) —
        // it re-reads S tile (LDS-resident, cheap) — actually better:
        // waves 4-7 stage sq[32][64]... no room. Do it in step 6 inline.
        if (wave < 4) {
            int tr = (wave >> 1) & 1;
            int tc = wave & 1;
#if HIPFIRE_GDN_SCAN_WMMA
                scan_float8_t acc = {0,0,0,0,0,0,0,0};
                for (int d0 = 0; d0 < SCAN_HD; d0 += 16) {
                    int dk = d0 + hid * 8;
                    scan_half8_t a_reg, b_reg;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        a_reg[j] = Qf_tile[(tr * 16 + half) * SCAN_HD + dk + j];
                        b_reg[j] = Kf_tile[(tc * 16 + half) * SCAN_HD + dk + j];
                    }
                    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_reg, acc);
                }
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int rr = tr * 16 + hid * 8 + j;
                    int cc = tc * 16 + half;
                    if (rr < SCAN_CS_MAX && cc < SCAN_CS_MAX)
                        TM_buf[rr * SCAN_CS_MAX + cc] =
                            (rr < C && cc < C && cc <= rr)
                            ? expf(Gv_buf[rr] - Gv_buf[cc]) * acc[j] : 0.0f;
                }
#else
                // f32 VALU Gram from the same f16 tiles (A/B vehicle).
                for (int rr = lane; rr < 16; rr += 32) {
                    int gr = tr * 16 + rr;
                    for (int cc = 0; cc < 16; cc++) {
                        int gc2 = tc * 16 + cc;
                        if (gr < SCAN_CS_MAX && gc2 < SCAN_CS_MAX) {
                            float e = 0.0f;
                            if (gr < C && gc2 < C && gc2 <= gr) {
                                float qk = 0.0f;
                                #pragma unroll 4
                                for (int d = 0; d < SCAN_HD; d++)
                                    qk += (float)Qf_tile[gr * SCAN_HD + d] * (float)Kf_tile[gc2 * SCAN_HD + d];
                                e = expf(Gv_buf[gr] - Gv_buf[gc2]) * qk;
                            }
                            TM_buf[gr * SCAN_CS_MAX + gc2] = e;
                        }
                    }
                }
#endif
        }
        __syncthreads();

        // ── 6. out_i[r] = expG_i*(S_r.q_i) + sum_{j<=i} M[i,j]*delta[j,r]. ──
        {
            float* out_base = output + c0 * stride + h * SCAN_HD + r_base;
            for (int k = tid; k < C * SCAN_S_ROWS; k += 256) {
                int i = k / SCAN_S_ROWS;
                int r = k - i * SCAN_S_ROWS;
                float sq = 0.0f;
                #pragma unroll 4
                for (int c = 0; c < SCAN_HD; c++)
                    sq += S_tile[r * SCAN_HD + c] * (float)Qf_tile[i * SCAN_HD + c];
                float acc = ExpG_buf[i] * sq;
                for (int j = 0; j <= i; j++)
                    acc += TM_buf[i * SCAN_CS_MAX + j] * Dlt_buf[j * SCAN_S_ROWS + r];
                out_base[i * stride + r] = acc;
            }
        }
        __syncthreads();

        // ── 7. carry S = expGC*S + sum_j decay_j*delta_j (x) k_j. ──
        {
            float expGC = expf(Gv_buf[C - 1]);
            for (int idx = tid; idx < SCAN_S_ROWS * SCAN_HD; idx += 256) {
                int r = idx / SCAN_HD;
                int c = idx - r * SCAN_HD;
                float acc = expGC * S_tile[idx];
                for (int j = 0; j < C; j++)
                    acc += Decay_buf[j] * Dlt_buf[j * SCAN_S_ROWS + r]
                         * (float)Kf_tile[j * SCAN_HD + c];
                S_tile[idx] = acc;
            }
        }
        __syncthreads();
    }

    // ── Commit (verbatim EF epilogue): one requant per row, EF folded once. ──
    {
        signed char* sq_base = s_q8 + h * SCAN_HD * SCAN_HD + r_base * SCAN_HD;
        __half* efr_base = s_ef_residual + h * SCAN_HD * SCAN_HD + r_base * SCAN_HD;
        // 8 rows per wave, full-wave cooperation per row.
        for (int rr = 0; rr < 8; rr++) {
            int r = wave * 8 + rr;
            float s0 = S_tile[r * SCAN_HD + lane];
            float s1 = S_tile[r * SCAN_HD + lane + 32];
            float s2 = S_tile[r * SCAN_HD + lane + 64];
            float s3 = S_tile[r * SCAN_HD + lane + 96];
            __half* efr = efr_base + r * SCAN_HD;
            s0 += __half2float(efr[lane]);
            s1 += __half2float(efr[lane + 32]);
            s2 += __half2float(efr[lane + 64]);
            s3 += __half2float(efr[lane + 96]);

            float my_max = fmaxf(fmaxf(fabsf(s0), fabsf(s1)), fmaxf(fabsf(s2), fabsf(s3)));
            for (int o = 16; o > 0; o >>= 1)
                my_max = fmaxf(my_max, __shfl_xor(my_max, o));

            float inv_s = (my_max > 0.0f) ? (127.0f / my_max) : 0.0f;
            float scale = (my_max > 0.0f) ? (my_max / 127.0f) : 1.0f;
            float qf0 = fminf(fmaxf(rintf(s0 * inv_s), -128.0f), 127.0f);
            float qf1 = fminf(fmaxf(rintf(s1 * inv_s), -128.0f), 127.0f);
            float qf2 = fminf(fmaxf(rintf(s2 * inv_s), -128.0f), 127.0f);
            float qf3 = fminf(fmaxf(rintf(s3 * inv_s), -128.0f), 127.0f);
            efr[lane]      = __float2half(s0 - qf0 * scale);
            efr[lane + 32] = __float2half(s1 - qf1 * scale);
            efr[lane + 64] = __float2half(s2 - qf2 * scale);
            efr[lane + 96] = __float2half(s3 - qf3 * scale);
            signed char* dst = sq_base + r * SCAN_HD;
            dst[lane]      = (signed char)qf0;
            dst[lane + 32] = (signed char)qf1;
            dst[lane + 64] = (signed char)qf2;
            dst[lane + 96] = (signed char)qf3;
            if (lane == 0) s_scales[h * SCAN_HD + r_base + r] = scale;
        }
    }
}
