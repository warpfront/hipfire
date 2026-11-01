// PM twin module (gfx1201): two per-arch kernel variants of generic ops, sealed by scripts/pm-hip-twin.sh.
//   gated_delta_net_vector_serial_f32_pm_gfx1201  <- card-judged R2Rda-05-rs-tv8-prio
//   causal_conv_depthwise_silu_rows_f32_pm_gfx1201 <- card-judged R1Conv-04-rpw32-k4
// Bodies are the judged candidates verbatim except the kernel symbol rename.
// hip_loop: rows_per_wg=8
// R2Rda-05-rs-tv8-prio
// idea: R2-01 plus s_setprio(3) around the pred butterfly so the serial chain wins issue slots over sibling waves' independent VALU
// prediction: 0.29-0.34
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Generic gated delta rule with per-head or per-key-channel log decay.
// One launch covers T>=1 tokens; the FP32 state tile stays in registers for the
// whole sequence (no per-token launch, no state tape, no atomics).
//
// Per token t and head h, state S[v,k] (value-major, key contiguous):
//   D[k]   = exp(log_decay[t,h,k])            (PerKeyChannel) or exp(log_decay[t,h]) (PerHead)
//   S[v,k] = S[v,k] * D[k]
//   pred[v]= sum_k S[v,k] * key[k]
//   err[v] = value[v] - pred[v]
//   S[v,k] = S[v,k] + beta * (err[v] * key[k])
//   y[v]   = sum_k S[v,k] * query[k]           (read AFTER write)
// The decay is applied to each key column BEFORE the reduction, so a vector
// decay is never factored out of the dot product.
//
// Geometry: head_dim (K) and value dim (V) are both 128 in this build. One
// wave32 workgroup owns (head, V-tile of GDV_TV rows); each lane owns four key
// columns (lane + 32*i) of every row in the tile. Reductions are xor
// butterflies, so every lane holds the bit-identical sum.
//
// Slot/token-count device indirection (graph-capture safe): when `slot_idx` /
// `n_dev` are non-null they override the baked `slot` / `n_tokens` (the baked
// n_tokens is then the launch upper bound).

#include <hip/hip_runtime.h>

#define GDV_D 128
#define GDV_TV 8
#define GDV_LANES 32
#define GDV_KPL (GDV_D / GDV_LANES)  // key columns per lane = 4

static __device__ __forceinline__ float gdv_bf16_rn(float x) {
    unsigned u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;  // NaN passes through
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

// Same butterfly as __shfl_xor 16,8,4,2,1 (x + partner, commutative => bit-identical),
// lowered to permlanex16 + DPP row_xmask instead of ds_bpermute.
template <int CTRL>
static __device__ __forceinline__ float gdv_xor_dpp(float x) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), CTRL, 0xf, 0xf, false));
}
static __device__ __forceinline__ float gdv_wave_sum(float x) {
    x += __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x),
                                                      0x76543210u, 0xfedcba98u, false, false));
    x += gdv_xor_dpp<0x168>(x);  // row_xmask:8
    x += gdv_xor_dpp<0x164>(x);  // row_xmask:4
    x += gdv_xor_dpp<0x162>(x);  // row_xmask:2
    x += gdv_xor_dpp<0x161>(x);  // row_xmask:1
    return x;
}


static __device__ __forceinline__ void gdv_wave_sum_tile(float (&a)[GDV_TV]) {
#pragma unroll
    for (int r = 0; r < GDV_TV; r++)
        a[r] += __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(a[r]), __float_as_int(a[r]),
                                                            0x76543210u, 0xfedcba98u, false, false));
#pragma unroll
    for (int r = 0; r < GDV_TV; r++) a[r] += gdv_xor_dpp<0x168>(a[r]);
#pragma unroll
    for (int r = 0; r < GDV_TV; r++) a[r] += gdv_xor_dpp<0x164>(a[r]);
#pragma unroll
    for (int r = 0; r < GDV_TV; r++) a[r] += gdv_xor_dpp<0x162>(a[r]);
#pragma unroll
    for (int r = 0; r < GDV_TV; r++) a[r] += gdv_xor_dpp<0x161>(a[r]);
}

extern "C" __launch_bounds__(GDV_LANES, 8)
__global__ void gated_delta_net_vector_serial_f32_pm_gfx1201(
    const float* __restrict__ query,       // [T, Hqk, K] row stride q_row_stride
    const float* __restrict__ key,         // [T, Hqk, K] row stride k_row_stride
    const float* __restrict__ value,       // [T, H,   V] row stride v_row_stride
    const float* __restrict__ log_decay,   // PerKey: [T,H,K]; PerHead: [T,H]
    const float* __restrict__ beta,        // [T, H]       row stride beta_row_stride
    float* __restrict__ state,             // slot base; [H,V,K] via head/row strides
    float* __restrict__ output,            // [T, H, V]    row stride out_row_stride
    const int* __restrict__ slot_idx,      // nullable device slot index
    const int* __restrict__ n_dev,         // nullable device token count
    int n_tokens,
    int n_heads,
    int qk_group,                          // value heads per q/k head (1 = no repeat)
    int decay_per_key,                     // 1 = PerKeyChannel, 0 = PerHead
    int round_out_bf16,
    int slot,
    int q_row_stride,
    int k_row_stride,
    int v_row_stride,
    int decay_row_stride,
    int beta_row_stride,
    int out_row_stride,
    int state_head_stride,
    int state_row_stride,
    unsigned long long state_slot_stride
) {
    const int h = blockIdx.x;
    const int row0 = blockIdx.y * GDV_TV;
    if (h >= n_heads) return;
    const int lane = threadIdx.x;
    const int hq = h / qk_group;

    int n = n_tokens;
    if (n_dev != nullptr) {
        int nd = n_dev[0];
        n = nd < n ? nd : n;
    }
    const int slot_sel = (slot_idx != nullptr) ? slot_idx[0] : slot;
    float* S_base = state + (unsigned long long)slot_sel * state_slot_stride
                  + (long long)h * state_head_stride + (long long)row0 * state_row_stride;

    float S[GDV_TV][GDV_KPL];
#pragma unroll
    for (int r = 0; r < GDV_TV; r++) {
#pragma unroll
        for (int i = 0; i < GDV_KPL; i++)
            S[r][i] = S_base[(long long)r * state_row_stride + lane + GDV_LANES * i];
    }

    // Operands for the current token; the next token's are fetched while the
    // current one computes.
    float kc[GDV_KPL], qc[GDV_KPL], dc[GDV_KPL], vc[GDV_TV];
    float bc = 0.0f;
    if (n > 0) {
        const float* kp = key + hq * GDV_D;
        const float* qp = query + hq * GDV_D;
#pragma unroll
        for (int i = 0; i < GDV_KPL; i++) {
            kc[i] = kp[lane + GDV_LANES * i];
            qc[i] = qp[lane + GDV_LANES * i];
        }
        if (decay_per_key) {
            const float* dp = log_decay + (long long)h * GDV_D;
#pragma unroll
            for (int i = 0; i < GDV_KPL; i++) dc[i] = expf(dp[lane + GDV_LANES * i]);
        } else {
            float d = expf(log_decay[h]);
#pragma unroll
            for (int i = 0; i < GDV_KPL; i++) dc[i] = d;
        }
        const float* vp = value + h * GDV_D + row0;
#pragma unroll
        for (int r = 0; r < GDV_TV; r++) vc[r] = vp[r];
        bc = beta[h];
    }

    for (int t = 0; t < n; t++) {
        float kn[GDV_KPL], qn[GDV_KPL], dn[GDV_KPL], vn[GDV_TV];
        float bn = 0.0f;
        const bool has_next = (t + 1) < n;
        if (has_next) {
            const int t1 = t + 1;
            const float* kp = key + (long long)t1 * k_row_stride + hq * GDV_D;
            const float* qp = query + (long long)t1 * q_row_stride + hq * GDV_D;
#pragma unroll
            for (int i = 0; i < GDV_KPL; i++) {
                kn[i] = kp[lane + GDV_LANES * i];
                qn[i] = qp[lane + GDV_LANES * i];
            }
            if (decay_per_key) {
                const float* dp = log_decay + (long long)t1 * decay_row_stride + (long long)h * GDV_D;
#pragma unroll
                for (int i = 0; i < GDV_KPL; i++) dn[i] = expf(dp[lane + GDV_LANES * i]);
            } else {
                float d = expf(log_decay[(long long)t1 * decay_row_stride + h]);
#pragma unroll
                for (int i = 0; i < GDV_KPL; i++) dn[i] = d;
            }
            const float* vp = value + (long long)t1 * v_row_stride + h * GDV_D + row0;
#pragma unroll
            for (int r = 0; r < GDV_TV; r++) vn[r] = vp[r];
            bn = beta[(long long)t1 * beta_row_stride + h];
        }

        float* out_row = output + (long long)t * out_row_stride + h * GDV_D + row0;
        // Same per-row arithmetic; the TV row reductions run stage-by-stage in lockstep.
        float acc[GDV_TV];
#pragma unroll
        for (int r = 0; r < GDV_TV; r++) {
            float s0 = S[r][0] * dc[0];
            float s1 = S[r][1] * dc[1];
            float s2 = S[r][2] * dc[2];
            float s3 = S[r][3] * dc[3];
            S[r][0] = s0; S[r][1] = s1; S[r][2] = s2; S[r][3] = s3;
            acc[r] = s0 * kc[0] + s1 * kc[1] + s2 * kc[2] + s3 * kc[3];
        }
        __builtin_amdgcn_s_setprio(3);
        gdv_wave_sum_tile(acc);
        __builtin_amdgcn_s_setprio(0);
#pragma unroll
        for (int r = 0; r < GDV_TV; r++) {
            float s0 = S[r][0], s1 = S[r][1], s2 = S[r][2], s3 = S[r][3];
            float err = vc[r] - acc[r];
            s0 = s0 + bc * (err * kc[0]);
            s1 = s1 + bc * (err * kc[1]);
            s2 = s2 + bc * (err * kc[2]);
            s3 = s3 + bc * (err * kc[3]);
            S[r][0] = s0; S[r][1] = s1; S[r][2] = s2; S[r][3] = s3;
            acc[r] = s0 * qc[0] + s1 * qc[1] + s2 * qc[2] + s3 * qc[3];
        }
        float rs0[8];
#pragma unroll
        for (int r = 0; r < 8; r++) rs0[r] = acc[r];
        float rs1[4];
        { const bool hb = (lane & 16) != 0;
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const float a = rs0[j], b = rs0[j + 4];
            const float kp = hb ? b : a;
            const float sd = hb ? a : b;
            rs1[j] = kp + __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(sd), __float_as_int(sd), 0x76543210u, 0xfedcba98u, false, false));
          } }
        float rs2[2];
        { const bool hb = (lane & 8) != 0;
#pragma unroll
          for (int j = 0; j < 2; j++) {
            const float a = rs1[j], b = rs1[j + 2];
            const float kp = hb ? b : a;
            const float sd = hb ? a : b;
            rs2[j] = kp + gdv_xor_dpp<0x168>(sd);
          } }
        float rs3[1];
        { const bool hb = (lane & 4) != 0;
#pragma unroll
          for (int j = 0; j < 1; j++) {
            const float a = rs2[j], b = rs2[j + 1];
            const float kp = hb ? b : a;
            const float sd = hb ? a : b;
            rs3[j] = kp + gdv_xor_dpp<0x164>(sd);
          } }
        float yv = rs3[0];
        yv += gdv_xor_dpp<0x162>(yv);
        yv += gdv_xor_dpp<0x161>(yv);
        const int yrow = (((lane >> 4) & 1) << 2) | (((lane >> 3) & 1) << 1) | (((lane >> 2) & 1) << 0);
        if ((lane & 3) == 0) out_row[yrow] = round_out_bf16 ? gdv_bf16_rn(yv) : yv;

        if (has_next) {
#pragma unroll
            for (int i = 0; i < GDV_KPL; i++) { kc[i] = kn[i]; qc[i] = qn[i]; dc[i] = dn[i]; }
#pragma unroll
            for (int r = 0; r < GDV_TV; r++) vc[r] = vn[r];
            bc = bn;
        }
    }

#pragma unroll
    for (int r = 0; r < GDV_TV; r++) {
#pragma unroll
        for (int i = 0; i < GDV_KPL; i++)
            S_base[(long long)r * state_row_stride + lane + GDV_LANES * i] = S[r][i];
    }
}

// ---- causal_conv_depthwise_silu_rows_f32_pm_gfx1201 ----
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Depthwise causal short convolution over N rows with an explicit raw-input
// history, optional SiLU and optional single BF16 rounding of the output.
//
//   padded = [hist[0..K-2] (oldest -> newest), in[0..N-1]]
//   out[r,c] = act( sum_j w[c,j] * padded[r+j, c] )      j = 0 (oldest) .. K-1 (newest)
//   hist_out[j,c] = padded[N+j, c]                        (raw, pre-conv values)
//
// FP32 accumulation in tap order oldest -> newest, no bias. `hist_in` and
// `hist_out` may alias only when the launch uses a single row chunk
// (rows_per_chunk >= n_rows); the launcher enforces that.
// Weights are [channel, tap] with explicit strides. K <= 8.

#include <hip/hip_runtime.h>

#define CCD_KMAX 8

static __device__ __forceinline__ float ccd_bf16_rn(float x) {
    unsigned u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}


// PM candidate: rows are tiled across workgroups (CCD_RPW rows per workgroup, grid y = ceil(N/CCD_RPW))
// regardless of the rows_per_chunk kernarg. Only workgroup row-chunk 0 reads hist_in, and that
// same chunk writes hist_out after its own reads (per channel, same thread), so the in-place
// history (hist_in == hist_out) has a single reader/writer. Per-output arithmetic is the incumbent's.
#define CCD_RPW 32
extern "C" __global__ void causal_conv_depthwise_silu_rows_f32_pm_gfx1201(
    float* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ weight,
    const float* hist_in,
    float* hist_out,
    int n_rows,
    int n_channels,
    int ksize,
    int rows_per_chunk,
    int in_row_stride,
    int out_row_stride,
    int w_ch_stride,
    int w_tap_stride,
    int hist_tap_stride,
    int apply_silu,
    int round_bf16
) {
    (void)rows_per_chunk;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;
    const int km1 = ksize - 1;
    const int r0 = blockIdx.y * CCD_RPW;
    if (r0 >= n_rows) return;
    int r1 = r0 + CCD_RPW;
    if (r1 > n_rows) r1 = n_rows;

    if (ksize == 4) {
        const float w0 = weight[(long long)c * w_ch_stride];
        const float w1 = weight[(long long)c * w_ch_stride + (long long)w_tap_stride];
        const float w2 = weight[(long long)c * w_ch_stride + 2LL * w_tap_stride];
        const float w3 = weight[(long long)c * w_ch_stride + 3LL * w_tap_stride];
        float a0, a1, a2;
        {
            int p = r0;
            a0 = (p < 3) ? hist_in[(long long)p * hist_tap_stride + c] : in[(long long)(p - 3) * in_row_stride + c];
            p = r0 + 1;
            a1 = (p < 3) ? hist_in[(long long)p * hist_tap_stride + c] : in[(long long)(p - 3) * in_row_stride + c];
            p = r0 + 2;
            a2 = (p < 3) ? hist_in[(long long)p * hist_tap_stride + c] : in[(long long)(p - 3) * in_row_stride + c];
        }
        for (int r = r0; r < r1; r++) {
            const float x = in[(long long)r * in_row_stride + c];
            float acc = 0.0f;
            acc = __fadd_rn(acc, __fmul_rn(w0, a0));
            acc = __fadd_rn(acc, __fmul_rn(w1, a1));
            acc = __fadd_rn(acc, __fmul_rn(w2, a2));
            acc = __fadd_rn(acc, __fmul_rn(w3, x));
            float y = apply_silu ? acc / (1.0f + expf(-acc)) : acc;
            out[(long long)r * out_row_stride + c] = round_bf16 ? ccd_bf16_rn(y) : y;
            a0 = a1; a1 = a2; a2 = x;
        }
        if (r0 == 0) {
            for (int j = 0; j < 3; j++) {
                const int p = n_rows + j;
                float v = (p < 3) ? hist_in[(long long)p * hist_tap_stride + c]
                                  : in[(long long)(p - 3) * in_row_stride + c];
                hist_out[(long long)j * hist_tap_stride + c] = v;
            }
        }
        return;
    }

    float w[CCD_KMAX];
    float win[CCD_KMAX];
#pragma unroll
    for (int j = 0; j < CCD_KMAX; j++) {
        w[j] = (j < ksize) ? weight[(long long)c * w_ch_stride + (long long)j * w_tap_stride] : 0.0f;
        float v = 0.0f;
        if (j < km1) {
            const int p = r0 + j;
            v = (p < km1) ? hist_in[(long long)p * hist_tap_stride + c]
                          : in[(long long)(p - km1) * in_row_stride + c];
        }
        win[j] = v;
    }

    for (int r = r0; r < r1; r++) {
        const float x = in[(long long)r * in_row_stride + c];
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < CCD_KMAX - 1; j++) {
            if (j < km1) acc = __fadd_rn(acc, __fmul_rn(w[j], win[j]));
        }
        acc = __fadd_rn(acc, __fmul_rn(w[km1], x));
        float y = apply_silu ? acc / (1.0f + expf(-acc)) : acc;
        out[(long long)r * out_row_stride + c] = round_bf16 ? ccd_bf16_rn(y) : y;
#pragma unroll
        for (int j = 0; j < CCD_KMAX - 2; j++) {
            if (j + 1 < km1) win[j] = win[j + 1];
        }
        win[km1 - 1] = x;
    }

    if (r0 == 0) {
        for (int j = 0; j < km1; j++) {
            const int p = n_rows + j;
            float v = (p < km1) ? hist_in[(long long)p * hist_tap_stride + c]
                                : in[(long long)(p - km1) * in_row_stride + c];
            hist_out[(long long)j * hist_tap_stride + c] = v;
        }
    }
}
