// SPDX-License-Identifier: Apache-2.0
//! Lab-only INDEPENDENT HIP reference for the A3B native Q-resident path
//! (gfx1201). Unique lab exports:
//!   attention_fp8_a3b_qresident_independent_ref_gfx1201
//!   attention_fp8_a3b_qresident_independent_ref_range_gfx1201
//!
//! Contract:
//!   Grid:  (batch*16, 1, 1) — exactly one wave32 workgroup per canonical
//!          (query, head): q = blockIdx.x / 16, head = blockIdx.x % 16.
//!   Block: (32, 1, 1) — a single wave32. Dynamic LDS: 0 bytes.
//!   ABI:   6 pointers (Qcodes, Qscales, K, V, out, positions) + 4xi32
//!          (H16, KV2, D256, batch) + f32 scale_attn = 6*8 + 16 + 4 = 68 bytes
//!          of kernargs — the dense Q8 packed ABI. No new runtime API,
//!          no user knob, no default change.
//!   Q:     REAL Raw two-plane upstream-F32 prep output: codes plane
//!          [T*16*256] bytes in the fused-prep permute
//!          (dim d = 16f + 8kg + j at byte kg*128 + f*8 + j per (row, head))
//!          then F32 scale plane [T*16] (row amax/448, 1.0 for all-zero rows).
//!          No F16 rounding step anywhere; scales are exact F32.
//!   KV:    native fp8 rows, 516 B stride: codes at key*516 + kvhead*256 + d
//!          (256 B per KV head), F16 scale at key*516 + 512 + kvhead*2,
//!          staged to F32 exactly (no F16 product rounding).
//!          kvhead = head / 8 (GQA8: H=16/Hkv=2/D=256).
//!   Out:   F32 [B*16*256] at q*4096 + head*256 + dim.
//!
//! Independence (what this file does NOT share with the candidate):
//!   Never calls fa2_qresident_v2_q8_a3b_body, its loaders, its head/row
//!   mapper (row_base/16*wave+ml, query=row/8, head=8*kvh+row%8), its LDS
//!   issue/commit double buffering, its transposed-V path, or any Qfr/P-quant
//!   helper from attention_q8_0_fa2_gqa.gfx1201.hip. That TU is untouched.
//!   Only the hardware WMMA fp8 intrinsic (a fixed-function GPU arithmetic
//!   primitive), __expf, the E4M3/F16 bit formats and the permlanex16 row-pair
//!   reduction instruction are shared — the same way two independent programs
//!   may both use FMAD. All operands are gathered directly from global memory
//!   into registers with BYTE loads (legal on odd 516-byte rows and any
//!   2-byte scale offset); no LDS, no transpose kernels, no shared headers,
//!   no u32/u64 casts, no TR/builtin helper.
//!
//! Canonical WMMA fragment derivation (own mapping, documented here):
//!   gfx12 wave32 WMMA: lane (ml=lane&15, kg=lane>>4) supplies
//!     A[row=ml][k=kg*8+j] and B[row=ml][k=kg*8+j] (8 fp8 bytes each);
//!     acc[j] = C[8*kg+j][ml], C[i][n] = sum_k A[i][k]*B[n][k].
//!   QK (S = K*Q^T over one 16-key sub, dim chunk (2p+h)*16+j, p=0..7,
//!   h=0..1): A byte j of lane (ml,kg) = Kcode[key=kt_sub+ml]
//!     [dim=(2p+h)*16+kg*8+j] — every lane supplies its key row, so all 32
//!     lanes feed K. B byte j = (ml==0) ? Qcode[linear dim (2p+h)*16+kg*8+j]
//!     : 0, gathered from the PREP PERMUTE as
//!     qperm[f=(d>>4)][kg2=(d>>3)&1][j2=d&7] at kg2*128+f*8+j2.
//!     Only logical B row 0 (lanes 0 and 16) is live, so only acc of lanes
//!     0/16 (canonical query vs keys 0..7 / 8..15) is meaningful; other
//!     lanes' B rows are exact +0.0 and their acc is never stored. Zero
//!     bytes past gmax reproduce the candidate's zero-fill exactly.
//!   PV (O = P*V over the same 16-key sub, dim tile (2p+h), p=0..7):
//!     A byte j of lane (ml,kg) = Vcode[key=kt_sub+kg*8+j]
//!     [dim=(2p+h)*16+ml] — every lane supplies its dim row; the wave
//!     crossbar gathers all 16 dims. B = the lane's own packed P half8
//!     (lane (ml,kg) byte j = quant(P[key=kt_sub+kg*8+j])); only B row 0
//!     feeds the canonical acc in lanes 0/16. Ofr[2p+h][j] in lane (ml,kg)
//!     is O[query=ml][dim=(2p+h)*16+8*kg+j]; lanes 0/16 hold canonical
//!     dims 0..7/8..15 per tile.
//!   Reductions stay pair-local: max/sum/wmax exchange lanes 0<->16 with the
//!   same permlanex16 instruction the candidate uses for its (ml, ml^16)
//!   row-pair reductions, so the 16-key horizontal reductions match exactly.
//!
//! Exact dense DAG: initialize m_old=-inf, l=0, b_prev=2^-64. Ascending
//! KT48 tiles each contain three ascending 16-key substeps (kt_sub<=gmax):
//!   16 QK WMMA calls in (p=0..7,h=0,1) order; scores =
//!   (sacc*(Qscale*scale_attn))*KscaleF16, causally masked to -inf.
//!   Reduce mx over j=0..7 then xor16. A first non--inf mx sets m_old;
//!   thereafter only mx>m_old+4 changes the frame: alpha=exp(m_old-mx),
//!   l*=alpha, b_prev*=alpha, m_old=mx. O stays in stored-P units.
//!   e=(m_old==-inf)?0:exp(score-m_old); sum+=e in j order; weightedP=e*Vscale.
//!   After xor16 sum, l+=sum. Reduce weightedP max in the same order;
//!   needed=wmax>0?fmax(wmax/448,2^-64):b_prev. Only needed>b_prev
//!   rescales each Ofr[dc] by b_prev/needed and sets b_prev=needed.
//!   Thus P scale grows monotonically within a fixed hysteresis frame;
//!   only a frame change can decrease it. Quantize weightedP/b_prev in
//!   j=0..7 pairs, then 16 PV WMMA calls in (p,h) order into retained Ofr.
//!   Final inv=l>0?b_prev/l:0 uses direct division; store Ofr*inv.
//!   Mask group: 48 queries (384 query-head rows / GQA8), clipped to the
//!   current 512-query run and its actual tail, min/max over positions.
//!   Fully masked substeps retain the same -inf score/zero-e policy.
//!
//! Independent divisions: for finite positive wmax the candidate documents
//! exhaustive equality of its div448 helper after the 2^-64 floor with
//! IEEE wmax/448. For finite weightedP and positive finite b_prev, its
//! reciprocal lo/hi products bracket the IEEE quotient. E4M3 RNE conversion
//! is monotone: agreeing endpoint codes imply that quotient's code;
//! disagreeing endpoints explicitly fall back to the same IEEE division.
//! This is a conditional bracket argument, not a proof of reciprocal bounds
//! for arbitrary inputs. This reference uses no reciprocal/helper path.
//! Nonfinite inputs or nonpositive/nonfinite scale frames are outside that
//! finite equivalence scope; NaN/Inf outcomes are classification-only,
//! never evidence of finite bitwise equivalence.
//!
//! Precision: bitwise target on all FINITE gates (identical input bytes
//!   through an identical f32 operation order on identical hardware),
///!   including general Random K/V and Q fixtures — not only closed-form
//!   winners. NaN codes stay classification-only under the oracle policy.
//!   The oracle gates all finite attention outputs against this separate
//!   implementation; no epsilon is introduced.
//!
//! Host registration, the 68-byte argument pack and guarded reusable output
//! live only in fa_pm_fp8_oracle's explicit native family. This translation
//! unit changes no production symbol, threshold, golden or shipping default.

#include <hip/hip_runtime.h>

typedef float __attribute__((ext_vector_type(8))) indep_f8_t;
typedef int __attribute__((ext_vector_type(2))) indep_v2i_t;
typedef union {
    uint32_t u;
    _Float16 h[2];
} indep_dw_t;

// Exact F16-bits -> F32 (bit-cast): scales stage as exact F32, never rounded.
static __device__ __forceinline__ float indep_f16(uint16_t u)
{
    indep_dw_t w;
    w.u = (uint32_t)u;
    return (float)w.h[0];
}

static __device__ __forceinline__ indep_v2i_t indep_pack8(
    unsigned b0, unsigned b1, unsigned b2, unsigned b3,
    unsigned b4, unsigned b5, unsigned b6, unsigned b7)
{
    const unsigned w0 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    const unsigned w1 = b4 | (b5 << 8) | (b6 << 16) | (b7 << 24);
    indep_v2i_t v = {(int)w0, (int)w1};
    return v;
}

// Pair-exchange hardware primitive: the same permlanex16 instruction the
// candidate body uses for its (ml, ml^16) row-pair reductions
// (mx/sum/wmax over the 8+8 keys of one 16-key substep). A raw intrinsic,
// not a candidate helper call: lanes 0/16 exchange, every other pair is
// don't-care (only lanes 0/16 hold live canonical state here).
static __device__ __forceinline__ float indep_xor16(float v)
{
    int bits = __builtin_bit_cast(int, v);
    bits = __builtin_amdgcn_permlanex16(bits, bits, 0x76543210, 0xfedcba98,
                                        false, false);
    return __builtin_bit_cast(float, bits);
}

static __device__ __forceinline__ void a3b_ref_body(
    const unsigned char* __restrict__ q_codes,
    const float* __restrict__ q_scales,
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
    if (n_heads != 16 || n_kv_heads != 2 || head_dim != 256)
        return;
    if (batch_size <= 0 || batch_size > 32768)
        return;
    const int lane = (int)threadIdx.x; // 0..31, the single wave
    const int ml = lane & 15;
    const int kg = lane >> 4;
    const int bid = (int)blockIdx.x;
    const int q = bid / 16;
    const int head = bid % 16;
    if (q >= batch_size)
        return;
    const int kvh = head >> 3;
    const int pos = positions[q];

    // Independent canonical group: the candidate's 384 query-head rows cover
    // 48 GQA8 queries, never crossing a 512-query run or its actual tail.
    const int run_base = (q / 512) * 512;
    const int run_end = min(run_base + 512, batch_size);
    const int gbase = run_base + ((q - run_base) / 48) * 48;
    const int gend = min(gbase + 48, run_end);
    int wmx = -1;
    int wmn = 0x7fffffff;
    for (int gq = gbase + lane; gq < gend; gq += 32) {
        const int p = positions[gq];
        wmx = max(wmx, p);
        wmn = min(wmn, p);
    }
    wmx = max(wmx, __shfl_xor(wmx, 16, 32));
    wmx = max(wmx, __shfl_xor(wmx, 8, 32));
    wmx = max(wmx, __shfl_xor(wmx, 4, 32));
    wmx = max(wmx, __shfl_xor(wmx, 2, 32));
    wmx = max(wmx, __shfl_xor(wmx, 1, 32));
    wmn = min(wmn, __shfl_xor(wmn, 16, 32));
    wmn = min(wmn, __shfl_xor(wmn, 8, 32));
    wmn = min(wmn, __shfl_xor(wmn, 4, 32));
    wmn = min(wmn, __shfl_xor(wmn, 2, 32));
    wmn = min(wmn, __shfl_xor(wmn, 1, 32));
    const int gmax = __builtin_amdgcn_readfirstlane(wmx);
    const int gmin = __builtin_amdgcn_readfirstlane(wmn);

    // REAL Raw Q row: permuted codes then out-of-line F32 scale plane.
    const unsigned qrow = (unsigned)q * 16u + (unsigned)head;
    const unsigned char* qcodes = q_codes + (unsigned long long)qrow * 256ULL;
    const float qs_tile = q_scales[qrow] * scale_attn;

    indep_f8_t Ofr[16];
#pragma unroll
    for (int dc = 0; dc < 16; ++dc)
        Ofr[dc] = (indep_f8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float m_old = -INFINITY;
    float l_val = 0.0f;
    float b_prev = 0x1p-64f;

    // KT48: 48-key tiles, 3 substeps of 16 keys, ascending.
    for (int ktile = 0; ktile <= gmax; ktile += 48) {
        for (int sub = 0; sub < 3; ++sub) {
            const int kt_sub = ktile + sub * 16;
            if (kt_sub > gmax)
                continue;
            // ---- QK: 16 WMMA calls, (p,h) order matches candidate ----
            indep_f8_t sacc =
                (indep_f8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (int p = 0; p < 8; ++p) {
                for (int h = 0; h < 2; ++h) {
                    const unsigned dbase = (unsigned)(2 * p + h) * 16u;
                    unsigned ab[8];
                    unsigned bb[8];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int key = kt_sub + ml;
                        const unsigned d =
                            dbase + (unsigned)kg * 8u + (unsigned)j;
                        unsigned av = 0u;
                        if (key <= gmax)
                            av = k_cache[(unsigned long long)key * 516ULL +
                                         (unsigned long long)kvh * 256ULL + d];
                        ab[j] = av;
                        unsigned qv = 0u;
                        if (ml == 0) {
                            // Linear dim -> prep permute:
                            // d = 16f + 8kg2 + j2.
                            const unsigned f = d >> 4;
                            const unsigned kg2 = (d >> 3) & 1u;
                            const unsigned j2 = d & 7u;
                            qv = (unsigned)qcodes[kg2 * 128u + f * 8u + j2];
                        }
                        bb[j] = qv;
                    }
                    const indep_v2i_t kf = indep_pack8(ab[0], ab[1], ab[2],
                                                      ab[3], ab[4], ab[5],
                                                      ab[6], ab[7]);
                    const indep_v2i_t qf = indep_pack8(bb[0], bb[1], bb[2],
                                                      bb[3], bb[4], bb[5],
                                                      bb[6], bb[7]);
                    sacc =
                        __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                            kf, qf, sacc);
                }
            }
            const bool sub_full = ((kt_sub + 15) <= gmin);
            float sc[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int key = kt_sub + 8 * kg + j;
                uint16_t u = 0u;
                if (key <= gmax) {
                    const unsigned char* kp =
                        k_cache + (unsigned long long)key * 516ULL + 512u +
                        (unsigned long long)kvh * 2ULL;
                    u = (uint16_t)kp[0] | ((uint16_t)kp[1] << 8);
                }
                sc[j] = indep_f16(u);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int key_g = kt_sub + 8 * kg + j;
                const bool ok = (sub_full || key_g <= pos);
                const float x =
                    ok ? (sacc[j] * qs_tile) * sc[j] : -INFINITY;
                sacc[j] = x;
            }
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                mx = fmaxf(mx, sacc[j]);
            mx = fmaxf(mx, indep_xor16(mx));
            // Hysteretic softmax frame. Ofr remains in stored-P units:
            // changing b_prev with l_val already compensates its frame.
            if (mx != -INFINITY) {
                if (m_old == -INFINITY) {
                    m_old = mx;
                } else if (mx > m_old + 4.0f) {
                    const float alpha = __expf(m_old - mx);
                    l_val *= alpha;
                    b_prev *= alpha;
                    m_old = mx;
                }
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int key = kt_sub + 8 * kg + j;
                uint16_t u = 0u;
                if (key <= gmax) {
                    const unsigned char* vp =
                        v_cache + (unsigned long long)key * 516ULL + 512u +
                        (unsigned long long)kvh * 2ULL;
                    u = (uint16_t)vp[0] | ((uint16_t)vp[1] << 8);
                }
                sc[j] = indep_f16(u);
            }
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float e = (m_old == -INFINITY)
                    ? 0.0f
                    : __expf(sacc[j] - m_old);
                sum += e;
                sacc[j] = e * sc[j];
            }
            sum += indep_xor16(sum);
            l_val += sum;
            float wmax = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                wmax = fmaxf(wmax, sacc[j]);
            wmax = fmaxf(wmax, indep_xor16(wmax));
            const float needed = wmax > 0.0f
                ? fmaxf(wmax / 448.0f, 0x1p-64f)
                : b_prev;
            if (needed > b_prev) {
                const float rho = b_prev / needed;
#pragma unroll
                for (int dc = 0; dc < 16; ++dc)
                    Ofr[dc] *= rho;
                b_prev = needed;
            }
            // Independent IEEE quotient, then hardware E4M3 RNE; see the
            // finite positive-frame bracket scope in the contract above.
            unsigned int p0 = 0u;
            p0 = __builtin_amdgcn_cvt_pk_fp8_f32(
                sacc[0] / b_prev, sacc[1] / b_prev, p0, false);
            p0 = __builtin_amdgcn_cvt_pk_fp8_f32(
                sacc[2] / b_prev, sacc[3] / b_prev, p0, true);
            unsigned int p1 = 0u;
            p1 = __builtin_amdgcn_cvt_pk_fp8_f32(
                sacc[4] / b_prev, sacc[5] / b_prev, p1, false);
            p1 = __builtin_amdgcn_cvt_pk_fp8_f32(
                sacc[6] / b_prev, sacc[7] / b_prev, p1, true);
            const indep_v2i_t pb = {(int)p0, (int)p1};
            // ---- PV: 8 dim tiles, both halves in order ----
            for (int p = 0; p < 8; ++p) {
                for (int h = 0; h < 2; ++h) {
                    const unsigned d =
                        (unsigned)(2 * p + h) * 16u + (unsigned)ml;
                    unsigned vb[8];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int key = kt_sub + kg * 8 + j;
                        unsigned vv = 0u;
                        if (key <= gmax)
                            vv = v_cache[(unsigned long long)key * 516ULL +
                                         (unsigned long long)kvh * 256ULL + d];
                        vb[j] = vv;
                    }
                    const indep_v2i_t va = indep_pack8(vb[0], vb[1], vb[2],
                                                      vb[3], vb[4], vb[5],
                                                      vb[6], vb[7]);
                    Ofr[2 * p + h] =
                        __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                            va, pb, Ofr[2 * p + h]);
                }
            }
        }
    }
    // ---- Completion: lanes 0/16 own canonical dims 0..7/8..15 ----
    if (ml == 0) {
        const float inv = l_val > 0.0f ? b_prev / l_val : 0.0f;
        const unsigned bo = (unsigned)q * 4096u + (unsigned)head * 256u;
#pragma unroll
        for (int dc = 0; dc < 16; ++dc) {
            const unsigned bd =
                bo + (unsigned)dc * 16u + (unsigned)kg * 8u;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                out[bd + (unsigned)j] = Ofr[dc][j] * inv;
        }
    }
}

// Grid (B*16,1,1), block (32,1,1), dynamic LDS 0. See contract above.
extern "C" __global__ __launch_bounds__(32, 1)
void attention_fp8_a3b_qresident_independent_ref_gfx1201(
    const unsigned char* __restrict__ q_codes,
    const float* __restrict__ q_scales,
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
    a3b_ref_body(q_codes, q_scales, k_cache, v_cache, out, positions,
                 n_heads, n_kv_heads, head_dim, batch_size, scale_attn);
}

extern "C" __global__ __launch_bounds__(32, 1)
void attention_fp8_a3b_qresident_independent_ref_range_gfx1201(
    const unsigned char* __restrict__ q_codes,
    const float* __restrict__ q_scales,
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
    // Same scalar body; range offsets are applied host-side to the Q plane
    // bases from the FULL allocation (codes + row_offset*4096,
    // scales + total*4096 + row_offset*64), exactly like the candidate range.
    a3b_ref_body(q_codes, q_scales, k_cache, v_cache, out, positions,
                 n_heads, n_kv_heads, head_dim, batch_size, scale_attn);
}
