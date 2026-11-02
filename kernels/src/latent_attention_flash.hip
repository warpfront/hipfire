// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

// Tiled causal flash-style latent attention prefill (BF16 matrix units, online
// softmax, no T^2 storage, no atomics). Operand contract and semantics:
// crates/rdna-compute/src/latent_attention_tiled.rs. Geometry is runtime:
//     key(j, h)   = ka(j, h)[0..wa] ++ kb(j, h)[0..wb]      (dq = wa + wb)
//     score       = scale * dot(q[i][h][0..dq], key(j, h))
//     y[i][h]     = sum_j softmax_j(score) * v(j, h)[0..dv]
// for query row i of this call, visible keys j <= *pos + pos_offset + i and
// j < max_rows; a row with no visible key writes zeros.
//
// Decomposition. grid.x = query tiles of 16 * NW rows (NW = blockDim.x / 32
// waves, one wave per 16 query rows; the heaviest, latest, tile is block 0),
// grid.y = head, grid.z = chunk of up to 128 value columns (each z block
// recomputes the scores; dv > 128 is the absorbed latent route).
//
// The score tile is computed TRANSPOSED, S^T = K . Q^T, so the accumulator
// fragment is lane-per-query (query = lane & 15) and the softmax row state
// (m, l) is lane-local:
//   A = 16 key rows x 16 dims, loaded straight from global memory,
//   B = Q^T from LDS (BF16, the wave's 16 queries, rounded once at staging).
// The probabilities (rounded to BF16, and summed from those same rounded
// values) become the B operand of O^T = V^T . P^T; V^T is staged per 64-key
// tile in LDS as [dv column][key pairs]. O^T is lane-per-query again, so the
// softmax correction is a lane-local scale. Fixed summation order everywhere:
// WMMA chains over ascending dims / keys, per-lane partial l combined with one
// fixed xor-16 exchange.
//
// Fragment maps (wave32):
//   gfx12 : A/B hold 8 of the 16 k per lane (k = 8 * (lane >> 4) + e), C[j] =
//           C[8 * (lane >> 4) + j][lane & 15].
//   gfx11 : A/B hold all 16 k per lane (lanes 16..31 replicate 0..15), C[j] =
//           C[2 * j + (lane >> 4)][lane & 15].
// The contraction index is a free labelling, so the key order inside the PV
// operands is chosen to match what each arch's S^T accumulator holds. The V^T
// LDS word of key-pair w of a 16-key sub-tile (w = 2q + r) packs keys
// (4q + r) | (4q + 2 + r) << 16, which is what both archs' P^T build produces.
//
// Requirements (the launcher refuses otherwise): wa, wb, dv multiples of 16;
// every offset/stride a multiple of 8 elements; 16-byte aligned base pointers;
// dq <= 1024; blockDim.x in {32, 64, 128}; dynamic LDS =
// NW * 16 * (dq + 8) * 2 + min(dv, 128) * 36 * 4 bytes.
//
// The LAF_EMU_GFX11 macro (fixture builds only) makes a gfx12 device execute
// the gfx11 fragment maps by emulating the gfx11 matrix op with the gfx12 one
// plus lane exchanges, so the gfx11 operand/accumulator logic is testable on
// gfx12 hardware.

#define LAF_KT 64      // keys per iteration
#define LAF_DVC 128    // value columns per z block

// exp(x) for x <= 0 (exp(-inf) == 0): the hardware base-2 exponential with the
// argument pre-scaled; relative error ~1e-6, fixed across runs.
static __device__ __forceinline__ float laf_exp(float x) {
    return __builtin_amdgcn_exp2f(x * 1.4426950408889634f);
}
#define LAF_VP 36      // LDS words per V^T column row: 32 key-pair words + 4 pad
#ifndef LAF_VPF
#define LAF_VPF 4      // value-fill rounds prefetched into registers
#endif

#if defined(LAF_EMU_GFX11)
#define LAF_G12_ISA 1
#define LAF_LAYOUT_G11 1
#elif defined(__GFX12__)
#define LAF_G12_ISA 1
#define LAF_LAYOUT_G11 0
#else
#define LAF_G12_ISA 0
#define LAF_LAYOUT_G11 1
#endif

typedef short __attribute__((ext_vector_type(8))) laf_s8_t;
typedef short __attribute__((ext_vector_type(16))) laf_s16_t;
typedef unsigned int __attribute__((ext_vector_type(8))) laf_u8_t;
typedef unsigned int __attribute__((ext_vector_type(4))) laf_u4_t;
typedef float __attribute__((ext_vector_type(8))) laf_f8_t;

#if LAF_LAYOUT_G11
typedef laf_s16_t laf_op_t;   // 16 k per lane
#define LAF_QOFF(hh) 0
#else
typedef laf_s8_t laf_op_t;    // 8 k per lane
#define LAF_QOFF(hh) (8 * (hh))
#endif

static __device__ __forceinline__ uint32_t laf_f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (u >> 16) | 0x40u;
    u += 0x7fffu + ((u >> 16) & 1u);
    return u >> 16;
}

// Round-to-nearest-even of a finite value, result left in the high 16 bits.
static __device__ __forceinline__ uint32_t laf_rne_hi(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
}

// ---- matrix op ---------------------------------------------------------------
#if defined(LAF_EMU_GFX11)
// gfx11 matrix op on gfx12 hardware: lane (hh, n) of the gfx12 op contracts its
// 8-k half of the (replicated) 16-k operand; the gfx12 result C[8 * hh' + j'][n]
// is moved to the gfx11 position C[2 * j + hh][n] by lane exchange.
static __device__ __forceinline__ laf_f8_t laf_wmma(laf_op_t a, laf_op_t b, laf_f8_t c) {
    const int hh = threadIdx.x >> 4 & 1;
    laf_s8_t a8, b8;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        a8[e] = hh ? a[8 + e] : a[e];
        b8[e] = hh ? b[8 + e] : b[e];
    }
    const laf_f8_t z = {0, 0, 0, 0, 0, 0, 0, 0};
    const laf_f8_t r = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a8, b8, z);
    const int n = threadIdx.x & 15;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int src = ((j >> 2) << 4) | n;  // holder of row 2j + hh: half (2j + hh) >> 3 == j >> 2
        const float v0 = __shfl(r[(2 * j) & 7], src, 32);
        const float v1 = __shfl(r[(2 * j + 1) & 7], src, 32);
        c[j] += hh ? v1 : v0;
    }
    return c;
}
#elif LAF_G12_ISA
static __device__ __forceinline__ laf_f8_t laf_wmma(laf_op_t a, laf_op_t b, laf_f8_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
#else
static __device__ __forceinline__ laf_f8_t laf_wmma(laf_op_t a, laf_op_t b, laf_f8_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}
#endif

// ---- operand loads -------------------------------------------------------------
// p: 16-byte aligned. gfx12 layout: 8 elements; gfx11 layout: 16 elements.
static __device__ __forceinline__ laf_op_t laf_ldop(const uint16_t* p) {
    const uint4 u0 = *(const uint4*)p;
#if LAF_LAYOUT_G11
    const uint4 u1 = *(const uint4*)(p + 8);
    const laf_u8_t w = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
    return __builtin_bit_cast(laf_op_t, w);
#else
    return __builtin_bit_cast(laf_op_t, u0);
#endif
}

// S^T tile: acc[ks] += K(row jr + 16 * ks, dims [0, 16 * nsteps)) . Q^T(dims
// [qcol0, ...)); row jr + 16 * ks is read only while 16 * ks < jrem (otherwise
// the operand is zero and no load is issued). `kp0` addresses row jr, `step16`
// is the element distance of 16 rows.
static __device__ __forceinline__ void laf_qk_pass(
    laf_f8_t (&acc)[4], const uint16_t* kp0, size_t step16, int jrem,
    const uint16_t* qrow, int qcol0, int nsteps, int hh
) {
#pragma unroll 2
    for (int s = 0; s < nsteps; ++s) {
        const int d = s * 16;
        const laf_op_t qf = laf_ldop(qrow + qcol0 + d + LAF_QOFF(hh));
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            {
                laf_op_t kf;
                if (16 * ks < jrem) {
                    kf = laf_ldop(kp0 + ks * step16 + d + LAF_QOFF(hh));
                } else {
#pragma unroll
                    for (int e = 0; e < (LAF_LAYOUT_G11 ? 16 : 8); ++e) kf[e] = 0;
                }
                acc[ks] = laf_wmma(kf, qf, acc[ks]);
            }
        }
    }
}

// Write one V^T task: 8 value columns of key pair `p` (a: low key, b: high key).
static __device__ __forceinline__ void laf_v_store(uint32_t* vs, int g, int p, uint4 a, uint4 b) {
    const uint32_t aw[4] = {a.x, a.y, a.z, a.w};
    const uint32_t bw[4] = {b.x, b.y, b.z, b.w};
    uint32_t* col = vs + (g * 8) * LAF_VP + p;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        col[(2 * i) * LAF_VP] = (aw[i] & 0xffffu) | (bw[i] << 16);
        col[(2 * i + 1) * LAF_VP] = (aw[i] >> 16) | (bw[i] & 0xffff0000u);
    }
}

extern "C" __global__ __launch_bounds__(128) void latent_attention_flash_prefill(
    const float* __restrict__ q,         // [n_rows, H, dq]
    float* __restrict__ y,               // [n_rows, H, dv]
    const uint16_t* __restrict__ ka,     // BF16 view, offset already applied
    const uint16_t* __restrict__ kb,
    const uint16_t* __restrict__ vv,
    const int* __restrict__ pos_ptr,     // device scalar: rows cached before this call
    int n_rows, int n_heads, int wa, int wb, int dv, int pos_offset, int max_rows,
    int ka_tok, int ka_head, int kb_tok, int kb_head, int v_tok, int v_head,
    float scale, int round_y
) {
    extern __shared__ __attribute__((aligned(16))) unsigned char laf_smem[];
    const int tid = threadIdx.x;
    const int nw = blockDim.x >> 5;
    const int wave = tid >> 5, lane = tid & 31, l16 = lane & 15, hh = lane >> 4;
    const int dq = wa + wb;
    const int qp = dq + 8;  // LDS pitch of a staged query row (elements)
    uint16_t* qs = (uint16_t*)laf_smem;
    uint32_t* vs = (uint32_t*)(laf_smem + (size_t)nw * 16 * qp * 2);

    const int h = blockIdx.y;
    const int dvc0 = blockIdx.z * LAF_DVC;
    const int cw = min(LAF_DVC, dv - dvc0);
    const int ngroups = cw >> 3;
    const int qt = 16 * nw;
    const int tile = (int)gridDim.x - 1 - (int)blockIdx.x;  // heaviest tile first
    const int i0 = tile * qt;

    // Visible limit per query row: keys j <= lim_i. Rows past n_rows replicate
    // the last row (their output is never stored).
    const int base = *pos_ptr + pos_offset;
    const int wrow0 = i0 + wave * 16;
    const int row_l = wrow0 + l16;
    const int lim_row = min(row_l, n_rows - 1);
    const int lim_i = min(base + lim_row, max_rows - 1);
    const int lim_first = min(base + min(wrow0, n_rows - 1), max_rows - 1);
    const int lim_last = min(base + min(wrow0 + 15, n_rows - 1), max_rows - 1);
    const int kend = max(0, min(base + min(i0 + qt - 1, n_rows - 1), max_rows - 1) + 1);

    // ---- stage this wave's 16 queries as BF16 --------------------------------
    for (int r = 0; r < 16; ++r) {
        const int row = wrow0 + r;
        uint2* dst = (uint2*)(qs + (size_t)(wave * 16 + r) * qp);
        if (row < n_rows) {
            const float4* src = (const float4*)(q + ((size_t)row * n_heads + h) * dq);
            for (int c = lane; c < (dq >> 2); c += 32) {
                const float4 f = src[c];
                dst[c] = make_uint2(laf_f32_to_bf16(f.x) | (laf_f32_to_bf16(f.y) << 16),
                                    laf_f32_to_bf16(f.z) | (laf_f32_to_bf16(f.w) << 16));
            }
        } else {
            for (int c = lane; c < (dq >> 2); c += 32) dst[c] = make_uint2(0u, 0u);
        }
    }
    __syncthreads();

    const uint16_t* qrow = qs + (size_t)(wave * 16 + l16) * qp;

    laf_f8_t acc_o[8];
#pragma unroll
    for (int d = 0; d < 8; ++d) acc_o[d] = laf_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY;
    float l = 0.0f;

    // V^T fill geometry: lane = key pair; word w = 2q + r of sub-tile ks holds
    // keys (ks*16 + 4q + r) | (+2).
    const int vks = lane >> 3, vq = (lane >> 1) & 3, vr = lane & 1;
    const int vklo = vks * 16 + 4 * vq + vr;
    const int vkhi = vklo + 2;

    for (int kb0 = 0; kb0 < kend; kb0 += LAF_KT) {

        // V rows of this tile into registers early; written to LDS after the
        // score math so the loads overlap it. Beyond kend: never read, zero-filled.
        const uint16_t* vpa = (kb0 + vklo < kend)
            ? vv + (size_t)(kb0 + vklo) * v_tok + (size_t)h * v_head + dvc0 : nullptr;
        const uint16_t* vpb = (kb0 + vkhi < kend)
            ? vv + (size_t)(kb0 + vkhi) * v_tok + (size_t)h * v_head + dvc0 : nullptr;
        uint4 va[LAF_VPF], vb[LAF_VPF];
#pragma unroll
        for (int it = 0; it < LAF_VPF; ++it) {
            const int g = wave + it * nw;
            va[it] = make_uint4(0u, 0u, 0u, 0u);
            vb[it] = make_uint4(0u, 0u, 0u, 0u);
            if (g < ngroups) {
                if (vpa) va[it] = *(const uint4*)(vpa + g * 8);
                if (vpb) vb[it] = *(const uint4*)(vpb + g * 8);
            }
        }

        float corr = 1.0f;
        laf_op_t pf[4];
        const bool wave_live = kb0 <= lim_last;
        if (wave_live) {
            // ---- S^T = K . Q^T ---------------------------------------------------
            const int jr = kb0 + l16;
            const uint16_t* kpa0 = ka + (size_t)jr * ka_tok + (size_t)h * ka_head;
            const uint16_t* kpb0 = kb + (size_t)jr * kb_tok + (size_t)h * kb_head;
            laf_f8_t acc_s[4];
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) acc_s[ks] = laf_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
            laf_qk_pass(acc_s, kpa0, (size_t)16 * ka_tok, kend - jr, qrow, 0, wa >> 4, hh);
            laf_qk_pass(acc_s, kpb0, (size_t)16 * kb_tok, kend - jr, qrow, wa, wb >> 4, hh);

            // ---- scale, mask, online softmax ---------------------------------------
            const bool need_mask = kb0 + (LAF_KT - 1) > lim_first;
            float sv[4][8];
            float tmax = -INFINITY;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float s = acc_s[ks][j] * scale;
                    if (need_mask) {
#if LAF_LAYOUT_G11
                        const int kj = kb0 + ks * 16 + 2 * j + hh;
#else
                        const int kj = kb0 + ks * 16 + 8 * hh + j;
#endif
                        s = (kj <= lim_i) ? s : -INFINITY;
                    }
                    sv[ks][j] = s;
                    tmax = fmaxf(tmax, s);
                }
            }
            tmax = fmaxf(tmax, __shfl_xor(tmax, 16, 32));
            const float m_new = fmaxf(m, tmax);
            const float mshift = (m_new == -INFINITY) ? 0.0f : m_new;
            corr = laf_exp(m - mshift);  // m == -inf -> 0 (nothing accumulated yet)
            m = m_new;

            float lsum = 0.0f;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                uint32_t pw[8];  // rounded probabilities, bf16 in the high half
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    pw[j] = laf_rne_hi(laf_exp(sv[ks][j] - mshift));
                    lsum += __uint_as_float(pw[j]);
                }
#if LAF_LAYOUT_G11
                // own word q packs acc[2q], acc[2q+1] = keys (4q + hh, 4q + 2 + hh).
                uint32_t own[4], oth[4];
#pragma unroll
                for (int w = 0; w < 4; ++w)
                    own[w] = (pw[2 * w] >> 16) | pw[2 * w + 1];
#pragma unroll
                for (int w = 0; w < 4; ++w) oth[w] = __shfl_xor((int)own[w], 16, 32);
                // W[2q + r]: r == hh -> own word q, else the partner's.
                laf_u8_t W;
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    W[2 * w] = hh ? oth[w] : own[w];
                    W[2 * w + 1] = hh ? own[w] : oth[w];
                }
                pf[ks] = __builtin_bit_cast(laf_op_t, W);
#else
                // word m' packs acc[4(m'>>1) + (m'&1)] | acc[... + 2] << 16.
                uint32_t pk[4];
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const int lo = 4 * (w >> 1) + (w & 1);
                    pk[w] = (pw[lo] >> 16) | pw[lo + 2];
                }
                const laf_u4_t W = {pk[0], pk[1], pk[2], pk[3]};
                pf[ks] = __builtin_bit_cast(laf_op_t, W);
#endif
            }
            l = l * corr + lsum;
#pragma unroll
            for (int d = 0; d < 8; ++d)
                if (d * 16 < cw) acc_o[d] *= corr;
        }

        // ---- V^T tile into LDS --------------------------------------------------
#pragma unroll
        for (int it = 0; it < LAF_VPF; ++it) {
            const int g = wave + it * nw;
            if (g < ngroups) laf_v_store(vs, g, lane, va[it], vb[it]);
        }
        for (int g = wave + LAF_VPF * nw; g < ngroups; g += nw) {
            uint4 a = make_uint4(0u, 0u, 0u, 0u), b = a;
            if (vpa) a = *(const uint4*)(vpa + g * 8);
            if (vpb) b = *(const uint4*)(vpb + g * 8);
            laf_v_store(vs, g, lane, a, b);
        }
        __syncthreads();

        // ---- O^T += V^T . P^T -----------------------------------------------------
        if (wave_live) {
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                if (d * 16 < cw) {
                    const uint32_t* arow = vs + (size_t)(d * 16 + l16) * LAF_VP;
#pragma unroll
                    for (int ks = 0; ks < 4; ++ks) {
                        {
#if LAF_LAYOUT_G11
                            const uint4 u0 = *(const uint4*)(arow + ks * 8);
                            const uint4 u1 = *(const uint4*)(arow + ks * 8 + 4);
                            const laf_u8_t w = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
                            const laf_op_t af = __builtin_bit_cast(laf_op_t, w);
#else
                            const uint4 u0 = *(const uint4*)(arow + ks * 8 + 4 * hh);
                            const laf_op_t af = __builtin_bit_cast(laf_op_t, u0);
#endif
                            acc_o[d] = laf_wmma(af, pf[ks], acc_o[d]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---- epilogue: y = O / l -------------------------------------------------------
    const float l_tot = l + __shfl_xor(l, 16, 32);
    const bool row_ok = row_l < n_rows;
    float* yrow = y + ((size_t)(row_ok ? row_l : 0) * n_heads + h) * dv + dvc0;
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        if (d * 16 >= cw) continue;
        float o[8];
#if LAF_LAYOUT_G11
        // acc[j] = column 2j + hh: swap halves so each lane owns 8 consecutive columns.
        float own[4], rcv[4];
#pragma unroll
        for (int mm = 0; mm < 4; ++mm) {
            const float send = hh ? acc_o[d][mm] : acc_o[d][4 + mm];
            own[mm] = hh ? acc_o[d][4 + mm] : acc_o[d][mm];
            rcv[mm] = __shfl_xor(send, 16, 32);
        }
#pragma unroll
        for (int mm = 0; mm < 4; ++mm) {
            o[2 * mm] = hh ? rcv[mm] : own[mm];
            o[2 * mm + 1] = hh ? own[mm] : rcv[mm];
        }
#else
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = acc_o[d][j];
#endif
        float r[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float v = (l_tot > 0.0f) ? o[j] / l_tot : 0.0f;
            if (round_y) v = __uint_as_float(laf_f32_to_bf16(v) << 16);
            r[j] = v;
        }
        if (row_ok) {
            float4* dst = (float4*)(yrow + d * 16 + 8 * hh);
            dst[0] = make_float4(r[0], r[1], r[2], r[3]);
            dst[1] = make_float4(r[4], r[5], r[6], r[7]);
        }
    }
}

// ============================================================================
// Wide-value variant: value width up to 512 in ONE pass over the keys.
//
// The chunked kernel above recomputes the score tile once per 128 value
// columns, which makes the absorbed latent route (dq = 576, dv = 512) spend four
// score passes per value pass. This kernel keeps the same fragment maps and
// numerics, but one block of four waves owns one 16-query tile and one head:
//   * score phase: wave w = 2 * half + sub computes the partial S^T of key
//     sub-tile `sub` (16 keys of the 32-key tile) over half `half` of the
//     contraction steps, into two independent accumulator chains;
//   * the four partials are exchanged through LDS (lane-identical layout, so no
//     lane movement) and every wave forms S = partial(half 0) + partial(half 1)
//     in that fixed order, then runs the identical softmax (the softmax state of a
//     query is replicated, bit for bit, in all four waves);
//   * value phase: the full V^T tile (dv x 32 keys) is staged transposed in LDS;
//     wave w accumulates value columns [w * dv/4, (w + 1) * dv/4).
// Every score and value product is computed exactly once. Requirements on top of
// the above: dv a multiple of 64 with dv / 4 <= 128; dynamic LDS =
// 16 * (dq + 8) * 2 + dv * 20 * 4 + 4096 bytes; blockDim.x == 128, grid.z == 1.
// ============================================================================

#define LAW_KT 32      // keys per iteration
#define LAW_VP 20      // words per V^T column: 16 key-pair words + 4 pad
#ifndef LAW_VPF
#define LAW_VPF 8      // value-fill rounds prefetched into registers (all of them for dv <= 512)
#endif
#ifndef LAW_SU
#define LAW_SU 2       // contraction steps issued per score-loop iteration (alternating 2 chains)
#endif

// 32-key tile version of laf_v_store (pitch LAW_VP).
static __device__ __forceinline__ void law_v_store(uint32_t* vs, int g, int p, uint4 a, uint4 b) {
    const uint32_t aw[4] = {a.x, a.y, a.z, a.w};
    const uint32_t bw[4] = {b.x, b.y, b.z, b.w};
    uint32_t* col = vs + (g * 8) * LAW_VP + p;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        col[(2 * i) * LAW_VP] = (aw[i] & 0xffffu) | (bw[i] << 16);
        col[(2 * i + 1) * LAW_VP] = (aw[i] >> 16) | (bw[i] & 0xffff0000u);
    }
}

extern "C" __global__ __launch_bounds__(128) void latent_attention_flash_prefill_wide(
    const float* __restrict__ q,         // [n_rows, H, dq]
    float* __restrict__ y,               // [n_rows, H, dv]
    const uint16_t* __restrict__ ka,     // BF16 view, offset already applied
    const uint16_t* __restrict__ kb,
    const uint16_t* __restrict__ vv,
    const int* __restrict__ pos_ptr,     // device scalar: rows cached before this call
    int n_rows, int n_heads, int wa, int wb, int dv, int pos_offset, int max_rows,
    int ka_tok, int ka_head, int kb_tok, int kb_head, int v_tok, int v_head,
    float scale, int round_y
) {
    extern __shared__ __attribute__((aligned(16))) unsigned char laf_smem[];
    const int tid = threadIdx.x;
    const int wave = tid >> 5, lane = tid & 31, l16 = lane & 15, hh = lane >> 4;
    const int dq = wa + wb;
    const int qp = dq + 8;  // LDS pitch of a staged query row (elements)
    uint16_t* qs = (uint16_t*)laf_smem;
    uint32_t* vs = (uint32_t*)(laf_smem + (size_t)16 * qp * 2);
    float* sp = (float*)(vs + (size_t)dv * LAW_VP);  // [4 waves][8][32 lanes]

    const int h = blockIdx.y;
    const int cw = dv >> 2;               // value columns owned by this wave
    const int dvc0 = wave * cw;
    const int ngroups = dv >> 3;          // 8-column groups of the whole V^T tile
    const int tile = (int)gridDim.x - 1 - (int)blockIdx.x;  // heaviest tile first
    const int i0 = tile * 16;

    const int base = *pos_ptr + pos_offset;
    const int row_l = i0 + l16;
    const int lim_row = min(row_l, n_rows - 1);
    const int lim_i = min(base + lim_row, max_rows - 1);
    const int lim_first = min(base + min(i0, n_rows - 1), max_rows - 1);
    const int lim_last = min(base + min(i0 + 15, n_rows - 1), max_rows - 1);
    const int kend = max(0, lim_last + 1);

    // ---- stage the 16 queries as BF16 (4 rows per wave) ------------------------
    for (int r = wave * 4; r < wave * 4 + 4; ++r) {
        const int row = i0 + r;
        uint2* dst = (uint2*)(qs + (size_t)r * qp);
        if (row < n_rows) {
            const float4* src = (const float4*)(q + ((size_t)row * n_heads + h) * dq);
            for (int c = lane; c < (dq >> 2); c += 32) {
                const float4 f = src[c];
                dst[c] = make_uint2(laf_f32_to_bf16(f.x) | (laf_f32_to_bf16(f.y) << 16),
                                    laf_f32_to_bf16(f.z) | (laf_f32_to_bf16(f.w) << 16));
            }
        } else {
            for (int c = lane; c < (dq >> 2); c += 32) dst[c] = make_uint2(0u, 0u);
        }
    }
    __syncthreads();

    const uint16_t* qrow = qs + (size_t)l16 * qp;

    laf_f8_t acc_o[8];
#pragma unroll
    for (int d = 0; d < 8; ++d) acc_o[d] = laf_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY;
    float l = 0.0f;

    // score phase geometry
    const int sub = wave & 1, half = wave >> 1;
    const int nsteps = dq >> 4;
    const int nlo = (nsteps + 1) >> 1;
    const int s0 = half ? nlo : 0, s1 = half ? nsteps : nlo;

    // V^T fill geometry: lane&15 = key pair; word w = 2q + r of sub-tile ks holds
    // keys (ks*16 + 4q + r) | (+2); the two lane halves take alternate 8-column groups.
    const int vp = lane & 15;
    const int vks = vp >> 3, vq = (vp >> 1) & 3, vr = vp & 1;
    const int vklo = vks * 16 + 4 * vq + vr;
    const int vkhi = vklo + 2;
    const int gbase = wave * 2 + hh;  // group g = gbase + 8 * it

    for (int kb0 = 0; kb0 < kend; kb0 += LAW_KT) {
        // V rows into registers early (overlap the score phase).
        const uint16_t* vpa = (kb0 + vklo < kend)
            ? vv + (size_t)(kb0 + vklo) * v_tok + (size_t)h * v_head : nullptr;
        const uint16_t* vpb = (kb0 + vkhi < kend)
            ? vv + (size_t)(kb0 + vkhi) * v_tok + (size_t)h * v_head : nullptr;
        uint4 va[LAW_VPF], vb[LAW_VPF];
#pragma unroll
        for (int it = 0; it < LAW_VPF; ++it) {
            const int g = gbase + 8 * it;
            va[it] = make_uint4(0u, 0u, 0u, 0u);
            vb[it] = make_uint4(0u, 0u, 0u, 0u);
            if (g < ngroups) {
                if (vpa) va[it] = *(const uint4*)(vpa + g * 8);
                if (vpb) vb[it] = *(const uint4*)(vpb + g * 8);
            }
        }

        // ---- partial S^T of key sub-tile `sub`, contraction steps [s0, s1) -------
        const int jr = kb0 + sub * 16 + l16;
        const bool kvalid = jr < kend;
        const uint16_t* kpa0 = ka + (size_t)jr * ka_tok + (size_t)h * ka_head;
        const uint16_t* kpb0 = kb + (size_t)jr * kb_tok + (size_t)h * kb_head;
        laf_f8_t acc_a = laf_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
        laf_f8_t acc_b = acc_a;
        int s = s0;
        for (; s + (LAW_SU - 1) < s1; s += LAW_SU) {
#pragma unroll
            for (int u = 0; u < LAW_SU; ++u) {
                const int d = (s + u) * 16;
                const laf_op_t qf = laf_ldop(qrow + d + LAF_QOFF(hh));
                laf_op_t kf;
                if (kvalid) {
                    kf = (d < wa) ? laf_ldop(kpa0 + d + LAF_QOFF(hh))
                                  : laf_ldop(kpb0 + (d - wa) + LAF_QOFF(hh));
                } else {
#pragma unroll
                    for (int e = 0; e < (LAF_LAYOUT_G11 ? 16 : 8); ++e) kf[e] = 0;
                }
                if ((u & 1) == 0) acc_a = laf_wmma(kf, qf, acc_a);
                else acc_b = laf_wmma(kf, qf, acc_b);
            }
        }
        for (; s < s1; ++s) {
            const int d = s * 16;
            const laf_op_t qf = laf_ldop(qrow + d + LAF_QOFF(hh));
            laf_op_t kf;
            if (kvalid) {
                kf = (d < wa) ? laf_ldop(kpa0 + d + LAF_QOFF(hh))
                              : laf_ldop(kpb0 + (d - wa) + LAF_QOFF(hh));
            } else {
#pragma unroll
                for (int e = 0; e < (LAF_LAYOUT_G11 ? 16 : 8); ++e) kf[e] = 0;
            }
            if (((s - s0) & 1) == 0) acc_a = laf_wmma(kf, qf, acc_a);
            else acc_b = laf_wmma(kf, qf, acc_b);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) sp[(wave * 8 + j) * 32 + lane] = acc_a[j] + acc_b[j];
        __syncthreads();

        // ---- scale, mask, online softmax (identical in all four waves) ------------
        const bool need_mask = kb0 + (LAW_KT - 1) > lim_first;
        float sv[2][8];
        float tmax = -INFINITY;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float sc = (sp[(ks * 8 + j) * 32 + lane] + sp[((2 + ks) * 8 + j) * 32 + lane]) * scale;
                if (need_mask) {
#if LAF_LAYOUT_G11
                    const int kj = kb0 + ks * 16 + 2 * j + hh;
#else
                    const int kj = kb0 + ks * 16 + 8 * hh + j;
#endif
                    sc = (kj <= lim_i) ? sc : -INFINITY;
                }
                sv[ks][j] = sc;
                tmax = fmaxf(tmax, sc);
            }
        }
        tmax = fmaxf(tmax, __shfl_xor(tmax, 16, 32));
        const float m_new = fmaxf(m, tmax);
        const float mshift = (m_new == -INFINITY) ? 0.0f : m_new;
        const float corr = laf_exp(m - mshift);  // m == -inf -> 0 (nothing accumulated yet)
        m = m_new;

        float lsum = 0.0f;
        laf_op_t pf[2];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            uint32_t pw[8];  // rounded probabilities, bf16 in the high half
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                pw[j] = laf_rne_hi(laf_exp(sv[ks][j] - mshift));
                lsum += __uint_as_float(pw[j]);
            }
#if LAF_LAYOUT_G11
            uint32_t own[4], oth[4];
#pragma unroll
            for (int w = 0; w < 4; ++w) own[w] = (pw[2 * w] >> 16) | pw[2 * w + 1];
#pragma unroll
            for (int w = 0; w < 4; ++w) oth[w] = __shfl_xor((int)own[w], 16, 32);
            laf_u8_t W;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                W[2 * w] = hh ? oth[w] : own[w];
                W[2 * w + 1] = hh ? own[w] : oth[w];
            }
            pf[ks] = __builtin_bit_cast(laf_op_t, W);
#else
            uint32_t pk[4];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const int lo = 4 * (w >> 1) + (w & 1);
                pk[w] = (pw[lo] >> 16) | pw[lo + 2];
            }
            const laf_u4_t W = {pk[0], pk[1], pk[2], pk[3]};
            pf[ks] = __builtin_bit_cast(laf_op_t, W);
#endif
        }
        l = l * corr + lsum;
#pragma unroll
        for (int d = 0; d < 8; ++d)
            if (d * 16 < cw) acc_o[d] *= corr;

        // ---- V^T tile into LDS ----------------------------------------------------
#pragma unroll
        for (int it = 0; it < LAW_VPF; ++it) {
            const int g = gbase + 8 * it;
            if (g < ngroups) law_v_store(vs, g, vp, va[it], vb[it]);
        }
        for (int g = gbase + 8 * LAW_VPF; g < ngroups; g += 8) {
            uint4 a = make_uint4(0u, 0u, 0u, 0u), b = a;
            if (vpa) a = *(const uint4*)(vpa + g * 8);
            if (vpb) b = *(const uint4*)(vpb + g * 8);
            law_v_store(vs, g, vp, a, b);
        }
        __syncthreads();

        // ---- O^T += V^T . P^T -------------------------------------------------------
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            if (d * 16 < cw) {
                const uint32_t* arow = vs + (size_t)(dvc0 + d * 16 + l16) * LAW_VP;
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
#if LAF_LAYOUT_G11
                    const uint4 u0 = *(const uint4*)(arow + ks * 8);
                    const uint4 u1 = *(const uint4*)(arow + ks * 8 + 4);
                    const laf_u8_t w = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
                    const laf_op_t af = __builtin_bit_cast(laf_op_t, w);
#else
                    const uint4 u0 = *(const uint4*)(arow + ks * 8 + 4 * hh);
                    const laf_op_t af = __builtin_bit_cast(laf_op_t, u0);
#endif
                    acc_o[d] = laf_wmma(af, pf[ks], acc_o[d]);
                }
            }
        }
        __syncthreads();
    }

    // ---- epilogue: y = O / l (this wave's value columns) -------------------------
    const float l_tot = l + __shfl_xor(l, 16, 32);
    const bool row_ok = row_l < n_rows;
    float* yrow = y + ((size_t)(row_ok ? row_l : 0) * n_heads + h) * dv + dvc0;
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        if (d * 16 >= cw) continue;
        float o[8];
#if LAF_LAYOUT_G11
        float own[4], rcv[4];
#pragma unroll
        for (int mm = 0; mm < 4; ++mm) {
            const float send = hh ? acc_o[d][mm] : acc_o[d][4 + mm];
            own[mm] = hh ? acc_o[d][4 + mm] : acc_o[d][mm];
            rcv[mm] = __shfl_xor(send, 16, 32);
        }
#pragma unroll
        for (int mm = 0; mm < 4; ++mm) {
            o[2 * mm] = hh ? rcv[mm] : own[mm];
            o[2 * mm + 1] = hh ? own[mm] : rcv[mm];
        }
#else
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = acc_o[d][j];
#endif
        float r[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float v = (l_tot > 0.0f) ? o[j] / l_tot : 0.0f;
            if (round_y) v = __uint_as_float(laf_f32_to_bf16(v) << 16);
            r[j] = v;
        }
        if (row_ok) {
            float4* dst = (float4*)(yrow + d * 16 + 8 * hh);
            dst[0] = make_float4(r[0], r[1], r[2], r[3]);
            dst[1] = make_float4(r[4], r[5], r[6], r[7]);
        }
    }
}

// ============================================================================
// Q8 expanded-mirror variant of the chunked kernel above (value width <= any
// multiple of 16, one z block per 128 value columns, same numerics contract).
//
// `ka` (K_nope) and `vv` (V) are INT8 mirror planes (see
// latent_attention_decompress.hip): a plane row is `n_heads * (w + 2)` bytes,
// `[n_heads * w codes | n_heads F16 scales]`, so ka_tok / v_tok are BYTE strides
// of whole plane rows, ka_head / v_head == w, and the scale of (key j, head h)
// is the F16 at byte `j * tok + n_heads * w + 2 * h`. `kb` (the shared tail) is
// the BF16 latent cache as before.
//
// Dequantization: bf16(code * scale), exactly what a BF16 mirror holding the
// dequantized values would feed the matrix units.
//   * K: the 64-key tile of K_nope is dequantized ONCE per block into LDS
//     ([64][wa + 8] BF16) by all threads (every wave then reads its A operands
//     from LDS, so the int8 -> BF16 conversion is not repeated per wave).
//   * V: the int8 rows are dequantized in registers on their way into the
//     transposed LDS tile.
// Requirements (launcher-checked): wa, wb, dv multiples of 16, tok strides and
// the view bases multiples of 16 bytes; dynamic LDS =
// NW * 16 * (dq + 8) * 2 + min(dv, 128) * 36 * 4 + 64 * (wa + 8) * 2 bytes.
// ============================================================================

#ifndef LAF_KPF
#define LAF_KPF 8      // K-dequant rounds prefetched into registers (nthreads == 128: all of them)
#endif

// Signed byte `b` (0..3) of word `w` as float.
static __device__ __forceinline__ float laf_i8(uint32_t w, int b) {
    return (float)((int)(w << (24 - 8 * b)) >> 24);
}

// F16 scale at a byte address.
static __device__ __forceinline__ float laf_f16_at(const uint8_t* p) {
    return (float)(*(const _Float16*)p);
}

// Eight int8 codes (two words) times `s`, rounded to BF16 (RNE), packed as 8 BF16.
static __device__ __forceinline__ uint4 laf_q8_to_bf16x8(uint2 w, float s) {
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t ww = (i < 2) ? w.x : w.y;
        const int b = (i & 1) * 2;
        o[i] = (laf_rne_hi(laf_i8(ww, b) * s) >> 16) | laf_rne_hi(laf_i8(ww, b + 1) * s);
    }
    return make_uint4(o[0], o[1], o[2], o[3]);
}

extern "C" __global__ __launch_bounds__(128) void latent_attention_flash_prefill_q8(
    const float* __restrict__ q,         // [n_rows, H, dq]
    float* __restrict__ y,               // [n_rows, H, dv]
    const uint8_t* __restrict__ ka,      // INT8 K_nope plane view base
    const uint16_t* __restrict__ kb,     // BF16 tail view
    const uint8_t* __restrict__ vv,      // INT8 V plane view base
    const int* __restrict__ pos_ptr,     // device scalar: rows cached before this call
    int n_rows, int n_heads, int wa, int wb, int dv, int pos_offset, int max_rows,
    int ka_tok, int ka_head, int kb_tok, int kb_head, int v_tok, int v_head,
    float scale, int round_y
) {
    extern __shared__ __attribute__((aligned(16))) unsigned char laf_smem[];
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int nw = nthr >> 5;
    const int wave = tid >> 5, lane = tid & 31, l16 = lane & 15, hh = lane >> 4;
    const int dq = wa + wb;
    const int qp = dq + 8;  // LDS pitch of a staged query row (elements)
    const int kp = wa + 8;  // LDS pitch of a staged K row (elements)
    const int dvmax = dv < LAF_DVC ? dv : LAF_DVC;
    uint16_t* qs = (uint16_t*)laf_smem;
    uint32_t* vs = (uint32_t*)(laf_smem + (size_t)nw * 16 * qp * 2);
    uint16_t* ks = (uint16_t*)(laf_smem + (size_t)nw * 16 * qp * 2 + (size_t)dvmax * LAF_VP * 4);

    const int h = blockIdx.y;
    const int dvc0 = blockIdx.z * LAF_DVC;
    const int cw = min(LAF_DVC, dv - dvc0);
    const int ngroups = cw >> 3;
    const int qt = 16 * nw;
    const int tile = (int)gridDim.x - 1 - (int)blockIdx.x;  // heaviest tile first
    const int i0 = tile * qt;

    const int base = *pos_ptr + pos_offset;
    const int wrow0 = i0 + wave * 16;
    const int row_l = wrow0 + l16;
    const int lim_row = min(row_l, n_rows - 1);
    const int lim_i = min(base + lim_row, max_rows - 1);
    const int lim_first = min(base + min(wrow0, n_rows - 1), max_rows - 1);
    const int lim_last = min(base + min(wrow0 + 15, n_rows - 1), max_rows - 1);
    const int kend = max(0, min(base + min(i0 + qt - 1, n_rows - 1), max_rows - 1) + 1);

    // ---- stage this wave's 16 queries as BF16 --------------------------------
    for (int r = 0; r < 16; ++r) {
        const int row = wrow0 + r;
        uint2* dst = (uint2*)(qs + (size_t)(wave * 16 + r) * qp);
        if (row < n_rows) {
            const float4* src = (const float4*)(q + ((size_t)row * n_heads + h) * dq);
            for (int c = lane; c < (dq >> 2); c += 32) {
                const float4 f = src[c];
                dst[c] = make_uint2(laf_f32_to_bf16(f.x) | (laf_f32_to_bf16(f.y) << 16),
                                    laf_f32_to_bf16(f.z) | (laf_f32_to_bf16(f.w) << 16));
            }
        } else {
            for (int c = lane; c < (dq >> 2); c += 32) dst[c] = make_uint2(0u, 0u);
        }
    }
    __syncthreads();

    const uint16_t* qrow = qs + (size_t)(wave * 16 + l16) * qp;

    laf_f8_t acc_o[8];
#pragma unroll
    for (int d = 0; d < 8; ++d) acc_o[d] = laf_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY;
    float l = 0.0f;

    // V^T fill geometry (as the BF16 kernel).
    const int vks = lane >> 3, vq = (lane >> 1) & 3, vr = lane & 1;
    const int vklo = vks * 16 + 4 * vq + vr;
    const int vkhi = vklo + 2;

    // K staging geometry: item i of the tile = (key i >> ncw_sh, 8-column chunk
    // i & (ncw - 1)); the launcher guarantees wa / 8 is a power of two.
    const int ncw = wa >> 3;
    const int ncw_sh = 31 - __builtin_clz(ncw);
    const int nitems = LAF_KT * ncw;
    const size_t ka_sc = (size_t)n_heads * wa + 2 * (size_t)h;   // scale byte offset in a K row
    const size_t v_sc = (size_t)n_heads * dv + 2 * (size_t)h;    // scale byte offset in a V row

    for (int kb0 = 0; kb0 < kend; kb0 += LAF_KT) {

        // V rows of this tile into registers early (int8 + scale); dequantized
        // into LDS after the score math. Beyond kend: never read, zero-filled.
        const uint8_t* vra = (kb0 + vklo < kend) ? vv + (size_t)(kb0 + vklo) * v_tok : nullptr;
        const uint8_t* vrb = (kb0 + vkhi < kend) ? vv + (size_t)(kb0 + vkhi) * v_tok : nullptr;
        const float vsa = vra ? laf_f16_at(vra + v_sc) : 0.0f;
        const float vsb = vrb ? laf_f16_at(vrb + v_sc) : 0.0f;
        uint2 va[LAF_VPF], vb[LAF_VPF];
#pragma unroll
        for (int it = 0; it < LAF_VPF; ++it) {
            const int g = wave + it * nw;
            va[it] = make_uint2(0u, 0u);
            vb[it] = make_uint2(0u, 0u);
            if (g < ngroups) {
                if (vra) va[it] = *(const uint2*)(vra + (size_t)h * v_head + dvc0 + g * 8);
                if (vrb) vb[it] = *(const uint2*)(vrb + (size_t)h * v_head + dvc0 + g * 8);
            }
        }

        // ---- K_nope tile: int8 -> BF16 (x scale) into LDS, once per block -------
        {
            uint2 kraw[LAF_KPF];
            float ksc[LAF_KPF];
#pragma unroll
            for (int it = 0; it < LAF_KPF; ++it) {
                const int i = tid + it * nthr;
                kraw[it] = make_uint2(0u, 0u);
                ksc[it] = 0.0f;
                if (i < nitems) {
                    const int key = i >> ncw_sh, c = i & (ncw - 1);
                    if (kb0 + key < kend) {
                        const uint8_t* kr = ka + (size_t)(kb0 + key) * ka_tok;
                        kraw[it] = *(const uint2*)(kr + (size_t)h * ka_head + c * 8);
                        ksc[it] = laf_f16_at(kr + ka_sc);
                    }
                }
            }
#pragma unroll
            for (int it = 0; it < LAF_KPF; ++it) {
                const int i = tid + it * nthr;
                if (i < nitems) {
                    const int key = i >> ncw_sh, c = i & (ncw - 1);
                    *(uint4*)(ks + (size_t)key * kp + c * 8) = laf_q8_to_bf16x8(kraw[it], ksc[it]);
                }
            }
            for (int i = tid + LAF_KPF * nthr; i < nitems; i += nthr) {
                const int key = i >> ncw_sh, c = i & (ncw - 1);
                uint2 raw = make_uint2(0u, 0u);
                float sc = 0.0f;
                if (kb0 + key < kend) {
                    const uint8_t* kr = ka + (size_t)(kb0 + key) * ka_tok;
                    raw = *(const uint2*)(kr + (size_t)h * ka_head + c * 8);
                    sc = laf_f16_at(kr + ka_sc);
                }
                *(uint4*)(ks + (size_t)key * kp + c * 8) = laf_q8_to_bf16x8(raw, sc);
            }
        }
        __syncthreads();  // K tile visible (previous tile's readers passed the trailing barrier)

        float corr = 1.0f;
        laf_op_t pf[4];
        const bool wave_live = kb0 <= lim_last;
        if (wave_live) {
            // ---- S^T = K . Q^T ---------------------------------------------------
            const int jr = kb0 + l16;
            const uint16_t* kpa0 = ks + (size_t)l16 * kp;
            const uint16_t* kpb0 = kb + (size_t)jr * kb_tok + (size_t)h * kb_head;
            laf_f8_t acc_s[4];
#pragma unroll
            for (int ks_ = 0; ks_ < 4; ++ks_) acc_s[ks_] = laf_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
            laf_qk_pass(acc_s, kpa0, (size_t)16 * kp, kend - jr, qrow, 0, wa >> 4, hh);
            laf_qk_pass(acc_s, kpb0, (size_t)16 * kb_tok, kend - jr, qrow, wa, wb >> 4, hh);

            // ---- scale, mask, online softmax ---------------------------------------
            const bool need_mask = kb0 + (LAF_KT - 1) > lim_first;
            float sv[4][8];
            float tmax = -INFINITY;
#pragma unroll
            for (int ks_ = 0; ks_ < 4; ++ks_) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float s = acc_s[ks_][j] * scale;
                    if (need_mask) {
#if LAF_LAYOUT_G11
                        const int kj = kb0 + ks_ * 16 + 2 * j + hh;
#else
                        const int kj = kb0 + ks_ * 16 + 8 * hh + j;
#endif
                        s = (kj <= lim_i) ? s : -INFINITY;
                    }
                    sv[ks_][j] = s;
                    tmax = fmaxf(tmax, s);
                }
            }
            tmax = fmaxf(tmax, __shfl_xor(tmax, 16, 32));
            const float m_new = fmaxf(m, tmax);
            const float mshift = (m_new == -INFINITY) ? 0.0f : m_new;
            corr = laf_exp(m - mshift);  // m == -inf -> 0 (nothing accumulated yet)
            m = m_new;

            float lsum = 0.0f;
#pragma unroll
            for (int ks_ = 0; ks_ < 4; ++ks_) {
                uint32_t pw[8];  // rounded probabilities, bf16 in the high half
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    pw[j] = laf_rne_hi(laf_exp(sv[ks_][j] - mshift));
                    lsum += __uint_as_float(pw[j]);
                }
#if LAF_LAYOUT_G11
                uint32_t own[4], oth[4];
#pragma unroll
                for (int w = 0; w < 4; ++w)
                    own[w] = (pw[2 * w] >> 16) | pw[2 * w + 1];
#pragma unroll
                for (int w = 0; w < 4; ++w) oth[w] = __shfl_xor((int)own[w], 16, 32);
                laf_u8_t W;
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    W[2 * w] = hh ? oth[w] : own[w];
                    W[2 * w + 1] = hh ? own[w] : oth[w];
                }
                pf[ks_] = __builtin_bit_cast(laf_op_t, W);
#else
                uint32_t pk[4];
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const int lo = 4 * (w >> 1) + (w & 1);
                    pk[w] = (pw[lo] >> 16) | pw[lo + 2];
                }
                const laf_u4_t W = {pk[0], pk[1], pk[2], pk[3]};
                pf[ks_] = __builtin_bit_cast(laf_op_t, W);
#endif
            }
            l = l * corr + lsum;
#pragma unroll
            for (int d = 0; d < 8; ++d)
                if (d * 16 < cw) acc_o[d] *= corr;
        }

        // ---- V^T tile into LDS (dequantized) -------------------------------------
#pragma unroll
        for (int it = 0; it < LAF_VPF; ++it) {
            const int g = wave + it * nw;
            if (g < ngroups)
                laf_v_store(vs, g, lane, laf_q8_to_bf16x8(va[it], vsa), laf_q8_to_bf16x8(vb[it], vsb));
        }
        for (int g = wave + LAF_VPF * nw; g < ngroups; g += nw) {
            uint2 a = make_uint2(0u, 0u), b = a;
            if (vra) a = *(const uint2*)(vra + (size_t)h * v_head + dvc0 + g * 8);
            if (vrb) b = *(const uint2*)(vrb + (size_t)h * v_head + dvc0 + g * 8);
            laf_v_store(vs, g, lane, laf_q8_to_bf16x8(a, vsa), laf_q8_to_bf16x8(b, vsb));
        }
        __syncthreads();

        // ---- O^T += V^T . P^T -----------------------------------------------------
        if (wave_live) {
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                if (d * 16 < cw) {
                    const uint32_t* arow = vs + (size_t)(d * 16 + l16) * LAF_VP;
#pragma unroll
                    for (int ks_ = 0; ks_ < 4; ++ks_) {
                        {
#if LAF_LAYOUT_G11
                            const uint4 u0 = *(const uint4*)(arow + ks_ * 8);
                            const uint4 u1 = *(const uint4*)(arow + ks_ * 8 + 4);
                            const laf_u8_t w = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
                            const laf_op_t af = __builtin_bit_cast(laf_op_t, w);
#else
                            const uint4 u0 = *(const uint4*)(arow + ks_ * 8 + 4 * hh);
                            const laf_op_t af = __builtin_bit_cast(laf_op_t, u0);
#endif
                            acc_o[d] = laf_wmma(af, pf[ks_], acc_o[d]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---- epilogue: y = O / l -------------------------------------------------------
    const float l_tot = l + __shfl_xor(l, 16, 32);
    const bool row_ok = row_l < n_rows;
    float* yrow = y + ((size_t)(row_ok ? row_l : 0) * n_heads + h) * dv + dvc0;
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        if (d * 16 >= cw) continue;
        float o[8];
#if LAF_LAYOUT_G11
        float own[4], rcv[4];
#pragma unroll
        for (int mm = 0; mm < 4; ++mm) {
            const float send = hh ? acc_o[d][mm] : acc_o[d][4 + mm];
            own[mm] = hh ? acc_o[d][4 + mm] : acc_o[d][mm];
            rcv[mm] = __shfl_xor(send, 16, 32);
        }
#pragma unroll
        for (int mm = 0; mm < 4; ++mm) {
            o[2 * mm] = hh ? rcv[mm] : own[mm];
            o[2 * mm + 1] = hh ? own[mm] : rcv[mm];
        }
#else
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = acc_o[d][j];
#endif
        float r[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float v = (l_tot > 0.0f) ? o[j] / l_tot : 0.0f;
            if (round_y) v = __uint_as_float(laf_f32_to_bf16(v) << 16);
            r[j] = v;
        }
        if (row_ok) {
            float4* dst = (float4*)(yrow + d * 16 + 8 * hh);
            dst[0] = make_float4(r[0], r[1], r[2], r[3]);
            dst[1] = make_float4(r[4], r[5], r[6], r[7]);
        }
    }
}
