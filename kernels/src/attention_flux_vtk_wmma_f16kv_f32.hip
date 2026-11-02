// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.
//
// Non-causal flash attention specialised to the FLUX.1-dev MMDiT shape:
// head_dim == 128, K/V f16, no mask, no KV cache.
//
// Q and out dtypes are template parameters, exactly as in the base `vt`
// kernel: `QT` x `OT` over {float, _Float16}, four entries, one kernarg
// layout. See attention_flux_vt_wmma_f16kv_f32.hip for why that is a surface
// change and not an algorithm change.
//
// Written against the two defects the earlier v5/v6/v7 family shared:
//
//   1. The PV WMMA B-fragment was built with 16 *scalar* strided LDS reads
//      (`V_lds[(s_col + j) * head_dim + my_d]`) — 16 `ds_read_u16` with bank
//      conflicts per single WMMA. Here V is **transposed while it is staged**
//      into LDS (`Vt[d][k]`), so the same fragment is 16 contiguous halves:
//      two `ds_read_b128`. 8x fewer LDS instructions on the hot path.
//   2. The online softmax round-tripped S through LDS and then ran a
//      *sequential* 16-iteration loop of wave-wide shuffle reductions, i.e.
//      16 dependent 5-shuffle chains + 64 LDS reads + 64 LDS writes per tile.
//      Here the running max/sum live in **registers**: the WMMA accumulator
//      layout puts each lane's 8 C values on 8 distinct rows, all 16 lanes of
//      a half-wave share a row, so a row reduction is 4 `shfl_xor` (masks
//      1/2/4/8) done for 8 independent rows at once. S is written to LDS once
//      (already exponentiated) and never read back for the softmax. m/l/alpha
//      never touch LDS, so the per-tile barrier around them disappears too.
//
// Layout notes (LDS banking, 32 banks x 4 B):
//   * `Vt` row stride 40 halves (80 B): the PV fragment read has 16 distinct
//     lanes x 8 dwords = 128 dwords spread over all 32 banks exactly 4x, i.e.
//     the 4-cycle floor. 80 B is 16 B aligned so `ds_read_b128` is legal.
//   * `Slds` row stride 72 halves (144 B): same property for the A-fragment.
//   * Each wave owns query rows [wave*16, wave*16+16) in *both* phases, and
//     reads back only its own S rows, so S needs no cross-wave barrier.
//
// Grid  = [n_heads, ceil(B/64)], block = [128] (4 waves), dynamic LDS 19456 B
// (19 KB -> 3 workgroups/CU on a 64 KB-per-CU RDNA3/3.5/4 part).

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef _Float16 __attribute__((ext_vector_type(4))) half4_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

#define AV_HD 128
#define AV_DC 8   // AV_HD / 16
#define AV_MT 64  // query rows per block
#define AV_NT 64  // keys per online-softmax tile
#define AV_NC 4   // AV_NT / 16
#define AV_VCHUNK 32
#define AV_VNC 2                            // AV_VCHUNK / 16
#define AV_VT_STRIDE 40                     // halves; 80 B, 16 B aligned
#define AV_K_STRIDE 136                     // halves; 272 B, 16 B aligned
#define AV_S_STRIDE 72                      // halves; 144 B, 16 B aligned
#define AV_VT_HALVES (AV_HD * AV_VT_STRIDE)  // 5120; also covers K (32*136=4352)
#define AV_LDS_BYTES ((AV_VT_HALVES + AV_MT * AV_S_STRIDE) * 2)  // 19456

// 16-byte-aligned pair load; guarantees ds_read_b128 / global_load_dwordx4
// rather than letting the compiler fall back to narrow scalar reads.
__device__ __forceinline__ half16_t av_ld16(const _Float16 *p) {
    const half8_t *q = (const half8_t *)p;
    half8_t a = q[0];
    half8_t b = q[1];
    half16_t r;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        r[i] = a[i];
        r[i + 8] = b[i];
    }
    return r;
}

// Epilogue store, specialised on the out dtype. KEEP IN SYNC with the copy in
// attention_flux_vt_wmma_f16kv_f32.hip (and the gfx12 sibling) — the empty asm
// in the `_Float16` specialisation is LOAD-BEARING, not a leftover. It pins the
// f32 product in a VGPR so the narrowing stays a separate RNE rounding; without
// it the backend folds `fptrunc (fmul a, b)` into one `v_fma_mixlo_f16` and the
// f16 entry stops being bit-identical to "f32 entry + RNE cast". See
// attention_flux_vt_wmma_f16kv_f32.hip for the full note.
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
av_vtk_body(const QT *__restrict__ q, const _Float16 *__restrict__ k,
            const _Float16 *__restrict__ v, OT *__restrict__ out, int B, int L,
            int n_heads, int n_kv_heads, int head_dim, float scale,
            int qmajor) {
    // See the base kernel: qmajor puts the query tile on the fast-varying grid
    // axis so the resident workgroups span one head's K/V, not all of them.
    const int head = qmajor ? blockIdx.y : blockIdx.x;
    const int q_start = (qmajor ? blockIdx.x : blockIdx.y) * AV_MT;
    const int tid = threadIdx.x;

    if (head >= n_heads || q_start >= B) return;
    if (head_dim != AV_HD) return;

    const int rep = n_heads / n_kv_heads;
    const int kv_head = head / rep;
    const int q_stride = n_heads * AV_HD;
    const int kv_stride = n_kv_heads * AV_HD;

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int h = lane & 15;   // WMMA a-row / b-col selector
    const int hi = lane >> 4;  // WMMA c-row parity
    const int rb = wave * 16;  // this wave's query-row base inside the tile

    // half8_t element type so the dynamic-LDS base is 16 B aligned, which is
    // what ds_read_b128 / ds_write_b128 require.
    // One staging buffer, used as K[key][d] during phase A and as V^T[d][key]
    // during phase C. The phases are already separated by a barrier, so the
    // alias is free and total LDS is unchanged at 19456 B (3 workgroups/CU).
    extern __shared__ half8_t av_smem[];
    _Float16 *Stg = (_Float16 *)av_smem;   // [32][AV_K_STRIDE] or [AV_HD][AV_VT_STRIDE]
    _Float16 *Vt = Stg;
    _Float16 *Kl = Stg;
    _Float16 *Slds = Stg + AV_VT_HALVES;   // [AV_MT][AV_S_STRIDE]

    // 32-bit index arithmetic throughout: the largest tensor this kernel is
    // used on is far below 2^31 elements, and size_t offsets cost a
    // v_mad_i64_i32 + v_lshlrev_b64 + carry-add per address.
    const _Float16 *kbase = k + kv_head * AV_HD;
    const _Float16 *vbase = v + kv_head * AV_HD;

    // ---- Q fragments, resident for the whole kernel (8 x half16 = 64 VGPR) --
    // Rows past B are clamped, not predicated: their outputs are dropped in the
    // epilogue and softmax is row-independent, so a duplicate row is harmless
    // and costs no divergent branch.
    half16_t Qf[AV_DC];
    {
        const int gq = min(q_start + rb + h, B - 1);
        const QT *qr = q + gq * q_stride + head * AV_HD;
#pragma unroll
        for (int dc = 0; dc < AV_DC; ++dc) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                Qf[dc][j] = (_Float16)qr[dc * 16 + j];
            }
        }
    }

    float8_t Of[AV_DC];
#pragma unroll
    for (int dc = 0; dc < AV_DC; ++dc) {
        Of[dc] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    // Running softmax state. Lane (h, hi) owns rows rb + 2*j + hi, j = 0..7 —
    // exactly the rows its WMMA accumulator holds. All 16 h-lanes of a
    // half-wave hold a redundant copy of the same row's state.
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
        // K is staged through LDS rather than read per wave from global. Read
        // straight from global, all four waves of the block issue the same 16
        // cache lines per fragment (each b128 gathers 16 rows 6 KB apart), so
        // the block requests the K tile four times over. Staging makes the
        // global side coalesced and read exactly once.
#pragma unroll
        for (int kc = 0; kc < AV_NT / 32; ++kc) {
            __syncthreads();  // Stg still holds the previous tile's V^T
            {
                // Wave w stages keys [32*kc + 8w, +8); lane covers d =
                // 4*lane..+3, so the global read is a fully coalesced 256 B per
                // instruction and the LDS write is one ds_write_b64.
                const int kk0 = wave * 8;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int gk = min(kt + kc * 32 + kk0 + r, L - 1);
                    const half4_t src =
                        *(const half4_t *)(kbase + gk * kv_stride + 4 * lane);
                    *(half4_t *)(Kl + (kk0 + r) * AV_K_STRIDE + 4 * lane) = src;
                }
            }
            __syncthreads();

#pragma unroll
            for (int dc = 0; dc < AV_DC; ++dc) {
#pragma unroll
                for (int cl = 0; cl < 2; ++cl) {
                    const half16_t breg =
                        av_ld16(Kl + (cl * 16 + h) * AV_K_STRIDE + dc * 16);
                    const int c = kc * 2 + cl;
                    sa[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        Qf[dc], breg, sa[c]);
                }
            }
        }

        // ---------------- Phase B: online softmax, in registers -------------
        float cmask[AV_NC];
#pragma unroll
        for (int c = 0; c < AV_NC; ++c) {
            cmask[c] = ((kt + c * 16 + h) < L) ? 0.0f : -INFINITY;
        }

        // Fold the softmax scale and the column mask into the accumulator once
        // (in place) so neither the row max nor the exponential repeats it.
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
        // Row reduction over the 16 lanes that share `hi`; masks 1/2/4/8 stay
        // inside the half-wave. 8 independent chains -> good ILP.
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
            alpha[j] = (m_run[j] == -INFINITY) ? 0.0f
                                               : __expf(m_run[j] - mnew[j]);
            m_run[j] = mnew[j];
        }

        float rs[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) rs[j] = 0.0f;
#pragma unroll
        for (int c = 0; c < AV_NC; ++c) {
            _Float16 *srow = Slds + (rb + hi) * AV_S_STRIDE + c * 16 + h;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float e = __expf(sa[c][j] - mnew[j]);
                rs[j] += e;
                srow[(2 * j) * AV_S_STRIDE] = (_Float16)e;
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
            __syncthreads();  // previous chunk's Vt fully consumed
            {
                // Stage Vt[d][kk] for kk in [0, AV_VCHUNK). Wave w owns the 8
                // keys [8w, 8w+8); lane covers d = lane, +32, +64, +96 so the
                // 8-key run is contiguous in Vt and lands as one ds_write_b128
                // per d-slot, with the 32 lanes' 4-dword writes spread evenly
                // over the banks.
                const int kk0 = wave * 8;
                half8_t col[4];
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int gk =
                        min(kt + vc * AV_VCHUNK + kk0 + r, L - 1);
                    const _Float16 *vr = vbase + gk * kv_stride;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        col[i][r] = vr[i * 32 + lane];
                    }
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
                const half16_t areg =
                    av_ld16(Slds + (rb + h) * AV_S_STRIDE + c * 16);
#pragma unroll
                for (int dc = 0; dc < AV_DC; ++dc) {
                    const half16_t breg =
                        av_ld16(Vt + (dc * 16 + h) * AV_VT_STRIDE + nl * 16);
                    Of[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        areg, breg, Of[dc]);
                }
            }
        }
    }

    // ---------------- Epilogue ----------------
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float lv = l_run[j];
        l_run[j] = (lv > 0.0f) ? (1.0f / lv) : 0.0f;
    }
    // j outer so the `gq < B` predicate is evaluated 8 times, not 64.
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int gq = q_start + rb + 2 * j + hi;
        if (gq < B) {
            OT *orow = out + gq * q_stride + head * AV_HD + h;
#pragma unroll
            for (int dc = 0; dc < AV_DC; ++dc) {
                av_store<OT>::put(&orow[dc * 16], Of[dc][j] * l_run[j]);
            }
        }
    }
}

// One entry per {Q dtype} x {out dtype}; see the base `vt` kernel.
#define AV_VTK_ENTRY(SUFFIX, QT, OT)                                          \
    extern "C" __global__ void __launch_bounds__(128, 2)                      \
        attention_flux_vtk_wmma_f16kv_f32##SUFFIX(                            \
            const QT *__restrict__ q, const _Float16 *__restrict__ k,         \
            const _Float16 *__restrict__ v, OT *__restrict__ out, int B,      \
            int L, int n_heads, int n_kv_heads, int head_dim, float scale,    \
            int qmajor) {                                                     \
        av_vtk_body<QT, OT>(q, k, v, out, B, L, n_heads, n_kv_heads, head_dim, \
                            scale, qmajor);                                   \
    }

AV_VTK_ENTRY(, float, float)
AV_VTK_ENTRY(_qf16, _Float16, float)
AV_VTK_ENTRY(_of16, float, _Float16)
AV_VTK_ENTRY(_qf16_of16, _Float16, _Float16)
