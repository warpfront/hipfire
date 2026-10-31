// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Grouped symmetric MQ4G256V2 (qt44) experts x int8 activations (W4A8), gfx11 (RDNA3: gfx1100,
// gfx1151; wave32 `v_wmma_i32_16x16x16_iu8`). The gfx11 twin of `moe_grouped_qt44_a8_sym.gfx12.hip`:
// same grouped ABI and kernargs, same tile-run scheme, same `block_i8_128` activation sidecar of
// `quantize_int8_mmq_ds128` (136 B per K128: `[f32 d][i32 s][128 i8]`, epoch-major
// `[K/128, x_src_rows]`), same F32 output and the same fold, so for equal inputs the output is
// bit-identical to the gfx12 kernel's. Weights are symmetric (`zp == -8 * sc`, `w = sc * (u - 8)`).
// The codes `u` enter the WMMA raw and UNSIGNED (0..15) against the signed int8 activations, and the
// zero point rides in the chain seed `-8 * s` (`s` = the slot's exact int32 code sum from the
// sidecar header), so the chain equals the gfx12 kernel's signed `sum (u - 8) * x` exactly. Per
// output and ascending K128 epoch h:
//   C_h = -8 * s_h + exact int32 sum_k128(X_signed * U_unsigned)   (8 K16 WMMA steps)
//   sum = fma(RN(sc_h * d_h), float(C_h), sum), sum starts +0
//
// RDNA3 WMMA operand form (wave32): A and B are 16 int8 per lane (one K16 step), lanes r and r + 16
// carry identical operands, and C element j of lane L is C[2j + (L >> 4)][L & 15]. The kernel puts
// the ACTIVATIONS on A (C rows = grouped token slots) and the expanded WEIGHTS on B (C columns =
// weight rows), as `gemm_mq4g256v2_residual_mmq_i8.gfx11.hip` does, so the row scale is the lane's
// own and each store instruction writes two 64-byte runs of `Y = [m_total][M]`.
//
// Data movement per K128 epoch:
// - Activations: the CTA's `NT` 16-slot tiles (2 KB each) are loaded once per CTA, one 16-byte chunk
//   per thread per tile, into a double-buffered LDS plane (chunk c of slot e at chunk c ^ (e & 7)),
//   and every wave reads its A operands from it (one b128 per step and tile). Slot scales `d` and
//   code sums `s` go to LDS in C-row order (slot 2j + hi at index 8 * hi + j), two b128 reads each
//   per tile.
// - Weights: per K256 group each lane loads the 64 nibble bytes of one of its row's two epochs (lane
//   half hi takes epoch 2g + hi, four b128; the row's whole 136-byte group is requested at once) and
//   the other half-row's lanes take them with `v_permlanex16`; the K16 B operand is
//   `{lo(pk0), hi(pk0), lo(pk1), hi(pk1)}` of the step's two nibble dwords, which is the nibble-pair
//   byte order of the activations (per 8 elements: e0 e2 e4 e6 | e1 e3 e5 e7), so no byte shuffle
//   exists in the K loop. The int32 chain is exact and order independent.
// - Weights and activations of K256 group g + PD are requested while group g computes (a PD-deep
//   register ring); one barrier per K128 epoch.
//
// Tiling: 4 waves per CTA, `RF` 16-row weight fragments per wave (CTA = 64 * RF weight rows), `NT`
// 16-slot tiles of one expert run per CTA (`NT`-aligned runs from `expert_offsets`).

#include <hip/hip_runtime.h>
#include <stdint.h>

#if !defined(__gfx1100__) && !defined(__gfx1151__) && defined(__HIP_DEVICE_COMPILE__)
#error "moe_grouped_qt44_a8_sym.gfx11.hip is gfx1100 / gfx1151 only"
#endif

typedef int a8g11_i32x4 __attribute__((ext_vector_type(4)));
typedef int a8g11_i32x8 __attribute__((ext_vector_type(8)));
typedef float a8g11_f32x4 __attribute__((ext_vector_type(4)));

#define A8G11_BLOCK_BYTES 136

__device__ __forceinline__ float a8g11_f16_lo(unsigned int w) {
    return (float)__builtin_bit_cast(_Float16, (unsigned short)(w & 0xFFFFu));
}

// One nibble dword -> the low (elements 0,2,4,6) / high (1,3,5,7) UNSIGNED code dword (0..15 per byte).
__device__ __forceinline__ int a8g11_lo(unsigned w) {
    return (int)(w & 0x0F0F0F0Fu);
}
__device__ __forceinline__ int a8g11_hi(unsigned w) {
    return (int)((w >> 4) & 0x0F0F0F0Fu);
}

__device__ __forceinline__ unsigned a8g11_xrow(unsigned v) {
    // Value of the same lane index in the opposite 16-lane row.
    return (unsigned)__builtin_amdgcn_permlanex16((int)v, (int)v, 0x76543210, 0xfedcba98, false, false);
}

// One lane's raw share of one weight fragment for one K256 group: the 64 nibble bytes of epoch
// 2g + hi (16 dwords), plus both epochs' row scales.
struct a8g11_wraw {
    a8g11_i32x4 v[4];
    float sc0, sc1;
};

__device__ __forceinline__ void a8g11_load_w(a8g11_wraw& o, const unsigned char* row, int g, int hi) {
    const unsigned char* p = row + (size_t)g * A8G11_BLOCK_BYTES;
    const uint2 hdr = *(const uint2*)p;
    o.sc0 = a8g11_f16_lo(hdr.x);
    o.sc1 = a8g11_f16_lo(hdr.y);
    const unsigned char* q = p + 8 + 64 * hi;
#pragma unroll
    for (int k = 0; k < 4; ++k) o.v[k] = *(const a8g11_i32x4*)(q + 16 * k);
}

template <int NT, int RF, int PD, bool FULL>
__device__ __forceinline__ void a8g11_body(const unsigned char* const (&wrow)[RF], const unsigned char* const (&xsrc)[NT],
                                           const bool (&xlive)[NT], float* __restrict__ Y, int tile, int ntv, int rbase,
                                           int M, int K, int m_total, size_t x_epoch_bytes, int lr, int hi, int t,
                                           unsigned char* __restrict__ xs, float* __restrict__ ds, int* __restrict__ ss) {
    // LDS plane per buffer: NT tiles x 16 slots x 128 B; scales and code sums NT x 16 each.
    constexpr int XS_TILE = 16 * 128;
    const int xe = t >> 3, xc = t & 7;                  // cooperative load: slot, 16-byte chunk
    const int xs_store = xe * 128 + 16 * (xc ^ (xe & 7));
    const int ds_store = (xe & 1) * 8 + (xe >> 1);       // C-row order

    float sum[NT][RF][8];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int rf = 0; rf < RF; ++rf)
#pragma unroll
            for (int j = 0; j < 8; ++j) sum[i][rf][j] = 0.0f;

    const int groups = K >> 8;
    // Prefetch ring: group g's weights and activations (both of its K128 epochs) sit in slot g % PD
    // and are requested PD groups before they are consumed (the loop below is unrolled by PD so slots
    // index statically). Each row's whole 136-byte group is requested at once.
    a8g11_wraw wq[PD][RF];
    a8g11_i32x4 xq[PD][2][NT];
    float dq[PD][2][NT];
    int sq[PD][2][NT];

    auto load = [&](int d, int g) {
#pragma unroll
        for (int rf = 0; rf < RF; ++rf) a8g11_load_w(wq[d][rf], wrow[rf], g, hi);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                if (!FULL && i >= ntv) break;
                if (xlive[i]) {
                    const unsigned char* b = xsrc[i] + (size_t)(2 * g + e) * x_epoch_bytes;
                    xq[d][e][i] = *(const a8g11_i32x4*)(b + 8 + 16 * xc);
                    const int2 hdr = *(const int2*)b;  // [f32 d][i32 s]
                    dq[d][e][i] = __int_as_float(hdr.x);
                    sq[d][e][i] = hdr.y;
                } else {
                    xq[d][e][i] = (a8g11_i32x4){0, 0, 0, 0};
                    dq[d][e][i] = 0.0f;
                    sq[d][e][i] = 0;
                }
            }
        }
    };

#pragma unroll
    for (int d = 0; d < PD; ++d)
        if (d < groups) load(d, d);
    for (int g0 = 0; g0 < groups; g0 += PD) {
#pragma unroll
        for (int d = 0; d < PD; ++d) {
            const int g = g0 + d;
            if (g >= groups) break;
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int buf = e;  // epoch 2g + e -> buffer parity e
                unsigned char* xb = xs + buf * (NT * XS_TILE);
                float* db = ds + buf * (NT * 16);
                int* sb = ss + buf * (NT * 16);
                // Publish epoch 2g + e's activation tiles (the barrier below also retires every wave's
                // reads of this buffer from epoch 2g + e - 2).
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    if (!FULL && i >= ntv) break;
                    *(a8g11_i32x4*)(xb + i * XS_TILE + xs_store) = xq[d][e][i];
                    if (xc == 0) {
                        db[i * 16 + ds_store] = dq[d][e][i];
                        sb[i * 16 + ds_store] = sq[d][e][i];
                    }
                }
                __syncthreads();
                // Epoch e's 64 bytes of each fragment row: the lane half hi == e loaded them; the
                // other half takes them from the opposite 16-lane row.
                unsigned w[RF][16];
                float sc[RF];
#pragma unroll
                for (int rf = 0; rf < RF; ++rf) {
                    sc[rf] = e ? wq[d][rf].sc1 : wq[d][rf].sc0;
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        const a8g11_i32x4 v = wq[d][rf].v[k >> 2];
                        const unsigned own = (unsigned)((k & 3) == 0 ? v.x : (k & 3) == 1 ? v.y : (k & 3) == 2 ? v.z : v.w);
                        const unsigned other = a8g11_xrow(own);
                        w[rf][k] = (hi == e) ? own : other;
                    }
                }
                if (e == 1 && g + PD < groups) load(d, g + PD);
                // Codes enter the WMMA unsigned (0..15); the zero point rides in the chain seed:
                // sum_k (u - 8) x = sum_k u x - 8 s, s = sum_k x of the slot (exact int32).
                a8g11_i32x8 c[NT][RF];
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    if (!FULL && i >= ntv) break;
                    const a8g11_i32x4 s0 = *(const a8g11_i32x4*)(sb + i * 16 + 8 * hi);
                    const a8g11_i32x4 s1 = *(const a8g11_i32x4*)(sb + i * 16 + 8 * hi + 4);
                    const a8g11_i32x8 seed = {-8 * s0.x, -8 * s0.y, -8 * s0.z, -8 * s0.w,
                                              -8 * s1.x, -8 * s1.y, -8 * s1.z, -8 * s1.w};
#pragma unroll
                    for (int rf = 0; rf < RF; ++rf) c[i][rf] = seed;
                }
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    a8g11_i32x4 a[NT];
#pragma unroll
                    for (int i = 0; i < NT; ++i) {
                        if (!FULL && i >= ntv) break;
                        a[i] = *(const a8g11_i32x4*)(xb + i * XS_TILE + lr * 128 + 16 * (s ^ (lr & 7)));
                    }
#pragma unroll
                    for (int rf = 0; rf < RF; ++rf) {
                        const unsigned p0 = w[rf][2 * s], p1 = w[rf][2 * s + 1];
                        const a8g11_i32x4 b = {a8g11_lo(p0), a8g11_hi(p0), a8g11_lo(p1), a8g11_hi(p1)};
#pragma unroll
                        for (int i = 0; i < NT; ++i) {
                            if (!FULL && i >= ntv) break;
                            c[i][rf] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a[i], false, b, c[i][rf], false);
                        }
                    }
                }
                // Fold epoch 2g + e: C row 2j + hi is slot 2j + hi of the tile; the row scale is this
                // lane's own.
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    if (!FULL && i >= ntv) break;
                    const a8g11_f32x4 d0 = *(const a8g11_f32x4*)(db + i * 16 + 8 * hi);
                    const a8g11_f32x4 d1 = *(const a8g11_f32x4*)(db + i * 16 + 8 * hi + 4);
                    const float dj[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
#pragma unroll
                    for (int rf = 0; rf < RF; ++rf)
#pragma unroll
                        for (int j = 0; j < 8; ++j)
                            sum[i][rf][j] = __fmaf_rn(__fmul_rn(sc[rf], dj[j]), (float)c[i][rf][j], sum[i][rf][j]);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        if (!FULL && i >= ntv) break;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int pos = (tile + i) * 16 + 2 * j + hi;
            if (pos < m_total) {
                float* y = Y + (size_t)pos * M + rbase + lr;
#pragma unroll
                for (int rf = 0; rf < RF; ++rf) y[rf * 16] = sum[i][rf][j];
            }
        }
    }
}

template <int NT, int RF, int PD>
__device__ __forceinline__ void a8g11_moe(
    const unsigned long long* __restrict__ expert_weight_ptrs, const int* __restrict__ expert_tile_ids,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_slot_index,
    const unsigned char* __restrict__ Xq, float* __restrict__ Y, int M, int K, int x_row_div, int m_total,
    int x_src_rows, int n_experts) {
    __shared__ __attribute__((aligned(16))) unsigned char xs[2 * NT * 16 * 128];
    __shared__ __attribute__((aligned(16))) float ds[2 * NT * 16];
    __shared__ __attribute__((aligned(16))) int ss[2 * NT * 16];
    const int tile = blockIdx.y;
    if (tile * 16 >= m_total) return;
    const int e = expert_tile_ids[tile];
    if (e < 0 || e >= n_experts) return;
    const int first = expert_offsets[e] >> 4;
    if (((tile - first) % NT) != 0) return;
    const int tiles_total = m_total >> 4;
    int ntv = 1;
    while (ntv < NT && tile + ntv < tiles_total && expert_tile_ids[tile + ntv] == e) ++ntv;

    const int t = threadIdx.x, wave = t >> 5, lane = t & 31, lr = lane & 15, hi = lane >> 4;
    const int rbase = blockIdx.x * (64 * RF) + wave * (16 * RF);
    const size_t row_bytes = (size_t)(K >> 8) * A8G11_BLOCK_BYTES;
    const unsigned char* W = (const unsigned char*)expert_weight_ptrs[e];
    const unsigned char* wrow[RF];
#pragma unroll
    for (int rf = 0; rf < RF; ++rf) wrow[rf] = W + (size_t)(rbase + rf * 16 + lr) * row_bytes;

    // Cooperative activation source of this thread: slot (t >> 3) of each tile.
    const unsigned char* xsrc[NT];
    bool xlive[NT];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        int slot = -1;
        if (i < ntv) slot = sorted_slot_index[(tile + i) * 16 + (t >> 3)];
        xlive[i] = slot >= 0;
        xsrc[i] = Xq + (size_t)(xlive[i] ? slot / x_row_div : 0) * A8G11_BLOCK_BYTES;
    }
    const size_t x_epoch_bytes = (size_t)x_src_rows * A8G11_BLOCK_BYTES;
    if (ntv == NT) {
        a8g11_body<NT, RF, PD, true>(wrow, xsrc, xlive, Y, tile, ntv, rbase, M, K, m_total, x_epoch_bytes, lr, hi, t, xs, ds,
                                     ss);
    } else {
        a8g11_body<NT, RF, PD, false>(wrow, xsrc, xlive, Y, tile, ntv, rbase, M, K, m_total, x_epoch_bytes, lr, hi, t, xs,
                                      ds, ss);
    }
}

#define A8G11_ENTRY(NT, RF, PD)                                                                                 \
    extern "C" __global__ __launch_bounds__(128) void moe_grouped_qt44_a8_sym_gfx11_nt##NT##r##RF(          \
        const unsigned long long* __restrict__ expert_weight_ptrs, const int* __restrict__ expert_tile_ids, \
        const int* __restrict__ expert_offsets, const int* __restrict__ sorted_slot_index,                  \
        const unsigned char* __restrict__ Xq, float* __restrict__ Y, int M, int K, int x_row_div,           \
        int m_total, int x_src_rows, int n_experts) {                                                       \
        a8g11_moe<NT, RF, PD>(expert_weight_ptrs, expert_tile_ids, expert_offsets, sorted_slot_index, Xq, Y, M, \
                          K, x_row_div, m_total, x_src_rows, n_experts);                                    \
    }

// (NT, RF, prefetch depth PD in K256 groups)
A8G11_ENTRY(1, 1, 2)
A8G11_ENTRY(2, 1, 2)
A8G11_ENTRY(4, 1, 1)
A8G11_ENTRY(1, 2, 2)
A8G11_ENTRY(2, 2, 1)
