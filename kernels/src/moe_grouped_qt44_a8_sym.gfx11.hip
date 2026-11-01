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
// - Weights, exchange form (`NX` = false): per K256 group each lane loads the 64 nibble bytes of one of
//   its row's two epochs (lane half hi takes epoch 2g + hi, four b128; the row's whole 136-byte group
//   is requested at once) and the other half-row's lanes take them with `v_permlanex16`. Direct form
//   (`NX` = true, `nt2r2`): the two 16-lane rows of the wave load the same 128 nibble bytes (eight
//   b128, identical addresses coalesce), so the 16 `v_permlanex16` plus 16 select per epoch and
//   fragment disappear (about a quarter of the VALU per WMMA at nt2r2). The K16 B operand is
//   `{lo(pk0), hi(pk0), lo(pk1), hi(pk1)}` of the step's two nibble dwords, which is the nibble-pair
//   byte order of the activations (per 8 elements: e0 e2 e4 e6 | e1 e3 e5 e7), so no byte shuffle
//   exists in the K loop. The int32 chain is exact and order independent.
// - Weights and activations of K256 group g + PD are requested while group g computes (a PD-deep
//   register ring); one barrier per K128 epoch.
//
// The chain seed `-8 * s` is formed once when a tile is published (stored to LDS as `-8 * s`), not per
// lane per epoch; the permlane `old` operand is left undefined (every lane is active and every source
// lane valid), so no copy of the operand is made. Neither changes a value.
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

// Memory accessors. Production builds dereference directly. The bounds guard (lab only,
// `-DA8G11_GUARD`, raon/pfx/ev/xmoe) compiles every global and LDS access through a range check that
// counts a violation per region and skips the access; the index expressions are the same.
enum { A8G11_R_PTRS, A8G11_R_TILES, A8G11_R_OFFS, A8G11_R_SORT, A8G11_R_X, A8G11_R_Y, A8G11_R_W, A8G11_R_LDS_XS, A8G11_R_LDS_DS, A8G11_R_LDS_SS, A8G11_R_N };
#ifdef A8G11_GUARD
__device__ unsigned long long a8g11_glo[A8G11_R_N], a8g11_ghi[A8G11_R_N], a8g11_viol[A8G11_R_N];
__device__ unsigned long long a8g11_wlo[256], a8g11_whi[256];
template <typename T>
__device__ __forceinline__ bool a8g11_ok(const void* p, int r, int e) {
    const unsigned long long a = (unsigned long long)p;
    const unsigned long long lo = r == A8G11_R_W ? a8g11_wlo[e] : a8g11_glo[r], hi = r == A8G11_R_W ? a8g11_whi[e] : a8g11_ghi[r];
    if (a < lo || a + sizeof(T) > hi) {
        atomicAdd(&a8g11_viol[r], 1ull);
        return false;
    }
    return true;
}
template <typename T>
__device__ __forceinline__ T a8g11_ld(const void* p, int r, int e = 0) {
    T z;
    __builtin_memset(&z, 0, sizeof z);
    return a8g11_ok<T>(p, r, e) ? *(const T*)p : z;
}
template <typename T>
__device__ __forceinline__ void a8g11_st(void* p, T v, int r) {
    if (a8g11_ok<T>(p, r, 0)) *(T*)p = v;
}
// LDS: range is the byte window [base, base + size) of the named array.
template <typename T>
__device__ __forceinline__ T a8g11_lds_ld(const void* base, size_t size, const void* p, int r) {
    const unsigned long long a = (unsigned long long)p, b = (unsigned long long)base;
    if (a < b || a + sizeof(T) > b + size) {
        atomicAdd(&a8g11_viol[r], 1ull);
        T z;
        __builtin_memset(&z, 0, sizeof z);
        return z;
    }
    return *(const T*)p;
}
template <typename T>
__device__ __forceinline__ void a8g11_lds_st(const void* base, size_t size, void* p, T v, int r) {
    const unsigned long long a = (unsigned long long)p, b = (unsigned long long)base;
    if (a < b || a + sizeof(T) > b + size) {
        atomicAdd(&a8g11_viol[r], 1ull);
        return;
    }
    *(T*)p = v;
}
#else
template <typename T>
__device__ __forceinline__ T a8g11_ld(const void* p, int, int = 0) {
    return *(const T*)p;
}
template <typename T>
__device__ __forceinline__ void a8g11_st(void* p, T v, int) {
    *(T*)p = v;
}
template <typename T>
__device__ __forceinline__ T a8g11_lds_ld(const void*, size_t, const void* p, int) {
    return *(const T*)p;
}
template <typename T>
__device__ __forceinline__ void a8g11_lds_st(const void*, size_t, void* p, T v, int) {
    *(T*)p = v;
}
#endif

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
    // Value of the same lane index in the opposite 16-lane row. The `old` operand is dead (every lane
    // is active and every source lane valid), so it is left undefined: no copy of `v` is made for it.
    int dead;
    return (unsigned)__builtin_amdgcn_permlanex16(dead, (int)v, 0x76543210, 0xfedcba98, false, false);
}

// One lane's raw share of one weight fragment for one K256 group. Exchange form (NX = false): the 64
// nibble bytes of epoch 2g + hi (16 dwords), the other half-row's lanes take them with
// `v_permlanex16`. Direct form (NX = true): both epochs' 64 bytes (32 dwords), the two 16-lane rows
// of the wave load the same bytes, no exchange. Both epochs' row scales either way.
template <bool NX>
struct a8g11_wraw {
    a8g11_i32x4 v[NX ? 8 : 4];
    float sc0, sc1;
};

template <bool NX>
__device__ __forceinline__ void a8g11_load_w(a8g11_wraw<NX>& o, const unsigned char* row, int e, int g, int hi) {
    const unsigned char* p = row + (size_t)g * A8G11_BLOCK_BYTES;
    const uint2 hdr = a8g11_ld<uint2>(p, A8G11_R_W, e);
    o.sc0 = a8g11_f16_lo(hdr.x);
    o.sc1 = a8g11_f16_lo(hdr.y);
    const unsigned char* q = p + 8 + (NX ? 0 : 64 * hi);
#pragma unroll
    for (int k = 0; k < (NX ? 8 : 4); ++k) o.v[k] = a8g11_ld<a8g11_i32x4>(q + 16 * k, A8G11_R_W, e);
}

// NT 16-slot tiles per run, RF 16-row fragments per wave, PD-deep K256 prefetch ring, NX direct weight
// form (no exchange). One barrier per K128 epoch.
template <int NT, int RF, int PD, bool NX, bool FULL>
__device__ __forceinline__ void a8g11_body(const unsigned char* const (&wrow)[RF], const unsigned char* const (&xsrc)[NT],
                                           const bool (&xlive)[NT], float* __restrict__ Y, int tile, int ntv, int rbase,
                                           int M, int K, int m_total, size_t x_epoch_bytes, int lr, int hi, int t, int ex,
                                           unsigned char* __restrict__ xs, float* __restrict__ ds, int* __restrict__ ss) {
    // LDS plane per buffer: NT tiles x 16 slots x 128 B; scales and (-8 * code sums) NT x 16 each.
    constexpr int XS_TILE = 16 * 128;
    constexpr int NPL = 2;
    const size_t xs_size = (size_t)NPL * NT * XS_TILE, ds_size = (size_t)NPL * NT * 16 * sizeof(float);
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
    a8g11_wraw<NX> wq[PD][RF];
    a8g11_i32x4 xq[PD][2][NT];
    float dq[PD][2][NT];
    int sq[PD][2][NT];

    auto load = [&](int d, int g) {
#pragma unroll
        for (int rf = 0; rf < RF; ++rf) a8g11_load_w<NX>(wq[d][rf], wrow[rf], ex, g, hi);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                if (!FULL && i >= ntv) break;
                if (xlive[i]) {
                    const unsigned char* b = xsrc[i] + (size_t)(2 * g + e) * x_epoch_bytes;
                    xq[d][e][i] = a8g11_ld<a8g11_i32x4>(b + 8 + 16 * xc, A8G11_R_X);
                    const int2 hdr = a8g11_ld<int2>(b, A8G11_R_X);  // [f32 d][i32 s]
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

    // Publish epoch e of ring slot d into LDS plane `pl`.
    auto publish = [&](int d, int e, int pl) {
        unsigned char* xb = xs + pl * (NT * XS_TILE);
        float* db = ds + pl * (NT * 16);
        int* sb = ss + pl * (NT * 16);
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            if (!FULL && i >= ntv) break;
            a8g11_lds_st<a8g11_i32x4>(xs, xs_size, xb + i * XS_TILE + xs_store, xq[d][e][i], A8G11_R_LDS_XS);
            if (xc == 0) {
                a8g11_lds_st<float>(ds, ds_size, db + i * 16 + ds_store, dq[d][e][i], A8G11_R_LDS_DS);
                a8g11_lds_st<int>(ss, ds_size, sb + i * 16 + ds_store, -8 * sq[d][e][i], A8G11_R_LDS_SS);  // chain seed
            }
        }
    };

    // Epoch 2g + e of ring slot d from plane `pl`.
    auto epoch = [&](int d, int e, int g, int pl) {
        unsigned char* xb = xs + pl * (NT * XS_TILE);
        float* db = ds + pl * (NT * 16);
        int* sb = ss + pl * (NT * 16);
        // Weight dword k (0..15) of this epoch's 64 nibble bytes, per fragment.
        unsigned w[RF][NX ? 1 : 16];
        float sc[RF];
#pragma unroll
        for (int rf = 0; rf < RF; ++rf) {
            sc[rf] = e ? wq[d][rf].sc1 : wq[d][rf].sc0;
            if (!NX) {
                // Epoch e's 64 bytes: the lane half hi == e loaded them; the other half takes them from
                // the opposite 16-lane row.
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    const a8g11_i32x4 v = wq[d][rf].v[k >> 2];
                    const unsigned own = (unsigned)((k & 3) == 0 ? v.x : (k & 3) == 1 ? v.y : (k & 3) == 2 ? v.z : v.w);
                    const unsigned other = a8g11_xrow(own);
                    w[rf][NX ? 0 : k] = (hi == e) ? own : other;
                }
            }
        }
        if (!NX && e == 1 && g + PD < groups) load(d, g + PD);
        // Codes enter the WMMA unsigned (0..15); the zero point rides in the chain seed (-8 s, formed
        // when the tile was published): sum_k (u - 8) x = sum_k u x - 8 s, s = sum_k x of the slot (exact int32).
        a8g11_i32x8 c[NT][RF];
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            if (!FULL && i >= ntv) break;
            const a8g11_i32x4 s0 = a8g11_lds_ld<a8g11_i32x4>(ss, ds_size, sb + i * 16 + 8 * hi, A8G11_R_LDS_SS);
            const a8g11_i32x4 s1 = a8g11_lds_ld<a8g11_i32x4>(ss, ds_size, sb + i * 16 + 8 * hi + 4, A8G11_R_LDS_SS);
            const a8g11_i32x8 seed = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
            for (int rf = 0; rf < RF; ++rf) c[i][rf] = seed;
        }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            a8g11_i32x4 a[NT];
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                if (!FULL && i >= ntv) break;
                a[i] = a8g11_lds_ld<a8g11_i32x4>(xs, xs_size, xb + i * XS_TILE + lr * 128 + 16 * (s ^ (lr & 7)), A8G11_R_LDS_XS);
            }
#pragma unroll
            for (int rf = 0; rf < RF; ++rf) {
                unsigned p0, p1;
                if (NX) {
                    const a8g11_i32x4 v = wq[d][rf].v[4 * e + (s >> 1)];
                    p0 = (unsigned)((s & 1) ? v.z : v.x);
                    p1 = (unsigned)((s & 1) ? v.w : v.y);
                } else {
                    p0 = w[rf][NX ? 0 : 2 * s];
                    p1 = w[rf][NX ? 0 : 2 * s + 1];
                }
                const a8g11_i32x4 b = {a8g11_lo(p0), a8g11_hi(p0), a8g11_lo(p1), a8g11_hi(p1)};
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    if (!FULL && i >= ntv) break;
                    c[i][rf] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a[i], false, b, c[i][rf], false);
                }
            }
        }
        if (NX && e == 1 && g + PD < groups) load(d, g + PD);
        // Fold epoch 2g + e: C row 2j + hi is slot 2j + hi of the tile; the row scale is this lane's own.
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            if (!FULL && i >= ntv) break;
            const a8g11_f32x4 d0 = a8g11_lds_ld<a8g11_f32x4>(ds, ds_size, db + i * 16 + 8 * hi, A8G11_R_LDS_DS);
            const a8g11_f32x4 d1 = a8g11_lds_ld<a8g11_f32x4>(ds, ds_size, db + i * 16 + 8 * hi + 4, A8G11_R_LDS_DS);
            const float dj[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
#pragma unroll
            for (int rf = 0; rf < RF; ++rf)
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    sum[i][rf][j] = __fmaf_rn(__fmul_rn(sc[rf], dj[j]), (float)c[i][rf][j], sum[i][rf][j]);
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
                // Epoch 2g + e -> plane parity e (the barrier below also retires every wave's reads
                // of this plane from epoch 2g + e - 2).
                publish(d, e, e);
                __syncthreads();
                epoch(d, e, g, e);
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
                for (int rf = 0; rf < RF; ++rf) a8g11_st<float>(y + rf * 16, sum[i][rf][j], A8G11_R_Y);
            }
        }
    }
}

template <int NT, int RF, int PD, bool NX>
__device__ __forceinline__ void a8g11_moe(
    const unsigned long long* __restrict__ expert_weight_ptrs, const int* __restrict__ expert_tile_ids,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_slot_index,
    const unsigned char* __restrict__ Xq, float* __restrict__ Y, int M, int K, int x_row_div, int m_total,
    int x_src_rows, int n_experts) {
    constexpr int NPL = 2;
    __shared__ __attribute__((aligned(16))) unsigned char xs[NPL * NT * 16 * 128];
    __shared__ __attribute__((aligned(16))) float ds[NPL * NT * 16];
    __shared__ __attribute__((aligned(16))) int ss[NPL * NT * 16];
    const int tile = blockIdx.y;
    if (tile * 16 >= m_total) return;
    const int e = a8g11_ld<int>(expert_tile_ids + tile, A8G11_R_TILES);
    if (e < 0 || e >= n_experts) return;
    const int first = a8g11_ld<int>(expert_offsets + e, A8G11_R_OFFS) >> 4;
    if (((tile - first) % NT) != 0) return;
    const int tiles_total = m_total >> 4;
    int ntv = 1;
    while (ntv < NT && tile + ntv < tiles_total && a8g11_ld<int>(expert_tile_ids + tile + ntv, A8G11_R_TILES) == e) ++ntv;

    const int t = threadIdx.x, wave = t >> 5, lane = t & 31, lr = lane & 15, hi = lane >> 4;
    const int rbase = blockIdx.x * (64 * RF) + wave * (16 * RF);
    const size_t row_bytes = (size_t)(K >> 8) * A8G11_BLOCK_BYTES;
    const unsigned char* W = (const unsigned char*)a8g11_ld<unsigned long long>(expert_weight_ptrs + e, A8G11_R_PTRS);
    const unsigned char* wrow[RF];
#pragma unroll
    for (int rf = 0; rf < RF; ++rf) wrow[rf] = W + (size_t)(rbase + rf * 16 + lr) * row_bytes;

    // Cooperative activation source of this thread: slot (t >> 3) of each tile.
    const unsigned char* xsrc[NT];
    bool xlive[NT];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        int slot = -1;
        if (i < ntv) slot = a8g11_ld<int>(sorted_slot_index + (tile + i) * 16 + (t >> 3), A8G11_R_SORT);
        xlive[i] = slot >= 0;
        xsrc[i] = Xq + (size_t)(xlive[i] ? slot / x_row_div : 0) * A8G11_BLOCK_BYTES;
    }
    const size_t x_epoch_bytes = (size_t)x_src_rows * A8G11_BLOCK_BYTES;
    if (ntv == NT) {
        a8g11_body<NT, RF, PD, NX, true>(wrow, xsrc, xlive, Y, tile, ntv, rbase, M, K, m_total, x_epoch_bytes, lr, hi, t, e, xs,
                                             ds, ss);
    } else {
        a8g11_body<NT, RF, PD, NX, false>(wrow, xsrc, xlive, Y, tile, ntv, rbase, M, K, m_total, x_epoch_bytes, lr, hi, t, e,
                                              xs, ds, ss);
    }
}

#define A8G11_ENTRY(NAME, NT, RF, PD, NX)                                                                     \
    extern "C" __global__ __launch_bounds__(128) void moe_grouped_qt44_a8_sym_gfx11_##NAME(                       \
        const unsigned long long* __restrict__ expert_weight_ptrs, const int* __restrict__ expert_tile_ids,       \
        const int* __restrict__ expert_offsets, const int* __restrict__ sorted_slot_index,                        \
        const unsigned char* __restrict__ Xq, float* __restrict__ Y, int M, int K, int x_row_div,                 \
        int m_total, int x_src_rows, int n_experts) {                                                             \
        a8g11_moe<NT, RF, PD, NX>(expert_weight_ptrs, expert_tile_ids, expert_offsets, sorted_slot_index, Xq, \
                                      Y, M, K, x_row_div, m_total, x_src_rows, n_experts);                        \
    }

// (name, NT, RF, prefetch depth PD in K256 groups, NX direct weights)
A8G11_ENTRY(nt1r1, 1, 1, 2, false)
A8G11_ENTRY(nt2r1, 2, 1, 2, false)
A8G11_ENTRY(nt4r1, 4, 1, 1, false)
A8G11_ENTRY(nt1r2, 1, 2, 2, false)
A8G11_ENTRY(nt2r2, 2, 2, 1, true)
