// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniil Markevich
// hipfire — see LICENSE and NOTICE in the project root.
// Restored from PR #616 (473efc835): gfx1100 X256/Y64, permuted-nibble,
// Quad-row-u32x2 layout, with native Q8-group32 activation precision.
// Checkpoint layout remains unchanged.
// Compile-time experimental branches have been pruned from the old source.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>
struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * 32];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");
struct block_q8_1_group128_compact {
    half2 d_sum0;
    half2 sum1_sum2;
    half2 sum3_pad;
    int8_t qs[4 * 32];
};
static_assert(sizeof(block_q8_1_group128_compact) == 140,
              "bad compact group128 size");

extern "C" __global__ void quantize_mq4_packed_group128(
    const float* __restrict__ X,
    block_q8_1_mmq* __restrict__ Y,
    int K,
    int N
) {
    const int token = blockIdx.y;
    const int i0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (token >= N || i0 >= K) return;
    const float4 xi = (i0 + 3 < K)
        ? *(const float4*)(X + (long long)token * K + i0)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)),
                       fmaxf(fabsf(xi.z), fabsf(xi.w)));
    // Eight lanes load 32 values. The symbol retains its legacy name, but
    // each of the four DS4 entries now carries its own native-MMQ scale.
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }
    float sum = xi.x + xi.y + xi.z + xi.w;
    for (int offset = 4; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, 32);
    }
    const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    char4 q;
    q.x = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.x * id)));
    q.y = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.y * id)));
    q.z = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.z * id)));
    q.w = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.w * id)));
    const int kblock = i0 / 128;
    const int iqs = i0 & 127;
    block_q8_1_mmq* out = Y + (long long)kblock * N + token;
    ((char4*)out->qs)[iqs / 4] = q;
    if ((iqs & 31) == 0) {
        out->ds4[iqs / 32] = make_half2((_Float16)d, (_Float16)sum);
    }
}


template <int nbytes>
static __device__ __forceinline__ void memcpy_aligned(void* __restrict__ dst, const void* __restrict__ src) {
    if constexpr (nbytes == 16) {
        ((int4*)dst)[0] = ((const int4*)src)[0];
    } else if constexpr (nbytes == 8) {
        ((int2*)dst)[0] = ((const int2*)src)[0];
    } else if constexpr (nbytes == 4) {
        ((int*)dst)[0] = ((const int*)src)[0];
    }
}
enum data_layout {
    DATA_LAYOUT_I_MAJOR = 0,
    DATA_LAYOUT_J_MAJOR = 10,
    DATA_LAYOUT_I_MAJOR_MIRRORED = 20,
};
template <int I_, int J_, typename T, data_layout dl = DATA_LAYOUT_I_MAJOR>
struct tile {};
template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I * J / 32;
    T x[ne] = {0};
    static __device__ __forceinline__ int get_i(const int) {
        return threadIdx.x % 16;
    }
    static __device__ __forceinline__ int get_j(const int l) {
        if constexpr (I == 16 && J == 16) {
            return 2 * l + (threadIdx.x / 16);
        } else {
            return l;
        }
    }
};
template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_J_MAJOR> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::ne;
    T x[ne] = {0};
    static __device__ __forceinline__ int get_i(const int l) {
        return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_j(l);
    }
    static __device__ __forceinline__ int get_j(const int l) {
        return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_i(l);
    }
};
template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR_MIRRORED> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I * J / 32 * 2;
    T x[ne] = {0};
    static __device__ __forceinline__ int get_i(const int) {
        return threadIdx.x % 16;
    }
    static __device__ __forceinline__ int get_j(const int l) {
        return l;
    }
};
template <typename T, data_layout dl>
static __device__ __forceinline__ void load_ldmatrix_16x8(tile<16, 8, T, dl>& t, const T* __restrict__ xs0, const int stride) {
    static_assert(dl == DATA_LAYOUT_I_MAJOR_MIRRORED, "bad layout");
    static_assert(sizeof(t.x) == 32, "bad mirrored tile size");
    memcpy_aligned<16>(t.x + 0, xs0 + t.get_i(0) * stride + 0);
    memcpy_aligned<16>(t.x + 4, xs0 + t.get_i(0) * stride + 4);
}
template <data_layout dl_d, data_layout dl_ab>
static __device__ __forceinline__ void mma_i8(
    tile<16, 16, int, dl_d>& D,
    const tile<16, 8, int, dl_ab>& A,
    const tile<16, 8, int, dl_ab>& B
) {
    using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
    using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
    int32x8_t* acc = (int32x8_t*)D.x;
    const int32x4_t* a_vec = (const int32x4_t*)A.x;
    const int32x4_t* b_vec = (const int32x4_t*)B.x;
    acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[0], true, b_vec[0], acc[0], true);
    acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[1], true, b_vec[1], acc[0], true);
}
template <bool FULL>
static __device__ __forceinline__ void load_hfq4_tile(
    const char* __restrict__ A,
    int* __restrict__ tile_x,
    int row0,
    int kb,
    int M,
    int groups_per_row
) {
    int* x_qs = tile_x;
    half2* x_dm = (half2*)(x_qs + 2 * 32);
    static_assert((64 % (4 * 8)) == 0,
                  "quad-row loader requires complete row groups");
    const int txi = threadIdx.x & 7;
    for (int i0 = 0; i0 < 64; i0 += 4 * 8) {
        const int i = i0 + 4 * threadIdx.y + (threadIdx.x >> 3);
        const int row = FULL ? row0 + i : min(row0 + i, M - 1);
        const char* gp = A + ((long long)row * groups_per_row + kb) * 136;
        const uint2* packed = (const uint2*)(gp + 8 + txi * 16);
        int* dst = x_qs + i * 76 + 8 * txi;
        uint2 words = packed[0];
        unsigned int q = words.x;
        unsigned int lo = q & 0x0F0F0F0Fu;
        unsigned int hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[0] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[1] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        q = words.y;
        lo = q & 0x0F0F0F0Fu;
        hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[2] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[3] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        words = packed[1];
        q = words.x;
        lo = q & 0x0F0F0F0Fu;
        hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[4] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[5] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
        q = words.y;
        lo = q & 0x0F0F0F0Fu;
        hi = (q >> 4) & 0x0F0F0F0Fu;
        dst[6] = (int)__builtin_amdgcn_perm(hi, lo, 0x05010400u);
        dst[7] = (int)__builtin_amdgcn_perm(hi, lo, 0x07030602u);
    }
    constexpr int rows_per_warp_meta = 16;
    for (int i0 = 0; i0 < 64; i0 += 8 * rows_per_warp_meta) {
        const int i = i0 + threadIdx.y * rows_per_warp_meta + threadIdx.x / 2;
        if (i < 64) {
            const int row = FULL ? row0 + i : min(row0 + i, M - 1);
            const long long group = (long long)row * groups_per_row + kb;
            const char* gp = A + group * 136;
            const int ksc = threadIdx.x & 1;
#ifdef PACKED_MQ4_V2
            // qt44 carries independent fp16 (scale, zero) pairs per K128.
            const half2 dm = *(const half2*)(gp + 4 * ksc);
#else
            const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
            const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            const half2 dm = make_half2((_Float16)sc, (_Float16)zp);
#endif
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                x_dm[i * 76 + 4 * ksc + l] = dm;
            }
        }
    }
}
template <bool APPLY_ZERO, bool COLLECT_SUMS = false>
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(
    const int* __restrict__ x,
    const int* __restrict__ y,
    float* __restrict__ sum,
    int k00,
    float* __restrict__ activation_sums = nullptr
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = 2;
    constexpr int col_groups = 4;
    static_assert(64 == (8 / col_groups) * row_frags * tile_C::I, "bad row topology");
    y += (threadIdx.y % col_groups) * (tile_C::J * 36);
    const int* x_qs = x;
    const half2* x_dm = (const half2*)(x_qs + 2 * 32);
    const int* y_qs = y + 4;
    const half2* y_dm = (const half2*)y;
    const int i0 = (threadIdx.y / col_groups) * row_frags * tile_C::I;
    for (int k01 = 0; k01 < 32; k01 += 8) {
        const int k0 = k00 + k01;
        tile_A A_frag[row_frags];
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
            load_ldmatrix_16x8(A_frag[n], x_qs + (i0 + n * tile_A::I) * 76 + k0, 76);
        }
#pragma unroll
        for (int j0 = 0; j0 < 256; j0 += col_groups * tile_C::J) {
            tile_B B_frag;
            load_ldmatrix_16x8(B_frag, y_qs + j0 * 36 + k01, 36);
            const int j = j0 + tile_C::get_j(0);
            const float2 dsB = __half22float2(y_dm[j * 36 + k01 / 8]);
            if constexpr (COLLECT_SUMS) {
                const int zero_acc = j0 / (col_groups * tile_C::J);
                activation_sums[zero_acc] += dsB.y;
            }
#pragma unroll
            for (int n = 0; n < row_frags; ++n) {
                tile_C C_frag;
                mma_i8(C_frag, A_frag[n], B_frag);
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                    const float2 dmA = __half22float2(x_dm[i * 76 + k0 / 8]);
                    const int acc = (j0 / (col_groups * tile_C::J)) * row_frags + n;
                    sum[acc * tile_C::ne + l] += dmA.x * dsB.x * (float)C_frag.x[l];
                    if constexpr (APPLY_ZERO) {
                        sum[acc * tile_C::ne + l] += dmA.y * dsB.y;
                    }
                }
            }
        }
    }
}
static __device__ __forceinline__ half2 load_group128_weight_dm(
    const half2* __restrict__ x_dm,
    int row,
    int k_offset
) {
    return x_dm[row * 76 + k_offset / 8];
}
static __device__ __forceinline__ void mmq_accum_add(
    float* __restrict__ sum,
    int idx,
    float value
) {
    sum[idx] += value;
}
static __device__ __forceinline__ float mmq_accum_get(
    const float* __restrict__ sum,
    int idx
) {
    return sum[idx];
}
static __device__ __forceinline__ void mmq_accum_add(
    half2* __restrict__ sum,
    int idx,
    float value
) {
    float2 pair = __half22float2(sum[idx >> 1]);
    if (idx & 1) {
        pair.y += value;
    } else {
        pair.x += value;
    }
    sum[idx >> 1] = __floats2half2_rn(pair.x, pair.y);
}
static __device__ __forceinline__ float mmq_accum_get(
    const half2* __restrict__ sum,
    int idx
) {
    const float2 pair = __half22float2(sum[idx >> 1]);
    return (idx & 1) ? pair.y : pair.x;
}
template <typename SumT>
static __device__ __forceinline__ void vec_dot_q8_1_group128_mma(
    const int* __restrict__ x,
    const int* __restrict__ y,
    SumT* __restrict__ sum,
    int k00
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = 2;
    constexpr int col_groups = 4;
    const int local_wave = threadIdx.y;
    const int plane_row = 0;
    y += (local_wave % col_groups) * (tile_C::J * 36);
    const int* x_qs = x;
    const half2* x_dm = (const half2*)(x_qs + 64);
    const int* y_qs = y + 4;
    const half2* y_dm = (const half2*)y;
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;
#pragma unroll
    for (int j0 = 0; j0 < 256; j0 += col_groups * tile_C::J) {
        tile_C c_acc[row_frags];
#pragma unroll
        for (int k01 = 0; k01 < 32; k01 += 8) {
            const int k0 = k00 + k01;
            tile_B b_frag;
            load_ldmatrix_16x8(
                b_frag, y_qs + j0 * 36 + k01, 36);
#pragma unroll
            for (int n = 0; n < row_frags; ++n) {
                tile_A a_frag;
                load_ldmatrix_16x8(
                    a_frag,
                    x_qs + (plane_row + i0 + n * tile_A::I) * 76 + k0,
                    76);
                mma_i8(c_acc[n], a_frag, b_frag);
            }
        }
        const int j = j0 + tile_C::get_j(0);
        const float d_x = __half22float2(y_dm[j * 36]).x;
#ifdef PACKED_MQ4_V2
        float half_sum = 0.0f;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            half_sum += __half22float2(y_dm[j * 36 + t]).y;
        }
#endif
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                const float scale_w = __half22float2(load_group128_weight_dm(
                    x_dm, plane_row + i, k00)).x;
                const int acc = (j0 / (col_groups * tile_C::J)) * row_frags + n;
                mmq_accum_add(
                    sum,
                    acc * tile_C::ne + l,
                    scale_w * d_x * (float)c_acc[n].x[l]);
#ifdef PACKED_MQ4_V2
                const float zero_w = __half22float2(load_group128_weight_dm(
                    x_dm, plane_row + i, k00)).y;
                mmq_accum_add(sum, acc * tile_C::ne + l, zero_w * half_sum);
#endif
            }
        }
    }
}
static __device__ __forceinline__ void load_group_row_zeros(
    const int* __restrict__ tile_x,
    float* __restrict__ row_zeros
) {
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = 2;
    constexpr int col_groups = 4;
    const half2* x_dm = (const half2*)(tile_x + 64);
    const int local_wave = threadIdx.y;
    const int plane_row = 0;
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;
#pragma unroll
    for (int n = 0; n < row_frags; ++n) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int i = i0 + n * tile_C::I + tile_C::get_i(l);
            row_zeros[n * tile_C::ne + l] =
                __half22float2(x_dm[(plane_row + i) * 76]).y;
        }
    }
}
template <typename SumT>
static __device__ __forceinline__ void add_group_zero_correction_regs(
    const float* __restrict__ row_zeros,
    const float* __restrict__ activation_sums,
    SumT* __restrict__ sum
) {
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = 2;
    constexpr int col_groups = 4;
#pragma unroll
    for (int j0 = 0; j0 < 256; j0 += col_groups * tile_C::J) {
        const int zero_acc = j0 / (col_groups * tile_C::J);
        const float activation_sum = activation_sums[zero_acc];
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int acc = zero_acc * row_frags + n;
                mmq_accum_add(
                    sum,
                    acc * tile_C::ne + l,
                    row_zeros[n * tile_C::ne + l] * activation_sum);
            }
        }
    }
}
static __device__ __forceinline__ void collect_q8_1_sums(
    const int* __restrict__ tile_y,
    float* __restrict__ zero_sums,
    bool reset
) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    if (tid < 256) {
        const half2* y_dm = (const half2*)tile_y;
        float value = 0.0f;
#pragma unroll
        for (int k01 = 0; k01 < 32; k01 += 8) {
            value += __half22float2(y_dm[tid * 36 + k01 / 8]).y;
        }
        zero_sums[tid] = reset ? value : zero_sums[tid] + value;
    }
}
template <typename SumT>
static __device__ __forceinline__ void add_group_zero_correction(
    const int* __restrict__ tile_x,
    const float* __restrict__ zero_sums,
    SumT* __restrict__ sum
) {
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = 2;
    constexpr int col_groups = 4;
    const half2* x_dm = (const half2*)(tile_x + 64);
    const int local_wave = threadIdx.y;
    const int plane_row = 0;
    const int i0 = (local_wave / col_groups) * row_frags * tile_C::I;
#pragma unroll
    for (int j0 = 0; j0 < 256; j0 += col_groups * tile_C::J) {
        const int j = j0 + (local_wave % col_groups) * tile_C::J + tile_C::get_j(0);
        const float activation_sum = zero_sums[j];
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                const float zero = __half22float2(x_dm[(plane_row + i) * 76]).y;
                const int acc = (j0 / (col_groups * tile_C::J)) * row_frags + n;
                mmq_accum_add(sum, acc * tile_C::ne + l, zero * activation_sum);
            }
        }
    }
}

template <bool ADD, bool COMBINE_ZERO = false, bool REG_ZERO = false,
          typename OutputT = float>
static __device__ __forceinline__ void gemm_mq4_packed_gfx1100_full_body(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    OutputT* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int row0 = blockIdx.x * 64;
    const int col0 = blockIdx.y * 256;
    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;
    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + 256 * 36;
    float* zero_sums = (float*)(tile_x + 64 * 76);
    float sum[256 * 64 / (8 * 32)] = {};
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int row_frags = 2;
    constexpr int col_groups = 4;
    constexpr int zero_groups = 256 / (col_groups * tile_C::J);
    float activation_sums[zero_groups] = {};
    float row_zeros[row_frags * tile_C::ne] = {};
    for (int kb = 0; kb < groups_per_row; ++kb) {
        if constexpr (REG_ZERO) {
#pragma unroll
            for (int z = 0; z < zero_groups; ++z) activation_sums[z] = 0.0f;
        }
        load_hfq4_tile<true>(A, tile_x, row0, kb, M, groups_per_row);
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l0 = 0; l0 < 256 * 36; l0 += 8 * 32) {
            const int l = l0 + threadIdx.y * 32 + threadIdx.x;
            tile_y[l] = by0[l];
        }
        __syncthreads();
        if constexpr (REG_ZERO) {
            load_group_row_zeros(tile_x, row_zeros);
        } else if constexpr (COMBINE_ZERO && !0) {
            collect_q8_1_sums(tile_y, zero_sums, true);
        }
        // Consume all four independent G32 scales; do not combine integer
        // dots under a single G128 scale or defer the zero correction.
        vec_dot_q8_1_q8_1_mma<!COMBINE_ZERO, REG_ZERO>(
            tile_x, tile_y, sum, 0, activation_sums);
        __syncthreads();
        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < 256 * 36; l0 += 8 * 32) {
                const int l = l0 + threadIdx.y * 32 + threadIdx.x;
                tile_y[l] = by1[l];
            }
            __syncthreads();
            if constexpr (COMBINE_ZERO && !REG_ZERO && !0) {
                collect_q8_1_sums(tile_y, zero_sums, false);
            }
            vec_dot_q8_1_q8_1_mma<!COMBINE_ZERO, REG_ZERO>(
                tile_x, tile_y, sum, 32, activation_sums);
            __syncthreads();
        }
        if constexpr (REG_ZERO) {
            add_group_zero_correction_regs(row_zeros, activation_sums, sum);
        } else if constexpr (COMBINE_ZERO && !0) {
            add_group_zero_correction(tile_x, zero_sums, sum);
            __syncthreads();
        }
    }
    const int i0 = (threadIdx.y / col_groups) * row_frags * tile_C::I;
#pragma unroll
    for (int j0 = 0; j0 < 256; j0 += col_groups * tile_C::J) {
#pragma unroll
        for (int n = 0; n < row_frags; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % col_groups) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                OutputT* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                const int acc = (j0 / (col_groups * tile_C::J)) * row_frags + n;
                const float v = mmq_accum_get(sum, acc * tile_C::ne + l);
                if constexpr (ADD) {
                    *yp += v;
                } else {
                    *yp = v;
                }
            }
        }
    }
}
extern "C" __global__ __launch_bounds__(8 * 32, 2)
void gemm_mq4_packed_gfx1100_full_add(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int
) {
    gemm_mq4_packed_gfx1100_full_body<true, false>(A, Xq, Y, M, K, N);
}
extern "C" __global__ __launch_bounds__(8 * 32, 2)
void gemm_mq4_packed_gfx1100_full_set(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int
) {
    gemm_mq4_packed_gfx1100_full_body<false, false>(A, Xq, Y, M, K, N);
}
extern "C" __global__ __launch_bounds__(8 * 32, 2)
void gemm_mq4_packed_gfx1100_full_set_f16(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    half* __restrict__ Y,
    int M,
    int K,
    int N,
    int
) {
    gemm_mq4_packed_gfx1100_full_body<false, false, false, half>(A, Xq, Y, M, K, N);
}
