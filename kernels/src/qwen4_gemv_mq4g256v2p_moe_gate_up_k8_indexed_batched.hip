// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Drop-zero planar twin (DESIGN F1, MQ4G256V2P, qt=137) of
// qwen4_gemv_mq4g256v2_moe_gate_up_k8_indexed_batched.hip.
//
// The shipped Flash-Next experts are symmetric MQ4-XTS (zp == f16(-8*sc), verified by the
// loader), so the repack drops the zero. Each expert tensor [M, K] is planar:
//
//   [codes plane: M*K/2 bytes][scales plane: M*(K/128) LE f16]
//
//   row r codes   : base + r*(K/2)            (low nibble = even weight, shipped order)
//   row r scales  : base + M*(K/2) + r*(K/128)*2; the two halves of a 256-group g are the
//                   consecutive f16 scales 2g, 2g+1  (one aligned u32 per group).
//
// Everything else is the shipped kernel line for line: grid (M, K_TOP, N), one wave per
// output row, the same lane/nibble mapping, accumulator association and wave reduction, so
// the result is bit-identical. The only arithmetic difference is where zp comes from:
// zp = -8.0f * sc. The f16 widen of f16(-8*sc) IS -8*sc in f32 exactly (x8 only moves the
// exponent), and (sc*q + zp) is exact (|sc*q| < 2^15 * 2^-14.. a product of an 11-bit
// mantissa and a 4-bit integer), so contraction cannot change a bit either.
//
// Thread `tid` reads the u32 at codes offset tid*4 (weights [tid*8, tid*8+8)), wholly in one
// 128-weight half: tid 0..15 use scale 2g (low half of the group's scale word), tid 16..31
// scale 2g+1 (high half). The half is lane-constant.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

static __device__ __forceinline__ float mq4p_scale(unsigned int h, int shift) {
    return __half2float(__ushort_as_half((unsigned short)((h >> shift) & 0xFFFFu)));
}

// Qwen4 K = 2560 (ten groups): every scale and payload load issues up front. Group g
// accumulates into acc[g & 3] in the loop's order; returns this lane's
// (acc0 + acc1) + (acc2 + acc3).
static __device__ __forceinline__ float mq4v2p_gate_up_row10(
    const char* __restrict__ codes_row, const char* __restrict__ scales_row,
    const float* x_row, int tid
) {
    const int boff = tid * 4;
    const int shift = (tid < 16) ? 0 : 16;
    unsigned int h[10], pk[10];
#pragma unroll
    for (int g = 0; g < 10; ++g) {
        h[g] = *(const unsigned int*)(scales_row + g * 4);
        pk[g] = *(const unsigned int*)(codes_row + g * 128 + boff);
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int g = 0; g < 10; ++g) {
        const float sc = mq4p_scale(h[g], shift);
        const float zp = -8.0f * sc;
        const int b = g * 256 + tid * 8;
        acc[g & 3] += (sc * (float)(pk[g] & 0xFu)        + zp) * x_row[b]
                    + (sc * (float)((pk[g] >> 4) & 0xFu)  + zp) * x_row[b + 1]
                    + (sc * (float)((pk[g] >> 8) & 0xFu)  + zp) * x_row[b + 2]
                    + (sc * (float)((pk[g] >> 12) & 0xFu) + zp) * x_row[b + 3]
                    + (sc * (float)((pk[g] >> 16) & 0xFu) + zp) * x_row[b + 4]
                    + (sc * (float)((pk[g] >> 20) & 0xFu) + zp) * x_row[b + 5]
                    + (sc * (float)((pk[g] >> 24) & 0xFu) + zp) * x_row[b + 6]
                    + (sc * (float)((pk[g] >> 28) & 0xFu) + zp) * x_row[b + 7];
    }
    return (acc[0] + acc[1]) + (acc[2] + acc[3]);
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq4g256v2p_moe_gate_up_k8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,                // [N × K_TOP]
    const float* __restrict__ x,                         // [N × K]
    float* __restrict__ y_gate,                          // [N × K_TOP × MI]
    float* __restrict__ y_up,                            // [N × K_TOP × MI]
    int M, int K, int K_TOP, int n_experts
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[bid * K_TOP + krank];
    if ((unsigned)expert_id >= (unsigned)n_experts) {
        if (tid == 0) {
            const int mi_inv = M >> 1;
            const long long base_inv = (long long)bid * K_TOP * mi_inv + (long long)krank * mi_inv;
            if (row < mi_inv) y_gate[base_inv + row] = 0.0f;
            else              y_up[base_inv + (row - mi_inv)] = 0.0f;
        }
        return;
    }
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x_row = x + (long long)bid * K;

    const int groups_per_row = K / 256;
    const char* codes_row = A + (long long)row * (K / 2);
    const char* scales_row = A + (long long)M * (K / 2) + (long long)row * groups_per_row * 4;
    const int boff = tid * 4;
    // Lane-constant half of the group's scale word (tid 0..15 -> scale 2g, 16..31 -> 2g+1).
    const int shift = (tid < 16) ? 0 : 16;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

#if defined(__gfx1151__)
    if (groups_per_row == 10) {
        acc0 = mq4v2p_gate_up_row10(codes_row, scales_row, x_row, tid);
    } else
#endif
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const unsigned int h0 = *(const unsigned int*)(scales_row + (g + 0) * 4);
        const unsigned int h1 = *(const unsigned int*)(scales_row + (g + 1) * 4);
        const unsigned int h2 = *(const unsigned int*)(scales_row + (g + 2) * 4);
        const unsigned int h3 = *(const unsigned int*)(scales_row + (g + 3) * 4);
        const float sc0 = mq4p_scale(h0, shift), zp0 = -8.0f * sc0;
        const float sc1 = mq4p_scale(h1, shift), zp1 = -8.0f * sc1;
        const float sc2 = mq4p_scale(h2, shift), zp2 = -8.0f * sc2;
        const float sc3 = mq4p_scale(h3, shift), zp3 = -8.0f * sc3;

        const unsigned int pk0 = *(const unsigned int*)(codes_row + (g + 0) * 128 + boff);
        const unsigned int pk1 = *(const unsigned int*)(codes_row + (g + 1) * 128 + boff);
        const unsigned int pk2 = *(const unsigned int*)(codes_row + (g + 2) * 128 + boff);
        const unsigned int pk3 = *(const unsigned int*)(codes_row + (g + 3) * 128 + boff);

        int base = g * 256 + tid * 8;

        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * x_row[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x_row[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x_row[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x_row[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x_row[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x_row[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x_row[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x_row[(b) + 7]

        DOG(pk0, sc0, zp0, base, acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
    }

    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * x_row[(b)]     \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x_row[(b) + 1] \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x_row[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x_row[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x_row[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x_row[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x_row[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x_row[(b) + 7]

#if defined(__gfx1151__)
    if (groups_per_row != 10 && tail >= 1) {
#else
    if (tail >= 1) {
#endif
        const int g = quads << 2;
        const unsigned int h = *(const unsigned int*)(scales_row + g * 4);
        const float sc = mq4p_scale(h, shift);
        const float zp = -8.0f * sc;
        const unsigned int pk = *(const unsigned int*)(codes_row + g * 128 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
#if defined(__gfx1151__)
    if (groups_per_row != 10 && tail >= 2) {
#else
    if (tail >= 2) {
#endif
        const int g = (quads << 2) + 1;
        const unsigned int h = *(const unsigned int*)(scales_row + g * 4);
        const float sc = mq4p_scale(h, shift);
        const float zp = -8.0f * sc;
        const unsigned int pk = *(const unsigned int*)(codes_row + g * 128 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const unsigned int h = *(const unsigned int*)(scales_row + g * 4);
        const float sc = mq4p_scale(h, shift);
        const float zp = -8.0f * sc;
        const unsigned int pk = *(const unsigned int*)(codes_row + g * 128 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG

#if defined(__gfx1151__)
    float acc = groups_per_row == 10 ? acc0 : (acc0 + acc1) + (acc2 + acc3);
#else
    float acc = (acc0 + acc1) + (acc2 + acc3);
#endif
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const int mi = M >> 1;
        const long long base_out = (long long)bid * K_TOP * mi + (long long)krank * mi;
        if (row < mi) {
            y_gate[base_out + row] = acc;
        } else {
            y_up[base_out + (row - mi)] = acc;
        }
    }
}

// K = 2560 rows, eight per 256-thread block (grid (M / 8, K_TOP, N)): the block's waves
// read one LDS copy of the token's x. Each wave's row is the kernel above's value, bit for bit.
__launch_bounds__(256)
extern "C" __global__ void gemv_mq4g256v2p_moe_gate_up_k2560_rows8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,                // [N × K_TOP]
    const float* __restrict__ x,                         // [N × 2560]
    float* __restrict__ y_gate,                          // [N × K_TOP × MI]
    float* __restrict__ y_up,                            // [N × K_TOP × MI]
    int M, int K_TOP, int n_experts
) {
    __shared__ __align__(16) float xs[2560];
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const float4* x4 = reinterpret_cast<const float4*>(x + (long long)bid * 2560);
    for (int i = threadIdx.x; i < 640; i += 256) reinterpret_cast<float4*>(xs)[i] = x4[i];
    __syncthreads();
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= M) return;
    const int tid = threadIdx.x & 31;
    const int expert_id = topk_indices[bid * K_TOP + krank];
    if ((unsigned)expert_id >= (unsigned)n_experts) {
        if (tid == 0) {
            const int mi_inv = M >> 1;
            const long long base_inv = (long long)bid * K_TOP * mi_inv + (long long)krank * mi_inv;
            if (row < mi_inv) y_gate[base_inv + row] = 0.0f;
            else              y_up[base_inv + (row - mi_inv)] = 0.0f;
        }
        return;
    }
    const char* __restrict__ A = reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const char* __restrict__ codes_row = A + (long long)row * 1280;
    const char* __restrict__ scales_row = A + (long long)M * 1280 + (long long)row * 40;
    float acc = mq4v2p_gate_up_row10(codes_row, scales_row, xs, tid);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if (tid == 0) {
        const int mi = M >> 1;
        const long long base_out = (long long)bid * K_TOP * mi + (long long)krank * mi;
        if (row < mi) {
            y_gate[base_out + row] = acc;
        } else {
            y_up[base_out + (row - mi)] = acc;
        }
    }
}
