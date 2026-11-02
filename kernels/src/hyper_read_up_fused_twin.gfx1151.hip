// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1151 HC read tail (BF16 up projection fused with
// hyper_read_projected_f32), scalar-X scheduling twin.
//
// Bitwise twin of `hyper_read_up_fused_f32` (tensor_ops.hip): same
// arguments, grid [hidden / 8, ceil(rows / 128)], block 256 and dynamic LDS
// 32 * (low_rank / 2 + 1) * 4 bytes.  Per (token, wide row) the arithmetic is
// the incumbent's text: the LDS-staged BF16 row, the 32 eight-value chunks
// (vl = l + 16p, i = vl * 8 + it * 256) accumulated into four chains a0..a3
// in the same order, (a0 + a1) + (a2 + a3), h[l] = sp0 + sp1, the halving
// tree over l, and the BF16-rounded gate / branch-order mix, four tokens per
// wave per pass, four passes.  Only the `mixed` stores move: they are held in
// registers until every pass is done.  In the incumbent a store before a
// later pass's loads stops the compiler from proving the wave-uniform X loads
// unclobbered, so X goes through the vector memory path (521 uniform VMEM
// loads); with the stores last and the register pins free of a memory
// clobber, X is read through scalar loads.
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// tensor_ops.hip's alias: the BF16 rounding is hip_bfloat16's.
using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ float bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

static __device__ __forceinline__ __hip_bfloat16 bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

extern "C" __global__ __launch_bounds__(256, 1) void hyper_read_up_fused_f32_sx(
    const unsigned short* __restrict__ W,
    const float* __restrict__ X,
    const float* __restrict__ normalized,
    float* __restrict__ mixed,
    int hidden,
    int K,
    int N,
    int normalized_bf16) {
    constexpr int TPW = 4;
    constexpr int TGROUPS = 4;
    const int branches = 4;
    const int wide = branches * hidden;
    extern __shared__ unsigned int ws[];
    const int stride = K / 2 + 1;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int j0 = blockIdx.x * 8;
    const int vpr = K / 8;
    for (int v = tid; v < 32 * vpr; v += 256) {
        const int r = v / vpr, c = v - r * vpr;
        const int row = (r >> 3) * hidden + j0 + (r & 7);
        const uint4 u = *(const uint4*)(W + (size_t)row * K + c * 8);
        unsigned int* d = ws + r * stride + c * 4;
        d[0] = u.x; d[1] = u.y; d[2] = u.z; d[3] = u.w;
    }
    __syncthreads();
    const unsigned int* wr = ws + lane * stride;
    const int my_row = (lane >> 3) * hidden + j0 + (lane & 7);
    float held[TGROUPS][TPW];
    for (int tg = 0; tg < TGROUPS; ++tg) {
        const int tbase = ((blockIdx.y * TGROUPS + tg) * 8 + wave) * TPW;
        if (tbase >= N) break;
        int tok[TPW];
#pragma unroll
        for (int t = 0; t < TPW; ++t) tok[t] = tbase + t < N ? tbase + t : tbase;
        float h[TPW][16];
#pragma unroll
        for (int l = 0; l < 16; ++l) {
            float sp[2][TPW];
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int vl = l + 16 * p;
                float a[TPW][4];
#pragma unroll
                for (int t = 0; t < TPW; ++t) a[t][0] = a[t][1] = a[t][2] = a[t][3] = 0.0f;
#pragma unroll
                for (int it = 0; it < 2; ++it) {
                    const int i = vl * 8 + it * 256;
                    if (i + 8 > K) break;
                    const unsigned int u0 = wr[i / 2], u1 = wr[i / 2 + 1], u2 = wr[i / 2 + 2], u3 = wr[i / 2 + 3];
                    const float w0 = __uint_as_float(u0 << 16), w1 = __uint_as_float(u0 & 0xFFFF0000u);
                    const float w2 = __uint_as_float(u1 << 16), w3 = __uint_as_float(u1 & 0xFFFF0000u);
                    const float w4 = __uint_as_float(u2 << 16), w5 = __uint_as_float(u2 & 0xFFFF0000u);
                    const float w6 = __uint_as_float(u3 << 16), w7 = __uint_as_float(u3 & 0xFFFF0000u);
#pragma unroll
                    for (int t = 0; t < TPW; ++t) {
                        const float* xr = X + (size_t)tok[t] * K + i;
                        a[t][0] += w0 * xr[0];
                        a[t][1] += w1 * xr[1];
                        a[t][2] += w2 * xr[2];
                        a[t][3] += w3 * xr[3];
                        a[t][0] += w4 * xr[4];
                        a[t][1] += w5 * xr[5];
                        a[t][2] += w6 * xr[6];
                        a[t][3] += w7 * xr[7];
                    }
                }
#pragma unroll
                for (int t = 0; t < TPW; ++t) sp[p][t] = (a[t][0] + a[t][1]) + (a[t][2] + a[t][3]);
            }
#pragma unroll
            for (int t = 0; t < TPW; ++t) {
                h[t][l] = sp[0][t] + sp[1][t];
                asm volatile("" : "+v"(h[t][l]));
            }
        }
#pragma unroll
        for (int t = 0; t < TPW; ++t) {
#pragma unroll
            for (int off = 8; off > 0; off >>= 1)
#pragma unroll
                for (int l = 0; l < off; ++l) h[t][l] = h[t][l] + h[t][l + off];
            // hyper_read_projected_f32 for (token, column) from the branch
            // lanes; the sum runs in branch order from +0.0.
            const size_t token = (size_t)tok[t];
            const float gate_input = bf16_to_f32(bf16_round(h[t][0]));
            const float gate = bf16_to_f32(bf16_round(1.0f / (1.0f + expf(-gate_input))));
            const float normed = normalized_bf16
                ? __uint_as_float((unsigned int)((const unsigned short*)normalized)[token * wide + my_row] << 16)
                : bf16_to_f32(bf16_round(normalized[token * wide + my_row]));
            const float term = bf16_to_f32(bf16_round(gate * normed));
            float value = 0.0f;
#pragma unroll
            for (int branch = 0; branch < 4; ++branch)
                value += __shfl(term, (lane & 7) + 8 * branch, 32);
            // tg is a rolled loop counter: select into static slots so
            // `held` stays in registers.
            const float out = bf16_to_f32(bf16_round(value / (float)branches));
#pragma unroll
            for (int k = 0; k < TGROUPS; ++k)
                if (tg == k) held[k][t] = out;
        }
    }
#pragma unroll
    for (int tg = 0; tg < TGROUPS; ++tg) {
        const int tbase = ((blockIdx.y * TGROUPS + tg) * 8 + wave) * TPW;
        if (tbase >= N) break;
#pragma unroll
        for (int t = 0; t < TPW; ++t)
            if (lane < 8 && tbase + t < N)
                mixed[(size_t)(tbase + t) * hidden + j0 + lane] = held[tg][t];
    }
}
