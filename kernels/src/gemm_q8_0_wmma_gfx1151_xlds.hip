// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exact-gfx1151 Q8_0 WMMA GEMM for small batches (the DFlash draft/verify
// lm_head of Q8-head models, 248320 x 5120 at 15-16 rows). Same ABI as
// gemm_q8_0_wmma: Y[N, M] = X[N, K] @ A_q8[M, K]^T, X FP16, Y FP32 overwrite.
//
// Per output the arithmetic is gemm_q8_0_wmma's exactly, so outputs are
// byte-identical to it:
//   - one wave owns 16 weight rows x 16 batch rows and runs the whole K chain
//     acc = wmma_f32_16x16x16_f16(A_t, B_t, acc) for t = 0..K/16-1 in order,
//     from acc = 0 (no split-K, no reduction across waves);
//   - A_t[i] = sc * (f16)w_i with the block's fp16 scale: the int8 -> f16 cast is
//     exact here too (0x6400 | (w ^ 0x80) is 1152 + w, minus 1152), and the one
//     f16 multiply is the same IEEE op, two lanes per v_pk_mul_f16;
//   - B_t is the same 16 x 16 fp16 X tile; invalid batch rows read row 0.
// What differs is memory access only. The base kernel loads each 34 B block
// byte by byte (the blocks are 2 B aligned) from one single-wave block per 16
// rows, ~33 GB/s on the 1.35 GB 27B head. Here a block runs 4 waves (64 weight
// rows) and stages each 256-K slab of the X panel in LDS once, double-buffered;
// each lane loads its row's 272 B (8 Q8 blocks, 16 B aligned when K % 256 == 0)
// with 17 b128 loads and splits blocks out of registers.
//
// Grid: [ceil(M / 64), ceil(N / 16), 1]. Block: [128, 1, 1].
// Static LDS: 2 x 16 x 264 halfs = 16,896 B. Requires K % 256 == 0 and a 16 B
// aligned A. Waves whose 16-row tile starts past M still run the loop (clamped
// rows) so the block barriers stay uniform; their stores are masked.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef _Float16 __attribute__((ext_vector_type(2))) half2_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef unsigned int __attribute__((ext_vector_type(8))) u8_t;
typedef unsigned int __attribute__((ext_vector_type(4))) u4_t;

#define Q8X_WAVES 4
#define Q8X_THREADS (32 * Q8X_WAVES)
// X slab row pitch 264 halfs (528 B): 16 lanes' b128 reads hit distinct banks.
#define Q8X_PITCH 264
#define Q8X_SLAB (16 * Q8X_PITCH)
#define Q8X_PER ((512 + Q8X_THREADS - 1) / Q8X_THREADS)
// One 256-K group of a Q8_0 row: 8 blocks x 34 B = 272 B = 68 dwords.
#define Q8X_GROUP_BYTES 272
#define Q8X_GROUP_DW 68

__device__ __forceinline__ half2_t q8x_h2(unsigned int u) { return __builtin_bit_cast(half2_t, u); }
__device__ __forceinline__ unsigned int q8x_u(half2_t h) { return __builtin_bit_cast(unsigned int, h); }

// Sixteen int8 weights (four dwords, bytes in K order) times the block scale:
// one K tile's A fragment.
__device__ __forceinline__ half16_t q8x_dq16(const unsigned int* w, half2_t sc2) {
    const half2_t m1152 = q8x_h2(0xE480E480u);
    u8_t o;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned int q = w[i] ^ 0x80808080u;
        const unsigned int lo = __builtin_amdgcn_perm(0u, q, 0x0c010c00u) | 0x64006400u;
        const unsigned int hi = __builtin_amdgcn_perm(0u, q, 0x0c030c02u) | 0x64006400u;
        o[2 * i] = q8x_u((q8x_h2(lo) + m1152) * sc2);
        o[2 * i + 1] = q8x_u((q8x_h2(hi) + m1152) * sc2);
    }
    return __builtin_bit_cast(half16_t, o);
}

__launch_bounds__(Q8X_THREADS)
extern "C" __global__ void gemm_q8_0_wmma_gfx1151_xlds(
    const char* __restrict__ A_q8,  // [M, K/32 * 34] packed Q8_0
    const _Float16* __restrict__ X, // [N, K] FP16
    float* __restrict__ Y,          // [N, M] FP32
    int M, int K, int N) {
    __shared__ __align__(16) _Float16 xs_lds[2 * Q8X_SLAB];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const int row_start = (blockIdx.x * Q8X_WAVES + (tid >> 5)) * 16;
    const int batch_start = blockIdx.y * 16;
    const int my_row = row_start + lane16;
    const int safe_row = (my_row < M) ? my_row : (M - 1);
    const int G = K >> 8;
    const char* row_base = A_q8 + (long long)safe_row * G * Q8X_GROUP_BYTES;

    // Cooperative X slab fill: 16 batch rows x 256 halfs = 512 vectors of 16 B.
    u4_t xr[Q8X_PER];
    auto x_fetch = [&](int g) {
#pragma unroll
        for (int i = 0; i < Q8X_PER; ++i) {
            const int v = tid + i * Q8X_THREADS;
            const int b = v >> 5, c = v & 31;
            const int sb = (batch_start + b < N) ? (batch_start + b) : 0;
            xr[i] = *(const u4_t*)(X + (long long)sb * K + g * 256 + c * 8);
        }
    };
    auto x_store = [&](int buf) {
#pragma unroll
        for (int i = 0; i < Q8X_PER; ++i) {
            const int v = tid + i * Q8X_THREADS;
            const int b = v >> 5, c = v & 31;
            *(u4_t*)(xs_lds + buf * Q8X_SLAB + b * Q8X_PITCH + c * 8) = xr[i];
        }
    };

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    x_fetch(0);
    x_store(0);
    __syncthreads();
    for (int g = 0; g < G; ++g) {
        unsigned int d[Q8X_GROUP_DW];
        const u4_t* gp = (const u4_t*)(row_base + (long long)g * Q8X_GROUP_BYTES);
#pragma unroll
        for (int i = 0; i < Q8X_GROUP_DW / 4; ++i) {
            const u4_t v = gp[i];
            d[4 * i] = v.x; d[4 * i + 1] = v.y; d[4 * i + 2] = v.z; d[4 * i + 3] = v.w;
        }
        if (g + 1 < G) x_fetch(g + 1);
        const _Float16* xs = xs_lds + (g & 1) * Q8X_SLAB + lane16 * Q8X_PITCH;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            // Block b spans bytes [34b, 34b + 34): even b starts dword aligned
            // (scale = low half, weights at +2 B), odd b at +2 B (scale = high
            // half, weights dword aligned).
            const int d0 = (34 * b) >> 2;
            unsigned int w[8];
            unsigned int s;
            if ((b & 1) == 0) {
                s = d[d0] & 0xFFFFu;
#pragma unroll
                for (int j = 0; j < 8; ++j) w[j] = __builtin_amdgcn_alignbyte(d[d0 + j + 1], d[d0 + j], 2);
            } else {
                s = d[d0] >> 16;
#pragma unroll
                for (int j = 0; j < 8; ++j) w[j] = d[d0 + 1 + j];
            }
            const half2_t sc2 = q8x_h2(s | (s << 16));
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                q8x_dq16(w, sc2), *(const half16_t*)(xs + b * 32), acc);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                q8x_dq16(w + 4, sc2), *(const half16_t*)(xs + b * 32 + 16), acc);
        }
        // Slab (g + 1) & 1 was last read in group g - 1, before that group's barrier.
        if (g + 1 < G) x_store((g + 1) & 1);
        __syncthreads();
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (lane>>4)][lane & 15].
    const int out_col = batch_start + lane16;
    if (row_start < M && out_col < N) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (lane >> 4);
            if (out_row < M) Y[(long long)out_col * M + out_row] = acc[j];
        }
    }
}
