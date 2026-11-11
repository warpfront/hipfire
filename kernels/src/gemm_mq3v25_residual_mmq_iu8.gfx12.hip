// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// gfx1201 symmetric qt49 A8, 128 weight rows x 128 tokens, eight waves.
// Activations are the shared A8 `block_i8_128` blocks (136 B, [K/128][N],
// nibble-pair qs order: per 8-K group q0,q2,q4,q6,q1,q3,q5,q7), produced by
// `quantize_int8_mmq_ds128` / the fused A8 producers. The block's `s` is
// unused: symmetric qt49 has no zero term (q = u - 4).
// K64 single-buffer slabs in fragment order (288-byte skew per K16);
// W is widened from 3-bit q=u-4 to signed int8 at LDS publication, in the
// same nibble-pair order, so each weight byte meets its own activation.
// Dynamic LDS: A 9216 B, W 9216 B, row scales 512 B, token scales
// 512 B = 19456 B. Epilogue reuses LDS for eight 512-float wave tiles.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#if defined(__gfx1201__)
typedef int32_t __attribute__((ext_vector_type(2))) i32x2;
typedef int32_t __attribute__((ext_vector_type(8))) i32x8;
typedef float __attribute__((ext_vector_type(8))) f32x8;

// 8 codes (3 bits each, code i at bits 3i) -> 8 signed bytes in nibble-pair
// order: dword 0 = codes 0,2,4,6; dword 1 = codes 1,3,5,7.
__device__ __forceinline__ i32x2 mq3v25_widen_i8(unsigned bits) {
    const unsigned c0 = 0xFFFEFDFCu, c1 = 0x03020100u;
    i32x2 wide;
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        unsigned sel = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            sel |= ((bits >> (3 * (2 * j + half))) & 7u) << (8 * j);
        wide[half] = (int)__builtin_amdgcn_perm(c1, c0, sel);
    }
    return wide;
}

extern "C" __global__ __launch_bounds__(256, 2)
void gemm_mq3v25_residual_mmq_iu8(
    const char* __restrict__ A, const block_i8_128* __restrict__ Xq,
    float* __restrict__ Y, int M, int K, int N, int add
) {
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int rs = blockIdx.x * 128, bs = blockIdx.y * 128;
    const int row_part = wave & 3, token_part = wave >> 2;
    const int wave_row = row_part * 32, wave_token = token_part * 64;
    const int ml = lane & 15, kh = lane >> 4;
    const int gpr = K / 256;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* const a_lds = smem;
    unsigned char* const w_lds = smem + 9216;
    float* const scales_w = (float*)(smem + 18432);
    float* const scales_x = (float*)(smem + 18944);
    const unsigned char* const ar = a_lds + (wave_token >> 4) * 4 * 288 + lane * 8;
    const unsigned char* const wr = w_lds + (wave_row >> 4) * 4 * 288 + lane * 8;
    const int st_row = tid >> 2, quad = tid & 3;
    f32x8 acc[4][2];
    const f32x8 zero_f = {0,0,0,0,0,0,0,0};
    #pragma unroll
    for (int nb = 0; nb < 4; ++nb) {
        acc[nb][0] = zero_f;
        acc[nb][1] = zero_f;
    }
    for (int kb = 0; kb < K / 128; ++kb) {
        if (tid < 128) {
            const int row = rs + tid;
            const int safe = row < M ? row : M - 1;
            const unsigned short d = *(const unsigned short*)(A +
                (long long)safe * gpr * 104 + (kb / 2) * 104 + (kb & 1) * 4);
            scales_w[tid] = row < M ? __half2float(__ushort_as_half(d)) : 0.0f;
        } else {
            const int col = bs + tid - 128;
            scales_x[tid - 128] = col < N ? Xq[(long long)kb * N + col].d : 0.0f;
        }
        i32x8 c[4][2];
        const i32x8 zero_i = {0,0,0,0,0,0,0,0};
        // Keep slab 1's raw global data live while slab 0 runs its WMMAs.
        unsigned pref_w0[2], pref_w1[2];
        unsigned long long pref_x[2][2];
        #pragma unroll
        for (int slab = 0; slab < 2; ++slab) {
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const int local_row = rr * 64 + st_row;
                const int row = rs + local_row, col = bs + local_row;
                const int safe_row = row < M ? row : M - 1;
                const int safe_col = col < N ? col : N - 1;
                const char* wg = A + (long long)safe_row * gpr * 104
                    + (kb / 2) * 104 + 8 + (kb & 1) * 48 + slab * 24;
                const block_i8_128* xb = Xq + (long long)kb * N + safe_col;
                const int frag = ((local_row >> 4) * 4 + quad) * 288 + (local_row & 15) * 8;
                const unsigned char* const p = (const unsigned char*)wg + quad * 6;
                const unsigned packed = slab ? pref_w0[rr] : *(const unsigned*)p;
                const unsigned tail = slab ? pref_w1[rr] : *(const unsigned short*)(p + 4);
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int k8 = slab * 64 + quad * 16 + half * 8;
                    const unsigned bits = half ? ((packed >> 24) | (tail << 8)) : (packed & 0xffffffu);
                    const i32x2 w = row < M ? mq3v25_widen_i8(bits) : i32x2{0,0};
                    const unsigned long long raw = slab ? pref_x[rr][half] : *(const unsigned long long*)(xb->qs + k8);
                    const unsigned long long x = col < N ? raw : 0ull;
                    *(i32x2*)(w_lds + frag + half * 128) = w;
                    *(unsigned long long*)(a_lds + frag + half * 128) = x;
                }
            }
            __syncthreads();
            if (slab == 0) {
                #pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                    const int local_row = rr * 64 + st_row;
                    const int row = rs + local_row, col = bs + local_row;
                    const int safe_row = row < M ? row : M - 1;
                    const int safe_col = col < N ? col : N - 1;
                    const char* wg = A + (long long)safe_row * gpr * 104
                        + (kb / 2) * 104 + 8 + (kb & 1) * 48 + 24;
                    const unsigned char* p = (const unsigned char*)wg + quad * 6;
                    const block_i8_128* xb = Xq + (long long)kb * N + safe_col;
                    pref_w0[rr] = *(const unsigned*)p;
                    pref_w1[rr] = *(const unsigned short*)(p + 4);
                    #pragma unroll
                    for (int half = 0; half < 2; ++half)
                        pref_x[rr][half] = *(const unsigned long long*)(xb->qs + 64 + quad * 16 + half * 8);
                }
            }
            #pragma unroll
            for (int step = 0; step < 4; ++step) {
                i32x2 wf[2], xf[4];
                #pragma unroll
                for (int rg = 0; rg < 2; ++rg)
                    wf[rg] = *(const i32x2*)(wr + (rg * 4 + step) * 288);
                #pragma unroll
                for (int nb = 0; nb < 4; ++nb)
                    xf[nb] = *(const i32x2*)(ar + (nb * 4 + step) * 288);
                #pragma unroll
                for (int nb = 0; nb < 4; ++nb)
                    #pragma unroll
                    for (int rg = 0; rg < 2; ++rg)
                        c[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                            true, wf[rg], true, xf[nb], slab == 0 && step == 0 ? zero_i : c[nb][rg], false);
                __builtin_amdgcn_sched_barrier(0);
            }
            if (slab == 0) __syncthreads(); // only the next slab reuses A/W LDS
        }
        // Integer dot is complete over exactly K128; fold before next block.
        #pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            const float dx = scales_x[wave_token + nb * 16 + ml];
            #pragma unroll
            for (int rg = 0; rg < 2; ++rg)
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int row = wave_row + rg * 16 + 8 * kh + j;
                    const float ds = __fmul_rn(scales_w[row], dx);
                    acc[nb][rg][j] = __fmaf_rn(ds, (float)c[nb][rg][j], acc[nb][rg][j]);
                }
        }
        __syncthreads();
    }
    // Wave-local skewed transpose for contiguous global output stores.
    float* const slots = (float*)smem;
    #pragma unroll
    for (int nb = 0; nb < 4; ++nb) {
        #pragma unroll
        for (int rg = 0; rg < 2; ++rg)
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = (rg * 16 + 8 * kh + j + ml) & 31;
                slots[wave * 512 + ml * 32 + col] = acc[nb][rg][j];
            }
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            const int col = bs + wave_token + nb * 16 + t;
            const int row = rs + wave_row + lane;
            if (col < N && row < M) {
                float* out = Y + (long long)col * M + row;
                const float v = slots[wave * 512 + t * 32 + ((lane + t) & 31)];
                if (add) *out = __fadd_rn(*out, v);
                else *out = v;
            }
        }
        __syncthreads();
    }
}
#endif
