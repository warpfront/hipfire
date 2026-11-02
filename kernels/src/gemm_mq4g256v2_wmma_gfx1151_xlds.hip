// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exact-gfx1151 G256 4-bit WMMA GEMMs for small batches (the 16-row DFlash
// verify): one module, eight symbols with the base kernels' ABIs.
//   MQ4G256V2 (dual-half (s0,z0 | s1,z1) header):
//   gemm_gate_up_mq4g256v2_wmma_gfx1151_xlds      (= gemm_gate_up_mq4g256v2_wmma)
//   gemm_qkvza_mq4g256v2_wmma_gfx1151_xlds        (= gemm_qkvza_mq4g256v2_wmma)
//   gemm_qkv_mq4g256v2_wmma_gfx1151_xlds          (= gemm_qkv_mq4g256v2_wmma)
//   gemm_mq4g256v2_residual_wmma_gfx1151_xlds     (= gemm_mq4g256v2_residual_wmma, Y += W@X)
//   HFQ4G256 / MQ4G256 v1 (one f32 (scale, zero) header per group):
//   gemm_gate_up_hfq4g256_wmma_gfx1151_xlds       (= gemm_gate_up_hfq4g256_wmma{,_ldscoop_nosync})
//   gemm_qkvza_hfq4g256_wmma_gfx1151_xlds         (= gemm_qkvza_hfq4g256_wmma)
//   gemm_qkv_hfq4g256_wmma_gfx1151_xlds           (= gemm_qkv_hfq4g256_wmma)
//   gemm_hfq4g256_residual_wmma_gfx1151_xlds      (= gemm_hfq4g256_residual_wmma_k2, Y += W@X)
//
// Per output the arithmetic is the base gfx11 kernels' exactly, so outputs are
// byte-identical to them:
//   - one wave owns 16 weight rows x 16 batch rows and runs the whole K chain
//     acc = wmma_f32_16x16x16_f16(A_t, B_t, acc) for t = 0..K/16-1 in order,
//     from acc = 0 (no split-K, no reduction across waves);
//   - A_t[i] = fma_f16(scale, (f16)nibble_i, zero) with the group's header:
//     V2 tiles 0..7 use (s0, z0), 8..15 use (s1, z1); V1 uses the group's
//     ((f16)scale_f32, (f16)zero_f32) for all 16 tiles. That is the value the
//     base kernels' scalar `sc * (f16)(float)q + zp` (contracted to v_fma_f16)
//     produces; here one v_pk_fma_f16 per pair, with the nibble made exact by
//     the 0x6400 | q magic (1024 + q) minus 1024;
//   - B_t is the same 16 x 16 fp16 X tile; the epilogue is the base one
//     (overwrite, or Y += acc for the residual).
// What differs is memory access only. A block runs 4 waves (64 weight rows,
// one 16-row tile each) and stages each 256-K slab of the X panel in LDS once,
// double-buffered, instead of every wave re-reading the whole N x K panel from
// L2; each lane loads its row's 136 B group (header + 128 B nibbles) with one
// b64 and eight b128 loads. On gfx1151 this moves the 16-row 27B verify GEMMs
// from ~131-158 GB/s to ~183-206 GB/s (kernel-level, Strix Halo).
//
// Grid: [ceil(total_m / 64), ceil(N / 16), 1]. Block: [128, 1, 1].
// Static LDS: 2 x 16 x 264 halfs = 16,896 B. Requires K % 256 == 0.
// Waves whose 16-row tile starts past total_m still run the loop (clamped
// rows) so the block barriers stay uniform; their stores are masked.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef _Float16 __attribute__((ext_vector_type(2))) half2_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef unsigned int __attribute__((ext_vector_type(8))) u8_t;
typedef unsigned int __attribute__((ext_vector_type(4))) u4_t;
// Group payloads start 8 B into a 136 B group: 4-byte alignment contract.
typedef unsigned int __attribute__((ext_vector_type(4), aligned(4))) u4a_t;
typedef unsigned int __attribute__((ext_vector_type(2), aligned(4))) u2a_t;

#define XLDS_WAVES 4
#define XLDS_THREADS (32 * XLDS_WAVES)
// X slab row pitch 264 halfs (528 B): 16 lanes' b128 reads hit distinct banks.
#define XLDS_PITCH 264
#define XLDS_SLAB (16 * XLDS_PITCH)
#define XLDS_PER ((512 + XLDS_THREADS - 1) / XLDS_THREADS)

__device__ __forceinline__ half2_t xlds_h2(unsigned int u) { return __builtin_bit_cast(half2_t, u); }
__device__ __forceinline__ unsigned int xlds_u(half2_t h) { return __builtin_bit_cast(unsigned int, h); }

// Eight halfs (n0..n7 of `pk`) as four packed pairs (n0,n1),(n2,n3),(n4,n5),(n6,n7).
// v_perm puts byte i of pk (low nibble n_{2i}) and byte i of pk >> 4 (low nibble
// n_{2i+1}) in the two half lanes; mask + 0x6400 gives the exact halfs 1024 + n.
__device__ __forceinline__ void xlds_dq8(unsigned int pk, half2_t sc2, half2_t zp2, unsigned int* o) {
    const half2_t m1024 = xlds_h2(0xE400E400u);
    const unsigned int s = pk >> 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned int sel = 0x0c000c00u | ((unsigned int)(4 + i) << 16) | (unsigned int)i;
        const unsigned int x = (__builtin_amdgcn_perm(s, pk, sel) & 0x000F000Fu) | 0x64006400u;
        o[i] = xlds_u(__builtin_elementwise_fma(xlds_h2(x) + m1024, sc2, zp2));
    }
}

// One K tile's A fragment (16 halfs) from its two nibble dwords and the
// governing (scale, zero) header dword.
__device__ __forceinline__ half16_t xlds_dq16(unsigned int pk0, unsigned int pk1, unsigned int hdr) {
    const unsigned int s = hdr & 0xFFFFu, z = hdr >> 16;
    const half2_t sc2 = xlds_h2(s | (s << 16));
    const half2_t zp2 = xlds_h2(z | (z << 16));
    unsigned int t[8];
    xlds_dq8(pk0, sc2, zp2, t);
    xlds_dq8(pk1, sc2, zp2, t + 4);
    u8_t o;
    o[0] = t[0]; o[1] = t[1]; o[2] = t[2]; o[3] = t[3];
    o[4] = t[4]; o[5] = t[5]; o[6] = t[6]; o[7] = t[7];
    return __builtin_bit_cast(half16_t, o);
}

#define XLDS_TILE(kt, pk0, pk1, hdr)                                                              \
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(                                             \
        xlds_dq16((pk0), (pk1), (hdr)), *(const half16_t*)(xs + lane16 * XLDS_PITCH + (kt) * 16), \
        acc)

// The whole K chain of this wave's 16 rows x 16 batch rows.
// row_base: this lane's (clamped) weight row. xs_lds: 2 X slabs.
// V1: the 8 B header is f32 (scale, zero), converted to f16 like the base kernels.
template <bool V1>
__device__ __forceinline__ float8_t xlds_mainloop(const char* row_base, const _Float16* X,
                                                  int batch_start, int N, int K,
                                                  _Float16* xs_lds) {
    const int tid = threadIdx.x;
    const int lane16 = tid & 15;
    const int G = K >> 8;
    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    // Cooperative X slab fill: 16 batch rows x 256 halfs = 512 vectors of 16 B.
    // Invalid batch rows read batch row 0, as the base kernels do.
    u4_t xr[XLDS_PER];
    auto x_fetch = [&](int g) {
#pragma unroll
        for (int i = 0; i < XLDS_PER; ++i) {
            const int v = tid + i * XLDS_THREADS;
            const int b = v >> 5, c = v & 31;
            const int sb = (batch_start + b < N) ? (batch_start + b) : 0;
            xr[i] = *(const u4_t*)(X + (long long)sb * K + g * 256 + c * 8);
        }
    };
    auto x_store = [&](int buf) {
#pragma unroll
        for (int i = 0; i < XLDS_PER; ++i) {
            const int v = tid + i * XLDS_THREADS;
            const int b = v >> 5, c = v & 31;
            *(u4_t*)(xs_lds + buf * XLDS_SLAB + b * XLDS_PITCH + c * 8) = xr[i];
        }
    };
    x_fetch(0);
    x_store(0);
    __syncthreads();
    for (int g = 0; g < G; ++g) {
        const char* gp = row_base + (long long)g * 136;
        const u2a_t hv = *(const u2a_t*)gp;
        unsigned int h0 = hv.x, h1 = hv.y;
        if (V1) {
            // Bit-cast the scalars, not hv.y: clang lowers __builtin_bit_cast of an
            // ext-vector element lvalue as a load from the vector's base (lane 0).
            half2_t sz;
            sz.x = (_Float16)__builtin_bit_cast(float, h0);
            sz.y = (_Float16)__builtin_bit_cast(float, h1);
            h0 = h1 = xlds_u(sz);
        }
        const u4a_t* p = (const u4a_t*)(gp + 8);
        const u4_t w0 = p[0], w1 = p[1], w2 = p[2], w3 = p[3];
        const u4_t w4 = p[4], w5 = p[5], w6 = p[6], w7 = p[7];
        if (g + 1 < G) x_fetch(g + 1);
        const _Float16* xs = xs_lds + (g & 1) * XLDS_SLAB;
        XLDS_TILE(0,  w0.x, w0.y, h0); XLDS_TILE(1,  w0.z, w0.w, h0);
        XLDS_TILE(2,  w1.x, w1.y, h0); XLDS_TILE(3,  w1.z, w1.w, h0);
        XLDS_TILE(4,  w2.x, w2.y, h0); XLDS_TILE(5,  w2.z, w2.w, h0);
        XLDS_TILE(6,  w3.x, w3.y, h0); XLDS_TILE(7,  w3.z, w3.w, h0);
        XLDS_TILE(8,  w4.x, w4.y, h1); XLDS_TILE(9,  w4.z, w4.w, h1);
        XLDS_TILE(10, w5.x, w5.y, h1); XLDS_TILE(11, w5.z, w5.w, h1);
        XLDS_TILE(12, w6.x, w6.y, h1); XLDS_TILE(13, w6.z, w6.w, h1);
        XLDS_TILE(14, w7.x, w7.y, h1); XLDS_TILE(15, w7.z, w7.w, h1);
        // Slab (g + 1) & 1 was last read in group g - 1, before that group's barrier.
        if (g + 1 < G) x_store((g + 1) & 1);
        __syncthreads();
    }
    return acc;
}

#define XLDS_PROLOGUE(TOTAL_M)                                                          \
    __shared__ __align__(16) _Float16 xs_lds[2 * XLDS_SLAB];                            \
    const int lane = threadIdx.x & 31;                                                  \
    const int total_m = (TOTAL_M);                                                      \
    const int row_start = (blockIdx.x * XLDS_WAVES + (int)(threadIdx.x >> 5)) * 16;     \
    const int batch_start = blockIdx.y * 16;                                            \
    const int my_row = row_start + (lane & 15);                                         \
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);                   \
    const long long row_stride = (long long)(K / 256) * 136;                            \
    const int out_col = batch_start + (lane & 15);                                      \
    const bool store = row_start < total_m && out_col < N;

// One symbol per header format (V2 = MQ4G256V2, V1 = HFQ4G256/MQ4G256) for each body.
#define XLDS_ENTRIES(BODY, V2_NAME, V1_NAME, PARAMS, ARGS)                                  \
    __launch_bounds__(XLDS_THREADS) extern "C" __global__ void V2_NAME PARAMS { BODY<false> ARGS; } \
    __launch_bounds__(XLDS_THREADS) extern "C" __global__ void V1_NAME PARAMS { BODY<true> ARGS; }

template <bool V1>
__device__ __forceinline__ void xlds_gate_up(
    const char* __restrict__ A_gate, const char* __restrict__ A_up,
    const _Float16* __restrict__ X,
    float* __restrict__ Y_gate, float* __restrict__ Y_up,
    int gate_m, int up_m, int K, int N) {
    XLDS_PROLOGUE(gate_m + up_m)
    const char* A; int local_row;
    if (safe_row < gate_m) { A = A_gate; local_row = safe_row; }
    else { A = A_up; local_row = safe_row - gate_m; }
    const float8_t acc = xlds_mainloop<V1>(A + local_row * row_stride, X, batch_start, N, K, xs_lds);
    if (store) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (lane >> 4);
            if (out_row < total_m) {
                float* Y; int proj_row; int out_stride;
                if (out_row < gate_m) { Y = Y_gate; proj_row = out_row; out_stride = gate_m; }
                else { Y = Y_up; proj_row = out_row - gate_m; out_stride = up_m; }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}

XLDS_ENTRIES(xlds_gate_up, gemm_gate_up_mq4g256v2_wmma_gfx1151_xlds, gemm_gate_up_hfq4g256_wmma_gfx1151_xlds,
    (const char* __restrict__ A_gate, const char* __restrict__ A_up, const _Float16* __restrict__ X,
     float* __restrict__ Y_gate, float* __restrict__ Y_up, int gate_m, int up_m, int K, int N),
    (A_gate, A_up, X, Y_gate, Y_up, gate_m, up_m, K, N))

template <bool V1>
__device__ __forceinline__ void xlds_qkvza(
    const char* __restrict__ A_qkv, const char* __restrict__ A_z,
    const char* __restrict__ A_beta, const char* __restrict__ A_alpha,
    const _Float16* __restrict__ X,
    float* __restrict__ Y_qkv, float* __restrict__ Y_z,
    float* __restrict__ Y_beta, float* __restrict__ Y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m, int K, int N) {
    XLDS_PROLOGUE(qkv_m + z_m + beta_m + alpha_m)
    const char* A; int local_row;
    if (safe_row < qkv_m) { A = A_qkv; local_row = safe_row; }
    else if (safe_row < qkv_m + z_m) { A = A_z; local_row = safe_row - qkv_m; }
    else if (safe_row < qkv_m + z_m + beta_m) { A = A_beta; local_row = safe_row - (qkv_m + z_m); }
    else { A = A_alpha; local_row = safe_row - (qkv_m + z_m + beta_m); }
    const float8_t acc = xlds_mainloop<V1>(A + local_row * row_stride, X, batch_start, N, K, xs_lds);
    if (store) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (lane >> 4);
            if (out_row < total_m) {
                float* Y; int proj_row; int out_stride;
                if (out_row < qkv_m) { Y = Y_qkv; proj_row = out_row; out_stride = qkv_m; }
                else if (out_row < qkv_m + z_m) { Y = Y_z; proj_row = out_row - qkv_m; out_stride = z_m; }
                else if (out_row < qkv_m + z_m + beta_m) { Y = Y_beta; proj_row = out_row - (qkv_m + z_m); out_stride = beta_m; }
                else { Y = Y_alpha; proj_row = out_row - (qkv_m + z_m + beta_m); out_stride = alpha_m; }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}

XLDS_ENTRIES(xlds_qkvza, gemm_qkvza_mq4g256v2_wmma_gfx1151_xlds, gemm_qkvza_hfq4g256_wmma_gfx1151_xlds,
    (const char* __restrict__ A_qkv, const char* __restrict__ A_z, const char* __restrict__ A_beta,
     const char* __restrict__ A_alpha, const _Float16* __restrict__ X, float* __restrict__ Y_qkv,
     float* __restrict__ Y_z, float* __restrict__ Y_beta, float* __restrict__ Y_alpha,
     int qkv_m, int z_m, int beta_m, int alpha_m, int K, int N),
    (A_qkv, A_z, A_beta, A_alpha, X, Y_qkv, Y_z, Y_beta, Y_alpha, qkv_m, z_m, beta_m, alpha_m, K, N))

template <bool V1>
__device__ __forceinline__ void xlds_qkv(
    const char* __restrict__ A_q, const char* __restrict__ A_k, const char* __restrict__ A_v,
    const _Float16* __restrict__ X,
    float* __restrict__ Y_q, float* __restrict__ Y_k, float* __restrict__ Y_v,
    int q_m, int k_m, int v_m, int K, int N) {
    XLDS_PROLOGUE(q_m + k_m + v_m)
    const char* A; int local_row;
    if (safe_row < q_m) { A = A_q; local_row = safe_row; }
    else if (safe_row < q_m + k_m) { A = A_k; local_row = safe_row - q_m; }
    else { A = A_v; local_row = safe_row - q_m - k_m; }
    const float8_t acc = xlds_mainloop<V1>(A + local_row * row_stride, X, batch_start, N, K, xs_lds);
    if (store) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (lane >> 4);
            if (out_row < total_m) {
                float* Y; int proj_row; int out_stride;
                if (out_row < q_m) { Y = Y_q; proj_row = out_row; out_stride = q_m; }
                else if (out_row < q_m + k_m) { Y = Y_k; proj_row = out_row - q_m; out_stride = k_m; }
                else { Y = Y_v; proj_row = out_row - q_m - k_m; out_stride = v_m; }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}

XLDS_ENTRIES(xlds_qkv, gemm_qkv_mq4g256v2_wmma_gfx1151_xlds, gemm_qkv_hfq4g256_wmma_gfx1151_xlds,
    (const char* __restrict__ A_q, const char* __restrict__ A_k, const char* __restrict__ A_v,
     const _Float16* __restrict__ X, float* __restrict__ Y_q, float* __restrict__ Y_k,
     float* __restrict__ Y_v, int q_m, int k_m, int v_m, int K, int N),
    (A_q, A_k, A_v, X, Y_q, Y_k, Y_v, q_m, k_m, v_m, K, N))

template <bool V1>
__device__ __forceinline__ void xlds_residual(
    const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y,
    int M, int K, int batch_size) {
    const int N = batch_size;
    XLDS_PROLOGUE(M)
    const float8_t acc = xlds_mainloop<V1>(A + (long long)safe_row * row_stride, X, batch_start, N, K, xs_lds);
    if (store) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (lane >> 4);
            if (out_row < M) Y[(long long)out_col * M + out_row] += acc[j];
        }
    }
}

XLDS_ENTRIES(xlds_residual, gemm_mq4g256v2_residual_wmma_gfx1151_xlds, gemm_hfq4g256_residual_wmma_gfx1151_xlds,
    (const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y,
     int M, int K, int batch_size),
    (A, X, Y, M, K, batch_size))
