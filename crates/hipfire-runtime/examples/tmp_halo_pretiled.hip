// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniil Markevich
// hipfire — see LICENSE and NOTICE in the project root.
//
// EXPERIMENT (resident pre-tiled 4-bit, gfx1151): throwaway twin of the
// shipping IU4 prefill GEMM that replaces the per-launch A fill (nibble
// gather + f16->f32->half2 header conversion/replication) with a straight
// coalesced copy of a resident pre-tiled image.
//
// Resident image layout (built once on the host, bit-exact by construction):
// per (row-tile of 128 rows T, K256 group kb), the exact 128x42-dword LDS
// image the shipping kernel builds in `tile_x` today:
//   row r (0..127), dwords [0..32):  payload words = MQ4 group bytes
//     [8+4*d, 8+4*d+4) of (row T*128+r, group kb), verbatim order
//     (both shipping fills address lane d -> word d; no reshuffle exists);
//   dwords [32..36): hs0 bits replicated x4; [36..40): hs1 bits x4, where
//     hs0/hs1 are the raw group header u32s (bytes [0..4)/[4..8)). The
//     device fill computes dm = half2(f16(sc),f16(zp)) from the same bits;
//     f16->f32 is exact and RTNE f32->f16 of an exactly-representable value
//     is the identity (Inf included), so the replicated dword == hs bits.
//     The host relayout performs the same RTNE conversion in software rather
//     than assuming it, so NaN payloads (never present in practice) still
//     match the device's v_cvt_f16_f32 path;
//   dwords [40..42): zero pad (shipping never writes or reads these LDS
//     words; the consumer only touches [0..40)).
// tile_bytes = 128*42*4 = 21,504 B per (T,kb), stored contiguously at dword
// offset (T*G + kb)*5376 where G = K/256. Total = M/128*K/256*21,504 B.
//
// This file is concatenated AFTER block_i4_128_quant.hip and
// gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip by the oracle example, so all
// IU4_* macros, tile<>, IU4_FOLD_RN, vec_dot_i4_x128 (8-wave) and
// vec_dot_i4_x128_lf16_gfx1151 (16-wave) are already declared. Do not
// redeclare them here.

#if defined(__gfx1151__)

// Image offsets fit 32 bits comfortably: the largest production image
// (down 5120x17408: 40 tiles x 68 groups x 5376 dwords = 14.6M dwords) is
// far below 2^32. All addressing below is base + 32-bit offset so no 64-bit
// address computation stays live in the kb loop (a 64-bit revision spilled
// 19 VGPR / 64 B scratch vs shipping's 4 / 20 B; 32-bit is at parity).
//
// 16-wave set tile: 512 threads, 5376 dwords = 10 full rounds + 256 tail.
// Lane-interleaved: consecutive lanes read consecutive global dwords
// (coalesced 128 B transactions) and write consecutive LDS banks
// (bank = dword % 32; no conflicts). Same lane math as the Xq fills.
static __device__ __forceinline__ void copy_iu4_pretiled_tile_lf16_gfx1151(
    const int* __restrict__ PT,
    int* __restrict__ tile_x,
    unsigned tile_img,
    int kb
) {
    const unsigned img = tile_img + (unsigned)kb * 5376u;
    #pragma unroll 1
    for (int l = threadIdx.y * IU4_WARP_SIZE + threadIdx.x; l < 5376; l += 512)
        tile_x[l] = PT[img + (unsigned)l];
}

// 8-wave add tile: 256 threads, 5376 dwords = 21 rounds exactly, no mask.
static __device__ __forceinline__ void copy_iu4_pretiled_tile_occ3_gfx1151(
    const int* __restrict__ PT,
    int* __restrict__ tile_x,
    unsigned tile_img,
    int kb
) {
    const unsigned img = tile_img + (unsigned)kb * 5376u;
    #pragma unroll 1
    for (int l = threadIdx.y * IU4_WARP_SIZE + threadIdx.x; l < 5376; l += 256)
        tile_x[l] = PT[img + (unsigned)l];
}

// Pretiled 16-wave set body: byte-identical copy of
// gemm_iu4_body_lf16_gfx1151<false> except the A fill above replaces
// load_iu4_tile_lf16_gfx1151. Xq fills, barriers, dots, fold, epilogue,
// grid walk (col-adjacent) and the always-true h1 guard are untouched;
// the h1 guard stays a branch for the same allocator reason quoted there.
static __device__ __forceinline__ void gemm_iu4_pretiled_body_set_gfx1151(
    const int* __restrict__ PT,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add,
    int row_tile,
    int col_tile
) {
    (void)add;
    const int row0 = row_tile * IU4_MMQ_Y;
    const int col0 = col_tile * IU4_MMQ_X;
    const int groups_per_row = K / 256;
    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + IU4_MMQ_X * IU4_TILE_Y_K;
    float sum[IU4_LF16_SUMS] = {0.0f};
    const unsigned tile_img = (unsigned)row_tile * (unsigned)groups_per_row * 5376u;
    for (int kb = 0; kb < groups_per_row; ++kb) {
        copy_iu4_pretiled_tile_lf16_gfx1151(PT, tile_x, tile_img, kb);
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        #pragma unroll 1
        for (int l = threadIdx.y * IU4_WARP_SIZE + threadIdx.x; l < IU4_LF16_XQ_WORDS; l += 512)
            tile_y[l] = by0[l];
        __syncthreads();
        vec_dot_i4_x128_lf16_gfx1151(tile_x, tile_y, sum, 0);
        __syncthreads();
        if (2 * kb + 1 < 2 * groups_per_row) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            #pragma unroll 1
            for (int l = threadIdx.y * IU4_WARP_SIZE + threadIdx.x; l < IU4_LF16_XQ_WORDS; l += 512)
                tile_y[l] = by1[l];
            __syncthreads();
            vec_dot_i4_x128_lf16_gfx1151(tile_x, tile_y, sum, 1);
            __syncthreads();
        }
    }
    {
        const int tid_y = (int)__builtin_amdgcn_workitem_id_y();
        int row_part, col_part, tmp;
        asm volatile(
            "v_mbcnt_lo_u32_b32 %2, -1, 0\n\t"
            "v_lshrrev_b32_e32 %0, 1, %3\n\t"
            "v_bfe_u32 %1, %2, 4, 1\n\t"
            "v_lshl_or_b32 %0, %0, 4, %1\n\t"
            "v_and_b32_e32 %1, 1, %3\n\t"
            "v_and_b32_e32 %2, 15, %2\n\t"
            "v_lshl_or_b32 %1, %1, 6, %2"
            : "=&v"(row_part), "=&v"(col_part), "=&v"(tmp)
            : "v"(tid_y), "v"(sum[0]));
        const int row0e = row_tile * IU4_MMQ_Y;
        const int col0e = col_tile * IU4_MMQ_X;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            #pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int i = row_part + 2 * l;
                const int j = col_part + 16 * c;
                float* yp = Y + (long long)(col0e + j) * M + (row0e + i);
                const float v = sum[8 * c + l];
                *yp = v;
            }
        }
    }
}

// Pretiled 8-wave add body: byte-identical copy of gemm_iu4_body<true>
// (the shipping occ3 path) except the A fill above replaces
// load_iu4_tile<true>. Same Xq stride loop over 256 threads, same two
// barriers per half, same h0-then-h1 accumulation, same residual add.
template <bool FULL>
static __device__ __forceinline__ void gemm_iu4_pretiled_body_occ3(
    const int* __restrict__ PT,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add,
    int row_tile,
    int col_tile
) {
    const int row0 = row_tile * IU4_MMQ_Y;
    const int col0 = col_tile * IU4_MMQ_X;
    if (!FULL && (row0 >= M || col0 >= N)) return;

    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;

    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + IU4_MMQ_X * IU4_TILE_Y_K;

    float sum[IU4_MMQ_X * IU4_MMQ_Y / (IU4_MMQ_NWARPS * IU4_WARP_SIZE)] = {0.0f};

    const unsigned tile_img = (unsigned)row_tile * (unsigned)groups_per_row * 5376u;
    for (int kb = 0; kb < groups_per_row; ++kb) {
        copy_iu4_pretiled_tile_occ3_gfx1151(PT, tile_x, tile_img, kb);

        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        for (int l0 = 0; l0 < IU4_MMQ_X * IU4_TILE_Y_K; l0 += IU4_MMQ_NWARPS * IU4_WARP_SIZE) {
            const int l = l0 + threadIdx.y * IU4_WARP_SIZE + threadIdx.x;
            if (FULL || col0 + l / IU4_TILE_Y_K < N) tile_y[l] = by0[l];
            else tile_y[l] = 0;
        }
        __syncthreads();
        vec_dot_i4_x128(tile_x, tile_y, sum, 0);
        __syncthreads();

        if (2 * kb + 1 < blocks128) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            for (int l0 = 0; l0 < IU4_MMQ_X * IU4_TILE_Y_K; l0 += IU4_MMQ_NWARPS * IU4_WARP_SIZE) {
                const int l = l0 + threadIdx.y * IU4_WARP_SIZE + threadIdx.x;
                if (FULL || col0 + l / IU4_TILE_Y_K < N) tile_y[l] = by1[l];
                else tile_y[l] = 0;
            }
            __syncthreads();
            vec_dot_i4_x128(tile_x, tile_y, sum, 1);
            __syncthreads();
        }
    }

    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    #pragma unroll
    for (int j0 = 0; j0 < IU4_MMQ_X; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if (FULL || (row0 + i < M && col0 + j < N)) {
                    float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                    const float v = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
                    if (add) {
                        *yp += v;
                    } else {
                        *yp = v;
                    }
                }
            }
        }
    }
}

// Pretiled 16-wave set body with next-group LDS prefetch (_pf): same
// arithmetic as the set body above, but the A plane is double-buffered in
// LDS (2 x 21,504 B + 9,216 B Xq = 52,224 B; needs SHARED_PF from the host)
// and group kb+1 is fetched from the resident image at the top of iteration
// kb, overlapping the current group's stores/dots in issue order.
// A register double-buffer was measured first: 128 VGPR, 0 spills — but a
// 16-wave workgroup needs <=96 VGPR for full SIMD residency (16x128 = 2048
// > 1536 file), so the register form cannot hold even one workgroup per CU
// and is not shipped. The LDS form adds ~2 pointer VGPRs and keeps the 96
// cap; LDS caps it at 1 WG/CU (52224 > 32768) — prefetch overlap vs halved
// WG residency is exactly what the oracle measures.
// Barrier safety: prefetch writes buf_nxt while dots read buf_cur (disjoint
// addresses, no barrier needed); the next iteration reads buf_nxt-as-cur
// only after its Xq-fill __syncthreads, which also retires the prefetch;
// the swap itself follows the h1 __syncthreads, retiring the last read of
// the buffer the next prefetch will overwrite.
static __device__ __forceinline__ void gemm_iu4_pretiled_body_set_pf_gfx1151(
    const int* __restrict__ PT,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add,
    int row_tile,
    int col_tile
) {
    (void)add;
    const int row0 = row_tile * IU4_MMQ_Y;
    const int col0 = col_tile * IU4_MMQ_X;
    const int groups_per_row = K / 256;
    const unsigned tile_img = (unsigned)row_tile * (unsigned)groups_per_row * 5376u;
    extern __shared__ int smem[];
    int* tile_y = smem;
    int* buf0 = tile_y + IU4_MMQ_X * IU4_TILE_Y_K;
    int* buf1 = buf0 + 5376;
    float sum[IU4_LF16_SUMS] = {0.0f};
    #pragma unroll 1
    for (int l = threadIdx.y * IU4_WARP_SIZE + threadIdx.x; l < 5376; l += 512)
        buf0[l] = PT[tile_img + (unsigned)l];
    int* tile_x = buf0;
    int* tile_nxt = buf1;
    for (int kb = 0; kb < groups_per_row; ++kb) {
        if (kb + 1 < groups_per_row) {
            const unsigned imgn = tile_img + (unsigned)(kb + 1) * 5376u;
            #pragma unroll 1
            for (int l = threadIdx.y * IU4_WARP_SIZE + threadIdx.x; l < 5376; l += 512)
                tile_nxt[l] = PT[imgn + (unsigned)l];
        }
        const int* by0 = (const int*)(Xq + (long long)(2 * kb) * N + col0);
        #pragma unroll 1
        for (int l = threadIdx.y * IU4_WARP_SIZE + threadIdx.x; l < IU4_LF16_XQ_WORDS; l += 512)
            tile_y[l] = by0[l];
        __syncthreads();
        vec_dot_i4_x128_lf16_gfx1151(tile_x, tile_y, sum, 0);
        __syncthreads();
        if (2 * kb + 1 < 2 * groups_per_row) {
            const int* by1 = (const int*)(Xq + (long long)(2 * kb + 1) * N + col0);
            #pragma unroll 1
            for (int l = threadIdx.y * IU4_WARP_SIZE + threadIdx.x; l < IU4_LF16_XQ_WORDS; l += 512)
                tile_y[l] = by1[l];
            __syncthreads();
            vec_dot_i4_x128_lf16_gfx1151(tile_x, tile_y, sum, 1);
            __syncthreads();
        }
        int* tmp = tile_x; tile_x = tile_nxt; tile_nxt = tmp;
    }
    {
        const int tid_y = (int)__builtin_amdgcn_workitem_id_y();
        int row_part, col_part, tmp;
        asm volatile(
            "v_mbcnt_lo_u32_b32 %2, -1, 0\n\t"
            "v_lshrrev_b32_e32 %0, 1, %3\n\t"
            "v_bfe_u32 %1, %2, 4, 1\n\t"
            "v_lshl_or_b32 %0, %0, 4, %1\n\t"
            "v_and_b32_e32 %1, 1, %3\n\t"
            "v_and_b32_e32 %2, 15, %2\n\t"
            "v_lshl_or_b32 %1, %1, 6, %2"
            : "=&v"(row_part), "=&v"(col_part), "=&v"(tmp)
            : "v"(tid_y), "v"(sum[0]));
        const int row0e = row_tile * IU4_MMQ_Y;
        const int col0e = col_tile * IU4_MMQ_X;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            #pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int i = row_part + 2 * l;
                const int j = col_part + 16 * c;
                float* yp = Y + (long long)(col0e + j) * M + (row0e + i);
                const float v = sum[8 * c + l];
                *yp = v;
            }
        }
    }
}

// Entries. The set/add pair keeps shipping's launch bounds and VGPR policy
// (set: 512 threads/min-2-WG + 96 cap; add: 256 threads/min-3-WG, uncapped
// like shipping occ3). The _pf entry keeps the 96 cap at min-1-WG: its LDS
// (52,224 B) already caps it at 1 WG/CU, so no VGPR cap is sacrificed.
// Kernarg ABI is the same seven-arg (A,Xq,Y,M,K,N,add) layout: A now points
// at the resident pre-tiled image (dword-addressed inside).
extern "C" __global__ __launch_bounds__(512, 2) __attribute__((amdgpu_num_vgpr(96))) void gemm_iu4_pretiled_set_gfx1151(
    const int* __restrict__ PT,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    gemm_iu4_pretiled_body_set_gfx1151(PT, Xq, Y, M, K, N, add, blockIdx.y, blockIdx.x);
}
extern "C" __global__ __launch_bounds__(256, 3) void gemm_iu4_pretiled_add_gfx1151(
    const int* __restrict__ PT,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    gemm_iu4_pretiled_body_occ3<true>(PT, Xq, Y, M, K, N, add, blockIdx.y, blockIdx.x);
}
extern "C" __global__ __launch_bounds__(512, 1) __attribute__((amdgpu_num_vgpr(96))) void gemm_iu4_pretiled_set_pf_gfx1151(
    const int* __restrict__ PT,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    gemm_iu4_pretiled_body_set_pf_gfx1151(PT, Xq, Y, M, K, N, add, blockIdx.y, blockIdx.x);
}

#endif
