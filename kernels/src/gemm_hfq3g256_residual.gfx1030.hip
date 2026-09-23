#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HFQ3-G256 batched single-output GEMM with fused residual add for
// gfx1030 (RDNA2). Sister of gemm_hfq3g256_residual_wmma.hip; same
// contract and per-lane output layout (acc[j] = C[2*j + (tid>>4)][tid & 15])
// + same residual semantics (Y += acc, caller initializes Y with residual
// stream). No WMMA — fdot2 FP32 FMA via LDS-shared dequantized A.
//
// See gemm_gate_up_hfq3g256.gfx1030.hip for full template comments.
// Differences from gate_up: single output projection (no row routing),
// writeback uses += instead of = (residual fusion).
//
// Uses __builtin_amdgcn_fdot2 (v_dot2_f32_f16) for 2× FP16 FMA throughput.
// Double-buffered LDS pipelines dequant of K-tile N+1 with the dot
// product of K-tile N (1 __syncthreads per iter).

typedef _Float16 __attribute__((ext_vector_type(2))) half2_t;

// Helper: dequantize one HFQ3 K-tile (16 FP16 weights from 6 bytes) into
// lds_buf[my_lds_row][0..16]. Called by lanes 0..15 only.
__device__ static inline void dq_hfq3_ktile(
    _Float16 (&lds_buf)[16][16],
    int my_lds_row,
    const unsigned char* __restrict__ dpkt,
    _Float16 sc_h,
    _Float16 zp_h
) {
    unsigned int a0 = dpkt[0], a1 = dpkt[1], a2 = dpkt[2];
    unsigned int a3 = dpkt[3], a4 = dpkt[4], a5 = dpkt[5];
    unsigned int q0  = a0 & 7u;
    unsigned int q1  = (a0 >> 3) & 7u;
    unsigned int q2  = ((a0 >> 6) | (a1 << 2)) & 7u;
    unsigned int q3  = (a1 >> 1) & 7u;
    unsigned int q4  = (a1 >> 4) & 7u;
    unsigned int q5  = ((a1 >> 7) | (a2 << 1)) & 7u;
    unsigned int q6  = (a2 >> 2) & 7u;
    unsigned int q7  = (a2 >> 5) & 7u;
    unsigned int q8  = a3 & 7u;
    unsigned int q9  = (a3 >> 3) & 7u;
    unsigned int q10 = ((a3 >> 6) | (a4 << 2)) & 7u;
    unsigned int q11 = (a4 >> 1) & 7u;
    unsigned int q12 = (a4 >> 4) & 7u;
    unsigned int q13 = ((a4 >> 7) | (a5 << 1)) & 7u;
    unsigned int q14 = (a5 >> 2) & 7u;
    unsigned int q15 = (a5 >> 5) & 7u;
    #define DQ(i, q) lds_buf[my_lds_row][i] = sc_h * (_Float16)(float)(q) + zp_h
    DQ(0, q0);   DQ(1, q1);   DQ(2, q2);   DQ(3, q3);
    DQ(4, q4);   DQ(5, q5);   DQ(6, q6);   DQ(7, q7);
    DQ(8, q8);   DQ(9, q9);   DQ(10, q10); DQ(11, q11);
    DQ(12, q12); DQ(13, q13); DQ(14, q14); DQ(15, q15);
    #undef DQ
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_hfq3g256_residual_gfx1030(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;

    if (row_start >= M || batch_start >= batch_size) return;

    const int my_lds_row = tid & 15;
    int safe_row = row_start + my_lds_row;
    if (safe_row >= M) safe_row = M - 1;
    const int my_col_in_tile = tid & 15;
    const int my_col_global = batch_start + my_col_in_tile;
    const int safe_col = (my_col_global < batch_size) ? my_col_global : 0;

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 104;
    const char* row_base = A + safe_row * row_stride;
    const _Float16* x_base = X + (long long)safe_col * K;

    // Double-buffered LDS: 2× 16×16 FP16 = 1024 bytes.
    __shared__ _Float16 lds_a[2][16][16];
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 104;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const _Float16* xg = x_base + g * 256;
        const unsigned char* dp = (const unsigned char*)(gp + 8);

        // Prefetch K-tile 0 into lds_a[0] before entering the pipelined loop.
        if ((tid >> 4) == 0) {
            dq_hfq3_ktile(lds_a[0], my_lds_row, dp, sc_h, zp_h);
        }
        __syncthreads();

        for (int kt = 0; kt < 16; kt++) {
            const int cur_buf = kt & 1;
            const int next_kt = kt + 1;
            const int next_buf = next_kt & 1;
            const int k_off = kt * 16;

            // Load X for current K-tile.
            half2_t x_pairs[8];
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                x_pairs[p] = *(const half2_t*)(xg + k_off + p * 2);
            }

            // Prefetch K-tile kt+1 into lds_a[next_buf] (skip on last K-tile).
            // Runs concurrently with the dot product on lds_a[cur_buf].
            if (next_kt < 16 && (tid >> 4) == 0) {
                dq_hfq3_ktile(lds_a[next_buf], my_lds_row, dp + next_kt * 6, sc_h, zp_h);
            }

            // Compute using cur_buf (already populated by previous iter's
            // prefetch or the per-group K-tile 0 prefetch above).
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int out_row_in_tile = 2 * j + (tid >> 4);
                const _Float16* a_row = &lds_a[cur_buf][out_row_in_tile][0];
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    half2_t a_pair = *(const half2_t*)(a_row + p * 2);
                    acc[j] = __builtin_amdgcn_fdot2(a_pair, x_pairs[p], acc[j], false);
                }
            }

            __syncthreads();
        }
    }

    const int out_col = batch_start + my_col_in_tile;
    if (out_col < batch_size) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                // Residual fusion: caller initialized Y with residual stream.
                Y[(long long)out_col * M + out_row] += acc[j];
            }
        }
    }
}
