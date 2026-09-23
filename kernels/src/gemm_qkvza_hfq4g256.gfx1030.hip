#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HFQ4-G256 batched 4-way fused QKV+Z+Beta+Alpha GEMM for gfx1030 (RDNA2).
// Sister of gemm_qkvza_hfq4g256_wmma.hip; 136 B/group stride, 4-bit nibble
// unpack. See gemm_gate_up_hfq4g256.gfx1030.hip for template comments.
// fdot2 inner product for 2× FP16 FMA throughput on gfx1030+.
// Double-buffered LDS pipelines dequant of K-tile N+1 with the dot
// product of K-tile N (1 __syncthreads per iter).

typedef _Float16 __attribute__((ext_vector_type(2))) half2_t;

// Helper: dequantize one HFQ4 K-tile (16 FP16 weights from 8 bytes) into
// lds_buf[my_lds_row][0..16]. Called by lanes 0..15 only. `tile_ptr`
// points at the 8 packed bytes for this K-tile (i.e. gp + 8 + kt * 8).
__device__ static inline void dq_hfq4_ktile(
    _Float16 (&lds_buf)[16][16],
    int my_lds_row,
    const char* __restrict__ tile_ptr,
    _Float16 sc_h,
    _Float16 zp_h
) {
    unsigned int pk0 = *(const unsigned int*)(tile_ptr);
    unsigned int pk1 = *(const unsigned int*)(tile_ptr + 4);
    #define DQ(i, pk, sh) lds_buf[my_lds_row][i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
    DQ(0,  pk0,  0); DQ(1,  pk0,  4); DQ(2,  pk0,  8); DQ(3,  pk0, 12);
    DQ(4,  pk0, 16); DQ(5,  pk0, 20); DQ(6,  pk0, 24); DQ(7,  pk0, 28);
    DQ(8,  pk1,  0); DQ(9,  pk1,  4); DQ(10, pk1,  8); DQ(11, pk1, 12);
    DQ(12, pk1, 16); DQ(13, pk1, 20); DQ(14, pk1, 24); DQ(15, pk1, 28);
    #undef DQ
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_qkvza_hfq4g256_gfx1030(
    const char*    __restrict__ A_qkv,
    const char*    __restrict__ A_z,
    const char*    __restrict__ A_beta,
    const char*    __restrict__ A_alpha,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y_qkv,
    float*         __restrict__ Y_z,
    float*         __restrict__ Y_beta,
    float*         __restrict__ Y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K, int N
) {
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    const int my_lds_row = tid & 15;
    int my_a_row_global = row_start + my_lds_row;
    if (my_a_row_global >= total_m) my_a_row_global = total_m - 1;

    const char* A;
    int local_row;
    if (my_a_row_global < qkv_m) {
        A = A_qkv;   local_row = my_a_row_global;
    } else if (my_a_row_global < qkv_m + z_m) {
        A = A_z;     local_row = my_a_row_global - qkv_m;
    } else if (my_a_row_global < qkv_m + z_m + beta_m) {
        A = A_beta;  local_row = my_a_row_global - qkv_m - z_m;
    } else {
        A = A_alpha; local_row = my_a_row_global - qkv_m - z_m - beta_m;
    }
    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_base = A + local_row * row_stride;

    const int my_col_in_tile = tid & 15;
    const int my_col_global = batch_start + my_col_in_tile;
    const int safe_col = (my_col_global < N) ? my_col_global : 0;

    // Double-buffered LDS: 2× 16×16 FP16 = 1024 bytes.
    __shared__ _Float16 lds_a[2][16][16];
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        _Float16 sc_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp));
        _Float16 zp_h = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4));

        const _Float16* x_base = X + (long long)safe_col * K + g * 256;
        const char* dp = gp + 8;

        // Prefetch K-tile 0 into lds_a[0] before entering the pipelined loop.
        if ((tid >> 4) == 0) {
            dq_hfq4_ktile(lds_a[0], my_lds_row, dp, sc_h, zp_h);
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
                x_pairs[p] = *(const half2_t*)(x_base + k_off + p * 2);
            }

            // Prefetch K-tile kt+1 into lds_a[next_buf] (skip on last K-tile).
            // Runs concurrently with the dot product on lds_a[cur_buf].
            if (next_kt < 16 && (tid >> 4) == 0) {
                dq_hfq4_ktile(lds_a[next_buf], my_lds_row, dp + next_kt * 8, sc_h, zp_h);
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
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < total_m) {
                float* Y;
                int proj_row;
                int out_stride;
                if (out_row < qkv_m) {
                    Y = Y_qkv;   proj_row = out_row;                                out_stride = qkv_m;
                } else if (out_row < qkv_m + z_m) {
                    Y = Y_z;     proj_row = out_row - qkv_m;                        out_stride = z_m;
                } else if (out_row < qkv_m + z_m + beta_m) {
                    Y = Y_beta;  proj_row = out_row - qkv_m - z_m;                  out_stride = beta_m;
                } else {
                    Y = Y_alpha; proj_row = out_row - qkv_m - z_m - beta_m;         out_stride = alpha_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
