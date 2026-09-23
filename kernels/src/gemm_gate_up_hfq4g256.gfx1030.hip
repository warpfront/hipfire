#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HFQ4-G256 batched gate+up GEMM for gfx1030 (RDNA2, Navi 21+).
// Sister of gemm_gate_up_hfq4g256_wmma.hip; same contract, same per-lane
// output layout (acc[j] = C[2*j + (tid>>4)][tid & 15]) — without WMMA.
//
// MQ4 differs from MQ3 only in per-group stride (136 vs 104 bytes) and
// inner-tile unpack (4-bit nibble vs 3-bit cross-byte). Same LDS-shared
// dequantized A strategy. Inner product uses __builtin_amdgcn_fdot2
// (v_dot2_f32_f16 on gfx1030+) for 2× FP16 FMA throughput.
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
extern "C" __global__ void gemm_gate_up_hfq4g256_gfx1030(
    const char*    __restrict__ A_gate,
    const char*    __restrict__ A_up,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y_gate,
    float*         __restrict__ Y_up,
    int gate_m, int up_m,
    int K, int N
) {
    const int total_m = gate_m + up_m;
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
    if (my_a_row_global < gate_m) {
        A = A_gate;  local_row = my_a_row_global;
    } else {
        A = A_up;    local_row = my_a_row_global - gate_m;
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
                if (out_row < gate_m) {
                    Y = Y_gate;  proj_row = out_row;            out_stride = gate_m;
                } else {
                    Y = Y_up;    proj_row = out_row - gate_m;   out_stride = up_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
