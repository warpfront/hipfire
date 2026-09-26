#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// HC read tail on gfx11 WMMA: the BF16 [4 * hidden, K] up projection of
// `low` ([N, K] BF16 bits, hc_activation_fused_f32's packed output) with
// hyper_read_projected_f32's
// branch mix fused, reading the normalized rows from hyper_norm_f16's F16
// copy (the BF16-rounded values, exact in F16 above its 6e-5 normal floor).
// Both GEMM operands are exact in BF16, so the logits differ from
// hyper_read_up_fused_f32 only by F32 summation order (BF16 WMMA
// accumulates in F32); the epilogue is that kernel's expression.
//
// Block: hidden columns j0..j0+15 of all four branches (64 W rows, staged
// in LDS) x 512 tokens; wave w takes the 16-token tiles w, w + 8, ...  X is the
// WMMA A operand (tokens = rows), W the B operand (columns = j), so each
// accumulator row is one token and the epilogue's normalized / mixed
// accesses are contiguous in j.  The four branch accumulators of a lane hold
// the same (token, j), so the branch mix is lane-local.
// Grid [hidden / 16, ceil(N / 512)], block 256, dynamic LDS 64 * (K + 8) * 2
// bytes.  hidden % 16 == 0, K % 16 == 0, K <= 504.

typedef short __attribute__((ext_vector_type(16))) hru_bf16x16_t;
typedef float __attribute__((ext_vector_type(8))) hru_f8_t;
typedef unsigned int __attribute__((ext_vector_type(8))) hru_u8_t;

// tensor_ops.hip's bf16_to_f32(bf16_round(v)).
static __device__ __forceinline__ float hru_bf16r(float v) {
    return static_cast<float>(static_cast<hip_bfloat16>(v));
}

extern "C" __global__ __launch_bounds__(256) void hyper_read_up_wmma_bf16(
    const unsigned short* __restrict__ W,
    const unsigned short* __restrict__ X,
    const _Float16* __restrict__ normalized,
    float* __restrict__ mixed,
    int hidden,
    int K,
    int N) {
    constexpr int TOKENS = 512;  // 4% under 256: half the W staging per token
    extern __shared__ unsigned short hru_w[];
    const int ldw = K + 8;
    const int j0 = blockIdx.x * 16;
    const int tid = threadIdx.x;
    const int vpr = K / 8;
    for (int v = tid; v < 64 * vpr; v += 256) {
        const int r = v / vpr, c = v - r * vpr;
        const int row = (r >> 4) * hidden + j0 + (r & 15);
        *(uint4*)(hru_w + r * ldw + c * 8) = *(const uint4*)(W + (size_t)row * K + c * 8);
    }
    __syncthreads();
    const int wave = tid >> 5, lane = tid & 31, l16 = lane & 15;
    const int wide = 4 * hidden;
    const int j = j0 + l16;
    const int block_end = min(N, (int)blockIdx.y * TOKENS + TOKENS);
    for (int t0 = blockIdx.y * TOKENS + wave * 16; t0 < block_end; t0 += 128) {
        // The epilogue's normalized values, fetched before the K loop so the
        // loads overlap the WMMA work.
        _Float16 nv[4][8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int token = min(t0 + 2 * i + (lane >> 4), N - 1);
#pragma unroll
            for (int br = 0; br < 4; br++)
                nv[br][i] = normalized[(size_t)token * wide + br * hidden + j];
        }
        const unsigned short* xr = X + (size_t)min(t0 + l16, N - 1) * K;
        hru_f8_t acc[4];
#pragma unroll
        for (int br = 0; br < 4; br++) acc[br] = hru_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 4
        for (int k = 0; k < K; k += 16) {
            const uint4 u0 = *(const uint4*)(xr + k), u1 = *(const uint4*)(xr + k + 8);
            const hru_u8_t xp = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
            const hru_bf16x16_t a = __builtin_bit_cast(hru_bf16x16_t, xp);
#pragma unroll
            for (int br = 0; br < 4; br++) {
                const hru_bf16x16_t b = *(const hru_bf16x16_t*)(hru_w + (br * 16 + l16) * ldw + k);
                acc[br] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[br]);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int token = t0 + 2 * i + (lane >> 4);
            if (token >= N) continue;
            // hyper_read_projected_f32's mix, branch order from +0.0.
            float value = 0.0f;
#pragma unroll
            for (int br = 0; br < 4; br++) {
                const float gate_input = hru_bf16r(acc[br][i]);
                const float gate = hru_bf16r(1.0f / (1.0f + expf(-gate_input)));
                const float normed = (float)nv[br][i];
                value += hru_bf16r(gate * normed);
            }
            mixed[(size_t)token * hidden + j] = hru_bf16r(value / 4.0f);
        }
    }
}
