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

#ifdef HRU_DUMP_ACC
// Test-only: pre-BF16 gate accumulators [4, N, hidden] F32, for exactness probes.
__device__ float* hru_dump_ptr;
#endif

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
#ifdef HRU_DUMP_ACC
            for (int br = 0; br < 4; br++)
                hru_dump_ptr[((size_t)br * N + token) * hidden + j] = acc[br][i];
#endif
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

// Operand-swapped sibling (opt-in, HIPFIRE_QWEN4_HC_UP_TILE).  Same launch
// geometry as hyper_read_up_wmma_bf16 (grid [hidden / 16, ceil(N / 512)], block
// 256, dynamic LDS 64 * (K + 8) * 2, hidden % 16 == 0, K % 16 == 0, K <= 504)
// and the same per-output math: every (token, j, branch) accumulator runs
// k = 0..K step 16 in ascending order from +0.0 through the same BF16 WMMA, and
// the epilogue is the same expression in the same branch order.  Only the
// placement changes: W rows are the WMMA A operand and X tokens the B operand,
// so C[row = j][col = token]; a lane then holds one token and j = 2 * i + half
// (i = accumulator register, half = lane >> 4).  One pair of 32-bit lane
// exchanges per accumulator (pure data movement) gives each lane eight
// *contiguous* j of its token, j = j0 + 8 * half + m, so the normalized read is
// four 16-byte loads and the mixed store two 16-byte stores instead of 32
// two-byte loads and eight scattered 4-byte stores per lane.  WMMA derives each
// C element from the same 16 bf16 products in the same order whichever operand
// carries the row (verified bytewise against the baseline kernel).
extern "C" __global__ __launch_bounds__(256) void hyper_read_up_wmma_bf16_swap(
    const unsigned short* __restrict__ W,
    const unsigned short* __restrict__ X,
    const _Float16* __restrict__ normalized,
    float* __restrict__ mixed,
    int hidden,
    int K,
    int N) {
    constexpr int TOKENS = 512;
    extern __shared__ unsigned short hru_ws[];
    const int ldw = K + 8;
    const int j0 = blockIdx.x * 16;
    const int tid = threadIdx.x;
    const int vpr = K / 8;
    for (int v = tid; v < 64 * vpr; v += 256) {
        const int r = v / vpr, c = v - r * vpr;
        const int row = (r >> 4) * hidden + j0 + (r & 15);
        *(uint4*)(hru_ws + r * ldw + c * 8) = *(const uint4*)(W + (size_t)row * K + c * 8);
    }
    __syncthreads();
    const int wave = tid >> 5, lane = tid & 31, l16 = lane & 15, half = lane >> 4;
    const int wide = 4 * hidden;
    const int jb = j0 + 8 * half;
    const int block_end = min(N, (int)blockIdx.y * TOKENS + TOKENS);
    for (int t0 = blockIdx.y * TOKENS + wave * 16; t0 < block_end; t0 += 128) {
        const unsigned short* xr = X + (size_t)min(t0 + l16, N - 1) * K;
        hru_f8_t acc[4];
#pragma unroll
        for (int br = 0; br < 4; br++) acc[br] = hru_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 2
        for (int k = 0; k < K; k += 16) {
            const uint4 u0 = *(const uint4*)(xr + k), u1 = *(const uint4*)(xr + k + 8);
            const hru_u8_t xp = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
            const hru_bf16x16_t b = __builtin_bit_cast(hru_bf16x16_t, xp);
#pragma unroll
            for (int br = 0; br < 4; br++) {
                const hru_bf16x16_t a = *(const hru_bf16x16_t*)(hru_ws + (br * 16 + l16) * ldw + k);
                acc[br] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[br]);
            }
        }
        // acc[br][i] holds j = j0 + 2 * i + half; exchange so lane half h holds
        // j = j0 + 8 * h + m in acc[br][m].
#pragma unroll
        for (int br = 0; br < 4; br++) {
            float q[8];
#pragma unroll
            for (int m = 0; m < 4; m++) {
                const float send = half ? acc[br][m] : acc[br][4 + m];
                const float recv = __shfl_xor(send, 16);
                q[2 * m] = half ? recv : acc[br][m];
                q[2 * m + 1] = half ? acc[br][4 + m] : recv;
            }
#pragma unroll
            for (int m = 0; m < 8; m++) acc[br][m] = q[m];
        }
        const int token = t0 + l16;
        if (token >= N) continue;
#ifdef HRU_DUMP_ACC
        for (int br = 0; br < 4; br++)
            for (int i = 0; i < 8; i++) hru_dump_ptr[((size_t)br * N + token) * hidden + jb + i] = acc[br][i];
#endif
        uint4 nvv[4];
#pragma unroll
        for (int br = 0; br < 4; br++)
            nvv[br] = *(const uint4*)(normalized + (size_t)token * wide + br * hidden + jb);
        float out[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            // hyper_read_projected_f32's mix, branch order from +0.0.
            float value = 0.0f;
#pragma unroll
            for (int br = 0; br < 4; br++) {
                const float gate_input = hru_bf16r(acc[br][i]);
                const float gate = hru_bf16r(1.0f / (1.0f + expf(-gate_input)));
                const unsigned int w32 = ((const unsigned int*)&nvv[br])[i >> 1];
                const unsigned short hb = (i & 1) ? (w32 >> 16) : (w32 & 0xffff);
                const float normed = (float)__builtin_bit_cast(_Float16, hb);
                value += hru_bf16r(gate * normed);
            }
            out[i] = hru_bf16r(value / 4.0f);
        }
        float4* mp = (float4*)(mixed + (size_t)token * hidden + jb);
        mp[0] = float4{out[0], out[1], out[2], out[3]};
        mp[1] = float4{out[4], out[5], out[6], out[7]};
    }
}
