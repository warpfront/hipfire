#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// gfx1201 (RDNA4) sister of hyper_read_up_wmma.gfx1151.hip: the HC read tail,
// the BF16 [4 * hidden, K] up projection of `low` ([N, K] BF16 bits,
// hc_activation_fused_f32's packed output) with hyper_read_projected_f32's
// branch mix fused, reading the normalized rows from hyper_norm_f16's F16
// copy.  Same block decomposition, same LDS staging and the same epilogue
// expression; only the WMMA fragment mapping changes:
//   - half-width (8-element) operands and the `_w32_gfx12` builtin: lane l
//     holds row (l & 15), K elements [8 * (l >> 4), +8) of each 16-element
//     substep, so each lane loads one uint4 of X and one of W per substep;
//   - the accumulator is contiguous per lane-half:
//     acc[i] = C[8 * (l >> 4) + i][l & 15], i.e. token t0 + 8 * (l >> 4) + i.
// Both GEMM operands are exact in BF16, so the logits differ from
// hyper_read_up_fused_f32 only by F32 summation order.
//
// Grid [hidden / 16, ceil(N / 512)], block 256, dynamic LDS 64 * (K + 8) * 2
// bytes.  hidden % 16 == 0, K % 16 == 0, K <= 504.

typedef short __attribute__((ext_vector_type(8))) hru12_bf16x8_t;
typedef float __attribute__((ext_vector_type(8))) hru12_f8_t;

// tensor_ops.hip's bf16_to_f32(bf16_round(v)).
static __device__ __forceinline__ float hru12_bf16r(float v) {
    return static_cast<float>(static_cast<hip_bfloat16>(v));
}

#ifdef HRU12_DUMP_ACC
// Test-only: pre-BF16 gate accumulators [4, N, hidden] F32, for exactness probes.
__device__ float* hru12_dump_ptr;
#endif

extern "C" __global__ __launch_bounds__(256) void hyper_read_up_wmma_bf16_gfx1201(
    const unsigned short* __restrict__ W,
    const unsigned short* __restrict__ X,
    const _Float16* __restrict__ normalized,
    float* __restrict__ mixed,
    int hidden,
    int K,
    int N) {
    constexpr int TOKENS = 512;
    extern __shared__ unsigned short hru12_w[];
    const int ldw = K + 8;
    const int j0 = blockIdx.x * 16;
    const int tid = threadIdx.x;
    const int vpr = K / 8;
    for (int v = tid; v < 64 * vpr; v += 256) {
        const int r = v / vpr, c = v - r * vpr;
        const int row = (r >> 4) * hidden + j0 + (r & 15);
        *(uint4*)(hru12_w + r * ldw + c * 8) = *(const uint4*)(W + (size_t)row * K + c * 8);
    }
    __syncthreads();
    const int wave = tid >> 5, lane = tid & 31, l16 = lane & 15, kh = lane >> 4;
    const int wide = 4 * hidden;
    const int j = j0 + l16;
    const int block_end = min(N, (int)blockIdx.y * TOKENS + TOKENS);
    for (int t0 = blockIdx.y * TOKENS + wave * 16; t0 < block_end; t0 += 128) {
        // The epilogue's normalized values, fetched before the K loop so the
        // loads overlap the WMMA work.
        _Float16 nv[4][8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int token = min(t0 + 8 * kh + i, N - 1);
#pragma unroll
            for (int br = 0; br < 4; br++)
                nv[br][i] = normalized[(size_t)token * wide + br * hidden + j];
        }
        const unsigned short* xr = X + (size_t)min(t0 + l16, N - 1) * K + 8 * kh;
        hru12_f8_t acc[4];
#pragma unroll
        for (int br = 0; br < 4; br++) acc[br] = hru12_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 4
        for (int k = 0; k < K; k += 16) {
            const hru12_bf16x8_t a = __builtin_bit_cast(hru12_bf16x8_t, *(const uint4*)(xr + k));
#pragma unroll
            for (int br = 0; br < 4; br++) {
                const hru12_bf16x8_t b =
                    *(const hru12_bf16x8_t*)(hru12_w + (br * 16 + l16) * ldw + k + 8 * kh);
                acc[br] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, acc[br]);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int token = t0 + 8 * kh + i;
            if (token >= N) continue;
#ifdef HRU12_DUMP_ACC
            for (int br = 0; br < 4; br++)
                hru12_dump_ptr[((size_t)br * N + token) * hidden + j] = acc[br][i];
#endif
            // hyper_read_projected_f32's mix, branch order from +0.0.
            float value = 0.0f;
#pragma unroll
            for (int br = 0; br < 4; br++) {
                const float gate_input = hru12_bf16r(acc[br][i]);
                const float gate = hru12_bf16r(1.0f / (1.0f + expf(-gate_input)));
                const float normed = (float)nv[br][i];
                value += hru12_bf16r(gate * normed);
            }
            mixed[(size_t)token * hidden + j] = hru12_bf16r(value / 4.0f);
        }
    }
}

// Retiled sibling (opt-in, HIPFIRE_QWEN4_HC_UP_TILE).  Per-output math is the
// same as hyper_read_up_wmma_bf16_gfx1201: every (token, j, branch) accumulator
// runs k = 0..K step 16 in ascending order from +0.0 through the same BF16
// 16x16x16 WMMA, and the epilogue is the same expression in the same branch
// order.  Only the placement changes:
//   - operands are swapped (W rows are the WMMA A operand, X tokens the B
//     operand), so C[row = j][col = token] and each lane owns 8 *contiguous* j
//     of one token: acc[br][i] = (j = j0 + 8 * (lane >> 4) + i, token = t0 +
//     (lane & 15)).  The normalized read becomes four 16-byte loads and the
//     mixed store two 16-byte stores per lane instead of 32 two-byte loads and
//     8 x 4-byte scattered stores.  WMMA computes each C element from the same
//     16 bf16 products in the same hardware order whichever operand carries
//     the row, so the accumulators are unchanged (verified bytewise);
//   - the block is 128 tokens (one 16-token tile per wave) and W is staged
//     through LDS in K chunks of 64: static LDS 64 * 72 * 2 = 9216 bytes
//     (7 blocks per CU) instead of 64 * (K + 8) * 2 = 41984 (1 block per CU),
//     so staging latency, the expf/divide epilogue and the memory traffic of
//     one block overlap the WMMA chains of the others.
// Grid [hidden / 16, ceil(N / 128)], block 256, NO dynamic LDS (the launch
// must pass 0: static LDS is declared here).  hidden % 16 == 0, K % 64 == 0.
extern "C" __global__ __launch_bounds__(256) void hyper_read_up_wmma_bf16_gfx1201_t128(
    const unsigned short* __restrict__ W,
    const unsigned short* __restrict__ X,
    const _Float16* __restrict__ normalized,
    float* __restrict__ mixed,
    int hidden,
    int K,
    int N) {
    constexpr int KC = 64;
    constexpr int LDW = KC + 8;
    __shared__ unsigned short hru12_wc[64 * LDW];
    const int j0 = blockIdx.x * 16;
    const int tid = threadIdx.x;
    const int wave = tid >> 5, lane = tid & 31, l16 = lane & 15, kh = lane >> 4;
    const int wide = 4 * hidden;
    const int t0 = blockIdx.y * 128 + wave * 16;
    const unsigned short* xr = X + (size_t)min(t0 + l16, N - 1) * K + 8 * kh;
    hru12_f8_t acc[4];
#pragma unroll
    for (int br = 0; br < 4; br++) acc[br] = hru12_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
    // Every wave runs every chunk (tokens past N are clamped) so the block
    // barriers stay uniform.
    for (int c = 0; c < K; c += KC) {
        __syncthreads();
#pragma unroll
        for (int v = tid; v < 64 * (KC / 8); v += 256) {
            const int r = v / (KC / 8), cc = v % (KC / 8);
            const int row = (r >> 4) * hidden + j0 + (r & 15);
            *(uint4*)(hru12_wc + r * LDW + cc * 8) = *(const uint4*)(W + (size_t)row * K + c + cc * 8);
        }
        __syncthreads();
#pragma unroll 2
        for (int k = 0; k < KC; k += 16) {
            const hru12_bf16x8_t b = __builtin_bit_cast(hru12_bf16x8_t, *(const uint4*)(xr + c + k));
#pragma unroll
            for (int br = 0; br < 4; br++) {
                const hru12_bf16x8_t a =
                    *(const hru12_bf16x8_t*)(hru12_wc + (br * 16 + l16) * LDW + k + 8 * kh);
                acc[br] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, acc[br]);
            }
        }
    }
    const int token = t0 + l16;
    if (token >= N) return;
    const int jb = j0 + 8 * kh;
#ifdef HRU12_DUMP_ACC
    for (int br = 0; br < 4; br++)
        for (int i = 0; i < 8; i++) hru12_dump_ptr[((size_t)br * N + token) * hidden + jb + i] = acc[br][i];
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
            const float gate_input = hru12_bf16r(acc[br][i]);
            const float gate = hru12_bf16r(1.0f / (1.0f + expf(-gate_input)));
            const unsigned int w32 = ((const unsigned int*)&nvv[br])[i >> 1];
            const unsigned short hb = (i & 1) ? (w32 >> 16) : (w32 & 0xffff);
            const float normed = (float)__builtin_bit_cast(_Float16, hb);
            value += hru12_bf16r(gate * normed);
        }
        out[i] = hru12_bf16r(value / 4.0f);
    }
    float4* mp = (float4*)(mixed + (size_t)token * hidden + jb);
    mp[0] = float4{out[0], out[1], out[2], out[3]};
    mp[1] = float4{out[4], out[5], out[6], out[7]};
}
