#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// gfx1151 (RDNA3.5, gfx11 WMMA wave32) port of hyper_read_up_wmma_bf16_gfx1201_v2: the HC read
// tail (BF16 [4 * hidden, K] up projection of `low` with hyper_read_projected_f32's branch mix
// fused), bytewise `hyper_read_up_wmma_bf16_swap`'s output (kernels/src/hyper_read_up_wmma.gfx1151.hip).
// K == 320 (20 WMMA steps) only; the host admits nothing else.
//
// Same WMMA sequence as `_swap`: W rows are the A operand, X tokens the B operand (16 contiguous
// K elements per lane, lanes 16..31 replicate lanes 0..15), every (token, j, branch) accumulator
// runs k = 0..319 step 16 ascending from +0.0 through the same BF16 16x16x16 WMMA; the
// accumulator holds j = j0 + 2 * i + half for token t0 + (lane & 15), and the same pair of
// __shfl_xor(16) exchanges gives each lane 8 contiguous j = j0 + 8 * half + m.  What changes:
//   - the block is persistent over a token range: W for its 16 columns x 4 branches is staged into
//     LDS once, and the HRU3_WAVES waves walk the block's 16-token tiles with no barrier;
//   - each wave software-pipelines its tile: X fragments HRU3_XB K steps ahead, W fragments one
//     step ahead (tuned on Halo: depth 5, 8 waves per block, two blocks per WGP at 58,368 B LDS);
//   - the gate sigmoid(x) -> BF16 depends only on the BF16-rounded logit, so it is a lookup in a
//     2 x 2048-entry table (16 KiB LDS) built at block start with the baseline's own expression
//     (BF16 magnitudes 0x3b80..0x42b2; outside that range the gate is the 0.5 / saturated plateau
//     and the index clamps);
//   - the epilogue arithmetic is branch-free (finite-only RNE to BF16).  A lane that sees a NaN
//     logit or a non-finite normalized value recomputes its 8 outputs with the baseline
//     expression, with the compiled v_fma_mix_f32 / v_add_f32 operand order of `_swap` pinned
//     (verified in `_swap`'s gfx1151 ISA: fma_mix(normalized word, gate, -0), add(value, r)).
// Grid [hidden / 16, G], G <= ceil(N / 16); block HRU3_WAVES * 32 threads; dynamic LDS
// 64 * (K + 8) * 2 + 16384 bytes.  hidden % 16 == 0, K == 320.

#ifndef HRU3_XB
#define HRU3_XB 5
#endif
#ifndef HRU3_WAVES
#define HRU3_WAVES 8
#endif
#define HRU3_K 320
#define HRU3_KSTEPS (HRU3_K / 16)
#define HRU3_LUT_LO 0x3b80u
#define HRU3_LUT_HI 0x42b2u
#define HRU3_LUT_SPAN 2048u

typedef short __attribute__((ext_vector_type(16))) hru3_bf16x16_t;
typedef float __attribute__((ext_vector_type(8))) hru3_f8_t;
typedef unsigned int __attribute__((ext_vector_type(8))) hru3_u8_t;
typedef _Float16 __attribute__((ext_vector_type(2))) hru3_h2_t;

// tensor_ops.hip's bf16_to_f32(bf16_round(v)).
static __device__ __forceinline__ float hru3_bf16r(float v) {
    return static_cast<float>(static_cast<hip_bfloat16>(v));
}

static __device__ __forceinline__ float hru3_bf16r_fin(float v) {
    unsigned u = __float_as_uint(v);
    u = u + 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

template <int HI>
static __device__ __forceinline__ float hru3_fma_mix(unsigned nvword, float gate) {
    float r;
    if (HI)
        asm("v_fma_mix_f32 %0, %1, %2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]"
            : "=v"(r) : "v"(nvword), "v"(gate));
    else
        asm("v_fma_mix_f32 %0, %1, %2, neg(0) op_sel_hi:[1,0,0]" : "=v"(r) : "v"(nvword), "v"(gate));
    return r;
}

static __device__ __forceinline__ float hru3_add(float a, float b) {
    float r;
    asm("v_add_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}

// The `_swap` epilogue for one lane's 8 outputs.
static __device__ __forceinline__ void hru3_baseline_epilogue(
    const hru3_f8_t* acc, const uint4* nvv, float* out) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
        float value = 0.0f;
#pragma unroll
        for (int br = 0; br < 4; br++) {
            const float gate_input = hru3_bf16r(acc[br][i]);
            const float gate = hru3_bf16r(1.0f / (1.0f + expf(-gate_input)));
            const unsigned int w32 = ((const unsigned int*)&nvv[br])[i >> 1];
            const float p = (i & 1) ? hru3_fma_mix<1>(w32, gate) : hru3_fma_mix<0>(w32, gate);
            value = hru3_add(value, hru3_bf16r(p));
        }
        out[i] = hru3_bf16r(value / 4.0f);
    }
}

static __device__ __forceinline__ hru3_bf16x16_t hru3_pack(uint4 a, uint4 b) {
    const hru3_u8_t xp = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(hru3_bf16x16_t, xp);
}

extern "C" __global__ __launch_bounds__(HRU3_WAVES * 32) void hyper_read_up_wmma_bf16_gfx1151_v2(
    const unsigned short* __restrict__ W,
    const unsigned short* __restrict__ X,
    const _Float16* __restrict__ normalized,
    float* __restrict__ mixed,
    int hidden,
    int K,
    int N) {
    constexpr int LDW = HRU3_K + 8;
    extern __shared__ unsigned short hru3_w[];
    unsigned* const tab = (unsigned*)((char*)hru3_w + 64 * LDW * 2);
    const int j0 = blockIdx.x * 16;
    const int tid = threadIdx.x;
    for (int v = tid; v < 64 * (HRU3_K / 8); v += HRU3_WAVES * 32) {
        const int r = v / (HRU3_K / 8), c = v % (HRU3_K / 8);
        const int row = (r >> 4) * hidden + j0 + (r & 15);
        *(uint4*)(hru3_w + r * LDW + c * 8) = *(const uint4*)(W + (size_t)row * HRU3_K + c * 8);
    }
    for (int idx = tid; idx < 2 * (int)HRU3_LUT_SPAN; idx += HRU3_WAVES * 32) {
        const unsigned sg = (unsigned)idx / HRU3_LUT_SPAN;
        const unsigned e = min((unsigned)idx % HRU3_LUT_SPAN, HRU3_LUT_HI - HRU3_LUT_LO);
        const float x = __uint_as_float(((sg << 15) | (HRU3_LUT_LO + e)) << 16);
        tab[idx] = __float_as_uint(hru3_bf16r(1.0f / (1.0f + expf(-x))));
    }
    __syncthreads();
    const int wave = tid >> 5, lane = tid & 31, l16 = lane & 15, half = lane >> 4;
    const int wide = 4 * hidden;
    const int ntiles = (N + 15) >> 4;
    const int tpg = (ntiles + (int)gridDim.y - 1) / (int)gridDim.y;
    const int tbeg = (int)blockIdx.y * tpg;
    const int tend = min(ntiles, tbeg + tpg);
    const int jb = j0 + 8 * half;
    const unsigned short* const wb = hru3_w + l16 * LDW;
    for (int tile = tbeg + wave; tile < tend; tile += HRU3_WAVES) {
        const int token = tile * 16 + l16;
        const int tokc = min(token, N - 1);
        uint4 nvv[4];
#pragma unroll
        for (int br = 0; br < 4; br++)
            nvv[br] = *(const uint4*)(normalized + (size_t)tokc * wide + br * hidden + jb);
        const unsigned short* const xr = X + (size_t)tokc * HRU3_K;
        hru3_f8_t acc[4];
#pragma unroll
        for (int br = 0; br < 4; br++) acc[br] = hru3_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
        uint4 xcur[HRU3_XB][2];
#pragma unroll
        for (int s = 0; s < HRU3_XB; s++) {
            xcur[s][0] = *(const uint4*)(xr + 16 * s);
            xcur[s][1] = *(const uint4*)(xr + 16 * s + 8);
        }
        hru3_bf16x16_t acur[4];
#pragma unroll
        for (int br = 0; br < 4; br++) acur[br] = *(const hru3_bf16x16_t*)(wb + br * 16 * LDW);
#pragma unroll
        for (int g = 0; g < HRU3_KSTEPS / HRU3_XB; g++) {
            __builtin_amdgcn_sched_barrier(0);
            uint4 xnext[HRU3_XB][2];
            if (g + 1 < HRU3_KSTEPS / HRU3_XB) {
#pragma unroll
                for (int s = 0; s < HRU3_XB; s++) {
                    xnext[s][0] = *(const uint4*)(xr + 16 * ((g + 1) * HRU3_XB + s));
                    xnext[s][1] = *(const uint4*)(xr + 16 * ((g + 1) * HRU3_XB + s) + 8);
                }
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int s = 0; s < HRU3_XB; s++) {
                const int ks = g * HRU3_XB + s;
                hru3_bf16x16_t anext[4];
                if (ks + 1 < HRU3_KSTEPS) {
#pragma unroll
                    for (int br = 0; br < 4; br++)
                        anext[br] = *(const hru3_bf16x16_t*)(wb + br * 16 * LDW + 16 * (ks + 1));
                }
                __builtin_amdgcn_sched_barrier(0);
                const hru3_bf16x16_t b = hru3_pack(xcur[s][0], xcur[s][1]);
#pragma unroll
                for (int br = 0; br < 4; br++)
                    acc[br] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(acur[br], b, acc[br]);
                if (ks + 1 < HRU3_KSTEPS) {
#pragma unroll
                    for (int br = 0; br < 4; br++) acur[br] = anext[br];
                }
            }
            if (g + 1 < HRU3_KSTEPS / HRU3_XB) {
#pragma unroll
                for (int s = 0; s < HRU3_XB; s++) {
                    xcur[s][0] = xnext[s][0];
                    xcur[s][1] = xnext[s][1];
                }
            }
        }
        // acc[br][i] holds j = j0 + 2 * i + half; exchange so lane half h holds j = j0 + 8 * h + m
        // in acc[br][m] (pure data movement, as `_swap`).
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
        float out[8];
        bool bad = false;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            float value = 0.0f;
#pragma unroll
            for (int br = 0; br < 4; br++) {
                const float a = acc[br][i];
                bad |= (a != a);
                const unsigned u = __float_as_uint(a);
                const unsigned r = u + 0x7fffu + ((u >> 16) & 1u);
                const unsigned mag = min(max((r >> 16) & 0x7fffu, HRU3_LUT_LO), HRU3_LUT_HI);
                const float gate = __uint_as_float(tab[(r >> 31) * HRU3_LUT_SPAN + (mag - HRU3_LUT_LO)]);
                const unsigned int w32 = ((const unsigned int*)&nvv[br])[i >> 1];
                const unsigned short hb = (i & 1) ? (w32 >> 16) : (w32 & 0xffff);
                const float normed = (float)__builtin_bit_cast(_Float16, hb);
                value += hru3_bf16r_fin(gate * normed);
            }
            out[i] = hru3_bf16r_fin(value * 0.25f);
        }
        {
            // inf / NaN anywhere in the 16 normalized words survives a packed F16 sum.
            hru3_h2_t s = __builtin_bit_cast(hru3_h2_t, ((const unsigned int*)&nvv[0])[0]);
#pragma unroll
            for (int br = 0; br < 4; br++)
#pragma unroll
                for (int w = 0; w < 4; w++)
                    if (br | w) s += __builtin_bit_cast(hru3_h2_t, ((const unsigned int*)&nvv[br])[w]);
            bad |= !((float)s[0] >= -1e30f) | !((float)s[1] >= -1e30f) | ((float)s[0] > 1e30f) |
                   ((float)s[1] > 1e30f);
        }
        if (bad) hru3_baseline_epilogue(acc, nvv, out);
        if (token < N) {
            float4* mp = (float4*)(mixed + (size_t)token * hidden + jb);
            mp[0] = float4{out[0], out[1], out[2], out[3]};
            mp[1] = float4{out[4], out[5], out[6], out[7]};
        }
    }
}
