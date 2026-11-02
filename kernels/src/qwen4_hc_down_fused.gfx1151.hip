// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx1151 Qwen4 HC read down projection with the activation in the epilogue:
// ONE launch for what the default gfx1151 prefill chain runs as
//   gemm_wmma_lds_160_64_32_64_k64_p   (qwen4_gemm_f16_x_f16_wmma_lds256.hip, F32 Y)
//   hc_activation_fused_f32            (bf16 -> * scale -> bf16 -> SiLU -> bf16, tensor_ops.hip)
// Y[b, m] = sum_k A[m, k] * X[b, k] with M = 320, K = 10240 (= 4 * hidden), and the packed BF16
// activation lands in `out[b, m]`, the operand hyper_read_up_wmma_bf16_swap reads.  The F32 `low`
// the chain also leaves behind is dead in this route (the activation is called with `bf16_out`,
// which leaves it as the raw GEMM output, and the read tail only reads the packed BF16), so it is
// not written.
//
// BYTE-IDENTICAL to that chain, per output:
//   * ONE accumulator chain per output, no K split: started at +0.0f, fed the 16-wide K substeps
//     of the whole K = 10240 in ascending order through __builtin_amdgcn_wmma_f32_16x16x16_f16_w32
//     (A operand = the weight rows, B operand = the X rows, 16 contiguous K halves per lane,
//     lanes 16..31 mirroring 0..15, exactly the incumbent's fragments);
//   * the incumbent's epilogue adds `has_bias ? Bias[m] : 0.0f` = +0.0f (this turns -0.0 into
//     +0.0), which is repeated here;
//   * the activation is hc_activation_fused_f32's expression with `bf16_out`.
//
// Why it wins.  The incumbent's M tile is 160, so every 20 KB X row is read from DRAM twice per
// call (2 x 168 MB per 8192 rows), and its 64-token tile leaves 128 workgroups for 40 CUs (4
// rounds, 80 % full).  A workgroup here owns the whole M = 320 and 32 tokens: X is read once, the
// 6.5 MB weight shadow stays L2 / MALL resident, and 256 workgroups run in 6.4 -> 7 rounds.
//
// Geometry.  Five waves (160 threads), wave w owns rows [64 w, 64 w + 64) of M (4 WMMA A
// fragments) x all 32 tokens (2 B fragments), 64 accumulator VGPRs.  K is staged 64 per barrier
// pair through one LDS buffer (A 320 x 64 + X 32 x 64 halves, 45312 B), the incumbent's
// store / barrier / compute / barrier structure: the global loads of the next stage are issued
// right after the first barrier and retire into LDS after the second stage's WMMAs.  The LDS tile
// layout is one plane per 16-byte chunk column q (K elements [8 q, 8 q + 8) of the stage),
// [plane][row][8 halves], plane pitch 1 (mod 8) rows: a fragment read's eight-lane groups are 128
// contiguous bytes and a stage store's eight lanes are bank-disjoint (the incumbent's row-major
// [row][16] layout has 2-way bank conflicts on every fragment read).  The fragment reads run as a
// streaming pipeline (an A fragment two steps ahead of its WMMAs), X rows are loaded non-temporal.
//
// PRECONDITIONS (host-checked): M == 320, K == 10240, rows * K * 2 < 2^32, out non-null,
// A / X / out 16-byte aligned.  Rows past B clamp the staged row to B - 1 and are discarded.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#if !defined(__gfx1151__)
#error "qwen4_hc_down_fused.gfx1151.hip: gfx1151 only"
#endif

typedef _Float16 __attribute__((ext_vector_type(8)))  hg_half8_t;
typedef _Float16 __attribute__((ext_vector_type(16))) hg_half16_t;
typedef float    __attribute__((ext_vector_type(8)))  hg_float8_t;

constexpr int HG_M      = 320;                 // HC low rank: the whole M in one workgroup
constexpr int HG_K      = 10240;               // 4 x hidden (2560): the only K the route admits
constexpr int HG_NWAVE  = 5;                   // waves along M
constexpr int HG_WP     = 4;                   // A fragments per wave (64 rows)
constexpr int HG_WQ     = 2;                   // X fragments per wave (32 tokens)
constexpr int HG_BN     = 16 * HG_WQ;          // tokens per workgroup
constexpr int HG_NT     = 32 * HG_NWAVE;       // threads per workgroup
constexpr int HG_KS     = 64;                  // K elements staged per barrier pair
constexpr int HG_KT     = HG_KS / 16;          // WMMA k-substeps per stage
constexpr int HG_CPR    = HG_KS / 8;           // 16-byte chunks per row per stage
constexpr int HG_NSTAGE = HG_K / HG_KS;
constexpr int HG_A_PS   = HG_M + 1;            // rows per A plane (pitch 1 mod 8)
constexpr int HG_X_PS   = HG_BN + 1;
constexpr int HG_A_BUF  = HG_CPR * HG_A_PS * 8;   // halves
constexpr int HG_X_BUF  = HG_CPR * HG_X_PS * 8;
constexpr int HG_SLAB   = HG_M + 8;               // epilogue row pitch (halves)
constexpr int HG_LDS_H  = (HG_A_BUF + HG_X_BUF > HG_BN * HG_SLAB) ? HG_A_BUF + HG_X_BUF
                                                                    : HG_BN * HG_SLAB;
constexpr int HG_A_CH   = HG_M * HG_CPR;          // A chunks per stage
constexpr int HG_X_CH   = HG_BN * HG_CPR;
constexpr int HG_A_PASS = HG_A_CH / HG_NT;
constexpr int HG_X_PASS = (HG_X_CH + HG_NT - 1) / HG_NT;
constexpr int HG_D      = 2;                      // A fragment prefetch distance (steps)

static_assert(HG_NWAVE * HG_WP * 16 == HG_M, "M tile must be the whole 320");
static_assert(HG_A_CH % HG_NT == 0, "A chunks divide the block");
static_assert(HG_K % HG_KS == 0, "K staging");
static_assert(HG_LDS_H * 2 <= 65536, "LDS per workgroup");
static_assert(HG_D < HG_WP, "prefetch distance within one k-substep");

// hc_activation_fused_f32 (tensor_ops.hip) with `bf16_out`, same expression, same casts.
static __device__ __forceinline__ unsigned short hg_hc_act(float v, float scale) {
    float value = static_cast<float>(static_cast<hip_bfloat16>(v));
    value *= scale;
    value = static_cast<float>(static_cast<hip_bfloat16>(value));
    value = value / (1.0f + expf(-value));
    return static_cast<hip_bfloat16>(value).data;
}

// LDS-only workgroup barrier (the incumbent's WLDS_BARRIER on gfx1151): __syncthreads would also
// wait for the in-flight next-stage global loads.
#define HG_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
#define HG_WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)
#define HG_SCHED() __builtin_amdgcn_sched_barrier(0)

extern "C" __global__ __launch_bounds__(HG_NT, 1)
void qwen4_hc_down_fused_gfx1151(
    const _Float16* __restrict__ A,       // [320][10240] F16 weight shadow
    const _Float16* __restrict__ X,       // [B][10240] F16
    unsigned short* __restrict__ out,     // [B][320] packed BF16 bits (activated)
    int K, int B, float scale
) {
    (void)K;   // host guarantees K == HG_K; every offset below is a constant
    __shared__ __align__(32) _Float16 lds[HG_LDS_H];
    _Float16* const a_l = lds;
    _Float16* const x_l = lds + HG_A_BUF;

    const int n_block = blockIdx.x * HG_BN;
    if (n_block >= B) return;

    const int tid    = threadIdx.x;
    const int lane   = tid & 31;
    const int wave_m = tid >> 5;
    const int ml     = lane & 15;

    hg_float8_t acc[HG_WP * HG_WQ];
    #pragma unroll
    for (int i = 0; i < HG_WP * HG_WQ; i++) {
        acc[i] = hg_float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    hg_half8_t a_reg[HG_A_PASS];
    hg_half8_t x_reg[HG_X_PASS];

    // Stage `stage`'s chunks of A / X into registers.  Chunk c: row = c / 8, q = c % 8 (16 bytes
    // [16 q, +16) of the row's 128-byte K slice).  The X loads are branch free (vmcnt
    // accounting): surplus threads re-load the last valid chunk, the store side keeps its guard.
    auto gload = [&](int stage) {
        const int k0 = stage * HG_KS;
        #pragma unroll
        for (int p = 0; p < HG_A_PASS; p++) {
            const int c   = tid + p * HG_NT;
            const int row = c / HG_CPR;
            const int q   = c % HG_CPR;
            a_reg[p] = *(const hg_half8_t*)(A + (size_t)row * HG_K + k0 + q * 8);
        }
        #pragma unroll
        for (int p = 0; p < HG_X_PASS; p++) {
            const int c0  = tid + p * HG_NT;
            const int c   = (HG_X_CH % HG_NT != 0 && c0 >= HG_X_CH) ? HG_X_CH - 1 : c0;
            const int row = c / HG_CPR;
            const int q   = c % HG_CPR;
            int r = n_block + row;
            r = (r < B) ? r : (B - 1);
            x_reg[p] = __builtin_nontemporal_load(
                (const hg_half8_t*)(X + (size_t)r * HG_K + k0 + q * 8));
        }
    };
    auto lstore = [&]() {
        #pragma unroll
        for (int p = 0; p < HG_A_PASS; p++) {
            const int c   = tid + p * HG_NT;
            const int row = c / HG_CPR;
            const int q   = c % HG_CPR;
            *(hg_half8_t*)(a_l + (q * HG_A_PS + row) * 8) = a_reg[p];
        }
        #pragma unroll
        for (int p = 0; p < HG_X_PASS; p++) {
            const int c = tid + p * HG_NT;
            if (HG_X_CH % HG_NT != 0 && c >= HG_X_CH) continue;
            const int row = c / HG_CPR;
            const int q   = c % HG_CPR;
            *(hg_half8_t*)(x_l + (q * HG_X_PS + row) * 8) = x_reg[p];
        }
    };

    // One stage of WMMA out of LDS.  Step s = (kt, i) = (s / WP, s % WP): A fragment i of
    // k-substep kt against both X fragments.  The A fragment of step s + D (and the X fragments
    // of the next k-substep when it opens one) is read from LDS before step s's WMMAs, so the LDS
    // latency hides behind the tensor work and few fragments are live.  Each accumulator still
    // sees the k-substeps in ascending order, the incumbent's chain.
    auto compute = [&]() {
        constexpr int NSTEP = HG_KT * HG_WP;
        hg_half16_t aq[HG_D + 1];
        hg_half16_t bq[2][HG_WQ];
        auto ldA = [&](int kt, int i) {
            const int row = wave_m * HG_WP * 16 + i * 16 + ml;
            const _Float16* p = a_l + ((2 * kt) * HG_A_PS + row) * 8;
            const hg_half8_t lo = *(const hg_half8_t*)p;
            const hg_half8_t hi = *(const hg_half8_t*)(p + HG_A_PS * 8);
            return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7,
                                           8, 9, 10, 11, 12, 13, 14, 15);
        };
        auto ldB = [&](int kt, int j) {
            const int row = j * 16 + ml;
            const _Float16* p = x_l + ((2 * kt) * HG_X_PS + row) * 8;
            const hg_half8_t lo = *(const hg_half8_t*)p;
            const hg_half8_t hi = *(const hg_half8_t*)(p + HG_X_PS * 8);
            return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7,
                                           8, 9, 10, 11, 12, 13, 14, 15);
        };
        #pragma unroll
        for (int j = 0; j < HG_WQ; j++) bq[0][j] = ldB(0, j);
        #pragma unroll
        for (int d = 0; d < HG_D; d++) aq[d] = ldA(d / HG_WP, d % HG_WP);
        #pragma unroll
        for (int s = 0; s < NSTEP; s++) {
            const int kt = s / HG_WP;
            const int i  = s % HG_WP;
            const int s2 = s + HG_D;
            if (s2 < NSTEP) {
                const int kt2 = s2 / HG_WP;
                const int i2  = s2 % HG_WP;
                if (i2 == 0) {
                    #pragma unroll
                    for (int j = 0; j < HG_WQ; j++) bq[kt2 & 1][j] = ldB(kt2, j);
                }
                aq[s2 % (HG_D + 1)] = ldA(kt2, i2);
            }
            #pragma unroll
            for (int j = 0; j < HG_WQ; j++) {
                acc[i * HG_WQ + j] =
                    HG_WMMA(aq[s % (HG_D + 1)], bq[kt & 1][j], acc[i * HG_WQ + j]);
            }
            HG_SCHED();
        }
    };

    gload(0);
    #pragma unroll 1
    for (int t = 0; t < HG_NSTAGE; t++) {
        lstore();
        HG_BARRIER();
        // Stage t + 1 (the last iteration re-loads the last stage; never consumed).
        gload(min(t + 1, HG_NSTAGE - 1));
        HG_SCHED();
        compute();
        HG_SCHED();
        HG_BARRIER();
    }

    // Epilogue.  acc[i * WQ + j][e] = C[m = wave_m * 64 + i * 16 + 2 * e + (lane >> 4)]
    //                                  [token = j * 16 + ml].
    // `+ 0.0f` is the incumbent's `o += bv` with has_bias == 0 (-0.0 -> +0.0).  The BF16 bits go
    // through LDS (the K staging buffer is dead behind the loop's last barrier) so the global
    // stores are 16-byte, row-contiguous.
    unsigned short* slab = (unsigned short*)lds;
    const int row_half = lane >> 4;
    #pragma unroll
    for (int i = 0; i < HG_WP; i++) {
        #pragma unroll
        for (int j = 0; j < HG_WQ; j++) {
            const hg_float8_t v = acc[i * HG_WQ + j];
            const int tok = j * 16 + ml;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const int m = wave_m * HG_WP * 16 + i * 16 + 2 * e + row_half;
                slab[tok * HG_SLAB + m] = hg_hc_act(v[e] + 0.0f, scale);
            }
        }
    }
    HG_BARRIER();
    constexpr int OCH = HG_M / 8;   // 16-byte chunks per output row
    #pragma unroll 1
    for (int c = tid; c < HG_BN * OCH; c += HG_NT) {
        const int tok = c / OCH;
        const int ch  = c % OCH;
        const int row = n_block + tok;
        if (row < B) {
            *(uint4*)(out + (size_t)row * HG_M + ch * 8) =
                *(const uint4*)(slab + tok * HG_SLAB + ch * 8);
        }
    }
}
