// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx1201 Qwen4 HC read down projection with the activation in the epilogue:
// ONE launch for what the default prefill chain runs as
//   qwen4_wmma_lds_160_80_32_80_k32_p_s4   (4 K-split partials -> scratch)
//   qwen4_wmma_lds_reduce                  (P0 + P1 + P2 + P3, ascending)
//   hc_activation_fused_f32                (bf16 -> * scale -> bf16 -> SiLU -> bf16)
// (gemm_f16_x_f16_wmma_lds_splitk.hip, tensor_ops.hip).  Y[b, m] = sum_k A[m, k] * X[b, k]
// with M = 320, K = 10240 (= 4 * hidden), and the packed BF16 activation lands in
// `out[b, m]`, the operand hyper_read_up_wmma_bf16_gfx1201_t128 reads.  The F32 `low` the
// chain also leaves behind is dead in this route (the activation is called with `bf16_out`,
// which leaves it as the raw GEMM output, and the read tail only reads the packed BF16), so
// it is not written.  The 21 MB split scratch, its re-read by the reducer, the F32 Y round
// trip and the activation's re-read of Y never happen: per row the kernel moves the 20 KB
// X row in and the 640 B BF16 row out.
//
// BYTE-IDENTICAL to that chain, per output:
//   * the four split partials are four independent accumulator chains, each started at
//     zero and fed the 16-wide K substeps of its quarter [z * K / 4, (z + 1) * K / 4)
//     in ascending order, with the same fragment operands (A: row m, X: row b,
//     K elements [8 * (l >> 4), +8) of each substep per lane half) as the split-K tile;
//   * the partials are summed s = P0; s += P1; s += P2; s += P3, which is the reducer's
//     order, in F32 (no contraction: there is no multiply to fuse);
//   * the activation is hc_activation_fused_f32's expression with `bf16_out`.
//
// Geometry.  One wave owns an 80 (M) x 32 (B) patch (5 x 2 WMMA fragments, 80 accumulator
// VGPRs) and runs the four splits back to back: after split z the finished accumulator is
// added into a running sum (a second 80 VGPRs) instead of leaving through memory.  The
// workgroup is the whole M = 320 (four M waves) by 64 rows of B (two B waves): X is read from
// DRAM once and the 6.5 MB weight shadow stays L2-resident (every workgroup walks the same K
// quarter at about the same time).  Grid [ceil(B / 64)], block 256, 49664 B of LDS.
//
// Main loop.  K is staged 32 per barrier through a double-buffered LDS tile.  The tile layout
// is one plane per (K substep, lane half) 16-byte chunk column, [plane][row][8 halves]
// (plane pitch = 2 mod 8 rows): a fragment read's eight-lane groups are 128 contiguous bytes
// and a stage store's eight lanes are bank-disjoint (the row-major [row][16] layout of the
// split-K tile has 2-way bank conflicts on every fragment read; this kernel is LDS-bound).
// Global staging loads are 16-byte chunks of TWO consecutive stages at once (K = 64 elements
// = one 128-byte cache line per row: 8 lanes per row, 4 lines per wave instruction instead of
// 8 half lines); the lanes holding the first stage's chunk store it now, the others store the
// second stage's chunk one stage later.  The barrier is split: a wave signals once its stores
// for stage s + 1 and its fragment reads of stage s are complete (s_wait_dscnt 0), and waits
// only after half of the second substep's WMMAs, so barrier skew and the next stage's first
// fragment reads hide under tensor work.  sched_barrier pins that order.
//
// PRECONDITIONS (host-checked): M == 320, K == 10240, rows * K * 2 < 2^32, out non-null,
// A / X / out 16-byte aligned.  Rows past B clamp the staged row to B - 1 and are discarded.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#if !defined(__gfx1201__)
#error "qwen4_hc_down_fused.hip: gfx1201 only"
#endif

typedef _Float16 __attribute__((ext_vector_type(8)))  hd_half8_t;
typedef float    __attribute__((ext_vector_type(8)))  hd_float8_t;

constexpr int HD_M     = 320;     // HC low rank: the whole M in one workgroup
constexpr int HD_K     = 10240;   // 4 x hidden (2560): the only K the route admits
constexpr int HD_SPLIT = 4;       // the incumbent tile's K splits
constexpr int HD_KLEN  = HD_K / HD_SPLIT;
constexpr int HD_KS    = 32;      // K elements staged per barrier
constexpr int HD_SPS   = HD_KLEN / HD_KS;   // stages per split (even)
constexpr int HD_WP    = 5;       // A fragments per wave (80 rows)
constexpr int HD_WQ    = 2;       // X fragments per wave (32 rows of B)
constexpr int HD_NWM   = 4;       // M waves
constexpr int HD_NWN   = 2;       // B waves
constexpr int HD_BN    = 32 * HD_NWN;
constexpr int HD_NT    = HD_NWM * HD_NWN * 32;
constexpr int HD_PADROWS = 2;     // plane pitch = 2 (mod 8) rows
constexpr int HD_A_PS  = HD_M + HD_PADROWS;   // rows per A plane
constexpr int HD_X_PS  = HD_BN + HD_PADROWS;
constexpr int HD_PLANES = 2 * (HD_KS / 16);   // (substep, lane half) chunk columns per stage
constexpr int HD_A_BUF = HD_PLANES * HD_A_PS * 8;   // halves per A buffer
constexpr int HD_X_BUF = HD_PLANES * HD_X_PS * 8;
constexpr int HD_ROWS2 = HD_NT / 8;           // rows per pair-load pass (8 lanes per row)
constexpr int HD_A_PASS = HD_M * 8 / HD_NT;   // 10
constexpr int HD_X_PASS = HD_BN * 8 / HD_NT;  // 2

static_assert(HD_SPS % 2 == 0, "stages per split must be even (static buffer parity)");
static_assert(HD_KLEN % (2 * HD_KS) == 0, "pairs of stages must not straddle a split");
static_assert((HD_M * 8) % HD_NT == 0 && (HD_BN * 8) % HD_NT == 0, "pair chunks must divide");
static_assert(HD_M == HD_NWM * 16 * HD_WP && HD_BN == HD_NWN * 16 * HD_WQ, "tile geometry");

// hc_activation_fused_f32 (tensor_ops.hip) with `bf16_out`, same expression, same casts.
static __device__ __forceinline__ unsigned short hd_hc_act(float v, float scale) {
    float value = static_cast<float>(static_cast<hip_bfloat16>(v));
    value *= scale;
    value = static_cast<float>(static_cast<hip_bfloat16>(value));
    value = value / (1.0f + expf(-value));
    return static_cast<hip_bfloat16>(value).data;
}

// Split LDS-only workgroup barrier (same idiom as gdn_chunk_scan.gfx1201.hip GS_BAR, split
// into signal and wait): the "local" fences keep in-flight global prefetches in flight, and
// each wave waits for its own LDS ops (s_wait_dscnt 0, which the waitcnt inserter cannot
// erase, llvm/llvm-project#220357) before it signals.
extern "C" __device__ void hd_wait_dscnt(unsigned short) __asm("llvm.amdgcn.s.wait.dscnt");
#define HD_BAR_SIGNAL() do { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local"); \
                             hd_wait_dscnt(0); __builtin_amdgcn_s_barrier_signal(-1); } while (0)
#define HD_BAR_WAIT()   do { __builtin_amdgcn_s_barrier_wait(-1); \
                             __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local"); } while (0)
#define HD_SCHED()      __builtin_amdgcn_sched_barrier(0)
#define HD_WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c)

extern "C" __global__ __launch_bounds__(HD_NT, 1)
void qwen4_hc_down_fused_b64_k32(
    const _Float16* __restrict__ A,       // [320][10240] F16 weight shadow
    const _Float16* __restrict__ X,       // [B][10240] F16
    unsigned short* __restrict__ out,     // [B][320] packed BF16 bits (activated)
    int K, int B, float scale
) {
    (void)K;   // host guarantees K == HD_K; every offset below is a constant
    __shared__ __align__(32) _Float16 a_lds[2 * HD_A_BUF];
    __shared__ __align__(32) _Float16 x_lds[2 * HD_X_BUF];

    const int n_block = blockIdx.x * HD_BN;
    if (n_block >= B) return;

    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int wave_id = tid >> 5;
    const int ml      = lane & 15;
    const int kh      = lane >> 4;
    const int wave_m  = wave_id % HD_NWM;
    const int wave_n  = wave_id / HD_NWM;
    const int n_wave  = n_block + wave_n * (16 * HD_WQ);

    // Pair staging: thread chunk (row, q): row = tid >> 3 (+ HD_ROWS2 per pass), q = tid & 7
    // selects the 16 bytes [q * 16, +16) of the row's 128-byte K pair; q >> 2 says which of
    // the pair's two stages the chunk belongs to, (q & 3) is its plane within that stage.
    const int p_row  = tid >> 3;
    const int p_q    = tid & 7;
    const int p_hi   = p_q >> 2;
    const int p_part = p_q & 3;
    const unsigned pa_off0 = (unsigned)(p_row * HD_K + p_q * 8) * 2u;
    const unsigned pa_dst0 = (unsigned)(p_part * HD_A_PS + p_row) * 16u;
    unsigned px_off[HD_X_PASS], px_dst[HD_X_PASS];
    #pragma unroll
    for (int p = 0; p < HD_X_PASS; p++) {
        const int r = n_block + p_row + p * HD_ROWS2;
        px_off[p] = (unsigned)(((r < B) ? r : (B - 1)) * HD_K + p_q * 8) * 2u;
        px_dst[p] = (unsigned)(p_part * HD_X_PS + p_row + p * HD_ROWS2) * 16u;
    }
    const char* a_base   = (const char*)A;
    const char* x_base   = (const char*)X;
    char*       a_lds_b  = (char*)a_lds;
    char*       x_lds_b  = (char*)x_lds;

    hd_half8_t a_reg[HD_A_PASS];
    hd_half8_t x_reg[HD_X_PASS];

    auto load_pair = [&](int k0) {           // K offset of the pair's first stage
        const char* ak = a_base + (long long)k0 * 2;
        const char* xk = x_base + (long long)k0 * 2;
        #pragma unroll
        for (int p = 0; p < HD_A_PASS; p++)
            a_reg[p] = *(const hd_half8_t*)(ak + pa_off0 + (unsigned)(p * HD_ROWS2 * HD_K * 2));
        #pragma unroll
        for (int p = 0; p < HD_X_PASS; p++)
            x_reg[p] = *(const hd_half8_t*)(xk + px_off[p]);
    };
    auto store_half = [&](int buf) {         // the chunks of the pair's stage with parity `buf`
        if (p_hi == buf) {
            #pragma unroll
            for (int p = 0; p < HD_A_PASS; p++)
                *(hd_half8_t*)(a_lds_b + buf * (HD_A_BUF * 2) + pa_dst0 + p * (HD_ROWS2 * 16)) = a_reg[p];
            #pragma unroll
            for (int p = 0; p < HD_X_PASS; p++)
                *(hd_half8_t*)(x_lds_b + buf * (HD_X_BUF * 2) + px_dst[p]) = x_reg[p];
        }
    };

    // Fragment lane offsets (halves) inside a buffer: plane = kt * 2 + lane half.
    const int a_frag = (kh * HD_A_PS + wave_m * (16 * HD_WP) + ml) * 8;
    const int x_frag = (kh * HD_X_PS + wave_n * (16 * HD_WQ) + ml) * 8;

    hd_float8_t acc[HD_WP * HD_WQ];
    hd_float8_t run[HD_WP * HD_WQ];
    #pragma unroll
    for (int i = 0; i < HD_WP * HD_WQ; i++) {
        acc[i] = hd_float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    // Software-pipelined stage loop.  Per stage s (LDS buffer `buf`; stage s + 1 is already
    // complete in the other buffer, the pair holding stages s + 2 / s + 3 sits in registers):
    //   WMMA(s, kt0)  ->  reads(s, kt1)  ->  signal  ->  WMMA(s, kt1)  ->  wait  ->
    //   store this stage's half of the pair into `buf` (stage s + 2)  ->  (odd stages: load the
    //   pair after next)  ->  reads(s + 1, kt0), carried to the next iteration in fa / fb.
    // One fragment set (fa / fb): substep 1's fragments are read after substep 0's WMMAs have
    // consumed it, the other waves of the SIMD cover the read latency.
    hd_half8_t fa[HD_WP], fb[HD_WQ];
    auto read_kt0 = [&](int buf) {
        const _Float16* al = a_lds + buf * HD_A_BUF + a_frag;
        const _Float16* xl = x_lds + buf * HD_X_BUF + x_frag;
        #pragma unroll
        for (int j = 0; j < HD_WQ; j++) fb[j] = *(const hd_half8_t*)(xl + j * 16 * 8);
        #pragma unroll
        for (int i = 0; i < HD_WP; i++) fa[i] = *(const hd_half8_t*)(al + i * 16 * 8);
    };
    auto mma = [&]() {
        #pragma unroll
        for (int j = 0; j < HD_WQ; j++) {
            #pragma unroll
            for (int i = 0; i < HD_WP; i++)
                acc[i * HD_WQ + j] = HD_WMMA(fa[i], fb[j], acc[i * HD_WQ + j]);
        }
    };
    auto stage = [&](int buf, int k_next) {
        const _Float16* al = a_lds + buf * HD_A_BUF + a_frag;
        const _Float16* xl = x_lds + buf * HD_X_BUF + x_frag;
        HD_SCHED();
        mma();                                // substep 0
        HD_SCHED();
        #pragma unroll
        for (int j = 0; j < HD_WQ; j++) fb[j] = *(const hd_half8_t*)(xl + (2 * HD_X_PS + j * 16) * 8);
        #pragma unroll
        for (int i = 0; i < HD_WP; i++) fa[i] = *(const hd_half8_t*)(al + (2 * HD_A_PS + i * 16) * 8);
        HD_SCHED();
        HD_BAR_SIGNAL();
        HD_SCHED();
        mma();                                // substep 1
        HD_SCHED();
        HD_BAR_WAIT();
        HD_SCHED();
        store_half(buf);                      // stage s + 2's chunks -> buf
        if (buf == 1) load_pair(k_next);      // pair after next in flight
        HD_SCHED();
        read_kt0(buf ^ 1);                    // stage s + 1, substep 0
        HD_SCHED();
    };

    // Prologue: stages 0 and 1 (one pair) into buffers 0 and 1, the next pair in flight.
    load_pair(0);
    store_half(0);
    store_half(1);
    load_pair(2 * HD_KS);
    HD_BAR_SIGNAL();
    HD_BAR_WAIT();
    read_kt0(0);

    #pragma unroll 1
    for (int z = 0; z < HD_SPLIT; z++) {
        const int base = z * HD_KLEN;
        #pragma unroll 1
        for (int t = 0; t < HD_SPS; t += 2) {
            // Entering: stage (z, t) in buffer 0 with its substep-0 fragments loaded.  The K
            // offset of the next pair to load (stages t + 4 / t + 5, into the next split at the
            // edge) is clamped so the last prefetches past the end of the last split re-read the
            // final pair instead of branching; their data is never consumed.
            stage(0, 0);
            int k1 = base + (t + 4) * HD_KS;
            k1 = (k1 < HD_K) ? k1 : (HD_K - 2 * HD_KS);
            stage(1, k1);
        }
        // Split z done: fold its accumulator into the running sum in the reducer's order.
        if (z == 0) {
            #pragma unroll
            for (int i = 0; i < HD_WP * HD_WQ; i++) run[i] = acc[i];
        } else {
            #pragma unroll
            for (int i = 0; i < HD_WP * HD_WQ; i++) run[i] = run[i] + acc[i];
        }
        #pragma unroll
        for (int i = 0; i < HD_WP * HD_WQ; i++) {
            acc[i] = hd_float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }
    }

    // Epilogue: lane (ml, kh) holds Y[n_wave + j * 16 + ml][wave_m * 80 + i * 16 + 8 * kh + e];
    // eight consecutive outputs of one row leave as one 16-byte store.
    #pragma unroll
    for (int j = 0; j < HD_WQ; j++) {
        const int out_b = n_wave + j * 16 + ml;
        if (out_b >= B) continue;
        unsigned short* orow = out + (long long)out_b * HD_M;
        #pragma unroll
        for (int i = 0; i < HD_WP; i++) {
            const int m0 = wave_m * (16 * HD_WP) + i * 16 + 8 * kh;
            const hd_float8_t v = run[i * HD_WQ + j];
            unsigned int w[4];
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned int lo = hd_hc_act(v[2 * e], scale);
                const unsigned int hi = hd_hc_act(v[2 * e + 1], scale);
                w[e] = lo | (hi << 16);
            }
            *(uint4*)(orow + m0) = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
}
