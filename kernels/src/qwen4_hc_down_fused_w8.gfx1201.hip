// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// W8 twin of qwen4_hc_down_fused.hip (gfx1201): the Qwen4 HC read down projection
//   Y[b, m] = hc_act(sum_k W[m, k] * X[b, k]),  M = 320, K = 10240,
// with the A operand taken from the PLANAR W8 weight instead of the BF16->F16 shadow:
//   q : i8  [320][10240]   row-major
//   d : f16 [320][320]     row-major, one scale per 32-K block (the Q8_0 numbers)
//   weight value = (float)q * (float)d.
// X is the F16 hyper_norm_f16 output [B][10240], the output is packed BF16 bits [B][320] with
// hc_activation_fused_f32's `bf16_out` expression, exactly as the BF16 kernel.
//
// What changed against the BF16 kernel (tiling, LDS layout, pipelining are untouched):
//   * The A tile is staged from int8: each pair load is 8 bytes per lane (8 K elements, the same
//     (row, plane) chunk mapping as the BF16 kernel's 16-byte chunks), converted to F16 on the
//     way into the same [plane][row][8 halves] LDS layout.  int8 -> F16 is EXACT; the conversion
//     is the magic-number form (0x6400 | (q ^ 0x80)) - 1152.0h done with v_perm_b32 + packed F16
//     subtracts, 2.5 VALU ops per element pair instead of a cvt chain.
//   * The scale is NEVER folded into an operand.  The wave's stage accumulator `acc` starts at
//     zero for every 32-K stage (= one scale block); after the stage's two WMMA substeps it is
//     folded into the running F32 sum with the block's scale:
//         run[e] = fmaf(d[m0 + e], acc[e], run[e])        (m0 = A row of accumulator element 0)
//     A lane's accumulator element e is A row m0 + e (m0 = wave_m * 80 + i * 16 + 8 * (l >> 4)),
//     so a fragment needs 8 consecutive scales: the [320]-row scale slice of each stage is staged
//     into LDS together with the operands (a 4-byte load per row covers a stage PAIR, same
//     lifetime as the A chunks; four buffers, stage index mod 4, so the stage s + 2 write cannot
//     land on stage s's scales) and each lane reads its 8 halves of fragment i as one 16-byte
//     LDS read at fold time.  The fold runs after the stage's stores and the next pair's loads
//     are issued.
//   * Because `run` already was the second accumulator bank of the BF16 kernel (split partials),
//     the VGPR budget is the same.  The four-split structure is gone: the running sum walks the
//     320 stages in ascending K.
//   * Not byte-identical to the BF16 kernel (different weights); each output is
//         hc_act( sum_stage fmaf(d_stage, WMMA-sum_stage(q * x), running) ).
//
// PRECONDITIONS (host-checked): M == 320, K == 10240, rows * K * 2 < 2^32, out non-null,
// q / d / X / out 16-byte aligned.  Rows past B clamp the staged row to B - 1 and are discarded.
// Grid [ceil(B / 64)], block 256, static LDS 52224 B.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "qwen4_hc_down_fused_w8.gfx1201.hip: gfx1201 only"
#endif

typedef _Float16 __attribute__((ext_vector_type(8)))  hd_half8_t;
typedef float    __attribute__((ext_vector_type(8)))  hd_float8_t;

constexpr int HD_M     = 320;     // HC low rank: the whole M in one workgroup
constexpr int HD_K     = 10240;   // 4 x hidden (2560): the only K the route admits
constexpr int HD_NB    = HD_K / 32;         // scale blocks per weight row (320)
constexpr int HD_KS    = 32;      // K elements staged per barrier (= one scale block)
constexpr int HD_NSTG  = HD_K / HD_KS;      // stages (320, even)
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

static_assert(HD_NSTG % 2 == 0, "stages must pair up (static buffer parity)");
static_assert((HD_M * 8) % HD_NT == 0 && (HD_BN * 8) % HD_NT == 0, "pair chunks must divide");
static_assert(HD_M == HD_NWM * 16 * HD_WP && HD_BN == HD_NWN * 16 * HD_WQ, "tile geometry");
static_assert(HD_M <= HD_NT + 64, "scale rows: one pass of HD_NT threads plus a 64-thread tail");

// hc_activation_fused_f32 (tensor_ops.hip) with `bf16_out`, same expression, same casts.
static __device__ __forceinline__ unsigned short hd_hc_act(float v, float scale) {
    float value = static_cast<float>(static_cast<hip_bfloat16>(v));
    value *= scale;
    value = static_cast<float>(static_cast<hip_bfloat16>(value));
    value = value / (1.0f + expf(-value));
    return static_cast<hip_bfloat16>(value).data;
}

// 8 signed bytes -> 8 exact F16.  f16(0x6400 | n) = 1024 + n for n in [0, 1024); with
// n = q ^ 0x80 = q + 128 (q the signed byte) the F16 subtract of 1152 gives q exactly.
static __device__ __forceinline__ hd_half8_t hd_q8x8_to_f16(uint2 r) {
    const unsigned x0 = r.x ^ 0x80808080u;
    const unsigned x1 = r.y ^ 0x80808080u;
    // v_perm_b32: selector bytes 0-3 pick bytes of the second source (the data), 4-7 of the first
    // (the 0x64 constant).  Output half = { byte, 0x64 }.
    uint4 u;
    u.x = __builtin_amdgcn_perm(0x64646464u, x0, 0x04010400u);
    u.y = __builtin_amdgcn_perm(0x64646464u, x0, 0x04030402u);
    u.z = __builtin_amdgcn_perm(0x64646464u, x1, 0x04010400u);
    u.w = __builtin_amdgcn_perm(0x64646464u, x1, 0x04030402u);
    const hd_half8_t h = __builtin_bit_cast(hd_half8_t, u);
    return h - (_Float16)1152.0f;
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
void qwen4_hc_down_fused_w8_b64_k32(
    const signed char* __restrict__ Q,    // [320][10240] int8 planar weight
    const _Float16* __restrict__ D,       // [320][320] F16 per-32 scales
    const _Float16* __restrict__ X,       // [B][10240] F16
    unsigned short* __restrict__ out,     // [B][320] packed BF16 bits (activated)
    int K, int B, float scale
) {
    (void)K;   // host guarantees K == HD_K; every offset below is a constant
    __shared__ __align__(32) _Float16 a_lds[2 * HD_A_BUF];
    __shared__ __align__(32) _Float16 x_lds[2 * HD_X_BUF];
    __shared__ __align__(32) _Float16 s_lds[4 * HD_M];     // [stage mod 4][A row] scales

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
    // selects the 8 K elements [q * 8, +8) of the row's 64-element K pair; q >> 2 says which of
    // the pair's two stages the chunk belongs to, (q & 3) is its plane within that stage.
    const int p_row  = tid >> 3;
    const int p_q    = tid & 7;
    const int p_hi   = p_q >> 2;
    const int p_part = p_q & 3;
    const unsigned pa_off0 = (unsigned)(p_row * HD_K + p_q * 8);       // bytes (int8)
    const unsigned pa_dst0 = (unsigned)(p_part * HD_A_PS + p_row) * 16u;
    unsigned px_off[HD_X_PASS], px_dst[HD_X_PASS];
    #pragma unroll
    for (int p = 0; p < HD_X_PASS; p++) {
        const int r = n_block + p_row + p * HD_ROWS2;
        px_off[p] = (unsigned)(((r < B) ? r : (B - 1)) * HD_K + p_q * 8) * 2u;
        px_dst[p] = (unsigned)(p_part * HD_X_PS + p_row + p * HD_ROWS2) * 16u;
    }
    // Scale staging: thread tid loads the stage pair's two scales of A row tid; threads
    // 0..HD_M-HD_NT+63 (tid < 64) take row HD_NT + tid as well.
    const bool s_tail = tid < (HD_M - HD_NT);
    const unsigned d_off0 = (unsigned)(tid * HD_NB * 2);                       // bytes
    const unsigned d_off1 = (unsigned)((s_tail ? (HD_NT + tid) : (HD_M - 1)) * HD_NB * 2);
    const char* d_base = (const char*)D;

    const char* a_base   = (const char*)Q;
    const char* x_base   = (const char*)X;
    char*       a_lds_b  = (char*)a_lds;
    char*       x_lds_b  = (char*)x_lds;

    uint2 a_reg[HD_A_PASS];
    hd_half8_t x_reg[HD_X_PASS];
    unsigned s_reg0, s_reg1;

    auto load_pair = [&](int k0) {           // K offset of the pair's first stage (multiple of 64)
        const char* ak = a_base + k0;
        const char* xk = x_base + (long long)k0 * 2;
        #pragma unroll
        for (int p = 0; p < HD_A_PASS; p++)
            a_reg[p] = *(const uint2*)(ak + pa_off0 + (unsigned)(p * HD_ROWS2 * HD_K));
        #pragma unroll
        for (int p = 0; p < HD_X_PASS; p++)
            x_reg[p] = *(const hd_half8_t*)(xk + px_off[p]);
        // stage pair k0 / 32, k0 / 32 + 1: one 4-byte aligned word (non-tail threads re-read the
        // last row, unused)
        s_reg0 = *(const unsigned*)(d_base + d_off0 + (unsigned)(k0 >> 4));
        s_reg1 = *(const unsigned*)(d_base + d_off1 + (unsigned)(k0 >> 4));
    };
    auto store_half = [&](int buf, int s_dst) {   // the chunks of the pair's stage with parity `buf`
        // (`s_dst`: scale buffer of that stage, stage index mod 4)
        if (p_hi == buf) {
            #pragma unroll
            for (int p = 0; p < HD_A_PASS; p++) {
                // convert-and-store one chunk at a time: the pins keep the F16 temporaries of
                // all ten chunks from being live at once (the stage loop has ~5 VGPRs of slack)
                HD_SCHED();
                *(hd_half8_t*)(a_lds_b + buf * (HD_A_BUF * 2) + pa_dst0 + p * (HD_ROWS2 * 16)) =
                    hd_q8x8_to_f16(a_reg[p]);
            }
            #pragma unroll
            for (int p = 0; p < HD_X_PASS; p++)
                *(hd_half8_t*)(x_lds_b + buf * (HD_X_BUF * 2) + px_dst[p]) = x_reg[p];
        }
        // This stage's scale of A rows tid (and HD_NT + tid): low half of the pair word for
        // parity 0, high half for parity 1.
        unsigned short* sd = (unsigned short*)s_lds + (s_dst & 3) * HD_M;
        sd[tid] = (unsigned short)(buf ? (s_reg0 >> 16) : (s_reg0 & 0xffffu));
        if (s_tail) sd[HD_NT + tid] = (unsigned short)(buf ? (s_reg1 >> 16) : (s_reg1 & 0xffffu));
    };

    // Fragment lane offsets (halves) inside a buffer: plane = kt * 2 + lane half.
    const int a_frag = (kh * HD_A_PS + wave_m * (16 * HD_WP) + ml) * 8;
    const int x_frag = (kh * HD_X_PS + wave_n * (16 * HD_WQ) + ml) * 8;
    const int s_frag = wave_m * (16 * HD_WP) + 8 * kh;        // halves inside a scale buffer

    hd_float8_t acc[HD_WP * HD_WQ];
    hd_float8_t run[HD_WP * HD_WQ];
    const hd_float8_t zero8 = hd_float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int i = 0; i < HD_WP * HD_WQ; i++) run[i] = zero8;

    // Software-pipelined stage loop.  Per stage s (LDS buffer `buf`; stage s + 1 is already
    // complete in the other buffer, the pair holding stages s + 2 / s + 3 sits in registers):
    //   WMMA(s, kt0; acc = 0 + ...)  ->  reads(s, kt1) + scales(s)  ->  signal  ->  WMMA(s, kt1)
    //   ->  wait  ->  store this stage's half of the pair into `buf` (stage s + 2)  ->  (odd
    //   stages: load the pair after next)  ->  fold acc into run with the stage's scales  ->
    //   reads(s + 1, kt0), carried to the next iteration in fa / fb.
    hd_half8_t fa[HD_WP], fb[HD_WQ];
    auto read_kt0 = [&](int buf) {
        const _Float16* al = a_lds + buf * HD_A_BUF + a_frag;
        const _Float16* xl = x_lds + buf * HD_X_BUF + x_frag;
        #pragma unroll
        for (int j = 0; j < HD_WQ; j++) fb[j] = *(const hd_half8_t*)(xl + j * 16 * 8);
        #pragma unroll
        for (int i = 0; i < HD_WP; i++) fa[i] = *(const hd_half8_t*)(al + i * 16 * 8);
    };
    auto mma0 = [&]() {                      // substep 0: the stage accumulator starts at zero
        #pragma unroll
        for (int j = 0; j < HD_WQ; j++) {
            #pragma unroll
            for (int i = 0; i < HD_WP; i++)
                acc[i * HD_WQ + j] = HD_WMMA(fa[i], fb[j], zero8);
        }
    };
    auto mma1 = [&]() {
        #pragma unroll
        for (int j = 0; j < HD_WQ; j++) {
            #pragma unroll
            for (int i = 0; i < HD_WP; i++)
                acc[i * HD_WQ + j] = HD_WMMA(fa[i], fb[j], acc[i * HD_WQ + j]);
        }
    };
    // run[e] += d[e] * acc[e], in place.  Inline asm so the destination is pinned to the run
    // register: left to the allocator, the 80 running sums come out of each fold in permuted
    // registers and the loop-carried copies spill (the VGPR file is full).  The scale pair word
    // holds two consecutive F16 scales; op_sel picks the half, the product is an exact F32 fma.
    auto fold = [&](const _Float16* sl) {
        #pragma unroll
        for (int i = 0; i < HD_WP; i++) {
            const uint4 sv = *(const uint4*)(sl + i * 16);
            #pragma unroll
            for (int j = 0; j < HD_WQ; j++) {
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    const unsigned w = (e >> 1) == 0 ? sv.x : (e >> 1) == 1 ? sv.y : (e >> 1) == 2 ? sv.z : sv.w;
                    float r = run[i * HD_WQ + j][e];
                    const float a = acc[i * HD_WQ + j][e];
                    if (e & 1)
                        asm("v_fma_mix_f32 %0, %1, %2, %0 op_sel:[1,0,0] op_sel_hi:[1,0,0]"
                            : "+v"(r) : "v"(w), "v"(a));
                    else
                        asm("v_fma_mix_f32 %0, %1, %2, %0 op_sel_hi:[1,0,0]"
                            : "+v"(r) : "v"(w), "v"(a));
                    run[i * HD_WQ + j][e] = r;
                }
            }
        }
    };
    auto stage = [&](int buf, int k_next, int sidx) {   // sidx: this stage's index
        const _Float16* al = a_lds + buf * HD_A_BUF + a_frag;
        const _Float16* xl = x_lds + buf * HD_X_BUF + x_frag;
        const _Float16* sl = s_lds + (sidx & 3) * HD_M + s_frag;
        HD_SCHED();
        mma0();                               // substep 0
        HD_SCHED();
        #pragma unroll
        for (int j = 0; j < HD_WQ; j++) fb[j] = *(const hd_half8_t*)(xl + (2 * HD_X_PS + j * 16) * 8);
        #pragma unroll
        for (int i = 0; i < HD_WP; i++) fa[i] = *(const hd_half8_t*)(al + (2 * HD_A_PS + i * 16) * 8);
        HD_SCHED();
        HD_BAR_SIGNAL();
        HD_SCHED();
        mma1();                               // substep 1
        HD_SCHED();
        HD_BAR_WAIT();
        HD_SCHED();
        store_half(buf, sidx + 2);            // stage s + 2's chunks -> buf, its scales -> (s + 2) mod 4
        if (buf == 1) load_pair(k_next);      // pair after next in flight
        HD_SCHED();
        fold(sl);
        HD_SCHED();
        read_kt0(buf ^ 1);                    // stage s + 1, substep 0
        HD_SCHED();
    };

    // Prologue: stages 0 and 1 (one pair) into buffers 0 and 1, the next pair in flight.
    load_pair(0);
    store_half(0, 0);
    store_half(1, 1);
    load_pair(2 * HD_KS);
    HD_BAR_SIGNAL();
    HD_BAR_WAIT();
    read_kt0(0);

    #pragma unroll 1
    for (int t = 0; t < HD_NSTG; t += 2) {
        // Entering: stage t in buffer 0 with its substep-0 fragments loaded.  The K offset of
        // the next pair to load (stages t + 4 / t + 5) is clamped so the last prefetches past
        // the end re-read the final pair instead of branching; their data is never consumed.
        stage(0, 0, t);
        int k1 = (t + 4) * HD_KS;
        k1 = (k1 < HD_K) ? k1 : (HD_K - 2 * HD_KS);
        stage(1, k1, t + 1);
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
