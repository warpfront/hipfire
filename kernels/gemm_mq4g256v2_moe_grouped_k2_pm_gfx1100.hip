// PM twin (gfx1100) of gemm_mq4g256v2_moe_grouped_wmma_k2: card-judged R1Moe-04-xlds-w4-t2 verbatim except the
// kernel symbol rename. Sealed by scripts/pm-hip-twin.sh.
// hip_loop: rows_per_block=64 block=128 tile_run=2
// R1Moe-04-xlds-w4-t2  (gemm_mq4g256v2_moe_grouped_wmma_k2, gfx1100, bit-identical to the shipped kernel)
// Idea: x4: 4 waves = 64 rows share the expert run's X through double-buffered LDS (one barrier per K64 chunk, global X traffic /4), weights stay direct from global (no LDS A staging), tile_run=2, packed dequant, A prefetch
// Predicted candidate/shipped time ratio: 0.5
// Config: rows tiles/wave R=1, slot tiles/leader T=2, waves/WG W=4, prefetch PF=1, setprio=0.
// M must be a multiple of 16 (grid X = ceil(M/64)).
#define CFG_R 1
#define CFG_T 2
#define CFG_W 4
#define CFG_PF 1
#define CFG_PRIO 0
#define CFG_WPE 8
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Grouped-WMMA-GEMM for MoE prefill over MQ4G256V2 (qt=44) routed experts
// (gemm_mq4g256v2_moe_grouped_wmma_k2, gfx1100).  Group layout, 136 B:
//   [0..2) f16 scale w0..127  [2..4) f16 zero w0..127  [4..6) f16 scale w128..255  [6..8) f16 zero w128..255
//   [8..136) 128 nibble bytes: byte i = weight 2i (low), 2i+1 (high).
// Numerics are the shipped kernel's, element for element: every output element is the F16 WMMA K16-step
// chain acc = wmma(a_kt, b_kt, acc) over kt = 0 .. K/16-1 ascending from +0, with a = fma_f16(sc, q, zp) (q the
// exact nibble) and the shipped's dual affine grid per K128 half-group.
//
// CFG_R rows tiles of 16 weight rows per wave, CFG_T consecutive slot tiles of one expert per leader workgroup
// (tile_run), CFG_W waves per workgroup (CFG_W > 1: the X tile is staged in LDS and shared by the waves),
// CFG_PF explicit one-pair software prefetch, CFG_PRIO s_setprio around the WMMA batch.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef _Float16 __attribute__((ext_vector_type(2))) half2_p_t;
typedef unsigned int __attribute__((ext_vector_type(8))) uint8_p_t;

// Weights 2j, 2j+1 of nibble words (lo, hi) as sc*q + zp in F16 (packed: perm + and/or + pk_add + pk_fma per pair).
__device__ __forceinline__ half16_t dq_nibbles(unsigned int lo, unsigned int hi, _Float16 sc, _Float16 zp) {
    // Byte p of a word holds weights 2p (low nibble) and 2p + 1 (high nibble).  One byte permute copies byte p
    // into bytes 0 and 2, one and-or keeps the low nibble at bits 0-3 of half 0 (0x6400 | q is exactly 1024 + q)
    // and the high nibble at bits 4-7 of half 1 (0x5400 | q << 4 is exactly 64 + q), so subtracting {1024, 64}
    // yields both codes exactly and one fused multiply-add per pair is the per-element sc * q + zp.
    const half2_p_t sc2 = {sc, sc};
    const half2_p_t zp2 = {zp, zp};
    const half2_p_t bias = {(_Float16)1024.0f, (_Float16)64.0f};
    uint8_p_t out;
#pragma unroll
    for (int p = 0; p < 8; p++) {
        const unsigned int w = p < 4 ? lo : hi;
        const unsigned int b = (unsigned)(p & 3);
        const unsigned int bits =
            (__builtin_amdgcn_perm(w, w, b | (12u << 8) | (b << 16) | (12u << 24)) & 0x00F0000Fu) | 0x54006400u;
        const half2_p_t q2 = __builtin_bit_cast(half2_p_t, bits) - bias;
        out[p] = __builtin_bit_cast(unsigned int, __builtin_elementwise_fma(sc2, q2, zp2));
    }
    return __builtin_bit_cast(half16_t, out);
}

// One lane's data of K32 (one pair of K16 steps) of one 16-row tile: the affine header word of the pair's K128
// half-group and 4 nibble words (K16 step 2j: w0 w1, step 2j+1: w2 w3).
struct APair {
    unsigned int h, w0, w1, w2, w3;
};

__device__ __forceinline__ APair load_a(const char* __restrict__ rowp, int p) {
    const char* gp = rowp + (p >> 3) * 136;
    const int j = p & 7;
    APair a;
    a.h = *(const unsigned int*)(gp + (j >= 4 ? 4 : 0));
    const uint2 lo = *(const uint2*)(gp + 8 + j * 16);
    const uint2 hi = *(const uint2*)(gp + 8 + j * 16 + 8);
    a.w0 = lo.x;
    a.w1 = lo.y;
    a.w2 = hi.x;
    a.w3 = hi.y;
    return a;
}

// Keeps the compiler from folding the odd pair index into an `or` (the --emu lifter has no v_or_b32_e64).
__device__ __forceinline__ int opaque_i(int x) {
    asm volatile("" : "+s"(x));
    return x;
}

template <int R, int T>
struct PairState {
    APair a[R];
    half16_t b[T][2];
};

template <int R, int T, int PRIO>
__device__ __forceinline__ void consume(const APair* a, const half16_t (*b)[2], float8_t (*acc)[T]) {
    half16_t da[R][2];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const _Float16 sc_h = __builtin_bit_cast(_Float16, (unsigned short)(a[r].h & 0xFFFFu));
        const _Float16 zp_h = __builtin_bit_cast(_Float16, (unsigned short)(a[r].h >> 16));
        da[r][0] = dq_nibbles(a[r].w0, a[r].w1, sc_h, zp_h);
        da[r][1] = dq_nibbles(a[r].w2, a[r].w3, sc_h, zp_h);
    }
    if (PRIO) __builtin_amdgcn_s_setprio(1);
    // Independent accumulators interleaved; each accumulator keeps step 2j then step 2j+1.
#pragma unroll
    for (int s = 0; s < 2; s++)
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
            for (int t = 0; t < T; t++) acc[r][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(da[r][s], b[t][s], acc[r][t]);
    if (PRIO) __builtin_amdgcn_s_setprio(0);
}

template <int R, int T>
__device__ __forceinline__ void issue_global(PairState<R, T>& s, const char* const* arow, const _Float16* const* xrow, int p) {
#pragma unroll
    for (int r = 0; r < R; r++) s.a[r] = load_a(arow[r], p);
#pragma unroll
    for (int t = 0; t < T; t++) {
        s.b[t][0] = *(const half16_t*)(xrow[t] + 32 * p);
        s.b[t][1] = *(const half16_t*)(xrow[t] + 32 * p + 16);
    }
}

#define MOE_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")

// Computes R row tiles of this wave x T slot tiles; T slot tiles starting at slot_start belong to one expert A.
template <int R, int T, int W, int PRIO, int PF>
__device__ __forceinline__ void moe_run(
    const char* __restrict__ A, const int* __restrict__ sorted_slot_index, const _Float16* __restrict__ X_src,
    float* __restrict__ Y_grouped, int M, int K, int x_row_div, int m_total, int slot_start, _Float16* __restrict__ xs) {
    constexpr int XS = 72;  // halves per staged X row (64 + 8)
    const int tid = threadIdx.x;
    const int wt = tid & 31;
    const int lane = wt & 15;
    const int hi = wt >> 4;
    const int wave = tid >> 5;
    const int P = K >> 5;  // K32 pairs
    const unsigned long long row_bytes = (unsigned long long)(unsigned)(K >> 8) * 136u;
    const int rows_wave0 = opaque_i(blockIdx.x * (16 * R * W)) + wave * (16 * R);

    const char* arow[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const int row = min(rows_wave0 + 16 * r + lane, M - 1);
        arow[r] = A + (unsigned long long)(unsigned)row * row_bytes;
    }

    // Slot column of this lane per tile: X row (row 0 for padding, whose column is zeroed at the store).
    const _Float16* xrow[T];
    bool cvalid[T];
#pragma unroll
    for (int t = 0; t < T; t++) {
        const int slot_idx = slot_start + 16 * t + lane;
        int flat = -1;
        if (slot_idx < m_total) flat = sorted_slot_index[slot_idx];
        cvalid[t] = flat >= 0;
        const int x_row = cvalid[t] ? ((x_row_div > 1) ? (flat / x_row_div) : flat) : 0;
        xrow[t] = X_src + (unsigned long long)(unsigned)x_row * (unsigned)K;
    }

    float8_t acc[R][T];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int t = 0; t < T; t++) acc[r][t] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};

    if constexpr (W == 1) {
        PairState<R, T> s0, s1;
        if (PF) {
            issue_global<R, T>(s0, arow, xrow, 0);
            for (int p = 0; p < P; p += 2) {
                issue_global<R, T>(s1, arow, xrow, opaque_i(p + 1));
                consume<R, T, PRIO>(s0.a, s0.b, acc);
                issue_global<R, T>(s0, arow, xrow, opaque_i(min(p + 2, P - 1)));
                consume<R, T, PRIO>(s1.a, s1.b, acc);
            }
        } else {
            for (int p = 0; p < P; p++) {
                issue_global<R, T>(s0, arow, xrow, p);
                consume<R, T, PRIO>(s0.a, s0.b, acc);
            }
        }
    } else {
        // X of the T tiles, 64 K per chunk, shared by the W waves through LDS (double-buffered, one barrier per
        // chunk).  Staging 16 B pieces: piece p = slot (p >> 3) of the tile run, 8 halves (p & 7).
        constexpr int NTH = 32 * W;
        constexpr int PIECES = T * 16 * 8;
        constexpr int PER = PIECES / NTH;
        static_assert(PIECES % NTH == 0, "X pieces must divide evenly over the threads");
        constexpr int BUF = T * 16 * XS;  // halves per buffer
        // Explicit scalars (not arrays): PER pieces per thread, PER in {1, 2, 4}.
        static_assert(PER == 1 || PER == 2 || PER == 4, "PER");
        const _Float16* srcs[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int p = tid + NTH * (i < PER ? i : 0);
            const int slot_idx = slot_start + (p >> 3);
            int flat = -1;
            if (slot_idx < m_total) flat = sorted_slot_index[slot_idx];
            const int x_row = flat >= 0 ? ((x_row_div > 1) ? (flat / x_row_div) : flat) : 0;
            srcs[i] = X_src + (unsigned long long)(unsigned)x_row * (unsigned)K + (p & 7) * 8;
        }
        uint4 xp0, xp1, xp2, xp3;
#define MOE_LOAD_X(c)                                          \
    xp0 = *(const uint4*)(srcs[0] + (c) * 64);                 \
    if (PER > 1) xp1 = *(const uint4*)(srcs[1] + (c) * 64);    \
    if (PER > 2) {                                             \
        xp2 = *(const uint4*)(srcs[2] + (c) * 64);             \
        xp3 = *(const uint4*)(srcs[3] + (c) * 64);             \
    }
#define MOE_ST1(buf, i, xv)                                                                                \
    {                                                                                                      \
        const int p_ = tid + NTH * (i);                                                                    \
        *(uint4*)(xs + (buf) * BUF + (p_ >> 3) * XS + (p_ & 7) * 8) = (xv);                                \
    }
#define MOE_STORE_X(buf)                \
    MOE_ST1(buf, 0, xp0)                \
    if (PER > 1) MOE_ST1(buf, 1, xp1)   \
    if (PER > 2) {                      \
        MOE_ST1(buf, 2, xp2)            \
        MOE_ST1(buf, 3, xp3)            \
    }
        const int nchunks = K >> 6;
        MOE_LOAD_X(0);
        MOE_STORE_X(0);
        MOE_BARRIER();
        APair a0[R], a1[R];
#pragma unroll
        for (int r = 0; r < R; r++) a0[r] = load_a(arow[r], 0);
        for (int c = 0; c < nchunks; c++) {
            const int p = 2 * c;
            const bool more = c + 1 < nchunks;
            if (more) {
                MOE_LOAD_X(c + 1);
            }
            const _Float16* xb = xs + (c & 1) * BUF + lane * XS;
            half16_t b[T][2];
#pragma unroll
            for (int r = 0; r < R; r++) a1[r] = load_a(arow[r], opaque_i(p + 1));
            if (PF) {
#pragma unroll
                for (int t = 0; t < T; t++) {
                    b[t][0] = *(const half16_t*)(xb + t * 16 * XS);
                    b[t][1] = *(const half16_t*)(xb + t * 16 * XS + 16);
                }
                consume<R, T, PRIO>(a0, b, acc);
#pragma unroll
                for (int r = 0; r < R; r++) a0[r] = load_a(arow[r], opaque_i(min(p + 2, P - 1)));
#pragma unroll
                for (int t = 0; t < T; t++) {
                    b[t][0] = *(const half16_t*)(xb + t * 16 * XS + 32);
                    b[t][1] = *(const half16_t*)(xb + t * 16 * XS + 48);
                }
                consume<R, T, PRIO>(a1, b, acc);
            } else {
#pragma unroll
                for (int t = 0; t < T; t++) {
                    b[t][0] = *(const half16_t*)(xb + t * 16 * XS);
                    b[t][1] = *(const half16_t*)(xb + t * 16 * XS + 16);
                }
                consume<R, T, PRIO>(a0, b, acc);
#pragma unroll
                for (int t = 0; t < T; t++) {
                    b[t][0] = *(const half16_t*)(xb + t * 16 * XS + 32);
                    b[t][1] = *(const half16_t*)(xb + t * 16 * XS + 48);
                }
                consume<R, T, PRIO>(a1, b, acc);
#pragma unroll
                for (int r = 0; r < R; r++) a0[r] = load_a(arow[r], opaque_i(min(p + 2, P - 1)));
            }
#pragma unroll
            for (int r = 0; r < R; r++)
#pragma unroll
                for (int t = 0; t < T; t++) asm volatile("" : "+v"(acc[r][t]));
            if (more) {
                MOE_STORE_X((c + 1) & 1);
                MOE_BARRIER();  // buffer (c + 1) & 1 complete, and every wave is done with buffer c & 1
            }
        }
    }

    // Epilogue.  Lane l < 16 holds rows 2j of column l, lane l + 16 rows 2j + 1; one cross-half swap gives the low
    // lane rows 0..7 and the high lane rows 8..15 of its column, stored as two 16-byte F32 runs.  A padding slot's
    // column is exactly the shipped's all-zero accumulator.
#pragma unroll
    for (int r = 0; r < R; r++) {
        const int row_start = rows_wave0 + 16 * r;
#pragma unroll
        for (int t = 0; t < T; t++) {
            float v[8];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float give = hi ? acc[r][t][i] : acc[r][t][4 + i];
                const float got = __int_as_float(__builtin_amdgcn_permlanex16(
                    __float_as_int(give), __float_as_int(give), 0x76543210u, 0xfedcba98u, false, false));
                const float own = hi ? acc[r][t][4 + i] : acc[r][t][i];
                v[2 * i] = hi ? got : own;
                v[2 * i + 1] = hi ? own : got;
            }
            if (!cvalid[t]) {
#pragma unroll
                for (int i = 0; i < 8; i++) v[i] = 0.0f;
            }
            const int out_col = slot_start + 16 * t + lane;
            const int row0 = row_start + 8 * hi;
            if (row_start >= M || out_col >= m_total || row0 >= M) continue;
            const unsigned long long o = (unsigned long long)(unsigned)out_col * (unsigned)M + (unsigned)row0;
            // M % 16 == 0 (all Raon shapes), so a live row tile is complete: row0 + 8 <= M.
            *(float4*)(Y_grouped + o) = make_float4(v[0], v[1], v[2], v[3]);
            *(float4*)(Y_grouped + o + 4) = make_float4(v[4], v[5], v[6], v[7]);
        }
    }
}

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N x K], x_row_div = K_TOP -> x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots x K], x_row_div = 1 -> x_row = flat
//
// Grid [ceil(M / (16 * CFG_R * CFG_W)), ceil(m_total / 16)], block 32 * CFG_W.  With CFG_T == 2 the workgroup at an
// even expert-local tile index also computes the expert's next tile (when it has one) and the odd one returns.
__attribute__((amdgpu_waves_per_eu(CFG_WPE, 16)))
extern "C" __global__ void __launch_bounds__(32 * CFG_W) gemm_mq4g256v2_moe_grouped_wmma_k2_pm_gfx1100(
    const unsigned long long* __restrict__ expert_weight_ptrs,  // [E]
    const int* __restrict__ expert_tile_ids,                    // [m_total / 16]
    const int* __restrict__ sorted_slot_index,                  // [m_total]
    const _Float16* __restrict__ X_src,                         // [n_rows x K]
    float* __restrict__ Y_grouped,                              // [m_total x M]
    int M, int K, int x_row_div, int m_total, int n_experts) {
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;
    if (blockIdx.x * (16 * CFG_R * CFG_W) >= M || slot_start >= m_total) return;
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0 || expert_id >= n_experts) return;
    int run = 1;
    if (CFG_T == 2) {
        // Expert-local tile index parity: count the consecutive preceding tiles of this expert, four ids per step.
        int local = 0;
        for (;;) {
            int e[4];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const int idx = tile_y - local - 1 - k;
                e[k] = idx >= 0 ? expert_tile_ids[idx] : -2;
            }
            const bool m0 = e[0] == expert_id;
            const bool m1 = m0 && e[1] == expert_id;
            const bool m2 = m1 && e[2] == expert_id;
            const bool m3 = m2 && e[3] == expert_id;
            local += (int)m0 + (int)m1 + (int)m2 + (int)m3;
            if (!m3) break;
        }
        if (local & 1) return;
        if ((tile_y + 1) * 16 < m_total && expert_tile_ids[tile_y + 1] == expert_id) run = 2;
    }
    const char* A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);
#if CFG_W > 1
    __shared__ __align__(16) _Float16 xs[2 * 2 * 16 * 72];
    _Float16* xsp = xs;
#else
    _Float16* xsp = nullptr;
#endif
    if (run == 2)
        moe_run<CFG_R, (CFG_T == 2 ? 2 : 1), CFG_W, CFG_PRIO, CFG_PF>(A, sorted_slot_index, X_src, Y_grouped, M, K, x_row_div, m_total, slot_start, xsp);
    else
        moe_run<CFG_R, 1, CFG_W, CFG_PRIO, CFG_PF>(A, sorted_slot_index, X_src, Y_grouped, M, K, x_row_div, m_total, slot_start, xsp);
}
