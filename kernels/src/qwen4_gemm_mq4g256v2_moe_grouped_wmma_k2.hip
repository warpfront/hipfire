// Qwen4 copy of gemm_mq4g256v2_moe_grouped_wmma_k2.hip as #774 left it (paired-tile F32
// entry and the BF16-output entries on gfx1151). Qwen4's entries load from here, the shipped
// module on gfx1151 only; the original file keeps the shipped text elsewhere.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Grouped-WMMA-GEMM for MoE prefill over MQ4G256V2 (qt=44) routed experts.
//
// Transcribed from `gemm_hfq4g256_moe_grouped_wmma_k2` (qt=13). The gather /
// tiling / WMMA pipeline is identical, line for line. The ONLY difference is
// the group header and which affine grid each K-tile dequantizes against.
//
// Why this kernel has to exist: qt44 had GEMV (decode) and dense GEMM paths
// but no MoE grouped-expert path at all, so a qt44 A3B MoE model failed prefill
// with `unsupported moe.prefill-grouped-gemm-dtype`. Routed experts are ~99% of
// such a model's tensors, so without this the format cannot be used for MoE.
//
// Group layout, 136 B stride for BOTH formats:
//
//   qt13 MQ4G256      [0..4)  f32 scale (all 256 weights)
//                     [4..8)  f32 min   (all 256 weights)
//                     [8..136) 128 nibble bytes
//
//   qt44 MQ4G256V2    [0..2)  f16 scale, weights   0..127
//                     [2..4)  f16 zero,  weights   0..127
//                     [4..6)  f16 scale, weights 128..255
//                     [6..8)  f16 zero,  weights 128..255
//                     [8..136) 128 nibble bytes   (identical packing)
//
// Nibble packing is unchanged: byte i holds weight 2i in the low nibble and
// weight 2i+1 in the high nibble. The affine-grid boundary at weight 128
// therefore falls exactly on byte 64, so no byte straddles the two grids.
//
// K-tile → grid mapping: tile kt covers weights [kt*16, kt*16+16). Tiles 0-7
// are weights 0..127 (grid 0); tiles 8-15 are weights 128..255 (grid 1). The
// loop consumes tiles in pairs (kt, kt+1) with kt even, and every such pair
// lies wholly inside one grid — (6,7) → weights 96..127, (8,9) → 128..159.
// So the grid is selected once per pair, not per tile.
//
// Degenerate groups: the encoder writes scale bits 0 and all nibbles 0 when
// hi == lo, giving 0 * q + zero == zero. That is the intended constant-group
// value and needs no special case here.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef _Float16 __attribute__((ext_vector_type(2))) half2_p_t;
typedef unsigned int __attribute__((ext_vector_type(8))) uint8_p_t;
// Weights 2j, 2j+1 of nibble words (lo, hi) as sc*q + zp in F16.
__device__ __forceinline__ half16_t dq_nibbles(unsigned int lo, unsigned int hi,
                                              _Float16 sc, _Float16 zp) {
    // Byte p of a word holds weights 2p (low nibble) and 2p + 1 (high
    // nibble).  One byte permute copies byte p into bytes 0 and 2, one and-or
    // keeps the low nibble at bits 0-3 of half 0 (0x6400 | q is exactly
    // 1024 + q) and the high nibble at bits 4-7 of half 1 (0x5400 | q << 4 is
    // exactly 64 + q), so subtracting {1024, 64} yields both codes exactly and
    // one fused multiply-add per pair is the per-element sc * q + zp.
    const half2_p_t sc2 = {sc, sc};
    const half2_p_t zp2 = {zp, zp};
    const half2_p_t bias = {(_Float16)1024.0f, (_Float16)64.0f};
    uint8_p_t out;
#pragma unroll
    for (int p = 0; p < 8; p++) {
        const unsigned int w = p < 4 ? lo : hi;
        const unsigned int b = (unsigned)(p & 3);
        const unsigned int bits =
            (__builtin_amdgcn_perm(w, w, b | (12u << 8) | (b << 16) | (12u << 24)) & 0x00F0000Fu)
            | 0x54006400u;
        const half2_p_t q2 = __builtin_bit_cast(half2_p_t, bits) - bias;
        out[p] = __builtin_bit_cast(unsigned int, __builtin_elementwise_fma(sc2, q2, zp2));
    }
    return __builtin_bit_cast(half16_t, out);
}

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots × K], x_row_div = 1 → x_row = flat
//
// Tiles of one expert are consecutive in expert_tile_ids.  The wave at an
// expert-local tile index that is a multiple of four also computes the next
// one to three tiles of the same expert, sharing each A load and decode
// (prefill routing is skewed: hot experts own dozens of tiles); the other
// waves of the run exit.  Every tile keeps the single-tile WMMA sequence, so
// outputs are unchanged.  The K-tile loop is unrolled so later
// tile pairs' X loads issue ahead of the current decode/WMMA.
// F32 -> BF16 bits, round to nearest even; non-finite values truncate (the
// BF16 round trip leaves them unchanged).  Widening the bits back gives the
// value moe_bf16_round_trip / hip_bfloat16 rounding yields for finite input.
__device__ __forceinline__ unsigned short moe_bf16_bits(float v) {
    const unsigned int u = __float_as_uint(v);
    if (!isfinite(v)) return (unsigned short)(u >> 16);
    return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// One 256-weight group of the lane's row against the tiles' X, in the
// single-tile WMMA sequence.  `gw` holds the group's 34 words (header, 128
// nibble bytes), in LDS or global memory.
template <int NT>
__device__ __forceinline__ void moe_grouped_wmma_k2_group(
    const unsigned int* gw, const _Float16* const* x_base, int g, float8_t* acc) {
    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const unsigned int hdr_lo = gw[0];
    const unsigned int hdr_hi = gw[1];
    const _Float16 sc_0 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_lo & 0xFFFFu));
    const _Float16 zp_0 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_lo >> 16));
    const _Float16 sc_1 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_hi & 0xFFFFu));
    const _Float16 zp_1 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_hi >> 16));
    constexpr int KT_UNROLL = NT == 4 ? 2 : (NT == 2 ? 4 : 8);  // larger unrolls spill
#pragma unroll KT_UNROLL
    for (int kt = 0; kt < 16; kt += 2) {
        const int k_off_a = kt * 16;
        const int k_off_b = (kt + 1) * 16;
        const bool upper = (k_off_a >= 128);
        const _Float16 sc_h = upper ? sc_1 : sc_0;
        const _Float16 zp_h = upper ? zp_1 : zp_0;
        const half16_t a_a = dq_nibbles(gw[2 + 2 * kt], gw[3 + 2 * kt], sc_h, zp_h);
        const half16_t a_b = dq_nibbles(gw[4 + 2 * kt], gw[5 + 2 * kt], sc_h, zp_h);
#pragma unroll
        for (int t = 0; t < NT; t++) {
            half16_t b_a = zero_b, b_b = zero_b;
            if (x_base[t] != nullptr) {
                b_a = *(const half16_t*)(x_base[t] + g * 256 + k_off_a);
                b_b = *(const half16_t*)(x_base[t] + g * 256 + k_off_b);
            }
            acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc[t]);
            acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_b, b_b, acc[t]);
        }
    }
}

// SILU: the 16 A rows are gate rows row_start/2 .. +7 and the matching up
// rows M/2 + row_start/2 .. +7, and the epilogue stores the SwiGLU activation
// rt(silu(bf16(g)) * bf16(u)) (moe_gate_up_unscatter_silu's expression with
// its BF16 round trips) as BF16 into Y_grouped, [m_total, M/2].
static __device__ __forceinline__ float moe_bf16_round_trip(float v) {
    if (!isfinite(v)) return v;
    const unsigned int u = __float_as_uint(v);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

template <int NT, bool OUT_BF16, bool SILU>
__device__ __forceinline__ void moe_grouped_wmma_k2_store(
    const float8_t* acc, void* __restrict__ Y_grouped, int M, int m_total, int tid,
    int row_start, int slot_start, int nlive);

template <int NT, bool OUT_BF16, bool SILU = false>
__device__ __forceinline__ void moe_grouped_wmma_k2_tiles(
    const char* __restrict__ A, const int* __restrict__ sorted_slot_index,
    const _Float16* __restrict__ X_src, void* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total, int tid, int row_start, int slot_start,
    unsigned int* __restrict__ a_lds, int nlive) {
    const int lane = tid & 15;
    const _Float16* x_base[NT] = {};
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const int slot_idx = slot_start + 16 * t + lane;
        if (t < nlive && slot_idx < m_total) {
            const int flat = sorted_slot_index[slot_idx];
            if (flat >= 0)
                x_base[t] = X_src + (long long)((x_row_div > 1) ? (flat / x_row_div) : flat) * K;
        }
    }
    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 136;
    float8_t acc[NT];
    for (int t = 0; t < NT; t++) acc[t] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    // One 136-byte group of the 16 rows at a time (272 8-byte pieces,
    // nine per lane, the last clamped so no lane branches), double-
    // buffered in LDS: group g+1 loads while group g computes.  Loading
    // 16 bytes of 16 rows per instruction ran DRAM ~25% under streaming
    // rate; group-pair steps overlapped less of each wave's own stream.
    const int rows_live = min(16, M - row_start);
    // Weight row of tile row r (SILU pairs gate and up rows, see above).
    auto weight_row = [&](int r) -> long long {
        if (SILU)
            return r < 8 ? row_start / 2 + r : M / 2 + row_start / 2 + r - 8;
        return row_start + min(r, rows_live - 1);
    };
    unsigned long long pf[9];
    auto load_group = [&](int g) {
#pragma unroll
        for (int i = 0; i < 9; i++) {
            const int p = min(tid + 32 * i, 16 * 17 - 1);
            const int r = p / 17, w = p - r * 17;
            pf[i] = *(const unsigned long long*)(A + weight_row(r) * row_bytes + g * 136 + w * 8);
        }
    };
    auto store_group = [&](int buf) {
#pragma unroll
        for (int i = 0; i < 9; i++) {
            const int p = min(tid + 32 * i, 16 * 17 - 1);
            *(unsigned long long*)(a_lds + buf * 16 * 34 + p * 2) = pf[i];
        }
    };
    load_group(0);
    store_group(0);
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    for (int g = 0; g < groups_per_row; g++) {
        if (g + 1 < groups_per_row) load_group(g + 1);
        moe_grouped_wmma_k2_group<NT>(a_lds + (g & 1) * 16 * 34 + lane * 34, x_base, g, acc);
        if (g + 1 < groups_per_row) {
            store_group((g + 1) & 1);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        }
    }
    moe_grouped_wmma_k2_store<NT, OUT_BF16, SILU>(acc, Y_grouped, M, m_total, tid, row_start, slot_start, nlive);
}

// Lane l < 16 holds rows 2j of column l, lane l + 16 rows 2j + 1.  One
// cross-half swap gives the low lane rows 0..7 and the high lane rows 8..15
// of its column, stored as one 16-byte BF16 run (or two 16-byte F32 runs)
// instead of eight scattered 2-byte stores.  `tid` is the lane in the wave.
template <int NT, bool OUT_BF16, bool SILU>
__device__ __forceinline__ void moe_grouped_wmma_k2_store(
    const float8_t* acc, void* __restrict__ Y_grouped, int M, int m_total, int tid,
    int row_start, int slot_start, int nlive) {
    const int lane = tid & 15;
    const int hi = tid >> 4;
#pragma unroll
    for (int t = 0; t < NT; t++) {
        float r[8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float give = hi ? acc[t][i] : acc[t][4 + i];
            const float got = __int_as_float(__builtin_amdgcn_permlanex16(
                __float_as_int(give), __float_as_int(give), 0x76543210u, 0xfedcba98u, false, false));
            const float own = hi ? acc[t][4 + i] : acc[t][i];
            r[2 * i] = hi ? got : own;
            r[2 * i + 1] = hi ? own : got;
        }
        const int out_col = slot_start + 16 * t + lane;
        if (SILU) {
            // Low lane: gate columns 0..7 of the block; high lane: up.  One
            // more swap gives each half four (gate, up) pairs.
            float h[4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float give = hi ? r[i] : r[4 + i];
                const float got = __int_as_float(__builtin_amdgcn_permlanex16(
                    __float_as_int(give), __float_as_int(give), 0x76543210u, 0xfedcba98u, false, false));
                const float g = hi ? got : r[i];
                const float u = hi ? r[4 + i] : got;
                const float v = __uint_as_float((unsigned int)moe_bf16_bits(g) << 16);
                const float w = __uint_as_float((unsigned int)moe_bf16_bits(u) << 16);
                h[i] = moe_bf16_round_trip((v / (1.0f + expf(-v))) * w);
            }
            if (t >= nlive || out_col >= m_total) continue;
            uint2 o;
            o.x = (__float_as_uint(h[0]) >> 16) | (__float_as_uint(h[1]) & 0xFFFF0000u);
            o.y = (__float_as_uint(h[2]) >> 16) | (__float_as_uint(h[3]) & 0xFFFF0000u);
            *(uint2*)((unsigned short*)Y_grouped + (long long)out_col * (M / 2) + row_start / 2 + 4 * hi) = o;
            continue;
        }
        if (t >= nlive) continue;
        const int row0 = row_start + 8 * hi;
        if (out_col >= m_total || row0 >= M) continue;
        const long long o = (long long)out_col * M + row0;
        if (row0 + 8 > M) {
            for (int i = 0; row0 + i < M; i++) {
                if (OUT_BF16)
                    ((unsigned short*)Y_grouped)[o + i] = moe_bf16_bits(r[i]);
                else
                    ((float*)Y_grouped)[o + i] = r[i];
            }
        } else {
            if (OUT_BF16) {
                uint4 w;
                w.x = moe_bf16_bits(r[0]) | ((unsigned)moe_bf16_bits(r[1]) << 16);
                w.y = moe_bf16_bits(r[2]) | ((unsigned)moe_bf16_bits(r[3]) << 16);
                w.z = moe_bf16_bits(r[4]) | ((unsigned)moe_bf16_bits(r[5]) << 16);
                w.w = moe_bf16_bits(r[6]) | ((unsigned)moe_bf16_bits(r[7]) << 16);
                *(uint4*)((unsigned short*)Y_grouped + o) = w;
            } else {
                *(float4*)((float*)Y_grouped + o) = make_float4(r[0], r[1], r[2], r[3]);
                *(float4*)((float*)Y_grouped + o + 4) = make_float4(r[4], r[5], r[6], r[7]);
            }
        }
    }
}

template <bool OUT_BF16, bool SILU = false>
__device__ __forceinline__ void gemm_mq4g256v2_moe_grouped_wmma_k2_entry(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index,
    const _Float16* __restrict__ X_src,
    void* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;
    if (row_start >= M || slot_start >= m_total) return;
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;
    int local = 0;
    while (tile_y - local - 1 >= 0 && expert_tile_ids[tile_y - local - 1] == expert_id) ++local;
    if (local % 4) return;
    int run = 1;
    while (run < 4 && (tile_y + run) * 16 < m_total && expert_tile_ids[tile_y + run] == expert_id) ++run;
    const char* A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);
    // Declared here, not in the NT-templated body: one allocation, not three.
    __shared__ __align__(16) unsigned int a_lds[16 * 68];
    // A three-tile run takes the four-tile body with its last tile masked.
    if (run >= 3)
        moe_grouped_wmma_k2_tiles<4, OUT_BF16, SILU>(A, sorted_slot_index, X_src, Y_grouped, M, K, x_row_div, m_total, tid, row_start, slot_start, a_lds, run);
    else if (run == 2)
        moe_grouped_wmma_k2_tiles<2, OUT_BF16, SILU>(A, sorted_slot_index, X_src, Y_grouped, M, K, x_row_div, m_total, tid, row_start, slot_start, a_lds, 2);
    else
        moe_grouped_wmma_k2_tiles<1, OUT_BF16, SILU>(A, sorted_slot_index, X_src, Y_grouped, M, K, x_row_div, m_total, tid, row_start, slot_start, a_lds, 1);
}

#define MOE_WMMA_ARGS \
    const unsigned long long* __restrict__ expert_weight_ptrs, \
    const int* __restrict__ expert_tile_ids, const int* __restrict__ sorted_slot_index, \
    const _Float16* __restrict__ X_src, void* __restrict__ Y_grouped, \
    int M, int K, int x_row_div, int m_total
// The F32-output entry is the one the shipped Qwen3.5-family MoE prefill runs
// on gfx11. gfx1151 takes the multi-tile LDS-staged body above (bitwise equal
// per the raw-bit parity arms); every other gfx11 arch keeps the original
// single-tile kernel verbatim, so its code is unchanged there.
#if defined(__gfx1151__)
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g256v2_moe_grouped_wmma_k2(MOE_WMMA_ARGS) {
    gemm_mq4g256v2_moe_grouped_wmma_k2_entry<false>(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, X_src,
                         Y_grouped, M, K, x_row_div, m_total);
}
#else
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g256v2_moe_grouped_wmma_k2(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E]
    const int*                __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const _Float16*           __restrict__ X_src,              // [n_rows × K]
    float*                    __restrict__ Y_grouped,          // [m_total × M]
    int M, int K, int x_row_div, int m_total
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int lane = tid & 15;
    const int slot_idx = slot_start + lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base = (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;

        // Dual affine grid. Stored as raw f16 bits, unlike qt13's f32 pair.
        const unsigned int hdr_lo = *(const unsigned int*)(gp);
        const unsigned int hdr_hi = *(const unsigned int*)(gp + 4);
        const _Float16 sc_0 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_lo & 0xFFFFu));
        const _Float16 zp_0 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_lo >> 16));
        const _Float16 sc_1 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_hi & 0xFFFFu));
        const _Float16 zp_1 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_hi >> 16));

        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 256) : nullptr;

        // 8 iterations × 2 K-tiles each = 16 K-tiles per group.
        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;

            // Both tiles of this pair lie in the same affine grid — see the
            // K-tile mapping note above.
            const bool upper = (k_off_a >= 128);
            const _Float16 sc_h = upper ? sc_1 : sc_0;
            const _Float16 zp_h = upper ? zp_1 : zp_0;

            unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
            unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);
            unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);
            unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);

            half16_t b_a, b_b;
            if (xg != nullptr) {
                b_a = *(const half16_t*)(xg + k_off_a);
                b_b = *(const half16_t*)(xg + k_off_b);
            } else {
                b_a = zero_b;
                b_b = zero_b;
            }

            half16_t a_a;
            #define DQ(reg, i, pk, sh) reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(a_a, 0,  pk0a,  0); DQ(a_a, 1,  pk0a,  4); DQ(a_a, 2,  pk0a,  8); DQ(a_a, 3,  pk0a, 12);
            DQ(a_a, 4,  pk0a, 16); DQ(a_a, 5,  pk0a, 20); DQ(a_a, 6,  pk0a, 24); DQ(a_a, 7,  pk0a, 28);
            DQ(a_a, 8,  pk1a,  0); DQ(a_a, 9,  pk1a,  4); DQ(a_a, 10, pk1a,  8); DQ(a_a, 11, pk1a, 12);
            DQ(a_a, 12, pk1a, 16); DQ(a_a, 13, pk1a, 20); DQ(a_a, 14, pk1a, 24); DQ(a_a, 15, pk1a, 28);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);

            DQ(a_a, 0,  pk0b,  0); DQ(a_a, 1,  pk0b,  4); DQ(a_a, 2,  pk0b,  8); DQ(a_a, 3,  pk0b, 12);
            DQ(a_a, 4,  pk0b, 16); DQ(a_a, 5,  pk0b, 20); DQ(a_a, 6,  pk0b, 24); DQ(a_a, 7,  pk0b, 28);
            DQ(a_a, 8,  pk1b,  0); DQ(a_a, 9,  pk1b,  4); DQ(a_a, 10, pk1b,  8); DQ(a_a, 11, pk1b, 12);
            DQ(a_a, 12, pk1b, 16); DQ(a_a, 13, pk1b, 20); DQ(a_a, 14, pk1b, 24); DQ(a_a, 15, pk1b, 28);
            #undef DQ

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
        }
    }

    const int out_col = slot_start + lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
#endif
// Y_grouped as BF16 (RNE) for consumers that round each value to BF16 first.
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g256v2_moe_grouped_wmma_k2_bf16out(MOE_WMMA_ARGS) {
    gemm_mq4g256v2_moe_grouped_wmma_k2_entry<true>(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, X_src,
                        Y_grouped, M, K, x_row_div, m_total);
}
// Y_grouped = the SwiGLU activation [m_total, M/2] as BF16 (see SILU above).
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out(MOE_WMMA_ARGS) {
    gemm_mq4g256v2_moe_grouped_wmma_k2_entry<true, true>(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, X_src,
                        Y_grouped, M, K, x_row_div, m_total);
}
#undef MOE_WMMA_ARGS

#if defined(__gfx1151__)
// Four row-tile waves (64 A rows) of one expert run share its X through
// LDS, 64 K at a time: the per-wave entries re-read every tile's X from L2 in
// each of M/16 waves.  Each wave stages its own 16 rows' current group
// (single-buffered: it is only overwritten after the barrier that ends the
// group).  The next chunk's X (and at a group boundary the next group's
// weights) are loaded before the current chunk computes from LDS, so the
// compute waits on no global load (vmcnt retires in order).  Every tile's
// WMMA sequence, operands and epilogue are moe_grouped_wmma_k2_tiles', so
// outputs are bitwise identical.
// Grid [M / 64, m_total / 16], block 128.  M % 64 == 0, K % 256 == 0.
template <int NT, bool OUT_BF16, bool SILU>
__device__ __forceinline__ void moe_grouped_wmma_k2_tiles_x4(
    const char* __restrict__ A, const int* __restrict__ sorted_slot_index,
    const _Float16* __restrict__ X_src, void* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total, int slot_start,
    unsigned int* __restrict__ a_lds, _Float16* __restrict__ xs, int nlive) {
    constexpr int WAVES = 4;
    constexpr int XS = 72;                   // halves per staged slot row (64 + 8)
    constexpr int NTH = 32 * WAVES;
    constexpr int PIECES = NT * 16 * 8;      // 16 B pieces of a 64-K chunk's X
    constexpr int PER = (PIECES + NTH - 1) / NTH;
    const int tid = threadIdx.x;
    const int wt = tid & 31;
    const int lane = wt & 15;
    const int row_start = (blockIdx.x * WAVES + (tid >> 5)) * 16;
    const _Float16* src[PER];
#pragma unroll
    for (int i = 0; i < PER; i++) {
        const int p = tid + NTH * i;
        const int s = p >> 3;
        const int slot_idx = slot_start + s;
        src[i] = nullptr;
        if (p < PIECES && (s >> 4) < nlive && slot_idx < m_total) {
            const int flat = sorted_slot_index[slot_idx];
            if (flat >= 0)
                src[i] = X_src + (long long)((x_row_div > 1) ? (flat / x_row_div) : flat) * K + (p & 7) * 8;
        }
    }
    uint4 xp[PER];
    auto load_x = [&](int c) {
#pragma unroll
        for (int i = 0; i < PER; i++)
            xp[i] = src[i] ? *(const uint4*)(src[i] + c * 64) : make_uint4(0, 0, 0, 0);
    };
    auto store_x = [&]() {
#pragma unroll
        for (int i = 0; i < PER; i++) {
            const int p = tid + NTH * i;
            if (p < PIECES) *(uint4*)(xs + (p >> 3) * XS + (p & 7) * 8) = xp[i];
        }
    };
    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 136;
    float8_t acc[NT];
    for (int t = 0; t < NT; t++) acc[t] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    const int rows_live = min(16, M - row_start);
    auto weight_row = [&](int r) -> long long {
        if (SILU)
            return r < 8 ? row_start / 2 + r : M / 2 + row_start / 2 + r - 8;
        return row_start + min(r, rows_live - 1);
    };
    unsigned long long pf[9];
    auto load_group = [&](int g) {
#pragma unroll
        for (int i = 0; i < 9; i++) {
            const int p = min(wt + 32 * i, 16 * 17 - 1);
            const int r = p / 17, w = p - r * 17;
            pf[i] = *(const unsigned long long*)(A + weight_row(r) * row_bytes + g * 136 + w * 8);
        }
    };
    auto store_group = [&]() {
#pragma unroll
        for (int i = 0; i < 9; i++) {
            const int p = min(wt + 32 * i, 16 * 17 - 1);
            *(unsigned long long*)(a_lds + p * 2) = pf[i];
        }
    };
    // LDS-only barriers: __syncthreads would also wait for the in-flight
    // next-chunk global loads (WGP-mode workgroup fence).
#define MOE_X4_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
    load_group(0);
    load_x(0);
    store_group();
    store_x();
    MOE_X4_BARRIER();
    const unsigned int* gw = a_lds + lane * 34;
    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const int nchunks = K / 64;
    for (int c = 0; c < nchunks; c++) {
        const int cq = c & 3;
        if (c + 1 < nchunks) {
            if (cq == 3) load_group((c >> 2) + 1);
            load_x(c + 1);
        }
        const unsigned int hdr = gw[cq >= 2 ? 1 : 0];
        const _Float16 sc_h = __builtin_bit_cast(_Float16, (unsigned short)(hdr & 0xFFFFu));
        const _Float16 zp_h = __builtin_bit_cast(_Float16, (unsigned short)(hdr >> 16));
#pragma unroll
        for (int kq = 0; kq < 4; kq += 2) {
            const int kt = cq * 4 + kq;
            const half16_t a_a = dq_nibbles(gw[2 + 2 * kt], gw[3 + 2 * kt], sc_h, zp_h);
            const half16_t a_b = dq_nibbles(gw[4 + 2 * kt], gw[5 + 2 * kt], sc_h, zp_h);
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const _Float16* xr = xs + (t * 16 + lane) * XS;
                const half16_t b_a = t < nlive ? *(const half16_t*)(xr + kq * 16) : zero_b;
                const half16_t b_b = t < nlive ? *(const half16_t*)(xr + (kq + 1) * 16) : zero_b;
                acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc[t]);
                acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_b, b_b, acc[t]);
            }
        }
        // WMMA has no memory chain: keep the WMMAs ahead of the barrier.
#pragma unroll
        for (int t = 0; t < NT; t++) asm volatile("" : "+v"(acc[t]));
        if (c + 1 < nchunks) {
            MOE_X4_BARRIER();  // every wave is done reading this chunk's X (and group)
            if (cq == 3) store_group();
            store_x();
            MOE_X4_BARRIER();
        }
    }
#undef MOE_X4_BARRIER
    moe_grouped_wmma_k2_store<NT, OUT_BF16, SILU>(acc, Y_grouped, M, m_total, wt, row_start, slot_start, nlive);
}

// gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out with four row tiles per
// workgroup: multi-tile runs share X (see above); single-tile runs, whose
// X is one tile, keep the per-wave body (no barriers).  Both are bitwise
// that entry's.  Grid [M / 64, m_total / 16], block 128.
__launch_bounds__(128, 1)
extern "C" __global__ void gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out_x4(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int* __restrict__ expert_tile_ids, const int* __restrict__ sorted_slot_index,
    const _Float16* __restrict__ X_src, void* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total) {
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;
    if (slot_start >= m_total) return;
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;
    int local = 0;
    while (tile_y - local - 1 >= 0 && expert_tile_ids[tile_y - local - 1] == expert_id) ++local;
    if (local % 4) return;
    int run = 1;
    while (run < 4 && (tile_y + run) * 16 < m_total && expert_tile_ids[tile_y + run] == expert_id) ++run;
    const char* A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);
    // One LDS arena: the shared-X body takes four single-buffered weight
    // slots plus the X chunk; the per-wave body two slots per wave.
    __shared__ __align__(16) unsigned int lds[4 * 16 * 34 + 4 * 16 * 72 / 2];
    const int wave = threadIdx.x >> 5;
    if (run == 1) {
        moe_grouped_wmma_k2_tiles<1, true, true>(A, sorted_slot_index, X_src, Y_grouped, M, K, x_row_div, m_total,
                                                 threadIdx.x & 31, (blockIdx.x * 4 + wave) * 16, slot_start,
                                                 lds + wave * 2 * 16 * 34, 1);
        return;
    }
    unsigned int* my_lds = lds + wave * 16 * 34;
    _Float16* xs = (_Float16*)(lds + 4 * 16 * 34);
    if (run == 4)
        moe_grouped_wmma_k2_tiles_x4<4, true, true>(A, sorted_slot_index, X_src, Y_grouped, M, K, x_row_div, m_total, slot_start, my_lds, xs, 4);
    else if (run == 3)
        moe_grouped_wmma_k2_tiles_x4<3, true, true>(A, sorted_slot_index, X_src, Y_grouped, M, K, x_row_div, m_total, slot_start, my_lds, xs, 3);
    else
        moe_grouped_wmma_k2_tiles_x4<2, true, true>(A, sorted_slot_index, X_src, Y_grouped, M, K, x_row_div, m_total, slot_start, my_lds, xs, 2);
}
#endif
