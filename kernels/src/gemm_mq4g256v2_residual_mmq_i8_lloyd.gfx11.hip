// SPDX-License-Identifier: MIT
// hipfire — see LICENSE and NOTICE in the project root.

// MQ4G256V2L (qt=52) Lloyd A8 MMQ for gfx11 (RDNA3 — gfx1100 / gfx1151):
// the `gemm_mq4g256v2_residual_mmq_i8.gfx11.hip` schedule with a codebook.
// Differences from the uniform parent (LDS 36864, see planes below):
//   * W plane holds expanded i8 codes; each K64 slab expands its packed step
//     once at publish via v_perm from the C0..C3 (gate) / D0..D3 (up)
//     kernargs (amortized like the X128 pack stage, no K-loop LUT).
//   * B operand is SIGNED codes (iu8 MMA sign flag true), A stays signed int8.
//   * Chain seed is plain I8_MAGIC: Lloyd has no -8*s term. The DS seed half
//     is repurposed as the exact int code-sum s plane (same bytes).
//   * Fold is acc = fma(RN(sc' * d), C, acc) + zp' * d * s per K128 block,
//     with sc' = sc/16 folded at publish and zp' as stored (the loader centers
//     headers before upload: zp' = zp + 7.5*sc; synthetic harnesses bypassing
//     the loader must use the same headers on both arms under test).
//   * SZ plane carries (sc', zp') per row (2x f32). Tile128-only LDS 36864
//     (1 CTA/WGP).
// Headroom: |C| <= 128*120*127 = 1,950,720 < 2^22 exact window.
//
// Lloyd twin of `gemm_mq4g256v2_residual_mmq_i8.gfx11.hip` (residual SET/ADD;
// T3 adds gate/up): same argument lists plus four u32 codebook words
// (C0..C3, i8 codes packed LE), same `block_i8_128` activations, banded
// raster and the same pinned fold, so the output matches the CPU reference in
// `tmp_lloyd_i8_twin_t0.rs`:
//   C   = sum_k code[nibble] * x8  per (row, token, K128), integer-exact
//   acc = fma(RN(sc' * d), (float)C, acc + zp' * d * s) per K128, in K order
// where sc' = sc/16 and zp' is the header zp as stored (centered by the
// loader on real files).
//
// CTA tile 128x128 (weight rows x tokens), 16 waves, 512 threads.
//
// RDNA3 WMMA operand form: A and B are 16 int8 per lane (one K16 step),
// lanes r and r+16 carry identical operands, and C element l of lane L is
// C[2l + (L >> 4)][L & 15]. The kernel puts the ACTIVATIONS on A (C rows =
// tokens) and the expanded WEIGHTS on B (C columns = weight rows), so one
// epilogue store instruction writes two 64-byte runs of Y = [N][M].
//
// Wave tile 16*I8_NB tokens (nb, A) x 32 weight rows (rg = 0..1, B):
// 2*I8_NB int32x8 chains; per K16 step I8_NB + 2 b128 LDS reads feed
// 2*I8_NB WMMAs. I8_NB = 2.
//
// Expanded codes enter the WMMA SIGNED against signed int8 activations;
// there is no -8*s term: the chain starts at plain I8_MAGIC and the fold
// adds zp' * d * s, exact for |sum| <= 120 * 127 * 128 < 2^22.
//
// LDS (I8_LDS_BYTES): two slots of one K64 slab each,
//   A_s [I8_TILE tokens][64 B]: 16-byte chunk c of token e at chunk
//       c ^ ((e >> 1) & 3) (b128 reads and writes conflict-free per 8 lanes),
//   W_s [128 rows][64 B]: 16-byte expanded-code step c of row e at step
//       c ^ ((e >> 2) & 3) (b128 reads conflict-free per 8 lanes),
// then per-K128 parity planes
//   DS_p: s[I8_TILE] i32 (exact int code-sum) + d[I8_TILE] f32 in C-row order
//         p(t) = (t & ~15) | ((t & 1) << 3) | ((t >> 1) & 7),
//   SZ_p: (sc', zp')[I8_TILE] 2x f32 (folded scale + stored zero-point),
// (no pairtab: publish expands via v_perm).
// Every staging map has one thread per (token, 16-byte chunk) and per (row,
// 16-byte step) because 128 == threads / 4. Two K64 slabs per K128 run
// as a register-prefetched ring with one barrier per slab.
// Grid: [ceil(M/I8_TILE), ceil(N/I8_TILE)] (gate/up: [ceil(2M/I8_TILE), ...]).
// Block [threads,1,1]. Compile gate: gfx1100 / gfx1151 only.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef I8_TILE
#define I8_TILE 128
#endif
#if I8_TILE == 256
#define I8_ENTRY(name) name##_t256
#elif I8_TILE == 128
#define I8_ENTRY(name) name
#else
#error "I8_TILE must be 128 or 256"
#endif

#if defined(__gfx1100__) || defined(__gfx1151__)

// block_i8_128 comes from the host JIT prelude (block_i8_128_quant.hip).

#define I8_NB (I8_TILE / 64)
#define I8_NTHREADS (I8_TILE * 4)
#ifndef I8_RASTER_BAND
#define I8_RASTER_BAND 8
#endif
// VGPR budgets at granule 24 of 1536: tile 128 -> 120 = 12 waves/SIMD, three
// 16-wave CTAs per WGP (unconstrained the allocator takes 124: two CTAs);
// tile 256 -> 192 = 8 waves/SIMD, one 32-wave CTA per WGP.
#ifndef I8_NUM_VGPR
#define I8_NUM_VGPR (I8_TILE == 256 ? 192 : 120)
#endif

#define I8_A_SLAB_BYTES (I8_TILE * 64)
// Staged expansion: W plane holds expanded i8 codes
// (64 B/row, evens-then-odds B order), expanded once at publish via v_perm
// (amortized like the X128 pack stage) instead of per-K16-step in the K loop.
// K loop matches the uniform parent (b128 A/X/B loads + WMMAs + fold).
// No pairtab (v_perm needs no table). Tile128-only: 36864 B = 1 CTA/WGP.
#define I8_W_SLAB_BYTES (I8_TILE * 64)
#define I8_SLOT_BYTES (I8_A_SLAB_BYTES + I8_W_SLAB_BYTES)
#define I8_DS_OFF (2 * I8_SLOT_BYTES)
#define I8_DS_BYTES (I8_TILE * 8)
#define I8_SZ_OFF (I8_DS_OFF + 2 * I8_DS_BYTES)
#define I8_SZ_BYTES (I8_TILE * 8)
#define I8_LDS_BYTES (I8_SZ_OFF + 2 * I8_SZ_BYTES)
_Static_assert(I8_LDS_BYTES == 36864,
               "i8-lloyd staged planes must match the host LDS contract");
_Static_assert(I8_LDS_BYTES <= 65536, "RDNA3 workgroup LDS limit");
#define I8_MAGIC 0x4B400000

typedef int32_t __attribute__((ext_vector_type(4))) int32x4_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;
typedef unsigned __attribute__((ext_vector_type(2))) u32x2_t;
typedef unsigned __attribute__((ext_vector_type(4))) u32x4_t;

// v_perm LUT8 helpers (spread4+lut8 chain, cf. the X128 pack stage):
// spread 4 nibbles to byte lanes, then select signed code bytes.
static __device__ __forceinline__ unsigned twin_spread4(unsigned x16) {
    const unsigned a = (x16 & 0x00FFu) | ((x16 & 0xFF00u) << 8);
    return (a & 0x000F000Fu) | ((a & 0x00F000F0u) << 4);
}
static __device__ __forceinline__ unsigned twin_lut8(unsigned sel, unsigned C0, unsigned C1, unsigned C2, unsigned C3) {
    const unsigned lo = __builtin_amdgcn_perm(C1, C0, sel & 0x07070707u);
    const unsigned hi = __builtin_amdgcn_perm(C3, C2, sel & 0x07070707u);
    const unsigned m = ((sel >> 3) & 0x01010101u) * 0xFFu;
    return (hi & m) | (lo & ~m);
}
// Packed 8 bytes -> 16 signed code bytes in evens-then-odds B order
// (dword0 = lo nibbles, dword1 = hi nibbles), matching the A-side
// nibble-pair order. Shared by the K-loop v_perm path and publish-time
// staged expansion.
static __device__ __forceinline__ int32x4_t twin_expand_vperm(u32x2_t pk, unsigned Q0, unsigned Q1, unsigned Q2, unsigned Q3) {
    const unsigned r0 = twin_lut8(twin_spread4(pk.x & 0xFFFFu), Q0, Q1, Q2, Q3);
    const unsigned r1 = twin_lut8(twin_spread4(pk.x >> 16), Q0, Q1, Q2, Q3);
    const unsigned r2 = twin_lut8(twin_spread4(pk.y & 0xFFFFu), Q0, Q1, Q2, Q3);
    const unsigned r3 = twin_lut8(twin_spread4(pk.y >> 16), Q0, Q1, Q2, Q3);
    int32x4_t w;
    w.x = (int)__builtin_amdgcn_perm(r1, r0, 0x06040200u);
    w.y = (int)__builtin_amdgcn_perm(r1, r0, 0x07050301u);
    w.z = (int)__builtin_amdgcn_perm(r3, r2, 0x06040200u);
    w.w = (int)__builtin_amdgcn_perm(r3, r2, 0x07050301u);
    return w;
}

// Byte offset of 16-byte chunk c of entity e inside a [I8_TILE][64 B] plane.
static __device__ __forceinline__ unsigned i8_chunk(unsigned e, unsigned c) {
    return e * 64u + ((c ^ ((e >> 1) & 3u)) << 4);
}

// C-row order of the DS planes: tokens 2l + hi of a 16-token group sit at
// hi * 8 + l, so one lane's 8 C rows are 32 contiguous bytes.
static __device__ __forceinline__ unsigned i8_ds_slot(unsigned t) {
    return (t & ~15u) | ((t & 1u) << 3) | ((t >> 1) & 7u);
}

enum { I8_EPI_SET = 0, I8_EPI_ADD = 1, I8_EPI_SILU = 2 };
#define I8_SILU_MUL(g, u) ((g) / (1.0f + expf(-(g))) * (u))

template <int EPI>
static __device__ __forceinline__ void gemm_mq4g256v2_residual_mmq_i8_lloyd_gfx11_body(
    const char*           __restrict__ A,   // [M x groups_per_row x 136]
    const char*           __restrict__ U,   // SILU: W_up, same layout as A
    const block_i8_128*   __restrict__ Xq,  // [K/128 x N]
    float*                __restrict__ Y,   // [N x M] column-major
    int M, int K, int N,
    unsigned C0, unsigned C1, unsigned C2, unsigned C3,
    unsigned D0, unsigned D1, unsigned D2, unsigned D3
) {
    constexpr bool ADD = EPI == I8_EPI_ADD;
    constexpr bool SILU = EPI == I8_EPI_SILU;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    int rs;
    int bs;
    {
        // Banded raster (gfx12 A8 / K1): bijective for any grid.
        const int n_row_tiles = gridDim.x;
        const int lin = blockIdx.y * n_row_tiles + blockIdx.x;
        const int per_band = I8_RASTER_BAND * gridDim.y;
        const int band = lin / per_band;
        const int in_band = lin - band * per_band;
        const int band_row0 = band * I8_RASTER_BAND;
        const int band_rows = min(n_row_tiles - band_row0, I8_RASTER_BAND);
        rs = (band_row0 + in_band % band_rows) * I8_TILE;
        bs = (in_band / band_rows) * I8_TILE;
    }
    if (rs >= (SILU ? 2 * M : M) || bs >= N) return;
    const int hs = rs >> 1;  // SILU: first h row of the tile

    const int groups_per_row = K / 256;
    const int kblocks = K / 128;
    extern __shared__ char LDS[];

    // Wave tile: tokens wave_tok0 + 16 nb + .., weight rows wave_row0 + 16 rg + ..
    const unsigned l16 = (unsigned)lane & 15u;
    const unsigned hi = (unsigned)lane >> 4;
    const int wave_tok0 = (wave & 3) * (16 * I8_NB);
    const int wave_row0 = (wave >> 2) * 32;

    // Tile row R -> source row and matrix. SILU: 16-row groups alternate
    // gate (even) and up (odd) rows of the same h rows, so a wave's rg 0 /
    // rg 1 chains are gate / up of identical h rows.
    auto src_row = [&](int R) -> int {
        if constexpr (SILU) return hs + (R >> 5) * 16 + (R & 15);
        else return rs + R;
    };
    auto src_base = [&](int R) -> const char* {
        if constexpr (SILU) return ((R >> 4) & 1) ? U : A;
        else return A;
    };

    // --- Staging maps (loop-invariant). ---
    // A slab: thread stages token T = tid >> 2, chunk c = tid & 3.
    const unsigned st_T = (unsigned)(tid >> 2);
    const unsigned st_c = (unsigned)(tid & 3);
    const unsigned st_aoff = (unsigned)(bs + (int)st_T) * (unsigned)sizeof(block_i8_128) + 8u + st_c * 16u;
    const unsigned st_alds = i8_chunk(st_T, st_c);
    // W slab: row R = tid >> 2, K16 step c = tid & 3 (8 packed bytes). A
    // wave stages rows 8w..8w+7, one 16-row group: the gate/up selector is
    // wave-uniform, and readfirstlane keeps the descriptor in SGPRs (a VGPR
    // descriptor turns every load into a waterfall loop).
    const int st_wsrc = src_row((int)st_T);
    const unsigned st_woff =
        (unsigned)(st_wsrc < M ? st_wsrc : (M - 1)) * (unsigned)groups_per_row * 136u + 8u + st_c * 8u;
    // W plane (expanded codes): row R's 16 bytes of K16 step c at
    // R*64 + 16*(c ^ ((R >> 2) & 3)).
    const unsigned st_wlds = I8_A_SLAB_BYTES + st_T * 64u + ((st_c ^ ((st_T >> 2) & 3u)) << 4);
    const char* st_wbase = src_base(__builtin_amdgcn_readfirstlane((int)st_T));
    const auto w_rsrc = __builtin_amdgcn_make_buffer_rsrc((void*)st_wbase, 0, 0xffffffffu, 0x31004000u);
    // Headers: every thread fetches and publishes token (tid % I8_TILE)'s
    // (d, s) and row (tid % I8_TILE)'s scale (4 identical writers per
    // slot). A wave's 32 header rows span a gate and an up group: plain
    // per-lane loads.
    const int hdr_i = tid & (I8_TILE - 1);
    const unsigned hdr_ds = i8_ds_slot((unsigned)hdr_i);
    const int hdr_src = src_row(hdr_i);
    const char* hdr_sp = src_base(hdr_i)
        + (long long)(hdr_src < M ? hdr_src : (M - 1)) * groups_per_row * 136;
    const unsigned a_bytes = (unsigned)N * (unsigned)sizeof(block_i8_128);

    // Consumer lane bases. The swizzle terms (A: (e >> 1) & 3, W: (e >> 2) &
    // 3) depend on l16 only, so each operand is one lane base XOR the step's
    // chunk bits plus an immediate (16 nb tokens = 1024 B, 16 rg rows = 512 B).
    const unsigned rd_sw = ((l16 >> 1) & 3u) << 4;
    const unsigned rd_a = ((unsigned)wave_tok0 + l16) * 64u + rd_sw;
    const unsigned rd_w = I8_A_SLAB_BYTES + ((unsigned)wave_row0 + l16) * 64u + (((l16 >> 2) & 3u) << 4);

    float acc[I8_NB][2][8];
    #pragma unroll
    for (int nb = 0; nb < I8_NB; nb++)
        #pragma unroll
        for (int rg = 0; rg < 2; rg++)
            #pragma unroll
            for (int l = 0; l < 8; l++) acc[nb][rg][l] = 0.0f;
    int32x8_t cacc[I8_NB][2];

    u32x4_t a_pf;
    u32x2_t w_pf;
    u32x2_t hd_pf;
    unsigned hs_pf;

    // Global fetch of K64 slab (kb, s) into registers.
    auto fetch_slab = [&](int kb, int s) {
        const auto a_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)(Xq + (long long)kb * N), 0, a_bytes, 0x31004000u);
        a_pf = __builtin_amdgcn_raw_buffer_load_b128(a_rsrc, st_aoff, (unsigned)(s * 64), 0);
        w_pf = __builtin_amdgcn_raw_buffer_load_b64(
            w_rsrc, st_woff, (unsigned)((kb >> 1) * 136 + (kb & 1) * 64 + s * 32), 0);
    };
    auto publish_slab = [&](int slot) {
        char* base = LDS + slot * I8_SLOT_BYTES;
        *(u32x4_t*)(base + st_alds) = a_pf;
        // Staged expansion (amortized like the X128 pack stage): expand the
        // packed step to codes here, once per K64 slab, instead of per K16
        // step in the K loop. Each thread's row is wholly gate or up
        // (16-row groups), so the per-thread codebook select is exact.
        const bool w_is_up = SILU && ((((unsigned)st_T) >> 4) & 1u);
        const int32x4_t w_exp = w_is_up ? twin_expand_vperm(w_pf, D0, D1, D2, D3)
                                        : twin_expand_vperm(w_pf, C0, C1, C2, C3);
        *(int32x4_t*)(base + st_wlds) = w_exp;
    };
    // Header fetch / publish of K128 block kb, both in every thread with no
    // branch: a descriptor merged across a divergent branch lands in VGPRs
    // (waterfall loops), and loads consumed only inside one get sunk next to
    // their use with a full vmcnt(0) wait in front of the barrier.
    auto fetch_hdr = [&](int kb) {
        const auto d_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)(Xq + (long long)kb * N), 0, a_bytes, 0x31004000u);
        hd_pf = __builtin_amdgcn_raw_buffer_load_b64(
            d_rsrc, (unsigned)(bs + hdr_i) * (unsigned)sizeof(block_i8_128), 0u, 0);
        hs_pf = *(const unsigned*)(hdr_sp + (kb >> 1) * 136 + (kb & 1) * 4);
    };
    auto publish_hdr = [&](int par) {
        // Tokens >= N read d = 0, s = 0 (descriptor bound).
        // DS seed half repurposed: exact int code-sum s (the chain starts at
        // plain I8_MAGIC; Lloyd has no -8*s term). SZ carries (sc', zp').
        int* ss = (int*)(LDS + I8_DS_OFF + par * I8_DS_BYTES);
        float* dd = (float*)(LDS + I8_DS_OFF + par * I8_DS_BYTES + I8_TILE * 4);
        float* sz = (float*)(LDS + I8_SZ_OFF + par * I8_SZ_BYTES);
        ss[hdr_ds] = (int)hd_pf.y;
        dd[hdr_ds] = __uint_as_float(hd_pf.x);
        sz[2 * hdr_i] = __half2float(__ushort_as_half((unsigned short)(hs_pf & 0xFFFFu))) * 0.0625f;
        sz[2 * hdr_i + 1] = __half2float(__ushort_as_half((unsigned short)(hs_pf >> 16)));
    };

    // v_perm LUT8 helpers live at file scope (twin_spread4/twin_lut8/
    // twin_expand_vperm) for publish-time staged expansion.

    // One K64 slab of compute from `slot`: 4 K16 steps, each fenced so only
    // one step's operands are live. Unfenced, the scheduler hoists every
    // step's LDS reads to the slab start and the kernel spills. (Loading the
    // next step's operands ahead measured no faster at 12 waves/SIMD.)
    auto compute_slab = [&](int slot) {
        const char* base = LDS + slot * I8_SLOT_BYTES;
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int32x4_t af[I8_NB];
            #pragma unroll
            for (int nb = 0; nb < I8_NB; nb++)
                af[nb] = *(const int32x4_t*)(base + (rd_a ^ ((unsigned)t << 4)) + nb * 1024);
            // Staged: B operands are pre-expanded codes; plain b128 loads,
            // w1 = +16 rows (64 B/row). No K-loop expansion at all.
            const int32x4_t w0 = *(const int32x4_t*)(base + (rd_w ^ ((unsigned)t << 4)));
            const int32x4_t w1 = *(const int32x4_t*)(base + (rd_w ^ ((unsigned)t << 4)) + 1024);
            #pragma unroll
            for (int nb = 0; nb < I8_NB; nb++) {
                cacc[nb][0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[nb], true, w0, cacc[nb][0], false);
                cacc[nb][1] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[nb], true, w1, cacc[nb][1], false);
            }
            __builtin_amdgcn_sched_barrier(0);
        }
    };
    // WMMAs touch no memory: without this pin they sink past the next
    // barrier and every slab operand stays live across it.
    auto pin_chains = [&]() {
        #pragma unroll
        for (int nb = 0; nb < I8_NB; nb++)
            asm volatile("" : "+v"(cacc[nb][0]), "+v"(cacc[nb][1]));
    };

    // --- Prologue: headers of block 0 and slab (0, 0) into slot 0. ---
    // No pairtab (publish expands via v_perm instead).
    fetch_hdr(0);
    fetch_slab(0, 0);
    publish_hdr(0);
    publish_slab(0);
    __syncthreads();

    #pragma clang loop unroll(disable)
    for (int kb = 0; kb < kblocks; kb++) {
        const int par = kb & 1;
        // The last block refetches itself into dead planes (no reader after
        // the final barrier): the body stays one basic block.
        const int kn = (kb + 1 == kblocks) ? kb : kb + 1;
        // Chain init: plain I8_MAGIC splat (Lloyd has no zero-point seed).
        {
            const int32x8_t sv = {I8_MAGIC, I8_MAGIC, I8_MAGIC, I8_MAGIC,
                                  I8_MAGIC, I8_MAGIC, I8_MAGIC, I8_MAGIC};
            #pragma unroll
            for (int nb = 0; nb < I8_NB; nb++) {
                cacc[nb][0] = sv;
                cacc[nb][1] = sv;
            }
        }
        // Slab 0 (slot 0); prefetch slab 1 -> slot 1 and the next headers.
        fetch_slab(kb, 1);
        fetch_hdr(kn);
        compute_slab(0);
        pin_chains();
        publish_slab(1);
        publish_hdr(par ^ 1);
        __syncthreads();
        // Slab 1 (slot 1); prefetch the next block's slab 0 -> slot 0.
        fetch_slab(kn, 0);
        compute_slab(1);
        pin_chains();
        // Fold this K128 block: acc = fma(RN(sc' * d), as_float(C) - magic, acc)
        // + zp' * d * s (sc' = sc/16 folded at publish; zp' as stored).
        {
            const float* dd = (const float*)(LDS + I8_DS_OFF + par * I8_DS_BYTES + I8_TILE * 4)
                + wave_tok0 + hi * 8;
            const int* ss = (const int*)(LDS + I8_DS_OFF + par * I8_DS_BYTES)
                + wave_tok0 + hi * 8;
            const float* sz = (const float*)(LDS + I8_SZ_OFF + par * I8_SZ_BYTES)
                + 2 * (wave_row0 + l16);
            const float sc[2] = {sz[0], sz[32]};
            const float zp[2] = {sz[1], sz[33]};
            #pragma unroll
            for (int nb = 0; nb < I8_NB; nb++) {
                float d8[8];
                *(float4*)(d8) = *(const float4*)(dd + nb * 16);
                *(float4*)(d8 + 4) = *(const float4*)(dd + nb * 16 + 4);
                int s8[8];
                *(int4*)(s8) = *(const int4*)(ss + nb * 16);
                *(int4*)(s8 + 4) = *(const int4*)(ss + nb * 16 + 4);
                #pragma unroll
                for (int rg = 0; rg < 2; rg++) {
                    #pragma unroll
                    for (int l = 0; l < 8; l++) {
                        const float t1 = __fmul_rn(sc[rg], d8[l]);
                        // Scalar copy first: clang lowers __builtin_bit_cast
                        // of an ext-vector element lvalue to element 0.
                        const int c = cacc[nb][rg][l];
                        const float fc = __int_as_float(c) - 12582912.0f;
                        const float t2 = zp[rg] * d8[l] * (float)s8[l];
                        acc[nb][rg][l] = __builtin_fmaf(t1, fc, acc[nb][rg][l] + t2);
                    }
                }
            }
        }
        publish_slab(0);
        __syncthreads();
    }

    // --- Epilogue: lane holds token wave_tok0 + 16 nb + 2l + hi and row
    // wave_row0 + 16 rg + l16 (SILU: h row hs + 16 (wave >> 2) + l16). ---
    if constexpr (SILU) {
        const int hrow = hs + (wave >> 2) * 16 + (int)l16;
        if (hrow >= M) return;
        #pragma unroll
        for (int nb = 0; nb < I8_NB; nb++) {
            #pragma unroll
            for (int l = 0; l < 8; l++) {
                const int tok = bs + wave_tok0 + nb * 16 + 2 * l + (int)hi;
                if (tok < N) Y[(long long)tok * M + hrow] = I8_SILU_MUL(acc[nb][0][l], acc[nb][1][l]);
            }
        }
        return;
    }
    #pragma unroll
    for (int rg = 0; rg < 2; rg++) {
        const int row = rs + wave_row0 + rg * 16 + (int)l16;
        if (row >= M) continue;
        #pragma unroll
        for (int nb = 0; nb < I8_NB; nb++) {
            #pragma unroll
            for (int l = 0; l < 8; l++) {
                const int tok = bs + wave_tok0 + nb * 16 + 2 * l + (int)hi;
                if (tok < N) {
                    float* yp = Y + (long long)tok * M + row;
                    if constexpr (ADD) *yp += acc[nb][rg][l];
                    else *yp = acc[nb][rg][l];
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(I8_NTHREADS) __attribute__((amdgpu_num_vgpr(I8_NUM_VGPR)))
void I8_ENTRY(gemm_mq4g256v2_residual_mmq_i8_lloyd_full_add)(
    const char*           __restrict__ A,
    const block_i8_128*   __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int,
    unsigned C0, unsigned C1, unsigned C2, unsigned C3
) {
    gemm_mq4g256v2_residual_mmq_i8_lloyd_gfx11_body<I8_EPI_ADD>(
        A, nullptr, Xq, Y, M, K, N, C0, C1, C2, C3, C0, C1, C2, C3);
}

extern "C" __global__ __launch_bounds__(I8_NTHREADS) __attribute__((amdgpu_num_vgpr(I8_NUM_VGPR)))
void I8_ENTRY(gemm_mq4g256v2_residual_mmq_i8_lloyd_full_set)(
    const char*           __restrict__ A,
    const block_i8_128*   __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int,
    unsigned C0, unsigned C1, unsigned C2, unsigned C3
) {
    gemm_mq4g256v2_residual_mmq_i8_lloyd_gfx11_body<I8_EPI_SET>(
        A, nullptr, Xq, Y, M, K, N, C0, C1, C2, C3, C0, C1, C2, C3);
}

// Gate/up SwiGLU twin: G/U = gate/up MQ4V2L rows (M each), H = h [N][M].
// Two codebooks (gate C*, up D*); rg=0 rows read the gate table, rg=1 the up.
extern "C" __global__ __launch_bounds__(I8_NTHREADS) __attribute__((amdgpu_num_vgpr(I8_NUM_VGPR)))
void I8_ENTRY(gemm_mq4g256v2_gate_up_silu_mmq_i8_lloyd)(
    const char*           __restrict__ G,
    const char*           __restrict__ U,
    const block_i8_128*   __restrict__ Xq,
    float*                __restrict__ H,
    int M, int K, int N,
    unsigned C0, unsigned C1, unsigned C2, unsigned C3,
    unsigned D0, unsigned D1, unsigned D2, unsigned D3
) {
    gemm_mq4g256v2_residual_mmq_i8_lloyd_gfx11_body<I8_EPI_SILU>(
        G, U, Xq, H, M, K, N, C0, C1, C2, C3, D0, D1, D2, D3);
}

#undef I8_SILU_MUL

#else

extern "C" __global__ void I8_ENTRY(gemm_mq4g256v2_residual_mmq_i8_lloyd_full_add)(
    const char*, const void*, float*, int, int, int, int,
    unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned
) {}

extern "C" __global__ void I8_ENTRY(gemm_mq4g256v2_residual_mmq_i8_lloyd_full_set)(
    const char*, const void*, float*, int, int, int, int,
    unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned
) {}

extern "C" __global__ void I8_ENTRY(gemm_mq4g256v2_gate_up_silu_mmq_i8_lloyd)(
    const char*, const char*, const void*, float*, int, int, int,
    unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned
) {}

#endif

#undef I8_ENTRY
