// SPDX-License-Identifier: MIT
// hipfire — see LICENSE and NOTICE in the project root.

// MQ6G256V2 exact-INT8 prefill GEMM for gfx11 (RDNA3 — gfx1100 / gfx1151):
// int8 per-128-K activations (`block_i8_128`) x raw MQ6V2 weight codes on
// `v_wmma_i32_16x16x16_iu8` (wave32). Fork of
// `gemm_mq4g256v2_residual_mmq_i8.gfx11.hip` (same tile / wave / ring /
// raster design); see INT8-SPEC.md for the contract.
//
// Math. Per (weight row r, token t, K128 block kb), half h = kb & 1 of the
// K256 group:
//   C   = sum_k (q_k - 32) * qx_k           (q = raw 6-bit code, 0..63)
//   |C| <= 32 * 127 * 128 = 520,192 < 2^22  -> the f32 view of the int32
//   chain, as_float(0x4B400000 + C) - 12582912.0f, is exact.
// The -32 * s term rides in the chain seed: seed = I8_MAGIC - 32 * s (s =
// sum of the token's 128 int8 values, exact int from the block header), so
// the WMMA chain ends at I8_MAGIC + C with raw UNSIGNED codes on B and
// signed activations on A (same signedness flags as the MQ4 kernel).
// Fold, per K128 block in K order (output y = acc, no extra scale):
//   acc = fma(RN(sc * d),  as_float(C) - 12582912.0f, acc)
//   acc = fma(RN(z' * d),  (float)s,                  acc)
//   z'  = z + 32 * sc  (f32, from the f16 header: w = sc*q + z)
//
// Weights: MQ6G256V2 group = 200 B: u32 {s0 | z0 << 16} (codes 0..127),
// u32 {s1 | z1 << 16} (codes 128..255), 192 B little-endian 6-bit
// bitstream (code c = bits [6c, 6c+6)). K64 slab s of K128 half h of a row
// is 48 contiguous bytes at 8 + 96 h + 48 s; K16 step c of the slab is the
// 12 bytes (96 bits, 16 codes) at +12 c.
//
// RDNA3 WMMA operand form: A and B are 16 int8 per lane (one K16 step),
// lanes r and r+16 carry identical operands, and C element l of lane L is
// C[2l + (L >> 4)][L & 15]. ACTIVATIONS on A (C rows = tokens), expanded
// WEIGHTS on B (C columns = weight rows), so one epilogue store writes runs
// of Y = [N][M].
//
// B operand byte order (the activation nibble-pair order of block_i8_128):
//   {c0,c2,c4,c6 | c1,c3,c5,c7 | c8,c10,c12,c14 | c9,c11,c13,c15}
// as four dwords w0..w3, one byte per code, value 0..63.
//
// LDS weight format (repacked at staging, once per block; still 12 B = 3
// dwords = 96 bits per row per K16 step): s0 = w0 | ((w3 << 6) & 0xC0C0C0C0),
// s1 = w1 | ((w3 << 4) & 0xC0C0C0C0), s2 = w2 | ((w3 << 2) & 0xC0C0C0C0).
// Expansion after the LDS read, per row per K16 step (9 VALU):
//   w0 = s0 & 0x3F3F3F3F, w1 = s1 & ..., w2 = s2 & ...,
//   w3 = ((s0 >> 6) & 0x03030303) | ((s1 >> 4) & 0x0C0C0C0C) | ((s2 >> 2) & 0x30303030).
// So the K loop has no byte shuffle, only masks and a 3-way merge.
//
// Wave tile: 16*I8_NB tokens (nb, A) x 32 weight rows (rg = 0..1, B); per
// K16 step I8_NB b128 A reads + 6 b32 W reads feed 2*I8_NB WMMAs.
// I8_NB = 2 (tile 128) or 4 (tile 256).
//
// LDS map (MQ6_I8_GFX11_LDS_BYTES): two slots of one K64 slab each,
//   A_s [I8_TILE tokens][64 B]: 16-byte chunk c of token e at chunk
//       c ^ ((e >> 1) & 3) (as the MQ4 kernel),
//   W_s 12 dword planes of I8_TILE words, plane p = 3 c + j (K16 step c of
//       the slab, dword j of its 12 bytes): row R at word R ^ (8 c). Reads
//       (16 consecutive rows per plane) and the staging writes (8 rows x 4
//       steps per wave) are both conflict-free,
// then per-K128 parity planes
//   DS_p: s[I8_TILE] i32 + d[I8_TILE] f32 in the C-row order
//         p(t) = (t & ~15) | ((t & 1) << 3) | ((t >> 1) & 7),
//   SZ_p: sc[I8_TILE] f32 + z'[I8_TILE] f32 (row order).
// tile 128: 2*(8192 + 6144) + 2*1024 + 2*1024 = 32768 B,
// tile 256: 2*(16384 + 12288) + 2*2048 + 2*2048 = 65536 B (the full
// RDNA3 64 KB workgroup LDS). Static-asserted below.
// Staging: one thread per (token, 16-byte chunk) and per (row, K16 step)
// because I8_TILE == threads / 4; two K64 slabs per K128 run as a
// register-prefetched ring with one barrier per slab (as the MQ4 kernel).
// Grid: [ceil(M/I8_TILE), ceil(N/I8_TILE)], block [4*I8_TILE,1,1].
// Compile gate: gfx1100 / gfx1151 only.

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
#ifndef I8_NUM_VGPR
#define I8_NUM_VGPR (I8_TILE == 256 ? 192 : 120)
#endif

#define I8_A_SLAB_BYTES (I8_TILE * 64)
#define I8_W_SLAB_BYTES (I8_TILE * 48)
#define I8_SLOT_BYTES (I8_A_SLAB_BYTES + I8_W_SLAB_BYTES)
#define I8_DS_OFF (2 * I8_SLOT_BYTES)
#define I8_DS_BYTES (I8_TILE * 8)
#define I8_SZ_OFF (I8_DS_OFF + 2 * I8_DS_BYTES)
#define I8_SZ_BYTES (I8_TILE * 8)
#define MQ6_I8_GFX11_LDS_BYTES (I8_SZ_OFF + 2 * I8_SZ_BYTES)
_Static_assert(MQ6_I8_GFX11_LDS_BYTES == (I8_TILE == 256 ? 65536 : 32768),
               "mq6 i8 gfx11 planes must match the host LDS contract");
_Static_assert(MQ6_I8_GFX11_LDS_BYTES <= 65536, "RDNA3 workgroup LDS limit");
#define I8_MAGIC 0x4B400000

typedef int32_t __attribute__((ext_vector_type(4))) int32x4_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;
typedef unsigned __attribute__((ext_vector_type(2))) u32x2_t;
typedef unsigned __attribute__((ext_vector_type(3))) u32x3_t;
typedef unsigned __attribute__((ext_vector_type(4))) u32x4_t;

// Byte offset of 16-byte chunk c of entity e inside a [I8_TILE][64 B] plane.
static __device__ __forceinline__ unsigned i8_chunk(unsigned e, unsigned c) {
    return e * 64u + ((c ^ ((e >> 1) & 3u)) << 4);
}

// C-row order of the DS planes: tokens 2l + hi of a 16-token group sit at
// hi * 8 + l, so one lane's 8 C rows are 32 contiguous bytes.
static __device__ __forceinline__ unsigned i8_ds_slot(unsigned t) {
    return (t & ~15u) | ((t & 1u) << 3) | ((t >> 1) & 7u);
}

// 6-bit code K (0..15) of the 96-bit step bitstream {g2, g1, g0}.
template <int K>
static __device__ __forceinline__ unsigned mq6_code(unsigned g0, unsigned g1, unsigned g2) {
    constexpr int bit = 6 * K;
    constexpr int w = bit >> 5;
    constexpr int sh = bit & 31;
    const unsigned g[3] = {g0, g1, g2};
    if constexpr (sh + 6 <= 32) return (g[w] >> sh) & 63u;
    else return ((g[w] >> sh) | (g[w + 1] << (32 - sh))) & 63u;
}

// Canonical 12-byte step -> LDS format {s0, s1, s2} (header comment).
static __device__ __forceinline__ u32x3_t mq6_repack(u32x3_t g) {
    const unsigned g0 = g.x, g1 = g.y, g2 = g.z;
    const unsigned w0 = mq6_code<0>(g0, g1, g2) | (mq6_code<2>(g0, g1, g2) << 8)
                      | (mq6_code<4>(g0, g1, g2) << 16) | (mq6_code<6>(g0, g1, g2) << 24);
    const unsigned w1 = mq6_code<1>(g0, g1, g2) | (mq6_code<3>(g0, g1, g2) << 8)
                      | (mq6_code<5>(g0, g1, g2) << 16) | (mq6_code<7>(g0, g1, g2) << 24);
    const unsigned w2 = mq6_code<8>(g0, g1, g2) | (mq6_code<10>(g0, g1, g2) << 8)
                      | (mq6_code<12>(g0, g1, g2) << 16) | (mq6_code<14>(g0, g1, g2) << 24);
    const unsigned w3 = mq6_code<9>(g0, g1, g2) | (mq6_code<11>(g0, g1, g2) << 8)
                      | (mq6_code<13>(g0, g1, g2) << 16) | (mq6_code<15>(g0, g1, g2) << 24);
    u32x3_t s;
    s.x = w0 | ((w3 << 6) & 0xC0C0C0C0u);
    s.y = w1 | ((w3 << 4) & 0xC0C0C0C0u);
    s.z = w2 | ((w3 << 2) & 0xC0C0C0C0u);
    return s;
}

// LDS format -> B operand {w0, w1, w2, w3}.
static __device__ __forceinline__ int32x4_t mq6_expand(unsigned s0, unsigned s1, unsigned s2) {
    int32x4_t w;
    w.x = (int)(s0 & 0x3F3F3F3Fu);
    w.y = (int)(s1 & 0x3F3F3F3Fu);
    w.z = (int)(s2 & 0x3F3F3F3Fu);
    w.w = (int)(((s0 >> 6) & 0x03030303u) | ((s1 >> 4) & 0x0C0C0C0Cu) | ((s2 >> 2) & 0x30303030u));
    return w;
}

// SET epilogue store: F32 or BF16 (RNE, bf16 = isfinite(y) ?
// (u + 0x7FFF + ((u >> 16) & 1)) >> 16 : u >> 16, u = bits(0.0f + y)).
static __device__ __forceinline__ void mq6_store_f32(float* p, float y) { *p = y; }
static __device__ __forceinline__ void mq6_store_bf16(unsigned short* p, float y) {
    const unsigned u = __float_as_uint(__fadd_rn(0.0f, y));
    const unsigned r = ((u & 0x7F800000u) != 0x7F800000u) ? ((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16) : (u >> 16);
    *p = (unsigned short)r;
}

template <bool BF16>
static __device__ __forceinline__ void gemm_mq6g256v2_mmq_i8_gfx11_body(
    const char*           __restrict__ A,   // [M x groups_per_row x 200]
    const block_i8_128*   __restrict__ Xq,  // [K/128 x N]
    void*                 __restrict__ Yv,  // [N x M] column-major, f32 or bf16
    int M, int K, int N
) {
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
    if (rs >= M || bs >= N) return;

    const int groups_per_row = K / 256;
    const int kblocks = K / 128;
    extern __shared__ char LDS[];

    // Wave tile: tokens wave_tok0 + 16 nb + .., weight rows wave_row0 + 16 rg + ..
    const unsigned l16 = (unsigned)lane & 15u;
    const unsigned hi = (unsigned)lane >> 4;
    const int wave_tok0 = (wave & 3) * (16 * I8_NB);
    const int wave_row0 = (wave >> 2) * 32;

    // --- Staging maps (loop-invariant). ---
    // A slab: thread stages token T = tid >> 2, chunk c = tid & 3.
    const unsigned st_T = (unsigned)(tid >> 2);
    const unsigned st_c = (unsigned)(tid & 3);
    const unsigned st_aoff = (unsigned)(bs + (int)st_T) * (unsigned)sizeof(block_i8_128) + 8u + st_c * 16u;
    const unsigned st_alds = i8_chunk(st_T, st_c);
    // W slab: row R = tid >> 2, K16 step c = tid & 3 (12 canonical bytes).
    const int st_wsrc = rs + (int)st_T;
    const unsigned st_woff =
        (unsigned)(st_wsrc < M ? st_wsrc : (M - 1)) * (unsigned)groups_per_row * 200u + 8u + st_c * 12u;
    // LDS W planes: dword j of (row R, step c) at plane 3c + j, word R ^ (8c).
    const unsigned st_wlds = I8_A_SLAB_BYTES + ((st_T ^ (st_c << 3)) << 2) + st_c * (3u * I8_TILE * 4u);
    const auto w_rsrc = __builtin_amdgcn_make_buffer_rsrc((void*)A, 0, 0xffffffffu, 0x31004000u);
    // Headers: every thread fetches and publishes token (tid % I8_TILE)'s
    // (d, s) and row (tid % I8_TILE)'s (sc, z') (4 identical writers per slot).
    const int hdr_i = tid & (I8_TILE - 1);
    const unsigned hdr_ds = i8_ds_slot((unsigned)hdr_i);
    const int hdr_src = rs + hdr_i;
    const char* hdr_sp = A + (long long)(hdr_src < M ? hdr_src : (M - 1)) * groups_per_row * 200;
    const unsigned a_bytes = (unsigned)N * (unsigned)sizeof(block_i8_128);

    // Consumer lane bases. A swizzle depends on l16 only. W: row e of step
    // t sits at word e ^ (8 t); with e = wave_row0 + 16 rg + l16 that is
    // wave_row0 + 16 (rg ^ (t >> 1)) + (l16 ^ (8 (t & 1))): two lane bases
    // (t even / odd), the rg term is an immediate.
    const unsigned rd_sw = ((l16 >> 1) & 3u) << 4;
    const unsigned rd_a = ((unsigned)wave_tok0 + l16) * 64u + rd_sw;
    const unsigned rd_w0 = I8_A_SLAB_BYTES + ((unsigned)wave_row0 + l16) * 4u;
    const unsigned rd_w1 = I8_A_SLAB_BYTES + ((unsigned)wave_row0 + (l16 ^ 8u)) * 4u;

    float acc[I8_NB][2][8];
    #pragma unroll
    for (int nb = 0; nb < I8_NB; nb++)
        #pragma unroll
        for (int rg = 0; rg < 2; rg++)
            #pragma unroll
            for (int l = 0; l < 8; l++) acc[nb][rg][l] = 0.0f;
    int32x8_t cacc[I8_NB][2];

    u32x4_t a_pf;
    u32x3_t w_pf;
    u32x2_t hd_pf;
    unsigned hs_pf;

    // Global fetch of K64 slab (kb, s) into registers.
    auto fetch_slab = [&](int kb, int s) {
        const auto a_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)(Xq + (long long)kb * N), 0, a_bytes, 0x31004000u);
        a_pf = __builtin_amdgcn_raw_buffer_load_b128(a_rsrc, st_aoff, (unsigned)(s * 64), 0);
        w_pf = __builtin_amdgcn_raw_buffer_load_b96(
            w_rsrc, st_woff, (unsigned)((kb >> 1) * 200 + (kb & 1) * 96 + s * 48), 0);
    };
    auto publish_slab = [&](int slot) {
        char* base = LDS + slot * I8_SLOT_BYTES;
        *(u32x4_t*)(base + st_alds) = a_pf;
        const u32x3_t sp = mq6_repack(w_pf);
        *(unsigned*)(base + st_wlds) = sp.x;
        *(unsigned*)(base + st_wlds + I8_TILE * 4) = sp.y;
        *(unsigned*)(base + st_wlds + I8_TILE * 8) = sp.z;
    };
    // Header fetch / publish of K128 block kb, both in every thread with no
    // branch (see the MQ4 kernel: divergent descriptors waterfall, and loads
    // used only inside one get sunk next to their use).
    auto fetch_hdr = [&](int kb) {
        const auto d_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)(Xq + (long long)kb * N), 0, a_bytes, 0x31004000u);
        hd_pf = __builtin_amdgcn_raw_buffer_load_b64(
            d_rsrc, (unsigned)(bs + hdr_i) * (unsigned)sizeof(block_i8_128), 0u, 0);
        hs_pf = *(const unsigned*)(hdr_sp + (kb >> 1) * 200 + (kb & 1) * 4);
    };
    auto publish_hdr = [&](int par) {
        // Tokens >= N read d = 0, s = 0 (descriptor bound).
        int* sv = (int*)(LDS + I8_DS_OFF + par * I8_DS_BYTES);
        float* dd = (float*)(LDS + I8_DS_OFF + par * I8_DS_BYTES + I8_TILE * 4);
        float* sz = (float*)(LDS + I8_SZ_OFF + par * I8_SZ_BYTES);
        sv[hdr_ds] = (int)hd_pf.y;
        dd[hdr_ds] = __uint_as_float(hd_pf.x);
        const float sc = __half2float(__ushort_as_half((unsigned short)(hs_pf & 0xFFFFu)));
        const float z = __half2float(__ushort_as_half((unsigned short)(hs_pf >> 16)));
        sz[hdr_i] = sc;
        sz[I8_TILE + hdr_i] = __fadd_rn(z, 32.0f * sc);
    };

    // One K64 slab of compute from `slot`: 4 K16 steps, each fenced so only
    // one step's operands are live (unfenced the scheduler hoists every
    // step's LDS reads to the slab start and the kernel spills).
    auto compute_slab = [&](int slot) {
        const char* base = LDS + slot * I8_SLOT_BYTES;
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int32x4_t af[I8_NB];
            #pragma unroll
            for (int nb = 0; nb < I8_NB; nb++)
                af[nb] = *(const int32x4_t*)(base + (rd_a ^ ((unsigned)t << 4)) + nb * 1024);
            const char* wb = base + ((t & 1) ? rd_w1 : rd_w0);
            int32x4_t wf[2];
            #pragma unroll
            for (int rg = 0; rg < 2; rg++) {
                const char* wp = wb + (unsigned)((3 * t) * I8_TILE * 4 + 64 * (rg ^ (t >> 1)));
                const unsigned s0 = *(const unsigned*)(wp);
                const unsigned s1 = *(const unsigned*)(wp + I8_TILE * 4);
                const unsigned s2 = *(const unsigned*)(wp + I8_TILE * 8);
                wf[rg] = mq6_expand(s0, s1, s2);
            }
            #pragma unroll
            for (int nb = 0; nb < I8_NB; nb++) {
                cacc[nb][0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[nb], false, wf[0], cacc[nb][0], false);
                cacc[nb][1] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[nb], false, wf[1], cacc[nb][1], false);
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
        // Chain seeds: I8_MAGIC - 32 * s per C row (token).
        {
            const int* sv = (const int*)(LDS + I8_DS_OFF + par * I8_DS_BYTES)
                + wave_tok0 + hi * 8;
            #pragma unroll
            for (int nb = 0; nb < I8_NB; nb++) {
                const int32x4_t s0 = *(const int32x4_t*)(sv + nb * 16);
                const int32x4_t s1 = *(const int32x4_t*)(sv + nb * 16 + 4);
                const int32x8_t seed = {
                    I8_MAGIC - 32 * s0.x, I8_MAGIC - 32 * s0.y, I8_MAGIC - 32 * s0.z, I8_MAGIC - 32 * s0.w,
                    I8_MAGIC - 32 * s1.x, I8_MAGIC - 32 * s1.y, I8_MAGIC - 32 * s1.z, I8_MAGIC - 32 * s1.w};
                cacc[nb][0] = seed;
                cacc[nb][1] = seed;
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
        // Fold this K128 block:
        //   acc = fma(RN(sc * d), as_float(C) - magic, acc)
        //   acc = fma(RN(z' * d), (float)s, acc)
        {
            const int* sv = (const int*)(LDS + I8_DS_OFF + par * I8_DS_BYTES)
                + wave_tok0 + hi * 8;
            const float* dd = (const float*)(LDS + I8_DS_OFF + par * I8_DS_BYTES + I8_TILE * 4)
                + wave_tok0 + hi * 8;
            const float* sz = (const float*)(LDS + I8_SZ_OFF + par * I8_SZ_BYTES)
                + wave_row0 + l16;
            const float sc[2] = {sz[0], sz[16]};
            const float zp[2] = {sz[I8_TILE], sz[I8_TILE + 16]};
            #pragma unroll
            for (int nb = 0; nb < I8_NB; nb++) {
                float d8[8];
                int s8[8];
                *(float4*)(d8) = *(const float4*)(dd + nb * 16);
                *(float4*)(d8 + 4) = *(const float4*)(dd + nb * 16 + 4);
                *(int32x4_t*)(s8) = *(const int32x4_t*)(sv + nb * 16);
                *(int32x4_t*)(s8 + 4) = *(const int32x4_t*)(sv + nb * 16 + 4);
                #pragma unroll
                for (int rg = 0; rg < 2; rg++) {
                    #pragma unroll
                    for (int l = 0; l < 8; l++) {
                        const float t1 = __fmul_rn(sc[rg], d8[l]);
                        const float t2 = __fmul_rn(zp[rg], d8[l]);
                        // Scalar copy first: clang lowers __builtin_bit_cast
                        // of an ext-vector element lvalue to element 0.
                        const int c = cacc[nb][rg][l];
                        const float fc = __int_as_float(c) - 12582912.0f;
                        float a = __builtin_fmaf(t1, fc, acc[nb][rg][l]);
                        a = __builtin_fmaf(t2, (float)s8[l], a);
                        acc[nb][rg][l] = a;
                    }
                }
            }
        }
        publish_slab(0);
        __syncthreads();
    }

    // --- Epilogue (SET): lane holds token wave_tok0 + 16 nb + 2l + hi and
    // row wave_row0 + 16 rg + l16; guarded for M and N tails. ---
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
                    const long long o = (long long)tok * M + row;
                    if constexpr (BF16) mq6_store_bf16((unsigned short*)Yv + o, acc[nb][rg][l]);
                    else mq6_store_f32((float*)Yv + o, acc[nb][rg][l]);
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(I8_NTHREADS) __attribute__((amdgpu_num_vgpr(I8_NUM_VGPR)))
void I8_ENTRY(gemm_mq6g256v2_mmq_i8_set_gfx11)(
    const char*           __restrict__ A,
    const block_i8_128*   __restrict__ Xq,
    void*                 __restrict__ Y,
    int M, int K, int N
) {
    gemm_mq6g256v2_mmq_i8_gfx11_body<false>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(I8_NTHREADS) __attribute__((amdgpu_num_vgpr(I8_NUM_VGPR)))
void I8_ENTRY(gemm_mq6g256v2_mmq_i8_set_bf16_gfx11)(
    const char*           __restrict__ A,
    const block_i8_128*   __restrict__ Xq,
    void*                 __restrict__ Y,
    int M, int K, int N
) {
    gemm_mq6g256v2_mmq_i8_gfx11_body<true>(A, Xq, Y, M, K, N);
}

#else

extern "C" __global__ void I8_ENTRY(gemm_mq6g256v2_mmq_i8_set_gfx11)(
    const char*, const void*, void*, int, int, int
) {}

extern "C" __global__ void I8_ENTRY(gemm_mq6g256v2_mmq_i8_set_bf16_gfx11)(
    const char*, const void*, void*, int, int, int
) {}

#endif

#undef I8_ENTRY
