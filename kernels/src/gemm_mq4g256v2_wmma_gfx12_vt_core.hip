// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Shared core of the gfx12 (RDNA4) MQ4G256V2 "verify-tile" WMMA GEMMs: the
// batched dense projections (qkvza / qkv / gate_up / residual+lm_head) at
// 16 < N < 64 rows (and, via the SLAB_K = 32 `_k32` instantiations, the wide
// 64 <= N <= 128 exact verify chunk), i.e. the multi-request speculative-verify trunk where
// several requests' 4-/16-row blocks are packed into one forward. NOT a
// standalone translation unit: `rdna-compute/src/kernels.rs` concatenates it
// in front of each `gemm_*_mq4g256v2_wmma_gfx12_vt.hip` wrapper (which supply
// the weight-row routing and the epilogue as an `Op` functor).
//
// Why this exists. The one-batch-tile kernels (grid.y = ceil(N/16), one wave
// per 16x16 tile) make every block stream the WHOLE activation panel
// X[N x K] through the L1/L2 for only 16 weight rows: at N = 48 each block
// moves ~15x more X bytes than weight bytes, so the GEMM is activation-
// bandwidth bound and its time scales ~linearly with N (the weights are
// re-fetched per batch tile on top). Batch-tiling inside one wave (the BT4
// kernels) shares the weight fragment but still streams all of X per
// 16 weight rows. These kernels instead run WAVES waves per block, each owning
// its own 16 weight rows, and stage each 128-K slab of the ENTIRE batch panel
// (up to 64 rows) into LDS ONCE per block, double buffered, so X is read from
// global memory 1/WAVES as often; every wave then feeds BT independent
// accumulators from LDS fragments while it dequantizes each of its weight
// fragments once.
//
// Byte-exactness. Every (weight row, batch row) output keeps exactly the
// one-tile kernel's arithmetic: one accumulator chain advanced by one WMMA
// per 16-K tile in ascending K (group g, K-tile kt = 0..15), the same dual-
// half (scale, zero) headers (kt < 8 -> header word 0, else word 1), the
// same fp16 dequant expression, the same C mapping (acc[j] = C[8*(lane>>4)+j]
// [lane&15]) and the same epilogue. Only where the B operand is read from
// (LDS copy of the same fp16 bytes) and how many tiles/waves a block covers
// change; no reduction order, accumulator or rounding changes.
//
// Geometry: block = WAVES waves (4 or 8: wide blocks share each staged slab
// across more weight rows, narrow blocks keep small-M projections spread over
// the CUs; the host picks by weight-row count); block x handles weight rows
// [x*16*WAVES, +16*WAVES), wave w owns 16 of them; block y handles batch rows
// [y*16*BT, +16*BT) (BT = ceil(N/16) in 2..4, so y = 1 for N <= 63). LDS:
// 2 x (16*BT) x 136 fp16 (row stride padded 128 -> 136 so the 16-byte B-
// fragment reads are bank-conflict free): 34.8 KB at BT = 4.
//
// Wide exact-verify variant (SLAB_K = 32, symbols `*_vt{4,8}w{4,8}_k32`,
// 64 <= N <= 128). At K128 staging BT8 would need 69.6 KB of LDS, over the
// 64 KB CU budget, so the wide instantiations stage the panel per 32-K slab
// instead: row stride 40 fp16 (32 + 8 pad), double buffered, 2 x (16*BT) x
// 40 fp16 = 10,240 B at BT4 and 20,480 B at BT8. Each slab is two ascending
// 16-K tiles; each weight 16-K fragment is still dequantized once and fed to
// all BT accumulators before K advances, and the header for a tile is still
// chosen by its 128-K half. Only the staging granularity and addressing
// change; per-output arithmetic is the one-tile kernel's, as above. The
// default SLAB_K = 128 instantiations (`_vt{2,3,4}w{4,8}`) are unchanged.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;
typedef unsigned int __attribute__((ext_vector_type(4))) uint4v_t;

// The one-tile kernels' dequantization, expression for expression.
__device__ __forceinline__ half8_t vt_dq8(unsigned int pk, _Float16 sc_h, _Float16 zp_h) {
    half8_t a_reg;
    a_reg[0] = sc_h * (_Float16)(float)(((pk) >>  0) & 0xFu) + zp_h;
    a_reg[1] = sc_h * (_Float16)(float)(((pk) >>  4) & 0xFu) + zp_h;
    a_reg[2] = sc_h * (_Float16)(float)(((pk) >>  8) & 0xFu) + zp_h;
    a_reg[3] = sc_h * (_Float16)(float)(((pk) >> 12) & 0xFu) + zp_h;
    a_reg[4] = sc_h * (_Float16)(float)(((pk) >> 16) & 0xFu) + zp_h;
    a_reg[5] = sc_h * (_Float16)(float)(((pk) >> 20) & 0xFu) + zp_h;
    a_reg[6] = sc_h * (_Float16)(float)(((pk) >> 24) & 0xFu) + zp_h;
    a_reg[7] = sc_h * (_Float16)(float)(((pk) >> 28) & 0xFu) + zp_h;
    return a_reg;
}

// Op contract:
//   const char* row(int safe_row, long long row_stride) const
//       weight bytes of global row `safe_row` (clamped to total_m-1), i.e.
//       projection base + local_row * row_stride.
//   void store(int out_row, int out_col, float v) const
//       the epilogue write of output (out_row, out_col = batch index).
// SLAB_K: K width of one LDS-staged panel slab, 128 (default, `_vt{2,3,4}`)
// or 32 (wide `_vt{4,8}_k32`). Changes staging/addressing only.
template <int BT, int WAVES, int SLAB_K = 128, class Op>
__device__ __forceinline__ void mq4v2_vt_gemm(const Op& op, const _Float16* __restrict__ X,
                                              int total_m, int K, int N) {
    static_assert(SLAB_K == 128 || SLAB_K == 32, "SLAB_K must be 128 or 32");
    constexpr int NT  = WAVES * 32;
    constexpr int NR  = BT * 16;          // batch rows staged per block
    constexpr int XS  = SLAB_K + 8;       // LDS row stride (fp16): SLAB_K K + 8 pad
    constexpr int KT  = SLAB_K / 16;      // 16-K WMMA tiles per slab
    constexpr int CSH = (SLAB_K == 128) ? 4 : 2;  // log2(16-byte chunks per row)
    constexpr int GSH = (SLAB_K == 128) ? 1 : 3;  // log2(slabs per 256-K group)
    constexpr int HSH = (SLAB_K == 128) ? 0 : 2;  // log2(slabs per 128-K header half)
    constexpr int NCH = NR << CSH;        // 16-byte chunks per slab
    constexpr int CPT = (NCH + NT - 1) / NT;  // chunks per thread
    constexpr bool EVEN = (NCH % NT) == 0;    // every thread owns CPT valid chunks

    __shared__ __attribute__((aligned(16))) _Float16 xs[2][NR][XS];

    const int tid  = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int ml   = lane & 15;           // weight row / batch column within the tile
    const int kg   = lane >> 4;           // which 8-K half of a 16-K tile
    const int rs   = (blockIdx.x * WAVES + wave) * 16;
    const int bs   = blockIdx.y * NR;

    // Block-uniform early exit (before any barrier).
    if (blockIdx.x * WAVES * 16 >= total_m || bs >= N) return;
    const bool active = rs < total_m;     // wave-uniform

    const int my_row   = rs + ml;
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);
    const int gpr      = K / 256;
    const int S        = gpr * (1 << GSH); // slabs (K128: one per (scale, zero) half-group)
    const char* rb     = op.row(safe_row, (long long)gpr * 136);

    // Cooperative slab load: thread chunk c = tid + j*NT -> batch row c>>CSH,
    // 16-byte column chunk c&(2^CSH-1). Rows past N reread row 0 (as the one-tile
    // kernels do); their columns are never stored.
    const _Float16* xsrc[CPT];
    int xdst[CPT];
    bool xok[CPT];
    #pragma unroll
    for (int j = 0; j < CPT; j++) {
        const int c = tid + j * NT;
        const int r = c >> CSH;
        const int cc = c & ((1 << CSH) - 1);
        const int t = bs + r;
        xsrc[j] = X + (long long)((t < N) ? t : 0) * K + cc * 8;
        xdst[j] = r * XS + cc * 8;
        xok[j] = EVEN || (c < NCH);
    }

    uint4v_t xr[CPT];
    unsigned int p[KT], pn[KT], h, hn;

    #define VT_LDX(s) \
        _Pragma("unroll") \
        for (int j = 0; j < CPT; j++) if (xok[j]) xr[j] = *(const uint4v_t*)(xsrc[j] + (long long)(s) * SLAB_K)
    #define VT_STX(buf) \
        _Pragma("unroll") \
        for (int j = 0; j < CPT; j++) if (xok[j]) *(uint4v_t*)(&xs[buf][0][0] + xdst[j]) = xr[j]
    // Slab s: group s>>GSH; header word = its 128-K half ((s>>HSH)&1, i.e.
    // 16-K tile kt < 8 -> word 0); 16-K tile i of the slab at nibble byte
    // (s & (2^GSH-1)) * SLAB_K/2 + i*8 (+ kg*4 for this lane's 8-K half).
    #define VT_LDW(s, pp, hh) { \
        const char* gp_ = rb + ((s) >> GSH) * 136; \
        hh = *(const unsigned int*)(gp_ + (((s) >> HSH) & 1) * 4); \
        const char* pw_ = gp_ + 8 + ((s) & ((1 << GSH) - 1)) * (SLAB_K / 2) + kg * 4; \
        _Pragma("unroll") \
        for (int i = 0; i < KT; i++) pp[i] = *(const unsigned int*)(pw_ + i * 8); }

    VT_LDX(0);
    VT_LDW(0, p, h);
    VT_STX(0);
    __syncthreads();

    float8_t acc[BT];
    #pragma unroll
    for (int b = 0; b < BT; b++) acc[b] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};

    for (int s = 0; s < S; s++) {
        const int cur = s & 1;
        const bool more = (s + 1) < S;
        if (more) {
            VT_LDX(s + 1);
            VT_LDW(s + 1, pn, hn);
        }
        if (active) {
            const _Float16 sc_h = (_Float16)__half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu)));
            const _Float16 zp_h = (_Float16)__half2float(__ushort_as_half((unsigned short)(h >> 16)));
            #pragma unroll
            for (int ki = 0; ki < KT; ki++) {
                const half8_t a = vt_dq8(p[ki], sc_h, zp_h);
                #pragma unroll
                for (int b = 0; b < BT; b++) {
                    const half8_t bb = *(const half8_t*)(&xs[cur][b * 16 + ml][ki * 16 + kg * 8]);
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, acc[b]);
                }
            }
        }
        if (more) {
            VT_STX(cur ^ 1);
            #pragma unroll
            for (int i = 0; i < KT; i++) p[i] = pn[i];
            h = hn;
        }
        __syncthreads();
    }
    #undef VT_LDX
    #undef VT_STX
    #undef VT_LDW

    if (!active) return;
    #pragma unroll
    for (int b = 0; b < BT; b++) {
        const int oc = bs + b * 16 + ml;
        if (oc < N) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int orow = rs + 8 * kg + j;
                if (orow < total_m) op.store(orow, oc, acc[b][j]);
            }
        }
    }
}
